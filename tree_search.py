import debugpy
# debugpy.listen(("0.0.0.0", 5655))

import random
import re
import copy
import os
import json
import argparse
import logging
import torch
import sys
import numpy as np
import requests
import time


from torch.distributed import (
    init_process_group,
    get_rank,
    get_world_size
)

import traceback
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import (
    AutoProcessor,
    AutoModelForCausalLM,
    AutoTokenizer,
    set_seed
)

from tqdm import tqdm
from qwen_vl.utils import process_vision_info
from vllm import LLM, SamplingParams
from func_timeout import func_timeout, func_set_timeout, FunctionTimedOut
from verifier import Verifier
from utils.symbolic import parse_clause

logger = logging.getLogger(__name__)

def read_json(path):
    data = json.load(open(path, 'r', encoding='utf-8'))
    return data

def read_jsonl(path):
    results = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:  # 去除空行
                try:
                    results.append(json.loads(line))
                except json.JSONDecodeError as e:
                    logger.info(f"解析错误: {e}, 行内容: {line}")
    return results

def write_json(path, data):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

def get_all_lines(clauses):
    lines = set()
    for clause in clauses:
        if 'Shape' in clause:
            name, items = parse_clause(clause)
            for item in items:
                if len(tuple(item)) >= 2:
                    lines.add(tuple(item))
        elif 'Equal' in clause and 'LengthOfLine(' in clause:
            name, items = parse_clause(clause)
            for item in items:
                if item.isalpha():
                    if len(tuple(item)) >= 2:
                        lines.add(tuple(item))
        elif 'Equal' in clause and 'MeasureofAngle' in clause:
            name, items = parse_clause(clause)
            for item in items:
                if item.isalpha():
                    if len(tuple(item)) == 3:
                        lines.add(tuple(item[2:]))
                        lines.add(tuple(item[:1]))
    return tuple(lines)

def build_model(model_path, logic_model_path, local_rank):
    model = LLM(
        model=model_path,
        device=f'cuda:{local_rank}',
        gpu_memory_utilization=0.55,
        max_model_len=4096,
    )

    logic_model = AutoModelForCausalLM.from_pretrained(
        logic_model_path,
        device_map={f'cuda:{local_rank}'},
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )

    # logic_model = LLM(
    #     model=logic_model_path,
    #     device=f'cuda:{local_rank}',
    #     gpu_memory_utilization=0.3,
    #     max_model_len=512
    # )

    return model, logic_model

def build_init_data(data_path, processor, output_dir, local_rank, world_size):
    raw_data = read_json(data_path)
    keys = [item['key'] for item in raw_data]

    logger.info(f"✅ keys: {len(keys)}")

    # remove keys that existed
    existed_keys = []
    for f in os.listdir(output_dir):
        try:
            data = read_json(f"{output_dir}/{f}")
            if data['response'] != '':
                existed_keys.append(f.split('.')[0].split('_')[-1])
        except:
            logger.error(f'{f} file {f} load error !')

    existed_idx = [keys.index(k) for k in existed_keys]

    keys = [k for idx, k in enumerate(keys) if idx not in existed_idx]
    raw_data = [item for idx, item in enumerate(raw_data) if idx not in existed_idx]
    logger.info(f"Keys to search: {len(keys)}")

    sliced_data = raw_data[local_rank::world_size]
    sliced_keys = keys[local_rank::world_size]
    inputs = []

    for v in sliced_data:
        prompt_msg = [v['messages'][0]]
        text = processor.apply_chat_template(
            prompt_msg,
            tokenize=False,
            add_generation_prompt=True
        )
        # image = v['messages'][0]['content'][0]['image']
        images, _ = process_vision_info(prompt_msg)
        inputs += [
            {"prompt": text, "multi_modal_data": {"image": images[0]}}
        ]

    return inputs, sliced_keys


def vllm_chat(url, prompt, stop=None):
    headers = {"Content-Type": "application/json"}
    payload = {
        "model": "any-model-name",
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7
    }
    if stop:
        payload["stop"] = stop
    response = requests.post(f"{url}/v1/chat/completions", json=payload, headers=headers)
    return response.json()["choices"][0]["message"]["content"]


class StepSearcher:
    def __init__(
        self,
        model,
        logic_model,
        init_query_list,
        keys,
        processor,
        logic_tokenizer,
        output_dir,
        sample_width=8,
        temperature=0.9,
        local_rank=0,
        symbolic_info_path='../total_geo_test_symbolic.json',
        disable_symbolic=False,
        debug=False
    ):
        self.model = model
        self.logic_model = logic_model
        self.init_query_list = init_query_list
        self.keys = keys
        self.processor = processor
        self.logic_tokenizer = logic_tokenizer
        self.output_dir = output_dir
        self.sample_width = sample_width
        self.temperature = temperature
        self.local_rank = local_rank
        self.disable_symbolic = disable_symbolic
        self.debug = debug

        # symbolic files
        self.symbolic_info_test = read_json(symbolic_info_path)
        self.predicate_GDL_search = read_json('geogen/json/predicate_GDL_for_search.json')
        self.theorem_GDL_search = read_json('geogen/json/theorem_GDL_for_search.json')
        self.t_info = read_json('geogen/json/t_info_new.json')
        self.natural_template = read_json('geogen/json/predicates_to_nature_language.json')

        self.logic_model.eval()

    @func_set_timeout(600)
    def init_symbolic_verifier(self, key):
        # empty cache first
        self.cache_verifiers = []
        if key not in self.symbolic_info_test:
            return False

        symbolic_info = self.symbolic_info_test[key]

        for i in range(self.sample_width):
            verifier_i = Verifier(
                predicate_GDL=self.predicate_GDL_search,
                theorem_GDL=self.theorem_GDL_search,
                t_info=self.t_info,
                natural_template=self.natural_template,
                debug=False,
                strict=False
            )

            lines = get_all_lines(
                symbolic_info['text_cdl'] + symbolic_info['construction_cdl'] + symbolic_info['image_cdl']
            )
            if 'lines' in symbolic_info:
                lines = list(set(tuple(l) for l in symbolic_info['lines']) | set(tuple(lines)))

            verifier_i.init_problem(
                text_cdls=symbolic_info['text_cdl'],
                constr_cdls=symbolic_info['construction_cdl'],
                image_cdls=symbolic_info['image_cdl'],
                p_pos=symbolic_info['p_pos'] if 'p_pos' in symbolic_info else None,
                lines=lines,
            )
            self.cache_verifiers.append(verifier_i)

        return True
    
def generate_beams(self, prompt, step_count, t=0.9, top_k=50, top_p=0.95):
    sampling_params = SamplingParams(
        max_tokens=2048,
        temperature=t,
        top_k=top_k,
        top_p=top_p,
        stop=['\n{step_count + 1}.']
    )

    generated_ids_trimmed = []

    outputs = self.model.generate(
        [prompt] * self.sample_width,
        sampling_params=sampling_params,
        use_tqdm=False
    )
    # for out in outputs:
    #     generated_ids_trimmed.append(out.outputs[0].token_ids)
    # output_text = self.processor.batch_decode(
    #     generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    # )
    output_text = [item.outputs[0].text for item in outputs]
    return output_text


def generate_one_response(self, prompt, t=0.9, top_k=50, top_p=0.95):
    sampling_params = SamplingParams(
        max_tokens=2048,
        temperature=t,
        top_k=top_k,
        top_p=top_p
    )

    generated_ids_trimmed = []

    outputs = self.model.generate(
        [prompt],
        sampling_params=sampling_params,
        use_tqdm=False
    )
    for out in outputs:
        generated_ids_trimmed.append(out.outputs[0].token_ids)
    output_text = self.processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return output_text


def split_into_steps(self, pred_list, key):
    total_steps = []
    logic_input_msgs = []
    instruction = 'Translate the content into geometry formal language with json format:\n'

    for response in pred_list:
        steps = re.split(r'\n\d+\. ', response)
        steps = [step.rstrip('\n').strip() for step in steps]
        steps = [step for step in steps if step != '']
        if steps[0].startswith('1. '):
            steps[0] = steps[0].split('1. ')[-1]
        logic_msgs = []
        steps_dict = {}
        for j, step in enumerate(steps):
            logic_msg = {
                "messages": [
                    {"role": "user", "content": f"{instruction}{step}"},
                ],
                "key": f"{key}_step{j}"
            }
            logic_msgs.append(logic_msg)
            steps_dict[j] = step

        logic_input_msgs.append(logic_msgs)
        total_steps.append(steps_dict)

    return total_steps, logic_input_msgs


def geologic_transcription(self, cur_logic_input_msgs):
    batch_prompts = [
        self.logic_tokenizer.apply_chat_template(item["messages"], tokenize=False) for item in cur_logic_input_msgs
    ]
    data = {
        "model": "logic",
        "prompt": batch_prompts,
        "max_tokens": 256,
        "temperature": 0,
    }
    res = requests.post("http://localhost:8001/v1/completions", json=data)
    output_text = [item["text"] for item in res.json()["choices"]]
    return output_text


def verify_one_step(
    self,
    cur_step_strs,
    cur_logic_input_msgs,
    key
):
    # Logic Transcription
    start_time_trans = time.time()
    steps_logic = self.geologic_transcription(cur_logic_input_msgs)
    step_time_trans = time.time() - start_time_trans

    # Logic Verify
    start_time_symbol = time.time()
    steps_res_list = []
    for i in range(len(steps_logic)):
            start_time_symbol = time.time()
    steps_res_list = []
    for i in range(len(steps_logic)):
        try:
            solution_i = {
                0: json.loads(steps_logic[i])
            }
            solution_str_i = {
                0: cur_step_strs[i]
            }
            step_res = self.cache_verifiers[i].verify(solution_i, solution_str_i)[0]
            steps_res_list.append(step_res)
        except:
            steps_res_list.append(False)
    step_time_symbol = time.time() - start_time_symbol

    time_info = {
        "step_time_trans": step_time_trans,
        "step_time_symbol": step_time_symbol
    }

    return steps_res_list, time_info


# @func_set_timeout(600)
def search_for_one_prompt(self, input_prompt, key):
    current_context = ""
    step_infer_time_list = []
    step_verify_time_list = []
    step_trans_time_list = []
    step_symbol_time_list = []

    prompt = copy.deepcopy(input_prompt)
    step_count = 1

    start_time_one_response = time.time()
    one_res_time = 0
    if self.debug:
        debug_context = self.generate_one_response(prompt, t=0)[0]
        one_res_time = time.time() - start_time_one_response

    start_time_tree_search = time.time()
    while True:
        # check length first
        if len(self.logic_tokenizer.encode(prompt['prompt'])) > 2048:
            break

        if 'Answer:' in current_context:
            break

        if step_count > 20:
            break

        # generate all future steps
        start_time_step_infer = time.time()
        beams = self.generate_beams(
            prompt,
            step_count=step_count,
            t=self.temperature
        )
        step_infer_time_list.append(time.time() - start_time_step_infer)

        if any([b == '' for b in beams]):
            break

        start_time_verify = time.time()
        # split beam into steps and choose the first (current) step
        total_steps, logic_input_msgs = self.split_into_steps(beams, key)
        # if not all(len(item) == 1 for item in total_steps):
        #     continue
        cur_step_strs = [item[0] for item in total_steps]
        cur_logic_input_msgs = [item[0] for item in logic_input_msgs]

        # check end
        # if any(['Answer:' in b for b in cur_step_strs]):
        #     end_ids = [idx for idx, step_str in enumerate(cur_step_strs) if 'Answer:' in step_str]
        #     chosen_end_id = random.choice(end_ids)
        #     chosen_end_step = cur_step_strs[chosen_end_id]
        #     current_context += f"\n{step_count}. {chosen_end_step}"
        #     break

        # Validate each candidate, skip if no symbolic annotation
        start_time_verify = time.time()
        if len(self.cache_verifiers) == 0:
            chosen_beam_ids = []
        elif len(cur_step_strs) == 1:
            chosen_beam_ids = [0]
        else:
            verify_results, time_info = self.verify_one_step(
                cur_step_strs,
                cur_logic_input_msgs,
                key
            )
            chosen_beam_ids = [idx for idx, res in enumerate(verify_results) if res]

        # check end
        if any(['Answer:' in b for b in cur_step_strs]):
            end_ids = [idx for idx, step_str in enumerate(cur_step_strs) if 'Answer:' in step_str]
            if len(chosen_beam_ids) == 0:
                if len(chosen_beam_ids) == end_ids:
                    chosen_beam_ids = end_ids
            elif len(set(chosen_beam_ids) & set(end_ids)) != 0:
                chosen_beam_ids = list(set(chosen_beam_ids) & set(end_ids))
        step_trans_time_list.append(time_info['step_time_trans'])
        step_symbol_time_list.append(time_info['step_time_symbol'])

        step_verify_time_list.append(time.time() - start_time_verify)

        # Choose one valid step at random
        if len(chosen_beam_ids) == 0:
            chosen_step = random.choice(cur_step_strs)
        else:
            chosen_id = random.choice(chosen_beam_ids)
            chosen_step = cur_step_strs[chosen_id]

        # Append to full response and update prompt
        # if step_count == 1:
        #     prompt['prompt'] += chosen_step
        #     current_context += chosen_step
        # else:
        #     prompt['prompt'] += f"\n{step_count}. {chosen_step}"
        #     current_context += f"\n{step_count}. {chosen_step}"
        if 'Answer:' not in chosen_step:
            prompt['prompt'] += f"{chosen_step}\n{step_count + 1}. "
            current_context += f"{chosen_step}\n{step_count + 1}. "
        else:
            prompt['prompt'] += chosen_step
            current_context += chosen_step

        step_count += 1

    tree_search_time = time.time() - start_time_tree_search
    logger.info(str({"key": key, "response": current_context}))
    search_info = {
        "one_res_time": one_res_time,
        "tree_search_time": tree_search_time,
        "total_step": step_count - 1,
        "step_infer_time": step_infer_time_list,
        "step_verify_time": step_verify_time_list,
        "step_trans_time_list": step_trans_time_list,
        "step_symbol_time_list": step_symbol_time_list
    }
    if self.debug:
        return debug_context, search_info
    return current_context, search_info


def search(self):
    for idx in tqdm(range(len(self.init_query_list))):
        query = self.init_query_list[idx]
        key = self.keys[idx]

        save_path = f"{self.output_dir}/geopub_{key}.json"

        if self.disable_symbolic:
            self.cache_verifiers = []
        else:
            try:
                # init for verifier
                self.init_symbolic_verifier(key)
            except:
                self.cache_verifiers = []
        try:
            # search for one query
            response, search_info = self.search_for_one_prompt(query, key)
        except:
            error_str = traceback.format_exc()
            logger.error(f'Error: Rank {self.local_rank} Key {key}:')
            logger.error(error_str)
            response = ""
            search_info = {}

        # save response
        save_data = {
            "query": query['prompt'],
            "response": response,
            "key": key,
            "search_info": search_info
        }

        write_json(save_path, save_data)

        # torch.distributed.barrier()
    return


def main():
    args = get_args()
    set_seed(args.seed)

    if args.debug:
        local_rank = 0
        world_size = 1
    else:
        init_process_group(backend='nccl')
        local_rank = get_rank()
        logger.info(f'local rank: {local_rank}')
        world_size = get_world_size()

    if local_rank == 0:
        logger.info(f'seed: \t{args.seed}')
        logger.info(f'ckpt: \t{args.model_path}')
        logger.info(f'output: \t{args.output_dir}')
        logger.info(f'processor: \t{args.processor_path}')

    device = torch.device(f"cuda:{local_rank}")
    processor = AutoProcessor.from_pretrained(args.processor_path)
    logic_tokenizer = AutoTokenizer.from_pretrained(args.logic_model_path)

    # load init prompt
    init_query_list, keys = build_init_data(args.data_path, processor, args.output_dir, local_rank, world_size)
    logger.info(f"len(keys): {len(keys)}")

    # build model, logic model
    model, logic_model = build_model(
        args.model_path, args.logic_model_path, local_rank
    )

    searcher = StepSearcher(
        model,
        logic_model,
        init_query_list,
        keys,
        processor,
        logic_tokenizer=logic_tokenizer,
        output_dir=args.output_dir,
        sample_width=args.sample_width,
        temperature=args.temperature,
        local_rank=local_rank,
        disable_symbolic=args.disable_symbolic,
        debug=args.debug
    )
    searcher.search()

    torch.distributed.barrier()
    torch.distributed.destroy_process_group()


def get_args():
    parser = argparse.ArgumentParser(description='test')
    parser.add_argument('--seed', type=int, default=1234)

    parser.add_argument('--data_path',type=str,)
    parser.add_argument('--processor_path',type=str,)
    parser.add_argument('--model_path',type=str,)
    parser.add_argument('--logic_model_path',type=str,)
    parser.add_argument('--output_dir',type=str,)
    parser.add_argument('--sample_width', type=int, default=4)
    parser.add_argument('--temperature', type=float, default=0.5)
    parser.add_argument('--disable_symbolic', action='store_true')
    parser.add_argument('--debug', action='store_true')

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreanHandler(sys.stdout)]
    )
    logger.setLevel(logging.INFO)
    main()
