
import json
import os
from multiprocessing import Pool

from openai import OpenAI
from tqdm import tqdm

INSTRUCTION = """Convert the following solution of one geometry problem into clear, concise English. The following rules must be followed:
1. Try to use mathematical expressions instead of formal language.
2. Retain key information, including theorem, condition and conclusion of each step. 
3. Respond with solution after transcription only.
4. Append final answer at the end in the format: \\boxed\{ your answer \}"
"""


class Agent():
    def __init__(self, api_path) -> None:
        self.api_dict = json.load(open(api_path, 'r', encoding='utf-8'))
        self.url_dict = {
            'openai': "https://api.chatanywhere.tech/v1",
            'qwen': "https://dashscope.aliyuncs.com/compatible-mode/v1",
            'deepseek': "https://api.deepseek.com/v1"
        }
        
    def get_engine_type(self, model_name):
        engine_type = None
        if 'gpt' in model_name:
            engine_type = 'openai'
        elif 'qwen' in model_name:
            engine_type = 'qwen'
        elif 'deepseek' in model_name:
            engine_type = 'deepseek'
        
        if engine_type is None:
            raise KeyError(model_name)
        
        return engine_type
            
    def chat(self, prompt, model_name="gpt-4o-mini"):
        engine_type = self.get_engine_type(model_name)
        client = OpenAI(
            api_key=self.api_dict[engine_type],
            base_url=self.url_dict[engine_type]
        )
        completion = client.chat.completions.create(
            model=model_name,
            messages=[
                {'role': 'system', 'content': 'You are a helpful assistant.'},
                {'role': 'user', 'content': prompt}],
            top_p=0.9,
        )
            
        # data = completion.model_dump_json()
        # data = json.loads(data)
        # content = data['choices'][0]['message']['content']
        content = completion.choices[0].message.content
        usage = {
            "prompt_tokens": completion.usage.prompt_tokens,
            "completion_tokens": completion.usage.completion_tokens
        }
        
        return content, usage


def stats_formalgeo():
    base_dir = 'datasets/processed_data'
    total_ori = 0
    print('solved / unsolved / keyerror / total')
    for split in ['train', 'val', 'test']:
        json_dir = f"{base_dir}/fgo_{split}"
        json_names = os.listdir(json_dir)
        
        solved_n = 0
        unsolved_n = 0
        keyerror_n = 0
        keyerror_lst = []
        for json_name in tqdm(json_names):
            json_i = json.load(open(f"{json_dir}/{json_name}", 'r', encoding='utf-8'))
            if 'solved' not in json_i:
                keyerror_n += 1
                keyerror_lst.append(json_name)
            else:
                if json_i['solved']:
                    solved_n += 1
                else:
                    unsolved_n += 1
                    
        data_ori = json.load(open(f"{base_dir}/fgo_{split}.json", 'r', encoding='utf-8'))
        print(f"{split}: {solved_n} / {unsolved_n} / {keyerror_n} / {len(data_ori)}")
        print(keyerror_lst)
        
        
    print(f"total: {total_ori}")
        

def test():
    test_path = 'datasets/processed_data/fgo_train/88.json'
    test_data = json.load(open(test_path, 'r', encoding='utf-8'))
    agent = Agent('api.json')
    model_names = [
        'gpt-4o-mini', 
        'gpt-4o', 
        'qwen2.5-7b-instruct',
        'qwen2.5-14b-instruct',
        'qwen2.5-72b-instruct',
        'qwen-math-turbo',
        'qwen-plus',
        'qwen-turbo',
        'deepseek-chat'
    ]
    
    prompt = f"{INSTRUCTION}\n{test_data['llm_info']['solution_str']}"
    print('=========== orgin ==========')
    print(test_data['llm_info']['solution_str'])
    
    for name in model_names:
        answer, usage = agent.chat(prompt, name)
        print(f'=========== {name} ==========')
        print(answer)
        print('prompt tokens    : ', usage['prompt_tokens'])
        print('completion tokens: ', usage['completion_tokens'])
        
    return

def get_source_problem(key):
    base_dir = 'datasets/formalgeo7k/problems'
    geo3k_dir = 'D:/Desktop/资源/几何答题/InterGPS-main/data/geometry3k'
    data = json.load(open(f"{base_dir}/{key}.json", 'r', encoding='utf-8'))
    source = data['source']
    if 'Geometry3k' in source:
        idx = int(source.split('-')[-1])
        if idx <= 2100:
            split = 'train'
        elif idx <= 2400 and idx > 2100:
            split = 'val'
        else:
            split = 'test'
        geo3k_path = f"{geo3k_dir}/{split}/{idx}/data.json"
        if os.path.exists(geo3k_path):
            geo3k_data = json.load(open(geo3k_path, 'r', encoding='utf-8'))
            return geo3k_data['problem_text']
        else:
            raise KeyError(geo3k_path)
    elif 'GeoQA':
        return ""
    
def chat_for_one_sample(
    base_dir, save_dir, agent, 
    split, json_name, model_name
):
    # read origin and expanded data
    json_path = f"{base_dir}/fgo_{split}/{json_name}"
    save_path = f"{save_dir}/{json_name}"
    if os.path.exists(save_path):
        return None
    
    data = json.load(open(json_path, 'r', encoding='utf-8'))
    expand_data_lst = []
    expand_name_lst = []
    for idx in range(2):
        expand_name = f"{json_name.split('.')[0]}_{idx}.json"
        expand_path = f"{base_dir}/fgo_{split}_expand/{expand_name}"
        if os.path.exists(expand_path):
            expand_data_lst.append(
                json.load(open(expand_path, 'r', encoding='utf-8'))
            )
            expand_name_lst.append(expand_name)
    
    save_dict = {}
    if data['solved']:
        prompt = f"{INSTRUCTION}\n{data['llm_info']['solution_str']}"
        answer_ori, usage_ori = agent.chat(prompt, model_name)
        
        save_dict[json_name.split('.')[0]] = {
            "key": data['key'],
            "problem_text_ori": get_source_problem(data['key']),
            "problem_text": data['llm_info']['problem_text'],
            "solution_source": data['llm_info']['solution_str'],
            "solution_chat": answer_ori
        }
    
    for data_i, name_i in zip(expand_data_lst, expand_name_lst):
        prompt = f"{INSTRUCTION}\n{data_i['llm_info']['solution_str']}"
        answer, usage = agent.chat(prompt, model_name)
        save_dict[name_i.split('.')[0]] = {
            "key": data['key'],
            "problem_text": data_i['llm_info']['problem_text'],
            "solution_source": data_i['llm_info']['solution_str'],
            "solution_chat": answer
        }

    # save_path = f"{save_dir}/{json_name}"
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(save_dict, f, indent=4, ensure_ascii=False)
        
    return None
    
def main():
    split = 'train'
    base_dir = 'datasets/processed_data'
    save_dir = f"{base_dir}/fgo_{split}_chat"
    os.makedirs(save_dir, exist_ok=True)
    
    agent = Agent('api.json')
    model_name = 'gpt-4o'
    
    json_names = os.listdir(f"{base_dir}/fgo_{split}")
    # for i, json_name in enumerate(tqdm(json_names)):
    results = []
    num_process = 10
    pool = Pool(num_process)
    with tqdm(total=len(json_names)) as pbar:
        def update(*args, **kwargs):
            pbar.update()
        for i, json_name in enumerate(json_names):
            res = pool.apply_async(
                chat_for_one_sample,
                args=(
                    base_dir, save_dir, agent, 
                    split, json_name, model_name
                ),
                callback=update
            )
            results.append(res)
        for r in results:
            r.wait()
            
    pool.close()

if __name__ == '__main__':
    # test()
    # stats_formalgeo()
    main()
    
        
