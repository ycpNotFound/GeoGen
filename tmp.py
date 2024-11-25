from formalgeo.solver import ForwardSearcher
from formalgeo.data import DatasetLoader
import json

dl = DatasetLoader(dataset_name="formalgeo7k", datasets_path="datasets")
t_info = json.load(open("datasets/formalgeo7k/files/t_info.json", 'r', encoding='utf-8'))
problem_idx = 1
problem_CDL = dl.get_problem(pid=problem_idx)

solver = ForwardSearcher(
    dl.predicate_GDL,
    dl.theorem_GDL,
    strategy="beam_search",
    max_depth=12, 
    beam_size=6,
    t_info=t_info,
    debug=True
)

solver.init_search(problem_CDL)
solver.search()


print(solver.problem.condition.items()) # 包含了搜索中生成的所有条件
