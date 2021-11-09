import json


def load_args(file_path):
    with open(file_path, 'r') as f:
        args_dict = json.load(f)
        f.close()
    if "use_func" not in args_dict:
        args_dict["use_func"] = False
    print("load arguments:", args_dict)
    args = ARGs(args_dict)
    return args


class ARGs:
    def __init__(self, dic):
        for k, v in dic.items():
            setattr(self, k, v)


def check_args(args):
    if args.embedding_module in ['TransE', 'TransH', 'TransR', 'TransD']:
        assert args.neg_triple_num == 1
