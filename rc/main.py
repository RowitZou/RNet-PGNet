import torch
import numpy as np

from rc.utils.argsParser import get_args
from rc.model_handler import ModelHandler


def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def main(args):
    print_config(args)
    set_random_seed(args['random_seed'])
    model = ModelHandler(args)
    model.train()
    model.test()


def print_config(config):
    print("**************** MODEL CONFIGURATION ****************")
    for key in sorted(config.keys()):
        val = config[key]
        keystr = "{}".format(key) + (" " * (24 - len(key)))
        print("{} -->   {}".format(keystr, val))
    print("**************** MODEL CONFIGURATION ****************")


if __name__ == '__main__':
    args = get_args()
    main(args)
