import argparse
import torch
from unsup3d import setup_runtime, Trainer, Unsup3D
from tqdm import tqdm


## runtime arguments
parser = argparse.ArgumentParser(description='Training configurations.')
parser.add_argument('--config', default=None, type=str, help='Specify a config file path')
parser.add_argument('--gpu', default=0, type=int, help='Specify a GPU device')
parser.add_argument('--num_workers', default=4, type=int, help='Specify the number of worker threads for data loaders')
parser.add_argument('--seed', default=2020, type=int, help='Specify a random seed')
args = parser.parse_args()

## set up
cfgs = setup_runtime(args)
run_train = cfgs.get('run_train', False)
run_test = cfgs.get('run_test', False)

if run_train:
    trainer = Trainer(cfgs, Unsup3D)
    trainer.train()
if run_test:
    trainer = Trainer(cfgs, Unsup3D)
    trainer.test()
    
    
        
    
        
