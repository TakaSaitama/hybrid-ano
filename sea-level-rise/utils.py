# basic random seed
import os 
import random
import numpy as np 

SEED = 2024


def seed_basic(seed=SEED):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    

def seed_tf(seed=SEED):
    # tensorflow random seed 
    import tensorflow as tf 
    tf.random.set_seed(seed)
    

def seed_torch(seed=SEED):
    # torch random seed
    import torch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# basic + tensorflow + torch 
def seed_torch(seed=SEED):
    seed_basic(seed)
    seed_tf(seed)
    seedTorch(seed)