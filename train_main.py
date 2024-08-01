import argparse
import os, sys, logging, random, numpy as np
from datetime import datetime
from easydict import EasyDict
import torch
import torch.multiprocessing as mp
import subprocess

from utils.util import set_prefix, write, copy
import train



def print_pass(*args, **kwargs): 
    pass
    return

def print_flush(*args, **kwargs): 
    # No print output from child multiprocessing.Process unless the program crashes
    # Have you tried flushing stdout?
    print(*args, **kwargs, flush=True)
    return

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)



parser = argparse.ArgumentParser(description="Deepfake-FAS: ")


parser.add_argument('--model_name', default='inclusion_coatnet', type=str, help='model name')
parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
parser.add_argument('--loss_weight_class_aux', default=0.1, type=float, help='aux class weight')
parser.add_argument('--batch_size', '-b', default=80, type=int, help='batch size per process/gpu')

parser.add_argument('--exp_name', default="", type=str, help='experiment name')
parser.add_argument('--gpus', default="0,1,2,3,4,5,6,7", type=str, help='gpus string')
parser.add_argument('--data_ratio', '-dr', default=1.0, type=int, help='data_ratio for training exp')
parser.add_argument('--data_dir', '-dd', default="", type=str, help='dataset root dir')
parser.add_argument('--epochs', '-e', default=20, type=int, help='training epochs')
parser.add_argument('--num_workers', default=16*1, type=int, help='training dataloader num_workers')
parser.add_argument('--scheduler_type', default='cosine', type=str, help='learning rate scheduler: cosine,step')
parser.add_argument('--step_size', default=5, type=int, help='learning rate decay interval')
parser.add_argument('--gamma', default=0.2, type=float, help='learning rate decay scope')
parser.add_argument('--cosine_lr_end', default=1e-6, type=float, help='minimum learning rate in cosine scheduler')
parser.add_argument('--print_freq', '-i', default=10, type=int, help='printing log frequence')
parser.add_argument('--prefix', '-p', default='classifier', type=str, help='folder prefix')
parser.add_argument('--log_name', '-l', default='"train_2021-11-10-23:23:55.log"', type=str, help='train log file')
parser.add_argument('--best_model_path', default='model_best.pth.tar', help='best model saved path')
parser.add_argument('--img_size', default=224, type=int, help='img size for training')
parser.add_argument('--num_classes', default=2, type=int, help='model num_classes: real or fake')
parser.add_argument('--is_focal_loss', default=False, type=bool, help='use focal loss or common loss(i.e. cross ectropy loss)(default: true)')



def update_config(cfg):

    cfg.test_first=False  
    cfg.data_ratio=1.0  
    cfg.exp_name="for_full_exp_general"  
    
    cfg.gpus="7"

    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    cfg.prefix = os.path.join(cfg.prefix, cfg.exp_name, cfg.model_name, current_time)
    os.makedirs(cfg.prefix, exist_ok=True)
    set_prefix(cfg.prefix, __file__)  # save source script
    copy_files = ['./dataset_processing.py', './dataset_transforms.py', 'train.py', 'train_main.py']
    save_dir = os.path.join(cfg.prefix, "record_code")
    os.makedirs(save_dir, exist_ok=True)
    for f in copy_files:  # backup
        os.system(f"cp -rf '{f}' '{save_dir}/' ")

    cfg.seed = 1314
    cfg.node_rank = 0
    cfg.node_size = 1
    cfg.ngpus_per_node = len(cfg.gpus.split(','))  # 8, ngpus in each machine
    cfg.world_size = cfg.ngpus_per_node * cfg.node_size  # 8*1, total ngpus in total machines
    cfg.dist_backend = 'nccl'  # 'nccl', 'gloo'
    cfg.dist_url = 'tcp://localhost:234{}'.format(random.randint(10,99))
    return cfg


def main_worker(gpu, cfg):
    if gpu == 0:
        cfg.debug_print = print_flush
    elif gpu != 0:
        cfg.debug_print = print_pass
    else: pass

    print('current gpu_id: {}'.format(gpu))
    cfg.gpu = gpu  # current gpu_id, local_rank in curr node/machine
    cfg.map_loc = 'cuda:{}'.format(cfg.gpu)
    cfg.device  = torch.device(cfg.map_loc)  # xx.to(cfg.device)
    cfg.global_rank = cfg.node_rank * cfg.ngpus_per_node + gpu  # current global_rank

    # print fianl config
    cfg.debug_print('final cfg: \n{}'.format(cfg))

    # init for training
    set_seed(cfg.seed)
    torch.backends.cudnn.benchmark = True  # accelerate the speed of training
    torch.backends.cudnn.deterministic = True
    torch.autograd.set_detect_anomaly(True)

    # distributed, init first
    torch.distributed.init_process_group(
        backend=cfg.dist_backend, init_method=cfg.dist_url, 
        rank=cfg.global_rank, world_size=cfg.world_size)
    
    torch.cuda.set_device(cfg.gpu)  # Must after init_process_group(...)
    device = torch.cuda.current_device()
    print('current gpu device: ', device)
    train.train_worker(cfg)


def main():
    # parse & update args 
    mycfg = update_config(EasyDict(vars(parser.parse_args())))

    device_ids = mycfg.gpus
    os.environ["CUDA_VISIBLE_DEVICES"] = device_ids

    ngpus_per_node = mycfg.ngpus_per_node
    mp.spawn(main_worker, nprocs=ngpus_per_node, args=(mycfg,))




if __name__ == '__main__':
    main()

