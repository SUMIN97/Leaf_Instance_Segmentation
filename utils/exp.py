import os
import sys
import torch
import yaml
from easydict import EasyDict as edict
from pathlib import Path

def init_experiment(args):
    cfg = load_config(args.cfg)
    update_config(cfg, args)

    experiments_path = Path(Path.cwd() / cfg.EXPS_PATH)
    experiments_path.mkdir(parents=True, exist_ok=True)

    if cfg.resume_exp:
        exp_path = find_resume_exp(experiments_path, cfg.resume_exp)
    else:
        last_exp_indx = find_last_exp_index(experiments_path)
        exp_name = f'{last_exp_indx:03d}'
        if cfg.exp_name:
            exp_name += '_' + cfg.exp_name
        exp_path = experiments_path / exp_name

        exp_path.mkdir(parents=True)

        cfg.EXP_PATH = exp_path
        cfg.CHECKPOINTS_PATH = exp_path / 'checkpoints'
        cfg.CHECKPOINTS_PATH.mkdir(exist_ok=True)

    #gpu
    device = str(cfg.device).strip().lower().replace('cuda:', '')  # to string, 'cuda:0' to '0'
    cpu = device == 'cpu'
    if cpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    elif device:
        os.environ['CUDA_VISIBLE_DEVICES'] = device
        assert torch.cuda.is_available(), f'CUDA unavailable, invalid device {device} requested'

    gpu_ids = [int(id) for id in device.split(',')]
    cfg.gpu_ids = gpu_ids
    cuda = not cpu and torch.cuda.is_available()
    if cuda:
        cfg.device = torch.device(f'cuda:{cfg.gpu_ids[0]}')
    else:
        cfg.device=torch.device('cpu')

    # save config
    cfg_save_path = exp_path / 'config.yml'
    with cfg_save_path.open('w') as f:
        yaml.dump(cfg, f)

    return cfg


def find_resume_exp(exp_parent_path, exp_pattern):
    candidates = sorted(exp_parent_path.glob(f'{exp_pattern}*'))
    if len(candidates) == 0:
        print(f'No experiments could be found that satisfied the pattenr = "*{exp_pattern}"')
        sys.exit(1)
    else:
        exp_path = candidates[0]
        print(f'Continue with the experiemnt "{exp_path}"')

    return exp_path

def find_last_exp_index(exp_parent_path):
    indx = 0
    for x in exp_parent_path.iterdir():
        if not x.is_dir():
            continue

        exp_name = x.stem
        if exp_name[:3].isnumeric():
            indx = max(indx, int(exp_name[:3]) +1)

    return indx

def update_config(cfg, args):
    for param_name, value in vars(args).items():
        if param_name.lower() in cfg or param_name.upper() in cfg:
            continue
        cfg[param_name] = value

def load_config(config_path):
    try:
        with open(config_path, 'r') as f:
            cfg = yaml.safe_load(f)
    except:
        cfg = dict()

    return edict(cfg)



