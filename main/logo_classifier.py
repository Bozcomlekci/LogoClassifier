import mmcls
from mmcls.apis import inference_model, init_model, show_result_pyplot

import os
import torch

import argparse
import copy
import os
import os.path as osp
import time

import mmcv
import torch
from mmcv import Config, DictAction
from mmcv.runner import get_dist_info, init_dist

from mmcls import __version__
from mmcls.apis import set_random_seed, train_model
from mmcls.datasets import build_dataset
from mmcls.models import build_classifier
from mmcls.utils import collect_env, get_root_logger

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

if device.type == 'cuda':
    print(torch.cuda.get_device_name(0))
    print('Memory Usage:')
    print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
    print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')

torch.cuda.empty_cache()


# ## Logo Classification Using MMClassification

# <strong>Modify myconfig.py file to adjust dataset, model and optimizer options as described in [the tutorial](https://mmclassification.readthedocs.io/en/latest/tutorials/finetune.html)</strong>

get_ipython().system('python configs/cls_tools/train.py configs/custom_config/myconfig.py --device cuda')


get_ipython().system('python configs/cls_tools/test.py configs/custom_config/myconfig.py work_dirs/myconfig/latest.pth --metrics=accuracy --metric-options=topk=5')


config_file = 'configs/custom_config/myconfig.py'
checkpoint_file = 'work_dirs/myconfig/latest.pth'
device = 'cuda:0'
model = init_model(config_file, checkpoint_file, device=device)

img = 'examples/exp.jpg'
result = inference_model(model=model, img=img)
print(result)
show_result_pyplot(model, img, result)


# ## Alternative way

# Alternatively traning can be initiated as a python code without bash

############################OPTIONALLY ADJUST CONFIG FILES###############################
#An extensive list of configurations can be found in https://github.com/open-mmlab/mmclassification/tree/master/configs
cls_model = 'resnet/resnet50_b32x8_imagenet.py'
cls_config = os.path.join('configs/configs_cls', cls_model)

get_ipython().system("wget 'https://download.openmmlab.com/mmclassification/v0/resnet/resnet50_batch256_imagenet_20200708-cfb998bf.pth' -P pretrains ")
cls_ckpt = os.path.join('pretrains','resnet50_batch256_imagenet_20200708-cfb998bf.pth')
#########################################################################################

def train( config_path, options = None, work_dir = None, pretrained = None, device = 'cpu'
          gpu_ids = None, gpus = None, launcher = None, seed = None, deterministic = None
          no_validate = None):

    cfg = Config.fromfile(config_path)
    if options is not None:
        cfg.merge_from_dict(options)

    if work_dir is not None:
        cfg.work_dir = work_dir
    elif cfg.get('work_dir', None) is None:
        cfg.work_dir = osp.join('./work_dirs', osp.splitext(osp.basename(config))[0])
        
    if pretrained is not None:
        cfg.resume_from = pretrained
        
    if gpu_ids is not None:
        cfg.gpu_ids = gpu_ids
    else:
        cfg.gpu_ids = range(1) if gpus is None else range(gpus)

    if launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(launcher, **cfg.dist_params)
        _, world_size = get_dist_info()
        gpu_ids = range(world_size)

    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    cfg.dump(osp.join(cfg.work_dir, osp.basename(config)))
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(cfg.work_dir, f'{timestamp}.log')
    logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)

    meta = dict()
    env_info_dict = collect_env()
    env_info = '\n'.join([(f'{k}: {v}') for k, v in env_info_dict.items()])
    dash_line = '-' * 60 + '\n'
    logger.info('Environment info:\n' + dash_line + env_info + '\n' +
                dash_line)
    meta['env_info'] = env_info

    logger.info(f'Distributed training: {distributed}')
    logger.info(f'Config:\n{cfg.pretty_text}')

    if seed is not None:
        logger.info(f'Set random seed to {seed}, '
                    f'deterministic: {deterministic}')
        set_random_seed(seed, deterministic=deterministic)
    cfg.seed = seed
    meta['seed'] = seed

    model = build_classifier(cfg.model)
    model.init_weights()

    datasets = [build_dataset(cfg.data.train)]
    if len(cfg.workflow) == 2:
        val_dataset = copy.deepcopy(cfg.data.val)
        val_dataset.pipeline = cfg.data.train.pipeline
        datasets.append(build_dataset(val_dataset))
    if cfg.checkpoint_config is not None:
        # save mmcls version, config file content and class names in
        # checkpoints as meta data
        cfg.checkpoint_config.meta = dict(
            mmcls_version=__version__,
            config=cfg.pretty_text,
            CLASSES=datasets[0].CLASSES)
    # add an attribute for visualization convenience
    train_model(model,datasets,cfg,
        distributed=distributed,
        validate=(not no_validate),
        timestamp=timestamp,
        device=device,
        meta=meta)

config_path = custom_config/myconfig.py
train( config_path, device='cuda' )

