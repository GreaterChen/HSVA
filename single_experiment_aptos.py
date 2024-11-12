
### execute this function to train and test the vae-model

from HSVA import Model
import numpy as np
import pickle
import torch
import os
import argparse
import time
import warnings
warnings.filterwarnings("ignore")

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser()

parser.add_argument('--dataset', default="APTOS")
parser.add_argument('--num_shots',type=int, default=0)
parser.add_argument('--generalized', type = str2bool, default=True)
args = parser.parse_args()


########################################
# the basic hyperparameters
########################################
hyperparameters = {
    'num_shots': 0,
    'device': 'cuda',
    'model_specifics': {'cross_reconstruction': True,
                       'name': 'CUDA',
                       'distance': 'wasserstein',
                       'warmup': {'beta': {'factor': 0.25,
                                           'end_epoch': 90,
                                           'start_epoch': 0},
                                  'cross_reconstruction': {'factor': 2.37,
                                                           'end_epoch': 75,
                                                           'start_epoch': 21},
                                  'distance': {'factor': 8.0,
                                               'end_epoch': 25,
                                               'start_epoch': 0}}},

    'lr_gen_model': 0.000015,
    'generalized': True,
    'batch_size': 8,
    'samples_per_class': {'SUN': (200, 0, 400, 0),
                          'APY': (200, 0, 400, 0),
                          'CUB': (200, 0, 400, 0),
                          'AWA2': (200, 0, 400, 0),
                          'FLO': (200, 0, 400, 0),
                          'AWA1': (200, 0, 400, 0),
                          'ZDFY': (200, 0, 400, 0),
                          'ADNI': (200, 0, 400, 0),
                          'APTOS': (200, 0, 400, 0)},
    'epochs': 200,
    'loss': 'l1',
    'auxiliary_data_source' : 'attributes',
    'lr_cls': 0.0001,
    'dataset': 'APTOS',
    'hidden_size_rule': {'resnet_features': (4096, 4096),
                        'attributes': (4096, 4096),
                        'sentences': (4096, 4096) },
    'coarse_latent_size': 2048,
    'latent_size': 64, ##64 for CUB,AWA; 128 for SUN
    'recon_x_cyc_w': 0.5,
    'adapt_mode': 'SWD',               #MCD or SWD
    'classifier': 'softmax',          #softmax
    'result_root': '/home/chenlb/compare_model/HSVA/model/result'
}

# The training epochs for the final classifier, for early stopping,
# as determined on the validation spit

cls_train_steps = [
      {'dataset': 'SUN',  'num_shots': 0, 'generalized': True, 'cls_train_steps': 21},
      {'dataset': 'ZDFY',  'num_shots': 0, 'generalized': True, 'cls_train_steps': 21},
      {'dataset': 'APTOS',  'num_shots': 0, 'generalized': True, 'cls_train_steps': 21},
      {'dataset': 'ADNI',  'num_shots': 0, 'generalized': True, 'cls_train_steps': 21},
      {'dataset': 'SUN',  'num_shots': 0, 'generalized': False, 'cls_train_steps': 30},
      {'dataset': 'SUN',  'num_shots': 1, 'generalized': True, 'cls_train_steps': 22},
      {'dataset': 'SUN',  'num_shots': 1, 'generalized': False, 'cls_train_steps': 96},
      {'dataset': 'SUN',  'num_shots': 5, 'generalized': True, 'cls_train_steps': 29},
      {'dataset': 'SUN',  'num_shots': 5, 'generalized': False, 'cls_train_steps': 78},
      {'dataset': 'SUN',  'num_shots': 2, 'generalized': True, 'cls_train_steps': 29},
      {'dataset': 'SUN',  'num_shots': 2, 'generalized': False, 'cls_train_steps': 61},
      {'dataset': 'SUN',  'num_shots': 10, 'generalized': True, 'cls_train_steps': 79},
      {'dataset': 'SUN',  'num_shots': 10, 'generalized': False, 'cls_train_steps': 94},
      {'dataset': 'AWA1', 'num_shots': 0, 'generalized': True, 'cls_train_steps': 33},
      {'dataset': 'AWA1', 'num_shots': 0, 'generalized': False, 'cls_train_steps': 25},
      {'dataset': 'AWA1', 'num_shots': 1, 'generalized': True, 'cls_train_steps': 40},
      {'dataset': 'AWA1', 'num_shots': 1, 'generalized': False, 'cls_train_steps': 81},
      {'dataset': 'AWA1', 'num_shots': 5, 'generalized': True, 'cls_train_steps': 89},
      {'dataset': 'AWA1', 'num_shots': 5, 'generalized': False, 'cls_train_steps': 62},
      {'dataset': 'AWA1', 'num_shots': 2, 'generalized': True, 'cls_train_steps': 56},
      {'dataset': 'AWA1', 'num_shots': 2, 'generalized': False, 'cls_train_steps': 59},
      {'dataset': 'AWA1', 'num_shots': 10, 'generalized': True, 'cls_train_steps': 100},
      {'dataset': 'AWA1', 'num_shots': 10, 'generalized': False, 'cls_train_steps': 50},
      {'dataset': 'CUB',  'num_shots': 0, 'generalized': True, 'cls_train_steps': 23},
      {'dataset': 'CUB',  'num_shots': 0, 'generalized': False, 'cls_train_steps': 22},
      {'dataset': 'CUB',  'num_shots': 1, 'generalized': True, 'cls_train_steps': 34},
      {'dataset': 'CUB',  'num_shots': 1, 'generalized': False, 'cls_train_steps': 46},
      {'dataset': 'CUB',  'num_shots': 5, 'generalized': True, 'cls_train_steps': 64},
      {'dataset': 'CUB',  'num_shots': 5, 'generalized': False, 'cls_train_steps': 73},
      {'dataset': 'CUB',  'num_shots': 2, 'generalized': True, 'cls_train_steps': 39},
      {'dataset': 'CUB',  'num_shots': 2, 'generalized': False, 'cls_train_steps': 31},
      {'dataset': 'CUB',  'num_shots': 10, 'generalized': True, 'cls_train_steps': 85},
      {'dataset': 'CUB',  'num_shots': 10, 'generalized': False, 'cls_train_steps': 67},
      {'dataset': 'AWA2', 'num_shots': 0, 'generalized': True, 'cls_train_steps': 50},
      {'dataset': 'AWA2', 'num_shots': 0, 'generalized': False, 'cls_train_steps': 39},
      {'dataset': 'AWA2', 'num_shots': 1, 'generalized': True, 'cls_train_steps': 44},
      {'dataset': 'AWA2', 'num_shots': 1, 'generalized': False, 'cls_train_steps': 96},
      {'dataset': 'AWA2', 'num_shots': 5, 'generalized': True, 'cls_train_steps': 99},
      {'dataset': 'AWA2', 'num_shots': 5, 'generalized': False, 'cls_train_steps': 29},
      {'dataset': 'AWA2', 'num_shots': 2, 'generalized': True, 'cls_train_steps': 69},
      {'dataset': 'AWA2', 'num_shots': 2, 'generalized': False, 'cls_train_steps': 79},
      {'dataset': 'AWA2', 'num_shots': 10, 'generalized': True, 'cls_train_steps': 86},
      {'dataset': 'AWA2', 'num_shots': 10, 'generalized': False, 'cls_train_steps': 78},
      {'dataset': 'APY',  'num_shots': 0, 'generalized': True, 'cls_train_steps': 23},
      {'dataset': 'APY',  'num_shots': 0, 'generalized': False, 'cls_train_steps': 22},
      {'dataset': 'APY',  'num_shots': 1, 'generalized': True, 'cls_train_steps': 34},
      {'dataset': 'APY',  'num_shots': 1, 'generalized': False, 'cls_train_steps': 46},
      {'dataset': 'APY',  'num_shots': 5, 'generalized': True, 'cls_train_steps': 64},
      {'dataset': 'APY',  'num_shots': 5, 'generalized': False, 'cls_train_steps': 73},
      {'dataset': 'APY',  'num_shots': 2, 'generalized': True, 'cls_train_steps': 39},
      {'dataset': 'APY',  'num_shots': 2, 'generalized': False, 'cls_train_steps': 31},
      {'dataset': 'APY',  'num_shots': 10, 'generalized': True, 'cls_train_steps': 85},
      {'dataset': 'APY',  'num_shots': 10, 'generalized': False, 'cls_train_steps': 67},
      {'dataset': 'FLO',  'num_shots': 0, 'generalized': True, 'cls_train_steps': 23},
      {'dataset': 'FLO',  'num_shots': 0, 'generalized': False, 'cls_train_steps': 22},
      {'dataset': 'FLO',  'num_shots': 1, 'generalized': True, 'cls_train_steps': 34},
      {'dataset': 'FLO',  'num_shots': 1, 'generalized': False, 'cls_train_steps': 46},
      {'dataset': 'FLO',  'num_shots': 5, 'generalized': True, 'cls_train_steps': 64},
      {'dataset': 'FLO',  'num_shots': 5, 'generalized': False, 'cls_train_steps': 73},
      {'dataset': 'FLO',  'num_shots': 2, 'generalized': True, 'cls_train_steps': 39},
      {'dataset': 'FLO',  'num_shots': 2, 'generalized': False, 'cls_train_steps': 31},
      {'dataset': 'FLO',  'num_shots': 10, 'generalized': True, 'cls_train_steps': 85},
      {'dataset': 'FLO',  'num_shots': 10, 'generalized': False, 'cls_train_steps': 67},
      ]

##################################
# change some hyperparameters here
##################################
hyperparameters['dataset'] = args.dataset
hyperparameters['num_shots']= args.num_shots
hyperparameters['generalized']= args.generalized

hyperparameters['cls_train_steps'] = [x['cls_train_steps']  for x in cls_train_steps
                                        if all([hyperparameters['dataset']==x['dataset'],
                                        hyperparameters['num_shots']==x['num_shots'],
                                        hyperparameters['generalized']==x['generalized'] ])][0]

print('***')
print(hyperparameters['cls_train_steps'] )
if hyperparameters['generalized']:
    if hyperparameters['num_shots']==0:
        hyperparameters['samples_per_class'] = {'CUB': (200, 0, 400, 0), 'SUN': (200, 0, 400, 0),
                                'APY': (200, 0,  400, 0), 'AWA1': (200, 0, 400, 0),
                                'AWA2': (200, 0, 400, 0), 'FLO': (200, 0, 400, 0), 'ZDFY': (200, 0, 400, 0),
                                'ADNI': (200, 0, 400, 0), "APTOS":(200, 0, 400, 0)}
    else:
        hyperparameters['samples_per_class'] = {'CUB': (200, 0, 200, 200), 'SUN': (200, 0, 200, 200),
                                                    'APY': (200, 0, 200, 200), 'AWA1': (200, 0, 200, 200),
                                                    'AWA2': (200, 0, 200, 200), 'FLO': (200, 0, 200, 200)}
else:
    if hyperparameters['num_shots']==0:
        hyperparameters['samples_per_class'] = {'CUB': (0, 0, 400, 0), 'SUN': (0, 0, 200, 0),
                                                    'APY': (0, 0, 200, 0), 'AWA1': (0, 0, 800, 0),
                                                    'AWA2': (0, 0, 200, 0), 'FLO': (0, 0, 200, 0)}
    else:
        hyperparameters['samples_per_class'] = {'CUB': (0, 0, 200, 200), 'SUN': (0, 0, 200, 200),
                                                    'APY': (0, 0, 200, 200), 'AWA1': (0, 0, 200, 200),
                                                    'AWA2': (0, 0, 200, 200), 'FLO': (0, 0, 200, 200)}

# model = torch.load("/home/LAB/chenlb24/compare_model/HSVA/model/result/ZDFY/model_full.pth")
model = Model( hyperparameters)
model.to(hyperparameters['device'])

"""
########################################
### load model where u left
########################################
saved_state = torch.load('./saved_models/CADA_trained.pth.tar')
model.load_state_dict(saved_state['state_dict'])
for d in model.all_data_sources_without_duplicates:
    model.encoder[d].load_state_dict(saved_state['encoder'][d])
    model.decoder[d].load_state_dict(saved_state['decoder'][d])
########################################
"""


start = time.time()

model.train_vae()

torch.save(model, '/home/chenlb/compare_model/HSVA/model/result/APTOS/model_full.pth')

time_used = time.time()- start

print("time used:", time_used)
print(time.strftime('ending time:%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
print("dataset", args.dataset)
print(hyperparameters['classifier'])
print("**********END*******************")
