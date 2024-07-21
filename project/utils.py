import pandas as pd
import numpy as np
import os
import torch
import random
import torch.backends.cudnn as cudnn
import json
import argparse
from sklearn.metrics import mean_squared_error
from random import randint, sample  

def expand_global_data(global_data):
    new_tmp = np.zeros((global_data.shape[0]*3, global_data.shape[1], global_data.shape[2]))
    index = np.arange(0, global_data.shape[0])
    new_tmp[index * 3] = global_data[index]
    index2 = np.arange(0, global_data.shape[0] - 1)
    new_tmp[index2*3+1] = (global_data[index2 + 1] - global_data[index2]) / 3 + global_data[index2]
    new_tmp[index2*3+2] = (global_data[index2 + 1] - global_data[index2]) * 2 / 3 + global_data[index2]
    new_tmp[-2] = global_data[-1]
    new_tmp[-1] = global_data[-1]
    return new_tmp

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def read_json():
    with open('config.json', 'r') as f:
        config = json.load(f)
    return config

def get_args():
    parser = argparse.ArgumentParser(description='The paramters with the project')
    parser.add_argument('--train', type=str, default = 'all', help='Please choose in [all, wind, temp]')
    parser.add_argument('--test', type=str, default = 'tiny', help = 'Please choose in [tiny, all]')
    parser.add_argument('--data_split', type=int, default = 0)
    args = parser.parse_args()
    return args


def cal_metric(pred, label):
    mse = mean_squared_error(label, pred)
    return mse

class TriangularCausalMask():
    def __init__(self, B, L, device="cpu"):
        mask_shape = [B, 1, L, L]
        with torch.no_grad():
            self._mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1).to(device)

    @property
    def mask(self):
        return self._mask

def global_data_wind(global_data): # T, 4, 9
    new_global_data = np.zeros((global_data.shape[0], global_data.shape[1] + 1, global_data.shape[2]))
    new_global_data[:, [0, 1, 3, 4], :] = global_data[:, :, :]
    new_global_data[:, 2, :] = np.sqrt(global_data[:, 0, :] * global_data[:, 0, :] + global_data[:, 1, :] * global_data[:, 1, :])
    return new_global_data

# def global_data_pos_expand(global_data):
#     global_data = global_data.reshape((global_data.shape[0], global_data.shape[1], 3, 3))
#     new_global_data = np.zeros((global_data.shape[0], global_data.shape[1], 5, 5))
#     new_global_data[:, :, 0, [0, 2, 4]] = global_data[:, :, 0, :]
#     new_global_data[:, :, 2, [0, 2, 4]] = global_data[:, :, 1, :]
#     new_global_data[:, :, 4, [0, 2, 4]] = global_data[:, :, 2, :]
#     new_global_data[:, :, :, [1, 3]] = (new_global_data[:, :, :, [0, 2]] + new_global_data[:, :, :, [2, 4]]) / 2
#     new_global_data[:, :, [1, 3], :] = (new_global_data[:, :, [0, 2], :] + new_global_data[:, :, [2, 4], :]) / 2
#     return new_global_data.reshape(global_data.shape[0], global_data.shape[1], 25)

def global_data_pos_expand(global_data):
    global_data = global_data.reshape((global_data.shape[0], global_data.shape[1], 3, 3))
    new_global_data = np.zeros((global_data.shape[0], global_data.shape[1], 5, 5))
    new_global_data[:, :, 0, [0, 2, 4]] = global_data[:, :, 0, :]
    new_global_data[:, :, 2, [0, 2, 4]] = global_data[:, :, 1, :]
    new_global_data[:, :, 4, [0, 2, 4]] = global_data[:, :, 2, :]
    new_global_data[:, :, :, [1, 3]] = (new_global_data[:, :, :, [0, 2]] + new_global_data[:, :, :, [2, 4]]) / 2
    new_global_data[:, :, [1, 3], :] = (new_global_data[:, :, [0, 2], :] + new_global_data[:, :, [2, 4], :]) / 2
    new_global_data[:, :, 1, 1] = (new_global_data[:, :, 0, 0] + new_global_data[:, :, 2, 2]) / 2
    new_global_data[:, :, 1, 3] = (new_global_data[:, :, 0, 4] + new_global_data[:, :, 2, 2]) / 2
    new_global_data[:, :, 3, 1] = (new_global_data[:, :, 4, 0] + new_global_data[:, :, 2, 2]) / 2
    new_global_data[:, :, 3, 3] = (new_global_data[:, :, 4, 4] + new_global_data[:, :, 2, 2]) / 2
    # return new_global_data.reshape(global_data.shape[0], global_data.shape[1], 25)[:, :, [0, 2, 4, 6, 7, 8, 10, 11, 12, 13, 14, 16, 17, 18, 21, 22, 24]]
    return new_global_data.reshape(global_data.shape[0], global_data.shape[1], 25)[:, :, [6, 7, 8, 11, 12, 13, 16, 17, 18]]

def MLM_mask(tokens, masking_prob=0.15, vocab_size = 1700):
    """  
    随机遮盖tokens，遵循BERT的MLM策略。  
    """  
    output_tokens = []  
    mask_tokens_index = []
    # special_tokens_mask 'CLS' 'SEP'
    special_tokens_mask = [1, 2, 3, 4]  
      
    for idx, token in enumerate(tokens):  
        tok_id = token
        # 跳过特殊tokens  
        if tok_id in special_tokens_mask:  
            output_tokens.append(tok_id)  
            continue  
          
        prob = randint(0, 100) / 100.0  
        if prob < masking_prob:  
            prob /= masking_prob  
              
            # 80% 的时间替换为[MASK]  
            if prob < 0.8:  
                output_tokens.append(3)
            # 10% 的时间替换为随机token  
            elif prob < 0.9:  
                output_tokens.append(randint(0, vocab_size - 1))  
            # 10% 的时间保持不变  
            else:  
                output_tokens.append(tok_id)
            mask_tokens_index.append(idx)
        else:  
            output_tokens.append(tok_id)
    return np.array(output_tokens), np.array(mask_tokens_index)

def get_wind_direction(global_data):
    new_global_data = np.zeros((global_data.shape[0], global_data.shape[1] + 1, global_data.shape[2]))
    U = global_data[:, 0, :]
    V = global_data[:, 1, :]
    angle = np.arctan2(V, U) 
    new_global_data[:, [0, 1, 3, 4, 5], :] = global_data[:, :, :]
    new_global_data[:, 2, :] = angle
    return new_global_data