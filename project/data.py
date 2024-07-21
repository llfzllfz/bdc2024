import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os
from utils import expand_global_data, global_data_wind, global_data_pos_expand, MLM_mask, get_wind_direction

class bdc2024_dataset(Dataset):
    def __init__(self, root_path, mode = 'train', data_split = 0,
                 global_wind = 0, wind_direction = 0,
                 MLM = 1, MLM_mask = 0,
                 task = 'temp', temp_add = 0):
        super().__init__()
        self.data_split = data_split
        self.global_wind = global_wind
        self.wind_direction = wind_direction
        self.MLM = MLM
        self.MLM_mask = MLM_mask
        self.mode = mode
        self.root_path = root_path
        self.task = task
        self.temp_add = temp_add
        self.read_data()
        self.seq_len = 168
        self.pred_len = 24
        self.station_num = self.temp.shape[-2]
        self.atm = 101325

    def read_data(self):
        if self.mode == 'train':
            '''
            global_data -> (T, 4, 9, S)
            temp -> (T, S, 1)
            wind -> (T, S, 1)
            
            '''
            self.global_data = np.load(os.path.join(self.root_path, 'global_data.npy')).astype(np.float32)
            self.temp = np.load(os.path.join(self.root_path, 'temp.npy')).astype(np.float32)
            self.wind = np.load(os.path.join(self.root_path, 'wind.npy')).astype(np.float32)
        elif self.mode == 'test':
            '''
            global_data -> (samples, 56, 4, 9, S)
            temp -> (samples, 168, S, 1)
            wind -> (samples, 168, S, 1)
            temp_label -> (samples, 24, S, 1)
            wind_label -> (samples, 24, S, 1)
            
            '''
            self.global_data = np.load(os.path.join(self.root_path, 'cenn_data.npy')).astype(np.float32)
            self.temp = np.load(os.path.join(self.root_path, 'temp_lookback.npy')).astype(np.float32)
            self.wind = np.load(os.path.join(self.root_path, 'wind_lookback.npy')).astype(np.float32)
            self.temp_label = np.load(os.path.join(self.root_path, 'temp_lookback_label.npy')).astype(np.float32)
            self.wind_label = np.load(os.path.join(self.root_path, 'wind_lookback_label.npy')).astype(np.float32)
        elif self.mode == 'predict':
            self.global_data = np.load(os.path.join(self.root_path, 'cenn_data.npy')).astype(np.float32)
            self.temp = np.load(os.path.join(self.root_path, 'temp_lookback.npy')).astype(np.float32)
            self.wind = np.load(os.path.join(self.root_path, 'wind_lookback.npy')).astype(np.float32)

    def __len__(self):
        if self.mode == 'train':
            return (self.global_data.shape[0] - self.seq_len - self.pred_len + 1) * self.global_data.shape[-1]
        elif self.mode == 'test':
            return self.global_data.shape[0] * self.global_data.shape[-1] // 5
        elif self.mode == 'predict':
            return self.global_data.shape[0] * self.global_data.shape[-1]
    
    def __getitem__(self, idx):
        if self.mode == 'train':
            station = idx % self.station_num
            samples = idx // self.station_num
            samples = samples
            global_data = self.global_data[samples : samples + self.seq_len // 3, :, :, station]
            
            
            temp = self.temp[samples * 3 : samples * 3 + self.seq_len, station, :]
            wind = self.wind[samples * 3 : samples * 3 + self.seq_len, station, :]

            temp_label = self.temp[samples * 3 + self.seq_len : samples * 3 + self.seq_len + self.pred_len, station, :]
            wind_label = self.wind[samples * 3 + self.seq_len : samples * 3 + self.seq_len + self.pred_len, station, :]

            global_data = self.cal_global_data(global_data, temp, wind)

            result = {
                'global_data': torch.FloatTensor(global_data.astype(np.float64)), # seq_len, 4, 9
                'temp': torch.FloatTensor(temp.astype(np.float64)), # seq_len, 1
                'wind': torch.FloatTensor(wind.astype(np.float64)), # seq_len, 1
                'temp_label': torch.FloatTensor(temp_label.astype(np.float64)).squeeze(-1), # pred_len, 1
                'wind_label': torch.FloatTensor(wind_label.astype(np.float64)).squeeze(-1), # pred_len, 1,
            }

            if self.MLM == 1:
                temp_MLM_tokens = ((temp + 100) * 10).astype(np.int32).reshape(-1).tolist()
                input_tokens, position_tokens, label_tokens = self.cal_MLM(temp_MLM_tokens)
                result['temp_MLM_input_tokens'] = torch.IntTensor(input_tokens)
                result['temp_MLM_label_tokens'] = torch.LongTensor(label_tokens)
                result['temp_MLM_position_tokens'] = torch.IntTensor(position_tokens)

        elif self.mode == 'test':
            station = idx % self.station_num
            samples = idx // self.station_num
            samples = samples * 5
            global_data = self.global_data[samples, :, :, :, station]
            

            temp = self.temp[samples, :, station, :]
            wind = self.wind[samples, :, station, :]

            temp_label = self.temp_label[samples, :, station, :]
            wind_label = self.wind_label[samples, :, station, :]

            global_data = self.cal_global_data(global_data, temp, wind)

            result = {
                'global_data': torch.FloatTensor(global_data.astype(np.float64)), # seq_len, 4, 9
                'temp': torch.FloatTensor(temp.astype(np.float64)), # seq_len, 1
                'wind': torch.FloatTensor(wind.astype(np.float64)), # seq_len, 1
                'temp_label': torch.FloatTensor(temp_label.astype(np.float64)).squeeze(-1), # pred_len, 1
                'wind_label': torch.FloatTensor(wind_label.astype(np.float64)).squeeze(-1), # pred_len, 1
                'station': station,
                'samples': samples
            }

            if self.MLM == 1:
                temp_MLM_tokens = ((temp + 100) * 10).astype(np.int32).reshape(-1).tolist()
                input_tokens, position_tokens, label_tokens = self.cal_MLM(temp_MLM_tokens)
                result['temp_MLM_input_tokens'] = torch.IntTensor(input_tokens)
                result['temp_MLM_label_tokens'] = torch.LongTensor(label_tokens)
                result['temp_MLM_position_tokens'] = torch.IntTensor(position_tokens)

        elif self.mode == 'predict':
            station = idx % self.station_num
            samples = idx // self.station_num
            global_data = self.global_data[samples, :, :, :, station]

            temp = self.temp[samples, :, station, :]
            wind = self.wind[samples, :, station, :]

            global_data = self.cal_global_data(global_data, temp, wind)

            result = {
                'global_data': torch.FloatTensor(global_data.astype(np.float64)), # seq_len, 4, 9
                'temp': torch.FloatTensor(temp.astype(np.float64)), # seq_len, 1
                'wind': torch.FloatTensor(wind.astype(np.float64)), # seq_len, 1
                'station': station,
                'samples': samples
            }

            if self.MLM == 1:
                temp_MLM_tokens = ((temp + 100) * 10).astype(np.int32).reshape(-1).tolist()
                input_tokens, position_tokens, label_tokens = self.cal_MLM(temp_MLM_tokens)
                result['temp_MLM_input_tokens'] = torch.IntTensor(input_tokens)
                result['temp_MLM_label_tokens'] = torch.LongTensor(label_tokens)
                result['temp_MLM_position_tokens'] = torch.IntTensor(position_tokens)

        return result

    def cal_global_data(self, global_data, temp = None, wind = None):
        global_data = expand_global_data(global_data)
        if self.global_wind == 1:
            global_data = global_data_wind(global_data)
        if self.wind_direction == 1:
            global_data = get_wind_direction(global_data)
        # global_data = global_data[:, :3, :]
        if temp is not None and self.temp_add == 1:
            global_data = torch.FloatTensor(global_data)
            z1 = global_data[:, 2, 4].unsqueeze(1) - torch.FloatTensor(temp)
            tmp = global_data[:, 2, :] - z1[:, :]
            global_data = torch.cat([global_data, tmp.unsqueeze(1)], dim = 1)
            global_data = global_data.numpy()
        return global_data
    
    def cal_MLM(self, temp_MLM_tokens):
        temp_MLM_tokens = [1] + temp_MLM_tokens + [2]
        if self.MLM_mask == 1:
            input_tokens, mask_tokens_index = MLM_mask(temp_MLM_tokens)
        else:
            input_tokens = np.array(temp_MLM_tokens)
        temp_MLM_tokens = np.array(temp_MLM_tokens)
        label_tokens = [-100] * len(temp_MLM_tokens)
        label_tokens = np.array(label_tokens)
        if self.MLM_mask == 1:
            label_tokens[mask_tokens_index] = temp_MLM_tokens[mask_tokens_index]
        position_tokens = np.arange(0, 170)
        return input_tokens, position_tokens, label_tokens

def bdc2024_dataloader(root_path, mode,
                       batch_size = 32, shuffle = True, num_workers = 0,
                       data_split = 0,
                       global_wind = 0, wind_direction = 0,
                       MLM = 1, MLM_mask = 0,
                       task = 'temp', temp_add = 0):
    dataset = bdc2024_dataset(root_path, mode, data_split, global_wind, wind_direction, MLM, MLM_mask, task, temp_add)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return dataloader