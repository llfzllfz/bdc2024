from data import bdc2024_dataloader
from utils import read_json, set_seed
import torch
import numpy as np
import os

def invoke(input_dir):
    config = read_json()
    set_seed(config['seed'])
    temp_dataloader = bdc2024_dataloader(input_dir, mode='predict', batch_size=1, shuffle=False,
                                         num_workers=config['num_workers'], global_wind = config["temp"]["global_wind"],
                                         wind_direction = config["temp"]["wind_direction"],
                                            MLM = config["temp"]["MLM"],
                                            MLM_mask = config["temp"]["MLM_mask"],
                                            task = "temp", temp_add = config["temp"]["temp_add"])
    wind_dataloader = bdc2024_dataloader(input_dir, mode='predict', batch_size=1, shuffle=False,
                                         num_workers=config['num_workers'], global_wind = config["wind"]["global_wind"],
                                         wind_direction = config["wind"]["wind_direction"],
                                            MLM = config["wind"]["MLM"],
                                            MLM_mask = config["wind"]["MLM_mask"],
                                            task = "wind", temp_add = config["wind"]["temp_add"])
    device = 'cuda'
    model_temp = torch.load('/home/mw/project/model_Transtemp_0.pth', map_location=device)
    model_wind = torch.load('/home/mw/project/model_Transwind_0.pth', map_location=device)
    model_temp.eval()
    model_wind.eval()
    tmp_data = np.load(os.path.join(input_dir, 'temp_lookback.npy'))
    T, C, S, _ = tmp_data.shape
    
    result_temp = np.zeros((T, 24, S, 1))
    result_wind = np.zeros((T, 24, S, 1))
    with torch.no_grad():
        for msg in temp_dataloader:
            station = msg['station'][0]
            samples = msg['samples'][0]
            out_temp = model_temp(msg['global_data'].to(device), msg['temp'].to(device)).cpu().detach().numpy()
            result_temp[samples, :, station, 0] = out_temp
            
        for msg in wind_dataloader:
            station = msg['station'][0]
            samples = msg['samples'][0]
            out_wind = model_wind(msg['global_data'].to(device), msg['wind'].to(device)).cpu().detach().numpy()
            result_wind[samples, :, station, 0] = out_wind

    np.save('/home/mw/project/temp_predict.npy', result_temp)
    np.save('/home/mw/project/wind_predict.npy', result_wind)

if __name__ == '__main__':

    invoke('../cenn_tiny')
    