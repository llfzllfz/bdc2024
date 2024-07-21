from data import bdc2024_dataloader
from utils import read_json, set_seed, get_args
import torch.nn.functional as F

import torch.nn as nn
import torch
from tqdm import tqdm
import logging
import random

class train():
    def __init__(self, train_dataloader, test_dataloader,
                 lr = 0.001, weight_decay = 1e-6, gpu = 2, epochs = 10,
                 early_stop = 5,
                 config = None, args = None):
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.epochs = epochs
        self.early_stop = early_stop
        self.config = config
        self.args = args
        self.device = 'cuda:{}'.format(gpu) if gpu >= 0 else 'cpu'
        
        self.lr = lr
        
        logging.basicConfig(filename='log/log_{}_{}.log'.format(self.args.test, self.args.train), level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.log = logging.getLogger('bdc2024')
        self.log.info(config)
        self.log.info(args)
    
    def init_model(self, task):
        if task == 'temp':
            from iTransformer_temp import Model
            self.model = Model(self.device).to(self.device)
            self.mse = nn.MSELoss()
            self.opt = torch.optim.AdamW(self.model.parameters(), lr = self.lr)
        elif task == 'wind':
            from iTransformer_wind import Model
            self.model = Model(self.device).to(self.device)
            self.mse = nn.MSELoss()
            self.opt = torch.optim.Adam(self.model.parameters(), lr = self.lr)
        

    def train_one_loop(self, task = 'temp'):
        self.model.train()
        all_loss = 0
        for msg in tqdm(self.train_dataloader):
            if self.config[task]["MLM"] == 1:
                out = self.model(msg['global_data'].to(self.device), msg['{}'.format(task)].to(self.device), msg['temp_MLM_input_tokens'].to(self.device), msg['temp_MLM_position_tokens'].to(self.device))
            else:
                out = self.model(msg['global_data'].to(self.device), msg['{}'.format(task)].to(self.device))
            loss = self.mse(out.cpu(), msg['{}_label'.format(task)])
            if config["temp"]["KL_Loss"] != 0:
                P = F.softmax(out.cpu(), dim = -1)
                Q = F.softmax(msg['{}_label'.format(task)], dim = -1)
                kl_div = F.kl_div(P.log(), Q, reduction='batchmean')
                print(kl_div)
                loss = loss + config["temp"]["KL_Loss"] * torch.mean(kl_div)
            
            all_loss = all_loss + loss.item()

            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
        print(all_loss)
        return all_loss
    
    def train_one_task(self, task = 'temp'):
        print('Start task {}...'.format(task))
        early_stop = self.early_stop
        mse = 1e5
        for epoch in range(self.epochs):
            loss = self.train_one_loop(task)
            loss, var_mse = self.eval(task)
            print('Evaluate Loss: {}'.format(loss))
            self.log.info('Epoch: {}\tEvaluate Loss: {}\tEvaluate var_Loss: {}'.format(epoch, loss, var_mse))

            if self.early_stop == -1:
                self.save(task)
                continue

            if mse > loss:
                mse = loss
                self.save(task)
                early_stop = self.early_stop
            else:
                early_stop = early_stop - 1
            if early_stop == 0:
                break

    def eval(self, task = 'temp'):
        self.model.eval()
        all_mse = 0
        var_mse = 0
        counts = 0
        with torch.no_grad():
            for msg in tqdm(self.test_dataloader):
                if self.config[task]["MLM"] == 1:
                    out = self.model(msg['global_data'].to(self.device), msg['{}'.format(task)].to(self.device), msg['temp_MLM_input_tokens'].to(self.device), msg['temp_MLM_position_tokens'].to(self.device))
                else:
                    out = self.model(msg['global_data'].to(self.device), msg['{}'.format(task)].to(self.device))
                loss = self.mse(out.cpu(), msg['{}_label'.format(task)])

                mse = loss.item()
                all_mse = all_mse + mse
                var_mse = var_mse + mse / msg['{}_label'.format(task)].var()
                counts = counts + 1
        return all_mse / counts, var_mse / counts

    def train(self,):
        if self.args.train == 'all' or self.args.train == 'temp':
            self.init_model('temp')
            self.train_one_task('temp')
        if self.args.train == 'all' or self.args.train == 'wind':
            self.init_model('wind')
            self.train_one_task('wind')

    def save(self, task):
        save_path = 'model/model_Trans{}_{}.pth'.format(task, self.args.data_split)
        import dill
        torch.save(self.model, save_path, pickle_module=dill)
        print('finish save model {}'.format(save_path))


if __name__ == '__main__':
    config = read_json()
    args = get_args()
    print(config)
    set_seed(config['seed'])
    print('Start load data...')
    if args.test == 'tiny':
        train_dataloader = bdc2024_dataloader('../train_tiny/', mode='train', batch_size=config['batch_size'], shuffle=True,
                                            num_workers=config['num_workers'], data_split = args.data_split)
        test_dataloader = bdc2024_dataloader('../cenn_tiny/', mode='test', batch_size=config['batch_size'], shuffle=False,
                                            num_workers=config['num_workers'], data_split = args.data_split)
    else:
        train_dataloader = bdc2024_dataloader('../train/', mode='train', batch_size=config['batch_size'], shuffle=True,
                                            num_workers=config['num_workers'], data_split = args.data_split, global_wind = config[args.train]["global_wind"],
                                            wind_direction = config[args.train]["wind_direction"],
                                            MLM = config[args.train]["MLM"],
                                            MLM_mask = config[args.train]["MLM_mask"],
                                            task = args.train, temp_add = config[args.train]["temp_add"])
        test_dataloader = bdc2024_dataloader('../cenn/', mode='test', batch_size=config['batch_size'], shuffle=False,
                                            num_workers=config['num_workers'], data_split = args.data_split, global_wind = config[args.train]["global_wind"],
                                            wind_direction = config[args.train]["wind_direction"],
                                            MLM = config[args.train]["MLM"],
                                            MLM_mask = config[args.train]["MLM_mask"],
                                            task = args.train, temp_add = config[args.train]["temp_add"])

    print('Finish load data...')
    run = train(train_dataloader, test_dataloader, config['lr'],
                config['weight_decay'], config['gpu'], config['epochs'],
                config['early_stop'],
                config, args)
    run.train()
    