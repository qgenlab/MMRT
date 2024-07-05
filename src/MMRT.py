import pickle
import torch
import torch.optim as optim

import pandas as pd
import numpy as np
from scipy import stats
from tqdm import tqdm
import os
from collections import namedtuple
import warnings

from model import model
from utils import Logger, Loader, train_test_split


class MMRT:
    def __init__(self,
                 train_data=None,
                 test_data=None,
                 train_ratio=0.9,
                 batch_size=32,
                 save_log=True,
                 resume_log=False,
                 save_model=False,
                 save_path='./',
                 model_name='MMRT',
                 device=None):
        
        if (train_data is not None) and (test_data is None):
            warnings.warn(
                'Using default split ratio (0.9) to perform train-test split. '
                'Adjust by setting train_ratio.'
            )
            
            self.train_data, self.test_data = train_test_split(train_data, train_ratio=train_ratio)
        else:
            self.train_data = train_data
            self.test_data = test_data
        
        self.save_log = save_log
        self.save_model = save_model
        self.save_path = save_path
        self.model_name = model_name
        self.device = torch.device('cuda:' + str(device) if device is not None else 'cpu')
        
        self.logger = Logger(
            path = self.save_path + 'logging/',
            name = self.model_name,
            resume_log = resume_log,
            columns = [f'test_data_{i}' for i in range(len(self.test_data))]
        )
        
        self.model = model(1280, 1, 64).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters())
        self.loss_func= torch.nn.MSELoss()
        
        self.train_dataloader = Loader(self.train_data, True, batch_size)
        self.test_dataloaders = [Loader([t], False, 4096) for t in self.test_data]
        
        if (not os.path.exists(self.save_path)) and (self.save_log or self.save_model):
            os.makedirs(self.save_path)
    
    
    def save_checkpoint(self, checkpoint_path):
        checkpoint = {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }
        
        torch.save(checkpoint, checkpoint_path)
        
    
    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'] )       
        
        
    def train(self,
              epochs=100,
              learning_rate=1e-3,
              few_shot=False,
              random_seed=None,
              cadence=40):            
        
        if random_seed is not None:
            torch.manual_seed(random_seed)
            
        self.optimizer.lr = learning_rate
        
        with tqdm(range(epochs)) as pbar:
            for epoch in pbar:
                self.model.train()
                self.optimizer.zero_grad()
                
                train_loss = []

                for mut_str, vec, act in self.train_dataloader:
                    if np.isnan(act).all():
                        warnings.warn(
                            'Found batch with all-NaN scores. '
                            'This is probably from a dataset without a ground-truth score, '
                            'which cannot be used to train.'
                        )
                        
                        continue
                        
                    mut_count = vec.shape[1]

                    output = self.model(
                        vec[:, :mut_count//2, :].to(self.device, dtype=torch.float64), 
                        vec[:, mut_count//2:, :].to(self.device, dtype=torch.float64)
                    ).squeeze()

                    loss = self.loss_func(
                        output.to(self.device),
                        act.to(self.device).squeeze()
                    )
                    
                    loss.backward()
                    self.optimizer.step()
                    train_loss.append(loss.cpu().detach().item())

                pbar.set_postfix(train_loss=np.mean(train_loss))
        
                if (epoch % cadence == 0):
                    test_results = self.test()
                    self.logger.write([i.correlation for i in test_results.values()])
                    
                    if self.save_model:
                        model_name = self.model_name + f'_{epoch}'
                        model_folder = self.save_path + 'model/'
                        
                        if not os.path.exists(model_folder):
                            os.makedirs(model_folder)
                        
                        self.save_checkpoint(model_folder + model_name + '.p')
        
        # output final results
        self.test(save_prediction = True)
      
    
    def test(self, save_prediction = False):        
        self.model.eval()
        
        output_dict = {}
        output_vals = namedtuple('output_vals', ['correlation', 'loss'])
        
        correlations = []
        predictions = []
        mutation_strs = []
        
        for set_idx, test_set in enumerate(self.test_dataloaders):
            test_loss = []
            output_list = []
            test_y = []
            dict_key = f'test_set_{set_idx}'

            for mut_str, vec, act in test_set:
                ave_test_loss = 0
                mut_count = vec.shape[1]

                pred = self.model(
                    vec[:, :mut_count//2, :].to(self.device, dtype=torch.float64), 
                    vec[:, mut_count//2:, :].to(self.device, dtype=torch.float64)
                ).squeeze()
                
                test_y.extend(act.cpu().tolist())
                output_list.extend(pred.cpu().detach().tolist())
                predictions.extend(pred.cpu().detach().tolist())
                mutation_strs.extend(mut_str)
                
                if not(np.isnan(act).all()):
                    loss = self.loss_func(
                        pred.to(self.device),
                        act.to(self.device).squeeze()
                    )

                    test_loss.append(loss.cpu().detach().item())

            res = stats.spearmanr(test_y, output_list, nan_policy='omit')
            output_dict[dict_key] = output_vals(res.statistic, np.mean(test_loss))
            correlations.append(res.statistic)
        
        if save_prediction:
            results_folder = self.save_path + 'output/'

            if not os.path.exists(results_folder):
                os.makedirs(results_folder)
            
            pd.DataFrame(
                {'mutation': mutation_strs,
                 'prediction': predictions}
            ).to_csv(f'{results_folder}{self.model_name}.csv', index=False)
            
        return output_dict