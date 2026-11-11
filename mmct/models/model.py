import os
from time import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error, roc_auc_score

import torch
from torch.optim.lr_scheduler import CyclicLR
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
import wandb

from mmct.utils.utils import (Lamb, Lookahead, RobustL1, BCEWithLogitsLoss,
                         EDM_CsvLoader, Scaler, DummyScaler, count_parameters, log_wandb_metrics, rank_zero_debug, rank_zero_info)
from mmct.utils.optim import SWA


# %%
RNG_SEED = 42
torch.manual_seed(RNG_SEED)
np.random.seed(RNG_SEED)
data_type_torch = torch.float32


# %%
class Model():
    def __init__(self,
                 model,
                 fabric,
                 cfg,
                 model_name='UnnamedModel',
                 **kwargs):
        self.model = model
        self.model_name = model_name
        self.fabric = fabric
        self.cfg = cfg
        self.data_loader = None
        self.train_loader = None
        self.classification = False
        self.n_elements = cfg['model']['misc'].get('n_elements', 'infer')
        self.fudge = 0.02  #  expected fractional tolerance (std. dev) ~= 2%
        self.capture_every = cfg['model']['misc'].get('capture_every', None)
        self.verbose = cfg['model']['misc'].get('verbose', True)
        self.drop_unary = cfg['model']['misc'].get('drop_unary', True)
        self.scale = cfg['model']['misc'].get('scale', True)
        self.capture_flag = False
        self.formula_current = None
        self.act_v = None
        self.pred_v = None
        self._log_prefix = f"fold_{self.model_name.split('_')[-1]}"
        if self.verbose:
            rank_zero_debug('\nModel architecture: out_dims, d_model, N, heads')
            rank_zero_debug(f'{self.model.out_dims}, {self.model.d_model}, '
                  f'{self.model.N}, {self.model.heads}')
            rank_zero_debug(f'Model size: {count_parameters(self.model)} parameters\n')
        if self.capture_every is not None:
            rank_zero_debug(f'capturing attention tensors every {self.capture_every}')

    def load_data(self, file_name, batch_size=2**9, train=False):
        self.batch_size = batch_size
        inference = not train
        data_loaders = EDM_CsvLoader(csv_data=file_name,
                                     batch_size=batch_size,
                                     n_elements=self.n_elements,
                                     inference=inference,
                                     verbose=self.verbose,
                                     drop_unary=self.drop_unary,
                                     num_workers=self.cfg['model']['misc'].get('num_workers', 0),
                                     tokens = self.cfg['vocab']['tokens'],
                                     scale_lp = self.cfg['task']['scale_lp'],
                                     data_path =  self.cfg['task']['src'].replace("$root", self.cfg['task']['root']),
                                     scale=self.scale)
        rank_zero_debug(f'loading data with up to {data_loaders.n_elements:0.0f} '
              f'elements in the formula')

        # update n_elements after loading dataset
        self.n_elements = data_loaders.n_elements

        data_loader = data_loaders.get_data_loaders(inference=inference)
        y = data_loader.dataset.data[1]
        if train:
            self.train_len = len(y)
            if self.classification:
                self.scaler = DummyScaler(y)
            else:
                self.scaler = Scaler(y)
            self.train_loader = data_loader
        self.data_loader = data_loader


    def train(self):
        
        self.model.train()
        ti = time()
        minima = []
        for i, data in enumerate(self.train_loader):
            self.step_count += 1
            log_wandb_metrics(self.fabric, {f"{self._log_prefix}/step": self.step_count}, step=self.step_count)
            X, y, formula, sg, lp = data
            y = self.scaler.scale(y)
            src, frac = X.squeeze(-1).chunk(2, dim=1)
            # add a small jitter to the input fractions to improve model
            # robustness and to increase stability
            # frac = frac * (1 + (torch.rand_like(frac)-0.5)*self.fudge)  # uniform
            frac = frac * (1 + (torch.randn_like(frac))*self.fudge)  # normal
            frac = torch.clamp(frac, 0, 1)
            frac[src == 0] = 0
            frac = frac / frac.sum(dim=1).unsqueeze(1).repeat(1, frac.shape[-1])

            src = src.to(dtype=torch.long)
            frac = frac.to(dtype=data_type_torch)
            y = y.to(dtype=data_type_torch)
            sg = sg.to(dtype=torch.long)
            lp = lp.to(dtype=data_type_torch)

            ##################################
            # Force evaluate dataset so that we can capture it in the hook
            # here we are using the train_loader, but we can also use
            # general data_loader
            if self.capture_every == 'step':
                # print('capturing every step!')
                # print(f'data_loader size: {len(self.data_loader.dataset)}')
                self.capture_flag = True
                # (act, pred, formulae, uncert)
                self.act_v, self.pred_v, _, _ = self.predict(self.data_loader)
                self.capture_flag = False
            ##################################

            output = self.model(src, frac, sg_t=sg, x_lp=lp)
            prediction, uncertainty = output.chunk(2, dim=-1)
            loss = self.criterion(prediction.view(-1),
                                  uncertainty.view(-1),
                                  y.view(-1))

            self.fabric.backward(loss)
            loss_value = loss.detach().item()
            log_wandb_metrics(self.fabric, {f"{self._log_prefix}/train/loss": loss_value}, step=self.step_count)
            # Drop references to this batch's graph before the next forward pass.
            del loss, output, prediction, uncertainty
            self.optimizer.step()
            self.optimizer.zero_grad()
            if self.stepping:
                self.lr_scheduler.step()

            swa_check = (self.epochs_step * self.swa_start - 1)
            epoch_check = (self.epoch + 1) % (2 * self.epochs_step) == 0
            learning_time = epoch_check and self.epoch >= swa_check
            if learning_time:
                with torch.no_grad():
                    act_v, pred_v, _, _ = self.predict(self.data_loader)
                mae_v = mean_absolute_error(act_v, pred_v)
                self.optimizer.update_swa(mae_v)
                minima.append(self.optimizer.minimum_found)

        if learning_time and not any(minima):
            self.optimizer.discard_count += 1
            rank_zero_debug(f'Epoch {self.epoch} failed to improve.')
            rank_zero_debug(f'Discarded: {self.optimizer.discard_count}/'
                           f'{self.discard_n} weight updates ♻🗑️')

        dt = time() - ti
        datalen = len(self.train_loader.dataset)
        rank_zero_debug(f'training speed: {datalen/dt:0.3f}')


    def fit(self, epochs=None, checkin=None, losscurve=False):
        assert_train_str = 'Please Load Training Data (self.train_loader)'
        assert_val_str = 'Please Load Validation Data (self.data_loader)'
        assert self.train_loader is not None, assert_train_str
        assert self.data_loader is not None, assert_val_str
        self.loss_curve = {}
        self.loss_curve['train'] = []
        self.loss_curve['val'] = []

        # change epochs_step
        # self.epochs_step = 10
        self.epochs_step = 1
        self.step_size = self.epochs_step * len(self.train_loader)
        rank_zero_debug(f"stepping every {self.step_size} training passes, cycling lr every {self.epochs_step} epochs")
        if epochs is None:
            n_iterations = 1e4
            epochs = int(n_iterations / len(self.data_loader))
            rank_zero_debug(f'running for {epochs} epochs')
        if checkin is None:
            checkin = self.epochs_step * 2
            rank_zero_debug(f'checkin at {self.epochs_step*2} '
                           f'epochs to match lr scheduler')
        if epochs % (self.epochs_step * 2) != 0:
            # updated_epochs = epochs - epochs % (self.epochs_step * 2)
            # rank_zero_info(f'epochs not divisible by {self.epochs_step * 2}, '
            #       f'updating epochs to {updated_epochs} for learning')
            updated_epochs = epochs
            epochs = updated_epochs

        self.step_count = 0
        self.criterion = RobustL1
        if self.classification:
            rank_zero_info("Using BCE loss for classification task")
            self.criterion = BCEWithLogitsLoss
            
        base_optim = Lamb(params=self.model.parameters())
        optimizer = Lookahead(base_optimizer=base_optim)
        optimizer = SWA(optimizer, 
                        swa_start = self.cfg['model']['optim']['SWA']['epoch_start'],
                        swa_lr = self.cfg['model']['optim']['SWA']['swa_lr'],)

        self.model, self.optimizer = self.fabric.setup(self.model, optimizer)
        self.train_loader = self.fabric.setup_dataloaders(self.train_loader)
        self.data_loader = self.fabric.setup_dataloaders(self.data_loader)

        lr_scheduler = CyclicLR(self.optimizer,
                                base_lr= self.cfg['model']['optim']['scheduler']['base_lr'],
                                max_lr= self.cfg['model']['optim']['scheduler']['max_lr'],
                                cycle_momentum=False,
                                step_size_up=self.step_size)

        self.swa_start = self.cfg['model']['optim']['SWA']['epoch_start']
        self.lr_scheduler = lr_scheduler
        self.stepping = True
        self.lr_list = []
        self.xswa = []
        self.yswa = []
        self.discard_n = 3
        pbar = tqdm(range(epochs), position=0, leave=True, desc='Training')
        with logging_redirect_tqdm():
            for epoch in pbar:
                epoch = int(epoch)
                self.epoch = epoch
                log_wandb_metrics(self.fabric, {f"{self._log_prefix}/epoch": self.epoch}, step=self.step_count)
                self.epochs = epochs
                ti = time()
                self.train()
                # print(f'epoch time: {(time() - ti):0.3f}')
                rank_zero_debug(f'Learning rate: {self.optimizer.param_groups[0]["lr"]}')
                self.lr_list.append(self.optimizer.param_groups[0]['lr'])


                ##################################
                # Force evaluate dataset so that we can capture it in the hook
                # here we are using the train_loader, but we can also use
                # general data_loader
                if self.capture_every == 'epoch':
                    # print('capturing every epoch!')
                    # print(f'data_loader size: {len(self.data_loader.dataset)}')
                    self.capture_flag = True
                    # (act, pred, formulae, uncert)
                    self.act_v, self.pred_v, _, _ = self.predict(self.data_loader)
                    self.capture_flag = False
                ##################################

                if (epoch+1) % checkin == 0 or epoch == epochs - 1 or epoch == 0:
                    ti = time()
                    with torch.no_grad():
                        act_t, pred_t, _, _ = self.predict(self.train_loader)
                    dt = time() - ti
                    datasize = len(act_t)
                    # print(f'inference speed: {datasize/dt:0.3f}')
                    mae_t = mean_absolute_error(act_t, pred_t)
                    self.loss_curve['train'].append(mae_t)
                    with torch.no_grad():
                        act_v, pred_v, _, _ = self.predict(self.data_loader)
                    mae_v = mean_absolute_error(act_v, pred_v)
                    self.loss_curve['val'].append(mae_v)
                    log_wandb_metrics(self.fabric, 
                                    {f"{self._log_prefix}/train/MAE_epoch": mae_t,
                                        f"{self._log_prefix}/val/MAE_epoch": mae_v},
                                        step = self.epoch
                                )
                    pbar.set_postfix_str(f"Epoch: {epoch}/{epochs} --- "
                                        f"train mae: {mae_t:0.3g} --- "
                                        f"val mae: {mae_v:0.3g}")
                    epoch_str = f'Epoch: {epoch}/{epochs} ---'
                    train_str = f'train mae: {self.loss_curve["train"][-1]:0.3g}'
                    val_str = f'val mae: {self.loss_curve["val"][-1]:0.3g}'
                    if self.classification:
                        train_auc = roc_auc_score(act_t, pred_t)
                        val_auc = roc_auc_score(act_v, pred_v)
                        train_str = f'train auc: {train_auc:0.3f}'
                        val_str = f'val auc: {val_auc:0.3f}'
                    rank_zero_debug(f'{epoch_str} {train_str} {val_str}')

                    if self.epoch >= (self.epochs_step * self.swa_start - 1):
                        if (self.epoch+1) % (self.epochs_step * 2) == 0:
                            self.xswa.append(self.epoch)
                            self.yswa.append(mae_v)
                            log_wandb_metrics(self.fabric,
                                                    {f"{self._log_prefix}/SWA": mae_v},
                                                    step = self.epoch
                                                )
                
                    if losscurve:
                        xval = np.arange(len(self.loss_curve['val'])) * checkin - 1
                        plt.figure(figsize=(8, 5))
                        xval[0] = 0
                        plt.plot(xval, self.loss_curve['train'],
                                'o-', label='train_mae')
                        plt.plot(xval, self.loss_curve['val'],
                                's--', label='val_mae')
                        plt.plot(self.xswa, self.yswa,
                                'o', ms=12, mfc='none', label='SWA point')
                        plt.ylim(0, 2 * np.mean(self.loss_curve['val']))
                        plt.title(f'{self.model_name}')
                        plt.xlabel('epochs')
                        plt.ylabel('MAE')
                        plt.legend()
                        plt.close()
                        # plt.show()
                    
                if (epoch == epochs-1 or
                    self.optimizer.discard_count >= self.discard_n):
                    # save output df for stats tracking
                    xval = np.arange(len(self.loss_curve['val'])) * checkin - 1
                    xval[0] = 0
                    tval = self.loss_curve['train']
                    vval = self.loss_curve['val']
                    df_loss = pd.DataFrame([xval, tval, vval]).T
                    df_loss.columns = ['epoch', 'train loss', 'val loss']
                    df_loss['swa'] = ['n'] * len(xval)
                    df_loss.loc[df_loss['epoch'].isin(self.xswa), 'swa'] = 'y'
                    if self.cfg['logger']['type'] == 'debug':
                        os.makedirs('figures/lc_data', exist_ok=True)
                        df_loss.to_csv(f'figures/lc_data/{self.model_name}_lc.csv',
                                    index=False)
                    df_loss_table = wandb.Table(dataframe=df_loss)
                    log_wandb_metrics(self.fabric, {f"{self._log_prefix}/loss_table": df_loss_table})


                    # save output learning curve plot
                    plt.figure(figsize=(8, 5))
                    xval = np.arange(len(self.loss_curve['val'])) * checkin - 1
                    xval[0] = 0
                    plt.plot(xval, self.loss_curve['train'],
                            'o-', label='train_mae')
                    plt.plot(xval, self.loss_curve['val'], 's--', label='val_mae')
                    if self.epoch >= (self.epochs_step * self.swa_start - 1):
                        plt.plot(self.xswa, self.yswa,
                                'o', ms=12, mfc='none', label='SWA point')
                    plt.ylim(0, 2 * np.mean(self.loss_curve['val']))
                    plt.title(f'{self.model_name}')
                    plt.xlabel('epochs')
                    plt.ylabel('MAE')
                    plt.legend()
                    log_wandb_metrics(self.fabric, {f"{self._log_prefix}/loss_curve": wandb.Image(plt)})
                    if self.cfg["logger"]["type"] == "debug":
                        plt.savefig(f'figures/lc_data/{self.model_name}_lc.png')
                    plt.close()
                if self.cfg['model']['optim']['early_stopping'] and self.optimizer.discard_count >= self.discard_n:
                    rank_zero_debug(f'Discarded: {self.optimizer.discard_count}/'
                        f'{self.discard_n} weight updates, '
                        f'early-stopping now 🙅🛑')
                    self.optimizer.swap_swa_sgd()
                    break


        # log all lrs in list
        for idx in range(len(self.lr_list)):
            log_wandb_metrics(self.fabric,
                {f"{self._log_prefix}/lr": self.lr_list[idx]},
                step=idx
            )
        if not (self.optimizer.discard_count >= self.discard_n):
            self.optimizer.swap_swa_sgd()


    def predict(self, loader):
        len_dataset = len(loader.dataset)
        n_atoms = int(len(loader.dataset[0][0])/2)
        act = np.zeros(len_dataset)
        pred = np.zeros(len_dataset)
        uncert = np.zeros(len_dataset)
        formulae = np.empty(len_dataset, dtype=list)
        atoms = np.empty((len_dataset, n_atoms))
        fractions = np.empty((len_dataset, n_atoms))
        self.model.eval()
        with torch.no_grad():
            for i, data in enumerate(loader):
                X, y, formula, sg, lp = data
                if self.capture_flag:
                    self.formula_current = None
                    # HACK for PyTorch v1.8.0
                    # this output used to be a list, but is now a tuple
                    if isinstance(formula, tuple):
                        self.formula_current = list(formula)
                    elif isinstance(formula, list):
                        self.formula_current = formula.copy()
                src, frac = X.squeeze(-1).chunk(2, dim=1)
                src = src.to(dtype=torch.long)
                frac = frac.to(dtype=data_type_torch)
                y = y.to(dtype=data_type_torch)
                sg = sg.to(dtype=torch.long)
                lp = lp.to(dtype=data_type_torch)
                output = self.model(src, frac, sg_t=sg, x_lp=lp)
                prediction, uncertainty = output.chunk(2, dim=-1)
                scaler_std = self.scaler.std.to(
                    device=uncertainty.device, dtype=uncertainty.dtype
                )
                uncertainty = torch.exp(uncertainty) * scaler_std
                prediction = self.scaler.unscale(prediction)
                if self.classification:
                    prediction = torch.sigmoid(prediction)

                data_loc = slice(i*self.batch_size,
                                 i*self.batch_size+len(y),
                                 1)

                atoms[data_loc, :] = src.cpu().numpy().astype('int32')
                fractions[data_loc, :] = frac.cpu().numpy().astype('float32')
                act[data_loc] = y.view(-1).cpu().numpy().astype('float32')
                pred[data_loc] = prediction.view(-1).cpu().detach().numpy().astype('float32')
                uncert[data_loc] = uncertainty.view(-1).cpu().detach().numpy().astype('float32')
                formulae[data_loc] = formula
        self.model.train()

        return (act, pred, formulae, uncert)


    def save_network(self, model_name=None):
        if model_name is None:
            model_name = self.model_name
            os.makedirs(f'{self.cfg["logger"]["results"]}/models/trained_models', exist_ok=True)
            path = f'{self.cfg["logger"]["results"]}/models/trained_models/{model_name}.pth'
            rank_zero_debug(f'Saving network ({model_name}) to {path}')
        else:
            path = f'{self.cfg["logger"]["results"]}/models/trained_models/{model_name}.pth'
            rank_zero_debug(f'Saving checkpoint ({model_name}) to {path}')

        save_dict = {'weights': self.model.state_dict(),
                     'scaler_state': self.scaler.state_dict(),
                     'model_name': model_name}
        torch.save(save_dict, path)


    def load_network(self, path):
        # load from logger results directory if it exists
        path = f'{self.cfg["logger"]["results"]}/models/trained_models/{path}'
        network = torch.load(path, map_location=torch.device(self.cfg['device']))
        base_optim = Lamb(params=self.model.parameters())
        optimizer = Lookahead(base_optimizer=base_optim)
        self.optimizer = SWA(optimizer)
        self.scaler = Scaler(torch.zeros(3))
        self.model.load_state_dict(network['weights'])
        self.scaler.load_state_dict(network['scaler_state'])
        self.model_name = network['model_name']


# %%
if __name__ == '__main__':
    pass
