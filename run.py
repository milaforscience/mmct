import os
import hydra
import numpy as np
from omegaconf import DictConfig
import pandas as pd
import torch


from crabnet.kingcrab import CrabNet
from crabnet.model import Model
from mmct.utils.utils import get_expt_choices
from utils.get_compute_device import get_compute_device

compute_device = get_compute_device(prefer_last=False)
RNG_SEED = 42
torch.manual_seed(RNG_SEED)
np.random.seed(RNG_SEED)


# %%
def get_model(task, i, cfg, batch_size=None,
              transfer=None, verbose=True):
    # Get the TorchedCrabNet architecture loaded
    model = Model(CrabNet(compute_device=compute_device).to(compute_device),
                  model_name=f'{task}{i}', verbose=verbose)

    # Train network starting at pretrained weights
    if transfer is not None:
        model.load_network(f'{transfer}.pth')
        model.model_name = f'{task}'

    # Get the datafiles you will learn from
    # if train{fold_nuber}.csv and val{fold_number}.csv are in the same folder, then this will work
    # else load the train.csv and get the entries whose Fold column is equal to the fold_number, and do the same for val.csv
    if os.path.exists(f'{cfg.data_dir}/{task}/train{i}.csv') and os.path.exists(f'{cfg.data_dir}/{task}/val{i}.csv'):
        train_df = f'{cfg.data_dir}/{task}/train{i}.csv'
        val_df = f'{cfg.data_dir}/{task}/val{i}.csv'
    elif os.path.exists(f'{cfg.data_dir}/{task}/train.csv') and os.path.exists(f'{cfg.data_dir}/{task}/val.csv'):
        train_data = f'{cfg.data_dir}/{task}/train.csv'
        val_data = f'{cfg.data_dir}/{task}/val.csv'
        train_df = pd.read_csv(train_data)
        val_df = pd.read_csv(val_data)
        if 'Fold' not in train_df.columns or 'Fold' not in val_df.columns:
            raise ValueError(f'Fold column not found in train.csv or val.csv for {task}. Please ensure that the Fold column is present in both files.')
        train_df = train_df[train_df['Fold'] == i]
        val_df = val_df[val_df['Fold'] == i]
    else:
        raise ValueError(f'Data files for {task} not found at {cfg.data_dir}/{task}. Please ensure that train{i}.csv and val{i}.csv or train.csv and val.csv are present in the directory.')

    # Load the train and validation data before fitting the network
    data_size = train_df.shape[0] if isinstance(train_df, pd.DataFrame) else pd.read_csv(train_df).shape[0]
    batch_size = 2**round(np.log2(data_size)-4)
    if batch_size < 2**7:
        batch_size = 2**7
    if batch_size > 2**12:
        batch_size = 2**12
    # batch_size = 2**7
    model.load_data(train_df, batch_size=batch_size, train=True)
    print(f'training with batchsize {model.batch_size} '
          f'(2**{np.log2(model.batch_size):0.3f})')
    model.load_data(val_df, batch_size=batch_size)

    # Set the number of epochs, decide if you want a loss curve to be plotted
    model.fit(epochs=300, losscurve=True)

    # Save the network (saved as f"{model_name}.pth")
    model.save_network()
    return model


def to_csv(output, save_name):
    # parse output and save to csv
    act, pred, formulae, uncertainty = output
    df = pd.DataFrame([formulae, act, pred, uncertainty]).T
    df.columns = ['formula', 'actual', 'predicted', 'uncertainty']
    save_path = 'publication_predictions/mat2vec_matbench__predictions'
    # save_path = 'publication_predictions/onehot_matbench__predictions'
    # save_path = 'publication_predictions/random_200_matbench__predictions'
    os.makedirs(save_path, exist_ok=True)
    df.to_csv(f'{save_path}/{save_name}', index_label='Index')


def load_model(task, i, cfg, file_name, verbose=True):
    # Load up a saved network.
    model = Model(CrabNet(compute_device=compute_device).to(compute_device),
                  model_name=f'{task}{i}', verbose=verbose)
    model.load_network(f'{task}{i}.pth')

    # Load the data you want to predict with
    data = f'{cfg.data_dir}/{task}/{file_name}'
    # data is reloaded to model.data_loader
    model.load_data(data, batch_size=2**9)
    return model


def get_results(model):
    output = model.predict(model.data_loader)  # predict the data saved here
    return model, output


def save_results(task, i, cfg, file_name, verbose=True):
    model = load_model(task, i, cfg, file_name, verbose=verbose)
    model, output = get_results(model)

    # Get appropriate metrics for saving to csv
    mae = np.abs(output[0] - output[1]).mean()
    print(f'\n{task}{i} mae: {mae:0.3g}')

    # save predictions to a csv
    fname = f'{task}_{file_name.replace(f"{i}.csv", "")}_output_cv{i}.csv'
    to_csv(output, fname)
    return model, mae

# run experiment for a specific task
def run_experiment_per_task(cfg, task):
    n_splits = cfg.n_splits
    maes = []
    for i in range(n_splits):
        print(f'property: {task}, cv {i}')
        model = get_model(task, i, cfg, verbose=True)
        print('=====================================================')
        print('calculating test mae')
        model_test, t_mae = save_results(task, i, cfg,
                                            f'test{i}.csv', verbose=False)
        print('calculating val mae')
        model_val, v_mae = save_results(task, i, cfg,
                                        f'val{i}.csv', verbose=False)
        maes.append(t_mae)
    print(f'Average test mae: {np.mean(maes)}')
    print('=====================================================')
    
# %%
def run_experiment(cfg, task = None):
    if task:
        run_experiment_per_task(cfg, task)
    else:
        data_dir = cfg.data_dir
        tasks = os.listdir(data_dir)
        print(f'training: {tasks}')
        for task in tasks:
            # matbench provides 5 dataset train/val splits
            run_experiment_per_task(cfg, task)


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig):
    task = get_expt_choices()
    run_experiment(cfg, task)


if __name__ == "__main__":
    main()
