import os
import fcntl
import json
import hydra
import numpy as np
from omegaconf import DictConfig
import pandas as pd
from sklearn.model_selection import train_test_split
import torch


from crabnet.kingcrab import CrabNet
from crabnet.model import Model
from utils.utils import get_expt_choices, to_dict
from utils.get_compute_device import get_compute_device

compute_device = get_compute_device(prefer_last=False)

# %%
def get_model(task, i, cfg, batch_size=None,
              transfer=None, verbose=True):
    # Get the TorchedCrabNet architecture loaded
    model_name = f'{cfg["model"]["name"]}{task}{i}'
    model = Model(CrabNet(compute_device=compute_device, **cfg["model"]["arch"]).to(compute_device),
                  model_name=model_name, 
                  **cfg["model"]["misc"]
                  )

    # Train network starting at pretrained weights
    if transfer is not None:
        model.load_network(f'{transfer}.pth')
        model.model_name = f'{task}'

    # Get the datafiles you will learn from
    # if train{fold_nuber}.csv and val{fold_number}.csv are in the same folder, then this will work
    # else load the train.csv and get the entries whose Fold column is equal to the fold_number, and do the same for val.csv
    if os.path.exists(f'{cfg["task"]["data_path"]}/train{i}.csv') and os.path.exists(f'{cfg["task"]["data_path"]}/val{i}.csv'):
        train_df = f'{cfg["task"]["data_path"]}/train{i}.csv'
        val_df = f'{cfg["task"]["data_path"]}/val{i}.csv'
    elif os.path.exists(f'{cfg["task"]["data_path"]}/train_data.csv'):
        train_data = f'{cfg["task"]["data_path"]}/train_data.csv'
        train_df = pd.read_csv(train_data)
        if os.path.exists(f'{cfg["task"]["data_path"]}/val_data.csv'):
            val_data = f'{cfg["task"]["data_path"]}/val_data.csv'
            val_df = pd.read_csv(val_data)
        else:
            # split using the split ration from config using sklearn split
            split_ratio = cfg["val_split_ratio"]
            train_df, val_df = train_test_split(train_df, test_size=split_ratio, random_state=cfg["seed"])

        if 'Fold' not in train_df.columns or 'Fold' not in val_df.columns:
            raise ValueError(f'Fold column not found in train.csv or val.csv for {task}. Please ensure that the Fold column is present in both files.')
        train_df = train_df[train_df['Fold'] == i]
        val_df = val_df[val_df['Fold'] == i]
        if "Formulae" in train_df.columns:
            # rename to formula
            train_df = train_df.rename(columns={"Formulae": "formula", cfg["task"]["target"]: "target"})
        if "Formulae" in val_df.columns:
            val_df = val_df.rename(columns={"Formulae": "formula", cfg["task"]["target"]: "target"})
        # select only formula and target columns
        train_df = train_df[['formula', 'target']]
        val_df = val_df[['formula', 'target']]
    else:
        raise ValueError(f'Data files for {task} not found at {cfg["task"]["data_path"]}. Please ensure that train{i}.csv and val{i}.csv or train.csv and val.csv are present in the directory.')

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
    model.fit(epochs=cfg["model"]["optim"]["epochs"], losscurve=cfg["model"]["optim"]["losscurve"])

    # Save the network (saved as f"{model_name}.pth")
    model.save_network(model_name)
    return model


def to_csv(output, save_name):
    # parse output and save to csv
    act, pred, formulae, uncertainty = output
    df = pd.DataFrame([formulae, act, pred, uncertainty]).T
    df.columns = ['formula', 'actual', 'predicted', 'uncertainty']
    save_path = f'publication_predictions/{save_name}'
    os.makedirs("/".join(save_path.split("/")[:-1]), exist_ok=True)
    df.to_csv(save_path, index_label='Index')


def load_model(task, i, cfg, file_name, verbose=True):
    # Load up a saved network.
    model_name = f'{cfg["model"]["name"]}{task}{i}'
    model = Model(CrabNet(compute_device=compute_device, **cfg["model"]["arch"]).to(compute_device),
                  model_name=model_name, 
                  **cfg["model"]["misc"]
            )
    model.load_network(f'{model_name}.pth')

    # Load the data you want to predict with
    if not os.path.exists(f'{cfg["task"]["data_path"]}/{file_name}'):
        file_name = 'test_data.csv'
    data = f'{cfg["task"]["data_path"]}/{file_name}'
    # data is reloaded to model.data_loader
    data = pd.read_csv(data)
    if 'Fold' in data.columns:
        data = data[data['Fold'] == i]
        if "Formulae" in data.columns:
            # rename to formula
            data = data.rename(columns={"Formulae": "formula", cfg["task"]["target"]: "target"})
        # select only formula and target columns
        data = data[['formula', 'target']]
    
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
    fname = f'{cfg["task"]["root"].split("/")[-1]}/{task}_{file_name.replace(f"{i}.csv", "")}_output_cv{i}.csv'
    to_csv(output, fname)
    return model, mae

# run experiment for a specific task
def run_experiment(cfg, model, task):
    cfg = to_dict(cfg)
    cfg["task"]["name"] = task
    cfg["task"]["data_path"] = cfg["task"]["src"].replace("$root", cfg["task"]["root"])
    cfg["model"]["name"] = model

    n_splits = cfg["n_splits"]
    maes = []
    for i in range(n_splits):
        print(f'property: {task}, cv {i}')
        model = get_model(task, i, cfg, verbose=True)
        print('=====================================================')
        print('calculating test mae')
        if cfg["test"]:
            model_test, t_mae = save_results(task, i, cfg,
                                            f'test{i}.csv', verbose=False)
            maes.append(t_mae)

        print('calculating val mae')
        model_val, v_mae = save_results(task, i, cfg,
                                        f'val{i}.csv', verbose=False)
    if cfg["test"]:
        avg_test_mae = float(np.mean(maes))
        std_test_mae = float(np.std(maes))
        print(f'Average test mae: {avg_test_mae}')
        results_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), f'{cfg["task"]["root"].split("/")[-1]}_test.txt')
        # Keep parallel task runs from overwriting each other's results.
        with open(results_path, 'a+', encoding='utf-8') as results_file:
            try:
                fcntl.flock(results_file.fileno(), fcntl.LOCK_EX)
                results_file.seek(0)
                contents = results_file.read()
                results = json.loads(contents) if contents.strip() else {}
                results[task] = f"{avg_test_mae} ± {std_test_mae}"
                updated_results = json.dumps(results, indent=4, ensure_ascii=False) + '\n'
                results_file.truncate(0)
                print(f"Saving results to {results_path}: {updated_results.strip()}")
                results_file.write(updated_results)
            except BlockingIOError:
                print(f"Could not acquire lock on {results_path}. Results may not be saved correctly.")
            finally:
                fcntl.flock(results_file.fileno(), fcntl.LOCK_UN)
    print('=====================================================')


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig):
    RNG_SEED = cfg["seed"]
    torch.manual_seed(RNG_SEED)
    np.random.seed(RNG_SEED)
    model, task = get_expt_choices()
    run_experiment(cfg, model, task)


if __name__ == "__main__":
    main()
