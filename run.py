import logging
import os
import fcntl
import json
import hydra
from lightning import Fabric
import numpy as np
from omegaconf import DictConfig
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import wandb
import re

from mmct.models.kingcrab import CrabNet
from mmct.models.model import Model
from mmct.utils.utils import build_logger, checkpoint_saving_enabled, get_expt_choices, log_wandb_metrics, rank_zero_debug, rank_zero_info, setup_logging, to_dict
from mmct.utils.get_compute_device import get_compute_device

compute_device = get_compute_device(force_cpu=False)

# %%
def get_model(task, i, cfg, fabric, batch_size=None,
              transfer=None, verbose=True):
    # Get the TorchedCrabNet architecture loaded
    model_name = f'{cfg["model"]["name"]}{task}_{i}'
    model = Model(CrabNet(compute_device=compute_device, **cfg["model"]["arch"]).to(compute_device),
                    fabric,
                    cfg,
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
    rank_zero_debug(f'training with batchsize {model.batch_size} '
                   f'(2**{np.log2(model.batch_size):0.3f})')
    model.load_data(val_df, batch_size=batch_size)

    # Set the number of epochs, decide if you want a loss curve to be plotted
    model.fit(epochs=cfg["model"]["optim"]["epochs"], losscurve=cfg["model"]["optim"]["losscurve"] or cfg["logger"]["type"] == "online")

    # Save the network (saved as f"{model_name}.pth")
    model.save_network(model_name)
    return model


def to_csv(output_df, save_name):
    # parse output and save to csv
    save_path = f'publication_predictions/{save_name}'
    os.makedirs("/".join(save_path.split("/")[:-1]), exist_ok=True)
    output_df.to_csv(save_path, index_label='Index')


def load_model(task, i, cfg, file_name, fabric, verbose=True):
    # Load up a saved network.
    model_name = f'{cfg["model"]["name"]}{task}{i}'
    model = Model(CrabNet(compute_device=compute_device, **cfg["model"]["arch"]).to(compute_device),
                    fabric,
                    cfg,
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


def save_results(task, i, cfg, file_name, fabric, verbose=True):
    model = load_model(task, i, cfg, file_name, fabric, verbose=verbose)
    model, output = get_results(model)

    # Get appropriate metrics for saving to csv
    mae = np.abs(output[0] - output[1]).mean()

    # save predictions to a csv
    fname = f'{cfg["task"]["root"].split("/")[-1]}/{task}_{file_name.replace(f"{i}.csv", "")}_output_cv{i}.csv'

    act, pred, formulae, uncertainty = output
    output_df = pd.DataFrame([formulae, act, pred, uncertainty]).T
    output_df.columns = ['formula', 'actual', 'predicted', 'uncertainty']
    if cfg["logger"]["type"] == "debug":
        to_csv(output_df, fname)
    else:
        log_wandb_metrics(fabric, {f"fold_{i}/{re.sub(r"\d", "", file_name.split(".")[0])}/predictions": wandb.Table(dataframe=output_df)})
    return model, mae

# run experiment for a specific task
def run_experiment(cfg, model_name, task, log):
    cfg = to_dict(cfg)
    cfg["task"]["name"] = task
    cfg["task"]["data_path"] = cfg["task"]["src"].replace("$root", cfg["task"]["root"])
    cfg["model"]["name"] = model_name
    cfg["logger"]["type"] = log
    cfg["logger"]["name"] = cfg["logger"].get("name") or f"{model_name}_{task}"

    if log == "debug":
        setup_logging(logging.DEBUG, force=True)
    else:
        setup_logging(force=True)

    logger_cfg = cfg["logger"]
    logger_name = logger_cfg["name"]

    save_checkpoints = checkpoint_saving_enabled(cfg)
    is_verbose = logger_cfg["type"] == "debug"

    shared_logObj = build_logger(logger_cfg, log_model=save_checkpoints, config=cfg)

    fabric = Fabric(loggers=shared_logObj, accelerator=compute_device)
    fabric.launch()

    # shared_wandb_experiment = (
    #     shared_logObj.experiment if logger_cfg["type"] == "online" else None
    # )
    # fabric.

    n_splits = cfg["n_splits"]
    t_maes = []
    v_maes = []
    for i in range(n_splits):
        
        rank_zero_info(f'property: {task}, cv {i}')

        rank_zero_debug("Building model for fold %d", i + 1)
                
        model = get_model(task, i, cfg, fabric, verbose=is_verbose)
        rank_zero_debug('=====================================================')
        rank_zero_debug('calculating test mae')
        if cfg["test"]:
            model_test, t_mae = save_results(task, i, cfg,
                                            f'test{i}.csv', fabric, verbose=is_verbose)
            t_maes.append(t_mae)
            log_wandb_metrics(fabric, {f"fold_{i}/test/MAE": t_mae})

        rank_zero_debug('calculating val mae')
        model_val, v_mae = save_results(task, i, cfg,
                                        f'val{i}.csv', fabric, verbose=is_verbose)
        v_maes.append(v_mae)
        log_wandb_metrics(fabric, {f"fold_{i}/val/MAE": v_mae})

    log_wandb_metrics(fabric, {f"avg_val_mae": float(np.mean(v_maes)), f"std_val_mae": float(np.std(v_maes))})
    if cfg["test"]:
        avg_test_mae = float(np.mean(t_maes))
        std_test_mae = float(np.std(t_maes))
        rank_zero_info(f'Average test mae: {avg_test_mae}')
        if cfg["logger"]["type"] == "debug":
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
                    rank_zero_debug(f"Saving results to {results_path}: {updated_results.strip()}")
                    results_file.write(updated_results)
                except BlockingIOError:
                    rank_zero_debug(f"Could not acquire lock on {results_path}. Results may not be saved correctly.")
                finally:
                    fcntl.flock(results_file.fileno(), fcntl.LOCK_UN)
        else:
            log_wandb_metrics(fabric, {f"avg_test_mae": avg_test_mae, f"std_test_mae": std_test_mae})
    rank_zero_debug('=====================================================')


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig):
    RNG_SEED = cfg["seed"]
    torch.manual_seed(RNG_SEED)
    np.random.seed(RNG_SEED)
    model, task, logger = get_expt_choices()
    run_experiment(cfg, model, task, logger)


if __name__ == "__main__":
    main()
