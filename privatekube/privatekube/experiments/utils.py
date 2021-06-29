from absl import flags, logging
import yaml
import json
import os
import pandas as pd
import time
import torch
import numpy as np
import gcsfs

FLAGS = flags.FLAGS
GCP_PROJECT = "project-id-1234"


def build_flags(*arg_dicts):
    """
    Declares Absl flags from a dictionary.
    Flags are associated with this privatekube module, but shared with everyone.
    """
    for arg_dict in arg_dicts:
        for arg_name, default_value in arg_dict.items():
            if type(default_value) == bool:
                flags.DEFINE_bool(arg_name, default_value, arg_name)
            if type(default_value) == int:
                flags.DEFINE_integer(arg_name, default_value, arg_name)
            if type(default_value) == float:
                flags.DEFINE_float(arg_name, default_value, arg_name)
            if type(default_value) == str:
                flags.DEFINE_string(arg_name, default_value, arg_name)


def flags_to_dict(*arg_dicts):
    """
    Returns a dict with the value of the flags we care about.
    """
    result = {}
    for arg_dict in arg_dicts:
        for arg_name in arg_dict.keys():
            result[arg_name] = getattr(FLAGS, arg_name)
    return result


def results_to_dict(
    train_size=None,
    test_size=None,
    dataset_files=None,
    training_time=None,
    epsilon=None,
    delta=None,
    mse=None,
    rmse=None,
    rmsle=None,
    accuracy=None,
    loss=None,
):

    dict_results = {
        "train_size": train_size,
        "test_size": test_size,
        "dataset_files": dataset_files,
        "training_time": training_time,
        "epsilon": epsilon,
        "delta": delta,
        "mse": mse,
        "rmse": rmse,
        "rmsle": rmsle,
        "accuracy": accuracy,
    }

    return dict_results


def save_model(model_path, model):
    model_dir = os.path.dirname(model_path)
    if not os.path.exists(model_dir) and model_dir != "":
        os.makedirs(model_dir)
        logging.debug(f"Created directory: {model_dir}.")
    torch.save(model.state_dict(), model_path)


def save_yaml(yaml_path, dict_results):
    yaml_dir = os.path.dirname(yaml_path)
    if not os.path.exists(yaml_dir) and yaml_dir != "":
        os.makedirs(yaml_dir)
        logging.debug(f"Created directory: {yaml_dir}.")
    with open(yaml_path, "w") as f:
        yaml.dump(dict_results, f)
        logging.debug(f"Wrote yaml: {yaml_path}")


def raw_flags_to_dict(flags):
    d = {}
    for attr, flag_obj in flags.__flags.items():
        d[attr] = flag_obj.value
    return d


def load_yaml(yaml_path):
    with open(yaml_path, "r") as f:
        configs = yaml.safe_load(f)
    return configs


def dicts_to_df(dict_list):
    """Concatenates a list of dictionaries with similar keys into a dataframe"""
    all_keys = set()
    for r in dict_list:
        all_keys.update(r.keys())
    data = {}
    for key in all_keys:
        data[key] = []

    for d in dict_list:
        for key in all_keys:
            if key in d:
                data[key].append(d[key])
            else:
                data[key].append(None)
    df = pd.DataFrame(data=data)
    return df


def save_results_to_yaml(
    log_dir,
    train_size,
    test_size,
    dataset_files,
    training_time=None,
    epsilon=None,
    delta=None,
    mse=None,
    rmse=None,
    rmsle=None,
    accuracy=None,
    loss=None,
):

    dict_results = {
        "train_size": train_size,
        "test_size": test_size,
        "dataset_files": dataset_files,
        "training_time": training_time,
        "epsilon": epsilon,
        "delta": delta,
        "mse": mse,
        "rmse": rmse,
        "rmsle": rmsle,
        "accuracy": accuracy,
        "timestamp": time.time(),
    }

    yaml_path = os.path.join(log_dir, "results.yaml")

    with open(yaml_path, "w") as f:
        yaml.dump(dict_results, f)

    print(f"Saved logs to {yaml_path}.")


def yaml_dir_to_df(dir):
    all_experiments = None
    print(f"Loading {dir}")

    if dir[0:5] == "gs://":
        fs = gcsfs.GCSFileSystem(project=GCP_PROJECT)
        files = list(map(lambda blob: blob["name"], fs.listdir(dir)))
    else:
        files = os.listdir(dir)

    for index, yaml_file in enumerate(filter(lambda f: f.endswith(".yaml"), files)):
        if dir[0:5] == "gs://":
            with fs.open(yaml_file) as f:
                config = yaml.load(f, Loader=yaml.Loader)
        else:
            with open(os.path.join(dir, yaml_file)) as f:
                config = yaml.load(f, Loader=yaml.Loader)

        # Pop nested lists that don't fit in the DF
        to_pop = []
        for key, value in config.items():
            if isinstance(value, list):
                to_pop.append(key)
        for key in to_pop:
            config.pop(key, None)

        # Transform to dataframe
        experiment = pd.DataFrame(config, index=[index])
        # workload = pd.DataFrame(config["workload"], index=[index])
        # results = pd.DataFrame(config["results"], index=[index])
        # experiment = dp.join(workload).join(results)

        # Update
        if all_experiments is None:
            all_experiments = experiment
        else:
            all_experiments = all_experiments.append(experiment)

    return all_experiments


def yaml_dir_to_csv(dir, path):
    df = yaml_dir_to_df(dir)
    df.to_csv(path)
    return


def multiclass_accuracy(outputs, labels):
    _, predicted = torch.max(outputs.data, 1)
    total = labels.size(0)
    correct = (predicted == labels).sum().float()
    return correct / total


def binary_accuracy(preds, y):
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == y).float()
    acc = correct.sum() / len(correct)
    return acc


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs
