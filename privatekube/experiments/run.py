import subprocess
import yaml
import os
import sys
import shutil
import datetime
from attrdict import AttrDict
import itertools
import atexit
from .utils import save_yaml, load_yaml
import argparse
import kfp
from ray.util.multiprocessing import Pool
import ray.util.multiprocessing

ALL_CONFIGS = []
LOG_DIR = ""
FAILED_CONFIGS = []
SUCCESSFUL_CONFIGS = []


def save_configs():
    print(f"Saving configurations to {LOG_DIR} before exit.")
    save_yaml(os.path.join(LOG_DIR, "failed_configs.config"), FAILED_CONFIGS)
    save_yaml(os.path.join(LOG_DIR, "successful_configs.config"), SUCCESSFUL_CONFIGS)
    save_yaml(os.path.join(LOG_DIR, "all_configs.config"), ALL_CONFIGS)


atexit.register(save_configs)


def run_experiment(
    yaml_path,
    timeout=240_000,
    copy_script=False,
    slurm=False,
    n_gpus=1,
    cuda_id=0,
    kubeflow=False,
):
    """
    Run either on a list of configurations or on a dict of possible parameters.
    """

    global ALL_CONFIGS
    global LOG_DIR
    global FAILED_CONFIGS
    global SUCCESSFUL_CONFIGS

    # Parse Yaml to dict
    configs = load_yaml(yaml_path)
    print(configs)

    if isinstance(configs, list):
        ALL_CONFIGS = configs
    if isinstance(configs, dict):

        for (key, value) in configs.items():
            if isinstance(value, list):
                configs[key] = value
            else:
                configs[key] = [value]

        # One script and log directory
        assert len(configs["log_dir"]) == 1
        assert len(configs["python_path"]) == 1
        assert len(configs["dataset"]) == 1

        # Compute the nested cartesian product
        ALL_CONFIGS = [
            dict(zip(configs, x)) for x in itertools.product(*configs.values())
        ]

    experiment_uid = (
        os.path.basename(yaml_path)[0 : -len(".yaml")]
        + "-"
        + datetime.datetime.now().strftime("%m%d-%H%M%S")
    )

    config = AttrDict(ALL_CONFIGS[0])
    LOG_DIR = os.path.join(
        config.log_dir,
        config.dataset,
        os.path.basename(config.python_path)[0 : -len(".py")],
        experiment_uid,
    )
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)

    # Save a copy of the configs and the script
    shutil.copy(
        yaml_path,
        os.path.join(LOG_DIR, os.path.basename(yaml_path))[0 : -len(".yaml")]
        + ".config",
    )

    python_path_copy = os.path.join(LOG_DIR, os.path.basename(config.python_path))
    shutil.copy(config.python_path, python_path_copy)

    # When the script is calling local modules we can't really run on the copy
    if copy_script:
        python_path = python_path_copy
    else:
        python_path = config.python_path

    def run_config(i, config, cuda_id=cuda_id):
        print(
            f"\n\n{datetime.datetime.now()}\nRunning configuration {i+1}/{len(ALL_CONFIGS)}.\n {config}\n Cuda id: {cuda_id}"
        )

        # config = AttrDict(config)

        # Load the model and prepare the logs
        # python_path = config.python_path
        log_path = os.path.join(
            LOG_DIR,
            str(i) + ".yaml",
        )

        # Generate temporary Abseil flagfile
        flagfile_path = os.path.join(LOG_DIR, "flags.txt")
        with open(flagfile_path, "w") as f:
            for (flag, value) in config.items():
                if flag == "device" and cuda_id > 0:
                    f.write(f"--device=cuda:{cuda_id}\n")
                else:
                    f.write(f"--{flag}={value}\n")
            f.write(f"--log_path={log_path}\n")
            # No need to define these flags inside the model
            f.write(f"--undefok=dataset,python_path\n")

        # Run the model on the given configuration
        shell_path = log_path[0 : -len(".yaml")] + ".log"
        with open(shell_path, "w") as f:
            if kubeflow:
                pipeline_path = config["pipeline_path"]
                config.pop("log_dir", None)
                config.pop("pipeline_path", None)
                config["gcs_prefix"] = os.path.join(LOG_DIR, f"{i}.yaml")

                # Submit a pipeline run
                result = kfp.Client().create_run_from_pipeline_package(
                    pipeline_path,
                    run_name=str(experiment_uid) + f" {i}",
                    experiment_name="",
                    arguments=config,
                )

                print("Kubeflow run id:")
                print(result)
            else:
                try:
                    command = ["python", python_path, "--flagfile", flagfile_path]
                    if slurm:
                        command = ["srun"] + command
                    result = subprocess.run(
                        command,
                        stdout=f,
                        stderr=f,
                        text=True,
                        timeout=timeout,
                        check=True,
                    )
                    print(result)
                    SUCCESSFUL_CONFIGS.append(config)
                except Exception as e:
                    print(f"This configuration failed. \n {e}")
                    FAILED_CONFIGS.append(config)

    if n_gpus <= 1:

        for i, config in enumerate(ALL_CONFIGS):
            run_config(i, config)

    else:
        with Pool(processes=n_gpus) as pool:
            pids_list = pool.map(lambda i: (i, os.getpid()), range(n_gpus))
            pids = {}
            for (i, pid) in pids_list:
                pids[pid] = i
            print(pids)

            def run_config_process(args):
                i = args[0]
                config = args[1]
                cuda_id = pids[os.getpid()]
                run_config(i, config, cuda_id)

            pool.map(run_config_process, list(enumerate(ALL_CONFIGS)))


def is_yaml(filename):
    if filename[-len(".yaml") :] == ".yaml" or filename[-len(".yml") :] == ".yml":
        return True
    return False


def main(configs, slurm=False, n_gpus=1, cuda_id=0):
    yaml_paths = []
    if os.path.isdir(configs):
        for config_file in os.listdir(configs):
            if is_yaml(config_file):
                yaml_paths.append(os.path.join(configs, config_file))
    elif is_yaml(configs):
        yaml_paths.append(configs)
    print(f"Running rex on the following yaml files:\n {yaml_paths}")
    for yaml_path in yaml_paths:
        run_experiment(
            os.path.realpath(yaml_path), slurm=slurm, n_gpus=n_gpus, cuda_id=cuda_id
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "configs",
        type=str,
        help="Path to a yaml config file or to a directory of config files",
    )
    parser.add_argument("--slurm", dest="slurm", action="store_true")
    parser.add_argument(
        "--n_gpus",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--cuda_id",
        type=int,
        default=0,
    )
    args = parser.parse_args()
    main(args.configs, args.slurm, args.n_gpus, args.cuda_id)
