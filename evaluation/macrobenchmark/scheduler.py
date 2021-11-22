import typer
from loguru import logger
from pathlib import Path
from tqdm import tqdm
from privatekube.experiments.utils import save_yaml, load_yaml, yaml_dir_to_df
import os
import itertools
import subprocess
import pandas as pd
import datetime
import time
import numpy as np

from ray.util.multiprocessing import Pool

from utils import plot_workload_run, load_block_claims, plot_schedulers_run
from utils import DEFAULT_LOG_PATH, DEFAULT_GO_EXEC

app = typer.Typer()


@app.command(
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True}
)
def schedulers(
    ctx: typer.Context,
    config_path="",
    output_dir="",
    keep_intermediary=True,
    keep_raw_logs=False,
    analyze=True,
    max_trials: int = 5,
    n_processes: int = 1,
    worker_delay: int = 0,
):
    configs = load_yaml(config_path)
    if isinstance(configs, list):
        all_configs = configs
    if isinstance(configs, dict):
        for (key, value) in configs.items():
            if isinstance(value, list):
                configs[key] = value
            else:
                configs[key] = [value]
        all_configs = [
            dict(zip(configs, x)) for x in itertools.product(*configs.values())
        ]

    experiment_uid = (
        os.path.basename(config_path)[0 : -len(".yaml")]
        + "-"
        + datetime.datetime.now().strftime("%m%d-%H%M%S")
    )

    if output_dir == "":
        output_dir = DEFAULT_LOG_PATH.joinpath("schedulers").joinpath(experiment_uid)

    # Store useful metrics for each configuration
    metrics = []

    # Run all the configurations
    if n_processes <= 1:
        for i, config in enumerate(tqdm(all_configs)):
            config_metrics = run_config(
                i,
                config,
                output_dir,
                ctx,
                max_trials,
                analyze,
                keep_intermediary,
                keep_raw_logs,
            )
            metrics.append(config_metrics)
    else:
        logger.info(f"Starting the multiprocessing pool with {n_processes} workers.")
        pool = Pool(processes=n_processes)

        def subrun(c):
            i, config = c[0], c[1]
            # Spread the execution time a little bit
            t = worker_delay * (os.getpid() % n_processes)
            logger.info(f"Sleep {t} seconds before running config {i+1}.")
            time.sleep(t)
            config_metrics = run_config(
                i,
                config,
                output_dir,
                ctx,
                max_trials,
                analyze,
                keep_intermediary,
                keep_raw_logs,
            )
            return config_metrics

        all_config_metrics = pool.map(subrun, list(enumerate(all_configs)))

        pool.close()
        pool.join()
        logger.info("Closed the pool. Gathering the results...")

        for config_metrics in all_config_metrics:
            metrics.append(config_metrics)

    # Compare the different workloads (for some simple cases)
    if analyze:
        try:
            data = {}
            for key in metrics[0].keys():
                data[key] = []
            for metric in metrics:
                for k, v in metric.items():
                    data[k].append(v)
            metrics_df = pd.DataFrame(data=data)
            metrics_df.to_csv(output_dir.joinpath("metrics.csv"))
            plot_schedulers_run(metrics_df, output_dir)
        except Exception as e:
            logger.error(e)
            save_yaml(output_dir.joinpath("metrics.yaml"), metrics)


def run_config(
    i, config, output_dir, ctx, max_trials, analyze, keep_intermediary, keep_raw_logs
):
    config_output = output_dir.joinpath(str(i))

    config_output.mkdir(parents=True, exist_ok=True)

    logger.info(ctx.args)
    args = []
    speed = 0
    for flag, value in config.items():
        args.append(f"--{flag}")
        args.append(f"{value}")
        if flag == "t":
            speed = value
    ctx.args = args

    logger.info(ctx.args)

    # Retry each configuration a few times if it fails, with reduced speed at each trial
    trial_number = 0
    success = False
    while not success and trial_number < max_trials:
        trial_number += 1
        try:
            workload(
                ctx,
                name=f"scheduler-{i}",
                output_dir=config_output,
                timeout=24 * 60 * 60,
                plot=True,
            )
            if not keep_intermediary:
                os.rmdir(config_output)

            success = True
        except Exception as e:
            logger.error(
                f"Error for config #{i} {config}: {e}.\n Will retry at slower speed."
            )
            speed = 2 * speed
            ctx.args.append("--t")
            ctx.args.append(f"{speed}")
    if not success:
        logger.error(f"Giving up on config #{i} {config}")

    # Plots the workload and add the metrics to our list
    if success and analyze:
        logger.info("Analyzing the config.")
        (blocks_df, claims_df) = load_block_claims(
            config_output.joinpath("claims.json"),
            config_output.joinpath("blocks.json"),
        )
        plot_workload_run(blocks_df, claims_df, config_output)
        config["n_pipelines"] = len(claims_df)
        config["n_mice"] = len(claims_df.query("mice == True"))
        config["n_elephants"] = len(claims_df.query("mice == False"))

        config["n_allocated_pipelines"] = sum(claims_df["success"])
        config["n_allocated_mice"] = sum(claims_df.query("mice == True")["success"])
        config["n_allocated_elephants"] = sum(
            claims_df.query("mice == False")["success"]
        )
        config["fraction_allocated_pipelines"] = config["n_allocated_pipelines"] / len(
            claims_df
        )
        total_profit = 0
        for _, claim in claims_df.iterrows():

            if claim["success"] == True:
                total_profit += claim["priority"]
        config["realized_profit"] = total_profit
        mice_path = config["mice"]
        if "/user-time/" in mice_path:
            semantic = "user-time"
        elif "/user/" in mice_path:
            semantic = "user"
        elif "/event/" in mice_path:
            semantic = "event"
        config["semantic"] = semantic
        # metrics.append(config)
        save_yaml(config_output.joinpath("metrics.yaml"), config)

    if not keep_raw_logs:
        try:
            os.remove(config_output.joinpath("out.log"))
            os.remove(config_output.joinpath("blocks.json"))
            os.remove(config_output.joinpath("claims.json"))
        except Exception as e:
            logger.error(e)

    return config


@app.command(
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True}
)
def workload(
    ctx: typer.Context,
    name="scheduling",
    output_dir="",
    timeout=360,
    plot=True,
):
    logger.info(ctx.args)

    experiment_uid = name + "-" + datetime.datetime.now().strftime("%m%d-%H%M%S")
    if output_dir == "":
        output_dir = DEFAULT_LOG_PATH.joinpath("workload").joinpath(experiment_uid)
        output_dir.mkdir(parents=True, exist_ok=True)

    if not DEFAULT_GO_EXEC.exists():
        logger.error("Please provide a valid path to the scheduler stub.")
        return
    log_file = output_dir.joinpath("out.log")
    log_blocks = output_dir.joinpath("blocks.json")
    log_claims = output_dir.joinpath("claims.json")
    with open(log_file, "w") as f:
        try:
            command = [
                str(DEFAULT_GO_EXEC),
                "--log_blocks",
                str(log_blocks),
                "--log_claims",
                str(log_claims),
            ]
            for extra_arg in ctx.args:
                command.append(extra_arg)
            result = subprocess.run(
                command,
                stdout=f,
                stderr=f,
                text=True,
                timeout=timeout,
                check=True,
            )
            logger.info(result)
            if plot:
                blocks_df, claims_df = load_block_claims(log_claims, log_blocks)
                plot_workload_run(blocks_df, claims_df, output_dir)

            d = {ctx.args[i][2:]: ctx.args[i + 1] for i in range(0, len(ctx.args), 2)}
            save_yaml(output_dir.joinpath("config.yaml"), d)
        except Exception as e:
            logger.info(f"This configuration failed. \n {e}")
            raise e


if __name__ == "__main__":
    app()