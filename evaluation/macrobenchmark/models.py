from evaluation.macrobenchmark.utils import get_name
import typer
from loguru import logger
from pathlib import Path
from tqdm import tqdm
from privatekube.experiments.utils import flags_to_dict, save_yaml, load_yaml
from privatekube.experiments import run

from utils import (
    round_smart,
    load_exps,
    build_figure_nn,
    build_gnuplot_df,
    plot_workload_run,
    copy_and_rename,
)
from utils import (
    SEMANTICS,
    DEFAULT_DATA_PATH,
    DEFAULT_LOG_PATH,
    DEFAULT_PYTHON_PATH,
    DEFAULT_PARAMS_PATH,
    DEFAULT_EXPERIMENTS_RESULTS,
    DEFAULT_IMAGE_PATH,
    DEFAULT_SELECTED_RUNS_PATH,
)

STAT_MODELS = [
    "count_reviews",
    "count_per_category",
    "avg_rating",
    "avg_tokens",
    "count_tokens",
    "std_tokens",
]
ACCURACY_TARGETS = {
    "product-linear": 0.6,
    "product-ff": 0.6,
    "product-lstm": 0.6,
    "product-bert": 0.7,
    "sentiment-linear": 0.7,
    "sentiment-ff": 0.7,
    "sentiment-lstm": 0.75,
    "sentiment-bert": 0.8,
}
STATS_TARGET = 0.05  # Relative error


app = typer.Typer()


@app.command()
def generate():
    """Generates the experiment configurations with hyperparameters and local paths."""
    raw_params = DEFAULT_PARAMS_PATH.joinpath("raw")
    run_params = DEFAULT_PARAMS_PATH.joinpath("runs")
    run_params.mkdir(parents=True, exist_ok=True)
    logger.info("Preparing the configurations.")
    for model_param in tqdm(raw_params.glob("*.yaml")):
        d = load_yaml(model_param)
        if d["python_path"] == "statistics.py":
            for s in list(SEMANTICS.keys()) + ["non-dp"]:
                for model in STAT_MODELS:
                    for mechanism in (
                        ["laplace"] if s == "non-dp" else ["laplace", "gaussian"]
                    ):
                        run_param = generate_paths(d)
                        run_param["model"] = model
                        run_param["dp"] = s
                        run_param["mechanism"] = mechanism
                        if mechanism == "laplace":
                            run_param["delta"] = 0.0
                        run_path = run_params.joinpath(
                            run_param["model"] + "-" + s + "-" + mechanism + ".yaml"
                        )
                        save_yaml(run_path, run_param)
        else:
            p = generate_experiment_params(d)
            for s in SEMANTICS:
                run_param = p[s]
                run_path = run_params.joinpath(model_param.stem + "-" + s + ".yaml")
                save_yaml(run_path, run_param)


@app.command()
def run_all(
    run_dir: str = typer.Option("", help="Parameters directory"),
):
    """Run all the experiments."""
    logger.warning("Running all the models. It can take a long time.")
    run_params = Path(run_dir) if run_dir else DEFAULT_PARAMS_PATH.joinpath("runs")
    run.main(run_params)


@app.command()
def analyze(
    logs_dir: str = typer.Option("", help="Logs directory to read from."),
    images_dir: str = typer.Option(
        "", help="Path to store graphs for individual models"
    ),
):
    logs_dir = Path(logs_dir) if logs_dir else DEFAULT_EXPERIMENTS_RESULTS
    images_dir = Path(images_dir) if images_dir else DEFAULT_IMAGE_PATH
    images_dir.mkdir(parents=True, exist_ok=True)

    all_dirs = []
    for exp_dir in logs_dir.glob("*/"):
        if exp_dir.is_dir():
            all_dirs.append(str(exp_dir))
    df = load_exps(all_dirs, base_path=None)
    df["epsilon"] = (
        df["epsilon"]
        .fillna(-1)
        .apply(round_smart)
        .apply(lambda e: 5.0 if e == 4.9 else e)
    )
    for task in ["product", "sentiment"]:
        for model in ["bow", "feedforward", "lstm", "bert"]:
            d = df.query(f"model == '{model}' and task == '{task}'")
            non_private = d.query("dp == 0")
            for semantic, semantic_query in SEMANTICS.items():
                try:
                    fig = build_figure_nn(
                        d.query(semantic_query), non_private, semantic
                    )
                    fig.write_image(
                        str(images_dir.joinpath(f"{task}-{model}-{semantic}.png"))
                    )
                    gnuplot_df = build_gnuplot_df(
                        d.query(semantic_query), non_private, semantic
                    )
                    gnuplot_df.to_csv(
                        images_dir.joinpath(f"{task}-{model}-{semantic}.csv"),
                        index=False,
                    )
                except Exception as e:
                    logger.error(f"Error for {model} {task} {semantic}")
                    raise e


@app.command()
def select(
    logs_dir: str = typer.Option("", help="Logs directory to read from."),
    selected_runs_dir: str = typer.Option("", help="Path to store the selected runs"),
):
    logs_dir = Path(logs_dir) if logs_dir else DEFAULT_LOG_PATH
    selected_runs_dir = (
        Path(selected_runs_dir) if selected_runs_dir else DEFAULT_SELECTED_RUNS_PATH
    )

    for semantic in SEMANTICS.keys():
        logger.info(f"Semantic: {semantic}")

        # logger.info("Selecting neural networks...")
        # select_runs(
        #     logs_dir.joinpath("amazon").joinpath("classification"),
        #     semantic,
        #     ACCURACY_TARGETS,
        #     selected_runs_dir.joinpath(semantic).joinpath("elephants"),
        # )
        logger.info("Selecting stats...")
        stats_target = {}
        for model in STAT_MODELS:
            stats_target[model] = STATS_TARGET
        for mechanism in ["laplace", "gaussian"]:
            select_runs(
                logs_dir.joinpath("amazon").joinpath("statistics"),
                f"{semantic}-{mechanism}",
                stats_target,
                selected_runs_dir.joinpath(semantic).joinpath(f"mice-{mechanism}"),
            )
    return


def select_runs(logs_dir, semantic, accuracy_targets, destination_dir):
    destination_dir.mkdir(parents=True, exist_ok=True)

    for model, accuracy_target in accuracy_targets.items():
        logger.info(model)
        logger.info(logs_dir)
        # Browsing the runs to find the cheapest one that reaches the target
        min_blocks = {}
        selected = {}
        for run_path in logs_dir.glob(f"**/{model}-{semantic}-[0-9]*/*.yaml"):
            run = load_yaml(run_path)
            if not (run["epsilon"] is None):
                if ("accuracy" in run and run["accuracy"] >= accuracy_target) or (
                    "relative_error" in run and run["relative_error"] < accuracy_target
                ):
                    # Group runs by budget
                    name = get_name(run)
                    size = run["n_blocks"]
                    if name in min_blocks:
                        if size < min_blocks[name]:
                            min_blocks[name] = size
                            selected[name] = run_path
                    else:
                        min_blocks[name] = size
                        selected[name] = run_path
        # Copy the selected runs
        for file_path in selected.values():
            copy_and_rename(file_path, destination_dir)


def generate_paths(params):
    # The other default paths are handled by the training script.
    exp = {}
    exp.update(params)
    exp["dataset_dir"] = ""
    exp["log_dir"] = str(DEFAULT_LOG_PATH)
    # The raw configurations only provide the filename
    exp["python_path"] = str(DEFAULT_PYTHON_PATH.joinpath(exp["python_path"]))
    return exp


def generate_experiment_params(params: dict):
    res = {}
    for semantic in SEMANTICS:
        exp = generate_paths(params)
        exp["delta"] = 1e-9
        exp["dp_eval"] = 0
        exp["dp"] = 1
        exp["epsilon"] = [0.5, 1.0, 5.0]
        exp["n_blocks"] = (
            [100, 200, 500, 1_000, 2_000, 5_000, 10_000, 50_000]
            if params["model"] != "bert"
            else [100, 200, 500, 1_000, 2_000, 5_000]
        )
        if semantic == "event":
            exp["epsilon"].append(-1.0)
            exp["user_level"] = 0
            exp["n_epochs"] = 15 if params["model"] != "bert" else 3
        if semantic == "user-time":
            exp["user_level"] = 1
            exp["timeframe_days"] = 1
            exp["n_epochs"] = 15 if params["model"] != "bert" else 3
        if semantic == "user":
            exp["user_level"] = 1
            exp["timeframe_days"] = 0
            exp["n_epochs"] = 60 if params["model"] != "bert" else 5
        res[semantic] = exp.copy()
    return res


if __name__ == "__main__":
    app()