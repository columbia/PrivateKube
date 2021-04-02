import typer
from loguru import logger
from pathlib import Path
from tqdm import tqdm
from privatekube.experiments.utils import save_yaml, load_yaml
from privatekube.experiments import run

SEMANTICS = ["event", "user-time", "user"]
DEFAULT_DATA_PATH = Path(__file__).resolve().parent.joinpath("data")
DEFAULT_LOG_PATH = Path(__file__).resolve().parent.joinpath("logs")
DEFAULT_LOG_PATH.mkdir(parents=True, exist_ok=True)
DEFAULT_PYTHON_PATH = (
    Path(__file__).resolve().parent.joinpath("workload").joinpath("models")
)
DEFAULT_PARAMS_PATH = (
    Path(__file__).resolve().parent.joinpath("workload").joinpath("params")
)

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
            # The parameters for the statistics are already set, except the paths
            run_param = generate_paths(d)
            run_path = run_params.joinpath("statistics.yaml")
            save_yaml(run_path, run_param)
        else:
            p = generate_experiment_params(d)
            for s in SEMANTICS:
                run_param = p[s]
                run_path = run_params.joinpath(model_param.stem + "-" + s + ".yaml")
                save_yaml(run_path, run_param)


@app.command()
def all():
    """Run all the experiments."""
    logger.warning("Running all the models. It can take a long time.")
    run_params = DEFAULT_PARAMS_PATH.joinpath("runs")
    run.main(run_params)


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