import docker
from pathlib import Path
from datetime import datetime
import subprocess
from privatekube.experiments.utils import load_yaml

import typer
from typing import Optional
from loguru import logger


def build_dir(client, repo, directory):
    for subdir in filter(lambda x: x.is_dir(), directory.iterdir()):
        build(client, repo, subdir)


def build(client, repo, subdir):
    name = subdir.name
    tag = datetime.now().strftime("%m-%d-%H-%M-%S")
    image = f"{repo}/{name}:{tag}"

    logger.info(f"Building: {image}")

    im = client.images.build(
        path=str(subdir.absolute()),
        tag=f"{repo}/{name}:{tag}",
    )

    for l in im[1]:
        logger.info(l)

    res = client.api.push(f"{repo}/{name}", tag=tag)
    logger.info(res)

    subprocess.call(
        f"sed 's=DOCKER_IMAGE_PLACEHOLDER='\"{image}\"'=g' {subdir.joinpath('src/component.yaml').absolute()} > {subdir.joinpath('component.yaml').absolute()}",
        shell=True,
    )
    logger.info("Done.")


def main(
    dir: Optional[str] = typer.Argument("", help="Base directory for the components"),
    single: Optional[bool] = typer.Option(
        False,
        help="Build a single component instead of all the components in the directory.",
    )
    # vector_cache: Optional[bool] = typer.Argument(False, help="Copy the vector cache"),
):
    secrets = load_yaml(Path(__file__).parent.joinpath("client_secrets.yaml"))
    repo = secrets["repo"]
    client = docker.from_env()

    if not single:
        if dir == "":
            directory = Path(__file__).parent.absolute()
        else:
            directory = Path(dir).absolute()
        logger.info(f"Building whole directory: {directory}")

        build_dir(client, repo, directory)
    else:
        if dir == "":
            raise Exception("Please provide a component directory")
        else:
            directory = Path(dir).absolute()
            logger.info(f"Building single component directory: {directory}")

            build(client, repo, directory)


if __name__ == "__main__":
    typer.run(main)