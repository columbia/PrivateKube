## Privacy Budget Scheduling Simulator

Our microbenchmark evaluation of the DPF privacy budget scheduling algorithm uses a simulator, whose code is available here.  This simulator supports controlled evaluation of several scheduling algorithms we develop for differential privacy budgets, including: round-robin, first-come-first-serve, and DPF algorithms. For more information about the algorithms and our evaluation, please refer to our [PrivateKube paper](https://columbia.github.io/PrivateKube/papers/osdi2021privatekube.pdf).

## Setup

(Note: These instructions assume that the initial current directory is the root of the PrivateKube directory; paths are relative to that.)

### Setup python environment
Install conda, create and activate an isolated python environment "ae". 
```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh
bash ~/miniconda.sh -b -p $HOME/miniconda
eval "$($HOME/miniconda/bin/conda shell.bash hook)"
conda init
conda create -n ae  -c conda-forge pypy3.6 pip python=3.6 seaborn notebook -y
conda activate ae
```

### Installation from source
Install a Python package called dpsched via
 
```bash
cd ./simulator
pip install -r ./requirements.txt
pip install .[plot]
```


## Examples
## The minimal example
`./examples/simulator/minimal_example.py` gives a quick start. There are two key concepts in the simulation program:
1. The simulation model: This implements how different components in the systems behave and interact with each other. One can import it via `from dpsched import Top`
2. The configuration dictionary: a dictionary that specifies many aspects of simulation behavior. for configuration details, please refer to the comments in minimal_example.py

 Basically, there are two steps in ./examples/simulator/minimal_example.py.
 1. Preparing the config dictionary
 2. Calling `simulate(config, Top)`, where `config` is the config dict and `Top` is the simulation model.

To run the minimal example.
```bash
cd ./examples/simulator
python ./minimal_example.py
``` 
or, replace CPython with PyPy for better performance:
```bash
cd ./examples/simulator
pypy ./minimal_example.py
```

The simulation program saves experiment results in a workspace specified by config dictionary. By default, it is saved under `./examples/exp_result/some_work_space_name`.

### How to analyze simulation results
`dpsched.analysis` contains modules for collecting experiment result from workspace directory and plotting various figures.
`evaluation/microbenchmark/microbenchmark_figures_single_block.ipynb` gives examples on how to use `dpsched.analysis` module with detailed comments. 

## How to reproduce microbenchmark in PrivateKube paper

Instructions and code for how to use the simulator to reproduce the microbenchmark results in the PrivateKube paper are in [`evaluation/microbenchmark/README.md`](../evaluation/microbenchmark/README.md).

