## Introduce Privacy Budget Scheduling Simulator

This simulator studies various scheduling algorithms given differential privacy budget constrains. It includes round-robin, first-come-first-serve, DPF algorithms. For more background about this problem, please refer to our PrivateKube paper.

In addition to differential privacy, the simulator also considers other computational resources, such as CPU and memory. 


## Setup

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
./examples/simulator/minimal_example.py gives a quick start. There are two key concepts in the simulation program:
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

The simulation program saves experiment results in a workspace specified by config dictionary. By default, it is saved under "./examples/exp_result/some_work_space_name"

### How to analyze simulation results
`dpsched.analysis` contains modules for collecting experiment result from workspace directory and plotting various figures.
./examples/microbenchmark_figures_single_block.ipynb gives examples on how to use `dpsched.analysis` module with detailed comments. 

## How to reproduce microbenchmark in PrivateKute paper
### Reproduce microbenchmark experiments
./examples/microbench_single_block.py contains the script to reproduce single static block workload experiment with various DP scheduling policies, for both epsilon-delta DP composition and Renyi DP composition. The major difference between microbench_single_block.py and minimal_example.py is  minimal_example.py only runs single simulation while microbench_single_block.py runs multiple simulations in parallel. 

./examples/microbench_multi_block.py is the dynamic multiple block counterpart of microbench_single_block.py

One can run them by (caveat: they may take hours to finish especially for multiple block script)
```bash
cd ./examples
pypy ./microbench_single_block.py
pypy ./microbench_multi_block.py
```

Similar to minimal_example.py, the experiment results are saved under "./examples/exp_result/some_work_space_name" 

### Reproduce microbenchmark figures

./examples/microbenchmark_figures_single_block.ipynb  and ./examples/microbenchmark_figures_multiple_block.ipynb show how to reproduce microbenchmark figures. Note that these two notebook download existing microbenchmark results and plot them directly. Therefore, we don't need to rerun microbenchmark experiments. I recommend viewing these notebooks directly via (nbviwer)[https://nbviewer.jupyter.org/]

