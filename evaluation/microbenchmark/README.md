## How to reproduce microbenchmark in PrivateKute paper
Before getting started, make sure to install dpsched package under ./simulator.

### Reproduce microbenchmark experiments
./microbench_single_block.py contains the script to reproduce single static block workload experiment with various DP scheduling policies, for both epsilon-delta DP composition and Renyi DP composition. 

./microbench_multi_block.py is the dynamic multiple block counterpart of microbench_single_block.py

One can run them by (caveat: multiple block script may take several hours to finish.)

```bash
cd ./evaluation/microbenchmark
pypy ./microbench_single_block.py
pypy ./microbench_multi_block.py
```

The experiment results are saved under "./evaluation/microbenchmark/exp_results/some_work_space_name" 

### Reproduce microbenchmark figures

microbenchmark_figures_single_block.ipynb and microbenchmark_figures_multiple_block.ipynb show how to reproduce microbenchmark figures. You can read a notebook directly via [nbviwer](https://nbviewer.jupyter.org/) without executing it. Note that these two notebooks download existing microbenchmark results and plot them directly. Therefore, we don't need to rerun microbenchmark experiments.
