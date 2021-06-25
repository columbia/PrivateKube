# Evaluation

We support two types of evaluation for PrivateKube and DPF (see [OSDI paper]() for methodologies): 
- `microbenchmark`: uses a simulator of the privacy scheduling algorithm (such as DPF) under a highly controlled workload;
- `macrobenchmark`: uses the real system under a DP ML workload we developed over the Amazon Reviews dataset.

This folder provides code and instructions for how to perform/reproduce both types of evaluation.
Instructions of how to reproduce the macrobenchmark results are in ../README.md.
Instructions for how to reproduce the microbenchmark results are in ./microbenchmark/README.md.
