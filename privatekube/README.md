# PrivateKube Python Package

This package contains useful functions to interact with the system. It is used by the evaluation and by the system itself.


- `privatekube/experiments/`: functions to run pipelines, including dataset objects that support different  differential privacy semantics (Event, User-Time, User)
- `privatekube/kfp/`: functions to interact with Kubeflow Pipelines and the privacy resource
- `privatekube/privacy/`: functions related to differential privacy
    
## Installation

Create a new virtual environment to interact with PrivateKube, for instance with:

```bash
conda create -n privatekube python=3.8

conda activate privatekube
```

Install the dependencies:
```bash
pip install -r privatekube/requirements.txt
```

Install the PrivateKube package:
```bash
pip install -e privatekube
```