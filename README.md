# PrivateKube

**Status.** This is a partial release of PrivateKube.

We provide:
- The privacy resource implementation
- The DPF scheduler
- The DP workloads (dataset, models and parameters) used for the macrobenchmark

We do not provide (yet):
- The Kubeflow Pipeline interface (currently tied to our Google Cloud infrastructure)
- The microbenchmark
- The end-to-end macrobenchmark
- The Grafana dashboard

## 1. Getting started

This section explains how to install the system and walks through a simple example of interaction with the privacy resource. It should take less than 30 mins to complete.

### 1.1 Requirements

PrivateKube needs a Kubernetes cluster to run. If you don't have a cluster, you can install a lightweight Microk8s cluster on a decent laptop. Kubeflow requires more resources but it is not required in this section.

Below are the instructions to install and configure a lightweight cluster on Ubuntu. For other platforms, see [https://microk8s.io/](https://microk8s.io/).

```bash
sudo snap install microk8s --classic
```


Check that it is running:
```bash
microk8s status --wait-ready
```

You can add your user to the `microk8s` group if you don't want to type `sudo` for every command (you should log out and log in again after this command):
```
sudo usermod -a -G microk8s $USER

mkdir ~/.kube

sudo chown -f -R $USER ~/.kube
```

You can now start and stop your cluster with:
```bash
microk8s start 

microk8s stop
```


Export your configuration:
```bash
microk8s config > ~/.kube/config
```

Declare an alias to use `kubectl` (you can add this line to your `.bash_profile` or equivalent):
```bash
alias kubectl=microk8s.kubectl
```

Check that you can control your cluster:
```bash
kubectl get pods -A
```


### 1.2. Deploying PrivateKube

#### Download the code

Clone this repository on your machine. Our scripts will only affect this repository (e.g. dataset, logs, etc.) and your cluster, not the rest of your machine.

```bash
git clone https://github.com/columbia/PrivateKube.git
```

Enter the repository:
```bash
cd PrivateKube
```

All the other instructions in this file have to be run from this `PrivateKube` directory, unless specified otherwise. 


#### Create a Python environment

Create a new virtual environment to interact with PrivateKube, for instance with:

```bash
conda create -n privatekube python=3.8

conda activate privatekube
```

Install the dependencies:
```bash
pip install -r requirements.txt
```

Install the PrivateKube package:
```bash
pip install -e .
```

#### Deploy PrivateKube to your cluster

You can deploy PrivateKube in one line and directly by running:

```bash
source system/deploy.sh
```

If you prefer to understand what is going on, you can run the following commands one by one:

First, let's create a clean namespace to separate PrivateKube from the rest of the cluster:

```bash
kubectl create ns dp-sage
```

Then, create the custom resources:

```bash
kubectl apply -f system/privacyresource/artifacts/privacy-budget-claim.yaml

kubectl apply -f system/privacyresource/artifacts/private-data-block.yaml
```

You can now interact with the privacy resource like with any other resource (e.g. pods). `pb` is a short name for private data block, and `pbc` stands for privacy claim. You can list blocks and see how much budget they have with: `kubectl get pb -A`. So far, there are no blocks nor claims, but in the next section (1.3.) we will add some.

We already compiled the controllers and the scheduler and prepared a Kubernetes deployment that will pull them from DockerHub. Launch the privacy controllers and the scheduler:

```bash
kubectl apply -f system/dpfscheduler/manifests/cluster-role-binding.yaml

kubectl apply -f  system/dpfscheduler/manifests/scheduler.yaml
```


### 1.3. Hello World

Open a first terminal. We are going to monitor the logs of the scheduler to see it in action. Find the scheduler pod with:

```bash
kubectl get pods -A | grep scheduler
```

Then, in the same terminal, monitor the logs of the scheduler with something similar to:
```bash
kubectl logs --follow dpf-scheduler-5fb6886497-w7x49 -n dp-sage 
```


Open another terminal. We are going to create a block and a claim and see how they are being scheduled.

Create a new namespace for this example:

```bash
kubectl create ns privacy-example
```

Check that there are no datablocks or claims:

```bash
kubectl get pb -A
```

Add a first datablock:

```bash
kubectl apply -f evaluation/examples/dpf-base/add-block.yaml
```

List the datablocks to see if you can see your new block:

```bash
kubectl get pb --namespace=privacy-example
```



Check the initial budget of your block:

```bash
kubectl describe pb/block-1 --namespace=privacy-example
```

Add a privacy claim:

```bash
kubectl apply -f evaluation/examples/dpf-base/add-claim-1.yaml
```

Describe the claim:
```bash
kubectl describe pbc/claim-1 --namespace=privacy-example
```

On your first terminal, you should see that the scheduler detected the claim and is trying to allocate it. Wait a bit, and check the status of the claim again to check if it has been allocated. You can also check the status of the block again.

Finally, clean up:
```bash
kubectl delete -f evaluation/examples/dpf-base/add-claim-1.yaml 
kubectl delete -f evaluation/examples/dpf-base/add-block.yaml
kubectl delete namespace privacy-example
```

We now have a proper abstraction to manage privacy as a native Kubernetes resource. You can refer to the next section for more details, such as how to interact with this privacy resource through real machine learning pipelines.

## 2. Detailed instructions

### 2.1. Structure of the repository

- `evaluation`: Scripts to reproduce the results from the paper
- `privatekube`: Python package to interact with the system
- `system`: Components to deploy PrivateKube on a cluster


### 2.2. Macrobenchmark

The commands in this section have to be run from the `macrobenchmark` directory. You can jump there with:

```bash
cd evaluation/macrobenchmark
```

Please note that the steps below can take several days to run, depending on your hardware. If you want to examine the experimental data without running the preprocessing or the training yourself, you can download some artifacts from this [public bucket](https://storage.googleapis.com/privatekube-public-artifacts). 

#### Download and preprocess the dataset

To download a preprocessed and truncated (140Mb instead of 7Gb) version of the dataset, run the following:

```bash
python dataset.py getmini
```

Alternatively, you can download the raw dataset from the [source](https://nijianmo.github.io/amazon/index.html) and preprocess it yourself as follows:

1. First, download the dataset with:

```bash
python dataset.py download
```

2. Then, preprocess the dataset:
```bash
python dataset.py preprocess
```

3. Finally, merge the preprocessed dataset into a single `reviews.h5` file that can be used by the machine learning pipelines:

```bash
python dataset.py merge
```

4. You can also change the tokenizer from Bert to a custom vocabulary (e.g. the 10,000 most common English words) with:

```bash
python dataset.py convert
```

Use `python dataset.py --help` or `python dataset.py preprocess --help` to have more details about the options.

#### Run individual DP machine learning models

Once you have a preprocessed dataset (either the full dataset or a truncated version), you can run the neural networks with the `classification.py` script, which accepts a lot of options defined at the top of the file. The default flags are set to reasonable values.

Here is an example:
```bash
python workload/models/classification.py --model="bert" --task="product" --n_blocks=200 --n_epochs=3 --dp=1 --epsilon=1.0 --delta=1e-9 --batch_size=32 --user_level=1
```

And another example:
```
python workload/models/classification.py --model="lstm" --task="sentiment" --n_blocks=500 --n_epochs=15 --dp=1 --epsilon=2.0 --delta=1e-9 --batch_size=64 --user_level=0
```

Similarly, you can run statistics:
```bash
python workload/models/statistics.py --dp="event" --n_blocks=1 --model="avg_rating" --epsilon=0.1 
```


#### Run large experiments

To create a workload from the individual models, we have to run them for various parameters.

First, you can generate the experiments (with nice parameters and a local path to store the results) with:

```bash
python run.py generate
```

Then, you can run all the experiments with:

```bash
python run.py all
```

#### End-to-end macrobenchmark

*The code to evaluate the scheduler on the workloads has not been released yet.*

### 2.3. Modifying the scheduler

Instead of using our Docker image, you can compile the DPF scheduler and launch it locally:

```bash
cd dpfscheduler

go build

./dpfscheduler --kubeconfig "$HOME/.kube/config" --n=10
```
