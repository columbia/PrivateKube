# PrivateKube

PrivateKube is an extension to the popular Kubernetes datacenter orchestrator that adds privacy as a new type of resource to be managed alongside other traditional compute resources, such as CPU, GPU, and memory.  A description of the project can be found on our [webpage](https://systems.cs.columbia.edu/PrivateKube/) and in our OSDI'21 paper titled [Privacy Budget Scheduling](https://www.usenix.org/conference/osdi21/presentation/luo) (PDF locally available [here](https://columbia.github.io/PrivateKube/papers/osdi2021privatekube.pdf).


## Project structure

This repository contains the artifact release for the OSDI paper.  Currently, we provide:
- [system](system/): The PrivateKube system, it includes the implementation of privacy resource and scheduling algorithm called *Dominant Privacy Fairness (DPF)*.
- [privatekube](privatekube/): A python client for interaction with the PrivateKube system and performing macrobenchmark evaluation.
- [simulator](simulator/): A simulator for microbenchmarking privacy scheduling algorithms in tightly controlled settings.
- [examples](examples/): Usage examples for various components, please refer its [README](./examples/README.md) for details.
- [evaluation](evaluation/): Scripts to reproduce the macrobenchmark and microbenchmark evaluation results from our paper.

We do not provide yet but will do so in the near future:
- The Kubeflow Pipeline interface (currently tied to our Google Cloud infrastructure) and an example of pipeline.

## 1. Getting started with PrivateKube

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
pip install -r privatekube/requirements.txt
```

Install the PrivateKube package:
```bash
pip install -e privatekube
```

#### Deploy PrivateKube to your cluster

You can deploy PrivateKube in one line and directly by running:

```bash
source system/deploy.sh
```

If you prefer to understand what is going on, you can run the following commands one by one:

First, let's create a clean namespace to separate PrivateKube from the rest of the cluster:

```bash
kubectl create ns privatekube
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

There are additional instructions in the [system directory](system/) if you want to modify the scheduler or run it locally.


### 1.3. Hello World

Open a first terminal. We are going to monitor the logs of the scheduler to see it in action. Find the scheduler pod with:

```bash
kubectl get pods -A | grep scheduler
```

Then, in the same terminal, monitor the logs of the scheduler with something similar to:
```bash
kubectl logs --follow dpf-scheduler-5fb6886497-w7x49 -n privatekube 
```

(alternatively, you can directly use: `kubectl logs --follow "$(kubectl get pods -n privatekube | grep scheduler | awk -F ' ' '{print $1}')" -n privatekube`)

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
kubectl apply -f examples/privacyresource/dpf-base/add-block.yaml
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
kubectl apply -f examples/privacyresource/dpf-base/add-claim-1.yaml
```

Describe the claim:
```bash
kubectl describe pbc/claim-1 --namespace=privacy-example
```

On your first terminal, you should see that the scheduler detected the claim and is trying to allocate it. Wait a bit, and check the status of the claim again to check if it has been allocated. You can also check the status of the block again.

Finally, clean up:
```bash
kubectl delete -f examples/privacyresource/dpf-base/add-claim-1.yaml 
kubectl delete -f examples/privacyresource/dpf-base/add-block.yaml
kubectl delete namespace privacy-example
```

We now have a proper abstraction to manage privacy as a native Kubernetes resource.  The next section will provide an end-to-end example for how to interact with the privacy resource through a real machine learning pipeline.  You can also refer to [evaluation/macrobenchmark](evaluation/macrobenchmark) to reproduce part of our evaluation of this resource and the DPF algorithm we developed for it.

# Example PrivateKube Usage inn DP ML Pipeline

To appear.

