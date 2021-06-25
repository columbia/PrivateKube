# System

See the [main README](https://github.com/columbia/PrivateKube) for deployment instructions.
Below are more specific details:

## Installing the packages
```bash
cd privacyresource
go mod vendor
cd ..

cd privacycontrollers
go mod vendor
cd ..

cd dpfscheduler
go mod vendor
cd ..
```

## Building the scheduler and the controllers

In the current directory (`system`), run:

```
docker build -t privatekube/dpfscheduler:latest . 
```


You can then push it to your Docker repository, for instance:

```
docker push privatekube/dpfscheduler:latest
```

Finally, you can update the Kubernetes deployment `dpfscheduler/manifests/scheduler.yaml` to fetch the latest version from your repository.

## Running a local version of the scheduler 

Instead of using our Docker image, you can compile the DPF scheduler and launch it locally:

```bash
go build

./dpfscheduler --kubeconfig "$HOME/.kube/config" --n=10
```


