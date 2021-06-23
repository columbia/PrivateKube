# System


## Install the packages
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

## Build the scheduler and the controllers

In the current directory (`system`), run:

```
docker build -t privatekube/dpfscheduler:latest . 
```


You can then push it to your Docker repository, for instance:

```
docker push privatekube/dpfscheduler:latest
```

Finally, you can update the Kubernetes deployment `dpfscheduler/manifests/scheduler.yaml` to fetch the latest version from your repository.