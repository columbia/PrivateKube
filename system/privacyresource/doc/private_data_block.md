# Private Data Block

Private Data Blocks are data blocks with privacy budget. A dataset is divided into data blocks by some properties (e.g. Time, User, ...). If a dataset has constantly incoming data (i.e datastream), its incoming data are continually forming new data blocks. This study aims to introduce the design idea on using Kubernetes Custom Resource to represent Private Data Blocks in Kubernetes. A common scenario is using Kubernetes as a machine-learning application orchestration system, which ingests datastream for training and prediction. Data blocks (split from datastream) are allocated to ML applications based on their requirements (e.g. privacy budget, time range, user group). Given each data block has its privacy budget, not only are the data blocks should be allocated, but their privacy budget has to be managed accordingly. For example, an application requires some fraction of privacy budget from a data block; then, it is killed accidentally, and how should we manage the privacy budget it has taken? Should we return them if no information has been leaked from the application? This study hopes to propose a representation of Private Data Block that can cover most user cases in allocating private data blocks toward privacy-consuming applications.

## Definition of Private Data Block

I am going to show how to define a private data block in Kubernetes using Custom Resource Definition (CRD). And try to defend why some fields are necessary in the struct of private data block. Here is the definition under the context of CRD:

```yaml
openAPIV3Schema:
  type: object
  properties:
    spec:
      type: object
      properties:
        startTime:
          type: integer
        endTime:
          type: integer
        userId:
          type: string
        dataset:
          type: string
        dataSource:
          type: string
        budget:
          type: object
          properties:
            epsilon:
              type: number
            delta:
              type: number
    status:
      type: object
      properties:
        availableBudget:
          type: object
          properties:
            epsilon:
              type: number
            delta:
              type: number
        lockedBudgetMap:
          type: object
          additionalProperties:
            type: object
            properties:
              epsilon:
                type: number
              delta:
                type: number
```
A private data block has some properties to identify itself, and these properties are time range (start/end time), user ID it may be related to, which dataset it belongs to, and the url (i.e. data source) to access its data. Of course, it should have a privacy budget as well. In this paper, I use the Epsilon-Delta notation from differential privacy to measure its privacy budget. The properties mentioned above are defined in `spec` part, and their values are assigned when a private data block is created.

Private data blocks are assigned to one or more privacy-consuming application, and their privacy budgets are consumed accordingly. An available (remaining) budget is needed to record the availability of privacy to consume. Also, the privacy held by applications should be recorded. When an application asks for some amount of privacy to consume, it may eventually only use part of the requested privacy. When the application ends, the unused privacy should be able to return to the data block. In addition, when an application is accidentally crashed; if the data haven't consumed or the results haven't released, its allocated privacy should be able to give back. Therefore, a dictionary `lockedBudgetMap` is declared to record which application requests how much privacy. The requested privacy is considered locked rather than consumed. The locked privacy will eventually be either released (abort) or consumed (commit). Both the `availableBudget` and `lockedBudgetMap` are defined in `status` part.


## APIs for Private Data Block
As a custom resource, private data block is managed by Kubernetes api server. The default operations are CRUD. Since applications will request, consume, and release privacy budgets from a private data block, extra APIs are needed for private data block. Kubernetes allows the customized API by implementing the subresource of a custom resource. One way to implement subresources is using API server extension (e.g. API server builder). 

### CRUD a Private Data Block
Similar to other custom resources, CRUD on private data block is straightforward. Before creation, specifying the desired state of a private data block under `yaml` format, and submit it through the client. Here is an example of a desired state of private data block:

```yaml
# filename: block-1.yaml

apiVersion: "columbia.github.com/v1"
kind: PrivateDataBlock
metadata:
  name: block-1
  namespace: privacy-example
spec:
  startTime: 0
  endTime: 100
  userId: "Pan"
  dataset: "taxi-db"
  dataSource: "some_url/db/block-1"
  budget:
    epsilon: 1
    delta: 1
```

`kubectl` is used to submit this yaml and create a private data block:

```sh
kubectl apply -f block-1.yaml
```

In addition, Kubernetes provides code generator (`kubernetes/code-generator`) to create the clients of custom resource. This has been done in the `sage` project under `privacyresource` folder.

The struct of Private Data Block is defined resemble to the `yaml` format, under the directory `privacyresource/pkg/apis/columbia.github.com/v1/types.go`.

```go
type PrivateDataBlock struct {
	metav1.TypeMeta `json:",inline"`
	// +optional
	metav1.ObjectMeta `json:"metadata,omitempty"`

	// +optional
	Status PrivateDataBlockStatus `json:"status,omitempty"`
	// This is where you can define
	// your own custom spec
	Spec PrivateDataBlockSpec `json:"spec,omitempty"`
}

// custom spec
type PrivateDataBlockSpec struct {
	Budget     PrivacyBudget `json:"budget"`
	StartTime  uint64        `json:"startTime"`
	EndTime    uint64        `json:"endTime"`
	UserId     string        `json:"userId, omitempty"`
	DataSource string        `json:"dataSource"`
	Dataset    string        `json:"dataset, omitempty"`
}

// custom status
type PrivateDataBlockStatus struct {
	AvailableBudget PrivacyBudget `json:"availableBudget"`
	LockedBudgetMap     map[string]PrivacyBudget	 `json:"lockedBudgetMap"`
}

type PrivacyBudget struct {
	Epsilon float64 `json:"epsilon"`
	Delta   float64 `json:"delta"`
}
```

Its client is created under the directory of `privacyresource/pkg/generated`. Here is an example to use client to create a private data block, list all the registered private data blocks, and delete the private data block created before.

```go
package main

import (
	columbiav1 "columbia.github.com/sage/privacyresource/pkg/apis/columbia.github.com/v1"
	privacyclientset "columbia.github.com/sage/privacyresource/pkg/generated/clientset/versioned"
	"flag"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/tools/clientcmd"
	"k8s.io/klog"
)

var (
	masterURL  string
	kubeconfig string 
    //namespace
	ns = "privacy-example"
)

func main() {
	klog.InitFlags(nil)
	flag.Parse()

	cfg, err := clientcmd.BuildConfigFromFlags(masterURL, kubeconfig)
	if err != nil {
		klog.Fatalf("Error building kubeconfig: %s", err.Error())
		return
	}


	privateDataBlockClient, err := privacyclientset.NewForConfig(cfg)
	if err != nil {
		klog.Fatalf("Error building privacy data clientset: %s", err.Error())
		return
	}

	block := columbiav1.PrivateDataBlock{
		ObjectMeta: metav1.ObjectMeta{Name: "block-1", Namespace:ns},
		Spec: columbiav1.PrivateDataBlockSpec{
			Budget:     columbiav1.PrivacyBudget{1, 1},
			StartTime:  0,
			EndTime:    100,
			UserId:     "Pan",
			DataSource: "some_url/db/block-1",
			Dataset:    "taxi-db",
		},
	}
	_, err = privateDataBlockClient.ColumbiaV1().PrivateDataBlocks(ns).Create(&block)

	if err != nil {
		klog.Fatalf("Error creating: %s", err.Error())
	}

	ls, err := privateDataBlockClient.ColumbiaV1().PrivateDataBlocks(ns).List(metav1.ListOptions{})
	if err != nil {
		klog.Fatalf("Error querying list: %s", err.Error())
	}

	if ls == nil || len(ls.Items) == 0 {
		klog.Warning("list is nil or list is empty")
	} else {
		klog.Info(ls.Items[0].Spec)
		klog.Info(ls.Items[0].Status)
	}

	err = privateDataBlockClient.ColumbiaV1().PrivateDataBlocks("privacy-example").Delete(block.Name, nil)
	if err != nil {
		klog.Fatalf("Error delete block: %s", err.Error())
	} else {
		klog.Infof("Succeed to delete: %s", block.Name)
	}
}

func init() {
	flag.StringVar(&kubeconfig, "kubeconfig", "/.kube/config", "Path to a kubeconfig. Only required if out-of-cluster.")
	flag.StringVar(&masterURL, "master", "", "The address of the Kubernetes API server. Overrides any value in kubeconfig. Only required if out-of-cluster.")
}
```

### Acquire Privacy Budget from Private Data Block
If an application needs more privacy budget from a private data block, it can send a request to `privatedatablocks/<block_name>/request` to acquire more privacy budget. The acquired privacy is locked by the private data block until further actions (consume or release) are received to handle it. The request looks like:

```json
{
  "podName": "<pod_name>",
  "budget": {
    "epsilon": 0.1,
    "delta": 0.1
  }
}
```
API server extension receives the "Acquire" request and handles it. The logic of to handle the request looks like:

```go
type Request struct {
    PodNmae  string,
    Budget   PrivacyBudget,
}

func Acquire(block *columbiav1.PrivateDataBlock, request *Request) {
    block = snapshot(block)
    budget := block.Status.LockedBudgetMap[request.PodNmae]
    availableBudget := block.Status.AvailableBudget
    
    epsilon := request.Budget.Epsilon
    delta = request.Budget.Delta
    if availableBudget.Epsilon >= epsilon && availableBudget.Delta >= delta {
        //deduce privacy from available budget
        availableBudget.Epsilon -= epsilon
        availableBudget.Delta -= delta 
        
        //lock the request budget
        budget.Epsilon += epsilon
        budget.Delta += delta
        
        //update the block
        block.Status.AvailableBudget = availableBudget
        block.Status.LockedBudgetMap[pod_name] = budget
        Update(block)
    
        return true //succeed
    } else {
        return false //unable to acquire
    }
}

```

After updating the private data block, API server extension will inform the requester whether the Acquire has succeeded or not.


### Scheduling Private Data Blocks to Pod
A scheduler is built to bind pods (a pod is basically a standalone application) with private data blocks. When a new pod is created, the scheduler finds suitable private data block based on the requirement of the pod, and bind the blocks to this pod. Also, when a new data block is created, it will be bound to the pods that are still hungry for data. The binding process is a combination of calling the Acquire APIs of multiple data blocks.

```go
func Bind(pod *corev1.Pod, blocks []*columbiav1.PrivateDataBlock) {

    // pre-check available privacy budget
    //...

    budget = getRequiredBudget(pod)
    lockedBlocks := make([]*columbiav1.PrivateDataBlock)
    for _, block := range blocks {
        res := privateDataBlockClient.Acquire(
                    block: block,
                    podName: pod.Name
                    budget: budget
                )
        if res.succeed {
            lockedBlocks = append(lockedBlocks, block)
        } 
    }
    
    if len(lockedBlocks) == 0 {
        klog.Warnf("no available private data blocks for %s", pod.Name)
        reschedule(pod)
        return
    }

    podToBind = snapshot(pod)
    AssignBlocks(podToBind, lockedBlocks)
    if err := Update(podToBind); err != nil {
        releaseLockedBudget(lockedBlocks)
        reschedule(pod)
        klog.Errorf("fail to bind pod [%s] wit blocks [%v]", pod.Name, lockedBlocks)
    }
}
```

### Consume Privacy Budget from Private Data Block
Once an application consumes a data block with some privacy budget, it should commit the usage by calling the "Consume" API. The body of request looks like:
```json
{
  "podName": "<pod_name>",
  "budget": {
    "epsilon": 0.1,
    "delta": 0.1
  }
}
```

The API server extension handles the "Consume" request in this way:

```go
func Consume(block *columbiav1.PrivateDataBlock, request *Request) {
    block = snapshot(block)
    budget := block.Status.LockedBudgetMap[request.PodNmae]
    
    epsilon := request.Budget.Epsilon
    delta = request.Budget.Delta
    if budget.Epsilon >= epsilon && budget.Delta >= delta {
        //deduce privacy from the locked budget
        budget.Epsilon -= epsilon
        budget.Delta -= delta
        
        //update the block
        block.Status.LockedBudgetMap[pod_name] = budget
        Update(block)

        return true //succeed
    } else {
        return false //unable to request
    }
}
```

### Release Privacy Budget to Private Data Block
Similar to consuming the requested privacy budget, an application can release its held privacy budget by calling the Release API. Similarly, the body of the request looks like:
```json
{
  "podName": "<pod_name>",
  "budget": {
    "epsilon": 0.1,
    "delta": 0.1
  }
}
``` 

The API server extension handles the Release API request like 

```go
func Release(block *columbiav1.PrivateDataBlock, request *Request) {
    block = snapshot(block)
    budget := block.Status.LockedBudgetMap[request.PodNmae]
    availableBudget := block.Status.AvailableBudget
    epsilon := request.Budget.Epsilon
    delta = request.Budget.Delta
    if budget.Epsilon >= epsilon && budget.Delta >= delta {
        //deduce privacy from the locked budget
        budget.Epsilon -= epsilon
        budget.Delta -= delta
        
        //add the privacy back to available budget
        availableBudget.Epsilon += epsilon
        availableBudget.Delta -= delta

        //update the block
        block.Status.AvailableBudget = availableBudget
        block.Status.LockedBudgetMap[pod_name] = budget
        Update(block)

        return true //succeed
    } else {
        return false //unable to request
    }
}
```