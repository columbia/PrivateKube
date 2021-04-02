# Privacy as a Resource

## How to generate client set for privacy resource?

```sh
go mod vendor
./hack/update-codegen.sh
```

## About PrivateDataBlocks

PrivateDataBlocks control the flow of data. They have privacy budgets that run out after a certain amount of use.

Every time a new pod is created (i.e. a new need for data) the scheduler in `../privatedatascheduler` finds a PrivateDataBlock that matches the pod's requirement and gives it to that pod.

## PrivateDataBlock Implementation

Kubernetes is fundamentally API-based. From the [Kubernetes docs](https://kubernetes.io/docs/reference/using-api/api-overview/): "All operations and communications between components, and external user commands are REST API calls that the API Server handles. Consequently, everything in the Kubernetes platform is treated as an API object and has a corresponding entry in the [API](https://kubernetes.io/docs/reference/generated/kubernetes-api/v1.18/)." Every resource type is a member of an API group.

We have extended the Kubernetes API with custom resources. We have defined PrivateDataBlock as a CustomResource in Kubernetes. This [walkthrough](https://insujang.github.io/2020-02-13/programming-kubernetes-crd/) of CustomResourceDefinitions in Kubernetes is very helpful, including its cited sources. The privatedatablocks can be created by kubectl applying their .yaml files.

CustomResources in Kubernetes have the properties *Spec* that represents desired state and *Status* that represents current state.

To update the Status field of a PrivateDataBlock, you need to use the client. We are using the [Go Client](https://github.com/kubernetes/client-go/tree/master/examples).

## Handling Events on the PrivateDataBlocks

We have created a custom controller that handles events on the PrivateDataBlock Custom Resource. There are many ways to create a custom controller; we are using code-generator. Clients to interact with the PrivateDataBlock Custom Resources must be generated with [`k8s.io/code-generator`](https://github.com/kubernetes/code-generator) using the using the `./hack/update-codegen.sh` script in this folder. Code-generator has the files types.go, register.go, and doc.go as requisites for running.

We use [`fake clients`](https://github.com/kubernetes/client-go/tree/master/examples/fake-client) for testing.

Informers and listers are the basis for building controllers.

Informers keep track of create, update, and delete operations on the PrivateDataBlock CustomResource.

Through the Lister, you can access items in the local cache storage. Objects are kept in a thread-safe store. The UpdateStatus() function is defined [here](https://github.com/columbia/sage/blob/master/privacyresource/pkg/generated/clientset/versioned/typed/columbia.github.com/v1/privacybudgetclaim.go). A test can be found [here](https://github.com/columbia/sage/blob/master/privacyresource/main.go#L119).

SyncHandler compares the actual state with the desired state (Spec) and updates the status block of the PrivateDataBlock resource.

## Dependencies

We use Go mod for dependency management.