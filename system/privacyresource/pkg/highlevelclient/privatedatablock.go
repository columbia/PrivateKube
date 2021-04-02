package highlevelclient

import (
	"fmt"

	columbiav1 "columbia.github.com/sage/privacyresource/pkg/apis/columbia.github.com/v1"
	lowlevelclient "columbia.github.com/sage/privacyresource/pkg/generated/clientset/versioned/typed/columbia.github.com/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)


var budgetPrecision = 1e-8

type PrivateDataBlockClient struct {
	namespace string
	client    lowlevelclient.PrivateDataBlockInterface
}

// PrivateDataBlockInterface has methods to work with PrivateDataBlock resources.
type PrivateDataBlockClientInterface interface {
	Create(*columbiav1.PrivateDataBlock) (*columbiav1.PrivateDataBlock, error)
	Update(*columbiav1.PrivateDataBlock) (*columbiav1.PrivateDataBlock, error)
	UpdateStatus(*columbiav1.PrivateDataBlock) (*columbiav1.PrivateDataBlock, error)
	Delete(name string) error
	//DeleteCollection(options *metav1.DeleteOptions, listOptions metav1.ListOptions) error
	Get(name string) (*columbiav1.PrivateDataBlock, error)
	//List(opts metav1.ListOptions) (*columbiav1.PrivateDataBlockList, error)
	//Patch(name string, pt types.PatchType, data []byte, subresources ...string) (result *columbiav1.PrivateDataBlock, err error)
	//lowlevelclient.PrivateDataBlockExpansion

	Acquire(block *columbiav1.PrivateDataBlock, podName string, budget columbiav1.EpsilonDeltaBudget) (
		*columbiav1.PrivateDataBlock, error)
	Consume(block *columbiav1.PrivateDataBlock, podName string, budget columbiav1.EpsilonDeltaBudget) (
		*columbiav1.PrivateDataBlock, error)
	Release(block *columbiav1.PrivateDataBlock, podName string, budget columbiav1.EpsilonDeltaBudget) (
		*columbiav1.PrivateDataBlock, error)
	GetLockedPrivacyBudget(block *columbiav1.PrivateDataBlock, podName string) (*columbiav1.EpsilonDeltaBudget, error)
}

func (client *PrivateDataBlockClient) Create(block *columbiav1.PrivateDataBlock) (
	*columbiav1.PrivateDataBlock, error) {
	return client.client.Create(block)
}

func (client *PrivateDataBlockClient) Update(block *columbiav1.PrivateDataBlock) (
	*columbiav1.PrivateDataBlock, error) {
	return client.client.Update(block)
}

func (client *PrivateDataBlockClient) UpdateStatus(block *columbiav1.PrivateDataBlock) (
	*columbiav1.PrivateDataBlock, error) {
	return client.client.UpdateStatus(block)
}

func (client *PrivateDataBlockClient) Delete(name string) error {
	return client.client.Delete(name, &metav1.DeleteOptions{})
}

func (client *PrivateDataBlockClient) Get(name string) (*columbiav1.PrivateDataBlock, error) {
	return client.client.Get(name, metav1.GetOptions{})
}

func (client *PrivateDataBlockClient) Acquire(block *columbiav1.PrivateDataBlock,
	podName string, budget columbiav1.EpsilonDeltaBudget) (*columbiav1.PrivateDataBlock, error) {
	block = block.DeepCopy()

	lockedBudget := block.Status.AcquiredBudgetMap[podName]
	availableBudget := block.Status.AvailableBudget
	if availableBudget.Epsilon >= budget.Epsilon && availableBudget.Delta >= budget.Delta {
		//deduce privacy from available budget
		availableBudget.Epsilon -= budget.Epsilon
		availableBudget.Delta -= budget.Delta

		//lock the request budget
		lockedBudget.Epsilon += budget.Epsilon
		lockedBudget.Delta += budget.Delta

		//update the block
		block.Status.AvailableBudget = availableBudget
		block.Status.AcquiredBudgetMap[podName] = lockedBudget
		block, err := client.client.UpdateStatus(block)
		if err != nil {
			return nil, fmt.Errorf("fail to acquire the privacy budget %v", err)
		}
		return block, err //succeed
	} else {
		return nil, fmt.Errorf("not enough privacy budget to acquire in [%s]", block.Name) //unable to request
	}
}

func (client *PrivateDataBlockClient) Consume(block *columbiav1.PrivateDataBlock,
	podName string, budget columbiav1.EpsilonDeltaBudget) (*columbiav1.PrivateDataBlock, error) {
	block = block.DeepCopy()

	lockedBudget := block.Status.AcquiredBudgetMap[podName]
	if lockedBudget.Epsilon >= budget.Epsilon && lockedBudget.Delta >= budget.Delta {
		//deduce privacy from the locked budget
		lockedBudget.Epsilon -= budget.Epsilon
		lockedBudget.Delta -= budget.Delta

		//update the block
		if lockedBudget.Epsilon >= budgetPrecision || lockedBudget.Delta >= budgetPrecision {
			block.Status.AcquiredBudgetMap[podName] = lockedBudget
		} else {
			delete(block.Status.AcquiredBudgetMap, podName)
		}
		block, err := client.client.UpdateStatus(block)
		if err != nil {
			return nil, fmt.Errorf("fail to consume the privacy budget %v", err)
		}
		return block, err //succeed
	} else {
		return nil, fmt.Errorf("not enough privacy budget to consume in [%s]", block.Name)
	}
}

func (client *PrivateDataBlockClient) Release(block *columbiav1.PrivateDataBlock,
	podName string, budget columbiav1.EpsilonDeltaBudget) (*columbiav1.PrivateDataBlock, error) {
	block = block.DeepCopy()

	lockedBudget := block.Status.AcquiredBudgetMap[podName]
	availableBudget := block.Status.AvailableBudget
	if lockedBudget.Epsilon >= budget.Epsilon && lockedBudget.Delta >= budget.Delta {
		availableBudget.Epsilon += budget.Epsilon
		availableBudget.Delta += budget.Delta

		lockedBudget.Epsilon -= budget.Epsilon
		lockedBudget.Delta -= budget.Delta

		block.Status.AvailableBudget = availableBudget
		if lockedBudget.Epsilon >= budgetPrecision || lockedBudget.Delta >= budgetPrecision {
			block.Status.AcquiredBudgetMap[podName] = lockedBudget
		} else {
			delete(block.Status.AcquiredBudgetMap, podName)
		}
		block, err := client.client.UpdateStatus(block)
		if err != nil {
			return nil, fmt.Errorf("fail to release the privacy budget %v", err)
		}
		return block, err //succeed
	} else {
		return nil, fmt.Errorf("not enough privacy budget to release in [%s]", block.Name)
	}
}

func (client *PrivateDataBlockClient) GetLockedPrivacyBudget(blockName string, podName string) (
	columbiav1.EpsilonDeltaBudget, error) {
	pod, err := client.Get(podName)
	if err != nil {
		return columbiav1.EpsilonDeltaBudget{},
			fmt.Errorf("unable to get the locked privacy budget of [%s]: %v", podName, err)
	}
	if budget, ok := pod.Status.AcquiredBudgetMap[blockName]; ok {
		return budget, nil
	} else {
		return columbiav1.EpsilonDeltaBudget{},
			fmt.Errorf("private data block [%s] locks no privacy budget for [%s]", blockName, podName)
	}
}
