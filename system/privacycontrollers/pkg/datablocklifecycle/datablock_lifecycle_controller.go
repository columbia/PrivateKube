package datablocklifecycle

import (
	"context"

	"columbia.github.com/privatekube/privacycontrollers/pkg/tools/timer"
	"columbia.github.com/privatekube/privacycontrollers/pkg/tools/util"
	columbiav1 "columbia.github.com/privatekube/privacyresource/pkg/apis/columbia.github.com/v1"
	privacyclientset "columbia.github.com/privatekube/privacyresource/pkg/generated/clientset/versioned"
	informer "columbia.github.com/privatekube/privacyresource/pkg/generated/informers/externalversions/columbia.github.com/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/tools/cache"
)

const (
	BaseScheduling = 0
	FlowScheduling = 1
	DpfN           = 2
	DpfT           = 3
)

type BlockLifecycleController struct {
	privacyResourceClient    privacyclientset.Interface
	timer                    timer.Timer
	privateDataBlockInformer informer.PrivateDataBlockInformer
	informersHaveSynced      func() bool
	stop                     chan struct{}
	BlockLifecycleOption
}

type BlockLifecycleOption struct {
	SchedulingPolicy           int
	DeletionGracePeriodSeconds int64
}

func NewBlockController(privacyResourceClient privacyclientset.Interface,
	privateDataBlockInformer informer.PrivateDataBlockInformer, option BlockLifecycleOption) *BlockLifecycleController {

	controller := BlockLifecycleController{}
	controller.privacyResourceClient = privacyResourceClient
	controller.privateDataBlockInformer = privateDataBlockInformer
	controller.informersHaveSynced = func() bool {
		return privateDataBlockInformer.Informer().HasSynced()
	}
	controller.BlockLifecycleOption = option

	// Informer should be run first to start watching!
	//go privateDataBlockInformer.Informer().Run(stopCh)
	controller.addAllEventHandlers()
	return &controller
}

// Run begins watching and scheduling. It waits for cache to be synced, then starts scheduling and blocked until the context is done.
func (controller *BlockLifecycleController) Run() {
	if !cache.WaitForCacheSync(controller.stop, controller.informersHaveSynced) {
		return
	}

	//wait.Until(func(ctx context.Context) {
	//	time.Sleep(1 * time.Second)
	//}, 0, controller.stop)

}

func (controller *BlockLifecycleController) initializeFlowOption(block *columbiav1.PrivateDataBlock) bool {
	if !util.IsFlowMode(block) && controller.SchedulingPolicy != DpfT {
		return false
	}
	modified := false

	if block.Spec.FlowReleasingOption == nil {
		block.Spec.FlowReleasingOption = new(columbiav1.FlowReleasingOption)
		modified = true
	}

	//startTime == 0 means to ask scheduler to select a recent timestamp as start time
	if block.Spec.FlowReleasingOption.StartTime == 0 {
		block.Spec.FlowReleasingOption.StartTime = controller.timer.Now()
		modified = true
	}

	// if duration is not zero and end time does not equal start time + duration, then reset the end time.
	if block.Spec.FlowReleasingOption.Duration > 0 &&
		block.Spec.FlowReleasingOption.EndTime != block.Spec.FlowReleasingOption.StartTime+block.Spec.FlowReleasingOption.Duration {
		block.Spec.FlowReleasingOption.EndTime = block.Spec.FlowReleasingOption.StartTime + block.Spec.FlowReleasingOption.Duration
		modified = true
	}

	return modified
}

func (controller *BlockLifecycleController) updateBlockSpec(block *columbiav1.PrivateDataBlock) error {
	_, err := controller.privacyResourceClient.ColumbiaV1().PrivateDataBlocks(block.Namespace).Update(context.TODO(), block, metav1.UpdateOptions{})
	return err
}

func (controller *BlockLifecycleController) initializeBlockStatus(block *columbiav1.PrivateDataBlock) bool {
	modified := false
	//if block.Status.LockedBudgetMap == nil {
	//	block.Status.LockedBudgetMap = map[string]columbiav1.LockedBudget{}
	//	modified = true
	//}
	//if block.Status.AcquiredBudgetMap == nil {
	//	block.Status.AcquiredBudgetMap = map[string]columbiav1.PrivacyBudget{}
	//	modified = true
	//}
	//if block.Status.ReservedBudgetMap == nil {
	//	block.Status.ReservedBudgetMap = map[string]columbiav1.PrivacyBudget{}
	//	modified = true
	//}
	//if block.Status.CommittedBudgetMap == nil {
	//	block.Status.CommittedBudgetMap = map[string]columbiav1.PrivacyBudget{}
	//	modified = true
	//}
	if len(block.Status.AcquiredBudgetMap) == 0 && len(block.Status.CommittedBudgetMap) == 0 &&
		block.Status.AvailableBudget.IsEmpty() && block.Status.PendingBudget.IsEmpty() {

		if util.IsFlowMode(block) || controller.SchedulingPolicy != BaseScheduling {
			block.Status.PendingBudget = block.Spec.InitialBudget
		} else {
			block.Status.AvailableBudget = block.Spec.InitialBudget
		}
		modified = true
	}
	return modified
}

func (controller *BlockLifecycleController) updateBlockStatus(block *columbiav1.PrivateDataBlock) error {
	_, err := controller.privacyResourceClient.ColumbiaV1().PrivateDataBlocks(block.Namespace).UpdateStatus(context.TODO(), block, metav1.UpdateOptions{})
	return err
}

func (controller *BlockLifecycleController) isBlockDeletable(block *columbiav1.PrivateDataBlock) bool {
	if block.Status.PendingBudget.IsEmpty() && block.Status.AvailableBudget.IsEmpty() &&
		block.Status.AcquiredBudgetMap != nil && len(block.Status.AcquiredBudgetMap) == 0 {
		return true
	}
	return false
}

func (controller *BlockLifecycleController) deleteDataBlock(block *columbiav1.PrivateDataBlock) error {
	deleteOptions := metav1.DeleteOptions{
		GracePeriodSeconds: &controller.DeletionGracePeriodSeconds,
	}
	err := controller.privacyResourceClient.ColumbiaV1().PrivateDataBlocks(block.Namespace).Delete(context.TODO(), block.Name, deleteOptions)
	return err
}
