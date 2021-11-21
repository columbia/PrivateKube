package flowreleasing

import (
	schedulercache "columbia.github.com/privatekube/dpfscheduler/pkg/scheduler/cache"
	"columbia.github.com/privatekube/dpfscheduler/pkg/scheduler/timing"
	"columbia.github.com/privatekube/dpfscheduler/pkg/scheduler/updater"
	columbiav1 "columbia.github.com/privatekube/privacyresource/pkg/apis/columbia.github.com/v1"
)

type Controller struct {
	cache           schedulercache.Cache
	resourceUpdater *updater.ResourceUpdater
	timer           timing.Timer
	ReleaseOption
}

type ReleaseOption struct {
	DefaultDuration int64
}

func MakeController(
	cache schedulercache.Cache,
	resourceUpdater *updater.ResourceUpdater,
	timer timing.Timer,
	option ReleaseOption) Controller {

	allocator := Controller{
		cache:           cache,
		resourceUpdater: resourceUpdater,
		timer:           timer,
		ReleaseOption:   option,
	}

	return allocator
}

func (controller *Controller) Release() []*schedulercache.BlockState {
	releasedBlocks := make([]*schedulercache.BlockState, 0, 16)

	for _, blockState := range controller.cache.AllBlocks() {
		_ = controller.resourceUpdater.ApplyOperationToDataBlock(controller.releaseLatestBudget, blockState)

		// klog.Infof("block [%s] has pending budget %v, available budget %v", blockState.GetId(),
		// 	blockState.View().Status.PendingBudget.ToString(),
		// 	blockState.View().Status.AvailableBudget.ToString())

		if !blockState.View().Status.AvailableBudget.IsEmpty() {
			releasedBlocks = append(releasedBlocks, blockState)
		}
	}

	return releasedBlocks
}

// return value indicates whether the data block has been updated
func (controller *Controller) releaseLatestBudget(block *columbiav1.PrivateDataBlock) error {
	if block.Spec.FlowReleasingOption == nil || block.Spec.FlowReleasingOption.StartTime == 0 {
		return budgetReleaseNotStart()
	}

	if block.Status.PendingBudget.IsEmpty() {
		return noPendingBudget()
	}

	now := controller.timer.Now()
	if now <= block.Spec.FlowReleasingOption.StartTime {
		return budgetReleaseNotStart()
	}

	// if startTime + duration != endTime, override the endTime to make them consistent.
	var endTime int64
	if block.Spec.FlowReleasingOption.Duration > 0 {
		endTime = block.Spec.FlowReleasingOption.StartTime + block.Spec.FlowReleasingOption.Duration
	} else if block.Spec.FlowReleasingOption.EndTime > 0 {
		endTime = block.Spec.FlowReleasingOption.EndTime
	} else {
		endTime = block.Spec.FlowReleasingOption.StartTime + controller.DefaultDuration
	}

	// if this is the first time to release budget, set the last releasing time as the start time of
	// the budget flow.
	if block.Status.LastBudgetReleaseTime == 0 {
		block.Status.LastBudgetReleaseTime = block.Spec.FlowReleasingOption.StartTime
	}

	remainingDuration := endTime - block.Status.LastBudgetReleaseTime
	lastDuration := now - block.Status.LastBudgetReleaseTime

	//klog.Infof("lastDuration, RemainDuration", lastDuration, remainingDuration)
	// this evaluation also prevents zero division.
	if lastDuration >= remainingDuration {
		releaseBudget(block, block.Status.PendingBudget.Copy(), now)
		return nil
	}

	releasingBudget := block.Status.PendingBudget.Mul(float64(lastDuration) / float64(remainingDuration))
	releaseBudget(block, releasingBudget, now)
	return nil
}

func releaseBudget(block *columbiav1.PrivateDataBlock, budget columbiav1.PrivacyBudget, timestamp int64) {
	block.Status.PendingBudget.ISub(budget)
	block.Status.AvailableBudget.IAdd(budget)
	block.Status.LastBudgetReleaseTime = timestamp
}
