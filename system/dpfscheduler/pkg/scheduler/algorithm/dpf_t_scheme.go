package algorithm

import (
	"fmt"

	"columbia.github.com/privatekube/dpfscheduler/pkg/scheduler/cache"
	"columbia.github.com/privatekube/dpfscheduler/pkg/scheduler/errors"
	"columbia.github.com/privatekube/dpfscheduler/pkg/scheduler/updater"
	columbiav1 "columbia.github.com/privatekube/privacyresource/pkg/apis/columbia.github.com/v1"
	"columbia.github.com/privatekube/privacyresource/pkg/framework"
	"k8s.io/klog"
)

type DpfTSchemeBatch struct {
	DpfBatch
}

func NewDpfTSchemeBatch(
	updater updater.ResourceUpdater,
	cache_ cache.Cache,
	p2Chan chan<- string,
) *DpfTSchemeBatch {

	return &DpfTSchemeBatch{
		DpfBatch: *NewDpfBatch(updater, cache_, p2Chan),
	}
}

func (dpf *DpfTSchemeBatch) BatchAllocate(requestHandler framework.RequestHandler) bool {
	requestViewer := requestHandler.Viewer()
	allocateRequest := requestViewer.View().AllocateRequest

	blockStates := dpf.getAvailableBlocks(requestViewer)
	klog.Infof("Available data blocks for request %s are: %v", requestHandler.GetId(), cache.GetIdOfBlockStates(blockStates))

	if len(blockStates) < allocateRequest.MinNumberOfBlocks {
		err := errors.NotEnoughDataBlock(allocateRequest.MinNumberOfBlocks, len(blockStates))
		klog.Warning(err)
		// roll back the allocation because there is no enough data blocks to allocate

		return dpf.BatchRollback(blockStates, requestHandler, err.Error())
	}

	return dpf.batchAllocate(requestHandler, blockStates)
}

func (dpf *DpfTSchemeBatch) batchAllocate(
	requestHandler framework.RequestHandler,
	blockStates []*cache.BlockState) bool {

	if !dpf.BatchAllocateP1(requestHandler, blockStates) {
		// No need to reschedule, because the only reason of failure is claim does not exist
		return true
	}

	dpf.AllocateAvailableBudgets(blockStates)
	return true
}

func (dpf *DpfTSchemeBatch) BatchAllocateP1(requestHandler framework.RequestHandler, blockStates []*cache.BlockState) bool {
	requestViewer := requestHandler.Viewer()
	if requestViewer.Claim == nil {
		return false
	}

	// Get all the data block handled by the scheduler before
	// This action can avoid duplicated handling on the same data block
	//allocatedBudgets := privateDataScheduler.findAllocatedBlocks(requestHandler.Viewer())
	allocatedBudgets := map[string]columbiav1.BudgetAccount{}
	// temporary variables to receive allocation result of individual data blocks
	var allocatedBudget columbiav1.BudgetAccount

	allocateStrategy := func(block *columbiav1.PrivateDataBlock) error {
		var err error
		if !framework.IsDataBlockReady(block) {
			return fmt.Errorf("data block is not ready")
		}

		allocatedBudget, err = dpf.core.AllocateP1(requestViewer, block, columbiav1.PrivacyBudget{}, ToReserve)
		return err
	}

	// Count for the number of successfully data blocks to check if it fulfills the requirement
	minNumOfBlocks := requestViewer.View().AllocateRequest.MinNumberOfBlocks
	successBlockCount := 0

	for _, blockState := range blockStates {
		if err := dpf.updater.ApplyOperationToDataBlock(allocateStrategy, blockState); err == nil {
			klog.Infof("succeeded to reserve budget from data block [%s] to request [%s]",
				blockState.GetId(), requestHandler.GetId())

			allocatedBudgets[blockState.GetId()] = allocatedBudget
			successBlockCount++
		} else {
			klog.Errorf("failed to reserve budget from data block [%s] to request [%s]: %v",
				blockState.GetId(), requestHandler.GetId(), err)

			// minNumOfBlocks == -1 means Strict All or Nothing. Once fails then all fails
			if minNumOfBlocks == columbiav1.StrictAllOrNothing {
				// try to rollback all the qualified data blocks in case some of them were allocated by last
				// scheduling terminated accidentally
				return dpf.BatchRollback(blockStates, requestHandler, errors.AllOrNothingError(err).Error())
			}
		}
	}

	if successBlockCount < minNumOfBlocks {
		err := errors.NotEnoughDataBlock(minNumOfBlocks, successBlockCount)
		klog.Errorf("failed to allocate enough data blocks to request [%s]", requestHandler.GetId())
		// try to rollback all the qualified data blocks in case some of them were allocated by last
		// scheduling terminated accidentally
		return dpf.BatchRollback(blockStates, requestHandler, err.Error())
	}

	dpf.batchCompleteReserving(blockStates, requestHandler, allocatedBudgets)
	return true
}
