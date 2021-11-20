package algorithm

import (
	"fmt"
	"sync"

	"columbia.github.com/privatekube/dpfscheduler/pkg/scheduler/cache"
	"columbia.github.com/privatekube/dpfscheduler/pkg/scheduler/errors"
	"columbia.github.com/privatekube/dpfscheduler/pkg/scheduler/updater"
	"columbia.github.com/privatekube/dpfscheduler/pkg/scheduler/util"
	columbiav1 "columbia.github.com/privatekube/privacyresource/pkg/apis/columbia.github.com/v1"
	"columbia.github.com/privatekube/privacyresource/pkg/framework"
	"k8s.io/klog"
)

type DpfNSchemeBatch struct {
	DpfBatch
	K int
}

const (
	Skip      = 0
	ToAcquire = 1
	ToReserve = 2
)

func NewDpfNSchemeBatch(
	k int,
	updater updater.ResourceUpdater,
	cache_ *cache.StateCache,
	p2Chan chan<- string,
	scheduler string,
) *DpfNSchemeBatch {

	return &DpfNSchemeBatch{
		DpfBatch: *NewDpfBatch(updater, cache_, p2Chan, scheduler),
		K:        k,
	}
}

func (dpf *DpfNSchemeBatch) BatchAllocate(requestHandler framework.RequestHandler) bool {
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

// Here we call BatchAllocateP1 that will unlock the budget
func (dpf *DpfNSchemeBatch) batchAllocate(
	requestHandler framework.RequestHandler,
	blockStates []*cache.BlockState) bool {

	if !dpf.BatchAllocateP1(requestHandler, blockStates) {
		// No need to reschedule, because the only reason of failure is claim does not exist
		return true
	}

	dpf.AllocateAvailableBudgets(blockStates)
	return true
}

func (dpf *DpfNSchemeBatch) BatchAllocateP1(requestHandler framework.RequestHandler, blockStates []*cache.BlockState) bool {
	requestViewer := requestHandler.Viewer()
	if requestViewer.Claim == nil {
		return false
	}

	isSmallDemand, availableBlockIds := dpf.isSmallDemand(requestViewer, blockStates)

	// Count for the number of successfully data blocks to check if it fulfills the requirement
	minNumOfBlocks := requestViewer.View().AllocateRequest.MinNumberOfBlocks
	maxNumOfBlocks := requestViewer.View().AllocateRequest.MaxNumberOfBlocks
	if maxNumOfBlocks == 0 {
		maxNumOfBlocks = len(blockStates)
	}

	// Get all the data block handled by the scheduler before
	// This action can avoid duplicated handling on the same data block
	//allocatedBudgets := privateDataScheduler.findAllocatedBlocks(requestHandler.Viewer())
	var mu sync.Mutex
	allocatedBudgets := map[string]columbiav1.BudgetAccount{}
	successBlockCount := 0
	done := make(chan bool)

	allocWrapper := func(blockState *cache.BlockState, releasingBudget columbiav1.PrivacyBudget) {
		// temporary variables to receive allocation result of individual data blocks
		var allocatedBudget columbiav1.BudgetAccount

		allocateStrategy := func(block *columbiav1.PrivateDataBlock) error {
			var err error
			if !framework.IsDataBlockReady(block) {
				return fmt.Errorf("data block is not ready")
			}

			var mode int
			if isSmallDemand {
				klog.Infof("Request [%s] is small demand for block [%s]", requestHandler.GetId(), blockState.GetId())
				if availableBlockIds[util.GetBlockId(block)] {
					mode = ToAcquire
				} else {
					mode = Skip
				}
			} else {
				klog.Infof("Request [%s] is not small demand for block [%s]", requestHandler.GetId(), blockState.GetId())
				mode = ToReserve
			}

			allocatedBudget, err = dpf.core.AllocateP1(requestViewer, block, releasingBudget, mode)
			return err
		}

		err := dpf.updater.ApplyOperationToDataBlock(allocateStrategy, blockState)

		if err != nil {
			klog.Errorf("failed to allocate budget from data block [%s] to request [%s]: %v",
				blockState.GetId(), requestHandler.GetId(), err)
			done <- false
			return
		}

		mu.Lock()
		if !allocatedBudget.Reserved.IsEmpty() || !allocatedBudget.Acquired.IsEmpty() {
			allocatedBudgets[blockState.GetId()] = allocatedBudget
		}
		successBlockCount++
		mu.Unlock()

		done <- true
	}

	// We continue to release 1/N until there is no locked budget. The code takes an intersection between
	// 1/N and the pending (locked) budget.
	counter := len(blockStates)
	for _, blockState := range blockStates {
		go allocWrapper(blockState, blockState.Quota(dpf.K))
	}

	for counter > 0 {
		<-done
		counter--
	}

	// minNumOfBlocks == -1 means Strict All or Nothing. Once fails then all fails
	if minNumOfBlocks == columbiav1.StrictAllOrNothing && successBlockCount != len(blockStates) {
		// try to rollback all the qualified data blocks in case some of them were allocated by last
		// scheduling terminated accidentally
		return dpf.BatchRollback(blockStates, requestHandler, errors.AllOrNothingError(nil).Error())
	}

	if successBlockCount < minNumOfBlocks {
		err := errors.NotEnoughDataBlock(minNumOfBlocks, successBlockCount)
		klog.Errorf("failed to allocate enough data blocks to request [%s]", requestHandler.GetId())
		// try to rollback all the qualified data blocks in case some of them were allocated by last
		// scheduling terminated accidentally
		return dpf.BatchRollback(blockStates, requestHandler, err.Error())
	}

	// Now enters the second phase of 2PC.
	// A decision will be first updated in the claim's response
	if isSmallDemand {
		blockStates = filterBlockStates(blockStates, availableBlockIds)
		dpf.batchCompleteAcquiring(blockStates, requestHandler, allocatedBudgets)
	} else {
		dpf.batchCompleteReserving(blockStates, requestHandler, allocatedBudgets)
	}
	return true
}

func (dpf *DpfNSchemeBatch) isSmallDemand(requestViewer framework.RequestViewer, blockStates []*cache.BlockState) (
	bool, map[string]bool) {
	blockIds := map[string]bool{}
	count := 0
	maxNum := requestViewer.View().AllocateRequest.MaxNumberOfBlocks
	if maxNum == 0 {
		maxNum = len(blockStates)
	}

	for _, blockState := range blockStates {
		block := blockState.View()

		if block == nil {
			continue
		}
		expectedBudget := interpretBudgetRequest(requestViewer.View().AllocateRequest.ExpectedBudget, block)

		fairShare := blockState.Quota(dpf.K)
		potentiallyAvailableBudget := block.Status.AvailableBudget.Add(block.Status.PendingBudget)
		if !potentiallyAvailableBudget.HasEnough(expectedBudget) {
			continue
		}

		switch expectedBudget.Type() {
		case columbiav1.EpsDelType:
			// Besides it is smaller than the release unit, and
			// AvailableBudget + PendingBudget should have the requested budget
			if !fairShare.HasEnough(expectedBudget) {
				continue
			}
		case columbiav1.RenyiType:
			if !fairShare.Renyi.HasEnoughInAllAlpha(expectedBudget.Renyi) {
				continue
			}
		}

		count++
		blockIds[blockState.GetId()] = true
		if count == maxNum {
			return true, blockIds
		}
	}

	return false, nil
}

func filterBlockStates(blockStates []*cache.BlockState, blockIdSet map[string]bool) []*cache.BlockState {
	result := make([]*cache.BlockState, 0, len(blockIdSet))

	for _, blockState := range blockStates {
		if blockIdSet[blockState.GetId()] {
			result = append(result, blockState)
		}
	}

	return result
}
