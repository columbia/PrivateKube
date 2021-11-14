package algorithm

import (
	"fmt"

	"columbia.github.com/privatekube/dpfscheduler/pkg/scheduler/cache"
	"columbia.github.com/privatekube/dpfscheduler/pkg/scheduler/util"
	columbiav1 "columbia.github.com/privatekube/privacyresource/pkg/apis/columbia.github.com/v1"
	"columbia.github.com/privatekube/privacyresource/pkg/framework"
	"k8s.io/klog"
)

func (dpf *DpfBatch) BatchRollback(blockStates []*cache.BlockState, requestHandler framework.RequestHandler, errMessage string) bool {
	// add a failure response to the privacy budget claim
	err := dpf.updater.ApplyOperationToClaim(func(claim *columbiav1.PrivacyBudgetClaim) error {
		if len(claim.Status.Responses) > requestHandler.RequestIndex {

			// If status is pending, then convert it to FAILURE
			response := &claim.Status.Responses[requestHandler.RequestIndex]
			if response.State == columbiav1.Pending {
				response.State = columbiav1.Failure
				response.Error = errMessage
				response.AllocateResponse.Budgets = map[string]columbiav1.BudgetAccount{}
				response.AllocateResponse.FinishTime = dpf.timer.Now()
			}

		} else {
			appendResponse(claim, columbiav1.Response{
				Identifier: claim.Spec.Requests[requestHandler.RequestIndex].Identifier,
				State:      columbiav1.Failure,
				Error:      errMessage,
				AllocateResponse: &columbiav1.AllocateResponse{
					Budgets:    map[string]columbiav1.BudgetAccount{},
					FinishTime: dpf.timer.Now(),
				},
			})
		}
		claim.Status.ReservedBudgets = nil
		return nil
	}, requestHandler.ClaimHandler)

	if err != nil {
		klog.Infof("unable to set a failure response to the privacy budget claim [%s]: %v",
			requestHandler.ClaimHandler.GetId(), err)
		return false
	}

	klog.Infof("succeed to set a failure response to the privacy budget claim [%s]", requestHandler.ClaimHandler.GetId())
	requestViewer := requestHandler.Viewer()

	done := make(chan bool)
	rollbackWrapper := func(blockState *cache.BlockState) {
		err := dpf.updater.ApplyOperationToDataBlock(func(block *columbiav1.PrivateDataBlock) error {
			return rollbackLockedBudget(block, requestViewer)
		}, blockState)

		blockId := blockState.GetId()
		if err == nil {
			klog.Infof("succeeded to rollback the request [%s] from data block [%s]",
				requestViewer.Id, blockId)
		} else {
			klog.Errorf("failed to rollback the request [%s] from data block [%s]: %v",
				requestViewer.Id, blockId, err)
		}

		done <- true
	}

	counter := len(blockStates)
	for _, blockState := range blockStates {
		go rollbackWrapper(blockState)
	}

	for counter > 0 {
		<-done
		counter--
	}

	return true
}

func (dpf *DpfBatch) BatchRollbackByKey(blockIds []string, requestHandler framework.RequestHandler, errMessage string) bool {
	blockStates := make([]*cache.BlockState, 0, len(blockIds))

	for _, blockId := range blockIds {
		blockState := dpf.cache.GetBlock(blockId)
		if blockState != nil {
			blockStates = append(blockStates, blockState)
		}
	}

	return dpf.BatchRollback(blockStates, requestHandler, errMessage)
}

func rollbackLockedBudget(block *columbiav1.PrivateDataBlock, requestViewer framework.RequestViewer) error {
	state := block.Status.LockedBudgetMap[requestViewer.Id].State
	// If the request Id does not exist as a key, an empty value will be assigned to the state.
	if state == "" {
		return fmt.Errorf("the block is not locked by the claim, skip it")
	}

	switch state {
	case columbiav1.Allocating:
	case columbiav1.DpfAcquiring:
	case columbiav1.DpfReserving:
		return rollbackAllocating(block, requestViewer)
	case columbiav1.Committed:
		return rollbackCommitted(block, requestViewer)
	case columbiav1.Aborting, columbiav1.Aborted:
		return fmt.Errorf("not allow to rollback Abort operation")
	default:
		return fmt.Errorf("unknown operation")
	}
	return nil
}

func rollbackAllocating(block *columbiav1.PrivateDataBlock, requestViewer framework.RequestViewer) error {
	lockedBudget := block.Status.LockedBudgetMap[requestViewer.Id]
	block.Status.AvailableBudget.IAdd(lockedBudget.Budget.Acquired)
	util.DeleteLock(block, requestViewer)
	return nil
}

func rollbackCommitted(block *columbiav1.PrivateDataBlock, requestViewer framework.RequestViewer) error {
	lockedBudget := block.Status.LockedBudgetMap[requestViewer.Id]
	claimId := util.GetClaimId(requestViewer.Claim)

	committedBudget := block.Status.CommittedBudgetMap[claimId]
	util.SetBudgetMapAndDeleteEmpty(block.Status.CommittedBudgetMap, claimId,
		committedBudget.Minus(lockedBudget.Budget.Acquired).Minus(lockedBudget.Budget.Reserved))

	acquiredBudget := block.Status.AcquiredBudgetMap[claimId]
	block.Status.AcquiredBudgetMap[claimId] = acquiredBudget.Add(lockedBudget.Budget.Acquired)

	reservedBudget := block.Status.ReservedBudgetMap[claimId]
	block.Status.ReservedBudgetMap[claimId] = reservedBudget.Add(lockedBudget.Budget.Reserved)
	block.Status.AvailableBudget.IAdd(lockedBudget.Budget.Reserved)

	util.DeleteLock(block, requestViewer)
	return nil
}
