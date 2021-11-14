package algorithm

import (
	"fmt"
	"math"
	"sync"

	"columbia.github.com/privatekube/dpfscheduler/pkg/scheduler/cache"
	"columbia.github.com/privatekube/dpfscheduler/pkg/scheduler/errors"
	"columbia.github.com/privatekube/dpfscheduler/pkg/scheduler/timing"
	"columbia.github.com/privatekube/dpfscheduler/pkg/scheduler/updater"
	"columbia.github.com/privatekube/dpfscheduler/pkg/scheduler/util"
	columbiav1 "columbia.github.com/privatekube/privacyresource/pkg/apis/columbia.github.com/v1"
	"columbia.github.com/privatekube/privacyresource/pkg/framework"
	"k8s.io/klog"
)

// DpfBatch is a snapshot of the tasks waiting to be allocated + the set of blocks and their available budgets
type DpfBatch struct {
	sync.Mutex
	core    CoreAlgorithm
	updater updater.ResourceUpdater
	cache   cache.Cache
	timer   timing.Timer

	// channel used in P2 Allocate to notify exterior about the allocation
	p2Chan chan<- string
}

func NewDpfBatch(
	updater updater.ResourceUpdater,
	cache_ cache.Cache,
	p2Chan chan<- string,
) *DpfBatch {

	return &DpfBatch{
		core:    CoreAlgorithm{},
		updater: updater,
		cache:   cache_,
		p2Chan:  p2Chan,
	}
}

// BatchAllocateP2 tries to allocate a request
func (dpf *DpfBatch) BatchAllocateP2(claimState *cache.ClaimState, allocatingBlockIds []string) bool {
	requestHandler := claimState.NextRequest()

	requestViewer := requestHandler.Viewer()
	if !requestViewer.IsPending() {
		return false
	}

	// Count for the number of successfully data blocks to check if it fulfills the requirement
	minNumOfBlocks := requestViewer.View().AllocateRequest.MinNumberOfBlocks

	// transfer allocatingBlockIds to set
	allocatingSet := make(map[string]bool)
	for _, blockId := range allocatingBlockIds {
		allocatingSet[blockId] = true
	}

	blockIds := util.GetStringKeysFromMap(requestViewer.Claim.Status.ReservedBudgets)
	blockStates := dpf.cache.GetBlocks(blockIds)

	// Get all the data block handled by the scheduler before
	// This action can avoid duplicated handling on the same data block
	//allocatedBudgets := privateDataScheduler.findAllocatedBlocks(requestHandler.Viewer())
	var mu sync.Mutex
	allocatedBudgets := map[string]columbiav1.BudgetAccount{}
	successBlockCount := 0
	done := make(chan bool)

	allocWrapper := func(blockState *cache.BlockState) {
		// temporary variables to receive allocation result of individual data blocks
		var allocatedBudget columbiav1.BudgetAccount
		toAcquire := allocatingSet[blockState.GetId()]

		allocateStrategy := func(block *columbiav1.PrivateDataBlock) error {
			var err error
			if !framework.IsDataBlockReady(block) {
				return fmt.Errorf("data block is not ready")
			}

			allocatedBudget, err = dpf.core.AllocateP2(requestViewer, block, toAcquire)
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

		klog.Infof("[Batch AllocateP2] succeeded to allocate budget from data block [%s] to request [%s]",
			blockState.GetId(), requestHandler.GetId())
		done <- true
	}

	counter := len(blockStates)
	for _, blockState := range blockStates {
		go allocWrapper(blockState)
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
	dpf.batchCompleteAcquiring(blockStates, requestHandler, allocatedBudgets)
	return true
}

func (dpf *DpfBatch) BatchAllocateTimeout(requestHandler framework.RequestHandler) bool {
	requestViewer := requestHandler.Viewer()
	if !requestViewer.IsPending() {
		return false
	}

	timeoutStrategy := func(block *columbiav1.PrivateDataBlock) error {
		return dpf.core.AllocateTimeout(requestViewer, block)
	}

	blockIds := util.GetStringKeysFromMap(requestViewer.Claim.Status.ReservedBudgets)
	blockStates := dpf.cache.GetBlocks(blockIds)

	done := make(chan bool)
	timeoutWrapper := func(blockState *cache.BlockState) {
		if err := dpf.updater.ApplyOperationToDataBlock(timeoutStrategy, blockState); err == nil {
			klog.Infof("succeeded to timeout pending request [%s] at data block [%s]",
				requestHandler.GetId(), blockState.GetId())

		} else {
			klog.Errorf("failed to timeout pending request [%s] at data block [%s]: %v",
				requestHandler.GetId(), blockState.GetId(), err)

		}

		done <- true
	}

	counter := len(blockStates)
	for _, blockState := range blockStates {
		go timeoutWrapper(blockState)
	}

	for counter > 0 {
		<-done
		counter--
	}

	// Now enters the second phase of 2PC.
	// A decision will be first updated in the claim's response
	dpf.BatchRollback(blockStates, requestHandler, "timeout")
	return true
}

// getInfluencedClaims fetches all the claims that demand one of the blocks
// It computes the dominant share for each claim
func (dpf *DpfBatch) getInfluencedClaims(blockStates []*cache.BlockState) map[string]cache.ShareInfo {
	claimShareMap := map[string]cache.ShareInfo{}

	for _, blockState := range blockStates {
		blockState.RLock()
		for claimId, demand := range blockState.Demands {
			// Important step: we discard demands that are not allocatable.
			// This boolean is computed by HasEnough, which allows negative budgets for RDP.
			if !demand.Availability {
				continue
			}

			// If this claim has been evaluated before, skip it.
			if _, ok := claimShareMap[claimId]; ok {
				continue
			}

			claimState := dpf.cache.GetClaim(claimId)
			share := claimState.UpdateDominantShare()
			if share.IsReadyToAllocate {
				claimShareMap[claimId] = share
			}

		}
		blockState.RUnlock()
	}

	return claimShareMap
}

func (dpf *DpfBatch) updateAffectedClaims(claimShareMap map[string]cache.ShareInfo, blockStates []string) {
	blockIdSet := map[string]bool{}
	for _, blockId := range blockStates {
		blockIdSet[blockId] = true
	}

	// update claim share map
	for claimId := range claimShareMap {
		claimState := dpf.cache.GetClaim(claimId)
		affected := false

		claimState.RLock()
		for blockId := range claimState.Demands {
			if blockIdSet[blockId] {
				affected = true
				break
			}
		}
		claimState.RUnlock()

		if !affected {
			continue
		}

		shareInfo := claimState.UpdateDominantShare()
		if shareInfo.IsReadyToAllocate {
			claimShareMap[claimId] = shareInfo
		} else {
			delete(claimShareMap, claimId)
		}
	}
}

// AllocateAvailableBudgets is the main loop of DPF
// Takes a batch of tasks + a state and tries to allocate everything
func (dpf *DpfBatch) AllocateAvailableBudgets(blockStates []*cache.BlockState) {

	// Compute the dominant shares
	claimShareMap := dpf.getInfluencedClaims(blockStates)
	for {
		if len(claimShareMap) == 0 {
			break
		}

		var minShareClaim *cache.ClaimState
		var minDominantShare = math.MaxFloat64
		var minShareInfo cache.ShareInfo

		// Browse claims to find the smallest and remove those which are impossible to allocate
		for claimId, shareInfo := range claimShareMap {
			if !shareInfo.IsReadyToAllocate {
				delete(claimShareMap, claimId) // Safe to delete
				continue
			}

			if shareInfo.DominantShare < minDominantShare {
				minDominantShare = shareInfo.DominantShare
				minShareClaim = dpf.cache.GetClaim(claimId)
				minShareInfo = shareInfo
			}
		}

		if minShareClaim == nil {
			break
		}

		// Pop the smallest claim, since we are going to allocate it
		// They will be tried in next iteration when more budget becomes available.
		delete(claimShareMap, minShareClaim.GetId())

		klog.Infof("ready to convert pending budget to acquired budget of claim [%s]", minShareClaim.GetId())
		if dpf.BatchAllocateP2(minShareClaim, minShareInfo.AvailableBlocks) {
			// notify exterior that this claim has been allocated
			dpf.p2Chan <- minShareClaim.GetId()
			dpf.updateAffectedClaims(claimShareMap, minShareInfo.AvailableBlocks)
		} else {
			klog.Infof("fail to convert pending budget of claim [%s]", minShareClaim.GetId())
		}
	}

}

func (dpf *DpfBatch) batchCompleteReserving(blockStates []*cache.BlockState, requestHandler framework.RequestHandler, allocatedBudgets map[string]columbiav1.BudgetAccount) {
	// add a failure response to the privacy budget claim
	err := dpf.updater.ApplyOperationToClaim(func(claim *columbiav1.PrivacyBudgetClaim) error {
		if len(claim.Status.Responses) > requestHandler.RequestIndex {
			return nil
		}

		appendResponse(claim, columbiav1.Response{
			Identifier: claim.Spec.Requests[requestHandler.RequestIndex].Identifier,
			State:      columbiav1.Pending,
			AllocateResponse: &columbiav1.AllocateResponse{
				Budgets:   allocatedBudgets,
				StartTime: dpf.timer.Now(),
			},
		})

		for blockId, account := range allocatedBudgets {
			reserved := claim.Status.ReservedBudgets[blockId].Add(account.Reserved)
			claim.Status.ReservedBudgets[blockId] = reserved
			// klog.Infof("New reserved budget map for [%s]:\b%s", blockId, reserved)
		}

		return nil
	}, requestHandler.ClaimHandler)

	if err == nil {
		klog.Infof("succeed to append a pending response to the privacy budget claim [%s]", requestHandler.ClaimHandler.GetId())
	}

	requestViewer := requestHandler.Viewer()

	done := make(chan bool)
	completeWrapper := func(blockState *cache.BlockState) {
		err = dpf.updater.ApplyOperationToDataBlock(func(block *columbiav1.PrivateDataBlock) error {
			return completeDpfReserving(requestViewer, block)
		}, blockState)

		if err != nil {
			klog.Infof("failed to complete the lock of [%s] on data block [%s]: %v",
				util.GetRequestId(requestViewer.Claim, requestViewer.RequestIndex), blockState.GetId(), err)
		}

		done <- true
	}

	counter := len(blockStates)
	for _, blockState := range blockStates {
		go completeWrapper(blockState)
	}

	for counter > 0 {
		<-done
		counter--
	}
}

// Gives the allocatedBudgets to the RequestHandler to update the claim
func (dpf *DpfBatch) batchCompleteAcquiring(blockStates []*cache.BlockState,
	requestHandler framework.RequestHandler,
	allocatedBudgets map[string]columbiav1.BudgetAccount) {

	// skip if the success has already been added
	err := dpf.updater.ApplyOperationToClaim(func(claim *columbiav1.PrivacyBudgetClaim) error {

		response := columbiav1.Response{
			Identifier: claim.Spec.Requests[requestHandler.RequestIndex].Identifier,
			State:      columbiav1.Success,
			AllocateResponse: &columbiav1.AllocateResponse{
				Budgets:    allocatedBudgets,
				FinishTime: dpf.timer.Now(),
			},
		}

		var isPhaseTwo bool

		if len(claim.Status.Responses) > requestHandler.RequestIndex {
			oldResponse := &claim.Status.Responses[requestHandler.RequestIndex]
			// Only update the response if it is Pending
			if oldResponse.State != columbiav1.Pending {
				return nil
			}
			// copy the start time to the new response
			response.AllocateResponse.StartTime = oldResponse.AllocateResponse.StartTime

			claim.Status.Responses[requestHandler.RequestIndex] = response
			isPhaseTwo = true
		} else {
			appendResponse(claim, response)
			isPhaseTwo = false
		}

		if isPhaseTwo {
			claim.Status.ReservedBudgets = map[string]columbiav1.PrivacyBudget{}
		}

		for blockId, account := range allocatedBudgets {
			if !account.Acquired.IsEmpty() {
				acquired := claim.Status.AcquiredBudgets[blockId].Add(account.Acquired)
				claim.Status.AcquiredBudgets[blockId] = acquired
			}
		}

		return nil
	}, requestHandler.ClaimHandler)

	if err == nil {
		klog.Infof("succeed to update a success response to the privacy budget claim [%s]", requestHandler.ClaimHandler.GetId())
	}

	requestViewer := requestHandler.Viewer()

	done := make(chan bool)
	completeWrapper := func(blockState *cache.BlockState) {
		err = dpf.updater.ApplyOperationToDataBlock(func(block *columbiav1.PrivateDataBlock) error {
			return completeDpfAcquiring(requestViewer, block)
		}, blockState)

		if err != nil {
			klog.Infof("failed to complete the lock of [%s] on data block [%s]: %v",
				util.GetRequestId(requestViewer.Claim, requestViewer.RequestIndex), blockState.GetId(), err)
		}

		done <- true
	}

	counter := len(blockStates)
	for _, blockState := range blockStates {
		go completeWrapper(blockState)
	}

	for counter > 0 {
		<-done
		counter--
	}
}

func completeAllocating(requestId string, block *columbiav1.PrivateDataBlock) error {
	lockedBudget := block.Status.LockedBudgetMap[requestId]
	if lockedBudget.State == columbiav1.Allocated {
		return LockExpired
	}

	lockedBudget.State = columbiav1.Allocated
	block.Status.LockedBudgetMap[requestId] = lockedBudget

	block.Status.AcquiredBudgetMap[requestId] = block.Status.AcquiredBudgetMap[requestId].Add(lockedBudget.Budget.Acquired)
	block.Status.ReservedBudgetMap[requestId] = block.Status.ReservedBudgetMap[requestId].Add(lockedBudget.Budget.Reserved)
	return nil
}

func completeCommitting(requestId string, block *columbiav1.PrivateDataBlock) error {
	lockedBudget := block.Status.LockedBudgetMap[requestId]
	if lockedBudget.State == columbiav1.Committed {
		return LockExpired
	}

	lockedBudget.State = columbiav1.Committed
	block.Status.LockedBudgetMap[requestId] = lockedBudget

	block.Status.CommittedBudgetMap[requestId] = block.Status.CommittedBudgetMap[requestId].
		Add(lockedBudget.Budget.Acquired).Add(lockedBudget.Budget.Reserved)
	return nil
}

func completeDpfAcquiring(requestViewer framework.RequestViewer, block *columbiav1.PrivateDataBlock) error {
	lockedBudget := block.Status.LockedBudgetMap[requestViewer.Id]
	if lockedBudget.State != columbiav1.DpfAcquiring {
		return LockExpired
	}

	lockedBudget.State = columbiav1.DpfAcquired
	block.Status.LockedBudgetMap[requestViewer.Id] = lockedBudget

	claimId := util.GetClaimId(requestViewer.Claim)
	// Some data blocks may not be acquired budgets
	if !lockedBudget.Budget.Acquired.IsEmpty() {
		block.Status.AcquiredBudgetMap[claimId] = block.Status.AcquiredBudgetMap[claimId].Add(lockedBudget.Budget.Acquired)
	}
	// Usually Reserved Budget is negative, meaning it converts from reserved to acquired
	reserved := block.Status.ReservedBudgetMap[claimId].Add(lockedBudget.Budget.Reserved)
	util.SetBudgetMapAndDeleteEmpty(block.Status.ReservedBudgetMap, claimId, reserved)
	return nil
}

func completeDpfReserving(requestViewer framework.RequestViewer, block *columbiav1.PrivateDataBlock) error {
	lockedBudget := block.Status.LockedBudgetMap[requestViewer.Id]
	if lockedBudget.State != columbiav1.DpfReserving {
		return LockExpired
	}

	lockedBudget.State = columbiav1.DpfReserved
	block.Status.LockedBudgetMap[requestViewer.Id] = lockedBudget

	claimId := util.GetClaimId(requestViewer.Claim)
	block.Status.ReservedBudgetMap[claimId] = block.Status.ReservedBudgetMap[claimId].Add(lockedBudget.Budget.Reserved)
	return nil
}
