package algorithm

import (
	"fmt"
	"math"
	"sync"

	columbiav1 "columbia.github.com/privatekube/privacyresource/pkg/apis/columbia.github.com/v1"
	"columbia.github.com/privatekube/privacyresource/pkg/framework"
	"columbia.github.com/privatekube/scheduler/pkg/scheduler/cache"
	"columbia.github.com/privatekube/scheduler/pkg/scheduler/errors"
	"columbia.github.com/privatekube/scheduler/pkg/scheduler/timing"
	"columbia.github.com/privatekube/scheduler/pkg/scheduler/updater"
	"columbia.github.com/privatekube/scheduler/pkg/scheduler/util"
	"k8s.io/klog"
)

// Batch is a snapshot of the tasks waiting to be allocated + the set of blocks and their available budgets
type Batch struct {
	sync.Mutex
	core    CoreAlgorithm
	updater updater.ResourceUpdater
	cache   cache.Cache
	timer   timing.Timer

	// channel used in P2 Allocate to notify exterior about the allocation
	p2Chan chan<- string
}

func NewBatch(
	updater updater.ResourceUpdater,
	cache_ cache.Cache,
	p2Chan chan<- string,
) *Batch {

	return &Batch{
		core:    CoreAlgorithm{},
		updater: updater,
		cache:   cache_,
		p2Chan:  p2Chan,
	}
}

// BatchAllocateP2 tries to allocate a request
func (batch *Batch) BatchAllocateP2(claimState *cache.ClaimState, allocatingBlockIds []string) bool {
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
	blockStates := batch.cache.GetBlocks(blockIds)

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

			allocatedBudget, err = batch.core.AllocateP2(requestViewer, block, toAcquire)
			return err
		}

		err := batch.updater.ApplyOperationToDataBlock(allocateStrategy, blockState)

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
		return batch.BatchRollback(blockStates, requestHandler, errors.AllOrNothingError(nil).Error())
	}

	if successBlockCount < minNumOfBlocks {
		err := errors.NotEnoughDataBlock(minNumOfBlocks, successBlockCount)
		klog.Errorf("failed to allocate enough data blocks to request [%s]", requestHandler.GetId())
		// try to rollback all the qualified data blocks in case some of them were allocated by last
		// scheduling terminated accidentally
		return batch.BatchRollback(blockStates, requestHandler, err.Error())
	}

	// Now enters the second phase of 2PC.
	// A decision will be first updated in the claim's response
	batch.batchCompleteAcquiring(blockStates, requestHandler, allocatedBudgets)
	return true
}

func (batch *Batch) BatchAllocateTimeout(requestHandler framework.RequestHandler) bool {
	requestViewer := requestHandler.Viewer()
	if !requestViewer.IsPending() {
		return false
	}

	timeoutStrategy := func(block *columbiav1.PrivateDataBlock) error {
		return batch.core.AllocateTimeout(requestViewer, block)
	}

	blockIds := util.GetStringKeysFromMap(requestViewer.Claim.Status.ReservedBudgets)
	blockStates := batch.cache.GetBlocks(blockIds)

	done := make(chan bool)
	timeoutWrapper := func(blockState *cache.BlockState) {
		if err := batch.updater.ApplyOperationToDataBlock(timeoutStrategy, blockState); err == nil {
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
	batch.BatchRollback(blockStates, requestHandler, "timeout")
	return true
}

// getInfluencedClaims fetches all the claims that demand one of the blocks
// It computes the cost for each claim
func (batch *Batch) getInfluencedClaims(blockStates []*cache.BlockState) map[string]cache.CostInfo {
	claimCostMap := map[string]cache.CostInfo{}

	for _, blockState := range blockStates {
		blockState.RLock()
		for claimId, demand := range blockState.Demands {
			// Important step: we discard demands that are not allocatable.
			// This boolean is computed by HasEnough, which allows negative budgets for RDP.
			if !demand.Availability {
				continue
			}

			// If this claim has been evaluated before, skip it.
			if _, ok := claimCostMap[claimId]; ok {
				continue
			}

			claimState := batch.cache.GetClaim(claimId)
			cost := claimState.UpdateTotalCost()
			if cost.IsReadyToAllocate {
				claimCostMap[claimId] = cost
			}

		}
		blockState.RUnlock()
	}

	return claimCostMap
}

func (batch *Batch) updateAffectedClaims(claimCostMap map[string]cache.CostInfo, blockStates []string) {
	blockIdSet := map[string]bool{}
	for _, blockId := range blockStates {
		blockIdSet[blockId] = true
	}

	// update claim cost map
	for claimId := range claimCostMap {
		claimState := batch.cache.GetClaim(claimId)
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

		costInfo := claimState.UpdateTotalCost()
		if costInfo.IsReadyToAllocate {
			claimCostMap[claimId] = costInfo
		} else {
			delete(claimCostMap, claimId)
		}
	}
}

// AllocateAvailableBudgets is the main loop of Scheduling
// Takes a batch of tasks + a state and tries to allocate everything
func (batch *Batch) AllocateAvailableBudgets(blockStates []*cache.BlockState) {

	// Compute the costs
	claimCostMap := batch.getInfluencedClaims(blockStates)
	for {
		if len(claimCostMap) == 0 {
			break
		}

		var minCostClaim *cache.ClaimState
		var minCost = math.MaxFloat64
		var minCostInfo cache.CostInfo

		// Browse claims to find the smallest and remove those which are impossible to allocate
		for claimId, costInfo := range claimCostMap {
			if !costInfo.IsReadyToAllocate {
				delete(claimCostMap, claimId) // Safe to delete
				continue
			}

			if costInfo.Cost < minCost {
				minCost = costInfo.Cost
				minCostClaim = batch.cache.GetClaim(claimId)
				minCostInfo = costInfo
			}
		}

		if minCostClaim == nil {
			break
		}

		// Pop the smallest claim, since we are going to allocate it
		// They will be tried in next iteration when more budget becomes available.
		delete(claimCostMap, minCostClaim.GetId())

		klog.Infof("ready to convert pending budget to acquired budget of claim [%s]", minCostClaim.GetId())
		if batch.BatchAllocateP2(minCostClaim, minCostInfo.AvailableBlocks) {
			// notify exterior that this claim has been allocated
			batch.p2Chan <- minCostClaim.GetId()
			batch.updateAffectedClaims(claimCostMap, minCostInfo.AvailableBlocks)
		} else {
			klog.Infof("fail to convert pending budget of claim [%s]", minCostClaim.GetId())
		}
	}

}

func (batch *Batch) batchCompleteReserving(blockStates []*cache.BlockState, requestHandler framework.RequestHandler, allocatedBudgets map[string]columbiav1.BudgetAccount) {
	// add a failure response to the privacy budget claim
	err := batch.updater.ApplyOperationToClaim(func(claim *columbiav1.PrivacyBudgetClaim) error {
		if len(claim.Status.Responses) > requestHandler.RequestIndex {
			return nil
		}

		appendResponse(claim, columbiav1.Response{
			Identifier: claim.Spec.Requests[requestHandler.RequestIndex].Identifier,
			State:      columbiav1.Pending,
			AllocateResponse: &columbiav1.AllocateResponse{
				Budgets:   allocatedBudgets,
				StartTime: batch.timer.Now(),
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
		err = batch.updater.ApplyOperationToDataBlock(func(block *columbiav1.PrivateDataBlock) error {
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
func (batch *Batch) batchCompleteAcquiring(blockStates []*cache.BlockState,
	requestHandler framework.RequestHandler,
	allocatedBudgets map[string]columbiav1.BudgetAccount) {

	// skip if the success has already been added
	err := batch.updater.ApplyOperationToClaim(func(claim *columbiav1.PrivacyBudgetClaim) error {

		response := columbiav1.Response{
			Identifier: claim.Spec.Requests[requestHandler.RequestIndex].Identifier,
			State:      columbiav1.Success,
			AllocateResponse: &columbiav1.AllocateResponse{
				Budgets:    allocatedBudgets,
				FinishTime: batch.timer.Now(),
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
		err = batch.updater.ApplyOperationToDataBlock(func(block *columbiav1.PrivateDataBlock) error {
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
