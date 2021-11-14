package algorithm

import (
	"fmt"

	"columbia.github.com/privatekube/dpfscheduler/pkg/scheduler/errors"
	"columbia.github.com/privatekube/dpfscheduler/pkg/scheduler/util"
	columbiav1 "columbia.github.com/privatekube/privacyresource/pkg/apis/columbia.github.com/v1"
	"columbia.github.com/privatekube/privacyresource/pkg/framework"
	"k8s.io/klog"
)

var (
	CommitExceedAllocated   = fmt.Errorf("try to commit more than the claim allocated")
	NotEnoughBudgetToCommit = fmt.Errorf("the private data block has not enough budget to commit")
	NotEnoughBudgetToAbort  = fmt.Errorf("the private data block has not enough budget to abort")
	ClaimNotAcquired        = fmt.Errorf("the privacy budget claim does not acquire this private data block")
	ReservationAborted      = fmt.Errorf("the reservation has been aborted")
	UpdateErr               = fmt.Errorf("failed to update because the object is expired")
	ClaimNotExist           = fmt.Errorf("the privacy budget claim does not exist any more")
	DataBlockNotExist       = fmt.Errorf("the data block does not exist any more")
	LockExpired             = fmt.Errorf("the locked budget has completed")
)

// AllocateP1 unlocks the budget when a new task arrives
// and the task tries to Acquire or Reserve the budget
func (core *CoreAlgorithm) AllocateP1(requestViewer framework.RequestViewer, block *columbiav1.PrivateDataBlock,
	releasingBudget columbiav1.PrivacyBudget, mode int) (columbiav1.BudgetAccount, error) {
	if block == nil {
		return columbiav1.BudgetAccount{}, DataBlockNotExist
	}

	// check if this data block has been handled before
	allocated, allocatedResult := ReadLock(requestViewer, block)
	if allocated {
		klog.Infof("succeeded to read the locked budget of request [%s] from data block [%s]",
			util.GetRequestId(requestViewer.Claim, requestViewer.RequestIndex), util.GetBlockId(block))
		return allocatedResult.Budget, nil
	}

	releasingBudget = releasingBudget.Intersect(block.Status.PendingBudget)
	if !releasingBudget.IsEmpty() {
		block.Status.PendingBudget.ISub(releasingBudget)
		block.Status.AvailableBudget.IAdd(releasingBudget)
	}

	allocateRequest := requestViewer.View().AllocateRequest
	expectedBudget := interpretBudgetRequest(allocateRequest.ExpectedBudget, block)
	expectedBudget = expectedBudget.NonNegative()

	if mode == ToAcquire {
		core.p1AcquireBudget(requestViewer, block, expectedBudget)
		return columbiav1.BudgetAccount{Acquired: expectedBudget}, nil
	} else if mode == ToReserve {
		core.p1ReserveBudget(requestViewer, block, expectedBudget)
		return columbiav1.BudgetAccount{Reserved: expectedBudget}, nil
	} else {

		// Don't have to update the block if budget is released
		if releasingBudget.IsEmpty() {
			return columbiav1.BudgetAccount{}, errors.NothingChanged()
		}

		return columbiav1.BudgetAccount{}, nil
	}
}

// AllocateP2 allocates some unlocked budget to a pipeline
// If there is not enough budget, it simply returns zero allocated budget (no error)
func (core *CoreAlgorithm) AllocateP2(requestViewer framework.RequestViewer, block *columbiav1.PrivateDataBlock,
	toAcquire bool) (columbiav1.BudgetAccount, error) {
	if block == nil {
		return columbiav1.BudgetAccount{}, DataBlockNotExist
	}

	// check if this data block has been handled before
	allocating, allocatingResult := ReadLock(requestViewer, block)
	if allocating && allocatingResult.State == columbiav1.DpfAcquiring {
		klog.Infof("succeeded to read the locked budget of request [%s] from data block [%s]",
			util.GetRequestId(requestViewer.Claim, requestViewer.RequestIndex), util.GetBlockId(block))
		return allocatingResult.Budget, nil
	}

	var acquiringBudget columbiav1.PrivacyBudget
	if toAcquire {
		acquiringBudget = core.p2AcquireBudget(requestViewer, block)
	} else {
		core.p2RollbackReserved(requestViewer, block)
	}

	return columbiav1.BudgetAccount{
		Acquired: acquiringBudget,
	}, nil
}

func (core *CoreAlgorithm) AllocateTimeout(requestViewer framework.RequestViewer, block *columbiav1.PrivateDataBlock) error {
	if block == nil {
		return DataBlockNotExist
	}

	// check if this data block has been handled before
	allocating, allocatingResult := ReadLock(requestViewer, block)
	if allocating && allocatingResult.State == columbiav1.DpfReserving {
		klog.Infof("succeeded to read the locked budget of request [%s] from data block [%s]",
			util.GetRequestId(requestViewer.Claim, requestViewer.RequestIndex), util.GetBlockId(block))
		return nil
	}

	core.timeoutPending(requestViewer, block)
	return nil
}

func (core *CoreAlgorithm) p1AcquireBudget(requestViewer framework.RequestViewer,
	block *columbiav1.PrivateDataBlock,
	acquiringBudget columbiav1.PrivacyBudget) columbiav1.PrivacyBudget {

	if acquiringBudget.IsEpsDelType() {
		acquiringBudget = block.Status.AvailableBudget.Intersect(acquiringBudget)
	}
	if acquiringBudget.IsRenyiType() {
		if !block.Status.AvailableBudget.HasEnough(acquiringBudget) {
			klog.Infof("Claim can't be allocated from Block [%s], returning 0 budget.", block.GetName())
			acquiringBudget.ISub(acquiringBudget)
		}

	}
	// subtract from available budget
	// NOTE: in the Renyi case, some orders of the Available budget can become negative (but not all).
	block.Status.AvailableBudget.ISub(acquiringBudget)

	// add locked budget
	requestId := util.GetRequestId(requestViewer.Claim, requestViewer.RequestIndex)
	block.Status.LockedBudgetMap[requestId] = columbiav1.LockedBudget{
		Budget: columbiav1.BudgetAccount{Acquired: acquiringBudget},
		// P2 means get the acquired budget
		State: columbiav1.DpfAcquiring,
	}
	return acquiringBudget
}

func (core *CoreAlgorithm) p1ReserveBudget(requestViewer framework.RequestViewer,
	block *columbiav1.PrivateDataBlock,
	budget columbiav1.PrivacyBudget) columbiav1.PrivacyBudget {

	// add locked budget
	addLockedBudget(block, requestViewer.Id, columbiav1.LockedBudget{
		Budget: columbiav1.BudgetAccount{Reserved: budget},
		State:  columbiav1.DpfReserving,
	})

	return budget
}

// p2AcquireBudget subtracts the budget acquired by the claim from the block's available budget
func (core *CoreAlgorithm) p2AcquireBudget(requestViewer framework.RequestViewer,
	block *columbiav1.PrivateDataBlock) columbiav1.PrivacyBudget {

	claimId := util.GetClaimId(requestViewer.Claim)
	// The reserved budget map stores the demand of the claim (we could store it eslw)
	acquiringBudget := block.Status.ReservedBudgetMap[claimId]

	if acquiringBudget.IsEpsDelType() {
		// Intersect computes the min
		acquiringBudget = block.Status.AvailableBudget.Intersect(acquiringBudget)
	}
	if acquiringBudget.IsRenyiType() {
		// Instead, if one alpha is positive, we would like to return the true demand
		// NOTE: this is Improvement 4
		if !block.Status.AvailableBudget.HasEnough(acquiringBudget) {
			// Return an empty budget
			klog.Infof("Claim [%s] can't be allocated from Block [%s], returning 0 budget.", claimId, block.GetName())
			acquiringBudget.ISub(acquiringBudget)
		}

	}

	// klog.Infof("Allocating %s from %s", block.Status.AvailableBudget, acquiringBudget)

	// subtract from available budget
	// With RDP, acquiringBudget can be greater than the available budget on some alphas,
	// so AvailableBudget can be negative
	block.Status.AvailableBudget.ISub(acquiringBudget)

	// add locked budget
	block.Status.LockedBudgetMap[requestViewer.Id] = columbiav1.LockedBudget{
		Budget: columbiav1.BudgetAccount{
			Acquired: acquiringBudget,
			Reserved: acquiringBudget.Negative(),
		},
		State: columbiav1.DpfAcquiring,
	}
	return acquiringBudget
}

func (core *CoreAlgorithm) p2RollbackReserved(requestViewer framework.RequestViewer,
	block *columbiav1.PrivateDataBlock) {

	claimId := util.GetClaimId(requestViewer.Claim)
	reservedBudget := block.Status.ReservedBudgetMap[claimId]

	// add locked budget
	block.Status.LockedBudgetMap[requestViewer.Id] = columbiav1.LockedBudget{
		Budget: columbiav1.BudgetAccount{
			Reserved: reservedBudget.Negative(),
		},
		State: columbiav1.DpfAcquiring,
	}
}

func (core *CoreAlgorithm) timeoutPending(requestViewer framework.RequestViewer,
	block *columbiav1.PrivateDataBlock) {

	claimId := util.GetClaimId(requestViewer.Claim)
	reservedBudget := block.Status.ReservedBudgetMap[claimId]
	delete(block.Status.ReservedBudgetMap, claimId)

	// add locked budget
	block.Status.LockedBudgetMap[requestViewer.Id] = columbiav1.LockedBudget{
		Budget: columbiav1.BudgetAccount{
			Reserved: reservedBudget,
		},
		State: columbiav1.DpfReserving,
	}
}

func addLockedBudget(block *columbiav1.PrivateDataBlock, requestId string, lockingBudget columbiav1.LockedBudget) {
	if lockedBudget, ok := block.Status.LockedBudgetMap[requestId]; ok {
		lockedBudget.Budget.Acquired.IAdd(lockingBudget.Budget.Acquired)
		lockedBudget.Budget.Reserved.IAdd(lockingBudget.Budget.Reserved)
		block.Status.LockedBudgetMap[requestId] = lockedBudget
	} else {
		block.Status.LockedBudgetMap[requestId] = lockingBudget
	}
}
