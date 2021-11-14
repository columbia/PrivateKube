package algorithm

import (
	"columbia.github.com/privatekube/dpfscheduler/pkg/scheduler/util"
	columbiav1 "columbia.github.com/privatekube/privacyresource/pkg/apis/columbia.github.com/v1"
	"columbia.github.com/privatekube/privacyresource/pkg/framework"
	"k8s.io/klog"
)

type CoreAlgorithm struct {
}

func (core *CoreAlgorithm) Consume(requestViewer framework.RequestViewer, block *columbiav1.PrivateDataBlock) (result columbiav1.BudgetErrorPair) {
	// check if this data block has been handled before
	committed, commitResult := ReadLock(requestViewer, block)
	if committed {
		klog.Infof("succeeded to read the log of request [%s] from data block [%s]",
			util.GetRequestId(requestViewer.Claim, requestViewer.RequestIndex), util.GetBlockId(block))

		// the committed budget = commit acquired + commit reserved
		result.Budget.IAdd(commitResult.Budget.Acquired)
		result.Budget.IAdd(commitResult.Budget.Reserved)
		return
	}

	consumeRequest := requestViewer.View().ConsumeRequest
	consumingBudget := consumeRequest.Consume[util.GetBlockId(block)]
	if consumingBudget.IsNegative() {
		result.Error = "the budget to consume is negative"
		return
	}

	// crate a policy that strictly commits the budget provided below
	policy := ConsumePolicy{
		ConsumeAvailableBudget: true,
	}

	consumedBudget, _ := core.consumeAcquiredBudget(requestViewer, block, consumingBudget, policy)
	result.Budget = consumedBudget
	return
}

func (core *CoreAlgorithm) consumeAcquiredBudget(requestViewer framework.RequestViewer,
	block *columbiav1.PrivateDataBlock,
	budget columbiav1.PrivacyBudget,
	policy ConsumePolicy) (columbiav1.PrivacyBudget, error) {

	claimId := util.GetClaimId(requestViewer.Claim)
	acquiredBudget, ok := block.Status.AcquiredBudgetMap[claimId]
	if !ok {
		return columbiav1.PrivacyBudget{}, ClaimNotAcquired
	}

	// make sure budget is not negative
	budget = budget.NonNegative()

	if !acquiredBudget.HasEnough(budget) {
		if policy.ConsumeAvailableBudget {
			budget = budget.Intersect(acquiredBudget)
		} else {
			return columbiav1.PrivacyBudget{}, NotEnoughBudgetToCommit
		}
	}

	if budget.IsEmpty() {
		return columbiav1.PrivacyBudget{}, nil
	}

	acquiredBudget.ISub(budget)
	util.SetBudgetMapAndDeleteEmpty(block.Status.AcquiredBudgetMap, claimId, acquiredBudget)

	committedBudget := block.Status.CommittedBudgetMap[claimId]
	block.Status.CommittedBudgetMap[claimId] = committedBudget.Add(budget)

	requestId := util.GetRequestId(requestViewer.Claim, requestViewer.RequestIndex)
	addLockedBudget(block, requestId, columbiav1.LockedBudget{
		Budget: columbiav1.BudgetAccount{Acquired: budget},
		State:  columbiav1.Committed,
	})
	return budget, nil
}

func (core *CoreAlgorithm) Release(requestViewer framework.RequestViewer, block *columbiav1.PrivateDataBlock) (result columbiav1.BudgetErrorPair) {
	// check if this data block has been handled before
	aborted, abortResult := ReadLock(requestViewer, block)
	if aborted {
		klog.Infof("succeeded to read the log of request [%s] from data block [%s]",
			util.GetRequestId(requestViewer.Claim, requestViewer.RequestIndex), util.GetBlockId(block))

		result.Budget = abortResult.Budget.Acquired
		return result
	}

	releaseRequest := requestViewer.View().ReleaseRequest
	releasingBudget := releaseRequest.Release[util.GetBlockId(block)]

	policy := ParseReleasePolicy(releaseRequest.Policy)

	// then abort acquired budget
	releasedAcquiredBudget, _ := releaseAcquiredBudget(requestViewer, block, releasingBudget, policy)

	// if user specify a strict policy to release as it specifies, we need to check if it is fulfilled.
	if !policy.ReleaseAvailableBudget && !releasingBudget.IsNegative() &&
		!releasedAcquiredBudget.Equal(releasingBudget) {
		result.Error = NotEnoughBudgetToAbort.Error()
		return
	}

	result.Budget = releasedAcquiredBudget
	return result
}

func releaseAcquiredBudget(requestViewer framework.RequestViewer,
	block *columbiav1.PrivateDataBlock,
	budget columbiav1.PrivacyBudget,
	policy ReleasePolicy) (columbiav1.PrivacyBudget, error) {

	claimId := util.GetClaimId(requestViewer.Claim)
	acquiredBudget, ok := block.Status.AcquiredBudgetMap[claimId]
	if !ok {
		return columbiav1.PrivacyBudget{}, ClaimNotAcquired
	}

	// negative budget indicates to commit whatever available
	if budget.IsNegative() {
		budget = acquiredBudget.Negative()
	} else {
		// make sure there is no partial negative
		budget = budget.NonNegative()
	}

	// if acquired budget does not have enough budget to abort
	// the abort amount may be adjusted or this request will be terminated, depending on the policy
	if !acquiredBudget.HasEnough(budget) {
		if policy.ReleaseAvailableBudget {
			budget = budget.Intersect(acquiredBudget)
		} else {
			return columbiav1.PrivacyBudget{}, NotEnoughBudgetToCommit
		}
	}

	if budget.IsEmpty() {
		return columbiav1.PrivacyBudget{}, nil
	}

	// subtract the aborting budget
	acquiredBudget.ISub(budget)
	util.SetBudgetMapAndDeleteEmpty(block.Status.AcquiredBudgetMap, claimId, acquiredBudget)

	if block.Spec.BudgetReturnPolicy == columbiav1.ToAvailable {
		block.Status.AvailableBudget.IAdd(budget)
	} else {
		block.Status.PendingBudget.IAdd(budget)
	}

	addLockedBudget(block, requestViewer.Id, columbiav1.LockedBudget{
		Budget: columbiav1.BudgetAccount{Acquired: budget},
		State:  columbiav1.Aborted,
	})
	return budget, nil
}

func ReadLock(requestViewer framework.RequestViewer, block *columbiav1.PrivateDataBlock) (bool, columbiav1.LockedBudget) {
	requestId := requestViewer.Id
	lockedBudget, ok := block.Status.LockedBudgetMap[requestId]
	return ok, lockedBudget
}
