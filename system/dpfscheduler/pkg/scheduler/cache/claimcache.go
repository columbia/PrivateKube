package cache

import (
	"fmt"

	columbiav1 "columbia.github.com/privatekube/privacyresource/pkg/apis/columbia.github.com/v1"
	"k8s.io/klog"
)

type ClaimCache struct {
	claims map[string]*ClaimState
}

func NewClaimCache() *ClaimCache {
	claimCache := ClaimCache{}
	claimCache.claims = make(map[string]*ClaimState)
	return &claimCache
}

func (claimCache *ClaimCache) Add(claim *columbiav1.PrivacyBudgetClaim) (*ClaimState, error) {
	claimId := getClaimId(claim)
	claimDefaultInitialization(claim)

	if state, ok := claimCache.claims[claimId]; ok {
		klog.Warningf("claim %s has already in claim cache, which will be override", claimId)
		return state, state.Update(claim)
	} else {
		state := NewClaimState(claim)
		claimCache.claims[claimId] = state
		return state, nil
	}
}

func (claimCache *ClaimCache) Update(claim *columbiav1.PrivacyBudgetClaim) (*ClaimState, error) {
	claimId := getClaimId(claim)
	claimDefaultInitialization(claim)

	if state, ok := claimCache.claims[claimId]; !ok {
		klog.Warningf("pod %s does not exist in pod claimCache, and a handler will be created", claimId)
		state = NewClaimState(claim)
		claimCache.claims[claimId] = state
		return state, nil
	} else {
		return state, state.Update(claim)
	}
}

func (claimCache *ClaimCache) Delete(claimId string) error {
	handler := claimCache.claims[claimId]
	if handler != nil {
		handler.Delete()
	} else {
		return fmt.Errorf("claim [%v] does not exist in claim cache", claimId)
	}
	delete(claimCache.claims, claimId)
	return nil
}

func (claimCache *ClaimCache) Get(claimId string) *ClaimState {
	return claimCache.claims[claimId]
}

func getClaimId(claim *columbiav1.PrivacyBudgetClaim) string {
	return claim.Namespace + "/" + claim.Name
}

func claimDefaultInitialization(claim *columbiav1.PrivacyBudgetClaim) {
	if claim.Status.AcquiredBudgets == nil {
		claim.Status.AcquiredBudgets = map[string]columbiav1.PrivacyBudget{}
	}
	if claim.Status.ReservedBudgets == nil {
		claim.Status.ReservedBudgets = map[string]columbiav1.PrivacyBudget{}
	}
	if claim.Status.CommittedBudgets == nil {
		claim.Status.CommittedBudgets = map[string]columbiav1.PrivacyBudget{}
	}
	if claim.Status.Responses == nil {
		claim.Status.Responses = []columbiav1.Response{}
	}
	if claim.Spec.Requests == nil {
		claim.Spec.Requests = []columbiav1.Request{}
	}

	for i := 0; i < len(claim.Spec.Requests); i++ {
		if claim.Spec.Requests[i].AllocateRequest != nil {
			if isEmptyBudget(claim.Spec.Requests[i].AllocateRequest.ExpectedBudget) {
				claim.Spec.Requests[i].AllocateRequest.ExpectedBudget.Constant = &columbiav1.PrivacyBudget{}
			}
		}
	}
}

func isEmptyBudget(budgetRequest columbiav1.BudgetRequest) bool {
	return budgetRequest.Constant == nil && budgetRequest.Function == "" && budgetRequest.BudgetMap == nil
}
