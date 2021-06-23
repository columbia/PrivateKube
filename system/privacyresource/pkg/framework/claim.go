package framework

import (
	"fmt"
	"sync"

	columbiav1 "columbia.github.com/privatekube/privacyresource/pkg/apis/columbia.github.com/v1"
)

type ClaimHandler interface {
	Snapshot() *columbiav1.PrivacyBudgetClaim

	View() *columbiav1.PrivacyBudgetClaim

	Update(claim *columbiav1.PrivacyBudgetClaim) error

	Delete()

	IsLive() bool

	GetId() string

	NextRequest() RequestHandler
}

type ClaimHandlerImpl struct {
	claim *columbiav1.PrivacyBudgetClaim
	lock  sync.RWMutex
}

func NewClaimHandler(claim *columbiav1.PrivacyBudgetClaim) *ClaimHandlerImpl {
	claimDefaultInitialization(claim)
	return &ClaimHandlerImpl{claim: claim}
}

func (handler *ClaimHandlerImpl) Snapshot() *columbiav1.PrivacyBudgetClaim {
	handler.lock.RLock()
	defer handler.lock.RUnlock()
	if handler.claim == nil {
		return nil
	}
	return handler.claim.DeepCopy()
}

func (handler *ClaimHandlerImpl) View() *columbiav1.PrivacyBudgetClaim {
	handler.lock.RLock()
	defer handler.lock.RUnlock()
	return handler.claim
}

func (handler *ClaimHandlerImpl) Update(claim *columbiav1.PrivacyBudgetClaim) error {
	handler.lock.Lock()
	defer handler.lock.Unlock()
	if claim.Generation < handler.claim.Generation {
		return fmt.Errorf("privacy budget claim [%v] to update is staled", claim.Name)
	}
	claimDefaultInitialization(claim)
	handler.claim = claim
	return nil
}

func (handler *ClaimHandlerImpl) Delete() {
	handler.lock.Lock()
	defer handler.lock.Unlock()
	handler.claim = nil
}

func (handler *ClaimHandlerImpl) IsLive() bool {
	handler.lock.RLock()
	defer handler.lock.RUnlock()
	return handler.claim != nil
}

func (handler *ClaimHandlerImpl) GetId() string {
	handler.lock.RLock()
	defer handler.lock.RUnlock()
	if handler.claim == nil {
		return ""
	}
	return handler.claim.Namespace + "/" + handler.claim.Name
}

func (handler *ClaimHandlerImpl) NextRequest() RequestHandler {
	handler.lock.RLock()
	defer handler.lock.RUnlock()
	if handler.claim == nil {
		return RequestHandler{handler, 0}
	}

	nextIndex := len(handler.claim.Status.Responses) - 1
	if nextIndex == -1 || handler.claim.Status.Responses[nextIndex].State != columbiav1.Pending {
		nextIndex++
	}

	return RequestHandler{handler, nextIndex}
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
			if isEmptyBudget(claim.Spec.Requests[i].AllocateRequest.MinBudget) {
				claim.Spec.Requests[i].AllocateRequest.MinBudget.Constant = &columbiav1.PrivacyBudget{
					EpsDel: &columbiav1.EpsilonDeltaBudget{
						Epsilon: 0,
						Delta:   0,
					},
				}
			}
			if isEmptyBudget(claim.Spec.Requests[i].AllocateRequest.ExpectedBudget) {
				claim.Spec.Requests[i].AllocateRequest.ExpectedBudget.Constant = &columbiav1.PrivacyBudget{
					EpsDel: &columbiav1.EpsilonDeltaBudget{
						Epsilon: 0,
						Delta:   0,
					},
				}
			}
		}
	}
}

func isEmptyBudget(budgetRequest columbiav1.BudgetRequest) bool {
	return budgetRequest.Constant == nil && budgetRequest.Function == "" && budgetRequest.BudgetMap == nil
}
