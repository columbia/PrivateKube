package cache

import (
	"fmt"
	"sync"

	columbiav1 "columbia.github.com/privatekube/privacyresource/pkg/apis/columbia.github.com/v1"
	"columbia.github.com/privatekube/privacyresource/pkg/framework"
	"github.com/wangjohn/quickselect"
)

type ClaimState struct {
	sync.RWMutex
	claim     *columbiav1.PrivacyBudgetClaim
	nextIndex int
	// Lock before accessing the demand field
	Demands map[string]*DemandState
	id      string
}

// CostInfo is a helper type bound to a claim. It records information on whether a claim is ready to
// be converted reserved budget
type CostInfo struct {
	Cost              float64
	AvailableBlocks   []string
	IsReadyToAllocate bool
}

func NewClaimState(claim *columbiav1.PrivacyBudgetClaim) *ClaimState {
	claimState := &ClaimState{
		claim:     claim,
		nextIndex: 0,
		Demands:   map[string]*DemandState{},
		id:        getClaimId(claim),
	}
	claimState.syncNextRequest()
	return claimState
}

type blockCostPair struct {
	cost    float64
	blockId string
}

type blockCostSlice []blockCostPair

func (slice blockCostSlice) Len() int {
	return len(slice)
}

func (slice blockCostSlice) Less(i, j int) bool {
	return slice[i].cost < slice[j].cost
}

func (slice blockCostSlice) Swap(i, j int) {
	slice[i], slice[j] = slice[j], slice[i]
}

func (claimState *ClaimState) UpdateTotalCost() (result CostInfo) {
	claimState.Lock()
	defer claimState.Unlock()

	if !claimState.hasPendingRequest() {
		return
	}

	// maxN is the max number of blocks
	// minN is the min number of blocks
	pendingRequest := claimState.claim.Spec.Requests[claimState.nextIndex].AllocateRequest
	maxN := pendingRequest.MaxNumberOfBlocks

	if maxN == 0 {
		maxN = len(claimState.Demands)
	}

	pairs := make(blockCostSlice, 0, len(claimState.Demands))
	for blockId, demand := range claimState.Demands {
		if !demand.Availability {
			continue
		}
		pairs = append(pairs, blockCostPair{demand.cost, blockId})
	}

	// the reason of len(pairs) < maxN is that the scheduler will try to allocate all the data blocks up to maxN.
	// If the current condition does not have maxN data blocks available to allocated to a claim, then the scheduler will
	// not schedule this claim.
	if len(pairs) < maxN {
		result.IsReadyToAllocate = false
		return
	}
	result.IsReadyToAllocate = true

	// else case will be maxN == len(pairs). In this case, no need to select the first maxN.
	if maxN < len(pairs) {
		_ = quickselect.QuickSelect(pairs, maxN)
	}

	result.Cost = 0
	result.AvailableBlocks = make([]string, 0, maxN)
	for _, pair := range pairs[:maxN] {
		result.Cost += pair.cost
		result.AvailableBlocks = append(result.AvailableBlocks, pair.blockId)
	}

	return
}

// Below: bunch of CRUD utils

func (claimState *ClaimState) Snapshot() *columbiav1.PrivacyBudgetClaim {
	claimState.RLock()
	defer claimState.RUnlock()
	if claimState.claim == nil {
		return nil
	}
	return claimState.claim.DeepCopy()
}

func (claimState *ClaimState) View() *columbiav1.PrivacyBudgetClaim {
	claimState.RLock()
	defer claimState.RUnlock()
	return claimState.claim
}

func (claimState *ClaimState) Update(claim *columbiav1.PrivacyBudgetClaim) error {
	claimState.Lock()
	defer claimState.Unlock()
	if claim.Generation < claimState.claim.Generation {
		return fmt.Errorf("privacy budget claim [%v] to update is staled", claim.Name)
	}

	claimState.claim = claim
	claimState.syncNextRequest()
	return nil
}

func (claimState *ClaimState) UpdateDemand(blockId string, demand *DemandState) {
	claimState.Lock()
	defer claimState.Unlock()
	claimState.Demands[blockId] = demand

}

func (claimState *ClaimState) Delete() {
	claimState.Lock()
	defer claimState.Unlock()
	claimState.claim = nil
}

func (claimState *ClaimState) IsLive() bool {
	claimState.RLock()
	defer claimState.RUnlock()
	return claimState.claim != nil
}

func (claimState *ClaimState) GetId() string {
	return claimState.id
}

func (claimState *ClaimState) NextRequest() framework.RequestHandler {
	claimState.RLock()
	defer claimState.RUnlock()

	return framework.RequestHandler{
		ClaimHandler: claimState,
		RequestIndex: claimState.nextIndex,
	}
}

func (claimState *ClaimState) syncNextRequest() {
	if claimState.claim == nil {
		claimState.nextIndex = 1
		return
	}

	nextIndex := len(claimState.claim.Status.Responses) - 1
	if nextIndex == -1 || claimState.claim.Status.Responses[nextIndex].State != columbiav1.Pending {
		nextIndex++
	}
	claimState.nextIndex = nextIndex
}

func (claimState *ClaimState) hasPendingRequest() bool {
	return claimState.nextIndex < len(claimState.claim.Status.Responses) &&
		claimState.claim.Status.Responses[claimState.nextIndex].State == columbiav1.Pending
}