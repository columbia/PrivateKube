package cache

import (
	"sync"

	columbiav1 "columbia.github.com/sage/privacyresource/pkg/apis/columbia.github.com/v1"
)

type Cache interface {
	AddClaim(claim *columbiav1.PrivacyBudgetClaim) (*ClaimState, error)

	UpdateClaim(claim *columbiav1.PrivacyBudgetClaim) (*ClaimState, error)

	DeleteClaim(claimName string) error

	GetClaim(claimName string) *ClaimState

	AddBlock(block *columbiav1.PrivateDataBlock) (*BlockState, error)

	UpdateBlock(block *columbiav1.PrivateDataBlock) (*BlockState, error)

	DeleteBlock(blockName string) error

	GetBlock(blockName string) *BlockState

	GetBlocks(blockName []string) []*BlockState

	AllBlocks() []*BlockState

	FilterBlock(predicate func(*BlockState) bool) []*BlockState
}

type StateCache struct {
	// This mutex guards
	claimLock sync.RWMutex
	blockLock sync.RWMutex

	claimCache ClaimCache
	blockCache PrivateDataBlockCache
}

func NewStateCache() *StateCache {
	return &StateCache{
		claimCache: *NewClaimCache(),
		blockCache: *NewPrivateDataBlockCache(),
	}
}

func (cache *StateCache) AddClaim(claim *columbiav1.PrivacyBudgetClaim) (*ClaimState, error) {
	cache.claimLock.Lock()
	claimState, err := cache.claimCache.Add(claim)
	cache.claimLock.Unlock()

	cache.syncDemandsByClaim(claimState)

	return claimState, err
}

func (cache *StateCache) UpdateClaim(claim *columbiav1.PrivacyBudgetClaim) (*ClaimState, error) {
	cache.claimLock.Lock()
	defer cache.claimLock.Unlock()
	return cache.claimCache.Update(claim)
}

func (cache *StateCache) DeleteClaim(claimName string) error {
	cache.claimLock.Lock()
	defer cache.claimLock.Unlock()
	return cache.claimCache.Delete(claimName)
}

func (cache *StateCache) GetClaim(claimName string) *ClaimState {
	cache.claimLock.RLock()
	defer cache.claimLock.RUnlock()
	return cache.claimCache.Get(claimName)
}

func (cache *StateCache) AddBlock(block *columbiav1.PrivateDataBlock) (*BlockState, error) {
	cache.blockLock.Lock()
	defer cache.blockLock.Unlock()

	blockState, err := cache.blockCache.Add(block)

	cache.updateDemands(blockState)

	return blockState, err
}

func (cache *StateCache) UpdateBlock(block *columbiav1.PrivateDataBlock) (*BlockState, error) {
	cache.blockLock.Lock()
	blockState, oldBlock, err := cache.blockCache.UpdateReturnOld(block)
	cache.blockLock.Unlock()

	if diffDemandRelatedFields(block, oldBlock) {
		cache.updateDemands(blockState)
	}

	return blockState, err
}

func (cache *StateCache) DeleteBlock(blockName string) error {
	cache.blockLock.Lock()
	defer cache.blockLock.Unlock()
	return cache.blockCache.Delete(blockName)
}

func (cache *StateCache) GetBlock(blockName string) *BlockState {
	cache.blockLock.RLock()
	defer cache.blockLock.RUnlock()
	return cache.blockCache.Get(blockName)
}

func (cache *StateCache) GetBlocks(blockName []string) []*BlockState {
	cache.blockLock.RLock()
	defer cache.blockLock.RUnlock()
	return cache.blockCache.GetBatch(blockName)
}

func (cache *StateCache) AllBlocks() []*BlockState {
	cache.blockLock.RLock()
	defer cache.blockLock.RUnlock()
	return cache.blockCache.All()
}

func (cache *StateCache) FilterBlock(predicate func(*BlockState) bool) []*BlockState {
	cache.blockLock.RLock()
	defer cache.blockLock.RUnlock()
	return cache.blockCache.Filter(predicate)
}

func (cache *StateCache) updateDemands(blockState *BlockState) {
	cache.claimLock.RLock()
	defer cache.claimLock.RUnlock()

	blockId := blockState.GetId()
	for claimId, demand := range blockState.UpdateDemandMap() {
		claimState := cache.claimCache.Get(claimId)
		if claimState == nil {
			continue
		}
		claimState.UpdateDemand(blockId, demand)
	}
}

func (cache *StateCache) syncDemandsByClaim(claimState *ClaimState) {
	claimState.Lock()
	defer claimState.Unlock()

	cache.blockLock.RLock()
	defer cache.blockLock.RUnlock()

	claimId := claimState.GetId()
	for blockId := range claimState.claim.Status.ReservedBudgets {
		blockState := cache.blockCache.Get(blockId)
		if blockState == nil {
			continue
		}

		blockState.RLock()
		demand := blockState.Demands[claimId]
		blockState.RUnlock()

		if demand != nil {
			claimState.Demands[blockId] = demand
		}

	}
}

// true - same, false - different
func diffDemandRelatedFields(a, b *columbiav1.PrivateDataBlock) bool {
	if !a.Status.AvailableBudget.Equal(b.Status.AvailableBudget) ||
		len(a.Status.ReservedBudgetMap) != len(b.Status.ReservedBudgetMap) {
		return true
	}

	for claimId, budget := range a.Status.ReservedBudgetMap {
		otherBudget, ok := b.Status.ReservedBudgetMap[claimId]
		if !ok || !otherBudget.Equal(budget) {
			return true
		}
	}

	return false
}
