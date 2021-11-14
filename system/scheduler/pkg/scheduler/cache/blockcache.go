package cache

import (
	"fmt"
	"sync"

	columbiav1 "columbia.github.com/privatekube/privacyresource/pkg/apis/columbia.github.com/v1"
	"columbia.github.com/privatekube/scheduler/pkg/scheduler/util"
	"k8s.io/klog"
)

type PrivateDataBlockCache struct {
	// This mutex guards all fields within this cache struct.
	lock sync.RWMutex
	// a set of assumed pod keys.
	// The key could further be used to get an entry in podStates.
	//assumedPods map[string]bool
	// a map from pod key to podState.
	//podStates map[string]*podState
	blocks  map[string]*BlockState
	counter *StreamingCounter
}

func NewPrivateDataBlockCache(counterOptions *StreamingCounterOptions) *PrivateDataBlockCache {
	var c *StreamingCounter
	if counterOptions == nil {
		c = nil
	} else {
		c = NewStreamingCounter(counterOptions)
	}
	cache := PrivateDataBlockCache{
		blocks:  make(map[string]*BlockState),
		counter: c,
	}

	return &cache
}

func (cache *PrivateDataBlockCache) Add(block *columbiav1.PrivateDataBlock) (*BlockState, error) {
	blockId := util.GetBlockId(block)

	blockDefaultInitialization(block)

	// If the counter is activated, try to update the count
	if cache.counter != nil {
		// We modify the counter and the block inplace
		cache.counter.UpdateCount(block)
	}

	if state, ok := cache.blocks[blockId]; ok {
		klog.Warningf("data block %s was already in cache and will be overriden", blockId)
		return state, state.Update(block)
	} else {
		state = NewBlockState(block)
		cache.blocks[blockId] = state
		return state, nil
	}

}

func (cache *PrivateDataBlockCache) Update(block *columbiav1.PrivateDataBlock) (*BlockState, error) {
	blockId := util.GetBlockId(block)
	blockDefaultInitialization(block)
	if state, ok := cache.blocks[blockId]; !ok {
		klog.Warningf("data block %s does not exist in cache, and a handler will be created", blockId)
		state = NewBlockState(block)
		cache.blocks[blockId] = state
		return state, nil
	} else {
		return state, state.Update(block)
	}
}

func (cache *PrivateDataBlockCache) UpdateReturnOld(block *columbiav1.PrivateDataBlock) (
	*BlockState, *columbiav1.PrivateDataBlock, error) {

	blockId := util.GetBlockId(block)
	blockDefaultInitialization(block)

	if state, ok := cache.blocks[blockId]; !ok {
		klog.Warningf("data block %s does not exist in cache, and a handler will be created", blockId)
		state = NewBlockState(block)
		cache.blocks[blockId] = state
		return state, nil, nil
	} else {
		old, err := state.UpdateReturnOld(block)
		return state, old, err
	}
}

func (cache *PrivateDataBlockCache) Delete(blockId string) error {
	handler := cache.blocks[blockId]
	if handler != nil {
		handler.Delete()
	} else {
		return fmt.Errorf("block [%v] does not exist in cache", blockId)
	}
	delete(cache.blocks, blockId)
	return nil
}

func (cache *PrivateDataBlockCache) Get(blockId string) *BlockState {
	return cache.blocks[blockId]
}

func (cache *PrivateDataBlockCache) GetBatch(blockNames []string) []*BlockState {
	blockStates := make([]*BlockState, 0, len(blockNames))
	for _, blockId := range blockNames {
		blockStates = append(blockStates, cache.blocks[blockId])
	}

	return blockStates
}

func (cache *PrivateDataBlockCache) All() []*BlockState {
	result := make([]*BlockState, len(cache.blocks))
	i := 0
	for _, blockState := range cache.blocks {
		result[i] = blockState
		i++
	}
	return result
}

func (cache *PrivateDataBlockCache) Filter(predicate func(*BlockState) bool) []*BlockState {
	result := make([]*BlockState, 0, 16)
	for _, blockState := range cache.blocks {
		if predicate(blockState) {
			result = append(result, blockState)
		}
	}
	return result
}

func GetIdOfBlockStates(blockHandlers []*BlockState) []string {
	ids := make([]string, len(blockHandlers))
	for i, blockHandler := range blockHandlers {
		ids[i] = blockHandler.GetId()
	}
	return ids
}

func blockDefaultInitialization(block *columbiav1.PrivateDataBlock) {
	if block.Status.AcquiredBudgetMap == nil {
		block.Status.AcquiredBudgetMap = map[string]columbiav1.PrivacyBudget{}
	}
	if block.Status.ReservedBudgetMap == nil {
		block.Status.ReservedBudgetMap = map[string]columbiav1.PrivacyBudget{}
	}
	if block.Status.CommittedBudgetMap == nil {
		block.Status.CommittedBudgetMap = map[string]columbiav1.PrivacyBudget{}
	}
	if block.Status.LockedBudgetMap == nil {
		block.Status.LockedBudgetMap = map[string]columbiav1.LockedBudget{}
	}

	if len(block.Status.AcquiredBudgetMap) == 0 && len(block.Status.CommittedBudgetMap) == 0 &&
		block.Status.AvailableBudget.IsEmpty() && block.Status.PendingBudget.IsEmpty() {
		block.Status.PendingBudget = block.Spec.InitialBudget
	}
}
