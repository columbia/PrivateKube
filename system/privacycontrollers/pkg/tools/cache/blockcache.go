package cache

import (
	"fmt"
	"sync"

	"columbia.github.com/privatekube/privacycontrollers/pkg/tools/util"
	columbiav1 "columbia.github.com/privatekube/privacyresource/pkg/apis/columbia.github.com/v1"
	"columbia.github.com/privatekube/privacyresource/pkg/framework"
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
	blocks map[string]framework.BlockHandler
}

func NewPrivateDataBlockCache() *PrivateDataBlockCache {
	cache := PrivateDataBlockCache{}
	cache.blocks = make(map[string]framework.BlockHandler)
	return &cache
}

func (cache *PrivateDataBlockCache) Add(block *columbiav1.PrivateDataBlock) (framework.BlockHandler, error) {
	cache.lock.Lock()
	defer cache.lock.Unlock()

	blockId := util.GetBlockId(block)
	if handler, ok := cache.blocks[blockId]; ok {
		klog.Warningf("data block %s has already in cache, which will be override", blockId)
		return handler, handler.Update(block)
	} else {
		handler = framework.NewBlockHandler(block)
		cache.blocks[blockId] = handler
		return handler, nil
	}
}

func (cache *PrivateDataBlockCache) Update(block *columbiav1.PrivateDataBlock) (framework.BlockHandler, error) {
	cache.lock.Lock()
	defer cache.lock.Unlock()

	blockId := util.GetBlockId(block)
	if handler, ok := cache.blocks[blockId]; !ok {
		klog.Warningf("data block %s does not exist in cache, and a handler will be created", blockId)
		handler = framework.NewBlockHandler(block)
		cache.blocks[blockId] = handler
		return handler, nil
	} else {
		return handler, handler.Update(block)
	}
}

func (cache *PrivateDataBlockCache) Delete(blockId string) error {
	cache.lock.Lock()
	defer cache.lock.Unlock()
	handler := cache.blocks[blockId]
	if handler != nil {
		handler.Delete()
	} else {
		return fmt.Errorf("block [%v] does not exist in cache", blockId)
	}
	delete(cache.blocks, blockId)
	return nil
}

func (cache *PrivateDataBlockCache) Get(blockId string) framework.BlockHandler {
	cache.lock.RLock()
	defer cache.lock.RUnlock()
	return cache.blocks[blockId]
}

func (cache *PrivateDataBlockCache) All() []framework.BlockHandler {
	cache.lock.RLock()
	defer cache.lock.RUnlock()
	result := make([]framework.BlockHandler, 0, len(cache.blocks))
	for _, blockInfo := range cache.blocks {
		result = append(result, blockInfo)
	}
	return result
}

func (cache *PrivateDataBlockCache) Filter(predicate func(framework.BlockHandler) bool) []framework.BlockHandler {
	cache.lock.RLock()
	defer cache.lock.RUnlock()
	result := make([]framework.BlockHandler, 0, 16)
	for _, blockInfo := range cache.blocks {
		if predicate(blockInfo) {
			result = append(result, blockInfo)
		}
	}
	return result
}
