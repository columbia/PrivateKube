package cache

import (
	columbiav1 "columbia.github.com/privatekube/privacyresource/pkg/apis/columbia.github.com/v1"
	"columbia.github.com/privatekube/privacyresource/pkg/framework"
)

type Cache interface {
	AddClaim(claim *columbiav1.PrivacyBudgetClaim) (framework.ClaimHandler, error)

	UpdateClaim(claim *columbiav1.PrivacyBudgetClaim) (framework.ClaimHandler, error)

	DeleteClaim(claimName string) error

	GetClaim(claimName string) framework.ClaimHandler

	AddBlock(block *columbiav1.PrivateDataBlock) (framework.BlockHandler, error)

	UpdateBlock(block *columbiav1.PrivateDataBlock) (framework.BlockHandler, error)

	DeleteBlock(blockName string) error

	GetBlock(blockName string) framework.BlockHandler

	AllBlocks() []framework.BlockHandler

	FilterBlock(predicate func(framework.BlockHandler) bool) []framework.BlockHandler
}

type schedulerCache struct {
	claimCache ClaimCache
	blockCache PrivateDataBlockCache
}

func (cache *schedulerCache) AddClaim(claim *columbiav1.PrivacyBudgetClaim) (framework.ClaimHandler, error) {
	return cache.claimCache.Add(claim)
}

func (cache *schedulerCache) UpdateClaim(claim *columbiav1.PrivacyBudgetClaim) (framework.ClaimHandler, error) {
	return cache.claimCache.Update(claim)
}

func (cache *schedulerCache) DeleteClaim(claimName string) error {
	return cache.claimCache.Delete(claimName)
}

func (cache *schedulerCache) GetClaim(claimName string) framework.ClaimHandler {
	return cache.claimCache.Get(claimName)
}

func (cache *schedulerCache) AddBlock(block *columbiav1.PrivateDataBlock) (framework.BlockHandler, error) {
	return cache.blockCache.Add(block)
}

func (cache *schedulerCache) UpdateBlock(block *columbiav1.PrivateDataBlock) (framework.BlockHandler, error) {
	return cache.blockCache.Update(block)
}

func (cache *schedulerCache) DeleteBlock(blockName string) error {
	return cache.blockCache.Delete(blockName)
}

func (cache *schedulerCache) GetBlock(blockName string) framework.BlockHandler {
	return cache.blockCache.Get(blockName)
}

func (cache *schedulerCache) AllBlocks() []framework.BlockHandler {
	return cache.blockCache.All()
}

func (cache *schedulerCache) FilterBlock(predicate func(framework.BlockHandler) bool) []framework.BlockHandler {
	return cache.blockCache.Filter(predicate)
}

func NewCache() Cache {
	cache := schedulerCache{
		claimCache: *NewClaimCache(),
		blockCache: *NewPrivateDataBlockCache(),
	}
	return &cache
}
