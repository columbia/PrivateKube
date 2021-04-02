package cache

import (
	columbiav1 "columbia.github.com/sage/privacyresource/pkg/apis/columbia.github.com/v1"
	"columbia.github.com/sage/privacyresource/pkg/framework"
	"fmt"
	"k8s.io/klog"
	"sync"
)

type ClaimCache struct {
	//ttl    time.Duration
	//period time.Duration

	// This mutex guards all fields within this cache struct.
	lock   sync.RWMutex
	claims map[string]framework.ClaimHandler
}

func NewClaimCache() *ClaimCache {
	claimCache := ClaimCache{}
	claimCache.claims = make(map[string]framework.ClaimHandler)
	return &claimCache
}

func (claimCache *ClaimCache) Add(claim *columbiav1.PrivacyBudgetClaim) (framework.ClaimHandler, error) {
	claimCache.lock.Lock()
	defer claimCache.lock.Unlock()

	claimId := getClaimId(claim)
	if handler, ok := claimCache.claims[claimId]; ok {
		klog.Warningf("claim %s has already in claim cache, which will be override", claimId)
		return handler, handler.Update(claim)
	} else {
		handler := framework.NewClaimHandler(claim)
		claimCache.claims[claimId] = handler
		return handler, nil
	}
}

func (claimCache *ClaimCache) Update(claim *columbiav1.PrivacyBudgetClaim) (framework.ClaimHandler, error) {
	claimCache.lock.Lock()
	defer claimCache.lock.Unlock()

	claimId := getClaimId(claim)
	if handler, ok := claimCache.claims[claimId]; !ok {
		klog.Warningf("pod %s does not exist in pod claimCache, and a handler will be created", claimId)
		handler = framework.NewClaimHandler(claim)
		claimCache.claims[claimId] = handler
		return handler, nil
	} else {
		return handler, handler.Update(claim)
	}
}

func (claimCache *ClaimCache) Delete(claimId string) error {
	claimCache.lock.Lock()
	defer claimCache.lock.Unlock()
	handler := claimCache.claims[claimId]
	if handler != nil {
		handler.Delete()
	} else {
		return fmt.Errorf("claim [%v] does not exist in claim cache", claimId)
	}
	delete(claimCache.claims, claimId)
	return nil
}

func (claimCache *ClaimCache) Get(claimId string) framework.ClaimHandler {
	claimCache.lock.RLock()
	defer claimCache.lock.RUnlock()
	return claimCache.claims[claimId]
}

func getClaimId(claim *columbiav1.PrivacyBudgetClaim) string {
	return claim.Namespace + "/" + claim.Name
}
