package blockclaimsync

import (
	"context"
	"fmt"
	"strconv"
	"strings"
	"time"

	privacycache "columbia.github.com/privatekube/privacycontrollers/pkg/tools/cache"
	"columbia.github.com/privatekube/privacycontrollers/pkg/tools/updater"
	"columbia.github.com/privatekube/privacycontrollers/pkg/tools/util"
	columbiav1 "columbia.github.com/privatekube/privacyresource/pkg/apis/columbia.github.com/v1"
	privacyclientset "columbia.github.com/privatekube/privacyresource/pkg/generated/clientset/versioned"
	informer "columbia.github.com/privatekube/privacyresource/pkg/generated/informers/externalversions/columbia.github.com/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/client-go/tools/cache"
	"k8s.io/klog"
)

type SyncController struct {
	privacyResourceClient      privacyclientset.Interface
	privateDataBlockInformer   informer.PrivateDataBlockInformer
	privacyBudgetClaimInformer informer.PrivacyBudgetClaimInformer
	informersHaveSynced        func() bool
	cache                      privacycache.Cache
	updater                    updater.ResourceUpdater
	stop                       chan struct{}
	SyncControllerOption
}

type SyncControllerOption struct {
	PeriodSecond int
}

func NewSynController(privacyClient privacyclientset.Interface,
	privateDataBlockInformer informer.PrivateDataBlockInformer,
	privacyBudgetClaimInformer informer.PrivacyBudgetClaimInformer,
	option SyncControllerOption) *SyncController {

	controller := SyncController{
		privacyResourceClient:      privacyClient,
		privateDataBlockInformer:   privateDataBlockInformer,
		privacyBudgetClaimInformer: privacyBudgetClaimInformer,
		informersHaveSynced: func() bool {
			return privateDataBlockInformer.Informer().HasSynced() && privacyBudgetClaimInformer.Informer().HasSynced()
		},
		cache:                privacycache.NewCache(),
		updater:              *updater.NewResourceUpdater(privacyClient),
		stop:                 make(chan struct{}),
		SyncControllerOption: option,
	}

	controller.addAllEventHandlers()
	return &controller
}

func (controller *SyncController) Run() {
	if !cache.WaitForCacheSync(controller.stop, controller.informersHaveSynced) {
		return
	}
	go wait.Until(controller.BatchCleanExpiredLocks, time.Duration(controller.PeriodSecond)*time.Second, controller.stop)
}

func (controller *SyncController) Close() {
	close(controller.stop)
}

func (controller *SyncController) cleanExpiredLocks(block *columbiav1.PrivateDataBlock) bool {
	if block.Status.LockedBudgetMap == nil {
		return false
	}

	isModified := false
	for requestId, lockedBudget := range block.Status.LockedBudgetMap {
		if controller.isRequestCompleted(requestId) {
			// in Go, it is safe to delete an entry when iterating with range.
			if isLockedBudgetExpired(&lockedBudget) {
				delete(block.Status.LockedBudgetMap, requestId)
				isModified = true
			}
		}
	}

	return isModified
}

// Clean the expired locks in block's status
func (controller *SyncController) BatchCleanExpiredLocks() {
	for _, blockHandler := range controller.cache.AllBlocks() {
		err := controller.updater.ApplyOperationToDataBlock(func(block *columbiav1.PrivateDataBlock) error {
			if controller.cleanExpiredLocks(block) {
				return nil
			}
			return fmt.Errorf("there is no expired locks to clean")
		}, blockHandler)

		if err == nil {
			klog.Infof("Expired locks in block [%s] has been cleaned.", blockHandler.GetId())
		}
	}
}

func (controller *SyncController) updateBlockStatus(block *columbiav1.PrivateDataBlock) error {
	_, err := controller.privacyResourceClient.ColumbiaV1().PrivateDataBlocks(block.Namespace).UpdateStatus(context.TODO(), block, metav1.UpdateOptions{})
	return err
}

func (controller *SyncController) isRequestCompleted(requestId string) bool {
	claimId, requestIndex := extractClaimIdAndIndex(requestId)
	claimHandler := controller.cache.GetClaim(claimId)
	if claimHandler == nil {
		return true
	}

	claim := claimHandler.View()
	if claim == nil {
		return true
	}

	if claim.Status.Responses == nil {
		return false
	}

	return len(claim.Status.Responses) > requestIndex && claim.Status.Responses[requestIndex].State != columbiav1.Pending
}

func extractClaimIdAndIndex(requestId string) (string, int) {
	splitRequest := strings.Split(requestId, "/")
	requestIndex, _ := strconv.Atoi(splitRequest[2])
	return splitRequest[0] + "/" + splitRequest[1], requestIndex
}

func isLockedBudgetExpired(lockedBudget *columbiav1.LockedBudget) bool {
	return lockedBudget.State == columbiav1.Allocated || lockedBudget.State == columbiav1.DpfAcquired ||
		lockedBudget.State == columbiav1.Committed
}

func rollbackLockedBudget(block *columbiav1.PrivateDataBlock, requestId string) {
	state := block.Status.LockedBudgetMap[requestId].State

	switch state {
	case columbiav1.Allocating, columbiav1.DpfAcquiring, columbiav1.DpfReserving:
		rollbackAllocating(block, requestId)
	case columbiav1.Committed:
		rollbackCommitted(block, requestId)
	}
	delete(block.Status.LockedBudgetMap, requestId)
}

func rollbackAllocating(block *columbiav1.PrivateDataBlock, requestId string) {
	lockedBudget := block.Status.LockedBudgetMap[requestId]
	block.Status.AvailableBudget.IAdd(lockedBudget.Budget.Acquired)
}

func rollbackCommitted(block *columbiav1.PrivateDataBlock, requestId string) {
	lockedBudget := block.Status.LockedBudgetMap[requestId]
	claimId, _ := extractClaimIdAndIndex(requestId)

	committedBudget := block.Status.CommittedBudgetMap[claimId]
	util.SetBudgetMapAndDeleteEmpty(block.Status.CommittedBudgetMap, claimId,
		committedBudget.Minus(lockedBudget.Budget.Acquired).Minus(lockedBudget.Budget.Reserved))

	acquiredBudget := block.Status.AcquiredBudgetMap[claimId]
	block.Status.AcquiredBudgetMap[claimId] = acquiredBudget.Add(lockedBudget.Budget.Acquired)

	reservedBudget := block.Status.ReservedBudgetMap[claimId]
	block.Status.ReservedBudgetMap[claimId] = reservedBudget.Add(lockedBudget.Budget.Reserved)
	block.Status.AvailableBudget.IAdd(lockedBudget.Budget.Reserved)

}
