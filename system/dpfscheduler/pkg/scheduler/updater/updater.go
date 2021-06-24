package updater

import (
	"context"
	"fmt"
	"strconv"

	schedulercache "columbia.github.com/privatekube/dpfscheduler/pkg/scheduler/cache"
	schedulingerrors "columbia.github.com/privatekube/dpfscheduler/pkg/scheduler/errors"
	"columbia.github.com/privatekube/dpfscheduler/pkg/scheduler/util"
	columbiav1 "columbia.github.com/privatekube/privacyresource/pkg/apis/columbia.github.com/v1"
	"columbia.github.com/privatekube/privacyresource/pkg/framework"
	privacyclientset "columbia.github.com/privatekube/privacyresource/pkg/generated/clientset/versioned"
	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/klog"
)

var (
	UnknownErr = fmt.Errorf("unknown error")
)

const (
	// error code from privacy resource client
	ConflictErrCode = 409

	// Max Request in concurrency
	MaxRequests = 32
)

type ResourceUpdater struct {
	privacyResourceClient privacyclientset.Interface
	cache                 schedulercache.Cache
	sem                   chan struct{}
}

func NewResourceUpdater(
	privacyResourceClient privacyclientset.Interface,
	cache schedulercache.Cache,
) *ResourceUpdater {

	return &ResourceUpdater{
		privacyResourceClient: privacyResourceClient,
		cache:                 cache,
		sem:                   make(chan struct{}, MaxRequests),
	}
}

func (updater *ResourceUpdater) ApplyOperationToDataBlock(
	operation func(*columbiav1.PrivateDataBlock) error,
	blockHandler framework.BlockHandler) error {
	for true {
		block := blockHandler.Snapshot()
		if block == nil {
			return fmt.Errorf("block does not exist")
		}

		err := operation(block)

		if err == schedulingerrors.NothingChanged() {
			return nil
		}

		if err != nil {
			return fmt.Errorf("failed to apply operation to data block [%s]: %v",
				block.Name, err)
		}

		updater.sem <- struct{}{}
		newBlock, err := updater.privacyResourceClient.ColumbiaV1().PrivateDataBlocks(block.Namespace).UpdateStatus(context.TODO(), block, metav1.UpdateOptions{})
		<-updater.sem

		if err == nil {
			_, _ = updater.cache.UpdateBlock(newBlock)
			return nil
		}

		if apiStatus, ok := err.(errors.APIStatus); !ok || apiStatus.Status().Code != ConflictErrCode {
			return fmt.Errorf("unable to update data block: %v", err)
		}
	}
	return UnknownErr
}

func (updater *ResourceUpdater) ApplyOperationToClaim(
	operation func(*columbiav1.PrivacyBudgetClaim) error, claimHandler framework.ClaimHandler) error {

	for true {
		claim := snapClaimAfterCheckNull(claimHandler)
		if claim == nil {
			return fmt.Errorf("claim does not exist")
		}

		err := operation(claim)

		if err == schedulingerrors.NothingChanged() {
			return nil
		}

		if err != nil {
			return fmt.Errorf("failed to apply operation to claim [%s]: %v", util.GetClaimId(claim), err)
		}

		updater.sem <- struct{}{}
		newClaim, err := updater.privacyResourceClient.ColumbiaV1().PrivacyBudgetClaims(claim.Namespace).UpdateStatus(context.TODO(), claim, metav1.UpdateOptions{})
		<-updater.sem
		if err == nil {
			_, err = updater.cache.UpdateClaim(newClaim)
			updater.AddResponseToAnnotation(claimHandler)
			return nil
		}

		if apiStatus, ok := err.(errors.APIStatus); !ok || apiStatus.Status().Code != ConflictErrCode {
			return fmt.Errorf("unable to update privacy budget claim: %v", err)
		}
	}
	return UnknownErr
}

func (updater *ResourceUpdater) AddResponseToAnnotation(claimHandler framework.ClaimHandler) {
	// Temporary annotation -- should be handled by the controller
	for true {
		claim := snapClaimAfterCheckNull(claimHandler)
		if claim == nil {
			return
		}

		idx := len(claim.Status.Responses) - 1
		if idx == -1 || claim.Status.Responses[idx].Identifier == "" {
			return
		}
		if claim.Annotations == nil {
			claim.Annotations = map[string]string{}
		}
		claim.Annotations["request_"+claim.Status.Responses[idx].Identifier] = strconv.Itoa(idx)

		updater.sem <- struct{}{}
		newClaim, err := updater.privacyResourceClient.ColumbiaV1().PrivacyBudgetClaims(claim.Namespace).Update(context.TODO(), claim, metav1.UpdateOptions{})
		<-updater.sem

		if err == nil {
			_, _ = updater.cache.UpdateClaim(newClaim)
			klog.Infof("upload the response (id: %d) to the annotation", idx)
			return
		}

		if apiStatus, ok := err.(errors.APIStatus); !ok || apiStatus.Status().Code != ConflictErrCode {
			return
		}
	}

}

func snapClaimAfterCheckNull(claimHandler framework.ClaimHandler) *columbiav1.PrivacyBudgetClaim {
	if claimHandler == nil {
		return nil
	}
	return claimHandler.Snapshot()
}
