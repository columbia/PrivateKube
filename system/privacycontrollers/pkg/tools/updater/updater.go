package updater

import (
	"columbia.github.com/sage/privacycontrollers/pkg/tools/util"
	columbiav1 "columbia.github.com/sage/privacyresource/pkg/apis/columbia.github.com/v1"
	"columbia.github.com/sage/privacyresource/pkg/framework"
	privacyclientset "columbia.github.com/sage/privacyresource/pkg/generated/clientset/versioned"
	"context"
	"fmt"
	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/klog"
	"strconv"
)

var (
	UnknownErr = fmt.Errorf("unknown error")
)

const (
	// error code from privacy resource client
	ConflictErrCode = 409
)

type ResourceUpdater struct {
	privacyResourceClient privacyclientset.Interface
}

func NewResourceUpdater(privacyResourceClient privacyclientset.Interface) *ResourceUpdater {
	return &ResourceUpdater{
		privacyResourceClient: privacyResourceClient,
	}
}

func (updater *ResourceUpdater) ApplyOperationToDataBlock(
	operation func(*columbiav1.PrivateDataBlock) error,
	blockHandler framework.BlockHandler) error {
	for true {
		block := snapshotBlockAfterCheckNull(blockHandler)
		if block == nil {
			return fmt.Errorf("block does not exist")
		}

		if err := operation(block); err != nil {
			return fmt.Errorf("failed to apply operation to data block [%s]: %v",
				block.Name, err)
		}

		newBlock, err := updater.privacyResourceClient.ColumbiaV1().PrivateDataBlocks(block.Namespace).UpdateStatus(context.TODO(), block, metav1.UpdateOptions{})
		if err == nil {
			_ = blockHandler.Update(newBlock)
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

		if err := operation(claim); err != nil {
			return fmt.Errorf("failed to apply operation to claim [%s]: %v", util.GetClaimId(claim), err)
		}

		newClaim, err := updater.privacyResourceClient.ColumbiaV1().PrivacyBudgetClaims(claim.Namespace).UpdateStatus(context.TODO(), claim, metav1.UpdateOptions{})
		if err == nil {
			_ = claimHandler.Update(newClaim)
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
	//TODO: temporary annotate, which should be taken care by controller
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

		newClaim, err := updater.privacyResourceClient.ColumbiaV1().PrivacyBudgetClaims(claim.Namespace).Update(context.TODO(), claim, metav1.UpdateOptions{})
		if err == nil {
			_ = claimHandler.Update(newClaim)
			klog.Infof("upload the response (id: %d) to the annotation", idx)
			return
		}

		if apiStatus, ok := err.(errors.APIStatus); !ok || apiStatus.Status().Code != ConflictErrCode {
			return
		}
	}

}

func snapshotBlockAfterCheckNull(blockHandler framework.BlockHandler) *columbiav1.PrivateDataBlock {
	if blockHandler == nil {
		return nil
	}
	return blockHandler.Snapshot()
}

func snapClaimAfterCheckNull(claimHandler framework.ClaimHandler) *columbiav1.PrivacyBudgetClaim {
	if claimHandler == nil {
		return nil
	}
	return claimHandler.Snapshot()
}
