package scheduler

import (
	"columbia.github.com/sage/dpfscheduler/pkg/scheduler/util"
	columbiav1 "columbia.github.com/sage/privacyresource/pkg/apis/columbia.github.com/v1"
	"columbia.github.com/sage/privacyresource/pkg/framework"
	"k8s.io/client-go/tools/cache"
	"k8s.io/klog"
)

func (dpfScheduler *DpfScheduler) addBlock(obj interface{}) {
	block, ok := obj.(*columbiav1.PrivateDataBlock)
	if !ok {
		klog.Errorf("cannot convert to *v1.PrivacyDataBlock: %v", obj)
		return
	}

	var err error
	if _, err = dpfScheduler.cache.AddBlock(block); err != nil {
		klog.Errorf("scheduler cache AddBlock failed: %v", err)
		return
	}

	klog.Infof("get a new block %s", util.GetBlockId(block))
}

func (dpfScheduler *DpfScheduler) updateBlock(oldObj, newObj interface{}) {
	newBlock, ok := newObj.(*columbiav1.PrivateDataBlock)
	if !ok {
		klog.Errorf("cannot convert newObj to *v1.Request: %v", newObj)
		return
	}

	var err error
	if _, err = dpfScheduler.cache.UpdateBlock(newBlock); err != nil {
		klog.Errorf("scheduler cache UpdateBlock failed: %v", err)
		return
	}

	klog.Infof("Data Block [%s] has been updated", util.GetBlockId(newBlock))
}

func (dpfScheduler *DpfScheduler) deleteBlock(obj interface{}) {
	var block *columbiav1.PrivateDataBlock
	switch t := obj.(type) {
	case *columbiav1.PrivateDataBlock:
		block = t
	case cache.DeletedFinalStateUnknown:
		var ok bool
		block, ok = t.Obj.(*columbiav1.PrivateDataBlock)
		if !ok {
			klog.Errorf("cannot convert to *PrivateDataBlock: %v", t.Obj)
			return
		}
	default:
		klog.Errorf("cannot convert to *PrivateDataBlock: %v", t)
		return
	}
	blockId := util.GetBlockId(block)
	if err := dpfScheduler.cache.DeleteBlock(blockId); err != nil {
		klog.Errorf("scheduler cache RemoveNode failed: %v", err)
	}

	klog.Infof("Private Data Block [%s] has been deleted", blockId)
}

func (dpfScheduler *DpfScheduler) addClaim(obj interface{}) {
	claim, ok := obj.(*columbiav1.PrivacyBudgetClaim)
	if !ok {
		klog.Errorf("cannot convert to *PrivacyBudgetClaim: %v", obj)
		return
	}

	var err error
	var claimHandler framework.ClaimHandler

	if claimHandler, err = dpfScheduler.cache.AddClaim(claim); err != nil {
		klog.Errorf("scheduler cache AddPod failed: %v", err)
	}

	dpfScheduler.enqueueClaim(claimHandler)
	klog.Infof("Privacy Budget Claim [%s] is added!", util.GetClaimId(claim))
}

func (dpfScheduler *DpfScheduler) updateClaim(oldObj, newObj interface{}) {
	_, ok := oldObj.(*columbiav1.PrivacyBudgetClaim)
	if !ok {
		klog.Errorf("cannot convert oldObj to *PrivacyBudgetClaim: %v", oldObj)
		return
	}

	newClaim, ok := newObj.(*columbiav1.PrivacyBudgetClaim)
	if !ok {
		klog.Errorf("cannot convert newObj to *PrivacyBudgetClaim: %v", newObj)
		return
	}

	var err error
	var claimHandler framework.ClaimHandler

	if claimHandler, err = dpfScheduler.cache.UpdateClaim(newClaim); err != nil {
		klog.Errorf("scheduler cache update Privacy Budget Claim failed: %v", err)
	}

	dpfScheduler.enqueueClaim(claimHandler)
	klog.Infof("Privacy Budget Claim [%v] has been updated! ", util.GetClaimId(newClaim))
}

func (dpfScheduler *DpfScheduler) deleteClaim(obj interface{}) {
	var claim *columbiav1.PrivacyBudgetClaim
	var claimId string
	switch t := obj.(type) {
	case *columbiav1.PrivacyBudgetClaim:
		claim = t
	case cache.DeletedFinalStateUnknown:
		var ok bool
		claim, ok = t.Obj.(*columbiav1.PrivacyBudgetClaim)
		if !ok {
			klog.Errorf("cannot convert to *PrivacyBudgetClaim: %v", t.Obj)
			return
		}
	default:
		klog.Errorf("cannot convert to *PrivacyBudgetClaim: %v", t)
		return
	}

	claimId = util.GetClaimId(claim)
	dpfScheduler.dequeueClaim(claim)

	if err := dpfScheduler.cache.DeleteClaim(claimId); err != nil {
		klog.Errorf("failed to delete claim from cache: %v", err)
		return
	}

	klog.Infof("Privacy Budget Claim [%v] has been deleted!", claimId)
}

func (dpfScheduler *DpfScheduler) enqueueClaim(handler framework.ClaimHandler) {
	request := handler.NextRequest()
	if !request.IsValid() {
		return
	}

	_ = dpfScheduler.schedulingQueue.Add(handler)
}

func (dpfScheduler *DpfScheduler) dequeueClaim(claim *columbiav1.PrivacyBudgetClaim) {
	_ = dpfScheduler.schedulingQueue.Delete(util.GetClaimId(claim))
}
