package scheduler

import (
	columbiav1 "columbia.github.com/privatekube/privacyresource/pkg/apis/columbia.github.com/v1"
	"columbia.github.com/privatekube/privacyresource/pkg/framework"
	"columbia.github.com/privatekube/scheduler/pkg/scheduler/util"
	"k8s.io/client-go/tools/cache"
	"k8s.io/klog"
)

func (scheduler *Scheduler) addBlock(obj interface{}) {
	block, ok := obj.(*columbiav1.PrivateDataBlock)
	if !ok {
		klog.Errorf("cannot convert to *v1.PrivacyDataBlock: %v", obj)
		return
	}

	blockState, err := scheduler.cache.AddBlock(block)
	if err != nil {
		klog.Errorf("scheduler cache AddBlock failed: %v", err)
		return
	}

	// A hack to ensure that the block's new state is not overwritten
	refresh := func(block *columbiav1.PrivateDataBlock) error {
		return nil
	}
	scheduler.updater.ApplyOperationToDataBlock(refresh, blockState)

	klog.Infof("get a new block %s", util.GetBlockId(block))
}

func (scheduler *Scheduler) updateBlock(oldObj, newObj interface{}) {
	newBlock, ok := newObj.(*columbiav1.PrivateDataBlock)
	if !ok {
		klog.Errorf("cannot convert newObj to *v1.Request: %v", newObj)
		return
	}

	var err error
	if _, err = scheduler.cache.UpdateBlock(newBlock); err != nil {
		klog.Errorf("scheduler cache UpdateBlock failed: %v", err)
		return
	}

	klog.Infof("Data Block [%s] has been updated", util.GetBlockId(newBlock))

}

func (scheduler *Scheduler) deleteBlock(obj interface{}) {
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
	if err := scheduler.cache.DeleteBlock(blockId); err != nil {
		klog.Errorf("scheduler cache RemoveNode failed: %v", err)
	}

	klog.Infof("Private Data Block [%s] has been deleted", blockId)
}

func (scheduler *Scheduler) addClaim(obj interface{}) {
	claim, ok := obj.(*columbiav1.PrivacyBudgetClaim)
	if !ok {
		klog.Errorf("cannot convert to *PrivacyBudgetClaim: %v", obj)
		return
	}

	var err error
	var claimHandler framework.ClaimHandler

	if claimHandler, err = scheduler.cache.AddClaim(claim); err != nil {
		klog.Errorf("scheduler cache AddPod failed: %v", err)
	}

	scheduler.enqueueClaim(claimHandler)
	klog.Infof("Privacy Budget Claim [%s] is added!", util.GetClaimId(claim))
}

func (scheduler *Scheduler) updateClaim(oldObj, newObj interface{}) {
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

	if claimHandler, err = scheduler.cache.UpdateClaim(newClaim); err != nil {
		klog.Errorf("scheduler cache update Privacy Budget Claim failed: %v", err)
	}

	scheduler.enqueueClaim(claimHandler)
	klog.Infof("Privacy Budget Claim [%v] has been updated! ", util.GetClaimId(newClaim))
}

func (scheduler *Scheduler) deleteClaim(obj interface{}) {
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
	scheduler.dequeueClaim(claim)

	if err := scheduler.cache.DeleteClaim(claimId); err != nil {
		klog.Errorf("failed to delete claim from cache: %v", err)
		return
	}

	klog.Infof("Privacy Budget Claim [%v] has been deleted!", claimId)
}

func (scheduler *Scheduler) enqueueClaim(handler framework.ClaimHandler) {
	request := handler.NextRequest()
	if !request.IsValid() {
		return
	}

	_ = scheduler.schedulingQueue.Add(handler)
}

func (scheduler *Scheduler) dequeueClaim(claim *columbiav1.PrivacyBudgetClaim) {
	_ = scheduler.schedulingQueue.Delete(util.GetClaimId(claim))
}
