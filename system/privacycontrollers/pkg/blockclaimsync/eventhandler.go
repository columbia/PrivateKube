package blockclaimsync

import (
	"columbia.github.com/sage/privacycontrollers/pkg/tools/util"
	columbiav1 "columbia.github.com/sage/privacyresource/pkg/apis/columbia.github.com/v1"
	"fmt"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/client-go/tools/cache"
	"k8s.io/klog"
)

func (controller *SyncController) addBlock(obj interface{}) {
	block, ok := obj.(*columbiav1.PrivateDataBlock)
	if !ok {
		klog.Errorf("cannot convert to *v1.PrivacyDataBlock: %v", obj)
		return
	}

	_, _ = controller.cache.AddBlock(block)
	klog.Infof("successfully add data block [%s]", util.GetBlockId(block))
}

func (controller *SyncController) updateBlock(oldObj, newObj interface{}) {
	newBlock, ok := newObj.(*columbiav1.PrivateDataBlock)
	if !ok {
		klog.Errorf("cannot convert newObj to *v1.PrivacyDataBlock: %v", newObj)
		return
	}

	_, _ = controller.cache.UpdateBlock(newBlock)
	klog.Infof("successfully update data block [%s]", util.GetBlockId(newBlock))
}

func (controller *SyncController) deleteBlock(obj interface{}) {
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
	_ = controller.cache.DeleteBlock(blockId)
	klog.Infof("successfully delete data block [%s]", blockId)
}

func (controller *SyncController) addClaim(obj interface{}) {
	claim, ok := obj.(*columbiav1.PrivacyBudgetClaim)
	if !ok {
		klog.Errorf("cannot convert to *v1.PrivacyBudgetClaim: %v", obj)
		return
	}

	_, _ = controller.cache.AddClaim(claim)
	klog.Infof("successfully add privacy budget claim [%s]", util.GetClaimId(claim))
}

func (controller *SyncController) updateClaim(oldObj, newObj interface{}) {
	newClaim, ok := newObj.(*columbiav1.PrivacyBudgetClaim)
	if !ok {
		klog.Errorf("cannot convert newObj to *v1.PrivacyBudgetClaim: %v", newObj)
		return
	}

	_, _ = controller.cache.UpdateClaim(newClaim)
	klog.Infof("successfully update privacy budget claim [%s]", util.GetClaimId(newClaim))
}

func (controller *SyncController) deleteClaim(obj interface{}) {
	var claim *columbiav1.PrivacyBudgetClaim
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

	claimId := util.GetClaimId(claim)
	_ = controller.cache.DeleteClaim(claimId)
	klog.Infof("successfully delete privacy budget claim [%s]", claimId)
}

func (controller *SyncController) addAllEventHandlers() {

	controller.privateDataBlockInformer.Informer().AddEventHandler(
		cache.FilteringResourceEventHandler{
			FilterFunc: func(obj interface{}) bool {
				switch t := obj.(type) {
				case *columbiav1.PrivateDataBlock:
					return true
				case cache.DeletedFinalStateUnknown:
					if _, ok := t.Obj.(*columbiav1.PrivateDataBlock); ok {
						return true
					}
					utilruntime.HandleError(fmt.Errorf("unable to convert object %T to *PrivateDataBlock", obj))
					return false
				default:
					utilruntime.HandleError(fmt.Errorf("unable to handle object %T", obj))
					return false
				}
			},
			Handler: cache.ResourceEventHandlerFuncs{
				AddFunc:    controller.addBlock,
				UpdateFunc: controller.updateBlock,
				DeleteFunc: controller.deleteBlock,
			},
		},
	)

	controller.privacyBudgetClaimInformer.Informer().AddEventHandler(
		cache.FilteringResourceEventHandler{
			FilterFunc: func(obj interface{}) bool {
				switch t := obj.(type) {
				case *columbiav1.PrivacyBudgetClaim:
					return true
				case cache.DeletedFinalStateUnknown:
					if _, ok := t.Obj.(*columbiav1.PrivacyBudgetClaim); ok {
						return true
					}
					utilruntime.HandleError(fmt.Errorf("unable to convert object %T to *PrivacyBudgetClaim", obj))
					return false
				default:
					utilruntime.HandleError(fmt.Errorf("unable to handle object %T", obj))
					return false
				}
			},
			Handler: cache.ResourceEventHandlerFuncs{
				AddFunc:    controller.addClaim,
				UpdateFunc: controller.updateClaim,
				DeleteFunc: controller.deleteClaim,
			},
		},
	)

}
