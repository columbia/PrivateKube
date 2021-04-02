package datablocklifecycle

import (
	"columbia.github.com/sage/privacycontrollers/pkg/tools/util"
	columbiav1 "columbia.github.com/sage/privacyresource/pkg/apis/columbia.github.com/v1"
	"fmt"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/client-go/tools/cache"
	"k8s.io/klog"
)

func (controller *BlockLifecycleController) addBlock(obj interface{}) {
	block, ok := obj.(*columbiav1.PrivateDataBlock)
	if !ok {
		klog.Errorf("cannot convert to *v1.PrivacyDataBlock: %v", obj)
		return
	}

	if controller.initializeFlowOption(block) {
		if err := controller.updateBlockSpec(block); err == nil {
			klog.Infof("successfully initialize the flow option of block %s", util.GetBlockId(block))
			return
		}
	}

	if controller.initializeBlockStatus(block) {
		if err := controller.updateBlockStatus(block); err == nil {
			klog.Infof("successfully initialize the status of block %s", util.GetBlockId(block))
			return
		}
	}

}

func (controller *BlockLifecycleController) updateBlock(oldObj, newObj interface{}) {
	newBlock, ok := newObj.(*columbiav1.PrivateDataBlock)
	if !ok {
		klog.Errorf("cannot convert newObj to *v1.Request: %v", newObj)
		return
	}

	if controller.isBlockDeletable(newBlock) {
		if err := controller.deleteDataBlock(newBlock); err == nil {
			klog.Infof("block [%s] meets deletion criterion and is going to be deleted", util.GetBlockId(newBlock))
		}
		return
	}

	if controller.initializeFlowOption(newBlock) {
		if err := controller.updateBlockSpec(newBlock); err == nil {
			klog.Infof("successfully initialize the flow option of block %s", util.GetBlockId(newBlock))
			return
		}
	}

	if controller.initializeBlockStatus(newBlock) {
		if err := controller.updateBlockStatus(newBlock); err == nil {
			klog.Infof("successfully initialize the status of block %s", util.GetBlockId(newBlock))
			return
		}
	}
}

//func (controller *BlockController) deleteBlock(obj interface{}) {
//	var block *columbiav1.PrivateDataBlock
//	switch t := obj.(type) {
//	case *columbiav1.PrivateDataBlock:
//		block = t
//	case cache.DeletedFinalStateUnknown:
//		var ok bool
//		block, ok = t.Obj.(*columbiav1.PrivateDataBlock)
//		if !ok {
//			klog.Errorf("cannot convert to *PrivateDataBlock: %v", t.Obj)
//			return
//		}
//	default:
//		klog.Errorf("cannot convert to *PrivateDataBlock: %v", t)
//		return
//	}
//
//}

func (controller *BlockLifecycleController) addAllEventHandlers() {
	// scheduled pod cache
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
			},
		},
	)

}
