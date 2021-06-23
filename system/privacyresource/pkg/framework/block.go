package framework

import (
	"fmt"
	"sync"

	columbiav1 "columbia.github.com/privatekube/privacyresource/pkg/apis/columbia.github.com/v1"
)

type BlockHandler interface {
	Snapshot() *columbiav1.PrivateDataBlock

	View() *columbiav1.PrivateDataBlock

	Update(claim *columbiav1.PrivateDataBlock) error

	Delete()

	IsLive() bool

	GetId() string
}

type BlockHandlerImpl struct {
	block *columbiav1.PrivateDataBlock
	lock  sync.RWMutex
}

func NewBlockHandler(block *columbiav1.PrivateDataBlock) *BlockHandlerImpl {
	SortInitialBudget(block)
	return &BlockHandlerImpl{block: block}
}

func (handler *BlockHandlerImpl) Snapshot() *columbiav1.PrivateDataBlock {
	handler.lock.RLock()
	defer handler.lock.RUnlock()
	if handler.block == nil {
		return nil
	}
	//klog.Infof("snapshot block: %v. %v", handler.block.Status.CommittedBudgetMap, handler.block.Status.CommittedBudgetMap == nil)
	return handler.block.DeepCopy()
}

func (handler *BlockHandlerImpl) View() *columbiav1.PrivateDataBlock {
	handler.lock.RLock()
	defer handler.lock.RUnlock()
	return handler.block
}

func (handler *BlockHandlerImpl) Update(block *columbiav1.PrivateDataBlock) error {
	handler.lock.Lock()
	defer handler.lock.Unlock()
	if block.Generation < handler.block.Generation {
		return fmt.Errorf("block [%v] to update is staled", block.Name)
	}
	//klog.Infof("update block: %v. %v", handler.block.Status.CommittedBudgetMap, handler.block.Status.CommittedBudgetMap == nil)
	handler.block = block
	return nil
}

func (handler *BlockHandlerImpl) Delete() {
	handler.lock.Lock()
	defer handler.lock.Unlock()
	handler.block = nil
}

func (handler *BlockHandlerImpl) IsLive() bool {
	handler.lock.RLock()
	defer handler.lock.RUnlock()
	return handler.block == nil
}

func (handler *BlockHandlerImpl) GetId() string {
	handler.lock.RLock()
	defer handler.lock.RUnlock()
	if handler.block == nil {
		return ""
	}
	return handler.block.Namespace + "/" + handler.block.Name
}

func IsDataBlockReady(block *columbiav1.PrivateDataBlock) bool {
	if block == nil {
		return false
	}
	return !block.Status.PendingBudget.IsEmpty() || !block.Status.AvailableBudget.IsEmpty() ||
		len(block.Status.AcquiredBudgetMap) > 0 || len(block.Status.ReservedBudgetMap) > 0 ||
		len(block.Status.CommittedBudgetMap) > 0
}

func IsFlowModeProtected(block *columbiav1.PrivateDataBlock) bool {
	if block == nil {
		return false
	}
	return block.Spec.ReleaseMode == columbiav1.Flow && block.Spec.FlowReleasingOption != nil &&
		!block.Status.PendingBudget.IsEmpty()
}

func IsFlowMode(block *columbiav1.PrivateDataBlock) bool {
	if block == nil {
		return false
	}
	return block.Spec.ReleaseMode == columbiav1.Flow && block.Spec.FlowReleasingOption != nil
}

func SortInitialBudget(block *columbiav1.PrivateDataBlock) {
	if block.Spec.InitialBudget.IsRenyiType() {
		block.Spec.InitialBudget.Renyi.SortByAlpha()
	}
}
