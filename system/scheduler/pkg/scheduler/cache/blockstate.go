package cache

import (
	"fmt"
	"sync"

	columbiav1 "columbia.github.com/privatekube/privacyresource/pkg/apis/columbia.github.com/v1"
	"columbia.github.com/privatekube/scheduler/pkg/scheduler/util"
)

type DemandState struct {
	Availability bool
	Cost         float64
	IsValid      bool
}

type BlockState struct {
	sync.RWMutex
	block         *columbiav1.PrivateDataBlock
	initialBudget columbiav1.PrivacyBudget
	Demands       map[string]*DemandState
	arrivalNumber int
	id            string
}

func NewBlockState(block *columbiav1.PrivateDataBlock) *BlockState {
	return &BlockState{
		block:         block,
		initialBudget: block.Spec.InitialBudget,
		Demands:       map[string]*DemandState{},
		arrivalNumber: 0,
		id:            util.GetBlockId(block),
	}
}

func (blockState *BlockState) GetReservedMap() map[string]columbiav1.PrivacyBudget {
	blockState.RLock()
	defer blockState.RUnlock()

	reservedMap := blockState.block.Status.ReservedBudgetMap
	return reservedMap
}

func (blockState *BlockState) computeDemandState(budget columbiav1.PrivacyBudget) *DemandState {
	initialBudget := blockState.block.Spec.InitialBudget
	availableBudget := blockState.block.Status.AvailableBudget

	blockCost := getPerBlockCost(budget, initialBudget)

	return &DemandState{
		Availability: availableBudget.HasEnough(budget),
		Cost:         blockCost,
		IsValid:      true,
	}
}

func getPerBlockCost(budget columbiav1.PrivacyBudget, base columbiav1.PrivacyBudget) float64 {
	budget.ToRenyi()
	base.ToRenyi()
	return getPerBlockCostRenyi(budget.Renyi, base.Renyi)
}

func getPerBlockCostRenyi(budget columbiav1.RenyiBudget, base columbiav1.RenyiBudget) float64 {
	var blockCost float64
	b, c := columbiav1.ReduceToSameSupport(budget, base)
	for i := range b {
		blockCost += b[i].Epsilon / c[i].Epsilon
	}
	return blockCost
}

// UpdateDemandMap updates the demand for each claim on this block
// - dominant share
// - availability (enough eps/delta budget, or one alpha positive)
// - valid
func (blockState *BlockState) UpdateDemandMap() map[string]*DemandState {
	blockState.Lock()
	defer blockState.Unlock()

	demandMap := map[string]*DemandState{}
	for claimId, reservedBudget := range blockState.block.Status.ReservedBudgetMap {
		demand := blockState.computeDemandState(reservedBudget)
		demandMap[claimId] = demand
	}

	// invalid the old demand states
	for _, demandState := range blockState.Demands {
		demandState.IsValid = false
	}
	blockState.Demands = demandMap

	return demandMap
}

func (blockState *BlockState) Snapshot() *columbiav1.PrivateDataBlock {
	blockState.RLock()
	defer blockState.RUnlock()
	if blockState.block == nil {
		return nil
	}
	//klog.Infof("snapshot block: %v. %v", handler.block.Status.CommittedBudgetMap, handler.block.Status.CommittedBudgetMap == nil)
	return blockState.block.DeepCopy()
}

func (blockState *BlockState) View() *columbiav1.PrivateDataBlock {
	blockState.RLock()
	defer blockState.RUnlock()
	return blockState.block
}

func (blockState *BlockState) Update(block *columbiav1.PrivateDataBlock) error {
	blockState.Lock()
	defer blockState.Unlock()
	if block.Generation < blockState.block.Generation {
		return fmt.Errorf("block [%v] to update is staled", block.Name)
	}
	//klog.Infof("update block: %v. %v", handler.block.Status.CommittedBudgetMap, handler.block.Status.CommittedBudgetMap == nil)
	blockState.block = block
	return nil
}

func (blockState *BlockState) UpdateReturnOld(block *columbiav1.PrivateDataBlock) (*columbiav1.PrivateDataBlock, error) {
	blockState.Lock()
	defer blockState.Unlock()
	old := blockState.block
	if block.Generation < old.Generation {
		return old, fmt.Errorf("block [%v] to update is staled", block.Name)
	}
	//klog.Infof("update block: %v. %v", handler.block.Status.CommittedBudgetMap, handler.block.Status.CommittedBudgetMap == nil)
	blockState.block = block
	return old, nil
}

func (blockState *BlockState) Delete() {
	blockState.Lock()
	defer blockState.Unlock()
	blockState.block = nil
}

func (blockState *BlockState) IsLive() bool {
	blockState.RLock()
	defer blockState.RUnlock()
	return blockState.block == nil
}

func (blockState *BlockState) GetId() string {
	return blockState.id
}

func (blockState *BlockState) Quota(K int) columbiav1.PrivacyBudget {
	return blockState.initialBudget.Div(float64(K))
}
