package algorithm

import (
	"columbia.github.com/sage/dpfscheduler/pkg/scheduler/cache"
	"columbia.github.com/sage/dpfscheduler/pkg/scheduler/errors"
	"columbia.github.com/sage/dpfscheduler/pkg/scheduler/util"
	columbiav1 "columbia.github.com/sage/privacyresource/pkg/apis/columbia.github.com/v1"
	"columbia.github.com/sage/privacyresource/pkg/framework"
	"fmt"
	"k8s.io/klog"
	"sync"
)

type BatchAlgorithm interface {
	BatchAllocate(requestHandler framework.RequestHandler) bool

	AllocateAvailableBudgets(blockStates []*cache.BlockState)

	BatchAllocateTimeout(requestHandler framework.RequestHandler) bool

	BatchConsume(requestHandler framework.RequestHandler) bool

	BatchRelease(requestHandler framework.RequestHandler) bool

	sync.Locker
}

func (dpf *DpfBatch) BatchConsume(requestHandler framework.RequestHandler) bool {
	consumeRequest := requestHandler.GetRequest().ConsumeRequest
	policy := ParseCommitPolicy(consumeRequest.Policy)
	requestViewer := requestHandler.Viewer()

	if policy.AllOrNothing && !dpf.tryBatchConsume(requestViewer) {
		klog.Warning("terminate this commit request because it has AllOrNothing policy and fails the try commit process")
		// try to rollback all the data block to be committed in case some of them were committed by last
		// commit terminated accidentally
		keys := util.GetStringKeysFromMap(consumeRequest.Consume)
		return dpf.BatchRollbackByKey(keys, requestHandler, errors.AllOrNothingError(nil).Error())
	}

	// build the commit policy
	var result columbiav1.BudgetErrorPair

	consumeStrategy := func(block *columbiav1.PrivateDataBlock) error {
		result = dpf.core.Consume(requestViewer, block)
		if result.Error == "" {
			return nil
		} else {
			return fmt.Errorf("request [%s] consumes data block [%s] failed: %s",
				util.GetRequestId(requestViewer.Claim, requestViewer.RequestIndex), util.GetBlockId(block), result.Error)
		}
	}

	responseMap := map[string]columbiav1.BudgetErrorPair{}

	for blockId := range consumeRequest.Consume {
		blockHandler := dpf.cache.GetBlock(blockId)
		err := dpf.updater.ApplyOperationToDataBlock(consumeStrategy, blockHandler)
		responseMap[blockId] = result

		if err == nil {
			klog.Infof("succeeded to consume data block [%s]", blockId)
			continue
		}

		klog.Infof("unable to consume data block [%s]: %v", blockId, err)
		// if an error exists in the response, we may go to rollback process depending on the policy.
		// start to rollback if one selected data block fails to be consumed
		if policy.AllOrNothing {
			// try to rollback all the data block to be consumed in case some of them were consumed by last
			// process terminated accidentally
			keys := util.GetStringKeysFromMap(consumeRequest.Consume)
			return dpf.BatchRollbackByKey(keys, requestHandler, errors.AllOrNothingError(err).Error())
		}
	}

	err := dpf.updater.ApplyOperationToClaim(func(claim *columbiav1.PrivacyBudgetClaim) error {
		if len(claim.Status.Responses) != requestHandler.RequestIndex {
			return fmt.Errorf("incorrect position to write the response. Expected position: %d, Actual Next Position %d",
				requestHandler.RequestIndex, len(claim.Status.Responses))
		}

		updateClaimByResponseMap(claim, responseMap)
		appendResponse(claim, columbiav1.Response{
			Identifier:       claim.Spec.Requests[requestHandler.RequestIndex].Identifier,
			State:            columbiav1.Success,
			AllocateResponse: nil,
			ConsumeResponse:  responseMap,
			ReleaseResponse:  nil,
		})
		return nil
	}, requestHandler.ClaimHandler)

	if err != nil {
		klog.Errorf("unable to apply operation: %v", err)
		return false
	}

	return true
}

func (dpf *DpfBatch) BatchRelease(requestHandler framework.RequestHandler) bool {
	releaseRequest := requestHandler.GetRequest().ReleaseRequest
	requestViewer := requestHandler.Viewer()

	var result columbiav1.BudgetErrorPair
	releaseStrategy := func(block *columbiav1.PrivateDataBlock) error {
		result = dpf.core.Release(requestViewer, block)
		if result.Error != "" {
			return fmt.Errorf("request [%s] releases data block [%s] failed: %s",
				util.GetRequestId(requestViewer.Claim, requestViewer.RequestIndex), util.GetBlockId(block), result.Error)
		}

		return nil
	}

	blockStates := make([]*cache.BlockState, 0, len(releaseRequest.Release))

	responseMap := map[string]columbiav1.BudgetErrorPair{}

	for blockId := range releaseRequest.Release {
		blockState := dpf.cache.GetBlock(blockId)
		if blockState == nil {
			continue
		}
		blockStates = append(blockStates, blockState)

		err := dpf.updater.ApplyOperationToDataBlock(releaseStrategy, blockState)
		responseMap[blockId] = result

		// if err occurs, then no update will be applied to this data block
		// so we will skip the following step, which records the change of this data block
		if err != nil {
			klog.Infof(err.Error())
			continue
		}

		klog.Infof("succeeded to process the release of data block [%s] by request [%s]", blockId, requestHandler.GetId())
	}

	err := dpf.updater.ApplyOperationToClaim(func(claim *columbiav1.PrivacyBudgetClaim) error {
		if len(claim.Status.Responses) != requestHandler.RequestIndex {
			return fmt.Errorf("incorrect position to write the response. Expected position: %d, Actual Next Position %d",
				requestHandler.RequestIndex, len(claim.Status.Responses))
		}

		updateClaimAcquiredBudgetMap(claim, responseMap)
		appendResponse(claim, columbiav1.Response{
			Identifier:       claim.Spec.Requests[requestHandler.RequestIndex].Identifier,
			State:            columbiav1.Success,
			AllocateResponse: nil,
			ConsumeResponse:  nil,
			ReleaseResponse:  responseMap,
		})
		return nil
	}, requestHandler.ClaimHandler)

	if err != nil {
		klog.Errorf("unable to apply operation: %v", err)
		return false
	}

	// The released budget may be used by data blocks
	dpf.AllocateAvailableBudgets(blockStates)

	return true
}

func (dpf *DpfBatch) getAvailableBlocks(requestViewer framework.RequestViewer) []*cache.BlockState {

	namespace := requestViewer.Claim.Namespace
	request := requestViewer.View().AllocateRequest
	requestId := requestViewer.Id

	predicate := func(blockState *cache.BlockState) bool {
		block := blockState.View()
		// If this data block has been allocated to this request, then we consider this should be available data blocks
		// belonging to this request.
		if _, ok := block.Status.LockedBudgetMap[requestId]; ok {
			return true
		}

		if block.Namespace != namespace || request.Dataset != block.Spec.Dataset {
			return false
		}

		dimensionMap := map[string]columbiav1.Dimension{}
		for _, dimension := range block.Spec.Dimensions {
			dimensionMap[dimension.Attribute] = dimension
		}

		for _, condition := range request.Conditions {
			dimension, ok := dimensionMap[condition.Attribute]
			if !ok || !validateCondition(condition, dimension) {
				return false
			}
		}

		return true
	}
	return dpf.cache.FilterBlock(predicate)
}

func (dpf *DpfBatch) tryBatchConsume(requestViewer framework.RequestViewer) bool {
	consumptionRequest := requestViewer.View().ConsumeRequest

	for blockId := range consumptionRequest.Consume {
		block := dpf.cache.GetBlock(blockId).Snapshot()
		result := dpf.core.Consume(requestViewer, block)
		if result.Error != "" {
			return false
		}
	}
	return true
}

func appendResponse(claim *columbiav1.PrivacyBudgetClaim, response columbiav1.Response) {
	claim.Status.Responses = append(claim.Status.Responses, response)
}

func updateClaimByResponseMap(claim *columbiav1.PrivacyBudgetClaim, responseMap map[string]columbiav1.BudgetErrorPair) {
	for blockId, result := range responseMap {
		if result.Error != "" {
			claim.Status.CommittedBudgets[blockId] = claim.Status.CommittedBudgets[blockId].Add(result.Budget)

			acquiredBudget := claim.Status.AcquiredBudgets[blockId].Minus(result.Budget)
			util.SetBudgetMapAndDeleteEmpty(claim.Status.AcquiredBudgets, blockId, acquiredBudget)
		}

	}
}

func updateClaimAcquiredBudgetMap(claim *columbiav1.PrivacyBudgetClaim, responseMap map[string]columbiav1.BudgetErrorPair) {
	for blockId, result := range responseMap {
		if result.Error == "" {
			acquiredBudget := claim.Status.AcquiredBudgets[blockId].Minus(result.Budget)
			util.SetBudgetMapAndDeleteEmpty(claim.Status.AcquiredBudgets, blockId, acquiredBudget)
		}
	}
}
