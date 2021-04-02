package queue

import (
	"columbia.github.com/sage/privacyresource/pkg/framework"
	"container/list"
	"fmt"
	"sync"
	"time"
)

type flowAllocationElement struct {
	block framework.BlockHandler
	state queueElementState
}

type FlowAllocationQueue struct {
	stop chan struct{}
	lock sync.RWMutex

	// the element is blockHandler
	allocationQ list.List
	blockMap    map[string]*flowAllocationElement
	closed      bool

	//parameters
	DeletionGracePeriod time.Duration
}

func MakeFlowAllocationQueue() FlowAllocationQueue {
	flowAllocationQueue := FlowAllocationQueue{
		stop:                make(chan struct{}),
		lock:                sync.RWMutex{},
		allocationQ:         list.List{},
		blockMap:            make(map[string]*flowAllocationElement),
		closed:              false,
		DeletionGracePeriod: 30 * time.Second,
	}
	return flowAllocationQueue
}

func (queue *FlowAllocationQueue) checkIfRequestExists(blockId string) bool {
	_, ok := queue.blockMap[blockId]
	return ok
}

func (queue *FlowAllocationQueue) Add(handler framework.BlockHandler) error {
	queue.lock.Lock()
	defer queue.lock.Unlock()
	blockId := handler.GetId()
	var element *flowAllocationElement
	var ok bool
	// the deleted element will not exist in the blockMap. Even the deleted block reenters the queue, the queue
	// will create a create element for it.

	if _, ok = queue.blockMap[blockId]; ok {
		return fmt.Errorf("block [%s] has already in flow allocation queue", handler.GetId())

	}

	element = &flowAllocationElement{
		block: handler,
		state: InQueue,
	}
	queue.blockMap[blockId] = element
	queue.allocationQ.PushBack(element)
	return nil
}

func (queue *FlowAllocationQueue) ReenterFront(handler framework.BlockHandler) error {
	queue.lock.Lock()
	defer queue.lock.Unlock()

	blockId := handler.GetId()
	metaData, ok := queue.blockMap[blockId]
	if !ok || metaData.state != PopOut {
		return fmt.Errorf("the reentering element is not poped out from this queue")
	}

	metaData.state = InQueue

	queue.allocationQ.PushFront(metaData)
	return nil
}

func (queue *FlowAllocationQueue) NextAvailable() framework.BlockHandler {
	queue.lock.Lock()
	defer queue.lock.Unlock()

	for queue.allocationQ.Len() > 0 {
		element, ok := queue.allocationQ.Front().Value.(*flowAllocationElement)
		if !ok || element.state == Deleted {
			queue.allocationQ.Remove(queue.allocationQ.Front())
			continue
		}

		return element.block
	}
	return nil
}

func (queue *FlowAllocationQueue) TryPop() framework.BlockHandler {
	queue.lock.Lock()
	defer queue.lock.Unlock()

	for queue.allocationQ.Len() > 0 {
		element, ok := queue.allocationQ.Remove(queue.allocationQ.Front()).(*flowAllocationElement)
		if !ok || element.state == Deleted {
			continue
		}
		element.state = PopOut
		return element.block
	}

	return nil
}

func (queue *FlowAllocationQueue) CommitPop(handler framework.BlockHandler) {
	queue.lock.Lock()
	defer queue.lock.Unlock()

	blockId := handler.GetId()

	metaData, ok := queue.blockMap[blockId]
	if !ok {
		return
	}

	if metaData.state == PopOut {
		delete(queue.blockMap, blockId)
	}
}

func (queue *FlowAllocationQueue) Delete(blockId string) error {
	queue.lock.Lock()
	defer queue.lock.Unlock()

	if !queue.checkIfRequestExists(blockId) {
		return fmt.Errorf("block [%s] does not exist in Allocation Map", blockId)
	}
	queue.blockMap[blockId].state = Deleted
	delete(queue.blockMap, blockId)
	return nil
}

// Close closes the SchedulingQueue so that the goroutine which is
// waiting to pop items can exit gracefully.
func (queue *FlowAllocationQueue) Close() {
	queue.lock.Lock()
	defer queue.lock.Unlock()

	close(queue.stop)
	queue.closed = true
	return
}
