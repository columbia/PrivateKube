package queue

import (
	"columbia.github.com/sage/privacyresource/pkg/framework"
	"container/list"
	"fmt"
	"k8s.io/klog"
	"sync"
)

type queueElementState string

const (
	InQueue = "IN_QUEUE"
	PopOut  = "POP_OUT"
	Waiting = "WAITING"
	Deleted = "DELETED"
)

type queueEntry struct {
	claimHandler framework.ClaimHandler
	state        queueElementState
}

type SchedulingQueue struct {
	stop chan struct{}
	lock sync.RWMutex

	schedulingQ list.List
	entryMap    map[string]*queueEntry
	closed      bool

	// wait queue + clock
	waitQueue      waitQueue
	defaultTimeout int64
}

func (queue *SchedulingQueue) checkIfRequestExists(requestId string) bool {
	_, ok := queue.entryMap[requestId]
	return ok
}

func NewSchedulingQueue(currTimestamp int64, defaultTimeout int64) *SchedulingQueue {
	return &SchedulingQueue{
		stop:           make(chan struct{}),
		waitQueue:      makeWaitQueue(currTimestamp),
		entryMap:       make(map[string]*queueEntry),
		closed:         false,
		defaultTimeout: defaultTimeout,
	}
}

func (queue *SchedulingQueue) Exist(requestId string) bool {
	queue.lock.RLock()
	defer queue.lock.RUnlock()
	return queue.checkIfRequestExists(requestId)
}

func (queue *SchedulingQueue) Add(handler framework.ClaimHandler) error {
	queue.lock.Lock()
	defer queue.lock.Unlock()
	if queue.checkIfRequestExists(handler.GetId()) {
		return fmt.Errorf("request [%s] has already in scheding queue", handler.GetId())
	}

	entry := &queueEntry{
		claimHandler: handler,
		state:        InQueue,
	}

	queue.schedulingQ.PushBack(entry)
	queue.entryMap[handler.GetId()] = entry
	return nil
}

// Pop removes the head of the queue and returns it. It blocks if the
// queue is empty and waits until a new item is added to the queue.
func (queue *SchedulingQueue) Pop() (framework.ClaimHandler, error) {
	queue.lock.Lock()
	defer queue.lock.Unlock()
	for queue.schedulingQ.Len() > 0 {
		entry := queue.schedulingQ.Remove(queue.schedulingQ.Front()).(*queueEntry)
		if entry.state == Deleted {
			continue
		}

		entry.state = PopOut
		return entry.claimHandler, nil
	}
	return nil, fmt.Errorf("no live requests in queue")
}

func (queue *SchedulingQueue) Delete(claimId string) error {
	queue.lock.Lock()
	defer queue.lock.Unlock()

	var entry *queueEntry
	if entry = queue.entryMap[claimId]; entry == nil {
		return fmt.Errorf("request [%s] does not exist in Scheduling Map", claimId)
	}

	entry.state = Deleted
	delete(queue.entryMap, claimId)
	return nil
}

// Close closes the SchedulingQueue so that the goroutine which is
// waiting to pop items can exit gracefully.
func (queue *SchedulingQueue) Close() {
	queue.lock.Lock()
	defer queue.lock.Unlock()

	close(queue.stop)
	queue.closed = true
	return
}

// Run starts the goroutines managing the queue.
func (queue *SchedulingQueue) Run() {
	queue.lock.Lock()
	defer queue.lock.Unlock()

	queue.closed = false
	return
}

func (queue *SchedulingQueue) Len() int {
	return queue.schedulingQ.Len()
}

func (queue *SchedulingQueue) Reschedule(handler framework.ClaimHandler) error {
	queue.lock.Lock()
	defer queue.lock.Unlock()

	claimId := handler.GetId()

	entry := queue.entryMap[claimId]
	if entry == nil {
		return fmt.Errorf("claim [%s] does not exist in the map", claimId)
	}

	if entry.state != PopOut {
		return fmt.Errorf("the state of claim [%s] is not POP_OUT", claimId)
	}

	entry.claimHandler = handler

	queue.schedulingQ.PushBack(entry)
	entry.state = InQueue
	klog.Infof("claim [%s] reschedule", claimId)
	return nil
}

func (queue *SchedulingQueue) ScheduleCompleted(claimId string) error {
	queue.lock.Lock()
	defer queue.lock.Unlock()

	entry := queue.entryMap[claimId]
	if entry == nil {
		return fmt.Errorf("request [%s] does not exist in the queue", claimId)
	}

	if entry.state != PopOut {
		return fmt.Errorf("the state of request [%s] is not POP_OUT", claimId)
	}

	delete(queue.entryMap, claimId)
	return nil
}
