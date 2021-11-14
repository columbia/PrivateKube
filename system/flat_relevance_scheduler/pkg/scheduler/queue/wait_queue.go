package queue

import (
	"fmt"

	"columbia.github.com/privatekube/privacyresource/pkg/framework"
	"k8s.io/klog"
)

const (
	// 2 seconds
	BucketSize = 2000
)

type waitQueue struct {
	nextBucketId   int64
	entryBucketMap map[*queueEntry]int64
	// map from time bucket to an array of waiting queue entries
	buckets map[int64][]*queueEntry
}

func makeWaitQueue(timestamp int64) waitQueue {
	return waitQueue{
		nextBucketId:   timestamp / BucketSize,
		entryBucketMap: map[*queueEntry]int64{},
		buckets:        map[int64][]*queueEntry{},
	}
}

func (wq *waitQueue) Wait(entry *queueEntry, wakeupTime int64) {
	bucketId := (wakeupTime + BucketSize - 1) / BucketSize

	if bucketId < wq.nextBucketId {
		bucketId = wq.nextBucketId
	}

	wq.buckets[bucketId] = append(wq.buckets[bucketId], entry)
	wq.entryBucketMap[entry] = bucketId

}

func (queue *SchedulingQueue) WaitEntry(handler framework.ClaimHandler) error {
	queue.lock.Lock()
	defer queue.lock.Unlock()

	claimId := handler.GetId()

	var entry *queueEntry

	if entry = queue.entryMap[claimId]; entry == nil {
		entry = &queueEntry{
			claimHandler: handler,
			state:        Waiting,
		}
		queue.entryMap[claimId] = entry
	} else {
		if entry.state != PopOut {
			return fmt.Errorf("the claim has existed in the scheduling queue and the state is not PopOut")
		}

		entry.state = Waiting
		entry.claimHandler = handler
	}

	requestViewer := handler.NextRequest().Viewer()
	if !requestViewer.IsPending() {
		return fmt.Errorf("the claim has no pending request")
	}

	var timeout int64
	if requestViewer.Claim.Spec.Requests[requestViewer.RequestIndex].AllocateRequest.Timeout != 0 {
		timeout = requestViewer.Claim.Spec.Requests[requestViewer.RequestIndex].AllocateRequest.Timeout
	} else {
		timeout = queue.defaultTimeout
	}

	waitTime := requestViewer.Claim.Status.Responses[requestViewer.RequestIndex].AllocateResponse.StartTime + timeout

	queue.waitQueue.Wait(entry, waitTime)

	klog.Infof("claim [%s] starts to wait", claimId)
	return nil
}

func removeQueueEntry(bucket []*queueEntry, candidate *queueEntry) []*queueEntry {
	for i, entry := range bucket {
		if entry == candidate {
			bucket[i] = bucket[len(bucket)-1]
			bucket[len(bucket)-1] = nil
			return bucket[:len(bucket)-1]
		}
	}
	return bucket
}

func (wq *waitQueue) Wakeup(entry *queueEntry) {
	bucketId := wq.entryBucketMap[entry]
	bucket := wq.buckets[bucketId]
	wq.buckets[bucketId] = removeQueueEntry(bucket, entry)
	delete(wq.entryBucketMap, entry)
}

func (queue *SchedulingQueue) WakeUpEntry(claimId string) error {
	queue.lock.Lock()
	defer queue.lock.Unlock()

	entry := queue.entryMap[claimId]
	if entry == nil || entry.state != Waiting {
		return fmt.Errorf("claim [%s] is not waiting", claimId)
	}

	queue.waitQueue.Wakeup(entry)

	queue.schedulingQ.PushBack(entry)
	entry.state = InQueue

	klog.Infof("claim [%s] wakes up and enters the queue", claimId)
	return nil
}

func (wq *waitQueue) WakeUpUntil(timestamp int64) []*queueEntry {
	result := make([]*queueEntry, 0, 8)
	untilBucketId := timestamp / BucketSize

	for wq.nextBucketId <= untilBucketId {
		bucket := wq.buckets[wq.nextBucketId]
		for i, entry := range bucket {
			// skip the lazy-deleted entry
			if entry.state == Deleted {
				continue
			}
			result = append(result, entry)
			bucket[i] = nil
			delete(wq.entryBucketMap, entry)
		}
		delete(wq.buckets, wq.nextBucketId)
		wq.nextBucketId++
	}

	return result
}

func (queue *SchedulingQueue) PopWaitingUntil(timestamp int64) []framework.ClaimHandler {
	queue.lock.Lock()
	defer queue.lock.Unlock()

	entries := queue.waitQueue.WakeUpUntil(timestamp)
	claimHandlers := make([]framework.ClaimHandler, 0, len(entries))

	for _, entry := range entries {
		claimHandlers = append(claimHandlers, entry.claimHandler)
		entry.state = PopOut
	}

	return claimHandlers
}
