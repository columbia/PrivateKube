package scheduler

import (
	"context"
	"fmt"
	"time"

	"columbia.github.com/privatekube/privacyresource/pkg/framework"
	privacyclientset "columbia.github.com/privatekube/privacyresource/pkg/generated/clientset/versioned"
	"columbia.github.com/privatekube/scheduler/pkg/scheduler/algorithm"
	"columbia.github.com/privatekube/scheduler/pkg/scheduler/flowreleasing"
	"columbia.github.com/privatekube/scheduler/pkg/scheduler/queue"
	"columbia.github.com/privatekube/scheduler/pkg/scheduler/timing"
	"columbia.github.com/privatekube/scheduler/pkg/scheduler/updater"
	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/client-go/tools/record"
	"k8s.io/klog"

	columbiav1 "columbia.github.com/privatekube/privacyresource/pkg/apis/columbia.github.com/v1"
	informer "columbia.github.com/privatekube/privacyresource/pkg/generated/informers/externalversions/columbia.github.com/v1"
	schedulercache "columbia.github.com/privatekube/scheduler/pkg/scheduler/cache"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/client-go/tools/cache"
)

const (
	AllocateRequest = "allocate"
	ConsumeRequest  = "consume"
	CommitRequest   = "commit"
	ReleaseRequest  = "release"
	AbortRequest    = "abort"
)

type Scheduler struct {
	// It is expected that changes made via PrivateDataBlockCache will be observed
	// by NodeLister and Algorithm.
	cache schedulercache.Cache

	privateResourceClient privacyclientset.Interface

	schedulingQueue *queue.SchedulingQueue

	flowController flowreleasing.Controller

	timer timing.Timer

	updater *updater.ResourceUpdater

	batch algorithm.BatchAlgorithm
	//Algorithm string
	// PodConditionUpdater is used only in case of scheduling errors. If we succeed
	// with scheduling, PodScheduled condition will be updated in apiserver in /bind
	// handler so that binding and setting PodCondition it is atomic.
	//podConditionUpdater podConditionUpdater

	// Framework runs scheduler plugins at configured extension points.
	//Framework framework.Framework

	// NextRequest should be a function that blocks until the next pod
	// is available. We don't use a channel for this, because scheduling
	// a pod may take some amount of time and we don't want pods to get
	// stale while they sit in a channel.
	NextRequest func() framework.ClaimHandler

	// Error is called if there is an error. It is passed the pod in
	// question, and the error
	//Error func(*framework.claimHander, error)

	// Recorder is the EventRecorder to use
	Recorder record.EventRecorder

	// Close this to shut down the scheduler.
	StopEverything <-chan struct{}

	informersHaveSynced func() bool

	// channels
	allocChan chan string

	// default releasing period for DPN-T policy. Unit is ms.
	// How frequent should the scheduler release budget
	defaultReleasingPeriod int64
}

type SchedulerOption struct {

	// default timeout for pending requests. Unit is ms.
	DefaultTimeout int64

	// N for unlocking amount
	N int

	// default releasing period for DPN-T policy. Unit is ms.
	DefaultReleasingPeriod int64

	// default releasing period for DPN-T policy. Unit is ms.
	DefaultReleasingDuration int64
}

func DefaultTSchemeOption() SchedulerOption {
	return SchedulerOption{
		DefaultTimeout:           20000,
		DefaultReleasingPeriod:   10000,
		DefaultReleasingDuration: 30000,
	}
}

func (option SchedulerOption) String() string {
	return fmt.Sprintf("Timeout %d\nN %d\nPeriod %d\nTotal duration %d", option.DefaultTimeout, option.N, option.DefaultReleasingPeriod, option.DefaultReleasingDuration)
}

func New(privacyResourceClient privacyclientset.Interface,
	privateDataBlockInformer informer.PrivateDataBlockInformer,
	privacyBudgetClaimInformer informer.PrivacyBudgetClaimInformer,
	recorder record.EventRecorder,
	stopCh <-chan struct{},
	option SchedulerOption) (*Scheduler, error) {
	// this schedulerCache will be used by the scheduler and its algorithm engine
	schedulerCache := schedulercache.NewStateCache(option.StreamingCounterOptions)

	scheduler := new(Scheduler)
	scheduler.privateResourceClient = privacyResourceClient
	scheduler.cache = schedulerCache
	scheduler.timer = timing.MakeDefaultTimer()
	scheduler.schedulingQueue = queue.NewSchedulingQueue(scheduler.timer.Now(), option.DefaultTimeout)
	scheduler.updater = updater.NewResourceUpdater(privacyResourceClient, schedulerCache)

	scheduler.allocChan = make(chan string, 16)

	scheduler.batch = algorithm.NewTSchemeBatch(*scheduler.updater, schedulerCache, scheduler.allocChan)
	releaseOption := flowreleasing.ReleaseOption{
		DefaultDuration: option.DefaultReleasingDuration,
	}
	scheduler.flowController = flowreleasing.MakeController(schedulerCache, scheduler.updater, scheduler.timer, releaseOption)
	scheduler.defaultReleasingPeriod = option.DefaultReleasingPeriod

	scheduler.informersHaveSynced = func() bool {
		return privateDataBlockInformer.Informer().HasSynced() && privacyBudgetClaimInformer.Informer().HasSynced()
	}
	// Informer should be run first to start watching!
	//go privateDataBlockInformer.Informer().Run(stopCh)
	AddAllEventHandlers(scheduler, "", privateDataBlockInformer, privacyBudgetClaimInformer)
	scheduler.Recorder = recorder
	scheduler.StopEverything = stopCh

	scheduler.NextRequest = func() framework.ClaimHandler {
		claimHandler, _ := scheduler.schedulingQueue.Pop()
		return claimHandler
	}

	return scheduler, nil
}

// Run begins watching and scheduling. It waits for cache to be synced, then starts scheduling and blocked until the context is done.
func (scheduler *Scheduler) Run(ctx context.Context) {
	if !cache.WaitForCacheSync(ctx.Done(), scheduler.informersHaveSynced) {
		return
	}
	scheduler.schedulingQueue.Run()
	//scheduler.flowController.Run()
	go scheduler.channelHandler()

	go wait.UntilWithContext(ctx, scheduler.flowReleaseAndAllocate, time.Duration(scheduler.defaultReleasingPeriod)*time.Millisecond)

	go wait.UntilWithContext(ctx, scheduler.checkTimeout, queue.BucketSize*time.Millisecond)

	wait.UntilWithContext(ctx, scheduler.scheduleOne, 0)

	scheduler.schedulingQueue.Close()
	//scheduler.flowController.Close()
}

func (scheduler *Scheduler) scheduleOne(ctx context.Context) {
	claimHandler := scheduler.NextRequest()
	// request could be nil when schedulerQueue is closed
	if claimHandler == nil {
		time.Sleep(1 * time.Second)
		return
	}

	requestHandler := claimHandler.NextRequest()
	if !requestHandler.IsValid() {
		return
	}

	claimId := claimHandler.GetId()
	requestId := requestHandler.GetId()
	klog.Infof("Attempting to schedule request: %v", requestId)

	if requestHandler.IsPending() {
		klog.Infof("wait the claim with pending request: %s", requestId)
		_ = scheduler.schedulingQueue.WaitEntry(claimHandler)
		return
	}

	var isCompleted bool

	scheduler.batch.Lock()
	switch getRequestType(requestHandler.GetRequest()) {
	case AllocateRequest:
		isCompleted = scheduler.batch.BatchAllocate(requestHandler)
	case ConsumeRequest:
		isCompleted = scheduler.batch.BatchConsume(requestHandler)
	case ReleaseRequest:
		isCompleted = scheduler.batch.BatchRelease(requestHandler)
	default:
		klog.Errorf("unable to get the request type from %v", requestHandler.GetRequest())
		scheduler.batch.Unlock()
		return
	}
	scheduler.batch.Unlock()

	requestHandler = claimHandler.NextRequest()
	if requestHandler.IsPending() {
		klog.Infof("wait the claim with pending request: %s", requestId)
		_ = scheduler.schedulingQueue.WaitEntry(claimHandler)
		return
	}

	// There are more requests to be handle in this claim
	if !isCompleted || requestHandler.IsValid() {
		klog.Infof("reschedule the claim [%s] which has more requests", claimId)
		if err := scheduler.schedulingQueue.Reschedule(claimHandler); err != nil {
			klog.Errorf("unable to reschedule: %v", err)
		}
		return
	}

	if err := scheduler.schedulingQueue.ScheduleCompleted(claimId); err == nil {
		klog.Infof("processing the request %s completed", requestId)
	} else {
		klog.Infof("failed to complete the scheduling of request %s", requestId)
	}

}

func getRequestType(request *columbiav1.Request) string {
	if request.AllocateRequest != nil {
		return AllocateRequest
	} else if request.ConsumeRequest != nil {
		return ConsumeRequest
	} else if request.ReleaseRequest != nil {
		return ReleaseRequest
	} else {
		return ""
	}
}

func AddAllEventHandlers(
	privateDataScheduler *Scheduler,
	schedulerName string,
	privateDataBlockInformer informer.PrivateDataBlockInformer,
	privacyBudgetClaimInformer informer.PrivacyBudgetClaimInformer,
) {
	// scheduled pod cache
	privacyBudgetClaimInformer.Informer().AddEventHandler(
		cache.FilteringResourceEventHandler{
			FilterFunc: func(obj interface{}) bool {
				switch t := obj.(type) {
				case *columbiav1.PrivacyBudgetClaim:
					return true
				case cache.DeletedFinalStateUnknown:
					if _, ok := t.Obj.(*columbiav1.PrivacyBudgetClaim); ok {
						return true
					}
					utilruntime.HandleError(fmt.Errorf("unable to convert object %T to *PrivacyBudgetClaim in %T", obj, privateDataScheduler))
					return false
				default:
					utilruntime.HandleError(fmt.Errorf("unable to handle object in %T: %T", privateDataScheduler, obj))
					return false
				}
			},
			Handler: cache.ResourceEventHandlerFuncs{
				AddFunc:    privateDataScheduler.addClaim,
				UpdateFunc: privateDataScheduler.updateClaim,
				DeleteFunc: privateDataScheduler.deleteClaim,
			},
		},
	)

	privateDataBlockInformer.Informer().AddEventHandler(
		cache.FilteringResourceEventHandler{
			FilterFunc: func(obj interface{}) bool {
				switch t := obj.(type) {
				case *columbiav1.PrivateDataBlock:
					return true
				case cache.DeletedFinalStateUnknown:
					if _, ok := t.Obj.(*columbiav1.PrivateDataBlock); ok {
						return true
					}
					utilruntime.HandleError(fmt.Errorf("unable to convert object %T to *PrivateDataDataBlock in %T", obj, privateDataScheduler))
					return false
				default:
					utilruntime.HandleError(fmt.Errorf("unable to handle object in %T: %T", privateDataScheduler, obj))
					return false
				}
			},
			Handler: cache.ResourceEventHandlerFuncs{
				AddFunc:    privateDataScheduler.addBlock,
				UpdateFunc: privateDataScheduler.updateBlock,
				DeleteFunc: privateDataScheduler.deleteBlock,
			},
		},
	)

}

// skipPodSchedule returns true if we could skip scheduling the pod for specified cases.
func (scheduler *Scheduler) skipPodSchedule(pod *v1.Pod) bool {
	// Case 1: pod is being deleted.
	if pod.DeletionTimestamp != nil {
		scheduler.Recorder.Eventf(pod, v1.EventTypeWarning, "FailedScheduling", "skip schedule deleting pod: %v/%v", pod.Namespace, pod.Name)
		klog.V(3).Infof("Skip schedule deleting pod: %v/%v", pod.Namespace, pod.Name)
		return true
	}

	// Case 2: pod has been assumed and pod updates could be skipped.
	// An assumed pod can be added again to the scheduling queue if it got an update event
	// during its previous scheduling cycle but before getting assumed.
	//if scheduler.skipPodUpdate(pod) {
	//	return true
	//}

	return false
}

func (scheduler *Scheduler) Now() int64 {
	return scheduler.timer.Now()
}

func (scheduler *Scheduler) channelHandler() {
	for claimId := range scheduler.allocChan {
		_ = scheduler.schedulingQueue.WakeUpEntry(claimId)
	}
}

func (scheduler *Scheduler) checkTimeout(ctx context.Context) {
	scheduler.batch.Lock()
	defer scheduler.batch.Unlock()

	claimHandlers := scheduler.schedulingQueue.PopWaitingUntil(scheduler.timer.Now())

	for _, claimHandler := range claimHandlers {
		requestHandler := claimHandler.NextRequest()
		scheduler.batch.BatchAllocateTimeout(requestHandler)

		// There are more requests to be handle in this claim
		requestHandler.RequestIndex++
		if requestHandler.IsValid() {
			klog.Infof("reschedule the claim [%s] which has more requests", claimHandler.GetId())
			if err := scheduler.schedulingQueue.Reschedule(claimHandler); err != nil {
				klog.Errorf("unable to reschedule: %v", err)
			}
			continue
		} else {
			_ = scheduler.schedulingQueue.ScheduleCompleted(claimHandler.GetId())
			klog.Infof("processing the requests of claim %s completed", claimHandler.GetId())
		}
	}

}

func (scheduler *Scheduler) flowReleaseAndAllocate(ctx context.Context) {
	scheduler.batch.Lock()
	defer scheduler.batch.Unlock()

	blockStates := scheduler.flowController.Release()
	scheduler.batch.AllocateAvailableBudgets(blockStates)
}
