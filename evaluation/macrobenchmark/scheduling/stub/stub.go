package stub

import (
	"context"
	"fmt"
	"time"

	"columbia.github.com/privatekube/dpfscheduler/pkg/scheduler"
	"columbia.github.com/privatekube/privacycontrollers/pkg/datablocklifecycle"
	columbiav1 "columbia.github.com/privatekube/privacyresource/pkg/apis/columbia.github.com/v1"
	"columbia.github.com/privatekube/privacyresource/pkg/generated/clientset/versioned"
	"columbia.github.com/privatekube/privacyresource/pkg/generated/clientset/versioned/fake"
	privacyinformers "columbia.github.com/privatekube/privacyresource/pkg/generated/informers/externalversions"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/kubernetes/scheme"
	"k8s.io/client-go/tools/record"
)

type Stub struct {
	Client           versioned.Interface
	PrivacyScheduler *scheduler.DpfScheduler
	Namespace        string
}

func NewStub() Stub {
	return Stub{
		Client:           fake.NewSimpleClientset(),
		PrivacyScheduler: nil,
		Namespace:        "test-scheduler",
	}
}

func (s *Stub) StartN(timeout time.Duration, N int, scheduler_method string) {
	option := scheduler.DefaultNSchemeOption()
	option.DefaultTimeout = int64(timeout / time.Millisecond)
	option.N = N
	option.Scheduler = scheduler_method
	s.PrivacyScheduler = startScheduler(s.Client, option)
}

func (s *Stub) StartT(timeout time.Duration, DPF_T int, dpf_release_period_millisecond int, scheduler_method string) {
	option := scheduler.DefaultTSchemeOption()
	option.DefaultTimeout = int64(timeout / time.Millisecond)
	option.DefaultReleasingPeriod = int64(dpf_release_period_millisecond)
	option.DefaultReleasingDuration = int64(DPF_T) * int64(dpf_release_period_millisecond)
	option.Scheduler = scheduler_method
	fmt.Println("Starting DPF-T scheduler:\n", option)
	s.PrivacyScheduler = startScheduler(s.Client, option)
}

func startScheduler(client versioned.Interface, option scheduler.DpfSchedulerOption) *scheduler.DpfScheduler {
	privacyResourceInformerFactory := privacyinformers.NewSharedInformerFactory(client, time.Second*30)

	privateDataBlockInformer := privacyResourceInformerFactory.Columbia().V1().PrivateDataBlocks()
	privacyBudgetClaimInformer := privacyResourceInformerFactory.Columbia().V1().PrivacyBudgetClaims()

	ctx, _ := context.WithCancel(context.Background())

	coreBroadcaster := record.NewBroadcaster()
	coreRecorder := coreBroadcaster.NewRecorder(scheme.Scheme, corev1.EventSource{Component: "privacyscheduler"})

	privacyScheduler, _ := scheduler.New(client, privateDataBlockInformer,
		privacyBudgetClaimInformer, coreRecorder, ctx.Done(), option)

	// notice that there is no need to run Start methods in a separate goroutine. (i.e. go kubeInformerFactory.Start(stopCh)
	// Start method is non-blocking and runs all registered informers in a dedicated goroutine.
	privacyResourceInformerFactory.Start(ctx.Done())
	go privacyScheduler.Run(ctx)

	var blockLifecycleOption datablocklifecycle.BlockLifecycleOption
	if option.Mode == scheduler.NScheme {
		blockLifecycleOption = datablocklifecycle.BlockLifecycleOption{
			SchedulingPolicy:           datablocklifecycle.DpfN,
			DeletionGracePeriodSeconds: 30,
		}
	} else {
		blockLifecycleOption = datablocklifecycle.BlockLifecycleOption{
			SchedulingPolicy:           datablocklifecycle.DpfT,
			DeletionGracePeriodSeconds: 30,
		}
	}

	blockLifecycleController := datablocklifecycle.NewBlockController(client, privateDataBlockInformer, blockLifecycleOption)
	blockLifecycleController.Run()

	return privacyScheduler
}

func (s *Stub) CreateDataBlock(block *columbiav1.PrivateDataBlock) (*columbiav1.PrivateDataBlock, error) {
	return s.Client.ColumbiaV1().PrivateDataBlocks(block.Namespace).Create(context.TODO(), block, metav1.CreateOptions{})
}

func (s *Stub) UpdateDataBlockStatus(block *columbiav1.PrivateDataBlock) (*columbiav1.PrivateDataBlock, error) {
	return s.Client.ColumbiaV1().PrivateDataBlocks(block.Namespace).UpdateStatus(context.TODO(), block, metav1.UpdateOptions{})
}

func (s *Stub) CreateClaim(claim *columbiav1.PrivacyBudgetClaim) (*columbiav1.PrivacyBudgetClaim, error) {
	return s.Client.ColumbiaV1().PrivacyBudgetClaims(claim.Namespace).Create(context.TODO(), claim, metav1.CreateOptions{})
}

func (s *Stub) UpdateClaimSpec(claim *columbiav1.PrivacyBudgetClaim) (*columbiav1.PrivacyBudgetClaim, error) {
	return s.Client.ColumbiaV1().PrivacyBudgetClaims(claim.Namespace).Update(context.TODO(), claim, metav1.UpdateOptions{})
}

func (s *Stub) GetDataBlock(name string) *columbiav1.PrivateDataBlock {
	block, err := s.Client.ColumbiaV1().PrivateDataBlocks(s.Namespace).Get(context.TODO(), name, metav1.GetOptions{})
	if err != nil {
		fmt.Println("Error while getting the block, returning an empty block:", err)
		b := &columbiav1.PrivateDataBlock{
			ObjectMeta: metav1.ObjectMeta{
				Name:      name,
				Namespace: "error",
			},
			Spec: columbiav1.PrivateDataBlockSpec{},
		}
		return b
	}
	return block
}

func (s *Stub) GetClaim(name string) *columbiav1.PrivacyBudgetClaim {
	claim, err := s.Client.ColumbiaV1().PrivacyBudgetClaims(s.Namespace).Get(context.TODO(), name, metav1.GetOptions{})
	if err != nil {
		fmt.Println("Error while getting the claim, returning an empty claim:", err)
		c := &columbiav1.PrivacyBudgetClaim{
			ObjectMeta: metav1.ObjectMeta{
				Name:      name,
				Namespace: "error",
			},
			Spec: columbiav1.PrivacyBudgetClaimSpec{},
		}
		return c
	}
	return claim
}
