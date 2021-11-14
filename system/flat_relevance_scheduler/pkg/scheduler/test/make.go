package test

import (
	"context"
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
	"k8s.io/klog"
)

const (
	namespace        = "test-scheduler"
	unprocessedBlock = "block-unprocessed"
	processedBlock   = "block-processed"
)

var (
	client           versioned.Interface
	privacyScheduler *scheduler.DpfScheduler

	initialBudget = columbiav1.EpsilonDeltaBudget{
		Epsilon: 1,
		Delta:   0.01,
	}

	acquiringBudget = columbiav1.EpsilonDeltaBudget{
		Epsilon: 0.1,
		Delta:   0.001,
	}

	overAcquiringBudget = columbiav1.EpsilonDeltaBudget{
		Epsilon: 0.1,
		Delta:   0.01,
	}

	reservingBudget = columbiav1.EpsilonDeltaBudget{
		Epsilon: 0.4,
		Delta:   0.009,
	}

	commitAcquiredBudget = columbiav1.EpsilonDeltaBudget{
		Epsilon: 0.05,
		Delta:   0.0005,
	}

	commitReservedBudget = columbiav1.EpsilonDeltaBudget{
		Epsilon: 0.2,
		Delta:   0.005,
	}

	overCommitReservedBudget = columbiav1.EpsilonDeltaBudget{
		Epsilon: 1,
		Delta:   0.005,
	}

	abortAcquiredBudget = columbiav1.EpsilonDeltaBudget{
		Epsilon: 0.05,
		Delta:   0.0005,
	}

	abortReservedBudget = columbiav1.EpsilonDeltaBudget{
		Epsilon: 0.2,
		Delta:   0.004,
	}
)

func init() {
	//client = newRealClient()
	client = fake.NewSimpleClientset()
}

//func newRealClient() versioned.Interface {
//	var kubeconfig, masterURL string
//	kubeconfig = "/Users/augustinpan/.kube/config"
//
//
//	cfg, err := clientcmd.BuildConfigFromFlags(masterURL, kubeconfig)
//	if err != nil {
//		klog.Fatalf("Error building kubeconfig: %s", err.Error())
//		panic("unable to create a client")
//	}
//
//	client, err = versioned.NewForConfig(cfg)
//	if err != nil {
//		klog.Fatalf("Error building privacy data clientset: %s", err.Error())
//		panic("unable to create a client")
//	}
//	return client
//}

func startScheduler(option scheduler.DpfSchedulerOption) {
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
	klog.Info("Starting blockLifecycleController...")

	blockLifecycleController.Run()
	klog.Info("blockLifecycleController is running.")

}

func startDefaultScheduler() {
	option := scheduler.DefaultNSchemeOption()
	option.DefaultTimeout = 5000
	startScheduler(option)
}

func startDefaultSchedulerWithCounter() {
	option := scheduler.DefaultNSchemeOption()
	option.StreamingCounterOptions = scheduler.NewStreamingCounterOptions(1, 1e6)
	option.DefaultTimeout = 5000
	startScheduler(option)
}

func createDataBlock(block *columbiav1.PrivateDataBlock) (*columbiav1.PrivateDataBlock, error) {
	return client.ColumbiaV1().PrivateDataBlocks(block.Namespace).Create(context.TODO(), block, metav1.CreateOptions{})
}

// func createDataBlockWithCounter(scheduler *scheduler.DpfScheduler, block *columbiav1.PrivateDataBlock) (*columbiav1.PrivateDataBlock, error) {
// 	// scheduler.updater.ApplyOperationToDataBlock()
// 	// NewDpfNSchemeBatch

// 	block.Spec.Dimensions[0] = columbiav1.Dimension{
// 		Attribute:    "DPCount",
// 		NumericValue: util.ToDecimal(12),
// 		StringValue:  fmt.Sprintf("%d", 3),
// 	}
// 	return client.ColumbiaV1().PrivateDataBlocks(block.Namespace).Create(context.TODO(), block, metav1.CreateOptions{})

// }

func updateDataBlockStatus(block *columbiav1.PrivateDataBlock) (*columbiav1.PrivateDataBlock, error) {
	return client.ColumbiaV1().PrivateDataBlocks(block.Namespace).UpdateStatus(context.TODO(), block, metav1.UpdateOptions{})
}

func createClaim(claim *columbiav1.PrivacyBudgetClaim) (*columbiav1.PrivacyBudgetClaim, error) {
	return client.ColumbiaV1().PrivacyBudgetClaims(claim.Namespace).Create(context.TODO(), claim, metav1.CreateOptions{})
}

func updateClaimSpec(claim *columbiav1.PrivacyBudgetClaim) (*columbiav1.PrivacyBudgetClaim, error) {
	return client.ColumbiaV1().PrivacyBudgetClaims(claim.Namespace).Update(context.TODO(), claim, metav1.UpdateOptions{})
}

func getDataBlock(name string) *columbiav1.PrivateDataBlock {
	block, err := client.ColumbiaV1().PrivateDataBlocks(namespace).Get(context.TODO(), name, metav1.GetOptions{})
	if err != nil {
		panic("cannot get the data block")
	}
	return block
}

func getClaim(name string) *columbiav1.PrivacyBudgetClaim {
	claim, err := client.ColumbiaV1().PrivacyBudgetClaims(namespace).Get(context.TODO(), name, metav1.GetOptions{})
	if err != nil {
		panic("cannot get the data block")
	}
	return claim
}

func getClaimId(name string) string {
	return namespace + "/" + name
}

func newDataBlock(name string, dataset string) *columbiav1.PrivateDataBlock {
	block := &columbiav1.PrivateDataBlock{
		ObjectMeta: metav1.ObjectMeta{
			Name:      name,
			Namespace: namespace,
		},
		Spec: columbiav1.PrivateDataBlockSpec{
			InitialBudget: columbiav1.PrivacyBudget{
				EpsDel: &initialBudget,
			},
			Dataset: dataset,
		},
	}
	block, err := createDataBlock(block)
	if err != nil {
		panic("cannot create a data block")
	}

	return block
}

func newFlowDataBlock(name string, dataset string, duration int64) *columbiav1.PrivateDataBlock {
	block := &columbiav1.PrivateDataBlock{
		ObjectMeta: metav1.ObjectMeta{
			Name:      name,
			Namespace: namespace,
		},
		Spec: columbiav1.PrivateDataBlockSpec{
			InitialBudget: columbiav1.PrivacyBudget{
				EpsDel: &initialBudget,
			},
			Dataset:     dataset,
			ReleaseMode: columbiav1.Flow,
			FlowReleasingOption: &columbiav1.FlowReleasingOption{
				StartTime: privacyScheduler.Now(),
				Duration:  duration,
			},
		},
	}
	block, err := createDataBlock(block)
	if err != nil {
		panic("cannot create a data block")
	}

	return block
}

func newClaim(name string) *columbiav1.PrivacyBudgetClaim {
	claim := &columbiav1.PrivacyBudgetClaim{
		ObjectMeta: metav1.ObjectMeta{
			Name:      name,
			Namespace: namespace,
		},
	}
	claim, err := createClaim(claim)
	if err != nil {
		panic("cannot create a privacy budget claim")
	}
	return claim
}

func addAllocateRequest(claim *columbiav1.PrivacyBudgetClaim, dataset string,
	numOfBlocks int,
	maxBudget columbiav1.EpsilonDeltaBudget) *columbiav1.PrivacyBudgetClaim {

	claim.Spec.Requests = append(claim.Spec.Requests, columbiav1.Request{
		Identifier: "foo",
		AllocateRequest: &columbiav1.AllocateRequest{
			Dataset:           dataset,
			MaxNumberOfBlocks: numOfBlocks,
			MinNumberOfBlocks: numOfBlocks,
			ExpectedBudget: columbiav1.BudgetRequest{
				Constant: &columbiav1.PrivacyBudget{
					EpsDel: &maxBudget,
				},
			},
		},
	})
	claim, err := updateClaimSpec(claim)
	if err != nil {
		panic("cannot update a privacy budget claim")
	}
	return claim
}
