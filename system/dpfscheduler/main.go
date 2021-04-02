package main

import (
	"columbia.github.com/sage/dpfscheduler/pkg/scheduler"
	"columbia.github.com/sage/privacycontrollers/pkg/blockclaimsync"
	"columbia.github.com/sage/privacycontrollers/pkg/datablocklifecycle"

	//"columbia.github.com/sage/privacycontrollers/pkg/datablocklifecycle"
	"context"
	"flag"
	"time"

	corev1 "k8s.io/api/core/v1"
	"k8s.io/client-go/kubernetes/scheme"
	"k8s.io/client-go/tools/clientcmd"
	"k8s.io/client-go/tools/record"
	"k8s.io/klog"

	privacyclientset "columbia.github.com/sage/privacyresource/pkg/generated/clientset/versioned"
	privacyinformers "columbia.github.com/sage/privacyresource/pkg/generated/informers/externalversions"

	// uncomment it if running with gcloud
	_ "k8s.io/client-go/plugin/pkg/client/auth/gcp"
)

var (
	masterURL         string
	kubeconfig        string
	mode              int
	n                 int
	timeout           int64
	releasingDuration int64
	releasingPeriod   int
)

func main() {
	klog.InitFlags(nil)
	flag.Parse()

	cfg, err := clientcmd.BuildConfigFromFlags(masterURL, kubeconfig)
	if err != nil {
		klog.Fatalf("Error building kubeconfig: %s", err.Error())
		return
	}

	privacyResourceClient, err := privacyclientset.NewForConfig(cfg)
	if err != nil {
		klog.Fatalf("Error building privacy privacyclientset: %s", err.Error())
		return
	}

	privacyResourceInformerFactory := privacyinformers.NewSharedInformerFactory(privacyResourceClient, time.Second*30)

	privateDataBlockInformer := privacyResourceInformerFactory.Columbia().V1().PrivateDataBlocks()
	privacyBudgetClaimInformer := privacyResourceInformerFactory.Columbia().V1().PrivacyBudgetClaims()

	blockLifecycleInformer := privacyResourceInformerFactory.Columbia().V1().PrivateDataBlocks()

	syncControllerBlockInformer := privacyResourceInformerFactory.Columbia().V1().PrivateDataBlocks()
	syncControllerClaimInformer := privacyResourceInformerFactory.Columbia().V1().PrivacyBudgetClaims()

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	coreBroadcaster := record.NewBroadcaster()
	coreRecorder := coreBroadcaster.NewRecorder(scheme.Scheme, corev1.EventSource{Component: "privacyscheduler"})

	var privacyScheduler *scheduler.DpfScheduler
	var blockLifecycleOption datablocklifecycle.BlockLifecycleOption

	if mode == scheduler.TScheme {
		option := scheduler.DefaultTSchemeOption()
		if releasingPeriod >= 0 {
			option.DefaultReleasingPeriod = releasingPeriod
		}

		if releasingDuration >= 0 {
			option.DefaultReleasingDuration = releasingDuration
		}

		if timeout >= 0 {
			option.DefaultTimeout = timeout
		}

		privacyScheduler, _ = scheduler.New(privacyResourceClient, privateDataBlockInformer,
			privacyBudgetClaimInformer, coreRecorder, ctx.Done(), option)

		blockLifecycleOption = datablocklifecycle.BlockLifecycleOption{
			SchedulingPolicy:           datablocklifecycle.DpfT,
			DeletionGracePeriodSeconds: 30,
		}
	} else {
		option := scheduler.DefaultNSchemeOption()
		if n > 0 {
			option.N = n
		}

		if timeout >= 0 {
			option.DefaultTimeout = timeout
		}

		privacyScheduler, _ = scheduler.New(privacyResourceClient, privateDataBlockInformer,
			privacyBudgetClaimInformer, coreRecorder, ctx.Done(), option)

		blockLifecycleOption = datablocklifecycle.BlockLifecycleOption{
			SchedulingPolicy:           datablocklifecycle.DpfN,
			DeletionGracePeriodSeconds: 30,
		}
	}

	blockLifecycleController := datablocklifecycle.NewBlockController(privacyResourceClient, blockLifecycleInformer, blockLifecycleOption)
	klog.Info("Block Lifecycle Controller is ready to run")

	syncControllerOption := blockclaimsync.SyncControllerOption{PeriodSecond: 60}
	syncController := blockclaimsync.NewSynController(privacyResourceClient, syncControllerBlockInformer, syncControllerClaimInformer, syncControllerOption)
	klog.Info("Sync Controller is ready to run")

	// notice that there is no need to run Start methods in a separate goroutine. (i.e. go kubeInformerFactory.Start(stopCh)
	// Start method is non-blocking and runs all registered informers in a dedicated goroutine.
	privacyResourceInformerFactory.Start(ctx.Done())
	klog.Info("Scheduler is ready to run")

	blockLifecycleController.Run()
	syncController.Run()
	privacyScheduler.Run(ctx)

}

func init() {
	flag.StringVar(&kubeconfig, "kubeconfig", "", "Path to a kubeconfig. Only required if out-of-cluster.")
	flag.StringVar(&masterURL, "master", "", "The address of the Kubernetes API server. Overrides any value in kubeconfig. Only required if out-of-cluster.")
	flag.IntVar(&mode, "mode", scheduler.NScheme, "Scheduling Mode (Policy): 0 is DPF N, 1 is DPF T")
	flag.IntVar(&n, "n", -1, "Default N value for DPF N mode")
	flag.Int64Var(&timeout, "timeout", -1, "Default timeout for pending requests. Unit is millisecond.")
	flag.Int64Var(&releasingDuration, "releasingDuration", -1, "Default releasing duration for pending requests for DPF T mode. Unit is millisecond.")
	flag.IntVar(&releasingPeriod, "releasingPeriod", -1, "Default releasing period for pending requests for DPF T mode. Unit is sec.")
}
