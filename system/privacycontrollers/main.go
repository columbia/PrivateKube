package main

import (
	"columbia.github.com/sage/privacycontrollers/pkg/blockclaimsync"
	"columbia.github.com/sage/privacycontrollers/pkg/datablocklifecycle"
	privacyclientset "columbia.github.com/sage/privacyresource/pkg/generated/clientset/versioned"
	privacyinformers "columbia.github.com/sage/privacyresource/pkg/generated/informers/externalversions"
	"context"
	"flag"
	"k8s.io/client-go/tools/clientcmd"
	"k8s.io/klog"
	"time"

	// uncomment it if running with gcloud
	_ "k8s.io/client-go/plugin/pkg/client/auth/gcp"
)

var (
	masterURL  string
	kubeconfig string
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

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	syncControllerOption := blockclaimsync.SyncControllerOption{PeriodSecond: 10}
	syncController := blockclaimsync.NewSynController(privacyResourceClient, privateDataBlockInformer, privacyBudgetClaimInformer, syncControllerOption)
	klog.Info("Sync Controller is ready to run")

	blockLifecycleOption := datablocklifecycle.BlockLifecycleOption{
		SchedulingPolicy:           datablocklifecycle.DpfN,
		DeletionGracePeriodSeconds: 30,
	}

	blockLifecycleController := datablocklifecycle.NewBlockController(privacyResourceClient, blockLifecycleInformer, blockLifecycleOption)
	klog.Info("Block Lifecycle Controller is ready to run")

	privacyResourceInformerFactory.Start(ctx.Done())
	syncController.Run()
	blockLifecycleController.Run()

	for {
		time.Sleep(time.Second)
	}
}

func init() {
	flag.StringVar(&kubeconfig, "kubeconfig", "", "Path to a kubeconfig. Only required if out-of-cluster.")
	flag.StringVar(&masterURL, "master", "", "The address of the Kubernetes API server. Overrides any value in kubeconfig. Only required if out-of-cluster.")
}
