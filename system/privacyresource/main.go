package main

import (
	"context"
	"encoding/json"
	"flag"
	"time"

	columbiav1 "columbia.github.com/privatekube/privacyresource/pkg/apis/columbia.github.com/v1"
	privacyclientset "columbia.github.com/privatekube/privacyresource/pkg/generated/clientset/versioned"
	privacyinformers "columbia.github.com/privatekube/privacyresource/pkg/generated/informers/externalversions"
	"github.com/shopspring/decimal"
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	_ "k8s.io/client-go/plugin/pkg/client/auth/gcp"
	"k8s.io/client-go/tools/clientcmd"
	"k8s.io/klog"
)

var (
	masterURL   string
	kubeconfig  string
	keepRunning string
	ns          = "default"
	blockName   = "block-tmp-1"
	claimName   = "claim-tmp-1"
)

func main() {
	klog.InitFlags(nil)
	flag.Parse()

	cfg, err := clientcmd.BuildConfigFromFlags(masterURL, kubeconfig)
	if err != nil {
		klog.Fatalf("Error building kubeconfig: %s", err.Error())
		return
	}

	privacyClient, err := privacyclientset.NewForConfig(cfg)
	if err != nil {
		klog.Fatalf("Error building privacy data clientset: %s", err.Error())
		return
	}

	testPrivateDataBlock(privacyClient)
	testPrivacyBudgetClaim(privacyClient)
	for keepRunning == "true" {
	}
}

func init() {
	flag.StringVar(&kubeconfig, "kubeconfig", "", "Path to a kubeconfig. Only required if out-of-cluster.")
	flag.StringVar(&masterURL, "master", "", "The address of the Kubernetes API server. Overrides any value in kubeconfig. Only required if out-of-cluster.")
	flag.StringVar(&keepRunning, "keepRunning", "", "Whether the script should keep running when everything is completed")
}

func testPrivateDataBlock(privacyClient *privacyclientset.Clientset) {
	block := &columbiav1.PrivateDataBlock{
		ObjectMeta: metav1.ObjectMeta{Name: blockName, Namespace: ns},
		Spec: columbiav1.PrivateDataBlockSpec{
			InitialBudget: columbiav1.PrivacyBudget{
				EpsDel: &columbiav1.EpsilonDeltaBudget{
					Epsilon: 1,
					Delta:   0.001,
				},
			},
			Dimensions: []columbiav1.Dimension{
				{
					Attribute:    "startTime",
					NumericValue: func() *decimal.Decimal { i := decimal.NewFromInt(0); return &i }(),
				},
				{
					Attribute:    "endTime",
					NumericValue: func() *decimal.Decimal { i := decimal.NewFromInt(100); return &i }(),
				},
				{
					Attribute:   "userId",
					StringValue: "foo",
				},
			},
			DataSource: "some_url/db/block-1",
			Dataset:    "taxi-db",
		},
	}
	block, err := privacyClient.ColumbiaV1().PrivateDataBlocks(ns).Create(context.TODO(), block, metav1.CreateOptions{})
	if err != nil {
		klog.Errorf("Error creating: %v", err)
	} else {
		// try to update the block
		block.Spec.Dimensions = append(block.Spec.Dimensions, columbiav1.Dimension{
			Attribute:    "extra",
			NumericValue: func() *decimal.Decimal { i := decimal.NewFromInt(5000); return &i }(),
		})

		if _, err = privacyClient.ColumbiaV1().PrivateDataBlocks(ns).Update(context.TODO(), block, metav1.UpdateOptions{}); err != nil {
			klog.Errorf("Failed to update the data block: %v", err)
		}
	}

	ls, err := privacyClient.ColumbiaV1().PrivateDataBlocks(ns).List(context.TODO(), metav1.ListOptions{})
	if err != nil {
		klog.Errorf("Error querying list: %s", err.Error())
	}

	if ls == nil || len(ls.Items) == 0 {
		klog.Warning("list is nil or list is empty")
	} else {
		block = &ls.Items[0]
		klog.Info(block.Spec)
		klog.Info(block.Status)
	}

	block.Status.AcquiredBudgetMap = make(map[string]columbiav1.PrivacyBudget)
	block.Status.AcquiredBudgetMap[claimName] = columbiav1.PrivacyBudget{
		Renyi: columbiav1.RenyiBudget{
			{2, 1},
			{3, 2},
			{4.5, 5},
		},
	}

	block.Status.LockedBudgetMap = map[string]columbiav1.LockedBudget{}
	block.Status.LockedBudgetMap[claimName] = columbiav1.LockedBudget{
		Budget: columbiav1.BudgetAccount{
			Acquired: columbiav1.PrivacyBudget{
				EpsDel: &columbiav1.EpsilonDeltaBudget{
					Epsilon: 1,
					Delta:   0.001,
				},
			},
		},
		State: columbiav1.Allocating,
	}

	blockPrivacyBudgetCheck(block)
	block, err = privacyClient.ColumbiaV1().PrivateDataBlocks(ns).UpdateStatus(context.TODO(), block, metav1.UpdateOptions{})
	if err != nil {
		klog.Errorf("unable to update status of data block: %v", err)
	} else {
		klog.Info(block.Status)
	}

	// Test if the informer and lister are sync.
	// We create an informer, start the informer, wait for sync, and list the blocks using lister
	privateDataBlockInformerFactory := privacyinformers.NewSharedInformerFactory(privacyClient, time.Second*30)
	privateDataBlockInformerFactory.Columbia().V1().PrivateDataBlocks().Informer()

	privateDataBlockInformerFactory.Start(nil)
	privateDataBlockInformerFactory.WaitForCacheSync(nil)

	lister := privateDataBlockInformerFactory.Columbia().V1().PrivateDataBlocks().Lister()
	res, err := lister.PrivateDataBlocks(ns).List(labels.Everything())
	if err != nil {
		klog.Errorf("lister failed: %v", err)
	} else {
		klog.Infof("lister get: %v", res)
	}
	var gracePeriod int64 = 0
	err = privacyClient.ColumbiaV1().PrivateDataBlocks(ns).Delete(context.TODO(), block.Name,
		metav1.DeleteOptions{GracePeriodSeconds: &gracePeriod})
	if err != nil {
		klog.Errorf("Error delete block: %v", err)
	} else {
		klog.Infof("Succeed to delete: %s", block.Name)
	}
}

func testPrivacyBudgetClaim(privacyClient *privacyclientset.Clientset) {
	claim := &columbiav1.PrivacyBudgetClaim{
		ObjectMeta: metav1.ObjectMeta{Name: claimName, Namespace: ns},
		Spec: columbiav1.PrivacyBudgetClaimSpec{
			Requests: []columbiav1.Request{},
			OwnedBy:  []v1.ObjectReference{},
		},
	}

	claim.Spec.Requests = append(claim.Spec.Requests, columbiav1.Request{
		AllocateRequest: &columbiav1.AllocateRequest{
			Dataset: "taxi",
			Conditions: []columbiav1.Condition{
				{
					Attribute:    "startTime",
					Operation:    ">",
					NumericValue: func() *decimal.Decimal { i := decimal.NewFromInt(12); return &i }(),
				},
				{
					Attribute:    "foo",
					Operation:    "=",
					NumericValue: func() *decimal.Decimal { i := decimal.NewFromFloat(123.23); return &i }(),
				},
				{
					Attribute:   "bar",
					Operation:   "is",
					StringValue: "fake",
				},
			},
			MinNumberOfData:   0,
			MinNumberOfBlocks: 0,
			MinBudget: columbiav1.BudgetRequest{
				Constant: &columbiav1.PrivacyBudget{
					EpsDel: &columbiav1.EpsilonDeltaBudget{
						Epsilon: 1,
						Delta:   0.001,
					},
				},
			},
			ExpectedBudget: columbiav1.BudgetRequest{
				Function: "exponential",
			},
		},
	})

	claim.Spec.Requests = append(claim.Spec.Requests, columbiav1.Request{
		ConsumeRequest: &columbiav1.ConsumeRequest{
			Consume: map[string]columbiav1.PrivacyBudget{
				blockName: {
					Renyi: columbiav1.RenyiBudget{
						{2, 2},
						{3, 3},
					},
				},
			},
		},
	})

	claim.Spec.Requests = append(claim.Spec.Requests, columbiav1.Request{
		ReleaseRequest: &columbiav1.ReleaseRequest{
			Release: map[string]columbiav1.PrivacyBudget{
				blockName: {
					EpsDel: &columbiav1.EpsilonDeltaBudget{
						Epsilon: 1,
						Delta:   1,
					},
				},
			},
		},
	})

	claim, err := privacyClient.ColumbiaV1().PrivacyBudgetClaims(ns).Create(context.TODO(), claim, metav1.CreateOptions{})
	if err != nil {
		klog.Errorf("Error creating: %s", err.Error())
	}

	if claim != nil {
		claim.Status.Responses = []columbiav1.Response{
			{
				State: columbiav1.Success,
				AllocateResponse: &columbiav1.AllocateResponse{
					Budgets: map[string]columbiav1.BudgetAccount{
						blockName: {
							Acquired: columbiav1.PrivacyBudget{
								EpsDel: &columbiav1.EpsilonDeltaBudget{
									Epsilon: 1,
									Delta:   1,
								},
							},
							Reserved: columbiav1.PrivacyBudget{
								EpsDel: &columbiav1.EpsilonDeltaBudget{
									Epsilon: 1,
									Delta:   1,
								},
							},
						},
						"block-2": {
							Reserved: columbiav1.PrivacyBudget{
								EpsDel: &columbiav1.EpsilonDeltaBudget{
									Epsilon: 1,
									Delta:   1,
								},
							},
						},
					},
				},
			},
			{
				State: columbiav1.Success,
				ConsumeResponse: columbiav1.ConsumeResponse{
					blockName: {
						Budget: columbiav1.PrivacyBudget{
							Renyi: columbiav1.RenyiBudget{
								{2, 2},
								{3, 3},
							},
						},
						Error: "",
					},
				},
			},
			{
				State: columbiav1.Success,
				ReleaseResponse: columbiav1.ReleaseResponse{
					blockName: {
						Budget: columbiav1.PrivacyBudget{
							EpsDel: &columbiav1.EpsilonDeltaBudget{
								Epsilon: 1,
								Delta:   1,
							},
						},
						Error: "",
					},
				},
			},
		}
		claim, err = privacyClient.ColumbiaV1().PrivacyBudgetClaims(ns).UpdateStatus(context.TODO(), claim, metav1.UpdateOptions{})
		if err != nil {
			klog.Errorf("fail to update status of claim: %v", err)
		}
	}

	ls, err := privacyClient.ColumbiaV1().PrivacyBudgetClaims(ns).List(context.TODO(), metav1.ListOptions{})
	if err != nil {
		klog.Errorf("Error querying list: %s", err)
	}

	if ls == nil || len(ls.Items) == 0 {
		klog.Warning("list is nil or list is empty")
	} else {
		claim = &ls.Items[0]
		printJson(claim)
	}

	// Test if the informer and lister are sync.
	// We create an informer, start the informer, wait for sync, and list the blocks using lister
	privacyBudgetClaimInformerFactory := privacyinformers.NewSharedInformerFactory(privacyClient, time.Second*30)
	privacyBudgetClaimInformerFactory.Columbia().V1().PrivacyBudgetClaims().Informer()

	privacyBudgetClaimInformerFactory.Start(nil)
	privacyBudgetClaimInformerFactory.WaitForCacheSync(nil)

	lister := privacyBudgetClaimInformerFactory.Columbia().V1().PrivacyBudgetClaims().Lister()
	res, err := lister.PrivacyBudgetClaims(ns).List(labels.Everything())
	if err != nil {
		klog.Errorf("lister failed: %v", err)
	} else {
		klog.Infof("lister get: %v", res)
	}

	err = privacyClient.ColumbiaV1().PrivacyBudgetClaims(ns).Delete(context.TODO(), claim.Name, metav1.DeleteOptions{})
	if err != nil {
		klog.Errorf("Error delete block: %v", err)
	} else {
		klog.Infof("Succeed to delete: %s", claim.Name)
	}

}

func blockPrivacyBudgetCheck(block *columbiav1.PrivateDataBlock) {
	// block.Spec.InitialBudget.InitIfNil()
	// block.Status.PendingBudget.InitIfNil()
	// block.Status.AvailableBudget.InitIfNil()

}

func printJson(v interface{}) {
	s, err := json.MarshalIndent(v, "", "\t")
	if err == nil {
		klog.Infof(string(s))
	}
}
