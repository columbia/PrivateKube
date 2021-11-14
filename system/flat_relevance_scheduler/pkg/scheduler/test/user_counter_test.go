package test

import (
	"fmt"
	"testing"
	"time"

	"columbia.github.com/privatekube/dpfscheduler/pkg/scheduler/util"
	columbiav1 "columbia.github.com/privatekube/privacyresource/pkg/apis/columbia.github.com/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/klog"
)

const N_BLOCKS = 10

func TestUserCounterWithoutCounts(t *testing.T) {
	klog.Info("Starting the controllers and informers.")
	startDefaultSchedulerWithCounter()

	dataset := "fake-set"

	klog.Info("Starting to create blocks.")
	for i := 0; i < N_BLOCKS; i++ {
		block := &columbiav1.PrivateDataBlock{
			ObjectMeta: metav1.ObjectMeta{
				Name:      fmt.Sprintf("block-%d", i),
				Namespace: namespace,
			},
			Spec: columbiav1.PrivateDataBlockSpec{
				InitialBudget: columbiav1.PrivacyBudget{
					EpsDel: &initialBudget,
				},
				Dataset: dataset,
				Dimensions: []columbiav1.Dimension{
					{
						Attribute:    "startTime",
						NumericValue: util.ToDecimal(i),
					},
					{
						Attribute:    "endTime",
						NumericValue: util.ToDecimal(i + 1),
					},
				},
			},
		}
		_, _ = createDataBlock(block)
	}
}

func TestUserCounterWithCounts(t *testing.T) {
	klog.Info("Starting the controllers and informers.")
	startDefaultSchedulerWithCounter()

	dataset := "fake-set"

	klog.Info("Starting to create blocks.")
	for i := 0; i < N_BLOCKS; i++ {
		block := &columbiav1.PrivateDataBlock{
			ObjectMeta: metav1.ObjectMeta{
				Name:      fmt.Sprintf("block-%d", i),
				Namespace: namespace,
			},
			Spec: columbiav1.PrivateDataBlockSpec{
				InitialBudget: columbiav1.PrivacyBudget{
					EpsDel: &initialBudget,
				},
				Dataset: dataset,
				Dimensions: []columbiav1.Dimension{
					{
						Attribute:    "startTime",
						NumericValue: util.ToDecimal(i),
					},
					{
						Attribute:    "endTime",
						NumericValue: util.ToDecimal(i + 1),
					},
					{
						Attribute:    "NNewUsers",
						NumericValue: util.ToDecimal(10),
					},
					{
						Attribute:    "NTicks",
						NumericValue: util.ToDecimal(100),
					},
				},
			},
		}
		_, _ = createDataBlock(block)
	}

	time.Sleep(1 * time.Second)

	fmt.Println("Blocks after the counter:")
	for i := 0; i < N_BLOCKS; i++ {
		block := getDataBlock(fmt.Sprintf("block-%d", i))
		fmt.Println(block)
	}

}

func TestUserCounterRDPCurve(t *testing.T) {
	klog.Info("Starting the controllers and informers.")
	startDefaultSchedulerWithCounter()

	dataset := "fake-set"

	klog.Info("Starting to create blocks.")
	for i := 0; i < N_BLOCKS; i++ {
		block := &columbiav1.PrivateDataBlock{
			ObjectMeta: metav1.ObjectMeta{
				Name:      fmt.Sprintf("block-%d", i),
				Namespace: namespace,
			},
			Spec: columbiav1.PrivateDataBlockSpec{
				InitialBudget: columbiav1.NewPrivacyBudget(10, 1e-7, true),
				Dataset:       dataset,
				Dimensions: []columbiav1.Dimension{
					{
						Attribute:    "startTime",
						NumericValue: util.ToDecimal(i),
					},
					{
						Attribute:    "endTime",
						NumericValue: util.ToDecimal(i + 1),
					},
					{
						Attribute:    "NNewUsers",
						NumericValue: util.ToDecimal(20),
					},
					{
						Attribute:    "NTicks",
						NumericValue: util.ToDecimal(100),
					},
				},
			},
		}
		_, _ = createDataBlock(block)
	}

	time.Sleep(1 * time.Second)

	fmt.Println("Blocks after the counter:")
	// for i := 0; i < N_BLOCKS; i++ {
	// 	block := getDataBlock(fmt.Sprintf("block-%d", i))
	// 	fmt.Println(block)
	// }

	block := getDataBlock(fmt.Sprintf("block-%d", N_BLOCKS-1))
	fmt.Println(block)
}
