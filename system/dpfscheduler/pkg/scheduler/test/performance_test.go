package test

import (
	"fmt"
	"testing"
	"time"

	columbiav1 "columbia.github.com/privatekube/privacyresource/pkg/apis/columbia.github.com/v1"
	"github.com/shopspring/decimal"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

func TestPerformance(t *testing.T) {
	startDefaultScheduler()

	dataset := "fake-set"

	toDecimal := func(i int) *decimal.Decimal {
		d := decimal.New(int64(i), 0)
		return &d
	}

	for i := 0; i < 100; i++ {
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
						NumericValue: toDecimal(i),
					},
					{
						Attribute:    "endTime",
						NumericValue: toDecimal(i + 1),
					},
				},
			},
		}
		_, _ = createDataBlock(block)
	}

	for i := 0; i < 20; i++ {
		claim := &columbiav1.PrivacyBudgetClaim{
			ObjectMeta: metav1.ObjectMeta{
				Name:      fmt.Sprintf("claim-%d", i),
				Namespace: namespace,
			},
			Spec: columbiav1.PrivacyBudgetClaimSpec{
				Requests: []columbiav1.Request{
					{
						Identifier: "1",
						AllocateRequest: &columbiav1.AllocateRequest{
							Dataset: dataset,
							Conditions: []columbiav1.Condition{
								{
									Attribute:    "startTime",
									Operation:    ">=",
									NumericValue: toDecimal(0),
								},
								{
									Attribute:    "startTime",
									Operation:    "<=",
									NumericValue: toDecimal(1000000000),
								},
							},

							//MinNumberOfBlocks: i % 10,
							//MaxNumberOfBlocks: i % 10,
							ExpectedBudget: columbiav1.BudgetRequest{
								Constant: &columbiav1.PrivacyBudget{
									EpsDel: &columbiav1.EpsilonDeltaBudget{
										Epsilon: 0.02,
										Delta:   0.00001,
									},
								},
							},
						},
					},
				},
			},
		}
		_, _ = createClaim(claim)
	}

	for {
		time.Sleep(1 * time.Second)

		ok := true
		for i := 0; i < 20; i++ {
			claim := getClaim(fmt.Sprintf("claim-%d", i))
			if len(claim.Status.Responses) == 0 || claim.Status.Responses[0].State == columbiav1.Pending {
				ok = false
				break
			}
		}

		if ok {
			break
		}

	}

	for i := 0; i < 20; i++ {
		claim := getClaim(fmt.Sprintf("claim-%d", i))
		printJson(t, claim)
	}
}
