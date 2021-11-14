package test

import (
	"testing"
	"time"

	columbiav1 "columbia.github.com/privatekube/privacyresource/pkg/apis/columbia.github.com/v1"
	"github.com/shopspring/decimal"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

func TestFairness(t *testing.T) {
	startDefaultScheduler()

	unit := 1e-2
	initialBudget := columbiav1.PrivacyBudget{
		EpsDel: &columbiav1.EpsilonDeltaBudget{
			Epsilon: 101 * unit,
			Delta:   0.01,
		},
	}

	demandBudget1 := []columbiav1.PrivacyBudget{
		{
			EpsDel: &columbiav1.EpsilonDeltaBudget{
				Epsilon: 0.5 * unit,
				Delta:   1e-9,
			},
		},
		{
			EpsDel: &columbiav1.EpsilonDeltaBudget{
				Epsilon: 1.5 * unit,
				Delta:   1e-9,
			},
		},
	}

	demandBudget2 := []columbiav1.PrivacyBudget{
		{
			EpsDel: &columbiav1.EpsilonDeltaBudget{
				Epsilon: 1 * unit,
				Delta:   1e-9,
			},
		},
		{
			EpsDel: &columbiav1.EpsilonDeltaBudget{
				Epsilon: 1 * unit,
				Delta:   1e-9,
			},
		},
	}

	demandBudget3 := []columbiav1.PrivacyBudget{
		{
			EpsDel: &columbiav1.EpsilonDeltaBudget{
				Epsilon: 1.6 * unit,
				Delta:   1e-9,
			},
		},
		{
			EpsDel: &columbiav1.EpsilonDeltaBudget{
				Epsilon: 1 * unit,
				Delta:   1e-9,
			},
		},
	}

	dataset := "test"
	blockName1 := "block-1"
	blockName2 := "block-2"

	toDecimal := func(i int) *decimal.Decimal {
		d := decimal.New(int64(i), 0)
		return &d
	}

	block := &columbiav1.PrivateDataBlock{
		ObjectMeta: metav1.ObjectMeta{
			Name:      blockName1,
			Namespace: namespace,
		},
		Spec: columbiav1.PrivateDataBlockSpec{
			InitialBudget: initialBudget,
			Dataset:       dataset,
			Dimensions: []columbiav1.Dimension{
				{
					Attribute:    "startTime",
					NumericValue: toDecimal(1),
				},
				{
					Attribute:    "endTime",
					NumericValue: toDecimal(2),
				},
			},
		},
	}
	_, _ = createDataBlock(block)

	block = &columbiav1.PrivateDataBlock{
		ObjectMeta: metav1.ObjectMeta{
			Name:      blockName2,
			Namespace: namespace,
		},
		Spec: columbiav1.PrivateDataBlockSpec{
			InitialBudget: initialBudget,
			Dataset:       dataset,
			Dimensions: []columbiav1.Dimension{
				{
					Attribute:    "startTime",
					NumericValue: toDecimal(1),
				},
				{
					Attribute:    "endTime",
					NumericValue: toDecimal(2),
				},
			},
		},
	}
	_, _ = createDataBlock(block)

	claim1 := &columbiav1.PrivacyBudgetClaim{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "claim-1",
			Namespace: namespace,
		},
		Spec: columbiav1.PrivacyBudgetClaimSpec{
			Requests: []columbiav1.Request{
				{
					Identifier: "1",
					AllocateRequest: &columbiav1.AllocateRequest{
						Dataset: dataset,
						ExpectedBudget: columbiav1.BudgetRequest{
							BudgetMap: map[string]columbiav1.PrivacyBudget{
								namespace + "/" + blockName1: demandBudget1[0],
								namespace + "/" + blockName2: demandBudget1[1],
							},
						},
					},
				},
			},
		},
	}

	_, _ = createClaim(claim1)

	claim2 := &columbiav1.PrivacyBudgetClaim{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "claim-2",
			Namespace: namespace,
		},
		Spec: columbiav1.PrivacyBudgetClaimSpec{
			Requests: []columbiav1.Request{
				{
					Identifier: "1",
					AllocateRequest: &columbiav1.AllocateRequest{
						Dataset: dataset,
						ExpectedBudget: columbiav1.BudgetRequest{
							BudgetMap: map[string]columbiav1.PrivacyBudget{
								namespace + "/" + blockName1: demandBudget2[0],
								namespace + "/" + blockName2: demandBudget2[1],
							},
						},
					},
				},
			},
		},
	}

	_, _ = createClaim(claim2)

	claim3 := &columbiav1.PrivacyBudgetClaim{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "claim-3",
			Namespace: namespace,
		},
		Spec: columbiav1.PrivacyBudgetClaimSpec{
			Requests: []columbiav1.Request{
				{
					Identifier: "1",
					AllocateRequest: &columbiav1.AllocateRequest{
						Dataset: dataset,
						ExpectedBudget: columbiav1.BudgetRequest{
							BudgetMap: map[string]columbiav1.PrivacyBudget{
								namespace + "/" + blockName1: demandBudget3[0],
								namespace + "/" + blockName2: demandBudget3[1],
							},
						},
					},
				},
			},
		},
	}

	_, _ = createClaim(claim3)

	time.Sleep(10 * time.Second)

	claim1 = getClaim(claim1.Name)
	printJson(t, claim1)
	if len(claim1.Status.Responses) != 1 || claim1.Status.Responses[0].State != columbiav1.Success {
		t.Fatal("claim-1 was not allocated successfully")
	}

	claim2 = getClaim(claim2.Name)
	printJson(t, claim2)
	if len(claim2.Status.Responses) != 1 || claim2.Status.Responses[0].State != columbiav1.Success {
		t.Fatal("claim-2 was not allocated successfully")
	}

	claim3 = getClaim(claim3.Name)
	printJson(t, claim3)
	if len(claim3.Status.Responses) != 1 || claim3.Status.Responses[0].State != columbiav1.Failure {
		t.Fatal("claim-3 should not be allocated successfully")
	}

	if claim1.Status.Responses[0].AllocateResponse.FinishTime <=
		claim2.Status.Responses[0].AllocateResponse.FinishTime {
		t.Fatal("claim-1 should be allocated after claim-2")
	}
}
