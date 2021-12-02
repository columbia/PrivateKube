package util

import (
	"reflect"

	columbiav1 "columbia.github.com/privatekube/privacyresource/pkg/apis/columbia.github.com/v1"
	"columbia.github.com/privatekube/privacyresource/pkg/framework"
	"github.com/shopspring/decimal"
	distuv "gonum.org/v1/gonum/stat/distuv"
)

const (
	DPF            = "DPF"
	FLAT_RELEVANCE = "FLAT_RELEVANCE"
)

func GetBlockId(block *columbiav1.PrivateDataBlock) string {
	return block.Namespace + "/" + block.Name
}

func GetClaimId(claim *columbiav1.PrivacyBudgetClaim) string {
	return claim.Namespace + "/" + claim.Name
}

func GetRequestId(claim *columbiav1.PrivacyBudgetClaim, requestIndex int) string {
	return framework.GetRequestId(claim, requestIndex)
}

func SetBudgetMapAndDeleteEmpty(m map[string]columbiav1.PrivacyBudget, key string, value columbiav1.PrivacyBudget) {
	if value.IsEmpty() {
		delete(m, key)
	} else {
		m[key] = value
	}
}

func GetStringKeysFromMap(m interface{}) []string {
	v := reflect.ValueOf(m)
	if v.Kind() != reflect.Map || v.Type().Key().Kind() != reflect.String {
		return nil
	}
	reflectedKeys := v.MapKeys()

	keys := make([]string, len(reflectedKeys))
	i := 0
	for _, reflectedKey := range reflectedKeys {
		keys[i] = reflectedKey.String()
		i++
	}
	return keys
}

func DeleteLock(block *columbiav1.PrivateDataBlock, requestViewer framework.RequestViewer) {
	delete(block.Status.LockedBudgetMap, requestViewer.Id)
}

func ToDecimal(i int) *decimal.Decimal {
	d := decimal.New(int64(i), 0)
	return &d
}

// LaplaceNoise generates noise from an exponential distribution with parameter lambda = noise_scale
// This is not a production-ready noise generation process
func LaplaceNoise(noise_scale float64) float64 {
	l := distuv.Laplace{
		Mu:    0,
		Scale: noise_scale,
	}
	return l.Rand()
}
