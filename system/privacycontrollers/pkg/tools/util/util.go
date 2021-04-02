package util

import (
	columbiav1 "columbia.github.com/sage/privacyresource/pkg/apis/columbia.github.com/v1"
	"reflect"
)

func GetBlockId(block *columbiav1.PrivateDataBlock) string {
	return block.Namespace + "/" + block.Name
}

func GetClaimId(claim *columbiav1.PrivacyBudgetClaim) string {
	return claim.Namespace + "/" + claim.Name
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

func IsFlowMode(block *columbiav1.PrivateDataBlock) bool {
	if block == nil {
		return false
	}
	return block.Spec.ReleaseMode == columbiav1.Flow && block.Spec.FlowReleasingOption != nil
}

func SetBudgetMapAndDeleteEmpty(m map[string]columbiav1.PrivacyBudget, key string, value columbiav1.PrivacyBudget) {
	if value.IsEmpty() {
		delete(m, key)
	} else {
		m[key] = value
	}
}
