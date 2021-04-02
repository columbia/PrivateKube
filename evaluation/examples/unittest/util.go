package test

import (
	"encoding/json"
	"testing"

	v1 "columbia.github.com/sage/privacyresource/pkg/apis/columbia.github.com/v1"
	"github.com/stretchr/testify/assert"
)

func PrivacyBudgetEqual(t *testing.T, a, b v1.EpsilonDeltaBudget) {
	assert.InDelta(t, a.Epsilon, b.Epsilon, 1e-12)
	assert.InDelta(t, a.Delta, b.Delta, 1e-12)
}

func printJson(t *testing.T, v interface{}) {
	s, err := json.MarshalIndent(v, "", "\t")
	if err == nil {
		t.Log(string(s))
	}
}
