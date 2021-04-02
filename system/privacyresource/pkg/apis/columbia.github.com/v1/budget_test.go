package v1

import (
	"math"
	"testing"
)

func TestRenyiBudget_Add(t *testing.T) {
	a := RenyiBudget{
		{1.5, 10},
		{6, 9},
		{1048576, 3},
	}

	b := RenyiBudget{
		{1.5, 10},
		{8, 7},
		{777, 5},
		{1048576, 3},
	}

	res := a.Add(b)
	eps := [...]float64{20, 6}

	// NOTE(Pierre): I disagree with this semantic
	// Alternatively: linear interpolation
	// eps := [...]float64{20, 0, 0, 0, 0, 0, 0, 9, 7, 0, 0, 0, 6}

	if len(res) != len(eps) {
		t.Fatal("output size is incorrect")
	}

	for i := range res {
		if res[i].Epsilon != eps[i] {
			t.Fatal("epsilon incorrect")
		}
	}
}

func TestRenyi_Minus(t *testing.T) {
	a := RenyiBudget{
		{1.5, 10},
		{6, 9},
		{1048576, 3},
	}

	b := RenyiBudget{
		{1.5, 8.5},
		{6.000000000000001, 7},
		{8, 7},
		{777, 5},
		{1048576, 4},
	}

	res := a.Minus(b)
	eps := [...]float64{1.5, 2, -1}
	// eps := [...]float64{1.5, 0, 0, 0, 0, 0, 0, 9, -7, 0, 0, 0, -1}

	if len(res) != len(eps) {
		t.Fatal("output size is incorrect")
	}

	for i := range res {
		if res[i].Epsilon != eps[i] {
			t.Fatal("epsilon incorrect")
		}
	}
}

func TestRenyi_IAdd(t *testing.T) {
	a := RenyiBudget{
		{1.5, 10.1},
		{6, 9},
		{1048576, 3},
	}

	b := RenyiBudget{
		{1.5, 10},
		{8, 7},
		{777, 5},
		{1048576, 3},
	}

	a.IAdd(b)
	eps := [...]float64{20.1, 6}

	// NOTE(Pierre): I disagree with this semantic
	// eps := [...]float64{20, 0, 0, 0, 0, 0, 0, 9, 7, 0, 0, 0, 6}

	if len(a) != len(eps) {
		t.Fatal("output size is incorrect")
	}

	for i := range a {
		if a[i].Epsilon != eps[i] {
			t.Fatal("epsilon incorrect")
		}
	}
}

func TestRenyi_ISub(t *testing.T) {
	a := RenyiBudget{
		{1.5, 10},
		{6, 9},
		{1048576, 5},
	}

	b := RenyiBudget{
		{1.5, 8.5},
		{8, 7},
		{777, 5},
		{1048576, 4},
	}

	a.ISub(b)
	eps := [...]float64{1.5, 1}
	// eps := [...]float64{1.5, 0, 0, 0, 0, 0, 0, 9, -7, 0, 0, 0, -1}

	if len(a) != len(eps) {
		t.Fatal("output size is incorrect")
	}

	for i := range a {
		if a[i].Epsilon != eps[i] {
			t.Fatal("epsilon incorrect")
		}
	}
}

func TestRenyi_Mul(t *testing.T) {
	a := RenyiBudget{
		{1.5, 1},
		{6, 2},
		{1048576, 4},
	}

	f := 2.0

	res := a.Mul(f)
	eps := [...]float64{2, 4, 8}
	// eps := [...]float64{1.5, 0, 0, 0, 0, 0, 0, 9, -7, 0, 0, 0, -1}

	if len(a) != len(eps) {
		t.Fatal("output size is incorrect")
	}

	for i := range res {
		if math.Abs(res[i].Epsilon-eps[i]) > 1e-12 {
			t.Fatal("epsilon incorrect")
		}
	}
}

func TestEpsilonDeltaBudget_ToRenyi(t *testing.T) {
	epsDel := EpsilonDeltaBudget{
		Epsilon: 1,
		Delta:   1e-3,
	}

	renyi := epsDel.ToRenyi()
	eps := [...]float64{0, 0, 0, 0, 0, 0, 0, 0, 0.0131778, 0.5394829, 0.7771691, 0.890353090, 1}

	if len(renyi) != len(eps) {
		t.Fatal("output size is incorrect")
	}

	for i := range renyi {
		if math.Abs(renyi[i].Epsilon-eps[i]) >= 0.0001 {
			t.Fatal("epsilon incorrect")
		}
	}
}
