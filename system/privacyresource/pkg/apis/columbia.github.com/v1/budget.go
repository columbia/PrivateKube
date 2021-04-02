package v1

import (
	"fmt"
	"math"
	"sort"

	"k8s.io/klog"
)

// The budget type is represented on 2 bits
// NilType corresponds to an empty budget, that is neither EpsDel nor Renyi
const (
	EmptyThreshold = 1e-12
	AlphaThreshold = EmptyThreshold
	EpsDelType     = 0b1
	RenyiType      = 0b10
	NilType        = 0
	ErrorType      = -1
)

var RenyiAlphas = [...]float64{1.5, 1.75, 2, 2.5, 3, 4, 5, 6, 8, 16, 32, 64, 1000000}

type PrivacyBudgetType int

type PrivacyBudget struct {
	EpsDel *EpsilonDeltaBudget `json:"epsDel,omitempty"`
	Renyi  RenyiBudget         `json:"renyi,omitempty"`
}

type EpsilonDeltaBudget struct {
	Epsilon float64 `json:"epsilon"`
	Delta   float64 `json:"delta"`
}

func (budget EpsilonDeltaBudget) Add(other EpsilonDeltaBudget) EpsilonDeltaBudget {
	budget.Epsilon += other.Epsilon
	budget.Delta += other.Delta
	return budget
}

func (budget EpsilonDeltaBudget) Minus(other EpsilonDeltaBudget) EpsilonDeltaBudget {
	budget.Epsilon -= other.Epsilon
	budget.Delta -= other.Delta
	return budget
}

func (budget *EpsilonDeltaBudget) IAdd(other EpsilonDeltaBudget) {
	budget.Epsilon += other.Epsilon
	budget.Delta += other.Delta
}

func (budget *EpsilonDeltaBudget) ISub(other EpsilonDeltaBudget) {
	budget.Epsilon -= other.Epsilon
	budget.Delta -= other.Delta
}

func (budget EpsilonDeltaBudget) Union(other EpsilonDeltaBudget) EpsilonDeltaBudget {
	return EpsilonDeltaBudget{
		Epsilon: math.Max(budget.Epsilon, other.Epsilon),
		Delta:   math.Max(budget.Delta, other.Delta),
	}
}

func (budget EpsilonDeltaBudget) NonNegative() EpsilonDeltaBudget {
	budget.Epsilon = math.Max(budget.Epsilon, 0)
	budget.Delta = math.Max(budget.Delta, 0)
	return budget
}

func (budget EpsilonDeltaBudget) Negative() EpsilonDeltaBudget {
	budget.Epsilon = -budget.Epsilon
	budget.Delta = -budget.Delta
	return budget
}

func (budget EpsilonDeltaBudget) Intersect(other EpsilonDeltaBudget) EpsilonDeltaBudget {
	return EpsilonDeltaBudget{
		Epsilon: math.Min(budget.Epsilon, other.Epsilon),
		Delta:   math.Min(budget.Delta, other.Delta),
	}
}

func (budget EpsilonDeltaBudget) Div(factor float64) EpsilonDeltaBudget {
	budget.Epsilon /= factor
	budget.Delta /= factor
	return budget
}

func (budget EpsilonDeltaBudget) Mul(factor float64) EpsilonDeltaBudget {
	budget.Epsilon *= factor
	budget.Delta *= factor
	return budget
}

func (budget EpsilonDeltaBudget) Equal(other EpsilonDeltaBudget) bool {
	return math.Abs(budget.Epsilon-other.Epsilon) < EmptyThreshold && math.Abs(budget.Delta-other.Delta) < EmptyThreshold
}

func (budget *EpsilonDeltaBudget) IsEmpty() bool {
	return math.Abs(budget.Epsilon) < EmptyThreshold && math.Abs(budget.Delta) < EmptyThreshold
}

func (budget *EpsilonDeltaBudget) IsNegative() bool {
	return budget.Epsilon < 0 && budget.Delta < 0
}

func (budget EpsilonDeltaBudget) HasEnough(other EpsilonDeltaBudget) bool {
	return budget.Epsilon+EmptyThreshold >= other.Epsilon && budget.Delta+EmptyThreshold >= other.Delta
}

func (budget EpsilonDeltaBudget) MoreThan(other EpsilonDeltaBudget) bool {
	return budget.Epsilon > other.Epsilon && budget.Delta > other.Delta
}

func (budget EpsilonDeltaBudget) ToRenyi() RenyiBudget {
	res := make(RenyiBudget, len(RenyiAlphas))
	logDelta := math.Log(budget.Delta)

	for i, alpha := range RenyiAlphas[:len(RenyiAlphas)-1] {
		res[i].Alpha = alpha
		if logDelta > (1-alpha)*budget.Epsilon {
			// eps_alpha = eps_old + log(delta) / (alpha - 1)
			res[i].Epsilon = budget.Epsilon + logDelta/(alpha-1)
		}
		// no need for else condition, because res[i].Epsilon = 0 by default
	}

	// alpha = 1e6 is special. It is used to represent +inf
	i := len(RenyiAlphas) - 1
	res[i].Alpha = RenyiAlphas[i]
	res[i].Epsilon = budget.Epsilon

	return res
}

type RenyiBudget []RenyiBudgetBlock

type RenyiBudgetBlock struct {
	Alpha   float64 `json:"alpha"`
	Epsilon float64 `json:"epsilon"`
}

// NewRenyiBudget initializes a budget with epsilon = 0 for each alpha of RenyiAlphas
func NewRenyiBudget() RenyiBudget {
	b := make(RenyiBudget, len(RenyiAlphas))
	for i, alpha := range RenyiAlphas {
		b[i].Alpha = alpha
		b[i].Epsilon = 0
	}
	return b
}

func (budget RenyiBudget) String() string {
	ret := "Renyi budget:"
	for _, b := range budget {
		ret += fmt.Sprintf("\n\t%f\t%f", b.Alpha, b.Epsilon)
	}
	return ret
}

// IsSorted is taken from sort package
func (budget RenyiBudget) IsSorted() bool {
	n := len(budget)
	for i := n - 1; i > 0; i-- {
		if budget[i].Alpha < budget[i-1].Alpha {
			return false
		}
	}
	return true
}

func (budget RenyiBudget) SortByAlpha() {
	if budget.IsSorted() {
		return
	}

	sort.Slice(budget, func(i, j int) bool {
		return budget[i].Alpha < budget[j].Alpha
	})
}

// ReduceToSameSupport prunes and sorts the two input budgets so that they
// use the same set of alphas (with a AlphaThreshold margin)
// The support is a *not* necessarily a subset of the tracking set RenyiAlphas
// Exception: empty budgets are considered as a budget of len(RenyiAlphas) instead of len = 0
func ReduceToSameSupport(b RenyiBudget, c RenyiBudget) (RenyiBudget, RenyiBudget) {
	b.SortByAlpha()
	c.SortByAlpha()

	// Initialize empty budgets
	if len(b) == 0 {
		b = NewRenyiBudget()
	}
	if len(c) == 0 {
		c = NewRenyiBudget()
	}

	newB := make(RenyiBudget, 0, len(b))
	newC := make(RenyiBudget, 0, len(c))

	// Browse the two budgets in order
	p, q := 0, 0
	for p < len(b) && q < len(c) {
		// If there is a match, we store the corresponding Alpha and both Epsilons
		if math.Abs(b[p].Alpha-c[q].Alpha) < AlphaThreshold {
			newB = append(newB, RenyiBudgetBlock{Alpha: b[p].Alpha, Epsilon: b[p].Epsilon})
			newC = append(newC, RenyiBudgetBlock{Alpha: b[p].Alpha, Epsilon: c[q].Epsilon})
		}
		// We only increment the index for the smallest alpha so we can't miss a match
		if b[p].Alpha <= c[q].Alpha {
			p++
		} else {
			q++
		}

	}
	return newB, newC
}

// ReduceToStandardSupport is like ReduceToSameSupport except that
// the output support is exactly RenyiAlphas and missing Epsilons are set to zero
func ReduceToStandardSupport(b RenyiBudget, c RenyiBudget) (RenyiBudget, RenyiBudget) {
	return b, c
}

// IReduceToSameSupport is an old version of ReduceToSameSupport inplace
// The support is a subset of the RenyiAlphas
func IReduceToSameSupport(b *RenyiBudget, c *RenyiBudget) {
	b.SortByAlpha()
	c.SortByAlpha()

	// Initialize empty budgets
	if len(*b) == 0 {
		*b = NewRenyiBudget()
	}
	if len(*c) == 0 {
		*c = NewRenyiBudget()
	}

	// Create new budget vectors that will replace the input budgets
	newB := make(RenyiBudget, 0, len(*b))
	newC := make(RenyiBudget, 0, len(*c))

	// Browse the tracking set
	for _, alpha := range RenyiAlphas {
		// Browse the budget b until we find a match with the tracking set (with some slack)
		p := 0
		for p < len(*b) && (*b)[p].Alpha <= alpha-EmptyThreshold/2 {
			p++
		}
		if p < len(*b) && (*b)[p].Alpha < alpha+EmptyThreshold/2 {
			// Now that we have a match on b, check if we have a match on c
			q := 0
			for q < len(*c) && (*c)[q].Alpha <= alpha-EmptyThreshold/2 {
				q++
			}
			if q < len(*c) && (*c)[q].Alpha < alpha+EmptyThreshold/2 {
				// Both budgets have a match for this alpha, we store them
				newB = append(newB, RenyiBudgetBlock{Alpha: alpha, Epsilon: (*b)[p].Epsilon})
				newC = append(newC, RenyiBudgetBlock{Alpha: alpha, Epsilon: (*c)[q].Epsilon})
			}
		}
	}
	*b = newB
	*c = newC
}

// Add sums the budgets over the intersection of their support
func (budget RenyiBudget) Add(other RenyiBudget) RenyiBudget {
	b, c := ReduceToSameSupport(budget, other)
	res := make(RenyiBudget, len(b))
	for i := range b {
		res[i].Alpha = b[i].Alpha
		res[i].Epsilon = b[i].Epsilon + c[i].Epsilon
	}
	return res
}

func (budget RenyiBudget) OldAdd(other RenyiBudget) RenyiBudget {
	klog.Infof("Summing budgets %s %s", budget, other)
	res := make(RenyiBudget, len(RenyiAlphas))
	p, q := 0, 0
	for i, alpha := range RenyiAlphas {
		res[p].Alpha = alpha

		// Assume both budget and other are sorted by alpha.
		for p < len(budget) && alpha > budget[p].Alpha {
			p++
		}

		for q < len(other) && alpha > other[q].Alpha {
			q++
		}

		if p < len(budget) && alpha == budget[p].Alpha {
			res[i].Epsilon = budget[p].Epsilon
		}

		if q < len(other) && alpha == other[q].Alpha {
			res[i].Epsilon += other[q].Epsilon
		}
	}
	klog.Infof("Result %s", res)
	return res
}

// Minus returns `budget` - `other` over the intersection of their support
func (budget RenyiBudget) Minus(other RenyiBudget) RenyiBudget {
	b, c := ReduceToSameSupport(budget, other)
	res := make(RenyiBudget, len(b))
	for i := range b {
		res[i].Alpha = b[i].Alpha
		res[i].Epsilon = b[i].Epsilon - c[i].Epsilon
	}
	return res
}

func (budget RenyiBudget) OldMinus(other RenyiBudget) RenyiBudget {
	res := make(RenyiBudget, len(RenyiAlphas))
	p, q := 0, 0
	for i, alpha := range RenyiAlphas {
		res[p].Alpha = alpha

		// Assume both budget and other are sorted by alpha.
		for p < len(budget) && alpha > budget[p].Alpha {
			p++
		}

		for q < len(other) && alpha > other[q].Alpha {
			q++
		}

		if p < len(budget) && alpha == budget[p].Alpha {
			res[i].Epsilon = budget[p].Epsilon
		}

		if q < len(other) && alpha == other[q].Alpha {
			res[i].Epsilon -= other[q].Epsilon
		}
	}

	return res
}

// IAdd adds `other` to `budget` inplace
func (budget *RenyiBudget) IAdd(other RenyiBudget) {
	*budget = budget.Add(other)
}

func (budget *RenyiBudget) ISub(other RenyiBudget) {
	*budget = budget.Minus(other)
}

// Union returns the max of the budgets over the intersection of their support
func (budget RenyiBudget) Union(other RenyiBudget) RenyiBudget {
	b, c := ReduceToSameSupport(budget, other)
	res := make(RenyiBudget, len(b))
	for i := range b {
		res[i].Alpha = b[i].Alpha
		res[i].Epsilon = math.Max(b[i].Epsilon, c[i].Epsilon)
		klog.Info(res[i])
	}
	return res
}

func (budget RenyiBudget) OldUnion(other RenyiBudget) RenyiBudget {
	res := make(RenyiBudget, len(RenyiAlphas))
	p, q := 0, 0
	for i, alpha := range RenyiAlphas {
		res[p].Alpha = alpha

		// Assume both budget and other are sorted by alpha.
		for p < len(budget) && alpha > budget[p].Alpha {
			p++
		}

		for q < len(other) && alpha > other[q].Alpha {
			q++
		}

		if p < len(budget) && alpha == budget[p].Alpha {
			res[i].Epsilon = budget[p].Epsilon
		}

		if q < len(other) && alpha == other[q].Alpha {
			res[i].Epsilon = math.Max(res[i].Epsilon, other[q].Epsilon)
		}
	}

	return res
}

func (budget RenyiBudget) NonNegative() RenyiBudget {
	res := make(RenyiBudget, len(budget))
	for p := 0; p < len(budget); p++ {
		res[p].Alpha = budget[p].Alpha
		res[p].Epsilon = math.Max(budget[p].Epsilon, 0)
	}
	return res
}

func (budget RenyiBudget) Negative() RenyiBudget {
	res := make(RenyiBudget, len(budget))
	for p := 0; p < len(budget); p++ {
		res[p].Alpha = budget[p].Alpha
		res[p].Epsilon = -budget[p].Epsilon
	}
	return res
}

// Intersect returns the min of the budgets over the intersection of their support
func (budget RenyiBudget) Intersect(other RenyiBudget) RenyiBudget {
	b, c := ReduceToSameSupport(budget, other)
	res := make(RenyiBudget, len(b))
	for i := range b {
		res[i].Alpha = b[i].Alpha
		res[i].Epsilon = math.Min(b[i].Epsilon, c[i].Epsilon)
	}
	return res
}

func (budget RenyiBudget) OldIntersect(other RenyiBudget) RenyiBudget {
	res := make(RenyiBudget, len(RenyiAlphas))
	p, q := 0, 0
	for i, alpha := range RenyiAlphas {
		res[p].Alpha = alpha

		// Assume both budget and other are sorted by alpha.
		for p < len(budget) && alpha > budget[p].Alpha {
			p++
		}

		for q < len(other) && alpha > other[q].Alpha {
			q++
		}

		if p < len(budget) && alpha == budget[p].Alpha {
			res[i].Epsilon = budget[p].Epsilon
		}

		if q < len(other) && alpha == other[q].Alpha {
			res[i].Epsilon = math.Min(res[i].Epsilon, other[q].Epsilon)
		}
	}

	return res
}

// Div returns `budget` (vector) divided by `factor` (scalar)
func (budget RenyiBudget) Div(factor float64) RenyiBudget {
	res := make(RenyiBudget, len(budget))
	for p := range budget {
		res[p].Alpha = budget[p].Alpha
		res[p].Epsilon = budget[p].Epsilon / factor
	}
	return res
}

// Mul returns `budget` (vector) multiplied by `factor` (scalar)
func (budget RenyiBudget) Mul(factor float64) RenyiBudget {
	res := make(RenyiBudget, len(budget))
	for p := range budget {
		res[p].Alpha = budget[p].Alpha
		res[p].Epsilon = budget[p].Epsilon * factor
	}
	return res
}

func (budget RenyiBudget) Equal(other RenyiBudget) bool {
	if len(budget) != len(other) {
		return false
	}

	for i := 0; i < len(budget); i++ {
		if budget[i].Alpha != other[i].Alpha || math.Abs(budget[i].Epsilon-other[i].Epsilon) >= EmptyThreshold {
			return false
		}
	}
	return true
}

func (budget RenyiBudget) IsEmpty() bool {
	for i := 0; i < len(budget); i++ {
		if budget[i].Epsilon >= EmptyThreshold {
			return false
		}
	}
	return true
}

func (budget RenyiBudget) IsNegative() bool {
	for i := 0; i < len(budget); i++ {
		if budget[i].Epsilon >= 0 {
			return false
		}
	}
	return true
}

// HasEnough returns true iif the curve of `budget` is greater or equal to `other` for one alpha (with EmptyThreshold margin)
func (budget RenyiBudget) HasEnough(other RenyiBudget) bool {
	b, c := ReduceToSameSupport(budget, other)
	for i := range b {
		// This solution is not super satisfying either, but at least it's safe. E.g. asking 1.0 out of 10.0 when N=10 would be unfair.
		if b[i].Epsilon >= c[i].Epsilon+EmptyThreshold {
			return true
		}
	}
	return false
}

// HasEnough returns true iif the curve of `budget` is greater or equal to `other` for one alpha (with EmptyThreshold margin)
func (budget RenyiBudget) HasEnoughInAllAlpha(other RenyiBudget) bool {
	b, c := ReduceToSameSupport(budget, other)
	for i := range b {
		// This solution is not super satisfying either, but at least it's safe. E.g. asking 1.0 out of 10.0 when N=10 would be unfair.
		if b[i].Epsilon < c[i].Epsilon+EmptyThreshold {
			return false
		}
	}
	return true
}

// OldHasEnough returns true iif the curve of `budget` is greater or equal to `other` for one alpha
func (budget RenyiBudget) OldHasEnough(other RenyiBudget) bool {
	p, q := 0, 0
	for ; p < len(budget) && q < len(other); p++ {
		if budget[p].Alpha != other[q].Alpha {
			continue
		}
		// Be more flexible (but not conservative)
		if budget[p].Epsilon+EmptyThreshold >= other[q].Epsilon {
			return true
		}
		q++
	}
	return false
}

// MoreThan returns true iif the curve of `budget` is strictly greater than `other` for one alpha
func (budget RenyiBudget) MoreThan(other RenyiBudget) bool {
	p, q := 0, 0
	for ; p < len(budget) && q < len(other); p++ {
		if budget[p].Alpha != other[q].Alpha {
			continue
		}
		if budget[p].Epsilon+EmptyThreshold > other[q].Epsilon {
			return true
		}
		q++
	}
	return false
}

func (budget PrivacyBudget) Type() PrivacyBudgetType {
	if budget.EpsDel != nil {
		return EpsDelType
	}

	if budget.Renyi != nil {
		return RenyiType
	}

	return NilType
}

func (budget *PrivacyBudget) IsEpsDelType() bool {
	return budget.EpsDel != nil
}

func (budget PrivacyBudget) IsRenyiType() bool {
	return budget.Renyi != nil
}

func (budget *PrivacyBudget) ToRenyi() {
	if budget.IsEpsDelType() {
		budget.Renyi = budget.EpsDel.ToRenyi()
		budget.EpsDel = nil
	}
}

func makeTypeCompatible(a, b *PrivacyBudget) PrivacyBudgetType {
	typeA := a.Type()
	typeB := b.Type()

	// The reason to use OR operation here is that different type representation
	// stays at different bit. If both types are the same or one type is nil,
	// the OR result should still be this type.
	if typeA|typeB == RenyiType|EpsDelType {
		return ErrorType
	}

	if typeA|typeB == NilType {
		return NilType
	}

	if typeA == NilType {
		if typeB == RenyiType {
			a.Renyi = make(RenyiBudget, 0)
		} else if typeB == EpsDelType {
			a.EpsDel = new(EpsilonDeltaBudget)
		}
	}

	if typeB == NilType {
		if typeA == RenyiType {
			b.Renyi = make(RenyiBudget, 0)
		} else if typeA == EpsDelType {
			b.EpsDel = new(EpsilonDeltaBudget)
		}
	}

	return typeA | typeB
}

func (budget PrivacyBudget) Add(other PrivacyBudget) PrivacyBudget {
	budgetType := makeTypeCompatible(&budget, &other)
	switch budgetType {
	case EpsDelType:
		res := budget.EpsDel.Add(*other.EpsDel)
		return PrivacyBudget{
			EpsDel: &res,
		}
	case RenyiType:
		return PrivacyBudget{
			Renyi: budget.Renyi.Add(other.Renyi),
		}
	case NilType:
		return PrivacyBudget{}
	}

	panic("budget type is incompatible")
}

func (budget PrivacyBudget) Minus(other PrivacyBudget) PrivacyBudget {
	budgetType := makeTypeCompatible(&budget, &other)
	switch budgetType {
	case EpsDelType:
		res := budget.EpsDel.Minus(*other.EpsDel)
		return PrivacyBudget{
			EpsDel: &res,
		}
	case RenyiType:
		return PrivacyBudget{
			Renyi: budget.Renyi.Minus(other.Renyi),
		}
	case NilType:
		return PrivacyBudget{}
	}

	panic("budget type is incompatible")
}

func (budget *PrivacyBudget) IAdd(other PrivacyBudget) {
	budgetType := makeTypeCompatible(budget, &other)
	switch budgetType {
	case EpsDelType:
		budget.EpsDel.IAdd(*(other.EpsDel))
		return
	case RenyiType:
		budget.Renyi.IAdd(other.Renyi)
		return
	case NilType:
		return
	}

	panic("budget type is incompatible")
}

func (budget PrivacyBudget) OldIAdd(other PrivacyBudget) {
	budgetType := makeTypeCompatible(&budget, &other)
	switch budgetType {
	case EpsDelType:
		budget.EpsDel.IAdd(*other.EpsDel)
		return
	case RenyiType:
		budget.Renyi.IAdd(other.Renyi)
		klog.Info("budget adfter adding", budget)
		return
	case NilType:
		return
	}

	panic("budget type is incompatible")
}

func (budget *PrivacyBudget) ISub(other PrivacyBudget) {
	budgetType := makeTypeCompatible(budget, &other)
	switch budgetType {
	case EpsDelType:
		budget.EpsDel.ISub(*(other.EpsDel))
		return
	case RenyiType:
		budget.Renyi.ISub(other.Renyi)
		return
	case NilType:
		return
	}

	panic("budget type is incompatible")
}

func (budget PrivacyBudget) OldISub(other PrivacyBudget) {
	budgetType := makeTypeCompatible(&budget, &other)
	switch budgetType {
	case EpsDelType:
		budget.EpsDel.ISub(*other.EpsDel)
		return
	case RenyiType:
		budget.Renyi.ISub(other.Renyi)
		return
	case NilType:
		return
	}

	panic("budget type is incompatible")
}

func (budget PrivacyBudget) Union(other PrivacyBudget) PrivacyBudget {
	budgetType := makeTypeCompatible(&budget, &other)
	switch budgetType {
	case EpsDelType:
		res := budget.EpsDel.Union(*other.EpsDel)
		return PrivacyBudget{
			EpsDel: &res,
		}
	case RenyiType:
		return PrivacyBudget{
			Renyi: budget.Renyi.Union(other.Renyi),
		}
	case NilType:
		return PrivacyBudget{}
	}

	panic("budget type is incompatible")
}

func (budget PrivacyBudget) Intersect(other PrivacyBudget) PrivacyBudget {
	budgetType := makeTypeCompatible(&budget, &other)
	switch budgetType {
	case EpsDelType:
		res := budget.EpsDel.Intersect(*other.EpsDel)
		return PrivacyBudget{
			EpsDel: &res,
		}
	case RenyiType:
		return PrivacyBudget{
			Renyi: budget.Renyi.Intersect(other.Renyi),
		}
	case NilType:
		return PrivacyBudget{}
	}

	panic("budget type is incompatible")
}

func (budget PrivacyBudget) NonNegative() PrivacyBudget {
	if budget.IsEpsDelType() {
		res := budget.EpsDel.NonNegative()
		return PrivacyBudget{
			EpsDel: &res,
		}
	}

	if budget.IsRenyiType() {
		return PrivacyBudget{
			Renyi: budget.Renyi.NonNegative(),
		}
	}

	return PrivacyBudget{}
}

func (budget PrivacyBudget) INonNegative() {
	budget = budget.NonNegative()
}

func (budget PrivacyBudget) Negative() PrivacyBudget {
	if budget.IsEpsDelType() {
		res := budget.EpsDel.Negative()
		return PrivacyBudget{
			EpsDel: &res,
		}
	}

	if budget.IsRenyiType() {
		return PrivacyBudget{
			Renyi: budget.Renyi.Negative(),
		}
	}

	return PrivacyBudget{}
}

func (budget PrivacyBudget) Div(factor float64) PrivacyBudget {
	if budget.IsEpsDelType() {
		res := budget.EpsDel.Div(factor)
		return PrivacyBudget{
			EpsDel: &res,
		}
	}

	if budget.IsRenyiType() {
		return PrivacyBudget{
			Renyi: budget.Renyi.Div(factor),
		}
	}

	return PrivacyBudget{}
}

func (budget PrivacyBudget) Mul(factor float64) PrivacyBudget {
	if budget.IsEpsDelType() {
		res := budget.EpsDel.Mul(factor)
		return PrivacyBudget{
			EpsDel: &res,
		}
	}

	if budget.IsRenyiType() {
		return PrivacyBudget{
			Renyi: budget.Renyi.Mul(factor),
		}
	}

	return PrivacyBudget{}
}

func (budget PrivacyBudget) Equal(other PrivacyBudget) bool {
	budgetType := makeTypeCompatible(&budget, &other)
	switch budgetType {
	case EpsDelType:
		return budget.EpsDel.Equal(*other.EpsDel)
	case RenyiType:
		return budget.Renyi.Equal(other.Renyi)
	case NilType:
		return true
	}

	panic("budget type is incompatible")
}

func (budget PrivacyBudget) IsEmpty() bool {
	if budget.IsEpsDelType() {
		return budget.EpsDel.IsEmpty()
	}

	if budget.IsRenyiType() {
		return budget.Renyi.IsEmpty()
	}

	return true
}

func (budget PrivacyBudget) IsNegative() bool {
	if budget.IsEpsDelType() {
		return budget.EpsDel.IsNegative()
	}

	if budget.IsRenyiType() {
		return budget.Renyi.IsNegative()
	}

	return false
}

func (budget PrivacyBudget) HasEnough(other PrivacyBudget) bool {
	budgetType := makeTypeCompatible(&budget, &other)
	switch budgetType {
	case EpsDelType:
		return budget.EpsDel.HasEnough(*other.EpsDel)
	case RenyiType:
		return budget.Renyi.HasEnough(other.Renyi)
	case NilType:
		return true
	}

	panic("budget type is incompatible")
}

func (budget PrivacyBudget) MoreThan(other PrivacyBudget) bool {
	budgetType := makeTypeCompatible(&budget, &other)
	switch budgetType {
	case EpsDelType:
		return budget.EpsDel.MoreThan(*other.EpsDel)
	case RenyiType:
		return budget.Renyi.MoreThan(other.Renyi)
	case NilType:
		return false
	}

	panic("budget type is incompatible")
}
