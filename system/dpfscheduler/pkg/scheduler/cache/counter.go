/// A DP counter for streams
/// Ref: https://eprint.iacr.org/2010/076.pdf
///
/// See also our standalone Python version:
/// https://github.com/columbia/PrivateKube/blob/main/privatekube/privacy/streaming.py
///
/// There are a lot of other algorithms and optimizations that do the same thing.
/// E.g. https://www.microsoft.com/en-us/research/wp-content/uploads/2010/01/dwork_soda10.pdf

package cache

import (
	"fmt"
	"math"

	"columbia.github.com/privatekube/dpfscheduler/pkg/scheduler/util"
	columbiav1 "columbia.github.com/privatekube/privacyresource/pkg/apis/columbia.github.com/v1"
	"k8s.io/klog"
)

type StreamingCounter struct {
	LaplaceNoise  float64
	Budget        *columbiav1.PrivacyBudget
	T             int
	n_bits        int
	current_index int
	counts        map[int]int
	alpha         map[int]int
	n_alpha       map[int]float64
}

type StreamingCounterOptions struct {
	LaplaceNoise     float64
	MaxNumberOfTicks int
}

/// NewStreamingCounter builds a new counter.
/// We could also load the counter from a previous set of blocks (persisted in etcd) in case we need to rebuild the cache.
func NewStreamingCounter(options *StreamingCounterOptions) *StreamingCounter {
	// The budget is lazily initialized depending on the type of the blocks (RDP or not)
	// All the counters are initialized to zero
	return &StreamingCounter{
		LaplaceNoise:  options.LaplaceNoise,
		T:             options.MaxNumberOfTicks,
		n_bits:        int(math.Ceil(math.Log2(float64(options.MaxNumberOfTicks)))),
		current_index: 1,
		counts:        make(map[int]int),
		alpha:         make(map[int]int),
		n_alpha:       make(map[int]float64),
	}
}

/// Feed extends the stream with `n_ticks` new bits, where `n_new_users` bits are 1
/// Feed is called only inside cache.AddBlock and cache.blockLock.Lock() ensures that there is no race condition
func (counter *StreamingCounter) Feed(n_new_users int, n_ticks int) error {
	if n_new_users > n_ticks {
		// The block loader will ensure that this error does not happen
		// If you want to use fewer ticks, you should use a different counter algorithm
		return fmt.Errorf("there are more 1 bits than the length of the stream")
	}

	if (counter.current_index-1)+n_ticks > counter.T {
		return fmt.Errorf("this counter is full: the maximum stream length has been reached")
	}

	for t := counter.current_index; t < counter.current_index+n_ticks; t++ {
		i := 0
		for bin_digit(t, i) == 0 {
			i++
		}

		// We simulate n_ticks where only the `n_new_users` ticks are 1
		if t-counter.current_index < n_new_users {
			counter.alpha[i] = 1
		} else {
			counter.alpha[i] = 0
		}

		// All the j < i are added to the new (larger) psum and erased
		// alpha[i] is the psum of the 2^i last items before t
		for j := 0; j < i; j++ {
			counter.alpha[i] += counter.alpha[j]
			counter.alpha[j] = 0
			counter.n_alpha[j] = 0.0
		}

		// Sanitize and release the psum
		counter.n_alpha[i] = float64(counter.alpha[i]) + util.LaplaceNoise(counter.LaplaceNoise)

		// Add psums of the remaining items
		c := 0.0
		for j := 0; j < counter.n_bits; j++ {
			if bin_digit(t, j) == 1 {
				c += counter.n_alpha[j]
			}
		}

		// Map to the closest credible value. No stream consistency here.
		counter.counts[t] = int(math.Ceil(math.Max(0, c)))
		println("Count", t, counter.counts[t])

	}
	counter.current_index += n_ticks

	return nil
}

/// bin_digit returns the ith digit of t in binary, 0th digit is the least significant.
func bin_digit(t int, i int) int {
	return (t >> i) % 2
}

/// GetLaplaceBudgetAs returns the budget spent by the counter for a given block, either in RDP or epsilon-delta DP
func (counter *StreamingCounter) GetLaplaceBudgetAs(target *columbiav1.PrivacyBudget) columbiav1.PrivacyBudget {
	// Cache the result of the computation assuming all the blocks have the same tracking set of RDP orders
	if counter.Budget != nil {
		return *counter.Budget
	}

	var budget columbiav1.PrivacyBudget
	if target.IsEpsDelType() {
		budget = columbiav1.NewPrivacyBudget(float64(counter.n_bits)/counter.LaplaceNoise, 0, false)
	} else {
		// See extended paper for the RDP curve, which corresponds to the sum of log T curves of the Laplace mechanism
		b := make(columbiav1.RenyiBudget, 0, len(target.Renyi))
		target.Copy()
		for i := range target.Renyi {
			alpha := target.Renyi[i].Alpha
			epsilon := float64(counter.n_bits) * laplaceRDP(counter.LaplaceNoise, alpha)
			if !math.IsInf(epsilon, 0) && !math.IsNaN(epsilon) {
				b = append(b, columbiav1.RenyiBudgetBlock{
					Alpha:   alpha,
					Epsilon: epsilon,
				})
			}
		}
		budget = columbiav1.PrivacyBudget{
			EpsDel: nil,
			Renyi:  b,
		}
	}
	counter.Budget = &budget
	return budget

}

/// laplaceRDP computes the RDP curve of the Laplace mechanism for alpha > 1
/// Ref: Table II of the RDP paper (https://arxiv.org/pdf/1702.07476.pdf)
func laplaceRDP(l float64, alpha float64) float64 {
	a := alpha * math.Exp((alpha-1)/l) / (2*alpha - 1)
	b := (alpha - 1) * math.Exp(-1*alpha/l) / (2*alpha - 1)
	return math.Log(a+b) / (alpha - 1)
}

/// count returns the DP count of 1s in the stream at tick `t`. We don't actually use all the ticks.
func (counter *StreamingCounter) count(t int) int {
	return counter.counts[t]
}

/// CountLast returns the DP count for the last tick along with the value of that tick.
/// In our case, we can release and store this result into the newly-appended block.
func (counter *StreamingCounter) CountLast() (int, int) {
	// current_index starts at 1
	return counter.count(counter.current_index - 1), counter.current_index - 1
}

/// UpdateCount computes the DP count (if possible). DPCount is the cumulative count.
/// It updates the state of the counter and writes the result back into the block.
func (counter *StreamingCounter) UpdateCount(block *columbiav1.PrivateDataBlock) {
	n_new_users, n_ticks, count_index, err := getNonDPCounts(block)
	if err != nil {
		klog.Info("Failed to retrieve counts from this block:", err)
		return
	}
	err = counter.Feed(n_new_users, n_ticks)
	if err != nil {
		klog.Info("Failed to update the counter:", err)
		return
	}
	dp_count, tick := counter.CountLast()

	// Pay for the result we just got
	budget := counter.GetLaplaceBudgetAs(&block.Spec.InitialBudget).Copy()
	block.Status.CommittedBudgetMap["counter"] = budget
	block.Status.PendingBudget = block.Status.PendingBudget.Minus(budget).NonNegative()

	// Store the DP count in the block's dimensions, along with the corresponding tick
	block.Spec.Dimensions[count_index] = columbiav1.Dimension{
		Attribute:    "DPCount",
		NumericValue: util.ToDecimal(dp_count),
		StringValue:  fmt.Sprintf("%d", tick),
	}
}

/// getNonDPCounts tries to read the number of new newers from a block's metadata
/// `count_index` is the index in the `Dimensions` list that contains the non-DP count. We will use it to overwrite the count with DPCount instead.
func getNonDPCounts(block *columbiav1.PrivateDataBlock) (int, int, int, error) {
	n_new_users, n_ticks, count_index := -1, -1, -1
	var err error
	for i, dim := range block.Spec.Dimensions {
		if dim.Attribute == "NNewUsers" {
			n_new_users = int(dim.NumericValue.IntPart())
			count_index = i
		}
		if dim.Attribute == "NTicks" {
			n_ticks = int(dim.NumericValue.IntPart())
		}
	}
	// If the block doesn't have any user count attributes, the variables remain at -1
	if n_new_users < 0 || n_ticks < 0 {
		err = fmt.Errorf("this block doesn't have a user count")
	}
	if n_new_users > n_ticks {
		n_new_users = n_ticks
		klog.Warning("there are more new users than ticks. You should increase the number of ticks (e.g. every millisecond instead of every second). Instead of crashing, we truncated the number of new users for this block: n_new_users := n_ticks")
	}
	return n_new_users, n_ticks, count_index, err
}
