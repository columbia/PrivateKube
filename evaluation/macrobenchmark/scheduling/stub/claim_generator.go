package stub

import (
	columbiav1 "columbia.github.com/privatekube/privacyresource/pkg/apis/columbia.github.com/v1"
	"fmt"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"log"
	"math/rand"
	"time"
)

type ClaimGenerator struct {
	BlockGen              *BlockGenerator
	Pipelines             *MiceElphantsSampler
	MeanPipelinesPerBlock float64
	Rand                  *rand.Rand
}

type Sampler interface {
	SampleOne() Pipeline
}

type MiceElphantsSampler struct {
	MiceRatio float64
	Mice      []Pipeline
	Elephants []Pipeline
}

func MakeSampler(rdp bool, mice_ratio float64, mice_path string, elephants_path string) MiceElphantsSampler {
	mice := LoadDir(mice_path)
	elephants := LoadDir(elephants_path)
	m := make([]Pipeline, 0, len(mice))
	e := make([]Pipeline, 0, len(elephants))
	for name, raw_pipeline := range mice {
		m = append(m, NewPipeline(name, raw_pipeline, rdp))
	}
	for name, raw_pipeline := range elephants {
		e = append(e, NewPipeline(
			name, raw_pipeline, rdp))
	}
	return MiceElphantsSampler{
		MiceRatio: mice_ratio,
		Mice:      m,
		Elephants: e,
	}
}

func (p MiceElphantsSampler) SampleOne(r *rand.Rand) Pipeline {
	if r.Float64() < p.MiceRatio {
		i := r.Intn(len(p.Mice))
		fmt.Println("Sample one mice: %d\n", i)
		return p.Mice[i]
	}
	i := r.Intn(len(p.Elephants))
	fmt.Println("Sample one elephants: %d\n", i)
	return p.Elephants[i]
}

func (g *ClaimGenerator) createClaim(block_index int, model Pipeline, timeout time.Duration) (*columbiav1.PrivacyBudgetClaim, error) {
	// Store the timestamp for analysis
	annotations := make(map[string]string)
	annotations["actualStartTime"] = fmt.Sprint(int(time.Now().UnixNano() / 1_000_000))

	// Create a new claim with flat demand that asks for the NBlock most recent blocks
	claim := &columbiav1.PrivacyBudgetClaim{
		ObjectMeta: metav1.ObjectMeta{
			Name:      fmt.Sprintf("%s-%d-%s", model.Name, block_index, RandId()),
			Namespace: g.BlockGen.Stub.Namespace,
			// CreationTimestamp: metav1.NewTime(time.Now()),
			Annotations: annotations,
		},
		Spec: columbiav1.PrivacyBudgetClaimSpec{
			Priority: int32(1000*model.Epsilon) * int32(model.NBlocks),
			Requests: []columbiav1.Request{
				{
					Identifier: "1",
					AllocateRequest: &columbiav1.AllocateRequest{
						Dataset: g.BlockGen.Dataset,
						Conditions: []columbiav1.Condition{
							{
								Attribute:    "startTime",
								Operation:    ">=",
								NumericValue: ToDecimal(block_index - model.NBlocks + 1),
							},
							{
								Attribute:    "startTime",
								Operation:    "<=",
								NumericValue: ToDecimal(block_index),
							},
						},

						MinNumberOfBlocks: model.NBlocks,
						MaxNumberOfBlocks: model.NBlocks,
						ExpectedBudget: columbiav1.BudgetRequest{
							Constant: &model.Demand,
						},
						Timeout: int64(timeout / time.Millisecond),
					},
				},
			},
		},
	}
	// if block_index-model.NBlocks+1 < 0 {
	// 	// Border case where the claim is asking for blocks that don't exist
	// 	// Let's make sure it is not allocated
	// 	claim.Spec.Requests[0].AllocateRequest.Timeout = 0
	// }
	return g.BlockGen.Stub.CreateClaim(claim)
}

func (g *ClaimGenerator) Run() {
	ticker := time.NewTicker(g.BlockGen.BlockInterval)
	go func() {
		index := 0
		for index < g.BlockGen.MaxBlocks {
			<-ticker.C
			go g.BlockGen.createDataBlock(index)
			// go g.createClaim(index, model)
			index++
		}
	}()
}
func (g *ClaimGenerator) RunExponentialDeterministic(claim_names chan string, default_timeout time.Duration, task_interval time.Duration) {
	total_duration := time.Duration(g.BlockGen.MaxBlocks+1) * g.BlockGen.BlockInterval
	end_time := g.BlockGen.StartTime.Add(total_duration)
	total_tasks := int(g.MeanPipelinesPerBlock) * g.BlockGen.MaxBlocks
	r := rand.New(rand.NewSource(0))

	index := 0

	//ticker := time.NewTicker(task_interval)
	for index < total_tasks {
		interval := (g.Rand.ExpFloat64() / g.MeanPipelinesPerBlock) * float64(g.BlockGen.BlockInterval.Microseconds())
		timer := time.NewTimer(time.Duration(interval) * time.Microsecond)
		<-timer.C

		block_index := g.BlockGen.CurrentIndex()
		// Cap the timeout by the simulation running time (with a five-block margin)
		timeout := time.Until(end_time) + 5*g.BlockGen.BlockInterval
		if timeout > default_timeout {
			timeout = default_timeout
		}
		go func(int, time.Duration, *rand.Rand) {

			pipeline := g.Pipelines.SampleOne(r)
			claim, err := g.createClaim(block_index, pipeline, timeout)
			if err != nil {
				log.Fatal(err)
			} else {
				claim_names <- claim.ObjectMeta.Name
			}
		}(block_index, timeout, r)
		index++
	}
}

//
//func (g *ClaimGenerator) RunConstant(claim_names chan string, default_timeout time.Duration, task_interval time.Duration) {
//	total_duration := time.Duration(g.BlockGen.MaxBlocks+1) * g.BlockGen.BlockInterval
//	end_time := g.BlockGen.StartTime.Add(total_duration)
//	total_tasks := int(g.MeanPipelinesPerBlock) * g.BlockGen.MaxBlocks
//
//	index := 0
//	ticker := time.NewTicker(task_interval)
//	for index < total_tasks {
//		<-ticker.C
//		block_index := g.BlockGen.CurrentIndex()
//		// Cap the timeout by the simulation running time (with a five-block margin)
//		timeout := time.Until(end_time) + 5*g.BlockGen.BlockInterval
//		if timeout > default_timeout {
//			timeout = default_timeout
//		}
//		go func(int, time.Duration) {
//			pipeline := g.Pipelines.SampleOne(g.Rand)
//			claim, err := g.createClaim(block_index, pipeline, timeout)
//			if err != nil {
//				log.Fatal(err)
//			} else {
//				claim_names <- claim.ObjectMeta.Name
//			}
//		}(block_index, timeout)
//		index++
//	}
//}

func (g *ClaimGenerator) RunExponential(claim_names chan string, default_timeout time.Duration) {
	// NOTE: we can try other start/stop strategies
	total_duration := time.Duration(g.BlockGen.MaxBlocks+1) * g.BlockGen.BlockInterval
	end_time := g.BlockGen.StartTime.Add(total_duration)
	for time.Since(g.BlockGen.StartTime) < total_duration {
		// The default rate parameter is 1 (so the mean is 1 too)
		interval := (g.Rand.ExpFloat64() / g.MeanPipelinesPerBlock) * float64(g.BlockGen.BlockInterval.Microseconds())
		timer := time.NewTimer(time.Duration(interval) * time.Microsecond)
		<-timer.C
		block_index := g.BlockGen.CurrentIndex()
		// Cap the timeout by the simulation running time (with a five-block margin)
		timeout := time.Until(end_time) + 5*g.BlockGen.BlockInterval
		if timeout > default_timeout {
			timeout = default_timeout
		}
		go func(int, time.Duration) {
			pipeline := g.Pipelines.SampleOne(g.Rand)
			claim, err := g.createClaim(block_index, pipeline, timeout)
			if err != nil {
				log.Fatal(err)
			} else {
				claim_names <- claim.ObjectMeta.Name
			}
		}(block_index, timeout)

	}
}
