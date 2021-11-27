package stub

import columbiav1 "columbia.github.com/privatekube/privacyresource/pkg/apis/columbia.github.com/v1"

// We group the blocks per day
const RAW_BLOCKS_MULTIPLIER int = 100

type Pipeline struct {
	Name    string
	Demand  columbiav1.PrivacyBudget
	NBlocks int
	Epsilon float64
}

func NewPipeline(name string, p RawPipeline, rdp bool) Pipeline {
	demand := columbiav1.PrivacyBudget{}
	if rdp {
		b := make(columbiav1.RenyiBudget, len(p.Alphas))
		for i, alpha := range p.Alphas {
			b[i].Alpha = alpha
			b[i].Epsilon = p.RdpEpsilons[i]
		}
		demand = columbiav1.PrivacyBudget{
			EpsDel: nil,
			Renyi:  b,
		}
	} else {
		demand = columbiav1.PrivacyBudget{
			EpsDel: &columbiav1.EpsilonDeltaBudget{
				Epsilon: p.Epsilon,
				Delta:   p.Delta,
			},
			Renyi: nil,
		}

	}
	return Pipeline{
		Name:    name,
		Demand:  demand,
		NBlocks: p.NBlocks / RAW_BLOCKS_MULTIPLIER,
		Epsilon: p.Epsilon,
	}
}

func (g *ClaimGenerator) createFlatDemand(start_block int, end_block int, budget columbiav1.PrivacyBudget) map[string]columbiav1.PrivacyBudget {
	demand := make(map[string]columbiav1.PrivacyBudget)
	for i := start_block; i <= end_block; i++ {
		demand[g.BlockGen.GetBlockNamespacedName(i)] = budget
	}
	return demand
}
