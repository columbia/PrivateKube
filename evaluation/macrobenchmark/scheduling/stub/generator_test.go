package stub

import (
	"fmt"
	"log"
	"os"
	"path"
	"testing"
	"time"

	columbiav1 "columbia.github.com/privatekube/privacyresource/pkg/apis/columbia.github.com/v1"
)

func TestBlockGenerator(t *testing.T) {
	s := NewStub()
	s.Start(time.Millisecond, 1)
	b := NewBlockGenerator(&s, "amazon", 5.0, 1e-9, -1.0, 10, time.Second*1)
	b.Run()
	time.Sleep(time.Second * 20)
}

func TestExponentialClaimGenerator(t *testing.T) {
	n_blocks := 5
	gamma := 0.0
	s := NewStub()
	s.Start(10*time.Second, 5)
	b := NewBlockGenerator(&s, "amazon", 5.0, 1e-7, gamma, n_blocks, time.Second*1)
	g := ClaimGenerator{
		BlockGen:              b,
		MeanPipelinesPerBlock: 3,
		Pipelines: &MiceElphantsSampler{
			MiceRatio: 0.1,
			Mice:      make([]Pipeline, 1),
			Elephants: make([]Pipeline, 1),
		},
	}
	// Just a fixed model
	// model := Model{
	// 	Name:    "dummy_model",
	// 	Demand:  columbiav1.NewPrivacyBudgetTruncated(3.0, 1e-9, gamma),
	// 	NBlocks: 2,
	// }
	go b.Run()
	time.Sleep(2 * b.BlockInterval)
	claim_names := make(chan string, 100)

	g.RunExponential(claim_names)
	time.Sleep(time.Second * 10)

	close(claim_names)
}

func TestBasicClaimGenerator(t *testing.T) {
	n_blocks := 5
	n_models := n_blocks
	gamma := 0.0
	s := NewStub()
	s.Start(10*time.Second, 5)
	b := NewBlockGenerator(&s, "amazon", 5.0, 1e-7, gamma, n_blocks, time.Second*1)
	g := ClaimGenerator{
		BlockGen: b,
	}
	// Just a fixed model
	model := Pipeline{
		Name:    "dummy_model",
		Demand:  columbiav1.NewPrivacyBudgetTruncated(3.0, 1e-9, gamma),
		NBlocks: 2,
	}
	ticker := time.NewTicker(g.BlockGen.BlockInterval)
	block_names := make(chan string, n_blocks)
	claim_names := make(chan string, n_models)
	go func() {
		index := 0
		for index < g.BlockGen.MaxBlocks {
			<-ticker.C
			go func() {
				block, err := g.BlockGen.createDataBlock(index)
				if err != nil {
					log.Fatal(err)
				} else {
					block_names <- block.ObjectMeta.Name
				}
			}()
			go func() {
				claim, err := g.createClaim(index, model)
				if err != nil {
					log.Fatal(err)
				} else {
					claim_names <- claim.ObjectMeta.Name
				}
			}()
			index++
		}
	}()
	time.Sleep(time.Second * 10)
	close(block_names)
	close(claim_names)
	// for block_name := range block_names {
	// 	PrintJson(s.GetDataBlock(block_name))
	// }
	// for claim_name := range claim_names {
	// 	PrintJson(s.GetClaim(claim_name))
	// }
	blocks := make([]interface{}, 0, n_blocks)
	claims := make([]interface{}, 0, n_blocks)

	// blocks := make([]columbiav1.PrivateDataBlock, 0, n_blocks)
	for block_name := range block_names {
		blocks = append(blocks, s.GetDataBlock(block_name))
		// PrintJson()
	}
	for claim_name := range claim_names {
		claims = append(claims, s.GetClaim(claim_name))
	}
	SaveObjects(blocks, "test_blocks.json")
	SaveObjects(claims, "test_claims.json")

}

func TestMakeSampler(t *testing.T) {
	rdp := true
	home, err := os.UserHomeDir()
	if err != nil {
		log.Fatal(err)
	}
	elephants_dir := path.Join(home, "PrivateKube/evaluation/macrobenchmark/workload/runs/event/elephants")
	mice_dir := path.Join(home, "PrivateKube/evaluation/macrobenchmark/workload/runs/event/mice")
	mice_ratio := 0.5
	m := MakeSampler(rdp, mice_ratio, mice_dir, elephants_dir)
	for i := 0; i < 5; i++ {
		fmt.Println(m.SampleOne())
	}

}
