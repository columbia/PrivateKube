package stub

import (
	"fmt"
	"log"
	"time"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

	columbiav1 "columbia.github.com/privatekube/privacyresource/pkg/apis/columbia.github.com/v1"
)

type BlockGenerator struct {
	Stub          *Stub
	Dataset       string
	InitialBudget columbiav1.PrivacyBudget
	MaxBlocks     int
	BlockInterval time.Duration
	StartTime     time.Time
}

func NewBlockGenerator(stub *Stub, dataset string, epsilon float64, delta float64, gamma float64, maxBlocks int, interval time.Duration) *BlockGenerator {

	return &BlockGenerator{
		Stub:          stub,
		Dataset:       dataset,
		InitialBudget: columbiav1.NewPrivacyBudgetTruncated(epsilon, delta, gamma),
		MaxBlocks:     maxBlocks,
		BlockInterval: interval,
		StartTime:     time.Time{},
	}
}

func (b *BlockGenerator) GetBlockName(index int) string {
	return fmt.Sprintf("block-%d", index)
}

func (b *BlockGenerator) GetBlockNamespacedName(index int) string {
	return fmt.Sprintf("%s/%s", b.Stub.Namespace, b.GetBlockName(index))
}

func (b *BlockGenerator) createDataBlock(index int) (*columbiav1.PrivateDataBlock, error) {
	annotations := make(map[string]string)
	annotations["blockIntervalDuration"] = fmt.Sprint(int(b.BlockInterval) / 1_000_000)
	// BlockInterval, claim startTime and finishTime are milliseconds
	block := &columbiav1.PrivateDataBlock{
		ObjectMeta: metav1.ObjectMeta{
			Name:        b.GetBlockName(index),
			Namespace:   b.Stub.Namespace,
			Annotations: annotations,
		},
		Spec: columbiav1.PrivateDataBlockSpec{
			InitialBudget: b.InitialBudget,
			Dataset:       b.Dataset,
			Dimensions: []columbiav1.Dimension{
				{
					Attribute:    "startTime",
					NumericValue: ToDecimal(index),
				},
				{
					Attribute:    "endTime",
					NumericValue: ToDecimal(index + 1),
				},
			},
		},
	}
	return b.Stub.CreateDataBlock(block)

}

func (b *BlockGenerator) Run() {
	ticker := time.NewTicker(b.BlockInterval)
	b.StartTime = time.Now()
	index := 0
	for index < b.MaxBlocks {
		<-ticker.C
		go b.createDataBlock(index)
		index++
	}
}

func (b *BlockGenerator) RunLog(block_names chan string) {
	ticker := time.NewTicker(b.BlockInterval)

	b.StartTime = time.Now()
	index := 0
	for index < b.MaxBlocks {
		<-ticker.C
		go func() {
			block, err := b.createDataBlock(index)
			if err != nil {
				log.Print("Error while creating the block.")
				log.Print(err)
			} else {
				block_names <- block.ObjectMeta.Name
			}
		}()
		index++
	}
}

// CurrentIndex returns the index of the most recent complete block
func (b *BlockGenerator) CurrentIndex() int {
	elapsed := time.Since(b.StartTime)
	return int(elapsed/b.BlockInterval) - 1
}
