package stub

import (
	"fmt"
	"log"
	"os"
	"path"
	"testing"
)

func TestLoadRun(t *testing.T) {
	home, err := os.UserHomeDir()
	if err != nil {
		log.Fatal(err)
	}
	r, err := LoadPipeline(path.Join(home, "PrivateKube/evaluation/macrobenchmark/workload/runs/event/elephants/product-lstm-1.0.yaml"))
	if err == nil {
		fmt.Println(r)
		fmt.Println(r.Alphas)
	}

}

func TestLoadDir(t *testing.T) {
	home, err := os.UserHomeDir()
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(LoadDir(path.Join(home, "PrivateKube/evaluation/macrobenchmark/workload/runs/event/elephants")))
}
