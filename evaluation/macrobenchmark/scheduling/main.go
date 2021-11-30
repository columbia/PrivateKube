package main

import (
	"flag"
	"fmt"
	"io"
	"log"
	"math/rand"
	"net/http"
	_ "net/http/pprof"
	"os"
	"runtime/pprof"
	"time"

	"columbia.github.com/privatekube/evaluation/macrobenchmark/scheduling/stub"
)

var (
	DPF_N                               int
	pipeline_timeout_blocks             int
	epsilon                             float64
	delta                               float64
	rdp                                 int
	n_blocks                            int
	block_interval_millisecond          int
	elephants_dir                       string
	mice_dir                            string
	mice_ratio                          float64
	mean_pipelines_per_block            float64
	initial_blocks                      int
	output_claims                       string
	output_blocks                       string
	profile                             string
	mode                                string
	scheduler_method                    string
	DPF_T                               int
	release_steps_per_scheduling_period int
	dpf_release_period_block            float64
)

func init() {
	flag.StringVar(&scheduler_method, "scheduler", "DPF", "Scheduler mode.")
	flag.StringVar(&mode, "mode", "N", "DPF's mode. Either `N` or `T`.")
	flag.IntVar(&DPF_T, "T", 1, "The value of T for DPF's T-scheme. The budget for each block is completely released after T-1 periods.")
	flag.Float64Var(&dpf_release_period_block, "release_period", -1, "The release period for DPF-T, in blocks duration. If not specified, we will use 0.5 * (mean pipeline arrival time)")
	flag.IntVar(&DPF_N, "N", 1, "The value of N for DPF's N-scheme")
	flag.IntVar(&pipeline_timeout_blocks, "timeout", 5, "pipeline_timeout_blocks")
	flag.Float64Var(&epsilon, "epsilon", 5.0, "epsilon global per block")
	flag.Float64Var(&delta, "delta", 1e-9, "delta global per block")
	flag.IntVar(&rdp, "rdp", 0, "Set rdp=0 to use eps-del DP, rdp=1 to use RDP (with a default gamma = 0.05).")
	flag.IntVar(&n_blocks, "n", 10, "n_blocks")
	flag.IntVar(&block_interval_millisecond, "t", 1_000, "block_interval_millisecond")
	flag.StringVar(&elephants_dir, "elephants", "", "Path to the elephants_dir")
	flag.StringVar(&mice_dir, "mice", "", "Path to the mice_dir")
	flag.Float64Var(&mice_ratio, "mice_ratio", 0.5, "Mice ratio in the pipeline mix")
	flag.Float64Var(&mean_pipelines_per_block, "m", 5, "Mean number of pipelines per block")
	flag.IntVar(&initial_blocks, "b", 5, "Wait for b blocks before we start sending pipelines")
	flag.StringVar(&output_blocks, "log_blocks", "", "Path to store the blocks log file (json)")
	flag.StringVar(&output_claims, "log_claims", "", "Path to store the claims log file (json)")
	flag.StringVar(&profile, "profile", "", "Base name to store profiling files. No profiling if empty")
}

func main() {

	flag.Parse()

	if profile != "" {
		go func() {
			http.ListenAndServe("localhost:8080", nil)

		}()
		f, err := os.Create(profile)
		if err != nil {
			log.Fatal(err)
		}
		fmt.Println("Starting profiling.")
		// pprof.
		pprof.StartCPUProfile(f)
		defer pprof.StopCPUProfile()
	}

	gamma := -1.0
	if rdp == 1 {
		// Default setting to drop the meaningless RDP orders.
		gamma = 0.05
	}
	run_exponential(scheduler_method, mode, DPF_T, dpf_release_period_block, DPF_N, pipeline_timeout_blocks, epsilon, delta, gamma, n_blocks, block_interval_millisecond, elephants_dir, mice_dir, mice_ratio, mean_pipelines_per_block, initial_blocks, output_blocks, output_claims)

	if profile != "" {
		fmt.Println("Saving the profiles.")
		url := "http://localhost:8080/debug/pprof/heap"
		err := downloadFile("heap.prof", url)
		if err == nil {
			fmt.Println("Downloaded: " + url)
		}
	}

}

func downloadFile(filepath string, url string) error {

	// Get the data
	resp, err := http.Get(url)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	// Create the file
	out, err := os.Create(filepath)
	if err != nil {
		return err
	}
	defer out.Close()

	// Write the body to file
	_, err = io.Copy(out, resp.Body)
	return err
}

func run_exponential(scheduler_method, mode string, DPF_T int, dpf_release_period_block float64, DPF_N int, pipeline_timeout_blocks int, epsilon float64, delta float64, gamma float64, n_blocks int, block_interval_millisecond int, elephants_dir string, mice_dir string, mice_ratio float64, mean_pipelines_per_block float64, initial_blocks int, output_blocks string, output_claims string) {

	r := rand.New(rand.NewSource(99))

	rdp := gamma >= 0
	s := stub.NewStub()

	timeout := time.Duration(pipeline_timeout_blocks*block_interval_millisecond) * time.Millisecond
	task_interval_millisecond := float64(block_interval_millisecond) / mean_pipelines_per_block
	switch mode {

	case "N":
		s.StartN(timeout, DPF_N, scheduler_method)
	case "T":
		dpf_release_period_millisecond := 0
		if dpf_release_period_block <= 0 {
			dpf_release_period_millisecond = int(0.5 * float64(block_interval_millisecond) / mean_pipelines_per_block)
			fmt.Print("Computing the release period from the arrival time.")
		} else {
			dpf_release_period_millisecond = int(dpf_release_period_block * float64(block_interval_millisecond))
		}
		fmt.Println("release time\n\n\n", dpf_release_period_block)
		fmt.Println("dpf_release_period_milliseconde\n\n\n", dpf_release_period_millisecond)

		s.StartT(timeout, DPF_T, dpf_release_period_millisecond, scheduler_method)
	default:
		log.Fatal("Invalid DPF mode", mode)
	}

	b := stub.NewBlockGenerator(&s, "amazon", epsilon, delta, gamma, n_blocks+initial_blocks, time.Duration(block_interval_millisecond)*time.Millisecond)

	m := stub.MakeSampler(rdp, mice_ratio, mice_dir, elephants_dir)
	g := stub.ClaimGenerator{
		BlockGen:              b,
		MeanPipelinesPerBlock: mean_pipelines_per_block,
		Pipelines:             &m,
		Rand:                  r,
	}
	// Collect the objects' names for future analysis
	block_names := make(chan string, b.MaxBlocks)
	claim_names := make(chan string, 100*int(mean_pipelines_per_block*float64(b.MaxBlocks)))

	// Start the block generator in the background
	go b.RunLog(block_names)
	// Wait a bit before sending pipelines
	time.Sleep(time.Duration(initial_blocks) * b.BlockInterval)
	//g.RunExponentialDeterministic(claim_names, timeout, n_blocks)
	g.RunConstant(claim_names, timeout, n_blocks, time.Duration(task_interval_millisecond)*time.Millisecond)

	fmt.Println("Waiting for the last pipelines to timeout")
	time.Sleep(10 * b.BlockInterval)

	// Close the channels and browse the objets
	fmt.Println("Collecting and saving the logs.")
	close(block_names)
	close(claim_names)
	blocks := make([]interface{}, 0, n_blocks)
	claims := make([]interface{}, 0, n_blocks)

	for block_name := range block_names {
		blocks = append(blocks, s.GetDataBlock(block_name))
	}
	for claim_name := range claim_names {
		claim := s.GetClaim(claim_name)
		// AddForcedTimeout(claim)
		claims = append(claims, claim)
	}
	stub.SaveObjects(blocks, output_blocks)
	stub.SaveObjects(claims, output_claims)
}
