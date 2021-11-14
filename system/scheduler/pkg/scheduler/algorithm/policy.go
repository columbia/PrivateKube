package algorithm

import "strings"

type AllocatePolicy struct {
	AllOrNothing bool
}

func ParseAllocatePolicy(policyStr string) (policy AllocatePolicy) {
	words := strings.Split(policyStr, ",")
	for _, word := range words {
		switch word {
		case "allOrNothing=true":
			policy.AllOrNothing = true
		}
	}
	return
}

type ConsumePolicy struct {
	AllOrNothing           bool
	ConsumeAvailableBudget bool
}

func ParseCommitPolicy(policyStr string) (policy ConsumePolicy) {

	words := strings.Split(policyStr, ",")
	for _, word := range words {
		switch word {
		case "allOrNothing=true":
			policy.AllOrNothing = true
		case "consumeAvailableBudget=true":
			policy.ConsumeAvailableBudget = true
		}
	}
	return
}

type ReleasePolicy struct {
	ReleaseAvailableBudget bool
}

func ParseReleasePolicy(policyStr string) (policy ReleasePolicy) {
	words := strings.Split(policyStr, ",")
	for _, word := range words {
		switch word {
		case "abortAvailable=true":
			policy.ReleaseAvailableBudget = true
		}
	}
	return
}
