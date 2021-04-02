package flowreleasing

const (
	notInFlowModeMode               = 1
	budgetReleaseNotStartMode       = 2
	noAvailableBudgetToAllocateMode = 3
	noPendingBudgetMode             = 4
)

type flowReleasingError struct {
	Code    int
	Message string
}

func (err *flowReleasingError) Error() string {
	return err.Message
}

func notInFlowMode() error {
	return &flowReleasingError{
		Code:    notInFlowModeMode,
		Message: "the block is not flow mode: either releaseMode is not flow or flowReleasingOption is null or all pending budget has been released",
	}
}

func budgetReleaseNotStart() error {
	return &flowReleasingError{
		Code:    budgetReleaseNotStartMode,
		Message: "the start time of budget release has not reached",
	}
}

func noAvailableBudgetToAllocate() error {
	return &flowReleasingError{
		Code:    noAvailableBudgetToAllocateMode,
		Message: "no available budget to allocate",
	}
}

func noPendingBudget() error {
	return &flowReleasingError{
		Code:    noPendingBudgetMode,
		Message: "no pending budget to release",
	}
}
