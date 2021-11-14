package errors

import "fmt"

func AllOrNothingError(child error) error {
	if child == nil {
		return fmt.Errorf("allOrNothing policy not fulfilled")
	}

	return fmt.Errorf("allOrNothing policy not fulfilled: %v", child)
}

func NotEnoughDataBlock(required, available int) error {
	return fmt.Errorf("not enough data blocks to allocate. %d required but only %d available",
		required, available)
}

type NothingChangedErr struct{}

func (err NothingChangedErr) Error() string {
	return "nothing changed"
}

func NothingChanged() error {
	return NothingChangedErr{}
}
