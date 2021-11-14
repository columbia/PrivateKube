package timing

import "time"

type Timer struct {
	offset int64
}

func MakeDefaultTimer() Timer {
	return Timer{0}
}

func MakeTimer(offset int64) Timer {
	return Timer{offset}
}

func FakeTimer() Timer {
	now := time.Now().UnixNano() / int64(time.Millisecond)
	return Timer{now}
}

func (timer Timer) Now() int64 {
	now := time.Now().UnixNano() / int64(time.Millisecond)
	return now - timer.offset
}
