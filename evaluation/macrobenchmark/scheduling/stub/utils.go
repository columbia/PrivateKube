package stub

import (
	"encoding/json"
	"fmt"
	"log"
	"math/rand"
	"time"

	"github.com/shopspring/decimal"
)

func init() {
	rand.Seed(time.Now().UnixNano())
}

func ToDecimal(i int) *decimal.Decimal {
	d := decimal.New(int64(i), 0)
	return &d
}

func RandId() string {
	return fmt.Sprintf("%d", rand.Int31())
}

func PrintJson(v interface{}) {
	s, err := json.MarshalIndent(v, "", "\t")
	if err == nil {
		log.Print(string(s))
	}
}

// func AddForcedTimeout(c *columbiav1.PrivacyBudgetClaim, timeout int) {

// }
