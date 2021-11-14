package algorithm

import (
	"strings"

	"columbia.github.com/privatekube/dpfscheduler/pkg/scheduler/util"
	columbiav1 "columbia.github.com/privatekube/privacyresource/pkg/apis/columbia.github.com/v1"
)

func validateCondition(condition columbiav1.Condition, dimension columbiav1.Dimension) bool {
	if dimension.NumericValue != nil {
		return validateNumericCondition(condition, dimension)
	} else if dimension.StringValue != "" {
		return validateStringCondition(condition, dimension)
	} else {
		return false
	}
}

func validateNumericCondition(condition columbiav1.Condition, dimension columbiav1.Dimension) bool {
	switch condition.Operation {
	case "<", "lessThan":
		return dimension.NumericValue.LessThan(*condition.NumericValue)
	case "=", "==", "equal":
		return dimension.NumericValue.Equal(*condition.NumericValue)
	case ">", "greaterThan":
		return dimension.NumericValue.GreaterThan(*condition.NumericValue)
	case "<=", "lessThanOrEqual":
		return dimension.NumericValue.LessThanOrEqual(*condition.NumericValue)
	case ">=", "greaterThanOrEqual":
		return dimension.NumericValue.GreaterThanOrEqual(*condition.NumericValue)
	default:
		return false
	}
}

func validateStringCondition(condition columbiav1.Condition, dimension columbiav1.Dimension) bool {
	switch condition.Operation {
	case "is":
		return dimension.StringValue == condition.StringValue
	case "include":
		return strings.Contains(dimension.StringValue, condition.StringValue)
	default:
		return false
	}
}

func interpretBudgetRequest(budgetRequest columbiav1.BudgetRequest, block *columbiav1.PrivateDataBlock) columbiav1.PrivacyBudget {
	if budgetRequest.Constant != nil {
		return *budgetRequest.Constant
	} else if budgetRequest.Function != "" {
		panic("not support yet")
	} else if budgetRequest.BudgetMap != nil {
		return budgetRequest.BudgetMap[util.GetBlockId(block)]
	} else {
		return columbiav1.PrivacyBudget{}
	}
}
