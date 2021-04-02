// Author: Mingen Pan
package v1

import (
	"encoding/json"

	"github.com/shopspring/decimal"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

const (
	// enum of flowMode in PrivateDataBlock
	Flow = "flow"
	Base = "base"

	// enum of BudgetReturnPolicy in PrivateDataBlock
	ToAvailable = "toAvailable"
	ToPending   = "toPending"

	// locked budget state
	Allocating = "allocating"
	Allocated  = "allocated"
	Committing = "committing"
	Committed  = "committed"
	Aborting   = "aborting"
	Aborted    = "aborted"

	// Response State
	Success = "success"
	Failure = "failure"

	// DPF constant
	Pending      = "pending"
	DpfAcquiring = "dpf_acquiring"
	DpfAcquired  = "dpf_acquired"
	DpfReserving = "dpf_reserving"
	DpfReserved  = "dpf_reserved"

	// Negative number of Min Number Of Blocks
	StrictAllOrNothing = -1
)

// +genclient
// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object
// +kubebuilder:subresource:status

// PrivateDataBlock is a top-level type
type PrivateDataBlock struct {
	metav1.TypeMeta `json:",inline"`
	// +optional
	metav1.ObjectMeta `json:"metadata,omitempty"`

	// Spec defines the behavior of a private data block.
	Status PrivateDataBlockStatus `json:"status,omitempty"`

	// Status represents the current information about a private data block.
	// +optional
	Spec PrivateDataBlockSpec `json:"spec,omitempty"`
}

// custom spec
type PrivateDataBlockSpec struct {
	// initial budget a data block
	InitialBudget PrivacyBudget `json:"initialBudget"`

	// a set of dimensions of this data block.
	// It can be used by a privacy budget claim to select data blocks
	// +optional
	Dimensions []Dimension `json:"dimensions,omitempty"`

	// a description of the data source of this data block.
	// It could be a simple url or a serialized file handler.
	DataSource string `json:"dataSource"`

	// the name of dataset this data block belongs to.
	Dataset string `json:"dataset,omitempty"`

	// release mode
	ReleaseMode string `json:"releaseMode,omitempty"`

	// the policy for the user-released budget. The available policy is:
	// toAvailable: the user-released budget will be placed to the available budget and wait for the next
	// scheduling.
	// toPending: the user-released budget will be placed to the pending budget and released gradually.
	BudgetReturnPolicy string `json:"budgetReturnPolicy,omitempty"`

	// if ReleaseMode is 'flow', then this field should not be null.
	FlowReleasingOption *FlowReleasingOption `json:"flowReleasingOption,omitempty"`
}

type FlowReleasingOption struct {
	// the timestamp where the data block starts to release privacy budget
	// zero or empty means the creation time.
	// unit is the unix timestamp in millisecond
	StartTime int64 `json:"startTime"`

	// the timestamp where the data block should release all the privacy budget
	// unit is the unix timestamp in millisecond
	EndTime int64 `json:"endTime,omitempty"`

	// the duration of the budget releasing.
	// if Duration is set, the scheduler will compute the end time automatically, so the user-specific end time
	// will be override.
	// unit is the unix timestamp in millisecond
	Duration int64 `json:"duration,omitempty"`
}

// custom status
type PrivateDataBlockStatus struct {
	// the pending budget is not empty only in budget flow mode.
	// the budget that are pending and will be released gradually through time.
	PendingBudget PrivacyBudget `json:"pendingBudget"`

	// the budget that can be allocated to privacy budget claims
	AvailableBudget PrivacyBudget `json:"availableBudget"`

	// a map from the id of privacy budget claims to privacy budgets acquired by the claims
	// this is the source of the truth
	AcquiredBudgetMap map[string]PrivacyBudget `json:"acquiredBudgetMap,omitempty"`

	// a map from the id of privacy budget claims to privacy budgets reserved by the claims
	// this is the source of the truth
	ReservedBudgetMap map[string]PrivacyBudget `json:"reservedBudgetMap,omitempty"`

	// a map from the id of privacy budget claims to privacy budgets committed by the claims
	// this is the source of the truth
	CommittedBudgetMap map[string]PrivacyBudget `json:"committedBudgetMap,omitempty"`

	// a map from the id of privacy budget request to privacy budgets locked by this request.
	// this is the source of the truth
	LockedBudgetMap map[string]LockedBudget `json:"lockedBudgetMap,omitempty"`

	// an estimated number of data in this data block, which is computed from a dp query
	// using some budget from this data block.
	// +optional
	NumberOfData int64 `json:"numberOfData,omitempty"`

	// last budget releasing time. It is not zero only at "flow" mode.
	// unit is the unix timestamp in millisecond
	LastBudgetReleaseTime int64 `json:"lastBudgetReleaseTime,omitempty"`
}

func (pb PrivateDataBlock) String() string {
	s, err := json.MarshalIndent(pb, "", "\t")
	if err != nil {
		return "Malformed block."
	}
	return string(s)
}

func (pb PrivateDataBlockStatus) String() string {
	s, err := json.MarshalIndent(pb, "", "\t")
	if err != nil {
		return "Malformed block status."
	}
	return string(s)
}

type Dimension struct {
	Attribute    string           `json:"attribute"`
	NumericValue *decimal.Decimal `json:"numericValue,omitempty"`
	StringValue  string           `json:"stringValue,omitempty"`
}

func (in *Dimension) DeepCopyInto(out *Dimension) {
	*out = *in
	return
}

type LockedBudget struct {
	Budget BudgetAccount `json:"budget"`
	State  string        `json:"state"`
}

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// no client needed for list as it's been created in above
type PrivateDataBlockList struct {
	metav1.TypeMeta `json:",inline"`
	// +optional
	metav1.ListMeta `json:"metadata,omitempty"`

	Items []PrivateDataBlock `json:"items"`
}

// +genclient
// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object
// +kubebuilder:subresource:status

// PrivacyBudgetClaim is a top-level type
type PrivacyBudgetClaim struct {
	metav1.TypeMeta `json:",inline"`
	// +optional
	metav1.ObjectMeta `json:"metadata,omitempty"`

	// specify your request of privacy budgets
	Spec PrivacyBudgetClaimSpec `json:"spec,omitempty"`

	// Status represents the current information about a private budget claim.
	// you can access the responses at this field.
	// +optional
	Status PrivacyBudgetClaimStatus `json:"status,omitempty"`
}

// custom spec
type PrivacyBudgetClaimSpec struct {
	// requests of privacy budget operation, including allocation, consumption, and releasing.
	// append only and each request is immutable once appended.
	Requests []Request `json:"requests"`

	// the owners of this claim
	// +optional
	OwnedBy []corev1.ObjectReference `json:"ownedBy,omitempty"`

	// priority of this claim. the claim with higher priority is allocated before those with low priority.
	// +optional
	Priority int32 `json:"priority"`
}

// custom status
type PrivacyBudgetClaimStatus struct {
	// a map from the id of private data blocks to privacy budgets acquired by this claims
	AcquiredBudgets map[string]PrivacyBudget `json:"acquiredBudgets,omitempty"`

	// a map from the id of private data blocks to privacy budgets reserved by this claims
	ReservedBudgets map[string]PrivacyBudget `json:"reservedBudgets,omitempty"`

	// a map from the id of private data blocks to privacy budgets committed by this claims
	CommittedBudgets map[string]PrivacyBudget `json:"committedBudgets,omitempty"`
	Responses        []Response               `json:"responses,omitempty"`
}

// A request of privacy budget operations
// only one of the AllocateRequest, ConsumeRequest, and ReleaseRequest should be not null
type Request struct {
	// a unique identifier to distinguish this request from others. Requests of the same claim should have different
	// identifiers.
	// +optional
	Identifier string `json:"identifier,omitempty"`

	AllocateRequest *AllocateRequest `json:"allocateRequest,omitempty"`
	ConsumeRequest  *ConsumeRequest  `json:"consumeRequest,omitempty"`
	ReleaseRequest  *ReleaseRequest  `json:"releaseRequest,omitempty"`
}

// An Allocate request is used to specify the request to acquire and/or reserve privacy budgets.
type AllocateRequest struct {
	// specify the datatset to be allocated privacy budgets
	Dataset string `json:"dataset"`

	// an array of conditions, which specifies the requirement on dimensions of Private Data Blocks.
	// Only the dimensions of a Private Data Block meets all the conditions in conditions field,
	// will this data block be considered to be allocated by the scheduler.
	// +optional
	Conditions []Condition `json:"conditions,omitempty"`

	// at least how much data are required in this allocation
	// +optional
	MinNumberOfData int `json:"minNumberOfData,omitempty"`

	// at least how many data blocks should be allocated in this request
	// +optional
	MinNumberOfBlocks int `json:"minNumberOfBlocks,omitempty"`

	// at most how many data blocks should be allocated in this request
	// This is useful when we use DFP scheduling policy, and the scheduler
	// won't use too many data blocks to compute the dominant share of a budget
	// +optional
	MaxNumberOfBlocks int `json:"maxNumberOfBlocks,omitempty"`

	// how much privacy should be allocated per data block
	// +optional
	MinBudget BudgetRequest `json:"minBudget"`

	// how much privacy is expected to be allocated per data block
	// +optional
	ExpectedBudget BudgetRequest `json:"expectedBudget"`

	// policy specifies the policy of this request. Here are the valid flags:
	//
	// (1) allOrNothing=true:
	// either every data blocks meeting the conditions of a request are acquired budgets or None of them.
	// If it fails, it will mark the success field in the response to false.
	// The scheduler guarantees an atomic semantic when this flag is on.
	//
	// (2) allOrNothing=false: (default option for allOrNothing)
	// the scheduler handles the Allocate request to data blocks individually. Some of them may succeed,
	// while some may not. Despite how many data blocks are allocated, the handling of request is always
	// considered as success.
	// +optional
	Policy string `json:"policy,omitempty"`

	// How much time a claim would like to wait for the allocation. If no value is specified, default value of
	// the scheduler will be used. This parameter is used in the scheduling policy like DPF, FLow-based.
	Timeout int64 `json:"timeout,omitempty"`
}

// only one field of NumericValue and StringValue can be not null.
// If NumericValue is filled in, the allowed operations are
// <, <=, =, == (equivalent to =), >, >=.
// An example is
// ```
//  attribute: startTime
//  numericValue: "0"
//  operation: ">="
// ```
// which means dataBlock.startTime >= 0.
//
// If StringValue is used, the allowed operations are "is", "include".
// "is" means equality between two strings. "include" means the condition of a data block is included in
// the StringValue, which could be represented by [a, b, c]. where a, b, c and string without quotes.
type Condition struct {
	Attribute    string           `json:"attribute"`
	Operation    string           `json:"operation"`
	NumericValue *decimal.Decimal `json:"numericValue,omitempty"`
	StringValue  string           `json:"stringValue,omitempty"`
}

func (in *Condition) DeepCopyInto(out *Condition) {
	*out = *in
	return
}

// only one field is allowed among constant, function, and budgetMap.
type BudgetRequest struct {
	Constant *PrivacyBudget `json:"constant,omitempty"`
	Function string         `json:"function,omitempty"`

	// specify the budget request per data block
	// and only the data blocks meeting all the conditions of this request and inside this budget map
	// are considered valid
	BudgetMap map[string]PrivacyBudget `json:"budgetMap,omitempty"`
}

type BudgetAccount struct {
	Acquired PrivacyBudget `json:"acquired,omitempty"`
	Reserved PrivacyBudget `json:"reserved,omitempty"`
}

type ConsumeRequest struct {
	// Policy specifies the policy of this request. Here are the valid flags:
	//
	//(1) allOrNothing=true:
	// either every data blocks to be consumed are consumed successfully or none of them. If it fails,
	// it will mark the success field in the response to false.
	// The scheduler guarantees an atomic semantic when this flag is on.
	//
	//(2) allOrNothing=false: (default option for allOrNothing) the scheduler handles the Consume request to
	// data blocks individually. Some of them may succeed, while some may not.
	// Despite how many data blocks are consumed, the handling of request is always considered as success.
	// +optional
	Policy string `json:"policy"`

	// specify the request to consume privacy budgets
	// +optional
	Consume map[string]PrivacyBudget `json:"consume,omitempty"`
}

type ReleaseRequest struct {
	// currently no policy is supported by abort quest
	// please leave this field empty.
	// +optional
	Policy string `json:"policy"`

	// +optional
	Release map[string]PrivacyBudget `json:"release,omitempty"`
}

// AllocateResponse, ConsumeResponse, and ReleaseResponse stores the detail of this response.
// Only one of the three fields can be not null.
type Response struct {
	// this identifier should be identical to the request this response answers.
	Identifier string `json:"identifier,omitempty"`

	// indicate if this response has succeeded
	State string `json:"state"`

	// show the error message if it fails
	// +optional
	Error string `json:"error,omitempty"`

	AllocateResponse *AllocateResponse `json:"allocateResponse,omitempty"`
	ConsumeResponse  ConsumeResponse   `json:"consumeResponse,omitempty"`
	ReleaseResponse  ReleaseResponse   `json:"releaseResponse,omitempty"`
}

type AllocateResponse struct {
	Budgets map[string]BudgetAccount `json:"budgets"`
	// start timestamp of the pending Allocate response
	StartTime  int64 `json:"startTime,omitempty"`
	FinishTime int64 `json:"finishTime,omitempty"`
}

type ConsumeResponse map[string]BudgetErrorPair

type BudgetAccountUpdate struct {
	Acquired *BudgetErrorPair `json:"acquired,omitempty"`
	Reserved *BudgetErrorPair `json:"reserved,omitempty"`
}

type BudgetErrorPair struct {
	Budget PrivacyBudget `json:"budget"`
	Error  string        `json:"error"`
}

type ReleaseResponse map[string]BudgetErrorPair

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object
// no client needed for list as it's been created in above
type PrivacyBudgetClaimList struct {
	metav1.TypeMeta `json:",inline"`
	// +optional
	metav1.ListMeta `json:"metadata,omitempty"`

	Items []PrivacyBudgetClaim `json:"items"`
}
