package framework

import (
	columbiav1 "columbia.github.com/sage/privacyresource/pkg/apis/columbia.github.com/v1"
	"strconv"
)

type RequestHandler struct {
	ClaimHandler ClaimHandler
	RequestIndex int
}

func (handler RequestHandler) GetId() string {
	claimId := handler.ClaimHandler.GetId()
	if claimId == "" {
		return ""
	}
	return claimId + "/" + strconv.Itoa(handler.RequestIndex)
}

func (handler RequestHandler) GetRequest() *columbiav1.Request {
	claim := handler.ClaimHandler.View()
	return &claim.Spec.Requests[handler.RequestIndex]
}

func (handler RequestHandler) Viewer() RequestViewer {
	return MakeRequestViewer(handler.ClaimHandler.View(), handler.RequestIndex)
}

func (handler RequestHandler) GetResponse() *columbiav1.Response {
	claim := handler.ClaimHandler.View()
	if len(claim.Status.Responses) > handler.RequestIndex {
		return &claim.Status.Responses[handler.RequestIndex]
	} else {
		return nil
	}
}

func (handler RequestHandler) IsNull() bool {
	return handler.ClaimHandler == nil || handler.ClaimHandler.View() == nil
}

func (handler RequestHandler) IsValid() bool {
	claim := handler.ClaimHandler.View()
	return claim != nil && handler.RequestIndex < len(claim.Spec.Requests)
}

func (handler RequestHandler) IsPending() bool {
	claim := handler.ClaimHandler.View()
	return claim != nil && handler.RequestIndex < len(claim.Status.Responses) &&
		claim.Status.Responses[handler.RequestIndex].State == columbiav1.Pending
}

type RequestViewer struct {
	Claim        *columbiav1.PrivacyBudgetClaim
	RequestIndex int
	Id           string
}

func MakeRequestViewer(claim *columbiav1.PrivacyBudgetClaim, index int) RequestViewer {
	return RequestViewer{
		Claim:        claim,
		RequestIndex: index,
		Id:           GetRequestId(claim, index),
	}
}

func GetRequestId(claim *columbiav1.PrivacyBudgetClaim, requestIndex int) string {
	return claim.Namespace + "/" + claim.Name + "/" + strconv.Itoa(requestIndex)
}

func (viewer RequestViewer) View() *columbiav1.Request {
	return &viewer.Claim.Spec.Requests[viewer.RequestIndex]
}

func (viewer RequestViewer) IsPending() bool {
	claim := viewer.Claim
	return claim != nil && viewer.RequestIndex < len(claim.Status.Responses) &&
		claim.Status.Responses[viewer.RequestIndex].State == columbiav1.Pending
}
