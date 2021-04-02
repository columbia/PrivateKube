module columbia.github.com/sage/evaluation/examples

go 1.14

require (
	columbia.github.com/sage/dpfscheduler v0.0.0
	columbia.github.com/sage/privacycontrollers v0.0.0
	columbia.github.com/sage/privacyresource v0.0.0
	github.com/shopspring/decimal v1.2.0
	github.com/stretchr/testify v1.4.0
	k8s.io/api v0.18.0
	k8s.io/apimachinery v0.18.0
	k8s.io/client-go v0.18.0
)

replace columbia.github.com/sage/privacyresource => ../../system/privacyresource

replace columbia.github.com/sage/privacycontrollers => ../../system/privacycontrollers

replace columbia.github.com/sage/dpfscheduler => ../../system/dpfscheduler
