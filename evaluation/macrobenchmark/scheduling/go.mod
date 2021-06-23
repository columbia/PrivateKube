module columbia.github.com/privatekube/evaluation/macrobenchmark/scheduling

go 1.14

require (
	columbia.github.com/privatekube/dpfscheduler v0.0.0
	columbia.github.com/privatekube/privacycontrollers v0.0.0
	columbia.github.com/privatekube/privacyresource v0.0.0
	github.com/shopspring/decimal v1.2.0
	github.com/stretchr/testify v1.4.0 // indirect
	gopkg.in/yaml.v2 v2.2.8
	k8s.io/api v0.18.0
	k8s.io/apimachinery v0.18.0
	k8s.io/client-go v0.18.0
	k8s.io/klog v1.0.0
)

replace columbia.github.com/privatekube/privacyresource => ../../../system/privacyresource

replace columbia.github.com/privatekube/privacycontrollers => ../../../system/privacycontrollers

replace columbia.github.com/privatekube/dpfscheduler => ../../../system/dpfscheduler
