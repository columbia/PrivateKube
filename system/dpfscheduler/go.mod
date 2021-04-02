module columbia.github.com/sage/dpfscheduler

go 1.14

require (
	columbia.github.com/sage/privacycontrollers v0.0.0
	columbia.github.com/sage/privacyresource v0.0.0
	github.com/google/pprof v0.0.0-20210226084205-cbba55b83ad5 // indirect
	github.com/shopspring/decimal v1.2.0 // indirect
	github.com/wangjohn/quickselect v0.0.0-20161129230411-ed8402a42d5f
	golang.org/x/tools v0.1.0 // indirect
	k8s.io/api v0.18.0
	k8s.io/apimachinery v0.18.0
	k8s.io/client-go v0.18.0
	k8s.io/klog v1.0.0
)

replace columbia.github.com/sage/privacyresource => ../privacyresource

replace columbia.github.com/sage/privacycontrollers => ../privacycontrollers
