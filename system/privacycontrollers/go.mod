module columbia.github.com/privatekube/privacycontrollers

go 1.14

require (
	columbia.github.com/privatekube/privacyresource v0.0.0
	github.com/shopspring/decimal v1.2.0
	github.com/stretchr/testify v1.4.0
	k8s.io/api v0.18.0 // indirect
	k8s.io/apiextensions-apiserver v0.17.3 // indirect
	k8s.io/apimachinery v0.18.0
	k8s.io/client-go v0.18.0
	k8s.io/klog v1.0.0
)

replace columbia.github.com/privatekube/privacyresource => ../privacyresource
