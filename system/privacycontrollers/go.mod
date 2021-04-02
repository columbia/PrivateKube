module columbia.github.com/sage/privacycontrollers

go 1.14

require (
	columbia.github.com/sage/privacyresource v0.0.0
	github.com/stretchr/testify v1.4.0 // indirect
	k8s.io/api v0.18.0 // indirect
	k8s.io/apiextensions-apiserver v0.17.3 // indirect
	k8s.io/apimachinery v0.18.0
	k8s.io/client-go v0.18.0
	k8s.io/klog v1.0.0
)

replace columbia.github.com/sage/privacyresource => ../privacyresource
