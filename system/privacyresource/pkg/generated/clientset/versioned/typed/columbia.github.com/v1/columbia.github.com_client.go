/*
Copyright (c) 2020 Mingen Pan

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/
// Code generated by client-gen. DO NOT EDIT.

package v1

import (
	v1 "columbia.github.com/privatekube/privacyresource/pkg/apis/columbia.github.com/v1"
	"columbia.github.com/privatekube/privacyresource/pkg/generated/clientset/versioned/scheme"
	rest "k8s.io/client-go/rest"
)

type ColumbiaV1Interface interface {
	RESTClient() rest.Interface
	PrivacyBudgetClaimsGetter
	PrivateDataBlocksGetter
}

// ColumbiaV1Client is used to interact with features provided by the columbia.github.com group.
type ColumbiaV1Client struct {
	restClient rest.Interface
}

func (c *ColumbiaV1Client) PrivacyBudgetClaims(namespace string) PrivacyBudgetClaimInterface {
	return newPrivacyBudgetClaims(c, namespace)
}

func (c *ColumbiaV1Client) PrivateDataBlocks(namespace string) PrivateDataBlockInterface {
	return newPrivateDataBlocks(c, namespace)
}

// NewForConfig creates a new ColumbiaV1Client for the given config.
func NewForConfig(c *rest.Config) (*ColumbiaV1Client, error) {
	config := *c
	if err := setConfigDefaults(&config); err != nil {
		return nil, err
	}
	client, err := rest.RESTClientFor(&config)
	if err != nil {
		return nil, err
	}
	return &ColumbiaV1Client{client}, nil
}

// NewForConfigOrDie creates a new ColumbiaV1Client for the given config and
// panics if there is an error in the config.
func NewForConfigOrDie(c *rest.Config) *ColumbiaV1Client {
	client, err := NewForConfig(c)
	if err != nil {
		panic(err)
	}
	return client
}

// New creates a new ColumbiaV1Client for the given RESTClient.
func New(c rest.Interface) *ColumbiaV1Client {
	return &ColumbiaV1Client{c}
}

func setConfigDefaults(config *rest.Config) error {
	gv := v1.SchemeGroupVersion
	config.GroupVersion = &gv
	config.APIPath = "/apis"
	config.NegotiatedSerializer = scheme.Codecs.WithoutConversion()

	if config.UserAgent == "" {
		config.UserAgent = rest.DefaultKubernetesUserAgent()
	}

	return nil
}

// RESTClient returns a RESTClient that is used to communicate
// with API server by this client implementation.
func (c *ColumbiaV1Client) RESTClient() rest.Interface {
	if c == nil {
		return nil
	}
	return c.restClient
}
