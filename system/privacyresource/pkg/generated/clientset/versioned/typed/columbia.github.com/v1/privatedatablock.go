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
	"context"
	"time"

	v1 "columbia.github.com/privatekube/privacyresource/pkg/apis/columbia.github.com/v1"
	scheme "columbia.github.com/privatekube/privacyresource/pkg/generated/clientset/versioned/scheme"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	types "k8s.io/apimachinery/pkg/types"
	watch "k8s.io/apimachinery/pkg/watch"
	rest "k8s.io/client-go/rest"
)

// PrivateDataBlocksGetter has a method to return a PrivateDataBlockInterface.
// A group's client should implement this interface.
type PrivateDataBlocksGetter interface {
	PrivateDataBlocks(namespace string) PrivateDataBlockInterface
}

// PrivateDataBlockInterface has methods to work with PrivateDataBlock resources.
type PrivateDataBlockInterface interface {
	Create(ctx context.Context, privateDataBlock *v1.PrivateDataBlock, opts metav1.CreateOptions) (*v1.PrivateDataBlock, error)
	Update(ctx context.Context, privateDataBlock *v1.PrivateDataBlock, opts metav1.UpdateOptions) (*v1.PrivateDataBlock, error)
	UpdateStatus(ctx context.Context, privateDataBlock *v1.PrivateDataBlock, opts metav1.UpdateOptions) (*v1.PrivateDataBlock, error)
	Delete(ctx context.Context, name string, opts metav1.DeleteOptions) error
	DeleteCollection(ctx context.Context, opts metav1.DeleteOptions, listOpts metav1.ListOptions) error
	Get(ctx context.Context, name string, opts metav1.GetOptions) (*v1.PrivateDataBlock, error)
	List(ctx context.Context, opts metav1.ListOptions) (*v1.PrivateDataBlockList, error)
	Watch(ctx context.Context, opts metav1.ListOptions) (watch.Interface, error)
	Patch(ctx context.Context, name string, pt types.PatchType, data []byte, opts metav1.PatchOptions, subresources ...string) (result *v1.PrivateDataBlock, err error)
	PrivateDataBlockExpansion
}

// privateDataBlocks implements PrivateDataBlockInterface
type privateDataBlocks struct {
	client rest.Interface
	ns     string
}

// newPrivateDataBlocks returns a PrivateDataBlocks
func newPrivateDataBlocks(c *ColumbiaV1Client, namespace string) *privateDataBlocks {
	return &privateDataBlocks{
		client: c.RESTClient(),
		ns:     namespace,
	}
}

// Get takes name of the privateDataBlock, and returns the corresponding privateDataBlock object, and an error if there is any.
func (c *privateDataBlocks) Get(ctx context.Context, name string, options metav1.GetOptions) (result *v1.PrivateDataBlock, err error) {
	result = &v1.PrivateDataBlock{}
	err = c.client.Get().
		Namespace(c.ns).
		Resource("privatedatablocks").
		Name(name).
		VersionedParams(&options, scheme.ParameterCodec).
		Do(ctx).
		Into(result)
	return
}

// List takes label and field selectors, and returns the list of PrivateDataBlocks that match those selectors.
func (c *privateDataBlocks) List(ctx context.Context, opts metav1.ListOptions) (result *v1.PrivateDataBlockList, err error) {
	var timeout time.Duration
	if opts.TimeoutSeconds != nil {
		timeout = time.Duration(*opts.TimeoutSeconds) * time.Second
	}
	result = &v1.PrivateDataBlockList{}
	err = c.client.Get().
		Namespace(c.ns).
		Resource("privatedatablocks").
		VersionedParams(&opts, scheme.ParameterCodec).
		Timeout(timeout).
		Do(ctx).
		Into(result)
	return
}

// Watch returns a watch.Interface that watches the requested privateDataBlocks.
func (c *privateDataBlocks) Watch(ctx context.Context, opts metav1.ListOptions) (watch.Interface, error) {
	var timeout time.Duration
	if opts.TimeoutSeconds != nil {
		timeout = time.Duration(*opts.TimeoutSeconds) * time.Second
	}
	opts.Watch = true
	return c.client.Get().
		Namespace(c.ns).
		Resource("privatedatablocks").
		VersionedParams(&opts, scheme.ParameterCodec).
		Timeout(timeout).
		Watch(ctx)
}

// Create takes the representation of a privateDataBlock and creates it.  Returns the server's representation of the privateDataBlock, and an error, if there is any.
func (c *privateDataBlocks) Create(ctx context.Context, privateDataBlock *v1.PrivateDataBlock, opts metav1.CreateOptions) (result *v1.PrivateDataBlock, err error) {
	result = &v1.PrivateDataBlock{}
	err = c.client.Post().
		Namespace(c.ns).
		Resource("privatedatablocks").
		VersionedParams(&opts, scheme.ParameterCodec).
		Body(privateDataBlock).
		Do(ctx).
		Into(result)
	return
}

// Update takes the representation of a privateDataBlock and updates it. Returns the server's representation of the privateDataBlock, and an error, if there is any.
func (c *privateDataBlocks) Update(ctx context.Context, privateDataBlock *v1.PrivateDataBlock, opts metav1.UpdateOptions) (result *v1.PrivateDataBlock, err error) {
	result = &v1.PrivateDataBlock{}
	err = c.client.Put().
		Namespace(c.ns).
		Resource("privatedatablocks").
		Name(privateDataBlock.Name).
		VersionedParams(&opts, scheme.ParameterCodec).
		Body(privateDataBlock).
		Do(ctx).
		Into(result)
	return
}

// UpdateStatus was generated because the type contains a Status member.
// Add a +genclient:noStatus comment above the type to avoid generating UpdateStatus().
func (c *privateDataBlocks) UpdateStatus(ctx context.Context, privateDataBlock *v1.PrivateDataBlock, opts metav1.UpdateOptions) (result *v1.PrivateDataBlock, err error) {
	result = &v1.PrivateDataBlock{}
	err = c.client.Put().
		Namespace(c.ns).
		Resource("privatedatablocks").
		Name(privateDataBlock.Name).
		SubResource("status").
		VersionedParams(&opts, scheme.ParameterCodec).
		Body(privateDataBlock).
		Do(ctx).
		Into(result)
	return
}

// Delete takes name of the privateDataBlock and deletes it. Returns an error if one occurs.
func (c *privateDataBlocks) Delete(ctx context.Context, name string, opts metav1.DeleteOptions) error {
	return c.client.Delete().
		Namespace(c.ns).
		Resource("privatedatablocks").
		Name(name).
		Body(&opts).
		Do(ctx).
		Error()
}

// DeleteCollection deletes a collection of objects.
func (c *privateDataBlocks) DeleteCollection(ctx context.Context, opts metav1.DeleteOptions, listOpts metav1.ListOptions) error {
	var timeout time.Duration
	if listOpts.TimeoutSeconds != nil {
		timeout = time.Duration(*listOpts.TimeoutSeconds) * time.Second
	}
	return c.client.Delete().
		Namespace(c.ns).
		Resource("privatedatablocks").
		VersionedParams(&listOpts, scheme.ParameterCodec).
		Timeout(timeout).
		Body(&opts).
		Do(ctx).
		Error()
}

// Patch applies the patch and returns the patched privateDataBlock.
func (c *privateDataBlocks) Patch(ctx context.Context, name string, pt types.PatchType, data []byte, opts metav1.PatchOptions, subresources ...string) (result *v1.PrivateDataBlock, err error) {
	result = &v1.PrivateDataBlock{}
	err = c.client.Patch(pt).
		Namespace(c.ns).
		Resource("privatedatablocks").
		Name(name).
		SubResource(subresources...).
		VersionedParams(&opts, scheme.ParameterCodec).
		Body(data).
		Do(ctx).
		Into(result)
	return
}
