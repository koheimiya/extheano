# README #

### What is this repository for? ###

* Easy Theano for easy use, based on the decorator.
* Less symbol variables.
* No more explicit instructions to compile your Theano codes.

### How can I make use of this? ###

* A tiny example here:  
```python
import extheano  
import theano.tensor as T  

# computes RMS
@extheano.jit
def rms(arr):
    return T.sqrt(T.mean(arr ** 2))

f([1., 2.]) # <-- Implicit compilation here
# (Note: the type of the input is fixed to 1darray of float on the first call)  
# -> array(1.5811388300841898)  
  
f([3., 4., 5.]) # <-- Pre-compiled function is used  
# -> array(4.08248290463863)  
```  
  
* Of course you can employ shared variables, without any explicit 'update' instructions.  
```python
import theano

# objective function to minimize
def f(x):
    return (x - 10.) ** 2

# initial guess
# NodeBuffers wrap shared variables
x0 = extheano.NodeBuffer(theano.shared(0., 'minimizer'))

# one step of gradient decent method
@extheano.jit
def update_x0():
    # access the current value of a NodeBuffer through its attribute `val`
    x0.val += 0.05 * T.grad(f(x0.val), x0.val)

# iteratively updates `x0`
itermax = 100
for i in xrange(itermax):
    update_x0()

print x0.get_value()
```

* Consider adopting `NodeDescriptor` with OOP for more intuitive experience.  
```python
import numpy
import theano
import theano.tensor as T
import theano.tensor.nlinalg as Tnlinalg
import extheano

# Example of using NodeDescriptors
class GaussianModel(object):
    ''' Calculates the likelihood and fits the parameters
        of the multivariate Gaussian distribution.
    '''

    # Each NodeDescriptor represent a single shared variable.
    mu = extheano.NodeDescriptor()
    sigma = extheano.NodeDescriptor()

    def __init__(self, mu0, sigma0):
        # Recommended is to initialize NodeDescriptors in `__init__`
        self.mu = theano.shared(mu0, 'mu')
        self.sigma = theano.shared(sigma0, 'sigma')

    @extheano.jit
    def loglikelihood(self, data):
        ''' Calculates the logarithm of the likelihood. '''
        x = data - self.mu
        lam = Tnlinalg.matrix_inverse(self.sigma)
        det = Tnlinalg.det(self.sigma)
        likelihoods = -0.5 * T.dot(x, T.dot(lam, x.T)) - 0.5 * T.log(2. * numpy.pi * det)
        return T.sum(likelihoods)

    @extheano.jit
    def fit(self, data):
        ''' Updates the parameters to the MLE. '''
        n = data.shape[0]
        # Note: you can assign new values directly to NodeDescriptors
        self.mu = T.mean(data, axis=0)
        self.sigma = (T.dot(data.T, data) - T.outer(self.mu, self.mu)) / n

    @extheano.jit
    def get_param(self):
        ''' Gets parameters. '''
        return self.mu, self.sigma

    @extheano.jit
    def set_param(self, mu, sigma):
        ''' Sets parameters. '''
        self.mu = mu
        self.sigma = sigma


# parameter-fitting demo
data = [[1., 2.], [0., 0.], [1., 1.]]
model = GaussianModel(mu0=numpy.zeros(2), sigma0=numpy.eye(2))

print 'Before fit:', model.loglikelihood(data)
model.fit(data)
print 'After fit:', model.loglikelihood(data)

mu, sigma = model.get_param()
print 'MLE: mu=%s, sigma=%s' %(repr(mu), repr(sigma)) 
```
  
* See tutorial\_test.py to use `scan` or to parse arguments with extheano.

### Who do I talk to? ###

* Kohei Miyaguchi (quote.curly@gmail.com)

### Copyright and License ###

* This package is released under the MIT License, see LICENSE.txt.
