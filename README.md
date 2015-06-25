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

# compute RMS
def rms(arr):
    return T.sqrt(T.mean(arr ** 2))

f = extheano.jit(rms)  
f([1., 2.]) # <-- Implicit compilation here
# (Note: the type of the input is fixed to 1darray of float on the first call)  
# -> array(1.5811388300841898)  
  
f([3., 4., 5.]) # <-- Pre-compiled function is used  
# -> array(4.08248290463863)  
```

* See tutorial\_test.py for more.

### Who do I talk to? ###

* Kohei Miyaguchi (quote.curly@gmail.com)

### Copyright and License ###

* This package is released under the MIT License, see LICENSE.txt.
