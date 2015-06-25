# README #

### What is this repository for? ###

* Easy Theano for easy use, based on the decorator.
* Less symbol variables.
* No more explicit instructions to compile your Theano codes.

### How can I make use of this? ###

* A tiny example here:  
    import extheano
    import theano.tensor as T
    f = extheano.jit( lambda x: T.sqrt((x ** 2).mean()) )
    f([1., 2.]) # <-- implicit compilation here
    # -> array(1.5811388300841898)
    f([3., 4., 5.]) # <-- pre-compiled function is used
    # -> array(4.08248290463863)  

* See tutorial\_test.py for the tutorial code.

### Who do I talk to? ###

* Kohei Miyaguchi (quote.curly@gmail.com)

### Copyright and License ###

* This package is released under the MIT License, see LICENSE.txt.
