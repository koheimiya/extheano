""" Compute RMSE and compare the time to each other """

import time
import numpy as np
import theano
import theano.tensor as T
import extheano


def timer(f, x, n_loops):
    t0 = time.time()
    for i in xrange(n_loops):
        res = f(x)
    t1 = time.time()
    return t1 - t0, res

def get_numpy_func():
    return lambda x: np.sqrt(np.mean(x ** 2))

def get_extheano_func():
    return extheano.jit(lambda x: T.sqrt(T.mean(x ** 2)))

def get_theano_func():
    s = T.lvector()
    return theano.function([s], T.sqrt(T.mean(s ** 2)))


def main():
    funcs = {'numpy': get_numpy_func(),
             'theano': get_theano_func(),
             'extheano': get_extheano_func()}

    xs = [np.arange(l) for l in (10, 100, 1000)]
    n_loops = 10000

    for i in xrange(10):
        results = {key: [timer(func, x, 10000)[0] for x in xs]
                   for key, func in funcs.items()}


    print "iterated %d times:" % n_loops
    for item in results.items():
        print "%s: %s seconds elapsed" % item


if __name__ == '__main__':
    main()
