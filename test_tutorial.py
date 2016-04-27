'''Includes the tutorial code and the test case'''

from sys import path
path.insert(0, '.')

import numpy as np
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

import extheano


class AverageSGM(object):

    ''' A toy example showing the usage of `extheano.NodeDescriptor` and
    `extheano.jit`.
    This class performs the stochastic gradient method (SGM) to find the
    average of given data.

    Usage:
        >> data = np.arange(1000)
        >> a = AverageSGM(data)
        >> for _ in xrange(10000): a.calc_loss_with_onestep_SGM()
        >> est = a.get_estimation()
    '''

    # node descriptors for shared variables
    # whole data of which we will compute the average
    data = extheano.NodeDescriptor()
    # estimate of the average
    mu = extheano.NodeDescriptor()
    # learning rate (will be discounted as SGM goes on)
    lrate = extheano.NodeDescriptor()

    def __init__(self, data, batch_size=10, init_val=0., lrate=0.05,
                 degree=0.75, seed=None):
        '''Set parameters for the SGM

        :param data:        array-like with its dimension one
        :param batch_size:  size of the mini batch in integer
        :param init_val:    initial guess of the average in float
        :param lrate:       initial learning rate in float
        :param degree:      degree of learning rate decreasing in float
        :param seed:        seed for RNG in integer
        '''

        # pure-python variables (assumed to be invariant until recompilation)
        self.batch_size = batch_size
        self.n_batches = len(data) / batch_size
        self.degree = degree
        self.init_lrate = lrate

        # initialize the nodes
        self.data = theano.shared(data.astype(float), 'data', borrow=True)
        self.mu = theano.shared(float(init_val), 'mu')
        self.lrate = theano.shared(float(lrate), 'lrate')

        # shared random streams
        self.rng = RandomStreams(seed)

    def quadratic_loss(self, minibatch):
        '''Get the quadratic loss against the given input'''
        return ((minibatch - self.mu) ** 2).mean()

    def gradient_descent(self, loss, lrate):
        '''Perform one step of the gradient descent on the given loss

        Note that you can update `self.mu` with the normal assignment
        operation since it is a descriptor.
        '''
        # calculate the gradient
        grad = -T.grad(loss, self.mu)
        # update the estimation
        self.mu = self.mu + lrate * grad

    def next_lrate(self, lr):
        '''Return the discounted learning rate

        The learning rate will be proportional to the number of iterations with
        minus `self.degree` on the exponent.
        '''
        time = (self.init_lrate / lr) ** (1. / self.degree)
        ratio = (1. - 1. / (1. + time)) ** self.degree
        return lr * ratio

    # With the decorator `@extheano.jit`, you can compile your theano-function
    # 'just in time'. Use `@extheano.jit.parse` instead if it has arguments with
    # default values.
    @extheano.jit.parse
    def calc_loss_with_onestep_SGM(self, scale=1.):
        '''Calculate the quadratic loss and perform one step of the SGM
        '''
        # assign a random batch to the input
        batch_start = self.batch_size * \
            self.rng.random_integers(low=0, high=self.n_batches - 1)
        batch_stop = batch_start + self.batch_size
        minibatch = self.data[batch_start: batch_stop]

        # perform SGM and discount the learning rate
        loss = self.quadratic_loss(minibatch)
        self.gradient_descent(loss, self.lrate * scale)
        self.lrate = self.next_lrate(self.lrate)
        return loss

    @extheano.jit
    def set_estimation(self, val):
        '''Set the estimation of the average'''
        self.mu = T.cast(val, theano.config.floatX)

    @extheano.jit
    def get_estimation(self):
        '''Get the estimation of the average'''
        return self.mu


def test_function_wrapping():
    func = extheano.jit(lambda a, b: a + b)
    assert 3 == func(1, 2)


def test_function_decoration():
    @extheano.jit
    def func(a, b):
        return a + b

    assert 3 == func(1, 2)


def test_function_wrapping_kwargs():
    func = extheano.jit.parse(lambda a, b=100: a + b)
    assert 101 == func(1)
    assert 3 == func(1, 2)
    assert 300 == func(a=200)


def test_higher_dim_args():
    func = extheano.jit(lambda a, b: a + b)
    assert np.asarray([[3.]]).ndim == func([[1]], [2.]).ndim


def test_recompile():
    func = extheano.jit(lambda a, b: a + b)
    assert 3 == func(1, 2)
    func.recompile()
    assert 3 == func(1, 2)


def test_member_method_decoration():
    seq = np.arange(1000)
    tol = 1e-3
    n_iter = 10000
    seed = 1234
    a = AverageSGM(seq, seed=seed)
    a.set_estimation(0.)
    true_result = 485.960824876

    for _ in range(n_iter):
        a.calc_loss_with_onestep_SGM(1.)

    a.set_estimation.recompile()

    gotten = a.get_estimation()
    assert (true_result - gotten).max() < tol
