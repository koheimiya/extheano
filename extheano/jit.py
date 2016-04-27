'''Classes related to the auto-compilation'''

import copy
import inspect
import functools
# import types

import numpy as np
import theano
import theano.tensor as T

from .nodebuffer import UpdateCollector


class JITCompiler(object):

    '''Decorator for your theano-function

    You can call your theano-function without any explicit instructions to compile it. It takes a
    while for the first time. Note that the type of arguments will be fixed on
    the first call.

    Usage:
        >> f = JITCompiler( lambda x: T.sqrt((x ** 2).mean()) )

        >> f([1., 2.]) # <-- implicit compilation here
            array(1.5811388300841898)

        >> f([3., 4., 5.]) # <-- pre-compiled function is used
            array(4.08248290463863)
    '''

    parse = None  # parser will be assigned here

    def __init__(self, func, owner=None):
        '''Initialize the members given the decorated function'''
        functools.update_wrapper(self, func)
        self.raw_func = func
        self.owner = owner  # owner of this instance
        self.compiled_func = None  # compiled function

    def __call__(self, *args):
        '''Call the compiled function after its compilation '''
        # compile the function
        if self.compiled_func is None:
            self.compiled_func = Compiler().compile_with_value(
                self.raw_func, args, self.owner
            )

        return self.compiled_func(*args)

    def __get__(self, obj, objtype=None):
        '''Support decorating instance methods'''

        # bypass to the instance's attribute
        if obj is not None:
            # create the attribute for the first time
            name = self.raw_func.__name__
            wrapped_func = JITCompiler(self.raw_func, owner=obj)
            setattr(obj, name, wrapped_func)
            return wrapped_func

        # work as descriptor
        elif self.owner is None:
            self.owner = objtype

        return self

    def recompile(self):
        '''Lazy re-compilation'''
        self.compiled_func = None


class ParsingJITCompiler(JITCompiler):

    '''JITCompiler with a new feature: argument parsing

    Now you can pass keyword arguments to the function
    '''

    def __init__(self, func, owner=None):
        '''Initialize the members given the decorated function'''
        functools.update_wrapper(self, func)
        self.rawinfo = FuncInfo(func)
        self.owner = owner  # owner of this instance
        self.compiled_func = None  # compiled function

        if owner is not None:
            self.rawinfo.remove_first_key()

    def __call__(self, *args, **kwargs):
        '''Call the compiled function after its compilation '''
        # parse the arguments with their keywords
        if self.rawinfo.has_default_arg():
            args = self.rawinfo.parse_args_kwargs(*args, **kwargs)

        # compile the function
        if self.compiled_func is None:
            self.compiled_func = Compiler().compile_with_value(
                self.rawinfo.func, args, self.owner
            )

        return self.compiled_func(*args)

    def __get__(self, obj, objtype=None):
        '''Support decorating instance methods'''
        # bypass to the instance's attribute
        if obj is not None:
            # create and set the new auto-compiler as an attribute of the instance
            name = self.rawinfo.func.__name__
            wrapped_func = ParsingJITCompiler(self.rawinfo.func, owner=obj)
            setattr(obj, name, wrapped_func)
            return wrapped_func

        # if the owner is a class
        elif self.owner is None:
            self.owner = objtype
            self.rawinfo.remove_first_key()

        return self


class FuncInfo(object):

    '''Container of a function and its information'''

    def __init__(self, func):
        self.func = func
        self.arginfo = self._get_keys_defdict()  # arguments info

    def has_default_arg(self):
        '''If there are any arguments with default value or not'''
        return (self.arginfo[1] is not None)

    def remove_first_key(self):
        '''remove the key of the first argument from arginfo'''
        self.arginfo = (self.arginfo[0][1:], self.arginfo[1])

    def parse_args_kwargs(self, *args, **kwargs):
        '''Parse the arguments with keywords.'''
        # unpack the arginfo
        keys, defdict = self.arginfo
        assigned = keys[:len(args)]
        not_assigned = keys[len(args):]

        # validate kwargs
        for key in kwargs:
            assert key not in assigned
            assert key in keys

        # integrate args and kwargs
        knowns = dict(defdict, **kwargs)
        parsed_args = args + tuple([knowns[key] for key in not_assigned])
        return parsed_args

    def _get_keys_defdict(self):
        '''Get the keys and the default dictionary of the given function's
        arguments
        '''
        # inspect argspecs
        argspec = inspect.getargspec(self.func)
        keys, defvals = argspec.args, argspec.defaults

        # convert to (list_of_argkeys, dict_of_default_keys)
        if defvals is None:
            return keys, None
        else:
            defvals = list(defvals)
            keys.reverse()
            defvals.reverse()
            defdict = dict(zip(keys, defvals))
            keys.reverse()
            return keys, defdict


class Compiler(object):

    '''Compile the theano-function/method just with its arguments and owner
    '''

    # default options for the compilation
    default_options = {'on_unused_input': 'warn'}

    def compile_with_value(self, func, args=None, owner=None):
        '''Compile the function with array-like objects'''
        # format args
        if args is None:
            args = []

        # cast numpy.ndarray into theano.tensor
        theano_args = [self.cast2theano_var(a, 'extheano.jit.Compiler-arg-%d' % i)
                       for a, i in zip(args, range(len(args)))]

        # compiled value with symbol
        return self.compile_with_symbol(func, theano_args, owner)

    def compile_with_symbol(self, func, theano_args=None, owner=None):
        '''Compile the function with theano symbols'''
        if theano_args is None:
            theano_args = []

        # initialize the shared buffers
        upc = UpdateCollector()

        # get the output symbols and other Theano options
        theano_ret = func(*theano_args) if owner is None \
            else func(owner, *theano_args)

        # integrate the information of updates, givens and the other options
        out = copy.copy(self.default_options)
        out['outputs'] = theano_ret
        out['updates'] = upc.extract_updates()

        # compile the function
        return theano.function(theano_args, **out)

    def cast2theano_var(self, array_like, name=None):
        '''Cast `numpy.ndarray` into `theano.tensor` keeping `dtype` and `ndim`
        compatible
        '''
        # extract the information of the input value
        array = np.asarray(array_like)
        args = (name, array.dtype)
        ndim = array.ndim

        # cast with the information above
        if ndim == 0:
            return T.scalar(*args)
        elif ndim == 1:
            return T.vector(*args)
        elif ndim == 2:
            return T.matrix(*args)
        elif ndim == 3:
            return T.tensor3(*args)
        elif ndim == 4:
            return T.tensor4(*args)
        else:
            raise ValueError('extheano.jit.Compiler: Unsupported type or shape')


JITCompiler.parse = ParsingJITCompiler
