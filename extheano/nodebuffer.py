""" Provide an interface of the shared variables """
import gc
from collections import OrderedDict

import theano
from theano.tensor.sharedvar import SharedVariable
from theano import Variable as theano_Variable


class NodeBuffer(object):
    """ Wrap a shared variable and buffer the assignments on it. """
    def __init__(self, shared_variable):
        assert isinstance(shared_variable, SharedVariable)
        self._node = shared_variable
        self._buffer = shared_variable

    def get(self):
        return self._buffer

    def set(self, value):
        assert isinstance(value, theano_Variable)
        self._buffer = value

    @property
    def val(self):
        return self.get()

    @val.setter
    def val(self, val):
        self.set(val)

    def _extract_updater(self):
        """ Get the update and reset the buffer.
        Return `None` if there is no update. """
        if self._node is self._buffer:
            return None
        else:
            updater = (self._node, self._buffer)
            self._buffer = self._node
            return updater


class NodeDescriptor(object):
    """ A descriptor that provides us a simple access to `NodeBuffer`"""
    def __init__(self):
        self._buffer_dict = dict()

    def __get__(self, obj, cls=None):
        if obj is None:
            # via class
            return self
        assert obj in self._buffer_dict
        return self._buffer_dict[obj].get()

    def __set__(self, obj, value):
        if obj in self._buffer_dict:
            self._buffer_dict[obj].set(value)
        else:
            self._buffer_dict[obj] = NodeBuffer(value)


class BufferSet(object):
    """ Pool of `NodeBuffer`s. Another access."""
    def __init__(self, **key_node_pairs):
        self.__dict__['_buffer_dict'] = {
                key: NodeBuffer(node)
                for key, node in key_node_pairs.items()}
        self.__dict__['_reg_lock'] = False

    def lock(self):
        self._reg_lock = True

    def unlock(self):
        self._reg_lock = False

    def __getattr__(self, name):
        buffer_dict = self._buffer_dict
        if name in buffer_dict:
            return buffer_dict[name].get()
        else:
            raise AttributeError("%s" % name)

    def __setattr__(self, name, value):
        buffer_dict = self._buffer_dict
        if name in buffer_dict:
            buffer_dict[name].set(value)
        elif not self._reg_lock:
            buffer_dict[name] = NodeBuffer(value)
        else:
            raise AttributeError("%s" % name)


class Scanner(object):
    """ Wrapper of `theano.scan` """
    _updates = OrderedDict()

    @classmethod
    def scan(cls, *args, **kwargs):
        outputs, new_updates = theano.scan(*args, **kwargs)
        cls._updates.update(new_updates)
        return outputs

    @classmethod
    def _extract_updates(cls):
        tmp = cls._updates
        cls._updates = OrderedDict()
        return tmp


class UpdateCollector(object):
    """ Collect updates from `NodeBuffer`s """
    def __init__(self):
        self.buffer_list = []
        for obj in gc.get_objects():
            if isinstance(obj, NodeBuffer):
                obj._extract_updater()
                self.buffer_list.append(obj)

        Scanner._extract_updates()

    def extract_updates(self):
        updates = Scanner._extract_updates()
        shared_updates = [buf._extract_updater() for buf in self.buffer_list]
        updates.update([upd for upd in shared_updates if upd is not None])
        return updates


