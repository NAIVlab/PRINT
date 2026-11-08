"""Attribute-setting decorator used by the continuous-action baseline."""
import functools

def autoargs(*include, **kwargs):
    import inspect
    def _autoargs(func):
        attrs, varargs, varkw, defaults, _, _, _ = inspect.getfullargspec(func)

        def sieve(attr):
            if kwargs and attr in kwargs['exclude']:
                return False
            if not include or attr in include:
                return True
            else:
                return False

        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            # handle default values
            if defaults:
                for attr, val in zip(reversed(attrs), reversed(defaults)):
                    if sieve(attr):
                        setattr(self, attr, val)
            # handle positional arguments
            positional_attrs = attrs[1:]
            for attr, val in zip(positional_attrs, args):
                if sieve(attr):
                    setattr(self, attr, val)
            # handle varargs
            if varargs:
                remaining_args = args[len(positional_attrs):]
                if sieve(varargs):
                    setattr(self, varargs, remaining_args)
            # handle varkw
            if kwargs:
                for attr, val in kwargs.items():
                    if sieve(attr):
                        setattr(self, attr, val)
            return func(self, *args, **kwargs)
        return wrapper
    return _autoargs

