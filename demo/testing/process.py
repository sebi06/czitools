# -*- coding: utf-8 -*-

#################################################################
# File        : process.py
# Author      : sebi06
#
# Disclaimer: This code is purely experimental. Feel free to
# use it at your own risk.
#
#################################################################

import dask.array as da
from skimage.measure import label
import numpy as np
import itertools
from typing import List, NamedTuple, Union, Tuple, Callable

"""
def my_decorator(func):
    def wrapper(*args, **kwargs):
        print("Before function call")
        result = func(*args, **kwargs)
        print("After function call")
        return result
    return wrapper

def my_function(x, y):
    return x + y

# Dynamic decoration
decorated_function = my_decorator(my_function)

result = decorated_function(2, 3)
print(result)

or 

def my_decorator(func):
    # ... (same as before)

@my_decorator
def my_function(x, y):
    return x + y

result = my_function(2, 3)
"""

def process_nd(func):
    def wrapper(*args, **kwargs):

        print("Before function call")
        arg_names = inspect.getfullargspec(func).args

        shape_woXY = args[0].shape[:-2]

        processed_md = da.zeros_like(args[0], chunks=args[0].shape)

        # create the "values" each for-loop iterates over
        loopover = [range(s) for s in shape_woXY]
        prod = itertools.product(*loopover)

        # loop over all dimensions
        for idx in prod:

            # create list of slice objects based on the shape
            sl = len(shape_woXY) * [np.s_[0:1]]

            # insert the correct index into the respective slice objects for all dimensions
            for nd in range(len(shape_woXY)):
                sl[nd] = idx[nd]

            # extract the 2D image from the n-dims stack using the list of slice objects
            processed_2d = np.squeeze(args[0][tuple(sl)])

            # process the whole 2d image - make sure to use the correct **kwargs
            processed = func(*args, **kwargs)

            # insert new 2D after tile-wise processing into nd array
            label_complete[tuple(sl)] = new_label2d

            print("Datatype Labels", type(label_complete))




        result = func(*args, **kwargs)
        print("After function call")
        return result
    return wrapper

    shape_woXY = seg.shape[:-2]

    label_complete = da.zeros_like(seg, chunks=seg.shape)

    # create the "values" each for-loop iterates over
    loopover = [range(s) for s in shape_woXY]
    prod = itertools.product(*loopover)

    # loop over all dimensions
    for idx in prod:

        # create list of slice objects based on the shape
        sl = len(shape_woXY) * [np.s_[0:1]]

        # insert the correct index into the respective slice objects for all dimensions
        for nd in range(len(shape_woXY)):
            sl[nd] = idx[nd]

        # extract the 2D image from the n-dims stack using the list of slice objects
        seg2d = np.squeeze(seg[tuple(sl)])

        # process the whole 2d image - make sure to use the correct **kwargs
        new_label2d = label((seg2d == labelvalue).astype(float))

        # insert new 2D after tile-wise processing into nd array
        label_complete[tuple(sl)] = new_label2d

        print("Datatype Labels", type(label_complete))

    return label_complete


import inspect

def my_decorator(func):
    def wrapper(*args, **kwargs):
        arg_names = inspect.getfullargspec(func).args
        for arg_name, arg_value in zip(arg_names, args):
            print(f"Argument {arg_name}: {arg_value}")
        return func(*args, **kwargs)
    return wrapper

@my_decorator
def my_function1(n1:int = 0, n2: int = 1) -> int:
    return n1 + n2

@my_decorator
def my_function2(x: int, y: int) -> int:
    return x + y

result1 = my_function1(2, 3)
print(result1)

result2 = my_function2(4, 7)
print(result2)