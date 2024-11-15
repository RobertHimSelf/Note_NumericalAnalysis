#!/usr/bin/python
# -*- coding: utf-8 -*-

try:
    from .ode__typing import X,Y
except:
    from ode__typing import X,Y

from typing import Callable
from numbers import Number
import numpy as np

def _calc(seq:np.ndarray, func:Callable[[np.number],np.number]):
    '''Apply the function for every variable in the array {seq}. And return it's result.'''
    return np.array([(_calc(obj) if isinstance(obj, np.ndarray) else func(obj)) for obj in seq])

def geterror(result:tuple[list[X],list[Y]], actuall_solution:Callable[[X],Y]):
    x = np.array(result[0])
    y_= np.array(result[1])
    y = _calc(x, actuall_solution)
    error = _calc(y-y_, abs)
    return max(error)

def test(solver:Callable, actuall_solution:Callable[[X],Y], individual_kwargs:list[dict], **solver_kwargs):
    print("开始测试，固定参数为：")
    for args in solver_kwargs:
        print(f"{args}:  {solver_kwargs[args]},")
    for args in individual_kwargs:
        result = solver(**solver_kwargs, **args)
        error = geterror(result, actuall_solution)
        print(f"{solver.__name__} 在参数 {args} 下具有最大误差：\n {error}")
