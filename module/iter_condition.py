#!/usr/bin/python
# -*- coding: utf-8 -*-
'''用于提供迭代停止条件的类型

需要相关知识说明，请参见本模块中的字符串 introduction'''

introduction = '''
在迭代时，总要设置一个迭代结束条件。常见的迭代结束条件有：

#1. 误差限 error_limit
 当迭代结果代入表达式后，误差小于误差限时，迭代停止。
 常用于可以直接算出误差的时候。
 比如 x^2 = 3 求 x，使用牛顿迭代法 x[n+1] = (3-x[n]**2)/2*x[n]，可以直接用 3-x[n]**2 求出误差。

#2. 相对误差限 relative_error_limit
 当迭代结果代入表达式后，相对误差小于相对误差限时，迭代停止。
 常用于可以直接算出相对误差的时候。

#3. 步长限 step_limit
 当迭代结果在某一次迭代的变化量小于步长限时，迭代停止。
 步长可以用于估计误差，常用于不可直接算出误差的时候。

#4. 相对步长限 relative_step_limit
 “相对步长限之于相对误差限”类似“步长限之于误差限”

#5. 最大迭代次数 max_iteration
 当迭代次数达到最大迭代次数时，结束迭代。
 如果迭代法不会收敛，每步的步进永远足够大，则迭代永远不会停止，最大迭代词素使得迭代总会停止。
 在这种情况时，停止迭代时应当提示迭代未收敛，或者直接报错。
 
 如果迭代法只需要极小的步数即可收敛，无需判断误差限，也可以用这类结束条件。在这种情况时，停止迭代无需报错。
'''

from abc import abstractmethod
from typing import Protocol
from numbers import Number
import numpy as np

class StopCondition(Protocol):
    '''表示迭代停止条件的抽象基类。'''

    @abstractmethod
    def __call__(self, iter_before, iter_after, iter_times) -> bool:
        '''输入本次迭代前后的数值，返回是/否应该停止迭代'''
        pass

def stopAt(e:Number=0, rel_e:Number=2**(-32), max_iter:Number=1000) -> StopCondition:
    '''通过输入一系列条件，直接得到迭代停止判断方法'''
    def stop(iter_before:Number, iter_after:Number, iter_times:Number):
        if iter_times == 0: return False
        if abs(iter_after-iter_before) < e: return True
        if abs(iter_after-iter_before) < rel_e*abs(iter_before): return True
        if iter_times >= max_iter: return True
        return False
    return stop

def astopAt(e:Number=0, rel_e:Number=2**(-32), max_iter:Number=1000) -> StopCondition:
    '''通过输入一系列条件，直接得到对 array 的迭代停止判断方法'''
    def stop(iter_before:np.ndarray, iter_after:np.ndarray, iter_times:Number):
        if iter_times == 0: return False
        if (np.abs(iter_after-iter_before) < e).all(): return True
        if (np.abs(iter_after-iter_before) < np.abs(rel_e*iter_before)).all(): return True
        if iter_times >= max_iter: return True
        return False
    return stop
