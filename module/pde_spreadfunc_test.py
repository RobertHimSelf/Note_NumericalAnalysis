#!/usr/bin/python
# -*- coding: utf-8 -*-
from numbers import Number
from typing import Callable
from collections.abc import Sequence
import numpy as np
import matplotlib.pyplot as plt
import math

'''u_t - v*u_{xx} = 0'''

try:
    from .ode__typing import X,U,T
    from .pde_spreadfunc import *
except:
    from ode__typing import X,U,T
    from pde_spreadfunc import *

def FTCS_Spread_Dirichlet_test():
    v = 1/6
    dt = 0.02
    x,t,u = FTCS_Spread(u0=lambda x: math.sin(math.tau*x),
                        border_min=lambda t:0,
                        border_max=lambda t:0,
                        x_min=0, x_max=1, dx=0.1, t_max=50, dt=dt, v=v)
    w = np.array(
        [[math.exp(-v*ti*math.tau**2)*math.sin(math.tau*xi) for ti in t]for xi in x]
    )
    t,x = np.meshgrid(t,x)
    fig, ax = plt.subplots(subplot_kw=dict(projection="3d"))
    surf1 = ax.plot_surface(x[:,:round(1/dt)], t[:,:round(1/dt)], u[:,:round(1/dt)], color=(0.1,0.2,0.5,1))
    surf2 = ax.plot_surface(x[:,:round(1/dt)], t[:,:round(1/dt)], w[:,:round(1/dt)], color=(0.5,0.1,0.2,0.5))
    ax.set(xlabel="x",ylabel="t",zlabel="u")
    plt.show()

def FTCS_Spread_Mixed_test():
    v = 1
    dt = 0.0003
    m = 40
    x,t,u = FTCS_Spread(u0=lambda x: math.cos(math.pi*x/2),
                        border_min=lambda t:0,
                        border_max=lambda t:0,
                        x_min=0, x_max=1, dx=1/m, t_max=50, dt=dt, v=v,
                        border_min_type=NEWMANN)
    w = np.array(
        [[math.exp(-v*ti*math.pi**2/4)*math.cos(math.pi*xi/2) for ti in t]for xi in x]
    )
    t,x = np.meshgrid(t,x)
    fig, ax = plt.subplots(subplot_kw=dict(projection="3d"))
    surf1 = ax.plot_surface(x[:,:round(1/dt)], t[:,:round(1/dt)], u[:,:round(1/dt)], color=(0.1,0.2,0.5,1))
    surf2 = ax.plot_surface(x[:,:round(1/dt)], t[:,:round(1/dt)], w[:,:round(1/dt)], color=(0.5,0.1,0.2,0.5))
    ax.set(xlabel="x",ylabel="t",zlabel="u")
    plt.show()

def BTCS_Spread_Dirichlet_test():
    v = 1/6
    dt = 0.1
    tm = 1
    x,t,u = BTCS_Spread(u0=lambda x: math.sin(math.tau*x),
                        border_min=lambda t:0,
                        border_max=lambda t:0,
                        x_min=0, x_max=1, dx=0.1, t_max=50, dt=dt, v=v)
    w = np.array(
        [[math.exp(-v*ti*math.tau**2)*math.sin(math.tau*xi) for ti in t]for xi in x]
    )
    t,x = np.meshgrid(t,x)
    fig, ax = plt.subplots(subplot_kw=dict(projection="3d"))
    surf1 = ax.plot_surface(x[:,:round(tm/dt)], t[:,:round(tm/dt)], u[:,:round(tm/dt)], color=(0.1,0.2,0.5,1))
    surf2 = ax.plot_surface(x[:,:round(tm/dt)], t[:,:round(tm/dt)], w[:,:round(tm/dt)], color=(0.5,0.1,0.2,0.8))
    ax.set(xlabel="x",ylabel="t",zlabel="u")
    plt.show()

if __name__ == "__main__":
    BTCS_Spread_Dirichlet_test()
    
    