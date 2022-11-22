import numpy as np
import gdspy
from numpy import sqrt, pi, cos, sin, log, exp, sinh
from phidl import Device, Layer, LayerSet, make_device, Path, CrossSection
from phidl import quickplot as qp # Rename "quickplot()" to the easier "qp()"
import phidl.geometry as pg
import phidl.routing as pr
import phidl.utilities as pu
import phidl.path as pp
import phidl.device_layout as pd
from matplotlib import pyplot as plt
import scipy as sp
import scipy.special

def taper_racetrack(radius = 10, length=20, width_wide = 3, width_narrow=1, euler_frac=0.5, points = 200, layer = 0):
    """Make a euler racetrack with width tapering on left side
    radius - true radius of the circular bend in the center of the 180 bends. NB the effective bend radius will be larger
    length - length of the straight section, um
    width_wide - width in the wide section, um
    width_narrow - width in the narrow section, um
    euler_frac - Angular fraction of the 180 deg bends to use Euler curve

    """
    D = Device(name= 'euler track')
    #Make left euler bend
    P = Path()
    P.append(pp.euler(radius=radius, angle=180, p=euler_frac))
    P.mirror()
    print('Euler racetrack: length of 180 bend: {:.4} um, bent_fraction={:.3}'.format(P.length(), P.length()/(length+P.length())))

    def width_func(t):
        #taper linearly from wide to narrow, but stay at narrow for the middle 10% of curve
        a=0.1 #consant narrow fraction
        def var(t):
            return (np.abs((t-0.5))-a/2)/(0.5-a/2)*(width_wide-width_narrow)+width_narrow

        def cons(t):
            return width_narrow

        return np.piecewise(t, [t<0.5-a/2, t>=0.5-a/2, t>0.5+a/2], [var, cons, var])

    X = CrossSection()
    X.add(width=width_func, offset=0, ports=(1,2), layer=layer)
    Left_euler = P.extrude(width=X)
    left_euler = D.add_ref(Left_euler)

    #Make straight sections
    Straight = pg.straight((width_wide,length), layer=layer)
    straight1 = D.add_ref(Straight)
    straight1.connect(1, left_euler.ports[1])
    straight2 = D.add_ref(Straight)
    straight2.connect(1,left_euler.ports[2])

    #Make right euler
    P.mirror()
    X2 = CrossSection()
    X2.add(width=width_wide, offset=0, ports=(1,2), layer=layer)
    Right_euler = P.extrude(width=X2)
    right_euler = D.add_ref(Right_euler)
    right_euler.connect(1, straight1.ports[2])


    return D