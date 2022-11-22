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
from gdsphuncs import couplers as cpl

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

def tester_ring(radius=80, ring_width=1.2, coupling_gap=1.1, waveguide_width=0.8, taper_length=100, ring_type='circle', coupler_type='circle', layer=1):
    """
    For making coupled rings. This functionality only creates the 
    waveguide components, NOT the text labels or the grating couplers.

    radius : float
        ring radius in um
    ring_width : float
        ring waveguide width in um. 
        Sets the coupling region waveguide width as well.
    coupling_gap : float
        coupling gap in um
    waveguide_width : float
        input/output waveguide width in um
    taper_length : float
        length of taper from waveguide_width to ring_width
    ring_type : string
        circle or euler. Sets ring type
    coupler_type : string
        circle or straight. Sets coupler type
    layer : int or layer object
        device layer

    """
    D = Device('tester ring')
    if ring_type == 'circle':
        MRR = D << pg.ring(radius = radius, width=ring_width, angle_resolution=0.1, layer=layer)

    if coupler_type == 'circle': 
        coupler = D << cpl.straight_coupler(radius = radius, 
                                    angle=40, 
                                    width_wide = ring_width, 
                                    width_wide_ring=ring_width, 
                                    width_narrow=ring_width, 
                                    layer=layer).rotate(90)

    MRR.x = coupler.ports[3].x + radius + ring_width/2 + coupling_gap
    MRR.y = coupler.ports[3].y

    # txt = D << pg.text(f'wgcp{coupling_gap} wg_w{waveguide_width}\nR{radius} ring_w{ring_width}', size = 10, justify = 'center', layer=txt_layer)
    # txt.x = MRR.x
    # txt.y = MRR.y - 1.5*radius

    pitches = (2 * radius) // 127 + 1

    T = cpl._taper(length=taper_length, width1=waveguide_width, width2=ring_width, trapz=False, layer=layer)
    t_upper = D << T
    t_lower = D << T

    t_lower.connect(2, coupler.ports[1])
    t_upper.connect(2, coupler.ports[2])

    X = CrossSection()
    X.add(width=waveguide_width, offset=0, ports=(1,2), layer=layer)

    P = Path()

    P.append( pp.euler(radius = 100, angle = -90, use_eff=True) )
    P.append( pp.straight(length = pitches*127 - 200) )
    P.append( pp.euler(radius = 100, angle = -90, use_eff=True) )
    P.append( pp.straight(length = 2*taper_length + np.abs(coupler.ports[1].y - coupler.ports[2].y)))

    ubend = D << P.extrude(width=X)
    ubend.connect(1, t_upper.ports[1])

    P_io = Path()
    P_io.append(pp.straight(length=10))
    io = P_io.extrude(width=X)
    straightL = D << io
    straightR = D << io

    straightL.connect(2, t_lower.ports[1])
    straightR.connect(2, ubend.ports[2])

    D.add_port(name=1, port=straightL.ports[1])
    D.add_port(name=2, port=straightR.ports[1])

    return D