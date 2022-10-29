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

def route_S(port1, port2, width=0.8, rmin=80, layer=1):
    """
    Utility for sigmoid routing between two ports,
    using two Euler bends. Uniquely defined for given
    dx and dy. 

    NOTE: In its current form, this utility only can
    route upwards, with y2 > y1. 

    port1 : phidl Port object
        port 1. Route starts here.
    port2 : phidl Port object
        port 2. Route ends here. 
    rmin : float
        Minimum radius of curvature. For error checking.
    layer : float or layer
        GDS layer   
    """
    delta_orientation = np.round(
        np.abs(np.mod(port1.orientation - port2.orientation, 360)), 3
    )
    if delta_orientation not in (0, 180, 360):
        raise ValueError("[PHIDL] path_U(): ports must be parallel.")

    R = Device("S Bend")
    
    x1, y1 = port1.midpoint
    x2, y2 = port2.midpoint

    dx = np.abs(x2-x1)
    dy = np.abs(y2-y1)

    x = dx/2
    # y = dy/2
    yscale = dy/dx
    
    # c = np.sqrt(x**2 + y**2)
    # theta = np.degrees(2*np.arcsin(c/(2*radius)))
    scale = 1 + yscale**2
    radius = x * scale / 2
    print(f'equivalent radius of curvature: {radius}')

    if radius < rmin:
        raise ValueError("[gdsphuncs] route_S(): Radius less than minimum setpoint!")    

    theta = np.degrees(2*np.arcsin(1/np.sqrt(scale)))
    print(f'equivalent circle arc angle: {theta}')
    
    X = CrossSection()
    X.add(width=width, offset=0, ports=(1,2), layer=layer)

    P = pp.euler(radius=radius, angle=theta, use_eff=True)
    s,K = P.curvature()
    print(f'minimum radius of curvature: {1/np.max(K)}')
    Bend = P.extrude(width=X)

    if (y2 > y1 and x2 > x1):
        Bend = Bend.mirror()

    sbend = R << Bend
    # sbend.mirror()
    sbend.rotate(-90)
    sbend.connect(1, port1)
    # print(sbend.ports)
    sbend2 = R << Bend
    # sbend2.mirror()
    # sbend2.connect(2, destination=sbend.ports[2])
    sbend2.connect(1, port2)

    return R

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


def alignment_marks(Xs, Ys, width=0.5, label=False, fontsize=10, xlabel='z', ylabel='y', layer=0):
    #places alignment markers in 4 quadrants using the Quadrant I coordinates
    #specified by Xs and Ys
    D = Device('marks')
    Mark = pg.cross(length = 200, width = width, layer=layer)
    for x, y in zip(Xs, Ys):
        for x_side in [-1,1]:
            for y_side in [-1,1]:
                mark = D.add_ref(Mark)
                mark.x = x*x_side
                mark.y = y*y_side

    if label:
        xtxt = pg.text(xlabel, layer=layer)
        ytxt = pg.text(ylabel, layer=layer)
        for x, y in zip(Xs, Ys):
            for x_side in [-1,1]:
                for y_side in [-1,1]:
                    xax = D << xtxt
                    xax.x = x*x_side + 110
                    xax.y = y*y_side

                    yax = D << ytxt
                    yax.x = x*x_side
                    yax.y = y*y_side - 110

        # xcenter = Mark.x
        # ycenter = Mark.y

        # xtxt = Mark << pg.text(xlabel, layer=layer)
        # xtxt.x = xcenter
        # xtxt.y = ycenter - 110

        # ytxt = Mark << pg.text(ylabel, layer=layer)
        # ytxt.x = xcenter + 110
        # ytxt.y = ycenter 

    return D

def merge_shapes(dev, layer=4):
    """Merge all shapes in a device"""
    empty = Device()
    rect= empty.add_ref(pg.rectangle(size=dev.size, layer=layer))
    rect.x = dev.x; rect.y = dev.y
    newdev = pg.boolean(dev, empty, operation='and', layer=layer)
    return newdev


def inverse_dev(dev, layer_invert=4, layer_new = 15, border=0):
    #returns a dev in which layer_invert is inverted with border border.
    newdev = pg.extract(dev, layers = [layer_invert])
    newdev = merge_shapes(newdev, layer=layer_invert)
    newdev = pg.invert(newdev, border=border, layer=layer_new)
    return newdev
