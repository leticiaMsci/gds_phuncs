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


# def route_S(port1, port2, width=0.8, rmin=80, layer=1):
#     """
#     Utility for sigmoid routing between two ports,
#     using two Euler bends. Uniquely defined for given
#     dx and dy. 

#     NOTE: In its current form, this utility only can
#     route upwards, with y2 > y1. 

#     port1 : phidl Port object
#         port 1. Route starts here.
#     port2 : phidl Port object
#         port 2. Route ends here. 
#     rmin : float
#         Minimum radius of curvature. For error checking.
#     layer : float or layer
#         GDS layer   
#     """
#     delta_orientation = np.round(
#         np.abs(np.mod(port1.orientation - port2.orientation, 360)), 3
#     )
#     if delta_orientation not in (0, 180, 360):
#         raise ValueError("[PHIDL] path_U(): ports must be parallel.")

#     R = Device("S Bend")
    
#     x1, y1 = port1.midpoint
#     x2, y2 = port2.midpoint

#     dx = np.abs(x2-x1)
#     dy = np.abs(y2-y1)

#     x = dx/2
#     # y = dy/2
#     yscale = dy/dx
    
#     # c = np.sqrt(x**2 + y**2)
#     # theta = np.degrees(2*np.arcsin(c/(2*radius)))
#     scale = 1 + yscale**2
#     radius = x * scale / 2
#     print(f'equivalent radius of curvature: {radius}')

#     if radius < rmin:
#         raise ValueError("[gdsphuncs] route_S(): Radius less than minimum setpoint!")    

#     theta = np.degrees(2*np.arcsin(1/np.sqrt(scale)))
#     print(f'equivalent circle arc angle: {theta}')
    
#     X = CrossSection()
#     X.add(width=width, offset=0, ports=(1,2), layer=layer)

#     P = pp.euler(radius=radius, angle=theta, use_eff=True)
#     s,K = P.curvature()
#     print(f'minimum radius of curvature: {1/np.max(K)}')
#     Bend = P.extrude(width=X)

#     if (y2 > y1 and x2 > x1):
#         Bend = Bend.mirror()

#     sbend = R << Bend
#     # sbend.mirror()
#     sbend.rotate(-90)
#     sbend.connect(1, port1)
#     # print(sbend.ports)
#     sbend2 = R << Bend
#     # sbend2.mirror()
#     # sbend2.connect(2, destination=sbend.ports[2])
#     sbend2.connect(1, port2)

#     return R

def route_S(port1, port2, rmin=80, layer=1):
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
    orientations = [np.round(port1.orientation, 12), 
                    np.round(port2.orientation, 12)] # hacky way to add numerical tolerance

    if delta_orientation not in (0, 180, 360):
        raise ValueError("[gdsphuncs] route_S(): Ports must be parallel.")
    if np.any(np.mod(orientations, 90)):
        raise ValueError("[gdsphuncs] route_S(): port orientations must be along xy axes.")

    R = Device("S Bend")
    
    x1, y1 = port1.midpoint
    x2, y2 = port2.midpoint
    width1 = [port1.width, np.mean([port1.width, port2.width])]
    width2 = [port2.width, np.mean([port1.width, port2.width])]

    dx = np.abs(x2-x1)
    dy = np.abs(y2-y1)

    x = dx/2
    y = dy/2

    y_oriented = np.any(np.mod(orientations, 180))
    if y_oriented: 
        if x > y:
            raise ValueError("[gdsphuncs] route_S(): y oriented ports must have dy > dx.")
    else:
        if x < y:
            raise ValueError("[gdsphuncs] route_S(): x oriented ports must have dy < dx.")

    dmin = np.min((x, y))
    dmax = np.max((x, y))

    # radius = (x**2 + y**2)/(2*y)
    # theta = np.arcsin((x/radius))

    radius = (x**2 + y**2)/(2*dmin)
    theta = np.arcsin((dmax/radius))
    
    # c = np.sqrt(x**2 + y**2)
    # theta = np.degrees(2*np.arcsin(c/(2*radius)))
    print(f'equivalent radius of curvature: {radius}')

    if np.abs(radius) < rmin:
        raise ValueError("[gdsphuncs] route_S(): Radius less than minimum setpoint!")    
    print(f'equivalent circle arc angle: {np.degrees(theta)}')
    
    #X = CrossSection()
    #X.add(width=width1, offset=0, ports=(1,2), layer=layer)

    P = pp.euler(radius=radius, angle=np.degrees(theta), use_eff= True)
    try:
        s,K = P.curvature()
        print(f'minimum radius of curvature: {1/np.max(K)}')
    except Exception:
        print('error with curvature')
    Bend = P.extrude(width=width1, layer=layer)
    Bend.add_port(name=1, midpoint=[0, 0], orientation=180)
    # if (y2 < y1 and x2 > x1):
    #     Bend = Bend.mirror()

    Bend2 = P.extrude(width=width2, layer=layer)
    Bend2.add_port(name=1, midpoint=[0, 0], orientation=180)
    
    if (y_oriented and x2 > x1) or y2 < y1:
        Bend = Bend.mirror()
        Bend2 = Bend2.mirror()     

    sbend = R << Bend
    sbend.connect(1, port1)

    sbend2 = R << Bend2
    sbend2.connect(1, port2)

    R.add_port(name=1, port=port1)
    R.add_port(name=2, port=port2)

    return R

def connector_S(dx, dy, width=0.8, layer=1):
    R = Device()

    x = dx/2
    y = dy/2

    dmin = np.min((x, y))
    dmax = np.max((x, y))

    radius = (x**2 + y**2)/(2*dmin)
    theta = np.arcsin((dmax/radius))

    P = pp.euler(radius=radius, angle=np.degrees(theta), use_eff= True)
    X = CrossSection()
    X.add(width=width, offset=0, ports=(1,2), layer=layer)
    
    Bend = P.extrude(width=X)  

    bend1 = R << Bend
    bend2 = R << Bend
    bend2.connect(2, bend1.ports[2])

    R.add_port(name=1, port=bend1.ports[1])
    R.add_port(name=2, port=bend2.ports[1])

    return R

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
