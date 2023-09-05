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
from gds_phuncs import gds
from gds_phuncs import mzis

def psqp_wg(wg_w, wg_L, si_w0, si_a, num_periods, wg_layer = 0, si_layer=1, angle=0, bend_radius=140, num_pts=2e4):
    D = Device("wg")

    def si_w_func(t, num_periods = num_periods):
        # Note: Custom width/offset functions MUST be vectorizable--you must be able
        # to call them with an array input like my_custom_width_fun([0, 0.1, 0.2, 0.3, 0.4])
        w =  si_w0 + si_a*np.cos(2*np.pi*t * num_periods-np.pi)
        return w

    # Create the Path
    P = pp.straight(length = wg_L, num_pts = int(num_pts))
    P.xmin = 0

    # Create two cross-sections: one fixed width, one modulated by my_custom_offset_fun
    X = CrossSection()
    X.add(width = wg_w, offset = 0, layer = wg_layer)

    if si_w0!=0 or si_a!=0:
        X.add(width = si_w_func, offset = 0,  layer = si_layer)
    
    # Extrude the Path to create the Device
    D = P.extrude(X)
    D.add_port(name=1, midpoint=(D.xmin, D.y), width=wg_w, orientation=180)
    D.add_port(name=2, midpoint=(D.xmax, D.y), width=wg_w, orientation=0) #will be replaced by angle routing

    D.rotate(angle)

    D = angle_routing(D, wg_w, angle, bend_radius=bend_radius, wg_layer=wg_layer)
    # D.add_port(name=1, midpoint=(D.xmin, D.y), width=wg_w, orientation=180)
    # D.add_port(name=2, midpoint=(D.xmax, D.y), width=wg_w, orientation=0)
    
    return D.flatten()

def angle_routing(D, #angled waveguide
                  wg_w, #waveguide width
                  angle, #deviation angle from horizontal (0<=angle<45) - degrees
                  bend_radius = 140,
                  wg_layer=0): #bending radius for routing
    """gets an angled waveguide and routes it to a horizontal output.
    """
    if angle<0 or angle>45:
        raise ValueError("angle must be between 0 and 45 degrees.")
    
    #port 1
    x0, y0 = D.ports[1].center
    points = np.array([
            (x0,y0),
            (x0-bend_radius, y0-bend_radius*np.tan(angle*np.pi/180)),
            (x0-2*bend_radius, y0-bend_radius*np.tan(angle*np.pi/180))
            ])
    P = pp.smooth(
        points = points,
        radius = bend_radius,
        corner_fun = pp.euler, # Alternatively, use pp.arc
        use_eff = False,
        )

    X = CrossSection()
    X.add(width = wg_w, offset = 0, layer = wg_layer)
    D<<P.extrude(X)
    #update ports
    D.remove(D.ports[1])
    D.add_port(name=1, midpoint = (D.xmin, D.ymin+wg_w/2), width=wg_w, orientation = 180)

    # port 2
    x0, y0 = D.ports[2].center
    points = np.array([
            (x0,y0),
            (x0+bend_radius, y0+bend_radius*np.tan(angle*np.pi/180)),
            (x0+2*bend_radius, y0+bend_radius*np.tan(angle*np.pi/180))
            ])
    P = pp.smooth(
        points = points,
        radius = bend_radius,
        corner_fun = pp.euler, # Alternatively, use pp.arc
        use_eff = False,
        )

    X = CrossSection()
    X.add(width = wg_w, offset = 0, layer = wg_layer)
    D<<P.extrude(X)
    #update ports
    D.remove(D.ports[2])
    D.add_port(name=2, midpoint = (D.xmax, D.ymax-wg_w/2), width=wg_w, orientation = 0)
    return D