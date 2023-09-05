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
from gds_phuncs import couplers as cpl
from gds_phuncs import gds
from gds_phuncs.couplers import grating

def Qtester_wrap_around(
            radius_coupler=80,
            coupler_angle=30,
            radius_bend=150,
            width_coupler=1,
            width_io=0.8,
            pitch_io=127*3,
            length_taper=50,
            wg_layer=0,
            ring_layer=0,
            txt_layer = 0,
            radius_ring = 150,
            waveguide_width=1,
            gap = 1):
    """Draws a ring resonator with a bus waveguide wrapped around it.
    radius_coupler - radius of the coupling section
    coupler_angle - angle spanned by the circular coupling section.
    radius_bend - bend radius for other turns
    width_coupler - width of the waveguide in coupling section
    width_io - width of the waveguide input and output
    length_taper - length of width taper
    """
    D = Device(name= 'circle coupler')
    #Make coupler section
    P1 = pp.euler(radius=radius_coupler,angle=2*coupler_angle,p=0.5)
    X = CrossSection()
    X.add(width=width_coupler, offset=0, ports=(1,2),name='main', layer=wg_layer)
    Coupler = P1.extrude(width=X)
    Coupler.rotate(90-coupler_angle)
    coupler = D.add_ref(Coupler, alias='coupler')

    #make coupler bottom bend
    P2 = pp.euler(radius=radius_bend, angle=-coupler_angle, p=1)
    bend_bottom = D.add_ref(P2.extrude(width=X))
    bend_bottom.connect(2, coupler.ports[1])
    # #make coupler bend top
    P3 = pp.euler(radius=radius_bend, angle=-coupler_angle-90, p=0.5)
    bend_top_left = D.add_ref(P3.extrude(width=X))
    bend_top_left.connect(1, coupler.ports[2])

    P4 = pp.euler(radius=radius_bend, angle=-90, p=0.5)
    bend_top_right = D.add_ref(P4.extrude(width=X))
    bend_top_right.connect(1, bend_top_left.ports[2])
    coupler_x = radius_coupler*(1-np.cos(coupler_angle))
    bend_top_right_x = bend_top_left.xmax+max(2*radius_ring, pitch_io)-radius_bend-coupler_x
    bend_top_right.move(origin=(bend_top_right.xmax, bend_top_right.y), 
                destination=(bend_top_right_x, bend_top_right.y))


    D.add_ref(
        pr.route_basic(port1=bend_top_right.ports[1], port2=bend_top_left.ports[2], 
        path_type='sine', width_type='straight', layer=wg_layer)
        )


    #Make the taper sections
    P = pp.straight(length=length_taper)
    X_io = CrossSection()
    X_io.add(width=width_io, offset=0, ports=(1,2), name='main', layer=wg_layer)
    X_taper = pp.transition(cross_section1=X, cross_section2=X_io, width_type='linear')
    Taper = P.extrude(X_taper)
    Taper.rotate(-90)
    taper_right = D.add_ref(Taper)
    taper_right.connect(1, bend_bottom.ports[1])
    taper_left = D.add_ref(Taper)
    taper_left.move(origin=(taper_left.x, taper_left.y), destination=(taper_right.x+pitch_io, taper_right.y))


    # ring
    ring = pg.ring(radius_ring, width = waveguide_width, layer=ring_layer)
    ring_resonator = D.add_ref(ring)
    ring_resonator.move(origin=(ring_resonator.x, ring_resonator.y), 
                        destination=(ring_resonator.x+Coupler.xmax+gap+radius_ring+0.5*(waveguide_width), ring_resonator.y+Coupler.y))

    # #route to taper
    D.add_ref(pr.route_basic(port1=bend_top_right.ports[2], port2=taper_left.ports[1], path_type='sine', width_type='straight', layer=wg_layer))
    D.add_port(name=1, midpoint = [taper_right.ports[2].x, taper_right.ports[2].y], width = width_io, orientation = 270)
    D.add_port(name=2, midpoint = [taper_left.ports[2].x, taper_left.ports[2].y], width = width_io, orientation = 270)

    # center device at ring resonator center
    D.move(origin=ring_resonator.center, destination=(0,0))
    return D


def Qtester_chip(gap_list, w_list, dx, dy, GC, param_Qtester, label_header = "", marks=True):
    T = Device('chip')
    for jj, w in enumerate(w_list):
        param_Qtester["waveguide_width"] = w
        param_Qtester["width_coupler"] = w
        for ii, gap in enumerate(gap_list):
            S = Device('mydevice')
            param_Qtester["gap"] = gap
            Qtest = Qtester_wrap_around(**param_Qtester)
            c = S.add_ref(Qtest, alias = "Q")
            gc1 = S.add_ref(GC, alias='gc1')
            gc1.connect(port=1, destination=c.ports[1])
            gc2 = S.add_ref(GC, alias='gc2')
            gc2.connect(port=1, destination=c.ports[2])

            #add text
            label=label_header+"\nG{:.1f}W{:.1f}".format(gap,w)
            Txtlabel = pg.text(label, size=15, justify='center', layer=param_Qtester["txt_layer"])
            txt = S.add_ref(Txtlabel)
            txt.move(origin = txt.center, destination = (0.5*gc1.x+0.5*gc2.x, c.ymin))
            s = T.add_ref(S)
            S.move(origin=(S.x, S.y), destination=(ii*dx, jj*dy))
    
    T.move(origin=T.center, destination = [0,0])
    xmax, ymax = (T.xmax+200)//100 * 100, (T.ymax+200)//100 * 100

    if marks:
        Alignment = alignment_marks([xmax], [ymax], align_layer = param_Qtester["txt_layer"], add_metal=True, metal_layer = 999)
        T<<Alignment
    
    return T


def alignment_marks(Xs, Ys, align_layer, metal_layer=0, add_metal=True):
    #places alignment markers in 4 quadrants using the Quadrant I coordinates
    #specified by Xs and Ys
    D = Device('marks')
    Mark = Device('mark')
    Cross = pg.cross(length = 150, width = 0.5, layer=align_layer)
    cross = Mark.add_ref(Cross)
    if add_metal:
        Metal_aligner = pg.rectangle(size=(98, 98), layer=metal_layer)
        ma = Mark.add_ref(Metal_aligner)
        ma.x = cross.x
        ma.y = cross.y
        # ma2 = Mark.add_ref(Metal_aligner)
        # ma2.xmin = cross.xmin
        # ma2.ymax = cross.ymax
        # ma3 = Mark.add_ref(Metal_aligner)
        # ma3.xmax = cross.xmax
        # ma3.ymin = cross.ymin
        # ma4 = Mark.add_ref(Metal_aligner)
        # ma4.xmin= cross.xmin
        # ma4.ymin = cross.ymin

    ii = 0
    for x, y in zip(Xs, Ys):
        for x_side in [-1,1]:
            for y_side in [-1,1]:
                mark = D.add_ref(Mark)
                mark.x = x*x_side
                mark.y = y*y_side
                Txtlabel = pg.text(str(ii), size=15, justify='left', layer=align_layer)
                txt = D.add_ref(Txtlabel)
                txt.x = x*x_side + 15
                txt.y = y*y_side +15
                ii=ii+1
    return D