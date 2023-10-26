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


def ybranch(waveguide_pitch=150.0,
            taper_length=200.0,
            waveguide_width=1.2,
            waveguide_layer=2):
    D = Device()

    T = pg.taper(length=taper_length, width1=waveguide_width, width2=2*waveguide_width, layer=waveguide_layer)
    T.add_port(name='out1', midpoint=(T.xmax, T.center[1]+0.5*waveguide_width), orientation=0)
    T.add_port(name='out2', midpoint=(T.xmax, T.center[1]-0.5*waveguide_width), orientation=0)

    P1 = Path()

    X = CrossSection()
    X.add(width=waveguide_width, offset=0, layer=waveguide_layer, ports=(1, 2))

    r = 0.5 * (waveguide_pitch - waveguide_width) / (2 * (1 - np.sin(np.pi / 180 * 45)))
    # print('radius=' + str(r))
    S1 = pp.euler(radius=r, angle=45, use_eff=True)
    S2 = pp.euler(radius=r, angle=-45, use_eff=True)

    P1.append([S1, S2])

    # s, K = P1.curvature()
    #
    # plt.plot(s, K, '.-')
    # plt.xlabel('Position along curve (arc length)')
    # plt.ylabel('Curvature')

    W1 = P1.extrude(width=X)
    W2 = pg.copy(W1).mirror(p1=(0, 0), p2=(1, 0))

    D['Taper'] = D << T
    D['S1'] = D << W1
    D['S2'] = D << W2

    D['S1'].connect(port=1, destination=D['Taper'].ports['out1'])
    D['S2'].connect(port=1, destination=D['Taper'].ports['out2'])

    D.add_port(name='in', port=D['Taper'].ports[1])
    D.add_port(name='out1', port=D['S2'].ports[2])
    D.add_port(name='out2', port=D['S1'].ports[2])

    return D


def ybranch_sine(waveguide_pitch=350.0,
                 waveguide_width=1.2,
                 taper_length=200.0,
                 branch_length=500.0,
                 waveguide_layer=2,
                 plot_curvature=True
                ):

    D = Device('ybranch_sine')

    T = pg.taper(length=taper_length, width1=waveguide_width, width2=2 * waveguide_width, layer=waveguide_layer)
    T.add_port(name='out1', midpoint=(T.xmax, T.center[1] + 0.5 * waveguide_width), orientation=0)
    T.add_port(name='out2', midpoint=(T.xmax, T.center[1] - 0.5 * waveguide_width), orientation=0)

    P1 = Path()
    S = pp.straight(length=branch_length)

    def sine_s_bend(t):
        A = 0.5 * 0.5 * (waveguide_pitch - waveguide_width)
        w = -A * (1+np.cos(np.pi*t + np.pi)) - 0.5*waveguide_width
        return w

    P1.append(S)
    P1.offset(offset=sine_s_bend)

    # if plot_curvature:
    #     s, K = P1.curvature()

    #     plt.plot(s, K, '.-')
    #     plt.xlabel('Position along curve (arc length)')
    #     plt.ylabel('Curvature')

    X = CrossSection()
    X.add(width=waveguide_width, offset=0, layer=waveguide_layer, ports=(1,2))

    W1 = P1.extrude(width=X)
    W2 = pg.copy(W1).mirror(p1=(0, 0), p2=(1, 0))

    D['Taper'] = D << T
    D['S1'] = D << W1
    D['S2'] = D << W2

    D['S1'].connect(port=1, destination=D['Taper'].ports['out1'])
    D['S2'].connect(port=1, destination=D['Taper'].ports['out2'])

    D.add_port(name='in', port=D['Taper'].ports[1])
    D.add_port(name='out1', port=D['S2'].ports[2])
    D.add_port(name='out2', port=D['S1'].ports[2])

    return D

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


def caliper(layers, 
            caliper_w = 2,
            caliper_h = 20,
            caliper_spacing_0 = 7,
            caliper_spacing_d = 0.1):
    
    C =  Device("Caliper")
    
    for ii, layer in enumerate(layers):
        print(layer)
        spacing = caliper_spacing_0 - ii*caliper_spacing_d

        Caliper = pg.litho_ruler(
            height = caliper_h ,
            width = caliper_w,
            spacing = spacing,
            scale = [1,1,1,1,1,1,1,1,1,1],
            num_marks = 21,
            layer = layer,
            )
        caliper = C<<Caliper
        caliper.movey(caliper_h*ii)
        caliper.x = 0
        
    return C.flatten()


def alignment_mark_ebeam(
    cross_w = 5, #width of cross
    cross_h = 800, # height of cross
    caliper_layers = None, # list of layers in order to which write calipers
    caliper_bool = True,
    mark_layer = 0, # alignment mar layer
    align_box_layer=15,
    label = "",
    txt_size = 50): #box for opening pmma box
    
    if caliper_layers is None and caliper_bool:
        caliper_layers=[mark_layer]

    Mark = Device("Mark")

    X = pg.cross(length = cross_h, width = cross_w, layer = mark_layer)

    temp = pg.rectangle(size = (2*cross_w, 2*cross_w))
    temp.center = X.center

    X = pg.boolean(A=X, B=temp, operation="not", precision = 1e-6, num_divisions = [1,1], layer = mark_layer)
    x = Mark<<X


    Sq = pg.rectangle(size = (cross_w/2, cross_w/2), layer=mark_layer)
    sq1 = Mark<<Sq
    sq1.move(origin=(sq1.xmin, sq1.ymin), destination = x.center)
    sq2 = Mark<<Sq
    sq2.move(origin=(sq2.xmax, sq2.ymax), destination = x.center)
    
    if caliper_layers is not None:
        Caliper = caliper(caliper_layers)
        c1 = Mark<<Caliper
        c1.x = x.x
        c1.ymin = x.ymax
        c2 = Mark<<Caliper
        c2.rotate(-90)
        c2.xmin, c2.y  = x.xmax, x.y
    
    if align_box_layer is not None:
        Box = pg.rectangle(size = (cross_h*0.9, cross_h*0.9), layer = align_box_layer)
        box = Mark<<Box
        box.center = x.center
    
    txt = Mark<<pg.text(text=label, size=txt_size, layer=mark_layer)
    txt.center = [box.xmax/2, box.ymax/2]

    Mark.add_port(name = "c", midpoint = x.center)

    return Mark.flatten()


def gsg_contacts(gsg_w=70, gsg_pitch=150, layer = 14):
    GSG = Device()
    Pad = pg.straight(size = (gsg_w,gsg_w), layer = layer).rotate(90)


    GSG = Device()
    g1 = GSG<<Pad
    s = GSG<<Pad
    s.movey(gsg_pitch)
    g2 = GSG<<Pad
    g2.movey(2*gsg_pitch)

    GSG.add_port(name = "g1", port = g1.ports[2])
    GSG.add_port(name = "g2", port = g2.ports[2])
    GSG.add_port(name = "s", port = s.ports[2])

    return GSG.flatten()
    
def ruler_ports(ruler_spacing = 50,
                ruler_w = 2,
                ruler_h = 50,
                port_spacing = 142,
                N_ports = 2,
                wg_layer = 1,
                port_w = 1.5,
                ruler_max_w = 100
                ):

    num_marks = min(21, ruler_max_w//ruler_spacing)
    
    Ruler = Device()
    Ruler0 = pg.litho_ruler(height = ruler_h ,
        width = ruler_w,
        spacing = ruler_spacing,
        scale = [3,1,1,1,1,2,1,1,1,1],
        num_marks = num_marks,
        layer = wg_layer,

        )
    ports = []
    for ii in range(N_ports):
        ports.append(
            Ruler.add_port(name=ii+1, midpoint=(ii*port_spacing, 0), width=port_w, orientation=90)
        )

    ruler1 = Ruler<<Ruler0
    ruler1.mirror(p1=(ruler_w/2, 1), p2=(ruler_w/2, 0))
    ruler1.move(origin = [ruler_w/2, 0], destination=[ports[0].x-ruler_spacing, 0])
    ruler2 = Ruler<<Ruler0
    ruler2.move(origin = [ruler_w/2, 0], destination=[ports[-1].x+ruler_spacing, 0])

    return Ruler.flatten()