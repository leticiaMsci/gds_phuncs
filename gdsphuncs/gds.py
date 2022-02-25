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

def grating(period,
            number_of_teeth,
            fill_frac,
            width,
            lda=1,
            sin_theta=0,
            focus_distance=-1,
            focus_width=-1,
            evaluations=50,
            overlap=0,
            layer=0,
            datatype=0):
    '''
    Straight or focusing grating.

    period          : grating period
    number_of_teeth : number of teeth in the grating
    fill_frac       : filling fraction of the teeth with respect to the period
    width           : width of the grating
    lda             : free-space wavelength
    sin_theta       : sine of incidence angle
    focus_distance  : focus distance (negative for straight grating)
    focus_width     : if non-negative, the focusing area is included in the
                      result (usually for negative resists) and this is the
                      width of the waveguide connecting to the grating
    overlap         : Port position relative to
    evaluations     : number of parametric evaluations of `path.parametric`
    layer           : layer object

    Return Phidl device
    '''
    position = (0,0)
    if focus_distance < 0:
        print('hi!')
        path = gdspy.L1Path(
            (position[0] - 0.5 * width, position[1] + 0.5 *
             (number_of_teeth - 1 + fill_frac) * period),
            '+x',
            period * fill_frac, [width], [],
            number_of_teeth,
            period,
            layer=layer,
            datatype=datatype)
    else:
        neff = lda / float(period) + sin_theta
        qmin = int(focus_distance / float(period) + 0.5)
        path = gdspy.Path(period * fill_frac, (position))
        max_points = 199 if focus_width < 0 else 2 * evaluations
        c3 = neff**2 - sin_theta**2
        w = 0.5 * width
        for q in range(qmin, qmin + number_of_teeth):
            c1 = q * lda * sin_theta
            c2 = (q * lda)**2
            path.parametric(
                lambda t: (width * t - w, (c1 + neff * np.sqrt(
                    c2 - c3 * (width * t - w)**2)) / c3),
                number_of_evaluations=evaluations,
                max_points=max_points,
                layer=layer,
                datatype=datatype)
            path.x = 0
            path.y = 0
        if focus_width >= 0:
            path.polygons[0] = np.vstack(
                (path.polygons[0][:evaluations, :],
                 ([position] if focus_width == 0 else
                  [(position[0] + 0.5 * focus_width, position[1]),
                   (position[0] - 0.5 * focus_width, position[1])])))
            path.fracture()

        D = Device('grating_coupler')
        D.add_polygon(path, layer=layer)
        D.add_port(name=1, midpoint = [0,overlap], width = focus_width, orientation = 270)
        D.rotate(180)
        return D

def mmi(length=74, width=9, spacing=3.2, taper_width=1.4, waveguide_width=0.8, taper_length=10, layer=2):
    """
    Function for making MMI, including waveguide tapers

    Parameters
    ---------- 
    length : float
        MMI length, in um
    width : float
        MMI top width, in um
    spacing : float
        input/output waveguide separation, in um
    taper_width : float
        taper top width, in um
    waveguide_width : float
        input/output waveguide top width, in um
    taper_length : float
        length of taper region, in um
    """
    D = Device() # main device
    T = Device() # tapers

    points =  [(-width/2, 0), (width/2, 0), (width/2, length), (-width/2, length)]
    D.add_polygon(points, layer=layer)

    path = gdspy.Path(waveguide_width, (0, -taper_length))
    path.segment(taper_length, direction='+y', final_width=taper_width, layer=layer)
    T.add_polygon(path, layer=layer)
    
    taper_ref1 = D << T
    taper_ref2 = D << T
    taper_ref3 = D << T
    taper_ref4 = D << T

    taper_ref1.move([-spacing/2, 0])
    taper_ref2.move([spacing/2, 0])
    taper_ref3.rotate(180).move([-spacing/2, length])
    taper_ref4.rotate(180).move([spacing/2, length])

    D.add_port(name=1, midpoint=[-spacing/2, -taper_length], width=waveguide_width, orientation=270)
    D.add_port(name=2, midpoint=[spacing/2, -taper_length], width=waveguide_width, orientation=270)
    D.add_port(name=3, midpoint=[-spacing/2, length+taper_length], width=waveguide_width, orientation=90)
    D.add_port(name=4, midpoint=[spacing/2, length+taper_length], width=waveguide_width, orientation=90)

    D.move([0, -length/2])

    return D

def add_grating_general(
        WG_parameter,
        period = 0.805,
        fill_frac = 0.3106,
        width_grating = 12,
        focus_distance = 10,# None for 1D grating
        taper_length=10,    # Not used for Focusing grating
        straight_length=5,  # Not used for Focusing grating
        number_of_teeth = None,
        type = 'out',
        window_extra_width = 10,
        window_extra_height = 10,
        lda = 1.55,
        sin_theta = np.sin(np.pi * -8 / 180),
        evaluations=99,
        layer_grating=4,
        layer_backreflector=25,
):
    '''
    Generalized function for making any kind of grating: Focused/1D and Uniform/Apodized
    Parameters
    ----------
    WG_parameter : array-like[3]
        [Start Coordinates in (x,y), Waveguide Width, Direction]
    period : number or array-like[N]
        grating period.  Assumed uniform, unless array is specified
    fill_frac : number or array-like[N]
        grating fill fraction.  Assumed uniform, unless array is specified
    width_grating : positive number
        width of the grating.
    focus_distance : non-negative number
        focus distance for a Focused grating.  If None, 1D grating is made instead
    taper_length : non-negative number
        length of tapering region from waveguide to grating.  Value only needed if focus_distance == None
    straight_length : non-negative number
        length of straight section before grating.  Value only needed if focus_distance == None
    number_of_teeth : positive integer or None
        number of teeth within the grating.  If None, this is determined based on period, fill_frac, and/or width_grating
    type : 'in' or 'out'
        Specifies the direction of the grating, depending on if it is an input or output
    window_extra_width : positive number
        window for backreflector, extra distance from left/right of grating.
    window_extra_height : positive number
        height for backreflector, extra distance from top/bottom of grating.
    lda : positive number
        free-space wavelength.
    sin_theta : number
        sine of incidence angle.
    evaluations : positive integer
        number of parametric evaluations of `path.parametric`.
    layer_grating : positive integer
        GDSII layer number.
    Returns
    -------
    out : array-like[3]
        WG_parameter of where the waveguide connects to the grating: [Start Coordinates in (x,y), Waveguide Width, Direction]
    '''
    # Function to check if input is some kind of list or array
    def is_listable(x):
        return isinstance(x, (list,np.ndarray))
    # Setup some variables
    position = WG_parameter[0]
    focus_width = WG_parameter[1]
    direction = WG_parameter[2]
    width = width_grating
    w = 0.5 * width
    origin = (0,0)
    sections = []
    # Parse number of grating periods, or calc from grating width to be close to square
    if is_listable(period):
        number_of_teeth = len(period)
    elif is_listable(fill_frac):
        number_of_teeth = len(fill_frac)
    elif number_of_teeth == None:
        number_of_teeth = int(np.ceil(width_grating / period))
    # Populate lists if input was not list
    if not is_listable(period):
        period = [period] * number_of_teeth
    if not is_listable(fill_frac):
        fill_frac = [fill_frac] * number_of_teeth
    # 1D grating
    if focus_distance is None:
        # Taper and straight section
        path = gdspy.Path(focus_width, origin)
        path.segment(taper_length, direction='+y', final_width=width, layer=layer_grating)
        path.segment(straight_length, direction='+y', layer=layer_grating)
        sections.append(path)
        # # Backreflector Window
        # total_height = sum(period)
        # sections.append(gdspy.Rectangle(
        #     (path.x - w - window_extra_width,       path.y - window_extra_height),
        #     (path.x + w + window_extra_width,       path.y + window_extra_height + total_height),
        #     layer=layer_backreflector,
        # ))
        # Grating section
        t_widths = [fill_frac[i]*period[i] for i in range(number_of_teeth)]
        # t_pos = path.y + 0.5*(period[0]-t_widths[0])      # CAREFUL, not sure where to start first tooth
        t_pos = path.y                                      # CAREFUL, not sure where to start first tooth
        for i in range(number_of_teeth):
            t_pos += period[i]
            sections.append(gdspy.Rectangle(
                (-w,    t_pos),
                ( w,    t_pos - t_widths[i]),
                layer=layer_grating,
            ))
    # Focused Grating
    else:
        # Make each tooth of the grating
        path = gdspy.Path(period[0] * fill_frac[0], origin)
        path_end = (0,0)
        for i in range(number_of_teeth):
            # Calc values
            per = period[i]
            ff = fill_frac[i]
            neff = lda / float(period[0]) + sin_theta               # CAREFUL, using period[0]
            qmin = int(focus_distance / float(period[0]) + 0.5)     # CAREFUL, using period[0]
            # neff = lda / float(period[i]) + sin_theta               # CAREFUL, using period[0]
            # qmin = int(focus_distance / float(period[i]) + 0.5)     # CAREFUL, using period[0]
            max_points = 199 if focus_width < 0 else 2 * evaluations
            c3 = neff ** 2 - sin_theta ** 2
            # Grating section
            q = qmin + i
            c1 = q * lda * sin_theta
            c2 = (q * lda) ** 2
            path.w = 0.5 * per * ff
            path.parametric(
                lambda t: (width * t - w, (c1 + neff * np.sqrt(
                    c2 - c3 * (width * t - w) ** 2)) / c3),
                number_of_evaluations=evaluations,
                max_points=max_points,
                layer=layer_grating)
            path_end = (path.x, path.y)
            path.x = 0
            path.y = 0
        # # Backreflector Window
        # total_height = sum(period)
        # # total_height = 0
        # sections.append(gdspy.Rectangle(
        #     (path_end[0] - width - window_extra_width,       path_end[1] - window_extra_height - total_height),
        #     (path_end[0] + window_extra_width,       path_end[1] + window_extra_height),
        #     layer=layer_backreflector,
        # ))
        # Coupling section
        path.polygons[0] = np.vstack(
            (path.polygons[0][:evaluations, :],
             ([origin] if focus_width == 0 else
              [(0.5 * focus_width, 0),
               (-0.5 * focus_width, 0)])))
        path.fracture()
        sections.append(path)
    # Move and Rotate based on orientation, default is 'out' in '+y'
    # also add to the phidl device
    D = Device('grating_coupler')

    for s in sections:
        if direction == '-x':
            s.rotate(0.5 * np.pi, origin)
        elif direction == '+x':
            s.rotate(-0.5 * np.pi, origin)
        elif direction == '-y':
            s.rotate(np.pi, origin)
        # Rotate an extra 180 degrees if type is 'in'
        if type == 'in':
            s.rotate(np.pi, origin)
        s.translate(position[0], position[1])
        D.add_polygon(s, layer=layer_grating)
    D.add_port(name=1, midpoint = [0,0.05], width = focus_width, orientation = 270)
    D.rotate(180)

    return D


def straight_coupler(radius = 170, angle=40,width_wide = 2, width_wide_ring=3, width_narrow=1, layer=0):
    intermediate_width = (width_wide_ring*1/3+width_narrow*2/3)
    D = Device(name= 'straight coupler')

    P = Path()
    P.append(pp.euler(radius=radius, angle=angle, p=0.6))
    # print('Left euler length: {}'.format(P.length()))

    def width_func(t):
        #taper linearly
        fraction_points= np.array([0, 0.3,0.7, 1])
        width_points = np.array([intermediate_width, width_narrow,width_narrow, intermediate_width])
        return np.interp(t, fraction_points, width_points)

    X = CrossSection()
    X.add(width=width_func, offset=0, ports=(1,2), layer=layer)
    Coupler = P.extrude(width=X)
    coupler = D.add_ref(Coupler)
    coupler.rotate(-angle/2)

    P = Path()
    P.append(pp.euler(radius=radius*1.2, angle=angle/2, p=1))

    # plt.figure(10)
    # s,K = P.curvature()
    # plt.plot(s,K,'.-')
    # plt.xlabel('Position along curve (arc length)')
    # plt.ylabel('Curvature')

    def width_func(t):
        #taper linearly
        fraction_points= np.array([0, 1])
        width_points = np.array([width_wide,  intermediate_width])
        return np.interp(t, fraction_points, width_points)

    X = CrossSection()
    X.add(width=width_func, offset=0, ports=(1,2), layer=layer)
    Connector = P.extrude(width=X)
    connectorL = D.add_ref(Connector)
    connectorL.mirror()
    connectorL.connect(2, coupler.ports[1])
    connectorR = D.add_ref(Connector)
    connectorR.connect(2, coupler.ports[2])
    D.add_port(name=1, port=connectorL.ports[1])
    D.add_port(name=2, port=connectorR.ports[1])
    D.add_port(name=3, midpoint = (D.x, D.ymin), orientation=270)
    return D

def hairpin_coupler(radius = 170, width_wide = 3, width_narrow=1, layer=0):
    D = Device(name= 'hairpin coupler')

    P = Path()
    P.append(pp.euler(radius=radius, angle=180, p=0.5))

    def width_func(t):
        #taper linearly
        fraction_points= np.array([0, 0.4,0.6, 1])
        width_points = np.array([width_wide, width_narrow,width_narrow, width_wide])
        return np.interp(t, fraction_points, width_points)

    X = CrossSection()
    X.add(width=width_func, offset=0, ports=(1,2), layer=layer)
    Coupler = P.extrude(width=X)
    coupler = D.add_ref(Coupler)
    coupler.rotate(90)

    D.add_port(name=1, port=coupler.ports[1])
    D.add_port(name=2, port=coupler.ports[2])
    D.add_port(name=3, midpoint = (D.x, D.ymax), orientation=90)



    return D

def alignment_marks(Xs, Ys, layer=0):
    #places alignment markers in 4 quadrants using the Quadrant I coordinates
    #specified by Xs and Ys
    D = Device('marks')
    Mark = pg.cross(length = 200, width = 0.5, layer=layer)
    for x, y in zip(Xs, Ys):
        for x_side in [-1,1]:
            for y_side in [-1,1]:
                mark = D.add_ref(Mark)
                mark.x = x*x_side
                mark.y = y*y_side
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
