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
from gdsphuncs import gds

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
        lda = 1.55,
        sin_theta = np.sin(np.pi * -8 / 180),
        evaluations=99,
        overlap=0.05,
        layer=0
):
    '''
    Generalized function for making any kind of grating: Focused/1D and Uniform/Apodized
    Taper is included by default.

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
    lda : positive number
        free-space wavelength.
    sin_theta : number
        sine of incidence angle.
    evaluations : positive integer
        number of parametric evaluations of `path.parametric`.
    layer : positive integer
        GDSII layer number.
    Returns
    -------
    out : array-like[3]
        phidl Device object
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
        path.segment(taper_length, direction='+y', final_width=width, layer=layer)
        path.segment(straight_length, direction='+y', layer=layer)
        sections.append(path)
        # Grating section
        t_widths = [fill_frac[i]*period[i] for i in range(number_of_teeth)]
        # t_pos = path.y + 0.5*(period[0]-t_widths[0])      # CAREFUL, not sure where to start first tooth
        t_pos = path.y                                      # CAREFUL, not sure where to start first tooth
        for i in range(number_of_teeth):
            t_pos += period[i]
            sections.append(gdspy.Rectangle(
                (-w,    t_pos),
                ( w,    t_pos - t_widths[i]),
                layer=layer,
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
                layer=layer)
            path_end = (path.x, path.y)
            path.x = 0
            path.y = 0
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
        D.add_polygon(s, layer=layer)
    D.add_port(name=1, midpoint = [0,overlap], width = focus_width, orientation = 270)
    D.rotate(180)

    return D

"""
def add_grating_general(period,
                        number_of_teeth,
                        fill_frac,
                        width,
                        lda=1,
                        sin_theta=0,
                        focus_distance=10,
                        focus_width=0.8,
                        evaluations=50,
                        overlap=0,
                        layer=0,
                        datatype=0,
                        focused=True,
                        apodized=False,
                        apodize_func=None):
    '''
    Straight or focusing grating, with or without apodization. 
    Waveguide taper is not explicitly included.

    period          : grating period
    number_of_teeth : number of teeth in the grating
    fill_frac       : filling fraction of the teeth with respect to the period
    width           : width of the grating
    lda             : free-space wavelength
    sin_theta       : sine of incidence angle
    focus_distance  : focus distance
    focus_width     : width of the waveguide connecting to the grating
    overlap         : Port position relative to grating edge
    evaluations     : number of parametric evaluations of `path.parametric
    layer           : layer object

    Return Phidl device
    '''
    D = Device('grating_coupler')
    if focused: 
        position = (0,0)
        if apodized:
            if apodize_func is None:
                print("Apodization function cannot be None if apodize_func is True!")
                return D
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
        D.add_polygon(path, layer=layer)
        D.add_port(name=1, midpoint = [0,overlap], width = focus_width, orientation = 270)
        D.rotate(180)
        return D    
    else:
        if apodized:
            if apodize_func is None:
                print("Apodization function cannot be None if apodize_func is True!")
                return D
        else:
            for i in range(number_of_teeth):
                cgrating = D.add_ref(pg.compass(size = [period*fill_frac,
                                                     width],
                                             layer = layer))
                cgrating.x += i*period

            # make the taper
            tgrating = D.add_ref(pg.taper(length = focus_distance,
                                       width1 = width,
                                       width2 = focus_width,
                                       port = None, layer = layer))
            print(tgrating.ports)
            print(cgrating.ports)
            # tgrating.xmin = cgrating.xmax
            tgrating.connect(1, cgrating.ports['E'])
            # define the port of the grating
            p = D.add_port(midpoint=[0, tgrating.ymax - overlap], width=focus_width, name = 1, orientation=90)
        return D
"""

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

def _taper(length=10, width1=0.8, width2=1.4, trapz=True, layer=1):
    T = Device('taper')
    if trapz:
            T.add_polygon([(0, 0), (width1, 0), (width2, length), (0, length)], layer=layer)
            T.add_port(name = 1, midpoint = [width1/2, 0], width = width1, orientation = 270)
            T.add_port(name = 2, midpoint = [width2/2, length], width = width2, orientation = 90)
            T.move([-width1/2, 0])
    else:
            T.add_polygon([(-width1/2, 0), (width1/2, 0), (width2/2, length), (-width2/2, length)], layer=layer)
            T.add_port(name = 1, midpoint = [0, 0], width = width1, orientation = 270)
            T.add_port(name = 2, midpoint = [0, length], width = width2, orientation = 90)
    return T

# def adiabatic_coupler(input_width=0.8, 
#     narrow_width=0.8, 
#     wide_width=3.0,
#     output_width=1.0, 
#     taper_length=100,
#     bend_length=40, 
#     coupling_length=200, 
#     coupling_gap=0.5, 
#     spacing=5.0, 
#     layer=1):
#     """
#     Creates an adiabatic coupler w 

#     Parameters
#     ------------------
#     input_width: float
#         input waveguide width in um. 
#     narrow_width: float
#         narrow waveguide width in um.
#     wide_width : float
#         wide waveguide width in um 
#     taper_length : float
#         length of taper from input_width to 
#         wide/narrow waveguide width in um.
#     bend_length : float
#         length of bend region in um.
#     coupling_length : float
#         adiabatic coupling region length in um.
#     coupling_gap : float
#         adiabatic coupling region gap in um/
#     spacing : float
#         input waveguide spacing, in um. 
#     layer : int or layer object
#         gds layer
#     """ 
#     AC = Device('Adiabatic coupler')

#     path_narrow = Device('Narrow arm')
#     path_wide = Device('Wide arm')

#     narrow_input = path_narrow << _taper(length=taper_length, 
#         width1=input_width, 
#         width2=narrow_width, 
#         trapz=False, 
#         layer=layer)
#     narrow_bend = path_narrow << _taper(length=bend_length,
#         width1=narrow_width,
#         width2=narrow_width,
#         trapz=False,
#         layer=layer)
#     narrow_bend.connect(port=1, destination=narrow_input.ports[2])
#     narrow_taper = path_narrow << _taper(length=coupling_length,
#         width1=narrow_width,
#         width2=output_width,
#         trapz=True,
#         layer=layer).mirror()
#     narrow_taper.connect(port=1, destination=narrow_bend.ports[2])

#     wide_input = path_wide << _taper(length=taper_length, 
#         width1=input_width, 
#         width2=wide_width, 
#         trapz=False, 
#         layer=layer)
#     path_wide.move([spacing, 0])
#     wide_taper = path_wide << _taper(length=coupling_length,
#         width1=wide_width,
#         width2=output_width,
#         trapz=True,
#         layer=layer).move([coupling_gap + (narrow_width+wide_width)/2, taper_length + bend_length])
#     # wide_bend = path_wide << gds.route_S(wide_input.ports[2], wide_taper.ports[1], width=wide_width, layer=layer)
#     wide_bend = path_wide << gds.route_S(wide_input.ports[2], wide_taper.ports[1], layer=layer)

#     AC << path_narrow
#     AC << path_wide

#     AC.add_port(name=1, midpoint=[0, 0], width=input_width, orientation=270)
#     AC.add_port(name=2, midpoint=[spacing, 0], width=input_width, orientation=270)
#     AC.add_port(name=3, midpoint=[(narrow_width-output_width)/2, taper_length+bend_length+coupling_length], width=output_width, orientation=90)
#     AC.add_port(name=4, midpoint=[(narrow_width+output_width)/2+coupling_gap, taper_length+bend_length+coupling_length], width=output_width, orientation=90)

#     return AC

def adiabatic_coupler(input_width=0.8, 
    narrow_width=0.8, 
    wide_width=3.0,
    output_width=1.0, 
    taper_length=100,
    bend_length=40, 
    coupling_length=200, 
    coupling_gap=0.5, 
    spacing=5.0,
    compact=False, 
    layer=1):
    """
    Creates an adiabatic coupler w 

    Parameters
    ------------------
    input_width: float
        input waveguide width in um. 
    narrow_width: float
        narrow waveguide width in um.
    wide_width : float
        wide waveguide width in um 
    taper_length : float
        length of taper from input_width to 
        wide/narrow waveguide width in um.
    bend_length : float
        length of bend region in um.
    coupling_length : float
        adiabatic coupling region length in um.
    coupling_gap : float
        adiabatic coupling region gap in um/
    spacing : float
        input waveguide spacing, in um. 
    layer : int or layer object
        gds layer
    """ 
    AC = Device('Adiabatic coupler')

    narrow_taper = AC << _taper(length=coupling_length,
        width1=narrow_width,
        width2=output_width,
        trapz=True,
        layer=layer).mirror()
    wide_taper = AC << _taper(length=coupling_length,
        width1=wide_width,
        width2=output_width,
        trapz=True,
        layer=layer).move([coupling_gap + (narrow_width+wide_width)/2, 0])

    if not compact:
        path_narrow = Device('Narrow arm')
        path_wide = Device('Wide arm')
        
        narrow_input = AC << _taper(length=taper_length, 
            width1=input_width, 
            width2=narrow_width, 
            trapz=False, 
            layer=layer)
        narrow_bend = AC << _taper(length=bend_length,
            width1=narrow_width,
            width2=narrow_width,
            trapz=False,
            layer=layer)

        narrow_bend.connect(port=2, destination=narrow_taper.ports[1])
        narrow_input.connect(port=2, destination=narrow_bend.ports[1])

        wide_input = AC << _taper(length=taper_length, 
            width1=input_width, 
            width2=wide_width, 
            trapz=False, 
            layer=layer)
        wide_input.x = narrow_input.ports[1].x + spacing
        wide_input.ymin = narrow_input.ports[1].y

        wide_bend = AC << gds.route_S(wide_input.ports[2], wide_taper.ports[1], layer=layer)

    if compact:
        AC.add_port(name=1, port=narrow_taper.ports[1])
        AC.add_port(name=2, port=wide_taper.ports[1])     
    else:
        AC.add_port(name=1, port=narrow_input.ports[1])
        AC.add_port(name=2, port=wide_input.ports[1])

    AC.add_port(name=3, port=narrow_taper.ports[2])
    AC.add_port(name=4, port=wide_taper.ports[2])

    return AC

def straight_coupler(radius = 170, angle=40, coupling_width=1.2, waveguide_width=0.8, layer=0):
    """
    Evanescent coupler for rings. Circular coupling region, 
    straight waveguide inputs.  
    """
    intermediate_width = (waveguide_width*1/3+coupling_width*2/3)
    D = Device(name= 'straight coupler')

    P = Path()
    P.append(pp.euler(radius=radius, angle=angle, p=0.6))
    # print('Left euler length: {}'.format(P.length()))

    def width_func(t):
        #taper linearly
        fraction_points= np.array([0, 0.3,0.7, 1])
        width_points = np.array([intermediate_width, coupling_width, coupling_width, intermediate_width])
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
        width_points = np.array([waveguide_width,  intermediate_width])
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
    """
    Hairpin (U) shape for couplers or testers.
    """
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

def circular_coupler(radius_coupler=80, coupler_angle=30, radius_bend=60, width_coupler=1, width_io=0.8, pitch_io=127, length_taper=50, plot_curvature=False, layer=0):
    """
    Make a circular ring coupler for grating coupled devices. 
    Similar to hairpin but with circular coupling region. 
    
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
    X.add(width=width_coupler, offset=0, ports=(1,2),name='main', layer=layer)
    Coupler = P1.extrude(width=X)
    Coupler.rotate(90-coupler_angle)
    coupler = D.add_ref(Coupler, alias='coupler')

    #make coupler bottom bend
    P2 = pp.euler(radius=radius_bend, angle=-coupler_angle, p=1)
    bend_bottom = D.add_ref(P2.extrude(width=X))
    bend_bottom.connect(2, coupler.ports[1])
    #make coupler bend top
    P3 = pp.euler(radius=radius_bend, angle=-coupler_angle+180, p=0.5)
    bend_top = D.add_ref(P3.extrude(width=X))
    bend_top.connect(1, coupler.ports[2])


    P_total = Path()
    P_total.append((P1, P2, P3))
    # if plot_curvature:
    #     plt.figure(num='Coupler curvature')
    #     s,K = P_total.curvature()
    #     plt.plot(s,K,'.-')
    #     plt.xlabel('Position along curve (arc length)')
    #     plt.ylabel('Curvature')

    #Make the taper sections
    P = pp.straight(length=length_taper)
    X_io = CrossSection()
    X_io.add(width=width_io, offset=0, ports=(1,2), name='main', layer=layer)
    X_taper = pp.transition(cross_section1=X, cross_section2=X_io, width_type='linear')
    Taper = P.extrude(X_taper)
    Taper.rotate(-90)
    taper_right = D.add_ref(Taper)
    taper_right.connect(1, bend_bottom.ports[1])
    taper_left = D.add_ref(Taper)
    taper_left.move(origin=(taper_left.x, taper_left.y), destination=(taper_right.x-pitch_io, taper_right.y))

    #route to taper
    D.add_ref(pr.route_basic(port1=bend_top.ports[2], port2=taper_left.ports[1], path_type='sine', width_type='straight'))
    D.add_port(name=1, midpoint = [taper_right.ports[2].x, taper_right.ports[2].y], width = width_io, orientation = 270)
    D.add_port(name=2, midpoint = [taper_left.ports[2].x, taper_left.ports[2].y], width = width_io, orientation = 270)

    return D
