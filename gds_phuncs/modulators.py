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
from gds_phuncs import rings
from gds_phuncs import couplers


def phase_modulator(
    electrode_length = 400,
    wg_length = 500,
    wg_w = 2,
    metal_layer = 3,
    wg_layer = 1):

    gs = mzis.gs(electrode_width = 50, electrode_length=electrode_length, layer = metal_layer)

    D = Device()
    port1 = D.add_port(name='1', midpoint=(0, 0), width=wg_w, orientation=0)
    port2 = D.add_port(name='2', midpoint=(wg_length, 0), width=wg_w, orientation=0)
    D.add_ref(pr.route_smooth(port1, port2, path_type='straight', layer = wg_layer))
    waypoint_path = pr.path_straight(port1, port2)

    gs.center = [D.x, D.y]

    D<<gs
    
    return D.flatten()




def mzi_modulator_wgs(
        g = 3,
        L_mod = 500,
        y_distance = 20,
        wg_layer = 1,
        wg_w = 1.5, 
        G=10,
        open_for_gsg = False,
        waveguide_pitch = 150,
        bend_radius = 40,
        heater=True,
        heater_length = 900
    ):
    """
    draws waveguides for mzi modulator.
    g is the gap between electrodes around the wg.
    L_mod is the modulatio length
    y_distance  is the distance between the y-splitter and the modulation section.
    wg_layer layer for wavegudies
    G is the gap between the two MZI waveguides
    open_for_gsg is a boolean that tells whether the y-splitter will "open up" more to have the gsg probes inside and outside of it.
    waveguide_pitch is the pitch between the y-splitter arm if one decides to open_for_gsg. This should be the pitch of the gsg probes.
    bend_radius is the routing radius of the bend if open_for_gsg is false. It should always be straight, so i dont think this parameter matters.
    """


    D = Device("amplitude modulator")

    # ports for semented electrode region
    port1 = D.add_port(name='1', midpoint=(0, G/2), width=wg_w, orientation=0)
    port2 = D.add_port(name='2', midpoint=(L_mod, G/2), width=wg_w, orientation=180)
    rtop = pr.route_smooth(port1, port2, layer = wg_layer)
    D<<rtop

    port3 = D.add_port(name='3', midpoint=(0, -G/2), width=wg_w, orientation=0)
    port4 = D.add_port(name='4', midpoint=(L_mod, -G/2), width=wg_w, orientation=180)
    rbot = pr.route_smooth(port3, port4, layer = wg_layer)
    D<<rbot

    # creating y-splitters
    if open_for_gsg:
        Y1=mzis.ybranch_sine(waveguide_layer=wg_layer, waveguide_width = wg_w, waveguide_pitch = waveguide_pitch, branch_length=y_distance)
        Y2=mzis.ybranch_sine(waveguide_layer=wg_layer, waveguide_width = wg_w, waveguide_pitch = waveguide_pitch, branch_length=y_distance).rotate(180)
    else:
        Y1=mzis.ybranch(waveguide_layer=wg_layer, waveguide_width = wg_w, waveguide_pitch = G)
        Y2=mzis.ybranch(waveguide_layer=wg_layer, waveguide_width = wg_w, waveguide_pitch = G).rotate(180)
    if heater:
        Straight = pg.straight(size = (wg_w,heater_length), layer = wg_layer).rotate(90)
        s1 = Y2<<Straight
        s1.connect(1, Y2.ports["out1"])
        Y2.remove(Y2.ports["out1"])
        Y2.add_port(name="out1", port = s1.ports[2])
        Y2.add_port(name="heat1", midpoint = s1.center)
        s2 = Y2<<Straight
        s2.connect(1, Y2.ports["out2"])
        Y2.remove(Y2.ports["out2"])
        Y2.add_port(name="out2", port = s2.ports[2])
        Y2.add_port(name="heat2", midpoint = s2.center)

    # open up for left probe contact
    y1 = D<<Y1
    y1.move(origin = (y1.ports["out2"].x, 0), destination = (port1.x-y_distance, 0))

    # open up for right probe contact
    y2 = D<<Y2
    y2.move(origin = (y2.ports["out2"].x, 0), destination = (port2.x+y_distance, 0))
    
    
    
    if open_for_gsg:
        r11=gds.route_S(y1.ports["out2"],rtop.ports[1], rmin=80, layer=wg_layer)
        D<<r11
        r12=gds.route_S(y1.ports["out1"],rbot.ports[1], rmin=80, layer=wg_layer)
        D<<r12
        r21=gds.route_S(rtop.ports[2], y2.ports["out1"], rmin=80, layer=wg_layer)
        D<<r21
        r22=gds.route_S(rbot.ports[2], y2.ports["out2"], rmin=80, layer=wg_layer)
        D<<r22
    else:
        r11 = pr.route_smooth(y1.ports["out2"],rtop.ports[1], radius = bend_radius, layer=wg_layer)
        D<<r11
        r12 = pr.route_smooth(y1.ports["out1"],rbot.ports[1], radius = bend_radius, layer=wg_layer)
        D<<r12
        r21 = pr.route_smooth(rtop.ports[2], y2.ports["out1"], radius = bend_radius, layer=wg_layer)
        D<<r21
        r22 = pr.route_smooth(rbot.ports[2], y2.ports["out2"], radius = bend_radius, layer=wg_layer)
        D<<r22


    
    D.add_port(name = "in", port = y1.ports["in"])
    D.add_port(name = "out", port = y2.ports["in"])
    D.add_port(name = "y1", midpoint=(y1.ports["out1"].x, y1.y), orientation=0)
    D.add_port(name = "y2", midpoint=(y2.ports["out2"].x, y2.y), orientation=180)

    if heater:
        D.add_port(name="heat1", port = y2.ports["heat1"])
        D.add_port(name="heat2", port = y2.ports["heat2"])


    return D.flatten()

def phase_modulator_wgs(
        g = 3,
        L_mod = 500,
        y_distance = 20,
        wg_layer = 1,
        wg_w = 1.5, 
        G=10,
        open_for_gsg = False,
        waveguide_pitch = 150,
        bend_radius = 40,
        input_y_branch = True
    ):
    """
    draws waveguides for mzi modulator.
    g is the gap between electrodes around the wg.
    L_mod is the modulatio length
    y_distance  is the distance between the y-splitter and the modulation section.
    wg_layer layer for wavegudies
    G is the gap between the two MZI waveguides
    open_for_gsg is a boolean that tells whether the y-splitter will "open up" more to have the gsg probes inside and outside of it.
    waveguide_pitch is the pitch between the y-splitter arm if one decides to open_for_gsg. This should be the pitch of the gsg probes.
    bend_radius is the routing radius of the bend if open_for_gsg is false. It should always be straight, so i dont think this parameter matters.
    """


    D = Device("amplitude modulator")

    # ports for semented electrode region
    port1 = D.add_port(name='1', midpoint=(0, G/2), width=wg_w, orientation=0)
    port2 = D.add_port(name='2', midpoint=(L_mod, G/2), width=wg_w, orientation=180)
    rtop = pr.route_smooth(port1, port2, layer = wg_layer)
    D<<rtop

    port3 = D.add_port(name='3', midpoint=(0, -G/2), width=wg_w, orientation=0)
    port4 = D.add_port(name='4', midpoint=(L_mod, -G/2), width=wg_w, orientation=180)
    rbot = pr.route_smooth(port3, port4, layer = wg_layer)
    D<<rbot

    # creating y-splitters
    if open_for_gsg:
        Y1=mzis.ybranch_sine(waveguide_layer=wg_layer, waveguide_width = wg_w, waveguide_pitch = waveguide_pitch, branch_length=y_distance)
        Y2=mzis.ybranch_sine(waveguide_layer=wg_layer, waveguide_width = wg_w, waveguide_pitch = waveguide_pitch, branch_length=y_distance).rotate(180)
    else:
        Y1=mzis.ybranch(waveguide_layer=wg_layer, waveguide_width = wg_w, waveguide_pitch = G)
        Y2=mzis.ybranch(waveguide_layer=wg_layer, waveguide_width = wg_w, waveguide_pitch = G).rotate(180)


    # open up for left probe contact
    y1 = D<<Y1
    y1.move(origin = (y1.ports["out2"].x, 0), destination = (port1.x-y_distance, 0))
    # open up for right probe contact
    y2 = D<<Y2
    y2.move(origin = (y2.ports["out2"].x, 0), destination = (port2.x+y_distance, 0))
    

    if open_for_gsg:
        r11=gds.route_S(y1.ports["out2"],rtop.ports[1], rmin=80, layer=wg_layer)
        D<<r11
        r12=gds.route_S(y1.ports["out1"],rbot.ports[1], rmin=80, layer=wg_layer)
        D<<r12
        r21=gds.route_S(rtop.ports[2], y2.ports["out1"], rmin=80, layer=wg_layer)
        D<<r21
        r22=gds.route_S(rbot.ports[2], y2.ports["out2"], rmin=80, layer=wg_layer)
        D<<r22
        
    else:
        r11 = pr.route_smooth(y1.ports["out2"],rtop.ports[1], radius = bend_radius, layer=wg_layer)
        D<<r11
        r12 = pr.route_smooth(y1.ports["out1"],rbot.ports[1], radius = bend_radius, layer=wg_layer)
        D<<r12
        r21 = pr.route_smooth(rtop.ports[2], y2.ports["out1"], radius = bend_radius, layer=wg_layer)
        D<<r21
        r22 = pr.route_smooth(rbot.ports[2], y2.ports["out2"], radius = bend_radius, layer=wg_layer)
        D<<r22
        

    if input_y_branch:
        D.add_port(name = "in", port = y1.ports["in"])
        D.add_port(name = "out1", port = port2)
        D.add_port(name = "out2", port = port4)
        
    else:
        D.remove(y1)
        D.remove(y2)
        D.add_port(name = "in1", midpoint = r11.ports[1].center, orientation = 180)
        D.add_port(name = "in2", midpoint = r12.ports[1].center, orientation = 180)
        D.add_port(name = "out1", midpoint = r21.ports[2].center, orientation = 0)
        D.add_port(name = "out2", midpoint = r22.ports[2].center, orientation = 0)


    D.add_port(name = "y1", midpoint=(rbot.xmin, y1.y), orientation=0)
    D.add_port(name = "y2", midpoint=(rbot.xmax, y1.y), orientation=180)
        

    return D.flatten()

def phase_modulator_separate_arms(
    g,
    G=30,
    L_p_mod = 2000,
    wg_w = 1.5,
    H_termination = 20,
    L_termination = 100,
    termination_layer = None,
    gold_layer = 6, 
    gsg_pitch=150,
    gsg_pad_h = 125,
    y_distance = 200,
    wg_layer = 1
    ):

    """
    gsg phase modulator but arms are not connected by y-branch.
    """


    SE = Device()
    WG = phase_modulator_wgs(L_mod=L_p_mod, G=G, waveguide_pitch=gsg_pitch, wg_w = wg_w, wg_layer=wg_layer,
                            y_distance=y_distance, open_for_gsg = True, input_y_branch=False)

    SE<<WG

    # space around wg
    O  = pg.offset(WG, distance = g/2-wg_w/2, join_first = True, precision = 1e-6,
            num_divisions = [1,1], layer = 0)


    # Ol10 big pads
    M2 = pg.rectangle(size = (L_p_mod+2*y_distance, gsg_pitch*2+2*gsg_pad_h), layer = gold_layer)
    M2.move(origin = (M2.xmax, M2.y), destination = (WG.xmax, WG.y))


    Gold_Pads = pg.boolean(A = M2, B = O, operation = 'not', precision = 1e-6,
                   num_divisions = [1,1], layer = gold_layer)
    SE<<Gold_Pads


    Termination = Device()
    Rsig = pg.rectangle(size = (L_termination, (G-g)), layer = gold_layer)
    r1 = Termination<<Rsig
    r1.move(origin = (r1.xmin, r1.y), destination = (Gold_Pads.xmax, Gold_Pads.y))

    Rgnd = pg.rectangle(size = (L_termination, gsg_pad_h), layer = gold_layer)
    r2 = Termination<<Rgnd
    r2.move(origin = (r2.xmin, r2.ymax), destination = (Gold_Pads.xmax, Gold_Pads.ymax))
    r3 = Termination<<Rgnd
    r3.move(origin = (r3.xmin, r3.ymin), destination = (Gold_Pads.xmax, Gold_Pads.ymin))
    
    
    if termination_layer is not None:
        T = pg.rectangle(size = (H_termination,M2.ysize), layer = termination_layer)
        term = Termination<<T
        term.move(origin = (term.x, term.y), destination = (r2.x, Gold_Pads.y))

    
    tout = SE<<Termination
    # tin = SE<<Termination
    # tin.move(origin=(tin.xmax, tin.y), destination = (SE.xmin, SE.y))

    SE.add_port(name = 1, port = WG.ports["in1"])
    SE.add_port(name = 2, port = WG.ports["in2"])
    SE.add_port(name = 3, port = WG.ports["out1"])
    SE.add_port(name = 4, port = WG.ports["out2"])
    
    return SE.flatten()


def mzi_modulator_metallized(WG,r = 50,
        s = 3,
        t = 25,
        c = 3,
        h=3,
        W=11,
        g=5,
        G = 50,
        L_mod = 100,
        pad_w = 100,
        pad_l = 120,
        pad_pitch = 150,
        pad_distance = 100,
        segment_layer = 1,
        el6_layer=0,          
        ):
        """
        should improve integration in the future
        """
        
        # Segmented electrodes
        

        xs = [0, r/2-c/2, r/2-c/2, t/2, t/2, r-t/2, r-t/2, r/2+c/2, r/2+c/2, r, r, 0]
        ys = [0, 0, -s, -s, -s-h, -s-h, -s, -s, 0, 0, -W, -W]

        U = Device("unit")
        poly1 = U.add_polygon([xs,ys], layer = segment_layer)

        #step 2 - array
        Nsegments =  L_mod//r
        A = Device("segments 1")           # Create a new blank Device
        d_ref1 = A.add_array(U, columns = Nsegments, rows = 1, spacing = [U.xsize,0])  # Reference the Device "D" that 3 references in it


        
        D=Device("Amplitude Modulator")
        wgs = D<<WG

        L_hich_curr = L_mod

        Segs = Device()
        seg1 = Segs<<A
        seg1.move(origin = [seg1.xmin, seg1.ymax], destination = [wgs.ports["1"].x, wgs.ports["1"].y-g/2])

        seg2 = Segs<<A
        seg2.rotate(-180)
        seg2.move(origin = [seg2.xmin, seg2.ymin], destination = [wgs.ports["3"].x, wgs.ports["3"].y+g/2])


        R = pg.rectangle(size=(L_mod, s), layer = segment_layer)
        r1 = Segs<<R
        r1.move(origin=(r1.xmin, r1.ymin), destination=(wgs.ports["1"].x, wgs.ports["1"].y+g/2))
        r2 = Segs<<R
        r2.move(origin=(r2.xmin, r2.ymax), destination=(wgs.ports["3"].x, wgs.ports["3"].y-g/2))


        
        # high-current part of segmented electrodes
        # center electrode height with overlap with segments
        wc = G-2*W-g+1 #+1 um for overlap

        EL6_HC = Device()
        Rc = pg.rectangle(size=(L_hich_curr, wc), layer = el6_layer)
        rc = EL6_HC<<Rc
        rc.move(origin = rc.center, destination = Segs.center)

        Re = pg.rectangle(size=(L_hich_curr, G), layer = el6_layer)
        re_top = EL6_HC<<Re
        re_top.move(origin=(re_top.x, re_top.ymin), destination = (Segs.x, Segs.ymax-1))

        re_bot = EL6_HC<<Re
        re_bot.move(origin=(re_bot.x, re_bot.ymax), destination = (Segs.x, Segs.ymin+1))

        Segs<<EL6_HC

        #adding ports to electrodes
        port_s1 = Segs.add_port(name="s_1", midpoint = (rc.xmin, rc.y), width = G-g, orientation = 180)
        port_gt1 = Segs.add_port(name="gt_1", midpoint = (re_top.xmin, re_top.y-(s-1)/2), width = (G+s-1)/2, orientation = 180)
        port_gb1 = Segs.add_port(name="gb_1", midpoint = (re_bot.xmin, re_bot.y+(s-1)/2), width = (G+s-1)/2, orientation = 180)

        port_s2 = Segs.add_port(name="s_2", midpoint = (rc.xmax, rc.y), width = G-g, orientation = 0)
        port_gt2 = Segs.add_port(name="gt_2", midpoint = (re_top.xmax, re_top.y-(s-1)/2), width = (G+s-1)/2, orientation = 0)
        port_gb2 = Segs.add_port(name="gb_2", midpoint = (re_bot.xmax, re_bot.y+(s-1)/2), width = (G+s-1)/2, orientation = 0)
        
        D<<Segs
        
        # GSG pads
        GSG = Device()

        Pad = pg.taper(length = pad_l, width1 = pad_w, width2 = pad_w, port = None, layer = el6_layer)

        # left connection
        s1 = GSG<<Pad
        s1.move(origin = (s1.xmax+pad_distance, s1.y), destination = WG.ports["y1"].center)
        GSG<<pr.route_smooth(port_s1, s1.ports[2], path_type='straight', layer = el6_layer)

        gt1 = GSG<<Pad
        gt1.move(origin = (gt1.xmin, gt1.y), destination = (s1.xmin, s1.y+pad_pitch))
        GSG<<pr.route_quad(port_gt1, gt1.ports[2], layer = el6_layer)

        gb1 = GSG<<Pad
        gb1.move(origin = (gb1.xmin, gb1.y), destination = (s1.xmin, s1.y-pad_pitch))
        GSG<<pr.route_quad(port_gb1, gb1.ports[2], layer = el6_layer)

        #right connection
        s2 = GSG<<Pad
        s2.move(origin = (s2.xmin-pad_distance, s2.y), destination = WG.ports["y2"].center)
        GSG<<pr.route_smooth(port_s2, s2.ports[1], path_type='straight', layer = el6_layer)

        gt2 = GSG<<Pad
        gt2.move(origin = (gt2.xmax, gt2.y), destination = (s2.xmax, s2.y+pad_pitch))
        GSG<<pr.route_quad(port_gt2, gt2.ports[1], layer = el6_layer)

        gb2 = GSG<<Pad
        gb2.move(origin = (gb2.xmax, gb2.y), destination = (s2.xmax, s2.y-pad_pitch))
        GSG<<pr.route_quad(port_gb2, gb2.ports[1], layer = el6_layer)

        D<<GSG

        D.add_port(name = "1", port = WG.ports["in"])
        D.add_port(name = "2", port = WG.ports["out"])
        
        return D.flatten()


def segmented_electrodes_teeth(r = 50,
        s = 3,
        t = 25,
        c = 3,
        h=3,
        W=11,
        g=5, L_mod = 100,
        segment_layer=0):
    """
    draws geometry for segmented electrodes. For geometry parameters, see https://doi.org/10.1364/OPTICA.416155 (Prashanta Kharel et.al., Breaking voltage-bandwidth..., Optica 2021).
    Returns positive (A) and negative (B) segments.
    L_mod has to have integer numbers of segments (L_mod must be divisible by r).
    """
    #positive of teeth
    xs = [0, r/2-c/2, r/2-c/2, t/2, t/2, r-t/2, r-t/2, r/2+c/2, r/2+c/2, r, r, 0]
    ys = [0, 0, -s, -s, -s-h, -s-h, -s, -s, 0, 0, -W, -W]

    U = Device("unit")
    teeth = U.add_polygon([xs,ys], layer = segment_layer)

    R = pg.rectangle(size = teeth.size)
    R.move(origin=(R.x, R.ymax), destination = (teeth.x, teeth.ymax))
    teeth_neg = pg.boolean(A = R, B = teeth, operation = 'not', precision = 1e-6,
                   num_divisions = [1,1], layer = segment_layer)
    N = Device()
    N<<teeth_neg




    #step 2 - array
    Nsegments =  L_mod//r
    A = Device("segments 1")           # Create a new blank Device
    segments = A.add_array(U, columns = Nsegments, rows = 1, spacing = [U.xsize,0])  # Reference the Device "D" that 3 references in it

    B = Device("segments 2") 
    segments_negative = B.add_array(N, columns = Nsegments, rows = 1, spacing = [U.xsize,0])

    return A, B


def segmented_mzi_mod_smooth_gsg(segment_layer, el6_layer, gold_layer,
    wg_w=1.5,
    W=11,
    g=5,
    L_mod = 5000,
    G=50,
    gsg_pitch=150,
    y_distance=350,
    gsg_pad_h = 125,
    wg_layer =1,
    heater = True
    ):

    SE = Device()
    WG = mzi_modulator_wgs(L_mod=L_mod, G=G, waveguide_pitch=gsg_pitch, wg_w = wg_w, wg_layer =wg_layer,
                            heater=heater, y_distance=y_distance, open_for_gsg = True)
    SE<<WG

    # O = pg.outline(WG, distance = g/2-wg_w/2, precision = 1e-6, layer = 5)
    O  = pg.offset(WG, distance = g/2-wg_w/2, join_first = True, precision = 1e-6,
            num_divisions = [1,1], layer = 0)
    Obig  = pg.offset(WG, distance = g/2-wg_w/2+W+1, join_first = True, precision = 1e-6,
            num_divisions = [1,1], layer = 99)
    # SE<<Obig


    M1 = pg.rectangle(size = (L_mod+y_distance, gsg_pitch), layer = el6_layer)
    # M1.center = O.center
    M1.xmin = WG.ports["y1"].x+y_distance/2
    M1.y = O.y
    NOT = pg.boolean(A = M1, B = O, operation = 'not', precision = 1e-6,
                   num_divisions = [1,1]) # rectangle minus the gap near the waveguide (O)
    # SE<<NOT

    Segments = pg.boolean(A = NOT, B = Obig, operation = 'and', precision = 1e-6,
                   num_divisions = [1,1], layer = segment_layer)
    # SE<<Segments

    Segments_high_c = pg.boolean(A = NOT, B = Obig, operation = 'not', precision = 1e-6,
                   num_divisions = [1,1], layer = el6_layer)
    SE<<pg.offset(Segments_high_c, distance = 1, join_first = True, precision = 1e-6,
            num_divisions = [1,1], layer = el6_layer)

    # cut out segments
    Seg_cutout = Device()
    _ ,seg_cutout = segmented_electrodes_teeth(segment_layer = segment_layer, L_mod=L_mod)
    seg_top = Seg_cutout<<seg_cutout
    seg_top.movey(1e-6)
    seg_bot = Seg_cutout<<seg_cutout
    seg_bot.rotate(180)
    seg_bot.move(origin=(seg_bot.x, seg_bot.ymin), destination=(seg_top.x, seg_top.ymax-G+g-1e-6))

    Seg_cutout.move(origin=(0,0), destination = (WG.ports["1"].x, WG.ports["1"].y - g/2))
    Segments = pg.boolean(A = Segments, B = Seg_cutout, operation = "not",precision = 1e-6,
                   num_divisions = [1,1], layer = segment_layer)
    SE<<Segments

    # Ol10 big pads
    M2 = pg.rectangle(size = (L_mod+2.3*y_distance, gsg_pitch*2+2*gsg_pad_h), layer = gold_layer)
    M2.xmin = WG.ports["y1"].x+10
    M2.y = O.y


    Gold_Pads = pg.boolean(A = M2, B = Obig, operation = 'not', precision = 1e-6,
                   num_divisions = [1,1], layer = gold_layer)

    SE<<Gold_Pads
    SE.add_port(name = "1", port = WG.ports["in"])
    SE.add_port(name = "2", port = WG.ports["out"])

    if heater:
        SE.add_port(name="heat1", port = WG.ports["heat1"])
        SE.add_port(name="heat2", port = WG.ports["heat2"])

    return SE.flatten()


def resonant_modulator(coupling_gap, electrode_gap, waveguide_width=2, radius=140, length=2000, wg_layer=1, ring_layer = 2, mod_layer = 5):
    D = Device()
    resonator = D<<rings.taper_racetrack(radius = radius, bend_radius=radius, 
                                        length = length, width_narrow=waveguide_width,
                                        width_wide = waveguide_width, layer=ring_layer)

    coupler = D<<couplers.straight_coupler(radius = radius, 
                                        angle=40, coupling_width = waveguide_width, 
                                        waveguide_width = waveguide_width,layer=wg_layer).rotate(90)

    coupler.connect(3, resonator.ports['l']).move([-coupling_gap, 0])

    waveguide_pitch = resonator.ysize-waveguide_width
    gsg = D<<mzis.gsg(layer = mod_layer, electrode_gap=electrode_gap, electrode_length = length, waveguide_pitch=waveguide_pitch)
    gsg.connect("c", destination = resonator.ports["c"])

    D.add_port(name=1, port = coupler.ports[1])
    D.add_port(name=2, port = coupler.ports[2])
    
    return D.flatten()