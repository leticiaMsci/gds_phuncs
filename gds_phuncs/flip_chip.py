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

def laser_pad(
    wg_w = 1.5,
    port_pitch = 142,
    gold_pitch = 50,
    gold_edge_distance = 15,
    pad_width = 3000,
    ruler_spacing = 50,
    ruler_w = 2,
    ruler_h = 50,
    pad_wg_overlap = 0.2,
    bonding_layer = 7,
    wg_layer=1,
    pad_layer = 3,
    pwb_trench_w = 300,
    pwb_trench_layer = None,
    pwb_trench_wg_overlap = 0.2,
    align_box_layer=15,
    oxide_pad_layer = 20,
    oxpad_distance = 30,
    distance_alignment_marks = 700
    ):
    '''
    creates laser pad with metal bonding layer and etch pad layer. 
    It also creates two ports for coupling to each laser waveguide displaced at 142 um.
    Adds lithographic rulerfor flip-chip bonding alignment.
    '''

    D = Device("laser pad")
    # ports for future wg connection
    port1 = D.add_port(name=1, midpoint=(-port_pitch/2, 0), width=wg_w, orientation=90)
    port2 = D.add_port(name=2, midpoint=(port_pitch/2, 0), width=wg_w, orientation=90)

    #litho ruler
    Ruler = pg.litho_ruler(height = ruler_h ,
        width = ruler_w,
        spacing = ruler_spacing,
        scale = [3,1,1,1,1,2,1,1,1,1],
        num_marks = 21,
        layer = wg_layer,
        
        )

    ruler1 = D<<Ruler
    ruler1.mirror(p1=(ruler_w/2, 1), p2=(ruler_w/2, 0))
    ruler1.move(origin = [ruler_w/2, 0], destination=[port1.x-ruler_spacing, port1.y])
    ruler2 = D<<Ruler
    ruler2.move(origin = [ruler_w/2, 0], destination=[port2.x+ruler_spacing, port2.y])

    

    # bonding pads
    Rpad = pg.rectangle(size=[pad_width, pad_width], layer = pad_layer)
    Rpad.move(origin = [Rpad.x, Rpad.ymax], destination = [D.x, port1.y+pad_wg_overlap])
    rpad = D<<Rpad

    if pwb_trench_layer is not None:
        Trench=pg.rectangle(size = (pad_width, pwb_trench_w), layer= pwb_trench_layer)
        trench1 = D<<Trench
        trench1.move(origin=(trench1.x, trench1.ymax), destination=(rpad.x, rpad.ymax+pwb_trench_wg_overlap-pad_wg_overlap))
        Align= gds.alignment_mark_ebeam(mark_layer = wg_layer, caliper_layers = [wg_layer, pwb_trench_layer, pad_layer], label="L", align_box_layer=align_box_layer)

        OxidePAd = pg.rectangle(size = (pad_width, rpad.ysize-trench1.ysize-oxpad_distance), layer=oxide_pad_layer)
        oxpad = D<<OxidePAd
        oxpad.x = rpad.x
        oxpad.ymax = trench1.ymin-oxpad_distance

        # gold bonding pads
        R = pg.rectangle(size=[(pad_width-gold_pitch)/2, oxpad.ysize-gold_edge_distance], layer = bonding_layer)
    
    else:
        Align= gds.alignment_mark_ebeam(mark_layer = wg_layer, caliper_layers = [wg_layer, pad_layer], label="L", align_box_layer=align_box_layer)

        # gold bonding pads
        R = pg.rectangle(size=[(pad_width-gold_pitch)/2, pad_width-gold_edge_distance-2*pad_wg_overlap], layer = bonding_layer)

    r1 = D<<R
    r1.move(origin=[r1.xmax, r1.ymin], destination = [-gold_pitch/2,rpad.ymin])
    r2 = D<<R
    r2.move(origin=[r2.xmin, r2.ymin], destination = [gold_pitch/2,rpad.ymin])

    align1 = D<<Align
    align1.center = rpad.xmax+distance_alignment_marks,rpad.ymin-distance_alignment_marks
    align2 = D<<Align
    align2.center = -align1.x,align1.y
    align3 = D<<Align
    align3.center = -align1.x,rpad.ymax+distance_alignment_marks
    align4 = D<<Align
    align4.center = align1.x,rpad.ymax+distance_alignment_marks
    
    return D.flatten()


def pwb_link_trench(
    wg_w = 1.5,
    port_pitch = 142,
    pad_width = 2000,
    ruler_spacing = 50,
    ruler_w = 2,
    ruler_h = 50,
    wg_layer=1,
    pwb_trench_w = 300,
    pwb_trench_layer = 50,
    pwb_trench_wg_overlap = 0.2,
    align_box_layer=None,
    distance_alignment_marks = 500
    ):
    '''
    creates laser pad with metal bonding layer and etch pad layer. 
    It also creates two ports for coupling to each laser waveguide displaced at 142 um.
    Adds lithographic rulerfor flip-chip bonding alignment.
    '''

    D = Device("laser pad")
    # ports for future wg connection
    # port1 = D.add_port(name='1', midpoint=(0, port_pitch/2), width=wg_w, orientation=90)
    # port2 = D.add_port(name='2', midpoint=(0,-port_pitch/2), width=wg_w, orientation=-90)

    RulerPort = gds.ruler_ports(ruler_spacing = ruler_spacing,
                ruler_w = ruler_w,
                ruler_h = ruler_h,
                port_spacing = port_pitch,
                N_ports = 1,
                wg_layer = wg_layer,
                port_w = wg_w, ruler_max_w = int(pad_width/2))

    ruler_port1 = D<<RulerPort
    port1 = ruler_port1.ports[1]

    ruler_port2 = D<<RulerPort
    ruler_port2.rotate(180)
    port2 = ruler_port2.ports[1]
    
    # adding trench
    Trench=pg.rectangle(size = (pad_width, pwb_trench_w), layer= pwb_trench_layer)
    trench1 = D<<Trench
    trench1.move(origin=(trench1.x, trench1.ymax), destination=(port1.x, port1.y+pwb_trench_wg_overlap))
    # moving port 2
    ruler_port2.move(origin=port2.center, destination=(port1.x, trench1.ymin+pwb_trench_wg_overlap))

    # alignment marks
    if align_box_layer is not None:
        Align= gds.alignment_mark_ebeam(mark_layer = wg_layer, caliper_layers = [wg_layer, pwb_trench_layer], label="L", align_box_layer=align_box_layer)
        align1 = D<<Align
        align1.center = trench1.xmax+distance_alignment_marks,trench1.ymin-distance_alignment_marks
        align2 = D<<Align
        align2.center = -align1.x,align1.y
        align3 = D<<Align
        align3.center = -align1.x,trench1.ymax+distance_alignment_marks
        align4 = D<<Align
        align4.center = align1.x,trench1.ymax+distance_alignment_marks

    # ports
    D.add_port(name = 1, port=ruler_port1.ports[1])
    D.add_port(name = 2, port=ruler_port2.ports[1])
    
    return D.flatten()

def deep_etch_frame(x,y, layer=15, deep_etch_w=500, deep_etch_overlap = 0.2, align_layer = 2, align_box_layer=15):
    F = Device()
    
    rectangle = pg.rectangle(size=(x,y))
    F<<pg.outline(rectangle, distance = deep_etch_w, precision = 1e-6, layer = layer)
    F.add_port(name = "E", midpoint = (rectangle.xmax+deep_etch_overlap, rectangle.y))
    F.add_port(name = "N", midpoint = (rectangle.x, rectangle.ymax+deep_etch_overlap))
    F.add_port(name = "W", midpoint = (rectangle.xmin-deep_etch_overlap, rectangle.y))
    F.add_port(name = "S", midpoint = (rectangle.x, rectangle.ymin-deep_etch_overlap))

    if align_layer is not None:
        Align = gds.alignment_mark_ebeam(mark_layer=align_layer, label = "C", caliper_bool=False, align_box_layer=align_box_layer)
        a1 = F<<Align
        a1.xmax, a1.ymax = F.xmax, F.ymax
        a2 = F<<Align
        a2.xmax, a2.ymin = F.xmax, F.ymin
        a3 = F<<Align
        a3.xmin, a3.ymin = F.xmin, F.ymin
        a4 = F<<Align
        a4.xmin, a4.ymax = F.xmin, F.ymax



    return F

def msc_laser_pad(MSC, LaserPad0):

    LaserPad = Device()
    laserpad=LaserPad<<LaserPad0

    msc1 = LaserPad<<MSC
    msc1.connect(port = 1, destination=laserpad.ports[1])

    msc2 = LaserPad<<MSC
    msc2.connect(port = 1, destination=laserpad.ports[2]) 

    # LaserPad.remove(LaserPad.ports["1"])
    # LaserPad.remove(LaserPad.ports["2"])

    LaserPad.add_port(name=1, port = msc1.ports[2])
    LaserPad.add_port(name=2, port = msc2.ports[2])

    return LaserPad.flatten()

def msc_amp_pad(MSC_list, LaserPad0):

    LaserPad = Device()
    laserpad=LaserPad<<LaserPad0

    for ii, MSC in enumerate(MSC_list):
        msc = LaserPad<<MSC
        msc.connect(port = 1, destination=laserpad.ports[ii+1])
        LaserPad.add_port(name=ii+1, port = msc.ports[2])

    # msc2 = LaserPad<<MSC
    # msc2.connect(port = 1, destination=laserpad.ports[4]) 

    # # LaserPad.remove(LaserPad.ports["1"])
    # # LaserPad.remove(LaserPad.ports["2"])

    # LaserPad.add_port(name=3, port = msc1.ports[2])
    # LaserPad.add_port(name=4, port = msc2.ports[2])

    return LaserPad.flatten()


def amplifier_pad(wg_w = 1.5,
    port_pitch = 142,
    gold_pitch = 50,
    gold_edge_distance = 15,
    pad_width = 3000,
    pad_height = 2000,
    ruler_spacing = 50,
    ruler_w = 2,
    ruler_h = 50,
    pad_wg_overlap = 0.2,
    bonding_layer = 7,
    wg_layer=1,
    pad_layer = 3,
    pwb_trench_w = 300,
    pwb_trench_layer = None,
    pwb_trench_wg_overlap = 0.2,
    align_box_layer=None,
    oxide_pad_layer = 20,
    oxpad_distance = 30,
    distance_alignment_marks = 150
    ):

    '''
    creates laser pad with metal bonding layer and etch pad layer. 
    It also creates two ports for coupling to each laser waveguide displaced at 142 um.
    Adds lithographic rulerfor flip-chip bonding alignment.
    '''

    D = Device("laser pad")

    RulerPort = gds.ruler_ports(ruler_spacing = ruler_spacing,
                    ruler_w = ruler_w,
                    ruler_h = ruler_h,
                    port_spacing = port_pitch,
                    N_ports = 2,
                    wg_layer = wg_layer,
                    port_w = wg_w, ruler_max_w = int(pad_width/2))

    #adding ruler at the top
    ruler_port1 = D<<RulerPort
    ruler_port1.x = 0
    port1 = ruler_port1.ports[1]
    port2 = ruler_port1.ports[2]

    # bonding pads
    Rpad = pg.rectangle(size=[pad_width, pad_height], layer = pad_layer)
    Rpad.move(origin = [Rpad.x, Rpad.ymax], destination = [D.x, port1.y+pad_wg_overlap])
    rpad = D<<Rpad

    if pwb_trench_layer is not None:
        if align_box_layer is not None:
            Align= gds.alignment_mark_ebeam(mark_layer = wg_layer, caliper_layers = [wg_layer, pwb_trench_layer, pad_layer], label="L", align_box_layer=align_box_layer)
        #pwb trenct at the top
        Trench=pg.rectangle(size = (pad_width, pwb_trench_w), layer= pwb_trench_layer)
        trench1 = D<<Trench
        trench1.move(origin=(trench1.x, trench1.ymax), destination=(rpad.x, rpad.ymax+pwb_trench_wg_overlap-pad_wg_overlap))
        
        # square for oxide etching
        OxidePAd = pg.rectangle(size = (pad_width, rpad.ysize-2*trench1.ysize-2*oxpad_distance), layer=oxide_pad_layer)
        oxpad = D<<OxidePAd
        oxpad.x = rpad.x
        oxpad.ymax = trench1.ymin-oxpad_distance

        # gold bonding pads
        R = pg.rectangle(size=[(pad_width-gold_pitch)/2, oxpad.ysize-gold_edge_distance], layer = bonding_layer)
        
        # pwb trench at the bottom
        trench2 = D<<Trench
        trench2.move(origin=(trench2.x, trench2.ymin), destination=(rpad.x, rpad.ymin+pwb_trench_wg_overlap-pad_wg_overlap))
        

    else:
        if align_box_layer is not None:
            Align= gds.alignment_mark_ebeam(mark_layer = wg_layer, caliper_layers = [wg_layer, pad_layer], label="L", align_box_layer=align_box_layer)

        # gold bonding pads
        R = pg.rectangle(size=[(pad_width-gold_pitch)/2, pad_width-gold_edge_distance-2*pad_wg_overlap], layer = bonding_layer)

    r1 = D<<R
    r1.move(origin=[r1.xmax, r1.y], destination = [-gold_pitch/2,rpad.y])
    r2 = D<<R
    r2.move(origin=[r2.xmin, r2.y], destination = [gold_pitch/2,rpad.y])


    # adding ruler at the bottom
    ruler_port2 = D<<RulerPort
    ruler_port2.rotate(180)
    ruler_port2.ymax = trench2.ymin+pwb_trench_wg_overlap
    ruler_port2.x = ruler_port1.x

    if align_box_layer is not None:
        align1 = D<<Align
        align1.move(origin = align1.ports["c"].center, destination = (rpad.xmax+distance_alignment_marks,rpad.ymin-distance_alignment_marks))
        align2 = D<<Align
        align2.move(origin = align2.ports["c"].center, destination = (-align1.ports["c"].x,align1.ports["c"].y))
        align3 = D<<Align
        align3.move(origin = align3.ports["c"].center, destination = (-align1.ports["c"].x,rpad.ymax+distance_alignment_marks))
        # align3.center = -align1.x,rpad.ymax+distance_alignment_marks
        align4 = D<<Align
        align4.move(origin = align4.ports["c"].center, destination = (rpad.xmax+distance_alignment_marks,rpad.ymax+distance_alignment_marks))

    D.add_port(name = 1, port=ruler_port1.ports[1])
    D.add_port(name = 2, port=ruler_port1.ports[2])
    D.add_port(name = 3, port=ruler_port2.ports[1])
    D.add_port(name = 4, port=ruler_port2.ports[2])

    return D.flatten()