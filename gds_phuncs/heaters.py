
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
from gds_phuncs import gds


def gsg_contact(
    contact_pad_w = 70,
    pitch = 150,
    contact_layer = 5):

    GSG = Device()
    Pad = pg.straight(size = (contact_pad_w,contact_pad_w), layer = contact_layer)

    s = GSG<<Pad
    g1 = GSG<<Pad
    g1.x = s.x-pitch
    g2 = GSG<<Pad
    g2.x = s.x+pitch

    GSG.add_port(name="s", midpoint = s.center, width = contact_pad_w, orientation=-90)
    GSG.add_port(name="gl", midpoint = g1.center, width = contact_pad_w, orientation=-90)
    GSG.add_port(name="gr", midpoint = g2.center, width = contact_pad_w, orientation=-90)
    GSG.add_port(name="route_s1", port=s.ports[1])
    GSG.add_port(name="route_s2", port=s.ports[2])
    GSG.add_port(name="route_gl1", port=g1.ports[1])
    GSG.add_port(name="route_gl2", port=g1.ports[2])
    GSG.add_port(name="route_gr1", port=g2.ports[1])
    GSG.add_port(name="route_gr2", port=g2.ports[2])

    return GSG.flatten()


def heater_straight(
            heater_w = 2,
            heater_l = 100,
            contact_pad_w = 70,
            heater_layer = 1,
            tapering_l = 5):

    H = Device()

    h = H<<pg.straight(size = (heater_w,heater_l), layer = heater_layer).rotate(90)

    Pad = pg.straight(size = (contact_pad_w,contact_pad_w), layer = heater_layer).rotate(90)

    cl = H<<Pad
    cl.xmax, cl.y = h.xmin-tapering_l, h.y
    H<<pr.route_smooth(cl.ports[2], h.ports[1], layer = heater_layer)

    cr = H<<Pad
    cr.xmin, cr.y = h.xmax+tapering_l, h.y
    H<<pr.route_smooth(cr.ports[1], h.ports[2], layer = heater_layer)

    H.add_port(name="c", midpoint = H.center, width = contact_pad_w, orientation=-90)
    H.add_port(name = "l", midpoint = cl.center, width = contact_pad_w, orientation=-90)
    H.add_port(name = "r", midpoint = cr.center, width = contact_pad_w, orientation=-90)

    H.add_port(name="route_cl1", midpoint = [cl.x, cl.ymin], width = contact_pad_w, orientation=-90)
    H.add_port(name="route_cl2", midpoint = [cl.x, cl.ymax], width = contact_pad_w, orientation=90)
    H.add_port(name="route_cr1", midpoint = [cr.x, cr.ymin], width = contact_pad_w, orientation=-90)
    H.add_port(name="route_cr2", midpoint = [cr.x, cr.ymax], width = contact_pad_w, orientation=90)
    
    return H.flatten()



def curved_heater(heater_L = 1000,
                bend_radius = 200,
                offset_heater_wgs = [0],
                heater_w = 1,
                heater_pitch = 150,
                contact_pad_w = 70,
                gold_layer = 6,
                heater_layer = 7,
                wg_layer = 1,
                wg_w=1.5,
                straight_L = 1000,
                ht_bend_radius = 5):

                    
    P_wg = Path()
    P_ht = Path()

    # Create the basic Path components
    left_turn = pp.euler(radius = bend_radius, angle = 90)
    right_turn = pp.euler(radius = bend_radius, angle = -90)
    u_turn = pp.euler(radius = bend_radius, angle = 180)
    straight = pp.straight(length = straight_L)
    long_straight = pp.straight(length = heater_L)

    ht_left_turn = pp.euler(radius = ht_bend_radius, angle = 90)
    ht_right_turn = pp.euler(radius = ht_bend_radius, angle = -90)

    short_turn = pp.euler(radius = 10, angle = 90)

    # Assemble a complex path by making list of Paths and passing it to `append()`
    P_wg.append([
        straight,
        right_turn,
        long_straight,
        u_turn,
        long_straight,
        right_turn,
        straight
    ])
    P_ht.append([
        ht_left_turn,
        long_straight,
        u_turn,
        long_straight,
        ht_left_turn
    ]).rotate(180)

    P_ht.x = P_wg.x
    P_ht.ymin = P_wg.ymin


    CH = Device()

    # waveguide
    X = CrossSection()
    X.add(width=wg_w, offset=0, layer = wg_layer, ports = [1,2])
    wg = CH<<P_wg.extrude(X)

    # heater
    Y = CrossSection()
    ht_1 = np.array([0,0])
    ht_2 = np.array([0,0])

    for ii, offset_heater_wg in enumerate(offset_heater_wgs):
        Y.add(width= heater_w, offset=offset_heater_wg, layer = heater_layer, ports=[10*ii+1,10*ii+2])

        
    curved_heater = CH<<P_ht.extrude(Y)


    GSG = gds.gsg_contacts(layer = [gold_layer, heater_layer]).rotate(-90)
    gsg = CH<<GSG
    gsg.x = curved_heater.x
    gsg.y = curved_heater.ymax+200

    for ii in range(len(offset_heater_wgs)):
        CH<<pr.route_smooth(gsg.ports["g1"], curved_heater.ports[10*ii+1], layer = [heater_layer, gold_layer],
                        smooth_options=  {'corner_fun': pp.euler, 'use_eff': True},radius=100)
        CH<<pr.route_smooth(gsg.ports["s"], curved_heater.ports[10*ii+2], layer = [heater_layer, gold_layer],radius=100)


    CH.add_port(name = 1, midpoint = wg.ports[1].midpoint, orientation = 180, width = wg_w)
    CH.add_port(name = 2, midpoint = wg.ports[2].midpoint, orientation = 0, width = wg_w)
    return CH.flatten()

def routed_double_gsg_heater(position_heater_1, position_heater_2, H, contact_pad_w = 130, gsg_y = 250, gold_layer = 15):
    
    H_Contact = Device()
    h1 = H_Contact<<H
    h1.center = position_heater_1
    h2 = H_Contact<<H
    h2.center = position_heater_2


    pitch = np.abs(h1.ports["l"].x - h1.ports["r"].x) # distance between heater (H) pads
    # create gsg contact
    gsg = H_Contact<<gsg_contact(contact_pad_w = contact_pad_w , pitch = pitch, contact_layer = gold_layer)
    gsg.move(origin = gsg.ports["s"].midpoint, destination = (h1.ports["r"].x, h1.ports["r"].y+gsg_y))
    #routing first ground pad
    R1 = pr.route_sharp(gsg.ports["route_gl2"], h1.ports["route_cl2"], layer = gold_layer)
    r1 = H_Contact<<R1
    H_Contact<<pr.route_quad(h1.ports["route_cl2"], h1.ports["route_cl1"], layer = gold_layer)
    #routing second ground pad
    r2 = H_Contact<<R1
    r2.x = gsg.ports["gr"].x
    H_Contact<<pr.route_sharp(r2.ports[2], h2.ports["route_cl1"], layer =  gold_layer)
    H_Contact<<pr.route_quad(h2.ports["route_cl1"], h2.ports["route_cl2"], layer = gold_layer)
    # # routing signal pad
    H_Contact<<pr.route_quad(gsg.ports["route_s2"], h1.ports["route_cr2"], layer = gold_layer)
    H_Contact<<pr.route_quad(h1.ports["route_cr2"], h2.ports["route_cr1"], layer =  gold_layer)

    H_Contact.add_port(name = 1, port = gsg.ports["route_gl1"])
    H_Contact.add_port(name = 2, port = gsg.ports["route_s1"])
    H_Contact.add_port(name = 3, port = gsg.ports["route_gr1"])
    
    return H_Contact.flatten()


def arc_heater_out(
    heater_R,
    heater_open_angle = 10,
    heater_w = 3.5,
    taper_angle = 5,
    heater_taper_w0 = 7,
    heater_layer = 7,
    coupling_arc_R = 30
):
    Arc = pp.arc(radius = heater_R, angle = (180-heater_open_angle-2*taper_angle)/2)
    TaperArc = pp.arc(radius = heater_R, angle = taper_angle)
    CouplingArc = pp.arc(radius = coupling_arc_R, angle = -(90-heater_open_angle/2))
    

    # taper arc
    P1 = Path()
    P1.append(CouplingArc)
    P1.append(TaperArc)
    # heater half - arc
    P2 = Path()
    P2.append(Arc)


    # Create the transitional CrossSection
    X1 = CrossSection()
    X1.add(width = heater_taper_w0, offset = 0, layer = heater_layer, name = 'heater', ports = (1, 2))
    X2 = CrossSection()
    X2.add(width = heater_w, offset = 0, layer = heater_layer, name = 'heater', ports = (1, 2))

    Xtrans = pp.transition(cross_section1 = X1,
                        cross_section2 = X2,
                        width_type = 'sine')
    

    RingHeater = Device()
    #taper
    T1 = P1.extrude(Xtrans)
    t1 = RingHeater<<T1
    #heater half-arc
    T2 = P2.extrude(X2)
    t2 = RingHeater<<T2
    t2.connect(1, destination=T1.ports[2])
    RingHeater.add_port(name = "in", port = t1.ports[1])
    RingHeater.add_port(name = "ref", port = t2.ports[1])

    # RingHeater.rotate(heater_open_angle/2)
    RingHeater.rotate(RingHeater.ports["in"].orientation-90)

    #heater second half
    RingHeater2 = pg.copy(RingHeater)
    RingHeater2.mirror((0, RingHeater.ymax), (RingHeater.xmax, RingHeater.ymax))
    # RingHeater2.add_port(name=2, port = RingHeater2.ports[2])

    r2 = RingHeater<<RingHeater2
    RingHeater.add_port(name = "out", port = r2.ports["in"])

    
    RingHeater.add_port(name = "c", midpoint = (RingHeater.ports["ref"].x-heater_R*np.sin((heater_open_angle/2+taper_angle)*np.pi/180) , RingHeater.y))
    RingHeater.move(origin = RingHeater.ports["c"].midpoint, destination = (0,0))

    RingHeater.add_port(name = 1, port = RingHeater.ports["in"])
    RingHeater.add_port(name = 2, port = RingHeater.ports["out"])

    RingHeater.remove(RingHeater.ports["in"])
    RingHeater.remove(RingHeater.ports["out"])

    return RingHeater.flatten()
