'''module for writing mzis'''
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

def euler_imbalancers(waveguide_width=1.2,
                      waveguide_pitch=150.0,
                      radius=200.0,
                      imbalance_length=200.0,
                      heater_width = 5,
                      waveguide_layer=2,
                      heater_layer=7,
                      metal_layer = 4,
                      num_pts=720,
                      add_heater=False):

    D = Device('euler_imbalancers')

    CL = pp.euler(radius=radius, angle=90, use_eff=True, num_pts=num_pts)
    CR = pp.euler(radius=radius, angle=-90, use_eff=True, num_pts=num_pts)
    # UR = pp.euler(radius=radius, angle=180, use_eff=True, num_pts=num_pts)
    # UL = pp.euler(radius=radius, angle=-180, use_eff=True, num_pts=num_pts)
    S = pp.straight(length=0.5*imbalance_length)

    PL = Path()
    # PL.append([CR, S, UR, S, CR])
    PL.append([CR, S, CL, CL, S, CR])

    PR = Path()
    # PR.append([CL, UL, CL])
    PR.append([CL, CR, CR, CL])
    PR.y += waveguide_pitch

    X = CrossSection()
    X.add(width=waveguide_width, offset=0, layer=waveguide_layer, ports=(1, 2))

    WL = PL.extrude(width=X)
    WR = PR.extrude(width=X)

    D['LongPath'] = D << WL
    D['ShortPath'] = D << WR
    D.add_port(name='in1', port=D['LongPath'].ports[1])
    D.add_port(name='out1', port=D['LongPath'].ports[2])
    D.add_port(name='in2', port=D['ShortPath'].ports[1])
    D.add_port(name='out2', port=D['ShortPath'].ports[2])

    #Add heater
    if add_heater:
      PHeater = Path()
      PHeater.append([S,CL, CL, S])
      X = CrossSection()
      X.add(width=heater_width, offset=0, layer=heater_layer, ports=(1,2))
      Heater = PHeater.extrude(width=X)
      heater = D.add_ref(Heater)
      heater.rotate(-90)
      heater.ymin = D['LongPath'].ymin + waveguide_width/2 - heater_width/2
      heater.x = D['LongPath'].x
      # add heater pads
      pad1 = D.add_ref(pg.rectangle(size=(100, 300), layer=metal_layer))
      pad1.x = heater.xmin
      pad1.ymin = heater.ymax-25

      pad2 = D.add_ref(pg.rectangle(size=(100, 100), layer=metal_layer))
      pad2.x = heater.xmax
      pad2.ymin = heater.ymax-25

      D.add_port(name='heater1', midpoint=(pad1.xmax, pad1.ymax-50), width=100, orientation=0)
      D.add_port(name='heater2', midpoint=(pad2.xmax, pad2.ymax-50), width=100, orientation=0)
    # qp(D)
    # plt.show()


    return D.flatten()

def eo_gsg(electrode_length=5e3,
          electrode_gap=3.0,
          boe_gap=11,
          electrode_gnd_width=150.0,
          waveguide_pitch=150.0,
          waveguide_width=1.2,
          waveguide_layer=2,
          electrode_layer=4,
          boe_layer = 5
          ):

    D = Device('eo_gsg')
    P=Path()
    P.append(pp.straight(length=electrode_length, num_pts=2))
    X_wg = CrossSection().add(width=waveguide_width, offset=0, ports=(1,2), layer=waveguide_layer)
    WG = P.extrude(width=X_wg)
    wg1 = D.add_ref(WG)
    wg2 = D.add_ref(WG)
    wg2.y = wg1.y+waveguide_pitch
    X_gnd =  CrossSection().add(width=electrode_gnd_width, ports=(1,2), layer=electrode_layer)
    X_gnd.add(width=electrode_gnd_width-(boe_gap-electrode_gap)/2, offset=+(boe_gap-electrode_gap)/4, layer=boe_layer)
    Gnd_metal = P.extrude(width=X_gnd)
    gmetal1 = D.add_ref(Gnd_metal)
    gmetal1.ymax = wg1.y-electrode_gap/2
    gmetal2 = D.add_ref(Gnd_metal)
    gmetal2.rotate(180).mirror()
    gmetal2.ymin = wg2.y+electrode_gap/2
    X_sig = CrossSection().add(width=waveguide_pitch-electrode_gap, ports=(1,2), layer=electrode_layer)
    X_sig.add(width=waveguide_pitch-boe_gap, layer=boe_layer)
    sig = D.add_ref(P.extrude(width=X_sig))
    sig.ymin = wg1.y+electrode_gap/2
    # qp(D)
    # plt.show()



    D.add_port(name='in1', port=wg1.ports[1])
    D.add_port(name='out1', port=wg1.ports[2])
    D.add_port(name='in2', port=wg2.ports[1])
    D.add_port(name='out2', port=wg2.ports[2])
    D.add_port(name='g_top', port=gmetal2.ports[1])
    D.add_port(name='sig', port=sig.ports[1])
    D.add_port(name='g_bot', port=gmetal1.ports[1])

    return D



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



def lnlf_amod_v1(electrode_length=10e3,
                 electrode_gap=3.0,
                 boe_gap=11.0,
                 imbalance_length=200.0,
                 waveguide_pitch=150.0,
                 waveguide_width=1.2,
                 bend_radius = 200.0,
                 waveguide_layer=1,
                 electrode_layer=4,
                 boe_layer=5,
                 heater_layer=7,
                 add_heater=False
                 ):
    dev_name = 'lnlf_amod_v1'
    D = Device(dev_name)

    unbalance_segment = euler_imbalancers(waveguide_width=waveguide_width,
                                          waveguide_pitch=waveguide_pitch,
                                          radius=bend_radius,
                                          imbalance_length=imbalance_length,
                                          waveguide_layer=waveguide_layer)

    branch_length = 500.0
    branch_taper_length = 200.0
    ybranch_segment_in = ybranch_sine(waveguide_pitch=waveguide_pitch,
                                      waveguide_width=waveguide_width,
                                      taper_length=branch_taper_length,
                                      branch_length=branch_length,
                                      waveguide_layer=waveguide_layer
                                      )

    ybranch_segment_out = ybranch_sine(waveguide_pitch=waveguide_pitch,
                                       waveguide_width=waveguide_width,
                                       taper_length=branch_taper_length,
                                       branch_length=branch_length,
                                       waveguide_layer=waveguide_layer
                                       )

    electrode_segment = eo_gsg(electrode_length=electrode_length,
                                              electrode_gap=electrode_gap,
                                              boe_gap=boe_gap,
                                              electrode_gnd_width=waveguide_pitch,
                                              waveguide_pitch=waveguide_pitch,
                                              waveguide_width=waveguide_width,
                                              waveguide_layer=waveguide_layer,
                                              electrode_layer=electrode_layer)


    # electrode_segment.add_ref(pg.copy_layer(electrode_segment_boe, layer=electrode_layer, new_layer=boe_layer))


    D['Unbalance'] = D << unbalance_segment
    D['ElectrodeAC'] = D << electrode_segment
    D['YbranchIn'] = D << ybranch_segment_in
    D['YbranchOut'] = D << ybranch_segment_out


    D['ElectrodeAC'].connect(port='in1', destination=D['YbranchIn'].ports['out1'])
    D['Unbalance'].connect(port='in1', destination=D['ElectrodeAC'].ports['out1'])
    D['YbranchOut'].connect(port='out2', destination=D['Unbalance'].ports['out1'])

    D.add_port(name=1,port=D['YbranchIn'].ports['in'])
    D.add_port(name=2,port=D['YbranchOut'].ports['in'])
    D.add_port(name='g_top', port=D['ElectrodeAC'].ports['g_top'])
    D.add_port(name='g_bot', port=D['ElectrodeAC'].ports['g_bot'])
    D.add_port(name='sig', port=D['ElectrodeAC'].ports['sig'])
    if add_heater:
      D.add_port(name='heater1', port=D['Unbalance'].ports['heater1'])
      D.add_port(name='heater2', port=D['Unbalance'].ports['heater2'])

    return D
