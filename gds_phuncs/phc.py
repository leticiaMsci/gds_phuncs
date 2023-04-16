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



def corrugation_piece(width, length, layer=0):
    '''
    returns piece of corrugated waveguide.
    '''
    X = CrossSection()
    X.add(width = width, ports = (1, 2), layer=layer)
    
    P = pp.straight(length = length)
    Corr = P.extrude(X)
    
    return Corr


def corrugated_wg(len_lst, w_lst, layer=0):
    '''
    returns a corrugated waveguide with corrugations in order with length as len_lst and widths as w_lst
    '''
    N = len(len_lst)

    D = Device()

    WG0 = corrugation_piece(width = w_lst[0], length=len_lst[0], layer=layer)
    wg0 = D<<WG0

    port1 = wg0.ports[1]
    port2 = wg0.ports[2]

    for ii in range(1,N):
        WGi = corrugation_piece(width = w_lst[ii], length=len_lst[ii], layer=layer)
        wgi = D.add_ref(WGi)
        wgi.connect(port=1, destination = port2)
        port2 = wgi.ports[2]
    D.add_port(name = 1, port = port1)
    D.add_port(name = 2, port = port2)
    D.flatten()
    # qp(D)
    return(D)