# -*- coding: utf-8 -*-
"""
Created on Mon Apr 11 16:28:13 2016

@author: slauniai
"""

import numpy as np
from soil_core import wrc
from soil_core import soilRH
from soil_core import e_sat

eps = np.finfo(float).eps  # machine epsilon

#: [m s\ :sup: `-2`\ ], standard gravity
GRAVITY = 9.81
#: [kg m\ :sup:`-3`\ ], ice density
ICE_DENSITY = 917.0
#: [K], normal temperature
NORMAL_TEMPERATURE = 273.15
#: [], latent heat of freezing
LATENT_HEAT_FREEZING = 333700.0
#: [], freezing point of water
FREEZING_POINT_H2O = 0.0
#: [J mol\ :sup:`-1`\ ], universal gas constant
GAS_CONSTANT = 8.314
#: [kg mol\ :sup:`-1`\ ], molar mass of H\ :sub:`2`\ O
MOLAR_MASS_H2O = 18.015e-3


def heatFlow1D(t_final, z, pF, T0, Wliq0, Wice0, ubc, lbc, spara, S=0.0, steps=10):
    """ Solves soil heat flow in 1-D using implicit,
    backward finite difference solution of heat equation (Hansson et al., 2004;
    Saito et al., 2006):

    Backward finite difference solution is as described by Van Dam and Feddes (2000)
    for solving Richars equation.

    .. math::
        \\frac{\\partial C_p T}{\\partial t} =
        \\frac{\\partial}{\\partial z}
        \\left[\\lambda(\\theta)\\frac{\\partial T}{\\partial z}\\right] +
        C_w\\frac{\\partial q T}{\\partial z}

    where :math:`C_p` is volumetric heat capacity of soil, :math:`C_w` is volumetric
    heat capacity of liquid water, :math:`\\lambda(z)` is heat conductivity in soil, and
    :math:`q(z)` is the liquid water flow.

    Reference:
        Hansson et al. (2004)
            Vadose Zone Journal 3:693-704
        Saito et al. (2006)
            Vadose Zone Journal 5:784-800

    Args:
        t_final: solution timestep [s]
        z: grid,<0, monotonically decreasing [m]
        pF (dict): van Genuchten soil water retension (pF) parameters
            * 'ThetaR'
            * 'ThetaS'
            * 'n'
            * 'alpha'
        T0: initial temperature profile [degC]
        Wliq0 - total water content [m3m-3]
        ubc: upper bc: {'type': (give str 'flux','temperature'), 'value': ubc value}.
            Downward flux <0 (i.e. surface heating / bottom cooling)
        lbc: lower bc, formulate as ubc
        spara: soil type parameter dict with necessary fields for
            * 'cs':
                soil parent material vol. heat capacity [Jm\ :sup:`-3` K\ :sup:`-1`\ ]
            * 'Ktherm':
                soil parent heat conductivity [Js\ :sup:`-1` m\ :sup:`-1` K\ :sup:`-1`]
            * 'frp':
                freezing curve shape parameter
                    * 2...4 for clay soils
                    * 0.5...1.5 for sandy soils

            NOTE:
            if cs and Ktherm are not given, then followign may be needed
                * 'vMineral': mineral fraction
                * 'vClay': clay fraction
                * 'vQuartz': quartz fraction
                * 'vOrg': organic fraction

        Wice0:
            ice content [m3m-3]
        S:
            local heat sink/source array [Wm-3 =  Js-1m3], <0 for sink
        steps:
            initial subtimesteps used to proceed to 't_final'

    Returns:
        T: new temperature profile [m]
        Wliq: new liquid water content [m3m-3]
        Wice: new ice ontent [m3m-3]
        Fheat: heat flux array [Wm-2]

    CODE:
        Samuli Launiainen, Luke 19.4.2016. Converted from Matlab (APES SoilProfile.Soil_HeatFlow)
    NOTE:
        (19.4.2016): Tested ok. For solution of 'coupled' water&heatflow solve heatflow and
            waterflow sequentially for small timesteps
    TODO:
        1. think how heat advection with infiltration should be accounted for
        2. gradient boundary should be implemented as flux-based
    """

    Conv_crit1 = 1.0e-3  # degC
    Conv_crit2 = 1e-4  # wliq [m3m-3]
    # heat sink/source to 1st node due infiltration or evaporation
    # LH=-cw*infil_rate *T_infil/ dz[0] [Wm-3]
    LH = 0.0

    #    ca=1297.0 #specific heat capacity of air (J/kg/K)
    #    cw=4.18e6 #specific heat capacity of water (J/kg/K)
    #    ci=1.9e6 #specific heat capacity of ice (J/kg/K)

    # Set computation grid properties
    N = len(z)  # nr of nodal points, 0 is top
    dz, dzu, dzl = np.empty(N)

    # distances between grid points: dzu is between point i-1 and i, dzl between point i and i+1
    dzu[1:] = z[0:-1] - z[1:N]
    dzu[0] = -z[0]
    dzl[0:-1] = z[0:-1] - z[1:]
    dzl[-1] = (z[-2] - z[-1]) / 2.0

    dz = (dzu + dzl) / 2.0
    dz[0] = dzu[0] + dzl[0] / 2.0
    print dz, dzu, dzl

    # get parameters correct
    if 'frp' in spara:
        fp = spara['frp']
    else:
        # use default; 2...4 for clay and 0.5-1.5 for sandy soils (Femma-code/Harri Koivusalo)
        fp = 2.0

    poros = pF['ThetaS']
    if type(poros) is float:
        poros = np.empty(N) + poros

    if 'cs' not in spara or spara['cs'] is None:
        # soil parent vol. heat capacity (J/(m3*K), Campbell & Norman 1998
        cs = (2.2e6 * (spara['vMineral'] + spara['vClay']) + 2.5e6 * spara['vOrg']) / (1.0 - poros)
    else:
        cs = spara['cs']

    if 'Ktherm' in spara is False or spara['Ktherm'] is None:
        # compute from given composition
        # Wm-1K-1, weighted by volume fractions (sum(v...)=1.0)
        ks = 2.9 * spara['vMineral'] + 8.8 * spara['vQuartz'] + 0.25 * spara['vOrg']
    else:
        ks = spara['Ktherm']

    if type(S) is float:
        S = np.zeros(N) + S  # sink-source array

    # find solution at t_final

    # initial conditions
    T = T0
    Wliq = Wliq0
    Wice = Wice0
    # hydraulic head [m]
    h0 = wrc(pF, x=Wliq+Wice, var='Th')
    # dWtot/dh
    C = diffCapa(pF, h0)
    # initial time step [s]
    dt = t_final / steps
    # running time
    t = 0.0

    # loop until solution timestep
    while t < t_final:

        # these will stay constant during iteration over time step "dt"
        T_old = T
        # Wliq_old=Wliq;
        Wice_old = Wice
        # vol. heat capacity [Jm-3K-1]
        CP_old = volHeatCapa(poros, Wliq, wice=Wice, cs=cs)

#        R=Cond_deVries(poros, Wliq, wice=Wice, h=h0, pF=pF, T=T, ks=ks)
#        R=spatialAverage(R,method='arithmetic')

        # these change during iteration
        T_iter = T.copy()
        Wliq_iter = Wliq.copy()
        Wice_iter = Wice.copy()

        err1 = 999.0
        err2 = 999.0
        iterNo = 0

        # start iterative solution of heat equation
        while err1 > Conv_crit1 or err2 > Conv_crit2:  # and pass_flag is False:
            # print 'err1=' +str(err1) +'   err2=' + str(err2)
            iterNo += 1
            # volumetric heat capacity [Jm-3K-1]
            CP = volHeatCapa(poros, Wliq_iter, wice=Wice_iter, cs=cs)
            # thermal conductivity
            R = thermalCond_deVries(poros, Wliq, wice=Wice, h=h0, pF=pF, T=T, ks=ks)
            R = spatialAverage(R, method='arithmetic')
            # additional term due to phase-changes (liq <--> solid) (J m-3 K-1)
            A = (LATENT_HEAT_FREEZING ** 2.0
                 * ICE_DENSITY
                 / (GRAVITY * T_iter + NORMAL_TEMPERATURE) * C)

            A[T_iter > FREEZING_POINT_H2O] = 0.0

            # set up tridiagonal matrix
            a, b, g, f = np.zeros(N)

            # intermediate nodes
            b[1:-1] = CP[1:-1] + A[1:-1] + dt/dz[1:-1] * (R[1:-1]/dzu[1:-1] + R[2:]/dzl[1:-1])
            a[1:-1] = - dt / (dz[1:-1] * dzu[1:-1]) * R[1:-1]
            g[1:-1] = - dt / (dz[1:-1]*dzl[1:-1]) * R[2:]
            f[1:-1] = (
                CP_old[1:-1] * T_old[1:-1]
                + A[1:-1] * T_iter[1:-1]
                + LATENT_HEAT_FREEZING * ICE_DENSITY * (Wice_iter[1:-1] - Wice_old[1:-1])
                - dt*S[1:-1])

            # top node (n=0)
            # LH is heat input by infiltration or loss by evaporation not currently implemented

            if ubc['type'] is 'flux':  # or ubc['type'] is 'grad':
                F_sur = ubc['value']
                b[0] = CP[0] + A[0] + dt / (dz[0] * dzl[0]) * R[1]
                a[0] = 0.0
                g[0] = -dt / (dz[0] * dzl[0]) * R[1]
                f[0] = (
                    CP_old[0] * T_old[0]
                    + A[0] * T_iter[0]
                    + LATENT_HEAT_FREEZING * ICE_DENSITY * (Wice_iter[0] - Wice_old[0])
                    - dt/dz[0] * F_sur
                    - dt*LH - dt*S[0])

            # fixed T at imaginary node at surface
            if ubc['type'] is 'temperature':
                T_sur = ubc['value']
                b[0] = CP[0] + A[0] + dt / dz[0] * (R[0]/dzu[0] + R[1]/dzl[0])
                a[0] = 0.0
                g[0] = -dt / (dz[0]*dzl[0]) * R[1]
                f[0] = (
                    CP_old[0]*T_old[0]
                    + A[0]*T_iter[0]
                    + LATENT_HEAT_FREEZING * ICE_DENSITY * (Wice_iter[0] - Wice_old[0])
                    + dt/(dz[0] * dzu[0]) * R[0] * T_sur
                    - dt * LH
                    - dt * S[0])

            # bottom node (n=N)
            if lbc['type'] is 'flux':  # or lbc['type'] is 'grad':
                F_bot = lbc['value']
                b[-1] = CP[-1] + A[-1] + dt / (dz[-1]*dzu[-1]) * R[-2]
                a[-1] = -dt / (dz[-1]*dzu[-1]) * R[-2]
                g[-1] = 0.0
                f[-1] = (
                    CP_old[-1]*T_old[-1]
                    + A[-1]*T_iter[-1]
                    + LATENT_HEAT_FREEZING * ICE_DENSITY * (Wice_iter[-1] - Wice_old[-1])
                    - dt / dz[-1] * F_bot
                    - dt * S[-1])

            # fixed temperature, Tbot "at node N+1"
            if lbc['type'] is 'temperature':
                T_bot = lbc['value']
                b[-1] = CP[-1] + A[-1] + dt / dz[-1] * (R[-2]/dzu[-1] + R[-1]/dzl[-1])
                a[-1] = -dt / (dz[-1]*dzu[-1]) * R[-2]
                g[-1] = 0.0
                f[-1] = (
                    CP_old[-1] * T_old[-1]
                    + A[-1]*T_iter[-1]
                    + LATENT_HEAT_FREEZING * ICE_DENSITY * (Wice_iter[-1] - Wice_old[-1])
                    + dt / (dz[-1] * dzl[-1]) * R[-1] * T_bot - dt*S[-1])

            # save old and update iteration values
            T_iterold = T_iter.copy()
            Wliq_iterold = Wliq_iter.copy()
            # Wice_iterold=Wice_iter.copy()

            T_iter = thomas(a, b, g, f)
            Wliq_iter, Wice_iter = frozenWater(T_iter, Wliq_iter+Wice_iter, fp=fp)

            if iterNo is 7:
                # re-try with smaller timestep
                dt = dt / 3.0
                iterNo = 0
                continue

            elif any(np.isnan(T_iter)):
                # stop iteration
                print 'nan found'
                break

            err1 = np.max(abs(T_iter-T_iterold))
            err2 = np.max(abs(Wliq_iter - Wliq_iterold))

        # ending iteration loop
        # print 'Pass: t = ' + str(t), ' dt = ' + str(dt), ' iterNo = ' + str(iterNo)

        # update state tp t
        T = T_iter.copy()
        Wliq = Wliq_iter.copy()
        Wice = Wice_iter.copy()

        t += dt  # solution time & new initial timestep

        if iterNo < 2:
            dt = dt * 1.25
        elif iterNo > 4:
            dt = dt / 1.25

        dt = min(dt, t_final - t)
        # print 'new dt= ' +str(dt)

    # ending while loop, compute heat flux profile
    # [W/m2]
    Fheat = nodalFluxes(z, T, R)

    return T, Wliq, Wice, Fheat


def heatFlow1D_Simple(t_final, z, T0, Wtot, ubc, lbc, spara, S=0.0, steps=10):
    """ Solves soil heat flow in 1-D using implicit,
    backward finite difference solution of heat equation.

    Reference:

    Args:
        t_final: solution timestep [s]
        z: grid,<0, monotonically decreasing [m]
        pF (dict):
            vanGenuchten soil water retension (pF) parameters
            * 'ThetaR'
            * 'ThetaS'
            * 'n'
            * 'alpha'
        T0: initial temperature profile [degC]
        Wliq0: total water content [m3m-3]
        ubc (dict): upper bc
            * 'type': (give str 'flux',temperature')
            * 'value': ubc value. Downward flux <0. (i.e. surface heating / bottom cooling)
        lbc: lower bc, formulate as ubc
        spara: soil type parameter dict with necessary fields for i) soil heat capacity,
            thermal conductivity, freezing curve:

            * 'cs': give soil parent material vol. heat capacity [Jm-3K-1]. If given used instead of
                computing from other inputs
            * 'Ktherm': soil parent heat conductivity [Js-1m-1-K-1]
            * 'poros' : soil porosity [m3m-3]
            * 'frp': freezing curve shape parameter [2...4 for clay, 0.5...1.5 for sandy soils]

        S - local heat sink/source array [Wm-3 =  Js-1m3], <0 for sink
        steps - initial subtimesteps used to proceed to 't_final'

    Returns:
        T: new temperature profile [m]
        Wliq: new liquid water content [m3m-3]
        Wice: new ice ontent [m3m-3]
        Fheat: heat flux array [Wm-2]

    CODE:
        Samuli Launiainen, Luke 19.4.2016. Converted from Matlab (APES SoilProfile.Soil_HeatFlow)

    NOTE:
        (19.4.2016): DOES NOT WORK; JÄI TÄMÄKIN KESKEN!!!!
    """
    # degC
    Conv_crit1 = 1.0e-3
    # grav=9.81 #acceleration due gravity, kgm/s2
    rhoi = 917.0  # ice density, kg/m3
    Lf = 333700.0  # latent heat of freezing, J/kg; % Lv is latent heat of vaportization
    # Tfr=0.0 #freezing point of water (degC)
    # rhoa=1.25 # density of air (kg/m3)

    #    ca=1297.0 #specific heat capacity of air (J/kg/K)
    #    cw=4.18e6 #specific heat capacity of water (J/kg/K)
    #    ci=1.9e6 #specific heat capacity of ice (J/kg/K)

    # Set computation grid properties
    N = len(z)  # nr of nodal points, 0 is top
    dz = np.empty(N)
    dzu = np.empty(N)
    dzl = np.empty(N)

    # distances between grid points: dzu is between point i-1 and i, dzl between point i and i+1
    dzu[1:] = z[0:-1] - z[1:N]
    dzu[0] = -z[0]
    dzl[0:-1] = z[0:-1] - z[1:]
    dzl[-1] = (z[-2] - z[-1]) / 2.0

    dz = (dzu + dzl) / 2.0
    dz[0] = dzu[0] + dzl[0] / 2.0
    # print dz, dzu, dzl

    # get parameters correct
    poros = spara['poros']
    # if type(poros) is float: poros=np.empty(N)+poros
    fp = spara['frp']
    # if type(fp) is float: fp=np.empty(N)+fp
    cs = spara['cs']

    ks = spara['Ktherm']

    if type(S) is float:
        S = np.zeros(N) + S  # sink-source array

    # find solution at t_final

    # initial conditions
    T = T0
    Wliq, Wice = frozenWater(T, Wtot, fp=fp)

    # h0=wrc(pF,x=Wliq+Wice,var='Th') #hydraulic head [m]
    # C=diffCapa(pF, h0) #dWtot/dh

    dt = t_final / steps  # initial time step [s]
    t = 0.0  # running time

    while t < t_final:  # loop until solution timestep

        # these will stay constant during iteration over time step "dt"
        T_old = T
        Wice_old = Wice

        CP_old = volHeatCapa(poros, Wliq, wice=Wice, cs=cs)  # vol. heat capacity [Jm-3K-1]

#        R=thermalCond_deVries(poros, Wliq, wice=Wice, T=T, ks=ks)
#        R=spatialAverage(R,method='arithmetic')

        # print 'R=' + str(R)
        # print 'Wl=' +str(Wliq), 'Wi=' + str(Wice)
        # these change during iteration
        T_iter = T.copy()
        Wliq_iter = Wliq.copy()
        Wice_iter = Wice.copy()

        err1 = 999.0
        iterNo = 0

        # start iterative solution of heat equation

        while err1 > Conv_crit1:  # and pass_flag is False:
            # print 'err1=' +str(err1) +'   err2=' + str(err2)
            iterNo += 1
            # vol. heat capacity [Jm-3K-1]
            CP = volHeatCapa(poros, Wliq_iter, wice=Wice_iter, cs=cs)

            # additional term due to phase-changes
            # A=rhoi*Lf*dWice/dT [J m-3 K-1]
            gam = (Wice_iter - Wice_old) / (T_iter - T_old + 2*eps)  # dWice/dT
            A = rhoi*Lf*gam
            # print T_iter
            A[T_iter > 0.0] = 0.0
            # print 'A=' + str(A)
            # thermal conductivity
            R = thermalCond_deVries(poros, Wliq_iter, wice=Wice_iter, T=T_iter, ks=ks)
            R = spatialAverage(R, method='arithmetic')

            # set up tridiagonal matrix
            a, b, g, f = np.zeros(N)

            # intermediate nodes
            b[1:-1] = CP[1:-1] + A[1:-1] + dt/dz[1:-1] * (R[1:-1]/dzu[1:-1] + R[2:]/dzl[1:-1])
            a[1:-1] = - dt / (dz[1:-1] * dzu[1:-1]) * R[1:-1]
            g[1:-1] = - dt / (dz[1:-1] * dzl[1:-1]) * R[2:]
            f[1:-1] = (
                CP_old[1:-1]*T_old[1:-1]
                + A[1:-1] * T_iter[1:-1]
                + Lf*rhoi * (Wice_iter[1:-1] - Wice_old[1:-1])
                - dt*S[1:-1])

            # top node (n=0)
            # LH is heat input by infiltration/ loss by evaporation not currently implemented

            if ubc['type'] is 'flux':  # or ubc['type'] is 'grad':
                F_sur = ubc['value']
                b[0] = CP[0] + A[0] + dt / (dz[0] * dzl[0]) * R[1]
                a[0] = 0.0
                g[0] = -dt / (dz[0] * dzl[0]) * R[1]
                f[0] = (
                    CP_old[0] * T_old[0]
                    + A[0] * T_iter[0]
                    + Lf * rhoi * (Wice_iter[0] - Wice_old[0])
                    - dt/dz[0]*F_sur
                    - dt*S[0])

            if ubc['type'] is 'temperature':   # fixed T at imaginary node at surface
                T_sur = ubc['value']
                b[0] = CP[0] + A[0] + dt / dz[0] * (R[0]/dzu[0] + R[1]/dzl[0])
                a[0] = 0.0
                g[0] = -dt / (dz[0] * dzl[0]) * R[1]
                f[0] = (
                    CP_old[0] * T_old[0]
                    + A[0] * T_iter[0]
                    + Lf * rhoi * (Wice_iter[0] - Wice_old[0])
                    + dt / (dz[0] * dzu[0]) * R[0] * T_sur
                    - dt*S[0])

            # bottom node (n=N)
            if lbc['type'] is 'flux':  # or lbc['type'] is 'grad':
                F_bot = lbc['value']
                b[-1] = CP[-1] + A[-1] + dt / (dz[-1] * dzu[-1]) * R[-2]
                a[-1] = -dt / (dz[-1] * dzu[-1]) * R[-2]
                g[-1] = 0.0
                f[-1] = (
                    CP_old[-1] * T_old[-1]
                    + A[-1] * T_iter[-1]
                    + Lf * rhoi * (Wice_iter[-1] - Wice_old[-1])
                    - dt / dz[-1] * F_bot
                    - dt*S[-1])

            if lbc['type'] is 'temperature':  # fixed temperature, Tbot "at node N+1"
                T_bot = lbc['value']
                b[-1] = CP[-1] + A[-1] + dt / dz[-1] * (R[-2]/dzu[-1] + R[-1]/dzl[-1])
                a[-1] = -dt / (dz[-1] * dzu[-1]) * R[-2]
                g[-1] = 0.0
                f[-1] = (
                    CP_old[-1] * T_old[-1]
                    + A[-1] * T_iter[-1]
                    + Lf * rhoi * (Wice_iter[-1] - Wice_old[-1])
                    + dt / (dz[-1] * dzl[-1]) * R[-1] * T_bot
                    - dt * S[-1])

            # save old and update iteration values
            T_iterold = T_iter.copy()
            # Wliq_iterold=Wliq_iter.copy()
            # Wice_iterold=Wice_iter.copy()

            T_iter = thomas(a, b, g, f)
            Wliq_iter, Wice_iter = frozenWater(T_iter, Wliq_iter+Wice_iter, fp=fp)

            if iterNo == 7:
                dt = dt / 3.0
                iterNo = 0
                continue  # re-try with smaller timestep
            elif any(np.isnan(T_iter)):
                print 'nan found'
                break  # break while loop

            err1 = np.max(abs(T_iter-T_iterold))
            print 'err1=' + str(err1), ' Tit=' + str(T_iter)
            # print 'Wiceit=' + str(Wice_iter)
            # err2=np.max(abs(Wliq_iter - Wliq_iterold))

        # ending iteration loop
        # update state tp t
        T = T_iter.copy()
        Wliq = Wliq_iter.copy()
        Wice = Wice_iter.copy()

        # solution time & new initial timestep
        t += dt

        if iterNo < 2:
            dt = dt * 1.25
        elif iterNo > 4:
            dt = dt / 1.25

        dto = dt
        dt = min(dt, t_final - t)

        print('Pass: t = ' + str(t)
              + ' dt = ' + str(dto)
              + ' iterNo = ' + str(iterNo)
              + ' new dt= ' + str(dt))

    # ending while loop, compute heat flux profile
    # [W/m2]
    Fheat = nodalFluxes(z, T, R)

    z_frost = 0.0
    z_thaw = 0.0
    if np.any(T < 0.0):
        z_frost = max(z[T < 0])
    if z_frost < 0.0 and np.any(T > 0.0):
        # f=min(find_index(T, lambda x: x>0.0))
        z_frost = max(z[T > 0.0])
        if z_thaw < z_frost:
            z_thaw = 0.0

    return T, Wliq, Wice, Fheat, z_frost, z_thaw


def soilTemperature_Rankinen(t_final, z, To, T_sur, Ds, para, steps=10):
    """ simplified solution of soil temperature. Computes soil temperature profile at depths z
    by simple approximation given in Rankinen et al. (2004). HESS, 8, 706-716.

    Assumes:
        i) depth-constant soil heat capacity and thermal conductivity.
        ii) no-flow conditions below depth in consideration,
        iii) explicit solution

    NOTE:
        Not suited for computing realistic temperature profile evolution; use heatFlow1D instead...

    Args:
        t_final - solution time [s]
        z - grid [m]
        To - initial temperature profile [degC]
        T_sur - surface temperature (or air temperature) [degC]
        Ds - snow depth [m]
        para - dict with following parameters (Ranges from Table 3:)
            * 'Cs': soil heat capacity Jm-3K-1. Range 1.0e6...1.3e6 for mineral soils,
                xxx for organic soils
            * 'Cice': latent heat storage/release correction term, accounted for T<0.01degC.
                Range 4.1e6...9e6; larger for peat soils?
            * 'Ktherm': soil thermal conductivity Wm-1K-1,
                range 0.5...0.8 Wm-1K-1 (mineral), 0.3...0.45 Wm-1K-1 for peat soils
            * 'fs': damping parameter for snow depth impacts. Range 2.1 ...7.1
        steps - model subtimesteps

    Returns:
        Tnew - new soil temperature profile
    """

    # input params
    R = para['Ktherm']  # Wm-1K-1
    fs = para['fs']  # m-1, snow damping parameter
    Cice = para['Cice']  # apparent ice vol. heat capacity, range 4e6...15e6 Jm-3K-1
    Cs = para['Cs']  # soil heat capacity Jm-3K-1, range 1e6...1.3e6. LOWER FOR organic soil!

    dt = t_final/steps
    N = len(z)

    Told = To

    Tnew = np.empty(N)
    t = 0.0
    while t < t_final:
        for n in range(0, N):
            if Told[n] <= 0.01:
                Ca = Cs + Cice
            else:
                Ca = Cs

            Tnew[n] = Told[n] + dt * R / (Ca * (2*z[n])**2) * (T_sur - Told[n])
            Tnew[n] = np.exp(-fs * Ds) * Tnew[n]  # snow correction

        Told = Tnew
        t += dt

        # plt.plot(Told,z,'r-')
    return Tnew


# explicit scheme for heat equation in homogenous material
def heatFlow_Homogenous(t_final, z, To, k, S, ubcType, ubc, lbcType, lbc):
    """
        Solves 1D heat equation in homogenous soil using explicit Eulerian method;
        Forward difference in time, centered in space
        IN:
            t_final -solution time [s]
            z - grid [m], expects constant increments
            To - initial T profile
            k - thermal diffusivity (m2/s) : thermal diffusivity is L/(rho_s*c_s)
            S - heat sink/source
            ubcType - type of upper bc
            ubc - value of upper bc: 'dirchlet' for fixed value, 'neumann' for flux
            lbcType - type of lower bc
            lbc - value of lower bc
        OUT:
            U - new temperature profile

            Samuli Launiainen 14.4.2016
    """
    N = len(z)
    dz = abs(z[1] - z[0])

    dt = 0.4 * dz ** 2 / k
    # mesh fourier number
    R = k * dt / (dz ** 2)
    # print dt, R
    U = To.copy()

    t = 0.0
    while t < t_final:

        Uxo = U.copy()
        # upper bc
        if ubcType is 'dirchlet':
            U[0] = ubc
        else:
            U[0] = Uxo[0] + R * (2 * U[1] - 2 * U[0] - 2 * dz * ubc)
        # U[0]=ubc; U[-1]=lbc
        for m in range(1, N-1):
            U[m] = Uxo[m] + R * (U[m-1] - 2*U[m] + U[m+1]) + dt*S[m]
        # lower bc
        if lbcType is 'dirchlet':
            U[-1] = lbc
        else:
            U[-1] = Uxo[-1] + R * (2*U[-2] - 2*U[-1] + 2*dz*lbc)
        # print t, U
        t += dt
        # print 't= ', t
    return U


# Utility functions
def thomas(a, b, C, D):
    """
    Tridiagonal matrix algorithm of Thomas
    a=subdiag, b=diag, C=superdiag, D=rhs
    """
    n = len(a)
    V, G, U, x = np.zeros(n)

    V[0] = b[0].copy()
    G[0] = C[0] / V[0]
    U[0] = D[0] / V[0]

    for i in range(1, n):  # nr of nodes
        V[i] = b[i] - a[i] * G[i - 1]
        U[i] = (D[i] - a[i] * U[i - 1]) / V[i]
        G[i] = C[i] / V[i]

    x[-1] = U[-1]
    inn = n - 2
    for i in range(inn, -1, -1):
        x[i] = U[i] - G[i] * x[i + 1]

    return x


def diffCapa(pF, head):
    """ Analytic derivative of vGenuchten soil water retention curve [m-1]
    Args:
        pF-dict & head [m]
    Returns:
        x - dW/dhead [m-1]
    """
    # [cm]
    head = -100 * head
    ts = pF['ThetaS']
    tr = pF['ThetaR']
    n = pF['n']
    m = 1.0 - np.divide(1, n)
    alfa = pF['alpha']

    # print ts, tr, n, m, alfa
    x = 100.0 * (ts - tr) * (n - 1.0) * alfa**n * head**(n-1.0) / ((1 + (alfa*head)**n)**(m+1.0))
    x[head <= 0.0] = 0.0

    return x


def find_index(a, func):
    """ Finds indexes or array elements that fill the condition
    call as indices(a, lambda x: criteria)
    """
    return [i for (i, val) in enumerate(a) if func(val)]


def spatialAverage(y, x=None, method='arithmetic'):
    """ Calculates spatial average of quantity y
    Args:
        y (array):
        x (grid):
        method:
            * 'arithmetic'
            * 'geometric'
            * 'dist_weighted'
    Returns:
        f: averaged y
    """

    N = len(y)
    f = np.empty(N)
    if method is 'arithmetic':
        f[1:-1] = 0.5 * (y[0:-2] + y[1:-1])
        f[0] = y[0]
        f[-1] = y[-1]

    elif method is 'geometric':
        f[1:-1] = np.sqrt(y[1:-2]*y[2:])
        f[0] = y[0]
        f[-1] = y[-1]

    elif method is 'dist_weighted':
        a = (x[0:-2] - x[2:]) * y[:-2] * y[1:-1]
        b = y[1:-1] * (x[:-2] - x[1:-1]) + y[:-2]*(x[1:-1] - x[2:])

        f[1:-1] = a / b
        f[0] = y[0]
        f[-1] = y[-1]

    return f


def nodalFluxes(x, y, K):
    """ Calculates fluxes between nodal points in 1-D grid
    Args:
        x: grid, monotonic!
        y: vector of values
        K: conductivity
    Returns:
        f: flux array
    """
    f = np.empty(np.shape(y))
    yprim = centralDifference(x, y)
    # flux
    f = -K * yprim

    return f


def centralDifference(x, y):
    """ Derivative by central difference
    Args:
        x: grid
        y: value vector
    Returns:
        yprim: derivative
    """
    yprim = np.empty(np.shape(y))

    # central difference
    yprim[1:-1] = (y[2:] - y[0:-2]) / (x[2:] - x[0:-2])
    # forward difference at left boundary
    yprim[0] = (y[1] - y[0]) / (x[1] - x[0])
    # backward difference at right boundary
    yprim[-1] = (y[-1] - y[-2]) / (x[-1] - x[-2])

    return yprim


# Thermal conductivity functions and heat capacity, freezing curve etc.
def thermalCond_deVries(poros, wliq, wice=0.0, h=None, pF=None,
                        T=15.0, ks=None, vOrg=0.0, vQuartz=0.0):
    """ Thermal conductivity of soil
    Args:
        poros: porosity [m3m-3]
        wliq: liquid water content [m3m-3]
        wice: ice content [m3m-3]
        h: soil water tension [m]
        pF: vanGenuchten pF parameter dict
        T: temperature [degC]
        ks: soil parent material thermal conductivity
        vOrg: organic matter fraction (of total vol. of solids)
        vQuartz: quartz fraction (of total vol. of solids)
    """
    kw = 0.57  # thermal cond of water W/(mK) = J/(smK)
    ki = 2.2  # thermal cond of ice J/(smK)
    ka = 0.025  # thermal conductivity of air J/(smK)
    kq = 8.8  # thermal conductivity of Quarz  8.8 W/m/K
    km = 2.9  # thermal conductivity ofother minerals W/m/K
    ko = 0.25  # thermal conductivity of organic matter W/m/K
    Lmol = 44100.0  # latent heat of vaporization at 20 deg C [Jmol-1]
    Dv = 24e-6  # molecular diffusivity [m2 s-1] of water vapor in air at 20degC
    P = 101300.0  # Pa
    # molar concentration of air [m3 mol-1]
    cf = np.divide(P, 287.05 * (T + 273.15)) / 29e-3

    wliq = np.array(wliq, ndmin=1)
    wice = np.array(wice, ndmin=1)
    poros = np.array(poros, ndmin=1)

    N = len(wliq)
    wair = poros - wliq

    if ks is None:
        # vol. weighted solid conductivty [W m-1 K-1]
        ks = np.zeros(N) + (1-vQuartz - vOrg)*km + vQuartz*kq + vOrg*ko

    ga = np.ones(N) * 0.0125
    # following line rises VisualDepricationWarning:
    # boolean index did not match indexed array along dimension
    # ga[vOrg > 0.9 * poros] = 0.33  # organic soil
    # possible solution may look something like:
    ga[np.where(vOrg > 0.9 * poros)] = 0.33
    # corrections for latent heat flux & vapor phase conductivity
    if h is not None:
        rh = soilRH(T, h)
    else:
        rh = 1.0

    es, ss = e_sat(T)
    wo = 0.10
    # correction for latent heat flux
    fw = np.divide(1.0, 1 + (wliq/wo) ** -4.0)
    # vapor phase apparent thermal conductivity inc. Stefan correction (W m-1 K-1)
    kg = ka + Lmol * ss * rh * fw * cf * Dv / (P-es)
    # fluid thermal cond.
    kf = kg + fw * (kw - kg)
    del fw, es, ss, cf

    # weight factors [-]
    r = 2.0 / 3.0
    # gas phase
    gg = r * (1.0 + ga * (kg/kf - 1.0))**(-1) + (1-r) * (1.0 + (1.0 - 2.0*ga) * (kg/kf - 1.0))**(-1)
    # liquid phase
    gw = r * (1.0 + ga * (kw/kf - 1.0))**(-1) + (1-r) * (1.0 + (1.0 - 2.0*ga) * (kw/kf - 1.0))**(-1)
    # solid phase
    gs = r * (1.0 + ga * (ks/kf - 1.0))**(-1) + (1-r) * (1.0 + (1.0 - 2.0*ga) * (ks/kf - 1.0))**(-1)
    # W m-1 K-1
    L = (
        (wliq * kw * gw + kg * wair * gg + (1-poros) * ks * gs + wice * ki)
        / (wliq*gw + wair*gg + (1.0-poros)*gs + wice)
        )

    return L


def thermalCond_Campbell(poros, wliq, wice=0.0, vQuartz=0.0, vClay=0.0, vMineral=1.0):
    """ Campbell, 1995, extended by Hansson et al. 2004 for frozen conditions

        Note:
            c1-c5 are from Hydrus 1-D code
    """
    # W/(mK)
    c1 = 0.57 + 1.73 * vQuartz + 0.93 * vMineral / (1 - 0.74 * vQuartz)
    # W/(mK)
    c2 = 2.8 * (1 - poros)
    # [-]
    c3 = 1 + 2.6 * np.sqrt(max(0.005, vClay))
    # W/(mK)
    c4 = 0.03 + 0.7 * (1 - poros)
    # [-]
    c5 = 4.0

    #  Hansson et al, 2004 Kanagawa sandy loam
    #         c1=0.55; %W/(mK)
    #         c2=0.80; %W/(mK)
    #         c3=3.07; %[-]
    #         c4=0.13; %W/(mK)
    #         c5=4; %[-])

    # [-]
    f1 = 13.0
    # [-]
    f2 = 1.06
    # [-]
    F = 1.0 + f1 * wice ** f2
    # W/(mK)
    L = c1 + c2*(wliq + F*wice) - (c1-c4)*np.exp(-(c3*(wliq + F*wice))**c5)

    return L


def thermalCondSolid(vMineral=1.0, vQuartz=0.0, vOrg=0.0):
    """ Soil solids thermal conductivity from mineral composition
    """
    if 1.0 - (vMineral + vQuartz + vOrg) > eps:
        print 'heat_flow.thermalCondSolid: sum of vFract must be 1.0'
        ks = None
    else:
        # [Wm-1K-1], weighted by volume fractions (sum(v...)=1.0)
        ks = 2.9 * vMineral + 8.8 * vQuartz + 0.25 * vOrg
    return ks


def thermalCondSimple(poros, wliq, wice, ks=None, soilComp=None, method='vol_average'):
    """ Simple estimate for thermal conductivity in soil
    """
    # thermal cond of water W/(mK) = J/(smK)
    kw = 0.57
    # thermal cond of ice J/(smK)
    ki = 2.2
    # thermal conductivity of air J/(smK)
    ka = 0.025
    # thermal conductivity of Quarz  8.8 W/m/K
    kq = 8.8
    # thermal conductivity ofother minerals W/m/K
    km = 2.9
    # thermal conductivity of organic matter W/m/K
    ko = 0.25

    wair = poros - wliq - wice

    if method is 'stetson':
        # W/(mK)
        L = (1 - poros) * (0.25 * kq + 0.75 * km) + kw * wliq * ki * wice

    elif method is 'vol_average':
        # soil parent thermal cond. given as input
        if ks is not None:
            L = ks + kw*wliq + ki*wice + ka*wair
        elif soilComp is not None:
            ks = km*soilComp['vMineral'] + kq*soilComp['vQuartz'] + ko*soilComp['vOrg']
            L = ks + kw*wliq + ki*wice + ka*wair
    else:
        print('heat_flow.thermalCond: invalid inputs, returning None')

    return L


def volHeatCapa(poros, wliq, wice=0.0, cs=None, soilType='Sand'):
    """ Computes volumetric heat capacity of soil

    Args:
        poros: porosity [m3 m-3]
        wliq: vol. liquid water content [m3 m-3]
        wice: vol. ice content [m3 m-3]
        cs: solid constituent heat capacity [J m-3 K-1]
        soilType: 'Sand', 'Silt', 'Clay, 'Organic' to use table values for cs

    Returns:
        cp: volumetric heat capacity of soil [J m-3 K-1]
    """
    # volumetric heat capacity of air (J/m3/K)
    ca = 1297.0
    # volumetric heat capacity of water (J/m3/K)
    cw = 4.18e6
    # volumetric heat capacity of ice (J/m3/K)
    ci = 1.9e6
    # tabulated values for cs
    csoil = {
        'Sand': 0.0,
        'Silt': 0.0,
        'Clay': 0.0,
        'Organic': 0.0
        }

    if cs is None:
        cs = csoil[soilType]  # use table value
    wair = poros - wliq - wice

    cp = cs * (1.0 - poros) + cw * wliq + ci * wice + ca * wair

    return cp


def frozenWater(T, wtot, fp=2.0, To=0.0):
    """ Approximates ice content from soil temperature and total water content

    Args:
        T: soil temperature [degC]
        wtot: total vol. water content [m3 m-3]
        fp: parameter of freezing curve [-]
        To: freezing temperature of soil water [degC]

    Returns:
        wliq - vol. liquid water content [m3 m-3]
        wice - vol. ice content [m3 m-3]
    """
    wtot = np.array(wtot, ndmin=1)
    T = np.array(T, ndmin=1)

    wice = wtot * (1.0 - np.exp(-(To - T)/fp))
    wice[T > To] = 0.0
    wliq = wtot - wice

    return wliq, wice
