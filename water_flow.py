# -*- coding: utf-8 -*-
"""
Created on Wed Apr 06 08:16:37 2016


@author: slauniai
"""

import numpy as np
from soil_core import hydrCond
from soil_core import wrc

eps = np.finfo(float).eps  # machine epsilon


def waterFlow1D(t_final, z, h0, pF, Ksat, Ftop, R, HM=0.0,
                lbcType='impermeable', lbcValue=None, Wice0=0.0,
                maxPond=0.0, pond0=0.0, cosalfa=1.0, h_atm=-1000.0, steps=10):
    """ Solves soil water flow in 1-D using implicit, backward finite difference
    solution of Richard's equation. Reference: vanDam & Feddes (2000):
    Numerical simulation of infiltration, evaporation and shallow
    groundwater levels with the Richards equation, J.Hydrol 233, 72-85.

    Args:
        t_final: solution timestep [s]
        z: grid,<0, monotonically decreasing
        h0: initial hydraulic head [m]
        pF: dict of vanGenuchten soil pF-parameters; pF.keys()=['ThetaR', 'ThetaS', 'n', 'alpha']
        Ksat: saturated hydr. cond. [ms-1]
        Ftop: potential top boundary flux (<0 for infiltration)
        R: local sink/source array due root uptake & well [s-1], <0 for sink
        HM: net lateral flux array [s-1], <0 for net outflow
        lbcType: lower bc type: 'impermeable', 'flux', 'free_drain', 'head'
        lbcValue: lower bc value; give for 'head' and 'flux'
        Wice0: vol. ice content [m3m-3] - not needed now; could be used to scale hydr.conductivity
        maxPond: maximum allowed pond depth at surface [m]
        pond0: initial pond depth [m]
        cosalfa: 1 for vertical water flow, 0 for horizontal transport
        h_atm: hydraulic head [m] in equilibrium with air humidity
            used to compute soil evaporation supply
        steps: initial subtimesteps used to proceed to 't_final'

    OUT:
        h: new hydraulic head [m]
        W: new total water content [m3m-3]
        h_pond: new ponding depth [m]
        C_inf: total infiltration [m], <=0
        C_eva: total evaporation [m],>=0
        C_dra: total drainage from profile [m], <0 from profile
        C_roff: total surface runoff [m]
        Fliq: vertical water fluxes [ms-1] at t_final
        gwl: ground water level [m];
            if not in computational layer then assumed hydrostatic equilibrium with node -1
        mbe: total mass balance error [m]

    CODE:
        Samuli Launiainen, Luke 8.4.2016. Converted from Matlab (APES SoilProfile.WaterFlow)

    NOTE:
        (8.4.2016): upper bc restriction checks needs to be tested
        (   -"-  ): include macropore adjustment as in APES-code?
    """

    Conv_crit = 1.0e-5
    S = R + HM  # net imposed sink/source term (root uptake + lateral flow)

    # Get computation grid
    N = len(z)  # nr of nodal points, 0 is top

    dz, dzu, dzl = np.empty(N)

    # distances between grid points: dzu is between point i-1 and i, dzl between point i and i+1
    dzu[1:] = z[0:-1] - z[1:N]
    dzu[0] = -z[0]
    dzl[0:-1] = z[0:-1] - z[1:]
    dzl[-1] = (z[-2] - z[-1]) / 2.0

    dz = (dzu + dzl) / 2.0
    dz[0] = dzu[0] + dzl[0] / 2.0
    # print dz

    # soil variables and save intial conditions
    if type(Ksat) is float:
        Ksat = np.zeros(N) + Ksat

    poros = pF['ThetaS']
    # [m3/m3]
    W_ini = wrc(pF, x=h0)
    # h_ini = h0;
    pond_ini = pond0

    # these change during process
    W = W_ini
    h = h0
    h_pond = pond0

    # find solution at t_final
    # initial time step [s]
    dto = t_final / steps
    # running time
    t = 0.0

    # cumulative boundary fluxes
    C_inf, C_eva, C_dra, C_roff = 0.0
    # Rsink = sum(S * dz)

    # loop until solution timestep
    while t < t_final:
        # these will stay constant during iteration over time step "dt"
        h_old = h
        W_old = W

        # these change during iteration
        h_iter, h_iterold = h.copy()
        W_iter = W.copy()
        afp_iter = poros - W_iter

        err1, err2 = 999.0
        iterNo = 0
        dt = dto

        if np.any(h < -1e-7):
            Conv_crit2 = 1e-8
        else:
            Conv_crit2 = 1.0
        # Conv_crit2=1.0e-7

        # start iterative solution of Richards equation
        pass_flag = True
        while (err1 > Conv_crit or err2 > Conv_crit2):  # and pass_flag is False:
            # print pass_flag
            iterNo += 1
            # hydr conductivity m/s at dt; move inside iteration loop?
            KLh = hydrCond(pF, x=h_iter, Ksat=Ksat)
            KLh = spatialAverage(KLh, method='arithmetic')

            # lower bc check
            if lbcType is 'free_drain':
                q_bot = -KLh[-1] * cosalfa
                # if h_iter(N)<-1, q_bot=0; end % allow free drainage until field capacity
            elif lbcType is 'impermeable':
                q_bot = 0.0
            elif lbcType is 'flux':
                q_bot = max(lbcValue, -KLh[-1]*cosalfa)

            elif lbcType is 'head':
                h_bot = lbcValue
                # note, this is approximation for output
                q_bot = -KLh[-1] * (h_iter[-1] - h_bot)/dzl[-1] - KLh[-1]*cosalfa

            # upper bc check
            # note: Ftop<0 infiltration, Ftop>0 evaporation
            # potential rate m/s
            q0 = Ftop - h_pond/(dt + eps)
            # net flow to/from column
            Qin = (-q_bot - q0)*dt

            # case infiltration
            if q0 < 0:
                # print 'infil'
                # maximum inflow (m) tolerated by soil column
                Airvol = max(0.0, sum(afp_iter * dz))
                # initially saturated profile
                if Airvol <= eps:
                    # inflow exceeds outflow, matrix stays saturated
                    if Qin >= 0 and Airvol/sum(poros*dz) < 1.0e-3:
                        # cont_flag=False
                        print 'saturated matrix, routing to pond and runoff and breaking'
                        h_iter = h_old
                        # set bc's to zero flux
                        q_bot = 0.0
                        q_sur = 0.0
                        dt = t_final - t
                        gwl = z[1]
                        pass_flag = False

                    # outflow exceeds inflow
                    elif Qin <= 0:
                        q_sur = max(q0, -Ksat[0])

                # initially unsaturated profile
                if Airvol > eps:
                    # max infiltration rate to top node
                    MaxInf = max(-KLh[0]*(h_pond - h_iter[0] - z[0])/dz[0], -Ksat[0])
                    # only part fits into profile
                    if Qin >= Airvol:
                        # last limitation is from available pore space
                        q_sur = max(MaxInf, q0, -0.9*(Qin - Airvol)/dt)
                    # all fits into profile
                    elif Qin < Airvol:
                        q_sur = max(MaxInf, q0)

            # case evaporation, limited by soil supply or atm demand
            if q0 >= 0:
                # MaxEva=-KLh[0]*(h_atm - h_iter[0] - z[0])/dz[0] # limited by soil supply [m/s]
                q_sur = min(-KLh[0] * (h_atm - h_iter[0] - z[0])/dz[0], q0)

            # differential water capacity
            C = diffCapa(pF, h_iter)

            # set up tridiagonal matrix
            a, b, g, f = np.zeros(N)

            # intermediate nodes for j in range(1,N-1):
            # diag
            b[1:-1] = C[1:-1] + dt / dz[1:-1] * (KLh[1:-1] / dzu[1:-1] + KLh[2:] / dzl[1:-1])
            # subdiag
            a[1:-1] = -dt / (dz[1:-1] * dzu[1:-1]) * KLh[1:-1]
            # superdiag
            g[1:-1] = -dt / (dz[1:-1] * dzl[1:-1]) * KLh[2:]
            # rhs
            f[1:-1] = (
                C[1:-1] * h_iter[1:-1]
                - (W_iter[1:-1] - W_old[1:-1])
                + dt/dz[1:-1] * (KLh[1:-1] - KLh[2:]) * cosalfa
                - S[1:-1] * dt)

            # top node (j=0), flux-based bc

            b[0] = C[0] + dt/(dz[0]*dzl[0]) * KLh[1]
            a[0] = 0
            g[0] = -dt / (dz[0] * dzl[0]) * KLh[1]
            f[0] = (
                C[0]*h_iter[0]
                - (W_iter[0] - W_old[0])
                + dt / dz[0] * (-q_sur - KLh[1] * cosalfa)
                - S[0] * dt)

            # top node (j=0, head bc); now all formulated as flux-based
            # b[0] = C[0] + dt/(dz[0]*dzu[0])*Kh[0] + dt/(dz[0]*dzl[0])*Kh[1]
            # a[0] = 0;
            # g[0] = -dt/(dz[0]*dzl[0])*Kh[1];
            # f[0] = (
            #   C[0]*h_iter[0]
            #   - (Witer[0] - Wold[0])
            #   + dt/dz[0]*( (KLh[0]
            #   - KLh[1])*cosalfa
            #   + Kh[0]/dzu[0]*h_sur)
            #   - S[0]*dt)

            # bottom node (j=N)
            # flux bc
            if lbcType is not 'head':
                # print 'flux lbc'
                b[-1] = C[-1] + dt/(dz[-1] * dzu[-1]) * KLh[-1]
                a[-1] = -dt / (dz[-1] * dzu[-1]) * KLh[-1]  # -dt/(dz[-1]*dzu[-1])*KLh[-2];
                g[-1] = 0
                f[-1] = (
                    C[-1] * h_iter[-1]
                    - (W_iter[-1] - W_old[-1])
                    + dt / dz[-1] * (KLh[-1] * cosalfa + q_bot)
                    - S[-1] * dt)

            # head boundary, fixed head "at node N+1"
            else:
                # print 'head lbc'
                b[-1] = C[-1] + dt/(dz[-1] * dzu[-1]) * KLh[-2] + dt/(dz[-1] * dzl[-1]) * KLh[-1]
                a[-1] = -dt/(dz[-1] * dzu[-1]) * KLh[-2]
                g[-1] = 0
                f[-1] = (
                    C[-1]*h_iter[-1]
                    - (W_iter[-1] - W_old[-1])
                    + dt/dz[-1] * ((KLh[-2] - KLh[-1]) * cosalfa + KLh[-1]/dzl[-1] * h_bot)
                    - S[-1] * dt)

            # save old and update iteration values
            h_iterold = h_iter.copy()
            W_iterold = W_iter.copy()

            h_iter = thomas(a, b, g, f)
            # h_iter[h_iter>0]=0.0
            # print h_iter
            # find GWL
            _, gwl = getGwl(h_iter, z)

            W_iter = wrc(pF, x=h_iter)
            afp_iter = poros - W_iter

            if iterNo == 7:
                # re-try with smaller timestep
                dt = dt / 3.0
                iterNo = 0
                continue

            elif pass_flag is False or any(np.isnan(h_iter)):
                # break while loop
                print (
                    pass_flag, ', dt: ' + str(dt)
                    + ' qsur = ' + str(q_sur)
                    + ' qbot = ' + str(q_bot))
                if any(np.isnan(h_iter)):
                    print 'nan found'

                break

            err1 = max(abs(W_iter - W_iterold))
            err2 = max(abs(h_iter - h_iterold))

        # ending iteration loop
        # print 'Pass: t = ' +str(t) + ' dt = ' +str(dt) + ' iterNo = ' + str(iterNo)

        # new state at t
        h = h_iter
        W = wrc(pF, x=h)

        # update cumulative Infiltration, evaporation, drainage and h_pond [m]--------------
        if q_sur <= 0:
            C_inf += q_sur * dt
            h_pond = max(0, h_pond - (Ftop - q_sur) * dt)
            # create runoff if h_pond>maxpond
            rr = max(0, h_pond - maxPond)
            h_pond = h_pond - rr
            C_roff += rr
            del rr
        if q_sur > 0:
            C_eva += q_sur * dt
            # m
            h_pond = max(0, h_pond - (Ftop - q_sur) * dt)

        C_dra += -q_bot * dt  # + sum(HM*dz)*dt #drainage + net lateral flow
        # solution time & new initial timestep
        t += dt

        if iterNo < 2:
            dt = dt * 1.25
        elif iterNo > 4:
            dt = dt / 1.25

        dt = min(dt, t_final - t)
        # print 'new dt= ' +str(dt)
    # ending while loop, compute outputs

    # vertical fluxes
    # hydr conductivity m/s in this iteration
    KLh = hydrCond(pF, x=h, Ksat=Ksat)
    KLh = spatialAverage(KLh, method='arithmetic')

    # ms-1
    Fliq = nodalFluxes(z, h, KLh)

    # mass balance error [m]
    mbe = (
        (sum(W_ini*dz) - sum(W*dz))
        - (pond_ini - h_pond)
        - sum(S * dz) * t_final
        - C_inf - C_dra - C_eva
        )
    # print 'mbe:' +str(mbe)

    return h, W, h_pond, C_inf, C_eva, C_dra, C_roff, Fliq, gwl, mbe


# Utility functions
def getGwl(head, x):
    """ Finds ground water level and adjusts head in saturated nodes by adding hydrostatic pressure
    returns adjusted head and gwl
    """
    sid = find_index(head, lambda x: x >= 0)
    if len(sid) > 0:
        # gwl below first node
        if sid[0] > 0:
            gwl = x[sid[0]-1] + head[sid[0]-1]
        else:
            # gwl in first node
            gwl = x[sid[0]]
        # m, >0
        head[sid] = gwl - x[sid]
    else:
        # if gwl not in profile, assume hydr. equilibrium between last node and gwl
        gwl = head[-1] + x[-1]

    return head, gwl


def thomas(a, b, C, D):
    """ Tridiagonal matrix algorithm of Thomas
    a=subdiag, b=diag, C=superdiag, D=rhs
    """
    n = len(a)
    V, G, U, x = np.zeros(n)
    V[0] = b[0].copy()
    G[0] = C[0] / V[0]
    U[0] = D[0] / V[0]

    # numberr of nodes
    for i in range(1, n):
        V[i] = b[i] - a[i] * G[i - 1]
        U[i] = (D[i] - a[i] * U[i - 1]) / V[i]
        G[i] = C[i] / V[i]

    x[-1] = U[-1]
    inn = n - 2
    for i in range(inn, -1, -1):
        x[i] = U[i] - G[i] * x[i + 1]
    return x


def diffCapa(pF, head):
    """ Derivative of vGenuchten soil water retention curve [m-1]

    Args:
        pF (dict): water retension parameters
        head: [m]

    Returns:
        x: dW/dhead [m-1]
    """
    # cm
    head = -100 * head

    ts = pF['ThetaS']
    tr = pF['ThetaR']
    n = pF['n']
    m = 1.0 - np.divide(1, n)
    alfa = pF['alpha']

    # print ts, tr, n, m, alfa
    x = (
        100.0 * (ts - tr) * (n - 1.0)
        * alfa**n * head**(n - 1.0)
        / ((1 + (alfa * head)**n) ** (m + 1.0)))
    x[head <= 0.0] = 0.0

    return x


def find_index(a, func):
    """ Finds indexes or array elements that fill the condition
    call as find_index(a, lambda x: criteria)
    """
    return [i for (i, val) in enumerate(a) if func(val)]


def spatialAverage(y, x=None, method='arithmetic'):
    """ Calculates spatial average of quantity y

    Args:
        y: array
        x: grid
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
        f[1:-1] = np.sqrt(y[1:-2] * y[2:])
        f[0] = y[0]
        f[-1] = y[-1]

    elif method is 'dist_weighted':
        a = (x[0:-2] - x[2:]) * y[:-2] * y[1:-1]
        b = y[1:-1] * (x[:-2] - x[1:-1]) + y[:-2] * (x[1:-1] - x[2:])

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
    IN:
        x - grid
        y - value vector
    OUT:
        yprim - derivative
    """
    yprim = np.empty(np.shape(y))
    # central difference
    yprim[1:-1] = (y[2:] - y[0:-2]) / (x[2:] - x[0:-2])
    # forward difference at left boundary
    yprim[0] = (y[1] - y[0]) / (x[1] - x[0])
    # backward difference at right boundary
    yprim[-1] = (y[-1] - y[-2]) / (x[-1] - x[-2])

    return yprim


def waterFlow1D_CN(dt0, z, h0, pF, Ksat, Ftop, R, F=0.0, lbcType='impermeable',
                   lbcValue=None, Wice=0.0, maxPond=0.0, pond0=0.0, h_atm=-1000.0,
                   Implic=0.5, steps=10, MaxIter=100, IterLim=5.0e-5):
    """ 1D transient water flow solved by Crank-Nicholson scheme,
    following FEMMA -code (Koivusalo, Lauren et al.)

    OBS:
        TÄTÄ PITÄÄ ITEROIDA. MASSA EI SÄILY, ALARAJAN REUNAEHTO EI TOIMI EIKÄ NUMERIIKKA PELAA.

    Solves soil water flow in 1-D using Crank-Nicholson (predictor-corrector) finite difference
    solution of Richard's equation.

    Reference:
        Koivusalo (2009) FEMMA-document:

    IN:
        t_final - solution timestep [s]
        z - grid,<0, monotonically decreasing
        h0 - initial hydraulic head [m]
        pF - dict of vanGenuchten soil pF-parameters; pF.keys()=['ThetaR', 'ThetaS', 'n', 'alpha']
        Ksat - saturated hydr. cond. [ms-1]
        Ftop - potential top boundary flux (<0 for infiltration)
        R - local sink/source array due root uptake & well [s-1], <0 for sink
        HM - net lateral flux array [s-1], <0 for net outflow
        lbcType - lower bc type: 'impermeable', 'flux', 'free_drain', 'head'
        lbcValue - lower bc value; give for 'head' and 'flux'
        Wice0 - vol. ice content [m3m-3] - not needed now; could be used to scale hydr.conductivity
        maxPond - maximum allowed pond depth at surface [m]
        pond0 - initial pond depth [m]
        cosalfa - 1 for vertical water flow, 0 for horizontal transport
        h_atm - hydraulic head [m] in equilibrium with air humidity
            - used to compute soil evaporation supply
        steps - initial subtimesteps used to proceed to 't_final'
    OUT:
        h - new hydraulic head [m]
        W - new total water content [m3m-3]
        h_pond - new ponding depth [m]
        C_inf - total infiltration [m], <=0
        C_eva - total evaporation [m],>=0
        C_dra - total drainage from profile [m], <0 from profile
        C_roff - total surface runoff [m]
        Fliq - vertical water fluxes [ms-1] at t_final
        gwl - ground water level [m]; if not in computational layer then assumed hydrostatic
            equilibrium with node -1
        mbe - total mass balance error [m]
    CODE:
        Samuli Launiainen, Luke 8.4.2016. Converted from Matlab (APES SoilProfile.WaterFlow)
    NOTE:
        (8.4.2016): upper bc restriction checks needs to be tested
        (   -"-  ): include macropore adjustment as in APES-code?

    prepare sink term S -> (Eact(lyr) + (HorizLatFlowOut(lyr)) + HorizDeepLatFlow(lyr) _
        + HorizDrFlow(lyr)) / Dz(1)
    prepare Qinf = Precip + SurfSto / Dt

    """
    from soil_core import hydrCond
    from soil_core import wrc

    # lowerBC's
    if lbcType is 'impermeable':
        Qbot = 0.0
    if lbcType is 'flux':
        Qbot = lbcValue
    if lbcType is 'free_drain':
        Qbot = None  # value setlater
    if lbcType is 'head':
        h_bot = lbcValue

    S = R + F #net imposed sink/source term (root uptake + lateral flow)

    #-------------Get computation grid -------------
    N=len(z) #nr of nodal points, 0 is top
    dz=np.empty(N); dzu=np.empty(N); dzl=np.empty(N)
    #if any(z)>0: z=-z; #must be monotonic negative values

    #distances between grid points: dzu is between point i-1 and i, dzl between point i and i+1
    dzu[1:]=z[0:-1] - z[1:N]; dzu[0]=-z[0]
    dzl[0:-1]=z[0:-1] - z[1:]; dzl[-1]=(z[-2] - z[-1])/2.0;

    dz=(dzu + dzl)/2.0;
    dz[0]=dzu[0] + dzl[0]/2.0;
    print 'dz = ', dz
    print 'dzu = ', dzu
    print 'dzl = ', dzl

    #----soil variables and save intial conditions--
    if type(Ksat) is float: Ksat=np.zeros(N)+Ksat
    Ws=np.array(pF['ThetaS'])
    S=np.array(R)+np.array(F); del R, F #lumped sink/source term
    W_ini=wrc(pF,x=h0); #m3m-3

    #these change during process
    h_new=h0.copy()
    h_pond=pond0;


    #cumulative boundary fluxes
    C_inf=0.0;
    C_eva=0.0;
    C_dra=0.0;
    C_roff=0.0;

    dt=dt0/steps # internal timestep [s]
    for idt in range(1,steps): #internal time loop

        print 'idt: ' +str(idt)
        h_old = h_new.copy()
        h_iter = h_new.copy()
        iterNo=0

        for ic in range(MaxIter+1): #iteration loop
            iterNo +=1;

            KLh = hydrCond(pF, x = h_iter, Ksat=Ksat) #hydr conductivity m/s
            KLh=spatialAverage(KLh,method='arithmetic')
            C=diffCapa(pF, h_iter)    #differential water capacity

            #check upper boundary condition
            #print airVol
            if Ftop==0.0: Qtop=0.0;
            if Ftop<0: #case infiltration
                airVol = sum((Ws - wrc(pF, x=h_old))*dz)  #air filled porosity
                MaxInf=-Ksat[0]/dz[0]*(Implic *h_iter + (1.0-Implic)*h_old)

                Qtop =max(-airVol / dt, Ftop, MaxInf)
                #print 'infiltr'
            if Ftop>=0: #case evaporation
                MaxEva=-KLh[0]/dz[0]*(h_atm - h_old[0] - 1) #maximum evaporation by soil supply
                Qtop=max(Ftop,MaxEva)
                #print 'evap'
            #print Qtop


            #set up tridiagonal matrix
            a= np.zeros(N); b = np.zeros(N); g = np.zeros(N); f = np.zeros(N)
            #mid layers
            a[1:-1] = -Implic * KLh[0:-2] / ( dz[1:-1] * dzu[1:-1]) #subdiag
            b[1:-1] = C[1:-1] / dt + Implic * ( KLh[0:-2] / ( dz[1:-1] * dzu[1:-1]) + KLh[1:-1] / ( dz[1:-1] * dzl[1:-1]) )   #diag
            g[1:-1] = - Implic *KLh[1:-1] / (dz[1:-1]*dzl[1:-1]) #superdiag

            f[1:-1] = C[1:-1]*h_old[1:-1] / dt - (1-Implic) *( KLh[0:-2]/dz[1:-1]* ( (h_old[1:-1]-h_old[0:-2]) /dzu[1:-1] - 1.0) +\
                    KLh[1:-1]/dz[1:-1]* ( (h_old[2:]-h_old[1:-1]) /dzl[1:-1] - 1.0) ) - S[1:-1] #RHS

            #bottom bc
            if lbcType=='free_drain': Qbot=-KLh[-1];

            #flux boundary (impermeable: Qbot=0, prescribed flux: Qbot=lbcValue, free drainage: Qbot=k[-1])
            if lbcType is not 'head':
                #print dzz
                a[-1] = -Implic * KLh[-1]/ (dz[-1] *dzu[-1])
                b[-1] = C[-1]/dt + Implic*KLh[-1] / (dz[-1] *dzu[-1])
                g[-1] = 0.0
                f[-1] = C[-1]*h_old[-1] /dt - (1- Implic) * KLh[-1]/dz[-1]* ( (h_old[-1]-h_old[-2])/dzu[-1] - 1.0 ) + Qbot/dz[-1] - S[-1]

            else:   #fixed head lbcValue
                a[-1] = -Implic * KLh[-1] / ( dz[-2] * dzu[-1]) #subdiag
                b[-1] = C[-1] / dt + Implic * ( KLh[-2] / ( dz[-1] * dzu[-1]) + KLh[-1] / ( dz[-1] * dzl[-1]) )   #diag
                g[-1] = - Implic *KLh[-1] / (dz[-1]*dzl[-1]) #superdiag

                f[-1] = C[-1]*h_old[-1] / dt - (1-Implic) *( KLh[-2]/dz[-1]* ( (h_old[-1]-h_old[-2]) /dzu[-1] - 1.0) +\
                        KLh[-1]/dz[-1]* ( (h_bot-h_old[-1]) /dzl[-1] - 1.0) ) - S[-1] #RHS

            # top bc is flux-based
            a[0] = 0.0
            b[0] = C[0]/dt + Implic*KLh[1]/(dz[0]*dzl[0])
            g[0] = -Implic*KLh[1]/(dz[0]*dzl[0])

            f[0] = C[0]*h_old[0]/dt + (1 - Implic)* ( KLh[1]/dz[0]*( ( h_old[1] - h_old[0])/dzl[0] -1))-S[0] -Qtop/dz[0]

            #call tridiagonal solver
            h_iterold = h_iter.copy()
            h_iter=thomas(a,b,g,f);
            h_iter, gwl=getGwl(h_iter, z)

            err=max(abs(h_iter - h_iterold))
            #print err
            if err < IterLim:
                print iterNo
                print h_iter
                break
            #reset matrix coefficients
            #a.fill(np.NaN); C.fill(np.NaN); b.fill(np.NaN); D.fill(np.NaN)
    #update state variable and cumulative fluxes
    #print 'ic :' + str(ic)

        h_new=h_iter.copy()
        W_new=wrc(pF, x=h_new)
        C_inf += min(0.0, Qtop)*dt
        C_eva += max(0.0, Qtop)*dt
        C_dra += Qbot*dt

        if Qtop<=0:
            h_pond=max(0, h_pond - (Ftop - Qtop)*dt)
            rr=max(0, h_pond - maxPond) #create runoff if h_pond>maxpond
            h_pond=h_pond - rr;
            C_roff += rr; del rr

       # if ic==MaxIter: psiiNew=psiiOld0
    mbe=None

    return h_new, W_new, h_pond, C_inf, C_eva, C_dra, C_roff, mbe

# def solveRichardsAri(z, dz, pF, psii, Ksat, S, Qinf, dt):
#    """
#    prepare sink term S -> (Eact(lyr) + (HorizLatFlowOut(lyr)) + HorizDeepLatFlow(lyr) _
#        + HorizDrFlow(lyr)) / Dz(1)
#    prepare Qinf = Precip + SurfSto / Dt
#
#    """
#    from soil_core import diffWaterCap
#    from soil_core import wrc
#    nLyrs = len(psii)
#    Ws=np.array(pF['ThetaS'])
#    #z = np.array(pF['z'].values())
#    #dz = np.array(pF['dz'].values())
#    Implic = 0.5; IterLimit = 0.0005       #impic optioksi
#    psiiOld = psii.copy()
#    psiiNew = psiiOld.copy()
#
#    #Runs one day (kysy Samulilta)
#    a= np.zeros(nLyrs); C = np.zeros(nLyrs); b = np.zeros(nLyrs); D = np.zeros(nLyrs)
#    INSUM=0.0
#    for idt in range(int(86400.0/dt)):
#        psii = psiiNew.copy()
#        for ic in range(100):                                               #maximum iterations
#
#            psiiOld = psiiNew.copy()
#
#            k = hydrCond(pF, x = psiiOld, Ksat=Ksat)                   #hydr conductivity m/d in this iteration
#            dW = diffWaterCap(pF, psii, psiiOld)                       #differential water capacity
#
#            airVol = sum((Ws - wrc(pF, x=psiiOld))*dz)
#            if Qinf > airVol / dt: Qinf = airVol
#            MaxInf = -(k[0]) / 2.0 * (((1.0 - Implic) * psii[0] + Implic * psiiOld[0]) / dz[0] - 1.0)
#            print MaxInf
#            Qinf = min([Qinf, MaxInf])
#
#            print 'Qinf: ' +str(Qinf)
#            #top layer, always flux boundary condition
#            CondLow = (k[0] + k[1]) / 2.0
#            a[0] = 0.0
#            C[0] = -Implic * CondLow / dz[0] / (z[1] - z[0])
#            b[0] = dW[0] / dt - C[0]
#            D[0] = dW[0] * psii[0] / dt + Qinf / dz[0] - S[0] + \
#                CondLow / dz[0] * ((1.0 - Implic) * (psii[1] - psii[0]) / (z[1] - z[0]) - 1.0)
#
#            for lyr in range(1, nLyrs-1):
#                CondUp = CondLow
#                CondLow = (k[lyr] + k[lyr+1]) / 2.0
#                a[lyr] = -Implic * CondUp / dz[lyr] / (z[lyr] - z[lyr-1])   # tridiag vasemman puol elementit
#                C[lyr] = -Implic * CondLow / dz[lyr] / (z[lyr+1] - z[lyr])  # tridiag oikean  puol elementit
#                b[lyr] = dW[lyr] / dt - a[lyr] - C[lyr]                     # tridiag keskimmäiset (diag)  puol elementit
#                D[lyr] = dW[lyr] * psii[lyr] / dt  - S[lyr] - CondUp / dz[lyr]  * ((1.0 - Implic) * (psii[lyr] - psii[lyr-1]) / (z[lyr] - z[lyr-1]) - 1.0) \
#                    + CondLow / dz[lyr] * ((1.0 - Implic) * (psii[lyr+1] - psii[lyr]) / (z[lyr+1] - z[lyr]) - 1.0)       #lin yhtälön oikea puoli (edellisestä aikaaskeleesta)
#
#            #Bottom layer
#            CondUp = CondLow
#            a[-1] = -Implic * CondUp / dz[-1] / (z[-1] - z[-2])
#            C[-1] = 0.0
#            b[-1] = -(dW[-1] / dt - a[-1])
#            D[-1] = dW[-1] * psii[-1] / dt  - S[-1] - CondUp / dz[-1]  * ((1.0 - Implic) * (psii[-1] - psii[-2]) / (z[-1] - z[-2]) - 1.0)
#
#            psiiNew = thomas(a, b, C, D)
#            pDiff = np.amax(np.abs(psiiNew - psiiOld))
#            pMax = np.max([np.max(psiiNew), np.max(psiiOld), 0.01])
#            if pDiff/pMax < IterLimit:
#                break
#
#            #reset matrix coefficients
#            a.fill(np.NaN); C.fill(np.NaN); b.fill(np.NaN); D.fill(np.NaN)
#        INSUM+=Qinf*dt
#    return psiiNew,INSUM


# Steady-state solution of richards equation between ground water level (gwl, head=0) and
# upper boundary (constant head)
def solveRichardsSteady(z, gwl, pF, Ksv, Ksh=None, h_root=-50.0, figSwitch=False):
    """ Computes steady-state solution of Richards equation between ground water level and
    bottom of root zone.

    IN:
        z - grid [m], <0
        gwl - depth of ground water level [m]
        pF - dict of vanGenuchten pF parameters or list in order [ThetaS, ThetaR, alpha, n]
        Ksv - vertical sat. hydr. conductity [ms-1]
        Ksh - horizontal sat. hydr. conductity [ms-1]. None if Ksh = Ksv
        h_root - suction at upper boundary (root zone bottom) [m]
        figSwitch - True plots figures

    OUT:
        X - hydraulic head profile [m]
        UpFlux - capillary upflux to root zone [ms-1]
    """
    # from scipy.interpolate import interp1d
    from soil_core import hydrCond
    import matplotlib.pyplot as plt

    Omega=0.5
    Conv_crit=0.00001
    maxIter=500

    #-------------Get computation grid -------------
    N=len(z) #nr of nodal points, 0 is top
    dz=np.empty(N); dzu=np.empty(N); dzl=np.empty(N)
    #if any(z)>0: z=-z; #must be monotonic negative values

    #distances between grid points: dzu is between point i-1 and i, dzl between point i and i+1
    dzu[1:]=z[0:-1] - z[1:N]; dzu[0]=-z[0]
    dzl[0:-1]=z[0:-1] - z[1:]; dzl[-1]=dzl[-2];

    dz=(dzu + dzl)/2.0;
    dz[0]=dzu[0] + dzl[0]/2.0;
#    print 'z = ', z
#    print 'dz = ', dz
#    print 'dzu = ', dzu
#    print 'dzl = ', dzl

    if type(Ksv) is float: Ksv=np.ones(N)*Ksv
    if type(Ksh) is float:
        Ksh=np.ones(N)*Ksh

    elif Ksh is None:
        Ksh=Ksv

    #tridiagonal elements
    A= np.zeros(N); B = np.zeros(N); C = np.zeros(N); D = np.zeros(N);  X=np.zeros(N); X_old=np.zeros(N);

    #initial condition
    X[z<=gwl]=gwl - z[-1]
    X[z>gwl]=0.5*h_root;

#    dd = [z[0], gwl,z[-1]]
#    yy = [h_root,  0.0, 0.0]
#    iPsii= interp1d(dd, yy); del yy, dd
#    X=iPsii(z).copy()
    #print 'X=', X

    #X_ini=X.copy() #save for possible smaller Omega

    #bc at root zone bottom
    A[0] = 0.0
    B[0] = 1.0
    C[0] = 0.0
    D[0] = h_root

    #bc at gwl
    A[-1] = 0.0
    B[-1] = 1.0
    C[-1] = 0.0
    D[-1] =max(0.0, gwl-z[-1])

    err=9999.99; iterNo=0
    while err>Conv_crit and iterNo < maxIter:
        iterNo +=1

        Con = hydrCond(pF, x = X, Ksat=Ksv)                   #hydr conductivity m/s in this iteration
        Con = spatialAverage(Con, 'arithmetic')
        X_old = X.copy()

        #tridiag elements, middle nodes
        A[1:-1]=Con[1:-1] / dzu[1]
        C[1:-1]=Con[2:] / dzl[1]
        B[1:-1]= - C[1:-1] - A[1:-1]
        D[1:-1]= - (Con[1:-1] - Con[2:])

        Xi = thomas(A,B,C,D)

        X=Omega*Xi + (1.0 - Omega)*X_old

        err=max(abs(X-X_old))
        #print err
        if iterNo==maxIter:
            Omega=0.1;
            iterNo=0;
            #X=X_ini.copy() #reset to initial condition

    # ------ convergence, compute upflux profile
    #print 'iterNo=' + str(iterNo) +',Omega=' + str(Omega)

    flx=np.zeros(N)
    for k in range(0,N-1):
        flx[k] = - Con[k]*( (X[k]-X[k+1])/dz[k])
    flx[X>=0.0]=0.0
    UpFlux=flx[1] #assume that flux at node i=1 equals capillary rise to root zone

#    flux using central difference
#    xx=X.copy(); xx[xx>0.0]=0.0
#    flx=nodalFluxes(z,xx,Con) #compute nodal fluxes
#    flx[X>0]=0.0
#    UpFlux=flx[1] #assume that flux at node i=1 equals capillary rise to

    if figSwitch is True:
        plt.figure()
        plt.subplot(121); plt.plot(X,z,'r.-'); plt.ylabel('z'); plt.xlabel('h [m]')
        plt.subplot(122); plt.plot(flx,z,'r.-'); plt.ylabel('z'); plt.xlabel('Flux [ms-1]')
        print 'upflux = ' + str(UpFlux)

    return X, UpFlux

""" ************ Drainage equations ********************"""

def drainage_Linear(zs,Ksat,GWL,DitchDepth,DitchSpacing):
    """"
    Calculates drainage from soil profile to ditch using simple linear equation,
    i.e. accounts only drainage from layers where GWL<DitchDepth
    INPUT:
       zs - depth of soil node (m), array, zs<0
       Ksat - saturated hydraulic conductivity (m/s),array
       GWL - ground water level below surface (m), GWL<0
       DitchDepth (m), depth of drainage ditch bottom
       DitchSpacing (m), horizontal spacing of drainage ditches
    OUTPUT:
       Q - total drainage from soil profile (m/s), >0 is outflow
       Qz_drain - drainage from each soil layer (m m-1s-1), i.e. sink term to Richard's eq.
    """

    dx=1; #unit length of horizontal element (m)

    N=len(zs)

    Qz_drain=np.zeros(N)
    dz=np.zeros(N)

    dz[1:]=(zs[0:-1] - zs[1:]) #m
    dz[0]=-2*zs[0]

    Keff=Ksat*dz/dx #transmissivity m s-1 in each layer

    # (-), positive gradient means flow towards ditches, return flow neglected
    hgrad=np.max([(GWL + DitchDepth)/(0.5*DitchSpacing), 0.0])

    ix1=np.where( (zs-GWL)<0)
    ix2=np.where(zs>-DitchDepth) #layers above ditch bottom where drainage is possible
    ix=np.intersect1d(ix1,ix2); del ix1, ix2

    Qz_drain[ix]=Keff[ix]*hgrad #layerwise drainage ms-1
    Q=sum(Qz_drain);
    Qz_drain=Qz_drain/dz #s-1, sink term

    return Q, Qz_drain


def drainage_Hooghoud(zs,Ksat,GWL,DitchDepth,DitchSpacing,DitchWidth,Zbot):
    """
    Calculates drainage to ditch using Hooghoud's drainage equation,
    i.e. accounts only drainage from saturated layers above and below ditch bottom
    INPUT:
       zs - depth of soil node (m), array, zs<0
       Ksat - saturated hydraulic conductivity (m/s),array
       GWL - ground water level below surface (m), GWL<0
       DitchDepth (m), depth of drainage ditch bottom, >0
       DitchSpacing (m), horizontal spacing of drainage ditches
       DitchWidth (m), ditch bottom width
       Zbot (m), distance to impermeable layer, scalar, >0
    OUTPUT:
       Q - total drainage from soil profile (m/s), >0 is outflow
       Qz_drain - drainage from each soil layer (m m-1s-1), i.e. sink term to Richard's eq.
    REFERENCE:
       Follows Koivusalo, Lauren et al. FEMMA -document. Ref: El-Sadek et al., 2001.
       J. Irrig.& Drainage Engineering.

    Samuli Launiainen, Metla 3.11.2014.; converted to Python 14.9.2016
    """
    if -Zbot<min(zs): Zbot=-min(zs)

    N=len(zs)
    Qz_drain=np.empty(N)
    Qa=0.0; Qb=0.0

    dz=np.empty(N)
    dz[1:]=(zs[0:-1] - zs[1:]) #m
    dz[0]=-2*zs[0]


    Hdr=np.maximum(0,GWL+DitchDepth) #depth of saturated layer above ditch bottom
    #print Hdr
    if Hdr>0:
        #print 'Hooghoud'
        Trans=Ksat[:]*dz# transmissivity of layer, m2s-1
        #print np.shape(Trans)

        #-------- drainage from saturated layers above ditch base
        #ix=np.where( (zs-GWL)<0 and zs>-DitchDepth) #layers above ditch bottom where drainage is possible
        ix1=np.where( (zs-GWL)<0)
        ix2=np.where(zs>-DitchDepth) #layers above ditch bottom where drainage is possible
        ix=np.intersect1d(ix1,ix2); del ix1, ix2

        if ix.size >0:
            Ka=sum(Trans[ix])/Hdr #effective hydraulic conductivity ms-1
            Qa=4*Ka*Hdr**2 / ( DitchSpacing**2 ) #m s-1, total drainage above ditches
            #sink term s-1, rel=Trans(dra)/sum(Trans(dra)) partitions Qa by relative transmissivity of layer
            Qz_drain[ix]=Qa*Trans[ix]/sum(Trans[ix])/dz[ix]
            del ix #, Ka

        #-----drainage from saturated layers below ditch base

        #layers below ditch bottom where drainage is possible
        ix=np.intersect1d(np.where(zs<=-DitchDepth), np.where((zs-GWL)<0))

        #depth of saturated layer below drain base
        zbase= np.minimum(abs(np.min(zs) +DitchDepth), abs(np.min(zs) - GWL))

        Kb=sum(Trans[ix])/zbase;

        #compute equivalent depth Deq
        A=3.55 -1.6 * Zbot / DitchSpacing -2 * ( 2 / DitchSpacing )**2;
        Reff=DitchWidth/2 #effective radius of ditch
        Dbt=-Zbot +DitchDepth #distance from impermeable layer to ditch bottom

        if Dbt/DitchSpacing <=0.3:
            Deq= Zbot / ( 1 + Zbot / DitchSpacing * (8 / np.pi * np.log( Zbot / Reff) - A)) #m
        else:
            Deq=np.pi * DitchSpacing / (8 * np.log( DitchSpacing/Reff ) - 1.15) #m

        Qb=8 * Kb * Deq * Hdr / DitchSpacing**2 # m s-1, total drainage below ditches
        Qz_drain[ix]= Qb * Trans[ix] / sum(Trans[ix]) / dz[ix] #sink term s-1

        del ix

    Q=Qa + Qb #total drainage m s-1, positive is outflow to ditch
    #print Hdr, Qa, Qb, Q
        #plt.figure(1)
        #plt.subplot(121); plt.plot([0 1],[GWL GWL],'r-',[0 1],[-DitchDepth -DitchDepth],'k-')
        #plt.subplot(122); plt.plot(Qz_drain,zs,'r-'); ylabel('zs'), xlabel('Drainage profile s-1');
    return Q, Qz_drain

""" ************** Functions for testing stuff in this module **************"""

def test_drainage(DitchDepth, DitchSpacing):
    """ tests drainage equations"""

    DitchWidth=0.8;
    Zbot=1.0

    zs=-np.arange(0.1,2.0,0.01); #grid
    Ksat=1e-5 #m/s

    gwl=np.arange(-DitchDepth, max(zs), 0.01)
    N=len(gwl)
    Drain=np.zeros([N,1])
    DrainHoog=np.zeros([N,1])
    Qprofile=[]
    for k in range(0,N):
        Q,Qz=drainage_Linear(zs,Ksat,gwl[k], DitchDepth, DitchSpacing)
        Drain[k]=Q
        Qprofile.append(Qz)
        del Q, Qz
        Q,Qz = drainage_Hooghoud(zs,Ksat,gwl[k],DitchDepth,DitchSpacing,DitchWidth,Zbot)
        DrainHoog[k]=Q
    plt.figure()
    plt.subplot(121); plt.plot(gwl,Drain,'r.-', gwl, DrainHoog, 'g.-'); plt.title('Q ms-1')
    plt.subplot(122); plt.plot(gwl,DrainHoog,'c.-')

    return Drain, Qprofile
