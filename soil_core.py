# -*- coding: utf-8 -*-

""""
Soil_module contains functions and class definitions related to soil physics and hydrology.

soil_core defines utility functions, soil properties etc.
Samuli Launiainen, Luke 2/2016
"""

import numpy as np
import matplotlib.pyplot as plt

eps = np.finfo(float).eps  # machine epsilon
"""
General functions
"""


def wrc(pF, value=None, var=None):
    """
    vanGenuchten-Mualem soil water retention curve (van Genuchten, 1980;
    Schaap and van Genuchten, 2006)

    .. math::
        \\theta(\\psi_s) = \\theta_{res} +
        \\frac{\\theta_{sat}-\\theta_{res}}
        {(1 + \\lvert \\alpha + \\psi_{s}\\rvert^n)^m}

    where :math:`\\theta_{res}` and :math:`\\theta_{sat}` are residual and saturation
    water contents (m\ :sup:`3` m :sup:`-3`\ ), :math:`\\alpha`\ , *n*, and :math:`m=1-^1/_n`
    are empirical shape parameters.

    Sole input 'pF' draws water retention curve and returns 'None'.
    For drawing give only one pF-parameter set. If several pF-curves are given,
    x can be scalar or len(x)=len(pF). In former case var is pF(x),
    in latter var[i]=pf[i,x[i]]

    References:
        Schaap and van Genuchten (2005)
            Vadose Zone 5:27-34
        van Genuchten, (1980)
            Soil Science Society of America Journal 44:892-898

    Args:
        pF (list/dict):
            0. 'ThetaS' saturated water content [m\ :sup:`3` m :sup:`-3`\ ]
            1. 'ThetaR' residual water content [m\ :sup:`3` m :sup:`-3`\ ]
            2. 'alpha' air entry suction [cm\ :sup:`-1`]
            3. 'n' pore size distribution [-]
        value:
            * [m\ :sup:`3` m\ :sup:`-3`\ ] if input is volumetric water content
            * [m] if input is water potential
        var: flag for conversion
            * 'Th' for volumetric water content
            * None for water potential

    Returns:
        float:
            * [m\ :sup:`3` m\ :sup:`-3`\ ] if input is water potential
            * [m] if input is volumetric water content

    Samuli Launiainen, Luke 2/2016
    """
    if type(pF) is dict:  # dict input
        # Ts, Tr, alfa, n =pF['ThetaS'], pF['ThetaR'], pF['alpha'], pF['n']
        Ts = np.array(pF['ThetaS'])
        Tr = np.array(pF['ThetaR'])
        alfa = np.array(pF['alpha'])
        n = np.array(pF['n'])
        m = 1.0 - np.divide(1.0, n)

    else:  # list input
        pF = np.array(pF, ndmin=2)  # ndmin=1 needed for indexing to work for 0-dim arrays
        # Ts=pF[0]; Tr=pF[1]; alfa=pF[2]; n=pF[3]
        Ts = pF[:, 0]
        Tr = pF[:, 1]
        alfa = pF[:, 2]
        n = pF[:, 3]
        m = 1.0 - np.divide(1.0, n)

    # Converts volumetric water content (m3/m3) to water potential (m)
    def theta_psi(theta):  # 'Theta-->Psi'
        theta = np.minimum(theta, Ts)
        theta = np.maximum(theta, Tr)  # checks limits
        s = (Ts - Tr) / ((theta - Tr) + eps)  # **(1/m)
        # rises RuntimeWarning: invalid value encountered in power
        # code has been tested and it seems to work fine when s approaches zero
        Psi = (-1e-2 / alfa * (s**(1/m) - 1)**(1/n))  # in m
        return Psi

    # Converts water potential (m) to volumetric water potential (m3/m3)
    def psi_theta(psi):  # 'Psi-->Theta'
        psi = 100 * np.minimum(psi, 0)  # cm
        Th = Tr + (Ts - Tr) / (1 + abs(alfa * psi) ** n) ** m
        return Th

    # This does all the work
    if value is None and np.size(Ts) == 1:  # draws pf-curve
        xx = -np.logspace(-4, 5, 100)  # cm
        yy = psi_theta(xx)
        # field capacity and wilting point
        fc = psi_theta(-1.0)
        wp = psi_theta(-150.0)
        fig = plt.figure()
        fig.suptitle('vanGenuchten-Mualem WRC', fontsize=16)
        # ttext=str(pF).translate(None,"{}'")
        ttext = (r'$\theta_s=$' + str(Ts) + r', $\theta_r=$'
                 + str(Tr) + r', $\alpha=$'+str(alfa) + ',n=' + str(n))

        plt.title(ttext, fontsize=14)
        plt.semilogx(-xx, yy, 'b-')
        plt.semilogx(1, fc, 'ro', 150, wp, 'ro')  # fc, wp
        plt.text(1, 1.1 * fc, 'FC'), plt.text(150, 1.2 * wp, 'WP')
        plt.ylabel(r'$\theta$  $(m^3m^{-3})$', fontsize=14)
        plt.xlabel('$\psi$ $(m)$', fontsize=14)
        plt.ylim(0.8 * Tr, min(1, 1.1 * Ts))
        del xx, yy
        return None

    elif value is None:
        print 'wrc: To draw water-retention curve give only one pF -parameter set'
        return None

    if var is 'Th':
        y = theta_psi(value)  # 'Theta-->Psi'
    else:
        y = psi_theta(value)  # 'Psi-->Theta'

    return y


def effSat(pF, x, var=None):  # layer effective saturation [ratio, max 1]
    """
    Effective saturation
    IN:
        pF - dict['ThetaS': ,'ThetaR': ,'alpha':, 'n':,] OR
           - list [ThetaS, ThetaR, alpha, n]
        x - Theta [vol/vol] or Psi [m H2O]
        var = 'Psi' if x = Psi
    OUT
        es= (x-Tr)/(Ts-Tr)
    """
    if type(pF) is dict:  # dict input
        Ts = np.array(pF['ThetaS'])
        Tr = np.array(pF['ThetaR'])
        alfa = np.array(pF['alpha'])
        n = np.array(pF['n'])

    else:  # list input
        pF = np.array(pF, ndmin=1)  # ndmin=1 needed for indexing to work for 0-dim arrays
        Ts = pF[0]
        Tr = pF[1]
        alfa = pF[2]
        n = pF[3]

    if var is None or var is 'Th':  # x=Th
        es = np.minimum((x-Tr)/(Ts-Tr + eps), 1.0)
    else:
        x = 100*np.minimum(x, 0)  # cm
        x = Tr + (Ts-Tr)/(1+abs(alfa*x)**n)**(1.0-1.0/n)
        es = np.minimum((x-Tr)/(Ts-Tr + eps), 1.0)
    return es


def hydrCond(pF, x=None, var=None, Ksat=1):
    """
    Hydraulic conductivity following vanGenuchten-Mualem
    IN:
        pF - dict or list
        x - Theta [vol/vol] or Psi [m H2O]
        var = 'Th' if x in [vol/vol]
        Ksat - saturated hydraulic conductivity [units]
    OUT:
        Kh - hydraulic conductivity ( if Ksat ~=1 then in [units], else relative [-])
    """

    if type(pF) is dict:  # dict input
        alfa = np.array(pF['alpha'])
        n = np.array(pF['n'])
        m = 1.0 - np.divide(1.0, n)

    else:  # list input
        pF = np.array(pF, ndmin=1)  # ndmin=1 needed for indexing of 0-dim arrays
        # alfa=pF[2]; n=pF[3]
        alfa = pF[:, 2]
        n = pF[:, 3]
        m = 1.0 - np.divide(1.0, n)

    def kRel(x):
        nm = (1 - abs(alfa*x)**(n-1) * (1 + abs(alfa*x)**n)**(-m))**2
        dn = (1 + abs(alfa*x)**n)**(m/2.0)
        r = nm/(dn+eps)
        return r

    if x is None and np.size(alfa) == 1:  # draws pf-curve
        xx = -np.logspace(-4, 5, 100)  # cm
        yy = kRel(xx)
        fig = plt.figure()
        fig.suptitle('Hydr. cond. (vanGenuchten-Mualem)', fontsize=16)
        # ttext=str(pF).translate(None,"{}'")
        ttext = r'$K_{sat}=$' + str(Ksat) + r', $\alpha=$'+str(alfa) + ', n='+str(n)

        plt.title(ttext, fontsize=14)
        plt.semilogx(-xx, yy, 'g-')

        plt.ylabel(r'K_{sat}', fontsize=14)
        plt.xlabel('$\psi$ $(cm)$', fontsize=14)

        del xx, yy
        return None

    elif x is None:
        print 'hydrCond: To draw curve give only one pF -parameter set'
        return None

    # this computes and returns
    x = np.array(x)
    if x is not None and var is 'Th':
        x = wrc(pF, x=x, var='Th')

    Kh = Ksat * kRel(100.0 * x)

    return Kh


def rew(pF, x, var=None):
    """
    Returns relative extractable water [0...1] at the prevailing Theta or Psi using
    vanGenuchten-Mualem model to compute wilting point and field capacity.
    IN:
        pF - dict.
        x - Theta [vol/vol] or Psi [m H2O]
        var = 'Psi' if x = Psi
    OUT:
        rew [-]
    """

    if var is not None:
        x = wrc(pF, x)  # Psi-->Th

    # Ts, Tr, alfa, n =pF['ThetaS'], pF['ThetaR'], pF['alpha'], pF['n']
    Tf, Tw = wrc(pF, x=[-1.0, -150.0])   # Fc & Wp

    rew = np.minimum((x-Tw)/(Tf-Tw + eps), 1.0)
    rew = np.maximum(rew, 0)
    return rew


def rew_general(x, fc, wp):
    """
    Returns relative extractable water [0...1] at the prevailing Theta.
    IN:
        x - Theta [vol/vol]
        fc - field capacity (vol/vol)
        wp - wilting point (vol/vol)
    OUT:
        rew [-]
    """
    rew = np.minimum((x-wp)/(fc-wp + eps), 1.0)
    rew = np.maximum(rew, 0)
    return rew


def soilRH(T, Psi):
    """ Calculates soil relative humidity (-) according Philip and de Vries, 1957

    IN: T (degC), head (m)
        %T=temperature (degC), Psi = soil water tension (m H2O, negative)
    OUT: rh (-), relative humidity
    Note: can accept scalar of list inputs
    """
    # constants
    grav = 9.81  # acceleration due gravity, kgm/s2
    GasConst = 8.314  # univ. gas constant, Jmol-1
    Mwater = 18.015e-3  # molar mass H2O, kg mol-1

    # relative humidity in soil pores (-)
    # if isinstance(T, float)==False: T=np.array(T)
    rh = np.exp(-(Mwater*grav*abs(Psi))/(GasConst*(T+273.15)))

    return rh


def e_sat(T):
    """ Saturation vapor pressure (es) over water surface and its derivative d(es)/dT

    IN: T (degC)
    OUT: es [Pa], d(es)/dT [Pa K-1]
    """
    if not isinstance(T, float):
        T = np.array(T)

    esat = 611.0 * np.exp((17.502*T)/(T + 240.97))  # (Stull 1988)
    ss = 17.502*240.97*esat/((240.97+T)**2)  # [Pa K-1], from Baldocchi's Biometeorology -notes
    return esat, ss


def moistureConvert(x=None, var=None, bd=None):
    """ Converts between vol. and grav. water content.

    Args:
        x: vol. water content [vol/vol] or gravimetric water cont. on dry-mass basis [H2O/dryweight]
        var - 'Th' if x is vol. water content
        bd - soil bulk density (dry density) [kg/m3 = 0.001 g/cm3]. Thus 1 g/cm3=1000kg/m3
    Returns:
        converted water content
    """
    dw = 1000.0  # kg/m3

    if x is None or var is None or bd is None:
        print 'moistureConvert: incorrect inputs'
        y = None

    elif var is 'Th':
        print 'Th--> w'
        y = np.array(x)*np.divide(dw, bd)  # return w

    else:
        y = np.array(x)*bd/dw  # return Th

    return y


class SoilType(object):
    """
    ************* KESKEN *****************************

    Samuli Launiainen, Luke 2/2016

    SoilType is a zero-dimensional soil object that defines soil physical properties and contains
    functions to compute them from physical state variables.

    General:
        ID - soil type ID code
        Name - soil type name / code
        Type - mineral, peat
        Subtype - friction,cohesive,organic
        BD - bulk density (gm-3)
    Composition:
        vSand - sand content (vol/vol)
        vSilt - silt content (vol/vol)
        vClay - clay contnet (vol/vol)
        vOrganic - organic matter content (vol/vol)
    Hydraulic properties:
        ThetaS - soil volumetric water content at saturation [vol/vol], i.e. porosity
        ThetaFc - vol. water content at field capacity [vol/vol](xx kPa)
        ThetaWp - vol. water content at wilting point [vol/vol](xx kPa)
        ThetaR  - residual vol. water content [vol/vol]
        pF - dictionary of vanGenuchten parameters [Theta_s Theta_r alpha (cm-1) n (-) compr], where
            alpha is inverse of air entry potential, n a shape parameter and compr. compressibility.
        Ksh - saturated horizontal hydraulic conductivity [m s-1]
        Ksv - saturated vertical hydraulic conductivity [m s-1]
    Thermal properties:
        Lambda - thermal conductivity [units] of parent materials
        Cp - volumetric heat capacity [units] of parent materials
        ThermDiff - thermal diffusivity [units] of parent materials
        FreezingPara - dictionary of freezing curve parameters [Tf a]

    Used Packages:
        np - numpy

    References:
        Skaggs, R.W., 1980. A Water Management Model for Artificially Drained Soils.
        North Carolina Agricultural Research Service, Raleigh, 54 p.

    """

    def __init__(self, ID, Name, Type, compo, bd, pF, Ksv=None, Ksh=None):
        """
        Initializes SoilType -instance
        IN:
            ID(int),
            Name(str),
            Type(str),
            compo(dict),
            bd (float),pF(dict),
            Lambda(float),
            bkpar=(float;Brooks-Corey),
            Ksv(float, m/s),
            Ksh(float, m/s)

        """
        self.ID = ID
        self.Name = Name
        self.Type = Type
        self.vSand = compo['vSand']
        self.vSilt = compo['vSilt']
        self.vClay = compo['vClay']
        self.BD = bd
        self.pF = pF
        self.ThetaS = pF['ThetaS']
        self.ThetaR = pF['ThetaR']
        self.ThetaFc = wrc(self.pF, -1.0)
        self.ThetaWp = wrc(self.pF, -150.0)
        self.Ksh = Ksh
        self.Ksv = Ksv


class SoilThermal(SoilType):
    """
    SoilType class extension to include thermal properties of soils
    """
    def __init__(
            self,
            Lambda=None,
            SType=None,
            ID=None,
            Name=None,
            Type=None,
            compo=None,
            pF=None,
            Kh=None):

        if not isinstance(SType, SoilType):
            # initialize from inputs
            SoilType.__init__(self, ID, Name, Type, compo, pF, Kh)
            # super(SoilThermal,self) = __init__(self, ID, Name, Type, compo, pF)
            self.Lambda = Lambda
            self.INFO = 'created from inputs'

        else:
            # initialize from SType properties

            properties = SType.__dict__
            for key in properties:
                    setattr(self, key, properties[key])

            self.Lambda = Lambda
            self.INFO = 'created from SoilType instance'

# 1-Dimensional soil water flow (Richard's equation)


def diffWaterCap(pF, xOld, xNew):
    """ Computes differential water capacity with respect to change in head between the iteration steps.
    xOld - psii in previous iteration
    xNew -  psii in present iteration

    """
    P1 = xOld
    P2 = np.where(np.abs(xNew - xOld) < 0.00001, np.ones(len(xOld))*0.00001, xNew)
    t2 = wrc(pF, x=P2)
    t1 = wrc(pF, x=P1)
    return (t2 - t1) / (P2 - P1)
