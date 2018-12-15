# -*- coding: utf-8 -*-
"""
Created on Wed Feb 10 14:52:54 2016

@author: slauniai
"""

""""
Trafficability contains functions and class definitions related to forest soil trafficability.\n
The functions are modular and most can be used both by scalar or n x m matrix inputs. \n

****
SOURCES:

Vega-Nieva et al. 2009. Can. J. Soil Sci.: Modular terrain model for daily variations in machine-specific
forest soil trafficability

Frankenstein & Koenig (2004). Fast All-season Soil STrength (FASST). Cold Regions Research and Engineering Lab. 
US Army Corps of Engineers. ERDC/CRREL SR-04-1
        
Ayers & Perumbal (1982)
Saarilahti 2002 Soil Interaction model
Maclaurin (1990), Saarilahti (1991), Amarjan (1972)

****

Samuli Launiainen, Luke 2-6/2016
"""
#import modules
import numpy as np
import matplotlib.pyplot as plt
#import soil_core as sm

eps = np.finfo(float).eps #machine epsilon


"""
SOURCES:

Vega-Nieva et al. 2009. Can. J. Soil Sci.: Modular terrain model for daily variations in machine-specific
forest soil trafficability
"""
#
#def testValues():
#    """
#    returns realistic input values for desting other functions
#    """
#    #ci=range(100,6000,100) #Cone index, kPa
#    #bd=

class Vehicle(object):
    """
    Vehicle -class defines a terrain vehicle
    For computation of rut depths and critical cone index Vehicle should have following properties:

    Type - string, not used
    Weight - empty weight [kg]
    Load - load [kg]
    Wheels - number of wheels [-]
    Axles - number of axles [-], not used
    W - wheel load [kg] (Weight + Load) /Wheels
    Tyre: dict with fields: {'dia' [m], 'hei' [m], 'width' [m], 'pres' [kPa], 'defl [m]'}

    nT - relative wheel numeric [-], Turnage (1978). Multiply bT, nM, nB by CI to get wheel numeric for current CI
    nM - relative wheel numeric [-], Maclaurin (1997)
    nB - relative wheel numeric [-], Brixius (xxxx)     
   
    """
    
    def __init__(self,  *args,**kwargs):
        """
        Initializes object using args that neeed to be dictionaries (key--> field), or by keyword arguments kwargs.
        Should give at minimum: Weight, Load, Wheels, Tyre properties except deflection
        """
        for dictionary in args:
            for key in dictionary:
                setattr(self, key, dictionary[key])
        for key in kwargs:
            setattr(self, key, kwargs[key])
        
        self.Tyre['defl'] = Vehicle.tyreDefl(self.W, self.Tyre['pres'])
    
    @staticmethod
    def tyreDefl(W,p):
        """ 
        Tyre deflection [m] from Tyre properties, Saarilahti (2002), eq. 2.22
        IN: W - wheel load [kg], p - tyre pressure [kPa]. Scalar inputs
        OUT: d - deflection [m]        
        """
        W=1e-3*W*9.81 #kg --> kN
        d=8.0e-4 + 1.0e-3*(0.365 + 170.0/p)*W     
        
        return d
    
    @staticmethod
    def wheelNumericTurnage(tyre, W, ci=1):
        """
        Wheel numeric by Turnage (1978).
        IN:
            tyre - dict with fields: {'dia' [m], 'hei' [m], 'width' [m], 'pres' [kPa], 'defl [m]'}
            W - wheel load [kg]
            ci - cone index [kPa]. if ci=1 then relative wheel numeric is returned
        OUT:
            nci - wheel numeric [-]
        """
        W=9.81e-3*W #kg --> kN
        b=tyre['width']; d=tyre['dia']; h=tyre['hei']; defl=tyre['defl']

        a= 1.0/(1 + b/(2*d))        
        nci=ci/W*b*d*np.sqrt(defl/h)*a 
        return nci

    @staticmethod
    def wheelNumericMaclaurin(tyre, W, ci=1):
        """
        Wheel numeric by Maclaurin (1997).
        IN:
            tyre - dict with fields: {'dia' [m], 'hei' [m], 'width' [m], 'pres' [kPa], 'defl [m]'}
            W - wheel load [kg]
            ci - cone index [kPa]. if ci=1, then relative wheel numeric is returned
        OUT:
            nci - wheel numeric [-]
            gpi - ground pressure [kPa]
            lci - limiting cone index [kPa]
        """
        W=9.81e-3*W #kg --> kN
        b=tyre['width']; d=tyre['dia']; defl=tyre['defl']

        gpi=W / b**0.8*d**0.8*defl**0.4 #kPa
        nci=ci / gpi
        
        lci=1.85*gpi #Saarilahti (2002) eq. 2.20, ref. Ziesak & Matthies (2001)
        return nci, gpi, lci
    
    @staticmethod
    def wheelNumericBrixius(tyre, W, ci=1):
        """
        Wheel numeric by Brixius (xxxx), Saarilahti (2002) eq. 2.25
        IN:
            tyre - dict with fields: {'dia' [m], 'hei' [m], 'width' [m], 'pres' [kPa], 'defl [m]'}
            W - wheel load [kg]
            ci - cone index [kPa]. if ci=1 then relative wheel numeric is returned
        OUT:
            nci - wheel numeric [-]
        """
        W=9.81e-3*W #kg --> kN
        b=tyre['width']; d=tyre['dia']; h=tyre['hei']; defl=tyre['defl']
     
        nci=ci*b*d/W * ( (1.0 + 5.0 *defl/h) / (1.0 + 3.0*b/d)) 
        return nci
        
    @staticmethod
    def ngp(W, r, b):
        """ 
        Nominal ground pressure [kPa]
        IN:
            W - wheel load [kg]
            r - wheel radius [m]
            b - tyre width [m]
        OUT:
            ngp - nominal ground pressure [kPa]
        """
        W=9.81e-3*W #kg --> kN
        ngp=W/(r*b)
        return ngp
        
def coneIndexVN(a=None, fsand=None, poros=None, S=None):
    """
    Soil penetration resistance, V-N eq. 4
    IN:
        a - parameter list [a0, aPS, aSat, aSand]
        fsand - sand content [g sand/g dryweight]
        poros - porosity [m3/m3]
        S - saturation ratio [-], i.e. fracion of pore space filled
    OUT:
        q - cone index, penetration resistance [kPa]
    """
    para0=[1.99, 0.38, 2.23, 0.72] #V-N eq-4
    
    if a is None: 
        a=para0; print 'coneIndexVN: using default field params.'
        q=1.08e3*10**(a[0] - a[1]*fsand - a[2]*poros - a[3]*S) #kPa
    else:
        q=1e3*10**(a[0] - a[1]*fsand - a[2]*poros - a[3]*S) #kPa

    return q
    
def rutDepthVN(ci, bd, mineral=1, peat=0, para=None, cf=0, pc=3, passes=1, vehicle=None):
    """
    Semi-empirical rut depth model (V-N eq. 6 & 7, 9)
    IN:
        ci - cone index [kPa], array
        bd - bulk density [kgm-3], array
        mineral = 1 if mineral soil
        peat = 1 if peat soil
        para - empirical parameters,array Nx
        pc - multi-pass exponent
        cf - coarse fragment ratio
        passes - nr. of vehicle passes.
        vehicle - dict of vehicle properties, keys are:
            W - mass [kg]
            hei - tire section height [m]
            dia - tire diameter [m]
            pres - tire inflation pressure [kPa]
            wheels - number of wheels [-]
    OUT:
        z - rut depth [m]
        nci - dimensionless wheel numeric (ci/wheel-specific load corrected by wheel-soil contact area)
    """
    p=np.empty(4)
    if para is None: # use nominal; V-N table 4, 3nd equation with added constant
        p[0]= 0.0; p[1]=198.0; p[2]=1.6; p[3]=2.5;
        #z=p[0] + p[1]*(1 + p[2]*mineral +p[3]*peat)/nci [mm]
    else:
        p=np.array(p, ndmin=1)
        cf=np.array(cf, ndmin=1)
    if vehicle is not None:     
        W=1e-3*vehicle['W']*9.81/vehicle['wheels']; h=vehicle['hei']; d=vehicle['dia']; pres=vehicle['pres']
        n=0.5*vehicle['wheels']*passes #number of wheels passing   
    else: #use nominal vehicle from V-N, assume empty, 15ton 8wheeler -->W=15e3/8*9.81
        W=18.5; h=0.533; d=1.83; pres=220; n=4*passes
        
    gamma= 1.0e-3*(0.365 + 170.0/pres) #tyre deflection  [m]
    
    nci= 1.0e3*ci*bd/W*np.sqrt(gamma/h)* 1.0/ (1.0 + 2.0*d) #normalized cone index
    
    # single-pass rut depth [m]. V-N eq. 6, last term is effect of coarse fragment ratio cf (from eq. 13)
    z1=1.0e-3*(p[0] + p[1]*(1 + p[2]*mineral +p[3]*peat)/nci )*(1-cf)**2 # single-pass rut depth [m]
    
    #multi-cycle rut depth, simple assumption zn=z1*n^(1/a)
    #Saarilahti, 1991. Maastoliikkuvuuden perusteet p. 47, eq. 6.8:
    #pc=2-3 loose soils or light load
    #   3-4 medium bearing soils or medium load
    #   4-5 hard surface or high load
    
    z=z1*n**(1.0/pc)
    
    return z, z1, nci

def frostNciVN(fd, W, S, nci=1):
    """
    frost depth fd impact of wheel numeric nci. V-N eq. 14\n
    IN:
        fd - frost depth [m]
        W - vehicle mass [kg]
        S - soil saturation ratio, fraction of pore space [-]
        nci - wheel numeric, optional
    OUT:
        fnci - wheel numeric or relative wheel numeric (nci=1) modified by frost effect
    """
    W=1e-6*W*9.81 # in MegaNewton
    k=0.81*S

    a=np.maximum(eps, (1.0 - np.divide(k*np.power(fd,2), W) ) )
    
    nfci= np.divide(nci,a)
    #print fd, a, nfci
    return nfci

def frostMaximumLoad(fd, sc=None):
    """
    Maximum load bearing capacity of frozen ground (CRMM-manual eq. 14-15 based on Rummukainen 1984)\n
    IN:
        fd - frost depth [m]
        Sc - 1.0 uses model for saturated soil
    OUT:
        lm - maximum load bearing capacity [kg]
    NOTE: very coarse approximation considering the data used to constrain the 'model'
    """

    if sc is None:
        lm = 0.35*np.power(fd,2)
    else: #use saturated-soil model
        lm = 0.86*np.power(fd,2)
    print lm
    lm = 1e6*lm/9.81 #MN --> kg
    
    return lm
    
def ctv(hw, k, p, xlow, xhi, cf=None, draw=None):
    """
    Cross-terrain variations of soil properties computed from depth-to-water index (dwt)
    V-N eq. 15
    IN:
        hw - cartographic depth-to-water index [m]
        k - shape parameter [-]
        p - shape parameter [-]
        xlow - variable value at dwt=0 (low elevation)
        xhi - variable value at high elevation dwt_h=max(dwt)
        cf - 'y' if variable is 'coarse fragments', which amount increase with dtw 
        draw - 'y' draws figure
    OUT:
        f - scaled x: x(0)=x0, x(dwt_h)=xh
    """
    
    hw=np.array(hw,ndmin=1)
    hm=np.amax(hw)
        
    if cf is None: #generally values decrease from low to high elevations
        f= xlow - (xlow - xhi)*( (1- np.exp(-k*hw)) / (1 - np.exp(-k*hm)) )**p
    else: #coarse fragment ratio increases towards high elevations
        f= xlow - (xhi- xlow)*( (1- np.exp(-k*hw)) / (1 - np.exp(-k*hm)) )**p
    
    if draw is not None:
        plt.figure()
        plt.plot(hw,f, 'o'); plt.title('ctw:'); plt.xlabel('h_w [m]'); plt.ylabel('f')
    return f
    
    

"""
Penetration resistance / ConeIndex - functions for different soils, from different sources etc.

"""

def coneIndexFasst(wg, ci0=1, a=None, stype=None, draw=None):
    """
    returns soil penetration resistance (cone index CI) in [units]. Exponential decay of ci with wg.
    IN:
        wg - grav. water content (%)
        ci0 - dry soil cone index [units]; default is 1 
        a - parameter, for decay with wg, a>0
        stype - soil type ID if USsoil literature values are used
        draw - 'y' draws wg vs. ci figure
    OUT:
        ci - cone index [units]. set ci0=1 to get relative ci
    SOURCE: Frankenstein & Koenig (2004). Fast All-season Soil STrength (FASST). 
        Cold Regions Research and Engineering Lab. US Army Corps of Engineers. ERDC/CRREL SR-04-1
    """ 
    #US soil types, FASST (Frankenstein et al 2004 Table 5.2.1)
    # For a<0 the model predicts increase of CI with wg.
    fasst={'SW': [3.987,	0.815], 'SP': [3.987, 0.815],'SM': [8.749,-1.1949],'SC': [9.056,-1.3566], 
    'ML': [10.225,-1.565], 'CL': [10.998,-1.848],'OL':[10.977,-1.754],'CH': [13.816,-5.583],
    'MH': [12.321,-2.044],'OH': [13.046,-2.172],'SMSC': [9.056,-1.3566], 'CLML': [9.454,-1.3850],
    'EV': [3.987, 0.815]}
    
    
    wg=np.array(wg)
    #select from parameter dict    
    if a is None and stype is not None:
        p=fasst[stype]
        a=p[1]        
        if ci0 is not 1: ci0=np.exp(p[0])
                        
    ci=ci0*np.exp(-a*np.log(wg))

    if draw is not None: #draw figure
        x=np.linspace(1,200,100)
        y=ci0*np.exp(-a*np.log(x))
        
        plt.figure()
        plt.plot(x,y,'r-'); plt.title('coneIndex: '+str(ci0)+' exp(-' +str(a)+'ln(x))')
        plt.xlabel('W_{grav}'); plt.ylabel('CI')
        
    return ci
    
def coneIndexAP(bd=None,w=None,C=None, tablevals='s1'):
    """
    Soil penetration resistance, cohesion soils, Ayers & Perumbal (1982)
    IN:
        bd - soil bulk density [g/cm3]
        w - grav. wat content [%]
        C - parameter list
        tablevals - for testing use A & P parameters for following soil types:
            s1 - 100% clay
            s2 - 75% clay, 25% sand
            s3 - 50% clay, ,50% sand
            s4 - 25% clay, 75% sand
            s5 - 100% sand
    OUT:
        q - penetration resistance (cone index) [kPa]
    NOTE: some param0 are weird; give counter-intuitive results.
    """
    param0={'s1': [4540.9, 31.94, 9.21, 6.37], 's2': [928.1, 20.22, 7.41, 6.60], 
            's3': [82.4, 9.47, 4.77, 7.50], 's4': [1.1, 2.19, 3.29, 9.34], 
            's5': [1.58, 17.72, 5.54, 8.92]}
    
    bd=np.array(bd, ndmin=1); w=np.array(w, ndmin=1)
    
    if C is None: C=param0[tablevals]; print 'q_AyePeru: using param0'
    print C
    q=C[0]*bd**C[3] / (C[1] + (w -C[2])**2) #kPa
    
    return q
    
def coneIndexWitney(w, fclay, bd):
    """
    Soil penetration resistance, Witney et al. (1984), ref. Saarilahti 2002 Soil Interaction model
    IN:
        w - grav. wat content [%]
        fclay - clay ratio per volume [-], clay/other components
        bd - soil bulk density [kg/m3]. Note: 1 g/cm3 = 1000 km/m3
    OUT:
        ci - cone index (kPa)
    NOTE: inputs can be scalars or NxM arrays. Uses elementwise operations so scalars are treated as constants.
    """
    bd=np.array(bd, ndmin=1); w=np.array(w, ndmin=1); fclay=np.array(fclay, ndmin=1)
    
    a=np.divide(np.pi, (1 + fclay))
    ci=1e3*(15.92*fclay*np.exp(-0.08*w) + 2.58e-5*bd*np.exp(a)) #kPa
    
    return ci

def coneIndexPeat(w, vpscale=None):
    """
    Soil penetration resistance for peat soils, Amarjan (1972) & Saarilahti (2002)
    IN:
        w - grav. wat content [%], can be very high for peat soils
        vpscale - vonPost humification (decomposition) scale
    OUT:
        ci - cone index (kPa)
    NOTE: inputs can be scalars or NxM arrays. Uses elementwise operations so scalars are treated as constants.
    """   
    
    if vpscale is None: #use Saarilahti bulk equation
        ci = 10504.0*np.power(w,-0.561)
    else:
        ci = np.divide(2500.0, w)*(100.0 - 3.60*np.power(vpscale, 1.414))
    
    return ci

"""
Soil shear strength functions
"""

def shearStrengthCoulomb(c, p, ang=None, stype=None):
    """
    Soil shear strength, classic Coulomb's formula
    IN:
        c - soil cohesion [kPa=kNm-2]
        p - pressure (load) [kPa]
        ang - soil friction angle [deg]
    OUT:
        tau - shear strength [kPa]
    """
    
    ang=ang/180.0*np.pi #deg --> rad  
    tau= c +np.tan(ang)
    return tau

def shearStrengthPeat(w, vpscale=5):
    """
    Shear strength of peat following Amarjan (1972)
    IN:
        w - grav. wat content [%], can be very high for peat soils
        vpscale - vonPost humification (decomposition) scale. Default is 5
    OUT:
        tau - tau - shear strength [kPa]
    NOTE: inputs can be scalars or NxM arrays. Uses elementwise operations so scalars are treated as constants.
    """
    tau = np.divide(140.0, w)*(100.0 - 2.83*np.power(vpscale, 1.41))
    return tau

"""
Rut-depth models

"""
def rutDepthML(nci, d, n=1, a=2):
    """
    Rut depth by Maclaurin (1990), multi-pass as Saarilahti (1991)
    IN:
        nci - Maclaurin wheel numeric [-], scalar or NxM array
        d - tyre diameter [m]
        n - number of wheel passes [-]
        a - multi-pass coefficient
    OUT:
        z - rut depth [m]       
    NOTE: multi-cycle rut depth, simple assumption zn=z1*n^(1/a)
        Saarilahti, 1991. Maastoliikkuvuuden perusteet p. 47, eq. 6.8:
        a=2-3 loose soils or light load
        3-4 medium bearing soils or medium load
        4-5 hard surface or high load
    ALL OPERATIONS ELEMENT-WISE!
    """
    
    z = d*np.divide(0.224, np.power(nci,1.25)) #m
    
    #multi-pass
    if n>1:
        #z=z*n**(1.0/a), next row is same but for array inputs
        z=z*np.power(n, np.divide(1.0,a)) 
    
    return z