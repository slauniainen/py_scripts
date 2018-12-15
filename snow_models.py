# -*- coding: utf-8 -*-
"""
Created on Fri Apr 22 12:48:04 2016
snow -related model codes

@author: slauniai
"""

import numpy as np
import matplotlib.pyplot as plt

def snow17(Prec, Tair, Params, IniState, dt=1.0):
    """
    Point implementation of Snow accumulation and ablation model SNOW-17 (Anderson, 1973 & 2006).
    """
    
    #--constantas
    #Lf =80.0 # lat. heat of fusion (cal/gm)
     #ci=0.5 # specific heat of ice [cal/gm/degC]
    Lf=334720.0 #J/kg
    ci=2050.0 #J/kg/K-1 at 0.0degC 
    sigma=6.12e10 #stefan-boltzman constant (mm/K/hr????)
    
    
    #retrieve params for param dict
    PXTEMP = Params[0] #  % Threshold temperature for rain/snow transition [deg. C]
    SCF    = Params[1] #  % Snow gage under-catch efficiency factor [-]
    UADJ   = Params[2] #   Avereage wind function [mm/mb/6 hr]
    NMF    = Params[3] #  % Maximum negative melt factor [mm/deg C/6 hr]
    MFMIN  = Params[4] # % Minimum melt factor [d/l]
    MFMAX  = Params[5] # % Maximum melt factor [d/l]
    MBASE  = Params[6] # % Base temperature for calculation of non-rain melt [deg. C]
    TIPM   = Params[7] #  % Model parameter [d/l]
    PLWHC  = Params[8] #  % Percent liquid water holding capacity of the snow pack [%]
    DAYGM  = Params[9] # Daily rate of melt at the snow-ground interface [mm/day]
    LatDeg = Params[10]  # Site latitude in degrees [deg. Latitude]
    He     = Params[11] # Elevation of the site [m]
 

    #----- get initial state
    Wice0=IniState[0]    #total ice content in snow [mm]
    Wliq0=IniState[1]    #total liq. water content in snow [mm]
    
    #----- length of input forcing & initialize output vectors---
    N=len(Prec)
    
    #Initialize storage containers
    SWE           = np.zeros(N)
    SWEi          = np.zeros(N)
    SWEq          = np.zeros(N)
    Outflow       = np.zeros(N)
    SnowDepth     = np.zeros(N)
    SnowfallDepth = np.zeros(N)
    
    for k in range(0,N): #loop in time
    
        Wi0=Wi.copy()
        Wq0=Wq.copy()
        ATI0=ATI.copy()
        U0=U.copy() #cold content
        rho_x=rho_x.copy()

        fr, fs = precipForm(Tair) #rain, snow fractions
        Pr=fr*Prec; Ps=fs*Prec
        
        rhon, Hn, dUn = newSnow(Tair, Ps) #density, depth and cold content of new snow
        
        #update ice content of snowpack
        Wi +=Ps
        
        #compute snow melt (mm)
        Mr = snowMelt(dt, Ta, Pr, Mcoef, UADJ=0.0)
        Mr=np.minimum(Wi, Mr)
        
        #new ice content and potential liquid flux after melt
        Wi -=Mr
        
        Qin = Mr + Pr #net liquid water flux to snowpack (melt + rain)
        
        #thermal state of snowpack [degC]
        #Antedecent temperature index ('snowpack temparature')
        ATI=ATI0 + (1.0 - (1.0-TIMP)**(dt/6.0))*(Tair - ATI0)
        if Mr>0: ATI=0.0
        elif Ps>1.5*dt: ATI=np.minimum(Tair, 0.0)
        #ATI[Mr>0.0]=0.0 #in melting snow T=0.0
        #ATI[Ps>1.5*dt]=np.minimum(Tair[Ps>1.5*dt], 0.0) #if lot of snowfall
    
        
        #NMCoef = negative melt coefficient
        #NMcoef = NMF*dt/6.0 * Mfact/MFMAX
        dUt=NMcoef*(ATI-np.minimum(Tair, 0.0)) #cold content change due to temperature change

        #update cold content
        U = U0 + dUn + dUt
        
        U=min(max([U,0.0]), 0.33*Wi)
        
        #compute liquid outflux and update state variables
        Wqx=PWLCH*Wi #liq. storage capacity
        
        E=0.0 #outflux
        if Wi>0:        
            if Qin>=U+Wqx: #case Qin is sufficient to fill cold content and liquid storage
                Wq = Wqx
                E = Qin - (1.0 + PLWHC)*U
                Wi += U
                Qf=U
                #ATI=0.0
            elif Qin>U and Qin<Wqx+U:
                Wq = Wq+Qin -U
                Wi += U
                Qf=U
            else:
                U -= Qin
                Wi += Qin
                Qf=Qin
        else:
            Wi=0.0
            Wq=0.0
            U=0.0
            Qf=0.0
            E=Qin.copy()

        del Qin
        
        #-----compute new snow temperature, density (rho_x) and depth (H) 
        
        
        if Wi0==0.0 and Ps>0.0: 
            rho_x=rhon.copy()
        else:
            rho_x=snowDensityChange(dt, Ts, Wi0, Wq0, Qf, rho_x)
                    
        if Wi==0: H=0.0
        else: H=Wi/rho_x
        
            
    
def newSnow(T, Ps=0.0):
    """
    density, depth and cold content of new snow (Anderson, 1976)
    IN:
        T - air temperature (degC)
        Ps - water equivalent of falling snow (mm)
    OUT:
        rhos - snow density (kg m-3): 1000 kgm-3 = 1g cm-3
        h - depth of falling snow (m)
        dU - heat deficit (mm)
    """
    
    Lf=334720.0 #J/kg
    ci=2050.0 #J/kg/K-1 at 0.0degC 
    
    T=np.array(T, ndmin=1)
    
    rhos=50.0 + 1.7*T**1.5
    rhos[T<-15.0]=50.0 #T below -15degC
    
    h= Ps/rhos #m
    
    T=np.minimum(T,np.zeros(len(T)))
    dU = -(T*Ps) / (Lf/ci) #mm
    
    return rhos, h, dU

def precipForm(T, Tlow=0.0, Tup=+2.0):
    """
    form of precipitation depends on air temperature
    returns: fr - liquid fraction, fs - snow fraction
    """
    T=np.array(T, ndmin=1)
    
    fr=(Tup-T)/(Tup-Tlow)
    fr[T>Tup]=0.0; fr[T<Tlow]=1.0
      
    fs=1.0-fr
    return fr, fs

def snowMelt(dt, Ta, Pr, Mf, UADJ=4e-4, MBASE=0.0, pressure=1013e2):
    """
    Snow melt approximation of Andersson 1972
    IN:
        dt - timestep [h]
        Ta - air temperature [degC]
        Pr - rain accumulated
        melt factor [mm K-1 dt-1]
        UADJ - wind correction function [mm Pa-1]. From Anderson 2006 one can calculate UADJ ~4e-4 mm/Pa
        MBASE - threshold temperature for snowmelt [degC]
        pressure - atmospheric pressure [Pa]
    OUT:
        mlt - snow melt (mm)
    """
    NT=273.15;
    sigma=0.0
    
    Ta=np.array(Ta,ndmin=1)
    Tr=np.maximum(Ta, np.zeros(len(Ta))) #T of rain
    es,_=e_sat(Ta)
    
        
    #rain-on-snow melt 
    mlt = sigma*dt*( (Ta + NT)**4 - NT**4) + 1.25e-2*Pr*Tr + 8.5*UADJ*dt*( (0.9*es -611) + 5.7e-4*pressure*Ta)
    
    #non-rain melt
    ix=np.where(Pr/dt<0.25) #if precipitation rate less than 0.25mm/h use different algorithm
    
    mlt[ix]= Mf*(Ta - MBASE) + 1.25e-2*Pr
    mlt[mlt<0.0]=0.0 
    
    return mlt

def snowDensityChange(dt,Ts,Wi0,Wq0,Qf,rhos0, param=None):
    """
    Snow density changes due to compaction and destr. metamorphosis as in Anderson (2006) eq. 14 & 16
    IN:
        dt - timestep [h]
        Ts - snow temperature
        Wi0 - ice content (mm)
        Wq0 - liquid water content (mm)
        Qf  - amount of liq. water freezing during dt (mm)
        rhos0 - snow density (kg m-3)
        param - model parameters [list]
    OUT:
        x - new snow density (kg m-3)
    """
    
    rhos0=1e-3*rhos0 #g cm-3
    #-- snow densification model parameters
    if param is None:    
        c1=0.026; c2=21.0; c3=0.005; c4=0.10; cx=23.0; rhod=0.15; #'new values', Anderson (2006)
        #c1=0.01; c2=21.0; c3=0.01; c4=0.04; cx=46.0; rhod=0.15; #old values in Anderson (1976)
    else:
        c1=param[0]; c2=param[1]; c3=param[2]; c4=param[3]; cx=param[4]; rhod=param[5];
    
    if Wq0>0.0: 
        c5=2.0
    else:
        c5=0.0
    
    beta=0.0
    if rhos0>rhod: beta=1.0    

    B=c1*dt*np.exp(0.08*Ts - c2*rhos0)
    A=c3*c5*dt*np.exp(c4*Ts - cx*beta*(rhos0-rhod))
    
    x=1e3*rhos0* ( (np.exp(B*0.1*Wi0) -1.0) / B*0.1*Wi0)*np.exp(A) #new density
    x= x* Wi0/(Wi0+Qf) #melt metamorphosis (eq. 16)
    
    return x

def snowTemperature(dt,To, rhos, Wi, Wq, Ho, Hn, Tn, dTa):
    """
    approximates temperature of snowpack from change in air temperature (eq. 18-20)
    IN:
        dt - hours
        To - old snow temperature (degC)
        rhos - old snow density (kgm-3)
        Wi - ice content (mm)
        Wq - liquid content (mm)
        Ho - old snow depth (cm)
        Hn - new snowfall depth (cm)
        Tn - new snowfall temperature (degC)
        dTa - air temperature change during previous dt (degC)
    OUT:
        Tx - new snowpack temperature (degC)
    """    
    
    rhos=1e-3*rhos #g cm-3
    
    
    c=2.1e6*rhos + 1.0e3*(1.0-rhos-Wq/(Wi+Wq)) + 4.2e6*(Wq/(Wi+Wq)) #J m-3 K-1
    K = 0.0442*np.exp(5.181*rhos) #Wm-1K-1
    
    alpha=np.sqrt((np.pi*c) / (K*2.0*3600.0*dt))    
    
    Tx= To + dTa*( np.exp(-0.01*alpha*Hn)- np.exp(-0.01*alpha*Ho)) / (0.01*alpha*(Ho-Hn))
    
    Tx=(Tx*Ho + Tn*Hn) / (Ho + Hn) #weighted temperature
    
    return Tx

def e_sat(T):
    """
    Computes saturation vapor pressure with respect to free and flat water surface for given temperature T.\n
    INPUT:
        T - temperature (degC), scalar or array.\n
    OUTPUT: 
        esat - saturation vapor pressure (Pa)
        delta - slope of saturation vapor pressure curve (Pa degC-1)
    SOURCE:
        Campbell & Norman, 1998. Introduction to Environmental Biophysics.
    """
    
    
    #constants
    a=611 #Pa
    b=17.502 #(-)
    c=240.97 #degC
    
    esat=a*np.exp(b*T/(T+c)) #Pa
    delta = b*c*esat / ( (c + T)**2) #Pa degC-1
    
    return esat, delta