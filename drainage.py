# -*- coding: utf-8 -*-
"""
Created on Wed Sep 14 09:20:15 2016

@author: slauniai
"""
import numpy as np
import matplotlib.pyplot as plt
eps = np.finfo(float).eps #machine epsilon 

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
    print Hdr
    if Hdr>0:
        print 'Hooghoud'
        Trans=Ksat*dz # transmissivity of layer, m2s-1
    
        #-------- drainage from saturated layers above ditch base
        #ix=np.where( (zs-GWL)<0 and zs>-DitchDepth) #layers above ditch bottom where drainage is possible
        ix1=np.where( (zs-GWL)<0)
        ix2=np.where(zs>-DitchDepth) #layers above ditch bottom where drainage is possible
        ix=np.intersect1d(ix1,ix2); del ix1, ix2  
        print Trans[ix]
        Ka=sum(Trans[ix])/Hdr #effective hydraulic conductivity ms-1
        print sum(Trans[ix])
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
        print Ka, Kb
        
        #compute equivalent depth Deq
        A=3.55 -1.6 * Zbot / DitchSpacing -2 * ( 2 / DitchSpacing )**2;
        Reff=DitchWidth/2 #effective radius of ditch
        Dbt=-Zbot +DitchDepth #distance from impermeable layer to ditch bottom

        if Dbt/DitchSpacing <=0.3:
            Deq= Zbot / ( 1 + Zbot / DitchSpacing * (8 / np.pi * np.log( Zbot / Reff) - A)) #m
        else:
            Deq=np.pi * DitchSpacing / (8 * np.log( DitchSpacing/Reff ) - 1.15) #m
    
        print Deq
        Qb=8 * Kb * Deq * Hdr / DitchSpacing**2 # m s-1, total drainage below ditches
        Qz_drain[ix]= Qb * Trans[ix] / sum(Trans[ix]) / dz[ix] #sink term s-1
        
        del ix
        
    Q=Qa + Qb #total drainage m s-1, positive is outflow to ditch
    print Hdr, Qa, Qb, Q    
        #plt.figure(1)
        #plt.subplot(121); plt.plot([0 1],[GWL GWL],'r-',[0 1],[-DitchDepth -DitchDepth],'k-')
        #plt.subplot(122); plt.plot(Qz_drain,zs,'r-'); ylabel('zs'), xlabel('Drainage profile s-1');
    return Q, Qz_drain

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
    
def find_index(a, func):
    """ 
    finds indexes or array elements that fill the condition
    call as find_index(a, lambda x: criteria)
    """
    return [i for (i, val) in enumerate(a) if func(val)]        