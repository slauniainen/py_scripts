# -*- coding: utf-8 -*-
"""
Created on Tue Dec 13 18:13:46 2016

@author: slauniai
"""
import numpy as np
import pandas as pd
import os
import configparser
import matplotlib.pyplot as plt
#from scipy import interpolate

eps = np.finfo(float).eps #machine epsilon

    
class Preles(object):
    """
    Preles GPP and ET equations from Preles-model (Peltoniemi et al., 2015 Bor. Env. Res. 20, 151-170).
    IN:
        self - object
        
    """
    import BucketGrid
    def __init__(self, preles_ini, LAI=None):
            
        #extract parameters

            #GPP & ET model parameters
            self.ppara=pp['GPP'] #gpp-parameter dictionary.
            self.etpara=pp['ET'] #et-parameter dictionary
            
            #water bucket parameters
            self.bupara=pp['Buckets']
            
            #structural parameters
            if LAI is None: LAI=pp['General']['lai']
            
            self.LAI=LAI
            self.kext=pp['General']['kext']
            self.fapar=1.0 - np.exp(-self.kext*self.LAI) #fapar [-] 

            #initial state variables
            
            #acclimation            
            self._x=np.ones(np.shape(LAI))*inistate['x']
            self.S=np.maximum(X-self.ppara['xo'], 0.0)
            
            #buckets
            self.Wsur=np.ones(np.size(self.LAI))*inistate['wsurf']
            self.SWE=np.ones(np.size(self.LAI))*inistate['swe']
            self.Wsoil=np.ones(np.size(self.LAI))*inistate['wsoil']
            
            """self.ppara.keys():            
            #betap potential LUE gC mol-1 m-2(ground) 
            #gamma mol-1 m-2, light response param
            #tau #delay parameter of seasonal acclimation
            #smax deg C, threshold for full acclimation
            #x0 degC, acclimation base temperature
            #kappa kPa-1, gpp vpd sensitivity parameter, <0
            #rhop (-), gpp Rew sensitivity param
            """
            
            """self.etpara.keys():
            #lambda, - transpi vpd sensitivity
            #nu, - transpi Rew sensitivity
            #alpha, - transpi scaling param mm (gCm-2 kPa^(1-lambda))^-1
            #ksi, - evap scaling param mm mol-1  
            """
            
            """self.bupara.keys():
            D - effective soil depth (mm)
            fc - field capacity (m3m-3)
            wp - wilting point (m3m-3)
            tau - drainage parameter (d)
            
            wsur_max - maximum surface storage (mm); should be function of LAI
            kmelt - snow melt coeffcient 
            """
            
            
            
                        
        
        def run(self, Par, VPD, T, Rew=1.0):
            """runs preles for given forcing"""
            
        def gpp_transpi_evap(self,Par,VPD,Rew=1.0, S=None):
            """ 
            Computes GPP, transpiration and evaporation
            Peltoniemi et al. 2015 eq. 8-13
            IN:
                self - object
                Par - photos. act. radiation (mol m-2, daily sum?)
                VPD - vapor pressure deficit, daily mean (kPa)
                Rew - relatively extractable water (-)
                S - stage of acclimation, included here so that we can run vectors
            OUT:
                p - gross-primary productivity (gCm-2d-1)
                tr - transpiration (mm d-1 = kgm-2d-1)
                evap - evaporation (mm d-1)                
            """
            
            fL, fD, fWP, fWE, fS = self._modifiers(X,Par, VPD, Rew=Rew, S=S)

            p=self.ppara['beta']*self.fapar*Par*fL*fS*np.minimum(fD,fWP) #gCm-2d-1
            
            tr=self.etpara['alpha']*p*fWP**self.etpara['nu']*VPD**(1-self.etpara['lambda']) #mmd-1
            evap=self.etpara['ksi']*(1-self.fapar)*Par*fWE #mmd-1
            
            return p, tr, evap
        
#        def waterbalance(self, T, Prec, Psnow=0.0, tr=0.0,evap=0.0):
#            """computes water balance of three buckets"""
#            if np.size(T)==1: T=np.ones(np.shape(self.LAI))*T
#            
#            #canopy water storage change
#            
            
            
            
        def _acclim(self, T):
            """ compute new stage of temperature acclimation"""            
            self._x = self._x +  1.0/self.ppara['tau']*(T - self._x) #degC
            self.S=np.maximum(X-self.ppara['xo'], 0.0)
            return self.S
            
        def _modifiers(self,Par,VPD, Rew=1.0, S=None):
            """ returns modifier functions (0...1)"""
            
            fL = 1.0/(self.ppara['gamma']*Par +1.0) #light
            fD = np.exp(self.ppara['kappa']*VPD) #vpd
            fWP = np.minimum(1.0, Rew/self.ppara['rhop']) #soil water, gpp            
            fWE = np.minimum(1.0, Rew/self.etpara['rhoe']) #soil water, transpi
            
            #temperature acclimation
            if S is None: S=self.S
                
            fS = np.minimum(S/self.ppara['smax'], 1.0)
            
            return fL, fD, fWP, fWE, fS
            

""" utility functions """
            
    def clear_console():
        """
        clears Spyder console window - does not affect namespace
        """
        import os
        clear = lambda: os.system('cls')
        clear()
        return None
        
    def read_ini(inifile):
        """read_ini(inifile): reads Spathy.ini parameter file into pp dict"""
        
        cfg=configparser.ConfigParser()
        cfg.read(inifile)
    
        pp = {}
        for s in cfg.sections():
            section=s.encode('ascii','ignore')
            pp[section] = {}
            for k, v in cfg.items(section):
                key=k.encode('ascii','ignore')
                val=v.encode('ascii','ignore')
                if section == 'General': #'general' section
                    pp[section][key] = val
                else:
                    pp[section][key] = float(val)      
        pp['General']['startyear']=int(pp['General']['startyear'])
        pp['General']['endyear']=int(pp['General']['endyear'])
        pp['General']['dt']=float(pp['General']['dt'])
        
        return pp
    
    def read_soilprop(soilparamfile):
        """ read_soilprop(soilparamfile): reads soiltype-dependent hydrol. properties from parameter file """
        cfg=configparser.ConfigParser()
        cfg.read(soilparamfile)
        #print cfg
        pp = {}
        for s in cfg.sections():
            section=s.encode('ascii','ignore')
            pp[section] = {}
            for k, v in cfg.items(section):
                key=k.encode('ascii','ignore')
                val=v.encode('ascii','ignore')
                
                if key == 'gtk_code':#and len(val)>1:                
                    val=map(int, val.split(','))    
                    pp[section][key]=val
                else:
                    pp[section][key] = float(val)
        del cfg
        
        return pp