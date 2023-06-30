import numpy as np
import math

def photoeffect_yin(DL, mu, zeta, ep):
    '''
    Yin 1997 beta function
    the effect of photoperiod on development rate
    mu=-15.46, zeta=2.06, ep=2.48
    '''
    def yin_photo(DL=DL,mu=mu,zeta=zeta,ep=ep):
        return math.exp(mu) * (DL) ** zeta * (24 - DL) ** ep
    photo = yin_photo(DL=DL,mu=mu,zeta=zeta,ep=ep) 
    max_photo=max([yin_photo(DL=DLm,mu=mu,zeta=zeta,ep=ep) for DLm in np.linspace(1, 24, 100)])
    return photo/max_photo

def photoeffect_wofost(DL,Dc,Do):
    '''
    van Oort et al. 2011, Correlation between temperature and phenology prediction error in rice.  Do=10.5
    Jia mingzhong, 1989, Study on phototemperature reaction characteristics of Shanyou 36. Do=8-10h, To=24-27
    Liang guangshang, 1980, Study on the critical day length of panicle emergence in rice varieties. Dc= 12.5-14.5 h
    '''
    return(min(max(0, ((DL-Dc)/(Do-Dc))), 1))

def photoeffect_oryza2000(DL, Dc, PPSE=0.2):
    '''
    van Oort et al. 2011, Correlation between temperature and phenology prediction error in rice
    '''
    # MOPP = 12.5
    # PPSE = 0.2
    if DL < Dc:
        PPFAC = 1.
    else:
        PPFAC = 1. - (DL - Dc) * PPSE
        PPFAC = np.min([1., np.max([0., PPFAC])])
    return PPFAC
