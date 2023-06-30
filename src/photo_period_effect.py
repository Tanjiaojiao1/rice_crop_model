import numpy as np
import math
def photoeffect_yin(DL, mu=-15.46, zeta=2.06, ep=2.48):
    '''
    Yin 1997 beta function
    the effect of photoperiod on development rate
    mu=-15.46, zeta=2.06, ep=2.48
    '''
    def yin_photo(DL, mu=-15.46, zeta=2.06, ep=2.48):
        return math.exp(mu) * (DL) ** zeta * (24 - DL) ** ep
    
    photo = yin_photo(DL=DL,mu=mu,zeta=zeta,ep=ep) 
    max_photo=max([yin_photo(DL=DLm,mu=mu,zeta=zeta,ep=ep) for DLm in np.linspace(1, 24, 100)])
    return photo/max_photo
def photoeffect_wofost(DL,Dc=20,Do=12.5):
    def wofost_photo(DL,Dc=20,Do=12.5):
        return (DL-Dc)/(Do-Dc)
    photo = wofost_photo(DL=DL,Dc=Dc,Do=Do)
    max_photo=max([wofost_photo(DL=DLm,Dc=Dc,Do=Do) for DLm in np.linspace(1, 24, 100)])
    return photo/max_photo

def photoeffect_oryza2000(DL, MOPP=11.5, PPSE=0.2):
    '''
    van Oort et al. 2011, Correlation between temperature and phenology prediction error in rice
    '''
    # MOPP = 11.5
    # PPSE = 0.2
    if DL < MOPP:
        PPFAC = 1.
    else:
        PPFAC = 1. - (DL - MOPP) * PPSE
        PPFAC = np.min([1., np.max([0., PPFAC])])
    return PPFAC