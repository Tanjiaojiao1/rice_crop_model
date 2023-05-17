import numpy as np
import math
def T_base_opt(T, Tbase=8, Topt=30):
    '''
    Yin 1997 beta function
    the effect of photoperiod on development rate
    mu=-15.46, zeta=2.06, ep=2.48
    '''
    
    return np.interp(T,[Tbase,Topt],[0,Topt-Tbase])
def T_base_op_ceiling(T, Tbase=8, Topt_low=20,Topt_high=30, Tcei=42):

    return np.interp(T,[Tbase,Topt_low,Topt_high,Tcei],[0,Topt_low-Tbase,Topt_low-Tbase,0])

def Wang_engle(T, Tbase=8, Topt=30, Tcei=42):
    '''
    Wang and Engle 1998, Agircultual systems
    Tbase=8, Topt=30, Tcei=42.
    '''
    thermal = 0
    if T <= Tbase or T >= Tcei:
        return thermal
    else:
        alpha = math.log(2, ) / (math.log((Tcei - Tbase) / (Topt - Tbase)))
        thermal = (2 * ((T - Tbase) ** alpha) * (Topt - Tbase) ** alpha - (T - Tbase) ** (2 * alpha)) / (
                (Topt - Tbase) ** (2 * alpha))
        return thermal * (T - Tbase)