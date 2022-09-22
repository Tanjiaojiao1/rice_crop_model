# -*- coding: utf-8 -*-
import pandas as pd
import numpy
import math
from pylab import *

pd.options.display.max_columns = 999


def Wang_engle(T=0, Tbase=8, Topt=30, Tcei=42):
    '''
    Wang and Engle 1998, Agircultual systems. 
    '''
    thermal = 0
    if T <= Tbase or T >= 42:
        return 0
    else:
        alpha = math.log(2, ) / (math.log((Tcei - Tbase) / (Topt - Tbase)))
        print(alpha)
        thermal = (2 * ((T - Tbase) ** alpha) * (Topt - Tbase) ** alpha - (T - Tbase) ** (2 * alpha)) / (
                    (Topt - Tbase) ** (2 * alpha))
        return thermal * (T - Tbase)
    return thermal * (T - Tbase)


def Photoperiod(Pdaily, DLBase, DLOptimum, DLCeiling, headingdate, today):
    '''
    Yin 1997
    return 0-1
    DLBase=8
    DLOptimum=10
    DLCeiling=12
    '''
    Photo = 1
    if today >= headingdate:
        return Photo
    else:
        Photo = ((Pdaily - DLBase) / (DLOptimum - DLBase) * (DLCeiling - Pdaily) / (DLCeiling - DLOptimum)) ** 5.6171
    return Photo


def Test_wang_engle():
    plot(range(0, 45), [Wang_engle(T=T) for T in range(45)])
    show()


def Trial_Sim():
    '''
    station, ID, lat, lon, date, stage
    join with weather by station id and date
    calculate daily thermal, daily photoperiod,photothermal= thermal*photo
    cumumlate thermal, photothermal from regrowth to maturation
    
    '''
    df = pd.read_csv('../data/Rice_phen_ob.csv', encoding="GBK")

    print(df.columns)


if __name__=='__main__':
    Trial_Sim()
    # print(Photo)
