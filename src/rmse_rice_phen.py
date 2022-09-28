# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plot
#from pylab import *
import Sun
import datetime
from collections import defaultdict

pd.options.display.max_columns = 999


def Wang_engle(T, Tbase=8, Topt=30, Tcei=42):
    '''
    Wang and Engle 1998, Agircultual systems. 
    '''
    thermal = 0
    if T <= Tbase or T >= 42:
        return 0
    else:
        alpha = math.log(2, ) / (math.log((Tcei - Tbase) / (Topt - Tbase)))
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


def photo_period_based_on_yin(dl, mu=-15.46, zeta=2.06, ep=2.48):
    '''
    Yin 1997 beta function
    the effect of photoperiod on development rate
    '''
    photo = math.exp(mu) * (dl) ** zeta * (24 - dl) ** ep
    photo = photo / (max([math.exp(mu) * (dl) ** zeta * (24 - dl) ** ep for dl in np.linspace(1, 24, 100)]))
    return photo


def Test_wang_engle():
    plot(range(0, 45), [Wang_engle(T=T) for T in range(45)])
    plot.show()


def Test_Yin_photo():
    plot(range(1, 24), [photo_period_based_on_yin(dl=dl) for dl in range(1, 24)])
    plot.show()


def read_station_weather(station, start, end):
    df = pd.read_table("C:/TanJiaojiao/Meteo(48 sta)/Meteo(48 sta)/" + str(station) + ".txt", encoding='gbk', sep=' * ', engine='python',
                       skiprows=[1])
    df['Date'] = df.apply(lambda row: pd.to_datetime('%d-%d-%d' % (row.YY, row.mm, row.dd)), axis=1)
    df = df.loc[(df.Date >= start) & (df.Date <= end), ['Date', 'TemAver']]
    return df


def photo_effect_correct(today, revd, jd, hd, photo):
    if pd.isna(jd):
        jd = revd + datetime.timedelta(days=25)
    if pd.isna(hd):
        jd = revd + datetime.timedelta(days=55)
    if today < jd or today > hd:
        return 1
    else:
        return photo

def date_period(time1,time2):
    a1 = time1
    a2 = time2
    b1 = a1.split('-')
    b2 = a2.split('-')
    c1=datetime.date(int(b1[0]), int(b1[1]), int(b1[2]))   #开始的日期de
    c2=datetime.date(int(b2[0]), int(b2[1]), int(b2[2]))  #  结束的日期
    result=(c2-c1).days
    return result

def Trial_Sim():
    '''
    station, ID, lat, lon, date, stage
    join with weather by station id and date
    calculate daily thermal, daily photoperiod,photothermal= thermal*photo
    cumumlate thermal, photothermal from regrowth to maturation
    
    '''
    sun = Sun.Sun()
    df = pd.read_csv('C:/TanJiaojiao/obser_pheno_catalog.csv', encoding="GBK",
                     parse_dates=['reviving data', 'tillering data', 'jointing data', 'booting data', 'heading data',
                                  'maturity data'])
    print("df columns", df.columns)
    dfmm = df[['station ID', 'lat', 'lon', 'alt', 'reviving data',
               'tillering data', 'jointing data', 'booting data', 'heading data',
               'maturity data']]
    dfm = pd.melt(dfmm, id_vars=['station ID', 'lat', 'lon', 'alt'], value_vars=['reviving data',
                                                                                 'tillering data', 'jointing data',
                                                                                 'booting data', 'heading data',
                                                                                 'maturity data'])
    dfm = dfm.rename(columns={'value': 'Date', 'variable': 'DStage'})
    print("dfm", dfm)
    dfall = pd.DataFrame()
    for ind, row in df.iterrows():
        dfw = read_station_weather(row['station ID'], row['reviving data'], row['maturity data'])
        #print("dfw", dfw)
        dfw['Thermal'] = dfw.TemAver.apply(lambda x: Wang_engle(T=x))
        dfw['Thermal_cum'] = dfw.Thermal.cumsum()
        dfw['dayL'] = dfw.Date.apply(
            lambda x: sun.dayCivilTwilightLength(year=x.year, month=x.month, day=x.day, lon=row.lon, lat=row.lat))
        dfw['photo_raw'] = dfw.dayL.apply(lambda x: photo_period_based_on_yin(dl=x))
        dfw['photo'] = dfw.apply(lambda rowt: photo_effect_correct(today=rowt.Date, revd=row['reviving data'], jd=row['jointing data'],
                                              hd=row['heading data'], photo=rowt.photo_raw), axis=1)
        dfw['photothermal'] = dfw.photo * dfw.Thermal
        dfw['photothermal_cum'] = dfw['phototherFmal'].cumsum()
        dfw['station ID'] = row['station ID']
        dfall = pd.concat([dfall, dfw])
        print(ind)
    print("dfall columns", dfall.columns, "dfm columns", dfm.columns)
    dfall = dfall.set_index(['station ID', 'Date']).join(
        dfm[['station ID', 'Date', 'DStage']].set_index(['station ID', 'Date'])).reset_index()
    dfall.to_excel('C:/TanJiaojiao/dfall.xlsx', index=False)

    print("df columns", df.columns)

    print('Finised')


def RMSE():
    '''
    thermal_cum >= Photothermal_cum
    
    Parameters
    ----------
    dfall : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    '''

    F2=defaultdict(list)
    
    phenophases = ['tillering data', 'jointing data',
                   'booting data', 'heading data', 'maturity data']
    
    df = pd.read_excel('D:/workspace/rice_crop_model/data/dfall.xlsx')       # calculated thermal/photothermal from Trial_Sim
    
    dfall = pd.DataFrame()
    phenoDate = []
    
    for phenophase in phenophases:
        dfthermal = df[df['DStage']==phenophase]
        dfthermal = dfthermal.reset_index(drop=True)
        f1 = defaultdict(list)
        #f1 = pd.DataFrame()
        for idx in range(dfthermal.shape[0]):
            DT = dfthermal['Thermal'][idx]
            DTT = dfthermal['Thermal_cum'][idx]
            startdate = dfthermal['Date'][idx]
            ID = dfthermal['station ID'][idx]
            
            for ind, row in df.iterrows():
                if row['DStage'] == 'reviving data':          # The begining of a specific phenology
                    F2['Station ID'] = row['station ID']
                    n = 0
                    period = 0
                    Thermal_sum = row['Thermal']
                    PhotoThermal_sum = row['photothermal']
                    errDay = 0
            
                else:
                    period += 1
                    Thermal_sum += row['Thermal']
                    PhotoThermal_sum += row['photothermal']
                    if Thermal_sum < DTT:
                        n += 1 
                    
                    if row['DStage'] == phenophase:              
                        errDay = period - n  
                        n = period 
                        F2['errDay'].append(errDay)

                        #break
                        
            errDay = np.array(F2['errDay']) 
            rmse = np.sqrt(np.mean(errDay) ** 2)
            f1['station ID'].append(ID)
            f1['Thermal'].append(DT)
            f1['Thermal_cum'].append(DTT)
            f1['rmse'].append(rmse)
            #Dfall = pd.concat([dfall, f1]) 
            #print('phenophase', phenophase)
            #print('err Day', errDay)
            #print('rmse', rmse)
        
        rmse_np = np.array(f1['rmse'])
        index = np.argsort(rmse_np)
        accuthermal = dfthermal['Thermal_cum'][index[0]]
        print(accuthermal)
        
    f1.to_excel('D:/Python code/Dfallrmse.xlsx', index=False)
    print('n', n)
    print('period', period)
    print('err days', errDay)
    print('F2', F2)
    errDay = np.array(F2['maturity data'])
    rmse = np.sqrt(np.mean( (errDay)**2))
    print(rmse)
 


if __name__ == '__main__':
    #Trial_Sim()
    RMSE()
    # print(Photo)
