# -*- coding: utf-8 -*-
import pandas as pd
import numpy
import math
from pylab import *
import Sun
import datetime

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
    show()


def Test_Yin_photo():
    plot(range(1, 24), [photo_period_based_on_yin(dl=dl) for dl in range(1, 24)])
    show()


def read_station_weather(station, start, end):
    df = pd.read_table("../data/Meteo(48 sta)/" + str(station) + ".txt", encoding='gbk', sep=' * ', engine='python',
                       skiprows=[1])
    df['Date'] = df.apply(lambda row: pd.to_datetime('%d-%d-%d' % (row.YY, row.mm, row.dd)), axis=1)
    df = df.loc[(df.Date >= start) & (df.Date <= (end+datetime.timedelta(days=60))), ['Date', 'TemAver']]
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


def Trial_Sim():
    '''
    station, ID, lat, lon, date, stage
    join with weather by station id and date
    calculate daily thermal, daily photoperiod,photothermal= thermal*photo
    cumumlate thermal, photothermal from regrowth to maturation
    
    '''
    sun = Sun.Sun()
    df = pd.read_csv('../data/obser_pheno_catalog.csv', encoding="GBK",
                     parse_dates=['reviving data', 'tillering data', 'jointing data', 'booting data', 'heading data',
                                  'maturity data'])
    print(df.columns)
    dfmm = df[['station ID', 'lat', 'lon', 'alt', 'reviving data',
               'tillering data', 'jointing data', 'booting data', 'heading data',
               'maturity data']]
    dfm = pd.melt(dfmm, id_vars=['station ID', 'lat', 'lon', 'alt'], value_vars=['reviving data',
                                                                                 'tillering data', 'jointing data',
                                                                                 'booting data', 'heading data',
                                                                                 'maturity data'])
    dfm = dfm.rename(columns={'value': 'Date', 'variable': 'DStage'})
    print(dfm)
    dfall = pd.DataFrame()
    for ind, row in df.iterrows():
        dfw = read_station_weather(row['station ID'], row['reviving data'], row['maturity data'])
        dfw['Thermal'] = dfw.TemAver.apply(lambda x: Wang_engle(T=x))
        dfw['Thermal_cum'] = dfw.Thermal.cumsum()
        dfw['dayL'] = dfw.Date.apply(
            lambda x: sun.dayCivilTwilightLength(year=x.year, month=x.month, day=x.day, lon=row.lon, lat=row.lat))
        dfw['photo_raw'] = dfw.dayL.apply(lambda x: photo_period_based_on_yin(dl=x))
        dfw['photo'] = dfw.apply(lambda rowt: photo_effect_correct(today=rowt.Date, revd=row['reviving data'], jd=row['jointing data'],
                                              hd=row['heading data'], photo=rowt.photo_raw), axis=1)
        dfw['photothermal'] = dfw.photo * dfw.Thermal
        dfw['photothermal_cum'] = dfw['photothermal'].cumsum()
        dfw['station ID'] = row['station ID']
        dfall = pd.concat([dfall, dfw])
    print(dfall.columns, dfm.columns)
    dfall = dfall.set_index(['station ID', 'Date']).join(
        dfm[['station ID', 'Date', 'DStage']].set_index(['station ID', 'Date'])).reset_index()
    dfall.to_excel('../data/dfall.xlsx', index=False)

    print(df.columns)
def cal_thermal_requirement():
    df=pd.read_excel('../data/dfall.xlsx').dropna(subset=['DStage'])
    
    df=df[['Thermal_cum','photothermal_cum','DStage']].groupby('DStage').quantile('0.5')
    print (df)

if __name__ == '__main__':
    Trial_Sim()
    # print(Photo)
