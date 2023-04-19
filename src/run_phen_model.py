# -*- coding: utf-8 -*-
import pandas as pd
import numpy
import math
from pylab import *
import Sun
import datetime
import seaborn as sns
import matplotlib.pyplot as plt
import os
from rice_phen import read_station_weather,photo_effect_correct
from photo_period_effect import photoeffect_yin,photoeffect_oryza200,photoeffect_wofost
from T_dev_effect import Wang_engle,T_base_op_ceiling,T_base_opt
os.chdir(os.path.dirname(os.path.realpath(__file__)))

pd.options.display.max_columns = 999
def Simulate_all_sites(T_fun,Photo_fun,T_fun_para={},Photo_fun_para={}):
    '''
    station, ID, lat, lon, date, stage
    join with weather by station id and date
    calculate daily thermal, daily photoperiod,photothermal= thermal*photo
    cumumlate thermal, photothermal from regrowth to maturation
    
    '''
    sun = Sun.Sun()
    df = pd.read_csv('../data/obser_pheno_catalog.csv', encoding="GBK",
                     parse_dates=['reviving date', 'tillering date', 'jointing date',
                                  'booting date', 'heading date','maturity date'])
    df['season']= df.groupby(['station ID', 'year']).cumcount()+1
    dfmm = df[['station ID', 'lat', 'lon', 'alt', 'year', 'season',
               'reviving date', 'tillering date', 'jointing date',
               'booting date', 'heading date','maturity date']]
    dfm = pd.melt(dfmm, id_vars=['station ID', 'lat', 'lon', 'alt', 'year', 'season'],
                  value_vars=['reviving date', 'tillering date', 'jointing date',
                              'booting date', 'heading date', 'maturity date'])
    dfm = dfm.rename(columns={'value': 'Date', 'variable': 'DStage'})
    print(dfm)
    dfall = pd.DataFrame()
    for ind, row in df.iterrows():
        dfw = read_station_weather(row['station ID'], row['reviving date'], row['maturity date'])
        dfw['Thermal'] = dfw.TemAver.apply(lambda x: T_fun(T=x,**T_fun_para))
        dfw['Thermal_cum'] = dfw.Thermal.cumsum()
        dfw['dayL'] = dfw.Date.apply(
            lambda x: sun.dayCivilTwilightLength(year=x.year, month=x.month, day=x.day, lon=row.lon, lat=row.lat))
        dfw['photo_raw'] = dfw.dayL.apply(lambda x: Photo_fun(DL=x,**Photo_fun_para))
        dfw['photo'] = dfw.apply(lambda rowt: photo_effect_correct(today=rowt.Date, revd=row['reviving date'], jd=row['jointing date'],
                                              hd=row['heading date'], photo=rowt.photo_raw), axis=1)
        dfw['photothermal'] = dfw.photo * dfw.Thermal
        dfw['photothermal_cum'] = dfw['photothermal'].cumsum()
        dfw['station ID'] = row['station ID']
        dfw['year'] = row['year']
        dfw['season'] = row['season']
        dfall = pd.concat([dfall, dfw])
    print(dfall.columns, dfm.columns)
    df = dfm.merge(dfall, on=['station ID', 'year', 'Date', 'season'], how='left')
    df.boxplot(column=['Thermal_cum', 'photothermal_cum'], by=['DStage'])
    show()
    df.to_excel('../data/dfall_phofun2.xlsx', index=False)
    thermaldf = dfall.merge(dfm, on=['station ID', 'year', 'Date', 'season'], how='left')
    thermaldf.to_excel('../data/thermaldf_phofun2.xlsx', index=False)
def Simulate_one_site(T_fun,Photo_fun,T_fun_para={},Photo_fun_para={},station_id=''):
    '''
    station, ID, lat, lon, date, stage
    join with weather by station id and date
    calculate daily thermal, daily photoperiod,photothermal= thermal*photo
    cumumlate thermal, photothermal from regrowth to maturation
    
    '''
    sun = Sun.Sun()
    df = pd.read_csv('../data/obser_pheno_catalog.csv', encoding="GBK",
                     parse_dates=['reviving date', 'tillering date', 'jointing date',
                                  'booting date', 'heading date','maturity date'])
    df['season']= df.groupby(['station ID', 'year']).cumcount()+1
    dfmm = df[['station ID', 'lat', 'lon', 'alt', 'year', 'season',
               'reviving date', 'tillering date', 'jointing date',
               'booting date', 'heading date','maturity date']]
    dfm = pd.melt(dfmm, id_vars=['station ID', 'lat', 'lon', 'alt', 'year', 'season'],
                  value_vars=['reviving date', 'tillering date', 'jointing date',
                              'booting date', 'heading date', 'maturity date'])
    dfm = dfm.rename(columns={'value': 'Date', 'variable': 'DStage'})
    print(dfm)
    dfall = pd.DataFrame()
    for ind, row in df.iterrows():
        dfw = read_station_weather(row['station ID'], row['reviving date'], row['maturity date'])
        dfw['Thermal'] = dfw.TemAver.apply(lambda x: T_fun(T=x,**T_fun_para))
        dfw['Thermal_cum'] = dfw.Thermal.cumsum()
        dfw['dayL'] = dfw.Date.apply(
            lambda x: sun.dayCivilTwilightLength(year=x.year, month=x.month, day=x.day, lon=row.lon, lat=row.lat))
        dfw['photo_raw'] = dfw.dayL.apply(lambda x: Photo_fun(DL=x,**Photo_fun_para))
        dfw['photo'] = dfw.apply(lambda rowt: photo_effect_correct(today=rowt.Date, revd=row['reviving date'], jd=row['jointing date'],
                                              hd=row['heading date'], photo=rowt.photo_raw), axis=1)
        dfw['photothermal'] = dfw.photo * dfw.Thermal
        dfw['photothermal_cum'] = dfw['photothermal'].cumsum()
        dfw['station ID'] = row['station ID']
        dfw['year'] = row['year']
        dfw['season'] = row['season']
        dfall = pd.concat([dfall, dfw])
    print(dfall.columns, dfm.columns)
    df = dfm.merge(dfall, on=['station ID', 'year', 'Date', 'season'], how='left')
    df.boxplot(column=['Thermal_cum', 'photothermal_cum'], by=['DStage'])
    show()
    df.to_excel('../data/dfall_phofun2.xlsx', index=False)
    thermaldf = dfall.merge(dfm, on=['station ID', 'year', 'Date', 'season'], how='left')
    thermaldf.to_excel('../data/thermaldf_phofun2.xlsx', index=False)
if __name__=="__main__":
    Simulate_all_sites(T_fun=T_base_op_ceiling,Photo_fun=photoeffect_oryza200)