# -*- coding: utf-8 -*-
import pandas as pd
import numpy
import math
from pylab import *
import Sun
from bayes_opt import BayesianOptimization
import os
from rice_phen import read_station_weather, photo_effect_correct
from photo_period_effect import photoeffect_yin, photoeffect_oryza200, photoeffect_wofost
from T_dev_effect import Wang_engle, T_base_op_ceiling, T_base_opt
import datetime
os.chdir(os.path.dirname(os.path.realpath(__file__)))

pd.options.display.max_columns = 999
pd.options.display.max_rows = 999
dft=pd.DataFrame()

def simulate_and_calibrate_T_base_opt_photoeffect_yin( mu, zeta, ep,Tbase, Topt):
    '''
    STEP 1:
    station, ID, lat, lon, date, stage
    join with weather by station id and date
    calculate daily thermal, daily photoperiod,photothermal= thermal*photo
    cumumlate thermal, photothermal from regrowth to maturation
    '''
    df=dft.copy()
    sun = Sun.Sun()
    df['season'] = df.groupby(['station ID', 'year']).cumcount() + 1
    dfmm = df[['station ID', 'lat', 'lon', 'alt', 'year', 'season',
               'reviving date', 'tillering date', 'jointing date',
               'booting date', 'heading date', 'maturity date']]
    dfm = pd.melt(dfmm, id_vars=['station ID', 'lat', 'lon', 'alt', 'year', 'season'],
                  value_vars=['reviving date', 'tillering date', 'jointing date',
                              'booting date', 'heading date', 'maturity date'])
    dfm = dfm.rename(columns={'value': 'Date', 'variable': 'DStage'})
    dfall = pd.DataFrame()
    for ind, row in df.iterrows():
        dfw = read_station_weather(row['station ID'], row['reviving date'], row['maturity date'])

        dfw['Thermal'] = dfw.TemAver.apply(lambda x: T_base_opt(T=x,Tbase=Tbase, Topt=Topt))
        dfw['Thermal_cum'] = dfw.Thermal.cumsum()
        dfw['dayL'] = dfw.Date.apply(
            lambda x: sun.dayCivilTwilightLength(year=x.year, month=x.month, day=x.day, lon=row.lon, lat=row.lat))
        dfw['photo_raw'] = dfw.dayL.apply(lambda x: photoeffect_yin(DL=x,mu=mu, zeta=zeta, ep=ep))
        dfw['photo'] = dfw.apply(
            lambda rowt: photo_effect_correct(today=rowt.Date, revd=row['reviving date'], jd=row['jointing date'],
                                              hd=row['heading date'], photo=rowt.photo_raw), axis=1)
        dfw['photothermal'] = dfw.photo * dfw.Thermal
        dfw['photothermal_cum'] = dfw['photothermal'].cumsum()
        dfw['station ID'] = row['station ID']
        dfw['year'] = row['year']
        dfw['season'] = row['season']
        dfall = pd.concat([dfall, dfw])

    thermaldf = dfall.merge(dfm, on=['station ID', 'year', 'Date', 'season'], how='right')

    # df.to_excel('../data/run_phen_model_result/dfall' + str(number) + '.xlsx', index=False)
    # thermaldf.to_excel('../data/run_phen_model_result/thermaldf.xlsx', index=False)
    dfp=thermaldf.groupby('DStage').median()['photothermal_cum'].reset_index()
    dfp=dfp.sort_values(by=['photothermal_cum'])
    # dfall['Thermal_Dstage'] = pd.cut(dfall['photothermal_cum'], bins=dfp.photothermal_cum, labels=dfp.DStage.tolist()[1:])
    mybins=np.insert(dfp.photothermal_cum,len(dfp.photothermal_cum),9999999)
    mybins[0]=0
    print(mybins)
    dfall['Thermal_Dstage'] = dfall.groupby(['station ID', 'year','season'])[['photothermal_cum']].transform(lambda x: pd.cut(x, bins=mybins, labels=dfp.DStage.tolist()).astype(str))
    dfall=dfall.drop_duplicates(subset=['station ID', 'year','season','Thermal_Dstage'])

    dfall=dfall[['station ID', 'year','season','Date','Thermal_Dstage']].rename(columns={'Thermal_Dstage':'DStage','Date':'Date_sim'})

    dff=dfm.merge(dfall,on=['station ID', 'year','season','DStage'],how='left')
    dff.loc[dff.DStage!='reviving date','Date_sim']=dff.loc[dff.DStage!='reviving date','Date_sim'].apply(lambda x:x+datetime.timedelta(days=-1))

    
    dff['delta_days']=dff.apply(lambda row:(row.Date_sim-row.Date).days,axis=1)
    heading_error=dff.loc[dff.DStage=='heading date','delta_days'].abs().mean()

    return heading_error

def bayesoptimize(func, init_points, n_iter):
    pbounds = {'Tbase': (8, 15), 'Topt': (20, 38),
               'mu': (-60, -10), 'zeta': (0.5, 33), 'ep': (0.5, 33)}
    optimizer = BayesianOptimization(func, pbounds=pbounds, random_state=1)
    optimizer.maximize(init_points, n_iter)
    print(optimizer.max)
def opsite():
    data = pd.read_csv('../data/obser_pheno_catalog.csv', encoding="GBK",
                       parse_dates=['reviving date', 'tillering date', 'jointing date',
                                    'booting date', 'heading date', 'maturity date'])
    global dft
    dft = data.head(5)
    print(dft)
    bayesoptimize(simulate_and_calibrate_T_base_opt_photoeffect_yin, init_points=15, n_iter=2)



if __name__ == "__main__":
    data = pd.read_csv('../data/obser_pheno_catalog.csv', encoding="GBK",
                       parse_dates=['reviving date', 'tillering date', 'jointing date',
                                    'booting date', 'heading date', 'maturity date'])

    dft = data.head(3)
    simulate_and_calibrate_T_base_opt_photoeffect_yin( mu=-2, zeta=20, ep=20,Tbase=8, Topt=30)


