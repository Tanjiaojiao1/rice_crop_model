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
    df = df.loc[(df.Date >= start) & (df.Date <= (end+datetime.timedelta(days=100))), ['Date', 'TemAver']]
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



def median(para):
    thermal_df = pd.read_excel('../data/dfall.xlsx', sheet_name='Sheet1')
    thermal_median = thermal_df[para].groupby(thermal_df['DStage']).median()
    print(thermal_median)
    return(thermal_median)


def simthermal_dev_date():
    thermal_df = pd.read_excel('../data/dfall.xlsx')
    thermal_df.index = thermal_df['Date']
    thermal_df['year'] = thermal_df.index.year
    thermal_sum = thermal_df['Thermal_cum']
    thermal_median = median('Thermal_cum')
    bins = [0, 137.792979, 561.930185, 857.311757, 1018.537240, 1493.815289, 2500]
    labesl = ['reviving data', 'tillering data', 'jointing data',
              'booting data', 'heading data', 'maturity data']
    thermal_df['simthermal_Dstage']=pd.cut(thermal_sum,bins=bins,labels=labesl)
    thermal_df['simthermal_standard'] = thermal_df['sim_Dstage'].map(thermal_median.to_dict())
    newthermal_df = thermal_df.drop_duplicates(subset=['station ID','year','sim_Dstage'], keep='first', inplace=False )
    thermal_df = thermal_df.reset_index(drop = True)
    thermal_df.drop(thermal_df.columns[12:15], axis=1, inplace=True)
    newthermal_df = newthermal_df.reset_index(drop = True)
    newthermal_df.drop(newthermal_df.columns[2:12], axis=1, inplace=True)
    simdf = pd.merge(thermal_df, newthermal_df, on=['Date','station ID'], how='left')

    simdf.to_excel('../data/dfall.xlsx', index=False)


def simphotothermal_dev_date():
    thermal_df = pd.read_excel('../data/dfall.xlsx', sheet_name='Sheet1')
    thermal_df.index = thermal_df['Date']
    phothermal_sum = thermal_df['photothermal_cum']
    phothermal_median = median('photothermal_cum')
    bins = [0, 137.792979, 559.648112, 782.703418, 916.268239, 1397.062866, 2500]
    labesl = ['reviving data', 'tillering data', 'jointing data',
              'booting data', 'heading data', 'maturity data']
    thermal_df['simphothermal_Dstage'] = pd.cut(phothermal_sum, bins=bins, labels=labesl)
    thermal_df['simphothermal_standard'] = thermal_df['simphothermal_Dstage'].map(phothermal_median.to_dict())
    newthermal_df = thermal_df.drop_duplicates(subset=['station ID','year','simphothermal_Dstage'], keep='first', inplace=False )
    thermal_df = thermal_df.reset_index(drop = True)
    thermal_df.drop(thermal_df.columns[14:16], axis=1, inplace=True)
    newthermal_df = newthermal_df.reset_index(drop = True)
    newthermal_df.drop(newthermal_df.columns[3:14], axis=1, inplace=True)
    simdf = pd.merge(thermal_df, newthermal_df, on=['station ID','Date'], how='left')

    writer = pd.ExcelWriter('../data/dfall.xlsx', mode='a', engine='openpyxl', if_sheet_exists='new')
    simdf.to_excel(writer, sheet_name='newsheet1', index=False)
    writer.save()
    writer.close()


def simthermal_errodays():
    df = pd.read_excel('../data/dfall.xlsx', sheet_name='Sheet1')
    newdf = df.drop(columns=['year', 'TemAver', 'Thermal', 'Thermal_cum', 'dayL', 'photo_raw',
                                    'photo', 'photothermal', 'photothermal_cum', 'simthermal_Dstage',
                                    'simthermal_standard', 'simphothermal_Dstage',
                                    'simphothermal_standard'],
                            axis=1, inplace=False)
    reviving_date = newdf[newdf['DStage'] == 'reviving data']
    reviving_date['reviving_date'] = reviving_date['Date']
    df2 = pd.merge(df, reviving_date, on=['station ID', 'Date', 'DStage'], how='left')
    df2['reviving_date'] = df2['reviving_date'].ffill()
    df2['obser_dvs'] = df2['DStage'].ffill()
    df2['sim_dvs'] = df2['simthermal_Dstage'].ffill()
    origin = df2['obser_dvs'].groupby([df2['station ID'],
                                      df2['reviving_date'],df2['obser_dvs']]).count()
    sim = df2['sim_dvs'].groupby([df2['station ID'],
                                 df2['reviving_date'],df2['sim_dvs']]).count()

    results = sim.unstack() - origin.unstack()
    result = results.reset_index()
    writer = pd.ExcelWriter('../data/dfall.xlsx', mode='a', engine='openpyxl', if_sheet_exists='new')
    result.to_excel(writer, sheet_name='thermal_errorday', index=False)
    writer.save()
    writer.close()

if __name__ == '__main__':
    simthermal_errodays()


