# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import math
from pylab import *
import Sun
import os
from rice_phen import read_station_weather, photo_effect_correct
from photo_period_effect import photoeffect_yin, photoeffect_oryza2000, photoeffect_wofost
from T_dev_effect import Wang_engle, T_base_opt_ceiling, T_base_opt
import datetime
os.chdir(os.path.dirname(os.path.realpath(__file__)))

pd.options.display.max_columns = 999
pd.options.display.max_rows = 999

def Tem_correct_T_base_opt(today, hd,T,Tbase2,Topt2,Thermal):
    if today > hd:
        return T_base_opt(T,Tbase2,Topt2)
    else:
        return Thermal

def Tem_correct_T_base_opt_ceiling(today, hd,T,Tbase2,Topt_low2,Topt_high2,Tcei2,Thermal):
    if today > hd:
        return T_base_opt_ceiling(T,Tbase2,Topt_low2,Topt_high2,Tcei2)
    else:
        return Thermal

def Tem_correct_Wang_engle(today, hd,T,Tbase2,Topt2,Tcei2,Thermal):
    if today > hd:
        return  Wang_engle(T, Tbase2,Topt2,Tcei2)
    else:
        return Thermal

def simulate_and_calibrate_T_base_opt_photoeffect_yin(mu, zeta, ep, Tbase, Topt, Tbase2):
    df = dft.copy()
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
        dfw['Thermal_raw'] = dfw.TemAver.apply(lambda x: T_base_opt(T=x, Tbase=Tbase, Topt=Topt))
        dfw['Thermal_correct'] = dfw.apply(
            lambda rowt: Tem_correct_T_base_opt(today=rowt.Date, hd=row['heading date'], T=rowt.TemAver,
                                                Tbase2=Tbase2, Topt2=Topt, Thermal=rowt.Thermal_raw), axis=1)
        dfw['Thermal_cum'] = dfw.Thermal_correct.cumsum()
        dfw['dayL'] = dfw.Date.apply(
            lambda x: sun.dayCivilTwilightLength(year=x.year, month=x.month, day=x.day, lon=row.lon, lat=row.lat))
        dfw['photo_raw'] = dfw.dayL.apply(lambda x: photoeffect_yin(DL=x, mu=mu, zeta=zeta, ep=ep))
        dfw['photo'] = dfw.apply(
            lambda rowt: photo_effect_correct(today=rowt.Date, revd=row['reviving date'], jd=row['jointing date'],
                                              hd=row['heading date'], photo=rowt.photo_raw), axis=1)
        dfw['photothermal'] = dfw.photo * dfw.Thermal_correct
        dfw['photothermal_cum'] = dfw['photothermal'].cumsum()
        dfw['station ID'] = row['station ID']
        dfw['year'] = row['year']
        dfw['season'] = row['season']
        dfall = pd.concat([dfall, dfw])
    thermaldf = dfall.merge(dfm, on=['station ID', 'year', 'Date', 'season'], how='right')
    # thermaldf.to_excel('../data/run_phen_model_result/thermaldf.xlsx', index=False)
    dfp = thermaldf.groupby('DStage').median()['photothermal_cum'].reset_index()
    dfp = dfp.sort_values(by=['photothermal_cum']).reset_index()
    mybins = dfp.photothermal_cum.tolist()
    mybins.append(9999999)
    mybins[0] = 0
    print(mybins)
    dfall['PhotoThermal_Dstage'] = dfall.groupby(['station ID', 'year', 'season'])[['photothermal_cum']].transform(
        lambda x: pd.cut(x, bins=mybins, labels=dfp.DStage.tolist()).astype(str))
    dfpall = dfall.drop_duplicates(subset=['station ID', 'year', 'season', 'PhotoThermal_Dstage'])

    dfpall = dfpall[['station ID', 'year', 'season', 'Date', 'PhotoThermal_Dstage']].rename(
        columns={'PhotoThermal_Dstage': 'DStage', 'Date': 'Date_simphoto'})

    dff = dfm.merge(dfpall, on=['station ID', 'year', 'season', 'DStage'], how='left')
    index = dff[(dff['DStage'] == 'maturity date') ].index
    for i in index:
        ID=dff['station ID'][i]
        y=dff['year'][i]
        sea=dff['season'][i]
        stardate = dff[dff['station ID']==ID][dff.year==y][dff.season==sea][dff.DStage=='reviving date']['Date']
        enddate = (stardate + datetime.timedelta(days=124)).values[0]
        endtime = pd.Timestamp(enddate)
        if dff['Date_simphoto'][i]==np.nan or dff['Date_simphoto'][i]>endtime:
            dff['Date_simphoto'][i] = endtime
    dff.loc[dff.DStage != 'reviving date', 'Date_simphoto'] = dff.loc[dff.DStage != 'reviving date', 'Date_simphoto'].apply(
        lambda x: x + datetime.timedelta(days=-1))
    dff['delta_days'] = dff.apply(lambda row: (row.Date_simphoto - row.Date).days, axis=1)
    heading_error = dff.loc[dff.DStage == 'heading date', 'delta_days'].abs().mean()
    maturity_error = dff.loc[dff.DStage == 'maturity date', 'delta_days'].abs().mean()
    print(dff, heading_error, maturity_error)

    dfcumt = thermaldf.groupby('DStage').median()['Thermal_cum'].reset_index()
    dfcumt = dfcumt.sort_values(by=['Thermal_cum']).reset_index()
    mybins2 = dfcumt.Thermal_cum.tolist()
    mybins2.append(9999999)
    mybins2[0] = 0
    dfall['Thermal_Dstage'] = dfall.groupby(['station ID', 'year', 'season'])[['Thermal_cum']].transform(
        lambda x: pd.cut(x, bins=mybins2, labels=dfcumt.DStage.tolist()).astype(str))
    dfcumtall = dfall.drop_duplicates(subset=['station ID', 'year', 'season', 'Thermal_Dstage'])
    dfcumtall = dfcumtall[['station ID', 'year', 'season', 'Date', 'Thermal_Dstage']].rename(
        columns={'Thermal_Dstage': 'DStage', 'Date': 'Date_sim'})
    dffcumt = dfm.merge(dfcumtall, on=['station ID', 'year', 'season', 'DStage'], how='left')
    index2 = dffcumt[(dffcumt['DStage'] == 'maturity date') ].index
    for m in index2:
        ID2=dffcumt['station ID'][m]
        y2=dffcumt['year'][m]
        sea2=dffcumt['season'][m]
        stardate2 = dffcumt[dffcumt['station ID']==ID2][dffcumt.year==y2][dffcumt.season==sea2][dffcumt.DStage=='reviving date']['Date']
        enddate2 = (stardate2 + datetime.timedelta(days=124)).values[0]
        endtime2 = pd.Timestamp(enddate2)
        if dffcumt['Date_sim'][m]==np.nan or dffcumt['Date_sim'][m]>endtime2:
            dffcumt['Date_sim'][m] = endtime2
    dffcumt.loc[dffcumt.DStage != 'reviving date', 'Date_sim'] = dffcumt.loc[dff.DStage != 'reviving date', 'Date_sim'].apply(
        lambda x: x + datetime.timedelta(days=-1))
    dffcumt['delta_days'] = dffcumt.apply(lambda row: (row.Date_sim - row.Date).days, axis=1)
    heading_error2 = dffcumt.loc[dffcumt.DStage == 'heading date', 'delta_days'].abs().mean()
    maturity_error2 = dffcumt.loc[dffcumt.DStage == 'maturity date', 'delta_days'].abs().mean()
    print(dffcumt, heading_error2, maturity_error2)

    ERROR = dff.merge(dffcumt, on=['station ID','lat', 'lon', 'alt','year', 'season','DStage','Date'], how='right')
    ERROR = ERROR.rename(columns={'delta_days_x': 'delta_days_simphoto', 'delta_days_y': 'delta_days_sim'})
    ERROR.to_excel(savepath+'error_T_base_opt_Yin'+flg+'.xlsx', index=False)
    print(ERROR, heading_error, heading_error2, maturity_error, maturity_error2)
    return maturity_error

def simulate_and_calibrate_T_base_opt_photoeffect_wofost(Dc, Do, Tbase, Topt, Tbase2):
    df = dft.copy()
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
        dfw['Thermal_raw'] = dfw.TemAver.apply(lambda x: T_base_opt(T=x, Tbase=Tbase, Topt=Topt))
        dfw['Thermal_correct'] = dfw.apply(
            lambda rowt: Tem_correct_T_base_opt(today=rowt.Date, hd=row['heading date'], T=rowt.TemAver,
                                                Tbase2=Tbase2, Topt2=Topt, Thermal=rowt.Thermal_raw), axis=1)
        dfw['Thermal_cum'] = dfw.Thermal_correct.cumsum()
        dfw['dayL'] = dfw.Date.apply(
            lambda x: sun.dayCivilTwilightLength(year=x.year, month=x.month, day=x.day, lon=row.lon, lat=row.lat))
        dfw['photo_raw'] = dfw.dayL.apply(lambda x: photoeffect_wofost(DL=x, Dc=Dc, Do=Do))
        dfw['photo'] = dfw.apply(
            lambda rowt: photo_effect_correct(today=rowt.Date, revd=row['reviving date'], jd=row['jointing date'],
                                              hd=row['heading date'], photo=rowt.photo_raw), axis=1)
        dfw['photothermal'] = dfw.photo * dfw.Thermal_correct
        dfw['photothermal_cum'] = dfw['photothermal'].cumsum()
        dfw['station ID'] = row['station ID']
        dfw['year'] = row['year']
        dfw['season'] = row['season']
        dfall = pd.concat([dfall, dfw])

    thermaldf = dfall.merge(dfm, on=['station ID', 'year', 'Date', 'season'], how='right')
    # thermaldf.to_excel('../data/thermal_T_base_opt_wofost.xlsx', index=False)
    dfp = thermaldf.groupby('DStage').median()['photothermal_cum'].reset_index()
    dfp = dfp.sort_values(by=['photothermal_cum']).reset_index()
    mybins = dfp.photothermal_cum.tolist()
    mybins.append(9999999)
    mybins[0] = 0
    print(mybins)
    dfall['PhotoThermal_Dstage'] = dfall.groupby(['station ID', 'year', 'season'])[['photothermal_cum']].transform(
        lambda x: pd.cut(x, bins=mybins, labels=dfp.DStage.tolist()).astype(str))
    dfpall = dfall.drop_duplicates(subset=['station ID', 'year', 'season', 'PhotoThermal_Dstage'])

    dfpall = dfpall[['station ID', 'year', 'season', 'Date', 'PhotoThermal_Dstage']].rename(
        columns={'PhotoThermal_Dstage': 'DStage', 'Date': 'Date_simphoto'})

    dff = dfm.merge(dfpall, on=['station ID', 'year', 'season', 'DStage'], how='left')
    index = dff[(dff['DStage'] == 'maturity date') ].index
    for i in index:
        ID=dff['station ID'][i]
        y=dff['year'][i]
        sea=dff['season'][i]
        stardate = dff[dff['station ID']==ID][dff.year==y][dff.season==sea][dff.DStage=='reviving date']['Date']
        enddate = (stardate + datetime.timedelta(days=124)).values[0]
        endtime = pd.Timestamp(enddate)
        if dff['Date_simphoto'][i]==np.nan or dff['Date_simphoto'][i]>endtime:
            dff['Date_simphoto'][i] = endtime
    dff.loc[dff.DStage != 'reviving date', 'Date_simphoto'] = dff.loc[dff.DStage != 'reviving date', 'Date_simphoto'].apply(
        lambda x: x + datetime.timedelta(days=-1))
    dff['delta_days'] = dff.apply(lambda row: (row.Date_simphoto - row.Date).days, axis=1)
    heading_error = dff.loc[dff.DStage == 'heading date', 'delta_days'].abs().mean()
    maturity_error = dff.loc[dff.DStage == 'maturity date', 'delta_days'].abs().mean()
    print(dff, heading_error, maturity_error)


    dfcumt = thermaldf.groupby('DStage').median()['Thermal_cum'].reset_index()
    dfcumt = dfcumt.sort_values(by=['Thermal_cum']).reset_index()
    mybins2 = dfcumt.Thermal_cum.tolist()
    mybins2.append(9999999)
    mybins2[0] = 0
    dfall['Thermal_Dstage'] = dfall.groupby(['station ID', 'year', 'season'])[['Thermal_cum']].transform(
        lambda x: pd.cut(x, bins=mybins2, labels=dfcumt.DStage.tolist()).astype(str))
    dfcumtall = dfall.drop_duplicates(subset=['station ID', 'year', 'season', 'Thermal_Dstage'])
    dfcumtall = dfcumtall[['station ID', 'year', 'season', 'Date', 'Thermal_Dstage']].rename(
        columns={'Thermal_Dstage': 'DStage', 'Date': 'Date_sim'})
    dffcumt = dfm.merge(dfcumtall, on=['station ID', 'year', 'season', 'DStage'], how='left')
    index2 = dffcumt[(dffcumt['DStage'] == 'maturity date') ].index
    for m in index2:
        ID2=dffcumt['station ID'][m]
        y2=dffcumt['year'][m]
        sea2=dffcumt['season'][m]
        stardate2 = dffcumt[dffcumt['station ID']==ID2][dffcumt.year==y2][dffcumt.season==sea2][dffcumt.DStage=='reviving date']['Date']
        enddate2 = (stardate2 + datetime.timedelta(days=124)).values[0]
        endtime2 = pd.Timestamp(enddate2)
        if dffcumt['Date_sim'][m]==np.nan or dffcumt['Date_sim'][m]>endtime2:
            dffcumt['Date_sim'][m] = endtime2
    dffcumt.loc[dffcumt.DStage != 'reviving date', 'Date_sim'] = dffcumt.loc[dff.DStage != 'reviving date', 'Date_sim'].apply(
        lambda x: x + datetime.timedelta(days=-1))
    dffcumt['delta_days'] = dffcumt.apply(lambda row: (row.Date_sim - row.Date).days, axis=1)
    heading_error2 = dffcumt.loc[dffcumt.DStage == 'heading date', 'delta_days'].abs().mean()
    maturity_error2 = dffcumt.loc[dffcumt.DStage == 'maturity date', 'delta_days'].abs().mean()
    print(dffcumt, heading_error2,maturity_error2)

    ERROR = dff.merge(dffcumt, on=['station ID','lat', 'lon', 'alt','year', 'season','DStage','Date'], how='right')
    ERROR = ERROR.rename(columns={'delta_days_x': 'delta_days_simphoto', 'delta_days_y': 'delta_days_sim'})
    ERROR.to_excel(savepath+'error_T_base_opt_WOFOST'+flg+'.xlsx', index=False)
    print(ERROR, heading_error,heading_error2,maturity_error,maturity_error2)
    return maturity_error

def simulate_and_calibrate_T_base_opt_photoeffect_oryza2000(Dc, Tbase, Topt, Tbase2):
    df = dft.copy()
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
        dfw['Thermal_raw'] = dfw.TemAver.apply(lambda x: T_base_opt(T=x, Tbase=Tbase, Topt=Topt))
        dfw['Thermal_correct'] = dfw.apply(
            lambda rowt: Tem_correct_T_base_opt(today=rowt.Date, hd=row['heading date'], T=rowt.TemAver,
                                                Tbase2=Tbase2, Topt2=Topt, Thermal=rowt.Thermal_raw), axis=1)
        dfw['Thermal_cum'] = dfw.Thermal_correct.cumsum()
        dfw['dayL'] = dfw.Date.apply(
            lambda x: sun.dayCivilTwilightLength(year=x.year, month=x.month, day=x.day, lon=row.lon, lat=row.lat))
        dfw['photo_raw'] = dfw.dayL.apply(lambda x: photoeffect_oryza2000(DL=x, Dc=Dc, PPSE=0.2))
        dfw['photo'] = dfw.apply(
            lambda rowt: photo_effect_correct(today=rowt.Date, revd=row['reviving date'], jd=row['jointing date'],
                                              hd=row['heading date'], photo=rowt.photo_raw), axis=1)
        dfw['photothermal'] = dfw.photo * dfw.Thermal_correct
        dfw['photothermal_cum'] = dfw['photothermal'].cumsum()
        dfw['station ID'] = row['station ID']
        dfw['year'] = row['year']
        dfw['season'] = row['season']
        dfall = pd.concat([dfall, dfw])

    thermaldf = dfall.merge(dfm, on=['station ID', 'year', 'Date', 'season'], how='right')
    # thermaldf.to_excel('../data/run_phen_model_result/thermaldf.xlsx', index=False)
    dfp = thermaldf.groupby('DStage').median()['photothermal_cum'].reset_index()
    dfp = dfp.sort_values(by=['photothermal_cum']).reset_index()
    mybins = dfp.photothermal_cum.tolist()
    mybins.append(9999999)
    mybins[0] = 0
    print(mybins)
    dfall['PhotoThermal_Dstage'] = dfall.groupby(['station ID', 'year', 'season'])[['photothermal_cum']].transform(
        lambda x: pd.cut(x, bins=mybins, labels=dfp.DStage.tolist()).astype(str))
    dfpall = dfall.drop_duplicates(subset=['station ID', 'year', 'season', 'PhotoThermal_Dstage'])

    dfpall = dfpall[['station ID', 'year', 'season', 'Date', 'PhotoThermal_Dstage']].rename(
        columns={'PhotoThermal_Dstage': 'DStage', 'Date': 'Date_simphoto'})

    dff = dfm.merge(dfpall, on=['station ID', 'year', 'season', 'DStage'], how='left')
    index = dff[(dff['DStage'] == 'maturity date') ].index
    for i in index:
        ID=dff['station ID'][i]
        y=dff['year'][i]
        sea=dff['season'][i]
        stardate = dff[dff['station ID']==ID][dff.year==y][dff.season==sea][dff.DStage=='reviving date']['Date']
        enddate = (stardate + datetime.timedelta(days=124)).values[0]
        endtime = pd.Timestamp(enddate)
        if dff['Date_simphoto'][i]==np.nan or dff['Date_simphoto'][i]>endtime:
            dff['Date_simphoto'][i] = endtime
    dff.loc[dff.DStage != 'reviving date', 'Date_simphoto'] = dff.loc[dff.DStage != 'reviving date', 'Date_simphoto'].apply(
        lambda x: x + datetime.timedelta(days=-1))
    dff['delta_days'] = dff.apply(lambda row: (row.Date_simphoto - row.Date).days, axis=1)
    heading_error = dff.loc[dff.DStage == 'heading date', 'delta_days'].abs().mean()
    maturity_error = dff.loc[dff.DStage == 'maturity date', 'delta_days'].abs().mean()
    print(dff,heading_error,maturity_error)


    dfcumt = thermaldf.groupby('DStage').median()['Thermal_cum'].reset_index()
    dfcumt = dfcumt.sort_values(by=['Thermal_cum']).reset_index()
    mybins2 = dfcumt.Thermal_cum.tolist()
    mybins2.append(9999999)
    mybins2[0] = 0
    dfall['Thermal_Dstage'] = dfall.groupby(['station ID', 'year', 'season'])[['Thermal_cum']].transform(
        lambda x: pd.cut(x, bins=mybins2, labels=dfcumt.DStage.tolist()).astype(str))
    dfcumtall = dfall.drop_duplicates(subset=['station ID', 'year', 'season', 'Thermal_Dstage'])
    dfcumtall = dfcumtall[['station ID', 'year', 'season', 'Date', 'Thermal_Dstage']].rename(
        columns={'Thermal_Dstage': 'DStage', 'Date': 'Date_sim'})
    dffcumt = dfm.merge(dfcumtall, on=['station ID', 'year', 'season', 'DStage'], how='left')
    index2 = dffcumt[(dffcumt['DStage'] == 'maturity date') ].index
    for m in index2:
        ID2=dffcumt['station ID'][m]
        y2=dffcumt['year'][m]
        sea2=dffcumt['season'][m]
        stardate2 = dffcumt[dffcumt['station ID']==ID2][dffcumt.year==y2][dffcumt.season==sea2][dffcumt.DStage=='reviving date']['Date']
        enddate2 = (stardate2 + datetime.timedelta(days=124)).values[0]
        endtime2 = pd.Timestamp(enddate2)
        if dffcumt['Date_sim'][m]==np.nan or dffcumt['Date_sim'][m]>endtime2:
            dffcumt['Date_sim'][m] = endtime2
    dffcumt.loc[dffcumt.DStage != 'reviving date', 'Date_sim'] = dffcumt.loc[dff.DStage != 'reviving date', 'Date_sim'].apply(
        lambda x: x + datetime.timedelta(days=-1))
    dffcumt['delta_days'] = dffcumt.apply(lambda row: (row.Date_sim - row.Date).days, axis=1)
    heading_error2 = dffcumt.loc[dffcumt.DStage == 'heading date', 'delta_days'].abs().mean()
    maturity_error2 = dffcumt.loc[dffcumt.DStage == 'maturity date', 'delta_days'].abs().mean()
    print(dffcumt, heading_error2, maturity_error2)

    ERROR = dff.merge(dffcumt, on=['station ID','lat', 'lon', 'alt','year', 'season','DStage','Date'], how='right')
    ERROR = ERROR.rename(columns={'delta_days_x': 'delta_days_simphoto', 'delta_days_y': 'delta_days_sim'})
    ERROR.to_excel(savepath+'error_T_base_opt_oryza'+flg+'.xlsx', index=False)
    print(ERROR, heading_error,heading_error2,maturity_error,maturity_error2)
    return maturity_error

def simulate_and_calibrate_T_base_opt_ceiling_photoeffect_yin(mu, zeta, ep, Tbase, Topt_low, Topt_high, Tcei,Tbase2):
    df = dft.copy()
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
        dfw['Thermal_raw'] = dfw.TemAver.apply(lambda x: T_base_opt_ceiling(T=x, Tbase=Tbase, Topt_low=Topt_low, Topt_high=Topt_high,  Tcei=Tcei))
        dfw['Thermal_correct'] = dfw.apply(
            lambda rowt: Tem_correct_T_base_opt_ceiling(today=rowt.Date, hd=row['heading date'], T=rowt.TemAver,Tbase2=Tbase2, Topt_low2=Topt_low,
                                                        Topt_high2=Topt_high,Tcei2 =Tcei, Thermal=rowt.Thermal_raw), axis=1)
        dfw['Thermal_cum'] = dfw.Thermal_correct.cumsum()
        dfw['dayL'] = dfw.Date.apply(
            lambda x: sun.dayCivilTwilightLength(year=x.year, month=x.month, day=x.day, lon=row.lon, lat=row.lat))
        dfw['photo_raw'] = dfw.dayL.apply(lambda x: photoeffect_yin(DL=x, mu=mu, zeta=zeta, ep=ep))
        dfw['photo'] = dfw.apply(
            lambda rowt: photo_effect_correct(today=rowt.Date, revd=row['reviving date'], jd=row['jointing date'],
                                              hd=row['heading date'], photo=rowt.photo_raw), axis=1)
        dfw['photothermal'] = dfw.photo * dfw.Thermal_correct
        dfw['photothermal_cum'] = dfw['photothermal'].cumsum()
        dfw['station ID'] = row['station ID']
        dfw['year'] = row['year']
        dfw['season'] = row['season']
        dfall = pd.concat([dfall, dfw])

    thermaldf = dfall.merge(dfm, on=['station ID', 'year', 'Date', 'season'], how='right')
    # thermaldf.to_excel('../data/run_phen_model_result/thermaldf.xlsx', index=False)
    dfp = thermaldf.groupby('DStage').median()['photothermal_cum'].reset_index()
    dfp = dfp.sort_values(by=['photothermal_cum']).reset_index()
    mybins = dfp.photothermal_cum.tolist()
    mybins.append(9999999)
    mybins[0] = 0
    print(mybins)
    dfall['PhotoThermal_Dstage'] = dfall.groupby(['station ID', 'year', 'season'])[['photothermal_cum']].transform(
        lambda x: pd.cut(x, bins=mybins, labels=dfp.DStage.tolist()).astype(str))
    dfpall = dfall.drop_duplicates(subset=['station ID', 'year', 'season', 'PhotoThermal_Dstage'])

    dfpall = dfpall[['station ID', 'year', 'season', 'Date', 'PhotoThermal_Dstage']].rename(
        columns={'PhotoThermal_Dstage': 'DStage', 'Date': 'Date_simphoto'})

    dff = dfm.merge(dfpall, on=['station ID', 'year', 'season', 'DStage'], how='left')
    index = dff[(dff['DStage'] == 'maturity date') ].index
    for i in index:
        ID=dff['station ID'][i]
        y=dff['year'][i]
        sea=dff['season'][i]
        stardate = dff[dff['station ID']==ID][dff.year==y][dff.season==sea][dff.DStage=='reviving date']['Date']
        enddate = (stardate + datetime.timedelta(days=124)).values[0]
        endtime = pd.Timestamp(enddate)
        if dff['Date_simphoto'][i]==np.nan or dff['Date_simphoto'][i]>endtime:
            dff['Date_simphoto'][i] = endtime
    dff.loc[dff.DStage != 'reviving date', 'Date_simphoto'] = dff.loc[dff.DStage != 'reviving date', 'Date_simphoto'].apply(
        lambda x: x + datetime.timedelta(days=-1))
    dff['delta_days'] = dff.apply(lambda row: (row.Date_simphoto - row.Date).days, axis=1)
    heading_error = dff.loc[dff.DStage == 'heading date', 'delta_days'].abs().mean()
    maturity_error = dff.loc[dff.DStage == 'maturity date', 'delta_days'].abs().mean()
    print(dff, heading_error, maturity_error)

    dfcumt = thermaldf.groupby('DStage').median()['Thermal_cum'].reset_index()
    dfcumt = dfcumt.sort_values(by=['Thermal_cum']).reset_index()
    mybins2 = dfcumt.Thermal_cum.tolist()
    mybins2.append(9999999)
    mybins2[0] = 0
    dfall['Thermal_Dstage'] = dfall.groupby(['station ID', 'year', 'season'])[['Thermal_cum']].transform(
        lambda x: pd.cut(x, bins=mybins2, labels=dfcumt.DStage.tolist()).astype(str))
    dfcumtall = dfall.drop_duplicates(subset=['station ID', 'year', 'season', 'Thermal_Dstage'])
    dfcumtall = dfcumtall[['station ID', 'year', 'season', 'Date', 'Thermal_Dstage']].rename(
        columns={'Thermal_Dstage': 'DStage', 'Date': 'Date_sim'})
    dffcumt = dfm.merge(dfcumtall, on=['station ID', 'year', 'season', 'DStage'], how='left')
    index2 = dffcumt[(dffcumt['DStage'] == 'maturity date') ].index
    for m in index2:
        ID2=dffcumt['station ID'][m]
        y2=dffcumt['year'][m]
        sea2=dffcumt['season'][m]
        stardate2 = dffcumt[dffcumt['station ID']==ID2][dffcumt.year==y2][dffcumt.season==sea2][dffcumt.DStage=='reviving date']['Date']
        enddate2 = (stardate2 + datetime.timedelta(days=124)).values[0]
        endtime2 = pd.Timestamp(enddate2)
        if dffcumt['Date_sim'][m]==np.nan or dffcumt['Date_sim'][m]>endtime2:
            dffcumt['Date_sim'][m] = endtime2
    dffcumt.loc[dffcumt.DStage != 'reviving date', 'Date_sim'] = dffcumt.loc[dff.DStage != 'reviving date', 'Date_sim'].apply(
        lambda x: x + datetime.timedelta(days=-1))
    dffcumt['delta_days'] = dffcumt.apply(lambda row: (row.Date_sim - row.Date).days, axis=1)
    heading_error2 = dffcumt.loc[dffcumt.DStage == 'heading date', 'delta_days'].abs().mean()
    maturity_error2 = dffcumt.loc[dffcumt.DStage == 'maturity date', 'delta_days'].abs().mean()
    print(dffcumt, heading_error2, maturity_error2)

    ERROR = dff.merge(dffcumt, on=['station ID','lat', 'lon', 'alt','year', 'season','DStage','Date'], how='right')
    ERROR = ERROR.rename(columns={'delta_days_x': 'delta_days_simphoto', 'delta_days_y': 'delta_days_sim'})
    ERROR.to_excel(savepath+'error_T_base_op_cei_Yin'+flg+'.xlsx', index=False)
    print(ERROR, heading_error,heading_error2,maturity_error,maturity_error2)
    return maturity_error

def simulate_and_calibrate_T_base_opt_ceiling_photoeffect_wofost(Dc, Do, Tbase, Topt_low, Topt_high, Tcei,Tbase2):
    df = dft.copy()
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
        dfw['Thermal_raw'] = dfw.TemAver.apply(
            lambda x: T_base_opt_ceiling(T=x, Tbase=Tbase, Topt_low=Topt_low, Topt_high=Topt_high, Tcei=Tcei))
        dfw['Thermal_correct'] = dfw.apply(
            lambda rowt: Tem_correct_T_base_opt_ceiling(today=rowt.Date, hd=row['heading date'], T=rowt.TemAver,
                                                        Tbase2=Tbase2, Topt_low2=Topt_low,Topt_high2=Topt_high, Tcei2=Tcei, Thermal=rowt.Thermal_raw),axis=1)
        dfw['Thermal_cum'] = dfw.Thermal_correct.cumsum()
        dfw['dayL'] = dfw.Date.apply(
            lambda x: sun.dayCivilTwilightLength(year=x.year, month=x.month, day=x.day, lon=row.lon, lat=row.lat))
        dfw['photo_raw'] = dfw.dayL.apply(lambda x: photoeffect_wofost(DL=x, Dc=Dc, Do=Do))
        dfw['photo'] = dfw.apply(
            lambda rowt: photo_effect_correct(today=rowt.Date, revd=row['reviving date'], jd=row['jointing date'],
                                              hd=row['heading date'], photo=rowt.photo_raw), axis=1)
        dfw['photothermal'] = dfw.photo * dfw.Thermal_correct
        dfw['photothermal_cum'] = dfw['photothermal'].cumsum()
        dfw['station ID'] = row['station ID']
        dfw['year'] = row['year']
        dfw['season'] = row['season']
        dfall = pd.concat([dfall, dfw])

    thermaldf = dfall.merge(dfm, on=['station ID', 'year', 'Date', 'season'], how='right')
    # thermaldf.to_excel('../data/run_phen_model_result/thermaldf.xlsx', index=False)
    dfp = thermaldf.groupby('DStage').median()['photothermal_cum'].reset_index()
    dfp = dfp.sort_values(by=['photothermal_cum']).reset_index()
    mybins = dfp.photothermal_cum.tolist()
    mybins.append(9999999)
    mybins[0] = 0
    print(mybins)
    dfall['PhotoThermal_Dstage'] = dfall.groupby(['station ID', 'year', 'season'])[['photothermal_cum']].transform(
        lambda x: pd.cut(x, bins=mybins, labels=dfp.DStage.tolist()).astype(str))
    dfpall = dfall.drop_duplicates(subset=['station ID', 'year', 'season', 'PhotoThermal_Dstage'])

    dfpall = dfpall[['station ID', 'year', 'season', 'Date', 'PhotoThermal_Dstage']].rename(
        columns={'PhotoThermal_Dstage': 'DStage', 'Date': 'Date_simphoto'})

    dff = dfm.merge(dfpall, on=['station ID', 'year', 'season', 'DStage'], how='left')
    index = dff[(dff['DStage'] == 'maturity date') ].index
    for i in index:
        ID=dff['station ID'][i]
        y=dff['year'][i]
        sea=dff['season'][i]
        stardate = dff[dff['station ID']==ID][dff.year==y][dff.season==sea][dff.DStage=='reviving date']['Date']
        enddate = (stardate + datetime.timedelta(days=124)).values[0]
        endtime = pd.Timestamp(enddate)
        if dff['Date_simphoto'][i]==np.nan or dff['Date_simphoto'][i]>endtime:
            dff['Date_simphoto'][i] = endtime
    dff.loc[dff.DStage != 'reviving date', 'Date_simphoto'] = dff.loc[dff.DStage != 'reviving date', 'Date_simphoto'].apply(
        lambda x: x + datetime.timedelta(days=-1))
    dff['delta_days'] = dff.apply(lambda row: (row.Date_simphoto - row.Date).days, axis=1)
    heading_error = dff.loc[dff.DStage == 'heading date', 'delta_days'].abs().mean()
    maturity_error = dff.loc[dff.DStage == 'maturity date', 'delta_days'].abs().mean()
    print(dff,heading_error,maturity_error)


    dfcumt = thermaldf.groupby('DStage').median()['Thermal_cum'].reset_index()
    dfcumt = dfcumt.sort_values(by=['Thermal_cum']).reset_index()
    mybins2 = dfcumt.Thermal_cum.tolist()
    mybins2.append(9999999)
    mybins2[0] = 0
    dfall['Thermal_Dstage'] = dfall.groupby(['station ID', 'year', 'season'])[['Thermal_cum']].transform(
        lambda x: pd.cut(x, bins=mybins2, labels=dfcumt.DStage.tolist()).astype(str))
    dfcumtall = dfall.drop_duplicates(subset=['station ID', 'year', 'season', 'Thermal_Dstage'])
    dfcumtall = dfcumtall[['station ID', 'year', 'season', 'Date', 'Thermal_Dstage']].rename(
        columns={'Thermal_Dstage': 'DStage', 'Date': 'Date_sim'})
    dffcumt = dfm.merge(dfcumtall, on=['station ID', 'year', 'season', 'DStage'], how='left')
    index2 = dffcumt[(dffcumt['DStage'] == 'maturity date') ].index
    for m in index2:
        ID2=dffcumt['station ID'][m]
        y2=dffcumt['year'][m]
        sea2=dffcumt['season'][m]
        stardate2 = dffcumt[dffcumt['station ID']==ID2][dffcumt.year==y2][dffcumt.season==sea2][dffcumt.DStage=='reviving date']['Date']
        enddate2 = (stardate2 + datetime.timedelta(days=124)).values[0]
        endtime2 = pd.Timestamp(enddate2)
        if dffcumt['Date_sim'][m]==np.nan or dffcumt['Date_sim'][m]>endtime2:
            dffcumt['Date_sim'][m] = endtime2
    dffcumt.loc[dffcumt.DStage != 'reviving date', 'Date_sim'] = dffcumt.loc[dff.DStage != 'reviving date', 'Date_sim'].apply(
        lambda x: x + datetime.timedelta(days=-1))
    dffcumt['delta_days'] = dffcumt.apply(lambda row: (row.Date_sim - row.Date).days, axis=1)
    heading_error2 = dffcumt.loc[dffcumt.DStage == 'heading date', 'delta_days'].abs().mean()
    maturity_error2 = dffcumt.loc[dffcumt.DStage == 'maturity date', 'delta_days'].abs().mean()
    print(dffcumt, heading_error2,maturity_error2)

    ERROR = dff.merge(dffcumt, on=['station ID','lat', 'lon', 'alt','year', 'season','DStage','Date'], how='right')
    ERROR = ERROR.rename(columns={'delta_days_x': 'delta_days_simphoto', 'delta_days_y': 'delta_days_sim'})
    ERROR.to_excel(savepath+'error_T_base_op_cei_WOFOST'+flg+'.xlsx', index=False)
    print(ERROR, heading_error,heading_error2,maturity_error,maturity_error2)
    return maturity_error

def simulate_and_calibrate_T_base_opt_ceiling_photoeffect_oryza2000(Dc, Tbase, Topt_low, Topt_high, Tcei,Tbase2):
    df = dft.copy()
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
        dfw['Thermal_raw'] = dfw.TemAver.apply(
            lambda x: T_base_opt_ceiling(T=x, Tbase=Tbase, Topt_low=Topt_low, Topt_high=Topt_high, Tcei=Tcei))
        dfw['Thermal_correct'] = dfw.apply(
            lambda rowt: Tem_correct_T_base_opt_ceiling(today=rowt.Date, hd=row['heading date'], T=rowt.TemAver,
                                                        Tbase2=Tbase2, Topt_low2=Topt_low,Topt_high2=Topt_high, Tcei2=Tcei, Thermal=rowt.Thermal_raw),axis=1)
        dfw['Thermal_cum'] = dfw.Thermal_correct.cumsum()
        dfw['dayL'] = dfw.Date.apply(
            lambda x: sun.dayCivilTwilightLength(year=x.year, month=x.month, day=x.day, lon=row.lon, lat=row.lat))
        dfw['photo_raw'] = dfw.dayL.apply(lambda x: photoeffect_oryza2000(DL=x, Dc=Dc))
        dfw['photo'] = dfw.apply(
            lambda rowt: photo_effect_correct(today=rowt.Date, revd=row['reviving date'], jd=row['jointing date'],
                                              hd=row['heading date'], photo=rowt.photo_raw), axis=1)
        dfw['photothermal'] = dfw.photo * dfw.Thermal_correct
        dfw['photothermal_cum'] = dfw['photothermal'].cumsum()
        dfw['station ID'] = row['station ID']
        dfw['year'] = row['year']
        dfw['season'] = row['season']
        dfall = pd.concat([dfall, dfw])

    thermaldf = dfall.merge(dfm, on=['station ID', 'year', 'Date', 'season'], how='right')
    # thermaldf.to_excel('../data/run_phen_model_result/thermaldf.xlsx', index=False)
    dfp = thermaldf.groupby('DStage').median()['photothermal_cum'].reset_index()
    dfp = dfp.sort_values(by=['photothermal_cum']).reset_index()
    mybins = dfp.photothermal_cum.tolist()
    mybins.append(9999999)
    mybins[0] = 0
    print(mybins)
    dfall['PhotoThermal_Dstage'] = dfall.groupby(['station ID', 'year', 'season'])[['photothermal_cum']].transform(
        lambda x: pd.cut(x, bins=mybins, labels=dfp.DStage.tolist()).astype(str))
    dfpall = dfall.drop_duplicates(subset=['station ID', 'year', 'season', 'PhotoThermal_Dstage'])

    dfpall = dfpall[['station ID', 'year', 'season', 'Date', 'PhotoThermal_Dstage']].rename(
        columns={'PhotoThermal_Dstage': 'DStage', 'Date': 'Date_simphoto'})

    dff = dfm.merge(dfpall, on=['station ID', 'year', 'season', 'DStage'], how='left')
    index = dff[(dff['DStage'] == 'maturity date') ].index
    for i in index:
        ID=dff['station ID'][i]
        y=dff['year'][i]
        sea=dff['season'][i]
        stardate = dff[dff['station ID']==ID][dff.year==y][dff.season==sea][dff.DStage=='reviving date']['Date']
        enddate = (stardate + datetime.timedelta(days=124)).values[0]
        endtime = pd.Timestamp(enddate)
        if dff['Date_simphoto'][i]==np.nan or dff['Date_simphoto'][i]>endtime:
            dff['Date_simphoto'][i] = endtime
    dff.loc[dff.DStage != 'reviving date', 'Date_simphoto'] = dff.loc[dff.DStage != 'reviving date', 'Date_simphoto'].apply(
        lambda x: x + datetime.timedelta(days=-1))
    dff['delta_days'] = dff.apply(lambda row: (row.Date_simphoto - row.Date).days, axis=1)
    heading_error = dff.loc[dff.DStage == 'heading date', 'delta_days'].abs().mean()
    maturity_error = dff.loc[dff.DStage == 'maturity date', 'delta_days'].abs().mean()
    print(dff,heading_error,maturity_error)


    dfcumt = thermaldf.groupby('DStage').median()['Thermal_cum'].reset_index()
    dfcumt = dfcumt.sort_values(by=['Thermal_cum']).reset_index()
    mybins2 = dfcumt.Thermal_cum.tolist()
    mybins2.append(9999999)
    mybins2[0] = 0
    dfall['Thermal_Dstage'] = dfall.groupby(['station ID', 'year', 'season'])[['Thermal_cum']].transform(
        lambda x: pd.cut(x, bins=mybins2, labels=dfcumt.DStage.tolist()).astype(str))
    dfcumtall = dfall.drop_duplicates(subset=['station ID', 'year', 'season', 'Thermal_Dstage'])
    dfcumtall = dfcumtall[['station ID', 'year', 'season', 'Date', 'Thermal_Dstage']].rename(
        columns={'Thermal_Dstage': 'DStage', 'Date': 'Date_sim'})
    dffcumt = dfm.merge(dfcumtall, on=['station ID', 'year', 'season', 'DStage'], how='left')
    index2 = dffcumt[(dffcumt['DStage'] == 'maturity date') ].index
    for m in index2:
        ID2=dffcumt['station ID'][m]
        y2=dffcumt['year'][m]
        sea2=dffcumt['season'][m]
        stardate2 = dffcumt[dffcumt['station ID']==ID2][dffcumt.year==y2][dffcumt.season==sea2][dffcumt.DStage=='reviving date']['Date']
        enddate2 = (stardate2 + datetime.timedelta(days=124)).values[0]
        endtime2 = pd.Timestamp(enddate2)
        if dffcumt['Date_sim'][m]==np.nan or dffcumt['Date_sim'][m]>endtime2:
            dffcumt['Date_sim'][m] = endtime2
    dffcumt.loc[dffcumt.DStage != 'reviving date', 'Date_sim'] = dffcumt.loc[dff.DStage != 'reviving date', 'Date_sim'].apply(
        lambda x: x + datetime.timedelta(days=-1))
    dffcumt['delta_days'] = dffcumt.apply(lambda row: (row.Date_sim - row.Date).days, axis=1)
    heading_error2 = dffcumt.loc[dffcumt.DStage == 'heading date', 'delta_days'].abs().mean()
    maturity_error2 = dffcumt.loc[dffcumt.DStage == 'maturity date', 'delta_days'].abs().mean()
    print(dffcumt, heading_error2, maturity_error2 )

    ERROR = dff.merge(dffcumt, on=['station ID','lat', 'lon', 'alt','year', 'season','DStage','Date'], how='right')
    ERROR = ERROR.rename(columns={'delta_days_x': 'delta_days_simphoto', 'delta_days_y': 'delta_days_sim'})
    ERROR.to_excel(savepath+'error_T_base_op_cei_oryza'+flg+'.xlsx', index=False)
    print(ERROR, heading_error,heading_error2,maturity_error,maturity_error2)
    return maturity_error

def simulate_and_calibrate_Wang_engle_photoeffect_yin(mu, zeta, ep, Tbase, Topt, Tcei,Tbase2):
    df = dft.copy()
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
        dfw['Thermal_raw'] = dfw.TemAver.apply(lambda x: Wang_engle(T=x, Tbase=Tbase, Topt=Topt, Tcei=Tcei))
        dfw['Thermal_correct'] = dfw.apply(
            lambda rowt: Tem_correct_Wang_engle(today=rowt.Date, hd=row['heading date'], T=rowt.TemAver,
                                                        Tbase2=Tbase2, Topt2=Topt,Tcei2=Tcei, Thermal=rowt.Thermal_raw), axis=1)
        dfw['Thermal_cum'] = dfw.Thermal_correct.cumsum()
        dfw['dayL'] = dfw.Date.apply(
            lambda x: sun.dayCivilTwilightLength(year=x.year, month=x.month, day=x.day, lon=row.lon, lat=row.lat))
        dfw['photo_raw'] = dfw.dayL.apply(lambda x: photoeffect_yin(DL=x, mu=mu, zeta=zeta, ep=ep))
        dfw['photo'] = dfw.apply(
            lambda rowt: photo_effect_correct(today=rowt.Date, revd=row['reviving date'], jd=row['jointing date'],
                                              hd=row['heading date'], photo=rowt.photo_raw), axis=1)
        dfw['photothermal'] = dfw.photo * dfw.Thermal_correct
        dfw['photothermal_cum'] = dfw['photothermal'].cumsum()
        dfw['station ID'] = row['station ID']
        dfw['year'] = row['year']
        dfw['season'] = row['season']
        dfall = pd.concat([dfall, dfw])

    thermaldf = dfall.merge(dfm, on=['station ID', 'year', 'Date', 'season'], how='right')
    # thermaldf.to_excel('../data/run_phen_model_result/thermaldf.xlsx', index=False)
    dfp = thermaldf.groupby('DStage').median()['photothermal_cum'].reset_index()
    dfp = dfp.sort_values(by=['photothermal_cum']).reset_index()
    mybins = dfp.photothermal_cum.tolist()
    mybins.append(9999999)
    mybins[0] = 0
    print(mybins)
    dfall['PhotoThermal_Dstage'] = dfall.groupby(['station ID', 'year', 'season'])[['photothermal_cum']].transform(
        lambda x: pd.cut(x, bins=mybins, labels=dfp.DStage.tolist()).astype(str))
    dfpall = dfall.drop_duplicates(subset=['station ID', 'year', 'season', 'PhotoThermal_Dstage'])

    dfpall = dfpall[['station ID', 'year', 'season', 'Date', 'PhotoThermal_Dstage']].rename(
        columns={'PhotoThermal_Dstage': 'DStage', 'Date': 'Date_simphoto'})

    dff = dfm.merge(dfpall, on=['station ID', 'year', 'season', 'DStage'], how='left')
    index = dff[(dff['DStage'] == 'maturity date') ].index
    for i in index:
        ID=dff['station ID'][i]
        y=dff['year'][i]
        sea=dff['season'][i]
        stardate = dff[dff['station ID']==ID][dff.year==y][dff.season==sea][dff.DStage=='reviving date']['Date']
        enddate = (stardate + datetime.timedelta(days=124)).values[0]
        endtime = pd.Timestamp(enddate)
        if dff['Date_simphoto'][i]==np.nan or dff['Date_simphoto'][i]>endtime:
            dff['Date_simphoto'][i] = endtime
    dff.loc[dff.DStage != 'reviving date', 'Date_simphoto'] = dff.loc[dff.DStage != 'reviving date', 'Date_simphoto'].apply(
        lambda x: x + datetime.timedelta(days=-1))
    dff['delta_days'] = dff.apply(lambda row: (row.Date_simphoto - row.Date).days, axis=1)
    heading_error = dff.loc[dff.DStage == 'heading date', 'delta_days'].abs().mean()
    maturity_error = dff.loc[dff.DStage == 'maturity date', 'delta_days'].abs().mean()
    print(dff,heading_error,maturity_error)

    dfcumt = thermaldf.groupby('DStage').median()['Thermal_cum'].reset_index()
    dfcumt = dfcumt.sort_values(by=['Thermal_cum']).reset_index()
    mybins2 = dfcumt.Thermal_cum.tolist()
    mybins2.append(9999999)
    mybins2[0] = 0
    dfall['Thermal_Dstage'] = dfall.groupby(['station ID', 'year', 'season'])[['Thermal_cum']].transform(
        lambda x: pd.cut(x, bins=mybins2, labels=dfcumt.DStage.tolist()).astype(str))
    dfcumtall = dfall.drop_duplicates(subset=['station ID', 'year', 'season', 'Thermal_Dstage'])
    dfcumtall = dfcumtall[['station ID', 'year', 'season', 'Date', 'Thermal_Dstage']].rename(
        columns={'Thermal_Dstage': 'DStage', 'Date': 'Date_sim'})
    dffcumt = dfm.merge(dfcumtall, on=['station ID', 'year', 'season', 'DStage'], how='left')
    index2 = dffcumt[(dffcumt['DStage'] == 'maturity date') ].index
    for m in index2:
        ID2=dffcumt['station ID'][m]
        y2=dffcumt['year'][m]
        sea2=dffcumt['season'][m]
        stardate2 = dffcumt[dffcumt['station ID']==ID2][dffcumt.year==y2][dffcumt.season==sea2][dffcumt.DStage=='reviving date']['Date']
        enddate2 = (stardate2 + datetime.timedelta(days=124)).values[0]
        endtime2 = pd.Timestamp(enddate2)
        if dffcumt['Date_sim'][m]==np.nan or dffcumt['Date_sim'][m]>endtime2:
            dffcumt['Date_sim'][m] = endtime2
    dffcumt.loc[dffcumt.DStage != 'reviving date', 'Date_sim'] = dffcumt.loc[dff.DStage != 'reviving date', 'Date_sim'].apply(
        lambda x: x + datetime.timedelta(days=-1))
    dffcumt['delta_days'] = dffcumt.apply(lambda row: (row.Date_sim - row.Date).days, axis=1)
    heading_error2 = dffcumt.loc[dffcumt.DStage == 'heading date', 'delta_days'].abs().mean()
    maturity_error2 = dffcumt.loc[dffcumt.DStage == 'maturity date', 'delta_days'].abs().mean()
    print(dffcumt, heading_error2,maturity_error2)

    ERROR = dff.merge(dffcumt, on=['station ID','lat', 'lon', 'alt','year', 'season','DStage','Date'], how='right')
    ERROR = ERROR.rename(columns={'delta_days_x': 'delta_days_simphoto', 'delta_days_y': 'delta_days_sim'})
    ERROR.to_excel(savepath+'error_Wang_engle_Yin'+flg+'.xlsx', index=False)
    print(ERROR, heading_error,heading_error2,maturity_error,maturity_error2)
    return heading_error

def simulate_and_calibrate_Wang_engle_photoeffect_wofost(Dc, Do, Tbase, Topt, Tcei,Tbase2):
    df = dft.copy()
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
        dfw['Thermal_raw'] = dfw.TemAver.apply(lambda x: Wang_engle(T=x, Tbase=Tbase, Topt=Topt, Tcei=Tcei))
        dfw['Thermal_correct'] = dfw.apply(
            lambda rowt: Tem_correct_Wang_engle(today=rowt.Date, hd=row['heading date'], T=rowt.TemAver,
                                                Tbase2=Tbase2, Topt2=Topt, Tcei2=Tcei, Thermal=rowt.Thermal_raw),axis=1)
        dfw['Thermal_cum'] = dfw.Thermal_correct.cumsum()
        dfw['dayL'] = dfw.Date.apply(
            lambda x: sun.dayCivilTwilightLength(year=x.year, month=x.month, day=x.day, lon=row.lon, lat=row.lat))
        dfw['photo_raw'] = dfw.dayL.apply(lambda x: photoeffect_wofost(DL=x, Dc=Dc, Do=Do))
        dfw['photo'] = dfw.apply(
            lambda rowt: photo_effect_correct(today=rowt.Date, revd=row['reviving date'], jd=row['jointing date'],
                                              hd=row['heading date'], photo=rowt.photo_raw), axis=1)
        dfw['photothermal'] = dfw.photo * dfw.Thermal_correct
        dfw['photothermal_cum'] = dfw['photothermal'].cumsum()
        dfw['station ID'] = row['station ID']
        dfw['year'] = row['year']
        dfw['season'] = row['season']
        dfall = pd.concat([dfall, dfw])

    thermaldf = dfall.merge(dfm, on=['station ID', 'year', 'Date', 'season'], how='right')
    # thermaldf.to_excel('../data/run_phen_model_result/thermaldf.xlsx', index=False)
    dfp = thermaldf.groupby('DStage').median()['photothermal_cum'].reset_index()
    dfp = dfp.sort_values(by=['photothermal_cum']).reset_index()
    mybins = dfp.photothermal_cum.tolist()
    mybins.append(9999999)
    mybins[0] = 0
    print(mybins)
    dfall['PhotoThermal_Dstage'] = dfall.groupby(['station ID', 'year', 'season'])[['photothermal_cum']].transform(
        lambda x: pd.cut(x, bins=mybins, labels=dfp.DStage.tolist()).astype(str))
    dfpall = dfall.drop_duplicates(subset=['station ID', 'year', 'season', 'PhotoThermal_Dstage'])

    dfpall = dfpall[['station ID', 'year', 'season', 'Date', 'PhotoThermal_Dstage']].rename(
        columns={'PhotoThermal_Dstage': 'DStage', 'Date': 'Date_simphoto'})

    dff = dfm.merge(dfpall, on=['station ID', 'year', 'season', 'DStage'], how='left')
    index = dff[(dff['DStage'] == 'maturity date') ].index
    for i in index:
        ID=dff['station ID'][i]
        y=dff['year'][i]
        sea=dff['season'][i]
        stardate = dff[dff['station ID']==ID][dff.year==y][dff.season==sea][dff.DStage=='reviving date']['Date']
        enddate = (stardate + datetime.timedelta(days=124)).values[0]
        endtime = pd.Timestamp(enddate)
        if dff['Date_simphoto'][i]==np.nan or dff['Date_simphoto'][i]>endtime:
            dff['Date_simphoto'][i] = endtime
    dff.loc[dff.DStage != 'reviving date', 'Date_simphoto'] = dff.loc[dff.DStage != 'reviving date', 'Date_simphoto'].apply(
        lambda x: x + datetime.timedelta(days=-1))
    dff['delta_days'] = dff.apply(lambda row: (row.Date_simphoto - row.Date).days, axis=1)
    heading_error = dff.loc[dff.DStage == 'heading date', 'delta_days'].abs().mean()
    maturity_error = dff.loc[dff.DStage == 'maturity date', 'delta_days'].abs().mean()
    print(dff,heading_error,maturity_error)

    dfcumt = thermaldf.groupby('DStage').median()['Thermal_cum'].reset_index()
    dfcumt = dfcumt.sort_values(by=['Thermal_cum']).reset_index()
    mybins2 = dfcumt.Thermal_cum.tolist()
    mybins2.append(9999999)
    mybins2[0] = 0
    dfall['Thermal_Dstage'] = dfall.groupby(['station ID', 'year', 'season'])[['Thermal_cum']].transform(
        lambda x: pd.cut(x, bins=mybins2, labels=dfcumt.DStage.tolist()).astype(str))
    dfcumtall = dfall.drop_duplicates(subset=['station ID', 'year', 'season', 'Thermal_Dstage'])
    dfcumtall = dfcumtall[['station ID', 'year', 'season', 'Date', 'Thermal_Dstage']].rename(
        columns={'Thermal_Dstage': 'DStage', 'Date': 'Date_sim'})
    dffcumt = dfm.merge(dfcumtall, on=['station ID', 'year', 'season', 'DStage'], how='left')
    index2 = dffcumt[(dffcumt['DStage'] == 'maturity date') ].index
    for m in index2:
        ID2=dffcumt['station ID'][m]
        y2=dffcumt['year'][m]
        sea2=dffcumt['season'][m]
        stardate2 = dffcumt[dffcumt['station ID']==ID2][dffcumt.year==y2][dffcumt.season==sea2][dffcumt.DStage=='reviving date']['Date']
        enddate2 = (stardate2 + datetime.timedelta(days=124)).values[0]
        endtime2 = pd.Timestamp(enddate2)
        if dffcumt['Date_sim'][m]==np.nan or dffcumt['Date_sim'][m]>endtime2:
            dffcumt['Date_sim'][m] = endtime2
    dffcumt.loc[dffcumt.DStage != 'reviving date', 'Date_sim'] = dffcumt.loc[dff.DStage != 'reviving date', 'Date_sim'].apply(
        lambda x: x + datetime.timedelta(days=-1))
    dffcumt['delta_days'] = dffcumt.apply(lambda row: (row.Date_sim - row.Date).days, axis=1)
    heading_error2 = dffcumt.loc[dffcumt.DStage == 'heading date', 'delta_days'].abs().mean()
    maturity_error2 = dffcumt.loc[dffcumt.DStage == 'maturity date', 'delta_days'].abs().mean()
    print(dffcumt, heading_error2,maturity_error2)

    ERROR = dff.merge(dffcumt, on=['station ID','lat', 'lon', 'alt','year', 'season','DStage','Date'], how='right')
    ERROR = ERROR.rename(columns={'delta_days_x': 'delta_days_simphoto', 'delta_days_y': 'delta_days_sim'})
    ERROR.to_excel(savepath+'error_Wang_engle_WOFOST'+flg+'.xlsx', index=False)
    print(ERROR, heading_error,heading_error2,maturity_error,maturity_error2)
    return heading_error

def simulate_and_calibrate_Wang_engle_photoeffect_oryza2000(Dc, Tbase, Topt, Tcei,Tbase2):
    df = dft.copy()
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
        dfw['Thermal_raw'] = dfw.TemAver.apply(lambda x: Wang_engle(T=x, Tbase=Tbase, Topt=Topt, Tcei=Tcei))
        dfw['Thermal_correct'] = dfw.apply(
            lambda rowt: Tem_correct_Wang_engle(today=rowt.Date, hd=row['heading date'], T=rowt.TemAver,
                                                Tbase2=Tbase2, Topt2=Topt, Tcei2=Tcei, Thermal=rowt.Thermal_raw),
            axis=1)
        dfw['Thermal_cum'] = dfw.Thermal_correct.cumsum()
        dfw['dayL'] = dfw.Date.apply(
            lambda x: sun.dayCivilTwilightLength(year=x.year, month=x.month, day=x.day, lon=row.lon, lat=row.lat))
        dfw['photo_raw'] = dfw.dayL.apply(lambda x: photoeffect_oryza2000(DL=x, Dc=Dc))
        dfw['photo'] = dfw.apply(
            lambda rowt: photo_effect_correct(today=rowt.Date, revd=row['reviving date'], jd=row['jointing date'],
                                              hd=row['heading date'], photo=rowt.photo_raw), axis=1)
        dfw['photothermal'] = dfw.photo * dfw.Thermal_correct
        dfw['photothermal_cum'] = dfw['photothermal'].cumsum()
        dfw['station ID'] = row['station ID']
        dfw['year'] = row['year']
        dfw['season'] = row['season']
        dfall = pd.concat([dfall, dfw])

    thermaldf = dfall.merge(dfm, on=['station ID', 'year', 'Date', 'season'], how='right')
    # thermaldf.to_excel('../data/run_phen_model_result/thermaldf.xlsx', index=False)
    dfp = thermaldf.groupby('DStage').median()['photothermal_cum'].reset_index()
    dfp = dfp.sort_values(by=['photothermal_cum']).reset_index()
    mybins = dfp.photothermal_cum.tolist()
    mybins.append(9999999)
    mybins[0] = 0
    print(mybins)
    dfall['PhotoThermal_Dstage'] = dfall.groupby(['station ID', 'year', 'season'])[['photothermal_cum']].transform(
        lambda x: pd.cut(x, bins=mybins, labels=dfp.DStage.tolist()).astype(str))
    dfpall = dfall.drop_duplicates(subset=['station ID', 'year', 'season', 'PhotoThermal_Dstage'])

    dfpall = dfpall[['station ID', 'year', 'season', 'Date', 'PhotoThermal_Dstage']].rename(
        columns={'PhotoThermal_Dstage': 'DStage', 'Date': 'Date_simphoto'})

    dff = dfm.merge(dfpall, on=['station ID', 'year', 'season', 'DStage'], how='left')
    index = dff[(dff['DStage'] == 'maturity date') ].index
    for i in index:
        ID=dff['station ID'][i]
        y=dff['year'][i]
        sea=dff['season'][i]
        stardate = dff[dff['station ID']==ID][dff.year==y][dff.season==sea][dff.DStage=='reviving date']['Date']
        enddate = (stardate + datetime.timedelta(days=124)).values[0]
        endtime = pd.Timestamp(enddate)
        if dff['Date_simphoto'][i]==np.nan or dff['Date_simphoto'][i]>endtime:
            dff['Date_simphoto'][i] = endtime
    dff.loc[dff.DStage != 'reviving date', 'Date_simphoto'] = dff.loc[dff.DStage != 'reviving date', 'Date_simphoto'].apply(
        lambda x: x + datetime.timedelta(days=-1))
    dff['delta_days'] = dff.apply(lambda row: (row.Date_simphoto - row.Date).days, axis=1)
    heading_error = dff.loc[dff.DStage == 'heading date', 'delta_days'].abs().mean()
    maturity_error = dff.loc[dff.DStage == 'maturity date', 'delta_days'].abs().mean()
    print(dff,heading_error,maturity_error)


    dfcumt = thermaldf.groupby('DStage').median()['Thermal_cum'].reset_index()
    dfcumt = dfcumt.sort_values(by=['Thermal_cum']).reset_index()
    mybins2 = dfcumt.Thermal_cum.tolist()
    mybins2.append(9999999)
    mybins2[0] = 0
    dfall['Thermal_Dstage'] = dfall.groupby(['station ID', 'year', 'season'])[['Thermal_cum']].transform(
        lambda x: pd.cut(x, bins=mybins2, labels=dfcumt.DStage.tolist()).astype(str))
    dfcumtall = dfall.drop_duplicates(subset=['station ID', 'year', 'season', 'Thermal_Dstage'])
    dfcumtall = dfcumtall[['station ID', 'year', 'season', 'Date', 'Thermal_Dstage']].rename(
        columns={'Thermal_Dstage': 'DStage', 'Date': 'Date_sim'})
    dffcumt = dfm.merge(dfcumtall, on=['station ID', 'year', 'season', 'DStage'], how='left')
    index2 = dffcumt[(dffcumt['DStage'] == 'maturity date') ].index
    for m in index2:
        ID2=dffcumt['station ID'][m]
        y2=dffcumt['year'][m]
        sea2=dffcumt['season'][m]
        stardate2 = dffcumt[dffcumt['station ID']==ID2][dffcumt.year==y2][dffcumt.season==sea2][dffcumt.DStage=='reviving date']['Date']
        enddate2 = (stardate2 + datetime.timedelta(days=124)).values[0]
        endtime2 = pd.Timestamp(enddate2)
        if dffcumt['Date_sim'][m]==np.nan or dffcumt['Date_sim'][m]>endtime2:
            dffcumt['Date_sim'][m] = endtime2
    dffcumt.loc[dffcumt.DStage != 'reviving date', 'Date_sim'] = dffcumt.loc[dff.DStage != 'reviving date', 'Date_sim'].apply(
        lambda x: x + datetime.timedelta(days=-1))
    dffcumt['delta_days'] = dffcumt.apply(lambda row: (row.Date_sim - row.Date).days, axis=1)
    heading_error2 = dffcumt.loc[dffcumt.DStage == 'heading date', 'delta_days'].abs().mean()
    maturity_error2 = dffcumt.loc[dffcumt.DStage == 'maturity date', 'delta_days'].abs().mean()
    print(dffcumt, heading_error2,maturity_error2)

    ERROR = dff.merge(dffcumt, on=['station ID','lat', 'lon', 'alt','year', 'season','DStage','Date'], how='right')
    ERROR = ERROR.rename(columns={'delta_days_x': 'delta_days_simphoto', 'delta_days_y': 'delta_days_sim'})
    ERROR.to_excel(savepath+'error_Wang_engle_oryza'+flg+'.xlsx', index=False)
    print(ERROR, heading_error,heading_error2,maturity_error,maturity_error2)
    return maturity_error

def opsite():
    global savepath
    savepath = '../data/individuals/general_parameters/result_mat_Tbase2_days/'
    dir_path = '../data/individuals/data'
    for root, dirs, files in os.walk(dir_path):
        for file in files:
            file_path = os.path.join(root, file)
            print(file_path)
            data = pd.read_csv(file_path,
                               parse_dates=['reviving date', 'tillering date', 'jointing date', 'booting date',
                                            'heading date', 'maturity date'])
            global dft, flg
            dft = data
            flg = file_path[-9:-4]
            simulate_and_calibrate_T_base_opt_photoeffect_yin(mu=-15.46, zeta=2.06, ep=2.48, Tbase=8, Topt=30, Tbase2=4)
            simulate_and_calibrate_T_base_opt_photoeffect_wofost(Dc=16, Do=12.5, Tbase=8, Topt=30, Tbase2=4)
            simulate_and_calibrate_T_base_opt_photoeffect_oryza2000(Dc=12.5, Tbase=8, Topt=30, Tbase2=4)
            simulate_and_calibrate_T_base_opt_ceiling_photoeffect_yin(mu=-15.46, zeta=2.06, ep=2.48, Tbase=8,Topt_low=25, Topt_high=35, Tcei=42, Tbase2=4)
            simulate_and_calibrate_T_base_opt_ceiling_photoeffect_wofost(Dc=16, Do=12.5, Tbase=8, Topt_low=25,Topt_high=35, Tcei=42, Tbase2=4)
            simulate_and_calibrate_T_base_opt_ceiling_photoeffect_oryza2000(Dc=12.5, Tbase=8, Topt_low=25, Topt_high=35,Tcei=42, Tbase2=4)
            simulate_and_calibrate_Wang_engle_photoeffect_yin(mu=-15.46, zeta=2.06, ep=2.48, Tbase=8, Topt=30, Tcei=42,Tbase2=4)
            simulate_and_calibrate_Wang_engle_photoeffect_wofost(Dc=16, Do=12.5, Tbase=8, Topt=30, Tcei=42, Tbase2=4)
            simulate_and_calibrate_Wang_engle_photoeffect_oryza2000(Dc=12.5, Tbase=8, Topt=30, Tcei=42, Tbase2=4)

if __name__ == "__main__":
    opsite()


