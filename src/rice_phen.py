# -*- coding: utf-8 -*-
import pandas as pd
import numpy
import math
from pylab import *
import Sun
import datetime
import seaborn as sns
import matplotlib.pyplot as plt

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
    df = df.loc[(df.Date >= start) & (df.Date <= (end+datetime.timedelta(days=30))), ['Date', 'TemAver']]
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
                     parse_dates=['reviving date', 'tillering date', 'jointing date',
                                  'booting date', 'heading date','maturity date'])
    df['season']= df.groupby(['station ID', 'year']).cumcount()+1
    dfmm = df[['station ID', 'lat', 'lon', 'alt', 'year', 'season',
               'reviving date', 'tillering date', 'jointing date',
               'booting date', 'heading date','maturity date']]
    dfm = pd.melt(dfmm, id_vars=['station ID', 'lat', 'lon', 'alt', 'year', 'season'],
                  value_vars=['reviving date', 'tillering date', 'jointing date',
                              'booting date', 'heading date','maturity date'])
    dfm = dfm.rename(columns={'value': 'Date', 'variable': 'DStage'})
    if pd.isna(jd):
        jd = revd + datetime.timedelta(days=25)
    if pd.isna(hd):
        jd = revd + datetime.timedelta(days=55)
    print(dfm)
    dfall = pd.DataFrame()
    for ind, row in df.iterrows():
        dfw = read_station_weather(row['station ID'], row['reviving date'], row['maturity date'])
        dfw['Thermal'] = dfw.TemAver.apply(lambda x: Wang_engle(T=x))
        dfw['Thermal_cum'] = dfw.Thermal.cumsum()
        dfw['dayL'] = dfw.Date.apply(
            lambda x: sun.dayCivilTwilightLength(year=x.year, month=x.month, day=x.day, lon=row.lon, lat=row.lat))
        dfw['photo_raw'] = dfw.dayL.apply(lambda x: photo_period_based_on_yin(dl=x))
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
    df.to_excel('../data/dfall.xlsx', index=False)
    thermaldf = dfall.merge(dfm, on=['station ID', 'year', 'Date', 'season'], how='left')
    thermaldf.to_excel('../data/thermaldfall.xlsx', index=False)



def Calculation_error_days():
    write = pd.ExcelWriter('../data/error_days.xlsx')
    # Calculate the median accumulated temperature
    dailydf = pd.read_excel('../data/thermaldfall.xlsx')  #Observed daily accumulated temperature
    #Extract Data according to different ripeness of rice
    earlydf = pd.DataFrame()  # Single season rice and double season early rice
    latedf = pd.DataFrame()   # Double season late rice
    rev = dailydf[dailydf['DStage'] == 'reviving date']
    rev = rev.reset_index(drop=True)
    for n in range(rev.shape[0]):
        if rev['Date'][n].month <= 6:
            early = dailydf[dailydf['station ID'] == rev['station ID'][n]][dailydf['year'] ==
                                                                           rev['year'][n]][dailydf['season'] == rev['season'][n]]
            earlydf = pd.concat([earlydf, early])
        else:
            late = dailydf[dailydf['station ID'] == rev['station ID'][n]][dailydf['year'] ==
                                                                          rev['year'][n]][dailydf['season'] == rev['season'][n]]
            latedf = pd.concat([latedf, late])
    latedf = latedf.reset_index(drop=True)
    earlydf = earlydf.reset_index(drop=True)
    thermal_median_early = earlydf['Thermal_cum'].groupby(earlydf['DStage']).median()
    thermal_median_early = thermal_median_early.sort_values()
    print('thermal_median_early', thermal_median_early)
    thermal_median_late = latedf['Thermal_cum'].groupby(latedf['DStage']).median()
    thermal_median_late = thermal_median_late.sort_values()
    print('thermal_median_late', thermal_median_late)
    photothermal_median_early = earlydf['photothermal_cum'].groupby(earlydf['DStage']).median()
    photothermal_median_early = photothermal_median_early.sort_values()
    print('photothermal_median_early', photothermal_median_early)
    photothermal_median_late = latedf['photothermal_cum'].groupby(latedf['DStage']).median()
    photothermal_median_late = photothermal_median_late.sort_values()
    print('photothermal_median_late', photothermal_median_late)

    # Calculate simulated days and observed days
    bins = [0, 132.871220, 526.287498, 856.900277, 1021.053015, 1501.593876, 25000]  # median_thermal():
    labes = ['reviving date', 'tillering date', 'jointing date', 'booting date', 'heading date', 'maturity date']
    phobins = [0,132.871220, 523.123174, 773.711641, 908.889552, 1400.536426, 25000]
    pholabes = ['reviving date', 'tillering date', 'jointing date', 'booting date', 'heading date', 'maturity date']
    earlydf['Thermal_Dstage'] = pd.cut(earlydf['Thermal_cum'], bins=bins, labels=labes)
    earlydf['Phothermal_Dstage'] = pd.cut(earlydf['photothermal_cum'], bins=phobins, labels=pholabes)
    earlydf['obser_Dstage'] = earlydf['DStage'].ffill()
    obser_days = earlydf['obser_Dstage'].groupby([earlydf['station ID'], earlydf['year'], earlydf['season'],
                                                  earlydf['obser_Dstage']]).count().unstack().iloc[:,[4, 5, 2, 0, 1, 3]]

    obser_days.to_excel(write, sheet_name='early_obser_days', index=True)
    sim_days_thermal = earlydf['Thermal_Dstage'].groupby([earlydf['station ID'], earlydf['year'],
                                                          earlydf['season'], earlydf['Thermal_Dstage']]).count()
    sim_days_phothermal = earlydf['Phothermal_Dstage'].groupby([earlydf['station ID'], earlydf['year'],
                                                                earlydf['season'], earlydf['Phothermal_Dstage']]).count()

    bins2 = [0, 166.382740, 600.055712, 827.298495, 965.626885, 1345.368864, 25000]
    labes2 = ['reviving date', 'tillering date', 'jointing date', 'booting date', 'heading date', 'maturity date']
    phobins2 = [0, 166.382740, 597.838510, 803.419459, 927.428064, 1317.251906, 25000]
    pholabes2 = ['reviving date', 'tillering date', 'jointing date', 'booting date', 'heading date', 'maturity date']
    latedf['Thermal_Dstage'] = pd.cut(latedf['Thermal_cum'], bins=bins2, labels=labes2)
    latedf['Phothermal_Dstage'] = pd.cut(latedf['photothermal_cum'], bins=phobins2, labels=pholabes2)
    latedf['obser_Dstage'] = latedf['DStage'].ffill()
    obser_days2 = latedf['obser_Dstage'].groupby([latedf['station ID'], latedf['year'],
                                                  latedf['season'], latedf['obser_Dstage']]).count().unstack().iloc[:, [4, 5, 2, 0, 1, 3]]
    obser_days2.to_excel(write, sheet_name='late_obser_days', index=True)
    sim_days_thermal2 = latedf['Thermal_Dstage'].groupby([latedf['station ID'], latedf['year'],
                                                          latedf['season'], latedf['Thermal_Dstage']]).count()
    sim_days_phothermal2 = latedf['Phothermal_Dstage'].groupby([latedf['station ID'], latedf['year'],
                                                                latedf['season'], latedf['Phothermal_Dstage']]).count()


    # Calculation error days for Single season rice and double season early rice
    Revised_obser = obser_days.reset_index()
    index_tillering = Revised_obser['tillering date'].isnull()
    Revised_obser['reviving date'][index_tillering] = np.NaN
    index_jointing = Revised_obser['jointing date'].isnull()
    Revised_obser['tillering date'][index_jointing] = np.NaN
    index_booting = Revised_obser['booting date'].isnull()
    Revised_obser['jointing date'][index_booting] = np.NaN
    Revised_obser = Revised_obser.set_index(['station ID', 'year', 'season'])
    early_thermal_results = sim_days_thermal.unstack() - Revised_obser
    early_phothermal_results = sim_days_phothermal.unstack() - Revised_obser
    result = early_thermal_results.merge(early_phothermal_results, left_index=True, right_index=True)
    result.to_excel(write, sheet_name='early_result', index=True)

    #Calculation error days for Double season late rice
    Revised_obser2 = obser_days2.reset_index()
    index_tillering2 = Revised_obser2['tillering date'].isnull()
    Revised_obser2['reviving date'][index_tillering2] = np.NaN
    index_jointing2 = Revised_obser2['jointing date'].isnull()
    Revised_obser2['tillering date'][index_jointing2] = np.NaN
    Revised_obser2  = Revised_obser2.set_index(['station ID', 'year', 'season'])
    late_thermal_results = sim_days_thermal2.unstack() - Revised_obser2
    late_phothermal_results = sim_days_phothermal2.unstack() - Revised_obser2
    result2 = late_thermal_results.merge(late_phothermal_results, left_index=True, right_index=True)
    result2.to_excel(write, sheet_name='late_result', index=True)
    write.save()

def error_boxplot(errordf, errordf2):
    errordf = errordf.melt()
    errordf2 = errordf2.melt()
    error1 = errordf.rename(columns={'value': 'Error days', 'variable': 'DStage'})
    error2 = errordf2.rename(columns={'value': 'Error days', 'variable': 'DStage'})
    a = pd.concat([error1, error2], axis=0).reset_index(drop=True)
    ind1 = a[a['DStage'].str.contains('maturity date')].index.tolist()
    a = a.drop(index=ind1)
    a["class"] = a["DStage"]
    d1 = a[a['class'].str.contains('_x')].index.tolist()
    d2 = a[a['class'].str.contains('_y')].index.tolist()
    rev = a[a['DStage'].str.contains('reviving date')].index.tolist()
    til = a[a['DStage'].str.contains('tillering date')].index.tolist()
    joi = a[a['DStage'].str.contains('jointing date')].index.tolist()
    boot = a[a['DStage'].str.contains('booting date')].index.tolist()
    hea = a[a['DStage'].str.contains('heading date')].index.tolist()
    for i in d1:
        a['class'].replace(a['class'][i], 'Thermal', inplace=True)
    for l in d2:
        a['class'].replace(a['class'][l], 'Photothermal', inplace=True)
    for r in rev:
        a['DStage'].replace(a['DStage'][r], 'Rev-Til', inplace=True)
    for t in til:
        a['DStage'].replace(a['DStage'][t], 'Til-Joi', inplace=True)
    for j in joi:
        a['DStage'].replace(a['DStage'][j], 'Joi-Boot', inplace=True)
    for b in boot:
        a['DStage'].replace(a['DStage'][b], 'Boot-Hea', inplace=True)
    for h in hea:
        a['DStage'].replace(a['DStage'][h], 'Hea-Mat', inplace=True)

    plt.figure(figsize=(12, 7))
    g = sns.boxplot(data=a, x="Error days", y="DStage", hue="class", palette='Set2',
                    medianprops={'linestyle': '-', 'color': 'black'},
                    showmeans=True, meanprops={'marker': 'o', 'markerfacecolor': 'red', 'markeredgecolor': 'white'})
    plt.ylabel('Phases', fontsize=18)
    plt.xlabel('Error days', fontsize=18)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    g.legend_.set_title(None)
    plt.setp(g.get_legend().get_texts(), fontsize='14')
    show()
    plt.savefig('..data/Erro days.png', dpi=300)


if __name__ == '__main__':
    # errordf = pd.read_excel('../data/error_days.xlsx', sheet_name='early_result', index_col=[0,1,2])
    # errordf2 = pd.read_excel('../data/error_days.xlsx', sheet_name='late_result', index_col=[0,1,2])
    # error_boxplot(errordf, errordf2)






