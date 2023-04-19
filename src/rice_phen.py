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
os.chdir(os.path.dirname(os.path.realpath(__file__)))

pd.options.display.max_columns = 999


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

def photoeffect_oryza200(DL, MOPP=11.5, PPSE=0.2):
    # MOPP = 11.5
    # PPSE = 0.2
    if DL < MOPP:
        PPFAC = 1.
    else:
        PPFAC = 1. - (DL - MOPP) * PPSE
        PPFAC = np.min([1., np.max([0., PPFAC])])
    return PPFAC
def photo_effect_correct(today, revd, jd, hd, photo):
    if pd.isna(jd):
        jd = revd + datetime.timedelta(days=25)
    if pd.isna(hd):
        hd = revd + datetime.timedelta(days=55)
    if today < jd or today > hd:
        return 1
    else:
        return photo


def Test_wang_engle():
    plot(range(0, 45), [Wang_engle(T=T) for T in range(45)])
    show()


def Test_Yin_photo():
    plot(range(1, 24), [photoeffect_yin(dl=dl) for dl in range(1, 24)])
    show()
def Test_photoeffect2():
    plot(range(1, 24), [photoeffect_wofost(DL=dl) for dl in range(1, 24)])
    show()
def Test_photoeffect3():
    plot(range(1, 24), [photoeffect_oryza200(DL=dl) for dl in range(1, 24)])
    show()

def read_station_weather(station, start, end):
    df = pd.read_table("../data/Meteo(48 sta)/" + str(station) + ".txt", encoding='gbk', sep=' * ', engine='python',
                       skiprows=[1])
    df['Date'] = df.apply(lambda row: pd.to_datetime('%d-%d-%d' % (row.YY, row.mm, row.dd)), axis=1)
    df = df.loc[(df.Date >= start) & (df.Date <= (end+datetime.timedelta(days=30))), ['Date', 'TemAver']]
    return df


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
                              'booting date', 'heading date', 'maturity date'])
    dfm = dfm.rename(columns={'value': 'Date', 'variable': 'DStage'})
    print(dfm)
    dfall = pd.DataFrame()
    for ind, row in df.iterrows():
        dfw = read_station_weather(row['station ID'], row['reviving date'], row['maturity date'])
        dfw['Thermal'] = dfw.TemAver.apply(lambda x: Wang_engle(T=x))
        dfw['Thermal_cum'] = dfw.Thermal.cumsum()
        dfw['dayL'] = dfw.Date.apply(
            lambda x: sun.dayCivilTwilightLength(year=x.year, month=x.month, day=x.day, lon=row.lon, lat=row.lat))
        dfw['photo_raw'] = dfw.dayL.apply(lambda x: photoeffect_yin(DL=x))
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


def cor_erroes_temp():
    errordf = pd.read_excel('../data/correlation_error_tem.xlsx',
                            sheet_name='cor_early', index_col=[0,1,2])
    errordf2 = pd.read_excel('../data/correlation_error_tem.xlsx',
                             sheet_name='cor_late',index_col=[0, 1, 2])
    inte_cor = pd.concat([errordf, errordf2]).reset_index()
    Rev_Thermal = inte_cor.loc[:, ['Rev_Thermal_T', 'Rev_Thermal_D']].dropna()
    Til_Thermal = inte_cor.loc[:, ['Til_Thermal_T', 'Til_Thermal_D']].dropna()
    Joi_Thermal = inte_cor.loc[:, ['Joi_Thermal_T', 'Joi_Thermal_D']].dropna()
    Boot_Thermal = inte_cor.loc[:, ['Boot_Thermal_T', 'Boot_Thermal_D']].dropna()
    Hea_Thermal = inte_cor.loc[:, ['Hea_Thermal_T', 'Hea_Thermal_D']].dropna()
    Rev_Phothermal = inte_cor.loc[:, ['Rev_Phothermal_T', 'Rev_Phothermal_D']].dropna()
    Til_Phothermal = inte_cor.loc[:, ['Til_Phothermal_T', 'Til_Phothermal_D']].dropna()
    Joi_Phothermal = inte_cor.loc[:, ['Joi_Phothermal_T', 'Joi_Phothermal_D']].dropna()
    Boot_Phothermal = inte_cor.loc[:, ['Boot_Phothermal_T', 'Boot_Phothermal_D']].dropna()
    Hea_Phothermal = inte_cor.loc[:, ['Hea_Phothermal_T', 'Hea_Phothermal_D']].dropna()

    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    fig = plt.figure(figsize=(20, 14.5))
    plt.rcParams['font.size'] = 18
    ax = plt.axes()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.subplot(2, 3, 1)
    plt.title('Rev-Til', fontsize='xx-large', fontweight='normal', size=25)
    plt.xlim(xmin=10, xmax=40)
    plt.ylim(ymin=-40, ymax=40)
    x1 = np.asarray(Rev_Thermal[['Rev_Thermal_T']])
    y1 = np.asarray(Rev_Thermal[['Rev_Thermal_D']])
    reg = LinearRegression().fit(x1, y1)
    print("一元回归方程为:  Y = %.2fX + (%.2f)" % (reg.coef_[0][0], reg.intercept_[0]))
    print(scipy.stats.spearmanr(x1, y1))
    sns.scatterplot(data=Rev_Thermal, x='Rev_Thermal_T', y='Rev_Thermal_D', color=".2", s=50, legend=False)
    plt.plot(x1, reg.predict(x1), linewidth='1', color='red')
    plt.xlabel('Tave(℃)', size=23)
    plt.ylabel('Error days(d)', size=25)
    plt.text(14, 30, "$Y = %.2fX + %.2f$" % (reg.coef_[0][0], reg.intercept_[0]), fontsize=18, weight="heavy")
    plt.text(14, 23, "$R^2= %.2f $" % r2_score(y1, reg.predict(x1)), fontsize=18, weight="heavy")
    plt.text(22, 23, "$ p = %.2f $" % scipy.stats.spearmanr(x1, reg.predict(x1))[1], fontsize=18, weight="heavy")
    plt.text(11, 35, '$ (a) $ ', fontsize=18, weight="heavy")
    plt.tick_params(labelsize=18)

    plt.subplot(2, 3, 2)
    plt.title('Til-Joi', fontsize='xx-large', fontweight='normal', size=25)
    plt.xlim(xmin=10, xmax=40)
    plt.ylim(ymin=-40, ymax=40)
    x2 = np.asarray(Til_Thermal[['Til_Thermal_T']])
    y2 = np.asarray(Til_Thermal[['Til_Thermal_D']])
    reg = LinearRegression().fit(x2, y2)
    print("一元回归方程为:  Y = %.2fX + (%.2f)" % (reg.coef_[0][0], reg.intercept_[0]))
    print(scipy.stats.spearmanr(x2, y2))
    sns.scatterplot(data=Til_Thermal, x='Til_Thermal_T', y='Til_Thermal_D', color=".2", s=50, legend=False)
    plt.plot(x2, reg.predict(x2), linewidth='1', color='red')
    plt.xlabel('Tave(℃)', size=23)
    plt.ylabel('Error days(d)', size=25)
    plt.text(14, 30, "$Y = %.2fX + %.2f$" % (reg.coef_[0][0], reg.intercept_[0]), fontsize=18, weight="heavy")
    plt.text(14, 23, "$R^2= %.2f $" % r2_score(y2, reg.predict(x2)), fontsize=18, weight="heavy")
    plt.text(14, 18, "$ p = %.2f $" % scipy.stats.spearmanr(x2, reg.predict(x2))[1], fontsize=18, weight="heavy")
    plt.text(11, 35, '$ (b) $ ', fontsize=18, weight="heavy")
    plt.tick_params(labelsize=18)

    plt.subplot(2, 3, 3)
    plt.title('Joi-Boot', fontsize='xx-large', fontweight='normal', size=25)
    plt.xlim(xmin=10, xmax=40)
    plt.ylim(ymin=-40, ymax=40)
    x3 = np.asarray(Joi_Thermal[['Joi_Thermal_T']])
    y3 = np.asarray(Joi_Thermal[['Joi_Thermal_D']])
    reg = LinearRegression().fit(x3, y3)
    print("一元回归方程为:  Y = %.2fX + (%.2f)" % (reg.coef_[0][0], reg.intercept_[0]))
    print(scipy.stats.spearmanr(x3, y3))
    sns.scatterplot(data=Joi_Thermal, x='Joi_Thermal_T', y='Joi_Thermal_D', color=".2", s=50, legend=False)
    plt.plot(x3, reg.predict(x3), linewidth='1', color='red')
    plt.xlabel('Tave(℃)', size=23)
    plt.ylabel('Error days(d)', size=25)
    plt.text(14, 30, "$Y = %.2fX + %.2f$" % (reg.coef_[0][0], reg.intercept_[0]), fontsize=18, weight="heavy")
    plt.text(14, 23, "$R^2= %.2f $" % r2_score(y3, reg.predict(x3)), fontsize=18, weight="heavy")
    plt.text(22, 23, '$ p = %.2f $ ' % scipy.stats.spearmanr(x3, reg.predict(x3))[1], fontsize=18, weight="heavy")
    plt.text(11, 35, '$ (c) $ ', fontsize=18, weight="heavy")
    plt.tick_params(labelsize=18)

    plt.subplot(2, 3, 4)
    plt.title('Joi-Boot', fontsize='xx-large', fontweight='normal', size=25)
    plt.xlim(xmin=10, xmax=40)
    plt.ylim(ymin=-40, ymax=40)
    x4 = np.asarray(Boot_Thermal[['Boot_Thermal_T']])
    y4 = np.asarray(Boot_Thermal[['Boot_Thermal_D']])
    reg = LinearRegression().fit(x4, y4)
    print("一元回归方程为:  Y = %.2fX + (%.2f)" % (reg.coef_[0][0], reg.intercept_[0]))
    print(scipy.stats.spearmanr(x4, y4))
    sns.scatterplot(data=Boot_Thermal, x='Boot_Thermal_T', y='Boot_Thermal_D', color=".2", s=50, legend=False)
    plt.plot(x4, reg.predict(x4), linewidth='1', color='red')
    plt.xlabel('Tave(℃)', size=23)
    plt.ylabel('Error days(d)', size=25)
    plt.text(14, 30, "$Y = %.2fX + %.2f$" % (reg.coef_[0][0], reg.intercept_[0]), fontsize=15, weight="heavy")
    plt.text(14, 23, "$R^2= %.2f $" % r2_score(y4, reg.predict(x4)), fontsize=18, weight="heavy")
    plt.text(22, 23, '$ p = %.2f $ ' % scipy.stats.spearmanr(x4,reg.predict(x4))[1], fontsize=18, weight="heavy")
    plt.text(11, 35, '$ (c) $ ', fontsize=18, weight="heavy")
    plt.tick_params(labelsize=18)

    plt.subplot(2, 3, 5)
    plt.title('Joi-Boot', fontsize='xx-large', fontweight='normal', size=25)
    plt.xlim(xmin=10, xmax=40)
    plt.ylim(ymin=-40, ymax=40)
    x5 = np.asarray(Hea_Thermal[['Hea_Thermal_T']])
    y5 = np.asarray(Hea_Thermal[['Hea_Thermal_D']])
    reg = LinearRegression().fit(x5, y5)
    print("一元回归方程为:  Y = %.2fX + (%.2f)" % (reg.coef_[0][0], reg.intercept_[0]))
    print(scipy.stats.spearmanr(x5, y5))
    sns.scatterplot(data=Hea_Thermal, x='Hea_Thermal_T', y='Hea_Thermal_D', color=".2", s=50, legend=False)
    plt.plot(x5, reg.predict(x5), linewidth='1', color='red')
    plt.xlabel('Tave(℃)', size=23)
    plt.ylabel('Error days(d)', size=25)
    plt.text(14, 35, "$Y = %.2fX + %.2f$" % (reg.coef_[0][0], reg.intercept_[0]), fontsize=18, weight="heavy")
    plt.text(22, 28, "$R^2= %.2f $" % r2_score(y5, reg.predict(x5)), fontsize=18, weight="heavy")
    plt.text(30, 28, '$ p = %.2f $ ' % scipy.stats.spearmanr(x5, reg.predict(x5))[1], fontsize=18, weight="heavy")
    plt.text(11, 35, '$ (c) $ ', fontsize=18, weight="heavy")
    plt.tick_params(labelsize=18)
    fig.savefig(".../fig/cor_errors_tem.png", dpi=300)


def Calculate_Sim_date():
    write = pd.ExcelWriter('../data/Sim_date.xlsx')
    dailydf = pd.read_excel('../data/thermaldfall.xlsx')
    earlydf = pd.DataFrame()
    latedf = pd.DataFrame()
    rev = dailydf[dailydf['DStage'] == 'reviving date']
    rev = rev.reset_index(drop=True)
    for n in range(rev.shape[0]):
        if rev['Date'][n].month <= 6:
            early = dailydf[dailydf['station ID'] == rev['station ID'][n]][dailydf['year'] == rev['year'][n]][
                dailydf['season'] == rev['season'][n]]
            earlydf = pd.concat([earlydf, early])
        else:
            late = dailydf[dailydf['station ID'] == rev['station ID'][n]][dailydf['year'] == rev['year'][n]][
                dailydf['season'] == rev['season'][n]]
            latedf = pd.concat([latedf, late])
    latedf = latedf.reset_index(drop=True)
    earlydf = earlydf.reset_index(drop=True)

    bins = [0, 132.871220, 526.287498, 856.900277, 1021.053015, 1501.593876, 25000]  # median_thermal():
    labes = ['reviving date', 'tillering date', 'jointing date', 'booting date', 'heading date', 'maturity date']
    phobins = [0, 132.871220, 523.123174, 773.711641, 908.889552, 1400.536426, 25000]
    pholabes = ['reviving date', 'tillering date', 'jointing date', 'booting date', 'heading date', 'maturity date']
    earlydf['Thermal_Dstage'] = pd.cut(earlydf['Thermal_cum'], bins=bins, labels=labes)
    earlydf['Phothermal_Dstage'] = pd.cut(earlydf['photothermal_cum'], bins=phobins, labels=pholabes)
    sim_days_thermal = earlydf.groupby([earlydf['station ID'], earlydf['year'],
                                        earlydf['season'], earlydf['Thermal_Dstage']])
    sim_days_phothermal = earlydf.groupby([earlydf['station ID'], earlydf['year'],
                                           earlydf['season'], earlydf['Phothermal_Dstage']])

    sim_thermal_date = sim_days_thermal.apply(lambda x: x.drop_duplicates(subset=['station ID', 'year',
                                                                                  'season', 'Thermal_Dstage']))
    sim_thermal_date.index = sim_thermal_date.index.droplevel(4)
    sim_thermal_date = sim_thermal_date.unstack()
    thermal_date = sim_thermal_date.drop(sim_thermal_date.columns[6:108], axis=1)
    thermal_date.columns = thermal_date.columns.droplevel(0)
    thermal_date.rename_axis('', axis="columns", inplace = True)

    sim_phothermal_date = sim_days_phothermal.apply(lambda x: x.drop_duplicates(subset=['station ID', 'year',
                                                                                        'season', 'Phothermal_Dstage']))
    sim_phothermal_date.index = sim_phothermal_date.index.droplevel(4)
    sim_phothermal_date = sim_phothermal_date.unstack()
    phothermal_date = sim_phothermal_date.drop(sim_phothermal_date.columns[6:108], axis=1)
    phothermal_date.columns = phothermal_date.columns.droplevel(0)
    phothermal_date.rename_axis('', axis="columns", inplace=True)

    bins2 = [0, 166.382740, 600.055712, 827.298495, 965.626885, 1345.368864, 25000]
    labes2 = ['reviving date', 'tillering date', 'jointing date', 'booting date', 'heading date', 'maturity date']
    phobins2 = [0, 166.382740, 597.838510, 803.419459, 927.428064, 1317.251906, 25000]
    pholabes2 = ['reviving date', 'tillering date', 'jointing date', 'booting date', 'heading date', 'maturity date']
    latedf['Thermal_Dstage'] = pd.cut(latedf['Thermal_cum'], bins=bins2, labels=labes2)
    latedf['Phothermal_Dstage'] = pd.cut(latedf['photothermal_cum'], bins=phobins2, labels=pholabes2)
    sim_days_thermal2 = latedf.groupby([latedf['station ID'], latedf['year'],
                                        latedf['season'], latedf['Thermal_Dstage']])
    sim_days_phothermal2 = latedf.groupby([latedf['station ID'], latedf['year'],
                                           latedf['season'], latedf['Phothermal_Dstage']])

    sim_thermal_date2 = sim_days_thermal2.apply(lambda x: x.drop_duplicates(subset=['station ID', 'year',
                                                                                  'season', 'Thermal_Dstage']))
    sim_thermal_date2.index = sim_thermal_date2.index.droplevel(4)
    sim_thermal_date2 = sim_thermal_date2.unstack()
    thermal_date2 = sim_thermal_date2.drop(sim_thermal_date2.columns[6:108], axis=1)
    thermal_date2.columns = thermal_date2.columns.droplevel(0)
    thermal_date2.rename_axis('', axis="columns", inplace=True)

    sim_phothermal_date2 = sim_days_phothermal2.apply(lambda x: x.drop_duplicates(subset=['station ID', 'year',
                                                                                        'season', 'Phothermal_Dstage']))
    sim_phothermal_date2.index = sim_phothermal_date2.index.droplevel(4)
    sim_phothermal_date2 = sim_phothermal_date2.unstack()
    phothermal_date2 = sim_phothermal_date2.drop(sim_phothermal_date2.columns[6:108], axis=1)
    phothermal_date2.columns = phothermal_date2.columns.droplevel(0)
    phothermal_date2.rename_axis('', axis="columns", inplace=True)

    allthermal_date = pd.concat([thermal_date, thermal_date2], axis=0)
    allphothermal_date = pd.concat([phothermal_date, phothermal_date2], axis=0)
    allthermal_date.to_excel(write, sheet_name='sim_thermal_date', index=True)
    allphothermal_date.to_excel(write, sheet_name='sim_phothermal_date', index=True)
    write.save()

def sim_date_weather(station, start, end):
    df = pd.read_table("../data/Meteo(48 sta)/" + str(station) + ".txt", encoding='gbk', sep=' * ', engine='python',
                       skiprows=[1])
    df['Date'] = df.apply(lambda row: pd.to_datetime('%d-%d-%d' % (row.YY, row.mm, row.dd)), axis=1)
    df = df.loc[(df.Date >= start) & (df.Date <= (end)), ['Date','SunHour','TemAver','TemMax','TemMin',
                                                                                     'Rainfall','Humidity','WindSpeed' ]]
    return df
def Sim_date_meteo(Sim_date,star_pheno,end_pheno,ero_colname):
    '''
    station, ID, lat, lon, date, stage
    join with weather by station id and date
    calculate daily thermal, daily photoperiod,photothermal= thermal*photo
    cumumlate simulative thermal, simulative photothermal from regrowth to maturation

    '''
    # Sim_date = pd.read_excel('../data/Sim_data.xlsx', sheet_name='sim_thermal_date',
    #                    parse_dates=['reviving date', 'tillering date', 'jointing date',
    #                                 'booting date', 'heading date', 'maturity date'])
    Sim_date['station ID'] = Sim_date['station ID'].ffill().astype('int')

    dfall = pd.DataFrame()
    for ind, row in Sim_date.iterrows():
        if pd.isna(row[end_pheno]):
            continue
        else:
            dfw = sim_date_weather(row['station ID'], row[star_pheno], row[end_pheno])
            dfw['SunHour_cum'] = dfw.SunHour.cumsum()
            dfw['AveTem'] = dfw.TemAver.mean()
            dfw['TemMax'] = dfw.TemMax.max()
            dfw['TemMin'] = dfw.TemMin.min()
            dfw['Rainfall'] = dfw.Rainfall.cumsum()
            dfw['Humidity'] = dfw.Humidity.mean()
            dfw['WindSpeed'] = dfw.Rainfall.mean()
            dfw['WindSpeed'] = dfw.Rainfall.mean()
            dfw['station ID'] = row['station ID']
            dfw['year'] = row['year']
            dfw['season'] = row['season']
            dfall = pd.concat([dfall, dfw])
    print(dfall.head())
    dfall = dfall.reset_index(drop=True)
    allmeteo = pd.DataFrame()
    for ind, row in Sim_date.iterrows():
        if pd.isna(row[end_pheno]):
            continue
        else:
            meteo = dfall[dfall['Date'] == row[end_pheno]][dfall['station ID'] == row['station ID']][
                dfall['year'] == row['year']][dfall['season'] == row['season']]
        allmeteo = pd.concat([allmeteo, meteo])
    print(allmeteo.head())
    allmeteo.to_excel('../data/'+star_pheno+'-'+end_pheno+'.xlsx', index=False)

    erro = pd.read_excel('../data/error_days.xlsx', sheet_name='all_result')
    erro['station ID'] = erro['station ID'].ffill().astype('int')
    ero = erro.loc[:, ['station ID', 'year', 'season', ero_colname]]
    dfme = allmeteo.merge(ero, how='left', on=['station ID', 'year', 'season'])
    station = pd.read_csv('../data/station catalog.csv', encoding='gbk')
    sta = station.loc[:, ['station ID', 'lat', 'lon', 'alt']]
    rfdata = dfme.merge(sta, how='left',on=['station ID'])
    rfdata.to_excel('../data/'+star_pheno+'-'+ero_colname+'.xlsx', index=False)



from bayes_opt import BayesianOptimization
from collections import defaultdict

# 计算两个日期之间的天数
def date_period(time1,time2):
    a1 = time1
    a2 = time2
    b1 = a1.split('-')
    b2 = a2.split('-')
    c1 = datetime.date(int(b1[0]), int(b1[1]), int(b1[2]))   #开始的日期de
    c2 = datetime.date(int(b2[0]), int(b2[1]), int(b2[2]))  #  结束的日期
    result = (c2-c1).days
    return result


# 定义贝叶斯优化目标函数，计算成熟日误差RMSE
def maturity_model(mu, zeta, ep, cumthermal):
    RMSE = 0
    # 计算每天的积温
    sun = Sun.Sun()
    df = pd.read_csv('D:/workspace/rice_crop_model/data/obser_pheno_catalog.csv', encoding="GBK",
                     parse_dates=['reviving date', 'tillering date', 'jointing date',
                                  'booting date', 'heading date', 'maturity date'])
    f1 = defaultdict(list)
    for j in range(df.shape[0]):
        dfw = read_station_weather(df['station ID'][j], df['reviving date'][j], df['maturity date'][j])
        dfw.index = pd.DatetimeIndex(dfw['Date'])
        n = (pd.Timestamp(df["reviving date"][j]) + pd.Timedelta("1day")).date().strftime('%Y-%m-%d')
        m = (pd.Timestamp(df['maturity date'][j]) + pd.Timedelta("30day")).date().strftime('%Y-%m-%d')
        period = date_period(n, m)
        period_d = 0
        day = pd.Timestamp(df['reviving date'][j])

        for k in range(period + 1):
            day = day + pd.Timedelta(k, unit='d')
            dfw1 = dfw[dfw.index == day]
            T = dfw1['TemAver'].values
            DTT = Wang_engle(T)
            dayL = sun.dayCivilTwilightLength(year=day.year, month=day.month, day=day.day, lon=df['lon'][j],
                                              lat=df['lat'][j])
            photo_raw = photo_period_based_on_yin(dayL, mu, zeta, ep)
            phothermal = DTT * photo_raw
            cumthermal = cumthermal - phothermal
            period_d += 1

        if cumthermal.size == 0 or cumthermal > 0:
            f1["erro"].append(np.nan)
        else:
            SimErr1 = period_d - period
            f1["erro"].append(SimErr1)
            break

    F1 = pd.DataFrame(f1)
    F1 = F1.dropna(axis=0)
    if F1.shape[0] != 0:
        s = 0
        for y in F1["erro"]:
            s += y ** 2
        RMSE = (s / F1.shape[0]) ** 0.5
        return RMSE
    else:
        return None



# 贝叶斯优化参数
def bayesoptimize(func,init_points, n_iter):
    pbounds = {'mu': (-60, -10), 'zeta': (0, 33), 'ep': (0, 33), 'cumthermal': (1000, 2000),}
    optimizer = BayesianOptimization(func, pbounds=pbounds, random_state=1)
    optimizer.maximize(init_points, n_iter)
    print(optimizer.max)

if __name__ == '__main__':
    # Calculation_error_days()
    # errordf = pd.read_excel('../data/error_days.xlsx', sheet_name='early_result', index_col=[0,1,2])
    # errordf2 = pd.read_excel('../data/error_days.xlsx', sheet_name='late_result', index_col=[0,1,2])
    # error_boxplot(errordf, errordf2)
    # Sim_date = pd.read_excel('../data/Sim_data.xlsx', sheet_name='sim_phothermal_date',
    #               parse_dates=['reviving date', 'tillering date', 'jointing date',
    #                            'booting date', 'heading date','maturity date'])
    # # Sim_date_meteo(Sim_date, 'reviving date', 'tillering date')
    # # Sim_date_meteo(Sim_date, 'tillering date', 'jointing date')
    # # Sim_date_meteo(Sim_date, 'jointing date', 'booting date')
    # # Sim_date_meteo(Sim_date, 'booting date', 'heading date')
    # Sim_date_meteo(Sim_date, 'heading date','maturity date')
    Test_photoeffect2()
    # bayesoptimize(maturity_model, 2, 5)

