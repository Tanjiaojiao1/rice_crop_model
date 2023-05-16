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

os.chdir(os.path.dirname(os.path.realpath(__file__)))

pd.options.display.max_columns = 999


def Simulate_error_days_for_sites(T_fun,Photo_fun,T_fun_para={},Photo_fun_para={},df):
    '''
    STEP 1:
    station, ID, lat, lon, date, stage
    join with weather by station id and date
    calculate daily thermal, daily photoperiod,photothermal= thermal*photo
    cumumlate thermal, photothermal from regrowth to maturation
    '''

    sun = Sun.Sun()
    df['season'] = df.groupby(['station ID', 'year']).cumcount() + 1
    dfmm = df[['station ID', 'lat', 'lon', 'alt', 'year', 'season',
               'reviving date', 'tillering date', 'jointing date',
               'booting date', 'heading date', 'maturity date']]
    dfm = pd.melt(dfmm, id_vars=['station ID', 'lat', 'lon', 'alt', 'year', 'season'],
                  value_vars=['reviving date', 'tillering date', 'jointing date',
                              'booting date', 'heading date', 'maturity date'])
    dfm = dfm.rename(columns={'value': 'Date', 'variable': 'DStage'})
    # print(dfm)
    dfall = pd.DataFrame()
    for ind, row in df.iterrows():
        dfw = read_station_weather(row['station ID'], row['reviving date'], row['maturity date'])
        dfw['Thermal'] = dfw.TemAver.apply(lambda x: T_fun(T=x, **T_fun_para))
        dfw['Thermal_cum'] = dfw.Thermal.cumsum()
        dfw['dayL'] = dfw.Date.apply(
            lambda x: sun.dayCivilTwilightLength(year=x.year, month=x.month, day=x.day, lon=row.lon, lat=row.lat))
        dfw['photo_raw'] = dfw.dayL.apply(lambda x: Photo_fun(DL=x,**Photo_fun_para))
        dfw['photo'] = dfw.apply(
            lambda rowt: photo_effect_correct(today=rowt.Date, revd=row['reviving date'], jd=row['jointing date'],
                                              hd=row['heading date'], photo=rowt.photo_raw), axis=1)
        dfw['photothermal'] = dfw.photo * dfw.Thermal
        dfw['photothermal_cum'] = dfw['photothermal'].cumsum()
        dfw['station ID'] = row['station ID']
        dfw['year'] = row['year']
        dfw['season'] = row['season']
        dfall = pd.concat([dfall, dfw])
    # df = dfm.merge(dfall, on=['station ID', 'year', 'Date', 'season'], how='left')
    # df.boxplot(column=['Thermal_cum', 'photothermal_cum'], by=['DStage'])
    # show()
    thermaldf = dfall.merge(dfm, on=['station ID', 'year', 'Date', 'season'], how='left')
    # df.to_excel('../data/run_phen_model_result/dfall' + str(number) + '.xlsx', index=False)
    # thermaldf.to_excel('../data/run_phen_model_result/thermaldf.xlsx', index=False)

    '''
    STEP 2:
    get the median cum_thermal and the median cum_photothermal
    calculate simulation error (days)
    the average error as the evaluation index
    '''
    # write = pd.ExcelWriter('../data/run_phen_model_result/error_days.xlsx')
    # Calculate the median accumulated temperature
    # Data were extracted separately for early rice and late rice
    earlydf = pd.DataFrame()  # Single season rice and early rice
    latedf = pd.DataFrame()  # late rice
    rev = thermaldf[thermaldf['DStage'] == 'reviving date']
    rev = rev.reset_index(drop=True)
    for n in range(rev.shape[0]):
        if rev['Date'][n].month <= 6:
            early = thermaldf[thermaldf['station ID'] == rev['station ID'][n]][thermaldf['year'] ==rev['year'][n]][thermaldf['season'] == rev['season'][n]]
            earlydf = pd.concat([earlydf, early])
        else:
            late = thermaldf[thermaldf['station ID'] == rev['station ID'][n]][thermaldf['year'] ==rev['year'][n]][thermaldf['season'] == rev['season'][n]]
            latedf = pd.concat([latedf, late])
    latedf = latedf.reset_index(drop=True)
    earlydf = earlydf.reset_index(drop=True)
    thermal_median_early = earlydf['Thermal_cum'].groupby(earlydf['DStage']).median()
    thermal_median_early = thermal_median_early.sort_values()
    thermal_median_late = latedf['Thermal_cum'].groupby(latedf['DStage']).median()
    thermal_median_late = thermal_median_late.sort_values()
    photothermal_median_early = earlydf['photothermal_cum'].groupby(earlydf['DStage']).median()
    photothermal_median_early = photothermal_median_early.sort_values()
    photothermal_median_late = latedf['photothermal_cum'].groupby(latedf['DStage']).median()
    photothermal_median_late = photothermal_median_late.sort_values()

    if len(photothermal_median_early) == len(np.unique(photothermal_median_early)) and len(photothermal_median_late) == len(np.unique(photothermal_median_late)):
    # Calculate simulated days and observed days
        bins = [0, thermal_median_early[1], thermal_median_early[2], thermal_median_early[3], thermal_median_early[4],
                thermal_median_early[5], numpy.inf]
        labes = thermal_median_early.index.tolist()
        phobins = [0, photothermal_median_early[1], photothermal_median_early[2], photothermal_median_early[3],
                   photothermal_median_early[4], photothermal_median_early[5], numpy.inf]
        pholabes = photothermal_median_early.index.tolist()
        earlydf['Thermal_Dstage'] = pd.cut(earlydf['Thermal_cum'], bins=bins, labels=labes)
        earlydf['Phothermal_Dstage'] = pd.cut(earlydf['photothermal_cum'], bins=phobins, labels=pholabes)
        earlydf['obser_Dstage'] = earlydf['DStage'].ffill()
        obser_days = earlydf['obser_Dstage'].groupby([earlydf['station ID'], earlydf['year'], earlydf['season'],
                                                      earlydf['obser_Dstage']]).count().unstack().iloc[:,[4, 5, 2, 0, 1, 3]]

        # obser_days.to_excel(write, sheet_name='early_obser_days', index=True)
        sim_days_thermal = earlydf['Thermal_Dstage'].groupby([earlydf['station ID'], earlydf['year'],
                                                              earlydf['season'], earlydf['Thermal_Dstage']]).count()
        sim_days_phothermal = earlydf['Phothermal_Dstage'].groupby([earlydf['station ID'], earlydf['year'],
                                                                    earlydf['season'],
                                                                    earlydf['Phothermal_Dstage']]).count()

        bins2 = [0, thermal_median_late[1], thermal_median_late[2], thermal_median_late[3], thermal_median_late[4],
                thermal_median_late[5], numpy.inf]
        labes2 = thermal_median_late.index.tolist()
        phobins2 = [0, photothermal_median_late[1], photothermal_median_late[2], photothermal_median_late[3],
                   photothermal_median_late[4], photothermal_median_late[5], numpy.inf]
        pholabes2 = photothermal_median_late.index.tolist()
        latedf['Thermal_Dstage'] = pd.cut(latedf['Thermal_cum'], bins=bins2, labels=labes2)
        latedf['Phothermal_Dstage'] = pd.cut(latedf['photothermal_cum'], bins=phobins2, labels=pholabes2)
        latedf['obser_Dstage'] = latedf['DStage'].ffill()
        obser_days2 = latedf['obser_Dstage'].groupby([latedf['station ID'], latedf['year'],
                                                      latedf['season'], latedf['obser_Dstage']]).count().unstack().iloc[:, [4, 5, 2, 0, 1, 3]]
        # obser_days2.to_excel(write, sheet_name='late_obser_days', index=True)
        sim_days_thermal2 = latedf['Thermal_Dstage'].groupby([latedf['station ID'], latedf['year'],
                                                              latedf['season'], latedf['Thermal_Dstage']]).count()
        sim_days_phothermal2 = latedf['Phothermal_Dstage'].groupby([latedf['station ID'], latedf['year'],
                                                                    latedf['season'], latedf['Phothermal_Dstage']]).count()

        # Calculation error days early rice
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
        result.drop(['maturity date_x', 'maturity date_y'], axis=1, inplace=True)
        result.columns = ['rev-til', 'til-joi', 'joi-boot', 'boot-hea', 'hea-mat',
                          'rev-til_photo', 'til-joi_photo', 'joi-boot_photo', 'boot-hea_photo', 'hea-mat_photo']
        # result.to_excel(write, sheet_name='early_result', index=True)

        # Calculation error days for late rice
        Revised_obser2 = obser_days2.reset_index()
        index_tillering2 = Revised_obser2['tillering date'].isnull()
        Revised_obser2['reviving date'][index_tillering2] = np.NaN
        index_jointing2 = Revised_obser2['jointing date'].isnull()
        Revised_obser2['tillering date'][index_jointing2] = np.NaN
        Revised_obser2  = Revised_obser2.set_index(['station ID', 'year', 'season'])
        late_thermal_results = sim_days_thermal2.unstack() - Revised_obser2
        late_phothermal_results = sim_days_phothermal2.unstack() - Revised_obser2
        result2 = late_thermal_results.merge(late_phothermal_results, left_index=True, right_index=True)
        # result2.to_excel(write, sheet_name='late_result', index=True)
        result2.drop(['maturity date_x', 'maturity date_y'], axis=1, inplace=True)
        result2.columns = ['rev-til', 'til-joi', 'joi-boot', 'boot-hea', 'hea-mat',
                           'rev-til_photo', 'til-joi_photo', 'joi-boot_photo', 'boot-hea_photo', 'hea-mat_photo']
        errors = pd.concat([result, result2])          #Merge all errors
        # errors.to_excel(write, sheet_name='errors', index=True)
        # write.save()
        print(errors.mean())
        return -abs(errors.mean()[5:9].sum())
    else:
        return -9999


def bayesoptimize(func, init_points, n_iter):
    pbounds = {'Tbase': (8, 15), 'Topt': (20, 38), 'Tcei': (38, 45),
               'mu': (-60, -10), 'zeta': (0.5, 33), 'ep': (0.5, 33)}
    optimizer = BayesianOptimization(func, pbounds=pbounds, random_state=1)
    optimizer.maximize(init_points, n_iter)
    print(optimizer.max)


if __name__ == "__main__":
    data = pd.read_csv('../data/obser_pheno_catalog.csv', encoding="GBK",
                       parse_dates=['reviving date', 'tillering date', 'jointing date',
                                    'booting date', 'heading date', 'maturity date'])
    df = data
    bayesoptimize(Simulate_error_days_for_sites, init_points=15, n_iter=30)


