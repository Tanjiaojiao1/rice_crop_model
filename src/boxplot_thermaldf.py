# -*- coding: utf-8 -*-
import pandas as pd
import Sun
import datetime
import seaborn as sns
import matplotlib.pyplot as plt
from photo_period_effect import photoeffect_yin, photoeffect_oryza2000, photoeffect_wofost
from T_dev_effect import Wang_engle, T_base_opt_ceiling, T_base_opt

pd.options.display.max_columns = 999


def Tem_correct_T_base_opt(today, hd, T, Tbase2, Topt2, Thermal):
    if today > hd:
        return T_base_opt(T, Tbase2, Topt2)
    else:
        return Thermal


def Tem_correct_T_base_opt_ceiling(today, hd, T, Tbase2, Topt_low2, Topt_high2, Tcei2, Thermal):
    if today > hd:
        return T_base_opt_ceiling(T, Tbase2, Topt_low2, Topt_high2, Tcei2)
    else:
        return Thermal


def Tem_correct_Wang_engle(today, hd, T, Tbase2, Topt2, Tcei2, value):
    if today > hd:
        return Wang_engle(T, Tbase2, Topt2, Tcei2)
    else:
        return value


def photo_effect_correct(today, revd, jd, hd, photo):
    if pd.isna(jd):
        jd = revd + datetime.timedelta(days=34)
    if pd.isna(hd):
        hd = revd + datetime.timedelta(days=63)
    if today < jd or today > hd:
        return 1
    else:
        return photo


def read_station_weather(station, start, end):
    df = pd.read_table("../data/Meteo(48 sta)/" + str(station) + ".txt", encoding='gbk', sep=' * ', engine='python',
                       skiprows=[1])
    df['Date'] = df.apply(lambda row: pd.to_datetime('%d-%d-%d' % (row.YY, row.mm, row.dd)), axis=1)
    df = df.loc[(df.Date >= start) & (df.Date <= (end + datetime.timedelta(days=0))), ['Date', 'TemAver']]
    return df


def Trial_Sim():
    '''
    station, ID, lat, lon, date, stage
    join with weather by station id and date
    calculate daily thermal, daily photoperiod,photothermal= thermal*photo
    cumumlate thermal, photothermal from regrowth to maturation

    '''
    sun = Sun.Sun()
    df = pd.read_excel('../data/obser_pheno_catalog.xlsx',
                       parse_dates=['reviving date', 'tillering date', 'jointing date',
                                    'booting date', 'heading date', 'maturity date'])
    df['season'] = df.groupby(['station ID', 'year']).cumcount() + 1
    dfmm = df[['station ID', 'lat', 'lon', 'alt', 'year', 'season',
               'reviving date', 'tillering date', 'jointing date',
               'booting date', 'heading date', 'maturity date']]
    dfm = pd.melt(dfmm, id_vars=['station ID', 'lat', 'lon', 'alt', 'year', 'season'],
                  value_vars=['reviving date', 'tillering date', 'jointing date',
                              'booting date', 'heading date', 'maturity date'])
    dfm = dfm.rename(columns={'value': 'Date', 'variable': 'DStage'})
    print(dfm)
    dfall = pd.DataFrame()
    for ind, row in df.iterrows():
        dfw = read_station_weather(row['station ID'], row['reviving date'], row['maturity date'])
        dfw['Thermal_Wang_raw'] = dfw.TemAver.apply(lambda x: Wang_engle(T=x, Tbase=8, Topt=30, Tcei=42))
        dfw['Thermal_TbTo_raw'] = dfw.TemAver.apply(lambda x: Wang_engle(T=x, Tbase=8, Topt=30, Tcei=42))
        dfw['Thermal_Tboc_raw'] = dfw.TemAver.apply(
            lambda x: T_base_op_ceiling(T=x, Tbase=8, Topt_low=25, Topt_high=35, Tcei=42))
        dfw['Thermal_Wang'] = dfw.apply(
            lambda rowt: Tem_correct_Wang_engle(today=rowt.Date, hd=row['heading date'], T=rowt.TemAver,
                                                Tbase2=4, Topt2=30, Tcei2=42, value=rowt.Thermal_Wang_raw), axis=1)
        dfw['Thermal_TbTo'] = dfw.apply(
            lambda rowt: Tem_correct_T_base_opt(today=rowt.Date, hd=row['heading date'], T=rowt.TemAver,
                                                Tbase2=4, Topt2=30, Thermal=rowt.Thermal_TbTo_raw), axis=1)
        dfw['Thermal_Tboc'] = dfw.apply(
            lambda rowt: Tem_correct_T_base_opt_ceiling(today=rowt.Date, hd=row['heading date'], T=rowt.TemAver,
                                                        Tbase2=4, Topt_low2=25, Topt_high2=35, Tcei2=42,
                                                        Thermal=rowt.Thermal_Tboc_raw), axis=1)
        dfw['Thermal_Wang_cum'] = dfw.Thermal_Wang.cumsum()
        dfw['Thermal_TbTo_cum'] = dfw.Thermal_TbTo.cumsum()
        dfw['Thermal_Tboc_cum'] = dfw.Thermal_Tboc.cumsum()
        dfw['dayL'] = dfw.Date.apply(
            lambda x: sun.dayCivilTwilightLength(year=x.year, month=x.month, day=x.day, lon=row.lon, lat=row.lat))
        dfw['photo_Yin_raw'] = dfw.dayL.apply(lambda x: photoeffect_yin(DL=x, mu=-15.46, zeta=2.06, ep=2.48))
        dfw['photo_wofost_raw'] = dfw.dayL.apply(lambda x: photoeffect_wofost(DL=x, Dc=16, Do=12.5))
        dfw['photo_oryza_raw'] = dfw.dayL.apply(lambda x: photoeffect_oryza2000(DL=x, Dc=12.5, ))
        dfw['photo_Yin'] = dfw.apply(
            lambda rowt: photo_effect_correct(today=rowt.Date, revd=row['reviving date'], jd=row['jointing date'],
                                              hd=row['heading date'], photo=rowt.photo_Yin_raw), axis=1)
        dfw['photo_wofost'] = dfw.apply(
            lambda rowt: photo_effect_correct(today=rowt.Date, revd=row['reviving date'], jd=row['jointing date'],
                                              hd=row['heading date'], photo=rowt.photo_wofost_raw), axis=1)
        dfw['photo_oryza'] = dfw.apply(
            lambda rowt: photo_effect_correct(today=rowt.Date, revd=row['reviving date'], jd=row['jointing date'],
                                              hd=row['heading date'], photo=rowt.photo_oryza_raw), axis=1)
        dfw['PhoThermal_Wang_Yin'] = dfw.photo_Yin * dfw.Thermal_Wang
        dfw['PhoThermal_Wang_Yin_cum'] = dfw['PhoThermal_Wang_Yin'].cumsum()
        dfw['PhoThermal_Wang_wofost'] = dfw.photo_wofost * dfw.Thermal_Wang
        dfw['PhoThermal_Wang_wofost_cum'] = dfw['PhoThermal_Wang_wofost'].cumsum()
        dfw['PhoThermal_Wang_oryza'] = dfw.photo_oryza * dfw.Thermal_Wang
        dfw['PhoThermal_Wang_oryza_cum'] = dfw['PhoThermal_Wang_oryza'].cumsum()
        dfw['PhoThermal_TbTo_Yin'] = dfw.photo_Yin * dfw.Thermal_TbTo
        dfw['PhoThermal_TbTo_Yin_cum'] = dfw['PhoThermal_TbTo_Yin'].cumsum()
        dfw['PhoThermal_TbTo_wofost'] = dfw.photo_wofost * dfw.Thermal_TbTo
        dfw['PhoThermal_TbTo_wofost_cum'] = dfw['PhoThermal_TbTo_wofost'].cumsum()
        dfw['PhoThermal_TbTo_oryza'] = dfw.photo_oryza * dfw.Thermal_TbTo
        dfw['PhoThermal_TbTo_oryza_cum'] = dfw['PhoThermal_TbTo_oryza'].cumsum()
        dfw['PhoThermal_Tboc_Yin'] = dfw.photo_Yin * dfw.Thermal_Tboc
        dfw['PhoThermal_Tboc_Yin_cum'] = dfw['PhoThermal_Tboc_Yin'].cumsum()
        dfw['PhoThermal_Tboc_wofost'] = dfw.photo_wofost * dfw.Thermal_Tboc
        dfw['PhoThermal_Tboc_wofost_cum'] = dfw['PhoThermal_Tboc_wofost'].cumsum()
        dfw['PhoThermal_Tboc_oryza'] = dfw.photo_oryza * dfw.Thermal_Tboc
        dfw['PhoThermal_Tboc_oryza_cum'] = dfw['PhoThermal_Tboc_oryza'].cumsum()
        dfw['station ID'] = row['station ID']
        dfw['year'] = row['year']
        dfw['season'] = row['season']
        dfall = pd.concat([dfall, dfw])
    print(dfall.head(3))
    print(dfall.columns, dfm.columns)
    df = dfm.merge(dfall, on=['station ID', 'year', 'Date', 'season'], how='left')
    df.boxplot(column=['Thermal_Wang_cum', 'Thermal_TbTo_cum', 'Thermal_Tboc_cum',
                       'PhoThermal_Wang_Yin_cum', 'PhoThermal_Wang_wofost_cum', 'PhoThermal_Wang_oryza_cum',
                       'PhoThermal_TbTo_Yin_cum', 'PhoThermal_TbTo_wofost_cum', 'PhoThermal_TbTo_oryza_cum',
                       'PhoThermal_Tboc_Yin_cum', 'PhoThermal_Tboc_wofost_cum', 'PhoThermal_Tboc_oryza_cum'
                       ], by=['DStage'])
    show()
    df.to_excel('D:/workspace/rice_crop_model/data/thermaldf.xlsx', index=False)
    # thermaldf = dfall.merge(dfm, on=['station ID', 'year', 'Date', 'season'], how='left')
    # thermaldf.to_excel('../data/thermaldf_phofun2.xlsx', index=False)


def boxplot_thermal():
    df = pd.read_excel('D:/workspace/rice_crop_model/data/thermaldf.xlsx')
    dff = df[['DStage', 'Thermal_Wang_cum', 'Thermal_TbTo_cum', 'Thermal_Tboc_cum',
              'PhoThermal_Wang_Yin_cum', 'PhoThermal_Wang_wofost_cum', 'PhoThermal_Wang_oryza_cum',
              'PhoThermal_TbTo_Yin_cum', 'PhoThermal_TbTo_wofost_cum', 'PhoThermal_TbTo_oryza_cum',
              'PhoThermal_Tboc_Yin_cum', 'PhoThermal_Tboc_wofost_cum', 'PhoThermal_Tboc_oryza_cum']]
    dff = dff.rename(columns={'Thermal_Wang_cum': 'Wang_engle', 'Thermal_TbTo_cum': 'T_base_opt',
                              'Thermal_Tboc_cum': 'T_base_opt_cei',
                              'PhoThermal_Wang_Yin_cum': 'Wang_engle_Yin',
                              'PhoThermal_Wang_wofost_cum': 'Wang_engle_WOFOST',
                              'PhoThermal_Wang_oryza_cum': 'Wang_engle_oryza',
                              'PhoThermal_TbTo_Yin_cum': 'T_base_opt_Yin',
                              'PhoThermal_TbTo_wofost_cum': 'T_base_opt_WOFOST',
                              'PhoThermal_TbTo_oryza_cum': 'T_base_opt_oryza',
                              'PhoThermal_Tboc_Yin_cum': 'T_base_op_cei_Yin',
                              'PhoThermal_Tboc_wofost_cum': 'T_base_op_cei_WOFOST',
                              'PhoThermal_Tboc_oryza_cum': 'T_base_op_cei_oryza'})
    dff.loc[dff['DStage'] == 'reviving date', 'DStage'] = 'Regreening '
    dff.loc[dff['DStage'] == 'tillering date', 'DStage'] = 'Tillering'
    dff.loc[dff['DStage'] == 'jointing date', 'DStage'] = 'Jointing'
    dff.loc[dff['DStage'] == 'booting date', 'DStage'] = 'Booting'
    dff.loc[dff['DStage'] == 'heading date', 'DStage'] = 'Heading'
    dff.loc[dff['DStage'] == 'maturity date', 'DStage'] = 'Maturation'
    # 使用melt()函数将数据框从宽格式转换为长格式
    df_melted = pd.melt(dff, id_vars=['DStage'], var_name='variable', value_name='value')
    df_melted.head()
    # 使用seaborn的boxplot函数绘制箱型图
    plt.figure(figsize=(17 / 2.54, 17 / 2.54))
    # palette = sns.color_palette('Set1', 12)
    g = sns.boxplot(data=df_melted, x='DStage', hue='variable', y='value', fliersize=3, width=0.85,
                    medianprops={'linestyle': '-', 'color': 'black'},
                    showmeans=True,
                    meanprops={'marker': 'o', 'markerfacecolor': 'red', 'markeredgecolor': 'white', 'markersize': 4.5})
    plt.ylabel('Thermal accumulation(℃ d)', fontsize=12)
    plt.xlabel('', fontsize=12)
    # plt.ylim(ymax=30,ymin=-20.5)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    g.legend_.set_title(None)
    plt.legend(ncol=4, bbox_to_anchor=(1.2, -0.07), frameon=False, fontsize=8)
    plt.savefig('D:/workspace/rice_crop_model/fig/thermaldf.png', bbox_inches='tight', dpi=600)

if __name__ == '__main__':
    Trial_Sim()
    boxplot_thermal()