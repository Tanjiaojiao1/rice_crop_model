# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import math
from pylab import *
import Sun
import os
import seaborn as sns
from rice_phen import read_station_weather, photo_effect_correct
from photo_period_effect import photoeffect_yin, photoeffect_oryza2000, photoeffect_wofost
from T_dev_effect import Wang_engle, T_base_op_ceiling, T_base_opt
import datetime
from rice_phen import read_station_weather
from sklearn.cluster import KMeans
from all_models import simulate_and_calibrate
from multiprocessing import Pool
os.chdir(os.path.dirname(os.path.realpath(__file__)))

pd.options.display.max_columns = 999
pd.options.display.max_rows = 999
def getweatherstat_TemAver_ATM(SID,trd,bd,hd,md):
    '''
    SID:station id
    trd:transplanting date
    bd:booting date
    hd:heading date
    md:maturation date
    '''
    print(SID,trd)
    csy=trd.year
    df = pd.read_table("../data/Meteo/" + str(SID) + ".txt", encoding='gbk', sep=' * ', engine='python',
                       skiprows=[1])
    df['Date'] = df.apply(lambda row: pd.to_datetime('%d-%d-%d' % (row.YY, row.mm, row.dd)), axis=1)
    df = df.loc[(df.Date >= datetime.datetime(csy,1,1)) & (df.Date <= datetime.datetime(csy,12,31)), ['Date', 'TemAver']]
    df['Thermal']=df.TemAver.apply(lambda x:np.interp(x,[8,30],[0,22]))
    ATM=df.TemAver.mean()
    ATS=df.Thermal.sum()
    dfs= df.loc[(df.Date >= trd) & (df.Date <= trd+datetime.timedelta(days=120)), ['Date', 'TemAver','Thermal']]
    STM=dfs.TemAver.mean()#season mean tempeature
    STS=dfs.Thermal.sum()#season thermal sum
    return ATM,ATS,STM,STS,df.loc[(df.Date >= trd) & (df.Date <= bd), 'Thermal'].sum(),df.loc[(df.Date >= trd) & (df.Date <= hd), 'Thermal'].sum(),df.loc[(df.Date >= trd) & (df.Date <= md), 'Thermal'].sum()
def get_weather(SID,trd):
    '''
    SID:station id
    trd:transplanting date
    bd:booting date
    hd:heading date
    md:maturation date
    '''
    print(SID,trd)
    csy=trd.year
    df = pd.read_table("../data/Meteo/" + str(SID) + ".txt", encoding='gbk', sep=' * ', engine='python',
                       skiprows=[1])
    df['Date'] = df.apply(lambda row: pd.to_datetime('%d-%d-%d' % (row.YY, row.mm, row.dd)), axis=1)

    dfs= df.loc[(df.Date >= trd) & (df.Date <= trd+datetime.timedelta(days=160)), ['Date', 'TemAver']]

    return dfs
def put_weather_together():
    df = pd.read_excel('../data/obser_pheno_catalog.xlsx', 
                     parse_dates=['transplanting date','reviving date', 'tillering date', 'jointing date',
                                  'booting date', 'heading date','maturity date'])
    df['season']= df.groupby(['station ID', 'year']).cumcount()+1
    dfm = df[['station ID', 'lat', 'lon', 'alt', 'year', 'season','transplanting date',
               'reviving date', 'tillering date', 'jointing date',
               'booting date', 'heading date','maturity date']]
    for ind,row in dfm.iterrows():
        wth=get_weather(row['station ID'],row['transplanting date'])
        wth['SID']=row['station ID']
        wth['year']=row.year
        wth['season']=row.season
        wth.to_csv('../data/weather_all.csv',index=False,header=False if os.path.exists('../data/weather_all.csv') else True,mode='a')
            
def create_cluster_variables():
    df = pd.read_excel('../data/obser_pheno_catalog.xlsx', 
                     parse_dates=['transplanting date','reviving date', 'tillering date', 'jointing date',
                                  'booting date', 'heading date','maturity date'])
    df['season']= df.groupby(['station ID', 'year']).cumcount()+1
    dfm = df[['station ID', 'lat', 'lon', 'alt', 'year', 'season','transplanting date',
               'reviving date', 'tillering date', 'jointing date',
               'booting date', 'heading date','maturity date']]
    # dfm = pd.melt(dfmm, id_vars=['station ID', 'lat', 'lon', 'alt', 'year', 'season'],
    #               value_vars=['transplanting date','reviving date', 'tillering date', 'jointing date',
    #                           'booting date', 'heading date', 'maturity date'])
    
    dfm = dfm.rename(columns={'station ID':'SID'})
    dfm['TDOY']=dfm['transplanting date'].dt.dayofyear
    # dfm['DsAT_reviving']=(dfm['reviving date']-df['transplanting date']).apply(lambda x:x.days)
    # dfm['DsAT_tillering']=(dfm['tillering date']-df['transplanting date']).apply(lambda x:x.days)
    # dfm['DsAT_jointing']=(dfm['jointing date']-df['transplanting date']).apply(lambda x:x.days)
    # dfm['DsAT_booting']=(dfm['booting date']-df['transplanting date']).apply(lambda x:x.days)
    # dfm['DsAT_heading']=(dfm['heading date']-df['transplanting date']).apply(lambda x:x.days)
    # dfm['DsAT_maturity']=(dfm['maturity date']-df['transplanting date']).apply(lambda x:x.days)
    # dfm=dfm.head(1)
    dfm['WS']=dfm.apply(lambda row:getweatherstat_TemAver_ATM(row.SID,row['transplanting date'],row['booting date'],row['heading date'],row['maturity date']),axis=1)
    dfm['ATM']=dfm.WS.apply(lambda x:x[0])
    dfm['ATS']=dfm.WS.apply(lambda x:x[1])
    dfm['STM']=dfm.WS.apply(lambda x:x[2])
    dfm['STS']=dfm.WS.apply(lambda x:x[3])
    dfm['Booting_TS']=dfm.WS.apply(lambda x:x[4])
    dfm['Heading_TS']=dfm.WS.apply(lambda x:x[5])
    dfm['Maturation_TS']=dfm.WS.apply(lambda x:x[6])
    dfm.to_excel('../data/dfm.xlsx',index=False)
def plot_corr_matrix():
    df=pd.read_excel('../data/dfm.xlsx')[[ 'lat', 'lon', 'alt', 'TDOY',  'ATM', 'ATS', 'STM',
       'STS', 'Booting_TS', 'Heading_TS', 'Maturation_TS']].corr()
    print(df)
    sns.heatmap(df)
    plt.savefig('../fig/variable_corr_matrix.png',dpi=300)
def cluster_and_sim():
    df=pd.read_excel('../data/dfm.xlsx')
    wths=pd.read_csv('../data/weather_all.csv')
    if os.path.exists('../data/cluster_and_sim.csv'):
        os.remove('../data/cluster_and_sim.csv')
    for va in [['lat'],['STM'],['lat','STM']]:
        print(va)
        for n_cluster in [1,6,12,18,24]:
            kmeans=KMeans(n_clusters=n_cluster,n_init='auto')
            y = kmeans.fit_predict(df[va])
            df['Cluster_%d_%s'%(n_cluster,'_'.join(va))]=y
            for ind,gp in df.groupby('Cluster_%d_%s'%(n_cluster,'_'.join(va))):
                print(ind)
                dfws=wths.merge(gp,on=['SID','year','season'])[['SID','year','season','Date','TemAver']]
                for thermalfun,thermalfun_para in zip([Wang_engle, T_base_op_ceiling, T_base_opt],[{"Tbase":8, "Topt":30, "Tcei":42},{"Tbase":8,
                                                                     "Topt_low":25, "Topt_high":35, "Tcei":42,},{"Tbase":8, "Topt":30}]):
                    for photofun,photofun_para in zip([photoeffect_yin, photoeffect_oryza2000, photoeffect_wofost,""],
                                                      [{"mu":-15.46, "zeta":2.06, "ep":2.48},{"Dc":12.5,'PPSE':0.2},{"Dc":16, "Do":12.5},""]):
                        
                        dfcm=simulate_and_calibrate(thermal_fun=thermalfun,thermal_fun_para=thermalfun_para,photofun=photofun,photo_fun_para=photofun_para,dfws=dfws,df=gp)
                        print(thermalfun,photofun)
                        dfcm['thermalfun']=thermalfun.__name__
                        if photofun=='':
                            print('here')
                            dfcm['photofun']=''
                            dfcm['model']=thermalfun.__name__
                        else:
                            dfcm['photofun']=photofun.__name__
                            dfcm['model']=thermalfun.__name__+'_'+photofun.__name__
                        dfcm['n_cluster']=n_cluster
                        dfcm['cluster_vas']='_'.join(va)
                        dfcm['claster_number']=ind
                
                        dfcm.to_csv('../data/cluster_and_sim.csv',mode='a',header=False if os.path.exists('../data/cluster_and_sim.csv') else True,index=False)

    df.to_excel('../data/dfm_cluster.xlsx',index=False)
def cluster_and_sim_parallel():
    df=pd.read_excel('../data/dfm.xlsx')
    wths=pd.read_csv('../data/weather_all.csv')
    if os.path.exists('../data/cluster_and_sim.csv'):
        os.remove('../data/cluster_and_sim.csv')
    pool=Pool(6)
    res=[]
    for va in [['lat'],['STM'],['lat','STM']]:
        print(va)
        for n_cluster in [1,6,12,18,24]:
            kmeans=KMeans(n_clusters=n_cluster,n_init='auto')
            y = kmeans.fit_predict(df[va])
            df['Cluster_%d_%s'%(n_cluster,'_'.join(va))]=y
            re=pool.apply_async(sim_cluster,(df,wths,n_cluster,va))
            res.append(re)
    for re in res:
        dfcm=re.get()
        dfcm.to_csv('../data/cluster_and_sim.csv',mode='a',header=False if os.path.exists('../data/cluster_and_sim.csv') else True,index=False)

    df.to_excel('../data/dfm_cluster.xlsx',index=False)
def sim_cluster(df,wths,n_cluster,va):
    dfall=pd.DataFrame()
    for ind,gp in df.groupby('Cluster_%d_%s'%(n_cluster,'_'.join(va))):
        print(ind)
        dfws=wths.merge(gp,on=['SID','year','season'])[['SID','year','season','Date','TemAver']]
        for thermalfun,thermalfun_para in zip([Wang_engle, T_base_op_ceiling, T_base_opt],[{"Tbase":8, "Topt":30, "Tcei":42},{"Tbase":8,
                                                                "Topt_low":25, "Topt_high":35, "Tcei":42,},{"Tbase":8, "Topt":30}]):
            for photofun,photofun_para in zip([photoeffect_yin, photoeffect_oryza2000, photoeffect_wofost,""],
                                                [{"mu":-15.46, "zeta":2.06, "ep":2.48},{"Dc":12.5,'PPSE':0.2},{"Dc":16, "Do":12.5},""]):
                
                dfcm=simulate_and_calibrate(thermal_fun=thermalfun,thermal_fun_para=thermalfun_para,photofun=photofun,photo_fun_para=photofun_para,dfws=dfws,df=gp)
                print(thermalfun,photofun)
                dfcm['thermalfun']=thermalfun.__name__
                if photofun=='':
                    print('here')
                    dfcm['photofun']=''
                    dfcm['model']=thermalfun.__name__
                else:
                    dfcm['photofun']=photofun.__name__
                    dfcm['model']=thermalfun.__name__+'_'+photofun.__name__
                dfcm['n_cluster']=n_cluster
                dfcm['cluster_vas']='_'.join(va)
                dfcm['claster_number']=ind
                dfall=pd.concat([dfall,dfcm])

    return dfall
def boxplot_error():
    dfa=pd.read_csv('../data/cluster_and_sim.csv')
    fig=plt.figure(figsize=(12,12))
    
    for model in dfa.model.unique():
        print(model)
        df=dfa[dfa.model==model]
        n=1
        for cv in df.cluster_vas.unique():
            print(cv)
            print(len(df.cluster_vas.unique()),n)
            ax=fig.add_subplot(len(df.cluster_vas.unique()),1,n);n+=1
            dfc=df[df.cluster_vas==cv]
            sns.boxplot(data=dfc,y='DStage',x='delta_days',hue='n_cluster',ax=ax)
            ax.set_title(cv)
            ax.set_xlabel('')
            ax.set_xlim([-20,20])
        plt.savefig('../fig/boxplot_error_%s.png'%model,dpi=300)
def boxplot_cluster_effect():
    df=pd.read_excel('../data/dfm_cluster.xlsx')
    df.Cluster_6_STM=df.Cluster_6_STM.astype(str)
    # df=df[df.cluster_vas=='lat_STM']
    sns.boxplot(data=df,y='Cluster_6_STM',x='STS');show()
    print(df.columns)
if __name__=="__main__":
    cluster_and_sim_parallel()
    boxplot_error()
