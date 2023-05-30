import pandas as pd
import glob
from pylab import *
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def read_station_weather(station, start, end):
    df = pd.read_table("D:/workspace/rice_crop_model/data/Meteo(48 sta)/" + str(station) + ".txt",
                       encoding='gbk', sep=' * ', engine='python',skiprows=[1])
    df['Date'] = df.apply(lambda row: pd.to_datetime('%d-%d-%d' % (row.YY, row.mm, row.dd)), axis=1)
    df = df.loc[(df.Date >= start) & (df.Date <= end), ['Date','YY','SunHour','TemAver', 'Rainfall' ]]
    return df

def AnnualTem_site():
    station_df = pd.read_csv('D:/workspace/rice_crop_model/data/station_catalog_obserphen.csv',encoding='gbk')
    dfall = pd.DataFrame()
    for ind, row in station_df.iterrows():
        dfw = read_station_weather(row['station ID'],pd.to_datetime('1986-01-01'), pd.to_datetime('2005-12-30'))
        yn = len(dfw.YY.unique())
#         mask1 = dfw['Rainfall'] == 32766.0
#         mask2 = dfw['SunHour'] > 24
#         mask = mask1 | mask2
#         dfw = dfw.loc[~mask, :]
        dfw = dfw.dropna()
        dfw['Annual_Tem'] = (dfw.TemAver.cumsum())/yn
#         dfw['Rainfall'] = dfw.Rainfall.mean()
#         dfw['SunHour'] = dfw.SunHour.mean()
        dfw['station ID'] = row['station ID']
        dfw = dfw.drop_duplicates(subset=['station ID'],keep='last')
        dfall = pd.concat([dfall, dfw])
    df = station_df.merge(dfall, on=['station ID'], how='left')
    df.to_excel('../data/sites_mete_cluster.xlsx', index=False)
    return df

def Elbowplt():
    data = pd.read_excel('../data/sites_mete_cluster.xlsx',encoding='gbk')
    data.head()
    X = data[['lat','Annual_Tem']]
    # 标准化数据
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    # 使用Elbow方法确定最佳的K值
    inertia = []
    for k in range(1, 10):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X_scaled)
        inertia.append(kmeans.inertia_)
    # 绘制误差平均值与K值的变化关系图
    plt.plot(range(1, 10), inertia, marker='o')
    plt.xlabel('Number of clusters')
    plt.ylabel('SSE')
    plt.show()
    # 使用最佳的K值进行聚类分析
    kmeans = KMeans(n_clusters=4)
    kmeans.fit(X_scaled)
    y_pred = kmeans.predict(X_scaled)
    # 将聚类结果添加到原始数据集
    data['cluster'] = y_pred
    data.to_csv('../data/sites_clusters.csv', index=False)
    return data

if __name__ == "__main__":
    AnnualTem_site()
    Elbowplt()
    data = pd.read_excel('../data/sites_mete_cluster.xlsx', encoding='gbk')
    plt.scatter(data['lat'], data['Annual_Tem'], c=data['cluster'])
    plt.xlabel('Latitude')
    plt.ylabel('Average Temperature')
    plt.show()