#!/usr/bin/env python
# encoding: utf-8
"""
# @Time    : 2022/9/27
# @Author  : Tanjiaojiao
# @File    : Sites distribution map by lat and lon
# @Software: PyCharm
"""

import numpy as np
import pandas as pd
import cartopy
import cartopy.crs as ccrs
import cartopy.feature as cfeat
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from cartopy.io.shapereader import Reader, natural_earth
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.image import imread
# from adjusttext import adjust_text
# from adjustText import adjust_text
# from collections import OrderedDict

def create_map():
    shp_path = 'D:/cn_shp/Province_9/'
    # --创建画图空间
    proj = ccrs.PlateCarree()  # 创建坐标系
    fig = plt.figure(figsize=(8, 8), dpi=400)  # 创建页面
    ax = fig.subplots(1, 1, subplot_kw={'projection': proj})

    # --设置地图属性
    provinces = cfeat.ShapelyFeature(
        Reader(shp_path + 'Province_9.shp').geometries(),
        proj, edgecolor='k',
        facecolor='none'
    )
    # 加载省界线
    ax.add_feature(provinces, linewidth=0.6, zorder=10)
    # 加载分辨率为50的海岸线
    ax.add_feature(cfeat.COASTLINE.with_scale('50m'), linewidth=0.6, zorder=10)
    # 加载分辨率为50的河流~
    ax.add_feature(cfeat.RIVERS.with_scale('50m'), zorder=8)
     # 加载分辨率为50的湖泊
    ax.add_feature(cfeat.LAKES.with_scale('50m'), zorder=10)
    ax.set_extent([70,135,5,50])

    # ax.stock_img()
    ax.imshow(
        imread('D:/NE1_50M_SR_W.tif'),
        # origin='upper',
        transform=proj,
        extent=[-180, 180, -90, 90]
    )
    # --设置网格点属性
    gl = ax.gridlines(
        crs=ccrs.PlateCarree(),
        draw_labels=True,
        linewidth=1.2,
        color='k',
        alpha=0.5,
        linestyle='--'
    )
    gl.top_labels = False  # 关闭顶端的经纬度标签
    gl.right_labels = False  # 关闭右侧的经纬度标签
    gl.xformatter = LONGITUDE_FORMATTER  # x轴设为经度的格式
    gl.yformatter = LATITUDE_FORMATTER  # y轴设为纬度的格式
    gl.xlocator = mticker.FixedLocator(np.arange(70, 130+5, 10))
    gl.ylocator = mticker.FixedLocator(np.arange(0, 45+5 , 5))


    # --设置小地图
    left, bottom, width, height = 0.64, 0.2, 0.36, 0.2
    ax2 = fig.add_axes(
        [left, bottom, width, height],
        projection=proj
    )
    ax2.add_feature(provinces, linewidth=0.6, zorder=2)
    ax2.add_feature(cfeat.COASTLINE.with_scale('50m'), linewidth=0.6, zorder=10)
    ax2.add_feature(cfeat.RIVERS.with_scale('50m'), zorder=10)
    ax2.add_feature(cfeat.LAKES.with_scale('50m'), zorder=10)
    ax2.set_extent([105, 125, 0, 25])
    # ax2.stock_img()
    ax2.imshow(
        imread('D:/NE1_50M_SR_W.tif'),
        origin='upper',
        transform=proj,
        extent=[-180, 180, -90, 90]
    )
    return ax


def main():
    ax = create_map()
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    plt.rcParams['axes.unicode_minus'] = False
    # matplotlib画图中中文显示会有问题，需要这两行设置默认字体

    df = pd.read_csv('D:/Python code/station catalog.csv', encoding='gbk')
    bins = [0, 3, 6, 10, 14]  # median_thermal():
    lables = ['0-3', '3-6', '6-10', '10-14']
    df['mark_year'] = pd.cut(df['num_years'], bins=bins, labels=lables)
    grouped_data = df.groupby('mark_year')
    marker_dict = {'0-3': 'o', '3-6': 's', '6-10': '^', '10-14': '<'}
    size_dict = {'0-3': 15, '3-6': 15, '6-10': 15, '10-14': 15}

    for mark_year, group in grouped_data:
        ax.scatter(group['经度'], group['纬度'], s=size_dict[mark_year], marker=marker_dict[mark_year],
                   transform=ccrs.PlateCarree(), label=str(mark_year))
    ax.legend()
    plt.savefig('station_distribute_map2.png')

if __name__ == '__main__':
    main()
