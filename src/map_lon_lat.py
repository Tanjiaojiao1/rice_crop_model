#!/usr/bin/env python
# encoding: utf-8
"""
# @Time    : 2022/7/4 12:04
# @Author  : weather
# @File    : test.py
# @Software: PyCharm
"""

import pandas as pd
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import geopandas as gpd
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter


def main():
    """

    """
    # read points Lon, lat
    df = pd.read_csv('../data/station catalog.csv', encoding='gbk')
    # geopandas instance of the csv
    points = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(x=df['经度'], y=df['纬度'], crs='epsg:4326'))
    # print(points)
    # read China shapefile
    province = gpd.read_file('../data/China_province/bou2_4p.shp')
    ax = plt.axes()
    # print(province)
    province.plot(ax=ax, facecolor="w", edgecolor='k')
    points.plot(ax=ax, markersize=3, color='r')
    # set y label  x label
    ax.set_xticks([80, 100, 120], crs=ccrs.PlateCarree())
    ax.set_yticks([20, 30, 40, 50], crs=ccrs.PlateCarree())
    lon_formatter = LongitudeFormatter(zero_direction_label=True)
    lat_formatter = LatitudeFormatter()
    ax.xaxis.set_major_formatter(lon_formatter)
    ax.yaxis.set_major_formatter(lat_formatter)
    plt.show()


if __name__ == '__main__':
    main()
