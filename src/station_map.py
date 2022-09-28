#!/usr/bin/env python
# encoding: utf-8
"""
# @Time    : 2022/7/4 12:04
# @Author  : weather
# @Software: PyCharm
"""

import geopandas as gpd
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams['axes.linewidth'] = 1.5


def draw_china_map():
    china = gpd.read_file('../data/china_map/中国省级地图GS（2019）1719号.geojson')
    nine = gpd.read_file('../data/china_map/九段线GS（2019）1719号.geojson')

    fig = plt.figure(figsize=[9, 9])
    ax = plt.axes(projection=ccrs.LambertConformal(central_latitude=90, central_longitude=105))
    ax.gridlines()
    ax.set_extent([80, 135, 15, 55])
    china.plot(ax=ax, transform=ccrs.PlateCarree(), facecolor='w', edgecolor='k', lw=1.5, zorder=5)
    nine.plot(ax=ax, transform=ccrs.PlateCarree(), facecolor='w', edgecolor='k', lw=2)

    gl = ax.gridlines(draw_labels=True, x_inline=False, y_inline=False, color='k', linestyle='dashed', linewidth=0.5)
    gl.top_labels = False
    gl.bottom_labels = True
    gl.left_labels = True
    gl.right_labels = False

    sub_ax = fig.add_axes([0.72, 0.2, 0.2, 0.1],
                          projection=ccrs.LambertConformal(central_latitude=90, central_longitude=115))
    china.plot(ax=sub_ax, transform=ccrs.PlateCarree(), facecolor='w', edgecolor='k', lw=1, zorder=5)
    nine.plot(ax=sub_ax, transform=ccrs.PlateCarree(), facecolor='w', edgecolor='k', lw=2)
    sub_ax.set_extent([100, 125, 0, 25])
    return ax


def plot(data_path):
    df = pd.read_csv(data_path, encoding='gbk')
    points = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df['经度'], df['纬度'], crs='epsg:4326'))
    ax = draw_china_map()
    p1 = points.plot(ax=ax, transform=ccrs.PlateCarree(), marker='o', color='red', markersize=5,
                     label='station', zorder=6)
    p1.legend(fontsize=14, frameon=False, bbox_to_anchor=[0.9, 0.5], markerscale=1.5, handletextpad=0.1)
    plt.savefig('../fig/station.png', dpi=300)
    plt.savefig('../fig/station.tiff', dpi=300)
    plt.show()


if __name__ == "__main__":
    plot('../data/station catalog.csv')
