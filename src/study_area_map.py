import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.shapereader as shapereader
import rasterio
import matplotlib.ticker as mticker
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import geopandas as gpd


def map():
    # 设置字体
    plt.rcParams["font.family"] =  "Times New Roman"
    # "Times New Roman"
    plt.rcParams["font.size"] = 12

    # 自定义颜色列表
    colors = ["#33A02C", "#B2DF8A", "#FDBF6F", "#1F78B4", "#999999", "#E31A1C", "#E6E6E6", "#A6CEE3"]
    # 创建 colormap
    cmap = ListedColormap(colors)

    proj = ccrs.LambertConformal(central_longitude=105, standard_parallels=(25, 47))
    fig = plt.figure(figsize=[14, 14])
    ax = plt.axes(projection=proj)
    ax.set_extent([80,130,18,53],crs = ccrs.PlateCarree())

    china = gpd.read_file('D:/workspace/rice_crop_model/src/china_map/中国省级地图GS（2019）1719号.geojson')
    nine = gpd.read_file('D:/workspace/rice_crop_model/src/china_map/九段线GS（2019）1719号.geojson')

    # 加载边界数据
    nine.plot(ax=ax,transform=ccrs.PlateCarree(),facecolor='none',edgecolor='k',lw=0.6,zorder=3,alpha=0.8)
    china.plot(ax=ax,transform=ccrs.PlateCarree(),facecolor='none',edgecolor='k',lw=0.6,zorder=3, alpha=0.8)

    #添加其他要素
    ax.add_feature(cfeature.OCEAN.with_scale('50m'))
    ax.add_feature(cfeature.LAND.with_scale('50m'))
    # ax.add_feature(cfeature.COASTLINE,lw = 0.3)#添加海岸线
    # ax.add_feature(cfeature.RIVERS,lw = 0.25)#添加河流
    # ax.add_feature(cfeature.LAKES)#指定湖泊颜色为红色#添加湖泊

    # # 读取 DEM 数据
    with rasterio.open('D:/workspace/rice_crop_model/src/china_map/dem_5km.tif') as src:
        data = src.read(1)
        transform = src.transform
        bounds = src.bounds

    # 过滤数据
    data[data<-1000] = np.nan
    tif_extent = [bounds.left,bounds.right,bounds.bottom,bounds.top]
    p = ax.imshow(data, origin='upper', transform=ccrs.PlateCarree(),extent=tif_extent ,cmap=cmap,zorder=3,alpha=0.55)

    # Add a colorbar to the figure
    cbar = plt.colorbar(p,ax=ax, orientation='vertical',  fraction=0.01, pad=0.05)

    # Set the title for the colorbar
    cbar.set_label('Elevation (m)')

    # Set the location for the colorbar
    cbar.ax.set_position([0.15, 0.3,  1.5, 0.1])

    # 添加比例尺
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes

    scalebar = inset_axes(ax, width="10%", height="1%", loc='lower center'
                          ,bbox_to_anchor=(-0.19, 0, 1, 1), bbox_transform=ax.transAxes,  borderpad=1.5)
    scalebar.axis["top"].set_visible(False)
    scalebar.axis["right"].set_visible(False)
    scalebar.axis["left"].set_visible(False)
    scalebar.axis["bottom"].set_visible(True)
    scalebar.set_xticks([0,10])
    scalebar.set_xticklabels(['0','10Km'])
    scalebar.tick_params(axis='both', labelsize=12)
    scalebar.set_facecolor('black')

    #网格刻度
    gl = ax.gridlines(draw_labels=True,linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    gl.x_inline = False
    gl.xlocator = mticker.FixedLocator([90, 100, 110, 120])
    gl.ylocator = mticker.FixedLocator([20,30,40])
    gl.xlabel_style = {'fontsize':12}
    gl.ylabel_style = {'fontsize':12}
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.rotate_labels = False
    gl.xpadding = 12

    #nanhai
    sub_ax = fig.add_axes([0.75, 0.1, 0.1, 0.2],projection=proj)
    # 绘制边界线
    china = gpd.read_file('D:/workspace/rice_crop_model/src/china_map/中国省级地图GS（2019）1719号.geojson')
    nine = gpd.read_file('D:/workspace/rice_crop_model/src/china_map/九段线GS（2019）1719号.geojson')
    nine.plot(ax=sub_ax,transform=ccrs.PlateCarree(),facecolor='w',edgecolor='k',lw=1,zorder=3,alpha=1)
    china.plot(ax=sub_ax,transform=ccrs.PlateCarree(),facecolor='w',edgecolor='k',lw=1,zorder=3, alpha=1)

    #添加其他要素
    sub_ax.add_feature(cfeature.OCEAN.with_scale('50m'))
    sub_ax.add_feature(cfeature.LAND.with_scale('50m'))
    p = sub_ax.imshow(data, origin='upper', transform=ccrs.PlateCarree(),extent=tif_extent ,cmap=cmap,zorder=3,alpha=0.55)
    # 设置图形范围
    sub_ax.set_extent([105, 122, 2, 22])

    # 加载数据点
    df = pd.read_csv('D:/Python code/station catalog.csv', encoding='gbk')
    bins = [0, 3, 6, 10, 14]  # median_thermal():
    lables = ['0-3', '3-6', '6-10', '10-14']
    df['mark_year'] = pd.cut(df['num_years'], bins=bins, labels=lables)
    grouped_data = df.groupby('mark_year')
    marker_dict = {'0-3': 'o', '3-6': 's', '6-10': '^', '10-14': '<'}
    size_dict = {'0-3': 35, '3-6': 35, '6-10': 35, '10-14': 35}
    for mark_year, group in grouped_data:
        ax.scatter(group['经度'], group['纬度'], s=size_dict[mark_year], marker=marker_dict[mark_year],
                   transform=ccrs.PlateCarree(), label=str(mark_year), alpha=1,zorder=4)
    # 加载图例
    ax.legend(loc='lower left', ncol=1, fontsize=12, frameon=False)

    # 显示图形
    plt.show()
    # save the figure
    fig.savefig('D:/workspace/rice_crop_model/src/fig1_research.png', dpi=300, bbox_inches='tight')

if __name__ == '__main__':
    map()