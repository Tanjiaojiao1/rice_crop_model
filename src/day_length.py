
import numpy as np
from pandas import *
from pylab import *
import datetime
def TwoDPlotDayLengthChina_south():
    import Sun
    sun=Sun.Sun()
    dl=[]
    tickps=[]
    dts=[]
    for lat in np.linspace(15,35,200):
        # print(lat)
        count=0
        for dt in np.arange('2016-03-01', '2016-10-31', dtype='datetime64[D]'):
            dt=to_datetime(dt)
            
            dl.append(sun.dayLength(year=dt.year, month=dt.month, day=dt.day, lon=0, lat=lat))
            if lat==15.0 and dt.day in [1]:
                tickps.append(count)
                dts.append(dt.strftime("%b-%d"))
            count+=1
    print (len(dts))
    
    print (len(dl)/400)
    # print(tickps)
    fig=plt.figure(figsize=(8,8))
    fig.subplots_adjust(left=0.0, bottom=0.0, right=1, top=1,wspace=0.0, hspace=0.0)
    f1=fig.add_subplot(1,1,1)   

    im=f1.imshow(np.flipud(np.array(dl).reshape((200,(datetime.date(year=2016,month=10,day=31)-datetime.date(year=2016,month=3,day=1)).days))),cmap='RdYlGn_r');
    cax = fig.add_axes([1.05,0.25,0.02,0.5])#
    f1.set_yticks(np.linspace(0,200,8))
    print (tickps,dts)
    f1.set_xticks(tickps)
    f1.set_xticklabels(dts,fontsize=8)
    f1.set_ylabel('Latitude',fontsize=12)
    f1.set_xlabel('Date',fontsize=12)
    print (np.linspace(0,200,8))
    print (['%d$^\circ$'%lat for lat in np.arange(5,55,5)])
    f1.set_yticklabels(['%d$^\circ$'%lat for lat in np.arange(35,15,-2.5)],fontsize=10)
    cbar=colorbar(im,cax,orientation='vertical',ticks=np.arange(6,20,0.5))
    cbar.set_label('Day length (h)',fontsize=12)
    fig.savefig('../fig/TwoDPlotDayLength_south_china.png', dpi=300, bbox_inches='tight', pad_inches=0.01)
if __name__=='__main__':
    TwoDPlotDayLengthChina_south()