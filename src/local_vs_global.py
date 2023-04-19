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
from photo_period_effect import photoeffect_yin,photoeffect_oryza200,photoeffect_wofost
from T_dev_effect import Wang_engle,T_base_op_ceiling,T_base_opt
os.chdir(os.path.dirname(os.path.realpath(__file__)))

pd.options.display.max_columns = 999

def Test_photo_response_fun():
    plot(range(1, 24), [photoeffect_yin(DL=dl) for dl in range(1, 24)])
    show()
    plot(range(1, 24), [photoeffect_wofost(DL=dl) for dl in range(1, 24)])
    show()
    plot(range(1, 24), [photoeffect_oryza200(DL=dl) for dl in range(1, 24)])
    show()
def Test_T_effect_fun():
    plot(range(0, 50), [Wang_engle(T=tem) for tem in range(0, 50)])
    show()    
    plot(range(0, 50), [T_base_opt(T=tem) for tem in range(0, 50)])
    show()    
    plot(range(0, 50), [T_base_op_ceiling(T=tem) for tem in range(0, 50)])
    show()    


if __name__=='__main__':
    Test_T_effect_fun()

