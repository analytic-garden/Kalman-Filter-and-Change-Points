#!/home/bill/anaconda3/envs/bio2/bin/python
# -*- coding: utf-8 -*-
"""
generate_data.py
        A python program to generate fake data for Kalman Filters.

@author: Bill Thompson
@license: GPL 3
@copyright: 2021_08_01
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def main():
    out_file = '/mnt/d/Documents/pCO2/flickering_switch/kf_change_point/data/test3.csv'
    
    dates = []
    temp_anomaly = []
    
    for i in range(25):
        dates.append(1900 + i)
        temp_anomaly.append(0.1 + 
                            np.random.normal(loc = 0, scale = 0.1, size = 1)[0])
        
    for i in range(25, 50):
        dates.append(1900 + i)
        temp_anomaly.append(temp_anomaly[24] + 0.03 * (i - 25) + 
                            np.random.normal(loc = 0, scale = 0.4, size = 1)[0])
        
    for i in range(50, 100):
        dates.append(1900 + i)
        temp_anomaly.append(temp_anomaly[49] + 0.01 * (i - 50) + 
                            np.random.normal(loc = 0, scale = 0.2, size = 1)[0])
        
    for i in range(100, 150):
        dates.append(1900 + i)
        temp_anomaly.append(temp_anomaly[99] + 
                            np.random.normal(loc = 0, scale = 0.1, size = 1)[0])
            
    df = pd.DataFrame({'Year': dates, 'Temperature_Anomaly': temp_anomaly})
    df.to_csv(out_file, index = False)
    
    size = 2 
    fig = plt.figure()
    ax1 = fig.add_subplot(1,1,1)
    ax1.plot(dates, temp_anomaly, 'o', markersize = size)
    ax1.set_xlabel('Year')
    ax1.set_ylabel('Temperature Anomaly')
    plt.show()
    
if __name__ == '__main__':
    main()