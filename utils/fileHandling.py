# -*- coding: utf-8 -*-
"""
Created on Wed Oct 28 17:21:04 2020

@author: poseidon
"""


import pandas as pd
import numpy  as np
from datetime import datetime

def ConvertDate( x ):
    year, month, day = x.split('-')
    return (pd.to_datetime(month + '/' + day + '/' + year))



def getData( filename, Strategy = 'Differenced' ):
    
    # Checking Strategy
    if (Strategy != 'Differenced' and Strategy != 'Returns'):
        print('[WARNING] Strategy: %s not known' % Strategy)
        Strategy = 'Differenced'
        
    print('[INFO] Strategy: %s was selected' % Strategy)
    
    
    
    
    Series = pd.read_csv( filename )
    Series = Series.interpolate()

    Series['Date'] = Series['Date'].apply(ConvertDate)
    Series.sort_values('Date')
    
    # Set date as index
    Series['Date'].astype('datetime64')
    Series.set_index('Date', inplace=True)

    #Series = Series[ ['pressure ', 'T_Independent', 'T-depend'] ]
    if (Strategy == 'Differenced'):
        # Difference the Series
        Transformed_Series = Series.diff()[1:]
    else:
        Series             = np.log( Series )
        Transformed_Series = Series.diff()[1:]
        #Transformed_Series = Series.diff()[1:] / Series[:-1].values
    
    
    return ( Series, Transformed_Series )




def splitData(Series, Dates):
    valid_date_start = datetime.strptime(Dates[0], '%Y-%m-%d')
    valid_date_end   = datetime.strptime(Dates[1], '%Y-%m-%d')

    
    # Training Data
    Training   = Series[ Series.index < valid_date_start.isoformat() ]
    # Validation Data
    Validation = Series[ (Series.index >= valid_date_start.isoformat()) & (Series.index < valid_date_end.isoformat())]
    # Testing Data
    Testing    = Series[Series.index >= valid_date_end.isoformat() ]

    return (Training, Validation, Testing)

