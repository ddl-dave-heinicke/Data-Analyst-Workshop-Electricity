#import the packages we need
import pandas as pd
# import numpy as np
import datetime
import os
import requests
import csv

def pull_data(start=None, end=None):
    
    # Convert start and end to dates. If no start time is provided, default to Jan 1, 2019
    # If no end date is provided, defualt to yesterday.
    
    if start:
        start_date = datetime.datetime.strptime(start, '%Y-%m-%d 00:00:00')
    else:
        start_date_str = '2019-01-01 00:00:00'
        start_date = datetime.datetime.strptime(start_date_str, '%Y-%m-%d 00:00:00')
    
    if end:
        end_date = datetime.datetime.strptime(end, '%Y-%m-%d 00:00:00')
    else:
        end_date = datetime.datetime.today() - datetime.timedelta(days=1)

    # BMS has a max 90 day period for downloads. Break up training data into 90 day chunks
    next_start_date = start_date

    print('Training data runs from {} to {}'.format(start_date, end_date))
    print('Pulling training data from BMR')


    while next_start_date < end_date:
        # Create the file name
        data_name = 'raw_data/data_' + next_start_date.strftime('%Y-%m-%d') + '.csv'
        next_end_date = min(end_date, next_start_date + datetime.timedelta(days=90))

        start = next_start_date.strftime('%Y-%m-%d')
        end = next_end_date.strftime('%Y-%m-%d')

        url = 'https://www.bmreports.com/bmrs/?q=ajax/filter_csv_download/FUELHH/csv/FromDate%3D{}%26ToDate%3D{}/&filename=GenerationbyFuelType_20191002_1657'.format(start, end)

        # Write respose to temp csv file
        with requests.Session() as s:

            download = s.get(url)

            decoded_content = download.content.decode('utf-8')

            cr = csv.reader(decoded_content.splitlines(), delimiter=',')

            my_list = list(cr)

            with open('raw_data/temp.csv', 'w', newline='\n') as f:
                for row in my_list:
                    writer = csv.writer(f)
                    writer.writerow(row)

        os.rename('raw_data/temp.csv', data_name)
        next_start_date = next_end_date
        print('Downloading {} to {}'.format(start, end))
    
    # Roll up indididual csvs as a single training set
    print('Preparing traiing and test data')

    consolidated_data = pd.DataFrame()

    for f in os.listdir('raw_data/'):
        if f.endswith('.csv'):
            df = pd.read_csv('raw_data/' + f, skiprows=1, skipfooter=1, header=None, engine='python')
            df = df.iloc[:,0:18]
            df.columns = ['HDF', 'date', 'half_hour_increment',
                  'CCGT', 'OIL', 'COAL', 'NUCLEAR',
                  'WIND', 'PS', 'NPSHYD', 'OCGT',
                  'OTHER', 'INTFR', 'INTIRL', 'INTNED',
                   'INTEW', 'BIOMASS', 'INTEM']
            consolidated_data = consolidated_data.append(df)

    # Sort by dates ascending
    consolidated_data = consolidated_data.sort_values(by=['date', 'half_hour_increment'], ascending=True)

    # Clean up dates
    consolidated_data['date'] = pd.to_datetime(consolidated_data['date'], format="%Y%m%d")
    consolidated_data['date'] = consolidated_data.apply(lambda x:x['date']+ datetime.timedelta(minutes=30*(int(x['half_hour_increment'])-1)), axis = 1)
    consolidated_data.rename(columns={'date':'datetime'}, inplace=True)
    consolidated_data.drop('half_hour_increment', axis=1, inplace=True)

    consolidated_data = consolidated_data.set_index('datetime')
    
    now = str(datetime.datetime.today())
    
    # Save the consolidated data back to the project
    consolidated_data.to_csv('consolidated_data_{}.csv'.format(now), index=False)