#import the packages we need
import pandas as pd
import datetime
import os
import sys
import argparse
import requests
import csv
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

def pull_data(start=None, end=None):
    
    # Convert start and end to dates. 
    start_date = datetime.datetime.strptime(start, '%Y-%m-%d %H:%M:%S')
    
    end_date = datetime.datetime.strptime(end, '%Y-%m-%d %H:%M:%S')


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
            consolidated_data = pd.concat([consolidated_data, df], axis=0)

    # Sort by dates ascending
    consolidated_data = consolidated_data.sort_values(by=['date', 'half_hour_increment'], ascending=True)

    # Clean up dates
    consolidated_data['date'] = pd.to_datetime(consolidated_data['date'], format="%Y%m%d")
    consolidated_data['date'] = consolidated_data.apply(lambda x:x['date']+ datetime.timedelta(minutes=30*(int(x['half_hour_increment'])-1)), axis = 1)
    consolidated_data.rename(columns={'date':'datetime'}, inplace=True)
    consolidated_data.drop('half_hour_increment', axis=1, inplace=True)

    # consolidated_data = consolidated_data.set_index('datetime')
    now = str(datetime.datetime.today())
    
    # Save the consolidated data back to the project
    consolidated_data.to_csv('/domino/datasets/local/{}/PowerGenerationData_{}_to_{}.csv'.format(os.environ.get('DOMINO_PROJECT_NAME'), str(start_date), str(end_date)), index=False)
    
    print(consolidated_data.head())

    # Vizualize the pulled data
    
    df = consolidated_data.copy()
    
    # Create total output feature: sum of all fuel sources.
    df['TOTAL'] = df[['CCGT', 'OIL', 'COAL', 'NUCLEAR', 'WIND', 'PS', 'NPSHYD', 'OCGT', 'OTHER', 'INTFR', 'INTIRL', 'INTNED', 'INTEW', 'BIOMASS', 'INTEM']].sum(axis=1)

    # Select CCGT, Wind, Nuclear, Biomass and Coal & create "Other" column
    plot_cols = ['CCGT', 'WIND', 'NUCLEAR','BIOMASS', 'COAL', 'TOTAL']

    df_plot = df[plot_cols].copy()

    df_plot['OTHER'] = df_plot['TOTAL'] - df_plot[['CCGT', 'WIND', 'NUCLEAR','BIOMASS', 'COAL']].sum(axis=1, numeric_only=True)

    # Plot Cumulative production up to prediction point
    x = df.datetime
    y = [df.NUCLEAR, df.BIOMASS, df.COAL, df.OTHER, df.WIND, df.CCGT,]

    fig, ax = plt.subplots(1,1, figsize=(12,8))

    colors = ['tab:red','tab:olive', 'tab:gray','tab:orange','tab:green','tab:blue']

    ax.stackplot(x,y,
                 labels=['NUCLEAR', 'BIOMASS', 'COAL', 'OTHER', 'WIND', 'CCGT (GAS)'],
                 colors=colors,
                 alpha=0.8)

    # Format the stack plot
    ax.legend(bbox_to_anchor=(1.25, 0.6), loc='right', fontsize=14)
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=7))
    plt.xticks(rotation=45, ha='right', fontsize=12)
    ax.set_ylabel('Total Production, MW', fontsize=16)
    ax.set_title('Cumulative Production, MW', fontsize=16)

    # Save the figure as an image to the Domino File System
    fig.savefig('/mnt/visualizations/Cumulative Production.png', bbox_inches="tight")

    plt.show()
    
    # Show total power generated & peak generation over the dataset time window in 
    total_energy_produced = round(sum(df_plot['TOTAL'] / 2 / 1000000), 1)
    max_production = round(max(df_plot['TOTAL']), 2)
    
    print("Total energy produced in the UK between {} and {}: {} TWH \n".format(start_date, end_date, total_energy_produced))
    print("Peak production in UK between {} and {}: {} MW \n".format(start_date, end_date, max_production))
    
    #Code to write Total and Peak values to dominostats value for population in jobs
    with open('dominostats.json', 'w') as f:
        f.write(json.dumps({"Total TWH": total_energy_produced,
                            "Max MWH": max_production}))
    
if __name__ == "__main__":
    
    # Pass start and end dates. If no start time is provided, default to Jan 1, 2019.
    # If no end date is provided, defualt to yesterday.
    
    parser=argparse.ArgumentParser()
    
    previous_month = (datetime.datetime.today() - datetime.timedelta(days=31)).strftime('%Y-%m-%d %H:%M:%S')
    yesterday = (datetime.datetime.today() - datetime.timedelta(days=1)).strftime('%Y-%m-%d %H:%M:%S')
    
    
    parser.add_argument("--start", type=str, default=previous_month, help="Start Date")
    parser.add_argument("--end", type=str, default=yesterday, help="End Date")
    
    args = parser.parse_args()
    
    pull_data(args.start, args.end)