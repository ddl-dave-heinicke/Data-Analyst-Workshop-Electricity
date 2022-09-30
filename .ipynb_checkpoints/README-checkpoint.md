## Domino Hands-On Workshop: Analyzing Electricity Production in the UK

#### In this workshop you will work through an end-to-end workflow broken into various labs to:

* Create a Domino Project, invite colaborators & set up project communication
* Create a laptop in the cloud - or a Domino Workspace
* Read in and vizualize historic data from a connected source.
* Run & schedule jobs to read in and process data from a live source
* Building a hosted app to predict electric genration

# Section 1 - Project Set Up

### Lab 1.1 - Forking Existing Projects
Once you have access to the Domino training environment - Guide your mouse to the left blue menu and click the **Search** page. Afterwards, type the word 'Training' in the cell provided and click enter to discover any projects tagged under 'Training'. (The left blue menu shrinks to show only the icon of the pages. Unshrink the left blue menu by guiding your mouse over the icon pages.)

Select the project called PowerGenerationWorkshop

<!-- ![image](readme_images/Search.png) -->

<p align="center">
<img src = readme_images/Search.png width="800">
</p>

Read the readme to learn more about the project's use case, status, etc.

In the top right corner, choose the icon to **fork** the project. Name the project *Domino-Training-yourname*

<!-- ![image](readme_images/Fork.png) -->

<p align="center">
<img src = readme_images/Fork.png width="800">
</p>

In your new project - go into the settings tab

View the default hardware tier and compute environment - ensure the environment is set to 'Domino-PowerGeneration-Workshop-Environment':

<!-- ![image](readme_images/ProjectSettings.png) -->

<p align="center">

<img src = readme_images/ProjectSettings.png width="800">
</p>

Go to the Access and Sharing tab - change your project visibility to **Public**

<!-- ![image](readme_images/ProjectVisibility.png) -->

<p align="center">
<img src = readme_images/ProjectVisibility.png width="800">
</p>

Add your instructor or another attendee as a collaborator in your project. 
<!-- ![image](readme_images/AddCollaborator.png) -->

<p align="center">
<img src = readme_images/AddCollaborator.png width="800">
</p>

Change their permissions to Results Consumer.
<!-- ![image](readme_images/ResultsConsumer.png) -->

<p align="center">
<img src = readme_images/ResultsConsumer.png width="800">
</p>

### Lab 1.2 - Defining Project Goals

Click back into the Overview area of your project. Then navigate to the Manage tab.

<!-- ![image](readme_images/Overview.png) -->

<p align="center">
<img src = readme_images/Overview.png width="800">
</p>

Click on Add Goals

<!-- ![image](readme_images/AddProjectGoals.png) -->

<p align="center">
<img src = readme_images/AddProjectGoals.png width="800">
</p>

For the goal title type in 'Explore Data' and click save. Once the goal is saved click the drop down on the right to mark the goal status as 'Data Acquisition and Exploration'.


<!-- ![image](readme_images/Goal1status.png) -->

<p align="center">

<img src = readme_images/Goal1status.png width="800">
</p>

[optional] - Add a comment to the goal and tag a collaborator you've added earlier by typing @ then their username. Please click on the paper airplane to submit the comment.

<!-- ![image](readme_images/Goal1comment.png) -->

<p align="center">

<img src = readme_images/Goal1comment.png width="800">
</p>

### Lab 1.3 - Add Data Source

We will now add a data connection defined by the admin of our project to later query in data. To do so - navigate to the Data tab of your projects. If you're taken to the Domino Datasets view, please click on the Data Sources view instead and click on 'Add a Data Source'

<!-- ![image](readme_images/AddDataSource.png) -->

<p align="center">

<img src = readme_images/AddDataSource.png width="800">
</p>

Select the 'Domino-PowerGeneration-Workshop' s3 bucket connection and click add to project (The avaialble data sources on the left may look different).

<!-- ![image](readme_images/AddS3.png) -->

<p align="center">
<img src = readme_images/AddS3.png width="800">
</p>

The data source should look like the image below

<!-- ![image](readme_images/S3done.png) -->

<p align="center">
<img src = readme_images/S3done.png width="800">
</p>

This concludes all labs in section 1.

We have now created a project, added collaborators and attached a data source.

## Section 2 - Work With Historic Data

### Lab 2.1 - Exploring Workspaces

Our goal in Section 2 is to take a look at power production data in the UK in the summer of 2022 (June-August). We'll visualize the 
production data, then identify the hour in each day where production peaked, so that we can see the breakdown of energy sources
when the grid is peaking.

We'll do this in an interactive Dominio Workspace - which you can think of as a laptop in the cloud.

#### Launch a Workspace

In the top right corner click Create New Workspace.

<p align="center">
<img src = readme_images/AddWorkspace.png width="800">
</p>

Type a name for the Workspace in the 'Workspace Name' cell and next click through the available Compute Environments in the Workspace Environment drop down button. Next, ensure that Domino-PowerGeneration-Workshop-Environment is selected.

Select JupyterLab as the Workspace IDE.

Click Launch.

<p align="center">
<img src = readme_images/LaunchWorkspace.png width="800">
</p>

Once the workspace is launched, create a new python notebook by clicking here:

<p align="center">
<img src = readme_images/NewNotebook.png width="800">
</p>

#### Read in Data from S3

Once your notebook is loaded, click on the left blue menu and click on the Data page, then onto the data source we added in lab 1 as displayed below

<p align="center">
<img src = readme_images/DataTab.png width="800">
</p>

Copy the provided code snippet into your notebook and run the cell

<p align="center">
<img src = readme_images/S3CodeSnippet.png width="800">
</p>

After running the code snippet. Copy the code below into the following cell 

```python
from io import StringIO
import pandas as pd

s = str(object_store.get("PowerGenerationData_Summer_2022.csv"),'utf-8')
data = StringIO(s) 

df = pd.read_csv(data, parse_dates=['datetime'])
df.head()
```

Now cell by cell, copy the code snippets below and run the cells to visualize and prepare the data! (You can click on the '+' icon to add a blank cell after the current cell)

#### Visualize Monthly Production

There are many fuel types in the dataset. To improve visualization, we will only select the following columns, and group the minor fuel sources together into "Other".

Fuel Sources to plot:  
CCGT - Combined Cycle Gas Turbines (natural gas)  
Wind  
Nuclear  
Biomass  
Coal  
All Others - Summed Together in "Other"


```python
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

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
ax.set_title('Cumulative Production, Summer 2022, MW', fontsize=16)

# Save the figure as an image to the Domino File System
plt.savefig('visualizations/Cumulative Production.png', bbox_inches="tight")

plt.show()
```
#### Identify hours when the grid peaks each day 

Now that we've saved our image, we can extract peak demand from the raw data.

```python
# Read in 30 minute data
s = str(object_store.get("PowerGenerationData_Summer_2022.csv"),'utf-8')
data = StringIO(s) 
daily_peak_df = pd.read_csv(data, parse_dates=['datetime'])

# Set the index to the timestamp, and create a column of TOTAL demand by row
daily_peak_df = daily_peak_df.set_index('datetime')
daily_peak_df = daily_peak_df.drop(['HDF'], axis=1)
daily_peak_df['TOTAL'] = daily_peak_df.sum(axis=1)

# Group 30 minute data by day, grab the index of the 30-minute interval when the daily demand peaks
idx = daily_peak_df.groupby(pd.Grouper(freq='D'))['TOTAL'].transform(max) == daily_peak_df['TOTAL']

# Verify there are no duplicate Maximum values in a single day in the dataset
print("The peak demand table includes {} days \n".format(daily_peak_df[idx].shape[0]))

# Print a table with the daily Maximum values, and time when demand peaked
daily_peak_df = daily_peak_df[idx]

# Print total power generated over the dataset time window
total_energy_produced = sum(df_plot['TOTAL'])
max_production = max(df_plot['TOTAL'])

print("Total energy produced in the UK between June and August 2022: {} TWH \n".format(round(total_energy_produced / 2 / 1000000, 1)))
print("Peak production in UK between June and August 2022: {} MW \n".format(round(max_production)))

# Print Peak demand over the dataset time window

df_plot.head()
```
We can also visualize the hours of the day when demand peaks, and save our plot.

```python
import seaborn as sns

plt.figure(figsize=(8,5))
plt.title('Hours in the day When Demand Peaks in the UK, Summer 2022')
plt.xlabel('Hour of the Day')
sns.histplot(daily_peak_df.index.hour, stat='count', bins=10)
plt.savefig('visualizations/Peak Demand Hours.png', bbox_inches="tight")
plt.show()
```

Power production peaks at a different time each day. But what sources contribute to peak production each day? And does the breakdown change from month to month?


```python
# Extract the month from the daily peak power dataframe
daily_peak_df['Month'] = daily_peak_df.index
daily_peak_df['Month'] = daily_peak_df['Month'].apply(lambda x: x.strftime('%b'))

# Group minimal sources into "Other" for simplicity
daily_peak_df['OTHER'] = daily_peak_df['TOTAL'] - daily_peak_df[['CCGT', 'WIND', 'NUCLEAR','BIOMASS', 'COAL']].sum(axis=1, numeric_only=True)
daily_peak_df = daily_peak_df[['CCGT', 'WIND', 'NUCLEAR','BIOMASS', 'COAL', 'OTHER', 'Month']]

# Group By Month
monthly_totals = daily_peak_df.groupby('Month').sum()

# Plot the contribution of each source by month
colors = ['tab:blue','tab:green', 'tab:red','tab:olive','tab:gray','tab:orange']
labels= monthly_totals.columns 

for i, row in monthly_totals.iterrows():
    plt.figure(figsize=(8,5))
    plt.title('Total Contribution to Peak Production by Source, Month: {}'.format(i))
    plt.pie(row, labels=labels, colors=colors, autopct='%1.1f%%')
```

After running EDA, you may want to save the daily peak demand dataframe for later reference. Here, we’ll save the dataset in the project in a Domino Dataset. However, we could also write the results back to s3, or any external storage if we wanted to. 

```python
import os
path = str('/domino/datasets/local/{}/Daily_Peak_Production_Summer_2022.csv'.format(os.environ.get('DOMINO_PROJECT_NAME')))
daily_peak_df.to_csv(path, index = False)
```

Rename your notebook 'EDA_code.ipynb' by right clicking on the file name as shown below then click the Save icon.

<p align="center">
<img src = readme_images/RenameAndSaveNotebook.png width="800">
</p>

### Lab 2.2 - Syncing Files

All of the changes we have made so far are saved in our workspace, and can be retrieved anytime we open this workspace. It’s as if we have saved changes to our local machine. However, to make our code updates visible to collaborators and available to other workspaces, we want to sync our work back to the project files.


<p align="center">
<img src = readme_images/SyncProject.png width="800">
</p>

Enter an informative but brief commit message such as "Completed EDA notebook" and click to Sync All Changes. 

Click the Domino logo on the upper left corner of the blue menu and select on the Project page. Then select your project followed by selecting “Files” on the left blue menu as shown below.   

Notice that the latest commit will reflect the commit message you just logged and you can see 'EDA_code.ipynb' in your file directory.

<p align="center">
<img src = readme_images/DFS.png width="800">
</p>

Click on your notebook to view it. On the top of your screen and click 'Link to Goal' in the dropdown, after selecting the goal you created in Lab 1.2

<p align="center">
<img src = readme_images/LinkToGoal.png width="800">
</p>

Now navigate to Overview, then to the manage tab and see your linked notebook.

Click the ellipses on the goal to mark the goal as complete

<p align="center">
<img src = readme_images/MarkGoalComplete.png width="800">
</p>


### Lab 2.3 - Run and Track Jobs

Workspaces are great environments for doing exploratory work and writing code. However, once our code is finished, we may want to run it regularly- which would be tedious if we have to spin up a workspace each time.

To simply run our code in our predefined environment and quickly visualize outputs, Domino has a feature called Jobs. Jobs spin up an instance, run a script, save outputs, and shut down the instance for us.

<p align="center">
<img src = readme_images/NavigateToJobs.png width="800">
</p>


In this example, we want to pull some more recent September data from BMRS and save it to the Domino File System.

Type in the following command below in the **File Name** section of the **Start a Job** pop up window. Click on **Start** to run the job.

```shell
scripts/pull_data.py '--start=2022-09-01 00:00:00' '--end=2022-09-7 00:00:00'
```

<p align="center">
<img src = readme_images/Jobsrun.png width="800">
</p>

Click into the pull_data.py job run.

In the details tab of the job run note that the compute environment and hardware tier are tracked to document not only who ran the experiment and when, but what versions of the code, software, and hardware were executed.

In the details tab of the job run note that the compute environment is tracked to document not only who ran the experiment and when, but what versions of the code, software, and hardware were executed.

Now, pull a couple new weeks of data by starting new jobs and pasting the following commands:

```shell
scripts/pull_data.py '--start=2022-09-08 00:00:00' '--end=2022-09-14 00:00:00'
```
```shell
scripts/pull_data.py '--start=2022-09-015 00:00:00' '--end=2022-09-21 00:00:00'
```

You'll notice that the tracker at the top has begun populating with stats from our runs - in this case total production over the time window in TWH and peak production in GW. When you run Domino jobs, you can save any stat you'd like from the run to include in the jobs tracker. 

You do this by populating the dominostats json file in your script. In `pull_data.py` , saving the peak demand and peak production stats looks like this:

```python
    # Show total power generated & peak generation over the dataset time window 
    total_energy_produced = round((sum(df_plot['TOTAL']) / 2 / 1000000), 1)
    max_production = round(max(df_plot['TOTAL']) / 1000, 2)
    
    #Code to write Total and Peak values to dominostats value for population in jobs
    with open('dominostats.json', 'w') as f:
        f.write(json.dumps({"Total TWH": total_energy_produced,
                            "Max GW": max_production}))
```

Finally, click into Domino Datasets and examine the contents in “Power Generation Workshop”. The September datasets should be there. 

<p align="center">
<img src = readme_images/UpdatedDataJob.png width="800">
</p>

### Lab 2.4 - Schedule Jobs

Say we wanted to pull the data each month. Rather than running this job manually, we can schedule the job to run in Domino.

The script we ran manually, pull_data.py, defaults to pulling the past 30 days if we don't pass it an start and end date, so we can simply schedule it to run each month.

Navigate to “Scheduled Jobs” under “Publish”, and select, “New Scheduled Job”

<p align="center">
<img src = readme_images/NewScheduleJob.png width="800">
</p>

Paste the following unde "File Name or Command", give the job the name "Monthly Data Pull", and verify that the Environment is "Domino-PowerGeneration-Workshop-Environment". It should be, since we set it as our project environemnt, but it's always good to check. Click “Next”. 

```shell
scripts/pull_data.py
```
<p align="center">
<img src = readme_images/SetJobDefinition.png width="800">
</p>

Have Domino Run the script every month on the 1st of the  month.

<p align="center">
<img src = readme_images/SetJobSchedule.png width="800">
</p>

Under notify emails, tag yourself to be notified when the job runs - and Create!

<p align="center">
<img src = readme_images/SetJobNotification.png width="800">
</p>


## Section 3 - Create Applications


### Lab 3.1 Deploying Web App
    
Static reports and dashboards are a useful way to consume data and insights, but taking the next step to an interactive application can go further and deriving value from your data. Domino's open platform allows you to build and host apps in many different frameworks, such as Dash, Shiny, Steamlit and others. Once Domino has spun up the app for you, you can share it with your colleagues, even if they do not have Domino licenses.

In this lab, we won't go through the details of building an app with one of the avaialbe frameworks. We'll take an existing app script built in Dash, and walk through the process of hosting the app with Domino. 

The app that we'll use takes historic power generation data over a user-defined time window, fits a time series forecast model to the usage data, the forecasts future production. 
    
To do so - in a new browser tab first navigate back to your Project and then in the left blue menu of your project click into the **Files** section and click **New File**

<p align="center">
<img src = readme_images/AddNewFileforAppsh.png width="800">
</p>     

Next, we will create a file called app.sh. It's a simple bash script that Domino uses to start and run the Dash App server based on the inputs provided. This script includes examples for calling a Flask, R/Shiny and Dash app. The Flask and Shiny lines are commented out, but are there for reference if you'd like to try hosting a Flask or Shiny app.

Copy the following code snippet in - 

```shell
#!/usr/bin/env bash
 
# This is a bash script for Domino's App publishing feature
# Learn more at http://support.dominodatalab.com/hc/en-us/articles/209150326
 
## R/Shiny Example
## This is an example of the code you would need in this bash script for a R/Shiny app
## R -e 'shiny::runApp("./scripts/shiny_app.R", port=8888, host="0.0.0.0")'
 
## Flask example
## This is an example of the code you would need in this bash script for a Python/Flask app
#export LC_ALL=C.UTF-8
#export LANG=C.UTF-8
#export FLASK_APP=app-flask.py
#export FLASK_DEBUG=1
#python -m flask run --host=0.0.0.0 --port=8888
 
## Dash Example
## This is an example of the code you would need in this bash script for a Dash app
python scripts/app.py
```
Name the file **app.sh** and click **Save**
<p align="center">
<img src = readme_images/appsh.png width="800">
</p>         


Now navigate back into the Files tab, and enter the **scripts** folder. Click add a new file and name it `app.py` (make sure the file name is exactly that, it is case sensitive) and then paste the following into the file, and Save.

<p align="center">
<img src = readme_images/SavePythonApp.png width="800">
</p> 

Copy all of the following code into `app.py`:

```python
# -*- coding: utf-8 -*-
import dash
import dash_core_components as dcc
import dash_html_components as html
from datetime import datetime as dt
from dash.dependencies import Input, Output
import requests
import datetime
import os
 
import pandas as pd
import datetime
import matplotlib.pyplot as plt
from fbprophet import Prophet
import plotly.graph_objs as go
 
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
 
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
 
app.config.update({'requests_pathname_prefix': '/{}/{}/r/notebookSession/{}/'.format(
    os.environ.get("DOMINO_PROJECT_OWNER"),
    os.environ.get("DOMINO_PROJECT_NAME"),
    os.environ.get("DOMINO_RUN_ID"))})
 
colors = {
    'background': '#111111',
    'text': '#7FDBFF'
}
 
# Plot configs
prediction_color = '#0072B2'
error_color = 'rgba(0, 114, 178, 0.2)'  # '#0072B2' with 0.2 opacity
actual_color = 'black'
cap_color = 'black'
trend_color = '#B23B00'
line_width = 2
marker_size = 4
uncertainty=True
plot_cap=True
trend=False
changepoints=False
changepoints_threshold=0.01
xlabel='ds'
ylabel='y'
 
app.layout = html.Div(style={'paddingLeft': '40px', 'paddingRight': '40px'}, children=[
    html.H1(children='Generate a Dataset for Power Generation in UK Forecasting'),
    html.Div(children='''
        This is a web app developed in Dash and published in Domino.
        You can add more description about the app here if you'd like.
    '''),
     html.Div([
        html.P('Select a Fuel Type:', className='fuel_type', id='fuel_type_paragraph'),
        dcc.Dropdown(
            options=[
                {'label': 'Combined Cycle Gas Turbine', 'value': 'CCGT'},
                {'label': 'Oil', 'value': 'OIL'},
                {'label': 'Coal', 'value': 'COAL'},
                {'label': 'Nuclear', 'value': 'NUCLEAR'},
                {'label': 'Wind', 'value': 'WIND'},
                {'label': 'Pumped Storage', 'value': 'PS'},
                {'label': 'Hydro (Non Pumped Storage', 'value': 'NPSHYD'},
                {'label': 'Open Cycle Gas Turbine', 'value': 'OCGT'},
                {'label': 'Other', 'value': 'OTHER'},
                {'label': 'France (IFA)', 'value': 'INTFR'},
                {'label': 'Northern Ireland (Moyle)', 'value': 'INTIRL'},
                {'label': 'Netherlands (BritNed)', 'value': 'INTNED'},
                {'label': 'Ireland (East-West)', 'value': 'INTEW'},
                {'label': 'Biomass', 'value': 'BIOMASS'},
                {'label': 'Belgium (Nemolink)', 'value': 'INTEM'},
            {'label': 'France (Eleclink)', 'value': 'INTEL'},
            {'label': 'France (IFA2)', 'value': 'INTIFA2'},
           {'label': 'Norway 2 (North Sea Link)', 'value': 'INTNSL'}
            ],
            value='CCGT',
            id='fuel_type',
            style = {'width':'auto', 'min-width': '300px'}
        )
    ], style={'marginTop': 25}),
    html.Div([
        html.Div('Training data will end today.'),
        html.Div('Select the starting date for the training data:'),
        dcc.DatePickerSingle(
            id='date-picker',
            date=dt(2021, 9, 10)
        )
    ], style={'marginTop': 25}),
    html.Div([
        dcc.Loading(
            id="loading",
            children=[dcc.Graph(id='prediction_graph',)],
            type="circle",
            ),
        ], style={'marginTop': 25})
])
 
@app.callback(
    # Output('loading', 'chhildren'),
    Output('prediction_graph', 'figure'),
    [Input('fuel_type', 'value'),
     Input('date-picker', 'date')])
def update_output(fuel_type, start_date):
    today = datetime.datetime.today().strftime('%Y-%m-%d')
    start_date_reformatted = start_date.split('T')[0]
    url = 'https://www.bmreports.com/bmrs/?q=ajax/filter_csv_download/FUELHH/csv/FromDate%3D{start_date}%26ToDate%3D{today}/&filename=GenerationbyFuelType_20191002_1657'.format(start_date = start_date_reformatted, today = today)
    r = requests.get(url, allow_redirects=True)
    open('data.csv', 'wb').write(r.content)
    df = pd.read_csv('data.csv', skiprows=1, skipfooter=1, header=None, engine='python')
    df.columns = ['HDF', 'date', 'half_hour_increment',
                'CCGT', 'OIL', 'COAL', 'NUCLEAR',
                'WIND', 'PS', 'NPSHYD', 'OCGT',
                'OTHER', 'INTFR', 'INTIRL', 'INTNED', 'INTEW', 'BIOMASS', 'INTEM',
                'INTEL','INTIFA2', 'INTNSL']
    df['datetime'] = pd.to_datetime(df['date'], format="%Y%m%d")
    df['datetime'] = df.apply(lambda x:
                          x['datetime']+ datetime.timedelta(
                              minutes=30*(int(x['half_hour_increment'])-1))
                          , axis = 1)
    df_for_prophet = df[['datetime', fuel_type]].rename(columns = {'datetime':'ds', fuel_type:'y'})
    m = Prophet()
    m.fit(df_for_prophet)
    future = m.make_future_dataframe(periods=72, freq='H')
    fcst = m.predict(future)
    # from https://github.com/facebook/prophet/blob/master/python/fbprophet/plot.py
    data = []
    # Add actual
    data.append(go.Scatter(
        name='Actual',
        x=m.history['ds'],
        y=m.history['y'],
        marker=dict(color=actual_color, size=marker_size),
        mode='markers'
    ))
    # Add lower bound
    if uncertainty and m.uncertainty_samples:
        data.append(go.Scatter(
            x=fcst['ds'],
            y=fcst['yhat_lower'],
            mode='lines',
            line=dict(width=0),
            hoverinfo='skip'
        ))
    # Add prediction
    data.append(go.Scatter(
        name='Predicted',
        x=fcst['ds'],
        y=fcst['yhat'],
        mode='lines',
        line=dict(color=prediction_color, width=line_width),
        fillcolor=error_color,
        fill='tonexty' if uncertainty and m.uncertainty_samples else 'none'
    ))
    # Add upper bound
    if uncertainty and m.uncertainty_samples:
        data.append(go.Scatter(
            x=fcst['ds'],
            y=fcst['yhat_upper'],
            mode='lines',
            line=dict(width=0),
            fillcolor=error_color,
            fill='tonexty',
            hoverinfo='skip'
        ))
    # Add caps
    if 'cap' in fcst and plot_cap:
        data.append(go.Scatter(
            name='Cap',
            x=fcst['ds'],
            y=fcst['cap'],
            mode='lines',
            line=dict(color=cap_color, dash='dash', width=line_width),
        ))
    if m.logistic_floor and 'floor' in fcst and plot_cap:
        data.append(go.Scatter(
            name='Floor',
            x=fcst['ds'],
            y=fcst['floor'],
            mode='lines',
            line=dict(color=cap_color, dash='dash', width=line_width),
        ))
    # Add trend
    if trend:
        data.append(go.Scatter(
            name='Trend',
            x=fcst['ds'],
            y=fcst['trend'],
            mode='lines',
            line=dict(color=trend_color, width=line_width),
        ))
    # Add changepoints
    if changepoints:
        signif_changepoints = m.changepoints[
            np.abs(np.nanmean(m.params['delta'], axis=0)) >= changepoints_threshold
        ]
        data.append(go.Scatter(
            x=signif_changepoints,
            y=fcst.loc[fcst['ds'].isin(signif_changepoints), 'trend'],
            marker=dict(size=50, symbol='line-ns-open', color=trend_color,
                        line=dict(width=line_width)),
            mode='markers',
            hoverinfo='skip'
        ))
 
    layout = dict(
        showlegend=False,
        yaxis=dict(
            title=ylabel
        ),
        xaxis=dict(
            title=xlabel,
            type='date',
            rangeselector=dict(
                buttons=list([
                    dict(count=7,
                         label='1w',
                         step='day',
                         stepmode='backward'),
                    dict(count=1,
                         label='1m',
                         step='month',
                         stepmode='backward'),
                    dict(count=6,
                         label='6m',
                         step='month',
                         stepmode='backward'),
                    dict(count=1,
                         label='1y',
                         step='year',
                         stepmode='backward'),
                    dict(step='all')
                ])
            ),
            rangeslider=dict(
                visible=True
            ),
        ),
    )
    return {
        'data': data,
        'layout': layout
    }
 
if __name__ == '__main__':
    app.run_server(port=8888, host='0.0.0.0', debug=True)
```
                
Now that you have your app.sh and shiny_app.R files created. Navigate to the **App** tab in your project

Enter a title for your app - 'Power-Forecast-App-yourname'. Note that Domino already knows about the app.sh file we added earlier.

<p align="center">
<img src = readme_images/LaunchApp.png width="800">
</p>       

Click Publish.
                   
You'll now see the below screen, once your app is active (should be within ~1-3 minutes) you can click the View App button. 

<p align="center">
<img src = readme_images/ViewApp.png width="800">
</p>       
        
Once you're in the app you can try out sending different forecasts from the time series model using the form on the left side of your page. Note that the application pulls data from BMRS, fits a model, then generates the forecast, so may take a minute to update when modified.

                   
<p align="center">
<img src = readme_images/DashAppView.png width="800">
</p>         

From here, you can zoom into a single week to see the model's forecast.
Try playing around with the training start date - sometimes time series models perform better with longer histories, sometimes shorter, depending on how quickly the generation profile in the UK is changing.


## Section 4 - Share Results

### Lab 4.1 - Share Web App and Model API

Congratulations! You have now gone through a full workflow to pull data from an S3 bucket, clean and visualize the data, run jobs, schedule jobs, and deploy a web app front end for visualizing a forecast. Now the final step is to get your model and front end into the hands of the end users.

To do so we will navigate back to our project and click on the **App** tab.

From the App page navigate to the **Permissions** tab.

In the permissions tab update the permissions to allow anyone, including anonymous users.

<p align="center">
<img src = readme_images/GoToAppPermissions.png width="800">
</p>         
       
Navigate back to the **settings** tab and click **Copy Link App**

<p align="center">
<img src = readme_images/CopyAppLink.png width="800">
</p>       

Paste the copied link into a new private/incognito window. Note that you're able to view the app without being logged into Domino. Try sending this URL to a colleague at your company to show them the work you've done.

Domino provides free licenses for business users to login and view models/apps etc - 

### *** End of Labs *** 