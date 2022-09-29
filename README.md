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
df['TOTAL'] = df.sum(axis=1, numeric_only=True)

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
fig.savefig('Cumulative Production.png', bbox_inches="tight")

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
print(daily_peak_df[idx].shape)

# Print a table with the daily Maximum values, and time when demand peaked
daily_peak_df = daily_peak_df[idx]

daily_peak_df.head()
```
We can also visualize the hours of the day when demand peaks, and save our plot.

```python
import seaborn as sns

plt.figure(figsize=(8,5))
plt.title('Hours in the day When Demand Peaks in the UK, Summer 2022')
plt.xlabel('Hour of the Day')
sns.histplot(daily_peak_df.index.hour, stat='count', bins=10)
plt.savefig('Peak Demand Hours.png', bbox_inches="tight")
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

### Lab 2.3 - Syncing Files

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


### Lab 2.4 - Run and Track Jobs

Workspaces are great environments for doing exploratory work and writing code. However, once our code is finished, we may want to run it regularly- which would be tedious if we have to spin up a workspace each time.

To simply run our code in our predefined environment and quickly visualize outputs, Domino has a feature called Jobs. Jobs spin up an instance, run a script, save outputs, and shut down the instance for us.

<p align="center">
<img src = readme_images/NavigateToJobs.png width="800">
</p>


In this example, we want to pull some more recent September data from BMRS and save it to the Domino File System.

Type in the following command below in the **File Name** section of the **Start a Job** pop up window. Click on **Start** to run the job.

```shell
scripts/pull_data.py '--start=2022-09-01 00:00:00' '--end=2022-09-27 00:00:00'
```

<p align="center">
<img src = readme_images/Jobsrun.png width="800">
</p>

Click into the pull_data.py job run.

In the details tab of the job run note that the compute environment and hardware tier are tracked to document not only who ran the experiment and when, but what versions of the code, software, and hardware were executed.

In the details tab of the job run note that the compute environment is tracked to document not only who ran the experiment and when, but what versions of the code, software, and hardware were executed.

Now, click into Domino Datasets and examine the contents in “Power Generation Workshop”. September data should be there. 

<p align="center">
<img src = readme_images/UpdatedDataJob.png width="800">
</p>

### Lab 2.5 - Schedule Jobs

Say we wanted to pull the data each month. Rather than running this job manually, we can schedule the job to run in Domino.

The script we ran manually, pull_data.py, defaults to pulling the past 30 days if we don't pass it an start and end date, so we can simply schedule it to run each month.

Navigate to “Scheduled Jobs” under “Publish”, and select, “New Scheduled Job”

<p align="center">
<img src = readme_images/NewScheduleJob.png width="800">
</p>

Paste the following into the command, give the job the name "Monthly Data Pull", and verify that the Environment is "Domino-PowerGeneration-Workshop-Environment". It should be, since we set it as our project environemnt, but it's always good to check. Click “Next”. 

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

### Lab 2.6 - Create a Launcher

# TBD


## Section 3 - Create Applications


### Lab 3.2 Deploying Web App
    
Now that we have a pod running to serve new model requests - we will build out a front end to make calling our model easier for end-users.
    
To do so - in a new browser tab first navigate back to your Project and then in the left blue menu of your project click into the **Files** section and click **New File**
<p align="center">
<img src = readme_images/AddNewFileforAppsh.png width="800">
</p>     

Next, we will create a file called app.sh. It's a bash script that will start and run the Shiny App server based on the inputs provided.
Copy the following code snippet in - 

```shell
#!/usr/bin/env bash
 
# This is a bash script for Domino's App publishing feature
# Learn more at http://support.dominodatalab.com/hc/en-us/articles/209150326
 
## R/Shiny Example
## This is an example of the code you would need in this bash script for a R/Shiny app
R -e 'shiny::runApp("./scripts/shiny_app.R", port=8888, host="0.0.0.0")'
 
## Flask example
## This is an example of the code you would need in this bash script for a Python/Flask app
#export LC_ALL=C.UTF-8
#export LANG=C.UTF-8
#export FLASK_APP=app-flask.py
#export FLASK_DEBUG=1
#python -m flask run --host=0.0.0.0 --port=8888
 
## Dash Example
## This is an example of the code you would need in this bash script for a Dash app
#python app-dash.py
```
Name the file **app.sh** and click **Save**
<p align="center">
<img src = readme_images/appsh.png width="800">
</p>         


Now navigate back into the Files tab, and enter the **scripts** folder. Click add a new file and name it `shiny_app.R` (make sure the file name is exactly that, it is case sensitive) and then paste the following into the file -

```R
#
# This is a Shiny web application. You can run the application by clicking
# the 'Run App' button above.
#
# Find out more about building applications with Shiny here:
#
#    http://shiny.rstudio.com/
#
 
install.packages("png")
 
library(shiny)
library(png)
library(httr)
library(jsonlite)
library(plotly)
library(ggplot2)
 
 
# Define UI for application that draws a histogram
ui <- fluidPage(
  
  # Application title
  titlePanel("Wine Quality Prediction"),
  
  # Sidebar with a slider input for number of bins 
  sidebarLayout(
    sidebarPanel(
      numericInput(inputId="feat1",
                   label='density', 
                   value=0.99),
      numericInput(inputId="feat2",
                   label='volatile_acidity', 
                   value=0.25),
      numericInput(inputId="feat3",
                   label='chlorides', 
                   value=0.05),
      numericInput(inputId="feat4",
                   label='is_red', 
                   value=1),
      numericInput(inputId="feat5",
                   label='alcohol', 
                   value=10),
      actionButton("predict", "Predict")
    ),
    
    # Show a plot of the generated distribution
    mainPanel(
      tabsetPanel(id = "inTabset", type = "tabs",
                  
                  tabPanel(title="Prediction",value = "pnlPredict",
                           plotlyOutput("plot"),
                           verbatimTextOutput("summary"),
                           verbatimTextOutput("version"),
                           verbatimTextOutput("reponsetime"))
      )        
    )
  )
)
 
prediction <- function(inpFeat1,inpFeat2,inpFeat3,inpFeat4,inpFeat5) {
  
#### COPY FULL LINES 4-7 from R tab in Model APIS page over this line of code. (It's a simple copy and paste) ####
    
    body=toJSON(list(data=list(density = inpFeat1, 
                               volatile_acidity = inpFeat2,
                               chlorides = inpFeat3,
                               is_red = inpFeat4,
                               alcohol = inpFeat5)), auto_unbox = TRUE),
    content_type("application/json")
  )
  
  str(content(response))
  
  result <- content(response)
}
 
gauge <- function(pos,breaks=c(0,2.5,5,7.5, 10)) {
 
  get.poly <- function(a,b,r1=0.5,r2=1.0) {
    th.start <- pi*(1-a/10)
    th.end   <- pi*(1-b/10)
    th       <- seq(th.start,th.end,length=10)
    x        <- c(r1*cos(th),rev(r2*cos(th)))
    y        <- c(r1*sin(th),rev(r2*sin(th)))
    return(data.frame(x,y))
  }
  ggplot()+
    geom_polygon(data=get.poly(breaks[1],breaks[2]),aes(x,y),fill="red")+
    geom_polygon(data=get.poly(breaks[2],breaks[3]),aes(x,y),fill="gold")+
    geom_polygon(data=get.poly(breaks[3],breaks[4]),aes(x,y),fill="orange")+
    geom_polygon(data=get.poly(breaks[4],breaks[5]),aes(x,y),fill="forestgreen")+
    geom_polygon(data=get.poly(pos-0.2,pos+0.2,0.2),aes(x,y))+
    geom_text(data=as.data.frame(breaks), size=5, fontface="bold", vjust=0,
              aes(x=1.1*cos(pi*(1-breaks/10)),y=1.1*sin(pi*(1-breaks/10)),label=paste0(breaks)))+
    annotate("text",x=0,y=0,label=paste0(pos, " Points"),vjust=0,size=8,fontface="bold")+
    coord_fixed()+
    theme_bw()+
    theme(axis.text=element_blank(),
          axis.title=element_blank(),
          axis.ticks=element_blank(),
          panel.grid=element_blank(),
          panel.border=element_blank())
}
 
# Define server logic required to draw a histogram
server <- function(input, output,session) {
  
  observeEvent(input$predict, {
    updateTabsetPanel(session, "inTabset",
                      selected = paste0("pnlPredict", input$controller)
    )
    print(input)
    result <- prediction(input$feat1, input$feat2, input$feat3, input$feat4, input$feat5)
    print(result)
    
    pred <- result$result[[1]][[1]]
    modelVersion <- result$release$model_version_number
    responseTime <- result$model_time_in_ms
    output$summary <- renderText({paste0("Wine Quality estimate is ", round(pred,2))})
    output$version <- renderText({paste0("Model version used for scoring : ", modelVersion)})
    output$reponsetime <- renderText({paste0("Model response time : ", responseTime, " ms")})
    output$plot <- renderPlotly({
      gauge(round(pred,2))
    })
  })
  
}
 
# Run the application 
shinyApp(ui = ui, server = server)
```

**Go to line 63** note that this is missing input for your model api endpoint. In a new tab navigate to your model API you just deployed. Go into overview and select the R tab as shown below. Copy lines 4-7 from the R code snippet. Switch back to your new file tab and paste the new lines in line 64 in your file.

<p align="center">
<img src = readme_images/RcodeSnippet.png width="800">
</p>                    
Lines 61-79 in your file should look like the following (note the url and authenticate values will be different) 
                   
<p align="center">
<img src = readme_images/ShinyCodePasted.png width="800">
</p>         

Click **Save**
                   
Now that you have your app.sh and shiny_app.R files created. Navigate to the **App** tab in your project

Enter a title for your app - 'wine-app-yourname'

<p align="center">
<img src = readme_images/LaunchApp.png width="800">
</p>       

Click Publish.
                   
You'll now see the below screen, once your app is active (should be within ~1-3 minutes) you can click the View App button. 

<p align="center">
<img src = readme_images/ViewApp.png width="800">
</p>       
        
Once you're in the app you can try out sending different scoring requests to your model using the form on the right side of your page. Click **predict** to send a scoring request and view the results in the visualization on the left side.
                   
<p align="center">
<img src = readme_images/ShinyScore.png width="800">
</p>         

## Section 4 - Collaborate Results

### Lab 4.1 - Share Web App and Model API

Congratulations! You have now gone through a full workflow to pull data from an S3 bucket, clean and visualize the data, train several models across different frameworks, deploy the best performing model, and use a web app front end for easy scoring of your model. Now the final step is to get your model and front end into the hands of the end users.

To do so we will navigate back to our project and click on the **App** tab

<p align="center">
<img src = readme_images/GoToAppPermissions.png width="800">
</p>         


From the App page navigate to the **Permissions** tab

In the permissions tab update the permissions to allow anyone, including anonymous users

<p align="center">
<img src = readme_images/UpdateAppPermissions.png width="800">
</p>         

Navigate back to the **settings** tab and click **Copy Link App**

<p align="center">
<img src = readme_images/CopyAppLink.png width="800">
</p>       

Paste the copied link into a new private/incognito window. Note that you're able to view the app without being logged into Domino. Try sending this URL to a colleague at your company to show them the work you've done.

PS - Domino provides free licenses for business users to login and view models/apps etc.

### *** End of Labs *** 

So now that we've got our model into production are we done? No! We want to make sure that any models we deploy stay healthy over time, and if our models do drop in performance, we want to quickly identify and remediate any issues. Stay tuned for a demo of integrated model monitoring to see how a ML Engineer would automate the model monitoring process and make remediation a breeze.
