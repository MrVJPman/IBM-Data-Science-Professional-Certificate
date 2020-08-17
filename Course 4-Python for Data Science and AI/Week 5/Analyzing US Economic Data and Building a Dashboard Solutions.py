import pandas as pd
from bokeh.plotting import figure, output_file, show,output_notebook
output_notebook()

def make_dashboard(x, gdp_change, unemployment, title, file_name):
    output_file(file_name)
    p = figure(title=title, x_axis_label='year', y_axis_label='%')
    p.line(x.squeeze(), gdp_change.squeeze(), color="firebrick", line_width=4, legend="% GDP change")
    p.line(x.squeeze(), unemployment.squeeze(), line_width=4, legend="% unemployed")
    show(p)
    
    
links={'GDP':'https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/PY0101EN/projects/coursera_project/clean_gdp.csv',\
       'unemployment':'https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/PY0101EN/projects/coursera_project/clean_unemployment.csv'}


#Question 1: Create a dataframe that contains the GDP data and display the first five rows of the dataframe.
# Type your code here
GDP_dataframe = pd.read_csv(links["GDP"]) 
GDP_dataframe.head()


#Question 2: Create a dataframe that contains the unemployment data. Display the first five rows of the dataframe.
# Type your code here

unemployment_dataframe = pd.read_csv(links["unemployment"]) 
unemployment_dataframe.head() 

#Question 3: Display a dataframe where unemployment was greater than 8.5%.
# Type your code here
unemployment_dataframe[unemployment_dataframe.unemployment > 8.5]

#Question 4: Use the function make_dashboard to make a dashboard

#Create a new dataframe with the column 'date' called x from the dataframe that contains the GDP data.
x = GDP_dataframe[["date"]] # Create your dataframe with column date

#Create a new dataframe with the column 'change-current'  called gdp_change from the dataframe that contains the GDP data.
gdp_change = GDP_dataframe[["change-current"]] 

#Create a new dataframe with the column 'unemployment'  called unemployment from the dataframe that contains the unemployment data.
unemployment = unemployment_dataframe[["unemployment"]] 

#Give your dashboard a string title, and assign it to the variable title
title = "Python for Data Science and AI > Week 5 > Analyzing US Economic Data and Building a Dashboard"

#Finally, the function make_dashboard will output an .html in your direictory, just like a csv file.
#The name of the file is "index.html" and it will be stored in the varable file_name.

file_name = "index.html"

#Call the function make_dashboard , to produce a dashboard. 
#Assign the parameter values accordingly take a the , take a screen shot of the dashboard and submit it.
make_dashboard(x=GDP_dataframe[["date"]], gdp_change=GDP_dataframe[["change-current"]] , 
               unemployment=unemployment_dataframe[["unemployment"]], 
               title="Python for Data Science and AI > Week 5 > Analyzing US Economic Data and Building a Dashboard", 
               file_name="index.html")
