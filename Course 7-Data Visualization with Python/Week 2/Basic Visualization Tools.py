#
import numpy as np  # useful for many scientific computing in Python
import pandas as pd # primary data structure library

df_can = pd.read_excel('https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DV0101EN/labs/Data_Files/Canada.xlsx',
                       sheet_name='Canada by Citizenship',
                       skiprows=range(20),
                       skipfooter=2
                      )


#Clean up the dataset to remove columns that are not informative to us for visualization (eg. Type, AREA, REG).
df_can.drop(['AREA', 'REG', 'DEV', 'Type', 'Coverage'], axis=1, inplace=True)

#Rename some of the columns so that they make sense.
df_can.rename(columns={'OdName':'Country', 'AreaName':'Continent','RegName':'Region'}, inplace=True)

#converts all column name into string types and then checks to ensure they are strings
df_can.columns = list(map(str, df_can.columns))
all(isinstance(column, str) for column in df_can.columns)

#4. Set the country name as index - useful for quickly looking up countries using .loc method.
df_can.set_index('Country', inplace=True)

#5. Add total column.
df_can['Total'] = df_can.sum(axis=1)


# create a list of years from 1980 - 2013
# this will come in handy when we start plotting the data
years = list(map(str, range(1980, 2014)))












# use the inline backend to generate the plots within the browser
#%matplotlib inline 

import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.style.use('ggplot') # optional: for ggplot-like style










#Area Plots

df_can.sort_values(['Total'], ascending=False, axis=0, inplace=True)
df_top5 = df_can.head() # get the top 5 entries
df_top5 = df_top5[years].transpose()  # transpose the dataframe

#Area plots are stacked by default.
#And to produce a stacked area plot, 
#each column must be either all positive or all negative values (any NaN values will defaulted to 0).
#To produce an unstacked plot, pass stacked=False.

df_top5.index = df_top5.index.map(int) # let's change the index values of df_top5 to type integer for plotting
df_top5.plot(kind='area', 
             alpha=0.25, # 0-1, default value a= 0.5
             stacked=False,
             figsize=(20, 10), # pass a tuple (x, y) size
             )

plt.title('Immigration Trend of Top 5 Countries')
plt.ylabel('Number of Immigrants')
plt.xlabel('Years')

plt.show()





#Two types of plotting
#*Option 1: Scripting layer (procedural method) - using matplotlib.pyplot as 'plt' *

#You can use plt i.e. matplotlib.pyplot and add more elements by calling different methods procedurally;
#for example, plt.title(...) to add title or plt.xlabel(...) to add label to the x-axis.

# Option 1: This is what we have been using so far
#df_top5.plot(kind='area', alpha=0.35, figsize=(20, 10)) 
#plt.title('Immigration trend of top 5 countries')
#plt.ylabel('Number of immigrants')
#plt.xlabel('Years')

#*Option 2: Artist layer (Object oriented method) - using an Axes instance from Matplotlib (preferred) *
#You can use an Axes instance of your current plot and store it in a variable (eg. ax). 
#You can add more elements by calling methods with a little change in syntax 
#(by adding "set_" to the previous methods). 
#For example, use ax.set_title() instead of plt.title() to add title, 
#or ax.set_xlabel() instead of plt.xlabel() to add label to the x-axis.

# option 2: preferred option with more flexibility
ax = df_top5.plot(kind='area', alpha=0.35, figsize=(20, 10))

ax.set_title('Immigration Trend of Top 5 Countries')
ax.set_ylabel('Number of Immigrants')
ax.set_xlabel('Years')

#Question: Use the scripting layer to create a stacked area plot of the 5 countries 
#that contributed the least to immigration to Canada from 1980 to 2013. Use a transparency value of 0.45.

df_bottom5 = df_can.tail() # get the top 5 entries
df_bottom5 = df_bottom5[years].transpose()

df_bottom5.index = df_bottom5.index.map(int)
df_bottom5.plot(kind='area', 
             alpha=0.45, # 0-1, default value a= 0.5
             stacked=True,
             figsize=(20, 10), # pass a tuple (x, y) size
             )

plt.title('Immigration Trend of Bottom 5 Countries')
plt.ylabel('Number of Immigrants')
plt.xlabel('Years')

plt.show()

#Question: Use the artist layer to create an unstacked area plot of the 5 countries that contributed 
#the least to immigration to Canada from 1980 to 2013. Use a transparency value of 0.55.

df_bottom5 = df_can.tail() # get the top 5 entries
df_bottom5 = df_bottom5[years].transpose()

df_bottom5.index = df_bottom5.index.map(int)
ax = df_bottom5.plot(kind='area', 
             alpha=0.55, 
             stacked=False,
             figsize=(20, 10), # pass a tuple (x, y) size
             )

ax.set_title('Immigration Trend of Bottom 5 Countries')
ax.set_ylabel('Number of Immigrants')
ax.set_xlabel('Years')















#Histograms

# np.histogram returns 2 values
count, bin_edges = np.histogram(df_can['2013'])

print(count) # frequency count

#[178  11   1   2   0   0   0   0   1   2]

print(bin_edges) # bin ranges, default = 10 bins

#[    0.   3412.9  6825.8 10238.7 13651.6 17064.5 20477.4 23890.3 27303.2
# 30716.1 34129. ]

#178 countries contributed between 0 to 3412.9 immigrants
#11 countries contributed between 3412.9 to 6825.8 immigrants
#1 country contributed between 6285.8 to 10238.7 immigrants, and so on..

#df_can['2013'].plot(kind='hist', figsize=(8, 5))
df_can['2013'].plot(kind='hist', figsize=(8, 5), xticks=bin_edges)
#xticks=bin_edges changes the x-axis so that labels are based on bin_edges rather default values


plt.title('Histogram of Immigration from 195 Countries in 2013') # add a title to the histogram
plt.ylabel('Number of Countries') # add y-label
plt.xlabel('Number of Immigrants') # add x-label

plt.show()






#Question: What is the immigration distribution for Denmark, Norway, and Sweden for years 1980 - 2013?

# let's quickly view the dataset 
df_can.loc[['Denmark', 'Norway', 'Sweden'], years].plot.hist() 

#@@@@@@@@doesn't look right@@@@@@@@@@@

# transpose dataframe
df_t = df_can.loc[['Denmark', 'Norway', 'Sweden'], years].transpose()

# generate histogram
df_t.plot(kind='hist', figsize=(10, 6))

plt.title('Histogram of Immigration from Denmark, Norway, and Sweden from 1980 - 2013')
plt.ylabel('Number of Years')
plt.xlabel('Number of Immigrants')

plt.show()



#Let's make a few modifications to improve the impact and aesthetics of the previous plot:

#---increase the bin size to 15 by passing in bins parameter
#---set transparency to 60% by passing in alpha paramemter
#---label the x-axis by passing in x-label paramater
#---change the colors of the plots by passing in color parameter

# let's get the x-tick values
count, bin_edges = np.histogram(df_t, 15)

# un-stacked histogram
df_t.plot(kind ='hist', 
          figsize=(10, 6),
          bins=15,
          alpha=0.6,
          xticks=bin_edges,
          color=['coral', 'darkslateblue', 'mediumseagreen']
         )

plt.title('Histogram of Immigration from Denmark, Norway, and Sweden from 1980 - 2013')
plt.ylabel('Number of Years')
plt.xlabel('Number of Immigrants')

plt.show()

#If we dot no want the plots to overlap each other, we can stack them using the stacked paramemter.
#Let's also adjust the min and max x-axis labels to remove the extra gap on the edges of the plot. 
#We can pass a tuple (min,max) using the xlim paramater, as show below.

count, bin_edges = np.histogram(df_t, 15)
xmin = bin_edges[0] - 10   #  first bin value is 31.0, adding buffer of 10 for aesthetic purposes 
xmax = bin_edges[-1] + 10  #  last bin value is 308.0, adding buffer of 10 for aesthetic purposes

# stacked Histogram
df_t.plot(kind='hist',
          figsize=(10, 6), 
          bins=15,
          xticks=bin_edges,
          color=['coral', 'darkslateblue', 'mediumseagreen'],
          stacked=True,
          xlim=(xmin, xmax)
         )

plt.title('Histogram of Immigration from Denmark, Norway, and Sweden from 1980 - 2013')
plt.ylabel('Number of Years')
plt.xlabel('Number of Immigrants') 

plt.show()


#Question: Use the scripting layer to display the immigration distribution for
#Greece, Albania, and Bulgaria for years 1980 - 2013? 
#Use an overlapping plot with 15 bins and a transparency value of 0.35.

df_g_a_b_transpose = df_can.loc[['Greece', 'Albania', 'Bulgaria'], years].transpose()

count, bin_edges = np.histogram(df_g_a_b_transpose, 15)

df_g_a_b_transpose.plot(kind = 'hist', 
          figsize=(10, 6),
          bins=15,
          alpha=0.35,
          xticks=bin_edges,
          color=['coral', 'darkslateblue', 'mediumseagreen']
         )

plt.title('Histogram of Immigration from Greece, Albanoa, and Bulgaria from 1980 - 2013')
plt.ylabel('Number of Years')
plt.xlabel('Number of Immigrants')

plt.show()














#Bar Charts (Dataframe)

#A bar plot is a way of representing data where the length of the bars represents the magnitude/size of the feature/variable. 
#Bar graphs usually represent numerical and categorical variables grouped in intervals.

#Question: Let's compare the number of Icelandic immigrants (country = 'Iceland') to Canada from year 1980 to 2013.
# step 1: get the data
df_iceland = df_can.loc['Iceland', years]
# step 2: plot data
df_iceland.plot(kind='bar', figsize=(10, 6))

plt.xlabel('Year') # add to x-label to the plot
plt.ylabel('Number of immigrants') # add y-label to the plot
plt.title('Icelandic immigrants to Canada from 1980 to 2013') # add title to the plot

# Creates an arrow on the graph
plt.annotate('',                      # s: str. Will leave it blank for no text
             xy=(32, 70),             # place head of the arrow at point (year 2012 , pop 70)
             xytext=(28, 20),         # place base of the arrow at point (year 2008 , pop 20)
             xycoords='data',         # will use the coordinate system of the object being annotated 
             arrowprops=dict(arrowstyle='->', connectionstyle='arc3', color='blue', lw=2)
            )

# Creates a text on the graph
plt.annotate('2008 - 2011 Financial Crisis', # text to display
             xy=(28, 30),                    # start the text at at point (year 2008 , pop 30)
             rotation=72.5,                  # based on trial and error to match the arrow
             va='bottom',                    # want the text to be vertically 'bottom' aligned
             ha='left',                      # want the text to be horizontally 'left' algned.
            )


plt.show()









#Question: Using the scripting layter and the df_can dataset, 
#create a horizontal bar plot showing the total number of immigrants
#to Canada from the top 15 countries, for the period 1980 - 2013.
#Label each country with the total immigrant count.

#Step 1: Get the data pertaining to the top 15 countries.
df_top_15 = df_can["Total"].head(15)
df_top_15.plot(kind='barh', figsize=(12, 12), color='steelblue')
plt.xlabel('Number of Immigrants')
plt.title('Top 15 Conuntries Contributing to the Immigration to Canada between 1980 - 2013')

for index, value in enumerate(df_top15):     
    plt.annotate(str(value), xy=(value, index - 0.10))
plt.show()