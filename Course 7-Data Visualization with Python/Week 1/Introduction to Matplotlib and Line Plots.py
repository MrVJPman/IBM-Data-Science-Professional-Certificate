import numpy as np  # useful for many scientific computing in Python
import pandas as pd # primary data structure library


df_can = pd.read_excel('https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DV0101EN/labs/Data_Files/Canada.xlsx',
                       sheet_name='Canada by Citizenship',
                       skiprows=range(20),
                       skipfooter=2)

print ('Data read into a pandas dataframe!')

#The top 5 rows of the dataset using the head() function.
df_can.head()
# tip: You can specify the number of rows you'd like to see as follows: df_can.head(10) 

#View the bottom 5 rows of the dataset using the tail() function.
df_can.tail()

#When analyzing a dataset, it's always a good idea to start by getting basic information about your dataframe.
#We can do this by using the info() method.
df_can.info()

#To get the list of column headers we can call upon the dataframe's .columns parameter.
df_can.columns.values 

#Similarly, to get the list of indicies we use the .index parameter.
df_can.index.values

#Note: The default type of index and columns is NOT list.
print(type(df_can.columns))
print(type(df_can.index))

#To get the index and columns as lists, we can use the tolist() method.
df_can.columns.tolist()
df_can.index.tolist()

print (type(df_can.columns.tolist()))
print (type(df_can.index.tolist()))

# size of dataframe (rows, columns)
print(df_can.shape)

#Let's clean the data set to remove a few unnecessary columns. We can use pandas drop() method as follows:
# in pandas axis=0 represents rows (default) and axis=1 represents columns.
df_can.drop(['AREA','REG','DEV','Type','Coverage'], axis=1, inplace=True)
df_can.head(2) #--> 2 x 38

#Let's rename the columns so that they make sense.
#We can use rename() method by passing in a dictionary of old and new names as follows:

df_can.rename(columns={'OdName':'Country', 'AreaName':'Continent', 'RegName':'Region'}, inplace=True)
df_can.columns

#We will also add a 'Total' column that sums up the total immigrants by country over the entire period 1980 - 2013, as follows:
df_can['Total'] = df_can.sum(axis=1)

#We can check to see how many null objects we have in the dataset as follows:
df_can.isnull().sum()

#Finally, let's view a quick summary of each column in our dataframe using the describe() method.
df_can.describe()










#Method 1: Quick and easy, but only works if the column name does NOT have spaces or special characters.
df.column_name #    (returns series)
#Method 2: More robust, and can filter on multiple columns.
df['column']  #       (returns series)
df[['column 1', 'column 2']] #       (returns dataframe)    


#pandas Intermediate: Indexing and Selection (slicing)
df_can.Country  # returns a series

df_can[['Country', 1980, 1981, 1982, 1983, 1984, 1985]] # returns a dataframe
# notice that 'Country' is string, and the years are integers. 
# for the sake of consistency, we will convert all column names to string later on.




df.loc[label] #filters by the labels of the index/column
df.iloc[index]  #filters by the positions of the index/column


#Before we proceed, notice that the defaul index of the dataset is a numeric range from 0 to 194. 
#This makes it very difficult to do a query by a specific country. For example to search for data on Japan,
#we need to know the corressponding index value.
#This can be fixed very easily by setting the 'Country' column as the index using set_index() method.
df_can.set_index('Country', inplace=True)
# tip: The opposite of set is reset. So to reset the index, we can use df_can.reset_index()

# optional: to remove the name of the index
df_can.index.name = None


# 1. the full row data (all columns)
print(df_can.loc['Japan'])
# alternate methods
print(df_can.iloc[87])
print(df_can[df_can.index == 'Japan'].T.squeeze())

# 2. for year 2013
print(df_can.loc['Japan', 2013])
# alternate method
print(df_can.iloc[87, 36]) # year 2013 is the last column, with a positional index of 36

# 3. for years 1980 to 1985
print(df_can.loc['Japan', [1980, 1981, 1982, 1983, 1984, 1984]])
print(df_can.iloc[87, [3, 4, 5, 6, 7, 8]])

#Column names that are integers (such as the years) might introduce some confusion. 
#For example, when we are referencing the year 2013, one might confuse that when the 2013th positional index.
#To avoid this ambuigity, let's convert the column names into strings: '1980' to '2013'.

df_can.columns = list(map(str, df_can.columns))
# [print (type(x)) for x in df_can.columns.values] #<-- uncomment to check type of column headers

# useful for plotting later on
years = list(map(str, range(1980, 2014)))





#Filtering based on a criteria
#To filter the dataframe based on a condition, we simply pass the condition as a boolean vector.

#For example, Let's filter the dataframe to show the data on Asian countries (AreaName = Asia).

# 1. create the condition boolean series
condition = df_can['Continent'] == 'Asia'
print(condition)

# 2. pass this condition into the dataFrame
df_can[condition] #

# we can pass mutliple criteria in the same line. 
# let's filter for AreaNAme = Asia and RegName = Southern Asia

df_can[(df_can['Continent']=='Asia') & (df_can['Region']=='Southern Asia')]

# note: When using 'and' and 'or' operators, pandas requires we use '&' and '|' instead of 'and' and 'or'
# don't forget to enclose the two conditions in parentheses

#Before we proceed: let's review the changes we have made to our dataframe.
print('data dimensions:', df_can.shape)
print(df_can.columns)
df_can.head(2)









#Visualizing Data using Matplotlib

# we are using the inline backend
#%matplotlib inline 

import matplotlib as mpl
import matplotlib.pyplot as plt
print ('Matplotlib version: ', mpl.__version__) # >= 2.0.0
print(plt.style.available)
mpl.style.use(['ggplot']) # optional: for ggplot-like style



#Line Pots (Series/Dataframe)
haiti = df_can.loc['Haiti', years] # passing in years 1980 - 2013 to exclude the 'total' column
haiti.head()
haiti.plot()




#Also, let's label the x and y axis using plt.title(), plt.ylabel(), and plt.xlabel() as follows:
haiti.index = haiti.index.map(int) # let's change the index values of Haiti to type integer for plotting
haiti.plot(kind='line')
plt.title('Immigration from Haiti')
plt.ylabel('Number of immigrants')
plt.xlabel('Years')
plt.show() # need this line to show the updates made to the figure


#We can clearly notice how number of immigrants from Haiti spiked up 
#from 2010 as Canada stepped up its efforts to accept refugees from Haiti. 
#Let's annotate this spike in the plot by using the plt.text() method.
haiti.plot(kind='line')
plt.title('Immigration from Haiti')
plt.ylabel('Number of Immigrants')
plt.xlabel('Years')
# annotate the 2010 Earthquake. 
# syntax: plt.text(x, y, label)
plt.text(2000, 6000, '2010 Earthquake') # see note below
plt.show() 







#Question: Let's compare the number of immigrants from India and China from 1980 to 2013.

#Step 1: Get the data set for China and India, and display dataframe.
china_india_dataframe = df_can.loc[["China", "India"], years]

#Step 2: Plot graph. We will explicitly specify line plot by passing in kind parameter to plot().
china_india_dataframe.plot(kind = "line")

china_india_dataframe = china_india_dataframe.transpose()
china_india_dataframe.head()

#pandas will auomatically graph the two countries on the same graph. 
#Go ahead and plot the new transposed dataframe. Make sure to add a title to the plot and label the axes.

china_india_dataframe.index = china_india_dataframe.index.map(int)
china_india_dataframe.plot(kind = "line")

plt.title('Immigration from China and India')
plt.ylabel('Number of Immigrants')
plt.xlabel('Years')

plt.show() 







#Question: Compare the trend of top 5 countries that contributed the most to immigration to Canada.
df_can[df_can["Total"] > 241500]
top_5_countries_data_frame = df_can.loc[["China", "India", "Pakistan", "Philippines", 
                                         "United Kingdom of Great Britain and Northern Ireland"], years]
top_5_countries_data_frame = top_5_countries_data_frame.transpose()

top_5_countries_data_frame.index = top_5_countries_data_frame.index.map(int) #changes string version of years into the integer version
top_5_countries_data_frame.plot(kind='line', figsize=(14, 8))

plt.title('Immigrants from top 5 countries that contributed the most to immigration to Canada')
plt.ylabel('Number of Immigrants')
plt.xlabel('Years')

plt.show()



#Solution from online

#df_can.sort_values(by='Total', ascending=False, axis=0, inplace=True)
#df_top5 = df_can.head(5)
#df_top5 = df_top5[years].transpose() 
#df_top5.index = df_top5.index.map(int) # let's change the index values of df_top5 to type integer for plotting
#df_top5.plot(kind='line', figsize=(14, 8)) # pass a tuple (x, y) size

#plt.title('Immigration Trend of Top 5 Countries')
#plt.ylabel('Number of Immigrants')
#plt.xlabel('Years')
#plt.show()
