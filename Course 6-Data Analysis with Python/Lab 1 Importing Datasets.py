#Lab 1
# Import pandas library
import pandas as pd

# Read the online file by the URL provides above, and assign it to variable "df"
other_path = "https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DA0101EN/auto.csv"
df = pd.read_csv(other_path, header=None)

# show the first 5 rows using dataframe.head() method
print("The first 5 rows of the dataframe") 
df.head(5)

#Question #1:
#check the bottom 10 rows of data frame "df".
# Write your code below and press Shift+Enter to execute 
print("The first 10 rows of the dataframe") 
df.tail(10)


# create headers list
headers = ["symboling","normalized-losses","make","fuel-type","aspiration", "num-of-doors","body-style",
         "drive-wheels","engine-location","wheel-base", "length","width","height","curb-weight","engine-type",
         "num-of-cylinders", "engine-size","fuel-system","bore","stroke","compression-ratio","horsepower",
         "peak-rpm","city-mpg","highway-mpg","price"]
print("headers\n", headers)

df.columns = headers
df.head(10)

df.dropna(subset=["price"], axis=0)

#Question #2:
#Find the name of the columns of the dataframe
print(df.columns)

#orrespondingly, Pandas enables us to save the dataset to csv by using the dataframe.to_csv() method,
#you can add the file path and name along with quotation marks in the brackets.
df.to_csv("automobile.csv", index=False)

#Data Types
#Data has a variety of types.
#The main types stored in Pandas dataframes are object, float, int, bool and datetime64. 
#In order to better learn about each attribute, it is always good for us to know the data type of each column. In Pandas:
df.dtypes

#returns a Series with the data type of each column.

# check the data type of data frame "df" by .dtypes
print(df.dtypes)

#If we would like to get a statistical summary of each column,
#such as count, column mean value, column standard deviation, etc. We use the describe method:

df.describe()

#This shows the statistical summary of all numeric-typed (int, float) columns.
#For example, the attribute "symboling" has 205 counts, the mean value of this column is 0.83, 
#the standard deviation is 1.25, the minimum value is -2, 25th percentile is 0, 50th percentile is 1,
#75th percentile is 2, and the maximum value is 3.
#However, what if we would also like to check all the columns including those that are of type object.

#You can add an argument include = "all" inside the bracket. Let's try it again.
# describe all the columns in "df" 
df.describe(include = "all")


#Question #3:
#You can select the columns of a data frame by indicating the name of each column, for example, you can select the three columns as follows:
#dataframe[[' column 1 ',column 2', 'column 3']]
#Where "column" is the name of the column, you can apply the method ".describe()" to get the statistics of those columns as follows:
#dataframe[[' column 1 ',column 2', 'column 3'] ].describe()
#Apply the method to ".describe()" to the columns 'length' and 'compression-ratio'.

df[["length", "compression-ratio"]].describe()

# look at the info of "df"
df.info

