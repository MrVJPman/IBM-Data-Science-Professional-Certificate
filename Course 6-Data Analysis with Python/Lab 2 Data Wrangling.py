#Lab 2

import pandas as pd
import matplotlib.pylab as plt

filename = "https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DA0101EN/auto.csv"
headers = ["symboling","normalized-losses","make","fuel-type","aspiration", "num-of-doors","body-style",
         "drive-wheels","engine-location","wheel-base", "length","width","height","curb-weight","engine-type",
         "num-of-cylinders", "engine-size","fuel-system","bore","stroke","compression-ratio","horsepower",
         "peak-rpm","city-mpg","highway-mpg","price"]
df = pd.read_csv(filename, names = headers)
# To see what the data set looks like, we'll use the head() method.
df.head()


import numpy as np

# replace "?" to NaN. "?" are missing values
df.replace("?", np.nan, inplace = True)
df.head(5)


#The missing values are converted to Python's default.
#We use Python's built-in functions to identify these missing values. 
#There are two methods to detect missing data:
#.isnull()
#.notnull()

missing_data = df.isnull()
missing_data.head(5)


#Using a for loop in Python, we can quickly figure out the number of missing values in each column. 
#As mentioned above, "True" represents a missing value, "False" means the value is present in the dataset. 
#In the body of the for loop the method ".value_counts()" counts the number of "True" values.
for column in missing_data.columns.values.tolist():
    print(column)
    print (missing_data[column].value_counts())
    print("")    
    

#Calculate the average of the column & Replace "NaN" by mean value in "normalized-losses" column
avg_norm_loss = df["normalized-losses"].astype("float").mean(axis=0)
print("Average of normalized-losses:", avg_norm_loss)    
df["normalized-losses"].replace(np.nan, avg_norm_loss, inplace=True)

#Calculate the mean value for 'bore' column & #Replace NaN by mean value
avg_bore=df['bore'].astype('float').mean(axis=0)
print("Average of bore:", avg_bore)
df["bore"].replace(np.nan, avg_bore, inplace=True)


#Question #1:
#According to the example above, replace NaN in "stroke" column by mean.
avg_stroke=df['stroke'].astype('float').mean(axis=0)
df["stroke"].replace(np.nan, avg_stroke, inplace=True)


avg_horsepower = df['horsepower'].astype('float').mean(axis=0)
print("Average horsepower:", avg_horsepower)
df['horsepower'].replace(np.nan, avg_horsepower, inplace=True)

avg_peakrpm=df['peak-rpm'].astype('float').mean(axis=0)
print("Average peak rpm:", avg_peakrpm)
df['peak-rpm'].replace(np.nan, avg_peakrpm, inplace=True)

#To see which values are present in a particular column, we can use the ".value_counts()" method:
df['num-of-doors'].value_counts()

#We can see that four doors are the most common type. 
#We can also use the ".idxmax()" method to calculate for us the most common type automatically:
df['num-of-doors'].value_counts().idxmax()

#replace the missing 'num-of-doors' values by the most frequent 
df["num-of-doors"].replace(np.nan, "four", inplace=True)

# simply drop whole row with NaN in "price" column
df.dropna(subset=["price"], axis=0, inplace=True)

# reset index, because we droped two rows
df.reset_index(drop=True, inplace=True)






#Lets list the data types for each column
df.dtypes


#Convert data types to proper format
df[["bore", "stroke"]] = df[["bore", "stroke"]].astype("float")
df[["normalized-losses"]] = df[["normalized-losses"]].astype("int")
df[["price"]] = df[["price"]].astype("float")
df[["peak-rpm"]] = df[["peak-rpm"]].astype("float")

# Convert mpg to L/100km by mathematical operation (235 divided by mpg)
df['city-L/100km'] = 235/df["city-mpg"]

#Question #2:
#According to the example above, transform mpg to L/100km in the column of "highway-mpg",
#and change the name of column to "highway-L/100km".
df["highway-mpg"] = 235/df["highway-mpg"]
df.rename(columns = {"highway-mpg":"highway-L/100km"} , inplace = True)



#Data Normalization
#Why normalization?

#Normalization is the process of transforming values of several variables into a similar range. 
#Typical normalizations include scaling the variable so the variable average is 0, 
#scaling the variable so the variance is 1, 
#or scaling variable so the variable values range from 0 to 1


# replace (original value) by (original value)/(maximum value)
df['length'] = df['length']/df['length'].max()
df['width'] = df['width']/df['width'].max()

#Questiont #3:
#According to the example above, normalize the column "height".
df["height"] = df["height"]/df["height"].max()








#Binning is a process of transforming continuous numerical variables into discrete categorical 'bins', for grouped analysis.
df["horsepower"]=df["horsepower"].astype(int, copy=True)

#%matplotlib inline
import matplotlib as plt
from matplotlib import pyplot
plt.pyplot.hist(df["horsepower"])

# set x/y labels and plot title
plt.pyplot.xlabel("horsepower")
plt.pyplot.ylabel("count")
plt.pyplot.title("horsepower bins")

bins = np.linspace(min(df["horsepower"]), max(df["horsepower"]), 4)

#We set group names:
group_names = ['Low', 'Medium', 'High']

#We apply the function "cut" the determine what each value of "df['horsepower']" belongs to.
df['horsepower-binned'] = pd.cut(df['horsepower'], bins, labels=group_names, include_lowest=True )

print(df["horsepower-binned"].value_counts())

#%matplotlib inline
import matplotlib as plt2
from matplotlib import pyplot
pyplot.bar(group_names, df["horsepower-binned"].value_counts())

# set x/y labels and plot title
plt2.pyplot.xlabel("horsepower")
plt2.pyplot.ylabel("count")
plt2.pyplot.title("horsepower bins")
#plt2.pyplot.show() <--Has an error i don't know how to solve


#%matplotlib inline
import matplotlib as plt
from matplotlib import pyplot

a = (0,1,2)

# draw historgram of attribute "horsepower" with bins = 3
plt.pyplot.hist(df["horsepower"], bins = 3)

# set x/y labels and plot title
plt.pyplot.xlabel("horsepower")
plt.pyplot.ylabel("count")
plt.pyplot.title("horsepower bins")
#plt.pyplot.show()


#Indicator variable (or dummy variable)

#get indicator variables and assign it to data frame "dummy_variable_1"
dummy_variable_1 = pd.get_dummies(df["fuel-type"])

#change column names for clarity
dummy_variable_1.rename(columns={'gas':'fuel-type-diesel', 'diesel':'fuel-type-diesel'}, inplace=True)
print(dummy_variable_1.head())

# merge data frame "df" and "dummy_variable_1" 
df = pd.concat([df, dummy_variable_1], axis=1)

# drop original column "fuel-type" from "df"
df.drop("fuel-type", axis = 1, inplace=True)

#Question #4:
#As above, create indicator variable to the column of "aspiration": "std" to 0, while "turbo" to 1.
dummy_variable_for_aspiration = pd.get_dummies(df["aspiration"])
dummy_variable_for_aspiration.rename(columns={'std':'aspiration-std', 'turbo':'aspiration-turbo'}, inplace=True)

#Question #5:
#Merge the new dataframe to the original dataframe then drop the column 'aspiration'
df = pd.concat([df, dummy_variable_for_aspiration], axis=1)
df.drop("aspiration", axis = 1, inplace=True)

df.to_csv('clean_df.csv')