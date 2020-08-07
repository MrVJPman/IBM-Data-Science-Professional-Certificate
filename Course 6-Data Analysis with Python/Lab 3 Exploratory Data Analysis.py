import pandas as pd
import numpy as np
       
path='https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DA0101EN/automobileEDA.csv'
df = pd.read_csv(path)
#print(df.head())

#%%capture
#! pip install seaborn

import matplotlib.pyplot as plt
import seaborn as sns
#%matplotlib inline 

# list the data types for each column
#print(df.dtypes)

#Question #1:
#What is the data type of the column "peak-rpm"?
#float64

#for example, we can calculate the correlation between variables of type "int64" or "float64" using the method "corr":
#print(df.corr())

#Question #2:
#Find the correlation between the following columns: bore, stroke,compression-ratio , and horsepower.
#Hint: if you would like to select those columns use the following syntax: df[['bore','stroke' ,'compression-ratio','horsepower']]

print(df[['bore','stroke' ,'compression-ratio','horsepower']].corr())


# Continuous numerical variables:
# Engine size as potential predictor variable of price
sns.regplot(x="engine-size", y="price", data=df)
plt.ylim(0,)
#plt.show()

#We can examine the correlation between 'engine-size' and 'price' and see it's approximately 0.87
#print(df[["engine-size", "price"]].corr())

#Highway mpg is a potential predictor variable of price
sns.regplot(x="highway-mpg", y="price", data=df)

#We can examine the correlation between 'highway-mpg' and 'price' and see it's approximately -0.704
df[['highway-mpg', 'price']].corr()

#Let's see if "Peak-rpm" as a predictor variable of "price".
sns.regplot(x="peak-rpm", y="price", data=df)

#We can examine the correlation between 'peak-rpm' and 'price' and see it's approximately -0.101616
df[['peak-rpm','price']].corr()


#Question 3 a):
#Find the correlation between x="stroke", y="price".

print(df[["stroke","price"]].corr())

#Question 3 b):
#Given the correlation results between "price" and "stroke" do you expect a linear relationship?
#Verify your results using the function "regplot()".
sns.regplot(x="price", y="stroke", data=df)

sns.boxplot(x="body-style", y="price", data=df)
sns.boxplot(x="engine-location", y="price", data=df)
#plt.clf() #Added to clear up previous data
sns.boxplot(x="drive-wheels", y="price", data=df)
#plt.show()

df.describe()

#The default setting of "describe" skips variables of type object.
#We can apply the method "describe" on the variables of type 'object' as follows:

df.describe(include=['object'])

#We can convert the series to a Dataframe as follows :

#print(df['drive-wheels'].value_counts().to_frame())

#Let's repeat the above steps but save the results to the dataframe "drive_wheels_counts"
#and rename the column 'drive-wheels' to 'value_counts'.
drive_wheels_counts = df['drive-wheels'].value_counts().to_frame()
drive_wheels_counts.rename(columns={'drive-wheels': 'value_counts'}, inplace=True)
#print(drive_wheels_counts)

#Now let's rename the index to 'drive-wheels':
drive_wheels_counts.index.name = 'drive-wheels'
#print(drive_wheels_counts)

#Series to Dataframe conversion
engine_loc_counts = df['engine-location'].value_counts().to_frame()
engine_loc_counts.rename(columns={'engine-location': 'value_counts'}, inplace=True)
engine_loc_counts.index.name = 'engine-location'
#print(engine_loc_counts.head(10))








#The "groupby" method groups data by different categories.
#The data is grouped based on one or several variables
#and analysis is performed on the individual groups.

#print(df['drive-wheels'].unique())
#unique values for "drive-wheels"

#We can select the columns 'drive-wheels', 'body-style' and 'price', 
#then assign it to the variable "df_group_one".

df_group_one = df[['drive-wheels','body-style','price']]

#We can then calculate the average price for each of the different categories of data.
# grouping results
df_group_one = df_group_one.groupby(['drive-wheels'],as_index=False).mean()
#print(df_group_one)

#You can also group with multiple variables. 
#For example, let's group by both 'drive-wheels' and 'body-style'. 
#This groups the dataframe by the unique combinations 'drive-wheels' and 'body-style'. 
#We can store the results in the variable 'grouped_test1'.
df_gptest = df[['drive-wheels','body-style','price']]
grouped_test1 = df_gptest.groupby(['drive-wheels','body-style'],as_index=False).mean()
#print(grouped_test1)

#In this case, we will leave the drive-wheel variable as the rows of the table,
#and pivot body-style to become the columns of the table:
grouped_pivot = grouped_test1.pivot(index='drive-wheels',columns='body-style')
#print(grouped_pivot)

#Often, we won't have data for some of the pivot cells.
#We can fill these missing cells with the value 0, 
#but any other value could potentially be used as well. 
#It should be mentioned that missing data is quite a complex subject
#and is an entire course on its own.
grouped_pivot = grouped_pivot.fillna(0) #fill missing values with 0
#print(grouped_pivot)

#Question 4:
#Use the "groupby" function to find the average "price" of each car based on "body-style" ?
df_body_style_price = df[['body-style','price']]
grouped_body_style_average_price  = df_body_style_price.groupby(["body-style"], as_index = False).mean()
#print(grouped_body_style_average_price)






import matplotlib.pyplot as plt
#%matplotlib inline 

#use the grouped results
plt.clf() #Added to clear up previous data
plt.pcolor(grouped_pivot, cmap='RdBu')
plt.colorbar()
#plt.show()


fig, ax = plt.subplots()
im = ax.pcolor(grouped_pivot, cmap='RdBu')

#label names
row_labels = grouped_pivot.columns.levels[1]
col_labels = grouped_pivot.index

#move ticks and labels to the center
ax.set_xticks(np.arange(grouped_pivot.shape[1]) + 0.5, minor=False)
ax.set_yticks(np.arange(grouped_pivot.shape[0]) + 0.5, minor=False)

#insert labels
ax.set_xticklabels(row_labels, minor=False)
ax.set_yticklabels(col_labels, minor=False)

#rotate label if too long
plt.xticks(rotation=90)

fig.colorbar(im)
#plt.show()





#5. Correlation and Causation
from scipy import stats
#Calculate the Pearson Correlation Coefficient and P-value of various
#p < 0.001 and coefficent = -1 or 1
pearson_coef, p_value = stats.pearsonr(df['wheel-base'], df['price'])
pearson_coef, p_value = stats.pearsonr(df['horsepower'], df['price'])
pearson_coef, p_value = stats.pearsonr(df['length'], df['price'])
pearson_coef, p_value = stats.pearsonr(df['width'], df['price'])
pearson_coef, p_value = stats.pearsonr(df['curb-weight'], df['price'])
pearson_coef, p_value = stats.pearsonr(df['engine-size'], df['price'])
pearson_coef, p_value = stats.pearsonr(df['bore'], df['price'])
pearson_coef, p_value = stats.pearsonr(df['city-mpg'], df['price'])
pearson_coef, p_value = stats.pearsonr(df['highway-mpg'], df['price'])

#6. ANOVA
#F-test score: ANOVA assumes the means of all groups are the same, 
#calculates how much the actual means deviate from the assumption, and 
#reports it as the F-test score. A larger score means there is a larger difference between the means.

#P-value: P-value tells how statistically significant is our calculated score value.

#If our price variable is strongly correlated with the variable we are analyzing,
#expect ANOVA to return a sizeable F-test score and a small p-value.

grouped_test2=df_gptest[['drive-wheels', 'price']].groupby(['drive-wheels'])
#does pretty much same thing as df_gptest[['drive-wheels', 'price']] 
print(grouped_test2.head(2))

#We can obtain the values of the method group using the method "get_group".
print(grouped_test2.get_group('4wd')['price'])
#obtains all the results for '4wd' and their price

#we can use the function 'f_oneway' in the module 'stats' to obtain the F-test score and P-value.

# ANOVA
f_val, p_val = stats.f_oneway(grouped_test2.get_group('fwd')['price'], grouped_test2.get_group('rwd')['price'], grouped_test2.get_group('4wd')['price'])  
#separately
f_val, p_val = stats.f_oneway(grouped_test2.get_group('fwd')['price'], grouped_test2.get_group('rwd')['price'])  
f_val, p_val = stats.f_oneway(grouped_test2.get_group('4wd')['price'], grouped_test2.get_group('rwd')['price']) 
f_val, p_val = stats.f_oneway(grouped_test2.get_group('4wd')['price'], grouped_test2.get_group('fwd')['price'])  

