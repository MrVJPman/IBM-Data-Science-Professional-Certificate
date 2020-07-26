# Dependency needed to install file 

#Code for Juptyer Notebook
#!pip install xlrd 

# Import required library
import pandas as pd

# Read data from CSV file
csv_path = 'https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/PY0101EN/Chapter%204/Datasets/TopSellingAlbums.csv'
df = pd.read_csv(csv_path)

# Print first five rows of the dataframe
df.head()

# Read data from Excel File and print the first five rows
xlsx_path = 'https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/PY0101EN/Chapter%204/Datasets/TopSellingAlbums.xlsx'

df = pd.read_excel(xlsx_path)
df.head()

# Access to the column Length
x = df[['Length']]
print(x)

# Get the column as a series
x = df['Length']
print(x)

# Get the column as a dataframe
x = type(df[['Artist']])
print(x)

# Access to multiple columns
y = df[['Artist','Length','Genre']]
print(y)

# Access the value on the first row and the first column
df.iloc[0, 0]

# Access the value on the second row and the first column
df.iloc[1,0]

# Access the value on the first row and the third column
df.iloc[0,2]

# Access the column using the name
df.loc[0, 'Artist']

# Access the column using the name
df.loc[1, 'Artist']

# Access the column using the name
df.loc[0, 'Released']

# Access the column using the name
df.loc[1, 'Released']

# Slicing the dataframe
df.iloc[0:2, 0:3]

# Slicing the dataframe using name
df.loc[0:2, 'Artist':'Released']

#Use a variable q to store the column Rating as a dataframe
q = df[["Rating"]]
print(q)

#Assign the variable q to the dataframe that is made up of the column Released and Artist:
q = df[["Released", "Artist"]]
print(q)

#Access the 2nd row and the 3rd column of df:
df.iloc[2-1, 3-1]

