# Assignment 1. Perform the following operations using Python on the Facebook metrics data sets
# a. Create data subsets <br>
# b. Merge Data<br>
# c. Sort Data<br>
# d. Transposing Data<br>
# e. Shape and reshape Data<br>
# 
# Download Dataset (Facebook metrics) : from following link
# https://www.kaggle.com/datasets/masoodanzar/facebook-metrics


import pandas as pd
import numpy as np
df = pd.read_csv('/home/mangal/Downloads/dataset_Facebook.csv', sep = ";") # put your path here
df.head()


# ### a. Create data subsets

# create subsets by selecting specific rows
# we also create a csv file to store the subset

subset1  = df.loc[df["Type"] == "Photo"]
subset1.to_csv('/home/mangal/Downloads/photo_data.csv')


#create subsets by selecting specific rows based on position

temp = df.iloc[10:25]

print(temp)


# create subsets by selecting specific columns and specific rows
df1 = df[['Type', 'Category', 'comment']].loc[4:17]
df2 = df[['Type', 'Category', 'Paid']].loc[24:30]
df3 = df[['Type', 'Category', 'Paid']].loc[31:35]
print("\n", "DF 1 ", df1, "\n")
print("\n", "DF 2 ", df2, "\n")
print("\n", "DF 3 ", df3, "\n")


# ### b. Merge Data
# Merge refers to combining subsets of data
# 
# For merge we can use either:  merge() or method concat() methods

# merge() allows us combine data vertically

# concat() allows us to combine data vertically as well as horizontally
# 
# Lets combine the above three dataframes, horizontally using concat()


pd.concat([df1, df2, df3])


# ### c. Sort Data

# Data can be sorted using pandas sort_values() method 


df.sort_values(['Category', 'Paid'], ascending=False)


# ### d. Transposing Data

# Data transpose refers to converting rows to columns 

# For it we can use pandas transpose() method


df.transpose()   # or simply df.T


# ### e. Shape and reshape Data
# refers to changing the structure of dataframe. 
# i.e representing same information using different shape (rows, cols)
# This can be done using pandas melt() and pivot() methods

print(df1)

# Lets reshape the data by applying melt()
df_melted = df1.melt (id_vars = None , value_vars = None , ignore_index = False )

print(df_melted)

# pivot
# syntax : as follows
# DataFrame.pivot(columns=, index=, values=)
# note: the parametes index and values do not have default values

# lets shape the above long format data using pivot() 

# to convert it into original wide format

df_pivoted = df_melted.pivot( columns = 'variable',values = 'value') 

print( df_pivoted)


# pivot(), reconstructs the melted columns. 

# pivot() can not work with duplicate values in the columns 

# In such case pivot_table() is used.
