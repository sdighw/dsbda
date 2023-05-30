#Assignment 2. Perform the following operations using Python on the Air quality and Heart Diseases data sets</p><br>
# a. Data cleaning <br>
# b. Data integration<br>
# c. Data transformation<br>
# d. Error correcting<br>
# e. Data model building (performed seperately in Part 2)<br>
# 
# Dataset (Air quality):  https://www.kaggle.com/datasets/fedesoriano/air-quality-data-set <br>
# Dataset ( Heart Diseases ) : https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset



import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
df = pd.read_csv('/home/mangal/Downloads/AirQuality.csv',sep = ';') # put your own path of dataset



# check the columns names and the respective datatypes
df.info()


# In[4]:


# check the statistical informatin of all columns: std , mean , min , max
df.describe()


# Examine some sample rows ; head shows top five rows by default
df.head()


# Note in above sample rows : the columns 'CO(GT)' , 'C6H6(GT)' ,  'T' , 'RH' and 'AH' all 
# are having numeric values
# but here these values contain commas ',' instead of '.' 
# We will handle this error in data cleaning phase


df.shape # check total number of rows and columns


# # A. DATA CLEANING
# 
# 
# handling missing values, And droping duplicate rows
# 

# Ckeck for missing values in each column ; by finding counts
df.isnull().sum()


# The above output shows that the last two columns contain entirely NaN
# or Null values
# lets find the count using following command
df['Unnamed: 15'].isnull().sum()
df['Unnamed: 16'].isnull().sum()


# This count shows that all values in these two columns are null ;
# hence lets drop these two columns

df.drop(['Unnamed: 15','Unnamed: 16'],axis = 1,inplace = True)


df.shape


df.isnull().sum()


# We will print all the rows having any of its value as NaN
df[df.isnull().any(axis=1)]


# The above output shows that these rows of missing values are having NaN in all the columns
# So we will drop all such rows using following command

df.dropna(how = 'all', inplace = True)

# Following is Syntax of dropna()
# df.dropna( subset = [ ‘column1’ , ‘column2’, ‘column5’ ] ,  how = ‘any’ ) 
# drops a row if any of given column is NA; if subset is not given it considers all columns


# we will check if the rows sre droped by printing these rows
df[df.isnull().any(axis=1)] 


column_list = df.columns.values.tolist()
print(column_list)


# Lets check unique values in each columns
# column_list = df.columns.values.tolist()
for column_name in column_list:
    print ("\n", column_name)
    
    print ( df[column_name].value_counts(dropna = False ) ) # print count ALL unique values in each column 
# dropna argument is optional, shows the count of NA  
    
#print ( df[column_name].unique())  # prints all unique values   


# 
# # ERROR CORRECTION
# 
# in our dataset the column 'CO(GT)' ,'C6H6(GT)', 'T' ,  'RH' ,  'AH' contain numeric values but contain a comma in each value;

# All the values in this column need to be corrected by replacing ',' with '.' 
# 
# Also it is observed that the first column is not showing its datatype as date.
# 
# Lets handle these errors by replacing all the values in these columns with corresponding corrected values


df.info()

print(df['CO(GT)'])

print( df['C6H6(GT)'])

print(df['RH'])

print(df['AH'])

print(df['T'])


# Note all above column we need to replace ',' with '.' and convert them to numeric from string


# ERROR CORRECTION column formatting: 'C6H6(GT)'
j = 'CO(GT) C6H6(GT) T RH AH'.split()
print(j) 
df.replace(to_replace=',',value='.',regex=True,inplace=True) 
for i in j :
    df[i] = pd.to_numeric(df[i],errors='coerce') 



# lets check again the datatypes of the said column 
# observe below 'C6H6(GT)' and 'CO(GT)' column is conveted to float now 
df.info() 
# following output shows the datatypes of corrected columns changed now

 
# # DATA INTEGRATION
# 
# combining Data from multiple Sources 
#DATA INTEGRATION
#If data is to be combined from more than two datasets then we can use merge or concat commands
#In our case data is from a single source so no need of data merge:refer assignment1 for merge and concat


# 
# # DATA TRANSFORMATION
# 
# Changing the form of data.

# Standardization is one kind of data transformation: it changes the scale of data
# 
# Lets check our dataset for applying standardization on our dataset 

# We will check the values in various columns using head command

df.head()


# output of head() shows that all columns have varying range of values
# Lets bring all these columns on same scale by using standardization as follows:

#cols = df.columns

scaler = StandardScaler()

Numerical_col = df.select_dtypes(exclude = [np.object_ , np.datetime64 ]  ) # or use include np.float64
#categorical_col = dataframe.select_dtypes(exclude=np.number)


for col in Numerical_col:
    df[[col]] = scaler.fit_transform(df[[col]])
df.head()
# the ouput of head() command shows that all columns has TRANSFORMED to same range

# in our case or data column is not reflecting its type as date 
# Hence need lets handle this error as follows


df.info()

# DATA TRANSFORMATION : 
# changing the form of the data 
# date is stored as string. We will change string type to date
# Formatting Date and Time to datetime type

df['Date'] = pd.to_datetime(df['Date'],dayfirst=True) 

df['Time'] = pd.to_datetime(df['Time'],format= '%H.%M.%S' ).dt.time
# Series.dt.time: Returns numpy array of datetime.time objects.

df.head()
df.info()

df.info() # Data type of data changed to date as shown by below command
