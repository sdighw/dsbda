# Group B: Assignment 3: Data Analytics using Python 
# Visualize the data using Python libraries matplotlib, seaborn by plotting the graphs for Air quality and Heart Diseases datasets

# #### Dataset Description
# 
# Attributes of the Dataset:
# 
# 1. age: age in years (29-77)
# 2. sex: sex (Female or Male)
#     * Value 0: Female
#     * Value 1: Male
# 3. cp: chest pain type
#     * Value 0: typical angina
#     * Value 1: atypical angina
#     * Value 2: non-anginal pain
#     * Value 3: asymptomatic
#     
# Typical anginal pain - Typical (classic) angina chest pain consists     of discomfort that is Provoked by exertion or emotional stress and relieved by rest or nitroglycerine (or both).
# 
# atypical angina - Angina pectoris which does not have associated classical symptoms of chest pain. Symptoms may include weakness, nausea, or sweating.
# 
# Non-anginal pain- *A chest pain is very likely nonanginal if its duration is over 30 minutes or less than 5 seconds, it increases with inspiration, can be brought on with one movement of the trunk or arm, can be brought on by local fingers pressure, or bending forward, or it can be relieved immediately on lying down. 
# 
# asymptomatic - a person who does not have any symptoms.
#     
# 4. trestbps: resting blood pressure (in mm Hg on admission to the hospital) (94-200)
# 5. chol: cholesterol level (mg/dl) (126-564)
# 6. fbs: fasting blood sugar > 120 mg/dl (False or True)
#     * Value 0: False
#     * Value 1: True
# 7. restecg: resting electrocardiographic results
#     * Value 0: normal
#     * Value 1: having ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV)
#     * Value 2: showing probable or definite left ventricular hypertrophy by Estes' criteria
# 8. thalach: maximum heart rate achieved (71-202)
# 9. exang: exercise induced angina (Presence or Absence)
#     * Value 0: Absence of Exercise Induced Angina
#     * Value 1: Presence of Exercise Induced Angina
# 10. oldpeak = ST depression induced by exercise relative to rest (0-6.2)
# 11. slope: the slope of the peak exercise ST segment
#     * Value 0: upsloping
#     * Value 1: flat
#     * Value 2: downsloping
# 12. ca: number of major vessels (0-3) colored by flouroscopy
# 13. thal: thallium (1-3)
#     * Value 1: normal 
#     * Value 2: fixed defect 
#     * Value 3: reversable defect
# 14. target: Heart Disease (Presence or Absence)
#     * Value 0: Absence of Heart Disease
#     * Value 1: Presence of Heart Disease

# ####  Load the dataset

import pandas as pd
import numpy as np
df=pd.read_csv("/home/mangal/Downloads/heart.csv") # give here your own path for dataset
df


# ### Exploring the dataset


df.shape


df.columns

df.dtypes


df.info()


df.describe()


df.isnull().sum()
# no null values

df.duplicated().any().sum()

df.drop_duplicates(inplace=True)

df.shape
# out of 1025 rows , 302 are left after removal of duplicates

df.duplicated().sum()

#it's showing that the duplicates have been removed


# ### Exploring the dataset**

# ### Visualising the dataset Correlation matrix : use seaborn.heatmap()

# use seaborn library

# Heatmap : is defined as a graphical representation of data using 
# colors to visualize the value of the matrix. 

# A heatmap give a plot of matrix values as  a matrix of color-encoded
# values. As parameter it takes a 2D dataset
# Heatmaps in Seaborn can be plotted by using the seaborn.heatmap()method.

import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(15,6)) # try (15 , 10)
sns.heatmap(df.corr(),annot=True)
#sns.heatmap(df.corr(),annot=True,linewidths=0.5,fmt= ".2f",cmap="YlGnBu");

plt.title('Degree of Correlation of variables in the dataset')

# here , we can see that the 'cp','thalach','slope' columns show a strong 
# positive correlation with the target variable which means that the type
# of chest plain (cp) , heart rate achieved (thalach) and slope of the 
# ST depression (slope) , have found to directly proportional to the
# chance of heart attack (denoted by target variable).


# #### Discrete variable : use seaborn.countplot() or histplot() 
# #### Distribution of target variable using seaborn.countplot()
# Countplot
# This is a seaborn-specific function which is used to plot the count or frequency distribution of each unique observation in the categorical variable. It is similar to a histogram over a categorical rather than quantitative variable.


# Lets check the distribution of target variable
# Count of people having more or less chances of heart attack

sns.countplot(x='target',data=df)

plt.xticks([0,1],['less chance','more chance'])

plt.title('Chances of heart disease')

plt.figure(figsize=(15,6))

# we can also get these numbers using the pandas value_counts() method
df['target'].value_counts(normalize=True)


# #### Discrete variable : use seaborn.countplot() or histplot()
# #### Visualising the Gender distribution:  seaborn countplot()

sns.countplot(x='sex',data=df  )
plt.title('Number of males and females')
plt.xticks([0,1] , ['females','males'])
plt.show()


# #### Discrete variable : use seaborn.countplot() or histplot()
# #### Visualising the Gender distribution:  according to the target variable:
# #### using seaborn.countplot()

sns.countplot(x='sex',data=df, hue ='target'  )
plt.title('chances of heart disease  by gender')
plt.xticks([0,1] ,['females','males'])
plt.legend(labels=['less chance','high chance'])

# Note the hue attribute acts as second dimension for the plot


# #### Contineous variable : use seaborn.histplot() or kdeplot()
# #### or seaborn.displot() with kind = "hist"
# #### Visualising the age distribution using seaborn.histplot()
# 
# A histogram is one of the most frequently used data visualization techniques in machine learning. It represents the distribution of a continuous variable over a given interval or period of time. Histograms plot the data by dividing it into intervals called ‘bins’. It is used to inspect the underlying frequency distribution (eg. Normal distribution), outliers, skewness, etc.
# 
# Histogram are more suitable for conteneous variable Whereas
# Countplots are more suitable for a discrete variable

sns.histplot(df['cp'])
# same result can be achieved as: sns.displot ( df['cp'], kind = 'hist')

plt.xticks([0,1,2,3],["typical angina","atypical angina",
"non-anginal pain","asymptomatic"])

plt.xticks(rotation=70)
plt.figure(figsize=(10,7))
plt.show()


list1 = list(df['cp'].value_counts(normalize=True))

list1

plt.pie(list1,labels=["typical angina","non-anginal",
"atypical angina","asymptomatic"],startangle=180, shadow=True, 
autopct='%1.1f%%')

plt.show()


# ### Distribution of the chest pain distribution as per the target variable

sns.countplot(x='cp',hue='target',data=df)

# formatting the plot

plt.title('Relation between types of chest pain and number of people having high or low chances of heart attack')

plt.xticks([0,1,2,3],["typical angina","atypical angina","non-anginal pain","asymptomatic"])

plt.legend(labels=['low chance','high chance'])


sns.countplot(x='fbs' , hue='target',data=df)
plt.legend(labels=['low chance','high chance'])

# There aare a large number of people who are not diabetic but have high chances of heart attack as compared to people who are diabetic , in this dataset.

df['fbs'].value_counts(normalize=True)

# around 85% of the people are non- diabetic and only 14 % of the people
# are diabetic.

g= sns.FacetGrid(df,hue="sex",aspect=4)
g.map(sns.kdeplot,'trestbps',shade=True)
plt.xlabel("resting blood presuure")
plt.legend(labels=["female","male"])

# generally , females have an all over lower blood pressure than males.
# for women,it's around 120, while for men, it is a little less than 140.
# # ***Visualisation of serrum cholestrol in males and females***


chol= sns.FacetGrid(df,hue="sex",aspect=4)
chol.map(sns.kdeplot,'chol',shade=True)
plt.xlabel("serum cholestrol")
plt.legend(labels=["female","male"])

# mostly females have around 200 mg/dl or less , whereas , males have a little more than 200 mg//dl.

# # **Relation between amount of cholestrol in the blood and chances of heart attack**

# # **Visualising the distribution of all the continuous variables**

cat_val=[] # list for storing categorical variables
cont_val=[] # list for storing continuous variables

for column in df.columns:
    if df[column].nunique() <= 5:
        cat_val.append(column)
    else:
        cont_val.append(column)
cont_val 
# now , this list contains continuous variables.

cat_val = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
# this is a list of all the categorical variables in the dataset.

# ## ***Plotting the continuous variables***

df.hist(cont_val, figsize=(17,6))
plt.show()
plt.tight_layout()

df.hist(cat_val,  layout=(4, 4), figsize=(20, 20), color="DarkCyan", 
               grid=True)
plt.show()


sns.pairplot(df , hue = 'target')
