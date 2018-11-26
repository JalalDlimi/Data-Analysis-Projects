#!/usr/bin/env python
# coding: utf-8


# You can find the dataset I used in this repo.
# 
# * `sopp.csv` is the Rhode Island dataset from the [Stanford Open Policing Project](https://openpolicing.stanford.edu/), made available under the [Open Data Commons Attribution License](https://opendatacommons.org/licenses/by/summary/)

# In[1]:


import pandas as pd
import matplotlib as plt
# allowing plots to appear for some version of jupyter notebooks!
get_ipython().run_line_magic('matplotlib', 'inline')


# ## Stanford Open Policing Project Dataset

# In[2]:


df = pd.read_csv('police.csv')
# I have reduced the number of rows and got rid of the useless columns from the original dataset


# In[3]:


df.head()


# In our dataset, each row represents a one traffic stop !

# In[3]:


df.shape


# In[4]:


df.dtypes


# In[5]:


df.isnull().sum()


# In[6]:


# As you can notice, we have a whole column with missing values ("county_name") - we will delete it!
# I used inplace=True to avoid re-assigning the dataframe
df.drop('county_name', axis=1, inplace=True)

# alternative method :
# ri.dropna(axis='columns', how='all').shape


# In[7]:


df.shape # we got now 14 columns


# In[8]:


df.columns  # the columns that r left


# In[9]:


df.loc[df.violation == 'Speeding', 'driver_gender'].value_counts()
# checking how many men and women get pulled over because of speedingg


# In[10]:


# checking for example how many 50yo drivers get pulled over for different violations
df[df.driver_age==50].violation.value_counts()


# In[11]:


# how often men and women violate nd if search is conducted or not.
df.groupby(['search_conducted','driver_gender']).violation.value_counts().unstack()


# In[12]:


# checking how much search operations were conducted
df.search_conducted.value_counts(normalize=True) # which is also the mean of how much the search was actually conducted


# In[13]:


# search rate by gender
df.groupby('driver_gender').search_conducted.mean()


# In[14]:


# including the violation type as a second factor
df.groupby(['violation', 'driver_gender']).search_conducted.mean()


# In[15]:


df.isnull().sum()


# In[103]:


# we noticed that search_type got +88k mmissig values -- WHY ? It's because if no search is conducted, there is no search type !
df.search_type.value_counts()


# In[104]:


df.search_type.value_counts().plot(kind="bar")


# In[17]:


df[df.search_conducted == False].search_type.value_counts(dropna=False)
# an alternative way to check if there are any missing values when a search WAS CONDUCTE :
# df[df.search_conducted == True].search_type.isnull().sum()


# Let's check for example how often is the driver frisked !
# ##### we will create a new column with name "Frist" where will be assigning all the search types that contains "Protective Frisk"
# 

# In[18]:


# includes partial matches
df['Frisk'] = df.search_type.str.contains('Protective Frisk')


# In[19]:


df.columns


# In[20]:


df.Frisk.dtype


# In[21]:


# includes exact matches only
df.Frisk.value_counts()


# In[96]:


df.Frisk.value_counts(dropna=False)


# In[23]:


df.Frisk.mean()
# means 8.5%  of the searches conducted, there is a frisk !
# PS: The mean method ignores the NaN  values!


# Let's check now how often there is a Reasonable Suspicion search !
# ##### we will create now a column with the name "Suspicion" where we will be assigning all the search types that contains "Reasonable Suspicion"
# 

# In[24]:


df['Suspicion']= df.search_type.str.contains('Reasonable Suspicion')


# In[25]:


df.columns


# In[26]:


df.Suspicion.value_counts()


# In[27]:


df.Suspicion.mean()


# ###### Let's now use the stop_date and stop_time to get more insights about the data!

# In[97]:


# Getting the year with least number of stops !
df.stop_date.str.slice(0,4).value_counts().plot(kind='pie')


# In[60]:


# We create a new column to save stop dates and time , both combined

combined = df.stop_date.str.cat(df.stop_time, sep = " ") # conct
df['datatime']=pd.to_datetime(combined) # I change the combined string type to a pandas datetime for future use!


# In[53]:


df.dtypes


# In[87]:


df.datatime.dt.year.value_counts().sort_values() # another way of counting how many stops per years using pandas data&time !


# In[59]:


# Getting the year with least stops ! 
df.datatime.dt.year.value_counts().sort_values().index[0]


# <b> Drug activity changes in time !

# In[102]:


df.groupby(df.datatime.dt.hour).drugs_related_stop.mean()


# In[113]:


df.groupby(df.datatime.dt.hour).drugs_related_stop.mean().plot()


# In[117]:


# count drug-related stops by hour instead of using the mean
df.groupby(df.datatime.dt.hour).drugs_related_stop.sum().plot()


# We all know that most accidents happen at night, would be great to make sure about that using data !

