#!/usr/bin/env python
# coding: utf-8

# In this project, I performed a data analysis on the Stanford Open Policing Project for the Rhode Island data using pandas.
# I explained every step and what results mean during my work! Enjoy learning !
# 
# You can find the dataset I used in this repo.
# 
# * `sopp.csv` is the Rhode Island dataset from the [Stanford Open Policing Project](https://openpolicing.stanford.edu/), made available under the [Open Data Commons Attribution License](https://opendatacommons.org/licenses/by/summary/)

# In[1]:


import pandas as pd
import matplotlib as plt
# allowing plots to appear for some version of jupyter notebooks!
get_ipython().run_line_magic('matplotlib', 'inline')


# ## Stanford Open Policing Project Dataset
# 
# To understand what each column of the stanford dataset means, check this where you can find each column meaning so that you can be able to well understand the dataset and perform good data analysis! (check thhe stanford open policing website for it Jey)*

# In[2]:


df = pd.read_csv('police.csv')
# I have reduced the number of rows and got rid of the useless columns from the original dataset


# In[3]:


df.head()


# In our dataset, each row represents a one traffic stop !

# In[4]:


df.shape


# In[5]:


df.dtypes


# In[6]:


df.isnull().sum()


# In[7]:


# As you can notice, we have a whole column with missing values ("county_name") - we will delete it!
# I used inplace=True to avoid re-assigning the dataframe
df.drop('county_name', axis=1, inplace=True)

# alternative method :
# ri.dropna(axis='columns', how='all').shape """


# In[8]:


df.shape # we got now 14 columns


# In[9]:


df.columns  # the columns that r left


# In[10]:


df.loc[df.violation == 'Speeding', 'driver_gender'].value_counts()
# checking how many men and women get pulled over because of speedingg


# In[11]:


# checking for example how many 50yo drivers get pulled over for different violations
df[df.driver_age==50].violation.value_counts()


# In[12]:


# how often men and women violate nd if search is conducted or not.
df.groupby(['search_conducted','driver_gender']).violation.value_counts().unstack()


# In[13]:


# checking how much search operations were conducted
df.search_conducted.value_counts(normalize=True) # which is also the mean of how much the search was actually conducted


# In[14]:


# search rate by gender
df.groupby('driver_gender').search_conducted.mean()


# In[16]:


# including the violation type as a second factor
df.groupby(['violation', 'driver_gender']).search_conducted.mean()


# In[17]:


df.isnull().sum()


# In[23]:


# we noticed that search_type got +88k mmissig values -- WHY ? It's because if no search is conducted, there is no search type !
df.search_type.value_counts()


# In[20]:


df[df.search_conducted == False].search_type.value_counts(dropna=False)
# an alternative way to check if there are any missing values when a search WAS CONDUCTE :
# df[df.search_conducted == True].search_type.isnull().sum()


# Let's check for example how often is the driver frisked !
# ##### we will create a new column with name "Frist" where will be assigning all the search types that contains "Protective Frisk"
# 

# In[35]:


# includes partial matches
df['Frisk'] = df.search_type.str.contains('Protective Frisk')


# In[36]:


df.columns


# In[37]:


df.Frisk.dtype


# In[38]:


# includes exact matches only
df.Frisk.value_counts()


# In[39]:


df.Frisk.value_counts(dropna=False)


# In[41]:


df.Frisk.mean()
# means 8.5%  of the searches conducted, there is a frisk !
# PS: The mean method ignores the NaN  values!


# Let's check now how often there is a Reasonable Suspicion search !
# ##### we will create now a column with the name "Suspicion" where we will be assigning all the search types that contains "Reasonable Suspicion"
# 

# In[42]:


df['Suspicion']= df.search_type.str.contains('Reasonable Suspicion')


# In[43]:


df.columns


# In[46]:


df.Suspicion.value_counts()


# In[47]:


df.Suspicion.mean()


