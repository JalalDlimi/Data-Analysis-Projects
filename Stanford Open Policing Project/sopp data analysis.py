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


# ## 6. Which year had the least number of stops? ([video](https://www.youtube.com/watch?v=W0zGzXQmE7c&list=PL5-da3qGB5IBITZj_dYSFqnd_15JgqwA6&index=7))

# In[ ]:


# this works, but there's a better way
ri.stop_date.str.slice(0, 4).value_counts()


# In[ ]:


# make sure you create this column
combined = ri.stop_date.str.cat(ri.stop_time, sep=' ')
ri['stop_datetime'] = pd.to_datetime(combined)


# In[ ]:


ri.dtypes


# In[ ]:


# why is 2005 so much smaller?
ri.stop_datetime.dt.year.value_counts()


# Lessons:
# 
# - Consider removing chunks of data that may be biased
# - Use the datetime data type for dates and times

# ## 7. How does drug activity change by time of day? ([video](https://www.youtube.com/watch?v=jV24N7SPXEU&list=PL5-da3qGB5IBITZj_dYSFqnd_15JgqwA6&index=8))

# In[ ]:


ri.drugs_related_stop.dtype


# In[ ]:


# baseline rate
ri.drugs_related_stop.mean()


# In[ ]:


# can't groupby 'hour' unless you create it as a column
ri.groupby(ri.stop_datetime.dt.hour).drugs_related_stop.mean()


# In[ ]:


# line plot by default (for a Series)
ri.groupby(ri.stop_datetime.dt.hour).drugs_related_stop.mean().plot()


# In[ ]:


# alternative: count drug-related stops by hour
ri.groupby(ri.stop_datetime.dt.hour).drugs_related_stop.sum().plot()


# Lessons:
# 
# - Use plots to help you understand trends
# - Create exploratory plots using pandas one-liners

# ## 8. Do most stops occur at night? ([video](https://www.youtube.com/watch?v=GsQ6x3pt2w4&list=PL5-da3qGB5IBITZj_dYSFqnd_15JgqwA6&index=9))

# In[ ]:


ri.stop_datetime.dt.hour.value_counts()


# In[ ]:


ri.stop_datetime.dt.hour.value_counts().plot()


# In[ ]:


ri.stop_datetime.dt.hour.value_counts().sort_index().plot()


# In[ ]:


# alternative method
ri.groupby(ri.stop_datetime.dt.hour).stop_date.count().plot()


# Lessons:
# 
# - Be conscious of sorting when plotting

# ## 9. Find the bad data in the stop_duration column and fix it ([video](https://www.youtube.com/watch?v=8U8ob9bXakY&list=PL5-da3qGB5IBITZj_dYSFqnd_15JgqwA6&index=10))

# In[ ]:


# mark bad data as missing
ri.stop_duration.value_counts()


# In[ ]:


# what four things are wrong with this code?
# ri[ri.stop_duration == 1 | ri.stop_duration == 2].stop_duration = 'NaN'


# In[ ]:


# what two things are still wrong with this code?
ri[(ri.stop_duration == '1') | (ri.stop_duration == '2')].stop_duration = 'NaN'


# In[ ]:


# assignment statement did not work
ri.stop_duration.value_counts()


# In[ ]:


# solves SettingWithCopyWarning
ri.loc[(ri.stop_duration == '1') | (ri.stop_duration == '2'), 'stop_duration'] = 'NaN'


# In[ ]:


# confusing!
ri.stop_duration.value_counts(dropna=False)


# In[ ]:


# replace 'NaN' string with actual NaN value
import numpy as np
ri.loc[ri.stop_duration == 'NaN', 'stop_duration'] = np.nan


# In[ ]:


ri.stop_duration.value_counts(dropna=False)


# In[ ]:


# alternative method
ri.stop_duration.replace(['1', '2'], value=np.nan, inplace=True)


# Lessons:
# 
# - Ambiguous data should be marked as missing
# - Don't ignore the SettingWithCopyWarning
# - NaN is not a string

# ## 10. What is the mean stop_duration for each violation_raw?

# In[ ]:


# make sure you create this column
mapping = {'0-15 Min':8, '16-30 Min':23, '30+ Min':45}
ri['stop_minutes'] = ri.stop_duration.map(mapping)


# In[ ]:


# matches value_counts for stop_duration
ri.stop_minutes.value_counts()


# In[ ]:


ri.groupby('violation_raw').stop_minutes.mean()


# In[ ]:


ri.groupby('violation_raw').stop_minutes.agg(['mean', 'count'])


# Lessons:
# 
# - Convert strings to numbers for analysis
# - Approximate when necessary
# - Use count with mean to looking for meaningless means

# ## 11. Plot the results of the first groupby from the previous exercise

# In[ ]:


# what's wrong with this?
ri.groupby('violation_raw').stop_minutes.mean().plot()


# In[ ]:


# how could this be made better?
ri.groupby('violation_raw').stop_minutes.mean().plot(kind='bar')


# In[ ]:


ri.groupby('violation_raw').stop_minutes.mean().sort_values().plot(kind='barh')


# Lessons:
# 
# - Don't use a line plot to compare categories
# - Be conscious of sorting and orientation when plotting

# ## 12. Compare the age distributions for each violation

# In[ ]:


# good first step
ri.groupby('violation').driver_age.describe()


# In[ ]:


# histograms are excellent for displaying distributions
ri.driver_age.plot(kind='hist')


# In[ ]:


# similar to a histogram
ri.driver_age.value_counts().sort_index().plot()


# In[ ]:


# can't use the plot method
ri.hist('driver_age', by='violation')


# In[ ]:


# what changed? how is this better or worse?
ri.hist('driver_age', by='violation', sharex=True)


# In[ ]:


# what changed? how is this better or worse?
ri.hist('driver_age', by='violation', sharex=True, sharey=True)


# Lessons:
# 
# - Use histograms to show distributions
# - Be conscious of axes when using grouped plots

# ## 13. Pretend you don't have the driver_age column, and create it from driver_age_raw (and call it new_age)

# In[ ]:


ri.head()


# In[ ]:


# appears to be year of stop_date minus driver_age_raw
ri.tail()


# In[ ]:


ri['new_age'] = ri.stop_datetime.dt.year - ri.driver_age_raw


# In[ ]:


# compare the distributions
ri[['driver_age', 'new_age']].hist()


# In[ ]:


# compare the summary statistics (focus on min and max)
ri[['driver_age', 'new_age']].describe()


# In[ ]:


# calculate how many ages are outside that range
ri[(ri.new_age < 15) | (ri.new_age > 99)].shape


# In[ ]:


# raw data given to the researchers
ri.driver_age_raw.isnull().sum()


# In[ ]:


# age computed by the researchers (has more missing values)
ri.driver_age.isnull().sum()


# In[ ]:


# what does this tell us? researchers set driver_age as missing if less than 15 or more than 99
5621-5327


# In[ ]:


# driver_age_raw NOT MISSING, driver_age MISSING
ri[(ri.driver_age_raw.notnull()) & (ri.driver_age.isnull())].head()


# In[ ]:


# set the ages outside that range as missing
ri.loc[(ri.new_age < 15) | (ri.new_age > 99), 'new_age'] = np.nan


# In[ ]:


ri.new_age.equals(ri.driver_age)


# Lessons:
# 
# - Don't assume that the head and tail are representative of the data
# - Columns with missing values may still have bad data (driver_age_raw)
# - Data cleaning sometimes involves guessing (driver_age)
# - Use histograms for a sanity check
