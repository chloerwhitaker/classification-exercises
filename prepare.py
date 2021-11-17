#!/usr/bin/env python
# coding: utf-8

# In[59]:


import acquire

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.model_selection import train_test_split


np.random.seed(123)


# ### Acquiring Iris Data

# In[60]:


#aquire iris data
iris_df = acquire.get_iris_data()
iris_df.head()


# In[61]:


#rename so split works on renamed column
iris_df = iris_df.rename(columns={'species_name' : 'species'})


# ## Prepare Iris Data

# In[62]:


#prepare function to prep iris data
def prep_iris(iris_df):
    cols_to_drop = ['species_id']
    iris_df = iris_df.drop(columns=cols_to_drop)
    iris_df = iris_df.rename(columns={'species_name' : 'species'})
    dummy_df = pd.get_dummies(iris_df[['species']], dummy_na=False)
    iris_df = pd.concat([iris_df, dummy_df], axis=1)

    return iris_df


# In[63]:


#bring in fresh iris data to test prep function
iris_df = acquire.get_iris_data()
iris_df.head()


# In[64]:


#test prep_iris function on fresh iris data
iris_df = prep_iris(iris_df)
iris_df.head()


# In[65]:


train.head()


# ### Splitting Iris Data

# In[66]:


train, test = train_test_split(iris_df, test_size = .2, random_state=123, stratify=iris_df.species)
train, validate = train_test_split(train, test_size=.3, random_state=123, stratify=train.species)


# In[67]:


#split data function
def split_data(iris_df):
    '''
    Takes in a dataframe and return train, validate, test subset dataframes
    '''
    train, test = train_test_split(iris_df, test_size = .2, random_state=123, stratify=iris_df.species)
    train, validate = train_test_split(train, test_size=.3, random_state=123, stratify=train.species)
    return train, validate, test


# In[68]:


train.head()


# ### Acquiring Titanic Data

# In[74]:


titanic_df = acquire.get_titanic_data()
titanic_df.head()


# ## Prepare Titanic Data

# In[75]:


def prep_titanic(titanic_df):
    '''
    This function will clean the titanic data...
    '''
    titanic_df = titanic_df.drop_duplicates()
    cols_to_drop = ['deck', 'embarked', 'class', 'age']
    titanic_df = titanic_df.drop(columns=cols_to_drop)
    dummy_df = pd.get_dummies(titanic_df[['sex', 'embark_town']], dummy_na=False, drop_first=[True, True])
    titanic_df = pd.concat([titanic_df, dummy_df], axis=1)
    
    # split the data
    train, validate, test = split_data(titanic_df)
    return titanic_df


# In[76]:


titanic_df = acquire.get_titanic_data()
titanic_df.head()


# In[77]:


titanic_df = prep_titanic(titanic_df)
titanic_df.head()


# In[78]:


train.head()


# ### Splitting Titanic Data

# In[79]:


#split data function
def split_data(titanic_df):
    '''
    Takes in a dataframe and return train, validate, test subset dataframes
    '''
    train, test = train_test_split(titanic_df, test_size = .2, random_state=123, stratify=titanic_df.survived)
    train, validate = train_test_split(train, test_size=.3, random_state=123, stratify=train.survived)
    return train, validate, test


# In[80]:


train, test = train_test_split(titanic_df, test_size = .2, random_state=123, stratify=titanic_df.survived)
train, validate = train_test_split(train, test_size=.3, random_state=123, stratify=train.survived)


# In[81]:


train.head()


# ### Acquiring Telco Data

# In[91]:


telco_df = acquire.get_telco_data()
telco_df.head()


# ## Prepare Telco Data

# In[92]:


def prep_telco(telco_df):
    '''
    This function will clean the telco data...
    '''
    #Drop Duplicates
    telco_df = telco_df.drop_duplicates()
    
    # Drop null values stored as whitespace    
    telco_df['total_charges'] = telco_df['total_charges'].str.strip()
    telco_df = telco_df[telco_df.total_charges != '']
    
    # Convert to correct datatype
    telco_df['total_charges'] = telco_df.total_charges.astype(float)
    
    # Drop Columns
    cols_to_drop = ['customer_id', 'payment_type_id', 'internet_service_type_id', 'contract_type_id']
    telco_df = telco_df.drop(columns=cols_to_drop)
    
    # Get dummies for non-binary categorical variables
    dummy_df = pd.get_dummies(telco_df[['multiple_lines',                               'online_security',                               'online_backup',                               'device_protection',                               'tech_support',                               'streaming_tv',                               'streaming_movies',                               'contract_type',                               'internet_service_type',                               'payment_type']], dummy_na=False)
    # Concatenate dummy dataframe to original 
    telco_df = pd.concat([telco_df, dummy_df], axis=1)
   
    return telco_df


# In[93]:


telco_df = acquire.get_telco_data()
telco_df.head()


# In[94]:


telco_df = prep_telco(telco_df)
telco_df.head()


# In[95]:


train.head()


# ### Splitting Telco Data

# In[96]:


#split data function
def split_data(telco_df):
    '''
    Takes in a dataframe and return train, validate, test subset dataframes
    '''
    train, test = train_test_split(telco_df, test_size = .2, random_state=123, stratify=telco_df.churn)
    train, validate = train_test_split(train, test_size=.3, random_state=123, stratify=train.churn)
    return train, validate, test


# In[97]:


train, test = train_test_split(telco_df, test_size = .2, random_state=123, stratify=telco_df.churn)
train, validate = train_test_split(train, test_size=.3, random_state=123, stratify=train.churn)


# In[98]:


train.head()


# In[ ]:




