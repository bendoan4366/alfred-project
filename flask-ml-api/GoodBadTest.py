#!/usr/bin/env python
# coding: utf-8

# # The Alfred Project Pt. 1 - GoodBad Test
#
# The **GoodBadTest** is a simple classfication model that learns on two playlists - a "good" playlist which contains songs from my top playlists, and a "bad" playlist, which contains songs pulled from spotify curated playlists of artists and genres that I don't enjoy. The model uses PCA to create a master audio data dataset that will then loaded into various models for performance comparison.
#
# The purpose of the model is to do 3 things:
#
# - Learn to train models off of Spotify's audio features
# - Test the effectiveness of PCA and other feature engineering methods on music/audio data
# - Compare the implementation of Tensorflow Keras vs. Adaboost models on audio data
# - Lay the foundation for a playlist-based recommendation models

#

# In[29]:


# import all the things

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split


from sklearn.metrics import accuracy_score
import sklearn.metrics as sm


from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier


from sklearn.feature_selection import SelectFromModel
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import scale


from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix, classification_report
from sklearn import datasets


from pathlib import Path


# In[30]:


# load in playlist data via csv

audio_data_good = pd.read_csv("audio_data_good.csv", encoding="latin-1")
audio_data_bad = pd.read_csv("audio_data_bad.csv", encoding="latin-1")


# ## Data Inspection

# Inspect the data shape and features for each of the two playlists

# In[31]:


audio_data_good.head(10)


# In[32]:


audio_data_bad.head(10)


#
#

# Histograms and dataframe summaries reveal the greatest discrpencies come in the valence, popular, and bpm features

# In[33]:


features_list = list(audio_data_good.columns)
features = audio_data_good.loc[:, features_list]
ax = features.hist(figsize=(20, 10), alpha=0.6)


# In[34]:


bad_features_list = list(audio_data_bad.columns)
bad_features = audio_data_bad.loc[:, bad_features_list]
ax2 = bad_features.hist(figsize=(20, 10), alpha=0.6)
ax2


# ## Feature Engineering - PCA

# Coming soon....

# ## Model Prep

# Add playlist labels to each data set and then combine datasets into a master for training and testing

# In[35]:


# add "good" label to good playlist data
audio_data_good["Playlist"] = "Good"


# In[36]:


# add "bad" label to bad playlist data
audio_data_bad["Playlist"] = "Bad"


# In[44]:


audio_data_master = audio_data_good.append(audio_data_bad)
audio_data_master.head(5)


# In[38]:


# specify classifying label and applicable numeric features

goodBadClassifer = audio_data_master.loc[:, ["Playlist"]]
audio_features = audio_data_master.loc[
    :, ["BPM", "ENERGY", "DANCE", "LOUD", "VALENCE", "ACOUSTIC", "POP."]
]


# In[39]:


audio_features.head(10)


# ## Model Training: AdaBoost Classification

# The first model we test will be a simple AdaBoost Classifier. With a train-test-split evaluation and 33% test size.

# In[40]:


alfred_adalearn = AdaBoostClassifier()
Xtrain, XTest, yTrain, yTest = train_test_split(
    audio_features, goodBadClassifer, test_size=0.33
)


# In[41]:


alfred_adalearn.fit(Xtrain, yTrain)


# In[42]:


ypred = alfred_adalearn.predict(XTest)


# In[43]:


print("Alfred acc:", accuracy_score(yTest, ypred))


# Initial AdaBoost Classifer accuracy was **79.31%** with no finetuning or feature engineering

# ## Model Training: Keras Deep Learning

# Coming soon...

# In[ ]:

