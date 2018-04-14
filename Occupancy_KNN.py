
# coding: utf-8

# In[5]:

###IMPORTING MODULE
import numpy as np
from sklearn import preprocessing, neighbors
from sklearn.model_selection import train_test_split
import pandas as pd


# In[6]:

#     IMPORTING DATASET
df = pd.read_csv('datatest.csv')
df.drop(['date'],1,inplace=True)


# In[7]:

#     Divinding depending and Independing variable
X = np.array(df.drop(['Occupancy'],1)) 
y = np.array(df['Occupancy'])


# In[8]:

#     divining the array as train test arrays
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25)


# In[9]:

#     Training KNN
clf = neighbors.KNeighborsClassifier() 
clf.fit(X_train,y_train)


# In[10]:

accuracy = clf.score(X_test,y_test)


# In[12]:

print(accuracy)


# In[13]:

# Lets Predict >>>>> Here with a random data ..ie example Measures (with random values) we are predicting
# if the given result produces Occupancy or not ..... Remember to remove the Occupancy column data....

example_measures =np.array([25.718,31.29,578.4,760.4,0.00477266099212519])
prediction = clf.predict(example_measures)
print(prediction)


# In[ ]:



