import numpy as np
from sklearn import preprocessing, cross_validation , neighbors
import pandas as pd

df = pd.read_csv('datatest.csv')
df.drop(['date'],1,inplace=True)


X = np.array(df.drop(['Occupancy'],1)) 
y = np.array(df['Occupancy'])

X_train,X_test,y_train,y_test = cross_validation.train_test_split(X,y,size=0.2)

clf = neighbors.KNeighborsClassifier() 
clf.fit(X_train,y_train)

accuracy = clf.score(X_test,y_test)
print(accuracy)

example_measures =np.array([25.718,31.29,578.4,760.4,0.00477266099212519])
prediction = clf.predict(example_measures)
print(prediction)