# -*- coding: utf-8 -*-
import pandas as pd
from sklearn.externals import joblib
import pickle

df=pd.read_csv('spam.csv',encoding='latin-1')
df.drop(['Unnamed: 2','Unnamed: 3','Unnamed: 4'],axis=1,inplace=True)
df.rename(columns={'v1':'label','v2':'message'},inplace=True)
column_titles=['message','label']
df=df.reindex(columns=column_titles)

df['label']=df['label'].map({'ham':0,'spam':1})

X=df['message']

y=df.iloc[:,1]

from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()
X=cv.fit_transform(X)

pickle.dump(cv,open('transform.pkl','wb'))

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.33,random_state=0)


from sklearn.naive_bayes import MultinomialNB
classifier =MultinomialNB()
classifier.fit(X_train, y_train)
classifier.score(X_test,y_test)

filename='npl_model.pkl'
pickle.dump(classifier,open(filename,'wb'))


from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
print("Accuracy: {:.2f} %".format(accuracies.mean()*100))
print("Standard Deviation: {:.2f} %".format(accuracies.std()*100))