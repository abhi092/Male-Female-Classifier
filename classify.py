import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split


dataset= pd.read_csv("names_dataset.csv")
 # Features and Labels
X = dataset.iloc[:,1]
y = dataset.iloc[:,2]
    
# Vectorization
countvectorizer = CountVectorizer()
X = countvectorizer.fit_transform(X) 

labelEncoder_y = LabelEncoder()
y = labelEncoder_y.fit_transform(y)
 
 # Loading our ML Model

#naivebayes_model = open("models/naivebayesgendermodel.pkl","rb")

#clf = joblib.load(naivebayes_model)

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25 ,random_state=42 )

clasifier = MultinomialNB()
clasifier.fit(X_train,y_train)
clasifier.score(X_test,y_test)



 # Receives the input query from form

namequery = "Komal"
data = [namequery]
vect = countvectorizer.transform(data).toarray()
my_prediction = clasifier.predict(vect)
if my_prediction == 1:
    print("Male")
else :
    print("Female")