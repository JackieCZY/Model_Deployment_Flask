import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

import pickle
#Load data.csv
df = pd.read_csv("toy_dataset.csv")
#drop unused column
df = df.drop("Number", axis= 1)
#convert categorical to numerical
df.loc[df["Gender"]=="Male","Gender"] = 1
df.loc[df["Gender"]=="Female", "Gender"]= 0

df["Illness"] = df["Illness"].apply(lambda x: 1 if x =="Yes" else 0)

#get dummies for cities 
df["City"].value_counts()
encode_city = {"City":     {"Austin": 1, "Boston": 2, "Dallas": 3, "Los Angeles":4, "Mountain View":5,
"New York City": 6, "San Diego":7,"Washington D.C.":8}}
df1 = df.replace(encode_city)

#independent variable
y = df1["Illness"]
X = df1.drop("Illness", axis= 1).values

#train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size =0.25, random_state = 101)

# scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)

#simple logistic regression
lr = LogisticRegression()
lr.fit(X_train, y_train)

#pickle model
pickle.dump(lr, open("model.pkl", "wb"))