# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import numpy as np
from sklearn import preprocessing 
from sklearn.tree import DecisionTreeClassifier
df = pd.read_csv('Somerville_High_School_YRBS_Raw_Data_2002-2014.csv') 
df.head()
colnames=df.columns.values.tolist() 

#PreProcees

#Clean all attributes
genderlist = ['Male','Female']
agelist = ['17 years old','18 years old or older','16 years old','15 years old',
 '14 years old', '13 years old or younger']
gradelist = ['Senior - 12th','Sophomore - 10th','Junior - 11th','Freshman - 9th']
racelist = ['Asian or other Pacific Islander','Hispanic or Latino','White',
 'Black', 'American Indian or Alaska Native']
parentslist = ['My mother and my father','My mother and a step-parent', 'My father and a step-parent',
'My mother only', 'My father only']
skl_gralist = ["Mostly B's","Mostly C's","Mostly A's", "Mostly D's", "Mostly E's or F's"]
emo_abuslist = ["Yes", "No"]
mistreatlist = ["Yes", "No"]
harassedlist = ["Yes", "No"]
hurtdatelist = ["Yes", "No"]
ganglist = ["Yes", "No"]
stayhomelist = ['0 days','1 day','4 or 5 days','6 or more days','2 or 3 days']
friendslist = ['3 or 4', 'None', '5 or more', '1 or 2']
alc_30list = ['0 days','3 to 5 days','1 or 2 days','6 to 9 days','10 to 19 days',
 'All 30 days', '20 to 29 days']
pregnantlist = ['I have never had sexual intercourse','Yes', 'No']
gamblelist = ['I have never gambled','Once or twice per year','At least once per month',
 'Daily', 'Weekly']
clubslist = ["Yes", "No"]
sportslist = ["Yes", "No"]
hrs_worklist = ['21-25 hours','5 hours or less','None', '11-15 hours',
 'More than 30 hours', '16-20 hours', '6-10 hours', '26-30 hours']

df1 = df[df['GENDER'].isin(genderlist) & df['age'].isin(agelist) & df['grade'].isin(gradelist) \
         & df['race'].isin(racelist) & df['parents'].isin(parentslist) & df['skl_gra'].isin(skl_gralist) \
         & df['stayhome'].isin(stayhomelist) \
         & df['friends'].isin(friendslist) &  df['alc_30'].isin(alc_30list) & df['pregnant'].isin(pregnantlist) \
         & df['gamble'].isin(gamblelist) & df['clubs'].isin(clubslist) & df['sports'].isin(sportslist) & df['hrs_work'].isin(hrs_worklist) ] 



#Preprocess Text data to catagorical
df1['GENDER'] = df1['GENDER'].map({"Male":1, "Female":0})
df1['age'] = df1['age'].map({'17 years old':17,'18 years old or older':18,'16 years old':16,'15 years old':15,
 '14 years old':14, '13 years old or younger':13})
df1['grade'] = df1['grade'].map({'Senior - 12th':12,'Sophomore - 10th':10,'Junior - 11th':11,'Freshman - 9th':9})
df1['race'] = df1['race'].map({'Asian or other Pacific Islander':0,'Hispanic or Latino':1,'White':2,
 'Black':3, 'American Indian or Alaska Native':4})
df1['parents'] = df1['parents'].map({'My mother and my father':0,'My mother and a step-parent':1, 'My father and a step-parent':2,
'My mother only':3, 'My father only':4})
df1['skl_gra'] = df1['skl_gra'].map({"Mostly A's":4,"Mostly B's":3,"Mostly C's":2, "Mostly D's":1, "Mostly E's or F's":0})
df1['stayhome'] = df1['stayhome'].map({'0 days':0,'1 day':0,'4 or 5 days':1,'6 or more days':2,'2 or 3 days':0})
df1['friends'] = df1['friends'].map({'3 or 4':2, 'None':0, '5 or more':3, '1 or 2':1})
df1['alc_30'] = df1['alc_30'].map({'0 days':0,'3 to 5 days':2,'1 or 2 days':1,'6 to 9 days':3,'10 to 19 days':4,
 'All 30 days':6, '20 to 29 days':5})
df1['pregnant'] = df1['pregnant'].map({'I have never had sexual intercourse':0,'Yes':1, 'No':0})
df1['gamble'] = df1['gamble'].map({'I have never gambled':0,'Once or twice per year':0,'At least once per month':0,
 'Daily':1, 'Weekly':1})
df1['clubs'] = df1['clubs'].map({"Yes":1, "No":0})
df1['sports'] = df1['sports'].map({"Yes":1, "No":0})
df1['hrs_work'] = df1['hrs_work'].map({'More than 30 hours':3, '26-30 hours':3, '21-25 hours':2,
  '16-20 hours':2,'11-15 hours':1, '6-10 hours':1,'5 hours or less':1,'None':0})

#Turn dataframe to metric
p = np.size(colnames)-1 
predictors=colnames[:p] 
target=colnames[p]


le = preprocessing.LabelEncoder()
for i in range(p):     
    df1.ix[:,i] = le.fit_transform(df1.ix[:,i]) 
X,Y = df1[predictors],df1[target]

# Create a decision tree classifier 
dt = DecisionTreeClassifier(criterion='entropy', min_samples_split=20, random_state=99)

dt=dt.fit(X,Y)

print("\n\nDecision Tree")
print(dt.feature_importances_)

#Cross validation for DT
from sklearn.cross_validation import KFold 
crossvalidation = KFold(n=X.shape[0], n_folds=20, shuffle=True, random_state=1) 
from sklearn.cross_validation import cross_val_score 
score = np.mean(cross_val_score(dt, X, Y, scoring='accuracy', cv=crossvalidation, n_jobs=1)) 
print ("\nDT Cross validation Score")
print (score)  

# RF Classifier
from sklearn.ensemble import RandomForestClassifier
rft = RandomForestClassifier(n_estimators=1000)
rft.fit(X, Y)
print("\n\nRandom Forrest")
print(rft.feature_importances_)

#Cross validate for RF
score = np.mean(cross_val_score(rft, X, Y, scoring='accuracy', cv=crossvalidation, n_jobs=1)) 
print ("\nRF validation Score")
print (score)
print(rft.predict([[17,1,11,2,3,2,0,5,0,0,1,0,2]]))
