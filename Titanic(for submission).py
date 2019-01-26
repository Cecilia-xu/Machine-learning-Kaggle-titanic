# -------------------------------------------
# Titanic: Machine Learning from Disaster
# Kaggle Competition
# -------------------------------------------
# INFSCI 2725: Data Analytics
# Fall 2018
# -------------------------------------------
# Yunshu Liang (yul219)
# Qi Lu (qil66)
# Erin Price (eep27)
# Ziyue Qi (ziq2)
# Xiaoqian Xu (xix64)
# -------------------------------------------
# Reference:  https://zhuanlan.zhihu.com/p/31743196
# https://www.kaggle.com/ldfreeman3/a-data-science-framework-to-achieve-99-accuracy
# https://zhuanlan.zhihu.com/p/33733586
# -------------------------------------------
# I. Prepare for the project
# -----------------

#import packages
import random
import re
import sys
#import sns as sns
import time
import seaborn as sns
import pandas as pd
import matplotlib
import numpy as np
import scipy as sp
import IPython
import sklearn
import matplotlib.pyplot as plt
import itertools
import graphviz
import warnings

from sklearn.model_selection import RandomizedSearchCV, cross_val_score

warnings.filterwarnings('ignore')
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn import preprocessing, model_selection, ensemble, linear_model, naive_bayes, tree
from math import isnan
from sklearn import datasets
from sklearn.preprocessing import LabelEncoder


# -------------------------------------------

# read data Files
csvTrain = "Data/train.csv"
csvTest = "Data/test.csv"
csvGender = "Data/genderSubmission.csv"

# Read CSV
dataRaw = pd.read_csv(csvTrain)
print(dataRaw)

dataTest = pd.read_csv(csvTest)
print(dataTest)

#Create a copy
dataTrain = dataRaw.copy(deep = True)

#Create a dictionary for processing
dataCleaner = [dataTrain,dataTest]
#-----------------------------------------
# II. Preview the Data
# -----------------

# Column Values
print(dataTrain.columns.values)
print("-"*10)
#--check if we have read all the data
# First 5 Rows
#print(dataTrain.head())

# Last 5 Rows
#print(dataTrain.tail())

# Data Types
print(dataTrain.info())
print("-"*10)
# Statistic Summary
print(dataTrain.describe())
print("-"*10)
# Sum Missing Values
print('Train columns with null values:\n', dataTrain.isnull().sum())
print("-"*10)
print('Test/Validation columns with null values:\n', dataTest.isnull().sum())
print("-"*10)
# -----------------------------------------------------------------
# Code below this point is following a tutorial
# from the following source: https://zhuanlan.zhihu.com/p/31743196
# -----------------------------------------------------------------
# Look at the relationship
# Relationship Between Sex and Survived (Graph)
dataTrain.groupby(["Survived", "Sex"])["Survived"].count()
dataTrain[["Sex", "Survived"]].groupby(["Sex"]).mean().plot.bar()
plt.show()

# Relationship between Sex and Survived (Numerical)
print(dataTrain.groupby(["Survived", "Sex"])["Survived"].count())

# Relationship between Survived and Sex (Graph)
fig, (axis1, axis2) = plt.subplots(1, 2, figsize=(15, 5))
sns.countplot(data=dataTrain, x="Sex", ax=axis1, palette="Set3")
sns.countplot(data=dataTrain, x="Survived", hue="Sex", order=[0, 1], ax=axis2, palette="Set3")
plt.xticks([0, 1], ["Not Survived", "Survived"])
plt.show()

# Relationship between Pclass and Survived (Graph)
dataTrain[["Pclass", "Survived"]].groupby(["Pclass"]).mean().plot.bar()
plt.show()

# Relationship with and without Sibling/Spouse (Graph)
withSibSp = dataTrain[dataTrain["SibSp"] != 0]
withoutSibSp = dataTrain[dataTrain["SibSp"] == 0]
plt.figure(figsize=(10, 5))
plt.subplot(121)
withSibSp["Survived"].value_counts().plot.pie(autopct="%1.1f%%")
plt.xlabel("With Sibling/Spouse")
plt.subplot(122)
withoutSibSp["Survived"].value_counts().plot.pie(autopct="%1.1f%%")
plt.xlabel("Without Sibling/Spouse")
plt.show()

# Relationship between Fare and Survived (Graph)
dataTrain[["Fare", "Survived"]].groupby("Survived").mean().plot.bar()
plt.show()

# Relationship between Cabin and Survival (Graph)
dataTrain.loc[dataTrain.Cabin.isnull(), "Cabin"] = "U0"
dataTrain["Has_Cabin"] = dataTrain["Cabin"].apply(lambda x: 0 if x == "U0" else 1)
dataTrain[["Has_Cabin", "Survived"]].groupby(["Has_Cabin"]).mean().plot.bar()
plt.show()

# Relationship between Embarked and Survival (Graph)
sns.factorplot("Embarked", "Survived", data=dataTrain, size=3, aspect=3, palette="Set3")
plt.title("Embarked and Survived rate")
plt.show()
dataTrain[["Embarked", "Survived"]].groupby(["Embarked"]).mean().plot.bar()
plt.show()

'''
# Relationship between Pclass and Survived (Numerical)
print(dataTrain.groupby(["Pclass", "Survived"])["Pclass"].count())

#  Relationship between Sex, Pclass, and Survived (Graph)
dataTrain[["Sex", "Pclass", "Survived"]].groupby(["Pclass", "Sex"]).mean().plot.bar()
plt.show()

# Relationship between Sex, Pclass, and Survived (Numerical)
print(dataTrain.groupby(["Sex", "Pclass", "Survived"])["Survived"].count())

# Relationship between Ages of passengers (Graph)
plt.figure(figsize=(12, 5))
plt.subplot(121)
dataTrain["Age"].hist(bins=70)
plt.xlabel("Age")
plt.ylabel("Num")
plt.subplot(122)
dataTrain.boxplot(column="Age", showfliers=False)
plt.show()

# Relationship between Age Groups and Survived (Graph)
ageGroups = [0, 6, 16, 35, 55, 100]
dataTrain["Age Group"] = pd.cut(dataTrain["Age"], ageGroups)
ageGroupSurvived = dataTrain.groupby("Age Group")["Survived"].mean()
ageGroupSurvived.plot.bar()
plt.show()

# Relationship between Parch and Sibling/Spouse (Graph)
fig, axis = plt.subplots(1, 2, figsize=(12, 5))
dataTrain[["Parch", "Survived"]].groupby(["Parch"]).mean().plot.bar(ax=axis[0])
dataTrain[["SibSp", "Survived"]].groupby(["SibSp"]).mean().plot.bar(ax=axis[1])
plt.show()

# Relationship between Fare and Pclass (Graph)
dataTrain.boxplot(column="Fare", by="Pclass", showfliers=False)
plt.show()

# Relationship between Survived, Sex, and Pclass (Graph)
sp = dataTrain[["Sex", "Pclass", "Survived"]].groupby(["Sex", "Pclass"], as_index=False)
predictSP = sp.mean()
fig = sns.barplot(data=predictSP, x="Sex", y="Survived", hue="Pclass", order=["male", "female"], palette="Set3")
fig.axes.set_ylabel("Survival Rate")
plt.show()
'''
#---------------------------------
# III. Data Cleaning
# -----------------

# -----------------
# 1. Fill Missing Data
# -----------------

#Fill missing Age data with predicting the values in dataTrain
ageSet = dataTrain[["Age", "Survived","Fare", "Parch", "SibSp", "Pclass"]]
ageSetNull = ageSet.loc[(dataTrain["Age"].isnull())]
ageSetNotNull = ageSet.loc[(dataTrain["Age"].notnull())]
X = ageSetNotNull.values[:, 1:]
Y = ageSetNotNull.values[:, 0]
RFR = RandomForestRegressor(n_estimators=1000, n_jobs=-1)
RFR.fit(X, Y)
predictAges = RFR.predict(ageSetNull.values[:, 1:])
dataTrain.loc[dataTrain["Age"].isnull(), ["Age"]] = predictAges
dataTrain.info()

print(dataTest.columns)

# Fill missing Age data with predicting the values in dataTest
ageSet1 = dataTest[["Age", "Parch", "SibSp", "Pclass"]]
ageSetNull1 = ageSet1.loc[(dataTest["Age"].isnull())]
ageSetNotNull1 = ageSet1.loc[(dataTest["Age"].notnull())]
X1 = ageSetNotNull1.values[:, 1:]
Y1 = ageSetNotNull1.values[:, 0]
RFR1 = RandomForestRegressor(n_estimators=1000, n_jobs=-1)
RFR1.fit(X1, Y1)
predictAges1 = RFR1.predict(ageSetNull1.values[:, 1:])
dataTest.loc[dataTest["Age"].isnull(), ["Age"]] = predictAges1
dataTest.info()

#Create a dictionary for processing(to deal with data both in dataTrain and dataTest together)
dataCleaner = pd.concat([dataTrain, dataTest], ignore_index = True)


# Fill missing Embarked data with mode
dataCleaner.Embarked[dataCleaner.Embarked.isnull()] = dataCleaner.Embarked.dropna().mode().values
# Fill missing Cabin data with "NA"
dataCleaner["Cabin"] = dataCleaner.Cabin.fillna("NA")
# Fill missing Fare data with "median"
dataCleaner['Fare'].fillna(dataCleaner['Fare'].median(), inplace=True)

# ---------------------
# 2. Create columns
# ---------------------

# Relationship between Title and Survived (Graph)
dataTrain["Title"] = dataTrain["Name"].apply(lambda x: x.split(",")[1].split(".")[0].strip())
dictTitles = {}
dictTitles.update(dict.fromkeys(["Capt", "Col", "Major", "Dr", "Rev"], "Officer"))
dictTitles.update(dict.fromkeys(["Don", "Sir", "the Countess", "Dona", "Lady"], "Royalty"))
dictTitles.update(dict.fromkeys(["Mme", "Ms", "Mrs"], "Mrs"))
dictTitles.update(dict.fromkeys(["Mlle", "Miss"], "Miss"))
dictTitles.update(dict.fromkeys(["Mr"], "Mr"))
dictTitles.update(dict.fromkeys(["Master", "Jonkheer"], "Master"))
dataTrain["Title"] = dataTrain["Title"].map(dictTitles)
sns.barplot(x="Title", y="Survived", data=dataTrain, palette="Set3")
plt.show()
#create title
def get_title(name):
    title_search = re.search(' ([A-Za-z]+)\.', name)
    # If the title exists, extract and return it.
    if title_search:
        return title_search.group(1)
    return ""
dataCleaner['Title'] = dataCleaner['Name'].apply(get_title)
dataCleaner['Title'] = dataCleaner['Title'].replace(['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
dataCleaner['Title'] = dataCleaner['Title'].replace('Mlle', 'Miss')
dataCleaner['Title'] = dataCleaner['Title'].replace('Ms', 'Miss')
dataCleaner['Title'] = dataCleaner['Title'].replace('Mme', 'Mrs')
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
dataCleaner['Title'] = dataCleaner['Title'].map(title_mapping)
dataCleaner['Title'] = dataCleaner['Title'].fillna(0)

#create Embarked
dataCleaner['Embarked'] = dataCleaner['Embarked'].map({'S': 0, 'C': 1, 'Q': 2}).astype(int)
#create FamilySize
dataCleaner['FamilySize'] = dataCleaner['SibSp'] + dataCleaner['Parch']+1

#!add picture of family size
#categorize familysize
def family_lable(n):
    if (n >= 2) & (n <= 4):
        return 2
    elif ((n > 4) & (n <= 7)) | (n == 1):
        return 1
    elif (n > 7):
        return 0
dataCleaner['FamilyBin'] = dataCleaner['FamilySize'].apply(family_lable)

Ticket_Count = dict(dataCleaner['Ticket'].value_counts())
#!add picture of ticket_Count
dataCleaner['TicketGroup'] = dataCleaner['Ticket'].apply(lambda x: Ticket_Count[x])
#categorize people who have same ticket number
def ticket_lable(n):
    if (n >= 2) & (n <= 4):
        return 2
    elif ((n > 4) & (n <= 8)) | (n == 1):
        return 1
    elif (n > 8):
        return 0
dataCleaner['TicketBin'] = dataCleaner['TicketGroup'].apply(ticket_lable)

# Relationship between Fare and Survived (Graph)
facet = sns.FacetGrid(dataTrain, hue="Survived", aspect=2)
facet.map(sns.kdeplot, "Fare", shade=True)
facet.set(xlim=(0, 200))
facet.add_legend()
plt.show()

#categorize fare
dataCleaner.loc[dataCleaner['Fare'] <= 7.91, 'Fare'] = 0
dataCleaner.loc[(dataCleaner['Fare'] > 7.91) & (dataCleaner['Fare'] <= 14.454), 'Fare'] = 1
dataCleaner.loc[(dataCleaner['Fare'] > 14.454) & (dataCleaner['Fare'] <= 31), 'Fare'] = 2
dataCleaner.loc[dataCleaner['Fare'] > 31, 'Fare'] = 3
dataCleaner['Fare'] = dataCleaner['Fare'].astype(int)

dataCleaner['Sex'] = dataCleaner['Sex'].map({'female': 0, 'male': 1}).astype(int)

#deal with outlier
dataCleaner['Surname'] = dataCleaner['Name'].apply(lambda x: x.split(',')[0].strip())
Surname_Count = dict(dataCleaner['Surname'].value_counts())
dataCleaner['FamilyGroup'] = dataCleaner['Surname'].apply(lambda x: Surname_Count[x])
Female_Child_Group = dataCleaner.loc[(dataCleaner['FamilyGroup'] >= 2) & ((dataCleaner['Age'] <= 15) | (dataCleaner['Sex'] == 0))]
Female_Child = pd.DataFrame(Female_Child_Group)
Female_Child = pd.DataFrame(Female_Child_Group.groupby('Surname')['Survived'].mean().value_counts())
Female_Child.columns = ['GroupCount']
print("Female_Child_Group",Female_Child_Group)
Male_Adult_Group = dataCleaner.loc[(dataCleaner['FamilyGroup'] >= 2) & (dataCleaner['Age'] > 15) & (dataCleaner['Sex'] == 1)]
Male_Adult = pd.DataFrame(Male_Adult_Group.groupby('Surname')['Survived'].mean().value_counts())
Male_Adult.columns = ['GroupCount']
print("Male_Adult_Group",Male_Adult_Group)

#Relationship between Age and Survived (Graph)
facet = sns.FacetGrid(dataTrain, hue="Survived", aspect=2)
facet.map(sns.kdeplot, "Age", shade=True)
facet.set(xlim=(0, dataTrain["Age"].max()))
facet.add_legend()
plt.show()

#categorize age
agebin = [0, 15, 80]
dataCleaner['AgeBin'] = pd.cut(dataCleaner['Age'], agebin, labels=[0, 1])

Female_Child_Group=Female_Child_Group.groupby('Surname')['Survived'].mean()
Dead_List=set(Female_Child_Group[Female_Child_Group.apply(lambda x:x==0)].index)
print(Dead_List)
Male_Adult_List=Male_Adult_Group.groupby('Surname')['Survived'].mean()
Survived_List=set(Male_Adult_List[Male_Adult_List.apply(lambda x:x==1)].index)
print(Survived_List)

dataTrain=dataCleaner.loc[dataCleaner['Survived'].notnull()]
dataTest=dataCleaner.loc[dataCleaner['Survived'].isnull()].drop('Survived',axis=1)
dataTest.loc[(dataTest['Surname'].apply(lambda x:x in Dead_List)),'Sex'] = 1
dataTest.loc[(dataTest['Surname'].apply(lambda x:x in Dead_List)),'AgeBin'] = 1
dataTest.loc[(dataTest['Surname'].apply(lambda x:x in Dead_List)),'Title'] = 1
dataTest.loc[(dataTest['Surname'].apply(lambda x:x in Survived_List)),'Sex'] = 0
dataTest.loc[(dataTest['Surname'].apply(lambda x:x in Survived_List)),'AgeBin'] = 0
dataTest.loc[(dataTest['Surname'].apply(lambda x:x in Survived_List)),'Title'] = 2
print(dataTest.info())



print("-"*10)
print()
# ---------------------
# 3. Drop unrelated columns
# ---------------------
drop_elements = ['PassengerId', 'Name','Ticket','TicketGroup','Age','FamilySize','Cabin', 'SibSp','Parch','FamilyGroup','Surname']
train = dataTrain.drop(drop_elements, axis = 1)
test = dataTest.drop(drop_elements, axis = 1)
print(train.isnull().sum())
print(test.isnull().sum())

print(train.info())
print(test.info())
# ---------------------
# 4.Covert the data
# ---------------------

#define y as "survived"
Target = ['Survived']

#define x and y variables(original)
dataTrain_x = ['Sex','Pclass', 'Embarked','Title','AgeBin', 'FamilyBin','TicketBin','Fare'] #pretty name/values for charts
dataTrain_x_calc = ['Sex','Pclass', 'Embarked', 'AgeBin' ,'Ticket','Title'] #coded for algorithm calculation
dataTrain_xy =  Target + dataTrain_x
print('Original X Y: ', dataTrain_xy, '\n')


#define x and y variables(remove continuous variables)
dataTrain_x_bin = ['Sex','Pclass', 'Embarked', 'Title', 'FamilyBin', 'AgeBin','TicketBin','Fare']
dataTrain_xy_bin = Target + dataTrain_x_bin
print('Bin X Y: ', dataTrain_xy_bin, '\n')


#define x and y variables(both dummy variables and continuous variables)
dataTrain_dummy = pd.get_dummies(dataTrain[dataTrain_x])
dataTrain_x_dummy = dataTrain_dummy.columns.tolist()
dataTrain_xy_dummy = Target + dataTrain_x_dummy
print('Dummy X Y: ',dataTrain_xy_dummy,'\n')

train1_x, test1_x, train1_y, test1_y = sklearn.model_selection.train_test_split(dataTrain[dataTrain_x_calc], dataTrain[Target], random_state = 0)
train1_x_bin, test1_x_bin, train1_y_bin, test1_y_bin = model_selection.train_test_split(dataTrain[dataTrain_x_bin], dataTrain[Target] , random_state = 0)
train1_x_dummy, test1_x_dummy, train1_y_dummy, test1_y_dummy = model_selection.train_test_split(dataTrain_dummy[dataTrain_x_dummy], dataTrain[Target], random_state = 0)

print("-"*80)

# -----------------------------------------------------------------
# IV. Analysis with Statistics
# ---------------------------

for x in dataTrain_x:
    if dataTrain[x].dtype != 'float64' :
        print('Survival Correlation by:', x)
        print(dataTrain[[x, Target[0]]].groupby(x, as_index=False).mean())
        print('-'*80, '\n')



#pair plots of entire dataset'
#print(dataTrain.info())
g = sns.pairplot(dataTrain,  palette = 'husl', size=1.2, diag_kind = 'kde', diag_kws=dict(shade=True), plot_kws=dict(s=10) )
g.set(xticklabels=[])
plt.show()


#correlation heatmap of dataset

def correlation_heatmap(df):
    _ , ax = plt.subplots(figsize =(14, 12))
    colormap = sns.diverging_palette(220, 10, as_cmap = True)

    _ = sns.heatmap(
        df.corr(),
        cmap = colormap,
        square=True,
        cbar_kws={'shrink':.9 },
        ax=ax,
        annot=True,
        linewidths=0.1,vmax=1.0, linecolor='white',
        annot_kws={'fontsize':12 }
    )

    plt.title('Pearson Correlation of Features', y=1.05, size=15)

correlation_heatmap(dataTrain)

plt.show()

# -----------------------------------------------------------------
# VI. Build the model
# ----------------------
# 1. Machine Learning Algorithm (MLA) Selection and Initialization
# ----------------------
MLA = [
    # Ensemble Methods
    ensemble.AdaBoostClassifier(),
    ensemble.ExtraTreesClassifier(),
    ensemble.RandomForestClassifier(),

    # Logistic Regression
    linear_model.LogisticRegressionCV(),

    # Navies Bayes
    naive_bayes.BernoulliNB(),
    naive_bayes.GaussianNB(),

    # Trees
    tree.DecisionTreeClassifier(),
    tree.ExtraTreeClassifier(),

]

cv_split = model_selection.ShuffleSplit(n_splits=10, test_size=.3, train_size=.6,
                                        random_state=0)  # run model 10x with 60/30 split intentionally leaving out 10%

# create table to compare MLA metrics
MLA_columns = ['MLA Name', 'MLA Parameters', 'MLA Train Accuracy Mean', 'MLA Test Accuracy Mean',
               'MLA Test Accuracy 3*STD', 'MLA Time']
MLA_compare = pd.DataFrame(columns=MLA_columns)

# create table to compare MLA predictions
MLA_predict = dataTrain[Target]

# index through MLA and save performance to table
row_index = 0
for alg in MLA:
    # set name and parameters
    MLA_name = alg.__class__.__name__
    MLA_compare.loc[row_index, 'MLA Name'] = MLA_name
    MLA_compare.loc[row_index, 'MLA Parameters'] = str(alg.get_params())

    # score model with cross validation: http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_validate.html#sklearn.model_selection.cross_validate
    cv_results = model_selection.cross_validate(alg, dataTrain[dataTrain_x_bin], dataTrain[Target], cv=cv_split)

    MLA_compare.loc[row_index, 'MLA Time'] = cv_results['fit_time'].mean()
    MLA_compare.loc[row_index, 'MLA Train Accuracy Mean'] = cv_results['train_score'].mean()
    MLA_compare.loc[row_index, 'MLA Test Accuracy Mean'] = cv_results['test_score'].mean()
    # if this is a non-bias random sample, then +/-3 standard deviations (std) from the mean, should statistically capture 99.7% of the subsets
    MLA_compare.loc[row_index, 'MLA Test Accuracy 3*STD'] = cv_results[
                                                                'test_score'].std() * 3  # let's know the worst that can happen!

    # save MLA predictions - see section 6 for usage
    alg.fit(dataTrain[dataTrain_x_bin], dataTrain[Target])
    MLA_predict[MLA_name] = alg.predict(dataTrain[dataTrain_x_bin])

    row_index += 1

# print and sort table
MLA_compare.sort_values(by=['MLA Test Accuracy Mean'], ascending=False, inplace=True)
print(MLA_compare)
print(MLA_compare[['MLA Name','MLA Test Accuracy Mean']])

# MLA_predict
sns.barplot(x='MLA Test Accuracy Mean', y = 'MLA Name', data = MLA_compare, color = 'm')


plt.title('Machine Learning Algorithm Accuracy Score \n')
plt.xlabel('Accuracy Score (%)')
plt.ylabel('Algorithm')
plt.show()

# ----------------------
# 2.choose RandomForest and optimize model
# ----------------------
#build model
RFC = RandomForestClassifier(random_state=0)
print('Parameters currently in use:\n')
print(RFC.get_params())

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1000, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(2, 10, num = 9)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]

# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
print("The parameters will be tuned:")
print(random_grid)

rf_random = RandomizedSearchCV(estimator = RFC, param_distributions = random_grid, n_iter = 100, cv = 5, verbose=2,
                               random_state=0, n_jobs = -1)
#Fit the model
train_features = train.drop(['Survived'], axis = 1)
rf_random.fit(train_features.values, train['Survived'].ravel())
print("The best parameter of the model is:")
print(rf_random.best_params_)

#get best one
best_random = rf_random.best_estimator_

rfc_features = best_random.feature_importances_
print("The importances of each features are:")
print(rfc_features)

RFC1 = best_random
print('Parameters currently in use:\n')
print(RFC1.get_params())
RFC1.fit(train[dataTrain_x_bin], train[Target])

#Cross Vaildation
cv_score = cross_val_score(RFC1,train[dataTrain_x_bin], train[Target], cv= 10)
print("CV Score : Mean - %.7g | Std - %.7g " % (np.mean(cv_score), np.std(cv_score)))
print("-"*80)
#----------------------------------------------------
#VII. Evaluate model(using decision tree)
#-------------------------------------
def mytree(df):
    Model = pd.DataFrame(data={'Predict': []})
    male_title = ['Master']

    for index, row in df.iterrows():

        Model.loc[index, 'Predict'] = 0

        if (df.loc[index, 'Sex'] == 0):
            Model.loc[index, 'Predict'] = 1

        if (df.loc[index, 'AgeBin'] == 0):
            Model.loc[index, 'Predict'] = 1

        if ((df.loc[index, 'Sex'] == 0) &
                (df.loc[index, 'Pclass'] == 3) &
                (df.loc[index, 'Embarked'] == 0)

        ):
            Model.loc[index, 'Predict'] = 0

        if ((df.loc[index, 'Sex'] == 1) &
                (df.loc[index, 'Title'] == 4)
        ):
            Model.loc[index, 'Predict'] = 1

    return Model

#comparsion: decision tree model
dataTest['Survived'] = mytree(test).astype(int)
print(dataTest['Survived'])
#comparsion: randomforest model
dataTest['Survived']=RFC1.predict(test[dataTrain_x_bin]).astype(int)
print(dataTest['Survived'])

#------------------
# VII. Submission
#------------------
submit = dataTest[['PassengerId','Survived']]
submit.to_csv("submit.csv", index=False)

print('Validation Data Distribution: \n', dataTest['Survived'].value_counts(normalize = True))
submit.sample(10)

print("Done")

