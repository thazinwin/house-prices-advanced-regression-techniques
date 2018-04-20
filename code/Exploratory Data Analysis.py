#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# This file provide a basic exploration of given house price dataset
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder


df_train = pd.read_csv('./../data/train.csv')
df_test = pd.read_csv('./../data/test.csv')

print("*** Extract Training Data Sample *** \n %s \n " % df_train.head())

print("*** Extract Training Data Column *** \n %s \n " % df_train.columns)

print("*** Extract Training Data Types *** \n %s \n " % df_train.info())


print("*** Extract Training Data of Similar Type Group *** \n %s \n " % df_train.columns.to_series().groupby(df_train.dtypes).groups)


print("*** Extract Training Data Summary *** \n %s \n " % df_train.describe())


train_sale_price = df_train['SalePrice']
print("*** Extract Training Sale Price *** \n %s \n " % train_sale_price)
print("*** Extract Training Sale Price Summary *** \n %s \n " % train_sale_price.describe())


#Plotting saleprice
sns.set(color_codes=True)


plt.figure(figsize=(12,5))
print("*** Basic univariate density of Sale Price ***")
sns.kdeplot(train_sale_price, shade=True);


train_correlation = df_train.corr()
print("*** Training Data Correlation Matrix *** \n %s \n " % train_correlation)


sns.set()
plt.subplots(figsize=(10,10))
print("*** Coorelation Map to analysis the how features are related ***")
sns.heatmap(train_correlation, linewidths= 1, cmap="YlGnBu");


#Top 10 Correlation Matrix with SalePrice
top_10_correlation_labels = train_correlation.nlargest(10, 'SalePrice')['SalePrice'].index
print("*** Top 10 Correlation Labels associated with Sale Price *** \n %s \n " % top_10_correlation_labels)


#Top 10 Correlation Matrix with SalePrice
top_10_correlation_list = train_correlation['SalePrice'].sort_values(axis=0,ascending=False).iloc[1:10]
print("*** Top 10 Correlation Label's value associated with Sale Price \n %s \n " % top_10_correlation_list)


fig, ax = plt.subplots(figsize=(5,5))
ax.scatter(x = df_train['GrLivArea'], y = train_sale_price)
plt.ylabel('SalePrice', fontsize=13)
plt.xlabel('GrLivArea', fontsize=13)
plt.title('Top Correlation Matrix for SalePrice', fontsize=15)
plt.show()


#Now we know the Nature of Target Value, Correlation between Target and Features. So Start Data Prep for model training.
#First Drop ID since this is not related feature of saleprice
train_ID = df_train['Id']
test_ID = df_test['Id']


#Concatenate the train and test data in same data-frame
#So that we can apply same preparation for all features 
#Take Note of train and test so that we can separate after data prep for train
ntrain = df_train.shape[0]
#Take the ‘SalePrice’ Value
targetSalePrice = df_train['SalePrice']
df_traintest = pd.concat((df_train,df_test)).reset_index(drop=True)
print("*** After Concat of train and test *** \n {} \n".format(df_traintest.shape))


#Check the missing Data
#Get percent of Null data per column which is more than ZERO percentage
df_traintest_na = df_traintest.isnull().sum()/len(df_traintest)*100
df_traintest_na =  df_traintest_na.drop(df_traintest_na[df_traintest_na==0].index).sort_values(ascending=False)
print("*** Non-null variables *** \n %s \n " % df_traintest_na)


#And then plot of Percent Missing Data by feature
features, percentage = plt.subplots(figsize=(10,8))
plt.xticks(rotation='90')
sns.barplot(x=df_traintest_na.index, y=df_traintest_na)
plt.xlabel('Features', fontsize=15)
plt.ylabel('Percentage of Missing Values', fontsize=15)
plt.title('Percentage Missing Data by Features', fontsize=15)


#Missing Data for Some Features are simply means this house don't have this feature
#So we will assign "None" 
for feature in ('PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'MasVnrType', 'MSSubClass'):
    df_traintest[feature] = df_traintest[feature].fillna('None')


#Measurement or count of Missing Features are simply means, this house don't have this feature So we will assign Zero
for feature_related in ('GarageYrBlt', 'GarageArea', 'GarageCars', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath', 'MasVnrArea'):
    df_traintest[feature_related] = df_traintest[feature_related].fillna(0)


#LotFrontage :  Linear feet of street connected to property
#Since the distance between the house and street might be the same for most house in same #neighborhood so take the mean of this feature value for same neighbourhood and fill those 
df_traintest["LotFrontage"] = df_traintest.groupby("Neighborhood")["LotFrontage"].transform(lambda x: x.fillna(x.median()))
print("*** Mean for distance between the house and street by Neighborhood *** \n %s \n " % df_traintest['MSZoning'].describe())


#MSZoning: Identifies the general zoning classification of the sale.
#Based on describe() 'RL' is the most freq, so we fill with 'RL' for those NA
df_traintest['MSZoning'] = df_traintest['MSZoning'].fillna(df_traintest['MSZoning'].mode()[0])


#Some features are not really related for SalesPrice in this context
#For Example, YrSold and MoSold can't really relate with SalePrice by themselves alone
#Since we don't have another feature that tell us what happen around 
#that house area on that year and month
#We drops some feature that not contribute the SalePrice
df_traintest = df_traintest.drop(['YrSold'], axis=1)
df_traintest = df_traintest.drop(['MoSold'], axis=1)
df_traintest = df_traintest.drop(['Utilities'], axis=1)


#Functional: Home functionality (Assume typical unless deductions are warranted)
#we take Functional 'NA' means it's typical
df_traintest["Functional"] = df_traintest["Functional"].fillna("Typ")


#Electrical: Electrical system
#Based on describe() 'SBrkr' is the most freq as it is standard circuit breakers, so we fill with 'SBrkr' for those NA
df_traintest['Electrical'] = df_traintest['Electrical'].fillna(df_traintest['Electrical'].mode()[0])
print("*** Electrical System Summary *** \n %s \n " % df_traintest['Electrical'].describe())


#Same for the following features since those have only one or two NA 
#So we fill in with the most common value
df_traintest['KitchenQual'] = df_traintest['KitchenQual'].fillna(df_traintest['KitchenQual'].mode()[0])
df_traintest['Exterior1st'] = df_traintest['Exterior1st'].fillna(df_traintest['Exterior1st'].mode()[0])
df_traintest['Exterior2nd'] = df_traintest['Exterior2nd'].fillna(df_traintest['Exterior2nd'].mode()[0])
df_traintest['SaleType'] = df_traintest['SaleType'].fillna(df_traintest['SaleType'].mode()[0])


#MSSubClass: Identifies the type of dwelling involved in the sale.
#NA for MSSubClass means simply says this house dont have the class or
#not in tht list of category so we fill with None
df_traintest['MSSubClass'] = df_traintest['MSSubClass'].fillna("None")


#Some features are numerical but they have actual meaning behind numerical
#For Example, 'MSSubClass' feature 20	means '1-STORY 1946 & NEWER ALL STYLES'
#So we havve to change these feature to category so that it will consider
#correctly when we feed the data to model

#MSSubClass=The building class
df_traintest['MSSubClass'] = df_traintest['MSSubClass'].apply(str)

# Same for 'OverallQual' and 'OverallCond' since the number applied to category
df_traintest['OverallQual'] = df_traintest['OverallQual'].astype(str)
df_traintest['OverallCond'] = df_traintest['OverallCond'].astype(str)


#We will transform all the category data to numeric value so that we can feed to model
#We decide to use the labelEncoder which will give the numeric value of each of 
#the category for all individual feature
#For Example
#So get all the Object Type Features
category_cols = [key for key in dict(df_traintest.dtypes) if dict(df_traintest.dtypes)[key] in ['object']]
for c in category_cols:
    lbl = LabelEncoder() 
    lbl.fit(list(df_traintest[c].values)) 
    df_traintest[c] = lbl.transform(list(df_traintest[c].values))


#Separate the train and test after all features applied equally
df_featured_train =  df_traintest[:ntrain]
df_featured_test = df_traintest[ntrain:]
print("After feature applied train shape : {}".format(len(df_featured_train)))
print("After feature applied train shape : {}".format(len(df_featured_test)))

