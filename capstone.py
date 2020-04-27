#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 18:51:55 2020

@author: tfogarty
"""
import pandas as pd
import numpy as np
df = pd.read_csv('lindsey_clean_april_7.csv')

schools = pd.unique(df.school_id)
trainset = pd.DataFrame(columns=df.columns)
testset = pd.DataFrame(columns=df.columns)

for i in range(len(schools)):
    sch = schools[i]
    schset = df[df.school_id == sch]
    perc = round(len(schset.school_id)*0.20)
    train_index = np.random.choice(schset.index, replace=False, size=perc)
    train = schset[schset.index.isin(train_index)]
    trainset = trainset.append(train)
    test = schset[~schset.index.isin(train_index)]
    testset = testset.append(test)

trainset = trainset.astype('float')
testset = testset.astype('float')

trainset = trainset.reset_index()
trainset['anx_dep'] = 0.0
for i in range(len(trainset.anxiety)):
    if (trainset.anxiety[i]+trainset.depression[i] > 0):
        trainset.anx_dep[i] = 1.0
testset = testset.reset_index()
testset['anx_dep'] = 0.0
for i in range(len(testset.anxiety)):
    if (testset.anxiety[i]+testset.depression[i] > 0):
        testset.anx_dep[i] = 1.0

trainset.to_csv('cleaned_trainingdata.csv',index=True,header=True)
testset.to_csv('cleaned_testingdata.csv',index=True,header=True)

trainset = trainset.drop('Unnamed: 0', axis=1)
testset = testset.drop('Unnamed: 0', axis=1)

train_means = trainset.groupby('school_id').mean()
train_means.to_csv('variable_means.csv', index=True, header=True)

test_means = testset.groupby('school_id').mean()
test_means.to_csv('model_means.csv', index=True, header=True)

train_means = train_means.drop('index',axis=1)
test_means = test_means.drop('index',axis=1)

train_means['safety'] = (train_means.daytime_campus_safety + train_means.nighttime_campus_safety+train_means.daytime_community_safety + train_means.nighttime_community_safety)/4
train_means = train_means.drop(['daytime_campus_safety','daytime_community_safety','nighttime_campus_safety','nighttime_community_safety'], axis =1)

import seaborn as sns
corr = train_means.corr()
plt.figure(figsize = (16,16))
ax = sns.heatmap(
    corr, 
    vmin=-1, vmax=1, center=0,
    cmap=sns.diverging_palette(20, 220, n=200),
    square=True,xticklabels=True, yticklabels=True
)
ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=45,
    horizontalalignment='right'
)

train = pd.read_csv('variable_means.csv')
test = pd.read_csv('model_means.csv')

train_y = train.anx_dep
train_x = train.drop(['anx_dep','anxiety','depression'], axis=1)

test_y = test.anx_dep
test_x = test.drop(['anx_dep','anxiety','depression'] , axis=1)


from sklearn.linear_model import LinearRegression

lm = LinearRegression()
lm.fit(train_x, train_y)
pd.DataFrame(lm.coef_,train_x.columns,columns=['Coeff']).sort_values(by='Coeff')

import statsmodels.api as sm
mod = sm.OLS(train_y,train_x)
fii = mod.fit()
p_values = fii.summary2().tables[1]['P>|t|']

sig_p = p_values[p_values <= 0.1].index
test_x = test_x[sig_p]
lm.fit(test_x,test_y)
pd.DataFrame(lm.coef_,test_x.columns,columns=['Coeff']).sort_values(by='Coeff')

mod = sm.OLS(test_y,test_x)
fii = mod.fit()
p_values = fii.summary2().tables[1]['P>|t|']