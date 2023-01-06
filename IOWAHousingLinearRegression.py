#!/usr/bin/env python
# coding: utf-8

# # Project 1: IOWA House Price Case Study

# In[1]:


# Importing necessary packages
import pandas as pd # python's data handling package
import numpy as np # python's scientific computing package
import matplotlib.pyplot as plt # python's plotting package
from sklearn.metrics import mean_squared_error as mse
from sklearn.impute import SimpleImputer # Dealing with missing observations
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
import warnings
warnings.filterwarnings('ignore')


# ## Part A: Add Lot Frontage into the Dataset
# ### Imputation Strategy 1: Mean

# In[2]:


raw_data = pd.read_csv('Original_Data.csv')
old_data = pd.read_csv('Houseprice_data_scaled.csv')
df1 = pd.DataFrame(raw_data['LotFrontage'])

# Use mean to deal with the missing observations
mean_imputer = SimpleImputer(strategy='mean')
result_mean_imputer = mean_imputer.fit_transform(df1)
LotFrontage1 = pd.DataFrame(result_mean_imputer, columns=['LotFrontage'])

# Scale LotFrontage and add it to the dataset
LotFrontage_scaled1 = LotFrontage1[['LotFrontage']]
LotFrontage_scaled1 = (LotFrontage_scaled1 - LotFrontage_scaled1.mean()) / LotFrontage_scaled1.std()
result1 = pd.concat([old_data, LotFrontage_scaled1], axis=1)
result1


# In[3]:


# First 1800 data items are training set; the next 600 are the validation set
train1 = result1.iloc[:1800] 
validation1 = result1.iloc[1800:2400]
test1 = result1.iloc[2400:2908]

# Creating the "X" and "y" variables. We drop sale price from "X"
X_train1, X_validation1, X_test1 = train1.drop('Sale Price', axis=1), validation1.drop('Sale Price', axis=1), test1.drop('Sale Price', axis=1)
y_train1, y_validation1, y_test1 = train1[['Sale Price']], validation1[['Sale Price']], test1[['Sale Price']] 


# ### Linear Regression

# In[4]:


lr=LinearRegression()
lr.fit(X_train1,y_train1)

# Create dataFrame with corresponding feature and its respective coefficients
coeffs1 = pd.DataFrame([['intercept'] + list(X_train1.columns), list(lr.intercept_) + list(lr.coef_[0])]).transpose().set_index(0)

# Find the mse of each set by the regression 
pred=lr.predict(X_train1)
print(f'The mse for training set is {mse(y_train1,pred)}.')

pred=lr.predict(X_validation1)
print(f'The mse for validation set is {mse(y_validation1,pred)}.')

pred=lr.predict(X_test1)
print(f'The mse for test set is {mse(y_test1,pred)}.')


# ### Ridge Regression

# In[5]:


# The alpha used by Python's ridge should be the lambda in Hull's book times the number of observations
alphas=[0.01*1800, 0.02*1800, 0.03*1800, 0.04*1800, 0.05*1800, 0.075*1800,0.1*1800,0.2*1800, 0.6*1800, 1.0*1800]
mses=[]
for alpha in alphas:
    ridge=Ridge(alpha=alpha)
    ridge.fit(X_train1,y_train1)
    pred=ridge.predict(X_validation1)
    mses.append(mse(y_validation1,pred))
    print(mse(y_validation1,pred))


# In[6]:


plt.plot(alphas, mses)
plt.xlabel('alphas')
plt.ylabel('Mean Squared Error')
plt.title('Ridge Results for Validation Set')


# ### Lasso Regression

# In[7]:


# We consider different lambda values. The alphas are half the lambdas.
alphas=[0/2, 0.01/2, 0.02/2, 0.03/2, 0.04/2, 0.05/2, 0.075/2, 0.1/2]
mses=[]
for alpha in alphas:
    lasso=Lasso(alpha=alpha)
    lasso.fit(X_train1,y_train1)
    pred=lasso.predict(X_validation1)
    mses.append(mse(y_validation1,pred))
    print(mse(y_validation1, pred))


# In[8]:


plt.plot(alphas, mses)
plt.xlabel('alphas')
plt.ylabel('Mean Squared Error')
plt.title('Lasso Results for Validation Set')


# In[9]:


# Calculate mse for test set when Hull's lambda =0.04
alpha=0.04/2
lasso=Lasso(alpha=alpha)
lasso.fit(X_train1,y_train1)
pred=lasso.predict(X_test1)
print(mse(y_test1,pred))


# ### Imputation Strategy 2: Median

# In[10]:


# Use median to deal with the missing observations
median_imputer = SimpleImputer(strategy='median')
result_median_imputer = median_imputer.fit_transform(df1)
LotFrontage2 = pd.DataFrame(result_median_imputer, columns=['LotFrontage'])

# Scale LotFrontage and add it to the dataset
LotFrontage_scaled2 = LotFrontage2[['LotFrontage']]
LotFrontage_scaled2 = (LotFrontage_scaled2 - LotFrontage_scaled2.mean()) / LotFrontage_scaled2.std()
result2 = pd.concat([old_data, LotFrontage_scaled2], axis=1)
result2


# In[11]:


# First 1800 data items are training set; the next 600 are the validation set
train2 = result2.iloc[:1800] 
validation2 = result2.iloc[1800:2400]
test2 = result2.iloc[2400:2908]

# Creating the "X" and "y" variables. We drop sale price from "X"
X_train2, X_validation2, X_test2 = train2.drop('Sale Price', axis=1), validation2.drop('Sale Price', axis=1), test2.drop('Sale Price', axis=1)
y_train2, y_validation2, y_test2 = train2[['Sale Price']], validation2[['Sale Price']], test2[['Sale Price']] 


# ### Linear Regression

# In[12]:


lr=LinearRegression()
lr.fit(X_train2,y_train2)

# Create dataFrame with corresponding feature and its respective coefficients
coeffs2 = pd.DataFrame([['intercept'] + list(X_train2.columns), list(lr.intercept_) + list(lr.coef_[0])]).transpose().set_index(0)

# Find the mse of each set by the regression 
pred=lr.predict(X_train2)
print(f'The mse for training set is {mse(y_train2,pred)}.')

pred=lr.predict(X_validation2)
print(f'The mse for validation set is {mse(y_validation2,pred)}.')

pred=lr.predict(X_test2)
print(f'The mse for test set is {mse(y_test2,pred)}.')


# ### Ridge Regression

# In[13]:


# The alpha used by Python's ridge should be the lambda in Hull's book times the number of observations
alphas=[0.01*1800, 0.02*1800, 0.03*1800, 0.04*1800, 0.05*1800, 0.075*1800,0.1*1800,0.2*1800, 0.6*1800, 1.0*1800]
mses=[]
for alpha in alphas:
    ridge=Ridge(alpha=alpha)
    ridge.fit(X_train2,y_train2)
    pred=ridge.predict(X_validation2)
    mses.append(mse(y_validation2,pred))
    print(mse(y_validation2,pred))


# In[14]:


plt.plot(alphas, mses)
plt.xlabel('alphas')
plt.ylabel('Mean Squared Error')
plt.title('Ridge Results for Validation Set')


# ### Lasso Regression

# In[15]:


# We consider different lambda values. The alphas are half the lambdas.
alphas=[0/2, 0.01/2, 0.02/2, 0.03/2, 0.04/2, 0.05/2, 0.075/2, 0.1/2]
mses=[]
for alpha in alphas:
    lasso=Lasso(alpha=alpha)
    lasso.fit(X_train2,y_train2)
    pred=lasso.predict(X_validation2)
    mses.append(mse(y_validation2,pred))
    print(mse(y_validation2, pred))


# In[16]:


plt.plot(alphas, mses)
plt.xlabel('alphas')
plt.ylabel('Mean Squared Error')
plt.title('Lasso Results for Validation Set')


# In[17]:


# Calculate mse for test set when Hull's lambda =0.04
alpha=0.04/2
lasso=Lasso(alpha=alpha)
lasso.fit(X_train2,y_train2)
pred=lasso.predict(X_test2)
print(mse(y_test2,pred))


# ### Imputation Strategy 3: Most Frequent

# In[18]:


# Use most frequent to deal with the missing observations
most_frequent_imputer = SimpleImputer(strategy='most_frequent')
result_most_frequent_imputer = most_frequent_imputer.fit_transform(df1)
LotFrontage3 = pd.DataFrame(result_most_frequent_imputer, columns=['LotFrontage'])

# Scale LotFrontage and add it to the dataset
LotFrontage_scaled3 = LotFrontage3[['LotFrontage']]
LotFrontage_scaled3 = (LotFrontage_scaled3 - LotFrontage_scaled3.mean()) / LotFrontage_scaled3.std()
result3 = pd.concat([old_data, LotFrontage_scaled3], axis=1)
result3


# In[19]:


# First 1800 data items are training set; the next 600 are the validation set
train3 = result3.iloc[:1800] 
validation3 = result3.iloc[1800:2400]
test3 = result3.iloc[2400:2908]

# Creating the "X" and "y" variables. We drop sale price from "X"
X_train3, X_validation3, X_test3 = train3.drop('Sale Price', axis=1), validation3.drop('Sale Price', axis=1), test3.drop('Sale Price', axis=1)
y_train3, y_validation3, y_test3 = train3[['Sale Price']], validation3[['Sale Price']], test3[['Sale Price']] 


# ### Linear Regression

# In[20]:


lr=LinearRegression()
lr.fit(X_train3,y_train3)

# Create dataFrame with corresponding feature and its respective coefficients
coeffs3 = pd.DataFrame([['intercept'] + list(X_train3.columns), list(lr.intercept_) + list(lr.coef_[0])]).transpose().set_index(0)

# Find the mse of each set by the regression 
pred=lr.predict(X_train3)
print(f'The mse for training set is {mse(y_train3,pred)}.')

pred=lr.predict(X_validation3)
print(f'The mse for validation set is {mse(y_validation3,pred)}.')

pred=lr.predict(X_test3)
print(f'The mse for test set is {mse(y_test3,pred)}.')


# ### Ridge Regression

# In[21]:


# The alpha used by Python's ridge should be the lambda in Hull's book times the number of observations
alphas=[0.01*1800, 0.02*1800, 0.03*1800, 0.04*1800, 0.05*1800, 0.075*1800,0.1*1800,0.2*1800, 0.6*1800, 1.0*1800]
mses=[]
for alpha in alphas:
    ridge=Ridge(alpha=alpha)
    ridge.fit(X_train3,y_train3)
    pred=ridge.predict(X_validation3)
    mses.append(mse(y_validation3,pred))
    print(mse(y_validation3,pred))


# In[22]:


plt.plot(alphas, mses)
plt.xlabel('alphas')
plt.ylabel('Mean Squared Error')
plt.title('Ridge Results for Validation Set')


# ### Lasso Rgression

# In[23]:


# We consider different lambda values. The alphas are half the lambdas.
alphas=[0/2, 0.01/2, 0.02/2, 0.03/2, 0.04/2, 0.05/2, 0.075/2, 0.1/2]
mses=[]
for alpha in alphas:
    lasso=Lasso(alpha=alpha)
    lasso.fit(X_train3,y_train3)
    pred=lasso.predict(X_validation3)
    mses.append(mse(y_validation3,pred))
    print(mse(y_validation3, pred))


# In[24]:


plt.plot(alphas, mses)
plt.xlabel('alphas')
plt.ylabel('Mean Squared Error')
plt.title('Lasso Results for Validation Set')


# In[25]:


# Calculate mse for test set when Hull's lambda =0.04
alpha=0.04/2
lasso=Lasso(alpha=alpha)
lasso.fit(X_train3,y_train3)
pred=lasso.predict(X_test3)
print(mse(y_test3,pred))


# ## Part B: Add Lot Shape into the Dataset

# In[26]:


# No missing observations in LotShape, no need to use the imputation appoarch
# Get dummy variables from LotShape
df2 = pd.DataFrame(raw_data['LotShape'])
LotShape = pd.get_dummies(df2)

# Scale LotShape and add it to the dataset
LotShape_scaled = LotShape[['LotShape_IR1', 'LotShape_IR2', 'LotShape_IR3', 'LotShape_Reg']]
LotShape_scaled = (LotShape_scaled - LotShape_scaled.mean()) / LotShape_scaled.std()
result4 = pd.concat([result3, LotShape_scaled], axis=1)
result4


# In[27]:


# First 1800 data items are training set; the next 600 are the validation set
train4 = result4.iloc[:1800] 
validation4 = result4.iloc[1800:2400]
test4 = result4.iloc[2400:2908]

# Creating the "X" and "y" variables. We drop sale price from "X"
X_train4, X_validation4, X_test4 = train4.drop('Sale Price', axis=1), validation4.drop('Sale Price', axis=1), test4.drop('Sale Price', axis=1)
y_train4, y_validation4, y_test4 = train4[['Sale Price']], validation4[['Sale Price']], test4[['Sale Price']] 


# ### Linear Regression

# In[28]:


lr=LinearRegression()
lr.fit(X_train4,y_train4)

# Create dataFrame with corresponding feature and its respective coefficients
coeffs4 = pd.DataFrame([['intercept'] + list(X_train4.columns), list(lr.intercept_) + list(lr.coef_[0])]).transpose().set_index(0)

# Find the mse of each set by the regression 
pred=lr.predict(X_train4)
print(f'The mse for training set is {mse(y_train4,pred)}.')

pred=lr.predict(X_validation4)
print(f'The mse for validation set is {mse(y_validation4,pred)}.')

pred=lr.predict(X_test4)
print(f'The mse for test set is {mse(y_test4,pred)}.')


# ### Ridge Regression

# In[29]:


# The alpha used by Python's ridge should be the lambda in Hull's book times the number of observations
alphas=[0.01*1800, 0.02*1800, 0.03*1800, 0.04*1800, 0.05*1800, 0.075*1800,0.1*1800,0.2*1800, 0.6*1800, 1.0*1800]
mses=[]
for alpha in alphas:
    ridge=Ridge(alpha=alpha)
    ridge.fit(X_train4,y_train4)
    pred=ridge.predict(X_validation4)
    mses.append(mse(y_validation4,pred))
    print(mse(y_validation4,pred))


# In[30]:


plt.plot(alphas, mses)
plt.xlabel('alphas')
plt.ylabel('Mean Squared Error')
plt.title('Ridge Results for Validation Set')


# ### Lasso Regression

# In[31]:


# We consider different lambda values. The alphas are half the lambdas.
alphas=[0/2, 0.01/2, 0.02/2, 0.03/2, 0.04/2, 0.05/2, 0.075/2, 0.1/2]
mses=[]
for alpha in alphas:
    lasso=Lasso(alpha=alpha)
    lasso.fit(X_train4,y_train4)
    pred=lasso.predict(X_validation4)
    mses.append(mse(y_validation4,pred))
    print(mse(y_validation4, pred))


# In[32]:


plt.plot(alphas, mses)
plt.xlabel('alphas')
plt.ylabel('Mean Squared Error')
plt.title('Lasso Results for Validation Set')


# In[33]:


# Calculate mse for test set when Hull's lambda =0.04
alpha=0.04/2
lasso=Lasso(alpha=alpha)
lasso.fit(X_train4,y_train4)
pred=lasso.predict(X_test4)
print(mse(y_test4,pred))


# ## Part C: Add Central Air and Exterior Quality to the Dataset

# In[34]:


# No missing observations in CentralAir and OverallQual, no need to use the imputation appoarch
# Get dummy variables from CentralAir
df3 = pd.DataFrame(raw_data['CentralAir'])
CentralAir = pd.get_dummies(df3)

# Scale CentralAir and add it to the dataset
CentralAir_scaled = CentralAir[['CentralAir_N', 'CentralAir_Y']]
CentralAir_scaled = (CentralAir_scaled - CentralAir_scaled.mean()) / CentralAir_scaled.std()
result5 = pd.concat([result4, CentralAir_scaled], axis=1)

# Get dummy variables from ExterQual
df4 = pd.DataFrame(raw_data['ExterQual'])
ExterQual = pd.get_dummies(df4)

# Scale ExterQual and add it to the dataset
ExterQual_scaled = ExterQual[['ExterQual_Ex', 'ExterQual_Fa', 'ExterQual_Gd', 'ExterQual_TA']]
ExterQual_scaled = (ExterQual_scaled - ExterQual_scaled.mean()) / ExterQual_scaled.std()
result6 = pd.concat([result5, ExterQual_scaled], axis=1)
result6


# In[35]:


# First 1800 data items are training set; the next 600 are the validation set
train6 = result6.iloc[:1800] 
validation6 = result6.iloc[1800:2400]
test6 = result6.iloc[2400:2908]

# Creating the "X" and "y" variables. We drop sale price from "X"
X_train6, X_validation6, X_test6 = train6.drop('Sale Price', axis=1), validation6.drop('Sale Price', axis=1), test6.drop('Sale Price', axis=1)
y_train6, y_validation6, y_test6 = train6[['Sale Price']], validation6[['Sale Price']], test6[['Sale Price']] 


# ### Linear Regression

# In[36]:


lr=LinearRegression()
lr.fit(X_train6,y_train6)

# Create dataFrame with corresponding feature and its respective coefficients
coeffs6 = pd.DataFrame([['intercept'] + list(X_train6.columns), list(lr.intercept_) + list(lr.coef_[0])]).transpose().set_index(0)

# Find the mse of each set by the regression 
pred=lr.predict(X_train6)
print(f'The mse for training set is {mse(y_train6,pred)}.')

pred=lr.predict(X_validation6)
print(f'The mse for validation set is {mse(y_validation6,pred)}.')

pred=lr.predict(X_test6)
print(f'The mse for test set is {mse(y_test6,pred)}.')


# ### Ridge Regression

# In[37]:


# The alpha used by Python's ridge should be the lambda in Hull's book times the number of observations
alphas=[0.01*1800, 0.02*1800, 0.03*1800, 0.04*1800, 0.05*1800, 0.075*1800,0.1*1800,0.2*1800, 0.6*1800, 1.0*1800]
mses=[]
for alpha in alphas:
    ridge=Ridge(alpha=alpha)
    ridge.fit(X_train6,y_train6)
    pred=ridge.predict(X_validation6)
    mses.append(mse(y_validation6,pred))
    print(mse(y_validation6,pred))


# In[38]:


plt.plot(alphas, mses)
plt.xlabel('alphas')
plt.ylabel('Mean Squared Error')
plt.title('Ridge Results for Validation Set')


# ### Lasso Regression

# In[39]:


# We consider different lambda values. The alphas are half the lambdas.
alphas=[0/2, 0.01/2, 0.02/2, 0.03/2, 0.04/2, 0.05/2, 0.075/2, 0.1/2]
mses=[]
for alpha in alphas:
    lasso=Lasso(alpha=alpha)
    lasso.fit(X_train6,y_train6)
    pred=lasso.predict(X_validation6)
    mses.append(mse(y_validation6,pred))
    print(mse(y_validation6, pred))


# In[40]:


plt.plot(alphas, mses)
plt.xlabel('alphas')
plt.ylabel('Mean Squared Error')
plt.title('Lasso Results for Validation Set')


# In[41]:


# Calculate mse for test set when Hull's lambda =0.04
alpha=0.04/2
lasso=Lasso(alpha=alpha)
lasso.fit(X_train6,y_train6)
pred=lasso.predict(X_test6)
print(mse(y_test6,pred))


# ## Part D: Random Shuffle

# In[42]:


# Shuffle the dataset with 4 additional features and reset the index
result7 = result6.sample(frac=1, random_state=0).reset_index(drop=True)
result7


# In[43]:


# First 1800 data items are training set; the next 600 are the validation set
train7 = result7.iloc[:1800] 
validation7 = result7.iloc[1800:2400]
test7 = result7.iloc[2400:2908]

# Creating the "X" and "y" variables. We drop sale price from "X"
X_train7, X_validation7, X_test7 = train7.drop('Sale Price', axis=1), validation7.drop('Sale Price', axis=1), test7.drop('Sale Price', axis=1)
y_train7, y_validation7, y_test7 = train7[['Sale Price']], validation7[['Sale Price']], test7[['Sale Price']] 


# ### Linear Regression

# In[44]:


lr=LinearRegression()
lr.fit(X_train7,y_train7)

# Create dataFrame with corresponding feature and its respective coefficients
coeffs7 = pd.DataFrame([['intercept'] + list(X_train7.columns), list(lr.intercept_) + list(lr.coef_[0])]).transpose().set_index(0)

# Find the mse of each set by the regression 
pred=lr.predict(X_train7)
print(f'The mse for training set is {mse(y_train7,pred)}.')

pred=lr.predict(X_validation7)
print(f'The mse for validation set is {mse(y_validation7,pred)}.')

pred=lr.predict(X_test7)
print(f'The mse for test set is {mse(y_test7,pred)}.')


# ### Ridge Regression

# In[45]:


# The alpha used by Python's ridge should be the lambda in Hull's book times the number of observations
alphas=[0.01*1800, 0.02*1800, 0.03*1800, 0.04*1800, 0.05*1800, 0.075*1800,0.1*1800,0.2*1800, 0.6*1800, 1.0*1800]
mses=[]
for alpha in alphas:
    ridge=Ridge(alpha=alpha)
    ridge.fit(X_train7,y_train7)
    pred=ridge.predict(X_validation7)
    mses.append(mse(y_validation7,pred))
    print(mse(y_validation7,pred))


# In[46]:


plt.plot(alphas, mses)
plt.xlabel('alphas')
plt.ylabel('Mean Squared Error')
plt.title('Ridge Results for Validation Set')


# ### Lasso Regression

# In[47]:


# We consider different lambda values. The alphas are half the lambdas.
alphas=[0/2, 0.01/2, 0.02/2, 0.03/2, 0.04/2, 0.05/2, 0.075/2, 0.1/2]
mses=[]
for alpha in alphas:
    lasso=Lasso(alpha=alpha)
    lasso.fit(X_train7,y_train7)
    pred=lasso.predict(X_validation7)
    mses.append(mse(y_validation7,pred))
    print(mse(y_validation7, pred))


# In[48]:


plt.plot(alphas, mses)
plt.xlabel('alphas')
plt.ylabel('Mean Squared Error')
plt.title('Lasso Results for Validation Set')


# In[49]:


# Calculate mse for test set when Hull's lambda =0.04
alpha=0.04/2
lasso=Lasso(alpha=alpha)
lasso.fit(X_train7,y_train7)
pred=lasso.predict(X_test7)
print(mse(y_test7,pred))

