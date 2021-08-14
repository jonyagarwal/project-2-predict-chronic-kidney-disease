#PERFORM DATA PRE-PROCESSING AND PREPARE YOUR DATA FOR ANALYSIS AND MODELLING PURPOSE AS WELL.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df=pd.read_csv('kidney_disease.csv')
df.head()
columns=pd.read_csv('data_description.txt',sep='-')#our data set columns are not clear and clear columns are present in data_description .txt file and we have to use this file data in our columns name. 
columns=columns.reset_index()#it will reset index of our dataset.

columns.columns=['cols','abb_col_names']#rename the column name.
columns
df.head()

df.columns=columns['abb_col_names'].values# give all values of this particular column.
df.head()
df.dtypes
def convert_dtype(df,feature):
    df[feature]=pd.to_numeric(df[feature],errors='coerce')#it will convert datatype into numeric.and errors willl handle nan values.
    
features=['packed cell volume','white blood cell count','red blood cell count']

for feature in features:
    convert_dtype(df,feature)
df.dtypes
df.drop('id',axis=1,inplace=True)#will drop id columns.

#APPLY DATA CLEANING TECHNIQUES ON DATA AND CLEAN YOUR DATA.

df.head()
def extract_cat_num(df):#will specify catorigal and numerical data separetely.
    cat_col=[col for col in df.columns if df[col].dtype=='object']
    num_col=[col for col in df.columns if df[col].dtype!='object']
    return cat_col,num_col
cat_col,num_col=extract_cat_num(df)
cat_col
num_col

for col in cat_col:
    print('{} has {} values '.format(col,df[col].unique()))#it will return the unique values of catorigal data.
    print('\n')

df['diabetes mellitus'].replace(to_replace={'\tno':'no','tyes':'yes'},inplace=True)#it will remove dirtyness in data like remove \tno \tyes etc.
df['coronary artery disease']=df['coronary artery disease'].replace(to_replace='\tno',value='no')
df['class']=df['class'].replace(to_replace='ckd\t',value='ckd')

for col in cat_col:
    print('{} has {} values '.format(col,df[col].unique()))
    print('\n')
    
#ANALYSING DISTEIBUTION  OF EACH AND EVERY NUMERICAL COLUMN.

plt.figure(figsize=(30,20))#now we want distribution of each and every numeriacal column:-

for i,feature in enumerate(num_col):#it will iterate index as well as feature.
    plt.subplot(5,3,i+1)#beacuse num_col has 14 features so we take 5,3 matrix in subplot.
    df[feature].hist()
    plt.title(feature)
    
#CHECK LABEL DISTRIBUTION OF CATORIGAL DATA.

len(cat_col)

plt.figure(figsize=(20,20))

for i,feature in enumerate(cat_col):
    plt.subplot(4,3,i+1)#beacuse cat_col has 11 features so 4,3 matrix taken.
    sns.countplot(df[feature])#it tells how many categorical features are there in categorical columns.
sns.countplot(df['class']) #ckd chronic kidney disease.   

#CHECK HOW COLUMNS ARE CO-RELATED WITH EACH OTHER AND ITS IMPACT ON TARGET FEATURE.

plt.figure(figsize=(10,8))
df.corr()#correlation
sns.heatmap(df.corr(),annot=True) 
df.groupby(['red blood cells','class'])['red blood cell count'].agg(['count','mean','median','min','max'])
import plotly.express as px
df.columns
px.violin(df,y='red blood cell count',x='class',color='class')

#FIND RELATIONSHIP BETWEEN HAEMOGLOBIN AND PACKED CELL VOLUME

px.scatter(df,x='haemoglobin',y='packed cell volume')

#ANALYSING DISTRIBUTION OF 'RED_BLOOD CELL_COUNT' CHRONIC AS WELL AS NON CHRONIC.

sns.FacetGrid(df,hue='class')
grid=sns.FacetGrid(df,hue='class')
grid.map(sns.kdeplot,'red blood cell count')

grid=sns.FacetGrid(df,hue='class',aspect=2)#aspect change the size of kdeplot graph.
grid.map(sns.kdeplot,'red blood cell count')
grid.add_legend()#legend add margin.

#AUTOMATE YOUR ANALYSIS.

def violin(col):
    fig=px.violin(df,y=col,x='class',color='class',box=True)
    return fig.show()
    
def scatters(col1,col2):
    fig=px.scatter(df,x=col1,y=col2,color='class')
    return fig.show()
    
def kde_plot(feature):
    grid=sns.FacetGrid(df,hue='class',aspect=2)
    grid.map(sns.kdeplot,feature)
    grid.add_legend()

kde_plot('red blood cell count')

#PERFORM EXPLORATRY DATA ANALYSIS ON DATA

df.columns
kde_plot('red blood cell count')
kde_plot('haemoglobin')
scatters('red blood cell count','packed cell volume')
scatters('red blood cell count','haemoglobin')
scatters('packed cell volume','haemoglobin')
violin('red blood cell count')
violin('packed cell volume')
scatters('red blood cell count','albumin')

#PERFORM DATA CLEANING  AND DEAL WITH MISSING VALUES:-
df.isna().sum()
df.isna().sum().sort_values(ascending=False)#it returns missing values and these missing values csn fill with mean,median standard deviation.but if there is a condition that 
#in 10,000 dataset 8000 are missing values,it mean no. of missing values are very large then we cant go with mean,median,std deviation beause it affects on normal distribution.
#So instead of this use some smarter approach ie. replace missing values with random values in columns.

sns.countplot(df['red blood cells'])
data=df.copy()
data.head()
data['red blood cells'].dropna().sample()#it will give the random value of missing value.
data['red blood cells'].isnull().sum()#return all missing value in particular column red blood cells.#152
data['red blood cells'].dropna().sample(data['red blood cells'].isnull().sum())#it will give random value of all 152 missing values.
random_sample=data['red blood cells'].dropna().sample(data['red blood cells'].isnull().sum())
random_sample

data[data['red blood cells'].isnull()].index
random_sample.index#random_sample and red blood cell index are different so we have to make both index same.
random_sample.index=data[data['red blood cells'].isnull()].index#make index as same
random_sample.index
random_sample

data.loc[data['red blood cells'].isnull(),'red blood cells']=random_sample
data.head()
data['red blood cells'].isnull().sum()#now missing value is 0.
sns.countplot(data['red blood cells'])#after again plotting graph after handling missing value,then we found that there will be no change.So we can observe that random value 
#techniques of handling missing value never affect the distribution of data.

def Random_value_imputation(feature):#we fill missing value by using this function so we have not to repeat the code again and again.
    random_sample=data[feature].dropna().sample(data[feature].isnull().sum())
    random_sample.index=data[data[feature].isnull()].index
    data.loc[data[feature].isnull(),feature]=random_sample

#CHECK MISSING VALUES IN CATEGORICAL FEATURES AND NUMERICAL FEATURES & FIX IT.

data[num_col].isnull().sum()
for col in num_col:
    Random_value_imputation(col)
data[num_col].isnull().sum()
data[cat_col].isnull().sum()
Random_value_imputation(' pus cell')
data['pus cell clumps'].mode()[0]
def impute_mode(feature):
    mode=data[feature].mode()[0]
    data[feature]=data[feature].fillna(mode)
for col in cat_col:
    impute_mode(col)
data[cat_col].isnull().sum()

#APPLY FEATURE ENCODING TECHNIQUE ON DATA:-
#Machine learning doesnt understand string data so string data or categorical data will convert into numerical data.
for col in cat_col:
    print('{} has {} categories'.format(col,data[col].nunique()))
#now we have to apply label encoding like for normal it give 0 for abnormal it give 1 and so o..

from sklearn .preprocessing import LabelEncoder
le=LabelEncoder()
for col in cat_col:
    data[col]=le.fit_transform(data[col])#it tansform or covert string into numerical data.
data.head()

****label encoder is used when less no. of string feature are avalable but if we have large string data then first we find the impact of every particular dtring data on dataset 
and according to it drop some feature of string data.

#SELECT BEST FEATURE OF YOUR MODEL USING SUITABLE FEATURE IMPORTANCE TECHNIQUES.
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2#chi2 check that my probability vslue is less than 0.5 or no.

ind_col=[col for col in data.columns if col!='class']
dep_col='class'

X=data[ind_col]#independent features 
y=data[dep_col]#dependent features.

X.head()

y

ordered_rank_features=SelectKBest(score_func=chi2,k=20)
ordered_feature=ordered_rank_features.fit(X,y)

ordered_feature

ordered_feature.scores_

datascores=pd.DataFrame(ordered_feature.scores_,columns=['Score'])
datascores

dfcols=pd.DataFrame(X.columns)
dfcols

features_rank=pd.concat([dfcols,datascores],axis=1)
features_rank

features_rank.columns=['features','Score']
features_rank

features_rank.nlargest(10,'Score')

selected_columns=features_rank.nlargest(10,'Score')['features'].values
selected_columns

X_new=data[selected_columns]
X_new.head()

len(X_new)
X_new.shape#and we can see that only best features are remaining and others featires are eleminited.

#BUILD A CROSS VALIDATED MODEL & PREDICT & CHECK ACCURACY OF YOUR MODEL

from sklearn .model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X_new,y,random_state=0,test_size=0.25)
print(X_train.shape)
print(X_test.shape)
y_train.value_counts()
from xgboost import XGBClassifier
XGBClassifier()

params={                                #these all my hyper parameter.
    'learning_rate':[0.05,0.20,0.25],
    'max_depth':[5,8,10],
    'min_child_weight':[1,3,5,7],
    'gamma':[0.0,0.1,0.2,0.4],
    'colsample_bytree':[0.3,0.4,0.7]
    
    
}
from sklearn.model_selection import RandomizedSearchCV
classifier=XGBClassifier()
random_search=RandomizedSearchCV(classifier,param_distributions=params,n_iter=5,scoring='roc_auc',n_jobs=-1,cv=5,verbose=3)
random_search.fit(X_train,y_train)
random_search.best_estimator_
random_search.best_params_

***it is nothing but output of random_search.best_estimator_
classifier=XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
              colsample_bynode=1, colsample_bytree=0.3, gamma=0.4, gpu_id=-1,
              importance_type='gain', interaction_constraints='',
              learning_rate=0.05, max_delta_step=0, max_depth=5,
              min_child_weight=3, monotone_constraints='()',
              n_estimators=100, n_jobs=8, num_parallel_tree=1, random_state=0,
              reg_alpha=0, reg_lambda=1, scale_pos_weight=1, subsample=1,
              tree_method='exact', validate_parameters=1, verbosity=None)

classifier.fit(X_train,y_train)

y_pred=classifier.predict(X_test)
y_pred

from sklearn .metrics import confusion_matrix,accuracy_score
confusion_matrix(y_test,y_pred)
accuracy_score(y_test,y_pred)
    
    

