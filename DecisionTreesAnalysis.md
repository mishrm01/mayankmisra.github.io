---
layout: page
title: DecisionTreesAnalysis
category : Data Science
tagline: "Supporting tagline"
tags : [edav, data exploration, visualization, graphs, lessons, python, statistics]
---

# Decision Trees

## Problem 1: Decision tree on the Titanic dataset

### Initializing requiered libraries for all solutions

In[447]:

```
import numpy as np
import pandas as pd
import pandas.tools.rplot as rplot
import pylab as pl
import matplotlib as mpl
import matplotlib.pyplot as plt
import math
```

### Initialize iPython hooks to be able to embed plots inline in the document

In[448]:

```
%matplotlib inline 
pd.set_option('display.notebook_repr_html', False)
pd.set_option('display.max_columns', 20)
pd.set_option('display.max_rows', 25)
```

#### Get Data, Define polynomial design matirx, plot the regression and define the cost function

## Problem 1: Titanic1 dataset

Titanic1 dataset records 4 features for 2,201 examples.

Open the dataset in Python. Read the description of the data.

### 1.1 Extract Data

In[449]:

```
#set data source path
path = '/Users/mayank/Dropbox/Columbia University/Machine Learning/Lecture Notes/hw3/Titanic1.csv'
#set column names as the data doesn't have headers
headers = ['Class', 'Age', 'Sex', 'Survived']  
#get the data, specify that there are no headers as the default will read first row as headers
getData = pd.read_csv(path, header=0, names=headers)
```

In[450]:

```
#confirm data extration
getData.head()
```




      Class    Age   Sex Survived
    0   1st  adult  male      yes
    1   1st  adult  male      yes
    2   1st  adult  male      yes
    3   1st  adult  male      yes
    4   1st  adult  male      yes
    
    [5 rows x 4 columns]



### Explore the dataset - Understand data statistics

In[451]:

```
getData.groupby("Survived").describe()
```




                    Class    Age   Sex Survived
    Survived                                   
    no       count   1490   1490  1490     1490
             unique     4      2     2        1
             top     crew  adult  male       no
             freq     673   1438  1364     1490
    yes      count    711    711   711      711
             unique     4      2     2        1
             top     crew  adult  male      yes
             freq     212    654   367      711
    
    [8 rows x 4 columns]



In[452]:

```
getData.groupby("Age").describe()
```




                 Class    Age   Sex Survived
    Age                                     
    adult count   2092   2092  2092     2092
          unique     4      1     2        2
          top     crew  adult  male       no
          freq     885   2092  1667     1438
    child count    109    109   109      109
          unique     3      1     2        2
          top      3rd  child  male      yes
          freq      79    109    64       57
    
    [8 rows x 4 columns]



In[453]:

```
getData.groupby("Sex").describe()
```




                  Class    Age     Sex Survived
    Sex                                        
    female count    470    470     470      470
           unique     4      2       1        2
           top      3rd  adult  female      yes
           freq     196    425     470      344
    male   count   1731   1731    1731     1731
           unique     4      2       1        2
           top     crew  adult    male       no
           freq     862   1667    1731     1364
    
    [8 rows x 4 columns]



In[454]:

```
getData.groupby("Class").describe()
```




                 Class    Age   Sex Survived
    Class                                   
    1st   count    325    325   325      325
          unique     1      2     2        2
          top      1st  adult  male      yes
          freq     325    319   180      203
    2nd   count    285    285   285      285
          unique     1      2     2        2
          top      2nd  adult  male       no
          freq     285    261   179      167
    3rd   count    706    706   706      706
          unique     1      2     2        2
          top      3rd  adult  male       no
          freq     706    627   510      528
    crew  count    885    885   885      885
          unique     1      1     2        2
          top     crew  adult  male       no
          freq     885    885   862      673
    
    [16 rows x 4 columns]



In[455]:

```
#visually explore the datafrom pandas.tools.plotting import scatter_matrix

#pd.scatter_matrix(getData, alpha=0.2, figsize=(3, 3))
```

### 1.1b DELETE LATER Enrich Data - Append new column to translate string to numerical values

In[456]:

```
#Insert a column 'newLabel' to traslate string values in 'cLabel' to numberical vlaues
#Initialize the column with numberical value 0
#getData.insert(5, 'newLabel', 0)

#Insert values in newLabel based ont he value in cLabel
#getData['flagSurvived'] = getData.apply(lambda row: (1
                                               #if row['Survived']=='yes'
                                               #else 2), axis=1)
#getData.head()
```

### 1.2 Split the data into a training and test set

Keep 66% for training and 34% for testing. Make sure the proportion positive to negative is maintained similar in training and testing.

In[457]:

```
survivedTrue = getData.ix[getData['Survived'] == 'yes']
survivedTrue
survivedTrueArray = survivedTrue.values
print survivedTrueArray
countTrue = len(survivedTrueArray)
countTrue
```


    [['1st' 'adult' 'male' 'yes']
     ['1st' 'adult' 'male' 'yes']
     ['1st' 'adult' 'male' 'yes']
     ..., 
     ['crew' 'adult' 'female' 'yes']
     ['crew' 'adult' 'female' 'yes']
     ['crew' 'adult' 'female' 'yes']]




    711



In[458]:

```
survivedFalse = getData.ix[getData['Survived'] == 'no']
survivedFalseArray = survivedFalse.values
print survivedFalseArray
countFalse = len(survivedFalseArray)
countFalse
```


    [['1st' 'adult' 'male' 'no']
     ['1st' 'adult' 'male' 'no']
     ['1st' 'adult' 'male' 'no']
     ..., 
     ['crew' 'adult' 'female' 'no']
     ['crew' 'adult' 'female' 'no']
     ['crew' 'adult' 'female' 'no']]




    1490



In[459]:

```
# calulate row count for 66% 'yes' records
x = round(.66*(countTrue))
print x
# calulate row count for 66% 'no' records
y = round(.66*(countFalse))
print y
```


    469.0
    983.0


In[460]:

```
trainTrue = survivedTrueArray[:(x)]
#print trainTrue
len(trainTrue)

trainFalse = survivedFalseArray[:y]
#print trainFalse
len(trainFalse)

train = np.concatenate((trainTrue, trainFalse),0)
len(train)
```




    1452



In[461]:

```
testTrue = survivedTrueArray[:(countTrue-x)]
#print testTrue
len(testTrue)

testFalse = survivedFalseArray[:(countFalse-y)]
#print testFalse
len(testFalse)

test = np.concatenate((testTrue, testFalse),0)
len(test)
```




    749



In[462]:

```
column_names = ['Class', 'Age', 'Sex', 'Survived']
train = pd.DataFrame(train, columns=column_names)
train.ix[train['Survived'] == 'yes']
```




       Class    Age   Sex Survived
    0    1st  adult  male      yes
    1    1st  adult  male      yes
    2    1st  adult  male      yes
    3    1st  adult  male      yes
    4    1st  adult  male      yes
    5    1st  adult  male      yes
    6    1st  adult  male      yes
    7    1st  adult  male      yes
    8    1st  adult  male      yes
    9    1st  adult  male      yes
    10   1st  adult  male      yes
    11   1st  adult  male      yes
    12   1st  adult  male      yes
    13   1st  adult  male      yes
    14   1st  adult  male      yes
    15   1st  adult  male      yes
    16   1st  adult  male      yes
    17   1st  adult  male      yes
    18   1st  adult  male      yes
    19   1st  adult  male      yes
    20   1st  adult  male      yes
    21   1st  adult  male      yes
    22   1st  adult  male      yes
    23   1st  adult  male      yes
    24   1st  adult  male      yes
         ...    ...   ...      ...
    
    [469 rows x 4 columns]



In[463]:

```
test = pd.DataFrame(test, columns=column_names)
test.ix[test['Survived'] == 'yes']
```




       Class    Age   Sex Survived
    0    1st  adult  male      yes
    1    1st  adult  male      yes
    2    1st  adult  male      yes
    3    1st  adult  male      yes
    4    1st  adult  male      yes
    5    1st  adult  male      yes
    6    1st  adult  male      yes
    7    1st  adult  male      yes
    8    1st  adult  male      yes
    9    1st  adult  male      yes
    10   1st  adult  male      yes
    11   1st  adult  male      yes
    12   1st  adult  male      yes
    13   1st  adult  male      yes
    14   1st  adult  male      yes
    15   1st  adult  male      yes
    16   1st  adult  male      yes
    17   1st  adult  male      yes
    18   1st  adult  male      yes
    19   1st  adult  male      yes
    20   1st  adult  male      yes
    21   1st  adult  male      yes
    22   1st  adult  male      yes
    23   1st  adult  male      yes
    24   1st  adult  male      yes
         ...    ...   ...      ...
    
    [242 rows x 4 columns]



### 1.3  Rrecursive python function decision tree 

In[464]:

```
#clculate entropy for the root
#count numbers of rows that have yes/no for the train datset
trainTotalCount = float(len(train))
print trainTotalCount

trainTrueCount = float(len(train.ix[train['Survived'] == 'yes']))
print trainTrueCount
probTrainTrue = round((trainTrueCount/trainTotalCount)*100)/100
print probTrainTrue

trainFalseCount = float(len(train.ix[train['Survived'] == 'no']))
print trainFalseCount
probTrainFalse = round((trainFalseCount/trainTotalCount)*100)/100
print probTrainFalse

entropyRoot = -probTrainTrue * math.log((probTrainTrue), 2) - probTrainFalse * math.log((probTrainFalse), 2)
entropyRoot
```


    1452.0
    469.0
    0.32
    983.0
    0.68




    0.904381457724494



In[465]:

```
#clculate entropy for the Class - Type 1st
#count numbers of rows that have yes/no for feature 'Class' where Class value is 1st

trainClassValues = train.Class.unique()
print trainClassValues

trainClass1st = train.ix[train['Class'] == '1st']
#print trainClass

trainClass1stTotal = float(len(trainClass1st))
print trainClass1stTotal

trainClass1stTrue = float(len(trainClass1st.ix[trainClass1st['Survived'] == 'yes']))
print trainClass1stTrue

probTrainClass1stTrue = round((trainClass1stTrue/trainClass1stTotal)*100)/100
print probTrainClass1stTrue

trainClass1stFalse = float(len(trainClass1st.ix[trainClass1st['Survived'] == 'no']))
print trainClass1stFalse

probTrainClass1stFalse = round((trainClass1stFalse/trainClass1stTotal)*100)/100
print probTrainClass1stFalse

entropyClass1st = -probTrainClass1stTrue * math.log((probTrainClass1stTrue), 2) - probTrainClass1stFalse * math.log((probTrainClass1stFalse), 2)
entropyClass1st
```


    ['1st' '2nd' '3rd' 'crew']
    325.0
    203.0
    0.62
    122.0
    0.38




    0.9580420222262995



In[468]:

```
#clculate entropy for the Class - Type 2nd
#count numbers of rows that have yes/no for feature 'Class' where Class value is 2nd

trainClass2nd = train.ix[train['Class'] == '2nd']
#print trainClass

trainClass2ndTotal = float(len(trainClass2nd))
print trainClass2ndTotal

trainClass2ndTrue = float(len(trainClass2nd.ix[trainClass2nd['Survived'] == 'yes']))
print trainClass2ndTrue

probTrainClass2ndTrue = round((trainClass2ndTrue/trainClass2ndTotal)*100)/100
print probTrainClass2ndTrue

trainClass2ndFalse = float(len(trainClass2nd.ix[trainClass2nd['Survived'] == 'no']))
print trainClass2ndFalse

probTrainClass2ndFalse = round((trainClass2ndFalse/trainClass2ndTotal)*100)/100
print probTrainClass2ndFalse

entropyClass2nd = -probTrainClass2ndTrue * math.log((probTrainClass2ndTrue), 2) - probTrainClass2ndFalse * math.log((probTrainClass2ndFalse), 2)
entropyClass2nd
```


    285.0
    118.0
    0.41
    167.0
    0.59




    0.9765004687578241



In[475]:

```
#clculate entropy for the Class - Type '3rd'
#count numbers of rows that have yes/no for feature 'Class' where Class value is '3rd'

trainClass3rd = train.ix[train['Class'] == '3rd']
#print trainClass3rd

trainClass3rdTotal = float(len(trainClass3rd))
print trainClass3rdTotal

trainClass3rdTrue = float(len(trainClass3rd.ix[trainClass3rd['Survived'] == 'yes']))
print trainClass3rdTrue

probTrainClass3rdTrue = round((trainClass3rdTrue/trainClass3rdTotal)*100)/100
print probTrainClass3rdTrue

trainClass3rdFalse = float(len(trainClass3rd.ix[trainClass3rd['Survived'] == 'no']))
print trainClass3rdFalse

probTrainClass3rdFalse = round((trainClass3rdFalse/trainClass3rdTotal)*100)/100
print probTrainClass3rdFalse

entropyClass3rd = -probTrainClass3rdTrue * math.log((probTrainClass3rdTrue), 2) - probTrainClass3rdFalse * math.log((probTrainClass3rdFalse), 2)
entropyClass3rd
```


    676.0
    148.0
    0.22
    528.0
    0.78




    0.7601675029619657



In[486]:

```
#clculate entropy for the Class - Type 'crew'
#count numbers of rows that have yes/no for feature 'Class' where Class value is 'crew'

trainClassCrew = train.ix[train['Class'] == 'crew']
#print trainClassCrew

trainClassCrewTotal = float(len(trainClassCrew))
#print trainClassCrewTotal

#print trainClassCrew.ix[trainClassCrew['Survived'] == 'yes']

trainClassCrewTrue = float(len(trainClassCrew.ix[trainClassCrew['Survived'] == 'yes']))
#print trainClassCrewTrue

probTrainClassCrewTrue = round((trainClassCrewTrue/trainClassCrewTotal)*100)/100
#print probTrainClassCrewTrue

trainClassCrewFalse = float(len(trainClassCrew.ix[trainClassCrew['Survived'] == 'no']))
print trainClassCrewFalse

probTrainClassCrewFalse = round((trainClassCrewFalse/trainClassCrewTotal)*100)/100
print probTrainClassCrewFalse

###Fix log0 and log1 based math error by if/then loop
entropyClassCrew = -probTrainClassCrewTrue * math.log((probTrainClassCrewTrue), 2) - probTrainClassCrewFalse * math.log((probTrainClassCrewFalse), 2)
entropyClassCrew
```



    ---------------------------------------------------------------------------
    ValueError                                Traceback (most recent call last)

    <ipython-input-486-8a36aa0bd9de> in <module>()
         23 
         24 ###Fix log0 and log1 based math error by if/then loop
    ---> 25 entropyClassCrew = -probTrainClassCrewTrue * math.log((probTrainClassCrewTrue), 2) - probTrainClassCrewFalse * math.log((probTrainClassCrewFalse), 2)
         26 entropyClassCrew


    ValueError: math domain error


    166.0
    1.0


## Problem 2:  Titanic2 dataset

In[466]:

```

```
