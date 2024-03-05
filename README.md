<H3>NAME : Kamesh D S</H3>
<H3>REGISTER NO : 212222240043</H3>
<H3>EX. NO.1</H3>
<H3>DATE</H3>
<H1 ALIGN =CENTER> Introduction to Kaggle and Data preprocessing</H1>

## AIM:

To perform Data preprocessing in a data set downloaded from Kaggle

## EQUIPMENTS REQUIRED:
Hardware – PCs
Anaconda – Python 3.7 Installation / Google Colab /Jupiter Notebook

## RELATED THEORETICAL CONCEPT:

**Kaggle :**
Kaggle, a subsidiary of Google LLC, is an online community of data scientists and machine learning practitioners. Kaggle allows users to find and publish data sets, explore and build models in a web-based data-science environment, work with other data scientists and machine learning engineers, and enter competitions to solve data science challenges.

**Data Preprocessing:**

Pre-processing refers to the transformations applied to our data before feeding it to the algorithm. Data Preprocessing is a technique that is used to convert the raw data into a clean data set. In other words, whenever the data is gathered from different sources it is collected in raw format which is not feasible for the analysis.
Data Preprocessing is the process of making data suitable for use while training a machine learning model. The dataset initially provided for training might not be in a ready-to-use state, for e.g. it might not be formatted properly, or may contain missing or null values.Solving all these problems using various methods is called Data Preprocessing, using a properly processed dataset while training will not only make life easier for you but also increase the efficiency and accuracy of your model.

**Need of Data Preprocessing :**

For achieving better results from the applied model in Machine Learning projects the format of the data has to be in a proper manner. Some specified Machine Learning model needs information in a specified format, for example, Random Forest algorithm does not support null values, therefore to execute random forest algorithm null values have to be managed from the original raw data set.
Another aspect is that the data set should be formatted in such a way that more than one Machine Learning and Deep Learning algorithm are executed in one data set, and best out of them is chosen.


## ALGORITHM:
STEP 1:Importing the libraries<BR>
STEP 2:Importing the dataset<BR>
STEP 3:Taking care of missing data<BR>
STEP 4:Encoding categorical data<BR>
STEP 5:Normalizing the data<BR>
STEP 6:Splitting the data into test and train<BR>

##  PROGRAM:

### IMPORT LIBRARIES:
```py
import pandas as pd
import io
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from scipy import stats
import numpy as np
```

### READ THE DATA:
```py
df=pd.read_csv("Churn_Modelling.csv")
```

### CHECK DATA:
```py
df.head()
df.tail()
df.columns
```

### CHECK THE MISSING DATA:
```py
df.isnull().sum()
```

### CHECK FOR DUPLICATES:
```py
df.isnull().sum()
```

### ASSIGNING X:
```py
X = df.iloc[:,:-1].values
X
```

### ASSIGNING Y:
```py
Y = df.iloc[:,-1].values
Y
```

### CHECK FOR OUTLIERS:
```py
df.describe()
```

### DROPPING STRING VALUES DATA FROM DATASET:
```py
data = df.drop(['Surname', 'Geography','Gender'], axis=1)
```

### CHECKING DATASETS AFTER DROPPING STRING VALUES DATA FROM DATASET:
```py
data.head()
```

### NORMALIE THE DATASET USING (MinMax Scaler):
```py
scaler=MinMaxScaler()
df1=pd.DataFrame(scaler.fit_transform(data))
print(df1)
```

### SPLIT THE DATASET:
```py
X=df.iloc[:,:-1].values
Y=df.iloc[:,-1].values
print(X)
print(Y)
```

### TRAINING AND TESTING MODEL:
```py
X_train ,X_test ,Y_train,Y_test=train_test_split(X,Y,test_size=0.2)
print("X_train\n")
print(X_train)
print("\nLenght of X_train ",len(X_train))
print("\nX_test\n")
print(X_test)
print("\nLenght of X_test ",len(X_test))
```

## OUTPUT:

### DATA CHECKING:
![image](https://github.com/KameshLeVI/Ex-1-NN/assets/120780633/04f46d0a-4bf3-47b4-9a04-ced35d174c13)


### MISSING DATA:
![image](https://github.com/KameshLeVI/Ex-1-NN/assets/120780633/7f19d636-0ea0-475b-85ce-65952ced27fb)

### DUPLICATES IDENTIFICATION:
![image](https://github.com/KameshLeVI/Ex-1-NN/assets/120780633/3e0519a3-ef73-41c7-9bf0-9afa6735962e)

### VALUE OF Y:
![image](https://github.com/KameshLeVI/Ex-1-NN/assets/120780633/8c2f3a95-1cc5-4807-8c67-7aa73d00fdec)

### OUTLIERS:
![image](https://github.com/KameshLeVI/Ex-1-NN/assets/120780633/08550b8f-21de-4261-8d57-630d2e873edf)


### CHECKING DATASET AFTER DROPPING STRING VALUES DATA FROM DATASET:
![307384149-6978a892-4d84-45f7-a61f-d4da24af6537](https://github.com/sivabalan28/Ex-1-NN/assets/113497347/4df7474f-e636-4579-acc5-70cca521fb7e)

### NORMALIZE THE DATASET:
![image](https://github.com/KameshLeVI/Ex-1-NN/assets/120780633/83a0daf2-c07a-434e-b527-e6a076229fa4)


### SPLIT THE DATASET:
![image](https://github.com/KameshLeVI/Ex-1-NN/assets/120780633/98b55479-90c8-47e4-b2c0-fc46f71d00e8)

### TRAINING AND TESTING MODEL:
![image](https://github.com/KameshLeVI/Ex-1-NN/assets/120780633/d20c7c28-c0b0-49d5-8afd-c5487603c9d6)

## RESULT:
Thus, Implementation of Data Preprocessing is done in python  using a data set downloaded from Kaggle.


