<H3>ENTER YOUR NAME:  Gokul J <H3>
<H3>ENTER YOUR REGISTER NO: 212222230038</H3>
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
```python
import pandas as pd
import io
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

df=pd.read_csv("/content/Churn_Modelling.csv", index_col="RowNumber")
df

df.drop(['CustomerId'],axis=1,inplace=True)
df.drop(['Surname'],axis=1,inplace=True)
df.drop('Age',axis=1,inplace=True)
df.drop('Geography',axis=1,inplace=True)
df.drop('Gender',axis=1,inplace=True)
df

df.isnull().sum()
df.duplicated()
df.describe()

scaler=StandardScaler()
df1=pd.DataFrame(scaler.fit_transform(df))
df1

x=df1.iloc[:,:-1].values
x
y=df1.iloc[:,-1].values
y

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
print(x_train)
print(len(x_train))
print(x_test)
print(len(x_test))
```


## OUTPUT:

## DATASET:

![image](https://github.com/user-attachments/assets/fc7aa9d9-e9f7-4c0e-98ed-3cc03f096f04)


## DROPPING THE UNWANTED DATASET:

![image](https://github.com/user-attachments/assets/c9b3916f-3efc-4ece-bae1-57f9b9052b38)


## CHECKING NULL VALUES:

![image](https://github.com/user-attachments/assets/e575e37b-3c74-415b-84ac-ab9d3d7359f9)

## CHECKING FOR DUPLICATION:

![image](https://github.com/user-attachments/assets/b8bc451f-be7f-4e20-9c13-f4819e86ab1c)


## DESCRIBING THE DATASET:

![image](https://github.com/user-attachments/assets/4de3ad01-19e9-4857-817c-564c33782acb)

## SCALING THE DATASET:

![image](https://github.com/user-attachments/assets/de14f593-bf22-4995-8687-b678395af658)


## X and Y FEATURES:

![image](https://github.com/user-attachments/assets/eb3b7ef9-a06e-4221-b50d-4018dfd46536)


## SPLITTING THE TRAINING AND TESTING DATASET:

![image](https://github.com/user-attachments/assets/d286151e-c239-4522-8819-7daa6932c037)




## RESULT:
Thus, Implementation of Data Preprocessing is done in python  using a data set downloaded from Kaggle.


