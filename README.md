# pipleline
>> These notes serve to demonstrate my machine learning knowledge to potential employers while also helping beginners learn complex concepts in simple terms.

## In this repository we will learn about pipeline in machine learning . 
  Let's think about building a machine learning model. As you already know, machine learning isn't just about a model that automatically learns—you need to configure
  several steps to achieve good results:

- **Preprocess** the data 
- Select the most relevant **features**
- **Train the model**

These steps require precise management. If you perform them manually:
- you also need to maintain efficiency. 
- There are potential errors that must be controlled.
- Also potential data leakage during the process that must be controlled. 
- Also you have to make sure these steps apply the same in both training and test set.

  
This is where `Pipeline` becomes practical.


A **pipeline** is a tool that automates and organizes the steps into a single, streamlined workflow.
In other words, a pipeline transforms raw data into a trained model through a sequence of steps. This might sound abstract at first,
so let's explore how it works and what I mean by these 'steps'.

>> NB ⁉️
>> **Workflow:** The sequence of steps or processes that a program executes to complete a task.

## sklearn.pipeline.Pipeline
`sklearn.pipeline.Pipeline` is a `sklearn` class that sets up some transformation steps (any thing that changes the data) and a final estimator (model) . It puts all 
these steps into a single predictable object.
This allows all steps to be executed with one line of code,
and ensures the same workflow in both training and test sets while prevents data leakage and potential errors that occure
during manual deployment.

## Components of a pipeline
Pipeline components are the individual building blocks that make up a pipeline.

- Each step contains a single component (transformer OR estimator)
- Steps are typically tuples: `(name, component)`
- `transformer`: Transforms/preprocesses data (implements `fit()` and `transform()`)
- `estimator`: Learns patterns and makes predictions (implements `fit()` and `predict()`)
- 
>> In **machine learning pipelines**, the final step is usually an estimator.

This an image which you can see the pipeline , its components and how it works :

<p align="center"><img src="https://github.com/farzane-yoosefi/pipleline/blob/main/pipe.png" alt="Description" width="300" /></p>

## create pipeline
### step1 : importing the data and necessary libraries
```python
import pandas as pd
from sklearn  import datasets
from sklearn.model_selection import train_test_split
```
### step2 : Find out about the dataset
```python
# Converting the data into a pandas dataframe
data = datasets.load_iris()
data = pd.DataFrame(data.data,columns =data.feature_names)
```
Now you can find out about the data using pandas functions after converting it to a DataFrame.

Here you look at first 5 rows of the data: 
```python
# Look at last 5 row
print("=== 1. First Look (Head) ===")
print(df.head())
```
output :
```
=== 1. First Look (Head) ===
   sepal length (cm)  sepal width (cm)  petal length (cm)  petal width (cm)
0                5.1               3.5                1.4               0.2
1                4.9               3.0                1.4               0.2
2                4.7               3.2                1.3               0.2
3                4.6               3.1                1.5               0.2
4                5.0               3.6                1.4               0.2

```
See how many rows and columns there are.
```python
# (Numbers of rows , number of columns)
print("===  2. Shape of data(shape)  ===")
print(f"rows : {df.shape[0]} columns : {df.shape[1]}")
```
OUtput :
```
===  2. Shape of data(shape)  ===
rows : 150 columns : 4
```
Look at the information of data 
```python
# Column names , data types , and non-null counts
print("===  Information about data  ===")
print(df.info())
```
output :
```

===  Information about data  ===
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 150 entries, 0 to 149
Data columns (total 4 columns):
 #   Column             Non-Null Count  Dtype  
---  ------             --------------  -----  
 0   sepal length (cm)  150 non-null    float64
 1   sepal width (cm)   150 non-null    float64
 2   petal length (cm)  150 non-null    float64
 3   petal width (cm)   150 non-null    float64
dtypes: float64(4)
memory usage: 4.8 KB

```
### Step 3 : Define feature , target and split the data
```python
X = data.data
Y = data.target
X_train,X_test,Y_train,Y_test = train_test_split(X, Y , test_size = 0.2, random_state = 42)
```
### step 4 : Define the `pipeline` by specifying the steps
```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

#Set up the pipeline
Pipe = Pipeline([
    ('scalar' , StandardScaler()),
    ('PCA',PCA(n_components=2)),
    ('logistic',LogisticRegression())
])
```
### step 5 : Train the model and make prediction :

```python
Pipe.fit(X_train,Y_train)
pred = Pipe.predict(X_test)
```
output : 
```
array([0, 2, 2, 1, 2, 2, 1, 1, 2,
       1, 0, 0, 1, 2, 2, 2, 1, 0, 2, 0, 1, 2,
       2, 1, 1, 2, 1, 0, 0, 2])
```
### Step 6 :Evaluate the model :
```python
from sklearn.metrics import accuracy_score
score = accuracy_score(pred , Y_test)
score
```
output :
```
0.9333333333333333
```
## What did actually the model do ?
**iris** dataset is a type of simple and classic dataset used for learning machine learning and sata analyses.
It is a type of measurment of 150 flowers.
There are 3 species of the this flower : ****Setosa**** ,****versicolor****,****virginica**** 
which are the target values.
These types are predicted based on 4 features :
- Sepal length
- Sepal width
- Petal length
- Petal width





