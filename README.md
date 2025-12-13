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

<p align="center"><img src="https://github.com/farzane-yoosefi/pipleline/blob/main/pipe.png" alt="Description" width="300" /></p>

## Components of a pipeline


