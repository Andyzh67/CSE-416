# Homework 1 - House Prices

### The Coding portion is submitted on EdStem and the Concept portion is submitted on Gradescope.

In this assignment we'll practice working with `pandas` `DataFrames` and train a linear regression model to predict house prices.

Fill in the cells provided marked `TODO` with code to answer the questions. Answers should do the computation stated rather than writing in hard-coded values. So for example, if a problem asks you to compute the average age of people in a dataset, you should be writing Python code in this notebook to do the computation instead of plugging it into some calculator and saving the hard-coded answer in the variable. In other words, we should be able to run your code on a smaller/larger dataset and get correct answers for those datasets with your code.

It is generally a good idea to restart the kernel and run all cells (especially before turning it in) to make sure your code runs correctly from start to finish.

## Submitting

To submit this coding portion on EdStem, press the "Mark" button on the bottom right of the screen. You can submit as many times as you want, and we will take your last one submitted when grading.

Unlike HW0, we do not show you all the test cases we run on your code. The tests that you see each time you submit indicate whether or not the types of the values you computed match our expected types. We do not share whether or not your answer is correct before you submit.


```python
# Conventionally people rename the pandas import to pd for brevity
import pandas as pd
```


```python
# Load in the data and preview it
sales = pd.read_csv('home_data.csv') 
sales.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>date</th>
      <th>price</th>
      <th>bedrooms</th>
      <th>bathrooms</th>
      <th>sqft_living</th>
      <th>sqft_lot</th>
      <th>floors</th>
      <th>waterfront</th>
      <th>view</th>
      <th>...</th>
      <th>grade</th>
      <th>sqft_above</th>
      <th>sqft_basement</th>
      <th>yr_built</th>
      <th>yr_renovated</th>
      <th>zipcode</th>
      <th>lat</th>
      <th>long</th>
      <th>sqft_living15</th>
      <th>sqft_lot15</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>7129300520</td>
      <td>20141013T000000</td>
      <td>221900</td>
      <td>3</td>
      <td>1.00</td>
      <td>1180</td>
      <td>5650</td>
      <td>1.0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>7</td>
      <td>1180</td>
      <td>0</td>
      <td>1955</td>
      <td>0</td>
      <td>98178</td>
      <td>47.5112</td>
      <td>-122.257</td>
      <td>1340</td>
      <td>5650</td>
    </tr>
    <tr>
      <th>1</th>
      <td>6414100192</td>
      <td>20141209T000000</td>
      <td>538000</td>
      <td>3</td>
      <td>2.25</td>
      <td>2570</td>
      <td>7242</td>
      <td>2.0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>7</td>
      <td>2170</td>
      <td>400</td>
      <td>1951</td>
      <td>1991</td>
      <td>98125</td>
      <td>47.7210</td>
      <td>-122.319</td>
      <td>1690</td>
      <td>7639</td>
    </tr>
    <tr>
      <th>2</th>
      <td>5631500400</td>
      <td>20150225T000000</td>
      <td>180000</td>
      <td>2</td>
      <td>1.00</td>
      <td>770</td>
      <td>10000</td>
      <td>1.0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>6</td>
      <td>770</td>
      <td>0</td>
      <td>1933</td>
      <td>0</td>
      <td>98028</td>
      <td>47.7379</td>
      <td>-122.233</td>
      <td>2720</td>
      <td>8062</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2487200875</td>
      <td>20141209T000000</td>
      <td>604000</td>
      <td>4</td>
      <td>3.00</td>
      <td>1960</td>
      <td>5000</td>
      <td>1.0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>7</td>
      <td>1050</td>
      <td>910</td>
      <td>1965</td>
      <td>0</td>
      <td>98136</td>
      <td>47.5208</td>
      <td>-122.393</td>
      <td>1360</td>
      <td>5000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1954400510</td>
      <td>20150218T000000</td>
      <td>510000</td>
      <td>3</td>
      <td>2.00</td>
      <td>1680</td>
      <td>8080</td>
      <td>1.0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>8</td>
      <td>1680</td>
      <td>0</td>
      <td>1987</td>
      <td>0</td>
      <td>98074</td>
      <td>47.6168</td>
      <td>-122.045</td>
      <td>1800</td>
      <td>7503</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 21 columns</p>
</div>



---
# Exploring the data.
This question asks you to explore the dataset we will be using. Answer the following three sentences by writing code to compute the dsecribed values in the given variables names. Note that Q1 has 3 sub-questions.

## Q1) Exploring the data

1. How many houses do we have in our data set? This should correspond to the number of rows in the dataset. Save the number of rows in a variable called `num_rows`.
2. Which column are we trying to predict given the other features (i.e. what's our output `y`)? Save the column values in a variable called `y`.
3. How many inputs do we have in total in the data set (i.e. what is the size of our input `x`)? Save the number of inputs in a variable called `num_inputs`.

Each one of these questions should be answered in the next cells respectively.

*Hint:* You can use `len()` to get the length of anything in Python. Note that when you use it on a DataFrame, it will give you the number of rows, not columns.



```python
### edTest(test_num_rows) ###
num_rows = sales.shape[0]
num_cols = sales.shape[1]

# TODO
```


```python
### edTest(test_get_labels) ###
y = sales['price']
# TODO 
```


```python
### edTest(test_num_inputs) ###
x = sales.iloc[0:num_rows, 3:num_cols]
n1 = x.shape[0]
num_inputs = x.shape[1]

# TODO
```

---
## Q2) What is the average price of houses with 3 bedrooms?

Compute the average price of houses in the dataset with 3 bedrooms. Save the result in `avg_price_3_bed`.


```python
### edTest(test_avg_price_3_bed) ###
bed = sales[sales['bedrooms'] == 3]
avg_price_3_bed = bed['price'].mean()
avg_price_3_bed
# TODO 
```




    466232.07949918567



---
## Q3) What fraction of the properties are have `sqft_living` between 2000-4000?

Compute the fraction of properties with `sqft_living` between 2000 (inclusive) and 4000 (exclusive). Your answer should be stored in `percent_q3` and it should be a number between 0 and 1.
 


```python
### edTest(test_percent_q3) ###
sample = sales[(sales['sqft_living'] >= 2000) & (sales['sqft_living'] <= 4000)]
percent_q3 = len(sample) / num_rows
percent_q3
# TODO 
```




    0.4266413732475825



---
# Training Linear Regression Models

## Q4) Training a Linear Regression Model.

We will now train a linear regression model to make useful predictions. Work through the steps below and then answer the following questions. Even though a lot of the code is pre-written, you should understand what it is doing! You may be asked to write some of this code on future assignments.

First we split the data into a training set and a test set.

**You should not modify the next two cells. Even though there is an `edTest` comment. The `edTest` comment is there to let us set up some state, and does no test any functionality. These cells need to be left as-is, otherwise it will potentially mess up future tests.**


```python
### edTest(test_setup_train_test_split) ###
```


```python
from sklearn.model_selection import train_test_split

# Split data into 80% train and 20% test
train_data, test_data = train_test_split(sales, test_size=0.2)
```

Lets plot some of the data to get a sense of what we are dealing with. You do not need to understand every part of the plotting code here, but plotting is a good skill in Python so it will help to read over this.


```python
import matplotlib.pyplot as plt
%matplotlib inline

# Plot sqft_living vs housing price for the train and test da
plt.scatter(train_data['sqft_living'], train_data['price'], marker='+', label='Train')
plt.scatter(test_data['sqft_living'], test_data['price'], marker='.', label='Test')

# Code to customize the axis labels
plt.legend()
plt.xlabel('Sqft Living')
plt.ylabel('Price')
```




    Text(0, 0.5, 'Price')




    
![png](output_15_1.png)
    


For this problem, we will look at using two sets of features derived from the data inputs. The basic set of features only contains a few data inputs while the advanced features contain them and more.


```python
basic_features = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'zipcode']
```


```python
advanced_features = basic_features + [
    'condition',      # condition of the house
    'grade',          # measure of qality of construction
    'waterfront',     # waterfront property 
    'view',           # type of view
    'sqft_above',     # square feet above ground
    'sqft_basement',  # square feet in basementab
    'yr_built',       # the year built
    'yr_renovated',   # the year renovated
    'lat',            # the longitude of the parcel
    'long',           # the latitide of the parcel
    'sqft_living15',  # average sq.ft. of 15 nearest neighbors 
    'sqft_lot15',     # average lot size of 15 nearest neighbors 
]
```

---
In the following cell, you should train two linear regression models
* The first should be saved in a variable called `basic_model` that only uses the basic features
* The seconod should be saved in a variable called `advanced_model` that uses the advanced features

You'll need to look through the [LinearRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html) class from scikit-learn to look into how to train a regression model for this task. In particular, make sure you check out the `fit` function.

Notice that our goal is to eventually make a prediction of how the model will do in the future. You should keep this in mind when deciding which datasets to use where.


```python
### edTest(test_train_models) ###
from sklearn.linear_model import LinearRegression
basic_model = LinearRegression()
basic_model.fit(train_data[basic_features], train_data['price'])

advanced_model = LinearRegression()
advanced_model.fit(train_data[advanced_features], train_data['price'])
# TODO
```




    LinearRegression()



Now, we will evaluate the models' predictions to see how they perform.

---
# Root Mean Suare Error (RMSE) of trained predictors

## Q5) What are your Root Mean Squared Errors (RMSE) on your training data using the basic model and the advanced model?


Use the models you trained in last section to predict what it thinks the values for the data points should be. You can look at the documentation from the `LinearRegression` model to see how to make predictions. 

The RMSE is another commonly reported metric used for regression models. The RMSE is similar to RSS but is modified slightly to scale the number down. The RMSE is defined as $$RMSE = \sqrt{\frac{1}{n}RSS}$$

where the thing inside the square root is refered to as the Mean Square Error (MSE). You will also need to use the `mean_squared_error` function from sklearn (documentation [here](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html#sklearn.metrics.mean_squared_error)) which you'll have to import as well. 

**Save your result in variables named `train_rmse_basic` and `train_rmse_advanced` respectively.**

**Remember, we want you to report the square root of the MSE numbers**.


```python
### edTest(test_train_rmse) ###
from sklearn.metrics import mean_squared_error
import math

true_price = train_data['price']
p1 = basic_model.predict(train_data[basic_features])
p2 = advanced_model.predict(train_data[advanced_features])

train_rmse_basic = math.sqrt(mean_squared_error(true_price, p1))
train_rmse_advanced = math.sqrt(mean_squared_error(true_price, p2))
# TODO
```

---
## Q6) What are your RMSE errors on your test data using the basic model and then the advanced model?

Similar to the last problem, but compute the test RMSE. Store your results in `test_rmse_basic` and `test_rmse_advanced`.


```python
### edTest(test_test_rmse) ###
true_price = test_data['price']
p1 = basic_model.predict(test_data[basic_features])
p2 = advanced_model.predict(test_data[advanced_features])

test_rmse_basic = math.sqrt(mean_squared_error(true_price, p1))
test_rmse_advanced = math.sqrt(mean_squared_error(true_price, p2))

# TODO
```

---
## Q7) Which model would you choose and why?
These questions do not need any code to answer them. Instead, save a variable with the specified name with a string of the specified option.


### Q7.1) Which model would you choose?

Ignore the fact that we do not have a validation set for this assignment (we will get to that in the next assignment). Which model do you think would perform better in the future?

* a) Model with the basic features
* b) Model with the advanced features

Save your result in a variable named `q7_1`. For example, if your anwer is option a, write 

```
q7_1 = 'a'
```



```python
### edTest(test_q7_1) ###
q7_1 = 'b'
# TODO
```

### Q7.2) Why?

Same as before, select on one option to describe why you would select the model in the previous question. Save your result as a string in a variable named `q7_2`.

* a) It has higher training error
* b) It uses more features
* c) It has lower test error
* d) It has lower training error
* e) It has higher test error


```python
### edTest(test_q7_2) ###
q7_2 = 'c'
# TODO
```

# Concept Portion
Make sure you also complete the concept portion of this assignment before the due date.

> Copyright ©2020 Emily Fox and Hunter Schafer.  All rights reserved.  Permission is hereby granted to students registered for University of Washington CSE/STAT 416 for use solely during Autumn Quarter 2021 for purposes of the course.  No other use, copying, distribution, or modification is permitted without prior written consent. Copyrights for third-party components of this work must be honored.  Instructors interested in reusing these course materials should contact the author.


```python

```


```python

```
