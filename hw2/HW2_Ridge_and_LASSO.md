# Homework 2 - Ridge and LASSO Regression

### The Coding portion is submitted on EdStem and the Concept portion is submitted on Gradescope.

In this assignment we'll look at the affect of using regularization on linear regression models that we train. You will write code to train models that use different regularizers and different penalties and to analyze how this affects the model.


Fill in the cells provided marked `TODO` with code to answer the questions. Answers should do the computation stated rather than writing in hard-coded values. So for example, if a problem asks you to compute the average age of people in a dataset, you should be writing Python code in this notebook to do the computation instead of plugging it into some calculator and saving the hard-coded answer in the variable. In other words, we should be able to run your code on a smaller/larger dataset and get correct answers for those datasets with your code.

It is generally a good idea to restart the kernel and run all cells (especially before turning it in) to make sure your code runs correctly. Answer the questions on Gradescope and make sure to download this file once you've finished the assignment and upload it to Canvas as well.

Note, you are not allowed to share any portions of this notebook outside of this class.

> Copyright ©2021 Emily Fox and Hunter Schafer.  All rights reserved.  Permission is hereby granted to students registered for University of Washington CSE/STAT 416 for use solely during Spring Quarter 2021 for purposes of the course.  No other use, copying, distribution, or modification is permitted without prior written consent. Copyrights for third-party components of this work must be honored.  Instructors interested in reusing these course materials should contact the author.

---


```python
# Conventionally people rename these common imports for brevity
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Magic command to make the plots appear in-line (it's actually called a "magic command")
%matplotlib inline
```

For this assignment, we will only be using a very small subset of the data to do our analysis. This is not something you would usually do in practice, but is something we do for this assignment to simplify this complexity of this dataset. The data is pretty noisy and to get meaningful results to demonstrate the theoretical behavior, you would need to use a much more complicated set of features that would be a bit more tedious to work with.

We use a parameter called `random_state` to control the randomness in loading the data so you don't get wildly different results than us just to how the data is sampled.


```python
sales = pd.read_csv('home_data.csv') 

# Selects 1% of the data
sales = sales.sample(frac=0.01, random_state=0) 

print(f'Number of points: {len(sales)}')
sales.head()
```

    Number of points: 216





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
      <th>17384</th>
      <td>1453602313</td>
      <td>20141029T000000</td>
      <td>297000</td>
      <td>2</td>
      <td>1.50</td>
      <td>1430</td>
      <td>1650</td>
      <td>3.0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>7</td>
      <td>1430</td>
      <td>0</td>
      <td>1999</td>
      <td>0</td>
      <td>98125</td>
      <td>47.7222</td>
      <td>-122.290</td>
      <td>1430</td>
      <td>1650</td>
    </tr>
    <tr>
      <th>722</th>
      <td>2225059214</td>
      <td>20140808T000000</td>
      <td>1578000</td>
      <td>4</td>
      <td>3.25</td>
      <td>4670</td>
      <td>51836</td>
      <td>2.0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>12</td>
      <td>4670</td>
      <td>0</td>
      <td>1988</td>
      <td>0</td>
      <td>98005</td>
      <td>47.6350</td>
      <td>-122.164</td>
      <td>4230</td>
      <td>41075</td>
    </tr>
    <tr>
      <th>2680</th>
      <td>2768000270</td>
      <td>20140625T000000</td>
      <td>562100</td>
      <td>2</td>
      <td>0.75</td>
      <td>1440</td>
      <td>3700</td>
      <td>1.0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>7</td>
      <td>1200</td>
      <td>240</td>
      <td>1914</td>
      <td>0</td>
      <td>98107</td>
      <td>47.6707</td>
      <td>-122.364</td>
      <td>1440</td>
      <td>4300</td>
    </tr>
    <tr>
      <th>18754</th>
      <td>6819100040</td>
      <td>20140624T000000</td>
      <td>631500</td>
      <td>2</td>
      <td>1.00</td>
      <td>1130</td>
      <td>2640</td>
      <td>1.0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>8</td>
      <td>1130</td>
      <td>0</td>
      <td>1927</td>
      <td>0</td>
      <td>98109</td>
      <td>47.6438</td>
      <td>-122.357</td>
      <td>1680</td>
      <td>3200</td>
    </tr>
    <tr>
      <th>14554</th>
      <td>4027700666</td>
      <td>20150426T000000</td>
      <td>780000</td>
      <td>4</td>
      <td>2.50</td>
      <td>3180</td>
      <td>9603</td>
      <td>2.0</td>
      <td>0</td>
      <td>2</td>
      <td>...</td>
      <td>9</td>
      <td>3180</td>
      <td>0</td>
      <td>2002</td>
      <td>0</td>
      <td>98155</td>
      <td>47.7717</td>
      <td>-122.277</td>
      <td>2440</td>
      <td>15261</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 21 columns</p>
</div>



## Q1 - Feature Engineering
First, we do a bit of feature engineering by creating features that represent the squares of each feature and the square root of each feature. One benefit of using regularization is you can include more features than necessary and you don't have to be as worried about overfitting since the model is regularized.

In the following cell, complete the code inside the loop to compute the square of each feature the the square root of each feature.

*Note: For this problem, we will make the correctness autograder test public to help avoid downstream issues.*


```python
### edTest(test_feature_extraction) ###

from math import sqrt

# All of the features of interest
selected_inputs = [
    'bedrooms', 
    'bathrooms',
    'sqft_living', 
    'sqft_lot', 
    'floors', 
    'waterfront', 
    'view', 
    'condition', 
    'grade',
    'sqft_above',
    'sqft_basement',
    'yr_built', 
    'yr_renovated'
]

# Compute the square and sqrt of each feature
all_features = []
for data_input in selected_inputs:
    square_feat = data_input + '_square'
    sqrt_feat = data_input + '_sqrt'
    
    # TODO compute the square and square root as two new features
    sales[square_feat] = sales[data_input] ** 2
    sales[sqrt_feat] = sales[data_input] ** 0.5

    all_features.extend([data_input, square_feat, sqrt_feat])


# Split the data into features and price
price = sales['price']
sales = sales[all_features]

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
      <th>bedrooms</th>
      <th>bedrooms_square</th>
      <th>bedrooms_sqrt</th>
      <th>bathrooms</th>
      <th>bathrooms_square</th>
      <th>bathrooms_sqrt</th>
      <th>sqft_living</th>
      <th>sqft_living_square</th>
      <th>sqft_living_sqrt</th>
      <th>sqft_lot</th>
      <th>...</th>
      <th>sqft_above_sqrt</th>
      <th>sqft_basement</th>
      <th>sqft_basement_square</th>
      <th>sqft_basement_sqrt</th>
      <th>yr_built</th>
      <th>yr_built_square</th>
      <th>yr_built_sqrt</th>
      <th>yr_renovated</th>
      <th>yr_renovated_square</th>
      <th>yr_renovated_sqrt</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>17384</th>
      <td>2</td>
      <td>4</td>
      <td>1.414214</td>
      <td>1.50</td>
      <td>2.2500</td>
      <td>1.224745</td>
      <td>1430</td>
      <td>2044900</td>
      <td>37.815341</td>
      <td>1650</td>
      <td>...</td>
      <td>37.815341</td>
      <td>0</td>
      <td>0</td>
      <td>0.000000</td>
      <td>1999</td>
      <td>3996001</td>
      <td>44.710178</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>722</th>
      <td>4</td>
      <td>16</td>
      <td>2.000000</td>
      <td>3.25</td>
      <td>10.5625</td>
      <td>1.802776</td>
      <td>4670</td>
      <td>21808900</td>
      <td>68.337398</td>
      <td>51836</td>
      <td>...</td>
      <td>68.337398</td>
      <td>0</td>
      <td>0</td>
      <td>0.000000</td>
      <td>1988</td>
      <td>3952144</td>
      <td>44.586994</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2680</th>
      <td>2</td>
      <td>4</td>
      <td>1.414214</td>
      <td>0.75</td>
      <td>0.5625</td>
      <td>0.866025</td>
      <td>1440</td>
      <td>2073600</td>
      <td>37.947332</td>
      <td>3700</td>
      <td>...</td>
      <td>34.641016</td>
      <td>240</td>
      <td>57600</td>
      <td>15.491933</td>
      <td>1914</td>
      <td>3663396</td>
      <td>43.749286</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>18754</th>
      <td>2</td>
      <td>4</td>
      <td>1.414214</td>
      <td>1.00</td>
      <td>1.0000</td>
      <td>1.000000</td>
      <td>1130</td>
      <td>1276900</td>
      <td>33.615473</td>
      <td>2640</td>
      <td>...</td>
      <td>33.615473</td>
      <td>0</td>
      <td>0</td>
      <td>0.000000</td>
      <td>1927</td>
      <td>3713329</td>
      <td>43.897608</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>14554</th>
      <td>4</td>
      <td>16</td>
      <td>2.000000</td>
      <td>2.50</td>
      <td>6.2500</td>
      <td>1.581139</td>
      <td>3180</td>
      <td>10112400</td>
      <td>56.391489</td>
      <td>9603</td>
      <td>...</td>
      <td>56.391489</td>
      <td>0</td>
      <td>0</td>
      <td>0.000000</td>
      <td>2002</td>
      <td>4008004</td>
      <td>44.743715</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 39 columns</p>
</div>



## Q2 - Split Data
Next, we need to split our data into our train, validation, and test data. To do this, we will use [train_test_split](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html) function to split up the dataset. For this assignment we will use 70% of the data to train, 10% for validation, and 20% to test. 

We have written most of the splitting for you, but we need you to figure out what the sizes should be in this case based off the numbers above. 

Note that we use `random_state=6` to make sure the results are deterministic for our assignment. Don't modify any code in this section besides changing the `<NUM>`s to the correct values.

*Hint: You should print out the length of the datasets to make sure you got it right!*

*Note: For this problem, we will make the correctness autograder test public to help avoid downstream issues.*


```python
### edTest(test_data_splitting) ###

# TODO Fill in the numbers to make datasets of the right size.
from sklearn.model_selection import train_test_split

train_and_validation_sales, test_sales, train_and_validation_price, test_price = \
    train_test_split(sales, price, test_size=0.2, random_state=6)

train_sales, validation_sales, train_price, validation_price = \
    train_test_split(train_and_validation_sales, train_and_validation_price, test_size=0.125, random_state=6)
```

## Q3 - Standardization

We first need to do a little bit more pre-processing to prepare the data for model training. Models like Ridge and LASSO assume the input features are standardized (mean 0, std. dev. 1) and the target values are centered (mean 0). If we do not do this, we might get some unpredictable results since we violate the assumption of the models!

So in the next cell, you should standardize the data in train, validation, and test using the following instructions:
* Use the [StandardScaler](https://scikit-learn.org/stable/modules/preprocessing.html#standardization-or-mean-removal-and-variance-scaling) preprocessor provided by scikit-learn to do the standardization for you. Note that you first `fit` it to the data so it can compute the mean/standard deviation and then `transform` to actually change the data. You'll find the examples on this documentation very helpful.
* You should only do this transformation on the features we are using of the data (not any of the other data inputs or the target values). 
* This next note will sound weird, but it's an important step. **You should only do the standardization calculation (e.g., the mean and the standard deviation) on the *training* set and use those statistics to scale the validation and test set**. In other words, the validation and test set should be standardized using the statistics from the training data so that you are using a consistent transformation throughout. This is important to do since you need to apply the same transformation process to every step of the data and you shouldn't use statistics from data outside of your training set in your transformations.


**Store the transformed results in `train_sales`, `validation_sales`, `test_sales` respectively (thus overwriting your old, unscaled feature).**

*Note: For this problem, we will make the correctness autograder test public to help avoid downstream issues.*


```python
### edTest(test_standardization) ###
from sklearn.preprocessing import StandardScaler

# TODO preprocess the training, validation, and test data
scaler = StandardScaler().fit(train_sales, train_price)

train_sales = scaler.transform(train_sales)
validation_sales = scaler.transform(validation_sales)
test_sales = scaler.transform(test_sales)
```

# Linear Regression 
## Q4) Linear Regression Baseline

As a baseline, we will first, train a regular `LinearRegression` model on the data using the features in `all_features` and report its **test RMSE**. Write the code in the cell below to calculate the answer. Save your result in a variable named `test_rmse`.



```python
### edTest(test_train_linear_regression) ###

from sklearn.metrics import mean_squared_error

def rmse(model, X, y):
    y_pred = model.predict(X)
    return mean_squared_error(y, y_pred, squared = False)


from sklearn.linear_model import LinearRegression

# TODO Train a linear regression model (you'll likely need to import some things)
basic_model = LinearRegression().fit(train_sales, train_price)

test_rmse = rmse(basic_model, test_sales, test_price)

test_rmse
```




    384955.79936441785



--- 
# Ridge Regression
At this point, you might be looking forward at the homework and seeing how long it is! We want to provide a lot of instruction so you aren't left completely in the dark on what to do, but we are also trying to avoid just giving you a bunch of starter code and just having you fill in the blanks. This section is very long because it tries to give really detailed instructions on what to compute. The next section on LASSO has almost exactly the same steps so it will be a lot easier doing that part of the assignment!

In this section, we will do some **hyper-parameter tuning** to find the optimal setting of the regularization constant $\lambda$ for Ridge Regression. Remember that $\lambda$ is the coefficient that controls how much the model is penalized for having large weights in the optimization function.

$$\hat{w}_{ridge} = \min_w RSS(w) + \lambda \left\lVert w \right\rVert_2^2$$

where $\left\lVert w \right\rVert_2^2 = \sum_{j=0}^D w_j^2$ is the $l_2$-norm of the parameters. By default, `sklearn`'s `Ridge` class does not regularize the intercept.

## Q5) Train Ridge Models
For this part of the assignment, you will be writing code to find the optimal setting of the penalty $\lambda$. Below, we describe what steps you will want to have in your code to compute these values:

*Implementation Details*
* Use the following choices of $l_2$ penalty: $[10^{-5}, 10^{-4}, ..., 10^4, 10^5]$. In Python, you can create a list of these numbers using `np.logspace(-5, 5, 11, base=10)`. 
* Use the [Ridge](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html#sklearn.linear_model.Ridge) class from sklearn to train a Ridge Regression model on the **training** data. The **only** parameters you need to pass when constructing the Ridge model are `alpha`, which lets you specify what you want the $l_2$ penalty to be, and `random_state=0` to avoid randomness.
* Evaluate both the training error and the validation error for the model by reporting the RMSE of each dataset.
* **Put all of your results in a pandas `DataFrame` named `ridge_data`** so you can analyze them later. The `ridge_data` should have a row for each $l_2$ penalty you tried and should have the following columns:
  * `l2_penalty`: The $l_2$ penalty for that row
  * `model`: The actual `Ridge` model object that was trained with that $l_2$ penalty
  * `train_rmse`: The training RMSE for that model
  * `validation_rmse`: The validation RMSE for that model
* To build up this `DataFrame`, we recommend first building up a list of dictionary objects and then converting that to a `DataFrame`. For example, the following code would produce the following `pandas.DataFrame`.
```python
data = []
for i in range(3):
    data.append({
        'col_a': i,
        'col_b': 2 * i
    }
data_frame = pd.DataFrame(data)
```

| col_a | col_b | 
|-------|-------|
|   0   |   0   | 
|   1   |   2   | 
|   2   |   4   |

*Hints: Here is a development strategy that you might find helpful*
* You will need a loop to loop over the possible $l_2$ penalties. Try writing the code without a loop first with just one setting of $\lambda$. Try writing a lot of the code without a loop first if you're stuck to help you figure out how the pieces go together. You can safely ignore building up the result `DataFrame` at first, just print all the information out to start! 
* If you are running into troubles writing your loop, try to print values out to investigate what's going wrong.
* Remember to use RMSE for calculating the error!



```python
### edTest(test_ridge) ###
from sklearn.linear_model import Ridge

# TODO Implement code to evaluate Ridge Regression with various l2 Penalties
reg_coefs = np.logspace(-5, 5, 11, base=10)
data = []

for reg_coef in reg_coefs:
    ridge_model = Ridge(alpha = reg_coef).fit(train_sales, train_price)
    train_rmse = rmse(ridge_model, train_sales, train_price)
    validation_rmse = rmse(ridge_model, validation_sales, validation_price)
    data.append({
        'l2_penalty': reg_coef,
        'model': ridge_model,
        'train_rmse': train_rmse,
        'validation_rmse' : validation_rmse,
    }
    )
    
ridge_data = pd.DataFrame(data)
ridge_data
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
      <th>l2_penalty</th>
      <th>model</th>
      <th>train_rmse</th>
      <th>validation_rmse</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.00001</td>
      <td>Ridge(alpha=1e-05)</td>
      <td>146188.566942</td>
      <td>392112.522241</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.00010</td>
      <td>Ridge(alpha=0.0001)</td>
      <td>146210.884175</td>
      <td>392721.061566</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.00100</td>
      <td>Ridge(alpha=0.001)</td>
      <td>146610.292053</td>
      <td>393099.892724</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.01000</td>
      <td>Ridge(alpha=0.01)</td>
      <td>147967.692703</td>
      <td>369263.393012</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.10000</td>
      <td>Ridge(alpha=0.1)</td>
      <td>151619.819399</td>
      <td>330722.661247</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1.00000</td>
      <td>Ridge()</td>
      <td>154932.690092</td>
      <td>302623.478585</td>
    </tr>
    <tr>
      <th>6</th>
      <td>10.00000</td>
      <td>Ridge(alpha=10.0)</td>
      <td>161876.430362</td>
      <td>282876.469131</td>
    </tr>
    <tr>
      <th>7</th>
      <td>100.00000</td>
      <td>Ridge(alpha=100.0)</td>
      <td>181371.431142</td>
      <td>283001.128683</td>
    </tr>
    <tr>
      <th>8</th>
      <td>1000.00000</td>
      <td>Ridge(alpha=1000.0)</td>
      <td>244296.061524</td>
      <td>341022.423065</td>
    </tr>
    <tr>
      <th>9</th>
      <td>10000.00000</td>
      <td>Ridge(alpha=10000.0)</td>
      <td>328840.881425</td>
      <td>486101.010743</td>
    </tr>
    <tr>
      <th>10</th>
      <td>100000.00000</td>
      <td>Ridge(alpha=100000.0)</td>
      <td>353757.594802</td>
      <td>528264.491801</td>
    </tr>
  </tbody>
</table>
</div>



Next, let's investigate how the penalty affected the train and validation error by running the following plotting code. 


```python
# Plot the validation RMSE as a blue line with dots
plt.plot(ridge_data['l2_penalty'], ridge_data['validation_rmse'], 
         'b-^', label='Validation')
# Plot the train RMSE as a red line dots
plt.plot(ridge_data['l2_penalty'], ridge_data['train_rmse'], 
         'r-o', label='Train')

# Make the x-axis log scale for readability
plt.xscale('log')

# Label the axes and make a legend
plt.xlabel('l2_penalty')
plt.ylabel('RMSE')
plt.legend()
```




    <matplotlib.legend.Legend at 0x7fcacfe75e80>




    
![png](output_16_1.png)
    


Next, we want to actually look at which model we think will perform best. First we define a helper function that will be used to inspect the model parameters.


```python
def print_coefficients(model, features):
    """
    This function takes in a model column and a features column. 
    And prints the coefficient along with its feature name.
    """
    feats = list(zip(features, model.coef_))
    print(*feats, sep = "\n")
```

## Q6 - Inspecting Coefficients
In the cell below, write code that uses the `ridge_data` `DataFrame` to select which L2 penalty we would choose based on the evaluations we did in the previous section. You should print out the following values to help you answer the next questions! 
* **Q6.1** -  The best L2 penalty based on the model evaluations. Save this L2 penalty in a variable called `best_l2`.
* **Q6.2** - Take the best model and evaluate its error on the **test** dataset. Report the number as an RMSE stored in a variable called `test_rmse`.
* **Q6.3** - Call the `print_coefficients` function passing in the model itself and the features used so you can look at all of its coefficient values. You should save the number of coefficients that are 0 in a variable called `num_zero_coeffs_ridge`.

Use the next cell answer all three questions. You should also print out the values so you can inspect them.

To do this in `pandas`, you'll need to use the `idxmin()` function to find the index of the smallest value in a column and the `loc` property to access that index. As an example, suppose we had a `DataFrame` named `df`:

| a | b | c |
|---|---|---|
| 1 | 2 | 3 |
| 2 | 1 | 3 |
| 3 | 2 | 1 |

If we wrote the code 
```python
index = df['b'].idxmin()
row = df.loc[index]
```

It would first find the index of the smallest value in the `b` column and then uses the `.loc` property of the `DataFrame` to access that particular row. It will return a `Series` object (basically a Python dictionary) which means you can use syntax like `row['a']` to access a particular column of that row.


```python
### edTest(test_ridge_analysis) ###

# TODO Print information about best l2 model
best_index = ridge_data['validation_rmse'].idxmin()
best = ridge_data.loc[best_index]

best_l2 = best['l2_penalty']
best_ridge = best['model']
test_rmse = rmse(best_ridge, test_sales, test_price)

print(best_l2)
print()
print(test_rmse)
print()

print_coefficients(best_ridge, all_features)
num_zero_coeffs_ridge = 0
```

    10.0
    
    354624.84725194686
    
    ('bedrooms', -12814.960695689691)
    ('bedrooms_square', -5578.585627553784)
    ('bedrooms_sqrt', -19987.87864108882)
    ('bathrooms', -261.23310460046054)
    ('bathrooms_square', 139717.209305505)
    ('bathrooms_sqrt', -50659.473271559014)
    ('sqft_living', 23125.504167268835)
    ('sqft_living_square', 19304.655191207865)
    ('sqft_living_sqrt', 14413.319225996636)
    ('sqft_lot', 18909.076417532644)
    ('sqft_lot_square', 21375.20111150271)
    ('sqft_lot_sqrt', -25079.695226768865)
    ('floors', -15962.664049881396)
    ('floors_square', 13961.951626360837)
    ('floors_sqrt', -27220.383772045905)
    ('waterfront', 43800.14749776022)
    ('waterfront_square', 43800.14749776014)
    ('waterfront_sqrt', 43800.14749776015)
    ('view', -7513.023932352457)
    ('view_square', 11529.923304894022)
    ('view_sqrt', -12278.672850845955)
    ('condition', 4224.607744785495)
    ('condition_square', 4624.595688989869)
    ('condition_sqrt', 4026.4554446251964)
    ('grade', 44871.77670857402)
    ('grade_square', 73624.90237512426)
    ('grade_sqrt', 31314.956423922842)
    ('sqft_above', 22432.119503016154)
    ('sqft_above_square', 40851.91437009508)
    ('sqft_above_sqrt', 1391.5234697645212)
    ('sqft_basement', 1331.3789329041126)
    ('sqft_basement_square', -35037.047383280405)
    ('sqft_basement_sqrt', 25425.77455079102)
    ('yr_built', -26030.27641373636)
    ('yr_built_square', -25617.78684676176)
    ('yr_built_sqrt', -26232.13443988865)
    ('yr_renovated', 4718.090111503783)
    ('yr_renovated_square', 5102.806925373102)
    ('yr_renovated_sqrt', 4514.904467763932)


--- 
# LASSO Regression
In this section you will do basically the exact same analysis you did with Ridge Regression, but using LASSO Regression instead. It's okay if your code for this section looks very similar to your code for the last section. 

Remember that for LASSO we choose the parameters that minimize this quality metric instead 

$$\hat{w}_{LASSO} = \min_w RSS(w) + \lambda \left\lVert w \right\rVert_1$$

where $\left\lVert w \right\rVert_1 = \sum_{j=0}^D \lVert w_j \rVert$ is the L1 norm of the parameter vector.

## Q7) Train LASSO Models
We will use the same set of instructions for LASSO as we did for Ridge, except for the following differences. Please refer back to the Ridge Regression instructions and your code to see how these differences fit in!

* Use the [Lasso](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html#sklearn.linear_model.Lasso) model. Like before, the only parameters you need to pass in are `alpha` for the L1 penalty and `random_state=0`.
* The range L1 penalties should be $[10, 10^2, ..., 10^7]$. In Python, this is `np.logspace(1, 7, 7, base=10)`.
* The result should be stored in a `DataFrame` named `lasso_data`. All the columns should have the same name and corresponding values except the penalty column should be called `l1_penalty`.
* It is okay if your code prints some `ConvergenceWarning` warnings, these should not impact your results!.

You do not need to worry about your code being redundant with the last section for this part.


```python
### edTest(test_lasso) ###

from sklearn.linear_model import Lasso

# TODO Implement code to evaluate Ridge Regression with various l2 Penalties
reg_coefs = np.logspace(1, 7, 7, base=10)
data = []

for reg_coef in reg_coefs:
    lasso_model = Lasso(alpha = reg_coef).fit(train_sales, train_price)
    train_rmse = rmse(lasso_model, train_sales, train_price)
    validation_rmse = rmse(lasso_model, validation_sales, validation_price)
    data.append({
        'l1_penalty': reg_coef,
        'model': lasso_model,
        'train_rmse': train_rmse,
        'validation_rmse' : validation_rmse,
    }
    )
    
lasso_data = pd.DataFrame(data)
lasso_data

```

    /usr/lib/python3.9/site-packages/sklearn/linear_model/_coordinate_descent.py:530: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 1652622332201.121, tolerance: 1912867203.4052832
      model = cd_fast.enet_coordinate_descent(
    /usr/lib/python3.9/site-packages/sklearn/linear_model/_coordinate_descent.py:530: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 1000006149477.815, tolerance: 1912867203.4052832
      model = cd_fast.enet_coordinate_descent(
    /usr/lib/python3.9/site-packages/sklearn/linear_model/_coordinate_descent.py:530: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 2342567368.334961, tolerance: 1912867203.4052832
      model = cd_fast.enet_coordinate_descent(





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
      <th>l1_penalty</th>
      <th>model</th>
      <th>train_rmse</th>
      <th>validation_rmse</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>10.0</td>
      <td>Lasso(alpha=10.0)</td>
      <td>151336.667973</td>
      <td>335371.150569</td>
    </tr>
    <tr>
      <th>1</th>
      <td>100.0</td>
      <td>Lasso(alpha=100.0)</td>
      <td>152164.419039</td>
      <td>323670.679137</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1000.0</td>
      <td>Lasso(alpha=1000.0)</td>
      <td>156360.395434</td>
      <td>285201.302593</td>
    </tr>
    <tr>
      <th>3</th>
      <td>10000.0</td>
      <td>Lasso(alpha=10000.0)</td>
      <td>169912.542560</td>
      <td>271138.560249</td>
    </tr>
    <tr>
      <th>4</th>
      <td>100000.0</td>
      <td>Lasso(alpha=100000.0)</td>
      <td>239553.336550</td>
      <td>340385.511537</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1000000.0</td>
      <td>Lasso(alpha=1000000.0)</td>
      <td>357105.698956</td>
      <td>533861.713077</td>
    </tr>
    <tr>
      <th>6</th>
      <td>10000000.0</td>
      <td>Lasso(alpha=10000000.0)</td>
      <td>357105.698956</td>
      <td>533861.713077</td>
    </tr>
  </tbody>
</table>
</div>



Like before, let's look at how the L1 penalty affects the performance.


```python
# Plot the validation RMSE as a blue line with dots

plt.plot(lasso_data['l1_penalty'], lasso_data['validation_rmse'],
         'b-^', label='Validation')

# Plot the train RMSE as a red line dots
plt.plot(lasso_data['l1_penalty'], lasso_data['train_rmse'],
         'r-o', label='Train')

# Make the x-axis log scale for readability
plt.xscale('log')

# Label the axes and make a legend
plt.xlabel('l1_penalty')
plt.ylabel('RMSE')
plt.legend()
```




    <matplotlib.legend.Legend at 0x7fcacfd91e50>




    
![png](output_24_1.png)
    


## Q8 - Inspecting Coefficients
Like before, in the cell below, write code that uses the `lasso_data` `DataFrame` to select which L1 penalty we would choose based on the evaluations we did in the previous section. You should print out the following values to help you answer the next questions! 

* **Q8.1** -  TThe best L1 penalty based on the model evaluations. Save this L1 penalty in a variable called `best_l1`.
* **Q8.2** - Take the best model and evaluate it on the **test** dataset and report its RMSE. Report the number as an RMSE stored in a variable called `test_rmse`.
* **Q8.3** - Call the `print_coefficients` function passing in the model itself and the features used so you can look at all of its coefficient values. Note some of the values are `-0.0` which is the same as `0.0` for our purposes. You should save the number of coefficients that are 0 in a variable called `num_zero_coeffs_lasso`.


```python
### edTest(test_lasso_analysis) ###

# TODO Print information about best l1 model
best_index = lasso_data['validation_rmse'].idxmin()
best = lasso_data.loc[best_index]

best_l1 = best['l1_penalty']
best_lasso = best['model']
test_rmse = rmse(best_lasso, test_sales, test_price)

print(best_l1)
print()
print(test_rmse)
print()

print_coefficients(best_lasso, all_features)
num_zero_coeffs_lasso = 29
```

    10000.0
    
    344434.8333379938
    
    ('bedrooms', -0.0)
    ('bedrooms_square', -0.0)
    ('bedrooms_sqrt', -19088.36032905744)
    ('bathrooms', -0.0)
    ('bathrooms_square', 108823.89843473864)
    ('bathrooms_sqrt', -0.0)
    ('sqft_living', 0.0)
    ('sqft_living_square', 0.0)
    ('sqft_living_sqrt', 0.0)
    ('sqft_lot', 0.0)
    ('sqft_lot_square', 7986.231414584601)
    ('sqft_lot_sqrt', 0.0)
    ('floors', -0.0)
    ('floors_square', -0.0)
    ('floors_sqrt', -12569.995812596218)
    ('waterfront', 124085.67950514512)
    ('waterfront_square', 0.0)
    ('waterfront_sqrt', 0.0)
    ('view', 0.0)
    ('view_square', 0.0)
    ('view_sqrt', -0.0)
    ('condition', 0.0)
    ('condition_square', 0.0)
    ('condition_sqrt', 0.0)
    ('grade', 0.0)
    ('grade_square', 147501.36528659248)
    ('grade_sqrt', 0.0)
    ('sqft_above', 0.0)
    ('sqft_above_square', 71224.80787057214)
    ('sqft_above_sqrt', 0.0)
    ('sqft_basement', 0.0)
    ('sqft_basement_square', -0.0)
    ('sqft_basement_sqrt', 0.0)
    ('yr_built', -63287.382821229934)
    ('yr_built_square', -0.0)
    ('yr_built_sqrt', -17250.36953619334)
    ('yr_renovated', 0.0)
    ('yr_renovated_square', 391.8053626652076)
    ('yr_renovated_sqrt', 0.0)


Let's look at which coefficients ended up having a 0 coefficient. In the cell below, we print the name of all features with coefficient 0. Note, we actually have to check if it is near 0 since numeric computations in Python sometimes yield slight rounding errors (e.g., how 1/3 is .333333333333 and that can't be represented precisely in a computer)


```python
# Note: You will need to replace row['model'] with code similar to the last cell
# to get the best model from the table

for feature, coef in zip(all_features, best['model'].coef_):
  if abs(coef) <= 10 ** -17:
    print(feature)
```

    bedrooms
    bedrooms_square
    bathrooms
    bathrooms_sqrt
    sqft_living
    sqft_living_square
    sqft_living_sqrt
    sqft_lot
    sqft_lot_sqrt
    floors
    floors_square
    waterfront_square
    waterfront_sqrt
    view
    view_square
    view_sqrt
    condition
    condition_square
    condition_sqrt
    grade
    grade_sqrt
    sqft_above
    sqft_above_sqrt
    sqft_basement
    sqft_basement_square
    sqft_basement_sqrt
    yr_built_square
    yr_renovated
    yr_renovated_sqrt


### Q9) Based on our experiments in this assignment, which of these models (LinearRegression, Ridge, Lasso) would we expect to have the lowest error on future error. 

Choose the best option that applies. Save your answer in a variable called `q9` with a value that's a string with the one of the following choices:

* `'LinearRegression'`
* `'Ridge'`
* `'Lasso'`
* `"Can't tell"`

For example, you might write

```
q9 = 'Answer'
```



```python
### edTest(test_q9) ###
q9 = 'Lasso'
# TODO Select the model

```
