# Median House Value Estimator
Median House Value Estimation in python.

In this project I use a few regressor algorithms to estimate the median value of a house in a district.

## Getting Started
These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites
* python 3.8.8

    * This setup requires that your machine has python 3.8 installed on it. you can refer to this url https://www.python.org/downloads/ to download python. Once you have python downloaded and installed, you will need to setup PATH variables (if you want to run python program directly, detail instructions are below in how to run software section). To do that check [this](https://www.pythoncentral.io/add-python-to-path-python-is-not-recognized-as-an-internal-or-external-command/).

   * Setting up PATH variable is optional as you can also run program without it and more instruction are given below on this topic.

    * Then you should run: `pip install requirements.txt`

* Second and easier option is to download anaconda and use its anaconda prompt to run the commands. To install anaconda check this url https://www.anaconda.com/download/.

## Dataset Used

### California Housing

>>#### Source

>>>This dataset is a modified version of the California Housing dataset available from [Luís Torgo's page](http://www.dcc.fc.up.pt/~ltorgo/Regression/cal_housing.html) (University of Porto). Luís Torgo obtained it from the StatLib repository (which is closed now). The dataset may also be downloaded from StatLib mirrors.

>>>This dataset appeared in a 1997 paper titled *Sparse Spatial Autoregressions* by Pace, R. Kelley and Ronald Barry, published in the *Statistics and Probability Letters* journal. They built it using the 1990 California census data. It contains one row per census block group. A block group is the smallest geographical unit for which the U.S. Census Bureau publishes sample data (a block group typically has a population of 600 to 3,000 people).

>### Tweaks
>>The dataset in this directory is almost identical to the original, with two differences:

>>* 207 values were randomly removed from the `total_bedrooms` column, so we can discuss what to do with missing data.
>>* An additional categorical attribute called `ocean_proximity` was added, indicating (very roughly) whether each block group is near the ocean, near the Bay area, inland or on an island. This allows discussing what to do with categorical data.

>#### Columns 
```
    longitude             
    latitude              
    housing_median_age    
    total_rooms           
    total_bedrooms        
    population            
    households            
    median_income         
    median_house_value - the target attribute   
    ocean_proximity 

```
## Creating a Test set
I created the test before before looking at the data, I am creating it at the beginning to avoid data snooping. I do not want to find what my mind thinks is interesting and use that in deciding which machine learning model to use.

* I made income categories because income is very important in estimating the housing value. I used the below code snippet.

    ```python
    housing['income_cat'] = np.ceil(housing.median_income/1.5) 
    housing.income_cat.where(housing.income_cat<5, 5, inplace=True)
    ```

* I used the income categories to randomly stratify and create representative training and test sets.
    ```python
    from sklearn.model_selection import StratifiedShuffleSplit

    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, test_index in split.split(housing, housing.income_cat):
        train_data = housing.iloc[train_index]
        test_data = housing.iloc[test_index]
    ```
* I removed the income category from both and train and testing set.
* I saved the training and testing sets to files for easy access.

## EDA
### The below visualisations are merely highlights of exploratory analyses.
I looked at distributions and counts of the attributes.

![Histograms](images/counts.png)

I looked at the geographical distribution of housing value and populations. 

![GeographicalScatter](images/scatter.png)

The median income has the most highest correlation with median house value. 
![Scatter](images/scatter_mhv_vs_mi.png)


## Model Building
I first made a datapipeline.

* What the pipeline does:
    * Selects the numerical attributes from the datasets.
    * Fits the missing values with the median value.
    * Combines some of the attributes to make new ones.
        * Created `rooms_per_household = total_room/total_bedrooms` 
        * Created `population_per_household = population/household`
        Created `bedrooms_per_room = total_bedrooms/total_rooms` with option to use it or not.
    * Scales the data using standard scaler.
    * Select the categorical attributes from the datasets.
    * Create LabelBinary

I tried three models and evaluated them using the RMSE.

* Linear Regression - I used it as a baseline model.
* DecisionTreeRegressor - I thought it could detect the non linear relationships between the attributes and the target label.
* RandomForestRegressor - I thought since it is an ensemble model it will do very well.

The linear model was underfitting to the training data.

The Decision Tree model was overfitting to the train data. The RMSE on the training data was zero.

The Random Forest Regressor is also overfitting to the training data.

I tried cross validation on the models.

```python
score = cross_val_score(
    lin_reg,
    housing_num,
    housing_labels,
    scoring='neg_mean_squared_error',
    cv=10
)
```
The RMSE of the models is as follows:

* Linear Regression = 70118.16
* DecisionTreeRegressor = 70086.40
* RandomForestRegressor = 50281.19

### Hyperparameter Tuning 
I chose to do hyperparameter tuning on the RandomForestRegressor.

```python
param_grid = [
    {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
    {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
]

grid_search = GridSearchCV(
    forest_reg,
    param_grid,
    cv=5,
    scoring='neg_mean_squared_error'
)

grid_search.fit(
    housing_prepared,
    housing_labels
)
```

## Model Perfomance
The RandomForestRegressor has improved after tuning.

The best model has `max_features = 6` and `n_estimators = 30`. 

I used the testing data to test the best model.

I passed the testing data through the pipeline.

The model's performance on the test data
* The RMSE = 47970.72.


# _______________________________________________________________


