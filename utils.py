import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import plotly.express as px
from plotly.subplots import make_subplots


# convenience function to evaluate data set during data analysis phase
def eval_categories(df):
  """
  Used in Exploratory Data Analysis to understand how the categorical data is distributed.
  For example, are there lots of nulls? A lot of unique values? Or a few set of values well distributed throughout the set.

  Args:
    df: Input data frame with categorical features (dtype = "object")

  Returns:
    a new data frame containing descriptive stats for each categorical feature in the input data frame
  """
  category_features = df.select_dtypes(include='object').columns.to_list()
  n = df.shape[0]
  to_percent = lambda x: f'{x/n:.1%}'
  return pd.DataFrame(
    {
      'unique': [len(np.unique(df[col].dropna())) for col in category_features],
      'top': [df.groupby(col)['price'].count().sort_values(ascending=False).head(1).index.to_list()[0] for col in category_features],
      'count': [to_percent(df.groupby(col)['price'].count().sort_values(ascending=False).head(1).values[0]) for col in category_features],
      'nulls': [to_percent(df[col].isnull().sum()) for col in category_features],
    },
    index=category_features
  )


# convenience function that computes the MSE and R2 coefficient given a model
def train_and_evaluate_model(model, name, X, y):
  """
  A convenience function that standardizes the steps to train and validate a model.

  Args:
    model: the model to analyze
    name: a name to use as a label for the model
    X: features
    y: target

  Returns:
    mse_train - Mean Squared Error on training data set
    mse_test - Mean Squared Error on test data set
    r2_train - R2 coefficient on training data set
    r2_test - R2 coefficient on test data set
  """
  # split the data into training and test data sets
  X_train, X_test, y_train, y_test = train_test_split(X, y)

  # train the model
  model.fit(X_train, y_train)

  # evaluate the model
  mse_train = mean_squared_error(model.predict(X_train), y_train)
  mse_test = mean_squared_error(model.predict(X_test), y_test)

  # report on the correlation coefficient (R2 score)
  r2_train = model.score(X_train, y_train)
  r2_test = model.score(X_test, y_test)

  print(f'MSE on {name} training data set', f'{mse_train:.3f}')
  print(f'MSE on {name} test data set', f'{mse_test}')
  print(f'R2 on {name} training data set', f'{r2_train}')
  print(f'R2 on {name} test data set', f'{r2_test}')

  return mse_train, mse_test, r2_train, r2_test


# convenience function to plot set of features against price
def plot_features(df, features: list[str], color='cluster'):
  """
  Utility function to plot selected features against price.

  Args:
    df: data frame
    features: list of feature names to plot
  
  Returns:
    Plot of features vs. price
  """
  figures = [
    px.scatter(df, x=feature, y='price', color=color) for feature in features
  ]

  fig = make_subplots(
    rows=1, cols=len(figures), 
    shared_yaxes=True, 
    y_title='Price', 
    subplot_titles=features
  ) 

  for i, figure in enumerate(figures):
      for trace in range(len(figure["data"])):
          fig.append_trace(figure["data"][trace], row=1, col=i+1)

  fig.show()