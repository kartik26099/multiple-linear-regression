import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import statsmodels.api as sms

# Load the dataset
df = pd.read_csv(r"D:\coding journey\aiml\python\udemy\Machine Learning A-Z (Codes and Datasets)\Part 2 - Regression\Section 5 - Multiple Linear Regression\Python\50_Startups.csv")

# Print the count of missing values for each column
print(df.isna().sum(axis=0))

# Extract the independent variables (features) and the dependent variable (target)
x = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

# Encode categorical data (the 'State' column at index 3) using OneHotEncoder
ct = ColumnTransformer(transformers=[("encoder", OneHotEncoder(), [3])], remainder="passthrough")
x = np.array(ct.fit_transform(x))

# Split the dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)

# Fit the Linear Regression model to the training set
lr = LinearRegression()
lr.fit(x_train, y_train)

# Predict the test set results
y_predict = lr.predict(x_test)

# Print the predicted and actual values side by side
print(np.concatenate((y_predict.reshape(len(y_predict), 1), y_test.reshape(len(y_test), 1)), 1))

# Print the shape of the features matrix
print(x.shape)

# Add a column of ones to x (for the intercept term in the regression model)
x = np.append(np.ones((x.shape[0], 1)).astype(int), x, axis=1)
x_opt = x

# Print the updated features matrix
print(x_opt)
print("\n", x)

# Convert x_opt to float64 to ensure compatibility with OLS
x_opt = np.asarray(x_opt, dtype=np.float64)

# Convert y to float64 to ensure compatibility with OLS
y = np.asarray(y, dtype=np.float64)

# Fit the OLS regression model using statsmodels
regressor_ols = sms.OLS(endog=y, exog=x_opt).fit()

# Print the summary of the regression model
print(regressor_ols.summary())

# Perform backward elimination by removing the predictor with the highest p-value
x_opt = x[:, [0, 1, 2, 3, 4, 6]]
print("\n", x_opt)

# Convert x_opt to float64 to ensure compatibility with OLS
x_opt = np.asarray(x_opt, dtype=np.float64)

# Convert y to float64 to ensure compatibility with OLS
y = np.asarray(y, dtype=np.float64)

# Fit the OLS regression model using statsmodels
regressor_ols = sms.OLS(endog=y, exog=x_opt).fit()

# Print the summary of the regression model
print(regressor_ols.summary())

# Perform further backward elimination by removing another predictor with the highest p-value
x_opt = x[:, [0, 1, 2, 3, 4]]
print("\n", x_opt)

# Convert x_opt to float64 to ensure compatibility with OLS
x_opt = np.asarray(x_opt, dtype=np.float64)

# Convert y to float64 to ensure compatibility with OLS
y = np.asarray(y, dtype=np.float64)

# Fit the OLS regression model using statsmodels
regressor_ols = sms.OLS(endog=y, exog=x_opt).fit()

# Print the summary of the regression model
print(regressor_ols.summary())

# Split the dataset with the optimal features into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x_opt, y, test_size=0.2, random_state=1)

# Fit a new Linear Regression model to the training set with the optimal features
lr2 = LinearRegression()
lr2.fit(x_train, y_train)

# Predict the test set results with the optimal features
y2_predict = lr2.predict(x_test)

# Print the predicted values
print(y2_predict)

# Print the actual values
print(y_test)
#to find the coefficient
print(lr2.coef_)
#to find the intercept(b0)

print(lr2.intercept_)
