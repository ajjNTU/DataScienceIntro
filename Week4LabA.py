# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Sklearn libraries for different types of regression and splitting data
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error, r2_score

# Bonus Task
from sklearn.ensemble import RandomForestRegressor

# Load the dataset
df = pd.read_csv('housing.csv')
df = df[df['median_house_value'] < 500001]

# Take a first look at your data.
print(df.head())

# Apply One-Hot Encoding to 'ocean_proximity'
df = pd.get_dummies(df, columns=['ocean_proximity'])

# Fill missing values with mean column values
df.fillna(df.mean(), inplace=True)

# Separate target variable and features again
X = df.drop('median_house_value', axis=1)  # Features
y = df['median_house_value']  # Target variable

# Split the dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)  # 50% split for training and testing

# 1. Train the OLS model
ols_model = LinearRegression()
ols_model.fit(X_train, y_train)

# 2. Predict on Test Data
ols_predictions = ols_model.predict(X_test)

# Initialize an empty DataFrame to store the R2 scores
r2_scores = pd.DataFrame(columns=['Model', 'R2 Score'])

# Evaluate the OLS model
ols_r2 = r2_score(y_test, ols_predictions)
r2_scores = r2_scores.append({'Model': 'OLS', 'R2 Score': ols_r2}, ignore_index=True)
print(f"OLS R2 Score: {ols_r2}")

# Plot true vs predicted values
plt.scatter(y_test, ols_predictions)
plt.xlabel('True Values')
plt.ylabel('Predicted Values')
plt.title('OLS: True vs Predicted Values')
plt.show()

# Print coefficients
coeff_df = pd.DataFrame(ols_model.coef_, X.columns, columns=['Coefficient'])
print(coeff_df)

# Create a Ridge regression object
ridge = Ridge(alpha=1.0)

# Train the model using the training sets
ridge.fit(X_train, y_train)

# Make predictions using the testing set
y_pred_ridge = ridge.predict(X_test)

# Evaluate the Ridge model
ridge_r2 = r2_score(y_test, y_pred_ridge)
r2_scores = r2_scores.append({'Model': 'Ridge', 'R2 Score': ridge_r2}, ignore_index=True)
print('Ridge R2 Score:', ridge_r2)

# Print coefficients
coeff_df = pd.DataFrame(ridge.coef_, X.columns, columns=['Coefficient'])
print(coeff_df)

# Create a Lasso regression object
lasso = Lasso(alpha=0.1)

# Train the model using the training sets
lasso.fit(X_train, y_train)

# Make predictions using the testing set
y_pred_lasso = lasso.predict(X_test)

# Evaluate the Lasso model
lasso_r2 = r2_score(y_test, y_pred_lasso)
r2_scores = r2_scores.append({'Model': 'Lasso', 'R2 Score': lasso_r2}, ignore_index=True)
print('Lasso R2 Score:', lasso_r2)

# Print coefficients
coeff_df = pd.DataFrame(lasso.coef_, X.columns, columns=['Coefficient'])
print(coeff_df)

# Create an ElasticNet regression object
elastic = ElasticNet(alpha=0.1, l1_ratio=0.5)

# Train the model using the training sets
elastic.fit(X_train, y_train)

# Make predictions using the testing set
y_pred_elastic = elastic.predict(X_test)

# Evaluate the ElasticNet model
elastic_r2 = r2_score(y_test, y_pred_elastic)
r2_scores = r2_scores.append({'Model': 'ElasticNet', 'R2 Score': elastic_r2}, ignore_index=True)
print('ElasticNet R2 Score:', elastic_r2)



# Print coefficients
coeff_df = pd.DataFrame(elastic.coef_, X.columns, columns=['Coefficient'])
print(coeff_df)

# Print the DataFrame
print(r2_scores)


from sklearn.metrics import mean_absolute_error

# Initialize an empty dataframe to store the scores
scores = pd.DataFrame(columns=['Model', 'R2 Score', 'MSE', 'MAE'])

# Add scores for OLS
scores = pd.concat([scores, pd.DataFrame([['OLS',
                                           r2_score(y_test, ols_predictions),
                                           mean_squared_error(y_test, ols_predictions),
                                           mean_absolute_error(y_test, ols_predictions)]],
                                          columns=scores.columns)], ignore_index=True)

# Add scores for Ridge
scores = pd.concat([scores, pd.DataFrame([['Ridge',
                                           r2_score(y_test, y_pred_ridge),
                                           mean_squared_error(y_test, y_pred_ridge),
                                           mean_absolute_error(y_test, y_pred_ridge)]],
                                          columns=scores.columns)], ignore_index=True)

# Add scores for Lasso
scores = pd.concat([scores, pd.DataFrame([['Lasso',
                                           r2_score(y_test, y_pred_lasso),
                                           mean_squared_error(y_test, y_pred_lasso),
                                           mean_absolute_error(y_test, y_pred_lasso)]],
                                          columns=scores.columns)], ignore_index=True)

# Add scores for ElasticNet
scores = pd.concat([scores, pd.DataFrame([['ElasticNet',
                                           r2_score(y_test, y_pred_elastic),
                                           mean_squared_error(y_test, y_pred_elastic),
                                           mean_absolute_error(y_test, y_pred_elastic)]],
                                          columns=scores.columns)], ignore_index=True)

# Print the scores
print(scores)

from sklearn.model_selection import GridSearchCV

# Define the hyperparameters we want to tune
hyperparameters = {'alpha': [0.1, 1, 10, 100, 1000], 'l1_ratio': [0.1, 0.5, 0.7, 0.9, 0.95, 0.99, 1]}

# We will now use GridSearchCV for ElasticNet
grid = GridSearchCV(ElasticNet(), hyperparameters, cv=5)

# Fit the model
grid.fit(X_train, y_train)

# Print the best parameters and the corresponding score
print("Best parameters: ", grid.best_params_)
print("Best score: ", grid.best_score_)

# Predict on the test data using the best parameters
y_pred_elastic = grid.predict(X_test)

# Print R2 score, MSE and MAE
print('ElasticNet R2 Score:', r2_score(y_test, y_pred_elastic))
print('ElasticNet MSE:', mean_squared_error(y_test, y_pred_elastic))
print('ElasticNet MAE:', mean_absolute_error(y_test, y_pred_elastic))

# Define the hyperparameters we want to tune
hyperparameters = {'alpha': [0.1, 1, 10, 100, 1000]}

# GridSearchCV for Ridge
grid_ridge = GridSearchCV(Ridge(), hyperparameters, cv=5)
grid_ridge.fit(X_train, y_train)
print("Ridge - Best parameters: ", grid_ridge.best_params_)
print("Ridge - Best score: ", grid_ridge.best_score_)

# Predict on the test data using the best parameters
y_pred_ridge = grid_ridge.predict(X_test)
print('Ridge R2 Score:', r2_score(y_test, y_pred_ridge))
print('Ridge MSE:', mean_squared_error(y_test, y_pred_ridge))
print('Ridge MAE:', mean_absolute_error(y_test, y_pred_ridge))

# GridSearchCV for Lasso
grid_lasso = GridSearchCV(Lasso(), hyperparameters, cv=5)
grid_lasso.fit(X_train, y_train)
print("Lasso - Best parameters: ", grid_lasso.best_params_)
print("Lasso - Best score: ", grid_lasso.best_score_)

# Predict on the test data using the best parameters
y_pred_lasso = grid_lasso.predict(X_test)
print('Lasso R2 Score:', r2_score(y_test, y_pred_lasso))
print('Lasso MSE:', mean_squared_error(y_test, y_pred_lasso))
print('Lasso MAE:', mean_absolute_error(y_test, y_pred_lasso))


from sklearn.ensemble import RandomForestRegressor

# Create a RandomForestRegressor object
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the model using the training sets
rf_model.fit(X_train, y_train)

# Predict on Test Data
rf_predictions = rf_model.predict(X_test)

# Evaluate the RandomForest model
rf_r2 = r2_score(y_test, rf_predictions)
rf_mse = mean_squared_error(y_test, rf_predictions)
rf_mae = mean_absolute_error(y_test, rf_predictions)

# Print results
print(f"Random Forest R2 Score: {rf_r2}")
print(f"Random Forest MSE: {rf_mse}")
print(f"Random Forest MAE: {rf_mae}")

# Store the results in the df_scores DataFrame
scores = scores.append({'Model': 'Random Forest', 'R2 Score': rf_r2, 'MSE': rf_mse, 'MAE': rf_mae}, ignore_index=True)

print(scores)