import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

train_data = pd.read_csv('data/train.csv')
test_data = pd.read_csv('data/test.csv')
X_train, X_test, Y_train, Y_test = train_data.drop(columns=['price']),test_data.drop(columns=['price']),train_data['price'],test_data['price']

# linear regression
linear_model = LinearRegression()
linear_model.fit(X_train,Y_train)
Y_pred_linear = linear_model.predict(X_test)

mse = mean_squared_error(Y_test,Y_pred_linear)
r2 = r2_score(Y_test, Y_pred_linear)
rmse = np.sqrt(mse)
print("Linear Regression:\n")
print(f'Mean Squared Error: {mse:.4f}')
print(f'R-squared: {r2:.4f}')
print(f'Root Mean Squared Error: {rmse:.4f}\n')

# using regression trees since numerical output is needed
decision_model = DecisionTreeRegressor(random_state=42)
decision_model.fit(X_train,Y_train)
Y_pred_decision = decision_model.predict(X_test)

mse = mean_squared_error(Y_test,Y_pred_decision)
r2 = r2_score(Y_test, Y_pred_decision)
rmse = np.sqrt(mse)
print("Decision trees regressor:\n")
print(f'Mean Squared Error: {mse:.4f}')
print(f'R-squared: {r2:.4f}')
print(f'Root Mean Squared Error: {rmse:.4f}\n')

# using random forest
forest_model = RandomForestRegressor(n_estimators=100,random_state=30)
forest_model.fit(X_train,Y_train)
Y_pred_forest = forest_model.predict(X_test)

mse = mean_squared_error(Y_test,Y_pred_forest)
r2 = r2_score(Y_test, Y_pred_forest)
rmse = np.sqrt(mse)
print("Random forest regressor:\n")
print(f'Mean Squared Error: {mse:.4f}')
print(f'R-squared: {r2:.4f}')
print(f'Root Mean Squared Error: {rmse:.4f}')