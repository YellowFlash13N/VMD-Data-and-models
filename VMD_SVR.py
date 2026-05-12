import pandas as pd
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split , GridSearchCV
from sklearn.metrics import mean_absolute_error , mean_squared_error , r2_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('master_dataset.csv')
X = df[['T','P','Q']]
Y = df['F']
scaler_X = StandardScaler()
scaler_Y = StandardScaler()
X_scaled = scaler_X.fit_transform(X)
Y_scaled = scaler_Y.fit_transform(Y.values.reshape(-1,1)).ravel()
X_train, X_test, Y_train, Y_test = train_test_split(X_scaled,Y_scaled,test_size=0.33,random_state=42)
#Training the RBF kernel using k folds cross validation
param_grid = {
    'C':[10,100,500],
    'gamma':['scale',0.01,0.1],
    'epsilon':[0.001,0.01,0.1]
}
svr = SVR(kernel = 'rbf')
grid = GridSearchCV(svr,param_grid,cv = 5,scoring = 'r2')
grid.fit(X_train,Y_train)
best_model = grid.best_estimator_

Y_pred_scaled = best_model.predict(X_test)
Y_test_orig = scaler_Y.inverse_transform(Y_test.reshape(-1,1)).ravel()
Y_pred_orig = scaler_Y.inverse_transform(Y_pred_scaled.reshape(-1,1)).ravel()
r2 = r2_score(Y_test_orig,Y_pred_orig)
rmse = (mean_squared_error(Y_test_orig,Y_pred_orig))**(1/2)
mae = mean_absolute_error(Y_test_orig,Y_pred_orig)

print(f'R^2 = {r2:5f}, RMSE = {rmse:5f}, MAE = {mae:5f}')

plt.figure(figsize = (7,5))
plt.scatter(Y_test_orig, Y_pred_orig, c = 'blue', alpha = 0.6)
plt.plot([0,max(Y_test_orig)],[0,max(Y_test_orig)],'r--',lw=2)
plt.xlabel('Actual Permeate Flux (kg/(m^2.h))')
plt.ylabel('Predicted Permeate Flux (kg/(m^2.h))')
plt.title('SVR (RBF) Predicted v/s Actual Flux')
plt.grid(True)
plt.show()