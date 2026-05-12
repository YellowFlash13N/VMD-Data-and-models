import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, LeaveOneOut, cross_val_predict
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

df = pd.read_csv('VMD Data.csv')
X = df[['T','P','Q','C']]
Y = df[['F','SEC']]
scaler_X = StandardScaler()
scaler_Y = StandardScaler()
X_scaled = scaler_X.fit_transform(X)
Y_scaled = scaler_Y.fit_transform(Y)

param_grid = {
    'n_estimators': [200,500,800],
    'max_depth': [None,5,10],
    'min_samples_leaf': [1,2,4] 
}
base_rf = RandomForestRegressor(random_state=42, n_jobs=-1)
grid = GridSearchCV(base_rf, param_grid, cv=5, scoring='r2', n_jobs=-1, verbose=0)

grid.fit(X_scaled, Y_scaled)
best_model = grid.best_estimator_
print('Best RF params:', grid.best_params_)

loo = LeaveOneOut()

Y_pred_scaled = cross_val_predict(best_model,X_scaled,Y_scaled,cv=loo,n_jobs=-1)

Y_orig = scaler_Y.inverse_transform(Y_scaled)
Y_pred_orig = scaler_Y.inverse_transform(Y_pred_scaled)

r2_all = r2_score(Y_orig,Y_pred_orig,multioutput='raw_values')
rmse_all = np.sqrt(np.mean((Y_orig-Y_pred_orig)**2 , axis = 0))
mae_all = np.abs(np.mean((Y_orig-Y_pred_orig) , axis = 0))

print(f'R^2 (Flux,SEC): {r2_all}')
print(f'RMSE (FLux,SEC): {rmse_all}')
print(f'MAE (Flux, SEC): {mae_all}')

flux_actual = Y_orig[:,0]
flux_predicted = Y_pred_orig[:,0]

plt.figure(figsize = (7,5))
plt.scatter(flux_actual,flux_predicted, c = 'blue', alpha = 0.6)
plt.plot([min(flux_actual),max(flux_actual)],[min(flux_actual),max(flux_actual)],'r--',lw=2)
plt.xlabel('Actual Permeate Flux (kg/(m^2.h))')
plt.ylabel('Predicted Permeate Flux (kg/(m^2.h))')
plt.title('Random Forests (LOOCV) - Predicted v/s Actual Flux')
plt.grid(True)
plt.show()