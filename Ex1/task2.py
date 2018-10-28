from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import numpy as np
import pandas as pd

# LOADING  + SHAPING DATA
X = pd.read_csv('./embryonic_data_10genes.txt', sep='\t', index_col=0).transpose()
y_raw = pd.read_csv('./labels.txt', sep='\t', header = None)
y_raw.columns = ["EDay"]
y = y_raw.values.flatten()

lin = linear_model.LinearRegression()
lin.fit(X,y)

r_sq = lin.score(X, y)
rmse = np.sqrt(mean_squared_error(y, lin.predict(X)))
coef = lin.coef_

genes = ["BCL2L10", "ZAR1L", "C3orf56", "BTG4", "TUBB8", "SH2D1B", "C9orf116", "TMEM132B", "CA4", "FAM19A4"]


print(r_sq)
print(rmse)
print("coeff:")
for i in range(0,10):
    print(genes[i], " & ", round(coef[i]*1000,4), "\\\\")

