from sklearn import linear_model
from sklearn.model_selection import cross_val_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import numpy as np
import pandas as pd

# LOADING  + SHAPING DATA
X = pd.read_csv('./embryonic_data_10genes.txt', sep='\t', index_col=0)
y = pd.read_csv('./labels.txt', sep='\t', header = None)
y.columns = ["EDay"]


# SHAPING 3_5 DATA
X_3_5 = X.loc[:, X.columns.str.contains('E3') | X.columns.str.contains('E5')].transpose()
y_3_5 = y.loc[(y["EDay"] == 3) | (y["EDay"] == 5)].values.flatten()


# SHAPING 4, 6 and 7 DATA
X_4 = X.loc[:, X.columns.str.contains('E4')].transpose()
X_6 = X.loc[:, X.columns.str.contains('E6')].transpose()
X_7 = X.loc[:, X.columns.str.contains('E7')].transpose()

#Defining models
logreg = linear_model.LogisticRegression(max_iter=2000, solver='liblinear')
fld = LinearDiscriminantAnalysis()

# Cross-validation
scores_logreg = cross_val_score(logreg, X_3_5, y_3_5)
scores_fld = cross_val_score(fld, X_3_5, y_3_5)
avg_score_logreg = np.mean(scores_logreg)
avg_score_fld = np.mean(scores_fld)

# Fitting
logreg.fit(X_3_5,y_3_5)
fld.fit(X_3_5,y_3_5)


#Checking standardized coefficients for contribution per gene
coeff_logreg = logreg.coef_
coeff_fld = fld.coef_

#lr_col = ["LR"] ++ coeff_logreg
#fld_col = ["FLD"].append(coeff_fld)

#ltx_matrix = [["","BCL2L10", "ZAR1L", "C3orf56","BTG4", "TUBB8","SH2D1B","C9orf116","TMEM132B","CA4","FAM19A4"], lr_col, fld_col]

#Appying on other classes
predicted_4_logreg = logreg.predict(X_4)
predicted_6_logreg = logreg.predict(X_6)
predicted_7_logreg = logreg.predict(X_7)

nb_3s_for_4_logreg, nb_5s_for_4_logreg = len(predicted_4_logreg[predicted_4_logreg == 3]), len(predicted_4_logreg[predicted_4_logreg == 5])
nb_3s_for_6_logreg, nb_5s_for_6_logreg = len(predicted_6_logreg[predicted_6_logreg == 3]), len(predicted_6_logreg[predicted_6_logreg == 5])
nb_3s_for_7_logreg, nb_5s_for_7_logreg = len(predicted_7_logreg[predicted_7_logreg == 3]), len(predicted_7_logreg[predicted_7_logreg == 5])

predicted_4_fld = fld.predict(X_4)
predicted_6_fld = fld.predict(X_6)
predicted_7_fld = fld.predict(X_7)

nb_3s_for_4_fld, nb_5s_for_4_fld = len(predicted_4_fld[predicted_4_fld == 3]), len(predicted_4_fld[predicted_4_fld == 5])
nb_3s_for_6_fld, nb_5s_for_6_fld = len(predicted_6_fld[predicted_6_fld == 3]), len(predicted_6_fld[predicted_6_fld == 5])
nb_3s_for_7_fld, nb_5s_for_7_fld = len(predicted_7_fld[predicted_7_fld == 3]), len(predicted_7_fld[predicted_7_fld == 5])

percentage_of_5_for_4_logreg = nb_5s_for_4_logreg/(nb_5s_for_4_logreg+nb_3s_for_4_logreg)
percentage_of_5_for_6_logreg = nb_5s_for_6_logreg/(nb_5s_for_6_logreg+nb_3s_for_6_logreg)
percentage_of_5_for_7_logreg = nb_5s_for_7_logreg/(nb_5s_for_7_logreg+nb_3s_for_7_logreg)

percentage_of_5_for_4_fld = nb_5s_for_4_fld/(nb_5s_for_4_fld+nb_3s_for_4_fld)
percentage_of_5_for_6_fld = nb_5s_for_6_fld/(nb_5s_for_6_fld+nb_3s_for_6_fld)
percentage_of_5_for_7_fld = nb_5s_for_7_fld/(nb_5s_for_7_fld+nb_3s_for_7_fld)


print(avg_score_logreg, avg_score_fld)

genes = ["BCL2L10", "ZAR1L", "C3orf56", "BTG4", "TUBB8", "SH2D1B", "C9orf116", "TMEM132B", "CA4", "FAM19A4"]
print("coeff:")
for i in range(0,10):
    print(genes[i], " & ", round(coeff_logreg[0][i],4), " & ", round(coeff_fld[0][i],4), "\\\\")

print(coeff_logreg,coeff_fld)


print("percentage_of_5_for_4_logreg: ",percentage_of_5_for_4_logreg)
print("percentage_of_5_for_6_logreg: ",percentage_of_5_for_6_logreg)
print("percentage_of_5_for_7_logreg: ",percentage_of_5_for_7_logreg)

print("percentage_of_5_for_4_fld: ",percentage_of_5_for_4_fld)
print("percentage_of_5_for_6_fld: ",percentage_of_5_for_6_fld)
print("percentage_of_5_for_7_fld: ",percentage_of_5_for_7_fld)
