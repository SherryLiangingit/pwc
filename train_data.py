import sklearn as sk
from sklearn.model_selection import train_test_split
import pandas as pd

from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import AdaBoostClassifier

from sklearn.preprocessing import StandardScaler


data = pd.read_csv("data/processed_training_v3.csv")  # All data

ol = pd.read_csv('data/on_line_data_train.csv')  # Else
lo = pd.read_csv("data/late_ontime.csv")  # Heu1 == 30
eo = pd.read_csv('data/early_ontime.csv')    # 0 < heu < 15

train_target = ['LongestDays', 'PostingDays', 'Huristic1', 'CompanyCode', 'DocumentType', 'PaymentDocumentNo',
                'InvoiceDate']
test_target = ["ActualDays"]

X = data[train_target]
y = data[test_target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=0)

clf = ExtraTreesRegressor(criterion='mae')  # (n_estimators=20, criterion='mae', bootstrap=False)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
score = mean_absolute_error(y_pred, y_test)
print(score)


# Predict final result here
x_f_test = pd.read_csv('data/processed_testing.csv')
x_f_test = x_f_test[train_target]

predict_y = clf.predict(x_f_test)
