import sklearn as sk
from sklearn.model_selection import train_test_split
import pandas as pd
import math
import datetime

from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import GradientBoostingRegressor

from sklearn.preprocessing import StandardScaler


data = pd.read_csv("data/processed_training_v3.csv")  # All data

ol = pd.read_csv('data/on_line_data_train.csv')  # Else
lo = pd.read_csv("data/late_ontime.csv")  # Heu1 == 30
eo = pd.read_csv('data/early_ontime.csv')    # 0 < heu < 15


train_target = ['LongestDays', 'PostingDays', 'Huristic1', 'CompanyCode', 'DocumentType', 'PaymentDocumentNo',
                'InvoiceDate']
test_target = ["ActualDays"]

X_all = data[['BusinessTransaction', 'CompanyCode', 'DocumentType',
       'EntryTime', 'InvoiceAmount', 'InvoiceDate', 'InvoiceDesc',
       'InvoiceItemDesc', 'PaymentDocumentNo', 'Period', 'PO_FLag',
       'PO_PurchasingDocumentNumber', 'ReferenceDocumentNo', 'TransactionCode',
       'TransactionCodeDesc', 'UserName', 'VendorName',
        'EntryDays', 'PostingDays', 'LongestDays', 'Huristic1']]

x_ol = ol[['BusinessTransaction', 'CompanyCode', 'DocumentType',
       'EntryTime', 'InvoiceAmount', 'InvoiceDate', 'InvoiceDesc',
       'InvoiceItemDesc', 'PaymentDocumentNo', 'Period', 'PO_FLag',
       'PO_PurchasingDocumentNumber', 'ReferenceDocumentNo', 'TransactionCode',
       'TransactionCodeDesc', 'UserName', 'VendorName',
        'EntryDays', 'PostingDays', 'LongestDays', 'Huristic1']]
x_lo = lo[['BusinessTransaction', 'CompanyCode', 'DocumentType',
       'EntryTime', 'InvoiceAmount', 'InvoiceDate', 'InvoiceDesc',
       'InvoiceItemDesc', 'PaymentDocumentNo', 'Period', 'PO_FLag',
       'PO_PurchasingDocumentNumber', 'ReferenceDocumentNo', 'TransactionCode',
       'TransactionCodeDesc', 'UserName', 'VendorName',
        'EntryDays', 'PostingDays', 'LongestDays', 'Huristic1']]
x_eo = eo[['BusinessTransaction', 'CompanyCode', 'DocumentType',
       'EntryTime', 'InvoiceAmount', 'InvoiceDate', 'InvoiceDesc',
       'InvoiceItemDesc', 'PaymentDocumentNo', 'Period', 'PO_FLag',
       'PO_PurchasingDocumentNumber', 'ReferenceDocumentNo', 'TransactionCode',
       'TransactionCodeDesc', 'UserName', 'VendorName',
        'EntryDays', 'PostingDays', 'LongestDays', 'Huristic1']]
y_ol = ol['ActualDays']
y_lo = lo['ActualDays']
y_eo = eo['ActualDays']

# X = data[train_target]
y = data[test_target]

X_train, X_test, y_train, y_test = train_test_split(X_all, y, test_size=0.4, random_state=0)

clf = ExtraTreesRegressor(n_estimators=100, criterion='mae')
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
score = mean_absolute_error(y_pred, y_test)
print(score)
y_pred_1 = [math.floor(x) if x > 0 else math.ceil(x) for x in y_pred]
score2 = mean_absolute_error(y_pred_1, y_test)

print(score2)

# On the line point region regression
clf_ol = ExtraTreesRegressor(n_estimators=100, criterion='mse', bootstrap=False)
clf_ol.fit(y_ol, y_ol)

# Later or on time data region regression
clf_lo = ExtraTreesRegressor(n_estimators=10, criterion='mse', bootstrap=False)
clf_lo.fit(y_lo, y_lo)

# Earlier or on time data region regression
clf_eo = ExtraTreesRegressor(n_estimators=10, criterion='mse', bootstrap=False)
clf_eo.fit(y_eo, y_eo)

data_test = pd.read_csv("data/InvoicePayment-final_evaluation.csv")

data_test.InvoiceDate = data_test.InvoiceDate.apply(lambda x: datetime.strptime(x, "%Y-%m-%d"))

# Predict final result here
x_f_test = pd.read_csv('data/processed_testing.csv')
x_f_test = x_f_test[train_target]

y_f_ol = clf_ol.predict(x_f_test)
y_f_lo = clf_lo.predict(x_f_test)
y_f_eo = clf_eo.predict(x_f_test)

y_final = [y_f_eo.loc[x] if x_f_test['Hurestic1'].loc[x] == 30 else (y_f_lo.loc[x] if
                                                                     (x_f_test['Hurestic1'].loc[x] < 15)
                                                                     & (x_f_test['Hurestic1'].loc[x]>0) else y_f_ol.loc[x]) for x in range(len(x_f_test)) ]

y_final = [math.floor(x) if x > 0 else math.ceil(x) for x in y_final]

# predict_y = clf.predict(x_f_test)
# print(y_final[:100])
data_test['PaymentDate'] = y_final
data_test.to_csv("tmp.csv")
