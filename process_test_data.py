
# coding: utf-8

import pandas as pd
import numpy as np
from datetime import datetime, date
import time

src_file = "data/InvoicePayment-final_evaluation.csv"
des_file = "data/processed_testing_2.csv"

data = pd.read_csv(src_file)


del data['DocumentTypeDesc'], data['Year'], data["PwC_RowID"], data["CompanyName"], data["LocalCurrency"], \
    data['DocumentNo'], data['PurchasingDocumentDate']

data.BusinessTransaction = data.BusinessTransaction.apply(lambda x: int(x[-4:]))
data.CompanyCode = data.CompanyCode.apply(lambda x: int(x[-3:]))
data.DocumentType = data.DocumentType.apply(lambda x: int(x[-2:]))
data.ReferenceDocumentNo = data.ReferenceDocumentNo.apply(lambda x: int(x[-8:]))
data.TransactionCodeDesc = data.TransactionCodeDesc.apply(lambda x:int(x[-5:]))
data.UserName = data.UserName.apply(lambda x: int(x[-4:]))
data.TransactionCode = data.TransactionCode.apply(lambda x: int(x[-4:]))
data.InvoiceDesc = data.InvoiceDesc.apply(lambda x: int(x[-8:]))
data.InvoiceItemDesc = data.InvoiceItemDesc.apply(lambda x: int(x[-8:]))
data.PO_PurchasingDocumentNumber = data.PO_PurchasingDocumentNumber.apply(lambda x: int(x[-8:]))
data.VendorName = data.VendorName.apply(lambda x: int(x[-5:]))

# for idx in range(len(data.PaymentDueDate)):
#     if type(data.PaymentDueDate.loc[idx]) is float:
#         data.PaymentDueDate.loc[idx] = data.PaymentDate.loc[idx]

for idx in range(len(data.InvoiceDate)):
    if type(data.InvoiceDate.loc[idx]) is float:
        data.InvoiceDate.loc[idx] = data.EntryDate.loc[idx]

data.PaymentDueDate = data.PaymentDueDate.apply(lambda x: datetime.strptime(x, "%Y-%m-%d"))
data.EntryDate = data.EntryDate.apply(lambda x: datetime.strptime(x, "%Y-%m-%d"))
# data.PaymentDate = data.PaymentDate.apply(lambda x: datetime.strptime(x, "%Y-%m-%d"))
data.PostingDate = data.PostingDate.apply(lambda x: datetime.strptime(x, "%Y-%m-%d"))
data.InvoiceDate = data.InvoiceDate.apply(lambda x: datetime.strptime(x, "%Y-%m-%d"))

data['LongestDays'] = data.PaymentDueDate - data.InvoiceDate
data.LongestDays = data.LongestDays.apply(lambda x: int(str(x).split(" ")[0]))
data.LongestDays = data.LongestDays.apply(lambda x: x + 365 if x < 0 else x)
data.LongestDays = data.LongestDays.apply(lambda x: x - 365 if x >365 else x)

# data['ActualDays'] = data.PaymentDate - data.InvoiceDate
# data.ActualDays = data.ActualDays.apply(lambda x: int(str(x).split(" ")[0]))
# data.ActualDays = data.ActualDays.apply(lambda x: x + 365 if x < 0 else x)
# data.ActualDays = data.ActualDays.apply(lambda x: x - 365 if x > 365 else x)
#
# data['ActualDays'] = data.LongestDays - data.ActualDays

data['EntryDays'] = data.EntryDate - data.InvoiceDate
data.EntryDays = data.EntryDays.apply(lambda x: int(str(x).split(" ")[0]))
data.EntryDays = data.EntryDays.apply(lambda x: x + 365 if x < 0 else x)
data.EntryDays = data.EntryDays.apply(lambda x: x - 365 if x >365 else x)

data['PostingDays'] = data.PostingDate - data.InvoiceDate
data.PostingDays = data.PostingDays.apply(lambda x: int(str(x).split(" ")[0]))
data.PostingDays = data.PostingDays.apply(lambda x: x + 365 if x < 0 else x)
data.PostingDays = data.PostingDays.apply(lambda x: x - 365 if x > 365 else x)


data['EntryTime'] = data['EntryTime'].apply(lambda x: int(x.split(":")[0]))
data['InvoiceDate'] = data["InvoiceDate"].apply(lambda x: (int(str(x)[3])-2) * 10 + int(str(x)[5:7])/12.0 * 10)

data.PaymentDocumentNo = data.PaymentDocumentNo.apply(lambda x: int(x[-8:]))

data.PO_FLag = data.PO_FLag.apply(lambda x: 0 if x == "N" else 1)

data['Huristic1'] = data['LongestDays'] - data['PostingDays']  # - 5?  Train a model to fit a good parameter

# data['EarlyLate'] = data['ActualDays'].apply(lambda x: 'Late' if x > 0 else ('OnTime' if x == 0 else "Late"))

data['EntryDays'] = data['EntryDays'] - data['PostingDays']

target_keys = ['BusinessTransaction', 'CompanyCode', 'DocumentType', 'EntryTime', 'InvoiceAmount', 'InvoiceDate', 'InvoiceDesc',
       'InvoiceItemDesc', 'PaymentDocumentNo', 'Period', 'PO_FLag', 'PO_PurchasingDocumentNumber', 'ReferenceDocumentNo', 'TransactionCode',
       'TransactionCodeDesc', 'UserName', 'VendorName', 'VendorCountry', 'EntryDays', 'PostingDays', 'LongestDays',
        'Huristic1']


target_data = data[target_keys]

target_data.to_csv(des_file)

