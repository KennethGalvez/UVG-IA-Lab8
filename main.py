import pandas as pd
from sklearn.model_selection import train_test_split

dataset = pd.read_csv('bank_transactions.csv')

X = dataset.iloc[:, :-1].values
X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)
print (X_train, X_test)