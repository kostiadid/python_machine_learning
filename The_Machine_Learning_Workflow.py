import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
cancer_data = load_breast_cancer(as_frame=True)
cancer_df = cancer_data.data
cancer_df['target'] = cancer_data.target

X = cancer_df.drop('target', axis=1)

y = cancer_df['target']

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.15,random_state=417)




# The output of train_test_split() is a list containing 4 elements:
# The training set features.
# The test set features.
# The training set labels.
# The test set labels.