import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
cancer_data = load_breast_cancer(as_frame=True)
cancer_df = cancer_data.data
cancer_df['target'] = cancer_data.target

X = cancer_df.drop(["target"], axis=1)
y = cancer_df["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=417)

# class LinearSVC:
#     def __init__(self,penalty,loss, random_state):
#             self.penalty = penalty
#             self.loss = loss
#             self.random_state = random_state
        

model = LinearSVC(penalty="l2",loss="hinge",C=10,random_state=417)
model.fit(X_train,y_train)
