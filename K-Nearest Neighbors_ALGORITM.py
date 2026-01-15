import pandas as pd
banking_df = pd.read_csv("subscription_prediction.csv")

# print(banking_df["y"])


banking_df["y"] = banking_df["y"].replace({"yes": 1, "no": 0})        
        
train_df = banking_df.sample(frac=0.85,random_state=417)
test_df =  banking_df.drop(train_df.index)  
print(train_df["y"].value_counts(normalize=True))
print(test_df["y"].value_counts(normalize=True))


X_train = train_df.drop(columns=["y"])
y_train = train_df["y"]

X_test = test_df.drop(columns=["y"])
y_test = test_df["y"]

def knn(feature,single_test_input,k):
