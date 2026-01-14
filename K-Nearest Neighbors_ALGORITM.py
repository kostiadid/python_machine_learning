import pandas as pd
banking_df = pd.read_csv("subscription_prediction.csv")

# print(banking_df["y"])


for i in  banking_df["y"]:
    if i == "yes":
        i = 1
    else:
        i = 0
train_df = banking_df.sample(frac=0.85,random_state=417)
test_df =  banking_df.drop(train_df.index)  
print(banking_df.value_counts())
