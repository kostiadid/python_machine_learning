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
print(train_df.value_counts(normalize=True))
print(test_df.value_counts(normalize=True))


X_train = train_df.drop(columns=["y"])
y_train = train_df["y"]

X_test = test_df.drop(columns=["y"])
y_test = test_df["y"]



# The submitted code has a few issues that need to be addressed.

# The target variable y in the banking_df DataFrame is not being updated with the new values (1 for "yes" and 0 for "no"). The for loop that attempts to do this is not modifying the DataFrame because it's only iterating over the values, not updating the DataFrame.
# The value_counts(normalize=True) function is being called on the entire DataFrame (train_df and test_df) instead of the specific column y. This will calculate the distribution of all columns, not just the target variable.
# The rest of the code seems to be correct, but it's relying on the previous steps being correct.
# To fix these issues, you should focus on updating the y column in banking_df and then calculate the value counts for the y column specifically.
