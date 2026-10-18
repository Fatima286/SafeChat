from sklearn.model_selection import train_test_split
from preprocess import Main
import pandas as pd
df=pd.read_csv('collapsed_data.csv')




train_df,temp_df=train_test_split(df,test_size=0.3,random_state=42)
print(len(train_df),len(temp_df))

val_df,test_df=train_test_split(temp_df,test_size=0.5,random_state=42)
print(len(val_df),len(test_df))

for name, split_df in [("train",train_df),("val",val_df),("test",test_df)]:
    processed=Main(split_df)
    processed.to_csv(name+"_processed.csv",index=False)
    print(f"Saved {name}_processed.csv — {len(processed)} rows")
