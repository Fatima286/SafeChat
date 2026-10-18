import pandas as pd 
#import numpy as np 

df=pd.read_parquet("train.parquet")


"""row_sum=df[label_cols].sum(axis=1)# axis 1 represents columns moving horizontally and collapsing the information across the horizontal axis 
safe_rows=df[row_sum==0] #series in panadas with zero sum of toxicity
safety_percentage=(len(safe_rows)/len(df))*100
print("The percentage of safe rows is ",safety_percentage)
print("The percentage of toxic rows is ",100-safety_percentage)"""

"""cols_sum=df[label_cols].sum(axis=0)#axis 0 represents rows moving vertically and collapsing the information vertically 
print(cols_sum)
row_sum=df[label_cols].sum(axis=1)
multiLabel_rows=df[row_sum>1]
print("The number of rows with multiple labels is ",len(multiLabel_rows))"""

# how many rows per comment_id
annotator_counts = df['comment_id'].value_counts()

#print(f"Total rows: {len(df)}")
#print(f"Unique comment_ids: {df['comment_id'].nunique()}")
#print(f"Comments rated by >1 annotator: {(annotator_counts > 1).sum()}")
#print(f"Max annotators on a single comment: {annotator_counts.max()}")
print(df.columns[3:13].tolist())
# see the distribution
#print(annotator_counts.value_counts().sort_index())
