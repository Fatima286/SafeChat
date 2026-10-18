import pandas as pd

df=pd.read_parquet("Berkley.parquet")
                                        
def After_Collapsing(dataframe):
    label_cols=dataframe.columns[3:14].tolist()
    print(label_cols)

    #vertical collapse:
    

    agg_dict={'text':'first'}
    for col in label_cols:
        agg_dict[col]='mean'

    collapsed_df=df.groupby('comment_id').agg(agg_dict).reset_index(drop=False) 

    return collapsed_df


#print(f"Before collapse: {len(df)} rows, {df['comment_id'].nunique()} unique comments")
coll=After_Collapsing(df)
coll.to_csv("collapsed_data.csv",index=False)
