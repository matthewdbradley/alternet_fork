import pandas as pd
from sklearn.preprocessing import StandardScaler


def create_hybrid_data(transcript_data, gene_data, tf_list, biomart_column='Transcript stable ID'):
    hybrid_data = pd.concat([transcript_data.loc[:, transcript_data.columns.isin(tf_list['Transcript stable ID'])], gene_data], axis =1)
    return hybrid_data
    




def standardize_dataframe(df):
    """
    Standardizes a DataFrame column-wise using sklearn's StandardScaler.
    
    Parameters:
        df (pd.DataFrame): The input DataFrame.
        
    Returns:
        pd.DataFrame: The standardized DataFrame.
    """
    scaler = StandardScaler()

    scaled_array = scaler.fit_transform(df)
    
    standardized_df = pd.DataFrame(
        scaled_array, 
        columns=df.columns,
        index=df.index
    )
    return standardized_df