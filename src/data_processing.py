import pandas as pd
import numpy as np
import os
import joblib
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


def _create_customer_level_features(df: pd.DataFrame) -> pd.DataFrame:
    # Convert to datetime and extract time-based features
    df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'])
    df['hour_of_day'] = df['TransactionStartTime'].dt.hour
    df['day_of_week'] = df['TransactionStartTime'].dt.dayofweek # Monday=0, Sunday=6

    customer_df = df.groupby('CustomerId').agg(
        total_monetary_value=('Value', 'sum'),
        avg_monetary_value=('Value', 'mean'),
        std_monetary_value=('Value', 'std'),
        transaction_count=('Value', 'count'),
        
        total_fraud_transactions=('FraudResult', 'sum'),
        has_committed_fraud=('FraudResult', 'max'),
        
        num_unique_products=('ProductId', 'nunique'),
        
        most_freq_product_category=('ProductCategory', lambda x: x.mode()[0] if not x.mode().empty else "Unknown"),
        most_freq_provider=('ProviderId', lambda x: x.mode()[0] if not x.mode().empty else "Unknown"),
        most_freq_channel=('ChannelId', lambda x: x.mode()[0] if not x.mode().empty else "Unknown"),
        
        avg_transaction_hour=('hour_of_day', 'mean'),
        avg_night_transaction_hour=('hour_of_day', lambda x: np.mean(x[(x >= 22) | (x <= 5)]))
    )
    
    customer_df['std_monetary_value'] = customer_df['std_monetary_value'].fillna(0)
    
    customer_df['avg_night_transaction_hour'] = customer_df['avg_night_transaction_hour'].fillna(-1)

    print("Successfully created customer-level features.")
    return customer_df

def _create_rfm_and_target_variable(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates RFM metrics, clusters customers into 3 segments, identifies the
    high-risk cluster, and creates the binary target variable 'is_high_risk'.

    Args:
        df (pd.DataFrame): The raw transaction DataFrame.

    Returns:
        pd.DataFrame: A DataFrame with CustomerId as the index and the
                      'is_high_risk' target column.
    """
    print("--- Starting RFM and Target Variable Engineering ---")
    
    # Ensure datetime format
    df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'])

    # --- RFM Calculation ---
    # Define a snapshot date for calculating recency. This is typically one day
    # after the last transaction in the dataset.
    snapshot_date = df['TransactionStartTime'].max() + pd.Timedelta(days=1)

    # Calculate RFM values for each customer
    rfm_df = df.groupby('CustomerId').agg({
        'TransactionStartTime': lambda date: (snapshot_date - date.max()).days,
        'TransactionId': 'count',
        'Value': 'sum'
    })

    # Rename columns for clarity
    rfm_df.rename(columns={'TransactionStartTime': 'Recency',
                           'TransactionId': 'Frequency',
                           'Value': 'MonetaryValue'}, inplace=True)

    # --- RFM Preprocessing and Clustering ---
    # The RFM features are skewed. We'll apply a log transform and scale them.
    rfm_log_transformed = rfm_df.copy()
    rfm_log_transformed[['Recency', 'Frequency', 'MonetaryValue']] = rfm_log_transformed[['Recency', 'Frequency', 'MonetaryValue']].apply(np.log1p)

    # Scale the features
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm_log_transformed)
    rfm_scaled = pd.DataFrame(rfm_scaled, index=rfm_df.index, columns=rfm_df.columns)

    # Use K-Means to cluster customers into 3 segments
    kmeans = KMeans(n_clusters=3, random_state=42, n_init='auto')
    rfm_df['Cluster'] = kmeans.fit_predict(rfm_scaled)

    # --- Identify High-Risk Cluster ---
    # We analyze the centroids of the clusters to find the "worst" customers.
    # High-risk customers typically have high recency (haven't purchased in a while),
    # low frequency, and low monetary value.
    cluster_centroids = rfm_df.groupby('Cluster')[['Recency', 'Frequency', 'MonetaryValue']].mean()
    
    # The cluster with the highest Recency is our highest-risk segment
    high_risk_cluster_id = cluster_centroids['Recency'].idxmax()
    
    print("Cluster Centroids (Mean RFM values for each cluster):")
    print(cluster_centroids)
    print(f"Identified High-Risk Cluster: {high_risk_cluster_id}")

    # --- Create Target Variable ---
    # Assign 1 to customers in the high-risk cluster, 0 to others.
    rfm_df['is_high_risk'] = (rfm_df['Cluster'] == high_risk_cluster_id).astype(int)
    
    print(f"Percentage of high-risk customers: {rfm_df['is_high_risk'].mean() * 100:.2f}%")
    
    return rfm_df[['is_high_risk']]

def create_feature_engineering_pipeline(customer_df: pd.DataFrame):
    # Identify numerical and categorical features from the aggregated dataframe
    numerical_features = customer_df.select_dtypes(include=np.number).columns.tolist()
    categorical_features = customer_df.select_dtypes(include=['object', 'category']).columns.tolist()

    print(f"Identified {len(numerical_features)} numerical features.")
    print(f"Identified {len(categorical_features)} categorical features.")

    # Create preprocessing pipelines for both numerical and categorical data
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    # Create a preprocessor object using ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='passthrough' # Keep other columns if any (should be none)
    )

    print("Scikit-learn preprocessing pipeline created.")
    return preprocessor


def main():
    """
    Main function to run the full data processing workflow.
    Loads raw data, creates features, creates the target variable, builds and fits
    the processing pipeline, and saves the final model-ready data and pipeline.
    """
    print("--- Starting Full Data Processing and Feature Engineering ---")

    # Define paths
    raw_data_path = os.path.join('data', 'raw', 'transactions.csv')
    processed_data_path = os.path.join('data', 'processed', 'model_input_data.csv')
    pipeline_path = os.path.join('src', 'processing_pipeline.joblib')
    
    os.makedirs(os.path.join('data', 'processed'), exist_ok=True)

    # Load raw data
    try:
        raw_df = pd.read_csv(raw_data_path)
        print(f"Raw data loaded successfully from {raw_data_path}")
    except FileNotFoundError:
        print(f"ERROR: Raw data file not found at {raw_data_path}.")
        return

    # --- Task 3: Create customer-level predictive features ---
    customer_features_df = _create_customer_level_features(raw_df.copy())

    # --- Task 4: Create RFM-based target variable ---
    target_df = _create_rfm_and_target_variable(raw_df.copy())

    # --- Merge features and target ---
    # We now have a complete dataset with predictors and the response variable
    model_ready_df = pd.merge(customer_features_df, target_df, left_index=True, right_index=True)

    # --- Separate features (X) from target (y) for pipeline fitting ---
    X = model_ready_df.drop('is_high_risk', axis=1)
    y = model_ready_df['is_high_risk']

    # --- Create and fit the feature engineering pipeline on features ONLY ---
    pipeline = create_feature_engineering_pipeline(X)
    X_processed = pipeline.fit_transform(X)
    
    # Get feature names after transformation for the final DataFrame
    cat_feature_names = pipeline.named_transformers_['cat']['onehot'].get_feature_names_out(X.select_dtypes(include=['object', 'category']).columns)
    all_feature_names = X.select_dtypes(include=np.number).columns.tolist() + cat_feature_names.tolist()
    
    # Create the final processed DataFrame for features
    processed_features_df = pd.DataFrame(X_processed, index=X.index, columns=all_feature_names)
    
    # --- Combine processed features with the target variable for saving ---
    final_processed_data = pd.concat([processed_features_df, y], axis=1)

    # --- Save the final model-ready data and the pipeline ---
    final_processed_data.to_csv(processed_data_path)
    print(f"Final model-ready data saved to {processed_data_path}")

    joblib.dump(pipeline, pipeline_path)
    print(f"Fitted processing pipeline saved to {pipeline_path}")
    
    print("--- Data Processing and Target Engineering Complete ---")


if __name__ == '__main__':
    main()