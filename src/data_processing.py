import pandas as pd
import numpy as np
import os
import joblib
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

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
    print("--- Starting Data Processing ---")

    # Define paths
    raw_data_path = os.path.join('data', 'raw', 'transactions.csv')
    processed_data_path = os.path.join('data', 'processed', 'customer_features.csv')
    pipeline_path = os.path.join('src', 'processing_pipeline.joblib')
    
    # Load raw data
    try:
        raw_df = pd.read_csv(raw_data_path)
        print(f"Raw data loaded successfully from {raw_data_path}")
    except FileNotFoundError:
        print(f"ERROR: Raw data file not found at {raw_data_path}. Please check the path.")
        return

    # Create customer-level features
    customer_features_df = _create_customer_level_features(raw_df)

    # Create and fit the feature engineering pipeline
    pipeline = create_feature_engineering_pipeline(customer_features_df)
    
    # Fit the pipeline on the customer features data
    processed_data = pipeline.fit_transform(customer_features_df)
    
    # Get feature names after one-hot encoding
    cat_feature_names = pipeline.named_transformers_['cat']['onehot'].get_feature_names_out(customer_features_df.select_dtypes(include=['object', 'category']).columns)
    all_feature_names = customer_features_df.select_dtypes(include=np.number).columns.tolist() + cat_feature_names.tolist()
    
    processed_df = pd.DataFrame(processed_data, index=customer_features_df.index, columns=all_feature_names)

    # 3. Save the processed data and the pipeline
    processed_df.to_csv(processed_data_path)
    print(f"Processed data saved to {processed_data_path}")

    joblib.dump(pipeline, pipeline_path)
    print(f"Fitted processing pipeline saved to {pipeline_path}")
    
    print("--- Data Processing Complete ---")


if __name__ == '__main__':
    main()