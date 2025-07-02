import pytest
import pandas as pd
import numpy as np
from src.data_processing import _create_customer_level_features, _create_rfm_and_target_variable

# Create a fixture to provide a sample raw DataFrame for tests.
# This fixture will be automatically passed to any test function that needs it.
@pytest.fixture
def sample_raw_data():
    """Provides a small, controlled DataFrame for testing."""
    data = {
        'TransactionId': [f't{i}' for i in range(10)],
        'TransactionStartTime': pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-05', '2023-01-10', '2023-01-11',
                                              '2023-01-01', '2023-01-03', '2023-01-04', '2023-01-12', '2023-01-13']),
        'CustomerId': ['C1', 'C1', 'C1', 'C1', 'C1', 'C2', 'C2', 'C2', 'C3', 'C3'],
        'Value': [100, 200, 50, 300, 150, 1000, 500, 250, 800, 1200],
        'FraudResult': [0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
        'ProductId': ['P1', 'P2', 'P1', 'P3', 'P2', 'P4', 'P5', 'P4', 'P6', 'P6'],
        'ProductCategory': ['catA', 'catB', 'catA', 'catC', 'catB', 'catD', 'catE', 'catD', 'catF', 'catF'],
        'ProviderId': ['Prov1', 'Prov1', 'Prov2', 'Prov1', 'Prov2', 'Prov3', 'Prov3', 'Prov3', 'Prov1', 'Prov1'],
        'ChannelId': ['Ch1', 'Ch1', 'Ch1', 'Ch2', 'Ch2', 'Ch3', 'Ch3', 'Ch3', 'Ch1', 'Ch1']
    }
    return pd.DataFrame(data)

# Test 1: Verify the customer-level feature creation
def test_create_customer_level_features(sample_raw_data):
    """
    Tests the _create_customer_level_features function.
    Checks for correct aggregation, shape, and column names.
    """
    customer_df = _create_customer_level_features(sample_raw_data)

    # Assert correct shape: 3 unique customers in sample data
    assert customer_df.shape[0] == 3
    
    # Assert expected columns are created
    expected_cols = ['total_monetary_value', 'transaction_count', 'has_committed_fraud']
    assert all(col in customer_df.columns for col in expected_cols)
    
    # Assert a specific aggregation is correct
    # Customer C1 has 5 transactions
    assert customer_df.loc['C1', 'transaction_count'] == 5
    # Customer C1 has a total value of 100+200+50+300+150 = 800
    assert customer_df.loc['C1', 'total_monetary_value'] == 800
    # Customer C3 had one fraudulent transaction, so has_committed_fraud should be 1
    assert customer_df.loc['C3', 'has_committed_fraud'] == 1


# Test 2: Verify the RFM and target variable creation
def test_create_rfm_and_target_variable(sample_raw_data):
    """
    Tests the _create_rfm_and_target_variable function.
    Checks that the output has the right shape and the target column is valid.
    """
    target_df = _create_rfm_and_target_variable(sample_raw_data)

    # Assert correct shape: 3 unique customers
    assert target_df.shape[0] == 3
    assert target_df.shape[1] == 1 # Only the target column should be returned
    
    # Assert the target column exists
    assert 'is_high_risk' in target_df.columns
    
    # Assert the target column is binary (contains only 0s and 1s)
    assert set(target_df['is_high_risk'].unique()) <= {0, 1}