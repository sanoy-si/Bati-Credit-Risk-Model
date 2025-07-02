from pydantic import BaseModel, Field

class PredictionInput(BaseModel):
    total_monetary_value: float = Field(..., example=800.0)
    avg_monetary_value: float = Field(..., example=160.0)
    std_monetary_value: float = Field(..., example=96.17)
    transaction_count: int = Field(..., example=5)
    total_fraud_transactions: int = Field(..., example=1)
    has_committed_fraud: int = Field(..., example=1)
    num_unique_products: int = Field(..., example=3)
    avg_transaction_hour: float = Field(..., example=10.5)
    most_freq_product_category: str = Field(..., example='catA')
    most_freq_provider: str = Field(..., example='Prov1')
    most_freq_channel: str = Field(..., example='Ch1')

class PredictionOutput(BaseModel):
    risk_probability: float = Field(..., example=0.75)