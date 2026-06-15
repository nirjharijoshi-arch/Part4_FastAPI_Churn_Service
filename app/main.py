from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import joblib
import pandas as pd

app = FastAPI()

model = joblib.load("model.pkl")
encoders = joblib.load("encoders.pkl")


class CustomerFeatures(BaseModel):
    city_tier: str
    age_group: str
    acquisition_channel: str
    loyalty_tier: str
    preferred_category: str
    marketing_consent: int

    recency_days: float
    frequency_180d: float
    monetary_180d: float

    return_rate_180d: float
    avg_discount_pct_180d: float
    avg_rating_180d: float

    category_diversity_180d: float
    ticket_count_90d: float
    negative_ticket_rate_90d: float
    avg_resolution_hours_90d: float

    days_since_signup: float

    sessions_30d: float
    product_views_30d: float
    cart_adds_30d: float
    wishlist_adds_30d: float
    abandoned_carts_30d: float

    email_opens_30d: float
    campaign_clicks_30d: float
    last_visit_days_ago: float


def get_risk_level(prob):
    if prob >= 0.75:
        return "high"
    elif prob >= 0.40:
        return "medium"
    return "low"


def get_explanation(prob):
    if prob >= 0.75:
        return "Low engagement and support history indicate elevated churn risk."
    elif prob >= 0.40:
        return "Customer shows moderate churn indicators."
    return "Customer currently shows low churn risk."


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(customer: CustomerFeatures):

    df = pd.DataFrame([customer.dict()])

    for col, encoder in encoders.items():
        df[col] = encoder.transform(df[col].astype(str))

    probability = float(
        model.predict_proba(df)[0][1]
    )

    prediction = int(probability >= 0.5)

    return {
        "churn_probability": round(probability, 3),
        "predicted_class": prediction,
        "risk_level": get_risk_level(probability),
        "risk_explanation": get_explanation(probability)
    }


@app.post("/batch_predict")
def batch_predict(customers: List[CustomerFeatures]):

    df = pd.DataFrame(
        [c.dict() for c in customers]
    )

    for col, encoder in encoders.items():
        df[col] = encoder.transform(df[col].astype(str))

    probabilities = model.predict_proba(df)[:, 1]

    results = []

    for p in probabilities:
        results.append(
            {
                "churn_probability": round(float(p), 3),
                "predicted_class": int(p >= 0.5),
                "risk_level": get_risk_level(p)
            }
        )

    return results
