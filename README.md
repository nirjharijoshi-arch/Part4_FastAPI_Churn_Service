# FastAPI Churn Scoring Service

## Project Overview

This project provides a FastAPI service for customer churn prediction.

The API loads a trained machine learning model and returns churn probability, churn prediction, and risk level.

---

## Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Run API

```bash
uvicorn app.main:app --reload
```

---

## Endpoints

### GET /health

Returns API health status.

Example Response:

```json
{
  "status": "ok"
}
```

---

### POST /predict

Predict churn risk for one customer.

Example Response:

```json
{
  "churn_probability": 0.72,
  "predicted_class": 1,
  "risk_level": "high"
}
```

---

### POST /batch_predict

Predict churn risk for multiple customers.

Returns a prediction for each record.

---

## Running Tests

```bash
pytest
```

---

## Model Notes

The model was trained using the provided churn dataset and uses only information available before the prediction date to avoid data leakage.
