# Monitoring Plan

## Data Drift Monitoring

Track changes in:

- recency_days
- frequency_180d
- monetary_180d
- return_rate_180d
- ticket_count_90d

Compare current feature distributions against training data monthly.

## Prediction Distribution

Monitor:

- percentage predicted as churn
- average churn probability
- changes in risk-level proportions

Unexpected shifts may indicate drift.

## Business Outcomes

Track:

- actual churn rate
- retention campaign success rate
- customer lifetime value
- churn reduction after interventions

## API Monitoring

Track:

- response latency
- API failures
- validation errors
- uptime

## Retraining Triggers

Retrain when:

- recall drops below 75%
- ROC-AUC drops below 0.80
- significant feature drift is detected
- major business changes occur

## Responsible Use

Predictions should support retention decisions and not replace human judgment.

The model should not be used for discrimination, denial of service, or automated customer exclusion.
