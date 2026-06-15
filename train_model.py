import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
import joblib
import zipfile

zip_path = "d2c churn data package-20260615T085556Z-3-001.zip"

z = zipfile.ZipFile(zip_path)

df = pd.read_csv(
    z.open(
        "d2c churn data package/rfm_modeling_snapshot.csv"
    )
)

target = "churn_next_60d"

train_df = df[df["split"] == "train"]

drop_cols = [
    "customer_id",
    "snapshot_date",
    "split",
    target
]

X = train_df.drop(columns=drop_cols)
y = train_df[target]

encoders = {}

for col in X.select_dtypes(include="object").columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    encoders[col] = le

model = LogisticRegression(
    max_iter=2000,
    random_state=42
)

model.fit(X, y)

joblib.dump(model, "model.pkl")
joblib.dump(encoders, "encoders.pkl")

print("Saved model.pkl and encoders.pkl")
