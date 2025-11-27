import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from xgboost import XGBClassifier
import joblib
import numpy as np

CSV_INPUT = "face_embeddings_clib_final.csv"
MODEL_OUTPUT = "deepfake_xgb_model_clib_final.pkl"

df = pd.read_csv(CSV_INPUT)
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
X = df.drop(columns=['label']).values
y = df['label'].values

le = LabelEncoder()
y_encoded = le.fit_transform(y)

fake_count = (y == "fake").sum()
real_count = (y == "real").sum()
scale_pos_weight = fake_count / real_count

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

model = XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=6, subsample=0.9, colsample_bytree=0.9, eval_metric='logloss', scale_pos_weight=scale_pos_weight)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=le.classes_))
print(confusion_matrix(y_test, y_pred))

joblib.dump((model, le), MODEL_OUTPUT)
print("Model saved:", MODEL_OUTPUT)
