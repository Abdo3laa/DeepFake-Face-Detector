import pandas as pd
import joblib
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.utils import shuffle

print("Running This File:", __file__)

CSV_INPUT = "face_embeddings_fine_tuning.csv"
OLD_MODEL_PATH = "deepfake_xgb_model_clib_final.pkl"
NEW_MODEL_OUTPUT = "adapter.pkl"

df = pd.read_csv(CSV_INPUT)
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
X = df.drop(columns=['label']).values
y = df['label'].values

old_model, le = joblib.load(OLD_MODEL_PATH)
y_encoded = le.transform(y)

X, y_encoded = shuffle(X, y_encoded, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

params = old_model.get_xgb_params()
adapter_model = XGBClassifier(**params)
adapter_model.fit(X_train, y_train, xgb_model=old_model.get_booster())

y_pred = adapter_model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=le.classes_))
print(confusion_matrix(y_test, y_pred))

joblib.dump((adapter_model, le), NEW_MODEL_OUTPUT)
print("Adapter model saved:", NEW_MODEL_OUTPUT)
