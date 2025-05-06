import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier
import joblib

def clean_data(data):
    # Gender map
    if 'Gender' in data:
        data['Gender'] = data['Gender'].map({'Female': 0, 'Male': 1}).fillna(0)

    # Binary columns
    binary_columns = [
        'Smoking', 'Alcohol', 'Stress', 'Fever',
        'Cough', 'Fatigue', 'Difficulty Breathing'
    ]
    for col in binary_columns:
        if col in data:
            data[col] = data[col].map({'Yes': 1, 'No': 0}).fillna(0)
        else:
            data[col] = 0

    # Blood Pressure
    mapping_bp = {'High': 1, 'Normal': 0, 'Low': -1}
    if 'Blood Pressure' in data:
        data['Blood Pressure'] = data['Blood Pressure'].map(mapping_bp).fillna(0)

    # Cholesterol
    mapping_chol = {'High': 1, 'Normal': 0, 'Low': -1}
    if 'Cholesterol Level' in data:
        data['Cholesterol Level'] = data['Cholesterol Level'].map(mapping_chol).fillna(0)

    data.fillna(0, inplace=True)
    return data


def augment_data(data, num_copies=150):
    augmented_data = pd.concat([data] * num_copies, ignore_index=True)

    # Gaussian noise on age
    augmented_data['Age'] = augmented_data['Age'] + np.random.normal(0, 1, size=len(augmented_data))
    augmented_data['Age'] = np.clip(augmented_data['Age'], 0, 100)

    # Conditional re‑sampling for lifestyle features
    for i, row in augmented_data.iterrows():
        age = row['Age']
        if age < 25:
            augmented_data.at[i, 'Smoking']  = np.random.choice([1, 0], p=[0.4, 0.6])
            augmented_data.at[i, 'Alcohol']  = np.random.choice([1, 0], p=[0.3, 0.7])
            augmented_data.at[i, 'Stress']   = np.random.choice([1, 0], p=[0.4, 0.6])
        elif 25 <= age < 50:
            augmented_data.at[i, 'Smoking']  = np.random.choice([1, 0], p=[0.7, 0.3])
            augmented_data.at[i, 'Alcohol']  = np.random.choice([1, 0], p=[0.5, 0.5])
            augmented_data.at[i, 'Stress']   = np.random.choice([1, 0], p=[0.6, 0.4])
        else:
            augmented_data.at[i, 'Smoking']  = np.random.choice([1, 0], p=[0.6, 0.4])
            augmented_data.at[i, 'Alcohol']  = np.random.choice([1, 0], p=[0.4, 0.6])
            augmented_data.at[i, 'Stress']   = np.random.choice([1, 0], p=[0.8, 0.2])
    return augmented_data


def main():
    file_path = "C:/Users/user/Downloads/Disease_symptom_and_patient_profile_dataset.csv"
    raw_data = pd.read_csv(file_path)

    # 1) Clean
    data = clean_data(raw_data)

    # 2) Augment
    data_aug = augment_data(data, num_copies=150)
    print("Veri boyutu:", len(data), "->", len(data_aug))

    # 3) Check disease column
    if 'Disease' not in data_aug.columns:
        raise ValueError("'Disease' adında bir sütun bulunamadı!")

    # 4) Encode disease labels
    le = LabelEncoder()
    data_aug['Disease'] = data_aug['Disease'].astype(str)
    data_aug['DiseaseEncoded'] = le.fit_transform(data_aug['Disease'])

    # 5) Feature list
    features = [
        'Age', 'Gender', 'Blood Pressure', 'Cholesterol Level',
        'Smoking', 'Alcohol', 'Stress',
        'Fever', 'Cough', 'Fatigue', 'Difficulty Breathing'
    ]
    target_col = 'DiseaseEncoded'

    # 6) Scale age
    scaler = MinMaxScaler()
    data_aug['Age'] = scaler.fit_transform(data_aug[['Age']])

    # 7) Split X, y
    X = data_aug[features]
    y = data_aug[target_col]

    # 8) Multi‑class XGBoost
    xgb_model = XGBClassifier(
        n_estimators=200,
        max_depth=10,
        learning_rate=0.05,
        random_state=42,
        objective='multi:softmax',
        num_class=len(le.classes_)
    )

    # 9) 5‑Fold
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    for fold, (train_idx, val_idx) in enumerate(kf.split(X), 1):
        print(f"\n=== KFold {fold} ===")
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        xgb_model.fit(X_train, y_train)
        y_pred = xgb_model.predict(X_val)

        acc = accuracy_score(y_val, y_pred)
        f1  = f1_score(y_val, y_pred, average='weighted')
        print(f"Accuracy: {acc:.3f}")
        print(f"F1‑score (weighted): {f1:.3f}")
        print(classification_report(y_val, y_pred, target_names=le.classes_))

    # 10) Kaydet: model + LabelEncoder
    joblib.dump(xgb_model, 'xgb_model_multiclass.pkl')
    joblib.dump(le,          'label_encoder.pkl')     # <-- Flask kodu için gerekli

    # 11) SHAP (son fold üzerinden)
    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(X_val)
    if isinstance(shap_values, list) and len(shap_values) > 1:
        shap.summary_plot(shap_values[0], X_val, show=False)
        plt.savefig("shap_summary_multiclass_class0.png", dpi=300, bbox_inches='tight')
    else:
        shap.summary_plot(shap_values, X_val, show=False)
        plt.savefig("shap_summary_multiclass.png", dpi=300, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    main()
