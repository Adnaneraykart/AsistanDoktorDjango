import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score, confusion_matrix, roc_curve
from sklearn.linear_model import LogisticRegression  # LightGBM yerine LogisticRegression import edildi
import os
import shap  # SHAP import edilmiştir
import matplotlib.pyplot as plt  # matplotlib import edilmiştir
import seaborn as sns  # Seaborn import edilmiştir

# ----------------------------------------------------------
# 1) Veri Temizleme ve Dönüştürme
# ----------------------------------------------------------

def clean_data(data):
    # Outcome Variable: 'Positive' -> 1, 'Negative' -> 0
    if 'Outcome Variable' in data:
        data['Outcome Variable'] = data['Outcome Variable'].map({'Positive': 1, 'Negative': 0})

    # Gender: 'Female' -> 0, 'Male' -> 1
    if 'Gender' in data:
        data['Gender'] = data['Gender'].map({'Female': 0, 'Male': 1}).fillna(0)

    # Binary sütunlar: 'Yes' -> 1, 'No' -> 0
    binary_columns = ['Smoking', 'Alcohol', 'Stress', 'Fever', 'Cough', 'Fatigue', 'Difficulty Breathing']
    for col in binary_columns:
        if col in data:
            data[col] = data[col].map({'Yes': 1, 'No': 0}).fillna(0)
        else:
            data[col] = 0  # Eksik sütunlar için varsayılan değer

    # Blood Pressure ve Cholesterol Level kategorik değerleri dönüştürme
    mapping_bp = {'High': 1, 'Normal': 0, 'Low': -1}
    mapping_chol = {'High': 1, 'Normal': 0, 'Low': -1}
    if 'Blood Pressure' in data:
        data['Blood Pressure'] = data['Blood Pressure'].map(mapping_bp).fillna(0)
    if 'Cholesterol Level' in data:
        data['Cholesterol Level'] = data['Cholesterol Level'].map(mapping_chol).fillna(0)

    # Eksik değerleri doldurma
    data.fillna(0, inplace=True)
    return data


# ----------------------------------------------------------
# 2) Veri Çoğaltma ve Olasılıklı Atamalar
# ----------------------------------------------------------

def augment_data(data, num_copies=150):
    augmented_data = pd.concat([data] * num_copies, ignore_index=True)

    # Gürültü ekleme
    augmented_data['Age'] = augmented_data['Age'] + np.random.normal(0, 1, size=len(augmented_data))
    augmented_data['Age'] = np.clip(augmented_data['Age'], 0, 100)

    # Sigara, Alkol ve Stres olasılıklarına bağlı atama
    for i, row in augmented_data.iterrows():
        age = row['Age']
        if age < 25:
            augmented_data.at[i, 'Smoking'] = np.random.choice([1, 0], p=[0.4, 0.6])
            augmented_data.at[i, 'Alcohol'] = np.random.choice([1, 0], p=[0.3, 0.7])
            augmented_data.at[i, 'Stress'] = np.random.choice([1, 0], p=[0.4, 0.6])
        elif 25 <= age < 50:
            augmented_data.at[i, 'Smoking'] = np.random.choice([1, 0], p=[0.7, 0.3])
            augmented_data.at[i, 'Alcohol'] = np.random.choice([1, 0], p=[0.5, 0.5])
            augmented_data.at[i, 'Stress'] = np.random.choice([1, 0], p=[0.6, 0.4])
        else:
            augmented_data.at[i, 'Smoking'] = np.random.choice([1, 0], p=[0.6, 0.4])
            augmented_data.at[i, 'Alcohol'] = np.random.choice([1, 0], p=[0.4, 0.6])
            augmented_data.at[i, 'Stress'] = np.random.choice([1, 0], p=[0.8, 0.2])

    return augmented_data


# ----------------------------------------------------------
# 3) Ana Akış (main)
# ----------------------------------------------------------

def main():
    file_path = "C:/Users/user/Downloads/Disease_symptom_and_patient_profile_dataset.csv"
    raw_data = pd.read_csv(file_path)

    # Veri temizleme
    cleaned_data = clean_data(raw_data)

    # Veri çoğaltma
    augmented_data = augment_data(cleaned_data, num_copies=150)
    print(f"Veri seti {len(cleaned_data)} örnekten {len(augmented_data)} örneğe artırıldı.")

    # Yeni CSV dosyasına kaydet
    augmented_data_file = "augmented_data.csv"
    augmented_data.to_csv(augmented_data_file, index=False)
    print(f"Artırılmış veri seti '{augmented_data_file}' dosyasına kaydedildi.")

    # Özellikler ve hedef kolon
    features = [
        'Age', 'Gender', 'Blood Pressure', 'Cholesterol Level',
        'Smoking', 'Alcohol', 'Stress',
        'Fever', 'Cough', 'Fatigue', 'Difficulty Breathing'
    ]
    target_col = 'Outcome Variable'

    # Yalnızca yaşa min-max normalizasyonu
    scaler = MinMaxScaler()
    augmented_data['Age'] = scaler.fit_transform(augmented_data[['Age']])

    # Model
    log_reg_model = LogisticRegression(
        solver='liblinear',  # Küçük veri setleri için uygun
        penalty='l2',
        C=1.0,
        random_state=42
    )

    # Özellikler ve hedef verilerini ayırma
    X = augmented_data[features]
    y = augmented_data[target_col].astype(float)

    # KFold çapraz doğrulama
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    # KFold ile eğitim ve değerlendirme
    fold_number = 1  # Fold numarasını takip etmek için
    shap_values_list = []  # SHAP değerlerini depolamak için
    X_val_list = []       # Validation verilerini depolamak için

    for train_index, val_index in kf.split(X):
        print(f"\nFold {fold_number}")
        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]

        # Modeli eğitme
        log_reg_model.fit(X_train, y_train)

        # Tahmin ve değerlendirme
        y_pred = log_reg_model.predict(X_val)
        acc = accuracy_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred)
        auc = roc_auc_score(y_val, y_pred)

        # Confusion Matrix (Karmaşık Matrisi)
        cm = confusion_matrix(y_val, y_pred)
        if cm.shape == (2, 2):
            TN, FP, FN, TP = cm.ravel()
        else:
            # Eğer bazı fold'larda sınıflar dengesizse, conf matrisi farklı boyutta olabilir
            TN = FP = FN = TP = 0
            if '1' in cm and '0' in cm:
                TP = cm[1,1]
                TN = cm[0,0]
                FP = cm[0,1]
                FN = cm[1,0]

        # Sensitivity (Duyarlılık) ve Specificity (Özgüllük) hesaplama
        sensitivity = TP / (TP + FN) if (TP + FN) != 0 else 0
        specificity = TN / (TN + FP) if (TN + FP) != 0 else 0
        precision = precision_score(y_val, y_pred)
        recall = recall_score(y_val, y_pred)

        print("\n=== Logistic Regression Model Sonuçları ===")
        print("Accuracy:", acc)
        print("F1 Score:", f1)
        print("ROC AUC:", auc)
        print("Sensitivity:", sensitivity)
        print("Specificity:", specificity)
        print("Precision:", precision)
        print("Recall:", recall)

        # ROC Eğrisini çizme
        # Logistic Regression için skorlar tahmin edilerek ROC eğrisi çizilebilir
        y_prob = log_reg_model.predict_proba(X_val)[:, 1]
        fpr, tpr, _ = roc_curve(y_val, y_prob)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='blue', label='ROC Eğrisi')
        plt.plot([0, 1], [0, 1], color='red', linestyle='--')
        plt.title(f'Lojistik Regresyon')
        plt.xlabel('Pozitif olarak tahmin edilen yanlış değer')
        plt.ylabel('Pozitif olarak tahmin edilen doğru değer')
        plt.legend(loc='lower right')
        plt.grid(True)
        plt.savefig(f'roc_curve_fold_{fold_number}.png')
        plt.close()

        # SHAP değerlerini toplamak için
        X_val_list.append(X_val)
        # Masker'ı oluşturma
        masker = shap.maskers.Independent(X_train)
        explainer = shap.LinearExplainer(log_reg_model, masker)
        shap_vals = explainer.shap_values(X_val)
        shap_values_list.append(shap_vals)

        fold_number += 1

    # Tüm fold'ların SHAP değerlerini birleştirme
    shap_values_combined = np.concatenate(shap_values_list, axis=0)
    X_val_combined = pd.concat(X_val_list, ignore_index=True)

    # Test verileri ve tahminleri birlikte yazdırmak
    # Bütün fold'ların validation verilerini ve tahminlerini birleştirmek daha anlamlı olabilir
    # Ancak mevcut yapı gereği sadece son fold'un test verileri kullanılıyor
    test_results = pd.DataFrame({
        'Age': X_val['Age'],
        'Gender': X_val['Gender'],
        'Blood Pressure': X_val['Blood Pressure'],
        'Cholesterol Level': X_val['Cholesterol Level'],
        'Smoking': X_val['Smoking'],
        'Alcohol': X_val['Alcohol'],
        'Stress': X_val['Stress'],
        'Fever': X_val['Fever'],
        'Cough': X_val['Cough'],
        'Fatigue': X_val['Fatigue'],
        'Difficulty Breathing': X_val['Difficulty Breathing'],
        'True Label': y_val,
        'Predicted Label': y_pred
    })

    print("\nTest Verileri ve Tahminler:")
    print(test_results)

    # Tabloyu görselleştirme ve resim olarak kaydetme
    sns.set(style="whitegrid")

    fig, ax = plt.subplots(figsize=(12, 8))  # Boyutu ayarlayın
    ax.axis('tight')
    ax.axis('off')

    # Tabloyu oluşturun
    table = ax.table(cellText=test_results.values,
                    colLabels=test_results.columns,
                    cellLoc='center',  # Hücrelerin metin hizalaması
                    loc='center',  # Tabloyu merkeze yerleştirme
                    colColours=["#f2f2f2"] * len(test_results.columns),  # Kolon başlıkları için arka plan rengi
                    cellColours=[["#e6f7ff"] * len(test_results.columns)] * len(test_results))  # Hücrelerin arka plan rengi

    # Hücrelerin metin özelliklerini değiştirmek
    for (i, j), cell in table.get_celld().items():
        if i == 0:
            cell.set_fontsize(12)
            cell.set_text_props(weight='bold', color='black')
        else:
            cell.set_fontsize(10)
            cell.set_text_props(weight='normal', color='black')
        cell.set_edgecolor('black')  # Kenarlık rengi

    # Tabloyu bir dosyaya kaydedin
    plt.savefig("test_results_table_stylish.png", format="png", bbox_inches='tight', dpi=300)
    plt.close()  # Görseli kapat

    # SHAP analizi
    # Tüm SHAP değerlerini kullanarak summary plot oluşturma
    if shap_values_combined.size > 0:
        plt.figure()
        shap.summary_plot(shap_values_combined, X_val_combined, show=False)
        plt.savefig("shap_summary_plotLR.png", format="png", bbox_inches="tight")
        plt.close()
        print("SHAP summary plot başarıyla kaydedildi.")
    else:
        print("SHAP summary plot oluşturulamadı: SHAP değerleri boş.")

# ----------------------------------------------------------
if __name__ == "__main__":
    main()
