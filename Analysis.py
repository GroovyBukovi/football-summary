import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# === CONFIGURATION ===
INPUT_CSV = "balanced_dataset.csv"           # Replace with your file
TARGET_COLUMN = "Class"
DROP_COLUMNS = ["timestamp", "timecode", "transcript", "segment_index"]
CORR_IMAGE_FILE = "correlation_matrix.png"

# === STEP 1: Load and clean data ===
df = pd.read_csv(INPUT_CSV)
df = df.drop(columns=[col for col in DROP_COLUMNS if col in df.columns])

X = df.drop(columns=[TARGET_COLUMN])
y = df[TARGET_COLUMN].apply(lambda x: 1 if x == 'Yes' else 0)

X = X.dropna(axis=1, how='all')

# === STEP 2: Preprocess (Impute + Standardize) ===
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)
X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)

# === STEP 3: Feature Importance via Random Forest ===
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_scaled_df, y)

importance_df = pd.DataFrame({
    "feature": X.columns,
    "importance": rf.feature_importances_
}).sort_values(by="importance", ascending=False)

top_features = importance_df['feature'].head(24).tolist()

# === STEP 4: Correlation Matrix ===
corr_df = X_scaled_df[top_features].copy()
corr_df[TARGET_COLUMN] = y
correlation_matrix = corr_df.corr()

# === STEP 5: Plot and Save Correlation Heatmap ===
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", square=True, linewidths=0.5)
plt.title("Correlation Matrix (Top 10 Features + Class)")
plt.tight_layout()
plt.savefig(CORR_IMAGE_FILE, dpi=300)
plt.close()

print(f"✅ Correlation matrix plotted and saved as: {CORR_IMAGE_FILE}")
