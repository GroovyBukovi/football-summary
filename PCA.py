import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# === CONFIGURATION ===
INPUT_CSV = "fused_dataset.csv"  # Replace with actual file
TARGET_COLUMN = "Class"
DROP_COLUMNS = ["timestamp", "timecode", "transcript", "segment_index"]
PLOT_FILE = "pca_plot.png"

# === STEP 1: Load and clean data ===
df = pd.read_csv(INPUT_CSV)
df = df.drop(columns=[col for col in DROP_COLUMNS if col in df.columns])

X = df.drop(columns=[TARGET_COLUMN])
y = df[TARGET_COLUMN].apply(lambda x: 1 if x == 'Yes' else 0)

X = X.dropna(axis=1, how='all')

# === STEP 2: Impute and standardize ===
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

# === STEP 3: Perform PCA ===
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# === STEP 4: Plot the first 2 principal components ===
plt.figure(figsize=(10, 8))
plt.scatter(X_pca[y == 0, 0], X_pca[y == 0, 1], label='No', alpha=0.6, c='blue')
plt.scatter(X_pca[y == 1, 0], X_pca[y == 1, 1], label='Yes', alpha=0.6, c='red')
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("PCA Projection (2D)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(PLOT_FILE, dpi=300)
plt.close()

print(f"✅ PCA plot saved as: {PLOT_FILE}")
