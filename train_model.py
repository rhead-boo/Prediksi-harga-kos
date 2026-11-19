
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1. Membaca dataset
df = pd.read_csv("kos_madiun.csv")

# 2. Mengubah fasilitas AC menjadi angka
df["ac_num"] = df["ac"].str.lower().map({
    "ya": 1,
    "tidak": 0
})

# 3. Menentukan fitur (X) dan target (y)
X = df[["jarak_km", "rating", "ac_num"]]
y = df["harga"]

# 4. Membagi data training dan testing
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# 5. Membuat dan melatih model
model = LinearRegression()
model.fit(X_train, y_train)

# 6. Melakukan prediksi pada data testing
y_pred = model.predict(X_test)

# 7. Menghitung evaluasi model
mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred) ** 0.5
r2 = r2_score(y_test, y_pred)

print("=== HASIL TRAINING MODEL KOS ===")
print("Jumlah data:", len(df))
print("MAE:", round(mae, 2))
print("RMSE:", round(rmse, 2))
print("R2 Score:", round(r2, 3))

# 8. Menyimpan model
joblib.dump(model, "model_kos.joblib")

print("\nModel berhasil disimpan sebagai model_kos.joblib")
