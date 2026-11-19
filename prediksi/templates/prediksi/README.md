# Prediksi Harga Kos Madiun

Aplikasi web untuk memprediksi estimasi harga kos pada wilayah Madiun menggunakan Machine Learning dengan algoritma Linear Regression dan framework Django.

# Teknologi yang digunakan
- Python
- Django
- Pandas
- Scikit-learn
- Joblib
- Chart.js

# Fitur
- Prediksi harga kos berdasarkan jarak, rating, dan fasilitas AC.
- Slider untuk mengatur jarak dan rating.
- API prediksi harga.
- API analisis rata-rata harga kos.
- Grafik perbandingan harga kos dengan AC dan tanpa AC.

# Dataset 

Dataset berisi 30 data kos dengan atribut:
Nama kos, Jarak (km), Harga (ribu rupiah per bulan), fasilitas AC, dan rating.

# Algoritma
Model menggunakan Linear Regression dengan fitur jarak, rating, dan fasilitas.

## Cara Menjalankan
1. Aktifkan virtual environment:
2. Menjalankan server.
3. Kemudian membuka browser Django.

## API
- `/api/predict/` — memprediksi harga kos.
- `/api/chart-insight/` — menyediakan data dan penjelasan untuk grafik.

Aplikasi ini masih merupakan prototipe pembelajaran. Data-data yang digunakan masih 30 data sehingga hasil prediksi masih belum akurat.