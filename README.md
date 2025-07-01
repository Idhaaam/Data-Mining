Nama : Affrizaa Primaliandra 231011403609
       Afrizal David Maulama 231011403455
       Idham Faturrohman Syufi 231011402914
       Muhammad Adam FIrmansyah 231011403026

       Link Youtube : https://youtu.be/u0ygAVX_fWM?si=b8Ry0roxnuLANNhO


       LAPORAN ANALISIS KODE ENSEMBLE LEARNING DENGAN BAGGINGCLASSIFIER

Pendahuluan
Pada laporan ini akan dibahas mengenai analisis fungsi-fungsi Python pada program klasifikasi dataset wine menggunakan metode ensemble learning, khususnya dengan algoritma BaggingClassifier dari pustaka scikit-learn. Program ini bertujuan untuk mempelajari bagaimana penggabungan beberapa model dasar dapat meningkatkan performa klasifikasi, serta bagaimana evaluasi hasil klasifikasi dilakukan.

Import Library
Bagian awal kode digunakan untuk mengimpor semua modul dan pustaka yang dibutuhkan. Fungsi-fungsinya adalah sebagai berikut:

from sklearn.datasets import load_wine
Berfungsi untuk memuat dataset wine bawaan dari pustaka scikit-learn. Dataset ini digunakan untuk tugas klasifikasi dan memiliki tiga kelas target.

from sklearn.model_selection import train_test_split
Digunakan untuk membagi dataset menjadi data latih dan data uji secara acak.

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, BaggingClassifier
Mengimpor empat jenis algoritma ensemble learning. Pada program ini hanya BaggingClassifier yang digunakan, sementara yang lainnya hanya disiapkan sebagai alternatif jika ingin dibandingkan.

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
Ketiga fungsi ini digunakan untuk mengevaluasi hasil prediksi dari model, baik berupa nilai akurasi, laporan klasifikasi lengkap, maupun confusion matrix.

Load Dataset
Bagian ini mengambil data dan memisahkan fitur serta target.

data = load_wine()
Memanggil fungsi load_wine() dan menyimpannya dalam variabel data.

X = data.data
Mengambil data fitur dan menyimpannya dalam variabel X.

y = data.target
Mengambil target (kelas dari masing-masing sampel) dan menyimpannya dalam variabel y.

Split Data
Baris kode berikut digunakan untuk membagi dataset menjadi data pelatihan dan pengujian.

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3)
Fungsi ini membagi 70% data menjadi data latih dan 30% menjadi data uji secara acak. Tujuannya agar model bisa dilatih menggunakan sebagian data, lalu diuji pada data yang belum pernah dilihat oleh model.

Membuat dan Melatih Model
Bagian ini berisi proses pembuatan model dan pelatihannya.

model = BaggingClassifier()
Membuat objek model BaggingClassifier. Ini adalah algoritma ensemble yang membangun banyak model dasar (default-nya decision tree) dari subset acak data latih, lalu menggabungkan hasilnya dengan voting.

model.fit(X_train, y_train)
Melatih model menggunakan data latih. Selama proses ini, beberapa subset data dibuat, dan masing-masing subset digunakan untuk melatih satu decision tree. Hasilnya adalah kumpulan model yang siap memprediksi data uji.

Prediksi dan Evaluasi
Tahap ini melakukan prediksi terhadap data uji dan menilai seberapa baik performa model.

y_predict = model.predict(X_test)
Model yang sudah dilatih digunakan untuk memprediksi kelas dari data uji X_test. Hasilnya disimpan di variabel y_predict.

print(confusion_matrix(y_test, y_predict))
Menampilkan confusion matrix, yaitu tabel 3x3 yang menunjukkan jumlah prediksi benar dan salah untuk setiap kelas.

print(classification_report(y_test, y_predict))
Menampilkan laporan evaluasi klasifikasi yang mencakup precision, recall, f1-score, dan jumlah data (support) untuk setiap kelas. Juga menampilkan akurasi keseluruhan model terhadap data uji.

Penutup
Dari hasil program ini, dapat dilihat bahwa model BaggingClassifier mampu menghasilkan akurasi yang tinggi pada dataset wine. Evaluasi menggunakan confusion matrix dan classification report menunjukkan bahwa model mampu mengklasifikasikan data dengan cukup akurat di semua kelas. Metode ensemble seperti ini cocok digunakan pada data yang rentan terhadap overfitting jika hanya menggunakan satu model dasar.
