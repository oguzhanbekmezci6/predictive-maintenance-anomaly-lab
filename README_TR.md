# Predictive Maintenance Anomaly Detection Lab

Sentetik endüstriyel telemetri verisi üzerinde, güvenlik-kritik bakış açısıyla
anomali tespiti yapan uçtan uca makine öğrenmesi projesidir.

## İçerik

- Sıcaklık, titreşim, basınç, RPM, motor akımı, akustik seviye,
  debi, yük ve bakım yaşı içeren sentetik telemetri
- Dengesiz anomali sınıfı
- Exploratory Data Analysis
- XGBoost baseline ve class-weighted XGBoost
- PyTorch Neural Network baseline ve `pos_weight` kullanılan sürüm
- Validation verisiyle güvenlik odaklı decision threshold seçimi
- Precision, Recall, F1, ROC-AUC, Average Precision
- False Negative Rate
- Örnek başına inference süresi
- Covariate drift simülasyonu
- Population Stability Index
- Otomatik safety-critical model değerlendirmesi
- Model, CSV, PNG, JSON ve Markdown rapor çıktıları

## En kolay çalıştırma

Windows'ta proje klasöründeki:

```text
START_PROJECT.bat
```

dosyasına çift tıkla. Script:

1. `.venv` oluşturur.
2. Kütüphaneleri kurar.
3. `run_project.py` dosyasını çalıştırır.
4. Sonuçları `outputs/` klasörüne kaydeder.

İlk kurulumda PyTorch nedeniyle indirme biraz uzun sürebilir.

## PyCharm ile çalıştırma

1. ZIP'i klasöre çıkar.
2. PyCharm → **Open** → proje klasörünü seç.
3. Python interpreter olarak Python 3.11 veya 3.12 seç.
4. Terminalde:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install -r requirements.txt
python run_project.py
```

PyCharm interpreter olarak daha sonra şunu seçebilirsin:

```text
.venv\Scripts\python.exe
```

## Hızlı deneme

```powershell
python run_project.py --samples 5000 --epochs 15
```

Varsayılan çalışma:

```powershell
python run_project.py
```

## Üretilen dosyalar

```text
data/
├── telemetry.csv
└── telemetry_drifted_test.csv

outputs/
├── REPORT.md
├── summary.json
├── eda/
│   ├── class_distribution.png
│   ├── correlation_matrix.png
│   ├── vibration_by_class.png
│   └── CSV analizleri
├── metrics/
│   ├── model_comparison_clean.csv
│   ├── model_comparison_drift.csv
│   ├── feature_drift_summary.csv
│   └── confusion matrix dosyaları
└── models/
    ├── xgboost_baseline.json
    ├── xgboost_weighted.json
    ├── neural_net_baseline.pt
    ├── neural_net_weighted.pt
    └── scaler dosyaları
```

## Model karşılaştırması

Dört deney yapılır:

1. XGBoost Baseline
2. XGBoost Weighted
3. Neural Net Baseline
4. Neural Net Weighted

Dengesiz sınıf için:

- XGBoost: `scale_pos_weight`
- Neural Net: `BCEWithLogitsLoss(pos_weight=...)`

kullanılır.

Threshold, test verisine bakılarak seçilmez. Validation verisinde,
minimum precision sınırı altında false-negative oranını azaltacak şekilde
seçilir.

## Drift simülasyonu

Test verisinde çalışma koşulları değiştirilir:

- Ortam ve makine sıcaklığı yükselir.
- Yük, titreşim, RPM ve akım artar.
- Basınç ve debi düşer.
- Bakım yaşı yükselir.

Model performansı aynı threshold ile tekrar ölçülür. Ayrıca özellik bazında
PSI hesaplanır.

## Safety-critical yorum

Bu projede model seçimi sadece accuracy ile yapılmaz. Öncelik:

1. En kötü senaryodaki recall
2. False Negative Rate
3. Precision
4. Inference süresi

Sentetik veriyle elde edilen bir model gerçek bir güvenlik sisteminin tek
koruma katmanı olarak kullanılmamalıdır. Gerçek arıza kayıtları, zaman dışı
doğrulama, alarm prosedürleri, sensör kalite kontrolleri ve mühendislik
yedekliliği gerekir.

## Testler

```powershell
python -m pytest -q
```

## Yeni örnek tahmini

Ana proje çalıştıktan sonra:

```powershell
python predict_example.py
```
