# Proje Akışı

## 1. Sentetik telemetri üretimi

`src/data_generation.py` şu değişkenleri üretir:

- operating load
- ambient temperature
- machine temperature
- vibration
- pressure
- RPM
- motor current
- acoustic level
- flow rate
- hours since maintenance

Arızalar; aşırı ısınma, rulman arızası, basınç arızası, aşırı devir ve
birleşik arıza tipleriyle sensörlerde ilişkili bozulmalar oluşturur.

## 2. Zamana göre veri bölme

Rastgele split yerine kronolojik split kullanılır:

- İlk yüzde 70: train
- Sonraki yüzde 15: validation
- Son yüzde 15: test

Bu yaklaşım üretim ortamına daha yakın bir değerlendirme sağlar.

## 3. EDA

`src/eda.py`:

- sınıf dağılımını,
- özellik özetlerini,
- sınıflara göre ortalamaları,
- korelasyon matrisini,
- titreşim dağılımını

CSV ve PNG olarak kaydeder.

## 4. Modeller

Dört deney yapılır:

- XGBoost Baseline
- XGBoost Weighted
- Neural Net Baseline
- Neural Net Weighted

## 5. Class imbalance

XGBoost için:

```python
scale_pos_weight = negative_count / positive_count
```

Neural Network için:

```python
BCEWithLogitsLoss(pos_weight=...)
```

kullanılır.

## 6. Threshold seçimi

Önce tüm modeller 0.50 threshold ile ayrıca değerlendirilir.

Ana güvenlik değerlendirmesinde threshold yalnızca validation setinde seçilir.
Amaç minimum precision sınırını koruyarak recall değerini yükseltmek ve
false negative sayısını azaltmaktır. Test seti threshold seçiminde kullanılmaz.

## 7. Drift

Test sensörlerinin dağılımı kontrollü şekilde değiştirilir. Aynı model ve aynı
threshold ile performans yeniden ölçülür. PSI tablosu hangi özelliklerin daha
fazla kaydığını gösterir.

## 8. Safety-critical karar

Accuracy tek başına kullanılmaz. Öncelik sırası:

1. Worst-case recall
2. Worst-case false-negative rate
3. Precision
4. Tek gözlem inference süresi

Son karar `outputs/REPORT.md` dosyasına yazılır.
