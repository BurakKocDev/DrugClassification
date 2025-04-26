# DrugClassification
DrugClassification
1. Amaç
Bu proje, bir ilaç görüntü veri setini kullanarak, görüntü sınıflandırması gerçekleştirmek amacıyla yapılmıştır. Transfer learning yöntemi ile MobileNetV2 önceden eğitilmiş modeli kullanılmıştır.

2. Kullanılan Kütüphaneler
numpy, pandas, matplotlib: Veri işleme ve görselleştirme

tensorflow, keras: Derin öğrenme modeli oluşturma ve eğitme

sklearn: Modelin performansını değerlendirme

pathlib, warnings: Dosya yönetimi ve uyarı mesajlarının kontrolü

3. Veri Yükleme
Görüntüler belirtilen dizinden (*.jpg, *.png, .jpeg) uzantılı dosyalar olarak yüklendi.

Her görüntünün etiketi (label), dosya yolundaki klasör ismine göre belirlendi.

Bir DataFrame oluşturularak tüm görüntü yolları ve etiketleri saklandı.

4. Veri Görselleştirme
Veri setinden rastgele 25 görüntü seçilerek 5x5 grid şeklinde görselleştirildi.

Her görselin üstüne kendi etiketi başlık olarak yazıldı.

5. Eğitim ve Test Ayrımı
Veri seti %80 eğitim ve %20 test olacak şekilde bölündü.

Eğitim verisi ayrıca %20 oranında validation (doğrulama) verisine ayrıldı.

6. Veri Ön İşleme
ImageDataGenerator kullanılarak:

Eğitim verisi için veri artırımı (preprocessing_function) uygulandı.

Validation ve test verisi için sadece MobileNetV2 ön işleme fonksiyonu kullanıldı.

Görüntüler 224x224 boyutuna yeniden boyutlandırıldı.

Batch size olarak 64 seçildi.

7. Model Yapısı
Base model: MobileNetV2 (ImageNet verisiyle önceden eğitilmiş, üst katmanları kaldırılmış)

trainable=False: Temel ağırlıklar eğitime kapatıldı.

Yeni katmanlar:

256 nöronlu, ReLU aktivasyonlu Dense katman + %20 Dropout

256 nöronlu, ReLU aktivasyonlu bir Dense katman daha + %20 Dropout

Sonuç katmanı: Softmax aktivasyonlu ve sınıf sayısına göre çıkış

Kayıp fonksiyonu: categorical_crossentropy

Optimizasyon: Adam optimizer, öğrenme oranı 1e-4

8. Eğitim Süreci
Callback'ler:

ModelCheckpoint: En iyi validation accuracy değerine ulaşan ağırlıkları .weights.h5 dosyasına kaydetti.

EarlyStopping: Validation loss 5 epoch boyunca iyileşmezse eğitimi durdurdu.

Eğitim 10 epoch boyunca gerçekleştirildi.

9. Model Değerlendirme
Test seti üzerinde modelin kayıp (loss) ve doğruluk (accuracy) değerleri hesaplandı.

Eğitim ve doğrulama setleri için accuracy ve loss grafiklerle görselleştirildi.

10. Sonuçlar
Test verisi üzerinde modelin başarı oranı ve kaybı raporlandı.

Modelin tahmin ettiği etiketler ile gerçek etiketler karşılaştırılarak ayrıntılı bir sınıflandırma raporu (classification_report) oluşturuldu.

Precision, Recall, F1-Score değerleri her sınıf için hesaplandı.