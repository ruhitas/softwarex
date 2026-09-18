# GitHub'a yükleme paketi

`repository` klasörünün içeriğini GitHub kod deponuzun köküne yükleyin. README, kaynak kod, bağımlılıklar, veri bölme manifesti ve kayıtlı tahminler bu klasördedir.

`release_assets` klasöründeki üç `.pt` model dosyası GitHub'ın yükleme boyutu sınırını aştığı için GitHub Release yerine Google Drive üzerinden, tek bir zip arşivi halinde paylaşılmıştır:

https://drive.google.com/file/d/15mHnY6mqvAymrpn6LghK3quhytknkBJX/view?usp=drive_link

Kullanıcılar bu bağlantıdan zip dosyasını indirip açar; içinden çıkan üç `.pt` dosyasını (`base_swin.pt`, `ce_tail.pt`, `ce_gated.pt`) deponun `weights` klasörüne doğrudan yerleştirir. README'de bu Google Drive bağlantısı belirtilmiştir; makalenin Code availability bölümünde de hem gerçek GitHub depo adresini hem bu Drive bağlantısını kullanın.

Henüz GitHub'a yükleme veya yayınlama yapılmadı. Ham veri, kişisel bilgisayar yolları, makale Word dosyaları, üçüncü taraf PDF'ler ve EPPO haritası pakete alınmadı. Veri üreticilerine atıf ve veri kaynağı README'de bulunur.

Yazılım için henüz bir yeniden kullanım lisansı seçilmedi. README bu durumu açıkça belirtir; sizin adınıza MIT/Apache gibi bir lisans verilmedi.

Kodların açıklamaları ve README İngilizcedir. Kayıtlı sonuçlar ağırlıksız doğrulanabilir. Yeniden eğitim için ham veri, CUDA ve sağlanan başlangıç checkpoint'i gerekir. Önceki başlangıç modelinin hiperparametre araması bu paketin kapsamı dışındadır.
