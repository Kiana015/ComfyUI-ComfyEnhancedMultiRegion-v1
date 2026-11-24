# Referensi parameter — ComfyEnhancedMultiRegion

Dokumen ini menjelaskan **setiap parameter** di repo ini, persis seperti yang dilakukan kodenya (bukan perkiraan).
Semua isi di sini berlaku untuk branch `claude/anima-character-prompts`, yang berisi semua fitur.
Lihat bagian [Ketersediaan per branch](#ketersediaan-per-branch) kalau kamu memakai `main`.

Istilah yang dipakai:
- **Region**: satu prompt yang berlaku di satu wilayah gambar.
- **Token**: satu "kotak" gambar yang diproses model. Di Anima, 1 token = 16×16 piksel. Contohnya 832×1216 px = 52×76 token.
- **Pass positif/negatif**: dalam CFG, model dijalankan sekali dengan prompt positif dan sekali dengan prompt negatif.
- **Cross-attention**: bagian model tempat gambar "membaca" teks prompt.
- **Self-attention**: bagian model tempat token gambar saling melihat. Di Anima (DiT) bagian ini global, jadi setiap token melihat seluruh gambar.

---

## 1. Cara kerja dasar

1. Node membagi gambar menjadi beberapa wilayah (mask).
2. Di cross-attention, setiap token gambar hanya membaca prompt milik wilayahnya sendiri. Inilah yang memisahkan karakter.
3. Pass negatif tetap memakai satu prompt negatif untuk seluruh gambar.
4. Opsi tambahan (blend, isolasi, base prompt, feather) mengatur seberapa banyak wilayah boleh saling memengaruhi.

LoRA **tidak** ikut dipisah. LoRA mengubah bobot model untuk seluruh gambar, jadi LoRA karakter A tetap berpengaruh ke karakter B.

---

## 2. Node "Comfy Multi-Region"

### Input wajib

#### `model`
Model yang akan di-patch, sudah termasuk LoRA. Node mengembalikan salinan model yang sudah dipatch; model aslinya tidak berubah.

#### `negative`
Prompt negatif. **Satu untuk seluruh gambar.** Tidak ada negative per region.

#### `orientation` — `horizontal` / `vertical`
- `horizontal`: region berjajar kiri → kanan. `positive_1` paling kiri.
- `vertical`: region bertumpuk atas → bawah. `positive_1` paling atas.

Hanya ada dua pilihan ini, pita lurus. Untuk posisi bebas, pakai node Anima Character (bagian 4).

#### `num_regions` — default 2, rentang 2–10
Jumlah region yang dipakai. Hanya `positive_1` sampai `positive_<num_regions>` yang dibaca. Kalau salah satunya tidak tersambung, node error (lihat [Error](#error)).

#### `width`, `height` — default 512, kelipatan 8
Ukuran kanvas tempat mask digambar. Di branch ini, mask selalu di-resize ke ukuran latent yang sebenarnya saat sampling, baik di SDXL maupun Anima. Jadi yang berpengaruh hanya **proporsi** region, bukan angka absolutnya. Tetap disarankan diisi sama dengan ukuran gambar supaya pembulatan batas region tepat.

#### `isolation_factor` — default 0.5, rentang 0–1
- **SDXL: tidak berpengaruh.** Nilainya dipakai untuk "mempertajam" mask lewat `mask^(e^isolation_factor)`. Mask dari node ini selalu keras (0 atau 1), dan 0 atau 1 dipangkatkan berapa pun hasilnya tetap sama.
- **Anima:** memblokir self-attention antar region. Perhatian sebuah token ke token di region lain dikalikan `(1 − isolation_factor)²`.
  - 0 = mati, perilakunya sama dengan model biasa.
  - 0.5 = perhatian ke region lain tinggal 25%.
  - 0.9 = tinggal 1%.
  - 1.0 = praktis nol (bias dibatasi di −18.4, jadi bobotnya sekitar 1e-8).
- Di Anima, bias ini dipakai di pass positif **dan** pass negatif. Kalau tidak, CFG akan memperbesar selisih antara keduanya.
- Isolasi hanya aktif kalau nilainya > 0, ada minimal 2 region, dan step saat ini berada di jendela `isolation_start`–`isolation_end`.
- Token di pita lembut `mask_feather` (bukan milik penuh satu region) tidak dibatasi.
- **Efek samping:** apa pun yang melewati batas region ikut terpotong. Di pengujian, isolasi 0.9 ke atas memotong lengan tepat di garis batas.
- **Rekomendasi Anima: 0.** Di pengujian, 0 sudah cukup menjaga atribut tiap karakter tetap terpisah. Naikkan pelan-pelan (0.2–0.3) hanya kalau atribut masih bocor.

### Input opsional

#### `cross_region_blend` — default 0.3, rentang 0–1
Berlaku di SDXL dan Anima.
- Node menghitung kemiripan antar prompt region: cosine similarity dari rata-rata vektor prompt (di SDXL vektor CLIP; di Anima output adapter teks dari token yang valid), dibatasi ≥ 0, dikuadratkan, lalu dinormalisasi per baris.
- Output cross-attention tiap region lalu dicampur dengan region lain sesuai kemiripan itu: `out_r = (1 − b) · out_r + b · Σ_j sim[r, j] · out_j`.
- Hanya di pass positif. 0 = mati.
- **Kenapa ini sumber kebocoran:** prompt yang banyak berbagi tag (tag kualitas, `2girls`, dan sejenisnya) dianggap mirip, jadi output region A ikut masuk ke region B. Kalau tag sudah ada di semua prompt, blend tidak menambahkan apa-apa; ia hanya merata-ratakan.
- **Rekomendasi: 0** (default 0.3 berasal dari versi SDXL).

#### `positive_1` … `positive_10`
Prompt per region, sesuai urutan `orientation`. Hanya sebanyak `num_regions` yang dibaca.

#### `ratio_1` … `ratio_9` — default 0.5, rentang 0–1
Lebar (horizontal) atau tinggi (vertical) tiap region, **kecuali region terakhir**.

Aturan persisnya:
1. Ambil `ratio_1` … `ratio_(num_regions − 1)`.
2. Region terakhir mendapat sisa, yaitu `max(0, 1 − jumlah ratio sebelumnya)`.
3. Semua angka lalu dinormalisasi supaya totalnya 1.

Jebakan penting:
- `num_regions` 3 dengan default (0.5, 0.5) membuat region ke-3 lebarnya **0** dan tidak pernah muncul. Untuk 3 region yang sama besar, isi 0.33 dan 0.33.
- Kalau jumlahnya > 1 (misalnya 0.7 + 0.7), region terakhir jadi 0 dan dua sisanya dinormalisasi menjadi 0.5 / 0.5.

Batas region dibulatkan ke bawah per piksel.

#### `weight_1` … `weight_10` — default 1.0, rentang 0–10
**Tidak berpengaruh di node ini.** Bobot dikalikan ke mask, lalu mask dinormalisasi per piksel supaya total semua region = 1. Region di node ini tidak pernah tumpang tindih, jadi setiap piksel hanya punya satu region dan hasil normalisasinya selalu 1, berapa pun bobotnya.

Pengecualian: **0 jangan dipakai.**
- SDXL: menghasilkan 0/0 = NaN.
- Anima: piksel region itu jadi membaca semua prompt sama rata.

`weight` baru berarti di node Anima Character Prompts, di mana area boleh bertumpuk.

#### `model_type` — `SDXL (Illustrious / Pony / SDXL Base)` (default) / `Anima`
Memilih jalur patch.
- SDXL juga mendeteksi SD1.5 otomatis.
- Kalau pilihan tidak cocok dengan model yang tersambung, node error.

Fitur berikut **hanya untuk Anima**: `isolation_factor` (efek nyata), `isolation_start`/`isolation_end`, `base_positive`/`base_weight`, dan `mask_feather`. Di jalur SDXL semua itu diabaikan.

#### `isolation_start` — default 0.0, dan `isolation_end` — default 1.0 (Anima)
Jendela step tempat isolasi aktif, dalam pecahan dari jumlah step sampling yang sedang berjalan. 0 = step pertama (paling banyak noise), 1 = akhir.
- Isolasi aktif kalau `isolation_start ≤ progress < isolation_end`, dengan `progress = indeks step / jumlah step`.
- Hanya berpengaruh kalau `isolation_factor > 0`.
- Di hires pass (KSampler kedua, denoise < 1), jendela dihitung relatif terhadap step pass itu sendiri.
- Contoh: start 0.3 berarti 30% step pertama bebas (pose dan komposisi terbentuk dengan region saling "melihat"), lalu isolasi menyala. End 0.9 berarti 10% step terakhir bebas lagi untuk merapikan sambungan.

#### `base_positive` (Anima)
Prompt untuk **seluruh gambar**: interaksi, scene, latar. Dicampur ke semua region di pass positif. Pass negatif tidak terpengaruh, dan base tidak ikut dihitung di `cross_region_blend`.

#### `base_weight` — default 0.4, rentang 0–1 (Anima)
Hanya dipakai kalau `base_positive` tersambung.
- Di setiap token: `output = (1 − base_weight) · output_region + base_weight · output_base`.
- 1.0 = prompt region diabaikan seluruhnya.
- **Apa pun yang ada di base akan muncul di semua region.** Di pengujian, base prompt di 0.4 memunculkan atribut yang tidak diinginkan.
- **Rekomendasi:** 0.1–0.2, isi base seminimal mungkin (interaksi dan scene saja, tanpa atribut karakter), atau tidak dipakai sama sekali.
- Di pengujian, `2girls` lebih ampuh ditaruh di prompt tiap region daripada di base.

#### `mask_feather` — default 0, rentang 0–16 token (Anima)
Melembutkan batas region. Mask token di-blur dengan kotak (2k+1)×(2k+1), lalu dinormalisasi lagi. Hasilnya, di sekitar setiap batas ada transisi selebar k token (k × 16 px) di tiap sisi, tempat dua prompt bercampur bertahap.
- Token di pita transisi tidak dibatasi isolasi, jadi dua region bisa menyambung di situ.
- 0 = batas keras.
- 2–3 menyamarkan garis sambungan.
- Lebih besar membantu objek yang melewati batas (misalnya tangan), tapi juga memperlebar area yang membaca dua prompt sekaligus.

### Output (sambungkan semuanya)
- `MODEL`: model yang sudah dipatch.
- `CONDITIONING` (positive): prompt region pertama. Hanya bermakna jika dipakai bersama model dari output yang sama, karena patch yang menentukan prompt per wilayah.
- `CONDITIONING` (negative).

Ketiga output harus masuk ke **setiap** KSampler, termasuk KSampler hires. Kalau hires pass memakai model atau prompt biasa, pemisahan region hilang dan gambar dirender ulang dengan satu prompt.

### Error
Kalau terjadi error (prompt kurang, `model_type` tidak cocok, dan lain-lain), node **tidak** menghentikan workflow. Ia mencetak `An error occurred: ...` di console dan mengeluarkan `None`, sehingga yang biasanya terlihat adalah error di KSampler. Selalu cek console.

---

## 3. Setelan awal yang sudah terbukti di pengujian pengguna

**Anima (Comfy Multi-Region):**

| Parameter | Nilai |
|---|---|
| `model_type` | Anima |
| `isolation_factor` | 0 |
| `cross_region_blend` | 0 |
| `2girls` / tag jumlah karakter | di prompt setiap region |
| `base_positive` | opsional, `base_weight` 0.1–0.2, isi minimal |
| `mask_feather` | 2 |
| CFG | 4–5 (rekomendasi model card Anima; CFG 8 memperkuat beda antar region) |

**SDXL / Illustrious:**
- `cross_region_blend` 0.
- Bagian gambar yang menyeberang ke region lain akan ikut prompt region itu. Misalnya tangan karakter B yang masuk ke wilayah A ikut atribut A, karena SDXL punya sedikit self-attention dan teks yang dominan. Di Anima hal ini lebih jarang karena self-attention global membawa identitas objek.
- Solusinya: geser `ratio_*` supaya batas tidak jatuh di bagian yang menyeberang, atau pakai 3 region dengan strip tengah yang punya prompt khusus untuk bagian yang menyeberang (contoh: `white hand, hand pulling`).

---

## 4. Node Anima Character Prompts (menu loaders → anima)

Mesinnya sama dengan jalur Anima di atas, tapi wilayah karakter bisa diletakkan di mana saja, tidak lagi berupa pita.

### "Anima Character" (satu node per karakter)

| Parameter | Default | Arti |
|---|---|---|
| `conditioning` | — | Prompt khusus karakter ini (rupa, baju, ekspresi). |
| `x`, `y` | 0.5, 0.5 | Titik tengah kotak. 0 = kiri/atas, 1 = kanan/bawah. |
| `width`, `height` | 0.5, 1.0 | Ukuran kotak sebagai pecahan dari lebar/tinggi gambar (0.01–1). |
| `weight` | 1.0 | Bobot di area yang **bertumpuk** dengan karakter lain. Contoh: 3 vs 1 → 75% / 25%. Di area yang tidak bertumpuk tidak berpengaruh. |
| `mask` (opsional) | — | Bentuk area bebas, misalnya dilukis di mask editor. Kalau tersambung, kotak diabaikan. |

### "Anima Character Prompts"

| Parameter | Default | Arti |
|---|---|---|
| `base` | — | Prompt scene. Menguasai semua area yang tidak ditempati karakter, dan dicampur ke area karakter sebesar `base_weight`. |
| `negative` | — | Negatif global. |
| `character_1` … `character_6` | — | Output dari node Anima Character. Minimal satu. |
| `base_weight` | 0.2 | Lihat `base_weight` di atas. |
| `mask_feather` | 2 | Lihat `mask_feather` di atas. |
| `isolation_factor` | 0 | Lihat di atas. |
| `isolation_start` / `isolation_end` | 0 / 1 | Lihat di atas. |

`cross_region_blend` di node ini selalu 0.

### "Anima Auto Character Prompts" (tanpa posisi, eksperimental)

| Parameter | Default | Arti |
|---|---|---|
| `base`, `negative`, `character_1` … `character_6` | — | Prompt biasa (CLIP Text Encode), tanpa posisi. |
| `layout_end` | 0.25 | Pecahan step pertama yang memakai **semua prompt digabung**. Selama step ini node mengukur token gambar mana yang memperhatikan token pembeda tiap karakter. Setelah itu setiap karakter hanya membaca prompt-nya sendiri di area yang ditemukan. Makin tinggi, area makin akurat, tapi makin banyak step yang prompt-nya tercampur (atribut besar seperti warna rambut bisa terkunci di sini). |
| `sharpness` | 4.0 | Seberapa tegas batas area. Diterapkan pada z-score peta tiap karakter; base mendapat nilai 0 sebagai latar. |
| `base_weight` | 0.2 | Seperti di atas, berlaku setelah fase layout. |
| `mask_feather` | 2 | Pelembut area yang ditemukan. Sebelumnya peta selalu dihaluskan 5×5. |
| `save_layout_preview` | false | Simpan `anima_auto_layout_xxxxx.png` ke folder output. Abu-abu = base, merah/biru/hijau/kuning/ungu/cyan = karakter 1–6. Kalau gambarnya satu warna rata, artinya karakter tidak ditemukan. |

Detail cara kerjanya:
- **Token pembeda** adalah token prompt karakter yang tidak ada di prompt lain (base maupun karakter lain), misalnya nama, warna rambut atau mata, dan trigger LoRA. Tag bersama tidak diukur karena menarik perhatian yang sama untuk semua karakter.
- Pengukuran dilakukan di blok tengah model (n/4 sampai 3n/4), hanya di pass positif. Step yang lebih akhir diberi bobot lebih besar.
- Console mencetak per karakter: jumlah token pembeda, kontras peta (mendekati 0 = tidak ditemukan), dan luas area.
- Hires pass mencari layout lagi di resolusinya sendiri.

---

## 5. Jebakan umum

1. `weight_i` di Comfy Multi-Region tidak berpengaruh (lihat di atas). Untuk memperkuat satu tag, pakai bobot di prompt, misalnya `(tag:1.3)`.
2. `num_regions` 3 dengan ratio default membuat region ketiga hilang.
3. Negative bersifat global dan tidak bisa per region.
4. LoRA bersifat global. LoRA karakter A ikut memengaruhi karakter B.
5. Kata penyangkal di positive tidak membalik makna. `unmasked` tetap membawa makna "masked" (text encoder Anima adalah LLM). Hapus kata itu, jangan dinegasikan.
6. Hires KSampler harus memakai output model, positive, dan negative dari node ini.
7. Setelah `git pull`: restart ComfyUI **dan** refresh browser. Widget baru ditambahkan di akhir node supaya nilai widget di workflow lama tidak bergeser, tapi tetap cek ulang setelah update.
8. Node yang benar-benar berjalan bisa dicek dengan mengarahkan kursor ke `isolation_factor`. Tooltip yang menyebut "Anima: ..." berarti versi baru.

---

## Ketersediaan per branch

| Fitur | `main` | `claude/nifty-bohr-8rwfch` | `claude/anima-character-prompts` |
|---|---|---|---|
| `model_type` Anima, `isolation_factor` (Anima) | ya | ya | ya |
| Grid token dibaca dari latent (`width`/`height` cukup proporsi) | **tidak**: di Anima, `width`/`height` harus sama rasionya dengan gambar | ya | ya |
| `isolation_start` / `isolation_end` | tidak | ya | ya |
| `base_positive` / `base_weight` / `mask_feather` | tidak | ya | ya |
| Anima Character / Character Prompts / Auto Character Prompts | tidak | tidak | ya |
