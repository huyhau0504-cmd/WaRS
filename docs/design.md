# Thiết kế eWaSR-H

Tài liệu này ghi lại thiết kế của **eWaSR-H**, biến thể eWaSR chạy được trên Hailo-8L. Nội dung gồm:
- các công thức và lý do chọn chúng,
- điểm mạnh và điểm yếu của từng khối,
- những thay đổi qua các bước WaSR → eWaSR → eWaSR-H, kèm lý do.

Mỗi khi thay đổi thiết kế, cập nhật tài liệu này và thêm một dòng vào mục *Nhật ký*.

Mã nguồn tương ứng:
- [wars/models/layers.py](../wars/models/layers.py): các khối
- [wars/models/ewasr.py](../wars/models/ewasr.py): ghép mạng
- [wars/models/\_\_init\_\_.py](../wars/models/__init__.py): danh sách biến thể
- [wars/losses.py](../wars/losses.py): hàm mất mát
- [tools/train.py](../tools/train.py): huấn luyện

Các con số đo trên chính mã của repo, với đầu vào 384×512, trừ những chỗ ghi *(theo bài báo, cần đối chiếu)*.

---

## 1. Bài toán và ràng buộc

- **Đầu vào:** một ảnh RGB. Không dùng IMU, không dùng stereo.
- **Đầu ra:** mask 3 lớp, gồm 0 = vật cản, 1 = nước, 2 = trời.
- Vật cản được định nghĩa là *mọi thứ không phải nước hay trời*. Nhờ vậy mô hình phát hiện được cả loại vật cản chưa gặp khi huấn luyện, điều mà detector như YOLO không làm được.
- **Phần cứng:** Raspberry Pi 5 + Hailo-8L (13 TOPS, INT8). Mục tiêu ≥ 20 FPS khi chỉ tính suy luận, và F1 giảm ≤ 2–3 điểm sau INT8 ([de_cuong.md](de_cuong.md) §4.2).
- **Hệ quả cho thiết kế:**
  - Chỉ dùng các phép mà Hailo DFC biên dịch tốt, gồm Conv, BN (gộp vào Conv), ReLU, Sigmoid, Add/Mul, pooling/resize với hệ số cố định.
  - Shape phải tĩnh.
  - Phần tính toán nặng nhưng đơn giản, như upsample về độ phân giải ảnh, được đưa ra ngoài mô hình.

## 2. Từ WaSR đến eWaSR-H

| Thành phần | WaSR (2022) | eWaSR (2023) | **eWaSR-H (đề tài)** |
|---|---|---|---|
| Backbone | ResNet-101, tích chập giãn (dilated) | ResNet-18 | ResNet-18 (giữ nguyên) |
| IMU | Có (mã hóa đường chân trời vào decoder) | Tùy chọn | **Không** |
| Decoder | ARM + FFM + ASPP *(theo bài báo, cần đối chiếu)* | Kiểu TopFormer: PPA → feature mixer (MetaFormer) → SIM | Giữ kiểu TopFormer |
| Attention không gian | — | `S`: mean/max theo trục kênh → conv 3×3 → sigmoid | **`D`: depthwise 3×3 → BN → conv 1×1 → sigmoid** |
| Attention kênh | ARM | `C`: ARM (GAP → 1×1 → BN → sigmoid) | `C`: giữ nguyên |
| Upsample đầu ra | Trong mạng | Trong mạng | **Ngoài mạng** (logits H/4×W/4) |
| Resize/pool nội bộ | Tùy kích thước | Adaptive pool, resize theo `size` | **Hệ số nguyên cố định** (AvgPool kernel cố định, Resize theo `scale_factor`) |
| Chuẩn hóa ảnh | Trên CPU | Trên CPU | Trong model script của Hailo (`normalization1`), chạy trên chip |
| Loss | Focal + separation | Focal + separation | Giữ nguyên |

### Lý do từng thay đổi

1. **WaSR → eWaSR (đổi backbone ResNet-101 sang ResNet-18, đổi decoder):** đây là thay đổi của tác giả eWaSR. ResNet-101 với tích chập giãn quá nặng cho thiết bị nhúng. Decoder kiểu TopFormer chỉ tính phần "toàn cục" trên một lưới token rất nhỏ, nên rẻ về FLOPs mà vẫn giữ ngữ cảnh toàn ảnh. Ngữ cảnh toàn ảnh quan trọng để phân biệt vật cản với bóng phản chiếu hay sóng.
2. **Bỏ IMU:** dữ liệu USV-VN và hệ thống trên Pi không có IMU đồng bộ. Đề tài cũng giới hạn chỉ dùng camera đơn ([de_cuong.md](de_cuong.md) §3.2). Bỏ IMU còn giúp nạp trọng số eWaSR chính thức bản không IMU.
3. **Đổi `S` sang `D`:** đây là thay đổi chính của eWaSR-H, chi tiết ở §3.3. Khối `S` sinh ra ReduceMean và ReduceMax theo trục kênh. Bộ tăng tốc dataflow như Hailo thường hỗ trợ kém các phép này, hoặc phải đẩy chúng về CPU, làm vỡ đồ thị thành nhiều đoạn. Khối `D` chỉ gồm Conv, BN và Sigmoid.
4. **Đưa upsample ra ngoài mạng:** từ logits 96×128×3 lên ảnh gốc là thao tác thuần băng thông, không có tham số. Làm trên chip sẽ tốn bộ nhớ đệm và băng thông PCIe gấp 16 lần cho đầu ra. Trên Pi, việc này làm bằng OpenCV (`wars/postprocess.logits_to_mask`). Thứ tự là **upsample logits trước, argmax sau** để biên mượt hơn so với upsample mask.
5. **Hệ số resize/pool cố định, shape tĩnh:** DFC cần đồ thị tĩnh. `resize_like` và `_pool_to` ép shape về `int` và dùng `scale_factor` nguyên hoặc AvgPool có kernel cố định, nhờ đó ONNX sinh ra các phép mà DFC nhận dạng được. Ràng buộc kèm theo là H và W phải chia hết cho 32. Dùng 64 thì an toàn hơn, vì PPA hạ thêm ×2 (§3.2).
6. **Chuẩn hóa trên chip:** Pi chỉ cần gửi ảnh RGB uint8 (NHWC), không phải đổi sang float. Việc này giảm thời gian tiền xử lý trên CPU và giảm 4 lần lượng dữ liệu truyền.

## 3. Kiến trúc eWaSR-H

Kích thước đặc trưng với đầu vào 384×512:

```
ảnh 3×384×512 (đã chuẩn hóa ImageNet)
 └─ ResNet-18
     ├─ layer1 → skip1  64×96×128   (1/4)
     ├─ layer2 → skip2 128×48×64    (1/8)  ──► enricher "DD" (MetaFormer ×2)
     ├─ layer3 → aux   256×24×32    (1/16) ──► (dùng cho separation loss khi huấn luyện)
     └─ layer4 → out   512×12×16    (1/32)
 PPA: AvgPool cả 4 mức về 6×8 (1/64), nối kênh → tokens 960×6×8
 Feature mixer "CCCCDD": 6 khối MetaFormer trên tokens 960×6×8
 Tách tokens theo kênh [512 | 256 | 128 | 64] → SIM1..4 (tiêm ngữ cảnh vào từng mức, ra 256 kênh)
 Cộng 4 nhánh sau khi upsample bilinear về 1/4 → 256×96×128
 SegHead: 1×1 → BN → ReLU6 → 1×1 → logits 3×96×128
 (ngoài mạng) upsample bilinear lên kích thước ảnh → argmax → mask
```

### 3.1 Backbone ResNet-18

- **Lý do:** đã được kiểm chứng trong eWaSR. Mạng chỉ gồm Conv, BN, ReLU, MaxPool và Add, đều là phép Hailo hỗ trợ tốt nhất. Có sẵn trọng số ImageNet và trọng số eWaSR chính thức.
- **Điểm mạnh:** 11,2 triệu tham số. Backbone chiếm 7,1 trên tổng 12,0 GMAC (59%), và đây là kiểu tính toán mà phần cứng INT8 xử lý hiệu quả nhất.
- **Điểm yếu:** trường tiếp nhận và năng lực biểu diễn kém ResNet-101. Phần bù đắp trông vào decoder toàn cục. Vật cản rất nhỏ (vài pixel) có thể mất ngay ở các mức 1/16 và 1/32.

### 3.2 Pyramid Pool Aggregation (PPA)

$$T = \operatorname{concat}_k\big(\operatorname{AvgPool}_{s_k}(f_k)\big),\quad f_k \in \{\text{out},\text{aux},\text{skip2},\text{skip1}\}$$

Mỗi mức được đưa về cùng lưới $\lceil H/64\rceil \times \lceil W/64\rceil$ (6×8). Kernel bằng stride: ×2, ×4, ×8 và ×16.

- **Lý do:** gom thông tin đa tỉ lệ vào một lưới token nhỏ, để các khối "toàn cục" phía sau gần như không tốn FLOPs.
- **Điểm mạnh:** chỉ gồm AvgPool và Concat, rẻ và thân thiện phần cứng.
- **Điểm yếu:** lưới 6×8 rất thô, mỗi token đại diện cho 64×64 pixel. Chi tiết nhỏ ở đây chỉ còn ở dạng trung bình, nên việc giữ vật cản nhỏ trông vào nhánh cục bộ của SIM (§3.5).

### 3.3 Token mixer: `C` (giữ nguyên), `S` (gốc, bị thay), `D` (mới)

Mọi mixer có dạng *cổng × đầu vào*: $y = A(x) \odot x$.

**`C`: ARM, attention theo kênh.**

$$A_C(x) = \sigma\big(\mathrm{BN}(W_{1\times1}\,\mathrm{GAP}(x))\big)\in\mathbb{R}^{C\times1\times1}$$

- **Lý do:** chọn lọc kênh dựa trên thống kê toàn ảnh. Ví dụ: tăng trọng số các kênh nhạy với "nước" khi ảnh nhiều nước.
- **Điểm mạnh:** chỉ gồm GlobalAveragePool, Conv 1×1, Sigmoid và Mul, đều thân thiện Hailo.
- **Điểm yếu:** không có thông tin vị trí, nên một mình `C` không chỉ ra được *vật cản nằm ở đâu*. Ablation `ewasr_c` (chỉ dùng `C`) dùng để định lượng điểm yếu này.

**`S`: spatial attention gốc của eWaSR.**

$$A_S(x) = \sigma\Big(W_{3\times3} * \big[\operatorname{mean}_c(x);\ \operatorname{max}_c(x)\big]\Big)\in\mathbb{R}^{1\times H\times W}$$

- **Điểm mạnh:** cực rẻ (18 tham số). Nhánh `max` giữ được các đỉnh kích hoạt mạnh ở vài kênh, hữu ích cho vật thể nhỏ nổi bật.
- **Điểm yếu đối với Hailo:** sinh ra `ReduceMean` và `ReduceMax` theo trục kênh, cộng thêm một `Concat`. Trong ONNX của `ewasr` có 4 `ReduceMax` và 4 `ReduceMean`. Đây là lý do thay khối này.

**`D`: depthwise spatial gate, khối của eWaSR-H.**

$$A_D(x) = \sigma\Big(\textstyle\sum_{c} w_c\,\mathrm{BN}\big(k_c * x_c\big) + b\Big)\in\mathbb{R}^{1\times H\times W}$$

Trong đó $k_c$ là kernel 3×3 riêng của từng kênh (depthwise), và $w_c$ là trọng số của conv 1×1 từ $C$ kênh về 1 kênh.

- **Lý do:** giữ đúng chức năng của `S`, tức là một bản đồ cổng 1 kênh theo không gian, nhân broadcast vào mọi kênh. Khác biệt là thay *phép gộp cố định* (mean/max theo kênh) bằng *phép gộp tuyến tính học được* (conv 1×1). Toàn khối chỉ gồm Conv, BN và Sigmoid. BN được gộp vào depthwise conv khi biên dịch.
- **Liên hệ với `S`:** nếu $w_c = 1/C$ và $k_c$ là identity, `D` trở thành nhánh `mean` của `S`. Vì vậy `D` bao hàm nhánh mean, đồng thời học được kênh nào quan trọng và lọc không gian riêng cho từng kênh trước khi gộp.
- **Điểm mạnh:** không có Reduce theo trục kênh. Số tham số nhỏ, $10C+1$ cộng BN; với $C=960$ là khoảng 11k. Trong ONNX của `ewasr_h` không còn `ReduceMax`/`ReduceMean` (§6).
- **Điểm yếu:**
  - **Mất nhánh `max`.** Tổ hợp tuyến tính không mô phỏng được max, nên đỉnh mạnh ở một vài kênh có thể bị pha loãng. Đây là rủi ro cho vật cản nhỏ, cần kiểm tra bằng kết quả theo kích thước vật cản.
  - **4 khối `D` khởi tạo ngẫu nhiên.** Trọng số chính thức không có các khối này, trong khi lr của decoder rất nhỏ (§5.3), nên có thể hội tụ chậm.
  - **Phép nhân broadcast 1 kênh × C kênh** cần được DFC hỗ trợ. Điều này mới là giả định, chưa kiểm chứng (§7).

### 3.4 Khối MetaFormer (feature mixer `CCCCDD` và enricher `DD`)

$$x \leftarrow x + \gamma_1 \odot \mathrm{Mixer}\big(\mathrm{BN}(x)\big),\qquad x \leftarrow x + \gamma_2 \odot \mathrm{MLP}\big(\mathrm{BN}(x)\big)$$

Trong đó MLP là Conv 1×1 (C→4C) → ReLU → Conv 1×1 (4C→C), và layer scale $\gamma$ được khởi tạo bằng $10^{-5}$.

- **Lý do:**
  - Cấu trúc MetaFormer tách phần trộn token (mixer) với phần trộn kênh (MLP) và có residual, nên ổn định khi xếp chồng nhiều khối.
  - Thứ tự 4 khối `C` rồi 2 khối `D` giữ đúng vị trí `CCCCSS` của eWaSR, để các khối `C` nạp lại được trọng số chính thức.
  - Dùng BN thay LayerNorm, vì BN gộp được vào Conv khi suy luận, trong khi LayerNorm khó lượng tử hóa và Hailo hỗ trợ kém hơn.
  - $\gamma$ nhỏ khiến mỗi khối mới thêm vào gần với identity lúc đầu, nên thay mixer không phá ngay đặc trưng đã học.
- **Enricher `DD` trên skip2 (128×48×64):** tinh chỉnh đặc trưng 1/8, là mức có độ phân giải còn đủ cho vật cản nhỏ, trước khi đưa vào SIM3.
- **Điểm mạnh:** chạy trên lưới 6×8 (48 vị trí), nên dù có 6 khối rộng 960 kênh, feature mixer chỉ tốn 2,1 GMAC (18% toàn mạng). Enricher tốn 0,8 GMAC.
- **Điểm yếu, và cũng là điểm yếu lớn nhất của cả mạng:** *tham số* rất lớn. MLP 960→3840→960 có khoảng 7,4 triệu tham số mỗi khối, nhân 6 thành **48 triệu trên tổng 60,3 triệu**. Như vậy 80% trọng số chỉ phục vụ 18% FLOPs, và mỗi trọng số chỉ được dùng lại 48 lần. Trên Hailo-8L (bộ nhớ trên chip hạn chế), riêng trọng số INT8 đã khoảng 60 MB. Có thể phải chia mạng thành nhiều context, làm giảm FPS. Đây là lý do có biến thể `ewasr_h_p` (§3.7).

### 3.5 Semantic Injection Module (SIM)

$$\mathrm{SIM}(l, g) = \mathrm{BN}(W_l\, l)\ \odot\ \sigma\big(\mathrm{BN}(W_a\,\uparrow g)\big)\ +\ \mathrm{BN}(W_g\,\uparrow g)$$

Trong đó $l$ là đặc trưng cục bộ của một mức, $g$ là phần token toàn cục tương ứng (cắt theo kênh), và $\uparrow$ là upsample nearest theo hệ số nguyên.

- **Lý do (lấy từ TopFormer):** ngữ cảnh toàn cục vừa *điều biến* đặc trưng cục bộ qua cổng sigmoid, vừa được *cộng* trực tiếp vào. Chi tiết vị trí đến từ $l$, ngữ nghĩa đến từ $g$.
- **Điểm mạnh:** chỉ gồm Conv 1×1, BN, Sigmoid, Mul, Add và Resize nearest với hệ số nguyên.
- **Điểm yếu:**
  - Ngữ cảnh $g$ có độ phân giải 6×8 rồi được upsample nearest, nên có thể sinh hiệu ứng "khối" ở biên nước–bờ. Phần này được bù bằng việc cộng 4 nhánh ở 1/4.
  - Hệ số upsample lớn (×16 cho skip1) tạo tensor trung gian lớn trên chip.

### 3.6 Hợp nhất và đầu phân đoạn

$$F = \sum_{k=1}^{4} \uparrow_{\text{bilinear}}\mathrm{SIM}_k,\qquad \text{logits} = W_2\,\mathrm{ReLU6}\big(\mathrm{BN}(W_1 F)\big)\in\mathbb{R}^{3\times H/4\times W/4}$$

- **Lý do dùng ReLU6:** giữ nguyên như bản gốc. Miền giá trị bị chặn trên nên dễ lượng tử hóa INT8.
- **Lý do xuất ở 1/4:** đã nêu ở §2, thay đổi 4.

### 3.7 Các biến thể dùng trong đề tài

| Tên | Mixer / enricher | Mục đích |
|---|---|---|
| `ewasr` | `CCCCSS` / `SS` | Đường cơ sở. Nạp được 100% trọng số chính thức |
| **`ewasr_h`** | `CCCCDD` / `DD` | **Mô hình chính** |
| `ewasr_c` | `CCCCCC` / `CC` | Ablation: bỏ hẳn attention không gian. `D` có thực sự cần không? |
| `ewasr_h_p` | `CCCCDD` / `DD` + chiếu ½ kênh trước decoder | Dự phòng khi Hailo-8L không chứa nổi 60 triệu tham số. Mixer chạy trên 480 kênh nên tham số giảm khoảng 4 lần |

## 4. Hàm mất mát

Tổng loss: $\mathcal{L} = \mathcal{L}_{\text{focal}} + \lambda\,\mathcal{L}_{\text{sep}}$, với $\lambda = 0.01$.

### 4.1 Focal loss

$$\mathcal{L}_{\text{focal}} = -\frac{1}{N}\sum_{i}\sum_{c} y_{ic}\,(1-p_{ic})^{\gamma}\log p_{ic},\qquad \gamma = 2$$

Logits được upsample bilinear lên kích thước nhãn trước khi tính softmax $p$.

- **Lý do:** pixel nước và trời chiếm phần lớn ảnh và rất dễ đoán. Hệ số $(1-p)^\gamma$ giảm trọng số các pixel dễ, để gradient tập trung vào biên và vật cản nhỏ.
- **Pixel bị bỏ qua:** nhãn MaSTr1325 có giá trị 4 (không xác định) và viền đen sinh ra sau Affine augmentation đều có one-hot $y_i = 0$, nên không đóng góp vào loss.
- **Điểm yếu:** $N$ đếm cả pixel bị bỏ qua (vì dùng `mean`). Do đó độ lớn loss phụ thuộc tỉ lệ pixel bị bỏ qua của từng batch. Ảnh hưởng nhỏ và giữ đúng như bản gốc.

### 4.2 Water-obstacle separation loss (của WaSR)

Loss này tính trên đặc trưng `aux` (layer3, 256×24×32). Mask được thu nhỏ bằng `area`, nên giá trị mask là tỉ lệ diện tích.

$$\mu_w = \frac{\sum_{i\in W} f_i}{N_w},\quad \sigma_w^2 = \frac{\sum_{i\in W}(f_i-\mu_w)^2}{N_w},\quad \mathcal{L}_{\text{sep}} = \operatorname{mean}_c\frac{N_o\,\sigma_{w,c}^2}{\sum_{i\in O}(f_{i,c}-\mu_{w,c})^2 + \varepsilon}$$

Loss bằng 0 khi batch có ít hơn 5 pixel nước hoặc vật cản.

- **Lý do:** buộc đặc trưng nước co cụm (phương sai nhỏ), đồng thời đẩy đặc trưng vật cản ra xa tâm cụm nước. Khi gặp vật cản lạ, chỉ cần nó "không giống nước" là bị tách ra. Đây chính là tính chất phát hiện vật cản chưa gặp.
- **Lý do đặt ở layer3:** độ phân giải 1/16 đủ mịn để còn thấy vật cản, và đủ sâu để mang ngữ nghĩa. Vị trí này giữ đúng như WaSR/eWaSR.
- **Điểm yếu:** thống kê tính trên cả batch (batch_size = 4) nên khá nhiễu. Loss cũng giả định "nước" là một cụm duy nhất, trong khi nước sông Việt Nam đục, ám màu và có bóng phản chiếu bờ, nên có thể đa dạng hơn MaSTr1325.

## 5. Huấn luyện

### 5.1 Dữ liệu
- **MaSTr1325:** 1299 ảnh train và 26 ảnh val (`configs/mastr1325/`). Ảnh được resize về 384×512 bằng `INTER_AREA`.
- **Augmentation** (theo eWaSR, [wars/data/transforms.py](../wars/data/transforms.py)):
  - HorizontalFlip.
  - Affine với scale 1–1,3, xoay ±15°, dịch ±6,25%, p = 0,7.
  - ColorJitter + RandomGamma.
  - Gauss/ISO noise.
  - Tùy chọn `vn_color_aug`: RGBShift + độ sáng/tương phản, mô phỏng nước sông ám màu (§6.4 đề cương).
- **Điểm yếu:** tập val chỉ có 26 ảnh, nên mIoU val dao động mạnh. `best.pth` chọn theo mIoU val có thể không phải bản tốt nhất trên MODS/USV-VN. Cần đánh giá thêm `final.pth`.

### 5.2 Khởi tạo
`init_weights: weights/ewasr_resnet18.pth` được nạp với `strict=False`, tức chỉ nạp các khóa trùng tên và trùng shape:

| Biến thể | Số tensor phải học từ đầu |
|---|---|
| `ewasr_h` | 32 (4 khối `D` × 8 tensor). Còn lại nạp hết, kể cả $\gamma$ của các khối thay thế |
| `ewasr_c` | 28 |
| `ewasr_h_p` | 188 (gần như toàn bộ decoder do số kênh khác) |

- **Lý do:** giữ backbone và decoder đã hội tụ, chỉ để 4 khối `D` thích nghi. Cách này nhanh hơn và công bằng hơn khi so với `ewasr`, vì chỉ khác đúng một khối.
- **Rủi ro:** $\gamma$ nạp từ khối `S` cũ được áp lên đầu ra của khối `D` mới. Đầu ra ngẫu nhiên lúc đầu có thể làm lệch đặc trưng trong vài epoch đầu. Cần theo dõi mIoU ở epoch 0–2.

### 5.3 Tối ưu (siêu tham số gốc WaSR/eWaSR)
- **RMSprop:** α = 0,9, momentum = 0,9, weight decay = 1e-6.
- **lr theo nhóm:**
  - backbone: 1e-6,
  - decoder, tham số có tên chứa `weight`: ×10,
  - các tham số decoder còn lại (bias, $\gamma$): ×20.
- **Poly decay theo epoch:** $\mathrm{lr}_e = \mathrm{lr}_0\,(1 - e/E)^{0.9}$, với E = 100.
- **AMP:** forward chạy fp16, loss tính ở fp32 cho ổn định số học.
- **Lý do giữ nguyên:** để so sánh với eWaSR công bằng. Mọi khác biệt về kết quả khi đó đến từ kiến trúc.
- **Điểm yếu:** lr rất nhỏ, phù hợp fine-tune nhưng có thể chậm với các khối `D` khởi tạo ngẫu nhiên. Nếu mIoU của `ewasr_h` thấp hơn rõ so với `ewasr` sau 100 epoch, thử nâng lr riêng cho các tham số mới trước khi kết luận `D` kém hơn `S`.

### 5.4 Kế hoạch chạy trên Kaggle

Notebook: [notebooks/kaggle_train.ipynb](../notebooks/kaggle_train.ipynb), đặt `RUN` = tên config, `TRAIN_SEG = True`.

| # | Lần chạy | Config / override | So sánh để trả lời |
|---|---|---|---|
| 1 | **eWaSR-H** | `ewasr_h` | Mô hình chính |
| 2 | eWaSR huấn luyện lại | `ewasr`, `model.init_weights=weights/ewasr_resnet18.pth` | Đường cơ sở cùng quy trình: `D` so với `S` |
| 3 | Ablation chỉ kênh | `ewasr_c` (thêm `init_weights`) | Attention không gian có cần không |
| 4 | Dự phòng bộ nhớ | `ewasr_h_p` | Chỉ chạy nếu DFC báo không vừa Hailo-8L |
| 5 | Augmentation màu | `ewasr_h`, `data.vn_color_aug=true` | Giảm domain gap USV-VN |

Kết quả cần ghi lại cho mỗi lần chạy:
- mIoU và IoU từng lớp trên val (`metrics.csv`),
- Recall, FP và F1 trên USV-VN test (`usv_test.json`),
- sau này thêm F1 trên MODS.

## 6. Số liệu đo trên mã hiện tại (384×512, 08/10/2026)

| Biến thể | Tổng tham số | Backbone | Feature mixer | Enricher | GMAC | ReduceMax / ReduceMean trong ONNX |
|---|---|---|---|---|---|---|
| `ewasr` | 60,25 M | 11,18 M | 48,00 M | 0,26 M | 11,99 | 4 / 4 |
| `ewasr_h` | 60,28 M | 11,18 M | 48,02 M | 0,27 M | 11,99 | **0 / 0** |
| `ewasr_c` | 62,14 M | 11,18 M | 49,85 M | 0,30 M | 11,99 | 0 / 0 |
| `ewasr_h_p` | 23,89 M | 11,18 M | 12,03 M | 0,07 M | 9,33 | 0 / 0 |

Phân bổ GMAC của `ewasr_h`:

| Khối | GMAC | Tỉ lệ |
|---|---|---|
| Backbone | 7,11 | 59% |
| Feature mixer | 2,13 | 18% |
| Enricher (trên skip2) | 0,81 | 7% |
| SIM1–4 | 1,13 | 9% |
| SegHead (ở 1/4) | 0,81 | 7% |

Với `ewasr_h_p`, feature mixer chỉ còn 0,53 GMAC và enricher còn 0,21 GMAC.

Nhận xét:
- `ewasr_h` có **cùng FLOPs** với `ewasr`. Lợi ích của eWaSR-H **không đến từ việc giảm FLOPs**, mà đến từ việc toàn bộ đồ thị chạy được trên Hailo, không phải chia đoạn hay đẩy phép về CPU. Lợi ích này phải được chứng minh bằng log parse và profiler của DFC cùng số FPS đo thật (tuần 3–4), không chỉ bằng bảng trên.
- Toán tử ONNX còn lại của `ewasr_h`: Conv, BatchNormalization, Relu, Clip (ReLU6), Sigmoid, Add, Mul, Concat, Slice, AveragePool, GlobalAveragePool, MaxPool, Resize.

## 7. Giả định chưa kiểm chứng và rủi ro

| # | Giả định / rủi ro | Cách kiểm tra | Phương án nếu sai |
|---|---|---|---|
| 1 | DFC parse được `ewasr_h` trọn vẹn, kể cả Mul broadcast 1 kênh × C kênh và Slice theo kênh | `hailo/compile.py --parse-only` | Thay Mul broadcast bằng conv 1×1 (1→C) có trọng số cố định bằng 1; thay Slice bằng 4 conv 1×1 riêng |
| 2 | 60 triệu tham số vừa Hailo-8L với số context chấp nhận được (≥ 20 FPS) | Profiler DFC, `hailortcli benchmark` | `ewasr_h_p`, hoặc giảm `mlp_ratio` |
| 3 | `D` không kém `S` về độ chính xác, đặc biệt với vật cản nhỏ | So sánh lần chạy 1 với lần chạy 2 theo kích thước vật cản | Thêm nhánh "max" thân thiện phần cứng, ví dụ MaxPool không gian trên đầu ra depthwise |
| 4 | Sigmoid và các phép Mul cổng chịu được INT8 | So sánh mask FP32 với mask INT8 từ emulator | Giữ các lớp này ở 16-bit trong model script |
| 5 | Ngưỡng coverage 0,5 tương đương MODS | Đối chiếu với MODS toolkit | Chỉnh `--coverage-thr` |

## 8. Nhật ký

| Ngày | Thay đổi | Lý do |
|---|---|---|
| 08/10/2026 | Tạo tài liệu. eWaSR-H = eWaSR (ResNet-18, không IMU) với `S`→`D`, upsample ngoài mạng, resize/pool hệ số cố định, chuẩn hóa trên chip | Tương thích Hailo DFC (§2) |
| 21/10/2026 | Thêm biến thể mixer-lite (`ewasr_lite` S, `ewasr_h_lite` D): feature mixer chỉ giữ MLP ở khối 0 (`mixer_mlp="100000"`), 23,35M tham số so với 60,25M | Trong trọng số eWaSR chính thức γ₂ của mixer 1–5 ≈ 1e-40 (`docs/data/layer_scale.csv`); nạp vào `ewasr_lite` cho đầu ra trùng khít bản đầy đủ. Trên Hailo-8L số context phụ thuộc các MLP này (E1 8 → 6 khi tắt). Quyết định C1 ngày 20/10: giữ S, D thành ablation |

## Tài liệu tham khảo

- B. Bovcon, M. Kristan, "WaSR — A Water Segmentation and Refinement Maritime Obstacle Detection Network," *IEEE T-Cybernetics*, 2022.
- M. Teršek, L. Žust, M. Kristan, "eWaSR — An Embedded-Compute-Ready Maritime Obstacle Detection Network," *Sensors*, 2023.
- W. Zhang et al., "TopFormer: Token Pyramid Transformer for Mobile Semantic Segmentation," *CVPR*, 2022 (PPA, SIM).
- W. Yu et al., "MetaFormer Is Actually What You Need for Vision," *CVPR*, 2022.
- C. Yu et al., "BiSeNet," *ECCV*, 2018 (Attention Refinement Module).
- T.-Y. Lin et al., "Focal Loss for Dense Object Detection," *ICCV*, 2017.
