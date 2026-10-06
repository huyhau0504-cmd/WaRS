# ĐỀ CƯƠNG ĐỒ ÁN

**Sinh viên thực hiện:** ……………………  **MSSV:** ……………
**Giảng viên hướng dẫn:** ……………………
**Thời gian thực hiện:** 07/10/2026 – 06/12/2026 (8 tuần)

---

## 1. Tên đề tài

**Tiếng Việt:** Nhận diện vật cản trên mặt nước thời gian thực cho phương tiện mặt nước không người lái (USV) bằng phân đoạn ngữ nghĩa trên nền tảng nhúng Raspberry Pi 5 và Hailo-8L

**Tiếng Anh:** Real-Time Maritime Obstacle Segmentation for Low-Cost USVs on Raspberry Pi 5 with Hailo-8L: Hardware-Aware Design and INT8 Quantization Analysis

## 2. Tính cấp thiết của đề tài

Phương tiện mặt nước không người lái (USV) được dùng ngày càng nhiều cho thu gom rác, quan trắc môi trường và khảo sát sông hồ. Để USV tự hành an toàn, nó phải phát hiện được vật cản trên mặt nước, đặc biệt là các **vật cản nhỏ** như rác nổi hay phao.

Các phương pháp phân đoạn ngữ nghĩa dành riêng cho môi trường mặt nước như **WaSR** và **eWaSR** phát hiện được mọi loại vật cản, kể cả loại chưa thấy khi huấn luyện, nhờ chia ảnh thành *nước / trời / vật cản*. Tuy vậy:

- WaSR dùng backbone ResNet-101, quá nặng cho thiết bị nhúng.
- eWaSR được tối ưu cho chip OAK-D (Myriad X), **chưa được đánh giá trên bộ tăng tốc Hailo**.
- **Ảnh hưởng của lượng tử hóa INT8** lên khả năng phát hiện vật cản nhỏ chưa được phân tích có hệ thống.
- Các bộ dữ liệu chuẩn (MaSTr1325, MODS) được thu ở biển châu Âu. Hiệu năng trên **sông đô thị Việt Nam**, nơi có rác nổi và camera đặt thấp, vẫn chưa được biết.

Raspberry Pi 5 kết hợp Hailo-8L (13 TOPS) là nền tảng rẻ và phổ biến, phù hợp với USV cỡ nhỏ. Nếu chứng minh được nền tảng này chạy thời gian thực với độ chính xác chấp nhận được, kết quả sẽ có giá trị thực tiễn cao.

## 3. Đối tượng và phạm vi nghiên cứu

### 3.1. Đối tượng nghiên cứu

- Mô hình phân đoạn ngữ nghĩa cho môi trường mặt nước (họ WaSR/eWaSR) với 3 lớp nước, bầu trời và vật cản.
- Mô hình phát hiện đối tượng YOLO (YOLOv8n), dùng làm phương pháp đối chứng.
- Nền tảng suy luận nhúng Raspberry Pi 5 (8 GB) + Hailo-8L (13 TOPS) và quy trình lượng tử hóa INT8 của Hailo Dataflow Compiler (DFC).
- Vật cản trên mặt nước: vật cản động (rác nổi, thuyền, phao) và vật cản tĩnh (bờ, cầu, cây).

### 3.2. Phạm vi nghiên cứu

- Chỉ dùng camera đơn, ảnh RGB; **không dùng IMU, không dùng stereo**.
- Mô hình phân đoạn được huấn luyện trên **MaSTr1325** và đánh giá trên **MODS** cùng **bộ dữ liệu USV tự thu**.
- Đầu ra dừng ở **danh sách vật cản** (vị trí trong ảnh, hướng tương đối). Không bao gồm điều khiển hay lập quỹ đạo tránh vật cản.
- Thử nghiệm offline: phát lại ảnh đã thu trên Raspberry Pi 5, **không thử nghiệm trên tàu thật**.

## 4. Mục tiêu

### 4.1. Mục tiêu tổng quát

Xây dựng hệ thống nhận diện vật cản trên mặt nước chạy **thời gian thực** trên phần cứng giá rẻ (Raspberry Pi 5 + Hailo-8L), có độ chính xác giảm không đáng kể so với mô hình gốc chạy trên GPU, và kiểm chứng trên dữ liệu thực tế ở Việt Nam.

### 4.2. Mục tiêu cụ thể

Các ngưỡng định lượng dưới đây là dự kiến và sẽ được chốt lại sau khi đo thử ở tuần 1.

1. Triển khai thành công eWaSR (biến thể tối ưu cho Hailo, gọi là **eWaSR-H**) lên Hailo-8L, đạt **≥ 20 FPS** khi chỉ tính suy luận.
2. Sau lượng tử hóa INT8, **F1 trên MODS giảm ≤ 2–3 điểm** so với FP32.
3. Định lượng mức suy giảm do lượng tử hóa **theo kích thước vật cản**, và đề xuất biện pháp giảm thiểu.
4. Đánh giá khả năng tổng quát hóa của mô hình (huấn luyện trên MaSTr1325) trên dữ liệu sông Việt Nam, và cải thiện bằng augmentation hoặc tinh chỉnh nhẹ.
5. So sánh hai hướng tiếp cận **phân đoạn ngữ nghĩa (eWaSR-H)** và **phát hiện đối tượng (YOLOv8n)** trên cùng phần cứng Hailo-8L.
6. Xây dựng pipeline hoàn chỉnh trên Pi 5, có số đo độ trễ đầu-cuối và công suất tiêu thụ.

## 5. Các vấn đề cần giải quyết

1. **Tương thích phần cứng.** eWaSR có các khối attention mà Hailo DFC có thể không hỗ trợ hoặc chạy chậm. Hailo-8L có tài nguyên hạn chế (13 TOPS), nên kiến trúc và độ phân giải phải phù hợp.
2. **Suy giảm do lượng tử hóa.** Chuyển từ FP32 sang INT8 có thể làm mất vật cản nhỏ ở xa và làm sai lệch đường mép nước, là những yếu tố quan trọng nhất cho an toàn của USV.
3. **Đánh đổi độ chính xác và tốc độ.** Cần chọn độ phân giải, backbone và cấu hình lượng tử hóa sao cho vừa đạt thời gian thực vừa giữ độ chính xác.
4. **Khác biệt miền dữ liệu (domain gap).** MaSTr1325 thu ở biển Slovenia với camera đặt cao. Dữ liệu tự thu là sông đô thị Việt Nam, camera sát mặt nước, ảnh bị ám màu và vật cản nhỏ.
5. **Đánh giá trên dữ liệu chỉ có bounding box.** Dữ liệu tự thu không có mask phân đoạn và gồm các khung hình liên tiếp, nên cần giao thức đánh giá công bằng, không rò rỉ dữ liệu, và so sánh được giữa phân đoạn với phát hiện đối tượng.

## 6. Phương pháp giải quyết

### 6.1. Thiết kế eWaSR-H tương thích Hailo (vấn đề 1)

- Xuất eWaSR (backbone ResNet-18, không IMU) ra ONNX, chạy parser của Hailo DFC để xác định các lớp không được hỗ trợ.
- Thay thế bằng khối tương đương thân thiện với phần cứng. eWaSR gốc dùng attention không gian tính từ *mean/max theo trục kênh* (ReduceMean/ReduceMax), là loại phép mà bộ tăng tốc dataflow thường hỗ trợ kém. eWaSR-H thay khối này bằng **cổng không gian depthwise** (*depthwise conv 3×3 → BN → conv 1×1 → sigmoid*), giữ nguyên attention kênh (*global average pooling → conv 1×1 → sigmoid*), và dùng upsample/pooling với hệ số cố định.
- Phương án dự phòng về bộ nhớ: bản eWaSR gốc có 60,3 triệu tham số, phần lớn nằm ở decoder. Biến thể eWaSR-H-P chiếu đặc trưng xuống một nửa số kênh trước decoder, còn 23,9 triệu tham số.
- Huấn luyện lại trên MaSTr1325 với hàm mất mát của WaSR (cross-entropy kết hợp *water-obstacle separation loss*).

### 6.2. Phân tích ảnh hưởng của lượng tử hóa INT8 (vấn đề 2)

- So sánh 3 mức: **FP32** (GPU), **INT8 PTQ** (lượng tử hóa sau huấn luyện) và **INT8 + QFT** (tinh chỉnh có xét lượng tử hóa của Hailo).
- Ablation: số ảnh calibration (64 / 256 / 1024), nguồn ảnh calibration (MaSTr1325 hay dữ liệu tự thu), và giữ các lớp nhạy cảm ở độ chính xác 16-bit.
- Báo cáo kết quả **theo kích thước vật cản** (nhỏ / vừa / lớn theo diện tích bounding box).

### 6.3. Khảo sát đánh đổi độ chính xác và tốc độ (vấn đề 3)

- Quét độ phân giải đầu vào (ví dụ 288×384, 384×512, 480×640). Mỗi cấu hình đo FPS, độ trễ và F1, rồi xây dựng đường biên Pareto.
- Đo trên Pi 5 bằng HailoRT. Đo công suất bằng đồng hồ USB-C.

### 6.4. Giảm domain gap (vấn đề 4)

- Đánh giá **zero-shot**: mô hình chỉ huấn luyện trên MaSTr1325, chạy trực tiếp trên dữ liệu tự thu.
- Augmentation màu (ám màu, cân bằng trắng, độ sáng) và mô phỏng góc nhìn thấp.
- Tùy chọn: tạo mask giả cho tập train của dữ liệu tự thu bằng **SAM** (bounding box làm prompt cho vật cản), kiểm tra lại bằng tay, rồi tinh chỉnh nhẹ eWaSR-H.

### 6.5. So sánh với YOLOv8n (vấn đề 5)

- Huấn luyện YOLOv8n (khởi tạo từ trọng số COCO) trên tập train của dữ liệu tự thu, biên dịch cho Hailo-8L và đo trên cùng phần cứng.
- So sánh trên tập test của dữ liệu tự thu theo **cùng một giao thức** (mục 8): bounding box của YOLO được tô thành mask vật cản và chấm theo cùng tiêu chí coverage như mô hình phân đoạn.
- So sánh **khả năng nhận diện vật cản chưa từng thấy** trên MODS. YOLO chỉ nhận diện các lớp đã huấn luyện, còn phân đoạn nhận diện mọi vật cản, nên đây là điểm khác biệt cốt lõi giữa hai hướng.

### 6.6. Hệ thống suy luận trên Raspberry Pi 5

Pipeline: đọc ảnh/video → tiền xử lý → suy luận trên Hailo-8L → hậu xử lý (tách vật cản nằm dưới đường mép nước, tính bounding box và góc phương vị tương đối) → xuất danh sách vật cản và ảnh trực quan hóa.

### 6.7. Quy trình phát triển

```
Laptop Ubuntu (viết code) ──push──► GitHub ──clone──► Kaggle (GPU: huấn luyện, đánh giá FP32, xuất ONNX)
                                                              │
Laptop Ubuntu: Hailo DFC (parse → optimize INT8 → compile hailo8l) ◄──┘
                                                              │
                                                    .hef ──► Raspberry Pi 5 + HailoRT
```

## 7. Dữ liệu

| Bộ dữ liệu | Vai trò | Ghi chú |
|---|---|---|
| **MaSTr1325** | Huấn luyện mô hình phân đoạn | 1325 ảnh, mask 3 lớp (nước / trời / vật cản) |
| **MODS** | Đánh giá chuẩn | Giao thức đánh giá theo vật cản và mép nước, có vùng nguy hiểm (danger zone) |
| **USV-VN (tự thu)** | Huấn luyện YOLO, đánh giá thực tế | 751 ảnh 640×640 từ camera trái của stereo, sông đô thị Việt Nam; 1 lớp (rác nổi), nhãn bounding box định dạng YOLO |

**Chia lại tập USV-VN theo thời gian.** Ảnh là các khung hình liên tiếp (`left_0001` đến `left_1259`). Cách chia ngẫu nhiên ban đầu khiến các khung gần giống hệt nhau nằm cả ở train lẫn test. Bộ dữ liệu được chia lại tại các khoảng trống tự nhiên trong chuỗi khung hình:

| Tập | Số ảnh | Khung hình |
|---|---|---|
| train | 431 | 1 – 754 |
| valid | 179 | 863 – 1047 |
| test | 141 | 1059 – 1259 |

Cách chia ban đầu được lưu tại `splits/original_split.csv`, và có thể tái tạo bằng `tools/resplit_by_time.py`.

## 8. Chỉ số và giao thức đánh giá

- **Độ chính xác (MODS):** Precision, Recall, F1 cho vật cản; F1 trong vùng nguy hiểm; độ chính xác đường mép nước.
- **Độ chính xác (USV-VN):**
  - Vật cản được tính là phát hiện đúng (TP) khi mask vật cản phủ đủ bounding box nhãn, theo tiêu chí của MODS. Tiêu chí này áp dụng cho cả YOLO (box dự đoán được tô thành mask).
  - Tiêu chí IoU ≥ 0.5 trên bounding box chỉ báo cáo bổ sung cho YOLO. Nó bất lợi cho mô hình phân đoạn: camera đặt sát mặt nước nên vật cản đè lên dải bờ trong ảnh, khiến mask vật cản dính liền với bờ.
  - Theo ngưỡng kích thước của COCO, mọi vật cản trong tập valid/test đều thuộc nhóm "vừa", nên phân tích theo kích thước trên USV-VN dùng ngưỡng riêng; phân tích vật cản nhỏ chủ yếu dựa trên MODS.
  - Vùng vật cản dự đoán nằm dưới đường mép nước mà không khớp nhãn nào được tính là báo động sai (FP). Đường mép nước được vẽ tay trên tập test.
  - Dữ liệu tự thu không có hiệu chỉnh camera nên **không tính vùng nguy hiểm**.
- **Hiệu năng:** FPS khi chỉ tính suy luận, độ trễ đầu-cuối (ms), công suất tiêu thụ (W).

## 9. Kế hoạch thực hiện

| Tuần | Thời gian | Công việc | Kết quả |
|---|---|---|---|
| 1 | 07/10 – 13/10 | Dựng repo GitHub và notebook Kaggle, tải dữ liệu lên Kaggle. Cài Hailo DFC trên laptop, biên dịch thử một mô hình mẫu và chạy trên Pi 5. | Quy trình từ code đến phần cứng chạy thông |
| 2 | 14/10 – 20/10 | Huấn luyện lại eWaSR gốc trên MaSTr1325, đánh giá FP32 trên MODS. | Kết quả gốc |
| 3 | 21/10 – 27/10 | Phân tích tương thích DFC, thiết kế và huấn luyện eWaSR-H. Huấn luyện YOLOv8n trên USV-VN. | eWaSR-H và YOLOv8n (FP32) |
| 4 | 28/10 – 03/11 | Lượng tử hóa INT8 (PTQ, QFT), biên dịch HEF, đo trên Pi 5. | Bảng FP32 so với INT8, FPS |
| 5 | 04/11 – 10/11 | Vẽ đường mép nước cho tập test USV-VN; đánh giá zero-shot; ablation (độ phân giải, calibration); thử augmentation/SAM. | Bảng ablation, kết quả domain gap |
| 6 | 11/11 – 17/11 | Hoàn thiện pipeline trên Pi 5; đo độ trễ và công suất; so sánh eWaSR-H với YOLOv8n; quay video demo. | Hệ thống demo, bảng so sánh |
| 7 | 18/11 – 24/11 | Viết bài báo (định dạng IEEE, 6–8 trang). | Bản thảo bài báo |
| 8 | 25/11 – 06/12 | Viết thuyết minh đồ án, làm slide, dự phòng. | Thuyết minh và slide |

## 10. Kết quả và đóng góp dự kiến

1. **eWaSR-H:** biến thể eWaSR tối ưu cho Hailo, chạy thời gian thực trên Raspberry Pi 5 + Hailo-8L.
2. **Phân tích có hệ thống** ảnh hưởng của lượng tử hóa INT8 lên khả năng phát hiện vật cản theo kích thước.
3. **So sánh phân đoạn ngữ nghĩa với phát hiện đối tượng** trên cùng nền tảng nhúng, xét cả độ chính xác, tốc độ và khả năng nhận diện vật cản chưa từng thấy.
4. **Đánh giá trên dữ liệu sông đô thị Việt Nam**, kèm giao thức đánh giá chỉ cần bounding box.
5. **Mã nguồn mở** toàn bộ quy trình: huấn luyện trên Kaggle, biên dịch Hailo, suy luận trên Raspberry Pi 5.
6. Một bài báo gửi hội nghị hoặc tạp chí (dự kiến VCCA, REV-ECIT, ATC, NICS hoặc tạp chí khoa học của trường).

## 11. Hạn chế đã biết

- Không có USV thật, nên hệ thống chỉ được thử nghiệm offline trên dữ liệu đã thu.
- Dữ liệu tự thu chỉ có **một loại vật cản** (chai/can nhựa nổi) tại **một địa điểm**, nên tính đa dạng hạn chế.
- Không còn ảnh gốc: ảnh đã bị resize về 640×640, có thể bị méo tỉ lệ so với cảm biến.
- Không có hiệu chỉnh camera, nên không ước lượng được khoảng cách tuyệt đối và không tính được vùng nguy hiểm trên dữ liệu tự thu.

## 12. Tài liệu tham khảo

1. B. Bovcon, M. Kristan, "WaSR — A Water Segmentation and Refinement Maritime Obstacle Detection Network," *IEEE Transactions on Cybernetics*, 2022.
2. M. Teršek, L. Žust, M. Kristan, "eWaSR — An Embedded-Compute-Ready Maritime Obstacle Detection Network," *Sensors*, 2023.
3. B. Bovcon, J. Muhovič, J. Perš, M. Kristan, "The MaSTr1325 Dataset for Training Deep USV Obstacle Detection Models," *IROS*, 2019.
4. B. Bovcon, J. Muhovič, D. Vranac, D. Mozetič, J. Perš, M. Kristan, "MODS — A USV-Oriented Object Detection and Obstacle Segmentation Benchmark," *IEEE Transactions on Intelligent Transportation Systems*, 2022.
5. L. Žust, J. Perš, M. Kristan, "LaRS: A Diverse Panoptic Maritime Obstacle Detection Dataset and Benchmark," *ICCV*, 2023.
6. G. Jocher, A. Chaurasia, J. Qiu, "Ultralytics YOLOv8," 2023.
7. A. Kirillov et al., "Segment Anything," *ICCV*, 2023.
8. Hailo Technologies, *Hailo Dataflow Compiler User Guide* và *Hailo Model Zoo*.
