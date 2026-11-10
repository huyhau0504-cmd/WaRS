# Ngữ cảnh bài báo (đọc trước khi sửa) — cập nhật 10/11/2026

File này để ai mở repo (Hậu trên Windows, hoặc Claude) nắm đủ bối cảnh mà không cần lịch sử trò chuyện.

## 1. Bài báo là gì

- **Tạp chí:** AUT Journal of Electrical Engineering (EEJ), nộp **file Word** qua eej.aut.ac.ir.
- **Tiêu đề:** *Real-Time Maritime Obstacle Segmentation for Low-Cost Unmanned Surface Vehicles on an Edge Neural Accelerator* (14 từ, không viết tắt — yêu cầu của EEJ).
- **Phạm vi đã chốt (06/11):** chỉ đánh giá trên **MODS**; LaRS là một mục ngắn về cảnh sông. **Đã bỏ** USV-VN, YOLOv8n, đo công suất (không có đồng hồ đo) — chuyển thành hướng phát triển.
- **Đóng góp:**
  - **C1 — eWaSR-Lite:** bỏ MLP của khối 1–5 trong feature mixer (+ cổng depthwise D thay mixer S). −61% tham số, 8 → 4 context trên Hailo-8L, nhanh hơn 59–70% (chip) / 43–55% (đầu-cuối) so với eWaSR cùng công thức, độ chính xác MODS như nhau (3 seed).
  - **C2 — độ bền INT8:** eWaSR tự huấn luyện sập theo bộ ảnh calib (F1 vùng nguy hiểm 37–86%), eWaSR-Lite ổn định (85,1–85,7%); độ trùng pixel INT8–FP32 không phát hiện được lỗi này.
  - **LaRS:** huấn luyện lại trên LaRS giảm báo động sai trên mặt nước sông 19% → 1,6%; E1 lại cần 8 context (14,7 FPS), E7 vẫn 4 (29,1 FPS).
- **Câu reviewer sẽ hỏi — "sao không dùng thẳng trọng số công bố?":** chúng nhanh (5 context, 28,7 FPS) và chính xác hơn ở vùng nguy hiểm, nhưng chỉ vì MLP đã "chết" trong lần huấn luyện gốc. Huấn luyện lại (cần cho mọi vùng hoạt động mới) thì MLP sống → chậm gấp đôi và INT8 dễ sập. **T03 (10/11) cho thấy kể cả dùng đúng công thức đã công bố, MLP vẫn sống** → bỏ bằng thiết kế là cách duy nhất chắc chắn.

## 2. File nào là bản gốc

| File | Vai trò |
|---|---|
| `paper/eej/WaRS_EEJ_manuscript.docx` | **Bản nộp. Sau lần dựng cuối có kết quả T03, đây là BẢN GỐC** — sửa trực tiếp trong Word (MathType, comment của thầy). |
| `paper/eej/manuscript.md` | Nguồn đã dùng để dựng file Word (`tools/build_eej_docx.py`). **Không dựng lại file Word từ đây sau mốc trên** — sẽ xóa mọi chỉnh sửa trong Word. |
| `paper/figs/*.png, *.pdf` | Hình (≥300 PPI). Sinh lại bằng `tools/make_arch_figure.py`, `tools/make_qual_figure.py`, `tools/make_plots.py`. |
| `paper/main.tex`, `paper/refs.bib` | Bản LaTeX kiểu IEEE (bản nháp cũ, không nộp). `paper/archive/`: bản có USV-VN/YOLO. |
| `docs/data/*.csv, *.json` | **Mọi con số trong bài** đều lấy từ đây (xem `docs/data/README_hailo.md`, `mods_results.csv`, `hailo_calib_seeds.csv`, `e2e_mods.csv`, `lars_val_results.csv`, `layer_scale_seeds.csv`). |

**Đồng bộ Linux ↔ Windows:** repo riêng tư `huyhau0504-cmd/WaRS`, nhánh **`task/T10-mods-paper`**, qua GitHub Desktop. Trước khi sửa: *Fetch origin → Pull*. Sửa xong: *Commit → Push*. File Word không gộp được hai bản sửa song song — **chỉ một người sửa tại một thời điểm**.

## 3. Yêu cầu định dạng của EEJ (từ file mẫu)

- A4, 1 cột, Times New Roman, **giãn dòng đôi**, căn đều; **7–20 trang**.
- Tiêu đề ≤ 15 từ, không viết tắt. Tác giả: học vị, khoa, trường, thành phố, quốc gia; tác giả liên hệ đánh `*` (nên là thầy).
- Tóm tắt **200–250 từ** (hiện 249), không phương trình, không trích dẫn, không viết tắt chưa định nghĩa; **đúng 5 từ khóa**.
- Mục đánh số `1-`, `2-1-`, `2-1-1-` (tối đa 3 cấp). Có mục **Nomenclature** trước References.
- **Phương trình gõ bằng MathType** (hiện 6 phương trình là ảnh tạm — phải gõ lại).
- Chú thích **bảng ở trên**, **hình ở dưới** ("Fig. 1."); bảng không được là ảnh; trục đồ thị ghi tên + đơn vị.
- Tài liệu tham khảo `[n]` theo thứ tự xuất hiện, định dạng như bài mẫu: `[1] H. Hesabi, …, Title, Journal, 219 (2022) 108191.`
- Sau khi được nhận: gửi hình riêng (TIFF/JPEG ≥300 PPI). Khi sửa theo phản biện: tô vàng chỗ sửa.

## 4. Còn phải làm

- [ ] Điền `[TBD]` (chữ đỏ nền vàng): tác giả, đơn vị, email liên hệ, lời cảm ơn.
- [ ] Gõ lại 6 phương trình bằng MathType.
- [ ] Kiểm tra **số trang** (ước tính ~20; nếu vượt thì rút Related Work/Discussion).
- [ ] Thầy rà câu "chưa có nghiên cứu về độ nhạy với bộ calib cho phân đoạn mặt nước" (mục 2-2).
- [ ] Bổ sung số trang cho 2 tài liệu: MaSTr1325 (IROS 2019), MODS (IEEE T-ITS 2022).
- [ ] Điền link code: **https://github.com/huyhau0504-cmd/ewasr-lite** (repo công khai, đã có Release v1.0 kèm trọng số).
- [ ] Kết quả MODS của T03 (đang chấm 10/11) → câu `[TBD: MODS danger-zone F1 …]` trong mục "Training recipe"; số context của E1 công thức gốc (đang compile).
- [ ] Khai báo việc dùng công cụ AI nếu quy định của EEJ yêu cầu.

## 5. Số liệu chính (để đối chiếu khi sửa câu chữ)

Mã mô hình: **E1** = eWaSR (`ewasr`), **E2** = eWaSR + D mixer (`ewasr_h`), **E6** = eWaSR-Lite S mixer (`ewasr_lite`), **E7** = **eWaSR-Lite** (`ewasr_h_lite`, mô hình chính).

| | E1 | E7 | Trọng số công bố |
|---|---|---|---|
| Tham số | 60,3 M | 23,4 M | 60,3 M |
| Context / FPS chip / độ trễ | 8 (7 với bộ calib khác) / 17,5 / 56,1 ms | 4 / 29,7 / 31,0 ms | 5 / 28,7 / 34,2 ms (bộ calib 2: 4 / 23,2) |
| FPS đầu-cuối (khung MODS 1278×958) | 14,6 | 22,7 | 21,8 |
| MODS FP32: F1 / vùng nguy hiểm (3 seed) | 91,8±0,7 / 85,3±0,5 | 92,1±0,3 / 85,5±1,1 | 92,5 / 90,3 |
| MODS INT8 trên Pi, vùng nguy hiểm (3 bộ calib) | 58,5 (37,3–86,0); bật eq: 67,7 (31,6–85,8) | 85,5 (85,1–85,7) | 90,3 (90,3–90,4) |
| LaRS sông: nước bị báo vật cản (huấn luyện MaSTr → LaRS) | 21,9% → 1,8% | 19,2% → 1,6% | 17,3% (chỉ MaSTr) |
| Huấn luyện trên LaRS: context / FPS chip / đầu-cuối | 8 / 14,7 / 12,8 | 4 / 29,1 / 22,3 | — |

Khoảng cách ~5 điểm F1 vùng nguy hiểm giữa mô hình tự huấn luyện và trọng số công bố là do **công thức huấn luyện** (ảnh hưởng E1 và E7 như nhau; nạp trọng số công bố vào kiến trúc Lite cho đúng 90,3).

## 6. Công cụ (chạy trên Linux)

```bash
third_party/.venv-docx/bin/python tools/build_eej_docx.py     # md → docx (CHỈ trước mốc "Word là bản gốc")
third_party/.venv-mods/bin/python tools/make_plots.py         # Fig. 2–4 (γ₂, đánh đổi độ chính xác–tốc độ, bộ calib)
python3 tools/mods_table.py                                    # gộp kết quả MODS → docs/data/mods_results.csv
```
