# Bài kiểm tra Trí tuệ nhân tạo

- Sinh viên: Trần Thủ Đô
- Mã sinh viên: 26DCN020
- Đề tài: Dự đoán nguy cơ bệnh tiểu đường từ các thông số sức khỏe

## Chạy chương trình

Yêu cầu Python 3.10 trở lên. Mở Terminal tại thư mục này và chạy:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
streamlit run app_du_doan_tieu_duong.py
```

Trên Windows, kích hoạt môi trường bằng `.venv\Scripts\activate`.

## Quy trình thực hiện

1. Đọc `CSDL_TieuDuong.csv`; chuẩn hóa nhãn nguy cơ về 0/1 và giới tính về Nam/Nữ.
2. Chuyển các trường số về dạng số, sửa giá trị HbA1C bị lệch 1.000 lần (ví dụ 8025 thành 8.025), đưa giá trị ngoài phạm vi hợp lý thành thiếu và loại dòng trùng.
3. Chia dữ liệu có phân tầng: 70% train, 15% validation, 15% test (random state 42). Dữ liệu thiếu được điền trong pipeline. Giới hạn ngoại lai IQR, bộ điền thiếu, bộ mã hóa và bộ chuẩn hóa đều được khớp từ dữ liệu train để tránh rò rỉ dữ liệu.
4. Huấn luyện Logistic Regression, Random Forest và SVM. Chọn mô hình có F1-score validation cao nhất.
5. Đánh giá mô hình được chọn trên tập test bằng Accuracy, Recall, F1-score và ma trận nhầm lẫn.
6. Giao diện cho phép nhập thông tin sức khỏe, xem điểm nguy cơ, nhóm dự đoán và các chỉ số tham khảo cần lưu ý.

## Lưu ý

Đây là bài tập học thuật, không phải thiết bị y tế. Điểm nguy cơ và các mốc tham khảo không thay thế chẩn đoán. Đơn vị và điều kiện xét nghiệm cần được xác nhận bởi nhân viên y tế.