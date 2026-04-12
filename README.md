# Sign Language Detector - Olympic Trí tuệ Nhân tạo Sinh viên 2025

Dự án này được thực hiện dựa trên đề thi "Olympic Trí tuệ Nhân tạo Sinh viên 2025 - Vòng loại - Tác vụ Nhận diện ngôn ngữ ký hiệu". Hệ thống tập trung vào việc nhận diện và phân loại ngôn ngữ ký hiệu từ video thành 100 lớp khác nhau.

## Nguồn gốc và Mục đích
* Đề bài: Olympic Trí tuệ Nhân tạo Sinh viên 2025 (Vòng loại) - Tác vụ: Nhận diện ngôn ngữ ký hiệu.
* Nền tảng học tập: Sử dụng hệ thống do Học viện Công nghệ Bưu chính Viễn thông (PTIT) triển khai cho mục đích học tập và nghiên cứu. Truy cập tại [đây](https://aichallenge.ptit.edu.vn/competitions/77/?fbclid=IwY2xjawQu29NleHRuA2FlbQIxMABicmlkETFtRTRWODJiTEdHSUlmQ3Vwc3J0YwZhcHBfaWQQMjIyMDM5MTc4ODIwMDg5MgABHnKzDG_oRblcbm0lWlFRgEviwCRQtSuZWSTd3BKF89s4ADqGzCnWMtCOV7DJ_aem_KtSa0oOkAQH0eD1uepoLWQ#/phases-tab)
* Dữ liệu và đánh giá: Đánh giá hiệu suất của mô hình thông qua Nền tảng học tập được đề cập ở trên. Yêu cầu chi tiết của tác vụ, Dataset và Baseline cũng được cung cấp cùng với nền tảng này.

## Cấu trúc Mô hình (CRNN)
* Mô hình kết hợp giữa Mạng Nơ-ron Tích chập (CNN) và Mạng Nơ-ron Hồi quy (RNN).
* Backbone: Sử dụng ResNet50 với bộ trọng số pre-trained trên tập ImageNet để trích xuất đặc trưng không gian từ từng khung hình.
* Sequence Learning: Sử dụng LSTM với kích thước hidden state là 256 và dropout 0.3 để học các đặc trưng chuỗi thời gian.
* Classifier: Lớp Linear cuối cùng ánh xạ kết quả đầu ra thành 100 lớp ngôn ngữ ký hiệu.

## Dữ liệu và Tiền xử lý
* Định dạng đầu vào: Đọc file video (mp4, avi, mov, mkv) thông qua OpenCV.
* Chuẩn hóa khung hình: Lấy mẫu cố định 16 khung hình/video bằng kỹ thuật padding hoặc linspace. Dữ liệu được chuẩn hóa theo thông số của ImageNet.
* Tăng cường dữ liệu (Video Augmentation): Áp dụng thay đổi tốc độ video (0.9x - 1.1x), cắt ảnh ngẫu nhiên, và điều chỉnh độ sáng/tương phản/bão hòa.
* Xử lý mất cân bằng: Sử dụng `WeightedRandomSampler` để kiểm soát tỷ lệ lấy mẫu giữa các lớp đa số và thiểu số (giới hạn oversample tối đa 10 lần).

## Chi tiết Huấn luyện
* Framework: PyTorch.
* Optimizer: AdamW (LR: 1e-4, Weight Decay: 1e-4).
* Scheduler: `ReduceLROnPlateau` với patience là 3 epochs.
* Loss Function: Cross Entropy Loss kết hợp Label Smoothing 0.1.
* Kỹ thuật tối ưu phần cứng: Sử dụng Gradient Accumulation (Batch Size = 8, Steps = 4) để đạt Effective Batch Size là 32, giúp tối ưu hóa việc sử dụng VRAM (15GB) khi chạy mô hình ResNet50.
* Số lượng Epochs: 10 epochs.
* Theo dõi thực nghiệm: Sử dụng Weights & Biases (W&B) để quản lý log và metrics. Link W&B dẫn tới dự án tại [đây](https://wandb.ai/phucduong0308-ho-chi-minh-city-university-of-technology/video-classification-100/workspace?nw=nwuserphucduong0308)

## Kết quả Đầu ra
* `best_model_resnet50.pth`: Trọng số mô hình tối ưu nhất.
* `public_submission.csv` / `.zip`: Kết quả dự đoán cho tập public test.
* `private_submission.csv` / `.zip`: Kết quả dự đoán cho tập private test.