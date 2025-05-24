# Dogs vs Cats Classifier with MobileNetV2

## Mô tả
Dự án này sử dụng mô hình MobileNetV2 đã được huấn luyện trước (pretrained on ImageNet) để phân loại ảnh chó và mèo. Dữ liệu đầu vào được tăng cường bằng các kỹ thuật augmentation như xoay, lật, zoom,... giúp cải thiện hiệu quả huấn luyện.

## Cấu trúc thư mục
```

dogs\_vs\_cats/
│
├── training\_set/
│   ├── cats/
│   └── dogs/
│
├── training.py           # Code huấn luyện mô hình
├── predict.py            # Code dự đoán ảnh mới
├── mobilenetv2\_dogs\_vs\_cats.keras  # Mô hình đã lưu (nếu có)
└── README.md

````

## Yêu cầu
- Python 3.8+
- TensorFlow 2.x
- Các thư viện khác: numpy, matplotlib (nếu dùng)

Cài đặt thư viện:
```bash
pip install tensorflow numpy matplotlib
````

## Cách chạy

### Huấn luyện mô hình

```bash
python training.py
```

* Đảm bảo dữ liệu ảnh chó và mèo đã được đặt đúng trong `dogs_vs_cats/training_set/cats` và `dogs_vs_cats/training_set/dogs`.
* Mô hình sẽ được lưu trong file `mobilenetv2_dogs_vs_cats.keras` sau khi huấn luyện xong.

### Dự đoán ảnh mới

```bash
python predict.py --image path/to/image.jpg
```

* Sử dụng mô hình đã lưu để dự đoán ảnh chó hoặc mèo mới.

## Ghi chú

* Mô hình sử dụng MobileNetV2 pretrained với phần đầu đóng băng, có thể mở đào tạo thêm bằng cách set `base_model.trainable = True` sau khi huấn luyện bước đầu.
* Điều chỉnh các tham số như learning rate, batch size, số epoch theo nhu cầu.

## License

MIT License

---
