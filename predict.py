import numpy as np
from keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array

# Load mô hình đã train
model = load_model('mobilenetv2_dogs_vs_cats.h5')

# Hàm dự đoán ảnh mới
def predict_image(image_path):
    img = load_img(image_path, target_size=(160, 160))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # thêm batch dimension

    prediction = model.predict(img_array)[0][0]

    print(f"Dự đoán xác suất: {prediction:.4f}")
    if prediction > 0.5:
        print("Predicted label: dog 🐶")
    else:
        print("Predicted label: cat 🐱")

if __name__ == '__main__':
    # Thay đường dẫn ảnh bạn muốn dự đoán
    test_image_path = 'concho2.jpg'
    predict_image(test_image_path)
