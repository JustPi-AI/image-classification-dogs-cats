import os
import numpy as np
from keras.preprocessing import image
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, TimeDistributed, Flatten, Conv2D, MaxPooling2D
from keras.optimizers import Adam
from keras.utils import load_img, img_to_array
import keras.models 

# Định nghĩa thư mục chứa dữ liệu
folder_cats = 'dogs_vs_cats/training_set/cats'  # Thay đổi đường dẫn tới thư mục chứa ảnh Cats
folder_dogs = 'dogs_vs_cats/training_set/dogs'  # Thay đổi đường dẫn tới thư mục chứa ảnh Dogs

# Khởi tạo danh sách ảnh và nhãn
photos = []
labels = []

# Hàm để chuẩn bị dữ liệu từ thư mục
def prepare_data(folder, label):
    for file in os.listdir(folder):
        if os.path.isfile(os.path.join(folder, file)) and file.endswith('.jpg'):
            # Đọc ảnh
            img = load_img(os.path.join(folder, file), target_size=(150, 150))
            # Chuyển đổi thành mảng numpy
            img_array = img_to_array(img)
            # Lưu vào danh sách
            photos.append(img_array)
            labels.append(label)

# Chuẩn bị dữ liệu cho Cats và Dogs
prepare_data(folder_cats, 0)
prepare_data(folder_dogs, 1)

# Chuyển danh sách thành mảng numpy
photos = np.array(photos)
labels = np.array(labels)

# Chuẩn hóa giá trị pixel về khoảng [0, 1]
photos = photos / 255.0

# Xây dựng mô hình CNN để trích xuất đặc trưng từ ảnh
cnn_model = Sequential()
cnn_model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)))
cnn_model.add(MaxPooling2D((2, 2)))
cnn_model.add(Conv2D(64, (3, 3), activation='relu'))
cnn_model.add(MaxPooling2D((2, 2)))
cnn_model.add(Flatten())

# Xây dựng mô hình LSTM
model = Sequential()
model.add(TimeDistributed(cnn_model, input_shape=(1, 150, 150, 3)))  # Thêm một chiều mới đại diện cho số lượng ảnh trong mỗi sequence
model.add(LSTM(128))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

# Biên dịch mô hình
model.compile(loss='binary_crossentropy', optimizer=Adam(), metrics=['accuracy'])
model.summary()

# Huấn luyện mô hình
model.fit(photos[:, np.newaxis], labels, epochs=10, batch_size=32, validation_split=0.2)  # Thêm np.newaxis để thêm chiều mới

# Lưu mô hình đã huấn luyện
model.save('dogs_vs_cats_model.h5')

# Load mô hình đã huấn luyện
loaded_model = keras.models.load_model('dogs_vs_cats_model.h5')

# Đọc và chuẩn bị ảnh mới cần dự đoán
new_photo = load_img('conmeo.jpg', target_size=(200, 200))
new_photo = img_to_array(new_photo)
new_photo = np.expand_dims(new_photo, axis=0) # thêm chiều mới ví dụ 1D lên 2D
new_photo = new_photo / 255.0 # chuẩn hóa giá trị pixel của hình ảnh từ khoảng từ 0 đến 255 xuống trong khoảng từ 0 đến 1.

# Dự đoán nhãn cho ảnh mới
predictions = loaded_model.predict(new_photo)

# In kết quả dự đoán (0: cats, 1: dogs)
print(predictions)
if predictions[0] < 0.5:
    print("Predicted label: cat")
else:
    print("Predicted label: dog")

