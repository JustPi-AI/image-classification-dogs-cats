import tensorflow as tf
from keras.src.legacy.preprocessing.image import ImageDataGenerator
from keras.applications import MobileNetV2
from keras.models import Sequential
from keras.layers import Dense, GlobalAveragePooling2D, Dropout
from keras.optimizers import Adam

# Đường dẫn tới thư mục train
train_dir = 'dogs_vs_cats/training_set'

# Tạo ImageDataGenerator với augment dữ liệu cho training
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,  # tách 20% làm validation
    horizontal_flip=True,
    rotation_range=20,
    zoom_range=0.2,
    shear_range=0.2,
)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(160, 160),
    batch_size=32,
    class_mode='binary',
    subset='training',
    shuffle=True
)

val_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(160, 160),
    batch_size=32,
    class_mode='binary',
    subset='validation',
    shuffle=False
)

# Tải MobileNetV2 pretrained (ImageNet), không lấy lớp FC cuối
base_model = MobileNetV2(input_shape=(160, 160, 3),
                         include_top=False,
                         weights='imagenet')

# Đóng băng base_model trong giai đoạn train đầu tiên
base_model.trainable = False

# Xây dựng mô hình
model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer=Adam(learning_rate=0.0001),
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.summary()

# Train model
history = model.fit(
    train_generator,
    epochs=10,
    validation_data=val_generator
)

# Lưu model
model.save('mobilenetv2_dogs_vs_cats.h5')
