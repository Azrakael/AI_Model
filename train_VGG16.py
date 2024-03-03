#%%
import pandas as pd
import tensorflow as tf
import os
import pickle
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, BatchNormalization, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

# 라벨 데이터 로드
labels_df = pd.read_csv('./processed_images/labels.csv')

# 기본 이미지 경로 설정
base_dir = './dataset'

models_dir = './models'
if not os.path.exists(models_dir):
    os.makedirs(models_dir)

# 훈련, 검증, 테스트 데이터 경로 설정
train_dir = os.path.join(base_dir, 'train')
val_dir = os.path.join(base_dir, 'val')
test_dir = os.path.join(base_dir, 'test')

# 이미지 증강 설정
train_datagen = ImageDataGenerator(rescale=1./255) # 이미지 증강 사용X (4th)

# 검증 데이터 생성기
val_datagen = ImageDataGenerator(rescale=1./255)

# 테스트 데이터 생성기
test_datagen = ImageDataGenerator(rescale=1./255)

# 테스트 데이터 생성기
test_generator = test_datagen.flow_from_dataframe(
    dataframe=labels_df,
    directory=test_dir,
    x_col='Image Name',
    y_col='Label',
    target_size=(224, 224), #없어도 되나 혹시모를 에러방지용도
    batch_size=16,
    class_mode='categorical',
    shuffle=False) 

# 훈련 데이터 생성기
train_generator = train_datagen.flow_from_dataframe(
    dataframe=labels_df,
    directory=train_dir,
    x_col='Image Name',
    y_col='Label',
    target_size=(224, 224), #없어도 되나 혹시모를 에러방지용도
    batch_size=16,
    class_mode='categorical')

# 검증 데이터 생성기
val_generator = val_datagen.flow_from_dataframe(
    dataframe=labels_df,
    directory=val_dir,
    x_col='Image Name',
    y_col='Label',
    target_size=(224, 224), #없어도 되나 혹시모를 에러방지용도
    batch_size=16,
    class_mode='categorical')

# VGG16 모델 구축 및 수정
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
for layer in base_model.layers[:-4]:
    layer.trainable = False

x = Flatten()(base_model.output)
# l2정규화를 이용해 과적합 방지
x = Dense(1024, activation='relu', kernel_regularizer=l2(0.001))(x)  # 4096 -> 1024 (2nd)
x = BatchNormalization()(x)
x = Dropout(0.6)(x) # 드롭아웃 0.5 -> 0.6
predictions = Dense(5, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)

# 조정된 Adam 옵티마이저
adam_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)  # 0.00001 -> 0.0001 (3rd)

# 모델 컴파일
model.compile(optimizer=adam_optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# 콜백 설정
early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='min')  # patience 10 -> 5 (2nd)
model_checkpoint = ModelCheckpoint(
    filepath=os.path.join(models_dir, 'VGG16_BM_4th.keras'),
    save_best_only=True,
    monitor='val_loss',
    mode='min'
)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, min_lr=0.00001, verbose=1) # patience 5 -> 3(2nd)

# 모델 훈련
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.n // train_generator.batch_size,
    epochs=25,  # 50 -> 25 (4th)
    validation_data=val_generator,
    validation_steps=val_generator.n // val_generator.batch_size,
    callbacks=[early_stopping, model_checkpoint, reduce_lr]
)

# 학습이 완료된 후 모델 저장
model.save(os.path.join(models_dir, 'VGG16_4th.keras'))

with open(os.path.join(models_dir, 'VGG16_4th_history.pkl'), 'wb') as file:
    pickle.dump(history.history, file)

# 검증 데이터로 모델 평가
val_loss, val_accuracy = model.evaluate(val_generator, steps=val_generator.n // val_generator.batch_size)
print(f'Validation loss: {val_loss}, Validation accuracy: {val_accuracy}')

# 테스트 데이터로 모델 평가
test_loss, test_accuracy = model.evaluate(test_generator, steps=test_generator.n // test_generator.batch_size)
print(f'Test loss: {test_loss}, Test accuracy: {test_accuracy}')
# %%
# 