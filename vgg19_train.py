# vgg19 훈련

import numpy as np
import pandas as pd
import keras
import cv2
from keras.applications.vgg19 import VGG19
from keras.models import Model
from keras.layers import Dense, Dropout, Flatten
from tqdm import tqdm
from sklearn.model_selection import train_test_split


# 훈련할 csv 데이터 셋팅
df_train = pd.read_csv("./input/labels.csv")

# breed 파싱(중첩제거?)
targets_series = pd.Series(df_train['breed'])
# one_hot 변환(0 0 1 0)
one_hot = pd.get_dummies(targets_series, sparse = True)

# 배열로 변환
one_hot_labels = np.asarray(one_hot)


# 이미지 사이즈 정의
im_size = 90

x_train = []
y_train = []

# 훈련할 데이터 셋팅
i = 0
for f, breed in tqdm(df_train.values):
    img = cv2.imread("./input/train/{}.jpg".format(f))
    label = one_hot_labels[i]
    x_train.append(cv2.resize(img, (im_size, im_size)))
    y_train.append(label)
    i += 1


y_train_raw = np.array(y_train, np.uint8)
x_train_raw = np.array(x_train, np.float32) / 255.

# shape 확인
print(x_train_raw.shape)
print(y_train_raw.shape)

# 분류
num_class = y_train_raw.shape[1]


# 교육할 데이터셋팅 및 성능테스트 데이터 교육데이터에서 30% 할당
X_train, X_valid, Y_train, Y_valid = train_test_split(x_train_raw, y_train_raw, test_size=0.3, random_state=1)

# 사전 교육된 기본 모델 생성
base_model = VGG19(
    weights = 'imagenet',
    # weights = None,
    include_top=False, input_shape=(im_size, im_size, 3))

# model out 정의
x = base_model.output
x = Flatten()(x)
predictions = Dense(num_class, activation='softmax')(x)

# 모델 정의
model = Model(inputs=base_model.input, outputs=predictions)

# 저장된 weights 로드
# model.load_weights('weights.h5')

# 교육할 계층 설정 최상위 계층만 교육 (False 가중치 고정)
for layer in base_model.layers:
    layer.trainable = False

# 훈련할 레이어 확인
for layer in base_model.layers:
    print(layer, layer.trainable)

# 오차역전파 정의
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
callbacks_list = [keras.callbacks.EarlyStopping(monitor='val_acc', patience=3, verbose=1)]
model.summary()


# 학습시작
model.fit(X_train, Y_train, epochs=1, validation_data=(X_valid, Y_valid), verbose=1)

# 학습된 weights 저장
model.save('weights.h5')

