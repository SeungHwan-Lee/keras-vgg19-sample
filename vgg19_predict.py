# vgg19 훈련된 신경망 데이터 예측

import numpy as np
import pandas as pd
import cv2
from keras.applications.vgg19 import VGG19
from keras.models import Model
from keras.layers import Dense, Dropout, Flatten
from tqdm import tqdm

# csv 데이터 셋팅
df_train = pd.read_csv("./input/labels.csv")
df_test = pd.read_csv("./input/predict.csv")

# breed 파싱(중첩제거?)
targets_series = pd.Series(df_train['breed'])
# one_hot 변환(0 0 1 0)
one_hot = pd.get_dummies(targets_series, sparse = True)

# 배열로 변환
one_hot_labels = np.asarray(one_hot)

# 이미지 사이즈 정의
im_size = 90

y_train = []
x_test = []

# 예측할 분류 셋팅
i = 0
for f, breed in tqdm(df_train.values):
    label = one_hot_labels[i]
    y_train.append(label)
    i += 1

# 예측할 데이터 셋팅
for f in tqdm(df_test['id'].values):
    img = cv2.imread("./input/test/{}.jpg".format(f))
    x_test.append(cv2.resize(img, (im_size, im_size)))

y_train_raw = np.array(y_train, np.uint8)
x_test = np.array(x_test, np.float32) / 255.


# shape 확인
print(y_train_raw.shape)
print(x_test.shape)

# 분류
num_class = y_train_raw.shape[1]

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
model.load_weights('weights.h5')
model.summary()

# 예측할 데이터 결과
preds = model.predict(x_test, verbose=1)

sub = pd.DataFrame(preds)
# 열 이름을 앞서 한번의 키 인코딩으로 생성된 열 이름으로 설정
col_names = one_hot.columns.values
sub.columns = col_names

sub.head(5)
print(sub)
# 데이터 프레임 시작 부분의 predict에서 열 ID삽입
sub.insert(0, 'id', df_test['id'])

# 테스트 이미지 예측값 저장
sub.to_csv("./input/predict.csv", mode='w')