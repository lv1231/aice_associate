
# DNN (Deep Neural Network)
- 심층 신경망

## Dropout 
- 과적합 (Overfiting) 방지용, Train 학습 시에만 사용
- dropout=0.3, 30%의 확률로 노드(퍼셉트론)를 없애 복잡도를 줄이게 함

# DNN 구현_라이브러리 임포트

```python
import tensorflow as tf
# DNN 모델을 만들 Sequence 라이브러리 임포트
from tensorflow.keras.models import Sequential
# Dense layer (Hidden layer), Dropout (과적합 방지용) 라이브러리 임포트
from tensorflow.keras.layers import Dense, Dropout
```

```python
model=Sequential()
model.add(Dense(4, input_shape=(3,), activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(4, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
history=model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=40,batch_size=10)
```