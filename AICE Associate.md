
## AICE Associate 출제 범위

데이터 확보 (14개 문항)
- 데이터 탐색
	- 필요한 라이브러리 설치, Tabular 데이터 가져오기, 데이터의 구성확인
- 데이터 분석
	- 데이터 전처리 (결측치 처리, 라벨 인코딩/원핫 인코딩), xy 데이터 분리, 데이터 정규 분포화, 표준화, 데이터 시각화 (상관 분석 등)
- AI 모델링
	- scikit-learn, Tensorflow 등을 활용하여 문제에 제시된 예측을 위해 머신러닝, 딥러닝 모델링
- AI 모델 평가
	- 모델 성능 평가 및 그래프 출력
AI 모델 활용 / 서비스화

파이썬에서 데이터 처리를 위해 Pandas 데이터 프레임 변경, 삭제
데이터 전처리 주요 기법
결측치 처리 
데이터 타입 변경하기
전체적인 EDA (탐색적 데이터 분석)에 대한 데이터 시각화 (Matplotlib. Seaborn)
파이썬에서 리스트, 딕셔너리, 함수

데이터 분석의 대표적인 Pandas
시각화는 Matplotlib, Seaborn

단일 모델과 앙상블 복합모델이 출제 범위
딥러닝, DNN, FNN 같은 일반 뉴럴 네트워크를 다룰 수 있어야 함


```python
# Linear Regression

from sklearn.linear_model import LinearRegression
model=LinearRegression()
model.fit(X_train, y_train)
pred=model.predict(X_test)
```

---
학습 목표
- AI 개념과 원리를 이해하고 설명할 수 있다.
- 머신러닝 이론을 이해하고 주요 알고리즘을 활용할 수 있다.
- 딥러닝 심층신경망 모델을 구성하고, 파라미터 조정으로 성능을 개선할 수 있다.

과정 특징
- 실무에 필요한 AI 모델링 핵심
- AI Summary 형태로 AI 기술 원리 및 주요 알고리즘 학습
- 실습으로 직접 Python 코딩, AI 모델링 학습

목차 
- AI 개요
- 머신러닝 원리와 주요 모델
- 딥러닝 이론과 심층신경망 모델

---
## AI 개요

인공지능
- 인간의 지적 능력 (추론, 인지)을 구현하는 모든 기술

머신러닝
- 알고리즘으로 데이터를 분석, 학습하여 판단이나 예측을 하는 기술
- 선형 회귀, 로지스틱 회귀, K-최근접 이웃, 결정 트리, 랜덤포레스트, 서포트 벡터머신

>머신러닝의 종류
- 지도 학습 (Supervised Learning) - 정답지로 학습 분류, 예측
- 비지도 학습 (Unsupervised Learning) - 정답없이 학습 군집, 특성 도출
- 강화 학습 (Reinforcement Learning) - 시뮬레이션 반복 학습 성능 강화 등에 활용

딥러닝
- 인공신경망 알고리즘을 활용하는 머신러닝 기술
- 심층 신경망(DNN), 합성곱 신경망 (CNN), 순환 신경망 (RNN), 강화 학습

파이썬의 장점
1. Python the Best programming language in AI
2. Community support
3. Good visualization options
4. Easy Readability
5. Platform indepence
6. Flexibility
7. Easy Coding
8. A rich Python Libraries for AI Projects

파이썬의 주요 패키지
- Numpy: 행렬과 다차원 배열을 쉽게 처리할 수 있게 해주는 패키지
- pandas: 데이터를 처리하고 분석하는 데 효과적인 패키지
- matplotlib: 데이터를 차트나 플롯으로 그려주는 시각화 라이브러리 패키지
- seaborn: matplotlib을 기반으로 다양한 색상 테마와 통계용 차트 등의 기능을 추가한 시각화 패키지
- scikit-learn: 교육 및 실무를 위한 머신러닝 라이브러리 패키지
- Tensorflow: 구글에서 만든 오픈소스 딥러닝 라이러리 패키지

---
## 머신러닝 원리와 주요 모델

과거 (Data - Model - Prediction)
- 사람이 직접 데이터 패턴을 찾아 알고리즘 코딩하여 결과를 얻음

AI/머신러닝 (Data - Model - Prediction)
- 데이터와 결과를 기반으로 스스로 패턴 학습하고 이를 이용하여 예측

### Linear Regression (선형 회귀) 
- 직선을 그어 미래를 예측하는 방법으로 머신러닝 모델 중 하나
- 수많은 직선을 그어서, 데이터를 잘 표현할 수 있는 직선 하나를 구함
	- 
- $y = wx + b$ 는 모델의 알고리즘
	- y는 정답, x는 학습 시간, w는 기울기 (=파라미터), b는 절편
		- 기울기와 절편만 구하면, 가장 잘 맞는 직선을 구할 수 있음
	- 컴퓨터는 w 가중치가 최적인 지 어떻게 알 수 있는가?
		- 이때 Cost Function (비용 함수)을 사용
		- 실제 값과 예측 값의 차이를 Error ( = Loss, Cost)라 칭함
		- Cost Function = (실제값 - 예측값)^2 / N 
			- -> MSE (Mean Squared Error)

>Gradient Descent Algorithm (경사하강법 알고리즘)
- Cost Function을 어떻게 최적화할 것인가?
- 그래프 상에서 비용이 제일 낮은 아래 부분 (기울기 0)을 찾으면 된다
- w를 줄여 나가 Minimum Cost를 구함
```python
from sklearn.linear_model import LinearRegression

model=LinearRegression()
model.fit(X_train, y_train)
pred=model.predict(X_test)
```

### 모델 학습이란?
1. 목표: 최적의 직선 구하기
2. 직선별 손실함수 구하기
3. 손실함수 최소값 구하기
4. Gradient Descent Algorithm 사용

### 지도 학습 (Supervised Learning)
- 데이터와 정답(레이블)을 알려주면서 진행하는 학습
- 데이터와 레이블이 함께 제공됨
- 레이블 = 정답, 실제값, 타깃, 클래스, y
- 예측된 값 = 예측값, $\widehat{y}$ (y hat)

>지도 학습 모델 종류
- 분류 모델 (Classification)
	- 레이블의 값들이 이산적으로 나눠질 수 있는 문제에 사용
- 예측 모델 (Prediction)
	- 레이블의 값들이 연속적인 문제에 사용

>지도 학습 데이터셋 구조
- 각 열 (Column)을 특징/속성(feature) 이라고 함
- 각 행 (Row)을 예제 (Example) 데이터라고 함
- 데이터 열 중에 하나를 선택하여 레이블로 사용함

### 비지도 학습 (Unsupervised Learning)
- 레이블(정답) 없이 진행되는 학습
- 데이터 자체에서 패턴을 찾아내야 할 때 사용함

### 데이터셋 분리
- DataSet = Train dataset + Validation dataset + Test dataset
- 수능 시험 점수 예측 모델
	- Train dataset: 참고서
	- Validation dataset: 모의고사
	- Test dataset: 수능 시험

### 모델 선택
- 데이터의 흐름을 잘 따라가는 모델이 좋은 모델
- 모든 데이터를 지나가는 모델은 과적합 (Overfitting) 
- 데이터를 흐름을 잘 따라가지 못하는 모델은 과소적합 (Underfitting)
	- 학습을 더 진행하여 해결 가능


### Confusion Matrix (5차 행렬, 혼동 행렬)
- 모델의 대략적인 성능 확인과 모델의 성능을 오차 행렬을 기반으로 수치로 표현
- True Positive (TP): True를 True로 예측 (정답)
- True Negetive (TN): False를 False로 예측 (정답)
- False Positive (FP): False를 True로 예측 (오답)
- False Negetive (FN): True를 False로 예측 (오답)

성능 지표
- 학습이 끝난 후 모델을 평가하는 용도로 사용됨

- 정확도 (Accuracy) 
	- 가장 직관적으로 모델의 성능을 나타낼 수 있는 평가 지표
	- (Accuracy) = $TP + TN /TP+FN+FP+TN$
- 정밀도 (Precision)
	- 모델이 True라고 분류한 것 중에서 실제 True인 것의 비율
	- (Precision) = $TP/TP + FP$
- 재현율 (Recall)
	- 실제 True인 것 중에서 모델이 True라고 예측한 것의 비율
	- (Recall) = $TP / TP + FN$
- F1 점수 (F1-score)
	- 정밀도와 재현율의 조화 평균
	- (F1-score) = $2*1/(1/Precision+1/recall)$

### 머신러닝 프로세스

![[Pasted image 20231225003144.png]]


### 머신러닝 주요 알고리즘

scikit-learn
- 가장 인기 있는 머신러닝 패키지, 많은 머신러닝 알고리즘이 내장되어 있음
```python
from sklearn.linear_model import LinearRegression

model=LinearRegression()
print(model)
```

머신러닝 주요 알고리즘 분류
- 회귀: Linear Regression
- 분류: Logistic Regression
- 회귀 분류: Decision Tree, Random Forest, K-Nearest Neighbor


### Logistic Regression
- 이진 분류 규칙은 0과 1의 두 클래스로 갖는 것으로, 일반 선형 회귀 모델을 이진 분류에 사용하기 어려움
- 로지스틱 함수를 사용하여 로지스틱 회귀 곡선으로 변환하여 이진 분류할 수 있음
	- 로지스틱 함수: $\sigma(t) = 1/1+e^{-t}$ 
```python
from sklearn.linear_model import LogisticRegression

model=LogisticRegression()
model.fit(X_train, y_train)
pred=model.predict(X_test)
```

### K-Nearest Neighbor
- 새로운 데이터가 주어졌을 때 기존 데이터 가운데 가장 가까운 k개 이웃의 정보로 새로운 데이터를 예측하는 방법론
```python
from sklearn.neighbors import KNeighborsClassifier

knn=KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
pred=knn.predict(X_test)
```
### Decision Tree
- 분류와 회귀 작업이 가능한 다재다능한 머신러닝 알고리즘
- 복잡한 데이터셋도 학습할 수 있으며 강력한 머신러닝 알고리즘인 랜덤 포레스트의 기본 구성 요소
```python
from sklearn.tree import DecisionTreeClassifier

model=DecisionTreeClassifier(max_depth=2)
model.fit(X_train, y_train)
pred=model.predict(X_test)
```
### Random Forest
- 일련의 예측기 (분류, 회귀 모델)로부터 에측을 수집하면 가장 좋은 모델 하나보다 더 좋은 예측을 얻을 수 있음
- 일련의 예측기 -> 앙상블
- 결정 트리의 앙상블 -> 랜덤 포래스트
```python
from sklearn.ensemble import RandomForestClassifier

model=RandomForestClassifier(n_estimators=50)
model.fit(X_train, y_train)
pred=model.predict(X_test)
```

### 총정리

**기본개념**
- 최적의 선을 긋는 것
- Cost Function (비용 함수)
- Gradient Descent (경사하강법)

**기술 원리**
- 지도 학습 vs 비지도 학습
- 회귀와 분류
- 데이터 확보/전처리 시간 소요
- 모델 성능 지표

**주요 알고리즘**
- Linear Regression
- Logistic Regression
- Decision Tree
- Random Forest

---
## 딥러닝 기본 개념

- 어떻게 강아지와 고양이를 인식해서 분류할 수 있을까?
	- 데이터 내의 학습을 진행함에 따라 특징을 파악함
	- 특징을 기반으로 패턴 및 규칙을 생성함
	- 패턴 및 규칙을 만족 시, 다음을 예측을 진행함
	

### 딥러닝의 목표 (= 머신러닝의 목표)
- 모델에 입력값을 넣었을 때 출력값이 최대한 정답과 일치하게 하는 것
- 예측값과 실제값의 에러를 최소화하는 것 (정답과 일치)을 목표로 함

### 딥러닝 학습 방법
- 딥러닝 모델의 매개변수 (weights, bias)를 무작위로 부여한 후, 반복 학습을 통해 모델의 출력값을 정답과 일치하도록 매개변수를 조금씩 조정함
	- $y = wx + b$, $y - \widehat{y} = error$
- Gradient Descent 최적화 알고리즘을 통해서 cost를 업데이트 하여 최적의 $w$를 구함

### 딥러닝의 기술 원리

Perceptron
- 사람 두뇌에 있는 뉴런을 모델링한 것
- 간단한 함수를 학습할 수 있음

![[Pasted image 20231225010409.png]]


### DNN (Deep Neural Network)
- 입력층과 출력층 사이에 여러 개의 은닉층 (hidden layer)으로 이루어진 인공신경망
	- 신경망 출력에 비선형 활성화 함수를 추가하여 복잡한 비선형 관계를 모델링 할 수 있음

### Activation Function (활성화 함수)
- Binary Step (계단 함수)
- Logistic, sigmoid or soft step
- Hyperbolic tangent (tanh)
- Rectified linear unit (ReLU)
- softmax

### Loss Function
- 신경망 학습의 목적 함수로 출력값(예측값)과 정답 (실제값)의 차이를 계산
- 회귀 모델
	- MSE, MAE
- 분류 모델
	- 이진 분류(Binary cross-entropy), 다중 분류 (Categorical cross-entropy)

### Gradient Descent
- 뉴럴넷이 가중치 파라미터들을 최적화하는 방법
	- 손실 함수를 현 가중치에서 기울기를 구하여 Loss를 줄이는 방향으로 업데이트 해 나감

### Backpropagation
- 실제값과 모델 결과값에서 오차를 구하여 오차를 output에서 input 방향으로 보냄
	- 가중치를 재 업데이트 하면서 학습


### Optimization Algorithm

![[Pasted image 20231225011309.png]]

---









 