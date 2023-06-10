# MachineLearning-With-Python
# OpenCV
MachineLearning

머신러닝 7과정
1.문제정의
2.데이터 수집
3.데이터 전처리
4.탐색적 데이터 분석(EDA)
5.모델 선택 및 하이퍼파라미터 조절
6.모델 학습
7.평가

libraries

pandas // 수학 관련 
matplotlib.pyplot // 그래프
from sklearn.neighbors import KNeigborsClassifier  // KNN분류모델 사용
from sklearn.datasets import load_iris #머신러닝 모델 불러오기 (KNN-> 분류모델)
from sklearn.neighbors import KNeighborsClassifier #정확도 측정도구 불러오기
from sklearn.metrics import accuracy_score

#ML 안에는 데이터를 랜덤으로 섞은다음 훈련용과 테스트용으로 나눠주는 도구가 존재합니다.
from sklearn.model_selection import train_test_split

from sklearn.model_selection import train_test_split # train test 분리도구
from sklearn.tree import DecisionTreeClassifier #DecisionTree모델 사용 




OpenCV!

###openCV(computer vision)

컴퓨터 비전은 시각적부분을 이해하고 처리할 수 있는 인공지능분야

C++ 언어로 제작되어있음

파이썬과 강한결합력

https://opencv.org/ : OpenCV 공식 사이트

https://blog.naver.com/samsjang/220498694383 : 잘 정리된 OpenCV 관련 블로그 (코드 복사 가능)

https://m.blog.naver.com/samsjang/220498694383 : 위의 블로그와 동일한 블로그 (내용 추가)

https://github.com/opencv/opencv : OpenCV github 사이트

!pip install opencv-python opencv-contrib-python

https://learnopencv.com/ : 높은 수준의 결과물을 낼 수 있는 다양한 코드 제공 (프로젝트에 활용)


!pip install opencv-python opencv-contrib-python

#설치확인 및 버전확인
import cv2
print(cv2.__version__)






