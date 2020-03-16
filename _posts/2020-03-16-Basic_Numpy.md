---
title: "Basic Numpy and Data Science"
date : 2020-03-16
tags : [data science, numpy, python]
header:
  #image: "/images/fig-1-33.png"
excerpt: "Data Science, Numpy, Python"
mathjax: "true"
toc: true
toc_label: "My Table of Contents"
toc_icon: "cog"
---
{% include toc %}

## 넘파이 ndarray 개요


```python
import numpy as np

# 1차원
array1 = np.array([1,2,3])
print('array1 type:', type(array1))
print('array1 array 형태:', array1.shape)

# 2차원
array2 = np.array([[1, 2, 3],
                  [2, 3, 4]])
print('array2 type:', type(array2))
print('array2 array 형태:', array2.shape)

# 2차원
array3 = np.array([[1,2,3]])
print('array3 type:', type(array3))
print('array3 array 형태:', array3.shape)
```

    array1 type: <class 'numpy.ndarray'>
    array1 array 형태: (3,)
    array2 type: <class 'numpy.ndarray'>
    array2 array 형태: (2, 3)
    array3 type: <class 'numpy.ndarray'>
    array3 array 형태: (1, 3)



```python
# 각 array의 차원 ndarray.ndim
print('array1: {:0}차원, array2: {:1}차원, array3: {:2}차원'.format(array1.ndim, array2.ndim, array3.ndim))
```

    array1: 1차원, array2: 2차원, array3:  2차원


## ndarray의 데이터 타입


```python
list1 = [1,2,3]
print(type(list1))
```

    <class 'list'>


array1 내부 원소들의 데이터 타입은 dtype 속성으로 확인할 수 있다.


```python
array1 = np.array(list1)
print(type(array1))
print(array1, array1.dtype)
```

    <class 'numpy.ndarray'>
    [1 2 3] int64


* ndarray 데이터 값이 모두 같은 데이터 타입이다.
* 섞인 경우 데이터 타입이 더 큰 데이터 타입으로 변환되어 출력한다. (int < U21, int64 < float64)


```python
list2 = [1,2,'test']
array2 = np.array(list2)
print(array2, array2.dtype)
```

    ['1' '2' 'test'] <U21



```python
list3 = [1,2,3.0]
array3 = np.array(list3)
print(array3, array3.dtype)
```

    [1. 2. 3.] float64


* astype(): ndarray 내 데이터 값의 타입 변경 (float -> int로 메모리 절약)


```python
# int32 -> float64
array_int = np.array([1,2,3])
array_float = array_int.astype('float64')
print(array_float, array_float.dtype)
```

    [1. 2. 3.] float64



```python
# float64 -> int32
array_int1 = array_float.astype('int32')
print(array_int1, array_int1.dtype)
```

    [1 2 3] int32



```python
# int32로 변환할 때 소수점 이하는 제거된다.
array_float1 = np.array([1.1, 2.1, 3.1])
array_int2 = array_float1.astype('int32')
print(array_int2, array_int2.dtype)
```

    [1 2 3] int32


## ndarray 편리하게 생성하기 : arange, zeros, ones


```python
# arange() : array를 range()로 표현, 순차적으로 ndarray 데이터 값으로 변환
sequence_array = np.arange(10) # 0부터 함수 인자값 10에서 -1을 한 값인 9까지 값을 순차적으로 나열한다.
print(sequence_array)
print(sequence_array.dtype, sequence_array.shape)
```

    [0 1 2 3 4 5 6 7 8 9]
    int64 (10,)



```python
# start 값도 부여하면 0부터가 아닌 배열로 만들 수 있다.
sequence_array1 = np.arange(5,10) # start = 5부터 stop= 10에서 -1을 한 값인 9까지 값을 순차적으로 나열한다.
print(sequence_array1)
print(sequence_array1.dtype, sequence_array1.shape)
```

    [5 6 7 8 9]
    int64 (5,)



```python
# zeros() : 튜플 shape값을 인자로 주면 모든 값을 0으로 채운 ndarray 반환
zero_array = np.zeros((3, 2), dtype='int32')
print(zero_array)
print(zero_array.dtype, zero_array.shape)
```

    [[0 0]
     [0 0]
     [0 0]]
    int32 (3, 2)



```python
# ones() : 모든 값을 1로 채운 ndarray 반환
one_array = np.ones((3,2))
print(one_array)
print(one_array.dtype, one_array.shape)
```

    [[1. 1.]
     [1. 1.]
     [1. 1.]]
    float64 (3, 2)


## ndarray의 차원과 크기 변경하는 reshape()


```python
array1 = np.arange(10)
print('array1:\n', array1)
```

    array1:
     [0 1 2 3 4 5 6 7 8 9]



```python
array2 = array1.reshape(2,5)
print('array2:\n', array2)
```

    array2:
     [[0 1 2 3 4]
     [5 6 7 8 9]]



```python
array3 = array1.reshape(5,2)
print('array3:\n', array3)
```

    array3:
     [[0 1]
     [2 3]
     [4 5]
     [6 7]
     [8 9]]



```python
# reshape은 지정된 사이즈로 변경 불가능시 에러 발생
array1.reshape(4,3)
```


    ---------------------------------------------------------------------------

    ValueError                                Traceback (most recent call last)

    <ipython-input-18-74cbe2c66603> in <module>
          1 # reshape은 지정된 사이즈로 변경 불가능시 에러 발생
    ----> 2 array1.reshape(4,3)


    ValueError: cannot reshape array of size 10 into shape (4,3)



```python
# reshape() 인자로 -1을 사용 : ndarray와 호환되는 새로운 shape으로 자동 변환
array1 = np.arange(10)
print(array1)

# 고정된 5개 칼럼에 맞는 로우를 자동 생성해 변환
array2 = array1.reshape(-1,5)
print('array2 shape:', array2.shape)

# 고정된 5개 로우에 맞는 칼럼을 자동 생성해 변환
array3 = array1.reshape(5,-1)
print('array3 shape:', array3.shape)
```

    [0 1 2 3 4 5 6 7 8 9]
    array2 shape: (2, 5)
    array3 shape: (5, 2)



```python
# reshape(-1 시리즈)에서도 변환이 안되는 경우는 해결 안된다.
array4 = array1.reshape(4,-1)
```


    ---------------------------------------------------------------------------

    ValueError                                Traceback (most recent call last)

    <ipython-input-20-6eb1a7045f5f> in <module>
          1 # reshape(-1 시리즈)에서도 변환이 안되는 경우는 해결 안된다.
    ----> 2 array4 = array1.reshape(4,-1)


    ValueError: cannot reshape array of size 10 into shape (4,newaxis)



```python
array1 = np.arange(8)
array3d = array1.reshape((2,2,2))
print('array3d:\n', array3d.tolist())
```

    array3d:
     [[[0, 1], [2, 3]], [[4, 5], [6, 7]]]



```python
# 3차원 배열을 2차원 배열로 변환(넘파이배열)
array5 = array3d.reshape(-1,1)
print('array5:\n', array5.tolist())
print('array5 shape:', array5.shape)
```

    array5:
     [[0], [1], [2], [3], [4], [5], [6], [7]]
    array5 shape: (8, 1)



```python
# 1차원 배열을 2차원 배열로 변환
array6 = array1.reshape(-1, 1)
print('array6:\n', array6.tolist())
print('array6 shape:', array6.shape)
```

    array6:
     [[0], [1], [2], [3], [4], [5], [6], [7]]
    array6 shape: (8, 1)


## 넘파이 ndarray의 데이터 세트 선택하기 - 인덱싱(indexing)
1. 특정한 데이터만 추출
2. 슬라이싱(Slicing)
3. 팬시 인덱싱(Fancy Indexing)
4. 불린 인덱싱(Boolean Indexing)

### 단일 값 추출


```python
# 단일 값 추출
# 1부터 9까지의 1차원 ndarray 생성
array1 = np.arange(start=1, stop=10)
print('array1:', array1)
```

    array1: [1 2 3 4 5 6 7 8 9]



```python
# index 0부터 시작하므로 array1[2]는 실제로 3번째 인덱스 위치 데이터 값 의미
value = array1[2]
print('value:', value)
print(type(value))
```

    value: 3
    <class 'numpy.int64'>



```python
print('맨 뒤의 값', array1[-1], '맨 뒤에서 두 번째 값:', array1[-2])
```

    맨 뒤의 값 9 맨 뒤에서 두 번째 값: 8



```python
# 단일 인덱스로 ndarray 내 데이터 값 간단히 수정 가능
array1[0] = 9 # 9로 수정
array1[8] = 0 # 마지막 인덱싱 0 수정
print('array1:', array1)
```

    array1: [9 2 3 4 5 6 7 8 0]



```python
# 1차원을 2차원 3x3 ndarray 변환 후 [row, col]로 2차원 ndarray에서 데이터 추출
array1d = np.arange(start=1, stop=10)
array2d = array1d.reshape(3,3)
print(array2d)
```

    [[1 2 3]
     [4 5 6]
     [7 8 9]]



```python
print('(row=0, col=0) index 가리키는 값:', array2d[0, 0])
print('(row=0, col=1) index 가리키는 값:', array2d[0, 1])
print('(row=1, col=0) index 가리키는 값:', array2d[1, 0])
print('(row=2, col=2) index 가리키는 값:', array2d[2, 2])
```

    (row=0, col=0) index 가리키는 값: 1
    (row=0, col=1) index 가리키는 값: 2
    (row=1, col=0) index 가리키는 값: 4
    (row=2, col=2) index 가리키는 값: 9


* axis 0 = 0 : row 방향의 축 (행), 생략시 axis 0 디폴트 값
* axis 1 = 1 : column 방향의 축 (열)
* axis 2 = 2 : 높이 방향의 축 (높이)

### 슬라이싱


```python
# 슬라이싱 ':' 사이 시작 인덱스에서 종료 인덱스 -1 위치에 있는 데이터 ndarray 반환
array1 = np.arange(start=1, stop=10)
array3 = array1[0:3] # 1 to 3 까지
print(array3)
print(type(array3))
```

    [1 2 3]
    <class 'numpy.ndarray'>


1. : 기호 앞 시작 인덱스 생략시 자동으로 맨 처음 인덱스 0으로 설정
2. : 기호 뒤에 종료 인덱스 생략시 자동으로 맨 마지막 인덱스 설정
3. : 기호 앞/뒤에 시작/종료 인덱스 생략시 자동으로 맨 처음/마지막 인덱스 설정


```python
array1 = np.arange(start=1, stop=10)
array4 = array1[:3] # 0부터 2까지 인덱스
print(array4)
```

    [1 2 3]



```python
array5 = array1[3:] #4부터 끝까지
print(array5)
```

    [4 5 6 7 8 9]



```python
array6 = array1[:] # 전체
print(array6)
```

    [1 2 3 4 5 6 7 8 9]



```python
# 2차원 ndarray 슬라이싱 콤마(,)로 로우, 칼럼 인덱스 저장
array1d = np.arange(start=1, stop=10)
array2d = array1d.reshape(3, 3) # 3*3 2dim ndarray return
print('array2d:\n', array2d)
```

    array2d:
     [[1 2 3]
     [4 5 6]
     [7 8 9]]



```python
print('array2d[0:2, 0:2] \n', array2d[0:2, 0:2])
```

    array2d[0:2, 0:2]
     [[1 2]
     [4 5]]



```python
print('array2d[1:3, 0:3] \n', array2d[1:3, 0:3])
```

    array2d[1:3, 0:3]
     [[4 5 6]
     [7 8 9]]



```python
print('array2d[1:3, :] \n', array2d[1:3, :])
```

    array2d[1:3, :]
     [[4 5 6]
     [7 8 9]]



```python
print('array2d[:, :] \n', array2d[:, :])
```

    array2d[:, :]
     [[1 2 3]
     [4 5 6]
     [7 8 9]]



```python
print('array2d[:2, 0] \n', array2d[:2, 0]) # 차원이 줄어드는 것도 가능하다.
```

    array2d[:2, 0]
     [1 4]



```python
# 2차원 ndarray 뒤에 오는 인덱스 제거시 1차원 ndarray 반환
print(array2d[0]) # axis 0 : row 방향의 축을 기준으로 첫번째 행을 반환한다.
```

    [1 2 3]



```python
print(array2d[1])
```

    [4 5 6]



```python
print('array2d[0] shape: ', array2d[0].shape, 'array2d[1] shape', array2d[1].shape)
```

    array2d[0] shape:  (3,) array2d[1] shape (3,)


### 팬시 인덱싱


```python
# 2차원에 팬시 인덱싱 적용해보기 : 리스트나 넘파이배열을 인덱싱으로 넣어 해당 인덱싱에 맞는 데이터 값만 반환
array1d = np.arange(start=1, stop=10)
array2d = array1d.reshape(3, 3)
```


```python
array3 = array2d[[0,1],2] # [0,1]의 행에서 마지막 3번째 칼럼값
print('array2d[[0, 1],2] =>', array3.tolist())
```

    array2d[[0, 1],2] => [3, 6]



```python
array4 = array2d[[0, 1], 0:2]
print('array2d[[0, 1], 0:2] =>', array4.tolist())
```

    array2d[[0, 1], 0:2] => [[1, 2], [4, 5]]



```python
array5 = array2d[[0, 1]] # axis 0 : row 방향이 디폴트(행 반환), ((0, :), (1, :)) 인덱싱 적용
print('array2d[[0, 1]] =>', array5.tolist())
```

    array2d[[0, 1]] => [[1, 2, 3], [4, 5, 6]]


### 불린 인덱싱


```python
# 조건 필터링과 검색 동시 가능
array1d = np.arange(start=1, stop=10)
```


```python
# [] 안에 array1d > 5 Boolean indexing 적용
array3 = array1d[array1d > 5] # 인덱싱 내부에 바로 조건 부여 가능
print('array1d > 5 불린 인덱싱 결과 값: ', array3)
```

    array1d > 5 불린 인덱싱 결과 값:  [6 7 8 9]



```python
array1d > 5
```




    array([False, False, False, False, False,  True,  True,  True,  True])



ndarray는 True 값이 있는 위치 인덱스 값으로 자동 변환해 해당 인덱스 위치 데이터만 반환, False 무시


```python
boolean_indexes = np.array([False, False, False,False,False,True,True,True,True])
array3 = array1d[boolean_indexes]
print('불린 인덱스로 필터링 결과 :', array3)
```

    불린 인덱스로 필터링 결과 : [6 7 8 9]



```python
# 일반 인덱스
indexes = np.array([5, 6, 7, 8])
array4 = array1d[indexes]
print('일반 인덱스로 필터링 결과:', array4)
```

    일반 인덱스로 필터링 결과: [6 7 8 9]


반드시 1이 아니라 True 값을 넣어줘야 한다.

## 행렬의 정렬 - sort(), argsort()

### 행렬 정렬


```python
# np.sort, ndarray.sort(원본 파괴)
org_array = np.array([3, 1, 9, 5])
print('원본 행렬:', org_array)
```

    원본 행렬: [3 1 9 5]



```python
# np.sort()로 정렬
sort_array1 = np.sort(org_array)
print('np.sort() 반환 정렬 행렬:', sort_array1)
print('np.sort() 호출 후 원본 행렬:', org_array)
```

    np.sort() 반환 정렬 행렬: [1 3 5 9]
    np.sort() 호출 후 원본 행렬: [3 1 9 5]



```python
# ndarray.sort() 정렬
sort_array2 = org_array.sort() # 이 경우에는 원본이 그대로 들어가고 닷 문법 사용한다.
print('org_array.sort() 호출 후 반환 정렬 행렬:', sort_array2) # None이 나타나는 것을 알 수 있다..
print('org_array.sort() 호출 후 반환 정렬 행렬:', org_array) # 원본이 바뀌어 버린디.
```

    org_array.sort() 호출 후 반환 정렬 행렬: None
    org_array.sort() 호출 후 반환 정렬 행렬: [1 3 5 9]



```python
# np.sort(), ndarray.sort() : 모두 기본적으로 오름차순으로 행렬 내 원소 정렬한다.
# 내림차순은 np.sort()[::-1] 이렇게 사용한다.
sort_array1_desc = np.sort(org_array)[::-1]
print('내림차순으로 정렬: ', sort_array1_desc)
```

    내림차순으로 정렬:  [9 5 3 1]



```python
# 2차원 이상 행렬은 axis 축 값을 설정으로 row, column 방향 정렬 수행이 가능하다.(행 끼리, 열끼리)
array2d = np.array([[8, 12],
                   [7, 1]])

sort_array2d_axis0 = np.sort(array2d, axis=0) # row 방향 기준으로 [7,1],[8,12] 예상
print('로우 방향으로 정렬:\n', sort_array2d_axis0)

sort_array2d_axis1 = np.sort(array2d, axis=1) # column 방향 기준으로 [8,12],[1,7] 예상
print('로우 방향으로 정렬:\n', sort_array2d_axis0)
```

    로우 방향으로 정렬:
     [[ 7  1]
     [ 8 12]]
    로우 방향으로 정렬:
     [[ 7  1]
     [ 8 12]]



```python
# np.argsort(): 정렬된 행렬은 인덱스 반환, argmax 떠올리자.
org_array = np.array([3,1,9,5])
sort_indices = np.argsort(org_array)
print(type(sort_indices))
print('행렬 정렬 시 원본 행렬의 인덱스:', sort_indices) # 놀랍게도 정렬을 알아서 하고 인덱싱 순서를 배열해서 반환한다.
```

    <class 'numpy.ndarray'>
    행렬 정렬 시 원본 행렬의 인덱스: [1 0 3 2]



```python
# 내림 차순 정렬 시 np.argsort()[::-1] 사용
org_array = np.array([3,1,9,5])
sort_indices_desc = np.argsort(org_array)[::-1]
print('행렬 내림차순 정렬 시 원본 행렬의 인덱스:', sort_indices_desc) # 정렬 시 [9, 5, 3, 1] -> [2, 3, 0, 1]
```

    행렬 내림차순 정렬 시 원본 행렬의 인덱스: [2 3 0 1]



```python
# np.argsort() : 넘파이는 메타 데이터를 가질 수 없으므로 데이터 추출시 이 함수를 많이 쓴다.(2개의 ndarray를 활용)
import numpy as np

name_array = np.array(['John', 'Mike', 'Sarah', 'Kate', 'Samuel'])
score_array = np.array([78, 95, 84, 98, 88])

sort_indices_asc = np.argsort(score_array)
print('성적 오름차순 정렬 시 score_array 인덱스:', sort_indices_asc)
print('성적 오름차순 정렬 시 name_array의 이름 출력:', name_array[sort_indices_asc]) # fancy indexing 활용하여 데이터 추출
```

    성적 오름차순 정렬 시 score_array 인덱스: [0 2 4 1 3]
    성적 오름차순 정렬 시 name_array의 이름 출력: ['John' 'Sarah' 'Samuel' 'Mike' 'Kate']


## 선형대수 연산 - 행렬 내적과 전치 행렬 구하기


```python
# 행렬 내적(행렬 곱)
A = np.array( [[1,2,3],
              [4,5,6]])
B = np.array( [[7,8],
              [9, 10],
              [11, 12]])
# np.dot() 으로 내적 값을 구한다.
dot_product = np.dot(A, B)
print('행렬 내적 결과:\n', dot_product)
```

    행렬 내적 결과:
     [[ 58  64]
     [139 154]]



```python
# 전치 행렬(Transpose)
A = np.array([[1, 2],
             [3, 4]])
transpose_mat = np.transpose(A)
print('A의 전치 행렬:\n', transpose_mat)
```

    A의 전치 행렬:
     [[1 3]
     [2 4]]


# 데이터 핸들링 - 판다스

## 판다스 시작 - 파일을 DataFrame으로 로딩, 기본 API


```python
# pandas를 pd로 alias해 임포트 하는 것이 관례이다.
import pandas as pd

titanic_df = pd.read_csv('titanic_train.csv')
titanic_df.head(3) # 3개의 로우 반환, 디폴트는 5개이다.
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>Braund, Mr....</td>
      <td>male</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>A/5 21171</td>
      <td>7.2500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>Cumings, Mr...</td>
      <td>female</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>PC 17599</td>
      <td>71.2833</td>
      <td>C85</td>
      <td>C</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>Heikkinen, ...</td>
      <td>female</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>STON/O2. 31...</td>
      <td>7.9250</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
  </tbody>
</table>
</div>




```python
print('titanic 변수 type:', type(titanic_df))
titanic_df
```

    titanic 변수 type: <class 'pandas.core.frame.DataFrame'>





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>Braund, Mr. Owen Harris</td>
      <td>male</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>A/5 21171</td>
      <td>7.2500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>
      <td>female</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>PC 17599</td>
      <td>71.2833</td>
      <td>C85</td>
      <td>C</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>Heikkinen, Miss. Laina</td>
      <td>female</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>STON/O2. 3101282</td>
      <td>7.9250</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>
      <td>female</td>
      <td>35.0</td>
      <td>1</td>
      <td>0</td>
      <td>113803</td>
      <td>53.1000</td>
      <td>C123</td>
      <td>S</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>0</td>
      <td>3</td>
      <td>Allen, Mr. William Henry</td>
      <td>male</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>373450</td>
      <td>8.0500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>5</th>
      <td>6</td>
      <td>0</td>
      <td>3</td>
      <td>Moran, Mr. James</td>
      <td>male</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>330877</td>
      <td>8.4583</td>
      <td>NaN</td>
      <td>Q</td>
    </tr>
    <tr>
      <th>6</th>
      <td>7</td>
      <td>0</td>
      <td>1</td>
      <td>McCarthy, Mr. Timothy J</td>
      <td>male</td>
      <td>54.0</td>
      <td>0</td>
      <td>0</td>
      <td>17463</td>
      <td>51.8625</td>
      <td>E46</td>
      <td>S</td>
    </tr>
    <tr>
      <th>7</th>
      <td>8</td>
      <td>0</td>
      <td>3</td>
      <td>Palsson, Master. Gosta Leonard</td>
      <td>male</td>
      <td>2.0</td>
      <td>3</td>
      <td>1</td>
      <td>349909</td>
      <td>21.0750</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>8</th>
      <td>9</td>
      <td>1</td>
      <td>3</td>
      <td>Johnson, Mrs. Oscar W (Elisabeth Vilhelmina Berg)</td>
      <td>female</td>
      <td>27.0</td>
      <td>0</td>
      <td>2</td>
      <td>347742</td>
      <td>11.1333</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>9</th>
      <td>10</td>
      <td>1</td>
      <td>2</td>
      <td>Nasser, Mrs. Nicholas (Adele Achem)</td>
      <td>female</td>
      <td>14.0</td>
      <td>1</td>
      <td>0</td>
      <td>237736</td>
      <td>30.0708</td>
      <td>NaN</td>
      <td>C</td>
    </tr>
    <tr>
      <th>10</th>
      <td>11</td>
      <td>1</td>
      <td>3</td>
      <td>Sandstrom, Miss. Marguerite Rut</td>
      <td>female</td>
      <td>4.0</td>
      <td>1</td>
      <td>1</td>
      <td>PP 9549</td>
      <td>16.7000</td>
      <td>G6</td>
      <td>S</td>
    </tr>
    <tr>
      <th>11</th>
      <td>12</td>
      <td>1</td>
      <td>1</td>
      <td>Bonnell, Miss. Elizabeth</td>
      <td>female</td>
      <td>58.0</td>
      <td>0</td>
      <td>0</td>
      <td>113783</td>
      <td>26.5500</td>
      <td>C103</td>
      <td>S</td>
    </tr>
    <tr>
      <th>12</th>
      <td>13</td>
      <td>0</td>
      <td>3</td>
      <td>Saundercock, Mr. William Henry</td>
      <td>male</td>
      <td>20.0</td>
      <td>0</td>
      <td>0</td>
      <td>A/5. 2151</td>
      <td>8.0500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>13</th>
      <td>14</td>
      <td>0</td>
      <td>3</td>
      <td>Andersson, Mr. Anders Johan</td>
      <td>male</td>
      <td>39.0</td>
      <td>1</td>
      <td>5</td>
      <td>347082</td>
      <td>31.2750</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>14</th>
      <td>15</td>
      <td>0</td>
      <td>3</td>
      <td>Vestrom, Miss. Hulda Amanda Adolfina</td>
      <td>female</td>
      <td>14.0</td>
      <td>0</td>
      <td>0</td>
      <td>350406</td>
      <td>7.8542</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>15</th>
      <td>16</td>
      <td>1</td>
      <td>2</td>
      <td>Hewlett, Mrs. (Mary D Kingcome)</td>
      <td>female</td>
      <td>55.0</td>
      <td>0</td>
      <td>0</td>
      <td>248706</td>
      <td>16.0000</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>16</th>
      <td>17</td>
      <td>0</td>
      <td>3</td>
      <td>Rice, Master. Eugene</td>
      <td>male</td>
      <td>2.0</td>
      <td>4</td>
      <td>1</td>
      <td>382652</td>
      <td>29.1250</td>
      <td>NaN</td>
      <td>Q</td>
    </tr>
    <tr>
      <th>17</th>
      <td>18</td>
      <td>1</td>
      <td>2</td>
      <td>Williams, Mr. Charles Eugene</td>
      <td>male</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>244373</td>
      <td>13.0000</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>18</th>
      <td>19</td>
      <td>0</td>
      <td>3</td>
      <td>Vander Planke, Mrs. Julius (Emelia Maria Vande...</td>
      <td>female</td>
      <td>31.0</td>
      <td>1</td>
      <td>0</td>
      <td>345763</td>
      <td>18.0000</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>19</th>
      <td>20</td>
      <td>1</td>
      <td>3</td>
      <td>Masselmani, Mrs. Fatima</td>
      <td>female</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>2649</td>
      <td>7.2250</td>
      <td>NaN</td>
      <td>C</td>
    </tr>
    <tr>
      <th>20</th>
      <td>21</td>
      <td>0</td>
      <td>2</td>
      <td>Fynney, Mr. Joseph J</td>
      <td>male</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>239865</td>
      <td>26.0000</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>21</th>
      <td>22</td>
      <td>1</td>
      <td>2</td>
      <td>Beesley, Mr. Lawrence</td>
      <td>male</td>
      <td>34.0</td>
      <td>0</td>
      <td>0</td>
      <td>248698</td>
      <td>13.0000</td>
      <td>D56</td>
      <td>S</td>
    </tr>
    <tr>
      <th>22</th>
      <td>23</td>
      <td>1</td>
      <td>3</td>
      <td>McGowan, Miss. Anna "Annie"</td>
      <td>female</td>
      <td>15.0</td>
      <td>0</td>
      <td>0</td>
      <td>330923</td>
      <td>8.0292</td>
      <td>NaN</td>
      <td>Q</td>
    </tr>
    <tr>
      <th>23</th>
      <td>24</td>
      <td>1</td>
      <td>1</td>
      <td>Sloper, Mr. William Thompson</td>
      <td>male</td>
      <td>28.0</td>
      <td>0</td>
      <td>0</td>
      <td>113788</td>
      <td>35.5000</td>
      <td>A6</td>
      <td>S</td>
    </tr>
    <tr>
      <th>24</th>
      <td>25</td>
      <td>0</td>
      <td>3</td>
      <td>Palsson, Miss. Torborg Danira</td>
      <td>female</td>
      <td>8.0</td>
      <td>3</td>
      <td>1</td>
      <td>349909</td>
      <td>21.0750</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>25</th>
      <td>26</td>
      <td>1</td>
      <td>3</td>
      <td>Asplund, Mrs. Carl Oscar (Selma Augusta Emilia...</td>
      <td>female</td>
      <td>38.0</td>
      <td>1</td>
      <td>5</td>
      <td>347077</td>
      <td>31.3875</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>26</th>
      <td>27</td>
      <td>0</td>
      <td>3</td>
      <td>Emir, Mr. Farred Chehab</td>
      <td>male</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>2631</td>
      <td>7.2250</td>
      <td>NaN</td>
      <td>C</td>
    </tr>
    <tr>
      <th>27</th>
      <td>28</td>
      <td>0</td>
      <td>1</td>
      <td>Fortune, Mr. Charles Alexander</td>
      <td>male</td>
      <td>19.0</td>
      <td>3</td>
      <td>2</td>
      <td>19950</td>
      <td>263.0000</td>
      <td>C23 C25 C27</td>
      <td>S</td>
    </tr>
    <tr>
      <th>28</th>
      <td>29</td>
      <td>1</td>
      <td>3</td>
      <td>O'Dwyer, Miss. Ellen "Nellie"</td>
      <td>female</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>330959</td>
      <td>7.8792</td>
      <td>NaN</td>
      <td>Q</td>
    </tr>
    <tr>
      <th>29</th>
      <td>30</td>
      <td>0</td>
      <td>3</td>
      <td>Todoroff, Mr. Lalio</td>
      <td>male</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>349216</td>
      <td>7.8958</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>861</th>
      <td>862</td>
      <td>0</td>
      <td>2</td>
      <td>Giles, Mr. Frederick Edward</td>
      <td>male</td>
      <td>21.0</td>
      <td>1</td>
      <td>0</td>
      <td>28134</td>
      <td>11.5000</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>862</th>
      <td>863</td>
      <td>1</td>
      <td>1</td>
      <td>Swift, Mrs. Frederick Joel (Margaret Welles Ba...</td>
      <td>female</td>
      <td>48.0</td>
      <td>0</td>
      <td>0</td>
      <td>17466</td>
      <td>25.9292</td>
      <td>D17</td>
      <td>S</td>
    </tr>
    <tr>
      <th>863</th>
      <td>864</td>
      <td>0</td>
      <td>3</td>
      <td>Sage, Miss. Dorothy Edith "Dolly"</td>
      <td>female</td>
      <td>NaN</td>
      <td>8</td>
      <td>2</td>
      <td>CA. 2343</td>
      <td>69.5500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>864</th>
      <td>865</td>
      <td>0</td>
      <td>2</td>
      <td>Gill, Mr. John William</td>
      <td>male</td>
      <td>24.0</td>
      <td>0</td>
      <td>0</td>
      <td>233866</td>
      <td>13.0000</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>865</th>
      <td>866</td>
      <td>1</td>
      <td>2</td>
      <td>Bystrom, Mrs. (Karolina)</td>
      <td>female</td>
      <td>42.0</td>
      <td>0</td>
      <td>0</td>
      <td>236852</td>
      <td>13.0000</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>866</th>
      <td>867</td>
      <td>1</td>
      <td>2</td>
      <td>Duran y More, Miss. Asuncion</td>
      <td>female</td>
      <td>27.0</td>
      <td>1</td>
      <td>0</td>
      <td>SC/PARIS 2149</td>
      <td>13.8583</td>
      <td>NaN</td>
      <td>C</td>
    </tr>
    <tr>
      <th>867</th>
      <td>868</td>
      <td>0</td>
      <td>1</td>
      <td>Roebling, Mr. Washington Augustus II</td>
      <td>male</td>
      <td>31.0</td>
      <td>0</td>
      <td>0</td>
      <td>PC 17590</td>
      <td>50.4958</td>
      <td>A24</td>
      <td>S</td>
    </tr>
    <tr>
      <th>868</th>
      <td>869</td>
      <td>0</td>
      <td>3</td>
      <td>van Melkebeke, Mr. Philemon</td>
      <td>male</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>345777</td>
      <td>9.5000</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>869</th>
      <td>870</td>
      <td>1</td>
      <td>3</td>
      <td>Johnson, Master. Harold Theodor</td>
      <td>male</td>
      <td>4.0</td>
      <td>1</td>
      <td>1</td>
      <td>347742</td>
      <td>11.1333</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>870</th>
      <td>871</td>
      <td>0</td>
      <td>3</td>
      <td>Balkic, Mr. Cerin</td>
      <td>male</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>349248</td>
      <td>7.8958</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>871</th>
      <td>872</td>
      <td>1</td>
      <td>1</td>
      <td>Beckwith, Mrs. Richard Leonard (Sallie Monypeny)</td>
      <td>female</td>
      <td>47.0</td>
      <td>1</td>
      <td>1</td>
      <td>11751</td>
      <td>52.5542</td>
      <td>D35</td>
      <td>S</td>
    </tr>
    <tr>
      <th>872</th>
      <td>873</td>
      <td>0</td>
      <td>1</td>
      <td>Carlsson, Mr. Frans Olof</td>
      <td>male</td>
      <td>33.0</td>
      <td>0</td>
      <td>0</td>
      <td>695</td>
      <td>5.0000</td>
      <td>B51 B53 B55</td>
      <td>S</td>
    </tr>
    <tr>
      <th>873</th>
      <td>874</td>
      <td>0</td>
      <td>3</td>
      <td>Vander Cruyssen, Mr. Victor</td>
      <td>male</td>
      <td>47.0</td>
      <td>0</td>
      <td>0</td>
      <td>345765</td>
      <td>9.0000</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>874</th>
      <td>875</td>
      <td>1</td>
      <td>2</td>
      <td>Abelson, Mrs. Samuel (Hannah Wizosky)</td>
      <td>female</td>
      <td>28.0</td>
      <td>1</td>
      <td>0</td>
      <td>P/PP 3381</td>
      <td>24.0000</td>
      <td>NaN</td>
      <td>C</td>
    </tr>
    <tr>
      <th>875</th>
      <td>876</td>
      <td>1</td>
      <td>3</td>
      <td>Najib, Miss. Adele Kiamie "Jane"</td>
      <td>female</td>
      <td>15.0</td>
      <td>0</td>
      <td>0</td>
      <td>2667</td>
      <td>7.2250</td>
      <td>NaN</td>
      <td>C</td>
    </tr>
    <tr>
      <th>876</th>
      <td>877</td>
      <td>0</td>
      <td>3</td>
      <td>Gustafsson, Mr. Alfred Ossian</td>
      <td>male</td>
      <td>20.0</td>
      <td>0</td>
      <td>0</td>
      <td>7534</td>
      <td>9.8458</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>877</th>
      <td>878</td>
      <td>0</td>
      <td>3</td>
      <td>Petroff, Mr. Nedelio</td>
      <td>male</td>
      <td>19.0</td>
      <td>0</td>
      <td>0</td>
      <td>349212</td>
      <td>7.8958</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>878</th>
      <td>879</td>
      <td>0</td>
      <td>3</td>
      <td>Laleff, Mr. Kristo</td>
      <td>male</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>349217</td>
      <td>7.8958</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>879</th>
      <td>880</td>
      <td>1</td>
      <td>1</td>
      <td>Potter, Mrs. Thomas Jr (Lily Alexenia Wilson)</td>
      <td>female</td>
      <td>56.0</td>
      <td>0</td>
      <td>1</td>
      <td>11767</td>
      <td>83.1583</td>
      <td>C50</td>
      <td>C</td>
    </tr>
    <tr>
      <th>880</th>
      <td>881</td>
      <td>1</td>
      <td>2</td>
      <td>Shelley, Mrs. William (Imanita Parrish Hall)</td>
      <td>female</td>
      <td>25.0</td>
      <td>0</td>
      <td>1</td>
      <td>230433</td>
      <td>26.0000</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>881</th>
      <td>882</td>
      <td>0</td>
      <td>3</td>
      <td>Markun, Mr. Johann</td>
      <td>male</td>
      <td>33.0</td>
      <td>0</td>
      <td>0</td>
      <td>349257</td>
      <td>7.8958</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>882</th>
      <td>883</td>
      <td>0</td>
      <td>3</td>
      <td>Dahlberg, Miss. Gerda Ulrika</td>
      <td>female</td>
      <td>22.0</td>
      <td>0</td>
      <td>0</td>
      <td>7552</td>
      <td>10.5167</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>883</th>
      <td>884</td>
      <td>0</td>
      <td>2</td>
      <td>Banfield, Mr. Frederick James</td>
      <td>male</td>
      <td>28.0</td>
      <td>0</td>
      <td>0</td>
      <td>C.A./SOTON 34068</td>
      <td>10.5000</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>884</th>
      <td>885</td>
      <td>0</td>
      <td>3</td>
      <td>Sutehall, Mr. Henry Jr</td>
      <td>male</td>
      <td>25.0</td>
      <td>0</td>
      <td>0</td>
      <td>SOTON/OQ 392076</td>
      <td>7.0500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>885</th>
      <td>886</td>
      <td>0</td>
      <td>3</td>
      <td>Rice, Mrs. William (Margaret Norton)</td>
      <td>female</td>
      <td>39.0</td>
      <td>0</td>
      <td>5</td>
      <td>382652</td>
      <td>29.1250</td>
      <td>NaN</td>
      <td>Q</td>
    </tr>
    <tr>
      <th>886</th>
      <td>887</td>
      <td>0</td>
      <td>2</td>
      <td>Montvila, Rev. Juozas</td>
      <td>male</td>
      <td>27.0</td>
      <td>0</td>
      <td>0</td>
      <td>211536</td>
      <td>13.0000</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>887</th>
      <td>888</td>
      <td>1</td>
      <td>1</td>
      <td>Graham, Miss. Margaret Edith</td>
      <td>female</td>
      <td>19.0</td>
      <td>0</td>
      <td>0</td>
      <td>112053</td>
      <td>30.0000</td>
      <td>B42</td>
      <td>S</td>
    </tr>
    <tr>
      <th>888</th>
      <td>889</td>
      <td>0</td>
      <td>3</td>
      <td>Johnston, Miss. Catherine Helen "Carrie"</td>
      <td>female</td>
      <td>NaN</td>
      <td>1</td>
      <td>2</td>
      <td>W./C. 6607</td>
      <td>23.4500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>889</th>
      <td>890</td>
      <td>1</td>
      <td>1</td>
      <td>Behr, Mr. Karl Howell</td>
      <td>male</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>111369</td>
      <td>30.0000</td>
      <td>C148</td>
      <td>C</td>
    </tr>
    <tr>
      <th>890</th>
      <td>891</td>
      <td>0</td>
      <td>3</td>
      <td>Dooley, Mr. Patrick</td>
      <td>male</td>
      <td>32.0</td>
      <td>0</td>
      <td>0</td>
      <td>370376</td>
      <td>7.7500</td>
      <td>NaN</td>
      <td>Q</td>
    </tr>
  </tbody>
</table>
<p>891 rows × 12 columns</p>
</div>




```python
titanic_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>Braund, Mr. Owen Harris</td>
      <td>male</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>A/5 21171</td>
      <td>7.2500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>
      <td>female</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>PC 17599</td>
      <td>71.2833</td>
      <td>C85</td>
      <td>C</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>Heikkinen, Miss. Laina</td>
      <td>female</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>STON/O2. 3101282</td>
      <td>7.9250</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>
      <td>female</td>
      <td>35.0</td>
      <td>1</td>
      <td>0</td>
      <td>113803</td>
      <td>53.1000</td>
      <td>C123</td>
      <td>S</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>0</td>
      <td>3</td>
      <td>Allen, Mr. William Henry</td>
      <td>male</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>373450</td>
      <td>8.0500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
  </tbody>
</table>
</div>




```python
# DataFrame의 행과 열 크기 반환은 shape 변수 활용, 튜플로 반환
print('DataFrame의 크기: ', titanic_df.shape)
```

    DataFrame의 크기:  (891, 12)



```python
# DataFrame 칼럼의 타입, Null 데이터 갯수, 데이터 분포도 등의 메타 데이터 등 조회 가능
# info() 메서드 : 총 데이터 건수, 데이터 타입, Null 건수
titanic_df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 891 entries, 0 to 890
    Data columns (total 12 columns):
    PassengerId    891 non-null int64
    Survived       891 non-null int64
    Pclass         891 non-null int64
    Name           891 non-null object
    Sex            891 non-null object
    Age            714 non-null float64
    SibSp          891 non-null int64
    Parch          891 non-null int64
    Ticket         891 non-null object
    Fare           891 non-null float64
    Cabin          204 non-null object
    Embarked       889 non-null object
    dtypes: float64(2), int64(5), object(5)
    memory usage: 83.6+ KB



```python
# describe() 메서드 : 칼럼별 숫자형 데이터 값의 n-percentile 분포도, 평균값, 최댓값, 최솟값
# 오직 숫자형(int, float 등) 칼럼의 분포도만 조사하며 object 타입 자동 제외
titanic_df.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Fare</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>891.000000</td>
      <td>891.000000</td>
      <td>891.000000</td>
      <td>714.000000</td>
      <td>891.000000</td>
      <td>891.000000</td>
      <td>891.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>446.000000</td>
      <td>0.383838</td>
      <td>2.308642</td>
      <td>29.699118</td>
      <td>0.523008</td>
      <td>0.381594</td>
      <td>32.204208</td>
    </tr>
    <tr>
      <th>std</th>
      <td>257.353842</td>
      <td>0.486592</td>
      <td>0.836071</td>
      <td>14.526497</td>
      <td>1.102743</td>
      <td>0.806057</td>
      <td>49.693429</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.420000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>223.500000</td>
      <td>0.000000</td>
      <td>2.000000</td>
      <td>20.125000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>7.910400</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>446.000000</td>
      <td>0.000000</td>
      <td>3.000000</td>
      <td>28.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>14.454200</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>668.500000</td>
      <td>1.000000</td>
      <td>3.000000</td>
      <td>38.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>31.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>891.000000</td>
      <td>1.000000</td>
      <td>3.000000</td>
      <td>80.000000</td>
      <td>8.000000</td>
      <td>6.000000</td>
      <td>512.329200</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Pclass 칼럼 값 분포
value_counts = titanic_df['Pclass'].value_counts()
print(value_counts)
```

    3    491
    1    216
    2    184
    Name: Pclass, dtype: int64



```python
# Series 객체 반환 : []내부 칼럼명 입력
titanic_pclass = titanic_df['Pclass']
print(type(titanic_pclass))
```

    <class 'pandas.core.series.Series'>



```python
# Series : Index와 단 하나의 칼럼으로 이루어진 데이터 세트
titanic_pclass.head()
```




    0    3
    1    1
    2    3
    3    1
    4    3
    Name: Pclass, dtype: int64




```python
# DataFrame은 value_counts() 메서드를 가지지 않는다. 오직 Series만 가진다.
value_counts = titanic_df['Pclass'].value_counts()
print(type(value_counts))
print(value_counts) # value_count() 반환 Series의 인덱스 값과 데이터 값으로 반환한다.
```

    <class 'pandas.core.series.Series'>
    3    491
    1    216
    2    184
    Name: Pclass, dtype: int64


* 모든 인덱스는 고유성이 보장되어야 한다.

### DataFrame과 리스트, 딕셔너리, 넘파이 ndarray 상호 변환
#### 넘파이 ndarray, list, dictionary를 DataFrame으로 변환하기


```python
# 사이킷런 API는 DataFrame 인자로 입력되지만 기본적으로 ndarray를 입력 인자로 많이 사용
import numpy as np

col_name1 = ['col1']
list1 = [1,2,3]
array1 = np.array(list1)
print('array1 shape:', array1.shape )
```

    array1 shape: (3,)



```python
# 리스트 이용해 DataFrame 생성
df_list1 = pd.DataFrame(list1, columns = col_name1)
print('1차원 리스트로 만든 DataFrame:\n', df_list1)
```

    1차원 리스트로 만든 DataFrame:
        col1
    0     1
    1     2
    2     3



```python
# 넘파이 ndarray 이용해 DataFrame 생성
df_array1 = pd.DataFrame(array1, columns = col_name1)
print('1차원 ndarray로 만든 DataFrame:\n', df_array1)
```

    1차원 ndarray로 만든 DataFrame:
        col1
    0     1
    1     2
    2     3



```python
# 2차원 형태 데이터 기반으로 DataFrame 생성
# 2행 3열로 제작하므로 칼럼명 3개 필요
col_name2 = ['col1', 'col2', 'col3']

# 2행 3열
list2 = [[1, 2, 3],
        [11, 12, 13]]
array2 = np.array(list2)
print('array2 shape:', array2.shape)
```

    array2 shape: (2, 3)



```python
df_list2 = pd.DataFrame(list2, columns=col_name2)
print('2차원 리스트로 만든 DataFrame:\n', df_list2)
```

    2차원 리스트로 만든 DataFrame:
        col1  col2  col3
    0     1     2     3
    1    11    12    13



```python
df_array2 = pd.DataFrame(array2, columns=col_name2)
print('2차원 ndarray로 만든 DataFrame:\n', df_array2)
```

    2차원 ndarray로 만든 DataFrame:
        col1  col2  col3
    0     1     2     3
    1    11    12    13



```python
# 딕셔너리를 DataFrame으로 변환
# key 문자열 칼럼명 매핑, Value 리스트/ndarray 칼럼 데이터로 매핑
dict = {'col1':[1, 11], 'col2':[2, 12], 'col3':[3, 13]}
df_dict = pd.DataFrame(dict)
print('딕셔너리로 만든 DataFrame:\n',df_dict)
```

    딕셔너리로 만든 DataFrame:
        col1  col2  col3
    0     1     2     3
    1    11    12    13



```python
# DataFrame을 ndarray로 변환
array3 = df_dict.values
print('df_dict.values 타입:', type(array3), 'df_dict.values shape:', array3.shape)
print(array3)
```

    df_dict.values 타입: <class 'numpy.ndarray'> df_dict.values shape: (2, 3)
    [[ 1  2  3]
     [11 12 13]]



```python
# DataFrame을 리스트로 변환
list3 = df_dict.values.tolist() # 딕셔너리 값을 추출해서 리스트로 만든다.(ndarray의 tolist())
print('df_dict.value.tolist() 타입:', type(list3))
print(list3)
```

    df_dict.value.tolist() 타입: <class 'list'>
    [[1, 2, 3], [11, 12, 13]]



```python
# DataFrame을 딕셔너리로 반환
dict3 = df_dict.to_dict('list') # DataFrame 개체의 to_dict() 메서드로 인자는 'list' 입력시 딕셔너리 값이 리스트형으로 변환
print('df_dict.to_dict() 타입:', type(dict3))
print(dict3)
```

    df_dict.to_dict() 타입: <class 'dict'>
    {'col1': [1, 11], 'col2': [2, 12], 'col3': [3, 13]}


### DataFrame의 칼럼 데이터 세트 생성과 수정


```python
# [] 연산자로 쉽게 수정 가능하다.
titanic_df['Age_0'] = 0 # 'Age_0'으로 모든 데이터 값이 0인 Series가 추가된다. 상수 값은 해당 Series 모든 값에 일괄 적용
titanic_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
      <th>Age_0</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>Braund, Mr. Owen Harris</td>
      <td>male</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>A/5 21171</td>
      <td>7.2500</td>
      <td>NaN</td>
      <td>S</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>
      <td>female</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>PC 17599</td>
      <td>71.2833</td>
      <td>C85</td>
      <td>C</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>Heikkinen, Miss. Laina</td>
      <td>female</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>STON/O2. 3101282</td>
      <td>7.9250</td>
      <td>NaN</td>
      <td>S</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>
      <td>female</td>
      <td>35.0</td>
      <td>1</td>
      <td>0</td>
      <td>113803</td>
      <td>53.1000</td>
      <td>C123</td>
      <td>S</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>0</td>
      <td>3</td>
      <td>Allen, Mr. William Henry</td>
      <td>male</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>373450</td>
      <td>8.0500</td>
      <td>NaN</td>
      <td>S</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 기존 칼럼 Series로 새로운 칼럼 생성
titanic_df['Age_by_10'] = titanic_df['Age']*10
titanic_df['Family_No'] = titanic_df['SibSp'] + titanic_df['Parch']+1
titanic_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
      <th>Age_0</th>
      <th>Age_by_10</th>
      <th>Family_No</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>Braund, Mr. Owen Harris</td>
      <td>male</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>A/5 21171</td>
      <td>7.2500</td>
      <td>NaN</td>
      <td>S</td>
      <td>0</td>
      <td>220.0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>
      <td>female</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>PC 17599</td>
      <td>71.2833</td>
      <td>C85</td>
      <td>C</td>
      <td>0</td>
      <td>380.0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>Heikkinen, Miss. Laina</td>
      <td>female</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>STON/O2. 3101282</td>
      <td>7.9250</td>
      <td>NaN</td>
      <td>S</td>
      <td>0</td>
      <td>260.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>
      <td>female</td>
      <td>35.0</td>
      <td>1</td>
      <td>0</td>
      <td>113803</td>
      <td>53.1000</td>
      <td>C123</td>
      <td>S</td>
      <td>0</td>
      <td>350.0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>0</td>
      <td>3</td>
      <td>Allen, Mr. William Henry</td>
      <td>male</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>373450</td>
      <td>8.0500</td>
      <td>NaN</td>
      <td>S</td>
      <td>0</td>
      <td>350.0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 기존 칼럼 값 일괄 업데이트, 칼럼 명 입력 후 값 할당
titanic_df['Age_by_10'] = titanic_df['Age_by_10'] + 100
titanic_df.head(3)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
      <th>Age_0</th>
      <th>Age_by_10</th>
      <th>Family_No</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>Braund, Mr. Owen Harris</td>
      <td>male</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>A/5 21171</td>
      <td>7.2500</td>
      <td>NaN</td>
      <td>S</td>
      <td>0</td>
      <td>320.0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>
      <td>female</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>PC 17599</td>
      <td>71.2833</td>
      <td>C85</td>
      <td>C</td>
      <td>0</td>
      <td>480.0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>Heikkinen, Miss. Laina</td>
      <td>female</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>STON/O2. 3101282</td>
      <td>7.9250</td>
      <td>NaN</td>
      <td>S</td>
      <td>0</td>
      <td>360.0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



#### DataFrame 데이터 삭제

* drop() 메서드 활용 주의가 필요하다.  

      DataFrame.drop(labels=None, axis=0, index= None, columns=None, level=None, inplace=False, errors='raise')


```python
# axis 0 로우 방향 축, axis 1 칼럼 방향 축
titanic_drop_df = titanic_df.drop('Age_0', axis=1) # 하나 제거시에는 [] 없고 축만 명확히 언급해줘라.
titanic_drop_df.head(3)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
      <th>Age_by_10</th>
      <th>Family_No</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>Braund, Mr. Owen Harris</td>
      <td>male</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>A/5 21171</td>
      <td>7.2500</td>
      <td>NaN</td>
      <td>S</td>
      <td>320.0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>
      <td>female</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>PC 17599</td>
      <td>71.2833</td>
      <td>C85</td>
      <td>C</td>
      <td>480.0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>Heikkinen, Miss. Laina</td>
      <td>female</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>STON/O2. 3101282</td>
      <td>7.9250</td>
      <td>NaN</td>
      <td>S</td>
      <td>360.0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
# inplace 알아보기
# 원본 데이터프레임
titanic_df.head(3) # Age_0 칼럼 그대로 존재, inplace=False
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
      <th>Age_0</th>
      <th>Age_by_10</th>
      <th>Family_No</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>Braund, Mr. Owen Harris</td>
      <td>male</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>A/5 21171</td>
      <td>7.2500</td>
      <td>NaN</td>
      <td>S</td>
      <td>0</td>
      <td>320.0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>
      <td>female</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>PC 17599</td>
      <td>71.2833</td>
      <td>C85</td>
      <td>C</td>
      <td>0</td>
      <td>480.0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>Heikkinen, Miss. Laina</td>
      <td>female</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>STON/O2. 3101282</td>
      <td>7.9250</td>
      <td>NaN</td>
      <td>S</td>
      <td>0</td>
      <td>360.0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
# inplace=True로 하면 원본도 바뀌어 버린다.
drop_result = titanic_df.drop(['Age_0', 'Age_by_10', 'Family_No'], axis =1, inplace=True)
print(' inplace = True 로 drop 후 반환된 값:', drop_result)
```

     inplace = True 로 drop 후 반환된 값: None



```python
titanic_df.head(3)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>Braund, Mr. Owen Harris</td>
      <td>male</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>A/5 21171</td>
      <td>7.2500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>
      <td>female</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>PC 17599</td>
      <td>71.2833</td>
      <td>C85</td>
      <td>C</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>Heikkinen, Miss. Laina</td>
      <td>female</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>STON/O2. 3101282</td>
      <td>7.9250</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
  </tbody>
</table>
</div>




```python
# axis=0으로 설정해 indexes 0,1,2 맨 앞 3개 데이터 로우를 삭제해보자.
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 15)
print('#### before axis 0 drop ####')
print(titanic_df.head(3))
```

    #### before axis 0 drop ####
       PassengerId  Survived  Pclass            Name     Sex   Age  SibSp  Parch          Ticket     Fare Cabin Embarked
    0            1         0       3  Braund, Mr....    male  22.0      1      0       A/5 21171   7.2500   NaN        S
    1            2         1       1  Cumings, Mr...  female  38.0      1      0        PC 17599  71.2833   C85        C
    2            3         1       3  Heikkinen, ...  female  26.0      0      0  STON/O2. 31...   7.9250   NaN        S



```python
titanic_df.drop([0,1,2], axis=0, inplace=True)

print('#### after axis 0 drop ####')
print(titanic_df.head(3))
```

    #### after axis 0 drop ####
       PassengerId  Survived  Pclass            Name     Sex   Age  SibSp  Parch  Ticket     Fare Cabin Embarked
    3            4         1       1  Futrelle, M...  female  35.0      1      0  113803  53.1000  C123        S
    4            5         0       3  Allen, Mr. ...    male  35.0      0      0  373450   8.0500   NaN        S
    5            6         0       3  Moran, Mr. ...    male   NaN      0      0  330877   8.4583   NaN        Q


#### Index 객체


```python
# 원본 파일 다시 로딩
titanic_df = pd.read_csv('titanic_train.csv')

# index 객체 추출
indexes = titanic_df.index
print(indexes)

# Index 객체를 실제 값 array로 변환
print('Index 객체 array 값:\n', indexes.values)
```

    RangeIndex(start=0, stop=891, step=1)
    Index 객체 array 값:
     [  0   1   2   3   4   5   6   7   8   9  10  11  12  13  14  15  16  17
      18  19  20  21  22  23  24  25  26  27  28  29  30  31  32  33  34  35
      36  37  38  39  40  41  42  43  44  45  46  47  48  49  50  51  52  53
      54  55  56  57  58  59  60  61  62  63  64  65  66  67  68  69  70  71
      72  73  74  75  76  77  78  79  80  81  82  83  84  85  86  87  88  89
      90  91  92  93  94  95  96  97  98  99 100 101 102 103 104 105 106 107
     108 109 110 111 112 113 114 115 116 117 118 119 120 121 122 123 124 125
     126 127 128 129 130 131 132 133 134 135 136 137 138 139 140 141 142 143
     144 145 146 147 148 149 150 151 152 153 154 155 156 157 158 159 160 161
     162 163 164 165 166 167 168 169 170 171 172 173 174 175 176 177 178 179
     180 181 182 183 184 185 186 187 188 189 190 191 192 193 194 195 196 197
     198 199 200 201 202 203 204 205 206 207 208 209 210 211 212 213 214 215
     216 217 218 219 220 221 222 223 224 225 226 227 228 229 230 231 232 233
     234 235 236 237 238 239 240 241 242 243 244 245 246 247 248 249 250 251
     252 253 254 255 256 257 258 259 260 261 262 263 264 265 266 267 268 269
     270 271 272 273 274 275 276 277 278 279 280 281 282 283 284 285 286 287
     288 289 290 291 292 293 294 295 296 297 298 299 300 301 302 303 304 305
     306 307 308 309 310 311 312 313 314 315 316 317 318 319 320 321 322 323
     324 325 326 327 328 329 330 331 332 333 334 335 336 337 338 339 340 341
     342 343 344 345 346 347 348 349 350 351 352 353 354 355 356 357 358 359
     360 361 362 363 364 365 366 367 368 369 370 371 372 373 374 375 376 377
     378 379 380 381 382 383 384 385 386 387 388 389 390 391 392 393 394 395
     396 397 398 399 400 401 402 403 404 405 406 407 408 409 410 411 412 413
     414 415 416 417 418 419 420 421 422 423 424 425 426 427 428 429 430 431
     432 433 434 435 436 437 438 439 440 441 442 443 444 445 446 447 448 449
     450 451 452 453 454 455 456 457 458 459 460 461 462 463 464 465 466 467
     468 469 470 471 472 473 474 475 476 477 478 479 480 481 482 483 484 485
     486 487 488 489 490 491 492 493 494 495 496 497 498 499 500 501 502 503
     504 505 506 507 508 509 510 511 512 513 514 515 516 517 518 519 520 521
     522 523 524 525 526 527 528 529 530 531 532 533 534 535 536 537 538 539
     540 541 542 543 544 545 546 547 548 549 550 551 552 553 554 555 556 557
     558 559 560 561 562 563 564 565 566 567 568 569 570 571 572 573 574 575
     576 577 578 579 580 581 582 583 584 585 586 587 588 589 590 591 592 593
     594 595 596 597 598 599 600 601 602 603 604 605 606 607 608 609 610 611
     612 613 614 615 616 617 618 619 620 621 622 623 624 625 626 627 628 629
     630 631 632 633 634 635 636 637 638 639 640 641 642 643 644 645 646 647
     648 649 650 651 652 653 654 655 656 657 658 659 660 661 662 663 664 665
     666 667 668 669 670 671 672 673 674 675 676 677 678 679 680 681 682 683
     684 685 686 687 688 689 690 691 692 693 694 695 696 697 698 699 700 701
     702 703 704 705 706 707 708 709 710 711 712 713 714 715 716 717 718 719
     720 721 722 723 724 725 726 727 728 729 730 731 732 733 734 735 736 737
     738 739 740 741 742 743 744 745 746 747 748 749 750 751 752 753 754 755
     756 757 758 759 760 761 762 763 764 765 766 767 768 769 770 771 772 773
     774 775 776 777 778 779 780 781 782 783 784 785 786 787 788 789 790 791
     792 793 794 795 796 797 798 799 800 801 802 803 804 805 806 807 808 809
     810 811 812 813 814 815 816 817 818 819 820 821 822 823 824 825 826 827
     828 829 830 831 832 833 834 835 836 837 838 839 840 841 842 843 844 845
     846 847 848 849 850 851 852 853 854 855 856 857 858 859 860 861 862 863
     864 865 866 867 868 869 870 871 872 873 874 875 876 877 878 879 880 881
     882 883 884 885 886 887 888 889 890]



```python
# 인덱스 객체는 식별성 데이터 1차원 배열로 가진다. ndarray와 유사하게 단일 값 반환 및 슬라이싱 가능
print(type(indexes.values))
print(indexes.values.shape)

# 슬라이싱
print(indexes[:5].values)
```

    <class 'numpy.ndarray'>
    (891,)
    [0 1 2 3 4]



```python
print(indexes.values[:6])
```

    [0 1 2 3 4 5]



```python
print(indexes[6])
```

    6



```python
# 한 번 만들어진 DataFrame, Series의 Index 객체는 함부로 변경 불가능하다.
indexes[0] = 5
```


    ---------------------------------------------------------------------------

    TypeError                                 Traceback (most recent call last)

    <ipython-input-156-c40105e7d95c> in <module>
          1 # 한 번 만들어진 DataFrame, Series의 Index 객체는 함부로 변경 불가능하다.
    ----> 2 indexes[0] = 5


    /usr/local/lib/python3.7/site-packages/pandas/core/indexes/base.py in __setitem__(self, key, value)
       3936
       3937     def __setitem__(self, key, value):
    -> 3938         raise TypeError("Index does not support mutable operations")
       3939
       3940     def __getitem__(self, key):


    TypeError: Index does not support mutable operations



```python
# Series 객체도 인덱스 포함하지만 오직 식별용으로만 사용한다. 연산 제외한다.
series_fair = titanic_df['Fare']
print('Fair Series max 값:', series_fair.max())
print('Fair Series sum 값:', series_fair.sum())
print('sum() Fair Series:', sum(series_fair))
print('Fair Series + 3:\n', (series_fair + 3).head(3))
```

    Fair Series max 값: 512.3292
    Fair Series sum 값: 28693.9493
    sum() Fair Series: 28693.949299999967
    Fair Series + 3:
     0    10.2500
    1    74.2833
    2    10.9250
    Name: Fare, dtype: float64



```python
# reset_index() 메서드 : 새롭게 인덱스를 연속 숫자 형으로 할당, 기존 인덱스는 'index' 칼럼 추가
titanic_reset_df = titanic_df.reset_index(inplace=False)
titanic_reset_df.head(3)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>index</th>
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>Braund, Mr....</td>
      <td>male</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>A/5 21171</td>
      <td>7.2500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>Cumings, Mr...</td>
      <td>female</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>PC 17599</td>
      <td>71.2833</td>
      <td>C85</td>
      <td>C</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>Heikkinen, ...</td>
      <td>female</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>STON/O2. 31...</td>
      <td>7.9250</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Series에 reset_index() 적용시 index 칼럼 추가되어 데이터프레임 반환됨에 유의
print('### before reset_index ###')
value_counts = titanic_df['Pclass'].value_counts()
print(value_counts)
print('value_counts 객체 변수 타입:', type(value_counts))
```

    ### before reset_index ###
    3    491
    1    216
    2    184
    Name: Pclass, dtype: int64
    value_counts 객체 변수 타입: <class 'pandas.core.series.Series'>



```python
new_value_counts = value_counts.reset_index(inplace=False)
print('### after reset_index ###')
print(new_value_counts)
print('new_value_counts 객체 변수 타입:', type(new_value_counts))
```

    ### after reset_index ###
       index  Pclass
    0      3     491
    1      1     216
    2      2     184
    new_value_counts 객체 변수 타입: <class 'pandas.core.frame.DataFrame'>



```python
# 위처럼 기존 인덱스가 칼럼으로 추가되는 게 싫다면 reset_index() drop=True 설정
new_value_counts = value_counts.reset_index(inplace=False, drop=True)
print('### after reset_index and drop True ###')
print(new_value_counts)
print('new_value_counts 객체 변수 타입:', type(new_value_counts)) # Series 유지된다.
```

    ### after reset_index and drop True ###
    0    491
    1    216
    2    184
    Name: Pclass, dtype: int64
    new_value_counts 객체 변수 타입: <class 'pandas.core.series.Series'>


### 데이터 셀렉션 및 필터링

* 현재 DataFrame 뒤에 있는 []는 칼럼만 지정할 수 있는 '칼럼 지정 연산자'로 이해
    * 반드시 문자열 칼럼만 지정하거나 문자열 객체 리스트 지원한다. 숫자는 칼럼 명이 아니다.


```python
print('단일 칼럼 데이터 추출:\n', titanic_df['Pclass'].head(3))
print('\n여러 칼럼 데이터 추출:\n', titanic_df[['Survived', 'Pclass']].head(3))
print('[ ] 안에 숫자 index는 KeyError 오류 발생:\n', titanic_df[0])
```

    단일 칼럼 데이터 추출:
     0    3
    1    1
    2    3
    Name: Pclass, dtype: int64

    여러 칼럼 데이터 추출:
        Survived  Pclass
    0         0       3
    1         1       1
    2         1       3



    ---------------------------------------------------------------------------

    KeyError                                  Traceback (most recent call last)

    /usr/local/lib/python3.7/site-packages/pandas/core/indexes/base.py in get_loc(self, key, method, tolerance)
       2656             try:
    -> 2657                 return self._engine.get_loc(key)
       2658             except KeyError:


    pandas/_libs/index.pyx in pandas._libs.index.IndexEngine.get_loc()


    pandas/_libs/index.pyx in pandas._libs.index.IndexEngine.get_loc()


    pandas/_libs/hashtable_class_helper.pxi in pandas._libs.hashtable.PyObjectHashTable.get_item()


    pandas/_libs/hashtable_class_helper.pxi in pandas._libs.hashtable.PyObjectHashTable.get_item()


    KeyError: 0


    During handling of the above exception, another exception occurred:


    KeyError                                  Traceback (most recent call last)

    <ipython-input-163-7e12472e1cac> in <module>
          1 print('단일 칼럼 데이터 추출:\n', titanic_df['Pclass'].head(3))
          2 print('\n여러 칼럼 데이터 추출:\n', titanic_df[['Survived', 'Pclass']].head(3))
    ----> 3 print('[ ] 안에 숫자 index는 KeyError 오류 발생:\n', titanic_df[0])


    /usr/local/lib/python3.7/site-packages/pandas/core/frame.py in __getitem__(self, key)
       2925             if self.columns.nlevels > 1:
       2926                 return self._getitem_multilevel(key)
    -> 2927             indexer = self.columns.get_loc(key)
       2928             if is_integer(indexer):
       2929                 indexer = [indexer]


    /usr/local/lib/python3.7/site-packages/pandas/core/indexes/base.py in get_loc(self, key, method, tolerance)
       2657                 return self._engine.get_loc(key)
       2658             except KeyError:
    -> 2659                 return self._engine.get_loc(self._maybe_cast_indexer(key))
       2660         indexer = self.get_indexer([key], method=method, tolerance=tolerance)
       2661         if indexer.ndim > 1 or indexer.size > 1:


    pandas/_libs/index.pyx in pandas._libs.index.IndexEngine.get_loc()


    pandas/_libs/index.pyx in pandas._libs.index.IndexEngine.get_loc()


    pandas/_libs/hashtable_class_helper.pxi in pandas._libs.hashtable.PyObjectHashTable.get_item()


    pandas/_libs/hashtable_class_helper.pxi in pandas._libs.hashtable.PyObjectHashTable.get_item()


    KeyError: 0



```python
# 판다스의 인덱스 형태로 변환 가능한 표현식은 입력 가능하다. 슬라이싱의 경우 판다스도 정확히 원하는 결과를 반영한다.
titanic_df[ 0:2 ] # 사용하지 않는게 좋다. 특히 row만 추출한다는 점 유의한다.
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>Braund, Mr....</td>
      <td>male</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>A/5 21171</td>
      <td>7.2500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>Cumings, Mr...</td>
      <td>female</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>PC 17599</td>
      <td>71.2833</td>
      <td>C85</td>
      <td>C</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 판다스 불린 인덱싱 : 데이터 셀렉션을 불린 인덱싱 기반으로 쉽게 할 수 있다. 다만 [] 안에 다시 데이터프레임을 구체적으로 적어줘야한다.

titanic_df[ titanic_df['Pclass']==3].head(3)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>Braund, Mr....</td>
      <td>male</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>A/5 21171</td>
      <td>7.250</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>Heikkinen, ...</td>
      <td>female</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>STON/O2. 31...</td>
      <td>7.925</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>0</td>
      <td>3</td>
      <td>Allen, Mr. ...</td>
      <td>male</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>373450</td>
      <td>8.050</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
  </tbody>
</table>
</div>




```python
# DataFrame ix[] 연산자 : 혼동을 주므로 사라진다.(칼럼 명칭, 위치 기반 모두 지원)
# 칼럼 명칭 기반 인덱싱 연산자 loc[], 위치 기반 인덱싱 연산자 iloc[] 등장
print('칼럼 위치 기반 인덱싱 데이터 추출:', titanic_df.ix[0,2])
print('칼럼 명 기반 인덱싱 데이터 추출:', titanic_df.ix[0, 'Pclass'])
```

    칼럼 위치 기반 인덱싱 데이터 추출: 3
    칼럼 명 기반 인덱싱 데이터 추출: 3


    /usr/local/lib/python3.7/site-packages/ipykernel_launcher.py:3: DeprecationWarning:
    .ix is deprecated. Please use
    .loc for label based indexing or
    .iloc for positional indexing

    See the documentation here:
    http://pandas.pydata.org/pandas-docs/stable/indexing.html#ix-indexer-is-deprecated
      This is separate from the ipykernel package so we can avoid doing imports until
    /usr/local/lib/python3.7/site-packages/ipykernel_launcher.py:4: DeprecationWarning:
    .ix is deprecated. Please use
    .loc for label based indexing or
    .iloc for positional indexing

    See the documentation here:
    http://pandas.pydata.org/pandas-docs/stable/indexing.html#ix-indexer-is-deprecated
      after removing the cwd from sys.path.



```python
# DataFrame의 ix[] 연산자 : 단일 지정, 슬라이싱, 불린 인덱싱, 팬시 인덱싱 모두 가능
data = {'Name' : ['A','B','C','D'],
        'Year' : [2011,2016,2015,2015],
        'Gender' : ['Male', 'Female', 'Male', 'Male'],
    }
data_df = pd.DataFrame(data, index=['one', 'two', 'three','four'])
data_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Name</th>
      <th>Year</th>
      <th>Gender</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>one</th>
      <td>A</td>
      <td>2011</td>
      <td>Male</td>
    </tr>
    <tr>
      <th>two</th>
      <td>B</td>
      <td>2016</td>
      <td>Female</td>
    </tr>
    <tr>
      <th>three</th>
      <td>C</td>
      <td>2015</td>
      <td>Male</td>
    </tr>
    <tr>
      <th>four</th>
      <td>D</td>
      <td>2015</td>
      <td>Male</td>
    </tr>
  </tbody>
</table>
</div>




```python
print(data_df.ix[0,0])
print(data_df.ix['one', 0])
print(data_df.ix[3, 'Name'])
print(data_df.ix[0:2, [0, 1]])
print(data_df.ix[0:2,0:3])
print(data_df.ix[0:3, ['Name','Year']])
print(data_df.ix[:])
print(data_df.ix[:,:])
print(data_df.ix[data_df.Year >= 2014])
```

    A
    A
    D
        Name  Year
    one    A  2011
    two    B  2016
        Name  Year  Gender
    one    A  2011    Male
    two    B  2016  Female
          Name  Year
    one      A  2011
    two      B  2016
    three    C  2015
          Name  Year  Gender
    one      A  2011    Male
    two      B  2016  Female
    three    C  2015    Male
    four     D  2015    Male
          Name  Year  Gender
    one      A  2011    Male
    two      B  2016  Female
    three    C  2015    Male
    four     D  2015    Male
          Name  Year  Gender
    two      B  2016  Female
    three    C  2015    Male
    four     D  2015    Male


    /usr/local/lib/python3.7/site-packages/ipykernel_launcher.py:1: DeprecationWarning:
    .ix is deprecated. Please use
    .loc for label based indexing or
    .iloc for positional indexing

    See the documentation here:
    http://pandas.pydata.org/pandas-docs/stable/indexing.html#ix-indexer-is-deprecated
      """Entry point for launching an IPython kernel.
    /usr/local/lib/python3.7/site-packages/ipykernel_launcher.py:2: DeprecationWarning:
    .ix is deprecated. Please use
    .loc for label based indexing or
    .iloc for positional indexing

    See the documentation here:
    http://pandas.pydata.org/pandas-docs/stable/indexing.html#ix-indexer-is-deprecated

    /usr/local/lib/python3.7/site-packages/ipykernel_launcher.py:3: DeprecationWarning:
    .ix is deprecated. Please use
    .loc for label based indexing or
    .iloc for positional indexing

    See the documentation here:
    http://pandas.pydata.org/pandas-docs/stable/indexing.html#ix-indexer-is-deprecated
      This is separate from the ipykernel package so we can avoid doing imports until
    /usr/local/lib/python3.7/site-packages/ipykernel_launcher.py:4: DeprecationWarning:
    .ix is deprecated. Please use
    .loc for label based indexing or
    .iloc for positional indexing

    See the documentation here:
    http://pandas.pydata.org/pandas-docs/stable/indexing.html#ix-indexer-is-deprecated
      after removing the cwd from sys.path.
    /usr/local/lib/python3.7/site-packages/ipykernel_launcher.py:5: DeprecationWarning:
    .ix is deprecated. Please use
    .loc for label based indexing or
    .iloc for positional indexing

    See the documentation here:
    http://pandas.pydata.org/pandas-docs/stable/indexing.html#ix-indexer-is-deprecated
      """
    /usr/local/lib/python3.7/site-packages/ipykernel_launcher.py:6: DeprecationWarning:
    .ix is deprecated. Please use
    .loc for label based indexing or
    .iloc for positional indexing

    See the documentation here:
    http://pandas.pydata.org/pandas-docs/stable/indexing.html#ix-indexer-is-deprecated

    /usr/local/lib/python3.7/site-packages/ipykernel_launcher.py:7: DeprecationWarning:
    .ix is deprecated. Please use
    .loc for label based indexing or
    .iloc for positional indexing

    See the documentation here:
    http://pandas.pydata.org/pandas-docs/stable/indexing.html#ix-indexer-is-deprecated
      import sys
    /usr/local/lib/python3.7/site-packages/ipykernel_launcher.py:8: DeprecationWarning:
    .ix is deprecated. Please use
    .loc for label based indexing or
    .iloc for positional indexing

    See the documentation here:
    http://pandas.pydata.org/pandas-docs/stable/indexing.html#ix-indexer-is-deprecated

    /usr/local/lib/python3.7/site-packages/ipykernel_launcher.py:9: DeprecationWarning:
    .ix is deprecated. Please use
    .loc for label based indexing or
    .iloc for positional indexing

    See the documentation here:
    http://pandas.pydata.org/pandas-docs/stable/indexing.html#ix-indexer-is-deprecated
      if __name__ == '__main__':



```python
# 명칭 기반 인덱싱과 위치 기반 인덱싱의 구분
# data_df를 reset_index()로 새로운 숫자형 인덱스 생성
data_df_reset = data_df.reset_index()
data_df_reset = data_df_reset.rename(columns={'index':'old_index'}) # 하나만 칼럼명 변경시 딕셔너리 쓴다.

# 인덱스값에 1을 더해서 1부터 시작하는 새로운 인덱스값 생성
data_df_reset.index = data_df_reset.index+1
data_df_reset
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>old_index</th>
      <th>Name</th>
      <th>Year</th>
      <th>Gender</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>one</td>
      <td>A</td>
      <td>2011</td>
      <td>Male</td>
    </tr>
    <tr>
      <th>2</th>
      <td>two</td>
      <td>B</td>
      <td>2016</td>
      <td>Female</td>
    </tr>
    <tr>
      <th>3</th>
      <td>three</td>
      <td>C</td>
      <td>2015</td>
      <td>Male</td>
    </tr>
    <tr>
      <th>4</th>
      <td>four</td>
      <td>D</td>
      <td>2015</td>
      <td>Male</td>
    </tr>
  </tbody>
</table>
</div>




```python
data_df_reset.ix[1,1]
```

    /usr/local/lib/python3.7/site-packages/ipykernel_launcher.py:1: DeprecationWarning:
    .ix is deprecated. Please use
    .loc for label based indexing or
    .iloc for positional indexing

    See the documentation here:
    http://pandas.pydata.org/pandas-docs/stable/indexing.html#ix-indexer-is-deprecated
      """Entry point for launching an IPython kernel.





    'A'



* DataFrame의 인덱스 값은 명칭 기반 인덱싱으로 간주한다. integer 형 인덱싱은 혼선 가능하므로 분리한다.


```python
# DataFrame iloc[] 연산자 : 위치 기반 연산자
data_df.iloc[0,0]
```




    'A'




```python
# iloc()에 명칭 입력시 오류 발생
data_df.iloc[0, 'Name']
```


    ---------------------------------------------------------------------------

    ValueError                                Traceback (most recent call last)

    /usr/local/lib/python3.7/site-packages/pandas/core/indexing.py in _has_valid_tuple(self, key)
        222             try:
    --> 223                 self._validate_key(k, i)
        224             except ValueError:


    /usr/local/lib/python3.7/site-packages/pandas/core/indexing.py in _validate_key(self, key, axis)
       2083             raise ValueError("Can only index by location with "
    -> 2084                              "a [{types}]".format(types=self._valid_types))
       2085


    ValueError: Can only index by location with a [integer, integer slice (START point is INCLUDED, END point is EXCLUDED), listlike of integers, boolean array]


    During handling of the above exception, another exception occurred:


    ValueError                                Traceback (most recent call last)

    <ipython-input-184-3808ccfc9449> in <module>
          1 # iloc()에 명칭 입력시 오류 발생
    ----> 2 data_df.iloc[0, 'Name']


    /usr/local/lib/python3.7/site-packages/pandas/core/indexing.py in __getitem__(self, key)
       1492             except (KeyError, IndexError, AttributeError):
       1493                 pass
    -> 1494             return self._getitem_tuple(key)
       1495         else:
       1496             # we by definition only have the 0th axis


    /usr/local/lib/python3.7/site-packages/pandas/core/indexing.py in _getitem_tuple(self, tup)
       2141     def _getitem_tuple(self, tup):
       2142
    -> 2143         self._has_valid_tuple(tup)
       2144         try:
       2145             return self._getitem_lowerdim(tup)


    /usr/local/lib/python3.7/site-packages/pandas/core/indexing.py in _has_valid_tuple(self, key)
        225                 raise ValueError("Location based indexing can only have "
        226                                  "[{types}] types"
    --> 227                                  .format(types=self._valid_types))
        228
        229     def _is_nested_tuple_indexer(self, tup):


    ValueError: Location based indexing can only have [integer, integer slice (START point is INCLUDED, END point is EXCLUDED), listlike of integers, boolean array] types



```python
# 문자열 인덱스를 행 위치에 입력해도 오류 발생
data_df.iloc['one', 0]
```


    ---------------------------------------------------------------------------

    ValueError                                Traceback (most recent call last)

    /usr/local/lib/python3.7/site-packages/pandas/core/indexing.py in _has_valid_tuple(self, key)
        222             try:
    --> 223                 self._validate_key(k, i)
        224             except ValueError:


    /usr/local/lib/python3.7/site-packages/pandas/core/indexing.py in _validate_key(self, key, axis)
       2083             raise ValueError("Can only index by location with "
    -> 2084                              "a [{types}]".format(types=self._valid_types))
       2085


    ValueError: Can only index by location with a [integer, integer slice (START point is INCLUDED, END point is EXCLUDED), listlike of integers, boolean array]


    During handling of the above exception, another exception occurred:


    ValueError                                Traceback (most recent call last)

    <ipython-input-185-3fa5bc58d2bd> in <module>
          1 # 문자열 인덱스를 행 위치에 입력해도 오류 발생
    ----> 2 data_df.iloc['one', 0]


    /usr/local/lib/python3.7/site-packages/pandas/core/indexing.py in __getitem__(self, key)
       1492             except (KeyError, IndexError, AttributeError):
       1493                 pass
    -> 1494             return self._getitem_tuple(key)
       1495         else:
       1496             # we by definition only have the 0th axis


    /usr/local/lib/python3.7/site-packages/pandas/core/indexing.py in _getitem_tuple(self, tup)
       2141     def _getitem_tuple(self, tup):
       2142
    -> 2143         self._has_valid_tuple(tup)
       2144         try:
       2145             return self._getitem_lowerdim(tup)


    /usr/local/lib/python3.7/site-packages/pandas/core/indexing.py in _has_valid_tuple(self, key)
        225                 raise ValueError("Location based indexing can only have "
        226                                  "[{types}] types"
    --> 227                                  .format(types=self._valid_types))
        228
        229     def _is_nested_tuple_indexer(self, tup):


    ValueError: Location based indexing can only have [integer, integer slice (START point is INCLUDED, END point is EXCLUDED), listlike of integers, boolean array] types



```python
# iloc은 숫자로만 받아요.
data_df_reset.iloc[0, 1]
```




    'A'




```python
# DataFrame loc[] 연산자 : 명칭 기반 데이터 추출
data_df.loc['one','Name'] # 띄어쓰기 하지 않는다. loc[]
```




    'A'




```python
# DataFrame loc[] 연산자에서 만약 인덱스가 integer라면 숫자가 명칭이므로 이를 입력한다.
data_df_reset.loc[1, 'Name']
```




    'A'




```python
# loc[]에서 0인 숫자 인덱스 없으면 사용할 수 없다.
data_df.loc[0,'Name'] # 띄어쓰기 하지 않는다. loc[]
```


    ---------------------------------------------------------------------------

    TypeError                                 Traceback (most recent call last)

    <ipython-input-196-f069a8fce6c4> in <module>
          1 # loc[]에서 0인 숫자 인덱스 없으면 사용할 수 없다.
    ----> 2 data_df.loc[0,'Name'] # 띄어쓰기 하지 않는다. loc[]


    /usr/local/lib/python3.7/site-packages/pandas/core/indexing.py in __getitem__(self, key)
       1492             except (KeyError, IndexError, AttributeError):
       1493                 pass
    -> 1494             return self._getitem_tuple(key)
       1495         else:
       1496             # we by definition only have the 0th axis


    /usr/local/lib/python3.7/site-packages/pandas/core/indexing.py in _getitem_tuple(self, tup)
        866     def _getitem_tuple(self, tup):
        867         try:
    --> 868             return self._getitem_lowerdim(tup)
        869         except IndexingError:
        870             pass


    /usr/local/lib/python3.7/site-packages/pandas/core/indexing.py in _getitem_lowerdim(self, tup)
        986         for i, key in enumerate(tup):
        987             if is_label_like(key) or isinstance(key, tuple):
    --> 988                 section = self._getitem_axis(key, axis=i)
        989
        990                 # we have yielded a scalar ?


    /usr/local/lib/python3.7/site-packages/pandas/core/indexing.py in _getitem_axis(self, key, axis)
       1910
       1911         # fall thru to straight lookup
    -> 1912         self._validate_key(key, axis)
       1913         return self._get_label(key, axis=axis)
       1914


    /usr/local/lib/python3.7/site-packages/pandas/core/indexing.py in _validate_key(self, key, axis)
       1797
       1798         if not is_list_like_indexer(key):
    -> 1799             self._convert_scalar_indexer(key, axis)
       1800
       1801     def _is_scalar_access(self, key):


    /usr/local/lib/python3.7/site-packages/pandas/core/indexing.py in _convert_scalar_indexer(self, key, axis)
        260         ax = self.obj._get_axis(min(axis, self.ndim - 1))
        261         # a scalar
    --> 262         return ax._convert_scalar_indexer(key, kind=self.name)
        263
        264     def _convert_slice_indexer(self, key, axis):


    /usr/local/lib/python3.7/site-packages/pandas/core/indexes/base.py in _convert_scalar_indexer(self, key, kind)
       2879             elif kind in ['loc'] and is_integer(key):
       2880                 if not self.holds_integer():
    -> 2881                     return self._invalid_indexer('label', key)
       2882
       2883         return key


    /usr/local/lib/python3.7/site-packages/pandas/core/indexes/base.py in _invalid_indexer(self, form, key)
       3065                         "indexers [{key}] of {kind}".format(
       3066                             form=form, klass=type(self), key=key,
    -> 3067                             kind=type(key)))
       3068
       3069     # --------------------------------------------------------------------


    TypeError: cannot do label indexing on <class 'pandas.core.indexes.base.Index'> with these indexers [0] of <class 'int'>



```python
print('명칭 기반 ix slicing\n', data_df.ix['one': 'two', 'Name'], '\n') # 이런 짓도 가능하다.
print('위치 기반 iloc slicing\n', data_df.iloc[0:1, 0], '\n') # 이런 짓도 가능하다.
print('명칭 기반 loc slicing\n', data_df.loc['one': 'two', 'Name'], '\n') # 이런 짓도 가능하다.
```

    명칭 기반 ix slicing
     one    A
    two    B
    Name: Name, dtype: object

    위치 기반 iloc slicing
     one    A
    Name: Name, dtype: object

    명칭 기반 loc slicing
     one    A
    two    B
    Name: Name, dtype: object



    /usr/local/lib/python3.7/site-packages/ipykernel_launcher.py:1: DeprecationWarning:
    .ix is deprecated. Please use
    .loc for label based indexing or
    .iloc for positional indexing

    See the documentation here:
    http://pandas.pydata.org/pandas-docs/stable/indexing.html#ix-indexer-is-deprecated
      """Entry point for launching an IPython kernel.



```python
# 숫자형 인덱싱이라면 명칭 기반 인덱싱에서 주의해야 한다.
print(data_df_reset.loc[1:2, 'Name'])
```

    1    A
    2    B
    Name: Name, dtype: object



```python
# ix[]에 위치 기반 인덱싱이 슬라이싱되면 1:2는 1개 데이터만 반환한다.
print(data_df.ix[1:2, 'Name']) # ix[]는 숫자형 인덱스라면 명칭에 우선한다.
```

    two    B
    Name: Name, dtype: object


    /usr/local/lib/python3.7/site-packages/ipykernel_launcher.py:2: DeprecationWarning:
    .ix is deprecated. Please use
    .loc for label based indexing or
    .iloc for positional indexing

    See the documentation here:
    http://pandas.pydata.org/pandas-docs/stable/indexing.html#ix-indexer-is-deprecated



* 명칭 기반 인덱싱 슬라이싱 '시작점:종료점' 지정시, 시작점에서 종료점을 포함한 위치에 있는 데이터 반환


```python
# 불린 인덱싱 : [], ix[], loc[] 모두 지원하고, iloc[]는 지원하지 않는다.
titanic_df = pd.read_csv('titanic_train.csv')

# 60세 이상만 추출한다.
titanic_boolean = titanic_df[titanic_df['Age'] > 60]
print(type(titanic_boolean))
titanic_boolean
```

    <class 'pandas.core.frame.DataFrame'>





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>33</th>
      <td>34</td>
      <td>0</td>
      <td>2</td>
      <td>Wheadon, Mr...</td>
      <td>male</td>
      <td>66.0</td>
      <td>0</td>
      <td>0</td>
      <td>C.A. 24579</td>
      <td>10.5000</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>54</th>
      <td>55</td>
      <td>0</td>
      <td>1</td>
      <td>Ostby, Mr. ...</td>
      <td>male</td>
      <td>65.0</td>
      <td>0</td>
      <td>1</td>
      <td>113509</td>
      <td>61.9792</td>
      <td>B30</td>
      <td>C</td>
    </tr>
    <tr>
      <th>96</th>
      <td>97</td>
      <td>0</td>
      <td>1</td>
      <td>Goldschmidt...</td>
      <td>male</td>
      <td>71.0</td>
      <td>0</td>
      <td>0</td>
      <td>PC 17754</td>
      <td>34.6542</td>
      <td>A5</td>
      <td>C</td>
    </tr>
    <tr>
      <th>116</th>
      <td>117</td>
      <td>0</td>
      <td>3</td>
      <td>Connors, Mr...</td>
      <td>male</td>
      <td>70.5</td>
      <td>0</td>
      <td>0</td>
      <td>370369</td>
      <td>7.7500</td>
      <td>NaN</td>
      <td>Q</td>
    </tr>
    <tr>
      <th>170</th>
      <td>171</td>
      <td>0</td>
      <td>1</td>
      <td>Van der hoe...</td>
      <td>male</td>
      <td>61.0</td>
      <td>0</td>
      <td>0</td>
      <td>111240</td>
      <td>33.5000</td>
      <td>B19</td>
      <td>S</td>
    </tr>
    <tr>
      <th>252</th>
      <td>253</td>
      <td>0</td>
      <td>1</td>
      <td>Stead, Mr. ...</td>
      <td>male</td>
      <td>62.0</td>
      <td>0</td>
      <td>0</td>
      <td>113514</td>
      <td>26.5500</td>
      <td>C87</td>
      <td>S</td>
    </tr>
    <tr>
      <th>275</th>
      <td>276</td>
      <td>1</td>
      <td>1</td>
      <td>Andrews, Mi...</td>
      <td>female</td>
      <td>63.0</td>
      <td>1</td>
      <td>0</td>
      <td>13502</td>
      <td>77.9583</td>
      <td>D7</td>
      <td>S</td>
    </tr>
    <tr>
      <th>280</th>
      <td>281</td>
      <td>0</td>
      <td>3</td>
      <td>Duane, Mr. ...</td>
      <td>male</td>
      <td>65.0</td>
      <td>0</td>
      <td>0</td>
      <td>336439</td>
      <td>7.7500</td>
      <td>NaN</td>
      <td>Q</td>
    </tr>
    <tr>
      <th>326</th>
      <td>327</td>
      <td>0</td>
      <td>3</td>
      <td>Nysveen, Mr...</td>
      <td>male</td>
      <td>61.0</td>
      <td>0</td>
      <td>0</td>
      <td>345364</td>
      <td>6.2375</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>438</th>
      <td>439</td>
      <td>0</td>
      <td>1</td>
      <td>Fortune, Mr...</td>
      <td>male</td>
      <td>64.0</td>
      <td>1</td>
      <td>4</td>
      <td>19950</td>
      <td>263.0000</td>
      <td>C23 C25 C27</td>
      <td>S</td>
    </tr>
    <tr>
      <th>456</th>
      <td>457</td>
      <td>0</td>
      <td>1</td>
      <td>Millet, Mr....</td>
      <td>male</td>
      <td>65.0</td>
      <td>0</td>
      <td>0</td>
      <td>13509</td>
      <td>26.5500</td>
      <td>E38</td>
      <td>S</td>
    </tr>
    <tr>
      <th>483</th>
      <td>484</td>
      <td>1</td>
      <td>3</td>
      <td>Turkula, Mr...</td>
      <td>female</td>
      <td>63.0</td>
      <td>0</td>
      <td>0</td>
      <td>4134</td>
      <td>9.5875</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>493</th>
      <td>494</td>
      <td>0</td>
      <td>1</td>
      <td>Artagaveyti...</td>
      <td>male</td>
      <td>71.0</td>
      <td>0</td>
      <td>0</td>
      <td>PC 17609</td>
      <td>49.5042</td>
      <td>NaN</td>
      <td>C</td>
    </tr>
    <tr>
      <th>545</th>
      <td>546</td>
      <td>0</td>
      <td>1</td>
      <td>Nicholson, ...</td>
      <td>male</td>
      <td>64.0</td>
      <td>0</td>
      <td>0</td>
      <td>693</td>
      <td>26.0000</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>555</th>
      <td>556</td>
      <td>0</td>
      <td>1</td>
      <td>Wright, Mr....</td>
      <td>male</td>
      <td>62.0</td>
      <td>0</td>
      <td>0</td>
      <td>113807</td>
      <td>26.5500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>570</th>
      <td>571</td>
      <td>1</td>
      <td>2</td>
      <td>Harris, Mr....</td>
      <td>male</td>
      <td>62.0</td>
      <td>0</td>
      <td>0</td>
      <td>S.W./PP 752</td>
      <td>10.5000</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>625</th>
      <td>626</td>
      <td>0</td>
      <td>1</td>
      <td>Sutton, Mr....</td>
      <td>male</td>
      <td>61.0</td>
      <td>0</td>
      <td>0</td>
      <td>36963</td>
      <td>32.3208</td>
      <td>D50</td>
      <td>S</td>
    </tr>
    <tr>
      <th>630</th>
      <td>631</td>
      <td>1</td>
      <td>1</td>
      <td>Barkworth, ...</td>
      <td>male</td>
      <td>80.0</td>
      <td>0</td>
      <td>0</td>
      <td>27042</td>
      <td>30.0000</td>
      <td>A23</td>
      <td>S</td>
    </tr>
    <tr>
      <th>672</th>
      <td>673</td>
      <td>0</td>
      <td>2</td>
      <td>Mitchell, M...</td>
      <td>male</td>
      <td>70.0</td>
      <td>0</td>
      <td>0</td>
      <td>C.A. 24580</td>
      <td>10.5000</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>745</th>
      <td>746</td>
      <td>0</td>
      <td>1</td>
      <td>Crosby, Cap...</td>
      <td>male</td>
      <td>70.0</td>
      <td>1</td>
      <td>1</td>
      <td>WE/P 5735</td>
      <td>71.0000</td>
      <td>B22</td>
      <td>S</td>
    </tr>
    <tr>
      <th>829</th>
      <td>830</td>
      <td>1</td>
      <td>1</td>
      <td>Stone, Mrs....</td>
      <td>female</td>
      <td>62.0</td>
      <td>0</td>
      <td>0</td>
      <td>113572</td>
      <td>80.0000</td>
      <td>B28</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>851</th>
      <td>852</td>
      <td>0</td>
      <td>3</td>
      <td>Svensson, M...</td>
      <td>male</td>
      <td>74.0</td>
      <td>0</td>
      <td>0</td>
      <td>347060</td>
      <td>7.7750</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 두 개 이상 칼럼으로 60세 이상인 승객의 나이와 이름만 추출
titanic_df[titanic_df['Age'] > 60][['Name','Age']].head()
# 첫 번째 조건은 [] 사용하고 두번째는 칼럼 명이므로 []연산자 내부에 []칼럼 리스트로 또다시 들어간다.
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Name</th>
      <th>Age</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>33</th>
      <td>Wheadon, Mr...</td>
      <td>66.0</td>
    </tr>
    <tr>
      <th>54</th>
      <td>Ostby, Mr. ...</td>
      <td>65.0</td>
    </tr>
    <tr>
      <th>96</th>
      <td>Goldschmidt...</td>
      <td>71.0</td>
    </tr>
    <tr>
      <th>116</th>
      <td>Connors, Mr...</td>
      <td>70.5</td>
    </tr>
    <tr>
      <th>170</th>
      <td>Van der hoe...</td>
      <td>61.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
titanic_df.loc[titanic_df['Age']>60, ['Name', 'Age']].head(3)
# 명칭 기반 칼럼 인덱싱은 loc 함수로 콤마로 조건을 덧 붙일 수 있다.
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Name</th>
      <th>Age</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>33</th>
      <td>Wheadon, Mr...</td>
      <td>66.0</td>
    </tr>
    <tr>
      <th>54</th>
      <td>Ostby, Mr. ...</td>
      <td>65.0</td>
    </tr>
    <tr>
      <th>96</th>
      <td>Goldschmidt...</td>
      <td>71.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 복합 조건도 가능하다.
# and &, or |, Not ~ 으로 부여할 수 있다.

cond1 = titanic_df['Age'] > 60
cond2 = titanic_df['Pclass']==1
cond3 = titanic_df['Sex']=='female'
titanic_df[cond1 & cond2 & cond3]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>275</th>
      <td>276</td>
      <td>1</td>
      <td>1</td>
      <td>Andrews, Mi...</td>
      <td>female</td>
      <td>63.0</td>
      <td>1</td>
      <td>0</td>
      <td>13502</td>
      <td>77.9583</td>
      <td>D7</td>
      <td>S</td>
    </tr>
    <tr>
      <th>829</th>
      <td>830</td>
      <td>1</td>
      <td>1</td>
      <td>Stone, Mrs....</td>
      <td>female</td>
      <td>62.0</td>
      <td>0</td>
      <td>0</td>
      <td>113572</td>
      <td>80.0000</td>
      <td>B28</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
titanic_df.loc[cond1, ['Name','Age']].head(3)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Name</th>
      <th>Age</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>33</th>
      <td>Wheadon, Mr...</td>
      <td>66.0</td>
    </tr>
    <tr>
      <th>54</th>
      <td>Ostby, Mr. ...</td>
      <td>65.0</td>
    </tr>
    <tr>
      <th>96</th>
      <td>Goldschmidt...</td>
      <td>71.0</td>
    </tr>
  </tbody>
</table>
</div>



## 정렬, Aggregation 함수, GroupBy 적용


```python
# DataFrame, Series 정렬 - sort_values()
# 기본 오름차순, ascendint=True, inplace=False
titanic_sorted = titanic_df.sort_values(by=['Name'])
titanic_sorted.head(3)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>845</th>
      <td>846</td>
      <td>0</td>
      <td>3</td>
      <td>Abbing, Mr....</td>
      <td>male</td>
      <td>42.0</td>
      <td>0</td>
      <td>0</td>
      <td>C.A. 5547</td>
      <td>7.55</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>746</th>
      <td>747</td>
      <td>0</td>
      <td>3</td>
      <td>Abbott, Mr....</td>
      <td>male</td>
      <td>16.0</td>
      <td>1</td>
      <td>1</td>
      <td>C.A. 2673</td>
      <td>20.25</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>279</th>
      <td>280</td>
      <td>1</td>
      <td>3</td>
      <td>Abbott, Mrs...</td>
      <td>female</td>
      <td>35.0</td>
      <td>1</td>
      <td>1</td>
      <td>C.A. 2673</td>
      <td>20.25</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 여러 칼럼으로 정렬은 by 뒤에 리스트로 입력해주기
# Pclass, Name 내림차순 정렬

titanic_sorted = titanic_df.sort_values(by=['Pclass','Name'], ascending=False)
titanic_sorted.head(3)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>868</th>
      <td>869</td>
      <td>0</td>
      <td>3</td>
      <td>van Melkebe...</td>
      <td>male</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>345777</td>
      <td>9.5</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>153</th>
      <td>154</td>
      <td>0</td>
      <td>3</td>
      <td>van Billiar...</td>
      <td>male</td>
      <td>40.5</td>
      <td>0</td>
      <td>2</td>
      <td>A/5. 851</td>
      <td>14.5</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>282</th>
      <td>283</td>
      <td>0</td>
      <td>3</td>
      <td>de Pelsmaek...</td>
      <td>male</td>
      <td>16.0</td>
      <td>0</td>
      <td>0</td>
      <td>345778</td>
      <td>9.5</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Aggregation 함수 적용 : min(), max(), sum(), count() 함수
# count() : 모든 칼럼 적용
titanic_df.count()
```




    PassengerId    891
    Survived       891
    Pclass         891
    Name           891
    Sex            891
    Age            714
    SibSp          891
    Parch          891
    Ticket         891
    Fare           891
    Cabin          204
    Embarked       889
    dtype: int64




```python
# 특정 함수에 aggregation 함수 적용 : 칼럼 표기 후 적용(닷 기반)
titanic_df[['Age','Fare']].mean()
```




    Age     29.699118
    Fare    32.204208
    dtype: float64




```python
# groupby() 적용 : DataFrameGroupBy 형태의 데이터프레임 반환
titanic_groupby = titanic_df.groupby(by='Pclass')
print(type(titanic_groupby))
```

    <class 'pandas.core.groupby.generic.DataFrameGroupBy'>



```python
# groupby 적용시 대상 칼럼을 제외한 모든 칼럼에 aggregation 함수를 적용한다.
titanic_groupby = titanic_df.groupby(by='Pclass').count()
titanic_groupby.head(3)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
    </tr>
    <tr>
      <th>Pclass</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>216</td>
      <td>216</td>
      <td>216</td>
      <td>216</td>
      <td>186</td>
      <td>216</td>
      <td>216</td>
      <td>216</td>
      <td>216</td>
      <td>176</td>
      <td>214</td>
    </tr>
    <tr>
      <th>2</th>
      <td>184</td>
      <td>184</td>
      <td>184</td>
      <td>184</td>
      <td>173</td>
      <td>184</td>
      <td>184</td>
      <td>184</td>
      <td>184</td>
      <td>16</td>
      <td>184</td>
    </tr>
    <tr>
      <th>3</th>
      <td>491</td>
      <td>491</td>
      <td>491</td>
      <td>491</td>
      <td>355</td>
      <td>491</td>
      <td>491</td>
      <td>491</td>
      <td>491</td>
      <td>12</td>
      <td>491</td>
    </tr>
  </tbody>
</table>
</div>




```python
# groupby 적용시 대상 칼럼을 제외한 특정 칼럼에 aggregation 함수를 적용한다.
titanic_groupby = titanic_df.groupby(by='Pclass')[['PassengerId', 'Survived']].count()
titanic_groupby.head(3)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
    </tr>
    <tr>
      <th>Pclass</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>216</td>
      <td>216</td>
    </tr>
    <tr>
      <th>2</th>
      <td>184</td>
      <td>184</td>
    </tr>
    <tr>
      <th>3</th>
      <td>491</td>
      <td>491</td>
    </tr>
  </tbody>
</table>
</div>




```python
# groupby후 여러 개의 aggregation 함수 적용시 agg(인자 안에 리스트로 함수 넣어준다.)
titanic_df.groupby('Pclass')['Age'].agg([max,min])
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>max</th>
      <th>min</th>
    </tr>
    <tr>
      <th>Pclass</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>80.0</td>
      <td>0.92</td>
    </tr>
    <tr>
      <th>2</th>
      <td>70.0</td>
      <td>0.67</td>
    </tr>
    <tr>
      <th>3</th>
      <td>74.0</td>
      <td>0.42</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 각 다른 칼럼에 다른 aggregation 함수 적용할 때는 딕셔너리 형태로 입력 후 이를 agg() 인자로 넣는다.
agg_format = {'Age':'max','SibSp':'sum','Fare':'mean'}
titanic_df.groupby('Pclass').agg(agg_format)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Fare</th>
    </tr>
    <tr>
      <th>Pclass</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>80.0</td>
      <td>90</td>
      <td>84.154687</td>
    </tr>
    <tr>
      <th>2</th>
      <td>70.0</td>
      <td>74</td>
      <td>20.662183</td>
    </tr>
    <tr>
      <th>3</th>
      <td>74.0</td>
      <td>302</td>
      <td>13.675550</td>
    </tr>
  </tbody>
</table>
</div>



## 결손 데이터 처리하기
* 데이터 값이 없는, NULL인 경우로, 넘파이의 NaN으로 표기한다. 이를 처리하지 않으므로 다른 값으로 대체한다.
* 평균, 총합 등 함수 연산시 제외가 된다.
* NaN 여부 확인하는 API는 isna()이고, NaN 값을 다른 값으로 대체하는 API는 fillna()이다.


```python
# isna()로 결측값 확인(True, False)
titanic_df.isna().head(3)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 결손 데이터 수는 isna().sum() 쓴다.
titanic_df.isna().sum()
```




    PassengerId      0
    Survived         0
    Pclass           0
    Name             0
    Sex              0
    Age            177
    SibSp            0
    Parch            0
    Ticket           0
    Fare             0
    Cabin          687
    Embarked         2
    dtype: int64




```python
# fillna()로 결손 데이터 대체하기
# 'Cabin' 칼럼의 NaN을 'C000' 대체
titanic_df['Cabin'] = titanic_df['Cabin'].fillna('C000')
# 위처럼 지정하거나 inplace=True를 fillna() 파라미터로 추가해야 원본 데이터에도 변경이 반영된다.

titanic_df.head(3)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>Braund, Mr....</td>
      <td>male</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>A/5 21171</td>
      <td>7.2500</td>
      <td>C000</td>
      <td>S</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>Cumings, Mr...</td>
      <td>female</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>PC 17599</td>
      <td>71.2833</td>
      <td>C85</td>
      <td>C</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>Heikkinen, ...</td>
      <td>female</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>STON/O2. 31...</td>
      <td>7.9250</td>
      <td>C000</td>
      <td>S</td>
    </tr>
  </tbody>
</table>
</div>




```python
titanic_df['Age'] = titanic_df['Age'].fillna(titanic_df['Age'].mean())
titanic_df['Embarked'] = titanic_df['Embarked'].fillna('S')
titanic_df.isna().sum()
```




    PassengerId    0
    Survived       0
    Pclass         0
    Name           0
    Sex            0
    Age            0
    SibSp          0
    Parch          0
    Ticket         0
    Fare           0
    Cabin          0
    Embarked       0
    dtype: int64



### apply lambda 식으로 데이터 가공
* 판다스는 apply 함수에 lambda 식을 결합해 데이터 가공 기능 제공
* 복잡한 데이터 가공시 주로 사용한다. 함수형 프로그래밍 지원을 위해 lambda 식 제작


```python
# 입력값의 제곱값 반환
def get_square(a):
    return a**2

print('3의 제곱은:', get_square(3))
```

    3의 제곱은: 9



```python
# 위 함수를 한줄로 lambda 변환
lambda_square = lambda x : x**2 # x는 입력 인자이고, ':'의 오른쪽은 반환 계산식이다.
print('3의 제곱은:', lambda_square(3))
```

    3의 제곱은: 9



```python
# lambda 식에서 여러 개 값을 입력 인자로 사용시 map() 함수를 활용한다.
a = [1,2,3]
squares = map(lambda x: x**2, a)
list(squares)
```




    [1, 4, 9]




```python
# 판다스 DataFrame의 lambda 식은 그대로 적용하되, apply에 적용하여 데이터 가공한다.
titanic_df['Name_len'] = titanic_df['Name'].apply(lambda x : len(x))
titanic_df[['Name', 'Name_len']].head(3)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Name</th>
      <th>Name_len</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Braund, Mr....</td>
      <td>23</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Cumings, Mr...</td>
      <td>51</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Heikkinen, ...</td>
      <td>22</td>
    </tr>
  </tbody>
</table>
</div>




```python
# lambda 식에서 if else 절로 복잡한 가공하기
# 15세 미만이면 Child, 아니면 Adult로 구분하는 Child_Adult를 만든다.
titanic_df['Child_Adult'] = titanic_df['Age'].apply(lambda x : 'Child' if x <= 15 else 'Adult')
titanic_df[['Age', 'Child_Adult']].head(8)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Age</th>
      <th>Child_Adult</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>22.000000</td>
      <td>Adult</td>
    </tr>
    <tr>
      <th>1</th>
      <td>38.000000</td>
      <td>Adult</td>
    </tr>
    <tr>
      <th>2</th>
      <td>26.000000</td>
      <td>Adult</td>
    </tr>
    <tr>
      <th>3</th>
      <td>35.000000</td>
      <td>Adult</td>
    </tr>
    <tr>
      <th>4</th>
      <td>35.000000</td>
      <td>Adult</td>
    </tr>
    <tr>
      <th>5</th>
      <td>29.699118</td>
      <td>Adult</td>
    </tr>
    <tr>
      <th>6</th>
      <td>54.000000</td>
      <td>Adult</td>
    </tr>
    <tr>
      <th>7</th>
      <td>2.000000</td>
      <td>Child</td>
    </tr>
  </tbody>
</table>
</div>



* 주의 사항
    * if 절에서 식보다 반호나 값을 먼저 기술해야 한다. 왜냐하면 ':' 기호 오른편에 반환 값이 먼저 나와줘야 하기 때문이다.
    * else if는 지원하지 않는다.
    * else 절을 ()로 포함해서 () 내에서 다시 if else 적용해야 가능하다.


```python
# 15세 이하 Child, 15 ~ 60세 Adult, 그 이상은 Elderly
titanic_df['Age_cat'] = titanic_df['Age'].apply(lambda x : 'Child' if x <= 15 else ('Adult' if x <= 60 else 'Elderly'))
titanic_df['Age_cat'].value_counts() # Series만 적용가능한 함수이다.
```




    Adult      786
    Child       83
    Elderly     22
    Name: Age_cat, dtype: int64




```python
# 더 세분화된 분류는 함수를 만드는 게 더 편하다.
def get_category(age):
    cat = ''
    if age <= 5: cat = 'Baby'
    elif age <= 12: cat = 'Child'
    elif age <= 18: cat = 'Teenager'
    elif age <= 25: cat = 'Student'
    elif age <= 35: cat = 'Young Adult'
    elif age <= 60: cat = 'Adult'
    else : cat = 'Elderly'

    return cat

# lambda 식에 위에서 생성한 get_category() 함수를 반환값 지정
# 입력값으로는 'Age' 칼럼을 받아서 해당하는 cat 반환

titanic_df['Age_cat'] = titanic_df['Age'].apply(lambda x: get_category(x))
titanic_df[['Age', 'Age_cat']].head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Age</th>
      <th>Age_cat</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>22.0</td>
      <td>Student</td>
    </tr>
    <tr>
      <th>1</th>
      <td>38.0</td>
      <td>Adult</td>
    </tr>
    <tr>
      <th>2</th>
      <td>26.0</td>
      <td>Young Adult</td>
    </tr>
    <tr>
      <th>3</th>
      <td>35.0</td>
      <td>Young Adult</td>
    </tr>
    <tr>
      <th>4</th>
      <td>35.0</td>
      <td>Young Adult</td>
    </tr>
  </tbody>
</table>
</div>
