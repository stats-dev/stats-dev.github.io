---
title: "Text Analysis and Kaggle"
date : 2020-03-18
tags : [data science, python, kaggle]
header:
  #image: "/images/fig-1-33.png"
excerpt: "Data Science, Python, Kaggle"
mathjax: "true"
toc: true
toc_label: "My Table of Contents"
toc_icon: "cog"
---
{% include toc %}

# 텍스트 분석 실습 - 캐글 Mercari Price

* 일본 대형 온라인 쇼핑몰 Mercari사의 제품에 대해 가격 예측 과제

## 데이터 전처리


```python
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

mercari_df = pd.read_csv('mercari_train.tsv', sep='\t')
print(mercari_df.shape)
mercari_df.head(3)
```

    (1482535, 8)





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
      <th>train_id</th>
      <th>name</th>
      <th>item_condition_id</th>
      <th>category_name</th>
      <th>brand_name</th>
      <th>price</th>
      <th>shipping</th>
      <th>item_description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>MLB Cincinnati Reds T Shirt Size XL</td>
      <td>3</td>
      <td>Men/Tops/T-shirts</td>
      <td>NaN</td>
      <td>10.0</td>
      <td>1</td>
      <td>No description yet</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>Razer BlackWidow Chroma Keyboard</td>
      <td>3</td>
      <td>Electronics/Computers &amp; Tablets/Components &amp; P...</td>
      <td>Razer</td>
      <td>52.0</td>
      <td>0</td>
      <td>This keyboard is in great condition and works ...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>AVA-VIV Blouse</td>
      <td>1</td>
      <td>Women/Tops &amp; Blouses/Blouse</td>
      <td>Target</td>
      <td>10.0</td>
      <td>1</td>
      <td>Adorable top with a hint of lace and a key hol...</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 피처의 타입과 결측값 여부 확인
print(mercari_df.info())
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 1482535 entries, 0 to 1482534
    Data columns (total 8 columns):
    train_id             1482535 non-null int64
    name                 1482535 non-null object
    item_condition_id    1482535 non-null int64
    category_name        1476208 non-null object
    brand_name           849853 non-null object
    price                1482535 non-null float64
    shipping             1482535 non-null int64
    item_description     1482531 non-null object
    dtypes: float64(1), int64(3), object(4)
    memory usage: 90.5+ MB
    None


* brand_name이 가장 많은 결측값을 가진다.
* category_name은 약 6300건의 결측값을 가지며, 기타 item_description도 결측치가 일부 있다.


```python
# price 칼럼 데이터 분포도
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

y_train_df = mercari_df['price']
plt.figure(figsize=(6, 4))
sns.distplot(y_train_df, kde=False)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x143735450>




![png](output_6_1.png)


* price 값이 0에 가까운 값에 많이 분포한다.(왜곡된 형태)
* 로그 변환을 통해 다시 분포도를 살펴본다.


```python
import numpy as np

y_train_df = np.log1p(mercari_df['price'])
sns.distplot(y_train_df, kde=False)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x12a4c5f90>




![png](output_8_1.png)


* 로그 변환을 통해 비교적 정규 분포에 가까운 분포를 가진다.


```python
mercari_df['price'] = np.log1p(mercari_df['price'])
mercari_df['price'].head(3)
```




    0    2.397895
    1    3.970292
    2    2.397895
    Name: price, dtype: float64




```python
print('Shipping 분포:\n', mercari_df['shipping'].value_counts())
print('\nitem_condition_id 값 분포:\n', mercari_df['item_condition_id'].value_counts())
```

    Shipping 분포:
     0    819435
    1    663100
    Name: shipping, dtype: int64

    item_condition_id 값 분포:
     1    640549
    3    432161
    2    375479
    4     31962
    5      2384
    Name: item_condition_id, dtype: int64



```python
# item_description 칼럼의 결측치 확인
boolean_cond = mercari_df['item_description']=='No description yet'
mercari_df[boolean_cond]['item_description'].count()
```




    82489



* 'No description yet'은 결측치이므로 적절한 값으로 변경해야 한다.
* category_name : 대분류/중분류/소분류로 구성되어있고, '/'기준으로 단어를 토큰화 -> 피처로 저장 -> 알고리즘 학습


```python
# 대분류, 중분류, 소분류 값을 분할 후 리스트로 변환한다.

def split_cat(category_name):
    try:
        return category_name.split('/')
    except:
        return ['non_Null', 'non_Null', 'non_Null']

# 위의 함수를 호출해서 각 칼럼을 데이터프레임에 추가
mercari_df['cat_large'], mercari_df['cat_med'], mercari_df['cat_small'] = \
    zip(*mercari_df['category_name'].apply(lambda x : split_cat(x)))

# 대분류 분포와 건수
print('대분류 분포 :\n', mercari_df['cat_large'].value_counts())
print(mercari_df['cat_med'].nunique(), mercari_df['cat_small'].nunique())
```

    대분류 분포 :
     Women                     664385
    Beauty                    207828
    Kids                      171689
    Electronics               122690
    Men                        93680
    Home                       67871
    Vintage & Collectibles     46530
    Other                      45351
    Handmade                   30842
    Sports & Outdoors          25342
    non_Null                    6327
    Name: cat_large, dtype: int64
    114 871


* nunique : number of unique로 생각한다.


```python
# 나머지 결측치는 일괄로 'non-Null'로 바꾼다.
mercari_df['brand_name'] = mercari_df['brand_name'].fillna(value='non-Null')
mercari_df['category_name'] = mercari_df['category_name'].fillna(value='non-Null')
mercari_df['item_description'] = mercari_df['item_description'].fillna(value='non-Null')

# 칼럼별 결측치 확인
mercari_df.isnull().sum()
```




    train_id             0
    name                 0
    item_condition_id    0
    category_name        0
    brand_name           0
    price                0
    shipping             0
    item_description     0
    cat_large            0
    cat_med              0
    cat_small            0
    dtype: int64



## 피처 인코딩과 피처 벡터화


```python
# 문자열 칼럼 중 레이블 혹은 원-핫 인코딩 수행하거나 피처 벡터화로 변환할 칼럼 선별
print('brand name의 종류 수 :', mercari_df['brand_name'].nunique())
print('brand name sample 5건 : \n', mercari_df['brand_name'].value_counts()[:5])
```

    brand name의 종류 수 : 4810
    brand name sample 5건 :
     non-Null             632682
    PINK                  54088
    Nike                  54043
    Victoria's Secret     48036
    LuLaRoe               31024
    Name: brand_name, dtype: int64


* 상품명 : Name 속성으로 종류가 매우 많고 적은 단어 위주 텍스트 형태 -> Count기반 피처 벡터화 변환
* category_name : 각각 대분류, 중분류, 소분류 칼럼으로 분리 형태 -> 원-핫 인코딩 적용
* shipping : 배송비 무료 여부, 0과 1 구성 -> 원-핫 인코딩
* item_condtion_id : 상품 상태, 다섯 가지 값(1,2,3,4,5) -> 원-핫 인코딩
* item_Description : 상품 설명, 가장 긴 텍스트 구성, 평균 문자열이 크다. -> TF-IDF 피처 벡터화 변환


```python
pd.set_option('max_colwidth', 200)

print('item_description 평균 문자열 크기:', round(mercari_df['item_description'].str.len().mean()))

mercari_df['item_description'][:2]
```

    item_description 평균 문자열 크기: 146.0





    0                                                                                                                                                                              No description yet
    1    This keyboard is in great condition and works like it came out of the box. All of the ports are tested and work perfectly. The lights are customizable via the Razer Synapse app on your PC.
    Name: item_description, dtype: object




```python
# name 피처 벡터화
cnt_vec = CountVectorizer()
X_name = cnt_vec.fit_transform(mercari_df.name)

# item_description에 대한 피처 벡터화 변환
tfidf_descp = TfidfVectorizer(max_features= 50000, ngram_range=(1, 3), stop_words='english')
X_descp = tfidf_descp.fit_transform(mercari_df['item_description'])

print('name vectorization shape:', X_name.shape)
print('item_description vectorization shape:', X_descp.shape)
```

    name vectorization shape: (1482535, 105757)
    item_description vectorization shape: (1482535, 50000)



```python
# 피처 데이터 세트를 희소 행렬로 변환 후 결합
from sklearn.preprocessing import LabelBinarizer

# 각 피처를 희소 행렬 : 원-핫 인코딩으로 변환
lb_brand_name = LabelBinarizer(sparse_output=True) # 희소행렬 형태의 원-핫 인코딩 변환
X_brand = lb_brand_name.fit_transform(mercari_df['brand_name']) # 희소 행렬 객체 변수로 적합
lb_item_cond_id = LabelBinarizer(sparse_output=True)
X_item_cond_id = lb_item_cond_id.fit_transform(mercari_df['item_condition_id'])
lb_shipping = LabelBinarizer(sparse_output=True)
X_shipping = lb_shipping.fit_transform(mercari_df['shipping'])

# 각 분류명도 모두 원-핫 인코딩 변환
lb_cat_large = LabelBinarizer(sparse_output=True)
X_cat_large = lb_cat_large.fit_transform(mercari_df['cat_large'])
lb_cat_med = LabelBinarizer(sparse_output=True)
X_cat_med = lb_cat_med.fit_transform(mercari_df['cat_med'])
lb_cat_small = LabelBinarizer(sparse_output=True)
X_cat_small = lb_cat_small.fit_transform(mercari_df['cat_small'])
```


```python
print(type(X_brand), type(X_item_cond_id), type(X_shipping))
print('X_brand shape:{0}, X_item_cond_id shape:{1}'.format(X_brand.shape, X_item_cond_id.shape))
print('X_shipping shape:{0}, X_cat_large shape:{1}'.format(X_shipping.shape, X_cat_large.shape))
print('X_cat_med shape:{0}, X_cat_small shape:{1}'.format(X_cat_med.shape, X_cat_small.shape))
```

    <class 'scipy.sparse.csr.csr_matrix'> <class 'scipy.sparse.csr.csr_matrix'> <class 'scipy.sparse.csr.csr_matrix'>
    X_brand shape:(1482535, 4810), X_item_cond_id shape:(1482535, 5)
    X_shipping shape:(1482535, 1), X_cat_large shape:(1482535, 11)
    X_cat_med shape:(1482535, 114), X_cat_small shape:(1482535, 871)



```python
# 앞에서 피처 벡터화 데이터 세트와 희소 인코딩 데이터 세트를 hstack()으로 결합
from scipy.sparse import hstack
import gc

sparse_matrix_list = (X_name, X_descp, X_brand, X_item_cond_id, \
                     X_shipping, X_cat_large, X_cat_med, X_cat_small)

# hstack 함수로 인코딩과 벡터화를 수행한 데이터 세트를 모두 결합
X_features_sparse = hstack(sparse_matrix_list).tocsr()
print(type(X_features_sparse), X_features_sparse.shape)

# 데이터 세트가 메모리 많이 차지하므로 사용 목적이 끝나면 메모리에서 삭제
del X_features_sparse
gc.collect() # gc.collect()로 결합 데이터를 즉시 삭제한다. 활용할 때마다 결합해 사용
```

    <class 'scipy.sparse.csr.csr_matrix'> (1482535, 161569)





    20



## 릿지 회귀 모델 구축 및 평가

* 평가 지표 : RMSLE(Root Mean Square Log Error)

$$\epsilon = \sqrt{{1 \over n} \sum_{i=1}^n (\log (p_i +1) - \log(a_i + 1))^2}$$


```python
# RMSLE 함수 생성
def rmsle(y, y_pred):
    # underflow, overflow 막고자 log1p로 rmsle 생성
    return np.sqrt(np.mean(np.power(np.log1p(y) - np.log1p(y_pred), 2)))

def evaluate_org_price(y_test, preds):

    # 원본 데이터를 log1p 변환하므로 원복시 exmpm1 활용한다.
    preds_exmpm = np.expm1(preds)
    y_test_exmpm = np.expm1(y_test)

    # rmsle 값 추출
    rmsle_result = rmsle(y_test_exmpm, preds_exmpm)
    return rmsle_result
```


```python
# 학습용 데이터 생성 및 모델을 학습/예측하는 로직을 별도 함수로 제작
import gc
from scipy.sparse import hstack

def model_train_predict(model, matrix_list):
    # hstack으로 희소 행렬 결합
    X = hstack(matrix_list).tocsr()

    X_train, X_test, y_train, y_test = train_test_split(X, mercari_df['price'],
                                                        test_size=0.2, random_state=156)
    # 모델 학습 및 예측
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    # 메모리 보전을 위해 즉시 삭제한다.
    del X, X_train, X_test, y_train
    gc.collect()

    return preds, y_test
```


```python
linear_model = Ridge(solver= "lsqr", fit_intercept=False)

sparse_matrix_list = (X_name, X_brand, X_item_cond_id, \
                     X_shipping, X_cat_large, X_cat_med, X_cat_small)
linear_preds, y_test = model_train_predict(model=linear_model, matrix_list=sparse_matrix_list)
print('Item Description을 제외한 rmsle 값:', evaluate_org_price(y_test, linear_preds))

sparse_matrix_list = (X_descp, X_name, X_brand, X_item_cond_id, \
                     X_shipping, X_cat_large, X_cat_med, X_cat_small)
linear_preds, y_test = model_train_predict(model=linear_model, matrix_list=sparse_matrix_list)
print('Item Description을 포함한 rmsle 값:', evaluate_org_price(y_test, linear_preds))
```

    Item Description을 제외한 rmsle 값: 0.5021629263070714
    Item Description을 포함한 rmsle 값: 0.47122146868833503


## LightGBM 회귀 모델 구축과 앙상블을 이용한 최종 예측 평가

* LGBM 회귀 수행 후, 위의 릿지 모델 예측값과 LGBM 모델 예측값을 앙상블 방식으로 섞어 최종 회귀 예측값 평가


```python
from lightgbm import LGBMRegressor

sparse_matrix_list = (X_descp, X_name, X_brand, X_item_cond_id,
                     X_shipping, X_cat_large, X_cat_med, X_cat_small)

lgbm_model = LGBMRegressor(n_estimators=200, learning_rate=0.5, num_leaves=125, random_state=156)
lgbm_preds, y_test = model_train_predict(model = lgbm_model, matrix_list=sparse_matrix_list)
print('LightGBM rmsle 값:', evaluate_org_price(y_test, lgbm_preds))
```

    LightGBM rmsle 값: 0.4565805991876516



```python
# Ridge 예측과 LGBM 예측 데이터 셋을 가중 평균 후 최종 예측 성능 결과
preds = lgbm_preds * 0.45 + linear_preds * 0.55
print('LightGBM과 Ridge를 ensemble한 최종 rmsle 값:', evaluate_org_price(y_test, preds))
```

    LightGBM과 Ridge를 ensemble한 최종 rmsle 값: 0.4503244956220032


* Simple Ensemble method를 통해 예측 성능을 더 개선할 수 있음을 알았다.
