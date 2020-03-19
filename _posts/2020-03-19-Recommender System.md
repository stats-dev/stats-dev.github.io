---
title: "Basic Recommender System"
date : 2020-03-19
tags : [data science, python, recommender system]
header:
  #image: "/images/fig-1-33.png"
excerpt: "Data Science, Python, Recommender System"
mathjax: "true"
toc: true
toc_label: "My Table of Contents"
toc_icon: "cog"
---
{% include toc %}
# 추천 시스템

## 추천 시스템의 개요와 배경

### 추천 시스템의 개요

* 사용자 자신도 몰랐던 취향을 시스템이 발견하고 이에 맞는 콘텐츠를 추천한다.
    * 사용자가 더 신뢰하고 더 많은 추천 콘텐츠를 감상하게 된다.
    * 사용자 데이터가 축적되고 더 정확하고 다양한 결과를 제공한다.

### 온라인 스토어의 필수 요소

* 온라인 스토어는 매우 많은 상품을 가진다.
    * 추천시스템은 한정된 시간 내 사용자의 선택 부담을 줄인다.

* 추천 시스템 구성에 사용되는 데이터
    * 어떤 상품을 구매했는가?
    * 어떤 제품을 클릭했는가?
    * 어떤 상품을 장바구니에 넣거나 둘러봤는가?
    * 스스로 작성한 취향은 무엇인가?
    * 제품을 클릭하고 머무른 시간은 어떤가?

* 추천 시스템은 위의 **데이터를 기반**으로 상품 구매를 유도한다.

### 추천 시스템의 유형

* 추천 시스템
    * 콘텐츠 기반 필터링(Content based filtering)
    * 협업 필터링(Collaborative Filtering)
        * 최근접 이웃(Nearest Neighbor) 협업 필터링
        * 잠재 요인(Latent Factor) 협업 필터링

## 콘텐츠 기반 필터링 추천 시스템

* 특정한 아이템을 매우 선호하면 그 아이템과 비슷한 콘텐츠를 가진 다른 아이템 추천
* 영화
    * 높게 평가한 영화의 컨텐츠 감안해서 적절히 매칭되는 영화 추천
    * 선호하는 감독, 장르, 키워드 등 컨텐츠 포함하므로, 영화 추천 가능

## 최근접 이웃 협업 필터링

* 협업 필터링 : 사용자가 아이템에 매긴 평점 정보나 상품 구매 이력과 같은 사용자 행동 양식만을 기반으로 추천
* 주요 목표
    * 사용자-아이템 평점 매트릭스 : 축적된 사용자 행동 데이터를 기반으로 아직 평가하지 않은 아이템 예측 평가(Predicted Rating)
* 사용자-아이템 평점 행렬
    * 최근접 이웃, 잠재 요인 방식 모두 활용
    * 행(Row) : 개별 사용자
    * 열(Column) : 개별 아이템
    * 각각의 값은 평점을 나타낸다.
    * 많은 아이템을 열로 하는 다차원 행렬
    * 평점을 안 매기는 경우가 많아서 희소 행렬(Sparse Matrix)

* 최근접 이웃 협업 필터링 = 메모리 협업 필터링
    * 사용자 기반 : 비슷한 고객이 다음 상품을 구매했다.
    * 아이템 기반 : 이 상품을 고른 고객은 다음 상품도 구매했다.

* 사용자 기반 최근접 이웃 방식
    * (다른) TOP-N 사용자가 좋아하는 아이템 추천하는 방식
    * 타 사용자 간의 유사도 측정 후 가장 유사도 높은 TOP-N 사용자 추출해 그들이 선호하는 아이템 추천
    * 행이 개별 사용자, 열은 개별 아이템이다.

* 아이템 기반 최근접 이웃 방식
    * 아이템 속성과는 상관없다.
    * 사용자가 아이템을 좋고 싫은지의 평가 척도가 유사한 **아이템** 추천 기준
    * 행과 열이 반대가 된다.(행이 개별 아이템, 열이 개별 사용자가 된다.)

* 사용자 기반보다는 아이템 기반 협업 필터링이 정확도가 더 높다.
    * 매우 유명한 영화는 취향과 관계없이 많이 관람하는 경향이 있다.
    * 평점이 결측인 경우 유사도 비교가 매우 어렵다.

## 잠재 요인 협업 필터링

### 잠재 요인 협업 필터링의 이해

* 잠재 요인 협업 필터링
    * 사용자-아이템 평점 행렬 속에 숨은 잠재 요인을 추출해 추천 예측
    * 행렬 분해(Matrix Factorization)
        * 대규모 다차원 행렬을 SVD 차원 감소 기법 분해하는 과정에서 잠재 요인 추출
    * 행렬 분해에 기반한 잠재 요인 협업 필터링 적용
    * 사용자-아이템 평점 행렬 데이터에서 '잠재 요인'을 끄집어 낸다.

* 영화 평점
    * 사용자의 장르별 선호도 벡터와 영화의 장르별 특성 벡터 곱 표현
* 숨겨진 '잠재 요인' 기반으로 분해된 매트릭스로 아직 평가하지 않은 아이템에 대한 예측 평가 수행
* 행렬 분해 : 다차원의 매트릭스를 저차원의 매트릭스로 분해하는 기법

### 행렬 분해의 이해

* 다차원의 매트릭스를 저차원 매트릭스로 분해 기법
    * SVD(Singular Vector Decomposition), NMF(Non-Negative Matrix Factorization) 등

* M : 총 사용자 수, N : 아이템 수, K : 잠재 요인의 차원 수
* R : M X N 차원의 사용자-아이템 평점 행렬
* P : M X K 사용자-잠재 요인 행렬, 사용자와 잠재 요인과의 관계 값
* Q : N X K 아이템-잠재 요인 행렬, 아이템과 잠재 요인과의 관계 값

$$R = P \times Q^T$$

$$r_{(u,i)} = p_u * q_i^t$$

* 행렬 분해는 주로 SVD 이용하나, 결측치 없는 행렬만 가능하다.
* 널 값이 있는 경우, SGD, ALS로 SVD 수행한다.
    * Stochastic Gradient Descent(SGD)
    * Alternating Least Squares(ALS)

### 확률적 경사 하강법을 이용한 행렬 분해

* P와 Q 행렬로 계산된 예측 R 행렬 값이 실제 R 행렬 값과 가장 최소의 오류를 가지도록 반복적인 비용 함수 최적화를 통해 P와 Q 유추

* SGD 이용한 행렬 분해의 절차

1. P와 Q 임의의 값을 가진 행렬로 설정
2. P와 Q.T 값을 곱해 예측 R 행렬을 계산, 예측 R 행렬, 실제 R 행렬에 해당하는 오류 값 계산
3. 오류를 최소화할 수 있도록 P와 Q 행렬을 적절한 값으로 각각 업데이트
4. 원하는 오류 값을 가질 때까지 반복하며 P와 Q 근사

* 실제 값과 예측값의 오류 최소화와 L2 규제 고려한 비용 함수식

$$min \sum(r_{(u,i)} - p_u q_i^t)^2 + \lambda(||q_i||^2 + ||p_u||^2)$$

* 업데이트된 p, q

$$p'_u = p_u + \eta (e_{(u,i)} * q_i - \lambda * p_u) $$  
$$q'_i = q_i + \eta (e_{(u,i)} * p_u - \lambda * q_i) $$  
$$e_{(u, i)} = r_{(u, i)} - \hat{r}_{(u, i)}$$


```python
# SGD 이용해 행렬 분해 수행
import numpy as np

# 원본 행렬 R 만들고 분해 행렬 P, Q은 초기화하고 잠재 요인 차원 K 설정
R = np.array([[4, np.NaN, np.NaN, 2, np.NaN],
             [np.NaN, 5, np.NaN, 3, 1],
             [np.NaN, np.NaN, 3, 4, 4],
             [5, 2, 1, 2, np.NaN]])

num_users, num_items = R.shape
K = 3

# P, Q 행렬 크기 지정 후 정규 분포를 가진 임의의 값으로 입력
np.random.seed(1)
P = np.random.normal(scale=1./K, size=(num_users, K))
Q = np.random.normal(scale=1./K, size=(num_items, K))
```


```python
# 실제 R 행렬과 예측 행렬의 오차를 구하는 get_rmse() 함수 제작

from sklearn.metrics import mean_squared_error

def get_rmse(R, P, Q, non_zeros):
    error = 0

    # 두 개의 분해된 행렬 P, Q.T의 내적으로 예측 R 행렬 생성
    full_pred_matrix = np.dot(P, Q.T)

    # 실제 R 행렬에서 널이 아닌 값의 위치 인덱스를 추출
    # 실제 R 행렬과 예측 행렬의 RMSE 추출
    x_non_zero_ind = [non_zero[0] for non_zero in non_zeros]
    y_non_zero_ind = [non_zero[1] for non_zero in non_zeros]
    R_non_zeros = R[x_non_zero_ind, y_non_zero_ind]
    full_pred_matrix_non_zeros = full_pred_matrix[x_non_zero_ind, y_non_zero_ind]
    mse = mean_squared_error(R_non_zeros, full_pred_matrix_non_zeros)
    rmse = np.sqrt(mse)

    return rmse
```


```python
# SGD 기반 행렬 분해 수행
# R > 0인 행, 열 위치, 값을 non_zeros 리스트에 저장
non_zeros = [ (i, j, R[i, j]) for i in range(num_users) for j in range(num_items) if R[i, j] > 0]

steps = 1000
learning_rate = 0.01
r_lambda=0.01

# SGD 기법으로 P, Q 매트릭스를 계속 업데이트.
for step in range(steps):
    for i, j, r in non_zeros:
        # 실제 값과 예측 값의 차이인 오류 값 구함.
        eij = r - np.dot(P[i, :], Q[j, :].T)
        # Regularization을 반영한 SGD 업데이트 공식 적용
        P[i, :] = P[i, :] + learning_rate*(eij * Q[j, :] - r_lambda*P[i, :])
        Q[j, :] = Q[j, :] + learning_rate*(eij * P[i, :] - r_lambda*Q[j, :])

    rmse = get_rmse(R, P, Q, non_zeros)
    if (step % 50) == 0: # 50번 마다 프린트해둔다.
        print('### iteration step : ', step, " rmse : ", rmse)
```

    ### iteration step :  0  rmse :  3.2388050277987723
    ### iteration step :  50  rmse :  0.4876723101369648
    ### iteration step :  100  rmse :  0.1564340384819247
    ### iteration step :  150  rmse :  0.07455141311978046
    ### iteration step :  200  rmse :  0.04325226798579314
    ### iteration step :  250  rmse :  0.029248328780878973
    ### iteration step :  300  rmse :  0.022621116143829466
    ### iteration step :  350  rmse :  0.019493636196525135
    ### iteration step :  400  rmse :  0.018022719092132704
    ### iteration step :  450  rmse :  0.01731968595344266
    ### iteration step :  500  rmse :  0.016973657887570753
    ### iteration step :  550  rmse :  0.016796804595895633
    ### iteration step :  600  rmse :  0.01670132290188466
    ### iteration step :  650  rmse :  0.01664473691247669
    ### iteration step :  700  rmse :  0.016605910068210026
    ### iteration step :  750  rmse :  0.016574200475705
    ### iteration step :  800  rmse :  0.01654431582921597
    ### iteration step :  850  rmse :  0.01651375177473524
    ### iteration step :  900  rmse :  0.01648146573819501
    ### iteration step :  950  rmse :  0.016447171683479155



```python
pred_matrix = np.dot(P, Q.T)
print('예측 행렬:\n', np.round(pred_matrix, 3))
```

    예측 행렬:
     [[3.991 0.897 1.306 2.002 1.663]
     [6.696 4.978 0.979 2.981 1.003]
     [6.677 0.391 2.987 3.977 3.986]
     [4.968 2.005 1.006 2.017 1.14 ]]



```python
print('원본 행렬:\n', R)
```

    원본 행렬:
     [[ 4. nan nan  2. nan]
     [nan  5. nan  3.  1.]
     [nan nan  3.  4.  4.]
     [ 5.  2.  1.  2. nan]]


## 콘텐츠 기반 필터링 실습 - TMDB 5000 영화 데이터 세트

### 장르 속성을 이용한 영화 콘텐츠 필터링

* 콘텐츠 기반 필터링
    * 특정 영화를 감상하고 좋아했다면 그와 비슷한 특성/속성, 구성 요소 등을 가진 다른 영화 추천
    * 유사성 판단 기준 : 영화를 구성하는 다양한 콘텐츠(장르, 감독 등) 기반으로 하는 방식
    * 영화 장르 속성을 기반으로 만든다.

### 데이터 로딩 및 가공


```python
# 장르 속성으로 콘텐츠 기반 필터링 수행
import pandas as pd
import numpy as np
import warnings;warnings.filterwarnings("ignore")
movies = pd.read_csv('./tmdb-movie-metadata/tmdb_5000_movies.csv')
print(movies.shape)
movies.head(1)
```

    (4803, 20)





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
      <th>budget</th>
      <th>genres</th>
      <th>homepage</th>
      <th>id</th>
      <th>keywords</th>
      <th>original_language</th>
      <th>original_title</th>
      <th>overview</th>
      <th>popularity</th>
      <th>production_companies</th>
      <th>production_countries</th>
      <th>release_date</th>
      <th>revenue</th>
      <th>runtime</th>
      <th>spoken_languages</th>
      <th>status</th>
      <th>tagline</th>
      <th>title</th>
      <th>vote_average</th>
      <th>vote_count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>237000000</td>
      <td>[{"id": 28, "name": "Action"}, {"id": 12, "nam...</td>
      <td>http://www.avatarmovie.com/</td>
      <td>19995</td>
      <td>[{"id": 1463, "name": "culture clash"}, {"id":...</td>
      <td>en</td>
      <td>Avatar</td>
      <td>In the 22nd century, a paraplegic Marine is di...</td>
      <td>150.437577</td>
      <td>[{"name": "Ingenious Film Partners", "id": 289...</td>
      <td>[{"iso_3166_1": "US", "name": "United States o...</td>
      <td>2009-12-10</td>
      <td>2787965087</td>
      <td>162.0</td>
      <td>[{"iso_639_1": "en", "name": "English"}, {"iso...</td>
      <td>Released</td>
      <td>Enter the World of Pandora.</td>
      <td>Avatar</td>
      <td>7.2</td>
      <td>11800</td>
    </tr>
  </tbody>
</table>
</div>




```python
movies_df = movies[['id','title','genres','vote_average', 'vote_count', 'popularity',
                   'keywords','overview']]
```

* 파이썬 리스트 내부에 여러 딕셔너리가 있다.
    * 다양한 정보를 제공하기 위함이고, 이 칼럼을 먼저 가공해야 한다.


```python
pd.set_option('max_colwidth', 100)
movies_df[['genres', 'keywords']][:1]
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
      <th>genres</th>
      <th>keywords</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>[{"id": 28, "name": "Action"}, {"id": 12, "name": "Adventure"}, {"id": 14, "name": "Fantasy"}, {...</td>
      <td>[{"id": 1463, "name": "culture clash"}, {"id": 2964, "name": "future"}, {"id": 3386, "name": "sp...</td>
    </tr>
  </tbody>
</table>
</div>




```python
# key는 name이고, 파이썬 ast 모듈의 literal_eval 함수 적용해 문자열을 객체로 변환
from ast import literal_eval
movies_df['genres'] = movies_df['genres'].apply(literal_eval)
movies_df['keywords'] = movies_df['keywords'].apply(literal_eval)
```


```python
# 리스트 내 여러 딕셔너리의 name 키에 해당하는 값을 찾아 이를 리스트 객체로 변환
movies_df['genres'] = movies_df['genres'].apply(lambda x : [y['name'] for y in x])
movies_df['keywords'] = movies_df['keywords'].apply(lambda x : [y['name'] for y in x])
movies_df[['genres', 'keywords']][:1]
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
      <th>genres</th>
      <th>keywords</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>[Action, Adventure, Fantasy, Science Fiction]</td>
      <td>[culture clash, future, space war, space colony, society, space travel, futuristic, romance, spa...</td>
    </tr>
  </tbody>
</table>
</div>



### 장르 콘텐츠 유사도 측정

* 문자열로 변환된 genres 칼럼을 카운트 기반 피처 벡터화 변환한다.
* genres 문자열을 피처 벡터화 행렬로 변환한 데이터 세트를 코사인 유사도로 비교한다.
    * 데이터 세트의 레코드별로 다른 레코드와 장르에서 코사인 유사도 값을 갖는 객체 생성
* 장르 유사도가 높은 영화 중 평점이 높은 순으로 영화 추천한다.


```python
from sklearn.feature_extraction.text import CountVectorizer

# 공백문자로 word 단위가 구분되는 문자열로 변환
movies_df['genres_literal'] = movies_df['genres'].apply(lambda x : (' ').join(x))
count_vect = CountVectorizer(min_df=0, ngram_range=(1, 2))
genre_mat = count_vect.fit_transform(movies_df['genres_literal'])
print(genre_mat.shape)
```

    (4803, 276)



```python
# 피처 벡터화된 행렬에 코사인 유사도 적용
from sklearn.metrics.pairwise import cosine_similarity

genre_sim = cosine_similarity(genre_mat, genre_mat)
print(genre_sim.shape)
print(genre_sim[:1])
```

    (4803, 4803)
    [[1.         0.59628479 0.4472136  ... 0.         0.         0.        ]]



```python
# gemre_sim : 행별 장르 유사도 값을 가진다.
# movies_df의 개별 레코드에 대해 가장 장르 유사도가 높은 순으로 다른 레코드 추출
# genre_sim 객체 활용

genre_sim_sorted_ind = genre_sim.argsort()[:, ::-1]
print(genre_sim_sorted_ind[:1])
```

    [[   0 3494  813 ... 3038 3037 2401]]


* 0번 레코드 기준
    * 자신을 제외하면 3494번 레코드가 가장 유사도가 높고, 다음 813번 순으로 유사도가 높다.
* genre_sim_sorted_ind : 각 레코드의 장르 코사인 유사도가 가장 높은 순으로 정렬된 타 레코드의 위치 인덱스 값을 가진다.
    * 위치 인덱스로 특정 레코드와 코사인 유사도가 높은 다른 레코드 추출 가능하다.

### 장르 콘텐츠 필터링을 이용한 영화 추천


```python
# 장르 유사도에 따라 영화를 추천하는 함수 생성
def find_sim_movie(df, sorted_ind, title_name, top_n=10):

    # 인자로 입력된 movies_df에서 'title' 칼럼이 입력된 title_name 값 df추출
    title_movie = df[df['title'] == title_name]

    title_index = title_movie.index.values
    similar_indexes = sorted_ind[title_index, :(top_n)]

    print(similar_indexes)
    similar_indexes = similar_indexes.reshape(-1) # 2차원을 1차원으로 바꿔버린다.

    return df.iloc[similar_indexes]
```


```python
# 본 함수로 영화 '대부'와 장르로 유사한 영화 10개 추천
similar_movies = find_sim_movie(movies_df, genre_sim_sorted_ind, 'The Godfather', 10)
similar_movies[['title', 'vote_average']]
```

    [[2731 1243 3636 1946 2640 4065 1847 4217  883 3866]]





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
      <th>title</th>
      <th>vote_average</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2731</th>
      <td>The Godfather: Part II</td>
      <td>8.3</td>
    </tr>
    <tr>
      <th>1243</th>
      <td>Mean Streets</td>
      <td>7.2</td>
    </tr>
    <tr>
      <th>3636</th>
      <td>Light Sleeper</td>
      <td>5.7</td>
    </tr>
    <tr>
      <th>1946</th>
      <td>The Bad Lieutenant: Port of Call - New Orleans</td>
      <td>6.0</td>
    </tr>
    <tr>
      <th>2640</th>
      <td>Things to Do in Denver When You're Dead</td>
      <td>6.7</td>
    </tr>
    <tr>
      <th>4065</th>
      <td>Mi America</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1847</th>
      <td>GoodFellas</td>
      <td>8.2</td>
    </tr>
    <tr>
      <th>4217</th>
      <td>Kids</td>
      <td>6.8</td>
    </tr>
    <tr>
      <th>883</th>
      <td>Catch Me If You Can</td>
      <td>7.7</td>
    </tr>
    <tr>
      <th>3866</th>
      <td>City of God</td>
      <td>8.1</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 영화의 평점 정보를 이용해서 다시 추천한다.
movies_df[['title', 'vote_average', 'vote_count']].sort_values('vote_average',
                                                              ascending=False)[:10]
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
      <th>title</th>
      <th>vote_average</th>
      <th>vote_count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>3519</th>
      <td>Stiff Upper Lips</td>
      <td>10.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4247</th>
      <td>Me You and Five Bucks</td>
      <td>10.0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>4045</th>
      <td>Dancer, Texas Pop. 81</td>
      <td>10.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4662</th>
      <td>Little Big Top</td>
      <td>10.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3992</th>
      <td>Sardaarji</td>
      <td>9.5</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2386</th>
      <td>One Man's Hero</td>
      <td>9.3</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2970</th>
      <td>There Goes My Baby</td>
      <td>8.5</td>
      <td>2</td>
    </tr>
    <tr>
      <th>1881</th>
      <td>The Shawshank Redemption</td>
      <td>8.5</td>
      <td>8205</td>
    </tr>
    <tr>
      <th>2796</th>
      <td>The Prisoner of Zenda</td>
      <td>8.4</td>
      <td>11</td>
    </tr>
    <tr>
      <th>3337</th>
      <td>The Godfather</td>
      <td>8.4</td>
      <td>5893</td>
    </tr>
  </tbody>
</table>
</div>



* 평가 횟수가 매우 적은 왜곡된 평점 데이터를 제거하기 위해, 가중 평점 활용한다.

$$\text{가중 평점(Weighted Rating)} = \frac{v}{(v+m)} * R + \frac{m}{(v+m)} * C$$

* v: 개별 영화 평점을 투표한 횟수
* m: 평점 부여를 위한 최소 투표 횟수
* R: 개별 영화에 대한 평균 평점
* C: 전체 영화에 대한 평균 평점


```python
C = movies_df['vote_average'].mean()
m = movies_df['vote_count'].quantile(0.6)
print('C:', round(C, 3), 'm:', round(m, 3))
```

    C: 6.092 m: 370.2



```python
percentile = 0.6
m = movies['vote_count'].quantile(percentile)
C = movies['vote_average'].mean()

def weighted_vote_average(record):
    v = record['vote_count']
    R = record['vote_average']

    return ( (v/(v+m)) * R) + ( (m/(m+v) * C))

movies_df['weighted_vote'] = movies.apply(weighted_vote_average, axis=1)
```


```python
# 이 가중평균 칼럼으로 평점 높은 순 상위 10개 영화를 추출한다.
movies_df[['title', 'vote_average', 'weighted_vote', 'vote_count']].sort_values(
            'weighted_vote', ascending=False)[:10]
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
      <th>title</th>
      <th>vote_average</th>
      <th>weighted_vote</th>
      <th>vote_count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1881</th>
      <td>The Shawshank Redemption</td>
      <td>8.5</td>
      <td>8.396052</td>
      <td>8205</td>
    </tr>
    <tr>
      <th>3337</th>
      <td>The Godfather</td>
      <td>8.4</td>
      <td>8.263591</td>
      <td>5893</td>
    </tr>
    <tr>
      <th>662</th>
      <td>Fight Club</td>
      <td>8.3</td>
      <td>8.216455</td>
      <td>9413</td>
    </tr>
    <tr>
      <th>3232</th>
      <td>Pulp Fiction</td>
      <td>8.3</td>
      <td>8.207102</td>
      <td>8428</td>
    </tr>
    <tr>
      <th>65</th>
      <td>The Dark Knight</td>
      <td>8.2</td>
      <td>8.136930</td>
      <td>12002</td>
    </tr>
    <tr>
      <th>1818</th>
      <td>Schindler's List</td>
      <td>8.3</td>
      <td>8.126069</td>
      <td>4329</td>
    </tr>
    <tr>
      <th>3865</th>
      <td>Whiplash</td>
      <td>8.3</td>
      <td>8.123248</td>
      <td>4254</td>
    </tr>
    <tr>
      <th>809</th>
      <td>Forrest Gump</td>
      <td>8.2</td>
      <td>8.105954</td>
      <td>7927</td>
    </tr>
    <tr>
      <th>2294</th>
      <td>Spirited Away</td>
      <td>8.3</td>
      <td>8.105867</td>
      <td>3840</td>
    </tr>
    <tr>
      <th>2731</th>
      <td>The Godfather: Part II</td>
      <td>8.3</td>
      <td>8.079586</td>
      <td>3338</td>
    </tr>
  </tbody>
</table>
</div>




```python
def find_sim_movie(df, sorted_ind, title_name, top_n=10):
    title_movie = df[df['title'] == title_name]
    title_index = title_movie.index.values

    # top_n의 2배에 해당하는 장르 유사성이 높은 인덱스 추출
    similar_indexes = sorted_ind[title_index, :(top_n*2)]
    similar_indexes = similar_indexes.reshape(-1)

    # 기준 영화 인덱스는 제외한다.
    similar_indexes = similar_indexes[similar_indexes != title_index]

    # top_n의 2배에 해당하는 후보군에서 weighted_vote가 높은 순으로 top_n만큼 추출
    return df.iloc[similar_indexes].sort_values('weighted_vote', ascending=False)[:top_n]

similar_movies = find_sim_movie(movies_df, genre_sim_sorted_ind, 'The Godfather', 10)
similar_movies[['title', 'vote_average', 'weighted_vote']]
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
      <th>title</th>
      <th>vote_average</th>
      <th>weighted_vote</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2731</th>
      <td>The Godfather: Part II</td>
      <td>8.3</td>
      <td>8.079586</td>
    </tr>
    <tr>
      <th>1847</th>
      <td>GoodFellas</td>
      <td>8.2</td>
      <td>7.976937</td>
    </tr>
    <tr>
      <th>3866</th>
      <td>City of God</td>
      <td>8.1</td>
      <td>7.759693</td>
    </tr>
    <tr>
      <th>1663</th>
      <td>Once Upon a Time in America</td>
      <td>8.2</td>
      <td>7.657811</td>
    </tr>
    <tr>
      <th>883</th>
      <td>Catch Me If You Can</td>
      <td>7.7</td>
      <td>7.557097</td>
    </tr>
    <tr>
      <th>281</th>
      <td>American Gangster</td>
      <td>7.4</td>
      <td>7.141396</td>
    </tr>
    <tr>
      <th>4041</th>
      <td>This Is England</td>
      <td>7.4</td>
      <td>6.739664</td>
    </tr>
    <tr>
      <th>1149</th>
      <td>American Hustle</td>
      <td>6.8</td>
      <td>6.717525</td>
    </tr>
    <tr>
      <th>1243</th>
      <td>Mean Streets</td>
      <td>7.2</td>
      <td>6.626569</td>
    </tr>
    <tr>
      <th>2839</th>
      <td>Rounders</td>
      <td>6.9</td>
      <td>6.530427</td>
    </tr>
  </tbody>
</table>
</div>



## 아이템 기반 최근접 이웃 협업 필터링 실습

### 데이터 가공


```python
import pandas as pd
import numpy as np

movies = pd.read_csv('./ml-latest-small/movies.csv')
ratings = pd.read_csv('./ml-latest-small/ratings.csv')
print(movies.shape)
print(ratings.shape)
```

    (9742, 3)
    (100836, 4)



```python
ratings = ratings[['userId', 'movieId', 'rating']]
ratings_matrix = ratings.pivot_table('rating', index='userId', columns='movieId')
ratings_matrix.head(3)
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
      <th>movieId</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>10</th>
      <th>...</th>
      <th>193565</th>
      <th>193567</th>
      <th>193571</th>
      <th>193573</th>
      <th>193579</th>
      <th>193581</th>
      <th>193583</th>
      <th>193585</th>
      <th>193587</th>
      <th>193609</th>
    </tr>
    <tr>
      <th>userId</th>
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
      <td>4.0</td>
      <td>NaN</td>
      <td>4.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>4.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>3 rows × 9724 columns</p>
</div>




```python
# title 칼럼을 얻기 위해 movies 조인
rating_movies = pd.merge(ratings, movies, on='movieId')

# columns='title'로 타이틀 칼럼으로 피벗 수행
ratings_matrix = rating_movies.pivot_table('rating', index='userId', columns='title')

# 결측치를 모두 0으로 변환
ratings_matrix = ratings_matrix.fillna(0)
ratings_matrix.head(3)
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
      <th>title</th>
      <th>'71 (2014)</th>
      <th>'Hellboy': The Seeds of Creation (2004)</th>
      <th>'Round Midnight (1986)</th>
      <th>'Salem's Lot (2004)</th>
      <th>'Til There Was You (1997)</th>
      <th>'Tis the Season for Love (2015)</th>
      <th>'burbs, The (1989)</th>
      <th>'night Mother (1986)</th>
      <th>(500) Days of Summer (2009)</th>
      <th>*batteries not included (1987)</th>
      <th>...</th>
      <th>Zulu (2013)</th>
      <th>[REC] (2007)</th>
      <th>[REC]² (2009)</th>
      <th>[REC]³ 3 Génesis (2012)</th>
      <th>anohana: The Flower We Saw That Day - The Movie (2013)</th>
      <th>eXistenZ (1999)</th>
      <th>xXx (2002)</th>
      <th>xXx: State of the Union (2005)</th>
      <th>¡Three Amigos! (1986)</th>
      <th>À nous la liberté (Freedom for Us) (1931)</th>
    </tr>
    <tr>
      <th>userId</th>
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
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>4.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>3 rows × 9719 columns</p>
</div>



* pivot_table()

        rating_movies.pivot_table('rating', index='userId', columns='title')

    * index(row_level) : userId 값 기준 정렬
    * columns : title 값으로 구성
    * values : ratings 값으로 구성
    * 세 가지 조건을 만족하는 표로 제작하는 함수

### 영화 간 유사도 산출


```python
# 영화 간 유사도 측정(코사인 유사도 기준)
# 영화 기준이므로 기존의 ratings_matrix를 전치해야 한다.(행이 영화가 되어야 함)

ratings_matrix_T = ratings_matrix.transpose()
ratings_matrix_T.head(3)
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
      <th>userId</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>10</th>
      <th>...</th>
      <th>601</th>
      <th>602</th>
      <th>603</th>
      <th>604</th>
      <th>605</th>
      <th>606</th>
      <th>607</th>
      <th>608</th>
      <th>609</th>
      <th>610</th>
    </tr>
    <tr>
      <th>title</th>
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
      <th>'71 (2014)</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>'Hellboy': The Seeds of Creation (2004)</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>'Round Midnight (1986)</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>3 rows × 610 columns</p>
</div>




```python
from sklearn.metrics.pairwise import cosine_similarity

item_sim = cosine_similarity(ratings_matrix_T, ratings_matrix_T)

# 코사인 유사도는 넘파이 행렬이므로 이를 영화 이름과 매핑해 데이터프레임으로 만든다.
item_sim_df = pd.DataFrame(data=item_sim, index=ratings_matrix.columns,
                          columns=ratings_matrix.columns)

print(item_sim_df.shape)
item_sim_df.head(3)
```

    (9719, 9719)





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
      <th>title</th>
      <th>'71 (2014)</th>
      <th>'Hellboy': The Seeds of Creation (2004)</th>
      <th>'Round Midnight (1986)</th>
      <th>'Salem's Lot (2004)</th>
      <th>'Til There Was You (1997)</th>
      <th>'Tis the Season for Love (2015)</th>
      <th>'burbs, The (1989)</th>
      <th>'night Mother (1986)</th>
      <th>(500) Days of Summer (2009)</th>
      <th>*batteries not included (1987)</th>
      <th>...</th>
      <th>Zulu (2013)</th>
      <th>[REC] (2007)</th>
      <th>[REC]² (2009)</th>
      <th>[REC]³ 3 Génesis (2012)</th>
      <th>anohana: The Flower We Saw That Day - The Movie (2013)</th>
      <th>eXistenZ (1999)</th>
      <th>xXx (2002)</th>
      <th>xXx: State of the Union (2005)</th>
      <th>¡Three Amigos! (1986)</th>
      <th>À nous la liberté (Freedom for Us) (1931)</th>
    </tr>
    <tr>
      <th>title</th>
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
      <th>'71 (2014)</th>
      <td>1.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.141653</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.342055</td>
      <td>0.543305</td>
      <td>0.707107</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.139431</td>
      <td>0.327327</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>'Hellboy': The Seeds of Creation (2004)</th>
      <td>0.0</td>
      <td>1.000000</td>
      <td>0.707107</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>'Round Midnight (1986)</th>
      <td>0.0</td>
      <td>0.707107</td>
      <td>1.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.176777</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>3 rows × 9719 columns</p>
</div>




```python
item_sim_df["Godfather, The (1972)"].sort_values(ascending=False)[:6]
```




    title
    Godfather, The (1972)                        1.000000
    Godfather: Part II, The (1974)               0.821773
    Goodfellas (1990)                            0.664841
    One Flew Over the Cuckoo's Nest (1975)       0.620536
    Star Wars: Episode IV - A New Hope (1977)    0.595317
    Fargo (1996)                                 0.588614
    Name: Godfather, The (1972), dtype: float64



* 영화 '대부(Godfather)'와 유사도
    * 기준인 '대부'를 제하고 '대부 2편'이 가장 유사도가 높다.
        * 이후 영화 좋은 친구들(Goodfellas) 등으로 나타난다.
    * 콘텐츠 기반 필터링과 차이점
        * 장르가 완전히 다른 영화도 유사도가 매우 높게 나타난다.


```python
# 인셉션과 유사도 상위 6개
item_sim_df["Inception (2010)"].sort_values(ascending=False)[1:6]
```




    title
    Dark Knight, The (2008)          0.727263
    Inglourious Basterds (2009)      0.646103
    Shutter Island (2010)            0.617736
    Dark Knight Rises, The (2012)    0.617504
    Fight Club (1999)                0.615417
    Name: Inception (2010), dtype: float64



* 영화 '인셉션'과 유사도
    * 다크나이트가 가장 높고, 나머지는 스릴러, 액션 영화가 높다.
    * 아이템 기반 유사도 데이터
        * 평점 정보를 취합해 유사한 다른 영화 추천 가능하게 한다.
        * 개인에게 특화된 영화 추천 알고리즘 제작할 수 있다.

### 아이템 기반 최근접 이웃 협업 필터링으로 개인화된 영화 추천

* 개인화된 영화 추천
    * 개인이 관람하지 않은 영화 추천
    * 아이템 유사도와 기존에 관람한 영화의 평점 데이터 기반으로 모든 영화의 예측 평점 계산
    * 높은 예측 평점을 가진 영화를 추천하는 방식

* 아이템 기반의 협업 필터링으로 개인화된 예측 평점

$$\hat{R}_{u,i} = \sum^N(S_{i,N} * R_{u, N}) / \sum^N (|S_{i,N}|) $$


```python
def predict_rating(ratings_arr, item_sim_arr):
    ratings_pred = ratings_arr.dot(item_sim_arr) / np.array([np.abs(item_sim_arr).sum(axis=1)])
    return ratings_pred
```


```python
# 개인화된 예측 평점
ratings_pred = predict_rating(ratings_matrix.values, item_sim_df.values)
ratings_pred_matrix = pd.DataFrame(data=ratings_pred, index=ratings_matrix.index,
                                  columns = ratings_matrix.columns)

ratings_pred_matrix.head(3)
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
      <th>title</th>
      <th>'71 (2014)</th>
      <th>'Hellboy': The Seeds of Creation (2004)</th>
      <th>'Round Midnight (1986)</th>
      <th>'Salem's Lot (2004)</th>
      <th>'Til There Was You (1997)</th>
      <th>'Tis the Season for Love (2015)</th>
      <th>'burbs, The (1989)</th>
      <th>'night Mother (1986)</th>
      <th>(500) Days of Summer (2009)</th>
      <th>*batteries not included (1987)</th>
      <th>...</th>
      <th>Zulu (2013)</th>
      <th>[REC] (2007)</th>
      <th>[REC]² (2009)</th>
      <th>[REC]³ 3 Génesis (2012)</th>
      <th>anohana: The Flower We Saw That Day - The Movie (2013)</th>
      <th>eXistenZ (1999)</th>
      <th>xXx (2002)</th>
      <th>xXx: State of the Union (2005)</th>
      <th>¡Three Amigos! (1986)</th>
      <th>À nous la liberté (Freedom for Us) (1931)</th>
    </tr>
    <tr>
      <th>userId</th>
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
      <td>0.070345</td>
      <td>0.577855</td>
      <td>0.321696</td>
      <td>0.227055</td>
      <td>0.206958</td>
      <td>0.194615</td>
      <td>0.249883</td>
      <td>0.102542</td>
      <td>0.157084</td>
      <td>0.178197</td>
      <td>...</td>
      <td>0.113608</td>
      <td>0.181738</td>
      <td>0.133962</td>
      <td>0.128574</td>
      <td>0.006179</td>
      <td>0.212070</td>
      <td>0.192921</td>
      <td>0.136024</td>
      <td>0.292955</td>
      <td>0.720347</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.018260</td>
      <td>0.042744</td>
      <td>0.018861</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.035995</td>
      <td>0.013413</td>
      <td>0.002314</td>
      <td>0.032213</td>
      <td>0.014863</td>
      <td>...</td>
      <td>0.015640</td>
      <td>0.020855</td>
      <td>0.020119</td>
      <td>0.015745</td>
      <td>0.049983</td>
      <td>0.014876</td>
      <td>0.021616</td>
      <td>0.024528</td>
      <td>0.017563</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.011884</td>
      <td>0.030279</td>
      <td>0.064437</td>
      <td>0.003762</td>
      <td>0.003749</td>
      <td>0.002722</td>
      <td>0.014625</td>
      <td>0.002085</td>
      <td>0.005666</td>
      <td>0.006272</td>
      <td>...</td>
      <td>0.006923</td>
      <td>0.011665</td>
      <td>0.011800</td>
      <td>0.012225</td>
      <td>0.000000</td>
      <td>0.008194</td>
      <td>0.007017</td>
      <td>0.009229</td>
      <td>0.010420</td>
      <td>0.084501</td>
    </tr>
  </tbody>
</table>
<p>3 rows × 9719 columns</p>
</div>



* 예측 결과와 실제 평점간 차이 확인


```python
from sklearn.metrics import mean_squared_error

def get_mse(pred, actual):
    # 평점 있는 실제 영화
    pred = pred[actual.nonzero()].flatten()
    actual = actual[actual.nonzero()].flatten()
    return mean_squared_error(pred, actual)

print('아이템 기반 모든 최근접 이웃 MSE: ', get_mse(ratings_pred, ratings_matrix.values))
```

    아이템 기반 모든 최근접 이웃 MSE:  9.895354759094706



```python
# 가장 비슷한 유사도 갖는 영화에 대해서만 유사도 벡터 적용
def predict_rating_topsim(ratings_arr, item_sim_arr, n=20):
    pred = np.zeros(ratings_arr.shape)

    for col in range(ratings_arr.shape[1]):
        top_n_items = [np.argsort(item_sim_arr[:, col])[:-n-1:-1]] # 유사도 큰 순 n개 데이터 행렬 인덱스 반환
        # 개인화된 예측 평점
        for row in range(ratings_arr.shape[0]):
            pred[row, col] = item_sim_arr[col, :][top_n_items].dot(ratings_arr[row,:][top_n_items].T)
            pred[row, col] /= np.sum(np.abs(item_sim_arr[col, :][top_n_items]))

    return pred
```


```python
ratings_pred = predict_rating_topsim(ratings_matrix.values, item_sim_df.values, 20)
print('아이템 기반 최근접 20 이웃 MSE: ', get_mse(ratings_pred, ratings_matrix.values))

# 계산된 예측 평점 데이터를 데이터프레임으로 재생성

ratings_pred_matrix = pd.DataFrame(data=ratings_pred, index= ratings_matrix.index,
                                  columns= ratings_matrix.columns)
```

    아이템 기반 최근접 20 이웃 MSE:  3.6949827608772314



```python
# 특정 사용자에게 영화 추천(userId=9)
user_rating_id = ratings_matrix.loc[9, :]
user_rating_id[user_rating_id > 0].sort_values(ascending=False)[:10]
```




    title
    Adaptation (2002)                                                                 5.0
    Austin Powers in Goldmember (2002)                                                5.0
    Lord of the Rings: The Fellowship of the Ring, The (2001)                         5.0
    Lord of the Rings: The Two Towers, The (2002)                                     5.0
    Producers, The (1968)                                                             5.0
    Citizen Kane (1941)                                                               5.0
    Raiders of the Lost Ark (Indiana Jones and the Raiders of the Lost Ark) (1981)    5.0
    Back to the Future (1985)                                                         5.0
    Glengarry Glen Ross (1992)                                                        4.0
    Sunset Blvd. (a.k.a. Sunset Boulevard) (1950)                                     4.0
    Name: 9, dtype: float64




```python
# 해당 사용자에게 아이템 기반 협업 필터링으로 영화 추천
# 사용자가 평점 주지 않은 영화를 리스트 객체로 반환하는 함수 생성

def get_unseen_movies(ratings_matrix, userId):
    user_rating = ratings_matrix.loc[userId, :]

    # user_rating이 0보다 크면 기존에 관람한 영화이므로 그 인덱스를 리스트로 만든다.
    already_seen = user_rating[user_rating > 0].index.tolist()

    movies_list = ratings_matrix.columns.tolist()

    # list comprehension : already_seen은 제외한다.
    unseen_list = [ movie for movie in movies_list if movie not in already_seen]

    return unseen_list
```


```python
# 최종 가장 높은 예측 평점을 가진 영화 추천 함수
def recomm_movie_by_userid(pred_df, userId, unseen_list, top_n=10):
    recomm_movies = pred_df.loc[userId, unseen_list].sort_values(ascending=False)[:top_n]
    return recomm_movies

# 사용자가 관람하지 않는 영화명 추출
unseen_list = get_unseen_movies(ratings_matrix, 9)

# 아이템 기반 KNN CF 영화 추천
recomm_movies = recomm_movie_by_userid(ratings_pred_matrix, 9, unseen_list, top_n=10)

# 평점 데이터를 데이터프레임 변환
recomm_movies = pd.DataFrame(data=recomm_movies.values, index=recomm_movies.index,
                            columns=['pred_score'])
recomm_movies
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
      <th>pred_score</th>
    </tr>
    <tr>
      <th>title</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Shrek (2001)</th>
      <td>0.866202</td>
    </tr>
    <tr>
      <th>Spider-Man (2002)</th>
      <td>0.857854</td>
    </tr>
    <tr>
      <th>Last Samurai, The (2003)</th>
      <td>0.817473</td>
    </tr>
    <tr>
      <th>Indiana Jones and the Temple of Doom (1984)</th>
      <td>0.816626</td>
    </tr>
    <tr>
      <th>Matrix Reloaded, The (2003)</th>
      <td>0.800990</td>
    </tr>
    <tr>
      <th>Harry Potter and the Sorcerer's Stone (a.k.a. Harry Potter and the Philosopher's Stone) (2001)</th>
      <td>0.765159</td>
    </tr>
    <tr>
      <th>Gladiator (2000)</th>
      <td>0.740956</td>
    </tr>
    <tr>
      <th>Matrix, The (1999)</th>
      <td>0.732693</td>
    </tr>
    <tr>
      <th>Pirates of the Caribbean: The Curse of the Black Pearl (2003)</th>
      <td>0.689591</td>
    </tr>
    <tr>
      <th>Lord of the Rings: The Return of the King, The (2003)</th>
      <td>0.676711</td>
    </tr>
  </tbody>
</table>
</div>



* 슈렉, 스파이더 맨, 등 높은 흥행작들이 추천되었다.
