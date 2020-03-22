---
title: "Recommender System and Application"
date : 2020-03-22
tags : [data science, python, recommender system]
header:
  #image: "/images/fig-1-33.png"
excerpt: "Data Science, Python, Recommender System"
mathjax: "true"
toc: true
toc_label: "My Table of Contents"
toc_icon: "cog"
---

## 행렬 분해를 이용한 잠재 요인 협업 필터링 실습


```python
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings("ignore")

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
# 최종 가장 높은 예측 평점을 가진 영화 추천 함수
def recomm_movie_by_userid(pred_df, userId, unseen_list, top_n=10):
    recomm_movies = pred_df.loc[userId, unseen_list].sort_values(ascending=False)[:top_n]
    return recomm_movies
```


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
def matrix_factorization(R, K, steps=200, learning_rate=0.01, r_lambda = 0.01):
    num_users, num_items = R.shape

    np.random.seed(1)
    P = np.random.normal(scale=1./K, size=(num_users, K))
    Q = np.random.normal(scale=1./K, size=(num_items, K))

    prev_rmse = 10000
    break_count = 0

    # R > 0 인 행, 열 위치 값을 리스트 객체로 저장
    non_zeros = [ (i, j, R[i, j]) for i in range(num_users) for j in range(num_items) if R[i, j] > 0]

    for step in range(steps):
        for i,j,r in non_zeros:
            eij = r - np.dot(P[i, :], Q[j, :].T) # 오류 값
            # 규제 반영한 SGD 공식 적용
            P[i, :] = P[i, :] + learning_rate * (eij * Q[j, :] - r_lambda * P[i, :])
            Q[j, :] = Q[j, :] + learning_rate * (eij * P[i, :] - r_lambda * Q[j, :])

        rmse = get_rmse(R, P, Q, non_zeros)
        if (step % 10) == 0 :
            print('### iteration step : ', step, " rmse : ", rmse)

    return P, Q        
```


```python
# 영화 평점 행렬 데이터를 데이터프레임으로 로딩 후 다시 사용자-아이템 평점 행렬로 제작
import pandas as pd
import numpy as np

movies = pd.read_csv('./ml-latest-small/movies.csv')
ratings = pd.read_csv('./ml-latest-small/ratings.csv')
ratings = ratings[['userId', 'movieId', 'rating']]
ratings_matrix = ratings.pivot_table('rating', index='userId', columns='movieId')

# title 칼럼을 얻고자 movies와 조인
ratings_movies = pd.merge(ratings, movies, on='movieId') # pandas merge ~ on
# pivot
ratings_matrix = ratings_movies.pivot_table('rating', index='userId', columns='title')
```


```python
P, Q = matrix_factorization(ratings_matrix.values, K=50, steps=200, learning_rate=0.01,
                           r_lambda=0.01)
pred_matrix = np.dot(P, Q.T)
```

    ### iteration step :  0  rmse :  2.9023619751336867
    ### iteration step :  10  rmse :  0.7335768591017927
    ### iteration step :  20  rmse :  0.5115539026853442
    ### iteration step :  30  rmse :  0.37261628282537446
    ### iteration step :  40  rmse :  0.2960818299181014
    ### iteration step :  50  rmse :  0.2520353192341642
    ### iteration step :  60  rmse :  0.22487503275269854
    ### iteration step :  70  rmse :  0.20685455302331537
    ### iteration step :  80  rmse :  0.19413418783028685
    ### iteration step :  90  rmse :  0.18470082002720403
    ### iteration step :  100  rmse :  0.17742927527209104
    ### iteration step :  110  rmse :  0.1716522696470749
    ### iteration step :  120  rmse :  0.1669518194687172
    ### iteration step :  130  rmse :  0.1630529219199754
    ### iteration step :  140  rmse :  0.15976691929679643
    ### iteration step :  150  rmse :  0.1569598699945732
    ### iteration step :  160  rmse :  0.1545339818671543
    ### iteration step :  170  rmse :  0.15241618551077643
    ### iteration step :  180  rmse :  0.1505508073962831
    ### iteration step :  190  rmse :  0.1488947091323209



```python
# 사용자 아이템 평점 행렬을 데이터프레임으로 변환
ratings_pred_matrix = pd.DataFrame(data=pred_matrix, index= ratings_matrix.index,
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
      <th>...All the Marbles (1981)</th>
      <th>...And Justice for All (1979)</th>
      <th>00 Schneider - Jagd auf Nihil Baxter (1994)</th>
      <th>1-900 (06) (1994)</th>
      <th>10 (1979)</th>
      <th>10 Cent Pistol (2015)</th>
      <th>10 Cloverfield Lane (2016)</th>
      <th>10 Items or Less (2006)</th>
      <th>10 Things I Hate About You (1999)</th>
      <th>10 Years (2011)</th>
      <th>10,000 BC (2008)</th>
      <th>100 Girls (2000)</th>
      <th>100 Streets (2016)</th>
      <th>101 Dalmatians (1996)</th>
      <th>101 Dalmatians (One Hundred and One Dalmatians) (1961)</th>
      <th>101 Dalmatians II: Patch's London Adventure (2003)</th>
      <th>101 Reykjavik (101 Reykjavík) (2000)</th>
      <th>102 Dalmatians (2000)</th>
      <th>10th &amp; Wolf (2006)</th>
      <th>10th Kingdom, The (2000)</th>
      <th>10th Victim, The (La decima vittima) (1965)</th>
      <th>11'09"01 - September 11 (2002)</th>
      <th>11:14 (2003)</th>
      <th>11th Hour, The (2007)</th>
      <th>12 Angry Men (1957)</th>
      <th>12 Angry Men (1997)</th>
      <th>12 Chairs (1971)</th>
      <th>12 Chairs (1976)</th>
      <th>12 Rounds (2009)</th>
      <th>12 Years a Slave (2013)</th>
      <th>...</th>
      <th>Zathura (2005)</th>
      <th>Zatoichi and the Chest of Gold (Zatôichi senryô-kubi) (Zatôichi 6) (1964)</th>
      <th>Zazie dans le métro (1960)</th>
      <th>Zebraman (2004)</th>
      <th>Zed &amp; Two Noughts, A (1985)</th>
      <th>Zeitgeist: Addendum (2008)</th>
      <th>Zeitgeist: Moving Forward (2011)</th>
      <th>Zeitgeist: The Movie (2007)</th>
      <th>Zelary (2003)</th>
      <th>Zelig (1983)</th>
      <th>Zero Dark Thirty (2012)</th>
      <th>Zero Effect (1998)</th>
      <th>Zero Theorem, The (2013)</th>
      <th>Zero de conduite (Zero for Conduct) (Zéro de conduite: Jeunes diables au collège) (1933)</th>
      <th>Zeus and Roxanne (1997)</th>
      <th>Zipper (2015)</th>
      <th>Zodiac (2007)</th>
      <th>Zombeavers (2014)</th>
      <th>Zombie (a.k.a. Zombie 2: The Dead Are Among Us) (Zombi 2) (1979)</th>
      <th>Zombie Strippers! (2008)</th>
      <th>Zombieland (2009)</th>
      <th>Zone 39 (1997)</th>
      <th>Zone, The (La Zona) (2007)</th>
      <th>Zookeeper (2011)</th>
      <th>Zoolander (2001)</th>
      <th>Zoolander 2 (2016)</th>
      <th>Zoom (2006)</th>
      <th>Zoom (2015)</th>
      <th>Zootopia (2016)</th>
      <th>Zulu (1964)</th>
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
      <td>3.055084</td>
      <td>4.092018</td>
      <td>3.564130</td>
      <td>4.502167</td>
      <td>3.981215</td>
      <td>1.271694</td>
      <td>3.603274</td>
      <td>2.333266</td>
      <td>5.091749</td>
      <td>3.972454</td>
      <td>1.623927</td>
      <td>3.910138</td>
      <td>4.775403</td>
      <td>3.837260</td>
      <td>3.875488</td>
      <td>1.550801</td>
      <td>2.929129</td>
      <td>2.680321</td>
      <td>3.225626</td>
      <td>3.251925</td>
      <td>2.778350</td>
      <td>3.331543</td>
      <td>2.391855</td>
      <td>3.199047</td>
      <td>4.148949</td>
      <td>1.852731</td>
      <td>3.269642</td>
      <td>3.448719</td>
      <td>4.458060</td>
      <td>3.719499</td>
      <td>3.231820</td>
      <td>3.521511</td>
      <td>3.866924</td>
      <td>3.961768</td>
      <td>4.957933</td>
      <td>4.075665</td>
      <td>3.509040</td>
      <td>3.923190</td>
      <td>3.210152</td>
      <td>4.374122</td>
      <td>...</td>
      <td>3.546313</td>
      <td>3.207635</td>
      <td>2.082641</td>
      <td>3.302390</td>
      <td>1.821505</td>
      <td>3.814172</td>
      <td>4.227119</td>
      <td>3.699006</td>
      <td>3.009256</td>
      <td>4.605246</td>
      <td>4.712096</td>
      <td>4.284418</td>
      <td>3.095067</td>
      <td>3.214574</td>
      <td>0.990303</td>
      <td>1.805794</td>
      <td>4.588016</td>
      <td>2.295002</td>
      <td>4.173353</td>
      <td>0.327724</td>
      <td>4.817989</td>
      <td>1.902907</td>
      <td>3.557027</td>
      <td>2.881273</td>
      <td>3.766529</td>
      <td>2.703354</td>
      <td>2.395317</td>
      <td>2.373198</td>
      <td>4.749076</td>
      <td>4.281203</td>
      <td>1.402608</td>
      <td>4.208382</td>
      <td>3.705957</td>
      <td>2.720514</td>
      <td>2.787331</td>
      <td>3.475076</td>
      <td>3.253458</td>
      <td>2.161087</td>
      <td>4.010495</td>
      <td>0.859474</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3.170119</td>
      <td>3.657992</td>
      <td>3.308707</td>
      <td>4.166521</td>
      <td>4.311890</td>
      <td>1.275469</td>
      <td>4.237972</td>
      <td>1.900366</td>
      <td>3.392859</td>
      <td>3.647421</td>
      <td>1.489588</td>
      <td>3.617857</td>
      <td>3.785199</td>
      <td>3.168660</td>
      <td>3.537318</td>
      <td>0.995625</td>
      <td>3.969397</td>
      <td>2.173005</td>
      <td>3.464055</td>
      <td>2.475622</td>
      <td>3.413724</td>
      <td>2.665215</td>
      <td>1.828840</td>
      <td>3.322109</td>
      <td>2.654698</td>
      <td>1.469953</td>
      <td>3.035060</td>
      <td>3.163879</td>
      <td>4.244324</td>
      <td>2.727754</td>
      <td>2.879571</td>
      <td>3.124665</td>
      <td>3.773794</td>
      <td>3.774747</td>
      <td>3.175855</td>
      <td>3.458016</td>
      <td>2.923885</td>
      <td>3.303497</td>
      <td>2.806202</td>
      <td>3.504966</td>
      <td>...</td>
      <td>3.289954</td>
      <td>2.677164</td>
      <td>2.087793</td>
      <td>3.388524</td>
      <td>1.783418</td>
      <td>3.267824</td>
      <td>3.661620</td>
      <td>3.131275</td>
      <td>2.475330</td>
      <td>3.916692</td>
      <td>4.197842</td>
      <td>3.987094</td>
      <td>3.134310</td>
      <td>2.827407</td>
      <td>0.829738</td>
      <td>1.380996</td>
      <td>3.974255</td>
      <td>2.685338</td>
      <td>3.902178</td>
      <td>0.293003</td>
      <td>3.064224</td>
      <td>1.566051</td>
      <td>3.095034</td>
      <td>2.769578</td>
      <td>3.956414</td>
      <td>2.493763</td>
      <td>2.236924</td>
      <td>1.775576</td>
      <td>3.909241</td>
      <td>3.799859</td>
      <td>0.973811</td>
      <td>3.528264</td>
      <td>3.361532</td>
      <td>2.672535</td>
      <td>2.404456</td>
      <td>4.232789</td>
      <td>2.911602</td>
      <td>1.634576</td>
      <td>4.135735</td>
      <td>0.725684</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2.307073</td>
      <td>1.658853</td>
      <td>1.443538</td>
      <td>2.208859</td>
      <td>2.229486</td>
      <td>0.780760</td>
      <td>1.997043</td>
      <td>0.924908</td>
      <td>2.970700</td>
      <td>2.551446</td>
      <td>0.881095</td>
      <td>1.813452</td>
      <td>2.687841</td>
      <td>1.908641</td>
      <td>2.228256</td>
      <td>0.695248</td>
      <td>1.146590</td>
      <td>1.536595</td>
      <td>0.809632</td>
      <td>1.561342</td>
      <td>1.820714</td>
      <td>1.097596</td>
      <td>1.216409</td>
      <td>1.347617</td>
      <td>1.760926</td>
      <td>0.622817</td>
      <td>1.786144</td>
      <td>1.934932</td>
      <td>2.332054</td>
      <td>2.291151</td>
      <td>1.983643</td>
      <td>1.785523</td>
      <td>2.265654</td>
      <td>2.055809</td>
      <td>2.459728</td>
      <td>2.092599</td>
      <td>2.512530</td>
      <td>2.928443</td>
      <td>1.777471</td>
      <td>1.808872</td>
      <td>...</td>
      <td>1.779506</td>
      <td>2.222377</td>
      <td>1.448616</td>
      <td>2.340729</td>
      <td>1.658322</td>
      <td>2.231055</td>
      <td>2.634708</td>
      <td>2.235721</td>
      <td>1.340105</td>
      <td>2.322287</td>
      <td>2.483354</td>
      <td>2.199769</td>
      <td>2.313019</td>
      <td>1.807883</td>
      <td>0.617402</td>
      <td>0.906815</td>
      <td>3.362981</td>
      <td>2.024704</td>
      <td>2.460702</td>
      <td>0.128483</td>
      <td>3.936125</td>
      <td>1.135435</td>
      <td>1.912071</td>
      <td>2.419887</td>
      <td>3.416503</td>
      <td>1.601437</td>
      <td>1.177825</td>
      <td>1.159584</td>
      <td>2.617399</td>
      <td>2.675379</td>
      <td>0.520354</td>
      <td>1.709494</td>
      <td>2.281596</td>
      <td>1.782833</td>
      <td>1.635173</td>
      <td>1.323276</td>
      <td>2.887580</td>
      <td>1.042618</td>
      <td>2.293890</td>
      <td>0.396941</td>
    </tr>
  </tbody>
</table>
<p>3 rows × 9719 columns</p>
</div>




```python
# 사용자가 관람하지 않은 영화명
unseen_list = get_unseen_movies(ratings_matrix, 9)

# 잠재 요인 협업 필터링으로 영화 추천
recomm_movies = recomm_movie_by_userid(ratings_pred_matrix, 9, unseen_list, top_n=10)

# 평점 데이터를 데이터프레임으로 생성.
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
      <th>Rear Window (1954)</th>
      <td>5.704612</td>
    </tr>
    <tr>
      <th>South Park: Bigger, Longer and Uncut (1999)</th>
      <td>5.451100</td>
    </tr>
    <tr>
      <th>Rounders (1998)</th>
      <td>5.298393</td>
    </tr>
    <tr>
      <th>Blade Runner (1982)</th>
      <td>5.244951</td>
    </tr>
    <tr>
      <th>Roger &amp; Me (1989)</th>
      <td>5.191962</td>
    </tr>
    <tr>
      <th>Gattaca (1997)</th>
      <td>5.183179</td>
    </tr>
    <tr>
      <th>Ben-Hur (1959)</th>
      <td>5.130463</td>
    </tr>
    <tr>
      <th>Rosencrantz and Guildenstern Are Dead (1990)</th>
      <td>5.087375</td>
    </tr>
    <tr>
      <th>Big Lebowski, The (1998)</th>
      <td>5.038690</td>
    </tr>
    <tr>
      <th>Star Wars: Episode V - The Empire Strikes Back (1980)</th>
      <td>4.989601</td>
    </tr>
  </tbody>
</table>
</div>



## 파이썬 추천 시스템 패키지 - Surprise

### Surprise 패키지 소개

* 파이썬 기반의 추천 시스템 구축을 위한 패키지인 Surprise
    * 사이킷런 유사 API, 프레임워크 제공
    * 다양한 추천 알고리즘으로, 다양한 잠재 요인 협업 필터링을 적용해 추천 시스템 구축 가능
    * 사이킷런의 핵심 API 유사하다.
    * 추천 시스템을 위한 모델 선택, 평가, 하이퍼 파라미터 튜닝 등 기능 제공

### Surprise 이용한 추천 시스템 구축


```python
# SVD 행렬 분해를 통한 잠재 요인 협업 필터링 수행
from surprise import SVD
from surprise import Dataset
from surprise import accuracy
from surprise.model_selection import train_test_split
```


```python
# Surprise 데이터 로딩 : Dataset 클래스로 가능하다.
data = Dataset.load_builtin('ml-100k')

trainset, testset = train_test_split(data, test_size=.25, random_state=0)
```

    Dataset ml-100k could not be found. Do you want to download it? [Y/n] y
    Trying to download dataset from http://files.grouplens.org/datasets/movielens/ml-100k.zip...
    Done! Dataset ml-100k has been saved to /Users/statstics/.surprise_data/ml-100k



```python
# SVD로 잠재 요인 협업 필터링 수행
algo = SVD()
algo.fit(trainset)
```




    <surprise.prediction_algorithms.matrix_factorization.SVD at 0x116abdf90>




```python
# 테스트 데이터 세트 전체에 대해 추천 영화 평점 데이터 생성 후 최초 5개만 추출 예제
predictions = algo.test(testset)
print('prediction type :', type(predictions), ' size:', len(predictions))
print('prediction 결과의 최초 5개 추출')
predictions[:5]
```

    prediction type : <class 'list'>  size: 25000
    prediction 결과의 최초 5개 추출





    [Prediction(uid='120', iid='282', r_ui=4.0, est=3.443514849725046, details={'was_impossible': False}),
     Prediction(uid='882', iid='291', r_ui=4.0, est=3.8225505828385393, details={'was_impossible': False}),
     Prediction(uid='535', iid='507', r_ui=5.0, est=3.95705699379237, details={'was_impossible': False}),
     Prediction(uid='697', iid='244', r_ui=5.0, est=3.544162907061212, details={'was_impossible': False}),
     Prediction(uid='751', iid='385', r_ui=4.0, est=3.2749055271557257, details={'was_impossible': False})]




```python
# Prediction 객체 3개에서 uid, iid, est 속성 추출
[ (pred.uid, pred.iid, pred.est) for pred in predictions[:3]]
```




    [('120', '282', 3.443514849725046),
     ('882', '291', 3.8225505828385393),
     ('535', '507', 3.95705699379237)]




```python
# 사용자 아이디, 아이템 아이디는 문자열로 입력해야 한다.
uid = str(196)
iid = str(302)
pred = algo.predict(uid, iid)
print(pred)
```

    user: 196        item: 302        r_ui = None   est = 4.26   {'was_impossible': False}



```python
accuracy.rmse(predictions)
```

    RMSE: 0.9455





    0.9455030769870753



* Surprise 패키지로 쉽게 추천 시스템 구현 가능하다.

### Surprise 주요 모듈 소개

#### Dataset

* API 명과 내용
* Dataset.load_builtin(name='ml-100k')
  * 무비렌즈 아카이브에서 데이터를 내려받는다. 디폴트는 ml-100k로 설정한다.
* Dataset.load_from_file(file_path, reader)
  * OS 파일에서 데이터 로딩할 때 사용한다.
  * 콤마, 탭 등으로 칼럼이 분리된 포맷의 OS 파일에서 데이터 로딩한다.
* Dataset.load_from_df(df, reader)
  * 판다스의 데이터프레임에서 데이터를 로딩한다.
  * 파라미터로 DF 입력받는다.
  * 반드시 3개의 칼럼인 사용자 아이디, 아이템 아이디, 평점 순 칼럼 순서 지정.
  * 입력 파라미터로 DF 객체, Reader 파일 포맷 지정.

#### OS 파일 데이터를 Surprise 데이터 세트로 로딩


```python
import pandas as pd

ratings = pd.read_csv('./ml-latest-small/ratings.csv')

# ratings_noh.csv 파일로 업로드 시 인덱스와 헤더를 모두 제거한 새로운 파일 생성.
ratings.to_csv('./ml-latest-small/ratings_noh.csv', index=False, header=False)
```


```python
# 칼럼명, 칼럼 분리문자, 최소에서 최대 평점 입력해 객체 생성
# 데이터 파일을 파싱하면서 로딩한다.

from surprise import Reader

reader = Reader(line_format='user item rating timestamp', sep=',', rating_scale=(0.5, 5))
data=Dataset.load_from_file('./ml-latest-small/ratings_noh.csv', reader=reader)
```

* Reader 클래스 파라미터
    * line_format(string) : 칼럼을 나열하되, 공백으로 분리해 문자열을 칼럼으로 인식
    * sep (char) : 칼럼 분리자로, 디폴트 '\t'이고, 판다스 데이터프레임 입력받으면 기재할 필요가 없다.
    * rating_scale(tuple, optional) : 평점 값의 (최소, 최대) 평점 설정한다.


```python
trainset, testset = train_test_split(data, test_size=.25, random_state=0)

algo = SVD(n_factors=50, random_state=0)

# 학습 데이터 세트로 학습하고 나서 테스트 데이터 세트로 평점 예측 후 RMSE 평가
algo.fit(trainset)
predictions = algo.test(testset)
accuracy.rmse(predictions)
```

    RMSE: 0.8682





    0.8681952927143516



#### 판다스 DataFrame에서 Surprise 데이터 세트로 로딩


```python
import pandas as pd
from surprise import Reader, Dataset

ratings = pd.read_csv('./ml-latest-small/ratings.csv')
reader = Reader(rating_scale=(0.5, 5.0))

data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)
trainset, testset = train_test_split(data, test_size=.25, random_state=0)

algo = SVD(n_factors=50, random_state=0)
algo.fit(trainset)
predictions = algo.test(testset)
accuracy.rmse(predictions)
```

    RMSE: 0.8682





    0.8681952927143516



### Surprise 추천 알고리즘 클래스

* SVD : 행렬 분해를 통한 잠재 요인 협업 필터링 위한 SVD 알고리즘.
* KNNBasic : 최근접 이웃 협업 필터링을 위한 KNN 알고리즘.
* BaselineOnly : 사용자 편향과 아이템 편향을 감안한 SGD 베이스라인 알고리즘.

* n_factors : 잠재 요인 K의 개수(100)
* n_epochs : SGD 수행 시 반복 횟수(20)
* biased(bool) : 베이스라인 사용자 편향 적용 여부.(True)

### 베이스라인 평점

* Baseline Rating : 개인의 성향을 반영해 아이템 평가에 편향성 요소를 반영하여 평점 부과하는 방식
    * 전체 평균 평점 + 사용자 편향 점수 + 아이템 편향 점수 공식

### 교차 검증과 하이퍼 파라미터 튜닝


```python
from surprise.model_selection import cross_validate

ratings = pd.read_csv('./ml-latest-small/ratings.csv') # reading data in pandas df
reader = Reader(rating_scale=(0.5, 5.0))
data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)

algo = SVD(random_state=0)
cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)
```

    Evaluating RMSE, MAE of algorithm SVD on 5 split(s).

                      Fold 1  Fold 2  Fold 3  Fold 4  Fold 5  Mean    Std     
    RMSE (testset)    0.8769  0.8711  0.8674  0.8766  0.8768  0.8738  0.0038  
    MAE (testset)     0.6754  0.6643  0.6669  0.6737  0.6730  0.6707  0.0043  
    Fit time          4.23    4.07    4.17    4.12    4.09    4.14    0.06    
    Test time         0.11    0.29    0.10    0.11    0.29    0.18    0.09    





    {'test_rmse': array([0.87690212, 0.87113905, 0.86743438, 0.87661392, 0.87679315]),
     'test_mae': array([0.6753898 , 0.66434446, 0.66694453, 0.67367633, 0.67301845]),
     'fit_time': (4.228317022323608,
      4.073528051376343,
      4.166388273239136,
      4.117280006408691,
      4.090139150619507),
     'test_time': (0.1060647964477539,
      0.29105687141418457,
      0.1040351390838623,
      0.10667800903320312,
      0.28515100479125977)}




```python
from surprise.model_selection import GridSearchCV

# 최적화할 파라미터를 딕셔너리 형태로 지정
param_grid = {'n_epochs':[20, 40, 60], 'n_factors':[50, 100, 200]}

# CV를 3개 폴드 셋으로 지정하고 성능 평가는 rmse, mse로 수행하도록 GridSearchCV 구성
gs = GridSearchCV(SVD, param_grid, measures=['rmse', 'mae'], cv=3)
gs.fit(data)

print(gs.best_score['rmse'])
print(gs.best_params['rmse'])
```

    0.878826727057796
    {'n_epochs': 20, 'n_factors': 50}


### Surprise 이용한 개인화 영화 추천 시스템 구축


```python
data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)
algo = SVD(n_factors=50, random_state=0)
algo.fit(data)
```


    ---------------------------------------------------------------------------

    AttributeError                            Traceback (most recent call last)

    <ipython-input-22-8ce12e86a0c5> in <module>
          1 data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)
          2 algo = SVD(n_factors=50, random_state=0)
    ----> 3 algo.fit(data)


    /usr/local/lib/python3.7/site-packages/surprise/prediction_algorithms/matrix_factorization.pyx in surprise.prediction_algorithms.matrix_factorization.SVD.fit()


    /usr/local/lib/python3.7/site-packages/surprise/prediction_algorithms/matrix_factorization.pyx in surprise.prediction_algorithms.matrix_factorization.SVD.sgd()


    AttributeError: 'DatasetAutoFolds' object has no attribute 'global_mean'



```python
from surprise.dataset import DatasetAutoFolds

reader = Reader(line_format='user item rating timestamp', sep=',', rating_scale=(0.5, 5))

data_folds = DatasetAutoFolds(ratings_file='./ml-latest-small/ratings_noh.csv', reader=reader)

trainset = data_folds.build_full_trainset()
```


```python
algo = SVD(n_epochs=20, n_factors=50, random_state=0)
algo.fit(trainset)
```




    <surprise.prediction_algorithms.matrix_factorization.SVD at 0x11fd5e310>




```python
# 영화
movies = pd.read_csv('./ml-latest-small/movies.csv')
# userId = 9
movieIds = ratings[ratings['userId']==9]['movieId']

if movieIds[movieIds==42].count() == 0:
    print('사용자 아이디 0에 영화 아이디 42 평점은 없다.')

print(movies[movies['movieId']==42])
```

    사용자 아이디 0에 영화 아이디 42 평점은 없다.
        movieId                   title              genres
    38       42  Dead Presidents (1995)  Action|Crime|Drama



```python
# 문자열로 만들어야 predict() 입력 가능
uid = str(9)
iid = str(42)

pred = algo.predict(uid, iid, verbose=True)
```

    user: 9          item: 42         r_ui = None   est = 3.13   {'was_impossible': False}



```python
def get_unseen_surprise(ratings, movies, userId):
    seen_movies = ratings[ratings['userId']== userId]['movieId'].tolist()

    total_movies = movies['movieId'].tolist()

    unseen_movies = [movie for movie in total_movies if movie not in seen_movies]
    print('평점 매긴 영화 수:', len(seen_movies), '추천 대상 영화 수:', len(unseen_movies),
         '전체 영화 수:', len(total_movies))

    return unseen_movies

unseen_movies = get_unseen_surprise(ratings, movies, 9)
```

    평점 매긴 영화 수: 46 추천 대상 영화 수: 9696 전체 영화 수: 9742



```python
def recomm_movie_by_surprise(algo, userId, unseen_movies, top_n=10):

    predictions = [algo.predict(str(userId), str(movieId)) for movieId in unseen_movies]

    # est로 정렬하고자 아래 함수 정의
    # 이 함수는 리스트 객체의 sort() 함수의 키 값 사용해서 정렬
    def sortkey_est(pred):
        return pred.est

    predictions.sort(key=sortkey_est, reverse=True)
    top_predictions = predictions[:top_n]

    # top_n으로 추출된 영화의 정보 추출. 영화 아이디, 추천 예상 평점, 제목 추출
    top_movie_ids = [ int(pred.iid) for pred in top_predictions]
    top_movie_rating = [ pred.est for pred in top_predictions]
    top_movie_titles = movies[movies.movieId.isin(top_movie_ids)]['title']

    top_movie_preds = [ (id, title, rating) for id, title, rating in \
                      zip(top_movie_ids, top_movie_titles, top_movie_rating)]

    return top_movie_preds

unseen_movies = get_unseen_surprise(ratings, movies, 9)
top_movie_preds = recomm_movie_by_surprise(algo, 9, unseen_movies, top_n=10)

print('##### Top-10 추천 영화 리스트 #####')
for top_movie in top_movie_preds:
    print(top_movie[1], ":", top_movie[2])

```

    평점 매긴 영화 수: 46 추천 대상 영화 수: 9696 전체 영화 수: 9742
    ##### Top-10 추천 영화 리스트 #####
    Usual Suspects, The (1995) : 4.306302135700814
    Star Wars: Episode IV - A New Hope (1977) : 4.281663842987387
    Pulp Fiction (1994) : 4.278152632122758
    Silence of the Lambs, The (1991) : 4.226073566460876
    Godfather, The (1972) : 4.1918097904381995
    Streetcar Named Desire, A (1951) : 4.154746591122658
    Star Wars: Episode V - The Empire Strikes Back (1980) : 4.122016128534504
    Star Wars: Episode VI - Return of the Jedi (1983) : 4.108009609093436
    Goodfellas (1990) : 4.083464936588478
    Glory (1989) : 4.07887165526957


* 9번 ID 사용자 기준
    * 브래드 피트 주연 Seven, 케빈 스페이시 주연 Usual Suspect, 대부, 좋은 친구들
    * 스릴러/범죄 영화 및 스타워즈 같은 액션 영화 추천

## 정리

* 파이썬의 추천 시스템 패키지 Surprise
    * 사이킷런과 유사한 API 지향
    * 간단한 API만으로 파이썬 기반 추천 시스템 구현
