from surprise import SVD
from surprise import Dataset
from surprise.model_selection import cross_validate
import numpy as np

data = Dataset.load_builtin('ml-100k',prompt=False)

#a = data.raw_ratings[:10]
#print(a)
# [('196', '242', 3.0, '881250949')
#196번 유저가 242 아이템에 3.0

#model = SVD()
#cross_validate(model,data,measures=['rmse','mae'],cv=5, verbose=True)

#content-based filtering
#추천시스템 중 컨텐츠 기반 필터링 방법은 사용자의 이전 행동과 명시적인 피드백을 통하여서 좋아하는 것과 유사한 항목을 추천함
#예를들어서 내가 시청한 영화 목록과 다른 사용자가 시청한 영화 목록을 비교하여 비슷한 취향을 가진 사람에게 안본 영화를 추천
#유사도를 기반으로 추천함
#많은 수의 사람에게 확장 가능
#사용자가 관심을 갖지 않던 상품을 추천 가능
#입력 특성을 직접 설계해야 되기 때문에 도메인 지식이 필요함
#사용자의 기존 관심사항으로만 추천 가능함
raw_data = np.array(data.raw_ratings,dtype=int)
raw_data[:, 0] -=1 #0부터 시작하도록 조정하는 코드
raw_data[:, 1] -=1 #0부터 시작하도록 조정하는 코드
n_users = np.max(raw_data[:, 0])
n_movies = np.max(raw_data[:, 1])
shape = (n_users+1,n_movies+1)
#print(shape) (943,1682)
adj_matrix = np.ndarray(shape,dtype=int)
for user_id,movie_id,rating,time in raw_data:
    adj_matrix[user_id][movie_id] = 1

### 인접행렬을 통하여 dot 연산을 통한 유사도 추출
#print(adj_matrix)
my_id ,my_vector = 0, adj_matrix[0]
best_match, best_match_id, best_match_vector = -1, -1, []

for user_id , user_vector in enumerate(adj_matrix):
    if my_id != user_id:
        similarity = np.dot(my_vector,user_vector)
        if similarity > best_match:
            best_match = similarity
            best_match_id = user_id
            best_match_vector = user_vector
print('Best match {}, Best match ID {}'.format(best_match,best_match_id)) #183, 275

recommend_list = []
for i, log in enumerate(zip(my_vector,best_match_vector)):
    log1 , log2 = log
    if log1 < 1.0 and log2 >0.0:
        recommend_list.append(i)
#나와 유사한 사람이 본 영화를 내가 안봤을 경우 추천하는 원리
print(recommend_list)