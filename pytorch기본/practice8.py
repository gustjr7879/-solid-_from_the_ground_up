import matplotlib.pyplot as plt
from sklearn.datasets import load_digits

#첫번째 샘플 출력 .images[인덱스] 로 해당 인덱스의 이미지를 행렬로 출력가능

digits = load_digits()
print(digits.images[0])
#0을 흰색 도화지, 0보다 큰 숫자들을 검정색 점이라고 생각해보면 0같은 실루엣이 보임
#다음으로는 label을 출력해보자

print(digits.target[0])
#label 도 0이라고 나온다.
print(len(digits.images))
# 1797개의 이미지 샘플이 존재하는 것을 확인할 수 있다.

#몇개만 뽑아서 시각화해보자
images_and_labels = list(zip(digits.images, digits.target))
#for index, (image, label) in enumerate(images_and_labels[:5]): # 5개의 샘플만 출력
#    plt.subplot(2, 5, index + 1)
#    plt.axis('off')
#    plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
#    plt.title('sample: %i' % label)
    #plt.show()

# 훈련 데이터와 레이블을 X,Y에 저장해보자. digits.data를 이용하여서 데이터를 얻을 수 있음

X = digits.data # image 의 feature
Y = digits.target # image의 label
# 다층 퍼셉트론 분류기 만들기
import torch
import torch.nn as nn
from torch import optim

model = nn.Sequential(
    nn.Linear(64,32),
    nn.ReLU(),
    nn.Linear(32,16),
    nn.ReLU(),
    nn.Linear(16,10) # label 이 10차원
)
X = torch.tensor(X,dtype=torch.float32)
Y = torch.tensor(Y,dtype=torch.int64)
loss_fn = nn.CrossEntropyLoss() # 이 loss함수는 소프트맥스 함수를 포함하고 있음
optimizer = optim.Adam(model.parameters())
losses = []

for epoch in range(100):
    optimizer.zero_grad()
    pred = model(X)
    loss = loss_fn(pred,Y)
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(loss)
    losses.append(loss.item())

plt.plot(losses)
plt.show()