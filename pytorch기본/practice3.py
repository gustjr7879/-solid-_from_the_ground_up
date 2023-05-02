import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
torch.manual_seed(1)

x_train  =  torch.FloatTensor([[73,  80,  75], 
                               [93,  88,  93], 
                               [89,  91,  80], 
                               [96,  98,  100],   
                               [73,  66,  70]])  
y_train  =  torch.FloatTensor([[152],  [185],  [180],  [196],  [142]])

# 모델 초기화
W = torch.zeros((3, 1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)
# optimizer 설정
optimizer = optim.SGD([W, b], lr=1e-5)
epochs = 1000
for i in range(epochs):
    hypo = x_train.matmul(W) + b #b는 브로드ㅜ캐스팅되어서 더해진다.
    loss = torch.mean((hypo - y_train)**2)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if i %100 == 0:
        print(loss)
print(W,b.item())