import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
torch.manual_seed(1)
#훈련 데이터 선언
x1_train = torch.FloatTensor([[73], [93], [89], [96], [73]])
x2_train = torch.FloatTensor([[80], [88], [91], [98], [66]])
x3_train = torch.FloatTensor([[75], [93], [90], [100], [70]])
y_train = torch.FloatTensor([[152], [185], [180], [196], [142]])
#가중치 w와 편향 b 초기화
w1 = torch.zeros(1, requires_grad=True)
w2 = torch.zeros(1, requires_grad=True)
w3 = torch.zeros(1, requires_grad=True)
b = torch.zeros(1, requires_grad=True)

#optimizer 설정
optimizer = optim.SGD([w1,w2,w3,b],lr = 1e-5)
epochs = 1000
for i in range(epochs):
    hypo = x1_train*w1 + x2_train*w2 + x3_train*w3 + b
    loss = torch.mean((hypo - y_train)**2)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if i %100 == 0:
        print(loss)
print(w1.item(),w2.item(),w3.item(),b.item())




