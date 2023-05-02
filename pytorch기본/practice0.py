import torch
import torch.nn as nn
import torch.optim as optim

torch.manual_seed(1) # 나중에 같은 결과를 위하여 seed고정

x_train = torch.FloatTensor([[1],[2],[3]])
y_train = torch.FloatTensor([[2],[4],[6]])

#weight bias 선언해주기
W = torch.zeros(1,requires_grad=True)
b = torch.zeros(1,requires_grad=True)

optimizer = optim.SGD([W,b],lr=0.01)

nb_epochs = 1999
for i in range(nb_epochs+1):
    hypo = W*x_train + b


    loss = torch.mean((hypo-y_train)**2)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if i%100 ==0:
        print(W.item(),b.item())
        print(loss.item())

    #결과를 보면 W는 거의 2에 가깝고 b는 거의 0에 가깝다.

