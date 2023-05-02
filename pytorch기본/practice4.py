
import torch
import torch.nn as nn
import torch.nn.functional as F
torch.manual_seed(1)
x_train = torch.FloatTensor([[1], [2], [3]])
y_train = torch.FloatTensor([[2], [4], [6]])
model = nn.Linear(1,1)
#모델에는 가중치와 편향이 저장되어있는데, 이는 model.parameter()라는 함수를 사용하여서 불러올 수 있다.
print(list(model.parameters()))
optimizer = torch.optim.SGD(model.parameters(),lr = 1e-2)

epochs = 2000
for i in range(epochs):
    pred = model(x_train)
    loss = F.mse_loss(pred,y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print(loss)


x_train = torch.FloatTensor([[73, 80, 75],
                             [93, 88, 93],
                             [89, 91, 90],
                             [96, 98, 100],
                             [73, 66, 70]])
y_train = torch.FloatTensor([[152], [185], [180], [196], [142]])

model2 = nn.Linear(3,1)

optimizer2 = torch.optim.SGD(model2.parameters(),lr = 1e-5)
epochs = 2000

for i in range(epochs):
    pred = model2(x_train)

    loss2 = F.mse_loss(pred,y_train)

    optimizer2.zero_grad()
    loss2.backward()
    optimizer2.step()

print(loss2)