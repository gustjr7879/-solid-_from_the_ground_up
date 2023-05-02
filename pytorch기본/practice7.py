from audioop import avg
import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import random


USE_CUDA = torch.cuda.is_available()
device = torch.device('cuda' if USE_CUDA else 'cpu')
print('device on', device)

random.seed(777)
torch.manual_seed(777)

training_epochs = 15
batch_size = 128

mnist_train = dsets.MNIST(root='MNIST',train=True,transform=transforms.ToTensor(),download=True)
mnist_test = dsets.MNIST(root='MNIST',train=False,transform=transforms.ToTensor(),download=True)

data_loader = DataLoader(mnist_train,batch_size=batch_size,shuffle=True,drop_last=True)
#drop last 는 배치사이즈로 다 쪼개고 나서 나머지를 버릴것인지 말것인지를 판단해주는데, 보통 마지막에 남는애들은 경사하강법을 진행할 때 과대평가되기 때문이다.

linear = nn.Linear(784,10,bias=True).to(device)
loss_func = nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.SGD(linear.parameters(),lr = 1e-1)

for epoch in range(training_epochs):
    avg_cost = 0
    total_batch = len(data_loader)

    for X,Y in data_loader:
        X = X.view(-1,28*28).to(device)
        Y = Y.to(device)
        optimizer.zero_grad()
        pred = linear(X)
        loss = loss_func(pred,Y)
        loss.backward()
        optimizer.step()
        avg_cost += loss / total_batch
    print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost))
# 테스트 데이터를 사용하여 모델을 테스트한다.
with torch.no_grad(): # torch.no_grad()를 하면 gradient 계산을 수행하지 않는다.
    X_test = mnist_test.test_data.view(-1, 28 * 28).float().to(device)
    Y_test = mnist_test.test_labels.to(device)

    prediction = linear(X_test)
    correct_prediction = torch.argmax(prediction, 1) == Y_test
    accuracy = correct_prediction.float().mean()
    print('Accuracy:', accuracy.item())

    # MNIST 테스트 데이터에서 무작위로 하나를 뽑아서 예측을 해본다
    r = random.randint(0, len(mnist_test) - 1)
    X_single_data = mnist_test.test_data[r:r + 1].view(-1, 28 * 28).float().to(device)
    Y_single_data = mnist_test.test_labels[r:r + 1].to(device)

    print('Label: ', Y_single_data.item())
    single_prediction = linear(X_single_data)
    print('Prediction: ', torch.argmax(single_prediction, 1).item())

    plt.imshow(mnist_test.test_data[r:r + 1].view(28, 28), cmap='Greys', interpolation='nearest')
    plt.show()