import numpy as np
import matplotlib.pyplot as plt

# numpy를 이용해서 심층 신경망 구현하는 실습

# image data
one = [0,0,1,0,0,0,
        0,0,1,0,0,0,
        0,0,1,0,0,0,
        0,0,1,0,0,0,
        0,0,1,0,0,0]

two = [0,1,1,1,1,0,
        0,0,0,0,1,0,
        0,0,0,1,0,0,
        0,0,1,0,0,0,
        0,1,1,1,1,0]

three = [0,1,1,1,1,0,
        0,0,0,0,1,0,
        0,0,1,1,1,0,
        0,0,0,0,1,0,
        0,1,1,1,1,0]
# label
y = [[1,0,0],[0,1,0],[0,0,1]]


# see image
'''
plt.imshow(np.array(one).reshape(5,6))
plt.show()

plt.imshow(np.array(two).reshape(5,6))
plt.show()

plt.imshow(np.array(three).reshape(5,6))
plt.show()
'''
# x is image feature vector, image data flatten
# y is label
x = [np.array(one).reshape(1,30),np.array(two).reshape(1,30),np.array(three).reshape(1,30)]
y = np.array(y)

#x를 평평하게 만들어서 30픽셀에 대한 feature를 input으로 하면서 시작함

#hidden layer와 output layer에 사용할 활성화함수

def sigmoid(x):
    return (1/(1+np.exp(-x)))

# 출력이 실제값과 얼마나 떨어져있나 보여주는 loss함수
# mean squared error
def loss(out,y):
    s = (np.square(out-y))
    s = np.sum(s)/len(y)
    return s

# feed foward fuction

def feed_foward(x,w1,w2):
    #hidden layer
    z1 = x.dot(w1) # input of layer1
    a1 = sigmoid(z1) # output of layer1

    #output layer
    z2 = a1.dot(w2) # input of layer 2
    a2 = sigmoid(z2) # output of layer 2
    return a2

#초기 가중치 
def generate_weights(x,y):
    weights = []
    for i in range(x*y):
        weights.append(np.random.randn())
    return np.array(weights).reshape(x,y)

def back_propagation(x,y,w1,w2,learning_rate):
    #hidden layer
    z1 = x.dot(w1) # input of layer 1
    a1 = sigmoid(z1) # output of layer 1

    #output layer

    z2 = a1.dot(w2) # input of layer2
    a2 = sigmoid(z2) # output of layer2

    d2 = (a2 -y) # 실제값과 distance
    d1 = np.multiply(w2.dot((d2.transpose())).transpose(),(np.multiply(a1,1-a1)))
    
    # gradient for w1 and w2
    w1_adj = x.transpose().dot(d1)
    w2_adj = a1.transpose().dot(d2)

    # updating parameters
    w1 = w1-(learning_rate*(w1_adj))
    w2 = w2-(learning_rate*(w2_adj))
    return(w1,w2)

def training(x,y,w1,w2,learning_rate=0.01,epochs = 10):
    accuracies = []
    losses = []

    for j in range(epochs):
        l = []
        for i in range(len(x)):
            out = feed_foward(x[i],w1,w2)
            l.append(loss(out,y[i]))
            w1,w2 = back_propagation(x[i],y[i],w1,w2,learning_rate)
        acc = (1-(sum(l)/len(x)))*100
        print('epochs:',j+1,'---->acc:',acc)
        accuracies.append(acc)
        losses.append(sum(l)/len(x))
    return accuracies,losses,w1,w2

def predict(x,w1,w2):
    output = feed_foward(x,w1,w2)
    maxim = 0
    k = 0
    for i in range(len(output[0])):
        if (maxim<output[0][i]):
            maxim = output[0][i]
            k = i
    if k == 0:
        print('image is 1')
    elif k ==1:
        print('image is 2')
    else:
        print('image is 3')
    plt.imshow(x.reshape(5,6))
    plt.show()

w1 = generate_weights(30,5) # input -> output
w2 = generate_weights(5,3)  # input -> output

print(w1,'\n\n',w2)

accuraccies,losses,trained_w1,trained_w2 = training(x,y,w1,w2,0.05,100)

plt.plot(accuraccies)
plt.xlabel('epochs')
plt.ylabel('acc')
plt.show()

plt.plot(losses)
plt.xlabel('epochs')
plt.ylabel('loss')
plt.show()

predict(x[2],trained_w1,trained_w2)