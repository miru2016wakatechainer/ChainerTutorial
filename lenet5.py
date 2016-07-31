from chainer import Chain
import chainer.functions as F
import chainer.links as L

class LeNet5(Chain):
    def __init__(self):
        super(LeNet5, self).__init__(
            conv1=L.Convolution2D(1, 6, 5),
            conv2=L.Convolution2D(6, 16, 5),
            conv3=L.Convolution2D(16, 120, 5),
            l1=L.Linear(120, 84),
            l2=L.Linear(84, 10))
        self.is_train = True
        
    def __call__(self, x):
        h1 = F.sigmoid(F.average_pooling_2d(self.conv1(x), 2))
        h2 = F.sigmoid(F.average_pooling_2d(self.conv2(h1),2))
        h3 = self.conv3(h2)
        h4 = F.tanh(self.l1(h3))
        p = self.l2(h4)
    
        return p