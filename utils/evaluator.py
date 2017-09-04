import numpy as np
from time import time

class Evaluator(object):
    def __init__(self, func, img_size):
        self.loss_value = None
        self.grads_values = None
        self.func = func
        self.img_size = img_size

    def eval_loss_and_grads(self, x):
        x = x.reshape((1, 1, ) + self.img_size)
        #  start = time() 
        outs = self.func([x])
        #  end = time()
        #  print 'Loss evaluation time', end - start
        loss_value = outs[0]
        if len(outs[1:]) == 1:
            grad_values = outs[1].flatten().astype('float64')
        else:
            grad_values = np.array(outs[1:]).flatten().astype('float64')
        return loss_value, grad_values

    def loss(self, x):
        assert self.loss_value is None
        loss_value, grad_values = self.eval_loss_and_grads(x)
        self.loss_value = loss_value
        self.grad_values = grad_values
        return self.loss_value

    def grads(self, x):
        assert self.loss_value is not None
        grad_values = np.copy(self.grad_values)
        self.loss_value = None
        self.grad_values = None
        return grad_values


