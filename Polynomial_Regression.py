import numpy as np
import matplotlib.pyplot as plt

class SGD:
    
    def __init__(self, eta=0.01):
        self.eta = eta
        
    def update(self, t, dW):
        return self.eta * dW
    
class Adam:
    
    def __init__(self, eta=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.m_dW, self.v_dW = 0, 0
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.eta = eta
        
    def update(self, t, dW):
        self.m_dW = self.beta1 * self.m_dW + (1-self.beta1) * dW
        self.v_dW = self.beta2 * self.v_dW + (1-self.beta2) * (dW**2)
        m_dW_corr = self.m_dW / (1-self.beta1**t)
        v_dW_corr = self.v_dW / (1-self.beta2**t)
        return self.eta * (m_dW_corr / (np.sqrt(v_dW_corr) + self.epsilon))

class Regressor:
    
    def __init__(self, f, interval, order, batch_size, optimizer):
        self.f = f
        self.interval = interval
        self.batch_size = batch_size
        self.W = np.zeros((order+1,))
        self.optimizer = optimizer

    def mse(self, y, y_):
        return (y - y_) ** 2
    
    def generate_samples(self):
        X = np.random.uniform(self.interval[0], self.interval[1], self.batch_size)
        Y = np.array([self.f(x) for x in X])
        X = np.array([[x**i for i in range(len(self.W))][::-1] for x  in X])
        return X, Y
    
    def compute_SGD_batch_gradient(self, X, Y):
        y_pred = np.squeeze([np.dot(self.W, x) for x in X]) 
        w_grad = (-2/len(Y)) * np.dot(X.T, Y - y_pred)
        return w_grad.transpose()
    
    def evaluate(self, X, Y):
        pred = np.array([np.dot(self.W, x) for x in X])
        loss_list = [self.mse(pred[i], Y[i]) for i in range(len(Y))]
        loss = np.sum(loss_list) / len(loss_list)
        return loss
    
    def gradient_descent(self, iterations):
        loss_list = []
        for iteration in range(iterations):
            x_batch, y_batch = self.generate_samples()
            dW = self.compute_SGD_batch_gradient(x_batch, y_batch)
            self.W -= self.optimizer.update(iteration+1, dW)
            
            if (iteration + 1) % 10 == 0:
                loss = self.evaluate(*self.generate_samples())
                loss_list.append(loss)
                message = 'Iteration %d/%d, Loss: %.4f, Params: %s' % (iteration+1, iterations, loss, str(self.W))
                print(message)
        return loss_list
    
    def plot_curves(self, coefficients, interval):
        plt.figure()
        X = np.linspace(interval[0], interval[1], 1000)
        Y = np.array([np.sum([self.W[::-1][i]*x**i for i in range(len(self.W))]) for x  in X])
        plt.plot(X, Y, label="Polynomial")
        X = np.linspace(interval[0], interval[1], 1000)
        Y = np.array([np.sum([coefficients[::-1][i]*x**i for i in range(len(coefficients))]) for x  in X])
        plt.plot(X, Y, label="Approximation")
        plt.legend()
        plt.show()


def polynomial(coefficients):
    def f(x):
        return np.sum([a*x**i for i,a in enumerate(coefficients[::-1])])
    return f    

def f1(w, a, b)
        
        
if __name__ == "__main__":
    
    coefficients = [1,2,3,4,5,6,7]
    f = polynomial(coefficients)
    interval = [-1, 1]
    optimizer = Adam()
    reg = Regressor(f, interval=interval, order=len(coefficients)-1, batch_size=100, optimizer=Adam())
    loss_list = reg.gradient_descent(iterations=10000)
    reg.plot_curves(coefficients, interval=interval)