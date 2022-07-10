from tensorflow.keras.datasets import mnist
import numpy as np

class Perceptron:
    
    def __init__(self, data, bath_size, input_size, output_size, optimizer) -> None:
        (self.x_train, self.y_train), (self.x_test, self.y_test) = data
        self.batch_size = bath_size
        self.W = np.zeros((output_size, input_size))
        self.b = np.zeros((output_size, 1))
        self.input_size = input_size
        self.output_size = output_size
        self.m_dw, self.v_dw = 0, 0
        self.m_db, self.v_db = 0, 0
        self.eta = 0.01
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.epsilon = 1e-8
        self.optimizer = optimizer
        
    def softmax(self, z):
        exp_list = np.exp(z)
        return 1/sum(exp_list) * exp_list

    def entropy_loss(self, pred, label):
        loss = -np.log(pred[int(label)])
        return loss

    def compute_batch_output(self, X):
        return np.array([np.matmul(self.W.T, x) + self.b for x in X])
    
    def compute_SGD_batch_gradient(self, x_batch, y_batch):
        y_true = np.array([[1 if j == y_batch[i] else 0 for j in range(self.output_size)] for i in range(len(y_batch))])    # one-hot encoding for y-labels
        y_pred = np.array([self.softmax(np.dot(self.W, x_batch[i]) + np.squeeze(self.b)) for i in range(len(y_batch))])     # y = softmax(Wx + b)
        w_grad = (1/len(y_batch)) * np.dot(x_batch.transpose(), y_pred - y_true)                                            # compute batch gradient w.r.t. W
        b_grad = (1/len(y_batch)) * np.dot(np.ones((1,len(y_batch))), y_pred - y_true)                                      # compute batch gradient w.r.t. b
        return w_grad.transpose(), b_grad.transpose()
    
    def evaluate(self, X, Y):
        dist = np.array([self.softmax(np.dot(self.W, X[i]) + np.squeeze(self.b)) for i in range(len(Y))])
        result = np.argmax(dist,axis=1)
        accuracy = sum(result == Y) / float(len(Y))
        loss_list = [self.entropy_loss(dist[i], Y[i]) for i in range(len(Y))]
        loss = sum(loss_list) / len(loss_list)
        return loss, accuracy
    
    def SGD_update(self, dW, db):
        self.W -= self.eta * dW
        self.b -= self.eta * db
    
    def Adam_update(self, t, dw, db):
        self.m_dw = self.beta1*self.m_dw + (1-self.beta1)*dw
        self.m_db = self.beta1*self.m_db + (1-self.beta1)*db
        self.v_dw = self.beta2*self.v_dw + (1-self.beta2)*(dw**2)
        self.v_db = self.beta2*self.v_db + (1-self.beta2)*(db**2)
        m_dw_corr = self.m_dw/(1-self.beta1**t)
        m_db_corr = self.m_db/(1-self.beta1**t)
        v_dw_corr = self.v_dw/(1-self.beta2**t)
        v_db_corr = self.v_db/(1-self.beta2**t)
        self.W -= self.eta*(m_dw_corr/(np.sqrt(v_dw_corr)+self.epsilon))
        self.b -= self.eta*(m_db_corr/(np.sqrt(v_db_corr)+self.epsilon))
    
    def optimize_parameters2(self, iterations=100):
        train_loss_list, train_acc_list, test_loss_list, test_acc_list = [],[],[],[]
        for iteration in range(iterations):
            rand_indices = np.random.choice(self.x_train.shape[0], self.x_train.shape[0], replace=False)
            num_batch = self.x_train.shape[0] // self.batch_size
                
            for batch in range(num_batch):
                index = rand_indices[self.batch_size*batch:self.batch_size*(batch+1)]
                x_batch = self.x_train[index]
                y_batch = self.y_train[index]

                dW, db = self.compute_SGD_batch_gradient(x_batch, y_batch)
                if self.optimizer == "adam":
                    self.Adam_update(iteration+1, dW, db)
                else:
                    self.SGD_update(dW, db)

            train_loss, train_acc = self.evaluate(self.x_train, self.y_train)
            test_loss, test_acc = self.evaluate(self.x_test, self.y_test)
            train_loss_list.append(train_loss)
            train_acc_list.append(train_acc)
            test_loss_list.append(test_loss)
            test_acc_list.append(test_acc)
            
            message = 'Epoch %d/%d, Train Loss %.4f, Train Acc %.4f, Test Loss %.4f, Test Acc %.4f' % (iteration+1, iterations, train_loss, train_acc, test_loss, test_acc)
            print(message)
        return train_loss_list, train_acc_list, test_loss_list, test_acc_list
        
        
if __name__ == "__main__":
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = (x_train / 255).reshape(len(x_train), -1), (x_test / 255).reshape(len(x_test), -1)
    data = (x_train, y_train), (x_test, y_test)
    
    perceptron = Perceptron(data, bath_size=512, input_size=x_train.shape[-1], output_size=len(set(y_train)), optimizer="adam")
    perceptron.optimize_parameters2()
    