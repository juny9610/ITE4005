import numpy as np
import pandas as pd
import sys

class MatrixFactorization():
    def __init__(self, rating, k, learning_rate, reg_param, epochs, test_data):
        """
        ratings: Rating matrix
        k: Latent parameter
        learning_rate: Learning rate
        reg_param: Regularization Strength
        epochs: Training epochs
        """

        self.rating = rating.values
        self.k = k
        self.learning_rate = learning_rate
        self.reg_param = reg_param
        self.epochs = epochs

        self.test_data = test_data

        self.n_users, self.n_items = rating.shape

        # init latent features
        self.m_user = np.random.normal(size=(self.n_users, k))
        self.m_item = np.random.normal(size=(self.n_items, k))

        # init biases
        self.b_user = np.zeros(self.n_users)
        self.b_item = np.zeros(self.n_items)
        self.bias = np.mean(self.rating[np.where(self.rating != 0)])

        self.user_id_idx = {}
        self.item_id_idx = {}
        
        for idx, user_id in enumerate(rating.index):
            self.user_id_idx[user_id] = idx
        for idx, item_id in enumerate(rating.columns):
            self.item_id_idx[item_id] = idx

    def train(self):
        self.training_process = []

        for epoch in range(self.epochs):
            for i in range(self.n_users):
                for j in range(self.n_items):
                    if self.rating[i][j] > 0:
                        self.gradient_descent(i, j)

            cost = self.get_cost()
            self.training_process.append((epoch, cost))

            print("Epoch: %d/%d, Cost = %.4f" %(epoch+1, self.epochs, cost))

    def gradient_descent(self, i, j):
        # get error
        prediction = self.get_prediction(i,j)
        error = self.rating[i][j] - prediction

        # update biases
        self.b_user[i] += self.learning_rate * (error - self.reg_param * self.b_user[i])
        self.b_item[j] += self.learning_rate * (error - self.reg_param * self.b_item[j])

        # update latent feature
        d_user = (error * self.m_item[j, :]) - (self.reg_param * self.m_user[i, :])
        d_item = (error * self.m_user[i, :]) - (self.reg_param * self.m_item[j, :])
        self.m_user[i, :] += self.learning_rate * d_user
        self.m_item[j, :] += self.learning_rate * d_item

    def get_prediction(self, i, j):
        return self.bias + self.b_user[i] + self.b_item[j] + np.dot(self.m_user[i, :], self.m_item[j, :].T)

    def get_cost(self):
        cost = 0
        xi, yi = self.rating.nonzero()
        predicted = self.bias + self.b_user[:, np.newaxis] + self.b_item[np.newaxis:, ] + np.dot(self.m_user, self.m_item.T)

        for x, y in zip(xi, yi):
            cost += pow(self.rating[x,y] - predicted[x,y], 2)
        
        return np.sqrt(cost) / len(xi)


    def test(self):
        test_user_id = self.test_data[:, 0]
        test_item_id = self.test_data[:, 1]
        R = self.bias + np.expand_dims(self.b_user, -1) + np.expand_dims(self.b_item, 0) + np.dot(self.m_user, self.m_item.T)
        
        ret = []
        for user_id, item_id in zip(test_user_id, test_item_id):
            if item_id in self.item_id_idx:
                user_idx = self.user_id_idx[user_id]
                item_idx = self.item_id_idx[item_id]
                
                r = max(0, R[user_idx][item_idx])
                r = min(5, r)
                ret.append(r)
            else:
                ret.append(self.bias)
	    
        return np.array(ret)

def read_data(train_file, test_file):
    # Read file
    header = ['user_id', 'item_id', 'rating', 'time_stamp']
    train_data = pd.read_csv('data-2/'+train_file, sep='\t', names=header)
    test_data = pd.read_csv('data-2/'+test_file, sep='\t', names=header)

    # Remove time stamp
    train_data.drop('time_stamp', axis=1, inplace=True)
    test_data.drop('time_stamp', axis=1, inplace=True)

    # Make pivot table
    rating = pd.pivot_table(train_data, 'rating', index='user_id', columns='item_id').fillna(0)

    train_data = np.array(train_data)
    test_data = np.array(test_data)

    return train_data, test_data, rating

def write_data(test_data, rating_result, train_file):
    user_id = test_data[:, 0]
    item_id = test_data[:, 1]

    with open('test/'+train_file+'_prediction.txt', 'w') as file:
        for u, i, r in zip(user_id, item_id, rating_result):
            file.write(str(u) + '\t' + str(i) + '\t' + str(r) + '\n')



if __name__ == "__main__":
    train_file = sys.argv[1]
    test_file = sys.argv[2]

    train_data, test_data, rating = read_data(train_file, test_file)

    LATENT = 3
    LEARNING_RATE = 0.001
    REGULAR_STRENGTH = 0.02
    NUM_EPOCHS = 100

    model = MatrixFactorization(rating=rating, k=LATENT, learning_rate=LEARNING_RATE, 
                    reg_param=REGULAR_STRENGTH, epochs=NUM_EPOCHS, test_data=test_data)
    model.train()
    rating_result = model.test()

    write_data(test_data, rating_result, train_file)