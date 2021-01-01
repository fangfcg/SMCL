from utils import *

class SMC():

    def __init__(self, X, k, alpha, beta, lmd, iterations, p=5):

        self.X = X
        self.num_samples, self.num_features = X.shape
        self.X_ = self.replace_nan(np.zeros(self.X.shape))
        self.p = p
        self.k = k
        self.alpha = alpha
        self.beta = beta
        self.iterations = iterations
        self.lmd = lmd
        self.D = np.zeros((self.num_samples, self.num_samples))
        self.W = np.zeros((self.num_samples, self.num_samples))
        for i in range(0, self.num_samples):
            distance_list = []
            for j in range(0, self.num_samples):
                distance_list.append((self.geo_distance(i, j), j))
            distance_list.sort(key=lambda x: x[0])
            for t in distance_list[:self.p]:
                self.D[i][t[1]] = 1
                self.D[t[1]][i] = 1
        for i in range(self.num_samples):
            self.W[i][i] = np.sum(self.D[i, :])
        self.L = self.W - self.D
        self.not_nan_index = (np.isnan(self.X) == False)
        self.r_omega = self.not_nan_index

    def geo_distance(self, i, j):
        dis_x = (self.X[i][0] - self.X[j][0]) ** 2
        dis_y = (self.X[i][1] - self.X[j][1]) ** 2
        return np.sqrt(dis_x + dis_y)

    def error(self):
        X_hat = np.matmul(self.U, self.V)
        res = np.nansum((self.X - X_hat) ** 2) + self.lmd * np.trace(np.matmul(np.matmul(self.U.T, self.L), self.U))
        return res

    def train(self, method=1):
        # Initialize factorization matrix U and V
        self.U = np.random.rand(self.num_samples, self.k)
        self.V = np.random.rand(self.k, self.num_features)
        # Perform stochastic gradient descent for number of iterations
        training_process = []
        for i in range(self.iterations):
            # total square error
            e = self.error()
            training_process.append((i, e))
            if method == 1:
                self.update()
            else:
                self.update2()
        return training_process

    def update(self):
        'sgd'
        a1 = np.matmul(self.X_, self.V.T)
        a2 = np.matmul(self.r_omega * np.matmul(self.U, self.V), self.V.T)
        a3 = self.lmd * np.matmul(self.L, self.U)
        U_new = self.U - self.alpha * (-2 * a1 +
                                    2 * a2 + 2 * a3)
        V_new = self.V - self.beta * (-2 * np.matmul(self.U.T, self.X_) +
                                   2 * np.matmul(self.U.T, self.r_omega * np.matmul(self.U, self.V)))
        self.U = U_new
        self.V = V_new


    def update2(self):
        'iteration'
        u1 = np.matmul(self.X_, self.V.T) + self.lmd * np.matmul(self.D, self.U)
        u2 = np.matmul(self.r_omega * np.matmul(self.U, self.V), self.V.T) + self.lmd * np.matmul(self.W, self.U)
        U_new = self.U * u1 / u2
        v1 = np.matmul(self.U.T, self.X_)
        v2 = np.matmul(self.U.T, self.r_omega * np.matmul(self.U, self.V))
        V_new = self.V * v1 / v2
        self.U = U_new
        self.V = V_new



    def get_x(self, i, j):
        """
        Get the predicted x of sample i and feature j
        """
        prediction = self.b + self.b_u[i] + self.b_v[j] + self.U[i, :].dot(self.V[j, :].T)
        return prediction

    def full_matrix(self):
        """
        Computer the full matrix using the resultant biases, U and V
        """
        return np.matmul(self.U, self.V) * (1-self.r_omega) + self.X_

    def replace_nan(self, X_hat):
        """
        Replace np.nan of X with the corresponding value of X_hat
        """
        X = np.copy(self.X)
        for i in range(self.num_samples):
            for j in range(self.num_features):
                if np.isnan(X[i, j]):
                    X[i, j] = X_hat[i, j]
        return X
