from scipy.spatial.distance import squareform, pdist
from numpy.linalg import linalg, LinAlgError
from scipy.stats import chi2

import numpy as np


def simulate( nPeriod, nPath,beta):
    noise =  np.random.randn(nPeriod, nPath)
    sims = np.zeros((nPeriod, nPath))
    sims[0] = noise[0]
    sqrt_beta = np.sqrt(1 - beta ** 2)
    for period in range(1, nPeriod):
        sims[period] = beta*sims[period-1] + sqrt_beta *noise[period]
    return sims



def simulatepm(N,p_change):
    X = np.zeros(N)-1
    change_sign = np.random.rand(N) < p_change
    for i in range(N):
        if change_sign[i]:
            X[i] = -X[i-1]
        else:
            X[i] = X[i-1]
    return X


class GaussianQuadraticTest:
    def __init__(self, grad_log_prob, scaling=2.0, grad_log_prob_multiple=None):
        self.scaling = scaling
        self.grad = grad_log_prob
        
        # construct (slow) multiple gradient handle if efficient one is not given
        if grad_log_prob_multiple is None:
            def grad_multiple(X):
                # simply loop over grad calls. Slow
                return np.array([self.grad(x) for x in X])
            
            self.grad_multiple = grad_multiple
        else:
            self.grad_multiple = grad_log_prob_multiple
            
    def k(self, x, y):
        return np.exp(-np.dot(x - y,x - y) / self.scaling)
    
    def k_multiple(self, X):
        """
        Efficient computation of kernel matrix without loops
        
        Effectively does the same as calling self.k on all pairs of the input
        """
        assert(X.ndim == 1)
        
        sq_dists = squareform(pdist(X.reshape(len(X), 1), 'sqeuclidean'))
            
        K = np.exp(-(sq_dists) / self.scaling)
        return K

    def k_multiple_dim(self, X):

        # check for stupid mistake
        assert X.shape[0] > X.shape[1]

        sq_dists = squareform(pdist(X, 'sqeuclidean'))

        K = np.exp(-(sq_dists) / self.scaling)
        return K


    def g1k(self, x, y):
        return -2.0 / self.scaling * self.k(x, y) * (x - y)
    
    def g1k_multiple(self, X):
        """
        Efficient gradient computation of Gaussian kernel with multiple inputs
        
        Effectively does the same as calling self.g1k on all pairs of the input
        """
        assert X.ndim == 1
        
        differences = X.reshape(len(X), 1) - X.reshape(1, len(X))
        sq_differences = differences ** 2
        K = np.exp(-sq_differences / self.scaling)

        return -2.0 / self.scaling * K * differences


    def g1k_multiple_dim(self, X,K,dim):

        X_dim = X[:,dim]
        assert X_dim.ndim == 1

        differences = X_dim.reshape(len(X_dim), 1) - X_dim.reshape(1,len(X_dim))

        return -2.0 / self.scaling * K * differences




    def g2k(self, x, y):
        return -self.g1k(x, y)
    
    def g2k_multiple(self, X):
        """
        Efficient 2nd gradient computation of Gaussian kernel with multiple inputs
        
        Effectively does the same as calling self.g2k on all pairs of the input
        """
        return -self.g1k_multiple(X)

    def g2k_multiple_dim(self, X,K,dim):
        return -self.g1k_multiple_dim(X,K,dim)

    def gk(self, x, y):
        return 2.0 * self.k(x, y) * (self.scaling - 2 * (x - y) ** 2) / self.scaling ** 2

    def gk_multiple(self, X):
        """
        Efficient gradient computation of Gaussian kernel with multiple inputs
        
        Effectively does the same as calling self.gk on all pairs of the input
        """
        assert X.ndim == 1
        
        differences = X.reshape(len(X), 1) - X.reshape(1, len(X))
        sq_differences = differences ** 2
        K = np.exp(-sq_differences / self.scaling)

        return 2.0 * K * (self.scaling - 2 * sq_differences) / self.scaling ** 2

    def gk_multiple_dim(self, X,K,dim):
        X_dim = X[:,dim]
        assert X_dim.ndim == 1

        differences = X_dim.reshape(len(X_dim), 1) - X_dim.reshape(1,len(X_dim))

        sq_differences = differences ** 2

        return 2.0 * K * (self.scaling - 2 * sq_differences) / self.scaling ** 2


    def get_statisitc(self, N, samples):
        U_matrix = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                x1 = samples[i]
                x2 = samples[j]
                a = self.grad(x1) * self.grad(x2) * self.k(x1, x2)
                b = self.grad(x2) * self.g1k(x1, x2)
                c = self.grad(x1) * self.g2k(x1, x2)
                d = self.gk(x1, x2)
                U_matrix[i, j] = a + b + c + d
        stat = N * np.mean(U_matrix)
        return U_matrix, stat


    def get_statisitc_two_dim(self, N, samples,dim):
        U_matrix = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                x1 = samples[i]
                x2 = samples[j]
                a = self.grad(x1)[dim] * self.grad(x2)[dim] * self.k(x1, x2)
                b = self.grad(x2)[dim] * self.g1k(x1, x2)[dim]
                c = self.grad(x1)[dim] * self.g2k(x1, x2)[dim]
                d = self.gk(x1, x2)[dim]
                U_matrix[i, j] = a + b + c + d
        stat = N * np.mean(U_matrix)
        return U_matrix, stat



    def get_statistic_multiple_dim(self, samples,dim):

        log_pdf_gradients = self.grad_multiple(samples)
        log_pdf_gradients = log_pdf_gradients[:,dim]
        K = self.k_multiple_dim(samples)
        G1K = self.g1k_multiple_dim(samples,K,dim)
        G2K = self.g2k_multiple_dim(samples,K,dim)
        GK = self.gk_multiple_dim(samples,K,dim)

        # use broadcasting to mimic the element wise looped call
        pairwise_log_gradients = log_pdf_gradients.reshape(len(log_pdf_gradients), 1) * log_pdf_gradients.reshape(1, len(log_pdf_gradients))
        A = pairwise_log_gradients * K
        B = G1K * log_pdf_gradients
        C = (G2K.T * log_pdf_gradients).T
        D = GK
        U = A + B + C + D
        stat = len(samples) * np.mean(U)
        return U, stat


    def get_statistic_multiple(self, samples):
        """
        Efficient statistic computation with multiple inputs
        
        Effectively does the same as calling self.get_statisitc.
        """
        log_pdf_gradients = self.grad_multiple(samples)
        K = self.k_multiple(samples)
        G1K = self.g1k_multiple(samples)
        G2K = self.g2k_multiple(samples)
        GK = self.gk_multiple(samples)
        
        # use broadcasting to mimic the element wise looped call
        pairwise_log_gradients = log_pdf_gradients.reshape(len(log_pdf_gradients), 1) * log_pdf_gradients.reshape(1, len(log_pdf_gradients))
        A = pairwise_log_gradients * K
        B = G1K * log_pdf_gradients
        C = (G2K.T * log_pdf_gradients).T
        D = GK
        U = A + B + C + D
        stat = len(samples) * np.mean(U) 
        return U, stat

    def get_statistic_multiple_custom_gradient(self, samples, log_pdf_gradients):
        """
        Implements the statistic for multiple samples, each from a different
        density whose gradient at the sample is passed
        
        """
        K = self.k_multiple(samples)
        G1K = self.g1k_multiple(samples)
        G2K = self.g2k_multiple(samples)
        GK = self.gk_multiple(samples)
        
        # use broadcasting to mimic the element wise looped call
        pairwise_log_gradients = log_pdf_gradients.reshape(len(log_pdf_gradients), 1) * log_pdf_gradients.reshape(1, len(log_pdf_gradients))
        A = pairwise_log_gradients * K
        B = G1K * log_pdf_gradients
        C = (G2K.T * log_pdf_gradients).T
        D = GK
        U = A + B + C + D
        stat = len(samples) * np.mean(U) 
        return U, stat

    def compute_pvalue(self, U_matrix, num_bootstrapped_stats=100):
        N = U_matrix.shape[0]
        bootsraped_stats = np.zeros(num_bootstrapped_stats)

        for proc in range(num_bootstrapped_stats):
            W = np.sign(np.random.randn(N))
            WW = np.outer(W, W)
            st = np.mean(U_matrix * WW)
            bootsraped_stats[proc] = N * st

        stat = N*np.mean(U_matrix)

        return float(np.sum(bootsraped_stats > stat)) / num_bootstrapped_stats

    def compute_pvalues_for_processes(self,U_matrix,chane_prob, num_bootstrapped_stats=100):
        N = U_matrix.shape[0]
        bootsraped_stats = np.zeros(num_bootstrapped_stats)

        # orsetinW = simulate(N,num_bootstrapped_stats,corr)

        for proc in range(num_bootstrapped_stats):
            # W = np.sign(orsetinW[:,proc])
            W = simulatepm(N,chane_prob)
            WW = np.outer(W, W)
            st = np.mean(U_matrix * WW)
            bootsraped_stats[proc] = N * st

        stat = N*np.mean(U_matrix)

        return float(np.sum(bootsraped_stats > stat)) / num_bootstrapped_stats


def mahalanobis_distance(difference, num_random_features):
    num_samples, _ = np.shape(difference)
    sigma = np.cov(np.transpose(difference))

    mu = np.mean(difference, 0)

    if num_random_features == 1:
        stat = float(num_samples * mu ** 2) / float(sigma)
    else:
        try:
            linalg.inv(sigma)
        except LinAlgError:
            print('covariance matrix is singular. Pvalue returned is 1.1')
            warnings.warn('covariance matrix is singular. Pvalue returned is 1.1')
            return 0
        stat = num_samples * mu.dot(linalg.solve(sigma, np.transpose(mu)))

    return chi2.sf(stat, num_random_features)

class GaussianSteinTest:
    def __init__(self, grad_log_prob, num_random_freq, scaling=(1.0, 10.0)):
        self.number_of_random_frequencies = num_random_freq
        self.scaling = scaling

        def stein_stat(random_frequency, samples):
            random_scale = np.random.uniform(self.scaling[0], self.scaling[1])
            a = grad_log_prob(samples)
            b = self._gaussian_test_function(samples, random_frequency, random_scale)
            c = self._test_function_grad(samples, random_frequency, random_scale)
            return a * b + c

        self.stein_stat = stein_stat


    def _make_two_dimensional(self, z):
        if len(z.shape) == 1:
            z = z[:, np.newaxis]
        return z

    def _get_mean_embedding(self, x, random_frequency, scaling=2.0):
        z = x - random_frequency
        z = linalg.norm(z, axis=1) ** 2
        z = np.exp(-z / scaling)
        return z

    def _gaussian_test_function(self, x, random_frequency, scaling=2.0):
        x = self._make_two_dimensional(x)
        mean_embedding = self._get_mean_embedding(x, random_frequency, scaling)
        return np.tile(mean_embedding, (self.shape, 1)).T


    def _test_function_grad(self, x, omega, scaling=2.0):
        arg = (x - omega) * 2 / scaling
        test_function_val = self._gaussian_test_function(x, omega, scaling)
        return -arg * test_function_val


    def compute_pvalue(self, samples):

        samples = self._make_two_dimensional(samples)

        self.shape = samples.shape[1]

        stein_statistics = []


        for f in range(self.number_of_random_frequencies):
            # This is a little bit of a bug , but th holds even for this choice
            random_frequency = np.random.randn()
            matrix_of_stats = self.stein_stat(random_frequency=random_frequency, samples=samples)
            stein_statistics.append(matrix_of_stats)

        normal_under_null = np.hstack(stein_statistics)
        normal_under_null = self._make_two_dimensional(normal_under_null)

        return mahalanobis_distance(normal_under_null, normal_under_null.shape[1])

