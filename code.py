import numpy as np
from scipy.stats import norm

## Question 1

class IsingMessageGraph:
    def __init__(self, y, alpha, beta, mu):
        self.y = y
        self.dims = self.y.shape
        self.alpha = alpha
        self.beta = beta
        self.expalpha = np.exp(alpha)
        self.expbeta = np.exp(beta)
        self.update_matrix = np.array([[self.expbeta, 1], [1, self.expbeta]])
        self.mu = mu
        self.densities = [norm(mean).pdf for mean in self.mu]
        self.deltas = [np.array(delta) for delta in [(-1,0), (1,0), (0,-1), (0,1)]]
        self.reverse_direction = [1, 0, 3, 2]
        self.messages = None
    
    def legal_neighbour(self, i, j):
        H, W = self.dims
        return 0 <= i and i < H and 0 <= j and j < W
    
    def message_y(self, i, j):
        message = np.array([pdf(self.y[i, j]) for pdf in self.densities])
        message[1] *= self.expalpha
        return message / message.sum()
    
    def init_messages(self):
        H, W = self.dims
        self.messages = np.ones(self.dims + (5,2))
        for i in range(H):
            for j in range(W):
                self.messages[i, j, 4] = self.message_y(i, j)
    
    def message_x(self, i, j, direction):
        neighbour_messages = self.messages[i, j, 4].copy()
        for other_direction in range(4):
            if other_direction != direction:
                neighbour_messages *= self.messages[i, j, other_direction].copy()
        message = self.update_matrix.dot(neighbour_messages)
        return message / message.sum()
    
    def message_passing(self, num_iter=10, verbose=False):
        """Execute loopy belief propagation."""
        self.init_messages()
        for t in range(num_iter):
            if verbose:
                print("Iteration %d" % (t+1))
            self.message_passing_step()

    def message_passing_step(self):
        """Pass messages between neighbours in every directions."""
        H, W = self.dims
        for i_from in range(H):
            for j_from in range(W):
                for direction_from, delta in enumerate(self.deltas):
                    i_to, j_to = (i_from, j_from) + delta
                    if self.legal_neighbour(i_to, j_to):
                        direction_to = self.reverse_direction[direction_from]
                        self.messages[i_to, j_to, direction_to] = self.message_x(i_from, j_from, direction_from)
    
    def marginal(self, i, j):
        """Compute the conditional marginal distribution at given location."""
        if self.messages is None:
            self.message_passing()
        neighbour_messages = self.messages[i, j, 4].copy()
        for direction in range(4):
            neighbour_messages *= self.messages[i, j, direction].copy()
        return neighbour_messages / neighbour_messages.sum()
    
    def locally_most_probable(self):
        """Return the image where each pixel is the most probable wrt its
        conditional marginal distribution."""
        H, W = self.dims
        result = np.zeros((H, W))
        for i in range(H):
            for j in range(W):
                result[i, j] = np.argmax(self.marginal(i, j))
        return result


## Question 2

class IsingGibbsSampler:
    def __init__(self, y, alpha, beta, mu):
        self.y = y
        self.dims = self.y.shape
        self.alpha = alpha
        self.beta = beta
        self.mu = mu
        self.diff_mu = mu[0] - mu[1]
        self.base_exponent = - self.alpha + (mu[0] ** 2 + mu[1] ** 2) / 2
        self.uniform = np.random.rand
        self.deltas = [np.array(delta) for delta in [(-1,0), (1,0), (0,-1), (0,1)]]
    
    @staticmethod
    def sigmoid(u):
        return 1 / (1 + np.exp(u))
    
    def legal_neighbour(self, i, j):
        H, W = self.dims
        return 0 <= i and i < H and 0 <= j and j < W
    
    def proba_pixel(self, i, j, x):
        exponent = 0
        for direction, delta in enumerate(self.deltas):
            i_neigh, j_neigh = (i, j) + delta
            if self.legal_neighbour(i_neigh, j_neigh):
                if x[i_neigh, j_neigh]:
                    exponent -= 1
                else:
                    exponent += 1
        exponent = self.beta * exponent + self.base_exponent + self.y[i, j] * self.diff_mu
        return self.sigmoid(exponent)
    
    def sample_pixel(self, i, j, x, *args):
        p = self.proba_pixel(i, j, x, *args)
        return int(self.uniform() <= p)
    
    def init_sample(self):
        """Initialise sample by MLE wrt y"""
        mu0, mu1 = self.mu
        return ((self.y - mu1) ** 2 <= (self.y - mu0) ** 2).astype(int)
    
    def gibbs_sampling(self, T=10, trajectory=True):
        x = self.init_sample()
        if trajectory:
            past_x = [x]
        for t in range(T):
            x = self.new_sample(x)
            if trajectory:
                past_x.append(x)
        if trajectory:
            return past_x
        return x
        
    def new_sample(self, x):
        H, W = self.dims
        x_new = x.copy()
        for i in range(H):
            for j in range(W):
                x_new[i, j] = self.sample_pixel(i, j, x_new)
        return x_new


## Question 3

class IsingEM:
    def __init__(self, y, alpha, beta):
        self.y = y
        self.dims = self.y.shape
        self.size = self.y.size
        self.sum_y = y.sum()
        self.alpha = alpha
        self.beta = beta
    
    def estimate_posterior(self, mu, burn_samples=5, num_samples=10):
        MCMC = IsingGibbsSampler(self.y, self.alpha, self.beta, mu)
        posterior_estimate = np.zeros(self.dims)
        x = MCMC.init_sample()
        for t in range(burn_samples):
            x = MCMC.new_sample(x)
        for t in range(num_samples):
            x = MCMC.new_sample(x)
            posterior_estimate += x
        return posterior_estimate / num_samples
    
    def EM_step(self, mu, **kwargs):
        posterior_estimate = self.estimate_posterior(mu, **kwargs)
        numerator = (self.y * posterior_estimate).sum()
        denominator = posterior_estimate.sum()
        mu_new = [
            (self.sum_y - numerator) / (self.size - denominator),
            numerator / denominator
        ]
        return mu_new
    
    def init_mu(self):
        averager = np.median
        avg_y = averager(self.y)
        return [averager(self.y[self.y < avg_y]), averager(self.y[self.y >= avg_y])]
    
    def estimation(self, num_iter=10, verbose=False, **kwargs):
        mu = self.init_mu()
        if verbose:
            print(f"Initialisation: mu={mu}")
        for t in range(num_iter):
            mu = self.EM_step(mu, **kwargs)
            if verbose:
                print(f"Iteration {t+1}: mu={mu}")
        return mu

## Question 5

class IsingGibbsSamplerParams(IsingGibbsSampler):
    def __init__(self, y, a, b, m, s):
        self.y = y
        self.dims = self.y.shape
        self.size = self.y.size
        self.sum_y = self.y.sum()
        self.a = a
        self.b = b
        self.m = m
        self.s = s
        self.sinv2 = 1 / s ** 2
        self.m_sinv2 = m * self.sinv2
        self.uniform = np.random.rand
        self.deltas = [np.array(delta) for delta in [(-1,0), (1,0), (0,-1), (0,1)]]
    
    def sample_boxed_exponential(self, lbda, C):
        """Sample from distribution proportional to exp(lbda * x) * 1_[0,C](x)"""
        U = self.uniform()
        return C + np.log(U + (1 - U) * np.exp(- lbda * C)) / lbda

    def sample_alpha(self, x):
        return self.sample_boxed_exponential(x.sum(), self.a)
        
    def sample_beta(self, x):
        agreeing_neighbours = 2 * (np.count_nonzero(x[1:] == x[:-1]) + np.count_nonzero(x[:, 1:] == x[:, :-1]))
        return self.sample_boxed_exponential(agreeing_neighbours, self.b)
    
    def sample_mu_k(self, sum_char, sum_char_y):
        sigma2 = 1 / (sum_char + self.sinv2)
        gamma = sigma2 * (sum_char_y + self.m_sinv2)
        return norm(gamma, np.sqrt(sigma2)).rvs()
    
    def sample_mu(self, x):
        mu = []
        mask = x==0
        sum_char = x[mask].sum()
        sum_char_y = (x[mask] * self.y[mask]).sum()
        mu.append(self.sample_mu_k(sum_char, sum_char_y))
        # Reuse computations for mu1
        sum_char = self.size - sum_char
        sum_char_y = self.sum_y - sum_char_y
        mu.append(self.sample_mu_k(sum_char, sum_char_y))
        return mu
    
    def proba_pixel(self, i, j, x, alpha, beta, mu):
        exponent = 0
        for direction, delta in enumerate(self.deltas):
            i_neigh, j_neigh = (i, j) + delta
            if self.legal_neighbour(i_neigh, j_neigh):
                if x[i_neigh, j_neigh]:
                    exponent -= 1
                else:
                    exponent += 1
        exponent = - alpha + beta * exponent + self.y[i, j] * (mu[0] - mu[1]) + (mu[0] ** 2 + mu[1] ** 2) / 2
        return self.sigmoid(exponent)
    
    def init_sample(self):
        averager = np.median
        avg_y = averager(self.y)
        x = np.zeros(self.dims, dtype=int)
        x[self.y >= avg_y] = 1
        return x, 1, 1, [0, 0]
        
    def new_sample(self, past_sample):
        x, alpha, beta, mu = past_sample
        H, W = self.dims
        x_new = x.copy()
        for i in range(H):
            for j in range(W):
                x_new[i, j] = self.sample_pixel(i, j, x, alpha, beta, mu)
        alpha_new = self.sample_alpha(x)
        beta_new = self.sample_beta(x)
        mu_new = self.sample_mu(x)
        return x_new, alpha_new, beta_new, mu_new