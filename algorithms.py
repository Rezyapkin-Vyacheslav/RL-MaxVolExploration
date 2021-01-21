import numpy as np
from copy import deepcopy
from numpy.linalg import pinv, norm, solve, lstsq
from sklearn.metrics import r2_score, mean_squared_error as mse
# from exception import NotImplementedError, ValueError

class Agent:
    def __init__(self, method='random', linreg_model=None, rnd_model=None):
        self.H = None #states
        self.R = [] # rewards for self.H - used for fit
        self.w = None # Q-function weights
        self.method = method
        self.linreg_model = linreg_model

        self.rnd_model = rnd_model


        
#       todo: add fit_flag
        
    def choose_phi(self, Phi, R, method=None, update_H=True, verbose=False, **evaluate_args):
        if method is None:
            method = self.method
            
        # some methods need non-empty H
        if self.H is None:
            method = 'random'
            
        if method == 'random':
            ind = np.random.randint(Phi.shape[0], size=1)
        elif method == 'stat_volume':
            H_pinv = pinv(self.H)
            volumes = [norm(H_pinv.T@phi) for phi in Phi]
            if verbose:
                print(volumes)
            ind = np.argmax(volumes)
        elif method == 'stacked_volume':
            volumes = [np.linalg.det(np.vstack([self.H, phi]).T @ np.vstack([self.H, phi])) for phi in Phi]
            ind = np.argmax(volumes)
        elif method == 'greedy':
            # too heavy, but may be effiсient
            best_r2 = -np.inf
            for i, phi in enumerate(Phi):
                fantom = deepcopy(self)
                fantom.update_H(phi)
                fantom.R += [R[i]]
                fantom.fit()
                r2 = fantom.evaluate(fantom.H, fantom.R)
                if r2 > best_r2:
                    ind = i
                    best_r2 = r2
        elif method == 'max_error':
            # argmax error on new sample (like the most difficult)
            # contrexample: stochastic case, with big variance 
            # unfair: we don't know R
            mse_scores = [mse([r], self.predict([phi])) for phi, r in zip(Phi, R)]
            ind = np.argmax(mse_scores)
        elif method == 'min_variance':
            # contrexample for r2 score - we can try to maximize variance, it will increase r2
            variances = [np.var(self.R+[r]) for r in R]
            ind = np.argmax(variances)
        elif method == 'random_network_distill':
            ind = self.rnd_model.choose_phi(Phi)
            self.rnd_model.fit(Phi[ind])
        else:
            raise ValueError(f'unknown method: {method}')


        phi = Phi[ind]
        r = R[ind]
            
        if update_H:
            self.update_H(phi)
            self.R += [r]
        return phi, r
    
    def update_H(self, phi):
        if self.H is None:
            self.H = phi
        else:
            self.H = np.vstack([self.H, phi])
    def fit(self):
        if self.linreg_model is not None:
            self.linreg_model.fit(self.H, self.R)
        else:
            self.w = pinv(self.H) @ self.R
    def predict(self, S):
        if self.linreg_model is not None:
            return self.linreg_model.predict(S)
        else:
            return S @ self.w
    def evaluate(self, S, R, metric=r2_score):
        R_pred = self.predict(S)
        r2 = metric(R, R_pred)
        # mse = mean_squared_error(R, R_pred)
#         print('r2_score', r2)
#         print('mse', mse)
        return r2 # , mse
        