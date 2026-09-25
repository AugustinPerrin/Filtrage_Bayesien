import numpy as np


class ExtendedKalmanFilter:
    def __init__(self, x0, P0):
        self.x = np.asarray(x0, dtype=float)
        self.P = np.asarray(P0, dtype=float)

    def predict(self, f, F_jacobian, Q):
        """
        Modèle:
            x_k = f_k(x_k-1) + w_k

        F_k = df_k/dx évaluée en x_k-1
        """
        F = F_jacobian(self.x)
        self.x = f(self.x)
        self.P = F @ self.P @ F.T + Q

    def update(self, y, h, H_jacobian, R):
        """
        Modèle de mesure:
            y_k = h_k(x_k) + v_k

        H_k = dh_k/dx évaluée en x_k|k-1
        """

        H = H_jacobian(self.x)

        innovation = y - h(self.x)
        
        S = H @ self.P @ H.T + R
        
        K = np.linalg.solve(S, H @ self.P).T
        
        self.x = self.x + K @ innovation
        
        I = np.eye(self.P.shape[0])
        
        A = I - K @ H
        
        self.P = A @ self.P @ A.T + K @ R @ K.T
        
        self.P = 0.5 * (self.P + self.P.T)
     
        return innovation, S, K
