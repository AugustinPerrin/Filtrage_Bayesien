import numpy as np


class KalmanFilter:
    def __init__(self, x0, P0):
        self.x = np.asarray(x0, dtype=float)
        self.P = np.asarray(P0, dtype=float)

    def predict(self, F, Q):
        """
        Prediction:
            x_k|k-1 = F x_k-1
            P_k|k-1 = F P_k-1 F.T + Q
        """
        F = np.asarray(F)
        Q = np.sarray(Q)

        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q

        
    def update(self, y, H, R):
        """
        Correction:
            innovation = y - H x
            S = H P H.T + R
            K = P H.T S^-1
        """
        y = np.asarray(y)
        H = np.asarray(H)
        R = np.asarray(R)

        innovation = y - H @ self.x

        S = H @ self.P @ H.T + R

        K = np.linalg.solve(S, H @ self.P).T

        self.x = self.x + K @ innovation

        I = np.eye(self.P.shape[0])

        A = I - K @ H

        self.P = A @ self.P @ A.T + K @ R @ K.T

        self.P = 0.5 * (self.P + self.P.T)
         
        return innovation, S, K
