"""Information locale de Fisher, dynamique lineaire SANS bruit.

J[k] = F[k-1]^{-T} J[k-1] F[k-1]^{-1} + H[k].T R[k]^{-1} H[k].
Pour observation non lineaire, H[k] est evalue sur une trajectoire NOMINALE.
C'est une borne LOCALE, non la PCRB bayesienne moyenne sur les etats.
"""
import numpy as np


class DeterministicDynamicsPCRB:
    def __init__(self, P0):
        P0 = np.asarray(P0, dtype=float)
        if P0.ndim != 2 or P0.shape[0] != P0.shape[1]:
            raise ValueError('P0 doit etre carree')
        self.J = np.linalg.inv(P0)
        self.J = (self.J + self.J.T) / 2

    @property
    def bound(self):
        P = np.linalg.inv(self.J)
        return (P + P.T) / 2

    def predict(self, F):
        """Prediction de l'information, suppose F inversible."""
        F = np.asarray(F, dtype=float)
        if F.shape != self.J.shape:
            raise ValueError('Dimension de F incompatible avec J')

        F_inv = np.linalg.inv(F)
        self.J = F_inv.T @ self.J @ F_inv

        self.J = (self.J + self.J.T) / 2
        return self.bound

    def update(self, H, R):
        """Ajout de l'information locale H.T @ inv(R) @ H.

        H est la jacobienne de h evaluee a l'etat nominal, ou une matrice.
        """
        H = np.asarray(H, dtype=float)
        R = np.asarray(R, dtype=float)

        self.J = self.J + H.T @ np.linalg.inv(R) @ H

        self.J = (self.J + self.J.T) / 2
        return self.bound

    def step(self, F, H, R):
        self.predict(F)
        return self.update(H, R)
