"""Finite-difference HJB solver for rho V = max_{0<c<=y a} F(c,a) + V'(a)(r a - c) (drift >= 0, upwind),
and policy/value factories for the joy of saving and for linear temptation. Used by ma_temptation.py."""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from ma_common import u_iso


def solve_hjb(F_policy, F_value, y, r, rho, agrid, Vtop, V0, Delta=1e3, tol=1e-10, maxit=5000):
    N = len(agrid)
    da = np.diff(agrid)
    V = V0.copy()
    for it in range(maxit):
        dV = np.empty(N)
        dV[:-1] = (V[1:] - V[:-1]) / da
        dV[-1] = dV[-2]
        c = F_policy(dV, agrid)
        c = np.minimum(c, y * agrid)
        mu = r * agrid - c
        Fv = F_value(c, agrid)
        # build A (forward differences); last row Dirichlet
        diag = np.zeros(N); up = np.zeros(N - 1)
        diag[:-1] = -mu[:-1] / da
        up[:] = mu[:-1] / da
        A = sp.diags([diag, up], [0, 1], format='csr')
        M = sp.identity(N, format='csr') * (rho + 1 / Delta) - A
        rhs = Fv + V / Delta
        M = M.tolil(); M[N - 1, :] = 0; M[N - 1, N - 1] = 1.0; M = M.tocsr()
        rhs[-1] = Vtop
        Vn = spla.spsolve(M, rhs)
        err = np.max(np.abs(Vn - V) / (1 + np.abs(V)))
        V = Vn
        if err < tol:
            break
    dV = np.empty(N); dV[:-1] = (V[1:] - V[:-1]) / da; dV[-1] = dV[-2]
    c = np.minimum(F_policy(dV, agrid), y * agrid)
    return V, c, it, err


def up_inv(m, g):
    return m ** (-1 / g)


