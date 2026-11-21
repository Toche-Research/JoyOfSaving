"""Neoclassical technology with the isoelastic joy of saving.

    max  int e^{-rho t} U(c, s) dt,   U = (c^(1-g)-1)/(1-g) + theta (s^(1-g)-1)/(1-g),
    s = f(k) - c (gross saving),   dk/dt = f(k) - delta k - c,   f(k) = A k^alpha.

Optimality:  lambda = U_c - U_s,   dlambda/dt = (rho + delta) lambda - f'(k) U_c.
Steady state: f'(k*) = (rho + delta)(1 - tau*),  tau* = U_s/U_c = theta (c*/s*)^gamma,  s* = delta k*,  c* = f(k*) - delta k*.
"""
import numpy as np
from scipy.optimize import brentq
from scipy.integrate import solve_ivp


class Economy:
    """One calibration of the neoclassical economy with the joy of saving.

    The default values are those of the paper's calibration: rho = 3%, delta = 5%, gamma = 2,
    alpha = 1/3 and A = 1, with theta the weight on the joy of saving (theta = 0: Ramsey).
    All quantities are per unit of effective labour and all rates are per year.
    """

    def __init__(self, rho=0.03, delta=0.05, gamma=2.0, theta=0.0, alpha=1 / 3, A=1.0):
        """rho: rate of impatience; delta: depreciation; gamma: elasticity of marginal felicity;
        theta: weight on the joy of saving; alpha, A: capital share and level of f(k) = A k^alpha."""
        self.rho, self.delta, self.gamma, self.theta, self.alpha, self.A = rho, delta, gamma, theta, alpha, A

    # technology
    def f(self, k):
        """Output per head, f(k) = A k^alpha."""
        return self.A * k ** self.alpha

    def fp(self, k):
        """Marginal product of capital, f'(k)."""
        return self.A * self.alpha * k ** (self.alpha - 1)

    def fpp(self, k):
        """Second derivative of the production function, f''(k); negative."""
        return self.A * self.alpha * (self.alpha - 1) * k ** (self.alpha - 2)

    # preferences
    def Uc(self, c):
        """Marginal felicity of consumption, c^(-gamma)."""
        return c ** (-self.gamma)

    def Us(self, s):
        """Marginal joy of saving, theta s^(-gamma); zero throughout in the Ramsey case theta = 0,
        where the expression 0.0 * s keeps the return shaped like the argument."""
        return self.theta * s ** (-self.gamma) if self.theta > 0 else 0.0 * s

    def tau(self, c, s):
        """Marginal joy of saving relative to the marginal felicity of consumption, tau = U_s/U_c."""
        return self.Us(s) / self.Uc(c)

    def U(self, c, s):
        """Felicity (c^(1-g)-1)/(1-g) + theta (s^(1-g)-1)/(1-g), the logarithm at g = 1. The joy term
        is dropped outright at theta = 0, so that the Ramsey criterion is the limit and not a product
        with a possibly infinite factor."""
        g = self.gamma
        if g == 1.0:
            u = np.log(c)
            return u + self.theta * np.log(s) if self.theta > 0 else u
        u = (c ** (1 - g) - 1) / (1 - g)
        return u + self.theta * (s ** (1 - g) - 1) / (1 - g) if self.theta > 0 else u

    # reference capital stocks
    @property
    def k_mgr(self):
        """Modified golden rule: f'(k) = rho + delta."""
        return (self.A * self.alpha / (self.rho + self.delta)) ** (1 / (1 - self.alpha))

    @property
    def k_gr(self):
        """Golden rule: f'(k) = delta."""
        return (self.A * self.alpha / self.delta) ** (1 / (1 - self.alpha))

    @property
    def k_max(self):
        """Maximum sustainable capital: f(k) = delta k."""
        return (self.A / self.delta) ** (1 / (1 - self.alpha))

    def Psi(self, k):
        """Steady-state condition: Psi(k) = f'(k) - (rho+delta)(1 - tau(k)), tau(k) = theta ((f - delta k)/(delta k))^gamma."""
        c, s = self.f(k) - self.delta * k, self.delta * k
        return self.fp(k) - (self.rho + self.delta) * (1 - self.tau(c, s))

    def beyond_golden_rule(self):
        """True iff tau at the golden rule exceeds rho/(rho+delta): theta ((1-alpha)/alpha)^gamma > rho/(rho+delta)."""
        return self.theta * ((1 - self.alpha) / self.alpha) ** self.gamma > self.rho / (self.rho + self.delta)

    def steady_state(self):
        """Steady state as a dict: capital, consumption, gross saving, tau, the gross saving rate, f'(k*),
        the three reference capital stocks, whether the golden rule is passed, and the costate lambda*.
        The root of Psi is bracketed between the modified golden rule and the maximum sustainable stock,
        the interval on which Psi is decreasing; theta = 0 returns the modified golden rule exactly."""
        if self.theta == 0:
            k = self.k_mgr
        else:
            k = brentq(self.Psi, self.k_mgr * (1 + 1e-12), self.k_max * (1 - 1e-9), xtol=1e-14, rtol=1e-14)
        c, s = self.f(k) - self.delta * k, self.delta * k
        return dict(k=k, c=c, s=s, tau=self.tau(c, s), saving_rate=s / self.f(k), fp=self.fp(k),
                    k_mgr=self.k_mgr, k_gr=self.k_gr, k_max=self.k_max,
                    beyond_gr=self.beyond_golden_rule(), lam=self.Uc(c) - self.Us(s))

    # dynamics in (k, c)
    def cdot(self, k, c):
        """dc/dt from dlambda/dt = (rho+delta) lambda - f' U_c with lambda = c^-g - theta s^-g, s = f - c."""
        g, th, d = self.gamma, self.theta, self.delta
        s = self.f(k) - c
        kdot = s - d * k
        fp = self.fp(k)
        if th > 0:
            N = g * th * s ** (-g - 1) * fp * kdot - (self.rho + d) * (c ** (-g) - th * s ** (-g)) + fp * c ** (-g)
            D = g * (c ** (-g - 1) + th * s ** (-g - 1))
        else:
            N = -(self.rho + d) * c ** (-g) + fp * c ** (-g)
            D = g * c ** (-g - 1)
        return N / D

    def rhs(self, t, y):
        """The pair (dk/dt, dc/dt) at the state y = (k, c), in the form solve_ivp expects."""
        k, c = y
        return [self.f(k) - self.delta * k - c, self.cdot(k, c)]

    def jacobian(self, k=None, c=None, h=1e-6):
        """Finite-difference Jacobian of (kdot, cdot) in (k, c) at the steady state by default."""
        if k is None:
            ss = self.steady_state(); k, c = ss['k'], ss['c']
        J = np.zeros((2, 2))
        for j, (dk, dc) in enumerate([(h * k, 0.0), (0.0, h * c)]):
            fpl = np.array(self.rhs(0, [k + dk, c + dc])); fmi = np.array(self.rhs(0, [k - dk, c - dc]))
            J[:, j] = (fpl - fmi) / (2 * (dk + dc))
        return J

    def det_closed_form(self):
        """det J = [(rho+delta) tau (f'(k) - f(k)/k) + f''(k) c/gamma] / (1 + tau c/s) at the steady state."""
        ss = self.steady_state(); k, c, s, tau = ss['k'], ss['c'], ss['s'], ss['tau']
        return ((self.rho + self.delta) * tau * (self.fp(k) - self.f(k) / k) + self.fpp(k) * c / self.gamma) / (1 + tau * c / s)

    def eigen(self):
        """Stable and unstable eigenvalues rho/2 -/+ sqrt(rho^2/4 - det), with det in closed form."""
        det = self.det_closed_form()
        disc = np.sqrt(self.rho ** 2 / 4 - det)
        return self.rho / 2 - disc, self.rho / 2 + disc

    def half_life(self):
        """Half-life in years of the adjustment to the steady state, ln 2 / |stable eigenvalue|."""
        return np.log(2) / abs(self.eigen()[0])

    def stable_vector(self):
        """Eigenvector (1, v2) of the stable root, normalised to a unit step in capital."""
        J = self.jacobian()
        eps = self.eigen()[0]
        # (J - eps I) v = 0 with v = (1, v2): row 1: (f'-delta-eps) + (-1) v2 = 0
        v2 = J[0, 0] - eps
        return np.array([1.0, v2])

    def _branch(self, sign, k_target):
        """Backward integration from the steady state along the stable eigenvector; sign < 0 goes below k*."""
        ss = self.steady_state(); ks, cs = ss['k'], ss['c']
        v = self.stable_vector(); v = v / np.linalg.norm(v)
        y0 = np.array([ks, cs]) + sign * 1e-7 * ks * v
        stop = lambda t, y: y[0] - k_target
        stop.terminal = True
        sol = solve_ivp(lambda t, y: -np.array(self.rhs(t, y)), [0, 1e5], y0, events=stop,
                        rtol=1e-11, atol=1e-13, dense_output=True, max_step=2.0)
        return sol  # sol.t is backward time from the steady state

    def saddle_path(self, k_lo_frac=0.2, k_hi_frac=2.0):
        """Saddle path (k increasing, c) on [k_lo_frac k*, k_hi_frac k*]."""
        ss = self.steady_state(); ks, cs = ss['k'], ss['c']
        lo = self._branch(-1.0, k_lo_frac * ks); hi = self._branch(1.0, k_hi_frac * ks)
        k = np.concatenate([lo.y[0][::-1], [ks], hi.y[0]])
        c = np.concatenate([lo.y[1][::-1], [cs], hi.y[1]])
        return k, c

    def policy(self, k0):
        """Consumption on the saddle path at k0."""
        ss = self.steady_state(); ks = ss['k']
        sol = self._branch(-1.0 if k0 < ks else 1.0, k0)
        return sol.y[1][-1]

    def transition(self, k0, T=200.0, n=2001):
        """Time path from k0 on the saddle path, read off the backward integration (no forward integration
        along the unstable direction): dict with t, k, c, s, saving rate, growth of consumption."""
        ss = self.steady_state(); ks = ss['k']
        sol = self._branch(-1.0 if k0 < ks else 1.0, k0)
        t0 = sol.t[-1]                      # backward time at which the branch reaches k0
        t = np.linspace(0, T, n)
        tb = np.clip(t0 - t, 0.0, t0)       # backward time of each forward date
        y = sol.sol(tb)
        k, c = y[0], y[1]
        far = t > t0                        # beyond the start of the branch: steady state to numerical accuracy
        k[far], c[far] = ks, ss['c']
        s = self.f(k) - c
        return dict(t=t, k=k, c=c, s=s, saving_rate=s / self.f(k), kdot=s - self.delta * k,
                    gc=np.array([self.cdot(ki, ci) for ki, ci in zip(k, c)]) / c)

    # welfare
    def path_value(self, tr, c_ss, s_ss, scale_c=1.0):
        """Value under THIS economy's criterion of a path given as the dict transition() returns, with
        consumption multiplied by scale_c: Simpson's rule for int_0^T e^{-rho t} U(c, s) dt plus the
        exact remainder e^{-rho T} U(c*, s*)/rho, which is the whole of the tail once the path has
        reached the steady state (c_ss, s_ss), as it has well before T at the horizons used here."""
        t = tr['t']
        T = t[-1]
        dx = T / (len(t) - 1)
        y = np.exp(-self.rho * t) * self.U(scale_c * tr['c'], tr['s'])
        integral = dx / 3.0 * (y[0] + y[-1] + 4.0 * y[1:-1:2].sum() + 2.0 * y[2:-1:2].sum())
        return integral + np.exp(-self.rho * T) * self.U(scale_c * c_ss, s_ss) / self.rho

    def consumption_equivalent(self, k0, T=200.0, n=20001):
        """The constant proportional increase in the Ramsey planner's consumption path from k0 that would
        give her, under THIS economy's criterion, the value this economy's own optimal path attains. The
        Ramsey planner has the same rho, delta, gamma, alpha and A and theta = 0, so she chooses the
        Ramsey saddle path; her realised (c, s) is then valued with the joy of saving counted at this
        economy's theta. Returns 0.0 at theta = 0, where the two paths coincide.

        At gamma > 1 felicity in consumption is bounded above, so the scaled Ramsey value has a finite
        supremum as the increase grows without bound; where that supremum falls short of the
        joy-of-saving value no such increase exists and the return is nan. The bracket is doubled rather
        than fixed so that the two cases are distinguished rather than confused."""
        if self.theta == 0:
            return 0.0
        ramsey = Economy(rho=self.rho, delta=self.delta, gamma=self.gamma, theta=0.0,
                         alpha=self.alpha, A=self.A)
        ss_r, ss_j = ramsey.steady_state(), self.steady_state()
        tr_r, tr_j = ramsey.transition(k0, T=T, n=n), self.transition(k0, T=T, n=n)
        target = self.path_value(tr_j, ss_j['c'], ss_j['s'])
        gap = lambda mu: self.path_value(tr_r, ss_r['c'], ss_r['s'], scale_c=1.0 + mu) - target
        if gap(0.0) >= 0.0:
            return 0.0
        hi = 1.0
        while gap(hi) < 0.0:
            hi *= 2.0
            if hi > 1e12:
                return np.nan
        return brentq(gap, 0.0, hi, xtol=1e-14, rtol=1e-15)

    def cdot_zero_locus(self, kgrid, n_scan=2000):
        """The dc/dt = 0 locus with the joy of saving, as the lower and upper branches (c_low, c_high) for each k
        in kgrid, NaN where there is no root. The locus is a closed curve: for k < k_MGR the numerator N(c, k) is
        positive as c -> 0 and negative as c -> f(k), so there is one root (returned as both branches);
        for k > k_MGR, N is negative at both ends and has either no root or two. The roots are located by a
        sign scan of N on a grid in c and refined by bisection. (Figure 2 is drawn from these two
        branches, joined into one closed curve; the lower branch also fixes where the locus label sits.)"""
        lo_b, hi_b = [], []
        for k in kgrid:
            fk = self.f(k)
            if self.theta <= 0:
                lo_b.append(np.nan); hi_b.append(np.nan)   # vertical line k = k_mgr in the Ramsey case
                continue
            N = lambda c: (self.gamma * self.theta * (fk - c) ** (-self.gamma - 1) * self.fp(k) * (fk - c - self.delta * k)
                           - (self.rho + self.delta) * (c ** (-self.gamma) - self.theta * (fk - c) ** (-self.gamma))
                           + self.fp(k) * c ** (-self.gamma))
            cs = fk * np.linspace(1e-6, 1 - 1e-6, n_scan)
            with np.errstate(over='ignore', invalid='ignore'):
                vals = np.array([N(c) for c in cs])
            roots = []
            for i in np.where(np.sign(vals[:-1]) * np.sign(vals[1:]) < 0)[0]:
                roots.append(brentq(N, cs[i], cs[i + 1], xtol=1e-13))
            if not roots:
                lo_b.append(np.nan); hi_b.append(np.nan)
            else:
                lo_b.append(min(roots)); hi_b.append(max(roots))
        return np.array(lo_b), np.array(hi_b)
