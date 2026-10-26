import numpy as np
from scipy.integrate import solve_ivp

rng = np.random.default_rng(7)
G, CD_A_RHO0, H_SCALE = 9.81, 0.0021, 8500.0  
THRUST_ACC, BURN = 25.0, 20.0                

def rocket(t, y):
    h, v = y
    a = (THRUST_ACC if t < BURN else 0) - G - CD_A_RHO0 * np.exp(-h / H_SCALE) * v * abs(v)
    return [v, a]

land = lambda t, y: y[0] if t > 1 else 1.0 
land.terminal = True
sol = solve_ivp(rocket, [0, 120], [0, 0], dense_output=True, events=land, rtol=1e-9, atol=1e-9)
T_LAND = sol.t[-1]

t = np.sort(rng.uniform(0.5, T_LAND - 0.5, 20))
h, v = sol.sol(t)
a = np.array([rocket(ti, [hi, vi])[1] for ti, hi, vi in zip(t, h, v)])
noise = lambda x, s: x + rng.normal(0, s * np.std(x), x.shape)
h, v, a = noise(h, 0.002), noise(v, 0.002), noise(a, 0.005)

names = ["1", "burn", "h", "v", "v|v|", "v|v|*exp(-h/H)"]
burn = (t < BURN).astype(float)
X = np.column_stack([np.ones_like(t), burn, h, v, v * np.abs(v), v * np.abs(v) * np.exp(-h / H_SCALE)])

scale = np.linalg.norm(X, axis=0)
coef = np.linalg.lstsq(X / scale, a, rcond=None)[0]
for _ in range(10):
    small = np.abs(coef) < 0.05 * np.abs(coef).max()
    coef[small] = 0
    big = ~small
    coef[big] = np.linalg.lstsq(X[:, big] / scale[big], a, rcond=None)[0]
coef /= scale

c = dict(zip(names, coef))
g_hat, thrust_hat, drag_hat = -c["1"], c["burn"], -c["v|v|*exp(-h/H)"]
print("Constant   Discovered   True")
print(f"gravity    {g_hat:9.4f}   {G}")
print(f"thrust     {thrust_hat:9.4f}   {THRUST_ACC}")
print(f"drag       {drag_hat:9.5f}   {CD_A_RHO0}")

def learned(ti, y):
    return [y[1], (thrust_hat if ti < BURN else 0) - g_hat - drag_hat * np.exp(-y[0] / H_SCALE) * y[1] * abs(y[1])]

pred = solve_ivp(learned, [0, T_LAND], [0, 0], t_eval=t, rtol=1e-9, atol=1e-9).y
h_true, v_true = sol.sol(t)
print(f"\n{'t (s)':>6} {'h pred (m)':>11} {'h true (m)':>11} {'v pred (m/s)':>13} {'v true (m/s)':>13}")
for row in zip(t, pred[0], h_true, pred[1], v_true):
    print(f"{row[0]:6.2f} {row[1]:11.1f} {row[2]:11.1f} {row[3]:13.2f} {row[4]:13.2f}")
print(f"\nMax altitude error: {np.abs(pred[0] - h_true).max():.2f} m | Max velocity error: {np.abs(pred[1] - v_true).max():.3f} m/s")
