import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
import streamlit as st

# --- Page Configuration ---
st.set_page_config(page_title="Monetary OLG Model", layout="wide")
st.title("Dynamic Equilibrium Paths in the OLG Model")
st.markdown("Adjust the parameters in the sidebar to see how the value of money evolves.")


# --- Core Functions ---
def implicit_function(phi_next, phi_t, e1, e2, mt, beta, sigma):
    if phi_t <= 0 or (e1 - phi_t * mt) <= 0:
        return np.inf

    A = (beta / phi_t) ** (1 / sigma) * (e1 - phi_t * mt)
    return phi_next ** (1 / sigma) * A - phi_next * mt - e2


def solve_phi_next(phi_t, e1, e2, mt, beta, sigma):
    initial_guess = phi_t
    phi_next_solution, info, ier, msg = fsolve(
        implicit_function,
        initial_guess,
        args=(phi_t, e1, e2, mt, beta, sigma),
        full_output=True
    )
    if ier == 1 and phi_next_solution[0] > 0:
        return phi_next_solution[0]
    else:
        return np.nan

    # --- Sidebar UI (Sliders) ---


st.sidebar.header("Model Parameters")
e1 = st.sidebar.slider(r"$e_1$ (Endowment Young)", 5.0, 20.0, 10.0, 1.0)
e2 = st.sidebar.slider(r"$e_2$ (Endowment Old)", 0.0, 10.0, 2.0, 0.5)
mt = st.sidebar.slider(r"$m_t$ (Money Supply $M$)", 0.1, 5.0, 1.0, 0.1)
beta = st.sidebar.slider(r"$\beta$ (Discount Factor)", 0.1, 1.0, 0.9, 0.05)
sigma = st.sidebar.slider(r"$\sigma$ (CRRA parameter)", 0.5, 3.0, 1.0, 0.1)
phi1 = st.sidebar.slider(r"$\phi_1$ (Initial Value of Money)", 0.01, 1.5, 0.2, 0.05)

# --- Plotting ---
fig, ax = plt.subplots(figsize=(10, 7))

# 1. Calculate Steady State
phi_star = (beta ** (1 / sigma) * e1 - e2) / ((1 + beta ** (1 / sigma)) * mt)

# 2. Generate the curve
phi_t_vals = np.linspace(0.01, (e1 / mt) * 0.95, 200)
phi_next_vals = [solve_phi_next(p, e1, e2, mt, beta, sigma) for p in phi_t_vals]

# Plot primary curve and 45-degree line
ax.plot(phi_t_vals, phi_next_vals, 'b-', linewidth=2, label=r'$\phi_{t+1} = f(\phi_t)$')
max_val = max(np.nanmax(phi_next_vals), phi_star * 2.5) if phi_star > 0 else 5.0
ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.5, label='45-degree line')

# Plot Steady State
if phi_star > 0:
    ax.plot(phi_star, phi_star, 'ro', markersize=8, label=rf'Stationary Eq ($\phi^*$ = {phi_star:.2f})')

# 3. Simulate the Cobweb Path
current_phi = phi1
iterations = 15
path_x, path_y = [current_phi], [0]

for _ in range(iterations):
    next_phi = solve_phi_next(current_phi, e1, e2, mt, beta, sigma)
    if np.isnan(next_phi) or next_phi > max_val * 2:
        break

    path_x.extend([current_phi, next_phi])
    path_y.extend([next_phi, next_phi])
    current_phi = next_phi

ax.plot(path_x, path_y, 'g-', linewidth=1.5, alpha=0.8, marker='.')
ax.plot(phi1, 0, 'go', label=rf'Initial $\phi_1$ = {phi1:.2f}')

# Formatting
ax.set_xlim(0, phi_star * 2.5 if phi_star > 0 else 2.0)
ax.set_ylim(0, phi_star * 2.5 if phi_star > 0 else 2.0)
ax.set_xlabel(r'Value of money today ($\phi_t$)')
ax.set_ylabel(r'Value of money tomorrow ($\phi_{t+1}$)')
ax.set_title(rf'Dynamic Equilibrium Paths ($\sigma$ = {sigma:.2f})')
ax.legend()
ax.grid(True, alpha=0.3)

# Render the plot in the web app
st.pyplot(fig)

# Show steady state data text
st.write(rf"Calculated Steady State ($\phi^*$): {phi_star:.4f}")