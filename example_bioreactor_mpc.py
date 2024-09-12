import numpy as np
import matplotlib.pyplot as plt
from utils.bioreactor_mpc import BioReactorMPC

# Define the initial state
x0 = np.array([1.2, 3.0, 10, 0.1])

# Define the initial control input
u0 = np.array([0.0, 0.0])

# Define the desired state
xd = np.array([1.0, 10.0, 0.0, 0.0])

# Define the MPC parameters
n_states = 4
n_controls = 2
n_horizon = 10
time_horizon = 5/6

bioreactor_mpc = BioReactorMPC(x0, u0, n_states, n_controls, n_horizon, time_horizon, n_int=1)

# Set bounds for the optimization variables
# Bound controls
u_lb = 0.0
u_ub = 0.4
lbu = np.array([u_lb, u_lb])
ubu = np.array([u_ub, u_ub])
bioreactor_mpc.set_bounds_u(lbu, ubu)

# Bound states
v_min = 0.5
v_max = 2.0
x_min = 0.0
x_max = 100.0
s_min = 0.0
s_max = 100.0
co2_min = 0.0
co2_max = 1.0
lb = np.array([v_min, x_min, s_min, co2_min])
ub = np.array([v_max, x_max, s_max, co2_max])
bioreactor_mpc.set_bounds_x(lb, ub)


# Load the model
bioreactor_mpc.bioreactor_model()

# Define the gains for the cost function
gains = {
    'k_volume': 1e1,
    'k_biomass': 1,
    'reg': 0,
}
# Create the cost function
bioreactor_mpc.cost_function(gains, xdes=xd)

# # Add initial state constraints
# bioreactor_mpc.add_constraint([bioreactor_mpc.state[0]], np.reshape(x0, (-1,1)), np.reshape(x0, (-1,1)))

# Define the multiple shooting constraints
bioreactor_mpc.integrator()
bioreactor_mpc.multiple_shooting()
    
# Set terminal cost
terminal_gains = {
    'alpha': 0,
    'beta': 0.0,
    'gamma': 0,
}
# bioreactor_mpc.set_terminal_cost(xd, terminal_gains)

# Set initial guess
# bioreactor_mpc.opt_var_0, v_init, x_init, s_init, co2_init, F0_init, F1_init = bioreactor_mpc.init_values(x0, u0)

# tgrid0 = np.linspace(0, bioreactor_mpc.T, bioreactor_mpc.N)

# # Plot initial condition
# plt.figure(1)
# plt.subplot(2, 2, 1)
# plt.plot(tgrid0, v_init, '-o', label='volume')
# plt.xlabel('t')
# plt.ylabel('L')
# plt.legend()
# plt.subplot(2, 2, 2)
# plt.plot(tgrid0, x_init, '-o', label='biomass') 
# plt.xlabel('t')
# plt.ylabel('g/L')
# plt.legend()
# plt.subplot(2, 2, 3)
# plt.plot(tgrid0, s_init, '-o', label='substrate')
# plt.xlabel('t')
# plt.ylabel('g/L')
# plt.legend()
# plt.subplot(2, 2, 4)
# plt.plot(tgrid0, co2_init, '-o', label='CO2')
# plt.xlabel('t')
# plt.ylabel('%')
# plt.grid()

# # Plot controls
# plt.figure(2)
# plt.subplot(2, 1, 1)
# plt.step(tgrid0, F0_init, '-o', label='Feed', where='post')
# plt.xlabel('t')
# plt.ylabel('L/h')
# plt.legend()
# plt.grid()
# plt.subplot(2, 1, 2)
# plt.step(tgrid0, F1_init, '-o', label='Outlet', where='post')
# plt.xlabel('t')
# plt.ylabel('L/h')
# plt.legend()
# plt.grid()
# # plt.show()

# Create the optimization problem
bioreactor_mpc.create_solver()

n_steps = 100

state_hist = [x0.tolist()]
control_hist = []

x_meas = x0
for i in range(n_steps):
    print(f'Step {i+1}')
    # Solve the optimization problem
    v_opt, x_opt, s_opt, co2_opt, F0_opt, F1_opt = bioreactor_mpc.step(x_meas, u0)
    
    # Control action
    control = [F0_opt[0], F1_opt[0]]
    control_hist.append(control)
    
    # Update the system state
    new_state = bioreactor_mpc.update_state(x_meas, control)
    x_meas = new_state['xf'].full().flatten().tolist()
    
    state_hist.append(x_meas)


# Extract the optimal state
v_opt = [state_hist[i][0] for i in range(len(state_hist))]
x_opt = [state_hist[i][1] for i in range(len(state_hist))]
s_opt = [state_hist[i][2] for i in range(len(state_hist))]
co2_opt = [state_hist[i][3] for i in range(len(state_hist))]

# Extract the optimal controls
F0_opt = [control_hist[i][0] for i in range(len(control_hist))]
F1_opt = [control_hist[i][1] for i in range(len(control_hist))]


# Plot the results
tgrid = np.linspace(0, bioreactor_mpc.T, bioreactor_mpc.N)
#print(tgrid)

# print optimal state variables
plt.figure(3)
plt.subplot(2, 2, 1)
plt.plot(tgrid, v_opt, '-o', label='volume')
plt.axhline(y=xd[0], color='r', linestyle='--')
plt.xlabel('t')
plt.ylabel('L')
plt.grid()
plt.legend()
plt.subplot(2, 2, 2)
plt.plot(tgrid, x_opt, '-o', label='biomass')
plt.axhline(y=xd[1], color='r', linestyle='--')
plt.xlabel('t')
plt.ylabel('g/L')
plt.grid()
plt.legend()
plt.subplot(2, 2, 3)
plt.plot(tgrid, s_opt, '-o', label='substrate')
plt.xlabel('t')
plt.ylabel('g/L')
plt.grid()
plt.legend()
plt.subplot(2, 2, 4)
plt.plot(tgrid, co2_opt, '-o', label='CO2')
plt.xlabel('t')
plt.ylabel('%')
plt.legend()
plt.grid()


# print optimal controls
plt.figure(4)
plt.subplot(2, 1, 1)
plt.plot(tgrid, F0_opt, '-o', label='Feed')
plt.xlabel('t')
plt.ylabel('L/h')
plt.legend()
plt.grid()
plt.subplot(2, 1, 2)
plt.plot(tgrid, F1_opt, '-o', label='Outlet')
plt.xlabel('t')
plt.ylabel('L/h')
plt.legend()
plt.grid()
plt.show()