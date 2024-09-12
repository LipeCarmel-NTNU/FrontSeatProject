import casadi as ca

def bioreactor_model(x, FlowVar, par):
    # States
    V, X, S, CO2 = x[0], x[1], x[2], x[3]

    if S < 0:
        S = 0

    # Manipulated variables
    Fin = FlowVar[0]
    Fout = FlowVar[1]

    # Parameters
    Sin = 200
    Q = 2
    V_total = 2.7  # L
    CO2in = 0.04  # 100*volume fraction of CO2 in feed

    Y_XS = 0.4204
    mu_max = 0.1945
    Ks = 0.0070
    Y_CO2X = 0.5430
    kd = 0.0060

    # Define the rate
    mu = mu_max * (S / (Ks + S))

    # Differential equations
    dV = Fin - Fout
    dX = -X * (Fin / V) + mu * X - kd * X  # Biomass
    dS = (Sin - S) * (Fin / V) - mu * X / Y_XS  # Substrate
    dCO2 = ((CO2in - CO2) * Q + mu * X / Y_CO2X) / (V_total - V)  # CO2

    # Output
    dx = ca.vertcat(dV, dX, dS, dCO2)
    return dx
