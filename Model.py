def User_fxp_Cont(x,t,u,pxp,pxmp):
    """
    SUMMARY:
    It constructs the function fx_p for the non-linear case
        
    SYNTAX:
    assignment = User_fxp_Cont(x,t,u)
        
    ARGUMENTS:
    + x         - State variable     
    + t         - Current time
    + u         - Input variable  
        
    OUTPUTS:
    + fx_p      - Non-linear plant function
    """ 
    Fin = u[0]
    Fout = u[1]

    V = x[0]
    X = x[1]
    S = x[2]
    CO2 = x[3]

    Sin= 200
    Q = 2           # L/h
    V_total = 2.7   # L
    CO2in = 0.04    # 100*volume fraction of CO2 in feed
    Y_XS = 0.4204
    mu_max = 0.1945
    Ks = 0.0070
    Y_CO2X = 0.5430
    kd = 0.0060

    fx_p = vertcat\
    (\
    Fin-Fout, \
    -X.*(Fin/V) +  mu*X - kd*X, \
    (Sin-S)*(Fin/V) - mu*X/Y_XS, \
    ((CO2in - CO2)*Q + mu*X/Y_CO2X)/(V_total - V)\

    )    
    
    return fx_p

def User_fyp(x,u,t,pyp,pymp):
    """
    SUMMARY:
    It constructs the function User_fyp for the non-linear case
    
    SYNTAX:
    assignment = User_fyp(x,t)
  
    ARGUMENTS:
    + x             - State variable
    + t             - Variable that indicate the current iteration
    
    OUTPUTS:
    + fy_p      - Non-linear plant function     
    """ 
    
    fy_p = vertcat\
    (\
    x[0],\
    x[1],\
    x[3] \
    )
    
    return fy_p