% Function to call the ODE Model
function  [dx] = model_summer_school(x, FlowVar, par)
% States
V = x(1); X = x(2); S = x(3); CO2 = x(4);

if S < 0
    % Avoids numerical issues, but might be incompatible with symbolic
    S = 0;
end

% Manipulated variables

Fin = FlowVar(1);
Fout= FlowVar(2);

% Parameters
Sin= 200;
Q = 2;
V_total = 2.7;          % L
CO2in = 0.04;           % 100*volume fraction of CO2 in feed

Y_XS = 0.4204;
mu_max = 0.1945;
Ks = 0.0070;
Y_CO2X = 0.5430;
kd = 0.0060;

% Define the rate
mu = mu_max .*(S ./(Ks + S));

% The inputs are Fin, Fout
% Differential equations =
dV   = Fin-Fout;
dX   = -X.*(Fin/V) +  mu.*X - kd .*X; %Biomass
dS   = (Sin-S).*(Fin/V) - mu .*X ./Y_XS; % Substrate
dCO2 = ((CO2in - CO2).*Q + mu .*X./Y_CO2X)/(V_total - V);     % CO2

% Output =
dx = [dV; dX; dS; dCO2];
end
