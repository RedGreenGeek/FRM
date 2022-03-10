% Example from book 

clear 
Corr = [[1 0.9 0.6];[0.9 1 0.7];[0.6 0.7 1]];

Vols = [0.06 0.1 0.2]/100;

Cov = diag(Vols)*Corr*diag(Vols);


% Interpolate interest rates

Rate1 = spline( 0.25:0.25:0.5,0.055:0.005:0.06, 0.3);
Rate2 = spline( 0.5:0.5:1,0.06:0.01:0.07, 0.8);

% Present value of cashflow

PV1 = 50000/(1+Rate1)^0.3;
PV2 = (50000+1000000)/(1+Rate2)^0.8;

% Figure out how much to invest in bonds with maturities 0.25, 0.5 and
% 1 by matching volatility

Vol1 = spline( 0.25:0.25:0.5,Vols(1:2), 0.3);
Vol2 = spline( 0.5:0.5:1,Vols(2:3), 0.8);

syms alpha
eqn = Vol1^2 == alpha^2*Vols(1)^2 + (1-alpha)^2*Vols(2)^2 + 2*Corr(1,2)*Vols(1)*Vols(2)*alpha*(1-alpha);
solx = vpasolve(eqn,alpha);
solx = double(solx);
solx = solx(solx<1);

Amount(1) = solx*PV1;
Amount(2) = (1-solx)*PV1;



syms alpha2
eqn2 = Vol2^2 == alpha2^2*Vols(2)^2 + (1-alpha2)^2*Vols(3)^2 + 2*Corr(2,3)*Vols(3)*Vols(2)*alpha2*(1-alpha2);
solx2 = vpasolve(eqn2,alpha2);
solx2 = double(solx2);
solx2 = solx2(solx2<1);

Amount(2) = Amount(2) + solx2*PV2;
Amount(3) = (1-solx2)*PV2;

% Find VaR

PF_var = Amount*Cov(1:3,1:3)*Amount';

VaR = norminv(0.99)*(PF_var.^0.5)*sqrt(10);

