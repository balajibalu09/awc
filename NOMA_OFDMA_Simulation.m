%% =====================================================================
%  Performance Analysis of NOMA and OFDMA in 5G Wireless Networks
%  -----------------------------------------------------------------
%  MATLAB simulation script.
%  Generates the following result figures:
%       1) BER vs SNR (Rayleigh fading, BPSK)
%       2) Sum spectral efficiency vs SNR
%       3) Per-user achievable rate vs SNR
%       4) Outage probability vs SNR
%       5) Jain's fairness index vs number of users
%       6) Energy efficiency vs SNR
%       7) Sum rate vs number of users
%% =====================================================================

clear; close all; clc;
rng(2026);                              

%% --------- Common simulation parameters -----------------------------
SNR_dB        = 0:2:30;                 % SNR sweep
N_bits        = 2e6;                    % bits/user for BER simulation
N_real        = 5000;                   % channel realisations for capacity
BW            = 1e6;                    % system bandwidth (Hz)

% NOMA two-user power allocation (weak user gets more power)
alpha_weak    = 0.75;
alpha_strong  = 0.25;

% Distance-based path-loss (normalised)
d_strong      = 50;     d_weak   = 200;     pl_exp = 3;
PL_strong     = d_strong^(-pl_exp);
PL_weak       = d_weak^(-pl_exp);
norm_pl       = PL_strong;
PL_strong     = PL_strong / norm_pl;
PL_weak       = PL_weak   / norm_pl;

%% =====================================================================
%  Section 1: BER vs SNR (BPSK over Rayleigh)
%  =====================================================================
ber_noma_s = zeros(size(SNR_dB));
ber_noma_w = zeros(size(SNR_dB));
ber_ofdma_s = zeros(size(SNR_dB));
ber_ofdma_w = zeros(size(SNR_dB));

for k = 1:length(SNR_dB)
    snr_lin = 10^(SNR_dB(k)/10);
    N0  = 1;
    Pt  = snr_lin * N0;

    b1 = randi([0 1], 1, N_bits);    s1 = 2*b1 - 1;
    b2 = randi([0 1], 1, N_bits);    s2 = 2*b2 - 1;

    h1 = (randn(1,N_bits)+1j*randn(1,N_bits))/sqrt(2);
    h2 = (randn(1,N_bits)+1j*randn(1,N_bits))/sqrt(2);
    h1 = h1*sqrt(PL_strong);
    h2 = h2*sqrt(PL_weak);
    n1 = (randn(1,N_bits)+1j*randn(1,N_bits))*sqrt(N0/2);
    n2 = (randn(1,N_bits)+1j*randn(1,N_bits))*sqrt(N0/2);

    % --- NOMA: superposition coding -------------------------------
    x  = sqrt(alpha_strong*Pt)*s1 + sqrt(alpha_weak*Pt)*s2;
    y1 = h1.*x + n1;     y2 = h2.*x + n2;

    % weak user decodes own bits, treating strong-user term as noise
    r2 = real(y2./h2);
    b2_hat = double(r2 > 0);
    ber_noma_w(k) = mean(b2_hat ~= b2);

    % strong user performs SIC: decode s2, subtract, decode s1
    r1   = y1./h1;
    s2_hat = 2*double(real(r1)>0) - 1;
    y1_clean = y1 - h1.*sqrt(alpha_weak*Pt).*s2_hat;
    b1_hat = double(real(y1_clean./h1)>0);
    ber_noma_s(k) = mean(b1_hat ~= b1);

    % --- OFDMA: orthogonal subcarriers, half power each -----------
    x1 = sqrt(Pt/2)*s1;     x2 = sqrt(Pt/2)*s2;
    y1o = h1.*x1 + n1;      y2o = h2.*x2 + n2;
    b1o = double(real(y1o./h1)>0);
    b2o = double(real(y2o./h2)>0);
    ber_ofdma_s(k) = mean(b1o ~= b1);
    ber_ofdma_w(k) = mean(b2o ~= b2);
end

figure; 
semilogy(SNR_dB,ber_noma_s,'o-','LineWidth',1.5); hold on;
semilogy(SNR_dB,ber_noma_w,'s-','LineWidth',1.5);
semilogy(SNR_dB,ber_ofdma_s,'d--','LineWidth',1.5);
semilogy(SNR_dB,ber_ofdma_w,'^--','LineWidth',1.5);
grid on; xlabel('Transmit SNR (dB)'); ylabel('Bit Error Rate (BER)');
title('BER Performance over Rayleigh Fading Channel');
legend('NOMA Strong (SIC)','NOMA Weak','OFDMA Strong','OFDMA Weak','Location','southwest');
ylim([1e-5 1]);
saveas(gcf,'fig_ber.png');

%% =====================================================================
%  Section 2: Sum-rate / per-user capacity
%  =====================================================================
sum_noma   = zeros(size(SNR_dB));   sum_ofdma  = zeros(size(SNR_dB));
rs_noma    = zeros(size(SNR_dB));   rw_noma    = zeros(size(SNR_dB));
rs_ofdma   = zeros(size(SNR_dB));   rw_ofdma   = zeros(size(SNR_dB));

for k = 1:length(SNR_dB)
    snr_lin = 10^(SNR_dB(k)/10);

    h1 = (randn(1,N_real)+1j*randn(1,N_real))/sqrt(2);
    h2 = (randn(1,N_real)+1j*randn(1,N_real))/sqrt(2);
    g1 = (abs(h1).^2)*PL_strong;
    g2 = (abs(h2).^2)*PL_weak;

    rs = log2(1 + alpha_strong*snr_lin*g1);
    rw = log2(1 + (alpha_weak*snr_lin*g2)./(alpha_strong*snr_lin*g2 + 1));
    rs_noma(k)  = mean(rs);
    rw_noma(k)  = mean(rw);
    sum_noma(k) = rs_noma(k)+rw_noma(k);

    r1o = 0.5*log2(1 + (snr_lin/2)*g1);
    r2o = 0.5*log2(1 + (snr_lin/2)*g2);
    rs_ofdma(k)  = mean(r1o);
    rw_ofdma(k)  = mean(r2o);
    sum_ofdma(k) = rs_ofdma(k)+rw_ofdma(k);
end

figure;
plot(SNR_dB,sum_noma,'o-','LineWidth',2); hold on;
plot(SNR_dB,sum_ofdma,'s--','LineWidth',2);
grid on; xlabel('Transmit SNR (dB)'); ylabel('Sum Spectral Efficiency (bps/Hz)');
title('Sum Rate vs SNR — NOMA vs OFDMA');
legend('NOMA','OFDMA','Location','northwest');
saveas(gcf,'fig_sumrate.png');

figure;
plot(SNR_dB,rs_noma,'o-','LineWidth',1.5); hold on;
plot(SNR_dB,rw_noma,'s-','LineWidth',1.5);
plot(SNR_dB,rs_ofdma,'d--','LineWidth',1.5);
plot(SNR_dB,rw_ofdma,'^--','LineWidth',1.5);
grid on; xlabel('Transmit SNR (dB)'); ylabel('Achievable Rate (bps/Hz)');
title('Per-User Rate vs SNR');
legend('NOMA Strong','NOMA Weak','OFDMA Strong','OFDMA Weak','Location','northwest');
saveas(gcf,'fig_per_user_rate.png');

%% =====================================================================
%  Section 3: Outage probability
%  =====================================================================
R_th = 1.0;
out_n_s = zeros(size(SNR_dB));    out_n_w = zeros(size(SNR_dB));
out_o_s = zeros(size(SNR_dB));    out_o_w = zeros(size(SNR_dB));

for k = 1:length(SNR_dB)
    snr_lin = 10^(SNR_dB(k)/10);
    h1 = (randn(1,N_real)+1j*randn(1,N_real))/sqrt(2);
    h2 = (randn(1,N_real)+1j*randn(1,N_real))/sqrt(2);
    g1 = (abs(h1).^2)*PL_strong;
    g2 = (abs(h2).^2)*PL_weak;

    rs = log2(1 + alpha_strong*snr_lin*g1);
    rw = log2(1 + (alpha_weak*snr_lin*g2)./(alpha_strong*snr_lin*g2 + 1));
    out_n_s(k) = mean(rs < R_th);
    out_n_w(k) = mean(rw < R_th);

    r1o = 0.5*log2(1 + (snr_lin/2)*g1);
    r2o = 0.5*log2(1 + (snr_lin/2)*g2);
    out_o_s(k) = mean(r1o < R_th);
    out_o_w(k) = mean(r2o < R_th);
end

figure;
semilogy(SNR_dB,max(out_n_s,1e-5),'o-','LineWidth',1.5); hold on;
semilogy(SNR_dB,max(out_n_w,1e-5),'s-','LineWidth',1.5);
semilogy(SNR_dB,max(out_o_s,1e-5),'d--','LineWidth',1.5);
semilogy(SNR_dB,max(out_o_w,1e-5),'^--','LineWidth',1.5);
grid on; xlabel('Transmit SNR (dB)'); ylabel('Outage Probability');
title(sprintf('Outage Probability vs SNR (R_{th}=%.1f bps/Hz)',R_th));
legend('NOMA Strong','NOMA Weak','OFDMA Strong','OFDMA Weak','Location','southwest');
ylim([1e-4 1]);
saveas(gcf,'fig_outage.png');

%% =====================================================================
%  Section 4: Fairness vs number of users
%  =====================================================================
user_counts = [2 4 6 8 10 12];
snr_db_fix  = 20;   snr_lin = 10^(snr_db_fix/10);
fair_n = zeros(size(user_counts));   fair_o = zeros(size(user_counts));

for j = 1:length(user_counts)
    K = user_counts(j);
    distances = linspace(50,250,K);
    pg = (distances.^(-pl_exp))/(distances(1)^(-pl_exp));
    a_n = 1./pg;    a_n = a_n / sum(a_n);     % more power to weaker users
    rates_n = zeros(N_real,K);
    rates_o = zeros(N_real,K);

    for r = 1:N_real
        h = (randn(1,K)+1j*randn(1,K))/sqrt(2);
        g = (abs(h).^2).*pg;
        [~,ord] = sort(g);                    % weakest first
        for ii = 1:K
            u   = ord(ii);
            inter = sum(a_n(ord(ii+1:end)));
            sinr  = (a_n(u)*snr_lin*g(u))/(inter*snr_lin*g(u) + 1);
            rates_n(r,u) = log2(1 + sinr);
        end
        rates_o(r,:) = (1/K)*log2(1 + (snr_lin/K)*g);
    end
    rn = mean(rates_n,1);   ro = mean(rates_o,1);
    fair_n(j) = (sum(rn))^2 / (K*sum(rn.^2));
    fair_o(j) = (sum(ro))^2 / (K*sum(ro.^2));
end

figure;
plot(user_counts,fair_n,'o-','LineWidth',2,'MarkerSize',8); hold on;
plot(user_counts,fair_o,'s--','LineWidth',2,'MarkerSize',8);
grid on; xlabel('Number of Users (K)'); ylabel("Jain's Fairness Index");
title('Fairness Comparison (SNR = 20 dB)');
ylim([0 1.05]); legend('NOMA','OFDMA','Location','best');
saveas(gcf,'fig_fairness.png');

%% =====================================================================
%  Section 5: Energy efficiency vs SNR
%  =====================================================================
Pc = 0.1;
ee_n = zeros(size(SNR_dB));   ee_o = zeros(size(SNR_dB));
for k = 1:length(SNR_dB)
    snr_lin = 10^(SNR_dB(k)/10);
    Pt = snr_lin * 1e-3;
    h1 = (randn(1,N_real)+1j*randn(1,N_real))/sqrt(2);
    h2 = (randn(1,N_real)+1j*randn(1,N_real))/sqrt(2);
    g1 = (abs(h1).^2)*PL_strong;
    g2 = (abs(h2).^2)*PL_weak;
    rn = mean(log2(1+alpha_strong*snr_lin*g1) + ...
              log2(1+(alpha_weak*snr_lin*g2)./(alpha_strong*snr_lin*g2+1)));
    ro = mean(0.5*log2(1+(snr_lin/2)*g1) + 0.5*log2(1+(snr_lin/2)*g2));
    ee_n(k) = (BW*rn)/(Pt + 2*Pc);
    ee_o(k) = (BW*ro)/(Pt + 2*Pc);
end

figure;
plot(SNR_dB,ee_n/1e6,'o-','LineWidth',2); hold on;
plot(SNR_dB,ee_o/1e6,'s--','LineWidth',2);
grid on; xlabel('Transmit SNR (dB)'); ylabel('Energy Efficiency (Mbits/Joule)');
title('Energy Efficiency vs SNR'); legend('NOMA','OFDMA','Location','best');
saveas(gcf,'fig_energy.png');

%% =====================================================================
%  Section 6: Sum-rate scaling with number of users
%  =====================================================================
user_counts2 = [2 4 6 8 10 12 14 16];
sn = zeros(size(user_counts2));   so = zeros(size(user_counts2));
for j = 1:length(user_counts2)
    K = user_counts2(j);
    distances = linspace(50,250,K);
    pg = (distances.^(-pl_exp))/(distances(1)^(-pl_exp));
    a_n = 1./pg;    a_n = a_n / sum(a_n);
    accN = 0;   accO = 0;
    for r = 1:N_real
        h = (randn(1,K)+1j*randn(1,K))/sqrt(2);
        g = (abs(h).^2).*pg;
        [~,ord] = sort(g);
        for ii = 1:K
            u = ord(ii);
            inter = sum(a_n(ord(ii+1:end)));
            sinr  = (a_n(u)*snr_lin*g(u))/(inter*snr_lin*g(u) + 1);
            accN  = accN + log2(1 + sinr);
        end
        accO = accO + sum((1/K)*log2(1 + (snr_lin/K)*g));
    end
    sn(j) = accN/N_real;   so(j) = accO/N_real;
end

figure;
plot(user_counts2,sn,'o-','LineWidth',2,'MarkerSize',8); hold on;
plot(user_counts2,so,'s--','LineWidth',2,'MarkerSize',8);
grid on; xlabel('Number of Users (K)'); ylabel('Sum Spectral Efficiency (bps/Hz)');
title('Sum Rate vs Number of Users (SNR = 20 dB)');
legend('NOMA','OFDMA','Location','best');
saveas(gcf,'fig_users.png');

disp('Simulation complete. All figures saved to current directory.');
