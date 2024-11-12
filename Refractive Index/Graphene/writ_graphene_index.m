hbar = 1.0545718 * 10^(-34);
kB = 1.38064852 * 10^(-23);
T = 300;
e = 1.6*10^(-19);
vF = 1.1e6; %Fermi Velocity of Graphene
E_f = 0.6*e;
n = (E_f/(hbar*vF))^2/pi;
mu = 550*10^-4; %Carrier Mobility in SI
epsilon_0 = 8.8541878128*10^-12;	
c = 299792458;
sigma0 = e^2/4/hbar;
tau = 0.2e-9;

wavelengths = 1e-6*linspace(6.1,6.6);

freqs = 2*pi*c./wavelengths;

sigma = sigma0*sigma_doped_GR(freqs, E_f);

eps = 1 + 1i*sigma./(epsilon_0*tau*freqs);

n_index = sqrt(eps);

real_n = real(n_index);
imag_n = imag(n_index);
figure
plot(wavelengths,real_n,wavelengths,imag_n);
legend("real","imag")
M = transpose([1e6*wavelengths; imag_n]);
csvwrite('graphene_k_0.6eV.csv',M)

