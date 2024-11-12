hbar = 1.0545718 * 10^(-34);
kB = 1.38064852 * 10^(-23);
T = 300;
e = 1.6*10^(-19);
vF = 1.1e6; %Fermi Velocity of Graphene
n = (E_f/(hbar*vF)).^2/pi;
mu = 1; %Carrier Mobility in SI
epsilon_0 = 8.8541878128*10^-12;	
c = 299792458;
sigma0 = e^2/(4*hbar);
tau = 0.34e-9;

wavelengths = 1e-9*1550;

freqs = 2*pi*c/wavelengths;

E_f = e*linspace(0.2,0.8);

sigma = [];

for index = 1:length(E_f)
    sigma(index) = sigma0*sigma_doped_GR(freqs, E_f(index));
end


eps = 1 + 1i*sigma/(epsilon_0*tau*freqs);

n_index = sqrt(eps);

real_n = real(sigma/sigma0);
imag_n = imag(sigma/sigma0);
figure
plot(E_f/e,real_n);