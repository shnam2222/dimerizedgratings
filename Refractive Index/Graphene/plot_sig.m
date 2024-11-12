hbar = 1.0545718 * 10^(-34);
kB = 1.38064852 * 10^(-23);
T = 300;
e = 1.6*10^(-19);
vF = 1.1e6; %Fermi Velocity of Graphene
E_f = 0.4*e;
n = (E_f/(hbar*vF))^2/pi;
mu = 550*10^-4; %Carrier Mobility in SI
epsilon_0 = 8.8541878128*10^-12;	
c = 299792458;
sigma0 = e^2/4/hbar;

freqs = E_f/hbar*linspace(0,3.5);

sigma = sigma_doped_GR(freqs,E_f);
real_sig = real(sigma);
imag_sig = imag(sigma);
figure
plot(freqs*hbar/E_f,real_sig,freqs*hbar/E_f,imag_sig);
ylim([-3,3])
%ylim([-sigma0,3*sigma0])