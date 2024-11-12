function sigma_GR = sigma_doped_GR(frequency, E_f)
    hbar = 1.0545718 * 10^(-34);
    kB = 1.38064852 * 10^(-23);
    T = 300;
    e = 1.6*10^(-19);
    sigma_0 = e^2/(4*hbar);
    vF = 1.1e6; %Fermi Velocity of Graphene
    %E_f = hbar*vF*sqrt(pi*n);
    
    n = E_f^2/((hbar*vF)^2*pi);
    mu = 650*e-4; %Carrier Mobility in SI
    epsilon_0 = 8.8541878128*10^-12;	
    c = 299792458;
    
    
%{    
    Ef = E_f/e;
    w = frequency*hbar/e;
    Gm_imp = hbar*(vF^2)/(mu*e*Ef); % [eV]
    Gm_oph = (abs(w-Ef)>0.2)* 2*(18.3e-3).*abs(w-sign(w-Ef)*0.2); % [eV]
    Gm = Gm_imp + Gm_oph;         % [eV]
    
    T = kB*T/e;
    
    H = @(x) (tanh(x/T)./(cosh(Ef/T)./cosh(x/T) + 1));
    Hwhalf = H(w/2);
        
    dz = 0.0001;
    Hint = 0;
    for z=0:dz:1000
        Hint = Hint + dz*(H(z)-Hwhalf)./((w).^2 - 4*z^2);
    end     
        
    % Sheet conductivity [S]
    sigma_GR = ((2i*e^2 *T)./(pi*hbar*(w+1i*Gm))) *log(2*cosh(Ef/(2*T)))...
          + (e^2/(4*hbar))*(Hwhalf + (4i*(w)/pi).*Hint); 
    
%}    
    
   
    w_oph = 0.2*e/hbar;

    w = frequency;        % [Hz]

    gamma_imp = e*vF/(mu*sqrt(n*pi));

    gamma_oph = [];

    for index = 1:length(w)
       %gamma_oph(index) = 2*(18.3*10^-3)*abs(hbar.*w(index)-sign(hbar.*w(index)-E_f)*0.2*e).*(abs(hbar.*w(index)-E_f)>(0.2*e));
       gamma_oph(index) = 2*(18.3*10^-3)*abs(hbar.*w(index)+0.2*e+E_f).*(erf((hbar*w(index)-0.2*e)/(0.05*e))+erf((-hbar*w(index)-0.2*e)/(0.05*e))+2);
    end

    gamma = gamma_imp+gamma_oph;%%Already contains hbar term
    %gamma = gamma_imp;
    %gamma = (3.7e-3)*e;
    %gamma = 8e-3*e/hbar;
    
    %sigma_intra = (1/pi).*(4./(hbar*gamma-1i*hbar.*w)).*(E_f+2*kB*T*log(1+exp(-E_f/(kB*T))));
    sigma_inter = [];
    for index = 1:length(w)
        %H_w = sinh((hbar*w(index))/(2*kB*T))/(cosh(E_f/(kB*T))+cosh((hbar*w(index))/(2*kB*T)));
        sigma_inter(index) = H(E_f, T, hbar*w(index)/2)+1i*(4*hbar*w(index)/pi)*integral_sigma_inter_prb(w(index),E_f);
    end
    
    %sigma_inter = (H(E_f, T, w/2)+1i*(4*hbar*w./pi).*integral_sigma_inter_prb(w,E_f));
    
    sigma_intra = 4i*E_f./(pi*(hbar*w + 1i*hbar*gamma))*(1+2*kB*T*log(1+exp(-E_f/(kB*T))/E_f));
    %sigma_inter = heaviside(hbar*w - 2*E_f) + 1i./pi * log((hbar*w-2*E_f)/(hbar*w+2*E_f));
    sigma_GR = sigma_intra + sigma_inter; %*sigma0
    
end