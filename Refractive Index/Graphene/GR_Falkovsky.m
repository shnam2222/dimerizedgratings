% =========================================================================
% Graphene material data/Falkovsky(Kubo)
% 
% Export material data of Falkovsky Graphene conductivity/permittivity
%
% ver.191024, Sangjun Han
% =========================================================================
clear;


%% CONSTANTS %%
    e = 1.6021773349e-19;
    k = 1.38064852e-23;
    hb = 1.0545718e-34;
    eps0 = 8.85418781762039e-12;
    vf = 1e6;%8.73017e5;
    c = 299792458;
    pi = 3.141592;



%% VARIABLES %%
    T = 298.15*k/e;               % Temperature [eV]
    tGR = 0.34e-9;             % graphene thickness [m]
    mu = 650e-4;              % mobility [m^2/Vs]
 
    
    wl_start = 1e-6;
    wl_end   = 15e-6;
    
    fstart = c/wl_end;    % [Hz]
    fend = c/wl_start;                 % [Hz]
    %fintv = 0.1;               % [Hz]
    %f = fstart:fintv:fend;     % [Hz]
    f = linspace(fend, fstart,600);
    
    wavelengths = c./f;

    
    
    w_Hz = 2*pi*f;             % [Freqeuency to angular, Hz]
    w = hb*w_Hz/e;             % [eV]
    
    Y0 = 1/377;                % admittance of air [S]

% ====== exporttype ====== %
% 1. isotropic epsilon         [freq, re(eps), im(eps)]
% 2. isotropic nk              [freq, n, k]
% 3. anisotropic epsilon       [freq, re(eps), im(eps)]
% 4. sheet conductivity        [freq, re(sigma), im(sigma)] [S]

    exporttype = 2;
    mix = 0;                   %re&im 같이 뽑을지 따로 뽑을지 결정 (1; 같이, 0; 따로)

%  structures={
%             };
        
        for j=1
            
%== Ef_sets (뽑고 싶은 Ef를 matrix로 입력) ==%

    structure = '240618_mob650'; %structures{j,2};
    save_directory = [structure];
    mkdir(save_directory);  
    %Ef_sets = [0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1];
    Ef_sets = [0.05 0.6];
   

%% CALCULATIONS %%
n_surface = e*(e*Ef_sets/(hb*vf)).^2 /pi;    % surface charge density [C/m^2]

    for i = 1:length(Ef_sets(:,1))*length(Ef_sets(1,:))
        Ef = Ef_sets(i);
        Gm_imp = hb*(vf^2)/(mu*e*Ef); % [eV]
        Gm_oph = (abs(w-Ef)>0.2)* 2*(18.3e-3).*abs(w-sign(w-Ef)*0.2); % [eV]
        Gm = Gm_imp;% + Gm_oph;         % [eV]
        
        H = @(x) (tanh(x/T)./(cosh(Ef/T)./cosh(x/T) + 1));
        Hwhalf = H(w/2);
        
        dz = 0.0001;
        Hint = 0;
        for z=0:dz:1000
            Hint = Hint + dz*(H(z)-Hwhalf)./((w).^2 - 4*z^2);
        end     
        
        % Sheet conductivity [S]
        sigma = ((2i*e^2 *T)./(pi*hb*(w+1i*Gm))) *log(2*cosh(Ef/(2*T)))...
                + (e^2/(4*hb))*(Hwhalf + (4i*(w)/pi).*Hint); 
        
        % Relative permitivity
        eps_GR = 1+ 1i*sigma./(w_Hz*tGR*eps0);
    
        
            
%% FILE EXPORT
    if exporttype==1
    %     isotropic eps
            filename=[save_directory,'\epsilon_iso_graphene_',num2str(fstart),'~',num2str(fend),'THz_',num2str(Ef),'eV_','mu=',num2str(mu/1e-4),'_t=',num2str(tGR/1e-9),'nm'];
            if mix == 0
                dlmwrite([filename,'_re.txt'],[f*1e12;real(eps_GR)]','delimiter','\t','newline','pc','precision','%.16g');
                dlmwrite([filename,'_im.txt'],[f*1e12;imag(eps_GR)]','delimiter','\t','newline','pc','precision','%.16g');
            elseif mix == 1
                dlmwrite([filename,'_re&im.txt'],[f*1e12;real(eps_GR);imag(eps_GR)]','delimiter','\t','newline','pc','precision','%.16g');
            end

    elseif exporttype==2
    %     isotropic nk
            filename=[save_directory,'\nk_',num2str(i),'iso_graphene_',num2str(fstart),'~',num2str(fend),'THz_',num2str(Ef),'eV_','mu=',num2str(mu/1e-4),'_t=',num2str(tGR/1e-9),'nm'];
            if mix == 0
                dlmwrite([filename,'_re.csv'],[wavelengths*1e6;real(sqrt(eps_GR))]','delimiter',',','newline','pc','precision','%.8g');
                dlmwrite([filename,'_im.csv'],[wavelengths*1e6;imag(sqrt(eps_GR))]','delimiter',',','newline','pc','precision','%.8g');
            elseif mix == 1
                dlmwrite([filename,'_re&im.txt'],[wavelengths*1e6;real(sqrt(eps_GR));imag(sqrt(eps_GR))]','delimiter','\t','newline','pc','precision','%.8g');
            end

    elseif exporttype==3
    %     anisotropic eps
            filename=[save_directory,'\epsilon_aniso_graphene_',num2str(fstart),'~',num2str(fend),'THz_',num2str(Ef),'eV_','mu=',num2str(mu/1e-4),'_t=',num2str(tGR/1e-9),'nm'];
            if mix == 0
                dlmwrite([filename,'_re.txt'],[f*1e12;real(eps_GR);ones(1,length(eps_GR));real(eps_GR)]','delimiter','\t','newline','pc','precision','%.16g');
                dlmwrite([filename,'_im.txt'],[f*1e12;imag(eps_GR);zeros(1,length(eps_GR));imag(eps_GR)]','delimiter','\t','newline','pc','precision','%.16g');
            elseif mix == 1
                dlmwrite([filename,'_re&im.txt'],[f*1e12;real(eps_GR);imag(eps_GR);ones(1,length(eps_GR));zeros(1,length(eps_GR));real(eps_GR);imag(eps_GR)]','delimiter','\t','newline','pc','precision','%.16g');
            end

    elseif exporttype==4
          filename=[save_directory,'\sigma_',num2str(i),'imp only_graphene_',num2str(fstart),'~',num2str(fend),'THz_','mu=',num2str(mu/1e-4)];
    %     sheet conductivity
            if mix == 0
                dlmwrite([filename,'_re.txt'],[wavelengths*1e6;real(sigma)]','delimiter','\t','newline','pc','precision','%.16g');
                dlmwrite([filename,'_im.txt'],[wavelengths*1e6;imag(sigma)]','delimiter','\t','newline','pc','precision','%.16g');
            elseif mix == 1
                dlmwrite([filename,'_re&im.txt'],[wavelengths*1e6;real(sigma);imag(sigma)]','delimiter','\t','newline','pc','precision','%.16g');
            end

    end
    disp([num2str(j),' - ',num2str(i),'/',num2str(length(Ef_sets(:,1))*length(Ef_sets(1,:)))]);
    end
    
    dlmwrite([save_directory,'\[!] Indices-Efs.txt'],[1:length(Ef_sets);Ef_sets]','delimiter','\t','newline','pc');
    disp([num2str(j),' DONE!']);            
        end
%  figure(1)
%  hold on
%  plot(f,real(sigma));
%  plot(f,imag(sigma));
%  grid on