function integral = integral_sigma_inter_prb(w,E_f)
   
   T = 300;
   tot = 1e4;
   e = 1.6*10^(-19);
   dnu = e/100;
   hbar = 1.0545718 * 10^(-34);
   integral = 0;
   for k =  1:tot
       nu = k*dnu;
       if nu < 10*E_f
            temp = (H(E_f, T, nu)-H(E_f, T, hbar*w/2))/((hbar*w)^2-4*nu^2);
       else
           temp = (1-H(E_f, T, hbar*w/2))/((hbar*w)^2-4*nu^2);
       end
       integral = integral + temp*dnu;
   end
end