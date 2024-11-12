function H = H(Ef, T, nu)

    kB = 1.38064852 * 10^(-23);
    H = sinh(nu/(kB*T))/(cosh(Ef/(kB*T))+cosh(nu/(kB*T)));
end

