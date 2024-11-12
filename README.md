# dimerizedgratings
Repository for code on tunable dimerized high contrast gratings. Includes methods of stochastic gradient descent and BOBYQA. The tunable device has geometry as below. Dielectric SiO2 is depositied above Au backreflector, with graphene layer on top. Two Si slabs on top of the graphene are dimerized with factor delta to create BICs (Figure 1). Graphene Fermi level tunes the reflection peak, and the ability of such tuning is defined as tunability below.

$$
T = \frac{\Delta \omega_0}{\gamma} = \frac{\omega_0 (\beta_1) - \omega_0 (\beta_0)}{\gamma},  \ \ \ \  \gamma = (\gamma(\beta_1)+\gamma(\beta_0))/2
$$

<img src="https://github.com/user-attachments/assets/ccbd93ef-2db0-4791-bc91-60ae2df9484a" width="430">
<img src="https://github.com/user-attachments/assets/ae66d8eb-4d6f-4acf-b7cf-abb4223c32c4" width="550">
Figure 1. (Left) Optimization of a tunable DHCG structure. a) Schematic of the unit cell used in the optimization of the structure b) Table of parameters for the optimal structures. Optimization of Design 1 allowed Si thickness t to range from 50 nm to 2 μm. Optimization of Design 2 restricted t to 220 nm. Optimization of Design 3 changed nSi = 2.6. c) Reflection spectra at TE polarization for Design 1, Design 2, and Design 3 at different EF. Figure from Gregory Holdman's thesis. (Right) Tunability Optimization at target wavelength 5500nm with Si index 3.5 and gradient descent. Obtained tunability is 1.9.
