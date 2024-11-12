# dimerizedgratings
Repository for code on tunable dimerized high contrast gratings. Includes methods of stochastic gradient descent and BOBYQA. The tunable device has geometry as below. Dielectric SiO2 is depositied above Au backreflector, with graphene layer on top. Two Si slabs on top of the graphene are dimerized with factor delta to create BICs (Figure 1). Graphene Fermi level tunes the reflection peak, and the ability of such tuning is defined as tunability below.

$$
T = \frac{\Delta \omega_0}{\gamma} = \frac{\omega_0 (\beta_1) - \omega_0 (\beta_0)}{\gamma},  \tab    \gamma = (\gamma(\beta_1)+\gamma(\beta_0))/2
$$
