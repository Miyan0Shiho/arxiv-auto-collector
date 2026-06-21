# From Rags to Jeans: Axion Miniclusters from Early matter domination

**Authors**: Ariel Angulo, Paola Arias, Nicolás Bernal, Javier Redondo

**Published**: 2026-06-17 18:00:03

**PDF URL**: [https://arxiv.org/pdf/2606.19439v1](https://arxiv.org/pdf/2606.19439v1)

## Abstract
In an early matter-dominated era, density and temperature inhomogeneities of the radiation bath grow more efficiently than in the standard radiation-dominated history. If the axion mass depends on temperature, these inhomogeneities induce spatial fluctuations of the axion mass, providing a new source term for axion density perturbations. We show that this mechanism is most efficient when the reheating temperature lies just below the mass-saturation scale $T_Λ$, and can drive axion overdensities to order unity by matter--radiation equality. For the QCD axion saturating the observed dark matter abundance, the nonlinear spectrum at equality exhibits two characteristic regions: one associated with the gravitational enhancement already present in moduli-driven cosmologies, and another produced by the temperature dependence of the axion mass. We estimate the resulting minicluster masses and discuss the possible formation of axion miniclusters and axion-star substructure.

## Full Text


<!-- PDF content starts -->

From Rags to Jeans:
Axion Miniclusters
from Early matter domination
Ariel Angulo1, Paola Arias2,
Nicol´ as Bernal3and Javier Redondo4
1Departamento de F´ ısica, Universidad de Santiago de Chile,
Av. V´ ıctor Jara 3493, Estaci´ on Central, Santiago, Chile
2Departamento de F´ ısica, Universidad T´ ecnica Federico Santa Mar´ ıa,
Casilla 110-V, Avda. Espa˜ na 1680, Valpara´ ıso, Chile
3New York University Abu Dhabi
PO Box 129188, Saadiyat Island, Abu Dhabi, United Arab Emirates
4CAPA & Departamento de F´ ısica Te´ orica, Universidad de Zaragoza, 50009 Zaragoza, Spain
Abstract
In an early matter-dominated era, density and temperature inhomogeneities of the radia-
tion bath grow more efficiently than in the standard radiation-dominated history. If the axion
mass depends on temperature, these inhomogeneities induce spatial fluctuations of the axion
mass, providing a new source term for axion density perturbations. We show that this mech-
anism is most efficient when the reheating temperature lies just below the mass-saturation
scaleT Λ, and can drive axion overdensities to order unity by matter–radiation equality. For
the QCD axion saturating the observed dark matter abundance, the nonlinear spectrum at
equality exhibits two characteristic regions: one associated with the gravitational enhance-
ment already present in moduli-driven cosmologies, and another produced by the temperature
dependence of the axion mass. We estimate the resulting minicluster masses and discuss the
possible formation of axion miniclusters and axion-star substructure.arXiv:2606.19439v1  [hep-ph]  17 Jun 2026

Contents
1 Introduction 2
2 Early matter domination 4
3 The homogeneous axion 6
4 Perturbation growth during early matter domination 10
4.1 Radiation and metric perturbations . . . . . . . . . . . . . . . . . . . . . . . . . . . 10
4.2 Axion perturbations . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 13
4.3 Axion overdensity evolution . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 15
4.3.1 General case . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 15
4.3.2 The QCD axion case . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 19
5 Miniclusters 21
6 Conclusions 24
Appendices 25
A Perturbed fluid equations 26
B Analytical solutions 30
C Axion perturbations 32
D Jeans threshold scale 33
E Non-relativistic evolution to matter–radiation equality 34
F Isocurvature bound 37
1

1 Introduction
The nature of dark matter (DM) remains one of the central open questions in fundamental physics.
Among the many candidates proposed over the decades, bosonic pseudoscalar particles — axions
and axion-like particles (ALPs) — occupy a privileged position [1–4]: the QCD axion [5,6] arises as
the pseudo-Nambu–Goldstone boson of the Peccei–Quinn (PQ) symmetry [7–9], elegantly solving
the strong-CP problem, while string compactifications and other ultraviolet completions of the
Standard Model (SM) generically predict a landscape of additional pseudoscalar fields with axion-
like couplings, the so-called axiverse [10]. Throughout this work, we use the term “axion” to refer
to both the QCD axion and ALPs, unless a distinction is explicitly needed.
A defining feature of axion DM is its extreme lightness: with masses typically well below the eV
scale, axions are typically not produced thermally in sufficient abundance and instead rely on non-
thermal mechanisms. The best known among these is the vacuum misalignment mechanism [3,11],
in which the axion field is initially frozen at an arbitrary value by Hubble friction and, once its
mass becomes comparable to the expansion rate, begins coherent oscillations around the minimum
of its potential. The resulting condensate is highly monochromatic — oscillating at a frequency
set by the axion mass — and behaves as cold, pressureless matter, providing an excellent DM
candidate. Several variants and extensions of this mechanism have been suggested, including
kinetic misalignment [12,13], acoustic misalignment [14], frictional misalignment [15], and trapped
misalignment [16–18], all of which exploit different dynamical histories for the axion field prior to
the onset of oscillations.
A crucial ingredient in all misalignment-based production scenarios is the possible temperature
dependence of the axion mass. For the QCD axion, this dependence is well established: the
mass is generated by the topological susceptibility of QCD,χ(T), which is strongly suppressed
at temperaturesTabove the QCD crossover,T≳Λ QCD≃150 MeV, and approaches its zero-
temperature value only once non-perturbative effects fully develop. The steep power-law behavior
ma(T)∝T−bat high temperatures — withb≃4 from lattice simulations — profoundly affects
the onset of oscillations, the relic abundance, and the subsequent cosmological evolution of the
axion condensate.
Importantly, temperature-dependent masses are not exclusive to the QCD axion. ALPs arising
from the breaking of an approximate globalU(1) symmetry can acquire a mass through non-
perturbative, instanton-like effects in a hidden confining sector, in complete analogy with the
QCD mechanism. If the hidden sector has its own confinement scale Λ, the ALP mass will exhibit
a similar thermal suppression forT≳Λ, with the power-law indexband the scaleT Λdetermined
by the gauge group and matter content of the hidden sector rather than by QCD. This observation
2

considerably broadens the phenomenological relevance of a temperature-dependent axion mass:
the framework developed in this work applies to any pseudoscalar whose mass turns on as the
Universe cools through a confining phase transition, whether visible or hidden.
The temperature dependence of the mass has far-reaching consequences that go beyond the
background evolution of the axion field. At the level of cosmological perturbations, a time-varying
mass introduces qualitatively new effects. Fluctuations in the radiation temperature directly trans-
late into fluctuations of the axion mass,δm a∝(dm a/dT)δT, coupling the axion perturbation
equation to the radiation overdensity. This coupling acts as an additional source term for ax-
ion density perturbations — absent in the constant-mass case — that can significantly enhance
the growth of small-scale overdensities [19–22]. As we show in this work, the efficiency of this
mechanism depends sensitively on the interplay between the temperature scaleT Λat which the
mass saturates, the reheating temperatureT RH, and the comoving wavenumber of the perturbation
mode. A powerful observable consequence of these effects is the imprint left on the matter power
spectrum and on the possible formation of compact objects such as axion miniclusters [23–26].
The relic abundance and perturbation spectrum of the axion DM are also sensitive to the
expansion history of the Universe prior to Big Bang Nucleosynthesis (BBN) [27–29]. Recently, it
has been shown that adiabatic fluctuations cannot grow enough to form dense miniclusters during
radiation domination [19,20], despite the additional temperature-dependent source. Naturally, the
situation can be dramatically different in early cosmologies where adiabatic fluctuations are larger
at the relevant scales, for instance driven by a different cosmological expansion. Motivated by
these findings, in this paper we consider an early matter-dominated (EMD) era, as realized, for
example, in low-temperature reheating (LTR) scenarios. During this matter domination, density
and temperature fluctuations grow larger than they would during radiation domination, and, as
we shall see, they can lead toO(1) axion-DM fluctuations and, subsequently, to axion miniclusters
and axion stars. The EMD era is driven by a long-lived scalar fieldϕthat modifies both the
Hubble expansion rate and the SM temperature evolution, relative to the standard radiation-
dominated picture. In minimal scenariosϕcould be identified with the inflaton during cosmic
reheating [30–36], but could also be a moduli field from string compactification [37–40].
In this paper, we study the growth of axion perturbations during an EMD era with low reheating
temperature, paying special attention to the role of a temperature-dependent axion mass. We work
within the framework of the standard (vacuum) misalignment mechanism [3, 11] and consider a
general parameterization of the mass,m a(T) =m a,0(TΛ/T)bforT > T Λ, which encompasses ALPs
with hidden-sector confinement at arbitrary scales. For the QCD axion, we use the topological
susceptibility obtained from lattice results [41]. Rather than adopting a fluid description for the
3

axion — which becomes technically cumbersome when the mass itself carries perturbations — we
solve the perturbed Klein–Gordon equation directly, using as input the background and perturbed
solutions for the decaying scalar field and radiation obtained from the coupled Boltzmann–Einstein
system. This approach allows us to capture the full dynamics of the axion field through the onset
of oscillations and across the reheating transition, including the new source term induced by the
temperature-dependent mass.
The paper is organized as follows. Section 2 describes the EMD phase and the resulting
modifications to the cosmological background. In Section 3, we review the background solutions for
the axion field, including approximate analytical expressions for both constant and temperature-
dependent masses. In Section 4, we derive the perturbed equations for the axion field in the
presence of a temperature-dependent mass and discuss the structure of the source terms. Section 4
also presents our numerical results for the growth of axion perturbations, exploring the dependence
on the mass of the axion, the temperature scaleT Λ, and the reheating temperature. In Section 5
we discuss the formation of axion miniclusters. We also apply our framework to the QCD axion.
We conclude in Section 6.
2 Early matter domination
Before the onset of BBN, the energy density of the Universe is not expected to always be dominated
by SM radiation. In fact, it may be governed by a different component, often modeled as a coher-
ently oscillating scalar field, whose energy density redshifts as non-relativistic matter. This leads
to an EMD era that departs from the conventional radiation-dominated thermal history [34, 35].
Such a phase can have a significant impact on early-Universe processes, including DM produc-
tion and the evolution of cosmological perturbations, thereby modifying standard cosmological
assumptions [25–29,42–52].
In this work, we focus on the time during which the non-relativistic scalar fieldϕdecays into
SM radiation while dominating the energy density of the Universe until the end of the EMD era.
The background dynamics is described by the Boltzmann equations [27]
dρϕ
dt+ 3H ρ ϕ=−Γ ϕρϕ,(1)
dρr
dt+ 4H ρ r= + Γ ϕρϕ,(2)
whereρ ϕandρ rdenote the energy densities ofϕand SM radiation, respectively, and Γ ϕis the
decay rate ofϕ. The Hubble expansion rateHis given by
H2=ρϕ+ρr
3M2
Pl,(3)
4

whereM Pl≃2.4×1018GeV is the reduced Planck mass, and the radiation energy density is
related to the SM temperatureTthrough
ρr(T) =π2
30g⋆(T)T4,(4)
withg ⋆(T) corresponding to the relativistic degrees of freedom that contribute toρ r. It is con-
venient to rewrite Eqs. (1) and (2) as a function of the cosmic scale factorRand the comoving
variablesρ ϕR3andρ rR4as
d(ρϕR3)
dR=−Γϕ
H R(ρϕR3),(5)
d(ρrR4)
dR= +Γϕ
H(ρϕR3).(6)
ForR≤R RH, the energy density of the Universe is dominated byϕ, inducing EMD cosmology.
The scale factorR=R RHcorresponds to the end of the EMD and then to the onset of the
standard radiation-dominated phase. The reheating temperatureT RH≡T(R RH) is implicitly
defined by the equalityρ ϕ(TRH) =ρ r(TRH). Although we determine it numerically,T RHcan be
analytically estimated up to order-one factors throughH RH≡H(T RH)≃Γ ϕ, which leads to
T2
RH≃3
πs
10
g⋆(TRH)MPlΓϕ.(7)
Consistency with BBN requires the reheating temperature to satisfyT RH≳T BBN≃4 MeV [53–56].
An estimate of the solution of the set of Eqs. (5) and (6) can be made analytically, in the regimes
whereϕdominates or is subdominant. In these two cases, the Hubble rate evolves as
H(R)≃

HRHRRH
R3/2
forR ini≤R≤R RH,
HRHRRH
R2
forR RH≤R .(8)
In addition, one finds that the temperature evolves approximately as [27]
T(R)≃

TRHRRH
R3/8
forR ini≤R≤R RH,
TRHRRH
R
forR RH≤R ,(9)
reflecting the continuous injection of entropy fromϕdecays into the visible sector during the
EMD era. In fact, the decay of the scalar field injects entropy into the radiation bath, diluting any
preexisting relic abundances. The left panel of Fig. 1 shows the evolution of theϕand SM radiation
5

10−410−310−210−1100101102
R/R RH10−1410−1010−610−21021061010ρi[GeV4]
R=RRH
ρr
ρφ
10−410−310−210−1100101102
R/R RH10−1100101102103104T(R) [MeV]T=TRH
R=RRHFigure 1: Evolution of the energy densities of theϕand the SM radiation (left) and the corre-
sponding SM temperature (right), as a function of the cosmic scale factorR, forT RH= 15 MeV.
energy densities as a function of the scale factor forT RH= 15 MeV, while the right panel shows
the corresponding evolution of the SM temperature. During the EMD era, the energy density ofϕ
scales as non-relativistic matter (that is,ρ ϕ(R)∝R−3), while the energy density of the radiation
scales asρ r(R)∝R−3/2due to the source term. After the end of the EMD era (whenR=R RH),ϕ
decays exponentially fast and SM radiation scales as free radiationρ r(R)∝R−4. In the upcoming
analysis, we assume that the maximum temperature reached by the thermal bath is much larger
than the relevant scale for axion DM production, so that the phenomenology is independent of the
cosmic initial conditions. Therefore, the only relevant parameter is Γ ϕor, equivalently,T RH.
3 The homogeneous axion
We will focus on the production of axion DM via the standard misalignment mechanism, assuming
a pre-inflationary PQ breaking scenario.1
Axions are initially frozen at their misalignment value due to Hubble friction in the expanding
Universe. As the Hubble rate drops below the axion mass, the field begins coherent oscillations
around the minimum of its potential, acting as a cold DM condensate. We focus on the scenario in
which axions start oscillating during the EMD era, while their mass is still temperature dependent.
In this case, we consider the regimeR osc< R RHandR osc< R Λ, whereR oscandR Λdenote the scale
factors at the onset of oscillations and at the time when the axion reaches its zero-temperature
mass, respectively. We now discuss the main features of the background solution in this regime.
After spontaneous breaking of the global symmetry, the axion sector is described as a minimally
1The corresponding isocurvature constraint on the inflationary scale is discussed in Appendix F.
6

coupled real scalar fielda=a(t, ⃗ x) with action
Sa=Z
d4x√−g
−1
2(∂a)2−V a(a)
,(10)
with
Va(a)≡m2
a(t)f2
a
1−cosa
fa
,(11)
wheref ais the scale at which the PQ symmetry is spontaneously broken andm a(t) is the time-
dependent axion mass.
For the QCD axion,f adetermines the axion mass through the topological susceptibility of
QCD,χ(T), as
m2
a(T) =χ(T)
f2
a,(12)
where from lattice QCD simulations has been found a zero-temperature value ofχ(0)≃0.0245 fm−4,
in the symmetric isospin case [41]. Therefore, the zero-temperature massm a,0≡m a(T→0) of
the QCD axion is given by
ma,0≃5.69 meV109GeV
fa
.(13)
Generally speaking, ALPs2could also feature a temperature dependent mass if the hidden
sector breaks non-perturbatively the extraU(1) global symmetry associated with the axion origin.
Therefore, for analytical estimations it is useful to consider an approximate expression, which has
to be cut off by hand once the mass reaches its zero-temperature value
ma(T) =m a,0×

(TΛ/T)bforT≥T Λ,
1 forT≤T Λ.(14)
For hidden-sector ALPs,Tshould be understood as the temperature of the sector responsible for
generating the axion mass; in the phenomenological analysis below, we identify this temperature
with the SM radiation temperature or assume that the two are proportional. From now on,
when considering generic ALPs, we will use the prescription of Eq. (14), consideringb= 4 as a
benchmark. Specifically referring to the QCD axion, we will adopt the mass parameterization of
Eq. (13), which follows the lattice results from Ref. [41]. During the EMD era,m a(R)∝R3b/8, as
shown in the left panel of Fig. 2.
The equation of motion that follows from Eq. (10) is
¨a+ 3H˙a−∇2a
R2+m2
afasina
fa
= 0,(15)
2We will loosely call axions to QCD axions and ALPs, and only make the distinction when necessary.
7

101102T[MeV]
10−410−310−210−1100101
R/R RH10−1610−1410−1210−1010−810−6ma(R) [eV]
R=RΛH(R)
b= 0
b= 4101102T[MeV]
10−410−310−210−1100101
R/R RH−0.20.00.20.40.60.8θ0(R)
R=R(b=0)
osc
R=R(b=4)
osc101102T[MeV]
10−410−310−210−1100101
R/R RH10−4810−4510−4210−3910−3610−33ρa/f2
a[GeV2]
R=R(b=0)
osc
R=R(b=4)
osc
R=RΛFigure 2: Evolution of the axion massm aand the Hubble parameterH(left), the normalized
homogeneous axion fieldθ 0(center), and the energy density of the axion (right), as functions of
the scale factorR, forb= 0 (blue), andb= 4 (red). We assumedT RH= 15 MeV,m a,0= 10−8eV,
TΛ= 25 MeV, andθ ini= 1.
where the dots (˙) denote derivatives with respect to timet.
In the pre-inflationary scenario, the axion field is homogeneous with tiny inhomogeneous per-
turbations
θ(t, ⃗ x)≡a(t, ⃗ x)
fa=θ0(t) +δθ(t, ⃗ x) withδθ(t, ⃗ x)≡Zd3k
(2π)3δθk(t)ei⃗k·⃗ x.(16)
The homogeneous fieldθ 0obeys
¨θ0+ 3H ˙θ0+m2
a(t) sin(θ 0) = 0.(17)
For a temperature-dependent mass, an approximate WKB analytical solution can be obtained
under the adiabatic condition,˙ma
m2a≪1, and assuming that the field is already oscillating, i.e.,
ma(t)≫H(t). In this regime, the solution can be written as
θ0(R)≃θ iniRosc
R3/2s
ma(Tosc)
ma(T)cos (ψ(t)),(18)
whereT osc≡T(R osc),θini≡θ 0(T≫T osc) is the initial misalignment angle of the field, and the
phase is defined by
ψ(t)≡Zt
toscdt′ma(t′).(19)
The temperatureT oscat which the axions start to oscillate can be estimated by the equality
2H(T osc) =m a(Tosc), and corresponds to
T4+b
osc≃45
2π2g⋆(TRH)1/2
ma,0MPlTb
ΛT2
RH.(20)
8

Interestingly, the conditions for oscillations to start during the EMD era,T RH< T osc, and during
the period where the mass is time dependent,T Λ< T osc, translate, respectively, into
ma,0
HRH≳TRH
TΛb
,(21)
ma,0
HRH≳TΛ
TRH4
.(22)
In the following analysis, both conditions are always imposed; however, the hierarchy betweenT RH
andT Λis not yet fixed. In addition, the scale factorR osc≡R(T osc) normalized to the scale factor
R0at present is
Rosc
R0=Rosc
RRHRRH
R0≃TRH
Tosc8/3g⋆s(T0)
g⋆s(TRH)1/3T0
TRH.(23)
The numerical solution of Eq. (17) is shown in the central panel of Fig. 2. For higher values of
b, the oscillations are delayed, have a higher frequency, and a faster decay of the envelope. This
behavior can be understood from the WKB solution in Eq. (18), which yieldsψ(R)∝R(3b+12)/8
andθ 0(R)∝R−3(8+b)/16. In particular, forb= 4 the amplitude decays asR−9/4, compared to the
standard scalingR−3/2in the constant-mass case. As a consequence, the onset of oscillations is
delayed forb >0 relative to the constant-mass case (b= 0), as shown in the left panel of Fig. 2.
In the limita/f a≪1, the energy densityρ aand the pressurep aof the zeroth mode are given
by
ρa≃1
2f2
a
˙θ2
0+m2
a(T)θ2
0
,(24)
pa≃1
2f2
a
˙θ2
0−m2
a(T)θ2
0
.(25)
The time average of the energy density gives⟨ρ a⟩ ≃1
2f2
am2
a(T)A2
θ, whereA θdenotes the oscillation
envelope ofθ 0, which leads to
⟨ρa(R)⟩ ∝Rosc
R24−3b
8
.(26)
The evolution of the axion energy density forb= 0 (blue line) andb= 4 (red line) is shown in the
right panel of Fig. 2. Once the mass reaches a constant value atR=R Λ(yellow dot-dashed line),
the energy density starts to redshift as non-relativistic matter,ρ a(R)∝R−3.
From the adiabatic conservation of the axion number, the axion energy density at present,
T=T 0, is
ρa(T0) =ρ a(Rosc)ma,0
ma(Tosc)Rosc
R03
=θ2
inif2
aT3
0
2M Plg⋆s(T0)
g⋆s(TRH)2π2g⋆(TRH)
458+b
2(4+b) 
mb
a,0T4+3b
RH
M4
PlT4b
Λ! 1
4+b
.(27)
9

In particular, forb= 4 the axion relic abundance is given by
Ωah2≡ρa(T0)
ρch2≃0.12ma,0
10−6eV1
2TRH
TΛ2faθini
1.2×1013GeV210.75
g⋆s(TRH)1
4
,(28)
whereρ c/h2≃1.054×10−5GeV/cm3, and the total DM relic abundance measured by Planck is
ΩDMh2≃0.12 [57]. The scalef acould be tuned to fit the entire observed axion DM abundance
in the case of ALPs, while for QCD axions, relation (13) should be used. Smaller values for the
energy density are also viable in the case of multi-component DM scenarios.
4 Perturbation growth during early matter domination
Having established the background expansion of the Universe and the evolution of the homogeneous
axion field, we now turn to the study of the perturbations of the different components during the
EMD era.
4.1 Radiation and metric perturbations
To study the evolution of perturbations, we need to construct a set of perturbed equations for each
of the relevant components; a careful derivation of the equations can be found in the Appendix A.
We start defining the perturbed metric in the Newtonian gauge,
ds2=−(1 + 2 Φ)dt2+R2(t) (1−2 Ψ)δ ijdxidxj,(29)
where Φ and Ψ are the Newtonian potential and the spatial curvature perturbation, respec-
tively [58]. In the following, we neglect the anisotropic stress, so we set Ψ = Φ. Secondly, we
introduce the energy overdensity and velocity divergence for each fluid componentσ(withσ=r
for the SM radiation orσ=ϕ), as
δσ≡δρσ
ρσ, θ σ≡1
R∇2vσ,(30)
whereδρ σdenotes the first-order perturbation of the energy density andv σis the corresponding
velocity potential. Density and velocity perturbations are governed by the Einstein equations and
the covariant conservation of the stress–energy tensor in a perturbed FLRW space-time. There is
energy exchange betweenϕand the visible sector, and therefore
∇µT(σ)µ
ν=Q(σ)
ν,(31)
whereT(σ)
µνis the stress–energy tensor of theσfluid andQ(σ)
νencodes the energy–momentum
transfer between components. As a result, the stress–energy tensor of each component is not
10

conserved. However, the sum over all species satisfies the covariant conservation law of the total
energy–momentum tensorX
σ=ϕ,r∇µT(σ)µ
ν= 0.(32)
The interaction terms are given by [59]
Q(ϕ)
ν= Γ ϕT(ϕ)
µνuµ,(33)
Q(r)
ν=−Q(ϕ)
ν.(34)
These terms can be expanded at linear order, and their first-order perturbations are given by
δQ(ϕ)
0= +Γ ϕρϕ(δϕ+ Φ),(35)
δQ(ϕ)
i=−Γ ϕρϕ∂ivϕ.(36)
With all of the above, we obtain the set of perturbed equations forϕand the SM radiation:
δ′
ϕ=−Γϕ
R HΦ−1
R2Hθϕ+ 3 Φ′,(37)
δ′
r=ρϕ
ρrΓϕ
R H(δϕ−δr+ Φ)−4
31
R2Hθr+ 4 Φ′,(38)
θ′
ϕ=k2
R2HΦ−1
Rθϕ,(39)
θ′
r=ρϕ
ρrΓϕ
R H3
4θϕ−θr
+k2
R2Hδr
4+ Φ
,(40)
Φ′=−1
6M2
PlR H2(ρϕδϕ+ρrδr+ρaδa) +k2
3R3H2+1
R
Φ
.(41)
We note that the axion appears in the Hubble parameter and in the Poisson Eq. (41), but its
contribution is small and can be safely neglected. Thus, our strategy is to solve the above set
of equations ignoring the axion contribution and later use it as input to solve for its evolution.
The initial conditions used areθ ϕ(Rini) =θ r(Rini) = 0,δ ϕ(Rini) =−2 Φ p,δr(Rini) =−Φ p, and
Φ(R ini) = Φ p≡√AswithA s≃2.101×10−9being the measured amplitude of the primordial
scalar power spectrum [57]. Even if in the following a full numerical treatment is presented, in
Appendix B analytical solutions for the evolution of the perturbations are presented.
Figure 3 shows the evolution of the absolute value of the overdensities forϕ(top) and the SM
radiation (bottom) as a function of the scale factor (left) and the mode (right), where we have
conveniently introduced a set of dimensionless quantities, defined as
x≡R
RRH, κ≡k
kRH,withk RH≡R RHHRH.(42)
11

10−310−210−1100101102
x10−610−510−410−310−210−1100|δφ(κ,x)|
|δφ|= 2Φ pκ= 100.0
κ= 10
κ= 1
κ= 0.1
10−1100101102
κ10−610−510−410−310−210−1100|δφ(κ,x)|
|δφ|= 2Φ px= 10.0
x= 1.0
x= 0.1
10−310−210−1100101102
x10−610−510−410−310−210−1100|δr(κ,x)|
|δr|= 24Φ p
|δr|= Φ pκ= 100.0
κ= 10
κ= 1
κ= 0.1
10−1100101102
κ10−610−510−410−310−210−1100|δr(κ,x)|
|δr|= 24Φ p
|δr|= Φ px= 10.0
x= 1.0
x= 0.1Figure 3: Evolution of the absolute value of the overdensities forϕ(top) and the SM radiation
(bottom) as a function of the scale factor (left) and the mode (right).
It is interesting to note that the choice of these variables makes the figure independent ofT RH.
During the EMD era (that is,x <1), modes for theϕoverdensity withκ >1 enter the horizon
before reheating and grow linearly as the Universe is dominated by non-relativistic matter. It is
interesting to note that the nonlinear regime for theϕoverdensity (that is,|δ ϕ| ≥1) can only
be reached for modes corresponding toκ≳p
3/(2 Φ p)≃181; see Appendix B. After the EMD
era (x >1) the growth is only logarithmic since the Universe has transitioned into a radiation-
dominated era. The situation is different for radiation overdensities as they have a non-vanishing
pressure, which implies that their linear growth during the EMD era saturates once it reaches
an enhancement of order 24, see Appendix B. After the EMD era,|δ r|oscillates with a constant
amplitude.
Finally, the evolution of the absolute value of the gravitational potential Φ is shown in Fig. 4.
Even if during the EMD era it remains constant, after the EMD era it rapidly decreases through
damped oscillations due to the progressive conversion ofϕinto SM radiation.
12

10−310−210−1100101102
x10−910−810−710−610−510−410−3|Φ(κ,x)||Φ|= Φ p
κ= 100.0
κ= 10
κ= 1
κ= 0.1
10−1100101102
κ10−910−810−710−610−510−410−3|Φ(κ,x)||Φ|= Φ p
x= 10.0
x= 1.0
x= 0.1Figure 4: The evolution of the absolute value of the gravitational potential Φ.
4.2 Axion perturbations
We now turn to the evolution of axion perturbations within the framework described above. In
principle, one could describe the axion sector as a perturbed fluid. However, we aim to include
the temperature-dependent mass in Eq. (14), which induces an explicit energy exchange between
radiation and axions, since the mass itself is perturbed:
m2
a(T) =m2
a(T) +δm2
a withδm2
a=dm2
a
dTδTk,(43)
whereδT kis the linear temperature perturbation with momentumkand Tis the background pho-
ton temperature. This makes the fluid description cumbersome, as the perturbed mass introduces
a direct coupling between the axion and radiation perturbations and breaks the usual conservation
structure. Therefore, we follow the axion perturbations directly from its field equations. These
will be solved using the solution of Eqs. (37) to (41) as input.
We start by considering a perturbation of the axion field, parameterized as the evolution of the
misalignment angle
θ(t, ⃗ x) =θ 0(t) +δθ(t, ⃗ x) withδθ(t, ⃗ x) =Zd3k
(2π)3δθk(t)ei⃗k·⃗ x,(44)
whereθ 0(t) corresponds to the homogeneous solution discussed in the previous section, andδθ k(t)
is the linear perturbation with momentumk. The equation of motion for the perturbation follows
from the variation of the action in Eq. (10), and is given by
d2δθk
dt2+ 3Hdδθk
dt+k2
R2δθk+m2
acosθ 0δθk= 4˙Φ˙θ0−2Φm2
asinθ 0−1
4dm2
a
dTsinθ 0T δr,(45)
13

where we have usedδ r= 4δT/ T(ignoring the change in the effective relativistic degrees of free-
dom); see Appendix C. Equation (45) can be rewritten as
d2δθκ
dx2+dln(H x)
dx+3
xdδθκ
dx+"
κ2
x4HRH
H2
+ma(x)
H x2
cosθ 0(x)#
δθκ=S 1+S 2+S 3,(46)
where the source terms are given by
S1≡4dΦ
dxdθ0
dx,(47)
S2≡ −2 Φma(x)
H(x)x2
sinθ 0(x),(48)
S3≡ −dlnm2
a
dlnTδr
4ma(x)
H(x)x2
sinθ 0(x)≃

b δr
2ma(x)
H(x)x2
sinθ 0(x) forx < x Λ,
0 forx > x Λ,(49)
andx Λ≡R Λ/RRH. We note that for a constant axion mass, the source termS 3vanishes. In the
following, Eq. (46) will be numerically solved using the initial conditionsδθ k(Rini) =δθ′
k(Rini) = 0,
since we focus on the adiabatic solution for the axion perturbation, neglecting the subdominant
isocurvature mode (cf Appendix C).3
To solve the perturbed axion equations we proceed as follows: First, we compute the background
evolution ofϕand radiation in Eqs. (5) and (6). We then use this solution to determine the
background evolution of the axion zeroth mode, Eq. (17). Next, we solve for the evolution of
the metric and radiation perturbations using Eqs. (37) to (41), and finally compute the axion
perturbations following Eq. (46).
The evolution of the three source terms for the QCD axion case is illustrated in Fig. 5, forT RH=
100 MeV,m a,0∼2.2×10−7eV,T Λ= 150 MeV andθ ini= 1. During the EMD era, the gravitational
potential is nearly constant and thereforeS 1is very suppressed and subdominant with respect toS 2
andS 3. The last two sources have a similar scaling and their ratio|S 3/S2|=b/4×|δ r/Φ| → |δ r/Φ|.
This implies that for superhorizon modes|S 3/S2| ≃1, while for subhorizon modes|S 3/S2| ≃24.
S2only dominates overS 3in the case where the mass of the axion is constant, sinceS 3= 0.
Additionally, the growth of these sources is driven by the common factor (m a/(H x))2∝x(3b+4)/4,
which corresponds tox4orx1if the axion mass evolves or not, respectively.
3We keep the initial misalignment angle away fromθ ini∼πto avoid parametric growth of the axion fluctua-
tions [60]; the so-called ‘large misalignment’ scenario. Although this amplification could, in principle, occur within
our framework, our primary focus remains on isolating the impact of the mass coupling to radiation.
14

10 310 210 1100101
x10 1110 810 510 2101104|Si(κ,x)|
xosc
xΛ
xRHκ= 100
|S1|
|S2|
|S3|
10 310 210 1100101
xxosc
xΛ
xRHκ= 10
10 310 210 1100101
xxosc
xΛ
xRHκ= 1Figure 5: Source termsS 1,S2, andS 3as a function ofxfor different values ofκ, assuming
TRH= 100 MeV,m a,0= 2.2×10−7eV,T Λ= 150 MeV, andθ ini= 1. We consider three
representative modes withκ= 100,κ= 10, andκ= 1. We show the envelope of the functions to
avoid the noise from oscillations.
4.3 Axion overdensity evolution
The axion overdensity is defined asδ a≡δρ a/ρa, whereδρ adenotes the fluctuation of the axion
energy density. We first expressδρ ain terms of the perturbed field as (see Appendix C)
δρa=f2
a
˙θ0δ˙θk−Φ˙θ2
0+m2
asinθ 0δθk−b
2m2
a(1−cosθ 0)δr
.(50)
Although the axion density is given by Eq. (24), its overdensity is
δa=˙θ0δ˙θk−Φ˙θ2
0+m2
asinθ 0δθk−b
2δrm2
a(1−cosθ 0)
1
2˙θ2
0+m2
a(1−cosθ 0),(51)
forT > T Λ. ForT < T Λ, the mass has saturated and the last term in the numerator should be
dropped. In the following, the general case will be studied, followed by the specific case of the
QCD axion.
4.3.1 General case
In Fig. 6 we show the absolute value of the axion overdensity forT RH= 15 MeV,m a,0= 10−8eV,
TΛ= 25 MeV,b= 4, andθ ini= 1. The left panel shows|δ a|as a function of the normalized scale
factorxfor several values ofκ, while the right panel shows it as a function of the comoving mode
κfor different values ofx. Note that before the onset of axion oscillations ˙θ0≃0, and therefore
Eq. (51) reduces to
δa≃2δθk
θ0−b
2δr≃ −b
2δr.(52)
Consequently, modes that are subhorizon at the onset of axion oscillations reach a maximum
amplitude set by the maximum amplitude ofδ r. As seen in Fig. 3, the largest enhancement
15

10−510−410−310−210−1100101
x10−610−510−410−310−210−1100101|δa(κ,x)|
|δa|= 2Φ p|δa|= 48Φ p
xosc
xΛκ= 100
κ= 50
κ=κosc
κ= 1
10−1100101102
κ10−610−510−410−310−210−1100101|δa(κ,x)|
|δa|= 2Φ p|δa|= 48Φ p
κJκoscx= 10
x= 1
x=xoscFigure 6: Evolution of the absolute value of the axion overdensity forT RH= 15 MeV,m a,0=
10−8eV,T Λ= 25 MeV,b= 4 andθ ini= 1, as a function ofx(left) orκ(right).
corresponds to modes that entered the horizon well before oscillations begin (κ≫1) and is given
by|δ a|= 12bΦ p. In contrast, modes that are superhorizon at reheating (κ≪1) remain frozen,
so the overdensity stays close to|δ a| ≃b
2Φp. Afterx osc,|δa|inherits a transient oscillation from
the background fieldθ 0(t) through the trigonometric factors in Eq. (51), which damp out as the
oscillation amplitude decreases and the field enters the small-angle regime, where the potential
becomes quadratic and these factors reduce to a commonθ2
0(t) envelope shared by numerator and
denominator. The right panel of Fig. 6 shows that modes entering the horizon before the onset
of oscillations (κ > k osc) benefit the most from the energy transfer with radiation and the non-
standard cosmological evolution, with the enhancement growing monotonically withκ. This trend
does not extend indefinitely, since forκ≳κ J(dashed line) the quantum pressure dominates and
suppresses the growth of the overdensity. The Jeans threshold of modes that entered the horizon
before the axion mass became constant is given by
κJ≃rma,0
HRHx1/4
Λ≃91ma,0
10−8eV10−21GeV
HRH1/2xΛ
0.71/4
,(53)
see Appendix D for the detailed derivation. The modes withκ≳κ Jare therefore suppressed after
xΛ, in good agreement with our numerical results.
In Fig. 7, we examine the impact of varying the temperatureT RH(left panels) and the mass
ma,0(right panels) on the overdensity of the axion, forT Λ= 25 MeV,b= 4, andθ ini= 1. In
the left panels, where the mass is fixed atm a,0= 10−8eV, we find that the overdensity of the
16

10−510−410−310−210−1100101
x10−610−510−410−310−210−1100101|δa(κ= 80,x)|
|δa|= 2Φ p|δa|= 48Φ pTRH= 10 MeV
TRH= 20 MeV
TRH= 50 MeV
xΛ
10−510−410−310−210−1100101
x10−610−510−410−310−210−1100101|δa(κ= 80,x)|
|δa|= 2Φ p|δa|= 48Φ pma,0= 10−9eV
ma,0= 10−8eV
ma,0= 10−7eV
xosc
10−1100101102
κ10−610−510−410−310−210−1100101|δa(κ,x= 10)|TRH= 10 MeV
TRH= 20 MeV
TRH= 50 MeV
κJ
10−1100101102
κ10−610−510−410−310−210−1100101|δa(κ,x= 10)|ma,0= 10−9eV
ma,0= 10−8eV
ma,0= 10−7eV
κJFigure 7: Axion overdensity|δ a|for differentT RH(left panels) andm a,0(right panels), withT Λ=
25 MeV andb= 4. The left panels fixm a,0= 10−8eV, while the right panels fixT RH= 15 MeV.
The top panels show the evolution as a function ofxforκ= 80, and the bottom panels the spectra
as a function ofκatx= 10. The Jeans scaleκ Jis indicated by vertical dashed lines.
axion is maximized for large modesκ(but smaller thanκ J) whenT RHlies slightly belowT Λ(e.g.
TRH= 20 MeV), ensuring that the temperature dependence of the mass ceases before the end of
the EMD era. ForT RH= 10 MeV, oscillations begin earlier, so the sourceS 3acts over a shorter
interval and its enhancement is only partially imprinted on the perturbation, yielding a reduced
amplitude (top-left panel). The earlier transition to a temperature-independent mass further
suppresses the effective duration ofS 3. WhenT Λ< T RH(e.g.T RH= 50 MeV), the overdensity
tracks theδ rplateau throughout EMD but is suppressed near reheating, whereδ rdecays sharply
for large modes, yielding only a mild net enhancement. In this regime, the modes retain the large
amplitude of the radiation overdensity characteristic ofκ≫1, while avoiding the suppression that
arises near reheating. Finally, in the right panel of Fig. 7, we fixT RH= 15 MeV and varym a,0.
Larger masses (m a,0= 10−8eV andm a,0= 10−7eV) exhibit the strongest enhancements at high
17

10−1100101102
κ10−510−410−310−210−1100101|δa(κ,x= 10)|b= 4,NSC
b= 4,SC
κosc,NSC
κJ,NSC
10−1100101102
κ10−1100101102103104|δ(b=4)
a,NSC|/|δ(b=4)
a,SC|x= 10
κ(b=4)
J,NSC
κosc,NSC
10−1100101102
κ10−510−410−310−210−1100101|δa(κ,x= 10)|b= 0,NSC
b= 0,SC
κosc,NSC
κJ,NSC
10−1100101102
κ10−1100101102103104|δ(b=0)
a,NSC|/|δ(b=0)
a,SC|x= 10
κ(b=0)
J,NSC
κosc,NSCFigure 8: Axion overdensity spectrum in the cases where the reheating temperature is low (T RH=
15 MeV) and high reheating temperature (T RH= 8 TeV) (left panels), and their ratio (right
panels). The top (bottom) panels correspond tob= 4 (b= 0), form a,0= 10−8eV,T Λ= 25 MeV,
θini= 1, andx= 10.
κ, which in fact reach the nonlinear regime aroundκ≳200. As the axion mass decreases, the
amplification is reduced, since higher-κmodes become pressure suppressed. The smaller the mass,
the lower the threshold wavenumber at which this suppression begins, as shown in Eq. (53). In
summary, we find that the optimal scenario for maximizing axion overdensity growth is achieved
whenT Λis slightly above but close toT RH, and for large masses of axions.
In order to understand the effect of the EMD era, in Fig. 8 we show the absolute value of the
axion overdensity forb= 4 (top) andb= 0 (bottom) for the cases where the reheating temperature
is high or low (left) and the ratio between them (right), form a,0= 10−8eV,T Λ= 25 MeV,θ ini= 1,
andx= 10. For the case with low-reheating temperature we assumeT RH= 15 MeV, while for the
case with high-reheating temperature we takeT RH= 8 TeV; however, the dynamics of the axion
becomes independent of the reheating temperature as long asT RH≳10 GeV. It is interesting to
18

30 40 60
TRH[MeV]10−910−810−7ma,0[eV]κ=κJ
|δa|= 1
30 40 60
TRH[MeV]κ= 0.5κJ
10−210−1110
|δa(κ,x= 10)|Figure 9: Absolute value of the ALP overdensityδ aas a function of the reheating temperature
TRHand the zero-temperature axion massm a,0forκ=κ Jandκ= 0.5κ J, respectively. We have
fixedθ ini= 1,T Λ= 40 MeV, andb= 4. All the parameter space satisfies the correct DM relic
abundance.
note that subhorizon modes smaller than the Jeans scale at the onset of oscillations benefit from
a large enhancement due to the early-matter domination. Forb= 4 the enhancement is typically
of order 102, while forb= 0 it could reach 103.
Finally, in Fig. 9 we show a gradient map for the ALP overdensity as a function of the reheating
temperature and the axion zero-temperature mass, for two different mode scales:κ=κ J(left) and
κ= 0.5κ J(right). As expected, the highest enhancement of the overdensity is obtained for the
highest non-suppressed mode, which is the one set by the Jeans threshold of Eq. (53), large values
for the axion mass, andT RH≲T Λ. This trend has also been observed in Ref. [61], and is due to the
source termS 2that bounds the ALP overdensity with the overdensity of the fieldϕ. However, it
is interesting to note that modesκ= 0.5k Jreceive comparable enhancements from theS 3source
term. The range of mass that can be enhanced is highly dependent on the temperature scaleT Λ.
Smaller masses can be efficiently enhanced ifT Λis slightly above the smallest allowed reheating
temperature, i.e.,T BBN.
4.3.2 The QCD axion case
For the QCD axion, the confinement scale is fixed atT Λ=T QCD≃150 MeV. Reaching the
nonlinear regime after the EMD era thus requires the optimal hierarchyT Λ≳T RHidentified
19

100101102
κ10−810−7ma,0[eV]|δa|= 1
κJ
κosc
κΛ
10−410−310−210−1110
|δa(κ,x eq)|
5102050100200300
TRH[MeV]Figure 10: Absolute value of the QCD axion overdensityδ aas a function of the comoving wavenum-
berκand the zero-temperature axion massm a,0, forT Λ= 150 MeV, andθ ini= 1, evaluated at the
MRE. The green dashed lines indicate the Jeans scaleκ Jat equality, while the black dashed line
denotes where the spectrum becomes nonlinear.
above, which selectsT RHin the rangeT Λ/TRH∼1.1–2,i.e.T RHbetween 75 MeV and 130 MeV.
Imposing Ω ah2= Ω DMh2in Eq. (28) together with the QCD relation in Eq. (13) fixes the axion
mass for a givenθ inito be
ma,0≃3.58×10−7eV 
g⋆(TRH)1
2
g⋆,s(TRH)2
3!TRH
150 MeV4/3
θ4/3
ini.(54)
Therefore, the masses that receive the greatest enhancement of their overdensity perturbations are
in the range 1.4×10−7eV≲m a,0≲3.2×10−7eV, forθ ini= 1. For these masses, the overdensity
enters the nonlinear regime at or near the EMD era. However, higher and lower axion masses can
still reach nonlinearity at or before matter-radiation equality (MRE).
In Fig. 10 we show the QCD axion overdensityδ aas a function ofm a,0(or, equivalently,T RH,
cf. Eq. (54)) and the reduced wavenumberκ=k/k RH, evaluated at the MRE scale factorx eq. The
thick black line marks the nonlinearity threshold|δ a|= 1, while the green dashed line indicates the
Jeans threshold of Eq. (53). Two maxima emerge: a global maximum aroundm a,0∼5×10−9eV
corresponding to the lowest cosmologically allowed reheating temperaturesT RH≃T BBNand a local
maximum aroundm a,0∼3×10−7eV (T RH∼100 MeV).
20

The global maximum corresponds to axion perturbations sourced by the density fluctuations
ofϕ, when axions fall in its gravitational potential. The competition between this gravitational
growth and the axion quantum pressure produces a peak in the axion power spectrum at a scale
that coincides with the axion quantum Jeans scale at MRE. It leads to the formation of axion stars,
as recently explored in Ref. [61]. In our framework, this corresponds to the gravitational source
termS 2. In contrast, the local maximum corresponds to a novel effect explored in this work: the
temperature dependence of the axion mass couples radiation overdensities toδ a, producing a source-
driven enhancement whenT RHlies just belowT Λ. The nonlinearity starts at smaller comoving
wavenumbersκ∼30, which leaves distinctive imprints on subsequent structure formation. In the
next subsection, we explore further the consequences on axion minicluster formation.
5 Miniclusters
The growth of axion density perturbations during the EMD era ceases around the end of reheating,
when the fieldϕdriving the matter-dominated phase decays. Fluctuations in both the radiation
and axion DM fluids are released from the deep gravitational potential wells generated on small
scales and begin to free-stream [25]. Thereafter, axion density perturbations evolve as standard
non-relativistic matter fluctuations in a radiation-dominated Universe, growing at most logarith-
mically with the scale factor. Details of the numerical integration and the non-relativistic WKB
approximation used to track this evolution up to MRE are provided in Appendix E.
Gravitational collapse becomes efficient only once DM dominates the cosmic energy density.
Consequently, theO(1) density contrasts generated during EMD can collapse only around the
MRE, defined byρ r=ρ a. Assuming axions constitute all the cold DM, equality occurs in the
redshiftz eq≃3365, corresponding to a temperatureT eq≃0.79 eV [57]. Regions with density
contrastδ aenter local matter domination earlier, at a redshiftz δagiven by (1+z δa)≃(1+δ a) (1+
zeq), and subsequently grow linearly with the scale factor until they become nonlinear and collapse
under their gravity [62]. Perturbations withδ a∼1 are therefore expected to collapse shortly after
MRE, giving rise to the first generation of dense axion miniclusters [23].
The density fluctuation spectrum evolved to MRE for the QCD axion is shown in Fig. 10 for
several reheating temperatures. The largest fluctuations occur on the smallest scales, down to the
Jeans scale, below which quantum pressure suppresses gravitational collapse [63]. This naturally
leads to a hierarchical picture of structure formation. Modes well below the nonlinear scale remain
supported by quantum pressure and oscillate, while the first gravitational collapses occur on scales
just above the Jeans cutoff. These initial collapses are expected to produce the first generation
21

of axion stars [64], solitons where gravitational pull is counteracted by quantum pressure; see
Refs. [65,66].
Structure formation then proceeds hierarchically toward larger scales. Modes withδ a∼1 col-
lapse around the MRE, forming abundant DM halos that can be identified with axion miniclusters.
Unlike the standard picture of smooth minicluster collapse, these objects are expected to possess
a rich internal substructure inherited from the earlier formation of axion stars. On still larger
scales, clusters of miniclusters form from perturbations withδ a<1. Because these fluctuations
require additional time to reach nonlinearity, the resulting halos are less dense and therefore are
more susceptible to tidal disruption by stellar encounters in the Milky Way [67–71].
Even without dedicated numerical simulations, linear analysis already identifies the charac-
teristic scales associated with this hierarchy of structures. The characteristic minicluster scale is
determined by the smallest perturbation mode that becomes nonlinear at MRE. Denoting byk nl
the comoving wavenumber satisfying
δa(knl, Req) = 1,(55)
whereR eqis the scale factor at equality; the corresponding physical radius is
rnl≡π
2Req
knl,(56)
given that the diameter is the half-wavelength scaleπ R eq/knl.
The characteristic minicluster massM mcis the mass enclosed within a region of radiusr nl
Mmc≃4π
3ρa(Req) [1 +δ a(knl, Req)]r3
nl
≃10−11M⊙×

0.06TRH
TQCD−3
forT BBN≲T RH≲35 MeV,
1.20TRH
TQCD−0.67
for 35 MeV≲T RH≲T QCD,
0.66TRH
TQCD−10.4
forT QCD≲T RH≲180 MeV,(57)
where the last part corresponds to a numerical fit and is shown in Fig. 11. At low reheating
temperatures, the minicluster mass follows the characteristic scalingM mc∝T−3
RH. This behavior
arises becauseκ nl≃50 is asymptotically independent of the reheating temperature forT RH≪T Λ
andk RH=H RHRRH∝T RHup tog ⋆,g⋆sfactors. As reheating approaches the QCD crossover,
TRH∼T Λ, enhanced density fluctuations lead to a decrease inκ nl, shifting the nonlinear scale
toward larger physical radii and producing substantially heavier miniclusters. Simultaneously, the
Jeans scale decreases, suggesting the formation of more massive axion stars.
22

10−810−710−6
ma,0[eV]10−1310−1210−1110−1010−910−8Mmc[M⊙]M
mc∝T−3RH
Mmc∝T−0.67RHM
mc
∝
T−10
.4RHTRH<T BBN
|δa|<15 10 20 50 100 150200TRH[MeV]Figure 11: Minicluster mass as a function of the zero-temperature QCD axion massm a,0and the
reheating temperatureT RH. The solid blue curve shows the numerical result. The dotted red,
dashed green, and dash-dotted purple curves represent power-law fits to the numerical solution in
different reheating-temperature regimes. The vertical orange dashed line indicates the QCD scale,
TΛ≃150 MeV.
Axion miniclusters and their axion stars will continue evolving through structure formation by
accretion, tidal forces, mergers, and wave-collapse [72–80]. The broad picture has been outlined
in several references and some limited results have been obtained by numerical simulations [81–
84]. Axion miniclusters will develop solitonic cores (central axion stars) and a Navarro-Frenk-
White profile [81] characteristic of DM structure formation. The current halo-mass function is still
uncertain, but references agree that the most severe threat to miniclusters and their axion stars is
tidal disruption with stars while orbiting galaxies like the Milky Way.
The Earth-minicluster encounter rate is given by the standard expression for the collision
probability in a gas of particles,
Γenc=n mc,local σ vrel,(58)
whereσ=πR2
mcis the geometric cross section associated with a minicluster of characteristic radius
Rmc, andv rel≃220 km/s is the typical relative velocity between the Earth and dark matter in the
Galactic halo.
The virialized radius of the miniclusterR mcis obtained from its mass and internal density,
Rmc=3Mmc
4πρ mc1/3
,(59)
23

whereρ mcdenotes the average density within the virialized object. To estimateρ mc, we follow the
spherical collapse analysis of Ref. [62], which is obtained for large-amplitude isocurvature fluctu-
ations that become nonlinear before matter–radiation equality, but we apply it to our scenario,
assuming the spherical collapse dynamics is independent of the perturbation origin. In that frame-
work, the final virialized core density of a clump seeded by an initial overdensityξ≡δρ DM/ρDM
is [62,67]
ρmc≃140ξ3(ξ+ 1)ρ EQ∼7×106ξ3(ξ+ 1) GeV/cm3,(60)
whereρ EQis the matter energy density at matter–radiation equality. We useξ∼1 since our
estimates are based on a linear analysis. Then, the encounter rate Eq. (58) is given by
Γenc=π v relρDM,local3
4π ρ mc2/3
M−1/3
mc,
≃2×10−6year−1×

2.5TRH
TQCD
forT BBN≲T RH≲35 MeV,
0.91TRH
TQCD0.22
for 35 MeV≲T RH≲T QCD,
1.12TRH
TQCD3.5
forT QCD≲T RH≲180 MeV.(61)
Axion miniclusters have a rich phenomenology, which potentially enables their indirect discov-
ery but also hampers the direct detection of DM. Simulations of post-inflationary axion scenarios
support the expectation that anO(1) fraction of DM can be bound into dense miniclusters when
the primordial density fluctuations reach order unity [61, 64, 72, 82, 84]. The residual density in
voids then sets a lower bound on the smooth axion component available to terrestrial haloscope
searches [85]. However, recent estimates of the disruption probability indicate that even if axion
miniclusters and stars are too dense to be disrupted, the next generations of miniclusters, i.e.
the clusters of miniclusters will be disrupted, releasing the trapped axions into tidal streams that
overlap and restore the local DM density around the Earth to∼80% of its standard value [71].
The surviving miniclusters and stars can be detected by a number of probes, such as microlens-
ing [75,86–88], femtolensing [89,90], see also Refs. [91,92] and radio emission [69,76,93–103].
6 Conclusions
In this work, we have studied the growth of axion dark-matter perturbations during an early
matter-dominated era. We focused on axions produced through the standard misalignment mech-
24

anism in a pre-inflationary PQ-breaking scenario, allowing for a temperature-dependent mass as
motivated by the QCD axion and by ALPs whose mass is generated by a hidden confining sector.
The key effect identified in this work is that fluctuations in the radiation temperature induce
fluctuations in the axion mass. This generates an additional source term in the perturbed Klein–
Gordon equation, directly coupling radiation overdensities to axion perturbations. During EMD,
radiation perturbations sourced by the decay of the matter-like field can become much larger
than in standard radiation domination. When the axion mass is still temperature dependent,
these enhanced radiation perturbations are efficiently transferred to the axion sector, leading to a
significant amplification of small-scale axion overdensities.
For generic ALPs withm a(T) =m a,0(TΛ/T)babove the saturation temperatureT Λ, we found
that the enhancement is largest when axion oscillations begin during EMD and the mass saturates
close to reheating. Once the mass reaches its zero-temperature value, the temperature-induced
source shuts off and axion pressure suppresses sufficiently small scales.
We also applied the framework to the QCD axion, using the lattice-motivated temperature
dependence of the mass. The strongest enhancement occurs for reheating temperatures below
but close to the QCD scale. In the corresponding region of parameter space, axion overdensities
can become nonlinear by matter–radiation equality, providing a mechanism for forming axion
miniclusters even in a pre-inflationary PQ-breaking scenario, without relying on topological defects
or primordial axion isocurvature fluctuations.
The small-scale structures produced in this way can include axion miniclusters and, near the
Jeans scale, dense axion-star-like cores. Their detailed properties require nonlinear simulations,
including gravitational collapse, wave dynamics, mergers, accretion, and tidal disruption in galactic
environments. More broadly, our results show that temperature-dependent axion masses provide
a sensitive probe of the pre-BBN expansion history, and that small-scale axion structure can carry
information about both axion microphysics and low-temperature reheating.
Acknowledgments
AA and PA acknowledge support from FONDECYT project No. 1251613. This publication is based
on work from the COST Action “COSMIC WISPers” (CA21106). NB thanks the Universidad
T´ ecnica Federico Santa Mar´ ıa for its hospitality.
25

A Perturbed fluid equations
In this work, we treat the scalar fieldϕand the SM radiationras perfect fluids, adopting a similar
treatment to that of Refs. [59, 104, 105]. Each fluidσis characterized by an energy densityρ σ, a
pressurep σ, and a stress-energy tensor
T(σ)µ
ν= (ρ σ+pσ)uµuν+δµ
νpσ,(62)
withσ=ϕ,r. At the background level, we consider a Universe described by the FLRW metric
ds2=−dt2+R2(t)δ ijdxidxj.(63)
We define the four-velocity in the rest frame of each fluid by the normalization conditionu µuµ=
−1, such thatuµ= (1,0,0,0). In our setup, the scalar field and radiation interact
∇µT(σ)µ
ν=Q(σ)
ν,(64)
so the stress tensor of each component is not conserved. However, the sum over all species must
satisfy the covariant conservation law of the total energy–momentum tensor
X
σ=ϕ,r∇µT(σ)µ
ν= 0,(65)
where
Q(ϕ)
ν= Γ ϕT(ϕ)
µνuµ,(66)
Q(r)
ν=−Q(ϕ)
ν (67)
represent the energy injected through decays ofϕinto SM radiation [59]. It follows from Eq. (64)
that the background continuity equation for each fluid component is
˙ρσ+ 3Hρ σ(1 +w σ) =−Q(σ)
0,(68)
wherew σis the equation-of-state parameter ofσ.
The perturbed metric in the Newtonian gauge is given by
ds2=−(1 + 2Φ)dt2+R2(t) (1−2Ψ)δ ijdxidxj,(69)
where Φ and Ψ are the Newtonian potential and the spatial curvature perturbation, respec-
tively [58]. From now on, we will neglect the anisotropic stress, setting Φ = Ψ. The energy-
momentum tensor can be decomposed linearly into a background ( T(σ)) and a perturbed (δT(σ))
contribution as
T(σ)µ
ν=T(σ)µ
ν+δT(σ)µ
ν.(70)
26

In the same way, the energy density and pressure can be decomposed as
ρσ= ¯ρ σ+δρ σ,(71)
pσ= ¯p σ+δp σ.(72)
For the fluids considered here, we take
wϕ=c2
ϕ= 0 andw r=c2
r=1
3,(73)
wherec2
σ≡δp σ/δρσfor the barotropic fluids used in this work. For each fluid, we can write the
individual components of the perturbed stress-energy tensor as
δT(σ)0
0=−δρ σ,(74)
δT(σ)0
i=ρ σ 
1 +w σ
∂ivσ,(75)
δT(σ)i
0=−ρ σ 
1 +w σ
gij∂jvσ,(76)
δT(σ)i
j=c2
σδρσδi
j,(77)
where we use the conditionuµuµ=−1 to write the perturbed four-velocity asu µ= (−1−Φ, ∂ ivσ),
withv σbeing the velocity potential of the fluidσ.
The energy transfer termsQ(σ)
νcan be decomposed into a background and a perturbed contri-
bution as
Q(σ)
ν=Q(σ)
ν+δQ(σ)
ν; (78)
with this, the covariant conservation equation for the perturbed energy–momentum tensor of each
fluid reads
∇µδT(σ)µ
ν+δΓµ
µαT(σ)α
ν−δΓα
µνT(σ)µ
α=δQ(σ)
ν,(79)
whereδΓλ
µνdenotes the perturbation of the Christoffel symbols. For our metric, the non-vanishing
components are
δΓ0
00=˙Φ, δΓi
00=1
R2∂iΦ,
δΓ0
0i=∂ iΦ, δΓi
0j=−˙Φδi
j,(80)
δΓ0
ij=−R2 
4HΦ + ˙Φ
δij, δΓi
jk=−∂ jΦδi
k−∂kΦδi
j+∂iΦδjk.
As commonly used, we move to new variables, the overdensityδ σand the divergence of the con-
formal velocityθ σ, defined as
δσ≡δρσ
ρσ, θ σ≡1
R∇2vσ.(81)
27

We obtain the evolution equations for density and velocity perturbations by applying the per-
turbed conservation law of the energy–momentum tensor, Eq. (79), to the temporal and spatial
components, and by including on the right-hand side the corresponding energy transfer termQ(σ)
ν,
following
˙δσ+ 3H 
c2
σ−w σ
δσ+ (1 +w σ)θσ
R−3˙Φ
=1
ρσh
Q(σ)
0δσ−δQ(σ)
0i
,(82)
˙θσ+ [H(1−3w σ)]θσ+c2
σ
1 +w σ∇2δσ
R+∇2Φ
R=1
ρσ"
∂iδQ(σ)
i
R(1 +w σ)+Q(σ)
0θσ#
.(83)
Expanding the source term in Eq. (66) up to first order gives
δQ(ϕ)
0= +Γ ϕρϕ(δϕ+ Φ),(84)
δQ(ϕ)
i=−Γ ϕρϕ∂ivϕ.(85)
Using Eqs. (84) and (85) in Eqs. (82) and (83), and changing to Fourier space, by taking ⃗∇ →i ⃗k,
we obtain the evolution equations for the density and velocity perturbations of the scalar field and
radiation as
˙δϕ=−Γ ϕΦ−θϕ
R+ 3˙Φ,(86)
˙θϕ=−Hθ ϕ+k2Φ
R,(87)
˙δr=ρϕΓϕ
ρr(δϕ−δr+ Φ)−4
3θr
R+ 4˙Φ,(88)
˙θr=ρϕΓϕ
ρr3
4θϕ−θr
+k2
Rδr
4+ Φ
.(89)
Furthermore, the evolution of the gravitational potential can be derived from theµ=ν= 0
component of the perturbed Einstein equations, relating the density perturbations to the metric
perturbations, and takes the form
3H(˙Φ +HΦ) +k2
R2Φ =−1
2M2
PlX
σδρσ.(90)
This is the full Poisson equation, which describes how the density perturbations of all components
source the metric perturbations. In this expression, the termP
σδρσdenotes the total density
perturbation obtained by summing all the components of the fluid. For convenience, we rewrite
28

Eqs. (86) to (90) in terms of the scale factor, which yields
δ′
ϕ=−Γϕ
RHΦ−1
R2Hθϕ+ 3Φ′,(91)
δ′
r=ρϕ
ρrΓϕ
RH(δϕ−δr+ Φ)−4
31
R2Hθr+ 4Φ′,(92)
θ′
ϕ=k2
R2HΦ−1
Rθϕ,(93)
θ′
r=ρϕ
ρrΓϕ
RH3
4θϕ−θr
+k2
R2Hδr
4+ Φ
,(94)
Φ′=−1
6M2
PlRH2(ρϕδϕ+ρrδr+ρaδa) +k2
3R3H2+1
R
Φ
.(95)
The axion overdensity appears in Eq. (95), but can be safely neglected sinceρ a≪ρ r, ρϕ.
Initial conditions
Now we derive the initial conditions for the system of Eqs. (91) to (95), following Refs. [26,59]. We
assume that all relevant modes are initially superhorizon (k≪HR). In this regime, perturbations
are constant in the Newtonian gauge, soδ′
σ= 0. During the EMD era, the gravitational potential
remains approximately constant
Φ(R)≃Φ(R ini)≡Φ p,(96)
where Φ pdenotes the initial Newtonian-potential amplitude. In the numerical analysis, we take
Φp=√Asas a convenient normalization, withA s≃2.101×10−9the amplitude of the primordial
scalar power spectrum measured by Planck [57].
Starting from Eq. (95), we neglect terms of orderO(k2) in the superhorizon limit. Since the
Universe isϕ-dominated, we use 3M2
PlH2≃ρ ϕand then
1
6M2
PlH2(ρϕδϕ+ρrδr+ρaδa) + Φ≃1
2
δϕ+ρr
ρϕδr+ρa
ρϕδa
+ Φ≃0.(97)
Sinceρ ϕ≫ρ r, ρa, we neglect the contributions fromδ randδ ato obtain
δϕ(Rini) =−2 Φ p.(98)
Next, we study the radiation overdensity in Eq. (92). We use Φ′≃0 and the superhorizon
conditionδ′
r≃0. From Eq. (30),θ σ≡ ∇2vσ/R, soθ r∼ O(k2) and can be neglected. With these
approximations, we obtain
δϕ(Rini)−δ r(Rini) + Φ p= 0.(99)
Using Eq. (98), we find
δr(Rini) =−Φ p.(100)
29

Additionally, for the initial condition of the velocity divergence, we set
θϕ(Rini) =θ r(Rini) = 0.(101)
This is justified by the fact that superhorizon modes remain frozen and do not develop spa-
tial gradients, so the velocity divergence is negligible. Furthermore, as shown by the analytical
approximation during the EMD era (see Eq. (107)), the initial value term becomes progressively
irrelevant as the scale factor evolves and remains subleading compared to the other terms ofθ σ(R).
B Analytical solutions
In the following, analytical solutions for the evolution of the background and its perturbations are
presented, valid deep in the EMD era. Consequently, we assume thatρ ϕ≫ρ r,H≫Γ ϕ, and a
negligible initial SM radiation density.
Background
The energy densities forϕand the SM radiation in Eqs. (5) and (6) scale as
ρϕ(R)≃ρ ϕ(Rini)Rini
R3
,(102)
ρr(R)≃2√
3
5MPlΓϕq
ρϕ(Rini)Rini
R3/2
,(103)
while the Hubble expansion rate scales as
H(R)≃H(R ini)Rini
R3/2
,(104)
with
H2(Rini)≡ρϕ(Rini)
3M2
Pl.(105)
Fluid perturbations during EMD
As the gravitational potential remains constant during the EMD era with a value Φ(R)≃Φ p,
Eq. (41) becomes
Φ′≃0.(106)
Then, the derivative of the velocity divergence forϕin Eq. (39) can be solved, giving rise to
θϕ(R)≃θ ϕ,iRini
R+2
3k2Φp
kini"r
R
Rini−Rini
R#
≃2
3k2Φp
kinir
R
Rini,(107)
30

withk ini≡R iniH(R ini). We use this approximate solution in Eq. (91) and solve forδ ϕ, obtaining
δϕ(R)≃δ ϕ,i−2
3Φpk
kini2R
Rini"
1 + 2Rini
R3/2
−3Rini
R#
≃ −2 Φ p−2
3Φpk
kini2R
Rini=−2 Φ p
1 +κ2
3R
RRH
.(108)
It is interesting to note that during the EMD era, nonlinearity|δ ϕ|= 1 can only be reached for
modes corresponding toκ≳p
3/(2 Φ p)≃181, atR=R RH.
The common source term in Eqs. (92) and (94) can be approximated using the analytical
background solutions given in Eqs. (102) to (104). We obtainρϕ
ρrΓϕ
H≃5
2. Using this result,
together with the analytical solutions forδ ϕandθ ϕ, the system is reduced to
δ′
r≃ −5
2Rδr−4
3kini√RiniRθr−5
2RΦp"
1 +2
3k
kini2R
Rini#
,(109)
θ′
r≃k2
4kini√RiniRδr−5
2Rθr+9
4k2Φp
kini√RiniR.(110)
Next, we differentiate Eq. (109) with respect toR, and then substituteθ′
randθ rusing Eqs. (110)
and (109), respectively, to obtain
δ′′
r+11
2Rδ′
r+1
R2"
5 +1
3k
kini2R
Rini#
δr≃ −1
R2Φp"
5 + 8k
kini2R
Rini#
.(111)
For the subhorizon modes, the terms proportional tok2dominate over the contributions indepen-
dent ofkwithin the squared brackets. The maximal growth of the subhorizon perturbation can
be found by settingδ′′
r=δ′
r= 0, and corresponds to
δr≃ −24 Φ p= 24δ r(Rini).(112)
This corresponds to the plateau observed in the lower left panel of Fig. 3. Additionally, it is
possible to solve Eq. (111) analytically; we find
δr(˜x)≃Φ p"
−24 +345
˜κ2˜x−3105
2˜κ4˜x2
+1
4˜κ5˜x5/2n
46˜κ 
135−30˜κ2+ 2˜κ4
cosφ(˜x) + 115√
3 
27−18˜κ2+ 2˜κ4
sinφ(˜x)o#
,
(113)
where we have used the initial conditionδ′
r(Rini) = 0, and the dimensionless variables ˜x≡R/R ini
and ˜κ≡k/k ini. The phase is given by
φ(˜x)≡2 ˜κ√
3√
˜x−1
.(114)
31

Finally, the velocity divergence for the radiation is
θr(˜x)≃θr,i
˜x5/2+kiniΦp
16˜κ4˜x5/2"
5 
1863−1242˜κ2˜x+ 138˜κ4˜x2−4˜κ6(˜x3−1)
−345(27−18˜κ2+ 2˜κ4) cos(φ(˜x)) + 46√
3˜κ(135−30˜κ2+ 2˜κ4) sin(φ(˜x))#
.(115)
C Axion perturbations
It is useful to express the axion field in terms of a dimensionless angular variable, defined as
θ=a/f a, with values within [−π, π]. We decompose the field into a homogeneous component and
a first-order perturbation,
θ(t, ⃗ x) =θ 0(t) +δθ(t, ⃗ x),(116)
with
δθ(t, ⃗ x) =Zd3k
(2π)3δθk(t)ei⃗k·⃗ x.(117)
Since the axion mass depends on temperature, the mass term also acts as a source of perturbations.
Therefore, the mass term can be expanded to the first order as
m2
a(T) =m2
a(T) +δm2
a,withδm2
a=dm2
a
dT
TδT ,(118)
where we used the first-order expansion of the temperatureT(t, k) = T(t) +δT(t, k), with Tthe
background temperature.
Then, using Eqs. (117) and (118), we expand at linear order the derivative of the axion potential
with respect toθas
∂V(θ, T)
∂θ=m2
a(T)f2
asinθ
≃m2
a(T)f2
asinθ 0+m2
a(T)f2
acosθ 0δθk+dm2
a
dT
Tf2
asinθ 0δT .(119)
Next, we can construct the Klein-Gordon equation
1√−g∂µ √−g gµν∂νθ
−1
f2
a∂V(θ, T)
∂θ= 0.(120)
Using the perturbed metric in Eq. (29), the first term can be written as
1√−g∂µ √−g gµν∂νθ
≃ −(1−2Φ)
¨θ0+ 3H ˙θ0
−
δ¨θk+ 3Hδ ˙θk+k2
R2δθk
+ 4˙Φ˙θ0,(121)
32

we obtain the perturbed equation of motion for the axion field
∂2δθk
∂t2+ 3H∂δθk
∂t+k2
R2δθk+m2
acosθ 0δθk= 4˙Φ˙θ0−2Φm2
asinθ 0−1
4dm2
a
dTsinθ 0T δr,(122)
where we have usedδρ r/ρr= 4δT/ T, ignoring the variation of the effective relativistic degrees of
freedom. The perturbed fluid quantities can be obtained using the energy-momentum tensor of a
scalar field
Tµ
ν=f2
agµλ∂νθ ∂λθ−δµ
νf2
a
2gλρ∂λθ ∂ρθ+V(θ)
.(123)
Using the perturbed metric in Eq. (29) and expanding the field to first order as in Eq. (44), we
isolate the first-order perturbed temporal and spatial components of the energy-momentum tensor
as
δT0
0=f2
a
Φ˙θ2
0−˙θ0δ˙θk−m2
a(T) sinθ 0δθk−T
4dm2
a
dT
T(1−cosθ 0)δr
,(124)
δTi
i= 3f2
a
˙θ0δ˙θk−Φ˙θ2
0−m2
a(T) sinθ 0δθk−T
4dm2
a
dT
T(1−cosθ 0)δr
.(125)
In the regimeT > T Λ, the mass of the axion still depends on the temperature. In this case, using
Eq. (14), we can write
T
4dm2
a
dT
T=−b
2m2
a.(126)
Then, using Eqs. (74) and (77), the perturbed axion energy density and pressure can be written
as
δρa=f2
a
˙θ0δ˙θk−Φ˙θ2
0+m2
a(T) sinθ 0δθk−b
2m2
a(T) (1−cosθ 0)δr
,(127)
δpa=f2
a
˙θ0δ˙θk−Φ˙θ2
0−m2
a(T) sinθ 0δθk+b
2m2
a(T) (1−cosθ 0)δr
.(128)
OnceT < T Λ, the effect of temperature fluctuations vanishes and the usual expression is recovered
for the perturbed axion energy density [106, 107]. Finally, using the axion energy density in
Eq. (24), the axion overdensity becomes
δa=˙θ0δ˙θk−Φ˙θ2
0+m2
aδθksinθ 0−b m2
aδr(1−cosθ 0)/2
˙θ2
0/2 +m2
a(1−cosθ 0),(129)
forT > T Λ; the opposite caseT < T Λ, the last term in the numerator should be discarded.
D Jeans threshold scale
Although the axion mass is temperature dependent, radiation perturbations source axion pertur-
bations through the explicit dependencem a=m a(T). In this regime, the usual constant-mass
33

Jeans argument does not directly apply: the additional radiation-induced source can continue to
enhance small-scale axion perturbations after horizon entry. Once the mass saturates atR=R Λ,
however, this source is shut off. The subsequent evolution is then governed by the competition
between gravitational forcing and the effective pressure of the oscillating axion field. We now
estimate which modes become pressure suppressed afterR Λ.
ForR Λ≲R < R RH, we are allowed to use fluid treatment and write the equation that governs
the dynamics of axions with constant mass [10,26]
¨δa+ 2H ˙δa+ω2
kδa≃ −k2
R2Φ,(130)
where we have neglected the derivatives of Φ and introduced the effective frequencyω2
k≡c2
sk2/R2,
withc sthe effective scale-dependent axion sound speed [107]
c2
s=k2
k2+ 4m2
a,0R2≃k2
4m2
a,0R2,(131)
where the last approximation is valid fork≪2m a,0R.
Equation (130) describes a forced harmonic oscillator. The dynamics is then set by the com-
petition between the response time of the axion pressure,ω−1
k, and the Hubble timeH−1, which
controls the evolution of the gravitational source. Whenω k≪H, the pressure cannot react effi-
ciently in a Hubble time andδ ais effectively dragged by the source and continues to grow. For
ωk≫H, instead, the pressure dominates and the subsequent growth ofδ ais suppressed, leading
to oscillations. The transition between both regimes occurs whenω k(RΛ)∼H(R Λ), which defines
the Jeans scale
kJ∼R Λq
ma,0H(R Λ).(132)
During the EMD era,H(R Λ) =H RHx−3/2
Λ, therefore, the Jeans threshold can be found to be
κJ≡kJ
kRH≃rma,0
HRHx1/4
Λ,(133)
in good agreement with our numerical results. Modes withκ > κ Jare pressure-dominated at
xΛ, stop growing, and start to oscillate at that moment. In contrast, modes withκ < κ Jare
mass-dominated and continue the growth accumulated during the previous stage.
E Non-relativistic evolution to matter–radiation equality
For the numerical integration of the axion equations of motion during the EMD era and the
subsequent evolution, we solve the coupled background and perturbation equations with a Python
34

code based on an adaptive Runge–Kutta 45 integrator.4The axion evolution is computed using
the numerical solution of the EMD background and perturbation system as input. Once the axion
mass becomes much larger than the Hubble rate,m a≫H, the field undergoes rapid coherent
oscillations, requiring increasingly small integration steps. In practice, well after reheating, around
x∼ O(102), the direct Klein–Gordon integration becomes inefficient and can develop spurious
oscillatory numerical artifacts.
In Section 5, we require the evolution of axion perturbations up to MRE. The corresponding
scale factor follows from entropy conservation,
xeq≡Req
RRH=TRH
Teqg⋆s(TRH)
g⋆s(Teq)1/3
≃3.56×108TRH
150 MeVg⋆s(TRH)
25.911/3
,(134)
where we usedT eq= 0.79 eV andg ⋆s(Teq)≃3.91 [57]. Sincex eqis many orders of magnitude
larger than the range accessible to direct relativistic integration, we switch to a non-relativistic
WKB description once the axion field is deep in the oscillatory regime.
We decompose both the homogeneous axion field and its perturbations into slowly varying
complex amplitudes. For the homogeneous mode, we write
θ0(x) =1
xp
2M(x)h
ψ0(x)e−iRxM(x′)dx′+ψ∗
0(x)e+iRxM(x′)dx′i
,(135)
and similarly for the perturbation,
δθκ(x) =1
xp
2M(x)h
ψκ(x)e−iRxM(x′)dx′+ψ∗
κ(x)e+iRxM(x′)dx′i
.(136)
Here
M(x)≡ma
H(x)x.(137)
The matching pointx iis chosen after reheating and after the axion mass has saturated to its
zero-temperature value, so thatx i≫1,T(x i)< T Λ, andm a=m a,0.
During radiation domination,H≃H RHx−2, and therefore
M(x) =µx, µ≡ma,0
HRH.(138)
The rapidly oscillating phase is then
Zx
M(x′)dx′=1
2µx2+ const.(139)
The irrelevant constant phase is fixed by the matching convention atx i.
4The code is currently in preparation for public release.
35

At the matching point, the non-relativistic amplitudeψ κ(xi) can be extracted from the direct
numerical solution through
ψκ(xi)≃eiIixir
Mi
2
δθκ(xi) +i
Mi
δθ′
κ(xi) +1
xiδθκ(xi) +M′
i
2Miδθκ(xi)
,(140)
where
Ii≡Zxi
M(x′)dx′, M i≡M(x i).(141)
An analogous expression is used to obtainψ 0(xi) fromθ 0(xi) andθ′
0(xi). In the absence of pertur-
bative sources,ψ 0is conserved at leading order in the non-relativistic expansion.
Under the transformation in Eqs. (135) and (136), the perturbation equation reduces to a forced
Schr¨ odinger-like equation. Keeping the leading non-relativistic terms, one obtains
iψ′
κ=κ2
2M(x)ψκ+Jκ(x).(142)
During radiation domination, this becomes
iψ′
κ=κ2
2µxψκ+Jκ(x).(143)
Since the matching point is chosen after the axion mass has become constant, the temperature-
induced source satisfiesS 3= 0. For subhorizon modes, the contribution fromS 1is also subdomi-
nant compared with the gravitational sourceS 2. The leading source term is therefore
Jκ(x)≃µxΦ κ(x)ψ 0(x).(144)
The solution of Eq. (142) can be written in integral form. Defining the free non-relativistic phase
Pκ(x)≡Zx
xiκ2
2M(x′)dx′=κ2
2µlnx
xi
,(145)
we find
ψκ(x)≃e−iPκ(x)
ψκ(xi)−iZx
xidx′eiPκ(x′)Jκ(x′)
.(146)
To evaluate this expression up to MRE, we use the analytical form of the gravitational potential
for subhorizon modes during radiation domination,
Φκ(x)≃1
(κx)2
C1cosκx√
3
+C 2sinκx√
3
.(147)
The constantsC 1andC 2are fixed by matching Φ κand Φ′
κto the numerical solution atx i.
This approximation captures the leading subhorizon behavior of the gravitational potential in a
radiation-dominated Universe.
36

Finally, the axion density contrast is obtained from the interference between the homogeneous
and perturbed non-relativistic fields. Averaging over the rapid oscillations, one finds
δa(x) =δρa
ρa≃2 Reψ∗
0(x)ψ κ(x)
|ψ0(x)|2
,(148)
up to corrections suppressed byH/m a,0andk2/(m2
a,0R2).
F Isocurvature bound
In the pre-inflationary PQ breaking scenario considered here, the axion field is present as a light
spectator during inflation and acquires quantum fluctuations of amplitudeδa=H I/(2π), where
HIis the inflationary Hubble scale. These fluctuations result in perturbations of the misalignment
angleδθ=H I/(2πf a), which generate a DM isocurvature mode uncorrelated with the primordial
adiabatic curvature perturbation. In the small-angle regime, where the axion abundance scales as
ρa∝θ2
ini, the corresponding isocurvature power spectrum is [108]
PS=Ωa
ΩDM2HI
π faθini2
.(149)
Planck constrains the isocurvature fraction at the pivot scalek p= 0.05 Mpc−1toβ iso≡ PS/(PR+
PS)≲0.038 at 95% CL [57], withP R=A s≃2.101×10−9. Since we want the axion to saturate
the DM abundance Ω a= Ω DM, one has
HI≲π f aθinis
βiso
1−β isoAs≃2.8×10−5faθini.(150)
For the QCD axion benchmark of Section 4.3.2, this bound should be evaluated using the cor-
responding value off aθini. For example, iff a≃1013GeV andθ ini≃1, one obtainsH I≲
2.8×108GeV. The bound is straightforwardly satisfied for sufficiently low-scale inflation. In the
perturbation analysis above, we set the primordial axion isocurvature mode to zero to isolate the
axion perturbations sourced by the adiabatic radiation and metric perturbations.
References
[1] J. Preskill, M. B. Wise, and F. Wilczek, “Cosmology of the Invisible Axion,”Phys. Lett. B
120(1983) 127–132.
[2] F. W. Stecker and Q. Shafi, “The Evolution of Structure in the Universe From Axions,”
Phys. Rev. Lett.50(1983) 928.
37

[3] M. Dine and W. Fischler, “The Not So Harmless Axion,”Phys. Lett. B120(1983) 137–141.
[4] L. Abbott and P. Sikivie, “A Cosmological Bound on the Invisible Axion,”Phys. Lett. B
120(1983) 133–136.
[5] S. Weinberg, “A New Light Boson?,”Phys. Rev. Lett.40(1978) 223–226.
[6] F. Wilczek, “Problem of StrongPandTInvariance in the Presence of Instantons,”Phys.
Rev. Lett.40(1978) 279–282.
[7] R. D. Peccei and H. R. Quinn, “CP Conservation in the Presence of Instantons,”Phys.
Rev. Lett.38(1977) 1440–1443.
[8] R. D. Peccei and H. R. Quinn, “Some Aspects of Instantons,”Nuovo Cim. A41(1977) 309.
[9] R. D. Peccei and H. R. Quinn, “Constraints Imposed by CP Conservation in the Presence
of Instantons,”Phys. Rev. D16(1977) 1791–1797.
[10] A. Arvanitaki, S. Dimopoulos, S. Dubovsky, N. Kaloper, and J. March-Russell, “String
Axiverse,”Phys. Rev. D81(2010) 123530,arXiv:0905.4720 [hep-th].
[11] P. Arias, D. Cadamuro, M. Goodsell, J. Jaeckel, J. Redondo, and A. Ringwald, “WISPy
Cold Dark Matter,”JCAP06(2012) 013,arXiv:1201.5902 [hep-ph].
[12] R. T. Co, L. J. Hall, and K. Harigaya, “Axion Kinetic Misalignment Mechanism,”Phys.
Rev. Lett.124no. 25, (2020) 251802,arXiv:1910.14152 [hep-ph].
[13] B. Barman, N. Bernal, N. Ramberg, and L. Visinelli, “QCD Axion Kinetic Misalignment
without Prejudice,”Universe8no. 12, (2022) 634,arXiv:2111.03677 [hep-ph].
[14] A. Bodas, R. T. Co, A. Ghalsasi, K. Harigaya, and L.-T. Wang, “Acoustic misalignment
mechanism for axion dark matter,”JHEP08(2025) 131,arXiv:2503.04888 [hep-ph].
[15] A. Papageorgiou, P. Qu´ ılez, and K. Schmitz, “Axion dark matter from frictional
misalignment,”JHEP01(2023) 169,arXiv:2206.01129 [hep-ph].
[16] L. Di Luzio, B. Gavela, P. Qu´ ılez, and A. Ringwald, “Dark matter from an even lighter
QCD axion: trapped misalignment,”JCAP10(2021) 001,arXiv:2102.01082 [hep-ph].
[17] S. Nakagawa, F. Takahashi, and M. Yamada, “Trapping Effect for QCD Axion Dark
Matter,”JCAP05(2021) 062,arXiv:2012.13592 [hep-ph].
38

[18] N. Kitajima and F. Takahashi, “Resonant production of dark photons from axions without
a large coupling,”Phys. Rev. D107no. 12, (2023) 123518,arXiv:2303.05492 [hep-ph].
[19] P. Sikivie and W. Xue, “Resonant excitation of the axion field during the QCD phase
transition,”Phys. Rev. D105no. 4, (2022) 043533,arXiv:2110.13157 [hep-ph].
[20] N. Kitajima, K. Kogai, and Y. Urakawa, “New scenario of QCD axion clump formation.
Part I. Linear analysis,”JCAP03no. 03, (2022) 039,arXiv:2111.05785 [astro-ph.CO].
[21] A. Ayad and D. J. Schwarz, “Impact of adiabatic temperature fluctuations on the power
spectrum of axion density perturbations,”JCAP08(2025) 073,arXiv:2503.05532
[astro-ph.CO].
[22] I. J. Allali, P. Chakraborty, J. Fan, and M. Reece, “Axion Perturbations: A General
Analytical Treatment,”arXiv:2510.07371 [hep-ph].
[23] E. W. Kolb and I. I. Tkachev, “Axion miniclusters and Bose stars,”Phys. Rev. Lett.71
(1993) 3051–3054,arXiv:hep-ph/9303313.
[24] E. Hardy, “Miniclusters in the Axiverse,”JHEP02(2017) 046,arXiv:1609.00208
[hep-ph].
[25] L. Visinelli and J. Redondo, “Axion Miniclusters in Modified Cosmological Histories,”
Phys. Rev. D101no. 2, (2020) 023008,arXiv:1808.01879 [astro-ph.CO].
[26] N. Blinov, M. J. Dolan, and P. Draper, “Imprints of the Early Universe on Axion Dark
Matter Substructure,”Phys. Rev. D101no. 3, (2020) 035002,arXiv:1911.07853
[astro-ph.CO].
[27] G. F. Giudice, E. W. Kolb, and A. Riotto, “Largest temperature of the radiation era and
its cosmological implications,”Phys. Rev. D64(2001) 023508,arXiv:hep-ph/0005123.
[28] L. Visinelli and P. Gondolo, “Axion cold dark matter in non-standard cosmologies,”Phys.
Rev. D81(2010) 063508,arXiv:0912.0015 [astro-ph.CO].
[29] P. Arias, N. Bernal, D. Karamitros, C. Maldonado, L. Roszkowski, and M. Venegas, “New
opportunities for axion dark matter searches in nonstandard cosmological models,”JCAP
11(2021) 003,arXiv:2107.13588 [hep-ph].
[30] A. D. Dolgov and D. P. Kirilova, “On Particle Creation by a Time Dependent Scalar
Field,”Sov. J. Nucl. Phys.51(1990) 172–177.
39

[31] J. H. Traschen and R. H. Brandenberger, “Particle Production During Out-of-equilibrium
Phase Transitions,”Phys. Rev. D42(1990) 2491–2504.
[32] L. Kofman, A. D. Linde, and A. A. Starobinsky, “Reheating after inflation,”Phys. Rev.
Lett.73(1994) 3195–3198,arXiv:hep-th/9405187.
[33] L. Kofman, A. D. Linde, and A. A. Starobinsky, “Towards the theory of reheating after
inflation,”Phys. Rev. D56(1997) 3258–3295,arXiv:hep-ph/9704452.
[34] R. Allahverdiet al., “The First Three Seconds: a Review of Possible Expansion Histories of
the Early Universe,”Open J. Astrophys.4(6, 2020) 001,arXiv:2006.16182
[astro-ph.CO].
[35] B. Batellet al., “Conversations and deliberations: Non-standard cosmological epochs and
expansion histories,”Int. J. Mod. Phys. A40no. 17, (2025) 2530004,arXiv:2411.04780
[astro-ph.CO].
[36] B. Barman, N. Bernal, and J. Rubio, “Two or three things particle physicists
(mis)understand about (pre)heating,”Nucl. Phys. B1018(2025) 116996,
arXiv:2503.19980 [hep-ph].
[37] J. R. Ellis, D. V. Nanopoulos, and M. Quiros, “On the Axion, Dilaton, Polonyi, Gravitino
and Shadow Matter Problems in Supergravity and Superstring Models,”Phys. Lett. B174
(1986) 176–182.
[38] T. Banks, D. B. Kaplan, and A. E. Nelson, “Cosmological implications of dynamical
supersymmetry breaking,”Phys. Rev. D49(1994) 779–787,arXiv:hep-ph/9308292.
[39] G. Kane, K. Sinha, and S. Watson, “Cosmological Moduli and the Post-Inflationary
Universe: A Critical Review,”Int. J. Mod. Phys. D24no. 08, (2015) 1530022,
arXiv:1502.07746 [hep-th].
[40] M. Cicoli, J. P. Conlon, A. Maharana, S. Parameswaran, F. Quevedo, and I. Zavala,
“String cosmology: From the early universe to today,”Phys. Rept.1059(2024) 1–155,
arXiv:2303.04819 [hep-th].
[41] S. Borsanyiet al., “Calculation of the axion mass based on high-temperature lattice
quantum chromodynamics,”Nature539no. 7627, (2016) 69–71,arXiv:1606.07494
[hep-lat].
40

[42] P. J. Steinhardt and M. S. Turner, “Saving the Invisible Axion,”Phys. Lett. B129(1983)
51.
[43] G. Lazarides, R. K. Schaefer, D. Seckel, and Q. Shafi, “Dilution of Cosmological Axions by
Entropy Production,”Nucl. Phys. B346(1990) 193–212.
[44] M. Kawasaki, T. Moroi, and T. Yanagida, “Can decaying particles raise the upper bound
on the Peccei-Quinn scale?,”Phys. Lett. B383(1996) 313–316,arXiv:hep-ph/9510461.
[45] D. Grin, T. L. Smith, and M. Kamionkowski, “Axion constraints in non-standard thermal
histories,”Phys. Rev. D77(2008) 085020,arXiv:0711.1352 [astro-ph].
[46] A. E. Nelson and H. Xiao, “Axion Cosmology with Early Matter Domination,”Phys. Rev.
D98no. 6, (2018) 063516,arXiv:1807.07176 [astro-ph.CO].
[47] N. Ramberg and L. Visinelli, “Probing the Early Universe with Axion Physics and
Gravitational Waves,”Phys. Rev. D99no. 12, (2019) 123513,arXiv:1904.05707
[astro-ph.CO].
[48] P. Arias, D. Karamitros, and L. Roszkowski, “Frozen-in fermionic singlet dark matter in
non-standard cosmology with a decaying fluid,”JCAP05(2021) 041,arXiv:2012.07202
[hep-ph].
[49] P. Carenza, M. Lattanzi, A. Mirizzi, and F. Forastieri, “Thermal axions with multi-eV
masses are possible in low-reheating scenarios,”JCAP07(2021) 031,arXiv:2104.03982
[astro-ph.CO].
[50] M. Venegas, “Relic Density of Axion Dark Matter in Standard and Non-Standard
Cosmological Scenarios,”arXiv:2106.07796 [hep-ph].
[51] N. Bernal, F. Hajkarim, and Y. Xu, “Axion Dark Matter in the Time of Primordial Black
Holes,”Phys. Rev. D104(2021) 075007,arXiv:2107.13575 [hep-ph].
[52] N. Bernal, Y. F. P´ erez-Gonz´ alez, Y. Xu, and ´O. Zapata, “ALP dark matter in a primordial
black hole dominated universe,”Phys. Rev. D104no. 12, (2021) 123536,
arXiv:2110.04312 [hep-ph].
[53] M. Kawasaki, K. Kohri, and N. Sugiyama, “Cosmological constraints on late time entropy
production,”Phys. Rev. Lett.82(1999) 4168,arXiv:astro-ph/9811437.
41

[54] M. Kawasaki, K. Kohri, and N. Sugiyama, “MeV scale reheating temperature and
thermalization of neutrino background,”Phys. Rev. D62(2000) 023506,
arXiv:astro-ph/0002127.
[55] P. F. de Salas, M. Lattanzi, G. Mangano, G. Miele, S. Pastor, and O. Pisanti, “Bounds on
very low reheating scenarios after Planck,”Phys. Rev. D92no. 12, (2015) 123534,
arXiv:1511.00672 [astro-ph.CO].
[56] T. Hasegawa, N. Hiroshima, K. Kohri, R. S. L. Hansen, T. Tram, and S. Hannestad,
“MeV-scale reheating temperature and thermalization of oscillating neutrinos by radiative
and hadronic decays of massive particles,”JCAP12(2019) 012,arXiv:1908.10189
[hep-ph].
[57]PlanckCollaboration, N. Aghanimet al., “Planck 2018 results. VI. Cosmological
parameters,”Astron. Astrophys.641(2020) A6,arXiv:1807.06209 [astro-ph.CO].
[Erratum: Astron.Astrophys. 652, C4 (2021)].
[58] S. Dodelson,Modern Cosmology. Academic Press, Amsterdam, 2003.
[59] A. L. Erickcek and K. Sigurdson, “Reheating Effects in the Matter Power Spectrum and
Implications for Substructure,”Phys. Rev. D84(2011) 083503,arXiv:1106.0536
[astro-ph.CO].
[60] A. Arvanitaki, S. Dimopoulos, M. Galanis, L. Lehner, J. O. Thompson, and
K. Van Tilburg, “Large-misalignment mechanism for the formation of compact axion
structures: Signatures from the QCD axion to fuzzy dark matter,”Phys. Rev. D101no. 8,
(2020) 083014,arXiv:1909.11665 [astro-ph.CO].
[61] E. Hardy, N. S´ anchez Gonz´ alez, H. Stubbs, and L. Tranchedone, “Pre-inflationary QCD
axion stars after moduli domination,”arXiv:2605.00103 [hep-ph].
[62] E. W. Kolb and I. I. Tkachev, “Large amplitude isothermal fluctuations and high density
dark matter clumps,”Phys. Rev. D50(1994) 769–773,arXiv:astro-ph/9403011.
[63] M. Y. Khlopov, B. A. Malomed, I. B. Zeldovich, and Y. B. Zeldovich, “Gravitational
instability of scalar fields and formation of primordial black holes,”Mon. Not. Roy. Astron.
Soc.215no. 4, (1985) 575–589.
[64] M. Gorghetto, E. Hardy, and G. Villadoro, “More axion stars from strings,”JHEP08
(2024) 126,arXiv:2405.19389 [hep-ph].
42

[65] R. Ruffini and S. Bonazzola, “Systems of selfgravitating particles in general relativity and
the concept of an equation of state,”Phys. Rev.187(1969) 1767–1783.
[66] J. Eby, M. Leembruggen, L. Street, P. Suranyi, and L. C. R. Wijewardhana, “Global view of
QCD axion stars,”Phys. Rev. D100no. 6, (2019) 063002,arXiv:1905.00981 [hep-ph].
[67] P. Tinyakov, I. Tkachev, and K. Zioutas, “Tidal streams from axion miniclusters and direct
axion searches,”JCAP01(2016) 035,arXiv:1512.02884 [astro-ph.CO].
[68] V. I. Dokuchaev, Y. N. Eroshenko, and I. I. Tkachev, “Destruction of axion miniclusters in
the Galaxy,”J. Exp. Theor. Phys.125no. 3, (2017) 434–442,arXiv:1710.09586
[astro-ph.GA].
[69] B. J. Kavanagh, T. D. P. Edwards, L. Visinelli, and C. Weniger, “Stellar disruption of
axion miniclusters in the Milky Way,”Phys. Rev. D104no. 6, (2021) 063038,
arXiv:2011.05377 [astro-ph.GA].
[70] X. Shen, H. Xiao, P. F. Hopkins, and K. M. Zurek, “Disruption of Dark Matter Minihalos
in the Milky Way Environment: Implications for Axion Miniclusters and Early Matter
Domination,”Astrophys. J.962no. 1, (2024) 9,arXiv:2207.11276 [astro-ph.GA].
[71] C. A. J. O’Hare, G. Pierobon, and J. Redondo, “Axion Minicluster Streams in the Solar
Neighborhood,”Phys. Rev. Lett.133no. 8, (2024) 081001,arXiv:2311.17367 [hep-ph].
[72] K. M. Zurek, C. J. Hogan, and T. R. Quinn, “Astrophysical Effects of Scalar Dark Matter
Miniclusters,”Phys. Rev. D75(2007) 043511,arXiv:astro-ph/0607341.
[73] D. G. Levkov, A. G. Panin, and I. I. Tkachev, “Gravitational Bose-Einstein condensation
in the kinetic regime,”Phys. Rev. Lett.121no. 15, (2018) 151301,arXiv:1804.05857
[astro-ph.CO].
[74] D. Ellis, D. J. E. Marsh, and C. Behrens, “Axion Miniclusters Made Easy,”Phys. Rev. D
103no. 8, (2021) 083525,arXiv:2006.08637 [astro-ph.CO].
[75] D. Ellis, D. J. E. Marsh, B. Eggemeier, J. Niemeyer, J. Redondo, and K. Dolag, “Structure
of axion miniclusters,”Phys. Rev. D106no. 10, (2022) 103514,arXiv:2204.13187
[hep-ph].
[76] D. Maseizik, S. Mondal, H. Seong, and G. Sigl, “Radio lines from accreting axion stars,”
JCAP05(2025) 033,arXiv:2409.13121 [hep-ph].
43

[77] D. Maseizik, J. Eby, H. Seong, and G. Sigl, “Detectability of accretion-induced bosenovae
in the Milky Way,”Phys. Rev. D111no. 6, (2025) 063017,arXiv:2410.13082 [hep-ph].
[78] I. DSouza, C. Gordon, and J. C. Forbes, “Enhanced disruption of axion minihalos by
multiple stellar encounters in the Milky Way,”Phys. Rev. D111no. 12, (2025) 123023,
arXiv:2411.16166 [astro-ph.CO].
[79] Z. Wang and Y. Gao, “Axion Star Bosenova in Axion Miniclusters,”arXiv:2508.14535
[hep-ph].
[80] L. Visinelli and M. Naydenov, “Transient axion streams from disrupted miniclusters,”
arXiv:2605.28005 [astro-ph.GA].
[81] B. Eggemeier, J. Redondo, K. Dolag, J. C. Niemeyer, and A. Vaquero, “First Simulations
of Axion Minicluster Halos,”Phys. Rev. Lett.125no. 4, (2020) 041301,arXiv:1911.09417
[astro-ph.CO].
[82] B. Eggemeier and J. C. Niemeyer, “Formation and mass growth of axion stars in axion
miniclusters,”Phys. Rev. D100no. 6, (2019) 063528,arXiv:1906.01348 [astro-ph.CO].
[83] A. S. Dmitriev, D. G. Levkov, A. G. Panin, and I. I. Tkachev, “Self-Similar Growth of Bose
Stars,”Phys. Rev. Lett.132no. 9, (2024) 091001,arXiv:2305.01005 [astro-ph.CO].
[84] B. Eggemeier, A. K. Anilkumar, and K. Dolag, “Evidence for axion miniclusters with an
increased central density,”Phys. Rev. D110no. 4, (2024) 043530,arXiv:2402.18221
[astro-ph.CO].
[85] B. Eggemeier, C. A. J. O’Hare, G. Pierobon, J. Redondo, and Y. Y. Y. Wong, “Axion
minivoids and implications for direct detection,”Phys. Rev. D107no. 8, (2023) 083510,
arXiv:2212.00560 [hep-ph].
[86] M. Fairbairn, D. J. E. Marsh, J. Quevillon, and S. Rozier, “Structure formation and
microlensing with axion miniclusters,”Phys. Rev. D97no. 8, (2018) 083502,
arXiv:1707.03310 [astro-ph.CO].
[87] E. D. Schiappacasse and T. T. Yanagida, “Can QCD axion stars explain Subaru HSC
microlensing?,”Phys. Rev. D104no. 10, (2021) 103020,arXiv:2109.13153 [hep-ph].
[88] A. Prabhu, “Optical Lensing by Axion Stars: Observational Prospects with Radio
Astrometry,”arXiv:2006.10231 [astro-ph.CO].
44

[89] E. W. Kolb and I. I. Tkachev, “Femtolensing and picolensing by axion miniclusters,”
Astrophys. J. Lett.460(1996) L25–L28,arXiv:astro-ph/9510043.
[90] A. Katz, J. Kopp, S. Sibiryakov, and W. Xue, “Femtolensing by Dark Matter Revisited,”
JCAP12(2018) 005,arXiv:1807.11495 [astro-ph.CO].
[91] L. Dai and J. Miralda-Escud´ e, “Gravitational Lensing Signatures of Axion Dark Matter
Minihalos in Highly Magnified Stars,”Astron. J.159no. 2, (2020) 49,arXiv:1908.01773
[astro-ph.CO].
[92] C. V. M¨ uller and J. Miralda-Escud´ e, “Limits on dark matter compact objects implied by
supermagnified stars in lensing clusters,”Mon. Not. Roy. Astron. Soc.536no. 2, (2024)
1579–1585,arXiv:2403.16989 [astro-ph.CO].
[93] A. Iwazaki, “Axion stars and fast radio bursts,”Phys. Rev. D91no. 2, (2015) 023008,
arXiv:1410.4323 [hep-ph].
[94] S. Raby, “Axion star collisions with Neutron stars and Fast Radio Bursts,”Phys. Rev. D
94no. 10, (2016) 103004,arXiv:1609.01694 [hep-ph].
[95] J. Eby, M. Leembruggen, J. Leeney, P. Suranyi, and L. C. R. Wijewardhana, “Collisions of
Dark Matter Axion Stars with Astrophysical Sources,”JHEP04(2017) 099,
arXiv:1701.01476 [astro-ph.CO].
[96] T. D. P. Edwards, B. J. Kavanagh, L. Visinelli, and C. Weniger, “Transient Radio
Signatures from Neutron Star Encounters with QCD Axion Miniclusters,”Phys. Rev. Lett.
127no. 13, (2021) 131103,arXiv:2011.05378 [hep-ph].
[97] M. A. Amin and Z.-G. Mou, “Electromagnetic Bursts from Mergers of Oscillons in
Axion-like Fields,”JCAP02(2021) 024,arXiv:2009.11337 [astro-ph.CO].
[98] M. A. Amin, A. J. Long, Z.-G. Mou, and P. Saffin, “Dipole radiation and beyond from
axion stars in electromagnetic fields,”JHEP06(2021) 182,arXiv:2103.12082 [hep-ph].
[99] J. Eby, S. Shirai, Y. V. Stadnik, and V. Takhistov, “Probing relativistic axions from
transient astrophysical sources,”Phys. Lett. B825(2022) 136858,arXiv:2106.14893
[hep-ph].
[100] P. J. Fox, N. Weiner, and H. Xiao, “Recurrent axion stars collapse with dark radiation
emission and their cosmological constraints,”Phys. Rev. D108no. 9, (2023) 095043,
arXiv:2302.00685 [hep-ph].
45

[101] L. Visinelli, B. Johnson, B. J. Kavanagh, D. J. E. Marsh, J. E. Shroyer, and L. Walters,
“Indirect detection of the QCD axion,”PoSCOSMICWISPers2024(2025) 023,
arXiv:2411.19441 [astro-ph.CO].
[102] L. Walters, J. E. Shroyer, M. Edenton, P. Agrawal, B. Johnson, B. J. Kavanagh, D. J. E.
Marsh, and L. Visinelli, “Axions in Andromeda: Searching for minicluster-neutron star
encounters with the Green Bank Telescope,”Phys. Rev. D110no. 12, (2024) 123002,
arXiv:2407.13060 [astro-ph.CO].
[103] P. J. Fox, N. Weiner, and H. Xiao, “Radio Killed the Axion Star: Constraining Axion
Properties with Radio Telescopes,”Phys. Rev. D113no. 5, (2026) 055038,
arXiv:2508.08371 [hep-ph].
[104] J. Fan, O. ¨Ozsoy, and S. Watson, “Nonthermal histories and implications for structure
formation,”Phys. Rev. D90no. 4, (2014) 043536,arXiv:1405.7373 [hep-ph].
[105] R. J. F. Marcondes, R. C. G. Landim, A. A. Costa, B. Wang, and E. Abdalla, “Analytic
study of the effect of dark energy-dark matter interaction on the growth of structures,”
JCAP12(2016) 009,arXiv:1605.05264 [astro-ph.CO].
[106] J.-c. Hwang and H. Noh, “Axion as a Cold Dark Matter candidate,”Phys. Lett. B680
(2009) 1–3,arXiv:0902.4738 [astro-ph.CO].
[107] D. J. E. Marsh, “Axion Cosmology,”Phys. Rept.643(2016) 1–79,arXiv:1510.07633
[astro-ph.CO].
[108] M. P. Hertzberg, M. Tegmark, and F. Wilczek, “Axion Cosmology and the Energy Scale of
Inflation,”Phys. Rev. D78(2008) 083507,arXiv:0807.1726 [astro-ph].
46