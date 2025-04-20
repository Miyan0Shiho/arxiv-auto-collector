# Smooth sailing or ragged climb? -- Increasing the robustness of power spectrum de-wiggling and ShapeFit parameter compression

**Authors**: Katayoon Ghaemi, Nils Schöneberg, Licia Verde

**Published**: 2025-04-14 18:00:01

**PDF URL**: [http://arxiv.org/pdf/2504.10578v1](http://arxiv.org/pdf/2504.10578v1)

## Abstract
The baryonic features in the galaxy power spectrum offer tight, time-resolved
constraints on the expansion history of the Universe but complicate the
measurement of the broadband shape of the power spectrum, which also contains
precious cosmological information. In this work we compare thirteen methods
designed to separate the broadband and oscillating components and examine their
performance. The systematic uncertainty between different de-wiggling
procedures is at most $2$%, depending on the scale. The ShapeFit parameter
compression aims to compute the slope $m$ of the power spectrum at large
scales, sensitive to matter-radiation equality and the baryonic suppression. We
show that the de-wiggling procedures impart large (50%) differences on the
obtained slope values, but as long as the theory and data pipelines are set up
consistently, this is of no concern for cosmological inference given the
precision of existing and on-going surveys. However, it still motivates the
search for more robust ways of extracting the slope. We show that
post-processing the power spectrum ratio before taking the derivative makes the
slope values far more robust. We further investigate eleven ways of extracting
the slope and highlight the two most successful ones. We derive a systematic
uncertainty on the slope $m$ of $\sigma_{m,\mathrm{syst}} = 0.023|m| + 0.001$
by studying the behavior of the slopes in different cosmologies within and
beyond $\Lambda$CDM and the impact in cosmological inference. In cosmologies
with a feature in the matter-power spectrum, such as in the early dark energy
cosmologies, this systematic uncertainty estimate does not necessarily hold,
and further investigation is required.

## Full Text


<!-- PDF content starts -->

Prepared for submission to JCAP
Smooth sailing or ragged climb? —
Increasing the robustness of power
spectrum de-wiggling and ShapeFit
parameter compression
Katayoon Ghaemi ,a,b,cNils Sch¨ oneberg ,d,eLicia Verdec,f
aAix-Marseille Universit´ e, CNRS/IN2P3, CPPM, Marseille, France
bDipartimento di Fisica e Astronomia “Galileo Galilei” Universit` a di Padova, I-35131 Padova, Italy
cICCUB Institut de Ci` encies del Cosmos, Universitat de Barcelona, Mart´ ı i Franqu` es 1, E08028
Barcelona, Spain
dUniversity observatory, Faculty of Physics, Ludwig-Maximilians-Universit¨ at, Scheinerstr. 1, 81677
Munich, Germany
eExcellence Cluster ORIGINS, Boltzmannstr. 2, 85748 Garching, Germany
fICREA, Instituci´ o Catalana de Recerca i Estudi Avan¸ cat, Pg. Lluis Companys 23, Barcelona, 08010,
Spain
E-mail: ghaemi@cppm.in2p3.fr, nils.science@gmail.com
Abstract. The baryonic features in the galaxy power spectrum offer tight, time-resolved constraints
on the expansion history of the Universe but complicate the measurement of the broadband shape of
the power spectrum, which also contains precious cosmological information. In this work we compare
thirteen methods designed to separate the broadband and oscillating components and examine their
performance. The systematic uncertainty between different de-wiggling procedures is at most 2%,
depending on the scale. The ShapeFit parameter compression aims to compute the slope mof the
power spectrum at large scales, sensitive to matter-radiation equality and the baryonic suppression.
We show that the de-wiggling procedures impart large (50%) differences on the obtained slope values,
but as long as the theory and data pipelines are set up consistently, this is of no concern for cos-
mological inference given the precision of existing and on-going surveys. However, it still motivates
the search for more robust ways of extracting the slope. We show that post-processing the power
spectrum ratio before taking the derivative makes the slope values far more robust. We further in-
vestigate eleven ways of extracting the slope and highlight the two most successful ones. We derive
a systematic uncertainty on the slope mofσm,syst= 0.023|m|+ 0.001 by studying the behavior of
the slopes in different cosmologies within and beyond ΛCDM and the impact in cosmological infer-
ence. In cosmologies with a feature in the matter-power spectrum, such as in the early dark energy
cosmologies, this systematic uncertainty estimate does not necessarily hold, and further investigation
is required.arXiv:2504.10578v1  [astro-ph.CO]  14 Apr 2025

Contents
1 Introduction 2
2 Broadband/BAO decomposition 3
2.1 On using analytical formulae 5
2.2 Numerical smoothing 6
2.3 Fitting smooth functions 6
2.4 Inflections 11
2.5 Correlation function peak removal 14
2.6 Summary 16
3 ShapeFit 16
3.1 Theory 18
3.2 Derivatives 19
3.2.1 Comparing performances of different derivatives methods 24
3.3 Matching the way mis extracted from the data 25
3.4 What is the true value of m? Removing ambiguity to maximize consistency. 26
3.4.1 Post-Processing Filters 26
3.4.2 Robust mtrueestimate 28
4 Systematic error budget on mfor cosmological inference from ShapeFit 28
4.1 Null tests 29
4.2 Tests on Standard cosmologies 30
4.3 Early dark energy 32
4.4 Systematic error budget on m: a recipe 32
5 Conclusion 34
A Smooth replacement 39
B Plateau Peak/Wiggle decomposition 39
C Splines 40
C.1 Cubic Spline 40
C.2 Univariate spline 41
D Power spectrum in w0-wacosmologies 41
– 1 –

1 Introduction
The power spectrum of galaxies has proven to be a treasure trove of cosmological information. It is
well known that the potential wells caused by the clustering of baryons and cold dark matter (CDM)
are fundamental for seeding the formation of galaxies and galaxy clusters. Therefore, the late-time
galaxy power spectrum captures a lot of information about the energy densities in the early universe
imprinted in current day galaxy positions, for example through the baryonic acoustic oscillations
(BAO) and its corresponding baryonic suppression of growth, as well as the overall broadband shape
related to the transition from radiation-dominated to matter-dominated growth and the spectral index
of the primordial fluctuation power spectrum.
Since its first detection in the early Two degree of Field Galaxy Redshift Survey (2dFGS) and the Sloan
Digital Sky Survey (SDSS) Luminous Red Galaxies (LRG) samples [1–3], the BAO signature has been
a staple of cosmological analyses from galaxy surveys, culminating in the SDSS Baryon Oscillation
Spectroscopic Survey (BOSS) and its extension (eBOSS) as well as the Dark energy Spectroscopic In-
strument (DESI) BAO analyses [4, 5], giving the most precise constraints on the Universe’s expansion
history from BAO to date. However, the BAO oscillations (‘wiggles’) are not the only information im-
printed in the power spectrum: its broadband shape encodes additional valuable information through
various effects. We schematically show the most notable features of the power spectrum in figure 1.
The turnover results from the transition between radiation-dominated and matter-dominated growth.
It occurs at scales which are slightly too large for current survey footprints, and can thus be measured
only with high statistical uncertainty [6]. While the non-linear scales are in principle accessible to
many current galaxy surveys, the limited understanding of non-linear effects restricts their use: the
maximum investigated wavenumbers often extend only mildly into the non-linear regime, limited by
the range of validity of perturbative or effective field theory approaches.
The grey band in figure 1 shows the investigated wavenumber range (for DESI) and the red rounded
square shows the non-linear enhancement of structure formation in the fully non-linear regime. Even at
the only mildly non-linear scales currently included in most analyses, the mode-coupling introduced by
non-linearities induces a suppression of the BAO feature (e.g., [7–11]). Most state-of-the-art analyses
use the effective theory of large scale structure (EFTofLSS) to model the non-linear power spectrum of
tracers (see [12, 13] for a review on EFTofLSS). To compute this suppression, for EFTofLSS modeling
it is useful and customary to decompose the power spectrum into the oscillatory (‘wiggly’) component,
and the broadband (‘de-wiggled’) component.
In addition to the above clearly visible features, there are two more subtle features imprinted in the
power spectrum. The logarithmic curvature of the power spectrum is related to the time of growth
between the horizon entry of a mode during radiation domination and the eventual start of matter
domination, leading to a weak dependence on the scale of matter-radiation equality. The baryonic
suppression on the other hand, is a clear feature after the turnover of the power spectrum and results
from the lack of growth of baryonic overdensities due to their involvement in the acoustic oscillations.
The baryonic suppression and the logarithmic curvature are measurable on large scales e.g., [14–16]
(where non-linear effects are irrelevant) and directly relate to the ratio of baryon to cold dark matter
in the early Universe.
In order to use these two features to constrain cosmological parameters, there are two main approaches:
fitting the full shape of the power spectrum (as in e.g., [17–21]) or compressing the information into
a single variable, such as with the ShapeFit approach developed and employed in Refs. [15, 22, 23].
While the former can in principle extract all information from the power spectrum, a cosmological
model needs to be assumed a priori , and therefore the data analysis needs to be completely re-done
for each cosmological model of interest. Instead, the latter approach is in principle independent of the
cosmological model: the analysis can be performed once and be re-interpreted for different models.
In addition, it is somewhat more interpretable due to the localization of the parameter constraints to
specific features in the power spectrum.
– 2 –

TurnoverBAO
DESInonlinearBaryonic
suppression
ShapeFitFigure 1 . A schematic representation of the relevant information imprinted in the matter power spec-
trum. The scales relevant to DESI correspond to 0 .02h/Mpc-0 .2h/Mpc [21], while we use the ShapeFit pivot
wavenumber of 0 .03h/Mpc [22].
Both approaches rely to some degree on the decomposition of the power spectrum into a wiggly and de-
wiggled component (either through their use of EFTofLSS for the full modeling of the power spectrum,
or for the computation of the ShapeFit parameter, see section 3.1 below). Therefore, the de-wiggling
of the power spectrum is an important part of any modern LSS analysis pipeline. It is not surprising
then that a vast number of de-wiggling methods have been proposed in the literature. A thorough
comparison of all proposed methods and a corresponding assignment of systematic uncertainties for the
de-wiggled power spectrum and the corresponding extraction of the shape parameter using ShapeFit
for a variety of cosmological models has, to our knowledge, not been performed before.
The paper is structured as follows. In section 2 we describe in detail the different de-wiggling methods
proposed in the literature and estimate a systematic uncertainty in the de-wiggled power spectrum.
The reader not too keen on technical details of the algorithms can skip sections 2.1 to 2.5 at first
sitting: the findings are summarized in section 2.6. In section 3 we then focus on different methods of
extracting the shape information using the ShapeFit formalism, and estimate a systematic uncertainty
on the ShapeFit slope parameter m. Once again the reader not to keen on technical details can skip
the first part of section 3.2 and jump directly to the comparison in section 3.2.1. In section 4 we then
test the systematic uncertainty for different cosmologies within and beyond the ΛCDM model.
We conclude in section 5.
2 Broadband/BAO decomposition
The separation (or decomposition) of the power spectrum (BAO) wiggles from the broadband shape
has been an important part of cosmological analyses since about two decades ago, when it was realized
that nonlinear structure formation affects the baryonic oscillation feature in a non-trivial way [24, 25].
The first methods were based on either semi-analytical fitting methods, such as the Eisenstein-Hu
transfer function [26, 27], or simple polynomial fits as in [28]. As interest in the analysis of the baryonic
features in the power spectrum grew, the approaches diversified. Today there is a large collection of
different methodologies for decomposing the power spectrum into baryonic oscillations (the ‘wiggly’
part) and the broadband shape (the ‘no-wiggle’ part). We discuss a large, nearly exhaustive collection
of methods in section 2 below, focusing on the latest implementations of any given method.
These de-wiggling methods have gained traction in particular for their use for accurately predicting
the nonlinear power spectrum in halo models [29–33] and effective field theories of large-scale structure
– 3 –

10−310−210−1100
k [1/Mpc]102103104105P(k)[Mpc]3
Pno−wiggle(k)
P(k)
Pref,no−wiggle(k)
Pref(k)
103
102
101
100
k [1/Mpc]0.04
0.02
0.000.020.04RatioP(k)/Pnowiggle(k)
Pref(k)/Pref,nowiggle(k)
Figure 2 .Left: Power spectra for the fiducial (black) and showcase (green) cosmology (see table 3 for
the corresponding cosmological parameters values), and their corresponding de-wiggled power spectra (blue
and red dashed lines, respectively). The example de-wiggling algorithm used here is the “Cubic Inflections”
algorithm of section 2.4. Right: Ratio of the power spectrum to the no-wiggle power spectrum, for both
cosmologies, highlighting the baryonic acoustic oscillations.
growth (EFTofLSS) [12, 13, 17–21]. These algorithms typically attempt to isolate the BAO wiggle
from the power spectrum, allowing one to compute a smoothed version of the power spectrum: they
remove the oscillations themselves but not the overall baryonic suppression. We show an example of
such a decomposition in figure 2, both for a fiducial/reference cosmology and a somewhat arbitrary
showcase cosmology (see table 3 for the parameters defining these cosmologies).
Recently, a new compressed parameter scheme “ShapeFit” has also been introduced [15, 22, 23], which
is based on determining the broadband slope at a wavenumber of interest. This approach also requires
efficient de-wiggling schemes, but robustly extracting the baryonic suppression requires special care
and we focus on such specific application in section 3.
De-wiggling methods common in the literature can roughly be divided into four main numerical
approaches. We schematically display these four approaches in figure 3 and list them below:
1.Smoothing: Just numerically smoothing the oscillations can give viable results if the smooth-
ing kernel width is approximately a full oscillation wavelength. At that point the oscillation
integrated over the smoothing kernel approximately cancels out, leaving just the broadband
shape. This method is discussed in section 2.2.
2.Fitting: Based on fitting a smooth function through the oscillations. The BAO oscillations are
assumed to be symmetric around the brodband, meaning that the amplitude of the oscillation
above and below the broadband should balance out. When using a least-square fitting method,
the BAO oscillations are forced to be symmetric and this indirectly determines the broadband.
This method is discussed in section 2.3.
3.Inflections: Based on constructing a smooth function passing through the inflections of the
oscillations. The inflection points usually coincide with the zero-point of the oscillations, and
those correspond to points where the wiggly power spectrum crosses the smooth broadband
spectrum. This method is discussed in section 2.4.
4.Peak removal: Instead of working with the power spectrum, the correlation function is used,
for which the BAO oscillation feature is known to be well localized. After removing this local
feature (for example by just fitting a smooth function through the surrounding scales) the
resulting correlation function can be transformed back into a power spectrum without the BAO
wiggles. This method is discussed in section 2.5.
– 4 –

InflectionsFitting
Smoothing
Peak removal
Powerspectrum Correlation functionFigure 3 . Schematic overview of de-wiggling methods. The red line corresponds to the combined broad-
band spectrum with additional wiggles (as a power spectrum on the left, and as a correlation function on
the right) while blue corresponds to the dewiggled broadband. The green/yellow/orange lines and points
show how the corresponding de-wiggled broadband function is obtained. For the “Smoothing” method, the
yellow/orange/green lines show sufficiently long smoothing windows at certain locations across the function,
whose averages are correspondingly shown as yellow/orange/green points, through which a smooth function
is drawn. For the “Fitting” method, the green/orange lines show the mean squared deviation that has to be
minimized in any approach that fits a smooth function to the oscillations. For the “Inflections” method, the
green point show the inflections of the oscillations (after subtracting a rough broadband trend) through which
the blue function is fitted. Finally, for the “Peak removal” method, the correlation function is fitted where
there is no BAO peak by a smooth function, resulting in the blue curve directly.
We discuss each dewiggling method individually in the following subsections, and summarize their
performances in table 1. The reader not interested in the details of the individual algorithms may
skip to the summary in section 2.6.
We use the class code [34] for generating the linear power spectra, and use our custom python code to
implement the different de-wiggling methods as well as to convert the power spectrum to a correlation
function. The corresponding code will be made public upon acceptance of this paper. For early dark
energy cosmologies, we use instead the AxiCLASS code from [35, 36].
2.1 On using analytical formulae
An analytic fitting formula for the shape of the transfer function (in the absence of massive neutrinos)
is provided in [27], both for the cases when the BAO wiggles are present and when they are neglected.
We refer to it as EH98 . This function has also been extended in the context of massive neutrinos
in [26]. The advantage of this approach is that this model is analytical. However, the fitting formula
was designed for galaxy surveys from the 2000s and is not suitable for the precision of present-day
galaxy surveys. Additionally, in practice the model cannot be used beyond the most commonly
discussed cosmological models – for many cases no generalization exists. This is a generic problem
for any kind of analytical fitting formula – its usage will be restricted only to those models for which
it has been explicitly constructed.
While we do not further consider the use of the analytical formulae to model the de-wiggled power
spectrum, we recognize that the EH98 formula can still be helpful when smoothing as it typically
captures the overall shape and order-of-magnitude of the true power spectrum, see below. Moreover,
the general form can still be helpful in a semi-analytic approach whereby the individual (otherwise
analytically determined) coefficients of the formula are numerically fit instead, see section 2.3.
– 5 –

2.2 Numerical smoothing
Given that the oscillations extend equally above and below the broadband, one can expect a simple
numerical smoothing method to smooth out these oscillations. The expectation is that, averaging the
power spectrum over a large enough distance, the oscillations are averaged out, while the broadband
remains at least approximately intact.
10−310−210−1100
Wavenumber k[1/Mpc]−0.100−0.075−0.050−0.0250.0000.0250.0500.0750.100RatioMethod: [Smoothing] Gaussian smoothing
P(k)/Pno−wiggle(k)
Pref(k)/Pref,no−wiggle(k)
Figure 4 . The ratio of the power spectra divided by the dewiggled power spectra for the simple Gaussian
smoothing method applied to the fiducial/reference (black) and showcase (green) cosmologies. The dashed grey
line represents the pivot wavenumber kpat which the ShapeFit slope (see section 3) is evaluated (translated
from h/Mpc to 1/Mpc).
TheSimple Gaussian method, proposed in [37], involves a simple convolution of the power spectrum
with a Gaussian function of the type
f(k, k0) =1√
2πλ2exp
−(lnk−lnk0)2
2λ2
, (2.1)
yielding a broadband de-wiggled power spectrum Pno−wiggle(k0) =R
f(k, k0)P(k)dlnkwhere λis the
width of the Gaussian, which is set to λ= 0.25/Mpc. We show the resulting wiggle/no-wiggle split in
figure 4. Since the averaging length is somewhat small, we do not see a large impact on the broadband
shape, while the oscillations are mostly removed.
The main disadvantage of this method (and similar ones) is that small wiggles can remain in the
power spectrum even after the averaging; if the smoothing scale is increased too much, the broadband
begins to be strongly affected.
2.3 Fitting smooth functions
The idea of this approach is to fit sufficiently smooth functions to the overall power spectrum. Such
functions are optimally able to fit the overall broadband shape but lack enough freedom to also follow
the oscillations. If the oscillations cannot be fit, a maximum likelihood estimation (like the least-
square difference algorithm) will typically balance the amplitude of the oscillations below and above
the fit, which is a good approximation of the true broadband. The big issue that these methods face
is that typically the broadband spectrum is difficult to model with simple elementary functions at
sufficient accuracy, but just going to higher-order expansions risks starting to fit the oscillations as
well. We discuss how the specific algorithms avoid this problem for each algorithm below.
– 6 –

Runge
phenomenonFigure 5 . Same as figure 4 but for the polynomial fitting method. The red line represents the weights used
for the fit according to equation (2.2). The blue region highlights the scales in which Runge’s phenomenon
becomes relevant.
Polynomial fit This method has been initially proposed in [38, app. A] for the WiggleZ survey.
The final selected method involved fitting a polynomial of degree nto the power spectrum. Of course
a high-order polynomial will naturally also fit the oscillations. To counteract this issue, the authors
down-weighted the wavenumbers according to the formula
w= 1−αexp[−k2/(2σ2)], (2.2)
which for a fine-tuned choice of n, the Gaussian width σ, and the weighting amplitude α, down-
weights the region containing the BAO oscillations (and the larger scales) compared to the overall
broadband fit. We choose the same parameters as [38, app. A], namely n= 13, σ= 1/Mpc, α= 0.5
and fit the polynomial in log-log space. We have checked that up-weighting smaller wavenumbers in
equation (2.2) below the BAO oscillations does not yield any significant improvement.
The resulting wiggle component of the reference and showcase power spectra (as per table 3) are
shown in figure 5. The polynomial representation introduces unphysical oscillations even where no
BAO are present due to the well known Runge-phenomenon [39]: polynomial approximations often do
not converge towards the true function but are subject to a type of ‘ringing’ around the true function.
The corresponding wavenumbers for which this is relevant are highlighted in blue in figure 5. Increasing
the polynomial order to reduce these ringing artifacts, however, also allows the polynomial to track the
physical BAO oscillations. Therefore we do not consider this method as a good de-wiggling method
(see also table 1).
Cubic Spline fit The authors of [40, app. B] propose a simple approach of fitting a cubic spline
function (see appendix C.1) to the power spectrum, choosing only a small number of points ( ∼20) to
the right and left of BAO scales, as well as an additional pivot point at k= 0.03h/Mpc. Using cubic
interpolation between the pivot point and the left/right sides beyond the BAO ensures that the wiggles
cannot be traced. Defining exactly where the BAO start and end is crucial for this approach and
the choice depends on cosmology. In our case, we select for the fiducial model the region [10−2/Mpc,
0.45/Mpc] and rescale it by rd/rfid
dfor other models.
Such a cubic spline approach is very similar to the one originally adopted in [28], using different node
points. We also implement the method of [28] for reference (which uses a selection of 8 node points
that lie mostly inside the BAO region). Using just these 8 points does not result in a good overall
fit. Instead we also include 20 points in the region outside the BAO interval [10−3/Mpc, 1/Mpc] in
the fit in order to force the de-wiggled power spectrum to coincide with the original power spectrum
there.
– 7 –

10−310−210−1100
Wavenumber k[1/Mpc]−0.100−0.075−0.050−0.0250.0000.0250.0500.0750.100RatioMethod: [Fitting] Cubic Spline ﬁt (Reid et al.)
P(k)/Pno−wiggle(k)
Pref(k)/Pref,no−wiggle(k)
10−310−210−1100
Wavenumber k[1/Mpc]−0.100−0.075−0.050−0.0250.0000.0250.0500.0750.100RatioMethod: [Fitting] Cubic Spline ﬁt (Blas et al.)
P(k)/Pno−wiggle(k)
Pref(k)/Pref,no−wiggle(k)Figure 6 . Same as figure 4, but for the cubic fitting methods. Left: The method adopted in [28], with the
modification of also including additional points left and right of the interval [10−3/Mpc, 1/Mpc]. Right: The
method adopted in [40, app. B].
As evident in figure 6, neither method is entirely satisfactory. With our implementation of the
algorithm of [28] (left panel), the fit is forced to have the linear and de-wiggled power spectra coincide
at all 8 nodes within the BAO region, resulting in a highly distorted fit there (and small shallow
residual wiggles in the no-wiggle power spectrum) – the amplitude of the wiggle/no-wiggle ratio does
not follow the expected shape of the BAO. It is likely that the choice of nodes in [28] was optimized
for a particular cosmology and would have to be changed when considering any other cosmology.
On the other hand, the approach of [40, app. B] using only a single pivot point in the BAO region
works significantly better. Here too, however, the result is highly dependent on the interval chosen
to contain the BAO wiggles, as well as the number of points outside the BAO region. This is
also true in particular for the subtraction of the first peak and therefore the slope maccording to
equation (3.1). For example, doubling the number of points from 20 to 40 outside the BAO region
gives slope differences |∆m|up to 0.02 and moving the left edge of the BAO region from 10−2/Mpc
to 5·10−2/Mpc gives |∆m|up to 0.05 (see also section 3.2).
B-Spline fit In [37, app. A] the authors provide several different de-wiggling methods, one of which
is based on a fit with B-splines (generalizing the cubic spline to higher polynomial degrees dand a
different number of knots nk, see appendix C.2 for more details). We show the case of d= 2, nk= 8
in the top panels of figure 7.
The authors of [37, app. A] combine the spline approximations from multiple combinations of d, nk,
imposing additional conditions in the weighed sum. Here for simplicity we use equal weights. The
bottom panels of figure 7 correspond to the case when we include d={2,3,4,5}with nk={2d+
4,2d+ 6}and simply average them with equal weights.
For best results, we find that the B-spline fit must be restricted to a fitting region (which includes the
BAO) between some kstartandkend. Since the B-spline is not guaranteed to be continuous with the
original power spectrum at the edges of the fitting interval, the method of appendix A must be applied
to ensure continuity. In particular the fitting region, [ kstart,kend], is padded with a transition region
defined by kmin< kstart, and kmax> kendwhere a “smooth replacement” according to appendix A is
performed. In the bottom panels of figure 7 we choose ln kmin= lnkstart−3∆ where ∆ = 0 .4 is the
“replacement width” of appendix A and ln kmax= lnkend+ 2∆.
Figure 7 compares wider and narrower B-spline fitting regions. For a single spline the method reacts
quite strongly to wider fitting regions (smaller kstart): the fit begins to diverge from the broadband for
the wider fitting region top right panel) – the oscillations are not symmetric around 0 in the ratio. In
– 8 –

10−310−210−1100
Wavenumber k[1/Mpc]−0.100−0.075−0.050−0.0250.0000.0250.0500.0750.100RatioMethod: [Fitting] Single B-Spline
P(k)/Pno−wiggle(k)
Pref(k)/Pref,no−wiggle(k)
Transition weights ( ·10−1)
10−310−210−1100
Wavenumber k[1/Mpc]−0.100−0.075−0.050−0.0250.0000.0250.0500.0750.100RatioMethod: [Fitting] Single B-Spline
P(k)/Pno−wiggle(k)
Pref(k)/Pref,no−wiggle(k)
Transition weights ( ·10−1)
10−310−210−1100
Wavenumber k[1/Mpc]−0.100−0.075−0.050−0.0250.0000.0250.0500.0750.100RatioMethod: [Fitting] Averaged B-Spline
P(k)/Pno−wiggle(k)
Pref(k)/Pref,no−wiggle(k)
Transition weights ( ·10−1)
10−310−210−1100
Wavenumber k[1/Mpc]−0.100−0.075−0.050−0.0250.0000.0250.0500.0750.100RatioMethod: [Fitting] Averaged B-Spline
P(k)/Pno−wiggle(k)
Pref(k)/Pref,no−wiggle(k)
Transition weights ( ·10−1)Figure 7 . Same as figure 4, but for spline fitting methods. The red lines denote the weights: the fitting
region corresponds to the flat portion of the red line. Top: The method for a single spline. Bottom: The
method for the average of multiple splines. Left: A narrow replacement window (which would impact the
slope at kp). Right: A broad replacement window (which would not impact the slope at kp).
contrast, the average of multiple splines remains mostly stable even with a much wider fitting window
(bottom right panel). For further comparisons, we choose a combination of the more robust average
of multiple splines together with a wider fitting region (for which the wiggle/de-wiggle ratio at the
pivot wavenumber of section 3 is not impacted by the replacement method of appendix A).
Univariate spline fit This idea is simply based on fitting the overall power spectrum with a
univariate spline, the concept of which is explained in detail in appendix C.2. For this case we weigh
the different parts of the power spectrum according to weights that are unity outside the BAO range,
and suppressed by a factor w= 0.5 inside the BAO range1and use a smoothing strength s= 0.01.
Finally, we only employ the de-wiggling algorithm in the aforementioned range, using the technique
of appendix A with ∆ = 0 .4 to ensure a smooth transition.
In figure 8 (left panel) we show the resulting wiggle/no-wiggle decomposition for the power spectrum.
There is not a strong dependence on the precise covered wavenumber range. However, there is a
strong dependence on the suppression weight w, as we show in the right panel of figure 8.
EH fit The authors of [40, app. B] also propose an approach based on the EH98 formula as a fitting
function, by generalizing its parameter dependencies, as well as including an additional correction.
1We take the range here to lie between kstart = 0.01/Mpc and kend= 0.3/Mpc for the fiducial cosmology, rescaled
byrd/rfid
dfor other cosmologies.
– 9 –

10−310−210−1100
Wavenumber k[1/Mpc]−0.100−0.075−0.050−0.0250.0000.0250.0500.0750.100RatioMethod: [Fitting] Univariate spline
P(k)/Pno−wiggle(k)
Pref(k)/Pref,no−wiggle(k)
weights (·10−1)
Transition weights ( ·10−1)
10−310−210−1100
Wavenumber k[1/Mpc]−0.100−0.075−0.050−0.0250.0000.0250.0500.0750.100RatioMethod: [Fitting] Univariate spline
Pref(k)/Pref,no−wiggle(k) (w=0.5, default)
P(k)/Pno−wiggle(k) (w = 0.3)
P(k)/Pno−wiggle(k) (w = 0.1)
P(k)/Pno−wiggle(k) (w = 0.9)Figure 8 . Same as figure 4, but for a univariate spline fitting method. Left: The wiggle/no-wiggle ratio for
the univariate spline fitting method applied to a given model and fiducial power spectrum. In red we show
the used weights, and in blue the weights used for the replacement scheme of appendix A. Right: Variations
on the fiducial cosmology for different weight suppressions wranging from w= 0.1 tow= 0.9.
10−310−210−1100
Wavenumber k[1/Mpc]−0.100−0.075−0.050−0.0250.0000.0250.0500.0750.100RatioMethod: [Fitting] EH98 ﬁt version 2
P(k)/Pno−wiggle(k)
Pref(k)/Pref,no−wiggle(k)
Figure 9 . Same as figure 4, but for the EH fitting method.
By slightly re-formulating their formula to be closer to the original proposal of [26, 27],2our fitting
formula reads
P(k)≈AknT(k)2(1 + ∆( k)) (2.3)
T(k) =L(k)·[L(k) +C(k)κ2]−1(2.4)
L(k) = ln( e+p
d1d2κ) (2.5)
C(k) = 14 .4 + 325 /(1 + 60 .5κ1.08) (2.6)
κ=d3k·p
d1+(1−√d1)
1 + (d4k)4−1
(2.7)
∆(k) =d5tanh (1 + log( d6κ)/d7) (2.8)
with the parameters A,n, and dibeing fitted (in log-log space) to the power spectrum.
2In particular, we have re-parameterized c1=√d1d2,c2=d3/√d1, and c3= (1−√d1)/√d1, and c≥4=c≥4(where
theciare the coefficients in [40, app. B]). Given that the mapping between the {ci}and{di}is one-to-one, we don’t
observe any differences between the formulations.
– 10 –

The resulting wiggle/no-wiggle decomposition is shown in figure 9, and the wiggles seem to be well
characterized. There are some very small residual oscillations for k <0.01/Mpc, but this is largely
irrelevant for most applications.
2.4 Inflections
The algorithms in this category are based on fitting a smooth function through the inflection points
of the oscillations to determine the broadband behavior. The idea is that in the absence of the
broadband component the inflection points coincide with the zero-points of the oscillations. Although
conceptually simple, the actual implementation must carefully avoid a number of common pitfalls.
In particular, the inflection points of the power spectrum are typically biased with respect to the
true zeros of the BAO due to the broadband slope.3This creates a circular problem: to accurately
determine the zero-points of the oscillations (to remove the wiggles), the de-wiggled power spectrum
already needs to be known. The circularity problem is typically solved by first adopting approxima-
tions of the broadband, enabling a nearly un-biased determination of the zeros of the oscillations,
which in turn allows for the determination of the true broadband.
In addition one must ensure that the numerically determined inflections are robust with respect to
the finite wavenumber sampling of the provided power spectrum.
One of the first implementations4(which we dub “Cubic Inflections”) is implemented by default in
theMontePython code [41, 42], while a second version (which we dub “EH inflections”)5is one of the
original options in ShapeFit (see [22]).
Both algorithms use estimates of the broadband to get approximate zero-points of the oscillations
by dividing of the true function by the approximation (assuming the oscillations are multiplicative).
The “Cubic Inflections” algorithm uses a cubic spline approximation of the overall power spectrum as
an approximate broadband, while the “EH inflections” algorithm uses the EH98 formula. The “EH
inflections” algorithm also uses a simpler helper algorithm, dubbed here “Gradient inflections”6. In
all cases, the range containing the BAO features, [ kstart, kend], is an input. A suitable range containing
the BAO is estimated by multiplying a fiducial range appropriately with the sound horizon scale.
We present and compare these algorithms below.
Cubic Inflections This algorithm uses the following steps:
1. A cubic spline is used to fit the power spectrum outside the range containing the BAO features,
essentially providing a smooth cubic approximation in the BAO region and a close-to-exact
approximation outside of it.
2. The ratio of the power spectrum and the smooth cubic approximation is used to define approx-
imately what should constitute as the wiggles.
3. The second derivative of the wiggles is then interpolated using a cubic univariate spline (see
appendix C.2) of smoothing strength s= 1. This essentially fits the second derivatives with a
function that ensures a certain degree of smoothness as in principle the second derivative of a
Cubic spline function can be quite discontinuous between the different knots (see appendix C.1).
3To see this very quickly, consider a function f(x) to consist of an oscillation plus some broadband f(x) = sin( x)+b(x).
If the broadband is un-curved b(x) =ax+b, then the inflection points xn=nπ(determined by f′′(xn) = 0) coincide with
the zeros of the oscillations. Therefore, the function evaluated at the inflection points follows the broadband f(xn) =
b(xn). If one now has a curved broadband instead, b(x) =γx2, then the inflection points are x2n= 2nπ+ arcsin 2 γ
andx2n+1= 2nπ+ [π−arcsin 2 γ] and therefore do not coincide with the zeros of the oscillations (which are still
nπ, of course). Therefore, the function evaluated at the inflection points does not follow the broadband, for example
f(x2n) =b(x2n) + sin(arcsin(2 γ) + 2nπ) =b(x2n) + 2γ̸=b(x2n).
4Due to Mario Ballardini
5Implemented by Samuel Brieden for ShapeFit[22].
6A more recent option in ShapeFit also provided by Samuel Brieden.
– 11 –

10−310−210−1100
Wavenumber k[1/Mpc]−0.100−0.075−0.050−0.0250.0000.0250.0500.0750.100RatioMethod: [Inﬂections] Cubic Inﬂections
P(k)/Pno−wiggle(k)
Pref(k)/Pref,no−wiggle(k)
10−310−210−1100
Wavenumber k[1/Mpc]−0.100−0.075−0.050−0.0250.0000.0250.0500.0750.100RatioMethod: [Inﬂections] Cubic Inﬂections
Pref(k)/Pref,no−wiggle(k) (f=0.01, default)
P(k)/Pno−wiggle(k) (f = 1)
P(k)/Pno−wiggle(k) (f = 0.6)
P(k)/Pno−wiggle(k) (f = 0.05)Figure 10 . Same as figure 4, but for the Cubic Inflections method. Left: The wiggle/no-wiggle ratio for
the cubic inflections method applied to a given model and fiducial power spectrum. Right: Variations with
different starting wavenumber wavenumbers (original wavenumber multiplied by f≤1).
4. The zeroes of the smoothed second derivative of the wiggles are calculated. These are the
inflection points of the oscillations. Note that any zeros outside the BAO region are removed.
5. A spline is fitted through the inflection points as well as the wave numbers from regions outside
the BAO range, giving a smooth function (which captures the deviation of the true broadband
from the cubic approximation).
6. The cubic approximation is then multiplied by this smooth function to recover the true broad-
band.
We show the results of this algorithm in figure 10. As implemented here, we do not use the default
settings of MontePython code [41, 42] whereby the beginning of the BAO region kstart= 0.028/Mpc
coincides with the turnover of the power spectrum. This would force the wiggle-to-nowiggle ratio to
be zero at this wavenumber. Here we allow for additional freedom to kstartthrough a multiplicative
rescaling factor f <1. Additionally, we scale the starting/ending wavenumbers of the BAO region
proportionally to the sound horizon of the given cosmology.
This choice is motivated by the fact that the peak of the power spectrum coincides with the first BAO
wiggle (see also appendix B), so allowing the first peak to also be removed can be a reasonable choice.
We show in figure 10 (right panel) that indeed such an approach converges for f→0.01, which we
therefore take as our fiducial value.
Gradient Inflections This is a helper algorithm which is used for the “EH inflections” methods
discussed below. Given a properly normalized wiggly spectrum, instead of determining the zeros of
the second derivatives (as above), this method attempts to find the peaks of the gradient. The steps
are as follows:
1. The derivative of the power spectrum is computed with a naive forward difference method.
2. The maxima and minima of this gradient are computed (and only those in the predetermined
BAO range are kept), effectively giving the (possibly biased) zeroes of the wiggles.
3. The parts before and after the BAO region as well as the intermediate power spectrum at the
location of the zeros of the wiggles are interpolated separately for the maxima and the minima
of the gradient, and the two functions are averaged.
– 12 –

10−310−210−1100
Wavenumber k[1/Mpc]−0.100−0.075−0.050−0.0250.0000.0250.0500.0750.100RatioMethod: [Inﬂections] Gradient inﬂections
P(k)/Pno−wiggle(k)
Pref(k)/Pref,no−wiggle(k)
10−310−210−1100
Wavenumber k[1/Mpc]−0.100−0.075−0.050−0.0250.0000.0250.0500.0750.100RatioMethod: [Inﬂections] EH inﬂections
P(k)/Pno−wiggle(k)
Pref(k)/Pref,no−wiggle(k)
10−310−210−1100
Wavenumber k[1/Mpc]−0.100−0.075−0.050−0.0250.0000.0250.0500.0750.100RatioMethod: [Inﬂections] EH inﬂections version 2
P(k)/Pno−wiggle(k)
Pref(k)/Pref,no−wiggle(k)Figure 11 . Same as figure 4, but for the different inflections methods. Top: Just the ’gradient inflections’
algorithm. Bottom left: The first version of the ’EH inflections’ algorithm. Bottom right: The second version
of the ’EH inflections’ algorithm.
There are two different implemented possibilities for removing the broadband from the power spectrum
together with the “Gradient Inflections” method to remove the remaining wiggles. We discuss these
two methods below.
EH Inflections For this method the “Gradient Inflections” algorithm is applied to the ratio of
the power spectrum and the EH98 formula. The idea is that the rescaled EH98 formula might not
precisely reproduce the power spectrum (or its amplitude), but it reproduces the broadband slope
closely enough to give an almost unbiased estimate of the inflection points.
1. First, for a chosen fiducial cosmology the power spectrum, Pfid(k), is divided by the one given
by the EH98 formula EH98fid, and the “Gradient Inflections” method is applied to this ratio in
order to give a very rough approximate fiducial broadband power spectrum. See equation (2.9a)
where G[x] is the “Gradient Inflections” method. Hence the ratio Pno−wiggle ,fid(k)/EH98fid(k)
defines a EH98-to-broadband correction factor for the fiducial cosmology.
2. For evaluations for any other cosmology, the power spectrum is divided both by the EH98 model
and by the fiducial EH98-to-broadband correction factor. Therefore, this ratio will contain the
wiggles over an approximately flat baseline (the flatter the closer to the fiducial cosmology).
Applying the “Gradient Inflections” method to this ratio then removes these wiggles with quite
high accuracy. The final result is multiplied back by the fiducial no-wiggle power spectrum to
restore the correct amplitude, see equation (2.9b).
– 13 –

Pno−wiggle ,fid(k) =GPfid(k)
EH98fid(k)
·EH98fid(k) (2.9a)
Pno−wiggle(k)≈Pno−wiggle ,fid(k)·GP(k)/EH98( k)
Pno−wiggle ,fid(k)/EH98fid(k)
(2.9b)
EH Inflections version 2 The second version of the same algorithm differs only by a few small
improvements. Equation (2.9a) remains the same, see equation (2.10a). However, compared to
equation (2.9b), in equation (2.10b) the the wavenumbers are multiplied by s=rfid
d/rdto shift the
power spectrum and align the BAO wiggles with those of the fiducial power spectrum. Thus the ratio
P(k/s)/Pfid(k) is already mostly smooth. Normalizing this by the ratio of the EH98 transfer functions
EH98( k)/EH98fidyields a function optimally close to unity with only small residual differences arising
from a) imperfect compensation of the terms and b) the general broadband shape deviation from the
EH98 formula. The “Gradient Inflections” algorithm is then applied to this quantity. The overall
correct amplitude and shape are then recovered by multiplying with the corresponding terms as shown
in equation (2.10b).
Pno−wiggle ,fid(k) =GPfid(k)
EH98fid(k)
·EH98fid(k) (2.10a)
Pno−wiggle(k)≈
GP(k/s)/EH98( k)
Pfid(k)/EH98fid(k)
·EH98( k)
EH98fid(k)
k=ks·Pno−wiggle ,fid(ks) (2.10b)
The results for each of the algorithms are displayed in figure 11. Interestingly, it is obvious that the
BAO oscillation coinciding with the first peak is not being removed (the ratio is close to zero around
and before kp). This means that the broadband shape is possibly only recovered after the first peak.
2.5 Correlation function peak removal
The final type of wiggle-removal technique discussed here uses the fact that in the correlation function
the BAO peak is well localized and therefore typically easier to remove. The transformation between
power spectrum and correlation function can be generally written as [43]
ξ(r) =Z
dlnksin(kr)
krP(k)k3
2π2
(2.11)
with the inverse transform
P(k) =Z
dlnrsin(kr)
kr 
4πξ(r)r3
(2.12)
This type of transformation can either be interpreted as a sine transform or as a Hankel transform [44],
since the Bessel function of the first kind has J1/2(x) = sin( x)·p
2/(πx).
The challenge for this method is to achieve sufficient accuracy in the transformations equations (2.11)
and (2.12) without sacrificing speed which is achieved by utilizing implementations based on the fast
Fourier transform.
Hankel transform In [45, Sec. 2.2.1] a wiggle-removal algorithm was presented for the BOSS
survey. We implement it here by logarithmically sampling k, computing k3/2P(k), and obtaining the
corresponding correlation function via the fast Hankel transform. We then fit a linear combination of
rkwith k∈ {− 3,−2,−1,0,1}to the correlation function between pre-defined bounds outside of the
peak range, such as 50 −86Mpc and 150 −190Mpc. The resulting interpolated correlation function is
then transformed back into a power spectrum using equation (2.12). Naturally the range of the scales
that lie outside the peak range need to be adjusted to the cosmology (this could be done automatically,
– 14 –

10−310−210−1100
Wavenumber k[1/Mpc]−0.100−0.075−0.050−0.0250.0000.0250.0500.0750.100RatioMethod: [Correlation function] Hankel transform
P(k)/Pno−wiggle(k)
Pref(k)/Pref,no−wiggle(k)
Transition weights ( ·10−1)Figure 12 . Same as figure 4, but for the Hankel transform method.
although for our tests a manual scaling by a factor of 1 /s=rd/rfid
dworks well enough). To avoid
high-frequency oscillations induced by the discrete sampling in the transformation, we smooth the
final result with a univariate spline (see appendix C.2) with s= 10−2. Finally, we replace the original
power spectrum according to appendix A, taking kmax→ ∞ ,kmin= 10−2/Mpc and using a shorter
width of ∆ = 0 .2.
The problem for this method is that while the higher peaks are correctly captured, the oscillations
at and below the first peak diverge (at least in the present numerical implementation) – and we note
that this algorithm was not necessarily designed to capture the P(k) at smaller wavenumbers than
the first peak, see also the top left panel of [45, Fig. 2].
10−310−210−1100
Wavenumber k[1/Mpc]−0.100−0.075−0.050−0.0250.0000.0250.0500.0750.100RatioMethod: [Correlation function] Fast sine transform
P(k)/Pno−wiggle(k)
Pref(k)/Pref,no−wiggle(k)
Figure 13 . Same as figure 4, but for the Fast sine transform method.
Fast sine transform This algorithm is one of the most frequently used in cosmology. It has been
proposed already in [46] and continued to be developed in the subsequent years, for example in [47,
App. D] or recently in [48, Sec. 4.2] (this is the version we use), and represents the state-of-the-art,
as it is used for example in [49, 50] for the DESI analysis. Ref. [50] also includes a comparison to
a subset of the other algorithms presented in this work (the EH Inflections andPolynomial fit
algorithms).
Following [48, Sec. 4.2] we sample ln( kP(k)) logarithmically between kmin= 7·10−5/Mpc and kmax=
7/Mpc with 216points. Then a fast discrete sine transform is used, of which the even and odd parts
– 15 –

are fit separately with linear combinations of rkwith k∈ {1,2,3,4,5,6}on two ranges of scales that
exclude the peak: 50 −120Mpc and 240 −500Mpc in this case.7We additionally tilt the even and odd
transforms by r1.3andr1.5, respectively, during the fitting of the linear combination of powerlaws,
to up-weigh important features. Finally, the resulting polynomial fits – now with the peak removed
– are transformed back (after removing the r1.3andr1.5tilt) and can be directly used as the power
spectrum P(k).
We show the resulting wiggle/no-wiggle ratio in figure 13. It is evident that a large peak is present at
wavenumbers around 10−2/Mpc and smaller. Such a peak can be identified with the plateau of the
BAO towards k→0. Whether that plateau should be identified as a ‘peak’ or broadband is a matter
of definition, as we demonstrate in appendix B using a toy example.
2.6 Summary
We summarize our observations for the different methods in table 1. In particular, we observe that a
few methods are not very well suited for the oscillation/broadband decomposition in the context of a
consistent numerical study of the BAO, for example due to a strong dependence on hyperparameters
or unsatisfactory removal of the wiggles. Therefore, we single out a ‘golden sample’ of six of the
most promising de-wiggling methods, including the Simple Gaussian ,B-spline fit (with averaging
over multiple splines), EH fit ,Cubic Inflections ,EH inflections (version 2) , and Fast sine
transform methods.
The golden sample outlined above is selected on the basis of being useful for applications concerned
with the full BAO range and for recovering the broadband shape in the full wavenumber range which
is typically of interest for galaxy surveys. However, for specific applications focused on specific scales
or features, another selection might be more optimal. For this reason we keep considering the entire
collection of thirteen methods until section 3.2.1 at which point we focus on these six most robust
ones.
All thirteen de-wiggling methods are compared in figure 14, where we show the mean and median
of the different methods as well as their standard deviation as an orange band around the median.
Compared to this median, we find deviations up to around 2%, strongly dependent on the scale, and
particularly large at either side of kp= 0.03h/Mpc. This can be understood as follows. The first peak
of the BAO coincides with the turnover of the power spectrum due to the closeness of the baryon drag
and matter-radiation equality times. Therefore what is considered the first BAO peak and what is
considered the turnover of the power spectrum is not uniquely defined (see appendix B for an explicit
toy example). This ambiguity is what causes the marked differences among the different de-wiggling
methods exactly around this scale.
This issue persists virtually unaltered also in the “gold sample”: these methods also do not treat the
first BAO/power spectrum turnover decomposition consistently. We will return to this in section 3.4,
as it has important consequences for the interpretation of ShapeFit results.
3 ShapeFit
ShapeFit, first introduced by [22], is a new approach to analyze the power spectrum that is quickly
gaining popularity in the cosmology community (see also [15, 22, 23] for more details on the method,
which involves computing a derivative of the large-scale de-wiggled power spectrum). This framework
bridges the standard approaches of BAO and redshift space distortions (RSD) with the full-modeling
approach. While the BAO and RSD are easily interpretable and can be used to isolate the features of
the power spectrum from which the constraints are coming from, the full modeling approach extracts
the most information from the power spectrum. ShapeFit can constrain parameters almost as tightly
7This is different to the Hankel transform above due to different tilt and different transformation. However, we also
perform the appropriate scaling with 1 /s=rd/rfid
din this case.
– 16 –

Method name Used for Reason
[Citation] gold sample for rejection
Analytical [26, 27] No ✗Inflexible for non-trivial cosmologies
Simple Gaussian [37] Yes ✓ —
Polynomial fit [38, app. A] No ✗ Runge-phenomenon at relevant k
Cubic Spline fit [28] No ✗Clear distortions even in BAO range
Cubic Spline fit [40, app. B] No ✗ Strong hyperparameter dependence
B-spline fit [37, app. A] Yes ✓ —
Univariate spline fit [this work] No ✗ Strong hyperparameter dependence
EH fit [40, app. B] Yes (only ΛCDM) ✓ Fails for non-trivial cosmologies
Cubic Inflections [41, 42] Yes ✓ —
EH inflections [22] No ✗ Updated version exists
EH inflections (version 2) [22] Yes ✓ —
Hankel transform [45, Sec. 2.2.1] No ✗ Divergence at small k
Fast sine transform [48, Sec. 4.2] Yes ✓ —
Table 1 . Summary of the methods of de-wiggling and corresponding selection for the gold sample or reason
for not including it. The “EH fit” method is included in the gold sample, but cannot be applied to all
cosmologies, see section 3.
10−410−310−210−1100101
Wavenumber k[h/Mpc]1.61.82.02.22.42.62.83.03.2P(k) ratio (shifted)No de-wiggling
Mean de-wiggling
Median de-wiggling
standard deviation of de-wiggling
10−410−310−210−1100101
Wavenumber k[h/Mpc]0.970.980.991.001.011.021.03P(k) ratio (shifted, relative to median)
Figure 14 . Ratio of the (shifted) power spectra for the computation of equation (3.1), using the fiducial and
showcase cosmologies of table 3. Left: Absolute ratio, showing the ratio without de-wiggling, and for all the
de-wiggling methods the mean, median, and the 10-90% quantile range. Right: Relative ratio, normalized by
the median. The dashed vertical grey line represents the pivot wavenumber kpat which the ShapeFit slope
(see section 3) is evaluated.
as the full-modeling approach while retaining the model-agnostic and interpretable nature of the stan-
dard BAO/RSD approaches. These properties make this approach a promising tool for current and
future galaxy surveys. When combined with the BAO+BBN data, ShapeFit can provide an addi-
tional constraint on the Hubble constant H0by isolating information from matter-radiation equality;
for a more detailed discussion, see [51]. ShapeFit relies on the broadband/BAO decomposition (as
studied in section 2) and involves the evaluation of a slope: a derivative of the numerically-evaluated
broadband component of the power spectrum. The evaluation of the derivative requires some care.
In practice, the differences between different de-wiggling methods as well as different ways of obtaining
the derivative will combine to impart a systematic uncertainty on the value of the ShapeFit slope
parameter m, which is a priori not straightforward to evaluate. In addition, systematic biases in m
do not necessarily propagate to a systematic shift in the recovered cosmology. Therefore, we proceed
here as follows. First, in section 3.1 we introduce the theoretical concepts at the heart of ShapeFit
– 17 –

and in section 3.2 we discuss in detail different implementations of numerical derivatives. Second, in
section 3.3 we stress the importance for cosmological inference of matching the way mis obtained from
the data. In section 3.4 we then discuss how to mitigate the underlying ambiguity in the definition
ofmand argue that it is possible to find a way of calculating the slope mthat is highly consistent
between different ways of numerically implementing the required derivative. Finally, in section 4
we recommend a procedure to obtain a robust and consistent mvalue from a given theory power
spectrum and associate to it a systematic error budget.
3.1 Theory
The ShapeFit parameter mis effectively defined to be the slope of the underlying transfer function8
for a given power spectrum at a pivot wavenumber kp, compared to that of an arbitrarily chosen
reference cosmology (ref). This ratio is mainly sensitive to the baryonic suppression and the equality
scale, see section 1 and [22].
Operationally, the slope mcan be computed by first computing the overall slope µof the ratio of
the linear power spectra and subsequently subtracting the slope of the ratio of the primordial power
spectra n. We can write
µ=∂lnPno−wiggle
lin(k/s)
Pref,no−wiggle
lin(k)
∂lnk
k=kp≡∂lnR(k, s)
∂lnk
k=kp
n=∂ln
Pprim(k/s)
Pref
prim(k)
∂lnk
k=kp
m=µ−n(3.1)
Here sis the rescaling of the sound horizon scale, s=rd/rref
d. Note that for the almost scale-invariant
power spectrum adopted in ΛCDM ( Pprim∝Askns−1), we simply have n=ns−nfid
s. While the
parameter that can be best measured in current data is µ=m+n, future surveys might be able
to distinguish between effects from mandn. Given that currently large-scale-structure analyses are
often used in combination with a prior on the slope n(containing primordial information), we focus on
the early-universe information contained in m. However, we caution that this aspect of constraining
primarily µ=m+nhas to be taken into account when interpreting the constraints on mfor a fixed
n, which are commonly shown in the literature (for example in [16, 52], see also [53]). In what follows
we refer to the ratio of the no-wiggle power spectra that appears in equation (3.1) (first line) as R.
While the computation of a slope at a point might sound like a trivial task, there are several aspects
to be considered. First, since the scale kpis chosen to lie in the linear regime (avoiding non-linear
corrections) where the baryon suppression begins, it unavoidably is impacted by the baryon acoustic
oscillations, see also figure 3. Therefore, it is important to define the slope in such a way that
is insensitive to the precise nature and phase of the BAO for it to be a robust model-independent
measure. This has been recognized already in the original ShapeFit paper [22], leading to the definition
presented in equation (3.1), where instead of a full power spectrum Plin(k) one uses a de-wiggled power
spectrum Pno−wiggle
lin(k). However, the ambiguity of the separation of the first BAO peak and the power
spectrum turnover, yielding the large variance among different de-wiggling algorithms, also affects m.
This problem motivates looking beyond the point-wise definition of a derivative and to investigate
methods that are less sensitive to small local fluctuations of the shape of the Rfunction.
It is important to note that, despite the value of mbeing sensitive to the choice of the algorithm,
the methodology adopted in the original ShapeFit implementation [15] which then is adopted for
8The transfer function in this sense is just (the square root of) the ratio of the linear power spectrum and the
primordial power spectrum.
– 18 –

applications to state of the art data [21, 50], has been shown to yield unbiased cosmological inference
at very high precision [54]. To understand why this is the case, to quantify potential biases on mfor
the next generation of surveys, and to offer a transparent way to connect the value of mto a theory
model, it is important to take a deep dive into derivatives methods and their performance.
The reader less interested in numerical implementation details of such methods may skip to sec-
tion 3.2.1.
3.2 Derivatives
While the naive gradient method that computes the numerical finite difference between subsequent
samples may intuitively appear to be the most accurate representation of the derivative, in practice
the different de-wiggling methods do not agree very well on the local derivative, yielding a potentially
large theoretical uncertainty.
In order to be robust to the differences between de-wiggling methods (see figure 14), it is possible to
define a more non-local version of the derivative. There are many such approaches, and we present
them below, showing that they result in modest differences for the extracted slope. As such, we are
trading sensitivity to the exact de-wiggling method with non-locality of the derivative computation.
This consideration can also be seen as a sort of bias-variance trade-off for the final systematic error
onm. Say that for a given way of computing the derivative, the variance is simply given by the
spread of the mvalues between different de-wiggling methods (below we use all the 13 presented
above). Then we can say that the bias instead arises from how strongly the function is approximated
in order to compute the derivative. For example, a derivative method that always assigns the slope 1
to any input will have a zero variance but a large bias. The gradient method has no bias, but a large
variance, see figure 15.
Each derivative method is based on approximating the true function by a sufficiently smooth function
in a given interval (see below for examples). The bias can be indirectly estimated from how different
the smooth function approximation is from the true function ( R) at/around the pivot point. Below
we report as bias the typical difference between Rand its smooth function approximation in a range
of scales around kp(around k∈[0.02h/Mpc,0.05h/Mpc]), but we caution that it may not be directly
interpreted as the bias for m(see section 3.4). In tests and figures we compute the ratio between the
fiducial/reference cosmology and the showcase cosmology, as listed in table 3.
Gradient This method is simply using the numerical second-order central finite difference formula
for subsequent samples to compute a numerical approximation of the derivative. For the gradient
method the functional approximation is simply the linear interpolation between two subsequent sam-
ples. Therefore, the approximation bias at any sampling location is by definition is zero, while the
variance between different de-wiggling methods remains rather large. In particular, in this case, we
findm= 0.193±0.035 (mean ±standard deviation among the 13 dewiggling methods), with a
maximal spread of ∆ m= max( m)−min(m) = 0 .13.
Spline Derivative In this method, a univariate spline (see appendix C.2) of degree 5 is fitted
to the power spectrum ratio, and its first derivative is evaluated at the pivot point. The fit can be
performed globally, across all wavenumbers k(global) or locally, only in a given krange (local). While
the local spline derivative is more sensitive to local fluctuations, the global one might under-estimate
real differences in the shape when they cannot be represented with a fifth-order polynomial.9The
global spline derivative is the one originally implemented in the context of ShapeFit for [22].
The results are displayed in figure 16 for both the local and global versions. We find for the global
spline m= 0.1293±0.0013 and a maximal deviation of ∆ m= 0.0046 and for the local spline method
9We use a smoothing s= 3 for the global version (in order to reproduce the original implementation) and s= 0.3
for the local version (decreased to give a good fit still).
– 19 –

10−310−210−1100
Wavenumber k[h/Mpc]0.50.60.70.80.91.01.1lnP(k) ratio (shifted)Approximation vs ln P(k) ratio: Gradient
No de-wiggling
Mean de-wiggling
Median de-wiggling
10%-90% quantiles of de-wiggling
No de-wiggling
Mean de-wiggling
Median de-wiggling
10%-90% quantiles of de-wiggling
10−310−210−1100
Wavenumber k[h/Mpc]−0.100−0.075−0.050−0.0250.0000.0250.0500.0750.100Error on approx. ln P(k) ratio (shifted)Approximation error (in log-space): Gradient
No de-wiggling
Mean de-wiggling
Median de-wiggling
10%-90% quantiles of de-wiggling
10−310−210−1100
Wavenumber k[h/Mpc]−0.2−0.10.00.10.20.3Derivative (gradient)Derivative of approximation: Gradient
No de-wiggling
Mean de-wiggling
Median de-wiggling
10%-90% quantiles of de-wigglingFigure 15 . Gradient method for the derivative. Top Left: Approximation of the logarithm of the ratio used
for equation (3.1) (approximation in solid, true function in dashed lines). Top right: Relative error of that
approximation. Bottom: Derivative of the functional approximation (whose value at the pivot scale kpis
taken as m).
m= 0.1856±0.028 and ∆ m= 0.11. In this case, we can see that the functional approximation
through a spline does introduce some bias, especially for the global method: the middle panels of
figure 16 show that the approximation error is around 7% for the global spline method and only
around <1% for the local one. We can immediately observe the trade off between bias and variance:
the global spline method has smaller variance but larger bias.
Linear derivative (Steps) This method simply selects two kvalues k1,k2that are not subsequent
samples and uses these to compute the slope using the usual finite difference approximation:
lnR(k2)−lnR(k1)
lnk2−lnk1(3.2)
where Ris the ratio of the (shifted) power spectra required for equation (3.1). In figure 17 we show
the corresponding results. Very similarly to the local spline method, this method trades off variance
with bias. In order to reach a 1 −2% approximation error, the steps have to be chosen relatively
closely to the pivot wavenumber (in this case ∆ ln k= lnk2−lnk1= 0.6). The resulting spread is
m= 0.172±0.015 with ∆ m= 0.061. Comparing to the local spline method, we observe about half
the variance while still keeping the bias at less than 2%. We show different step sizes in figure 20
and section 3.4 below.
– 20 –

10−310−210−1100
Wavenumber k[h/Mpc]0.50.60.70.80.91.01.1lnP(k) ratio (shifted)Approximation vs ln P(k) ratio: Global spline
No de-wiggling
Mean de-wiggling
Median de-wiggling
10%-90% quantiles of de-wiggling
No de-wiggling
Mean de-wiggling
Median de-wiggling
10%-90% quantiles of de-wiggling
10−310−210−1100
Wavenumber k[h/Mpc]0.40.50.60.70.80.91.01.11.2lnP(k) ratio (shifted)Approximation vs ln P(k) ratio: Local spline
No de-wiggling
Mean de-wiggling
Median de-wiggling
10%-90% quantiles of de-wiggling
No de-wiggling
Mean de-wiggling
Median de-wiggling
10%-90% quantiles of de-wiggling
10−310−210−1100
Wavenumber k[h/Mpc]−0.100−0.075−0.050−0.0250.0000.0250.0500.0750.100Error on approx. ln P(k) ratio (shifted)Approximation error (in log-space): Global spline
No de-wiggling
Mean de-wiggling
Median de-wiggling
10%-90% quantiles of de-wiggling
10−310−210−1100
Wavenumber k[h/Mpc]−0.100−0.075−0.050−0.0250.0000.0250.0500.0750.100Error on approx. ln P(k) ratio (shifted)Approximation error (in log-space): Local spline
No de-wiggling
Mean de-wiggling
Median de-wiggling
10%-90% quantiles of de-wiggling
10−310−210−1100
Wavenumber k[h/Mpc]−0.020.000.020.040.060.080.100.120.14Derivative (gradient)Derivative of approximation: Global spline
No de-wiggling
Mean de-wiggling
Median de-wiggling
10%-90% quantiles of de-wiggling
10−310−210−1100
Wavenumber k[h/Mpc]−0.4−0.20.00.20.4Derivative (gradient)Derivative of approximation: Local spline
No de-wiggling
Mean de-wiggling
Median de-wiggling
10%-90% quantiles of de-wigglingFigure 16 . Top: Approximation of the logarithm of the ratio used for equation (3.1) (approximation in
solid, true function in dashed lines). Middle: Relative error of that approximation. Bottom: Derivative of the
functional approximation (whose value at the pivot scale kpis taken as m). Left: Global spline. Right: Local
spline (between kmin= 0.01h/Mpc and kmax= 0.1h/Mpc).
Polynomial Derivative In this method the ratio Ris fitted over a range of scales by a polynomial of
degree d, and the derivative of this polynomial evaluated at the pivot point kpis used to determine the
slope. We show the results in figure 18 for a fit between kmin= 8·10−3/Mpc and kmax= 0.1/Mpc. The
results are m= 0.1499±0.0047 and ∆ m= 0.022 for a polynomial of degree d= 2,m= 0.1751±0.016
and ∆ m= 0.07 for a polynomial of degree d= 3 (not shown here), and m= 0.1842±0.025 and
∆m= 0.1 for a polynomial of degree d= 5. We see that higher order polynomials naturally increase
– 21 –

10−310−210−1100
Wavenumber k[h/Mpc]−0.50.00.51.01.52.0lnP(k) ratio (shifted)Approximation vs ln P(k) ratio: Steps
No de-wiggling
Mean de-wiggling
Median de-wiggling
10%-90% quantiles of de-wiggling
No de-wiggling
Mean de-wiggling
Median de-wiggling
10%-90% quantiles of de-wiggling
10−310−210−1100
Wavenumber k[h/Mpc]−0.100−0.075−0.050−0.0250.0000.0250.0500.0750.100Error on approx. ln P(k) ratio (shifted)Approximation error (in log-space): Steps
No de-wiggling
Mean de-wiggling
Median de-wiggling
10%-90% quantiles of de-wiggling
10−310−210−1100
Wavenumber k[h/Mpc]0.150.160.170.180.190.200.21Derivative (gradient)Derivative of approximation: Steps
No de-wiggling
Mean de-wiggling
Median de-wiggling
10%-90% quantiles of de-wigglingFigure 17 . Top left: Approximation of the logarithm of the ratio used for equation (3.1) (approximation in
solid, true function in dashed lines). Top right: Relative error of that approximation. Bottom: Derivative of
the functional approximation (whose value at the pivot scale kpis taken as m). Steps method with ∆ ln k= 0.6.
variance, while they also reduce bias (the bias is 3% for d= 2, 1% for d= 3 (not shown here), and
<1% for d= 5).
Hyperbolic Tangent The shape of the baryonic suppression can be approximated by a hyperbolic
tangent curve [22]. This is the reason why in the data analysis pipeline a common choice is to rescale
the power spectrum template with such a curve. This method thus stands out among the others as
one particularly close to how the data is analyzed, see section 3.4 for further discussion on this point.
One simply fits the ratio Rwith a function of the form
lnR=p0+p1·(lnk/kt) +p2/a·tanh[ a(lnk/kt)] (3.3)
Here the pi,kt, and aare the parameters of the family of curves. There are two variations of
this method. In the first method, we take ktandato be those values advocated for in [22], i.e.
kt=kp= 0.03h/Mpc and a= 0.6. In the second method we leave these two also as free parameters
of the fit, though importantly we still evaluate the derivative of equation (3.1) at the same location.
The results for both methods are shown in figure 19. Particularly interesting is that in this case even
the non-de-wiggled ratios return the same slope as all other methods. This way of computing the
derivatives is therefore robust to baryonic oscillations, but it may be prone to bias in the derivative as
the bias in Rincreases steeply away from the pivot point, see the middle panels of figure 19. We find
the low variance estimates of m= 0.1460±0.0014 and ∆ m= 0.0049 for the fixed parameters case ( a
– 22 –

10−310−210−1100
Wavenumber k[1/Mpc]0.00.51.01.52.0lnP(k) ratio (shifted)Approximation vs ln P(k) ratio: Polynomial (degree 2)
No de-wiggling
Mean de-wiggling
Median de-wiggling
10%-90% quantiles of de-wiggling
No de-wiggling
Mean de-wiggling
Median de-wiggling
10%-90% quantiles of de-wiggling
10−310−210−1100
Wavenumber k[h/Mpc]0.40.50.60.70.80.91.01.11.2lnP(k) ratio (shifted)Approximation vs ln P(k) ratio: Polynomial (degree 5)
No de-wiggling
Mean de-wiggling
Median de-wiggling
10%-90% quantiles of de-wiggling
No de-wiggling
Mean de-wiggling
Median de-wiggling
10%-90% quantiles of de-wiggling
10−310−210−1100
Wavenumber k[h/Mpc]−0.100−0.075−0.050−0.0250.0000.0250.0500.0750.100Error on approx. ln P(k) ratio (shifted)Approximation error (in log-space): Polynomial (degree 2)
No de-wiggling
Mean de-wiggling
Median de-wiggling
10%-90% quantiles of de-wiggling
10−310−210−1100
Wavenumber k[h/Mpc]−0.100−0.075−0.050−0.0250.0000.0250.0500.0750.100Error on approx. ln P(k) ratio (shifted)Approximation error (in log-space): Polynomial (degree 5)
No de-wiggling
Mean de-wiggling
Median de-wiggling
10%-90% quantiles of de-wiggling
10−310−210−1100
Wavenumber k[h/Mpc]−0.050.000.050.100.150.200.250.300.35Derivative (gradient)Derivative of approximation: Polynomial (degree 2)
No de-wiggling
Mean de-wiggling
Median de-wiggling
10%-90% quantiles of de-wiggling
10−310−210−1100
Wavenumber k[h/Mpc]−0.4−0.20.00.20.4Derivative (gradient)Derivative of approximation: Polynomial (degree 5)
No de-wiggling
Mean de-wiggling
Median de-wiggling
10%-90% quantiles of de-wigglingFigure 18 . Top: Approximation of the logarithm of the ratio used for equation (3.1) (approximation in
solid, true function in dashed lines). Middle: Relative error of that approximation. Bottom: Derivative of
the functional approximation (whose value at the pivot scale kpis taken as m). Left: Polynomial of degree 2.
Right: Polynmoial of degree 5.
andkt), and m= 0.1402±0.0026 and ∆ m= 0.0087 when these are left free.The approximation has
a roughly 5% maximum bias in either case.
– 23 –

10−310−210−1100
Wavenumber k[h/Mpc]0.50.60.70.80.91.01.1lnP(k) ratio (shifted)Approximation vs ln P(k) ratio: Tanh (ﬁxed)
No de-wiggling
Mean de-wiggling
Median de-wiggling
10%-90% quantiles of de-wiggling
No de-wiggling
Mean de-wiggling
Median de-wiggling
10%-90% quantiles of de-wiggling
10−310−210−1100
Wavenumber k[h/Mpc]0.50.60.70.80.91.01.1lnP(k) ratio (shifted)Approximation vs ln P(k) ratio: Tanh (ﬁt)
No de-wiggling
Mean de-wiggling
Median de-wiggling
10%-90% quantiles of de-wiggling
No de-wiggling
Mean de-wiggling
Median de-wiggling
10%-90% quantiles of de-wiggling
10−310−210−1100
Wavenumber k[h/Mpc]−0.100−0.075−0.050−0.0250.0000.0250.0500.0750.100Error on approx. ln P(k) ratio (shifted)Approximation error (in log-space): Tanh (ﬁxed)
No de-wiggling
Mean de-wiggling
Median de-wiggling
10%-90% quantiles of de-wiggling
10−310−210−1100
Wavenumber k[h/Mpc]−0.100−0.075−0.050−0.0250.0000.0250.0500.0750.100Error on approx. ln P(k) ratio (shifted)Approximation error (in log-space): Tanh (ﬁt)
No de-wiggling
Mean de-wiggling
Median de-wiggling
10%-90% quantiles of de-wiggling
10−310−210−1100
Wavenumber k[h/Mpc]0.000.020.040.060.080.100.120.140.16Derivative (gradient)Derivative of approximation: Tanh (ﬁxed)
No de-wiggling
Mean de-wiggling
Median de-wiggling
10%-90% quantiles of de-wiggling
10−310−210−1100
Wavenumber k[h/Mpc]0.0000.0250.0500.0750.1000.1250.150Derivative (gradient)Derivative of approximation: Tanh (ﬁt)
No de-wiggling
Mean de-wiggling
Median de-wiggling
10%-90% quantiles of de-wigglingFigure 19 . Top: Approximation of the logarithm of the ratio used for equation (3.1) (approximation in
solid, true function in dashed lines). Middle: Relative error of that approximation. Bottom: Derivative of the
functional approximation (whose value at the pivot scale kpis taken as m). Left: Tanh fitting method (fixed
aandkt). Right: Tanh fitting method with all parameters free to vary.
3.2.1 Comparing performances of different derivatives methods
A direct comparison between the different methods for calculating the derivative can be found in
figure 20. It is notable to see that many of the more unbiased derivative methods (such as e.g.,
the gradient method, the local spline, the polynomial of degree 5, the steps method with very local
support) return values of the slope around ∼0.17 with a large variance, while other methods (such
– 24 –

Gradient
Global splineLocal spline
Polynomial (degree 2) Polynomial (degree 3) Polynomial (degree 5)Steps ∆ lnk= 0.6
Steps ∆ lnk= 1.4
Steps ∆ lnk= 2.2
Tanh (ﬁxed)Tanh (ﬁt)0.120.140.160.180.200.220.240.260.28Slopem
Mean
Median
Outliers
Gradient
Global splineLocal spline
Polynomial (degree 2) Polynomial (degree 3) Polynomial (degree 5)Steps ∆ lnk= 0.6
Steps ∆ lnk= 1.4
Steps ∆ lnk= 2.2
Tanh (ﬁxed)Tanh (ﬁt)0.130.140.150.160.170.180.190.20SlopemMean
Median
OutliersFigure 20 . Comparison of the mvalues found for each different way of computing the derivative listed in
section 3. Left: using all de-wiggling methods of table 1. Right: using only the de-wiggling methods of the
‘gold sample’ of table 1. The triangles represent the median, the blue horizontal line the mean, the empty
boxes the standard deviation, the error bars the 10-90% quantiles, and the empty circles outliers beyond that.
as e.g., the global spline, the polynomial of degree 2, the steps method with large stepsizes, the tanh
methods) return smaller slopes, albeit with less variance. For reference statistical error-bars on m
from current surveys are ∼0.03−0.05, i.e., smaller than the observed variance.
It is evident that considering only the gold sample of smoothing methods strongly reduces the spread
(notice the reduced 10 −90% intervals and the smaller number of outliers). While the gold sample of
smoothing methods therefore is more homogeneous in the recovered mvalues, the remaining scatter
is still somewhat large for many of the derivative methods.
3.3 Matching the way mis extracted from the data
The requirement for obtaining unbiased estimates of the underlying cosmological parameters is not
identical to the requirement of having small mvariations across different dewiggling and derivative
implementations, as we show below.
As it is customary, algorithms and analysis methods are tested on mock data, where the true underling
cosmology is known. In this case it can be checked how biased a particular implementation Iis with
respect to a certain data analysis pipeline p. The theoretically determined slope mtheorygenerally
depends on the implementation Iand the cosmological model parameters θunder consideration.
On the other hand, the slope extracted from the mock data mdatadepends on the cosmological
parameters (and settings) used to generate the mock data θmockand the slope extraction pipeline p.
The requirement of having an un-biased recovery of the cosmological parameters then requires to
minimize the recovery bias
B(I|θmock, p) =|mtheory(θmock|I)−mdata(θmock|p)|. (3.4)
Note that by definition mtheory(θfid|I) = 0 for all implementations, and usually mdata(θfid|p) = 0, so
to find the best implementation, cosmologies different than the fiducial should be considered.
The best implementation Iwill minimize this bias for a range of cosmological parameters θmock.10
In other words, measuring the true underlying value of the parameter mdoes not matter for cosmo-
logical inference: mtheoryandmdatacould both have a large bias, but it is irrelevant as long as Bof
equation (3.4) is kept well below the statistical errors.
10Note that while the performance of any given method Imay depend on cosmology, it is customary to assume that
this dependence is weak, as long as the cosmologies to be considered are not heavily disfavored by the data.
– 25 –

For example, the data analysis for ShapeFit uses a template that rescales the linear power spectrum
via a multiplicative factor (see for example [49])
Padjusted
template(k, m, n ) =Pfid
lin(k) expm
atanh
alnk
kp
+nlnk
kp
. (3.5)
with a= 0.6 and kp= 0.03h/Mpc. The adjusted template is then rescaled with the usual BAO
parameters (e.g., α∥orα⊥) and subsequently compared to the data. Comparing equations (3.3)
and (3.5) it is evident that the method that minimizes the bias of equation (3.4) is most likely11the
tanh (fixed) derivative method, which gives closely consistent results regardless of the de-wiggling
method.
By using a derivative method that is insensitive to the dewiggling algorithm and that is tuned to the
way the data and treated, the current ShapeFit implementation has been found to be unbiased for the
cosmologies where it has been tested. Different choices that are also consistent with the data analysis
pipeline might overall have additional advantages.
3.4 What is the true value of m? Removing ambiguity to maximize consistency.
At first glance, the various ways to de-wiggle the power spectrum from section 2 and to extract the
slope mof section 3 could all appear to be a priori equally valid, as they are all just different ways
of numerically implementing equation (3.1). Yet, because of the BAO peak/power spectrum turnover
ambiguity, for a given cosmological model there seem to be not one “true” value of the slope m. While
unimportant for cosmological inference purposes, this is unsatisfactory: if we wish to use mlike other
compressed variables (e.g., α∥,α⊥etc.) it would be very useful to associate a unique value of mto
any theory model (given a fiducial cosmology).
Is it possible to find consistency among implementations of dewiggling and derivative? We offer a
possible solution next, which involves some post-processing of the ratio Rbefore taking the derivative.
We begin by noting that since for the cosmologies where it has been tested tanh (fixed) is robust to
dewiggling methods and suitable for cosmological inference, we can take this method as benchmark
and adopt as mtruethe median of tanh (fixed) method for different de-wiggling algorithms.
3.4.1 Post-Processing Filters
Given that the differences between the smoothing methods of section 2 are typically of the order of
1-5% and highly oscillatory in nature, one could imagine that a method of further smoothing the
de-wiggled power spectrum ratios might aid in getting more consistent slope values. We particularly
focus on the windowed average and the Savitzky-Golay filter [55] – The idea is that these filters should
typically mostly conserve the overall broadband shape, and simply help reduce the variance between
the different de-wiggling methods.
To define the window length of each filter (over which the smoothing is performed) we choose a given
physical size ∆ ln kand convert it into a length in terms of indices as:
Nwindow =
N×∆ lnk
lnkmax−lnkmin
(3.6)
where Ncorresponds to the number of elements in the wavenumber array and Nwindow is the number
of indices used.
We schematically show the impact of the smoothing method on the ratio between the power spectra
in figure 21. In particular, it is evident that the broadband shape is mostly preserved, while residual
11The template of equation (3.5) is typically transformed to redshift space and subjected to well known effects like
redshift space distortions, the Alcock-Paczy´ nski effect, and others. Therefore, while we expect that these additional
steps do not change which implementation is most consistent with the data analysis pipeline, we have not conclusively
proven this. We leave such a proof using the full data analysis pipeline for future work.
– 26 –

10−310−210−1100
Wavenumber k[h/Mpc]1.61.82.02.22.42.62.83.03.2P(k)/Pﬁd(k)original
mean
savitzkyFigure 21 . Impact of post-processing smoothing of the ratio Rby either a running average (mean) or
Savitzky-Golay (savitzky) filtering procedure. We show the example of the univariate spline fit to highlight
the impact of the post-processing, which is less visible for other de-wiggling methods.
Gradient
Global splineLocal spline
Polynomial (degree 2)Polynomial (degree 3)Polynomial (degree 5)Steps ∆ lnk= 0.6
Steps ∆ lnk= 1.4
Steps ∆ lnk= 2.2
Tanh (ﬁxed)Tanh (ﬁt)Gradient
Global splineLocal spline
Polynomial (degree 2)Polynomial (degree 3)Polynomial (degree 5)Steps ∆ lnk= 0.6
Steps ∆ lnk= 1.4
Steps ∆ lnk= 2.2
Tanh (ﬁxed)Tanh (ﬁt)Gradient
Global splineLocal spline
Polynomial (degree 2)Polynomial (degree 3)Polynomial (degree 5)Steps ∆ lnk= 0.6
Steps ∆ lnk= 1.4
Steps ∆ lnk= 2.2
Tanh (ﬁxed)Tanh (ﬁt)0.120.130.140.150.160.170.180.190.20SlopemNone Moving Average Savitzky-Golay
Mean
Median
Outliers
mtrue
Figure 22 . Comparison of the different derivative algorithms for different levels of post-processing. The red
line shows the mtruedefined in section 3.4. For reference, the statistical error on mfrom current surveys is
σstat
m∼0.03−0.05.
oscillations are effectively removed. In what follows, we smooth over 2.5 decades in ln kfor the
Savitzky-Golay filter and over 1.3 decades in ln kfor the mean filter. These values are optimized
by hand to reduce the impact on the extracted broadband filter while strongly suppressing residual
oscillations.
We compare different post-processing methods in figure 22 by their impact on the variance and mean
value of the extracted slopes for each different way of computing the derivative. In particular, we
observe that the variance of all derivative computations is significantly reduced. There is, however,
also a slight impact on the extracted mean value, with the moving average preferring slightly lower
mean values (by ∆ m≃0.01) than the Savitzky-Golay filter.
It is evident from figure 22 that only the Savitzky-Golay smoothing results in both (1) a broad
– 27 –

Method ∆m/m ∆m/m ∆m/m
(no post-proc.) (moving average) (Savitzky-Golay)
Gradient 0.197±0.118 −0.053±0.021 0.006±0.023
Global spline −0.118±0.010 −0.167±0.009 −0.126±0.010
Local spline 0.185±0.099 −0.054±0.021 0.006±0.023
Polynomial (degree 2) 0.011±0.032 −0.104±0.017 −0.052±0.019
Polynomial (degree 3) 0.140±0.067 −0.055±0.021 0.005±0.023
Polynomial (degree 5) 0.178±0.096 −0.055±0.021 0.006±0.023
Steps ∆ ln k= 0.6 0.132±0.061 −0.058±0.020 0.000±0.022
Steps ∆ ln k= 1.4 0.068±0.044 −0.080±0.019 −0.024±0.021
Steps ∆ ln k= 2.2 −0.028±0.027 −0.119±0.017 −0.069±0.018
Tanh (fixed) −0.004±0.011 −0.063±0.010 −0.017±0.011
Tanh (fit) −0.039±0.013 −0.120±0.010 −0.068±0.012
Table 2 . Relative mdeviation with respect to mtrue, and its scatter across the different dewiggling algorithms,
as a function of derivative method and post processing procedure. Recall that mtrueis taken to be the median
of the fixed tanh method (without postprocessing) across the different dewiggling algorithms. We mark in
bold the cases where the shift is smaller than the scatter.
consistency between most local methods for obtaining a derivative and (2) a mostly un-biased recovery
of the slope (for these particular cosmologies, see section 4 for different cosmologies).
The resulting systematic error in the context of cosmological inference will be discussed in section 4.
3.4.2 Robust mtrueestimate
Table 2 lists the relative mdeviations from the reference tanh (fixed) method. The scatter is across
the different dewiggling algorithms. The derivative/postprocessing combinations where the shift is
smaller than the scatter are marked in bold.
Not only do we see the broad consistency of various (local) derivative methods with the Savitzky-
Golay filter, we also see that they agree nicely on the expected relative variance on m, which is
σSG
m,syst≃0.023|m|. As long as the dewiggled power spectrum is computed with any of the golden
sample method and the ratio Rof equation (3.1) is smoothed using a Savitzky-Golay filter as described
in section 3.4.1, any local derivative method will yield a robust mtrueestimate with an associated
scatter (arising purely from different algorithm choices) of σSG
m,syst≃0.023|m|.
4 Systematic error budget on mfor cosmological inference from ShapeFit
We have motivated in section 3.4 a recommendation for using the value of mobtained either by (a)
using the tanh (fixed) derivative method (with no post processing) or by (b) using a Savitzky-Golay
filter of the ratio in equation (3.1) and then using any of the local derivative methods. We now assess
the performance of these two approaches in the context of cosmological parameters inference where
a wide parameter space may be explored, sampling models significantly different from ΛCDM. We
denote the two methods in the following as TANH and SG, respectively.
For the TANH case, we use no post-processing of the ratio in equation (3.1) and directly apply the fixed
tanh derivative method of section 3.2. For the SG case, we post-process the ratio in equation (3.1) with
a Savitzky-Golay filter according to section 3.4.1 and apply the steps derivative method of section 3.2
for ∆ ln k= 0.6. For both cases we report the mean and variance for the smoothing methods in the
gold sample and compare it to the systematic uncertainty of section 4.4. The considered cosmologies
are listed in table 3.
– 28 –

Cosmology Parameters (others fixed to fiducial)
Fiducial/Reference As= 2.1·10−9,ns= 0.97, Ω bh2= 0.022, Ω m= 0.31,h= 0.676
Showcase cosmology As= 4·10−9, Ωcdmh2= 0.15
Variations ns ns= [0.95,1.0]
Variations Ω k Ωk= [−0.5,−0.01,0.01,0.5]
Variations As As= 2.5·10−9
Variations w0/wa (w0, wa) =[(−1.2,0),(−0.8,0),(−1.2,0.4),(−0.8,0.4),(−1.2,−0.4),
(−0.8,−0.4),(−0.831,−0.73),(−0.64,−1.3),(−0.727,−1.05)]
Variations Ω m (Ωcdmh2,Ωbh2) = [(0 .14,0.02566) ,(0.1,0.01833) ,(0.1,0.022),(0.14,0.022)]
Variations Neff Neff= [1,2.4,3.6,5]
VariationsPmνPmν= [0.12,0.4,0.8]eV
Variations EDE (fede,Ωcdmh2,Ωbh2, ns) =[(0 .02,0.121,0.0221,0.975),(0.05,0.124,0.0222,0.98),
(0.1,0.13,0.0223,0.99),(0.15,0.137,0.0225,1.0)]
Table 3 . The different cosmologies under investigation in this paper. The showcase cosmology is used for
most plots in sections 2 and 3 (unless otherwise specified). The first set of variations is used in section 4.1,
the second set in section 4.2, and the EDE variations in section 4.3.
4.1 Null tests
We perform a small number of ‘null tests’ in order to estimate the residual numerical uncertainty in
the limit m→0. In these tests power spectrum variations are considered –resulting from changes in
cosmological parameters– for which the resulting slope mshould be identically zero. We quantify the
residual (small) deviations as a constant term in the systematic error budget, i.e., we parameterize the
systematic error budget in masσm,sys=σrel|m|+σm,0; while section 3.4 found and initial estimate
forσrelgiven by σrel≈σSG
m,sys≃0.023 for a fixed cosmology (see also below for justification that this
choice is reasonable for other cosmologies), here we quantify the constant σm,0by examining different
cosmologies. The parameter changes considered here (in As, ns, and background quantities such as
curvature and dark energy equation of state parameters) with respect to the fiducial model should
leave the shape of the power spectrum at the pivot point unaltered (hence m= 0). In figure 23 we
show that the deviations are typically extremely small with σm,0<0.001 (and thus irrelevant for
slopes |m|≳0.005 due to the term ∝ |m|), but are relevant compared to the intrinsic scatter between
different methods.
The top left panel of figure 23 shows the considered variations of the primordial power spectrum. For
changes in nsthe SG method shows some tiny bias ∼3.6×10−3·∆ns, which is not of any practical
concern given any reasonable nsvariations. The top right panel of figure 23 shows results for changes
in curvature. For |Ωk|= 0.2 the power spectrum close to the horizon shows an upturn which is
responsible for the large deviations seen. In this case the EH fitting method fails (so it is not included
for producing the high Ω kpoints in this figure). The bottom panel of figure 23 shows variations of
w0andwa. Even if the growth of structure is still scale-independent, in these models horizon-scales
effects affect the shape of the observed power spectrum at extremely large ( ≲10−3/Mpc) scales, with
minute permille-level leakage into the relevant scales for the most extreme models, see also figure 32
in appendix D. In this case too the EH fitting method fails and there is an extremely minute bias
seen for all derivative methods – which is in any case only a tiny contribution. These deviations are
smaller than the linear term of the systematic error budget as long as |m|≳0.005.
To summarise, a conservative estimate of the systematic uncertainty at m= 0 is σ0= 0.001 which
encompassess all the variations found. We note that in pure ΛCDM the differences will typically be
even smaller by another order of magnitude. We also note that the EH98 fitting method should not
be used in cases of large curvature or non-constant dark energy.
– 29 –

As·1.25ns−0.02ns+ 0.03−0.0004−0.00020.00000.00020.0004Slopem
Ωk=−0.01Ωk= 0.01Ωk=−0.2Ωk= 0.2−0.0004−0.00020.00000.00020.00040.0006Slopem
w0=−1.2
w0=−0.8
w0=−0.831, wa=−0.73
w0=−0.727, wa=−1.05
w0=−0.64, wa=−1.3
w0=−1.2, wa= 0.4
w0=−1.2, wa=−0.4
w0=−0.8, wa= 0.4
w0=−0.8, wa=−0.4−0.0010−0.00050.00000.00050.0010SlopemFigure 23 . Extracted slopes mfor the null tests either with the TANH approach (red) or the SG approach
(blue). The marker shows the mean and the line the standard deviation of the different de-wiggling methods.
The expected result in each case is a slope m≈0, since the power spectrum at the pivot point should be
unchanged by the respective parameterizations. Top left: Changes in the primordial amplitude and slope, as
well as small changes in the curvature. Top right: Large changes in the curvature – for this case we had to
omit the “EH fit” algorithms as they would need to be adapted to cases of larger curvature (the numerical
implementation currently assumes P(k)∝knsfork≪0.01/Mpc, which is not the case for large curvature).
Bottom: Different models of dark energy evolution (excluding also the “EH fit” algorithm).
4.2 Tests on Standard cosmologies
The recovered mvalues for variations with respect to the fiducial value of Ω mh2are shown in figure 24,
while figure 25 corresponds to variations of the effective number of neutrino-like species Neffand
figure 26 to variations of the total neutrino massPmν. In the figures the error bars shows the
variance across the different de-wiggling methods and the shaded region the size of the assigned
systematic uncertainty σm,sys.= 0.023|m|+ 0.001. The EH98 fitting method returns very biased
results and therefore is not considered here and will not be considered in comparisons below.
With this caveat, overall, we find a good agreement between the different methods once the systematic
uncertainty is taken into account.
– 30 –

Ωmh2−0.02 (ﬁxed Ωb/Ωcdm)
Ωmh2+ 0.02 (ﬁxed Ωb/Ωcdm)
Ωmh2−0.02 (ﬁxed Ωb)
Ωmh2+ 0.02 (ﬁxed Ωb)−0.10−0.050.000.050.10Slopem
Ωmh2−0.02 (ﬁxed Ωb/Ωcdm)
Ωmh2+ 0.02 (ﬁxed Ωb/Ωcdm)
Ωmh2−0.02 (ﬁxed Ωb)
Ωmh2+ 0.02 (ﬁxed Ωb)0.9250.9500.9751.0001.025Slopem/m tanh(ﬁxed)Figure 24 . Extracted slopes mfor the TANH (red) and SG (blue) approaches for a variety of cosmological
scenarios. The marker shows the mean, the line the uncertainty, and the shaded region the assigned systematic
uncertainty. We vary Ω mh2, leaving Ω b/Ωmconstant or Ω bconstant. Left: Absolute. Right: Relative to the
TANH method.
Neﬀ−0.6
Neﬀ+ 0.6
Neﬀ−2
Neﬀ+ 2−0.20−0.15−0.10−0.050.000.05Slopem
Neﬀ−0.6
Neﬀ+ 0.6
Neﬀ−2
Neﬀ+ 20.960.981.001.021.041.061.08Slopem/m tanh(ﬁxed)
Figure 25 . Same as figure 24 but for variations including additional ultra-light relics measured through the
effective number of neutrino-like species.
/summationtextmν= 0.12eV
/summationtextmν= 0.4eV
/summationtextmν= 0.8eV0.040.050.060.07Slopem
/summationtextmν= 0.12eV
/summationtextmν= 0.4eV
/summationtextmν= 0.8eV0.9751.0001.0251.0501.0751.100Slopem/m tanh(ﬁxed)
Figure 26 . Same as figure 24 but for variations including massive neutrinos with a total neutrino sum massPmν.
– 31 –

4.3 Early dark energy
Early dark energy (EDE) is a hypothetical form of dark energy which is relevant in the early universe
and diluting away around or shortly after the redshift of matter-radiation equality. EDE contributes
to the expansion rate of the universe, affecting the growth of density perturbations and therefore
suppressing the growth of structures. A comprehensive discussion on this subject is provided in
[35, 36, 56]. The EDE model uses an axion-like potential
V(ϕ) =m2
axf2
ax[1−cos(ϕ/fax)]nax(4.1)
with parameters fax, max, nax, which guide the redshift of the sudden dilution, the overall contribution
to the energy density, and the rapidity of the transition, respectively. The parameter maxis usually
replaced with the more phenomenological parameter fede, which is defined as the maximum ratio of
the energy density of EDE compared to the critical density, see [56] for further discussion.
Importantly, the presence of an EDE component causes an additional enhancement in the power
spectrum roughly at the same location as the baryon suppression. On large scales, the power spectrum
from the turnover to the BAO is fundamentally of a different shape, see also figure 27 and [57, Fig. 2,
Sec. 3].
This affects the different methods to measure the derivative mto different degrees. We show the
comparison of the TANH and SG methods for different values of fedein figure 28. The two meth-
ods disagree for this case beyond the designated systematic uncertainties due to the shape of the
enhancement. From figure 27 it is apparent that this is related to the form of the ratio not being well
represented by a hyperbolic tangent function around kp= 0.03h/Mpc, as the characteristic enhance-
ment/suppression is shifted with respect to the k·rdexpectation in this case (see equation (3.1)) –
using another shifting parameter other than rd(which is strongly affected by EDE) might be beneficial
here (for example 1 /keq). In figure 29 we show the result for the SG method as a horizontal dashed
red line. As in figure 20 we show the results for mfor different ways to compute the derivative for a
case with fede= 0.15. If we set aside the tanh (fixed) method, we find broad agreement between
different ways of computing the derivative albeit with more outliers. The agreement holds even for
thetanh (fit) method where the ktof the method is allowed to be adjusted during the fit (differing
from kp). We conclude that the currently used method of obtaining a slope with fixed kpmight not be
suitable for early dark energy cases, but adjusting kpin such cases shows promise. We leave further
exploration of ShapeFit in the EDE model for future work.
4.4 Systematic error budget on m: a recipe
We propose the following approach to derive mconsistently given a theory power spectrum or within
a theoretical modeling pipeline:
1. Compute the de-wiggled power spectrum with the preferred method (see section 2 for possibil-
ities, we recommend methods in the gold sample)
2. Compute the ratio required for equation (3.1).
3. Smooth the ratio using a Savitzky-Golay filter as described in section 3.4.1.
4. Compute the derivative using any local derivative method (for example using steps with ∆ ln k=
0.6, or the simple gradient).
5. Associate a systematic uncertainty of σSG
m,systto the obtained result, quantifying possible differ-
ences in de-wiggling and derivative computation:
σSG
m,syst= 0.023|m|+ 0.001 (4.2)
It is also possible to use the tanh (fixed) method (see section 3.2) as well, adopting a slightly
smaller systematic uncertainty of σtanh (fixed)
m,syst = 0.011|m|+0.001. By definition this smaller systematic
– 32 –

104
103
102
101
100101
Wavenumber k[h/Mpc]1.041.061.081.101.121.141.161.18P(k) ratio (shifted)No de-wiggling
Mean de-wiggling
Median de-wiggling
10\%-90\% quantiles of de-wigglingFigure 27 . Ratio of the (shifted) power spectra (the Rof equation (3.1)) for an EDE cosmology with
fede= 0.15.
fede= 0.02
fede= 0.05
fede= 0.1
fede= 0.150.010.020.030.040.05Slopem
fede= 0.02
fede= 0.05
fede= 0.1
fede= 0.150.60.70.80.91.01.1Slopem/m tanh(ﬁxed)
Figure 28 . Same as figure 24, but for EDE with varying fede. We also vary As,ns, Ωm,h, Ωbh2according to
the contours, as shown in table 3. Not changing the other cosmological parameters results in roughly similar
(even slightly more discrepant) results.
uncertainty covers only differences between de-wiggling methods and not different derivative methods.
Furthermore, this uncertainty might be under-estimating the systematic biases for cosmologies whose
functional shape is not close to a hyperbolic tangent curve, as we discussed in section 4.3.
As illustrated in section 3.3 the systematic error in the value of mof equation (4.2) may be somewhat
conservative for cosmological inference, but this is not a concern for current surveys. In fact, note
that the systematic error of equation (4.2) is about an order of magnitude smaller than the reported
statistical uncertainties of σm,stat ≃0.03−0.05 (see [16, 52]) for |m|≲0.1 as it is currently mea-
sured. We show a comparison between measured statistical uncertainties and the derived systematic
uncertainty in figure 30.
For future applications it is important to keep in mind that in any cosmological data analysis it is
common practice to start including systematic contributions to the systematic error budget only if
– 33 –

Gradient
Global splineLocal spline
Polynomial (degree 2)Polynomial (degree 3)Polynomial (degree 5)Steps ∆ lnk= 0.6
Steps ∆ lnk= 1.4
Steps ∆ lnk= 2.2
Tanh (ﬁxed)Tanh (ﬁt)Gradient
Global splineLocal spline
Polynomial (degree 2)Polynomial (degree 3)Polynomial (degree 5)Steps ∆ lnk= 0.6
Steps ∆ lnk= 1.4
Steps ∆ lnk= 2.2
Tanh (ﬁxed)Tanh (ﬁt)Gradient
Global splineLocal spline
Polynomial (degree 2)Polynomial (degree 3)Polynomial (degree 5)Steps ∆ lnk= 0.6
Steps ∆ lnk= 1.4
Steps ∆ lnk= 2.2
Tanh (ﬁxed)Tanh (ﬁt)0.020.040.060.08Slopem
None Moving Average Savitzky-Golay
Mean
Median
Outliers
mtrueFigure 29 . Same as figure 20 but for a model with fede= 0.15.
0.00 0.02 0.04 0.06 0.08 0.10 0.12 0.14
|m|0.000.020.040.060.080.10UncertaintySystematic
Statistical (eBOSS)
Statistical (DESI, symmetrized)
Figure 30 . Comparison of statistical and systematic uncertainties for eBOSS (blue, [16]) and DESI (green,
[52]). The systematic uncertainty is that of equation (4.2).
they cross a threshold, usually defined as a fraction ϵof the statistical error. For example DESI takes
ϵto be 0.25 [58]. The (conservative) systematic error on mof equation (4.2) can then be used to
evaluate quickly whether it is one component of the systematic uncertainty to be included in the final
error budget (and therefore better quantified specifically for the adopted pipeline) or it is something
that can be safely ignored.
5 Conclusion
Separating the oscillations and the broadband shape of the power spectrum is a very important
task both for traditional analysis pipelines (full-modeling) and novel parameter compression schemes
(ShapeFit). We have demonstrated that there is a roughly 1-2% level difference between different
proposed methods to de-wiggle the power spectrum, see also figure 14. Since no method stands out
– 34 –

as particularly more well-motivated or well-defined than the others, we argue that this percent-level
difference should be seen as an inherent systematic uncertainty to the de-wiggling procedure.
Importantly, the differences between the methods are strongly enhanced when taking the derivative
of the power spectra, such as required for the mparameter of ShapeFit (see equation (3.1)), which is
used to compress the broadband information of the power spectrum. Overall, this leads to dramatic
differences for the value of the mparameter up to 50%. These large differences are not a major
concern for cosmological inference for current surveys: as long as the theory pipeline is consistent
with the data analysis pipeline, there is virtually no bias on the cosmological parameters. However,
this result still motivates ways of taking a derivative that is more robust to the precise de-wiggling
method employed. In this work we have investigated a number of such non-local derivative algorithms
and compare the results. We find that there are still large discrepancies between the computed values
ofm, see figure 20.
However, there are two important approaches in which the systematic uncertainty of the computed
slopes mis greatly reduced. First, when using a method of computing the derivative that is close to
howmwould be obtained from the data, the values of mare more consistent, with a relative spread
of only around σm/m≃1.1%. However, it should be noted that in the currently widespread imple-
mentation of the method this relies on an approximation of the functional shape of the suppression
that might become inaccurate when exploring some cosmologies beyond ΛCDM that alter early-time
physics especially before or at matter-radiation equality, see below. Second, one can post-process the
power spectrum ratio of which the derivative is taken by a smoothing procedure, such as for example
the Savitzky-Golay filter. In this case we also find a low spread in the values of σm/m≃2.3%, and
also an agreement in the reported mean value with the previous method.
We investigate both approaches in a range of cosmologies and assign a total systematic uncertainty
of
σm,syst= 0.023|m|+ 0.001, (5.1)
with some small constant term derived from a number of null-tests where m≈0 is theoretically
expected but not necessarily recovered, see section 4.1. We note that the quoted number is the
conservative result and a reduced uncertainty of σm,syst= 0.011m+ 0.001 can be obtained with the
approach outlined in section 3.4. In general the assigned systematic uncertainty σm,syst, is much
smaller than current statistical uncertainties, see for example figure 30.
We find that for most simple one- and two-parameter extensions of the ΛCDM model (including
curvature, additional relativistic degrees of freedom, dark energy variations, models with massive
neutrinos), the assigned systematic uncertainty very well captures deviations between the two ap-
proaches, see section 4. We therefore conclude that present analyses are unaffected. However, there
are a number of important caveats:
•In cosmologies beyond ΛCDM for which the baryonic suppression is modified in a non-trivial
way, such as for early dark energy cosmologies investigated in section 4.3, further care is needed.
We argue that it might be necessary to extract additional information beyond the slope at the
pre-defined fixed pivot (such as checking the consistency with a varied pivot analysis). We leave
a more detailed investigation into this case for future work.
•While the systematic error estimate for the mvalue may be slightly conservative for cosmological
inference, this estimate offers guidance as of when the systematic error in mis negligible and
can be ignored, when it is sub dominant so it can simply be propagated in the error budget, or
when it can become of concern.
•Future analyses with further reductions in the statistical uncertainties might need more careful
extraction of mwith the objective of further reducing the systematic error and a) use an un-
ambiguous definition of the slope mthat can be applied in the same way to the data and to
the theoretical modeling of the power spectrum and b) that generalizes well to models with an
atypical shape of the baryonic suppression region of the power spectrum.
– 35 –

We conclude that for current analyses the differences in the de-wiggling methods are not critical and
a sub dominant systematic uncertainty on the slope of the power spectrum mis sufficient for most
simple extensions of the ΛCDM model. Going forward, and depending on the specific application and
the model to be constrained, the recipe inevitably will need to be re-fined (both in the extraction
from the data and in the theoretical analysis pipeline).
Acknowledgements
We thank Hector Gil Marin, Sabino Matarrese, Sergi Novell and Alice Pisani. Funding for this work
was partially provided by the Spanish MINECO under project PID2022-141125NB-I00 MCIN/AEI,
and the “Center of Excellence Maria de Maeztu 2020-2023” award to the ICCUB (CEX2019-000918-
M funded by MCIN/AEI/10.13039/501100011033). This work was supported by the Erasmus+ Pro-
gramme of the European Union under 2023-1-IT02-KA131-HED-000127536. The content of this
publication does not necessarily reflect the official opinion of the European Union. Responsibility for
the information and views expressed lies entirely with the authors. Katayoon Ghaemi acknowledges
support from the French government under the France 2030 investment plan, as part of the Initiative
d’Excellence d’Aix- Marseille Universit´ e - A*MIDEX AMX-22-CEI-03.
References
[1] C.J. Miller, R.C. Nichol and D.J. Batuski, Possible detection of baryonic fluctuations in the large scale
structure power spectrum ,Astrophys. J. 555(2001) 68 [ astro-ph/0103018 ].
[2]2dFGRS collaboration, The 2dF Galaxy Redshift Survey: Power-spectrum analysis of the final dataset
and cosmological implications ,Mon. Not. Roy. Astron. Soc. 362(2005) 505 [ astro-ph/0501174 ].
[3]SDSS collaboration, Detection of the Baryon Acoustic Peak in the Large-Scale Correlation Function of
SDSS Luminous Red Galaxies ,Astrophys. J. 633(2005) 560 [ astro-ph/0501171 ].
[4]eBOSS collaboration, Completed SDSS-IV extended Baryon Oscillation Spectroscopic Survey:
Cosmological implications from two decades of spectroscopic surveys at the Apache Point Observatory ,
Phys. Rev. D 103(2021) 083533 [ 2007.08991 ].
[5]DESI collaboration, DESI 2024 VI: Cosmological Constraints from the Measurements of Baryon
Acoustic Oscillations ,2404.03002 .
[6] B. Bahr-Kalus, D. Parkinson and E.-M. Mueller, Measurement of the matter-radiation equality scale
using the extended baryon oscillation spectroscopic survey quasar sample ,Mon. Not. Roy. Astron. Soc.
524(2023) 2463 [ 2302.07484 ].
[7] M. Crocce and R. Scoccimarro, Nonlinear Evolution of Baryon Acoustic Oscillations ,Phys. Rev. D 77
(2008) 023533 [ 0704.2783 ].
[8] A.G. Sanchez, C.M. Baugh and R. Angulo, What is the best way to measure baryonic acoustic
oscillations? ,Mon. Not. Roy. Astron. Soc. 390(2008) 1470 [ 0804.0233 ].
[9] N. Padmanabhan and M. White, Calibrating the Baryon Oscillation Ruler for Matter and Halos ,Phys.
Rev. D 80(2009) 063508 [ 0906.1198 ].
[10] B.D. Sherwin and M. Zaldarriaga, The Shift of the Baryon Acoustic Oscillation Scale: A Simple
Physical Picture ,Phys. Rev. D 85(2012) 103523 [ 1202.3998 ].
[11] F. Prada, C.G. Sc´ occola, C.-H. Chuang, G. Yepes, A.A. Klypin, F.-S. Kitaura et al., Hunting down
systematics in baryon acoustic oscillations after cosmic high noon ,Mon. Not. Roy. Astron. Soc. 458
(2016) 613 [ 1410.4684 ].
[12] R.A. Porto, The effective field theorist’s approach to gravitational dynamics ,Phys. Rept. 633(2016) 1
[1601.04914 ].
[13] M.M. Ivanov, Effective field theory for large-scale structure , inHandbook of Quantum Gravity ,
C. Bambi, L. Modesto and I. Shapiro, eds., (Singapore), pp. 1–48, Springer Nature Singapore (2023),
DOI.
– 36 –

[14] O.H.E. Philcox, B.D. Sherwin, G.S. Farren and E.J. Baxter, Determining the Hubble Constant without
the Sound Horizon: Measurements from Galaxy Surveys ,Phys. Rev. D 103(2021) 023538 [ 2008.08084 ].
[15] S. Brieden, H. Gil-Mar´ ın and L. Verde, Model-independent versus model-dependent interpretation of the
sdss-iii boss power spectrum: Bridging the divide ,Physical Review D 104(2021) L121301.
[16] S. Brieden, H. Gil-Mar´ ın and L. Verde, Model-agnostic interpretation of 10 billion years of cosmic
evolution traced by BOSS and eBOSS data ,JCAP 08(2022) 024 [ 2204.11868 ].
[17] T. Simon, P. Zhang and V. Poulin, Cosmological inference from the EFTofLSS: the eBOSS QSO
full-shape analysis ,JCAP 07(2023) 041 [ 2210.14931 ].
[18] G. D’Amico, J. Gleyzes, N. Kokron, K. Markovic, L. Senatore, P. Zhang et al., The Cosmological
Analysis of the SDSS/BOSS data from the Effective Field Theory of Large-Scale Structure ,JCAP 05
(2020) 005 [ 1909.05271 ].
[19] T. Tr¨ oster et al., Cosmology from large-scale structure: Constraining ΛCDM with BOSS ,Astron.
Astrophys. 633(2020) L10 [ 1909.11006 ].
[20] M.M. Ivanov, M. Simonovi´ c and M. Zaldarriaga, Cosmological Parameters and Neutrino Masses from
the Final Planck and Full-Shape BOSS Data ,Phys. Rev. D 101(2020) 083504 [ 1912.08208 ].
[21]DESI collaboration, DESI 2024 VII: Cosmological Constraints from the Full-Shape Modeling of
Clustering Measurements ,2411.12022 .
[22] S. Brieden, H. Gil-Mar´ ın and L. Verde, Shapefit: extracting the power spectrum shape information in
galaxy surveys beyond bao and rsd ,Journal of Cosmology and Astroparticle Physics 2021 (2021) 054.
[23] S. Brieden, H. Gil-Mar´ ın and L. Verde, Model-agnostic interpretation of 10 billion years of cosmic
evolution traced by boss and eboss data ,Journal of Cosmology and Astroparticle Physics 2022 (2022)
024.
[24] H.-J. Seo and D.J. Eisenstein, Baryonic acoustic oscillations in simulated galaxy redshift surveys ,
Astrophys. J. 633(2005) 575 [ astro-ph/0507338 ].
[25] D.J. Eisenstein, H.-j. Seo and M.J. White, On the Robustness of the Acoustic Scale in the Low-Redshift
Clustering of Matter ,Astrophys. J. 664(2007) 660 [ astro-ph/0604361 ].
[26] D.J. Eisenstein and W. Hu, Power spectra for cold dark matter and its variants ,Astrophys. J. 511
(1997) 5 [ astro-ph/9710252 ].
[27] D.J. Eisenstein and W. Hu, Baryonic features in the matter transfer function ,The Astrophysical
Journal 496(1998) 605.
[28] B.A. Reid, W.J. Percival, D.J. Eisenstein, L. Verde, D.N. Spergel, R.A. Skibba et al., Cosmological
constraints from the clustering of the Sloan Digital Sky Survey DR7 luminous red galaxies ,Monthly
Notices of the Royal Astronomical Society 404(2010) 60 [ 0907.1659 ].
[29]VIRGO Consortium collaboration, Stable clustering, the halo model and nonlinear cosmological power
spectra ,Mon. Not. Roy. Astron. Soc. 341(2003) 1311 [ astro-ph/0207664 ].
[30] R. Takahashi, M. Sato, T. Nishimichi, A. Taruya and M. Oguri, Revising the Halofit Model for the
Nonlinear Matter Power Spectrum ,Astrophys. J. 761(2012) 152 [ 1208.2701 ].
[31] S. Bird, M. Viel and M.G. Haehnelt, Massive neutrinos and the non-linear matter power spectrum ,
Monthly Notices of the Royal Astronomical Society 420(2012) 2551.
[32] A. Mead, S. Brieden, T. Tr¨ oster and C. Heymans, hmcode-2020: improved modelling of non-linear
cosmological power spectra with baryonic feedback ,Mon. Not. Roy. Astron. Soc. 502(2021) 1401
[2009.01858 ].
[33] A. Mead, J. Peacock, C. Heymans, S. Joudaki and A. Heavens, An accurate halo model for fitting
non-linear cosmological power spectra and baryonic feedback models ,Mon. Not. Roy. Astron. Soc. 454
(2015) 1958 [ 1505.07833 ].
[34] D. Blas, J. Lesgourgues and T. Tram, The Cosmic Linear Anisotropy Solving System (CLASS). Part
II: Approximation schemes ,J. Cosmology Astropart. Phys. 2011 (2011) 034 [ 1104.2933 ].
[35] T.L. Smith, V. Poulin and M.A. Amin, Oscillating scalar fields and the Hubble tension: a resolution
with novel signatures ,Phys. Rev. D 101(2020) 063523 [ 1908.06995 ].
– 37 –

[36] V. Poulin, T.L. Smith, D. Grin, T. Karwal and M. Kamionkowski, Cosmological implications of
ultralight axionlike fields ,Phys. Rev. D 98(2018) 083525 [ 1806.10608 ].
[37] Z. Vlah, U. Seljak, M.Y. Chu and Y. Feng, Perturbation theory, effective field theory, and oscillations
in the power spectrum ,Journal of Cosmology and Astroparticle Physics 2016 (2016) 057.
[38] S. Hinton, Extraction of cosmological information from wigglez ,arXiv preprint arXiv:1604.01830 (2016)
.
[39] J.F. Epperson, On the runge example ,The American Mathematical Monthly 94(1987) 329.
[40] D. Blas, M. Garny, M.M. Ivanov and S. Sibiryakov, Time-sliced perturbation theory ii: baryon acoustic
oscillations and infrared resummation ,Journal of Cosmology and Astroparticle Physics 2016 (2016)
028.
[41] B. Audren, J. Lesgourgues, K. Benabed and S. Prunet, Conservative constraints on early cosmology
with monte python ,Journal of Cosmology and Astroparticle Physics 2013 (2013) 001.
[42] T. Brinckmann and J. Lesgourgues, Montepython 3: boosted mcmc sampler and other features ,Physics
of the Dark Universe 24(2019) 100260.
[43] S. Dodelson and F. Schmidt, Modern cosmology , Academic press (2020).
[44] J.D. Secada, Numerical evaluation of the hankel transform ,Computer Physics Communications 116
(1999) 278.
[45]BOSS collaboration, Fitting Methods for Baryon Acoustic Oscillations in the Lyman- \alpha Forest
Fluctuations in BOSS Data Release 9 ,JCAP 03(2013) 024 [ 1301.3456 ].
[46] J. Hamann, S. Hannestad, J. Lesgourgues, C. Rampf and Y.Y.Y. Wong, Cosmological parameters from
large scale structure - geometric versus shape information ,J. Cosmology Astropart. Phys. 2010 (2010)
022 [ 1003.3999 ].
[47] B. Wallisch, Cosmological Probes of Light Relics , Ph.D. thesis, Cambridge U., 2018. 1810.02800 .
10.17863/CAM.30368.
[48] A. Chudaykin, M.M. Ivanov, O.H. Philcox and M. Simonovi´ c, Nonlinear perturbation theory extension
of the boltzmann code class ,Physical Review D 102(2020) 063533.
[49] M. Maus et al., An analysis of parameter compression and full-modeling techniques with Velocileptors
for DESI 2024 and beyond ,2404.07312 .
[50] Y. Lai et al., A comparison between Shapefit compression and Full-Modelling method with PyBird for
DESI 2024 and beyond ,2404.07283 .
[51] N. Sch¨ oneberg, L. Verde, H. Gil-Mar´ ın and S. Brieden, Bao+ bbn revisited—growing the hubble tension
with a 0.7 km/s/mpc constraint ,Journal of Cosmology and Astroparticle Physics 2022 (2022) 039.
[52]DESI collaboration, DESI 2024 V: Full-Shape Galaxy Clustering from Galaxies and Quasars ,
2411.12021 .
[53] J.-Q. Jiang and Y.-S. Piao, Can the sound horizon-free measurement of H0constrain early new
physics? ,2501.16883 .
[54] S. Brieden, H. Gil-Mar´ ın and L. Verde, PT challenge: validation of ShapeFit on large-volume,
high-resolution mocks ,J. Cosmology Astropart. Phys. 2022 (2022) 005 [ 2201.08400 ].
[55] I. Esteban, M.C. Gonz´ alez-Garc´ ıa, A. Hernandez-Cabezudo, M. Maltoni and T. Schwetz, Global
analysis of three-flavour neutrino oscillations: synergies and tensions in the determination of θ23,δcp,
and the mass ordering ,Journal of High Energy Physics 2019 (2019) 1.
[56] V. Poulin, T.L. Smith and T. Karwal, The ups and downs of early dark energy solutions to the hubble
tension: a review of models, hints and constraints circa 2023 ,Physics of the Dark Universe (2023)
101348.
[57] A. Klypin, V. Poulin, F. Prada, J. Primack, M. Kamionkowski, V. Avila-Reese et al., Clustering and
Halo Abundances in Early Dark Energy Cosmological Models ,Mon. Not. Roy. Astron. Soc. 504(2021)
769 [ 2006.14910 ].
– 38 –

[58] DESI Collaboration, A.G. Adame, J. Aguilar, S. Ahlen, S. Alam, D.M. Alexander et al., DESI 2024 V:
Full-Shape Galaxy Clustering from Galaxies and Quasars ,arXiv e-prints (2024) arXiv:2411.12021
[2411.12021 ].
A Smooth replacement
Given that not all algorithms find a power spectrum in the entire range of wavenumbers, sometimes
the de-wiggled power spectrum is substituted by the original power spectrum outside of a pre-defined
range. It is simply assumed that the original power spectrum outside of that range will show suffi-
ciently small wiggles that these can be effectively ignored.
If one would just replace the power spectrum inside the given range, one would naturally find dis-
continuities at the edge of the replacement from slight mismatches in the power spectrum amplitude
between the true wiggly power spectrum and the de-wiggled power spectrum. In order to minimize
such discontinuities, a smooth replacement is performed instead, where the output smoothly interpo-
lates between the true and the de-wiggled power spectrum.
The formula used for such cases is simply
lnP(k) = t(k)·lnPoriginal(k) + [1−t(k)]·lnPde−wiggled(k) (A.1)
In order to ensure a smooth transition, one simply has to choose a smooth transition function t(k).
We work in logarithmic space since in this space the power spectrum differences do typically not vary
over orders of magnitude. A given smooth transition as a function of ln kcan be written as
t±(k|k0,∆) =1
2±1
2tanh([ln k−lnk0]/∆), (A.2)
where ln k0and ∆ are adjustable parameters. The former is the center point of the transition and
the latter is the width of the transition. This transition t±(k) would smoothly interpolate from one
power spectrum to the other (favoring either the original or the de-wiggled power spectrum at low k
depending on the sign). However, we want to use the original power spectrum in two regimes: below
and above the interval where the de-wiggling algorithm works reliably. In this case we simply use the
combination
t(k) =t−(k|kmin,∆) + t+(k|kmax,∆) (A.3)
with kmin,kmax, and ∆ being chosen for each usage case to optimize the consistency between the
methods. We call the parameter ∆ the “replacement width” for convenience.
B Plateau Peak/Wiggle decomposition
Roughly speaking, the BAO are expected to follow a form like sin( krd)/(krd) [26] which asymptotes
towards unity at k→0. Whether this first plateau should be considered as oscillation or part
of the broadband then makes a crucial difference in whether such a peak around k≃π/(2rd)≃
0.01/Mpc is apparent in the wiggle to no-wiggle ratio, and more indirectly how large the peak at
k≃3π/(2rd)≃0.03/Mpc should be. Both of these effects also have a crucial impact on the slope at
kp≃π/rd≃0.03h/Mpc≃0.02/Mpc. Overall, the oscillation-broadband decomposition is not well
defined in this range between k∈[π/(2rd),3π/(2rd)], therefore owing to the large differences between
the different methods in this range, see also figure 14.
We can show this effect with a very simple toy example: Consider a simplified BAO of the form
f(x) = sin( πx)/(πx), see also figure 31. It is clear that for x≫1 the broadband should be close
to zero and for x≪1 the broadband should be unity. However, the precise transition between
– 39 –

10−210−1100101
Normalized wavenumber x=krd/π−2.0−1.5−1.0−0.50.00.51.01.52.0Normalized amplitudeBAO
Broadband 1
Broadband 2
10−210−1100101−2.0−1.5−1.0−0.50.00.51.01.52.0
BAO - (Broadband 1)
BAO - (Broadband 2)Figure 31 . Simplified BAO model to show the ambiguity in the removal of the first peak/plateau. The
function Broadband 1 is 1 −tanh( x) and Broadband 2 is1
2−1
2tanh[3( x−0.6)]. We also mark x= 1/2 and
x= 3/2 by dotted vertical grey lines.
these two regimes is not well defined. Either the broadband simply tracks the function until the
first oscillation begins (essentially until the first zero-crossing), or the broadband already begins to
smoothly interpolate at an earlier point. We show two such functions in figure 31. The resulting
difference (as a stand-in for the wiggle/no-wiggle decomposition) shows either a clear first peak at
around x≈1/2 and x≈3/2 or no peak at x≈1/2 and a reduced peak at x≈3/2. We also observe
that the slope at x≈1 is strongly impacted as a consequence. Translating this back to k=πx/r d,
this gives the differences mentioned above.
Since neither assignment of a broadband can be considered more physical than another, this kind of
ambiguity in the first two peaks and the slope at x≈1 ork≈π/rdis unavoidable.
C Splines
A spline is a simple numerical approach to interpolating data. The idea is that between any two data
points ( xi,yi) and ( xi+1, yi+1) the interpolation uses a smooth function fi(x). What differentiates
this approach from linear interpolation (for example) is that at each of the data positions (nodes xi)
the function has to suffice conditions. The most obvious would for example be passing through the
data points (and thus ensuring continuity), which requires that fi(xi) =yiandfi(xi+1) =yi+1.
Given a sufficiently flexible interpolating function fi(x) more advanced conditions can be imposed,
most commonly continuous differentiability, which requires that f′
i(xi+1) =f′
i+1(xi+1). We discuss
specific approaches and implementations below.
C.1 Cubic Spline
A cubic spline is simply a spline with a cubic function, i.e. a function of the form
fi(x) =ai+bix+cix2+dix3(C.1)
Obviously there are four degrees of freedom for such a function. Two are required to fix fi(xi) =yiand
fi(xi+1) =yi+1. One more is used to require differentiability f′
i(xi+1) =f′
i+1(xi+1). In this case one
further degree of freedom is open and is used to ensure that the second derivatives are also continuous
f′′
i(xi+1) =f′′
i+1(xi+1). In total, there are 4 N−2 conditions (with Nbeing the number of data points).
This leaves 2 final boundary conditions, which are typically requiring the derivatives at the endpoints
to be those linearly estimated f′
1(x1) = (y2−y1)/(x2−x1) and f′
N−1(xN) = (yN−yN−1)/(xN−xN−1).
In general such splines are very flexible without strong over-fitting or Runge-phenomena.
– 40 –

C.2 Univariate spline
The idea of a univariate spline is that instead of fixing the interpolating intervals to be the data
positions ( xi,xi+1), one instead introduces artificial points (so-called knots tj) which lie somewhere
between the data points. Depending on the approach there can be more or fewer knots than data
points (but typically fewer). In most use cases, the final function f(x) that is generated from joining
eachfj(x) defined between tjandtj+1to cover the full interval x1toxNhas to obey certain conditions.
First, the fj(x) are ensured to be continuous and as differentiable as their polynomial degree (which,
unlike for Cubic splines, can be something other than 3). Then, the number of knots is increased
(starting from 2) until a certain condition is met. In our case this condition is always one of fitting
the data points reasonably well, with
X
(yi−f(xi))2<s (C.2)
For some smoothing strength s. Note that since the knots don’t coincide with the data points xi
and since the spline is of possibly higher order, there is often no analytical solution and the knot
number and the individual polynomial coefficients of the fj(x) are found numerically. The choice of
a smoothing strength is an important consideration, and we often optimized the values quoted here
visually and ‘by hand’.
The univariate splines are often represented in the B-spline basis, which is an alternative way of
representing arbitrary splines. In the text we call a B-spline simply that for which s= 0 (note that
this still does not imply that tj=xi).
D Power spectrum in w0-wacosmologies
We show in figure 32 the impact of the CLP w0-waparameterized dark energy on the power spectrum,
compared to the fiducial model. We observe in the right panel that there is a tiny leakage of power
spectrum difference even to scales around 0 .03h/Mpc at the level of <0.1% which causes the minute
differences in mobserved in figure 23.
– 41 –

10−410−310−210−1100
k[1/Mpc]−0.20.00.2∆P(k)/Pﬁd(k) w0=−1.2
w0=−0.8
w0=−0.831, wa=−0.73
w0=−0.727, wa=−1.05
w0=−0.64, wa=−1.3
w0=−1.2, wa= 0.4
w0=−1.2, wa=−0.4
w0=−0.8, wa= 0.4
w0=−0.8, wa=−0.4
10−410−310−210−1100
k[1/Mpc]−0.0010−0.00050.00000.00050.0010∆P(k)/Pﬁd(k) w0=−1.2
w0=−0.8
w0=−0.831, wa=−0.73
w0=−0.727, wa=−1.05
w0=−0.64, wa=−1.3
w0=−1.2, wa= 0.4
w0=−1.2, wa=−0.4
w0=−0.8, wa= 0.4
w0=−0.8, wa=−0.4Figure 32 . Relative difference in the power spectrum for a few selected w0-wacosmologies of figure 23. Top:
Relative difference. Bottom: Relative difference, but re-normalized to coincident amplitude at smallest scales
and zoomed into the ±0.1% range.
– 42 –