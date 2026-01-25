# RAG: A Random-Forest-Based Generative Design Framework for Uncertainty-Aware Design of Metamaterials with Complex Functional Response Requirements

**Authors**: Bolin Chen, Dex Doksoo Lee, Wei "Wayne'' Chen, Wei Chen

**Published**: 2026-01-19 17:06:12

**PDF URL**: [https://arxiv.org/pdf/2601.13233v1](https://arxiv.org/pdf/2601.13233v1)

## Abstract
Metamaterials design for advanced functionality often entails the inverse design on nonlinear and condition-dependent responses (e.g., stress-strain relation and dispersion relation), which are described by continuous functions. Most existing design methods focus on vector-valued responses (e.g., Young's modulus and bandgap width), while the inverse design of functional responses remains challenging due to their high-dimensionality, the complexity of accommodating design requirements in inverse-design frameworks, and non-existence or non-uniqueness of feasible solutions. Although generative design approaches have shown promise, they are often data-hungry, handle design requirements heuristically, and may generate infeasible designs without uncertainty quantification. To address these challenges, we introduce a RAndom-forest-based Generative approach (RAG). By leveraging the small-data compatibility of random forests, RAG enables data-efficient predictions of high-dimensional functional responses. During the inverse design, the framework estimates the likelihood through the ensemble which quantifies the trustworthiness of generated designs while reflecting the relative difficulty across different requirements. The one-to-many mapping is addressed through single-shot design generation by sampling from the conditional likelihood. We demonstrate RAG on: 1) acoustic metamaterials with prescribed partial passbands/stopbands, and 2) mechanical metamaterials with targeted snap-through responses, using 500 and 1057 samples, respectively. Its data-efficiency is benchmarked against neural networks on a public mechanical metamaterial dataset with nonlinear stress-strain relations. Our framework provides a lightweight, trustworthy pathway to inverse design involving functional responses, expensive simulations, and complex design requirements, beyond metamaterials.

## Full Text


<!-- PDF content starts -->

RAG: A RANDOM-FOREST-BASEDGENERATIVEDESIGN
FRAMEWORK FORUNCERTAINTY-AWAREDESIGN OF
METAMATERIALS WITHCOMPLEXFUNCTIONALRESPONSE
REQUIREMENTS
Bolin Chen1, Dex Doksoo Lee1, Wei “Wayne” Chen2, and Wei Chen1*
1Department of Mechanical Engineering, Northwestern University, Evanston, IL 60208
2J. Mike Walker ’66 Department of Mechanical Engineering, Texas A&M University, College Station, TX 77843
*Corresponding author: weichen@northwestern.edu
ABSTRACT
Metamaterials design for advanced functionality often entails the inverse design on nonlinear and
condition-dependent responses (e.g., stress-strain relation and dispersion relation), which are de-
scribed by continuous functions. Most existing design methods focus on vector-valued responses
(e.g., Young’s modulus and bandgap width), while the inverse design of functional responses remains
challenging due to their high-dimensionality, the complexity of accommodating design requirements
in inverse-design frameworks, and non-existence or non-uniqueness of feasible solutions. Although
generative design approaches have shown promise, they are often data-hungry, handle design re-
quirements heuristically, and may generate infeasible designs without uncertainty quantification. To
address these challenges, we introduce a RAndom-forest-based Generative approach (RAG). By
leveraging the small-data compatibility of random forests and reformulating the forward mapping in
a discretization-invariant way, RAG enables data-efficient predictions of high-dimensional functional
responses. During the inverse design, the framework estimates the likelihood of solutions conditioned
on the design requirement that can be flexibly specified. The likelihood estimated through the
ensemble quantifies the trustworthiness of generated designs while reflecting the relative difficulty
across different requirements. The one-to-many mapping is addressed through single-shot design
generation by sampling from the conditional likelihood. We demonstrate RAG on: 1) acoustic
metamaterials with prescribed partial passbands/stopbands, and 2) mechanical metamaterials with
targeted snap-through responses, using 500 and 1057 samples, respectively. Its data-efficiency is
benchmarked against neural networks on a public mechanical metamaterial dataset with nonlinear
stress-strain relations. Our framework provides a lightweight, trustworthy pathway to inverse design
involving functional responses, expensive simulations, and complex design requirements, beyond
metamaterials.
KeywordsRandom forest·Generative design·Functional response·Uncertainty quantification
1 Introduction
Metamaterials are artificially engineered materials whose extraordinary properties arise from their geometry rather than
material composition [1 –3]. Through meticulous geometric design, they enable a wide range of applications—spanning
thermal [4], mechanical [5], acoustic [6], and optical [7] regimes—that are unattainable with conventional materials.
Designing metamaterials for target functionalities hinges on tailoring unit cell geometries to achieve desired responses.
For instance, thermal cloaks can be realized by tessellating unit cells with tailored thermal conductivity tensors [8, 9].
Mechanical metamaterials exhibiting target deformation behaviors can be achieved through the spatial distributionarXiv:2601.13233v1  [cs.AI]  19 Jan 2026

of microstructures with local target stiffness [10]. Likewise, wave propagation in acoustic metamaterials can be
manipulated by arranging unit cells with desired bandgap properties [11].
The richness of unit-cell responses dictates the achievable system-level functionalities. Existing metamaterials design
efforts primarily focus on vector-valued responses, such as stiffness tensors [10, 12 –17], bandgap ranges [18 –21],
thermal conductivity tensors [8], or their combination [22], all of which can be represented as finite-dimensional
vectors. Although vector-valued responses serve as an effective proxy for the underlying behaviors, they are often
insufficient to fully characterize them. For example, stiffness characterizes only the linear elastic regime of the
stress–strain relation but cannot describe nonlinear mechanical behaviors such as snap-through. In comparison,
functional responses—expressed as infinite-dimensional functions [23, 24]—can describe nonlinear behavior with
complex physical mechanisms (e.g., nonlinear stress–strain relation) or properties that vary under contextual factors
(e.g., temperature-dependent thermal conductivity). Functional responses carry the low-level underlying behaviors
from which vector-valued responses are derived. Thus, the design formulated with functional responses enables to
unlock more advanced, sophisticated functionalities possibly unattainable by the other. For example, a soft actuator can
be spatially and temporally programmed by distributing unit cells with prescribed stress–strain relations [25] (Fig. 1
(a)). Directional noise filtering can be achieved by designing unit cells with desired dispersion relations that meet
specific passband/stopband requirements [26] (Fig. 1 (b)). By tailoring the material’s thermal conductivity across
different temperatures, temperature-switchable cloaking and concentrating can be realized [27] (Fig. 1 (c)). ther
demonstrations in the literature include advanced functionalities and applications such as shoe midsoles with tunable
dynamic performance [28], lacrosse chest protectors and vibration-damping panels [29], and energy absorption and
dissipation [30–32].
Despite their promise, designing metamaterials with desired functional responses remains challenging due to the
high-dimensionality of the responses and the non-uniqueness or non-existence of design solutions. To handle the
infinite-dimensional nature of functions, a common practice is to discretize them into high-dimensional vector [25, 33].
Such high dimensionality, often coupled with the nonlinearity of underlying structure–property relations, not only
makes the evaluation of functional responses computationally expensive but also complicates the mapping between the
design and response spaces. Moreover, a given design requirement on the functional response may admit many feasible
designs if it is relatively easy to satisfy, or none if it is inherently stringent [23, 34]. In particular, design requirements
imposed on functional responses are expressed in various forms: they may specify exact response values (e.g., an entire
stress–strain curve shown in Fig. 1 (a)) or only enforce certain characteristics (e.g., acoustic bandgap properties of the
dispersion relation shown in Fig. 1 (b)). Such complexity in requirement specification further challenges the inverse
design.
Conventional optimization methods, such as topology optimization [35], have been widely used for the metamaterials
design involving functional responses [30, 36 –39]. In these approaches, the design requirements are formulated as
objective functions, e.g., minimizing the discrepancy between the targeted stress–strain relation and the obtained
one [30, 36], and design solutions are obtained through iterative evaluations of structure–property relations. Gradient-
based topology optimization has proven effective for inverse design of acoustic bandgaps [37,38], topological states [39],
and nonlinear stress—strain relations [30, 36]. However, analytical sensitivity analysis applies only to analytically
differentiable objectives, limiting flexibility in design requirement specification. To bypass sensitivity derivation,
non-gradient-based optimization methods such as genetic algorithms have been employed [31, 40, 41], providing
greater flexibility in handling complex design requirements that are difficult to formulate as differentiable objectives.
Nonetheless, they tend to require a large number of evaluations of functional responses, which are often time-consuming.
Iterative optimications in general are sensitive to the initial guess, which limits their ability to efficiently explore diverse
design solutions.
Data-driven design [42, 43] offers a promising alternative for inverse design of metamaterials with functional responses.
Central to the framework is a surrogate forward model that can be trained for rapid, on-the-fly prediction without costly
evaluations [42], then be integrated with optimization at the downstream to accelerate the search process [44 –47], or
combined with another machine learning model supporting one-shot, iteration-free inverse design [25, 28, 33, 48, 49].
More recently, conditional deep generative models including generative adversarial networks [50, 51], diffusion
model [52 –54], variational autoencoders [55], have been applied to generate design candidates conditioned on functional
responses, such as stress–strain relations and acoustic dispersion relations. Since generative models are inherently
stochastic, they can more effectively accommodate one-to-many mappings than deterministic models [56]. The diversity
of design solutions offers the flexibility to consider additional design targets beyond the primary requirement.
Nonetheless, handling the high dimensionality of functional responses with machine learning tends to require large
datasets—often on the order of 103-105samples [25, 28, 33]. When the computational cost of evaluating functional
responses is high, acquiring such large datasets becomes impractical. Furthermore, most generative models are
conditioned directly on the response rather than on the requirement [50 –53, 55, 56], only supporting requirements
2

Figure 1: Illustration of inverse design on functional responses across different classes of metamaterials. In each
case, the middle panel shows the functional response, while the bottom panel illustrates the corresponding design
requirement. (a) Mechanical metamaterials: The functional response is the stress–strain relation, where stress σis a
function of strain ϵ. The design requirement is a prescribed nonlinear mechanical behavior, represented by the blue
dashed line. (b) Acoustic metamaterials: The functional response is the dispersion relation, where frequency ωis a
function of band index iand wave vector k. The design requirements are indicated by blue and gray shaded areas,
representing the desired passbands and stopbands, respectively. (c) Thermal metamaterials: The functional response is
the temperature-dependent thermal properties, where thermal connectivity κis a function of temperature T. The design
requirement—high conductivity at high temperatures and low conductivity at low temperatures—is illustrated by the
blue dashed line. (d) Schematic illustrating the relationship among unit-cell geometry, functional responses, and design
requirements in metamaterials design. The form of the design requirements could vary across applications and is often
complex to specify.
that are tied to specific responses (e.g., the entire dispersion relations must be provided as the conditioning response
for acoustic bandgap inverse design [33]). In addition, because existing generative design approaches provide no
uncertainty estimate in the conditional inference, we cannot assess the trustworthiness of the generated design solutions.
This is particularly problematic given the potential non-existence or non-uniqueness of feasible solutions: without
uncertainty assessment across them, the method may produce designs that are actually infeasible, especially when the
requirement lies outside the training distribution or cannot be satisfied within the predefined design space [23].
To date, most data-driven design approaches have employed neural networks (NN) [42,43]. While their strong expressive
power facilitates the accurate prediction of functional responses, NNs are data-hungry, require extensive hyperparameter
tuning, and often demand resource-intensive training. In contrast, random forests—another widely adopted machine
learning method—are relatively easy to construct, exhibiting greater stability to training parameter variations [57].
They are particularly advantageous in small-data regimes, which can reduce the data demand in metamaterials design,
where simulations to obtain functional responses are computationally intensive. Furthermore, their ensemble structure
naturally supports uncertainty quantification of the model predictions [58]. Random forests have proven effective
for predicting functional response such as stress–strain relations [59] and optical transmission spectra [60], but their
inherent capability for uncertainty quantification has rarely been utilized for inverse design. Our previous work [24]
3

leverages the ensemble structure of random forests to provide confidence estimates for generated solutions. The data
efficiency of this approach was demonstrated through spectral inverse design of acoustic metamaterials and optical
metasurfaces, requiring fewer than 250 samples in each example. Nevertheless, this framework was limited to simplified,
qualitative functional responses—such as binary representation of spectral behavior in acoustics and optics—and did
not address fully quantitative ones.
In this work, we propose a RAndom-forest-based Generative design framework (RAG), which can tackle the inverse
design of quantitative, high-dimensional functional responses with complex requirement specification, small-data
compatibility, and uncertainty quantification over generated designs. A random forest is trained for forward mapping,
where each decision tree predicts the entire functional response from a given unit-cell geometry. Leveraging this full
functional-response information, complex design requirements can be specified on either specific response values or
characteristics extracted from the response. Each tree can then classify whether its predicted response satisfies that
requirement. Aggregating the votes across trees in the random forest yields a likelihood estimate conditioned on the
requirement, indicating the model’s confidence that a given design can achieve the on-demand functional response. The
one-to-many mapping can be resolved through iteration-free, single-shot design generation by sampling the design
space based on the conditional likelihood distribution.
In the context of metamaterial inverse functional response design, the contributions of RAG are as follows:
1.Data-efficiency.RAG can predict high-dimensional functional responses accurately using small datasets,
which is beneficial given the high computational cost of evaluating functional responses.
2.Handling complex requirement specification.RAG can accommodate complex forms of design requirements
on functional response, providing greater freedom in inverse functional response design.
3.Uncertainty quantification over generated solutions.RAG can provide predictive uncertainty when
generating design solutions. The uncertainty information can help prevent accepting designs with low
confidence in meeting requirements, improving trustworthiness especially when requirements lie outside of
distribution.
To validate the proposed framework, we investigate two functional response inverse design tasks: (1) acoustic metamate-
rials with prescribed partial passband/stopband behavior, and (2) mechanical metamaterials with targeted snap-through
responses [25]. Both studies are conducted with substantially smaller datasets than existing works [25, 28, 33, 44]—500
samples for the acoustic case and 1057 samples for the mechanical case. The design requirements in both tasks are
difficult to formulate in differentiable objectives thus challenging to solve using topology optimization. The results
demonstrate that RAG can efficiently handle high-dimensional functional response and complex design requirements
with small data. The conditional likelihood, which indicates the model’s uncertainty, not only reflects the trustworthiness
of generated designs for a given requirement, but also reveals the relative difficulty across different requirements, thereby
helping to prevent the generation of infeasible designs. The data efficiency is validated on a public dataset by comparing
with a NN-based design framework [25]. Overall, this lightweight and uncertainty-aware design framework provides a
fast and easy-to-implement approach for inverse functional response design of metamaterials in the small-data regime.
The remainder of the paper is organized as follows. Sec. 2 introduces the mathematical formulation of the functional
response inverse design problem and the architecture of RAG. In Sec. 3, we validate RAG’s small-data compatibility
and uncertainty quantification capability on acoustic metamaterials design (Sec. 3.1) and mechanical metamaterials
design (Sec. 3.2). Discussions and conclusions are stated in Sec. 4 and Sec. 5, respectively.
2 Methods
In this section, we first provide a formal description of the inverse functional response design problem of our interest and
discuss the associated key challenges, including the high-dimensionality of the functional responses, complex design
requirements, and potential non-existence or non-uniqueness of design solutions (Sec. 2.1). To address these challenges,
we introduce the RAG framework (Sec. 2.2), which enables data-efficient forward prediction of high-dimensional
functional responses (Sec. 2.2.1) and uncertainty-informed inverse design under complex requirement specifications
(Sec. 2.2.2).
2.1 Problem Formulation of Inverse Functional Response Design
The forward mapping from a metamaterial unit cell design to its functional response shown in Fig. 2 (a) and (b) can be
expressed as
F(x) =f,(1)
4

where F: Ω x→ {f: Ω a→R} .xis the vector of design variables with dimension dxrepresenting the unit cell
geometry. The corresponding design space is Ωx⊂Rdx.fdenotes the functional response of interest, defined over the
domain Ωa⊆Rdaof the query point a. For example, fcan represent a stress–strain relation σ(ϵ) (where query point a
is the strain ϵ) or dispersion relation ω(i, k) (where a= (i, k) , with iandkdenoting the band index and wave vector,
respectively).Ω aspecifies the corresponding range of interest.
Figure 2: Formulation of forward mapping and inverse design, illustrating the relationship between (a) the unit cell
geometry x, (b) corresponding functional response fand its discrete form y, (c) design requirement T, (d) desired
functional response setYT
G, and (e) design solution setXT.
Since the functional response fin Eq. 1 is infinite-dimensional and difficult to directly handle, we discretize it over a
sequence of query points (a1, . . . ,a dy)in its domain Ωa, as shown in Fig. 2 (b). Evaluating fat these query points
defines a discretization operator
G(f) =y,(2)
where G:{f: Ω a→R} →Rdy.yis the discretized functional response, with components yq=f x(aq), q=
1, . . . , d y. The specific definition of the discretization operator Gdepends on the sampling strategy over Ωa, which must
ensure ypreserves the necessary information of the original infinite-dimensional functional response f. In practice,
highly nonlinear areas of fnecessitate a higher density of query points for an accurate representation. Therefore, more
complex functional responses often require more query points for discretization, leading to high dimensionality. Without
loss of generality, we consider uniform sampling for simplicity throughout this work. Specifically, the domain Ωais
sampled uniformly, where each dimension ajis discretized into njequidistant points (aj,1, . . . , a j,nj), j= 1, . . . , d a.
Consequently, the total dimension of yisdy=Qda
j=1nj. This indicates that the dimension of yscales exponentially
with the dimensionality of a, since each additional dimension multiplies the number of required query points. Combining
Eqs. 1 and 2, the discretized forward mapping can be obtained as
(G ◦ F)(x) =y,(3)
whereG ◦ F: Ω x→Rdy.
The design requirement defines which functional responses are considered desirable. To facilitate determining whether
a response is desired, the requirements are imposed on the discrete form of functional response y. We define the
desired response set YT
Gas the set of all discretized responses y, constructed under G, that satisfy the requirement T(as
shown in Fig. 2 (d)). A straightforward type of Tis to prescribe an target functional response y∗, such as a nonlinear
stress–strain relation [23, 28, 54, 61, 62], in which case YT
G={y∗}. If some deviation from the target is acceptable, the
requirement then admits a set of responses within a tolerance band YT
G={y| |y i−y∗
i| ≤δ i,∀i= 1, . . . , d y}, where
δidenotes the allowable deviation in each component. In other cases, the requirement concerns certain characteristics
5

of the functional response rather than its full profile. Typical examples include specifying plateau stress or absorbed
energy in mechanical metamaterials [14, 25], or targeting a specific full bandgap with the lower bound ωland upper
bound ωuin acoustic metamaterials [63] (as illustrated by the gray shaded regions in Fig. 2 (c) and (d)), which can
be expressed as YT
G={y|y i̸∈[ω l, ωu],∀i= 1, . . . , d y}. Under such characteristic-related requirements, the full
functional response is not uniquely determined, meaning that responses with distinct overall patterns may still be
feasible. Consequently, the elements in YT
Gcan exhibit substantial diversity, as illustrated in Fig. 2 (d). This, in turn,
often leads to a diverse set of feasible design solutions.
When the requirement Tis specified, inverse design aims to find the corresponding design solution set XT, which
consists of all designs in the design space Ωxwhose functional responses satisfy T, as shown in Fig. 2 (e). Based on
the discretized forward mapping in Eq. 3,XTis defined as
XT=
x∈Ω x|(G ◦ F)(x)∈ YT
G	
.(4)
It is worth noting that the solution set XTin Eq. 4 may contain multiple feasible designs if Tis easy to satisfy, or may
be empty if Tis tough to reach within the given design space. This non-uniqueness or potential non-existence of design
solutions introduces significant challenges in inverse functional design.
In summary, the high-dimensional nature of functional responses, the complexity of design requirements, and the
non-uniqueness or potential non-existence of design solutions render the inverse design process highly challenging.
In Sec. 2.2, we will present the proposed RAG framework and demonstrate how it systematically addresses these
challenges.
2.2 The RAG Framework
2.2.1 Forward Prediction with Uncertainty Quantification
According to Eq. 4, identifying the solution set XTrequires building a surrogate model for the forward mapping
y= (G ◦ F)(x) . Here Frepresents the physical mapping which can not be changed, while the discretization operator
Gis manually specified based on the nonlinearity of the response. The most straightforward way to construct the
surrogate model is to take xas input and yas output. However, learning a mapping to high-dimensional output spaces
suffers from the curse of dimensionality, necessitating prohibitive amounts of data for effective generalization. To
mitigate this issue, existing approaches often rely on (i) dimensionality reduction methods (e.g., principal component
analysis [44, 64]) to obtain a low-dimensional form of y, (ii) manual parameterization yinto a compact form [28], or
(iii) partitioning yinto several segments and train separate models for each [33]. Despite their effectiveness, these
methods still require large datasets and complex model architectures [25, 28, 33]. Furthermore, because the learned
mapping(G ◦ F)is tied to a fixed discretizationG, disabling to make predictions at arbitrary query points insideΩ a.
These two issues can be addressed by reformulating the original mapping in Eq. 1 as,
ϕ(x,a) =F(x)(a),(5)
where ϕ: Ω x×Ωa→R . As shown in Fig. 3 (a), by taking both the design xand the query point aas input, the learned
surrogate ˆϕcan predict the response at arbitrary location a∈Ω a. This discretization invariance allows the functional
response to be transferred across different discretizations. For a given discretization operator G, the corresponding
discrete response is obtained as y= (G ◦ϕ)(x) , with each component given by yq=ϕ(x,a q). Additionally, since
the output dimensionality is one, the surrogate can be trained with less data and learn the forward mapping more
efficiently [65].
In RAG, random forest is implemented to approximate the mapping in Eq. 5. Random forests are an ensemble learning
method that constructs multiple decision trees trained on the same task and aggregates their predictions to improve
accuracy. Compared with NNs, random forests can work effectively on small datasets, require less hyperparameter
tuning, and train more efficiently. Moreover, they naturally handle mixed data types, as xmay contain both numerical
and categorical variables. After training the random forest with Ntrees, each tree ˆϕ(n)predict its own discrete
functional response ˆy(n)= (G ◦ ˆϕ(n))(x), n= 1, . . . , N . The ensemble ˆϕis then defined as the average of all trees
ˆϕ(x,a) =1
NPN
n=1ˆϕ(n)(x,a), and its corresponding discrete prediction is ˆy= (G ◦ ˆϕ)(x).
Beyond improving accuracy through averaging, the ensemble structure of random forests also provides predictive
uncertainty, which arises from the variability among individual tree outputs [58, 66], as shown in Fig. 3 (a). The
variance of the tree predictions reflects the degree of disagreement among trees: small variance indicates consistent,
confident predictions, while large variance suggests greater uncertainty. For computational simplicity, we ignore
the correlation among tree outputs and estimate the predictive uncertainty using the sample variance: ˆs2(x) =
1
NPN
n=1 
(G ◦ˆϕ(n))(x)−(G ◦ ˆϕ)(x)◦2, where (·)◦2indicates element-wise squaring. The resulting ˆs2(x)∈Rdyis
6

Figure 3: Illustration of the RAG framework. (a) Step 1: forward prediction of yfromxwith uncertainty quantification
(c) Step 2: uncertainty-informed inverse design given a design requirementT.
a vector-valued uncertainty estimate, with each entry ˆs2
q(x)quantifying the predictive variance at the corresponding
query point aq. This provides an estimate of the model’s confidence, enabling us to assess the trustworthiness of the
predictions used in the inverse design process.
2.2.2 Uncertainty-Informed Inverse Design
As illustrated in Fig. 2 (c) and (d), when the design requirement Tis specified, the desired functional response set
YT
Gcan be defined. For a given design x, each decision tree ˆϕ(n)in the random forest determines whether xsatisfies
the requirement (shown in Fig. 3 (b)). Based on the proportion of trees that classify xas feasible, we can estimate a
conditional likelihood that quantifies how likelyxsatisfiesT:
L(x| T) =1
NNX
n=11{(G◦ ˆϕ(n))(x)∈YT
G},(6)
where 1is the indicator function that returns 1 when the predicted response ˆy(n)= (G ◦ ˆϕ(n))(x) from tree nbelongs
to the desired response set YT
Gand 0 otherwise. The conditional likelihood in Eq. 6 reflects the model’s predictive
confidence. The more trees voting that xis feasible, the higher the model’s confidence that xsatisfies the requirement.
We remark that computing L(x| T) in Eq. 6 relies solely on forward prediction, therefore no additional machine-
learning module is required to learn an inverse mapping. Moreover, the likelihood is conditioned directly on T. Existing
7

generative models often learn the conditional distribution p(x|y) , which condition directly on the responses yrather
than requirement T[50–53, 55, 56]. As a result, these methods generally require manually selecting a representative
response y∈ YT
Gand providing it to the generative model to generate design solutions. This can be challenging
when the desired functional response cannot be explicitly defined. Moreover, such heuristic selection also inevitably
introduces bias that limits solution diversity, especially when YT
Gcontains highly diverse response patterns, as illustrated
in Fig. 2. In comparison, in our framework, as long as the requirement Tcan be formulated as constraints defining YT
G,
the corresponding conditional likelihood can be estimated directly for design generation. This provides substantial
flexibility in how design requirements can be specified, as further demonstrated in Secs. 3.1 and 3.2.
Based on the likelihood function, design solutions satisfying the requirement are obtained in a single shot by sampling
from the likelihood distribution L(x| T) over the design space. Because L(x| T) may exhibit a complex shape
depending on T, we adopt the Metropolis-Hastings algorithm [67]—a widely used Markov Chain Monte Carlo (MCMC)
method—to generate samples from this likelihood. In each iteration, the algorithm proposes a new state according to a
specified proposal distribution and decides whether to accept it based on the Metropolis-Hastings acceptance probability.
This process constructs a Markov chain whose stationary distribution coincides withL(x| T).
At iterationk, the new proposed statex k+1is drawn from:
xk+1∼ N 
xk,diag(σ2)
,(7)
where the proposal standard deviation is defined as
σ=c 0d−1/2
x|u−l|,(8)
withlandudenoting the lower and upper bounds of the design variables, respectively. Scaling the proposal standard
deviation by the variable ranges allows the proposal step size to adapt naturally to the design space. c0serves as a
scaling parameter controlling the overall step size. Since the Gaussian proposal is symmetric, the acceptance probability
can be expressed as
α(xk,xk+1) = min(1,L(xk+1| T)
L(xk| T)).(9)
Designs with higher likelihood values have a greater probability of being sampled, meaning that the RAG framework
favors generating design solutions in regions where the surrogate model is more confident. In addition, the likelihood
associated with each generated solution provides a quantitative indicator of its trustworthiness. If the requirement Tis
inherently difficult—or even impossible—to satisfy within the design space Ωx, it will lead to a relatively low likelihood
distribution across the entire domain. Thus this likelihood distribution can serve as a proxy that quantifies task difficulty
across different design requirements, which is often not available in inverse design with deep generative modeling.
3 Results
In this section, we evaluate the performance of RAG on the inverse design of dispersion relations in acoustic metamate-
rials (Sec. 3.1) and stress–strain relations in mechanical metamaterials (Sec. 3.2). For both tasks, the random forest is
configured with 100 trees, setting the minimum number of samples for node splitting to 2 and for leaf nodes to 1. Gini
impurity [68] is used as the node-splitting criterion. To account for different levels of nonlinearity, the maximum tree
depth is adjusted for each task to balance predictive capability and computational efficiency.
3.1 Design Acoustic Metamaterials with On-Demand Partial Passbands/Stopbands
3.1.1 Problem Statement
Capitalizing on the unconventional capability to control propagation of acoustic/elastic waves in specified frequency
ranges or directions, acoustic metamaterials have enabled a wide range of applications in vibration control [69], signal
processing [40], and energy harvesting [70]. In general, designing acoustic metamaterials with targeted functionalities
hinges on inversely designing unit cells that realize a desired dispersion relation [71, 72]. Therefore, the functional
response fin Eq. 1 is a frequency function ω(i,k) , with the query point a= (i,k) , where idenotes the band index
andkdenotes the wave vector. Although dispersion relations comprehensively describe the wave properties supported
by a lattice of unit cell, their high dimensionality often leads existing works to reduce them to a few frequency-based
features (e.g., full bandgap range) [24, 73, 74]. This simplification limits the potential of acoustic metamaterials for
more sophisticated wave manipulation. In this work, we demonstrate that our RAG can achieve full dispersion relation
inverse design through an acoustic metamaterial case study.
Inspired by the micro-architected metamaterial design proposed by Sun et al. [75], which have been experimentally
demonstrated to exhibit a sufficiently rich dynamic property space, we consider a two-dimensional unit-cell design
8

with a similar geometric pattern in this case study, as shown in Fig. 4(a). Circular micro-inertia are added to the center
and corners of a braced square unit cell with strut radius rstrut. The micro-inertia placed at the center of the brace has
radius rcenter while those at the corners of the square unit cell has radius rcorner. The design space Ωxis defined by 4
≤r strut≤6.41,√
2rstrut≤r center≤20, and (√
2 + 1)r strut≤r corner≤20 (unit: mm). The lower bounds for rcenter and
rcorner are defined in relation to rstrutto ensure that the center and corner circles are not completely overlapped by the
struts. The unit cell size is set at l= 60 mm. The Young’s modulus E, density ρand Poisson’s ratio νare set as 70
GPa, 2700 kg/m3, and 0.33 respectively. To demonstrate the flexibility of requirement specification, we consider design
requirements consisting of multiple partial passbands and stopbands (i.e., passbands and stopbands defined over specific
wave-vector ranges).
Figure 4: (a) Design space and response space in the 2d acoustic metamaterials in the case study. The unit cell structure
is specified by three geometric parameters x= [r strut, rcenter, rcorner]. (b) Forward prediction accuracy of random forests
under different maximum tree depths, quantified by mean squared error (MSE). (c) Forward prediction with uncertainty
for three testing samples. The uncertainty is quantified by the ±2standard deviation denoted as ±2ˆs. (d) Average
uncertainty (quantified by2 ¯ˆs) of dispersion relations across different band orders.
3.1.2 Data Acquisition
The governing equation of the elastic wave propagation can be stated as [38, 76]:
ρ¨u=∇(λ+µ)∇ ·u+∇ ·µ∇u,(10)
9

where ρdenotes the material density, u={u x, uy}⊤is the displacement vector, λandµare the Lamé coefficients. To
homogenize an infinite tessellation of an acoustic metamaterial unit cell, the periodic Bloch-Floquet boundary condition
is applied:
u(r,k) = ˜u(r)ei(kTr+ωt),(11)
where r={x, y} represents the position vector, ˜urepresents the Bloch displacement vector and k= (k x, ky)is
the wave vector. Substituting Eq. 11 into Eq. 10 and discretizing based on the finite element method, the following
eigenvalue problem can be obtained: 
K(k)−ω2M(k)˜u= 0,(12)
where KandMare the Bloch-reduced global stiffness matrix and mass matrix, respectively. Since both KandM
depend on the wave vectork, the obtained eigenfrequenciesωare also functions ofk.
A total of 500 geometric parameter sets, x= (r strut, rcenter, rcorner), are generated through Latin hypercube sampling
within the design space. Bloch-wave analysis is conducted in COMSOL Multiphysics to solve Eq. 12 and obtain the
dispersion relations. The dispersion relation is computed by sampling wave vectors along the high-symmetry path
Γ→X→M→Γ , which traces the boundary of the first irreducible Brillouin zone (IBZ), as illustrated in Fig. 4(a).
The IBZ boundary is uniformly discretized into 61 points. At each sampled wave vector, the first 15 eigenfrequencies
are extracted, yielding a discrete functional response vectorywith dimensiond y= 915.
3.1.3 Forward Prediction
After data acquisition, 400 samples were randomly selected as training data for the random forest to learn the forward
mapping, and the remaining 100 samples were held out for testing. Following the reformulated mapping in Eq. 5, we
use the design variable xtogether with the query point a= (i, k) to predict the corresponding frequency, meaning each
component of the response yis predicted individually. Therefore, the input dimension of the random forest is 5 and the
output dimension is reduced to 1. Accordingly, the number of reformulated input–output pairs scales by a factor of
dy, resulting in 366,000 pairs for training and 91,500 pairs for testing, respectively. Maximum tree depth is tuned to
balance fitting capability and computational cost. As shown in Fig. 4 (b), when the depth exceeds 20, the mean squared
error (MSE, defined as1
dy(y− ˆy)⊤(y− ˆy)) in both the training and testing sets no longer decreases. Therefore, the
maximum tree depth is set to 20, yielding an MSE of 0.0108 for training and 0.4839 for testing.
The forward predictions for three samples from the testing set are shown in Fig. 4 (c). The blue points represent the
mean predictions of the random forest, while the red points denote the ground-truth values obtained from COMSOL
simulations. The shaded blue regions indicate the ±2standard deviation ranges (denoted as ±2ˆs) across predictions
from individual decision trees, providing a measure of predictive uncertainty as introduced in Sec. 2.2.1. Overall,
the ground-truth values are generally well captured within the ±2standard deviation bounds. Since higher-order
eigenfrequencies generally exhibit greater variability, they are usually more challenging to fit and have a lower
prediction accuracy compared to lower-order frequencies [33]. This is captured by the quantified uncertainty, as shown
in Fig. 4 (d). The blue and red solid lines denote the prediction uncertainty across different band orders for the training
and testing sets, respectively, quantified as the average of two standard deviation 2¯ˆs. It is observed that as the band
order increases, the standard deviation increases. A similar trend can also be seen in individual predictions in Fig. 4(c).
This consistency between uncertainty and prediction accuracy indicates that the variance offers a reasonable measure of
predictive uncertainty.
3.1.4 Inverse Design
After training the forward model, we perform inverse design using the same model. To evaluate the inverse design
performance, 10 partial passband/stopband requirements Tkwere prescribed, as shown in Table 1. Each requirement
contains 2–3 segments, where each segment specifies a wave-vector range and a frequency range. The wave-vector
interval is constructed by randomly selecting the starting point from {Γ, X, M} and the ending point from {X, M,Γ} ,
with the constraint that the starting point precedes the ending point along the IBZ boundary. The frequency range is
sampled within [20,50] kHz subject to a prescribed bandwidth constraint (1–5 kHz), and non-overlapping frequency
intervals are enforced across segments. Once the segments are created, each segment is randomly assigned as either a
stopband or a passband. For each requirement, 30 designs are generated using the proposed method by sampling from
the corresponding likelihood distribution over the design space. Note that it is possible for the likelihood to be zero
everywhere in the design space when the model believes the requirement is unachievable. Such cases are excluded as it
is impossible to sample designs from such likelihood distribution.
Fig. 5 (a) and (b) show the designs generated through RAG for requirements T1andT2, respectively. The blue shaded
area denotes the target passband and the gray shaded area denotes the target stopband. For each requirement, we present
the five highest-likelihood designs. For T1, which imposes relatively few constraints, feasible designs are easy to identify,
10

Table 1: Design requirements used for inverse design. Each band is specified by a wave-vector range and a frequency
range (unit: kHz).
Requirement Stopbands Passbands
T1 [Γ−X],[45.48,47.52] [X−M],[37.49,39.12]
T2[M−Γ],[26.20,27.94]
[X−M−Γ],[47.85,49.91][M−Γ],[22.83,25.49]
T3[X−M−Γ],[26.57,28.22]
[Γ−X],[21.91,24.34][X−M],[45.27,47.46]
T4 [Γ−X],[47.04,48.39] [Γ−X],[43.09,45.96]
T5 [M−Γ],[45.20,46.21] [Γ−X−M−Γ],[28.42,30.15]
T6 [M−Γ],[44.96,47.17][Γ−X],[26.87,28.72]
[X−M],[32.96,34.03]
T7[M−Γ],[26.95,28.12]
[Γ−X−M],[32.14,33.41]None
T8[M−Γ],[27.66,28.82]
[Γ−X−M],[35.54,38.05]None
T9[Γ−X−M],[37.71,40.34]
[X−M],[24.94,26.90]None
T10[Γ−X],[30.37,33.22]
[Γ−X],[46.33,48.07]None
thus all five designs generated by RAG attain high likelihood (all higher than 0.90) and all satisfy the requirement
as confirmed by simulation. Since many dispersion relations can satisfy these constraints, the corresponding design
solutions are distributed across multiple regions of the design space. This characteristic is also well captured by RAG,
as the generated unit-cell geometries and their corresponding dispersion relations are clearly distinct. This demonstrates
that RAG effectively handles the one-to-many nature of the inverse problem by producing diverse solutions for a single
requirement. In comparison, for T2that introduces more constraints, the likelihood of the generated designs is relatively
low (all lower than 0.4) and none of them satisfy the requirement. Overall, likelihood correlates strongly with feasibility:
higher-likelihood designs are more likely to meet the requirement. Leveraging this information allows us to assess the
complexity of a given inverse-design requirement and mitigate risk by excluding low-likelihood designs.
Fig. 6 (a) shows an overview of all 10 prescribed requirements. The gray bars denote the number of training samples
satisfying each requirement. In general, the more feasible samples present in the dataset, the easier it is to achieve the
corresponding requirement within the design space. Accordingly, requirements such as T1,T4, andT5are relatively
easy to achieve, while T2,T3,T8, andT9are more difficult because no feasible samples exist in the training dataset.
This difficulty is effectively captured by the average likelihood shown in the blue curve: requirements that are easier
to achieve exhibit higher average likelihoods, while those lacking feasible samples show lower values. This trend is
consistent with Fig. 5, confirming that likelihood serves as an effective indicator of requirement difficulty.
To show that RAG can produce designs that satisfy the requirement, we compute the satisfaction rate—defined as the
fraction of generated designs meeting the requirement (yellow curve)—and the average overlap rate, which measures
the average frequency range overlap between a generated response and the requirement. The satisfaction rate is
computed based on a binary measure (a design either satisfies the full requirement or not), while the overlap rate offers
a softer measure that accounts for partial satisfaction. By construction, the average overlap rate is lower-bounded
by the satisfaction rate, revealing when the model generates designs that, although infeasible, remain close to the
requirement. For relatively easy requirements ( T1,T4,T5), both satisfaction and overlap rates are high, whereas for
difficult requirements ( T2,T3,T8,T9) both measures are low. Importantly, satisfaction and overlap rates can only be
obtained through explicit feasibility validation. In contrast, our framework leverages likelihood information to flag
designs that are less likely to meet the requirement prior to validation.
Fig. 6 (b) shows the kernel density estimation (KDE) of the likelihood for all 300 generated designs, conditioned on
their corresponding requirements. It can be observed that infeasible designs generally exhibit low likelihood, whereas
feasible designs are associated with high likelihood values. Therefore, by applying a likelihood threshold to filter out
low-likelihood designs, the overall satisfaction rate of the generated designs can be increased. The relationships among
the selection rate, satisfaction rate, overlap rate, and the likelihood threshold are shown in Fig. 6(c). As the likelihood
threshold increases and low-likelihood designs are progressively filtered out, the selection rate decreases, while the
remaining designs exhibit increasing satisfaction and overlap rates. This also demonstrates that the likelihood threshold
can be flexibly adjusted according to the acceptable confidence level of different tasks.
11

Figure 5: Inverse design results for acoustic metamaterials for two different requirements T1(a) and T2(b). The blue
shaded area indicates the partial passband and the gray shaded area indicates the partial stopband. For each requirement,
the top-5 highest likelihood unit cell geometries and the corresponding dispersion relations are displayed. For T1, all
the five designs meet the requirement. For T2, none meets the requirement. The detailed specifications of all the ten
requirements can be found in Table 1.
3.2 Design Mechanical Metamaterials with Desired Snap-Through Response
3.2.1 Problem statement
Mechanical metamaterials have been widely studied due to the unprecedented mechanical properties that arise from
their unit-cell structures [5]. These properties are often characterized by their stress–strain relations [25, 28, 44, 59],
which can be represented as a function σ(ϵ), with the query point a:=ϵ representing the strain. To demonstrate both
the capability of RAG for stress–strain relation inverse design and its data-efficiency advantage, we apply it to an
existing mechanical metamaterial system proposed by Chai et al. [25], using only 40% of the original training dataset.
In this task, the design space is defined by five variables: the horizontal length of the tilt beam l, the angle between
the cantilever beam and tilt beam α, the width of the tilt beam w, and the number of columns ncoland rows nrow, as
shown in Fig. 7 (a). ncolandnrowcan only take integer values of 1, 2, 3, or 4. The stress–strain relation is uniformly
discretized into 31 points; since the stress at the first point is always zero, we predict only the remaining 30 points,
giving the discrete functional response vector ya dimensionality of 30. In this study, we focus on the reconfigurable
soft actuator inverse design task introduced in [25], where different actuation modes can be achieved under a simple
12

Figure 6: Statistical summary of all 300 designs generated from the 10 prescribed partial passband/stopband require-
ments in Table 1. (a) The overview of the 10 requirements. The gray bar represents the number of training samples
satisfying their corresponding requirements. The blue, yellow, and green curves represent the average likelihood,
satisfaction rates, and average overlap rates, respectively. (b) KDE of the estimated likelihood for all 300 generated
designs. (c) Satisfaction rates, overlap rates, and selection rates for generated designs under varying sampling thresholds.
pneumatic input. The key to achieve this functionality is to find unit cell design that can snap through under specific
force and provide desired stroke. Therefore, the design requirement imposed on the stress–strain relation is specified in
terms of a target snap-through threshold force and stroke, as illustrated in Fig. 7(a).
3.2.2 Forward modeling
Similar to the forward prediction in the acoustic case (Sec. 3.1.3), we jointly take the design variable xand the query
pointa—–which corresponds to the strain ϵin this task—–as inputs to predict the corresponding stress value. A random
forest regressor with the maximum tree depth of 15 is employed for this task. The public ground dataset consists of
2065 training samples and 231 testing samples. To evaluate the data efficiency of RAG, we downsample the original
training set with varying downsampling portions, while keeping the testing set unchanged. Each subset is then used to
train a separate random forest model and the corresponding accuracy in shown in Fig. 7 (b). The results indicate that
random forests remain effective with limited data: when only 10% of the training data are used, the model still achieves
mean relative errors of 2.70% and 7.37% on the training and testing sets, respectively. In comparison, the original
study using NN achieved lower errors of 1.08% (training) and 2.59% (testing), but required the full dataset. This result
shows that, the proposed RAG framework can attain prediction accuracy close to that of NN with a significantly smaller
amount of training data, thereby highlighting its data-efficiency advantage. For subsequent tasks, we select the model
trained on 40% of the training data (826 samples), which achieves 2.09% error on the training set and 5.14% on the
testing set.
Fig. 7 (c) shows the forward predictions with uncertainty quantification for test samples without snap-through, with one
snap-through wave, and with two snap-through waves, respectively. It can be observed that the curves with greater
13

Figure 7: Design problem statement and forward modeling of mechanical metamaterials. (a) Design space and
response space in mechanical metamaterials. (b) Relationship between prediction accuracy, quantified by mean absolute
percentage error (MAPE), and the ratio of data used for model training. The solid blue and red lines represent the
training and testing accuracy of the random forest model, respectively. The dashed blue and red lines represent the
training and testing accuracy of the NN reported in Ref. [25]. (c) Forward prediction and uncertainty quantification for
test samples without snap-through, with one snap-through wave, and with two snap-through waves, respectively.
fluctuation exhibit wider uncertainty ranges. The ground truth values remain within the 2standard deviation bands,
demonstrating both reasonable predictive accuracy and meaningful uncertainty quantification.
3.2.3 Inverse inference
The design requirement is specified as a target threshold force of 0.2 MPa and a target stroke of 1.3 mm. We relax the
requirement by introducing a tolerance on these values—that is, the generated designs are allowed to have threshold
force and stroke values within a specified percentage range of the targets. Requirements with smaller tolerances are
more difficult to satisfy. Here, we consider two different tolerance levels, ±25% and ±15%, to investigate how the
difficulty of the requirement influences the estimated likelihood.
Fig. 8 (a) and (b) visualizes the estimated likelihood distribution over the design space for tolerance levels of ±25% and
±15%. The discrete design variables ncolandnrowdivide the 5-dimensional design space into sixteen 3-dimensional
subspaces. Here we only visualize the subspace corresponding to ncol= 4andnrow= 2as an example. To examine
the accuracy of the estimated likelihoods, we also plot the training samples observed by the random forests (i.e. the
reduced training dataset). Gray points represent samples that do not meet the requirement, while blue points denote
those that satisfy it. It can be observed that regions with high estimated likelihoods are located near feasible samples,
whereas low likelihoods appear in regions surrounded by infeasible ones. Since the requirements on threshold and
stroke do not impose constraints on the specific shape of the stress-–strain relation, different types of stress—strain
relations may satisfy these requirements. Consequently, multiple high-likelihood regions can be observed across the
design space. Moreover, as the tolerance becomes tighter, the overall likelihood decreases while the landscape of the
likelihood distribution remains similar because of the unchanged target. These observations suggest that the likelihood
14

distribution estimated by RAG can effectively reflect the distribution of feasible design solutions, and that the magnitude
of the likelihood also serves as a reliable indicator of the difficulty of the design requirement.
Figure 8: Likelihood distribution in the design space for desired stroke and threshold with different tolerances. The
gray/blue points denote the infeasible/feasible data points in the training dataset (a) tolerance = 25% (b) tolerance =
15%.
4 Discussion
In this section, we discuss the advantages of RAG in terms of flexibility in requirement specification, data efficiency,
and uncertainty awareness. We further analyze its computational efficiency to illustrate how the problem structure
influences the overall computational performance of RAG.
4.1 Flexibility in Requirement Specification
Existing generative models often learn the inverse mapping from the functional response yto the design x. Therefore,
representative functional responses that satisfy the design requirements (e.g., stress–strain relations with desired
threshold and stroke, or dispersion relations with on-demand bandgap property) must be manually specified as inputs to
generate the corresponding design solutions. In contrast, RAG directly learns the inverse mapping from the design
requirement Tto the design xby estimating the likelihood conditioned on the requirement, i.e., L(x| T) . This
eliminates the need to manually specify representative functional responses for design generation and enables greater
flexibility in requirement specification.
In the two design tasks considered in this work—partial passband/stopband control (Sec. 3.1) and snap-through
threshold/stroke control (Sec. 3.2)—the requirements do not prescribe the exact shape of the functional response and
15

thus allow functional responses with diverse shapes. In such cases, it is challenging to formulate the requirements as
differentiable objectives amenable to topology optimization, and identifying representative dispersion or stress–strain
relations for generative models is also nontrivial. Moreover, using a specific representative functional response to
approximate the requirement would inevitably introduce bias and artificially restrict the feasible design space. In
contrast, RAG avoids this source of constraint and enables a more comprehensive exploration of feasible designs across
the design space.
4.2 Data Efficiency
Our method enables forward prediction and inverse design of full dispersion relations and stress–strain relations using
significantly fewer data than those required by existing NN-based approaches for similar tasks. The data efficiency of
RAG mainly arises from two factors. First, random forests are inherently more robust than NN in small-data regimes.
Second, instead of treating the entire high-dimensional functional response as the model output, we introduce the query
points of the functional response as model inputs and reduce the model output dimension to one. This reformulation
substantially lowers the learning complexity, thereby improving data efficiency.
4.3 Uncertainty Awareness
By leveraging the ensemble structure of random forests, our framework can quantify uncertainty both in the response
space during forward prediction and in the design space for arbitrary requirements during inverse design. This capability
is particularly critical for inverse design problems, which can be either one-to-many—where the requirements are
relatively easy and multiple design solutions satisfy the same requirement—or one-to-none—where the requirements
are too stringent to allow any feasible designs within the predefined design space. Most existing generative inverse
design frameworks produce design solutions without providing any measure of the conditional inference. As a result,
even when the requirement is challenging or even impossible to satisfy, the approaches provide no ways to monitor this.
In contrast, our method provides model uncertainty information that helps indicate how difficult a given requirement is
to satisfy within a predefined design space. The trustworthiness of the generated design solutions can then be improved
by selecting those with higher confidence (likelihood) values.
4.4 Computational Efficiency and Scalability
The applicability of random forests is constrained by their computational efficiency. The time complexity of training a
random forest is O(NMklog(M)) , where Nis the number of trees, Mis the number of input–output pairs, and kis
the number of features considered at each split. During data acquisition, we often compute the functional response y
for a given design x. Therefore, the number of training design samples, denoted as m, corresponds to the number of
distinct x−y pairs. Based on the reformulation in Eq. 5, the query point ais introduced as additional inputs to predict
the corresponding functional response value. As a result, each original x-ypair is decomposed into dyinput–output
pairs. Under this setting, we have M=mp
dyandk=√dx+da. The time complexity for training can be then
expressed as [77]
O(Nmq
dy(dx+da) log(mp
dy))(13)
This scaling relation illustrates how the computational efficiency of the random forest depends on the dimensionality of
the functional response and the sample size. Based on Eq. 13, one can evaluate the trade-off between the resolution
of the functional response determined by dyand the training efficiency. For reference, the actual training time for
each scenario in this study is summarized in Table 2, obtained on an Intel Core i7-12700 CPU (2.10 GHz) with 32 GB
of memory. It can be observed that both cases are trained within one minute, which is substantially faster than the
NN-based approach. The acoustic metamaterial case requires a longer training time mainly because the dispersion
relation has a much higher dimensionality than the stress-–strain relation. In the design generation stage, the efficiency
Table 2: Training time of random forests in each task.
Taskd y m d xTraining time (s)
Acoustic metamaterials design 915 400 3 58.88
Mechanical metamaterials design 30 826 5 3.01
of MCMC sampling is primarily influenced by the dimensionality of the design space dx. A high-dimensional design
space makes it more difficult for the MCMC process to explore and locate regions with high likelihood. When the
requirement is challenging to achieve, the corresponding high-likelihood region may be extremely small or even
nonexistent, which further increases the sampling time. To ensure computational efficiency, the proposed method is
therefore more suitable for parametric design spaces with relatively low dimensionality.
16

5 Conclusion
In this work, we presented RAG, a random-forest-based generative inverse-design framework capable of handling
quantitative, high-dimensional functional responses using small datasets. RAG requires training only a single forward
random-forest model, which avoids extensive hyperparameter tuning and enables fast, stable training. Instead of directly
learning the mapping from design to functional response, we take design variables xtogether with the query point aof
the functional response as the input and learn their joint mapping. This reformulation reduces the learning difficulty by
lowering the output dimensionality of the model and enables discretization-invariant prediction. During inverse design,
RAG directly estimates the likelihood conditioned on the requirement, providing substantial flexibility in how design
requirements are specified. Leveraging the random forests ensemble, the framework quantifies uncertainty in both
forward prediction—through the variance across tree outputs—and inverse design—through the estimated likelihood.
These uncertainty measures reflect the model’s confidence in its inferences. Iteration-free, single-shot design generation
can be achieved by sampling the design space from the estimated likelihood distribution.
We validated RAG on two metamaterial systems: acoustic metamaterials with prescribed partial passbands/stopbands,
and mechanical metamaterials with desired snap-through behaviors, where the design requirements are challenging to
address using gradient-based topology optimization. In both cases, RAG operates effectively with far fewer data than
those required by existing deep-learning-based methods, while accurately predicting high-dimensional responses and
producing diverse, feasible design candidates. The likelihood estimates can help secure trustworthiness of the generated
design by filtering low-confidence solutions and indicate requirement difficulty.
In addition, we highlight two potential directions for future work that can further exploit the uncertainty-aware and
easy-to-construct advantages of RAG. First, the uncertainty information provided by RAG in forward prediction
can be leveraged for adaptive sampling to improve data efficiency. By selectively adding new design samples in
high-uncertainty regions, the total number of required samples can be significantly reduced while maintaining prediction
accuracy. It is worth noting that RAG learns a mapping from the joint space of design variables and query points
of the functional response. Therefore, adaptive sampling can be simultaneously applied in the design space Ωxand
the query-point space Ωa. This implies that the discretization of the functional response itself can also be optimized.
Second, owing to its ease of construction, RAG can also be used to quickly assess whether a user-defined design space
is sufficiently rich to satisfy a given design requirement. In practice, it is often unclear whether the predefined design
space is adequate for achieving the requirement. RAG provides an efficient tool for such an examination. If, for a given
requirement, the conditional likelihood distribution estimated by RAG is generally low across the entire design space,
this indicates that the predefined design space is insufficient and should be expanded to meet the requirement.
Overall, RAG offers a lightweight, uncertainty-aware, and data-efficient pathway to metamaterial inverse design
with nonlinear and high-dimensional functional responses. Its flexibility in requirement specification, robustness
in small-data regimes, and compatibility with adaptive learning strategies make it a promising building block for
uncertainty-informed metamaterials design pipelines. Beyond the metamaterials design demonstrated in this work,
RAG can be applied to more general inverse-design problems involving high-dimensional property spaces, expensive
physics-based simulations, and complex functional requirements, possibly upon trivial modifications.
Acknowledgments
Wei Chen, Doksoo Lee, and Bolin Chen acknowledge support from the NSF Boosting Research Ideas for Transformative
and Equitable Advances in Engineering (BRITE) Fellow Program (CMMI-2227641). Wei Chen and Doksoo Lee
are also grateful for partial support from NASA’s Minority University Research and Education Project Institutional
Research Opportunity (MIRO) through the Center for In-Space Manufacturing (CISM-R2): Recycling and Regolith
Processing (Award 80NSSC24M0176). Wei "Wayne" Chen acknowledges support from the NSF EDSE program
(CMMI 2434393). The authors also acknowledge Dr. Mohammad Charara, Rachel Sun, and Prof. Carlos M. Portela for
insightful discussions and constructive comments on the manuscript.
Data Availability Statement
The source code and data for the acoustic metamaterial case in this work are available at: https://github.com/
Sautoy/Random_Forest_Based_Generative_Design_Framework.
17

References
[1]Xianglong Yu, Ji Zhou, Haiyi Liang, Zhengyi Jiang, and Lingling Wu. Mechanical metamaterials associated with
stiffness, rigidity and compressibility: A brief review.Progress in Materials Science, 94:114–173, 2018.
[2]Jun Wang, Gaole Dai, and Jiping Huang. Thermal metamaterial: fundamental, application, and outlook.Iscience,
23(10), 2020.
[3]Steven A Cummer, Johan Christensen, and Andrea Alù. Controlling sound with acoustic metamaterials.Nature
Reviews Materials, 1(3):1–13, 2016.
[4]Sophia R Sklan and Baowen Li. Thermal metamaterials: functions and prospects.National Science Review,
5(2):138–141, 2018.
[5] Pengcheng Jiao, Jochen Mueller, Jordan R Raney, Xiaoyu Zheng, and Amir H Alavi. Mechanical metamaterials
and beyond.Nature communications, 14(1):6004, 2023.
[6]Guangxin Liao, Congcong Luan, Zhenwei Wang, Jiapeng Liu, Xinhua Yao, and Jianzhong Fu. Acoustic
metamaterials: A review of theories, structures, fabrication approaches, and applications.Advanced Materials
Technologies, 6(5):2000787, 2021.
[7]Costas M Soukoulis and Martin Wegener. Past achievements and future challenges in the development of
three-dimensional photonic metamaterials.Nature photonics, 5(9):523–530, 2011.
[8]Yihui Wang, Wei Sha, Mi Xiao, Cheng-Wei Qiu, and Liang Gao. Deep-learning-enabled intelligent design of
thermal metamaterials.Advanced Materials, 35(33):2302387, 2023.
[9]Daicong Da and Wei Chen. Two-scale data-driven design for heat manipulation.International Journal of Heat
and Mass Transfer, 219:124823, 2024.
[10] Liwei Wang, Yu-Chin Chan, Faez Ahmed, Zhao Liu, Ping Zhu, and Wei Chen. Deep generative modeling for
mechanistic-based learning and design of metamaterial systems.Computer Methods in Applied Mechanics and
Engineering, 372:113377, 2020.
[11] Mourad Oudich, M Badreddine Assouar, and Zhilin Hou. Propagation of acoustic waves and waveguiding in a
two-dimensional locally resonant phononic crystal plate.Applied Physics Letters, 97(19), 2010.
[12] Yifan Liu, Wei Huang, Zhiyong Wang, Jie Zhang, and Jiayi Liu. Machine learning-based mechanical performance
prediction and design of lattice structures.International Journal of Mechanical Sciences, 294:110230, 2025.
[13] Xiaoyang Zheng, Ta-Te Chen, Xiaofeng Guo, Sadaki Samitsu, and Ikumu Watanabe. Controllable inverse design
of auxetic metamaterials using deep learning.Materials & Design, 211:110178, 2021.
[14] Kang Ang, Ji Qiu, Buyun Su, Zhiqiang Li, Xiaohu Yao, Zhihua Wang, and Xuefeng Shu. Deep learning-based
inverse design of programmable disordered metamaterials.International Journal of Mechanical Sciences, page
110712, 2025.
[15] Zhenyang Gao, Hongze Wang, Nikita Letov, Yaoyao Fiona Zhao, Xiaolin Zhang, Yi Wu, Chu Lun Alex Leung,
and Haowei Wang. Data-driven design of biometric composite metamaterials with extremely recoverable and
ultrahigh specific energy absorption.Composites Part B: Engineering, 251:110468, 2023.
[16] Shengzhi Luan, Enze Chen, Joel John, and Stavros Gaitanaros. A data-driven framework for structure-property
correlation in ordered and disordered cellular metamaterials.Science Advances, 9(41):eadi1453, 2023.
[17] Zihan Wang, Anindya Bhaduri, Hongyi Xu, and Liping Wang. An uncertainty-aware deep learning framework-
based robust design optimization of metamaterial units.Structural and Multidisciplinary Optimization, 68(3):1–26,
2025.
[18] Jiapeng Sun, Yan Ma, Yulong He, Jianjiao Deng, Xi Chen, Dianlong Pan, Xin Li, Ming-Hui Lu, and Yan-
Feng Chen. Low-frequency vibration attenuation and ensemble learning-based inverse design of vibro-acoustic
metamaterials.Mechanical Systems and Signal Processing, 239:113320, 2025.
[19] Than V Tran, SS Nanthakumar, and Xiaoying Zhuang. Deep learning-based framework for the on-demand inverse
design of metamaterials with arbitrary target band gap.npj Artificial Intelligence, 1(1):2, 2025.
[20] Hongyuan Liu, Yating Gao, Yongpeng Lei, Hui Wang, and Qinxi Dong. Data-driven inverse design of the
perforated auxetic phononic crystals for elastic wave manipulation.Smart Materials and Structures, 33(9):095029,
2024.
[21] Zihan Wang, Weikang Xian, M Ridha Baccouche, Horst Lanzerath, Ying Li, and Hongyi Xu. Design of phononic
bandgap metamaterials based on gaussian mixture beta variational autoencoder and iterative model updating.
Journal of Mechanical Design, 144(4):041705, 2022.
18

[22] Jeonghoon Park, Jaebum Noh, Jehyeon Shin, Grace X Gu, and Junsuk Rho. Investigating static and dynamic
behaviors in 3d chiral mechanical metamaterials by disentangled generative models.Advanced Functional
Materials, 35(2):2412901, 2025.
[23] Haoxuan Dylan Mu, Mingjian Tang, Wei Gao, Wei Chen, et al. Guide: Generative and uncertainty-informed
inverse design for on-demand nonlinear functional responses.arXiv preprint arXiv:2509.05641, 2025.
[24] Wei Chen, Rachel Sun, Doksoo Lee, Carlos M Portela, and Wei Chen. Generative inverse design of metamaterials
with functional responses by interpretable learning.Advanced Intelligent Systems, 7(6):2400611, 2025.
[25] Zhiping Chai, Zisheng Zong, Haochen Yong, Xingxing Ke, Jiaqi Zhu, Han Ding, Chuan Fei Guo, and Zhigang Wu.
Tailoring stress–strain curves of flexible snapping mechanical metamaterial for on-demand mechanical responses
via data-driven inverse design.Advanced Materials, 36(33):2404369, 2024.
[26] Sabiju Valiya Valappil, Alejandro M Aragón, and Johannes FL Goosen. Directional band gap phononic struc-
tures for attenuating crosstalk in clamp-on ultrasonic flowmeters.Mechanical Systems and Signal Processing,
224:112173, 2025.
[27] Pengfei Zhuang, Jun Wang, Shuai Yang, and Jiping Huang. Nonlinear thermal responses in geometrically
anisotropic metamaterials.Physical Review E, 106(4):044203, 2022.
[28] Chan Soo Ha, Desheng Yao, Zhenpeng Xu, Chenang Liu, Han Liu, Daniel Elkins, Matthew Kile, Vikram
Deshpande, Zhenyu Kong, Mathieu Bauchy, et al. Rapid inverse design of metamaterials based on prescribed
mechanical behavior through machine learning.Nature Communications, 14(1):5765, 2023.
[29] Marco Maurizi, Derek Xu, Yu-Tong Wang, Desheng Yao, David Hahn, Mourad Oudich, Anish Satpati, Mathieu
Bauchy, Wei Wang, Yizhou Sun, et al. Designing metamaterials with programmable nonlinear responses and
geometric constraints in graph space.Nature Machine Intelligence, 7(7):1023–1036, 2025.
[30] Zhi Zhao, Rahul Dev Kundu, Ole Sigmund, and Xiaojia Shelly Zhang. Extreme nonlinearity by layered materials
through inverse design.Science Advances, 11(20):eadr6925, 2025.
[31] Qingliang Zeng, Shengyu Duan, Zeang Zhao, Panding Wang, and Hongshuai Lei. Inverse design of energy-
absorbing metamaterials by topology optimization.Advanced Science, 10(4):2204977, 2023.
[32] Tark Raj Giri and Russell Mailen. Controlled snapping sequence and energy absorption in multistable mechanical
metamaterial cylinders.International Journal of Mechanical Sciences, 204:106541, 2021.
[33] Kai Zhang, Yaoyao Guo, Xiangbing Liu, Fang Hong, Xiuhui Hou, and Zichen Deng. Deep learning-based inverse
design of lattice metamaterials for tuning bandgap.Extreme Mechanics Letters, 69:102165, 2024.
[34] Jan-Hendrik Bastek, Siddhant Kumar, Bastian Telgen, Raphaël N Glaesener, and Dennis M Kochmann. Inverting
the structure–property map of truss metamaterials by deep learning.Proceedings of the National Academy of
Sciences, 119(1):e2111505119, 2022.
[35] Ole Sigmund and Kurt Maute. Topology optimization approaches: A comparative review.Structural and
multidisciplinary optimization, 48(6):1031–1055, 2013.
[36] Jiashuo Xu, Mi Xiao, and Liang Gao. Inverse design of structures with accurately programmable nonlinear
mechanical responses by topology optimization.Computer Methods in Applied Mechanics and Engineering,
446:118243, 2025.
[37] Vanessa Cool, Ole Sigmund, and Niels Aage. Metamaterial design with vibroacoustic bandgaps through topology
optimization.Computer Methods in Applied Mechanics and Engineering, 436:117744, 2025.
[38] Qiangbo Wu, Jingjie He, Wenjiong Chen, Quhao Li, and Shutian Liu. Topology optimization of phononic crystal
with prescribed band gaps.Computer Methods in Applied Mechanics and Engineering, 412:116071, 2023.
[39] Pegah Azizi, Rahul Dev Kundu, Weichen Li, Kai Sun, Xiaojia Shelly Zhang, and Stefano Gonella. Lattice
materials with topological states optimized on demand.Proceedings of the National Academy of Sciences,
122(32):e2506787122, 2025.
[40] Xiaopeng Zhang, Jian Xing, Pai Liu, Yangjun Luo, and Zhan Kang. Realization of full and directional band gap
design by non-gradient topology optimization in acoustic metamaterials.Extreme Mechanics Letters, 42:101126,
2021.
[41] Xiaopeng Zhang, Yan Li, Yaguang Wang, and Yangjun Luo. Ultra-wide low-frequency bandgap design of acoustic
metamaterial via multi-material topology optimization.Composite structures, 306:116584, 2023.
[42] Doksoo Lee, Wei Chen, Liwei Wang, Yu-Chin Chan, and Wei Chen. Data-driven design for metamaterials and
multiscale systems: a review.Advanced Materials, 36(8):2305254, 2024.
19

[43] Xiaoyang Zheng, Xubo Zhang, Ta-Te Chen, and Ikumu Watanabe. Deep learning in mechanical metamaterials:
from prediction and generation to inverse design.Advanced Materials, 35(45):2302530, 2023.
[44] Bolei Deng, Ahmad Zareei, Xiaoxiao Ding, James C Weaver, Chris H Rycroft, and Katia Bertoldi. Inverse design
of mechanical metamaterials with target nonlinear response via a neural accelerated evolution strategy.Advanced
Materials, 34(41):2206238, 2022.
[45] Yueyou Tang, Anfu Zhang, Qi Zhou, Mu He, and Liang Xia. Data-driven design of topologically optimized
auxetic metamaterials for tailored stress–strain and poisson’s ratio-strain behaviors.Materials & Design, page
114290, 2025.
[46] Hamza Baali, Mahmoud Addouche, Abdesselam Bouzerdoum, and Abdelkrim Khelif. Design of acoustic
absorbing metasurfaces using a data-driven approach.Communications Materials, 4(1):40, 2023.
[47] Ertai Cao, Zhicheng Dong, Ben Jia, and Heyuan Huang. Inverse design of isotropic auxetic metamaterials via a
data-driven strategy.Materials Horizons, 2025.
[48] Zhibin Liang, Chongrui Liu, Yuze Liu, Yueyin Ma, and Fuyin Ma. Demand-driven inverse design and optimization
for ultra-thin broadband sound-absorbing metamaterials based on metaheuristic-enhanced autoencoder network.
Composites Part B: Engineering, page 112643, 2025.
[49] Weifeng Jiang, Yangyang Zhu, Guofu Yin, Houhong Lu, Luofeng Xie, and Ming Yin. Dispersion relation
prediction and structure inverse design of elastic metamaterials via deep learning.Materials Today Physics,
22:100616, 2022.
[50] Caglar Gurbuz, Felix Kronowetter, Christoph Dietz, Martin Eser, Jonas Schmid, and Steffen Marburg. Generative
adversarial networks for the design of acoustic metamaterials.The Journal of the Acoustical Society of America,
149(2):1162–1174, 2021.
[51] Jiahui Yan, Yingli Li, Guohui Yin, Song Yao, and Yong Peng. Inverse design on customised absorption of acoustic
metamaterials with high degrees of freedom by deep learning.Mechanical Systems and Signal Processing,
237:112989, 2025.
[52] Jan-Hendrik Bastek and Dennis M Kochmann. Inverse design of nonlinear mechanical metamaterials via video
denoising diffusion models.Nature Machine Intelligence, 5(12):1466–1475, 2023.
[53] Nikolaos N Vlassis and WaiChing Sun. Denoising diffusion algorithm for inverse design of microstructures with
fine-tuned nonlinear material properties.Computer Methods in Applied Mechanics and Engineering, 413:116126,
2023.
[54] Qibang Liu, Seid Koric, Diab Abueidda, Hadi Meidani, and Philippe Geubelle. Toward signed distance function
based metamaterial design: Neural operator transformer for forward prediction and diffusion model for inverse
design.Computer Methods in Applied Mechanics and Engineering, 446:118316, 2025.
[55] Che Wang, Yuhao Fu, Ke Deng, and Chunlin Ji. Functional response conditional variational auto-encoders for
inverse design of metamaterials. InNeurIPS 2021 Workshop on Deep Learning and Inverse Problems, 2021.
[56] Wei Ma, Feng Cheng, Yihao Xu, Qinlong Wen, and Yongmin Liu. Probabilistic representation and inverse design
of metamaterials based on a deep generative model with semi-supervised learning strategy.Advanced Materials,
31(35):1901111, 2019.
[57] Victor Rodriguez-Galiano, Manuel Sanchez-Castillo, M Chica-Olmo, and MJOGR Chica-Rivas. Machine learning
predictive models for mineral prospectivity: An evaluation of neural networks, random forest, regression trees and
support vector machines.Ore geology reviews, 71:804–818, 2015.
[58] Hengrui Zhang, Wei Chen, Akshay Iyer, Daniel W Apley, and Wei Chen. Uncertainty-aware mixed-variable
machine learning for materials design.Scientific reports, 12(1):19760, 2022.
[59] Jian He, Yaohui Wang, Hubocheng Tang, Guoquan Zhang, Ke Dong, Dong Wang, Liang Xia, and Yi Xiong.
Customizable bistable units for soft-rigid grippers enable handling of multi-feature objects via data-driven design.
Materials Horizons, 12(12):4426–4433, 2025.
[60] Bhagwat Singh Chouhan, Ali Nawaz, Asit Das, Rohith KM, Amir Ahmad, and Gagan Kumar. Machine learning-
driven ultra-broadband terahertz multilayer metamaterial.Journal of Lightwave Technology, 2024.
[61] Haoyu Wang, Zongliang Du, Fuyong Feng, Zhong Kang, Shan Tang, and Xu Guo. Diffmat: Data-driven inverse
design of energy-absorbing metamaterials using diffusion model.Computer Methods in Applied Mechanics and
Engineering, 432:117440, 2024.
[62] Xin-chun Zhang, Zhi-yi Song, Yi-nan Li, Li-jun Xiao, Zheng Xu, Li-xiang Rao, Tie-jun Ci, and Xu-Long Hui.
Generative inverse design of metamaterials with customized stress-strain response.International Journal of
Mechanical Sciences, page 110875, 2025.
20

[63] Yuhao Bao, Zhiyuan Jia, Qiming Tian, Yangjun Luo, Xiaopeng Zhang, and Zhan Kang. Phononic crystal-based
acoustic demultiplexer design via bandgap-passband topology optimization.Composite Structures, 351:118622,
2025.
[64] Sushan Nakarmi, Jeffery A Leiding, Kwan-Soo Lee, and Nitin P Daphalapurkar. Predicting non-linear stress–
strain response of mesostructured cellular materials using supervised autoencoder.Computer Methods in Applied
Mechanics and Engineering, 432:117372, 2024.
[65] Hong-yun Yang, Xiao-sun Wang, Lu Liu, and Shi-jing Wu. Deep-learning-based low-frequency bandgap prediction
and elastic wave propagation properties of two-dimensional locally resonant metamaterials.European Journal of
Mechanics-A/Solids, page 105823, 2025.
[66] Julia Ling, Maxwell Hutchinson, Erin Antono, Sean Paradiso, and Bryce Meredig. High-dimensional materials and
process optimization using data-driven experimental design with well-calibrated uncertainty estimates.Integrating
Materials and Manufacturing Innovation, 6(3):207–217, 2017.
[67] W Keith Hastings. Monte carlo sampling methods using markov chains and their applications. 1970.
[68] Raisa Abedin Disha and Sajjad Waheed. Performance analysis of machine learning models for intrusion detection
system using gini impurity-based weighted random forest (giwrf) feature selection technique.Cybersecurity,
5(1):1, 2022.
[69] Xin Fang, Walter Lacarbonara, and Li Cheng. Advances in nonlinear acoustic/elastic metamaterials and metas-
tructures.Nonlinear Dynamics, pages 1–28, 2024.
[70] Fahimeh Akbari-Farahani and Salman Ebrahimi-Nejad. From defect mode to topological metamaterials: A
state-of-the-art review of phononic crystals & acoustic metamaterials for energy harvesting.Sensors and Actuators
A: Physical, 365:114871, 2024.
[71] Arash Kazemi, Kshiteej J Deshmukh, Fei Chen, Yunya Liu, Bolei Deng, Henry Chien Fu, and Pai Wang.
Drawing dispersion curves: band structure customization via nonlocal phononic crystals.Physical Review Letters,
131(17):176101, 2023.
[72] Alexander C Ogren, Berthy T Feng, Katherine L Bouman, and Chiara Daraio. Gaussian process regression as
a surrogate model for the computation of dispersion relations.Computer Methods in Applied Mechanics and
Engineering, 420:116661, 2024.
[73] Ting-Wei Liu, Chun-Tat Chan, and Rih-Teng Wu. Deep-learning-based acoustic metamaterial design for attenuat-
ing structure-borne noise in auditory frequency bands.Materials, 16(5):1879, 2023.
[74] Lige Chang, Xiaowen Li, Zengrong Guo, Yajun Cao, Yuyang Lu, Rinaldo Garziera, and Hanqing Jiang.
On-demand tunable metamaterials design for noise attenuation with machine learning.Materials & Design,
238:112685, 2024.
[75] Rachel Sun, Jet Lem, Yun Kai, Washington DeLima, and Carlos M Portela. Tailored ultrasound propagation in
microscale metamaterials via inertia design.Science Advances, 10(45):eadq6425, 2024.
[76] Qiangbo Wu, Quhao Li, Renjing Gao, and Shutian Liu. Topology optimization of phononic crystals for integrated
design with specific mechanical and band gap performance.Structural and Multidisciplinary Optimization,
68(9):1–20, 2025.
[77] Gilles Louppe. Understanding random forests.arXiv preprint arXiv:1407.7502, 2014.
21