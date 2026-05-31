# Credit-assigned Policy Gradient for Early Stage Retrieval in Two-stage Ranking

**Authors**: Haruka Kiyohara, Mihaela Curmei, Ariel Evnine, Shankar Kalyanaraman, Israel Nir, Ana-Roxana Pop, Nitzan Razin, Sarah Dean, Thorsten Joachims, Udi Weinsberg

**Published**: 2026-05-25 23:17:37

**PDF URL**: [https://arxiv.org/pdf/2605.26385v1](https://arxiv.org/pdf/2605.26385v1)

## Abstract
Large-scale search, recommendation, and retrieval-augmented generation (RAG) systems typically employ a two-stage architecture: an early-stage ranker (ESR) generates a candidate set, which is subsequently re-ranked by a late-stage ranker (LSR). While there are many reinforcement learning (RL) methods for training the LSR, end-to-end training of the ESR has proven challenging. In particular, naive application of "vanilla" policy gradient (V-PG) is not scalable for candidate-set sizes relevant for practical use due to exploding variance. This issue arises because V-PG propagates the gradient to the joint probability of the candidate sets, ignoring the contribution of each specific item in the candidate set to the reward. To mitigate this issue, we propose a novel "credit-assigned" policy gradient (CA-PG), which computes gradients with respect to the probability that the target item is chosen in any candidate set, i.e. marginalizing over all candidate sets that contain it. Our theoretical analysis reveals that CA-PG significantly reduces the variance of V-PG by marginalizing over the specific composition of the candidate set, while preserving the ability to learn the correct ranking of items under a reasonably aligned LSR policy. Experiments on both synthetic and real-world data demonstrate that CA-PG improves the convergence speed and training stability for ESRs utilizing the canonical Plackett-Luce model, especially when the candidate-set size is large.

## Full Text


<!-- PDF content starts -->

Credit-assigned Policy Gradient for
Early Stage Retrieval in Two-stage Ranking
Haruka Kiyohara1Mihaela Curmei2Ariel Evnine2Shankar Kalyanaraman2Israel Nir2Ana-Roxana Pop2
Nitzan Razin2Sarah Dean1Thorsten Joachims1Udi Weinsberg2
Abstract
Large-scale search, recommendation, and
retrieval-augmented generation (RAG) systems
typically employ a two-stage architecture: an
early-stage ranker (ESR) generates a candi-
date set, which is subsequently re-ranked by
a late-stage ranker (LSR). While there are
many reinforcement learning (RL) methods for
training the LSR, end-to-end training of the ESR
has proven challenging. In particular, naive
application of “vanilla” policy gradient (V-PG)
is not scalable for candidate-set sizes relevant
for practical use due to exploding variance. This
issue arises because V-PG propagates the gradient
to the joint probability of the candidate sets,
ignoring the contribution of each specific item
in the candidate set to the reward. To mitigate
this issue, we propose a novel“credit-assigned”
policy gradient (CA-PG), which computes
gradients with respect to the probability that the
target item is chosen in any candidate set, i.e.
marginalizing over all candidate sets that contain
it. Our theoretical analysis reveals that CA-PG
significantly reduces the variance of V-PG by
marginalizing over the specific composition of
the candidate set, while preserving the ability
to learn the correct ranking of items under a
reasonably aligned LSR policy. Experiments on
both synthetic and real-world data demonstrate
that CA-PG improves the convergence speed
and training stability for ESRs utilizing the
canonical Plackett-Luce model, especially when
the candidate-set size is large.
1Computer Science Department, Cornell University, Ithaca,
NY , USA2Central Applied Science, Meta, Menlo Park, CA, USA.
Correspondence to: Haruka Kiyohara<hk844@cornell.edu>.
Proceedings of the 43rdInternational Conference on Machine
Learning, Seoul, South Korea. PMLR 306, 2026. Copyright 2026
by the author(s).
Figure 1.Illustration of the two-stage ranking problem in a
large-scale recommender system.
1. Introduction
Reinforcement learning (RL) via policy gradient meth-
ods (Sutton et al., 1999) has proven to be a powerful tool
for finding decision-making policies that maximize an ob-
jective of interest, a.k.a. reward. Successful applications
of policy gradient include a wide range of problems, such
as the training of large language models (LLMs) (Rafailov
et al., 2023; Shao et al., 2024), retrieval-augmented gen-
eration (RAG) (Huang et al., 2020), information retrieval
(IR) (Gao et al., 2023), and content recommendation and
ranking (Chen et al., 2019; Oosterhuis, 2021). These appli-
cations require us to work with larger action (item) spaces
and more complex models as technology evolves.
In particular, modern recommender systems have to process
a massive volume of items at web latency. By necessity,
these systems handle this scale through two main stages:
an early-stage ranker (ESR), which filters up to billions
of items cheaply into a smaller candidate set, and a more
expensive late-stage ranker (LSR) that generates the final set
of recommendations from this candidate set, as illustrated in
Figure 1. An effective ESR is of crucial importance, since
the LSR policy can only be successful if the candidate set
contains the relevant items (Evnine et al., 2024).
However, while we have many methods for training the
LSR policy, we lack principled approaches for end-to-end
training of the ESR policy to make the overall retrieval
pipeline maximally effective. Moreover, the LSR in existing
pipelines often combines the results from multiple ESR
1arXiv:2605.26385v1  [cs.IR]  25 May 2026

Credit-assigned Policy Gradient for Early Stage Retrieval in Two-stage Ranking
models to make sure a diverse set of items is available to the
LSR policy (e.g., mixture-of-experts; MoE). This makes the
design of effective training methods for the ESR particularly
challenging (Hron et al., 2021).
This paper proposes a principled and efficient PG ap-
proach for training ESR policies, including those consist-
ing of mixtures-of-experts (MoE).We begin the analysis
by investigating the naive approach for the ESR training,
called “vanilla” (V-PG) (Ma et al., 2020). V-PG derives the
exact PG of ESR w.r.t. the probability of choosing the candi-
date set sampled by the ESR policy. We observed that V-PG
has a significant learning instability (i.e., gradient overflow)
issue, especially when we increase the candidate-set size
(K) and the number of actions ( |A|). This is because the
varianceof V-PG increases exponentially in O(|A|K)as
the action space of V-PG, is combinatorial. Because we
select only one candidate set AKfor each query in the user
interaction, this large action space makes it difficult to accu-
rately estimate the gradient. Moreover, we also identified
acredit-assignmentissue in the V-PG, which is closely re-
lated to the variance problem. The credit-assignment issue
happens because V-PG aims to increase the joint probability
ofAK, where all actions in the candidate set are equally val-
ued, thus ignoring which specific action contributed to the
final reward. For example, consider the case where there are
three actions, A, B, C , selected in the candidate set, and the
actions AandBare presented in the final ranking selected
by the LSR. Suppose that we get a positive reward for A
and a negative reward for Bas their position-wise reward.
Intuitively, we expect that the probability of selecting Ain
the candidate set should be increased, while the probabil-
ity of selecting Bshould be decreased by the ESR policy.
However, V-PG increases thejointprobability of selecting
all members of (A, B, C) when Areceives a high reward.
Thiscredit-assignmentissue has been overlooked.
Our theoretical analysis reveals that this inefficient propa-
gation of V-PG to irrelevant actions is the key source of the
variance (Theorem 3.1), leading to learning instability and
slow convergence. We overcome this issue by introducing a
variance-reduced alternative called“credit-assigned”pol-
icy gradient (CA-PG). The key point of CA-PG is that the
gradient is calculated w.r.t. the probability that action ais
selected inanycandidate set, i.e. the marginal probability
ofSK(a) :={AK|a∈ AK}, instead of the probability
of selecting aspecificcandidate set AK. CA-PG is not an
unbiasedestimation of the true ESR’s policy gradient due to
marginalization. However, we show that CA-PG can learn
to select the optimal candidate, as long as the LSR policy is
“reasonably aligned”with the true rank of actions, includ-
ing arbitrary interpolation between the optimal and uniform
random policies (Theorem 3.3). These results indicate that
ignoring distinctions between specific candidate sets can
have more benefits than harms due to substantial variancereduction under the use of a practical LSR policy, achieving
a better bias-variance tradeoff when the sample size is small.
We turn from statistical to computational efficiency. Specifi-
cally, we discuss the computation of the CA-PG’s marginal
probability (of SK(a)) for the commonly-used Plackett-
Luce (PL) ESR policy (Oosterhuis, 2021), with and without
the “sampling w/ replacement” (SwR) approximation. Our
analysis reveals that CA-PG-SwR reduces the computational
cost of CA-PG from O(KL) toO(L) in the special case
of using a single logit model (i.e., not mixture-of-experts
(MoE) candidate retrievers), where Lis the length of LSR
output. This “TOP1-PG” computes the simple probability
of the target action selected as the top-1 action by ESR, pro-
viding a computationally efficient alternative to CA-PG. Our
finding that TOP1-PG can be sufficient is a valuable insight,
as a single logit model-based PL is still widely adopted in
practical applications (Chen et al., 2019; Ma et al., 2020).
However, the combination of MoE and CA-PGs can yield a
further expected reward improvement compared to TOP1-
PG, due to more efficient exploration. This suggests an
interesting tradeoff between computational efficiency and
performance in the design of the ESR policy.
Finally, we compared CA-PG and TOP1-PG to V-PG and V-
PG-SwR on both synthetic and real data, using the KuaiRec
dataset (Gao et al., 2022). The results demonstrate that CA-
PG consistently improves the training stability over V-PG,
by avoiding the gradient overflow of the policy gradient.
Moreover, we also observe that both CA-PG and CA-PG-
SwR converge faster than V-PG-SwR, especially when the
size of candidates ( K) is large, requiring minimal interaction
to identify user interests.
Our contributions are summarized as follows.
•We discuss the tractable end-to-end training of ESR to
maximize users’ response signal.
•We find that “vanilla” policy gradient has a high-
variance issue, associated with the credit-assignment
problem.
•We propose a “credit-assigned” policy gradient to re-
duce variance, while keeping the ability to correctly
align items under a reasonably aligned late-stage pol-
icy.
•We provide a theoretical analysis formally characteriz-
ing the aforementioned variance and alignment prop-
erties, alongside a comparison of computational com-
plexity.
•We empirically demonstrate that CA-PG improves the
learning stability and convergence of V-PG when the
candidate-set size is large.
2

Credit-assigned Policy Gradient for Early Stage Retrieval in Two-stage Ranking
2. Policy Gradients for ESR Training
We begin by formulating two-stage ranking as a contex-
tual bandit problem. Let x∈ X be a user context in
the context space X, such as a user’s demographic pro-
file. Let a∈ A be an item (e.g., a document, prod-
uct, or piece of content) where Ais the original (large)
pool of items. In applications like search or recommen-
dation, we aim to present a ranked list of items (with-
out duplicates) a1:L:= (a 1, a2,···, a l,···, a L)to the
users, and receive item-wise user feedback (i.e., reward)
r1:L:= (r 1, r2,···, r l,···, r L). The aggregated (ranking-
wise) reward is expressed as r∗=PL
l=1αlrl, where
αl(>0) is a pre-specified non-negative weight for each
position (Kiyohara et al., 2022). For example, a simple sum
of rewards is expressed as ∀l∈[L], α l= 1, and a rep-
resentative information retrieval metric called Discounted
Cumulative Gain (DCG) is expressed as ∀l∈[L], α l=
1/log2(l+ 1) (J¨arvelin & Kek ¨al¨ainen, 2002). We can also
implicitly model the learning-to-rank (LTR) setting where
αcan be seen as theposition bias, representing the position-
dependent item observation probability (Joachims et al.,
2017). Given a policy π, which determines which item rank-
ing should be presented to users, we aim at maximizing the
following policy value as our objective function:
V(π) :=E p(x)π(a 1:L|x)p(r 1:L|x,a1:L)[r∗],
where p(x) andp(r1:L|x, a 1:L)are the user and reward
distributions respectively, and π(a1:L|x)is the probability
of choosing the item ranking a1:Lgiven the user context
x. As large-scale recommender systems need to handle
a large number of items, these systems often employ a
two-stage policy combination. The (simple)early stage
ranking (ESR)policy first filters thecandidate set, and the
(complex)late-stage ranking (LSR)policy generates the
ranked list of items from the candidate set. These poli-
cies are modeled as the stochastic sampling process of
the candidate set AK∼π ESR(AK|x)and final ranking
a1:L∼π LSR(a1:L|x,AK), where AKis the (unordered)
set of Kdifferent items. This paper studies PG methods for
the ESR policy ( πESR) given an existing, fixed LSR ( πLSR),
to improve the overall quality of the joint policy (π).
Following existing work (Gao et al., 2023), we assume
that the reward at each position ldepends only on the
corresponding action, i.e., rl∼p(r l|x, a l)and denote
q(x, a l) =E[r l|x, a l]. This“independence”assumption
means that there is no item-to-item interaction, such as di-
versity effects, in the users’ reward signal. This allows us to
decompose the expected ranking-wise reward into position-
wise expected rewards as E[r∗|x, a 1:L] =PL
l=1αlq(x, a l).
Therefore, the policy value is also decomposed into V(π) =PL
l=1αlvl(π), where
vl(π) :=Ep(x)π(l)(al|x)[q(x, a l)](1)andπ(l)(al|x) =P
a′
1:LI{al=a′
l}π(a′
1:L|x)is the
marginal probability of observing the action alat the posi-
tionl. Because ∇V(π) =PL
l=1αl∇vl(π), in the rest of
the paper, we discuss how to estimate the PG ofv l(π).
2.1. (Vanilla) Policy Gradient for Two-stage Policy
From the (general) PG theorem (Williams, 1992), the PG of
the joint policy for Eq. (1) is given as follows:
∇vl(π) =E D[∇logπ(a 1:L|x)rl]
=Ep(x)π(l)(al|x)p(r l|x,al)[∇logπ(l)(al|x)rl](2)
where D:=p(x)π(a 1:L|x)p(r 1:L|x, a 1:L)is the original
data generation process. The second equality follows from
the independence assumption rl⊥ ⊥a1:L(̸=l) |x, a l. We aim
to calculate the gradient of ESR given a static LSR. The
joint policy is decomposed into ESR and LSR as follows,
π(l)(al|x) =P
AKπESR(AK|x)π(l)
LSR(al|x,AK),(3)
where π(l)
LSR(al|x,AK)is the marginal probability of the
LSR policy choosing the action alat the position l, de-
fined asP
a′
1:LI{al=a′
l}πLSR(a′
1:L|x,AK). Combining
Eqs. (2)and(3)for a fixed LSR policy, the ESR’s PG be-
comes as follows:
∇vl(π) =E D′[∇logπ ESR(AK|x)·r l],(4)
where we denote the new data generation process as
(D′), which is the joint distribution of AKanda1:L, i.e.,
p(x)π ESR(AK|x)π LSR(a1:L|x,AK)p(r1:L|x, a 1:L).1
2.2. Variance and Credit-assignment Issues of V-PG
To train the ESR model via gradient ascent, the expectation
in the policy gradient (Eq. (4)) must be estimated from
empirical data. In online policy learning, we sample n
tuples per batch {(xi,AK
i, ai,l, ri,l)}n
i=1from the current
policy π, observing only a single candidate set AK
iandl’th
position item ai,lper context xi. Because the action space
AKgrows exponentially with K, unbiased estimators of
the PG suffer from extreme variance; thus, managing the
bias-variance tradeoff is critical for fast and stable learning.
The main limitation of the aforementioned “vanilla” policy
gradient (V-PG) (Eq. (4)) isvariancewhen using empirical
estimation from sampled data. Specifically, because the
action space of the candidate retrieval ( AK) becomes ex-
ponentially large as the size of the candidate ( K) becomes
large, more samples are required to estimate PG accurately.
Such a variance issue can result in learning instability or
slow convergence of the policy.
1We provide all derivations in this paper in Appendix B.1.
3

Credit-assigned Policy Gradient for Early Stage Retrieval in Two-stage Ranking
Figure 2.(Left) Credit-assignment issue of V-PG and (Right) Concept of the“credit-assigned”PG.
Moreover, taking a closer look at the source of variance re-
veals that the variance stems from thecredit-assignmentis-
sue of V-PG. As illustrated in Figure 2 (Left), V-PG (Eq. (4))
aims to increase thejoint probabilityof all the actions se-
lected in the candidate set ( AK), regardless of which action
(al) was selected by LSR and resulted in a high reward
value. This means that V-PG needs to take an expectation
over the combinatorial space of all candidate sets AKto
derive the gradient of each specific action al. As we need
to use Monte-Carlo (MC) sampling to estimate the expecta-
tion over the large action space of AK, V-PG exponentially
increases sample complexity asKgrows.
Instead, it might be more desirable to propagate the gradi-
ent only to the corresponding action in the candidate set,
as illustrated in Figure 2 (Right). In the next section, we
study how to enable such “credit-assigned” policy gradient
and discuss whether the credit attribution can resolve the
variance problem of V-PG.
3. Proposal: Credit-assigned Policy Gradient
The key ingredient for enabling the“credit-assigned”PG
is a new decomposition of the policy that does not depend
on specific samples of AK. Specifically, we consider the
following policy decomposition:
π(l)(al|x) :=π ESR(SK(al)|x)π(l)(al|x,SK(al))
where SK(al) :={AK|al∈ AK}is the set of candidate
setsAKthat contain alas one of the members. Here,
πESR(SK(al)|x) :=P
AKI{al∈ AK}πESR(AK|x)is
the (marginal) probability that the ESR model chooses the
itemalamong any candidate sets, while π(l)(al|x,SK(al))
is the probability that alis selected as the l’th item
in the final ranking, given that alis included in the
candidates ( SK(al)). More precisely, π(l)(al|x,SK(al))
depends on both ESR and LSR and is defined asP
AK∈SK(al)πESR(AK|x,SK(al))π(l)
LSR(al|x,AK),
where πESR(AK|x,SK(al))is the normalized probability
that each specific candidate setAKis selected.2
2πESR(AK|x,SK(al)) =π ESR(AK|x)/π ESR(SK(al)|x).The greatest benefit of the above policy decomposition is
that the variable SK(al)takes the marginal across candi-
date sets AKand thus no specific AKappears as a random
variable. This is beneficial for variance reduction, as it
eliminates the need to estimate EAK[·]from Monte-Carlo
(MC) samples as done in V-PG. Then, deriving the gradi-
ent of ESR w.r.t. πESR(SK(al)|x), we have the following
“credit-assigned” policy gradient(CA-PG):
∇vl(π) =E D[∇logπ ESR(SK(al)|x)·r l].(5)
The key trick is that CA-PG ignores the dependence of ESR
on the second term (i.e., π(l)(al|x,SK(al))), which helps
avoid the dependency on each specific AKin the gradient
estimation. Then, we can increase the ESR’s (marginal)
probability only for the action corresponding to the reward
observation, thereby resolving the credit-assignment issue
as illustrated in Figure 2 (Right). Moreover,because the
action space is now reduced from |A|Kto|A|, we expect
variance reduction and improved sample complexity.
3.1. Theoretical Analysis – Connection to V-PG
To gain insight into how the proposed CA-PG works, we
theoretically examine two important research questions: (1)
What is the connection to V-PG and (2) when is the CA-PG
sufficient for learning an effective ESR model? Below, we
first show that CA-PG is a component of V-PG, designed
for variance reduction. Note that all the proofs in this paper
are provided in Appendix C.
Theorem 3.1.(Relationship between two policy gradients)
CA-PG is a component of V-PG, i.e.,
(V-PG)=(CA-PG)+∇C(π).
The residual term (∇C(π)) is given as follows.
Eπ(l)
LSR(al|x,AK)
π(l)(al|x,SK(al))| {z }
(♠)∇logπ ESR(AK|x,SK(al))·r l
,
where the expectation is E[·] :=E D[EπESR(AK|x,SK(al))[·]].
Note that (♠)can be seen as the normalized importance of
4

Credit-assigned Policy Gradient for Early Stage Retrieval in Two-stage Ranking
each specific candidate set AK, determined by how likely
alis to be chosen by the late stage policy π(l)
LSR(al|x,AK)
given the candidate setAK.
Theorem 3.1 indicates that V-PG has an additional gradi-
ent term ( ∇C(π) ) that propagates the gradient to each spe-
cific candidate set AK, depending on how likely LSR is
to choose the target action given the corresponding candi-
date set AK. In other words, CA-PG equally propagates
the gradient to each candidate AK, ignoring from which
specific AKthe action alcomes from. We intentionally use
this trick to avoid dependency on AKvia marginalization,
allowing CA-PG to reduce the variance greatly compared to
V-PG by working on a smaller (marginalized) action space.
Interestingly, it also resolves the credit-assignment issue
without taking an expectation over all candidate sets. This
advantage of CA-PG becomes more pronounced when the
candidate-set size (|A|K) becomes large.
Next, to discuss RQ. (2): “When does CA-PG learn an
effective ESR policy?”, we first introduce the following
notions of “optimal alignment” and “optimal partition”.
Definition 3.2.(Optimal alignment and partition) Let
a∗
1, a∗
2, . . . , a∗
|A|be the ground-truth ranking of actions for
contextx(i.e.,q(x, a∗
1)> q(x, a∗
2)>···> q(x, a∗
|A|)).
•A policy isoptimally alignedwhen the policy pre-
serves the ranking of items in the probability space,
asπ(a∗
1|x)> π(a∗
2|x)>···> π(a∗
|A||x).
•A policy isoptimally partitioned for top- Kwhen
π(a∗
k|x)> π(a∗
j|x)holds for any (k, j) , that satisfies
1≤k≤KandK+ 1≤j≤ |A|.
Note that the optimal partition is a weaker condition than
the optimal alignment, as it allows misalignment within the
top-Kand within the tail items (i.e., those ranked K+ 1
or lower). Then, the following theorem suggests that CA-
PG learns the optimal partition of items for many practical
choices of the LSR policy, as long as they have a reasonable
level of alignment ability, as explained below.
Theorem 3.3.(Sufficient condition for CA-PG) CA-PG
learns an ESR policy that is optimally partitioned for top- K,
if the following condition is satisfied for any (k, j),1≤k≤
K, K+ 1≤j≤ |A|.
EπESR(AK|x,SK(a∗
j))[π(l)
LSR(a∗
j|x,AK)]
EπESR(AK|x,SK(a∗
k))[π(l)
LSR(a∗
k|x,AK)]<q(x, a∗
k)
q(x, a∗
j)
Note that, when optimizing a policy with ∇V(π) =PL
l=1αl∇vl(π), the LHS of the inequality becomes the
policy ratio w.r.t.PL
l=1αlπ(l)(a|x,SK(a)).
The proof outline of Theorem 3.3 is as follows: From
the general PG theorem (Sutton et al., 1999), CA-PGupdates the policy in a way to align πESR(SK(a)|x)
with the rank of the corresponding expected reward,
q(l)(x, SK(a)). This expected reward, q(l)(x, SK(a)),
is defined as the LSR-propensity discounted reward,
i.e.,EπESR(AK|x,SK(a))[π(l)
LSR(a|x,AK)]·q(x, a) from
Lemma C.1 in Appendix C. Therefore, to evaluate
the (ground-truth) top- Kitems higher than other items
(i.e.,∀(k, j), π ESR(SK(a∗
k)|x)> π ESR(SK(a∗
j)|x)), the
expected reward that CA-PG optimizes should satisfy
q(l)(x, SK(ak))> q(l)(x, SK(aj)).
Assuming positivity of the reward, ∀(x, a), q(x, a)>0 ,
Theorem 3.3 indicates thata reasonably aligned LSR, e.g.,
the one defined by the optimal partition in Definition 3.2,
is sufficient for learning the optimal ESR policy. Exam-
ples of such LSRs include arbitrary interpolation between
the uniform random and optimal policy (e.g., epsilon-greedy
or softmax). This is because the LHS of the inequality be-
comes 1 when using the uniform random policy as LSR,
and becomes less than 1 when using a policy that aligns the
items optimally. In contrast, the RHS of the inequality is
always greater than 1. Moreover, Theorem 3.3 suggests that
CA-PG learns an optimally partitioned ESR policy even in
the presence of certain LSR misalignments, specifically: (1)
misalignment within (ground-truth) top- Kitems and within
tail items (i.e., those ranked K+ 1 or lower), and (2) cross-
top-Kmisalignment, provided the LSR alignment error is
bounded by the reward ratio. The condition in the theorem
may hold even when the LSR has inconsistent alignment of
items among candidate sets, e.g., due to item-diversity or
item-fairness constraints (Wang & Joachims, 2021). These
theoretical results indicate that CA-PG does not introduce
much bias in the ESR alignment when using a practical LSR
policy, while greatly reducing the variance.
3.2. Computation of the Score Function
Having established the general form of CA-PG, we
now address the computation of the marginal probability
πESR(SK(al)|x), which is required to derive the score func-
tion in Eq. (5). This section focuses on the calculation of
the score function of the Plackett-Luce (PL) policy family,
as this is one of the most prevalent approaches for sampling
candidate sets from a large pool of items (Chen et al., 2019).
The PL policy samples ranked items without replacement by
recursively applying the softmax function to the remaining
items (which is often implemented with top- Kon Gumbel-
trick (Kool et al., 2019)). We consider the generalized family
of PL using multiple sub-retrievers (i.e., mixture-of-experts
(MoE) models), taking into account that having multiple can-
didate retrievers is often considered useful in practice (Hron
5

Credit-assigned Policy Gradient for Early Stage Retrieval in Two-stage Ranking
Table 1.Comparison of policy gradient methods for early stage retrieval
PG methodsaction gradient
space size MC sample alignment condition variance computation
V-PG, V-PG-SwR AK≈ |A|K(AK,a1:L) none (always accurate) high O(K)
CA-PG (ours) SK|A| a1:L + w/ reasonably-aligned LSR low O(KL)
CA-PG-SwR, TOP1-PG (S1)|A| a1:L + w/ Plackett-Luce ESR,+ w/o MoE low O(ML),O(L)
Theredfont represents the hardest requirement, and the green font shows the easiest one. The alignment of the items becomes accurate
when the additional condition (+ w/, w/o) is satisfied for the proposed method. MC is the abbreviation of “Monte-Carlo”.
et al., 2021; Evnine et al., 2024):
∀k,1≤k≤K,ˆa k∼exp(ˆqm(k)
ESR(x, a k)/τ)
P
a∈A\A(k−1) exp(ˆqm(k)
ESR(x, a)/τ),
where m(k) is the model index for the k’th member. For
example, when K= 10 and using M= 5 experts of the
MoE models, we can define m(k) as follows: members 1
and 2 ( k= 1,2 ) use model 1 ( m= 1 ), and members 3 and
4 (k= 3,4 ) use model 2 ( m= 2 ), and so on. ˆqESR(x, a)
is the ESR model’s logit value of each item ato the given
user context x, and τ >0 is the temperature parameter.
A(k−1):= (ˆa 1,ˆa2,···,ˆa (k−1) )is the previously sampled
candidate set, where A(0)=∅. Then, the marginal probabil-
ityπESR(SK(al)|x) can be derived via a method detailed
in Appendix D.1, in a fully differential manner without re-
quiring Monte-Carlo (MC) sampling. However, a drawback
is that this process requires O(KL) of computation as the
probability must be calculated for every landk. This is
worse than the computation of V-PG ( O(K) ), which does
not depend on the final ranking length,L.
To mitigate the computational burden of CA-PG, we con-
sider the sampling w/ replacement (SwR) approximation of
PL, as done in (Chen et al., 2019; Ma et al., 2020). Using
the SwR approximation, PL can be seen as applying inde-
pendent softmax to all the actions multiple times, resulting
in the followingCA-PG-SwR:
∇vl(π) =E D
∇logPK
k=1πESR(S1(al)|x;m(k))
·rl
.
We provide the derivation steps in Appendix B.2. CA-
PG-SwR reduces the computational time significantly in
the MoE case due to dependency on multiple logit models
(fromO(KL) toO(ML) ). Moreover, as the special case
of CA-PG-SwR with a single logit-based PL model, we can
specifically define the following “TOP1-PG”:
∇vl(π) =E D[∇logπ ESR(S1(al)|x)·rl].
TOP1-PG dramatically reduces the computational order
fromO(KL) toO(L) , a complexity substantially lower
than the O(K) cost of V-PG. TOP1-PG enables a fast and
easy gradient computation on the probability that the target
itemais selected as the top-1 item in the candidate. Thismakes TOP1-PG a practical alternative to (fully-computed)
CA-PG, especially in traditional applications using a single
logit model. In contrast, it is also known that the MoE-
based ESR model improves the exploration efficiency due
to diversity in the candidate (Hron et al., 2021), and there
is a tradeoff of computational efficiency and performance
between TOP1-PG (single) and CA-PGs (MoE). Table 2
summarizes the key characteristics of each PG method.
4. Experiments
This section empirically compares each PG method. The
experiment code is available at a GitHub repository:
https://github.com/facebookresearch/early stage retrieval.
4.1. Synthetic Data
We simulate 1000 unique users and 1000 unique items in
the synthetic experiment, where users ( x) and items ( a) are
associated with dxanddadimensional embeddings (x∗, a∗)
sampled from a uniform distribution U[−1,1](dx,da). Using
these embeddings, we model the expected reward as
q(x, a) = SoftPlus(⟨M aa∗, Mxx∗⟩) +c,
where SoftPlus(z) := log(1 + exp(z)) is a function that
ensures positive output, and c= 1.0 is a positive constant
(to guarantee the reward positivity required by Theorem 3.3).
MaandMxare some (dh, da)and(dh, dx)dimensional
projection matrices, where we let dx=da=dh= 10 .
⟨·,·⟩ is the inner-product between two embeddings. Given
the above expected reward q(x, a) , the reward is sampled
from a normal distribution N(q(x, a), σ2(x, a)) , where σ
is the (item-dependent) noise level. Finally, we use the (or-
acle) Plackett-Luce policy defined as π(l)
LSR(al|x,AK) :=
exp(q(x, a l)/τLSR)/(P
a′∈AK\a1:l−1exp(q(x, a′)/τLSR))
as the (given) LSR policy.
Compared PG methodsTo enable a fair comparison
among different PG methods, we use the same Plackett-Luce
(PL) policy (Oosterhuis, 2021) with (mixture-of-experts;
MoE) two-tower model architecture (Guo et al., 2021).
Specifically, a model corresponding to the k’th element
(m(k) ) generates 10-dimensional user and item embeddings
ˆµm(k)(x)andˆνm(k)(a)and calculates a logit value of action
6

Credit-assigned Policy Gradient for Early Stage Retrieval in Two-stage Ranking
agiven context xby the inner product between two model
embeddings: ˆqm(k)(x, a) =⟨ˆµ m(k)(x),ˆν m(k)(a)⟩, where
⟨,⟩represents the inner product.
Given this model, we compare four PG methods for the
ESR training;vanilla-PG(V-PG, baseline),V-PG-SwR
(sampling w/ replacement, baseline),credit assigned-PG
(CA-PG, ours), andCA-PG-SwR(TOP1-PG when M= 1 ,
ours). We use a stochastic policy during the policy train-
ing phase, and evaluate the performance of deterministic
(greedy) ESR for all compared methods to see the difference
in the item alignment ability of the learned ESR model. We
use the same procedure to set the learning rate for V-PGs
and CA-PGs. The exact implementation of the policies and
additional details are provided in Appendix B.2, D.1, and E.
Experiment configsTo check if Theorems 3.1 and 3.3
hold empirically, we compare performance of each PG
method with the following configurations with the bold font
representing the default value:(1) # of candidate actions
(K)-{5,10, 20},(2) # of output actions ( L)-{1, 5},(3)
# of MoE models ( M){1, 5}, (4)alignment (optimality)
of LSR- {optimal , noisy optimal, uniform, anti-optimal
}. Config (3) specifies whether we use a single logit model
ˆqESR for the PL policy or use a different model for each
position of the top- K(i.e., MoE). Finally, Config (4) tests
the performance of CA-PGs when Theorem 3.3 (sufficient
condition about LSR) does not hold. We run experiments
with 10 random seeds. Additionally, computational cost
analysis is provided in Appendix E.
ResultsWe first investigate how the stability and conver-
gence speed of each policy gradient method change with
varying size of candidates ( K) in Figure 3. The former,
stability, is studied by checking whether the policy gradi-
ent caused gradient overflow during the training process.
The results demonstrate that, without using asampling with
replacement(SwR) approximation, the policy gradient be-
comes unstable regardless of the choice of PG method. In
particular, V-PG behaves especially catastrophically when
the candidate size is large (e.g., K= 20 ). In contrast,
CA-PG increases the training stability greatly, making the
training process more reliable. Moreover, Figure 3 also
suggests that, even if we use the SwR approximation, V-
PG-SwR slows down the convergence speed as we increase
the size of candidates ( K). Thus, CA-PGs perform much
better than V-PG-SwR when we compare the performance
with a small number of online interactions (@50K), and
achieve 95% of the final policy value of V-PG-SwR about
3xfaster than V-PG-SwR when K= 20 . These differences
are significant in a practical situation, as a poor performance
of the policy in the initial training phase can harm both
user satisfaction and business metrics. In contrast, because
V-PG is an unbiased estimation of the true ESR’s gradient,V-PG-SwR performs better than CA-PGs at convergence
(@500K), while CA-PGs also perform comparably.
Next, we also compare the performance (@50K and
@500K) with varying alignment (optimality) of LSR in
Table 2. When using the optimal, noisy-optimal, and uni-
form LSR, the observations are quite similar to the case
with varying candidate sizes – V-PG-SwR often performs
better than CA-PG at the convergence performance due
to unbiasedness of the policy gradient, while CA-PG and
TOP1-PG achieve better performance @50K due to faster
convergence, empirically showing the tradeoff between bias
and variance as Theorem 3.1 indicates. Moreover, we also
find that CA-PG and CA-PG-SwR fail only in the case with
anti-optimal LSR, empirically verifying the validity of The-
orem 3.3, stating that CA-PG can align items correctly as
long as using a reasonably aligned LSR policy.
Finally, we further compare the performance of each PG
method when using five MoE models as the base logit mod-
els, instead of using a single logit model, which is often
useful in practical situations (Hron et al., 2021; Evnine et al.,
2024). Specifically, Table 3 reports how the convergence
performance changes with a single and MoE ( M= 5 ) mod-
els and varying size of LSR output (i.e., ranking length, L).
The results reveal that, while V-PG is the best when using a
single logit model, CA-PG and CA-PG-SwR perform better
than V-PG-SwR when using MoE models. These results
show that CA-PGs can exploit more benefits of MoE models
than V-PG, allowing more flexible modeling of ESR.
4.2. Real Data
Next, we compare the approaches on real data from the
KuaiRec dataset (Gao et al., 2022). This dataset provides
a fully observable (dense) user-item interaction matrix of
q(x, a) using the empirical ”watch ratio” (i.e., watch time
divided by the video duration) in a video streaming platform.
The data consists of 1411 users and 3326 items, and we
learned their embeddings with dh= 10 . To simulate a
practical situation, we compare PG methods with larger
candidate-set sizes, i.e., K∈ {50,100,200} . To make
policy learning computationally tractable in this setting, we
need to use SwR approximation and compare only V-PG-
SwR and CA-PG-SwR. All other settings remain the same
as the defaults used in the synthetic experiments.
Table 4 summarizes the performance of each PG method
with varying # of candidates ( K). The real-data experi-
ment is a more challenging setting due to large candidate
sizes and complex (and unknown) reward functions. We
observe that CA-PG-SwR (i.e., TOP1-PG) improves the con-
vergence speed over V-PG-SwR and becomes increasingly
beneficial as Kgrows. This result demonstrates that the
“credit-assignment”-based variance reduction is particularly
useful in practical situations with a large set of candidates.
7

Credit-assigned Policy Gradient for Early Stage Retrieval in Two-stage Ranking
Figure 3.Comparing the training stability and convergence performance (policy value) of each PG with varying # of candidates
(K).Each line shows the learning curve for a single random seed (i.e., 10 random seed results can overlap). The vertical lines indicate
training termination; any line appearing before 500K gradient steps signifies an interruption due to gradient overflow. Two horizontal lines
show the 100% (top) and 95% (bottom) lines of the V-PG-SwR policy values, to compare the convergence speed of each method.
Table 2.Comparing PG methods with a single logit model
PG methods# of candidates (K) alignment (optimality) of LSR
5 10 20 opt. noisy uniform anti-opt.
V-PG 7.09(±0.22) 7.16(±0.27) 6.86(±0.34) 7.16(±0.27) 7.25(±0.31) 3.27 (±0.28) 1.07 (±0.02)
(baseline) →7.22(±0.32) →7.17(±0.27) →6.86(±0.34) →7.17(±0.27) →7.25(±0.32) →3.28(±0.29) →1.07(±0.03)
V-PG-SwR 7.68 (±0.03) 7.80(±0.03) 7.88(±0.04) 7.80(±0.03) 7.86(±0.03) 3.80(±0.05) 1.05(±0.01)
(baseline) →8.33 (±0.03) →8.54 (±0.02) →8.74 (±0.03) →8.54 (±0.02) →8.55 (±0.04) 6.39(±0.08) →1.08 (±0.01)
CA-PG 7.39(±0.03) 7.99 (±0.03) 8.39 (±0.02) 7.99 (±0.03) 8.14 (±0.04) 5.64(±0.11) 1.00(±0.01)
(ours) →8.19(±0.03) →8.44(±0.17) →8.70(±0.12) →8.44(±0.17) →8.42(±0.18) →6.73(±0.13) →1.00(±0.01)
CA-PG-SwR 7.37(±0.04) 7.94(±0.03) 8.33(±0.01) 7.94(±0.03) 8.11(±0.04) 5.79 (±0.09) 1.00(±0.01)
(ours, TOP1-PG) →8.11(±0.03) →8.40(±0.03) →8.62(±0.02) →8.40(±0.03) →8.41(±0.04) →6.90 (±0.14) →1.00(±0.01)
The top values report the policy value @50K gradient steps, while the bottom value reports those @500K gradient steps. If gradient
overflow occurs, we use the performance at the interrupted point instead of @50K or @500K. Theredfont represents the worst
performance, and thegreen font shows the best performance, for each of @50K and @500K performances.
Table 3.Comparing PG methods with MoE candidate retrievers
PG methods# of output (L)
1 5
# of MoE models (M)
1 5 1 5
V-PG 7.17 7.63 25.0 26.5
(baseline) (±0.27) (±0.54) (±2.22) (±3.95)
V-PG-SwR 8.54 8.58 38.1 37.2
(baseline) (±0.02) (±0.03) (±0.19) (±0.07)
CA-PG 8.44 8.88 36.2 37.2
(ours) (±0.17) (±0.02) (±0.64) (±1.00)
CA-PG-SwR 8.40 8.82 36.3 37.6
(TOP1-PG,M= 1) (±0.03) (±0.02) (±0.18) (±0.09)
Theredfont represents the worst performance, and the green font
shows the best performance. The policy value is either @500K
gradient steps or at the gradient overflow point.
Table 4.Comparing PG methods in the real-data experiment
PG methods# of candidates (K)
50 100 200
V-PG-SwR 5.05(±0.06) 5.39(±0.07) 5.92(±0.08)
(baseline) →7.91 (±0.05) →8.22 (±0.05) →8.26(±0.07)
CA-PG-SwR 5.61 (±0.05) 6.37 (±0.04) 7.10 (±0.06)
(TOP1-PG) →7.19(±0.07) →7.82(±0.03) →8.32 (±0.04)
Thegreen font shows the best performance. The policy value is
measured at either @50K (top) or @500K (bottom) gradient steps.5. Related Work
This section discusses the most closely related works on
policy learning for large-scale ranking.3
Policy gradient for rankingThe Plackett-Luce (PL) pol-
icy has been widely used in search (Oosterhuis, 2021), video
streaming (Chen et al., 2019), and document retrieval (Gao
et al., 2023) in a single-stage ranking setting. Given that
the action space can be large in this ranking setting, the
efficient calculation of PL has been a key challenge for
the (single-stage) PG approach. For example, (Oosterhuis,
2021) calculates PL for the partial reward of ranking, lever-
aging the cascading structure of the ranking decomposition.
(Ma et al., 2021) approximates the (marginal) probability
that an item set (which includes the actions selected in the
final ranking) is ranked higher than the remaining subset
of items. (Chen et al., 2019) calculates the probability of
the selected action being one of the top- Kactions in the
final ranking. Furthermore, (Sakhi et al., 2023) proposes a
PL-like stochastic policy that allows faster computation of
the gradient using Gaussian modeling. However, because
all of these works focus on thesingle-stageranking setting,
the credit assignment issue in thetwo-stagesetting has not
3An extended discussion can be found in Appendix A.
8

Credit-assigned Policy Gradient for Early Stage Retrieval in Two-stage Ranking
been previously addressed. An efficient way of propagating
the gradient to the ESR policy has remained non-trivial and
underexplored.
(Ma et al., 2020) is the most related work to ours in that it
studies the (vanilla) PG approach in the two-stage setting.
However, as discussed in the main text, V-PG has the issues
of high variance and credit assignment, limiting the scalabil-
ity of V-PG to large candidate sizes ( K). While (Ma et al.,
2020) discusses some heuristics to reduce the variance of
V-PG (i.e., adding a discount factor to the non-target action
in the candidate set), the mechanism of how and why this
heuristic works has not been demystified. We provide a rig-
orous proof of how a general idea of“credit-assigned”PG
can resolve the two issues of V-PG. Another related work
is (Kiyohara et al., 2025), which retrieves a diverse set of
candidates using an MoE-like PL policy via two-stage PG.
However, (Kiyohara et al., 2025) focuses on the case where
reward interaction among ranking (i.e., diversity) matters,
and credit assignment does not matter due to the depen-
dence of the reward on multiple items. These two works
also consider different learning challenges in the offline set-
ting. Our work is the first to highlight and mitigate the credit
assignment issue in two-stage PG.
Other variance reduction methods for RLIn another
line of research, variance reduction has long been a key
research question for policy gradient-based methods in
RL (Konda & Tsitsiklis, 1999). The prevalent approach
for variance reduction has been baseline correction, and the
representative method called PPO (Schulman et al., 2017)
calculates the“advantage”function, which normalizes the
reward given context and action ( r(x, a 1:L)) by a model-
predicted baseline estimating the expected reward given
context x, i.e., ˆv(x)≈E π(a1:L)[r(x, a 1:L)|x]. Similarly,
in the recent large-language model (LLM) applications,
GRPO (Shao et al., 2024) queries msamples per context
and action rj(x, a), j∈[m] , and normalizes the reward as
r′
j= (r j−mean(r))/std(r) . These methods reduce the
variance by normalizing the reward instead of manipulating
the calculation of the score function (i.e., log-action choice
probability), as CA-PG does. Therefore, the existing RL
variance reduction methods are complementary to CA-PG,
and we can combine the baseline correction-based variance
reduction with CA-PG. As a proof of concept, we addition-
ally report the results of combining both V-PG and CA-PG
with GRPO in Appendix E, showing the benefit of using
both CA-PG and GRPO together.
6. Conclusion and Future Work
This paper proposes a statistically and computationally effi-
cient approach for training the early-stage ranker (ESR) in
a two-stage decision policy. We examined the constructionof the “vanilla” policy gradient (V-PG) and identified that
variance and credit-assignment issues become prohibitive
when the candidate-set size ( K) is large. To tackle these
challenges, we developed a novel PG method called “credit-
assigned” PG (CA-PG) and demonstrated its benefits in both
theoretical and empirical ways.
We identify two interesting directions for future work. The
first is to explore theoff-policy learning(OPL) setting (Chen
et al., 2019; Ma et al., 2020), where we aim to optimize the
(two-stage) policy using logged data. While we expect CA-
PG’s benefits to hold in the offline setting, exploring an
efficient importance sampling strategy in our setting can be
a promising future direction. The second direction is to in-
vestigate how to incorporate reward interaction within rank-
ing. Our paper primarily focuses on the setting where the
position-wise reward is affected only by the corresponding
action at the same position. However, reward interactions
such as diversity within the final ranking (Oosterhuis, 2021)
may arise in applications like news recommendation. Ex-
ploring how to assign credit to each action under item-item
interactions can also be impactful.
Acknowledgments
This research was supported in part by NSF Awards IIS-
2312865, IIS-2442137, OAC-2311521, CCF-2312774, and
a gift to the LinkedIn-Cornell Bowers CIS Strategic Part-
nership. Sarah Dean is also partly supported by an AI2050
Early Career Fellowship program at Schmidt Sciences, and
Haruka Kiyohara is partly supported by the Funai Over-
seas Scholarship and Quad Fellowship. Most of the work
was done while Haruka Kiyohara was an intern at Central
Applied Science, Meta.
Additionally, we thank anonymous reviewers for valuable
feedback during the discussion phase.
Impact Statement
This paper investigates the optimization of early stage rank-
ing (ESR) policies in a two-stage ranking framework. Ap-
plications span search, recommender systems, and large lan-
guage models (especially retrieval-augmented generation;
RAG). We expect that the paper will have a broad impact
on such related domains that rely on large-scale retrieval.
As we use synthetic simulations and a publicly available
dataset (Gao et al., 2022) in our paper, we do not expect
that this paper will raise any ethical considerations. While
reinforcement learning (RL) approaches in general have
concerns regarding their learning instability and harm during
the training process (Levine et al., 2020), this work aims to
overcome such difficulties. We believe this effort makes the
RL approach more robust and tractable in practical settings.
9

Credit-assigned Policy Gradient for Early Stage Retrieval in Two-stage Ranking
References
Burges, C. J. From ranknet to lambdarank to lambdamart:
An overview.Learning, 11(23-581):81, 2010.
Carbonell, J. and Goldstein, J. The use of mmr, diversity-
based reranking for reordering documents and producing
summaries. InProceedings of the 21st annual interna-
tional ACM SIGIR conference on Research and develop-
ment in information retrieval, pp. 335–336, 1998.
Chakraborty, J. and Verma, V . A survey of diversifica-
tion techniques in recommendation systems. In2016
International Conference on Data Mining and Advanced
Computing (SAPIENCE), pp. 35–40. IEEE, 2016.
Chen, M., Beutel, A., Covington, P., Jain, S., Belletti, F.,
and Chi, E. H. Top-k off-policy correction for a rein-
force recommender system. InProceedings of the 12th
ACM International Conference on Web Search and Data
Mining, pp. 456–464, 2019.
Evnine, A., Ioannidis, S., Kalimeris, D., Kalyanaraman, S.,
Li, W., Nir, I., Sun, W., and Weinsberg, U. Achieving
a better tradeoff in multi-stage recommender systems
through personalization. InProceedings of the 30th ACM
SIGKDD Conference on Knowledge Discovery and Data
Mining, pp. 4939–4950, 2024.
Gao, C., Li, S., Lei, W., Chen, J., Li, B., Jiang, P., He,
X., Mao, J., and Chua, T.-S. Kuairec: A fully-observed
dataset and insights for evaluating recommender systems.
InProceedings of the 31st ACM International Conference
on Information & Knowledge Management, pp. 540–550,
2022.
Gao, G., Chang, J. D., Cardie, C., Brantley, K., and Joachim,
T. Policy-gradient training of language models for rank-
ing.arXiv preprint arXiv:2310.04407, 2023.
Gollapudi, S. and Sharma, A. An axiomatic approach for
result diversification. InProceedings of the 18th inter-
national conference on World wide web, pp. 381–390,
2009.
Guiver, J. and Snelson, E. Bayesian inference for plackett-
luce ranking models. InProceedings of the 26th Inter-
national Conference on Machine Learning, pp. 377–384,
2009.
Guo, W., Krauth, K., Jordan, M., and Garg, N. The stereo-
typing problem in collaboratively filtered recommender
systems. InProceedings of the 1st ACM Conference
on Equity and Access in Algorithms, Mechanisms, and
Optimization, pp. 1–10, 2021.
Hofmann, K., Whiteson, S., and De Rijke, M. Balancing
exploration and exploitation in learning to rank online.InProceedings of the 33rd European conference on Ad-
vances in Information Retrieval, pp. 251–263, 2011.
Hron, J., Krauth, K., Jordan, M., and Kilbertus, N. On com-
ponent interactions in two-stage recommender systems.
Advances in Neural Information Processing Systems, 34:
2744–2757, 2021.
Huang, C., Jiang, W., Wu, J., and Wang, G. Personalized
review recommendation based on users’ aspect sentiment.
ACM Transactions on Internet Technology, 20(4):1–26,
2020.
J¨arvelin, K. and Kek ¨al¨ainen, J. Cumulated gain-based evalu-
ation of ir techniques.ACM Transactions on Information
Systems (TOIS), 20(4):422–446, 2002.
Joachims, T., Swaminathan, A., and Schnabel, T. Unbiased
learning-to-rank with biased feedback. InProceedings of
the 10th ACM International Conference on Web Search
and Data Mining, pp. 781–789, 2017.
Kiyohara, H., Saito, Y ., Matsuhiro, T., Narita, Y ., Shimizu,
N., and Yamamoto, Y . Doubly robust off-policy eval-
uation for ranking policies under the cascade behav-
ior model. InProceedings of the 15th ACM Interna-
tional Conference on Web Search and Data Mining, pp.
487–497, 2022.
Kiyohara, H., Khanna, R., and Joachims, T. Off-policy
learning for diversity-aware candidate retrieval in two-
stage decisions. InICML 2025 Workshop on Scaling Up
Intervention Models, 2025.
Kleinberg, J. and Raghu, M. Team performance with test
scores.ACM Transactions on Economics and Computa-
tion (TEAC), 6(3-4):1–26, 2018.
Konda, V . and Tsitsiklis, J. Actor-critic algorithms.Ad-
vances in Neural Information Processing Systems, 12,
1999.
Kool, W., Van Hoof, H., and Welling, M. Stochastic beams
and where to find them: The gumbel-top-k trick for sam-
pling sequences without replacement. InProceedings of
the 36th International Conference on Machine Learning,
pp. 3499–3508, 2019.
Levine, S., Kumar, A., Tucker, G., and Fu, J. Offline rein-
forcement learning: Tutorial, review, and perspectives on
open problems.arXiv preprint arXiv:2005.01643, 2020.
Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V .,
Goyal, N., K ¨uttler, H., Lewis, M., Yih, W.-t., Rockt ¨aschel,
T., et al. Retrieval-augmented generation for knowledge-
intensive nlp tasks.Advances in Neural Information Pro-
cessing Systems, 33:9459–9474, 2020.
10

Credit-assigned Policy Gradient for Early Stage Retrieval in Two-stage Ranking
Li, L., Chu, W., Langford, J., and Wang, X. Unbiased
offline evaluation of contextual-bandit-based news article
recommendation algorithms. InProceedings of the fourth
ACM international conference on Web search and data
mining, pp. 297–306, 2011.
Liu, T.-Y . et al. Learning to rank for information retrieval.
Foundations and Trends® in Information Retrieval, 3(3):
225–331, 2009.
Ma, J., Zhao, Z., Yi, X., Yang, J., Chen, M., Tang, J., Hong,
L., and Chi, E. H. Off-policy learning in two-stage recom-
mender systems. InProceedings of The Web Conference
2020, pp. 463–473, 2020.
Ma, J., Yi, X., Tang, W., Zhao, Z., Hong, L., Chi, E., and
Mei, Q. Learning-to-rank with partitioned preference:
Fast estimation for the plackett-luce model. InInterna-
tional Conference on Artificial Intelligence and Statistics,
pp. 928–936. PMLR, 2021.
Maystre, L. and Grossglauser, M. Fast and accurate in-
ference of plackett–luce models.Advances in Neural
Information Processing Systems, 28, 2015.
Mehta, A., Nadav, U., Psomas, A., and Rubinstein, A. Hit-
ting the high notes: Subset selection for maximizing
expected order statistics.Advances in Neural Information
Processing Systems, 33:15800–15810, 2020.
Oosterhuis, H. Computationally efficient optimization of
plackett-luce ranking models for relevance and fairness.
InProceedings of the 44th International ACM SIGIR
Conference on Research and Development in Information
Retrieval, pp. 1023–1032, 2021.
Paszke, A., Gross, S., Massa, F., Lerer, A., Bradbury, J.,
Chanan, G., Killeen, T., Lin, Z., Gimelshein, N., Antiga,
L., et al. Pytorch: An imperative style, high-performance
deep learning library.Advances in Neural Information
Processing Systems, 32, 2019.
Rafailov, R., Sharma, A., Mitchell, E., Manning, C. D., Er-
mon, S., and Finn, C. Direct preference optimization:
Your language model is secretly a reward model.Ad-
vances in Neural Information Processing Systems, 36:
53728–53741, 2023.
Sakhi, O., Rohde, D., and Chopin, N. Fast slate policy
optimization: Going beyond plackett-luce.arXiv preprint
arXiv:2308.01566, 2023.
Schulman, J., Wolski, F., Dhariwal, P., Radford, A., and
Klimov, O. Proximal policy optimization algorithms.
arXiv preprint arXiv:1707.06347, 2017.Shao, Z., Wang, P., Zhu, Q., Xu, R., Song, J., Bi, X., Zhang,
H., Zhang, M., Li, Y ., Wu, Y ., et al. Deepseekmath: Push-
ing the limits of mathematical reasoning in open language
models.arXiv preprint arXiv:2402.03300, 2024.
Sutton, R. S., Barto, A. G., et al. Reinforcement learning.
Journal of Cognitive Neuroscience, 11(1):126–134, 1999.
Swaminathan, A. and Joachims, T. Counterfactual risk
minimization: Learning from logged bandit feedback.
InInternational Conference on Machine Learning, pp.
814–823, 2015.
Wang, L. and Joachims, T. User fairness, item fairness, and
diversity for rankings in two-sided markets. InProceed-
ings of the 44th ACM SIGIR International Conference on
Theory of Information Retrieval, pp. 23–41, 2021.
Williams, R. J. Simple statistical gradient-following algo-
rithms for connectionist reinforcement learning.Machine
learning, 8(3):229–256, 1992.
Wu, H., Zhang, Y ., Ma, C., Lyu, F., He, B., Mitra, B., and
Liu, X. Result diversification in search and recommen-
dation: A survey.IEEE Transactions on Knowledge and
Data Engineering, 36(10):5354–5373, 2024.
Yu, C., Lakshmanan, L., and Amer-Yahia, S. It takes variety
to make a world: diversification in recommender systems.
InProceedings of the 12th international conference on
extending database technology: Advances in database
technology, pp. 368–378, 2009.
11

Credit-assigned Policy Gradient for Early Stage Retrieval in Two-stage Ranking
A. Extended Related Work
We discuss additional related work in a broader context.
Traditional ranking recommendation and retrieval-augmented generation (RAG)In the broader context, the most
commonly used approach for ranking and retrieval-augmented generation (RAG) (Lewis et al., 2020) has been accurately
estimating the users’ valuation on items from noisy user feedback, such as implicit click data (i.e., regression-based
approach) (Liu et al., 2009). These regressed reward models are used to align items based on the predicted rewards, and
many research papers have explored model architectures and user interactions models to accurately predict the users’
valuation. A representative approach is LambdaMART (Burges, 2010), which trains a boosting algorithm to regress the
users’ interaction signals. The Plackett-Luce (PL) model has also been invented for regressing the users’ choice model,
which calculates the probability that users select the target item among the items that appear in the final ranking (Guiver
& Snelson, 2009; Maystre & Grossglauser, 2015). However, these approaches require a careful debiasing mechanism to
align items correctly from biased logged data (Joachims et al., 2017), and how to take the LSR choice into account has
remained unclear (Ma et al., 2020). Even more importantly, regression-based approaches train the ESR model with the
objective of accurately estimating the users’ interaction signal, leading to an objective mismatch with the goal ofmaximizing
the reward (Chen et al., 2019). Policy gradient (PG) approaches naturally resolve the above issue by deriving the gradient of
the ESR model directly from the final objective of the expected reward. Our work makes PG more tractable and practically
applicable in large-scale recommendation settings by tackling the variance and credit-assignment issues of two-stage PG.
Subset selection problemsSubset selection methods applied in recommender systems have mainly focused on optimizing
for utility, diversity and fairness. Earlier work proposing maximal marginal relevance (Carbonell & Goldstein, 1998) and
incorporating diversity as a key objective (Gollapudi & Sharma, 2009; Yu et al., 2009) established the necessity of selecting
a “representative” set of items rather than just the top-ranked ones to mitigate redundancy (covered more extensively
by Chakraborty & Verma (2016) and Wu et al. (2024)). Building on these, the field has increasingly moved toward framing
subset selection as a sequential decision-making problem, often leveraging Reinforcement Learning (RL) and multi-armed
bandit frameworks. Rather than treating selection as a static combinatorial task, contemporary approaches emphasize the
need to balance exploration and exploitation in online ranking (Hofmann et al., 2011) and to optimize recommendation
policies directly from logged feedback using counterfactual techniques (Swaminathan & Joachims, 2015; Li et al., 2011).
Within this narrative, Kleinberg & Raghu (2018) and, more recently, Mehta et al. (2020) focused on how individual item
utilities aggregate into a collective reward that is optimized through the lens of maximizing expected order statistics.
However, these works focus on the single-stage setting, and the credit-assignment issue in the two-stage ranking is an
underexplored problem in the subset selection literature.
Two-stage recommendation and mixture-of-experts (MoE)The mixture-of-experts (MoE) policy has been considered
effective in sampling a diverse set of items and enhancing the efficiency of exploration in two-stage recommendation. For
instance, Guo et al. (2021) empirically shows that using MoE can help improve the joint accessibility of two dissimilar
items. Hron et al. (2021) shows that nominating candidate items using multiple models can accelerate the item exploration
efficiency in two-stage bandit problems. This paper thus explores how to integrate the generalized family of PL policies
with the MoE models, and observes that MoE further improves the performance compared to the single-logit model cases of
both CA-PG and CA-PG-SwR.
B. Derivation of the Gradients
We first discuss the general form of the gradient, and then discuss how to combine the gradient with the Plackett-Luce policy
w/ and w/o sampling with replacement (SwR) approximations.
B.1. General Form of the Gradients
Here, we provide the derivation steps of naive PG. For the derivation of credit assigned PG, please refer to Appendix C.1
(i.e., proof of Theorem 3.1).
12

Credit-assigned Policy Gradient for Early Stage Retrieval in Two-stage Ranking
∇vl(π) =∇Ep(x)π(l)(al|x)[q(x, a l)]
=∇Ep(x)π ESR(AK|x)π(l)
LSR(al|x,AK)[q(x, a l)]
=∇E p(x)π ESR(AK|x)"X
al∈Aπ(l)
LSR(al|x,AK)q(x, a l)#
=∇E p(x)π ESR(AK|x)[q(l)(x,AK)]
=∇E p(x)"X
AKπESR(AK|x)q(l)(x,AK)#
=Ep(x)"X
AK∇πESR(AK|x)q(l)(x,AK)#
=Ep(x)"X
AKπESR(AK|x)∇πESR(AK|x)
πESR(AK|x)q(l)(x,AK)#
=Ep(x)"X
AKπESR(AK|x)∇logπ ESR(AK|x)q(l)(x,AK)#
=Ep(x)π ESR(AK|x)[∇logπ ESR(AK|x)q(l)(x,AK)]..(⋆)
=Ep(x)π ESR(AK|x)"
∇logπ ESR(AK|x)· X
al∈Aπ(l)
LSR(al|x,AK)q(x, a l)!#
=Ep(x)π ESR(AK|x)"X
al∈Aπ(l)
LSR(al|x,AK)∇logπ ESR(AK|x)q(x, a l)#
=Ep(x)π ESR(AK|x)π(l)
LSR(al|x,AK)
∇logπ ESR(AK|x)q(x, a l)
=Ep(x)π ESR(AK|x)π(l)
LSR(al|x,AK)p(rl|x,al)
∇logπ ESR(AK|x)rl
 
=Ep(x)π ESR(AK|x)πLSR(a1:L|x,AK)p(r1:L|x,a1:L)
∇logπ ESR(AK|x)rl
The last line is due to the independence of the reward rlfrom the other positions’ actions al′,∀l′̸=l. This means that the
reward function is given as a simple form of q(x, a l) =E[r l|x, a l]. We also use q(l)(x,AK) =P
al∈Aπ(l)
LSR(al|x,AK)·
q(x, a l)for a shorthand notation of the expected reward at position lgiven the candidate set AKand the LSR policy πLSR.
B.2. Plackett-Luce Family and Sampling with Replacement (SwR) Approximation
V-PG-SwR and CA-PG-SwR use the sampling with replacement (SwR) approximation of Plackett-Luce to calculate the
score function (i.e., log action choice probability) in the computation of the PG. In contrast, the candidate selection process
itself follows the original Plackett-Luce policy (i.e., not using the SwR process). That is, the original retrieval process
recursively applies the softmax function on theremainingactions:
∀k,1≤k≤K,ˆa k∼πESR(ak|x,A(k−1)),whereπ ESR(ak|x,A(k−1)) =exp(ˆqm(k)
ESR(x, a k)/τ)
P
a∈A\A(k−1) exp(ˆqm(k)
ESR(x, a)/τ),
andA(k−1)is the set of previously sampled actions in the candidate set. In contrast, we calculate the score function as if the
softmax function is recursively applied toallactions, instead ofremainingactions, in the following way.
∀k,1≤k≤K,ˆa k∼πESR(ak|x),whereπ ESR(ak|x) =exp(ˆqm(k)
ESR(x, a k)/τ)
P
a∈Aexp(ˆqm(k)
ESR(x, a)/τ),
Thus, when we compare V-PG and V-PG-SwR, the difference becomes as follows.
13

Credit-assigned Policy Gradient for Early Stage Retrieval in Two-stage Ranking
• V-PG:
∇logπ ESR(AK|x) =∇logQK
k=1πESR(ak|x,A(k−1);m(k))
=PK
k=1∇logπ ESR(ak|x,A(k−1);m(k))
• V-PG-SwR:
∇logπ ESR(AK|x) =∇logQK
k=1πESR(ak|x;m(k))
=PK
k=1∇logπ ESR(ak|x;m(k))
Next, we derive CA-PG-SwR. By the definition ofSK(a), we have
∇logπ ESR(SK(al)|x) =∇logπ ESR(alis selected byk’th|x)
=∇logPK
k=1πESR(alis selected atk’th|x)
=∇logPK
k=1πESR(al|x;m(k))
=∇logPK
k=1πESR(S1(al)|x;m(k))
Notably, an important distinction between V-PG-SwR and CA-PG-SwR is the (set of) actions for which the action choice
probabilities are computed. Specifically, V-PG requires evaluation of the probabilities for all actions in the candidate set
(ak), whereas CA-PG only includes evaluation of choice probabilities for the specific action selected by the LSR ( al). This
reflects the principled difference between these two approaches regarding credit-assignment.
C. Proof of Theoretical Properties
This section provides the proofs omitted from the main text.
C.1. Proof of Theorem 3.1
Proof.We useπ(l)(al|x) :=π ESR(SK(al)|x)π(l)(al|x,SK(al))to derive the policy gradient as follows.
∇vl(π) =∇Ep(x)π(l)(al|x)[q(x, a l)]
=∇Ep(x)π ESR(SK(al)|x)π(l)(al|x,SK(al))[q(x, a l)]
=∇E p(x)"X
al∈AπESR(SK(al)|x)π(l)(al|x,SK(al))·q(x, a l)#
=Ep(x)"X
al∈A∇(π ESR(SK(al)|x)π(l)(al|x,SK(al)))·q(x, a l)#
=Ep(x)"X
al∈A∇πESR(SK(al)|x)·π(l)(al|x,SK(al))·q(x, a l)#
(6)
+Ep(x)"X
al∈AπESR(SK(al)|x)· ∇π(l)(al|x,SK(al))·q(x, a l)#
(7)
Note that the decomposition in the last line is due to the derivative of the product, where the second term is non-vanishing
sinceπ(l)(al|x,SK(al))depends onπ ESR. Then,
14

Credit-assigned Policy Gradient for Early Stage Retrieval in Two-stage Ranking
(6)=E p(x)"X
al∈A∇πESR(SK(al)|x)·π(l)(al|x,SK(al))·q(x, a l)#
=Ep(x)"X
al∈AπESR(SK(al)|x)∇πESR(SK(al)|x)
πESR(SK(al)|x)·π(l)(al|x,SK(al))·q(x, a l)#
=Ep(x)"X
al∈AπESR(SK(al)|x)∇logπ ESR(SK(al)|x)·π(l)(al|x,SK(al))·q(x, a l)#
=Ep(x)"X
al∈AπESR(SK(al)|x)π(l)(al|x,SK(al))∇logπ ESR(SK(al)|x)q(x, a l)#
=Ep(x)"X
al∈Aπ(l)(al|x)∇logπ ESR(SK(al)|x)q(x, a l)#
=Ep(x)π(l)(al|x)[∇logπ ESR(SK(al)|x)q(x, a l)]
=Ep(x)π(l)(al|x)p(r l|x,al)[∇logπ ESR(SK(al)|x)r l]
 
=Ep(x)π(a 1:L|x)p(r 1:L|x,a1:L)[∇logπ ESR(SK(al)|x)r l]
(= (CA-PG))
The last line follows from the independence of the rewards, i.e., q(x, a l) =E[r l|x, a l](i.e., does not depend on (al′, rl′), l′̸=
l). We also have,
15

Credit-assigned Policy Gradient for Early Stage Retrieval in Two-stage Ranking
(7)=E p(x)"X
al∈AπESR(SK(al)|x)· ∇π(l)(al|x,SK(al))·q(x, a l)#
=Ep(x)"X
al∈AπESR(SK(al)|x)π(l)(al|x,SK(al))·∇π(l)(al|x,SK(al))
π(l)(al|x,SK(al))·q(x, a l)#
=Ep(x)"X
al∈Aπ(l)(al|x)·∇π(l)(al|x,SK(al))
π(l)(al|x,SK(al))·q(x, a l)#
=Ep(x)π(l)(al|x)∇π(l)(al|x,SK(al))
π(l)(al|x,SK(al))·q(x, a l)
=Ep(x)π(l)(al|x)"
1
π(l)(al|x,SK(al))∇ X
AKπESR(AK|x,SK(al))π(l)
LSR(al|x,AK)!
·q(x, a l)#
=Ep(x)π(l)(al|x)"X
AK∇πESR(AK|x,SK(al))·π(l)
LSR(al|x,AK)
π(l)(al|x,SK(al))q(x, a l)#
=Ep(x)π(l)(al|x)"X
AKπESR(AK|x,SK(al))∇πESR(AK|x,SK(al))
πESR(AK|x,SK(al))·π(l)
LSR(al|x,AK)
π(l)(al|x,SK(al))·q(x, a l)#
=Ep(x)π(l)(al|x)"X
AKπESR(AK|x,SK(al))∇logπ ESR(AK|x,SK(al))·π(l)
LSR(al|x,AK)
π(l)(al|x,SK(al))·q(x, a l)#
=Ep(x)π(l)(al|x)"
EπESR(AK|x,SK(al))"
π(l)
LSR(al|x,AK)
π(l)(al|x,SK(al))∇logπ ESR(AK|x,SK(al))#
·q(x, a l)#
=Ep(x)π(l)(al|x)p(r l|x,al)"
EπESR(AK|x,SK(al))"
π(l)
LSR(al|x,AK)
π(l)(al|x,SK(al))∇logπ ESR(AK|x,SK(al))#
·rl#
 
=Ep(x)π(a 1:L|x)p(r 1:L|x,a1:L)"
EπESR(AK|x,SK(al))"
π(l)
LSR(al|x,AK)
π(l)(al|x,SK(al))∇logπ ESR(AK|x,SK(al))#
·rl#!
(=∇C(π))
Again, the last line follows from the independence of the rewards, as used in the detailed derivation of Eq. (6).
C.2. Proof of Theorem 3.3
As a preparation, we first introduce the following lemma to discuss the expected rewards, which are optimized by each of
V-PG and CA-PG.
Lemma C.1.(Expected rewards for optimization) In expectation, V-PG (baseline) and the CA-PG (proposal) optimize the
objective with the following expected rewards.
(CA-PG):E πESR(AK|x,SK(al))[π(l)
LSR(al|x,AK)]·q(x, a l)(conditioned onxandSK(al))
(V-PG):Eπ(l)
LSR(al|x,AK)[q(x, a l)](conditioned onxandAK)
In the following, we denote the expected rewards of CA-PG and V-PG asq(l)(x,SK(al))andq(l)(x,AK), respectively.
Proof. We have already derived the expected reward of the vanilla PG in Appendix B.1 ( ⋆). Thus, we derive the expected
16

Credit-assigned Policy Gradient for Early Stage Retrieval in Two-stage Ranking
reward of the credit-assigned PG as follows.
∇vl(π) =Ep(x)π(l)(al|x)p(r l|x,al)[∇logπ ESR(SK(al)|x)r l]
=Ep(x)π(l)(al|x)[∇logπ ESR(SK(al)|x)q(x, a l)]
=Ep(x)π ESR(SK(al)|x)π(l)(al|x,SK(al))[∇logπ ESR(SK(al)|x)q(x, a l)]
=Ep(x)π ESR(SK(al)|x)[∇logπ ESR(SK(al)|x)·π(l)(al|x,SK(al))q(x, a l)]
=Ep(x)π ESR(SK(al)|x)"
∇logπ ESR(SK(al)|x)
· X
AKπESR(AK|x,SK(al))π(l)
LSR(al|x,AK(,SK(al)))!
q(x, a l)#
=Ep(x)π ESR(SK(al)|x)[∇logπ ESR(SK(al)|x)·E πESR(AK|x,SK(al))[π(l)
LSR(al|x,AK)]q(x, a l)]
=Ep(x)π ESR(SK(al)|x)[∇logπ ESR(SK(al)|x)·q(l)(x,SK(al))]
Note that we use π(l)
LSR(al|x,AK,SK(al)) =π(l)
LSR(al|x,AK). The last line follows from the definition of q(l)(x,SK(al)).
There are two key differences between the two expected rewards in Lemma C.1. First, V-PG considers the expected reward
marginalized over the actions chosen by the LSR policy, i.e., Eπ(l)
LSR(al|x,AK)[q(x, a l)]. In contrast, CA-PG considers an
LSR-propensity discounted reward, where the marginalized LSR propensity EπESR(AK|x,SK(al))[π(l)
LSR(al|x,AK)]serves
as a discount factor applied to the specific action reward. Second, while V-PG takes the expectation over all possible LSR
actions to calculate the expected reward, CA-PG only considers the reward of the target action, al. This is because SK(al)
has one-to-one correspondence to the final action of interest ( al), while AKis a new random variable that is conditionally
independent of al. These differences contribute to the variance reduction of CA-PG, as analyzed in Section 3.1. This
suggests that while V-PG targets the marginal expected reward, CA-PG optimizes a well-defined surrogate reward that
facilitates more efficient and robust policy improvement via gradient ascent.
Next, using the above expected reward, we provide the proof of Theorem 3.3 below.
Proof. As we have discussed in the proof outline in the main text, we need the condition that the expected reward of
credit-assigned PG should satisfyq(l)(x, SK(ak))> q(l)(x, SK(aj)). From Lemma C.1, we should have:
q(l)(x, SK(ak))> q(l)(x, SK(aj))
⇐⇒π(l)(ak|x,SK(ak))q(x, a k)> π(l)(aj|x,SK(aj))q(x, a j)
⇐⇒π(l)(ak|x,SK(ak))
π(l)(aj|x,SK(aj))>q(x, a j)
q(x, a k)
⇐⇒π(l)(aj|x,SK(aj))
π(l)(ak|x,SK(ak))<q(x, a k)
q(x, a j)
where we denoteπ(l)(a|x,SK(a)) =E πESR(AK|x,SK(a))[π(l)
LSR(a|x,AK)].
D. Detailed Computation of Score Functions and Sampling Process (in the Plackett-Luce Case)
This section discusses the computation of score function logπ ESR(SK(a|x)) for both CA-PG and CA-PG-SwR, under the
Plackett-Luce (MoE) models. Note that, while we do not describe the details here, V-PG and V-PG-SwR also use a stable
calculation of the loss function using thelogsumexpoperation.
17

Credit-assigned Policy Gradient for Early Stage Retrieval in Two-stage Ranking
Table 5.Empirical approximation error of the action choice probability relative to the ground-truth probability
τ= 1(relatively stochastic) τ= 1/2 τ= 1/5(near deterministic)
K= 10 K= 20 K= 50 K= 100 K= 10 K= 20 K= 50 K= 100 K= 10 K= 20 K= 50 K= 100
Relative abs. err. 2.3e-3 9.2e-3 2.7e-2 5.7e-2 4.2e-3 1.4e-5 5.0e-4 4.2e-3 0.0 0.0 0.0 4.7e-6
Mean logit gap (top-K) 3.20 2.78 2.28 1.89 6.40 5.55 4.57 3.78 16.0 13.9 11.4 9.45
The reported values represent the absolute approximation error relative to the ground-truth probability incurred when replacingAk−1
withAk−1(Eq. (8)), illustrating the relative magnitude of the error. The logit values of|A|= 1000of actions are randomly sampled
from the standard normal distribution, and we recursively apply softmax on the remaining items with the temperature parameter τ. As the
exact ground-truth probability is hard to calculate due to computational complexity, we estimate the ground-truth by Monte-Carlo
sampling of 1000 trials.
D.1. Gradient Computation for CA-PG
In the actual implementation, we use thelog1poperation to stabilize the loss calculation as follows.
∇logπ ESR(SK(al)|x) =∇logπ ESR(alis selected byk’th|x)
=∇log(1−π ESR(alis NOT selected byk’th|x))
=∇log1p(−QK
k=1πESR(alis NOT selected atk’th|x))
=∇log1p(−exp(PK
k=1logπ ESR(alis NOT selected atk’th|x)))
=∇log1p(−exp(PK
k=1log(1−π ESR(alis selected atk’th|x))))
=∇log1p(−exp(PK
k=1log1p (−π ESR(alis selected atk’th|x))))
Then, we estimateπ ESR(alis selected atk’th)as follows.
πESR(alis selected atk’th|x) =P
Ak−1P(alis selected atk’th|Ak−1)P(Ak−1)
≈P(a lis selected atk’th| Ak−1)(8)
=exp(logit (k)(al))P
a′∈A(k)\Ak−1exp(logit (k)(a′))
=exp(logit (k)(al))
exp(A (k)+ log1p(−exp(B (k)−A (k))))
= exp(logit (k)(a)−A (k)−log1p (−exp(B (k)−A (k))))
where A(k)= logsumexpa′∈A(logit (k)(a′))andB(k)= logsumexpa′∈Ak−1(logit (k)(a′)). We also denote
logit (k)(a) = ˆqm(k)
ESR(x, a)/τ , andAk−1= arg max Ak−1̸∋alP(Ak−1), which is the most probable k−1 candidate
set excluding the target action al. While we approximate the probability of the previously sampled items by “arg-top-
(k−1) ”, this approximation is often quite accurate. This is because it is often unlikely that lower-ranked items can be
sampled first when the item logit distribution exhibits a large logit gap (where the most likely item has a much larger logit
than the next best ones, and most logit values are low). Indeed, Table 5 shows that the (relative) approximation error of
the action choice probability is quite low across different distributions of logit values. In contrast, when the item logit
distribution is near-uniform, the probability of the top-ranked and bottom-ranked items does not differ so much. These
results indicate that we can avoid the Monte-Carlo sampling to calculate the score function, reducing the computational time
fromO(|A|K)toO(1)in the aforementioned calculation, with only a small approximation error.
This approximation is a 0-th order low temperature Taylor expansion. It is exact in the limit τ→0 , where the probability of
the most likely choice approaches 1, and all other choices approach 0. Interestingly, this approximation is also exact in
the high-temperature limit ( τ→ ∞ ). In this regime, all item and set selections are equally likely; the distributions P(ak)
andP(Ak−1)tend to uniform, and therefore the average over Ak−1values becomes an average over a constant argument,
which is equal to the argument at anyAk−1and in particular also at Ak−1.
18

Credit-assigned Policy Gradient for Early Stage Retrieval in Two-stage Ranking
D.2. Gradient Computation for CA-PG-SwR
Next, we implement CA-PG-SwR usinglogsumexpoperation to stabilize the loss calculation as follows.
∇logπ ESR(SK(al)|x) =∇log KX
k=1πESR(S1(al)|x;m(k))!
=∇log KX
k=1exp(logit (k)(al))P
a′∈Aexp(logit (k)(a′))!
=∇log KX
k=1exp(logit (k)(al))
exp 
log P
a′∈Aexp(logit (k)(a′))!
=∇log KX
k=1exp(logit (k)(al))
exp 
logsumexp a′∈A(logit (k)(a′))!
=∇log KX
k=1exp 
logit (k)(al)−A (k)!
=∇logsumexpK
k=1 
logit (k)(al)−A (k)
where we let A(k)= logsumexpa′∈A(logit (k)(a′))andlogit (k)(a) = ˆqm(k)
ESR(x, a)/τ . Note that because
(logit (k)(al)−A (k))becomes the same among the ks that share the same model, the computational time is reduced from
O(KL)toO(ML), whereKis the size of candidates,Mis # of MoE models, andLis the length of LSR’s output.
D.3. Sampling Process of Top-Kunder Plackett-Luce
The recursive application of softmax of the Plackett-Luce policy is often modeled by the Gumbel top- Ktrick for the
computational efficiency (Kool et al., 2019). We also follow this procedure and implement the sampling process as follows.
• (Case 1: when using a single logit model)
1. Sample a value from Gumbel distribution for each action asη a∼Gumbel(0,1).
2. Retrieve top-Kbased on the logit value and Gumbel noise asAK= arg-top-K a∈A(logit(a) +η a)
• (Case 2: when using MoE logit models)
1. Sample a value from Gumbel distribution for each action asη a∼Gumbel(0,1).
2. Recursively choosek’th item asa k= argmaxa∈A\A(k−1) (logit (k)(a) +η a)
wherelogit (k)(a) = ˆqm(k)
ESR(x, a)/τis the predicted value of actionagiven contextx.
E. Additional Details and Experiment Results
This section provides additional details on the experiment and reports the whole training process. Code is available on a
GitHub repository: https://github.com/facebookresearch/early stage retrieval.
Varying alignment of LSRAs described in Section 4.1, we use the following oracle Plackett-Luce policy as the given
LSR policy:
π(l)
LSR(al|x,AK) :=exp(q(x, a l)/τLSR)P
a′∈AK\a1:l−1exp(q(x, a′)/τLSR),
where q(x, a) is the ground-truth expected reward. We can simulate the varying alignment (optimality) of LSR by adjusting
the temperature hyperparameter τLSRin each configuration. Specifically, the “optimal” policy is modeled by τLSR→+0 ,
“anti-optimal” policy is modeled by τLSR→ −0 , and the “uniform-random” policy is modeled by τLSR→ ±∞ (In
implementation, we used the exact argmax and exact uniform random policy for “optimal”, “anti”, and “uniform”, instead of
adjusting the temperature hyperparameter of the softmax). The “noisy-optimal” policy uses τLSR= 1.0 and the top-1 action
choice probability of the best action among the candidate was around 0.6 on average.
19

Credit-assigned Policy Gradient for Early Stage Retrieval in Two-stage Ranking
Learning rates and batch sizesWe selected the learning rate based on the training stability and policy value at the
stopping or convergence point of CA-PG and V-PG on the default setting (i.e., K= 10 ,L= 10 ,M= 1 , and optimal LSR)4.
Then, we applied the same learning rate to different configurations. This is to see the robustness of each PG method to the
varying configurations. The specific learning rate used for CA-PG/CA-PG-SwR is 1e-2, and for V-PG/V-PG-SwR is 1e-1.
For both methods, we use a batch size of 128. A large value of the batch size is set to mitigate the variance issues of V-PG.
We also test a simple adjustment of the learning rate depending on the problem instance. First, we observe that the scale of
the loss function of V-PG becomes linearly large as Kbecomes large. Therefore, we consider a small adjustment of the
learning rate, in which we normalize the loss scale by multiplying the original learning rate by K−1, when we vary the # of
candidates. However, this adjustment further slows down the training process for large values of K, and thus, we reject this
adjustment. Another possible adjustment is to scale up the learning rate depending on the size of the action space ( |AK|).
However, this approach is not tractable, as the scaling factor becomes exponentially large as Kgrows. In particular, our
experimental setting involves an action space of order ( |A|= 1e3), making this approach infeasible. Due to this reason, we
reuse the same learning rate across varying sizes of candidatesKfor V-PGs.
Next, for CA-PGs, we know from Lemma C.1 that only CA-PGs optimize the policy with LSR propensity discounted reward
using the following policy gradient in expectation (the derivation is provided in Appendix C.2):
(CA-PG):E p(x)π ESR(SK(al)|x)h
∇logπ ESR(SK(al))·
EπESR(AK|x,SK(al))[π(l)
LSR(al|x,AK)]·q(x, a l)i
(V-PG):E p(x)π ESR(AK|x)h
∇logπ ESR(AK)·
Eπ(l)
LSR(al|x,AK)[q(x, a l)]i
This means that the magnitude of the loss values can change depending on the level of stochasticity of the given LSR policy
– while we have no discount when using a deterministic LSR, we have a large discount in the magnitude of the reward
signal when using a uniform random LSR. Therefore, we expect that the training of CA-PG/CA-PG-SwR should be slowed
down without scaling with the inverse propensity of LSR’s action choice probability from theoretical analysis. To mitigate
this reward scaling issue, we recommend the use of an adaptive learning rate, where the learning rate should be scaled
proportionally to the LSR propensity to the greedy action as follows.
(adaptive lr) = (original lr)×(E p(x)[avgAK(Eπ(l)
LSR(a|x,AK)[π(l)
LSR(a|x,AK)])])−1.(9)
We use this simple adjustment of the learning rate in the synthetic experiment when we vary the alignment (optimality)
of LSR. Note that the adaptive rate is consistent across the whole training step, and can be estimated by the Monte-Carlo
estimation only once at the beginning of policy training, enabling the simple adjustment of the reward scale, when we use a
stochastic and consistently aligned policy for LSR5. As we continue using the default learning rate of 1e-2, the resulting
adaptive learning rates of “noisy”, “uniform”, and “anti” settings become 2e-2, 1e-1, 1e-2, after rounding the values.
Computational timeWe compare the computational time of V-PG, V-PG-SwR, CA-PG, and CA-PG-SwR (1) with a
single logit model ( M= 1 ) and varying candidate-set size ( K) and the length of ranking ( L), and (2) with MoE logit
models ( M) and varying candidate-set size ( K) with a single output ( L= 1 ), in the synthetic setting (Section 4.1). For this
experiment, we use a MacBook Pro (M1, 10 cores, 16GB memory) and run the experiments with CPUs without explicit
parallel computation. All implementations are completed by PyTorch (Paszke et al., 2019).
Figure 4 reports the results of the computational time comparison. In Figure 4 (Left), we vary the size of problem instances
as(K, L)∈ {(5,1),(10,1),(20,1),(20,5)} . The first three settings are to see the changes with varying sizes of candidates
(K), and the last two settings are to study how the length of ranking ( L) affects the computational time. The results
demonstrate that the SwR approximation, including CA-PG-SwR and V-PG-SwR, has a clear benefit over exact calculation
of the Plackett-Luce, including V-PG and CA-PG. Specifically, while CA-PG and V-PG require more runtime when the
configurations ( (K, L) and(K,) , respectively) increase, the runtime of CA-PG-SwR and V-PG-SwR stays almost constant
across varying problem instances. These results indicate that CA-PG-SwR not only reduces the computational order from
O(KL) toO(L) , but the SwR approximation itself contributes to reducing the constant multiplier of the computational
time.
4In a practical situation, the selection of the learning rate based on these results is almost infeasible as we need to run multiple
compared method requiring user interaction. However, we employ this selection to ensure fair comparison among the PG methods.
5A ”consistently aligned” policy means that the order of any two items is preserved regardless of the remaining items in the candidates.
20

Credit-assigned Policy Gradient for Early Stage Retrieval in Two-stage Ranking
Table 6.Comparing PG methods combined with GRPO (Shao et al., 2024) (i.e., other variance reduction methods in RL)
PG methods# of candidates (K)
20 50 100 200
V-PG-SwR + GRPO 8.71 (±0.03) 8.84(±0.01) 8.93(±0.02) 9.00(±0.02)
(baseline) →8.88(±0.03) →8.98(±0.03) →9.02(±0.03) →9.06(±0.03)
CA-PG-SwR + GRPO 8.58(±0.03) 8.89 (±0.02) 9.02 (±0.02) 9.08 (±0.02)
(ours) →8.97 (±0.03) →9.07 (±0.03) →9.10 (±0.02) →9.11 (±0.03)
Thegreen font shows the best performance. The policy value is measured at either @50K (top) or @500K (bottom) gradient steps. The
results are based on 10 random seeds.
Next, Figure 4 (Right) compares V-PG-SwR and CA-PG-SwR in a larger problem configurations, e.g., K∈ {50,100,200}
andM∈ {1,5} , with L= 1 . The results indicate that the computational time scales quite similarly in these two methods,
where we observe the increase of computational time proportional to # of MoE models ( M), while the computational time
does not change much from the K= 5 case (in the right panel) even when we use a large candidate-set size like K= 200 .
This indicates that the training-time computational cost of PG methods with the SwR approximation scales well with the
candidate-set size (K).
From these results, we recommend the use of TOP1-PG (i.e., CA-PG-SwR in the case of M= 1 ) in a large-scale search and
ranking problem where Lbecomes large. In contrast, TOP1-PG can limit the ability to make most of MoE, as we have
seen in Table 3 in Section 4.1. This suggests a tradeoff between performance and (training) computational cost of CA-PGs
(M≥2) and TOP1-PG (M= 1).
Detailed training processesWe report the training processes of the experiments with (1) varying alignment (optimality)
of ESR and (2) varying numbers of outputs ( L) and MoE models ( M) in the synthetic setting, and (3) varying candidate
sizes (K) in the real-data setting in Figures 5-7.
Additional results combining GRPOFinally, as discussed in the related work (Section 5), we test the performance of
both V-PG and CA-PG combined with other variance reduction methods, especially with a prevalent choice of GRPO (Shao
et al., 2024). GRPO queries msamples per context and action rj(x, a), j∈[m] , and normalizes the reward as r′
j=
(rj−mean(r))/std(r) . While GRPO is not directly applicable to CA-PG due to the reward positivity requirement in
Theorem 3.3, we can easily integrate this reward normalization to CA-PG by adding a positive constant ( c) to the reward as
(r′
j+c).6Note that, implementing GRPO is unrealistic in the recommender systems (RecSys) setting, as GRPO needs to
query rewards for multiple rankings (i.e., mdifferent samples per context). This is because, unlike LLM training, which
can retrieve multiple rewards from the reward simulator, in RecSys, oftentimes only a single reward feedback is available
through online interaction with users. However, the main objective of this additional experiment is to provide evidence that
CA-PG can be easily integrated into other variance reduction methods of RL policies.
Table 6 reports the expected reward of the policies learned by (1) V-PG-SwR + GRPO and (2) CA-PG-SwR + GRPO.
To compare the results in a practical situation, we use five MoE base models ( M= 5 ) and vary the candidate-set size
asK∈ {20,50,100,200} . The results show that both V-PG-SwR and CA-PG-SwR improve the convergence speed
combined with GRPO, and the difference between the V-PG-SwR and CA-PG-SwR becomes small when comparing the
expected reward at 50K gradient steps. However, CA-PG-SwR + GRPO achieves a better performance than V-PG-SwR in
the large candidate-set size regime, suggesting that the policy gradient methods can gain benefits from both GRPO-type
variance reduction (i.e., reward normalization) and CA-PG-type variance reduction (i.e., efficient marginalized action choice
probability calculation).
6For V-PG, we tested the performance on both of the following two situations: (1) not adding a constant (the same as original GRPO),
and (2) adding a constant (the same modification as CA-PG), and report the best results (1) among these two choices.
21

Credit-assigned Policy Gradient for Early Stage Retrieval in Two-stage Ranking
Figure 4.Comparison of the computational time for running 1K gradient steps.We ran each PG method for 10 random seeds. The
result reports the mean and standard deviation of the runtime, excluding trials in which gradient overflow occurred. The left figure uses
M= 1(i.e., single logit model), and the right figure usesL= 1(i.e., final ranking length is 1).
Figure 5.Comparing the training process of each PG method with varying alignment (optimality) of LSR.
Figure 6.Comparing the training process of each PG method with varying # of outputs (L) and MoE models (M).
Figure 7.Comparing the training process of each PG method with varying candidate sizes (K) in the real-data experiment.
22