# RobustMask: Certified Robustness against Adversarial Neural Ranking Attack via Randomized Masking

**Authors**: Jiawei Liu, Zhuo Chen, Rui Zhu, Miaokun Chen, Yuyang Gong, Wei Lu, Xiaofeng Wang

**Published**: 2025-12-29 08:51:35

**PDF URL**: [https://arxiv.org/pdf/2512.23307v1](https://arxiv.org/pdf/2512.23307v1)

## Abstract
Neural ranking models have achieved remarkable progress and are now widely deployed in real-world applications such as Retrieval-Augmented Generation (RAG). However, like other neural architectures, they remain vulnerable to adversarial manipulations: subtle character-, word-, or phrase-level perturbations can poison retrieval results and artificially promote targeted candidates, undermining the integrity of search engines and downstream systems. Existing defenses either rely on heuristics with poor generalization or on certified methods that assume overly strong adversarial knowledge, limiting their practical use. To address these challenges, we propose RobustMask, a novel defense that combines the context-prediction capability of pretrained language models with a randomized masking-based smoothing mechanism. Our approach strengthens neural ranking models against adversarial perturbations at the character, word, and phrase levels. Leveraging both the pairwise comparison ability of ranking models and probabilistic statistical analysis, we provide a theoretical proof of RobustMask's certified top-K robustness. Extensive experiments further demonstrate that RobustMask successfully certifies over 20% of candidate documents within the top-10 ranking positions against adversarial perturbations affecting up to 30% of their content. These results highlight the effectiveness of RobustMask in enhancing the adversarial robustness of neural ranking models, marking a significant step toward providing stronger security guarantees for real-world retrieval systems.

## Full Text


<!-- PDF content starts -->

RobustMask: Certified Robustness against Adversarial Neural Ranking Attack via
Randomized Masking
Jiawei Liu1, Zhuo Chen1, Rui Zhu2, Miaokun Chen1, Yuyang Gong1, Wei Lu1∗, XiaoFeng Wang3∗,
1Wuhan University,2Yale University,3Nanyang Technological University
Abstract
Neural ranking models have achieved remarkable progress
and are now widely deployed in real-world applications such
as Retrieval-Augmented Generation (RAG). However, like
other neural architectures, they remain vulnerable to adversar-
ial manipulations: subtle character-, word-, or phrase-level per-
turbations can poison retrieval results and artificially promote
targeted candidates, undermining the integrity of search en-
gines and downstream systems. Existing defenses either rely
on heuristics with poor generalization or on certified meth-
ods that assume overly strong adversarial knowledge, limit-
ing their practical use. To address these challenges, we pro-
poseRobustMask, a novel defense that combines the context-
prediction capability of pretrained language models with a
randomized masking-based smoothing mechanism. Our ap-
proach strengthens neural ranking models against adversarial
perturbations at the character, word, and phrase levels. Lever-
aging both the pairwise comparison ability of ranking models
and probabilistic statistical analysis, we provide a theoretical
proof of RobustMask’s certified top-K robustness. Extensive
experiments further demonstrate that RobustMask success-
fully certifies over 20% of candidate documents within the
top-10 ranking positions against adversarial perturbations af-
fecting up to 30% of their content. These results highlight the
effectiveness of RobustMask in enhancing the adversarial ro-
bustness of neural ranking models, marking a significant step
toward providing stronger security guarantees for real-world
retrieval systems.
1 Introduction
Neural Ranking Models (NRMs), particularly those built on
pre-trained language models, have demonstrated remarkable
success in information retrieval. Yet, they remain inherently
susceptible to a wide range of adversarial exploits that can
significantly degrade performance under malicious attacks, ex-
posing a critical lack of robustness. For example, adversaries
∗Corresponding authors.can manipulate the behavior of the model through carefully
crafted inputs or operations to launch information manipu-
lation attacks [29]. Such exploits not only compromise the
quality of search results, but also pose serious security risks to
retrieval systems and retrieval-augmented LLM applications.
As a result, a central challenge in information retrieval lies in
strengthening the robustness of NRMs and designing effective
defenses against information manipulation attacks.
Robustness in ranking models. More specifically, the lack
of robustness in information retrieval models can lead to de-
graded user experience, lower customer retention, reduced
product competitiveness, and even deliberate exploitation for
information manipulation, with the potential to influence pub-
lic opinion and perception [13,18,29,43]. Adversarial attacks
against retrieval models can arise in various scenarios, lever-
aging tactics such as fake news, misleading advertisements,
and biased content to advance objectives in domains ranging
from political propaganda and marketing to social media ma-
nipulation and malicious search engine optimization [7, 58].
Prior efforts to improve the robustness of text ranking mod-
els have been largely empirical. Some studies attempt to de-
tect adversarial samples using features such as TF-IDF [57]
and perplexity [32, 46], while others focus on generating ad-
versarial examples based on known attack methods for adver-
sarial training [3]. However, these empirical defenses remain
fundamentally limited, as they cannot fully capture the di-
verse strategies adversaries may employ, and consequently
provide only modest gains in robustness, not to mention any
guarantee for the protection.
To counter unknown or adaptive attacks and break the on-
going arms race between attackers and defenders, it is crucial
to establish certifiable robustness guarantees, theoretically
assurances that a model’s predictions remain stable under
bounded adversarial perturbations. However, progress in this
direction remains limited. A notable attempt is the work of
Wu et al. [49], which leverages synonym substitution to gener-
ate output ranking scores for randomly perturbed documents,
thereby constructing a smoothed ranker based on averaged
scores. Using the ranking and statistical properties of these
1arXiv:2512.23307v1  [cs.CR]  29 Dec 2025

Query
Target Ranker
…… …
CollectionRank
Rank
…
1 2 K K+1 K+2
…
1 2 K K+1 K+2Benign Ranking List
Attacked Ranking List
Certified Top-K
Robustness
𝑥1,𝑥2,𝑥3,…𝑥𝑇Document K
𝑥1,MASK,𝑥3,…𝑥𝑇[MASK],𝑥2,𝑥3,…𝑥𝑇
𝑥1,𝑥2,[MASK],…𝑥𝑇 𝑥1,𝑥2,𝑥3′,…𝑥𝑇Pairwise Ranker
Random Masking𝐶1
𝐶2
𝐶3𝐶
Predict
+
ProveRelevance (Document K)
∨
Relevance (Document K+1)Figure 1: The overview of the Certified Top-K Robustness and the architecture of our proposed RobustMask method.
perturbed documents, they derive a criterion for certifying
the robustness of the target model. However, the key limita-
tion of this approach lies in its strong assumption that the
defender has prior knowledge of the adversary’s synonym
substitution strategy and lexicon, an unrealistic premise that
severely undermines its practical applicability. Moreover, the
enhancement comes at the cost of substantial performance
degradation on clean data.
Our approach. In this paper, we introduceRobustMask, a
theoretically certifiable robustness enhancement technique
for text information retrieval models based onrandom mask-
ing smoothing. By leveraging the inherent capabilities of
pre-trained models, our randomized masking mechanism con-
structs a smoothed ranking model that mitigates the impact
of diverse adversarial perturbations. Unlike prior approaches,
RobustMask is built on more realistic assumptions and pro-
vides formal guarantees that a model’s top-K predictions re-
main stable within a bounded range of adversarial manipu-
lations. While it primarily certifies the stability of the top-K
retrieved results, which is a limited subset of the candidate
list, these prioritized rankings align closely with user-centric
information needs. As reflected in standard retrieval metrics
such as MRR and NDCG, highly relevant documents provide
far greater utility to users than lower-ranked items, whose
relevance and impact diminish significantly [29, 36].
Empirical experiments show that our method substantially
outperforms the previous state-of-the-art. Compared with
CertDR, which achieves a Top-10 CRQ of 9.5% [48], we
achieve a Top-10 CRQ of 58%, and we reduce the attack suc-
cess rate of the PRADA [49] from 57.4% to 16.7%. Beyond
PRADA, a representative word- and character-level adver-sarial attack, our approach also defends effectively against
phrase-level attacks such as keyword stuffing, Collision [46],
and PAT [29]. RobustMask integrates smoothly with language-
model pretraining objectives, thereby minimizing its impact
on retrieval efficiency. On MSMARCO dataset, it incurs only
a 1%–2% drop in MRR@10 and NDCG@10 relative to the
original ranking model while delivering superior defense ef-
fectiveness, whereas CertDR induces a roughly 3%–5% degra-
dation under the same setting. These results demonstrate that
RobustMask attains strong adversarial robustness with min-
imal retrieval performance overhead. Given the widespread
adoption of RAG frameworks in modern LLM applications,
our study represents a critical step toward strengthening the
adversarial robustness of retrieval systems and safeguarding
RAG pipelines against security vulnerabilities.
Contributions. The contributions of this paper are threefold:
(1) We propose RobustMask, leveraging the mask predic-
tion capability from pre-trained language models and a robust
randomized masking smoothing mechanism, to effectively
defend neural text ranking models against adversarial pertur-
bations at character, word, and phrase levels.
(2) We theoretically establish the robustness guarantees of
RobustMask by using pairwise ranking comparison capabil-
ities and probabilistic statistical methods, providing formal
and provable certification for the Top-K ranking robustness
of neural ranking models.
(3) We verify the practicality and effectiveness of Robust-
Mask, clearly demonstrating that the method can certify the
robustness of over 20% of candidates in the top-10 rankings
against any perturbation up to 30% of content. It demonstrates
significant improvements in robustness over existing methods.
2

2 Related works
2.1 Neural Text Ranking Models
Classical text ranking methods primarily rely on exact term
matching, typically employing efficient bag-of-words repre-
sentations such as the BM25 algorithm [42]. To mitigate
the vocabulary mismatch issue inherent in exact term match-
ing, neural ranking models introduced continuous vector rep-
resentations, such as word2vec [33] and Glove [38], com-
bined with neural network architectures to perform semantic
soft-matching. Existing neural ranking frameworks can be
broadly categorized into three types: representation-based
models [22,44], interaction-based models [20,21], and hybrid
models integrating both approaches [34]. Transformer-based
models pretrained with language-modeling objectives, notably
BERT [11], have significantly advanced the state-of-the-art
in neural text ranking. [37] first demonstrated BERT’s ef-
fectiveness for text ranking tasks. Subsequently, numerous
pretrained transformer-based language models fine-tuned on
domain-specific corpora achieved substantial performance
gains [9, 10, 14, 39]. Even in the LLM era, pre-trained models
based on BERT architecture are still one of the core solu-
tions for retrieval technology and downstream applications of
LLMs, such as RAG [15, 45, 53].
2.2 Adversarial Ranking Attack
Neural text ranking models have inevitably inherited vulner-
abilities characteristic of neural networks, particularly sus-
ceptibility to adversarial examples [29, 46]. Such adversarial
attacks pose severe security threats by intentionally mislead-
ing ranking models into incorrect judgment, while preserving
a surface-level fluency that bypasses conventional detection
mechanisms. Specifically, adversarial attacks exploit the in-
herent relational dependencies between queries and ranked
candidates, deliberately manipulating textual inputs to delib-
erately distort rankings. Typically, adversaries subtly inject
semantic perturbations into textual content at varying granu-
larities, including character-based [12], word-level [31,41,49],
and phrase-level modifications [5, 6, 46]. Earlier research pri-
marily focused on adversarial attacks under white-box scenar-
ios, which assume full access to model internals and gradi-
ents [19,41,46]. Recent advances have expanded to more prac-
tical and realistic black-box scenarios, manipulating semantic
relevance to compromise retrieval effectiveness [2, 6, 16].
2.3 Defenses against adversarial text ranking
attacks
To mitigate textual adversarial attacks, previous works pro-
pose various empirical defense strategies, primarily involv-
ing input pre-processing [26, 40], adversarial training [17],
and robust model architectures [35]. However, empirical de-fenses inherently face limitations. In practice, they tend to be
scenario-specific and often fail when confronted with novel
or adaptive adversaries [1]. For instance, defenses effective
against character-level perturbations typically lose effective-
ness when facing word substitution attacks, and vice versa.
Moreover, empirical defense methods generally do not of-
fer rigorous robustness guarantees about the model’s perfor-
mance within clearly defined perturbation bounds.
Unlike empirical methods, certified robustness defenses
seek to provide provable guarantees about model predictions.
Such methods certify that, within a certain explicitly defined
perturbation set around an input, the model’s output remains
unchanged, thus delivering theoretically grounded robustness
guarantees [28, 56]. Representative certification methods in-
clude Interval Bound Propagation (IBP) [25], linear relax-
ation [23], and randomized smoothing [8,54,55]. Particularly,
randomized smoothing’s model-agnostic nature and scalabil-
ity led to its notable application in textual robustness settings.
For adversarial attacks specifically targeting text ranking
tasks, relatively few studies have been conducted. Chen et
al. [3] try enhance the robustness and effectiveness of BERT-
based re-ranking models in the presence of textual noise. Ad-
ditionally, Chen et al. [4] investigated defense against adver-
sarial ranking attacks through detection approaches. They
evaluated the efficacy of various detection methods, including
unsupervised methods relying on spamicity scores, perplexity,
and linguistic acceptability measures, as well as supervised
classification-based detectors leveraging BERT and RoBERTa
representations. Regarding certified robustness specifically
in adversarial text ranking attacks, Wu et al. [48] propose
CertDR, a certified defensive approach built upon random-
ized smoothing techniques designed to enhance the robustness
of text ranking models specifically against synonym-based
word substitution attacks. Nonetheless, their certification as-
sumptions, limiting perturbations exclusively to synonym sub-
stitutions, may appear overly restrictive and unrealistic for
many practical scenarios.
3 Preliminaries
3.1 Adversarial Textual Ranking Attack
In this paper, we primarily focus on the re-ranking phase
within the two-stage retrieval process (recall-rerank). For
the re-ranking phase, given a textual query qand a set
of candidate documents {x1,x2,···,x N}, the ranking model
can calculate the relevance score s(q,x i)for each candidate
document xi∈D, and subsequently generate a ranked list
L=x 1,x2,···,x N, such that x1≻x 2≻ ··· ≻x N, ifs(q,x 1)>
s(q,x 2)>···>s(q,x N). Here, it is assumed that the relevance
probability score s(q,x i)ranges between 0 and 1, which can
be straightforwardly achieved by applying a sigmoid or soft-
max function to the logits output from the ranking model.
Attacks aimed at manipulating neural text ranking mod-
3

els primarily seek to identify a sequence of text that inten-
tionally disrupts the intended order of ranking. This paper
attempts to propose a theoretically grounded method to en-
hance robustness against such attacks. To maintain general-
izability, the focus is on manipulation attacks based on word
substitution. For example, assume s(q,x i)>s(q,x j), where
xj={w 1,w2,...,w T}. An adversarial manipulator might gen-
erate an adversarial sample x′
j={w′
1,w′
2,...,w′
T}based on
the query qand the document xj, by perturbing up to R≤T
words in xj, in such a way that the target ranking model makes
an error. For an adversarial information manipulation attacker,
an adversarial sample x′
jis considered effective if the follow-
ing conditions are satisfied:
s(q,x j)<s(q,x′
j),||x′
j−x j||0≤R(1)
where||x′
j−x j||0=∑T
t=1I{w t̸=w′
t}represents the Hamming
distance, and I{·} is an indicator function. Here, Sx:={x′:
||x′
j−x j||0≤R} denotes the set of candidate adversarial sam-
ples available to the attacker. Ideally, all x′
j∈Sxwould possess
the same meaning from the perspective of human readers, yet
could lead to different relevance judgments by the model,
potentially elevating their rank according to the target ranking
model. For character-level perturbations in English, w′
tmight
be a visually or phonetically similar typographical or spelling
error replacement for wt. For word-level perturbations, w′
t
could be a synonym of wtselected based on the attacker’s
adversarial strategy. In terms of phrase-level, it is formed
by multiple combinations of character-level and word-level
perturbations. Practically, the defender does not have prior
knowledge of these substitution attack strategies.
3.2 Certified Top-K Robustness
Inspired by the certified Top-K robustness definition for rank-
ing models introduced by Wu et al. [48], and the definition
of certified robustness for text classification models provided
by Ye et al. [54], a ranking model can be considered certi-
fied robust if it guarantees that adversarial manipulations of
the input will always result in the failure of the attack. This
means that regardless of how the attacker modifies the input,
the ranking model consistently maintains its effectiveness and
integrity of Top-K results, thereby thwarting any attempt to
disrupt the intended order of results.
It is well known that in practical search scenarios, users are
more concerned with the top-ranked results rather than the
lower-ranked candidates, as continued exploration of search
results often leads to a sharp decline in click-through rates
and traffic. Widely-used retrieval ranking evaluation metrics,
such as Recall@N (R@n), normalized Discounted Cumula-
tive Gain (nDCG), and Mean Reciprocal Rank (MRR), also
emphasize the importance of top-ranked candidates. There-
fore, protecting the top-ranked relevant candidates is crucial
not only for downstream applications but also for ensuring the
reliability of widely-used ranking metrics [48, 50]. A rankingmodel scan be considered certified Top-K robust if it can
ensure that no document ranked beyond the top Kcan be pro-
moted into the top Kpositions of the ranking list Lthrough
adversarial attacks. This implies that the ranking model sis
robust against adversarial manipulations that attempt to alter
the ranking of documents within the critical topKpositions.
Based on this, the certified Top-K robustness in the context
of information retrieval can be formally defined as follows:
Given a query qand the corresponding ranking list Lqpro-
duced by a ranking model s, we suppose that an attacker
conducts adversarial word substitution attacks with specific
intensity on any document d∈L q[K+1 :] . If the ranking
model scan ensure that these attacked documents x′∈S xre-
main outside the top Kpositions of the ranking list, then the
ranking model sis considered certified Top-K robust against
word substitution attacks, i.e.:
Rank(s(q,x′))>K,∀x∈L q[K+1 :]&∀x′∈S x(2)
where Rank(s(q,x′))represents the ranking position of the
adversarially modified document x′in the ranking list Lqgiven
by the ranking models.
4 RobustMask
4.1 Method Overview
The core motivation behind the proposed RobustMask method
is as follows: if a sufficient number of words are randomly
masked from a text, and relatively few words are deliber-
ately perturbed, it becomes unlikely that all perturbed words,
which are selected through adversarial attacks, will appear in
the masked text. Retaining only a subset of these malicious
words is usually insufficient to deceive the text classifier. Thus,
to avoid the computationally expensive combinatorial opti-
mization problem, e.g., enumerating all candidate adversarial
documents inS x. Inspired by the concept of random smooth-
ing, this paper proposes a defensive method, RobustMask, to
enhance the robustness of the baseline text ranking models.
Considering that pre-trained models utilize a masked lan-
guage modeling (MLM) task during pre-training, leveraging
their capability to predict masked tokens can enhance the
model’s contextual understanding and robustness [11, 30]. In
this paper, we introduce this masking mechanism to improve
the certified robustness of ranking models. This approach
constructs a smoothed ranking model as a substitute for the
target model, thereby avoiding the exponential computational
cost of certified verification. In conjunction with the relative
position relationships in ranking lists, this paper transforms
the pointwise relevance score prediction task into a pairwise
relative relevance prediction task. This is achieved by repeat-
edly performing random masking operations on the input text
to generate numerous masked copies of the text. The base
pairwise ranking model is then used to classify each masked
4

text. The final robust relative relevance judgment is made
using a “voting” method.
Many studies have shown [24, 51, 52, 55] that these adver-
sarial examples themselves are highly sensitive and fragile,
easily affected by small random perturbations. If some words
are randomly masked before feeding the adversarial exam-
ples into the classifier, it is more likely that the erroneous
predictions of the adversarial examples will be corrected to
the correct predictions.
4.2 Theoretical Certification Robustness
In this paper, we adoptx ⊖x′to denote the set of token coor-
dinate indices wherexandx′differ. Consequently, we have
|x⊖x′|=|| x−x′||0. For example, supposex =“A B C D E F”
andx′=“A B G H E J” . Then we havex ⊖x′={3,4,6} and
|x⊖x′|=3 . Additionally, let D={1,2,...,T} be the set of
coordinate indices. We denote I(T,k)⊆P(D) as the set of
all possible subsets of Dcontaining kunique coordinate in-
dices, where P(D) is the power set of D.U(T,k) represents
the uniform distribution over I(T,k) . Sampling from U(T,k)
means uniformly sampling kcoordinate indices from the set
ofTcoordinate indices without replacement. For example, a
possible element Hsampled from U(6,3) could be {1,2,5} .
Define a masking operation M: x×I(T,k)→ xmask,
wherex mask is a text with some of its words masked. This
operation takes as input a textxof length Land a set of
coordinate indices, and outputs the masked text where all
words except those at the specified coordinate indices are
replaced with a special token [MASK]. Since the [MASK]
token is used by pre-trained language models during their
pre-training phase [11, 30], we will also use this token in this
paper to facilitate the efficient utilization of the pre-trained
knowledge. For example, M(“A B G H E J”,1,2,5) =
“A B [MASK] [MASK] E [MASK]”.
Previous work [48] directly utilized pointwise scores to
determine relative relevance, employing a predefined pertur-
bation vocabulary for a two-stage proof based on a bi-level
min-max optimization. This paper eliminates the dependency
on a predefined vocabulary, allowing adversarial word replace-
ment attacks to use any substitute words.
Specifically, given a query qand the corresponding rank-
ing list Lqprovided by the ranking model s, for a document
x∈L q[K+1 :] , the model sis Top-K robust to an adversarial
document sample x′if it satisfies s(q,x K)>s(q,x′). Taking
the relative relevance order into consideration, it is only nec-
essary to prove that s(q,x K)>s(q,x′
K+1), where x′
K+1is the
adversarial sample corresponding to the document ranked at
position K+1 . To ensure the ranking model performs consis-
tently and robustly on the data subject to masking operations,
we fine-tuned the model. We let the model fbe a pairwise rel-
ative relevance judgment classifier that performs the mapping
f:X mask→Y . The function fis trained to classify the triplet
(query q, document 1, document 2 )based on the relative rele-vance between document 1 and document 2, where one of the
documents is xKand the other is xK+1. Specifically, the input
can be represented as either [q,x K,xK+1]or[q,x K+1,xK], with
the label set Y=0,1 . Since the concatenation method being
one of the most effective approaches for achieving superior
re-ranking performance, we also employ the concatenation
method as the fundamental structure of the model. Formally,
the query qand a pair of candidate items (xi,xj)are concate-
nated using the [SEP] and [CLS] tokens. The relevance score
is calculated through a linear layerW∈R768×2:
si=LM([CLS;q;SEP;x i;SEP])∗W(3)
sj=LM([CLS;q;SEP;x j;SEP])∗W(4)
wheres iands j∈R2represent the relevance scorers for the
positive and negative examples, respectively (also denoted as
sposandsneg).LM(·) denotes the representational embedding
vector derived by the model LMfor a given text input. In prac-
tice, the model LMused in this study is a BERT model with
shared parameters, which is trained on the two concatenated
inputs. The loss is calculated as following:
L(q,x i,xj) =−y i,jlog(so ftmax(s i−s j)(5)
where y(i,j)∈Y represents the one-hot label for [q,x i,xj]. To
ensure a balanced distribution of labels in the training data,
this study generates triplets with a label of 0 by swapping
the positions of the candidate with relative positive relevance
and the candidate with relative negative relevance from the
original data. As a result, the g(x) for aggregated copies clas-
sification can be defined as:
g(x) =argmaxc∈Y[PH∼U(T,k) (f(M(x,H) =c))]
| {z }
pc(x)(6)
where Trepresents the length ofx, and kis the number of
unmasked words retained inx, calculated as [T−ρ×T] .ρde-
notes the ratio of masked words. pc(x)refers to the probability
that freturns class cafter random masking. The predictions
of the smoothed classifier can be shown to be consistent with
the input perturbations.
The model for ranking in g(x)can be defined as:
g(x) =E H∼U(T,k) [s(q,M(x,H))](7)
Here, we give the definition of Theorem 1.
Theorem 1: Given two tripletsxandx′, where ||x−x′||0≤
R, we can derive:
g(x′)−g(x)≤α·β·∆(8)
wherein:
α=1
(T
k)
β=f avg(q,M(x′,H))
∆=1−(T−R
k)
(T
k)(9)
5

where αrepresents randomly sampling positions of any k
words from a text of length Tfor the purpose of a masking
operation. βrepresents the average relevance score that the
ranker fgives to the masked text x′.∆represents the propor-
tion of all possible masking combinations that are excluded
when considering the differences between textsxand x′. It
denotes the overall fraction of masking combinations that are
affected by these differences.
Proof of Theorem 1: GivenH∼U(T,k), we can obtain:
g(x) =g(q,M(x,H))
g(x′) =g(q,M(x′,H))(10)
By subtracting g(x)from g(x′), we obtain:
g(q,M(x′,H))−g(q,M(x,H))
=E[s(q,M(x′,H))]−E[s(q,M(x,H))]
=RP(H)s(q,M(x′,H))−RP(H)s(q,M(x,H))
=RP(H)[s(q,M(x′,H))−s(q,M(x,H))](11)
whereP(H)represents the distribution ofH.
In considering whether there is overlap between the two
index setsx ⊖x′andH, we can transform the expression into:
RP(H)s(q,M(x′,H))
=RH∈{H|H∩(x⊖x′)̸=/0}P(H)s(q,M(x′,H))+
RH∈{H|H∩(x⊖x′)=/0}P(H)s(q,M(x′,H))(12)
It is noteworthy that when there is no overlap between the
two index setsx ⊖x′andH, this indicates that the mask opera-
tion has obscured all the words attacked by the attacker,xand
x′result in identical text after the mask operation. Therefore,
we have the following:
RH∈{H|H∩(x⊖x′)=/0}P(H)s(q,M(x′,H))
=RH∈{H|H∩(x⊖x′)=/0}P(H)s(q,M(x,H))(13)
Back into the main equation 11, we have:
RP(H)[s(q,M(x′,H))−s(q,M(x,H))]
=RH∈{H|H∩(x⊖x′)̸=/0}P(H)s(q,M(x′,H))−
RH∈{H|H∩(x⊖x′)̸=/0}P(H)s(q,M(x,H))
≤RH∈{H|H∩(x⊖x′)̸=/0}P(H)s(q,M(x′,H))(14)
In this context, {H|H∩( x⊖x′)̸=/0}signifies that the
intersection ofx ⊖x′andHis not an empty set, implying
that the masking operation has not completely obscured the
words attacked by the attacker. We denote the probability of
the intersection ofx⊖x′andHbeing non-empty by∆:∆=P(H∩(x⊖x′)̸=/0) = T−|x⊖x′|
k
 T
k= T−||x−x′||0
k
 T
k
where P(H) denotes the probability distribution of the mask:
P(H) =P(M(x,H)) =1 T
k=α
Further reasoning is as follows:
g(q,M(x′,H))−g(q,M(x,H))
=RP(H)[s(q,M(x′,H))−s(q,M(x,H))]
=RH∈{H|H∩(x⊖x′)̸=/0}P(H)s(q,M(x′,H))−
RH∈{H|H∩(x⊖x′)̸=/0}P(H)s(q,M(x,H))
≤RH∈{H|H∩(x⊖x′)̸=/0}P(H)s(q,M(x′,H))
≤P(H∩(x⊖x′)̸=/0)·P(H)·f avg(q,M(x′,H))
=α·β·∆(15)
Proposition 1.1: Given a ranked list Lqwith respect to a
query qand the candidate set of adversarial documents as Sx,
according to Theorem 1, if
g(q,M(x k,H))−g(q,M(x k+1,H))−α·β·∆≥0 (16)
We can certify that the ranking of x′is lower than that of
the k-th document , for allx∈L q[K+1 :]and anyx′∈S x:
max
x∈L q[K+1:]max
x′∈Sxg(q,M(x′,H))≤g(q,M(x k,H)) (17)
Proof of Proposition 1.1: We use δLto denote the differ-
ence between the relevance score of the K-th document in the
ranked list Lqand the maximum possible relevance score of
the document subject to word replacement attack. According
to Theorem 1, it can be inferred that:
δL=g(q,M(x k,H))−max
x∈L q[K+1:]max
x′∈Sxg(q,M(x′,H))
Based on Eq 8, we have:
g(q,M(x′,H))≤g(q,M(x,H))+α·β·∆
Accordingly, we can transformδ Linto:
δL≥g(q,M(x k,H))−max
x∈L q[K+1:]max
x′∈Sd[g(q,M(x,H))+α·β·∆]
=g(q,M(x k,H))−max
x∈L q[K+1:][g(q,M(x,H))+α·β·∆]
=g(q,M(x k,H))−[g(q,M(x k+1,H))+α·β·∆]
=g(q,M(x k,H))−g(q,M(x k+1,H))−α·β·∆
6

So if
g(q,M(x′,H))−g(q,M(x k,H))−α·β·∆>0 (18)
Then for for allx∈L q[K+1 :]and anyx′∈S x,we have:
δL=g(q,M(x k,H))−max
x∈L q[K+1:]max
x′∈Sxg(q,M(x′,H))≥0
i.e., the ranking ofx′is always lower than that of the k-th
document.
Since the mask space can be extremely large, we are unable
to access the relevance score β=f avg(q,M(x′,H)) . There-
fore, we employ the Monte Carlo estimation method to esti-
mateβ, aiming to approximate it as closely as possible to the
true value. We estimate favg(q,M(x′,H)) andg(q,M(x,H))
using the following approach:
favg(q,M(x′,H)) =E H∼U(T,k) [f(q,M(x′,H))]
≈1
nn
∑
i=1f(q,M(x′
(i),H))
g(q,M(x′,H)) =E H∼U(T,k) [g(q,M(x′,H))]
≈1
nn
∑
i=1g(q,M(x′
(i),H))
According to the definition of β, its value must lie between
0 and 1 (being positive and less than 1). Consequently, by
inferring the inequality conditions for Proposition 1.1, our
approach can generate more stringent provable constraints
compared to the method employed by Levine and Feizi (2019),
who directly set βto 1. It is worth noting that the work by
Levine and Feizi was focusing on image classification tasks,
assuming that all inputs are of equal length and width (i.e.,
with a fixed number of pixels). However, for text, it is essen-
tial to account for the issue of variable length. Therefore, in
establishing the provable robustness for text ranking models
as defined in this paper, we define the ranking model, the
smoothed ranking model, and the values of βand∆based
on the masking rate ρ(i.e., the proportion of words that can
be masked), rather than fixing the number of perturbation
units as in previous works [27, 48]. In practical scenarios,
given xand the perturbation number R, one can attempt dif-
ferent masking rates ρto calculate the corresponding values
ofg(q,M(x,H)) ,β, and∆. We can use rradius to denote the
Certified Robustness Radius (CRR) on the samplex:
rradius =max
g(x)−g(x′)−α·β·∆≥0R/T
4.3 Estimation and Validation ofβ
The preceding text has defined and explained β, and here we
will discuss how to estimate it. Since βrepresents the averageAlgorithm 1:The Estimation ofβ
Input:queryq, documentx
Parameter:Input text lengthT, Number of reserved
wordsk, Number of differential wordsr, Number of
perturbed samples nr, Number of masked samples nk,
randomly smoothed rankerg
Output:Estimatedβ
1FunctionBetaEstimator(x,T,k,r,n r,nk,g):
2INIT:β_list←0
3A←sample n relements f romU(T,r)
4foreach a∈Ado
5B←sample n kelements f romU(T,k)
6foreach b∈Bdo
7ifa∩b= /0then
8B.delete(b)
9s g←based on ranker g and set B
10β_list.add(s g)
11β←average(β_list)
12returnβ
relevance score that the ranker ggives to the masked text
where there are intersections between differences xandx′,
we can utilize the Monte Carlo method to sample a large
number of elements from U(T,k) to estimate the value of β.
To simplify the notation, we denote the value of|x⊖x′|asr.
The algorithm, as described in Algorithm 1, begins by sam-
pling nrelements from the set U(T,r) . Each element, denoted
asa, is a set of coordinate indices indicating that the cor-
responding positions of words have been perturbed by the
attacker. For each a,nkelements are further sampled from
U(T,r) . Each of these elements, denoted as b, is also a set
of coordinate indices, but represents positions where words
have not been masked. Subsequently, elements for which the
intersection of aandbis an empty set are removed. Using
the remaining elements and the ranker g, it is possible to
approximate the computation ofβ.
As the value of rincreases, for any a, it becomes increas-
ingly likely to coincide with any randomly sampled b, and the
value of βwill approach g(x′). To observe the degree of prox-
imity between βandg(x′), we conducted a simple experiment
on the MSMARCO Passage DEV dataset. We randomly se-
lected 100 sampled triplets, set nr=500 andnk=1000 , and
used Jensen-Shannon divergence to measure the distributional
difference between βandg(x′). As shown in Figure 2, regard-
less of the masking ratio ρ, when the number of perturbed
tokens is sufficiently large, all Jensen-Shannon divergences
become extremely small (less than 1×10−5). Even when the
target ranking model was not further fine-tuned on randomly
masked ranking data, the Jensen-Shannon divergence condi-
tion still held. However, models not fine-tuned on randomly
masked ranking data exhibited smoother performance and
7

2 4 6 8 10
Perturbation Numbers0.00.51.01.52.02.53.0JS Divergence1e6
W/ Random Mask Training
Mask Rate 10%
Mask Rate 30%
Mask Rate 50%
Mask Rate 70%
Mask Rate 90%
2 4 6 8 10
Perturbation Numbers0.00.51.01.52.02.5JS Divergence1e6
W/o Random Mask Training
Mask Rate 10%
Mask Rate 30%
Mask Rate 50%
Mask Rate 70%
Mask Rate 90%Figure 2: The Jensen-Shannon divergence between βand
pc(x)calculated by applying different masking ratios (10%,
30%, 50%, 70%, and 90%) on the MSMARCO Passage DEV
dataset.
slightly larger estimation errors under lower masking ratios.
Therefore, based on the conclusions from the above valida-
tion experiments, we can approximate gusing βin subsequent
experiments, i.e.,β≈g(x′).
4.4 Practical Defense Method of Certified Ro-
bustness
Based on the theoretical analysis presented above, we now
propose a practically certifiable robust defense method for text
ranking models. To ensure that the smoothed ranking model
gto rank samples accurately and robustly, we first construct
samples with a masking rate of ρto train the base classifier
g. During each iteration of the training process, we sample a
mini-batch of data and apply random masking. Subsequently,
gradient descent is employed on the masked mini-batch of
text to train the classifier g.
Specifically, Algorithm 2 provides a method for estimating
g(x) using Monte Carlo sampling and proving the robustness
ofgaround x. Estimating the predicted value of the smoothed
ranking model grequires determining the expected value of its
predictions. This process is defined in the algorithm pseudo
code as the Predict() function, which randomly samples n
masked copies and inputs these copies into the original rank-
ing model f(as part of the classifier g), obtaining the average
of their predicted values as the prediction of the smoothed
model. In addition, it is also necessary to estimateβ.
As presented in the RelRankerCertify method in Algorithm
2, we gradually increase the number of perturbation words R
such that R/T steadily rises from 0 to 1. We randomly gener-
aten′masked copies of x(where n′is significantly greater than
n) and estimate the βvalue through function BetaEstimator ,
calculating the ∆value using equation (9). Subsequently, we
use the smoothed model to rank the query q, comparing the
relevance scores of the k-th and k+1 -th documents in the rank-
ing results. According to Proposition 1.1, this process contin-
ues until g(q,M(x k,H))−g(q,M(x k+1,H))−α·β·∆<0 ;
upon stopping, R/T is output as the maximum provable ro-
bustness radius of the model g for x.Algorithm 2:Model Prediction and Robustness Cer-
tification
Input:queryq, documentx
Parameter:Input text lengthT, Number of reserved
wordsk, Number of perturbed wordsR, Number of
masked copies for predictionn, Number of masked
copies for certificationn′, Number of perturbed
samplesn r, original ranker f
Output:Predicted relevance scoresand certifiable
robustness radiusr rate
1FunctionRelPredict(x,T,k,n,f):
2H←samplenelements fromU(T,k)
3rel_list←sample n relements f romU(T,r)
4foreach h∈Hdo
5x mask←M(x,h)
6s←f(x mask)
7rel_list←insert relevance scores
8 g(x)←calculate the mean ofrel_list
9return g(x),rel_list
10FunctionRelRankerCertify(x,T,k,n,n′,nr,f):
11g←RelPredict()
12forR←0to Tdo
13β←BetaEstimator(x,T,k,R,n r,n′,g)
14α←based on equation (9) and valueT,k
15∆←based on equation (9) and valueT,R,k
16ifg(q,M(x k,H))−g(q,M(x k+1,H))≥
α·β·∆then
17R←R+1
18else break
19returnr rate=R/T
5 Experiment Analysis
5.1 Experiment Setup
5.1.1 Parameters and Dataset
The experiments primarily utilize PyTorch and the Transform-
ers library by HuggingFace [47] to implement the methods
discussed herein and to conduct the corresponding experi-
ments.
The experimental parameters are configured as follows: the
maximum concatenated length of queries and candidate docu-
ments is set to 256, and the learning rate for model training is
selected from 1, 3, 5, 7 × 10−6. The batch size is set to 256. All
models are trained and evaluated on two high-performance
servers: Server 1 is equipped with four NVIDIA RTX 24GB
3090 GPUs and 128GB system memory. Server 2 is equipped
with four 80GB NVIDIA A100 GPUs and 521GB memory.
Our experiments primarily utilize the MS MARCO Pas-
sage dataset and the TREC DL 2019 dataset, both of which
8

are widely used in the field of information retrieval. The
MS MARCO Passage dataset, part of the Machine Reading
Comprehension (MS MARCO) dataset, is constructed from
samples of real user queries on Bing. Initially, this dataset
was formed by retrieving the top 10 documents from the Bing
search engine and annotating them. The relevance labels in
this dataset are determined using a sparse judgment method,
assigning labels based on whether the answer to the query
appears in the current document, with each document being
segmented into passages. The full training set comprises ap-
proximately four million query tuples, each consisting of a
query, relevant or non-relevant document passages, and their
labels. The MS MARCO DEV validation set is primarily used
for re-ranking document passages. It includes 6,980 queries,
along with the top 1,000 document passages retrieved using
BM25 from the MS MARCO corpus. The experiments in this
paper employ BM25 retrieval results based on the open-source
Anserini library. The average length of passages included in
the dataset is 58 words. The TREC DL 2019 dataset is a pas-
sage ranking task from the 2019 TREC Deep Learning Track
dataset [9]. It shares the same source of candidate documents
as the MS MARCO dataset, both derived from Bing search
queries and document collections. The difference between it
and the MS MARCO dataset lies in the evaluation set orga-
nized by TREC DL, which provides 200 distinct new queries,
among which 43 queries are furnished with four-level rele-
vance labels. These labels are carefully determined by profes-
sional assessors employed by NIST through detailed manual
evaluation. For queries with multiple relevant passages, four
levels of graded relevance labels are assigned: fully relevant
(3), highly relevant (2), relevant (1), and not relevant (0).
5.1.2 Evaluation Metrics
In terms of basic evaluation metrics, we first assess the perfor-
mance of PairLM, following training with different random
masking strategy parameters, on the “clean” and non-attacked
MSMARCO dataset and TREC DL 2019 dataset, in compari-
son to the BM25 and TK models. This evaluation employs the
widely used metrics MRR@10 and NDCG@10 in informa-
tion retrieval tasks. MRR@10 is used to evaluate the ability
of the target model to place the first correct answer as close
to the top of the ranking as possible. This is specifically done
by calculating the mean of the reciprocal rank across all the
queries, where the reciprocal rank is derived from the rank of
the first correct answer. A higher MRR@10 value indicates
better system performance. On the other hand, NDCG@10 is
motivated by the notion that for a search user, highly relevant
documents are more valuable compared to marginally rele-
vant ones. It considers the overall ranking performance of the
top 10 candidate answers, as opposed to MRR, which only
considers the first correct answer. A larger NDCG@10 value
signifies better overall performance of the system within the
top 10 answers.To evaluate the robustness of information retrieval models
against word substitution attacks, we refer to the Certified Ro-
bust Query rate (CRQ) proposed by Wu et al. [48]. A query
is considered certifiably robust if none of the documents po-
sitioned outside the top K are attacked such that they move
into the top K positions. Precisely assessing this metric would
require enumerating all documents and exponential perturba-
tions, which is computationally prohibitive. As an alternative,
CRQ can be evaluated under the condition of randomized
smoothing. A higher value of this metric indicates better ro-
bustness performance.
Also, to further evaluate the defensive capability of Ro-
bustMask, we also employ two metrics during the assess-
ment: Mean Certified Robustness (MCR) and Mean Certified
Robustness Radius (MCRR). MCR represents the average
certified robustness of RobustMask across all queries, the cer-
tified robustness of a query qdenotes the maximum Rwhen
||x′−x|| 0=R. MCRR is defined as the average quotient of
the certified robustness for each qdivided by the length of the
target document. Higher values of MCR and MCRR indicate
a stronger certified robustness enhancement of RobustMask
for retrieval models.
5.1.3 Baseline
In the experimental section, we adopt four typical baseline
models to analyze the inherent robustness of neural text rank-
ing models and to evaluate the effectiveness of defense meth-
ods in enhancing ranking robustness. These four models are
highly representative. Our selection includes classic neural
ranking architectures (such as BERT ranker) and dense re-
trieval models (such as BGE). Specifically, we select the
Dense Retrieval model BGE (BAAI General Embeddings) as
a validation baseline, as it is a widely deployed mainstream
model in practice. Although current decoder-only, LLM-based
retrieval models demonstrate significant advantages in abso-
lute relevance judgment, their latency and deployment costs
mean that, at present, they are not as widely used as main-
stream dense retrieval models.
Our choice aligns with current industry trends, where
BERT-based architectures still dominate most text ranking
scenarios. The main focus of this paper is to improve the ro-
bustness of text ranking models, rather than to pursue absolute
state-of-the-art performance in retrieval ranking. Meanwhile,
the proposed methods can be transferred to other models,
which can be further explored in future work. A brief intro-
duction of these four methods is provided below:
(1) BM25 is a classic and widely used term matching algo-
rithm in the field of information retrieval. It is based on the
concepts of Term Frequency (TF) and Inverse Document Fre-
quency (IDF), along with normalization of document length,
allowing for precise matching between query terms and doc-
uments. The strength of BM25 lies in its efficiency and ac-
curacy, especially in the first stage of a two-stage retrieval
9

paradigm (namely the recall phase), where it performs ex-
ceptionally well. In our experiments, we utilized parameters
specifically fine-tuned for the MS MARCO DEV data and
used official experimental results as the benchmark perfor-
mance for BM25.
(2) TK is a relatively unique information retrieval model,
distinguished by its design philosophy, which does not rely
on BERT pre-trained models. Instead, it employs a shallow
Transformer encoder and the pre-trained word embeddings,
diverging from the pre-training paradigm that a significant
amount of research currently explores. This design enables
TK to maintain model performance while reducing complex-
ity and improving efficiency. The introduction of a Trans-
former encoder allows TK to handle more complex word
sequence relations, while the use of word embeddings pro-
vides TK with an initial level of semantic understanding from
the early stages of model training. In the experiments of com-
paring baseline models’ performance, we directly adopted
the reported performance of the TK model from the work of
Hofstätter et al. [21] as the benchmark performance for TK.
(3) In the previous sections, PairLM has been introduced
as a pairwise ranking model utilizing the concatenation ap-
proach, which currently achieves the best re-ranking perfor-
mance. In this method, the query and a pair of candidates are
concatenated using [SEP] and [CLS] tokens, respectively. Es-
sentially, a shared parameter LM encoder is trained on these
two concatenated inputs. The relevance scores are addition-
ally obtained via a trainable linear layer, allowing for the
acquisition of both positive and negative relevance scores.
During the training phase, gradients are calculated through
back propagation, and iterative optimization is conducted us-
ing the Adam optimizer. In the inference phase, the relevance
score for a document corresponding to a query is derived by
selecting the final dimensional output of the relevance scorer.
(4) BGE is a previous state-of-the-art embedding model.
The BGE model is initially pre-trained on large-scale text
corpora using MAE-style learning framework, followed by
contrastive learning optimization and task-specific fine-tuning
to enhance its semantic representation capabilities. This
hierarchical training paradigm establishes BGE as a high-
performance embedding model, achieving state-of-the-art re-
sults in semantic similarity tasks. Consequently, BGE remains
extensively employed as a core retrieval component in indus-
trial and research applications, including search engines, open-
domain question answering systems, and retrieval-augmented
generation pipelines for LLMs. In the experiment, we adopt
the BGE-M3 model from FlagEmbedding as one of the base-
line models.
5.2 Analysis and Discussion
To validate the effectiveness of RobustMask for robustness
enhancement in information retrieval models, several research
questions are addressed in detail: (1) How does the perfor-Table 1: The comparison of conventional ranking performance
between the original ranking model and various smoothed
ranking models on the MS-MARCO Passage DEV dataset
and the TREC DL 2019 dataset.
MethodMSMARCO DEV TREC DL 2019
MRR@10 NDCG@10 MRR@10 NDCG@10
BM25 18.7 23.4 68.5 48.7
TK 33.1 38.4 75.1 65.2
BERT-Base 35.2 41.5 87.1 71.0
BGE - - 85.9 67.9
BERT-Base+PGD - - 82.6 66.8
PairLM (BERT-Base) 34.3 40.4 85.7 71.2
+CertDR w/o Data Augments 19.5 23.5 66.7 54.2
+CertDR 31.9 36.9 77.4 66.1
+Random Mask 30% Ranker 34.0 40.2 84.5 71.7
+Random Mask 60% Ranker 34.1 40.3 83.6 71.5
mance of the ranking model with random masking smoothing
compare to the original model? (2) How is the Top-K certified
robustness at different mask ratios? (3) How does the certifi-
able robustness enhancement method, RobustMask, perform
in defending against actual information manipulation attacks
compared to other methods?
5.2.1 RQ1: How does the performance of the ranking
model with random masking smoothing compare
to the original model?
The performance metrics for BM25, TK, BGE and BERT-
Base presented in Table 1 are derived from the replication of
results from corresponding open-source projects, or reported
outcomes in existing literature, or our own experiment re-
sults. The original performance of PairLM (BERT-Base) was
obtained by training the PairLM model on the official MS-
MARCO triplet training set. The performance degradation of
the original PairLM (BERT-Base) compared to BERT-Base
can be attributed to two primary factors: overfitting to the
triplet training set and differences in the optimization tech-
niques and number of GPU machines utilized during the train-
ing process of the BERT-Base ranking model. Furthermore,
we also present the performance of BERT-Base following
the application of adversarial training via Projected Gradient
Descent (PGD) with continuous embedding perturbation.
The focus of this research is not on optimizing performance
on clean datasets where performance is considered acceptable,
but rather on examining the efficacy of robustness enhance-
ment methods. The focus of this research is not on optimizing
performance on clean datasets where fundamental ranking
performance needs only be acceptable, but rather on examin-
ing the efficacy of robustness enhancement methods. Specifi-
cally, this investigation aims to assess the improvements in
the defense and resilience of the target ranking models against
10

10% 20% 30%
Mask Rate01020304050607080MRR@10 BERT-Base
PairLM
Num=50
Num=100
Num=200Figure 3: The experimental results comparing the perfor-
mance of smoothed ranking models on the clean TREC DL
2019 dataset under different random masking rates and vary-
ing numbers of ensemble samples.
information manipulation attacks.
In typical datasets, these [MASK] tokens are not usually
present. Therefore, evaluating smoothed models using clean
data without [MASK] tokens may have a slight impact on the
results. The discrepancy between the data used during evalua-
tion and that employed in training may lead to slight variations
in the model performance. As demonstrated in Table 1, the
model trained with a 0.6 masking ratio, when compared to the
PairLM trained on the normal dataset, exhibited a decrease
of 0.2 percentage points in MRR@10 and a decrease of 0.1
percentage points in NDCG@10 on the MSMARCO DEV
dataset. On the TREC DL 2019 dataset, the MRR@10 de-
creased by 1.2 percentage points, while NDCG@10 increased
by 0.3 percentage points. The performance degradation is
primarily due to inconsistencies between training and evalu-
ation processes. Since normal datasets do not contain Mask
tokens, this can slightly affect the model’s performance on
clean data. However, overall, our method still achieves perfor-
mance levels comparable to the original target model, PairLM
(BERT-Base). Compared with CertDR [48], a certified ro-
bustness method based on word substitution for constructing
smoothed models, RobustMask exhibits distinct comparative
advantages. In the case of CertDR and its data augmentation-
based variant, despite being based on a strong assumption
of knowing the substitution attack vocabulary used by the
information manipulation attacker, this word substitution ap-
proach still significantly impairs the performance of the target
model. It requires more normal data to enhance its semantic
understanding capabilities in conventional ranking tasks. This
result also demonstrates that our RobustMask method min-
imally diminishes the model’s abilities in normal scenarios,
maintaining robust performance even when faced with thechallenge of training and evaluation inconsistencies.
Due to the large number of queries in the MSMARCO DEV
data, which results in extended testing times when combined
with the ensemble numbers, the experimental comparison of
the performance of smoothed ranking models under different
random masking rates and varying ensemble sample sizes
was conducted solely on the TREC DL 2019 dataset. Figure
3 reveals that the performance of the BERT-Base smoothed
ranking model shows a minor decline compared to the non-
random masking integrated PairLM (BERT-Base) in most
cases. Specifically, with a masking rate of 20%, the MRR@10
decreases less significantly, indicating that random smoothing
has minimal impact on top-ranked documents. In addition,
since the employed neural ranking model is lightweight, Ro-
bustMask has low overhead with short runtime and its mem-
ory consumption and time cost can be further optimized by
GPU-based parallel acceleration.
Figure 3 evaluates three sample sizes for random mask-
ing integration, namely Num=50, Num=100, Num=200. The
results indicate that MRR@10 varies only slightly across
these settings, suggesting that once a statistically represen-
tative number of samples is integrated, the overall ranking
performance remains largely stable.
5.2.2 RQ2: How is the Top-K certified robustness at dif-
ferent mask ratios?
We conducted experiments on the MSMARCO DEV and
TREC DL 2019 datasets to evaluate the Top-K (K=1, 3, 5, 10)
Certified Robust Query rate (CRQ) under different random
masking rates. Additionally, we measured the corresponding
Mean Certified Robustness (MCR) and Mean Certified Ro-
bustness Radius (MCRR). The Top-K Certifiable Robustness
for a given query Qrefers to the maximum radius Rthat the
smoothed relevance judgment model gcan provide for the
triplet xcomposed of the query, the K-th document, and the
K+1-th document. Here, x′represents an adversarial example,
and for any ||x−x′||0≤R, the model can produce a correct
relative relevance judgment.
From Figures 4 and 5, it can be observed that with the
increase in the mask ratio, there is a decrease in the Certi-
fied Robust Query (CRQ) ratio on the MSMARCO dataset.
This indicates that the increase in masks complicates the dif-
ficulty of relevance determination, leading to a reduction in
statistically significant and reliable relevance assessments.
Consequently, the model is more likely to refuse predictions,
resulting in a higher frequency of “unverifiable” judgments.
However, concurrently, as the certifiable robust perturbation
radius increases, more adversarial modifications are neces-
sary to successfully attack a ranking model, thereby incurring
greater costs. More results on TREC DL 2019 dataset are
depicted Figure 6 and Figure 7 at appendix.
Figures 6 and 7 present the corresponding results on the
TREC DL 2019 dataset. Overall, the observed trends are
11

102030405060708090
Mask Rate0102030405060708090CRQ
T op1
CRQ
102030405060708090
Mask Rate0102030405060708090CRQ
T op3
CRQ
102030405060708090
Mask Rate0102030405060708090CRQ
T op5
CRQ
102030405060708090
Mask Rate0102030405060708090CRQ
T op10
CRQ
0123456789
MCR
MCR
0123456789
MCR
MCR
0123456789
MCR
MCR
0123456789
MCR
MCRFigure 4: Top-K CRQ and the corresponding MCR under different Mask rates on the MSMARCO dataset.
102030405060708090
Mask Rate0102030405060708090CRQ
T op1
CRQ
102030405060708090
Mask Rate0102030405060708090CRQ
T op3
CRQ
102030405060708090
Mask Rate0102030405060708090CRQ
T op5
CRQ
102030405060708090
Mask Rate0102030405060708090CRQ
T op10
CRQ
024681012141618
MCRR
MCRR
024681012141618
MCRR
MCRR
024681012141618
MCRR
MCRR
024681012141618
MCRR
MCRR
Figure 5: Top-K CRQ and the corresponding MCRR under different Mask rates on the MSMARCO dataset.
consistent with those on MSMARCO. As the mask ratio in-
creases, the Certified Robust Query (CRQ) ratio declines,
leading to more frequent “unverifiable” judgments, while
the certifiable robust perturbation radius expands, implying
higher adversarial costs. The main difference lies in the spe-
cific numerical values, which vary due to the characteristics
of the TREC DL 2019 dataset.
5.2.3 RQ3: How does the certifiable robustness enhance-
ment method, RobustMask, perform in defending
against actual information manipulation attacks
compared to other methods?
For RQ3, we focus on BERT and BGE as the primary target
ranking models to examine different robustness enhancement
baseline methods and ranking attack methods: (1) The origi-
nal target ranking model (BERT and BGE) serves mainly as
a baseline to assess the ability of the original model to with-
stand information manipulation attacks in the absence of any
defense methods; (2) Models (BERT only) enhanced by ro-
bustness enhancement methods based on adversarial training,
such as PGD adversarial training (PGD-ADV) and adversar-
ial training based on PAT samples (PAT-ADV), which cor-
respond to continuous perturbation adversarial training and
discrete perturbation adversarial training, respectively; (3)
The proposed certifiable robustness enhancement method, Ro-
bustMask, based on random masking for text ranking models,primarily combines the innate characteristics of pre-trained
models to mitigate the effects of various adversarial attack
perturbations through robust random mask smoothing.
The actual information manipulation attack methods pri-
marily include keyword stuffing (Query+), adversarial seman-
tic collision (Collision_nat), Pairwise Anchor Trigger(PAT)
attack methods, PRADA, among other attack methodologies.
It is important to note that for information manipulation at-
tacks based on trigger insertion, such as keyword stuffing and
adversarial semantic collision, adjustments were made in the
experiments: the generated triggers were inserted at the begin-
ning of candidate documents, with equivalent lengths of text
being truncated from the end to achieve a “word replacement”
attack transformation. The hyperparameters such as trigger
length and search space follow the settings of the experiments
discussed in earlier sections, with adversarial perturbation
content not exceeding 5%. The experimental results on TREC
DL 2019 are shown in Tables 3 and 4.
It can be observed that if no defensive measures are taken
to enhance the robustness of the base model(BERT-Base and
BGE), the success rate of information manipulation adversar-
ial attacks is remarkably high. For instance, on the TREC DL
2019 dataset, the attack success rate of Query+ against BERT-
base reached 90%. This underscores the necessity of propos-
ing appropriate defense methods. Although empirically-based
adversarial training defense methods reduce the success rate
of empirical attacks to some extent, their performance does
12

102030405060708090
Mask Rate0102030405060708090CRQ
T op1
CRQ
102030405060708090
Mask Rate0102030405060708090CRQ
T op3
CRQ
102030405060708090
Mask Rate0102030405060708090CRQ
T op5
CRQ
102030405060708090
Mask Rate0102030405060708090CRQ
T op10
CRQ
0123456789
MCR
MCR
0123456789
MCR
MCR
0123456789
MCR
MCR
0123456789
MCR
MCRFigure 6: Top-K CRQ and the corresponding MCR under different mask rates on the TREC DL 2019 dataset.
102030405060708090
Mask Rate0102030405060708090CRQ
T op1
CRQ
102030405060708090
Mask Rate0102030405060708090CRQ
T op3
CRQ
102030405060708090
Mask Rate0102030405060708090CRQ
T op5
CRQ
102030405060708090
Mask Rate0102030405060708090CRQ
T op10
CRQ
024681012141618
MCRR
MCRR
024681012141618
MCRR
MCRR
024681012141618
MCRR
MCRR
024681012141618
MCRR
MCRR
Figure 7: Top-K CRQ and the corresponding MCRR under different mask rates on the TREC DL 2019 dataset.
Table 2: The results of the Top-10 defense success rate (%)
against empirical attacks using robustness enhancement meth-
ods on the TREC DL 2019 dataset.
Method Query+ Collision_nat PAT PRADA
BERT-base 92.9 51.2 29.3 78.6
BGE 97.6 68.3 31.7 35.7
PGD-ADV 73.8 75.6 31.7 38.1
PAT-ADV 57.1 43.9 26.8 31.0
RobustMask-30% (BERT) 64.3 31.7 12.2 38.1
RobustMask-60% (BERT) 33.3 22.0 14.6 33.3
RobustMask-90% (BERT) 11.9 9.8 9.8 23.8
RobustMask-30% (BGE) 83.3 36.9 4.87 21.9
RobustMask-60% (BGE) 14.3 4.9 2.4 2.4
RobustMask-90% (BGE) 0.0 0.0 0.0 0.0
not match that of RobustMask. Therefore, merely increasing
training documents and adversarial samples is not a robust de-
fensive strategy against adversarial information manipulation
attacks in information retrieval tasks. Future robustness en-
hancement methods in the field of information retrieval should
explore empirical defense strategies that are more suitable for
retrieval ranking scenarios. However, empirical defense meth-
ods cannot provide strict certifiable robustness guarantees,and their performance may largely depend on the dataset and
the specific attack method. For example, for keyword stuffing
(Query+), a commonly used attack method in search engine
optimization, general end-to-end defense ranking models find
it challenging to defend effectively. The defense is typically
possible only by deploying empirically-based information
manipulation attack anomaly detectors explored previously.
In contrast, RobustMask enables end-to-end defense against
keyword stuffing for target ranking models. Moreover, Robust-
Mask can also effectively defend against other attack methods
based on gradients and important word replacements. Partic-
ularly, for PAT attacks, which are less effective at inverting
adjacent positional relevance, the attack success rate could be
low due to incorporating camouflage and contextual consis-
tency mechanisms, sacrificing some attack success rate. Our
proposed RobustMask can further reduce the attack success
rate of PAT, building on its already low success rate. This indi-
cates that RobustMask can theoretically certify the robustness
of ranking models and practically enhance their robustness.
Furthermore, from the tables of the two experimental re-
sults mentioned above, it can be observed that the defense
performance of RobustMask is relatively sensitive to the prob-
ability of random mask smoothing. The optimal masking prob-
ability varies among different methods, primarily because the
current certifiable methods have a small robustness radius,
and different attack methods result in varying adversarial
modifications. In practical application, some costs need to
13

Table 3: The results of the Top-20 defense success rate (%)
against empirical attacks using robustness enhancement meth-
ods on the TREC DL 2019 dataset.
Method Query+ Collision_nat PAT PRADA
BERT-base 95.2 70.7 24.4 85.7
BGE 97.6 53.6 24.4 30.9
PGD-ADV 88.1 68.3 22.0 64.3
PAT-ADV 71.4 51.2 17.7 50.0
RobustMask-30% (BERT) 81.0 43.9 9.8 28.6
RobustMask-60% (BERT) 57.1 41.5 26.8 23.8
RobustMask-90% (BERT) 38.1 26.8 24.4 21.4
RobustMask-30% (BGE) 83.3 26.8 7.3 7.3
RobustMask-60% (BGE) 33.3 7.3 2.4 0.0
RobustMask-90% (BGE) 2.4 0.0 0.0 0.0
be incurred to adjust this parameter. Future research could
explore defense methods with a larger certifiable robustness
radius, aiming to reduce method specificity and thereby en-
hance the robustness of target models more efficiently.
6 Conclusion
In this paper, we introduced RobustMask, a novel defense
approach designed to enhance and certify the robustness of
neural text ranking models against various adversarial per-
turbations at the character, word, and phrase levels. Leverag-
ing the intrinsic masking prediction capability of pretrained
language models with a strategically randomized smooth-
ing mechanism, RobustMask significantly mitigates potential
adversarial manipulation, effectively shielding retrieval sys-
tems from harm. We rigorously established theoretically prov-
able certified robustness guarantees by integrating pairwise
ranking comparisons and probabilistic statistical methods,
enabling formal verification of top-K ranking stability un-
der bounded adversarial perturbations. Empirical evaluations
also demonstrated its effectiveness, maintaining certified ro-
bustness for the top-10 and top-20 ranked candidates. These
results significantly surpass the capabilities of existing meth-
ods, highlighting the practical usability and substantial advan-
tage of our method in real-world retrieval and RAG scenarios.
Our study advances a critical step toward ensuring the secure
deployment of neural ranking models in adversarial environ-
ments, paving the way for trustworthy and stable information
retrieval systems and their downstream applications. In the
future, we will explore further refinement of masking strate-
gies and adaptive certification schemes, enhancing certified
robustness in more target models, dynamic scenarios, and
across diverse adversarial environments.References
[1]Anish Athalye, Nicholas Carlini, and David Wagner. Ob-
fuscated gradients give a false sense of security: Circum-
venting defenses to adversarial examples. InInterna-
tional conference on machine learning, pages 274–283.
PMLR, 2018.
[2]Amin Bigdeli, Negar Arabzadeh, Ebrahim Bagheri, and
Charles LA Clarke. Empra: Embedding perturba-
tion rank attack against neural ranking models.arXiv
preprint arXiv:2412.16382, 2024.
[3]Xuanang Chen, Ben He, Kai Hui, Le Sun, and Yingfei
Sun. Dealing with textual noise for robust and effective
bert re-ranking.Information Processing & Management,
60(1):103135, 2023.
[4]Xuanang Chen, Ben He, Le Sun, and Yingfei Sun. De-
fense of adversarial ranking attack in text retrieval:
Benchmark and baseline via detection.arXiv preprint
arXiv:2307.16816, 2023.
[5]Xuanang Chen, Ben He, Zheng Ye, Le Sun, and Yingfei
Sun. Towards imperceptible document manipulations
against neural ranking models. InFindings of the Asso-
ciation for Computational Linguistics: ACL 2023, pages
6648–6664, 2023.
[6]Zhuo Chen, Yuyang Gong, Jiawei Liu, Miaokun Chen,
Haotan Liu, Qikai Cheng, Fan Zhang, Wei Lu, and Xi-
aozhong Liu. Flippedrag: Black-box opinion manipula-
tion adversarial attacks to retrieval-augmented genera-
tion models. InProceedings of the 2025 ACM SIGSAC
Conference on Computer and Communications Security,
CCS ’25, page 4109–4123, New York, NY , USA, 2025.
Association for Computing Machinery.
[7]Zhuo Chen, Jiawei Liu, and Haotan Liu. Research on
the reliability and fairness of opinion retrieval in public
topics. In2024 Network and Distributed System Secu-
rity (NDSS) workshop on AI Systems with Confidential
Computing, 2024.
[8]Jeremy Cohen, Elan Rosenfeld, and Zico Kolter. Cer-
tified adversarial robustness via randomized smooth-
ing. InInternational Conference on Machine Learning,
pages 1310–1320. PMLR, 2019.
[9]Nick Craswell, Bhaskar Mitra, Emine Yilmaz, Daniel
Campos, and Ellen M V oorhees. Overview of
the trec 2019 deep learning track.arXiv preprint
arXiv:2003.07820, 2020.
[10] Zhuyun Dai and Jamie Callan. Deeper text understand-
ing for IR with contextual neural language modeling. In
Proceedings of the 42nd International ACM SIGIR Con-
ference on Research and Development in Information
14

Retrieval, SIGIR 2019, Paris, France, July 21-25, 2019,
pages 985–988. ACM, 2019.
[11] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and
Kristina Toutanova. Bert: Pre-training of deep bidirec-
tional transformers for language understanding. InPro-
ceedings of the 2019 Conference of the North American
Chapter of the Association for Computational Linguis-
tics: Human Language Technologies, Volume 1 (Long
and Short Papers), pages 4171–4186, 2019.
[12] Javid Ebrahimi, Anyi Rao, Daniel Lowd, and Dejing
Dou. HotFlip: White-box adversarial examples for text
classification. InProceedings of the 56th Annual Meet-
ing of the Association for Computational Linguistics
(Volume 2: Short Papers), pages 31–36, Melbourne, Aus-
tralia, 2018. Association for Computational Linguistics.
[13] Robert Epstein and Ronald E Robertson. The search
engine manipulation effect (seme) and its possible im-
pact on the outcomes of elections.Proceedings of the
National Academy of Sciences, 112(33):E4512–E4521,
2015.
[14] Luyu Gao, Zhuyun Dai, and Jamie Callan. COIL: Re-
visit exact lexical match in information retrieval with
contextualized inverted list. InProceedings of the 2021
Conference of the North American Chapter of the Associ-
ation for Computational Linguistics: Human Language
Technologies, pages 3030–3042, Online, 2021. Associa-
tion for Computational Linguistics.
[15] Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang Jia,
Jinliu Pan, Yuxi Bi, Yi Dai, Jiawei Sun, Haofen Wang,
and Haofen Wang. Retrieval-augmented generation
for large language models: A survey.arXiv preprint
arXiv:2312.10997, 2, 2023.
[16] Yuyang Gong, Zhuo Chen, Jiawei Liu, Miaokun Chen,
Fengchang Yu, Wei Lu, Xiaofeng Wang, and Xiaozhong
Liu. Topic-fliprag: Topic-orientated adversarial opinion
manipulation attacks to retrieval-augmented generation
models. InProceedings of the 34th USENIX Security
Symposium, pages 3807–3826, 2025.
[17] Ian J Goodfellow, Jonathon Shlens, and Christian
Szegedy. Explaining and harnessing adversarial exam-
ples.arXiv preprint arXiv:1412.6572, 2014.
[18] Gregory Goren, Oren Kurland, Moshe Tennenholtz, and
Fiana Raiber. Ranking robustness under adversarial
document manipulations. InThe 41st International
ACM SIGIR Conference on Research & Development in
Information Retrieval, pages 395–404, 2018.
[19] Gregory Goren, Oren Kurland, Moshe Tennenholtz, and
Fiana Raiber. Ranking-incentivized quality preservingcontent modification. InProceedings of the 43rd Inter-
national ACM SIGIR conference on research and devel-
opment in Information Retrieval, SIGIR 2020, Virtual
Event, China, July 25-30, 2020, pages 259–268. ACM,
2020.
[20] Jiafeng Guo, Yixing Fan, Qingyao Ai, and W. Bruce
Croft. A deep relevance matching model for ad-hoc
retrieval. InProceedings of the 25th ACM International
Conference on Information and Knowledge Manage-
ment, CIKM 2016, Indianapolis, IN, USA, October 24-
28, 2016, pages 55–64. ACM, 2016.
[21] Sebastian Hofstätter, Markus Zlabinger, and Allan Han-
bury. Interpretable & time-budget-constrained contextu-
alization for re-ranking. InECAI 2020, pages 513–520.
IOS Press, 2020.
[22] Po-Sen Huang, Xiaodong He, Jianfeng Gao, Li Deng,
Alex Acero, and Larry P. Heck. Learning deep structured
semantic models for web search using clickthrough data.
In22nd ACM International Conference on Information
and Knowledge Management, CIKM’13, San Francisco,
CA, USA, October 27 - November 1, 2013, pages 2333–
2338. ACM, 2013.
[23] Po-Sen Huang, Robert Stanforth, Johannes Welbl, Chris
Dyer, Dani Yogatama, Sven Gowal, Krishnamurthy Dvi-
jotham, and Pushmeet Kohli. Achieving verified robust-
ness to symbol substitutions via interval bound propa-
gation. InProceedings of the 2019 Conference on Em-
pirical Methods in Natural Language Processing and
the 9th International Joint Conference on Natural Lan-
guage Processing (EMNLP-IJCNLP), pages 4083–4093,
2019.
[24] Yuheng Huang, Lei Ma, and Yuanchun Li. Patchcen-
sor: Patch robustness certification for transformers via
exhaustive testing.ACM Transactions on Software En-
gineering and Methodology, 32(6):1–34, 2023.
[25] Robin Jia, Aditi Raghunathan, Kerem Göksel, and Percy
Liang. Certified robustness to adversarial word sub-
stitutions. InProceedings of the 2019 Conference on
Empirical Methods in Natural Language Processing and
the 9th International Joint Conference on Natural Lan-
guage Processing (EMNLP-IJCNLP), pages 4129–4142,
2019.
[26] Erik Jones, Robin Jia, Aditi Raghunathan, and Percy
Liang. Robust encodings: A framework for combating
adversarial typos. InProceedings of the 58th Annual
Meeting of the Association for Computational Linguis-
tics, pages 2752–2765, 2020.
[27] Alexander Levine and Soheil Feizi. Robustness cer-
tificates for sparse adversarial attacks by randomized
ablation, 2019.
15

[28] Zongyi Li, Jianhan Xu, Jiehang Zeng, Linyang Li, Xi-
aoqing Zheng, Qi Zhang, Kai-Wei Chang, and Cho-Jui
Hsieh. Searching for an effective defender: Benchmark-
ing defense against adversarial word substitution. In
Marie-Francine Moens, Xuanjing Huang, Lucia Specia,
and Scott Wen-tau Yih, editors,Proceedings of the 2021
Conference on Empirical Methods in Natural Language
Processing, pages 3137–3147, Online and Punta Cana,
Dominican Republic, November 2021. Association for
Computational Linguistics.
[29] Jiawei Liu, Yangyang Kang, Di Tang, Kaisong Song,
Changlong Sun, Xiaofeng Wang, Wei Lu, and Xi-
aozhong Liu. Order-disorder: Imitation adversarial at-
tacks for black-box neural ranking models. InProceed-
ings of the 2022 ACM SIGSAC Conference on Computer
and Communications Security, pages 2025–2039, 2022.
[30] Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du, Man-
dar Joshi, Danqi Chen, Omer Levy, Mike Lewis, Luke
Zettlemoyer, and Veselin Stoyanov. Roberta: A robustly
optimized bert pretraining approach.arXiv preprint
arXiv:1907.11692, 2019.
[31] Yu-An Liu, Ruqing Zhang, Jiafeng Guo, Maarten de Ri-
jke, Wei Chen, Yixing Fan, and Xueqi Cheng. Black-
box adversarial attacks against dense retrieval models: A
multi-view contrastive learning method. InProceedings
of the 32nd ACM International Conference on Informa-
tion and Knowledge Management, pages 1647–1656,
2023.
[32] Yongqiang Ma, Jiawei Liu, Fan Yi, Qikai Cheng, Yong
Huang, Wei Lu, and Xiaozhong Liu. Is this abstract gen-
erated by ai? a research for the gap between ai-generated
scientific text and human-written scientific text.arXiv
preprint arXiv:2301.10416, 2023.
[33] Tomás Mikolov, Ilya Sutskever, Kai Chen, Gregory S.
Corrado, and Jeffrey Dean. Distributed representations
of words and phrases and their compositionality. InAd-
vances in Neural Information Processing Systems 26:
27th Annual Conference on Neural Information Pro-
cessing Systems 2013. Proceedings of a meeting held
December 5-8, 2013, Lake Tahoe, Nevada, United States,
pages 3111–3119, 2013.
[34] Bhaskar Mitra, Fernando Diaz, and Nick Craswell.
Learning to match using local and distributed repre-
sentations of text for web search. InProceedings of
the 26th International Conference on World Wide Web,
WWW 2017, Perth, Australia, April 3-7, 2017, pages
1291–1299. ACM, 2017.
[35] Takeru Miyato, Andrew M Dai, and Ian Goodfellow.
Adversarial training methods for semi-supervised textclassification. InInternational Conference on Learning
Representations, 2017.
[36] Craswell Nick. Mean reciprocal rank.Encyclopedia of
database systems, 1703, 2009.
[37] Rodrigo Nogueira and Kyunghyun Cho. Passage re-
ranking with bert.ArXiv preprint, abs/1901.04085,
2019.
[38] Jeffrey Pennington, Richard Socher, and Christopher
Manning. GloVe: Global vectors for word representa-
tion. InProceedings of the 2014 Conference on Empiri-
cal Methods in Natural Language Processing (EMNLP),
pages 1532–1543, Doha, Qatar, 2014. Association for
Computational Linguistics.
[39] Ronak Pradeep, Rodrigo Nogueira, and Jimmy Lin.
The expando-mono-duo design pattern for text ranking
with pretrained sequence-to-sequence models.ArXiv
preprint, abs/2101.05667, 2021.
[40] Danish Pruthi, Bhuwan Dhingra, and Zachary C Lipton.
Combating adversarial misspellings with robust word
recognition. InProceedings of the 57th Annual Meeting
of the Association for Computational Linguistics, pages
5582–5591, 2019.
[41] Nisarg Raval and Manisha Verma. One word at a time:
adversarial attacks on retrieval models.ArXiv preprint,
abs/2008.02197, 2020.
[42] Stephen E Robertson and Steve Walker. Some simple
effective approximations to the 2-poisson model for
probabilistic weighted retrieval. InSIGIR’94, pages
232–241. Springer, 1994.
[43] Victoria L Rubin and Yimin Chen. Information manipu-
lation classification theory for lis and nlp.Proceedings
of the American Society for Information Science and
Technology, 49(1):1–5, 2012.
[44] Yelong Shen, Xiaodong He, Jianfeng Gao, Li Deng,
and Grégoire Mesnil. A latent semantic model with
convolutional-pooling structure for information retrieval.
InProceedings of the 23rd ACM International Confer-
ence on Conference on Information and Knowledge
Management, CIKM 2014, Shanghai, China, November
3-7, 2014, pages 101–110. ACM, 2014.
[45] Xiang Shi, Jiawei Liu, Yinpeng Liu, Qikai Cheng, and
Wei Lu. Know where to go: Make llm a relevant, re-
sponsible, and trustworthy searchers.Decision Support
Systems, 188:114354, 2025.
[46] Congzheng Song, Alexander M Rush, and Vitaly
Shmatikov. Adversarial semantic collisions. InPro-
ceedings of the 2020 Conference on Empirical Methods
16

in Natural Language Processing (EMNLP), pages 4198–
4210, 2020.
[47] T Wolf. Huggingface’s transformers: State-of-the-
art natural language processing.arXiv preprint
arXiv:1910.03771, 2019.
[48] Chen Wu, Ruqing Zhang, Jiafeng Guo, Wei Chen, Yix-
ing Fan, Maarten de Rijke, and Xueqi Cheng. Certified
robustness to word substitution ranking attack for neu-
ral ranking models. InProceedings of the 31st ACM
International Conference on Information & Knowledge
Management, pages 2128–2137, 2022.
[49] Chen Wu, Ruqing Zhang, Jiafeng Guo, Maarten De Ri-
jke, Yixing Fan, and Xueqi Cheng. Prada: Practical
black-box adversarial attacks against neural ranking
models.ACM Transactions on Information Systems,
41(4):1–27, 2023.
[50] Fen Xia, Tie-Yan Liu, and Hang Li. Statistical consis-
tency of top-k ranking.Advances in Neural Information
Processing Systems, 22, 2009.
[51] Chong Xiang, Arjun Nitin Bhagoji, Vikash Sehwag, and
Prateek Mittal. {PatchGuard }: A provably robust de-
fense against adversarial patches via small receptive
fields and masking. In30th USENIX Security Sympo-
sium (USENIX Security 21), pages 2237–2254, 2021.
[52] Chong Xiang and Prateek Mittal. Patchguard++: Ef-
ficient provable attack detection against adversarial
patches.arXiv preprint arXiv:2104.12609, 2021.
[53] Shicheng Xu, Liang Pang, Huawei Shen, Xueqi Cheng,
and Tat-Seng Chua. Search-in-the-chain: Interac-
tively enhancing large language models with search for
knowledge-intensive tasks. InProceedings of the ACM
Web Conference 2024, pages 1362–1373, 2024.
[54] Mao Ye, Chengyue Gong, and Qiang Liu. Safer:
A structure-free approach for certified robustness
to adversarial word substitutions.arXiv preprint
arXiv:2005.14424, 2020.
[55] Jiehang Zeng, Jianhan Xu, Xiaoqing Zheng, and Xu-
anjing Huang. Certified robustness to text adversarial
attacks by randomized [mask].Computational Linguis-
tics, 49(2):395–427, 2023.
[56] Xinyu Zhang, Hanbin Hong, Yuan Hong, Peng Huang,
Binghui Wang, Zhongjie Ba, and Kui Ren. Text-crs:
A generalized certified robustness framework against
textual adversarial attacks. In2024 IEEE Symposium
on Security and Privacy (SP), pages 2920–2938. IEEE,
2024.[57] Bin Zhou and Jian Pei. Osd: An online web spam de-
tection system. InIn Proceedings of the 15th ACM
SIGKDD International Conference on Knowledge Dis-
covery and Data Mining, KDD, volume 9, 2009.
[58] Mo Zhou, Le Wang, Zhenxing Niu, Qilin Zhang, Nan-
ning Zheng, and Gang Hua. Adversarial attack and
defense in deep ranking.IEEE Transactions on Pattern
Analysis and Machine Intelligence, 2024.
17