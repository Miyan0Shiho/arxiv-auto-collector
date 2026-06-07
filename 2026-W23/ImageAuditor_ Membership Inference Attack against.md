# ImageAuditor: Membership Inference Attack against Image-based Retrieval-Augmented Generation

**Authors**: Jinghuai Zhang, Pengyue Yu, Zhexiao Lin, Kunlin Cai, Fnu Suya, Yuan Tian

**Published**: 2026-06-02 09:03:56

**PDF URL**: [https://arxiv.org/pdf/2606.03354v1](https://arxiv.org/pdf/2606.03354v1)

## Abstract
Image-based Retrieval-Augmented Generation (IRAG) conditions a frozen generator on reference images retrieved from an external database, supporting both text-to-image (T2I) and question answering (Q&A) tasks. Because these databases are opaque and web-scraped, copyright holders need ways to audit whether specific images appear in them. While prior work employs membership inference attacks (MIAs) to audit uni-modal, text-based RAG, they fail to transfer to IRAG due to two key challenges. First, cross-modal retrieval: text-RAG MIAs force retrieval of the target passage by injecting its content into the query, which is unavailable in IRAG since images cannot be embedded into text queries; even accurate image captions fail to bridge the modality gap. Second, discriminative signal extraction: text-RAG MIAs extract membership signals by prompting the generator to answer multiple questions over the target passage, whereas T2I generators in IRAG produce images rather than follow Q&A commands. To fill this gap, we introduce the first MIA tailored to IRAG, ImageAuditor, which decomposes each attack query into a retrieval segment and an extraction segment, enabling dedicated optimization for each challenge. For retrieval, we propose Reward-Guided Policy Optimization (RGPO), which updates a stochastic policy from reward-ranked candidates to navigate the cross-modal embedding landscape and admits finite-sample optimality guarantees to balance exploration and exploitation. For extraction, we analyze the distribution of the MIA score to guide the co-design of the prompting strategy and scoring rule, and derive task-specific instantiations for T2I and Q&A tasks. We aggregate signals across queries via K-means clustering for reliable membership decisions. Across various IRAG systems, ImageAuditor exceeds 80% AUROC with only four queries per audited image and remains robust across diverse settings.

## Full Text


<!-- PDF content starts -->

ImageAuditor: Membership Inference Attack against
Image-based Retrieval-Augmented Generation
Jinghuai Zhang1Pengyue Yu1Zhexiao Lin2Kunlin Cai1
Fnu Suya3Yuan Tian1
1UCLA2University of California, Berkeley3University of Tennessee, Knoxville
{jinghuai1998,pengyue2024,kunlin96,yuant}@g.ucla.edu
zhexiaolin@berkeley.edu fsuya@utk.edu
Abstract
Image-based Retrieval-Augmented Generation (IRAG) conditions a frozen gen-
erator on reference images retrieved from an external database, supporting both
text-to-image (T2I) and question answering (Q&A) tasks. Because these databases
are opaque and typically web-scraped, copyright holders need ways to audit whether
specific images appear in them. While prior work employs membership inference
attacks (MIAs) to audit uni-modal, text-based RAG, they fail to transfer to IRAG
due totwofundamental challenges. First,cross-modal retrieval: text-RAG MIAs
force retrieval of the target passage by injecting its content into the query, which is
unavailable in IRAG since images cannot be embedded into text queries; even accu-
rate image captions fail to bridge the modality gap. Second,discriminative signal
extraction: text-RAG MIAs extract membership signals by prompting the generator
to answer multiple questions over the target passage, whereas T2I generators in
IRAG produce images rather than follow Q&A commands. To fill this gap, we
introduce the first MIA tailored to IRAG,ImageAuditor, which decomposes each
attack query into a retrieval segment and an extraction segment, enabling dedicated
optimization for each challenge. For retrieval, we propose Reward-Guided Policy
Optimization (RGPO), which updates a stochastic policy from reward-ranked can-
didates to navigate the cross-modal embedding landscape and admits finite-sample
optimality guarantees to balance exploration and exploitation. For extraction, we
analyze the distribution of the MIA score to guide the co-design of the prompting
strategy and scoring rule, and derive task-specific instantiations for T2I and Q&A
tasks. We aggregate signals across queries via K-means clustering for reliable
membership decisions. Across IRAG systems built on SDXL, SD1.5, Kandinsky,
LLaV A-1.6, and Qwen2.5-VL, ImageAuditor exceeds 80% AUROC with only four
queries per audited image and remains robust across diverse settings.
1 Introduction
Image-based Retrieval-Augmented Generation (IRAG) [ 22,6] grounds generative models in external
visual knowledge by retrieving reference images from an image database. It has gained increasing
popularity in two major multimodal generation tasks (Figure 1): text-to-image generation (T2I) [ 22,
31,5], where retrieved images supply concepts, identities, and styles beyond the reach of base
diffusion models [ 26]; and question answering (Q&A) [ 6,4,18], where retrieved images provide
knowledge-dependent visual evidence for vision-language models [ 37]. For consistency, this paper
focuses on IRAG systems that take text as input and produce either images (T2I) or text (Q&A).
Preprint.arXiv:2606.03354v1  [cs.CR]  2 Jun 2026

This rapid adoption also exposes a privacy and copyright surface [ 21,3]. Because these databases are
typically web-crawled with poorly documented provenance, copyright holders (e.g., artists), dataset
owners, and privacy regulators have no reliable way to verify whether their images have been silently
incorporated into a deployed system, motivating the need for principled auditing tools.
Membership inference attacks (MIA) [ 32], which infer whether a record belongs to a target system,
provide a natural auditing primitive. Traditional MIAs [ 32,1] target a model’s training data by
exploiting memorization signals in its weights, and therefore do not apply when membership resides
in an external database. Recent text-RAG MIAs [ 23,16,20] probe the database via question–answer
interactions, but they implicitly assumesingle-modal retrieval, where the attacker can inject the target
document into the query to force retrieval, andtext-format outputs, where responses can be matched
against pre-computed answers. Both assumptions break down in IRAG settings. A text query cannot
carry an image target, and we show in Section 4.2 that even substituting a strong caption fails to
bridge the modality gap. Furthermore, IRAG for T2I generation emits images rather than text and
does not explicitly answer the query, leaving Q&A-style probing without a signal to match. We
formalize these obstacles astwo fundamental challengesin Section 3.1.
Figure 1: IRAG employs an external im-
age database to generate faithful images
(top) and produce reliable answers to
knowledge-dependent queries (bottom).ContributionsTo fill this gap, we introduceImageAudi-
tor, the first MIA tailored to IRAG, operating in a realistic
setting with only a handful of attack queries per audited
image (Section 2.2). Our key design decomposes each
query into two functionally isolated segments: aretrieval
segmentthat brings the test image into the generator’s con-
text when present, and anextraction segmentthat elicits a
discriminative MIA signal from the resulting output. This
decomposition allows each segment to be explicitly opti-
mized against a clean and dedicated objective, avoiding
conflicts inherent in coupled optimization.
For the retrieval segment, a natural starting point is the
suffix-optimization toolkit developed for RAGpoisoning
attacks [ 43,39], which consider a structurally similar prob-
lem of crafting text that retrieves a chosen target. However,
their gradient-based optimizers are designed for single-
modal text retrieval and are easily trapped in local min-
ima on the non-smooth cross-modal embedding landscape,
motivating a gradient-free, exploration-heavy alternative.
We therefore proposeReward-Guided Policy Optimization
(RGPO), which updates a stochastic token policy from
reward-ranked candidates to balance exploration and ex-
ploitation, and comes with a finite-sample optimality guarantee (Theorem 3.1 in Section 3.2).
For the extraction segment, we derive a mixture-identity decomposition of the MIA score that cleanly
separates retrieval probability from the extraction hit/miss gap, and use it to guide task-specific
instantiations for T2I and Q&A (Section 3.3). We further pair these instantiations with K-means
aggregation across queries (Section 3.4). Across IRAG systems built on six generators and diverse
datasets, ImageAuditor consistently outperforms adapted baselines, exceeding 80% AUROC and
reaching around 50% TPR@5%FPR with only four queries per audited image. These values are well
above the bar for a strong MIA [1], establishing ImageAuditor as a practical auditing tool.
2 Background and Threat Model
2.1 Background
IRAG.We describe an IRAG system as a tuple S= (R, D, G) consisting of a retriever R, an image
database D, and a generator G, where each image I∈D is encoded by an embedding model Ethat
Ruses for retrieval. The key idea is to leverage external image knowledge (e.g., web-crawled images)
to compensate for knowledge gaps in the generator’s parameters. In this work, we cover IRAG for
both T2I and Q&A tasks. Formally, given a text query T, the system retrieves the top- Krelevant
images and produces outputy(an image for T2I, a text answer for Q&A).
2

Membership inference attack (MIA).MIA [ 32], which aims to determine whether a record belongs
to a target database, is a prominent framework for auditing data privacy in ML systems. Prior
MIAs [ 1,12,24] target a model’s training data and do not transfer to IRAG systems, where the target
data must first be retrieved before it can influence the output. While recent work [ 20,23] explores
MIAs against text-based RAG, their query designs fail to reliably retrieve the target image due to the
modality gap, especially under the small retrieval budgets used in IRAG (see Sec. 3.1). Moreover,
their Q&A-style attack strategies do not transfer to IRAG systems whose generator is a text-to-image
model. A separate line of work [ 3,21] addresses IRAG copyright concerns from thedefender’s
side, embedding watermark patterns into images so that owners can later assert ownership. These
approaches require proactive modification of the protected images and do not help general auditors
(e.g., ordinary users) whose published data lacks such sentinels.
2.2 Threat Model
Attacker’s goal.The adversary acts as a data auditor (e.g., a copyright holder, a database owner, or a
privacy regulator). Given a test image Iand a target IRAG system Stgt, the attacker aims to infer the
membership m∈ {0,1} ofIin the image database Dtgt. We frame this as a binary classification task
(member vs. non-member) intended to detect unauthorized use of copyrighted images.
Attacker’s capabilities.We consider a realistic and challenging attack setting. The attacker can only
interact with the target IRAG system Stgtthrough its API, operate under a limited query budget (e.g.,
4–5 per test image) to remain stealthy, and has no knowledge of the images in Dtgt. For transparency,
IRAG systems typically disclose the names of their AI components, such as the embedding model
and the generator. Following the standard threat model in prior RAG attacks [ 39,43,10,35,2], we
assume by default that the attacker has access to Etgt. This assumption is realistic because (i) deployed
IRAG systems overwhelmingly adopt off-the-shelf CLIP variants [ 27] as their retrievers [ 21,31,3,
6,22,5], since training a custom cross-modal encoder is expensive and rarely improves retrieval
quality; and (ii) commercial providers further expose embedding APIs that return query embeddings
directly.1Concretely, the attacker submits an image or text query and obtains its embedding via local
deployment or the API. To stress-test this assumption, we additionally evaluate a restrictive scenario
in which the attacker has no access to Etgtand instead relies on a weaker shadow embedding model
Esh(e.g., a smaller model trained with a different architecture or on a different data distribution);
ImageAuditor remains effective in this setting (Sec. 4.3). Finally, the attacker has access to a
captioning model and, optionally, a small shadow dataset drawn from the same domain as the test
images, which is readily collected from the Internet.
3 Method
3.1 Key Challenges
Designing an effective MIA against IRAG faces two fundamental challenges.Challenge 1 (C1):
Cross-modal retrieval.Unlike traditional MIAs [ 32,1], where members are memorized in model
parameters and can be directly probed, our setting requires the test image to be retrieved into the
generation context before it can influence the output of Stgt. Crafting a text query that reliably
retrieves a target image is non-trivial: text tokens are discrete, the embedding space is cross-modal,
and the retrieval budget is small (e.g., 1–3). Prior text-RAG MIAs [ 23,16,20] sidestep this by
injecting the target passage directly into the query—an option unavailable here, since an image cannot
be embedded into a text query, and even a strong caption fails to bridge the modality gap (Sec. 4.2).
White-box suffix-optimization attacks [ 39,43] also suffer, as they are easily trapped in local minima
on the non-smooth cross-modal landscape.Challenge 2 (C2): Discriminative signal extraction.
Even when retrieval succeeds, the attacker must still coerce the black-box generator into producing
outputs whose distributions diverge between members and non-members. This is non-trivial across
modalities: T2I generators produce unstructured images rather than formatted answers, while Q&A
generators tend to summarize multiple references and dilute the per-image signal. Prior Q&A-style
strategies [ 23,20] match generated text against pre-computed answers and require many queries per
sample, which is incompatible with T2I outputs and impractical under realistic stealth budgets.
1For example, Google’s multimodal embedding modelgemini-embedding-2is accessible via API.
3

Figure 2: Framework overview.(a) Single-query attack.Given a test image, we construct a text
query composed of an extraction segment (orange) and a retrieval segment (red). The retrieval
segment is optimized via RGPO (Sec. 3.2) such that, when the test image lies in the database, the
query retrieves it with high probability. The extraction segment and the similarity-based scoring
rule areco-designed(Sec. 3.3) to elicit outputs that produce discriminative MIA signals: members
yield outputs that closely resemble the test image (top), while non-members do not (bottom).(b)
Multi-query attack.ImageAuditor pairs multiple extraction and retrieval segments, scores each
query, and aggregates the scores via K-means clustering for improved attack results. (Sec. 3.4).
Our solution: query decomposition.ImageAuditor addresses both challenges by decomposing
each attack query into two functionally isolated segments—an extraction segment Textand a retrieval
segment Tret—such that T=T ext⊕T ret. The retrieval segment is optimized so that Iis retrieved with
high probability whenever it lies in the database (addressing C1), while the extraction segment and
the corresponding scoring rule areco-designedfrom a principled analysis to elicit outputs that yield a
discriminative member/non-member signal (addressing C2). This decomposition lets each component
target a clean, dedicated objective. Below, we describe the retrieval segment, the extraction segment,
and the score-aggregation strategy for multi-query attacks (see framework overview in Figure 2).
3.2 Retrieval Segment Optimization
The retrieval segment aims to ensure that, given a designed extraction segment Text(illustrated in
Sec. 3.3), the target system Stgtretrieves the test image Ias a reference for the query T=T ext⊕T ret
with high probability whenever Ilies in the database. To this end, we optimize a contrastive reward
that is observable to the attacker:
f(T ret;I, T ext) := sim 
E(T ext⊕T ret), E(I)
−α
|S|X
J∈Ssim 
E(T ext⊕T ret), E(J)
,
where Sis an optional set of shadow images and αbalances attraction to the test image against
repulsion from the shadow images. When no shadow set is available, we simply set α= 0 . Including
a shadow set shapes a more favorable and generalizable optimization landscape, particularly under
restrictive settings (see Table 10). Maximizing the contrastive reward encourages the composed
query to be close to the test image while remaining separated from generic distractors, which better
approximates the retrieval event than plain similarity alone. Here Edenotes the embedding model
accessible to the attacker (i.e.,E=E tgtorE=E sh, as introduced in the threat model).
A natural approach is to adapt optimization techniques from poisoning attacks on text-RAG (e.g.,
BadRAG [ 39], PoisonedRAG [ 43]). However, these methods suffer from two key limitations. (1)
Gradient-based methods are sensitive to initialization and prone to poor local optima in the non-convex
landscape induced by discrete token search, and the text-image modality gap further exacerbates this
issue. (2) Gradient-free methods such as PoisonedRAG-B [ 43] append content extracted from the
target sample to the attack segment; however, in the cross-modal setting, the target is an image, and
an unoptimized caption derived from it cannot reliably bridge the modality gap.
4

To address these issues, the optimization should bothexplorethe embedding space andexploithigh-
reward regions within it. Motivated by recent approaches that leverage RL for prompt optimization [ 8,
41], we proposeReward-Guided Policy Optimization (RGPO), which models the retrieval segment
as a factorized stochastic policy π(T ret|T ext) =QL
l=1πl(Tret[l]|T ext), where πlis the token
distribution at position l. At iteration t, RGPO samples Bcandidates {T(i)
ret}B
i=1∼πt, scores them
by the contrastive reward ri=f(T(i)
ret;I, T ext), and assigns each a sharpened, elite-gated weight
wt,i= exp(r i/τ)1{r i≥γt}, where γtis the (1−ρ) -quantile of {ri}andτ >0 is a temperature.
For each position l, the marginal is then updated toward the reweighted empirical token frequencies:
πt+1,l(v|T ext) = (1−β)π t,l(v|T ext) +βBX
i=1˜wt,i1{T(i)
ret[l] =v},
withv∈ V a vocabulary token, ˜wt,i=wt,i/P
jwt,j, andβ∈(0,1] controlling the adaptation speed.
Intuitively, RGPO shifts probability mass toward tokens that appear frequently among high-reward
candidates, while the factorized policy preserves enough exploration to escape local optima.
Theoretical guarantee.Write f(x) =f(x;I, T ext)for the contrastive reward and assume that
it decomposes additively across positions f(x) =PL
l=1fl(x[l]) forx∈ VL(we discuss in Ap-
pendix A.1 why a first-order surrogate is locally additive). The following theorem formalizes the
token-amplification effect underlying RGPO.
Theorem 3.1 (RGPO(ε, δ)-guarantee) Fix any ε >0 . If the number of RGPO iterations satisfies
n= Ω(log|V|) and the number of candidates at each iteration satisfies B= Ω(log(1/δ)) , thenbTret
from the final policy satisfies
f(bTret)≥max
x∈VLf(x)−ε
with probability at least1−δ. See proof in Appendix A.1.
3.3 Extraction Segment Design
The retrieval segment determines whether the test image enters the generator’s context. The remaining
question is how to design the extraction segment Textand the scoring rule so that, together, they elicit
discriminative MIA signals: low scores for non-members and high scores for members. We therefore
consider a queryT=T ext⊕Tretand analyze the MIA score induced by ageneralscoring rule.
LetXdenote the set of retrieved images, and let Y∼p(y| X, T) be the generator output. We
define a measurable score function s:I × Y →R , where the first argument is the test image and the
second is the generated output. The single-query MIA score is the random variable S=s(I, Y) . Let
Z∈ {0,1} indicate whether the test image Ibelongs to the target database. For z∈ {0,1} , define
the conditional score distribution under membership statuszas
Fz(a|I, T) = Pr(S≤a|Z=z, I, T).
We further define the retrieval hit event H={I∈ X} , i.e., the event that the test image is retrieved.
LetρT(I) = Pr(H= 1|Z= 1, I, T) denote the retrieval probability induced by the query T. We
also define the hit and miss score distributions as
Fhit(a|I, T) = Pr(S≤a|H= 1, I, T), F miss(a|I, T) = Pr(S≤a|H= 0, I, T).
Since a non-member image cannot be retrieved from the target database, we have H= 0 whenever
Z= 0. Hence, the member and non-member score distributions satisfy the mixture identity
F1(a|I, T) =ρ T(I)F hit(a|I, T)+ 
1−ρ T(I)
Fmiss(a|I, T), F 0(a|I, T) =F miss(a|I, T).
This identity separates the two roles of our attack: the retrieval segment optimized by RGPO increases
the retrieval probabilityρ T(I), while the extraction segment and the scoring function determine the
separation between FhitandFmiss. Thus, a successful attack requires both a large retrieval probability
and a scoring rule whose hit/miss gap is large. Equivalently, the extraction segment should make
outputs in the hit case systematically more similar to the test image than outputs in the miss case.
Guided by this principle, we instantiate the extraction segment as a prompt that asks the generator to
describe the retrieved reference images. We then define the MIA score as the similarity between the
5

test image Iand the generated output Y. Because T2I and Q&A systems expose different interaction
patterns, we employ task-specific adaptations of this extraction-and-scoring strategy, described below.
(a) T2I.Since T2I generators [ 26] lack instruction tuning, prompts such as“please describe the
retrieved images”are largely ignored (low text attention) and easily flagged as anomalous requests
(as illustrated in Appendix B). To address this, we use the caption of the test image as the extraction
segment, leveraging the copy-like behavior of generators under aligned inputs. If Iis successfully
retrieved as a reference (i.e., a member), the T2I generator faithfully reproduces its content. Even
when the retrieved image is not I, the generator still tends to reproduce salient content from the
reference, as it shares similar semantics with the caption. Together, these effects make caption-based
extraction an effective realization of our design principle. As illustrated in Appendix F, members
trigger near-verbatim reconstruction through the copy-like behavior, while non-members yield only
semantically aligned but visually divergent outputs.(b) Q&A.A direct command such as“describe
each retrieved image”causes the VLM to enumerate all retrieved images in a single response,
introducing substantial noise into the similarity computation. To address this, we split the output
into per-image descriptions and define the MIA score as the maximum similarity between Iand any
individual description. As shown in Table 3, we further observe that computing similarity within the
same modality (i.e., T2T or I2I) is essential for achieving strong performance.
3.4 The Overall Pipeline and Multi-query Aggregation
Our single-query attack proceeds in four steps: (1) construct an extraction segment Textfollowing
Sec. 3.3; (2) construct a retrieval segment Tretfollowing Sec. 3.2; (3) feed the composite query
T=T ext⊕T retinto the target system and obtain its output; and (4) compute the MIA score: image-to-
image similarity between the test image and the output in the T2I setting, or the maximum text-to-text
similarity between the test image description and each output fragment in the Q&A setting.
Multi-query aggregation.Some queries may fail to elicit a meaningful MIA signal due to errors in
optimizing the retrieval segment or approximating the appropriate extraction segment. To improve
reliability, we issue M×N queries by pairing each of Mextraction segments with Noptimized
retrieval segments. We then apply K-means clustering to the resulting scores, retain only the high-
scoring cluster, and aggregate the retained scores to produce the final MIA decision.
4 Experiments
We aim to answer the following research questions: (1) How does ImageAuditor perform compared
to adapted baselines under both T2I and Q&A settings? (Sec. 4.2) (2) Is ImageAuditor robust under
various settings? (Sec. 4.3) (3) How do our attack designs contribute to ImageAuditor? (Sec. 4.3)
4.1 Experimental Setup
Image database and evaluation datasets.For both T2I and Q&A tasks, we construct large-scale
image databases following prior work [ 22,6].(1) T2I tasks.We construct a database of 259,057
images spanning six datasets: MSCOCO [ 17], ImageNet-100 subset [ 30], WikiArt [ 25], Stanford
Cars [ 15], Stanford Dogs [ 7], and CelebA-HQ [ 13]. We use each dataset’s training split to construct
the database, and sample member and non-member images from the training and test splits. Due
to computational constraints, we randomly select three of the six datasets for attack evaluation.
Details can be found in Appendix D.1. For utility evaluation, we follow the standard protocol [ 22] as
illustrated in Appendix D.1.(2) Q&A tasks.Following [ 6], we use MMQA [ 34] and the MSCOCO
training split as the database, containing 58,075 and 118,287 images, respectively. Since MMQA
does not provide a training/test split, we randomly hold out 1,000 images as non-members and treat
the rest as members. For utility evaluation, we follow the same setup as AQUA [3].
Models.For T2I tasks, we employ four generators: SDXL [ 26] plus IP-Adapter [ 40] (default),
SD1.5 [ 29] plus IP-Adapter, Kandinsky v2.2 [ 28], and SDXL plus Conceptrol [ 11], paired with
CLIP-ViT-H/14 [ 27] as the retriever. For Q&A tasks, we employ Qwen2.5-VL-7B-Instruct [ 37]
(default) and LLaV A-1.6-Mistral-7B [ 19] as generators, with CLIP-ViT-L/14-336 as the retriever.
The setup follows standard practice in the IRAG literature [22, 6].
Metrics.We report AUROC, ACC, TPR@5%FPR, and Retrieval Success Rate (RSR) for attack
evaluation. Details of the attack and utility evaluation metrics are provided in Appendix D.3.
6

Table 1: Attack results of ImageAuditor and adapted baselines∗on IRAG for T2I tasks. For adapted
baselines, we use the same extraction segments and optimize retrieval segments using their respective
methods. The system uses CLIP ViT-H/14 for retrieval and SDXL+IP-Adapter for generation. TPR 5%
denotes TPR@5%FPR. Detailed TPR-FPR curves are provided in Appendix Figure 7.
Attack Method MSCOCO WikiArt Stanford Dogs
AUC ACC TPR 5%RSR AUC ACC TPR 5%RSR AUC ACC TPR 5%RSR
Naïve 0.72 0.70 0.37 0.53 0.63 0.63 0.31 0.33 0.66 0.67 0.29 0.40
PoisonedRAG-B∗0.71 0.71 0.26 0.55 0.68 0.66 0.33 0.42 0.70 0.68 0.38 0.48
PoisonedRAG∗0.79 0.77 0.45 0.66 0.69 0.70 0.40 0.53 0.72 0.74 0.40 0.52
BadRAG∗0.80 0.78 0.46 0.70 0.66 0.67 0.30 0.48 0.71 0.71 0.33 0.54
ImageAuditor0.91 0.85 0.55 0.96 0.87 0.82 0.49 0.93 0.90 0.84 0.65 0.90
SDXL+IPA SD1.5+IPA KandinskySDXL+Conc.0.00.20.40.60.81.0Rate Score
(a) T2I-MSCOCOSDXL+IPA SD1.5+IPA KandinskySDXL+Conc.0.00.20.40.60.81.0
(b) T2I-WikiArtQwen2.5-VL LLaV A-v1.60.00.20.40.60.81.0
(c) Q&A-MMQAAUROC ACC TPR@5%FPR
Figure 3: ImageAuditor remains effective across different generators for both T2I and Q&A tasks.
Baselines.We compare ImageAuditor against four baselines.Naïvedirectly uses the caption of the
test image as the query, with no retrieval segment.PoisonedRAG-B∗,PoisonedRAG∗andBadRAG∗
are adapted baselines that retain our extraction segment but optimize the retrieval segment using
techniques from existing RAG poisoning attacks [ 43,39] (See Appendix D.4). We do not include
previous text-RAG MIAs, as their Q&A-based pipelines do not transfer to IRAG. We also exclude
jailbreaking methods like GCG [ 42], as they are less effective than PoisonedRAG in this setting [ 43].
Evaluation setup.Unlike prior work, our image database contains samples drawn from multiple
sources (e.g., MSCOCO, WikiArt). In practice, an auditor is typically interested in a specific data
domain (e.g., artwork) and collects calibration samples from that domain to set the threshold and
infer membership. We therefore report per-dataset results, which better simulate this realistic auditing
scenario, and defer the combined attack setting to Appendix Table 6.
(1) Attack settings.The attacker is allowed M×N queries per test image, where Mis the number of
extraction segments and Nis the number of retrieval segments optimized per extraction segment;
we set M=N= 2 by default. For the extraction segment, we use Qwen-VL to generate and
rank candidate captions in the T2I setting, and manually design structured prompts in the Q&A
setting. For the retrieval segment, we set the token length to 8 and sample a small shadow set of 32
images (disjoint from the database). By default, the attacker has access to Etgt; under the restrictive
setting, we instead assume access to shadow embedding models, which are smaller and trained with
different data distributions [ 38,33]. The scoring model is the target (or a selected shadow) embedding
model for T2I and a text-embedding model E5 [ 36] for Q&A. Due to the high experimental cost
(∼1 min/sample), we use a 200 member/non-member set per subdataset and verify the stability in
Appendix Table 5. Remaining hyperparameters (optimization iterations, RGPO temperature, etc.) are
deferred to Appendix D.5.(2) IRAG settings.The retriever uses K= 1 for T2I and K= 2 for Q&A
by default, following the literature [ 40,3]; this small retrieval budget poses a substantial challenge
for our attack. Other generation settings follow each generator’s default setup (Appendix D.2).
4.2 Main Results
In this part, we discuss how different MIAs perform against IRAG. We further validate that IRAG
substantially improves utility on both T2I and Q&A tasks, and defer this analysis to Appendix C.
7

CLIP-L MetaCLIP EvaCLIP0.000.250.500.751.00Rate Score
(a) MSCOCOCLIP-L MetaCLIP EvaCLIP0.000.250.500.751.00
(b) WikiArtCLIP-L MetaCLIP EvaCLIP0.000.250.500.751.00
(c) Stanford DogsAUC ACC TPR@5%FPRFigure 4: ImageAuditor remains effective under strict black-box settings. We use a weaker shadow
embedding model (e.g., with different architectures or training distributions) to optimize the retrieval
segments and compute the MIA scores. The target system uses CLIP ViT-H/14 for retrieval.
Table 2: Fine-grained captions for the extraction
segment elicit stronger copy-like behavior and
stronger attacks than generic captions (Gen.).
TypeMSCOCO WikiArt
AUC ACC TPR 5%AUC ACC TPR 5%
Gen. 0.71 0.70 0.38 0.75 0.72 0.39
Fine.0.91 0.85 0.55 0.87 0.82 0.49Table 3: ImageAuditor achieves stronger attacks
when scoring within a single modality (Single)
than across modalities (Cross).
Type T2I-MSCOCO Q&A-MMQA
AUC ACC TPR 5%AUC ACC TPR 5%
Cross 0.60 0.59 0.14 0.82 0.79 0.37
Single 0.91 0.85 0.55 0.85 0.82 0.56
ImageAuditor outperforms baseline attacks on IRAG systems for T2I and Q&A tasks.We
present the main results of ImageAuditor and adapted baselines on IRAG for T2I tasks in Table 1
and for Q&A tasks in Appendix Table 4. All methods share the same extraction segment and differ
only in retrieval segment optimization. (1) Our attack achieves state-of-the-art performance on both
T2I and Q&A tasks, outperforming existing attacks by a large margin (e.g., over 10% AUROC on
T2I tasks). Existing attacks suffer from low target retrieval rates because they easily get trapped in
local minima within the cross-modal embedding space. (2) Our attack is effective in the low false
positive region (also illustrated in Figure 7 and 8), achieving around 50% TPR@5%FPR, which is
commonly interpreted as a strong attack in the literature [ 1]. (3) In the T2I setting, WikiArt shows
lower performance than other datasets because diffusion models exhibit weaker copy-like behavior on
artwork references, as stylized images are less aligned with the text. (4) In the Q&A setting, our attack
is more effective on MMQA due to the uniqueness of its images (e.g., film posters). (5) Existing
baselines perform poorly in the Q&A setting because when the extraction segment (“Describe each
retrieved...”) is unrelated to the test image, the initial query embedding lies far from the target, making
gradient-based optimization challenging-a regime where RGPO’s reward-guided updates succeed.
ImageAuditor remains effective across diverse generators for both T2I and Q&A tasks.For T2I,
beyond the default SDXL+IPA, we evaluate generators varying along two axes—diffusion backbone
and conditioning mechanism—including SD1.5+IPA (different backbone), Kandinsky (different
family, adapter-free), and SDXL+Conceptrol (alternative conditioning). As shown in Figure 3(a,b),
ImageAuditor maintains high AUROC (>0.80) across all variants. Conceptrol yields the weakest
attack due to its more creative generation capability, yet it still exceeds 0.30 TPR@5%FPR. For
Q&A, Qwen2.5-VL and LLaV A-v1.6 produce nearly identical results (Figure 3(c)), as both VLMs
faithfully describe image content. Additional results are provided in the Appendix Figure 9. Overall,
these results confirm that ImageAuditor is robust to the choice of underlying generators.
4.3 Ablation Study
Unless otherwise mentioned, we conduct ablation studies on IRAG for T2I tasks.
ImageAuditor remains effective in the restrictive attack setting.Under restrictive constraints,
where the attacker has no access to the embedding model used by the target system, we construct
retrieval segments using a weaker shadow embedding model. Despite this mismatch, Figure 4 shows
that ImageAuditor still achieves an AUROC of around 0.75, which is generally considered strong for
8

200K 250K 300K
Size of the Image Database0.000.250.500.751.00Rate Score
1 2 3
Number of Retrieved Images0.000.250.500.751.00
3 8 13
Retrieval Segment Length0.000.250.500.751.00
AUROC ACC TPR@5%FPRFigure 5: Impact of (a) the size of the image database, and (b) the number of retrieved images, and (c)
the retrieval segment length on attack results. ImageAuditor remains robust across all these settings.
MIA. The observed performance drop stems not only from weaker retrieval segment construction
but also from the weaker scoring model used in this setup; as shown in Appendix Table 11, stronger
scoring models consistently yield better results by capturing more fine-grained image semantics.
Interestingly, EvaCLIP transfers best on MSCOCO and Dogs but degrades on WikiArt, likely due to
lower embedding similarity with the target (ViT-H/14) on artwork images.
Our tailored attack designs lead to stronger attack results.We further investigate the impact of
our attack designs for theextraction segment,retrieval segment, andscoring mechanism. For the
extraction segment, as shown in Table 2 and Appendix Table 9, fine-grained captions generated by
BLIP and Qwen-VL significantly improve the attack results, yielding up to ∼10% gains in AUROC.
We attribute this to the fact that high-quality captions more effectively elicit copy-like behavior
from diffusion models and provide a better starting point for retrieval segment construction. For
the retrieval segment, we have demonstrated the superiority of the proposed RGPO mechanism in
Table 1. In Appendix Table 10, we further evaluate the impact of incorporating a small shadow set for
contrastive reward. Under the restrictive setting, we show that negative samples consistently improve
attack performance by modeling a more favorable optimization landscape that generalizes. Beyond
these two components, we also examine different strategies for computing the similarity score. In
both T2I and Q&A settings, similarity can be measured between the output and the image (image
modality) or between the output and image descriptions (text modality). As shown in Table 3, we
observe that computing similarity within the same modality yields large improvements, especially for
TPR 5%; accordingly, we extract image details from the test image in the Q&A setting.
ImageAuditor remains effective under different IRAG setups.In Figure 5, we investigate the
impact of the size of the image database, the number of retrieved images, and the length of the retrieval
segment on attack results. In (a), we enlarge the image database by incorporating additional images
from ImageNet and observe that the database size has only a minor effect on attack performance.
This is because RGPO can effectively retrieve the test image whenever it lies in the database. We
further observe that a larger retrieval budget slightly improves the attack in the Q&A setting, as K=3
increases the likelihood of the members being retrieved. Here, we experiment with Q&A setting due
to architectural constraints in T2I generators. Finally, ImageAuditor saturates even when the retrieval
segment length is small (8 tokens), which makes our attack stealthier and less likely to be detected.
Ablation on Multi-Query Attack Strategies.We evaluate the impact of aggregation strategy and
query number for multi-query attacks in Appendix Figures 10 and 11. We find that the K-means-based
strategy balances precision and coverage, yielding more stable attack performance across settings.
We also observe that increasing the number of queries improves attack results, though notably, even a
single-query attack achieves competitive performance. Detailed analysis is provided in Appendix C.
5 Conclusion
We introduceImageAuditor, the first membership inference attack (MIA) against image-based RAG
(IRAG) systems. To overcome fundamental limitations of prior text-RAG MIAs,ImageAuditor
decomposes each attack query into a retrieval segment and an extraction segment. We propose
Reward-Guided Policy Optimization to enhance cross-modal retrieval, and design tailored prompting
strategies paired with a scoring rule to elicit discriminative MIA signals. Extensive experiments show
thatImageAuditorachieves over 80% AUROC with only four queries and remains effective under
restrictive attack settings. These results establish ImageAuditor as a practical auditing tool.
9

Limitations and Future Work.This work focuses on establishing the feasibility of MIAs against
IRAG. While our attack achieves strong performance across multiple settings, we evaluate robustness
only against text/image attention ratio-based detection (see Appendix B); more sophisticated defenses,
such as differential privacy [ 14] and output perturbation, remain unexplored. In addition, we focus on
text-based retrieval and do not evaluate multimodal inputs. However, as discussed in Appendix E, our
attack naturally extends to this setting. We leave these directions to future work.
References
[1]Nicholas Carlini, Steve Chien, Milad Nasr, Shuang Song, Andreas Terzis, and Florian Tramer.
Membership inference attacks from first principles. In2022 IEEE symposium on security and
privacy (SP), pages 1897–1914. IEEE, 2022.
[2]Harsh Chaudhari, Giorgio Severi, John Abascal, Matthew Jagielski, Christopher A Choquette-
Choo, Milad Nasr, Cristina Nita-Rotaru, and Alina Oprea. Phantom: General trigger attacks on
retrieval augmented language generation. 2024.
[3]Tianyu Chen, Jian Lou, and Wenjie Wang. Safeguarding multimodal knowledge copyright in
the rag-as-a-service environment.arXiv preprint arXiv:2506.10030, 2025.
[4]Wenhu Chen, Hexiang Hu, Xi Chen, Pat Verga, and William Cohen. Murag: Multimodal
retrieval-augmented generator for open question answering over images and text. InProceedings
of the 2022 Conference on Empirical Methods in Natural Language Processing, pages 5558–
5570, 2022.
[5]Wenhu Chen, Hexiang Hu, Chitwan Saharia, and William W Cohen. Re-imagen: Retrieval-
augmented text-to-image generator.arXiv preprint arXiv:2209.14491, 2022.
[6]Zhanpeng Chen, Chengjin Xu, Yiyan Qi, and Jian Guo. Mllm is a strong reranker: Ad-
vancing multimodal retrieval-augmented generation via knowledge-enhanced reranking and
noise-injected training.arXiv preprint arXiv:2407.21439, 2024.
[7]E Dataset. Novel datasets for fine-grained image categorization. InFirst workshop on fine
grained visual categorization, CVPR. Citeseer. Citeseer. Citeseer, volume 5, page 2. Citeseer,
2011.
[8]Mingkai Deng, Jianyu Wang, Cheng-Ping Hsieh, Yihan Wang, Han Guo, Tianmin Shu, Meng
Song, Eric Xing, and Zhiting Hu. Rlprompt: Optimizing discrete text prompts with rein-
forcement learning. InProceedings of the 2022 Conference on Empirical Methods in Natural
Language Processing, pages 3369–3391, 2022.
[9]Javid Ebrahimi, Anyi Rao, Daniel Lowd, and Dejing Dou. Hotflip: White-box adversarial
examples for text classification. InProceedings of the 56th Annual Meeting of the Association
for Computational Linguistics (Volume 2: Short Papers), pages 31–36, 2018.
[10] Runpeng Geng, Yanting Wang, Ying Chen, and Jinyuan Jia. Unic-rag: Universal knowledge
corruption attacks to retrieval-augmented generation.arXiv preprint arXiv:2508.18652, 2025.
[11] Qiyuan He and Angela Yao. Conceptrol: Concept control of zero-shot personalized image
generation.arXiv preprint arXiv:2503.06568, 2025.
[12] Yu He, Boheng Li, Liu Liu, Zhongjie Ba, Wei Dong, Yiming Li, Zhan Qin, Kui Ren, and Chun
Chen. Towards{Label-Only}membership inference attack against pre-trained large language
models. In34th USENIX Security Symposium (USENIX Security 25), pages 1609–1628, 2025.
[13] Tero Karras, Timo Aila, Samuli Laine, and Jaakko Lehtinen. Progressive growing of gans for
improved quality, stability, and variation.arXiv preprint arXiv:1710.10196, 2017.
[14] Tatsuki Koga, Ruihan Wu, Zhiyuan Zhang, and Kamalika Chaudhuri. Privacy-preserving
retrieval-augmented generation with differential privacy.arXiv preprint arXiv:2412.04697,
2024.
10

[15] Jonathan Krause, Michael Stark, Jia Deng, and Li Fei-Fei. 3d object representations for fine-
grained categorization. InProceedings of the IEEE international conference on computer vision
workshops, pages 554–561, 2013.
[16] Yuying Li, Gaoyang Liu, Chen Wang, and Yang Yang. Generating is believing: Membership
inference attacks against retrieval-augmented generation. InICASSP 2025-2025 IEEE Inter-
national Conference on Acoustics, Speech and Signal Processing (ICASSP), pages 1–5. IEEE,
2025.
[17] Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr
Dollár, and C Lawrence Zitnick. Microsoft coco: Common objects in context. InEuropean
conference on computer vision, pages 740–755. Springer, 2014.
[18] Weizhe Lin and Bill Byrne. Retrieval augmented visual question answering with outside
knowledge. InProceedings of the 2022 conference on empirical methods in natural language
processing, pages 11238–11254, 2022.
[19] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning.Advances
in neural information processing systems, 36:34892–34916, 2023.
[20] Mingrui Liu, Sixiao Zhang, and Cheng Long. Mask-based membership inference attacks for
retrieval-augmented generation. InProceedings of the ACM on Web Conference 2025, pages
2894–2907, 2025.
[21] Ziyuan Luo, Yangyi Zhao, Ka Chun Cheung, Simon See, and Renjie Wan. Imagesentinel:
Protecting visual datasets from unauthorized retrieval-augmented image generation.arXiv
preprint arXiv:2510.12119, 2025.
[22] Yuanhuiyi Lyu, Xu Zheng, Lutao Jiang, Yibo Yan, Xin Zou, Huiyu Zhou, Linfeng Zhang,
and Xuming Hu. Realrag: Retrieval-augmented realistic image generation via self-reflective
contrastive learning. InInternational Conference on Machine Learning, pages 41772–41790.
PMLR, 2025.
[23] Ali Naseh, Yuefeng Peng, Anshuman Suri, Harsh Chaudhari, Alina Oprea, and Amir
Houmansadr. Riddle me this! stealthy membership inference for retrieval-augmented genera-
tion. InProceedings of the 2025 ACM SIGSAC Conference on Computer and Communications
Security, pages 1245–1259, 2025.
[24] Yan Pang and Tianhao Wang. Black-box membership inference attacks against fine-tuned
diffusion models.arXiv preprint arXiv:2312.08207, 2023.
[25] Fred Phillips and Brandy Mackintosh. Wiki art gallery, inc.: A case for critical thinking.Issues
in Accounting Education, 26(3):593–608, 2011.
[26] Dustin Podell, Zion English, Kyle Lacey, Andreas Blattmann, Tim Dockhorn, Jonas Müller, Joe
Penna, and Robin Rombach. Sdxl: Improving latent diffusion models for high-resolution image
synthesis.arXiv preprint arXiv:2307.01952, 2023.
[27] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal,
Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual
models from natural language supervision. InInternational conference on machine learning,
pages 8748–8763. PmLR, 2021.
[28] Anton Razzhigaev, Arseniy Shakhmatov, Anastasia Maltseva, Vladimir Arkhipkin, Igor Pavlov,
Ilya Ryabov, Angelina Kuts, Alexander Panchenko, Andrey Kuznetsov, and Denis Dimitrov.
Kandinsky: an improved text-to-image synthesis with image prior and latent diffusion. In
Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing:
System Demonstrations, pages 286–295, 2023.
[29] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. High-
resolution image synthesis with latent diffusion models. InProceedings of the IEEE/CVF
conference on computer vision and pattern recognition, pages 10684–10695, 2022.
11

[30] Olga Russakovsky, Jia Deng, Hao Su, Jonathan Krause, Sanjeev Satheesh, Sean Ma, Zhiheng
Huang, Andrej Karpathy, Aditya Khosla, Michael Bernstein, et al. Imagenet large scale visual
recognition challenge.International journal of computer vision, 115(3):211–252, 2015.
[31] Rotem Shalev-Arkushin, Rinon Gal, Amit H Bermano, and Ohad Fried. Imagerag: Dynamic
image retrieval for reference-guided image generation.arXiv preprint arXiv:2502.09411, 2025.
[32] Reza Shokri, Marco Stronati, Congzheng Song, and Vitaly Shmatikov. Membership inference
attacks against machine learning models. In2017 IEEE symposium on security and privacy
(SP), pages 3–18. IEEE, 2017.
[33] Quan Sun, Yuxin Fang, Ledell Wu, Xinlong Wang, and Yue Cao. Eva-clip: Improved training
techniques for clip at scale.arXiv preprint arXiv:2303.15389, 2023.
[34] Alon Talmor, Ori Yoran, Amnon Catav, Dan Lahav, Yizhong Wang, Akari Asai, Gabriel Ilharco,
Hannaneh Hajishirzi, and Jonathan Berant. Multimodalqa: Complex question answering over
text, tables and images.arXiv preprint arXiv:2104.06039, 2021.
[35] Haowei Wang, Rupeng Zhang, Junjie Wang, Mingyang Li, Yuekai Huang, Dandan Wang,
and Qing Wang. Joint-gcg: Unified gradient-based poisoning attacks on retrieval-augmented
generation systems. InProceedings of the AAAI Conference on Artificial Intelligence, volume 40,
pages 35793–35801, 2026.
[36] Liang Wang, Nan Yang, Xiaolong Huang, Binxing Jiao, Linjun Yang, Daxin Jiang, Rangan
Majumder, and Furu Wei. Text embeddings by weakly-supervised contrastive pre-training.
arXiv preprint arXiv:2212.03533, 2022.
[37] Peng Wang, Shuai Bai, Sinan Tan, Shijie Wang, Zhihao Fan, Jinze Bai, Keqin Chen, Xuejing
Liu, Jialin Wang, Wenbin Ge, et al. Qwen2-vl: Enhancing vision-language model’s perception
of the world at any resolution.arXiv preprint arXiv:2409.12191, 2024.
[38] Hu Xu, Saining Xie, Xiaoqing Ellen Tan, Po-Yao Huang, Russell Howes, Vasu Sharma, Shang-
Wen Li, Gargi Ghosh, Luke Zettlemoyer, and Christoph Feichtenhofer. Demystifying clip data.
arXiv preprint arXiv:2309.16671, 2023.
[39] Jiaqi Xue, Mengxin Zheng, Yebowen Hu, Fei Liu, Xun Chen, and Qian Lou. Badrag: Identifying
vulnerabilities in retrieval augmented generation of large language models.arXiv preprint
arXiv:2406.00083, 2024.
[40] Hu Ye, Jun Zhang, Sibo Liu, Xiao Han, and Wei Yang. Ip-adapter: Text compatible image
prompt adapter for text-to-image diffusion models.arXiv preprint arXiv:2308.06721, 2023.
[41] Taeyoung Yun, Dinghuai Zhang, Jinkyoo Park, and Ling Pan. Learning to sample effective
and diverse prompts for text-to-image generation. InProceedings of the Computer Vision and
Pattern Recognition Conference, pages 23625–23635, 2025.
[42] Andy Zou, Zifan Wang, Nicholas Carlini, Milad Nasr, J Zico Kolter, and Matt Fredrikson.
Universal and transferable adversarial attacks on aligned language models.arXiv preprint
arXiv:2307.15043, 2023.
[43] Wei Zou, Runpeng Geng, Binghui Wang, and Jinyuan Jia. {PoisonedRAG }: Knowledge
corruption attacks to {Retrieval-Augmented }generation of large language models. In34th
USENIX Security Symposium (USENIX Security 25), pages 3827–3844, 2025.
12

A Proofs
A.1 Proof of Theorem 3.1
We prove the theorem for β= 1 .β <1 introduces additional approximation error but does not
change the qualitative amplification mechanism.
For each positionl∈ {1, . . . , L}, define
Gl:=n
v∈ V:f l(v)≥max
v∈Vfl(v)−ε
Lo
.
IfGl=V, then every token at position lis already ε/L-good and there is nothing to prove for that
position. Otherwise define the good–bad margin ∆l:= min v∈Glfl(v)−max v/∈Glfl(v)>0 and
∆ := min l∆l>0.
We note that any sequence whose token at each position lies in the corresponding good set is ε-optimal.
To see that, for anyx∈ VLwithx[l]∈ G lfor alll, we have
max
z∈VLf(z)−f(x) =LX
l=1
max
v∈Vfl(v)−f l(x[l])
≤LX
l=1ε
L=ε.
Therefore, it suffices to show that the modal token of the final policy belongs to Glat every position.
To see the token-amplification effect, we first analyze the one-step amplification. Fix an iteration
tand a position l. Let the current product policy be πt=QL
j=1πt,j, and let X∼π t. Define the
importance weight wt(x) := exp(f(x)/τ)1{f(x)≥γ t}. For each token v∈ V , letmt,l(v) :=
E[wt(X)|X[l] =v, π t]. Since the reward is additive, if we write St,l:=P
j̸=lfj(X[j]), then we
have
mt,l(v) = expfl(v)
τ
E
expSt,l
τ
1{St,l≥γt−fl(v)}πt
.
For anyv∈ G landu /∈ G l, sincef l(v)−f l(u)≥∆ l≥∆, we have
mt,l(v)≥expfl(v)−f l(u)
τ
expfl(u)
τ
E
expSt,l
τ
1{St,l≥γ⋆
t−fl(u)}πt
≥exp∆
τ
mt,l(u).
Define the exact marginal update induced byw t
pt+1,l(v) =E[wt(X)1{X[l] =v} |π t]
E[wt(X)|π t]=πt,l(v)m t,l(v)P
u∈Vπt,l(u)m t,l(u),
and define
qt,l:=X
v∈Glπt,l(v),¯q t+1,l:=X
v∈Glpt+1,l(v).
Then we have
¯qt+1,l=P
v∈Glπt,l(v)m t,l(v)P
v∈Glπt,l(v)m t,l(v) +P
v/∈Glπt,l(v)m t,l(v)≥κqt,l
(1−q t,l) +κq t,l,
whereκ= exp (∆/τ)>1. Equivalently,
¯qt+1,l
1−¯q t+1,l≥κqt,l
1−q t,l.
Therefore the update multiplies the good-token odds by at least the factorκat every iteration.
Given the amplification result, we can now bound the number of RGPO iterations nand the number
of samples at each iteration B. Since RGPO is initialized uniformly, for every position lwe have
q1,l≥1/|V| . The amplification result then implies that, after n= Ω(log|V|) iterations, the
good-token mass at each position satisfies qn,l≥1−1/(2L) . Therefore, for a single sample
13

X∼π nfrom the final policy, a union bound over the Lpositions gives Pr(X[l]∈ G lfor alll)≥
1−PL
l=1Pr(X[l]/∈ G l)≥1−L/(2L) = 1/2 . Whenever this event occurs, the additive structure of
fimplies f(X)≥max x∈VLf(x)−ε . Since each draw is ε-optimal with probability at least 1/2, by
the independent sampling from the policy, choosing B= Ω(log(1/δ)) ensures that, with probability
at least1−δ,
f(bTret)≥max
x∈VLf(x)−ε.
This completes the proof.
On the additivity assumption.The cosine reward is not globally position-additive. However,
for a fixed extraction segment Textand a short retrieval suffix, a first-order Taylor expansion of
sim 
E(T ext⊕T ret), E(I)
around the current policy mean is position-separable:
f(x)≈c+LX
l=1⟨gl, ϕ(x[l])⟩,
where glis the gradient with respect to the l-th token embedding and ϕis the embedding lookup.
Theorem 3.1 should therefore be read as a guarantee on this local surrogate, which RGPO effectively
optimizes through reward-ranked sampling. The empirical retrieval success rates (90%+) in our
experiments (Table 1) are consistent with this approximation being tight in the regimes we evaluate.
14

(a) Diffusion Step 1
 (b) Diffusion Step 6
 (c) Diffusion Step 11
 (d) Diffusion Step 16
Figure 6: Instruction-based extraction segments produce abnormal text-image attention ratios due to
the lack of semantic meaning in text prompts, which can be easily detected in early diffusion steps.
B Text/Image Attention Ratio based Detection
In the T2I setting, a naïve attacker may use an instruction prompt (e.g.,“Please describe the retrieved
images”) or even an empty prompt (e.g.,“”) to coerce the T2I generator into directly outputting the
retrieved image. The underlying intuition is that, in such cases, the text input is largely ignored and
generation is governed primarily by the image reference, leading to direct leakage of the retrieved
image. While effective in certain cases, we show that this strategy can be easily detected by a simple
Text/Image Attention Ratiomechanism.
Detection mechanism.Let’s take SDXL+IP-Adapter as an example. The defender collects cross-
attention scores from a specific U-Net block (we use down_blocks.2 ) of the diffusion model. In
the IP-Adapter architecture, each cross-attention layer contains two parallel branches: one attending
to text embeddings and one attending to image embeddings. Let A(l)
textandA(l)
imgdenote the attention
scores from the text and image branches at layer l, respectively. The text/image attention ratio Ais
computed as:
A=1
LLX
l=1∥A(l)
text∥1
∥A(l)
img∥1+ϵ,(1)
where Lis the number of cross-attention layers in that block and ϵis a small constant for numerical
stability. A low ratio indicates that the generation is dominated by image conditioning, signaling a
potential leakage attack.
Detection results.As shown in Figure 6, both instruction and empty prompts produce abnormally
low text/image attention ratios at early diffusion steps, making them easily distinguishable from
benign queries. In contrast, our caption-based attack uses natural text queries and yields ratios
within the normal range, thereby evading detection. Moreover, while the empty prompt provides a
strong shortcut for image leakage, it underperforms our caption-based approach in eliciting reference
images, achieving a CLIP image similarity of 0.815 compared to 0.833 for ours (note that we directly
compare the similarity between the reference image and the output image, rather than measuring MIA
performance). The empty prompt also complicates retrieval segment construction, as the absence
of textual context weakens the optimization signal. Together, these results demonstrate that our
extraction segment design is bothstealthyandeffective.
C Additional Experiment Analysis
IRAG improves the utility of both T2I and Q&A tasks.As shown in Appendix Tables 7 and 8,
IRAG substantially improves T2I generation utility (measured by CLIP-I) by over 10% on the
ImageNet100, WikiArt, Dogs, and Cars datasets, and boosts response accuracy on MMQA from 30%
to approximately 55%. These gains stem from the inherent limitations of parametric knowledge in
generative models: (1) fine-grained visual concepts, long-tail entities, and domain-specific details are
often poorly learned and underrepresented in the model’s parameters; and (2) novel concepts that
emerge after the base diffusion model or VLM is trained remain entirely absent from its parametric
knowledge. Both cases lead to hallucinated or generic outputs. By retrieving semantically relevant
images at inference time, IRAG grounds generation in concrete visual evidence, allowing the model to
faithfully reproduce category-specific details in T2I tasks and to answer visually grounded questions
that would otherwise exceed its parametric capacity. This confirms that IRAG serves as an effective
15

(a) T2I-MSCOCO
 (b) T2I-WikiArt
 (c) T2I-Stanford Dogs
Figure 7: TPR–FPR curves of ImageAuditor and adapted baselines on IRAG for T2I generation tasks.
(a) Q&A-MMQA
 (b) Q&A-MSCOCO
Figure 8: TPR–FPR curves of ImageAuditor and adapted baselines on IRAG for Q&A tasks. Adapted
PoisonedRAG and BadRAG fail because the extraction segment—“Describe each retrieved image.
Format as ’Image 1:’, ’Image 2:’, etc.”——provides no semantic alignment with the target image,
causing prior attacks to get trapped in local minima and fail to retrieve that image.
external memory, complementing the model’s internal knowledge with up-to-date and fine-grained
visual information.
Detailed Analysis of Multi-Query Attack Strategies.We evaluate different variants of attack
strategies for multi-query attacks in Appendix Figures 10 and 11. Specifically, we observe that
different aggregation strategies achieve comparable performance under the standard setting, but the
average-score-based approach exhibits much lower performance ( ∼10% drop in TPR@5%FPR)
under the restrictive setting. We explain that under this more challenging setting, many queries fail to
retrieve the test image despite its presence in the database and, therefore, produce noisy scores. The
K-means-based strategy mitigates this by aggregating scores from high-confidence clusters, which
balances precision with coverage and yields more stable performance across settings. In Figure 11,
we show that increasing the number of queries slightly improves attack results, while notably, even
the single-query attack achieves desirable performance.
D Additional Details
D.1 Description of the Datasets.
For both T2I and Q&A tasks, we construct large-scale image databases by aggregating data from
multiple sources, following prior work [ 22,6]. For datasets without official splits, we partition the
data into member and non-member sets, using the member split to build the image database.
T2I datasets.(1) MSCOCO (2017)[ 17] is a large-scale image-caption dataset of real-world photo-
graphic images, containing 118K training images and 5K validation images.(2) ImageNet-100is a
100-class subset of ImageNet [ 30] comprising ∼130K images across diverse object categories. For
controlled experiments, we use a 50-class subset by default and vary the number of classes (from 1 to
100) in ablation studies.(3) WikiArt[ 25] contains 81,444 paintings spanning 27 styles and 45 genres
16

Table 4: Main results of ImageAuditor on IRAG for Q&A tasks. Following the literature [ 6], the
system uses CLIP ViT-L/14-336 for retrieval and Qwen2.5-VL-7B-Instruct for generation. All
methods share the same extraction segment (“Describe each retrieved image...”) and differ only in
their retrieval segment, so the results reflect optimizer quality. PoisonedRAG-B∗outperforms the two
gradient-based variants as it avoids noisy gradient estimation in this challenging setting.
Attack Method MMQA MSCOCO
AUC ACC TPR 5% AUC ACC TPR 5%
Naïve N/A N/A
PoisonedRAG-B∗0.62 0.62 0.19 0.57 0.59 0.14
PoisonedRAG∗0.53 0.57 0.06 0.51 0.56 0.06
BadRAG∗0.52 0.55 0.04 0.49 0.54 0.03
ImageAuditor0.85 0.82 0.56 0.82 0.76 0.39
Table 5: Stability of our evaluation. MSCOCO
and MMQA are used for T2I and Q&A tasks.
Task # Tests AUC ACC TPR 5%
T2I taskDefault 0.91 0.85 0.55
800 0.91 0.85 0.56
Q&A taskDefault 0.85 0.82 0.56
800 0.85 0.80 0.49Table 6: Attack evaluation on T2I tasks under
both per-dataset and combined settings.
Setting Dataset AUC ACC TPR 5%
SeparateMSCOCO 0.91 0.85 0.55
WikiArt 0.87 0.82 0.49
Dogs 0.90 0.84 0.65
Combined Full 0.87 0.79 0.46
from over 2,000 artists across centuries of art history. We exclude paintings labeled as unknown
artists, yielding a filtered dataset of ∼39K images. We then split the dataset by artist at a 4:1 ratio.(4)
Stanford Cars[ 15] consists of 16,185 images of 196 car models, with an official 8,144/8,041 train/test
split.(5) Stanford Dogs[ 7] contains 20,580 images of 120 dog breeds, with an official 12,000/8,580
train/test split.(6) CelebA-HQ[13] is a high-quality facial dataset containing 30,000 face images at
1024×1024 resolution. We split the dataset at a 4:1 ratio. In total, we construct an image database
of 259,057 images spanning all six datasets. Together, they provide concepts coveringreal-world
scenes,fine-grained and rare objects,artistic styles, andhigh-quality faces, etc.
Q&A dataset.(1) MMQA(MultiModalQA) [ 34] is a multi-modal question answering benchmark
requiring reasoning across text, tables, and images. We use the complete image partition of the
dataset, containing 58,075 images. Due to the lack of official splits, we randomly sample 1,000
images as the non-member set.(2) MSCOCOis also used in RagVL [ 6] as a Q&A dataset to evaluate
text-to-image retrieval performance over a large-scale image database.
Utility evaluation.The T2I database covers diverse visual concepts spanning real scenes, fine-grained
and rare objects, and artistic styles. For datasets with clear class labels, we randomly sample 20
classes and generate 10 prompts per class—for example,A photo of a {class} sitting on the ground”
for Stanford Dogs, andA modern artwork influenced by {class}”for WikiArt. This allows us to
evaluate whether generated images faithfully capture the visual concepts from each class and to
compare generators with and without IRAG. For MSCOCO and CelebA-HQ, we randomly sample
200 prompts following their respective distributions and compare the generated image distributions
using FID and CLIP scores. For MMQA, which supplies visual knowledge for knowledge-dependent
question answering, we follow the same setup as AQUA and measure response accuracy to compare
generators with and without IRAG.
D.2 Description of the Generation Settings
For T2I tasks, we use four generators that span different architectures and conditioning mecha-
nisms:(1) SDXL + IP-Adapter[ 26,40]: We pair SDXL with the ViT-H IP-Adapter Plus variant
(ip-adapter-plus_sdxl_vit-h ) and use an IP-Adapter scale of 0.6, which provides a balanced
setting between image and text conditioning. Images are generated at 1024 ×1024 resolution with
30 denoising steps, following the default setting of SDXL. Empirically, ImageAuditor maintains
AUROC above 80% across practical scale values ( ≥0.5 ); lower scales weaken retrieval influence and
are rarely used in IRAG systems, as they reduce the benefit of retrieval augmentation.(2) SD1.5 +
17

Table 7: IRAG improves the utility of text-to-image generation. Following [ 22], we use CLIP-I as a
metric of concept-level semantic alignment, which measures the similarity between the generated
image and the corresponding class prototype embedding.
Generator RAG WikiArt ImageNet100 Stanford Dogs
CLIP-I CLIP-T FID CLIP-I CLIP-T FID CLIP-I CLIP-T FID
SDXL+IP-Adapter×0.4530.314330 0.649 0.343 145 0.4300.324235
✓0.5220.296292 0.744 0.358 103 0.6230.269138
SD1.5+IP-Adapter×0.419 0.292 340 0.617 0.312 189 0.4120.286243
✓0.539 0.295 297 0.732 0.321 108 0.6210.264135
Generator RAG Cars MSCOCO CelebA-HQ
CLIP-I CLIP-T FID CLIP-I CLIP-T FID CLIP-I CLIP-T FID
SDXL+IP-Adapter×0.618 0.333 160 / 0.362 190 / 0.365 171
✓0.688 0.334 112/0.364 178/0.375 117
SD1.5+IP-Adapter×0.608 0.311 124 / 0.343 180 / 0.351 112
✓0.684 0.319 111/0.347 175/0.364 96
Table 8: IRAG improves the utility of question answering. We report accuracy (ACC) on MMQA
across different numbers of retrieved passages.
GeneratorNoneK= 2K= 3
Qwen2.5-7B-Instruct 0.3140.5290.495
LLaV A-v1.6-Mistral-7B 0.3590.5750.573
IP-Adapter[ 29,40]: We pair Stable Diffusion 1.5 with its corresponding IP-Adapter Plus variant and
use an IP-Adapter scale of 0.6. Images are generated at 512 ×512 resolution with 30 denoising steps,
following the default setting of SD1.5.(3) Kandinsky v2.2[ 28]: We use Kandinsky v2.2, a latent
diffusion model with a CLIP-based image prior that natively supports image-conditioned generation.
Images are generated at 512 ×512 resolution with 100 denoising steps, using an image weight of
0.7, a guidance scale of 4.0, a prior guidance scale of 4.0, and 25 prior inference steps.(4) SDXL
+ Conceptrol[ 11]: We use SDXL augmented with Conceptrol, an advanced image conditioning
method. Images are generated at 1024 ×1024 resolution with 30 denoising steps, using a condition
scale of 1.0, a guidance scale of 6.0, and a warmup ratio of 0.2.
For Q&A tasks, we use two vision-language models (VLMs):(1) Qwen2.5-VL[ 37]: We experiment
with Qwen2.5-VL, a vision-language model from the Qwen family with native dynamic resolution
support.(2) LLaVA-1.6-Mistral-7B[ 19]: We also use LLaV A-1.6-Mistral-7B, a 7B-parameter vision-
language model built on Mistral that supports dynamic high-resolution image inputs. All other
hyperparameters follow the default settings from each model’s official implementation.
D.3 Description of the Evaluation Metrics
For attack evaluation, we use standard MIA metrics, including AUROC, ACC, and TPR@5%FPR.
AUROC measures the area under the ROC curve and captures the overall separability between
members and non-members across all thresholds. ACC reports the classification accuracy at the
optimal decision threshold. TPR@5%FPR measures the true positive rate when the false positive rate
is constrained to 5%, reflecting performance under a strict false-alarm budget. We additionally report
the Retrieval Success Rate (RSR) to measure the attack’s effectiveness at retrieving member images.
For utility evaluation, we report CLIP-T, CLIP-I (key metric for class-labeled dataset), and FID for
T2I tasks and response accuracy for Q&A tasks, following the literature [22, 6].
D.4 Description of the Attack Baselines
Traditional text-RAG MIAs [ 23,20] are not directly applicable to our setting due to the incompatibility
of T2I generators with question-answering tasks and the inability to embed images into text queries.
We therefore adapt these baselines following the query decomposition paradigm ofImageAuditor.
The naïve baseline includes only the extraction segment: we generate candidate captions and select
18

SDXL+IPA SD1.5+IPA KandinskySDXL+Conc.0.00.20.40.60.81.0Rate Score
(a) T2I-Stanford DogsQwen2.5-VL LLaV A-v1.60.00.20.40.60.81.0
(b) Q&A-MSCOCOAUROC ACC TPR@5%FPRFigure 9: ImageAuditor remains effective across different generators for both T2I and Q&A tasks.
SDXL+Conceptrol reduces TPR@5%FPR due to its defensive and more creative generation behavior.
However, it still achieves around 30% of TPR@5%FPR, which significantly outperforms random
guessing (5%) and is considered an effective attack under standard MIA literature.
Table 9: Impact of extraction segment type on ImageAuditor against IRAG for the T2I task. BLIP and
Qwen-VL captions yield stronger attacks, as better alignment between the image and the extraction
segment more effectively elicits copy-like behavior from T2I generators.
Extraction Segment
TypeMSCOCO WikiArt Stanford Dogs
AUC ACC TPR 5%AUC ACC TPR 5%AUC ACC TPR 5%
Generic Cap. 0.71 0.70 0.38 0.75 0.72 0.39 0.83 0.76 0.49
BLIP Cap. 0.86 0.82 0.420.870.800.540.88 0.83 0.51
Qwen-VL Cap.0.91 0.85 0.55 0.87 0.820.490.90 0.84 0.65
those with high reward (e.g., similarity to the test image), following our framework. Since these
captions semantically describe the test image, this baseline can still retrieve some member images in
the T2I setting, providing a meaningful comparison point.
We additionally adapt PoisonedRAG-B [ 43], PoisonedRAG [ 43], and BadRAG [ 39] as baseline
attacks. All three are originally designed for text-RAG poisoning, and we adapt them by treating
retrieval segment construction as analogous to their adversarial passage optimization. Specifically,
we apply their respective objectives to construct retrieval segments for each test image, while keeping
the extraction segment and scoring rule identical to ImageAuditor for a fair comparison. The three
methods differ primarily in their optimization objective.
(1) PoisonedRAG-B[ 43] is the gradient-free variant of PoisonedRAG. Rather than optimizing Tretvia
token-level gradients, it directly constructs the retrieval segment by extracting content from the target
sample. In our case, we use the generated image caption as Tret. In the T2I setting, this differs from
the Naïve baseline in that Naïve uses no retrieval segment, whereas PoisonedRAG-B∗employs two
separate captions—one as the extraction segment and another as the retrieval segment.
(2) PoisonedRAG[ 43] employs HotFlip [ 9] to optimize the retrieval segment Tretsuch that the
embedding of the full query is close to that of the target image. Specifically, given a target embedding
et, a query encoderE(·), and a fixed extraction segmentT ext, the optimization objective is:
T∗
ret= arg max
Tretsim 
E(T ext⊕T ret),et
,(2)
where sim(·,·) denotes cosine similarity and ⊕denotes concatenation. At each iteration, HotFlip
selects a random token position and replaces the token with the candidate yielding the largest
gradient-based score improvement.
(3) BadRAG[ 39] adopts a contrastive optimization objective (COP) that distinguishes the target
from a set of shadow (negative) samples. Let q=E(T ext⊕T ret)denote the query embedding and
{e(i)
n}N
i=1denote shadow embeddings drawn from the shadow set. The InfoNCE-style objective is:
T∗
ret= arg max
Tretlogexp(sim(q,e t)/τ)
exp(sim(q,e t)/τ) +PN
i=1exp(sim(q,e(i)
n)/τ),(3)
19

Table 10: The shadow dataset improves results when the target embedding space is unknown. Here,
ViT-L/14 is used as the shadow embedding model.
Type MSCOCO WikiArt Stanford Dogs
AUC ACC TPR 5%AUC ACC TPR 5%AUC ACC TPR 5%
No 0.75 0.70 0.31 0.64 0.64 0.18 0.72 0.68 0.26
Yes 0.78 0.75 0.32 0.70 0.67 0.34 0.75 0.70 0.33
Avg Kmeans Max
MSCOCO0.000.250.500.751.00Rate Score
Avg Kmeans Max
WikiArt0.000.250.500.751.00
AUROC ACC TPR@5%FPR
(a) with access to ViT-H/14 (standard setting)
Avg Kmeans Max
MSCOCO0.000.250.500.751.00Rate Score
Avg Kmeans Max
WikiArt0.000.250.500.751.00
AUROC ACC TPR@5%FPR (b) with access to ViT-L/14 (restrictive setting)
Figure 10: K-means-based aggregation yields the most stable attack results across various settings.
where τis a temperature hyperparameter. This formulation encourages the retrieval segment to be
selectively retrieved for the target while avoiding retrieval for negatives. The optimization is solved
using HotFlip-style token-level updates.
D.5 Hyperparameter Sensitivity
The attacker is allowed M×N queries per test image, where Mis the number of extraction segments
andNis the number of retrieval segments per extraction segment. We set M=N= 2 by default
to balance attack effectiveness against query and optimization costs. Using multiple extraction and
retrieval segments reduces sensitivity to any single choice. As shown in Figure 11, larger values
yield only marginal gains and even a single query (i.e., M=N= 1 ) already achieves competitive
performance. For the extraction segment, we use Qwen-VL to generate and rank 5 candidate captions
in the T2I setting, and manually design structured prompts in the Q&A setting. For the retrieval
segment, we set the token length to 8, which is sufficient for the attack to saturate (see Figure 5(c)).
We further sample a small shadow set of 32 images (disjoint from the database) for the contrastive
reward, which is practical to collect from public sources. We optimize each retrieval segment for
20 iterations in the T2I setting and 80 iterations in the Q&A setting, due to differences in extraction
prompt designs. As shown in Figure 12, our designs provide a favorable efficiency–effectiveness
trade-off, with diminishing returns beyond this point. The remaining RGPO parameters take standard
values from the policy-optimization literature: τ= 1 (a neutral temperature), ρ= 0.2 (a common
choice in importance sampling methods), and β= 0.5 (a balanced adaptation rate). We found these
defaults to be robust across all datasets, and therefore did not perform dedicated sweeps over them.
D.6 Description of the Attack Prompts.
We use the following prompts in our attack pipeline:
Prompt for Extraction Segment Generation in the T2I Setting
You are an image captioning system. Given an input image, generate 5 concise and diverse captions.
Requirements:
• Write all captions in English only. Do not use any other language.
• Each caption should describe the main scene and key objects.
• Keep each caption short and natural (similar to MSCOCO style).
•Use different wording and focus on slightly different aspects (e.g., objects, actions, relationships, or
context).
• Avoid repeating the same sentence structure or phrases.
• Do not include unnecessary details or long descriptions.
20

1×1 2×2 3×3
MSCOCO0.000.250.500.751.00Rate Score
1×1 2×2 3×3
WikiArt0.000.250.500.751.00
AUROC ACC TPR@5%FPRFigure 11: Increasing the number of queries
(M×N ) per test case slightly improves the at-
tack performance of ImageAuditor. Our attack
remains effective with a single query due to the
RGPO mechanism.
10 20 30
MSCOCO0.000.250.500.751.00Rate Score
10 20 30
WikiArt0.000.250.500.751.00
AUROC ACC TPR@5%FPRFigure 12: Increasing the number of RGPO it-
erations slightly improves the results (efficiency
vs. effectiveness trade-off), and ImageAuditor
remains effective even with a small number of
iterations (e.g., 10).
Table 11: In both standard and restrictive settings, a stronger scoring model improves attack perfor-
mance, as it more accurately captures fine-grained image semantics.
Dataset Scoring model with access to ViT-H/14 with access to ViT-L/14
AUC ACC TPR 5% AUC ACC TPR 5%
MSCOCOEvaCLIP 0.87 0.85 0.32 0.78 0.75 0.32
CLIP ViT-H/140.91 0.85 0.55 0.82 0.78 0.39
WikiArtEvaCLIP 0.81 0.77 0.32 0.68 0.68 0.27
CLIP ViT-H/140.87 0.82 0.49 0.69 0.71 0.41
DogsEvaCLIP 0.88 0.82 0.53 0.75 0.70 0.33
CLIP ViT-H/140.90 0.84 0.65 0.77 0.73 0.41
Output format:
1. ...
2. ...
3. ...
4. ...
5. ...
Extraction Segment in the Q&A Setting
Describe each retrieved image. Format as ‘Image 1:’, ‘Image 2:’, etc.
Prompt for Details Extraction in the Q&A Setting
You are an image captioning system.
Given an input image, write one detailed natural caption.
Requirements:
• Describe visible objects, text, scene, attributes, and layout when helpful.
• Stay faithful to the image.
• Do not speculate beyond what is visible.
• Do not output numbering, explanations, or extra text.
• Output exactly one caption.
E Discussion
E.1 Generalization to multi-modal inputs.
Our attack also extends to IRAG systems that take image-text pairs as input. Specifically, we can
use the image as the retrieval segment and the entire text as the extraction segment, which will
further improves attack performance due to the continuous nature of the pixel space. More broadly,
this decomposition applies to other multimodal RAG variants, suggesting that the vulnerability is
fundamental to cross-modal retrieval.
21

Figure 13: Visual illustration of ImageAuditor on test images from the MSCOCO dataset under
IRAG for the T2I task. Our method performs multiple queries and applies K-means clustering to the
resulting MIA scores for selective aggregation.
E.2 Responsible Deployment of ImageAuditor
ImageAuditor should be deployed with care to avoid false accusations. First, because the MIA
score distribution varies slightly across datasets (see Table 6), auditors should calibrate the decision
threshold on a small set of known non-members from the same domain and select an operating
point with a controlled false-positive rate (e.g., FPR ≤5% or even FPR ≤1%). Second, single-query
decisions can be noisy; auditors should issue multiple queries and apply the K-means aggregation in
Sec. 3.4 to retain only high-confidence ones, treating low-agreement cases as inconclusive. Finally,
MIA outputs provide statistical evidence rather than definitive proof, and we recommend using
ImageAuditor as a screening tool to flag candidates for further investigation rather than as the sole
basis for accusation.
22

Figure 14: Visual illustration of ImageAuditor on test images from the WikiArt dataset under IRAG
for the T2I task. Our method performs multiple queries and applies K-means clustering to the
resulting MIA scores for selective aggregation.
F Qualitative Analysis
Figure 13–15 visualize our attack results on the MSCOCO, WikiArt, Dogs datasets under the T2I
setting, and Figure 16 visualizes our attack results on the MMQA dataset under the Q&A setting.
Our method effectively distinguishes members from non-members by eliciting distinct generation
behaviors. For example, for the member in the first column of Figure 14 (a painting of a Venetian
palace),ImageAuditorsuccessfully triggers the copy-like behavior of the diffusion model: across
all three queries, the generated images closely replicate the reference image’s composition, color
palette, and architectural details, yielding consistently high similarity scores ( ∼0.85). In contrast,
for the non-member in the last column of Figure 14 (a painting of figures in a garden), the generator
produces images that share semantic content with the caption (i.e., a mother and daughter sitting in a
garden) but differ substantially from the reference in visual composition, lighting, and stylistic details.
As a result, the similarity scores remain consistently lower ( ∼0.67–0.71). This contrast confirms
that our caption-based extraction strategy reliably amplifies the membership signal: members trigger
near-verbatim reconstruction through the diffusion model’s copy-like behavior, while non-members
yield only semantically aligned but visually divergent outputs.
23

Figure 15: Visual illustration of ImageAuditor on test images from the Stanford Dogs dataset under
IRAG for the T2I task. Our method performs multiple queries and applies K-means clustering to the
resulting MIA scores for selective aggregation.
G Computation Resources
All experiments were conducted on three single-GPU systems, each equipped with an NVIDIA
A6000 GPU with 48GB of VRAM. We did not use multi-GPU training or distributed execution. Both
attack optimization and retrieval-augmented generation were performed on a single GPU per run.
24

Figure 16: Visual illustration of ImageAuditor on test images from the MMQA dataset under IRAG
for the Q&A task. Our method performs multiple queries and applies K-means clustering to the
resulting MIA scores for selective aggregation.
25