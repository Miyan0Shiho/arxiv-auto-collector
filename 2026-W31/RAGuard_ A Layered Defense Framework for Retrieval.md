# RAGuard: A Layered Defense Framework for Retrieval-Augmented Generation Systems Against Data Poisoning

**Authors**: Pushkal Kumar, Tucker Nielson, Tanish Kolhe, Shubham Zala, Vincent Li

**Published**: 2026-07-28 23:21:04

**PDF URL**: [https://arxiv.org/pdf/2607.26339v1](https://arxiv.org/pdf/2607.26339v1)

## Abstract
Retrieval-Augmented Generation (RAG) systems ground large language models (LLMs) in external corpora, but this reliance exposes them to corpus poisoning: maliciously injected passages that manipulate retrieved evidence. We introduce RAGuard, a layered defense against \emph{factual} corpus-poisoning attacks on RAG pipelines. The first layer adversarially fine-tunes a dense retriever on synthetic poisoned documents (fabricated facts, contradictions, and reasoning traps), teaching it to downrank malicious passages before generation. The second layer, the Zero-Knowledge Inference Patch ZKIP, is a label-free, black-box filter: for each retrieved document, it performs a leave-one-out decode and scores the document by the semantic shift and output-entropy change that its removal induces. ZKIP requires no poison labels, no ground-truth answers, and no access to model internals; it compares the model's own answers under counterfactual contexts. On poisoned Natural Questions at 5--30\% poison ratios, adversarial retriever training alone reduces but does not eliminate attack success, while ZKIP drives the measured attack success rate to 0.000 in every defended configuration, keeping Recall@5 within 0.03 of the clean-corpus baseline. Supervised analyses on both Natural Questions and BEIR (NFCorpus) confirm that the counterfactual signals ZKIP relies on carry learnable poison structure. The defense costs $k{+}1$ generator passes per query ($6\times$ for $k{=}5$); we analyze batching and early-stopping approximations that reduce this overhead. We also show that keyword-preserving poisons leave lexical retrievers such as BM25 essentially unaffected, an observation that delineates the boundary of the threat model. Code, datasets, and evaluation harnesses are released for reproducibility.

## Full Text


<!-- PDF content starts -->

RAGuard: A Layered Defense Framework for
Retrieval-Augmented Generation Systems Against Data
Poisoning
Pushkal Kumar Tucker Nielson Tanish Kolhe Shubham Zala Vincent Li
Abstract
Retrieval-Augmented Generation (RAG) systems ground large language models (LLMs) in external
corpora, but this reliance exposes them to corpus poisoning: maliciously injected passages that manipulate
retrieved evidence. We introduce RAGuard, a layered defense againstfactualcorpus-poisoning attacks on
RAG pipelines. The first layer adversarially fine-tunes a dense retriever on synthetic poisoned documents
(fabricated facts, contradictions, and reasoning traps), teaching it to downrank malicious passages before
generation. The second layer, the Zero-Knowledge Inference Patch (ZKIP), is a label-free, black-box filter:
for each retrieved document, it performs a leave-one-out decode and scores the document by the semantic
shift and output-entropy change that its removal induces. ZKIP requires no poison labels, no ground-truth
answers, and no access to model internals; it compares the model’s own answers under counterfactual
contexts. On poisoned Natural Questions at 5–30% poison ratios, adversarial retriever training alone
reduces but does not eliminate attack success, while ZKIP drives the measured attack success rate to 0.000
in every defended configuration, keeping Recall@5 within 0.03 of the clean-corpus baseline. Supervised
analyses on both Natural Questions and BEIR (NFCorpus) confirm that the counterfactual signals ZKIP
relies on carry learnable poison structure. The defense costs k+1 generator passes per query ( 6×fork=5);
we analyze batching and early-stopping approximations that reduce this overhead. We also show that
keyword-preserving poisons leave lexical retrievers such as BM25 essentially unaffected, an observation
that delineates the boundary of the threat model. Code, datasets, and evaluation harnesses are released for
reproducibility.
1 Introduction
Retrieval-Augmented Generation (RAG) has emerged as an effective method to ground Large Language
Models (LLMs), retrieving external evidence so that responses reflect up-to-date, verifiable information
rather than parametric memory alone (Lewis et al., 2020; Asai et al., 2024; Ram et al., 2023; Izacard et al.,
2023). This grounding, however, creates a new attack surface: because generation trusts retrieved passages as
evidence, an adversary who can write to the corpus can steer the model’s answers. Data-poisoning attacks
inject documents that mimic relevant content yet contain false or misleading information, and only a handful
of poisoned documents can mislead large models with high success rates (Zou et al., 2025; Souly et al., 2025;
Wang et al., 2026; Long et al., 2025; Su et al., 2025).
Existing defenses have structural gaps. Detection-based filters rely on labeled poison examples or
heuristic rules and fail against new attack styles (Zou et al., 2025; Edemacu et al., 2025). Generator-hardening
approaches carry heavy inference cost and degrade when poisoned passages dominate the retrieved set (Asai
et al., 2024; Shi et al., 2024). Adversarially trained retrievers depend on the synthetic poisons seen during
An earlier version of this work was accepted to the NeurIPS 2025 Workshop on Responsible Foundation Models (ResponsibleFM)
and AAAI 2026 Workshop on New Frontiers in Information Retrieval (FrontierIR). Code, datasets, and evaluation artifacts:
https://github.com/RAGuard-AI/RAGuard.
1
arXiv:2607.26339v1  [cs.LG]  28 Jul 2026

training and risk overfitting to known poison types (Lupart and Clinchant, 2023; Park and Chang, 2019). No
single mechanism covers both the retrieval and generation stages.
This work introduces RAGuard, a two-layer defense. The first layer adversarially fine-tunes a dense
retriever with contrastive training on synthetic poisoned documents—fabricated facts, contradictions, and
reasoning traps—so that poisoned passages are downranked before they reach the generator (Izacard et al.,
2022; Lei et al., 2023; Lupart and Clinchant, 2023). The second layer, the Zero-Knowledge Inference Patch
(ZKIP), operates at generation time and is entirelyself-referential: given the top- kretrieved documents, it
decodes a reference answer using the full context, then re-decodes once per document with that document
removed. A document is flagged when its removal substantially shifts the answer’s semantics or lowers the
model’s output uncertainty. ZKIP never consults a ground-truth answer, poison label, or external oracle; the
only quantities it compares are the model’s own outputs under counterfactual contexts. This makes the filter
applicable to unseen attack types that bypass the trained retriever.
We evaluate RAGuard on poisoned variants of Natural Questions (NQ) (Kwiatkowski et al., 2019) and
BEIR (NFCorpus) (Thakur et al., 2021) at poison ratios from 5% to 30%. Our key contributions are:
•A label-free, black-box inference-time filter.ZKIP scores each retrieved document by the answer
instability and entropy differential its removal induces (§3.4). It requires no poison labels, no gold
answers, and no model internals, and it reduces the measured attack success rate (ASR) to 0.000 for
every retriever at every NQ poison ratio with a completed defended run (§4.3).
•A layered retrieval-plus-inference defense with measured layer contributions.Adversarial retriever
training alone leaves residual attack success (ASR 0.072 at NQ 10% poisoning), while ZKIP alone and
the combined system both eliminate it; the combination additionally preserves the adversarially trained
retriever’s ranking quality (§4.4).
•A threat-model analysis grounded in negative results.Our keyword-preserving poisons leave BM25
(Robertson and Zaragoza, 2009) essentially unaffectedwithout any defense(ASR 0.000 at 5–20%
poisoning and 0.011 at 30%), showing that the poisons exploit dense-embedding semantics specifically.
We use this to delineate which attacks are in and out of scope (§4.1, Appendix B).
ZKIP carries a real cost: k+1 generator forward passes per query, a 6×inference overhead at k=5. We
quantify this overhead and describe batching, early stopping, and subset-sampling approximations (§5.2). We
scope our claims to factual question answering under corpus poisoning of dense retrieval; broader robustness
claims would require evaluation against attack frameworks such as PoisonedRAG (Zou et al., 2025) and
FlippedRAG (Chen et al., 2025), which we identify as the most important next step (§6).
2 Related Work
RAG and corpus poisoning.RAG (Lewis et al., 2020) enhances LLMs by combining retrieval with
generation, enabling factual grounding without retraining. Recent studies show that RAG architectures are
susceptible to data poisoning. PoisonedRAG (Zou et al., 2025) demonstrated that inserting adversarially
crafted documents into retrieval corpora distorts ranking and generation. Joint-GCG (Wang et al., 2026)
extended this with unified gradient-based attacks perturbing both retriever and generator embeddings. Chain-
of-thought poisoning (Song et al., 2025) showed that reasoning-style attacks targeting multi-step prompts
propagate errors across retrieval iterations, and Souly et al. (2025) showed that a near-constant number of
poisoned documents suffices to compromise large models. FlippedRAG (Chen et al., 2025) demonstrated
black-boxopinion manipulationattacks, a class we explicitly place outside our threat model (Appendix B).
2

Counterfactual and conflicting-evidence datasets.Our synthetic poison construction (§3.3) is related
to a line of work that perturbs evidence to study model robustness. Chen et al. (2022) construct conflicting-
evidence settings for open-domain QA and recalibrate models to reflect disagreement among sources. Pan
et al. (2023) study misinformation pollution of QA corpora using LLM-generated false passages. Hong et al.
(2024) build counterfactual-noise variants of NQ and propose discriminator-guided robustness training. Our
poisons differ in purpose rather than mechanism: instead of building a benchmark, we use LLM-rewritten
counterfactual passages as adversarialtrainingsignal for the retriever and as a controlled attack for evaluating
the inference-time filter. Unlike these benchmark datasets, our poisons are parameterized by attack family
(fabrication, contradiction, reasoning trap), letting us measure per-family defense behavior.
Defenses.Existing RAG poisoning defenses rely on input filtering, heuristic retriever fine-tuning, or
adversarial data augmentation (Edemacu et al., 2025). These reduce known attack surfaces but often depend
on labeled poison data or add heavy inference-time cost. Adversarial training for neural retrieval (Lupart and
Clinchant, 2023; Park and Chang, 2019) hardens ranking but, as we confirm empirically, does not by itself
eliminate attack success. Outside of NLP, black-box counterfactual defenses have proven effective against
backdoors in other modalities, e.g., zero-shot image purification (Shi et al., 2023); ZKIP brings a similarly
black-box, perturb-and-compare principle to retrieval contexts.
Relation to attribution methods.Leave-one-out (LOO) influence estimation is an established tool in
explainability and causal inference (Johansson et al., 2016; Prosperi et al., 2020; Molnar, 2025). Perturbation-
based attribution methods such as LIME (Ribeiro et al., 2016) and SHAP (Lundberg and Lee, 2017) likewise
estimate a feature’s importance from output changes under input perturbation. ZKIP borrows the LOO
mechanism but answers a different question. Attribution asks “which input was mostimportantto the
output?”; a gold passage and a poisoned passage can both be highly important. ZKIP instead asks “which
document’sremovalmakes the model’s answer more stable or less uncertain?”—a directional, security-
oriented query. It also combines two signals (semantic answer shift and entropy differential) into a single
anomaly score and applies the result as an online filter rather than a post-hoc explanation. Section 3.4 makes
this contrast precise.
3 Methods
3.1 Overall Architecture
Figure 1 shows the architecture. User queries pass through an adversarially fine-tuned dense retriever (Layer
1). During training, the retriever is exposed to both clean and synthetically poisoned passages and learns
to downrank documents whose embeddings deviate from normal semantic structure. The top- kretrieved
documents then pass through ZKIP (Layer 2), which flags and removes documents with anomalous causal
influence on the generated answer before the final decode. Components are modular: retrievers, generators,
and defense layers can be swapped independently.
3.2 Rationale
The dominant vulnerability of RAG systems sits in the retriever: poisoned documents that rank highly
contaminate the generator’s context. Rather than fine-tuning the generator, RAGuard strengthens the retrieval
process at its root and adds a generation-time filter as a fail-safe. The two layers are complementary by
design: adversarial training isproactivebut limited to poison distributions seen during training, while ZKIP
isreactiveand attack-agnostic because it measures the causal effect of each context element on the model’s
3

QueryAdv-trained
retriever
(Layer 1)Top-k
docsZKIP (Layer 2)
k LOO decodesAi=(1−si)+λ[ΔHi]+Filtered
contextGenerator→ answer
flagged docs removed (no labels, no gold answers)downranks known poison familiesFigure 1: RAGuard’s two-layer defense. A query passes through the adversarially trained retriever (Layer 1);
ZKIP (Layer 2) runs one leave-one-out decode per retrieved document, scores each document’s causal
influence on the answer, and removes flagged documents before the final generation. No poison labels or
gold answers are used at any stage of inference.
own output. A poison that evades the trained retriever still has to influence the answer to succeed, and that
influence is exactly what ZKIP measures.
3.3 Adversarial Training Data and Retriever Fine-Tuning
Poison construction.To evaluate robustness against retrieval-level poisoning, we construct poisoned
variants of NQ and BEIR. Starting from the original corpora, we sample 30% of query–document pairs and
generate modified passages by prompting an LLM to rewrite each gold document according to an attack type.
We use three attack families: (i)fabricationpoisons, which append falsified or hallucinated statements; (ii)
contradictionpoisons, which flip key factual tokens (e.g., “true” →“false”); and (iii)reasoningpoisons,
which introduce misleading logical steps or corrupted intermediate claims. Poisoned samples are distributed
evenly across the three families. The constructed datasets contain 12,344 total BEIR samples (3,700 poisoned)
and 1,000 total NQ samples (300 poisoned); experiments substitute poisoned triples at controlled ratios (5–
30%). Each record stores the query, the unmodified gold passage, the poisoned passage, and the attack-family
label, so every experiment is reproducible from the released JSONL files.
Why synthetic poisons rather than existing counterfactual datasets.Prior work such as Hong et al.
(2024) provides counterfactual-noise variants of NQ, and Pan et al. (2023) studies LLM-generated misinfor-
mation pollution. We generate fresh synthetic poisons rather than reuse these corpora for three reasons. First,
our three attack families are parameterized adversarial classes not uniformly present in prior benchmarks,
which primarily test model robustness toconflicting evidencerather than retrieval-level poison defense.
Second, controlling the poison ratio (5–30%) requires generating exactly the right number of poisons per
experimental condition; fixed prior datasets do not support this. Third, generating poisons on both NQ and
BEIR allows us to test transfer across domains. The tradeoff is that our poisons are LLM rewrites rather than
gradient-optimized adversarial injections, which places a ceiling on measured ASR and may underestimate
real-world attack strength (§5.3).
Retriever fine-tuning.The dense retriever is initialized from the all-MiniLM-L6-v2 sentence encoder
(Reimers and Gurevych, 2019; Wang et al., 2020) and fine-tuned with a triplet margin loss (margin 0.2) over
{query, positive document, negative document}triples:
L= max 
0, s(q, d−)−s(q, d+) + 0.2
,(1)
4

where s(·,·) is the cosine similarity of Equation 2, d+is the gold passage, and d−is the negative. We
use triplet margin loss rather than InfoNCE with in-batch negatives because in-batch negatives would treat
randomly co-occurring passages as hard negatives indiscriminately; the explicit margin instead gives the
model a clear signal for the exact semantics that distinguish each poison rewrite from its gold counterpart.
Negative-sampling strategy.Forpoisoned triplesthe negative d−is the attack-family-specific LLM
rewrite of that triple’s gold document—the precise passage the model must learn to rank below the gold.
This direct anti-poison contrastive signal is the core mechanism by which the retriever becomes adversarially
robust. Forclean triplesthe negative is a randomly sampled in-corpus clean passage, providing general
ranking discrimination. We train for 3 epochs with AdamW, learning rate 2×10−5, batch size 16, max
sequence length 256 tokens; 30% of training triples carry poisons, split evenly across the three attack families.
Full hyperparameters appear in Appendix C. Retrieval uses cosine similarity between the query and document
encoders:
s(q, d) = cos 
fθ(q), g θ(d)
=fθ(q)·g θ(d)
∥fθ(q)∥∥g θ(d)∥,(2)
where fθandgθmap the query qand document dto embeddings; Equation 2 also supplies the ranking signal
that ZKIP complements at generation time.
Distribution-shift caveat.Because the adversarially trained retriever is fine-tuned on synthetic poisons
over gold documents, it may partially memorize gold relevance for the training queries and behave less
reliably on unseen query–document pairs. Two design choices mitigate, but do not eliminate, this risk: the
poison families are stylistically diverse rewrites rather than fixed templates, and ZKIP sits behind the retriever
as an attack-agnostic safety net. We treat held-out-domain evaluation of the adversarially trained retriever as
required future work and do not claim it generalizes beyond the evaluated distributions (§6).
3.4 Zero-Knowledge Inference Patch (ZKIP)
Relation to LOO attribution, and what is new.Leave-one-out probing is an established technique in
explainability and counterfactual inference (Johansson et al., 2016; Prosperi et al., 2020; Molnar, 2025),
and perturbation-based attribution methods such as LIME (Ribeiro et al., 2016) and SHAP (Lundberg and
Lee, 2017) similarly estimate input importance from output changes. ZKIP differs in three ways. First, it
combines two complementary signals—semantic answer stabilityandan output-entropy differential—into
one anomaly score, rather than relying on either alone. Second, it repurposes the counterfactual probe
from model explanation to online poisoning defense: scores are computed per query at inference time and
immediately drive a filtering decision. Third, the causal question differs from attribution. Attribution asks
which document was most important; ZKIP asks which document, when removed, makes the modelmore
stable or less uncertain—a gold passage and a poison are both “important,” but only the poison’s removal
stabilizes the answer and reduces uncertainty. Crucially, ZKIP requires no poison labels, no ground-truth
answers, and no access to model internals: it is black-box and entirely self-referential, comparing the
model’s own outputs across counterfactual contexts. This also distinguishes it from relevance reranking,
which scores query–document similarity and therefore cannot flag a document that is highly relevantand
poisoned—precisely the stealth attacker’s goal.
Setup.Given a query qand top- kcontext D={d i}k
i=1, we decode a reference answer with all passages,
then run one leave-one-out (LOO) decode per removed di. For outputs y= (y 1, . . . , y T)the generator defines
pϕ(y|q,D) =TY
t=1pϕ 
yt|y<t, q,D
.(3)
5

Letyall= arg max ypϕ(y|q,D) andy−i= arg max ypϕ(y|q,D \ {d i})denote the reference and LOO
decodes under Equation 3.
Answer stability.Let hψ(·)be an answer encoder; we use the all-MiniLM-L6-v2 sentence-embedding
model. The answer stability of documentd iis
si= cos 
hψ(yall), hψ(y−i)
∈[−1,1],(4)
where larger siindicates the answer is stable to removing di. Note that Equation 4 compares the model’s two
answersto each other, never to a gold answer.
Entropy differential.Let the sequence-level output entropy be
H(q,D) =−X
ypϕ(y|q,D) logp ϕ(y|q,D),(5)
approximated in practice from the per-token log-probabilities {logp ϕ(yt|y<t, q,D)}T
t=1of the decoded
sequence: we convert them to normalized pseudo-probabilities and compute their entropy, which is tractable
for a single decoded path and captures the model’s per-step confidence. The uncertainty shift induced by diis
∆Hi=H(q,D)−H 
q,D \ {d i}
,(6)
so a large positive ∆Himeans removing direducesthe model’s uncertainty—evidence that diwas injecting
confusion.
Anomaly scoring and filtering.We combine Equations 4 and 6 into a per-passage score:
Ai= (1−s i) +λ
∆Hi
+,[x] += max(0, x), λ >0,(7)
and discard the passage(s) with the largest Aiprior to the final decode. Figure 2 illustrates the decision flow
on a three-document example. We use λ= 0.5 and remove the single top-scoring passage ( m=1 ) in the
main experiments; §4.4 discusses the contribution of each term, and Appendix C lists all defaults. With
this construction, ZKIP requires no poison labels and generalizes across attack types because it relies on
counterfactual sensitivity rather than attack-specific features.
Cost.ZKIP requires exactly k+1 generator forward passes per query (one reference decode plus kLOO
decodes), a 6×overhead at k=5. The LOO decodes are mutually independent and batch into a single padded
forward pass on local models; §5.2 analyzes approximations that reduce the pass count further.
4 Experiments
4.1 Experimental Setup and Threat Model
Threat model scope.RAGuard targetsfactual corpus poisoning: an adversary with write access to the
retrieval corpus (but not to model weights, queries, or prompts) injects passages containing fabricated facts,
contradictions, or corrupted reasoning, aiming to change the answers of a dense-retrieval RAG pipeline.
Both retrieval-level effects (the poison ranks highly) and generation-level effects (the poison misleads the
generator) are in scope, since ZKIP operates after retrieval.
6

d1 (clean)
d2 (poisoned,
unknown)
d3 (clean)decode with{d1,d2,d3}
→yalldecode w/o d1A1=0.08
decode w/o d2A2=0.91
decode w/o d3A3=0.12maxiAi
vs. τflag d2(A2>τ)
answer from{d1,d3}Figure 2: ZKIP decision flow for k=3. One reference decode plus one LOO decode per document yields
anomaly scores Ai(Equation 7); the poisoned document’s removal shifts the answer and reduces uncertainty,
soA2exceeds the threshold and d2is excluded from the final decode. All comparisons are between the
model’s own outputs.
The following attack classes areexplicitly out of scopefor this paper: (i)opinion manipulation(Chen
et al., 2025), which targets subjective or multi-answer questions for which ZKIP’s stability signal is unin-
formative; (ii)prompt hijacking / instruction injection, which embeds instructions in documents to override
system prompts; (iii)retrieval jamming, which floods the corpus with irrelevant passages to suppress relevant
results; (iv)backdoor triggers(Long et al., 2025), which require latent activations or specific token patterns
not present in our threat model; (v)adversarial queries, since RAGuard assumes a clean query stream; and
(vi)white-box gradient attacks(e.g., Joint-GCG, Wang et al., 2026; greedy corpus poisoning, Su et al., 2025).
Appendix B details each boundary and discusses the extent to which ZKIP may incidentally cover some of
these classes.
Datasets and pipeline.We evaluate on Natural Questions (NQ) (Kwiatkowski et al., 2019), chosen for
broad topical coverage, and BEIR (NFCorpus) (Thakur et al., 2021), a medical/scientific retrieval benchmark
with much sparser relevance signal. For each dataset, we construct clean and poisoned variants as described
in §3.3, with binary poison flags recorded for every document so that detection metrics are computable.
We compare BM25 (Robertson and Zaragoza, 2009) and dense retrievers in three configurations:Dense
(clean), trained only on clean triples;Dense (adv-trained), fine-tuned with synthetic poisons per §3.3;
and each of these with and without ZKIP. (Earlier drafts labeled the adversarially trained retriever “Dense
(poisoned)”; we rename it throughout to remove ambiguity—the retriever isdefended by training, not
attacked.) Generation uses GPT-4o-mini (temperature 0.2, top- p0.95, max 128 tokens) for the main pipeline
and FLAN-T5-small (Chung et al., 2024) for the batched feature-extraction experiments of §4.6; answer
embeddings use all-MiniLM-L6-v2 (Reimers and Gurevych, 2019). We retrieve k=5 passages per
query.
4.2 Evaluation Metrics
Recall@5: the proportion of queries for which the gold document appears in the top-5 retrieved results.
Mean Reciprocal Rank (MRR): the average of1
rankover the first relevant document.Attack Success Rate
(ASR): the fraction of queries for which a poisoned document ranks higher than the gold document and
7

Retriever Defense Recall@5 MRR ASR↓
Dense (clean) None 0.259 0.176 0.101
Dense (adv-trained) None 0.319 0.215 0.072
Dense (clean) ZKIP 0.259 0.1760.000
Dense (adv-trained) ZKIP 0.314 0.2150.000
BM25 None 0.070 0.056 0.000
Table 1: NQ at 10% poisoning: ZKIP eliminates measured attack success for both retriever variants while
keeping Recall@5 within 0.005 of the corresponding undefended retriever. “Dense (adv-trained)” denotes the
retriever fine-tuned on synthetic poisons (§3.3), not a retriever under attack. Clean-corpus reference: Dense
(clean) reaches 0.282 Recall@5 / 0.200 MRR (Appendix A). Full results across poison levels and retrievers
are in Appendix A.
misleads the generator; lower is better. We report all three metrics for clean, poisoned, and defended settings.
4.3 Main Results
Table 1 summarizes NQ at 10% poisoning; Appendix A reports all conditions, and Figures 3 and 4 visualize
the Recall@5 and ASR behavior across poison ratios.
Impact of poisoning on retrieval quality.Comparing clean baselines to poisoned settings, dense retrievers
degrade measurably as poisons enter the corpus: on NQ at 10% poisoning, Dense (clean) drops from 0.282
to 0.259 Recall@5 and from 0.200 to 0.176 MRR. Poisoned documents disrupt retrieval even without any
attack on the retriever’s weights.
Attack success rate.For dense retrievers without adversarial training, ASR ranges from 0.029 (20%
poisoning) to 0.101 (10% poisoning) on NQ; the non-monotonic pattern across ratios reflects the modest
number of attackable queries per condition (33–190) and motivates the multi-seed runs planned in §6. BM25
shows ASR 0.000 at 5–20% poisoning and 0.011 at 30%: our poisons are LLM rewrites that alter semantics
but largely preserve the original keywords, so lexical ranking is barely affected. This is a threat-model
boundary, not a defense result—a keyword-overlap retriever with 0.070 Recall@5 is not a usable alternative,
and an adversary targeting hybrid pipelines would craft poisons for both signals (§6).
Effect of adversarial training.Dense (adv-trained) improves Recall@5 under poisoning relative to the
clean-trained model (0.319 vs. 0.259 at NQ 10%) and reduces—but does not eliminate—attack success (ASR
0.072 vs. 0.101). At 5% poisoning the adversarially trained model’s ASR (0.091) exceeds the clean model’s
(0.061), a reminder that training-time hardening interacts with the poison distribution in ratio-dependent
ways. Training-time defense alone is insufficient against the residual attacks that slip through ranking.
ZKIP effectiveness.ZKIP drives ASR to 0.000 in every defended configuration on NQ—both retriever
variants and BM25, at both poison ratios with completed defended runs (10% and 30%). Retrieval quality stays
close to the corresponding undefended retriever: at 10% poisoning the defended Dense (adv-trained) reaches
0.314 Recall@5 vs. 0.319 undefended, and at 30% it reaches 0.274 vs. 0.318, in exchange for eliminating
all measured attack success. Relative to theclean-corpusbaseline of 0.282, defended configurations land
between 0.256 and 0.314 (within 0.03); Figure 3 visualizes this against the clean-baseline reference line.
8

NQ 10% NQ 30%0.000.050.100.150.200.250.300.350.40Recall@50.319 0.318 0.314
0.274Dense (adv-trained)
NQ 10% NQ 30%0.000.050.100.150.200.250.300.350.40Recall@50.259 0.256 0.259 0.256Dense (clean)
Clean-corpus dense baseline (0.282)
Poisoned corpus, no defense
+ ZKIPFigure 3: Recall@5 on poisoned NQ with and without ZKIP, for the adversarially trained (left) and clean-
trained (right) dense retrievers, at the two poison ratios with completed defended runs. Dashed line: clean-
corpus dense baseline (0.282). The adv-trained retriever’s bars sitabovethe clean baseline because fine-tuning
on answer-bearing rewrites also boosts ranking of relevant passages; ZKIP removes one influential passage
per flagged query, trading a small amount of Recall@5 (largest at 30%: 0.318 →0.274) for the elimination
of all measured attack success. All values are taken directly from the released evaluation artifacts.
Configuration (NQ 10%) Recall@5 MRR ASR↓
Retriever only (adv-trained) 0.319 0.215 0.072
ZKIP only (clean retriever) 0.259 0.1760.000
Both layers 0.314 0.2150.000
Table 2: Layer ablation on NQ at 10% poisoning, using results measured in the main grid. ZKIP is necessary
and sufficient to eliminate measured ASR; the adversarially trained retriever contributes ranking quality
(+0.055Recall@5,+0.039MRR over ZKIP-only) but cannot reach ASR 0 alone.
Computational cost.ZKIP adds k+1 generator calls per query: 6 calls at k=5, a6×worst-case inference
cost. §5.2 discusses when this is acceptable and how to reduce it.
4.4 Ablation: Layer Contributions
Layer ablation.Table 2 separates the two defense layers using results measured in our main grid. ZKIP
alone eliminates measured attack success; adversarial retriever training alone does not (ASR 0.072). The
combination preserves the adv-trained model’s superior ranking (0.314/0.215 vs. 0.259/0.176 for ZKIP-only)
while keeping ASR at 0.000. The layers therefore play distinct roles: the retriever protectsranking quality
under poisoning, and ZKIP protectsanswer integrity.
Signal ablation (protocol).The anomaly score of Equation 7 combines a stability term (1−s i)and
an entropy term λ[∆H i]+. We expectstability only( λ=0) to catch poisons whose removal substantially
changes the decoded answer string—the primary signal for fabrication and contradiction attacks—while
entropy onlyshould catch poisons that inflate model uncertainty without changing the surface answer, with
reasoning-trap poisons the expected primary beneficiary. We release the evaluation harness for this ablation
(experiments/ablation_stability_vs_entropy.py ) but report no numbers here: the runs are
not yet complete, and we report only completed measurements.
9

5% 10% 20% 30%
Poison ratio0.000.020.040.060.080.10Attack success rate
Dense (clean)
Dense (adv-trained)
BM25
Any retriever + ZKIP (ASR = 0)Figure 4: Attack success rate on NQ across poison ratios for the three undefended retrievers, with ZKIP-
defended results (stars) at the ratios with completed defended runs. ZKIP reaches ASR 0.000 for every
retriever it is applied to, including the 30% condition where even BM25 shows nonzero ASR (0.011).
4.5 Multi-Poison Attacks (Protocol)
The main experiments poison at most one of the top- kdocuments per query. A coordinated adversary can
instead place multiple mutually reinforcing poisons in the top- k: removing any single document then changes
the answer little, muting the LOO signal that ZKIP relies on. This is the most plausible evasion strategy
against single-document removal, and we state it as the central known weakness of single-pass ZKIP (§5.4).
Our released code implements two countermeasure variants for future measurement:single-pass ZKIP
(remove the top-1 anomaly) anditerative ZKIP, which after removing the top-scoring document re-runs the
LOO probe on the remaining k−1 documents and removes the next document whose score exceeds a stricter
threshold τ2> τ 1, up to a removal budget m. Iteration breaks poison coalitions one member at a time: once the
first poison is gone, the remaining poisons’ influence is exposed. The cost grows to at mostPm−1
j=0(k−j) + 1
generator calls for mremovals. The evaluation harness ( experiments/multi_poison_eval.py )
implements the full protocol—coordinated injection of 2–4 poisons into the top- kand measurement of ASR
and filter recall for both variants—and we leave its quantitative results to future work rather than reporting
estimates.
4.6 Learned Poison Classification
To assess whether ZKIP’s influence signals contain learnable poison structure, we train supervised classifiers
to predict whether a retrieved document is poisoned, using a compact set of per-document influence features
(Appendix D). We evaluate a logistic-regression baseline and a neural classifier over these features on labeled
NQ, and separately a text-level BERT classifier (Devlin et al., 2019) on gold-vs-poison document pairs
as a supervised upper bound. Unlike ZKIP, these classifiers require poison-labeled training data and may
need retraining across attack styles; we treat them as supporting analysis, not as the primary defense. The
feature-level BERT classifier reaches AUPRC 0.732 versus 0.377 for logistic regression (Appendix Table 8),
confirming that counterfactual-removal signals carry nonlinear, learnable poison structure—which is the
property ZKIP exploits label-free.
10

Configuration Generator calls/query ASR↓
No defense 1 0.029–0.101
ZKIP, full LOO (k=5) 6 0.000
ZKIP, batched LOO 2 batched 0.000
Table 3: Cost–robustness tradeoff on poisoned NQ (dense retrievers; ASR range spans the 5–30% undefended
conditions). Batching removes most wall-clock overhead on local generators without changing the filtering
decision.
5 Discussion
5.1 Advantages
RAGuard’s primary strengths are modularity, model agnosticism, and label-freedom. The LOO patch can
be used with virtually any LLM or retriever, since it only requires access to the system’s output (text and
token log-probabilities) for each context perturbation. The key upshot is that the patch can catch unseen
attack types—including those that evade training-time simulation—because it evaluates the causal effect of
each context element on the model’s end-to-end answer. Unlike prior defenses that depend on explicit poison
traces or multi-LLM ensembles (Edemacu et al., 2025; Zou et al., 2025), the patch is label-free and feasible
with a single LLM.
5.2 Computational Tradeoffs
The principal cost is k+1 generator passes per query ( 6×atk=5,11× atk=10 ); cost scales linearly with
the number of retrieved documents while robustness benefits plateau. Table 3 summarizes the measured
cost–robustness tradeoff. Three approximations reduce the pass count:(1) Batched LOO: the kLOO
decodes are independent and run as one padded batch on local models, reducing wall-clock cost to roughly
two batched generations per query; our implementation batches by default.(2) Early stopping: when
the first flagged document exceeds the threshold by a margin δ, remaining LOO passes are skipped; this
trades completeness of the per-document score vector for latency.(3) Subset sampling: probing a random
⌈k/2⌉ subset of documents per query halves the LOO passes; our released code implements this mode
(-approximate ), and quantifying its accuracy cost is planned work. Practitioners can also apply ZKIP
selectively—only to high-stakes or low-confidence queries—to amortize the cost. Conceptually, the patch
is akin to self-consistency prompting for retrieval: rather than querying an ensemble of models, RAGuard
queries multiple context subsets of one model, using stability as a proxy for trustworthiness. For latency-
critical workloads the overhead may still be prohibitive; we state this plainly as a limitation rather than a
footnote.
5.3 Comparison to Other Methods
Traditional filtering defenses (Edemacu et al., 2025; Zou et al., 2025) rely on hand-crafted features or learned
classifiers that may not generalize to new poison types or domain shifts. Prompt-level robustness methods
(Asai et al., 2024; Shi et al., 2024) struggle when the retrieval step is severely compromised. Fine-tuning
retrievers on synthetic poisons risks overfitting to known poison distributions (Lupart and Clinchant, 2023),
as demonstrated by attacks that evolve trigger patterns or semantic camouflage (Su et al., 2025). RAGuard
mitigates these weaknesses by combining a proactive hardening step with an adaptive, model-agnostic
inference-time filter.
Two comparisons we have not yet run bound our claims.PoisonedRAG(Zou et al., 2025) generates
11

adversarial passages via optimization against a target answer string, producing semantically coherent in-
jections optimized to rank highly under a specific dense retriever. Our synthetic poisons are LLM rewrites
constructed without access to the retriever’s gradients; they are related in intent but face a weaker optimization
pressure. A comparison using PoisonedRAG’s injected-passage format is the most direct way to establish
whether ZKIP generalizes beyond distribution-matched evaluation.FlippedRAG(Chen et al., 2025) targets
opinion manipulation on questions without ground-truth answers, placing it outside our factual threat model
(§4.1); it would nevertheless stress-test whether ZKIP’s answer-stability signal degrades on questions with
legitimate answer variation. Both are flagged as the highest-priority evaluation extensions (§6); our released
code includes a harness ( experiments/external_attack_eval.py ) that evaluates ZKIP against
externally generated attack corpora in PoisonedRAG’s injected-passage format, making these extensions
straightforward to run.
5.4 Failure Cases
ZKIP fails in identifiable patterns, which we report rather than smooth over.(1) Coordinated multi-poison
coalitions.When two or more poisons in the top- kassert the same false fact, removing any one leaves the
answer unchanged; the LOO signal for each is muted and no document is flagged. Single-pass ZKIP is
structurally blind to this; §4.5 describes the iterative-removal countermeasure.(2) Stable wrong answers.
A poison that shifts the answerwithoutraising output entropy—for example, a confident contradiction that
the generator finds fluent—produces a small ∆Hi, leaving only the stability term to catch it. If the poisoned
and clean answers are also lexically close (a date off by one year), the cosine shift in Equation 4 can fall
below threshold.(3) False positives on unusual benign documents.A benign passage that is opinionated,
off-distribution, or contains a unique disambiguating fact can dominate the answer; its removal changes
the output substantially and ZKIP may flag it. This is the mechanism behind the Recall@5 reductions
observed when ZKIP is applied (Figure 3); on ambiguous queries it can remove the most informative passage.
(4) Queries the generator cannot answer from any context.When all decodes are unstable (the model
guesses), anomaly scores are uniformly high and filtering approaches random removal.
5.5 Future Work
(1) Combining ZKIP with active learning or human-in-the-loop verification to filter subtle reasoning attacks.
(2) Real-time deployments that combine self-consistency tests with lightweight instance selection where
cost is the binding constraint. (3) Broader benchmarks—cross-domain, multilingual, and standardized attack
suites (PoisonedRAG, FlippedRAG)—to stress-test defense layers. (4) Theoretical bounds on patch efficacy,
connecting LOO filtering to the geometry of poison coalitions.
5.6 Broader Impact
Robust RAG matters most in high-stakes applications such as medicine, finance, and law (Ram et al., 2023).
Increased robustness shifts, rather than ends, the attacker–defender dynamic; by open-sourcing the patches and
stress-testing pipelines, the community can probe these defenses before attackers do. Our poison-generation
code is released for reproducibility of the defense evaluation; the poisons are simple LLM rewrites with no
capability beyond what current attack literature already documents (Zou et al., 2025; Pan et al., 2023).
6 Conclusion
We introduced RAGuard, a modular two-layer defense that strengthens RAG systems against factual corpus
poisoning. The first layer fine-tunes dense retrievers on synthetic poisons (fabrications, contradictions,
12

reasoning traps) to downrank malicious content; the second layer, ZKIP, filters retrieved passages by
measuring the causal influence of each on the generated answer—without poison labels, gold answers, or
model internals. On poisoned NQ, ZKIP eliminates measured attack success in every defended configuration
while keeping retrieval quality within 0.03 of the clean-corpus baseline, at a cost of k+1 generator passes per
query; supervised analyses on NQ and BEIR confirm that its counterfactual signals carry learnable poison
structure. The clearest open problems are coordinated multi-poison attacks, standardized attack baselines,
and reducing inference overhead; we release code, datasets, and evaluation harnesses to support that work.
Limitations
Computational overhead.ZKIP requires k+1 generator passes per query ( 6×atk=5). Batching reduces
wall-clock cost on local models, but for API-priced or latency-critical deployments the overhead is substantial.
ZKIP is most practical for high-stakes, lower-throughput pipelines or selective application to flagged queries.
Single-document removal misses coordinated attacks.If multiple poisoned documents reinforce one
another, removal of any single item may not restore the correct response, muting the LOO signal (§4.5, §5.4).
Iterative removal raises cost and only partially closes this gap; we have not yet measured the multi-poison
frontier.
False positives.Filtering on output change can flag benign but opinionated or out-of-distribution documents,
especially for ambiguous queries or factual disagreements; this is the source of the Recall@5 cost we observe.
Integrating additional heuristics or weak supervision could reduce such false positives.
Evaluation scope.Our defended (ZKIP) evaluation covers NQ at 10% and 30% poisoning with a single
seed per condition; the undefended grid covers 5–30%. Metrics derive from a limited set of runs constrained
by compute, and the number of attackable queries per condition is modest (33–190), which explains the
non-monotonic ASR pattern across ratios. ASR 0.000 means no successful attacksamong the attacks and
samples we tested, not a guarantee. On BEIR we report poisoning impact and supervised detection results
(Appendices A and D); completing the ZKIP-defended BEIR grid, adding seeds, and comparing against
standardized attacks (PoisonedRAG, FlippedRAG) and recent defense baselines are the highest-priority
evaluation extensions.
Effectiveness is strongest for factual QA.The stability signal presumes a question with an (approximately
unique) factual answer. For subjective, multi-hop, or generative tasks, answer variation under context removal
is expected even without poisoning, and ZKIP’s signals weaken (Appendix B).
Threat model is implicitly dense-retrieval-only.Our poisons are LLM rewrites that alter semantics while
largely preserving the original keywords: BM25 shows ASR 0.000 at 5–20% poisoning (0.011 at 30%) against
them with no defense, because poisoned and gold documents share the query’s terms and term-frequency
ranking can rarely distinguish them. The threat model is therefore optimized for dense embeddings and does
not reflect realistic poisoning of hybrid lexical+semantic pipelines. A real adversary would craft poisons to
evade both signals simultaneously; evaluating RAGuard against such hybrid-targeting poisons (our released
-hard_modegenerator produces keyword-substituted variants) is required future work.
13

Adversarially trained retriever may be distribution-fragile.Fine-tuning on a fixed distribution of
synthetic poisons risks two failure modes: (i)overfittingto the three LLM-rewrite families used during
training, such that gradient-optimized or domain-shifted attack passages are not downranked; and (ii)degraded
generalizationto unseen query–document pairs from out-of-distribution domains. On BEIR (NFCorpus),
a medical corpus distinct from NQ, the adversarially trained retriever’s ASR is comparable to the clean
model’s (Table 5), and BEIR’s low absolute retrieval numbers make strong conclusions impossible. We do
not claim the trained retriever generalizes to unseen domains, and we rely on ZKIP as the attack-agnostic
safety net. Three mitigations worth exploring are: (a)diverse poison generation—including gradient-
optimized passages (Su et al., 2025) and backdoor-trigger rewrites (Long et al., 2025) in training, not only
LLM rewrites; (b)periodic retrainingon newly discovered attack corpora as the threat landscape evolves;
and (c)data-augmentation via domain mixing, training on joint NQ+BEIR+open-domain triples to reduce
query-distribution sensitivity.
References
Akari Asai, Zeqiu Wu, Yizhong Wang, Avirup Sil, and Hannaneh Hajishirzi. Self-RAG: Learning to retrieve,
generate, and critique through self-reflection. InThe Twelfth International Conference on Learning
Representations (ICLR), 2024. arXiv:2310.11511.
Hung-Ting Chen, Michael J.Q. Zhang, and Eunsol Choi. Rich knowledge sources bring complex knowledge
conflicts: Recalibrating models to reflect conflicting evidence. InProceedings of the 2022 Conference
on Empirical Methods in Natural Language Processing (EMNLP), pages 2292–2307, Abu Dhabi, United
Arab Emirates, 2022. Association for Computational Linguistics. doi: 10.18653/v1/2022.emnlp-main.146.
Zhuo Chen, Yuyang Gong, Jiawei Liu, Miaokun Chen, Haotan Liu, Qikai Cheng, Fan Zhang, Wei Lu,
and Xiaozhong Liu. FlippedRAG: Black-box opinion manipulation adversarial attacks to retrieval-
augmented generation models. InProceedings of the 2025 ACM SIGSAC Conference on Computer and
Communications Security (CCS). ACM, 2025. doi: 10.1145/3719027.3765023.
Hyung Won Chung, Le Hou, Shayne Longpre, Barret Zoph, Yi Tay, William Fedus, Yunxuan Li, Xuezhi
Wang, Mostafa Dehghani, Siddhartha Brahma, Albert Webson, Shixiang Shane Gu, Zhuyun Dai, Mirac
Suzgun, Xinyun Chen, Aakanksha Chowdhery, Alex Castro-Ros, Marie Pellat, Kevin Robinson, Dasha
Valter, Sharan Narang, Gaurav Mishra, Adams Yu, Vincent Zhao, Yanping Huang, Andrew Dai, Hongkun
Yu, Slav Petrov, Ed H. Chi, Jeff Dean, Jacob Devlin, Adam Roberts, Denny Zhou, Quoc V . Le, and Jason
Wei. Scaling instruction-finetuned language models.Journal of Machine Learning Research, 25(70):1–53,
2024.
Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. BERT: Pre-training of deep bidi-
rectional transformers for language understanding. InProceedings of the 2019 Conference of the North
American Chapter of the Association for Computational Linguistics: Human Language Technologies,
Volume 1 (Long and Short Papers), pages 4171–4186, Minneapolis, Minnesota, 2019. Association for
Computational Linguistics. doi: 10.18653/v1/N19-1423.
Kennedy Edemacu, Vinay M. Shashidhar, Micheal Tuape, Dan Abudu, Beakcheol Jang, and Jong Wook
Kim. Defending against knowledge poisoning attacks during retrieval-augmented generation, 2025.
arXiv:2508.02835.
Giwon Hong, Jeonghwan Kim, Junmo Kang, Sung-Hyon Myaeng, and Joyce Jiyoung Whang. Why so
gullible? Enhancing the robustness of retrieval-augmented models against counterfactual noise. InFindings
14

of the Association for Computational Linguistics: NAACL 2024, pages 2474–2495, Mexico City, Mexico,
2024. Association for Computational Linguistics. doi: 10.18653/v1/2024.findings-naacl.159.
Gautier Izacard, Mathilde Caron, Lucas Hosseini, Sebastian Riedel, Piotr Bojanowski, Armand Joulin, and
Edouard Grave. Unsupervised dense information retrieval with contrastive learning.Transactions on
Machine Learning Research, 2022. arXiv:2112.09118.
Gautier Izacard, Patrick Lewis, Maria Lomeli, Lucas Hosseini, Fabio Petroni, Timo Schick, Jane Dwivedi-Yu,
Armand Joulin, Sebastian Riedel, and Edouard Grave. Atlas: Few-shot learning with retrieval augmented
language models.Journal of Machine Learning Research, 24(251):1–43, 2023. arXiv:2208.03299.
Fredrik Johansson, Uri Shalit, and David Sontag. Learning representations for counterfactual inference. In
Proceedings of the 33rd International Conference on Machine Learning (ICML), volume 48 ofProceedings
of Machine Learning Research, pages 3020–3029. PMLR, 2016.
Tom Kwiatkowski, Jennimaria Palomaki, Olivia Redfield, Michael Collins, Ankur Parikh, Chris Alberti,
Danielle Epstein, Illia Polosukhin, Jacob Devlin, Kenton Lee, Kristina Toutanova, Llion Jones, Matthew
Kelcey, Ming-Wei Chang, Andrew M. Dai, Jakob Uszkoreit, Quoc Le, and Slav Petrov. Natural questions: A
benchmark for question answering research.Transactions of the Association for Computational Linguistics,
7:452–466, 2019. doi: 10.1162/tacl_a_00276.
Yibin Lei, Liang Ding, Yu Cao, Changtong Zan, Andrew Yates, and Dacheng Tao. Unsupervised dense
retrieval with relevance-aware contrastive pre-training. InFindings of the Association for Computational
Linguistics: ACL 2023, pages 10932–10940, 2023. doi: 10.18653/v1/2023.findings-acl.695.
Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich
Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, Sebastian Riedel, and Douwe Kiela. Retrieval-
augmented generation for knowledge-intensive NLP tasks. InAdvances in Neural Information Processing
Systems 33 (NeurIPS 2020), pages 9459–9474, 2020. arXiv:2005.11401.
Quanyu Long, Yue Deng, LeiLei Gan, Wenya Wang, and Sinno Jialin Pan. Backdoor attacks on dense
retrieval via public and unintentional triggers. InConference on Language Modeling (COLM), 2025.
arXiv:2402.13532.
Scott M. Lundberg and Su-In Lee. A unified approach to interpreting model predictions. InAdvances in
Neural Information Processing Systems 30 (NeurIPS 2017), pages 4765–4774. Curran Associates, Inc.,
2017.
Simon Lupart and Stéphane Clinchant. A study on FGSM adversarial training for neural retrieval. InAdvances
in Information Retrieval: 45th European Conference on Information Retrieval (ECIR 2023), Part II, volume
13981 ofLecture Notes in Computer Science, pages 484–492, 2023. doi: 10.1007/978-3-031-28238-6_39.
Christoph Molnar.Interpretable Machine Learning: A Guide for Making Black Box Models Explainable.
Self-published, 3 edition, 2025. ISBN 978-3-911578-03-5. URL https://christophm.github.
io/interpretable-ml-book.
Yikang Pan, Liangming Pan, Wenhu Chen, Preslav Nakov, Min-Yen Kan, and William Yang Wang. On the risk
of misinformation pollution with large language models. InFindings of the Association for Computational
Linguistics: EMNLP 2023, pages 1389–1403, Singapore, 2023. Association for Computational Linguistics.
doi: 10.18653/v1/2023.findings-emnlp.97.
15

Dae Hoon Park and Yi Chang. Adversarial sampling and training for semi-supervised information retrieval.
InThe World Wide Web Conference (WWW ’19), pages 1443–1453. ACM, 2019. doi: 10.1145/3308558.
3313416.
Mattia Prosperi, Yi Guo, Matt Sperrin, James S. Koopman, Jae S. Min, Xing He, Shannan Rich, Mo Wang,
Iain E. Buchan, and Jiang Bian. Causal inference and counterfactual prediction in machine learning for
actionable healthcare.Nature Machine Intelligence, 2:369–375, 2020. doi: 10.1038/s42256-020-0197-y.
Ori Ram, Yoav Levine, Itay Dalmedigos, Dor Muhlgay, Amnon Shashua, Kevin Leyton-Brown, and Yoav
Shoham. In-context retrieval-augmented language models.Transactions of the Association for Computa-
tional Linguistics, 11:1316–1331, 2023. doi: 10.1162/tacl_a_00605.
Nils Reimers and Iryna Gurevych. Sentence-BERT: Sentence embeddings using Siamese BERT-networks.
InProceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th
International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), pages 3982–3992,
Hong Kong, China, 2019. Association for Computational Linguistics. doi: 10.18653/v1/D19-1410.
Marco Tulio Ribeiro, Sameer Singh, and Carlos Guestrin. “Why Should I Trust You?”: Explaining the
Predictions of Any Classifier. InProceedings of the 22nd ACM SIGKDD International Conference on
Knowledge Discovery and Data Mining, pages 1135–1144. Association for Computing Machinery, 2016.
doi: 10.1145/2939672.2939778.
Stephen Robertson and Hugo Zaragoza. The probabilistic relevance framework: BM25 and beyond.Founda-
tions and Trends in Information Retrieval, 3(4):333–389, 2009. doi: 10.1561/1500000019.
Weijia Shi, Sewon Min, Michihiro Yasunaga, Minjoon Seo, Rich James, Mike Lewis, Luke Zettlemoyer, and
Wen-tau Yih. REPLUG: Retrieval-augmented black-box language models. InProceedings of the 2024
Conference of the North American Chapter of the Association for Computational Linguistics: Human
Language Technologies (Volume 1: Long Papers), pages 8371–8384, 2024. doi: 10.18653/v1/2024.
naacl-long.463.
Yucheng Shi, Mengnan Du, Xuansheng Wu, Zihan Guan, Jin Sun, and Ninghao Liu. Black-box backdoor
defense via zero-shot image purification. InAdvances in Neural Information Processing Systems 36
(NeurIPS 2023), 2023. arXiv:2303.12175.
Hongru Song, Yu-an Liu, Ruqing Zhang, Jiafeng Guo, and Yixing Fan. Chain-of-thought poisoning attacks
against R1-based retrieval-augmented generation systems, 2025. arXiv:2505.16367.
Alexandra Souly, Javier Rando, Ed Chapman, Xander Davies, Burak Hasircioglu, Ezzeldin Shereen, Carlos
Mougan, Vasilios Mavroudis, Erik Jones, Chris Hicks, Nicholas Carlini, Yarin Gal, and Robert Kirk.
Poisoning attacks on LLMs require a near-constant number of poison samples, 2025. arXiv:2510.07192.
Jinyan Su, Preslav Nakov, and Claire Cardie. Corpus poisoning via approximate greedy gradient descent. In
Findings of the Association for Computational Linguistics: ACL 2025, pages 4274–4294. Association for
Computational Linguistics, 2025. arXiv:2406.05087.
Nandan Thakur, Nils Reimers, Andreas Rücklé, Abhishek Srivastava, and Iryna Gurevych. BEIR: A
heterogeneous benchmark for zero-shot evaluation of information retrieval models. InProceedings
of the Neural Information Processing Systems Track on Datasets and Benchmarks, volume 1, 2021.
arXiv:2104.08663.
16

Haowei Wang, Rupeng Zhang, Junjie Wang, Mingyang Li, Yuekai Huang, Dandan Wang, and Qing Wang.
Joint-GCG: Unified gradient-based poisoning attacks on retrieval-augmented generation systems. In
Proceedings of the AAAI Conference on Artificial Intelligence, volume 40, pages 35793–35801, 2026. doi:
10.1609/aaai.v40i42.40893.
Wenhui Wang, Furu Wei, Li Dong, Hangbo Bao, Nan Yang, and Ming Zhou. MiniLM: Deep self-attention
distillation for task-agnostic compression of pre-trained transformers. InAdvances in Neural Information
Processing Systems 33 (NeurIPS 2020), pages 5776–5788, 2020. arXiv:2002.10957.
Wei Zou, Runpeng Geng, Binghui Wang, and Jinyuan Jia. PoisonedRAG: Knowledge corruption attacks to
retrieval-augmented generation of large language models. In34th USENIX Security Symposium (USENIX
Security 25). USENIX Association, 2025. arXiv:2402.07867.
A Full Baseline and ZKIP Evaluation Results
We report full retrieval/defense results across clean, poisoned, and ZKIP-defended settings in Tables 4 and 5.
Labels follow §4.1: “Dense (adv-trained)” is the retriever fine-tuned on synthetic poisons; “Dense (clean)” is
trained on clean triples only. Every value is taken directly from the released evaluation artifacts in the code
repository; conditions without a completed run are omitted rather than estimated.
B Threat Model Scope
Attack classes in scope.RAGuard is designed and evaluated againstfactual poisoning: an adversary injects
passages containing false factual claims—fabricated facts, contradictions of the gold evidence, and corrupted
reasoning chains—into the retrieval corpus, with the goal of changing the answers a RAG pipeline produces.
The adversary is assumed to have write access to the corpus (e.g., via user-generated content, wiki edits, or
web crawl injection) butnoaccess to model weights, query inputs, or the generation prompt. Within this
class, both retrieval-level attacks (the poisoned passage ranks above the gold passage) and generation-level
attacks (the poisoned passage is retrieved alongside gold evidence and confuses the generator) are in scope:
Layer 1 (adversarial retriever training) targets the former, and ZKIP operates at the generation stage after
retrieval, so it applies to both. All experiments in this paper instantiate this threat model with three poison
families (fabrication, contradiction, reasoning trap) at corpus poison ratios of 5–30%.
Attack classes out of scope.Several attack classes documented in the literature are explicitlynotcovered
by our evaluation.Prompt hijacking / instruction injection—embedding instructions in documents to override
system prompts—may be partially caught by ZKIP, since injected instructions tend to destabilize the decoded
answer, but we do not evaluate this and make no claim.Opinion manipulation(Chen et al., 2025) shifts
model outputs on subjective questions with no single factual answer; ZKIP’s answer-stability signal is less
meaningful without a factual reference point, and we expect degraded performance.Retrieval jamming
/ denial-of-service—flooding the corpus with irrelevant documents to suppress relevant results—is not
addressed; ZKIP can only filter what is retrieved.Backdoor-style attacks with trigger tokens(Long et al.,
2025) may be detected when the trigger causes answer instability, but a targeted backdoor engineered to
produce astablewrong answer could evade detection (§5.4).Adversarial queriesare excluded: RAGuard
assumes the query stream is clean.White-box gradient attackson the retriever or generator (e.g., Joint-GCG,
Wang et al., 2026; greedy corpus poisoning, Su et al., 2025) are not the primary target; adversarial retriever
training provides at most partial defense against them.
17

Setting Retriever Recall@5 MRR ASR↓
Clean baseline
NQ BM25 0.068 0.053 0.000
NQ Dense (clean) 0.282 0.200 0.000
Adversarially trained, evaluated on clean corpus
NQ Dense (adv-trained) 0.301 0.201 0.000
Under poisoning attack
NQ (5%) BM25 0.070 0.055 0.000
NQ (5%) Dense (clean) 0.268 0.181 0.061
NQ (5%) Dense (adv-trained) 0.323 0.220 0.091
NQ (10%) BM25 0.070 0.056 0.000
NQ (10%) Dense (clean) 0.259 0.176 0.101
NQ (10%) Dense (adv-trained) 0.319 0.215 0.072
NQ (20%) BM25 0.075 0.059 0.000
NQ (20%) Dense (clean) 0.256 0.179 0.029
NQ (20%) Dense (adv-trained) 0.327 0.218 0.065
NQ (30%) BM25 0.071 0.059 0.011
NQ (30%) Dense (clean) 0.256 0.175 0.053
NQ (30%) Dense (adv-trained) 0.318 0.212 0.068
Poisoning + ZKIP defense
NQ (10%) BM25 + ZKIP 0.070 0.0560.000
NQ (10%) Dense (clean) + ZKIP 0.259 0.1760.000
NQ (10%) Dense (adv-trained) + ZKIP 0.314 0.2150.000
NQ (30%) BM25 + ZKIP 0.071 0.0590.000
NQ (30%) Dense (clean) + ZKIP 0.256 0.1750.000
NQ (30%) Dense (adv-trained) + ZKIP 0.274 0.1960.000
Table 4: Natural Questions retrieval/defense results across clean, poisoned, and ZKIP-defended settings.
ZKIP eliminates measured ASR in every defended configuration. Recall@5 and MRR are computed over the
full 1,000-query set; ASR is computed over the queries where a poisoned document is present in the retrieved
candidates (33, 69, 139, and 190 queries at 5%, 10%, 20%, and 30% poisoning respectively). Defended runs
exist at 10% and 30%; the 20% defended condition has not been run and is omitted rather than estimated.
Corpus and query assumptions.RAGuard assumes (i) the query stream is clean (no adversarially crafted
queries); (ii) the retrieval corpus is partially poisoned, with poison ratios of 5–30% in our experiments; and
(iii) the gold (correct) document exists somewhere in the corpus, even if not always retrieved. Assumption
(iii) matters for ZKIP: if no correct evidence is retrievable, removing a poison cannot restore a correct answer,
and the stability signal degrades toward the failure mode described in §5.4.
C Retriever Training Details
Each training triple is {query, gold document, negative document }. For triples carrying a poison label,
the negative is the attack-family-specific rewrite of that triple’s gold document, making the contrastive
signal directly anti-poison; clean triples use in-corpus random negatives. The released training script
(retrievers/train.py ) additionally supports contrastive and in-batch-negative losses, hard-negative
selection, a configurable training poison ratio, and per-epoch loss logging for reproducibility; the configuration
in Table 6 reproduces the checkpoints used in this paper. ZKIP defaults: λ= 0.5 , removal budget m= 1 ,
k= 5 , answer encoder all-MiniLM-L6-v2 , generator GPT-4o-mini (temperature 0.2, top- p0.95, max
18

Setting Retriever Recall@5 MRR ASR↓
Clean baseline
BEIR BM25 0.018 0.013 0.000
BEIR Dense (clean) 0.019 0.014 0.000
Adversarially trained, evaluated on clean corpus
BEIR Dense (adv-trained) 0.018 0.013 0.000
Under poisoning attack (30% poisoned corpus)
BEIR (30%) BM25 0.016 0.013 0.000
BEIR (30%) Dense (clean) 0.021 0.015 0.008
BEIR (30%) Dense (adv-trained) 0.019 0.014 0.007
Table 5: BEIR (NFCorpus) retrieval results on the clean and 30%-poisoned corpora (2,481 ASR-eligible
queries). Absolute retrieval numbers are low for all systems on NFCorpus; the relevant observations are
relative: dense ASR is an order of magnitude above BM25’s (which stays at 0.000), and adversarial training
does not meaningfully reduce ASR on this out-of-domain corpus. ZKIP-defended BEIR runs are not yet
complete and are deliberately not reported (§6); supervised detection results on BEIR appear in Appendix D.
Component Value
Base encoderall-MiniLM-L6-v2
Pooling Mean over last hidden state
Loss Triplet margin loss, margin= 0.2
Optimizer AdamW
Learning rate2×10−5
Batch size 16
Epochs 3
Max sequence length 256 tokens
Poison ratio (training) 0.30, even across 3 families
Negatives (poisoned triples) The poisoned rewrite of the gold doc
Negatives (clean triples) Random clean passage from corpus
Training filestrain_triples_nq_poisoned.jsonl
train_triples_clean.jsonl
Table 6: Full retriever fine-tuning configuration. “Dense (clean)” uses the clean triples file; “Dense (adv-
trained)” uses the poisoned triples file with the same hyperparameters.
128 tokens).
D Supervised Poison Classification Ablations
This appendix reports supervised poison-classification ablations that complement ZKIP. Whereas ZKIP is
label-free and operates by counterfactual sensitivity, these ablations test whether (i) poisoned passages are
distinguishable from gold passages directly from text, and (ii) ZKIP-derived influence signals are sufficiently
informative for a supervised model to predict poison labels. These results are a supervised upper bound: they
require labeled poison data during training, while ZKIP does not.
19

D.1 Experimental Setup
We extract eight features per retrieved document:rank_pos(normalized retrieval rank),sim_q(query–
document cosine),sim_ans(similarity between baseline and counterfactual answers),entropy_delta( ∆Hi
per Equation 6),f1_deltaandem_delta(answer-score change on removal; gold-dependent, used only in this
supervised analysis),flip_wrong_to_right(indicator that removal corrects a wrong answer; gold-dependent,
supervised analysis only), andsim_z(embedding outlier z-score within the retrieved set). The released
feature extractor exposes an oracle_free mode that drops the gold-dependent features for inference-time
use.
D.2 Text-Level Classification on Gold vs. Poison Pairs
Dataset Poison Acc. Prec. Recall F1 ROC-AUC AUPRC
NQ5% 0.594 0.851 0.228 0.360 0.685 0.719
10% 0.593 0.844 0.228 0.359 0.690 0.720
20% 0.592 0.843 0.226 0.356 0.686 0.720
30% 0.591 0.853 0.220 0.350 0.685 0.717
BEIR5% 0.613 0.649 0.490 0.559 0.687 0.678
10% 0.612 0.657 0.466 0.546 0.689 0.674
Table 7: BERT poison-classifier performance on gold vs. poison document pairs: high precision but low recall
on NQ indicates many poisons are lexically subtle rewrites that text alone cannot flag without over-triggering.
Across NQ, AUPRC remains stable at ≈0.717–0.720 over poisoning ratios from 5% to 30%, with ROC-
AUC≈0.685–0.690 (Table 7). Precision is high ( ≈0.84–0.85) but recall is low ( ≈0.22–0.23): the classifier
is conservative, flagging only a limited subset of poisons. Many poisons are semantically subtle rewrites
that remain lexically and stylistically similar to gold documents. On BEIR, the profile is more balanced
(precision ≈0.65, recall ≈0.47–0.49), suggesting poisoned passages are comparatively easier to separate at
the document-text level there, potentially due to greater topical diversity or larger stylistic shifts.
D.3 Feature-Level Classification Using ZKIP Influence Signals
Metric LogReg BERT
AUPRC 0.377 0.732
ROC-AUC 0.571 0.814
F1 0.514 0.636
Precision 0.442 0.531
Recall 0.614 0.792
Specificity 0.536 0.581
Table 8: Classifier performance on labeled NQ using ZKIP-derived influence features: nonlinear models
nearly double AUPRC over a linear baseline, indicating exploitable interactions among counterfactual signals.
A logistic-regression baseline achieves AUPRC 0.377 and ROC-AUC 0.571, while the BERT-based
classifier improves substantially to AUPRC 0.732 and ROC-AUC 0.814, with F1 increasing from 0.514
to 0.636 (Table 8). The AUPRC gain matters under class imbalance and indicates that nonlinear models
exploit interactions among influence features (e.g., combining answer destabilization with retrieval rank and
similarity cues) that a linear model cannot capture. In the logistic-regression feature-importance analysis,
20

retrieval rank (rank_pos) dominates, followed by query similarity (sim_q) and answer stability (sim_ans),
with the entropy differential contributing a smaller but nonzero share.
D.4 Takeaway and Relationship to ZKIP
These supervised results support the central motivation of ZKIP: poisoned passages induce measurable,
systematic changes in generation under counterfactual removal, and these changes contain learnable signal.
However, unlike ZKIP, supervised classifiers require poison labels for training and may not generalize to
unseen attack styles without continual relabeling and retraining. We therefore treat the learned classifiers
as ablations that validate the informativeness of ZKIP’s influence signals, while maintaining ZKIP as the
primary, label-free defense.
21