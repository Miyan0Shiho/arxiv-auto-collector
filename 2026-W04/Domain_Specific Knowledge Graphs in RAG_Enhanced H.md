# Domain-Specific Knowledge Graphs in RAG-Enhanced Healthcare LLMs

**Authors**: Sydney Anuyah, Mehedi Mahmud Kaushik, Hao Dai, Rakesh Shiradkar, Arjan Durresi, Sunandan Chakraborty

**Published**: 2026-01-21 19:55:12

**PDF URL**: [https://arxiv.org/pdf/2601.15429v1](https://arxiv.org/pdf/2601.15429v1)

## Abstract
Large Language Models (LLMs) generate fluent answers but can struggle with trustworthy, domain-specific reasoning. We evaluate whether domain knowledge graphs (KGs) improve Retrieval-Augmented Generation (RAG) for healthcare by constructing three PubMed-derived graphs: $\mathbb{G}_1$ (T2DM), $\mathbb{G}_2$ (Alzheimer's disease), and $\mathbb{G}_3$ (AD+T2DM). We design two probes: Probe 1 targets merged AD T2DM knowledge, while Probe 2 targets the intersection of $\mathbb{G}_1$ and $\mathbb{G}_2$. Seven instruction-tuned LLMs are tested across retrieval sources {No-RAG, $\mathbb{G}_1$, $\mathbb{G}_2$, $\mathbb{G}_1$ + $\mathbb{G}_2$, $\mathbb{G}_3$, $\mathbb{G}_1$+$\mathbb{G}_2$ + $\mathbb{G}_3$} and three decoding temperatures. Results show that scope alignment between probe and KG is decisive: precise, scope-matched retrieval (notably $\mathbb{G}_2$) yields the most consistent gains, whereas indiscriminate graph unions often introduce distractors that reduce accuracy. Larger models frequently match or exceed KG-RAG with a No-RAG baseline on Probe 1, indicating strong parametric priors, whereas smaller/mid-sized models benefit more from well-scoped retrieval. Temperature plays a secondary role; higher values rarely help. We conclude that precision-first, scope-matched KG-RAG is preferable to breadth-first unions, and we outline practical guidelines for graph selection, model sizing, and retrieval/reranking. Code and Data available here - https://github.com/sydneyanuyah/RAGComparison

## Full Text


<!-- PDF content starts -->

Domain-Specific Knowledge Graphs in
RAG-Enhanced Healthcare LLMs
Sydney Anuyah*, Mehedi Mahmud Kaushik*, Hao Dai‡, Rakesh Shiradkar†, Arjan Durresi*, Sunandan Chakraborty*
*Luddy School of Informatics, Computing, and Engineering, Indiana University, Indianapolis, IN, USA
‡School of Medicine, Indiana University, Indianapolis, IN, USA
†Department of Biomedical Engineering and Informatics, Indiana University, Indianapolis, IN, USA
Emails: {sanuyah, mekaush, daihao, rshirad, adurresi, sunchak}@iu.edu
Abstract—Large Language Models (LLMs) generate fluent
answers but can struggle with trustworthy, domain-specific rea-
soning. We evaluate whether domain knowledge graphs (KGs)
improve Retrieval-Augmented Generation (RAG) for healthcare
by constructing three PubMed-derived graphs:G 1(T2DM),
G2(Alzheimer’s disease), andG 3(AD+T2DM). We design
two probes: Probe 1 targets merged AD–T2DM knowledge,
while Probe 2 targets the intersection ofG 1andG 2. Seven
instruction-tuned LLMs are tested across retrieval sources {No-
RAG,G 1,G 2,G 1+G2,G 3,G 1+G2+G3} and three decoding
temperatures. Results show thatscope alignmentbetween probe
and KG is decisive: precise, scope-matched retrieval (notably
G2) yields the most consistent gains, whereas indiscriminate
graph unions often introduce distractors that reduce accuracy.
Larger models frequently match or exceed KG-RAG with a
No-RAGbaseline on Probe 1, indicating strong parametric
priors, whereas smaller/mid-sized models benefit more from well-
scoped retrieval. Temperature plays a secondary role; higher
values rarely help. We conclude that precision-first, scope-
matched KG-RAG is preferable to breadth-first unions, and
we outline practical guidelines for graph selection, model siz-
ing, and retrieval/reranking. Code and Data available here -
https://github.com/sydneyanuyah/RAGComparison
I. INTRODUCTION
Alzheimer’s disease (AD) and type 2 diabetes mellitus
(T2DM) represent two of the most pressing chronic health
challenges we face today [1], [2], each carrying its own
significant public health burden while also sharing surprising
connections with one another [3]. Today, AD stands as the
primary cause of dementia in older adults [4], [5]. More than
55 million people worldwide are living with dementia, and
experts predict this could surge to somewhere between 139
and 150 million by 2050 [6], [7]. T2DM shows a similarly
alarming trend, with roughly 589 million adults as of 2024
(that’s about 11.1% of the adult population) living with the
condition, and projections suggest we could see around 853
million cases by mid-century [8]. What is particularly striking
is that over 90% of all diabetes cases are type 2 [8], [9], largely
driven by our ageing populations and modern lifestyle factors.
As life expectancy continues to climb, there are statistically
more people dealing with both metabolic and neurodegenera-
tive diseases simultaneously [10], which makes understanding
the relationship between AD and T2DM increasingly urgent
for biomedical AI development.Epidemiological research has consistently shown that hav-
ing diabetes substantially increases the risk of cognitive de-
cline and dementia, including AD [11]. A recent 2024 meta-
analysis showed that diabetic patients face about 59% higher
dementia risk compared to people without diabetes [11],
[12], which reinforces what earlier studies had found that
T2DM increases an individual’s risk of cognitive disorders
by roughly 1.3 to 1.9 times [13]. When we dig into the
molecular details, the picture gets even more interesting. The
chronic high blood sugar and insulin resistance that defines
T2DM actually mirrors some of the same pathophysiological
processes we see in AD [14]. This similarity has led some
researchers to provocatively label AD as “type 3 diabetes"
though this remains controversial [15]–[18]. The idea empha-
sizes the overlapping features: impaired insulin signaling in the
brain, chronic inflammation, and oxidative stress [13]. Both
conditions also share common risk factors such as midlife
obesity and high blood pressure [13].
Despite the wealth of published research on AD–T2DM
connections, Large Language Models (LLMs) on their own
have real trouble delivering reliable medical answers [19].
Today’s state-of-the-art (SOTA) LLMs can certainly generate
impressively fluent responses, but they frequently hallucinate
facts or fabricate citations that do not exist, which is a serious
safety concern in healthcare settings [20]. Even powerful mod-
els like GPT-4 have inherent knowledge limitations that mean
they might miss crucial domain-specific details or misrepre-
sent the latest findings. This is where Retrieval-Augmented
Generation (RAG) has become the go-to solution, helping
ground LLM outputs in external, verified knowledge [21]. By
pulling in relevant documents or facts when answering a query,
RAG significantly improves factual accuracy [21]. However,
conventional RAG systems that retrieve free-text passages can
still drag in irrelevant or contradictory information, which
the LLM might then mistakenly weave into its answer [21].
This becomes especially problematic for knowledge-intensive
questions that require multiple reasoning steps, where models
need to piece together complex biomedical relationships.
Structured knowledge bases offer a more promising al-
ternative. Knowledge graphs (KGs) organize facts as sub-
ject–relation–object triples(E 1, R, E 2)as seen in Table I,
whereE 1is the subject entity,E 2is the object entity, andarXiv:2601.15429v1  [cs.CL]  21 Jan 2026

TABLE I: Examples of triples used for the knowledge graphs
KG Example
G1 ["Entity 1": "T2DM", "Relationship": "was associated with", "Entity 2":
"decreased forced expiratory volume in 1s (FEV1)"]
G2 ["Entity 1": "Alzheimer’s disease CSF", "Relationship": "is associated
with", "Entity 2": "neuroinflammation"]
G3 ["Entity 1": “Insulin-like growth factor ", "Relationship": "influences",
"Entity 2": “cognitive functions"]
Rrepresents their relationship. This structured representation
provides clear semantics and traceable origins, making it easier
to verify how an answer was constructed. Yet, critical ques-
tions remain about how best to leverage KGs in healthcare AI:
Does curating domain-specific KGs from biomedical literature
actually improve answer quality in RAG-enhanced LLMs?
Should we build narrow, disease-focused graphs or broader
cross-domain ones? And how do these design choices interact
with model parameters like decoding temperature?
In this research, we investigate the role of domain-specific
KGs curated from PubMed abstracts in RAG-enhanced health-
care LLMs. We ask whether such KGs improve the factual
accuracy and evidential support of answers compared to no-
RAG baselines (RQ1). We further examine how curation scope
affects question answering: do focused, disease-specific KGs
(G1 for T2DM, G2 for AD) outperform a broader merged
KG (G3) on single-hop and multi-hop probes, or vice versa
(RQ2)? We also investigate the robustness of RAG gains
across different decoding temperatures (0, 0.2, 0.5) and test
for statistical significance of improvements (RQ3). Finally,
we explore whether we can enhancing the components of the
CoDe-KG pipeline to yield higher-quality graphs that translate
into better downstream QA performance (RQ4).
A. Contributions
In this research, we investigate the role of domain-specific
KGs curated from abstracts on PubMed for RAG-enhanced
healthcare LLMs.
•We build three abstract-derived KGs with a common
schema:G 1(T2DM-focused),G 2(AD-focused), andG 3
(combined AD+T2DM).
•We introduce probe sets spanning single-hop
clinical facts (e.g., drug→outcome, gene→disease)
and multi-hop mechanisms (e.g., insulin
signaling→neuroinflammation→cognition)
•We evaluate 7 LLMswithandwithoutKG-RAG across
varying temperatures0,0.2, and0.5, measuring F 1score,
and accuracy across the probes, and we test for signif-
icance of RAG gains and of curation scope (G 1/G2vs.
G3)
•We improve the co-reference of the CoDe-KG pipeline.
II. BACKGROUND
A. Pathophysiological Links Between AD and T2DM
AD and T2DM share several disease mechanisms that play
out at both the molecular and whole-body at scales. Chronic
insulin resistance, the defining feature of T2DM, contributes
to AD by interfering with how neurons take up glucose andrespond to insulin signals in the brain [13]. This neurode-
generative change: the combination of excess glucose and
disrupted insulin signaling fuels inflammation and oxidative
stress, which then accelerates the formation ofβ-amyloid
plaques and tau tangles (the signature brain abnormalities we
see in AD) [13].
As established by the KG, Apolipoproteinϵ4 (APOEϵ4)
offers a genetic link between these two diseases, example of
a triple that shows this link is "Individuals with T2DM have
APOE4-related cognitive and olfactory impairment". While we
know APOEϵ4 as the strongest genetic risk factor for late-
onset AD, its influence extends beyond just amyloid process-
ing; it’s also tied to how our bodies handle fats and glucose
[22]. People with diabetes who carry APOEϵ4 experience
faster mental decline and face higher dementia risk than
diabetics without this gene variant, suggesting that the allele
and metabolic stress amplifies each other’s effects [13], [23].
Inflammation represents another critical connection point
between T2DM and AD. In T2DM, visceral fat and insulin-
resistant tissues pump out elevated levels of adipokines and in-
flammatory cytokines. These molecules can breach the blood-
brain barrier, adding fuel to the neuroinflammation already
present in AD. From a clinical standpoint, T2DM patients
carry roughly twice the risk of developing vascular dementia
and show higher rates of Alzheimer’s dementia across many
studies [13], [24].
B. Knowledge Graphs in Biomedical Research
The complex web of relationships in biomedicine has
pushed researchers to develop knowledge graphs (KGs) as a
way to organize factual information in a structured, queryable
format. Unlike traditional text documents, KGs represent
biomedical entities like genes, diseases, drugs, proteins as
nodes or entities in a network, with their relationships depicted
as labelled edges. For example, an edge might connect a
gene node to a disease node with the label “associated with,"
making the relationship explicit and machine-readable. Large-
scale biomedical KGs like SPOKE show how they created
KGs by integrating over 40 curated databases, including
DrugBank and GWAS catalogs, resulting in a comprehensive
network of approximately 42 million nodes across 28 entity
types and 160 million relationships [25]. CoDe-KG [26] is
another SOTA pipeline built with open-source LLMs. What
sets KGs apart from text-based resources is their built-in
traceability. While bioinformaticians have traditionally used
KGs for drug repurposing and gene discovery, we leverage
them as knowledge sources for question answering (QA).
Given this context, we propose to leverage CoDe-KG [26],
an automated KG construction pipeline, to build focused
biomedical knowledge graphs and evaluate their impact on
trustworthy domain-specific questions. CoDe-KG is an open-
source framework that extracts structured facts from text by
combining robust co-reference resolution with syntactic de-
composition. This approach breaks down complex statements
and resolves pronouns, thereby capturing more complete and
context-rich relations. According to the authors, CoDe-KG

achieved an increase of∆ =8% F 1when compared to prior
methods on the REBEL relation extraction dataset [26], and
integrating co-reference + decomposition increased recall on
rare relations by over 20% [26]. We harness this pipeline to
distil knowledge from biomedical abstracts into three KGs of
varying scope:G 1,G2andG 3for the combined AD+T2DM
domain. We then use these graphs in an LLM-driven RAG
setup to answer questions. By comparing performance across
G1,G2, andG 3, we examine whether a targeted disease-
specific KG yields better answers than a broader cross-domain
KG, or vice versa. We further probe how the structure and
scope of knowledge graphs affect the LLM’s ability to deliver
correct, well-supported answers. While RAG models often out-
perform non-RAG approaches, this is not always guaranteed.
As [27], [28] point out, earlier KG-RAG frameworks with
fixed search parameters could retrieve redundant trivial facts or
miss important multi-hop connections, ultimately weakening
the LLM’s reasoning capabilities and making RAG-enhanced
systems perform worse than their non-RAG counterparts.
C. Our Approach
In this work, we built three different knowledge graphs
(G1,G2andG 3) to investigate how the scope of knowledge
affects QA performance in this domain. We detail the curation
of Probe1 and Probe2 in the subsequent sections. In Probe1,
formulated onG 3and Probe2, formulated onG 1∩G 2, we
tested different combinations of LLMs built on the different
RAG systems. Our hypothesis going in was that the focused
KG (G 1andG 2) might perform better on questions specif-
ically about the AD-T2DM intersection, since it provides
denser, more concentrated context for that particular overlap.
Meanwhile, the merged KG (G 3) should theoretically handle a
wider range of questions about either disease or their connec-
tions more effectively. However, there’s a potential downside
to the merged approach as it could introduce distractors (facts
relevant to one disease but not the question at hand), which
might confuse the LLM. By testing the same set of questions
against both knowledge graphs, we can observe any trade-offs
between having a highly focused knowledge source versus a
more comprehensive one.
III. METHODOLOGY
In this section, we cover the development of the three KGs,
(G1(T2DM-focused),G 2(AD-focused), andG 3(combined
AD+T2DM)), the creation of the probes and the experimen-
tation of the seven LLMs with different combinations of the
knowledge graphs, shown in Figure 1.
A. Abstract Selection and Filtration
Search queries used to build the KG sets
QT2DM (T2DM-only).
(T2DM OR "Type 2 Diabetes" OR "Type
II Diabetes" OR "Type-2 Diabetes" OR
"Diabetes Mellitus, Type 2" OR NIDDM
OR "non insulin dependent diabetes"OR "non-insulin-dependent diabetes" OR
"adult-onset diabetes" OR (diabet *AND
("type 2" OR T2DM)))
QAD(AD-only).
(AD OR "Alzheimer’s disease" OR
"Alzheimer disease" OR "Alzheimers
disease" OR Alzheimers OR Alzheime *
OR Alzhiemer *OR "dementia of the
Alzheimer type" OR DAT OR LOAD OR
"late-onset Alzheimer *")
QT2DM+AD (AD & T2DM together).
((AD OR "Alzheimer *disease" OR
Alzheime *OR Alzhiemer *OR DAT OR LOAD)
AND (T2DM OR "Type 2 Diabetes" OR "Type
II Diabetes" OR "Diabetes Mellitus,
Type 2" OR NIDDM))
After applying theQ T2DM /QAD/QAD+T2DM search
strings, we created a simple, reproducible filter to rank ab-
stracts and kept the most relevantper group. The aim is to
bias toward causal and mechanistic content while still covering
phenotypes and biomarkers that matter for AD, T2DM, and
their intersection. We remove short-worded abstracts (less than
180 words) to avoid editorials or thin notes:
keep(i) =⊮
words(Abstract i)≥180
.
We then join title and abstract into one text stringx i, and
build a TF–IDF matrix on unigrams and bigrams (English
stopwords,min_df=2):
X∈Rn×V, X i= tfidf(x i), i= 1. . . n.
We create three query vectors by TF–IDF, transforming the
curated term lists:
qcaus, qpheno, qbiom∈RV,
representingCAUSALITY_TERMS,PHENOTYPE_TERMS,
andBIOMARKER_TERMSrespectively. For each abstract, we
compute three cosine similarities:
s(caus)
i = cos(X i, qcaus), s(pheno)
i = cos(X i, qpheno),
s(biom)
i = cos(X i, qbiom).
We reward abstracts that explicitly name crucial elements
(e.g., “Mendelian randomization,” “longitudinal,” “p-tau-217,”
“APOE4,” “insulin resistance phenotype,” “HbA1c”).
k(caus)
i , k(pheno)
i , k(biom)
i ∈N, k(tot)
i=k(caus)
i +k(pheno)
i +k(biom)
i .
Per-feature normalization.:Each signal is min–max nor-
malized to[0,1]to keep ranges comparable:
˜s(·)
i=s(·)
i−mins(·)
maxs(·)−mins(·),˜ki=k(tot)
i−mink(tot)
maxk(tot)−mink(tot).
We then produce a single value that we useonly to rank
abstracts:
Ri=w caus˜s(caus)
i +w pheno ˜s(pheno)
i +w biom˜s(biom)
i

+w kw˜ki.
The final KGs were created from the top 1000 selected
abstracts.
B. Pipeline replication and coref upgrade
We start from the selected abstractsD={d 1, . . . , d N}and
treat each abstract as a set of sentencesS(d) ={s 1, . . . , s |d|}.
The extraction loop in CODE-KG is
T=[
d∈D[
s∈S(d)REψ(decomp(coref θ(s))),(1)
wheres(coref)→is resolving abstract co-reference, i.e
“T2DM" = “Type 2 Diabetes Mellitus" = “t2dm"; then de-
compose complex sentences into simpler ones (decomp), then
extract triples(E 1, R, E 2)with a simple source tagπ(t)that
notes (paper id, sentence id, clause id). This tag lets us always
point back to the exact sentence that created a triple. We
first matched the original authors’ inference setting (temp
= 0.7) and reproduced similar relation extraction behavior.
We replaced the coreference backbone with Qwen 32B coder
and kept all other steps the same.
coref θ⋆reachesF1 = 61%vs.58%before,
This test was done on the authors dataset. In practice, better
coref means fewer broken heads or tails later, so the KG has
cleaner nodes and more usable causal links.
C. KG build, cleanup, and naming rules
Raw extraction gives a multi-set of triplesT. For our study,
we want edges that point in a clear causal direction, so we kept
only a small relation set.
Rc={causes,because, ...}.
Formally,
Tc={(E 1, R, E 2, π)∈ T:r∈ R c}.(2)
Next, we remove vague heads/tails. A simple maskMdrops
items like “it”, “this”, or “this study”:
M(x) =⊮[x /∈ {it,this,this study}],
Tclean
c ={t∈ T c:M(E 1) =M(E 2) = 1}.
We manually normalize names so variants collapse to a
single label: “T2DM” and “Type 2 Diabetes” should be treated
as one thing. We rewrite each edge by
Φ : (E 1, R, E 2, π)7→ 
m(E 1), R, m(E 2), π
,
and obtain the canonical set bTc={Φ(t) :t∈ Tclean
c}.
D. Probe Creation
We created two types of Probes: the first built onG 3and the
second built on the intersection ofG 1andG 2→(G 1∩G2)1) Probe 1:Containing 100 multiple-choice questions, this
probe tests the joint AD+T2DM query. The questions are
framed to cover single hop, multi-hop and fill in the blank
(FITB). LetA∈ {0,1}|bE3|×|bE3|be the adjacency overR c.
•Single-hop.Choose(u, r, v)∈ bTc,3and create one correct
option and three distractors fromN−(u) ={x:A x,u=
1}that match type/frequency. This checks one clean link.
•Multi-hop, pair-selection.For a targetx, the set ofdirect
causes isP 1(x) ={u:A u,x= 1}. The correct answer
is an unordered pair{u, v} ⊆ P 1(x). Distractors come
fromP 2(x) ={u: (A2)u,x= 1}or close neighbors that
look right but are not immediate parents. This stresses
real 2-hop reasoning.
•FITB.Mask one canonical token in(u, r, v); only the
canonical label passes. This drills precision under syn-
onyms.
We ensure synonym control explicitly, as options must be
distinct in canonical space (m(c i)̸=m(c j)), so there are no
duplicate answers under different spellings.
2) Probe 2:This probe targetsG 1∩G 2. We embed the
triple with an encoders(·)∈Rdand use cosine similarity to
screen for intersection candidatesI
I=n
(τ1, τ2)∈bTc,1×bTc,2: cos(s(τ 1), s(τ 2))≥0.65o
.(3)
Yielding|I|= 424. After stop-word removal and de-
duplication in canonical space, we keep| bI|= 193items. We
then form questions from this intersection subgraph:
•Single-hop, FITB:Same as Probe 1 but restricted to bI.
•Multi-hop with direction.If the true edge isu→x, then
the optionx→uis wrong even though it uses the same
names. We present four atomic options (1–4) and ask for
the correct pair among A–E (the 2-combinations). Exactly
two pairs are correct; both must point into the target.
0.65 was chosen as the cosine similarity value because it kept
the same causal theme in both diseases (e.g., insulin signaling,
inflammation) without having too loose matches.
E. RAG setups Prompting and Answer Generation
We compare six retrieval setups:
K∈
∅,G 1,G2,G1+G2,G3,G1+G2+G3	
.
Where∅→(No-RAG). For each value ofK, we run
the same LLM with and without this context. For QA we use
T∈ {0,0.2,0.5}; LowerTsticks closer to retrieved facts;
higherTcan add details but risks drift. All the prompts were
zero-shot instruction based.
You are answering a multiple-choice
question.
Return ONLY one uppercase letter from
this set: {allowed_str}.
Do not include explanations or extra
text.

Step 1a: Query PubMed 
and filter by length
Step 3b: CanonicalizeStep 3a: Filter causal
relations
Final output: 
Three Knowledge GraphsStep 1b: Rank and
select top K abstracts
Pipeline replication 
and coref upgradeKG build, cleanup, and naming rules Abstract Selection and FiltrationStep 2: Extract raw triples
Probe CreationProb 1: Query PubMed 
and filter by length
Prob 2: Intersection
(110 questions)Evaluation
RAG setups Prompting and
Answer GenerationFig. 1: Overview of the methodology for constructing domain-specific knowledge graphs and evaluating their impact on RAG-
enhanced healthcare LLMs. The pipeline includes abstract selection, knowledge graph construction, probe generation, and
systematic evaluation across multiple models and retrieval configurations. Detailed explanation is provided in Section III.
TABLE II: QA items formatted as fill-in-the-gap, multi-hop, and single-hop examples.
Question type Question Options Answer choices
Fill-in-the-gap Type 2 diabetes increases Alzheimer’s risk through
and .1. Neuroinflammation
2. Amyloid degradation
3. Insulin resistance
4. Tau dephosphorylationA: 1 and 2
B: 3 and 4
C: 3 and 2
D: 2 and 4
E: 3 and 1
Multi-hop Which two are direct precursors of neuroinflammation
escalation?1. Aβoligomers
2. Peripheral infection
3. Aerobic fitness
4. Microglial primingA: 1 and 2
B: 1 and 4
C: 2 and 3
D: 3 and 4
Single-hop Abnormal insulin signaling in the brain primarily results in: A: Lower GSK3βactivity
B: Reduced oxidative damage
C: Enhanced mitochondrial function
D: Higher synaptic resilience
E: Reduced Aβdegradation and increased tau phosphorylation—
Since all questions were multiple-choice, we incorporated
common distractors in the options. For questions with multiple
answers, we tested the macro and micro F1 in those cases.
IV. RESULTS
A. Effect of Varying Temperature
For each Model×Probe×Graph configuration, we compared
macro-F 1across temperatures using pairwise Welch’st-tests
(unequal variances) for the three temperature values:0vs.0.2,
0.2vs.0.5, and0vs.0.5. Within each configuration we applied
Holm–Bonferroni correction across the three tests; statistical
significance is denoted as * (p adj<.05), ** (p adj<.01), and ***(padj<.001). Effect sizes (Cohen’sd) were computed for all
comparisons.
Because each temperature condition is represented by a
single run per configuration (i.e.,n=1per temperature),
inferential tests were of low statistical power and produced no
adjustedp-values below.05; consequently, no cells received a
significance marker after Holm correction. We therefore report
directional patterns descriptively. Across all 84 configurations,
increasing temperature from0→0.5reduced macro-F 1in 52
cases, increased it in 23, and left it unchanged in 9 (median
∆ =−0.02). The attenuation with higherTwas more
pronounced on Probe 1 (28 decreases, 9 increases; median

TABLE III: Temperature sensitivity by graph (macro-F 1).
Counts across all Model×Probe settings;∆is median change
fromT=0to0.5.
Graph # Increases # Decreases # No change Median∆(0→0.5)
G1 7 7 0 0.00
G1+G2 3 10 1 -0.03
G1+G2+G3 3 9 2 -0.03
G2 3 10 1 -0.03
G3 3 8 3 -0.02
No-RAG 4 8 2 -0.02
∆ =−0.03) than Probe 2 (24 decreases, 14 increases; median
∆ =−0.01). By graph condition,G 1was most temperature-
stable (balanced 7 increases/7 decreases; median∆≈0),
whereasG 2,G3, andG 1+G2(+G3) skewed toward decreasing
values, however, it was small (median∆∈[−0.03,−0.015]).
By model,Anthropic.Claude-3-Haikushowed the
mildest tendency to improve with temperature (median
∆= + 0.005), whileMistral-7B-Instruct-v0.3
exhibited the largest typical drop (median∆=−0.12).
Illustratively, the largest decrease occurred for
Mistral-7B-Instruct-v0.3on Probe 1 across
several graphs (∆ 0→0.5 ∈[−0.25,−0.21]), whereas a
notable increase was observed forMixtral-8x7B-v0.1
(No-RAG, Probe 2;∆ 0→0.5 = + 0.09).
On the domain of single-run estimations (no stars beyond
the Holm adjustment), on the graphs of macro-F 1, the trend
of higher decoding temperature is, in most cases, downward,
particularly Probe 1 and non-G 1graphs. When stability is
preferred,T=0is the safest default; small exploration at
T=0.2may be helpful in some environments, butT=0.5tends
to worsen the accuracy. Accordingly, we take an averaged
value of each of these temperatures.
B. Model Performance
The models which perform well on both probes
are: Anthropic Claude-3-Haiku, (which is the only non-
open source model), Qwen-2.5-32B-Instruct, Llama-3.1-8B-
Instruct, Llama-3.3-70B-Instruct, and GPT-OSS-20B. Llama-
3.3-70B-Instruct scores the highest macro F1 on the probe
1, while Qwen-2.5-32B-Instruct has the highest macro F1 in
probe 2, beating the Anthropic Claude-3-Haiku model. An
observation is that these models which are particularly large in
size are already trained to contain large amounts of biomedical
knowledge and therefore adding external information in the
shape of a domain-specific KG is likely to have very minimal
value and can even cause a model to become confused and
confused by irrelevant or contradictory information, shown by
the dip in the results, in Table IV and V. From the results, we
see that normal RAG might lure in erroneous or conflicting
facts, which is likely the reason whyG 1andG 3fell in
performance in these models. On probe 2, the image is a bit
different - RAG occasionally helps. For example, the combined
graph (G 1+G2+G3) slightly improves Anthropic Haiku from
0.61 to 0.63 and GPT-OSS-20B from 0.46 to 0.57, suggesting
that even large models can benefit from external knowledge
when tackling more complex, multi-hop reasoning tasks.Small models benefit fromG 2. The baseline scores of
mistral-7B and Mixtral-8X7B are moderate (0.69 and 0.80)
in probe 1 and (0.29 and 0.39) in probe 1. The performance
of both probes is significantly improved when they retrieve
out of the AD-oriented KG (G 2). On probe 1 for Mixtral-
8X7B, F 1is increased by 0.80 to 0.89, and probe 2 F 1is
also increased by 0.39 to 0.51. Mistral-7B exhibited a similar
behavior, from 0.69 to 0.74 for probe 1 and 0.29 to 0.33 in
probe 2. Interestingly,G 1+G 2is slightly worse thanG 2
alone, indicating that the most valuable information in these
models is the AD graph instead of being part of the union
with T2DM knowledge. The above improvements suggest that
smaller models that possess less built-in biomedical knowledge
can benefit meaningfully in terms of structured biomedical
knowledge to fill in the gaps in their parameter knowledge.
Performance inG 3showed a downward spike, even though
there was no conclusive evidence as to why. In the models,
except in Mistral and Mixtral, theG 3graph reduces the F 1.
An interesting case of the Llama-3.1-8B Instruct model shows
that the combined larger graph:G 1+G 2+G 3dropped the
performance from 0.85 to 0.65 on probe 1. 0.65 was the same
value as when usingG 2alone, and was even less (0.62) on
G1alone. Probe 2 was quite similar, as we saw a drop in the
combined graph ofG 1+G2+G3fom 0.50 to 0.40 macro F 1.
We hypothesized that a larger KG may add irrelevant relations,
and our findings validate this, meaning that the broader graph
may mislead LLMs, exposing them to unfounded relations.
Upon adding up the improvement over the No-RAG baseline
across all the models and all the probes, one can easily see that
G2is positive (mean improvement +0.006 F 1) andG 1,G3and
the combination of graphs have negative mean improvements
(–0.04 to -0.01). Therefore, the graph on AD (G 2) is the only
domain-specific KG that was helpful.
V. DISCUSSION
It is experimentally demonstrated that the source of the
retrieval and the scope of retrievalG 1,G2,G3, and their
combinations directly influence the quality of answers. Since
Probe 1 is derived out ofG 3, the resulting merged KG is
likely to include the accurate facts sought by a large number
of questions; whereas Probe 2 is created out ofG 1∩G2, thus
signals that are similar across the two domains of diseases have
a stronger influence. In both probes, the addition of additional
sources (G 1,G2,G3) does not assure higher accuracy: the
addition of breadth raises recall but may introduce many other
unrelated distracting facts that reduce precision.
A. Significance
We provide statistical significance of each condition (three
independent runs each model-probe-system) (paired with
paired two-samplet-tests against No-RAG):∗p <0.05,
∗∗p <0.01,∗∗∗p <0.001. Table VI and VII only indicate
differences that are of significance at these values; non-stars
would not be statistically different to the baseline atα=0.05.

TABLE IV: Results of Probe 1, Averaged across the three temperatures (0.0, 0.2 and 0.5)
Systems
Model MetricG 1 G1+G2 G1+G2+G3 G2 G3 No-RAG
Anthropic.Claude-3-HaikuAcc/ F 1Micro 0.84 / 0.840.92/0.92 0.93/0.93 0.93/0.930.89 / 0.890.96/0.96
Macro P/R/F 1 0.87 / 0.84 / 0.840.93/0.93/0.93 0.94/0.93/0.93 0.94/0.94/0.930.91 / 0.90 / 0.90 0.91 / 0.91 / 0.91
GPT-OSS-20BAcc/ F 1Micro 0.87/0.870.85 / 0.85 0.82 / 0.82 0.82 / 0.82 0.84 / 0.84 0.91 / 0.91
Macro P/R/F 1 0.88/0.87/0.870.87 / 0.86 / 0.86 0.85 / 0.83 / 0.82 0.84 / 0.82 / 0.82 0.87 / 0.85 / 0.85 0.92 / 0.92 / 0.91
Llama-3.1-8B-InstructAcc/ F 1Micro 0.69 / 0.69 0.63 / 0.63 0.65 / 0.65 0.66 / 0.66 0.67 / 0.67 0.83 / 0.83
Macro P/R/F 1 0.76 / 0.66 / 0.67 0.69 / 0.62 / 0.62 0.74 / 0.63 / 0.65 0.71 / 0.65 / 0.65 0.74 / 0.65 / 0.66 0.85 / 0.82 / 0.83
Llama-3.3-70B-InstructAcc/ F 1Micro 0.86 / 0.86 0.91 / 0.91 0.92 / 0.92 0.91 / 0.910.92/0.920.95 / 0.95
Macro P/R/F 1 0.72 / 0.71 / 0.71 0.91 / 0.91 / 0.91 0.93 / 0.92 / 0.92 0.91 / 0.90 / 0.900.92/0.92/0.92 0.96/0.96/0.96
Mistral-7B-Instruct-v0.3Acc/ F 1Micro 0.67 / 0.67 0.68 / 0.68 0.69 / 0.69 0.73 / 0.73 0.72 / 0.72 0.67 / 0.67
Macro P/R/F 1 0.79 / 0.67 / 0.70 0.75 / 0.69 / 0.70 0.77 / 0.70 / 0.71 0.78 / 0.73 / 0.74 0.78 / 0.72 / 0.73 0.78 / 0.66 / 0.69
Mixtral-8x7B-v0.1Acc/ F 1Micro 0.83 / 0.83 0.83 / 0.83 0.87 / 0.87 0.88 / 0.88 0.84 / 0.84 0.79 / 0.79
Macro P/R/F 1 0.86 / 0.83 / 0.84 0.82 / 0.80 / 0.80 0.88 / 0.87 / 0.87 0.90 / 0.89 / 0.89 0.86 / 0.85 / 0.84 0.85 / 0.79 / 0.80
Qwen2.5-32B-InstructAcc/ F 1Micro 0.85 / 0.85 0.90 / 0.90 0.89 / 0.89 0.89 / 0.89 0.87 / 0.87 0.91 / 0.91
Macro P/R/F 1 0.84 / 0.84 / 0.84 0.90 / 0.89 / 0.89 0.89 / 0.88 / 0.88 0.89 / 0.87 / 0.88 0.87 / 0.87 / 0.87 0.92 / 0.90 / 0.91
TABLE V: Results of Probe 2, Averaged across the three temperatures (0.0, 0.2 and 0.5)
Systems
Model MetricG 1 G1+G2 G1+G2+G3 G2 G3 No-RAG
Anthropic.Claude-3-HaikuAcc/ F 1Micro 0.58/0.58 0.60/0.60 0.62/0.620.61 / 0.61 0.53 / 0.53 0.61 / 0.61
Macro P/R/F1 0.53 / 0.52 / 0.50 0.57 / 0.57 /0.55 0.60/0.60/0.570.56 / 0.55 / 0.54 0.50 / 0.49 / 0.47 0.54 / 0.55 / 0.55
GPT-OSS-20BAcc/ F 1Micro 0.46 / 0.46 0.50 / 0.50 0.48 / 0.48 0.50 / 0.50 0.44 / 0.44 0.43 / 0.43
Macro P/R/F1 0.56 / 0.45 / 0.45 0.56 / 0.47 / 0.46 0.57 / 0.47 / 0.460.63/ 0.51 / 0.480.54/ 0.46 / 0.41 0.49 / 0.42 / 0.39
Llama-3.1-8B-InstructAcc/ F 1Micro 0.53 / 0.53 0.51 / 0.51 0.48 / 0.48 0.56 / 0.560.54/0.540.61 / 0.61
Macro P/R/F1 0.44 / 0.41 / 0.41 0.47 / 0.41 / 0.42 0.40 / 0.38 / 0.37 0.52 / 0.44 / 0.45 0.45 / 0.41 / 0.40 0.51 / 0.50 / 0.50
Llama-3.3-70B-InstructAcc/ F 1Micro 0.58/0.580.54 / 0.54 0.54 / 0.54 0.59 / 0.59 0.52 / 0.52 0.56 / 0.56
Macro P/R/F1 0.52 / 0.51 / 0.51 0.47 / 0.47 / 0.47 0.51 / 0.50 / 0.49 0.53 / 0.52 / 0.52 0.48 / 0.49 /0.480.52 / 0.49 / 0.49
Mistral-7B-Instruct-v0.3Acc/ F 1Micro 0.36 / 0.36 0.40 / 0.40 0.39 / 0.39 0.44 / 0.44 0.35 / 0.35 0.38 / 0.38
Macro P/R/F1 0.32 / 0.26 / 0.23 0.46 / 0.31 / 0.29 0.36 / 0.28 / 0.25 0.46 / 0.34 / 0.33 0.38 / 0.27 / 0.24 0.40 / 0.31 / 0.29
Mixtral-8x7B-v0.1Acc/ F 1Micro 0.47 / 0.47 0.58 / 0.58 0.57 / 0.57 0.56 / 0.56 0.47 / 0.47 0.46 / 0.46
Macro P/R/F1 0.44 / 0.43 / 0.42 0.55 / 0.54 / 0.54 0.51 / 0.50 / 0.50 0.52 / 0.51 / 0.51 0.45 / 0.44 / 0.43 0.47 / 0.39 / 0.39
Qwen2.5-32B-InstructAcc/ F 1Micro 0.56 / 0.56 0.56 / 0.56 0.55 / 0.550.64/0.640.51 / 0.510.65/0.65
Macro P/R/F10.60/0.60/0.54 0.59/0.61/0.550.53 / 0.55 / 0.52 0.62 /0.65/0.610.51 /0.53/0.48 0.61/0.63/0.60
G
 G+G
 G+G+G
 G
 G
 No-RAG
System Configuration0.600.650.700.750.800.850.900.951.00Score
Probe 1 (Avg. across T = 0.0, 0.2, 0.5): Trend across RAG Configurations
F1 Micro (mean)
Macro F1 (mean)
(a) Probe 1
G
 G+G
 G+G+G
 G
 G
 No-RAG
System Configuration0.300.350.400.450.500.550.600.650.70Score
Probe 2 (Avg. across T = 0.0, 0.2, 0.5): Trend across RAG Configurations
F1 Micro (mean)
Macro F1 (mean) (b) Probe 2
Fig. 2: Trends across RAG configurations for the averaged F 1scores of the models in Table IV and V (averaged over
T={0.0,0.2,0.5}).
B. How RAG scope interacts with each probe
a) Probe 1:This probe was constructed out of direct
existing knowledge, and systems that usedG 3directly, ac-
cessed a single graph that happened to answer all Probe 1
facts; however, these did not help these systems, but rather
introduced extraneous context, which caused failure:
•Large generalist models (Llama-3.3-70B). Adding ad-
ditional, tangential evidence, on the basis ofG 1only,
generated a huge, significant decline (∗∗∗), smaller, but
significant, declines in the case ofG 2,G3, andG 1+G2
(∗–∗∗). The model already captures a lot of knowledge
that is required to respond to health-based questions.•Mid-sized mixture models (Mixtral-8×7B).G 2yields a
significantgain (∗∗) andG 1+G2+G3a smaller gain (∗),
suggesting that well-scoped AD evidence helps when
merged facts are relevant but not memorized.
•Smaller instruction models (Llama-3.1-8B). Against all
KG-RAG settings, the performance is significantly low.
One-sample t-test incorrect on Probe 1 (∗–∗∗∗): Sensitive
to distractors in the case of retrieval of multiple seman-
tically related facts.
•Other models. Mistral-7B shows no reliable change on
Probe 1 (mostly unstarred), while Qwen-2.5-32B shows
small but significant drops forG 1andG 3(∗).

TABLE VI: RAG system comparison on Probe 1: Macro-F1 and significance vs. No-RAG
Model No-RAGG 1 G2G1+G2G3G1+G2+G3
Llama-3.1-8B-Instruct 0.83 0.67∗∗∗0.65∗∗0.62∗∗∗0.66∗∗∗0.65
Mistral-7B-Instruct-v0.3 0.69 0.70 0.74 0.70 0.73 0.71
Mixtral-8×7B-v0.1 0.80 0.84 0.89∗∗0.80 0.84 0.87∗
Qwen-2.5-32B-Instruct 0.91 0.84∗0.88 0.89 0.87∗0.88
GPT-OSS-20B 0.91 0.87 0.82∗0.86∗0.85∗0.82
Anthropic Claude-3-Haiku 0.91 0.84 0.93 0.93 0.90 0.93
Llama-3.3-70B-Instruct 0.96 0.71∗∗∗0.90∗∗0.91∗0.92∗0.92∗∗
Notes: Stars denote Welch two-samplet-test vs. No-RAG using the three replicates per condition:∗p<.05,∗∗p<.01,∗∗∗p<.001.
TABLE VII: RAG system comparison on Probe 2: Macro-F1 and significance vs. No-RAG
Model No-RAGG 1 G2G1+G2G3G1+G2+G3
Llama-3.1-8B-Instruct 0.50 0.41∗0.45 0.42∗0.40∗0.37∗∗
Mistral-7B-Instruct-v0.3 0.29 0.23∗0.33 0.29 0.24∗0.25
Mixtral-8×7B-v0.1 0.39 0.42 0.51∗0.54∗0.43 0.50∗
Qwen-2.5-32B-Instruct 0.60 0.54∗∗0.61 0.55∗0.48∗∗0.52∗∗
GPT-OSS-Bb 0.39 0.45 0.48∗0.46 0.41 0.46
Anthropic Claude-3-Haiku 0.55 0.50∗0.54 0.55 0.47∗∗0.57∗
Llama-3.3-70B-Instruct 0.49 0.51 0.52 0.47 0.48 0.49
Notes: Stars denote Welch two-samplet-test vs. No-RAG using the three replicates per condition:∗p<.05,∗∗p<.01,∗∗∗p<.001.
Why this pattern?Probe 1 questions reflect the merged
space in existing literature, (AD + T2DM) i.e. different from
just naively joiningG 1andG 2. When retrieval comes from
a narrower domain (G 1orG 2) or an over-broad union
(G1+G2+G3) the context either misses key merged relations
or dilutes them with near-miss facts (e.g., disease-specific
mechanisms that are irrelevant to the asked relation). High
level models, with their substantial common-sense and medical
knowledge inherent in them, can be readily diverted by near-
misses. On the contrary, mid-sized models take advantage of
focused AD signal (G 2) when it coincides with merged facts
of a probe.
b) Probe 2 (formulated onG 1∩G 2).:In this case,
consistent evidence between the two diseases will be of
greatest importance:
•Mixtral-8×7B. All ofG 2,G1+G2,G1+G2+G3are im-
proving significantly (∗), a fact that implies that AD-
centric cues and their combination with T2DM are quite
consistent with the intersection facts.
•Qwen-2.5-32B. The several settings of RAG (G 1,
G1+G2,G3,G1+G2+G3) result in important drops (∗–∗∗)
andG 2is not statistically different to baseline. This indi-
cates accuracy rather than breadth: AD-specific retrieval
is less risky than the domain mixture of this probe.
•Mistral-7B.G 1andG 3indicate minor yet significant
declines (∗); other settings are not significant.
•Llama-3.1-8B. The vast majority of KG-RAG environ-
ments impair performance (∗–∗∗), which once more indi-
cates susceptibility to distractors in smaller models.
•GPT-OSS-20B & Claude-Haiku. Mixed outcomes: some
modest gains (e.g., Haiku withG 1+G2+G3,∗) and several
significant drops where the added context conflicts with
the intersection signal.
•Llama-3.3-70B. None of the significant differences isobserved- in keeping with a strong prior which is neither
aided nor injured by the retrieved snippets.
C. Explaining the patterns of stars
We want to consider if the RAG is useful or harmful. There
are three typical processes that we see:
1) Models are more potential to assist in the event that the
construction space of the probe equals the retrieval space
(Probe 1 withG 3, Probe 2 withG 2orG 1+G2). False
positive (e.g. Probe 2 andG 3) enhances contradictions
and off-target cues and produces∗/∗∗drops.
2) The recall is augmented by unions of (G 1+G2),
G1+G2)+G3) but it bring heterogeneous evidence to it.
Otherwise, the models will overfittingly adapt to spurious
but fluent responses, and will produce large drops even
with truthful facts.
3) Larger models tolerate imperfect retrieval better, (as we
see in Table VI and VII where we have often unstarred or
mixed effects. However, these models are also susceptible
to highly plausible distractors; smaller models rely more
heavily on retrieved text and thus amplify retrieval noise,
leading to frequent∗–∗∗∗drops.
D. Temperature effects
System-to-system changes in temperature (0, 0.2, 0.5) had
little influence on the conclusions of the RAG type: most in-
system patterns were statistically insignificant, and significant
ones were small compato system differences. In practice, these
probes are mainly tuned by RAG choice; temperature tuning
must always come second after a well-scoped graph is chosen
and a strong retriever is selected.
E. Frontier LLMs and Non-expert Baseline
We also posed both probes to two general-purpose LLMs
and a non-expert human being.

TABLE VIII: Gemini, ChatGPT, and non-expert human per-
formance on Probe 1 and Probe 2.
Agent Probe 1 (100) Probe 2 (110)
Gemini 2.5 Pro 99/100 (99%) 69/110 (62.7%)
ChatGPT 5 Thinking 98/100 (98%) 77/110 (70.0%)
Naive Human (no medical knowledge) 38/100 (38%) 30/110 (27.3%)
Table VIII contrasts the accuracy of Gemini, ChatGPT, and
a non-expert human across Probe 1 and Probe 2. Both frontier
models achieve near-ceiling performance on Probe 1 (99%
and 98%, respectively), revealing that the initial probe tasks
are largely saturated. In contrast, Probe 2 introduces greater
conceptual and retrieval difficulty, producing a notable accu-
racy decline—Gemini drops by 36.3% points, while ChatGPT
falls by 28%. The naive human baseline shows only a modest
relative decline, but its overall accuracy remains far below
that of the models. The naive human baseline is a test for
random guessing, as the people who took the test did not
have prior knowledge of the field and were encouraged to
guess randomly.
When compared with the open-weight counterparts in
Tables IV and V, the frontier models display markedly
higher stability and consistency. Anthropic Claude-3-Haiku,
the strongest among the smaller systems, achieved accuracies
around 96% on Probe 1 but fell to roughly 61% on Probe
2, a pattern mirrored by all the other models. This shows
that Probe 2 acts as a discriminative benchmark, separating
models with generalizable reasoning (ChatGPT and Claude-
tier systems) from those whose performance is more retrieval-
anchored or domain-fragile.
ChatGPT’s smaller degradation indicates stronger robust-
ness to task complexity and class imbalance, aligning with
its narrower gap between Micro and Macro F 1in Probe 2.
Gemini’s sharper fall suggests sensitivity to rare or ambiguous
cases, reinforcing that Probe 2 better exposes differential rea-
soning depth rather than surface-level recall. Overall, frontier
LLMs far exceed human baselines, yet their relative separation
on Probe 2 highlights the probe’s discriminative power for
evaluating balanced generalization.
F . Error Analysis
We covered the following errors shown in Table IX: (1)
Directionality flips (pickedB→Ainstead ofA→B), (2)
Two-hop chain order errors (mis-orderedA→B→Cpairs),
(3) Negation/exception misreads (e.g., “is not associated” /
“pick the exception”), (4) Immediate vs. downstream cause
(selected a true but non-proximal effect), (5) Undefined token
guesswork (e.g., ambiguous labels like “FX protein”) and (6)
Overweighting canonical AD triad (Aβ/tau/neuroinflammation
selected when vascular/metabolic was targeted)
G. Recommendations
•When adding several graphs, do so conditionally: by
putting in a ranker that favors off-topic passages being
demoted and evidence being given precedence that is
found in both domains of the disease.TABLE IX: Miss classification on Probe 2 (33 total errors).
Category Count Percent
Directionality flips 9 27.27%
Two-hop chain order 8 24.24%
Negation/exception 6 18.18%
Immediate vs. downstream 5 15.15%
Undefined token guesswork 3 9.09%
AD-triad overweighting 2 6.06%
Total 33 100%
•Smaller models benefit most from clean, highly relevant
passages, while larger models need less context overall
but require stricter filtering to avoid distractors.
•Given three runs per condition, using star-coding
(∗/∗∗/∗∗∗) is essential to avoid over-interpreting small
differences in mean values that might not be statistically
meaningful.
a) Threats to validity:(1) Only three replicates per cell
limit power; results marked unstarred could still harbor small
effects. (2) Factual mix reproducible by our KG construction
decisions (entity normalization, relation filtering, and co-
reference backbone) determines the factual mix that can be
accessed, and any improvement or error in this regard directly
translates into the results of QA. (3) Prompting and re-ranking
were held fixed; stronger retrieval/re-ranking may change the
balance between precision and recall.
VI. CONCLUSION
This work studied domain-specific KG-RAG for healthcare
LLMs under realistic design choices: graph scope (G 1/G2/G3),
probe definition (merged vs. intersection), model capacity, and
decoding temperature. Three consistent lessons emerged.
(1) Match retrieval scope to task scope.Probe 1 (merged
AD -T2DM relations) is best served by retrieval that is to
a large extent coincidental with the relations (e.g.G 3or
more specific union ofG 2), whereas Probe 2 (intersection-
style questions) is best served byG 2orG 1+G2rather than
excessively broad unions.
(2) Favor precision over breadth.Uniting graphs increases
recall but also injects heterogeneous evidence; without strong
ranking/filters, distractors lower accuracy. Precision-first re-
trieval with scope-matched graphs is more reliable.
(3) Right-size KG-RAG to model capacity.Smaller and
mid-sized models gain the most from clean, well-scoped
retrieval (notably withG 2). Larger models often match or
surpass KG-RAG withNo-RAGon merged-scope questions,
reflecting strong parametric knowledge and a higher sensitivity
to noisy context.
Practically, teams should: (i) choose graphs that match
their question distribution; (ii) deploy rankers/filters that shed
off-topic spans and reward corroborated evidence; and (iii)
tailor retrieval strictness to model size. Future work includes
risk-aware reranking, dynamic graph selection conditioned
on query intent, and multi-hop reasoning that exploits KG
structure without flooding prompts with near-miss facts.

VII. LIMITATIONS
We highlight key limitations and threats to validity of this
study.
1) Our graphs are centered on T2DM (G 1), AD (G 2) and
AD+T2DM (G 3) space. Findings might not be related to
other conditions, specialities, non-PubMed corpora, and
multilingual environments
2) Although co-reference and canonicalization have been
improved, the errors in entity connecting, synonym merg-
ing, and relation extracting may be carried over to the
retrieval to produce plausible yet off-target evidence
that diminishes accuracy. We used rule-based filtering to
reduce false positives, but the graph is not 100% perfect.
3) We did not actively search retriever depth, hybrid lexical
neural retrieval or learning-to-rank rerankers. A more
powerful ranking stack that has the potential to minimize
distractors, particularly on union graphs.
4) Accuracy, Macro P/R/F1, and Micro F1 fail to reflect on
calibration, factual grounding to primary sources and clin-
ical harm potential. The fact-checking based on human
judgment and reference was out of the question.
5) The findings on smaller vs. larger models might not be
generalizable to other architectures, tokenizer selection,
alignment processes or domain-trained checkpoints.
6) Our setup omits latency, cost, privacy, and PHI-handling
constraints crucial in clinical workflows in this pilot
study. Deployment-time retrieval drift, updates to liter-
ature, and governance requirements are not modeled.
7) Since we had to run on a few resources, we did not
carry out large multi-seed reruns, ablations of KG pre-
processing steps, or confidence-interval reporting over
all settings; here, some of the effects may be due to
stochasticity.
ACKNOWLEDGMENT
The authors would like to thank Paul Josiah, Toluwalase Kunle-
John and Elizabeth Oyegoke for volunteering to take the test for the
naive human baseline.
REFERENCES
[1] A. Vb, D. K. Jha, and S. Bhattacharjee, “Global trends and burden
of diabetes: A comprehensive review of global insights and emerging
challenges,”Current Journal of Applied Science and Technology, vol. 44,
no. 7, pp. 134–150, 2025.
[2] R. López-Antón, “Recent advances in alzheimer’s disease research: from
biomarkers to therapeutic frontiers,”Biomedicines, vol. 12, no. 12, p.
2816, 2024.
[3] M. A Kamal, S. Priyamvada, A. N Anbazhagan, N. R Jabir, S. Tabrez,
and N. H Greig, “Linking alzheimer’s disease and type 2 diabetes
mellitus via aberrant insulin signaling and inflammation,”CNS &
Neurological Disorders-Drug Targets (Formerly Current Drug Targets-
CNS & Neurological Disorders), vol. 13, no. 2, pp. 338–346, 2014.
[4] A. Association, “2015 alzheimer’s disease facts and figures,”Alzheimer’s
& Dementia, vol. 11, no. 3, pp. 332–384, 2015.
[5] M. D. Mezey, E. L. Mitty, M. M. Bottrell, G. C. Ramsey, and T. Fisher,
“Advance directives: Older adults with dementia,”Clinics in Geriatric
Medicine, vol. 16, no. 2, pp. 255–268, 2000.
[6] M. Schwarzinger and C. Dufouil, “Forecasting the prevalence of demen-
tia,”The Lancet Public Health, vol. 7, no. 2, pp. e94–e95, 2022.
[7] G. Logroscino, “Prevention of alzheimer’s disease and dementia: the
evidence is out there, but new high-quality studies and implementation
are needed,” 2020.[8] R. R. Kalyani, J. J. Neumiller, N. M. Maruthur, and D. J. Wexler,
“Diagnosis and treatment of type 2 diabetes in adults: A review,”JAMA,
2025.
[9] Y . Zheng, S. H. Ley, and F. B. Hu, “Global aetiology and epidemiology
of type 2 diabetes mellitus and its complications,”Nature reviews
endocrinology, vol. 14, no. 2, pp. 88–98, 2018.
[10] C. Procaccini, M. Santopaolo, D. Faicchia, A. Colamatteo, L. Formisano,
P. de Candia, M. Galgani, V . De Rosa, and G. Matarese, “Role of
metabolism in neurodegenerative disorders,”Metabolism, vol. 65, no. 9,
pp. 1376–1390, 2016.
[11] M. Barbagallo and L. J. Dominguez, “Type 2 diabetes mellitus and
alzheimer’s disease,”World journal of diabetes, vol. 5, no. 6, p. 889,
2014.
[12] C. Bellia, M. Lombardo, M. Meloni, D. Della-Morte, A. Bellia, and
D. Lauro, “Diabetes and cognitive decline,”Advances in clinical chem-
istry, vol. 108, pp. 37–71, 2022.
[13] M. Kciuk, W. Kruczkowska, J. Gał˛ eziewska, K. Wanke, ˙Z. Kałuzi ´nska-
Kołat, M. Aleksandrowicz, and R. Kontek, “Alzheimer’s disease as
type 3 diabetes: Understanding the link and implications,”International
Journal of Molecular Sciences, vol. 25, no. 22, p. 11955, 2024.
[14] E. Blázquez, E. Velázquez, V . Hurtado-Carneiro, and J. M. Ruiz-
Albusac, “Insulin in the brain: its pathophysiological implications for
states related with central insulin resistance, type 2 diabetes and
alzheimer’s disease,”Frontiers in endocrinology, vol. 5, p. 161, 2014.
[15] E. Abdelgadir, R. Ali, F. Rashid, and A. Bashier, “Effect of metformin
on different non-diabetes related conditions, a special focus on malignant
conditions: review of literature,”Journal of clinical medicine research,
vol. 9, no. 5, p. 388, 2017.
[16] S. M. De la Monte, “Type 3 diabetes is sporadic alzheimer’s disease:
mini-review,”European neuropsychopharmacology, vol. 24, no. 12, pp.
1954–1960, 2014.
[17] Z. Kroner, “The relationship between alzheimer’s disease and diabetes:
Type 3 diabetes?”Alternative Medicine Review, vol. 14, no. 4, p. 373,
2009.
[18] J. Leszek, E. Trypka, V . V Tarasov, G. Md Ashraf, and G. Aliev, “Type
3 diabetes mellitus: a novel implication of alzheimers disease,”Current
topics in medicinal chemistry, vol. 17, no. 12, pp. 1331–1335, 2017.
[19] A. Lacerda, G. Pappa, A. C. M. Pereira, W. M. Jr, and A. G.
de Almeida Barros, “Evaluation of medical large language models:
Taxonomy, review, and directions.”
[20] Y . Wang, R. E. Mercer, F. Rudzicz, S. S. Roy, P. Ren, Z. Chen, and
X. Wang, “Trustworthy medical question answering: An evaluation-
centric survey,”arXiv preprint arXiv:2506.03659, 2025.
[21] W. Wu, H. Wang, B. Li, P. Huang, X. Zhao, and L. Liang, “Multirag:
a knowledge-guided framework for mitigating hallucination in multi-
source retrieval augmented generation,” in2025 IEEE 41st International
Conference on Data Engineering (ICDE). IEEE, 2025, pp. 3070–3083.
[22] H. N. Yassine and C. E. Finch, “Apoe alleles and diet in brain aging and
alzheimer’s disease,”Frontiers in aging neuroscience, vol. 12, p. 150,
2020.
[23] K. Jabeen, K. Rehman, and M. S. H. Akash, “Genetic mutations of
apoeε4 carriers in cardiovascular patients lead to the development of in-
sulin resistance and risk of alzheimer’s disease,”Journal of biochemical
and molecular toxicology, vol. 36, no. 2, p. e22953, 2022.
[24] L. Exalto, R. Whitmer, L. Kappele, and G. Biessels, “An update on type
2 diabetes, vascular dementia and alzheimer’s disease,”Experimental
gerontology, vol. 47, no. 11, pp. 858–864, 2012.
[25] K. Soman, P. W. Rose, J. H. Morris, R. E. Akbas, B. Smith, B. Peetoom,
C. Villouta-Reyes, G. Cerono, Y . Shi, A. Rizk-Jacksonet al., “Biomed-
ical knowledge graph-optimized prompt generation for large language
models,”Bioinformatics, vol. 40, no. 9, p. btae560, 2024.
[26] S. Anuyah, M. M. Kaushik, K. Dwarampudi, R. Shiradkar, A. Durresi,
and S. Chakraborty, “Automated knowledge graph construction using
large language models and sentence complexity modelling,”arXiv
preprint arXiv:2509.17289, 2025.
[27] Z. Gao, Y . Cao, H. Wang, A. Ke, Y . Feng, X. Xie, and S. K. Zhou, “Frag:
A flexible modular framework for retrieval-augmented generation based
on knowledge graphs,”arXiv preprint arXiv:2501.09957, 2025.
[28] J. Linders and J. M. Tomczak, “Knowledge graph-extended re-
trieval augmented generation for question answering,”arXiv preprint
arXiv:2504.08893, 2025.