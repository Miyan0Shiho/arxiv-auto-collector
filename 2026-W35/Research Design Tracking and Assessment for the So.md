# Research Design Tracking and Assessment for the Social Sciences

**Authors**: Marco Rovera, Sergiu Burlacu, Dominique Cappelletti, Alessio Tomelleri, Sonia Marzadro, Martina Bazzoli, Annalisa Tassi, Jessica Gagete-Miranda

**Published**: 2026-08-27 12:36:17

**PDF URL**: [https://arxiv.org/pdf/2608.27049v1](https://arxiv.org/pdf/2608.27049v1)

## Abstract
Reliable assessment of causal research designs in the social sciences is critical for evidence-based policy-making, yet has so far relied entirely on manual expert analysis. We introduce Automated Research Design Tracking and Assessment (ARDTrA), a task that involves detecting the research design used in a paper and assessing the quality of its application. We create an expert-annotated dataset of papers covering six families of counterfactual research designs and evaluate the task using a multi-turn RAG-based conversational pipeline. Across four retrieval strategies, four LLMs and six embedding models, we find that passage length is the main driver of performance, explaining 52-66% of the variance. A per-research-design analysis also shows that human and machine difficulty do not align: the designs that prove hardest for the system are not those on which expert annotators disagree most, pointing to two independent sources of task difficulty.

## Full Text


<!-- PDF content starts -->

Research Design Tracking and Assessment for the Social Sciences
Marco Rovera, Sergiu Burlacu, Dominique Cappelletti, Alessio Tomelleri,
Sonia Marzadro, Martina Bazzoli, Annalisa Tassi, Jessica Gagete-Miranda
Fondazione Bruno Kessler
Trento, Italy
m.rovera@fbk.eu
{sburlacu, dcappelletti, atomelleri, marzadro,
bazzoli, atassi, jgagetemiranda}@irvapp.it
Abstract
Reliable assessment of causal research designs
in the social sciences is critical for evidence-
based policy-making, yet has so far relied en-
tirely on manual expert analysis. We introduce
Automated Research Design Tracking and As-
sessment (ARDTrA), a task that involves de-
tecting the research design used in a paper and
assessing the quality of its application. We cre-
ate an expert-annotated dataset of papers cov-
ering six families ofcounterfactualresearch
designs and evaluate the task using a multi-turn
RAG-based conversational pipeline. Across
four retrieval strategies, four LLMs and six em-
bedding models, we find that passage length
is the main driver of performance, explaining
52–66% of the variance. A per-research-design
analysis also shows that human and machine
difficulty do not align: the designs that prove
hardest for the system are not those on which
expert annotators disagree most, pointing to
two independent sources of task difficulty.
1 Introduction
Identifying and assessing the causal research de-
signs employed in social science papers is a key
yet underexplored NLP task. The demand for such
automation is driven by a concrete use case: when
policy advisors survey the scientific literature to
inform decision-making on a specific topic, they
need to quickly isolate studies whose methodol-
ogy supports credible causal claims. In the social
sciences, this translates into selecting studies that
adopt acounterfactualresearch design (RD), as for
causal effect estimation, contributions employing
such designs are generally regarded as providing
more reliable evidence. Furthermore, even among
papers that adopt a counterfactual RD, there are
important differences in how the method is applied,
which can affect the quality and robustness of the
resulting evidence. Due to the specialist knowledge
required, regarding both RDs and their applicationcaveats, this literature selection and assessment
process has so far been carried out manually by
social scientists. Despite its practical relevance,
the task has received limited attention within the
NLP community, likely due to the lack of both
a formalized analytical framework and a public,
expert-curated evaluation dataset. In this paper
we address these gaps by introducing Automated
Research Design Tracking and Assessment (ARD-
TrA). The task involves two steps: (1) detecting
which Research Design a paper employs, and (2)
assessing the credibility, validity, and robustness of
its application in the specific application context.
Both steps require locating and interpreting fine-
grained methodological details scattered through-
out a paper. Using an evaluation framework de-
signed by experts in counterfactual policy evalua-
tion, we first create and manually annotate an orig-
inal corpus of 140 scientific papers drawn from a
diverse range of social science fields, spanning ap-
plied economics, sociology, political science, and
related disciplines. The framework considers six
families of RDs identified by the causal inference
literature (Rubin, 1974; Holland, 1986) and subse-
quently consolidated by the so-called “credibility
revolution” (Angrist and Pischke, 2010), and de-
velops anad hocanalytical scheme for each one.
We then use the annotated dataset to evaluate dif-
ferent retrieval strategies within a multi-turn con-
versational RAG pipeline, across a set of LLMs
and embedding models. From an NLP perspective,
our primary goal is to evaluate the performance of
various indexing and retrieval strategies on a chal-
lenging domain-specific document analysis task,
where answers require synthesizing fine-grained
methodological information from long scientific
texts. From a Computational Social Science per-
spective, the study additionally demonstrates the
feasibility of automating RD assessment and pro-
vides social scientists with a reliable estimate of
current system performance. The contribution of
arXiv:2608.27049v1  [cs.CL]  27 Aug 2026

this paper is threefold:i)we define the task of
Research Design Tracking and Assessment, along
with an expert-designed analytical schema;ii)we
create a dataset of 140 papers annotated by do-
main experts according to this schema; andiii)we
benchmark different retrieval strategies within a
multi-turn RAG-based conversational agent, pro-
viding a systematic comparison across chunking
methods, embedding models, and LLMs.
2 Related Work
In the last few years, a growing body of litera-
ture has focused on automatic analysis of causal
inference methods in the social sciences. Currie
et al. (2020) used basic text-mining techniques
to track the incidence of keywords associated
with causal inference methods in NBER1work-
ing papers and five leading economics journals.
Goldsmith-Pinkham (2024) extended this analysis
to additional fields including finance and macroe-
conomics. More recently, Garg and Fetzer (2025)
employed GPT-4o-mini to extract causal inference
methods from NBER and CEPR2working paper se-
ries, validating their approach against the annotated
dataset of Brodeur et al. (2024), which covers four
RDs (DiD, RCT, RDD, IV) across 1,106 economics
articles. While reporting accuracy scores between
70% and 93% depending on the RD, this study
uses a single LLM without systematic evaluation
of retrieval configurations, and focuses exclusively
on RDdetectionwithout addressing the quality of
the method’s application. Other related efforts in-
clude Hooper et al. (2024), who combine NLP with
causal mapping for extracting causal evidence from
policy literature, and Imai and Nakamura (2026),
who address causal inference methodology from
a complementary perspective. OpenScholar (Asai
et al., 2026) used RAG on 45 million open-access
papers, showing that retrieval-augmented systems
substantially outperform parametric LLMs on sci-
entific synthesis tasks. Our work differs in two
ways: we introduce theassessmentdimension, i.e
we evaluate not only which RD a paper uses but
howcrediblyit is applied, and we provide a sys-
tematic NLP evaluation across retrieval strategies,
embedding models, and LLMs.
1https://www.nber.org/
2https://www.cepr.org/3 Counterfactual Research Designs
In the social sciences, the credibility of a causal
claim depends not only on the method employed
but also on the specific empirical setting in which
it is applied, as thecausal identificationstrategy
plays a major role. For instance, in Card and
Krueger’s study of the employment effects of the
minimum wage (Card and Krueger, 1993), the
Difference-in-Differences design is credible pre-
cisely because it exploits a specific institutional
contrast: New Jersey increased its minimum wage
while neighboring Pennsylvania did not, provid-
ing a context-specific comparison group. Causal
claims are typically supported through research
designs that attempt to construct or approximate
such a comparison group, one that estimates the
outcome that would have been observed in the ab-
sence of the intervention or treatment under study.
Rather than relying exclusively on statistical asso-
ciations, these approaches aim to exploit sources
of variation that can plausibly isolate causal effects.
Although the boundaries between causal and non-
causal empirical strategies are not always sharp, the
literature associated with the “credibility revolu-
tion” has identified a relatively consolidated set of
design-based approaches that are widely regarded
as suitable for causal inference in both experimen-
tal and observational settings (Imbens, 2024). In
this paper, we operationalize this literature by fo-
cusing on six broad families of research designs
that cover the vast majority of causal designs en-
countered in applied social science research:
1.Experimental Designs (ED), e.g. field exper-
iments, survey experiments and lab experi-
ments (Fisher and Fisher, 1966; Athey and
Imbens, 2017);
2.Threshold-based Designs (TbD), e.g. regres-
sion discontinuity, regression kink or bunch-
ing (Thistlethwaite and Campbell, 1960; Cat-
taneo and Titiunik, 2022);
3.Instrumental Variables (IV) (Angrist et al.,
1996; Mogstad and Torgovitsky, 2018);
4.Selection-on-Observables (SoO), e.g. regres-
sion adjustment, matching, weighting, doubly
robust (Rubin, 1974; Heckman, 1979; Stuart,
2010; Imbens, 2015);
5.Difference-in-Differences (DiD) and related
designs, e.g., event studies, changes-in-

changes and triple differences (Ashenfelter
and Card, 1985; Roth et al., 2023);
6.Synthetic Control Methods (SCM) and related
designs, e.g. augmented synthetic control,
synthetic difference-in-differences (Abadie
and Gardeazabal, 2003; Abadie et al., 2010;
Abadie, 2021).
It is worth noting that the analytical framework
proposed in this work (see Section 4.1 and Ap-
pendix D) is not specific to economics: many of
its elements originate in, and are widely shared
across, the empirical sciences as part of a com-
mon causal-inference toolkit (Imbens and Rubin,
2015; Hernán and Robins, 2020). Experimental
Designs, for instance, are foundational in psychol-
ogy and medicine. Selection-on-Observables meth-
ods are central to epidemiology and public health,
while other research designs, such as Difference-
in-Differences, are increasingly adopted in related
fields, like health policy (Feng et al., 2025). Be-
yond academia, the same designs increasingly un-
derpin large-scale experimentation and observa-
tional causal analysis in industry, particularly at
large technology firms (Larsen et al., 2024). Collab-
orations between biostatisticians, econometricians
and methodological experts in political science
and sociology are common, and cross-referencing
across these literatures is the norm in published
work. While some differences exist, terminology
is largely shared across disciplines, and our anno-
tation scheme explicitly accounts for differences
in use of certain terms (for example, “unconfound-
edness”, “ignorability”, and “conditional indepen-
dence assumption”, which refer to the same core as-
sumption in Selection-on-Observables, are favored
in different disciplines).
4 Dataset
The dataset3consists of 140 papers in English, sam-
pled from an initial pool of 6,554 articles drawn
from journals selected using the Combes-Linnemer
classification (Combes and Linnemer, 2010) and
IDEAS/RePEc rankings, spanning multiple quality
tiers and subfields, such as labour, health, educa-
tion, environmental, urban and regional economics
and was designed to maintain a reasonable balance
3Due to copyright restrictions, as most papers in the dataset
are not open access, we release only the paper metadata (au-
thors, title, journal) in the Supplementary Materials, along
with the annotations and the analytical framework.between papers that use counterfactual and non-
counterfactual RDs on the one hand, and between
the various RDs on the other. The distribution of
RDs across the dataset is depicted in Figure 1.
79Non-CF
11
ED9
TbD9 IV11SoO11DiD
10SCM
Figure 1: Distribution of RDs in the ARDTrA dataset.
4.1 Analytical Framework
The analytical schema developed for ARDTrA is
used for the manual annotation of papers in the
dataset. The full text is provided in Appendix D.
The schema consists of a set of methodological
questions, each provided with instructions for an-
swering (one or more valid answers) and with a
finite set of possible answer options. The dataset
was annotated by 8 domain experts with special-
ized knowledge of causal inference methodology
and research design application. The task of each
annotator is to read through the paper and answer
the full set of questions. The schema is divided
into two parts, which define the two subtasks: RD
Identification and RD Assessment.
RD IdentificationThis subtask comprises 6
questions, arranged from general to specific, that
progressively narrow down the methodological pro-
file of a paper. The first three questions char-
acterize the paper along broad dimensions: its
type (e.g. empirical, theoretical, methodological,
review), the nature of its data and research de-
sign (quantitative vs. qualitative), and the type of
analysis it performs (causal, descriptive, or predic-
tive/simulation). Question 4 then asks whether the
paper employs any of the six counterfactual RDs
introduced in Section 3, as a binary yes/no decision.
If the answer is negative, the paper is classified as
non-counterfactual and the analysis terminates. If a
counterfactual design is detected, Question 5 asks
the annotator to identify which of the six families
constitutes the paper’s main RD. Finally, Question
6 captures whether the paper employs any addi-
tional counterfactual RDs beyond the primary one,
allowing for multi-method designs. Papers identi-
fied as counterfactual at this stage proceed to the
RD Assessment subtask.

Krippendorff’sα
Full Task .814
RD Identification .926
RD Assessment .753
Experimental Designs (ED) .821
Synthetic Control Methods (SCM) .833
Threshold-based Desings (TbD) .784
Difference-in-Differences (DiD) .688
Instrumental Variables (IV) .681
Selection-on-Observables (SoO) .649
Table 1: Krippendorff’s αbetween 3 expert annotators,
computed on the I-AA set (24 papers). Specific RD’s
scores have been computed on the 12 papers using a
counterfactual RD (2 papers for RD).
Annotator pair Full Task RD-I RD-A
A1 - A2 .820 .922 .767
A1 - A3 .771 .925 .686
A2 - A3 .851 .935 .805
Table 2: Pairwise Krippendorff’s αacross the three
tasks. A1, A2, A3 denote the three annotators.
RD AssessmentOnce a counterfactual RD has
been identified, the analysis proceeds to an assess-
ment subtask tailored to the specific design identi-
fied in Question 5. Each of the six RD families has
its own set of questions, ranging from 9 (Selection-
on-Observables) to 16 (Instrumental Variables and
Synthetic Control Methods), for a total of 76 ques-
tions across all designs. While the specific ques-
tions differ, reflecting the distinct assumptions and
methodological concerns of each RD, the assess-
ment schemes share a common structure organized
around four broad dimensions: (a) the study con-
text and specific features of the research design, (b)
the features of the data available in the study, (c)
the estimation and inference procedures employed,
and (d) falsification tests and and sensitivity anal-
yses. For instance, the TbD schema asks about
the features of the running variable, bandwidth se-
lection and specific robustness checks, while the
DiD schema focuses on the treatment assignment
structure (blocked vs. staggered), parallel trends ev-
idence, and the choice of estimator. All questions
are closed-ended, with predefined answer options,
and most allow multiple selections.
4.2 Inter-Annotator Agreement
In order to estimate the complexity of the task,
we asked three experts to independently annotate
24 scientific papers and computed inter-annotator
agreement (I-AA) scores. The I-AA set consistsof 12 papers using counterfactual methods (2 pa-
pers for each method) and 12 papers using non-
counterfactual methods. Results are reported in
Table 1). The overall agreement on the Full Task
(alpha = .81) is well above the .67 threshold com-
monly considered acceptable for reliable annota-
tion (Krippendorff, 2004). Agreement is notably
higher for RD Identification (.92) than for RD
Assessment (.75), confirming that detecting a re-
search design is substantially easier than evaluating
how credibly it is applied, even for trained experts.
The pairwise scores (Table 2) support this obser-
vation: RD Identification is uniformly high and
stable across all annotator pairs (.92–.94), whereas
RD Assessment is both lower and more variable,
consistent with its more subjective nature. At the
per-RD level, agreement varies considerably, rang-
ing from ED and SCM at the top (.83 and .82) down
to SoO (.64), IV (.68) and DiD (.69).
5 Methodology
We frame the ARDTrA task as a RAG-based
(Lewis et al., 2020; Guu et al., 2020) multi-turn
conversation with closed-ended questions. In this
setup, the system processes one question at a time,
in sequence: at each turn, the LLM receives (a)
a question q, (b) its predefined answer options
a1, a2, ...a n∈A and (c) a list ofkpassages
p1, p2, ...p n∈P dynamically retrieved from the
paper under examination. The questionqand the
answer optionsAare concatenated and used jointly
as the retrieval query, since the answer options
contain method-specific terminology that improves
passage selection. The conversation is maintained
across turns within a fixed context window of 20K
tokens, so that the model can leverage its previous
answers when responding to subsequent, more spe-
cific assessment questions. This design provides
two key advantages. First, it makes the model’s
answers traceable: because retrieval selects a small
set of passages for each question, every answer
can be verified against specific spans of the source
paper, a form of explainability that full-document
prompting does not enable. Second, and more im-
portantly, it closely replicates the human annotation
process: the model answers the exact same closed-
ended questions, with the same predefined answer
options, in the same order as the human annota-
tors, making human-machine comparison imme-
diate. We compare four retrieval strategies that
vary along the dimensions of chunking granularity

(fixed vs. variable) and retrieval method (sparse
vs. dense), using four LLMs of different sizes and
six state-of-the-art embedding models. The four
strategies are described below4.
Best Match 25 (BM25)BM25 (Robertson and
Zaragoza, 2009) is included due to its low compu-
tational cost and its established effectiveness as a
retrieval baseline. At indexing time, documents are
chunked at fixed lengths and indexed in a sparse,
keyword-based index. At retrieval time, the system
uses the Okapi BM25 scoring function.
Dense RetrievalAt indexing time, documents
are chunked at fixed lengths and each chunk is rep-
resented as a dense vector embedding. At retrieval
time, passages are ranked by cosine similarity be-
tween the query embedding and the chunk embed-
dings.
Adaptive Auto-merging RetrievalAdaptive re-
trieval is based on hierarchical indexing. At in-
dexing time, the document is chunked at differ-
ent lengths (we used 1024, 512, 256 and 128 to-
kens respectively). During parsing, chunks from
each layer are structured into a tree, by keeping
child/parent relations, as well as sibling relations.
All nodes, at all levels, are then stored, but only
leaf nodes are embedded, using dense embeddings.
At retrieval time, similarity search is performed
against the leaf-node index to retrieve an initial
candidate set. A bottom-up merging pass then con-
solidates these candidates: for each parent node,
the ratio of retrieved children to total children is
computed, and when this ratio exceeds a thresh-
oldt(we experimented with t∈ {0.3,0.5,0.7} ),
the children are replaced by the parent node. This
check recurses upward through the tree until no fur-
ther merges are triggered. Lower tvalues promote
merging into fewer, larger chunks, while higher val-
ues keep the output closer to leaf retrieval, resulting
in shorter and more specific chunks.
Propositional Topic-aware RetrievalIn this
strategy, the document is first split into paragraphs.
Each paragraph is then decomposed by an LLM
(we used Llama-3.1-8B-Instruct ) andrewrit-
teninto a set of propositions, each representing
an atomic statement contained in the paragraph
(Chen et al., 2024b). Propositions are the result of
4All strategies have been implemented using Llamaindex.
For Adaptive Auto-merging Retrieval and for Propositional
Topic-aware we used the AutoMergingRetriever and the Top-
icNodeParser classes, respectively.splitting compound sentences and resolving extra-
propositional pronoun co-reference so that they
are interpretable independently. Once paragraphs
are converted into propositions, a parser iterates
over propositions and builds “topic-aware” chunks
by choosing whether to merge a new proposition
into the current chunk or to create a new chunk.
This decision is made by the parser based on a
similarity threshold st, computed as the cosine
similarity between the current chunk and the new
proposition. Higher stvalues result in smaller,
topically more coherent chunks, whereas lower val-
ues produce larger, more inclusive and potentially
more heterogeneous chunks. We experiment with
st∈ {0.3,0.5,0.8} . Also, a sliding window of
sizewsis used by the parser to limit the context
used in the chunk definition process. A small value
of the wsparameter will make the parser more sen-
sitive to local topic switch, while a larger value
results in more stable and inclusive topic chunks.
We experiment with ws∈ {2,5} . It is worth not-
ing that, unlike all other strategies, in this case the
chunks produced are not original text but consist
of variable-length sequences of rewritten atomic
propositions.
Long-ContextAs a non-retrieval baseline, we
additionally evaluate a Long-Context (LC) setting
in which the full text of the paper, truncated to a
32K-token window for comparability across mod-
els, is provided directly in the model’s context win-
dow without any retrieval. The model answers the
same closed-ended questions, in the same multi-
turn conversational format, but instead of receiving
dynamically retrieved passages, it has access to the
entire document. This baseline tests whether re-
trieval is beneficial at all, or whether a sufficiently
large context window makes it unnecessary.
5.1 Experimental Setup
The described methodology is instantiated
on six embedding models of varying size
and architecture: mxbai-embed-large-v1
(335M), bge-large-en-v1.5 (335M; Chen
et al., 2024a), e5-large-v2 (335M; Wang
et al., 2022), SFR-Embedding-Mistral (7B;
Meng et al., 2024), Qwen3-Embedding-8B
(8B; Zhang et al., 2025), and a proprietary
model, Cohere-embed-english-v3.0 . In the
main phase, all experiments are run using
two LLMs, Llama-3.1-8B-Instruct and
Qwen2.5-32B-Instruct . The top k= 5 passages

are retrieved at each turn. For each combination of
strategy, hyperparameter values, embedding model
and LLM, a complete run is conducted on the full
dataset (140 documents). Considering the hyper-
parameter and embedding model combination,
we run a total of 82 rounds for each LLM (4 for
BM25, 24 for Dense, 18 for Adaptive and 36 for
Propositional), summing up to 164 experimental
rounds. Based on the evidence collected in the
main experimental phase, the best-performing
embedding model is selected and a further set of
experiments is then run with a single embedding
model, using Llama-3.3-70B-Instruct and
gpt-5.1 , in order to benchmark larger LLMs
while mitigating computational costs.
6 Evaluation and Results
Given the mixed nature of answer formats (binary,
single-answer, and multiple-answer questions) we
frame evaluation as a classification problem at the
level of individual answer options. Each correct
answer option selected by the model is considered
a true positive, each option selected by the model
but absent from the gold annotation counts as a
false positive, and each gold answer option missed
by the model counts as a false negative. This per-
option evaluation is strict: for a binary question
where the model answers “yes” and the gold answer
is “no”, both a false positive and a false negative
are counted. On this basis, we compute Precision,
Recall and F1. We report results for the Full Task,
as well as separately for the two subtasks: RD
Identification and RD Assessment. Results are
depicted in Figure 2 and additionally in Appendix
C. In what follows, we analyze the results along
three dimensions: the impact of retrieval strategy,
the role of text embeddings and the variation of
performance across Research Designs.
6.1 Retrieval Strategies
When comparing strategies at their best config-
uration, by selecting the top-performing embed-
ding and hyperparameter combination for each,
Adaptive achieves the highest F1 on the Full Task
for both Llama-8B (.563) and Qwen-32B (.660),
slightly ahead of BM25 (.555 and .645 respec-
tively). However, when we average over embed-
dings to remove the dependency on a favorable
embedding choice, the picture changes: BM25
matches or outperforms all dense strategies across
both LLMs and all subtasks, proving to be the mostreliable strategy overall (see Table 7, Appendix A).
LLM Full Task RD-I RD-A
Llama-8B .779 (p < .001) .620 (p < .01) .757 (p < .001)
Qwen-32B .815 (p < .001) .703 (p < .01) .810 (p < .001)
Llama-70B .727 (p < .01) .532 (p < .05) .655 (p < .01)
gpt-5.1 .292 (p = n.s.) .534 (p < .05) .299 (p = n.s.)
Table 3: Pearson correlation ( r) between average chunk
length and F1 score across the 17 hyperparameter con-
figurations. For Llama-8B andQwen-32B , F1 scores of
dense strategies are averaged over 6 embedding models.
Within each strategy, then, performance is max-
imized around certain hyperparameters, i.e. 512-
1024 for fixed-size strategies (BM25 and Dense),
t= 0.3 for Adaptive and st= 0.3 for Propo-
sitional. Sensitivity to hyperparameters is much
higher in variable-size strategies, in particular
higher values of tfor Adaptive and stfor Proposi-
tional (which in both cases yield shorter passages)
cause a significant drop in performance, and this
holds in both RD-I and RD-A. To test whether
passage length, rather than other strategy related
properties, drive these differences, we first com-
puted the average passage length (Figure 2, ruler)
produced by each hyperparameter value for all
the strategies across the whole dataset and then
correlated them with the performance. This re-
veals that the overall performance, at least for the
three smallest LLMs, has a high linear correla-
tion with average passage length: 0.779 Pearsonr
forLlama-8B , 0.815 for Qwen-32B and 0.727 for
Llama-70B respectively. Correlation values are
significantly higher for Assessment than for Iden-
tification (see Table 3). Different evidence holds
forgpt-5.1 , whose performance does not corre-
late significantly with passage length. Crucially,
this provides a very clear indication for our task:
between 52% and 66% of the task variance (R-
squared) is explained by chunk length, showing
that context is relevant to enable correct answer-
ing patterns in a RAG-baseline pipeline. Also, the
lack of correlation for gpt-5.1 suggests that very
strong LLMs can compensate for the lack of con-
text, possibly with a better use of conversational
history or via parametric knowledge. Since the
observed correlation between passage length and
performance could in principle be confounded with
other strategy-related properties, we conducted a
further analysis by comparing different strategies
at matched passage lengths. The underlying idea
is that, if a given strategy holds a systematic advan-

0.60.650.70.750.80.85
(a)BM25 Dense Adaptive PropositionalRD Identification
128 256 5121024128 256 5121024t=0.3 t=0.5 t=0.7st=0.3ws=2st=0.3ws=5st=0.5ws=2st=0.5ws=5st=0.8ws=2st=0.8ws=50.250.30.350.40.450.50.550.60.65
(b)
57 131 259 511 54 114 212 387 465 55 51 154 155 116 116 23 24RD Assessment
Llama-3.1-8B-Instruct Qwen2.5-32B-Instruct Llama-3.3-70B-Instruct gpt-5.1
Figure 2: F1-scores for (a)RD Identificationand (b)RD Assessmentacross retrieval strategies and LLM
configurations. Numbers on the ruler refer to the average passage length (in words) associated to the given
hyperparameter value. For example: Adaptive strategy with t= 0.3 produces passages of 465 words on average.
Dashed horizontal lines indicate the Long Context (non-retrieval) baseline, in the corrisponding colour.
tage over another, this should be reflected in con-
sistent performance differences when comparing
configurations that yield similar average passage
lengths. We compared 5 pairs of configurations (re-
ported in Table 4) with Llama-8B andQwen-32B ,
for which we have results across all 6 embedding
models, allowing us to average out the effect of
embedding choice. Across 30 comparisons (5 pairs
with 2 LLMs and 3 tasks), the average performance
difference between strategies is 1.0 F1-points (with
the worst case lying at 3.1 points) and no strategy
shows a consistent advantage over the others. Over-
all, this indicates that, while differences in strate-
gies might exist, they probably lie at other levels
(lexical, boundaries), and that passage length is the
strongest driver for performance across strategies.
6.2 Text Embeddings
Using the F1 scores from single experimental
rounds with Llama-8B andQwen-32B , we aggre-
gated the results of different embedding models
by subtask, retrieval strategy and across strategy
parameters. Based on simple win count (see Ta-
ble 9, Appendix B), bgeleads the ranking, beingConfig Alen vs.Config Blen
(a) Dense 128 54 Adaptivet=0.5 55
(b) BM25 256 131 Propositionalst=0.3,ws=2 154
(c) Dense 256 114 Propositionalst=0.5 116
(d) Dense 1024 387 Adaptivet=0.3 465
(e) BM25 1024 511 Adaptivet=0.3 465
Table 4: Values for the 5 configuration pairs used for
matched length comparison. Thelencolumns refer
to the average passage length generated by the given
configuration, measured in words (split on whitespace).
the best model in 41.5% of cases, followed by SFR
(20.7%) and mxbai (13.4%). This trend holds for
Dense and Adaptive strategies, while for Propo-
sitional SFRclearly dominates. As for subtasks,
bgeis noticeably the best in the RD-I task and
in the Full Task, while in RD-A it ties with SFR.
bgealso shows the top mean F1 across the three
subtasks and the second lowest standard deviation
across hyperparameters, after SFR. Overall, bgeof-
fers the best trade-off between performance and
stability among the models tested. Looking at the
broader picture, we observe that the gap between
the best and the worst embedding drops with larger

LLMs, passing from 2.5 to 2.1 F1 points in the
RD-I task and from 4.7 to 3.3 points in the more
difficult RD-A task. A similar but less pronounced
trend is followed by the standard deviation, which
decreases from 0.010 to 0.008 in RD-I and from
0.017 to 0.014 in RD-A (see Table 8). Both ob-
servations, combined, indicate that the choice of
embedding is less relevant with stronger models,
but also that it becomes more important with the
growing difficulty of the task.
6.3 Research Design Analysis
Based on their specific methodological architecture,
Research Designs differ in many respects: in their
degree of formalization, the type of technical lan-
guage used, the way argumentation is constructed
and the density of relevant information throughout
the paper. We therefore analyzed the performance
per-RD, by computing performance for each LLM
(across strategies, Figure 3) and a ranking for each
strategy (see Table 5). The results provide a nu-
anced picture. At the two opposite ends of the
spectrum, we observe that TbD are consistently the
easiest to identify and assess by models, followed
by DiD, while IV are quite consistently the most
difficult. SCM and ED, in turn, are more sensi-
tive to the specific LLM. SoO, meanwhile, shows
a more consistent ranking across all four models.
When results are grouped by strategy, BM25 ap-
pears to show the highest variability across RDs,
possibly because its lexical matching is more sensi-
tive to how explicitly each design’s methodology
is described.
RDL-8B Q-32B L-70B gpt-5.1MR Std
TbD 1 1 3 1 1.5 0.9
DiD 2 2 1 2 1.8 0.4
SCM 3 4 2 6 3.8 1.5
SoO 4 5 4 4 4.2 0.4
ED 6 3 5 3 4.2 1.3
IV 5 6 6 5 5.5 0.5
Table 5: Difficulty ranking of the six RDs (rank 1 =
easiest) for each LLM. For each (LLM, RD) pair, F1 is
averaged within each retrieval strategy and then across
the four strategies (equal weight per strategy); RDs are
then ranked within each LLM.Mean RankandStdare
computed across the four LLM ranks.
We further correlated each of the 4 sets of F1-
averages per Research Design with the human
agreement, in order to test the hypothesis that RDs
that are difficult for humans to annotate would
also be harder for models to predict correctly. As0.40.50.60.70.8F1-score(a) Long Context
0.40.50.60.70.8Average F-1 score(b) BM25
0.40.50.60.70.8Average F-1 score(c) Dense
0.40.50.60.70.8Average F-1 score(d) Adaptive
DiD ED IV SCM SoO TbD0.40.50.60.70.8
Research DesignAverage F-1 score(e) Propositional
Llama-8B Qwen-32B Llama-70B gpt-5.1
Figure 3: Performance of four LLMs, by RD. Except
for the Long Context baseline, results for Llama-8B
andQwen-32B refer to the average performance of 6
embedding models for the given RD. Llama-70B and
gpt-5.1used onlybge-large-en-v1.5.
showed in Table 6,we observed no statistically
significant correlation. Indeed, several cases ex-
hibited an inverse pattern, with RDs characterized
by lower human agreement being comparatively
easier for models to predict (for example DiD), and
vice versa (e.g. ED). The only RD where human
and machine difficulty align is IV , which proves

challenging for both. Although surprising, this sug-
gests that the RDs on which humans disagree the
most are not the same on which automated system
struggle the most. In other words, human difficulty,
i.e. the subjectivity inherent to expert judgment,
and machine difficulty, that of correctly retriev-
ing the right passages from the text, appear to be
two largely independent dimensions in expert tasks
like ARDTrA. This finding is in line with recent
debate in IR and NLP (Amiraz et al., 2025; Trap-
polini et al., 2026), who show that at thepassage
level, human relevance judgments and actual ma-
chine utility in RAG systems are fundamentally
misaligned. Our results suggest that a similar mis-
alignment exists at thetasklevel. Understanding
how this misalignment operates remains an open
challenge, with direct implications for the design
and evaluation of RAG systems in expert domains.
Llama-8B Qwen-32B Llama-70B gpt-5.1
BM25 .27/.49 .06/-.09 .14/.09 -.04/-.09
Dense .21/.09 .49/.43 .21/.26 -.20/-.31
Adaptive .20/.14 .44/.71 .04/.43 .00/-.09
TopicNode .05/.09 .40/.20 .07/.09 .12/-.09
Table 6: Pearson r/ Spearman ρcorrelations between
Krippendorff’s αand model performance across the six
Research Designs. None of the correlations is statisti-
cally significant (p > .05).
6.4 Long Context vs. RAG
Long-Context document prompting proves a strong
baseline: across both subtasks and all four LLMs,
most retrieval configurations fail to outperform it
(dashed lines in Figure 2). On RD-I, the baseline is
only locally surpassed, and mainly by the smaller
models, suggesting that when the task is compar-
atively easy, retrieval offers little advantage over
simply reading the whole document. On RD-A,
the picture is similar but the exceptions are more
systematic: only BM25 and the Adaptive strategy
manage to beat the LC baseline at specific configu-
rations. These results indicate that retrieval is not,
in itself, necessary to achieve competitive perfor-
mance on this task; rather, its value lies in focusing
the model on a small set of passages that can be
stored and inspected, providing the interpretability
that a black-box LC pipeline cannot.
7 Conclusions and Future Work
In this work, we introduced the task of Auto-
matic Research Design Tracking and Assessment(ARDTrA), meant to help social scientists and pol-
icy advisors quickly and reliably estimate the ro-
bustness of scientific papers based on an exten-
sive analysis of their application of Research De-
signs. This contribution bridges a gap in exist-
ing NLP-CSS research, in particular by provid-
ing an expert-designed analytical framework and
an annotated dataset of 140 papers from the so-
cial sciences. We evaluated four different retrieval
strategies, based on fixed and variable chunking,
with four LLMs and six embedding models, as
well as a Long Context setting. Based on ex-
tensive experimental evidence, our findings show
that: (a) low-cost, sparse BM25-based retrieval
matches and often outperforms dense embedding-
based strategies; (b) passage length, rather than
retrieval strategy, explains between 52% and 66%
of the performance variance across configurations,
and at matched passage lengths different strate-
gies perform comparably, though this effect di-
minishes with stronger LLMs such as gpt-5.1 ,
whose performance is largely independent of pas-
sage length; (c) for embedding-based strategies,
the impact of embedding model choice is smaller
than that of retrieval strategy or hyperparameter
configuration, with smaller encoder-based mod-
els like bge-large-en-v1.5 proving the most ro-
bust across settings; moreover, (d) Long-Context
document prompting proves to be a strong base-
line when compared to RAG. Finally, (e) per-RD
analysis reveals that human and machine difficulty
are driven by different factors: the Research De-
signs on which experts disagree most are not the
same ones on which the automated system strug-
gles, suggesting that subjective judgment and in-
formation retrievability represent two independent
dimensions of task difficulty. Findings (b) and (e),
in particular, open up promising research directions
that we aim to address in future work. First, while
passage length accounts for a substantial share of
the variance, a significant portion remains unex-
plained; we plan to explore whether lexical fac-
tors (how methodological choices are expressed
linguistically) and structural ones (where in the
document the relevant information appears) con-
tribute to the residual variance. Second, our finding
that human and machine difficulty diverge raises
the question of whether experts and retrieval-based
systems attend to fundamentally different textual
cues when performing methodological assessment,
and whether this divergence generalizes to other
expert analytical tasks.

8 Limitations
The main limitation of this work is the size of the
dataset. This is due to the depth of the annotation
schema and the high effort required from anno-
tators, who must closely read papers of up to 50
pages and answer subtle methodological questions.
With the current size of the dataset, therefore, per-
RD estimates should be interpreted with caution.
We tried to counterbalance the limited scale by
carefully selecting papers to be representative of
each RD and to cover a range of application do-
mains. Additionally, the dataset focuses on papers
in English, which limits the generalizability of our
findings to other languages.
9 Use of AI Assistants
We used AI Assistants to support the revision of the
manuscript and the creation of complex plots/tables.
All scientific content, experimental design, data
analysis, and interpretation of results are the sole
responsibility of the authors.
10 Ethics Statement
This work analyzes published scientific papers and
does not involve human subjects, personal data, or
sensitive content. All annotators are co-authors
and participated as part of their regular institutional
duties. The dataset release (see Supplemental Mate-
rials) contains only bibliographic metadata as most
of the papers are copyrighted. To the best of our
knowledge, the work described in this paper does
not raise any ethical concerns.
References
Alberto Abadie. 2021. Using synthetic controls: Feasi-
bility, data requirements, and methodological aspects.
Journal of economic literature, 59(2):391–425.
Alberto Abadie, Alexis Diamond, and Jens Hainmueller.
2010. Synthetic control methods for comparative
case studies: Estimating the effect of california’s
tobacco control program.Journal of the American
statistical Association, 105(490):493–505.
Alberto Abadie and Javier Gardeazabal. 2003. The eco-
nomic costs of conflict: A case study of the basque
country.American economic review, 93(1):113–132.
Chen Amiraz, Florin Cuconasu, Simone Filice, and
Zohar Karnin. 2025. The distracting effect: Under-
standing irrelevant passages in RAG. InProceedings
of the 63rd Annual Meeting of the Association for
Computational Linguistics (Volume 1: Long Papers),pages 18228–18258, Vienna, Austria. Association
for Computational Linguistics.
Joshua D Angrist, Guido W Imbens, and Donald B
Rubin. 1996. Identification of causal effects using
instrumental variables.Journal of the American sta-
tistical Association, 91(434):444–455.
Joshua D Angrist and Jörn-Steffen Pischke. 2010. The
credibility revolution in empirical economics: How
better research design is taking the con out of econo-
metrics.Journal of economic perspectives, 24(2):3–
30.
Akari Asai, Jacqueline He, Rulin Shao, Weijia Shi,
Amanpreet Singh, Joseph Chee Chang, Kyle Lo,
Luca Soldaini, Sergey Feldman, Mike D’Arcy, and 1
others. 2026. Synthesizing scientific literature with
retrieval-augmented language models.Nature, pages
1–7.
Orley Ashenfelter and David Card. 1985. Using the lon-
gitudinal structure of earnings to estimate the effect
of training programs.The Review of Economics and
Statistics, 67:648–660.
Susan Athey and Guido Wilhelmus Imbens. 2017. The
econometrics of randomized experiments. InHand-
book of economic field experiments, volume 1, pages
73–140. Elsevier.
Abel Brodeur, Nikolai Cook, and Carina Neisser. 2024.
P-hacking, data type and data-sharing policy.The
Economic Journal, 134(659):985–1018.
David Card and Alan B Krueger. 1993. Minimum wages
and employment: A case study of the fast food indus-
try in new jersey and pennsylvania.
Matias D Cattaneo and Rocio Titiunik. 2022. Regres-
sion discontinuity designs.Annual Review of Eco-
nomics, 14(1):821–851.
Jianlyu Chen, Shitao Xiao, Peitian Zhang, Kun
Luo, Defu Lian, and Zheng Liu. 2024a. M3-
embedding: Multi-linguality, multi-functionality,
multi-granularity text embeddings through self-
knowledge distillation. InFindings of the Asso-
ciation for Computational Linguistics: ACL 2024,
pages 2318–2335, Bangkok, Thailand. Association
for Computational Linguistics.
Tong Chen, Hongwei Wang, Sihao Chen, Wenhao Yu,
Kaixin Ma, Xinran Zhao, Hongming Zhang, and
Dong Yu. 2024b. Dense X retrieval: What retrieval
granularity should we use? InProceedings of the
2024 Conference on Empirical Methods in Natural
Language Processing, pages 15159–15177, Miami,
Florida, USA. Association for Computational Lin-
guistics.
Pierre-Philippe Combes and Laurent Linnemer. 2010.
Inferring missing citations: A quantitative multi-
criteria ranking of all journals in economics.

Janet Currie, Henrik Kleven, and Esmée Zwiers. 2020.
Technology and big data are changing economics:
Mining text to track methods. InAEA Papers and
Proceedings, volume 110, pages 42–48. American
Economic Association 2014 Broadway, Suite 305,
Nashville, TN 37203.
Shuo Feng, Ishani Ganguli, Youjin Lee, John Poe, An-
drew Ryan, and Alyssa Bilinski. 2025. Difference-in-
differences for health policy and practice: A review
of modern methods.Statistics in medicine, 44(23-
24):e70247.
Sir Ronald Aylmer Fisher and Ronald A Fisher. 1966.
The design of experiments, volume 21. Springer.
Prashant Garg and Thiemo Fetzer. 2025. Causal claims
in economics.arXiv preprint arXiv:2501.06873.
Paul Goldsmith-Pinkham. 2024. Tracking the cred-
ibility revolution across fields.arXiv preprint
arXiv:2405.20604.
Kelvin Guu, Kenton Lee, Zora Tung, Panupong Pasu-
pat, and Mingwei Chang. 2020. Retrieval augmented
language model pre-training. InInternational confer-
ence on machine learning, pages 3929–3938. PMLR.
J Heckman. 1979. Sample selection bias as a specifica-
tion error.Econometrica.
Miguel A. Hernán and James M. Robins. 2020.Causal
Inference: What If. Chapman & Hall/CRC, Boca
Raton.
Paul W. Holland. 1986. Statistics and causal infer-
ence.Journal of the American Statistical Association,
81(396):945–960.
Rory Hooper, Nihit Goyal, Kornelis Blok, and Lisa
Scholten. 2024. A semi-automated approach to
policy-relevant evidence synthesis: combining natu-
ral language processing, causal mapping, and graph
analytics for public policy.Policy Sciences, pages
1–26.
Kosuke Imai and Kentaro Nakamura. 2026. Causal in-
ference with generative artificial intelligence: Appli-
cation to texts as treatments.Journal of the American
Statistical Association, (just-accepted):1–27.
Guido W Imbens. 2015. Matching methods in prac-
tice: Three examples.Journal of Human Resources,
50(2):373–419.
Guido W Imbens. 2024. Causal inference in the so-
cial sciences.Annual Review of Statistics and Its
Application, 11.
Guido W. Imbens and Donald B. Rubin. 2015.Causal
Inference for Statistics, Social, and Biomedical Sci-
ences: An Introduction. Cambridge University Press,
New York.
Klaus Krippendorff. 2004. Reliability in content analy-
sis: Some common misconceptions and recommen-
dations.Human communication research, 30(3):411–
433.Nicholas Larsen, Jonathan Stallrich, Srijan Sengupta,
Alex Deng, Ron Kohavi, and Nathaniel T Stevens.
2024. Statistical challenges in online controlled ex-
periments: A review of a/b testing methodology.The
American Statistician, 78(2):135–149.
Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio
Petroni, Vladimir Karpukhin, Naman Goyal, Hein-
rich Küttler, Mike Lewis, Wen-tau Yih, Tim Rock-
täschel, and 1 others. 2020. Retrieval-augmented
generation for knowledge-intensive nlp tasks.Ad-
vances in Neural Information Processing Systems,
33:9459–9474.
Rui Meng, Ye Liu, Shafiq Rayhan Joty, Caiming
Xiong, Yingbo Zhou, and Semih Yavuz. 2024. Sfr-
embedding-mistral:enhance text retrieval with trans-
fer learning. Salesforce AI Research Blog.
Magne Mogstad and Alexander Torgovitsky. 2018. Iden-
tification and extrapolation of causal effects with in-
strumental variables.Annual Review of Economics,
10(1):577–613.
Stephen Robertson and Hugo Zaragoza. 2009.The prob-
abilistic relevance framework: BM25 and beyond,
volume 4. Now Publishers Inc.
Jonathan Roth, Pedro HC Sant’Anna, Alyssa Bilinski,
and John Poe. 2023. What’s trending in difference-in-
differences? a synthesis of the recent econometrics
literature.Journal of Econometrics, 235(2):2218–
2244.
Donald B Rubin. 1974. Estimating causal effects of
treatments in randomized and nonrandomized studies.
Journal of educational Psychology, 66(5):688.
Elizabeth A Stuart. 2010. Matching methods for causal
inference: A review and a look forward.Statistical
science: a review journal of the Institute of Mathe-
matical Statistics, 25(1):1.
Donald L Thistlethwaite and Donald T Campbell. 1960.
Regression-discontinuity analysis: An alternative to
the ex post facto experiment.Journal of Educational
psychology, 51(6):309.
Giovanni Trappolini, Florin Cuconasu, Simone Filice,
Yoelle Maarek, and Fabrizio Silvestri. 2026. Re-
defining retrieval evaluation in the era of LLMs. In
Proceedings of the 19th Conference of the European
Chapter of the Association for Computational Lin-
guistics (Volume 1: Long Papers), pages 8359–8375,
Rabat, Morocco. Association for Computational Lin-
guistics.
Liang Wang, Nan Yang, Xiaolong Huang, Binxing
Jiao, Linjun Yang, Daxin Jiang, Rangan Majumder,
and Furu Wei. 2022. Text embeddings by weakly-
supervised contrastive pre-training.arXiv preprint
arXiv:2212.03533.
Yanzhao Zhang, Mingxin Li, Dingkun Long, Xin Zhang,
Huan Lin, Baosong Yang, Pengjun Xie, An Yang,
Dayiheng Liu, Junyang Lin, Fei Huang, and Jingren

Zhou. 2025. Qwen3 embedding: Advancing text
embedding and reranking through foundation models.
arXiv preprint arXiv:2506.05176.

A Retrieval Strategies
In this Section we report the results of the analysis
conducted on Retrieval Strategies (Section 6.1).
LLM Task BM25 DENSE ADAPT PROP
Best overall score
Ll-8BFull Task .555 .541 (b) .563(Q) .543 (S)
RD-I .712 .727 (b) .729 (Q) .742(e)
RD-A .443 .442 (b) .447(Q) .409 (S)
Qw-32BFull Task .645 .641 (b) .660(b) .629 (m)
RD-I .773 .764 (b) .767 (m) .774(S)
RD-A .557 .562 (Q) .590(b) .536 (m)
Best score embedding average
Ll-8BFull Task .555.526 .539 .513
RD-I .712 .715.715 .711
RD-A .443.393 .421 .375
Qw-32BFull Task .645.627 .630 .607
RD-I .773.753 .715 .759
RD-A .557.542 .544 .504
Single embedding (bge-large-en-v1.5)
Ll-70BFull Task .640.626 .634 .589
RD-I .743 .749 .754.742
RD-A .573.542 .553 .482
gpt-5.1Full Task .694.658 .686 .664
RD-I .811 .814.803 .813
RD-A .618 .556 .620.581
Table 7: Strategy comparison (F1-scores). Highlight
indicates the best strategy per row.Best embedding: best
hyperparameter configuration with the best-performing
embedding model.Embedding avg.: best hyperparame-
ter configuration, averaged over all 6 embedding models.
Llama-70B and gpt-5.1 usebge-large-en-v1.5only.
B Text Embeddings
This section refers to the analysis on the impact of
text embeddings conducted in Section 6.2 of the
paper. Values in Table 8 have been computed by
first averaging all F1-scores by embedding model
across strategies and then by calculating the max-
min range and standard deviation. Values in Table
9 have been computed by counting the times each
embedding model provides the best F1 score for a
given hyperparameter. As for Table 10, Mean F1
values have been computed by averaging the given
model performance across all hyperparameters and
two LLMs. Standard deviation and Range, instead
have been computed by first calculating Std value
for each LLM in RD-I and RD-A tasks for the
given embedding model, thus obtaining 4 values
for each embedding model. Final Std in the Table
is the average of these values, while Range is their
max-min difference.LLM Task Range Std
Llama-8BFull Task .028 .011
RD-I .025 .010
RD-A .047 .017
Qwen-32BFull Task .024 .010
RD-I .021 .008
RD-A .033 .014
Table 8: Embedding impact: range and standard devia-
tion of F1 scores across the six embedding models, by
task and LLM.
bge SFR mxbai Qwen3 e5 Cohere
Overall (78)3417 11 10 8 2
Dense (24)142 3 6 1 0
Adaptive (18)121 4 3 0 0
Propositional (36) 8144 1 7 2
Full Task (26)135 4 3 1 1
RD-I (26)134 2 1 7 0
RD-A (26) 885 6 0 1
Table 9: Number of times each embedding model
achieves the highest F1 score, out of 78 combinations
(13 hyperparameter configurations ×2 LLMs ×3 tasks).
This information is provided in Tables 11, 12 and 13 in
Appendix C. Since there are a few ties, some rows don’t
sum up to the same number.
Size Mean F1 HP Sensitivity
Model Full RD-I RD-A Std Range
bge 335M.560 .729 .442.029 .110
SFR 7B .551 .719 .437.026.092
mxbai 335M .550 .720 .433 .031 .117
Qwen3 8B .541 .707 .434 .029 .097
e5 335M .536 .724 .404 .029.089
Cohere prop. .536 .709 .416 .028.089
Table 10: Embedding model comparison: mean F1 and
sensitivity to hyperparameter configuration. Mean is
computed across all configurations (13 hyperparameters
×2 LLMs). Std. and range measure variability across
the 13 hyperparameter configurations, averaged over 2
LLMs×2 tasks.
C Research Design Analysis
This Section reports the raw experimental results
before any aggregation (F1-scores). Each score in
Tables 11, 12 and 13 refers to a complete run of the
system on the whole dataset. Table 11 refers to the
Full Task, Table 12 to RD Identification and Table
13 to RD Assessment.

LC BM25 Dense Adaptive Topicnode
chunk size chunk size t st=0.3 st=0.5 st=0.8
LLM128 256 512 1024Emb.128 256 512 1024 0.3 0.5 0.7 ws=2 ws=5 ws=2 ws=5 ws=2 ws=5
Llama-8B* .545 .483 .524.555.533 mxbai .503 .495 .516 .525 .538 .489 .501 .517 .526 .505 .511 .455 .440
bge .494 .541 .540 .539 .542 .504 .494 .511 .511 .537 .532 .497 .468
e5 .444 .494 .503 .512 .535 .448 .452 .529 .466 .495 .538 .459 .470
SFR .512 .520 .534 .532 .543 .496 .484 .524 .519 .543 .542 .441 .471
Qwen3 .513 .511 .530 .539 .563.486 .481 .513 .516 .487 .458 .452 .457
Cohere .496 .508 .533 .511 .515 .471 .479 .469 .509 .488 .496 .463 .475Qwen-32B.674 .598 .626 .638 .645 mxbai .572 .622 .626 .633 .653 .582 .589 .612 .603 .629 .597 .553 .516
bge .593 .627 .622 .641 .660 .592 .595 .613 .603 .602 .601 .567 .522
e5 .565 .590 .625 .619 .610 .555 .550 .598 .594 .590 .596 .552 .550
SFR .561 .590 .590 .617 .604 .586 .560 .617 .606 .625 .595 .552 .573
Qwen3 .592 .565 .614 .635 .634 .560 .563 .602 .589 .556 .566 .535 .540
Cohere .561 .617 .622 .618 .619 .565 .552 .600 .570 .580 .562 .535 .52770B.636 .529 .611 .621 .640bge .574 .626 .610 .593 .634 .571 .576 .575 .584 .589 .587 .530 .5135.1 .691 .643 .675 .694.672 bge .618 .658 .635 .642 .670 .683 .686 .654 .661 .664 .655 .620 .634
Avg. p. len – 57 131 269 511 54 114 212 387 465 55 51 154 155 116 116 23 24
Table 11:Full TaskF1 scores across retrieval strategies, hyperparameters, embedding models and LLMs. High-
light marks the model scoring the best result for the given hyperparameter , in boldface theconfigurations that
outperform the Long Context (LC) baseline . Underlined, the best result for the current LLM. In green, the best
overall result for the task. The last row reports the average passage length generated by the hyperparameter (in
number of words).
LC BM25 Dense Adaptive Topicnode
chunk size chunk size t st=0.3 st=0.5 st=0.8
LLM128 256 512 1024Emb.128 256 512 1024 0.3 0.5 0.7 ws=2 ws=5 ws=2 ws=5 ws=2 ws=5Llama-8B* .713 .705 .712 .71 .707 mxbai .701 .700 .704.716.706 .682 .676 .734 .724 .719 .719.646 .622
bge .703 .714 .727 .720.707 .692 .677.724 .726 .731 .725 .698 .667
e5 .670 .696 .713.716.709 .670 .672 .742.669 .713 .738.685 .694
SFR .694 .696 .699.716 .724.679 .664.722.672 .737 .731.651 .676
Qwen3 .691 .694 .707 .711 .729.677 .662 .690.715.668 .652 .646 .650
Cohere .677 .697 .705 .708 .713 .671 .663 .647 .696 .696 .679 .668 .663Qwen-32B.771 .758 .763 .763 .773 mxbai .736 .767 .752 .757 .767 .730 .732 .763 .747 .763 .742 .718 .701
bge .750 .757 .759 .764 .761 .736 .743 .768 .754 .756 .758 .729 .714
e5 .729 .761 .762 .753 .754 .725 .725 .758 .758 .759 .762 .740 .740
SFR .696 .722 .740 .747 .762 .736 .720 .760 .748 .774 .764 .731 .726
Qwen3 .724 .745 .738 .746 .751 .724 .716 .750 .748 .712 .715 .705 .708
Cohere .725 .753 .757 .752 .761 .712 .715 .754 .746 .734 .718 .710 .71870B.752 .712 .731 .743 .736 bge .719 .749 .743 .742 .754 .715 .716 .752.743 .741 .738 .740 .7205.1 .818 .796 .802 .810 .806 bge .776 .814 .792 .813 .803 .792 .788 .814 .806 .802 .805 .788 .776
Table 12:RD IdentificationF1 scores across retrieval strategies, hyperparameters, embedding models and LLMs.
BM25 does not use embeddings.

LC BM25 Dense Adaptive Topicnode
chunk size chunk size t st=0.3 st=0.5 st=0.8
LLM128 256 512 1024Emb.128 256 512 1024 0.3 0.5 0.7 ws=2 ws=5 ws=2 ws=5 ws=2 ws=5Llama-8B* .414 .327 .382.443.400 mxbai .367 .353 .379 .385.418.362 .384 .361 .375 .359 .371 .327 .319
bge .347 .442.406 .404.424 .380 .370 .353 .353 .393 .394 .356 .326
e5 .284 .346 .344 .371 .410 .298 .298 .376 .315 .342 .389 .299 .312
SFR .397 .399 .413 .394 .413 .373 .359 .381 .409 .406 .406 .307 .337
Qwen3 .397 .389 .408 .419 .447.369 .365 .393 .379 .366 .333 .328 .336
Cohere .371 .374 .410 .363 .412 .339 .353 .346 .376 .346 .359 .319 .340Qwen-32B.605 .487 .535 .554 .557 mxbai .461 .526 .542 .549 .576 .487 .496 .508 .504 .536 .498 .431 .383
bge .488 .542 .529 .556 .590 .496 .496 .510 .498 .497 .494 .450 .385
e5 .453 .475 .529 .526 .512 .439 .427 .489 .480 .476 .480 .411 .415
SFR .470 .502 .487 .528 .501 .484 .454 .520 .510 .521 .479 .432 .469
Qwen3 .506 .498 .534 .562 .551 .463 .467 .505 .485 .453 .466 .424 .434
Cohere .450 .524 .528 .529 .531 .459 .441 .490 .444 .472 .447 .402 .38470B.555 .402 .528 .536 .573bge .473 .542 .518 .491 .553 .474 .483 .517 .484 .372 .406 .393 .3445.1 .608 .545 .594.618.575 bge .517 .556 .531 .527 .578.613 .62.601 .56 .481 .506 .441 .444
Table 13:RD AssessmentF1 scores across retrieval strategies, hyperparameters, embedding models and LLMs.

D ARDTrA Analytical Framework
In this Appendix we present the ARDTrA Ana-
lytical Framework, described in Section 4.1. The
framework consists of two main tasks: Research
Design Identification (RD-I, Section D.1) and Re-
search Design Assessment (RD-A, Section D.2).
D.1 Research Design Identification (6
questions)
1.What type of paper is this?
(a)Empirical - analyzes real-world data to
address research questions
(b)Theoretical - develops or discusses con-
ceptual, graphical or mathematical theo-
retical models
(c)Methodological - develops, discusses or
evaluates statistical or research method
(d)Review paper - summarizes or synthe-
sizes other studies (e.g., systematic re-
views, meta-analyses)
(e)Opinion paper, comment, reply - offers
commentary or responds to another pa-
per
(f) Other
2.Does the paper use quantitative or qualitative
research designs and data?
(a)Quantitative research design and data
- uses primarily numeric or categorical
data (e.g., from surveys, administrative
data, experiments) and statistical models
(b)Qualitative research design and data -
uses primarily non-numerical data de-
rived from interviews, participant obser-
vation, focus groups, narratives or other
qualitative and participatory data collec-
tions methods
(c) Other
3.Which type of analyses best describes the pa-
per?
(a)Causal Analysis - investigates causal re-
lationships between two or more vari-
ables, e.g., investigates whether a vari-
able or intervention causes a change in
another
(b)Descriptive Analysis - describes the state,
evolution or relationships among vari-
ables without making causal claims(c)Predictive or Simulation Analysis - pre-
dicts or simulates the value or evolution
of one or multiple variables
(d) Other
4.Does the study use any of the following
causal research design/methods: Experi-
mental designs (e.g., field experiments, sur-
vey experiments, lab-experiments), Selection-
on-observables (e.g., regression adjustment,
matching, weighting, doubly robust), Instru-
mental variables, Threshold-based designs
(e.g., regression discontinuity, regression kink,
bunching), Difference-in-differences and re-
lated designs (e.g., event studies, changes-in-
changes, triple differences), Synthetic control
method and related designs (e.g., augmented
synthetic control, synthetic difference in dif-
ferences)?
(a) Yes
(b) No
5.Which of the following is the main or preferred
research design, associated with the credibil-
ity revolution, potential outcome framework
or design based approach, used in the paper?
(a)Experimental designs (e.g., field ex-
periments, survey experiments and lab-
experiments)
(b)Selection-on-observables (e.g., regres-
sion adjustment, matching, weighting,
doubly robust)
(c) Instrumental variables
(d)Threshold-based designs (e.g., regres-
sion discontinuity, regression kink or
bunching)
(e)Difference-in-differences and related de-
signs (e.g., event studies, changes-in-
changes, triple differences)
(f)Synthetic control method and related de-
signs (e.g., augmented synthetic control,
synthetic difference in differences)
(g) Other
6.Does the paper use additional research de-
signs beyond the selected one, fitting under
the design-based causal inference approaches
associated with the “credibility revolution”?
(a)Experimental designs (field experiments,
survey experiments and lab-experiments)

(b) Selection-on-observables
(c) Instrumental variables
(d)Threshold-based designs (regression dis-
continuity, regression kink or bunching)
(e)Difference-in-differences and related de-
signs (i.e., event studies, changes-in-
changes, triple differences)
(f)Synthetic control method and related de-
signs (augmented synthetic control, syn-
thetic difference in differences)
D.2 Research Design Assessment
D.2.1 Experimental Designs (ED)
1.What is the randomization strategy used in the
study?
(a)Complete (simple) randomization: units
are independently assigned to treat-
ment/control with fixed probability
(b)Matched-pair or matched-set randomiza-
tion: treatment assigned within small,
covariate-matched group
(c)Stratified (blocked) randomization: treat-
ment assigned within pre-defined blocks
based on key covariates
(d)Covariate-adaptive randomization: as-
signment probabilities adjusted dynami-
cally to improve balance
(e)Re-randomization: randomization re-
peated until balance criteria are met
(f) Not reported or described
(g) Other
2.What is the unit of assignment to treatment
and control conditions?
(a)Individual level assignment: each unit is
independently assigned (e.g., individuals,
firms)
(b)Cluster level assignment: units are as-
signed in groups (e.g., schools, firms, vil-
lages)
(c)Multiple levels: assignment occurs at
more than one level, common in designs
aimed to capture spillover effects (e.g.,
schools then classes within schools)
(d) Not reported or unclear
(e) Other
3.Was the experimental protocol and analysis:(a)Pre-registered: a pre-analysis plan or
trial registration was filed prior to data
collection / analyses
(b)Approved by an ethical board (e.g., IRB)
(c) Not reported or unclear
4.Does the study include any balance / random-
ization checks?
(a)Pre-determined covariates (e.g., baseline
socio-demographic characteristics)
(b)Pre-determined outcomes or related prox-
ies (i.e., baseline outcome variables)
(c) Post-treatment variables not expected to
be affected by the treatment (i.e., placebo
outcomes)
(d) No balance test reported
(e) Other
5.Does the study provide evidence of good co-
variate balance?
(a)Yes - balance is high with only small
differences between groups
(b)Partially - balance is mostly good, but
some relevant differences remain
(c)No - balance is poor, with substantial
differences between groups
(d) Not discussed or not assessed
6.What is the level of general attrition, in per-
centage terms? Note: if there are multiple
treatment groups with different levels of at-
trition please select multiple values for each
row.
(a) Less than 5 percent
(b) 5-15 percent
(c) 16-25 percent
(d) 26-40 percent
(e) More than 40 percent
(f)Not applicable / relevant (i.e., one-shot
data collection and randomization, use
of administrative data)
(g) Not reported
7.What is the level of differential attrition, in
percentage points? Note: if there are multi-
ple treatment groups with different levels of
attrition please select multiple values for each
row.
(a) Less than 5 percentage points

(b) 5-15 percentage points
(c) 16-25 percentage points
(d) 26-40 percentage points
(e) More than 40 percentage points
(f)Not applicable / relevant (i.e., one-shot
data collection and randomization, use
of administrative data)
(g) Not reported
8.What estimation method(s) are used to esti-
mate treatment effects in the main analysis?
(a) Difference-in-means
(b)Regression models (OLS, probit, logit
etc)
(c)Matching, weighting or related ap-
proaches (e.g., propensity score match-
ing, inverse probability weighting, dou-
bly robust)
(d) ANOV A / ANCOV A
(e)Non-parametric methods (e.g., Wilcoxon
Rank-Sum Test, Mann-Whitney U,
Kolmogorov-Smirnov Test, Quantile
treatment effects, Kruskal-Wallis Test,
etc)
(f) Bayesian approaches
(g)Permutation-based test or related (Fisher
randomization tests)
(h)Multilevel (Hierarchical) Models (e.g.,
mixed-effects models, random inter-
cepts/slopes)
(i)Machine learning–based estimators (e.g.,
lasso, causal forest, double machine
learning)
(j) Other
9.Is clustering appropriately accounted for in
the inference procedure? Note: The recom-
mended practice is to account for clustering
at the level of unit assignment when the unit
of analysis is at a lower level than the cluster
(e.g., schools assigned but data is at student
level). However, if the unit of analysis is at the
level of clustering (e.g., schools assigned, and
data is aggregated at school level), clustering
is not necessary. Clustering is also necessary
when there are multiple observations per unit
used in the analysis (e.g., repeated measures).
Extra: clustering is also necessary when there
is a risk of interference or spillovers, but this
is rarely done, so if the paper reports such,please choose the “g) Other” option, ignor-
ing all other response options.
(a)Yes - clustering at level of treatment as-
signment
(b)Yes - clustering below treatment assign-
ment (e.g., at class level when schools
are assigned)
(c)Yes - clustering above treatment assign-
ment (e.g., at school level, when classes
are assigned)
(d)No - clustering not used and not nec-
essary (e.g., individual-level or clus-
tered assignment but the unit of analysis
matches the treatment assignment level)
(e)No - clustering not used, but it should
have been
(f) Unclear / not reported
(g) Other
10.Does the paper report any of the following
robustness or sensitivity checks? Select the “f)
Other” option only if you identify other highly
relevant falsification tests for the identification
strategy
(a)Sensitivity to the violation of the identi-
fication assumption (e.g., bounding, par-
tial identification; i.e., for cases with high
attrition rates or implementation issues)
(b) Sensitivity to alternative estimators
(c)Sensitivity to inclusion or exclusion of
covariates (e.g., robustness to controls or
pre-treatment variables)
(d)Sensitivity to using alternative infer-
ence approaches (e.g., different inference
method, level of clustering etc.)
(e)Robustness to sample exclusions (e.g.,
outlier removal, trimming)
(f) Other
11.Does the paper suffer from any of the potential
issues?
(a)Analyses include units not randomly as-
signed (e.g., units entering the program
after randomization)
(b)Analyses do not account for differing as-
signment probabilities (e.g., the proba-
bility of treatment varies considerably
within strata but the analysis does not
account for this)

(c)A randomized unit’s assigned condition
is not the same as the unit’s analyzed
condition (i.e., mismatch between as-
signed and analyzed treatment status,
e.g., controls non-complying with assign-
ment considered part of the treatment
group in the analysis)
(d)Exclusion of randomized units based on
post-assignment criteria (e.g., excluding
treatment units not taking up the treat-
ment)
(e) No, none of the above
D.2.2 Threshold-based designs (e.g.,
regression discontinuity, regression kink
or bunching)
1.What RDD framework does the paper use in
its main analysis?
(a)Continuity-based framework - assumes
potential outcomes are continuous at the
cutoff (the most frequent approach)
(b)Local randomization framework - treats
units near the cutoff as randomly as-
signed to treatment/control (often with
discrete running variables or small sam-
ple sizes, or as robustness check to the
continuity approach. It relies on stronger
assumptions.)
(c) Other
2.Which of the following RDD features best de-
scribes the paper?
(a) Single running variable
(b) Single cutoff
(c) Multiple cutoffs
(d)Multiple running variables (i.e., geo-
graphical RDDs, scholarship assigned
based on cutoffs in both language and
math tests)
(e) Other
3.Is the running variable approximately contin-
uous or discrete?
(a)Approximately continuous: there is a
large number of values on each side of
the cutoff
(b)Discrete with more than 10 unique values
on each side of the cutoff
(c)Discrete with 10 or fewer unique values
on each side of the cutoff(d) Other
4.Does the paper provide evidence for the conti-
nuity of the running variable density around
the cutoff?
(a)Histograms or similar graphical analysis
(b)Bernoulli or binomial tests around the
cutoff (e.g., Cattaneo et al., 2017)
(c)McCrary (2008) local polynomial den-
sity estimation
(d)Local polynomial density estimation
(e.g., Cattaneo, Jansson, and Ma, 2020)
(e)No specific details provided in the text
other than support for the hypothesis
(performed in the appendix)
(f) Other
(g) Not provided
5.Does the density continuity test support the
identification assumption (no manipulation of
the running variable)?
(a)Yes - no strong evidence of manipulation
at the cutoff
(b)No - evidence suggests manipulation or
discontinuity
(c) Not provided
6.What estimation approach is used to obtain
point estimates?
(a)Global polynomial methods, i.e., regres-
sion using observations over the entire
support of the running variable, often in-
cluding higher order polynomials
(b)Local polynomial methods, i.e., regres-
sion using only observations close to the
cutoff, generally with low order polyno-
mials
(c)Randomization inference within a win-
dow close to the cutoff (Fisherian ap-
proach)
(d)Ordinary least squares or similar large
sample approaches in the local random-
ization framework (based on Neyman or
superpopulation approaches)
(e) Other
7.How was the bandwidth (in continuity-based
RDD) or window (in local randomization
RDD) selected?

(a)MSE-optimal bandwidth - e.g., from
Calonico et al. (2014), based on mean
squared error minimization
(b)Covariate-based window selection (local
randomization framework)
(c) Manual selection
(d)Cross-validation (e.g., as in Ludwig and
Miller, 2007)
(e)IK algorithm for optimal bandwidth se-
lector proposed by Imbens and Kalya-
naraman (2012)
(f) Other
8.Is clustering appropriately accounted for in
the inference procedure? Note: The recom-
mended practice is to account for clustering
at the level of unit assignment when the unit
of analysis is at a lower level than the cluster
(e.g., schools assigned but data is at student
level). However, if the unit of analysis is at the
level of clustering (e.g., schools assigned, and
data is aggregated at school level), clustering
is not necessary. Clustering is also necessary
when there are multiple observations per unit
used in the analysis (e.g., repeated measures,
panel data). Extra: clustering is also neces-
sary when there is a risk of interference or
spillovers, but this is rarely done, so if the
paper reports such, please indicate in other,
ignoring all other response options.
(a)Yes - clustering at level of treatment as-
signment
(b)Yes - clustering below treatment assign-
ment (e.g., at class level when schools
are assigned)
(c)Yes - clustering above treatment assign-
ment (e.g., at school level, when classes
are assigned)
(d)No - clustering not used and not nec-
essary (e.g., individual-level or clus-
tered assignment but the unit of analysis
matches the treatment assignment level)
(e)No - clustering not used, but it should
have been
(f) Unclear / not reported
(g) Other
9.Does the paper provide any of the follow-
ing falsification (e.g., balance, placebo) tests?
Please select the ë) Otheröption only if it is ahighly relevant falsification test for the identi-
fication strategy
(a)Pre-determined covariate or outcome bal-
ance test - checking continuity or balance
of predetermined covariates at the cutoff
(b)Placebo outcomes test - applying the
same design / analysis replacing the out-
come variable with one which should not
be impacted by the treatment
(c)Placebo population test - applying the
same design / analysis in a sample where
the treatment is expected to not have an
effect
(d)Placebo treatment - applying the same
design / analysis replacing the treatment
variable with one that should not affect
the outcome. In the case of the RD this is
usually done by using multiple placebo
cutoffs. If something different was done,
please indicate in the comments.
(e) Other
10.Does the paper provide any of the following
sensitivity or robustness checks? Please se-
lect the “h) Other” option only in the case of
highly relevant sensitivity tests for the identifi-
cation strategy
(a)Sensitivity to the violation of the identifi-
cation assumptions (e.g., bounding, par-
tial identification)
(b)Sensitivity to the exclusion of observa-
tions near the cutoff (the so-called donut
hole approach)
(c)Sensitivity to bandwidth or window
choices (usually making it smaller, thus
excluding observations away from the
cutoff)
(d)Sensitivity to using an alternative RDD
framework (i.e., local randomization vs
continuity)
(e)Robustness to alternative kernels or poly-
nomial orders
(f)Sensitivity to the inclusion or exclusion
of covariates (robustness to controlling
for pre-treatment variables)
(g)Sensitivity to using alternative infer-
ence approaches (e.g., different inference
method, level of clustering etc.)
(h) Other

D.2.3 Instrumental Variables
1.How is the plausibility of the identification
assumption supported in the paper? In an
instrumental variables design, the key iden-
tification assumption is that the instrument
is independent of unobserved determinants
of the outcome — either unconditionally or
conditional on observed covariates. This is
often not guaranteed by design, especially in
observational studies, and must be defended
through theoretical arguments, institutional
details, balance tests, placebo tests, or robust-
ness checks. Being able to argue convincingly
that the instrument is as good as randomly
assigned can help strengthen the credibility of
the IV strategy. You may select more than one
option if the paper combines multiple forms
of justification (e.g., uses empirical tests and
argues for quasi-random variation). Consider
both what the authors claim and the kind of
variation they exploit. Do not infer justifica-
tion that is not discussed or implied by the
paper.
(a)Through arguments supporting a quasi-
random assignment due to plausibly ex-
ogenous policy variation. For example:
assignment is determined by administra-
tive quirks, implementation delays, pilot
rollout, or other arbitrary features of pol-
icy design that are plausibly unrelated to
potential outcome levels and trends. This
often corresponds to the so-called natural
experiments due to policy variations.
(b)Through other plausibly exogenous varia-
tion driven by contextual or historical fac-
tors. Assignment is determined by con-
textual forces such as natural disasters,
conflict, migration patterns, geographi-
cal features or long-standing institutional
or historical differences, claimed to gen-
erate quasi-random variation. This often
corresponds to the so-called natural ex-
periments due to contextual factors.
(c)Through policy eligibility rules that al-
low comparison between eligible and
non-eligible units. This applies when
the policy is assigned based on fixed
rules - such as socio-demographic char-
acteristics (e.g., age, income, family sta-
tus) or geography (e.g., some regions
treated, others not) - and units cannotself-select into treatment (or would have
to incur substantial costs to do so). This
justification is common in difference-in-
differences designs.
(d)Credibility defended based on theoreti-
cal or substantive reasoning (e.g., claims
based on social science theories and lim-
ited institutional, external or historical
variation)
(e)Through reference to previous studies us-
ing similar designs without further jus-
tification that the design is valid in the
current setting.
(f)Identification defended through empiri-
cal econometric tests (e.g., the paper sup-
ports its identification strategy through
balance checks, placebo tests, sensitivity
checks etc.)
(g) Other
2.What type of instrument are used in the study?
(a)Randomized assignment - instrument is
based on a randomized experiment (e.g.,
encouragement design, randomized eli-
gibility)
(b)Rules or policy changes - instrument
arises from policy thresholds or quasi-
random eligibility criteria (e.g., as in
RDD settings)
(c)Examiner designs (e.g., judge leniency,
using disparities among judges and other
decision makers to identify causal ef-
fects)
(d)Geography, weather, or climate-based -
instrument relies on spatial or natural
variation (e.g., rainfall, elevation, prox-
imity to coast or borders)
(e)Historical variation - instrument is based
on long-past events or historical institu-
tions (e.g., colonial presence, migration
history)
(f)Treatment diffusion (e.g., using US aid
in neighboring countries as instrument
for US aid in the country analyzed)
(g) Shift-share / Bartik instruments
(h) Other
3.How many instruments are used in the main
analysis?
(a) One instrument

(b) Two Instruments
(c) 3-10 instruments
(d) 11-20 instruments
(e) More than 20 instruments
(f) Unclear
4.What is the first-stage F-statistic in the pre-
ferred specifications? If there is no clearly
stated preferred specification, select all rele-
vant values from the main tables or text.
(a)Not reported, but authors claim the in-
strument is strong
(b)Not reported, but authors acknowledge
the instrument is weak
(c) Less than 10
(d) 10-20
(e) 21-30
(f) 31-50
(g) 51-100
(h) 101-200
(i) Over 200
(j)Not reported and not possible to assess
instrument validity
(k)Not reported but can be calculated based
on information report to the reader, such
as the first stage t-test
(l) Other
5.What type of F test statistic is computed?
(a) Not discussed or unclear
(b)Standard First-Stage F-Statistic assum-
ing homoskedasticity
(c)Kleibergen-Paap Wald F-Statistic (Het-
eroskedastic and Clustered Robust)
(d)Effective F-Statistic (Montiel-Olea and
Pflueger)
(e) Cragg-Donald F-Statistic
(f)Sanderson-Windmeijer (SW) Condi-
tional F-Statistic
(g) Bootstrapped F-statistic
(h) Other
6.Which estimator(s) does the study use to esti-
mate the causal effect in the main analysis?
(a)Two-stage Least Squares (2SLS), includ-
ing Wald estimator (Wald 1940, Angrist
and Krueger 1991)(b)Limited Information Maximum Like-
lihood (LIML) (Anderson and Rubin,
1949)
(c)Bias-Corrected 2SLS (Nagar, 1959, Don-
ald and Newey, 2001, Kolesár et al.,
2015, Anatolyev, 2011)
(d)Jackknife IV (JIVE), sometimes referred
to as “leave-one-out estimator” (Angrist,
Imbens and Krueger, 1999)
(e)Continuously Updated GMM (CUE)
(Hansen, Heaton, and Yaron, 1996)
(f)Standard two-step GMM (Generalized
Method of Moments)
(g)Local Instrumental Variables (LIV)
(Heckman and Vytlacil, 1999)
(h) Unclear or difficult to assess
(i) Other
7.Inference is performed through:
(a) Standard IV asymptotic inference
(b) Bootstrap-based Inference
(c) Anderson-Rubin (AR) Test
(d)Conditional Likelihood Ratio (CLR) Test
(e.g., Moreira, 2003)
(e) tF procedure (Lee et al. 2022)
(f) Unclear or difficult to assess
(g) Other
8.Is clustering appropriately accounted for in
the inference procedure in the second stage?
Note: The recommended practice is to ac-
count for clustering at the level of unit as-
signment when the unit of analysis is at a
lower level than the cluster (e.g., schools as-
signed but data is at student level). However,
if the unit of analysis is at the level of cluster-
ing (e.g., schools assigned, and data is aggre-
gated at school level), clustering is not neces-
sary. Clustering is also necessary when there
are multiple observations per unit used in the
analysis (e.g., repeated measures, panel data).
Extra: clustering is also necessary when there
is a risk of interference or spillovers, but this
is rarely done, so if the paper reports such,
please select “g) Other”, ignoring all other
response options"
(a)Yes - clustering at level of treatment as-
signment
(b)Yes - clustering below treatment assign-
ment (e.g., at class level when schools
are assigned)

(c)Yes - clustering above treatment assign-
ment (e.g., at school level, when classes
are assigned)
(d)No - clustering not used and not nec-
essary (e.g., individual-level or clus-
tered assignment but the unit of analysis
matches the treatment assignment level)
(e)No - clustering not used, but it should
have been
(f) Unclear / not reported
(g) Other
9.Are the first stage coefficients in line with the-
oretical predictions?
(a) Yes
(b) No
(c)Not discussed or first stage estimates not
reported
(d) Unclear or difficult to assess
10.Does the paper compare the IV estimates to
the OLS estimates?
(a)Yes - and IV estimates are larger in abso-
lute values
(b) Yes - and IV estimates are smaller in ab-
solute values
(c)Yes - IV estimates are similar to OLS
estimates
(d) No - OLS estimates are not reported
(e) Unclear or difficult to assess
11.If yes, how are the differences between IV and
OLS estimates justified?
(a)Selection arguments (or lack of selection
bias) - OLS is considered biased due to
negative or positive selection into treat-
ment of units
(b)Complier difference - it argues that dif-
ferences are likely to differences of com-
pliers relative to the population
(c)Measurement error (i.e., downward bias
OLS estimates due to measurement error
in the treatment variable)
(d) No clear justification offered
12.Does the study include any falsification
(placebo) tests to support the identification
strategy? Please select the “e) Other” option
only if it is a highly relevant falsification tests
for the identification strategy(a)Post-treatment placebo outcomes test -
applying the same design / analysis re-
placing the outcome variable with one
which should not be impacted by the
treatment or instrument
(b)Placebo treatment / instrument - applying
the same design / analysis replacing the
treatment or instrument variable with one
that should not affect the outcome.
(c)Placebo population test - applying the
same design / analysis in a sample where
the treatment or instrument is expected
to not have an effect
(d)Placebo / Balance on pre-treatment co-
variates or pre-treatment outcomes, ei-
ther unconditionally or conditional on a
sub-set of covariates which are consid-
ered sufficient to make the instrument as
good as random
(e) Other
13.What type of sensitivity analyses does the
study conduct? Please select the “f) Other”
option only in the case of other highly relevant
sensitivity tests for the identification strategy
(a)Sensitivity to the violation of the identifi-
cation assumptions (e.g., bounding, par-
tial identification)
(b) Robustness to alternative estimators
(c)Sensitivity to using alternative inference
approaches, first or second stage
(d)Robustness to weak instruments proce-
dures
(e) Robustness to the exclusion of outliers
(f) Other
14.Does the study perform any overidentification
tests? Examples of overidentification tests:
Sargan test, Hansen J-test
(a) Yes
(b) No overidentification test reported
(c)Not applicable - model is exactly iden-
tified (same number of instruments as
endogenous variables)
15.Does the paper explicitly discuss or provide
formal empirical support for the exclusion
restriction in the IV setting? The exclusion
restriction assumes that the instrument affects

the outcome only through its effect on the treat-
ment (i.e., there is no direct effect of the in-
strument on the outcome)
(a)Discussion, i.e., discussed qualitatively
or formal tests (e.g., supportive empirical
evidence) provided
(b) No - does not mention or address
(c) Unclear or hard to assess
16.Does the paper explicitly discuss or provide
formal empirical support for the monotonicity
assumption in the IV setting? Monotonicity
assumes that the instrument affects the treat-
ment in the same direction for all units (i.e.,
there are no defiers)
(a)Discussion, i.e., discussed qualitatively
or formal tests (e.g., supportive empirical
evidence) provided
(b) No - does not mention or address
(c) Unclear or hard to assess
D.2.4 Selection-on-observables (e.g.,
regression adjustment, matching,
weighting doubly robust
1.How is the plausibility of the identification
assumption supported in the paper?
(a)Through arguments supporting a quasi-
random assignment due to plausibly ex-
ogenous policy variation. For example:
assignment is determined by administra-
tive quirks, implementation delays, pilot
rollout, or other arbitrary features of pol-
icy design that are plausibly unrelated to
potential outcome levels and trends. This
often corresponds to the so-called “natu-
ral experiments” due to policy variations.
(b)Through other plausibly exogenous varia-
tion driven by contextual or historical fac-
tors. Assignment is determined by con-
textual forces such as natural disasters,
conflict, migration patterns, geographi-
cal features or long-standing institutional
or historical differences, claimed to gen-
erate quasi-random variation. This often
corresponds to the so-called “natural ex-
periments” due to contextual factors
(c)Through policy eligibility rules that al-
low comparison between eligible and
non-eligible units This applies when the
policy is assigned based on fixed rules— such as socio-demographic charac-
teristics (e.g., age, income, family sta-
tus) or geography (e.g., some regions
treated, others not) - and units cannot
self-select into treatment (or would have
to incur substantial costs to do so). This
justification is common in difference-in-
differences designs
(d)Credibility defended based on theoreti-
cal or substantive reasoning (e.g., claims
based on social science theories and lim-
ited institutional, external or historical
variation)
(e)Through reference to previous studies us-
ing similar designs without further jus-
tification that the design is valid in the
current setting.
(f)Identification defended through empiri-
cal econometric tests (e.g., the paper sup-
ports its identification strategy through
balance checks, placebo tests, sensitivity
checks etc.)
(g) Other
2.Which of the following describe the set of co-
variates used to support the unconfounded-
ness assumption?
(a)Basic set of socio-demographic covari-
ates (e.g., gender, age in a study on indi-
viduals)
(b)Rich set of socio-demographic covariates
(e.g., family education, income, employ-
ment status etc.)
(c)Pre-treatment outcome or close proxy
available for one period
(d)Pre-treatment outcome or close proxy
available for multiple periods
(e)Post-treatment variables potentially im-
pacted by the treatment (i.e., problematic,
collider bias risk)
(f) Other
3.Does the study provide any evidence for the
credibility of the common support assumption,
either on covariates or a distance measure
(e.g., propensity score)?
(a) Yes
(b) No
4.Is any trimming or sample restriction applied
to improve overlap/common support?

(a)Yes, keeping units on the common sup-
port
(b)Yes, dropping units with high or low
propensity scores usingad hocthresh-
olds (e.g., 0.1 and 0.9)
(c)Yes, dropping units with high or low
propensity scores in a data driven way
(d)Yes, dropping units based on otherad
hoccriteria
(e) There were no common support issues
(f) No trimming discussed
(g) Other
5.Does the study provide evidence of good co-
variate balance after adjustment? Balance
may be assessed based on standardized mean
differences and variance ratios, plots, or other
tables with estimates of differences by group
(a)Yes, balance is high with small remaining
differences
(b)Partially, balance is good with some re-
maining differences in relevant variables
(c)Inadequate balance, with many remain-
ing differences between the groups
(d) Unclear or difficult to assess
6.Which estimation methods does the study use
in the main analyses? Please note that some
approaches are combined with regression ad-
justments. In that case, please select more
than one option
(a)Outcome modelling, i.e., regression ad-
justments (e.g., linear regression, kernel
regression, spline-based adjustments)
(b)Exact matching or Coarsened Exact
Matching
(c)Matching on the propensity score (near-
est neighbor, genetic matching, subclas-
sification etc.)
(d)Weighting methods (e.g., inverse proba-
bility weighting [IPW], entropy balanc-
ing)
(e)Doubly robust methods (e.g., augmented
inverse probability weighting, double
machine learning)
(f) Other
7.Is clustering appropriately accounted for in
the inference procedure? Note: The recom-
mended practice is to account for clusteringat the level of unit assignment when the unit
of analysis is at a lower level than the cluster
(e.g., schools assigned but data is at student
level). However, if the unit of analysis is at the
level of clustering (e.g., schools assigned, and
data is aggregated at school level), clustering
is not necessary. Clustering is also necessary
when there are multiple observations per unit
used in the analysis (e.g., repeated measures,
panel data). Extra: clustering is also neces-
sary when there is a risk of interference or
spillovers, but this is rarely done, so if the
paper reports such, please indicate in other,
ignoring all other response options
(a)Yes - clustering at level of treatment as-
signment
(b)Yes - clustering below treatment assign-
ment (e.g., at class level when schools
are assigned)
(c)Yes - clustering above treatment assign-
ment (e.g., at school level, when classes
are assigned)
(d)No - clustering not used and not nec-
essary (e.g., individual-level or clus-
tered assignment but the unit of analysis
matches the treatment assignment level)
(e)No - clustering not used, but it should
have been
(f) Unclear / not reported
(g) Other
8.Does the study include any falsification
(placebo) tests to support the identification
strategy? Please select the “f) Other” option
only if it is a highly relevant falsification test
for the identification strategy
(a)Post-treatment placebo outcomes test -
applying the same design / analysis re-
placing the outcome variable with one
which should not be impacted by the
treatment
(b)Placebo treatment - applying the same
design / analysis replacing the treatment
variable with one that should not affect
the outcome.
(c)Placebo population test - applying the
same design / analysis in a sample where
the treatment is expected to not have an
effect

(d)Placebo / Balance on pre-treatment co-
variates - excluding a sub-set of covari-
ates from the balancing and checking for
differences
(e)Placebo / Balance on Placebo pre-
treatment outcomes or close proxies -
e.g., excluding one or more pre-treatment
outcomes from the balancing and assess-
ing for differences
(f) Other
9.What type of sensitivity analyses does the
study conduct?
(a)Sensitivity to the violation of the identifi-
cation assumptions (e.g., bounding, par-
tial identification)
(b) Robustness to alternative estimators
(c)Sensitivity to the selection of covariates
included in the balancing
(d)Robustness to alternative trimming strate-
gies
(e)Sensitivity to using alternative infer-
ence approaches (e.g., different inference
method, level of clustering etc.)
(f) Other
D.2.5 Difference in Differences (DiD) and
related designs (e.g., event studies,
changes-in-changes, triple differences)
1.How is the credibility of main identification
strategy supported in the paper? In a DiD set-
ting the key identification assumption is that
of parallel trends. This is generally defended
empirically by showing that there were paral-
lel trends before the treatment and assuming
that such trends are assumed to hold also in
the post-treatment period in the absence of
the treatment. This is not always credible,
as assignment may be due to expectations re-
garding the evolution of the outcome in the
post-treatment period (e.g., there are expec-
tations that growth will slow down only in
some regions and the policy is implemented
to contrast that). Being able to argue that this
was not the case and that there was some id-
iosyncratic variation can help strengthen the
identification strategy. You may select more
than one option if the paper combines multi-
ple forms of justification (e.g., uses empirical
tests and argues for quasi-random variation).
Consider both what the authors claim and thekind of variation they exploit. Do not infer
justification that is not discussed or implied
by the paper.
(a)Through arguments supporting a quasi-
random assignment due to plausibly ex-
ogenous policy variation. For example:
assignment is determined by administra-
tive quirks, implementation delays, pilot
rollout, or other arbitrary features of pol-
icy design that are plausibly unrelated to
potential outcome levels and trends. This
often corresponds to the so-called “natu-
ral experiments” due to policy variations.
(b)Through other plausibly exogenous varia-
tion driven by contextual or historical fac-
tors. Assignment is determined by con-
textual forces such as natural disasters,
conflict, migration patterns, geographi-
cal features or long-standing institutional
or historical differences, claimed to gen-
erate quasi-random variation. This often
corresponds to the so-called “natural ex-
periments” due to contextual factors
(c)Through policy eligibility rules that al-
low comparison between eligible and
non-eligible units. This applies when
the policy is assigned based on fixed
rules — such as socio-demographic char-
acteristics (e.g., age, income, family sta-
tus) or geography (e.g., some regions
treated, others not) — and units cannot
self-select into treatment (or would have
to incur substantial costs to do so). This
justification is common in difference-in-
differences designs
(d)Credibility defended based on theoreti-
cal or substantive reasoning (e.g., claims
based on social science theories and lim-
ited institutional, external or historical
variation)
(e)Through reference to previous studies us-
ing similar designs without further jus-
tification that the design is valid in the
current setting.
(f)Identification defended through empir-
ical econometric tests (e.g., The pa-
per supports its identification strategy
through balance checks, placebo tests,
sensitivity checks etc.)
(g) Other

2.What is the structure of treatment assignment
across units and over time?
(a)Blocked design - all treated units receive
treatment at the same time (e.g., single
policy shock or eligibility event)
(b)Staggered design - treatment is rolled
out at different times across units/groups
(e.g., cohort or regional rollouts of the
program across time)
(c) Other
3.Is the no-anticipation assumption discussed or
addressed? This refers to whether the paper
considers whether units could respond before
the official treatment start
(a)Yes - discussed and addressed where rel-
evant (e.g., by backdating treatment or
showing no early effects)
(b)Not discussed, and anticipation is plausi-
ble (e.g., policy announcement precedes
implementation)
(c)Not discussed but plausibly there are no
anticipation concerns
4.What is the time unit of observation (i.e., data
frequency)?
(a) Daily
(b) Weekly
(c) Monthly
(d) Quarterly
(e) Yearly
(f) Other
5.What is the number of pre-periods available
(i.e., before the treatment/intervention starting
date)? Please note that if the paper argues
there are anticipation effects, the treatment
starting date should be back-dated thus the
number of pre-periods should only consider
the period preceding the back-dated treatment
starting date.
(a) 1 period
(b) 2-3 periods
(c) 4-5 periods
(d) 6-10 periods
(e) 11-20 periods
(f) 21-40 periods
(g) more than 40 periods(h)Unclear or difficult to assess based on
the information provided
6.What is the number of pre-periods available
(i.e., before the treatment/intervention starting
date)? Please note that if the paper argues
there are anticipation effects, the treatment
starting date should be back-dated thus the
number of pre-periods should only consider
the period preceding the back-dated treatment
starting date.
(a) 1 period
(b) 2-3 periods
(c) 4-5 periods
(d) 6-10 periods
(e) 11-20 periods
(f) 21-40 periods
(g) more than 40 periods
(h)Unclear or difficult to assess based on
the information provided
7.What is the number of post-periods available
(i.e., after the treatment/intervention starting
date)? Please note that if the paper argues
there are anticipation effects, the treatment
starting date should be back-dated thus the
number of post-periods should consider the
periods after the back-dated treatment start-
ing date
(a) 1 period
(b) 2-3 periods
(c) 4-5 periods
(d) 6-10 periods
(e) 11-20 periods
(f) 21-40 periods
(g) more than 40 periods
(h)Unclear or difficult to assess based on
the information provided
8.Does the study provide formal or graphical
analysis of pre-treatment trends by treatment
groups? Please note that if the paper argues
there are anticipation effects, the treatment
starting date should be back-dated and such
parallel pre-trends analyses should be done
on the period preceeding any anticipation ef-
fects.
(a)Yes, group-level mean outcome plots
over time (e.g., raw or smoothed trends
by treatment and control groups)

(b)Yes, event study plots (or estimates re-
ported in a different format such as a
table)
(c)Not provided in the main text but ar-
gued that parallel trends hold in the pre-
treatment period
(d) No - neither discussed nor shown
(e) Other
9.Do the treatment and control groups appear
to follow parallel trends prior to the interven-
tion?
(a)Yes - trends are visually and/or statisti-
cally similar
(b)No - trends are systematically different
/ diverging or there are many periods
where differences are large
(c)Unclear or difficult to assess based on
the information provided
10.Which estimator(s) does the study use in the
main analysis to estimate treatment effects?
(a)Standard DiD / two-way fixed effects
- two-way fixed effects with group and
time fixed effects and a treatment indica-
tor (e.g., classic 2×2 DiD or panel DiD)
(b)Event-study with TWFE - the treatment
is typically interacted with relative time
periods (e.g., event time indicators) to
estimate dynamic effects
(c)Callaway and Sant’Anna (2021) estima-
tor - group-time average treatment ef-
fects estimator
(d)Imputation-based estimators - e.g.,
Borusyak, Jaravel, Spiess (2021), Gard-
ner (2021), Liu et al. (2022), Wooldridge
(2022)
(e)Sun and Abraham (2021) - interaction-
weighted estimator for dynamic effects
under staggered adoption
(f)Stacked DID (i.e., Wing, Freedman and
Hollingsworth, 2024)
(g) Other
11.Is clustering appropriately accounted for in
the inference procedure? Note: The recom-
mended practice is to account for clustering
at the level of unit assignment when the unit
of analysis is at a lower level than the cluster
(e.g., schools assigned but data is at studentlevel). However, if the unit of analysis is at the
level of clustering (e.g., schools assigned, and
data is aggregated at school level), clustering
is not necessary. Clustering is also necessary
when there are multiple observations per unit
used in the analysis (e.g., repeated measures,
panel data). Extra: clustering is also neces-
sary when there is a risk of interference or
spillovers, but this is rarely done, so if the
paper reports such, please select “g) Other”,
ignoring all other response options
(a)Yes - clustering at level of treatment as-
signment
(b)Yes - clustering below treatment assign-
ment (e.g., at class level when schools
are assigned)
(c)Yes - clustering above treatment assign-
ment (e.g., at school level, when classes
are assigned)
(d)No - clustering not used and not nec-
essary (e.g., individual-level or clus-
tered assignment but the unit of analysis
matches the treatment assignment level)
(e)No - clustering not used, but it should
have been
(f) Unclear / not reported
(g) Other
12.Does the study include other falsification
(placebo) tests? Please select the “d) Other”
option only in the case of highly relevant fal-
sification tests for the identification strategy
(a)Post-treatment placebo outcomes test -
applying the same design / analysis re-
placing the outcome variable with one
which should not be impacted by the
treatment
(b)Placebo population test - applying the
same design / analysis in a sample where
the treatment is expected to not have an
effect
(c)Placebo treatment - applying the same
design / analysis replacing the treatment
variable with one that should not affect
the outcome.
(d) Other
13.What type of sensitivity analyses does the
study conduct? Please select the “g) Other”
option only in the case of highly relevant sen-
sitivity tests for the identification strategy

(a)Sensitivity to the violation of the identifi-
cation assumptions (e.g., bounding, par-
tial identification)
(b)Sensitivity of the parallel trends assump-
tion to the chosen functional form of the
outcome
(c)Assessment of the power of pre-trends
tests against economically relevant viola-
tions of parallel trends
(d)Robustness to alternative estimators (e.g.,
using modern estimators are used only
as robustness please specify here)
(e)Robustness to assuming parallel trends
only conditional on covariates
(f)Sensitivity to using alternative infer-
ence approaches (e.g., different inference
method, level of clustering etc.)
(g) Other
14.Do any of the following issues or assumptions
arise in the study’s treatment effect estima-
tion?
(a)The study includes in the analysis also
periods where all units are treated
(b)The study provides theoretical, graphi-
cal or formal arguments to argue for the
homogeneity of treatment effects (e.g.,
effects not evolving dynamically, and the
same effects across units)
(c) None of the above
D.2.6 Synthetic Control Methods (SCMs) and
related designs (e.g., augmented
synthetic control, synthetic
difference-in-differences)
1.How is the plausibility of the identification
assumption supported in the paper? Select
the option that best describes how the study
justifies the exogeneity of treatment or instru-
ment assignment. Consider both the source of
variation and the justification provided. In a
SC setting the implicit assumption is similar
to a DiD setting, that is, in the absence of the
treatment the trends observed pre-intervention
are assumed to hold. This is not always credi-
ble, as assignment may be due to expectations
regarding the evolution of the outcome in the
post-treatment period (e.g., there are expec-
tations that growth will slow down only in
some regions and the policy is implemented
to contrast that). Being able to argue that thiswas not the case and that there was some id-
iosyncratic variation can help strengthen the
identification strategy
(a)Through arguments supporting a quasi-
random assignment due to plausibly ex-
ogenous policy variation. For example:
assignment is determined by administra-
tive quirks, implementation delays, pilot
rollout, or other arbitrary features of pol-
icy design that are plausibly unrelated to
potential outcome levels and trends. This
often corresponds to the so-called “natu-
ral experiments” due to policy variations.
(b)Through other plausibly exogenous varia-
tion driven by contextual or historical fac-
tors. Assignment is determined by con-
textual forces such as natural disasters,
conflict, migration patterns, geographi-
cal features or long-standing institutional
or historical differences, claimed to gen-
erate quasi-random variation. This often
corresponds to the so-called “natural ex-
periments” due to contextual factors
(c)Through policy eligibility rules that al-
low comparison between eligible and
non-eligible units. This applies when
the policy is assigned based on fixed
rules — such as socio-demographic char-
acteristics (e.g., age, income, family sta-
tus) or geography (e.g., some regions
treated, others not) — and units cannot
self-select into treatment (or would have
to incur substantial costs to do so). This
justification is common in difference-in-
differences designs
(d)Credibility defended based on theoreti-
cal or substantive reasoning (e.g., claims
based on social science theories and lim-
ited institutional, external or historical
variation)
(e)Through reference to previous studies us-
ing similar designs without further jus-
tification that the design is valid in the
current setting.
(f)Identification defended through empiri-
cal econometric tests (e.g., the paper sup-
ports its identification strategy through
balance checks, placebo tests, sensitivity
checks etc.)
(g) Other

2.Please indicate the number of treated units
used in the study
(a) 1
(b) 2-5
(c) 6-20
(d) 21-50
(e) 51-100
(f) More than 100
3.Please indicate the number of control units
used in the study
(a) 1
(b) 2-5
(c) 6-20
(d) 21-50
(e) 51-100
(f) More than 100
4.Is the no-anticipation assumption discussed or
addressed? This refers to whether the paper
considers whether units could respond before
the official treatment start.
(a)Yes - discussed and addressed where rel-
evant (e.g., by backdating treatment or
showing no early effects)
(b)Not discussed, and anticipation is plausi-
ble (e.g., policy announcement precedes
implementation)
(c)Not discussed but plausibly there are no
anticipation concerns
5.What is the time unit of observation (i.e., data
frequency)?
(a) Daily
(b) Weekly
(c) Monthly
(d) Quarterly
(e) Yearly
(f) Other
6.What is the number of pre-periods available
(i.e., before the treatment/intervention starting
date)? Please note that if the paper argues
there are anticipation effects, the treatment
starting date should be back-dated thus the
number of pre-periods should only consider
the period preceding the back-dated treatment
starting date.
(a) 1 period(b) 2-3 periods
(c) 4-5 periods
(d) 6-10 periods
(e) 11-20 periods
(f) 21-40 periods
(g) More than 40 periods
(h)Unclear or difficult to assess based on
the information provided
7.What is the number of post-periods available
(i.e., after the treatment/intervention starting
date)? Please note that if the paper argues
there are anticipation effects, the treatment
starting date should be back-dated thus the
number of post-periods should consider the
periods after the back-dated treatment start-
ing date
(a) 1 period
(b) 2-3 periods
(c) 4-5 periods
(d) 6-10 periods
(e) 11-20 periods
(f) 21-40 periods
(g) More than 40 periods
(h)Unclear or difficult to assess based on
the information provided
8.Is there a high volatility of the outcome in the
pre-treatment period (or average outcome in
the case of multiple units)? Outcome vari-
ables that include substantial random noise
elevate the risk of over-fitting. It is advisable
to remove it via filtering, in both the exposed
unit as well as in the units in the donor pool,
before applying synthetic control techniques.
This is especially concerning in comparative
case studies (one or few treated units). With
many units, averaging reduces noise.
(a) No - the series is not highly volatile
(b)Yes - but filtered or smoothed prior to
applying SC
(c)Yes - fitting is performed directly on
volatile data
(d)Unclear or difficult to assess based on
the information provided
9.Which of the following are used to construct
the synthetic control and assess pre-treatment
balance?

(a)Pre-treatment outcome variables (lagged
outcome levels or trends)
(b)Standard demographic or economic
covariates (e.g., age, gender in an
individual-level study; population, edu-
cation rate, unemployment in a mezzo or
macro level study)
(c)Covariates argued in the paper to be
highly relevant for treatment or outcomes
(e.g., think of ability or expectations
proxies in an individual study; or indus-
try composition, political alignment, fis-
cal variables in a macro study)
(d) Other
10.How well does the synthetic control match
the treated unit(s) on the outcome in the pre-
treatment period? Base your answer on re-
ported fit statistics and/or graphical compar-
isons of treated vs. synthetic outcomes before
treatment
(a)Good fit - small and not systematic dif-
ferences between treated and synthetic
outcomes, visually good match and/or
low RMSPE reported
(b)Poor fit - large systematic gaps between
treated and synthetic outcomes before
treatment
(c) Unclear or difficult to assess
(d)No quantitative or visual assessment pro-
vided
(e) Other
11.Is there any donor pool trimming performed
prior to the construction of the synthetic con-
trol?
(a)Yes - donor pool units were excluded due
to exposure to similar policies or contem-
poraneous shocks
(b)Yes - donor pool units considered sub-
stantially different compared to the
treated units (i.e., with the goal of avoid-
ing interpolation bias)
(c)No - the donor pool was not trimmed or
restricted
(d)Unclear or difficult to assess based on
the information provided
(e) Other12.Which approaches are used to construct the
synthetic control and estimate treatment ef-
fects?
(a)Classical SCM - Abadie & Gardeaza-
bal (2003); Abadie, Diamond & Hain-
mueller (2010)
(b)Augmented or Penalized Synthetic Con-
trol - e.g., Ben-Michael, Feller & Roth-
stein (2021); Abadie & L’Hour (2021)
(c)Synthetic Difference-in-Differences
(SDID)
(d) Generalized SCM (Xu, 2017)
(e)Bayesian Synthetic Control — e.g., Scott
& Varian, 2014; Hazlett & Xu, 2018
(f) Other
13.What inference methods are used to assess
statistical significance and uncertainty?
(a) Permutation / Placebo tests
(b) Jackknife (e.g., leave-one-out methods)
(c) Bootstrap
(d)Conformal Inference (Chernozhukov,
Wüthrich & Zhu, 2021)
(e) Other
14.Is clustering appropriately accounted for in
the inference procedure? Note: The recom-
mended practice is to account for clustering
at the level of unit assignment when the unit
of analysis is at a lower level than the cluster
(e.g., schools assigned but data is at student
level). However, if the unit of analysis is at the
level of clustering (e.g., schools assigned, and
data is aggregated at school level), clustering
is not necessary. Clustering is also necessary
when there are multiple observations per unit
used in the analysis (e.g., repeated measures)
Extra: clustering is also necessary when there
is a risk of interference or spillovers, but this
is rarely done, so if the paper reports such,
please select “g) Other”, ignoring all other
response options
(a)Yes - clustering at level of treatment as-
signment
(b)Yes - clustering below treatment assign-
ment (e.g., at class level when schools
are assigned)
(c)Yes - clustering above treatment assign-
ment (e.g., at school level, when classes
are assigned)

(d)No - clustering not used and not nec-
essary (e.g., individual-level or clus-
tered assignment but the unit of analysis
matches the treatment assignment level)
(e)No - clustering not used, but it should
have been
(f) Unclear / not reported
(g) Other
15.Does the study include falsification (placebo)
tests? Please select the “e) Other” option
only if it is a highly relevant falsification tests
for the identification strategy
(a)Post-treatment placebo outcomes test -
applying the same design / analysis re-
placing the outcome variable with one
which should not be impacted by the
treatment
(b)Placebo treatment - applying the same
design / analysis replacing the treatment
variable with one that should not affect
the outcome. The usual approach is back-
dating or “in time placebo tests” - setting
a placebo treatment date prior to the ac-
tual treatment date and checking if the
trend diverges prior to the actual treat-
ment
(c)Placebo population test - applying the
same design / analysis in a sample where
the treatment is expected to not have an
effect. Consider only analysis other than
the permutation tests used for inference.
(d) None of the above
(e) Other
16.What type of sensitivity analyses does the
study conduct? Please select the “g) Other”
option only in the case of highly relevant sen-
sitivity tests for the identification strategy
(a)Sensitivity to the violation of the identifi-
cation assumptions (e.g., bounding, par-
tial identification)
(b)Robustness to the choice of covariates
(e.g., as suggested by Ferman, Pinto, and
Possebom, 2020)
(c)Robustness to the choice of units in the
donor pool - are results robust to exclud-
ing units from the donor pool, usually
units which receive higher weights in the
main analysis(d)Robustness to alternative estimators (e.g.,
a bias correction approach, synthetic
DID, matrix completion)
(e)Sensitivity to using alternative infer-
ence approaches (e.g., different inference
method, level of clustering etc.)
(f) None of the above
(g) Other