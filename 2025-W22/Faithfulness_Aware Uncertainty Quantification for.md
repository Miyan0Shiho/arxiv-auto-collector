# Faithfulness-Aware Uncertainty Quantification for Fact-Checking the Output of Retrieval Augmented Generation

**Authors**: Ekaterina Fadeeva, Aleksandr Rubashevskii, Roman Vashurin, Shehzaad Dhuliawala, Artem Shelmanov, Timothy Baldwin, Preslav Nakov, Mrinmaya Sachan, Maxim Panov

**Published**: 2025-05-27 11:56:59

**PDF URL**: [http://arxiv.org/pdf/2505.21072v2](http://arxiv.org/pdf/2505.21072v2)

## Abstract
Large Language Models (LLMs) enhanced with external knowledge retrieval, an
approach known as Retrieval-Augmented Generation (RAG), have shown strong
performance in open-domain question answering. However, RAG systems remain
susceptible to hallucinations: factually incorrect outputs that may arise
either from inconsistencies in the model's internal knowledge or incorrect use
of the retrieved context. Existing approaches often conflate factuality with
faithfulness to the retrieved context, misclassifying factually correct
statements as hallucinations if they are not directly supported by the
retrieval. In this paper, we introduce FRANQ (Faithfulness-based Retrieval
Augmented UNcertainty Quantification), a novel method for hallucination
detection in RAG outputs. FRANQ applies different Uncertainty Quantification
(UQ) techniques to estimate factuality based on whether a statement is faithful
to the retrieved context or not. To evaluate FRANQ and other UQ techniques for
RAG, we present a new long-form Question Answering (QA) dataset annotated for
both factuality and faithfulness, combining automated labeling with manual
validation of challenging examples. Extensive experiments on long- and
short-form QA across multiple datasets and LLMs show that FRANQ achieves more
accurate detection of factual errors in RAG-generated responses compared to
existing methods.

## Full Text


<!-- PDF content starts -->

arXiv:2505.21072v2  [cs.CL]  28 May 2025Faithfulness-Aware Uncertainty Quantification for
Fact-Checking the Output of Retrieval Augmented Generation
Ekaterina Fadeeva1♢Aleksandr Rubashevskii2♢Roman Vashurin2
Shehzaad Dhuliawala1Artem Shelmanov2Timothy Baldwin2
Preslav Nakov2Mrinmaya Sachan1Maxim Panov2
1ETH Zürich2MBZUAI
{efadeeva,sdhuliawala,msachan}@ethz.ch
{aleksandr.rubashevskii,preslav.nakov,maxim.panov}@mbzuai.ac.ae
Abstract
Large Language Models (LLMs) enhanced
with external knowledge retrieval, an approach
known as Retrieval-Augmented Generation
(RAG), have shown strong performance in
open-domain question answering. However,
RAG systems remain susceptible to halluci-
nations: factually incorrect outputs that may
arise either from inconsistencies in the model’s
internal knowledge or incorrect use of the re-
trieved context. Existing approaches often
conflate factuality with faithfulness to the re-
trieved context, misclassifying factually correct
statements as hallucinations if they are not di-
rectly supported by the retrieval. In this pa-
per, we introduce FRANQ (Faithfulness-based
Retrieval Augmented UNcertainty Quantifica-
tion), a novel method for hallucination detec-
tion in RAG outputs. FRANQ applies different
Uncertainty Quantification (UQ) techniques to
estimate factuality based on whether a state-
ment is faithful to the retrieved context or not.
To evaluate FRANQ and other UQ techniques
for RAG, we present a new long-form Question
Answering (QA) dataset annotated for both fac-
tuality and faithfulness, combining automated
labeling with manual validation of challeng-
ing examples. Extensive experiments on long-
and short-form QA across multiple datasets
and LLMs show that FRANQ achieves more
accurate detection of factual errors in RAG-
generated responses compared to existing meth-
ods.
1 Introduction
Large Language Models (LLMs) are increasingly
being used for a wide range of tasks. However,
LLMs are prone to generate plausible but factually
incorrect generations, termed hallucinations, due
to limited training data, ambiguity of input queries,
architectural limitations, etc. (Huang et al., 2025).
External knowledge retrieval can partially help pre-
vent factual errors (Shuster et al., 2021). Retrieval
♢Equal contributionAugmented Generation (RAG; Lewis et al. (2020))
which uses dynamically retrieved knowledge as in-
put to generate LLM responses has been proposed
as a way to address these limitations.
However, despite the use of external knowledge,
RAG still produces hallucinations (Shi et al., 2023).
Moreover, the use of retrieved information makes
it challenging to detect hallucinations in LLM re-
sponses and determine their original source. The
model becomes more confident in generating state-
ments that appear in the retrieval, regardless of
their correctness (Kim et al., 2024). The internal
knowledge of the LLM and the retrieved informa-
tion may contradict each other (Wang et al., 2024a,
2025). The retrieved information may even be erro-
neous, incomplete, or completely irrelevant to the
query (Shi et al., 2023; Ding et al., 2024).
Thus, an important question is how to define hal-
lucination in a RAG system, given the interplay
between the model’s internal knowledge and the
retrieved context. One approach is to consider any
content not directly supported by the retrieved con-
text as a hallucination (Niu et al., 2024). However,
we argue that hallucination should be defined based
on factual inaccuracies rather than strict contextual
alignment. Specifically, a generated statement that
originates from the LLM’s internal knowledge but
lies outside the retrieved context should not be con-
sidered a hallucination if it is factually correct.
To address this distinction, we differentiate be-
tween factuality and faithfulness . Faithfulness
refers to whether the generated output is seman-
tically entailed by the retrieved context, while fac-
tuality indicates whether the content is objectively
correct (Dziri et al., 2022; Yang et al., 2024). In the
context of fact-checking of RAG output, detecting
non-factual claims is more critical than identifying
unfaithful ones, although assessing faithfulness can
still indirectly aid in fact-checking.
In this paper, we investigate the detection of
non-factual statements produced by RAG using
1

95%
10%
1%98% 50%
80% 90%
4% 60%Factuality: 
96%
Factuality: 
90%
Factuality: 
5%
2. If claim is 
faithful, is it 
factual? 3. If claim is 
unfaithful, is it 
factual? FRANQ 
Claims extracted 
from answer: 
Hmm, can I 
trust that? 4. Calculate factuality 
as Total Probability 
A bicycle works by 
pedaling to spin the 
wheels 
A bicycle uses 
handlebars to steer 
A bicycle burns fuel to 
maintain speed 
How does a bicycle work? 
A bicycle works by pedaling to spin the 
wheels, using handlebars to steer, and 
burning fuel to maintain speed ...1. Is claim 
faithful? 
I found top-3 retrievals for this question: 
Passage 1: A bicycle moves forward when the 
rider pedals, turning the chain and rotating the  
rear wheel  …
Passage 2: Bicycles use air-filled rubber tires 
to reduce …
Passage 3: Brakes are used for stopping, and 
handlebars …
Now, I will answer the question possibly using 
these retrievals Figure 1: FRANQ illustration. Left: A user poses a question, and the RAG retrieves relevant documents and
formulates an answer, potentially using information from the retrieved documents. Middle : The RAG output is
decomposed into atomic claims. Right : The FRANQ method assesses factuality by evaluating three components:
faithfulness and factuality under both faithful and unfaithful conditions.
Uncertainty Quantification (UQ) techniques. We
introduce FRANQ (Faithfulness-based Retrieval
Augmented U Ncertainty Quantification), a novel
method that first evaluates the faithfulness of the
generated response and subsequently applies differ-
ent UQ methods based on the outcome.
We evaluate FRANQ on both long-form and short-
form QA tasks. For long-form QA, where answers
include multiple claims, we assess factuality on
the claim level. For that, we introduce a dataset
with factuality annotations, combining automated
labeling with manual validation of challenging ex-
amples. For short-form QA, where answers are
brief, we treat each response as a single claim and
evaluate it as a whole. We test our method on four
QA datasets in this setting.
The key contributions of this work are:
•We develop a new UQ method for RAG,
FRANQ , that estimates uncertainty by first as-
sessing faithfulness, and then enabling sep-
arate uncertainty quantification methods for
faithful and unfaithful outputs; see Section 2.
•We develop a long-form QA factuality dataset
for RAG. The dataset incorporates both factu-
ality and faithfulness labels, and was build by
combining automatic annotation with manual
validation for difficult cases; see Section 3.
• We conduct comprehensive experimental val-
idation on both long-form and short-form
QA across several LLMs, demonstrating that
FRANQ significantly improves the detection
of factual errors in RAG outputs compared to
other approaches; see Section 4.2 Uncertainty Quantification for RAG
Letxbe the user query submitted to the RAG sys-
tem. The system first retrieves kpassages, denoted
byr={r1, . . . , r k}, from an external knowledge
source using xas the query. The RAG model then
employs a LLM to generate a greedy output y, con-
ditioned on both xandr.
Autoregressive LLMs produce text sequentially,
generating one token at a time. At each step t, the
model samples a token yt∼p(· |y<t), where
y<tdenotes the sequence of previously generated
tokens. In the case of greedy decoding, this token
is selected as the most likely outcome, i.e., yt=
arg max yp(y|y<t). From y, we extract a set of l
atomic claims, denoted as c1, . . . , c l. Each claim ci
is associated with a specific span of tokens, S(ci),
which represents the indices of the tokens in ythat
correspond to this particular claim.
A claim cis considered factually true if it is
generally true, and false otherwise. A claim is
deemed faithful with respect to the retrieved doc-
uments rif it is entailed by them, and unfaithful
otherwise. While most current benchmarks for
evaluating RAG outputs focus on evaluating faith-
fulness (Dziri et al., 2022; Niu et al., 2024), our
main objective is to assess the factuality of claims.
General baselines. A straightforward approach to
hallucination detection is to concatenate xandr,
and apply standard UQ techniques directly to this
joint prompt. While this approach allows reuse of
existing UQ methods, it disregards the structural
distinction between the question and its retrieved
context.
For example, a standard UQ baseline is the prob-
2

Category Uncertainty Quantification MethodSuitable for
long-
formshort-
form
Information-
basedMax Claim/Sequence Probability ✓ ✓
CCP (Fadeeva et al., 2024) ✓ ✓
Mean/Max Token Entropy
(Fomicheva et al., 2020)✓ ✓
Perplexity (Fomicheva et al., 2020) ✓ ✓
TokenSAR (Duan et al., 2024) ✓
Sample
diversitySemantic Entropy (Kuhn et al., 2023) ✓
SentenceSAR (Duan et al., 2024) ✓
Lexical Similarity (Fomicheva et al., 2020) ✓
Degree Matrix (Lin et al., 2024) ✓
Eccentricity (Lin et al., 2024) ✓
Sum of Eigenvalues (Lin et al., 2024) ✓
Number of Semantic Sets (Lin et al., 2024) ✓
Reflexive P(True) (Kadavath et al., 2022) ✓ ✓
Table 1: Summary of general baseline methods.
ability of a claim caccording to the LLM:
p(c|x,r) =Y
t∈S(c)p(yt|x,r,y<t). (1)
We select methods available in LM-Polygraph li-
brary (Fadeeva et al., 2023) and summarize them
in Table 1. More methods support short-form QA
since the entire answer is assessed as a single claim,
allowing the use of sampling-based techniques. For
long-form QA, there are fewer applicable methods,
as they must operate at the token level to aggregate
information over claim spans.
2.1 Faithfulness-Based Retrieval Augmented
uNcertainty Quantification ( FRANQ )
We introduce FRANQ , a new approach for assessing
the factuality of claims in RAG outputs by lever-
aging UQ techniques and explicitly treating xand
ras separate inputs. The key idea is to first as-
sess whether a generated claim is faithful to rand
then apply different UQ strategies depending on
the outcome. The overall factuality of a claim cis
estimated as:
P(cis true ) = (2)
P(cis faithful to r)·P(cis true |faithful )+
P(cis unfaithful to r)·P(cis true |unfaithful ).
Here, we calculate P(cis unfaithful to r)as1−
P(cis faithful to r). This decomposition inte-
grates three probability components, each instanti-
ated using a suitable UQ method:
1.P(cis faithful to r);
2.P(cis true |faithful );
3.P(cis true |unfaithful ).
An overview of our approach is visually depicted
in Figure 1. Detailed examples of FRANQ applied
to specific claims are given in Appendix E.2.2 FRANQ Components
We now describe the three components used in
equation (2).
Faithfulness. To assess whether a claim ciis
entailed by the retrieved evidence r, we use
AlignScore , a semantic similarity metric based on
RoBERTa fine-tuned for factual alignment (Zha
et al., 2023). AlignScore evaluates the consistency
between a claim and its supporting context. A
comparison with alternative methods is provided
in Appendix C.1.
In long-form QA, we compare the claim ci
with the concatenated retrieved documents rus-
ing AlignScore. For short-form QA, where the
full answer yis treated as a single claim, we pro-
vide question context by calculating AlignScore
on pairs of the form (x◦y, rj), with ‘ ◦’ denoting
string concatenation.
Factuality under unfaithful condition. When a
claim cis unfaithful (i.e., not entailed by r), it likely
originates from the LLM’s internal knowledge. We
estimate its factuality using the model’s own prob-
ability estimates. Specifically, we introduce the
Parametric Knowledge UQ method, which com-
putes the likelihood of cbased solely on the LLM’s
parametric knowledge (Mallen et al., 2023) without
providing retrieved evidence r:
p(c|x) =Y
t∈S(c)p(yt|x,y<t). (3)
This approach is effective in the long-form QA
setting (see Appendix C.2). However, for short-
form QA, more alternative general UQ baselines
are available, such as the ones employing sample
diversity. We find that Sum of Eigenvalues (Lin
et al., 2024) better approximates factuality in this
scenario (see Appendix C.2).
Therefore, we estimate the factuality of unfaith-
ful claims using Parametric Knowledge in long-
form QA and using Semantic Similarity in short-
form QA.
Factuality under faithful condition. If a claim
cis evaluated as faithful, we can additionally as-
sess whether it is entailed by the retrievals using
a method different from AlignScore. To this end,
we introduce the MaxNLI UQ method, which uses
a pre-trained Natural Language Inference (NLI)
model (He et al., 2023) to assess whether a claim
cis entailed (‘e’) or contradicted (‘c’) by any of
the retrieved passages. We report the maximum
3

entailment confidence across all retrievals:
MaxNLI (c) = max
j=1,kP(NLI(c, rj) =‘e’)
P(NLI(c, rj)∈ {‘e’,‘c’}),
(4)
where probabilities are taken from the NLI model’s
predicted class distribution. As with AlignScore,
for long-form QA, we apply the NLI model to
each individual claim, and for short-form QA, the
premise is formed by concatenating the question x
with the answer y.
We later find that MaxNLI performs well for
long-form QA, making it a reasonable choice for
estimating factuality under the faithful condition,
however, the specific choice of method for this
component has limited impact on overall FRANQ
performance (see Appendix D.1).
Again, for short-form QA, we find alternative
baselines more suitable, particularly Semantic En-
tropy (Kuhn et al., 2023), which better captures
uncertainty in this scenario (see Appendix C.2).
Therefore, we estimate the factuality for faith-
ful claims with MaxNLI for long-form QA, and
Semantic Entropy for short-form QA.
Resulting formula. In summary, we estimate the
factuality of the claim cwith FRANQ using the
following formula:
FRANQ (c) =Pfaithful (c,r)·UQfaith(c) (5)
+ 
1−Pfaithful (c,r)
·UQunfaith (c),
where we use AlignScore to estimate faithfulness
probability Pfaithful and two UQ methods, UQfaith
andUQunfaith , selected based on empirical perfor-
mance for long-form and short-form scenarios. For
long-form QA, we use MaxNLI (4)forUQfaithand
Parametric Knowledge (3)forUQunfaith . For short-
form QA, we use Semantic Entropy (Kuhn et al.,
2023) for UQfaithand Sum of Eigenvalues (Lin
et al., 2024) for UQunfaith .
2.3 Calibrating FRANQ
Since the UQ methods UQfaithand UQunfaith of
equation (5)are not inherently probabilistic and
may have different distributions, to avoid incon-
sistencies and miscalibration among various UQ
measures, we calibrate their outputs using isotonic
regression on training data (Vashurin et al., 2025).
Formally, given a training dataset D=
{(ui,facti)}N
i=1comprising pairs of UQ scores ui
and corresponding binary factuality labels factifor
Nclaims, we calibrate UQ scores by fitting a non-
decreasing function f:R→[0,1]through isotonicregression, minimizing the squared error:
bf= arg min
f∈FXN
i=1 
f(ui)−facti2,(6)
whereFdenotes the set of all non-decreasing func-
tions mapping real numbers to probabilities in the
interval [0,1]. During inference, the calibration
function bfis applied to each UQ score to produce
probabilistically meaningful output.
Since UQfaithandUQunfaith represent probabil-
ities under faithful and unfaithful conditions, re-
spectively, we propose condition-specific calibra-
tion. This involves partitioning the training dataset
Dinto two subsets: faithful claims Dfaithand un-
faithful claims Dunfaith . Then, we calibrate UQfaith
using the faithful subset DfaithandUQunfaith using
the unfaithful subset Dunfaith .
We consider FRANQ with condition-specific cal-
ibration as our primary method. To evaluate the
impact of calibration, we additionally assess two
variants: one without any calibration, and another
in which both UQ methods are calibrated using the
full training dataset D. The calibration strategies
are summarized as follows:
1.No calibration. Raw outputs from UQfaith
andUQunfaith are directly used in equation (5)
without any calibration.
2.Calibrated scores. Both UQ methods are
calibrated on the entire training dataset D, dis-
regarding claim faithfulness.
3.Condition-calibrated scores. Each UQ
method is calibrated using a subset of the train-
ing data corresponding to the respective con-
dition: UQfaithis calibrated using Dfaith, and
UQunfaith is calibrated using Dunfaith .
3 Datasets for RAG Uncertainty
Quantification
Existing datasets for studying RAG hallucinations
have serious limitations (see discussion in Sec-
tion 5.2). We argue that an effective dataset should
reflect generation behavior in the presence of re-
trieval, capturing both factual errors and contextual
misuse. To this end, we introduce a new dataset
specifically designed for long-form generations, en-
abling fine-grained analysis of atomic claims.
Overall, we consider two QA settings: long-
form , where questions require comprehensive an-
swers composed of multiple claims; and short-
form, where questions are answered concisely with
4

single claims. For both QA settings, we employ
Llama-3.2-3B-Instruct (Grattafiori et al., 2024)
andFalcon3-3B-Base (Team, 2024) as the LLMs.
Long-form QA dataset. Generating high-quality
data for RAG hallucination detection remains chal-
lenging. Most benchmarks focus on faithfulness
(i.e., entailment with retrieved content) rather than
factuality. However, we argue that factuality is
more critical in RAG applications, with faithful-
ness serving as a complementary perspective. Since
automatic annotation lacks precision and manual
annotation is costly, we adopt a hybrid approach to
ensure data quality.
Our long-form QA dataset consists of 76 ques-
tions: 44 from RAGTruth (Niu et al., 2024) filtered
to include questions where Llama 3B provides at
least two false claims, and 32 technical questions
generated via GPT-4 (e.g., “How does solar power
generate electricity?” ).
We extract claims and their token spans from
model outputs using methods introduced in (Wang
et al., 2024b; Vashurin et al., 2025). First, GPT-4o
extracts decontextualized atomic claims from the
entire text paragraph through a dedicated prompt.
Then, for each claim, a second prompt instructs
GPT-4o to list the relevant words from the original
text, which we map to token spans. Applying this
procedure, we obtain 1,782 claims for Llama 3B
and 1,548 claims for Falcon 3B. From these claims,
we select 500 claims for training, reserving the
remainder for testing.
We annotate factuality and faithfulness using
GPT-4o-search and manually verify claims classi-
fied as incorrect. For faithfulness annotation, each
claim is automatically assigned one of three cat-
egories: faithful ,unfaithful-contra , orunfaithful-
neutral . These categories are binarized for evalu-
ation: faithful is mapped to 1, and both unfaithful
categories to 0, as unfaithful-contra comprises less
than 5% of the data. Similarly, factuality annota-
tions include True,False , and unverifiable . Only
verifiable claims (True or False) are retained, with
labels binarized. Further details on prompts and
dataset statistics are available in Appendix B.
Short-form QA datasets. For short-form QA, we
evaluate FRANQ on four widely used QA datasets:
TriviaQA (Joshi et al., 2017), SimpleQA (Wei
et al., 2024), Natural Questions (Kwiatkowski et al.,
2019), and PopQA (Mallen et al., 2023). The out-
put is treated as a single claim. From each dataset,
we sample 200 questions for training and 1,000 for
testing.We evaluate factuality by comparing RAG out-
puts with gold-standard answers using GPT-4, fol-
lowing the approach in (Wei et al., 2024).
RAG models. For each question, we select top-
kpassages with Facebook/contriever-msmarco
retriever and Wikipedia embeddings based on the
2018 English Wikipedia, to ensure accurate re-
trievals. For long-form QA, we set k= 3, and
for short-form QA, k= 5.
4 Experiments
For all the experiments, we fix both the retrieval
process and the white-box LLM, verifying the fac-
tual accuracy of the LLM-generated claims.
We test FRANQ with 3 calibration strategies (see
Section 2.3). First, we compare it with the un-
supervised baselines introduced earlier. While the
uncalibrated version of FRANQ is unsupervised, the
calibrated versions require training labels. There-
fore, we further compare our method against the
supervised XGBoost baseline (Kuleshov and Liang,
2015) trained using the outputs of unsupervised UQ
methods as features.
Furthermore, in ablation studies, we investi-
gate the impact of different FRANQ components
P(faithful ),UQfaith, and UQunfaith , as well as vary-
ing training sizes.
4.1 Experimental Setup
UQ baselines. We group all UQ methods into 4
categories: (1) general baselines, (2) RAG-specific
baselines, (3) XGBoost-based methods, and (4)
three variants of proposed FRANQ method, each
employing a different calibration strategy.
General baselines. To implement baselines,
we use the LM-Polygraph library (Fadeeva et al.,
2023). For short-form QA, we use 13 sequence-
level methods; for long-form QA, we use 5 claim-
level methods. A complete list of these methods is
provided in Table 1. We also evaluate three base-
line methods previously introduced as components
ofFRANQ : AlignScore, MaxNLI, and Parametric
Knowledge (see Section 2.2).
XGBoost methods. We include XGBoost models
trained on factuality labels using two feature sets:
(1) three components used in FRANQ (AlignScore,
UQfaith,UQunfaith ), and (2) all available unsuper-
vised UQ methods (for long-form QA: 5 general +
3 RAG-specific baselines; for short-form QA: 13
general + 3 RAG-specific baselines).
FRANQ. Finally, we evaluate three FRANQ vari-
5

MethodLlama 3b Falcon 3b
ROC-AUC ↑PR-AUC ↑PRR↑ ROC-AUC ↑PR-AUC ↑PRR↑
Max Claim Probability .497 .058 -.029 .678 .126 .258
P(True) .573 .117 .207 .591 .077 .170
Perplexity .477 .056 -.081 .636 .090 .165
Mean Token Entropy .562 .109 .115 .646 .130 .219
CCP .587 .085 .169 .641 .162 .181
MaxNLI .642 .109 .151 .669 .101 .287
AlignScore .616 .075 .108 .652 .104 .233
Parametric Knowledge .523 .064 .018 .536 .067 .029
XGBoost (all UQ features) .611 .124 .206 .616 .088 .198
XGBoost ( FRANQ features) .576 .111 .149 .550 .080 .086
FRANQ no calibration .646 .100 .181 .692 .135 .362
FRANQ calibrated .653 .103 .256 .602 .074 .090
FRANQ condition-calibrated .641 .140 .223 .693 .173 .354
Table 2: Results on long-form QA benchmark with factuality target. Higher values indicate better performance. In
every setting, the top-performing method is one of the FRANQ variants.
ants with different calibration strategies for UQfaith
andUQunfaith (see Section 2.3): “no calibration”,
“calibrated”, and “condition-calibrated”.
Metrics. Each UQ method provides factuality
estimates, which we compare with binary gold-
standard factuality labels using ROC-AUC and PR-
AUC (false claims are designated as the positive
class to emphasize the method’s effectiveness in
identifying them). To further check rejection per-
formance, we evaluate Prediction Rejection Ratio
(PRR; Mallen et al. (2023)) with a maximum re-
jection threshold of 0.5. It quantifies how well the
model can reject uncertain predictions while retain-
ing accurate ones, providing a direct measure of the
model’s ability to identify and prioritize reliable
outputs.
4.2 Long-Form QA Results
For long-form QA, we evaluate each UQ method
using three metrics: ROC-AUC, PR-AUC, and
PRR, across two models: Llama 3B and Falcon
3B. These results are presented in Table 2.
For Llama 3B, the calibrated version of FRANQ
achieves the highest ROC-AUC and PRR, indicat-
ing strong overall performance. The condition-
calibrated version of FRANQ yields the best PR-
AUC and ranks second in PRR. For Falcon 3B, the
condition-calibrated FRANQ variant performs best
overall, with the highest ROC-AUC and PR-AUC,
while the non-calibrated FRANQ achieves the top
PRR and second-best ROC-AUC.
Note that, except for FRANQ calibrated on Fal-
con 3B, the proposed FRANQ method and its two
calibration extensions achieve leading results on
both models. In the case of FRANQ calibratedon Falcon 3B, one hypothesis is that Parametric
Knowledge calibration over the entire dataset de-
grades its performance. This may be because Para-
metric Knowledge on Falcon 3B data does not ap-
proximate faithfulness well enough, leading iso-
tonic regression calibration to further degrade the
final output.
4.3 Short-Form QA Results
For short-form QA, we compute three metrics
(ROC-AUC, PR-AUC, and PRR) for each UQ
method across two models (Llama 3B and Falcon
3B) and four QA datasets. For each metric and
dataset, we also determine the relative performance
rank of each UQ method (1st best, 2nd best, etc.).
Since these detailed results vary across datasets,
we report only the mean metric values and mean
ranks (averaged over the four datasets) to provide
a clearer and more concise summary, following the
approach of Vashurin et al. (2025). These sum-
mary results are shown in Table 3. Full per-dataset
results are provided in Appendix A.
For Llama 3B, the condition-calibrated FRANQ
version outperforms all baselines, achieving the
best mean rank and mean value across all three
metrics. The calibrated FRANQ variant ranks sec-
ond, showing the top mean rank for ROC-AUC
and second-best for PRR, as well as second-best
mean value for both ROC-AUC and PRR. Among
unsupervised methods, DegMat shows strong per-
formance, ranking second in PR-AUC under both
mean rank and mean value.
For Falcon 3B, the condition-calibrated FRANQ
method again leads overall, achieving the top mean
rank across all metrics, and the best PR-AUC and
6

MethodLlama-3b Falcon-3b
MeanRank ↓ MeanValue ↑ MeanRank ↓ MeanValue ↑
ROC-AUC PR-AUC PRR ROC-AUC PR-AUC PRR ROC-AUC PR-AUC PRR ROC-AUC PR-AUC PRR
Max Sequence Probability 10.50 12.50 11.50 .758 .558 .454 14.75 12.50 14.00 .617 .628 .256
CCP 13.00 13.00 13.00 .745 .551 .443 11.75 10.50 10.50 .637 .641 .304
Max Token Entropy 7.25 6.75 8.00 .774 .594 .481 13.75 14.25 15.00 .617 .613 .242
P(True) 20.25 20.00 20.00 .477 .292 -.011 20.00 20.00 20.00 .515 .529 .042
Lexical Similarity 8.50 10.50 8.25 .769 .564 .479 13.00 14.00 13.50 .635 .618 .277
Degree Matrix 6.25 3.75 5.75 .789 .629 .520 3.00 3.00 3.25 .724 .702 .464
Eccentricity 6.00 5.00 6.50 .787 .615 .515 6.25 8.75 7.25 .702 .662 .406
Sum of Eigenvalues 5.25 4.25 5.50 .791 .628 .518 3.00 3.50 3.75 .723 .700 .460
Number of Semantic Sets 17.00 16.50 16.75 .662 .472 .312 9.25 10.25 8.25 .675 .646 .382
Semantic Entropy 4.00 5.25 4.75 .792 .613 .525 13.25 13.25 13.25 .637 .623 .278
Perplexity 12.00 10.75 11.75 .745 .558 .432 16.50 15.75 16.25 .603 .603 .225
TokenSAR 13.75 13.50 13.75 .735 .540 .413 14.25 13.25 14.50 .616 .615 .243
SentenceSAR 9.00 10.25 9.00 .773 .571 .483 13.75 15.00 14.25 .631 .602 .263
MaxNLI 20.25 20.25 20.25 .466 .303 -.036 20.00 19.00 20.00 .514 .532 .035
AlignScore 17.00 18.25 17.50 .631 .415 .207 7.50 7.50 7.25 .682 .666 .372
Parametric Knowledge 18.00 18.00 17.75 .627 .425 .247 18.25 19.25 18.75 .561 .556 .104
XGBoost (all UQ features) 9.50 8.50 7.25 .766 .594 .494 3.00 3.25 3.75 .733 .705 .462
XGBoost ( FRANQ features) 13.75 14.75 14.00 .723 .526 .409 9.25 7.25 8.00 .672 .670 .368
FRANQ no calibration 14.25 12.25 14.75 .728 .553 .403 11.00 10.00 10.75 .664 .641 .345
FRANQ calibrated 2.75 4.00 2.75 .797 .628 .537 7.00 7.75 6.00 .696 .672 .411
FRANQ condition-calibrated 2.75 3.00 2.25 .802 .631 .541 2.50 3.00 2.75 .730 .711 .477
Table 3: Results averaged across four QA datasets for Llama 3B and Falcon 3B. For each model, the first three
columns show the mean rank across all 4 datasets, and the last three columns report the mean metric values. Lower
mean ranks and higher mean values indicate better performance. The condition-calibrated FRANQ is top-performing
across all settings, except mean ROC-AUC on Falcon 3B, where it ranks second.
PRR in mean value, while ranking second in ROC-
AUC. XGBoost, trained on all UQ features, ranks
second overall, placing second in ROC-AUC and
PR-AUC under mean rank, and first in ROC-AUC
under mean value. DegMat remains competitive
for Falcon 3B as well, with the best PR-AUC and
second-best ROC-AUC and PRR in mean rank, and
second-best PRR in mean value.
4.4 Ablation Studies
We conducted ablation studies to evaluate the con-
tributions of individual FRANQ components, the
effect of training set size on supervised variants,
and the calibration quality of the tested UQ meth-
ods. This section summarizes the key findings, and
a more detailed analysis is provided in Appendix D.
Analysis of FRANQ components. We begin by
analyzing different combinations of UQfaithand
UQunfaith within condition-calibrated FRANQ (see
Figure 10 in in Appendix D). For long-form QA
with Llama 3B, performance is highly sensitive
to the choice of UQunfaith , with Parametric Knowl-
edge emerging as the most effective option. In
contrast, UQunfaith has a relatively minor impact.
For short-form QA with Llama 3B, FRANQ is ro-
bust across multiple UQ configurations, suggesting
greater flexibility in component selection.
Next, we evaluate alternative faithfulness estima-
tion strategies (see Tables 12 and 13). Substituting
AlignScore probabilities with binary thresholdingconsistently degrades performance, underscoring
the benefit of using continuous faithfulness sig-
nals. Surprisingly, calibrating AlignScore with
gold-standard labels also reduces performance, in-
dicating that the raw AlignScore values are more
effective when used directly within FRANQ .
Impact of the training set size. We also examine
the effect of the training set size on both FRANQ
and XGBoost variants (see Figure 11). The un-
supervised FRANQ baseline remains stable, while
supervised versions with calibration generally im-
prove with more data. In long-form QA, condition-
calibrated FRANQ peaks at 300 training instances
before slightly declining; for short-form QA, per-
formance saturates around 120 instances. Across
all training sizes, supervised FRANQ methods con-
sistently outperform XGBoost ones.
XGBoost approximates the decision-making be-
havior of FRANQ .To better understand XGBoost’s
behavior, we analyze its first decision tree trained
onFRANQ features (see Figure 12 in Appendix D).
We find that the tree mimics FRANQ ’s decision
logic: initially splitting on AlignScore, followed
by MaxNLI or Parametric Knowledge depending
on the score. This suggests that supervised meth-
ods like XGBoost learn decision patterns closely
aligned with those encoded in FRANQ .
Calibration evaluation. Finally, we assess the cal-
ibration of all UQ methods using Expected Calibra-
tion Error (ECE; Guo et al. (2017)); see Table 14
7

in Appendix D. As expected, the two calibrated
FRANQ variants achieve the lowest ECE scores
for both long- and short-form QA, indicating that
they produce well-calibrated confidence estimates
aligned with empirical accuracy.
5 Related Work
5.1 Uncertainty Quantification for RAG
Several uncertainty quantification methods have
been proposed for RAG that take into account the
influence of retrieved knowledge on the LLM out-
put. Chuang et al. (2024) detect hallucinations with
a linear classifier that leverages the lookback ratio,
the ratio of attention weights on the context ver-
sus newly generated tokens. The method of Sun
et al. (2025) uses regression on features measuring
the model’s reliance on external extracted knowl-
edge (via attention scores) and internal parametric
knowledge (from its own weights). Another UQ
method for RAG gets an uncertainty score that
estimates the similarity between text fragments
through the signal-to-noise ratio of the output prob-
abilities of the model sample (Li et al., 2024). The
hallucination detection method for RAG proposed
by Hu et al. (2024) estimates the correlation be-
tween the model’s input and output by computing
a relevance matrix between the prompt (including
retrieved context) and the response, and then it is
classified as hallucinations.
All of the aforementioned methods identify hal-
lucinations with respect to the retrieved context, but
do not assess them in relation to world knowledge,
which represents a notable shortcoming. Moreover,
these approaches incur additional computational
overhead due to the need for an auxiliary model
and its associated training process.
When RAG does not incorporate retrieved con-
text during generation, various methods estimate
uncertainty solely based on the model’s internal
knowledge, without accounting for retrieval-related
uncertainty. These include both white-box ap-
proaches (Fomicheva et al., 2020; Kadavath et al.,
2022; Kuhn et al., 2023; Fadeeva et al., 2024; Duan
et al., 2024) and black-box approaches (Fomicheva
et al., 2020; Lin et al., 2024).
5.2 Hallucination and Factuality Datasets for
RAG
The task of hallucination detection in RAG requires
labeled data that captures model hallucinations
in the form of corresponding class annotations.RAGTruth (Niu et al., 2024) is a multi-domain
dataset for RAG hallucination detection, featuring
manually annotated and span-level classified hallu-
cinations. However, the hallucination classes in this
dataset do not account for cases where the model
generates knowledge independently of the retrieved
context and is then verified separately for halluci-
nations, as such responses may still be factually
correct and useful to the user.
Knowledge-based dialogue datasets bear some
similarity to RAG datasets, as they also include
an external knowledge base to support the conver-
sation. Wizard of Wikipedia (Dinan et al., 2019)
and FaithDial (Dziri et al., 2022), for example, are
structured as user–model dialogues grounded in
Wikipedia content. Unlike RAG datasets, however,
these datasets place greater emphasis on maintain-
ing coherent dialogue and conversational consis-
tency. Moreover, as in the RAGTruth dataset, any
information not directly grounded in the provided
context is automatically treated as hallucination,
regardless of its factual accuracy.
Hallucination benchmarks for RAG are often
constructed using QA datasets, where relevant con-
text is provided alongside the question (Friel et al.,
2024; Moskvoretskii et al., 2025). In these datasets,
hallucinations are defined strictly with respect to
the provided context.
6 Conclusion and Future Work
In this paper, we introduce FRANQ (Faithfulness-
based Retrieval Augmented UNcertainty Quantifi-
cation), a new method that quantifies the factuality
of claims in RAG output based on their faithfulness.
Evaluated on both long-form and short-form QA
tasks across multiple LLMs, FRANQ outperforms
existing unsupervised UQ baselines, RAG-specific
methods, and supervised classifiers. We also in-
troduce a new long-form QA dataset annotated for
both factuality and faithfulness using a hybrid of
automatic and manual labeling.
Our approach opens several promising directions
for future research. One is extending uncertainty
modeling to the retrieval stage itself, allowing sys-
tems to account for noisy, incomplete, or conflict-
ing evidence. Future work could also explore how
FRANQ ’s uncertainty signals can guide generation-
time control or post-editing, enabling more reliable
and interpretable RAG systems.
8

Limitations
While FRANQ provides strong hallucination detec-
tion performance on average, it does not guarantee
ideal hallucination detection in every situation, as
it is a challenging task.
The method assumes that retrieved evidence
is always factual and takes precedence over the
LLM’s parametric knowledge. In theory, this can
be achieved through careful selection and curation
of document sources within the search index. How-
ever, ensuring complete factual accuracy in real-
world applications might be challenging as the size
of the index grows.
Since FRANQ leverages the calibration of its
components, it might be considered as supervised.
To address this concern, we show that this method
also outperforms supervised methods.
Ethical Considerations
FRANQ is designed to reduce the spread of factual
errors by enhancing the interpretability and reliabil-
ity of language model outputs. By distinguishing
between factuality and faithfulness, it helps pre-
vent misclassification of factually correct but un-
supported claims. However, the system does not
actively prevent the generation of hallucinations
and instead relies on downstream filtering. Its ef-
fectiveness, therefore, depends on integration into
larger pipelines with proper safeguards.
FRANQ assumes the retrieved context is factual
and trustworthy. In real-world applications, re-
trieved documents may be biased, outdated, or
incorrect, which could compromise the method’s
output. Careful curation of retrieval sources and
monitoring of retrieval quality are crucial to avoid
reinforcing harmful biases or misinformation.
The dataset used in evaluation relies on outputs
from GPT-4o. While we manually validate a subset
of annotations, some inherent biases from the un-
derlying model may persist. We encourage future
work to explore more diverse annotation strategies,
including community-sourced validation.
Improving factuality estimation can support
safer AI deployment, especially in knowledge-
intensive domains such as education, healthcare,
or law. However, the system should not be consid-
ered a replacement for human fact-checkers. It is
best used as a decision-support tool rather than a
source of truth.References
Yung-Sung Chuang, Linlu Qiu, Cheng-Yu Hsieh, Ran-
jay Krishna, Yoon Kim, and James R. Glass. 2024.
Lookback lens: Detecting and mitigating contex-
tual hallucinations in large language models us-
ing only attention maps. In Proceedings ofthe
2024 Conference onEmpirical Methods inNatural
Language Processing , pages 1419–1436. Association
for Computational Linguistics.
Emily Dinan, Stephen Roller, Kurt Shuster, Angela
Fan, Michael Auli, and Jason Weston. 2019. Wizard
of wikipedia: Knowledge-powered conversational
agents. In International Conference onLearning
Representations.
Hanxing Ding, Liang Pang, Zihao Wei, Huawei Shen,
and Xueqi Cheng. 2024. Retrieve only when it needs:
Adaptive retrieval augmentation for hallucination mit-
igation in large language models. arXiv preprint
arXiv:2402.10612.
Jinhao Duan, Hao Cheng, Shiqi Wang, Alex Zavalny,
Chenan Wang, Renjing Xu, Bhavya Kailkhura, and
Kaidi Xu. 2024. Shifting attention to relevance: To-
wards the predictive uncertainty quantification of
free-form large language models. In Proceedings
ofthe62nd Annual Meeting oftheAssociation
forComputational Linguistics (V olume 1:Long
Papers) , pages 5050–5063. Association for Compu-
tational Linguistics.
Nouha Dziri, Ehsan Kamalloo, Sivan Milton, Osmar Za-
iane, Mo Yu, Edoardo M Ponti, and Siva Reddy. 2022.
FaithDial: A Faithful Benchmark for Information-
Seeking Dialogue. Transactions oftheAssociation
forComputational Linguistics, 10:1473–1490.
Ekaterina Fadeeva, Aleksandr Rubashevskii, Artem
Shelmanov, Sergey Petrakov, Haonan Li, Hamdy
Mubarak, Evgenii Tsymbalov, Gleb Kuzmin, Alexan-
der Panchenko, Timothy Baldwin, Preslav Nakov,
and Maxim Panov. 2024. Fact-checking the output
of large language models via token-level uncertainty
quantification. In Findings oftheAssociation for
Computational Linguistics: ACL 2024, pages 9367–
9385. Association for Computational Linguistics.
Ekaterina Fadeeva, Roman Vashurin, Akim Tsvi-
gun, Artem Vazhentsev, Sergey Petrakov, Kirill
Fedyanin, Daniil Vasilev, Elizaveta Goncharova,
Alexander Panchenko, Maxim Panov, Timothy Bald-
win, and Artem Shelmanov. 2023. LM-polygraph:
Uncertainty estimation for language models. In
Proceedings ofthe2023 Conference onEmpirical
Methods inNatural Language Processing: System
Demonstrations , pages 446–461. Association for
Computational Linguistics.
Marina Fomicheva, Shuo Sun, Lisa Yankovskaya,
Frédéric Blain, Francisco Guzmán, Mark Fishel,
Nikolaos Aletras, Vishrav Chaudhary, and Lucia Spe-
cia. 2020. Unsupervised quality estimation for neural
machine translation. Transactions oftheAssociation
forComputational Linguistics, 8:539–555.
9

Robert Friel, Masha Belyi, and Atindriyo Sanyal. 2024.
Ragbench: Explainable benchmark for retrieval-
augmented generation systems. arXiv preprint
arXiv:2407.11005.
Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri,
Abhinav Pandey, Abhishek Kadian, Ahmad Al-
Dahle, Aiesha Letman, Akhil Mathur, Alan Schel-
ten, Alex Vaughan, Amy Yang, Angela Fan, Anirudh
Goyal, Anthony Hartshorn, Aobo Yang, Archi Mi-
tra, Archie Sravankumar, Artem Korenev, Arthur
Hinsvark, and 542 others. 2024. The llama 3 herd of
models. Preprint, arXiv:2407.21783.
Chuan Guo, Geoff Pleiss, Yu Sun, and Kilian Q. Wein-
berger. 2017. On calibration of modern neural net-
works. In Proceedings ofthe34th International
Conference onMachine Learning , volume 70 of
Proceedings ofMachine Learning Research , pages
1321–1330. PMLR.
Pengcheng He, Jianfeng Gao, and Weizhu Chen. 2023.
Debertav3: Improving deberta using electra-style pre-
training with gradient-disentangled embedding shar-
ing. In The Eleventh International Conference on
Learning Representations.
Haichuan Hu, Yuhan Sun, and Quanjun Zhang. 2024.
Lrp4rag: Detecting hallucinations in retrieval-
augmented generation via layer-wise relevance prop-
agation. arXiv preprint arXiv:2408.15533.
Lei Huang, Weijiang Yu, Weitao Ma, Weihong Zhong,
Zhangyin Feng, Haotian Wang, Qianglong Chen,
Weihua Peng, Xiaocheng Feng, Bing Qin, and 1 oth-
ers. 2025. A survey on hallucination in large lan-
guage models: Principles, taxonomy, challenges, and
open questions. ACM Transactions onInformation
Systems, 43(2):1–55.
Mandar Joshi, Eunsol Choi, Daniel Weld, and Luke
Zettlemoyer. 2017. TriviaQA: A large scale distantly
supervised challenge dataset for reading comprehen-
sion. In Proceedings ofthe55th Annual Meeting
oftheAssociation forComputational Linguistics
(V olume 1:Long Papers) , pages 1601–1611. Associ-
ation for Computational Linguistics.
Saurav Kadavath, Tom Conerly, Amanda Askell, Tom
Henighan, Dawn Drain, Ethan Perez, Nicholas
Schiefer, Zac Hatfield-Dodds, Nova DasSarma, Eli
Tran-Johnson, and 1 others. 2022. Language mod-
els (mostly) know what they know. arXiv preprint
arXiv:2207.05221.
Hyuhng Joon Kim, Youna Kim, Sang-goo Lee, and
Taeuk Kim. 2024. When to speak, when to abstain:
Contrastive decoding with abstention. arXiv preprint
arXiv:2412.12527.
Lorenz Kuhn, Yarin Gal, and Sebastian Farquhar. 2023.
Semantic uncertainty: Linguistic invariances for
uncertainty estimation in natural language genera-
tion. In The Eleventh International Conference on
Learning Representations.V olodymyr Kuleshov and Percy S Liang. 2015. Cali-
brated structured prediction. In Advances inNeural
Information Processing Systems , volume 28. Curran
Associates, Inc.
Tom Kwiatkowski, Jennimaria Palomaki, Olivia Red-
field, Michael Collins, Ankur Parikh, Chris Alberti,
Danielle Epstein, Illia Polosukhin, Jacob Devlin, Ken-
ton Lee, Kristina Toutanova, Llion Jones, Matthew
Kelcey, Ming-Wei Chang, Andrew M. Dai, Jakob
Uszkoreit, Quoc Le, and Slav Petrov. 2019. Nat-
ural questions: A benchmark for question answer-
ing research. Transactions oftheAssociation for
Computational Linguistics, 7:452–466.
Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio
Petroni, Vladimir Karpukhin, Naman Goyal, Hein-
rich Küttler, Mike Lewis, Wen-tau Yih, Tim Rock-
täschel, and 1 others. 2020. Retrieval-augmented gen-
eration for knowledge-intensive nlp tasks. Advances
inneural information processing systems , 33:9459–
9474.
Zixuan Li, Jing Xiong, Fanghua Ye, Chuanyang
Zheng, Xun Wu, Jianqiao Lu, Zhongwei Wan, Xi-
aodan Liang, Chengming Li, Zhenan Sun, and
1 others. 2024. Uncertaintyrag: Span-level
uncertainty enhanced long-context modeling for
retrieval-augmented generation. arXiv preprint
arXiv:2410.02719.
Zhen Lin, Shubhendu Trivedi, and Jimeng Sun. 2024.
Generating with confidence: Uncertainty quan-
tification for black-box large language models.
Transactions onMachine Learning Research.
Alex Mallen, Akari Asai, Victor Zhong, Rajarshi Das,
Daniel Khashabi, and Hannaneh Hajishirzi. 2023.
When not to trust language models: Investigating
effectiveness of parametric and non-parametric mem-
ories. In Proceedings ofthe61st Annual Meeting
oftheAssociation forComputational Linguistics
(V olume 1:Long Papers) , pages 9802–9822. Associ-
ation for Computational Linguistics.
Viktor Moskvoretskii, Maria Lysyuk, Mikhail Sal-
nikov, Nikolay Ivanov, Sergey Pletenev, Daria Gal-
imzianova, Nikita Krayko, Vasily Konovalov, Irina
Nikishina, and Alexander Panchenko. 2025. Adap-
tive retrieval without self-knowledge? bringing uncer-
tainty back home. arXiv preprint arXiv:2501.12835.
Cheng Niu, Yuanhao Wu, Juno Zhu, Siliang Xu, Kashun
Shum, Randy Zhong, Juntong Song, and Tong Zhang.
2024. Ragtruth: A hallucination corpus for develop-
ing trustworthy retrieval-augmented language models.
InProceedings ofthe62nd Annual Meeting ofthe
Association forComputational Linguistics (V olume
1:Long Papers), pages 10862–10878.
Freda Shi, Xinyun Chen, Kanishka Misra, Nathan
Scales, David Dohan, Ed H Chi, Nathanael Schärli,
and Denny Zhou. 2023. Large language mod-
els can be easily distracted by irrelevant context.
InInternational Conference onMachine Learning ,
pages 31210–31227. PMLR.
10

Kurt Shuster, Spencer Poff, Moya Chen, Douwe Kiela,
and Jason Weston. 2021. Retrieval augmentation
reduces hallucination in conversation. In Findings
oftheAssociation forComputational Linguistics:
EMNLP 2021 , pages 3784–3803. Association for
Computational Linguistics.
ZhongXiang Sun, Xiaoxue Zang, Kai Zheng, Jun Xu,
Xiao Zhang, Weijie Yu, Yang Song, and Han Li.
2025. RedeEP: Detecting hallucination in retrieval-
augmented generation via mechanistic interpretabil-
ity. In The Thirteenth International Conference on
Learning Representations.
Falcon-LLM Team. 2024. The falcon 3 family of open
models.
Roman Vashurin, Ekaterina Fadeeva, Artem Vazhentsev,
Lyudmila Rvanova, Akim Tsvigun, Daniil Vasilev,
Rui Xing, Abdelrahman Boda Sadallah, Kirill Gr-
ishchenkov, Sergey Petrakov, Alexander Panchenko,
Timothy Baldwin, Preslav Nakov, Maxim Panov, and
Artem Shelmanov. 2025. Benchmarking uncertainty
quantification methods for large language models
with lm-polygraph. Transactions oftheAssociation
forComputational Linguistics, 13:220–248.
Fei Wang, Xingchen Wan, Ruoxi Sun, Jiefeng Chen,
and Sercan Ö Arık. 2024a. Astute rag: Overcom-
ing imperfect retrieval augmentation and knowledge
conflicts for large language models. arXiv preprint
arXiv:2410.07176.
Han Wang, Archiki Prasad, Elias Stengel-Eskin, and
Mohit Bansal. 2025. Retrieval-augmented gener-
ation with conflicting evidence. arXiv preprint
arXiv:2504.13079.
Yuxia Wang, Revanth Gangi Reddy, Zain Muhammad
Mujahid, Arnav Arora, Aleksandr Rubashevskii, Ji-
ahui Geng, Osama Mohammed Afzal, Liangming
Pan, Nadav Borenstein, Aditya Pillai, Isabelle Au-
genstein, Iryna Gurevych, and Preslav Nakov. 2024b.
Factcheck-bench: Fine-grained evaluation bench-
mark for automatic fact-checkers. In Findings ofthe
Association forComputational Linguistics: EMNLP
2024 , pages 14199–14230. Association for Compu-
tational Linguistics.
Jason Wei, Nguyen Karina, Hyung Won Chung,
Yunxin Joy Jiao, Spencer Papay, Amelia Glaese,
John Schulman, and William Fedus. 2024. Mea-
suring short-form factuality in large language models.
Preprint, arXiv:2411.04368.
Chenxu Yang, Zheng Lin, Chong Tian, Liang Pang,
Lanrui Wang, Zhengyang Tong, Qirong Ho, Yanan
Cao, and Weiping Wang. 2024. A factuality and
diversity reconciled decoding method for knowledge-
grounded dialogue generation. arXiv preprint
arXiv:2407.05718.
Yuheng Zha, Yichi Yang, Ruichen Li, and Zhiting
Hu. 2023. AlignScore: Evaluating factual con-
sistency with a unified alignment function. In
Proceedings ofthe61st Annual Meeting oftheAssociation forComputational Linguistics (V olume
1:Long Papers) , pages 11328–11348. Association
for Computational Linguistics.
11

A Additional short-form QA results
In Table 3 of the main text, we report aggregated results for short-form QA using MeanRank and
MeanValue for ease of presentation. Here, we provide the full results for each of the four QA datasets
(Natural Questions, PopQA, TriviaQA, SimpleQA) for both Llama 3B (see Table 4) and Falcon 3B (see
Table 5).
For Llama 3B, FRANQ calibrated and FRANQ condition-calibrated methods consistently rank among
the top performers. They are the top two methods on TriviaQA and SimpleQA. On PopQA, FRANQ
condition-calibrated ranks among the top three methods, alongside Semantic Entropy and Max Token
Entropy. On Natural Questions, it ranks in the top four, along with DegreeMatrix, Eccentricity, and Sum
of Eigenvalues. Overall, both FRANQ variants achieve the best average performance across all datasets.
MethodNQ PopQA TriviaQA SimpleQA
ROC-AUC ↑PR-AUC ↑PRR↑ROC-AUC ↑PR-AUC ↑PRR↑ROC-AUC ↑PR-AUC ↑PRR↑ROC-AUC ↑PR-AUC ↑PRR↑
Max Sequence Probability .680 .440 .292 .745 .550 .421 .774 .529 .478 .833 .712 .625
CCP .705 .471 .357 .709 .526 .393 .767 .528 .471 .800 .680 .552
Max Token Entropy .723 .503 .389 .768 .607 .455 .796 .569 .523 .809 .697 .555
P(True) .463 .256 -.042 .550 .374 .100 .474 .244 -.022 .419 .294 -.082
Lexical Similarity .720 .494 .386 .763 .571 .462 .775 .508 .485 .818 .685 .585
Degree Matrix .751 .557 .421 .738 .570 .421 .816 .626 .570 .852 .764 .668
Eccentricity .750 .528 .424 .741 .569 .420 .805 .589 .545 .852 .773 .671
Sum of Eigenvalues .749 .553 .411 .740 .564 .416 .816 .621 .561 .861 .774 .686
Number of Semantic Sets .680 .459 .336 .534 .374 .069 .695 .461 .382 .737 .595 .461
Semantic Entropy .727 .518 .373 .776 .602 .496 .801 .565 .546 .863 .766 .684
Perplexity .705 .485 .366 .745 .573 .422 .763 .522 .460 .768 .653 .480
TokenSAR .696 .476 .359 .728 .560 .397 .745 .488 .426 .769 .635 .468
SentenceSAR .678 .395 .269 .762 .562 .459 .794 .560 .521 .858 .767 .682
MaxNLI .540 .308 .042 .362 .262 -.213 .429 .236 -.082 .533 .408 .108
AlignScore .682 .427 .312 .566 .371 .079 .631 .387 .215 .645 .473 .221
Parametric Knowledge .626 .371 .203 .664 .470 .290 .727 .467 .397 .490 .393 .096
XGBoost (all UQ features) .712 .504 .375 .744 .565 .433 .773 .546 .486 .835 .760 .683
XGBoost ( FRANQ features) .651 .412 .283 .690 .503 .350 .692 .441 .328 .860 .747 .676
FRANQ no calibration .637 .456 .268 .676 .481 .278 .773 .557 .467 .826 .717 .601
FRANQ calibrated .735 .529 .405 .765 .597 .468 .821 .623 .580 .869 .761 .695
FRANQ condition-calibrated .748 .526 .409 .763 .605 .477 .821 .618 .576 .877 .776 .703
Table 4: Results on 4 QA datasets for Llama-3b.
For Falcon 3B, FRANQ condition-calibrated achieves the top performance on TriviaQA and second-best
performance on Natural Questions. It also ranks among the top three methods on PopQA and among the
top four on SimpleQA, alongside Degree Matrix, Sum of Eigenvalues, and XGBoost (all features). On
average, FRANQ condition-calibrated is the leading method across the four datasets.
MethodNQ PopQA TriviaQA SimpleQA
ROC-AUC ↑PR-AUC ↑PRR↑ROC-AUC ↑PR-AUC ↑PRR↑ROC-AUC ↑PR-AUC ↑PRR↑ROC-AUC ↑PR-AUC ↑PRR↑
Max Sequence Probability .599 .555 .186 .653 .649 .259 .590 .487 .163 .625 .820 .416
CCP .632 .576 .258 .659 .648 .297 .620 .518 .212 .635 .822 .448
Max Token Entropy .599 .542 .184 .657 .662 .279 .557 .432 .108 .656 .814 .396
P(True) .516 .458 .025 .515 .510 .012 .480 .383 .002 .551 .764 .129
Lexical Similarity .581 .486 .115 .721 .691 .412 .587 .476 .157 .650 .818 .422
Degree Matrix .653 .571 .258 .787 .777 .571 .660 .565 .311 .795 .896 .718
Eccentricity .632 .536 .204 .776 .773 .569 .633 .474 .216 .767 .867 .621
Sum of Eigenvalues .651 .568 .260 .789 .780 .570 .661 .559 .299 .791 .894 .713
Number of Semantic Sets .622 .534 .205 .691 .670 .413 .651 .524 .282 .738 .858 .625
Semantic Entropy .561 .494 .086 .718 .698 .415 .584 .468 .155 .685 .831 .456
Perplexity .593 .545 .199 .626 .629 .215 .547 .428 .097 .645 .809 .388
TokenSAR .602 .555 .210 .656 .654 .249 .553 .433 .106 .654 .818 .405
SentenceSAR .509 .455 .012 .755 .707 .463 .523 .395 .026 .739 .850 .552
MaxNLI .559 .502 .086 .507 .535 .065 .483 .358 -.044 .506 .733 .033
AlignScore .655 .613 .320 .639 .652 .262 .685 .540 .341 .748 .860 .566
Parametric Knowledge .556 .486 .089 .611 .590 .210 .567 .420 .086 .512 .729 .030
XGBoost (all UQ features) .679 .617 .340 .772 .748 .507 .693 .572 .340 .787 .885 .661
XGBoost ( FRANQ features) .640 .596 .292 .694 .712 .414 .624 .517 .236 .731 .853 .532
FRANQ no calibration .576 .496 .113 .732 .716 .448 .609 .492 .205 .738 .862 .616
FRANQ calibrated .617 .541 .215 .773 .749 .520 .626 .513 .228 .769 .885 .682
FRANQ condition-calibrated .668 .591 .331 .781 .764 .533 .695 .606 .377 .776 .886 .668
Table 5: Results on 4 QA datasets for Falcon-3b.
12

B Prompts and setup
Contents (not necessarily includes answer to the following question):
Title: {title1}
Content: {retrieval1}
Title: {title2}
Content: {retrieval2}
...
Title: {title5}
Content: {retrieval5}
Question: {question}
Answer (single line):
Figure 2: Prompt used in short-form QA datasets. Titles
and retrievals correspond to the Wikipedia page title
and the passage retrieved from it.Using the context provided below, answer the question with a balanced approach.
Ensure your response contains an equal number of claims or details drawn directly from
the context and from your own knowledge:
Context: passage 1:{retrieval1}
passage 2:{retrieval2}
passage 3:{retrieval3}
Question: {question}
Answer:
Figure 3: Prompt used in long-form QA datasets. Re-
trievals corresponds to the Wikipedia passage retrieved
for input question.
B.1 Short-form QA
For short-form QA experiments, we used each input question along with the top-5 retrieved passages from
Wikipedia. The prompt format used across all evaluated models is shown in Figure 2. For annotation,
we provided GPT-4 with the input question, the model-generated answer, and the corresponding gold
answer, and asked it to categorize the model’s response as correct, incorrect, or not attempted (the latter
category was excluded from evaluation). This annotation method follows the approach proposed by Wei
et al. (2024). Table 6 presents the statistics of all short-form QA datasets.
Model Dataset Train Size Test Size True False UnverifiableMean Generation
Length (characters)
Llama3bNQ 200 1000 62.4 % 27.6 % 10.0 % 180.1
PopQA 200 1000 50.2 % 22.4 % 27.3 % 149.2
TriviaQA 200 1000 68.0 % 22.3 % 9.7 % 114.4
SimpleQA 200 1000 29.5 % 14.4 % 56.1 % 159.9
Falcon3bNQ 200 1000 44.1 % 37.6 % 18.3 % 352.2
PopQA 200 1000 42.6 % 41.6 % 15.8 % 260.3
TriviaQA 200 1000 57.2 % 32.8 % 10.0 % 324.7
SimpleQA 200 1000 25.5 % 65.6 % 8.8 % 286.9
Table 6: Statistics of datasets used in short-form QA benchmark.
B.2 Long-form QA
For long-form QA experiments, we used each input question along with the top-3 retrieved passages from
Wikipedia. The prompt format employed across all evaluated models is illustrated in Figure 3. We then
decomposed the extracted answers into atomic claims using the prompt shown in Figure 4. For each claim,
we identified its corresponding span in the original sentence using the prompt in Figure 5. Claims for
which we could not extract spans (due to inconsistencies in annotation, e.g., words appearing in a different
order than in the text) were excluded from evaluation. Finally, we annotated the resulting claims for both
factuality and faithfulness using a combination of automatic annotation and subsequent manual validation
(see Appendix B.3 and B.4). Table 7 presents the statistics of all long-form QA dataset.
B.3 Automatic Annotation of Claims
As for faithfulness annotation, each claim is assigned one of three categories during the automatic
annotation. The categories of faithfulness are “faithful” if the context contains information that directly
supports the statement, “unfaithful-contra” if the context contains information that directly contradicts the
13

Your task is to decompose the text into atomic claims.
Let’s define a function named decompose(input:str).
The returned value should be a list of strings, where each string should be a context-independent, fully atomic claim, representing one fact. Atomic claims are simple, indivisible facts
that do not bundle multiple pieces of information together.
### Guidelines for Decomposition:
1. **Atomicity**: Break down each statement into the smallest possible unit of factual information. Avoid grouping multiple facts in one claim. For example:
- Instead of: "Photosynthesis in plants converts sunlight, carbon dioxide, and water into glucose and oxygen."
- Output: ["Photosynthesis in plants converts sunlight into glucose.", "Photosynthesis in plants converts carbon dioxide into glucose.", "Photosynthesis in plants converts water
into glucose.", "Photosynthesis in plants produces oxygen."]
- Instead of: "The heart pumps blood through the body and regulates oxygen supply to tissues."
- Output: ["The heart pumps blood through the body.", "The heart regulates oxygen supply to tissues."]
- Instead of: "Gravity causes objects to fall to the ground and keeps planets in orbit around the sun."
- Output: ["Gravity causes objects to fall to the ground.", "Gravity keeps planets in orbit around the sun."]
2. **Context-Independent**: Each claim must be understandable and verifiable on its own without requiring additional context or references to other claims. Avoid
vague claims like "This process is important for life."
3. **Precise and Unambiguous**: Ensure the claims are specific and avoid combining related ideas that can stand independently.
4. **No Formatting**: The response must be a Python list of strings without any extra formatting, code blocks, or labels like "python".
### Example:
If the input text is: "Mary is a five-year-old girl. She likes playing piano and doesn’t like cookies."
The output should be: ["Mary is a five-year-old girl.", "Mary likes playing piano.", "Mary doesn’t like cookies."]
Note that your response will be passed to the python interpreter, SO NO OTHER WORDS!
decompose("{text}")
Figure 4: Prompt template used wit GPT-4o for decomposing an answer into a set of atomic claims.
Model Train Size Test Size True False Unverifiable Faithful Unfaithful UndefinedMean Generation
Length (characters)
Llama3b 500 1282 90.5 % 6.2 % 3.3 % 37.3 % 62.6 % 0.1 % 1725.4
Falcon3b 500 1048 91.4 % 6.0 % 2.6 % 38.2 % 61.5 % 0.3 % 1720.2
Table 7: Statistics of datasets used in long-form QA benchmark.
statement, “unfaithful-neutral” if the context does not contain any information supporting or contradicting
the statement. Next, the obtained labels are binarized for experimental evaluation: 1 if labeled faithful and
0 if labeled unfaithful-contra, or unfaithful-neutral. This is done since the unfaithful-contra category is
quite small (less than 5% of the data).
Regarding the factuality annotation, the automatic annotation process also assigns one of three categories
to each claim. The categories of factuality are “True” if the statement is factually correct according to
your knowledge, “False” if the statement is factually incorrect according to your knowledge, “unverifiable”
if the factual accuracy of the statement cannot be determined based on available knowledge (should be
assessed without referring to the provided context). After that, the annotation for factuality is binarized,
namely only verifiable claims are taken (we exclude labels of the unverifiable category), 1 if the claim
class is false and 0 if the class is true.
Figure 6 presents the annotation prompt used with GPT-4o-search.
B.4 Manual Enhancement of Automatic Annotation
For manual verification of the automatic annotation, we used reliable sources obtained from Google
search results. An initial validation of the automatic labeling was conducted by comparing automatic and
manual annotations on randomly selected claims to assess class balance. Specifically, 100 claims were
reviewed for Llama 3B and 76 for Falcon 3B. The resulting class distributions are shown in Figure 7(a) and
Figure 8(a), respectively. We also present the results of comparison of manual and automatic annotation
for faithfulness for both models, see Figure 9(a, b).
Given the difficulty of assessing the false and unverifiable categories in automatic annotations, these
cases were prioritized for manual review. Enhanced annotations for the same 100 (Llama 3B) and 76
(Falcon 3B) claims are shown in Figure 7(b) and Figure 8(b). In addition, a full manual re-check of all
False and Unverifiable claims was performed, resulting in 359 manually reviewed claims for Llama 3B
and 240 for Falcon 3B.
We used six student annotators for this study. Each annotator spent approximately three hours on
14

Task: Analyze the given text and the claim (which was extracted from the text). For each sentence in the text:
1. Copy the sentence exactly as it appears in the text.
2. Identify the words from the sentence that are related to the claim, in the same order they appear in the sentence. If no words are related, output "No related words."
Example:
Text:
"Sure! Here are brief explanations of each type of network topology mentioned in the passages: [...]"
Claim:
"Distributed Bus topology connects all network nodes to a shared transmission medium via multiple endpoints."
Answer:
Sentence: "Sure! Here are brief explanations [...]"
Related words from this sentence (same order they appear in the sentence): No related words
Sentence: "2. Distributed Bus: In a Distributed Bus topology, [...]"
Related words from this sentence (same order they appear in the sentence): "Distributed", "Bus", "topology", "all", "network", [...]
Sentence: [... More sentences follow ...]
Now analyze the following text using this format:
Text:
{text}
Claim:
{claim}
Answer:
Figure 5: Prompt template used with GPT-4o to identify the span in the original text corresponding to each atomic
claim. The model is instructed to process each sentence and extract words relevant to the claim, preserving their
order. Parts of the 1-shot example have been omitted for brevity.
Evaluate the given claim using two criteria: **faithfulness** and **factuality**.
- **Faithfulness** assesses how accurately the claim reflects the *context document*. Assign one of the following labels:
- "faithful" — The claim is directly supported by the context.
- "unfaithful-contra" — The claim directly contradicts the context.
- "unfaithful-neutral" — The claim is not present in or supported by the context.
- **Factuality** assesses the truth of the claim *independently of the context*, based on the most up-to-date and reliable sources of knowledge available to
humanity. Assign one of the following labels:
- "True" — The claim is factually correct.
- "False" — The claim is factually incorrect.
- "unverifiable" — The truth of the claim cannot be determined with current knowledge.
Return your answer in the exact format: ("faithfulness label", "factuality label")
Context Document: {retrievals}
Claim: {claim}
Label:
Figure 6: Prompt used with GPT-4o-search to automatically annotate claims for faithfulness and factuality in
long-form QA benchmark.
Annotation Type Num of Claims Accuracy Cohen’s Kappa
Factuality 100 .87 .552
Faithfulness 100 .78 .586
Table 8: Inter-annotator agreement for factuality and faithfulness annotations based on 100 claims of Llama 3B.
Accuracy measures raw agreement, Cohen’s Kappa adjusts for chance agreement.
the annotation task. Instructions were provided orally in the form of informal discussions; no written
guidelines were used. All annotators participated voluntarily and were not financially compensated.
To assess annotation consistency, we also conducted an agreement analysis on the 100 Llama 3B claims,
each independently reviewed by two annotators (see Table 8 for details). These results suggest that the
annotators generally aligned well in both tasks, particularly for factuality, though there is some room for
improvement, especially in clarifying ambiguous or borderline cases.
15

false true unverifiable
GPT Predicted Labelsfalse true unverifiableHuman Annotated Labels3 6 1
3 71 8
1 3 4Factuality Confusion Matrix for Llama 3B
10203040506070(a) Before manual enhancement of automatic annotation
false true unverifiable
GPT+Human Predicted Labelsfalse true unverifiableHuman Annotated Labels4 6 0
0 82 0
0 3 5Factuality Confusion Matrix for Llama 3B
01020304050607080 (b) After manual enhancement of automatic annotation
Figure 7: Balance of classes of factuality annotations for the Llama 3B model. Each matrix is based on 100
randomly selected claims, comparing annotations produced by the model with those from human annotators.
false true unverifiable
GPT Predicted Labelsfalse true unverifiableHuman Annotated Labels4 0 1
6 54 0
2 8 1Factuality Confusion Matrix for Falcon 3B
01020304050
(a) Before manual enhancement of automatic annotation
false true unverifiable
GPT+Human Predicted Labelsfalse true unverifiableHuman Annotated Labels5 0 0
0 60 0
0 8 3Factuality Confusion Matrix for Falcon 3B
0102030405060 (b) After manual enhancement of automatic annotation
Figure 8: Balance of classes of factuality annotations for the Falcon 3B model. Each matrix is based on 76 randomly
selected claims, comparing annotations produced by the model with those from human annotators.
faithful unfaithful-contra unfaithful-neutral
GPT Predicted Labelsfaithful unfaithful-contra unfaithful-neutralHuman Annotated Labels40 1 8
1 0 1
6 3 40Faithfulness Confusion Matrix for Llama 3B
0510152025303540
(a) Llama 3B faithfulness classes
faithful unfaithful-contra unfaithful-neutral
GPT Predicted Labelsfaithful unfaithful-contra unfaithful-neutralHuman Annotated Labels30 1 9
0 0 0
3 0 33Faithfulness Confusion Matrix for Falcon 3B
051015202530 (b) Falcon 3B faithfulness classes
Figure 9: Balance of classes of faithfulness annotations for Llama 3B and Falcon 3B models. The matrices are
based on 100 and 76 randomly selected claims, correspondingly, comparing annotations produced by the model
with those from human annotators.
16

C Additional faithfulness-related results
C.1 Faithfulness Evaluation on Long-Form QA
Table 9 shows the long-form QA dataset results when using faithfulness as the target metric. In this
evaluation, all methods remained consistent with those used in the factuality dataset. Among the compared
methods, AlignScore achieved the best performance, indicating its effectiveness in approximating the
faithfulness within the FRANQ formula.
MethodLlama3b
ROC-AUC ↑PR-AUC ↑PRR↑
Max Claim Probability .614 .751 .298
P(True) .447 .624 -.242
Perplexity .642 .782 .315
Mean Token Entropy .596 .743 .208
CCP .569 .727 .135
MaxNLI .640 .734 .177
AlignScore .856 .907 .789
Parametric Knowledge .273 .559 -.704
Table 9: Llama3b results on long-form QA benchmark with faithfulness target.
C.2 Factuality under faithful/unfaithful conditions
MethodLlama-3b
MeanRank ↓ MeanValue ↑
ROC-AUC PR-AUC PRR ROC-AUC PR-AUC PRR
Max Sequence Probability 7.00 7.00 7.50 .754 .518 .454
CCP 9.00 7.75 8.75 .742 .512 .434
Max Token Entropy 5.25 5.25 5.00 .767 .540 .472
P(True) 15.25 15.25 15.25 .486 .252 -.001
Lexical Similarity 7.00 7.00 7.00 .758 .500 .457
Degree Matrix 4.50 4.00 4.25 .770 .549 .488
Eccentricity 5.50 5.25 4.75 .768 .547 .482
Sum of Eigenvalues 5.50 6.50 6.25 .767 .538 .476
Number of Semantic Sets 13.00 12.50 12.50 .625 .383 .232
Semantic Entropy 2.50 3.00 2.50 .781 .562 .510
Perplexity 8.25 7.50 8.00 .741 .509 .426
TokenSAR 9.25 9.25 9.75 .733 .493 .408
SentenceSAR 5.00 6.75 5.50 .766 .518 .473
MaxNLI 15.50 15.25 15.50 .454 .246 -.067
AlignScore 13.25 13.50 13.50 .606 .321 .170
Parametric Knowledge 10.25 10.25 10.00 .657 .413 .295
(a) Only claims with AlignScore > 0.5MethodLlama-3b
MeanRank ↓ MeanValue ↑
ROC-AUC PR-AUC PRR ROC-AUC PR-AUC PRR
Max Sequence Probability 8.50 8.00 8.50 .752 .648 .446
CCP 7.75 9.25 8.00 .741 .631 .445
Max Token Entropy 6.75 5.00 6.75 .755 .673 .462
P(True) 15.50 15.50 15.25 .474 .405 -.016
Lexical Similarity 6.25 8.00 7.00 .767 .662 .469
Degree Matrix 3.00 2.75 2.50 .796 .728 .551
Eccentricity 3.50 3.50 4.25 .793 .709 .527
Sum of Eigenvalues 1.50 2.50 2.00 .807 .735 .560
Number of Semantic Sets 11.50 9.50 10.00 .693 .605 .377
Semantic Entropy 3.75 4.25 5.25 .782 .689 .502
Perplexity 9.25 8.75 9.25 .725 .634 .411
TokenSAR 11.00 10.50 10.50 .710 .616 .390
SentenceSAR 6.25 6.75 5.50 .770 .667 .473
MaxNLI 15.00 14.75 14.75 .490 .443 .043
AlignScore 14.00 14.00 14.25 .555 .488 .142
Parametric Knowledge 12.50 13.00 12.25 .602 .512 .230
(b) Only claims with AlignScore < 0.5
Table 10: Results averaged across 4 QA datasets for Llama 3B considering only claims with high and low AlignScore.
MethodLlama3b
ROC-AUC ↑PR-AUC ↑PRR↑
MCP .538 .115 .028
PTrue .463 .112 .002
Perplexity .480 .092 -.068
MTE .580 .167 .122
CCP .585 .134 .152
MaxNLI .657 .254 .268
AlignScore .477 .094 -.007
Parametric Knowledge .667 .190 .303
Table 11: Results on long-form QA benchmark with factuality target ( only unfaithful claims ).
Tables 10 present the results for high and low AlignScore, respectively. The results indicate that
Semantic Entropy is the top-performing method for faithful claims, and sum of Eigenvalues of Graph
Laplacian is the top-performing method for unfaithful claims.
Table 11 presents the results of all unsupervised methods with only unfaithful claims left in the test.
Parametric Knowledge outperforms other methods with respect to ROC-AUC and PRR metrics.
17

D Ablation studies
D.1 FRANQ with different choice of UQfaithand UQunfaith
Max Claim ProbabilityP(True)Perplexity
Mean T oken EntropyCCPMaxNLIAlignScore
Parametric KnowledgeConst
UQunfaithMax Claim Probability
P(True)
Perplexity
Mean T oken Entropy
CCP
MaxNLI
AlignScore
Parametric Knowledge
ConstUQfaith.132 .107 .111 .171 .192 .188 .108 .215 .108
.131 .110 .111 .171 .192 .190 .107 .218 .107
.132 .106 .111 .170 .191 .188 .108 .214 .108
.132 .107 .111 .171 .191 .189 .109 .215 .109
.133 .107 .113 .173 .193 .190 .109 .217 .109
.128 .107 .109 .168 .192 .195 .109 .223 .110
.132 .106 .110 .171 .192 .190 .108 .214 .108
.133 .107 .110 .172 .193 .188 .108 .202 .108
.131 .106 .110 .170 .190 .190 .108 .213 .108PRR of FRANQ condition-calibrated for different choice of (UQfaith, UQunfaith)
0.120.140.160.180.200.22
(a) PRR on long-form QA dataset.
Max Sequence ProbabilityCCP
Max T oken EntropyP(True)
Lexical SimilarityDegree MatrixEccentricity
Sum of Eigenvalues
Number of Semantic SetsSemantic EntropyPerplexity T okenSAR
SentenceSARMaxNLI
AlignScore
ParametricProb
UQunfaithMax Sequence Probability
CCP
Max T oken Entropy
P(True)
Lexical Similarity
Degree Matrix
Eccentricity
Sum of Eigenvalues
Number of Semantic Sets
Semantic Entropy
Perplexity
T okenSAR
SentenceSAR
MaxNLI
AlignScore
ParametricProbUQfaith.484 .472 .508 .440 .507 .536 .525 .545 .477 .531 .474 .467 .504 .417 .417 .400
.475 .463 .496 .406 .506 .531 .516 .537 .463 .523 .474 .460 .512 .398 .410 .394
.509 .499 .485 .452 .517 .546 .535 .553 .487 .540 .463 .457 .533 .435 .424 .434
.446 .415 .426 .105 .451 .473 .454 .486 .249 .477 .382 .349 .465 .110 .085 .244
.497 .479 .488 .451 .478 .513 .503 .524 .457 .509 .464 .453 .496 .423 .413 .428
.534 .516 .528 .464 .525 .523 .517 .530 .474 .540 .506 .490 .535 .464 .448 .473
.524 .504 .519 .472 .520 .523 .515 .529 .472 .537 .496 .483 .525 .453 .452 .461
.533 .516 .529 .483 .527 .526 .523 .529 .483 .541 .509 .498 .541 .467 .456 .479
.502 .475 .490 .312 .499 .500 .490 .503 .331 .520 .451 .431 .523 .285 .309 .376
.522 .506 .520 .495 .506 .533 .526 .541 .497 .528 .489 .486 .528 .473 .460 .470
.496 .484 .472 .404 .507 .536 .525 .544 .456 .533 .441 .427 .522 .387 .400 .407
.493 .478 .460 .385 .504 .531 .516 .537 .436 .529 .432 .412 .517 .367 .369 .404
.505 .490 .514 .488 .514 .530 .525 .539 .497 .530 .486 .479 .503 .459 .436 .434
.456 .424 .446 .149 .467 .492 .465 .503 .277 .496 .401 .360 .481 .110 .120 .275
.454 .421 .431 .156 .454 .474 .469 .475 .298 .471 .390 .358 .472 .151 .201 .276
.462 .445 .486 .276 .496 .516 .509 .522 .412 .510 .453 .424 .482 .266 .254 .290PRR of FRANQ condition-calibrated for different choice of (UQfaith, UQunfaith)
0.10.20.30.40.5(b) PRR on short-form QA benchmark (mean value across 4
datasets).
Figure 10: Comparison of FRANQ condition-calibrated with different choice of UQfaithand UQunfaith .
Figure 10 presents the PRR of FRANQ condition-calibrated, for different choices of UQfaithandUQunfaith
methods. On the long-form dataset (Figure 10(a)), the highest PRR is achieved using MaxNLI as UQfaith
and Parametric Knowledge as UQunfaith . The results indicate that on the long-form dataset, the selection of
UQfaithhas a minimal impact, while opting for Parametric Knowledge as the UQunfaith method is essential.
The short-form QA results (see Figure 10(b)) show that many combinations of UQfaithandUQunfaith
achieve PR-AUC scores close to the top, indicating flexibility in selecting these components. The pair
chosen for short-form QA experiments (Semantic Entropy and Sum of Eigenvalues) achieves a PR-AUC
of 0.541, which is close to the highest score of 0.553 obtained by Max Token Entropy and Sum of
Eigenvalues.
18

D.2 FRANQ with alternative faithfulness estimations
MethodLlama3b, long-form QALlama3b, short-form QA
(mean values across 4 datasets)
ROC-AUC ↑PR-AUC ↑PRR↑ROC-AUC ↑PR-AUC ↑PRR↑
FRANQ no calibration .646 .100 .181 .646 .100 .181
FRANQ no calibration T=0.5 .629 .105 .170 .629 .105 .170
FRANQ calibrated .653 .103 .256 .653 .103 .256
FRANQ calibrated T=0.5 .607 .085 .084 .607 .085 .084
FRANQ condition-calibrated .641 .140 .223 .641 .140 .223
FRANQ condition-calibrated T=0.5 .587 .111 .180 .587 .111 .180
Table 12: Comparison of FRANQ performance on Llama3b benchmarks, when using AlignScore with and without
threshold.
MethodLlama3b, long-form QA
ROC-AUC ↑PR-AUC ↑PRR↑
FRANQ no calibration .646 .100 .181
FRANQ calibrated .653 .103 .256
FRANQ condition-calibrated .641 .140 .223
FRANQ condition-calibrated, faithfulness-calibrated .587 .124 .112
Table 13: Comparison of FRANQ performance on Llama3b long-form QA benchmark, when applying calibration
for faithfulness estimator, AlignScore.
Table 12 compares the performance of three original FRANQ versions (each employing a differ-
ent calibration strategy) with three modified versions that use a thresholded AlignScore instead of
raw AlignScore probabilities. In the thresholded versions, the faithfulness probability is defined as
P(cis faithful to r) =1(AlignScore (c)> T)withT= 0.5. These methods are denoted by the ‘T=0.5’
label. The results indicate that, overall, the continuous versions of FRANQ outperform their thresholded
counterparts.
Table 13 further compares the performance of three original FRANQ versions with a condition-calibrated
version of FRANQ that also calibrates AlignScore for faithfulness estimation (this method is denoted
‘FRANQ condition-calibrated, faithfulness-calibrated’). In this version, the AlignScore is calibrated using
a training set with binary gold faithfulness targets and then incorporated into the FRANQ formula. The
results suggest that calibrating AlignScore may reduce the PRR of FRANQ , indicating that it might be
more effective to use AlignScore without faithfulness calibration.
19

D.3 Impact of train size on FRANQ
100 150 200 250 300 350 400 450 500
Train Size0.000.050.100.150.200.25PRR
PRR vs Train Size for Various UQ Methodss
FRANQ no calibration
FRANQ calibrated
FRANQ condition-calibrated
XGBoost (all UQ features)
XGBoost (FRANQ features)
(a) Long-form QA, Llama 3B
40 60 80 100 120 140 160 180 200
Train Size0.400.420.440.460.480.500.520.54PRR
PRR vs Train Size for Various UQ Methodss
FRANQ no calibration
FRANQ calibrated
FRANQ condition-calibrated
XGBoost (all UQ features)
XGBoost (FRANQ features) (b) Short-form QA, Llama 3B
Figure 11: PRR comparison of FRANQ and XGBoost methods with different train size.
Figure 11 shows the PRR for 3 FRANQ variants and 2 XGBoost variants, evaluated across varying
training set sizes on both long-form and short-form QA datasets. The uncalibrated FRANQ , being
unsupervised, exhibits constant performance regardless of training size. In contrast, the supervised FRANQ
variants generally improve with larger training sets, except for the condition-calibrated FRANQ on the
long-form QA dataset, which peaks at 300 training samples and slightly declines thereafter. Across all
training sizes, calibrated versions of FRANQ consistently outperform XGBoost. The results indicate that
the optimal training size for condition-calibrated FRANQ is approximately 300 for long-form QA, while
for short-form QA, its performance stabilizes at around 120 training samples.
D.4 Analysis of XGBoost
AlignScore > 0.72
MaxNLI > 0.39
. . . . . .Parametric Knowledge > exp(-22.13)
. . . . . .true
true falsefalse
true false
Figure 12: Top vertices of first XGBoost tree trained on FRANQ components (AlignScore, Parametric Knowledge
and MaxNLI) for long-form QA Llama 3B behchmark.
We examine the first tree from an XGBoost model trained on FRANQ features (AlignScore, MaxNLI,
and Parametric Knowledge) for long-form QA with Llama 3B. While XGBoost uses multiple trees, the
first tree often captures key decision patterns.
Figure 12 presents the first several nodes in first XGBoost tree. The root splits on AlignScore. If it’s
high, the model next considers MaxNLI; if low, it turns to Parametric Knowledge. This mirrors FRANQ ’s
logic: leading with faithfulness assessment with AlignScore, followed by either MaxNLI or Parametric
Knowledge. The tree thus exhibits structure similar of FRANQ ’s decision process.
20

D.5 Calibration properties of UQ methods
We evaluate the calibration properties of all our UQ methods using the Expected Calibration Error (ECE;
Guo et al. (2017)). ECE quantifies the alignment between predicted confidence scores and observed
accuracy. Specifically, predictions are partitioned into 10 equally spaced confidence bins. Within each
bin, we compute the average predicted confidence and compare it to the empirical accuracy. Lower ECE
values indicate better-calibrated models.
Table 14 reports ECE scores for both long-form QA dataset and short-form QA benchmark using the
Llama 3B model. Only UQ methods that produce confidence values within the [0, 1] interval are included,
as this is a prerequisite for ECE computation. Notably, the two calibrated variants of FRANQ achieve the
best calibration performance across datasets.
Method ECE↓
Max Claim Probability .72
P(True) .94
Perplexity .18
CCP .21
MaxNLI .19
AlignScore .40
Parametric Knowledge .80
XGBoost (all UQ features) .05
XGBoost ( FRANQ features) .06
FRANQ no calibration .44
FRANQ calibrated .02
FRANQ condition-calibrated .03
(a) Long-form QA Llama 3B dataset.Method Mean ECE ↓
Max Sequence Probability .46
CCP .23
P(True) .71
Lexical Similarity .07
Degree Matrix .14
Eccentricity .46
Sum of Eigenvalues .54
Number of Semantic Sets .48
Perplexity .17
MaxNLI .51
AlignScore .13
Parametric Knowledge .23
XGBoost (all UQ features) .15
XGBoost ( FRANQ features) .17
FRANQ no calibration .64
FRANQ calibrated .07
FRANQ condition-calibrated .07
(b) Short-form QA Llama 3B benchmark (ECE is averaged
across 4 QA datasets).
Table 14: Expected Calibration Error (ECE) for all tested UQ methods with Llama 3B.
E FRANQ Examples
In Figure 13, we demonstrate the behavior of FRANQ using three examples from a long-form QA dataset
evaluated with Llama 3B. We selected three representative claims and present their corresponding FRANQ
scores for both the uncalibrated version and condition-calibrated version. The latest uses monotonic func-
tionsfandg, fitted via isotonic regression for MaxNLI and Parametric Knowledge methods, respectively.
F Resource and Expenses
For a single run of data generation and evaluation of UQ baseline methods, we used approximately 8 days
of compute time on an NVIDIA V100 32GB GPU across all datasets and models for the long-form QA
task. In comparison, the short-form QA task required less than one day of GPU time. The OpenAI API
was employed for atomic claim splitting, matching, and annotation. The total API cost for processing
a single model run (Llama 3B) was approximately $100. In addition to automated processing, human
annotations were conducted by six student annotators, each contributing roughly three hours of work.
21

Question : how and when to harvest chestnuts 
Top-3 retrievals: 
1. Properly stored fresh Chestnuts will feel hard as rocks if you squeeze  
them. Though our Chestnuts are quite sweet right out of the bag, due to our  
SECRET post harvest conditioning process, for best flavor you should dry  
them slightly. A unique aspect of Chestnuts is that they become sweeter as  
they dry. 
2. Harvesting Chestnut Trees. After the chestnuts fall from the tree, watch  
for the spiny burs to split. Don’t harvest chestnuts if the burs are still  
green and closed because the nuts inside will be unripe. Harvest the nuts  
every couple of days. 
3. When to harvest chestnuts? Chestnuts don’t ripen at the same time and  
chestnut harvest time can span as much as five weeks, although the nuts  
generally ripen in a 10- to 30-day span of time in late August and . 
September . Allow the nuts to fall from the tree naturally. 
LLM Answer: 
To determine the best time and method for harvesting chestnuts, it's essential  
to consider both the natural process of ripening and the specific  
characteristics of the nuts. 
Firstly, chest  nuts  typically  take  around  10 - 30 days  to rip  en in late  August 
 
and September , as mentioned in passage 3. However, … 
Claim: 
Chestnuts typically take around 10-30 days to ripen in late August and  . 
September. 
Factuality Annotation:  True 
Faithfulness Annotation:  Faithful FRANQ 
1. P(c is faithful to r) = AlignScore(c, r) = 0.98 
2. MaxNLI(c) = max(0.44, 0.34, 0.99) = 0.99 
3. ParametricKnowledge(c) =                            = 
              = 0.52 · 0.66 · … · 0.32 =  3.5·10-15
= 0.44 
= 0.34 
= 0.99 
52%66% 33% 44% 22% 2%83%3%69%87%13%100%0.1%32% 0.2%
0.6% 32%Token probabilities from 
parametric knowledge 
FRANQno calibration(c) = 
         = 0.98 · 0.99 + 0.02 · 3.5 · 10-15 = 0.97 
FRANQcondition-calibrated(c) = 
        = 0.98  · f(0.99) + 0.02 · g(3.5·10-15) = 0.93 
                        0.95                      0.13(a)Faithful–True. FRANQ correctly identifies the claim as faithful and uses MaxNLI, which detects high entailment with the
third retrieved passage. This results in an appropriately high FRANQ score.
Question : How does RAM work and why is it important? 
Top-3 retrievals: 
1. amounts of RAM (mostly SRAM) are also integrated in the CPU and other ICs  
on the motherboard, as well as in hard-drives, CD-ROMs, and … 
2. must be refreshed periodically. Because of this refresh process, DRAM uses  
more power, but it can achieve greater storage densities and … 
3. will always find it easiest to grasp the physical configuration. Much of  
the confusion surrounding memory geometry occurs when … 
LLM Answer: 
RAM, or Random Access Memory, plays a crucial role in a computer system… 
RAM is a volatile  memory  technology , meaning that its contents are lost when the  
power is turned off … 
Claim: 
RAM is a volatile memory technology. 
Factuality Annotation:  True 
Faithfulness Annotation:  Unfaithful FRANQ 
1. P(c is faithful to r) = AlignScore(c, r) = 0.05 
2. MaxNLI(c) = max(0.01, 0.19, 0.06) = 0.19 
3. ParametricKnowledge(c) =                            = 
              = 1.0 · 0.81 · … · 0.44 = 0.05 
= 0.01 
= 0.19 
= 0.06 
Token probabilities from 
parametric knowledge 
FRANQno calibration(c) = 
         = 0.05 · 0.19 + 0.95 · 0.05 = 0.06 
FRANQcondition-calibrated(c) = 
        = 0.05 · f(0.19) + 0.95 · g(0.05) = 0.95 
                      0.21                  0.99 100%81%14% 98% 99% 44%
(b)Unfaithful–True. FRANQ accurately detects the claim’s low faithfulness and assigns its factuality score based on Parametric
Knowledge, which is relatively high. In the uncalibrated version, the final score is underestimated due to the uncalibrated
Parametric Knowledge score. The condition-calibrated version corrects this by assigning a calibrated score of 0.99, resulting in a
correctly high factuality estimate.
Question : which type of diabetes is worse type 1 or type 2 
Top-3 retrievals: 
1. With type 2 diabetes, your body either resists the effects of insulin — a  
hormone that regulates the movement of sugar into your cells — or doesn’t  
produce enough insulin to maintain a normal glucose level. Untreated, type  
2 diabetes can be life-threatening. 
2. More specifically, diabetes is a condition that affects the body's ability  
to use glucose (a type of sugar) as fuel. Glucose is a form of carbohydrate  
that comes from foods such as breads, cereals, pasta, rice, potatoes,  
fruits and some vegetables. 
3. Now, the cause of type 2 diabetes is quite different from type 1. The cause  
of type 2 diabete is primarily a complicated medical condition called  
'insulin resistance.' In fact, in the early stages of type 2 diabetes,  
there's plenty of insulin around, it just doesn't work well. 
LLM Answer: 
Determining which type of diabetes is worse is a complex task … 
Type  1 diabetes  is a condition  where  the body  either  res  ists  the effects  of
insulin  or doesn  't produce  enough  insulin  to maintain a normal glucose level … 
Claim: 
Type 1 diabetes is a condition where the body either resists the effects of  
insulin or doesn't produce enough insulin. 
Factuality Annotation:  False 
Faithfulness Annotation:  Unfaithful FRANQ 
1. P(c is faithful to r) = AlignScore(c, r) = 0.04 
2. MaxNLI(c) = max(0.15, 0.72, 0.62) = 0.72 
3. ParametricKnowledge(c) =                            = 
              = 0.005 · 1.0 · … · 0.96 = 3.8 · 10-15
= 0.15 
= 0.72 
= 0.62 
0.5%Token probabilities from 
parametric knowledge 
FRANQno calibration(c) = 
         = 0.04 · 0.72 + 0.96 · 3.8 · 10-15 = 0.03 
FRANQcondition-calibrated(c) = 
        = 0.04 · f(0.72) + 0.96 · g(3.8·10-15) = 0.15  
                        0.49                    0.14 100%30% 37%6% 2% 91%100%78% 0.1% 1%99%38% 41% 100%
100% 60% 57% 100% 93% 96%
(c)Unfaithful–False. FRANQ correctly identifies the claim as unfaithful and assigns a low factuality score using Parametric
Knowledge, consistent across both the uncalibrated and calibrated versions.
Figure 13: Example outputs from FRANQ .Left: Each example includes the input question, retrieved passages, the
LLM-generated answer, a selected claim from the answer, and corresponding factuality and faithfulness annotations.
Claims and their spans in the answer are highlighted in yellow. If a claim is faithful, its corresponding span in the
retrieved passages is also highlighted. Right : The FRANQ component scores and final factuality estimations, shown
for both the uncalibrated and condition-calibrated versions.
22