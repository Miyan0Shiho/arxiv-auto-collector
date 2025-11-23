# Streamlining Industrial Contract Management with Retrieval-Augmented LLMs

**Authors**: Kristi Topollai, Tolga Dimlioglu, Anna Choromanska, Simon Odie, Reginald Hui

**Published**: 2025-11-18 17:10:57

**PDF URL**: [https://arxiv.org/pdf/2511.14671v1](https://arxiv.org/pdf/2511.14671v1)

## Abstract
Contract management involves reviewing and negotiating provisions, individual clauses that define rights, obligations, and terms of agreement. During this process, revisions to provisions are proposed and iteratively refined, some of which may be problematic or unacceptable. Automating this workflow is challenging due to the scarcity of labeled data and the abundance of unstructured legacy contracts. In this paper, we present a modular framework designed to streamline contract management through a retrieval-augmented generation (RAG) pipeline. Our system integrates synthetic data generation, semantic clause retrieval, acceptability classification, and reward-based alignment to flag problematic revisions and generate improved alternatives. Developed and evaluated in collaboration with an industry partner, our system achieves over 80% accuracy in both identifying and optimizing problematic revisions, demonstrating strong performance under real-world, low-resource conditions and offering a practical means of accelerating contract revision workflows.

## Full Text


<!-- PDF content starts -->

Streamlining Industrial Contract Management with Retrieval-Augmented
LLMs
Kristi Topollai1, Tolga Dimlioglu1, Anna Choromanska1, Simon Odie2, Reginald Hui2
1New York University
2Consolidated Edison Company of New York Inc. (Con Edison),
{kt2664, td2249, ac5455}@nyu.edu
{ODIES, huir}@coned.com
Abstract
Contract management involves reviewing and
negotiating provisions, individual clauses that
define rights, obligations, and terms of agree-
ment. During this process, revisions to provi-
sions are proposed and iteratively refined, some
of which may be problematic or unacceptable.
Automating this workflow is challenging due to
the scarcity of labeled data and the abundance
of unstructured legacy contracts. In this pa-
per, we present a modular framework designed
to streamline contract management through a
retrieval-augmented generation (RAG) pipeline.
Our system integrates synthetic data generation,
semantic clause retrieval, acceptability classi-
fication, and reward-based alignment to flag
problematic revisions and generate improved
alternatives. Developed and evaluated in col-
laboration with an industry partner, our system
achieves over 80% accuracy in both identifying
and optimizing problematic revisions, demon-
strating strong performance under real-world,
low-resource conditions and offering a prac-
tical means of accelerating contract revision
workflows.
1 Introduction
Contract management is a labor-intensive and criti-
cal process essential for risk management, opera-
tional efficiency, and regulatory compliance in both
legal and business environments. Traditional ap-
proaches to contract analysis involve meticulously
reviewing complex and lengthy documents to iden-
tify crucial clauses, obligations, exceptions, and
potential risks. Such manual methods are not only
time-consuming but also prone to human error, re-
quiring significant legal expertise to accurately in-
terpret nuanced language and cross-reference mul-
tiple documents to ensure compliance with rele-
vant laws and standards. Furthermore, organiza-
tions frequently possess extensive legacy contrac-
tual data that remain largely unlabeled and unstruc-
tured, making them difficult to utilize for informed
Figure 1: The system flags problematic clauses and
rewrites them into acceptable revisions, reducing the
risk of negotiation failure.
decision-making.
In this paper, we present a novel proof-of-
concept framework for automating contract man-
agement and revision in settings with limited su-
pervision and large volumes of legacy data. We
evaluate the system on a real-world internal dataset
provided by an industry partner in the utility sec-
tor. Our core contribution is an integrated pipeline
that combines synthetic data generation, semantic
clause retrieval, and acceptability classification to
support systematic contract analysis. This frame-
work enables the transformation of unstructured
legacy contracts into structured, actionable insights,
improving both the efficiency and reliability of
identifying and addressing problematic revisions.
At the core of our proposed solution is a
Retrieval-Augmented Generation (RAG) (Lewis
et al., 2020) methodology, designed to automati-
cally identify unacceptable contract revisions, re-
trieve historically relevant clauses, and generate
contextually appropriate amendments. By com-
1arXiv:2511.14671v1  [cs.CL]  18 Nov 2025

Database Relevant 
Documents Unacceptable 
Revisions Prompt
Acceptable 
Revisions Revisions/Provisions 
Reward Signal Inference 
Alignment/Finetuning 
Provision 1 
Provision 2 
Provision N Revision 1 
Revision 2 
Revision N (Acceptable) 
(Acceptable) (Unacceptable) Optimized Revision 2 Initial Contract Revised Contract Our pipeline for optimizing revisions 
Acceptability 
 Classiﬁer
Acceptability 
 Classiﬁer
Acceptability 
 Classiﬁer
Revision
Optimizer 
Generative 
 LLM
Retriever
Figure 2: The structure of a contract. Our tool operates on each contract revision it identifies as problematic.
bining synthetic data creation, semantic-based re-
trieval, revision classification, and reinforcement-
learning–based alignment of the generator to ac-
ceptability judgments, our approach reduces man-
ual labor, mitigates human error, and enhances over-
all contractual consistency. This research demon-
strates a practical path forward for organizations to
effectively harness their extensive legacy contract
data, transforming it into a dynamic resource for in-
formed decision-making and streamlined contract
management processes.
2 Related Work
Early efforts in automating contract analysis fo-
cused on rule-based and shallow learning meth-
ods for element extraction and clause classifica-
tion. A benchmark of 3,500 contracts with 11 an-
notated element types (e.g., termination dates, par-
ties, payments) was introduced in (Chalkidis et al.,
2017), followed by work on obligation/prohibition
extraction with hierarchical BiLSTMs (Chalkidis
et al., 2018) and unfair clause detection via deep
models in the CLAUDETTE system (Lippi et al.,
2019). Domain-specific resources include a lease-
focused benchmark with ALeaseBERT (Leivaditi
et al., 2020) and the large-scale CUAD dataset for
clause extraction (Hendrycks et al., 2021).
Beyond extraction, several works tackle legal
reasoning. A tax law dataset (Holzenberger et al.,
2020) showed symbolic solvers outperform neural
models on entailment tasks. ContractNLI (Koreeda
and Manning, 2021) frames clause-level review as
document-level NLI, while LEGALBENCH (Guha
et al., 2023) evaluates LLMs across 162 legal rea-
soning tasks. Recent work incorporates LLMs for
generation and clause variation. (Lam et al., 2023)
uses RAG to draft new clauses from negotiation
intent, while (Narendra et al., 2024) applies Natu-
ral Language Inference (NLI) and RAG to detect
deviations from contract templates and build clause
libraries. In contrast, we focus on optimizing re-
visions to existing clauses during negotiation, en-abling faster review and improved consistency. Our
system combines synthetic data generation, seman-
tic retrieval, and a lightweight acceptability clas-
sifier within a clause-level RAG pipeline. Unlike
prior work limited to classification or extraction,
our framework generates revised clauses likely to
be accepted, operating effectively in low-resource
settings with minimal supervision.
3 Dataset
Contracts are the backbone of modern business and
legal relationships. They define how parties collab-
orate, what each side is responsible for, and what
happens when things go wrong. These agreements
are built from a set of provisions, or clauses, that
outline specific terms such as scope of work, pay-
ment, liability, or termination. During negotiations,
these provisions are often revised as parties, such
as those involved in our industrial case study, seek
to align the contract with their respective needs and
constraints. Some revisions are acceptable and lead
to a workable consensus, while others may intro-
duce legal or operational risks and must be rejected.
This process is time-consuming for both parties,
and reaching an agreement can be one of the most
resource-intensive parts of contract management.
3.1 Internal Dataset
Ideally, we would have access to the full lifecycle
of contract formation, including all intermediate
revisions to each provision and the rationale be-
hind every change. However, we operate in a more
constrained setting, with only limited supervision
available. The labeled data used in our study was
obtained internally and is provided in two forms.
First, we are provided with a small, curated set of
fallback revisions: six acceptable and six unaccept-
able revisions for each of two provisions, totaling
just 24 ground truth examples. Second, we lever-
age a collection of 20 previously negotiated con-
tracts, 10 from theStandard Terms and Conditions
for Serviceand 10 from theStandard Terms and
2

Conditions for Purchase of Equipment. These docu-
ments contain tracked edits made during real-world
negotiations. From these, we extract revised provi-
sions and apply a weak labeling heuristic: edited
provisions are treated as unacceptable (requiring
correction), while non-edited provisions that differ
from the original template are treated as accept-
able. In total, our data has 287 acceptable and 143
unacceptable labeled revisions.
3.2 Synthetic Data
Given the limited size of our labeled dataset, re-
lying solely on supervised learning is insufficient
for training robust models. To overcome this, we
leverage synthetic data generation with large lan-
guage models (LLMs), a practical strategy for low-
resource scenarios (Meng et al., 2022; Ye et al.,
2022; Wang et al., 2023). Prior work has shown
that prompting LLMs to generate labeled exam-
ples can yield high-quality supervision for tasks
like classification (Li et al., 2023b), NLI (Hos-
seini et al., 2024), and summarization (Chintagunta
et al., 2021), often rivaling human-labeled datasets
in downstream performance. In our setting, syn-
thetic clause revisions help augment the training set
and improve generalization, particularly in learning
patterns that distinguish acceptable from unaccept-
able revisions. This approach allows us to scale
beyond the small set of manually curated fallback
examples and tracked contract edits.
Due to data sensitivity constraints, all generation
is performed locally, limiting the size of LLMs we
can deploy. We experiment with models from the
LLaMA 3 (Grattafiori et al., 2024) family, specif-
ically the 8-billion parameter LLaMA 3.1 model
and a AutoGPTQ (Frantar et al., 2023; Li et al.,
2025) int4-quantized variant of the 70-billion pa-
rameter LLaMA 3.3 model. To generate synthetic
labeled examples, we use a simple prompting strat-
egy in which we provide the model demonstrations
consisting of a contract provision followed by an
acceptable and an unacceptable revision. An illus-
trative prompt can be found in the Appendix.
The generated set of synthetic revisions is further
refined through a filtering process. We begin by
encoding each synthetic revision using a general-
purpose text embedding model (Li et al., 2023a)
and retrieving its nearest real revisions based on L2
distance in the embedding space. If the majority
label of the retrieved neighbors disagrees with the
synthetic label, the example is discarded. In the end
we are left with around 27,000 synthetic revisions.
payment
indemnificationinspection and teststime of performancewarrantiestermination for convenience
insuranceclaimsexcusable delay
suspensionTrue Revision Embeddings by Provision
 True vs. Synthetic Revision Embeddings
True
SyntheticComing SoonFigure 3: The revisions are clustered by their provision
in the embedding space,
Figure 4 provides t-SNE visualization of the text
embeddings and shows that synthetic embeddings
preserve clause-type clustering, with a clear bi-
modal separation between acceptable and unac-
ceptable revisions. Table 1 supports our choice of
the LLaMA 3.1 8B model over the int4-quantized
70B variant, as the former yields better Fréchet
Inception Distance (FID) scores, adapted here to
measure distributional similarity between real and
synthetic text embeddings, under constrained re-
sources. We also observe diminishing returns be-
yond three demonstrations per prompt, which we
use in our final dataset to balance quality and
prompt length.
Synthetic Dataset Generated by FID Score
Baseline (between two subsets of the real data) 0.08
INT4-Quantized LLaMA 3.3 70B (1 demonstration) 0.46
LLaMA 3.1 8B (1 demonstration) 0.31
LLaMA 3.1 8B (3 demonstrations) 0.14
LLaMA 3.1 8B (5 demonstrations) 0.16
Table 1: Comparison between the synthetic datasets
payment
indemnificationinspection and teststime of performancewarrantiestermination for convenience
insuranceclaimsexcusable delay
suspensionTrue Revision Embeddings by Provision
 True vs. Synthetic Revision Embeddings
True
SyntheticComing Soon
Figure 4: The t-SNE visualization demonstrates that
real and synthetic revisions exhibit similar distributions
in the embedding space.
3

Database Relevant 
Documents Unacceptable 
Revisions Prompt
Acceptable 
Revisions Revisions/Provisions 
Reward Signal Inference 
Alignment/Finetuning 
Provision 1 
Provision 2 
Provision N Revision 1 
Revision 2 
Revision N (Acceptable) 
(Acceptable) (Unacceptable) Optimized Revision 2 Initial Contract Revised Contract Our pipeline for optimizing revisions 
Acceptability 
 Classiﬁer
Acceptability 
 Classiﬁer
Acceptability 
 Classiﬁer
Revision
Optimizer 
Generative 
 LLM
Retriever
Figure 5: Our modular RAG-based pipeline. Using a frozen classifier for supervision allows end-to-end finetuning.
4 Method
4.1 Retrieval-Augmented Revision
Optimization
We propose a modular, (semi)-automatic system de-
signed to assist legal professionals during contract
negotiations. The tool accelerates the revision pro-
cess by identifying potentially unacceptable clauses
and generating improved alternatives that are more
likely to reach consensus. To allow human over-
sight, our system is modular and designed to keep
the legal expert in the loop while automating the
most repetitive aspects of clause revision. The ar-
chitecture consists of four key components: (1) a
structured database of provisions and their histor-
ical revisions, (2) a semantic similarity retriever,
(3) an acceptability classifier, and (4) a genera-
tive module. These modules are integrated into a
retrieval-augmented generation (RAG) pipeline.
The pipeline begins by segmenting a contract
into its constituent revised clauses. The acceptabil-
ity classifier then flags revisions that are likely to be
rejected. For each problematic clause, the retriever
identifies similar revisions from the database and
extracts other relevant provisions from the current
contract to provide context. This set of retrieved
documents is used to construct a prompt for the
generative module, which rewrites the clause with
the goal of increasing its likelihood of acceptance.
We leverage a pretrained LLaMA-3.1 8B model
to generate optimized contract revisions under two
complementary paradigms. First, in zero-shot in-
ference mode, we simply chain the off-the-shelf
retriever and generator without any additional train-
ing. Second, in reward-based alignment mode, we
freeze both the retriever and our acceptability clas-
sifier, using the latter as a frozen reward model to
refine only the generator via reinforcement learn-
ing. Concretely, we take the classifier’s positive-
class probability as the reward signal and update
the generator’s parameters with Proximal PolicyOptimization (PPO) (Schulman et al., 2017), a sta-
ble actor–critic algorithm tailored for large neu-
ral policies. This closely follows the Reinforce-
ment Learning from Human Feedback (RLHF)
paradigm (Christiano et al., 2017), effectively align-
ing the generator’s outputs with the acceptability
judgments encoded by our pretrained classifier. At
inference time, the frozen retriever provides full-
contract context, and the trained generator produces
revisions that maximize the expected acceptability
reward.
4.2 Similarity Retriever
The similarity retriever in our system serves two
main purposes: (1) retrieving past contract revi-
sions that resemble a given query, providing prece-
dent for potential rewrites, and (2) identifying con-
textually related clauses within the same contract,
which can influence how a revision is interpreted.
In addition, due to the absence of manual similarity
labels, we generate synthetic supervision by cre-
ating semantically equivalent paraphrases of each
revision. This augmented dataset is used for both
fine-tuning and evaluation.
4.2.1 Retrieving Past Revisions
Our database stores the text and embeddings of
prior revisions. Given the document length, we
experimented with embedding models with large
context windows ( >2048 tokens), and evaluated
both general-purpose and legal-domain variants.
Retrieval is performed via cosine similarity with
the query revision, followed by reranking the top-
Khits using a cross-encoder.
While embedding model fine-tuning had min-
imal impact, due to limited data, fine-tuning the
cross-encoder substantially improved results. We
compare two strategies:i)binary classification
(similar vs. not similar), andii)graded similarity
with soft labels: y= 1 for paraphrases, y= 0.5 for
acceptable revisions of the same provision, y= 0.3
4

Retriever Provision Retrieval Top-10 Accuracy Top-5 Accuracy Top-1 Accuracy
legal-BERT 96.12% 61.24% 51.08% 29.63%
legal-Longformer 90.21% 46.50% 38.99% 22.45%
Qwen2-1.5B-Instruct 99.90% 88.23% 80.34% 53.64%
Qwen3-Embedding-4B99.94%90.17% 84.01% 55.98%
+ Pretrained BGE-Reranker-Large - 52.22% 12.31%
+ Finetuned BGE-Reranker-Large (1st method) - 72.76% 51.19%
+ Finetuned BGE-Reranker-Large (2nd method) -88.37%58.93%
Table 2: Retrieval accuracy between different embedding models and Qwen3 + Rerankers.
for acceptable vs. unacceptable pairs of the same
provision, and y= 0 for unrelated provisions. The
graded approach captures finer semantic distinc-
tions critical for legal document retrieval.
4.2.2 Intra-Contract Clause Retrieval
To identify clauses within the same contract that are
contextually related, we consider two approaches.
The first leverages expert-provided labels for our
internal contract templates. Alternatively, we adopt
an automated strategy inspired by Lam et al. (2023).
We begin by using our LLaMA model to extract
keywords, key phrases, and explicit references to
other clauses. Given the relatively small num-
ber of provisions per contract, we then apply a
cross-encoder to all clause pairs and retain those
whose similarity score exceeds a predefined thresh-
old. This allows us to efficiently capture clause
interdependencies, which are also important for
generating context-aware revisions.
4.3 Acceptability Classifier
The acceptability classifier is a central component
of our pipeline, responsible for flagging potentially
problematic revisions. We frame this as a binary
classification task, acceptable vs. unacceptable,
and explore two complementary approaches.
The first leverages the zero-shot capabilities of
a generative LLM (LLaMA in our case). We re-
trieve the top- Ksemantically similar acceptable
and unacceptable revisions (Section 4.2) as demon-
strations and prompt the model to classify the
query revision. This approach offers interpretabil-
ity through reasoning both in the input and output,
facilitating collaboration with legal professionals.
The second approach employs a discriminative
model trained on learned embeddings. Revisions
are encoded using the same embedding model from
the retrieval module and classified using logistic
regression. To account for variation across provi-
sions, we adopt an ensemble strategy: revisions are
clustered into Kgroups based on their embeddings,and a separate classifier is trained for each cluster.
At inference time, the query is routed to its nearest
cluster for prediction. The choice of Kbalances
generality and specialization.
5 Experiments
We first evaluate the retrieval and classification
modules independently, detailing their architec-
tures and evaluation protocols, before assessing the
full pipeline’s ability to generate more acceptable
contract revisions.
5.1 Similarity Retriever
We evaluate the similarity retriever on our aug-
mented synthetic revision dataset, where each orig-
inal revision is paired with a semantically equiv-
alent rephrased version. The evaluation metric is
top-1 and top-K retrieval accuracy: the fraction of
queries for which the correct rephrased version is
retrieved among the top-1 or top-K results, respec-
tively. We compare four embedding models:
•Legal-domain:LEGAL-BERT (Chalkidis
et al., 2020) and LEGAL-LONGFORMER(Ma-
makas et al., 2022),
•General-purpose models:QWEN2-1.5B-
INSTRUCT(Yang et al., 2024) and QWEN3-
EMBEDDING-4B (Yang et al., 2025), both
of which are top-performing models on the
MTEB benchmark (Muennighoff et al., 2023).
As shown in Table 2, general-purpose models
significantly outperform legal-domain models, pri-
marily due to the latter’s limited context length,
which hampers clause-level semantic understand-
ing. Among the general-purpose models, Qwen3-
Embedding-4B achieves the best top-1 and top-K
retrieval accuracy and consistently retrieves seman-
tically aligned revisions from the same provision.
This evaluation is particularly challenging, as the
synthetic dataset includes many near-duplicate re-
visions generated from a limited set of roughly 400
unique examples.
5

Reranking.We apply theBGE-RERANKER-
LARGE(Xiao et al., 2024) cross-encoder to
rerank the top-10 candidates retrieved by the best-
performing embedding model, Qwen3-Embedding-
4B. We evaluate both the off-the-shelf pretrained
reranker and two versions finetuned using our pro-
posed strategies. Surprisingly, as seen in Table 2,
the pretrained reranker fails to improve the initial
ranking. Moreover, the binary classification-based
finetuning strategy performs worse than using no
reranker at all, leading to degraded top-1 accuracy.
In contrast, the graded similarity-based finetuning
substantially improves the reranking performance,
outperforming both the baseline and the binary-
trained variant.
5.2 Acceptability Classifier
We evaluate the acceptability classifier using a
train/test split on the synthetic dataset, with train-
ing performed exclusively on the synthetic training
set. Evaluation is conducted on both the held-out
synthetic test set and the original collection of 430
real revisions. For the embedding-based approach,
we use the Qwen3-Embedding-4B model to gener-
ate embeddings, followed by a simple ensemble of
logistic regression classifiers and we set the num-
ber of clusters K=X through hyperparameter
search.
Classifier Synthetic Original
Train Test Test F1
Llama Zero-Shot (No clustering) - 65.7% 0.682
Logistic Ensemble (No clustering) 89.1% 84.4% 79.3% 0.851
Logistic Ensemble (5 clusters) 91.9% 85.6% 82.8% 0.878
Logistic Ensemble (8 clusters) 93.0% 85.9%84.7% 0.889
Table 3: Comparison between acceptability classifiers.
The results can be found in Table 3. Interest-
ingly, the zero-shot LLaMA-based approach strug-
gles to reliably distinguish between acceptable and
unacceptable revisions. In contrast, the embedding-
based classifier performs significantly better. More-
over, its misclassifications tend to be associated
with low-confidence (i.e., ambiguous) predictions,
making it especially suitable for a semi-automatic
setup, where uncertain cases can still be flagged for
expert review.
5.3 Retrieval-Augmented Revision
Optimization
Assessing the success of our revision optimization
presents its own challenges. To automate evalu-
ation at scale, we apply our frozen acceptabilityclassifier to each post-optimization revision and re-
port the fraction classified as acceptable. Although
this metric depends on a classifier that is not per-
fect, it nevertheless provides a practical proxy when
labeled data are scarce.
MethodPercentage of
Successful Optimizations
Zero-Shot Inference (Modular Pretraining)
1 Demonstration 59.1%
5 Demonstrations 67.5%
Acceptability-based Alignment
1 Demonstration 80.8%
5 Demonstrations81.9%
Table 4: Comparison between our two RAG pipelines
for revision optimization.
Detailed results appear in Table 4 and the Supple-
ment. In zero-shot mode, performance improves
as we increase the number of in-context demon-
strations, peaking at four examples before plateau-
ing. Under our acceptability-based alignment pro-
cedure, reinforcement-learning refinement further
boosts the rate of acceptable revisions, bringing
generated outputs closer to ground-truth acceptable
edits. Although our reward model is trained on syn-
thetic data and, these results highlight the promise
of replacing it with expert-provided signals.
6 Conclusion
We presented a modular Retrieval-Augmented,
LLM-based system for semi-automatic contract
management, engineered to operate under minimal
supervision and leverage large volumes of legacy
contractual data. Our framework integrates syn-
thetic data generation, semantic clause retrieval,
and an ensemble-based acceptability classifier to
detect and propose revisions for problematic con-
tract clauses. Empirical results on internal in-
dustrial contracts demonstrate strong performance
across semantic retrieval and revision optimization,
showing that even highly domain-specific legal
tasks can be meaningfully automated. Crucially,
the pipeline is designed with a human-in-the-loop
workflow that preserves legal rigor: automated
tools offer candidate revisions, while domain ex-
perts make final decisions on edge cases or high-
stakes clauses. This integration of automation and
expert review improves productivity by focusing
legal attention where it is most needed, without
sacrificing contract integrity. Finally, the modu-
lar design allows for flexible customization across
contracts, vendors, and legal regimes.
6

Limitations
While our system achieves strong overall perfor-
mance, it currently struggles to capture fine-grained
semantic changes within revision proposals, such
as increases in contract budget or adjustments to
milestone dates, that require contextual or numeric
reasoning. Additionally, the pipeline does not incor-
porate the identity of the third-party vendor, which
may influence whether a revision is acceptable. A
clause modification that is acceptable to one ven-
dor might be rejected by another, leading to po-
tential mismatches in optimization. Nonetheless,
the classifier reliably flags such vendor-sensitive
revisions, ensuring they are routed to legal experts
for targeted revision. This hybrid human-in-the-
loop workflow still leads to significant productivity
gains by automating the majority of routine clause
adjustments. Future extensions could incorporate
vendor identifiers or negotiation history into the
dataset to further personalize optimization.
References
Ilias Chalkidis, Ion Androutsopoulos, and Achilleas
Michos. 2017. Extracting contract elements. In
Proceedings of the 16th edition of the International
Conference on Artificial Intelligence and Law, pages
19–28.
Ilias Chalkidis, Ion Androutsopoulos, and Achilleas
Michos. 2018. Obligation and prohibition ex-
traction using hierarchical rnns.arXiv preprint
arXiv:1805.03871.
Ilias Chalkidis, Manos Fergadiotis, Prodromos Malaka-
siotis, Nikolaos Aletras, and Ion Androutsopoulos.
2020. LEGAL-BERT: The muppets straight out of
law school. InFindings of the Association for Com-
putational Linguistics: EMNLP 2020, pages 2898–
2904, Online. Association for Computational Lin-
guistics.
Bharath Chintagunta, Namit Katariya, Xavier Amatri-
ain, and Anitha Kannan. 2021. Medically aware
gpt-3 as a data generator for medical dialogue sum-
marization. InMachine Learning for Healthcare
Conference, pages 354–372. PMLR.
Paul F Christiano, Jan Leike, Tom Brown, Miljan Mar-
tic, Shane Legg, and Dario Amodei. 2017. Deep
reinforcement learning from human preferences.Ad-
vances in neural information processing systems, 30.
Elias Frantar, Saleh Ashkboos, Torsten Hoefler, and
Dan Alistarh. 2023. Gptq: Accurate post-training
quantization for generative pre-trained transformers.
InProceedings of the 11th International Conference
on Learning Representations (ICLR). Published at
ICLR 2023.Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri,
Abhinav Pandey, Abhishek Kadian, Ahmad Al-
Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten,
Alex Vaughan, and 1 others. 2024. The llama 3 herd
of models.arXiv preprint arXiv:2407.21783.
Neel Guha, Julian Nyarko, Daniel Ho, Christopher Ré,
Adam Chilton, Alex Chohlas-Wood, Austin Peters,
Brandon Waldon, Daniel Rockmore, Diego Zam-
brano, and 1 others. 2023. Legalbench: A collab-
oratively built benchmark for measuring legal reason-
ing in large language models.Advances in Neural
Information Processing Systems, 36:44123–44279.
Dan Hendrycks, Collin Burns, Anya Chen, and
Spencer Ball. 2021. Cuad: an expert-annotated nlp
dataset for legal contract review.arXiv preprint
arXiv:2103.06268.
Nils Holzenberger, Andrew Blair-Stanek, and Benjamin
Van Durme. 2020. A dataset for statutory reasoning
in tax law entailment and question answering.arXiv
preprint arXiv:2005.05257.
Mohammad Javad Hosseini, Andrey Petrov, Alex Fab-
rikant, and Annie Louis. 2024. A synthetic data
approach for domain generalization of NLI models.
InProceedings of the 62nd Annual Meeting of the
Association for Computational Linguistics (Volume 1:
Long Papers), pages 2212–2226, Bangkok, Thailand.
Association for Computational Linguistics.
Yuta Koreeda and Christopher D Manning. 2021.
Contractnli: A dataset for document-level natural
language inference for contracts.arXiv preprint
arXiv:2110.01799.
Kwok-Yan Lam, Victor CW Cheng, and Zee Kin Yeong.
2023. Applying large language models for enhancing
contract drafting. InLegalAIIA@ ICAIL, pages 70–
80.
Spyretta Leivaditi, Julien Rossi, and Evangelos
Kanoulas. 2020. A benchmark for lease contract
review.arXiv preprint arXiv:2010.10386.
Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio
Petroni, Vladimir Karpukhin, Naman Goyal, Hein-
rich Küttler, Mike Lewis, Wen-tau Yih, Tim Rock-
täschel, Sebastian Riedel, and Douwe Kiela. 2020.
Retrieval-augmented generation for knowledge-
intensive nlp tasks. InProceedings of the 34th Inter-
national Conference on Neural Information Process-
ing Systems, NIPS ’20, Red Hook, NY , USA. Curran
Associates Inc.
Yuhang Li, Ruokai Yin, Donghyun Lee, Shiting Xiao,
and Priyadarshini Panda. 2025. Gptaq: Efficient
finetuning-free quantization for asymmetric calibra-
tion. InProceedings of the 42nd International Con-
ference on Machine Learning (ICML), volume 267,
Vancouver, Canada. PMLR. Finetuning-Free Quanti-
zation for Asymmetric Calibration.
Zehan Li, Xin Zhang, Yanzhao Zhang, Dingkun Long,
Pengjun Xie, and Meishan Zhang. 2023a. Towards
7

general text embeddings with multi-stage contrastive
learning.arXiv preprint arXiv:2308.03281.
Zhuoyan Li, Hangxiao Zhu, Zhuoran Lu, and Ming
Yin. 2023b. Synthetic data generation with large lan-
guage models for text classification: Potential and
limitations. InProceedings of the 2023 Conference
on Empirical Methods in Natural Language Process-
ing, pages 10443–10461, Singapore. Association for
Computational Linguistics.
Marco Lippi, Przemysław Pałka, Giuseppe Contissa,
Francesca Lagioia, Hans-Wolfgang Micklitz, Gio-
vanni Sartor, and Paolo Torroni. 2019. Claudette: an
automated detector of potentially unfair clauses in
online terms of service.Artificial Intelligence and
Law, 27:117–139.
Dimitris Mamakas, Petros Tsotsi, Ion Androutsopou-
los, and Ilias Chalkidis. 2022. Processing long legal
documents with pre-trained transformers: Modding
LegalBERT and longformer. InProceedings of the
Natural Legal Language Processing Workshop 2022,
pages 130–142, Abu Dhabi, United Arab Emirates
(Hybrid). Association for Computational Linguistics.
Yu Meng, Jiaxin Huang, Yu Zhang, and Jiawei Han.
2022. Generating training data with language mod-
els: Towards zero-shot language understanding.Ad-
vances in Neural Information Processing Systems,
35:462–477.
Niklas Muennighoff, Nouamane Tazi, Loic Magne, and
Nils Reimers. 2023. MTEB: Massive text embedding
benchmark. InProceedings of the 17th Conference
of the European Chapter of the Association for Com-
putational Linguistics, pages 2014–2037, Dubrovnik,
Croatia. Association for Computational Linguistics.
Savinay Narendra, Kaushal Shetty, and Adwait Ratna-
parkhi. 2024. Enhancing contract negotiations with
llm-based legal document comparison. InProceed-
ings of the Natural Legal Language Processing Work-
shop 2024, pages 143–153.
John Schulman, Filip Wolski, Prafulla Dhariwal, Alec
Radford, and Oleg Klimov. 2017. Proximal policy
optimization algorithms.CoRR, abs/1707.06347.
Yizhong Wang, Yeganeh Kordi, Swaroop Mishra, Alisa
Liu, Noah A. Smith, Daniel Khashabi, and Hannaneh
Hajishirzi. 2023. Self-instruct: Aligning language
models with self-generated instructions. InProceed-
ings of the 61st Annual Meeting of the Association for
Computational Linguistics (Volume 1: Long Papers),
pages 13484–13508, Toronto, Canada. Association
for Computational Linguistics.
Shitao Xiao, Zheng Liu, Peitian Zhang, Niklas Muen-
nighoff, Defu Lian, and Jian-Yun Nie. 2024. C-pack:
Packed resources for general chinese embeddings. In
Proceedings of the 47th International ACM SIGIR
Conference on Research and Development in Infor-
mation Retrieval, SIGIR ’24, page 641–649, New
York, NY , USA. Association for Computing Machin-
ery.An Yang, Anfeng Li, Baosong Yang, Beichen Zhang,
Binyuan Hui, Bo Zheng, Bowen Yu, Chang
Gao, Chengen Huang, Chenxu Lv, and 1 others.
2025. Qwen3 technical report.arXiv preprint
arXiv:2505.09388.
An Yang, Baosong Yang, Binyuan Hui, Bo Zheng,
Bowen Yu, Chang Zhou, Chengpeng Li, Chengyuan
Li, Dayiheng Liu, Fei Huang, Guanting Dong, Hao-
ran Wei, Huan Lin, Jialong Tang, Jialin Wang, Jian
Yang, Jianhong Tu, Jianwei Zhang, Jianxin Ma, and
40 others. 2024. Qwen2 technical report.arXiv
preprint arXiv:2407.10671.
Jiacheng Ye, Jiahui Gao, Qintong Li, Hang Xu, Jiangtao
Feng, Zhiyong Wu, Tao Yu, and Lingpeng Kong.
2022. Zerogen: Efficient zero-shot learning via
dataset generation. InConference on Empirical Meth-
ods in Natural Language Processing.
8

A Hardware and Environment
Due to the sensitivity of the internal dataset, all in-
ference and fine-tuning were performed locally on
private machines. Specifically, our experiments
were conducted using two NVIDIA RTX 3090
GPUs, providing a total of 48GB of VRAM. This
setup allowed us to run all LLMs used in our
pipeline with full FP16 precision.
B Prompts
Synthetic Revisions Prompt
Use the following pairs of provisions and fallback
revisions to understand what constitutes an acceptable
and unacceptable revision. Then provide revisions for
the given query provision.
Demonstration 1
Provision:[Random provision text]
Acceptable revision:[Acceptable revision ]
Unacceptable revision:[Unacceptable revision]
...
Demonstration N
Provision:[Random provision text]
Acceptable revision:[Acceptable revision]
Unacceptable revision:[Unacceptable revision]
Query Provision:[Query provision text]
Revision Optimization Prompt
Use the following examples of provisions and their
revisions to learn how to transform unacceptable
revisions into acceptable ones. Then, provide revised
versions for the given query unacceptable revision.
You are also provided with clauses from the same
contract that may be contextually relevant to the query.
Incorporate their meaning and constraints when rewriting.
Demonstration 1
Provision:[Provision text A]
Unacceptable revision:[Revised text A]
Acceptable revision:[Corrected version of A]
Demonstration 2
Provision:[Provision text B]
Unacceptable revision:[Revised text B]
Acceptable revision:[Corrected version of B]
...
Related Clauses (from current contract):
Related clause:[Related clause 1]
Related clause:[Related clause 1]
...
Query Unacceptable Revision:[Unacceptable
Revision text Q]
Optimized Unacceptable Version:Rephrasing Prompt for Semantic Equivalence
Rephrase the following contract clause revision so that it
is semantically identical but expressed using different
wording. Do not change the meaning, intent, or legal
interpretation of the revision. Ensure the rephrasing
retains the same level of formality and contractual tone.
Original Revision:[Original revision text]
Rephrased Revision:
Clause Dependency Extraction Prompt
Given the contract text below, analyze the specified
clause to extract:
(1) The key terms and phrases that summarize its content.
(2) Any explicit or implicit references to other clauses
within the same contract (e.g. “as described in Section
5”, “subject to Clause 10”).
Return the output in JSON format with the keys
"keywords" ,"key_phrases" , and "references" . Do
not modify the text of the clause.
Full Contract: [Insert full or partial contract text here ]
Target Clause:[Insert clause to analyze here]
Output:
{
"keywords": [...],
"key_phrases": [...],
"references": [...]
}
Zero-Shot Acceptability Classification Prompt
Below are examples of contract clause revisions labeled
as either acceptable or unacceptable. Analyze the
patterns in these examples and determine whether the
given query revision should be classified as ACCEPTABLE
orUNACCEPTABLE . Provide a brief justification for your
classification.
Demonstration 1
Revision:[Revised text A]
Label:ACCEPTABLE
Demonstration 2
Revision:[Revised text B]
Label:UNACCEPTABLE
...
Demonstration N
Revision:[Revised text N]
Label:ACCEPTABLE
Query Revision:[Revision to classify]
Output:
Label: ACCEPTABLE
Justification: [Explain the decision]
9

C Original Dataset
Among all provisions, only a subset is frequently re-
vised, and within this subset, certain provisions are
disproportionately associated with unacceptable re-
visions. In fact, 75% of unacceptable revisions fall
under just a handful of provisions. Figures 6 and 7
visualize the distribution of acceptable and unac-
ceptable revisions, respectively, highlighting that
while acceptable revisions are relatively evenly dis-
tributed, unacceptable revisions are concentrated
in specific provision types such asindemnification,
time of performance, andinsurance.
0 10 20 30 40 50
Percentage of T otal Revisionsrestinfringementconfidentialityassignmentsubcontractinginspection and testschangeswarrantiessuspensionexcusable delayfirm pricecontract formationdefinitionscon edison performancesubmission to jurisdiction/choice of forumtime and material and cost reimbursable workgift policy and unlawful conductrequired approvalsclaimstaxespaymentMost common Acceptable Revisions
Figure 6: Distribution of acceptable revisions across
provisions. Each bar indicates the percentage of accept-
able revisions contributed by that provision.
0 5 10 15 20 25
Percentage of T otal Revisionsrestclaimssafeguardsinvestigation and auditlimitation of liabilitycancellation for defaultset-offcon edison performancesubcontractingchangesconfidentialityinfringementsuspensionexcusable delaypaymenttermination for conveniencewarrantiesinsuranceinspection and teststime of performanceindemnificationMost common Unacceptable Revisions
Figure 7: Distribution of unacceptable revisions across
provisions. A small subset of provisions accounts for
the majority of unacceptable cases.D Hyperparameters and Training
Settings
This section summarizes the key hyperparameters
and configuration details used across different com-
ponents of our system for reproducibility.
Parameter Value
Model LLaMA 3.1–8B Instruct
Max new tokens 8192
Temperature 0.8
Top-k 50
Top-p 0.9
Number of demonstrations (K) 1/2/3/4/5/7/9
Embedding model for filtering Qwen3-Embedding-4B
Neighbors used for filtering (k) 10/20/40
Table 5: Hyperparameters for synthetic revision genera-
tion.
Component Setting
Embedding model Qwen3-Embedding-4B
Similarity metric Cosine/ L2
Top-Kcandidates retrieved 5/10/20
Reranker model BGE-Reranker-Large
Finetuning objective Multiclass/Binary Classification
Training epochs 10
Optimizer AdamW
betas (0.9, 0.999)
weight decaye 0.1
Batch size 128
Learning rate 0.001
Similarity label scheme {1.0, 0.5, 0.3, 0.0}, {1.0, 0.0}
Table 6: Hyperparameters for retriever and reranker
finetuning.
Component Setting
Embedding model Qwen3-Embedding-4B
Clustering algorithm k-means
Number of clusters (K) 3,5,8,11
Classifier type Logistic Regression
Train/val split ratio 90/10
Routing metric Cosine, L2
Table 7: Hyperparameters for the acceptability classifier.
Parameter Value
Retriever top-K(past revisions) 1/2/3/4/5/6/7
Max tokens for merged prompt 128k
Temperature 0.8
Top-k 50
Top-p 0.9
Max new tokens 8192
Table 8: Hyperparameters for inference in the RAG
pipeline.
10

Parameter Value
Policy model LLaMA 3.1–8B Instruct
LoRA rank (r) 8
LoRAα 32
LoRA dropout 0.05
Reward model Frozen acceptability classifier
Reward normalization None
Batch size 4
PPO epochs 4
Learning rate 1×10−5
Discount factor (γ) 1.0
Clip parameter (ϵ) 0.2
Entropy coefficient 0.01
KL penalty coefficient 0.1
Max sequence length 8192 tokens
Gradient clipping 1.0 (global norm)
Table 9: Hyperparameters for PPO.
11