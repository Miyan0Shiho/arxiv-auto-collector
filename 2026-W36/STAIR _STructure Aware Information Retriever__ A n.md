# STAIR (STructure Aware Information Retriever): A novel dataset and LLM based retriever for document structure augmentation

**Authors**: Vineet Kumar, Meghanadh Pulivarthi, vishwajeet kumar, Jaydeep Sen, Riyaz Ahmad Bhat, Sachindra Joshi

**Published**: 2026-09-03 14:02:53

**PDF URL**: [https://arxiv.org/pdf/2609.03874v1](https://arxiv.org/pdf/2609.03874v1)

## Abstract
Retrieval Augmented Generation (RAG) is a key component for generating accurate and hallucination free answers using Large Language Models (LLMs). LLMs are improving at handling long context, but still suffer from "lost in the middle" problem. Thus, precise and accurate retrieval is important. Current retrievers chunk long context into length-based manageable chunks - in the process throwing away rich and informative semantic global structure in the corpus. We introduce a novel retrieval system STAIR that empowers an LLM to exploit global structure in a corpus such as a Table of Contents (ToC) to efficiently store and retrieve information from its model parameters. Our thorough and careful ablation studies with a finetuned Differentiable Search Index (DSI) system show that ToC helps build a low hallucination (less than 0.05%) generative Information Retrieval (IR) system and can generalize to examples where very few training samples are available. To further research in this novel direction of ToC based retrieval we release SearchTome - a diverse benchmark created from 18 books across 6 diverse domains to further research in this novel direction. STAIR achieves a high Recall@1 score of 82.6% on SearchTome as compared to DSI (76.9%), where the difference is found to be statistically significant. STAIR easily beats other strong baselines such as BM25 (59.5%), DPR (68.7%) and out-of-the-box Mistral (13.8%).

## Full Text


<!-- PDF content starts -->

STAIR (STructure Aware Information Retriever): A novel dataset and
LLM based retriever for document structure augmentation
Vineet Kumar *Meghanadh Pulivarthi Vishwajeet Kumar Jaydeep Sen
Riyaz Ahmad Bhat Sachindra Joshi
IBM
vineet.mundhra@gmail.com vishk024@in.ibm.com
jaydesen@in.ibm.com Riyaz.Bhat@ibm.com jsachind@in.ibm.com
Abstract
Retrieval Augmented Generation (RAG) is a
key component for generating accurate and hal-
lucination free answers using Large Language
Models (LLMs). LLMs are improving at han-
dling long context, but still suffer from “lost in
the middle” problem. Thus, precise and accu-
rate retrieval is important. Current retrievers
chunk long context into length-based manage-
able chunks – in the process throwing away
rich and informative semantic global structure
in the corpus. We introduce a novel retrieval
systemSTAIRthat empowers an LLM to ex-
ploit global structure in a corpus such as a Ta-
ble of Contents (ToC) to efficiently store and
retrieve information from its model parame-
ters. Our thorough and careful ablation studies
with a finetuned Differentiable Search Index
(DSI) system show thatToChelps build a low
hallucination (less than 0.05%) generative In-
formation Retrieval (IR) system and can gen-
eralize to examples where very few training
samples are available. To further research in
this novel direction ofToCbased retrieval we
releaseSearchTome– a diverse benchmark
created from 18 books across 6 diverse do-
mains to further research in this novel direc-
tion.STAIRachieves a high Recall@1 score
of 82.6% onSearchTomeas compared to DSI
(76.9%), where the difference is found to be sta-
tistically significant.STAIReasily beats other
strong baselines such as BM25 (59.5%), DPR
(68.7%) and out-of-the-box Mistral (13.8%).
The benchmark data and code used for training
STAIRis available at https://anonymous.
4open.science/r/s_331/README.md.
1 Introduction
The burgeoning interest in Retrieval Augmented
Generation (RAG) has led to a significant surge in
the development of advanced Information Retrieval
(IR) systems. Large Language Models (LLMs) in
*Work done while at IBM. Currently at Amazon Books
Science.turn can now handle large contexts (Chen et al.,
2023; Liu et al., 2024a), though they suffer from a
“lost in the middle” problem (Liu et al., 2024c; Bai
et al., 2024a,b; Li et al., 2024). Therefore, retriev-
ing precise information (Pipitone and Alami, 2024)
is extremely important to curb hallucinations (La-
ban et al., 2024) and generate accurate responses.
Current retrievers address this by creating length
based chunks (Setty et al., 2024) and throwing
away rich and informative semantic global struc-
ture in the corpus. This leads to a sub-optimal re-
trieval quality – length-based chunks compete with
each other due to a lack of semantic coherence and
boundaries.
In this work, we address this key limitation by
augmenting the retriever with a structured global
view of the corpus. Global structured view over a
long context helps knowledge ingestion (Liu et al.,
2024b). Further, LLMs are capable of storing the
entire corpus in its model parameters to directly
generate a document identifier for a user query (Tay
et al., 2022). We posit that by empowering an
LLM with a global structure of the search corpus, it
can store and retrieve information more accurately
from its model parameters. Such a global structure
already exists for a Wikipedia (Wikipedia, 2024)
page, textbooks, and enterprise help and product
feature webpages (SAP, 2024) and technical re-
ports (SEC, 2024).
Figure 1 demonstrates that for a question such as
“What is the plurality voting system?”, it is natural
for a human to consult theToCand first narrow
down that it could be answered by the Chapter
“Democracies and Democratization”. Further sub-
sections reveal that perhaps the Section “Institu-
tions within Democracy” and the subsection “Elec-
toral Systems & Political Parties” could contain the
answer. Content for this subsection indeed contains
the definition for the plural voting system. Text
content associated withToCentries is semantically
coherent organized around a topic as captured by
arXiv:2609.03874v1  [cs.AI]  3 Sep 2026

Figure 1: A human can answer “What is the plurality voting system?” by looking at the table of contents and
picking Section 4.2.3 without even reading the book Introduction to Comparative Government and Politics.
the title of theToCentry. Further, aToCentry has
a clear semantic boundary with otherToCentries.
ToCas a unit of retrieval is a novel yet intuitive
way of organizing long context for retrieval.
In this work, we presentSTAIR(STructure
AwareInformationRetriever) forToCbased re-
trieval, inspired by how a human searches for infor-
mation usingToC, and the fact that LLMs can store
the entire corpus in the model’s parameters. Our
careful and thorough ablation studies reveal that
empowering an LLM withToChelps in two critical
dimensions - curbing hallucinations and general-
izing for content with a low number of training
examples.STAIRachieves a high Recall@1 score
of 82.6, which is is ( +7.4% ) better than its closest
baseline Differentiable Search Index (DSI) (Tay
et al., 2022). DSI needs to discover the book struc-
ture (indirectly) through the completion part of
training examples, whileSTAIRcan use exploit the
structure provided as part of the input to accurately
store and retrieve information from its model’s pa-
rameters.
We develop and release a new multi-domain
benchmarkSearchTome, made up of 18books of
varying sizes across 6domains, parsed and cleaned
to extract theirToCmapped to the text of the per-
tinent sections. We also release a set of train, dev
and test queries for each book as part of the bench-
mark, with goldToCentries labeled for each query
serving as the retrieval output. We believe this will
be a first step towards promoting further researchonToCbased retrieval, withSearchTomeserving
as a standard evaluation benchmark. We train and
evaluateSTAIRusingSearchTomeand compare
it with strong baseline retrievers like BM25 and
a fully fine-tuned Sequence-to-Sequence model,
such as DSI (Tay et al., 2022), which does not
utilizeToC. Our experimental results show that
STAIRsignificantly benefits from incorporating
ToCin its training and easily outperforms the tradi-
tional DSI. Our main contributions are as follows:
•We propose a novel direction leveraging the
Table of Contents (ToC) of a lengthy corpus,
like a book, as a valuable element for effi-
ciently indexing corpus knowledge and utiliz-
ingToCentries as effective retrieval units.
•We build and release a multi-domain and di-
verse benchmarkSearchTome, consisting of
18books from 6diverse domains for the task
ofToCretrieval
•We develop an LLM basedSTAIRand eval-
uate it onSearchTomeand compare it with
strong competitive baselines
2 Related Work
Efficient and precise retrieval is a critical re-
search problem in natural language processing,
with a wide range of applications such as se-
mantic search (Guha et al., 2003), open domain
question answering (Chen and Yih, 2020; Zhu
et al., 2021) and Retrieval Augmented Genera-
tion(RAG) (Lewis et al., 2020) to name a few.

With the rise of RAG applications, precise retrieval
has become increasingly important, particularly in
curbing hallucination in generative models. We
can broadly classify all retrieval techniques into
three main categories:
Dense Retrieval(Cai et al., 2021; Karpukhin
et al., 2020) represent a query and document
using a dense vector and compute the similar-
ity based on the distance between their vectors.
DPR (Karpukhin et al., 2020) uses dual encoder,
ColBERT (Khattab and Zaharia, 2020) represents
every token in the query and document using dense
vector and performs late interaction to score query
and document pairs. Recently, Sarthi et al. (2024)
introduced RAPTOR, a method for building a hier-
archical tree by recursively embedding, clustering,
and summarizing content from lengthy documents
to retrieve relevant text at various levels of abstrac-
tion. Our approach differs in two key aspects:
while RAPTOR uses dense retrievers for index-
ing and retrieval, we adopt model-based indexing.
Additionally, we enhance model-based indexing
by leveraging table-of-contents (ToC) structures,
unlike RAPTOR, which relies solely on dense re-
trievers to access information at multiple levels of
abstraction.
Learned Sparse Retrievalsystems use sparse
vector representation for a query and document.
Unlike traditional sparse retrieval methods like
BM25(Robertson and Zaragoza, 2009), which rely
on exact token matches, these models learn to rep-
resent queries and documents in a sparse high-
dimensional space, allowing for efficient lexical
matching. SPLADE (Hai et al., 2023) (Sparse
Lexical and Expansion Model for Information Re-
trieval) is a prominent example of such systems.
Several other models have been developed to en-
hance sparse retrieval, including SPLADEv2 (For-
mal et al., 2021), DeepCT(Dai and Callan, 2019),
uniCOIL(Lin and Ma, 2021) and DeepImpact (Bas-
net et al., 2024).
Model-based Indexing(Metzler et al., 2021)
such as Differential Search Index (DSI) (Tay et al.,
2022) embed knowledge of the entire corpus di-
rectly into its model parameters – greatly sim-
plifying the retrieval process.STAIRtakes this
paradigm one step further by learning to exploit a
rich hierarchal and semantic global structure within
the corpus.3 SearchTome: A new benchmark for
Table of Contents based retrieval
3.1 Breaking the Mold: Why a New
Benchmark Is Needed
There are multiple benchmarks proposed around
long context applications such as ContractNLI (Ko-
reeda and Manning, 2021) focused on NLI or
Scrolls (Shaham et al., 2022)with seven chal-
lenging tasks. However, there doesn’t exist any
benchmark for long context retrieval with any form
of structured view, which we hypothesize as the
key for precise retrieval. Benchmarks such as Gov-
Report (Huang et al., 2021), SummScreen (Chen
et al., 2022), and QM-Sum (Zhong et al., 2021)
focus on generating summaries of a length docu-
ment, whereas, Qasper (Dasigi et al., 2021), QuAL-
ITY (Pang et al., 2022) and NarrativeQA (Ko ˇciský
et al., 2018) focus on the generation aspect with
only answersfor a user query anddo not con-
tain the gold passage for retrieval. The closest
benchmark isLocoV1 (Saad-Falcon et al., 2024)
which does include gold passages for each query.
However, due tolack of ToC or a global struc-
ture for the input document, we cannot use this
benchmark for our task.
To the best of our knowledge,SearchTomeis the
first benchmark which provides a clean structured
view with Table of Contents for evaluating long
context retrieval with structures .
3.2 SearchTome: A diverse and novel
benchmark for ToC Retrieval
Our main motivation for a new benchmark is to
evaluate how well a global semantic structure such
as aToCcan help in better retrieval for long context
applications which is typical for technical reports or
voluminous books. Thus to create a clean and effec-
tive benchmark we turn to opentextbooks, which is
perhaps the largest collection of such textbooks. To
make the benchmark a comprehensive one across
domains, we picked six diverse domains namely –
Education, Finance, Law, Medicine, Natural Sci-
ences and Social Sciences and selected three books
from each domain (Table 1). We parsed the PDF
for each book and extracted Table of Contents us-
ing pymupdf1. A sample of the final cleaned con-
tent for one of the books can be viewed here. For
each paragraph we ask a powerful LLM Mixtral
8x7b model (Jiang et al., 2024) to generate multi-
ple questions covering all important topics in the
1https://pymupdf.readthedocs.io/en/latest/

Table 1:SearchTomeBook links & Statistics
Domain Book #Pages #Leaves Test Train Dev
EducationOpen Music Theory 1297 429 7430 4493 1671
The Whole Child: Development in the Early Years 182 129 1746 1056 388
Teaching in a Digital Age 779 118 6150 3797 1489
FinancePrinciples of Financial Accounting 318 98 4119 2395 981
Accounting in the Finance World 572 80 4481 2747 1081
Financial and Managerial Accounting 1077 107 7815 4692 1901
LawConstruction Contracting 403 103 3653 2249 866
Criminal Procedure 897 106 5610 3401 1349
Tort Law: Cases and Commentaries 948 396 18209 11174 4363
MedicineNursing Assistant 659 118 4586 2790 1093
Nursing Fundamentals 1327 121 9065 5493 2200
Nursing Management and Professional Concepts 599 74 3695 2277 885
Natural SciencesIntroduction to Genetics 513 70 2254 1345 527
Principles of Mechanics 179 125 2341 1418 543
Organic Chemistry 1249 321 11124 6601 2635
Social SciencesFoundations of Aural Skills 674 118 2451 1421 559
Introduction to Comparative Government and Politics 421 189 3893 2288 898
A Practicum in Behavioral Economics 381 178 4627 2821 1073
paragraph (Zhang et al., 2024). Following the same
technique as was proposed in DSI (Tay et al., 2022)
to train the parametric index, we use a portion of
these questions for training DSI (Tay et al., 2022)
andSTAIR, a small portion as development set
which is to help us pick the best checkpoint. Ma-
jority of the generated questions were picked as
test questions – to thoroughly test coverage of con-
tent by an IR system. We believe our benchmark
SearchTomewill further research in building IR
systems which can leverage structure in a corpus
and retrieve the most relevant chunks for a user
query.
4 STAIR: Table of Contents Searcher
We now formally define the problem statement and
the technique for retrieving a section title given a
book and its Table of Contents (ToC)
4.1 Notations
We are given a long document Dand its Table
of Contents ToCD={T 1, T2, ...., T n}. An edge
e:T p→T cbetween two nodes Tp,Tc∈ToC D
is defined as Tp= parent( Tc), if the section repre-
sented by title Tpis further divided into multiple
sub-sections including Tc. The set of leaf nodes
can thus be defined as LND={Tl∈ToC D|∃!Tc∈
ToCD, Tl=parent_of(T c)} ⊆ToC D. Our goal
is to retrieve the correct leaf node Tl∈LN Dwhose
content can answer a user queryq.4.2 Training STAIR
Figure 2 depicts the training pipeline forSTAIR
which learns two correlated butdifferenttasks:
Corpus knowledge ingestion: STAIRneeds to
learn the information trove Dby mapping a book’s
text content to the corresponding section titles
Ti∈ToC D. This is facilitated by the training
data, which consists of a series of queries that are
linked to their corresponding section titles. Conse-
quently, this phase of training integrates the book’s
knowledge into the LLM’s parameters, similar to
model-based indexing systems like DSI, but with
a key difference. Unlike other model-based index-
ing systems, the parameters inSTAIRremember
the corpus knowledge not by a document identifier,
but by using fine-grained section title entries in the
ToC, which are likely to have a much stronger se-
mantic connection with the content being ingested
into parametric knowledge.
Learning to generate from ToC: STAIRlearns
to pick the most relevant leaf node from the com-
plete global structure (ToC) of a document.
We use the training split fromSearchTometo
trainSTAIRthrough supervised fine-tuning. Dur-
ing training, we provide the query qalong with
the complete ToCDas input. Consequently, the
complete input during training consists ofprompt
+(q,ToC D). Note that ToCD, remains consis-
tent across queries from the same book, while the
prompt remains unchanged throughout the training
process. We fine-tune the model to generate the
correct leaf node Tl∈LN Dthat corresponds to

Stage 1 — Input Preparation
Stage 2 — TrainingStage 3 — Inference
Book (PDF/Text)
ToC Extraction
Synthetic QA GenerationConcatenateToC + Query
LLM Fine-tuning
Target:ToC Leaf NodeQuery Input ToC
ConcatenateToC + Query
Fine-tuned LLM
Constrained Generation
Predicted Leaf NodeFigure 2: Overview of our pipeline. Stage 1 prepares synthetic QA data from the input book. Stage 2 fine-tunes a
language model to map a ToC-augmented query to the correct leaf node. Stage 3 runs inference with constrained
generation, restricting the output vocabulary to valid ToC leaf nodes.
the queryq.
Table 2: Sample training example
Prompt
### Book: Teaching in a Digital Age
Pick the best section from the table of contents below
which can answer the user query below. Only generate
the section name, do not generate any explanation!
### Table of Contents
1 Chapter 1: Fundamental Change in Education
· · ·
leaf 12.7 Step five:master the technology
leaf 12.8 Step six:set appropriate learning goals
leaf 12.9 Step seven:design course structure and learning
· · ·
leaf 13.7 Building the future
### Query: What is the suggested time allocation
for students studying a course or program?
### Section: leaf
Completion
12.9 Step seven: design course structure and learning
5 Experiments
We report Recall@1 (R@1), Recall@3 (R@3)
and Normalized Discounted Cumulative Gain
(nDCG@3) metrics on the test set for each book.
BeIR (Thakur et al., 2021) is used for computing
the metrics.5.1 Baselines
Mistral(Out of the Box LLM): Beam search is
used to generate the Top- Kpredictions for Mistral
Instruct v0.2
BM25(Robertson and Zaragoza, 2009): We index
content using Elastic Search v 8.11.2. Leaf node
acts as a unique document identifier for a book
DPR: We use NV-Embed-v2(Lee et al., 2024) out
of the box with text lengths set to 512 for passages
and 256 for queries. NV-Embed-v2 is the leading
text-only embedding model with open weights on
the MTEB leaderboard2. The model is based on
Mistral-7B-v0.13, making it a suitable baseline for
STAIRas both are finetuned over Mistral.
DSI (Tay et al., 2022)infuses the entire knowledge
of a corpus in the parameters of an LLM and di-
rectly generates a document identifier. We fine-tune
DSI using the train split for each book.
5.2 Fine-tuning STAIR and DSI
We fine-tune Mistral Instruct v0.24(Jiang et al.,
2023) for a maximum of 200 epochs using a LoRA
adapter (Hu et al., 2021) with a r= 16 andα= 32 .
We early stop with a patience of 20 epochs, by
computing Recall@1 on the dev set. We set the
maximum input length to 14k tokens forSTAIR
2As per Jan 23’2025 MTEB Leaderboard Snapshot.
3https://huggingface.co/mistralai/Mistral-7B-v0.1
4https://huggingface.co/mistralai/Mistral-7B-Instruct-
v0.2

and 512tokens for DSI and a maximum output
length of 64 tokens. See Table 2 for a sample
training input and output forSTAIR.
6 Results and Analysis
Table 4 comparesSTAIRwith all the baselines
listed in Section 5.1 and addresses the following
two research questions:
•RQ1: DoesToCbased training makesSTAIR
more accurate as a retriever?
•RQ2: Can we finetune an LLM to learn the
new task of generating leaf nodes fromToC?
Our key findings are as follows:
1.STAIRoutperforms strong baselines such as
BM25, DPR and fine-tuned DSI
2.R@1 for Mistral is much lower than BM25 –
this suggests the LLM not only needs to learn
the new task of picking the best section but
also needs to ingest knowledge from the cor-
pus
3.DPR performs much better than BM25 but
worse than DSI. This result is expected, as NV-
Embed-v2 is not fine-tuned forSearchTome
4.Tuning withToCas input helpsSTAIR: The
only difference in the input between DSI and
STAIRis thatSTAIRleveragesToC.STAIR
outperforms DSI by 7.4% , as it is able to ac-
cess the entire structure of the book and better
align it with the queries during fine-tuning.
DSI on the other hand must learn the seman-
tic alignment between queries and sections by
looking at all the training data. We can thus
conclude thatRQ1andRQ2are answered in
affirmative.
Statistical significance testingwas done for the
Recall@1 performance difference between DSI and
STAIR. We follow the randomization test tailored
to retrieval systems as described in Smucker et al.
(2007). The null hypothesis for the significance
testing is that outputs from SystemA (DSI) and
SystemB (STAIR) may belong to the same under-
lying distribution and the difference of performance
between SystemA (DSI) and SystemB (STAIR) is
because of the sampling variance. We test the null
hypothesis with the traditional pvalue of 0.05 at
the domain level i.e. for each of the 6domains sep-
arately (by combining all the books from a single
domain). The significance test results conclude that
the difference in DSI andSTAIRis indeed statis-
tically significant for all the 6domains where the
null hypothesis is successfully invalidated. Morespecifically, through our sample runs for random-
ization test, we find that the probability of seeing
this scale of Recall@1 performance difference as
observed in Table 4 when being randomly sampled
from the same distribution falls below p= 0.05 .
As with any statistical significance testing, the key
takeaway from these experiments is a confirmation
that the performance improvement seen inSTAIR
is statistically significant and therefore, it is a prop-
erty of theSTAIRdesign and not because of the
data set size or data distribution inSearchTome.
6.1 Ablations
We conduct ablation studies to study the impact of
usingToCas an input. Two key conclusions are as
follows:
ToC reduces hallucinations: We define hallucina-
tion as the generation of a non-leaf node (invalid
document identifier). Figure 3 illustrates the num-
ber of hallucinations per leaf node. We observed
that the hallucination rate ofSTAIRremains nearly
constant at close to zero, regardless of the number
of training examples. This indicates thatSTAIR
has effectively learned to generate outputs solely
from the leaf nodes, making it inherently less prone
to hallucination. In contrast, DSI, which does not
useToC, exhibits a higher hallucination rate than
STAIR, particularly for leaf nodes with fewer train-
ing examples.
Remembering the structure is hard without ToC
Figure 4 shows that for leaf nodes with less num-
ber of training examples, the Recall@1 difference
between DSI andSTAIRis much higher. This
supports our hypothesis, that withoutToC, DSI
needs to discover the book structure and also re-
member it within its parameters. DSI attempts to
learn this via the training data on query and as-
sociatedToCleaf node pairs. Thus, leaf nodes
which have low representations in training data are
prone to be missed by DSI. In contrast,STAIR
does not need to “remember” the leaf nodes. As
the number of training examples for a leaf node
increase, Recall@1 gap between DSI andSTAIR
decreases – althoughSTAIRconsistently achieves
higher Recall@1 numbers.
6.2 Error Analysis with Anecdotal Examples
We observe that out-of-the-box Mistral’s error rate
is 86.20%, and 26.81% of its mistakes are as it
predicts a non-leaf node, while 23.86% are hal-
lucinations. This suggests that Mistral out-of-box
needs knowledge ingestion and task finetuning. For

Table 3: Examples of model predictions. Gold leaf is highlighted inboldin Table of Contents
Example 1:
Book: The Whole Child: Development in the Early Years
Query: How do preschoolers react when caregivers and teachers belittle their autonomous actions?
Table of Contents Content
· · ·
2 Chapter Two: Theorists and Theories of Development Initiative vs. Guilt (Preschool Years)
leaf 2.1 Theories of Development The development of courage and independence ...
2.2 Psychosocial Theory ...
leaf 2.2.1 Trust vs. Mistrust (Infancy) ...
leaf 2.2.2 Autonomy vs. Shame/Doubt (Toddlerhood) ...
leaf 2.2.3 Initiative vs. Guilt (Preschool Years)
leaf 2.2.4 Industry vs. Inferiority (Elementary Years) If caregivers and preschool teachers encourage and support
leaf 2.2.5 Identity vs. Role-Confusion (Adolescence) children’s efforts while also helping them make realistic
leaf 2.2.6 Intimacy vs. Isolation (Early Adulthood) and appropriate choices, children develop a healthy sense of initiative in
leaf 2.2.7 Generativity vs. Stagnation (Middle Adulthood) planning and undertaking activities If, instead, adults discourage the
leaf 2.2.8 Integrity vs. Despair (Older Adulthood) pursuit of independent activities or dismiss them as silly and
· · · bothersome, children develop guilt about their needs and desires.
3 Chapter Three: Domains in Development
Mistral: 8.3.3 Moral Development, leaf 7.3.2 Hitting/Scratching and Temper Tantrums,leaf 2.2.3 Initiative vs. Guilt (Preschool Years)
BM25: 8.3.2 Moral Development
DPR:2.2.2 Autonomy vs. Shame/Doubt (Toddlerhood)
DSI: 2.2.2 Autonomy vs. Shame/Doubt (Preschool Years)
STAIR:2.2.3 Initiative vs. Guilt (Preschool Years)
Example 2:
Book: Nursing Fundamentals
Query: In what way does a chronic illness affect an elderly person’s ability to perform daily activities?
Table of Contents Content
· · ·
leaf 30.5 Spiritual Care of Self Applying the Nursing Process
leaf 30.6 Putting It All Together ...
31 Care of the Older Adult It is also important to consider the impact of chronic disease
leaf 31.1 Care of the Older Adult Introduction on their ability to function and complete Activities of Daily Living
leaf 31.2 Basic Concepts (ADLs). older adults who are able to perform ADLs without assistance
leaf31.3 Applying the Nursing Process consider themselves healthy.
Mistral:31.3 Applying the Nursing Process
BM25: 11.2 Sensory Impairments Basic Concepts
DPR:31.2 Basic Concepts
DSI: 31.2 Basic Concepts
STAIR:31.3 Applying the Nursing Process
Example 3:
Book: Introduction to Comparative Government and Politics
Query: What are irregular armed organizations and how are they used by states?
Table of Contents Content
9 Chapter 5: Non-Democracies and Democratic Backsliding
· · · Another powerful instrument of repression are paramilitaries.
9.2 Strategies for staying in power These refer to groups with access to military-grade weapons and training
leaf 9.2.1Institutional channels yet they are not part of the national military.
leaf 9.2.2 Cultural and ideological controls They are “irregular armed organizations that carry out acts of violence
9.3 Varieties of non-democracy against civilians on behalf of a state,”
· · ·
Mistral: 9.3.5 Illiberal and hybrid regimes
BM25: 15.3.1 Insurgencies/Civil Wars
DPR: 15.2.2 External State-Sponsored Political Violence (State-Sponsored Terrorism)
DSI: 15.2.1 Internal State-Sponsored Political Violence (Government Terrorism)
STAIR:9.2.1 Institutional channels
DSI, the error rate drops to 24.31%, with 3.25%
of its predictions being non-leaf nodes (suggesting
knowledge infusion helps).STAIRhas the low-
est error rate of 18.67%, with only 0.05% of its
predictions being non-leaf nodes.
Table 3 lists a few qualitative examples. We dis-
cuss next the first example “How do preschoolers
react when caregivers and teachers belittle theirautonomous actions” from the book “The Whole
Child: Development in the Early Years”:
Mistralcould correctly guess that the query
can be answered by “2.2.3 Initiative vs. Guilt
(Preschool Years)” using its world knowledge. It
doesn’t understand that it needs to pick only one
option – due to lack of the task training
BM25suffers from the known bias of keyword

Table 4: Recall@1, Recall@3 and nDCG@3 forSearchTome; Haiku is Claude haiku 4.5
Education Finance Law Med NatSci SocSci Avg
Mistral 14.8/17.8/16.6 16.4/19.4/18.2 13.6/15.7/14.9 13.7/17.0/15.7 13.1/15.3/14.4 10.9/13.4/12.4 13.8/16.4/15.4
BM25 58.7/75.8/68.8 55.0/76.8/67.8 62.0/77.7/71.3 59.6/78.7/70.8 58.7/78.1/70.1 62.8/78.3/72.0 59.5/77.6/70.1
Haiku 59.8/80.4/72.0 37.7/56.3/48.4 41.1/60.6/52.3 46.1/65.0/57.2 54.6/73.7/65.7 34.0/48.8/42.6 45.6/64.1/56.4
DPR 71.5/86.8/80.6 66.4/85.5/77.7 69.6/84.1/78.2 72.7/88.8/82.2 63.0/84.1/75.7 68.8/83.1/77.3 68.7/85.4/78.6
DSI 76.2/84.1/80.9 78.1/87.8/83.9 73.3/82.9/79.0 82.1/89.9/86.8 76.2/84.7/81.3 75.4/82.2/79.4 76.9/85.3/81.9
STAIR 83.3/91.2/88.0 82.8/91.4/88.0 80.8/89.4/85.9 86.1/93.2/90.3 80.6/90.4/86.4 81.8/89.0/86.1 82.6/90.8/87.5
Figure 3: As number of training examples decrease,
hallucination rate for DSI increases
matching where and picks the section “8.3.2. Moral
Development” purely because of highest lexical
overlap with query tokens, which is the incorrect
answer
DPRis used out of the box and does a word sensed
matching where it predicts “2.2.2 Autonomy vs.
Shame/Doubt (Toddlerhood)” being most relevant
to the query tokens mentioning “belittling of au-
tonomous actions”. However, without the corpus
specific training it falls short of identifying the cor-
rect section.
DSIis finetuned for the corpus but faulters to
find the precise section. It errs just like DPR in
this example but might have chosen a more sen-
sible match “2.2.2 Autonomy vs. Shame/Doubt
(Preschool years)” which takes into account the
importance of keyword “preschool” for getting a
better topic match. However, careful consideration
shows this is a hallucinated header not present in
the inputToC. This correlates with our ablation in
figure 3 on DSI hallucination.
STAIR: Our proposed systemSTAIRcorrectly
identifies the gold leaf node as “Initiative vs. Guilt
(Preschool Years)”.
The comparison between DPR, DSI andSTAIRFigure 4: Recall@1 for each leaf acrossSearchTome.
STAIRshows high accuracy even with a low number
of training examples
is interesting as they all are built on top odf the
same base LLM – Mistral. The difference between
DSI andSTAIRlargely points towards the impor-
tance of havingToCas an input during training,
which makes it easier for the LLM to align much
better to the corpus. Other anecdotal examples in
Table 3 also enumerate the same partial ordering
seen among the baselines and provide more quali-
tative analysis to support the claim that the design
ofSTAIRhelps it do a much better retrieval.
7 Conclusion
We introduceSTAIR, a novel LLM based IR sys-
tem that leveragesToCto store and retrieve infor-
mation from its parameters. Given a query,STAIR
generates the most probably leaf section header
from aToCwhich could answer it. We use Mis-
tral Instruct v0.2 to demonstrate that it is possible
to instruction finetune the modern day LLMs to
efficiently use theToCstructure to generate the
correct leaf node while reducing the hallucinations
to almost zero.STAIRoutperforms all strong
baselines such as BM25, finetuned DSI and DPR
and achieves a Recall@1 score of 82.6% which
is around 7.4% gain over the next best system

(DSI). We release a new comprehensive bench-
markSearchTomeacross 6domains with 18books
comprising of train, dev and test splits to further
research in this novel direction.
As a future work, we want to explore directions
where aToClike structure is created dynamically
over an unseen search corpus, as has been proposed
by some previous works. We envision that having a
ToC-based retrieval paradigm will gain more trac-
tion in future for agentic frameworks needing multi-
hop retrieval and reasoning over retrieved context.
To that end, we want to developSTAIRto work
in a complete zero shot setup where it iteratively
retrievesToCleaf nodes and makes intelligent de-
cisions by reasoning on the content of the leaf node
to do precise information retrieval.
8 Limitations
Our current evaluation is limited to corpora where a
global structure exists. While this setting is suitable
for initial validation, it may not fully represent the
diversity of real-world use cases. In future work,
we plan to extend our evaluation to standard bench-
marks by artificially inducing a Table of Contents
structure. Additionally, we aim to test our model
on enterprise datasets where such a structure al-
ready exists at a very large scale (possibly millions
of URLs in the corpus).
References
Yushi Bai, Xin Lv, Jiajie Zhang, Hongchang Lyu,
Jiankai Tang, Zhidian Huang, Zhengxiao Du, Xiao
Liu, Aohan Zeng, Lei Hou, Yuxiao Dong, Jie Tang,
and Juanzi Li. 2024a. LongBench: A bilingual, mul-
titask benchmark for long context understanding. In
Proceedings of the 62nd Annual Meeting of the As-
sociation for Computational Linguistics (Volume 1:
Long Papers), pages 3119–3137, Bangkok, Thailand.
Association for Computational Linguistics.
Yushi Bai, Shangqing Tu, Jiajie Zhang, Hao Peng, Xi-
aozhi Wang, Xin Lv, Shulin Cao, Jiazheng Xu, Lei
Hou, Yuxiao Dong, Jie Tang, and Juanzi Li. 2024b.
Longbench v2: Towards deeper understanding and
reasoning on realistic long-context multitasks.arXiv
preprint arXiv:2412.15204.
Soyuj Basnet, Jerry Gou, Antonio Mallia, and
Torsten Suel. 2024. Deeperimpact: Optimizing
sparse learned index structures.arXiv preprint
arXiv:2405.17093.
Yinqiong Cai, Yixing Fan, Jiafeng Guo, Fei Sun, Ruqing
Zhang, and Xueqi Cheng. 2021. Semantic models
for the first-stage retrieval: A comprehensive review.ACM Transactions on Information Systems (TOIS),
40:1 – 42.
Danqi Chen and Wen-tau Yih. 2020. Open-domain
question answering. InProceedings of the 58th An-
nual Meeting of the Association for Computational
Linguistics: Tutorial Abstracts, pages 34–37, Online.
Association for Computational Linguistics.
Mingda Chen, Zewei Chu, Sam Wiseman, and Kevin
Gimpel. 2022. SummScreen: A dataset for abstrac-
tive screenplay summarization. InProceedings of the
60th Annual Meeting of the Association for Compu-
tational Linguistics (Volume 1: Long Papers), pages
8602–8615, Dublin, Ireland. Association for Compu-
tational Linguistics.
Shouyuan Chen, Sherman Wong, Liangjian Chen, and
Yuandong Tian. 2023. Extending context window of
large language models via positional interpolation.
arXiv preprint arXiv:2306.15595.
Zhuyun Dai and Jamie Callan. 2019. Context-aware
sentence/passage term importance estimation for first
stage retrieval.arXiv preprint arXiv:1910.10687.
Pradeep Dasigi, Kyle Lo, Iz Beltagy, Arman Cohan,
Noah A. Smith, and Matt Gardner. 2021. A dataset
of information-seeking questions and answers an-
chored in research papers. InProceedings of the
2021 Conference of the North American Chapter of
the Association for Computational Linguistics: Hu-
man Language Technologies, pages 4599–4610, On-
line. Association for Computational Linguistics.
Thibault Formal, Carlos Lassance, Benjamin Pi-
wowarski, and Stéphane Clinchant. 2021. Splade
v2: Sparse lexical and expansion model for informa-
tion retrieval.arXiv preprint arXiv:2109.10086.
R. Guha, Rob McCool, and Eric Miller. 2003. Seman-
tic search. InProceedings of the 12th International
Conference on World Wide Web, WWW ’03, page
700–709, New York, NY , USA. Association for Com-
puting Machinery.
Nam Le Hai, Thomas Gerald, Thibault Formal, Jianyun
Nie, Benjamin Piwowarski, and Laure Soulier. 2023.
Cosplade: Contextualizing splade for conversational
information retrieval.ArXiv, abs/2301.04413.
J. Edward Hu, Yelong Shen, Phillip Wallis, Zeyuan
Allen-Zhu, Yuanzhi Li, Shean Wang, and Weizhu
Chen. 2021. Lora: Low-rank adaptation of large
language models.ArXiv, abs/2106.09685.
Luyang Huang, Shuyang Cao, Nikolaus Parulian, Heng
Ji, and Lu Wang. 2021. Efficient attentions for long
document summarization. InProceedings of the 2021
Conference of the North American Chapter of the
Association for Computational Linguistics: Human
Language Technologies, pages 1419–1436, Online.
Association for Computational Linguistics.

Albert Q Jiang, Alexandre Sablayrolles, Antoine
Roux, Arthur Mensch, Blanche Savary, Chris Bam-
ford, Devendra Singh Chaplot, Diego de las Casas,
Emma Bou Hanna, Florian Bressand, and 1 oth-
ers. 2024. Mixtral of experts.arXiv preprint
arXiv:2401.04088.
Albert Qiaochu Jiang, Alexandre Sablayrolles, Arthur
Mensch, Chris Bamford, Devendra Singh Chap-
lot, Diego de Las Casas, Florian Bressand, Gi-
anna Lengyel, Guillaume Lample, Lucile Saulnier,
L’elio Renard Lavaud, Marie-Anne Lachaux, Pierre
Stock, Teven Le Scao, Thibaut Lavril, Thomas Wang,
Timothée Lacroix, and William El Sayed. 2023. Mis-
tral 7b.ArXiv, abs/2310.06825.
Vladimir Karpukhin, Barlas O ˘guz, Sewon Min, Patrick
Lewis, Ledell Yu Wu, Sergey Edunov, Danqi
Chen, and Wen tau Yih. 2020. Dense passage re-
trieval for open-domain question answering.ArXiv,
abs/2004.04906.
O. Khattab and Matei A. Zaharia. 2020. Colbert: Effi-
cient and effective passage search via contextualized
late interaction over bert.Proceedings of the 43rd
International ACM SIGIR Conference on Research
and Development in Information Retrieval.
Tomáš Ko ˇciský, Jonathan Schwarz, Phil Blunsom, Chris
Dyer, Karl Moritz Hermann, Gábor Melis, and Ed-
ward Grefenstette. 2018. The NarrativeQA reading
comprehension challenge.Transactions of the Asso-
ciation for Computational Linguistics, 6:317–328.
Yuta Koreeda and Christopher Manning. 2021. Con-
tractNLI: A dataset for document-level natural lan-
guage inference for contracts. InFindings of the
Association for Computational Linguistics: EMNLP
2021, pages 1907–1919, Punta Cana, Dominican Re-
public. Association for Computational Linguistics.
Philippe Laban, A. R. Fabbri, Caiming Xiong, and
Chien-Sheng Wu. 2024. Summary of a haystack:
A challenge to long-context llms and rag systems. In
Conference on Empirical Methods in Natural Lan-
guage Processing.
Chankyu Lee, Rajarshi Roy, Mengyao Xu, Jonathan
Raiman, Mohammad Shoeybi, Bryan Catanzaro, and
Wei Ping. 2024. Nv-embed: Improved techniques for
training llms as generalist embedding models.arXiv
preprint arXiv:2405.17428.
Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio
Petroni, Vladimir Karpukhin, Naman Goyal, Hein-
rich Küttler, Mike Lewis, Wen-tau Yih, Tim Rock-
täschel, and 1 others. 2020. Retrieval-augmented
generation for knowledge-intensive nlp tasks.Ad-
vances in Neural Information Processing Systems,
33:9459–9474.
Tianle Li, Ge Zhang, Quy Duc Do, Xiang Yue, and
Wenhu Chen. 2024. Long-context llms struggle with
long in-context learning.ArXiv, abs/2404.02060.Jimmy Lin and Xueguang Ma. 2021. A few brief notes
on deepimpact, coil, and a conceptual framework
for information retrieval techniques.arXiv preprint
arXiv:2106.14807.
Hao Liu, Wilson Yan, Matei Zaharia, and Pieter Abbeel.
2024a. World model on million-length video and lan-
guage with blockwise ringattention.arXiv preprint
arXiv:2402.08268.
Kai Liu, Ze Chen, Zhihang Fu, Rongxin Jiang, Fan
Zhou, Yaowu Chen, Yue Wu, and Jieping Ye. 2024b.
Structure-aware domain knowledge injection for
large language models.
Nelson F. Liu, Kevin Lin, John Hewitt, Ashwin Paran-
jape, Michele Bevilacqua, Fabio Petroni, and Percy
Liang. 2024c. Lost in the middle: How language
models use long contexts.Transactions of the Asso-
ciation for Computational Linguistics, 12:157–173.
Donald Metzler, Yi Tay, Dara Bahri, and Marc Najork.
2021. Rethinking search: Making experts out of
dilettantes.ArXiv, abs/2105.02274.
Richard Yuanzhe Pang, Alicia Parrish, Nitish Joshi,
Nikita Nangia, Jason Phang, Angelica Chen, Vishakh
Padmakumar, Johnny Ma, Jana Thompson, He He,
and Samuel Bowman. 2022. QuALITY: Question
answering with long input texts, yes! InProceedings
of the 2022 Conference of the North American Chap-
ter of the Association for Computational Linguistics:
Human Language Technologies, pages 5336–5358,
Seattle, United States. Association for Computational
Linguistics.
Nicholas Pipitone and Ghita Houir Alami. 2024.
Legalbench-rag: A benchmark for retrieval-
augmented generation in the legal domain.ArXiv,
abs/2408.10343.
Stephen Robertson and Hugo Zaragoza. 2009. The
probabilistic relevance framework: Bm25 and be-
yond.ACM Transactions on Information Systems,
27(3):1–27.
Jon Saad-Falcon, Daniel Y . Fu, Simran Arora, Neel
Guha, and Christopher R’e. 2024. Benchmarking
and building long-context retrieval models with loco
and m2-bert.ArXiv, abs/2402.07440.
SAP. 2024. Sap-product-documentation. https:
//help.sap.com/doc/saphelp_snc70/7.0/
en-US/ef/c08d5377a0ec23e10000000a174cb4/
frameset.htm.
Parth Sarthi, Salman Abdullah, Aditi Tuli, Shubh
Khanna, Anna Goldie, and Christopher D Man-
ning. 2024. Raptor: Recursive abstractive pro-
cessing for tree-organized retrieval.arXiv preprint
arXiv:2401.18059.
SEC. 2024. Sec 10k structure. https://www.sec.
gov/files/reada10k.pdf.

Spurthi Setty, Katherine Jijo, Eden Chung, and Natan
Vidra. 2024. Improving retrieval for rag based
question answering models on financial documents.
ArXiv, abs/2404.07221.
Uri Shaham, Elad Segal, Maor Ivgi, Avia Efrat, Ori
Yoran, Adi Haviv, Ankit Gupta, Wenhan Xiong,
Mor Geva, Jonathan Berant, and Omer Levy. 2022.
SCROLLS: Standardized CompaRison over long lan-
guage sequences. InProceedings of the 2022 Con-
ference on Empirical Methods in Natural Language
Processing, pages 12007–12021, Abu Dhabi, United
Arab Emirates. Association for Computational Lin-
guistics.
Mark Smucker, James Allan, and Ben Carterette. 2007.
A comparison of statistical significance tests for in-
formation retrieval evaluation. pages 623–632.
Yi Tay, Vinh Q. Tran, Mostafa Dehghani, Jianmo Ni,
Dara Bahri, Harsh Mehta, Zhen Qin, Kai Hui, Zhe
Zhao, Jai Gupta, Tal Schuster, William W. Cohen,
and Donald Metzler. 2022. Transformer memory as a
differentiable search index.ArXiv, abs/2202.06991.
Nandan Thakur, Nils Reimers, Andreas Rücklé, Ab-
hishek Srivastava, and Iryna Gurevych. 2021. BEIR:
A heterogeneous benchmark for zero-shot evaluation
of information retrieval models. InThirty-fifth Con-
ference on Neural Information Processing Systems
Datasets and Benchmarks Track (Round 2).
Wikipedia. 2024. wikipedia sample layout.
https://en.wikipedia.org/wiki/Wikipedia:
Manual_of_Style/Layout.
Jiaxin Zhang, Wendi Cui, Yiran Huang, Kamalika Das,
and Kumar Sricharan. 2024. Synthetic knowledge in-
gestion: Towards knowledge refinement and injection
for enhancing large language models. InConference
on Empirical Methods in Natural Language Process-
ing.
Ming Zhong, Da Yin, Tao Yu, Ahmad Zaidi, Mutethia
Mutuma, Rahul Jha, Ahmed Hassan Awadallah, Asli
Celikyilmaz, Yang Liu, Xipeng Qiu, and Dragomir
Radev. 2021. QMSum: A new benchmark for query-
based multi-domain meeting summarization. InPro-
ceedings of the 2021 Conference of the North Amer-
ican Chapter of the Association for Computational
Linguistics: Human Language Technologies, pages
5905–5921, Online. Association for Computational
Linguistics.
Fengbin Zhu, Wenqiang Lei, Chao Wang, Jianming
Zheng, Soujanya Poria, and Tat-Seng Chua. 2021.
Retrieving and reading: A comprehensive survey on
open-domain question answering.arXiv e-prints,
pages arXiv–2101.