# Adapting Large Language Models for Multi-Domain Retrieval-Augmented-Generation

**Authors**: Alexandre Misrahi, Nadezhda Chirkova, Maxime Louis, Vassilina Nikoulina

**Published**: 2025-04-03 09:03:40

**PDF URL**: [http://arxiv.org/pdf/2504.02411v1](http://arxiv.org/pdf/2504.02411v1)

## Abstract
Retrieval-Augmented Generation (RAG) enhances LLM factuality, but
multi-domain applications face challenges like lack of diverse benchmarks and
poor out-of-domain generalization. The first contribution of this work is to
introduce a diverse benchmark comprising a variety of question-answering tasks
from 8 sources and covering 13 domains. Our second contribution consists in
systematically testing out-of-domain generalization for typical RAG tuning
strategies. While our findings reveal that standard fine-tuning fails to
generalize effectively, we show that sequence-level distillation with
teacher-generated labels improves out-of-domain performance by providing more
coherent supervision. Our findings highlight key strategies for improving
multi-domain RAG robustness.

## Full Text


<!-- PDF content starts -->

March, 2025
Adapting Large Language Models for
Multi-Domain Retrieval-Augmented-Generation
Alexandre Misrahi1*Nadezhda Chirkova2Maxime Louis2Vassilina Nikoulina2
1EPFL2NAVER LABS Europe
General
Biomedical WebSearch Ctx-critical Long FormAvgVanilla RAG
 
Full-LoRA
LoRA-QKAtt
LoRA-Att
LoRA-MLP
 
Full-LoRA
LoRA-QKAtt
LoRA-Att
LoRA-MLP0.53 0.59 0.53 0.44 0.52 0.51
0.65 0.63 0.68 0.41 0.42 0.50
0.62 0.63 0.59 0.44 0.49 0.53
0.65 0.62 0.62 0.40 0.41 0.49
0.66 0.63 0.67 0.41 0.41 0.49
0.64 0.67 0.70 0.49 0.61 0.58
0.63 0.66 0.69 0.48 0.59 0.60
0.64 0.67 0.69 0.49 0.60 0.61
0.64 0.67 0.70 0.50 0.61 0.61LoRA
LoRA + distilled labelsLlama-3.2-1B
General
Biomedical WebSearch Ctx-critical Long FormAvg0.65 0.69 0.75 0.49 0.61 0.61
0.71 0.70 0.82 0.49 0.49 0.58
0.70 0.68 0.80 0.54 0.58 0.62
0.70 0.70 0.83 0.50 0.47 0.57
0.71 0.69 0.83 0.49 0.49 0.58
0.66 0.72 0.80 0.54 0.67 0.66
0.66 0.71 0.80 0.53 0.66 0.65
0.66 0.71 0.79 0.54 0.66 0.65
0.66 0.71 0.79 0.54 0.67 0.66LoRA
LoRA + distilled labelsLlama-3-8B
Figure 1: RAG adaptation results, LLMEval. Color is relative to the corresponding column. Average column is
computed across all individual datasets.
Abstract
Retrieval-Augmented Generation (RAG) enhances LLM factuality, but multi-domain applications face
challenges like lack of diverse benchmarks and poor out-of-domain generalization. The first contribution
of this work is to introduce a diverse benchmark comprising a variety of question-answering tasks
from 8 sources and covering 13 domains. Our second contribution consists in systematically testing
out-of-domain generalization for typical RAG tuning strategies. While our findings reveal that standard
fine-tuningfailstogeneralizeeffectively, weshowthatsequence-leveldistillationwithteacher-generated
labels improves out-of-domain performance by providing more coherent supervision. Our findings
highlight key strategies for improving multi-domain RAG robustness.
1. Introduction
Retrieval-AugmentedGeneration(RAG)[ 8,20]isatech-
nique that enhances Large Language Models (LLMs) by
retrieving relevant information from an external docu-
ment repository to augment the input prompt. Often
likened to an “open book” exam, this approach allows
LLMs to ground their responses in external knowledge,
thereby improving their faithfulness. RAG is especially
valuable for out-of-domain queries—those beyond an
LLM’s general knowledge—where the model must rely
on its ability to exploit retrieved context.
*Work done during internship at NAVER LABS EuropeWhile RAG is sometimes viewed as an alternative to
supervised finetuning on domain-specific data, and has
shownstrongperformanceacrossvariousdomains[ 29],
recent work [ 21,29,47] has highlighted the benefits
of finetuning LLMs specifically for RAG. This not only
improves the model’s ability to use retrieved context
effectively, but also increases robustness to irrelevant
information. Our goal in this paper is to investigate
RAG finetuning for out-of-domain tasks at test time.
Existing studies on multi-domain RAG [ 9,10,14,41,47]
overlook two key aspects. First, existing multi-domain
RAG benchmarks lack diversity and offer only limited
Corresponding author(s): alexandre.misrahi@epfl.ch, vassilina.nikoulina@naverlabs.comarXiv:2504.02411v1  [cs.CL]  3 Apr 2025

Adapting Large Language Models for
Multi-Domain Retrieval-Augmented-Generation
insights into the multi-domain challenges. Second, to
the best of our knowledge, none of the research per-
formedinadaptingLLMsforRAGconsiderdomain-shift
settings, when test-time queries or document reposito-
ries can differ from the ones used at training time in
the style or domain.
Our primary contribution is a carefully curated and
highly diverse set of query datasets and correspond-
ing document collections (datastores) covering mul-
tiple domains, task types and answer formats1. This
resource enables a precise assessment of the limitations
of the RAG pipeline and facilitates an analysis of the
impact of retrievers, rerankers, and LLMs on overall
performance. Our experimental study shows that per-
formance improvement brought by RAG compared to
zero-shot LLM prompting varies across domains, due to
multiple factors such as variable complexity and genre
of questions, datastore preprocessing, or retrieval per-
formance.
Our second contribution is studying cross-domain
generalization of various LLM finetuning strategies
for RAG. We show that commonly used tuning of an
LLM for RAG on standard question answering datasets
(with short labels) does not enable models to effectively
use context in the presence of domain shift. Our main
finding is that general-domain sequence-level distilla-
tion using teacher-generated labels improves out-of-
domain performance, likely due to the greater coher-
ence of teacher-generated answers compared to ground
truth labels.
Finally,we perform an in-depth analysis of results ,
looking at various RAGChecker metrics [ 33] across mul-
tipledatasetstobetterunderstandtheeffectofdifferent
RAG adaptation techniques and support our hypothesis.
2. Related work
Multi-domain RAG benchmarks. While some pa-
pers propose benchmarks for evaluating retrieval-
augmented LLMs in multi-domain settings [ 9,10,41,
47], we argue that existing multi-domain RAG bench-
marks lack domain diversity and do not capture a wide
variability of real-world scenarios where RAG can be
useful. In particular, existing benchmarks usually in-
clude a small number of domains, crafted from a lim-
ited set of sources (usually 1–3 sources), and often
focus on a particular kind of answers, e.g. only include
long-form answers. In our work, we introduce a di-
verse multi-domain RAG benchmark, which includes
data from 8 sources, covers 13 domains in total, and
1These datasets and document collections have been integrated
into the Bergen benchmarking library for RAG.contains questions and answers of various types and
genres.
Out-of-Domain Generalization in NLP. Generaliza-
tionofNLPpipelinesisanactiveareaofstudy[ 1,38,40].
Yang et al. [44]examines out-of-distribution (OOD)
generalization of pretrained LLMs across domains for
different tasks, and observe a significant gap between
LLM and human OOD performance. In [ 4], the authors
evaluate domain robustness of models and in particular
show that zero-shot LLMs demonstrate superior cross-
domain robustness compared to fine-tuned models –
we will show similar conclusions for RAG tasks.
Techniques for Improving RAG Pipelines. Recent ef-
fortstoimproveRAGpipelineperformancehavefocused
on optimizing both retrieval and generation compo-
nents. RA-DIT [ 21] jointly tunes the retriever and gen-
erator, enhancing overall performance. Instruct-Retro
[37]introducesatwo-stageapproachbycontinuingpre-
training with RAG, followed by an instruction-tuning
phase, which improves post-instruction tuning perfor-
mance. RAFT [ 47] has shown that finetuning an LLM
forRAGonoracledocumentsmixedwithnoisycontexts
improves robustness at test time. Authors claim better
usage of context is enabled by Chain-of-Thought (CoT)
augmentation of the training data. However, such CoT
augmentation requires processing whole set of samples
in training data, and also expects oracle documents to
be available at training time which might be true only
for a very limited set of datasets. While these methods
have demonstrated success, they are usually tested only
in-domain, i.e. the training and the testing data come
from the same distribution. In contrast, in our work,
we study cross-domain generalization of various LLM
adaptation techniques for RAG.
3. Multidomain RAG Benchmark
We address the limitations of existing out-of-domain
benchmarks mentioned in §2 by building a new diverse
benchmark suite for RAG. Specifically, we include mul-
tiple domains, multiple formats of answers and ques-
tions, multiple collections of documents from various
sources, and sets where the relative importance of the
context is variable. The selected Question Answering
(QA) datasets and associated document collections are
summarized in Table 1 and described below.
Biomedical QA. The BioASQ challenge [ 26] datasets
from 2023 and 2024 consist in biomedical QA. We fur-
ther contextualize BioASQ queries with the most rele-
vant of∼58M PubMed abstracts. The biomedical do-
mainalsoincludesCovidQA[ 24],adatasetoflong-form
2

Adapting Large Language Models for
Multi-Domain Retrieval-Augmented-Generation
Cat.Dataset Description Query Source Datastore Source N queries Example query & documentBIOMEDICALBioasq11b Biomedical ques-
tions of typesfac-
toid/yesno/listTask B, 2023
[26]
PubMed abstracts3837Doc:Biochemical purification of pseudopodia from migratory
cells.: Cell migration requires [...]
Query: Is it possible to purify pseudopodia to be used for
proteomic analysis?
Label:’yes’
Bioasq12b Biomedical ques-
tions of typesfac-
toid/yesno/listTask B, 2024
[27]940Doc:Deubiquitinating function of ataxin-3: insights from the
solution structure of the Josephin domain.: Spinocerebellar ataxia
type 3 is a human neurodegenerative disease [...]
Query: Which is the protein implicated in Spinocerebellar ataxia
type 3?
Label:Ataxin-3
CovidQA Short and long-
form QA related
to covid-19CovidQA
[24]CORD-19 [39] 2019Doc:Hantaviruses in the Americas and Their Role as Emerging
Pathogens [...]
Query: Typically how long do the hamsters die post-inoculation?
Label:11 and 14-dLONG-FORMFiQA Fact- and
Opinion-based
QA for financeTask 2 of
FiQA challenge
SourceBeIR (corpus)
[36]2561Doc:There are benefits associated with a cash only business (the
link states a few). However [...]
Query: A merchant requests that checks be made out to "Cash".
Should I be suspicious?
Label:There are benefits associated with a cash only business (the
link states a few). However [...]
Lifestyle E.g. cooking, nu-
trition, everyday
tasks.
Source
[9] LoTTE (StackExchange)2198Doc:A clove is not a clove. There are 2 main types of garlic:
Hard Neck: hard neck garlic varieties [...]
Query: what is a clove of garlic?
Label:Each wedge of a garlic is known as a clove and the whole
garlic is referred to as a head.
Recreation E.g. various
video games.2090Doc:Technically - you do not need to purchase anymore games.
However, you will need to purchase and download [...]
Query: if i buy garrys mod, will i need other games to play it?
Label:You don’t have to have any other games in order to
actually play GMod. [...]
Science E.g. math,
physics, biology.1404Doc:(as a limit of partial sums). Now, when people then say stuff
like1+2+3+···=−1/12they will [...]
Query: why does 1+2+3+···=−1
12?
Label:There are numerous methods to determine that a particular
result is correct. [...]
Technology E.g. security,
hardware, soft-
ware.2064Doc:The easiest way to have the Finder refresh its listing is to
enter a subfolder [...]
Query: is there a way to refresh a finder file listing?
Label:There are a number of approaches, one is to use a simple
AppleScript [...]
Writing E.g. syntax,
grammar, vocab-
ulary.2694Doc:Fact and fairy aren’t etymologically related, but John
Lawler’s answer is [...]
Query: etymology of fairy
Label:The "-ry" part of the word fairy is derived from the
"-ery" suffix [...]WEB-SEARCHSearchQA General QA with
context from
search engineSource
[6]Search Engine Results 21613Doc:Jun 24, 2010 ... Good King Wenceslas was murdered by his
brother. [...]
Query: This "Good King" of Bohemia was killed by his
brother Boleslav while on the way to mass
Label:WenceslasCONTEXT—CRITICALParaphraseRC Short, fact-based
movie QA (pre-
pended with
movie name).ParaphraseRC
[34]Movie plots 13111Doc:On the Path: Luna and Amar are a young Bosnian couple
living in Sarajevo. [...]
Query: On the Path: Why did Amar loses his job for being at
work?
Label:Amar loses his job for being drunk at work
SyllabusQA Short questions
about course lo-
gisticsSource
[7]Courses syllabi 957Doc:MUSIC-ED 500KU Syllabus S23: l be weighted as follows:
Assignment Percentage of Final Grade Midpoint Reports [...]
Query: UMASS-Math3312023: How many credits will I earn for
this course?
Label:No/insufficient information
TechQA Technical sup-
port queriesSource
[5]IBM Technotes 621Doc:IBM Solving IBM MQ Java code version mismatches using
the mqjavalist.sh script [...]
Query: How do I tell when there are mismatched MQ jars in my
application server? [...]
Label:Finding and eliminating duplicate copies of the MQ jar
files can be a difficult task [...]
Table 1: Multidomain RAG benchmarks
The datastores for ParaphraseRC, CORD-19, LoTTE are processed in chunks of 100 words with 20 words overlap
between consecutive chunks. The datastores for SyllabusQA and TechQA are processed in chunks of 1000 characters
with 200 characters overlap between consecutive chunks. For SyllabusQA and ParaphraseRC, each chunk is pre-pended
with the document title (course title and movie title, respectively).
3

Adapting Large Language Models for
Multi-Domain Retrieval-Augmented-Generation
0.00.20.40.60.8KILT-NQ BioASQ-12b
Llama-3.2-1BLlama-3-8BMistral-7B0.00.20.40.60.8T echQA
Llama-3.2-1BLlama-3-8BMistral-7BFiQAbm25
retromaebge
spladeoracle
+ Reranker
Figure 2: LLMEval scores for three LLM generators
evaluated across four datasets and multiple retriev-
er/reranker configurations. Oracle documents shown
where available.. Full results given in Appendix Table
11.
QA about the Covid-19 pandemic. The datastore for
CovidQA is the CORD-19 dataset. Compared to BioASQ,
this dataset requires more specific biomedical knowl-
edge, and in general we observe lower absolute metrics
on CovidQA compared to BioASQ. In the biomedical
setting, the robustness challenge stems from rare, out-
of-distribution tokens.
Web-Search QA. SearchQA is a question-answering
dataset augmented with text snippets retrieved from a
search engine. The difficulty in this dataset stems from
the convoluted phrasing of the queries posed as “Trivia”
questions. For example: “Slow down, as in a car” yields
gold label “decelerate”.
Context-critical QA. ParaphraseRC consists of short
questions about movies, with datastores consisting of
movie plot chunks. SyllabusQA is made of short ques-
tions about university course syllabi. TechQA corre-
sponds to the highly specialized domain of company-
and tool-specific forums. All of these domains contain
queries that can only be replied to with the correct
information from the relevant documents and cannot
be replied to from the internal LLM knowledge alone.
These domains are particularly helpful for evaluating
the retriever and generator’s ability to extract relevant
information from noisy context. For example, with the
query “CS-101: How is this course graded?” in Syl-
labusQA, the retriever may retrieve chunks relevant
either to the course CS-101, or chunks relevant to how
certain courses are graded. In the latter case, the task
mightbechallengingforthegeneratortocorrectlyiden-tify the chunk specific to CS-101, especially in the zero-
shot setting (no fine-tuning on SyllabusQA).
Long-Form QA. Finally, we adopt several long-form
question answering datasets which are more open-
ended QA tasks. This includes the following domains
from the RobustQA collection [ 9]: Finance, Lifestyle,
Recreation, Science, Technology, Writing.
3.1. Experimental settings
We use BERGEN2[32] to benchmark multi-domain
RAG on QA tasks.
General setup. Unless otherwise specified, we use
SPLADE-v3 [19] retriever to identify a first set of rele-
vantdocumentsgivenaquery. Thesedocumentsarefur-
ther reranked using DeBERTa-v3 [11], a cross-encoder
computing relevance score for each document relative
to the query. For generation, we use the instruct ver-
sions of Llama-3.2-1B3, Llama-3-8B4and Mistral-7B5.
Evaluation. To evaluate generated responses, we
mostly use LLM evaluation, denoted as LLMEval6, but
we also consider Match7and Recall8. LLMEval was
shown to be better correlate with GPT-4 evaluations
than match-based metrics in [ 32]. Similar to Match
metrics, LLMeval is convenient to interpret since it re-
ports the portion of successful replies. LLMEval is par-
ticularly useful for comparing long generations/ground
truth answers, since Match will always output the zero
evaluation result, and Recall is highly impacted by com-
mon words, and hard to interepret. Details about the
LLMEval, as well as some examples of evaluation, are
given in Appendix E.
4. Zero-Shot RAG results
4.1. RAG robustness to the context.
First,weanalysehowgooddifferentLLMsareatexploit-
ing context in RAG settings. Figure 2 compares how dif-
ferentretrieversandrerankersimpactqualityofanswer
generation. We add oracle documents when available
to assess the performance of the retrieval pipeline. We
select four datasets representing different domains doc-
2https://github.com/naver/bergen
3meta-llama/Llama-3.2-1B-Instruct
4meta-llama/Meta-Llama-3-8B-Instruct
5Mistral-7B-Instruct
6LLMEval prompts an open-source LLM to output a binary judg-
ment about thecorrectness of the generated response, given the input
question and the ground truth labels.
7Match measures if any of the ground truth labels is contained
verbatim in the generated response as a substring .
8Recall measures a percentage of words from the ground truth
labels that are contained verbatim in the generated response.
4

Adapting Large Language Models for
Multi-Domain Retrieval-Augmented-Generation
umentedinthetable1: (1)NQforgeneraldomain[ 17],
(2) BioASQ-12b for biomedical domain, (3) TechQA for
Context-critical and (4) FiQA for Long-form.
Effect of retriever/reranker. For all domains, we
observe systematic performance gains with stronger
retrievers, i.e. model-based retrievers vs bm25. Fur-
thermore, the second stage reranking improves perfor-
mance for most of the cases. The behaviour is similar
bothforthesmallmodel(Llama3.2-1B)andlargermod-
els (Llama3-8B, Mistral-7B). At the same time, even
with oracle documents, none of the models are able to
fully exploit relevant context, as there is still 20%-30%
gap in performance to reach the perfect answer, which
highlights importance of LLM adaptation.
Importance of top-k documents. Figure 3 compares
how different LLMs handle the increased amount of
retrieved documents provided in the context. All mod-
els benefit from larger number of documents when
assessed with LLMEval. In addition to varied top-k doc-
uments, we also assess LLM ability to identify relevant
content when distractor documents are added in the
context. We note that large models (Llama3-8B and
Mistral-7B) are relatively robust to the noise introduc-
tion, however the small model (Llama3.2-1B) is more
sensitive to noise.
4.2. RAG domain robustness
Table 2 reports RAG performance (LLMeval) across var-
ious domains on off-the-shelf LLMs which are not fine-
tuned for RAG tasks (Recall metric shown in Appendix
Table 10). First, we observe that the use of RAG im-
proves performance compared to no-RAG on almost all
datasets and for both Llama-1B and Llama-8B. Second,
gains with RAG are about equivalent for Llama-1B and
Llama-8B, except on the Long-form generation tasks,
where Llama-8B benefits significantly more (+14% ver-
sus +3% in LLMEval for Llama-1B) from the addition
of the retrieved context. On the long-form tasks, a good
generation must be able to use most parts of the pro-
vided context, and not simply identify a specific fact
as for short answers. This is a likely reason as to why
the larger model benefits more from the provided RAG
context. Third, improvements from RAG are heteroge-
neous across the different datasets and tasks: it brings
+14% for biomedical tasks, +20% for Long-form tasks
but only 9% for Context-Critical tasks for Llama-8B,
and similar relative gains for Llama-1B. This indicates
different generalization abilities of the RAG pipeline
to the different domains. Additional results, confirm-
ing these conclusions, on other LLMs can be found in
appendix Table 9a & 9b.
0.40.6Recall
zero-shot
top-1
top-3
top-5
top-10
top-200.50.60.70.8LLMeval
Llama-1B
Llama-8B
Mistral-7B
top-1 + 4D
top-3 + 2DFigure 3: BioASQ-12b across Top-k Retrieved Docu-
ments (with splade-v3 retriever, DeBERTa-v3 reranker).
𝐷denotes distractor documents chosen at random from
PubMed abstracts to add noise to the context. The
smaller model is more sensible to noise.
5. RAG adaptation
5.1. LLM Finetuning for RAG
Recent works have demonstrated that it is beneficial
to finetune an LLM to encourage it to better use re-
trieved context [ 21,22,32,37]. It has been shown to
improve RAG performances in-domain, and we inves-
tigate whether this holds for out-of-domain RAG gen-
eralization. To do so, we run supervised fine-tuning
on the MultiQA dataset9: a dataset consisting of 450k
general domain questions and answers, described in
Appendix Table 5. Its associated document collection
consists of Wikipedia [ 30] and MSMARCO [ 2] docu-
ments. Each supervised fine-tuning sample then con-
sists of a prompt (taken from [ 32]) with 5 retrieved
documents, the question and its answer. Models are
trained with LoRA [ 12]. After training, we evaluate the
RAG-adapted models on the multi-domain benchmark.
Training hyper-parameters are given in Appendix B.
Table 3 compares vanilla RAG to RAG-adapted model.
We note that while RAG adaptation does indeed im-
prove on General and WebSearch domains, it de-
grades performance on Context-critical and Long-form
datasets. The performance on biomedical datasets gets
improved for Llama3.2-1B model, but doesn’t really
change for larger models. This indicates that stan-
dard RAG adaptation does not generalize beyond
the training data.
9huggingface/datasets/dmrau/multi_qa
5

Adapting Large Language Models for
Multi-Domain Retrieval-Augmented-Generation
Domain DatasetLlama-3.2-1B-Instruct Llama-3-8B-Instruct
Without RAG With RAG Without RAG With RAG
BiomedicalBioasq12b 0.48 0.68 0.64 0.76
CovidQA 0.38 0.50 0.46 0.61
Mean 0.43 0.59 0.55 0.69
Long-formFiQA 0.54 0.50 0.55 0.51
Lifestyle 0.50 0.56 0.51 0.62
Recreation 0.38 0.45 0.47 0.58
Science 0.49 0.49 0.33 0.64
Technology 0.47 0.53 0.43 0.64
Writing 0.44 0.49 0.53 0.67
Mean 0.47 0.50 0.47 0.61
Web-Search SearchQA 0.38 0.57 0.55 0.75
Context-criticalParaphraseRC 0.18 0.47 0.32 0.63
SyllabusQA 0.37 0.30 0.40 0.26
TechQA 0.52 0.54 0.49 0.59
Mean 0.36 0.44 0.40 0.49
Table 2: Benchmarking RAG with models of different sizes (LLMEval). The RAG pipeline uses SPLADE-v3 retriever
and DeBERTa-v3 reranker.
General Biomedical WebSearch Cxt-critical Long Form Avg
Llama3.2-1B
Vanilla RAG 0.53 0.59 0.53 0.44 0.52 0.51
FT with RAG 0.65 0.63 0.68 0.41 0.42 0.50
Llama3-8B
Vanilla RAG 0.65 0.69 0.75 0.49 0.61 0.61
FT with RAG 0.71 0.70 0.82 0.49 0.49 0.58
Mistral-7B
Vanilla RAG 0.69 0.70 0.72 0.52 0.65 0.64
FT with RAG 0.72 0.69 0.85 0.50 0.53 0.59
Table 3: Comparing results (LLMEval) of vanilla RAG (out-of-the-box LLM) vs finetuning with RAG (LoRA FT on
MultiQA dataset). We report average score across multiple datasets within the same domain. Detailed results per
dataset are available in Appendix Tables 15 - 21.
5.2. Robust RAG adaptation
If we want RAG finetuning to transfer across domains,
it would mean that it should learn to identify relevant
information in the retrieved documents, that would
transfer to different types of collections, and different
types of answers. Therefore the drop on certain do-
mains can be explained by multiple factors:
•Thedifferentdistributionofrelevantinformationin
the retrieved documents can affect the generator’s
ability to exploit context,
•The stylistic differences in questions can affect the
model’s ability to understand the task at hand,
•Overfitting to the answer style in the training data
may hurt performance on long-form datasets.
In this work we compare different adaptation strategies
to improve robustness under domain shift.ImprovingattentionpatternviaLoRA-QKfinetuning.
We hypothesize that in order to learn model to better
focus on relevant documents we need to modify the
attention pattern. Therefore we train a LoRA variant
that only updates 𝑄and𝐾matrices ( LoRAQKAtt ) of
attention which are responsible for the attention pat-
tern. We also train LoRA variants updating MLP layers
only (LoRAMLP ), and LoRA updating all the attention
matrices ( LoRAAtt ) for comparison. We adjust the rank
of different LoRA variants in order to keep a similar
amount of trainable parameters for fair comparison
(details in Appendix Table 7).
Sequence-level Knowledge Distillation. Most of the
gold answers of standard datasets that compose Mul-
tiQA dataset correspond to very short text snippets,
without any additional information or explanation.
6

Adapting Large Language Models for
Multi-Domain Retrieval-Augmented-Generation
Standard answers generated by an instruction-tuned
LLMs tend to be in a quite different format: more ver-
bose, containing more details and explanations. There-
fore finetuning on short labels forces the model to sub-
stantially move away from its initial generation distribu-
tion. In order to avoid this effect we rely on Sequence-
level Knowledge Distillation (SKD) [ 16]. It consists in
training the student model on the labels generated by a
strong teacher, rather than on the gold labels. [ 48] sug-
gests that SKD allows to reduce the complexity of the
data and therefore makes learning simpler for the stu-
dent. In our experiments, we rely on Mistral-7B model
to generate labels. Preliminary experiments showed
Mistral-7B is an excellent teacher, on par with findings
in [23,43]. We emphasize that student model training
is performed with the same context as used in teacher
generations.
We also report additional regularization experiments
with distractors in the Appendix G.
5.3. Results and Analysis.
Figure 1 compares different proposed strategies for
robust RAG adaptation.
Improved attention pattern. We note that LoRA-
QKAttis indeed more resilient to distribution shift
onContext-critical andLong-form datasets compared to
other LoRA variants: for Llama-3-8B it does improve
over the baseline on Context-critical datasets which fol-
low different distribution of relevant information in the
retrieved documents. It also better preserves perfor-
mance on Long-Form datasets compared to other FT
strategies. Table 4 further confirms that LoRA-QKAtt is
better at exploring oracle context compared to Vanilla
RAG and more robust compared to other FT variants.
Sequence Knowledge Distillation. Knowledge dis-
tillation slightly decreases performance on general
domain, but does improve across all other domains
and lead to the best results overall. We note, that
when adaptation is performed on distilled labels, the
adaptation strategy doesn’t really matter, and all LoRA
variantsperformsimilarly. Weviewtworeasonsforthis.
First, we believe that this is due to the fact that there
is less discrepancy between the data used for adapta-
tion and models’ natural distribution. Therefore, model
adaptation is fully focused on better context exploita-
tion rather than mimicking training data format. Table
4 provides additional evidence for this, demonstrat-
ing that model FT with distilled labels exploits better
relevant context provided by Oracle documents. Sec-
ond, teacher-generated labels include explanations andLlama-3.2-1B Llama-3-8B
KILT-NQ BioASQ12b KILT-NQ BioASQ12b
Vanilla 0.73 0.69 0.83 0.78
FT with LoRA
Full 0.77 0.67 0.82 0.75
QKAtt 0.74 0.74 0.85 0.79
Att 0.76 0.71 0.85 0.76
MLP 0.77 0.70 0.82 0.75
FT with LoRA + distilled labels
Full 0.80 0.77 0.86 0.81
Table 4: LLMeval scores for RAG with oracle documents
in the context, with and without FineTuning for RAG.
We report Recall scores in the table 12 in the Appendix
reasoning from the context, much like in [ 47]. Note
that, at variance with [ 47] which generates chain-of-
thoughts for fine-tuning, distillation labels don’t rely
on the existence of an oracle document. These labels
are self-consistent because generated from the same
retrieved context used at train time.
In-depth analysis with RagChecker. To analyze
deeper various aspects of the RAG pipeline, we
adopt a set of metrics introduced in RAGChecker
[33]. This framework relies on two auxiliary models,
namely a claim extractor and anentailment classifier .
RAGChecker first extracts atomic claims from ground
truth answers and generated responses, and then de-
fines a set of metrics based on the entailment classifica-
tionresultsbetweenatomicclaimsandvariouspiecesof
text (retrieved context, ground truth answer, or gener-
ated response). We refer the reader to [ 33] for the full
definition of each particular metric. Figure 4 reports
RAGCheckermetricstocomparethreebestmodelsfrom
Figure 1, on a subset of domains. Results for other do-
mains are identical.
Domains with long ground truth labels (e.g. FiQA,
TechQA) exhibit similar trends and show evidence that
themodel trained with distilled labels features a higher
reliance on retrieved contexts than the baseline zero-shot
model. Thisisemphasizedbyconsistentimprovementin
metricsFaithfulness (i.e. higher percentage of response
claims entailed by the context) and Hallucination (i.e.
lower percentage of response claims not entailed by the
retrieved context or by the ground truth), but also by
the reduction in metrics measuring sensitivity to the
noise in retrieved contexts (i.e. higher percentage of
response claims not entailed by the ground truth and
entailed by the context). Higher reliance on the context
is exactly the effect we are aiming to achieve with our
10See Qwen/Qwen2.5-3B-Instruct and ynie/roberta-large-snli
7

Adapting Large Language Models for
Multi-Domain Retrieval-Augmented-Generation
precisionrecallfaithfulness 1 -- relevant noise sensitivity
1 -- irrelevant noise sensitivity
1 -- hallucination
self knowledge
context utilization claim recallcontext precision0.2 0.4 0.6 0.8bioasq12b
precisionrecallfaithfulness 1 -- relevant noise sensitivity
1 -- irrelevant noise sensitivity
1 -- hallucination
self knowledge
context utilization claim recallcontext precision0.2 0.4 0.6 0.8FiQA
precisionrecallfaithfulness 1 -- relevant noise sensitivity
1 -- irrelevant noise sensitivity
1 -- hallucination
self knowledge
context utilization claim recallcontext precision0.2 0.4 0.6 0.8robustqa_Lifestyle
precisionrecallfaithfulness 1 -- relevant noise sensitivity
1 -- irrelevant noise sensitivity
1 -- hallucination
self knowledge
context utilization claim recallcontext precision0.2 0.4 0.6 0.8techQA
precisionrecallfaithfulness 1 -- relevant noise sensitivity
1 -- irrelevant noise sensitivity
1 -- hallucination
self knowledge
context utilization claim recallcontext precision0.2 0.4 0.6 0.8bioasq12bVanilla RAG
LoraQKAtt + regular_labels
LoraAtt + distilled_labels
Figure 4: RAGChecker metrics on a subset of domains for Llama-3.2-1B. Effects on other domains are identical to
one of the reported domains. All metrics are "the higher the better". The BioASQ dataset has short ground truth
labels while other datasets have long labels. To compute metrics we use a Qwen claim extractor to segment out
claims made by the generator, and a RoBERTa claim entailment checker.10
8

Adapting Large Language Models for
Multi-Domain Retrieval-Augmented-Generation
tuned models. However, we highlight that the observed
differencesinmetricvaluescanalsobepartlyattributed
to a different stylein responses between the baseline
model (Llama-3.2-1B) and the teacher model used to
generate labels for distillation (Mistral-7B). We provide
more details in Appendix J.
The model tuned with regular (short) labels generates
shorter responses on average, hence it exhibits consis-
tent natural drops in metrics such as recallandcontext
utilization , which measure percentages of ground-truth
claims or context claims entailed by the generated re-
sponse. Retrieval metrics (claim recall, context preci-
sion) are the same across three models, since they all
have the same retrieval setting. Self-knowledge mea-
sures a percentage of correct (i.e. entailed by ground
truth) generated claims, not entailed by the retrieved
context. We observe a zero value for self-knowledge
for all models and datasets, showcasing that all three
models generate responses by looking at the context
rather than using internal LLM memory.
FortheBioASQdataset, whichwasevaluatedwithshort
labels, the model fine-tuned with short labels naturally
achieves better RAGChecker results than the two other
models.
6. Conclusion
This work addresses the problem of RAG adaptation un-
derdomainshift. Inordertostudythisphenomenonwe
first build a large and diverse multi-domain benchmark
for RAG. We then demonstrate that LLM finetuning for
RAGwithsequence-leveldistillationallowsthemodelto
better leverage the context, and transfers this capacity
to other domains.
9

Adapting Large Language Models for
Multi-Domain Retrieval-Augmented-Generation
References
[1]Melissa Ailem, Katerina Marazopoulou, Charlotte
Siska, and James Bono. Examining the robustness
ofllmevaluationtothedistributionalassumptions
of benchmarks, 2024. 2
[2]Payal Bajaj, Daniel Campos, Nick Craswell, Li
Deng, Jianfeng Gao, Xiaodong Liu, Rangan Ma-
jumder, Andrew McNamara, Bhaskar Mitra, Tri
Nguyen, et al. Ms marco: A human generated
machine reading comprehension dataset. arXiv
preprint arXiv:1611.09268 , 2016. 5
[3]Max Bartolo, Alastair Roberts, Johannes Welbl,
Sebastian Riedel, and Pontus Stenetorp. Beat
the ai: Investigating adversarial human annota-
tion for reading comprehension. Transactions of
the Association for Computational Linguistics , 8:
662–678, 2020. 13
[4]Nitay Calderon, Naveh Porat, Eyal Ben-David,
Alexander Chapanin, Zorik Gekhman, Nadav
Oved, Vitaly Shalumov, and Roi Reichart. Measur-
ing the robustness of nlp models to domain shifts,
2024. 2
[5]Vittorio Castelli, Rishav Chakravarti, Saswati
Dana, Anthony Ferritto, Radu Florian, Martin
Franz, Dinesh Garg, Dinesh Khandelwal, Scott
McCarley, Michael McCawley, Mohamed Nasr, Lin
Pan, Cezar Pendus, John Pitrelli, Saurabh Pujar,
Salim Roukos, Andrzej Sakrajda, Avi Sil, Rosario
Uceda-Sosa, Todd Ward, and Rong Zhang. The
TechQA dataset. In Proceedings of the 58th An-
nual Meeting of the Association for Computational
Linguistics , pages 1269–1278, Online, 2020. As-
sociation for Computational Linguistics. 3
[6]Matthew Dunn, Levent Sagun, Mike Higgins,
V. Ugur Güney, Volkan Cirik, and Kyunghyun
Cho. Searchqa: A new q&a dataset augmented
with context from a search engine. CoRR,
abs/1704.05179, 2017. 3
[7]Nigel Fernandez, Alexander Scarlatos, and An-
drew Lan. Syllabusqa: A course logistics question
answering dataset, 2024. 3
[8]Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang
Jia, Jinliu Pan, Yuxi Bi, Yi Dai, Jiawei Sun, and
Haofen Wang. Retrieval-augmented generation
for large language models: A survey. arXiv
preprint arXiv:2312.10997 , 2023. 1
[9]Rujun Han, Peng Qi, Yuhao Zhang, Lan Liu, Juli-
ette Burger, William Yang Wang, Zhiheng Huang,
Bing Xiang, and Dan Roth. RobustQA: Bench-
marking the robustness of domain adaptation for
open-domain question answering. In Findings of
the Association for Computational Linguistics: ACL
2023, pages 4294–4311, Toronto, Canada, 2023.Association for Computational Linguistics. 1, 2,
3, 4
[10]Rujun Han, Yuhao Zhang, Peng Qi, Yumo Xu,
Jenyuan Wang, Lan Liu, William Yang Wang, Bo-
nan Min, and Vittorio Castelli. Rag-qa arena:
Evaluating domain robustness for long-form re-
trieval augmented question answering, 2024. 1,
2
[11]Pengcheng He, Jianfeng Gao, and Weizhu Chen.
Debertav3: Improving deberta using electra-style
pre-training with gradient-disentangled embed-
ding sharing, 2021. 4
[12]Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan
Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang,
Weizhu Chen, et al. Lora: Low-rank adaptation
of large language models. ICLR, 1(2):3, 2022. 5
[13]Kelvin Jiang, Dekun Wu, and Hui Jiang. Free-
baseQA: A new factoid QA data set matching
trivia-style question-answer pairs with Freebase.
InProceedings of the 2019 Conference of the North
American Chapter of the Association for Computa-
tional Linguistics: Human Language Technologies,
Volume 1 (Long and Short Papers) , pages 318–
323, Minneapolis, Minnesota, 2019. Association
for Computational Linguistics. 13
[14]Dian Jiao, Li Cai, Jingsheng Huang, Wenqiao
Zhang, Siliang Tang, and Yueting Zhuang. Due-
trag: Collaborative retrieval-augmented genera-
tion.arXiv preprint arXiv:2405.13002 , 2024. 1
[15]Mandar Joshi, Eunsol Choi, Daniel S. Weld, and
Luke Zettlemoyer. Triviaqa: A large scale dis-
tantly supervised challenge dataset for reading
comprehension, 2017. 13
[16]Yoon Kim and Alexander M. Rush. Sequence-level
knowledge distillation. In Proceedings of the 2016
Conference on Empirical Methods in Natural Lan-
guageProcessing ,pages1317–1327,Austin,Texas,
2016. Association for Computational Linguistics.
7
[17]Tom Kwiatkowski, Jennimaria Palomaki, Olivia
Redfield, Michael Collins, Ankur Parikh, Chris Al-
berti, Danielle Epstein, Illia Polosukhin, Jacob De-
vlin,KentonLee,etal. Naturalquestions: abench-
mark for question answering research. Transac-
tions of the Association for Computational Linguis-
tics, 7:453–466, 2019. 5
[18]Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying
Sheng, Lianmin Zheng, Cody Hao Yu, Joseph E.
Gonzalez, Hao Zhang, and Ion Stoica. Efficient
memory management for large language model
serving with pagedattention. In Proceedings of
the ACM SIGOPS 29th Symposium on Operating
Systems Principles , 2023. 14
10

Adapting Large Language Models for
Multi-Domain Retrieval-Augmented-Generation
[19]Carlos Lassance, Hervé Déjean, Thibault Formal,
and Stéphane Clinchant. Splade-v3: New base-
lines for splade, 2024. 4
[20]Patrick S. H. Lewis, Ethan Perez, Aleksandra
Piktus, Fabio Petroni, Vladimir Karpukhin, Na-
man Goyal, Heinrich Küttler, Mike Lewis, Wen-
tau Yih, Tim Rocktäschel, Sebastian Riedel,
and Douwe Kiela. Retrieval-augmented gener-
ation for knowledge-intensive NLP tasks. CoRR,
abs/2005.11401, 2020. 1
[21]Xi Victoria Lin, Xilun Chen, Mingda Chen, Weijia
Shi, Maria Lomeli, Rich James, Pedro Rodriguez,
Jacob Kahn, Gergely Szilvasy, Mike Lewis, Luke
Zettlemoyer, and Scott Yih. Ra-dit: Retrieval-
augmented dual instruction tuning, 2024. 1, 2,
5
[22]Zihan Liu, Wei Ping, Rajarshi Roy, Peng Xu,
Chankyu Lee, Mohammad Shoeybi, and Bryan
Catanzaro. Chatqa: Building gpt-4 level conver-
sational qa models. CoRR, 2024. 5
[23]Maxime Louis, Hervé Déjean, and Stéphane Clin-
chant. Pisco: Pretty simple compression for
retrieval-augmented generation. arXiv preprint
arXiv:2501.16075 , 2025. 7
[24]Timo Möller, Anthony Reina, Raghavan Jayaku-
mar, and Malte Pietsch. COVID-QA: A question
answering dataset for COVID-19. In Proceedings
of the 1st Workshop on NLP for COVID-19 at ACL
2020, Online, 2020. Association for Computa-
tional Linguistics. 2, 3
[25]Giovanni Monea, Maxime Peyrard, Martin Josi-
foski, Vishrav Chaudhary, Jason Eisner, Emre Kıcı-
man, Hamid Palangi, Barun Patra, and Robert
West. A glitch in the matrix? locating and detect-
ing language model grounding with fakepedia,
2024. 14
[26]Anastasios Nentidis, Georgios Katsimpras, Anasta-
sia Krithara, Salvador Lima López, Eulália Farré-
Maduell, Luis Gasco, Martin Krallinger, and Geor-
gios Paliouras. Overview of BioASQ 2023: The
Eleventh BioASQ Challenge on Large-Scale Biomed-
ical Semantic Indexing and Question Answering ,
page 227–250. Springer Nature Switzerland,
2023. 2, 3
[27]Anastasios Nentidis, Georgios Katsimpras, Anas-
tasia Krithara, and Georgios Paliouras. Overview
of bioasq tasks 12b and synergy12 in clef2024. In
Working Notes of CLEF , 2024. 3
[28]Xing Niu, Prashant Mathur, Georgiana Dinu, and
Yaser Al-Onaizan. Evaluating robustness to in-
put perturbations for neural machine translation.
2020. 14[29]Oded Ovadia, Menachem Brief, Moshik Mishaeli,
and Oren Elisha. Fine-tuning or retrieval? com-
paring knowledge injection in LLMs. In Proceed-
ings of the 2024 Conference on Empirical Methods
in Natural Language Processing , pages 237–250,
Miami, Florida, USA, 2024. Association for Com-
putational Linguistics. 1
[30]Fabio Petroni, Aleksandra Piktus, Angela
Fan, Patrick Lewis, Majid Yazdani, Nicola
De Cao, James Thorne, Yacine Jernite, Vladimir
Karpukhin, Jean Maillard, et al. Kilt: a bench-
mark for knowledge intensive language tasks.
arXiv preprint arXiv:2009.02252 , 2020. 5
[31]Pranav Rajpurkar, Jian Zhang, Konstantin Lopy-
rev, and Percy Liang. Squad: 100,000+ questions
for machine comprehension of text, 2016. 13
[32]David Rau, Hervé Déjean, Nadezhda Chirkova,
Thibault Formal, Shuai Wang, Vassilina Nikoulina,
and Stéphane Clinchant. Bergen: A benchmark-
ing library for retrieval-augmented generation,
2024. 4, 5, 14
[33]Dongyu Ru, Lin Qiu, Xiangkun Hu, Tianhang
Zhang, Peng Shi, Shuaichen Chang, Cheng Ji-
ayang, Cunxiang Wang, Shichao Sun, Huanyu Li,
Zizhao Zhang, Binjie Wang, Jiarong Jiang, Tong
He, Zhiguo Wang, Pengfei Liu, Yue Zhang, and
Zheng Zhang. Ragchecker: A fine-grained frame-
work for diagnosing retrieval-augmented genera-
tion, 2024. 2, 7
[34]Amrita Saha, Rahul Aralikatte, Mitesh M. Khapra,
and Karthik Sankaranarayanan. Duorc: To-
wards complex language understanding with
paraphrased reading comprehension. CoRR,
abs/1804.07927, 2018. 3
[35]Ivan Stelmakh, Yi Luan, Bhuwan Dhingra, and
Ming-Wei Chang. Asqa: Factoid questions meet
long-form answers, 2023. 13
[36]Nandan Thakur, Nils Reimers, Andreas Rücklé,
Abhishek Srivastava, and Iryna Gurevych. BEIR:
A heterogeneous benchmark for zero-shot evalu-
ation of information retrieval models. In Thirty-
fifth Conference on Neural Information Processing
Systems Datasets and Benchmarks Track (Round
2), 2021. 3
[37]Boxin Wang, Wei Ping, Lawrence McAfee, Peng
Xu, Bo Li, Mohammad Shoeybi, and Bryan Catan-
zaro. Instructretro: Instruction tuning post
retrieval-augmented pretraining, 2024. 2, 5
[38]Jindong Wang, Cuiling Lan, Chang Liu, Yidong
Ouyang, and Tao Qin. Generalizing to unseen
domains: A survey on domain generalization. In
ProceedingsoftheThirtiethInternationalJointCon-
ference on Artificial Intelligence, IJCAI-21 , pages
11

Adapting Large Language Models for
Multi-Domain Retrieval-Augmented-Generation
4627–4635. International Joint Conferences on
Artificial Intelligence Organization, 2021. Survey
Track. 2
[39]Lucy Lu Wang, Kyle Lo, Yoganand Chandrasekhar,
Russell Reas, Jiangjiang Yang, Doug Burdick,
Darrin Eide, Kathryn Funk, Yannis Katsis, Rod-
ney Michael Kinney, Yunyao Li, Ziyang Liu,
William Merrill, Paul Mooney, Dewey A. Mur-
dick, DevvretRishi, JerrySheehan, ZhihongShen,
Brandon Stilson, Alex D. Wade, Kuansan Wang,
Nancy Xin Ru Wang, Christopher Wilhelm, Boya
Xie, Douglas M. Raymond, Daniel S. Weld, Oren
Etzioni, and Sebastian Kohlmeier. CORD-19: The
COVID-19 open research dataset. In Proceedings
of the 1st Workshop on NLP for COVID-19 at ACL
2020, Online, 2020. Association for Computa-
tional Linguistics. 3
[40]Xuezhi Wang, Haohan Wang, and Diyi Yang. Mea-
sure and improve robustness in nlp models: A
survey, 2022. 2
[41]Jerry Wei, Chengrun Yang, Xinying Song, Yifeng
Lu, Nathan Hu, Jie Huang, Dustin Tran, Daiyi
Peng, Ruibo Liu, Da Huang, et al. Long-form
factualityinlargelanguagemodels. arXivpreprint
arXiv:2403.18802 , 2024. 1, 2
[42]Johannes Welbl, Nelson F. Liu, and Matt Gardner.
Crowdsourcing multiple choice science questions,
2017. 13
[43]Zhangchen Xu, Fengqing Jiang, Luyao Niu,
Bill Yuchen Lin, and Radha Poovendran. Stronger
models are not stronger teachers for instruction
tuning.arXiv preprint arXiv:2411.07133 , 2024. 7
[44]Linyi Yang, Shuibai Zhang, Libo Qin, Yafu Li, Yi-
dong Wang, Hanmeng Liu, Jindong Wang, Xing
Xie, and Yue Zhang. GLUE-X: Evaluating natural
language understanding models from an out-of-
distribution generalization perspective. In Find-
ings of the Association for Computational Linguis-
tics: ACL 2023 , pages 12731–12750, Toronto,
Canada, 2023. Association for Computational Lin-
guistics. 2
[45]YiYang,Wen-tauYih,andChristopherMeek. Wik-
iQA: A challenge dataset for open-domain ques-
tion answering. In Proceedings of the 2015 Con-
ference on Empirical Methods in Natural Language
Processing , pages 2013–2018, Lisbon, Portugal,
2015. Association for Computational Linguistics.
13
[46]Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua
Bengio, William W. Cohen, Ruslan Salakhutdi-
nov, and Christopher D. Manning. Hotpotqa: A
dataset for diverse, explainable multi-hop ques-
tion answering, 2018. 13[47]Tianjun Zhang, Shishir G. Patil, Naman Jain,
Sheng Shen, Matei Zaharia, Ion Stoica, and
Joseph E. Gonzalez. Raft: Adapting language
model to domain specific rag, 2024. 1, 2, 7, 14
[48]Chunting Zhou, Jiatao Gu, and Graham Neu-
big. Understanding knowledge distillation in
non-autoregressive machine translation. In Inter-
national Conference on Learning Representations ,
2020. 7
12

Adapting Large Language Models for
Multi-Domain Retrieval-Augmented-Generation
A. Datasets
Dataset Number of queries
HotpotQA [46] 88839
NQ-open 87925
SQuAD [31] 87596
TriviaQA [15] 61797
MSmarco 59699
AdversarialQA [3] 29966
FreebaseQA [13] 20356
SciQ [42] 11679
ASQA [35] 4353
WikiQA [45] 813
Table 5: Content of MultiQA union of datasets
B. Experimental details
In Section 5, we fine-tune models on general domain
RAG data, to see if it improves performances on the out-
of-domain RAG pipeline. Hyper-parameters are given
in Table 6.
Hyperparameter Value
Batch Size (Llama32-1B) 512
Batch size (Llama3-8B) 256
LR 5×10−4
LR scheduler linear
optimizer AdamW
Epochs 1
LoRA Dropout 0.1
LoRA Alpha 32
Weight Decay 0.1
Warmup Ratio 0.05
Max Gradient Norm 1.0
Documents max tokens 128
Table 6: Fine-tuning Hyperparameters.
C.Retrievers, Rerankers, and Top-k Re-
trieved Documents
In Figure 3, we highlight the importance of including
nottoomanydocumentsasthisintroducesnoise forthe
generator, and not too few documents as this removes
useful context for the generator. Using five documents
seems like a good balance point. As shown in Figure
5, we observe better performance when using oracle
retrieval instead of top-5 retrieved documents. We can
therefore say that suboptimal retrievers are unable to
rank documents in the most optimal way to maximise
generator performance, and/or generators are sensitive
to surrounding noise around the useful context. We
also notice the superior performance of the SPLADE-v3
retriever on the BioASQ task compared to other docu-Model Rank Trainable parameters
Llama3.2-1B
FullLoRA 16 11272192
LoRA-QKAtt 128 13631488
LoRAAtt 64 13631488
LoRAMLP 24 11796480
Llama3-8B
FullLoRA 16 41943040
LoRA-QKAtt 128 54525952
LoRAAtt 64 54525952
LoRAMLP 24 42467328
Table 7: Rank and amount of trainable parameters for
different LoRA variants.
Match Recall LLMeval0.00.51.0
0.48
0.48
0.49
0.50
0.53
0.60
0.61
0.62
0.65
0.70
0.73
0.73
0.75
0.79
0.83bge-base
retromaeBM25
splade-v3ORACLE
Figure 5: BioASQ-11b Performance Across Retrievers
(SOLAR-10.7B generator)
Recall LLMeval0.000.250.500.751.00
0.43
0.43
0.46
0.65
0.63
0.67
0.67
0.65
0.69
0.76
0.74
0.78No RR (1B)
MiniLM6 (1B)
DeBERTa-v3 (1B)No RR (8B)
MiniLM6 (8B)
DeBERTa-v3 (8B)
Figure 6: Reranking (RR) on BioASQ-12b dataset with
splade-v3 retriever, with two generator of varying sizes
(Llama3.2-1B-Instruct + Llama3-8B-Instruct) and com-
paring rerankers. Reranking with DeBERTa-v3 brings
some improvement in RAG.
ment retrievers. Finally, figure 6 recalls the importance
of rerankers for RAG performance.
D. Prompts
For each domain, we use simple domain-specific
prompts to describe the task, which varies across
datasets especially across short-form and long-form QA.
The generation temperature is set to 0for reproducibil-
13

Adapting Large Language Models for
Multi-Domain Retrieval-Augmented-Generation
ity. For gemma, qwen and llama models we use 4-bit
quantization. For SOLAR-10.7B-Instruct we use vllm
[18]).
E. LLM evaluation
To evaluate the quality of responses, we rely on an
evaluation computed by a large language model with
the prompt described in Figure 7. Unless otherwise
specified, we use the SOLAR-10.7B model11as judge.
[32] find that this metric has high correlation with
GPT4. LLMEval better captures semantic meaning of
text compared to static metrics like match (Figure 8).
Figure 7: LLM Evaluation Prompt
system: "You are an evaluation tool. Answer with
one of 1: Correct, 0.5: Partially correct, 0: wrong.
user: "Here is a question, a golden answer, and an
AI-generated answer. Can you judge whether the
AI-generated answer is correct according to the
question and golden answer? Simply answer with
one of 1: correct, 0.5: partially correct, 0: wrong.
Question: {question} . Golden answer: {answer} .
Generated answer: {prediction} ."
Table 8 provides several examples demonstrating that
LLMeval is generally better suited to judge the correct-
nessoftheresponsecomparedtocommonlyusedMatch
metric.
F. Other benchmark results
Tables 9a,9b,10 report additional benchmark results for
small and medium-sized LLMs in terms of Recall and
LLMEval.
G. Training with distractors.
In addition to the main methods, we experiment with
previously proposed strategies [ 47], that add noise in
the retrieved documents during training, in order to
make models more resilient to bad context. To do so, af-
ter retrieval and reranking is performed, we replace the
least𝐷relevant documents with documents sampled
at random from the documents repository.
Table 13 reports the results of training with distractors.
We note that, as expected, training with distractors
does not necessarily make the model more robust
fornewdomains. Weexplorerobustnesstothecontext
more in depth in the Appendix H.
11huggingface/upstage/SOLAR-10.7B-Instruct-v1.0
12https://huggingface.co/meta-llama/Llama-3.1-70B
13https://huggingface.co/upstage/SOLAR-10.7B-Instruct-v1.
0H. Context robustness analysis.
We investigate how do RAG adaptation techniques im-
pact model robustness to noisy context. Similar to [ 28]
we introduce robustness metric 𝑅(𝑦,𝑦′)measuring the
ratio of correct answers produced by RAG model deal-
ing with noisy context in comparison to the same model
dealing with clean context. Clean context refers to our
default RAG settings (splade-v3 retriever, DeBERTa-v3
reranker, top5 documents). We compare 3 types of
noisy context: (1) randomly replace 4 out of 5 relevant
documents by distractors (2) remove reranking from
the pipeline (3) extend top-k documents (from 5 to 20).
Table 14 reports the results of robustness evaluation
across 4 datasets belonging to different domains (NQ,
Bioasq12b,TechQAandFiQA).Inadditiontotherobust-
ness metric we report LLMEval performance obtained
in clean settings. The best model should have high-
est LLMEval in clean settings, and highest robustness
scores (e.g. being resilient to the distribution shift in
the way the relevant information is presented).
First, we note that, similar to the observation made in
section 3 (Figure 3), the smaller model ( Llama-3.2-1B )
is more sensitive to noisy context compared to Llama-
3-8B.
We also note that models trained with distractors are in-
deedmorerobustwhenthesametypeofnoiseispresent
at test time, but not necessarily to other types of distri-
bution shift (top-20, no reranking) when trained with
regular labels. Overall, original model (Lama-3.2-1B)
seems to be more robust to the distribution shift in the
relevant information, compared to all finetuned vari-
ants (especially on general domain). We attribute it to
strong capabilities of instruction-tuned LLMs to exploit
theircontext. However,highestrobustnessdoesnotnec-
essarily correlate with the best performance in clean
settings. Models trained with distilled labels and dis-
tractors seems to represent the best trade-off between
robustness and overall performance. This aligns with
our hypothesis, that such a model diverges less from
the original distribution and focuses more on better
exploiting relevant information.
I. When does RAG fail?
A well known struggle in RAG is the ability of models
to adapt to contextual knowledge. Context can either
clash with the model’s internal knowledge [ 25], or, in
the case of multidomain RAG, the model may have
no internal knowledge at all. Figure 8 gives a formal
example in 8, where the model struggles to adapt to
out-of-distribution text.
14

Adapting Large Language Models for
Multi-Domain Retrieval-Augmented-Generation
Table 8: Comparison between Match and LLMEval metrics across different examples.
Question Gold Label Model Output Notes Match LLMEval
Is there an
association
between
bruxism and
reflux?yes Based on the provided documents,
it appears that there is some
evidence suggesting a relation-
shipbetweenbruxismandgastroe-
sophageal reflux disease (GERD).
[...Analysis of provided RAG docu-
ments...]LLMEval cap-
tures semantics
better than
Match (similar
cases are fre-
quent).0.0 1.0
Which topoi-
somerase is
essential in
yeast?topoisomerase
IIortopo IIBased on the provided documents,
there is a link between BCL11B
haploinsufficiency and syndromic
neurodevelopmental delay. [...ex-
planation based on RAG docu-
ments...] In yeast, Topoisomerase
I is essential, while Topoisomerase
II is not essential.LLMEval cap-
tures semantics
better than
Match metric.1.0 0.0
Which is the
database
of somatic
mutations in
normal cells?DSMNC or
Database
of Somatic
Mutations in
Normal CellsThe correct answer is: Document
1: DSMNC: a database of somatic
mutations in normal cells. This
is the database of somatic muta-
tionsinnormalcells,asmentioned
in Document 2: SomaMutDB: a
database of somatic mutations in
normal human tissues.LLMEval can be
biased towards
"final answer"
(rare failure
case).1.0 0.0
Figure 8: Generator fails to exploit context in the
biomedical domain.
Background:
Document 1: [...] we found that the predomi-
nantly nuclear A3B [...] In contrast to the exclu-
sively cytoplasmic A3G
Document 2: [...] APOBEC3G limits the replica-
tion of [...]. It localizes predominantly to the cyto-
plasm of cells [...]
Document 3: [...]
Question: Is APOBEC3B protein predominantly
cytoplasmic or nuclear?
Response: Based on the provided documents, it
appears that the APOBEC3B protein is predomi-
nantly cytoplasmic. The documents mention that:
- APOBEC3B is predominantly nuclear in the cyto-
plasm of cells.
- APOBEC3B localizes to the cytoplasm of cells,
which is consistent with a model where it pack-
ages into assembling virions.
Label: nuclearJ.RAGChecker’s sensitivity to style of the
LLM response
In manual inspection of RAGChecker results in Sec-
tion 5.3, we found that style difference between two
models may cause difference in RAGChecker metrics . Par-
ticularly, RAGChecker extracts atomic claims from a
model-generated response and makes judgment about
them depending on whether a claim is entailed by the
ground truth answer or by the retrieved context.
If two models differ in style, e.g. one model inserts
genericcommentsintoitsreplymoreoftenthananother,
then these generic comments will not be entailed by
the ground truth answer or by the retrieved context
and will be treated the same way as wrong claims . This
may reduce the values of some RAGChecker metrics for
one of the models.
In our case, Llama-3.2-1B model trained on labels
distilled from Mistral-7B exhibits higher Faithfulness
and lower Hallucination than a baseline Llama-3.2-1B
modelappliedzero-shot. Wefoundthatthelattermodel
tendstorepeattheuser’squestionmorefrequentlythan
the former model, e.g. “The question is asking what
15

Adapting Large Language Models for
Multi-Domain Retrieval-Augmented-Generation
Dataset gemma-2b-it qwen-2.5-3b-instruct
Without RAG With RAG Without RAG With RAG
Recall LLMEval Recall LLMEval Recall LLMEval Recall LLMEval
Bioasq11b 0.341 0.357 0.470 0.589 0.382 0.491 0.649 0.769
Bioasq12b 0.247 0.367 0.438 0.584 0.362 0.494 0.634 0.769
CovidQA 0.229 0.213 0.357 0.453 0.305 0.336 0.480 0.559
FiQA 0.170 0.308 0.169 0.317 0.200 0.406 0.201 0.391
ParaphraseRC 0.104 0.104 0.211 0.322 0.190 0.141 0.553 0.572
Lifestyle 0.234 0.241 0.201 0.320 0.304 0.390 0.337 0.565
Recreation 0.192 0.140 0.185 0.261 0.299 0.238 0.381 0.498
Science 0.280 0.288 0.242 0.376 0.325 0.454 0.339 0.457
Technology 0.239 0.241 0.208 0.342 0.289 0.380 0.313 0.508
Writing 0.231 0.257 0.193 0.313 0.335 0.470 0.353 0.549
SearchQA 0.149 0.183 0.315 0.350 0.091 0.140 0.671 0.724
SyllabusQA 0.256 0.300 0.195 0.231 0.339 0.312 0.391 0.278
TechQA 0.237 0.184 0.268 0.253 0.274 0.256 0.412 0.528
(a) Benchmark Results for Small Models
Dataset Llama3-8B-instruct SOLAR-10.7B-instruct
Without RAG With RAG Without RAG With RAG
Recall LLMEval Recall LLMEval Recall LLMEval Recall LLMEval
Bioasq11b 0.446 0.595 0.615 0.762 0.445 0.622 0.668 0.791
Bioasq12b 0.462 0.600 0.617 0.763 0.431 0.609 0.674 0.782
CovidQA 0.321 0.328 0.501 0.551 0.311 0.405 0.503 0.605
FiQA 0.198 0.432 0.230 0.438 0.196 0.492 0.218 0.499
ParaphraseRC 0.274 0.279 0.607 0.624 0.377 0.356 0.635 0.648
Lifestyle 0.299 0.442 0.358 0.592 0.301 0.549 0.360 0.688
Recreation 0.282 0.283 0.367 0.532 0.308 0.361 0.412 0.603
Science 0.320 0.464 0.364 0.568 0.325 0.426 0.377 0.634
Technology 0.290 0.440 0.332 0.586 0.283 0.422 0.341 0.637
Writing 0.287 0.521 0.363 0.629 0.318 0.567 0.386 0.740
SearchQA 0.662 0.668 0.722 0.780 0.397 0.552 0.687 0.754
SyllabusQA 0.382 0.304 0.311 0.313 0.309 0.295 0.305 0.284
TechQA 0.288 0.266 0.469 0.576 0.273 0.300 0.461 0.597
(b) Benchmark Results for Medium-Sized Models
Table 9: Benchmarks using models of different sizes (LLMEval is performed with Llama-3.1-70B12and a prompt
specified in the Appendix D). The RAG pipeline uses SPLADE-v3 retriever and DeBERTa-v3 reranker.
prevents interest rates from rising.”. We confirmed this
quantitatively: the number of claims extracted from
model responses, containing keywords “question” and
“ask”, is 96 for the baseline model and only 15 for the
model trained with distilled labels. All these question
repetitions will be counted as claims not entailed by
the ground truth response or by the retrieved context,
hence reducing Faithfulness14and increasing Hallucina-
tion15.
14Faithfulness is defined as a portion of response claims entailed
by the retrieved context.
15Hallucination is defined as a portion of response claims not en-
tailed by the ground truth answer and not entailed by the contextK. Full tables for RAG adaptation
In this section, we display all RAG adaptation results
by individual datasets in Tables 15 - 21.
16

Adapting Large Language Models for
Multi-Domain Retrieval-Augmented-Generation
Domain DatasetLlama-3.2-1B-Instruct Llama-3-8B-Instruct
Without RAG With RAG Without RAG With RAG
Recall LLMEval Recall LLMEval Recall LLMEval Recall LLMEval
BiomedicalBioasq12b 0.26 0.48 0.52 0.68 0.46 0.64 0.62 0.76
CovidQA 0.26 0.38 0.42 0.50 0.32 0.46 0.50 0.61
Mean 0.26 0.43 0.47 0.59 0.39 0.55 0.56 0.69
Long-formFiQA 0.19 0.54 0.21 0.50 0.20 0.55 0.23 0.51
Lifestyle 0.27 0.50 0.32 0.56 0.30 0.51 0.36 0.62
Recreation 0.25 0.38 0.30 0.45 0.28 0.47 0.37 0.58
Science 0.31 0.49 0.34 0.49 0.32 0.33 0.36 0.64
Technology 0.27 0.47 0.30 0.53 0.29 0.43 0.33 0.64
Writing 0.25 0.44 0.32 0.49 0.29 0.53 0.36 0.67
Mean 0.26 0.47 0.30 0.50 0.28 0.47 0.34 0.61
Web-Search SearchQA 0.34 0.38 0.46 0.57 0.47 0.55 0.72 0.75
Context-criticalParaphraseRC 0.16 0.18 0.44 0.47 0.27 0.32 0.61 0.63
SyllabusQA 0.27 0.37 0.28 0.30 0.38 0.40 0.31 0.26
TechQA 0.26 0.52 0.34 0.54 0.29 0.49 0.47 0.59
Mean 0.23 0.36 0.35 0.44 0.32 0.40 0.46 0.49
Table 10: Benchmarks using models of different sizes (LLMEval is performed with SOLAR-10.7B-Instruct13and a
prompt specified in the Appendix D). The RAG pipeline uses SPLADE-v3 retriever and DeBERTa-v3 reranker.
17

Adapting Large Language Models for
Multi-Domain Retrieval-Augmented-Generation
RetrieverMatch Recall LLMeval
Llm-1B Llm-8B Mstrl-7B Llm-1B Llm-8B Mstrl-7B Llm-1B Llm-8B Mstrl-7BKILT-NQbge 0.536 0.623 0.654 0.624 0.702 0.736 0.567 0.682 0.712
bge + RR 0.562 0.654 0.68 0.655 0.733 0.766 0.593 0.705 0.731
retromae 0.52 0.597 0.62 0.604 0.677 0.702 0.536 0.652 0.68
retromae + RR 0.544 0.63 0.666 0.642 0.711 0.75 0.571 0.689 0.71
bm25 0.392 0.451 0.479 0.483 0.542 0.574 0.439 0.531 0.572
bm25 + RR 0.459 0.556 0.579 0.562 0.64 0.673 0.506 0.619 0.649
splade 0.533 0.628 0.654 0.626 0.714 0.737 0.58 0.685 0.707
splade + RR 0.556 0.649 0.683 0.652 0.731 0.767 0.597 0.696 0.724
oracle 0.677 0.808 0.83 0.754 0.858 0.886 0.731 0.827 0.834BioASQ-12bbge 0.302 0.416 0.443 0.428 0.545 0.566 0.621 0.705 0.718
bge + RR 0.33 0.462 0.502 0.471 0.608 0.636 0.675 0.742 0.757
retromae 0.288 0.42 0.468 0.421 0.551 0.612 0.615 0.698 0.724
retromae + RR 0.334 0.457 0.495 0.476 0.605 0.637 0.662 0.737 0.764
bm25 0.287 0.39 0.437 0.393 0.503 0.56 0.584 0.675 0.704
bm25 + RR 0.318 0.418 0.473 0.446 0.553 0.61 0.62 0.701 0.723
splade 0.414 0.489 0.497 0.585 0.644 0.645 0.694 0.763 0.774
splade + RR 0.317 0.474 0.51 0.464 0.634 0.654 0.682 0.775 0.783
oracle 0.351 0.468 0.517 0.524 0.633 0.672 0.69 0.78 0.794TechQAbge 0.014 0.066 0.024 0.35 0.49 0.44 0.554 0.605 0.632
bge + RR 0.014 0.066 0.024 0.35 0.47 0.432 0.547 0.596 0.623
retromae 0.011 0.055 0.018 0.339 0.468 0.421 0.529 0.617 0.61
retromae + RR 0.011 0.06 0.019 0.34 0.464 0.424 0.52 0.607 0.605
bm25 0.018 0.056 0.016 0.353 0.468 0.418 0.514 0.598 0.608
bm25 + RR 0.016 0.052 0.019 0.345 0.469 0.424 0.527 0.598 0.596
splade 0.01 0.042 0.021 0.346 0.443 0.442 0.528 0.635 0.621
splade + RR 0.014 0.06 0.023 0.356 0.475 0.423 0.539 0.601 0.609
oracle - - - - - - - -FiQAbge 0.0 0.001 0.0 0.208 0.233 0.226 0.506 0.539 0.582
bge + RR 0.0 0.0 0 0.208 0.232 0.227 0.52 0.544 0.591
retromae 0.0 0.001 0 0.202 0.226 0.22 0.479 0.521 0.575
retromae + RR 0.001 0.0 0 0.207 0.231 0.224 0.5 0.544 0.594
bm25 0.0 0 0 0.188 0.216 0.212 0.399 0.485 0.562
bm25 + RR 0.0 0.001 0 0.2 0.226 0.219 0.456 0.52 0.575
splade 0 0 0 0.189 0.182 0.22 0.485 0.563 0.586
splade + RR 0.001 0.0 0 0.208 0.231 0.225 0.523 0.545 0.59
oracle - - - - - - - -SyllabusQAbge 0.056 0.057 0.071 0.271 0.305 0.347 0.294 0.272 0.349
bge + RR 0.057 0.057 0.076 0.285 0.314 0.366 0.301 0.289 0.357
retromae 0.061 0.05 0.072 0.276 0.296 0.35 0.293 0.255 0.372
retromae + RR 0.057 0.056 0.068 0.277 0.312 0.354 0.298 0.284 0.374
bm25 0.087 0.106 0.147 0.361 0.425 0.502 0.443 0.555 0.565
bm25 + RR 0.082 0.114 0.153 0.37 0.442 0.522 0.478 0.608 0.604
splade 0.057 0.061 0.066 0.266 0.272 0.351 0.237 0.28 0.33
splade + RR 0.057 0.06 0.075 0.278 0.305 0.363 0.322 0.289 0.342
oracle - - - - - - - -
Table 11: Three metrics across 5 datasets, 3 generators, 5 retrievers and reranking (RR).
18

Adapting Large Language Models for
Multi-Domain Retrieval-Augmented-Generation
Llama-3.2-1B Llama-3-8B
KILT-NQ BioASQ12b KILT-NQ BioASQ12b
Vanilla RAG 0.75 0.52 0.86 0.63
FT with LoRA
Full-LoRA 0.71 0.41 0.76 0.36
LoRA-QKAtt 0.68 0.58 0.78 0.53
LoRA-Att 0.7 0.43 0.78 0.41
LoRA-MLP 0.71 0.43 0.75 0.37
FT with LoRA + distilled labels
Full-LoRA 0.87 0.67 0.89 0.69
Table 12: Recall scores for RAG with oracle documents in the context, with and without FineTuning for RAG.
Setup General Biomed. WebSearch CtxtCritical LongForm Avg
Full-LoRA 0.65 0.63 0.68 0.41 0.42 0.50
Full-LoRA + Distractors 0.61 0.62 0.67 0.40 0.41 0.49
LoRA-QKAtt + Distractors 0.60 0.62 0.59 0.43 0.45 0.50
Full-LoRA distill 0.64 0.67 0.70 0.49 0.61 0.58
Full-LoRA distill + Distractors 0.60 0.66 0.69 0.46 0.59 0.59
Table 13: Training with distractors, Llama-3.2-1B
General domain Other domains
Distr. NoRR Top20 LLMEval Distr. No RR Top20 LLMEval
Llama3-8B 0.89 0.97 1.02 0.65 0.93 1.03 1.05 0.64
Llama32-1B 0.96 0.96 0.99 0.53 0.88 0.96 1.00 0.58
FullLoRA 0.83 0.93 0.95 0.65 0.91 0.96 0.96 0.49
LoRAQKAtt 0.88 0.92 0.91 0.62 0.88 0.90 0.92 0.53
LoRAAtt 0.84 0.92 0.93 0.65 0.971.00 1.00 0.48
LoRAMLP 0.84 0.92 0.94 0.66 0.95 0.98 0.98 0.49
FullLoRA + distilled labels 0.87 0.93 0.95 0.64 0.83 0.93 0.96 0.63
+ distractors
LoRAQKAtt 0.93 0.91 0.92 0.60 0.92 0.92 0.94 0.53
FullLoRA 0.94 0.95 0.98 0.61 0.98 0.99 0.97 0.48
FullLoRA + distilled labels 0.98 0.94 0.91 0.60 0.94 0.94 0.91 0.61
Table 14: Robustness metric for different types of noise (1) distractors (2) No reranking (3) top 20; for different
domains (1) General (NQ and PopQA) (2) Other domains (Bioasq12b, TechQA, FiQA).
19

Adapting Large Language Models for
Multi-Domain Retrieval-Augmented-Generation
Adaptation TypeMatch Recall LLMeval
Llm-1B Llm-8B Llm-1B Llm-8B Llm-1B Llm-8BKILT-NQVanilla RAG 0.557 0.698 0.651 0.729 0.6 0.698
FullLora 0.506 0.779 0.614 0.69 0.7 0.779
LoraQKAtt 0.506 0.756 0.57 0.654 0.67 0.756
LoraAtt 0.536 0.765 0.601 0.675 0.69 0.765
LoraMLP 0.558 0.776 0.618 0.684 0.71 0.776
FullLora-distill 0.644 0.729 0.731 0.759 0.7 0.729
LoraQKAtt-distill 0.63 0.726 0.719 0.763 0.69 0.726
LoraAtt-distill 0.645 0.732 0.729 0.764 0.7 0.732
LoraMLP-distill 0.655 0.728 0.741 0.764 0.7 0.728
FullLora-distractors 0.527 0.588 0.68
LoraQKAtt-distractors 0.49 0.555 0.66
FullLora-distill-distractors 0.617 0.703 0.68PopQAVanilla RAG 0.595 0.6 0.627 0.687 0.47 0.6
FullLora 0.516 0.642 0.578 0.609 0.61 0.642
LoraQKAtt 0.516 0.639 0.542 0.608 0.57 0.639
LoraAtt 0.545 0.641 0.574 0.611 0.6 0.641
LoraMLP 0.546 0.645 0.577 0.613 0.6 0.645
FullLora-distill 0.661 0.586 0.679 0.708 0.57 0.586
LoraQKAtt-distill 0.629 0.601 0.646 0.711 0.58 0.601
LoraAtt-distill 0.652 0.59 0.67 0.712 0.58 0.59
LoraMLP-distill 0.66 0.589 0.678 0.709 0.57 0.589
FullLora-distractors 0.478 0.509 0.53
LoraQKAtt-distractors 0.484 0.511 0.54
FullLora-distill-distractors 0.597 0.619 0.53GeneralVanilla RAG 0.58 0.65 0.64 0.71 0.53 0.65
FullLora 0.55 0.71 0.6 0.65 0.65 0.71
LoraQKAtt 0.51 0.7 0.56 0.63 0.62 0.7
LoraAtt 0.54 0.7 0.59 0.64 0.65 0.7
LoraMLP 0.55 0.71 0.6 0.65 0.66 0.71
FullLora-distill 0.65 0.66 0.71 0.73 0.64 0.66
LoraQKAtt-distill 0.63 0.66 0.68 0.74 0.63 0.66
LoraAtt-distill 0.65 0.66 0.7 0.74 0.64 0.66
LoraMLP-distill 0.66 0.66 0.71 0.74 0.64 0.66
FullLora-distractors 0.5 0.55 0.61
LoraQKAtt-distractors 0.49 0.53 0.6
FullLora-distill-distractors 0.61 0.66 0.6
Table 15: RAG Adaptation, General Domain
20

Adapting Large Language Models for
Multi-Domain Retrieval-Augmented-Generation
Adaptation TypeMatch Recall LLMeval
Llm-1B Llm-8B Llm-1B Llm-8B Llm-1B Llm-8BBioASQ-12bVanilla RAG 0.295 0.762 0.432 0.614 0.67 0.762
FullLora 0.448 0.8 0.511 0.599 0.73 0.8
LoraQKAtt 0.448 0.764 0.566 0.499 0.71 0.764
LoraAtt 0.418 0.801 0.508 0.6 0.72 0.801
LoraMLP 0.423 0.793 0.523 0.589 0.73 0.793
FullLora-distill 0.499 0.795 0.646 0.673 0.75 0.795
LoraQKAtt-distill 0.463 0.778 0.605 0.681 0.74 0.778
LoraAtt-distill 0.491 0.786 0.635 0.664 0.76 0.786
LoraMLP-distill 0.479 0.795 0.627 0.665 0.75 0.795
FullLora-distractors 0.426 0.527 0.73
LoraQKAtt-distractors 0.429 0.557 0.72
FullLora-distill-distractors 0.509 0.644 0.75CovidQAVanilla RAG 0.131 0.614 0.429 0.487 0.51 0.614
FullLora 0.17 0.595 0.244 0.307 0.53 0.595
LoraQKAtt 0.17 0.595 0.325 0.371 0.55 0.595
LoraAtt 0.12 0.591 0.234 0.296 0.53 0.591
LoraMLP 0.117 0.591 0.232 0.295 0.53 0.591
FullLora-distill 0.156 0.637 0.457 0.478 0.59 0.637
LoraQKAtt-distill 0.155 0.633 0.451 0.478 0.57 0.633
LoraAtt-distill 0.158 0.625 0.461 0.48 0.59 0.625
LoraMLP-distill 0.159 0.633 0.467 0.477 0.59 0.633
FullLora-distractors 0.107 0.224 0.51
LoraQKAtt-distractors 0.138 0.281 0.52
FullLora-distill-distractors 0.122 0.418 0.57Biomedical AvgVanilla RAG 0.21 0.69 0.43 0.55 0.59 0.69
FullLora 0.27 0.7 0.38 0.45 0.63 0.7
LoraQKAtt 0.31 0.68 0.45 0.44 0.63 0.68
LoraAtt 0.27 0.7 0.37 0.45 0.62 0.7
LoraMLP 0.27 0.69 0.38 0.44 0.63 0.69
FullLora-distill 0.33 0.72 0.55 0.58 0.67 0.72
LoraQKAtt-distill 0.31 0.71 0.53 0.58 0.66 0.71
LoraAtt-distill 0.32 0.71 0.55 0.57 0.67 0.71
LoraMLP-distill 0.32 0.71 0.55 0.57 0.67 0.71
FullLora-distractors 0.27 0.38 0.62
LoraQKAtt-distractors 0.28 0.42 0.62
FullLora-distill-distractors 0.32 0.53 0.66
Table 16: RAG Adaptation, Biomedical Domain.
21

Adapting Large Language Models for
Multi-Domain Retrieval-Augmented-Generation
Adaptation TypeMatch Recall LLMeval
Llm-1B Llm-8B Llm-1B Llm-8B Llm-1B Llm-8BFiQAVanilla RAG 0.0 0.51 0.201 0.23 0.49 0.51
FullLora 0.004 0.434 0.034 0.06 0.39 0.434
LoraQKAtt 0.004 0.522 0.105 0.102 0.44 0.522
LoraAtt 0 0.423 0.035 0.044 0.39 0.423
LoraMLP 0 0.442 0.031 0.056 0.37 0.442
FullLora-distill 0 0.606 0.228 0.233 0.55 0.606
LoraQKAtt-distill 0 0.59 0.235 0.241 0.54 0.59
LoraAtt-distill 0 0.594 0.233 0.233 0.54 0.594
LoraMLP-distill 0 0.606 0.231 0.236 0.55 0.606
FullLora-distractors 0 0.027 0.35
LoraQKAtt-distractors 0.002 0.077 0.42
FullLora-distill-distractors 0 0.212 0.53RobustQA-LifestyleVanilla RAG 0.003 0.616 0.492 0.506 0.57 0.616
FullLora 0.004 0.496 0.041 0.063 0.43 0.496
LoraQKAtt 0.004 0.603 0.123 0.14 0.54 0.603
LoraAtt 0.002 0.476 0.041 0.047 0.42 0.476
LoraMLP 0.002 0.495 0.037 0.065 0.41 0.495
FullLora-distill 0.002 0.661 0.331 0.359 0.62 0.661
LoraQKAtt-distill 0.002 0.659 0.389 0.397 0.6 0.659
LoraAtt-distill 0.002 0.653 0.35 0.365 0.6 0.653
LoraMLP-distill 0.002 0.662 0.336 0.362 0.63 0.662
FullLora-distractors 0.002 0.042 0.43
LoraQKAtt-distractors 0.002 0.098 0.49
FullLora-distill-distractors 0.002 0.298 0.59RobustQA-RecreationVanilla RAG 0.0 0.584 0.43 0.404 0.47 0.584
FullLora 0.002 0.49 0.07 0.093 0.42 0.49
LoraQKAtt 0.002 0.55 0.126 0.142 0.48 0.55
LoraAtt 0.001 0.477 0.063 0.076 0.4 0.477
LoraMLP 0.001 0.492 0.065 0.096 0.41 0.492
FullLora-distill 0.001 0.618 0.355 0.376 0.56 0.618
LoraQKAtt-distill 0.001 0.628 0.347 0.372 0.55 0.628
LoraAtt-distill 0.001 0.607 0.366 0.38 0.56 0.607
LoraMLP-distill 0.001 0.614 0.355 0.376 0.56 0.614
FullLora-distractors 0.0 0.066 0.4
LoraQKAtt-distractors 0.0 0.1 0.44
FullLora-distill-distractors 0.001 0.316 0.52RobustQA-ScienceVanilla RAG 0.001 0.636 0.481 0.5 0.5 0.636
FullLora 0.004 0.484 0.065 0.074 0.42 0.484
LoraQKAtt 0.004 0.591 0.134 0.16 0.48 0.591
LoraAtt 0.001 0.47 0.058 0.061 0.4 0.47
LoraMLP 0.001 0.494 0.064 0.079 0.43 0.494
FullLora-distill 0.003 0.693 0.361 0.376 0.64 0.693
LoraQKAtt-distill 0.003 0.669 0.377 0.407 0.62 0.669
LoraAtt-distill 0.001 0.681 0.37 0.382 0.63 0.681
LoraMLP-distill 0.003 0.696 0.362 0.382 0.64 0.696
FullLora-distractors 0.001 0.074 0.43
LoraQKAtt-distractors 0.001 0.097 0.45
FullLora-distill-distractors 0.004 0.329 0.63
Table 17: RAG Adaptation, Long-Form domain (1).
22

Adapting Large Language Models for
Multi-Domain Retrieval-Augmented-Generation
Adaptation TypeMatch Recall LLMeval
Llm-1B Llm-8B Llm-1B Llm-8B Llm-1B Llm-8BRobustQA-TechnologyVanilla RAG 0.0 0.635 0.442 0.433 0.53 0.635
FullLora 0.003 0.49 0.068 0.083 0.43 0.49
LoraQKAtt 0.003 0.572 0.118 0.139 0.49 0.572
LoraAtt 0.002 0.472 0.063 0.067 0.41 0.472
LoraMLP 0.001 0.487 0.061 0.083 0.41 0.487
FullLora-distill 0.0 0.707 0.319 0.336 0.63 0.707
LoraQKAtt-distill 0.001 0.693 0.347 0.353 0.61 0.693
LoraAtt-distill 0.0 0.697 0.333 0.339 0.62 0.697
LoraMLP-distill 0.001 0.705 0.323 0.34 0.63 0.705
FullLora-distractors 0 0.069 0.41
LoraQKAtt-distractors 0.002 0.09 0.43
FullLora-distill-distractors 0.001 0.295 0.6RobustQA-WritingVanilla RAG 0 0.669 0.449 0.45 0.55 0.669
FullLora 0.0 0.557 0.044 0.059 0.45 0.557
LoraQKAtt 0.0 0.619 0.097 0.105 0.53 0.619
LoraAtt 0 0.519 0.039 0.044 0.42 0.519
LoraMLP 0 0.56 0.043 0.064 0.43 0.56
FullLora-distill 0 0.734 0.323 0.35 0.66 0.734
LoraQKAtt-distill 0 0.72 0.314 0.369 0.64 0.72
LoraAtt-distill 0 0.727 0.339 0.36 0.65 0.727
LoraMLP-distill 0 0.734 0.325 0.355 0.66 0.734
FullLora-distractors 0 0.047 0.45
LoraQKAtt-distractors 0.0 0.076 0.48
FullLora-distill-distractors 0 0.286 0.64Long-Form AvgVanilla RAG 0.0 0.61 0.42 0.42 0.52 0.61
FullLora 0.0 0.49 0.05 0.07 0.42 0.49
LoraQKAtt 0.0 0.58 0.12 0.13 0.49 0.58
LoraAtt 0.0 0.47 0.05 0.06 0.41 0.47
LoraMLP 0.0 0.49 0.05 0.07 0.41 0.49
FullLora-distill 0.0 0.67 0.32 0.34 0.61 0.67
LoraQKAtt-distill 0.0 0.66 0.33 0.36 0.59 0.66
LoraAtt-distill 0.0 0.66 0.33 0.34 0.6 0.66
LoraMLP-distill 0.0 0.67 0.32 0.34 0.61 0.67
FullLora-distractors 0.0 0.05 0.41
LoraQKAtt-distractors 0.0 0.09 0.45
FullLora-distill-distractors 0.0 0.29 0.59
Table 18: RAG Adaptation, Long-Form domain (2).
23

Adapting Large Language Models for
Multi-Domain Retrieval-Augmented-Generation
Adaptation TypeMatch Recall LLMeval
Llm-1B Llm-8B Llm-1B Llm-8B Llm-1B Llm-8BParaphraseRCVanilla RAG 0.378 0.625 0.481 0.605 0.49 0.625
FullLora 0.366 0.655 0.434 0.528 0.55 0.655
LoraQKAtt 0.366 0.649 0.434 0.541 0.53 0.649
LoraAtt 0.373 0.654 0.437 0.525 0.55 0.654
LoraMLP 0.367 0.654 0.428 0.524 0.55 0.654
FullLora-distill 0.43 0.617 0.537 0.605 0.55 0.617
LoraQKAtt-distill 0.42 0.625 0.524 0.609 0.53 0.625
LoraAtt-distill 0.43 0.616 0.537 0.605 0.54 0.616
LoraMLP-distill 0.432 0.62 0.539 0.607 0.55 0.62
FullLora-distractors 0.343 0.405 0.53
LoraQKAtt-distractors 0.34 0.406 0.51
FullLora-distill-distractors 0.384 0.494 0.5SyllabusQAVanilla RAG 0.053 0.257 0.289 0.31 0.3 0.257
FullLora 0.092 0.346 0.149 0.178 0.33 0.346
LoraQKAtt 0.092 0.379 0.203 0.257 0.35 0.379
LoraAtt 0.091 0.374 0.152 0.18 0.3 0.374
LoraMLP 0.095 0.362 0.154 0.18 0.29 0.362
FullLora-distill 0.08 0.366 0.347 0.342 0.35 0.366
LoraQKAtt-distill 0.069 0.32 0.319 0.334 0.33 0.32
LoraAtt-distill 0.073 0.358 0.339 0.339 0.34 0.358
LoraMLP-distill 0.072 0.361 0.337 0.343 0.34 0.361
FullLora-distractors 0.097 0.167 0.32
LoraQKAtt-distractors 0.106 0.208 0.35
FullLora-distill-distractors 0.103 0.341 0.35TechQAVanilla RAG 0.019 0.591 0.347 0.468 0.54 0.591
FullLora 0.035 0.46 0.121 0.214 0.36 0.46
LoraQKAtt 0.035 0.59 0.202 0.275 0.44 0.59
LoraAtt 0.019 0.457 0.113 0.205 0.35 0.457
LoraMLP 0.023 0.448 0.12 0.189 0.38 0.448
FullLora-distill 0.019 0.649 0.394 0.445 0.58 0.649
LoraQKAtt-distill 0.016 0.658 0.386 0.462 0.58 0.658
LoraAtt-distill 0.016 0.647 0.401 0.448 0.58 0.647
LoraMLP-distill 0.019 0.651 0.398 0.444 0.61 0.651
FullLora-distractors 0.018 0.119 0.37
LoraQKAtt-distractors 0.039 0.176 0.46
FullLora-distill-distractors 0.019 0.357 0.54Context-Critical AvgVanilla RAG 0.15 0.49 0.37 0.46 0.44 0.49
FullLora 0.16 0.49 0.23 0.31 0.41 0.49
LoraQKAtt 0.16 0.54 0.28 0.36 0.44 0.54
LoraAtt 0.16 0.5 0.23 0.3 0.4 0.5
LoraMLP 0.16 0.49 0.23 0.3 0.41 0.49
FullLora-distill 0.18 0.54 0.43 0.46 0.49 0.54
LoraQKAtt-distill 0.17 0.53 0.41 0.47 0.48 0.53
LoraAtt-distill 0.17 0.54 0.43 0.46 0.49 0.54
LoraMLP-distill 0.17 0.54 0.42 0.46 0.5 0.54
FullLora-distractors 0.15 0.23 0.4
LoraQKAtt-distractors 0.16 0.26 0.43
FullLora-distill-distractors 0.17 0.4 0.46
Table 19: RAG Adaptation, Context-Critical domain.
24

Adapting Large Language Models for
Multi-Domain Retrieval-Augmented-Generation
Adaptation TypeMatch Recall LLMeval
Llm-1B Llm-8B Llm-1B Llm-8B Llm-1B Llm-8BSearchQAVanilla RAG 0.431 0.745 0.457 0.726 0.53 0.745
FullLora 0.433 0.824 0.562 0.725 0.68 0.824
LoraQKAtt 0.433 0.797 0.46 0.7 0.59 0.797
LoraAtt 0.471 0.832 0.498 0.734 0.62 0.832
LoraMLP 0.521 0.826 0.549 0.727 0.67 0.826
FullLora-distill 0.569 0.797 0.598 0.72 0.7 0.797
LoraQKAtt-distill 0.563 0.795 0.593 0.748 0.69 0.795
LoraAtt-distill 0.57 0.79 0.602 0.73 0.69 0.79
LoraMLP-distill 0.574 0.791 0.603 0.72 0.7 0.791
FullLora-distractors 0.525 0.552 0.67
LoraQKAtt-distractors 0.433 0.461 0.59
FullLora-distill-distractors 0.586 0.618 0.69
Table 20: RAG Adaptation, Web Search.
Adaptation TypeMatch Recall LLMeval
Llm-1B Llm-8B Llm-1B Llm-8B Llm-1B Llm-8BAvgVanilla RAG 0.176 0.61 0.443 0.511 0.51 0.61
FullLora 0.19 0.575 0.252 0.306 0.5 0.58
LoraQKAtt 0.185 0.616 0.286 0.335 0.53 0.62
LoraAtt 0.184 0.568 0.244 0.297 0.49 0.57
LoraMLP 0.19 0.576 0.25 0.303 0.49 0.58
FullLora-distill 0.219 0.657 0.45 0.483 0.6 0.66
LoraQKAtt-distill 0.211 0.65 0.446 0.495 0.59 0.65
LoraAtt-distill 0.217 0.65 0.455 0.486 0.6 0.65
LoraMLP-distill 0.218 0.656 0.452 0.484 0.61 0.66
FullLora-distractors 0.18 0.244 0.49
LoraQKAtt-distractors 0.176 0.264 0.5
FullLora-distill-distractors 0.21 0.423 0.58
Table 21: RAG Adaptation, Average across all datasets.
25