# DocQAC: Adaptive Trie-Guided Decoding for Effective In-Document Query Auto-Completion

**Authors**: Rahul Mehta, Kavin R, Indrajit Pal, Tushar Abhishek, Pawan Goyal, Manish Gupta

**Published**: 2026-04-20 13:30:45

**PDF URL**: [https://arxiv.org/pdf/2604.18257v1](https://arxiv.org/pdf/2604.18257v1)

## Abstract
Query auto-completion (QAC) has been widely studied in the context of web search, yet remains underexplored for in-document search, which we term DocQAC. DocQAC aims to enhance search productivity within long documents by helping users craft faster, more precise queries, even for complex or hard-to-spell terms. While global historical queries are available to both WebQAC and DocQAC, DocQAC uniquely accesses document-specific context, including the current document's content and its specific history of user query interactions.
  To address this setting, we propose a novel adaptive trie-guided decoding framework that uses user query prefixes to softly steer language models toward high-quality completions. Our approach introduces an adaptive penalty mechanism with tunable hyperparameters, enabling a principled trade-off between model confidence and trie-based guidance. To efficiently incorporate document context, we explore retrieval-augmented generation (RAG) and lightweight contextual document signals such as titles, keyphrases, and summaries.
  When applied to encoder-decoder models like T5 and BART, our trie-guided framework outperforms strong baselines and even surpasses much larger instruction-tuned models such as LLaMA-3 and Phi-3 on seen queries across both seen and unseen documents. This demonstrates its practicality for real-world DocQAC deployments, where efficiency and scalability are critical. We evaluate our method on a newly introduced DocQAC benchmark derived from ORCAS, enriched with query-document pairs. We make both the DocQAC dataset (https://bit.ly/3IGEkbH) and code (https://github.com/rahcode7/DocQAC) publicly available.

## Full Text


<!-- PDF content starts -->

DocQAC: Adaptive Trie-Guided Decoding for Effective
In-Document Query Auto-Completion
Rahul Mehta
mehtarahul@microsoft.com
Microsoft Corporation
Hyderabad, India
Indian Institute of Technology
Kharagpur, IndiaKavin R V
kavinrv13@gmail.com
Indian Institute of Technology
Kharagpur
Kharagpur, IndiaIndrajit Pal
pal.indrajit99@gmail.com
Independent
Bengaluru, India
Tushar Abhishek
tabhishek@microsoft.com
Microsoft Corporation
Hyderabad, IndiaPawan Goyal
pawang.iitk@gmail.com
Indian Institute of Technology
Kharagpur
Kharagpur, IndiaManish Gupta
gmanish@microsoft.com
Microsoft Corporation
Hyderabad, India
Abstract
Query auto-completion (QAC) has been widely studied in the con-
text of web search, yet remains underexplored for in-document
search, which we termDocQAC.DocQACaims to enhance search
productivity within long documents by helping users craft faster,
more precise queries, even for complex or hard-to-spell terms. Un-
like traditional WebQAC systems,DocQACcan leverage rich doc-
ument context, having access not only to the partially typed user
query and global historical queries, but also the content of the cur-
rent document itself, and crucially, the document-specific history
of user query interactions.
To address this setting, we propose a novel adaptive trie-guided
decoding framework that uses user query prefixes to softly steer
language models toward high-quality completions. Our approach
introduces an adaptive penalty mechanism with tunable hyperpa-
rameters, enabling a principled trade-off between model confidence
and trie-based guidance. To efficiently incorporate document con-
text, we explore retrieval-augmented generation (RAG) and light-
weight contextual document signals such as titles, keyphrases, and
summaries.
When applied to encoder–decoder models like T5 and BART,
our trie-guided framework outperforms strong baselines and even
surpasses much larger instruction-tuned models such as LLaMA-3
and Phi-3 in seen-query settings. This demonstrates its practicality
for real-worldDocQACsystem deployments, where efficiency and
scalability are critical. We evaluate our method on a newly intro-
ducedDocQACbenchmark derived from ORCAS, enriched with
query–document pairs. We make both theDocQACdataset1and
code2publicly available.
1Dataset- https://bit.ly/3IGEkbH
2Code- https://github.com/rahcode7/DocQAC
This work is licensed under a Creative Commons Attribution 4.0 International License.
SIGIR ’26, Melbourne, VIC, Australia
©2026 Copyright held by the owner/author(s).
ACM ISBN 979-8-4007-2599-9/2026/07
https://doi.org/10.1145/3805712.3809644Document Prefix WebQAC outputDocQACoutput
https://en.
wikipedia.org/
wiki/Parisfr freejobalert, freepik, free games,
free fire max, from tv series, friends,
friendship quotes, freecell, frank
lampard, free job alert 2025france capital, france tourism,
france history, france landmarks,
france culture, france travel guide,
france famous cities, france eiffel
tower, france paris attractions,
france paris museums
https://en.
wikipedia.org/
wiki/Brad_Pitta amazon, adobe acrobat, anydesk,
australia vs india, amazon prime, al-
lahabad, aiden markram, american
airlines, amitabh bachchan, alice in
borderlandamerican actor, academy awards,
angelina jolie, angelina jolie hus-
band, aniston, alcoholism, academy
award nominations, angelina jolie
and brad pitt relationship, autobi-
ography of brad pitt, a river runs
through it
https://history.
house.gov/People/
Office/Speakers-
List/spea speaker, speak no evil, speak, speak-
ing, speaker cleaner, speak now,
spear, speaker test, speaker clean-
ing, spearmint teaspeaker of the house, speaker of
the house history, speaker of house
of representatives, speaker history,
speakers of the us house, speakers,
speaker henry clay, speaker of the
house current status, speaker of the
house duties, speaker of the house
responsibilities
Table 1: Examples of top few results from WebQAC versus
DocQACsystems.
CCS Concepts
•Computing methodologies →Natural language processing;
•Information systems→Query suggestion.
Keywords
In-Document Search, Trie-Guided Decoding, Query Auto Comple-
tion, Large Language Models, Retrieval Augmented Generation
ACM Reference Format:
Rahul Mehta, Kavin R V, Indrajit Pal, Tushar Abhishek, Pawan Goyal,
and Manish Gupta. 2026.DocQAC: Adaptive Trie-Guided Decoding for
Effective In-Document Query Auto-Completion. InProceedings of the 49th
International ACM SIGIR Conference on Research and Development in Informa-
tion Retrieval (SIGIR ’26), July 20–24, 2026, Melbourne, VIC, Australia.ACM,
New York, NY, USA, 11 pages. https://doi.org/10.1145/3805712.3809644
1 Introduction
Query Auto Completion (QAC) is the first service with which search
users interact, offering ranked query suggestions based on the
partially typed user query (which we call a prefix). Traditionally,arXiv:2604.18257v1  [cs.IR]  20 Apr 2026

SIGIR ’26, July 20–24, 2026, Melbourne, VIC, Australia Rahul Mehta, Kavin R V, Indrajit Pal, Tushar Abhishek, Pawan Goyal, and Manish Gupta
the most common approach to solving this task involves utilizing
highly efficient trie-based data structures [ 17] with techniques like
Most Popular Completions (MPC) [ 2]. Recently, deep learning-based
approaches that utilize sequence-to-sequence models trained on
historical queries have been employed to generate high-quality
completions [ 19,21,27,36]. We will refer to such QAC systems for
Web search as WebQAC systems.
1.1 Motivation
While QAC has been extensively studied in the context of web
search (WebQAC), where suggestions are driven by global popular-
ity and historical query logs, there has been surprisingly no research
dedicated to QAC for in-document search. We term this under-
explored task as Document Query Auto-Completion (DocQAC).
DocQACsystems can help users in (a) reducing search time by pre-
dicting and suggesting search terms, which in turn helps quickly
find relevant information within long documents, and (b) improv-
ing search accuracy by minimizing typographical errors, especially
for documents with rare and orthographically challenging terms.
Overall, aDocQACsystem can enhance user productivity, ensure
that users are using the correct terminology and phrases, and assist
users who may not be familiar with the exact terms or keywords
to use, making it easier for them to navigate and locate specific
information without having to sift through long documents.
The extra document context inDocQACsystems brings an addi-
tional challenge. How do you best leverage this document context
and associated metadata? How do you handle long documents?
1.1.1 Difference betweenDocQACand WebQAC.ADocQACsys-
tem differs from a WebQAC system in several fundamental ways.
•Intent locality:In WebQAC, user intent is often naviga-
tional (finding a specific website, e.g., “youtube”) or broad
information. InDocQAC, user intent is tightly coupled to
the current document and is exploratory in nature. As a
result, globally frequent queries are often irrelevant, while
rare or document-specific terms become critical for accurate
completion.
•Vocabulary shift: Documents frequently contain named
entities, domain-specific phrases, and long-tail terminology
that do not appear frequently in global query logs.
•Contextual Grounding: The same prefix may require radi-
cally different completions depending on the document being
viewed. Unlike WebQAC, which prioritizes global consensus,
DocQACrequires local contextual grounding: suggestions
must be relevant to the specific content of the document at
hand. This capability is critical for enhancing productivity
in long-form document consumption, aiding users in nav-
igating dense technical texts, and mitigating orthographic
errors for rare, document-specific terms.
Table 1 illustrates this gap. For identical prefixes, WebQAC sys-
tems produce globally popular suggestions that are largely irrele-
vant to the document context, while aDocQACsystem must surface
completions grounded in the document’s content and semantics.
1.1.2 Use Cases ofDocQAC.ADocQACsystem can appear in doc-
ument interaction tools like Adobe Acrobat’s in-document search,
text editors, IDEs, and enterprise document management systems.Here, users frequently write short prefixes to locate sections, enti-
ties, or phrases within a document. For example, when navigating
long PDFs, reviewing technical documentation, or searching policy
and legal documents.DocQACauto-completion in these scenarios
improves efficiency by suggesting contextually relevant, document-
specific terms during query formulation.
1.2 Core Contributions
Overall, we make the following main contributions in this paper.
•Formalization of theDocQACTask: We formalize docu-
ment specific Query Auto-Completion (DocQAC) as a dis-
tinct problem paradigm, differentiating it from standard We-
bQAC by its strict faithfulness constraints and “cold-start”
document challenges. To support this, we also release a
dataset benchmark for theDocQACtask.
•Trie-Guided Inference Time Decoding Framework:We
develop a novel adaptive trie-guided decoding framework
to softly bias encoder-decoder language model generation
toward valid completions without adding new parameters.
This method resolves the “generative drift” problem inherent
in standard LLMs by dynamically pruning tokens that do
not appear in the document’s query log or body.
•Efficiency and Performance Gains: We demonstrate that
small, guided encoder-decoder models (e.g., T5-Small, BART-
Base) significantly outperform unguided Large Language
Models (LLaMA-3, Phi-3). This establishes a new state-of-
the-art for efficient, latency-constrained query completion.
•Understanding effect of Context Representations: We
also perform detailed ablation studies to understand the role
of document context, such as using document title and URL
tokens, summaries or keyphrases extracted from documents.
To facilitate further research in this direction, we make the dataset1
and code2publicly available.
2 Related Work
2.1 Autocompletion Methods
2.1.1 Trie-Based Methods.Given a prefix, the MPC model, pro-
posed in [ 2], extracts a limited number ( 𝑘) of completions from
a character-trie structure (also referred to as the main trie) built
using a corpus of past queries.
2.1.2 Generative Methods.QueryBlazer [ 20] is a fully generative,
low-latency query auto-completion method that is capable of lever-
aging both previously encountered queries and generating comple-
tions for new, unseen queries. At inference time, the user’s query
prefix is input into the subword encoder as a sequence of characters.
The encoder then produces all potential top-k subword sequences
that can be generated from the partial input provided.
Deep learning methods like Hierarchical RNN Encoder-decoder [ 33]
with pointer generator [ 10], GRUs with user and time representa-
tions [ 14] and Transformer-based hierarchical encoder [ 38] have
also been studied. While showing suggestions it is important to
not show defective suggestions and prefixes. To avoid defects, re-
searchers have used LSTMs for inappropriate query suggestion

DocQAC: Adaptive Trie-Guided Decoding for Effective In-Document Query Auto-Completion SIGIR ’26, July 20–24, 2026, Melbourne, VIC, Australia
Figure 1:DocQACDataset Construction Pipeline (detailed in Section 4). A document 𝐷in ORCAS has clicked queries 𝑄. Query
Augmentation augments 𝐷with non-clicked queries 𝑄′which are similar to 𝑄. Relevance Labeling filters queries in 𝑄and𝑄′
that are irrelevant to𝐷. Click Popularity Estimation estimates pseudo-counts for𝑄′queries. Finally we create dataset splits.
detection [ 37], A* search and Markov noisy channel models for on-
line spell correction [ 12], and character RNNs [ 35]. There have also
been attempts togenerateeffective QAC suggestions [ 23–25,25,26].
2.2DocQAC-like systems
Unfortunately, there has been no work on theDocQACproblem
which we study in this paper. There have been some attempts on
type-ahead completions for Teams [ 34] and email composition sys-
tems like GMail [ 6] and Outlook [ 34], which help users to compose,
whereasDocQAChelps the users in finding information (Navi-
gation). Further, existing WebQAC methods cannot be trivially
adapted for theDocQACtask since it involves careful modeling of
the additional document context. Our proposed system can be effec-
tively integrated with in-document search systems like KTRL+F [ 28]
to enhance their functionality.
2.3 Constrained Decoding
Recent studies on constrained decoding have focused on constrain-
ing the output of language models, primarily using hard constraints.
Early methods such as grid beam search [ 16] and dynamic beam
allocation [ 29] introduced mechanisms to enforce the inclusion of
specific words or phrases in the generated text. In parallel, grammar-
based decoding approaches [ 15] have emerged as another direction
for structured generation. However, such hard constraint mecha-
nisms often limit the generative flexibility of models, making them
less suitable for open-ended tasks such as WebQAC orDocQAC,
where a balance between adherence and fluency is crucial. Recently,
prefix trie-based methods [ 5] have been explored to improve beam
search efficiency during decoding.
Although, there exists work on utilizing tries in constrained gen-
eration in other tasks of information retrieval [ 3,9] ,no prior work
exists that applies an adaptive penalty schedule at decoding-time on
trie constrained generation.In document-level QAC, such adaptive
constraints become even more meaningful where the document
context encoded in the trie naturally guides generation, allowing
the model to generate plausible and context-aligned completions
while maintaining the advantages of neural generation.3DocQACProblem Formulation
LetVbe the vocabulary of all terms and Dbe a corpus of docu-
ments. Given a query prefix 𝑝consisting of a sequence of tokens
(𝑤1,...,𝑤 𝑘), the goal ofQuery Auto-Completion (QAC)is to
generate the optimal completion suffix 𝑠such that the full query
𝑞=𝑝⊕𝑠maximizes the posterior probability.
Existing WebQAC (Global Optimization):Standard WebQAC op-
erates in an “open-world” setting where the objective is to maximize
the likelihood of 𝑞given the prefix 𝑝and the global user search
historyH𝑔𝑙𝑜𝑏𝑎𝑙 . Mathematically, this approximates the marginal
probability over all possible latent contexts (or documents𝑑):
𝑞∗
𝑤𝑒𝑏=argmax
𝑞∈V∗𝑃(𝑞|𝑝,H 𝑔𝑙𝑜𝑏𝑎𝑙)(1)
Consequently, WebQAC systems are biased toward “head” queries
that are frequent across the entire corpus, often ignoring specific
document contexts. The support of the distribution is effectively the
entire vocabularyV, meaning any plausible string has non-zero
probability (𝑃>0).
DocQAC(Conditional Constrained Optimization):In contrast,
DocQACis a “closed-world” task where the completion is strictly
conditioned on a specific observed document 𝑑𝑐𝑢𝑟𝑟as well as the
global user search history for this document 𝐻𝑐𝑢𝑟𝑟. The objective
changes to maximizing the conditional probability:
𝑞∗
𝑑𝑜𝑐=argmax
𝑞∈V∗𝑃(𝑞|𝑝,𝑑 𝑐𝑢𝑟𝑟,H𝑐𝑢𝑟𝑟)(2)
Unlike WebQAC, where unseen queries are smoothed,Doc-
QACtreats suggestions outside the document’s scope as hallu-
cinations. This fundamental shift requires models to suppress the
global token popularity and exclusively rely on the local likelihood
𝑃(𝑞|𝑝,𝑑 𝑐𝑢𝑟𝑟), necessitating the trie-guided constrained decoding
approach proposed in this work.
4DocQACDataset
We utilize the ORCAS [ 8] dataset which contains 1.4M documents
and 10M distinct real-world user queries. We illustrate our dataset
construction process in Fig. 1 and describe it in this section.

SIGIR ’26, July 20–24, 2026, Melbourne, VIC, Australia Rahul Mehta, Kavin R V, Indrajit Pal, Tushar Abhishek, Pawan Goyal, and Manish Gupta
Obtaining Frequency Counts and Initial Preprocessing.We
perform the following preprocessing steps: (1) We retain queries
that are at least 3 characters long. (2) We remove duplicate query-
document (QD) pairs. (3) We retain documents with more than 10
queries but fewer than 500 queries to ensure a balanced represen-
tation. (4) We choose only those query-document pairs where the
document exists in the TREC dataset3. Thus, we obtain document
text content from the TREC dataset.
Next, we create train/validation/test dataset splits while consid-
ering the temporal aspects of the dataset. To do this, we look up
the (query, document) pairs from the ORCAS dataset against Bing
logs from Jul-Aug 20224, and prepare the training splits from a 30
day window, a validation split from a 4-day window, and a test set
from a 10-day period. This process also helps us obtain the number
of times a document was clicked for a given query which in turn is
helpful in ranking suggestions for QAC.
Query Augmentation via Similar Queries.To augment the OR-
CAS dataset, for each unique clicked query 𝑄, we start by collecting
100 most similar queries 𝑄′from Bing query logs (Jul-Aug 20224)
using cosine similarity over DeBERTa-v3-base5embeddings. To
ensure the quality and relevance of these similar queries for the
DocQACtask, we applied a series of rigorous filtering steps. First,
to create a publicly shareable dataset, we retained only those sim-
ilar queries that appeared verbatim within the content of some
document in the collection. Second, we remove any similar queries
that were near-duplicates (Levenshtein distance ≤1) or if they were
already part of clicked queries in ORCAS for the same document.
Relevance Labeling.A critical component of our dataset is the
relevance label for each query-document pair. We use GPT-46as a bi-
nary relevance classifier to evaluate whether each query-document
pair is relevant or not, going beyond historical clicks. This helps in
removing false positives due to user behavior patterns or popularity
biases. Overall, this leads to significant enhancement to the dataset
where 48.79% are the original clicked queries 𝑄and the remaining
are similar queries𝑄′.
Click Popularity Estimation.As query and document click counts
are not available for these similar queries (since they did not di-
rectly lead to clicks to the documents), we estimate them based on
our original dataset as follows. For a similar query 𝑄′, we identify
its top-5 closest matches from the existing queries 𝑄in our dataset
(which have frequency counts), and calculate a weighted average of
their historical click counts, where the weight is the cosine similar-
ity between the similar query 𝑄′and its top-5 most similar queries.
A similar methodology was used to assign click counts to a similar
query𝑄′for a document by using counts of top-5 closest clicked
queries of that document.
Test Dataset Creation.Within the test set, we create 4 distinct
test splits based on the presence of test set queries and documents
in the training dataset. Specifically, if both the query and the doc-
ument are present in both the training and test sets, we call this
subset the “seen query-seen document (SS)” test set. Conversely,
if neither the query nor the document appears in the training set,
the corresponding subset is labeled as the “unseen query-unseen
3https://msmarco.z22.web.core.windows.net/msmarcoranking/msmarco-docs.tsv.gz
4We use an older time point to match timeline of original queries in ORCAS.
5https://huggingface.co/microsoft/deberta-v3-base
6https://cdn.openai.com/papers/gpt-4.pdfDataset Docs (Query, Doc) Pairs Prefixes Avg Query Len
Train 22,453 316,813 Dynamic 17.1
Validation 7,266 31,682 Dynamic 16.8
Test-Seen Q Seen D 2,611 3,000 53,862 19.0
Test-Seen Q Unseen D 712 3,000 52,678 18.5
Test-UnSeen Q Seen D 2,485 3,000 54,145 20.5
Test-Unseen Q UnSeen D 1,068 3,000 52,488 17.1
Table 2: Dataset Statistics for different Dataset Splits. Length
is in characters. “Dynamic” implies that query in the (query,
doc) pair is split into prefix and suffix by choosing a random
split point in every batch.
Prefix
Document
Summary
Keyphrases
Related 
DocsRelevant Text 
ChunksRelevant Text 
Chunks
Truncated Text
TitleURL
Reranker
ANN 
Index2ChatGPT
YAKEANN 
Index1
ANN 
Index1Models
Tries
        Global Tries        DocC Tries        DocQ Tries
Query Blazer
NLG Models          T5          T5 Guided  
          BART          BART Guided
          Phi -3
          LLaMA -3.2
Figure 2: Input Representations andDocQACMethods
document (UU)” test set. Additionally, we define two other splits:
the “unseen query-seen document (US)” test set, where the query
is absent from the training set but the document is present, and
the “seen query-unseen document (SU)” test set, where the query
is seen during training but the document is not. Each split in the
test set contains 3,000 (query, document) pairs. Table 2 shows the
statistics of various subsets of our dataset.
For each (query, document) pair in the train set, a sample is
created in the dataset by randomly choosing a split point within
the query. We refer to the string to the left (right) of the split point
in the query as the prefix (suffix or completion). Thus, each sample
consists of a prefix and a document as input and the goal is to
generate the suffix.
5 Methods forDocQAC
We follow various modeling strategies forDocQACthat gives a
complete view at the spectrum of trade-off between accuracy and
latency. We investigate several ML and DL approaches for the task,
including trie-based methods, QueryBlazer, and neural language
models.
5.1DocQACTries
To improve the coverage of main trie, we design an alternative
method that utilizes a suffix-trie to handle prefixes with no matches
in the main trie of the training queries. Specifically, we experiment
with three different tries.
•Global Query Trie: All the queries of all the training docu-
ments are indexed into a single global trie. Given a test prefix

DocQAC: Adaptive Trie-Guided Decoding for Effective In-Document Query Auto-Completion SIGIR ’26, July 20–24, 2026, Melbourne, VIC, Australia
and a document, the completions are selected using MPC
from this global trie.
•Document Content (or DocC) Tries: We build a DocC trie for
each document by utilizing the ngrams of the document text.
•Document Query (or DocQ) Tries: We build a DocQ trie for
each document using the document-specific subset of his-
torical queries. Among our 4 test sets, DocQ tries cannot be
made for test sets with unseen documents.
5.2 Neural Language Models
We also experiment with popular Transformer-based models for
generatingDocQACsuggestions. Specifically, we use the BART-
base [ 22] and T5-small [ 31] encoder-decoder models. Among Large
Language models, we experiment with 2 models : Phi-3.5 [ 1] and
LLaMA-3.2 [ 13] by LoRA-finetuning them on our training datasets.
Particularly, we use google-t5/t5-small, facebook/bart-base, meta-
llama/Llama-3.2-3B-Instruct and microsoft/phi-3.5-mini-instruct
checkpoints. Refer Appendix B for hyperparameters.
The prefix and the completion constitute the source and target
sequences, respectively, for these models. While training both the
models, each training query is split stochastically into prefix and
suffix. During inference, we use beam search to generate a ranked
list of completions.
6 Trie-Guided Decoding
Our analysis of performance of various methods reveals the follow-
ing trade-off: trie-based methods excel at recall but fail at generaliza-
tion, while generative language models excel at generalization but
often lack relevance and precision. In “seen query, seen document
(SS)” scenarios, traditional trie-based approaches perform excep-
tionally well, as the task is primarily one of recall from a known set
of queries. However, their rigidity becomes a critical failure point
in “unseen query, seen document (US)” cases, where they are fun-
damentally unable to generate novel queries that are not present in
their pre-compiled structure. Conversely, LMs demonstrate strong
performance on unseen query test sets by leveraging their genera-
tive capabilities to formulate novel, contextually relevant queries.
However, their limitation is a lack of reliable grounding. In “seen
query, unseen document (SU)” scenarios, an LM may fail to sug-
gest a known, popular query, instead generating a fluent but less
effective alternative.
To address this, we propose an adaptive trie-guided mechanism
that biases the model’s generation toward trie-conforming com-
pletions while retaining flexibility for contextual adaptation. At
each decoding step, we compute a bias term that is subtracted from
the logits of tokens which do not match with the completions in
the trie. This bias is annealed over time according to the length
of the prefix and the diversity of the beam index, controlled by 3
hyperparameters: Initial bias, 𝛼and𝛽.𝛼controls decay with respect
to prefix length. A higher 𝛼reduces the trie’s influence as prefix
length increases.𝛽controls decay with respect to beam depth.
annealed_bias=initial_bias·𝑒−𝛼·length·𝑒−𝛽·beam_depth
This encourages higher-ranked beams to follow the trie more strictly
while allowing diversity in lower-ranked beams. Further, we also
Figure 3: Illustration of Trie-Guided LLM vs Unguided LLM
for a sample user prefix inDocQACsetting.
introduce an initial bias, which is a large initial penalty to strongly
prioritize trie-conforming tokens at early decoding steps.
Figure 3 showcases the difference between a Trie Guided LLM
vs an unguided LLM with an example. Our trie constrained system
can validates if a query exists in a trie and updates its decoding
process while inferencing, thereby helping the user in getting to the
actual completion faster. The formal pseudocode for our constrained
decoding strategy is presented in Algorithm 1.
Algorithm 1:Soft Trie-Guided Decoding
Input:Language modelM, prefix𝑝, trieT 𝑟, beam size𝐾,
initial bias𝑏 0, decay parameters𝛼,𝛽, maximum
decoding steps𝑇
Output:Top-𝐾query completions
1Initialize beam setB 0←{𝑝}
2for𝑡=1to𝑇do
3Initialize candidate setC←∅
4foreach beam𝑏 𝑖∈B𝑡−1do
5Compute logitsz 𝑡←M(𝑏 𝑖)
6Retrieve valid next tokens
V𝑖←T 𝑟.ValidNextTokens(𝑏 𝑖)
7 Compute annealed bias 𝛿𝑖=𝑏0·𝑒−𝛼·|𝑏 𝑖|·𝑒−𝛽·rank(𝑏 𝑖)
8foreach token𝑣in vocabularydo
9if𝑣∉V 𝑖then
10z 𝑡[𝑣]←z 𝑡[𝑣]−𝛿 𝑖
11p 𝑡←Softmax(z 𝑡)
12Expand beam𝑏 𝑖using top tokens fromp 𝑡
13Add expanded beams toC
14PruneCto top-𝐾beams to formB 𝑡
15ReturnB 𝑇
Tokenization Mismatch Problem. A core challenge in applying
trie-based guidance to language models for tasks like query sugges-
tion is the mismatch between the granularity of user inputs, which
is at character level, while the language model decodes at a subword
level. Note that tries are built at a subword level to support such
guidance. We build a serialized BytesTrie for fast lookup during
decoding.

SIGIR ’26, July 20–24, 2026, Melbourne, VIC, Australia Rahul Mehta, Kavin R V, Indrajit Pal, Tushar Abhishek, Pawan Goyal, and Manish Gupta
Given a prefix of a seen query, we need to ensure that its tokeniza-
tion should match with a path in the trie. The last few characters
of a partial token should also match with a node in the trie. For
example, for the query “machine learning”, it is important to have
“machine lea” also in the trie although tokenization of the original
query just leads to two tokens “machine” and “learning”. Hence,
for each query 𝑞of the training dataset 𝐷, we generate all possible
character-level prefix-suffix splits for all the queries, tokenize prefix
and suffix separately, and store the overall tokenized string in the
trie. However, at test time, this could lead to multiple redundant
prefix path matches in the trie for super-strings. For example, con-
sider a new query “machine learning goals” and prefix is “machine
learning”. This would match several paths in the trie including
“machine lea rning”, “machine learn ing” etc. To avoid a match with
multiple paths in the trie, at trie creation time, we mark the transi-
tion from prefix to suffix by inserting a unique separator token (e.g.,
[SEP_SPLIT] ) between the last prefix and first suffix token IDs. At
test time, we match the path corresponding to prefix+ [SEP_SPLIT]
against the trie.
7 Utilizing Document Context forDocQAC
We evaluate our models discussed in Section 5, using different
approaches to incorporate document context. Specifically, we ex-
periment with short input representations, longer document repre-
sentations and retrieval augmented generation based approaches.
For trie-based methods and QueryBlazer, we use the document
content to rerank top 100 suggestions using cosine similarity based
on sentence-BERT (msmarco-distilbert-base-v4) embeddings.
7.1 Short Input
To quantify the improvements obtained using the document context,
we experiment with aPrefix Only (P)setting andTitle + URL
+ Prefix (P+TU)setting - with document title and URL as input
context.
7.2 Longer Document Representations
We experiment with three different longer document representa-
tions: document text, keyphrases (KPs) and summaries.
•Title + URL + Document + Prefix (P+TUD): We pass
trimmed document content as input. Given our model’s max-
imum context length of 512 tokens, we allocate a maximum
of 32 tokens each for the title and URL and 352 tokens for
document context and rest for the prefix and system prompt.
•Title + URL + KPs + Prefix (P+TUK): We use YAKE [ 4] to
extract keyphrases with a maximum n-gram length of three,
and we limit the number of extracted phrases to fifty.
•Title + URL + Summary + Prefix (P+TUS): For this, we
utilize ChatGPT 3.5 Turbo7with 16,134 context window and
generate offline summaries of up to 300 words and pass as
representative document context
7https://chat.openai.com/chat7.3 Retrieval Augmented Generation (RAG)
•RAG using current document: In this setting, for a given
query prefix, we utilize only the current document to ex-
tract relevant chunks. The relevant chunks are retrieved and
ranked using sparse or dense similarity metrics as follows.
–Sparse Retrieval (Sparse RAG). We retrieve top 𝑘(=20) sen-
tences from the documents with highest BM25 [ 32] simi-
larity between the prefix and sentences.
–Dense Retrieval (Dense RAG). We split each document
into fixed-size (200 characters) chunks with some overlap
(30 characters) between two neighboring chunks. We in-
dex chunks using all-mpnet-base-v2 [ 30] embeddings and
FAISS [ 11]. Next, we use the vector similarity between
the full prefix and chunks to extract extract top 𝑘(=20)
chunks.
•RAG using related documents (Rel+Dense RAG): Given
the current document, we first obtain top 10 similar docu-
ments from the training set based on top similarity scores
usingmsmarco-distilbert-base-v4document embeddings and
FAISS.msmarco-distilbert-base-v4has been trained on the
MS MARCO dataset, the same dataset from which our docu-
ments are derived.
8 Evaluation Metrics
We categorize our metrics forDocQACinto 2 categories and evalu-
ate top 10 suggestions for each prefix on these metrics.
8.1 Primary Metrics
We prioritize the evaluation metrics that directly quantify the suc-
cess of the system in meeting the user’s core objective: finding the
correct document content with minimal physical effort.
Typing Effort Saved (TES): InDocQACscenarios (e.g., techni-
cal manuals, legal briefs, medical records), users often search for
long, complex, or hard-to-spell domain-specific terms. The TES met-
ric, inspired by [ 34], is computed as TES= 1−No. of typed characters
Query Length.
Set Model Input MRR nDCG 𝛼BLEU 𝑟𝑟SBMRR PPN PRN TES
SS DocQ tries P0.738 0.277 0.477 0.803 0.8200.824 0.889
SS DocQ tries P + TUS 0.688 0.274 0.472 0.778 0.8140.8260.889
SS DocQ tries Sparse RAG 0.688 0.274 0.471 0.777 0.8130.8260.889
SS DocQ tries P + TUD 0.686 0.273 0.471 0.776 0.8130.8260.889
SS DocQ tries Dense RAG 0.686 0.273 0.471 0.776 0.8130.8260.889
SS DocQ tries Rel + Dense RAG 0.686 0.273 0.471 0.775 0.8130.8260.889
SS DocQ tries P + TUK 0.680 0.273 0.470 0.770 0.8120.8260.889
SS DocQ-Guided BART P + TUK 0.720 0.080 0.349 0.793 0.692 0.7900.896
SU Global-Guided BART P + TUK0.7110.078 0.296 0.778 0.664 0.730 0.880
SU LLaMA-3.2 P + TUS 0.462 0.088 0.277 0.6680.7400.691 0.813
SU Global-Guided T5 Sparse RAG 0.706 0.076 0.2990.7790.6740.737 0.882
SU Global-Tries P + TUS 0.5250.156 0.3090.595 0.664 0.665 0.625
US Phi-3.5 P + TU0.4010.052 0.282 0.557 0.7100.6700.756
US LLaMA-3.2 P + TUD 0.3810.0770.283 0.5480.7240.658 0.730
US Phi-3.5 P + TUS 0.392 0.052 0.280 0.554 0.714 0.6670.760
US LLaMA-3.2 P + TU 0.400 0.0700.285 0.5620.711 0.668 0.728
UU Phi-3.5 P + TU0.4610.059 0.279 0.628 0.729 0.693 0.809
UU LLaMA-3.2 P + TUS 0.4420.0850.280 0.6150.7430.680 0.792
UU Phi-3.5 Sparse RAG 0.456 0.061 0.2830.6290.7300.6950.809
UU Phi-3.5 P + TUS 0.452 0.060 0.278 0.625 0.736 0.6930.821
UU LLaMA-3.2 Dense RAG 0.452 0.0840.2850.624 0.731 0.686 0.785
UU LLaMA-3.2 P + TUK 0.458 0.080 0.2830.6290.729 0.688 0.789
Table 3: Model and input combinations that result in at least
one best metric value for any of the 4 test sets.

DocQAC: Adaptive Trie-Guided Decoding for Effective In-Document Query Auto-Completion SIGIR ’26, July 20–24, 2026, Melbourne, VIC, Australia
Set Model Input MRR nDCG 𝛼BLEU 𝑟𝑟SBMRR PPN PRN TES
SS BART P + TUK 0.486 0.059 0.286 0.649 0.672 0.707 0.750
SU BART P + TUK 0.440 0.054 0.259 0.615 0.663 0.703 0.697
SU T5 Sparse RAG 0.425 0.051 0.260 0.617 0.677 0.710 0.685
SS DocQ-Guided BART P + TUK 0.720 0.080 0.349 0.793 0.692 0.790 0.896
SU Global-Guided BART P + TUK 0.711 0.078 0.296 0.778 0.664 0.730 0.880
SU Global-Guided T5 Sparse RAG 0.706 0.076 0.299 0.779 0.674 0.737 0.882
Table 4: Comparison of unguided vs. Trie-guided decoding
(Ours). Dashed line separates the two groups.
Set Model Input MRR nDCG 𝛼BLEU 𝑟𝑟SBMRR PPN PRN TES
SS Phi-3.5 P + TUK 0.450 0.058 0.293 0.643 0.724 0.696 0.829
SS LLaMA-3.2 P + TUK 0.446 0.079 0.296 0.645 0.724 0.689 0.791
SU Phi-3.5 P + TUK 0.476 0.061 0.277 0.676 0.724 0.706 0.842
SU LLaMA-3.2 P + TUK 0.477 0.083 0.279 0.676 0.725 0.699 0.808
SU Phi-3.5 Sparse RAG 0.467 0.085 0.278 0.667 0.729 0.697 0.808
SU LLaMA-3.2 Sparse RAG 0.473 0.063 0.279 0.670 0.730 0.706 0.833
SS DocQ-Guided BART P + TUK 0.720 0.080 0.349 0.793 0.692 0.790 0.896
SU Global-Guided BART P + TUK 0.711 0.078 0.296 0.778 0.664 0.730 0.880
SU Global-Guided T5 Sparse RAG 0.706 0.076 0.299 0.779 0.674 0.737 0.882
Table 5: Comparison of large models (Phi-3.5, LLaMA-3.2)
with our Trie-Guided models (T5 and BART). Dashed line
separates the two groups.
Unlike ranking metrics which evaluate a static list, TES simulates
the dynamic, end-to-end user interaction (typing →looking→
selecting). It answers the most critical practical question: “Did this
system actually make the query creation faster for users?” In our
DocQACsettings, saving keystrokes is the ultimate goal.
Mean Reciprocal Rank (MRR): AsDocQACis a navigation
task, where the goal is to locate a specific string instantly, the rank
of the first correct answer is paramount. In auto-completion, users
rarely scan beyond the top 1 or 2 results. MRR strictly penalizes
any system that buries the correct answer lower in the list.
8.2 Secondary Metrics
These metrics are valuable for understanding why a model succeeds
or fails, and for diagnosing specific error types (e.g., hallucination
vs. drift) and can be used as a set of secondary metrics.
Semantic Match (SBMRR): SBMRR gives credit for understand-
ing the user’s intent, even if the exact string matching failed. This
helps researchers understand if a model is “smart but imprecise”
(high SBMRR, low MRR). Instead of lexical match, we find a seman-
tic match between the reference query and its auto-completions. We
use a transformer based model, Sentence-BERT [ 30](all-MiniLM-
L6-v2)to compute both the query and suggestions’ representations.
We consider a match if the semantic similarity is≥0.9.
Partial Match Metrics (PPN and PRN): To diagnose failure
modes beyond binary success, we employ Partial Precision (PPN)
and Partial Recall (PRN) NDCG. PPN penalizes “hallucinated suf-
fixes” (e.g., suggesting “Data Lake” instead of “Data Base”) by mea-
suring how much of the suggestion is valid. Conversely, PRN iden-
tifies “truncation errors” (e.g., stopping at “Machine” instead of
“Machine Learning”) by measuring how much of the target phrase
was captured, ensuring models balance precision with complete-
ness.
Diversity and N-gram Overlap ( 𝛼-NDCG, BLEU 𝑅𝑅): Clarke
et al. [ 7] defined the 𝛼NDCG metric for evaluating diversity. Forexample, in a coding document, the prefix “pro” might match “pro-
cess”, “program”, and “protect”. A bad model might fill the top
10 slots with just variations of one word (“process”, “processing”,
“processed”). 𝛼-NDCG penalizes this redundancy. BLEU 𝑅𝑅acts as
a “soft” MRR by weighting n-gram overlap by reciprocal rank. In
DocQAC, it rewards models for capturing document-specific vocab-
ulary, which distinguishes useful “near-misses” from completely
irrelevant hallucinations.
9 Results
9.1 Overall Results
For each of the 4 test sets, Table 3 shows the model and input com-
binations that result in at least one best metric value. All results are
reported on the 4DocQACtest sets, each containing approximately
52,000-53,000 prefix-query pairs (see Table 2).
We observe that for SS set, DocQ tries perform the best for all
metrics except TES. Our guided decoding approach, leveraging
DocQ tries, achieves the highest TES with BART-Base, outperform-
ing DocQ tries, LLaMA-3.2 (3B), and Phi-3.5 (3B). For SU set, the
DocQ tries are not available. In this setting, our BART model with
KPs as input and Global Tries for guided decoding, provides the
best MRR. T5 model with Sparse RAG as input and Global Tries
for guided decoding achieves the highest TES and PPN. For US and
UU, LLaMA-3.2 and Phi-3.5 lead to the best aggregated results.
9.1.1 Best Guided vs Unguided counterparts.Table 4 compares the
3 best performing guided decoding models (from Table 3) with their
unguided counterparts. Across all metrics, the guided methods
consistently outperform the unguided ones. For example, for the
BART model, applying guided decoding yields a 48.1% improvement
in MRR (from 0.486 to 0.720) and a 19.4% increase in TES (from 0.750
to 0.896) for SS test set. We observe similar gains for SeenQ-SeenD
for both BART and T5 models. For T5 with Sparse RAG as input,
applying guided decoding leads to 66% improvement in MRR alone
(from 0.425 to 0.706) and 27.7% improvement in TES (from 0.685 to
0.882).
9.1.2 Best Guided Fine Tuned (BART and T5) vs Instruction Fine
Tuned (Phi-3.5 and LLaMA).Table 5 shows comparison between
our three best-performing guided decoding models with large mod-
els with the same input. Notably, both the T5- and BART-based
models, when enhanced with trie decoding, consistently and sub-
stantially outperform the instruction fine-tuned LLaMA-3.2, while
being approximately 52.5×smaller (T5) and 23×smaller (BART).
9.2 Ablation Studies
9.2.1 Analysis of input representations.We present impact of each
input representation in Table 6.For SS set, the best models are with
DocQ tries with various inputs followed by our guided decoding
models BART and then T5 with various inputs.For SU set, our
global trie-guided T5 model with Sparse RAG has the highest TES,
and global trie-guided BART with keyphrases as input has the best
MRR. Notably, these models outperform global tries by a large
margin.For US set, LLaMA-3.2 and Phi-3.5 with title and URL
perform the best in all metrics except PRN.For UU set, LLaMA-
3.2 and Phi-3.5, with either the title and URL, or both title and

SIGIR ’26, July 20–24, 2026, Melbourne, VIC, Australia Rahul Mehta, Kavin R V, Indrajit Pal, Tushar Abhishek, Pawan Goyal, and Manish Gupta
Input MRR𝛼N BLEU 𝑅𝑅 SBMRR PPN PRN TES MRR𝛼N BLEU 𝑅𝑅 SBMRR PPN PRN TES
SS, DocQ tries US, LLaMA-3.2
P 0.738 0.277 0.477 0.803 0.8200.824 0.889 0.245 0.061 0.229 0.325 0.669 0.572 0.499
P+TU0.687 0.274 0.470 0.776 0.815 0.825 0.889 0.400 0.0700.285 0.5620.711 0.668 0.728
P+TUD0.686 0.273 0.471 0.776 0.8130.8260.889 0.388 0.072 0.283 0.551 0.713 0.663 0.724
P+TUK0.680 0.273 0.470 0.770 0.8120.8260.889 0.387 0.071 0.283 0.552 0.707 0.664 0.714
P+TUS0.688 0.274 0.472 0.778 0.8140.8260.889 0.3810.0770.283 0.5480.7240.658 0.730
Sparse RAG0.688 0.274 0.471 0.777 0.8130.8260.889 0.388 0.074 0.284 0.550 0.713 0.662 0.716
Dense RAG0.686 0.273 0.471 0.776 0.813 0.824 0.889 0.384 0.075 0.285 0.546 0.711 0.660 0.714
Rel+Dense RAG0.686 0.273 0.471 0.775 0.8130.8260.889 0.370 0.067 0.281 0.543 0.696 0.659 0.697
SU, Global-Guided T5 UU, LLaMA-3.2
P0.384 0.042 0.221 0.463 0.593 0.636 0.552 0.241 0.060 0.210 0.318 0.665 0.568 0.487
P+TU0.538 0.059 0.267 0.675 0.662 0.714 0.771 0.460 0.078 0.282 0.626 0.730 0.688 0.789
P+TUD0.552 0.060 0.270 0.690 0.662 0.718 0.788 0.449 0.080 0.281 0.619 0.734 0.685 0.792
P+TUK0.549 0.060 0.268 0.686 0.664 0.715 0.787 0.458 0.080 0.2830.6290.729 0.688 0.789
P+TUS0.528 0.058 0.264 0.672 0.669 0.711 0.805 0.4420.0850.280 0.6150.7430.680 0.792
Sparse RAG0.708 0.076 0.2990.7790.6740.737 0.8820.453 0.082 0.282 0.624 0.732 0.686 0.792
Dense RAG0.554 0.061 0.270 0.695 0.667 0.718 0.807 0.452 0.0840.2850.624 0.731 0.686 0.785
Rel+Dense RAG0.553 0.060 0.268 0.688 0.667 0.717 0.798 0.446 0.076 0.280 0.617 0.716 0.685 0.769
Table 6: Results with different input representations, for the best performing model for each of the 4 test sets.
MRR
𝛼𝛼NDCG
BLEURR
SBMRRPPNPRNMRR
𝛼𝛼NDCG
BLEURR
SBMRRPPNPRNMRR
𝛼𝛼NDCG
BLEURR
SBMRRPPNPRNMRR
𝛼𝛼NDCG
BLEURR
SBMRRPPNPRN
Figure 4: Performance Comparison across metrics for varying prefix lengths. Left to right: SS DocQ tries (P), SU Global-Guided
T5 (Sparse RAG), US LLaMA-3.2 (P+TU), UU LLaMA-3.2 (P+TUK). Note TES cannot be computed for this experiment.
URL along with a document or summary as context, achieve the
best results. Also, for bothUS and UU set, we observe the next
best models to be BART and T5; tries perform the worst. Thus, for
unseen queries, leveraging neural LMs is recommended.
9.3 Other Analysis
9.3.1 Varying Prefix Length Analysis.We assess performance of
DocQACmodels in these prefix length categories: 1-5, 6-10, 11-15,
16-20 and 20+ characters. Fig. 4 shows results for our best models
across all the 4 test sets. We observe that for each set across all
metrics, results improve as the prefix length increases. We believe
this is because longer prefixes provide the best clues for the models
to predict accurate suggestions.
9.3.2 Qualitative Analysis.Table 8 shows predictions for 4 samples
(one from each test set) across all models, for best guided models and
their unguided counterparts. We observe that DocQ tries cannot
Global DocC DocQ QB T5 T5 Guided BART BART Guided Phi LLaMA
DocQ Global DocQ Global 3.5 3.2
P 8 10 11 0.2 66 73 118 63 73 190 494 905
P+TU 31 36 30 38 62 71 113 58 66 194 748 983
P+TUD 33 38 35 37 71 73 123 64 72 197 1246 1156
P+TUS 28 33 31 31 64 73 116 65 74 206 1331 1138
P+TUK 29 34 30 31 66 72 118 66 74 204 1153 1114
Sparse RAG 32 38 34 40 76 88 127 74 84 211 1162 1103
Dense RAG 71 84 63 77 99 107 150 97 111 239 1203 1105
Rel+Dense RAG 277 325 231 290 226 240 287 228 240 366 1344 1134
Table 7: Latency (in ms) per sample. T5, BART, Phi-3.5 and
LLaMA-3.2 latencies are on GPU. Others on CPU.predict anything for 3 of these samples due to query log sparsity
for those documents. For the SS sample, we observe that all guided
models are able to correctly show query suggestion matching the
user input query while their unguided counterparts are unable to
do so for the prefix “spe” for the given query “speed typing practice”.
In the SU scenario, the guided decoding models of BART are able
to complete the prefix “king” by leveraging Global Tries. In the
UU case, we observe that providing additional context like Sparse
RAG to T5 and keyphrases to BART helped with getting the correct
suggestion.
9.3.3 Latency Analysis.Table 7 reports latency across various
methods computed using batch size of 1. Latency for neural LMs
is on GPUs while for other models, we report latency on CPUs.
Clearly, neural models have high latency compared to tries or QB.
Our guided decoding models are highly practical for real-world
deployment too. They incur only a modest 12 ms latency increase
for T5 Guided DocQ + Sparse RAG (from 76 ms to 88 ms) for un-
guided vs guided scenarios, while remaining ∼15x faster than both
LLaMA-3.2 3B and Phi-3.5 3B. Similarly, DocQ-Guided BART +
(P+TUK) model incurs a latency of increase of just 8 ms (from 66
to 74 ms) while remaining 15 times faster than LLaMA-3.2 3B and
Phi-3.5 3B. Lastly, Global-Guided BART + (P+TUK) model adds a
latency of 138 ms on top of unguided version, but still being 5.5
times faster than LLaMA3 and Phi3. Although these models have
not been optimized using any of the popular optimization methods
like FasterTransformers or TensorRT-LLM, their latencies are al-
ready low enough to be deployed in practical settings. These results
demonstrate that developing specialized decoding algorithms is a

DocQAC: Adaptive Trie-Guided Decoding for Effective In-Document Query Auto-Completion SIGIR ’26, July 20–24, 2026, Melbourne, VIC, Australia
SS SU US UU
Title . Francis I of Francethe third month of the yearWalt Whitman
URL http://www.keybr.com/ https://en.wikipedia.org/wiki/Francis_I_of_
Francehttp://www.timeanddate.com/calendar/
months/https://en.wikipedia.org/wiki/Walt_Whitman
Queryspeed typing practice king francis of france the third month of the year poems of walt whitman
Prefix spe king the thi poems o
DocQ tries (P) speed test practice, speed and eliminates, speed
stayed, speed, speed test, speed for each, speed
for every, speed for, speed stayed atTrie cannot be created Trie returned no matches due to log sparsity Trie cannot be created
Phi-3.5 (P+TUK) speed typing, speed of typing, speed in typing,
speed to type, speed for typing, speed keybr,
speeds of typing, speed keyboard, speed of key-
board typing, speed typking francis, king francis i,king francis of
france, king francis the first, king of france,
king francis 1, king francis iii, kings of france,
king french, king francis 1stthe thirteen months, the thirty days, the thir-
teenth month, the thirteen month, the this
month, the thirteen, the third month, the thirty
one dayspoems of walt whitman, poems of whitman,
poems of walt, poems on walt whitman, poems
of walter whitman, poems of walt whittman,
poems on leaves of grass, poems of, poems of
walt w, poems of walt watson
LLaMA-3.2
(P+TUK)speed typing, speed type, speed of typing,
speed typing test,speed typing practice, spell
typing, spell typer, spell type, speed of type,
spell keybrking francis, king francis i, kings of france,
king frances, king francisco, king francois, king
frances i, kings francis, kings francis i, kings
of francthe third month, the thirteenth month, the this
month, the this month is, the this months, the
third month of, the thirteens, the thirtieth, the
thirdpoems of walt whitman, poems of whit-
man, poems of walt whitan, poems of walt
whiman, poems of walter whitman,poems
of walt whitmans, poems of whitman’s, po-
ems of whitmans
DocQ-Guided
BART (P+TUK)speed typing practice, speed typing wpm,
speed of your typing, speed of the keyboard,
speed of a keyboard, speed of your keyboard,
speed typing.com, speed of typing x, speed
practice for keyboard, speed of a computerTrie cannot be created the third month of the year, the thir month
september, the this month of the year, the thist
month of the year, the things of the year, the
thin months of the year, the this month, the
third months of the year, the thir seasons of
the year, the thingths of the yearTrie cannot be created
Global-Guided
BART (P+TUK)speed of typing, speed typing, speed of a com-
puter,speed typing practice, speed checks,
speak and type, speed website, speed of audio,
speed up your computer, speed type gamekings of france,king francis of france, kings
in france, king henry viii, king francis, king
henry vii, king if england, kings, king of france
timeline, kings ukthe third person, the things of the year,the
third month of the year, the thingths of the
year, the third months of the year, the thing
months of the year, the thir month september,
the things in the year, the thirds of the year,
the this month of the yearpoems of walt whitman,poems of walt
whitmanbooks, poems of walt walt breath,
poems of walt walt Whitman, poems of walt
walt, poems of walt rhodea, poems of walt
rhodeo, poems of walt walt rock, poems of
walt chattman, poems of walt walt cod
BART (P+TUK) speed your typing, speed of typing, speed to
type, speed of your typing, speed typing, speed
writers, speed of the keyboard, speed of a key-
board, speed of your keyboard, speed your key-
boardkings of france, king of france, king francis
viii, king and queen of france, king francis i,
king francis of france, king’s death of france,
king’s history of france, king francis the great,
kings of france listthe third month of the year, the thir month
september, the this month of the year, the thist
month of the year, the things of the year, the
thin months of the year, the this month, the
third months of the year, the thir seasons of
the year, the thingths of the yearpoems of walt whitman, poems of walt walt,
poems of walt, poems of walt codman, poems
of walt norfolk, poems of walt washington,
poems of walt potman,poems of walt whit-
manbooks, poems of walt walt breath, poems
of walt marshall
Table 8: Sample predictions from our best models comparing guided and unguided models.
highly effective and resource-efficient alternative to simply scaling
up model size.
10 Conclusion
In this paper, we propose a novel task ofDocQACfor in-document
query auto completions. We also release the newly createdDocQAC
benchmark dataset. We conduct extensive experiments by integrat-
ing various strategies, including RAG, using document summaries
as context for LLMs, and combining these approaches with tra-
ditional methods such as tries. We establish strong baselines for
these techniques and further evaluate larger language models like
LLaMA and Phi-3. Building on this, we introduce a novel adaptive
trie-guided decoding framework that enhances the performance of
language models like T5 and BART by guiding their outputs rather
than imposing hard constraints. Our results demonstrate that this
approach significantly outperforms the established baselines and
even surpasses large instruction-tuned models. Overall, our find-
ings showcase the potential of adaptive trie-guided decoding as a
practical and efficient method for controlled language generation,
and encourage further research in context-aware and nuanced text
generation.
A Query Document Relevance Classifier
We utilize GPT4 with the system prompt as shown in Fig. 5 to
classify whether the query is relevant for a document or not.B Hyper-parameter Settings
Compute: Experiments were performed on a machine with 8
NVIDIA V100 GPUs for training neural models. For CPU-based
experiments, we used an AMD Ryzen 9 7900X3D 12-Core Processor
(4.40 GHz) with 64GB RAM.
T5 and BART: For T5, we use batch size=24,learning rate=1e-
4,epochs=30,maximum input length=512, AdamW. For inference,
we set beam size=25,max output length=48. For BART,we use 8
for document content (RAG, Summary, and KPs) and 18 for other
experiments.
Phi-3.5 and Llama-3.2: Both models are fine-tuned using LoRA [ 18]
for all linear layers (lora_r=16, lora_alpha=32,lora_dropout=0.05),
with AdamW,learning rate=5e-5,fp16 precision, max len=512,batch
size=4 and epochs=5. Inference uses beam search with beam size=10,
length penalty=1.0, and a 40-token output limit.
Trie-Guided Decoding: We perform extensive hyperparameter
tuning across 𝛼∈{ 0.05,0.1,0.2,0.5},𝛽∈{ 0.05,0.1,0.2,0.5},bias∈
{20,30,40}. For DocQ tries based guided decoding, we use (P+Sparse
RAG), while for global-tries based guided decoding, we use (P+TUD)
and run for SS and SU settings as they are the best performing mod-
els. With global tries, the best performance on both SS and SU is
obtained with 𝛼= 0.1,𝛽= 0.05, and bias= 40. In contrast, for
DocQ trie-guided decoding, the optimal configuration is 𝛼=0.1,
𝛽=0.2, and bias= 40on SS, while 𝛼=0.5,𝛽=0.2, and bias= 20
performs best on SU.
References
[1]Marah Abdin, Jyoti Aneja, Hany Awadalla, Ahmed Awadallah, Ammar Ahmad
Awan, Nguyen Bach, Amit Bahree, Arash Bakhtiari, Jianmin Bao, Harkirat Behl,

SIGIR ’26, July 20–24, 2026, Melbourne, VIC, Australia Rahul Mehta, Kavin R V, Indrajit Pal, Tushar Abhishek, Pawan Goyal, and Manish Gupta
<|im_start|>system
[system](#instructions)
# Task
Given a document, the following query was retrieved by an in-
formation retrieval (IR) system as a potential query a user might
type for searching for content within the document. Your task is
to accurately classify whether the query is truly relevant to the
document or not.
# Input
Document: “{body}”
Query: “{Query}”
id : “{id}”
docid : “{docid}”
# Output
Provide your classification judgement of the query relevance for
the documentSTRICTLYin the following JSON format:
{
“query_relevance”: bool,
“id”: string,
“Query”: string,
“docid”: string
}
# The query is relevant (true) if:
– The document contains specific information that directly an-
swers the query.
– The document provides background knowledge, explanations,
or context that meaningfully relates to the query.
– The document discusses entities, topics, or events explicitly men-
tioned in the query.
– A user who issued this query would find the document useful or
informative in addressing their information need.
# The query is not relevant (false) if:
– The document does not address the topic, entities, or intent ex-
pressed in the query.
– The content is too vague, general, or off-topic to satisfy the
query’s information needs.
– There is no logical or semantic connection between the query
and the document content.
– The document might mention some terms from the query, but
in a completely unrelated context.
<|im_end|>
Figure 5: The system prompt used for document-query rele-
vance classification using GPT4
et al.2024. Phi-3 technical report: A highly capable language model locally on
your phone.arXiv preprint arXiv:2404.14219(2024).
[2]Ziv Bar-Yossef and Naama Kraus. 2011. Context-sensitive query auto-completion.
InWWW. 107–116.
[3]Michele Bevilacqua, Giuseppe Ottaviano, Patrick Lewis, Scott Wen-tau Yih, Sebas-
tian Riedel, and Fabio Petroni. 2022. Autoregressive Search Engines: Generating
Substrings as Document Identifiers. InAdvances in Neural Information Processing
Systems, Vol. 35. 31668–31683.
[4]Ricardo Campos, Vítor Mangaravite, Arian Pasquali, Alípio Jorge, Célia Nunes,
and Adam Jatowt. 2020. YAKE! Keyword extraction from single documents using
multiple local features.Information Sciences509 (2020), 257–289.
[5]Brian J Chan, Jui-Hung Cheng, Mao Xun Huang, Chao-Ting Chen, and Hen-Hsen
Huang. 2025. Efficient beam search for large language models using Trie-based
decoding.arXiv preprint arXiv:2502.00085(2025).
[6]Mia Xu Chen, Benjamin N Lee, Gagan Bansal, Yuan Cao, Shuyuan Zhang, Justin
Lu, Jackie Tsay, Yinan Wang, Andrew M Dai, Zhifeng Chen, et al .2019. Gmail
smart compose: Real-time assisted writing. In25th KDD. 2287–2295.[7]Charles LA Clarke, Maheedhar Kolla, Gordon V Cormack, Olga Vechtomova,
Azin Ashkan, Stefan Büttcher, and Ian MacKinnon. 2008. Novelty and diversity
in information retrieval evaluation. In31st SIGIR. 659–666.
[8]Nick Craswell, Daniel Campos, Bhaskar Mitra, Emine Yilmaz, and Bodo Billerbeck.
2020. ORCAS: 18 Million Clicked Query-Document Pairs for Analyzing Search.
arXiv preprint arXiv:2006.05324(2020).
[9]Nicola De Cao, Gautier Izacard, Sebastian Riedel, and Fabio Petroni. 2021. Autore-
gressive Entity Retrieval. InICLR. https://openreview.net/forum?id=5k8F6UU39V
Spotlight.
[10] Mostafa Dehghani, Sascha Rothe, Enrique Alfonseca, and Pascal Fleury. 2017.
Learning to attend, copy, and generate for session-based query suggestion. In
2017 CIKM. 1747–1756.
[11] Matthijs Douze, Alexandr Guzhva, Chengqi Deng, Jeff Johnson, Gergely Szilvasy,
Pierre-Emmanuel Mazaré, Maria Lomeli, Lucas Hosseini, and Hervé Jégou. 2024.
The faiss library.arXiv preprint arXiv:2401.08281(2024).
[12] Huizhong Duan and Bo-June Hsu. 2011. Online spelling correction for query
completion. InWWW. 117–126.
[13] Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad
Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Amy Yang, Angela Fan,
et al. 2024. The llama 3 herd of models.arXiv preprint arXiv:2407.21783(2024).
[14] Nicolas Fiorini and Zhiyong Lu. 2018. Personalized neural language models for
real-world query auto completion. InNAACL-HLT. 208–215.
[15] Saibo Geng, Martin Josifoski, Maxime Peyrard, and Robert West. 2023. Grammar-
constrained decoding for structured NLP tasks without finetuning.arXiv preprint
arXiv:2305.13971(2023).
[16] Chris Hokamp and Qun Liu. 2017. Lexically constrained decoding for sequence
generation using grid beam search.arXiv preprint arXiv:1704.07138(2017).
[17] Bo-June Hsu and Giuseppe Ottaviano. 2013. Space-efficient data structures for
top-k completion. In22nd WWW. 583–594.
[18] Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean
Wang, Lu Wang, and Weizhu Chen. 2021. Lora: Low-rank adaptation of large
language models.arXiv preprint arXiv:2106.09685(2021).
[19] Jyun-Yu Jiang and Wei Wang. 2018. RIN: Reformulation inference network
for context-aware query suggestion. In27th ACM International Conference on
Information and Knowledge Management. 197–206.
[20] Young Mo Kang, Wenhao Liu, and Yingbo Zhou. 2021. QueryBlazer: efficient
query autocompletion framework. InWSDM. 1020–1028.
[21] Dong-Ho Lee, Zhiqiang Hu, and Roy Ka-Wei Lee. 2021. Improving Text Auto-
Completion with Next Phrase Prediction. InFindings of the Association for Com-
putational Linguistics: EMNLP 2021. 4434–4438.
[22] Mike Lewis, Yinhan Liu, Naman Goyal, Marjan Ghazvininejad, Abdelrahman Mo-
hamed, Omer Levy, Ves Stoyanov, and Luke Zettlemoyer. 2019. BART: Denoising
sequence-to-sequence pre-training for natural language generation, translation,
and comprehension.arXiv preprint arXiv:1910.13461(2019).
[23] Aishwarya Maheswaran, Kaushal Kumar Maurya, Manish Gupta, and Maunen-
dra Sankar Desarkar. 2024. DAC: quantized optimal transport reward-based
reinforcement learning approach to detoxify query auto-completion. InProceed-
ings of the 47th International ACM SIGIR Conference on Research and Development
in Information Retrieval. 608–618.
[24] Aishwarya Maheswaran, Kaushal Kumar Maurya, Manish Gupta, and Maunen-
dra Sankar Desarkar. 2024. DQAC: detoxifying query auto-completion with
adapters. InPacific-Asia Conference on Knowledge Discovery and Data Mining.
Springer, 108–120.
[25] Anubhab Mandal, Sandeep Mishra, Bishal Santra, Tushar Abhishek, Pawan Goyal,
and Manish Gupta. 2026. Chat-Ghosting: Methods for Auto-Completion in Dialog
Systems. InProceedings of the 19th Conference of the European Chapter of the
Association for Computational Linguistics (Volume 1: Long Papers). 4502–4528.
[26] Kaushal Kumar Maurya, Maunendra Sankar Desarkar, Manish Gupta, and Puneet
Agrawal. 2023. TRIE-NLG: trie context augmentation to improve personalized
query auto-completion for short and unseen prefixes.DMKD37, 6 (2023), 2306–
2329.
[27] Agnès Mustar, Sylvain Lamprier, and Benjamin Piwowarski. 2020. Using BERT
and BART for Query Suggestion. InJoint Conference of the Information Retrieval
Communities in Europe, Vol. 2621. CEUR-WS. org.
[28] Hanseok Oh, Haebin Shin, Miyoung Ko, Hyunji Lee, and Minjoon Seo. 2024.
KTRL+ F: Knowledge-Augmented In-Document Search. InNAACL-HLT. 2416–
2436.
[29] Matt Post and David Vilar. 2018. Fast lexically constrained decoding with dynamic
beam allocation for neural machine translation.arXiv preprint arXiv:1804.06609
(2018).
[30] N Reimers. 2019. Sentence-BERT: Sentence Embeddings using Siamese BERT-
Networks.arXiv preprint arXiv:1908.10084(2019).
[31] Adam Roberts, Colin Raffel, Katherine Lee, Michael Matena, Noam Shazeer,
Peter J Liu, Sharan Narang, Wei Li, and Yanqi Zhou. 2019. Exploring the limits
of transfer learning with a unified text-to-text transformer.Google, Tech. Rep.
(2019).
[32] Stephen Robertson, Hugo Zaragoza, et al .2009. The probabilistic relevance
framework: BM25 and beyond.Foundations and Trends®in Information Retrieval

DocQAC: Adaptive Trie-Guided Decoding for Effective In-Document Query Auto-Completion SIGIR ’26, July 20–24, 2026, Melbourne, VIC, Australia
3, 4 (2009), 333–389.
[33] Jun Song, Jun Xiao, Fei Wu, Haishan Wu, Tong Zhang, Zhongfei Mark Zhang, and
Wenwu Zhu. 2017. Hierarchical contextual attention recurrent neural network
for map query suggestion.TKDE29, 9 (2017), 1888–1901.
[34] Stojan Trajanovski, Chad Atalla, Kunho Kim, Vipul Agarwal, Milad Shokouhi,
and Chris Quirk. 2021. When does text prediction benefit from additional context?
an exploration of contextual signals for chat and email messages. InNAACL-HLT.
1–9.
[35] Po-Wei Wang, J Zico Kolter, Vijai Mohan, and Inderjit S Dhillon. 2018. Realtime
query completion via deep language models. (2018).[36] Sida Wang, Weiwei Guo, Huiji Gao, and Bo Long. 2020. Efficient neural query auto
completion. In29th ACM International Conference on Information & Knowledge
Management. 2797–2804.
[37] Harish Yenala, Manoj Chinnakotla, and Jay Goyal. 2017. Convolutional Bi-
directional LSTM for detecting inappropriate query suggestions in web search.
InPAKDD. Springer, 3–16.
[38] Di Yin, Jiwei Tan, Zhe Zhang, Hongbo Deng, Shujian Huang, and Jiajun Chen.
2020. Learning to generate personalized query auto-completions via a multi-view
multi-task attentive approach. InKDD. 2998–3007.