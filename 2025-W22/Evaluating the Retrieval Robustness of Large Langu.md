# Evaluating the Retrieval Robustness of Large Language Models

**Authors**: Shuyang Cao, Karthik Radhakrishnan, David Rosenberg, Steven Lu, Pengxiang Cheng, Lu Wang, Shiyue Zhang

**Published**: 2025-05-28 01:34:31

**PDF URL**: [http://arxiv.org/pdf/2505.21870v1](http://arxiv.org/pdf/2505.21870v1)

## Abstract
Retrieval-augmented generation (RAG) generally enhances large language
models' (LLMs) ability to solve knowledge-intensive tasks. But RAG may also
lead to performance degradation due to imperfect retrieval and the model's
limited ability to leverage retrieved content. In this work, we evaluate the
robustness of LLMs in practical RAG setups (henceforth retrieval robustness).
We focus on three research questions: (1) whether RAG is always better than
non-RAG; (2) whether more retrieved documents always lead to better
performance; (3) and whether document orders impact results. To facilitate this
study, we establish a benchmark of 1500 open-domain questions, each with
retrieved documents from Wikipedia. We introduce three robustness metrics, each
corresponds to one research question. Our comprehensive experiments, involving
11 LLMs and 3 prompting strategies, reveal that all of these LLMs exhibit
surprisingly high retrieval robustness; nonetheless, different degrees of
imperfect robustness hinders them from fully utilizing the benefits of RAG.

## Full Text


<!-- PDF content starts -->

Evaluating the Retrieval Robustness of Large Language Models
Shuyang Cao♠*, Karthik Radhakrishnan♡, David Rosenberg♡,
Steven Lu♡, Pengxiang Cheng♡, Lu Wang♠, Shiyue Zhang♡
Bloomberg♡University of Michigan♠
{kradhakris1, drosenberg44, slu126, pcheng134, szhang1061}@bloomberg.net
{caoshuy, wangluxy}@umich.edu
Abstract
Retrieval-augmented generation (RAG) gener-
ally enhances large language models’ (LLMs)
ability to solve knowledge-intensive tasks. But
RAG may also lead to performance degrada-
tion due to imperfect retrieval and the model’s
limited ability to leverage retrieved content. In
this work, we evaluate the robustness of LLMs
in practical RAG setups (henceforth retrieval
robustness ). We focus on three research ques-
tions: (1) whether RAG is always better than
non-RAG; (2) whether more retrieved docu-
ments always lead to better performance; (3)
and whether document orders impact results.
To facilitate this study, we establish a bench-
mark of 1500 open-domain questions, each
with retrieved documents from Wikipedia. We
introduce three robustness metrics, each cor-
responds to one research question. Our com-
prehensive experiments, involving 11 LLMs
and 3 prompting strategies, reveal that all of
these LLMs exhibit surprisingly high retrieval
robustness; nonetheless, different degrees of
imperfect robustness hinders them from fully
utilizing the benefits of RAG.1
1 Introduction
Large language models (LLMs) learn to acquire
massive amounts of knowledge through large-scale
pre-training, enabling them to answer knowledge-
intensive questions (OpenAI et al., 2024; An-
thropic, July. 2024; Meta, September 2024). How-
ever, relying exclusively on parametric knowledge
can lead to inaccuracies when dealing with unseen
or time-sensitive information, or when the model
fails to precisely retrieve relevant knowledge from
its own parameters. To alleviate these limitations,
retrieval-augmented generation (RAG) is proposed,
where external documents containing information
relevant to the task are fetched from a datastore
*Work done during an internship at Bloomberg
1We will release our evaluation harness soon.
0.3 0.4 0.5 0.6 0.7
Task Performance0.840.860.880.900.92Retrieval Robustness
1B
3B
8B
70B
32B
104B
12B
123B
4o / o3-mini
sonnet
Llama
 Command
 Mistral
 OpenAI
 ClaudeFigure 1: Comparison of retrieval robustness and QA
task performance across various LLMs. The y-axis rep-
resents robustness (geometric mean of the three robust-
ness metrics), while the x-axis represents task perfor-
mance (average across all k,o, retrievers, and datasets).
OpenAI GPT-4o and o3-mini have very close robustness
and performance.
and provided to the model as context during infer-
ence (Chen et al., 2017; Lewis et al., 2020).
Despite its potential, RAG does not always guar-
antee performance improvements. The retriever
might fail to retrieve relevant documents, and the
LLMs might be distracted by irrelevant content,
leading to performance drop (Mallen et al., 2023).
As achieving a perfect retriever remains an elusive
goal in practice, it is crucial for LLMs to behave ro-
bustly in the RAG setup to reduce the risks during
actual deployment.
Previous work has shown that LLMs are partic-
ularly vulnerable when provided with noisy con-
texts that are synthetically constructed (Chen et al.,
2024). Distracted by the specially designed mis-
leading information, models tend to produce incor-
rect outputs (Wu et al., 2024b). Despite yielding
valuable insights, synthetically constructed con-arXiv:2505.21870v1  [cs.CL]  28 May 2025

texts are dissimilar to realistic retrieved contexts
that are usually drawn from credible corpora like
Wikipedia and trusted news outlets.
To bridge this gap, this work benchmarks LLMs’
robustness under realistic RAG setups. We con-
sider an LLM retrieval robust if (1) its RAG perfor-
mance is equal to or better than its non-RAG perfor-
mance; (2) adding more retrieved documents leads
to equal or better performance; and (3) its RAG per-
formance is invariant to the order of retrieved docu-
ments. Three metrics are defined correspondingly—
no-degradation rate, retrieval size robustness, and
retrieval order robustness.
We focus on open-domain question answering—
a knowledge-intensive task where RAG is widely
adopted. We curate a benchmark of 1,500 sam-
ples by randomly drawing 500 questions each from
three datasets—Natural Questions (Kwiatkowski
et al., 2019), Hotpot QA (Yang et al., 2018),
ASQA (Stelmakh et al., 2022)—covering diverse
domains and complexities. To construct retrieved
contexts, we leverage two retrievers, including a
canonical sparse BM25 (Robertson and Zaragoza,
2009) retriever and a dense retriever based on a
strong embedding model, BGE (Xiao et al., 2023).
Both retrievers retrieve context from Wikipedia arti-
cles. For analyses of retrieval size and order robust-
ness, RAG setups with multiple retrieval sizes (5 to
100 documents) and three ways of ordering them
(original rank, reversed rank, random shuffle) are
evaluated. Our experiments cover 11 LLMs from
both open-source and proprietary families. Each
LLM is evaluated via vanilla prompting and two
more sophisticated prompting strategies: one aug-
ments the model’s own knowledge, and the other
filters relevant retrieval contexts.
We find that LLMs generally demonstrate strong
robustness, achieving over 80% scores on the geo-
metric mean of the three retrieval robustness met-
rics, as shown by Figure 1. This indicates that,
oftentimes , (1) RAG is better than non-RAG; (2)
more retrieved documents lead to better perfor-
mance; and (3) order of the documents does not
matter a lot. Nonetheless, the imperfect retrieval
robustness reflects undesired behaviors, notably the
performance trade-off among individual samples
(i.e., decreasing performance on some examples
while improving it on others), which prevents the
models from fully utilizing the benefits of RAG and
destabilizes response quality when changing the re-
trieval size or order. Such unpredictable trade-off
poses risks for realistic applications that demandconsistent outcomes. Therefore, retrieval robust-
ness provides a novel perspective for benchmark-
ing and understanding LLMs’ RAG performance.
For example, as shown in Figure 1, even if GPT-
4o/O3-mini and Claude 3.5 Sonnet have similar
RAG task performance, the higher retrieval robust-
ness of GPT-4o/O3-mini makes them more pre-
ferred in practice. Finally, we find that retrieval
robustness can be enhanced by augmenting the an-
swers generated with the model’s own knowledge,
though it also limits the potential task performance
gain from RAG.
Our contributions are summarized as follows:
•We propose sample-level metrics to rigor-
ously measure retrieval robustness —how ro-
bust LLMs handle queries in RAG setups,
which provides a new perspective of under-
standing LLM’s RAG performance.
•We compile a benchmark for evaluating re-
trieval robustness, following common RAG
setups in practice. It comprises diverse open-
domain QA tasks along with retrieved doc-
uments from Wikipedia obtained by widely-
used and strong retrievers.
•We conduct a comprehensive empirical study
of 11 modern LLMs with 3 different prompt-
ing strategies, revealing the generally good
robustness of LLMs in more realistic settings
and highlighting the consequences of their im-
perfect robustness.
2 Related Works
Retrieval-Augmented Generation (RAG) en-
hances parametric models by retrieving seman-
tically relevant information from a knowledge
base (Gao et al., 2023b; Wu et al., 2024a). Typi-
cally, it involves a retriever and a parametric lan-
guage model. RAG can potentially help adapt pre-
trained models to up-to-date knowledge, ground
models with long-tail information, and thus im-
prove factuality and accuracy (Asai et al., 2024).
The pioneering RAG framework, DrQA (Chen
et al., 2017), was introduced to tackle knowledge-
intensive open-domain question answering (QA)
tasks, which is still the main evaluation target
of recent works (Wu et al., 2024b; Chen et al.,
2024). RAG has also been used for non-knowledge-
intensive tasks like language modeling, understand-
ing, and reasoning (Borgeaud et al., 2022; Guo
et al., 2023; Izacard et al., 2024). There are many

different ways to implement RAG. Some works,
e.g., knn-LM (Khandelwal et al., 2020), retrieve
hidden states, while many other works retrieve text.
To utilize the retrieved documents, some works
modified the model architecture. e.g., FiD (Izacard
and Grave, 2021) encoded each document sepa-
rately and concatenated their hidden states in the
decoder, while RETRO (Borgeaud et al., 2022)
added a chunked cross-attention module into the
regular Transformer block. Another widely used
method is to simply include the retrieved docu-
ments directly into the input. This can be done by
putting them all together in one context (Ram et al.,
2023; Lee et al., 2024) or by generating answers
with each of them separately and ensembling the re-
sults (Guu et al., 2020; Lewis et al., 2020; Shi et al.,
2024). Some works train the retriever and the lan-
guage model jointly (Lewis et al., 2020; Borgeaud
et al., 2022; Lin et al., 2024), while others fix the
model and and only train the retriever (Ram et al.,
2023; Shi et al., 2024). In this paper, we opt for
the simplest setup: we use off-the-shelf retrievers
and LLMs, and we use the retrieved documents by
directly including them in a single context window.
This approach has become increasingly practical
with the long-context ability of modern LLMs (Lee
et al., 2024).
Retrieval Robustness. Neural language models
are shown to be easily distracted by adversarially
inserted irrelevant content (Jia and Liang, 2017; Shi
et al., 2023; Weston and Sukhbaatar, 2023). How-
ever, irrelevant context comes in naturally in any
RAG setup due to the imperfect retriever. Chen
et al. (2024) showed that the LLM-based RAG
performance goes down when increasing the noise
(i.e., documents that are relevant to the question but
do not contain any information about the answer)
rate. Wu et al. (2024b) conducted a deeper analysis
and found that highly semantically related infor-
mation is more likely to distract LLMs. Thakur
et al. (2024) evaluated LLM RAG performance
with a completely irrelevant set of documents and
observed non-trivial hallucination rates. Yoran
et al. (2024) introduced the concept of retrieval
robustness , “retrieval-robust LLMs states that: (a)
when relevant, the retrieved context should improve
model performance; (b) when irrelevant, the re-
trieved context should not hurt model performance.”
However, all these works usually handcrafted con-
trolled yet synthetic evaluation setups that mixing
irrelevant context with relevant ones. Following
the same spirit, we instead resort to a more realisticand practical setup where we simply pick the top-
Kcontexts returned by a retriever with a natural
mixture of relevant and irrelevant content. And we
extend the definition of retrieval robustness to the
three conditions stated in the introduction. In addi-
tion, some recent works tried to make RAG robust
to intentional knowledge corruption attacks, e.g.,
injecting malicious facts (Zou et al., 2024; Anony-
mous, 2024), which is not the type of robustness
we would like to evaluate in this paper.
3 Robustness Metrics
In this section, we present the three critical metrics
for evaluating the retrieval robustness of an LLM
system, illustrated in Figure 2. We define an LLM
system as a backbone LLM, paired with a prompt-
ing strategy. Let f(q, k, o )denote the performance
of an LLM system, where qis the task query, kis
the number of retrieved documents, and ospeci-
fies the order of the retrieved documents. In this
paper, f(q, k, o )is the correctness of the model’s
response to q, assessed by an LLM judge by com-
paring with the reference answer (§4.1). When
k >0,f(q, k, o )represents the performance of the
LLM system in the RAG setup. For consistency,
we use f(q,0)to denote the performance of the
LLM system in the non-RAG setup, where model
answers the query using its own knowledge. See
§4.3 for the choices of kandoin our experiments.
No-Degradation Rate (NDR). This metric mea-
sures how often the LLM system’s performance
with RAG f(q, k, o )(for any k >0ando) is at
least as good as the performance without RAG
f(q,0), which is calculated as:
NDR =1
ZX
q∈QX
k∈KX
o∈O1
f(q, k, o )≥f(q,0)
(1)
where Kincludes all choices of numbers of re-
trieved documents, Orepresents all possible docu-
ment orders used in the benchmark, and Qis the set
of all task samples. Z=|Q| · |K| · |O|is the nor-
malization factor for the aggregation. A high NDR
implies that, for most queries, using retrieval does
not degrade performance relative to the non-RAG
baseline.
Retrieval Size Robustness (RSR). This metric
examines how the system behaves as the num-
ber of retrieved documents increases. Specifically,
for each task query qand each value of k, we

Retrieval Size RobustnessNo-Degradation Rate
Retrieval Order Robustness
Figure 2: Our retrieval robustness metrics, targeting
three research questions: (1) whether RAG is always
better than non-RAG; (2) whether more retrieved doc-
uments always lead to better performance; (3) whether
different document orders lead to consistent results.
check whether the performance is maintained or im-
proved, compared to all smaller values of k. RSR
only considers k >0, not involving the effect of
NDR. Results for various ks are then aggregated
across all task samples, formally defined as:
RSR (q,k i,o)= 1
∧j<i[f(q, ki, o)≥f(q, kj, o)]
RSR=1
ZX
q∈QX
ki∈K,i> 1X
o∈ORSR (q,k i,o)
(2)
where Z=|Q| ·(|K| −1)· |O|. A high RSR
indicates that performance rarely degrades when
adding more retrieved documents.
Retrieval Order Robustness (ROR). ROR con-
cerns the sensitivity of the system to the order of
the same set of retrieved documents. For a task
sample qandk >0, letOdenote selected choices
of permutations of the kdocuments. We can com-
pute the standard deviation of the model perfor-
mance over all permutations o∈O, which is rep-
resented as σo∈O[f(q, k, o )]. For performance met-
rics bounded between 0 and 1, the standard devia-
tion is bounded between 0 and 0.5. Therefore, we
scale it by a factor of 2 to ensure the robustness
metric ranges between 0 and 1. We compute theROR score as:
ROR =1
ZX
q∈QX
k∈K 
1−2σo∈O
f(q, k, o )
(3)
where Z=|Q| · |K|. A higher ROR means that
different permutations of the same set of documents
produce more consistent performance.
The three metrics capture complementary as-
pects of retrieval robustness, reflecting different de-
sired behaviors of LLM systems with RAG in real
world applications. NDR provides a safety guaran-
tee that retrieval will not harm performance; RSR
is critical for scenarios where retrieval size can be
scaled up for enhanced performance; and ROR is
important for situations where document ranking
is imperfect. Note that, for simplicity, we omit the
marginalization over two different retrievers (see
§4.3) from the equations of all three metrics.
4 Benchmark Setups
We conduct experiments to benchmark retrieval
robustness of LLM systems. Though RAG can
be applied for various tasks, we focus on the
task where RAG is commonly adopted—answering
knowledge-intensive open-domain questions.
4.1 Data and Evaluation Metrics
Open-domain QA Tasks. We sample from three
QA datasets. Natural Questions (Kwiatkowski
et al., 2019) contains samples derived from Google
Search queries, covering a broad range of ques-
tions real-world users ask online; Hotpot QA (Yang
et al., 2018) is a multi-hop QA dataset that requires
chaining multiple passages to answer questions;
ASQA (Stelmakh et al., 2022) targets extraction
of key information from multiple sources. We
randomly sample 500 examples from each of the
datasets, totaling 1500 samples.
Evaluation Metrics. Previous work usually
used string match metrics for answers evalua-
tion (Mallen et al., 2023; Gao et al., 2023a). How-
ever, it is rigid and can not evaluate model per-
formance very well. Therefore, we prompt (see
the prompts we used in Appendix C) Llama-3.3-
70B-Instruct to evaluate whether the generated re-
sponses align with the gold answers.2
2We also tried GPT-4o as the judge initially. However
due to cost constraints for large-scale evaluation, we opt for
Llama-3.3-70B-Instruct. And on a subset of 2,000 samples,
we find these two models agree at 93% of time.

Retrieval Corpus. We use Wikipedia as the cor-
pus to retrieve documents from. We processed the
wikidump from June 2024, which contains 6 mil-
lion articles. We split each article into chunks by
double newlines, resulting in 20 million chunks.
Each chunk is treated as an independent “docu-
ment” for retrieval.
4.2 LLM Systems
Backbone LLMs. 11 LLMs from three open-
source families and two proprietary families are
tested, including Llama-3 Instruct (3.1-8B, 3.1-
70B, 3.2-1B, 3.2-3B) (Meta, July 2024,S), Mistral
Instruct (Nemo, Large) (Mistral.ai, July 2024,F),
Command (R, R plus) (Cohere, Aug. 2024), Ope-
nAI GPT-4o (OpenAI et al., 2024), o3-mini (Ope-
nAI, 2025), and Claude-3.5-sonnet (Anthropic,
July. 2024).
Prompting Strategies. Besides the vanilla
prompting strategy that concatenates all retrieved
documents in the prompt, we explore two alterna-
tive strategies that might help model incorporate
information in the retrieved documents more ro-
bustly. Both strategies involve two steps. (1) Own-
Know obtains a draft answer based on models’ own
knowledge by prompting without retrieval in the
first step, and then inserts this draft answer into
the prompt for the RAG setup. (2) S2A, inspired
by System 2 Attention (Weston and Sukhbaatar,
2023), first tries to identify the relevant retrieved
documents in the first step, and then only uses the
identified documents in the RAG setup. This de-
couples relevance estimation from answer extrac-
tion, allowing the answer extraction step to focus
on the most pertinent information. See the Jinja2
templates of our prompts in Appendix C.
4.3 RAG Parameters
Retrievers. Our retrieval system is built on top of
Solr 93. We use two retrievers: one is the canonical
sparse retriever based on BM25 (Robertson and
Zaragoza, 2009), and the other is cosine similarity
based dense retriever where we embeded each doc-
ument by bge-large-en-v1.54(Xiao et al., 2023).
For any robustness metric defined in §3, we get the
results for both retrievers and take the average.
Sizes. We experiment with retrieval sizes of 5, 10,
25, 50, 75, and 100 documents. The retrieval size
3https://solr.apache.org/docs/9_0_0/index.html
4http://huggingface.co/BAAI/bge-large-en-v1.5
0 20 40 60 80 100
Number of Retrieved Documents0.20.30.40.50.60.70.8Recall of Gold Answer
Natural Questions
Hotpot QA
ASQABM25
Dense
Non-RAG PerfFigure 3: Performance of the retrievers, measured by the
recall of gold answers within the concatenated retrieved
documents. The gold answer is considered covered if
any of its alternative forms exactly match a substring in
the concatenated retrieved documents.
is capped at 100 documents as most models have
reached their maximum context lengths. When the
retrieved documents exceed the maximum context
length of a model, we iteratively drop the lowest
ranked document.
Orders. For each of these sizes, we apply three
ordering strategies based on the retriever’s ranking
of the documents: the original order (returned by
the retriever), the reversed order (reversing the
original order), and a randomly shuffled order. We
test the reversed order because sometimes we want
to put the most relevant document to the end of the
prompt (the closest to the answer). We include a
random order to simulate any potential reranking
logic on top of the retriever.
Retrieval Quality. As our retrieval robustness
benchmark relies on the retrievers, we examine
the retrieval quality by checking the recall of gold
answers within the retrieved documents. We fol-
low prior work and determine if the concatenated
retrieved documents contain the gold answer if
its substring is an exact match of any form of
the gold answer (substring exact match) (Mallen
et al., 2023). For reference, we also report the best
model performance without RAG (Non-RAG Perf)
to highlight the potential improvement that can be
obtained with RAG. As shown in Figure 3, both
retrievers provide sufficiently high-quality retrieval,
ensuring that the findings of our experiments are
based on valid setups.

1B3B8B70B32B104B12B123B4oo3msonn0.00.20.40.60.8
No-Degradation Rate
1B3B8B70B32B104B12B123B4oo3msonn
Retrieval Size Robustness
1B3B8B70B32B104B12B123B4oo3msonn
Retrieval Order Robustness
Performance Robustness Metric
 Llama
 Command
 Mistral
 OpenAI
 ClaudeFigure 4: The three retrieval robustness metrics and task performance of experimented LLMs using vanilla prompting.
Model families are indicated by icons, while the variants are indicated by model sizes or names (o3m: o3-mini;
sonn: sonnet). 12B and 123B Mistral models respectively correspond to Mistral-Nemo and Mistral-Large. Task
performance is the averaged QA accuracy across different retrieval sizes and orders. Models generally demonstrate
strong retrieval robustness (achieving 80% scores). While larger model sizes lead to improved task performance,
there exists no consistent trend across the retrieval robustness metrics.
5 Results
5.1 Overall Robustness
We report the three retrieval robustness metrics
for LLM systems using vanilla prompting (see the
prompt rag_qa.j2 in Appendix C) in Figure 4.
Besides robustness, task performance is shown in
the same figure with bars with a different hatch
style. Retrieval robustness is calculated following
the definitions in §3, while task performance is
the average score across all k,o, retrievers, and
datasets. All models achieve higher than 80% re-
trieval robustness across all metrics, with GPT-
4o and o3-mini surpassing 90%. Compared to
prior studies that highlight the weak robustness of
RAG systems under synthetic setups, such as using
artificially created documents (Wu et al., 2024b),
we show that LLMs demonstrate surprisingly good
retrieval robustness in more realistic settings. This
high retrieval robustness means we can safely apply
RAG without overly stressing about whether RAG
is better than non-RAG and about the decisions on
retrieval size and order, which can potential sim-
plify RAG systems. Nevertheless, the remaining
10% may pose challenges for real-world deploy-
ment, particular for high-stake domains where com-
prehensive reliability is required.
5.2 Relation between Robustness and
Performance
Although retrieval robustness metrics are derived
from the sample-level task performance, retrievalrobustness does not always correlate with task per-
formance. As shown in Figure 1 and Figure 4,
task performance usually gets better when models
get larger. In contrast, we note that, larger LLMs
can have lower retrieval robustness than smaller
LLMs . For example, in Figure 1, Llama-3-8B has
higher robustness than 70B. If we “zoom in” to
each of the three robustness metrics (Figure 4), we
can see that this inverse scaling trend mainly comes
from No-Degradation Rate (NDR). This is because
larger models usually have richer parametric knowl-
edge and answers more questions correctly without
retrieval, which means RAG will have a higher
baseline to beat and thus RAG is more likely to get
worse than non-RAG. Therefore, in practice, when
we apply RAG to knowledge-rich LLMs (usually
models of larger sizes), we need to be cautious
about whether it will lead to performance degrada-
tion compared to non-RAG.
Here, we use one example to show how low ro-
bustness reduces RAG efficacy . In Figure 5, solid
lines illustrate the actual performance of Mistral-
Large and o3-mini at different number of retrieved
documents. Dashed lines show their hypothetical
performance under an oracle setup. This oracle
setup assumes perfect NDR , meaning the models
consistently generate responses at least as good as
those produced without retrieval. As the solid lines
show, although Mistral-Large surpasses o3-mini
without retrieval (0 retrieved documents), it yields
worse performance than o3-mini and even its own
non-RAG baseline when RAG is applied. Con-

0510 25 50 75 100
Retrieved Documents0.60.70.8Task Performance
Mistral Large
o3-mini
Actual NDR
Perfect NDRMistral Large
o3-mini
Actual NDR
Perfect NDRFigure 5: Task performance of models using vanilla
prompting under setups with actual no-degradation rate
(NDR) and perfect NDR. Enhancing retrieval robustness
could lead to a 12% absolute performance gain for both
models.
versely, if Mistral-Large has perfect NDR, it would
outperform o3-mini in the RAG setup. The gap
between the actual and oracle setups demonstrate
that Mistral-Large fails to preserve its non-RAG
performance for approximately 14% of the dataset
samples, due to the insufficient retrieval robustness.
Overall, retrieval robustness metrics complement
standard task performance metrics and provide a
new perspective of how well LLMs perform in
RAG settings.
5.3 Effect of Retrieval Size
For most of the models, the overall task perfor-
mance is generally increasing as more retrieved
documents are added (see Figure 13, 14, 15,
and 16 in Appendix). This again demonstrates that
in practice we do not have to overly concern about
picking the optimal retrieval size. If budget allows,
we can simply keep adding more documents till it
reaches the max input length limit.
Nevertheless, this does not indicate perfect re-
trieval size robustness, as models keep trading
off performance across individual samples , i.e.,
hurting performance on some examples while gain-
ing performance on others. Similar to the perfect
NDR setup, we investigate an oracle setup with per-
fect RSR—choosing the best answer among those
generated at current and all preceding values of
ks as the final answer. Note that only answers
produced by RAG (i.e., k >0) are considered in
the perfect RSR setup to eliminate the effect of
NDR. Although, in the normal setup (actual RSR),
task performance is increasing from k= 10 to
k= 75 , the gain is much more significant in the
10 25 50 75 100
Retrieved Documents0.60.70.8Task Performance
Mistral Large
GPT-4o
Actual RSR
Perfect RSRMistral Large
GPT-4o
Actual RSR
Perfect RSRFigure 6: Task performance of models using vanilla
prompting under setups with actual RSR and perfect
RSR.
0.4 0.5 0.6 0.7
Task Performance0.800.850.900.95Robustness
Llama 3.2 3B
Llama 3.1 70B
Command R
Command R+Mistral Nemo
Mistral Large
GPT-4o
o3-miniClaude 3.5 Sonnet
Original
Reversed
Shuffled
Figure 7: Geometric mean of no-degradation rate and re-
trieval size robustness, grouped by the order of retrieved
documents.
hypothetical perfect RSR situation, enlarging the
gap between the two setups. This implies that mod-
els are constantly sacrificing some samples while
enhancing others with larger retrieval sizes. We
think that the increasing number of retrieved docu-
ments challenges models’ ability to identify helpful
documents and handle longer inputs, and thus leads
to the imperfect robustness on retrieval size.
5.4 Effect of Retrieval Order
We break down retrieval robustness and task per-
formance by the order of the retrieved documents
(Figure 7). Overall, LLMs demonstrate good
retrieval order robustness – the performance
achieved with different orders of the retrieved
documents is similar . This means, in practice, we
do not have to overly concern about the order of
documents. While GPT-4o and o3-mini demon-
strate the strongest retrieval robustness and perfor-

510 25 50 75 100
Retrieved Documents0.60.70.8Task Performance
Mistral Large
GPT-4o
OriginalReversed
Shuffled
Perfect RORMistral Large
GPT-4o
OriginalReversed
Shuffled
Perfect RORFigure 8: Task performance of models using vanilla
prompting under setups with actual ROR for each order
and perfect ROR.
mance with normally ordered documents, all other
models prefer the reversed order. This suggests
thatplacing higher-ranked retrieved documents
closer to the question generally optimizes RAG
performance.
Despite this high robustness, we underscore
that performance variance across orders per-
sists at the sample level . We establish an oracle
setup for retrieval order robustness that selects the
best response among responses generated with re-
trieved contexts ordered differently ( perfect ROR ),
as shown in Figure 8. Picking the best response
for each example across different orders exhibits a
large performance gain from each individual docu-
ment order. This indicates that each example has
a different best order, highlighting the need for
continuing efforts to improve order robustness.
5.5 Effects of Prompting Strategies
Using prompting strategies to decompose response
generation has demonstrated effectiveness in han-
dling complex tasks. Figure 9 shows that only the
OwnKnow strategy (see the prompt ownknow.j2
in Appendix C) that incorporates answers gener-
ated in the non-RAG setup can consistently en-
hance retrieval robustness. We believe outputs
given by the non-RAG setup serve as drafts and
anchors, leading to reduced variance. It is also
possible that OwnKnow benefits from its simi-
larity to self-refinement that was shown to be an
effective prompting technique (Yang et al., 2022;
Madaan et al., 2023). Although selecting task-
relevant context benefits robustness when synthetic
noisy passages are injected into the input as shown
by Weston and Sukhbaatar (2023), a similar S2A
0.4 0.5 0.6 0.7
Task Performance0.800.850.900.95RobustnessVanilla vs. OwnKnow
0.4 0.5 0.6 0.7
Task Performance0.800.850.900.95RobustnessVanilla vs. S2A
Llama 3.2 3B
Llama 3.1 70B
Command R
Command R+Mistral Nemo
Mistral Large
GPT-4o
o3-miniClaude 3.5 Sonnet
Vanilla
OwnKnow
S2AFigure 9: Geometry mean of the three retrieval robust-
ness metrics and task performance of LLMs paired with
different prompting strategies. The mean of task perfor-
mance achieved with different retrieval sizes and orders
are shown for each model. Models are differentiated
with colors and prompting strategies are indicated by
marker styles. The bar on the right of each marker indi-
cates the maximum performance across retrieval sizes.
prompting strategy (see the prompt s2a.j2 in Ap-
pendix C) fails to enhance retrieval robustness in
our evaluations. We conjecture that, compared to
synthetic noisy contexts, realistic retrievers provide
models with harder negative contexts that are more
challenging for the model to identify.
As we look into the maximum task performance
across retrieval sizes rather than the mean task per-
formance, we observe that using OwnKnow might
limit the maximum performance models can pos-
sibly achieve, suggesting that the higher retrieval
robustness of OwnKnow comes at a cost of RAG
effectiveness.
6 Conclusions
We introduce retrieval robustness metrics—no-
degradation rate, retrieval size robustness, and re-
trieval order robustness—to quantify how reliably

LLMs handle queries via RAG. A realistic bench-
mark of 1,500 questions is compiled, spanning
three open-domain QA datasets, with augmented
documents retrieved from Wikipedia using both
sparse and dense retrievers. Our experiments with
11 LLMs from 5 families reveal that models gener-
ally demonstrate strong robustness, achieving 80%
scores on those metrics. Nonetheless, imperfect
robustness result in sample-level trade-offs, often
hurting the performance of some samples for the
improvement on others, which forfeits RAG’s po-
tential gains. While incorporating outputs gener-
ated with the model’s own knowledge can enhance
retrieval robustness, it also limits the best perfor-
mance that can be achieved by RAG. We believe
retrieval robustness provides a new perspective for
evaluating and understanding LLMs’ RAG perfor-
mance and we hope it can guide and inspire further
research on building robust RAG systems.
7 Limitations
Our study of retrieval robustness focuses on open-
domain QA, though we recognize that RAG can
also be applied to other tasks, such as fact check-
ing and code completion. We choose open-domain
QA, as it is arguably the most common use case of
RAG and is being used in prior work on retrieval
robustness with synthetic setups (Wu et al., 2024b;
Chen et al., 2024). That being said, our proposed
retrieval robustness metrics are specifically formu-
lated such that they can be used for any task, as
long as its evaluation metric returns a scalar value.
8 Ethical Considerations
This benchmark comprises of multiple public mod-
els and datasets. We performed an internal legal
review for each model and dataset to ensure that
they contained permissive licenses to be used for
research purposes. We also do not pretrain or fine-
tune any language models as part of this research
and hence not anticipate the environmental impact
to be significant.
Additionally, before ingesting the Wikipedia
data for retrieval, we ensured that all Personally
Identifiable Information was removed from the
dataset (By removing sections listed as “Personal
Information”). However, we acknowledge that the
models and datasets could still contain biases (such
as race, gender, etc.) that could be reflected in the
generated answers.Acknowledgements
We thank Bang An and Mark Dredze for their help-
ful discussions.
References
Anonymous. 2024. Certifiably robust RAG against re-
trieval corruption attacks. In Submitted to The Thir-
teenth International Conference on Learning Repre-
sentations . Under review.
Anthropic. July. 2024. claude-3-5-sonnet.
Akari Asai, Zexuan Zhong, Danqi Chen, Pang Wei
Koh, Luke Zettlemoyer, Hannaneh Hajishirzi, and
Wen-tau Yih. 2024. Reliable, adaptable, and at-
tributable language models with retrieval. arXiv
preprint arXiv:2403.03187 .
Sebastian Borgeaud, Arthur Mensch, Jordan Hoff-
mann, Trevor Cai, Eliza Rutherford, Katie Milli-
can, George Bm Van Den Driessche, Jean-Baptiste
Lespiau, Bogdan Damoc, Aidan Clark, Diego
De Las Casas, Aurelia Guy, Jacob Menick, Roman
Ring, Tom Hennigan, Saffron Huang, Loren Mag-
giore, Chris Jones, Albin Cassirer, Andy Brock,
Michela Paganini, Geoffrey Irving, Oriol Vinyals,
Simon Osindero, Karen Simonyan, Jack Rae, Erich
Elsen, and Laurent Sifre. 2022. Improving language
models by retrieving from trillions of tokens. In
Proceedings of the 39th International Conference
on Machine Learning , volume 162 of Proceedings
of Machine Learning Research , pages 2206–2240.
PMLR.
Danqi Chen, Adam Fisch, Jason Weston, and Antoine
Bordes. 2017. Reading Wikipedia to answer open-
domain questions. In Proceedings of the 55th Annual
Meeting of the Association for Computational Lin-
guistics (Volume 1: Long Papers) , pages 1870–1879,
Vancouver, Canada. Association for Computational
Linguistics.
Jiawei Chen, Hongyu Lin, Xianpei Han, and Le Sun.
2024. Benchmarking large language models in
retrieval-augmented generation. In Proceedings of
the AAAI Conference on Artificial Intelligence , vol-
ume 38, pages 17754–17762.
Cohere. Aug. 2024. command-r.
Tianyu Gao, Howard Yen, Jiatong Yu, and Danqi Chen.
2023a. Enabling large language models to generate
text with citations. In Proceedings of the 2023 Con-
ference on Empirical Methods in Natural Language
Processing , pages 6465–6488, Singapore. Associa-
tion for Computational Linguistics.
Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang Jia,
Jinliu Pan, Yuxi Bi, Yi Dai, Jiawei Sun, and Haofen
Wang. 2023b. Retrieval-augmented generation for
large language models: A survey. arXiv preprint
arXiv:2312.10997 .

Zhicheng Guo, Sijie Cheng, Yile Wang, Peng Li, and
Yang Liu. 2023. Prompt-guided retrieval augmen-
tation for non-knowledge-intensive tasks. In Find-
ings of the Association for Computational Linguistics:
ACL 2023 , pages 10896–10912, Toronto, Canada. As-
sociation for Computational Linguistics.
Kelvin Guu, Kenton Lee, Zora Tung, Panupong Pasu-
pat, and Ming-Wei Chang. 2020. Realm: retrieval-
augmented language model pre-training. In Proceed-
ings of the 37th International Conference on Machine
Learning , ICML’20. JMLR.org.
Gautier Izacard and Edouard Grave. 2021. Leveraging
passage retrieval with generative models for open do-
main question answering. In Proceedings of the 16th
Conference of the European Chapter of the Associ-
ation for Computational Linguistics: Main Volume ,
pages 874–880, Online. Association for Computa-
tional Linguistics.
Gautier Izacard, Patrick Lewis, Maria Lomeli, Lucas
Hosseini, Fabio Petroni, Timo Schick, Jane Dwivedi-
Yu, Armand Joulin, Sebastian Riedel, and Edouard
Grave. 2024. Atlas: few-shot learning with retrieval
augmented language models. J. Mach. Learn. Res. ,
24(1).
Robin Jia and Percy Liang. 2017. Adversarial exam-
ples for evaluating reading comprehension systems.
InProceedings of the 2017 Conference on Empiri-
cal Methods in Natural Language Processing , pages
2021–2031, Copenhagen, Denmark. Association for
Computational Linguistics.
Urvashi Khandelwal, Omer Levy, Dan Jurafsky, Luke
Zettlemoyer, and Mike Lewis. 2020. Generalization
through memorization: Nearest neighbor language
models. In International Conference on Learning
Representations .
Tom Kwiatkowski, Jennimaria Palomaki, Olivia Red-
field, Michael Collins, Ankur Parikh, Chris Alberti,
Danielle Epstein, Illia Polosukhin, Jacob Devlin, Ken-
ton Lee, Kristina Toutanova, Llion Jones, Matthew
Kelcey, Ming-Wei Chang, Andrew M. Dai, Jakob
Uszkoreit, Quoc Le, and Slav Petrov. 2019. Natu-
ral questions: A benchmark for question answering
research. Transactions of the Association for Compu-
tational Linguistics , 7:452–466.
Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying
Sheng, Lianmin Zheng, Cody Hao Yu, Joseph E.
Gonzalez, Hao Zhang, and Ion Stoica. 2023. Effi-
cient memory management for large language model
serving with pagedattention. In Proceedings of the
ACM SIGOPS 29th Symposium on Operating Systems
Principles .
Jinhyuk Lee, Anthony Chen, Zhuyun Dai, Dheeru Dua,
Devendra Singh Sachan, Michael Boratko, Yi Luan,
Sébastien MR Arnold, Vincent Perot, Siddharth
Dalmia, et al. 2024. Can long-context language mod-
els subsume retrieval, rag, sql, and more? arXiv
preprint arXiv:2406.13121 .Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio
Petroni, Vladimir Karpukhin, Naman Goyal, Hein-
rich Küttler, Mike Lewis, Wen-tau Yih, Tim Rock-
täschel, Sebastian Riedel, and Douwe Kiela. 2020.
Retrieval-augmented generation for knowledge-
intensive nlp tasks. In Proceedings of the 34th Inter-
national Conference on Neural Information Process-
ing Systems , NIPS ’20, Red Hook, NY , USA. Curran
Associates Inc.
Xi Victoria Lin, Xilun Chen, Mingda Chen, Weijia
Shi, Maria Lomeli, Richard James, Pedro Rodriguez,
Jacob Kahn, Gergely Szilvasy, Mike Lewis, Luke
Zettlemoyer, and Wen tau Yih. 2024. RA-DIT:
Retrieval-augmented dual instruction tuning. In The
Twelfth International Conference on Learning Repre-
sentations .
Aman Madaan, Niket Tandon, Prakhar Gupta, Skyler
Hallinan, Luyu Gao, Sarah Wiegreffe, Uri Alon,
Nouha Dziri, Shrimai Prabhumoye, Yiming Yang,
Shashank Gupta, Bodhisattwa Prasad Majumder,
Katherine Hermann, Sean Welleck, Amir Yazdan-
bakhsh, and Peter Clark. 2023. Self-refine: It-
erative refinement with self-feedback. Preprint ,
arXiv:2303.17651.
Alex Mallen, Akari Asai, Victor Zhong, Rajarshi Das,
Daniel Khashabi, and Hannaneh Hajishirzi. 2023.
When not to trust language models: Investigating
effectiveness of parametric and non-parametric mem-
ories. In Proceedings of the 61st Annual Meeting of
the Association for Computational Linguistics (Vol-
ume 1: Long Papers) , pages 9802–9822, Toronto,
Canada. Association for Computational Linguistics.
Meta. July 2024. Introducing llama 3.1.
Meta. September 2024. Llama 3.2.
Mistral.ai. Feb. 2024. mistral-large.
Mistral.ai. July 2024. mistral-nemo.
OpenAI, :, Aaron Hurst, Adam Lerer, Adam P. Goucher,
Adam Perelman, Aditya Ramesh, Aidan Clark,
AJ Ostrow, Akila Welihinda, Alan Hayes, Alec
Radford, Aleksander M ˛ adry, Alex Baker-Whitcomb,
Alex Beutel, Alex Borzunov, Alex Carney, Alex
Chow, Alex Kirillov, Alex Nichol, Alex Paino, Alex
Renzin, Alex Tachard Passos, Alexander Kirillov,
Alexi Christakis, Alexis Conneau, Ali Kamali, Allan
Jabri, Allison Moyer, Allison Tam, Amadou Crookes,
Amin Tootoochian, Amin Tootoonchian, Ananya
Kumar, Andrea Vallone, Andrej Karpathy, Andrew
Braunstein, Andrew Cann, Andrew Codispoti, An-
drew Galu, Andrew Kondrich, Andrew Tulloch, An-
drey Mishchenko, Angela Baek, Angela Jiang, An-
toine Pelisse, Antonia Woodford, Anuj Gosalia, Arka
Dhar, Ashley Pantuliano, Avi Nayak, Avital Oliver,
Barret Zoph, Behrooz Ghorbani, Ben Leimberger,
Ben Rossen, Ben Sokolowsky, Ben Wang, Benjamin
Zweig, Beth Hoover, Blake Samic, Bob McGrew,
Bobby Spero, Bogo Giertler, Bowen Cheng, Brad
Lightcap, Brandon Walkin, Brendan Quinn, Brian

Guarraci, Brian Hsu, Bright Kellogg, Brydon East-
man, Camillo Lugaresi, Carroll Wainwright, Cary
Bassin, Cary Hudson, Casey Chu, Chad Nelson,
Chak Li, Chan Jun Shern, Channing Conger, Char-
lotte Barette, Chelsea V oss, Chen Ding, Cheng Lu,
Chong Zhang, Chris Beaumont, Chris Hallacy, Chris
Koch, Christian Gibson, Christina Kim, Christine
Choi, Christine McLeavey, Christopher Hesse, Clau-
dia Fischer, Clemens Winter, Coley Czarnecki, Colin
Jarvis, Colin Wei, Constantin Koumouzelis, Dane
Sherburn, Daniel Kappler, Daniel Levin, Daniel Levy,
David Carr, David Farhi, David Mely, David Robin-
son, David Sasaki, Denny Jin, Dev Valladares, Dim-
itris Tsipras, Doug Li, Duc Phong Nguyen, Duncan
Findlay, Edede Oiwoh, Edmund Wong, Ehsan As-
dar, Elizabeth Proehl, Elizabeth Yang, Eric Antonow,
Eric Kramer, Eric Peterson, Eric Sigler, Eric Wal-
lace, Eugene Brevdo, Evan Mays, Farzad Khorasani,
Felipe Petroski Such, Filippo Raso, Francis Zhang,
Fred von Lohmann, Freddie Sulit, Gabriel Goh,
Gene Oden, Geoff Salmon, Giulio Starace, Greg
Brockman, Hadi Salman, Haiming Bao, Haitang
Hu, Hannah Wong, Haoyu Wang, Heather Schmidt,
Heather Whitney, Heewoo Jun, Hendrik Kirchner,
Henrique Ponde de Oliveira Pinto, Hongyu Ren,
Huiwen Chang, Hyung Won Chung, Ian Kivlichan,
Ian O’Connell, Ian O’Connell, Ian Osband, Ian Sil-
ber, Ian Sohl, Ibrahim Okuyucu, Ikai Lan, Ilya
Kostrikov, Ilya Sutskever, Ingmar Kanitscheider,
Ishaan Gulrajani, Jacob Coxon, Jacob Menick, Jakub
Pachocki, James Aung, James Betker, James Crooks,
James Lennon, Jamie Kiros, Jan Leike, Jane Park,
Jason Kwon, Jason Phang, Jason Teplitz, Jason
Wei, Jason Wolfe, Jay Chen, Jeff Harris, Jenia Var-
avva, Jessica Gan Lee, Jessica Shieh, Ji Lin, Jiahui
Yu, Jiayi Weng, Jie Tang, Jieqi Yu, Joanne Jang,
Joaquin Quinonero Candela, Joe Beutler, Joe Lan-
ders, Joel Parish, Johannes Heidecke, John Schul-
man, Jonathan Lachman, Jonathan McKay, Jonathan
Uesato, Jonathan Ward, Jong Wook Kim, Joost
Huizinga, Jordan Sitkin, Jos Kraaijeveld, Josh Gross,
Josh Kaplan, Josh Snyder, Joshua Achiam, Joy Jiao,
Joyce Lee, Juntang Zhuang, Justyn Harriman, Kai
Fricke, Kai Hayashi, Karan Singhal, Katy Shi, Kavin
Karthik, Kayla Wood, Kendra Rimbach, Kenny Hsu,
Kenny Nguyen, Keren Gu-Lemberg, Kevin Button,
Kevin Liu, Kiel Howe, Krithika Muthukumar, Kyle
Luther, Lama Ahmad, Larry Kai, Lauren Itow, Lau-
ren Workman, Leher Pathak, Leo Chen, Li Jing, Lia
Guy, Liam Fedus, Liang Zhou, Lien Mamitsuka, Lil-
ian Weng, Lindsay McCallum, Lindsey Held, Long
Ouyang, Louis Feuvrier, Lu Zhang, Lukas Kon-
draciuk, Lukasz Kaiser, Luke Hewitt, Luke Metz,
Lyric Doshi, Mada Aflak, Maddie Simens, Madelaine
Boyd, Madeleine Thompson, Marat Dukhan, Mark
Chen, Mark Gray, Mark Hudnall, Marvin Zhang,
Marwan Aljubeh, Mateusz Litwin, Matthew Zeng,
Max Johnson, Maya Shetty, Mayank Gupta, Meghan
Shah, Mehmet Yatbaz, Meng Jia Yang, Mengchao
Zhong, Mia Glaese, Mianna Chen, Michael Jan-
ner, Michael Lampe, Michael Petrov, Michael Wu,
Michele Wang, Michelle Fradin, Michelle Pokrass,
Miguel Castro, Miguel Oom Temudo de Castro,
Mikhail Pavlov, Miles Brundage, Miles Wang, Mi-nal Khan, Mira Murati, Mo Bavarian, Molly Lin,
Murat Yesildal, Nacho Soto, Natalia Gimelshein, Na-
talie Cone, Natalie Staudacher, Natalie Summers,
Natan LaFontaine, Neil Chowdhury, Nick Ryder,
Nick Stathas, Nick Turley, Nik Tezak, Niko Felix,
Nithanth Kudige, Nitish Keskar, Noah Deutsch, Noel
Bundick, Nora Puckett, Ofir Nachum, Ola Okelola,
Oleg Boiko, Oleg Murk, Oliver Jaffe, Olivia Watkins,
Olivier Godement, Owen Campbell-Moore, Patrick
Chao, Paul McMillan, Pavel Belov, Peng Su, Pe-
ter Bak, Peter Bakkum, Peter Deng, Peter Dolan,
Peter Hoeschele, Peter Welinder, Phil Tillet, Philip
Pronin, Philippe Tillet, Prafulla Dhariwal, Qiming
Yuan, Rachel Dias, Rachel Lim, Rahul Arora, Ra-
jan Troll, Randall Lin, Rapha Gontijo Lopes, Raul
Puri, Reah Miyara, Reimar Leike, Renaud Gaubert,
Reza Zamani, Ricky Wang, Rob Donnelly, Rob
Honsby, Rocky Smith, Rohan Sahai, Rohit Ramchan-
dani, Romain Huet, Rory Carmichael, Rowan Zellers,
Roy Chen, Ruby Chen, Ruslan Nigmatullin, Ryan
Cheu, Saachi Jain, Sam Altman, Sam Schoenholz,
Sam Toizer, Samuel Miserendino, Sandhini Agar-
wal, Sara Culver, Scott Ethersmith, Scott Gray, Sean
Grove, Sean Metzger, Shamez Hermani, Shantanu
Jain, Shengjia Zhao, Sherwin Wu, Shino Jomoto, Shi-
rong Wu, Shuaiqi, Xia, Sonia Phene, Spencer Papay,
Srinivas Narayanan, Steve Coffey, Steve Lee, Stew-
art Hall, Suchir Balaji, Tal Broda, Tal Stramer, Tao
Xu, Tarun Gogineni, Taya Christianson, Ted Sanders,
Tejal Patwardhan, Thomas Cunninghman, Thomas
Degry, Thomas Dimson, Thomas Raoux, Thomas
Shadwell, Tianhao Zheng, Todd Underwood, Todor
Markov, Toki Sherbakov, Tom Rubin, Tom Stasi,
Tomer Kaftan, Tristan Heywood, Troy Peterson, Tyce
Walters, Tyna Eloundou, Valerie Qi, Veit Moeller,
Vinnie Monaco, Vishal Kuo, Vlad Fomenko, Wayne
Chang, Weiyi Zheng, Wenda Zhou, Wesam Manassra,
Will Sheu, Wojciech Zaremba, Yash Patil, Yilei Qian,
Yongjik Kim, Youlong Cheng, Yu Zhang, Yuchen
He, Yuchen Zhang, Yujia Jin, Yunxing Dai, and
Yury Malkov. 2024. Gpt-4o system card. Preprint ,
arXiv:2410.21276.
OpenAI. 2025. OpenAI o3-mini — openai.com. https:
//openai.com/index/openai-o3-mini/ .
Ori Ram, Yoav Levine, Itay Dalmedigos, Dor Muhlgay,
Amnon Shashua, Kevin Leyton-Brown, and Yoav
Shoham. 2023. In-context retrieval-augmented lan-
guage models. Transactions of the Association for
Computational Linguistics , 11:1316–1331.
Stephen Robertson and Hugo Zaragoza. 2009. The
probabilistic relevance framework: Bm25 and be-
yond. Found. Trends Inf. Retr. , 3(4):333–389.
Freda Shi, Xinyun Chen, Kanishka Misra, Nathan
Scales, David Dohan, Ed Chi, Nathanael Schärli, and
Denny Zhou. 2023. Large language models can be
easily distracted by irrelevant context. In Proceed-
ings of the 40th International Conference on Machine
Learning , ICML’23. JMLR.org.
Weijia Shi, Sewon Min, Michihiro Yasunaga, Min-
joon Seo, Richard James, Mike Lewis, Luke Zettle-

moyer, and Wen-tau Yih. 2024. REPLUG: Retrieval-
augmented black-box language models. In Proceed-
ings of the 2024 Conference of the North American
Chapter of the Association for Computational Lin-
guistics: Human Language Technologies (Volume
1: Long Papers) , pages 8371–8384, Mexico City,
Mexico. Association for Computational Linguistics.
Ivan Stelmakh, Yi Luan, Bhuwan Dhingra, and Ming-
Wei Chang. 2022. ASQA: Factoid questions meet
long-form answers. In Proceedings of the 2022 Con-
ference on Empirical Methods in Natural Language
Processing , pages 8273–8288, Abu Dhabi, United
Arab Emirates. Association for Computational Lin-
guistics.
Nandan Thakur, Luiz Bonifacio, Crystina Zhang,
Odunayo Ogundepo, Ehsan Kamalloo, David
Alfonso-Hermelo, Xiaoguang Li, Qun Liu, Box-
ing Chen, Mehdi Rezagholizadeh, and Jimmy Lin.
2024. “knowing when you don’t know”: A multilin-
gual relevance assessment dataset for robust retrieval-
augmented generation. In Findings of the Association
for Computational Linguistics: EMNLP 2024 , pages
12508–12526, Miami, Florida, USA. Association for
Computational Linguistics.
Jason Weston and Sainbayar Sukhbaatar. 2023. Sys-
tem 2 attention (is something you might need too).
Preprint , arXiv:2311.11829.
Shangyu Wu, Ying Xiong, Yufei Cui, Haolun Wu, Can
Chen, Ye Yuan, Lianming Huang, Xue Liu, Tei-Wei
Kuo, Nan Guan, et al. 2024a. Retrieval-augmented
generation for natural language processing: A survey.
arXiv preprint arXiv:2407.13193 .
Siye Wu, Jian Xie, Jiangjie Chen, Tinghui Zhu, Kai
Zhang, and Yanghua Xiao. 2024b. How easily do
irrelevant inputs skew the responses of large language
models? In First Conference on Language Modeling .
Shitao Xiao, Zheng Liu, Peitian Zhang, and Niklas
Muennighoff. 2023. C-pack: Packaged resources
to advance general chinese embedding. Preprint ,
arXiv:2309.07597.
Kevin Yang, Yuandong Tian, Nanyun Peng, and Dan
Klein. 2022. Re3: Generating longer stories with
recursive reprompting and revision. In Proceedings
of the 2022 Conference on Empirical Methods in Nat-
ural Language Processing , pages 4393–4479, Abu
Dhabi, United Arab Emirates. Association for Com-
putational Linguistics.
Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio,
William Cohen, Ruslan Salakhutdinov, and Christo-
pher D. Manning. 2018. HotpotQA: A dataset for
diverse, explainable multi-hop question answering.
InProceedings of the 2018 Conference on Empiri-
cal Methods in Natural Language Processing , pages
2369–2380, Brussels, Belgium. Association for Com-
putational Linguistics.
Ori Yoran, Tomer Wolfson, Ori Ram, and Jonathan Be-
rant. 2024. Making retrieval-augmented languagemodels robust to irrelevant context. In The Twelfth
International Conference on Learning Representa-
tions .
Wei Zou, Runpeng Geng, Binghui Wang, and Jinyuan
Jia. 2024. Poisonedrag: Knowledge corruption at-
tacks to retrieval-augmented generation of large lan-
guage models. Preprint , arXiv:2402.07867.

A Additional Results
A.1 Dataset Breakdown of Retrieval
Robustness
We show the retrieval robustness metrics and av-
erage RAG performance in Figure 10, 11, and 12.
Across all individual datasets, there is still no con-
sistent improvement in retrieval robustness with
increased model sizes.
A.2 Dataset Breakdown of RAG Performance
across ks
We show open-domain QA performance at differ-
ent numbers of retrieved documents in Figure 13,
with dataset breakdown in Figure 14, 15, and 16.
Performance with each retriever and document or-
der can be found in Figure 17, 18, and 19.
Compared to non-RAG, open-source LLMs with
RAG can always boost performance, with the ex-
ception of Command R+ on Natural Questions. We
also observe a performance drop on Hotpot QA
with the dense retriever when using Llama-3.1-
70B.
B Inference Setup
Inference Parameters. Due to the computational
cost and running time, we use greedy decoding
and perform inference with each model under each
setup once. During inference, models are allowed
to generate at most 100 tokens, though they never
exceed the limit.
Inference Infrastructure. We use vLLM for
more efficient inference (Kwon et al., 2023) and
our experiments are conducted on compute nodes
with 8 H100 GPUs.
C Prompt Templates
The prompt templates (in jinja2 format) used in our
experiments can be found at the end of Appendix.

1B 3B 8B70B 32B104B 12B123B 4osonnet0.00.20.40.60.8
No-Degradation Rate
1B 3B 8B70B 32B104B 12B123B 4osonnet
Retrieval Size Robustness
1B 3B 8B70B 32B104B 12B123B 4osonnet
Retrieval Order Robustness
Performance Robustness Metric
 Llama
 Command
 Mistral
 GPT
 ClaudeFigure 10: The three retrieval robustness metrics and task performance of experimented LLMs using vanilla
prompting on Natural Questions.
1B 3B 8B70B 32B104B 12B123B 4osonnet0.00.20.40.60.8
No-Degradation Rate
1B 3B 8B70B 32B104B 12B123B 4osonnet
Retrieval Size Robustness
1B 3B 8B70B 32B104B 12B123B 4osonnet
Retrieval Order Robustness
Performance Robustness Metric
 Llama
 Command
 Mistral
 GPT
 Claude
Figure 11: The three retrieval robustness metrics and task performance of experimented LLMs using vanilla
prompting on Hotpot QA.
1B 3B 8B70B 32B104B 12B123B 4osonnet0.00.20.40.60.8
No-Degradation Rate
1B 3B 8B70B 32B104B 12B123B 4osonnet
Retrieval Size Robustness
1B 3B 8B70B 32B104B 12B123B 4osonnet
Retrieval Order Robustness
Performance Robustness Metric
 Llama
 Command
 Mistral
 GPT
 Claude
Figure 12: The three retrieval robustness metrics and task performance of experimented LLMs using vanilla
prompting on ASQA.

0510 25 50 75 100
Retrieved Documents0.10.20.30.40.50.60.70.80.9Task Performance
Llama 3.2 1B
Llama 3.2 3B
Llama 3.1 8B
Llama 3.1 70B
Command R
Command R+
Mistral Nemo
Mistral Large
GPT-4o
o3-mini
Claude 3.5 SonnetFigure 13: Performance averaged across datasets, re-
trievers, and document orders.
0510 25 50 75 100
Retrieved Documents0.10.20.30.40.50.60.70.80.9Task Performance
Llama 3.2 1B
Llama 3.2 3B
Llama 3.1 8B
Llama 3.1 70B
Command R
Command R+
Mistral Nemo
Mistral Large
GPT-4o
o3-mini
Claude 3.5 Sonnet
Figure 14: Performance on Natural Questions, averaged
across retrievers and document orders.
0510 25 50 75 100
Retrieved Documents0.10.20.30.40.50.60.70.80.9Task Performance
Llama 3.2 1B
Llama 3.2 3B
Llama 3.1 8B
Llama 3.1 70B
Command R
Command R+
Mistral Nemo
Mistral Large
GPT-4o
o3-mini
Claude 3.5 Sonnet
Figure 15: Performance on Hotpot QA, averaged across
retrievers and document orders.
0510 25 50 75 100
Retrieved Documents0.10.20.30.40.50.60.70.80.9Task Performance
Llama 3.2 1B
Llama 3.2 3B
Llama 3.1 8B
Llama 3.1 70B
Command R
Command R+
Mistral Nemo
Mistral Large
GPT-4o
o3-mini
Claude 3.5 Sonnet
Figure 16: Performance on ASQA, averaged across
retrievers and document orders.

0510 25 50 75 1000.10.20.30.40.50.60.70.80.9Task Performance
Bm25 - Original
0510 25 50 75 1000.10.20.30.40.50.60.70.80.9
Bm25 - Reversed
0510 25 50 75 1000.10.20.30.40.50.60.70.80.9
Bm25 - Shuffled
0510 25 50 75 100
Retrieved Documents0.10.20.30.40.50.60.70.80.9Task Performance
Dense - Original
0510 25 50 75 100
Retrieved Documents0.10.20.30.40.50.60.70.80.9
Dense - Reversed
0510 25 50 75 100
Retrieved Documents0.10.20.30.40.50.60.70.80.9
Dense - ShuffledLlama 3.2 1B
Llama 3.2 3B
Llama 3.1 8B
Llama 3.1 70B
Command R
Command R+
Mistral Nemo
Mistral Large
GPT-4o
o3-mini
Claude 3.5 SonnetFigure 17: Performance on Natural Questions with different retrievers and document orders.
0510 25 50 75 1000.10.20.30.40.50.60.70.80.9Task Performance
Bm25 - Original
0510 25 50 75 1000.10.20.30.40.50.60.70.80.9
Bm25 - Reversed
0510 25 50 75 1000.10.20.30.40.50.60.70.80.9
Bm25 - Shuffled
0510 25 50 75 100
Retrieved Documents0.10.20.30.40.50.60.70.80.9Task Performance
Dense - Original
0510 25 50 75 100
Retrieved Documents0.10.20.30.40.50.60.70.80.9
Dense - Reversed
0510 25 50 75 100
Retrieved Documents0.10.20.30.40.50.60.70.80.9
Dense - ShuffledLlama 3.2 1B
Llama 3.2 3B
Llama 3.1 8B
Llama 3.1 70B
Command R
Command R+
Mistral Nemo
Mistral Large
GPT-4o
o3-mini
Claude 3.5 Sonnet
Figure 18: Performance on Hotpot QA with different retrievers and document orders.

0510 25 50 75 1000.10.20.30.40.50.60.70.80.9Task Performance
Bm25 - Original
0510 25 50 75 1000.10.20.30.40.50.60.70.80.9
Bm25 - Reversed
0510 25 50 75 1000.10.20.30.40.50.60.70.80.9
Bm25 - Shuffled
0510 25 50 75 100
Retrieved Documents0.10.20.30.40.50.60.70.80.9Task Performance
Dense - Original
0510 25 50 75 100
Retrieved Documents0.10.20.30.40.50.60.70.80.9
Dense - Reversed
0510 25 50 75 100
Retrieved Documents0.10.20.30.40.50.60.70.80.9
Dense - ShuffledLlama 3.2 1B
Llama 3.2 3B
Llama 3.1 8B
Llama 3.1 70B
Command R
Command R+
Mistral Nemo
Mistral Large
GPT-4o
o3-mini
Claude 3.5 SonnetFigure 19: Performance on ASQA with different retrievers and document orders.

non_rag_qa.j2
1Answer the following question in a concise manner without explanation . Indicate your
answer with " Answer :" and only include the answer words or phrases . For example
: " Question : What city is Kowloon a part of? Answer : Hong Kong ."
2
3{{ question }}
rag_qa.j2
1Based on your own knowledge and retrieved contexts , answer the question in a concise
manner without any explanation . Indicate your answer with " Answer :". For
example : " Question : What city is Kowloon a part of? Answer : Hong Kong ." If the
answer is not specified or mentioned in the retrieved context , you must ignore
the context and provide an answer by yourself . You must not refrain from
answering the question .
2
3Retrieved contexts :
4{% for c in sources %} Context {{ loop . index }}
5{{c}}
6{% endfor %}
7{{ question }}
ownknow.j2
1Previously , you answer the question with your own knowledge . Now , based on your own
knowledge and additional retrieved contexts , answer the question in a concise
manner without any explanation . Indicate your answer with " Answer :". For example
: " Question : What city is Kowloon a part of? Previous Answer : previous answer .
Answer : Hong Kong ." If the answer is not specified or mentioned in the retrieved
context , you must ignore the context and provide an answer by yourself . You
must not refrain from answering the question .
2
3Retrieved contexts :
4{% for c in sources %} Context {{ loop . index }}
5{{c}}
6{% endfor %}
7{{ question }} Previous Answer : {{ non_rag_output }}.
s2a.j2
1Identify the retrieved context (s) that would be good context for providing an
unbiased answer to the question . Indicate your selected context (s) " Selected
Contexts :". For example : " Question : What city is Kowloon a part of? Selected
Conetxts : Context 2, Context 5." If there is no retrieved context , reply with "
Selected Conetxts : None ".
2
3Retrieved contexts :
4{% for c in sources %} Context {{ loop . index }}
5{{c}}
6{% endfor %}
7{{ question }}
answer_evaluation_nq_hotpot.j2
1You will be given a question , a list of gold answers to this question , and a
predicted answer . Any one answer or multiple answers from the gold answer list
can correctly answer the question . Your task is to judge whether the predicted
answer can answer the question correctly .
2Note that predicted answer does not have to exactly match one or multiple gold
answers . It can answer the question correctly as long as its meaning entails one
or multiple gold answers and there is no any additional incorrect information .
3
4Question :
5{{ question }}
6
7Gold Answers :
8{{ gold_answer }}
9
10Predicted Answer :

11{{ pred_answer }}
12
13Is the predicted answer a correct answer to the question ?
14
15IMPORTANT : Please strictly follow the following format in your response :
16[ Start answer ]
17<Your answer . Choose from : Yes , No >
18[ End answer ]
answer_evaluation_asqa.j2
1You will be given a question , gold answers to this question , and a predicted answer .
Gold answers are composed of multiple groups . Your task is to judge whether the
predicted answer cover each group of the gold answers . Within one gold answer
group , there can be multiple alternative answers . As long as one of the
alternative answers is covered , the group is covered . Note that " cover " means "
entail ", in other words , you need to judge the predicted answer entails any
answer within each group .
2
3Question :
4{{ question }}
5
6Gold Answers :
7{% for group in short_answer %} Group {{ loop . index }}: {{ group }}
8{% endfor %}
9Predicted Answer :
10{{ pred_answer }}
11
12Does the predicted answer cover each group of the gold answers ?
13
14IMPORTANT : Please strictly follow the following format in your response :
15[ Start answer ]
16{% for group in short_answer %} Group {{ loop . index }}: <Your answer . Choose from : Yes ,
No >
17{% endfor %}[ End answer ]