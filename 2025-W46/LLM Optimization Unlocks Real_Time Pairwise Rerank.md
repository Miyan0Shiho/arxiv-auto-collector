# LLM Optimization Unlocks Real-Time Pairwise Reranking

**Authors**: Jingyu Wu, Aditya Shrivastava, Jing Zhu, Alfy Samuel, Anoop Kumar, Daben Liu

**Published**: 2025-11-10 19:04:41

**PDF URL**: [https://arxiv.org/pdf/2511.07555v1](https://arxiv.org/pdf/2511.07555v1)

## Abstract
Efficiently reranking documents retrieved from information retrieval (IR) pipelines to enhance overall quality of Retrieval-Augmented Generation (RAG) system remains an important yet challenging problem. Recent studies have highlighted the importance of Large Language Models (LLMs) in reranking tasks. In particular, Pairwise Reranking Prompting (PRP) has emerged as a promising plug-and-play approach due to its usability and effectiveness. However, the inherent complexity of the algorithm, coupled with the high computational demands and latency incurred due to LLMs, raises concerns about its feasibility in real-time applications. To address these challenges, this paper presents a focused study on pairwise reranking, demonstrating that carefully applied optimization methods can significantly mitigate these issues. By implementing these methods, we achieve a remarkable latency reduction of up to 166 times, from 61.36 seconds to 0.37 seconds per query, with an insignificant drop in performance measured by Recall@k. Our study highlights the importance of design choices that were previously overlooked, such as using smaller models, limiting the reranked set, using lower precision, reducing positional bias with one-directional order inference, and restricting output tokens. These optimizations make LLM-based reranking substantially more efficient and feasible for latency-sensitive, real-world deployments.

## Full Text


<!-- PDF content starts -->

LLM Optimization Unlocks Real-Time Pairwise Reranking
Jingyu Wu, Aditya Shrivastava, Jing Zhu, Alfy Samuel, Anoop Kumar, Daben Liu
AI Foundations, Capital One
{jingyu.wu, aditya.shrivastava2, jing.zhu, alfy.samuel, anoop.kumar, daben.liu}@capitalone.com
Abstract
Efficiently reranking documents retrieved from
information retrieval (IR) pipelines to enhance
overall quality of Retrieval-Augmented Gen-
eration (RAG) system remains an important
yet challenging problem. Recent studies have
highlighted the importance of Large Language
Models (LLMs) in reranking tasks. In par-
ticular, Pairwise Reranking Prompting (PRP)
has emerged as a promising plug-and-play ap-
proach due to its usability and effectiveness.
However, the inherent complexity of the al-
gorithm, coupled with the high computational
demands and latency incurred due to LLMs,
raises concerns about its feasibility in real-time
applications. To address these challenges, this
paper presents a focused study on pairwise
reranking, demonstrating that carefully applied
optimization methods can significantly mitigate
these issues. By implementing these methods,
we achieve a remarkable latency reduction of
up to 166 times, from 61.36 seconds to 0.37
seconds per query, with an insignificant drop in
performance measured by Recall@k. Our study
highlights the importance of design choices
that were previously overlooked, such as us-
ing smaller models, limiting the reranked set,
using lower precision, reducing positional bias
with one-directional order inference, and re-
stricting output tokens. These optimizations
make LLM-based reranking substantially more
efficient and feasible for latency-sensitive, real-
world deployments.
1 Introduction
Efficient and accurate retrieval systems are crit-
ical for delivering relevant content in industrial
applications such as customer support (Xu et al.,
2024), finance, and banking (Zhao et al., 2024a).
Retrieval-Augmented Generation (RAG) pipelines,
which integrate retrieval and generation can en-
hance response quality. Large Language Models
(LLMs) have demonstrated significant potential in
optimizing these pipelines through reranking (Zhuet al., 2023). However, their high computational
costs and latency present substantial barriers to
widespread deployment.
Pairwise Reranking Prompting (PRP) (Qin et al.,
2024), which ranks candidate documents through
pairwise comparisons, effectively enhances rele-
vance. However, its extensive computational re-
quirements make it impractical for many real-time
applications. This necessitates optimization strate-
gies that maintain accuracy while reducing compu-
tational overhead.
In this paper, we present an optimized system
(Figure 1) with PRP for industrial-scale retrieval-
augmented generation applications, aiming to
lower latency without compromising reranking per-
formance. We evaluate our approaches on two
real-world proprietary datasets. Our optimizations
include leveraging smaller LLMs like FLAN-T5-
XL, implementing single pass of sliding window
for the most salient document reranking, constrain-
ing reranking scope to a carefully chosenTop-K,
loading LLMs with lower precision, applying one-
directional order inference to mitigate positional
bias, and using constrained single-token generation.
Collectively, these methods reduce inference time
from over 60 seconds to 0.37 seconds per query
using four A100 GPUs, with minimal impact on
model performance.
Our main contributions are:
•Demonstrating that carefully optimized LLM-
based reranking is viable for real-time indus-
trial applications.
•Providing a set of generalizable latency-
reduction techniques for LLM-based tasks
that enables more effective, real-time retrieval
systems in diverse domains where relevant
information must be delivered promptly and
accurately.arXiv:2511.07555v1  [cs.CL]  10 Nov 2025

2 Background
2.1 LLM-Based Text Reranking
LLMs have driven significant advances in text
reranking (Zhu et al., 2023). Traditional ap-
proaches, predominantly listwise (e.g., RankGPT
(Sun et al., 2023b), LRL (Ma et al., 2023)) and
pointwise (e.g., (Liang et al., 2022), (Sachan et al.,
2022)), have inherent drawbacks. Listwise meth-
ods, which rank all documents simultaneously, typ-
ically underperform unless powered by highly ca-
pable LLMs such as GPT-4. By contrast, pointwise
methods, which assess each document’s relevance
independently, struggle to generate well-calibrated
scores without fine-tuning. Recent exploration of
setwise approaches (Zhuang et al., 2024) has shown
promise for efficiency, but still lags behind pairwise
methods in performance. Given these challenges,
particularly when performance cannot be traded for
speed, Pairwise Reranking Prompting (PRP) (Qin
et al., 2024) emerges as a compelling solution. PRP
leverages LLMs’ comparative strengths to rerank
documents effectively without additional training,
making it a practical choice for real-world appli-
cations, especially if latency improvements can be
realized.
2.2 Advancements in Pairwise Reranking
Prompting
Pairwise Reranking Prompting (PRP), introduced
by (Qin et al., 2024), has rapidly gained traction
in the field of information retrieval because of the
off-the-shelf usability of the pretrained LLMs with-
out additional finetuning. Its applications have ex-
tended beyond ranking to encompass broader eval-
uation tasks (Liusie et al., 2024; Park et al., 2024).
While Instruction Distillation (Sun et al., 2023a)
has shown promise, its effectiveness appears lim-
ited to LLMs of GPT-4 scale or larger. Efforts
to refine PRP have yielded approaches like LLM-
RankFusion (Zeng et al., 2024), which addresses
inconsistencies in PRP by incorporating additional
context. However, this increased context length
introduces latency concerns. Similarly, PRPGraph
(Luo et al., 2024) enhances PRP’s performance but
also faces latency bottlenecks. Despite widespread
acknowledgment of latency limitations in pairwise
methods (Liu et al., 2024b; Pradeep et al., 2023; Liu
et al., 2024a; Zhao et al., 2023; Zhu et al., 2023),
mitigating these challenges without compromising
performance remains an open research area.3 Method
In this section, we provide an in-depth description
of the optimization techniques explored to improve
the inference latency of the PRP methodology.
3.1 Model Size Reduction
We first analyze the importance of model size in
the Pairwise Reranking approach. Our experiments
primarily use the FLAN-T5 (Raffel et al., 2020)
family of models (FLAN-T5-XXL (11.3B), FLAN-
T5-XL (2.85B), and FLAN-T5-Large (783M)). We
additionally extend the comparison to FLAN-UL2
(20B) (Tay et al., 2022), which is also a FLAN-T5
based model. The core step in Pairwise Reranking
involves comparing the relevance of two documents
with respect to a given user query.
Based on our findings, we conclude that for rel-
atively simple tasks—such as text comparisons
requiring concise outputs—larger models do not
necessarily offer additional performance benefits
compared to smaller models. In contrast, using a
smaller model can substantially improve efficiency
by reducing latency and resource requirements.
3.2 Sliding Window with Single Pass
Traditional reranking methods resemble full sort-
ing, where the retrieved documents are reranked
based on their relevance with the query. However,
such global reordering incurs substantial computa-
tional overhead and proves redundant when only a
concentrated set of highly relevant results is sought.
Our approach employs a sliding window mech-
anism (Qin et al., 2024) with a single pass to it-
eratively identifying the most salient document,
significantly reduces computational complexity to
O(n). While extended to identify a small subset of
highly ranked documents, our experimental results
(Table 1) consistently indicate that optimizing be-
yond the single highest-ranked document (top-1)
yields only marginal improvements in recall rate.
This suggests the initial highest-ranked document
captures predominant relevant information, mak-
ing top-1 optimization the most resource-efficient
strategy for our datasets.
Rerank Recall@k(Dataset 1)Recall@k(Dataset 2)Latency
Goal k=5 k=3 k=1 k=5 k=3 k=1 (s)
Top-5 0.77 0.73 0.56 0.79 0.75 0.54 1.97
Top-3 0.76 0.73 0.56 0.79 0.75 0.54 1.06
Top-1 0.75 0.71 0.56 0.77 0.71 0.540.37
Table 1: Sliding Window on Different Reranking Goals

Figure 1: The Optimized Pairwise Reranking System
3.3 Optimizing the Reranking Threshold
Top-K
The reranking stage, irrespective of the underly-
ing algorithm, introduces a hyperparameter,Top-
K, which specifies how many of the retriever’s
top-ranked results are subsequently passed to the
reranker. In our investigation, we systematically
testTop-Kto measure its impact on both retrieval
accuracy and efficiency. We find that while increas-
ingTop-Kcan improve accuracy up to a certain
extent, any further increase yields diminishing re-
turns.
In practice, an appropriateTop-Kvalue can be
identified using grid search, binary search, or sim-
ple trial-and-error. Once recall rate meets the de-
sired performance level, the additional computa-
tional cost of reranking more candidates typically
outweighs the marginal performance gains. Hence,
pinpointing the “sweet spot” that optimizes both
efficiency and accuracy becomes crucial.
3.4 Loading Models with Lower Precision
We further explore lowering the floating-point pre-
cision from float32 tofloat16 orbfloat16
when loading model weights, reducing computa-
tional cost without substantially degrading perfor-
mance.
Our findings with different precision settings
highlight that even modest reductions from
float32 tobfloat16 can notably decrease infer-
ence time.
Separately, we also explore quantization, which
maps model weights from floating-point to lower-
precision integer representations (e.g., 8-bit or 4-
bit). While quantization can provide additionalspeedups beyond mere precision reduction, we ob-
serve that it introduces more substantial changes to
the numerical representations, resulting in a notice-
able performance decline. Given these trade-offs,
we opt not to include quantization in our final im-
plementation.
3.5 One-directional Order Inference
Recent studies have shown that LLMs are prone to
positional bias, exhibiting sensitivity to the order
in which input elements are presented (Qin et al.,
2024; Zeng et al., 2024; Zhao et al., 2024b; Luo
et al., 2024). A common mitigation strategy in-
volves performing pairwise comparisons in both
directions to neutralize input-order effects.
In this work, we investigate a latency-efficient al-
ternative using LLMs by adopting a one-directional
order inference scheme with deliberate position as-
signment. Specifically, during inference, we consis-
tently designate the document with the lower initial
retrieval ranking as input A and the higher-ranked
document as input B. This configuration ensures
that the model processes the lower-ranked docu-
ment first, which we hypothesize reduces the ten-
dency to favor the first input, thereby partially miti-
gating positional bias while improving efficiency.
3.6 Constrained Decoding
In pairwise reranking, most of the inference time
is spent generating output tokens. To mitigate this,
we constrain the model to produce exactly one to-
ken (e.g., “A” or “B”) through prompt engineering,
thereby reducing inference overhead. However,
simply taking the first token generated as the final
output can lead to invalid or irrelevant response
unless the prompt is carefully designed.

To address this, we systematically test multiple
prompt variations and select the one (in Figure 1)
that yields the highest accuracy under single-token
constraints. Algorithm 1 outlines our approach:
we evaluate each candidate prompt on a test set,
record its accuracy, and select the best-performing
prompt for constrained decoding. Combined with
greedy decoding at zero temperature, this approach
achieves performance on par with multi-token out-
puts while substantially reducing computational
time.
Algorithm 1:Iterative Prompt Selection
for Constrained Decoding
Input:ModelM, Test Set
D={(cA
i, cB
i, yi)}, Prompt
CandidatesP
Output: Best Prompt p∗with Single-Token
Decoding
Initialize bestAccuracy←0,p∗←None;
foreachp∈Pdo
correctCount←0;
foreach(cA, cB, y)inDdo
fullPrompt
←formatPrompt(p, cA, cB);
outputToken
←generate(M,prompt=
fullPrompt,max_new_tokens=
1,greedy=True,temperature=
0);
ifoutputToken =ythen
correctCount
←correctCount+ 1;
end
end
accuracy←correctCount
|D|;
ifaccuracy > bestAccuracythen
bestAccuracy←accuracy,p∗←p;
end
end
returnp∗
4 Experiments
4.1 Data
We employ two distinct datasets, dataset 1 and
dataset 2, both derived from proprietary internal
use cases for our experiments. Each dataset corre-
sponds to a unique business vertical. Within each
dataset, the structure comprises:•A document corpus that serves as the primary
contextual knowledge base.
•A test set consisting of user-generated queries,
each meticulously paired with its correspond-
ing relevant context from the corpus.
4.1.1 Corpus
The corpus is the knowledge base for a business
vertical. The corpus in dataset 1 consists of 8379
documents, while corpus in dataset 2 consists of
8121 documents. Considering the token limit of
the model input, the maximum token length of
each document is 512. These documents repre-
sent segmented, structured entries derived from
real business knowledge bases used in production
environments.
4.1.2 Test Set
Two test sets are included in the datasets corre-
sponding to the corpora described above. Each test
set contains real user queries derived from actual
usage of the internal search system. For each query,
a relevant context—a segment from the knowledge
base that directly answers the query—is provided,
along with a corresponding link that identifies the
document in the knowledge system where the con-
text is found. Both the context and the link are
manually annotated by domain experts for each re-
spective business vertical, serving as ground truth
in our experiments. The volume of test sets is 975
queries for dataset 1 and 736 queries for dataset 2.
Dataset 1 Dataset 2
Total Documents in Corpus 8379 8121
Tokens per Document (Min) 192 214
Tokens per Document (Avg.) 474 509
Tokens per Document (Max) 512 512
Total Queries 975 736
Table 2: Statistics on the Two Datasets Used in Our
Experiments
4.2 Evaluation Metrics
In our experiments, Recall rate is selected as the pri-
mary evaluation metric, aligning directly with our
industry-specific use cases. This metric quantifies
the proportion of relevant documents successfully
retrieved by the system out of the total available
documents in the corpus. We specifically utilize
Recall@k, as our datasets feature a single correct
document as ground truth for each query. This met-
ric directly measures the percentage of queries for

Reranking Recall@k(Dataset 1)Recall@k(Dataset 2)
Strategy k=25 k=10 k=5 k=3 k=1 k=25 k=10 k=5 k=3 k=1
Cross-encoder 0.90 0.85 0.80 0.73 0.52 0.90 0.83 0.76 0.67 0.47
Pairwise 0.90 0.85 0.80 0.750.580.90 0.85 0.79 0.740.54
Table 3: Pairwise Reranking Outperforms Cross-encoder Reranking on Both Datasets
which the unique relevant document is successfully
identified within the top-k retrieved results.
Alternative metrics, such as Normalized Dis-
counted Cumulative Gain (NDCG), which are de-
signed for scenarios with multiple relevant docu-
ments and graded relevance, are deemed less ap-
propriate and potentially overly complex for our
specific single-answer retrieval task.
4.3 System Settings
We experiment with different computational re-
sources. The latency on A100 is significantly lower
than A10G GPUs. All the experiments reported in
this paper are using four A100 GPUs.
4.4 Retriever Model Setup
Our baseline retriever model ismulti-qa-mpnet-
base-cos-v1, a sentence-transformer model trained
on 215M {question, answer} pairs. It is the cham-
pion retriever model on our datasets.
4.5 Comparison with Classical Ranking
Techniques
To substantiate the efficacy and necessity of em-
ploying pairwise reranking within our system, we
conduct a comparative analysis against a more es-
tablished reranking paradigm. Specifically, we
benchmark its performance against cross-encoder
reranking, a widely adopted strategy in information
retrieval tasks.
Our evaluation involves comparing the perfor-
mance between the cross-encoder and the pairwise
reranker when applies atop the baseline retriever
previously described. We usebge-reranker-v2-m3
as the cross-encoder reranker. At the time of this
study’s inception, this model is recognized for its
state-of-the-art performance in the field of cross-
encoder reranking. Its demonstrated capability in
achieving high retrieval effectiveness makes it a
compelling and robust choice for our comparative
analysis. As evidenced in Table 3, the pairwise
reranker consistently demonstrates superior perfor-
mance across both datasets. This significant outper-
formance underscores the potency of the pairwise
reranker for complex reranking tasks.Despite its inherently higher latency compared
to classical cross-encoder reranking, the enhanced
effectiveness of the pairwise approach validates its
integration into the retrieval pipeline, making it a
crucial component for achieving optimal ranking
quality.
5 Results and Discussion
5.1 Baseline
As illustrated in the first half of Table 4, the base-
line application of pairwise reranking produces a
marked improvement in Recall@k. For instance,
in Dataset 1, Recall@1 jumps from 0.42 (Retriever
Only) to 0.58 (Retriever + Pairwise Reranker), con-
firming that the llm-based comparison of compet-
ing candidate documents substantially enhances
ranking performance. We observe a similar trend
in Dataset 2, where pairwise reranking boosts Re-
call@1 from 0.50 to 0.54. These improvements
underscore the significance of the problem being
addressed—enabling more accurate ranking of rel-
evant information despite the inherent challenges
posed by large and complex databases.
5.2 Optimization Results
The lower half of Table 4 provides a comprehensive
breakdown of the latency improvements achieved
through the optimization techniques discussed in
Section 3. Since the latency values for Dataset 1
and Dataset 2 are approximately equivalent across
all methods, we report latency results for Dataset
2 for clarity and conciseness. Each method con-
tributes uniquely to enhancing efficiency of our
Pairwise Reranking algorithm. Starting with the
replacement of FLAN-UL2 with FLAN-T5-XL, a
model that is approximately seven times smaller,
we observe 2.7 times improvement in latency. No-
tably, this change also results in a slight increase in
retrieval performance, demonstrating that smaller
models can effectively handle tasks requiring con-
cise outputs, such as pairwise comparisons, without
compromising accuracy.
Among all the optimization strategies,Top-K
tuning yields the most significant performance

Recall@k(Dataset 1)Recall@k(Dataset 2)Latency Speedup
Setup k=10 k=5 k=3 k=1 k=10 k=5 k=3 k=1 (s) Factor
Baseline (Retriever Only) 0.77 0.68 0.62 0.42 0.80 0.73 0.67 0.50 - -
+ Pairwise Reranker 0.85 0.80 0.750.580.85 0.79 0.740.5461.36 -
+ Flan-UL2→Flan-T5-XL 0.85 0.81 0.76 0.59 0.85 0.81 0.75 0.56 22.50∼2.7×
+ TopK=25→TopK=5 0.81 0.72 0.66 0.52 0.84 0.77 0.71 0.54 3.27∼6.9×
+float32→bfloat160.81 0.72 0.66 0.52 0.84 0.77 0.71 0.54 2.23∼1.5×
+ One-directional Order 0.81 0.72 0.66 0.52 0.84 0.77 0.71 0.54 1.11∼2×
+ Constrained Decoding 0.81 0.72 0.66 0.52 0.84 0.77 0.71 0.540.37∼3×
(Speedup Gain in Total)∼167×
Table 4: Model Performance, Optimization Methods and Their Corresponding Latency Improvements on Two
Datasets
gains. While reducingTop-Kfrom 25 to 5 results
in a drop in Recall@1 (0.59 to 0.52 on Dataset
1 and 0.56 to 0.54 for Dataset 2), the broader re-
trieval quality—particularly the consistency of Re-
call@10—remains stable. This indicates that while
fewer candidates are processed at the reranking
stage, the model retains its ability to identify the
most relevant documents within the top ranked doc-
uments. This optimization achieves a 6.9 times
improvement in latency. The computational cost
saving far outweighs the marginal reduction in top-
1 accuracy. We consider this the second most ef-
fective optimization, as it balances efficiency and
effectiveness for tasks that prioritize recall perfor-
mance on a small number of top search results.
One of the most meaningful observations is the
role of constrained decoding. By enforcing single-
token outputs in addition to all other optimiza-
tion strategies, we lower latency by a factor of 3
times without any recall drop and finally achieve
a runtime latency of lower than 1 second–at just
0.37 seconds per query. This improvement fun-
damentally changes the feasibility of deploying
pairwise reranking in real-time applications, where
sub-second responsiveness is often critical. Re-
markably, this drastic gain in efficiency does not
compromise retrieval accuracy, as Recall@1 re-
mains consistent with other optimized configura-
tions. This demonstrates that with careful prompt
design and decoding constraints, tasks requiring
categorical outputs can achieve near-optimal per-
formance with significantly reduced computational
overhead.
Combined with one-directional order inference,
which halves the latency associated with positional
bias, and reduces model precision ( bfloat16 ),
these strategies collectively transforms the system’sscalability. The cumulative impact of these opti-
mizations underscores a key takeaway: a holistic
approach to PRP optimization, targeting both al-
gorithmic bottlenecks and hardware efficiency, is
essential and feasible for real-world scalability of
LLM-based pairwise reranking solutions.
6 Conclusion and Future Work
We address the challenge of integrating Pairwise
Reranking Prompting (PRP) into real-world re-
trieval and RAG systems, showing that careful op-
timizations can reduce per-query latency from over
60 seconds to well under a second with negligible
performance loss. These optimizations—selecting
smaller models, restricting the reranked set, adopt-
ing lower precision, mitigating positional bias via
one-directional order inference, and constraining
output tokens—collectively transform PRP into a
viable option for latency-sensitive applications.
In future research, we plan to extend our
optimization framework to other reranking
paradigms—listwise, pointwise, and hybrid—to
assess whether similar latency–accuracy trade-offs
can be achieved. We also aim to explore dynamic
reranking strategies that adjust the Top-K value
based on query complexity or system load, poten-
tially improving both recall and responsiveness in
practical settings.
While model parallelism offers limited benefit
in our current setup—likely due to the relatively
modest model size—we see data or pipeline par-
allelism as promising areas for further investiga-
tion. These strategies may offer greater gains under
higher-throughput conditions or when scaling to
larger models. Finally, we anticipate further per-
formance gains could be realized through model-
or task-specific tuning and more effective batching

strategies within the reranking pipeline.
By systematically balancing performance and
computational efficiency, our approach outlines a
robust, scalable blueprint for real-time LLM-based
retrieval solutions. In doing so, we bridge the gap
between cutting-edge research and practical deploy-
ment constraints across diverse industry settings.
Limitations
While our proposed optimizations significantly re-
duce PRP latency and improve deployability in real-
world RAG systems, several limitations remain.
First, our reranking improvements rely on a fixed
Top-K strategy. The current system does not adjust
reranking scope based on query complexity or load,
which may limit efficiency under highly variable
conditions.
Second, our reranking pipeline—though opti-
mized—does not yet incorporate task-specific or
domain-adaptive tuning. This may constrain per-
formance in specialized applications (e.g., legal,
medical) where domain-specific language and re-
trieval behavior differ significantly from the train-
ing regime.
References
Tri Dao. 2023. Flashattention-2: Faster attention with
better parallelism and work partitioning.arXiv
preprint arXiv:2307.08691.
Percy Liang, Rishi Bommasani, Tony Lee, Dimitris
Tsipras, Dilara Soylu, Michihiro Yasunaga, Yian
Zhang, Deepak Narayanan, Yuhuai Wu, Ananya Ku-
mar, et al. 2022. Holistic evaluation of language
models.arXiv preprint arXiv:2211.09110.
Qi Liu, Bo Wang, Nan Wang, and Jiaxin Mao. 2024a.
Leveraging passage embeddings for efficient listwise
reranking with large language models.arXiv preprint
arXiv:2406.14848.
Yinhong Liu, Han Zhou, Zhijiang Guo, Ehsan Shareghi,
Ivan Vuli ´c, Anna Korhonen, and Nigel Collier. 2024b.
Aligning with human judgement: The role of pair-
wise preference in large language model evaluators.
InFirst Conference on Language Modeling.
Adian Liusie, Potsawee Manakul, and Mark Gales. 2024.
LLM comparative assessment: Zero-shot NLG eval-
uation through pairwise comparisons using large lan-
guage models. InProceedings of the 18th Confer-
ence of the European Chapter of the Association for
Computational Linguistics (Volume 1: Long Papers),
pages 139–151, St. Julian’s, Malta. Association for
Computational Linguistics.Jian Luo, Xuanang Chen, Ben He, and Le Sun. 2024.
PRP-graph: Pairwise ranking prompting to LLMs
with graph aggregation for effective text re-ranking.
InProceedings of the 62nd Annual Meeting of the
Association for Computational Linguistics (Volume 1:
Long Papers), pages 5766–5776, Bangkok, Thailand.
Association for Computational Linguistics.
Xueguang Ma, Xinyu Zhang, Ronak Pradeep, and
Jimmy Lin. 2023. Zero-shot listwise document
reranking with a large language model.arXiv
preprint arXiv:2305.02156.
ChaeHun Park, Minseok Choi, Dohyun Lee, and Jaegul
Choo. 2024. Paireval: Open-domain dialogue eval-
uation with pairwise comparison.arXiv preprint
arXiv:2404.01015.
Ronak Pradeep, Sahel Sharifymoghaddam, and Jimmy
Lin. 2023. Rankzephyr: Effective and robust zero-
shot listwise reranking is a breeze!arXiv preprint
arXiv:2312.02724.
Zhen Qin, Rolf Jagerman, Kai Hui, Honglei Zhuang,
Junru Wu, Le Yan, Jiaming Shen, Tianqi Liu, Jialu
Liu, Donald Metzler, Xuanhui Wang, and Michael
Bendersky. 2024. Large language models are effec-
tive text rankers with pairwise ranking prompting. In
Findings of the Association for Computational Lin-
guistics: NAACL 2024, pages 1504–1518, Mexico
City, Mexico. Association for Computational Lin-
guistics.
Colin Raffel, Noam Shazeer, Adam Roberts, Katherine
Lee, Sharan Narang, Michael Matena, Yanqi Zhou,
Wei Li, and Peter J Liu. 2020. Exploring the lim-
its of transfer learning with a unified text-to-text
transformer.Journal of machine learning research,
21(140):1–67.
Devendra Sachan, Mike Lewis, Mandar Joshi, Armen
Aghajanyan, Wen-tau Yih, Joelle Pineau, and Luke
Zettlemoyer. 2022. Improving passage retrieval with
zero-shot question generation. InProceedings of
the 2022 Conference on Empirical Methods in Nat-
ural Language Processing, pages 3781–3797, Abu
Dhabi, United Arab Emirates. Association for Com-
putational Linguistics.
Weiwei Sun, Zheng Chen, Xinyu Ma, Lingyong Yan,
Shuaiqiang Wang, Pengjie Ren, Zhumin Chen,
Dawei Yin, and Zhaochun Ren. 2023a. Instruction
distillation makes large language models efficient
zero-shot rankers.arXiv preprint arXiv:2311.01555.
Weiwei Sun, Lingyong Yan, Xinyu Ma, Shuaiqiang
Wang, Pengjie Ren, Zhumin Chen, Dawei Yin, and
Zhaochun Ren. 2023b. Is chatgpt good at search?
investigating large language models as re-ranking
agents.Preprint, arXiv:2304.09542.
Yi Tay, Mostafa Dehghani, Vinh Q Tran, Xavier Garcia,
Jason Wei, Xuezhi Wang, Hyung Won Chung, Sia-
mak Shakeri, Dara Bahri, Tal Schuster, et al. 2022.
Ul2: Unifying language learning paradigms.arXiv
preprint arXiv:2205.05131.

Zhentao Xu, Mark Jerome Cruz, Matthew Guevara,
Tie Wang, Manasi Deshpande, Xiaofeng Wang, and
Zheng Li. 2024. Retrieval-augmented generation
with knowledge graphs for customer service ques-
tion answering. InProceedings of the 47th Inter-
national ACM SIGIR Conference on Research and
Development in Information Retrieval, SIGIR ’24,
page 2905–2909, New York, NY , USA. Association
for Computing Machinery.
Yifan Zeng, Ojas Tendolkar, Raymond Baartmans,
Qingyun Wu, Huazheng Wang, and Lizhong Chen.
2024. Llm-rankfusion: Mitigating intrinsic in-
consistency in llm-based ranking.arXiv preprint
arXiv:2406.00231.
Huaqin Zhao, Zhengliang Liu, Zihao Wu, Yiwei Li,
Tianze Yang, Peng Shu, Shaochen Xu, Haixing Dai,
Lin Zhao, Gengchen Mai, et al. 2024a. Revolutioniz-
ing finance with llms: An overview of applications
and insights.arXiv preprint arXiv:2401.11641.
Wayne Xin Zhao, Kun Zhou, Junyi Li, Tianyi Tang,
Xiaolei Wang, Yupeng Hou, Yingqian Min, Beichen
Zhang, Junjie Zhang, Zican Dong, et al. 2023. A
survey of large language models.arXiv preprint
arXiv:2303.18223.
Xiutian Zhao, Ke Wang, and Wei Peng. 2024b. Mea-
suring the inconsistency of large language models in
preferential ranking. InProceedings of the 1st Work-
shop on Towards Knowledgeable Language Models
(KnowLLM 2024), pages 171–176, Bangkok, Thai-
land. Association for Computational Linguistics.
Yutao Zhu, Huaying Yuan, Shuting Wang, Jiongnan
Liu, Wenhan Liu, Chenlong Deng, Haonan Chen,
Zhicheng Dou, and Ji-Rong Wen. 2023. Large lan-
guage models for information retrieval: A survey.
arXiv preprint arXiv:2308.07107.
Shengyao Zhuang, Honglei Zhuang, Bevan Koopman,
and Guido Zuccon. 2024. A setwise approach for
effective and highly efficient zero-shot ranking with
large language models. InProceedings of the 47th
International ACM SIGIR Conference on Research
and Development in Information Retrieval, SIGIR
’24, page 38–47, New York, NY , USA. Association
for Computing Machinery.

A Appendices
A.1 Methods Tried but Not Successful
Model Parallelism
From the model structure perspective, a com-
monly used inference optimization method is
model parallelism. Model parallelism distributes
different layers of models in different GPUs and
accelerates inference by doing computations on
different GPUs parallelly. However, in our experi-
ments, model parallelism doesn’t give a huge im-
provement on latency. The reason behind could
be the fact that Flan-T5-XL is small enough to be
loaded in one A100 GPU. Since doing computation
on only one GPU at a time can save the commu-
nication cost between GPUs, smaller models may
benefit more when being loaded into one GPUs
rather than multiple GPUs. Thus, we decide to
pause on model parallelism experiments.
Altering Attention Mechanisms
In this section, we describe our efforts to opti-
mize the latency of Pairwise Reranking by incor-
porating faster attention mechanisms. Specifically,
we explore three different attention mechanisms
applied to the T5 family of models: SDPA, Eager
and Flash Attention 2 (Dao, 2023).
We experiment with various attention mecha-
nisms, some of which do not yield significant im-
provements. Importantly, we find that these mech-
anisms improve performance with larger models,
such as Flan-T5-XXL. However, when we transi-
tion to smaller models through prompt engineering,
the gains are minimal. This suggests a trade-off
between model size and the benefits of faster at-
tention mechanisms. Larger models tend to benefit
more, while smaller models may not exhibit the
same level of improvement.
Batch Inference
We also systematically experiment with a range
of batch sizes to evaluate the impact on perfor-
mance. In this procedure, the input batch in the
first round contains approximately n/2samples or
pairs, where nis the total number of samples in
the initial batch. After processing the first batch,
the most similar text pairs are promoted to the next
round, which reduces the number of samples to
n/4, and so on. This process continues for ap-
proximately log2nrounds, depending on the total
number of initial samples.
Figure 2: Performance comparison in terms of latency
per query of different models with faster attention mech-
anisms (from top to bottom: Flan-T5-XXL, Flan-UL2,
Flan-XL, and Flan-T5-Large).

Version Primary Prompt
Original """Given a query {query}, which of the following two passages is more relevant
to the query?
Passage A: {doc1}
Passage B: {doc2}
Output Passage A or Passage B:"""
Passage Mode """Given a query {query}, which of the following two passages is more relevant
to the query?
Passage A: {doc1}
Passage B: {doc2}
Output the more relevant Passage:"""
Number Mode """Given a query {query}, which of the following passages is more relevant to
the query?
Passage 1: {doc1}
Passage 2: {doc2}
The answer is captured in a * format. For example, *2* or *1*. Just return the
number of the more relevant Passage:"""
Final Version """Given a query {query}, which of the following two passages is more relevant
to the query?
A: {doc1}
B: {doc2}
Output A or B:"""
Table 5: Primary Prompt Variations in Our Experiments
A.2 Prompt List
Table 5 shows the list of primary variations of
prompts that we have experimented with.
We conduct a systematic evaluation of several
prompt variations, ultimately choosing the final ver-
sion that achieves the highest performance which
enables single-token constraints.