# RAGtifier: Evaluating RAG Generation Approaches of State-of-the-Art RAG Systems for the SIGIR LiveRAG Competition

**Authors**: Tim Cofala, Oleh Astappiev, William Xion, Hailay Teklehaymanot

**Published**: 2025-06-17 11:14:22

**PDF URL**: [http://arxiv.org/pdf/2506.14412v1](http://arxiv.org/pdf/2506.14412v1)

## Abstract
Retrieval-Augmented Generation (RAG) enriches Large Language Models (LLMs) by
combining their internal, parametric knowledge with external, non-parametric
sources, with the goal of improving factual correctness and minimizing
hallucinations. The LiveRAG 2025 challenge explores RAG solutions to maximize
accuracy on DataMorgana's QA pairs, which are composed of single-hop and
multi-hop questions. The challenge provides access to sparse OpenSearch and
dense Pinecone indices of the Fineweb 10BT dataset. It restricts model use to
LLMs with up to 10B parameters and final answer generation with Falcon-3-10B. A
judge-LLM assesses the submitted answers along with human evaluators. By
exploring distinct retriever combinations and RAG solutions under the challenge
conditions, our final solution emerged using InstructRAG in combination with a
Pinecone retriever and a BGE reranker. Our solution achieved a correctness
score of 1.13 and a faithfulness score of 0.55, placing fourth in the SIGIR
2025 LiveRAG Challenge.

## Full Text


<!-- PDF content starts -->

arXiv:2506.14412v1  [cs.IR]  17 Jun 2025RAGtifier: Evaluating RAG Generation Approaches of
State-of-the-Art RAG Systems for the SIGIR LiveRAG Competition
Tim Cofala
L3S Research Center
Hannover, Germany
tim.cofala@l3s.deOleh Astappiev
L3S Research Center
Hannover, Germany
astappiev@l3s.de
William Xion
L3S Research Center
Hannover, Germany
william.xion@l3s.deHailay Teklehaymanot
L3S Research Center
Hannover, Germany
teklehaymanot@l3s.de
ABSTRACT
Retrieval-Augmented Generation (RAG) enriches Large Language
Models (LLMs) by combining their internal, parametric knowl-
edge with external, non-parametric sources, with the goal of im-
proving factual correctness and minimizing hallucinations. The
LiveRAG 2025 challenge explores RAG solutions to maximize accu-
racy on DataMorgana’s QA pairs, which are composed of single-
hop and multi-hop questions. The challenge provides access to
sparse OpenSearch and dense Pinecone indices of the Fineweb
10BT dataset. It restricts model use to LLMs with up to 10B pa-
rameters and final answer generation with Falcon-3-10B. A judge-
LLM assesses the submitted answers along with human evaluators.
By exploring distinct retriever combinations and RAG solutions
under the challenge conditions, our final solution emerged using
InstructRAG in combination with a Pinecone retriever and a BGE
reranker. Our solution achieved a correctness score of 1.13and a
faithfulness score of 0.55, placing fourth in the SIGIR 2025 LiveRAG
Challenge1. The RAGtifier code is publicly available2.
KEYWORDS
Retrieval Augmented Generation, Adaptive Retrieval, Self-Reflection,
LLMs-as-Judges
ACM Reference Format:
Tim Cofala, Oleh Astappiev, William Xion, and Hailay Teklehaymanot.
2025. RAGtifier: Evaluating RAG Generation Approaches of State-of-the-
Art RAG Systems for the SIGIR LiveRAG Competition. In Proceedings of
SIGIR 2025 LiveRAG Challenge (SIGIR). ACM, New York, NY, USA, 7 pages.
https://doi.org/XXXXXXX.XXXXXXX
1Non-final LLM scoring rankings from the organizer. [Accessed 22-05-2025]
2https://git.l3s.uni-hannover.de/liverag/ragtifier
Permission to make digital or hard copies of all or part of this work for personal or
classroom use is granted without fee provided that copies are not made or distributed
for profit or commercial advantage and that copies bear this notice and the full citation
on the first page. Copyrights for components of this work owned by others than the
author(s) must be honored. Abstracting with credit is permitted. To copy otherwise, or
republish, to post on servers or to redistribute to lists, requires prior specific permission
and/or a fee. Request permissions from permissions@acm.org.
SIGIR, July 17, 2025, Padua, IT
©2025 Copyright held by the owner/author(s). Publication rights licensed to ACM.
ACM ISBN 978-1-4503-XXXX-X/2018/06. . . $15.00
https://doi.org/XXXXXXX.XXXXXXX
Gemma3 -27B
Claude 3.5 HaikuSimple Prompt
TrustRAG
AstuteRAG
InstructRAG
IterDRAGOpenSearch
PineconeRank -R1
BGERetriever Reranker Evaluation GenerationFigure 1: Overview of the RAG solutions and components
under consideration. The highlighted path highlights our
submitted solution based on the performance evaluation by
Gemma-3-27B and Claude-3.5-Haiku.
1 INTRODUCTION
Retrieval Augmented Generation (RAG) has emerged as a key tech-
nique for enhancing LLM performance in question answering (QA)
by incorporating external knowledge [ 21,34]. The LLM prompt is
enriched with retrieved information to mitigate issues related to
unknown or sparse knowledge within the model itself [24].
On the live challenge day of the LiveRAG Challenge, partici-
pants are provided with 500 DataMorgana-generated [ 15] ques-
tions. The generation of answers with Falcon-3-10B [ 4] and their
submission is limited to a two-hour time slot. The organizers of the
LiveRAG Challenge also provide access to the sparse search engine
OpenSearch [ 25] and the dense vector database Pinecone [ 22], both
populated with data from Fineweb-10BT [ 16]. Subsequently, the
State-of-the-Art (SotA) LLM, Claude-3.5-Sonnet, evaluates the sub-
mitted answers according to correctness [ 23] and faithfulness [ 13]
scores. The top-ranked submissions are also manually evaluated to
determine the final ranking.
Our approach (Figure 1) combines the cross-evaluation of SotA
RAG solutions with research insights on the optimization of distinct
RAG components. We benchmarked five different generation strate-
gies, two retrievers, two rerankers, and context-ordering techniques
using two evaluation LLMs and various single- and multi-hop ques-
tions generated by DataMorgana for our internal benchmark.
2 RELATED-WORK
In this section, we outline the relevant components of a complete
RAG pipeline and associated concepts.
Retriever .It collects and evaluates information to expand LLM
queries using sparse (lexical) or dense (semantic) methods [ 9,17].

SIGIR, July 17, 2025, Padua, IT Trovato et al.
Sparse methods like BM25 [ 26] are used in OpenSearch [ 25], while
dense methods may use ANN-based systems like Pinecone [ 22].
Recent work compares retriever performance on QA tasks [ 18] and
investigates context length and document order effects [9].
Reranker .This component rearranges the retrieved documents
to enhance contextual relevance [ 33]. Recent research in rerank-
ing addresses faster comparison methods [ 40] or improved search
relevance [7, 27].
Generation .The generation process consists of using the re-
trieved documents as context to generate a coherent and relevant
answer to the question. We select five SotA RAG solutions (Figure 1)
that exclusively consider retrieval-augmented prompts [ 20] or ad-
ditionally reflect on retrieved passages [ 29,37]. Further approaches
include the comparison of passages with the parametric knowledge
of LLMs [ 28] or the execution of retrievals and generation in several
rounds [32].
Evaluation .Judging generated answers by LLMs [ 35] became
an alternative to simple string matching and inefficient, expen-
sive human evaluators [ 8]. Influencing factors of LLM-judges in-
clude bias through prompt styles [ 14], answer-length [ 12] or cross-
capability performance [ 36]. Mitigation approaches suggest the
expensive use of high-performance LLMs [ 19], finetuning [ 38] or
confidence estimation [ 19]. We select two judge-LLMs for our per-
formance metric to evaluate the generated answers.
3 RAG COMPONENTS
In this section, we provide a detailed overview of the RAG compo-
nents considered for the LiveRAG Challenge, followed by a descrip-
tion of the components used in our submission.
3.1 DataMorgana
DataMorgana [ 15] is a novel approach to generate QA-pairs from
documents by defining diverse user and question categories to
create highly customizable synthetic benchmarks. We define addi-
tional user and question categories (Figure 3) to increase the variety
of generated questions, thereby further challenging the answer
generation capabilities. From a pool of 10,000 generated questions,
we created a dataset of 500 randomly selected QA-pairs, evenly
distributed across single- and multi-document subsets.
3.2 Retriever
Our RAG solution considers the dense retriever implemented with
Pinecone. This choice is based on experiments with the QA-pairs
we generated. We compared both provided indices, where the dense
retriever demonstrated faster response times and a higher retrieval
rate of gold documents (@k), both with and without an additional
reranker.
3.3 Reranker
We investigate the performance of BGE-M3 [ 7] and Rank-R1 [ 39].
Both rerankers aim to improve document relevance by reorder-
ing retrieved documents according to their relevance to the input
query, thereby enhancing the quality of the context provided to the
generation model.We investigate the key performance characteristics of the BGE-
M3 reranker, focusing on its latency in combination with different
amounts of retrieved documents, its ability to handle diverse queries
and document lengths for context understanding, and its ranking
accuracy with distinct queries and retriever settings to determine
the optimal configurations.
In exploring alternative SotA reranking methods, we also inves-
tigate Rank-R1 [ 39], a novel LLM-based reranker notable for its
explicit reasoning capabilities. However, Rank-R1’s application was
ultimately deemed impractical due to its processing time, which
can take up to 100𝑠for a single query, making it unsuitable for the
time constraints of the LiveRAG Challenge.
3.4 Generation
We consider recent advances in RAG and cross-evaluate various
answer generation approaches with distinct retriever and reranker
settings to compare their performance on DataMorgana-generated
QA-pairs. We use a non-finetuned Falcon-3-10B LLM for all gen-
eration tasks, with a temperature setting of 0.1. We consider the
generation prompts from the following RAG approaches:
Simple Prompt. The instruction utilizes direct-input augmenta-
tion for answer generation, combining the retrieved documents
followed by the query [20].
TrustRAG. The solution proposes a three-step process where the
retrieved information is compared against the parametric knowl-
edge to filter out malicious or irrelevant documents, aiming to
enhance the security and reliability of answer generation against
retrieval-influenced corpus poisoning attacks [37].
InstructRAG. That strategy introduces a framework for explicit
context denoising in retrieval-augmented generation through a
two-phase methodology [ 29]. First, rationale generation utilizes
the LLM’s instruction-following capabilities to identify relevant
information in noisy inputs. Second, explicit denoising learning
employs synthesized rationales as demonstrations or training data
that enable effective denoising strategies.
AstuteRAG. A framework addressing imperfect retrieval results
through a three-phase process: adaptive elicitation of internal model
knowledge, source-aware knowledge consolidation, and reliability-
based answer finalization [ 28]. In contrast to conventional RAG im-
plementations, Astute RAG explicitly identifies and resolves knowl-
edge conflicts between the model’s parametric knowledge and the
retrieved information and adaptively combines the most reliable
elements from each source.
Iterative Demonstration-Based RAG (IterDRAG). An iterative ap-
proach based on Demonstration-based RAG (DRAG) [ 6], where con-
textualized examples guide the LLM in its long-context usage [ 32].
IterDRAG extends this approach by incorporating a multi-round
question refinement process, specifically targeting multi-hop ques-
tions. It decomposes the main question into sub-queries, generates
an answer for each sub-question, which can additionally be re-
trieved independently, and ultimately constructs the final prompt
containing the original question, the complete set of retrieved doc-
uments, follow-up questions, and intermediate answers.

RAGtifier: Evaluating RAG Generation Approaches of State-of-the-Art RAG Systems for the SIGIR LiveRAG Competition SIGIR, July 17, 2025, Padua, IT
All RAG generation approaches, with the exception of IterDRAG
due to its inherent complexity, utilize the inverted context ordering
proposed by Cuconasu et al. [ 10]. In this ordering, the retrieved or
reranked documents are arranged in descending order of relevance,
with the highest-ranked document placed immediately before the
question.
3.5 Evaluation
Gemma-3-27B [ 2] serves as the primary evaluation model in our
RAG Challenge pipeline. Considering the expensive use of the best-
performing models, which are also the best-performing judges [ 11],
we searched for the smallest, yet best-performing LLM on the Chat-
bot Arena [ 1]. This approach also aligns with the recommendation
against using the same LLM for both answer generation and evalua-
tion [ 31]. Our selection is therefore based on its competitive Chatbot
Arena Elo score of 1341, which ranks it as the smallest among the
top open models. While acknowledging the potential drawbacks
associated with this specific model, given the complexities of inves-
tigating biases and unexpected weaknesses in LLMs-as-Judges [ 31],
we proceeded with Gemma-3-27B. Additionally, we evaluate the
candidate systems using Claude-3.5-Haiku [ 5], as this model family
is utilized for the final evaluation in the LiveRAG Challenge.
Furthermore, we consider different proposed evaluation prompt-
ing techniques to investigate influencing factors on LLMs-as-Judges
evaluations. The first evaluation prompt (simple comparison) com-
pares only the Falcon-3-10B generated answer with DataMorgana’s
generated ground-truth answer [ 35]. The prompt directly instructs
the LLM to decide which answer is better or if it’s a tie after pro-
viding a brief explanation of each answer’s most important aspects.
The second evaluation prompt is derived from CRAG [ 30], which
employs several metrics to define a good answer, such as concise-
ness, correctness, and support from retrieved documents. The third
evaluation approach extends these methods by employing two dis-
tinct metrics, a correctness score and a faithfulness score, with
a range of 4 and 3 possible values per metric, respectively. This
evaluation prompt (Figure 2), hereafter referred to as the LiveRAG
prompt, combines the evaluation strategy specified by the LiveRAG
Challenge [3] with generated and ground truth information.
4 EXPERIMENTS
Now, we discuss the results for the components we considered,
starting with insights from the use of DataMorgana and LLMs-as-
Judges. We then move on to a general investigation of retriever
performance and retrieval-influenced reranking, and finally de-
scribe the results of the cross-evaluation that led to our LiveRAG
solution.
4.1 DataMorgana Question Generation
The question generation process with DataMorgana yields an over-
all solid alignment of questions, documents, and answers. During
question generation, we observed that user and question categories
are more likely to influence the generated question as desired if
the category description emphasizes how it should behave rather
than what it should represent . For instance, a categorization of a
Spywith a behavioral description like hides his true intentions by
omitting information, giving misleading details, or communicatingin encrypted form yields more expected results compared to repre-
sentational description like secretly gathers information, often for a
government or organization, typically about enemies or competitors.
Spies use covert methods to obtain intelligence .
4.2 Evaluating Prompts and Judges
First, we investigate the impact of three different evaluation prompts.
We select a small subset of 100DataMorgana-generated QA- pairs to
cross-evaluate the influence of these prompts on the overall evalua-
tion metric. We use Falcon-3-10B to generate answers by providing:
1. query only, 2. golden document and query, and 3. OpenSearch@k
and query, where 𝑘∈{1,5,10,20,50}. Subsequently, we evaluate all
prompts using Gemma-3-27B with the generated answer and any
additional necessary prompt information. At this phase, we use the
OpenSearch retriever due to its superior performance on DataMor-
gana queries with fewer than 50 retrieved documents, which aligns
with Falcon-3-10B’s limited context window of up to 50 retrieved
passages.
In general, the answers generated using the query and gold docu-
ments achieved the highest performance across all prompt styles. In
the simple comparison prompt, answers generated with retrieved
documents were penalized, as scores decreased as retrieval@k in-
creased. While only specifying the query yielded the second-best
performance, surpassed only by the inclusion of the golden docu-
ment. The CRAG prompt favors retrieval@k with 𝑘∈{5,10}over
𝑘∈{20,50}and penalizes answers generated with only the query.
The LiveRAG prompt (Figure 2) favors an increase in retrieval@k
for the correctness metric. Compared to the simple comparison
and the CRAG prompt, query-only answers are scored almost as
correctly as the best-performing answer generation strategies, but
these answers are penalized with the lowest faithfulness scores
among all generation settings.
Since these different prompts provide comparable results with
their biased characteristics, we choose the LiveRAG prompt because
of its detailed assessment of correctness and faithfulness and its
similarity to the LiveRAG Challenge evaluation methodology.
Investigating the performance of Gamma-3-as-a-Judge, we com-
pare a subset of questions judged by Gemma-3-27B against the
judgments of Claude-3.5-Haiku. To do this, we selected samples
that were rated as poor, fair, and good by Gemma-3-27B (using the
LiveRAG prompt) and re-evaluated them with Claude-3.5-Haiku.
As a result, the poor and good samples evaluated by Gemma-3-27B
yielded nearly identical correctness and faithfulness scores when
judged by Claude. For the mediocre samples, we notice a minor
shift towards lower scores from Claude.
4.3 Retriever Performance
Considering the time constraints of the LiveRAG Challenge and
the influence of golden documents in retrieval@k, we investigate
the runtime of the retrieval and the number of golden documents
returned at distinct retrieval@k values for the provided OpenSearch
and Pinecone indices. Figure 4 shows that the number of gold doc-
uments increases continuously with higher retrieval@k. Notably,
Pinecone outperforms OpenSearch at @𝑘=20for multi-hop ques-
tions and at @𝑘=50for single-hop questions. Runtime measure-
ments (Figure 5) reveal that Pinecone is faster for all retrieval@k at

SIGIR, July 17, 2025, Padua, IT Trovato et al.
𝑘=[1.600]. OpenSearch takes 0.12𝑠and Pinecone 0.15𝑠at@𝑘=1,
scaling nearly linearly up to 𝑘=600, where OpenSearch takes 0.9𝑠
and Pinecone 0.58𝑠. We decide to proceed with Pinecone due to its
performance after reaching Falcon-3-10B’s context limit of about
50retrieved documents. Consequently, we check the performance
increase by using a reranker in combination with retriever@ 𝑘>50.
4.4 Reranker Performance
Evaluating the impact on the performance of BGE reranker, we
measure the runtime affected by retrieval@k and reranker@k and
report the percentage of remaining golden documents. We consider
retrieval@k for 𝑘∈[1,300]and reranker@k for 𝑘∈{1,3,5,10,20},
allowing Falcon-3-10B with its limited context length to be used
for all generation tasks. We cross-evaluate various retrieval and
reranker settings to find configurations that perform better than
retrieval@k alone in terms of the number of retrieved golden doc-
uments for single- and multi-hop questions within a reasonable
runtime.
The runtime of BGE (Figure 5) increases with the number of re-
trieved documents. Further experiments reveal that BGE takes ∼
11.2𝑠to rerank 400 retrieved documents. Considering the LiveRAG
time constraints and available computational resources, we limit
further experiments to 300 retrieved documents, which takes ∼8.6𝑠
per reranking operation per question. If we increase k up to the
context limit of 𝑘=50(Figure 4), the percentage of returned gold
documents increases due to the query alone. We hypothesize an
increasing RAG performance due to more golden documents are
present in the context when a higher retrieval@k is used in com-
bination with a reranker set to 𝑘≤20. Therefore, we searched
within the subset of viable retriever and reranker settings for a
configuration that outperforms retrieval@50 in terms of gold doc-
uments@k. For single-hop questions using Pinecone@300 with
BGE@10 and OpenSearch and Pinecone@[100, 300] with BGE@20,
more gold documents remaining in the reranked set compared to
using OpenSearch or Pinecone@50 alone. Similarly, for multi-hop
questions, this occurs for OpenSearch and Pinecone@[100, 300]
with BGE@20.
4.5 Final System Performance
With our insights into retriever and reranker performance, we
cross-evaluate various settings. We use Pinecone@ {100,200,300}
with BGE@{5,8,10,12}combined with optional inverted context
order for each RAG solution. Due to the iterative context gener-
ation of IterDRAG, we omitted the context ordering and tested
Pinecone@{100,200}with BGE@{5,10}for the initial retrieval
step. For additional retrieval steps in IterDRAG, we considered
Pinecone@ 200with BGE@{4,5}for a maximum of four and five
iterations respectively, and BGE@3 for six iterations.
Considering the LiveRAG constraints, increasing retrieval@k for
a fixed rerank@k does not consistently lead to better performance.
The performance differences due to a change in retrieval@k remain
mainly within a±2%range of the correctness evaluation metric
(Figure 2). Increasing rerank@k with fixed retrieval@k results in
a higher variation in the correctness score, where we measure
variations from 1%up to 25%. The influence of the inverted context
order averages to 1%performance increase.Table 1: Gemma-3-27B evaluation on 500 DataMorgana-
generated questions, equally distributed between single and
multi-hop questions. We report the LiveRAG prompt (Fig. 2)
metrics [%] for Correctness {1, 2} and Faithfulness {0,1}, dis-
carding other Correctness {-1, 0} and Faithfulness {-1} values.
Single-Hop Multi-Hop
Correctness Faithfulness Correctness Faithfulness
RAG { 1, 2 } { 0, 1 } { 1, 2 } { 0, 1 }
Simple Prompt 10.0 89.6 8.4 91.6 6.0 91.2 5.2 92.0
TrustRAG 15.3 80.9 14.8 80.9 14.7 83.3 17.3 80.7
Astute 17.0 77.1 15.7 78.3 35.5 62.0 36.6 60.4
InstructRAG 4.2 94.5 3.4 95.3 5.6 92.9 5.1 93.4
IterDRAG 3.4 94.5 4.2 93.6 3.8 93.6 3.8 93.6
The best-performing RAG approaches are listed in Figure 1, using
identical settings of Pinecone@200, BGE@5 and inverted context
order. IterDRAG uses Pinecone@200 and BGE@10 for initial re-
trieval, and Pinecone@200 with BGE@4 for up to 5 iterations. With
InstructRAG and IterDRAG perform comparably on Gemma-3-27B,
we select both as possible approaches for LiveRAG. During the
live challenge day, we generated answers with both RAG solutions
and evaluated the results using Gemma-3-27B, omitting the golden
document andgolden answer , supplemented by human evaluation.
Resulting in InstructRAG outperforms IterDRAG in terms of the
evaluation metric. IterDRAG achieved an average correctness of
1.70and a faithfulness score of 0.73, while InstructRAG achieved a
correctness of 1.91(+28.1%) and a faithfulness score of 0.93(+27.4%).
An additional manual comparison between these two approaches
revealed an occasionally subjectively better question-answer align-
ment for InstructRAG. Compared to our measurements, the orga-
nizer’s LLM evaluation returned lower scores for our submitted
InstructRAG-based approach: correctness of 1.13(−40.9%) and faith-
fulness of 0.55(−40.9%). This difference was likely influenced by
using a more capable judge LLM and accessing the golden document
andgolden answer .
5 CONCLUSION AND FUTURE WORK
For the LiveRAG Challenge, we investigated the performance pa-
rameters defined by the organizers for the evaluation, alongside
variations of RAG components. We evaluated several RAG solutions,
including AstuteRAG, IterDRAG, TrustRAG, InstructRAG, and sim-
ple prompting, using subsets of single- and multi-hop questions
generated with DataMorgana. Our investigation included the per-
formance of BGE and Rank-R1 rerankers, metrics for OpenSearch
and Pinecone retrievers, and evaluations using Gemma-3-27B and
Claude-3.5-Haiku as judge LLMs. On the live challenge day, we ran
our two best-performing RAG solutions in parallel, evaluating them
with Claude-3.5-Haiku and manual assessment. Our final submitted
solution uses InstructRAG with an inverted context order, employ-
ing Pinecone@200 with BGE@5 for context retrieval and reranking.
In the future, we plan to explore efficient and SotA RAG approaches
to enhance performance on diverse QA datasets, starting with our
DataMorgana-generated questions.

RAGtifier: Evaluating RAG Generation Approaches of State-of-the-Art RAG Systems for the SIGIR LiveRAG Competition SIGIR, July 17, 2025, Padua, IT
REFERENCES
[1][n. d.]. Chatbot Arena — openlm.ai. https://openlm.ai/chatbot-arena/. [Accessed
10-04-2025].
[2][n. d.]. Gemma open models | Google AI for Developers — ai.google.dev. https:
//ai.google.dev/gemma. [Accessed 10-04-2025].
[3][n. d.]. LiveRAG Challenge - Challengedetails — liverag.tii.ae. https://liverag.tii.
ae/challenge-details.php. [Accessed 10-04-2025].
[4]Falcon 3. [n. d.]. Falcon 3 — falconllm.tii.ae. https://falconllm.tii.ae/falcon3/index.
html. [Accessed 20-05-2025].
[5]Anthropic. 2024. Claude Haiku 3.5. https://www.anthropic.com/claude/haiku
Accessed: May 10, 2025.
[6]Tom B. Brown, Benjamin Mann, Nick Ryder, et al .2020. Language Models
are Few-Shot Learners. In Advances in Neural Information Processing Systems
33: Annual Conference on Neural Information Processing Systems 2020, NeurIPS
2020, December 6-12, 2020, virtual , Hugo Larochelle, Marc’Aurelio Ranzato,
Raia Hadsell, et al .(Eds.). https://proceedings.neurips.cc/paper/2020/hash/
1457c0d6bfcb4967418bfb8ac142f64a-Abstract.html
[7]Jianlv Chen, Shitao Xiao, Peitian Zhang, Kun Luo, Defu Lian, and Zheng Liu.
2024. BGE M3-Embedding: Multi-Lingual, Multi-Functionality, Multi-Granularity
Text Embeddings Through Self-Knowledge Distillation. arXiv:2402.03216 [cs.CL]
[8]Cheng-Han Chiang and Hung-yi Lee. 2023. Can Large Language Models Be an Al-
ternative to Human Evaluations?. In Proceedings of the 61st Annual Meeting of the
Association for Computational Linguistics (Volume 1: Long Papers) , Anna Rogers,
Jordan Boyd-Graber, and Naoaki Okazaki (Eds.). Association for Computational
Linguistics, Toronto, Canada, 15607–15631. https://doi.org/10.18653/v1/2023.acl-
long.870
[9]Florin Cuconasu, Giovanni Trappolini, Federico Siciliano, et al .2024. The Power
of Noise: Redefining Retrieval for RAG Systems. In Proceedings of the 47th In-
ternational ACM SIGIR Conference on Research and Development in Information
Retrieval, SIGIR 2024, Washington DC, USA, July 14-18, 2024 , Grace Hui Yang,
Hongning Wang, Sam Han, et al .(Eds.). ACM, 719–729. https://doi.org/10.1145/
3626772.3657834
[10] Florin Cuconasu, Giovanni Trappolini, Federico Siciliano, et al .2024. The Power
of Noise: Redefining Retrieval for RAG Systems. In Proceedings of the 47th In-
ternational ACM SIGIR Conference on Research and Development in Information
Retrieval, SIGIR 2024, Washington DC, USA, July 14-18, 2024 . ACM, 719–729.
https://doi.org/10.1145/3626772.3657834
[11] Florian E. Dorner, Vivian Y. Nastl, and Moritz Hardt. 2024. Limits to scalable evalu-
ation at the frontier: LLM as Judge won’t beat twice the data. CoRR abs/2410.13341
(2024). https://doi.org/10.48550/ARXIV.2410.13341 arXiv:2410.13341
[12] Yann Dubois, Balázs Galambosi, Percy Liang, and Tatsunori B. Hashimoto. 2024.
Length-Controlled AlpacaEval: A Simple Way to Debias Automatic Evalua-
tors. CoRR abs/2404.04475 (2024). https://doi.org/10.48550/ARXIV.2404.04475
arXiv:2404.04475
[13] Shahul Es, Jithin James, Luis Espinosa-Anke, and Steven Schockaert.
2025. Ragas: Automated Evaluation of Retrieval Augmented Generation.
arXiv:2309.15217 [cs.CL] https://arxiv.org/abs/2309.15217
[14] Benjamin Feuer, Micah Goldblum, Teresa Datta, et al .2024. Style Out-
weighs Substance: Failure Modes of LLM Judges in Alignment Benchmark-
ing. CoRR abs/2409.15268 (2024). https://doi.org/10.48550/ARXIV.2409.15268
arXiv:2409.15268
[15] Simone Filice, Guy Horowitz, David Carmel, et al .2025. Generating Diverse
Q&A Benchmarks for RAG Evaluation with DataMorgana. CoRR abs/2501.12789
(2025). https://doi.org/10.48550/ARXIV.2501.12789 arXiv:2501.12789
[16] FineWeb. [n. d.]. HuggingFaceFW/fineweb ·Datasets at Hugging Face — hugging-
face.co. https://huggingface.co/datasets/HuggingFaceFW/fineweb. [Accessed
20-05-2025].
[17] Kailash A. Hambarde and Hugo Proença. 2023. Information Retrieval: Recent
Advances and Beyond. IEEE Access 11 (2023), 76581–76604. https://doi.org/10.
1109/ACCESS.2023.3295776
[18] Oz Huly, Idan Pogrebinsky, David Carmel, et al .2024. Old IR Methods Meet RAG.
InProceedings of the 47th International ACM SIGIR Conference on Research and
Development in Information Retrieval, SIGIR 2024, Washington DC, USA, July 14-18,
2024, Grace Hui Yang, Hongning Wang, Sam Han, et al .(Eds.). ACM, 2559–2563.
https://doi.org/10.1145/3626772.3657935
[19] Jaehun Jung, Faeze Brahman, and Yejin Choi. 2024. Trust or Escalate: LLM Judges
with Provable Guarantees for Human Agreement. CoRR abs/2407.18370 (2024).
https://doi.org/10.48550/ARXIV.2407.18370 arXiv:2407.18370
[20] Patrick Lewis, Ethan Perez, Aleksandra Piktus, et al .2020. Retrieval-Augmented
Generation for Knowledge-Intensive NLP Tasks. In Advances in Neural Infor-
mation Processing Systems 33: Annual Conference on Neural Information Process-
ing Systems 2020, NeurIPS 2020, December 6-12, 2020, virtual , Hugo Larochelle,
Marc’Aurelio Ranzato, Raia Hadsell, et al .(Eds.). https://proceedings.neurips.cc/
paper/2020/hash/6b493230205f780e1bc26945df7481e5-Abstract.html
[21] Alex Mallen, Akari Asai, Victor Zhong, et al .2023. When Not to Trust Language
Models: Investigating Effectiveness of Parametric and Non-Parametric Memories.
InProceedings of the 61st Annual Meeting of the Association for ComputationalLinguistics (Volume 1: Long Papers), ACL 2023, Toronto, Canada, July 9-14, 2023 .
Association for Computational Linguistics, 9802–9822. https://doi.org/10.18653/
V1/2023.ACL-LONG.546
[22] Pinecone. [n. d.]. The vector database to build knowledgeable AI | Pinecone —
pinecone.io. https://www.pinecone.io/. [Accessed 20-05-2025].
[23] Ronak Pradeep, Nandan Thakur, Shivani Upadhyay, Daniel Campos, Nick
Craswell, and Jimmy Lin. 2025. The Great Nugget Recall: Automating Fact Extrac-
tion and RAG Evaluation with Large Language Models. arXiv:2504.15068 [cs.IR]
https://arxiv.org/abs/2504.15068
[24] Ofir Press, Muru Zhang, Sewon Min, et al .2023. Measuring and Narrowing the
Compositionality Gap in Language Models. In Findings of the Association for
Computational Linguistics: EMNLP 2023, Singapore, December 6-10, 2023 , Houda
Bouamor, Juan Pino, and Kalika Bali (Eds.). Association for Computational Lin-
guistics, 5687–5711. https://doi.org/10.18653/V1/2023.FINDINGS-EMNLP.378
[25] OpenSearch Project. [n. d.]. OpenSearch — opensearch.org. https://opensearch.
org/. [Accessed 20-05-2025].
[26] Stephen E. Robertson and Hugo Zaragoza. 2009. The Probabilistic Relevance
Framework: BM25 and Beyond. Found. Trends Inf. Retr. 3, 4 (2009), 333–389.
https://doi.org/10.1561/1500000019
[27] Venktesh V, Mandeep Rathee, and Avishek Anand. 2025. SUNAR: Se-
mantic Uncertainty based Neighborhood Aware Retrieval for Complex QA.
CoRR abs/2503.17990 (2025). https://doi.org/10.48550/ARXIV.2503.17990
arXiv:2503.17990
[28] Fei Wang, Xingchen Wan, Ruoxi Sun, et al .2024. Astute RAG: Overcoming
Imperfect Retrieval Augmentation and Knowledge Conflicts for Large Language
Models. CoRR abs/2410.07176 (2024). https://doi.org/10.48550/ARXIV.2410.07176
arXiv:2410.07176
[29] Zhepei Wei, Wei-Lin Chen, and Yu Meng. 2025. InstructRAG: Instructing
Retrieval-Augmented Generation via Self-Synthesized Rationales. In The Thir-
teenth International Conference on Learning Representations, ICLR 2025, Singa-
pore, April 24-28, 2025 . OpenReview.net. https://openreview.net/forum?id=
P1qhkp8gQT
[30] Xiao Yang, Kai Sun, Hao Xin, et al .2024. CRAG - Comprehensive RAG
Benchmark. In Advances in Neural Information Processing Systems 38: An-
nual Conference on Neural Information Processing Systems 2024, NeurIPS
2024, Vancouver, BC, Canada, December 10 - 15, 2024 , Amir Globersons,
Lester Mackey, Danielle Belgrave, et al .(Eds.). http://papers.nips.cc/
paper_files/paper/2024/hash/1435d2d0fca85a84d83ddcb754f58c29-Abstract-
Datasets_and_Benchmarks_Track.html
[31] Jiayi Ye, Yanbo Wang, Yue Huang, et al .2024. Justice or Prejudice? Quantifying
Biases in LLM-as-a-Judge. CoRR abs/2410.02736 (2024). https://doi.org/10.48550/
ARXIV.2410.02736 arXiv:2410.02736
[32] Zhenrui Yue, Honglei Zhuang, Aijun Bai, et al .2025. Inference Scaling for
Long-Context Retrieval Augmented Generation. In The Thirteenth International
Conference on Learning Representations, ICLR 2025, Singapore, April 24-28, 2025 .
OpenReview.net. https://openreview.net/forum?id=FSjIrOm1vz
[33] Hamed Zamani and Michael Bendersky. 2024. Stochastic RAG: End-to-End
Retrieval-Augmented Generation through Expected Utility Maximization. In
Proceedings of the 47th International ACM SIGIR Conference on Research and
Development in Information Retrieval, SIGIR 2024, Washington DC, USA, July 14-
18, 2024 , Grace Hui Yang, Hongning Wang, Sam Han, et al .(Eds.). ACM, 2641–2646.
https://doi.org/10.1145/3626772.3657923
[34] Zihan Zhang, Meng Fang, and Ling Chen. 2024. RetrievalQA: Assessing Adap-
tive Retrieval-Augmented Generation for Short-form Open-Domain Question
Answering. In Findings of the Association for Computational Linguistics, ACL 2024,
Bangkok, Thailand and virtual meeting, August 11-16, 2024 . Association for Com-
putational Linguistics, 6963–6975. https://doi.org/10.18653/V1/2024.FINDINGS-
ACL.415
[35] Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, et al .2023. Judging
LLM-as-a-Judge with MT-Bench and Chatbot Arena. In Advances in Neu-
ral Information Processing Systems 36: Annual Conference on Neural In-
formation Processing Systems 2023, NeurIPS 2023, New Orleans, LA, USA,
December 10 - 16, 2023 , Alice Oh, Tristan Naumann, Amir Glober-
son, et al .(Eds.). http://papers.nips.cc/paper_files/paper/2023/hash/
91f18a1287b398d378ef22505bf41832-Abstract-Datasets_and_Benchmarks.html
[36] Ming Zhong, Aston Zhang, Xuewei Wang, et al .2024. Law of the Weakest
Link: Cross Capabilities of Large Language Models. CoRR abs/2409.19951 (2024).
https://doi.org/10.48550/ARXIV.2409.19951 arXiv:2409.19951
[37] Huichi Zhou, Kin-Hei Lee, Zhonghao Zhan, et al .2025. TrustRAG: Enhancing
Robustness and Trustworthiness in RAG. CoRR abs/2501.00879 (2025). https:
//doi.org/10.48550/ARXIV.2501.00879 arXiv:2501.00879
[38] Lianghui Zhu, Xinggang Wang, and Xinlong Wang. 2023. JudgeLM: Fine-tuned
Large Language Models are Scalable Judges. CoRR abs/2310.17631 (2023). https:
//doi.org/10.48550/ARXIV.2310.17631 arXiv:2310.17631
[39] Shengyao Zhuang, Xueguang Ma, Bevan Koopman, et al .2025. Rank-R1: En-
hancing Reasoning in LLM-based Document Rerankers via Reinforcement Learn-
ing. CoRR abs/2503.06034 (2025). https://doi.org/10.48550/ARXIV.2503.06034
arXiv:2503.06034

SIGIR, July 17, 2025, Padua, IT Trovato et al.
[40] Shengyao Zhuang, Honglei Zhuang, Bevan Koopman, and Guido Zuccon. 2024.
A Setwise Approach for Effective and Highly Efficient Zero-shot Ranking with
Large Language Models. In Proceedings of the 47th International ACM SIGIR
Conference on Research and Development in Information Retrieval, SIGIR 2024,
Washington DC, USA, July 14-18, 2024 , Grace Hui Yang, Hongning Wang, Sam
Han, et al. (Eds.). ACM, 38–47. https://doi.org/10.1145/3626772.3657813
A APPENDIX
Figure 2: Evaluation prompt based on LiveRAG evaluation
guidelines used for Gemma-3-27B and Claude-3.5-Haiku.
System Prompt
You are an expert evaluator assessing the quality of an answer to a
given question based on retrieved passages. Your evaluation must
measure ‘correctness‘ and ‘faithfulness‘ according to predefined
criteria.
Query Prompt
Please evaluate the **generated answer** based on these specific
metrics:
1. Correctness. Combines elements of:
- **coverage**: portion of vital information, in the ground truth
answer which is covered by the generated answer.
- **relevance**: portion of the generated response which is directly
addressing the question, regardless its factual correctness.
Graded on a continuous scale with the following representative
points:
- **2:** Correct and relevant (no irrelevant information)
- **1:** Correct but contains irrelevant information
- **0:** No answer provided (abstention)
- **-1:** Incorrect answer
2. Faithfulness. Assesses whether the response is **grounded in
the retrieved passages**. Graded on a continuous scale with the
following representative points:
- **1:** Full support. All answer parts are grounded
- **0:** Partial support. Not all answer parts are grounded
- **-1:** No support. All answer parts are not grounded
The question that was asked:
{question}
Ground truth answer:
{gold_answer}
Ground truth passage from a document:
{gold_document}
Generated answer (to be evaluated):
{generated_answer}
Retrieved passages:
{retrieved_passages}
Return ONLY a JSON object with your evaluation scores. Do not
repeat the question, answer or any other text. The JSON object should
contain the following
{{"correctness": integer, "faithfulness": integer}}Figure 3: Subset of question and user categories we used to
generate QA-pairs with DataMorgana.
Question: quantitative-nature
- Question that doesn’t involve numbers, statistics, or mathematical
reasoning.
- Question involving simple numerical facts or basic arithmetic
(dates, measurements, counts).
- Question requiring statistical analysis, probability assessment, or
mathematical reasoning beyond basic arithmetic.
Question: verification-difficulty
Facts that can be quickly checked against reliable, accessible sources.
Information requiring specialized sources or moderate effort to
verify accurately.
Information that’s challenging to verify due to conflicting sources,
limited documentation, or inherent ambiguity.
User: decision-making-style
User who makes decisions based on logical analysis of facts and
evidence.
User who relies heavily on intuition and gut feelings when making
decisions.
User who prioritizes gathering multiple perspectives before making
decisions.
User who primarily draws on personal or historical experience to
make decisions.
User: trust-orientation
- User who questions information extensively and requires strong
evidence before accepting claims.
- User who verifies important information but doesn’t question
everything.
- User who generally trusts information from seemingly authoritative
sources.
- User who rarely questions the validity of information they
encounter.
User: behavior-patterns
A spy usually hides his true intentions by omitting information,
giving misleading details or communicating in encrypted form.
A person who believes that this world is secretly controlled by
reptilian aliens, often depicted as shape-shifting lizard-like beings.
This belief is part of a conspiracy theory that suggests these beings
manipulate human affairs and are involved in various global events.
A member of a diverse group of reptiles that first appeared during
the Triassic period, over 230 million years ago, and dominated the
Earth for over 160 million years.
Bishop of Rome and the spiritual leader of the worldwide Roman
Catholic Church, he always refers to a god and is a religious figure.
Cites biblical texts and religious doctrines in discussions, often
emphasizing the importance of faith, morality, and the teachings of
Jesus Christ.
A person who claims to have traveled through time, either from
the future or the past. They might offer vague predictions or
anachronistic knowledge, often with a sense of urgency or a warning
about future events.

RAGtifier: Evaluating RAG Generation Approaches of State-of-the-Art RAG Systems for the SIGIR LiveRAG Competition SIGIR, July 17, 2025, Padua, IT
Figure 4: Pinecone excels OpenSearch in retrieving gold doc-
uments [%], for multi-hop @k=20 and @k=50 for single-hop
questions. Considering multi-hop questions, both documents
used for QA generation must be retrieved.
Figure 5: OpenSearch and Pinecone retrievers, and BGE
reranker scale linearly in runtime [s] for 𝑘∈ [1,600]for
our generated DataMorgana QA dataset.
