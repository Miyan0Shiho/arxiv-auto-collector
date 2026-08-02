# MRCoder: An Efficient Context Selecting Approach for Repository-Level Code Generation

**Authors**: Peiding Wang, Li Zhang, Fang Liu

**Published**: 2026-07-29 11:47:51

**PDF URL**: [https://arxiv.org/pdf/2607.26805v1](https://arxiv.org/pdf/2607.26805v1)

## Abstract
Large language models (LLMs) have demonstrated strong capabilities in code generation. However, repository-level code generation remains challenging, as it requires effectively identifying and utilizing repository-specific context. While retrieval-augmented generation (RAG) incorporates relevant code snippets, it often introduces redundant context that interferes with the LLM's ability to utilize relevant information, leading to degraded generation quality and increased computational cost. Moreover, existing context selection and compression methods struggle to balance efficiency and quality, either introducing additional computational overhead or failing to effectively select valid context. In this paper, we propose MRCoder, an efficient context selection framework that improves both the effectiveness and efficiency of repository-level code generation. MRCoder adopts a Map-Reduce paradigm: in the Map Phase, a lightweight draft model generates drafts over partitioned contexts, and Structure-Aware Draft-Guided Selection (SADGS) selects informative contexts based on drafts through API consistency and logical similarity; in the Reduce Phase, the refined contexts are aggregated for final generation, with a parallel verification strategy further accelerating decoding. We evaluate MRCoder on two widely used repository-level code generation benchmarks, CoderEval and DevEval, using Qwen2.5-Coder and DeepSeek-Coder as backbone LLMs. Experimental results show that MRCoder improves code generation accuracy over strong baselines while reducing token consumption by 30 to 50% and inference time by up to 52%. These results demonstrate that our proposed structured and draft-guided context selection strategy is crucial for improving both the quality and efficiency of repository-level code generation

## Full Text


<!-- PDF content starts -->

MRCoder: An Efficient Context Selecting Approach for Repository-Level Code
Generation
PEIDING WANG,Beihang University, China
LI ZHANG,Beihang University, China
FANG LIU∗,Beihang University, China
Large language models (LLMs) have demonstrated strong capabilities in code generation. However, repository-level code generation
remains challenging, as it requires effectively identifying and utilizing repository-specific context. While retrieval-augmented genera-
tion (RAG) incorporates relevant code snippets, it often introduces redundant context that interferes with the LLM’s ability to utilize
relevant information, leading to degraded generation quality and increased computational cost. Moreover, existing context selection
and compression methods struggle to balance efficiency and quality, either introducing additional computational overhead or failing
to effectively select valid context. In this paper, we proposeMRCoder, an efficient context selection framework that improves both
the effectiveness and efficiency of repository-level code generation.MRCoderadopts a Map–Reduce paradigm: in the Map Phase,
a lightweight draft model generates drafts over partitioned contexts, and Structure-Aware Draft-Guided Selection (SADGS) selects
informative contexts based on drafts through API consistency and logical similarity; in theReduce Phase, the refined contexts are
aggregated for final generation, with a parallel verification strategy further accelerating decoding. We evaluateMRCoderon two
widely used repository-level code generation benchmarks, CoderEval and DevEval, using Qwen2.5-Coder and DeepSeek-Coder as
backbone LLMs. Experimental results show thatMRCoderimproves code generation accuracy over strong baselines while reducing
token consumption by 30%–50% and inference time by up to 52%. These results demonstrate that our proposed structured and
draft-guided context selection strategy is crucial for improving both the quality and efficiency of repository-level code generation. The
code and data are available at https://github.com/zhu-zhu-ding/MRCoder.
CCS Concepts:•Software and its engineering;•Computing methodologies→Artificial intelligence;
Additional Key Words and Phrases: Large Language Models, Context Selection, Repository-Level Code Generation, Efficient Inference
ACM Reference Format:
Peiding Wang, Li Zhang, and Fang Liu. 2018.MRCoder: An Efficient Context Selecting Approach for Repository-Level Code Generation.
InProceedings of Make sure to enter the correct conference title from your rights confirmation email (Conference acronym ’XX).ACM,
New York, NY, USA, 23 pages. https://doi.org/XXXXXXX.XXXXXXX
1 Introduction
Recent advances in large language models (LLMs) have fundamentally transformed the landscape of software de-
velopment [ 2,21,25], enabling intelligent systems to assist developers throughout the entire programming lifecycle.
∗Corresponding author.
Authors’ Contact Information: Peiding Wang, wangpeiding@buaa.edu.cn, Beihang University, School of Computer Science & Engineering, State Key
Laboratory of Complex & Critical Software Environment, Beijing, China; Li Zhang, lily@buaa.edu.cn, Beihang University, School of Computer Science &
Engineering, State Key Laboratory of Complex & Critical Software Environment, Beijing, China; Fang Liu, fangliu@buaa.edu.cn, Beihang University,
School of Computer Science & Engineering, State Key Laboratory of Complex & Critical Software Environment, Beijing, China.
Permission to make digital or hard copies of all or part of this work for personal or classroom use is granted without fee provided that copies are not
made or distributed for profit or commercial advantage and that copies bear this notice and the full citation on the first page. Copyrights for components
of this work owned by others than the author(s) must be honored. Abstracting with credit is permitted. To copy otherwise, or republish, to post on
servers or to redistribute to lists, requires prior specific permission and/or a fee. Request permissions from permissions@acm.org.
©2018 Copyright held by the owner/author(s). Publication rights licensed to ACM.
Manuscript submitted to ACM
Manuscript submitted to ACM 1
arXiv:2607.26805v1  [cs.SE]  29 Jul 2026

2 Wang et al.
3 5 7 10
T op-K101520253035404550Score (%)
Quality performance with different T op-K code blocks retrieved
Pass@1
Precision
Recall
3 5 7 10
T op-K4.64.85.05.25.45.65.86.0Time (s)
Time (s)
T okens (k)234567
T okens (k)
Efficiency performance with different T op-K code blocks retrieved
Fig. 1.Motivation Example. Results from Qwen2.5-Coder-7B-Instruct on CoderEval with BM25 retrieval. As Top- 𝐾increases, recall
improves, but redundant and noisy context degrades generation quality after an initial gain. Meanwhile, longer contexts increase
both token consumption and generation time, leading to reduced efficiency.
The emergence of powerful code-centric LLMs, such as Qwen-Coder [ 11] and DeepSeek-Coder [ 36], has substantially
enhanced the ability of AI systems to infer developer intent and generate high-quality code, thereby improving develop-
ment efficiency [ 26]. While early studies primarily focused on isolated function-level generation [ 3,6], recent research
has increasingly shifted toward the more realistic and challenging setting of repository-level code generation [ 8,31]. In
this scenario, models must leverage rich repository-level context, including cross-file dependencies, user-defined APIs,
and project-specific conventions, to produce functionally correct and contextually consistent code [15].
Retrieval-Augmented Generation (RAG) has become the dominant paradigm for incorporating repository-level
context into code generation [ 28,32]. By retrieving relevant code snippets—such as definitions, usage patterns, and
dependencies—and integrating them into the model input, RAG enhances contextual awareness and improves generation
accuracy. For instance, RepoCoder [32] adopts an iterative retrieval-generation framework that refines queries across
rounds, while RLCoder [ 28] leverages reinforcement learning to optimize the retriever for more effective context
selection. Despite these advances, existing RAG-based methods share a fundamental limitation:not all retrieved
code contexts are truly relevant to the generation process. In practice, retrieval inevitably introduces noisy or
weakly relevant contexts, especially when the number of retrieved chunks is increased to improve recall [ 7,12,29]. The
presence of such noisy contexts leads to two major issues:
Issue 1: Degraded Generation Quality. Irrelevant or redundant contexts can distract the LLM and introduce
conflicting signals, causing incorrect reasoning, hallucinations, or off-target code generation. Moreover, noisy contexts
unnecessarily increase the input length to make LLMs struggle to effectively utilize all provided information, resulting
in diminished accuracy.
Issue 2: Reduced Generation Efficiency.Noisy contexts unnecessarily increase the input length, thereby raising
the computational cost of LLM inference and slowing down generation. In repository- or codebase-level code generation
scenarios, such increased latency can significantly hinder developer productivity, resulting in a poor user experience [ 35].
As illustrated in Figure 1, as the amount of effective context increases (i.e., recall improves), the generation quality
(Pass@1) initially improves but subsequently declines. At the same time, the steady drop in precision suggests that
redundant and noisy contexts increasingly interfere with the LLM’s ability to leverage truly relevant information. In
addition, these noisy contexts inflate the input length, leading to higher inference costs and longer generation latency.
Manuscript submitted to ACM

MRCoder: An Efficient Context Selecting Approach for Repository-Level Code Generation 3
Together, these observations highlight a key challenge in repository-level code generation: effectively identifying and
selecting useful context while filtering out.
Context selection and compression have been explored as potential solutions. RepoFormer [ 29] and CodeFilter [ 16]
trains a small LLM to output control tokens for deciding whether to retrieve or select relevant contexts; however, it does
not fundamentally mitigate the negative effects of retrieval noise (Issues 1 and 2). In contrast, LongCodeZip [ 22] reduces
context length by selecting informative code lines based on perplexity computed by a small LLM, but requires multiple
forward passes, leading to significant time overhead and limiting its applicability in latency-sensitive scenarios (Issue 2).
To address the above challenges, we proposeMRCoder, an efficient context selection framework for repository-level
code generation.MRCoderis designed to identify and retain code contexts that are truly beneficial for generation from
a large pool of initially retrieved candidates. The framework consists of two main phases:
Phase I: Map Phase.We first partition the retrieved code blocks into multiple context groups and combine each
group with the target code query to construct a set of prompt sequences. These prompts are then fed into a smaller LLM
to generate code drafts in parallel. Based on these drafts, we apply a selection strategy that extracts relevant contexts
from the candidate pool using signals such as API call consistency and semantic similarity. This process enables the
smaller LLM to evaluate context usefulness under moderate input lengths, effectively filtering out noisy or irrelevant
code fragments.
Phase II: Reduce Phase.In this phase, the filtered contexts produced in the Map phase are aggregated and provided
to a larger target LLM for final code generation. By focusing on high-quality and concise context, this step not only
improves generation accuracy but also reduces input length, allowing the model to better utilize the most relevant
information. In addition, we leverage the drafts generated during the Map phase to design a parallel verification
mechanism for the target LLM. By validating multiple draft candidates concurrently, our approach accelerates the
decoding process and reduces overall inference latency.
Overall,MRCoderimproves generation quality by selecting informative context fragments while simultaneously
enhancing efficiency through context compression and draft-based parallel verification.
To evaluate the effectiveness ofMRCoder, we conduct extensive experiments on two widely used repository-level
code generation benchmarks, CoderEval [ 31] and DevEval [ 15]. We compare our approach against representative base-
lines, including standard RAG, the advanced repository-level code generation method RL-Coder [ 28], and state-of-the-art
code context compression approaches RepoFormer [ 29] and LongCodeZip [ 22]. The evaluation is conducted across
advanced code LLM families, namely Qwen2.5-Coder and DeepSeek-Coder. Experimental results show thatMRCoder
consistently improves code generation accuracy while reducing computational cost. Specifically, it outperforms all
baseline methods in most settings, achieving up to52.7%relative improvement in Pass@1 compared to RAG and
31.3%compared to state-of-the-art baseline LongCodeZip, while maintaining more stable performance across different
retrieval blocks. In terms of efficiency,MRCoderreduces token consumption by30%–50%and decreases total inference
time by up to52.1%compared to RAG and47.6%compared to state-of-the-art baseline RepoFormer, achieving a better
balance between effectiveness and efficiency than existing methods.
Our contributions can be summarized as follows:
•We proposeMRCoder, a novel Map–Reduce framework for repository-level code generation that performs
context selection after retrieval and before generation, effectively improving the quality and efficiency of
generation.
Manuscript submitted to ACM

4 Wang et al.
•We introduce Structure-Aware Draft-Guided Selection (SADGS), which leverages API call relationships and
logical similarity between draft code and candidate contexts to identify informative context fragments and
improve code quality.
•We propose a draft-based efficiency optimization mechanism that leverages code drafts to select relevant contexts,
reducing input length, and employs parallel verification to improve generation efficiency.
•We conduct comprehensive experiments on two repository-level code generation benchmarks using different
backbone LLMs. The results demonstrate thatMRCoderoutperforms state-of-the-art context selection and
compression methods, achieving improvements in both generation quality and efficiency.
2 Related Work
Repository-Level Code Generation.Repository-level code generation has recently attracted increasing attention as
large language models (LLMs) are applied to realistic software engineering scenarios [ 8,10,15,31]. Unlike function-level
benchmarks such as HumanEval [ 6] and MBPP [ 3], repository-level tasks require LLMs to reason over long-range
dependencies, cross-file interactions, and project-specific APIs. A common paradigm for repository-level code generation
is retrieval-augmented generation (RAG) [ 28,32,33], where relevant code snippets are retrieved from the repository
and provided as additional context to the LLM. Early work such as RepoCoder [ 32] demonstrates that retrieving
cross-file contexts can significantly improve code completion performance by exposing the LLM to relevant APIs and
usage patterns. Similarly, GraphCoder [ 18] incorporate structural signals such as call graphs or dependency graphs to
enhance retrieval quality. RL-Coder [ 28] improves the quality of generated code by using reinforcement learning to
train a retriever to retrieve more relevant code. These approaches highlight the importance of grounding generation in
repository-specific knowledge. However, they typically rely on Top- 𝐾retrieval, which inevitably introduces substantial
noise particularly when Top- 𝐾is increased to improve retrieval recall [ 16,29], leading to degraded model performance
due to context overload.
Code Context Selection and Filtering.To select and retain the relevant code context, recent work has explored
both implicit and explicit strategies to address this challenge. Repoformer [ 29] proposes a selective retrieval framework
that enables the model to decide whether retrieval is necessary, thereby avoiding unnecessary or harmful context and
improving both efficiency and robustness. Similarly, CodeFilter [ 16] employs a likelihood-based metric to train the model
to assess the impact of retrieved context blocks on generation quality, thereby enabling the selection of beneficial code
context. Complementarily, DietCode [ 34] combines frequency-based filtering with CodeBERT [ 9] attention heuristics
to remove low-impact tokens and SlimCode [ 27] employs rule-based token pruning using token types and program
dependency graphs, which may generalize poorly across languages and tasks. However, both methods primarily focus on
single-function compression for short-context scenarios. To address this, LongCodeZip [ 22] introduces a code-specific
compression framework that performs hierarchical filtering at both function and block levels using perplexity-based
relevance scoring, achieving substantial context reduction while preserving task performance.
Despite these advances, existing methods struggle to balance generation quality and efficiency. While Repoformer
or CodeFilter use a lightweight model to generate a single decision token for retrieval control, it cannot effectively
mitigate noisy retrieval. In contrast, LongCodeZip relies on multiple rounds of forward passes to compute perplexity
for code selection, incurring substantial computational overhead.
Manuscript submitted to ACM

MRCoder: An Efficient Context Selecting Approach for Repository-Level Code Generation 5
3 Method
3.1 Task Definition
In this section, we formalize the task of repository-level code generation [ 15,31,35]. Given a code repository, we
focus exclusively on its source code and abstract it into a collection of code contexts obtained via static analysis tools.
Specifically, we parse the repository and decompose it into a set of semantically meaningful code units (functions and
classes). Each such unit is treated as an individual code context. Formally, we denote this collection as {𝐶1,...,𝐶𝑁},
where each𝐶 𝑖represents a code snippet corresponding to a function or class extracted from the repository.
Given a query 𝑄, which consists of a prompt with code to be generated, the objective is to let LLM generate the
target function 𝐹by leveraging the Top- 𝐾retrieved contexts{𝐶1,...,𝐶𝐾}, where the Top- 𝐾contexts correspond to the
𝐾code snippets with the highest similarity to the query. The generation process can be formulated as:
𝐹=LLM(𝑄,{𝐶 1,...,𝐶𝐾})(1)
In this work, our goal is to select contexts from the Top- 𝐾that are are effective for generation process, thereby reducing
irrelevant contexts and improving both the quality and efficiency of code generation.
3.2 Overview
In this section, we introduceMRCoder, a two-stage framework that selects effective contexts after retrieval and before
generation. As shown in Figure 2, given a query and a set of retrieved code contexts from repository,MRCoderfollows a
Map–Reduce paradigm. In theMap Phase, the retrieved contexts are partitioned into multiple groups, and a lightweight
draft model generates draft code for each group independently. These drafts are then used to evaluate the usefulness
of contexts viaStructure-Aware Draft-Guided Selection (SADGS)based on API consistency and semantic similarity,
producing a filtered set of high-quality contexts. In theReduce Phase, the selected contexts are aggregated and fed into
a larger target LLM for final code generation. Meanwhile, the intermediate drafts are further leveraged for parallel
verification to accelerate decoding.
3.3 Map Phase
3.3.1 Context Partition and Draft Generation.As described in Section 3.1, we first employ static analysis to extract a
set of code contexts from the repository. Specifically, we use tree-sitter1to parse each source file into functions and
classes code units. To mitigate potential data leakage, we filter out code contexts whose Jaccard Similarity [ 23] with
the standard answer exceeds 0.9 [ 5,13], thereby removing near-duplicate or highly similar code snippets. After this
preprocessing step, we obtain the final code contexts {𝐶1,...,𝐶𝐾}. As shown in Figure 2, following prior work [ 28,32],
given a code generation query𝑄, we first retrieve the Top-𝐾relevant code contexts from{𝐶 1,...,𝐶𝑁}:
{𝐶1,...,𝐶𝐾}=F retrieval(𝑄,{𝐶 1,...,𝐶𝑁})(2)
whereFretrieval(·)denotes the retrieval method. Notably, the retrieval component is not tightly coupled with any specific
approach;MRCoderdoes not rely on particular retrieval optimizations and can be seamlessly integrated with existing
retrieval methods.
Next, we partition the retrieved code contexts into multiple subgroups of size 𝑀, where𝑀is a predefined hyper-
parameter. As shown in Figure 2, the 𝑀is set to 2. Given the retrieved contexts {𝐶1,...,𝐶𝐾}, we first rank them in
1https://github.com/tree-sitter/tree-sitter
Manuscript submitted to ACM

6 Wang et al.
Query
Query
Q+C1C2
Q+C...C...
Q+Ck-1Ck
Draft LLMD1
D...
DnC1
C4C2
C5C3
C6
C...Ck-1Ck
Q DDraft CodeC Effective ContextCDiscarded Context
C2
Q+C...
Ck
Target LLMC2
D dtjdtj+1 ...
A tjtj+1 tm
 ...
Search
Speed Up
Chunking
1 forward：n tokens
dtm
tm
A tj tj+1 t...
1 forward：1 token
Effective ContextOurs BaseAPI Call 
Matching
Logic 
Similarity
token/sOurBase
 slow
fastG1
G...
GnA Final Code GGroupdtDraft Token
tFinal Token
Fig. 2. The overall pipeline ofMRCoder.
descending order according to their similarity to the query 𝑄. We then divide the ranked list into 𝑛consecutive groups
with a fixed stride of𝑀:
{𝐶1,...,𝐶𝐾}=𝑛Ä
𝑖=1𝐺𝑖,|𝐺𝑖|=𝑀(𝑖<𝑛),|𝐺 𝑛|≤𝑀(3)
where𝑛=𝐾
𝑀
, and each group 𝐺𝑖contains a contiguous segment of contexts from the ranked list. We also explored
dynamic partitioning strategies based on clustering methods (e.g., Random and K-Means [ 1]). However, these approaches
did not yield noticeable performance gains while introducing additional computational overhead (See Section 6.2).
Therefore, we adopt a simple and efficient fixed partitioning strategy.
Then we combine each context group 𝐺𝑖with the query 𝑄to construct independent prompts, which are processed
in parallel by a lightweight draft model to obtain draft code sequences {𝐷1,...,𝐷𝑛}. These drafts serve as intermediate
signals that reflect how each context group contributes to the generation, and are subsequently used to guide context
selection. Formally, for each group𝐺 𝑖, we define:
𝐷𝑖=LLM(𝑄,𝐺 𝑖), 𝑖=1,...,𝑛(4)
where𝐷𝑖is the draft code generated by the small draft LLM corresponding to 𝐺𝑖. This design ensures that each prompt
has a moderate context length, enabling the LLM to better utilize relevant information compared to directly processing
the full set of retrieved contexts. Moreover, the parallel generation of drafts significantly reduces inference latency in
theMap Phase.
3.3.2 Structure-Aware Draft-Guided Selection.In this section, we introduce Structure-Aware Draft-Guided Selection
(SADGS), a dual-perspective context selection framework that leverages structural signals in code. Specifically, given
Manuscript submitted to ACM

MRCoder: An Efficient Context Selecting Approach for Repository-Level Code Generation 7
the draft set{𝐷1,...,𝐷𝑛}generated in theMap Phaseand the retrieved contexts groups {𝐺1,...,𝐺𝑛}, SADGS selects a
subset of informative contexts 𝐺∗⊆𝐺 based on two complementary signals. The design of SADGS is motivated by two
key observations in repository-level code generation: (1)API Call Perspective:target implementations often rely
on invoking repository-defined APIs, and (2)Logic Perspective:similar logic patterns in the repository can provide
valuable references.
API Call Perspective.For this perspective, our goal is to select valid contexts based on the API calls between the
generated draft code and code contexts. We employ the static analysis tool tree-sitter to extract API call information
from both the draft code and the retrieved contexts. In this paper, we categorize APIs into two types based on call
relationships:
•External APIs: function or class calls that are invoked within the current code unit but defined outside of
it, representing outgoing dependencies. These APIs reflect how the current code relies on external callable
interfaces.
•Internal APIs: functions or classes defined within the current code unit that can be invoked by other code units,
representing exposed callable interfaces. These APIs characterize the reusable interfaces provided by the code
unit itself.
For example, consider a class 𝐴with a method 𝐵, where𝐵calls a function 𝐶that is defined outside of class 𝐴. In this
case,𝐶is treated as an external API of 𝐵, as it is defined outside the method but invoked within it. Conversely, 𝐵is
regarded as an internal API of class𝐴, since it is defined within the class and can be called by other code units.
For each draft 𝐷𝑖, we identify its external API call set A(𝐷𝑖)ext. This API sequence contains the necessary external
API information from code context required to complete the draft code. For each code context 𝐶∈𝐺𝑖, we extract both
its external API calls and internal API calls A(𝐶𝑖). Internal API calls within the code context directly provide callable
external information for the generated code. The reason why we retain external API calls in the context, as identical
external API usage is that it can serve as reference patterns to guide the generation of code using this API. Based on
above, we select context blocks that share overlapping API calls with the draft. Formally, the selected context subset is
then defined as:
𝐺𝑖𝐴𝑃𝐼={𝐶𝑡∈𝐺𝑖|A(𝐶𝑡)∩A(𝐷𝑖)ext≠∅},𝑡≤𝑀(5)
This mechanism preserves both repository-specific APIs that the target implementation may invoke and reference
code patterns for external API usage, thereby providing efficient and informative context for subsequent LLM-based
generation from the API perspective.
Logic Similarity Perspective.To further capture logical consistency between the draft and candidate contexts, we
perform similarity-based selection. The goal of this step is to identify code contexts that are most semantically and
structurally aligned with the draft implementation. Specifically, for each draft 𝐷𝑖and each context 𝐶𝑡∈𝐺𝑖, we compute
their relevance score using the BM25 [20] similarity function:
𝑆sim(𝐷𝑖,𝐶𝑡)=BM25(𝐷 𝑖,𝐶𝑡)(6)
where BM25(·,·)measures the logic and structural similarity between the draft code and the candidate context.
Based on the computed similarity scores, we rank all contexts 𝐶𝑡∈𝐺𝑖in descending order according to 𝑆sim(𝐷𝑖,𝐶𝑡),
and retain the Top-𝐿(𝐿<𝑀) most relevant ones:
𝐺SIM
𝑖=(
𝐶′
𝑡∈𝐿arg top
𝐶𝑡∈𝐺𝑖𝑆sim(𝐷𝑖,𝐶𝑡))
(7)
Manuscript submitted to ACM

8 Wang et al.
This mechanism preserves code contexts that exhibit similar implementation logic to the draft, thereby providing the
subsequent LLM with concise and informative context from a logical perspective, complementary to the API-based
selection.
Finally, we merge𝐺API
𝑖and𝐺SIM
𝑖and remove duplicate elements to obtain the final informative contexts𝐺∗:
𝐺∗
𝑖=𝐺API
𝑖∪𝐺SIM
𝑖.(8)
3.4 Reduce Phase
Although theMap Phaseeffectively filters out irrelevant context, the target LLM is still required to generate the final
code in a token-by-token, autoregressive manner based on the refined context, which inherently limits generation
efficiency. Moreover, the incorporation of draft models for parallel generation inMap Phaseintroduces additional
computational overhead. Therefore, in this phase, our primary objective is toimprove the efficiency of final code
generation.
3.4.1 Preliminaries.Autoregressive decoding [24]is a prevalent paradigm in LLM inference, where output tokens
are generated sequentially through iterative forward predictions. Formally, given an LLM LLM, a prefix token sequence
(𝑡1,...,𝑡𝑛−1), and the current input token𝑡 𝑛, the model predicts the probability distribution over the next token as:
𝒑𝑛+1=LLM(𝑡𝑛|𝑡1,...,𝑡𝑛−1)(9)
where 𝒑𝑛+1denotes the probability distribution for the next token. The next token 𝑡𝑛+1is then obtained either by
selecting the highest probability token from 𝒑𝑛+1or by sampling according to the distribution. The selected token
is subsequently appended to the sequence and fed back into the model to generate the next token. Inherently, this
token-by-token nature of autoregressive decoding introduces computational inefficiencies, resulting in inference latency
that scales linearly with both the generated sequence length and the model complexity.
Parallel decoding [14]aims to accelerate the autoregressive generation process of LLMs. While LLMs inherently
generate one token per forward pass, their inference efficiency can be significantly improved by verifying and accepting
multiple tokens in parallel. The target LLM accelerates the generation process by performing parallel validation and
accepting token sequences from the draft sequence that match the final answer it has generated. Given a sequence of
draft tokens generated by the small draft LLM inMap Phrase, the target LLM generating the final answer can evaluate
the entire sequence in a single forward and determine which tokens are consistent with its own predicted distributions.
This mechanism allows the LLM to accept multiple tokens simultaneously, thereby reducing the number of required
forward passes and accelerating the overall generation process.
Formally, given a sequence of 𝑚draft tokens(𝑑𝑡1,···,𝑑𝑡𝑚), the target model is conditioned on the prefix (𝑡1,...,𝑡𝑛).
𝒑1,···,𝒑𝑚=LLM target(𝑑𝑡1,···,𝑑𝑡𝑚|𝑡1,...,𝑡𝑛),(10)
where 𝒑𝑖denotes the predicted probability distribution for the 𝑖-th draft token position. Each draft token 𝑑𝑡𝑖is then
verified against the corresponding distribution𝒑 𝑖. The generation token can be defined as:
 
𝑑𝑡𝑖,if𝑑𝑡 𝑖=𝑑𝑒𝑐𝑜𝑑𝑒(𝒑 𝑖)
𝑑𝑒𝑐𝑜𝑑𝑒(𝒑 𝑖),if𝑑𝑡𝑖≠𝑑𝑒𝑐𝑜𝑑𝑒(𝒑 𝑖)(11)
A draft token 𝑑𝑡𝑖is defined asacceptedif it matches decode(𝒑 𝑖), and is otherwiserejected. If a draft token 𝑑𝑡𝑗is
rejected at position 𝑗(where𝑗<𝑚 ), the verification process terminates immediately, and the token 𝑡𝑗is instead sampled
Manuscript submitted to ACM

MRCoder: An Efficient Context Selecting Approach for Repository-Level Code Generation 9
Algorithm 1:Efficient Parallel Decoding for Reduce
Input:Query𝑄, refined context𝐺∗, draft code tokens𝑑𝑡 1,,...,𝑑𝑡 𝑚
Output:Generated token sequence𝑇
Initialize𝑇←∅;
Set pointer𝑗←1;
while len(𝑇) <MAX_TOKENSdo
if𝑗≤𝑚then
// Parallel verification
𝒑𝑗,...,𝒑 𝑚←LLM target(𝑑𝑡𝑗,...,𝑑𝑡 𝑚|𝑞,𝐺∗);
Find the longest accepted prefix:
𝑘=max
𝑧𝑑𝑡𝑗+𝑖−1=𝑑𝑒𝑐𝑜𝑑𝑒(𝒑 𝑗+𝑖−1),∀𝑖∈[1,𝑧]	
;
Accept(𝑑𝑡 𝑗,...,𝑑𝑡 𝑗+𝑘−1);
𝑇←𝑇⊕(𝑑𝑡 𝑗,...,𝑑𝑡 𝑗+𝑘−1);
𝑗←𝑗+𝑘;
𝑡=𝑑𝑒𝑐𝑜𝑑𝑒(𝒑 𝑗);
𝑇←𝑇⊕𝑡;
// Span alignment with remaining draft
Find𝑠such that:
(𝑡1,...,𝑡 𝑥)=(𝑑𝑡 𝑠,...,𝑑𝑡 𝑠+𝑥−1), 𝑠∈[𝑗+1,𝑚−𝑥+1];
ifsuch𝑠existsthen
𝑗←𝑠+𝑥;
else
// Continue autoregressive decoding
𝑗←𝑚+1;
else
// Autoregressive decoding
𝒑←LLM target(·|𝑄,𝐺∗,𝑇);
𝑡=𝑑𝑒𝑐𝑜𝑑𝑒(𝒑);
𝑇←𝑇⊕𝑡;
iflasttoken in𝑇isEOSthen
break;
return𝑇;
from the corresponding probability distribution 𝒑𝑗. No further draft tokens are validated beyond this point; If all 𝑚
draft tokens are accepted, we set 𝑗=𝑚+ 1and sample the next token 𝑡𝑗from the distribution 𝒑𝑚+1. Then the resulting
token sequence{(𝑑𝑡 1,...,𝑑𝑡𝑗−1,𝑡𝑗),𝑗≤𝑚+1}generated in a single forward.
This parallel verification mechanism not only accelerates the autoregressive generation process of LLMs but also
ensures that the final output strictly adheres to the model’s intrinsic probability distribution. As a result, generation
quality is preserved when the same decoding or sampling strategy is employed.
3.4.2 Phrase Pipline.In this section, we describe the overall pipline of theReduce Phase. Given the refined context
𝐺∗and the draft sequences obtained from theMap Phase, we construct a new instruction by combining 𝐺∗with the
query𝑞. This instruction is then fed into a larger target LLM to generate the final output. This design provides the
target LLM with high-quality, multi-perspective code context, enabling more accurate and robust code generation. At
the same time, by filtering out noisy or irrelevant information, it improves the model’s ability to effectively utilize
relevant contextual signals. Next, we select the first draft code 𝐷0from the drafts obtained by the draft LLM as the
candidate(𝑑𝑡1,···,𝑑𝑡𝑚)for verification to accelerate the generation process of the target LLM, where 𝑑𝑡𝑖represents
the𝑖-th token in the draft sequence. This choice is motivated by two considerations: (i)Draft quality:the first draft
Manuscript submitted to ACM

10 Wang et al.
group𝐺1exhibits the highest semantic similarity to the query, making it the most likely to yield a high-quality solution;
(ii)Efficiency:prioritizing the most probable candidate minimizes the verification overhead for the target LLM.
As shown in Algorithm 1, during verification, if the target LLM rejects a draft token at position 𝑗, we retain the
remaining suffix of the draft (𝑑𝑡𝑗+1,...,𝑑𝑡𝑚), which is used as the subsequent draft for future verification. After rejection,
MRCodertemporarily falls back to autoregressive decoding by the target LLM. Let the newly generated token sequence
be(𝑡 1,...,𝑡𝑘). We then performspan matchingbetween this generated sequence and the remaining draft sequence.
Formally, given the remaining draft(𝑑𝑡 𝑗+1,...,𝑑𝑡𝑚), we search for a matching position𝑠such that:
∃𝑠∈[𝑗+1,𝑚−𝑘+1],(𝑡 1,...,𝑡𝑥)=(𝑑𝑡𝑠,...,𝑑𝑡𝑠+𝑥−1).(12)
If such a position 𝑠exists, we realign the draft sequence and resume parallel verification using the remaining suffix
(𝑑𝑡𝑠+𝑥,...,𝑑𝑡𝑚). Otherwise, the model continues with standard autoregressive decoding. This mechanism allows the
LLM to recover alignment with the draft even after local mismatches, enabling efficient reuse of partially consistent
draft tokens and further improving decoding efficiency.
4 Experiment Setups
4.1 Benchmarks
We evaluateMRCoderon two widely adopted repository-level code generation benchmarks.
•CoderEval[ 31] contains 230 Python and Java tasks curated from real-world projects. It covers functions with
diverse contextual dependencies and employs an automated execution platform to assess functional correctness.
In our experiments, we use the Python subset, comprising 230 instances.
•DevEval[ 15] is a large-scale, manually annotated benchmark for repository-level code generation. It includes
1,825 test samples from 117 Python repositories spanning 10 popular domains (e.g., Internet and databases).
However, as it does not provide a fully reproducible test environment, we configure the environment locally and
obtain 1,462 valid data points by filtering instances based on whether the ground truth code successfully passes
all associated tests.
4.2 Baselines
To evaluate the effectiveness ofMRCoder, we compare it against the following representative baselines:
•Standard RAG.This baseline adopts a conventional BM25-based retrieval strategy. We use the target function
signature along with its natural language description (e.g., comments) as the query, and retrieve relevant code
contexts from the repository.
•RL-Coder [ 28].This is an advanced repository-level code generation method that leverages reinforcement
learning to optimize the retriever. It uses perplexity-based rewards to encourage the selection of informative
code contexts.
•RepoFormer [ 29].It is a selective retrieval approach that first determines whether retrieval is necessary before
generation. For a fair comparison, we adopt the StarCoder-1B that they have trained, which has a comparable
parameter scale to the models used in our framework.
•LongCodeZip [ 22].This is a code context compression and filtering method based on perplexity estimation. It
employs a smaller model to compute perplexity scores for context selection. To ensure fairness, the same model
used as the draft generator inMRCoderis employed for perplexity computation in this baseline.
Manuscript submitted to ACM

MRCoder: An Efficient Context Selecting Approach for Repository-Level Code Generation 11
4.3 Metrics
•Pass@K.We adopt the Pass@K metric to evaluate code generation performance, which measures the probability
that at least one correct solution is produced within𝐾attempts:
Pass@K=1− 𝑛−𝑐
𝐾
 𝑛
𝐾,(13)
where𝑛is the total number of generated samples, 𝑐is the number of correct solutions, and 𝐾is the number of
allowed attempts. Unless otherwise specified, we report Pass@1.
•Time(s).We measure generation efficiency in seconds using three metrics: (1) Preprocessing Time (Proc.T),
which denotes the time required for context preprocessing after retrieval; (2) Generation Time (Gen.T), which
measures the time taken by the LLM to generate code; and (3) Total Time (All.T), defined as the sum of Proc.T
and Gen.T.
•Token.We compute the length of the retrieved (or compressed) code context using the GPT tokenizer across all
methods. This metric reflects the efficiency of code context selection and generation cost.
4.4 Implementation Details
Backbone Models:Due to the need to obtain the probability distribution from the LLM and constraints on computational
resources, we select two open-source and widely used code LLM families:Qwen2.5-CoderandDeepSeek-Coder. For
Qwen2.5-Coder, we adopt Qwen2.5-Coder-7B-Instruct as the generative (target) model and Qwen2.5-Coder-1.5B-Instruct
as the draft model. ForDeepSeek-Coder, we use DeepSeek-Coder-6.7B-Instruct as the generative model and DeepSeek-
Coder-1.3B-Instruct as the draft model. For baseline methods, RAG and RL-Coder rely solely on the generative model
without drafting models. For RepoFormer, we replace the drafting model with its trained lightweight model of comparable
scale, RepoFormer-1B. For LongCodeZip andMRCoder, we adopt the same generative–draft model combinations
described above.
Hardware and Implementation Details:All experiments are conducted on a server equipped with 4 NVIDIA
RTX 4090 GPUs, which are used for both draft and generative LLM inference. The CPU is an Intel®Xeon®Gold 6330.
All time tests are conducted under the above conditions to ensure a fair comparison. ForMRCoder, based on our
preliminary experiments we set the hyperparameter 𝐿in the Logic Similarity Perspective is 1 and the size of each
group in theMap Phraseis 4. All other baselines follow the default configurations reported in their default settings.
To ensure deterministic generation and eliminate randomness, the decoding temperature is set to 0. To verify the
effectiveness of different numbers of retrieval blocks, the number of retrieved contexts is varied among {0,3,5,7,10},
and the corresponding results are reported. The selection of these values is guided by preliminary experiments, which
reveal a clear turning point within this range: the Pass@1 score first improves with increasing 𝑘and then degrades
as more contexts are introduced. This trend aligns with our motivation, indicating that while additional context can
provide useful information, excessive retrieval introduces noise that negatively affects generation quality.
5 Experimental Results
We aim to address the following three research questions:
•RQ1: Code Generation Quality.How doesMRCoderperform in terms of the correctness of generated code
for repository-level code generation tasks?
Manuscript submitted to ACM

12 Wang et al.
Table 1. Pass@1 results comparison on CoderEval and DevEval. The green percentage boxes indicate the percentage improvement in
Pass@1 forMRCodercompared to the RAG baseline. The text in bold indicates the highest score, and the underlined values denote
the performance of the best baseline.
Model MethodCoderEval DevEval
K=0 K=3 K=5 K=7 K=10 K=0 K=3 K=5 K=7 K=10
Qwen2.5-CoderRAG 24.4 31.7 37.0 35.2 33.5 13.0 15.8 15.9 15.0 15.3
RL-Coder 24.4 26.5 30.0 27.0 31.3 13.0 16.2 15.9 15.7 16.1
Repoformer 24.4 29.6 30.9 31.3 29.6 13.0 13.4 13.0 12.7 13.0
LongCodeZip 24.4 36.5 38.3 36.5 35.2 13.0 15.3 15.8 16.1 16.3
MRCoder24.437.8 ↑19.2% 40.0↑8.1% 39.6↑12.5% 38.7↑15.5% 13.019.9↑25.9% 19.0↑19.5% 19.3↑28.7% 18.1↑18.3%
DeepSeek-CoderRAG 25.2 31.3 30.0 28.3 23.9 13.4 24.1 23.2 24.5 23.5
RL-Coder 25.2 26.5 27.4 27.8 26.3 13.4 24.4 26.124.7 24.2
Repoformer 25.2 28.7 29.6 29.1 27.4 13.4 18.0 17.7 17.9 18.1
LongCodeZip 25.2 26.5 30.0 26.9 27.8 13.4 22.9 24.4 27.6 27.6
MRCoder25.231.7 ↑1.3% 35.2↑17.3% 32.6↑15.2% 36.5↑52.7% 13.4 24.1 ↑0.0% 25.5↑10.0% 26.3↑7.3% 26.3↑11.9%
•RQ2: Efficiency and Cost Analysis.How doesMRCoderperform with respect to computational cost (e.g.,
number of input tokens) and efficiency (e.g., generation time)?
•RQ3: Ablation Study.How does each component ofMRCodercontribute to the overall performance?
5.1 RQ1: Code Generation Quality
Table 1 presents Pass@1 scores of the generated code for both the baseline methods andMRCoderon DevEval and
CoderEval. Except for RL-Coder, which employs a separately trained retriever, all methods adopt BM25 for retrieval.
Standard RAG baselines do not exhibit a monotonic improvement in Pass@1 as the number of retrieved context
blocks increases. For both Qwen2.5-Coder and DeepSeek-Coder, performance typically shows an initial improvement
followed by degradation, or fluctuates across different values of 𝐾. For example, Qwen2.5-Coder on CoderEval improves
from𝐾=3to𝐾=5, but declines at 𝐾=7, while DeepSeek-Coder achieves its best performance at 𝐾=7on DevEval.
These observations indicate that although increasing 𝐾introduces more potentially useful context, it also brings in
noisy or irrelevant information, which ultimately hinders the model’s ability to effectively utilize the retrieved context.
Existing methods attempt to mitigate this issue from different perspectives, yet their effectiveness remains limited.
Specifically, RL-Coder improves retrieval quality via reinforcement learning, but its gains are marginal—for instance,
on DevEval it achieves less than a 1% improvement over standard RAG for Qwen2.5-Coder. Moreover, its reliance
on specific training dataset leads to limited generalization, resulting in inferior performance compared to RAG on
CoderEval. RepoFormer takes a different approach by learning to decide whether external context is needed through
special control tokens; however, its performance still falls behind RAG, suggesting that the key challenge is not simply
deciding whether to retrieve context, but rather how to filter and utilize it effectively. LongCodeZip further explores
fine-grained selection by using a small LLM to estimate perplexity and select informative code lines. While it performs
competitively at larger 𝐾, its reliance on perplexity-based estimation is inherently imprecise and lacks interpretability.
In contrast,MRCoderaddresses these limitations by explicitly modeling context relevance through the SADGS
algorithm, which integrates two complementary signals: API call relationships and structural similarity between draft
generations and repository code. This design enablesMRCoderto provide concise and high-quality context, thereby
reducing interference from irrelevant information and improving the model’s ability to leverage useful context during
generation. As a result,MRCoderachieves strong and stable performance across benchmarks. On CoderEval, both
LLM combinations attain the best results among all baselines, with DeepSeek-Coder improving Pass@1 by nearly
Manuscript submitted to ACM

MRCoder: An Efficient Context Selecting Approach for Repository-Level Code Generation 13
K=3 K=5 K=7 K=10051015202530Inference Time (s)CoderEval | Qwen2.5-Coder
4.65.25.56.1
4.04.54.85.2
4.2 4.44.8 4.912.114.719.423.8
2.22.63.03.5
-52.1%-50.0%-45.5%-42.6%
K=3 K=5 K=7 K=10051015202530CoderEval | DeepSeek-Coder
7.99.09.811.0
6.77.28.18.9
6.4 6.5 6.5 6.612.616.119.224.5
4.96.26.68.0
-38.0%-31.1%-32.7%-27.3%
K=3 K=5 K=7 K=10010203040Inference Time (s)DevEval | Qwen2.5-Coder
7.4 7.6 7.98.4
7.4 7.6 7.78.27.4 7.6 7.4 7.621.824.628.035.2
3.8 4.14.75.6
-48.6% -46.1%-40.5%-33.3%
K=3 K=5 K=7 K=1005101520253035DevEval | DeepSeek-Coder
10.010.3 10.511.7
9.710.3 10.4 10.6
7.9 8.18.48.717.222.025.830.1
6.16.66.98.2
-39.0%-35.9%-34.3%-29.9%RAG RL-Coder Repoformer LongCodeZip MRCoder Proc.T Gen.T RAG RL-Coder Repoformer LongCodeZip MRCoder Proc.T Gen.T
Fig. 3. Inference time comparison ofMRCoderand baseline methods. The green percentage boxes indicate the percentage reduction
in time forMRCodercompared to the RAG baseline.
52.7% over standard RAG at 𝐾=10and 31.2% over best baseline LongCodeZip. On DevEval,MRCoderconsistently
outperforms the RAG baseline across all settings. Although it is slightly inferior to RL-Coder ( 𝐾=3,5) and LongCodeZip
(𝐾=7,10) at specific values of 𝐾,MRCodermaintains significantly more stable performance as 𝐾varies. These results
demonstrate thatMRCodernot only improves code generation quality but also provides a more interpretable and
reliable mechanism for context selection. Moreover,MRCoderconsistently outperforms standard RAG across different
𝐾values and exhibits a similar performance trend (i.e., first improving and then degrading as 𝐾increases). This suggests
that the optimal𝐾identified under RAG can be directly applied toMRCoderto achieve the best performance.
RQ1 Summary:MRCoderconsistently achieves strong and stable performance in terms of code correctness across
both CoderEval and DevEval, outperforming or matching competitive baselines. By modeling context relevance via
code structure and API relationships,MRCoderfilters noisy retrievals and provides concise, high-quality context,
improving generation and interpretability. These results demonstrate that precise, structured context selection is
critical for repository-level code generation.
5.2 RQ2: Efficiency and Cost Analysis
Figure 3 presents the inference time comparisons and Table 2 presents the token comparisons across different methods
on CoderEval and DevEval. Standard RAG exhibits a clear increase in both inference time and token consumption as the
number of retrieval blocks 𝐾grows. Since it directly concatenates all retrieved contexts, larger 𝐾leads to longer input
Manuscript submitted to ACM

14 Wang et al.
Table 2. Token consumption comparison on CoderEval and DevEval. The green percentage boxes indicate the percentage reduction in
tokens forMRCodercompared to the RAG baseline. The text in bold indicates the lowest token usage overall, and the underlined
values denote the best-performing baseline in terms of token efficiency.
Model MethodCoderEval DevEval
K=3 K=5 K=7 K=10 K=3 K=5 K=7 K=10
Qwen2.5-CoderRAG 1.7k 3.1k 4.6k 7.3k 1.8k 3.2k 4.6k 7.0k
RL-Coder 1.0k 2.0k 3.3k 5.6k 1.3k 2.4k 3.6k 5.5k
Repoformer 0.7k 1.3k 1.9k 2.8k 0.8k 1.3k1.9k 3.0k
LongCodeZip 0.8k 1.3k 1.9k2.9k 0.9k 1.3k 1.8k 2.6k
MRCoder1.2k ↓29.4% 1.9k↓38.7% 2.7k↓41.3% 3.6k↓50.7% 1.2k↓33.3% 1.8k↓43.8% 2.7k↓41.3% 3.8k↓45.7%
DeepSeek-CoderRAG 1.7k 3.1k 4.6k 7.3k 1.8k 3.2k 4.6k 7.0k
RL-Coder 1.0k 2.0k 3.3k 5.6k 1.3k 2.4k 3.6k 5.5k
Repoformer 0.7k1.3k 1.9k 2.8k 0.8k 1.3k1.9k 3.0k
LongCodeZip 0.7k 1.2k 1.7k 2.5k0.9k 1.3k 1.8k 2.5k
MRCoder1.0k ↓41.2% 1.8k↓41.9% 2.2k↓52.2% 3.5k↓52.1% 1.1k↓38.9% 2.0k↓37.5% 2.5k↓45.6% 3.4k↓51.4%
sequences, which significantly increases generation latency and computational cost. This trend is consistent across
both Qwen2.5-Coder and DeepSeek-Coder, indicating that naive retrieval scaling introduces substantial inefficiency.
Existing baselines attempt to reduce cost from different perspectives, but each has inherent limitations. RL-Coder
reduces token consumption compared to RAG by improving retrieval quality, which slightly lowers generation time.
We attribute this to the fact that, although RL-Coder does not directly train the retriever to compress context, it assigns
higher scores to function-level code contexts that are more semantically consistent with the current query, while
assigning lower retrieval scores to potential longer external API contexts (such as classes), thereby reducing token
costs and generation time. However, its improvements remain moderate, as it still relies on incorporating multiple
retrieved contexts. RepoFormer achieves the lowest token consumption by aggressively controlling whether retrieval is
used (only one token’s inference time is required), resulting in minimal processing overhead and stable inference time.
Nevertheless, this strategy sacrifices performance, as shown in RQ1, indicating it cannot achieve a trade-off between
inference cost and time versus quality. LongCodeZip adopts a fine-grained selection strategy and achieves competitive
token reduction at larger 𝐾, but its computational overhead is substantial due to multiple forward passes of a small
model for perplexity estimation. As a result, its processing time increases dramatically with 𝐾, leading to the highest
overall latency among all methods. This limits its applicability in time-sensitive code generation scenarios.
In contrast,MRCoderachieves a more favorable balance between efficiency and effectiveness. AlthoughMRCoder
introduces additional overhead—stemming from draft generation by a small model in theMap Phaseand context
selection via SADGS—the batching and parallelization design ensures that the overall latency increase remains limited.
As shown in Figure 3, it incurs at most around 2 seconds of additional inference time. Moreover, the use of parallel
decoding for draft verification in theReduce Phasesignificantly improves inference efficiency, effectively offsetting
the extra computation introduced in earlier stages. On CoderEval,MRCoderreduces total inference time by up to
52.1% for Qwen2.5-Coder and 38.0% for DeepSeek-Coder at 𝑘= 3, while maintaining consistent reductions as 𝐾
increases. Similar trends are observed on DevEval, where time reductions reach up to 48.6% and 39.0%, respectively. In
most experimental settings,MRCoderachieved the highest generation efficiency compared to all baseline methods.
Notably, the processing overhead introduced byMRCoderremains minimal compared to LongCodeZip, demonstrating
that its selection mechanism is lightweight and scalable. In terms of token efficiency,MRCoderconsistently reduces
token consumption by 30%–50% compared to RAG across both benchmarks and model settings. While it does not
Manuscript submitted to ACM

MRCoder: An Efficient Context Selecting Approach for Repository-Level Code Generation 15
Table 3. Pass@1 ablation study results on CoderEval and DevEval. Red and green boxes represent the absolute decrease and increase,
respectively, when specific components are removed fromMRCoder.
Model MethodCoderEval DevEval
K=3 K=5 K=7 K=10 K=3 K=5 K=7 K=10
Qwen2.5-CoderMRCoder37.8 40.0 39.6 38.7 19.9 19.0 19.3 18.1
w/o_API 35.2 ↓2.6 39.6↓0.4 39.6↓0.0 37.0↓1.7 14.8↓5.1 14.7↓4.3 16.0↓3.3 14.8↓3.3
w/o_Sim 36.5 ↓1.3 38.3↓1.7 39.6↓0.0 37.8↓0.9 17.3↓2.6 17.2↓1.8 17.1↓2.2 16.9↓1.2
DeepSeek-CoderMRCoder31.7 35.2 32.6 36.5 24.1 25.5 26.3 26.3
w/o_API 32.2 ↑0.5 36.5↑1.3 31.7↓0.9 36.0↓0.5 20.9↓3.2 23.1↓2.4 23.1↓3.2 24.6↓1.7
w/o_Sim 31.3 ↓0.4 34.8↓0.4 33.5↑0.9 33.9↓2.6 22.4↓1.7 24.1↓1.4 24.1↓2.2 25.2↓1.1
Table 4. Token ablation consumption ofMRCoderon CoderEval and DevEval. The green boxes indicate the reduction in context
length when removing specific modules.
Model MethodCoderEval DevEval
K=3 K=5 K=7 K=10 K=3 K=5 K=7 K=10
Qwen2.5-CoderMRCoder1.2k 1.9k 2.7k 3.6k 1.2k 1.8k 2.7k 3.8k
w/o_API 0.6k ↓50.0% 1.3k↓31.6% 1.4k↓48.1% 2.1k↓41.7% 0.8k↓33.3% 1.6k↓11.1% 1.9k↓29.6% 2.5k↓34.2%
w/o_Sim 1.0k ↓16.7% 1.6k↓15.8% 2.3k↓14.8% 3.0k↓16.7% 0.8k↓33.3% 1.7k↓5.6% 1.3k↓51.9% 2.3k↓39.5%
DeepSeek-CoderMRCoder1.0k 1.8k 2.2k 3.5k 1.1k 2.0k 2.5k 3.4k
w/o_API 0.5k ↓50.0% 1.3k↓27.8% 1.1k↓50.0% 2.0k↓42.9% 0.9k↓18.2% 1.6k↓20.0% 1.8k↓28.0% 2.4k↓29.4%
w/o_Sim 0.9k ↓10.0% 1.5k↓16.7% 1.9k↓13.6% 3.0k↓14.3% 0.8k↓27.3% 1.2k↓40.0% 1.6k↓36.0% 2.2k↓35.3%
always achieve the absolute lowest token usage (as RepoFormer and LongCodeZip are more aggressive in pruning), it
maintains substantially lower token cost without sacrificing generation quality. This highlights thatMRCoderavoids
the inefficiency of redundant context while preserving the essential information required for accurate code generation.
RQ2 Summary:MRCoderimproves both efficiency and generation quality simultaneously. By effectively filtering
redundant context, it reduces token consumption by 30%–50% and inference time by up to over 50%, mitigating the
cost explosion of increasing 𝐾. Compared to prior methods, it avoids heavy computational overhead and achieves the
highest generation efficiency, providing a more efficient and scalable solution for repository-level code generation.
5.3 RQ3: Ablation Study
To answer this question, we conduct ablation studies to analyze the contribution of each component inMRCoder.
Specifically, we remove individual components, including API-based selection, similarity-based selection, and parallel
decoding (PD), to evaluate their impact on overall performance in terms of both code quality and efficiency/cost. Since
parallel decoding does not affect code quality, the quality-focused analysis considers only the removal of the two
selection components.
From the perspective of code quality, as shown in Table 3, removing either the API-based or similarity-based
selection leads to consistent performance degradation across most settings. Specifically, removing the API component
results in more significant drops on DevEval (e.g., up to 5.1 absolute decrease for Qwen2.5-Coder at 𝐾=3), indicating
that API call relationships play a crucial role in identifying functionally relevant context. In contrast, removing the
similarity component leads to relatively smaller but still consistent declines, suggesting that structural similarity
provides complementary signals that refine context selection. Although occasional marginal improvements are observed
in certain settings (e.g., DeepSeek-Coder at specific 𝐾values), these gains are not stable. As a result, slight improvements
Manuscript submitted to ACM

16 Wang et al.
K=3 K=5 K=7 K=1001234567Inference Time (s)CoderEval | Qwen2.5-Coder
2.22.63.03.5
2.22.42.53.0
2.22.73.13.54.54.85.15.6
+104.5%+84.6%+70.0%+60.0%
K=3 K=5 K=7 K=10024681012CoderEval | DeepSeek-Coder
4.96.26.68.0
5.16.36.17.6
5.26.56.38.0
7.78.28.810.0
+57.1%+32.3%+33.3%+25.0%
K=3 K=5 K=7 K=100246810Inference Time (s)DevEval | Qwen2.5-Coder
3.84.14.75.6
3.43.94.14.3
3.53.74.14.37.57.98.28.9
+97.4%+92.7%+74.5%+58.9%
K=3 K=5 K=7 K=1002468101214DevEval | DeepSeek-Coder
6.16.66.98.2
6.06.4 6.37.1
5.96.36.66.910.111.1 11.012.3
+65.6%+68.2% +59.4%+50.0%MRCoder w/o_API w/o_Sim w/o_PD Proc.T Gen.T MRCoder w/o_API w/o_Sim w/o_PD Proc.T Gen.T
Fig. 4. Efficiency ablation analysis forMRCoderon CoderEval and DevEval. The red percentages boxes in w/o_PD denote the
percentage reduction in time compared toMRCoder.
can be observed in both w/o_API and w/o_Sim variants in some cases. However, the overall performance trends
consistently demonstrate that both components contribute synergistically to improving code generation quality.
In terms of inference time, as shown in Figure 4, the differences betweenMRCoderand its w/o_API and w/o_Sim
variants are relatively minor, suggesting that both selection mechanisms introduce negligible computational overhead
and remain lightweight. For the perspective of cost, as shown in Table 4 , the impact of each component exhibits
different characteristics. Removing the API or similarity modules consistently reduces token consumption, as fewer
constraints are applied during context selection, leading to shorter input sequences. However, this reduction comes
at the cost of degraded generation quality, indicating that aggressive pruning without structural guidance harms
effectiveness. In contrast, removing parallel decoding (w/o_PD) results in a substantial increase in inference latency
across all settings. For example, total inference time increases by up to 104.5% on CoderEval and nearly doubles on
DevEval. This demonstrates that parallel decoding is the key factor enabling efficient generation inMRCoder, not
only effectively amortizing the additional overhead introduced in earlier stages such as draft generation and context
selection, but improving the overall generation efficiency.
RQ3 Summary:Both API-based and similarity-based selection are essential for achieving high-quality code
generation, contributing complementary signals for effective context filtering. While removing these components
may reduce token consumption, it leads to noticeable performance degradation, highlighting the importance of
these two complementary components. In contrast, parallel decoding plays a critical role in efficiency, significantly
Manuscript submitted to ACM

MRCoder: An Efficient Context Selecting Approach for Repository-Level Code Generation 17
Instruction: Please generate the function code based on the 
contexts.The code to be completed: def minor_third(note): Answer: Here's the ` minor_third` function: 
def minor_third(note):
    #...
    return interval("C", note, 3) 
Selected  Context
Token: 0.5k
Time: 2.8s
Answer: Here's the ` minor_third` function: 
def minor_third(note):
    #...
    return interval("C", note, 3) 
Token: 0.4k
Time: 17.6s
Incorrect API
Incorrect API
Answer: Here's the ` minor_third` function: 
def minor_third(note):
    #...
    return intervals.minor_third(note) 
Token: 0.4k
Time: 2.6s
Incorrect API
Answer: Here's the ` minor_third` function: 
def minor_third(note):
    #...
    return get_interval(note, 3)
Token: 0.3k
Time: 0.7s
Correct APIDraft_Answer: Here's the ` minor_third` function: 
def minor_third(note):
    #...
    return get_interval(note, 3, "Cm")API Call: get_interval && Highest Similarity 
def get_interval(note, interval, key="C"):
    #...
    key_notes = keys.get_notes(key)
    for x in key_notes:
    #...Context-4, RAG/LongCodeZip/MRCoderdef minor_triad(value):
    """Return the triplet note value.
    #...
    return [note, intervals.minor_third(note), 
intervals.perfect_fifth(note)]Context-3, RLCoderdef triplet(value):
    """Return the triplet note value.
    #...
    return tuplet(value, 3, 2)Context-2, RAG/LongCodeZip/RL-Coder/MRCoderdef third(note, key):
    """Take the diatonic third of note in key.
    Raise a KeyError exception if the note is not found 
in the given key.
    #...
    return interval(key, note, 2)Context-1, RAG/LongCodeZip/RL-Coder/MRCoder
Parallel  Decoding
Fig. 5. An example from the DevEval benchmark, illustrating the code generation results and corresponding retrieved contexts for
MRCoderand the baselines (standard RAG, LongCodeZip, and RL-Coder), using Qwen2.5-Coder as the backbone model. Standard
RAG, LongCodeZip, and MRCoder (Map phase) employ BM25 to retrieve Contexts 1, 2, and 4, while RL-Coder retrieves Contexts 1, 2,
and 3 using its retriever. For MRCoder, Context 4 is ultimately selected from the context group (Contexts 1, 2, 4) and is then used in
the Reduce phase.
reducing inference latency and offsetting the overhead of earlier stages. Together, these components enableMRCoder
to maintain strong generation quality while achieving efficient and scalable inference.
6 Disscussion
6.1 Case Study
Figure 5 presents a representative example from the DevEval benchmark, comparing the code generation results of
MRCoderand the baselines. The task is to implement the functionminor_third(note)based on multiple retrieved code
snippets.
As shown in the figure, in Standard RAG and other baselines (LongCodeZip and RL-Coder), the LLM directly utilize
all their retrieved contexts, leading to incorrect API usage. Specifically, they either invokeinterval("C", note, 3)or
intervals.minor_third(note), both of which are inconsistent with the repository’s actual API design. This error stems from
the presence of superficially similar but semantically misaligned contexts (e.g.,third,triplet, andminor_triad), which
introduce ambiguity and misguide the generation process. In contrast,MRCoderemploys a draft-guided selection
mechanism to identify the most relevant context. During theMap Phase, the draft LLM generates an intermediate solution
that correctly reflects the intended API pattern (get_interval). Based on API call consistency and logical similarity, the
LLM successfully filters out noisy contexts and retains the key snippet definingdef get_interval(note, interval, key="C").
Manuscript submitted to ACM

18 Wang et al.
Table 5. Pass@1 results comparison of different context partition strategies in the Map phase. The text in bold indicates the highest
score.
Model Partition StrategyCoderEval DevEval
K=3 K=5 K=7 K=10 K=3 K=5 K=7 K=10
Qwen2.5-CoderRandom Partition 36.1 38.7 38.342.219.9 19.0 19.6 20.0
Interleaved Partition 36.4 38.3 38.343.419.6 19.5 19.6 20.1
Clustering Partition 36.3 39.6 38.3 37.8 19.7 19.7 19.9 18.9
Sequential Partition37.8 40.0 39.638.719.9 19.0 19.3 18.1
DeepSeek-CoderRandom Partition31.933.0 34.8 33.0 22.9 23.9 24.1 25.1
Interleaved Partition 31.235.2 36.933.5 23.9 24.7 25.426.3
Clustering Partition 30.4 32.6 33.5 35.6 23.9 23.9 24.4 25.9
Sequential Partition 31.735.232.636.5 24.1 25.5 26.3 26.3
Table 6. Generation time comparison of different draft selection strategies in the Reduce phase. The text in bold indicates the shortest
generation time.
Model Draft Selection StrategyCoderEval DevEval
K=3 K=5 K=7 K=10 K=3 K=5 K=7 K=10
Qwen2.5-CoderRandom Draft1.92.3 2.82.9 3.6 3.7 4.04.7
First Draft1.9 2.2 2.5 2.9 3.6 3.74.04.6
DeepSeek-CoderRandom Draft4.55.8 6.36.4 5.7 5.76.4 6.3
First Draft4.5 5.6 5.76.75.7 5.7 5.8 6.1
In theReduce Phase, the refined context is used by the target model to generate the final implementation, yielding the
correct API call.
Beyond correctness,MRCoderalso achieves notable efficiency gains. As shown in the figure, baseline methods
process longer input contexts (0.3k–0.5k tokens) and incur higher latency (2.6s–17.6s), particularly for methods such as
LongCodeZip that require multiple forward passes. In contrast,MRCoderreduces the input to only the most relevant
context (0.3k tokens) and, combined with parallel decoding, significantly lowers the inference time to 0.7s.
This case demonstrates thatMRCodernot only improves functional accuracy but also effectively reduces computa-
tional overhead by eliminating redundant context and enabling more efficient decoding.
6.2 Impact of Context Partition Strategy in Map Phrase
To investigate the impact of different context partitioning strategies in theMap phaseon code generation quality, we
conduct additional experiments incorporating the following four representative partitioning approaches, enabling a
systematic comparison of their effects on generation performance.
•Sequential Partition: The partition strategy used in our paper. Given the retrieved context blocks ordered by
their similarity to the query, this strategy directly splits them into consecutive groups of a fixed size. Specifically,
the first𝑚blocks form the first group, the next 𝑚blocks form the second group, and so on. Consequently, this
strategy preserves the original retrieval ranking and places highly ranked contexts into earlier groups, making
the first group the most relevant one.
•Random Partition: This strategy first randomly shuffles all retrieved context blocks using a fixed random
seed, and then applies the same consecutive grouping procedure as Sequential Partition. The purpose is to break
Manuscript submitted to ACM

MRCoder: An Efficient Context Selecting Approach for Repository-Level Code Generation 19
the original relevance order and examine whether the grouping strategy is sensitive to the relevant ranking of
retrieved contexts.
•Interleaved Partition: This strategy first determines the number of groups based on the total number of
retrieved blocks and the group size, and then distributes contexts into groups in a round-robin manner. For
example, if there are 𝑛groups, the first group contains blocks at positions1 ,1+𝑛,1+2𝑛,... , whilst the second
group contains blocks at positions2 ,2+𝑛,2+2𝑛,... . In this way, each group mixes contexts from different
relevance levels.
•Clustering Partition: This strategy groups retrieved context blocks according to their vector representations.
Specifically, each block is first transformed into a feature vector, and K-Means clustering is then applied to
divide all blocks into several clusters, where the number of clusters is determined by the expected group size.
This strategy aims to group semantically similar code snippets together, but it introduces additional clustering
overhead.
As shown in Table 5, different context partition strategies lead to overall comparable performance across both models
and benchmarks, and no single strategy consistently dominates across all settings. While Random and Interleaved
Partition occasionally achieve higher Pass@1 scores at specific values of 𝐾(e.g., at larger 𝐾on CoderEval), these
improvements are not stable and often do not generalize across datasets or models. Similarly, Clustering-based Partition
shows competitive performance in some cases, but fails to provide consistent gains despite its additional computational
overhead.
These observations suggest that the partition strategy itself is not a primary bottleneck in the Map phase. Instead, as
long as each group contains a reasonable subset of retrieved contexts, the subsequent draft generation and selection
mechanism can effectively filter useful information. In this sense, different partition strategies mainly affect how context
diversity and relevance are distributed across groups, but their overall impact on final generation quality remains
limited. Among all strategies, Sequential Partition exhibits relatively stable performance across most configurations.
By preserving the original retrieval order, it groups highly relevant contexts together, which helps the draft model
generate more coherent and focused drafts. At the same time, it avoids introducing additional randomness or structural
perturbations, leading to more consistent behavior.
Practical Recommendations.In practice, Sequential Partition is recommended as the default choice due to
its simplicity, efficiency, and stable performance. Interleaved Partition may be considered when relevant contexts
are unevenly distributed and increased diversity across groups is desired, but its benefits are not consistent. Random
Partition generally introduces unnecessary variance, and Clustering-based Partition is less attractive due to its additional
preprocessing cost without clear performance advantages. Overall, Sequential Partition is sufficient and effective for
theMap phaseand we used in our paper.
6.3 Impact of Draft Selection Strategy in Reduce Phase
To further investigate the impact of draft selection in theReduce phase, we compare different strategies for choosing the
draft used in parallel verification. Specifically, we consider two representative strategies:
•First Draft: Select the first draft generated from the most relevant context group (i.e., the group with highest
similarity to the query). This strategy is adopted as the default choice in our method. Since the context groups are
constructed in descending order of relevance, the first draft is more likely to align with the target model’s token
distribution, leading to a higher acceptance rate during parallel verification and improved decoding efficiency.
Manuscript submitted to ACM

20 Wang et al.
•Random Draft: Randomly select a draft from all candidate drafts generated in theMap phase. This strategy is
introduced as a control setting to evaluate the sensitivity of theReduce phaseto draft quality. By removing the
relevance-based selection bias, Random Draft allows us to assess how the impact of draft quality (the alignment
with the target LLM’s answer) on generation efficiency
As shown in Table 6, different draft selection strategies lead to overall comparable generation time across both
models and benchmarks. But selecting the first draft exhibits more stable and consistently competitive performance
across different configurations. This can be attributed to the quality of the selected draft. The first draft is generated
from the most relevant context group, making it more likely to align with the target model’s token distribution. As
a result, a larger proportion of draft tokens can be directly accepted during parallel verification, reducing the need
for fallback decoding and improving overall efficiency. In contrast, Random Draft may select drafts derived from less
relevant context groups, which are more likely to deviate from the target model’s predictions. This leads to increased
token rejections and additional validation overhead, slightly degrading decoding efficiency.
Practical Recommendations.In practice, selecting the first draft is recommended as the default strategy, as it
consistently provides high-quality drafts that better match the target model’s predictions. Given its simplicity and
effectiveness, a deterministic and relevance-based draft selection strategy is sufficient for achieving efficient decoding
in theReduce phase.
7 Threats to Validity
Threats to External Validity.First, our experiments are conducted on two widely used repository-level code generation
benchmarks, namely CoderEval and DevEval, which mainly focus on Python repositories. Although these benchmarks
may not fully represent other programming languages, they cover diverse real-world scenarios and are widely adopted in
existing code generation studies [ 4,17,19,30,35]. Second, we evaluateMRCoderusing two representative open-source
LLM families, Qwen2.5-Coder and DeepSeek-Coder. We adopt open-source models because our method requires access
to token-level probability distributions (e.g., for parallel verification), which is typically unavailable in commercial APIs,
and because they allow full control over inference to ensure stable efficiency measurements. Due to resource constraints,
we do not include extremely large models. However, our framework is model-agnostic and does not rely on specific
architectures, and thus can be extended to larger or proprietary models when decoding signals are accessible. Third,
we adopt BM25 as the retrieval method. AlthoughMRCoderis designed to be retrieval-agnostic, different retrieval
strategies (e.g., dense or graph-based retrieval) may interact differently with our selection mechanism.
Threats to Internal Validity.Internal validity concerns the correctness of the experimental design and implemen-
tation. To ensure fair comparisons, we carefully follow the original papers for all baseline methods and adopt official or
widely used implementations. We align hyperparameters, preprocessing, and training settings as closely as possible,
though minor differences (e.g., for RL-Coder) may remain. Second, the experimental environment (e.g., GPU hardware
and system dependencies) may influence efficiency measurements such as inference time; we mitigate this by running
all experiments on the same hardware. Finally, using small models for draft generation may be limited by their capacity,
potentially overlooking relevant context. While our results demonstrate the overall effectiveness ofMRCoder, this
limitation will be addressed in future work.
Manuscript submitted to ACM

MRCoder: An Efficient Context Selecting Approach for Repository-Level Code Generation 21
8 Conclusion
In this paper, we proposeMRCoder, an efficient context selection framework for repository-level code generation. By
adopting a Map–Reduce paradigm with draft-guided context selection,MRCodereffectively filters noisy and redundant
contexts while preserving informative contexts. The proposed SADGS mechanism integrates API call relationships and
logical similarity to identify relevant context, and the parallel verification strategy further improves generation efficiency.
Extensive experiments on CoderEval and DevEval demonstrate thatMRCoderconsistently improves generation quality
while significantly reducing token consumption and inference time, achieving a better balance between effectiveness
and efficiency compared to existing methods. In future work, we plan to extendMRCoderto larger models and more
diverse programming environments.
References
[1]Mohiuddin Ahmed, Raihan Seraj, and Syed Mohammed Shamsul Islam. 2020. The k-means algorithm: A comprehensive survey and performance
evaluation. Electronics 9, 8 (2020), 1295.
[2] Anysphere. 2023. Cursor. https://www.cursor.com/
[3]Jacob Austin, Augustus Odena, Maxwell Nye, Maarten Bosma, Henryk Michalewski, David Dohan, Ellen Jiang, Carrie Cai, Michael Terry, Quoc Le,
and Charles Sutton. 2021. Program Synthesis with Large Language Models. arXiv:2108.07732 [cs.PL] https://arxiv.org/abs/2108.07732
[4]Zhangqian Bi, Yao Wan, Zheng Wang, Hongyu Zhang, Batu Guan, Fangxin Lu, Zili Zhang, Yulei Sui, Hai Jin, and Xuanhua Shi. 2024. Iterative
Refinement of Project-Level Code Context for Precise Code Generation with Compiler Feedback. In Findings oftheAssociation forComputational
Linguistics: ACL 2024, Lun-Wei Ku, Andre Martins, and Vivek Srikumar (Eds.). Association for Computational Linguistics, Bangkok, Thailand,
2336–2353. doi:10.18653/v1/2024.findings-acl.138
[5]Aichen Cai, Anmeng Zhang, Anyu Li, Bo Zhang, Bohua Cai, Chang Li, Changjian Jiang, Changkai Lu, Chao Xue, Chaocai Liang, Cheng Zhang,
Dongkai Liu, Fei Wang, Guoqiang Huang, Haijian Ke, Han Lin, Hao Wang, Ji Miao, Jiacheng Zhang, Jialong Shi, Jifeng Zhu, Jingjing Qian, Junhui
Luo, Junwu Xiong, Lam So, Liang Huang, Ming Ke, Mingyang Li, Panfeng Shi, Peng Hao, Qi Wang, Qian Lai, Qiaoqiao Yuan, Qingyu Yin, Qiong
Cao, Qixiang Wang, Rongcheng Bian, Rongduo Han, Shaoqiang Zheng, Shi Hu, Shi Suo, Shijie Ren, Shijin Zhang, Shiying Fan, Shuai Xie, Tianyi
Zhang, Wei Liu, Wentao Tan, Xianghan Meng, Xiaodong He, Xing Pan, Xiran Wang, Xuyang Peng, Ya Zhang, Yang Liu, Yangyang Duan, Yanxu
Chen, Yicheng Gong, Yidan Huang, Yifei Liu, Yinhao Bai, Yongqiang Liu, Yuesong Zhang, Yuqi Zhang, Zerui Xie, Zhenfang Wang, Zhennan
Shen, Zheyuan Liu, and Zhuwei Zeng. 2026. JoyAI-LLM Flash: Advancing Mid-Scale LLMs with Token Efficiency. arXiv:2604.03044 [cs.CL]
https://arxiv.org/abs/2604.03044
[6]Mark Chen, Jerry Tworek, Heewoo Jun, Qiming Yuan, Henrique Ponde de Oliveira Pinto, Jared Kaplan, Harri Edwards, Yuri Burda, Nicholas Joseph,
Greg Brockman, Alex Ray, Raul Puri, Gretchen Krueger, Michael Petrov, Heidy Khlaaf, Girish Sastry, Pamela Mishkin, Brooke Chan, Scott Gray,
Nick Ryder, Mikhail Pavlov, Alethea Power, Lukasz Kaiser, Mohammad Bavarian, Clemens Winter, Philippe Tillet, Felipe Petroski Such, Dave
Cummings, Matthias Plappert, Fotios Chantzis, Elizabeth Barnes, Ariel Herbert-Voss, William Hebgen Guss, Alex Nichol, Alex Paino, Nikolas Tezak,
Jie Tang, Igor Babuschkin, Suchir Balaji, Shantanu Jain, William Saunders, Christopher Hesse, Andrew N. Carr, Jan Leike, Josh Achiam, Vedant
Misra, Evan Morikawa, Alec Radford, Matthew Knight, Miles Brundage, Mira Murati, Katie Mayer, Peter Welinder, Bob McGrew, Dario Amodei,
Sam McCandlish, Ilya Sutskever, and Wojciech Zaremba. 2021. Evaluating Large Language Models Trained on Code. arXiv:2107.03374 [cs.LG]
https://arxiv.org/abs/2107.03374
[7]Florin Cuconasu, Giovanni Trappolini, Federico Siciliano, Simone Filice, Cesare Campagnano, Yoelle Maarek, Nicola Tonellotto, and Fabrizio Silvestri.
2024. The Power of Noise: Redefining Retrieval for RAG Systems. In Proceedings ofthe47th International ACM SIGIR Conference onResearch
andDevelopment inInformation Retrieval (SIGIR 2024). ACM, 719–729. doi:10.1145/3626772.3657834
[8]Yangruibo Ding, Zijian Wang, Wasi Uddin Ahmad, Hantian Ding, Ming Tan, Nihal Jain, Murali Krishna Ramanathan, Ramesh Nallapati, Parminder
Bhatia, Dan Roth, and Bing Xiang. 2023. CrossCodeEval: A Diverse and Multilingual Benchmark for Cross-File Code Completion. In Thirty-seventh
Conference onNeural Information Processing Systems Datasets andBenchmarks Track. https://openreview.net/forum?id=wgDcbBMSfh
[9]Zhangyin Feng, Daya Guo, Duyu Tang, Nan Duan, Xiaocheng Feng, Ming Gong, Linjun Shou, Bing Qin, Ting Liu, Daxin Jiang, and Ming Zhou. 2020.
CodeBERT: A Pre-Trained Model for Programming and Natural Languages. arXiv:2002.08155 [cs.CL] https://arxiv.org/abs/2002.08155
[10] Nam Le Hai, Dung Manh Nguyen, and Nghi D. Q. Bui. 2025. On the Impacts of Contexts on Repository-Level Code Generation. In Findings ofthe
Association forComputational Linguistics: NAACL 2025 , Luis Chiruzzo, Alan Ritter, and Lu Wang (Eds.). Association for Computational Linguistics,
Albuquerque, New Mexico, 1496–1524. doi:10.18653/v1/2025.findings-naacl.82
[11] Binyuan Hui, Jian Yang, Zeyu Cui, Jiaxi Yang, Dayiheng Liu, Lei Zhang, Tianyu Liu, Jiajun Zhang, Bowen Yu, Keming Lu, Kai Dang, Yang Fan,
Yichang Zhang, An Yang, Rui Men, Fei Huang, Bo Zheng, Yibo Miao, Shanghaoran Quan, Yunlong Feng, Xingzhang Ren, Xuancheng Ren, Jingren
Zhou, and Junyang Lin. 2024. Qwen2.5-Coder Technical Report. arXiv:2409.12186 [cs.CL] https://arxiv.org/abs/2409.12186
Manuscript submitted to ACM

22 Wang et al.
[12] Mintong Kang, Nezihe Merve Gürel, Ning Yu, Dawn Song, and Bo Li. 2024. C-RAG: Certified Generation Risks for Retrieval-Augmented Language
Models. arXiv:2402.03181 [cs.AI] https://arxiv.org/abs/2402.03181
[13] Indrajith P. Karunanayaka. 2026. SplitGuard: A Resource Efficient Framework for Auditing Train and Eval Overlap and Near Duplicate Contamination
in NLP Datasets. TechRxiv 2026, 0225 (2026). arXiv:https://www.techrxiv.org/doi/pdf/10.36227/techrxiv.177205027.77074976/v1 doi:10.36227/techrxiv.
177205027.77074976/v1
[14] Yaniv Leviathan, Matan Kalman, and Yossi Matias. 2023. Fast Inference from Transformers via Speculative Decoding. arXiv:2211.17192 [cs.LG]
https://arxiv.org/abs/2211.17192
[15] Jia Li, Ge Li, Yunfei Zhao, Yongmin Li, Huanyu Liu, Hao Zhu, Lecheng Wang, Kaibo Liu, Zheng Fang, Lanshen Wang, Jiazheng Ding, Xuanming Zhang,
Yuqi Zhu, Yihong Dong, Zhi Jin, Binhua Li, Fei Huang, Yongbin Li, Bin Gu, and Mengfei Yang. 2024. DevEval: A Manually-Annotated Code Generation
Benchmark Aligned with Real-World Code Repositories. In Findings oftheAssociation forComputational Linguistics: ACL 2024 , Lun-Wei Ku, Andre
Martins, and Vivek Srikumar (Eds.). Association for Computational Linguistics, Bangkok, Thailand, 3603–3614. doi:10.18653/v1/2024.findings-acl.214
[16] Yanzhou Li, Shangqing Liu, Kangjie Chen, Tianwei Zhang, and Yang Liu. 2025. Impact-driven Context Filtering For Cross-file Code Completion.
arXiv:2508.05970 [cs.SE] https://arxiv.org/abs/2508.05970
[17] Dianshu Liao, Shidong Pan, Xiaoyu Sun, Xiaoxue Ren, Qing Huang, Zhenchang Xing, Huan Jin, and Qinying Li. 2024. A3A3-CodGen: A Repository-
Level Code Generation Framework for Code Reuse With Local-Aware, Global-Aware, and Third-Party-Library-Aware. IEEE Transactions on
Software Engineering 50, 12 (2024), 3369–3384. doi:10.1109/TSE.2024.3486195
[18] Wei Liu, Ailun Yu, Daoguang Zan, Bo Shen, Wei Zhang, Haiyan Zhao, Zhi Jin, and Qianxiang Wang. 2024. GraphCoder: Enhancing Repository-Level
Code Completion via Code Context Graph-based Retrieval and Language Model. arXiv:2406.07003 [cs.SE] https://arxiv.org/abs/2406.07003
[19] Yang Liu, Li Zhang, Fang Liu, Zhuohang Wang, Donglin Wei, Zhishuo Yang, Kechi Zhang, Jia Li, and Lin Shi. 2025. RepoScope: Leveraging Call
Chain-Aware Multi-View Context for Repository-Level Code Generation. arXiv preprint arXiv:2507.14791 (2025).
[20] Stephen Robertson and Hugo Zaragoza. 2009. The Probabilistic Relevance Framework: BM25 and Beyond. Found. Trends Inf.Retr. 3, 4 (April 2009),
333–389. doi:10.1561/1500000019
[21] Baptiste Rozière, Jonas Gehring, Fabian Gloeckle, Sten Sootla, Itai Gat, Xiaoqing Ellen Tan, Yossi Adi, Jingyu Liu, Romain Sauvestre, Tal Remez,
Jérémy Rapin, Artyom Kozhevnikov, Ivan Evtimov, Joanna Bitton, Manish Bhatt, Cristian Canton Ferrer, Aaron Grattafiori, Wenhan Xiong, Alexandre
Défossez, Jade Copet, Faisal Azhar, Hugo Touvron, Louis Martin, Nicolas Usunier, Thomas Scialom, and Gabriel Synnaeve. 2024. Code Llama: Open
Foundation Models for Code. arXiv:2308.12950 [cs.CL] https://arxiv.org/abs/2308.12950
[22] Yuling Shi, Yichun Qian, Hongyu Zhang, Beijun Shen, and Xiaodong Gu. 2025. LongCodeZip: Compress Long Context for Code Language Models.
arXiv:2510.00446 [cs.CL] https://arxiv.org/abs/2510.00446
[23] Gonzalo Travieso, Alexandre Benatti, and Luciano da F. Costa. 2024. An Analytical Approach to the Jaccard Similarity Index.
arXiv:2410.16436 [physics.data-an] https://arxiv.org/abs/2410.16436
[24] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, and Illia Polosukhin. 2023. Attention Is
All You Need. arXiv:1706.03762 [cs.CL] https://arxiv.org/abs/1706.03762
[25] Peiding Wang, Li Zhang, Fang Liu, Lin Shi, Minxiao Li, Bo Shen, and An Fu. 2025. CodeIF-Bench: Evaluating Instruction-Following Capabilities of
Large Language Models in Interactive Code Generation. arXiv:2503.22688 [cs.SE] https://arxiv.org/abs/2503.22688
[26] Peiding Wang, Li Zhang, Fang Liu, Yinghao Zhu, Wang Xu, Lin Shi, Xiaoli Lian, Minxiao Li, Bo Shen, and An Fu. 2025. EfficientEdit: Accelerating
Code Editing via Edit-Oriented Speculative Decoding. arXiv:2506.02780 [cs.SE] https://arxiv.org/abs/2506.02780
[27] Yan Wang, Xiaoning Li, Tien Nguyen, Shaohua Wang, Chao Ni, and Ling Ding. 2024. Natural Is The Best: Model-Agnostic Code Simplification for
Pre-trained Large Language Models. arXiv:2405.11196 [cs.SE] https://arxiv.org/abs/2405.11196
[28] Yanlin Wang, Yanli Wang, Daya Guo, Jiachi Chen, Ruikai Zhang, Yuchi Ma, and Zibin Zheng. 2025. RLCoder: Reinforcement Learning for
Repository-Level Code Completion. IEEE Press, 1140–1152. https://doi.org/10.1109/ICSE55347.2025.00014
[29] Di Wu, Wasi Uddin Ahmad, Dejiao Zhang, Murali Krishna Ramanathan, and Xiaofei Ma. 2024. REPOFORMER: selective retrieval for repository-level
code completion. In Proceedings ofthe41st International Conference onMachine Learning (Vienna, Austria) (ICML’24) . JMLR.org, Article 2183,
21 pages.
[30] Zezhou Yang, Sirong Chen, Cuiyun Gao, Zhenhao Li, Xing Hu, Kui Liu, and Xin Xia. 2025. An Empirical Study of Retrieval-Augmented Code
Generation: Challenges and Opportunities. ACM Trans. Softw. Eng. Methodol. 34, 7, Article 188 (Aug. 2025), 28 pages. doi:10.1145/3717061
[31] Hao Yu, Bo Shen, Dezhi Ran, Jiaxin Zhang, Qi Zhang, Yuchi Ma, Guangtai Liang, Ying Li, Qianxiang Wang, and Tao Xie. 2024. CoderEval: A
Benchmark of Pragmatic Code Generation with Generative Pre-trained Models. In Proceedings oftheIEEE/ACM 46th International Conference on
Software Engineering (Lisbon, Portugal) (ICSE ’24) . Association for Computing Machinery, New York, NY, USA, Article 37, 12 pages. doi:10.1145/
3597503.3623316
[32] Fengji Zhang, Bei Chen, Yue Zhang, Jacky Keung, Jin Liu, Daoguang Zan, Yi Mao, Jian-Guang Lou, and Weizhu Chen. 2023. RepoCoder: Repository-
Level Code Completion Through Iterative Retrieval and Generation. In Proceedings ofthe2023 Conference onEmpirical Methods inNatural
Language Processing , Houda Bouamor, Juan Pino, and Kalika Bali (Eds.). Association for Computational Linguistics, Singapore, 2471–2484. doi:10.
18653/v1/2023.emnlp-main.151
[33] Kechi Zhang, Jia Li, Ge Li, Xianjie Shi, and Zhi Jin. 2024. CodeAgent: Enhancing Code Generation with Tool-Integrated Agent Systems for Real-World
Repo-level Coding Challenges. arXiv:2401.07339 [cs.SE] https://arxiv.org/abs/2401.07339
Manuscript submitted to ACM

MRCoder: An Efficient Context Selecting Approach for Repository-Level Code Generation 23
[34] Zhaowei Zhang, Hongyu Zhang, Beijun Shen, and Xiaodong Gu. 2022. Diet code is healthy: simplifying programs for pre-trained models of code.
InProceedings ofthe30th ACM Joint European Software Engineering Conference andSymposium ontheFoundations ofSoftware Engineering
(ESEC/FSE ’22). ACM, 1073–1084. doi:10.1145/3540250.3549094
[35] Qianhui Zhao, Li Zhang, Fang Liu, Xiaoli Lian, Qiaoyuanhe Meng, Ziqian Jiao, Zetong Zhou, Jia Li, and Lin Shi. 2025. FastCoder: Accelerating
Repository-level Code Generation via Efficient Retrieval and Verification. arXiv:2502.17139 [cs.AI] https://arxiv.org/abs/2502.17139
[36] Qihao Zhu, Daya Guo, Zhihong Shao, Dejian Yang, Peiyi Wang, Runxin Xu, Y Wu, Yukun Li, Huazuo Gao, Shirong Ma, et al .2024. DeepSeek-Coder-V2:
Breaking the Barrier of Closed-Source Models in Code Intelligence. arXiv preprint arXiv:2406.11931 (2024).
Manuscript submitted to ACM