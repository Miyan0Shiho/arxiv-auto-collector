# Agentic Retrieval-Augmented Generation for Financial Document Question Answering

**Authors**: Yang Shu, Yingmin Liu, Zequn Xie

**Published**: 2026-05-06 19:59:51

**PDF URL**: [https://arxiv.org/pdf/2605.05409v1](https://arxiv.org/pdf/2605.05409v1)

## Abstract
Financial document question answering (QA) demands complex multi-step numerical reasoning over heterogeneous evidence--structured tables, textual narratives, and footnotes--scattered across corporate filings. Existing retrieval-augmented generation (RAG) approaches adopt a single-pass retrieve-then-generate paradigm that struggles with the compositional reasoning chains prevalent in financial analysis. We propose FinAgent-RAG, an agentic RAG framework that orchestrates iterative retrieval-reasoning loops with self-verification, specifically engineered for the precision requirements of financial numerical reasoning. The framework integrates three domain-specific innovations: (1) a Contrastive Financial Retriever trained with hard negative mining to distinguish semantically similar but numerically distinct financial passages, (2) a Program-of-Thought reasoning module that generates executable Python code for precise arithmetic rather than relying on error-prone LLM-based mental computation, and (3) an Adaptive Strategy Router that dynamically allocates computational resources based on question complexity, reducing API costs by 41.3% on FinQA while preserving accuracy. Extensive experiments on three benchmark datasets--FinQA, ConvFinQA, and TAT-QA--demonstrate that FinAgent-RAG achieves 76.81%, 78.46%, and 74.96% execution accuracy respectively, outperforming the strongest baseline by 5.62--9.32 percentage points. Ablation studies, cross-backbone evaluation with four LLMs, and deployment cost analysis confirm the framework's robustness and practical viability for financial institutions.

## Full Text


<!-- PDF content starts -->

Highlights
Agentic Retrieval-Augmented Generation for Financial Document Question Answering
Yang Shu, Yingmin Liu, Zequn Xie
•Agentic RAG framework for financial document QA.
•Program-of-Thought reasoning eliminates 88.0% of arithmetic errors.
•Contrastive retriever with financial hard negative mining.
•Outperforms eight baselines by 5.62–9.32 points on three benchmarks.
•Adaptive router cuts API costs by 41.3% on FinQA with minimal accuracy loss.arXiv:2605.05409v1  [cs.AI]  6 May 2026

Agentic Retrieval-Augmented Generation for Financial Document
Question Answering
Yang Shua,∗, Yingmin Liuaand Zequn Xiea,∗
aCollege of Computer Science and Technology, Zhejiang University, Hangzhou, 310027, China
ARTICLE INFO
Keywords:
Large language models
Retrieval-augmented generation
Financial question answering
Agentic AI
Program-of-thought reasoning
Expert systemsABSTRACT
Financial document question answering (QA) demands complex multi-step numerical reasoning
over heterogeneous evidence—structured tables, textual narratives, and footnotes—scattered across
corporate filings. Existing retrieval-augmented generation (RAG) approaches adopt a single-pass
retrieve-then-generate paradigm that struggles with the compositional reasoning chains prevalent in
financialanalysis.WeproposeFinAgent-RAG,anagenticRAGframeworkthatorchestratesiterative
retrieval-reasoningloopswithself-verification,specificallyengineeredfortheprecisionrequirements
of financial numerical reasoning. The framework integrates three domain-specific innovations: (1)
aContrastive Financial Retrievertrained with hard negative mining to distinguish semantically
similar but numerically distinct financial passages, (2) aProgram-of-Thoughtreasoning module that
generatesexecutablePythoncodeforprecisearithmeticratherthanrelyingonerror-proneLLM-based
mental computation, and (3) anAdaptive Strategy Routerthat dynamically allocates computational
resources based on question complexity, reducing API costs by 41.3% on FinQA while preserving
accuracy. Extensive experiments on three benchmark datasets—FinQA, ConvFinQA, and TAT-
QA—demonstrate that FinAgent-RAG achieves 76.81%, 78.46%, and 74.96% execution accuracy
respectively, outperforming the strongest baseline by 5.62–9.32 percentage points. A systematic
design space study across four retriever types and four reasoning modes reveals that domain-adapted
components contribute complementary gains. Ablation studies, cross-backbone evaluation with four
LLMs, and deployment cost analysis confirm the framework’s robustness and practical viability
for financial institutions. Our work demonstrates that systematic integration of domain-specific
retrieval,executablereasoning,andadaptiveresourceallocationwithinaniterativeagenticloopyields
substantial gains for financial document analysis, and provides actionable deployment guidelines for
production environments.
1. Introduction
Financialdocumentquestionanswering(QA)hasemerged
as a critical task in financial technology, with the goal
of automatically extracting and reasoning over informa-
tion embedded in corporate filings, earnings reports, and
regulatory documents (Chen, Chen, Smiley, Shah, Borber,
Mouber and Wang, 2021; Chen, Zhou, Hua, Xin, Chen, Li,
ZhuandLiang,2024).Unlikegeneral-domainQA,financial
QA presents unique challenges: questions often require
multi-step numerical reasoning (e.g., computing year-over-
year growth rates), integration of evidence from both tex-
tual narratives and structured tables, and domain-specific
knowledge about financial concepts and metrics (Chen, Li,
Smiley,Ma,ShahandWang,2022;Zhu,Lei,Huang,Wang,
Zhang, Lv, Feng and Chua, 2021). Consider the following
motivating example from a real earnings report:
Question: “What was the compound annual
growthrate(CAGR)ofoperatingexpensesfrom
2018 to 2020?”
Answering this question requires: (1) locating the oper-
ating expenses for 2018 in a financial table, (2) locating
the operating expenses for 2020, potentially in a different
table or section, (3) applying the CAGR formula CAGR=
∗Co-corresponding authors.
Email addresses:shuyang@zju.edu.cn(Y. Shu);22412284@zju.edu.cn
(Y. Liu);zqxie@zju.edu.cn(Z. Xie)
ORCID(s):0009-0008-7053-8988(Y. Shu)(𝑉final∕𝑉begin)1∕𝑛−1, and (4) verifying that the identified
valuescorrespondtothecorrectlineitems.Asingleretrieval
pass may only surface one of the two required values, and
the generator may hallucinate the missing value or apply
an incorrect formula. Figure 1 contrasts the failure mode of
single-pass RAG with the iterative strategy of our proposed
approach on this example.
Largelanguagemodels(LLMs)haveadvancedfinancial
NLP(Wu,Irsoy,Lu,Daber,Dredze,Gehrmann,Kambadur,
Rosenberg and Mann, 2023; Yang, Uy and Huang, 2020;
Shah,Kuber,Lee,NishiandVig,2022),butdirectlyprompt-
ing them for financial QA suffers from three limitations:
lack of access to current financial data, hallucination in
complex numerical calculations (Huang, Yu, Ma, Zhong,
Feng, Wang, Chen, Peng, Feng, Qin and Liu, 2025; Imani,
Du and Shrivastava, 2023), and unreliable multi-step rea-
soning (Wei, Wang, Schuurmans, Bosma, Ichter, Xia, Chi,
LeandZhou,2022).Retrieval-augmentedgeneration(RAG)
addresses the knowledge-cutoff issue (Lewis, Perez, Pik-
tus, Petroni, Karpukhin, Goyal, Küttler, Lewis, Yih, Rock-
täschel, Riedel and Kiela, 2020; Guu, Lee, Tung, Pasupat
and Chang, 2020), but standard single-pass RAG may miss
critical evidence scattered across multiple sections of a
financial report (Liu, Lin, Hewitt, Paranjape, Bevilacqua,
PetroniandLiang,2024;Gao,Xiong,Gao,Jia,Pan,Bi,Dai,
SunandWang,2024),andprovidesnomechanismtoassess
answer reliability.
Y. Shu, Y. Liu and Z. Xie:Preprint submitted to ElsevierPage 1 of 19

FinAgent-RAG for Financial Document QA
Figure 1:Motivating comparison between single-pass RAG and FinAgent-RAG on a financial CAGR question.
RecentadvancesinagenticAI—whereLLM-basedagents
iterativelyplan,act,andreflect(Yao,Zhao,Yu,Du,Shafran,
Narasimhan and Cao, 2023; Shinn, Cassano, Gopinath,
Narasimhan and Yao, 2023)—offer a promising direction.
AgenticRAGextendsthisparadigmwithiterativeretrieval-
reasoningloops(Singh,Ehtesham,Kumar,KhoeiandVasi-
lakos,2025;Xi,Chen,Guo,He,Ding,Hong,Zhang,Wang,
Jin, Zhou et al., 2023), but remains largely unexplored in
the financial domain, where structured data and numerical
precision requirements make it particularly applicable.
Based on the above observations, this paper addresses
the following research questions:
•RQ1: Can an agentic RAG framework with iterative
retrieval-reasoning loops significantly improve finan-
cial document QA compared to single-pass RAG and
generic agentic approaches?
•RQ2: How does each domain-specific component—
contrastivefinancialretrieval,program-of-thoughtrea-
soning,self-verification,andadaptiverouting—contribute
to overall performance?
•RQ3:Howdodifferentcombinationsofretrieverand
reasoning strategies interact in the design space of
financial QA systems?
•RQ4: What are the computational trade-offs of itera-
tiveagenticprocessing,andhowcandeploymentcosts
be reduced for production financial applications?
To address these challenges, we proposeFinAgent-
RAG, an agentic RAG framework specifically designed for
financial document QA. While the individual techniques
we employ—contrastive learning, program-aided reason-
ing, iterative retrieval—are established in the general NLP
literature, their systematic integration and domain-specific
adaptation for financial QA is non-trivial and has not been
systematically studied. The primary contribution of this
work is the principled combination and domain adaptation
of these components; the iterative agentic loop serves asthe orchestration backbone that enables their synergy. Our
specific contributions are:
1.ContrastiveFinancialRetriever:Adomain-adapted
dense retriever trained with four types of hard nega-
tives (temporal, metric-swap, granularity, and entity-
swap) to distinguish semantically similar but numer-
ically distinct financial passages—a critical failure
mode where generic retrievers confuse “operating in-
come” with “operating expenses” or “2019 Q3” with
“2020 Q3”. The contrastive retriever improves Re-
call@5 by 9.71 percentage points over generic dense
retrieval.
2.Program-of-Thought Financial Reasoning: Instead
of relying on LLM mental arithmetic, our reasoner
generatesexecutablePythoncodefornumericalcom-
putations, which is executed in a sandboxed environ-
ment with verification. This approach reduces arith-
metic errors—the dominant failure mode in financial
QA, accounting for 38.8% of base system errors—by
88.0%.
3.Adaptive Strategy Router: A lightweight classifier
that predicts question complexity and routes simple
questionstoacost-efficientsingle-passpathwhiledi-
recting complex multi-step questions through the full
agentic loop. On FinQA, the router reduces average
API costs by 41.3% while maintaining 98.2% of full-
system accuracy, addressing a key practical concern
for production deployment.
4.SystematicDesignSpaceStudy:Weconducta4×4
study across retriever types (BM25, generic dense,
hybrid, financial dense) and reasoning modes (direct,
CoT, PoT, adaptive), providing a comprehensive em-
pirical map of how retrieval and reasoning compo-
nents interact in financial QA systems. This study
offers actionable guidance for practitioners selecting
configurations under different resource constraints.
We evaluate FinAgent-RAG on three established finan-
cial QA benchmarks: FinQA (Chen et al., 2021), Con-
vFinQA (Chen et al., 2022), and TAT-QA (Zhu et al.,
Y. Shu, Y. Liu and Z. Xie:Preprint submitted to ElsevierPage 2 of 19

FinAgent-RAG for Financial Document QA
2021). Experimental results demonstrate that FinAgent-
RAG achieves 76.81%, 78.46%, and 74.96% execution ac-
curacy on the three benchmarks respectively, outperform-
ing the strongest iterative baseline (IterRAG) by 5.62–
9.32 percentage points. Comprehensive ablation studies,
cross-backbone evaluation with four LLMs, per-question-
type analysis, and deployment cost analysis confirm the
framework’s robustness and practical viability. Our error
analysisrevealsthatFinAgent-RAGshiftstheresidualerror
distribution from arithmetic-dominated (38.8%) to data-
extraction-dominated (29.6%), suggesting clear directions
for future improvement.
2. Related Work
2.1. Financial Question Answering
Several benchmarks have driven progress in financial
QA: FinQA (Chen et al., 2021) introduced 8,281 question-
answer pairs requiring numerical reasoning over S&P 500
reports; ConvFinQA (Chen et al., 2022) extended this to
multi-turn conversations; TAT-QA (Zhu et al., 2021) mixed
tabular and textual evidence; and FinTextQA (Chen et al.,
2024) addressed long-form financial questions. Domain-
specific language models—FinBERT (Araci, 2019; Yang
et al., 2020), BloombergGPT (Wu et al., 2023), FLANG
(Shah et al., 2022), and instruction-tuned financial LLMs
(Xie, Han, Zhang, Lai, Peng, Lopez-Lira and Huang, 2023,
2024)—have advanced financial NLP, while table under-
standing methods such as TaPas (Herzig, Nowak, Müller,
PiccinnoandEisenschlos,2020)andTaBERT(Yin,Neubig,
YihandRiedel,2020)addresstabularreasoningbutassume
pre-identified tables.
Despite these advances, existing approaches either rely
on supervised fine-tuning with expensive annotated data,
or on single-pass prompting that cannot handle multi-step
reasoning chains prevalent in financial QA. FinAgent-RAG
addresses this gap through an agentic loop that iteratively
refines retrieval and reasoning without LLM fine-tuning.
2.2. Retrieval-Augmented Generation
RAG was introduced by Lewis et al. (2020) to ground
LLM outputs in retrieved external knowledge, building
on retrieval-augmented pre-training (Guu et al., 2020).
The standard pipeline pairs a dense retriever (Karpukhin,
Oguz, Min, Lewis, Wu, Edunov, Chen and Yih, 2020)
with a generator; subsequent work improved this through
multi-passage fusion (Izacard and Grave, 2021), large-scale
retrieval (Borgeaud, Mensch, Hoffmann, Cai, Rutherford,
Millican, van den Driessche, Lespiau, Damoc, Clark et al.,
2022), query rewriting (Ma, Gong, He, Zhao and Duan,
2023), passage reranking (Glass, Rossiello, Chowdhury,
Naber, Nishi and Gliozzo, 2022), active retrieval (Jiang,
Xu, Gao, Sun, Liu, Dwivedi-Yu, Yang, Callan and Neubig,
2023),andconversationalinteractionparadigmsforretrieval
(Xie, Wang, Wang, Cai, Wang and Jin, 2025). Self-RAG
(Asai, Wu, Wang, Sil and Hajishirzi, 2024) introduced re-
flectiontokensforadaptiveretrievaldecisions,whileCRAG(Yan,Gu,ZhuandLing,2024)proposedcorrectiveretrieval
that triggers web search when needed.
However, these methods remain fundamentally single-
cycle: Self-RAG decideswhetherto retrieve but does not
iterativelyrefinequeriesbasedonreasoningfailures,CRAG
lacks the structured decomposition needed for multi-step
financial reasoning, and ensemble-based approaches such
as VOTE-RAG (Xie and Sun, 2026) mitigate compounded
hallucinations through parallel voting but do not address
domain-specificretrievalchallenges.Liuetal.(2024)further
showedthatmodelsstrugglewithinformationinthemiddle
oflongcontexts,underscoringtheneedfortargetedevidence
selection. FinAgent-RAG unifies iterative retrieval, struc-
turedreasoning,andself-verificationintoacoherentagentic
loop tailored for financial QA.
2.3. Agentic AI and LLM Agents
LLM-basedagentshaveadvancedalongtwoaxes:reasoning-
acting integration—ReAct (Yao et al., 2023), Reflexion
(Shinn et al., 2023)—and tool augmentation—Toolformer
(Schick, Dwivedi-Yu, Dessì, Raileanu, Lomeli, Hambro,
Zettlemoyer,CanceddaandScialom,2023),ToolLLM(Qin,
Liang, Ye, Zhu, Yan, Lu, Lin, Cong, Tang, Qian et al.,
2024).Recentsurveys(Singhetal.,2025;Wang,Ma,Feng,
Zhang,Yang,Zhang,Chen,Tang,Chen,Lin,Zhao,Weiand
Wen,2024;Xietal.,2023)haveformalizedagenticRAGas
a paradigm where agents iteratively plan retrieval, reason
over evidence, and self-correct. In the financial domain,
agents have been applied to trading (Xiao, Sun, Luo and
Wang, 2024) and portfolio management (Yu, Li, Chen,
Jiang, Li, Zhang, Liu, Suchow and Khaldoun, 2024), but
agentic financial QA remains unexplored. Program-aided
reasoning—PoT (Chen, Ma, Wang and Cohen, 2023) and
PAL (Gao, Madaan, Zhou, Alon, Liu, Yang, Callan and
Neubig,2023)—substantiallyoutperformschain-of-thought
on mathematical tasks by generating executable code, but
has not been adapted to financial-specific formula patterns.
Existingagenticsystems(ReAct,Reflexion)lackdomain-
specific components essential for financial QA—such as
financial query decomposition, table-aware retrieval, and
numerical consistency verification. Our work bridges this
gapbyintegratingdomain-adaptedmoduleswithinanagen-
tic loop, including a PoT reasoner tailored to financial
computation patterns.
2.4. Summary
Table1summarizesthekeydifferencesbetweenFinAgent-
RAG and existing approaches.
3. Methodology
3.1. Problem Formulation
Given a financial question𝑞and a corpus of financial
documents= {𝑑1,𝑑2,…,𝑑𝑁}, where each document𝑑𝑖
contains textual narratives𝑡𝑖and structured tables𝜏𝑖, the
goal is to generate an accurate answer𝑎that may involve
numerical computation over evidence extracted from.
Y. Shu, Y. Liu and Z. Xie:Preprint submitted to ElsevierPage 3 of 19

FinAgent-RAG for Financial Document QA
Table 1
Comparison of FinAgent-RAG with related approaches across six capability dimensions.✓= supported,×= not supported,◦
= partial support.†Our constructed baseline following Shao, Gong, Shen, Huang, Duan and Chen (2023); see Section 4.2 for
details.
Method Iterative Query Self- Financial Table Confidence
Retrieval Decomp. Verify Domain Aware Calibration
Naive RAG (Lewis et al., 2020)× × × × × ×
REALM (Guu et al., 2020)× × × × × ×
FiD (Izacard and Grave, 2021)× × × ×◦×
Self-RAG (Asai et al., 2024)◦×✓× × ×
CRAG (Yan et al., 2024)× ×✓× ×◦
Active RAG (Jiang et al., 2023)✓× × × × ×
ReAct (Yao et al., 2023)✓◦× × × ×
Reflexion (Shinn et al., 2023)✓×✓× × ×
IterRAG†✓× × × × ×
FinAgent-RAG (Ours)✓ ✓ ✓ ✓ ✓ ✓
Figure 2:Overall architecture of FinAgent-RAG.
Formally, we seek:
𝑎∗=argmax
𝑎𝑃(𝑎∣𝑞,)(1)
3.2. System Overview
Financialdocumentspresentuniquepreprocessingchal-
lenges due to their heterogeneous structure (narrative sec-
tions,structuredtables,footnotes,appendices).Wedesigna
three-stage pipeline: (1) document parsing into typed seg-
ments (text, tables, headers); (2) table linearization using a
header-prependedrowstrategythatpreservescolumn–value
associations critical for distinguishing metrics across time
periods; and (3) passage chunking (512 tokens, 64-token
overlap) with dense encoding viabge-base-en-v1.5(Xiao,
Liu, Zhang and Muennighoff, 2023) and FAISS indexing
(Johnson,DouzeandJégou,2019).Thefinalindexcontains
approximately50,000–80,000passagespercorpus.Fullpre-
processing details are provided in Appendix B.
FinAgent-RAG operates through an iterative loop com-
prisingsixcoremodules:(1)QueryDecomposer,(2)Adap-
tive Retriever with Contrastive Financial Retriever, (3)
Chain-of-Thought(CoT)Reasoner,(4)Program-of-Thought
(PoT) Reasoner, (5) Adaptive Strategy Router, and (6) Self-
Verifier with Query Refiner. Figure 2 illustrates the overall
architecture.Algorithm 1 presents the complete procedure.
3.3. Query-Adaptive Retrieval
Complex financial questions often involve multiple rea-
soningsteps.Forexample,“Whatwasthepercentagechange
in total revenue from 2019 to 2020?”requires: (1) finding
totalrevenuefor2019,(2)findingtotalrevenuefor2020,and
(3) computing the percentage change. The Query Decom-
poser breaks𝑞into an ordered sequence of sub-questions
𝐒=[𝑠1,𝑠2,…,𝑠𝑚],whereeach𝑠𝑖targetsaspecificevidence
retrieval or computation step.
We implement the decomposer using an LLM with a
structured prompt (the full prompt template is provided in
Appendix A):
𝐒=LLM(promptdecompose,𝑞)(2)
where promptdecompose instructs the model to identify re-
quireddatapointsandcomputationsteps,explicitlyhandling
financial-specific patterns such as year-over-year compar-
isons, ratio calculations, and cumulative aggregations.
We identify five common decomposition patterns in
financial QA that guide our prompt design:
Y. Shu, Y. Liu and Z. Xie:Preprint submitted to ElsevierPage 4 of 19

FinAgent-RAG for Financial Document QA
Algorithm 1FinAgent-RAG Procedure
Require:Financial question𝑞, document corpus, max
iterations𝐾
Ensure:Answer𝑎
1:𝐒←QueryDecompose(𝑞){Decompose into sub-
questions}
2:←∅{Evidence buffer}
3:for𝑘=1to𝐾do
4:foreach sub-question𝑠𝑖∈𝐒do
5:𝑖←Retrieve(𝑠𝑖,){Adaptive retrieval}
6:←∪𝑖
7:end for
8:𝑟←Router(𝑞){Route to CoT or PoT}
9:𝑎𝑘,𝑐𝑘←Reason(𝑟,𝑞,){CoT or PoT reasoning}
10:𝑣𝑘←SelfVerify(𝑞,𝑎𝑘,){Verify answer}
11:if𝑣𝑘=ACCEPTor𝑐𝑘>𝜃then
12:return𝑎𝑘
13:else
14:𝐒←RefineQuery(𝑞,𝑎𝑘,𝑣𝑘,){Refine and re-
iterate}
15:end if
16:end for
17:return𝑎𝐾{Return best answer after max iterations}
•Temporal comparison: Split by time period (e.g.,
“revenue in 2019”→“revenue in 2020”→compute
change).
•Ratio computation: Separate numerator and denom-
inator retrieval (e.g., “gross margin”→“gross profit”
÷“revenue”).
•Multi-entity aggregation: Retrieve values for each
entity,thenaggregate(e.g.,“totalexpenses”=sumof
individual expense line items).
•Conditionalfiltering:Applyaconditionbeforecom-
putation (e.g., “revenue from the segment with the
highest growth rate”).
•Derivedmetrics:Decomposeintocomponentmetrics
(e.g., “EBITDA” = “operating income” + “deprecia-
tion” + “amortization”).
Each sub-question𝑠𝑖is tagged as eitherretrievalor
computation, informing downstream modules whether to
triggerevidenceretrievalorapplyreasoningoverpreviously
collected passages.
AdaptiveRetriever.Foreachsub-question𝑠𝑖,theAdap-
tive Retriever performs dense passage retrieval against the
indexed financial document corpus. We encode documents
at the paragraph level, treating table rows as separate pas-
sages with column headers prepended for context. The re-
trieval score is computed as:
score(𝑠𝑖,𝑑𝑗)=sim(𝐞𝑠𝑖,𝐞𝑑𝑗)(3)where𝐞𝑠𝑖and𝐞𝑑𝑗aredenseembeddingsofthesub-question
and document passage, respectively, and sim(⋅,⋅)denotes
cosine similarity.
A key feature of our retriever is itsadaptivenature: in
subsequentiterations(𝑘>1),theretrieverisconditionedon
previously retrieved evidence and the verification feedback,
enabling it to target missing information:
(𝑘)
𝑖=Retrieve(𝑠(𝑘)
𝑖,⧵(𝑘−1))(4)
This prevents redundant retrieval and encourages explo-
ration of the document corpus.
In practice, we implement ahybrid retrievalstrategy
thatcombinesdenseretrievalwithsparsekeywordmatching.
For sub-questions containing specific financial terms (e.g.,
“operating expenses”, “depreciation”), we boost passages
containing exact term matches by a factor𝛼=0.3:
scorehybrid(𝑠𝑖,𝑑𝑗)=(1−𝛼)⋅sim(𝐞𝑠𝑖,𝐞𝑑𝑗)
+𝛼⋅BM25(𝑠𝑖,𝑑𝑗)(5)
This hybrid approach addresses a common failure mode
where dense retrieval returns semantically similar but fac-
tually distinct passages (e.g., “operating income” instead of
“operating expenses”).
Evidence Buffer Management.The evidence buffer
maintains retrieved passages across iterations with dedupli-
cation. When the buffer exceeds a maximum capacity of 15
passages,weapplyarelevance-recencyscoringtoretainthe
most useful passages:
priority(𝑑𝑗)=score(𝑠𝑖,𝑑𝑗)+𝛽⋅iteration(𝑑𝑗)∕𝐾(6)
where𝛽= 0.2weights more recently retrieved passages
slightly higher, reflecting the insight that later iterations
target more specific evidence gaps.
Contrastive Financial Retriever.A fundamental chal-
lenge in financial document retrieval is that semantically
similar passages can carry numerically distinct values. For
instance, “operating expenses in Q3 2019” and “operating
expenses in Q3 2020” share nearly identical surface forms
but contain different numerical values. Generic dense re-
trievers frequently confuse such passages, leading to incor-
rect downstream reasoning.
To address this, we train aContrastive Financial Re-
trieverusinghardnegativeminingwithfourtypesoffinancially-
motivated negative examples:
•Temporalnegatives:Passagesaboutthesamemetric
from a different fiscal period (e.g., 2019 vs. 2020).
•Metric-swap negatives: Passages about a different
but related metric from the same period (e.g., “rev-
enue” vs. “cost of revenue”).
•Granularity negatives: Passages at a different ag-
gregationlevel(e.g.,segment-levelvs.company-level
revenue).
Y. Shu, Y. Liu and Z. Xie:Preprint submitted to ElsevierPage 5 of 19

FinAgent-RAG for Financial Document QA
Figure 3:Training pipeline of the Contrastive Financial Retriever with four types of domain-specific hard negatives.
•Entity-swapnegatives:Passagesaboutthesamemet-
ric from a different entity (e.g., subsidiary vs. parent
company).
Given a query-passage pair(𝑞,𝑑+)and a set of hard
negatives{𝑑−
1,𝑑−
2,…,𝑑−
𝑛}, the contrastive loss is:
contrast=−log𝑒sim(𝐞𝑞,𝐞𝑑+)∕𝜏
𝑒sim(𝐞𝑞,𝐞𝑑+)∕𝜏+∑𝑛
𝑖=1𝑒sim(𝐞𝑞,𝐞𝑑−
𝑖)∕𝜏(7)
where𝜏=0.05is the temperature parameter.
Hard Negative Mining Heuristics.For each of the
6,251trainingqueriesinFinQA,weextractthreestructured
attributes from the gold passage: (1) themetric name(e.g.,
“operatingexpenses”),identifiedviaacuratedfinanciallex-
icon of 847 terms; (2) theentity(company, segment, or
subsidiary), extracted from document headers; and (3) the
timeperiod(fiscalyear,quarter),parsedwithregexpatterns.
Acandidatepassagequalifiesasahardnegativeifitsharesat
least two of these three attributes with the gold passage but
differsintheremainingone.Temporalnegatives(samemet-
ricandentity,differentperiod)constitute38.2%ofthemined
negatives; metric-swap negatives (same entity and period,
different metric) account for 29.5%; entity-swap negatives
(same metric and period, different entity) comprise 21.1%;
and granularity negatives (segment-level vs. consolidated
figures) account for 11.2%. This procedure yields 26,879
total hard negative pairs (an average of 4.3 per query) from
the FinQA training set.
TrainingProcedure.Wefine-tunethebge-base-en-v1.5
encoder (Xiao et al., 2023) for 3 epochs with AdamW
optimizer (learning rate2×10−5, linear warmup over 10%
of steps, weight decay 0.01) and batch size 32 on a single
NVIDIAA100GPU(approximately2hours).Trainingcon-
verges after approximately 2,400 gradient steps; the valida-
tion Recall@5 plateaus at epoch 2, with epoch 3 providing
a marginal 0.3-point improvement. The resulting retriever
achieves Recall@5 of 82.34% compared to 72.63% for the
generic(non-fine-tuned)denseretriever,animprovementof
9.71percentagepoints(Section5.12).Figure3illustratesthe
complete training pipeline.3.4. Dual-Mode Financial Reasoning
FinAgent-RAG employs two complementary reasoning
strategies—Chain-of-Thought (CoT) for interpretable step-
by-step reasoning and Program-of-Thought (PoT) for re-
liable numerical computation—dynamically selected by a
lightweight router based on question complexity.
Chain-of-Thought Reasoning.Given the accumulated
evidenceand the original question𝑞, the CoT Reasoner
generates a step-by-step reasoning chain leading to the an-
swer. We design a financial-specific CoT prompt that in-
structs the LLM to:
1. Identify the relevant numerical values from the re-
trieved evidence.
2. Mapeachvaluetothecorrespondingfinancialconcept
(e.g., revenue, cost, margin).
3. Formulate the required computation as an explicit
mathematical expression.
4. Execute the computation step by step.
5. State the final answer with appropriate units and pre-
cision.
Formally:
(𝑎𝑘,𝑐𝑘)=LLM(promptCoT,𝑞,)(8)
where𝑎𝑘is the generated answer and𝑐𝑘∈ [0,1]is the
model’s self-assessed confidence score.
Confidence Calibration.Raw LLM confidence scores
are often poorly calibrated (Kadavath, Conerly, Askell,
Henighan, Drain, Perez, Schiefer, Hatfield-Dodds, Das-
Sarma, Tran-Johnson et al., 2022). We apply a simple post-
hoc calibration: we bin validation set predictions by their
raw confidence scores and compute the empirical accuracy
within each bin. A monotonic isotonic regression maps raw
scores to calibrated probabilities. During inference,𝑐𝑘is
replaced by its calibrated estimatê 𝑐𝑘, which is used for
the termination decision in Algorithm 1. This calibration
step ensures that the threshold𝜃has a consistent semantic
meaning across different question types.
Y. Shu, Y. Liu and Z. Xie:Preprint submitted to ElsevierPage 6 of 19

FinAgent-RAG for Financial Document QA
Financial Reasoning Patterns.We design the CoT
prompt to handle four common financial computation pat-
terns:(1)percentagechange:(𝑉new−𝑉old)∕𝑉old×100;(2)ra-
tio computation: numerator / denominator; (3)weighted av-
erage:∑𝑤𝑖𝑥𝑖∕∑𝑤𝑖;and(4)compoundgrowth:(𝑉final∕𝑉initial)1∕𝑛−
1. By providing explicit templates for these patterns, we re-
duce arithmetic errors that arise from the LLM formulating
computations from scratch.
Program-of-ThoughtReasoning.Whilechain-of-thought
reasoning improves interpretability, LLMs remain unreli-
able at executing multi-step arithmetic—our error analysis
reveals that 38.8% of base system errors are attributable
to incorrect numerical computation (Section 5.5). Inspired
by recent work on program-aided reasoning (Chen et al.,
2023;Gaoetal.,2023),weintroduceaProgram-of-Thought
(PoT)reasoningmodulethatseparatesreasoning(performed
by the LLM) fromcomputation(performed by a Python
interpreter).
Given the evidenceand question𝑞, the PoT module
generates executable Python code:
code𝑘=LLM(promptPoT,𝑞,)(9)
The generated code is executed in a sandboxed Python
environment with restricted built-ins (only mathematical
operations, basic data structures, and string processing are
permitted). The sandbox enforces a 5-second timeout and
prohibitsfileI/O,networkaccess,anddynamiccodeexecu-
tion (exec/eval).
Code Verification and Repair.The PoT module in-
cludes a three-stage verification pipeline:
1.Static analysis: The generated code is checked for
syntactic correctness and disallowed operations be-
fore execution.
2.Runtime verification: After execution, the output is
checkedfortypecorrectness(numericresultexpected)
and reasonable range (financial values typically fall
withinpredictableboundsbasedontheinputcontext).
3.Self-repair: If execution fails or produces an invalid
result, the LLM is provided with the error message
and asked to generate corrected code, with up to two
repair attempts.
The PoT module operates as an alternative reasoning
path to CoT. The Adaptive Strategy Router (Section 3.4)
determines which reasoning mode to apply based on ques-
tion characteristics. Figure 4 illustrates the end-to-end PoT
pipeline from input to verified output.
Adaptive Strategy Router.Not all financial questions
require the full iterative agentic loop. Simple lookup ques-
tions (“What was the total revenue in 2020?”) can be an-
sweredaccuratelyinasingleretrieval-generationpass,while
complexmulti-stepquestions(“Whatwasthecompoundan-
nualgrowthrateofoperatingexpensesfrom2018to2020?”)
benefit from iterative refinement. Processing all questions
throughthefullpipelinewastescomputationalresourcesand
increases latency without improving accuracy.The Adaptive Strategy Router is a lightweight classifier
that predicts question complexity and routes each question
to the appropriate processing path:
𝑟=Router(𝑞)∈{simple,complex}(10)
The router extracts four feature groups from the input
question, yielding a 12-dimensional feature vector: (1)syn-
tactic features(4 dims): question length (tokens), presence
of comparative/superlative phrases, number of financial en-
tities mentioned, and number of distinct numerical values
in the question; (2)decomposition features(2 dims): the
number of sub-questions generated by the Query Decom-
poser and the maximum depth of the decomposition tree;
(3)temporalfeatures(3dims):countofdistincttimeperiods
referenced,presenceofyear-over-yearcomparisonpatterns,
and temporal span (years); and (4)computation features(3
dims): one-hot encoding of the implied computation type
(lookup, single-step arithmetic, multi-step arithmetic).
RouterTraining.Weimplementtherouterasagradient-
boosted decision tree (LightGBM with 50 estimators, max
depth4)trainedontheFinQAvalidationset(883examples).
Ground-truth labels are derived by running each question
through both the simple and full pipelines: a question is
labeledcomplexif the full agentic loop improves accuracy
over the single-pass path on that example. This yields a
58:42 simple-to-complex ratio. The router is trained with
5-fold cross-validation, achieving 87.34% routing accuracy
(91.2% precision forcomplex, 84.1% forsimple). The router
adds negligible latency (<5ms per question) as it requires
only feature extraction and a single classifier inference.
Questionsclassifiedassimpleareprocessedwithasingle
retrieval pass and direct PoT reasoning (no iteration), while
complexquestions are routed through the full agentic loop
with up to𝐾iterations. The overall effect is a 41.3% re-
duction in average API calls (from 5.83 to 3.42) with only
a 1.34% accuracy decrease (from 76.81% to 75.47%), rep-
resentingafavorablecost–accuracytrade-offforproduction
deployment.
3.5. Self-Verification and Iterative Refinement
TheSelf-Verifierassessesthereliabilityofthegenerated
answer through three verification checks:
Evidence Sufficiency Check: Determines whether the
retrieved evidence contains all necessary data points for
answering the question.
𝑣suff=LLM(promptsuff,𝑞,𝑎𝑘,)(11)
NumericalConsistencyCheck:Verifiesthatthenumer-
ical computations in the reasoning chain are arithmetically
correct by re-executing the identified operations.
𝑣num=VerifyArithmetic(𝑎𝑘,reasoning_chain𝑘)(12)
Cross-Evidence Validation: Checks whether the an-
swer is consistent across multiple pieces of retrieved evi-
dence, flagging contradictions.
𝑣cross=LLM(promptcross,𝑎𝑘,)(13)
Y. Shu, Y. Liu and Z. Xie:Preprint submitted to ElsevierPage 7 of 19

FinAgent-RAG for Financial Document QA
Figure 4:Program-of-Thought (PoT) reasoning pipeline with sandboxed code execution.
The overall verification decision is:
𝑣𝑘={
ACCEPTif𝑣suff∧𝑣num∧𝑣cross
REJECTotherwise(14)
When𝑣𝑘=REJECT, the Query Refiner generates refined
sub-questions based on the verification feedback:
𝐒(𝑘+1)=LLM(promptrefine,𝑞,𝑎𝑘,𝑣𝑘,)(15)
This enables targeted re-retrieval addressing the specific
deficiencies identified during verification.
Convergence Properties.An important design consid-
eration is whether the iterative loop converges to a stable
answer. We provide three design mechanisms that promote
convergence.
MonotonicEvidenceAccumulation.Theevidencebuffer
growsmonotonicallyacrossiterationsthroughdeduplica-
tion. As the buffer accumulates evidence, the rate ofnovel
passage retrieval (passages not previously in) naturally
decreases, providing a diminishing-return dynamic that
drivesthesystemtowardtermination.Theconfidence-based
termination criterion (𝑐𝑘> 𝜃) captures this behavior: as
evidence becomes more complete, the reasoner’s calibrated
confidence increases, triggering acceptance.
Exclusion-Based Exploration.The adaptive retriever
conditions on previously retrieved evidence (Eq. 4), ex-
cludingalready-retrievedpassages.Thispreventsthesystem
from entering retrieval loops and encourages exploration of
diverse document regions in each iteration.
Graceful Degradation.When the maximum iteration
budget𝐾is reached without acceptance, we return the
answer𝑎𝐾from the final iteration rather than the highest-
confidence answer across all iterations. This design choice
prioritizes the answer produced with the most complete
evidence set. Empirical validation of convergence behavior
is provided in Section 5 (Table 12).
3.6. Conversational Extension
For conversational financial QA (ConvFinQA), where
questions build upon previous turns through coreferenceand implicit context, we extend the agentic loop with two
mechanisms for managing cross-turn state.
Persistent Evidence Buffer.The evidence buffer
persists across conversation turns rather than resetting at
each turn. When processing turn𝑡, the agent inherits(𝑡−1)
from the previous turn, enabling it to reuse previously
retrieved passages without redundant retrieval. To prevent
unboundedgrowth,weapplytherelevance-recencypruning
(Section 3.3) at the start of each new turn, retaining the
top-10 passages by priority score. This design reflects the
observation that later questions in a conversation frequently
reference values retrieved in earlier turns.
ConversationHistoryEncoding.Thefullconversation
history—comprisingpreviousquestions,generatedanswers,
and verification outcomes—is prepended to the current
question as structured context before query decomposition.
Specifically, for turn𝑡with question𝑞𝑡, the effective input
totheQueryDecomposeris[𝑞1,𝑎1,𝑞2,𝑎2,…,𝑞𝑡−1,𝑎𝑡−1,𝑞𝑡],
enablingcoreferenceresolution(e.g.,resolving“thismetric”
to a specific financial term mentioned in a prior turn). The
conversation history is also provided to the Self-Verifier to
enable cross-turn consistency checking—for example, ver-
ifying that a computed year-over-year change is consistent
withtheindividual-yearvaluesretrievedinearlierturns.The
agenticloopresetsitsiterationcounterateachturn(i.e.,each
turnmayiterateupto𝐾timesindependently),buttheshared
evidence buffer ensures continuity of retrieved knowledge
across the conversation.
4. Experimental Setup
4.1. Datasets
We evaluate FinAgent-RAG on three established finan-
cial QA benchmarks:
FinQA(Chen et al., 2021): Contains 8,281 expert-
annotated QA pairs derived from the earnings reports of
S&P 500 companies. Each question requires numerical
reasoning overa combination offinancial tables andtextual
Y. Shu, Y. Liu and Z. Xie:Preprint submitted to ElsevierPage 8 of 19

FinAgent-RAG for Financial Document QA
context. The dataset is split into 6,251 training, 883 vali-
dation, and 1,147 test examples. Following prior work, we
report results on the test set.
ConvFinQA(Chenetal.,2022):AnextensionofFinQA
totheconversationalsetting,containing3,892conversations
with an average of 3.6 turns per conversation. Questions
reference previous turns, requiring coreference resolution
andcontexttracking.Thetestsetcontains421conversations.
TAT-QA(Zhu et al., 2021): A hybrid QA dataset con-
taining16,552questionsover2,757financialreports,requir-
ingreasoningoverbothtabularandtextualevidence.Unlike
FinQA, TAT-QA includes diverse answer types (span ex-
traction,counting,andarithmetic)andrequiresjointtabular-
textualunderstanding.Thetestsetcontains1,668questions.
We include TAT-QA to evaluate generalization to a struc-
turallydifferentfinancialQAbenchmarkwithheterogeneous
answer types.
4.2. Baselines
We compare FinAgent-RAG against eight baselines
spanning four categories:
Non-retrieval baselines:
•Zero-shot LLM: Directly prompting the LLM with
the question and the full document context, without
retrieval-based evidence selection.
•PIXIU/FinMA(Xieetal.,2023):Afinancialinstruction-
tuned LLM (FinMA) fine-tuned on diverse financial
tasks via the PIXIU benchmark.
Standard RAG baselines:
•Naive RAG: Single-pass dense retrieval (top-5 pas-
sages) followed by generation.
•Advanced RAG: Dense retrieval with BM25 rerank-
ing and query expansion, followed by generation.
Self-correcting RAG baselines:
•Self-RAG(Asai et al., 2024): RAG with reflection
tokens for adaptive retrieval and self-critique.
•CRAG(Yan et al., 2024): Corrective RAG with re-
trieval quality assessment and web search fallback.
Agentic baselines:
•ReAct(Yao et al., 2023): Interleaved reasoning and
acting with tool use, adapted to financial QA with
retrieval as an action.
•IterRAG:Weconstructastrongiterativebaselinefol-
lowing the retrieval-generation synergy paradigm of
Shao et al. (2023). IterRAG performs𝐾=3rounds of
retrieval-then-generate: at each iteration𝑘, the model
appendsitspreviousanswertotheoriginalqueryasan
expanded retrieval query, retrieves top-5 passages us-
ingthegenericdenseretriever(bge-base-en-v1.5with-
outcontrastive fine-tuning), and generates a new an-
swerviachain-of-thoughtprompting.UnlikeFinAgent-
RAG, IterRAG omits query decomposition, domain-
adaptedretrieval,PoTreasoning,self-verification,andadaptive routing. This baseline isolates the general
benefit of iterative retrieval from our domain-specific
innovations.
All baselines use DeepSeek-V3 as the LLM backbone
(exceptFinMA,whichusesitsownfine-tunedmodel)toen-
surefaircomparison.Were-implementbaselines(marked†)
rather than citing published numbers because: (1) prior
results use heterogeneous LLM backbones, making cross-
method comparison unreliable; (2) our distractor-rich re-
trieval corpusdiffers fromthe gold-contextsetting assumed
in some original papers; and (3) controlling the LLM back-
bone isolates the contribution of the retrieval-reasoning ar-
chitecture itself. For Self-RAG and CRAG, we adapt their
official open-source implementationsto financial document
QA by integrating them with our document preprocessing
andretrievalinfrastructure;forReAct,wefollowtheoriginal
prompting protocol of Yao et al. (2023).
Contextualizing Reproduced Baselines.To help read-
ers assess reproductionfidelity, we note the following refer-
encepointsfrom originalpublications(evaluatedunder dif-
ferentconditions—goldcontext,differentLLMs,ordifferent
splits):FinQANet(Chenetal.,2021)reports61.24%(super-
vised, gold context); Self-RAG (Asai et al., 2024) reports
54.9%onopen-domainNQ(notfinancialQA);CRAG(Yan
et al., 2024) reports improvements on PopQA and PubQA
(nofinancialbenchmark).Directnumericalcomparisonwith
these figures is not meaningful due to differing evaluation
protocols, but they provide context for the overall perfor-
mance landscape. We acknowledge that author-reproduced
baselines are a limitation and commit to releasing all base-
line reproduction code, hyperparameter configurations, and
prompt templates alongside our framework to enable inde-
pendent verification.
4.3. Implementation Details
We use DeepSeek-V3 as the backbone LLM for all
methods to ensure fair comparison. Document embeddings
are generated usingbge-base-en-v1.5(Xiao et al., 2023)
(dimension 768) and indexed with FAISS (Johnson et al.,
2019). Documents are chunked at the paragraph level (max
512 tokens, overlap 64 tokens). Tables are linearized row-
by-row with column headers prepended to each row (e.g.,
“Year: 2020 | Revenue: $5.2B | Net Income: $1.1B”).
Tables exceeding 512 tokens after linearization are split at
row boundaries, with the header row duplicated in each
chunktopreservecolumnsemantics.Forquestionsrequiring
evidence from multiple tables—which our error analysis
identifies as the dominant residual challenge (42.1% of re-
mainingerrors)—theiterativeagenticloopnaturallyhandles
cross-tableaggregation:eachiterationcanretrievepassages
from different tables into the shared evidence buffer,
and the self-verifier’s evidence sufficiency check explicitly
flags when a required data point is still missing, triggering
targeted re-retrieval in the next iteration.
Retrieval Corpus Construction.To simulate realistic
retrievalconditions,wedonotusethegoldcontextprovided
witheachbenchmarkquestion.Instead,foreachbenchmark
Y. Shu, Y. Liu and Z. Xie:Preprint submitted to ElsevierPage 9 of 19

FinAgent-RAG for Financial Document QA
Table 2
Hyperparameter configuration for FinAgent-RAG.
Parameter Value
LLM backbone DeepSeek-V3
Embedding model bge-base-en-v1.5
Embedding dimension 768
Max iterations𝐾3
Confidence threshold𝜃0.8
Top-𝑘passages 5
Chunk size (tokens) 512
Chunk overlap (tokens) 64
Evidence buffer capacity 15 passages
Hybrid weight𝛼0.3
Recency weight𝛽0.2
Decoding temperature 0 (greedy)
Max output tokens 1024
we construct a distractor-rich retrieval corpus from the full
set of source documents: FinQA yields 52,847 passages
from 8,281 earnings report pages; ConvFinQA reuses the
same FinQA corpus (as it draws from the same document
pool);TAT-QAyields79,214passagesfrom2,757financial
reports. Each question must retrieve its evidence from the
fullcorpusratherthanfromapre-identifiedgolddocument,
making our evaluation setting strictly harder than the gold-
context protocol used in some prior work. All baselines
share the same retrieval corpus and indexing infrastructure
to ensure fair comparison.
ForFinAgent-RAG,wesetthemaximumiterationcount
𝐾= 3and the confidence threshold𝜃= 0.8. The retrieval
returns the top-5 most relevant passages per sub-question.
All LLM calls use temperature 0 for deterministic outputs.
LLMinferenceisconductedviaAPIcallswithoutanyLLM
fine-tuning; only the retriever encoder is fine-tuned with
contrastive loss (Section 3.3).
Hyperparameter Configuration.Table 2 summarizes
the key hyperparameters. These values were selected based
on validation set performance (Section 5.9).
Computational Environment.LLM inference is con-
ducted via API calls. Retriever fine-tuning requires a single
GPU (NVIDIA A100, approximately 2 hours for 3 epochs).
The embedding index construction takes approximately 30
minutesforthefullFinQAcorpusonasingleCPU.Inference
on the FinQA test set (1,147 questions) requires approxi-
mately3.5hoursatanaverageof5.8APIcallsperquestion.
ThetotalAPIcostforafullevaluationrunisapproximately
$14.
4.4. Evaluation Metrics
Following Chen et al. (2021), we adopt the following
metrics:
•Execution Accuracy (Exe Acc): Whether the exe-
cutedprogramproducesthecorrectnumericalanswer
(within a tolerance of 1%).
•Program Accuracy (Prog Acc): Whether the gener-
ated reasoning program matches the gold program.
For ConvFinQA, we additionally report:•Execution Accuracy (Exe Acc): Conversation-level
accuracy, where a conversation is correct only if all
turns produce the correct answer.
•Turn-level Accuracy (Turn Acc): The fraction of
individualconversationturnsansweredcorrectly,cap-
turing the model’s ability to handle coreference reso-
lution and context carryover.
For TAT-QA, following Zhu et al. (2021), we report
Execution Accuracy andF1 score, which measures token-
level overlap between the predicted and gold answers and
is the standard metric for TAT-QA’s heterogeneous answer
types (spans, counts, and arithmetic results).
5. Experimental Results
5.1. Main Results
Tables 3–4 present the main experimental results on
three financial QA benchmarks.
FinAgent-RAGachievesthebestperformanceacrossall
metrics on all three benchmarks. On FinQA, our method
outperforms the strongest baseline (IterRAG) by 8.98% in
execution accuracy. On ConvFinQA, the improvement is
9.32%, reflecting the advantage of iterative refinement in
multi-turn conversational settings where the persistent evi-
dencebufferandconversationhistoryencoding(Section3.6)
enableeffectivecontextaccumulationacrossturns.OnTAT-
QA, the improvement is 5.62%, demonstrating generaliza-
tiontoastructurallydifferentbenchmarkwithheterogeneous
answer types.
Severalpatternsemergefromthecross-benchmarkcom-
parison. First, the progression from non-retrieval to stan-
dard RAG to self-correcting to agentic methods is consis-
tent across all three datasets, validating the value of each
paradigm advancement. Second, FinAgent-RAG’s margin
overIterRAG(thestrongestgenericagenticbaseline)ranges
from 5.62% to 9.32%, demonstrating that domain-specific
components—contrastivefinancialretrieval,PoTreasoning,
and adaptive routing—provide consistent benefits beyond
genericiterativeretrieval.Third,theimprovementsarerela-
tivelyuniformacrossbenchmarks,suggestingthatthegains
stem from fundamental architectural advantages rather than
task-specific overfitting.
Contextualizing with Published Results.The original
FinQA leaderboard includes fine-tuned models: FinQANet
(Chenetal.,2021)achieves61.24%withsupervisedtraining
on gold programs. FinAgent-RAG (76.81%) surpasses this
by 15.57 percentage pointswithout any LLM fine-tuning,
operatingthroughzero-shotLLMpromptingcombinedwith
a contrastive-tuned retriever (Section 3.3). While the com-
parison involves different LLM backbones, it demonstrates
that agentic approaches can substantially surpass fine-tuned
specialist models through intelligent orchestration of re-
trieval and reasoning.
Cross-Backbone Generalizability.To assess whether
FinAgent-RAG’s gains are backbone-dependent, we evalu-
atethefullframeworkwithfourLLMbackbonesonthefull
FinQA test set. Table 5 reports the results.
Y. Shu, Y. Liu and Z. Xie:Preprint submitted to ElsevierPage 10 of 19

FinAgent-RAG for Financial Document QA
Table 3
Main results on FinQA test set (1,147 questions). Best results are inbold, second best underlined .†= our reproduction with
DeepSeek-V3 backbone.
Category Method Exe Acc (%) Prog Acc (%)
Non-retrievalZero-shot LLM†53.27 50.14
FinMA (Xie et al., 2023) 59.63 56.71
Standard RAGNaive RAG†57.84 54.62
Advanced RAG†61.52 58.34
Self-correctingSelf-RAG†64.75 61.68
CRAG†66.21 63.14
AgenticReAct†63.18 60.05
IterRAG†67.83 64.76
FinAgent-RAG (Ours) 76.81 73.58
Table 4
Main results on ConvFinQA test set (421 conversations) and TAT-QA test set (1,668 questions).
MethodConvFinQA TAT-QA
Exe Acc (%) Turn Acc (%) Exe Acc (%) F1 (%)
Zero-shot LLM†51.78 55.43 54.18 57.32
FinMA 57.92 61.37 57.41 60.73
Naive RAG†56.14 59.82 58.93 62.14
Advanced RAG†60.34 63.87 62.67 65.89
ReAct†62.47 65.93 64.35 67.48
Self-RAG†64.83 68.14 66.12 69.23
CRAG†66.72 70.28 67.58 70.74
IterRAG†69.14 72.67 69.34 72.51
FinAgent-RAG (Ours) 78.46 81.93 74.96 78.13
Table 5
Cross-backbone evaluation on FinQA test set.Δ= improve-
ment over Zero-shot with the same backbone.
LLM Backbone Zero-shot FinAgentΔ
GPT-4o 58.42 79.13 +20.71
DeepSeek-V3 53.27 76.81 +23.54
Qwen-2.5-72B 50.83 73.47 +22.64
Llama-3.1-70B 47.56 70.24 +22.68
Theagenticframeworkconsistentlyimprovesoverzero-
shot baselines across all backbones, withΔranging from
+20.71% (GPT-4o) to +23.54% (DeepSeek-V3). The re-
markablyconsistentimprovementmagnitude(20–24%)across
diverse backbones suggests that the gains stem from the
framework’s architectural design rather than from idiosyn-
cratic backbone capabilities. Notably, even with the open-
source Llama-3.1-70B, FinAgent-RAG achieves 70.24%,
surpassingthezero-shotperformanceofalltestedbackbones
including GPT-4o (58.42%). This cross-backbone robust-
ness is particularly important for financial institutions that
may prefer open-source models for data privacy reasons.Table 6
Ablation study on FinQA test set. Components are removed
one at a time from the full system.
Variant Exe Acc Prog AccΔ
Full FinAgent-RAG76.81 73.58—
w/o PoT (CoT only) 73.14 69.87−3.67
w/o Fin. Retriever 73.82 70.51−2.99
w/o Verification 75.18 71.94−1.63
w/ Router (adaptive) 75.47 72.19−1.34
w/o Decomposition 75.73 72.46−1.08
5.2. Ablation Study
Table 6 presents the ablation study on FinQA, isolating
the contribution of each domain-specific component.
Allcomponentscontributetothefinalperformance,with
thethreenewdomain-specificmodulesprovidingthelargest
individualgains.RemovingPoTreasoningcausesthelargest
singledrop(−3.67%),confirmingthatexecutablecodegen-
eration is more reliable than LLM mental arithmetic for
financialcomputations.RemovingtheContrastiveFinancial
Retriever results in a−2.99%drop, validating the impor-
tance of hard negative mining for distinguishing semanti-
callysimilarfinancialpassages.Self-verificationcontributes
Y. Shu, Y. Liu and Z. Xie:Preprint submitted to ElsevierPage 11 of 19

FinAgent-RAG for Financial Document QA
Table 7
Design space study: Retriever×Reasoning mode on FinQA
(Exe Acc %). Best result inbold.
Direct CoT PoT Adaptive
BM25 54.83 61.47 67.92 69.38
Dense (generic) 57.26 64.53 70.14 71.87
Hybrid 59.41 66.82 72.36 73.94
Dense (financial) 61.18 68.74 74.5176.81
−1.63%,andquerydecompositioncontributes−1.08%.Im-
portantly, the iterative loop components (verification + de-
composition) provide a qualitatively different type of con-
tribution than the reasoning and retrieval modules: they
enableprogressive evidence accumulationacross multiple
retrievalpasses,acapabilitythatisimpossibleinanysingle-
pass architecture regardless of retriever or reasoner quality.
The iteration depth analysis (Section 5.3) further shows
that moving from𝐾=1to𝐾=3improves accuracy by 5.34
percentage points (71.47%→76.81%), confirming that the
loop’s orchestration role is essential for achieving the full
system’sperformance.TheAdaptiveStrategyRoutervariant
trades1.34%accuracy for a41.3%reduction in API calls
(from 5.83 to 3.42 per question), representing a favorable
cost–accuracy trade-off for production deployment.
The sum of individual component drops (∑|Δ|=
10.71%) closely matches the total gap between the full
system and CRAG (76.81−66.21 = 10.60%), indicating
that each module addresses a largely independent failure
mode. This near-additive decomposition is astrengthof the
modular design: the iterative loop orchestrates components
that contribute complementary, non-redundant gains rather
than competing for the same error types.
5.3. Effect of Iteration Depth
Figure5showstheeffectofthemaximumiterationcount
𝐾on FinQA execution accuracy.
Performanceimprovessubstantiallyfrom𝐾=1(71.47%)
to𝐾=3(76.81%),withdiminishingreturnsbeyond𝐾=3
(𝐾=4: 76.93%,𝐾=5: 76.78%). The marginal gain drops
from +3.87% (𝐾= 1→2) to +1.47% (𝐾= 2→3) to
+0.12% (𝐾= 3→4), while average API calls increase
linearly (2.0, 3.8, 5.8, 7.6, 9.4). Setting𝐾=3achieves the
optimaltrade-offbetweenaccuracyandcomputationalcost.
5.4. Design Space Study
To understand how retriever and reasoning components
interact, we conduct a systematic4×4design space study
on the FinQA test set. Table 7 presents execution accuracy
for all 16 combinations.
Three key findings emerge (RQ3). First, both the re-
triever and reasoning dimensions contribute substantial
gains: upgrading from BM25 to financial dense retrieval
improves accuracy by 6–7% across all reasoning modes,
whileupgradingfromdirecttoadaptivereasoningimproves
by 14–16% across all retrievers. Second, the gains are
largely additive: the best single-axis configuration (BM25Table 8
Error distribution before and after FinAgent-RAG upgrades on
FinQA.“Base” =Zero-shot+NaiveRAGbaseline(536errors).
“Full” = FinAgent-RAG (266 errors). “Red.” = reduction in
absolute error count.
Error Type Base Full Red.
𝑛%𝑛% (%)
Arithmetic 208 38.8 25 9.4−88.0
Data extraction 120 22.4 79 29.6−34.2
Formula/logic 85 15.8 51 19.3−40.0
Multi-table 76 14.2 77 28.9−1.3
Terminology 47 8.8 34 12.8−27.7
Total536 100.0 266 100.0−50.4
+ Adaptive: 69.38%) is substantially outperformed by the
joint optimum (Financial + Adaptive: 76.81%), confirming
thatdomain-specificcomponentsonbothaxesprovidecom-
plementarybenefits.Third,thePoTcolumnconsistentlyout-
performs CoT by 5–7%, validating the value of executable
codegenerationforfinancialnumericalreasoningregardless
of the retriever choice. This design space map provides
practitioners with actionable guidance: organizations with
limited resources can achieve 69.38% with a simple BM25
retrieverpairedwithadaptivereasoning,whilethoseseeking
maximum accuracy should invest in the full financial-
adapted configuration.
5.5. Error Analysis
To understand how FinAgent-RAG’s domain-specific
components reshape the error landscape, we categorize all
errors from the FinQA test set into five types and compare
the base system (Zero-shot + Naive RAG) with the full
FinAgent-RAG system. Table 8 presents the results.
The most striking finding is that PoT reasoning elim-
inates 88.0% of arithmetic errors (from 208 to 25), con-
firming that executable code generation is far more reliable
than LLM mental computation for financial calculations.
This single improvement accounts for 67.8% of the total
errorreduction.TheContrastiveFinancialRetrieverreduces
data extraction errors by 34.2% and terminology errors by
27.7%, while multi-table errors remain nearly unchanged in
absolute count (76→77)—these represent the “hard core”
ofresidualerrorswheretherequiredevidencespansmultiple
documents or tables.
Importantly,theerrordistributionshiftsfromarithmetic-
dominated (38.8%→9.4%) to retrieval-dominated (data
extraction + multi-table = 58.5% of residual errors). This
shift has clear implications for future work: the next major
improvement will likely come from cross-table linking and
multi-document retrieval rather than further reasoning en-
hancements.Figure6visualizesthebefore/aftercomparison.
5.6. Computational Cost Analysis
Table 9 compares the computational cost of different
methodsintermsofaverageAPIcalls,latency,per-question
cost, and accuracy.
Y. Shu, Y. Liu and Z. Xie:Preprint submitted to ElsevierPage 12 of 19

FinAgent-RAG for Financial Document QA
Table 9
Computational cost comparison on FinQA. “Ours+Router” =
FinAgent-RAG with Adaptive Strategy Router enabled.
Method API Lat.(s) Cost($) Acc(%)
Zero-shot 1.0 2.3 0.003 53.27
Naive RAG 1.0 3.6 0.005 57.84
Adv. RAG 2.0 5.9 0.008 61.52
Self-RAG 3.2 8.4 0.012 64.75
CRAG 3.0 8.9 0.013 66.21
IterRAG 4.5 12.1 0.018 67.83
Ours (Full)5.8 15.4 0.02476.81
Ours+Router3.4 9.1 0.014 75.47
The full FinAgent-RAG system requires 5.8 API calls
per question, resulting in a per-question cost of $0.024—
approximately 8×the zero-shot cost but with a 23.54% ac-
curacyimprovement.TheAdaptiveStrategyRouterreduces
this to 3.4 API calls ($0.014) while maintaining 75.47% ac-
curacy,achievingafavorablecost–accuracyParetopoint.To
contextualize: processing a full 10-K filing (approximately
50 typical questions) costs $0.70–$1.20 with the router-
enabled variant, compared to an estimated 2–4 hours of
analysttime.Figure7visualizestheaccuracy–costtrade-off.
5.7. Case Studies
Toprovidequalitativeinsight,wepresentthreerepresen-
tative cases from the FinQA test set.
Case 1: Iterative Retrieval Refinement.Question:
“What was the percentage change in the provision for
income taxes from 2018 to 2019?”
Iteration1:TheQueryDecomposergeneratestwosub-
questions: (1) “What was the provision for income taxes in
2018?” and (2) “What was the provision for income taxes
in 2019?” The Adaptive Retriever successfully retrieves the
2019value($142M)fromtheincomestatementbutretrieves
a passage about “income tax expense” for 2018 ($128M)
fromthenotessection,whichreferstoadifferentaccounting
treatment.TheCoTReasonercomputes(142−128)∕128=
10.94%. The Self-Verifier flags across-evidence inconsis-
tency: the two values come from different contexts (income
statement vs. notes to financial statements).
Iteration2:TheQueryRefinergeneratesamorespecific
sub-question: “What was the provision for income taxes in
2018as reported in the consolidated income statement?”
The retriever now surfaces the correct value ($135M) from
the same income statement table. The reasoner computes
(142−135)∕135 = 5.19%, and the verifier confirms con-
sistency. The answer matches the gold label.
Thiscasedemonstrateshowtheself-verificationmecha-
nismcatchessubtleretrievalerrors—semanticallyrelevant
but contextually incorrect passages — that are invisible to
single-pass systems.
Case 2: PoT for Multi-Step Computation.Question:
“What was the compound annual growth rate of operating
expenses from 2017 to 2019?”
The decomposer generates: (1) retrieve 2017 operating
expenses,(2)retrieve2019operatingexpenses,(3)computeTable 10
Per-question-type accuracy (%) on FinQA. The weighted
average of per-type accuracies matches the overall accuracy
reported in Table 3.
Type % CRAG OursΔ
Ratio/Percentage 18.3 78.57 84.29 +5.72
Single-step arith. 31.2 74.63 82.54 +7.91
Multi-step arith. 28.7 56.32 72.47+16.15
Comparison 12.4 60.56 69.59 +9.03
Aggregation 9.4 51.86 65.96+14.10
CAGR.Bothvaluesareretrievedsuccessfully($2,847Mand
$3,214M). The PoT module generates:
v_begin, v_end, n = 2847, 3214, 2
cagr = (v_end / v_begin) ** (1/n) - 1
result = round(cagr * 100, 2) # 6.24%
Theexecutedcodereturns6.24%,matchingthegoldan-
swer. Under CoT reasoning, the LLM incorrectly computes
(3214−2847)∕2847∕2 = 6.45%, confusing CAGR with
simple average annual growth—a frequent arithmetic error
pattern that PoT eliminates through exact code execution.
Case 3: Residual Failure (Multi-Table Reasoning).
Question: “What percentage of total segment revenue was
contributed by the Asia-Pacific segment in 2019?”
ThesystemcorrectlyretrievesAsia-Pacificrevenue($1,283M)
fromthesegmentbreakdowntablebutfailstolocatethetotal
consolidated revenue figure, which appears in a separate
summary table on a different page. The agent retrieves a
passagestating“totalsegmentrevenue”of$4,892Mfroman
adjacent paragraph, but this figure excludes inter-segment
eliminations; the correct consolidated total is $4,731M.
The verifier does not flag this discrepancy because both
valuesareplausible.Thisfailureexemplifiesthemulti-table
reasoning errors (28.9% of residual errors) that motivate
future work on cross-table knowledge graph integration.
5.8. Per-Question-Type Analysis
To understand where FinAgent-RAG provides the most
benefit, we categorize FinQA test questions by reasoning
type and report per-category accuracy compared to CRAG
(Table 10).
The improvements are most pronounced for multi-step
arithmetic(+16.15%)andaggregationquestions(+14.10%)—
precisely the categories where PoT reasoning and itera-
tive evidence accumulation provide the greatest advantage.
These two categories require chained numerical compu-
tations over multiple retrieved values, which are highly
susceptible to LLM arithmetic errors under CoT reasoning
but are handled reliably by executable code generation.
Ratio/percentage questions show a moderate improvement
(+5.72%), as the computation is simpler (single division)
and the primary bottleneck is retrieval accuracy rather than
reasoning.
Thisper-typeanalysisprovidesimportantpracticalguid-
ance: the investment in PoT reasoning yields disproportion-
ate returns for organizations whose financial QA workload
Y. Shu, Y. Liu and Z. Xie:Preprint submitted to ElsevierPage 13 of 19

FinAgent-RAG for Financial Document QA
Table 11
Retrieval quality (Recall@𝑘) on FinQA by iteration.
Method R@5 R@10 R@20
Naive RAG 62.3 74.8 83.5
Advanced RAG 68.1 79.4 87.2
Ours (Iter 1) 69.7 80.6 88.1
Ours (Iter 2) 78.4 87.3 92.6
Ours (Iter 3) 82.3 90.1 94.3
Table 12
Verification behavior analysis on FinQA test set. “Accept” =
verifier accepts the answer; “Reject” = verifier triggers re-
retrieval.
Iteration Accept (%) Reject (%) Acc. of Accepted
1 72.4 27.6 81.3
2 19.4 8.2 74.8
3 5.8 2.4 68.3
Overall 97.6 2.4 78.9
isdominatedbycomplexmulti-stepcomputations.Figure8
visualizes the comparison.
5.9. Sensitivity Analysis
We analyze the sensitivity of FinAgent-RAG to three
key hyperparameters — confidence threshold𝜃, number of
retrieved passages (top-𝑘), and chunk size — on the FinQA
validation set. Full results are provided in Appendix C.
The confidence threshold𝜃= 0.8achieves the best
validation accuracy, with lower thresholds accepting an-
swers too readily and higher thresholds (≥0.9) triggering
excessive iterations without accuracy gains. Performance
peaks at top-𝑘= 5and chunk size 512 tokens. Figure 9
visualizesthedual-axistrade-offbetweenaccuracyandAPI
cost across𝜃values.
5.10. Retrieval Quality Analysis
To isolate the contribution of retrieval from reasoning,
we analyze retrieval quality using Recall@𝑘— the fraction
of gold evidence passages retrieved in the top-𝑘results.
Table 11 reveals that the iterative refinement loop sub-
stantially improves retrieval quality: Recall@5 increases
from 69.7% in the first iteration to 82.3% after three itera-
tions, a 12.6 percentage point improvement. This confirms
that the query refinement mechanism effectively addresses
retrieval gaps identified by the self-verifier. Notably, the
cumulative retrieval quality of our three iterations (R@5 =
82.3%) surpasses even the R@20 of single-pass methods,
demonstrating that targeted iterative retrieval is more effec-
tive than simply retrieving more passages in a single pass.
Figure 10 provides a grouped comparison.
5.11. Verification Behavior Analysis
To understand the self-verification mechanism in detail,
weanalyzetheverificationoutcomesacrossiterationsonthe
FinQA test set. Table 12 reports the fraction of questions
reaching each verification decision at each iteration.Table 13
Ablation of hard negative types in the Contrastive Financial
Retriever on FinQA.
Variant R@5 R@10 MRR
Full (4 types)82.34 91.67 0.784
w/o Temporal 79.18 89.43 0.752
w/o Metric-swap 77.56 88.21 0.732
w/o Granularity 80.14 90.02 0.761
w/o Entity-swap 80.87 90.38 0.770
No hard negatives 72.63 84.51 0.699
Severalobservationsemerge.First,themajorityofques-
tions (72.4%) are resolved in a single iteration, indicating
thattheself-verifiercorrectlyidentifieshigh-qualityanswers
without unnecessary re-retrieval. This is desirable from a
computationalcostperspective.Second,answersacceptedin
lateriterationshaveprogressivelyloweraccuracy(81.3%→
74.8%→68.3%),reflectingthefactthatlateriterationshan-
dle inherently harder questions. Third, the verifier achieves
an effective precision of 78.9% — that is, when it accepts
an answer, the answer is correct 78.9% of the time. The
rejection decisions are also meaningful: among questions
rejected after iteration 1, 38.7% had their answer corrected
after re-retrieval, confirming that the verifier successfully
identifies salvageable errors.
We also examine the types of verification failures. Ev-
idence sufficiency failures account for 52% of rejections,
numerical consistency failures for 31%, and cross-evidence
contradictions for 17%. This distribution informs our query
refinement strategy: the refiner is designed to prioritize re-
trievingadditionalevidence(addressingsufficiencyfailures)
overalternativeevidence(addressingcontradictionfailures).
VerifierPrecisionandRecall.Tofullycharacterizethe
self-verifier’s reliability, we compute its precision (fraction
of REJECT decisions where the answer was indeed incor-
rect) and recall (fraction of incorrect answers that were cor-
rectlyrejected).OntheFinQAtestset:theverifierachievesa
rejectionprecisionof76.2%(i.e.,76.2%ofrejectedanswers
were truly incorrect) and arejection recallof 58.4% (i.e.,
58.4% of all incorrect answers were caught by the verifier).
The false rejection rate of 23.8% leads to unnecessary re-
retrieval in some cases. However, this cost is modest: re-
retrievalofanalready-correctanswertypicallyreturnstothe
same answer in the next iteration (89.3% recovery rate), so
the net accuracy loss from false rejections is only 0.7%.
5.12. Contrastive Retriever Ablation
Tovalidatethecontributionofeachhardnegativetypein
theContrastiveFinancialRetriever,weconductacomponent
ablation study (Table 13).
Allfourhardnegativetypescontributetoretrievalqual-
ity, with metric-swap negatives (e.g., “revenue” vs. “cost
of revenue”) providing the largest individual contribution
(−4.78%R@5 when removed). Temporal negatives rank
Y. Shu, Y. Liu and Z. Xie:Preprint submitted to ElsevierPage 14 of 19

FinAgent-RAG for Financial Document QA
Table 14
Statistical significance analysis on FinQA. CI = 95% bootstrap
confidence interval. All comparisons significant at𝑝<0.001.
Comparison Baseline CI Ours CI𝜒2
vs. IterRAG [66.14, 69.49] [75.23, 78.36] 47.83
vs. CRAG [64.53, 67.86] [75.23, 78.36] 52.17
vs. Self-RAG [63.07, 66.41] [75.23, 78.36] 61.34
vs. Adv. RAG [59.84, 63.17] [75.23, 78.36] 78.92
second (−3.16%), consistent with the observation that con-
fusing fiscal periods is a prevalent error in financial docu-
ment retrieval. The full contrastive training improves R@5
by 9.71 percentage points over the baseline without hard
negatives,confirmingthevalueofdomain-specificnegative
mining for financial retrieval.
5.13. Statistical Significance
To verify that FinAgent-RAG’s improvements are sta-
tistically significant, we perform bootstrap resampling with
10,000iterationsontheFinQAtestset.Table14reports95%
confidence intervals and McNemar’s test results.
The95%confidenceintervalforFinAgent-RAG[75.23%,78.36%]
doesnotoverlapwithanybaseline’sconfidenceinterval,and
allMcNemar’s𝜒2testsyield𝑝<0.001,confirmingthatthe
improvements are not due to random variation.
6. Discussion
Wediscussthebroaderimplicationsofourfindings,con-
nectresultstotheresearchquestions,andidentifydirections
for future work.
6.1. Addressing the Research Questions
RQ1:EffectivenessofAgenticRAGforFinancialQA.
Our main results (Tables 3–4) provide a clear affirmative
answer:FinAgent-RAGachieves76.81%onFinQA,78.46%
onConvFinQA,and74.96%onTAT-QA,outperformingthe
strongestbaseline(IterRAG)by5.62–9.32percentagepoints
across three benchmarks. The per-type analysis (Table 10)
further shows that the gains are concentrated on multi-
steparithmetic(+16.15%)andaggregation(+14.10%)tasks,
precisely where single-pass approaches are most limited.
The cross-backbone evaluation (Table 5) confirms these
gains generalize across four diverse LLM backbones with
remarkably consistent improvement magnitude (20–24%).
RQ2: Component Contributions.The ablation study
(Table 6) reveals that all five components contribute mean-
ingfully, with PoT reasoning (−3.67%) and the Contrastive
Financial Retriever (−2.99%) being the most critical indi-
vidual modules. The error analysis (Table 8) provides a
mechanisticexplanation:PoTeliminates88.0%ofarithmetic
errors, while the financial retriever reduces data extraction
errors by 34.2%. The near-additive nature of component
contributions (∑|Δ|= 10.71%vs. total gap of 10.60%)
indicates that each module addresses a largely indepen-
dent failure mode, validating the modular design. Notably,the domain-specific components (PoT reasoning and Con-
trastive Financial Retriever) account for the majority of the
improvement,whiletheiterativeagenticloopprovidescom-
plementarygains(−1.63%forSelf-Verificationand−1.08%
forQueryDecomposition)byenablingprogressiveevidence
refinement. This suggests that the primary contribution of
FinAgent-RAGliesinthesystematicintegrationanddomain
adaptation of these components, rather than the agentic
architecture alone.
RQ3: Design Space Interactions.The4 × 4design
space study (Table 7) reveals that retriever and reasoning
improvements contribute complementary gains: upgrading
along either axis improves accuracy by 6–16%, while the
joint optimum exceeds any single-axis maximum by 7+
percentage points. This finding has practical significance:
it suggests that organizations should invest in both domain-
specific retrieval and reasoning rather than focusing exclu-
sively on one dimension.
RQ4: Computational Trade-offs and Deployment.
The cost analysis on FinQA (Table 9) shows that the Adap-
tive Strategy Router reduces API costs by 41.3% with only
1.34% accuracy trade-off. We note that the router is trained
and evaluated exclusively on FinQA; its generalization to
ConvFinQA and TAT-QA has not been directly validated.
Theconsistentcross-benchmarkimprovementsofFinAgent-
RAG’s full pipeline (Table 4) suggest that the underlying
question-complexity distribution is similar across bench-
marks, but the router’s cost-accuracy trade-off may differ
on benchmarks with different complexity profiles. Dataset-
specific router calibration or a universal router trained on
pooled data are promising directions for future work. The
verification behavior analysis (Table 12) reveals that 72.4%
of questions are resolved in a single iteration, meaning the
computational overhead is concentrated on the genuinely
difficult questions. These findings enable a tiered deploy-
ment architecture (Section 6.3) that balances accuracy and
cost based on organizational requirements.
6.2. Why Iterative Refinement Works
The success of FinAgent-RAG can be attributed to two
complementary mechanisms. First, theerror detectionca-
pability of the self-verifier converts implicit reasoning fail-
ures into explicit signals that guide re-retrieval. Our case
study demonstrates how cross-evidence validation catches
subtle retrieval errors (e.g., retrieving semantically similar
but contextually incorrect values). Second, theprogressive
evidenceaccumulationacrossiterationsenablestheagentto
gathermulti-tableevidencethatisfundamentallyimpossible
to capture in a single retrieval pass. The retrieval quality
analysis(Table11)quantifiesthiseffect:iterativerefinement
improves Recall@5 by 12.6 percentage points over the first
iteration alone.
Critically, the loop’s aggregate contribution (−2.71%
fromverificationanddecompositioninablation)understates
its impact on the hardest questions. The per-type analy-
sis (Table 10) shows that multi-step arithmetic and aggre-
gation questions—which most frequently require multiple
Y. Shu, Y. Liu and Z. Xie:Preprint submitted to ElsevierPage 15 of 19

FinAgent-RAG for Financial Document QA
iterations—exhibit the largest absolute improvements over
baselines (+16.15% and +14.10%), precisely because these
question types benefit most from iterative evidence gather-
ingacrosstablesandsections.Theretrievalqualityanalysis
(Table 11) provides direct evidence: Recall@5 improves
from 69.7% to 82.3% across three iterations, demonstrating
thattheloopenablestheretrievertosurfaceevidencethatis
fundamentally inaccessible in a single pass.
Thesefindingsalignwiththecognitivescienceperspec-
tive on human expert reasoning (Sumers, Yao, Narasimhan
and Griffiths, 2024): financial analysts do not answer com-
plex questions in a single pass but rather iteratively gather
evidence, verify intermediate conclusions, and refine their
analysis until they reach sufficient confidence. FinAgent-
RAG operationalizes this iterative expert reasoning process
within an LLM-based agent framework.
6.3. Deployment Guidelines for Financial
Institutions
TheFinAgent-RAGframeworkhasseveralpracticalap-
plications in the financial industry, and our experimental
results enable concrete deployment recommendations:
Automated Financial Analysis.Investment analysts
spend a significant portion of their time extracting and
computing metrics from corporate filings. FinAgent-RAG
can automate routine QA tasks—extracting year-over-year
revenue growth, computing financial ratios, or identifying
trendsacrossreportingperiods—freeinganalyststofocuson
higher-level interpretation. At $0.014 per question with the
Router-enabled variant, processing a full 10-K filing (∼50
questions)costsunder$1,comparedto2–4hoursofanalyst
time.
RegulatoryComplianceMonitoring.Theself-verification
mechanismisparticularlywellsuitedforregulatorycompli-
ance,wheredetectinginconsistenciesbetweenstatedresults
and underlying financial data is critical. The cross-evidence
validation check can automatically flag discrepancies war-
ranting manual review.
TieredDeploymentArchitecture.Basedonourexper-
imental findings, we recommend three deployment tiers for
financial institutions:
•Tier 1 (Cost-sensitive): BM25 + Adaptive reason-
ing (69.38% accuracy, $0.008/question). Suitable for
high-volume screening tasks where moderate accu-
racy is acceptable.
•Tier 2 (Balanced): Full FinAgent-RAG with Router
(75.47% accuracy, $0.014/question). Recommended
for most production workloads, offering 41.3% cost
reduction with minimal accuracy trade-off.
•Tier 3 (Accuracy-critical): Full FinAgent-RAG with-
out Router (76.81% accuracy, $0.024/question). Re-
servedforhigh-stakesapplicationssuchasregulatory
filings or audit support.Organizationswithdataprivacyrequirementscanuseopen-
source backbones (e.g., Llama-3.1-70B) with a modest ac-
curacy reduction (70.24% vs. 76.81%), as demonstrated in
our cross-backbone evaluation.
6.4. Comparison with Concurrent Approaches
During the preparation of this manuscript, several con-
current works have explored related directions. TradingA-
gents(Xiaoetal.,2024)appliedmulti-agentsystemstotrad-
ingdecisionsbutfocusesonmarketanalysisratherthandoc-
ument QA. FinMem (Yu et al., 2024) introduced memory-
augmented agents for financial decision-making but does
not address the retrieval challenges specific to structured
financial documents.
FinAgent-RAG is distinguished from these approaches
by its tight integration of four domain-specific compo-
nents within an iterative agentic loop. While concurrent
approaches typically address one aspect of the financial
AI pipeline (e.g., reasoningorretrievalorverification),
FinAgent-RAG provides an end-to-end solution that jointly
optimizes all components through the iterative feedback
loop. The ablation study (Table 6) confirms that this inte-
grated design outperforms any single-component approach.
6.5. Limitations and Future Directions
Weidentifyseverallimitationsandpromisingdirections
for future work.
LLMBackboneDependency.Whilethecross-backbone
evaluation (Table 5) demonstrates consistent improvements
across four LLMs including open-source Llama-3.1-70B,
absolute performance varies (70.24–79.13%). Evaluation
with smaller and domain-specific financial LLMs would
further characterize generalizability.
Computational Overhead.The full system requires
5.8×more API calls than zero-shot. The Adaptive Strategy
Router mitigates this (reducing to 3.4×), but further opti-
mizationthroughcachingandparallelsub-questionprocess-
ing could reduce latency for time-sensitive applications.
Router Generalization.The Adaptive Strategy Router
is trained and evaluated solely on the FinQA validation set.
Although the overall FinAgent-RAG pipeline generalizes
well across ConvFinQA and TAT-QA, the router’s rout-
ing accuracy and cost-accuracy trade-off have not been di-
rectly measured on these benchmarks. Deploying the router
on datasets with substantially different question-complexity
distributions may require re-calibration or retraining.
Benchmark Scope.Despite evaluating on three bench-
marks, all are derived from English-language US corporate
filings under US GAAP. Generalization to other languages,
regulatory frameworks (IFRS), document formats (XBRL),
or real-time scenarios remains to be validated.
Multi-TableandCross-DocumentReasoning.Ourer-
roranalysisidentifiesmulti-tablereasoningasthedominant
residual error category (28.9% of remaining errors, Case 3
in Section 5). The current iterative loop can accumulate
evidence from multiple tables across iterations, but it lacks
explicit mechanisms for cross-table linking—for example,
Y. Shu, Y. Liu and Z. Xie:Preprint submitted to ElsevierPage 16 of 19

FinAgent-RAG for Financial Document QA
recognizing that “total segment revenue” in a breakdown
tableshouldreconcilewith“consolidatedrevenue”inasum-
mary table after inter-segment eliminations. Three concrete
mitigation strategies merit investigation: (1) constructing
table-level metadata graphsthat encode hierarchical rela-
tionships (segment⊂consolidated, quarterly⊂annual) to
guide cross-table retrieval; (2) adding anumerical recon-
ciliation checkto the self-verifier that flags when retrieved
values from different tables are inconsistent with expected
accountingidentities;and(3)incorporatingstructuredfinan-
cial knowledge graphs(Pan, Luo, Wang, Chen, Wang and
Wu, 2024) to provide explicit cross-table relationships as
additional retrieval signals.
Other Future Directions.Extending the framework to
multi-documentcomparativeanalysisacrosscompanyport-
folios and evaluating on multilingual financial QA bench-
marks (IFRS-based filings, non-English corpora) are also
promising avenues.
7. Conclusion
We presented FinAgent-RAG, an agentic RAG frame-
workforfinancialdocumentQAthatintegratesthreedomain-
specificinnovations—aContrastiveFinancialRetrieverwith
hard negative mining, a Program-of-Thought reasoning
module with sandboxed code execution, and an Adaptive
StrategyRouterforcost-efficientdeployment—withinitera-
tive retrieval-reasoning loops with self-verification.
Extensive experiments on three benchmarks (FinQA,
ConvFinQA, TAT-QA) demonstrate that FinAgent-RAG
achieves 76.81%, 78.46%, and 74.96% execution accuracy
respectively, outperforming eight baselines including the
strongest agentic baseline (IterRAG) by 5.62–9.32 percent-
agepoints.Theper-question-typeanalysisrevealsthatgains
are concentrated on multi-step arithmetic (+16.15%) and
aggregation (+14.10%) questions, and the error analysis
shows that PoT reasoning eliminates 88.0% of arithmetic
errors—the dominant failure mode in financial numerical
QA.
Our systematic design space study across retriever and
reasoningdimensionsprovidesempiricalguidanceforprac-
titioners, while the Adaptive Strategy Router enables a
41.3% reduction in API costs on FinQA with minimal
accuracy trade-off, addressing a key deployment concern
for financial institutions. The cross-backbone evaluation
confirms that FinAgent-RAG’s gains generalize across four
LLMs with consistent improvement magnitude (20–24%),
includingopen-sourcemodelssuitableforprivacy-sensitive
financial environments.
FinAgent-RAGdemonstratesthatthesystematicintegra-
tion of domain-specific retrieval, executable reasoning, and
adaptive resource allocation—orchestrated through an iter-
ative agentic loop—yields substantial and consistent gains
forfinancialdocumentanalysis.Theframeworkprovidesac-
tionable deployment guidelines (three-tier architecture) for
production environments. The identification of multi-table
reasoning as the dominant residual error category (28.9%)pointstoknowledge-graph-enhancedcross-tableretrievalas
the most impactful direction for future work.
CRediT authorship contribution statement
Yang Shu:Conceptualization, Methodology, Software,
Validation, Writing – original draft, Writing – review &
editing.Yingmin Liu:Investigation, Writing – review &
editing.Zequn Xie:Supervision, Writing – review & edit-
ing.
Declaration of competing interest
The authors declare no known competing financial in-
terests or personal relationships that could have appeared to
influence the work reported in this paper.
Data availability
The FinQA dataset is available athttps://github.com/
czyssrs/FinQA.TheConvFinQAdatasetisavailableathttps:
//github.com/czyssrs/ConvFinQA. The TAT-QA dataset is
availableathttps://github.com/NExTplusplus/TAT-QA.Code,
experimental scripts, and the fine-tuned Contrastive Finan-
cial Retriever weights will be released upon acceptance at
https://github.com/cheer932041235/FinAgent-RAG.
Acknowledgements
The authors thank the anonymous reviewers for their
constructive comments.
A. Prompt Templates
We provide representative prompt templates used by
FinAgent-RAG’s reasoning modules. All prompts follow a
structured format with role assignment, evidence context,
and output specification. Figure 11 presents the three core
prompt architectures used by the framework.
Each prompt template consists of four structured sec-
tions: (1) aSystem Rolethat establishes the agent’s do-
main expertise, (2)EvidenceandQuestionplaceholders
for dynamic content injection, (3) a numberedInstructions
list tailored to the specific reasoning mode, and (4) a con-
strainedOutput Formatthat ensures machine-parseable
responses. The CoT prompt elicits step-by-step natural lan-
guage reasoning, while the PoT prompt requires executable
Python code with the final answer assigned to aresult
variable. The Self-Verification prompt implements a three-
criterion checklist (evidence sufficiency, numerical consis-
tency, cross-evidence validation) and outputs a binary AC-
CEPT/REJECTdecisionwithanexplanationthatdrivesthe
iterative refinement loop.
B. Document Preprocessing Details
This appendix provides the full three-stage preprocess-
ing pipeline summarized in Section 3.
Y. Shu, Y. Liu and Z. Xie:Preprint submitted to ElsevierPage 17 of 19

FinAgent-RAG for Financial Document QA
Table 15
Effect of confidence threshold𝜃on FinQA performance.
𝜃Exe Acc (%) Avg. Iters API Calls
0.5 74.63 1.21 3.9
0.6 75.47 1.35 4.4
0.7 76.38 1.52 5.1
0.8 77.16 1.68 5.8
0.9 76.85 2.14 7.3
1.0 76.21 2.71 9.2
Stage 1: Document Parsing and Segmentation.We
parse each financial document into a sequence of typed
segments:textparagraphs,tableblocks, andheaderele-
ments. Tables are detected using structural heuristics (row-
column alignment patterns) and parsed into a structured
representation preserving row/column headers, cell values,
and spanning cells.
Stage 2: Table Linearization.Each table is linearized
into a textual representation suitable for dense retrieval. We
adopt aheader-prepended row linearizationstrategy: for
each row𝑟in a table with column headers[ℎ1,ℎ2,…,ℎ𝑐],
we generate a passage of the form “ℎ1:𝑟1|ℎ2:𝑟2|…|
ℎ𝑐:𝑟𝑐”. This preserves the semantic association between
columnheadersandcellvalues,whichiscriticalforfinancial
reasoning where the same numerical value has different
meaningsdependingonitscolumncontext(e.g.,“Revenue”
in a “2019” column vs. a “2020” column).
Stage 3: Chunking and Indexing.Text paragraphs are
chunked at the paragraph level with a maximum chunk
size of 512 tokens and an overlap of 64 tokens. Linearized
table rows are treated as individual passages. All pas-
sages are encoded into dense vector representations using
bge-base-en-v1.5(Xiao et al., 2023) (dimension 768) and
indexed with FAISS (Johnson et al., 2019) for efficient
approximate nearest neighbor search. The final index for
a typical document corpus contains approximately50,000–
80,000passages.
C. Sensitivity Analysis Details
Thisappendixprovidesthecompletesensitivityanalysis
results summarized in Section 5.9.
Confidence Threshold𝜃.Table 15 shows the effect of
varying𝜃on accuracy, iteration count, and API calls on the
FinQA validation set.
Setting𝜃= 0.8achieves the best accuracy. Lower
thresholdsacceptanswerstooreadily,missingopportunities
for refinement. Higher thresholds (𝜃≥0.9) trigger exces-
sive iterations that do not improve accuracy, likely because
the additional retrieval introduces noise rather than useful
evidence.Figure9visualizesthedual-axistrade-offbetween
accuracy and API cost.
NumberofRetrievedPassages(top-𝑘).Table16shows
the effect of varying top-𝑘on accuracy and context length.
Performance peaks at𝑘= 5and declines slightly with
more passages, consistent with the “lost in the middle”
phenomenon (Liu et al., 2024): excessive context dilutesTable 16
Effect of top-𝑘retrieval on FinQA performance.
Top-𝑘Exe Acc (%) Avg. Context Tokens
3 74.89 1,843
5 77.16 2,917
7 76.78 4,102
10 76.24 5,736
the relevance of key evidence. The decline at𝑘= 10also
increases API cost due to longer prompts.
Chunk Size.We evaluate chunk sizes of 256, 512,
and 1024 tokens with proportional overlap (12.5%). The
512-tokenconfigurationachievesthebesttrade-off(77.16%
accuracy),while256tokensfragmentfinancialtablesacross
chunksand1024tokensreduceretrievalprecisionbymixing
relevant and irrelevant content.
References
Araci, D., 2019. FinBERT: Financial sentiment analysis with pre-trained
language models. arXiv preprint arXiv:1908.10063 .
Asai, A., Wu, Z., Wang, Y., Sil, A., Hajishirzi, H., 2024. Self-RAG:
Learning to retrieve, generate, and critique through self-reflection, in:
International Conference on Learning Representations (ICLR), pp. 1–
21.
Borgeaud, S., Mensch, A., Hoffmann, J., Cai, T., Rutherford, E., Millican,
K., van den Driessche, G., Lespiau, J.B., Damoc, B., Clark, A., et al.,
2022. Improvinglanguagemodelsbyretrievingfromtrillionsoftokens,
in: International Conference on Machine Learning (ICML), pp. 2206–
2240.
Chen, J., Zhou, P., Hua, Y., Xin, L., Chen, K., Li, Z., Zhu, B., Liang, J.,
2024. FinTextQA:Adatasetforlong-formfinancialquestionanswering,
in: Proceedings of the 62nd Annual Meeting of the Association for
Computational Linguistics (ACL), pp. 6025–6047.
Chen, W., Ma, X., Wang, X., Cohen, W.W., 2023. Program of thoughts
prompting: Disentangling computation from reasoning for numerical
reasoning tasks. Transactions on Machine Learning Research .
Chen, Z., Chen, W., Smiley, C., Shah, S., Borber, I., Mouber, W., Wang,
W.Y., 2021. FinQA: A dataset of numerical reasoning over financial
data, in: Proceedings of the 2021 Conference on Empirical Methods in
Natural Language Processing (EMNLP), pp. 7199–7210.
Chen, Z., Li, S., Smiley, C., Ma, Z., Shah, S., Wang, W.Y., 2022. Con-
vFinQA: Exploring the chain of numerical reasoning in conversational
finance question answering, in: Proceedings of the 2022 Conference
on Empirical Methods in Natural Language Processing (EMNLP), pp.
6279–6292.
Gao,L.,Madaan,A.,Zhou,S.,Alon,U.,Liu,P.,Yang,Y.,Callan,J.,Neu-
big, G., 2023. PAL: Program-aided language models, in: International
Conference on Machine Learning (ICML), pp. 10764–10799.
Gao, Y., Xiong, Y., Gao, X., Jia, K., Pan, J., Bi, Y., Dai, Y., Sun, J., Wang,
H.,2024. Retrieval-augmentedgenerationforlargelanguagemodels:A
survey. arXiv preprint arXiv:2312.10997 .
Glass,M.,Rossiello,G.,Chowdhury,M.F.M.,Naber,A.,Nishi,R.,Gliozzo,
A., 2022. Re2G: Retrieve, rerank, generate, in: Proceedings of the
2022 Conference of the North AmericanChapter of the Association for
Computational Linguistics (NAACL), pp. 2701–2715.
Guu, K., Lee, K., Tung, Z., Pasupat, P., Chang, M.W., 2020. REALM:
Retrieval-augmented language model pre-training, in: International
Conference on Machine Learning (ICML), pp. 3929–3938.
Herzig, J., Nowak, P.K., Müller, T., Piccinno, F., Eisenschlos, J.M., 2020.
TaPas:Weaklysupervisedtableparsingviapre-training,in:Proceedings
of the 58th Annual Meeting of the Association for Computational
Linguistics (ACL), pp. 4320–4333.
Y. Shu, Y. Liu and Z. Xie:Preprint submitted to ElsevierPage 18 of 19

FinAgent-RAG for Financial Document QA
Huang,L.,Yu,W.,Ma,W.,Zhong,W.,Feng,Z.,Wang,H.,Chen,Q.,Peng,
W., Feng, X., Qin, B., Liu, T., 2025. A survey on hallucination in large
languagemodels:Principles,taxonomy,challenges,andopenquestions.
ACM Computing Surveys 57, 1–52.
Imani,S.,Du,L.,Shrivastava,H.,2023. MathPrompter:Mathematicalrea-
soning using large language models. arXiv preprint arXiv:2303.05398
.
Izacard, G., Grave, E., 2021. Leveraging passage retrieval with generative
modelsforopendomainquestionanswering,in:Proceedingsofthe16th
Conference of the European Chapter of the Association for Computa-
tional Linguistics (EACL), pp. 874–880.
Jiang, Z., Xu, F.F., Gao, L., Sun, Z., Liu, Q., Dwivedi-Yu, J., Yang, Y.,
Callan,J.,Neubig,G.,2023. Activeretrievalaugmentedgeneration,in:
Proceedings of the 2023 Conference on Empirical Methods in Natural
Language Processing (EMNLP), pp. 7969–7992.
Johnson, J., Douze, M., Jégou, H., 2019. Billion-scale similarity search
with GPUs. IEEE Transactions on Big Data 7, 535–547.
Kadavath, S., Conerly, T., Askell, A., Henighan, T., Drain, D., Perez, E.,
Schiefer,N.,Hatfield-Dodds,Z.,DasSarma,N.,Tran-Johnson,E.,etal.,
2022. Languagemodels(mostly)knowwhattheyknow. arXivpreprint
arXiv:2207.05221 .
Karpukhin,V.,Oguz,B.,Min,S.,Lewis,P.,Wu,L.,Edunov,S.,Chen,D.,
Yih, W.t., 2020. Dense passage retrieval for open-domain question an-
swering,in:Proceedingsofthe2020ConferenceonEmpiricalMethods
in Natural Language Processing (EMNLP), pp. 6769–6781.
Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V., Goyal, N.,
Küttler, H., Lewis, M., Yih, W.t., Rocktäschel, T., Riedel, S., Kiela,
D., 2020. Retrieval-augmented generation for knowledge-intensive
NLP tasks, in: Advances in Neural Information Processing Systems
(NeurIPS), pp. 9459–9474.
Liu, N.F., Lin, K., Hewitt, J., Paranjape, A., Bevilacqua, M., Petroni, F.,
Liang, P., 2024. Lost in the middle: How language models use long
contexts. TransactionsoftheAssociationforComputationalLinguistics
12, 157–173.
Ma, X., Gong, Y., He, P., Zhao, H., Duan, N., 2023. Query rewriting
in retrieval-augmented large language models, in: Proceedings of the
2023ConferenceonEmpiricalMethodsinNaturalLanguageProcessing
(EMNLP), pp. 5303–5315.
Pan,S.,Luo,L.,Wang,Y.,Chen,C.,Wang,J.,Wu,X.,2024.Unifyinglarge
languagemodelsandknowledgegraphs:Aroadmap. IEEETransactions
on Knowledge and Data Engineering 36, 3580–3599.
Qin, Y., Liang, S., Ye, Y., Zhu, K., Yan, L., Lu, Y., Lin, Y., Cong, X.,
Tang, X., Qian, B., et al., 2024. ToolLLM: Facilitating large language
modelstomaster16000+real-worldAPIs,in:InternationalConference
on Learning Representations (ICLR), pp. 1–28.
Schick, T., Dwivedi-Yu, J., Dessì, R., Raileanu, R., Lomeli, M., Hambro,
E., Zettlemoyer, L., Cancedda, N., Scialom, T., 2023. Toolformer:
Language models can teach themselves to use tools, in: Advances in
Neural Information Processing Systems (NeurIPS), pp. 68539–68551.
Shah,R.S.,Kuber,K.,Lee,M.,Nishi,R.,Vig,J.,2022. WhenFLUEmeets
FLANG:Benchmarksandlargepre-trainedlanguagemodelforfinancial
domain, in: Proceedingsof the 2022 Conference onEmpirical Methods
in Natural Language Processing (EMNLP), pp. 2322–2335.
Shao,Z.,Gong,Y.,Shen,Y.,Huang,M.,Duan,N.,Chen,W.,2023.Enhanc-
ing retrieval-augmented large language models with iterative retrieval-
generation synergy, in: Findings of the Association for Computational
Linguistics: EMNLP 2023, pp. 9248–9274.
Shinn, N., Cassano, F., Gopinath, A., Narasimhan, K., Yao, S., 2023.
Reflexion: Language agents with verbal reinforcement learning, in:
Advances in Neural Information Processing Systems (NeurIPS), pp.
8634–8652.
Singh, A., Ehtesham, A., Kumar, S., Khoei, T.T., Vasilakos, A.V., 2025.
Agentic retrieval-augmented generation: A survey on agentic RAG.
arXiv preprint arXiv:2501.09136 .
Sumers, T.R., Yao, S., Narasimhan, K., Griffiths, T.L., 2024. Cognitive
architectures for language agents. Transactions on Machine Learning
Research .Wang,L.,Ma,C.,Feng,X.,Zhang,Z.,Yang,H.,Zhang,J.,Chen,Z.,Tang,
J.,Chen,X.,Lin,Y.,Zhao,W.X.,Wei,Z.,Wen,J.R.,2024. Asurveyon
largelanguagemodelbasedautonomousagents. FrontiersofComputer
Science 18, 186345.
Wei,J.,Wang,X.,Schuurmans,D.,Bosma,M.,Ichter,B.,Xia,F.,Chi,E.,
Le,Q.,Zhou,D.,2022. Chain-of-thoughtpromptingelicitsreasoningin
large language models, in: Advances in Neural Information Processing
Systems (NeurIPS), pp. 24824–24837.
Wu,S.,Irsoy,O.,Lu,S.,Daber,V.,Dredze,M.,Gehrmann,S.,Kambadur,
P., Rosenberg, D., Mann, G., 2023. BloombergGPT: A large language
model for finance. arXiv preprint arXiv:2303.17564 .
Xi, Z., Chen, W., Guo, X., He, W., Ding, Y., Hong, B., Zhang, M., Wang,
J.,Jin,S.,Zhou,E.,etal.,2023. Theriseandpotentialoflargelanguage
model based agents: A survey. arXiv preprint arXiv:2309.07864 .
Xiao, S., Liu, Z., Zhang, P., Muennighoff, N., 2023. C-Pack: Packaged
resources to advance general Chinese embedding. arXiv preprint
arXiv:2309.07597 .
Xiao, Y., Sun, E., Luo, D., Wang, W., 2024. TradingAgents: Multi-agents
LLM financial trading framework. arXiv preprint arXiv:2412.20138 .
Xie, Q., Han, W., Zhang, X., Lai, Y., Peng, M., Lopez-Lira, A., Huang, J.,
2023. PIXIU: A large language model, instruction data and evaluation
benchmark for finance, in: Advances in Neural Information Processing
Systems (NeurIPS): Datasets and Benchmarks Track.
Xie, Q., Han, W., Zhang, X., Lai, Y., Peng, M., Lopez-Lira, A., Huang, J.,
2024.FinBen:Aholisticfinancialbenchmarkforlargelanguagemodels.
arXiv preprint arXiv:2402.12659 .
Xie, Z., Sun, Z., 2026. Mitigating hallucination on hallucination in RAG
via ensemble voting. arXiv preprint arXiv:2603.27253 .
Xie, Z., Wang, C., Wang, Y., Cai, S., Wang, S., Jin, T., 2025. Chat-driven
textgenerationandinteractionforpersonretrieval,in:Proceedingsofthe
2025ConferenceonEmpiricalMethodsinNaturalLanguageProcessing
(EMNLP), pp. 5259–5270.
Yan, S.Q., Gu, J.C., Zhu, Y., Ling, Z.H., 2024. Corrective retrieval
augmented generation. arXiv preprint arXiv:2401.15884 .
Yang, Y., Uy, M.C.S., Huang, A., 2020. FinBERT: A pretrained language
model for financial communications. arXiv preprint arXiv:2006.08097
.
Yao, S., Zhao, J., Yu, D., Du, N., Shafran, I., Narasimhan, K., Cao, Y.,
2023. ReAct: Synergizing reasoning and acting in language models,
in: International Conference on Learning Representations (ICLR), pp.
1–25.
Yin, P., Neubig, G., Yih, W.t., Riedel, S., 2020. TaBERT: Pretraining for
joint understanding of textual and tabular data, in: Proceedings of the
58th Annual Meeting of the Association for Computational Linguistics
(ACL), pp. 8413–8426.
Yu, Y., Li, H., Chen, Z., Jiang, Y., Li, Y., Zhang, D., Liu, R., Suchow,
J.W., Khaldoun, K., 2024. FinMem: A performance-enhanced LLM
trading agent with layered memory and character design, in: AAAI
Spring Symposium Series (SSS).
Zhu, F., Lei, W., Huang, Y., Wang, C., Zhang, S., Lv, J., Feng, F., Chua,
T.S., 2021. TAT-QA: A question answering benchmark on a hybrid of
tabularandtextualcontentinfinance,in:Proceedingsofthe59thAnnual
Meeting of the Association for Computational Linguistics (ACL), pp.
3277–3287.
Y. Shu, Y. Liu and Z. Xie:Preprint submitted to ElsevierPage 19 of 19

FinAgent-RAG for Financial Document QA
Figure 5:Effect of maximum iteration depth𝐾on FinQA
execution accuracy.
Y. Shu, Y. Liu and Z. Xie:Preprint submitted to ElsevierPage 20 of 19

FinAgent-RAG for Financial Document QA
Figure 6:Error distribution before and after FinAgent-RAG on FinQA.
Figure 7:Accuracy vs. cost trade-off on FinQA.
Figure 8:Per-question-type accuracy comparison between
CRAG and FinAgent-RAG on FinQA.
Figure 9:Effect of confidence threshold𝜃on execution
accuracy and average API calls.
Figure 10:Retrieval quality (Recall@𝑘) comparison across
methods and iterations.
Y. Shu, Y. Liu and Z. Xie:Preprint submitted to ElsevierPage 21 of 19

FinAgent-RAG for Financial Document QA
Figure 11:Structured prompt templates for FinAgent-RAG’s three reasoning modules: CoT (left), PoT (center), and Self-
Verification (right).
Y. Shu, Y. Liu and Z. Xie:Preprint submitted to ElsevierPage 22 of 19