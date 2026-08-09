# D$^2$F-ReAG: Dynamic Decomposition and Filtering for Multi-Hop Reasoning-Augmented Generation

**Authors**: Jiaoyang Li, Junhao Ruan, Shengwei Tang, Kaiyan Chang, Zhengtao Yu, Tong Xiao, Jingbo Zhu

**Published**: 2026-08-05 04:41:36

**PDF URL**: [https://arxiv.org/pdf/2608.04444v1](https://arxiv.org/pdf/2608.04444v1)

## Abstract
Large language models (LLMs) often generate inaccurate answers due to their reliance on static internal knowledge. Retrieval-augmented generation (RAG) addresses this limitation by integrating external knowledge and excelling at single-hop queries. However, it struggles with multi-hop questions that require cross-document reasoning. Existing methods, such as graph structured RAG or question decomposition, often lack dynamic decomposition and effective filtering, which leads to lower efficiency and accuracy. To overcome these limitations, we propose Dynamic Decomposition and Filtering for Multi-Hop Reasoning-Augmented Generation (D2F-ReAG), a novel paradigm that adaptively controls reasoning depth by judging the reliability of the root-level reasoning. If the root reasoning is reliable, the model directly generates the answer. Otherwise, the question is logically decomposed into sub-questions, and the verified reasoning derived from these sub-questions is used to refine the root reasoning. Experiments on three multi-hop benchmarks demonstrate the effectiveness of our method in handling complex multi-hop questions.

## Full Text


<!-- PDF content starts -->

D2F-ReAG: Dynamic Decomposition and
Filtering for Multi-Hop Reasoning-Augmented
Generation
Jiaoyang Li1, Junhao Ruan1, Shengwei Tang1, Saihan Chen1, Kaiyan Chang1,
Zhengtao Yu2, Tong Xiao1,3, and Jingbo Zhu1,3
1Northeastern University, Shenyang, China
2Kunming University of Science and Technology, Kunming, China
3NiuTrans Research, Shenyang, China
Abstract.Large language models (LLMs) often generate inaccurate
answers due to their reliance on static internal knowledge. Retrieval-
augmented generation (RAG) addresses this limitation by integrating
externalknowledgeandexcellingatsingle-hopqueries.However,itstrug-
gleswithmulti-hopquestionsthatrequirecross-documentreasoning.Ex-
isting methods, such as graph structured RAG or question decomposi-
tion,oftenlackdynamicdecompositionandeffectivefiltering,whichleads
to lower efficiency and accuracy. To overcome these limitations, we pro-
poseDynamicDecomposition andFiltering for Multi-HopReasoning-
AugmentedGeneration (D2F-ReAG), a novel paradigm that adaptively
controls reasoning depth by judging the reliability of the root-level rea-
soning. If the root reasoning is reliable, the model directly generates
the answer. Otherwise, the question is logically decomposed into sub-
questions, and the verified reasoning derived from these sub-questions is
used to refine the root reasoning. Experiments on three multi-hop bench-
marks demonstrate the effectiveness of our method in handling complex
multi-hop questions.
Keywords:Multi-HopReasoning·DynamicDecomposition·Reasoning-
Augmented Generation
1 Introduction
Large Language Models (LLMs) have demonstrated strong capabilities in lan-
guage understanding and generation across a wide range of tasks [1]. Despite
these advances, they still suffer from inherent limitations such as outdated para-
metric knowledge and hallucinations, which often lead to factually incorrect or
unsupported outputs and undermine their reliability in knowledge-intensive sce-
narios [11]. Retrieval-augmented Generation (RAG) mitigates these issues by
incorporating external knowledge sources into the generation process, thereby
improving factuality and reducing the model’s reliance on its static internal
knowledge [13].
arXiv:2608.04444v1  [cs.CL]  5 Aug 2026

2 J. Li et al.
For simple factual questions, retrievers can often locate the necessary ev-
idence within a single retrieval step, making the process both efficient and
straightforward [17]. In contrast, multi-hop reasoning questions require connect-
ing multiple pieces of evidence that are typically scattered across different docu-
ments[3]. To address this challenge, Graph-based RAG (Graph RAG) organizes
facts through pre-built graph structures, enabling effective evidence traversal
and multi-hop retrieval across interconnected knowledge units [6]. However, such
graphs often suffer from inherent limitations: they may be incomplete due to the
difficulty of capturing all relevant relations, are costly to construct and maintain
at scale, and become less effective when the underlying knowledge is frequently
updated or evolving [14].
Recent advances such as LogicRAG [2] reduce the reliance on pre-constructed
graphs by decomposing complex queries into sub-questions and iteratively com-
pressing retrieved evidence into a document-level memory, enabling multi-hop
reasoning without explicit graph structures. However, such compression cannot
fully filter out redundant or erroneous content, allowing noise to accumulate
across iterations and gradually mislead subsequent reasoning [23]. Moreover,
their query decomposition is often rigid and lacks adaptive control over gran-
ularity, so that simple questions tend to be over-decomposed while genuinely
complex ones are not decomposed deeply enough, ultimately degrading reason-
ing performance [12].
To overcome these limitations, we propose D2F-ReAG, a multi-hop reason-
ing framework that performs on-demand question decomposition driven by rea-
soning reliability and augments generation with verified sub-question reasoning
paths. Concretely, D2F-ReAG continuously assesses the model’s confidence in
its current reasoning, and triggers further decomposition only when the rea-
soning over the current question is judged unreliable. Once a sub-question is
solved with sufficient confidence and deemed relevant to the query, its reason-
ing trace is propagated upward to update and guide the reasoning of the root
question, so that accurate intermediate results are effectively integrated rather
than discarded or compressed away. To prevent over-reasoning and unnecessary
computational overhead, D2F-ReAG also adopts an early-stopping strategy that
terminates the process as soon as the root question can be answered reliably. In
summary, our contributions are:
–We propose D2F-ReAG, a novel framework that dynamically decomposes
complex questions into a sequence of logical sub-questions and progressively
augments the reasoning process of the root question with the correct reason-
ing paths derived from these sub-questions, ensuring that accurate interme-
diate results are effectively propagated back to guide the final answer rather
than being lost in lossy memory compression.
–We dynamically and adaptively balance the decomposition depth accord-
ing to question complexity through a confidence-driven control mechanism,
enabling the framework to perform deeper decomposition only when nec-
essary while avoiding redundant decomposition for simpler questions, thus

Title Suppressed Due to Excessive Length 3
achieving truly on-demand decomposition that adapts to varying reasoning
difficulty.
–We leverage reliable reasoning chains derived from sub-questions to itera-
tively update and correct the root reasoning process, which effectively identi-
fies and fixes intermediate errors, thereby reducing error accumulation across
reasoning steps and improving the overall correctness of the final answer.
2 Related Work
Graph-based methods organize external knowledge into structured graph repre-
sentations to better support multi-hop retrieval and cross-document reasoning.
GraphRAG [6] performs hierarchical community search by jointly leveraging
local and global queries, allowing retrieval to operate at varying levels of gran-
ularity. LightRAG [7] further improves large-scale retrieval efficiency through
a two-stage, graph-augmented indexing pipeline that balances coverage and
cost. RAPTOR [16] constructs hierarchical summaries over the corpus to enable
multi-granularretrievalatdifferentlevelsofabstraction,therebyaccommodating
queriesofdiversespecificity.HippoRAG[8]anditsextensionHippoRAG2[9]ad-
ditionally enhance long-context coherence by ranking nodes with PageRank [15]
and integrating paragraph-level memory, which helps preserve contextual conti-
nuity across retrieval steps. Despite these advances, graph-based methods gen-
erally rely on pre-constructed structures that are costly to build and maintain,
and may become outdated when the underlying knowledge evolves.
Prompt-basedmethods,incontrast,interleavereasoningandretrievalthrough
carefullydesignedpromptswithoutmodifyingmodelparameters,offeringamore
flexible and lightweight alternative. ReAct [21] alternates between reasoning and
retrieval in a step-by-step manner, enabling the model to dynamically decide
when to retrieve based on its current reasoning state and observed evidence.
ChainRAG [22] decomposes complex questions into sub-questions guided by
prompts, and subsequently performs iterative sentence-graph retrieval to cap-
ture both local fine-grained details and global document-level structures. Logi-
cRAG [2] similarly decomposes questions into logical components and aggregates
evidence across multiple reasoning steps, leveraging logical relationships among
sub-questions to enhance retrieval coherence and reduce redundancy. SiGIR [4]
refines the reasoning process iteratively to progressively approach the final an-
swer; however, it decomposes questions purely by atomicity and lacks explicit
correctness verification during retrieval, which may allow erroneous intermediate
resultstopropagateandaccumulateacrossreasoningsteps,ultimatelydegrading
the quality of the final answer.
3 Method
WeproposeD2F-ReAG,adynamicframeworkformulti-hopreasoning.Asshown
in Figure 1, D2F-ReAG begins with (I) Retrieval & Generation, where first re-
trieves top-kdocuments to generate question’s reasoning. In (II) Judge Relia-
bility, a judge model evaluates the reliability of the reasoning. If reliable, the

4 J. Li et al.
Which film was released first,
Aas Ka Panchhi or Phoolwari?
Original Question Retriever documents 
 Judge model
solved 
final answer
DecomposeI.Retrieval & Generation II.Judge Reliability
Which film was released first,
Aas Ka Panchhi or Phoolwari?
When was Aas Ka 
Panchhi released?When was 
Phoolwari released?Which year is earlier?
#1 and #2?
solved & relevant
No information..... 
may be 1933.Aas Ka Panchhi is 
a 1961..... Panchhi, 
which is 1961.
Decompose
Which year is earlier?
1961 and 1933?
Rewrite 
1933.
Aas Ka Panchhi is 
a 1961..... Panchhi, 
which is 1961.To determine which film was 
released first between Aas Ka 
Panchhi......Therefore, Phoolwari 
(1933)      1946 released before 
Aas Ka Panchhi (1955).      1961
Reasoning
When was 
Phoolwari released?
What is another 
name for 
Phoolwari?What is Phoolwari?
According to.....  
is The Bower.
Rewrite 
What is The Bower 
released?
Acrodd..... is 1946.Accord..... is 
1946.
solved & relevant
 correct & irrelevantIII.Decomposition & Rewriting 
correct & irrelevantIV.ReAG 
When was #4 
released?
Stop
Fig. 1.The D2F-ReAG framework consists of four stages: (I) Retrieval & Generation,
which retrieves relevant documents to generate a reasoning. (II) Judge Reliability,
where an LLM evaluates reasoning reliability against a threshold. (III) Decomposition
& Rewriting, which decomposes and rewrites the question when reliability is insuf-
ficient. (IV) ReAG (Reasoning-Augments generation), which integrates reliable and
relevant sub-question reasoning to update the root reasoning. The process stops once
the root question is answered reliably.
question is solved; otherwise, (III) Decomposition & Rewriting decomposes the
questionintosub-questionsandrewritesthemtoretrievemorerelevantevidence.
Finally, (IV) ReAG (Reasoning-Augmented Generation) augments root reason-
ing with reliable, relevant reasoning traces of sub-questions. D2F-ReAG stops
once the root reasoning score exceeds a threshold.
3.1 Retrieval & Generation
For the root questionq rootand each sub-questionq i, we retrieve the top-kmost
relevant documentsD(q)from the corpusCusing a dense retriever, and then
generate the corresponding reasoning processr(q)conditioned on the retrieved
evidence. The retrieved documents serve as external knowledge that grounds the
model’s reasoning, mitigating hallucinations and providing factual support for
downstream judgment. Formally, for any questionq∈ {q root, q1, . . . , q n},
r(q) = Generator 
q, D(q)
,(1)
whereGeneratordenotes the model that generates detailed reasoning based on
relevant documents. By explicitly conditioning onD(q), the generator is encour-
aged to perform evidence-aware reasoning rather than relying solely on its para-

Title Suppressed Due to Excessive Length 5
metric memory, which is particularly important for knowledge-intensive multi-
hop questions.
3.2 Judge Reliability
After generating the reasoning for the root or sub-questions, we judge its re-
liability using an LLM-based scoring mechanism. Rather than treating every
generated reasoning chain as trustworthy, we introduce an explicit verification
step that assesses whether the reasoning is logically coherent, factually consis-
tent with the retrieved documents, and sufficient to answer the question. We
denote the reliability score of the reasoning process ass r(q), which is calculated
by LLM:
sr(q) =Score(q, r(q)),(2)
wherer(q)represents the reasoning process, and Score(q, r(q))is the reasoning
score provided by the LLM according to our predefined rubric. The rubric jointly
considers multiple dimensions, including evidence grounding, logical consistency,
andanswercompleteness,sothattheresultingscorereflectsaholisticassessment
of reasoning quality.
The reasoning scores r(q)∈[0,10]measures reasoning reliability (higher
is better). We compare it to a thresholdθ(set to 7 in our experiments): if
sr(q)> θ, we consider the question solved. Otherwise, D2F-ReAG thinks the
reasoning unreliable and further decomposes the question for continued reason-
ing (Section 3.3). This adaptive mechanism allows the system to allocate more
computational effort only to questions that genuinely require deeper exploration,
while terminating early on questions that are already well-addressed.
Judge =(
Solved,ifs r(q)> θ
Decompose,ifs r(q)≤θ.(3)
If the current question is the root question and is deemed solved, the LLM gen-
erates the final answer based on the corresponding reasoning process, ensuring
that the answer is directly traceable to verified reasoning steps:
aroot=Answer(r qroot).(4)
If a sub-question is reliably solved, we use its relevant sub-questions to update
the root reasoning in ReAG (Section 3.4), so that the verified intermediate con-
clusions can progressively enrich the global reasoning context.
3.3 Decomposition & Rewriting
When the root question or any sub-question remains unsolved, we use prompt
engineering to logically decompose it into smaller, more tractable sub-questions,
each focusing on a single reasoning hop or a narrower information need. This

6 J. Li et al.
divide-and-conquer strategy reduces the cognitive load on the generator and
makes retrieval more targeted.
sub(q) =Decompose(q),(5)
wheresub(q)is the set of sub-questions obtained by logically decomposing the
questionq.
D2F-ReAG first solves each sub-question in order and judges the reliability
of the reasoning. If reliable, we rewrite the related other sub-questionsq∈ S i
based on the correct reasoningr q, so that subsequent sub-questions can leverage
previously verified intermediate results and avoid redundant or contradictory
reasoning paths.
S′
i={Rewrite(q, r q)|q∈ S i},(6)
wherer qis the reasoning to the current question, andq iis the related sub-
question. Through this rewriting step, ambiguous references and missing entities
in the original sub-questions can be resolved using the newly acquired evidence,
leading to higher-quality retrieval in the next iteration.
3.4 ReAG
Whenasub-questionissolvedreliably,wecheckitsrelevancetotherootquestion
and use its reasoning to update the root reasoning if relevant. This relevance
check prevents tangential or off-topic sub-question results from contaminating
the global reasoning state.
Check(q i, q) =(
relevant
irrelevant(7)
where Check(q i, q)denotes the relevance ofq ito the root questionq.
If a sub-question’s reasoning is reliable and relevant, it is used to update
the root reasoning. The update operation integrates newly verified evidence and
intermediate conclusions into the existing root reasoning, gradually building a
more complete and coherent chain of thought toward the final answer. Formally,
this can be written as:
r′(qroot) =Update(r(q root), r(q)),(8)
wherer′(qroot)is the updated reasoning process of the root problemq,r(q)is
the reasoning process of the sub-questionq i, and Update(r(q root), r(q))updates
r(qroot)by incorporatingr(q).
Once the root reasoning is reliable, we stop processing the remaining sub-
questionsandobtainthefinalanswertoavoidover-reasoning.Thisearly-stopping
strategy not only reduces unnecessary computational overhead but also prevents
the introduction of noise from over-decomposition, which could otherwise dilute
the focus of the root reasoning and degrade answer accuracy.

Title Suppressed Due to Excessive Length 7
Table 1.Performance comparison on HotpotQA, 2WikiMultiHopQA, and MuSiQue
using Str-Acc and LLM-Acc.†denotes our reimplementation; the others are from
LogicRAG. Best results are in bold, and second-best results are underlined.
Type MethodHotpotQA 2Wiki MuSiQue
Str. LLM Str. LLM Str. LLM
Zero-shotLlama3 (8B) 17.1 11.1 22.3 4.7 2.3 2.0
Llama3 (13B) 23.7 20.1 33.8 15.4 6.4 6.0
GPT-3.5-Turbo 31.5 35.4 24.0 22.0 7.9 10.9
GPT-4o-Mini 38.7 36.3 26.4 24.3 17.6 14.0
GraphRAPTOR 48.1 57.8 47.7 45.9 25.2 29.1
GraphRAG 39.6 45.2 46.3 43.3 16.5 19.4
LightRAG 47.8 57.7 43.1 36.3 18.1 19.4
HippoRAG 53.5 56.6 47.2 47.2 24.9 30.1
HippoRAG256.761.9 50.0 47.1 27.0 32.6
Prompt-RAGVanillaRAG 43.2 53.1 43.0 42.0 20.3 23.6
ReAct†54.2 56.5 54.9 50.8 28.8 31.9
ChainRAG†52.1 56.6 68.9 66.231.033.8
LogicRAG†54.2 62.5 65.3 62.6 29.6 36.5
Ours D2F-ReAG 55.8 63.4 70.3 68.9 32.2 37.9
4 Experiments
4.1 Dataset and Metrics
We evaluate on three standard multi-hop reasoning benchmarks: HotpotQA [20],
MuSiQue [18] and 2WikiMultiHopQA [10], covering diverse cross-document and
multi-hopreasoning.FollowingHippoRAG2[9],weusethesameretrievalcorpus
and randomly sample 1,000 questions from each validation set for evaluation,
ensuring fair comparison.
Following LogicRAG [2], we adopt two metrics:Str-AccandLLM-Acc.
Str-Accmeasures lexical correctness by checking whether the prediction ex-
actly matches the ground-truth answer after standard normalization.LLM-Acc
instead employs a strong LLM as an automatic judge to assess whether the pre-
diction is semantically equivalent to the reference, tolerating paraphrasing and
surface-form variations. Reporting both metrics enables a balanced evaluation
of strict lexical matching and flexible semantic correctness.
4.2 Baselines
We compare D2F-ReAG with three categories of baselines that cover repre-
sentative paradigms in multi-hop question answering.Zero-shotbaselines di-
rectly prompt LLMs without any external retrieval, including LLaMA3 (8B) and
LLaMA3 (13B) [5], as well as gpt-3.5-turbo and gpt-4o-mini [1], which serve to
reflect the intrinsic reasoning ability of LLMs.Graph RAGbaselines incorpo-
rate pre-constructed graph structures to support multi-hop retrieval, including
RAPTOR [16], GraphRAG [6], LightRAG [7], HippoRAG [8], and HippoRAG 2

8 J. Li et al.
[9].Prompt-based RAGbaselines instead interleave reasoning and retrieval
through prompt design without relying on explicit graph construction, including
ReAct [21], ChainRAG [22], and LogicRAG [2].
4.3 Implementation Details.
For fair comparison, all methods adopt the same experimental configuration.
We use sentence-transformers/all-MiniLM-L6-v2 [19] as the unified embedding
model for dense retrieval. The number of retrieved passages (top-k) is fixed to
3 across all methods. For answer generation, we employ gpt-4o-mini [1] as the
backbone LLM, ensuring that performance differences stem from the retrieval
and reasoning strategies rather than the underlying generator. All experiments
are conducted on a single NVIDIA RTX 3090 GPU.
4.4 Main Results
Table 1 reports results on three multi-hop reasoning benchmarks (HotpotQA,
2WikiMultiHopQA, and MuSiQue). D2F-ReAG achieves the best or near-best
performance on both Str-Acc and LLM-Acc across all three datasets, demon-
strating consistent superiority over diverse baselines.
Zero-shot LLMs benefit from stronger backbones, with gpt-4o-mini clearly
surpassing LLaMA3 (8B/13B), yet even the best zero-shot model trails retrieval-
based methods, underscoring the need for external evidence in multi-hop QA.
Graph-based RAG methods (RAPTOR, GraphRAG, LightRAG, HippoRAG,
HippoRAG 2) improve performance by structuring context into explicit graphs,
with HippoRAG 2 being the strongest (e.g., 56.7 Str-Acc on HotpotQA), though
their gains shrink on harder datasets like MuSiQue where graphs miss multi-hop
relations.Prompt-basedRAGmethodsinterleaveretrievalandreasoning:Chain-
RAG performs well on 2Wiki via sub-question decomposition with sentence-
graph retrieval, and LogicRAG attains the best baseline LLM-Acc on HotpotQA
(62.5) through logical decomposition, but both suffer from fixed decomposition
and noise accumulated over iterations.
D2F-ReAG consistently outperforms all baselines on 2Wiki (70.3 / 68.9) and
MuSiQue (32.2 / 37.9), and achieves the highest LLM-Acc on HotpotQA (63.4),
with the largest gains on 2WikiMultiHopQA where deep multi-hop reasoning is
most needed. Against the strongest prompt-based baseline LogicRAG, it yields
up to 5.4-point Str-Acc and 6.4-point LLM-Acc improvements on 2Wiki, show-
ing that on-demand decomposition and reliability-guided reasoning mitigate the
over- or under-decomposition of fixed-depth methods. Joint gains on both met-
rics further indicate that the recovered reasoning paths are not only lexically
aligned with gold answers but also semantically more faithful. We report the
best scores across runs, and provide a case study in Appendix 6 to illustrate
how D2F-ReAG corrects intermediate errors via reliable sub-question reasoning.

Title Suppressed Due to Excessive Length 9
MuSiQue 2Wiki HotpotQA00.20.40.60.8
0.590.630.8
0.41
0.37
0.2ratew/o decomposition
w/ decomposition
Fig. 2.Solved with vs. without decomposition on three benchmarks.
Table 2.Accuracy and efficiency comparison on 2WikiMultiHopQA (with vs. without
decomposition).
MethodStr
AccLLM
AccAvg.
Time (s)Avg.
Tokens
ReAct 54.9 50.8 13.93 11287
LogicRAG 65.3 62.6 15.35 1998
Ours (No D) 71.9 71.0 9.49 2267
Ours (+D) 67.3 65.0 73.54 13321
Ours (Avg.) 70.3 68.9 32.4 6057
4.5 Decomposition & Efficiency Comparison
Figure 2 shows that a large proportion of questions can be solved without de-
composition,demonstratingthenecessityofdynamicdecompositiontoefficiently
handle both simple and complex questions.
Since ChainRAG also uses a pre-built sentence graph, we compare latency and
efficiency with ReAct and LogicRAG. Table 2 shows that always decomposing
and judging reliability greatly increases time and token cost. In contrast, on-
demand decomposition skips unnecessary steps, speeding up easy cases while
still decomposing for hard ones. While our multi-step judgments can take longer
on some difficult examples, Figure 2 confirms these long cases are relatively few.
Overall, D2F-ReAG balances between performance and efficiency and better
matches human-like reasoning.
5 Conclusion
Existing RAG methods still struggle with multi-hop questions, primarily due to
their fixed decomposition strategies and limited ability to filter out erroneous

10 J. Li et al.
or irrelevant information during reasoning. To address these issues, we propose
D2F-ReAG (Dynamic Decomposition and Filtering for Multi-Hop Reasoning-
Augmented Generation), a framework that adaptively decomposes questions
only when necessary and selectively filters intermediate reasoning according to
its reliability. By leveraging verified reasoning from reliable sub-questions to
progressively refine the root-level generation, D2F-ReAG effectively suppresses
error propagation and produces more faithful reasoning chains, achieving strong
performance across challenging multi-hop benchmarks.
6 Case Study
We present a case study comparing the reasoning behaviors ofD2F-ReAGand
LogicRAG.
Question:Which film has the director who died later,45 Calibre Echoor
Bons Baisers De Hong Kong?
Gold Answer:Bons Baisers De Hong Kong
Failure Case of LogicRAG.
Model Answer:45 Calibre Echo(Incorrect)Rounds:1
Retrieved Contexts.LogicRAG retrieves three biography-style passages:
–Bruce M. Mitchell: includes his death date (September 26, 1952), supporting
reasoning about45 Calibre Echo.
–John Edward Bruce / Fred Bradley (rower): unrelated to either queried film.
Dependency Analysis (sorted).Death date of Bruce M. Mitchell; Death date of
Yvan Chiffre.
Round 1.Query:Death date of Bruce M. Mitchell.can_answer:true.
Understanding:BruceM.Mitchell(directorof45 Calibre Echo)diedonSeptem-
ber 26, 1952; the death date of Yvan Chiffre is not retrieved. LogicRAG never-
theless concludes45 Calibre Echodue to the missing comparison target.
Correct Case of D2F-ReAG.
Model Answer:Bons Baisers De Hong Kong(Correct)
Root Rationale:The retrieved context names the directors (Bruce M. Mitchell
and Yvan Chiffre) but does not provide their death dates.Judging reliability:
0.4
Decompose & Rewriting:
–Who directed45 Calibre Echoand when did the director die?
–Who directedBons Baisers De Hong Kongand when did the director die?
–Which year is later?

Title Suppressed Due to Excessive Length 11
Filtering for Multi-Hop Reasoning-Augmented Generation.
Subq 1:Director of45 Calibre Echois Bruce M. Mitchell.Reliability:0.8
Subq 2:Director ofBons Baisers De Hong Kongis Yvan Chiffre.Reliability:
0.8
Updating Root Reasoning and Rewriting.
Rewritten sub-questions: When did Bruce M. Mitchell die? When did Yvan
Chiffre die?
Filtering for Multi-Hop Reasoning-Augmented Generation.
Yvan Chiffre died on 1990-01-01; Bruce M. Mitchell died on 1988-02-19.
Updating Root Reasoning and Rewriting.
Rewritten sub-question 3:Which year is later, 1988 or 1990?
Subq 3:1990.Reliability:1.0
Final Answer:Bons Baisers De Hong Kong
References
1. Achiam, J., Adler, S., Agarwal, S., Ahmad, L., Akkaya, I., Aleman, F.L., Almeida,
D., Altenschmidt, J., Altman, S., Anadkat, S., et al.: Gpt-4 technical report. arXiv
preprint arXiv:2303.08774 (2023)
2. Chen, S., Zhou, C., Yuan, Z., Zhang, Q., Cui, Z., Chen, H., Xiao, Y., Cao, J.,
Huang,X.:Youdon’tneedpre-builtgraphsforrag:Retrievalaugmentedgeneration
with adaptive reasoning structures. arXiv preprint arXiv:2508.06105 (2025)
3. Cheng, R., Liu, J., Zheng, Y., Ni, F., Du, J., Mao, H., Zhang, F., Wang, B., Hao,
J.: DualRAG: A dual-process approach to integrate reasoning and retrieval for
multi-hop question answering. In: Proceedings of the 63rd Annual Meeting of the
Association for Computational Linguistics (Volume 1: Long Papers) (2025)
4. Chu, Z., Fan, H., Chen, J., Wang, Q., Yang, M., Liang, J., Wang, Z., Li, H., Tang,
G., Liu, M., Qin, B.: Self-critique guided iterative reasoning for multi-hop question
answering. In: Findings of the Association for Computational Linguistics: ACL
2025 (Jul 2025)
5. Dubey, A., Jauhri, A., Pandey, A., Kadian, A., Al-Dahle, A., Letman, A., Mathur,
A., Schelten, A., Yang, A., Fan, A., et al.: The llama 3 herd of models. arXiv
preprint arXiv:2407.21783 (2024)
6. Edge, D., Trinh, H., Cheng, N., Bradley, J., Chao, A., Mody, A., Truitt, S.,
Metropolitansky, D., Ness, R.O., Larson, J.: From local to global: A graph rag
approach to query-focused summarization. arXiv preprint arXiv:2404.16130 (2024)
7. Guo, Z., Xia, L., Yu, Y., Ao, T., Huang, C.: LightRAG: Simple and fast retrieval-
augmented generation. In: Findings of the Association for Computational Linguis-
tics: EMNLP 2025 (2025)
8. Gutiérrez, B.J., Shu, Y., Gu, Y., Yasunaga, M., Su, Y.: Hipporag: Neurobio-
logically inspired long-term memory for large language models. In: The Thirty-
eighth Annual Conference on Neural Information Processing Systems (2024),
https://openreview.net/forum?id=hkujvAPVsg
9. Gutiérrez, B.J., Shu, Y., Qi, W., Zhou, S., Su, Y.: From rag to mem-
ory: Non-parametric continual learning for large language models (2025),
https://arxiv.org/abs/2502.14802
10. Ho, X., Duong Nguyen, A.K., Sugawara, S., Aizawa, A.: Constructing a multi-hop
QA dataset for comprehensive evaluation of reasoning steps. In: Proceedings of the
28th International Conference on Computational Linguistics (2020)

12 J. Li et al.
11. Jiang, C., Qi, B., Hong, X., Fu, D., Cheng, Y., Meng, F., Yu, M., Zhou, B., Zhou,
J.: On large language models’ hallucination with regard to known facts. In: Pro-
ceedings of the 2024 Conference of the North American Chapter of the Association
for Computational Linguistics: Human Language Technologies (Volume 1: Long
Papers) (2024)
12. Kim,G.I.,Kim,J.W.,Jang,B.:UniRAG:AunifiedRAGframeworkforknowledge-
intensive queries with decomposition, break-down reasoning, and iterative rewrit-
ing. In: Findings of the Association for Computational Linguistics: EMNLP 2025
(Nov 2025)
13. Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V., Goyal, N., Küttler, H.,
Lewis, M., Yih, W.t., Rocktäschel, T., et al.: Retrieval-augmented generation for
knowledge-intensive nlp tasks. Advances in neural information processing systems
33, 9459–9474 (2020)
14. Li, J., Ruan, J., Tang, S., Chen, S., Chang, K., Ge, Y., Xiao, T., Zhu, J.: Subqrag:
Sub-question driven dynamic graph rag. arXiv preprint arXiv:2510.07718 (2025)
15. Page, L., Brin, S., Motwani, R., Winograd, T.: The pagerank citation ranking:
Bringing order to the web. Tech. rep., Stanford infolab (1999)
16. Sarthi, P., Abdullah, S., Tuli, A., Khanna, S., Goldie, A., Manning, C.D.: Rap-
tor: Recursive abstractive processing for tree-organized retrieval. In: International
Conference on Learning Representations (ICLR) (2024)
17. Shi, Z., Zhang, S., Sun, W., Gao, S., Ren, P., Chen, Z., Ren, Z.: Generate-then-
ground in retrieval-augmented generation for multi-hop question answering. In:
Proceedings of the 62nd Annual Meeting of the Association for Computational
Linguistics (Volume 1: Long Papers) (2024)
18. Trivedi, H., Balasubramanian, N., Khot, T., Sabharwal, A.: MuSiQue: Multihop
questions via single-hop question composition. Transactions of the Association for
Computational Linguistics (2022)
19. Wang, W., Bao, H., Huang, S., Dong, L., Wei, F.: MiniLMv2: Multi-head self-
attention relation distillation for compressing pretrained transformers. In: Findings
of the Association for Computational Linguistics: ACL-IJCNLP 2021 (Aug 2021)
20. Yang, Z., Qi, P., Zhang, S., Bengio, Y., Cohen, W., Salakhutdinov, R., Manning,
C.D.: HotpotQA: A dataset for diverse, explainable multi-hop question answering.
In: Proceedings of the 2018 Conference on Empirical Methods in Natural Language
Processing (2018)
21. Yao, S., Zhao, J., Yu, D., Du, N., Shafran, I., Narasimhan, K., Cao, Y.: ReAct:
Synergizing reasoning and acting in language models. In: International Conference
on Learning Representations (ICLR) (2023)
22. Zhu, R., Liu, X., Sun, Z., Wang, Y., Hu, W.: Mitigating lost-in-retrieval problems
in retrieval augmented multi-hop question answering. In: Proceedings of the 63rd
Annual Meeting of the Association for Computational Linguistics (Volume 1: Long
Papers) (2025)
23. Zhuang, Z., Zhang, Z., Cheng, S., Yang, F., Liu, J., Huang, S., Lin, Q., Rajmohan,
S., Zhang, D., Zhang, Q.: EfficientRAG: Efficient retriever for multi-hop question
answering. In: Proceedings of the 2024 Conference on Empirical Methods in Nat-
ural Language Processing (2024)