# KGCaRe: Explainable Complex Conditional Question Answering using Automatic Knowledge Graph Construction and Context Retrieval with LLMs

**Authors**: Ghanshyam Verma, Simanta Sarkar, Devishree Pillai, Hotaka Shiokawa, Yourong Xu, Fiona Veazey, Peter Hubbert, Hui Su, Paul Buitelaar

**Published**: 2026-08-10 16:05:58

**PDF URL**: [https://arxiv.org/pdf/2608.09779v1](https://arxiv.org/pdf/2608.09779v1)

## Abstract
Answering complex conditional questions using Large Language Models (LLMs) and Retrieval-Augmented Generation (RAG) remains a challenge, particularly in domain-specific contexts where general-purpose LLMs and RAG tend to underperform. We hypothesize that augmenting RAG with unstructured and structured knowledge, extracted from both documents and knowledge graphs (KGs), can improve reasoning and answer accuracy for such tasks.
  To test this, we propose KGCaRe, a hybrid approach that combines neural retrieval with symbolic reasoning over LLM-generated KGs. KGCaRe constructs a KG from documents using a multi-prompt extraction strategy and stores it in a graph database. Simultaneously, the documents are embedded into a vector store to enable neural retrieval. KGCaRe performs innovative iterative graph traversal guided by the LLM to extract relevant triples, prune irrelevant information, and uses additional clue entities to traverse the graph again if the initial traversal does not provide satisfactory context to generate the answer. The relevant triples extracted from the KG in path form, along with semantically retrieved text passages, are then fed into custom KGCaRe prompts to generate answers to the complex conditional questions with explanations.
  We evaluate KGCaRe on two complex conditional QA datasets. Our results on these datasets show that KGCaRe consistently outperforms existing baselines, including Vanilla LLM, Code Prompt, Text Prompt, Think-on-Graph, Vanilla RAG, and HybridContextQA, across multiple LLMs such as Mistral, Mixtral, GPT-3.5, and GPT-4o. We publicly release the software pipeline that we developed to implement the proposed KGCaRe approach.

## Full Text


<!-- PDF content starts -->

KGCaRe: Explainable Complex Conditional
Question Answering using Automatic Knowledge
Graph Construction and Context Retrieval with
LLMs
Ghanshyam Verma1*, Simanta Sarkar1, Devishree Pillai1,
Hotaka Shiokawa2, Yourong Xu2, Fiona Veazey3,
Peter Hubbert3, Hui Su2, Paul Buitelaar1
1*Insight Research Ireland Centre for Data Analytics, Data Sc ience
Institute, University of Galway, IDA Business Park, Galway , H91
AEX4, Co. Galway, Ireland.
2Fidelity Investments, Boston, Massachusetts, USA.
3Fidelity Investments, Dublin, Co. Dublin, Ireland.
*Corresponding author(s). E-mail(s):
ghanshyam.verma@universityofgalway.ie ;
Contributing authors: simanta.sarkar@universityofgalway.ie ;
devishree.pillai23@gmail.com ;hotaka.shiokawa@fmr.com ;
yourong.xu@fmr.com ;ﬁona.veazey@fmr.com ;peter.hubbert@fmr.com ;
Hui.Su@fmr.com ;paul.buitelaar@universityofgalway.ie ;
Abstract
Answering complex conditional questions using Large Langu age Models (LLMs)
and Retrieval-Augmented Generation (RAG) remains a challe nge, particularly
in domain-speciﬁc contexts where general-purpose LLMs and RAG tend to
underperform. We hypothesize that augmenting RAG with unst ructured and
structured knowledge, extracted from both documents and kn owledge graphs
(KGs), can improve reasoning and answer accuracy for such ta sks.
To test this, we propose KGCaRe, a hybrid approach that combi nes neural
retrieval with symbolic reasoning over LLM-generated KGs. KGCaRe constructs
a KG from documents using a multi-prompt extraction strateg y and stores it in a
graph database. Simultaneously, the documents are embedde d into a vector store
1
arXiv:2608.09779v1  [cs.CL]  10 Aug 2026

to enable neural retrieval. KGCaRe performs innovative ite rative graph traver-
sal guided by the LLM to extract relevant triples, prune irre levant information,
and uses additional clue entities to traverse the graph agai n if the initial traver-
sal does not provide satisfactory context to generate the an swer. The relevant
triples extracted from the KG in path form, along with semant ically retrieved
text passages, are then fed into custom KGCaRe prompts to gen erate answers
to the complex conditional questions with explanations.
We evaluate KGCaRe on two complex conditional QA datasets. O ur results
on these datasets show that KGCaRe consistently outperform s existing base-
lines, including Vanilla LLM, Code Prompt, Text Prompt, Thi nk-on-Graph,
Vanilla RAG, and HybridContextQA, across multiple LLMs suc h as Mistral,
Mixtral, GPT-3.5, and GPT-4o. We publicly release the softw are pipeline that
we developed to implement the proposed KGCaRe approach.
Keywords: Natural Language Processing, Knowledge Graph, Large languag e models,
Retrieval-Augmented Generation, Complex Question Answering
1 Introduction
Answering domain-speciﬁc conditional questions using LLMs presen ts a signiﬁcant
challenge. Although LLMs perform well on general-purpose QA task s, their eﬀective-
ness diminishes considerably when the input involves conditional logic o r requires
reasoning across domain-speciﬁc content [ 1,2]. One of the reasons behind this per-
formance drop is the absence of ﬁne-tuning on specialized corpora . Additionally,
when presented with insuﬃcient context, LLMs are prone to hallucin ate—generating
answers that appear plausible but lack grounding in the source mate rial [3]. Condi-
tional questions are particularly sensitive to missing or implicit constr aints; the LLM
must be aware of both the question and the conditions under which a n answer is valid.
These challenges point to the need for enhanced strategies that g o beyond prompting
or ﬁne-tuning.
A promising direction is to incorporate external knowledge sources such as knowl-
edgegraphs(KGs),which organizeinformationinsemanticallystruc tured,explainable
formats [ 4,5]. KGs can complement LLMs by providing factual precision and enablin g
symbolic reasoning [ 6]. However, extracting high-quality triples from raw domain-
speciﬁc documents can be error-prone and incomplete, limiting the u sefulness of the
KG. On the other hand, semantic vector retrieval from the same d ocuments oﬀers
a ﬂexible way to supplement the LLM’s understanding with textual ev idence. This
motivates our hypothesis: a hybrid context composed of both sym bolic and neural
representations could improve performance on complex conditiona l QA tasks.
To validate this, we propose a new approach called KGCaRe ( Knowledge Graph
ContextawareReasoning for complex Question Answering). It combines symbolic
retrieval from a LLM-generated KG with vector-based semantic s earch from the same
document corpus. The KG is constructed using a multi-stage promp t strategy and
stored in a Neo4j database [ 7,8], while the vector index is built using embeddings
stored in Facebook AI Similarity Search (FAISS) [ 9] supported vector storage. During
2

inference, KGCaRe performs iterative graph traversalover the KG to identify relevant
triples and combine them with semantically retrieved passages. This f used context is
then used to generate answers that are both accurate and expla inable.
We evaluate KGCaRe using two publicly available QA datasets: Condition alQA
[10] and HotpotQA [ 11]. ConditionalQA contains complex, multi-hop, conditional
questions derived from UK policy documents. HotpotQA contains co mplex multi-hop
questions derived from Wikipedia articles. We also conduct an ablation study to eval-
uate the performance of the proposed and existing approaches o n speciﬁc types of
questions.
The key contributions of this work are:
•We design and implement an end-to-end pipeline for automatic KG cons truction
from long documents with complex structure, using a multi-prompt L LM-based
triple extraction approach.
•We introduce KGCaRe, a hybrid approach that leverages both KG an d vector
context for answering complex conditional questions.
•We empirically show that KGCaRe outperforms baseline methods such as
Vanilla LLM, Code Prompt, Text Prompt, Think-on-Graph, Vanilla RAG , and
HybridContextQA across multiple LLMs.
•We publicly release the end-to-end software pipeline code and the as sociated
prompts used to implement our approach1, to support reproducibilityand further
research.
The rest of the paper is structured as follows. In Section 2, we describe related
work. Section 3describes the ConditionalQA and HotpotQA datasets. In Section 4,
we explain our proposed approach. Section 5describes the experimental design that
we used. In Section 6, we discuss and compare results in detail. Finally, we conclude
in Section 7.
2 Related Work
Despite impressive gains in general-domain question answering, LLMs continue to
struggle with tasks requiring multi-hop, conditional reasoning over long and complex
documents. In this context, the ConditionalQA dataset introduce d by Sun et al. [ 10]
stands out for its focus on realistic, policy-based QA that includes m ultiple reasoning
steps and conditional constraints.
Puerto et al. proposed a code prompting strategy [ 12] that converts natural lan-
guage queries and context from documents into code, which is then used as input to
the LLM prompts for answer generation. While this approach provid es the query and
context in a more machine-understandable format, it lacks explicit in corporation of
structured knowledge, such as knowledge from KGs.
Combining KGs with LLMs is a growing area of research. Prior works ha ve shown
that knowledge graphs can enhance factual grounding, improve f aithfulness, and sup-
port reasoning in domains like law and medicine [ 1]. For instance, MindMap [ 13]
1https://github.com/GhanshyamVerma/KGCaRe
3

integrates biomedical triples from EMCKG into LLM prompts and show s gains on
medical QA benchmarks.
Think-on-Graph [ 14] and Think-on-Graph 2.0 [ 15] perform iterative beam search
with the help of LLMs to extract context from the KG and use that c ontext to gen-
erate the answers. Luo et al. proposed a method called Reasoning o n Graphs (RoG)
that synergises LLMs with KGs so that faithful and interpretable r easoning can be
performed [ 16]. These approachescan uncover multi-hop paths, but they often depend
on dense graph connectivity. In contrast, our approach works e ven when the KG is
sparse by supplementing it with semantically retrieved passages.
Retrieval-Augmented Generation (RAG) frameworks have also bee n explored as a
solution for grounding LLMs in external documents [ 17]. However, most RAG imple-
mentations rely on simple keyword or similarity search, which may fail t o retrieve
critical conditions or logical dependencies [ 18].
HybridContextQA[ 19]isanexistingRAG-basedhybridapproachthatusescontext
from both documents and KGs to generate answers to complex and conditional ques-
tions. While HybridContextQA has shown promising results on the Con ditionalQA
dataset, it performs simple keyword-based search to extract co ntext from KGs.
Tree-of-Traversal [ 20] is a zero-shot approach that enables augmentation of LLMs
with one or more KGs to perform tree search over KGs. This approa ch signiﬁcantly
improves performance on KG question answering tasks; however, it does not consider
context extraction from documents for answer generation.
To handle these issues, our KGCaRe approach combines symbolic gra ph traversal
and semantic retrieval into a uniﬁed pipeline for better answer gene ration of complex
and conditional questions.
3 Datasets
We evaluate our approach on two publicly available benchmark complex QA datasets.
The ﬁrst dataset is ConditionalQA [ 10], which has been speciﬁcally designed to test
a model’s capability in multi-hop reasoning and conditional answer gene ration. The
dataset is constructed from UK public policy documents and reﬂect s realistic scenar-
ios where understanding legal and procedural nuances is essentia l for answering the
questions correctly.
The ConditionalQA dataset includes four answer categories: yes/n o answers, span-
based answers that involve extracting text from the document, c onditional answers
which are only valid when certain criteria are met, and not-answerab le cases where
the document does not contain suﬃcient information to produce a v alid response. One
of the key challenges in ConditionalQA is that conditions relevant to th e answer are
often not explicitly stated in the question but are buried within variou s sections of
the document. As a result, a model must perform multi-hop reason ing to locate and
assemble these conditions into a coherent justiﬁcation for its answ er.
The second QA dataset is HotpotQA [ 11], a publicly available large-scale complex
QA dataset that requires multi-hop reasoning to answer the quest ions. This dataset
mainly contains two types of multi-hop questions: yes/no type and s pan type. In this
dataset, there is no speciﬁc categoryof conditional-type questio ns that require explicit
4

conditions to be generated along with the answer. However, the qu estions require
implicit conditions to be checked in order to generate accurate answ ers, or involve
comparison between two or more entities based on shared propert ies [11]. Please refer
to Appendix Afor further details on the datasets.
4 Proposed Approach
KGCaRe introduces a comprehensive and modular architecture des igned to support
explainable reasoning across both unstructured and structured sources of informa-
tion. The core of the KGCaRe approach lies in its hybrid retriever tha t seamlessly
integrates symbolic reasoning over a knowledge graph with semantic similarity-based
vector retrieval. Below, we detail each component of the propose d KGCaRe approach.
4.1 Multi-Prompt Triple Extraction and Knowledge Graph
Construction
To enable precise and context-rich knowledge representation, KG CaRe employs a
multi-stage, prompt-driven approach for KG construction using L LMs. This process is
designedtoextracthighlyaccurateandlogicallyconnected(subje ct, predicate,object)
triples from raw document text, using a pipeline of progressive prom pt stages. These
prompt stages are as follows:
•Stage 1: Contextual Understanding and Initial Entity-Rela tion
Extraction
The ﬁrst stage uses a prompt (MLT PROMPT 1) that instructs the LLM to
analyze the input text deeply, identify key entities, and extract all possible rela-
tions. This prompt emphasizes understanding temporal, causal, pr ocedural, and
conditional constructs within the document. It encourages the m odel to extract
granular, context-aware triples by ﬁrst summarizing the purpose and conditions
within the text and then listing entity-relationship pairs. For example :
(‘Applicant’, ‘must submit’, ‘form’)
(‘form’, ‘must be submitted within’, ‘30 days’)
•Stage 2: Conditional and Alternative Scenario Expansion
Next, a second prompt (MLT PROMPT 2) is used to enhance the previously
extracted content by focusing on conditional logic, exceptions, a nd alternatives
within the text. This step is crucial for capturing real-world nuance s such as
regulatoryconditions,proceduraldependencies,andbranching decisionlogic.The
prompt instructs the LLM to explicitly recognize if-then structure s, alternatives,
and exceptions, and to augment the triple set accordingly. For inst ance:
Condition: If consent is not given, permission must be obtained from the court.
Triples: (‘Consent’, ‘not given’, ‘get permission from court’)
•Stage 3: Reﬁnement and Logical Graph Expansion
The third prompt (MLT PROMPT 3) reﬁnes the extracted triples by validating
their completeness, enhancing speciﬁcity, and interlinking them for better logical
consistency.ItpushestheLLMtoderivenewtriplesfromtheprev iouslyextracted
5

setusingtransitiveorimpliedlogic.Thisstepensuresthattheﬁnalg raphincludes
both explicitly stated and inferable relationships. For example:
Given:
(‘Guardian’, ‘needs consent from’, ‘Parents’)
(‘Parents’, ‘can delegate consent to’, ‘Court of Protection’)
Inferred:
(‘Guardian’, ‘can delegate consent to’, ‘Court of Protection’)
•Stage 4: Output Normalization and Graph Ingestion
The ﬁnal output from these prompts is formatted as a set of triple s that pre-
serve the semantics of the source text. These triples are then st andardized and
ingested into a Neo4j graph database, enabling scalable storage an d advanced
traversal logic during the retrieval phase. This multi-prompt appr oach not only
improvesthe quality and structure ofextracted knowledgebut als oallows the KG
to represent intricate dependencies and nuanced regulatory logic that single-pass
extraction pipelines may miss (see Appendix B).
4.2 Semantic Vector Indexing
In parallel with KG construction, documents are embedded into high -dimensional
vectorrepresentationsusingan LLM-poweredembedder. These embeddings are stored
in a FAISS index [ 9], allowing fast approximate nearest-neighbor search. This vector
index captures the global semantic context of the documents and is used to retrieve
relevant passages based on the similarity to the user’s query.
4.3 Knowledge Graph Traversal
The central component of KGCaRe is its symbolic, LLM-guided KG tra versal mech-
anism, formally outlined in Algorithm 1. This traversal process enables KGCaRe to
iteratively explore and reason over the constructed KG to extrac t relevant contextual
information for complex question answering.
Given an input question q, the pipeline begins with a preprocessing step where
topic entities ( Etopic) are extracted using an LLM. These topic entities act as anchors
for initiating graph traversal. KGCaRe then initializes multiple memory m odules,
including the triple memory M, clue entity memory Eclue, clue triple memory Mclue,
traversal path memory Mpathtraversed, andvisitedentities.
Traversal proceeds in a depth-bounded iterative manner up to a m aximum depth
D. At each level, KGCaRe searches for triples within the KG. For the init ial depth
(d= 1), the retrieval is based on partial matching to allow broader exp loration,
including subword matches. For subsequent depths ( d >1), exact matching is used
to ensure more focused reasoning. In parallel, KGCaRe also search es for triples linked
to clue entities ( Eclue) using partial matching, enabling the discovery of supporting
evidencenotcapturedbytopicentitiesalone.Theuseoftheclueen tityinourKGCaRe
approachis inspired by the workof [ 20], who proposed the Tree-of-Traversalapproach.
6

Algorithm 1 Knowledge Graph Traversal in KGCaRe
Input:q(input question), D(maximum depth), LLM(language model)
Output :triples pruned, Memory ( M)
1:Initialize Etopicfromq, memory M, clue entities Eclue, clue memory Mclue, pathmemory
Mpathtraversed , andvisitedentities
2:fordepth= 1 toDdo
3:for alletopici∈Etopicdo
4: ifdepth== 1then
5: Find partial matches for etopiciwithvisitedentities
6: else
7: Find exact matches for etopiciwithvisitedentities
8: end if
9:end for
10:for allecluei∈Ecluedo
11: Find partial matches for eclueiwithvisitedentities
12:end for
13:triples pruned←PruneTriples( LLM,q, extracted triples)
14:Save candidates for next round using triples pruned,i,M,Mpathtraversed
15:Updatevisitedentities,M, andMpathtraversed
16:iftriples pruned/negationslash=∅then
17: Perform Reasoning( LLM,q,triples pruned,Mclue)
18: ifknowledge is suﬃcient then
19: returntriples pruned, Memory ( M)
20: else
21: UpdateEclueandMcluewith new clues
22: ifEclue/negationslash=∅orEtopic/negationslash=∅then
23: continue
24: else
25: break
26: end if
27: end if
28:else
29: Perform Reasoning( LLM,q)
30: UpdateEclueandMcluewith clues found
31: ifEclue/negationslash=∅orEtopic/negationslash=∅then
32: continue
33: else
34: break
35: end if
36:end if
37:end for
38:returntriples pruned, Memory ( M)
The retrieved triples are then passed through an LLM-powered pr uning function:
triples pruned←PruneTriples( LLM,q, triples)
7

This step ﬁlters out irrelevant or redundant triples, retaining only t hose deemed con-
textually useful for answering the input question. The pruned trip les are saved to
the memory Mand are also used to update the visited entity set and traversal pa th
memory.
Next, an LLM reasoning prompt is used to evaluate whether the acc umulated
context is suﬃcient to answer q. If so, KGCaRe generates the ﬁnal answer and halts
traversal. If not, the LLM is prompted to extract new clue entities that may assist in
further traversal. These new entities are appended to Eclue, and the process continues
to the next depth level. If no clue or topic entities remain, or no relev ant triples are
found, KGCaRe triggers a fallback mode where the LLM performs re asoning directly
over the question to extract additional clues for KG traversal.
If traversal reaches the maximum depth Dor suﬃcient context is retrieved from
the KG to generate the answer, it returns the KG context. KGCaR e invokes a ﬁnal
mechanism, where KGCaRe uses an LLM with a custom prompt to gene rate the
answer using all accumulated information from memory M(context from KG) and
the context retrieved from the vector store.
Throughout the process, KGCaRe maintains an explicit reasoning tr ace, including
the triples retrieved, pruned, and traversed, as well as the orde r in which entities were
visited. This traceability allows KGCaRe to generate not only an answe r but also
an explanation of the logical steps and conditions leading to that ans wer, ensuring
interpretability and transparency.
By dynamically combining symbolic graph traversal with LLM-guided clu e gen-
eration and pruning, KGCaRe enables robust, multi-hop reasoning o ver sparse or
incomplete KGs in complex conditional question answering scenarios.
4.4 Combined Retrieval and Answer Generation
Once relevant information has been gathered from both the KG tra versal and the
semanticvectorstore,KGCaRemergesthesetwocontextsourc esintoauniﬁedprompt
for ﬁnal answer generation. This fusion stage ensures that the L LM beneﬁts from the
structuredprecisionofsymbolicreasoningandthesemanticrichne ssofneuralretrieval.
The graph-based retriever contributes a curated set of triples a nd their associated
traversal paths, representing the logical structure of the rea soning process. Simulta-
neously, the semantic retriever returns top-ranked passages f rom the FAISS vector
index, capturing global textual context relevant to the question . These passages may
include supporting explanations, alternate phrasings, or implicit con ditions that are
not explicitly modeled in the KG.
Both the symbolic and neural contexts are formatted into a custo m structured
prompt that instructs the LLM to synthesize a coherent and accu rate answer. The
promptexplicitly asksthe modeltoincorporateevidencefromboth sourcesand,where
applicable, to identify and state any conditions under which the answ er holds. This
is particularly important for conditional questions, where part of t he answer often
depends on latent assumptions or constraints embedded in the ret rieved context.
By combining symbolic and neural retrieval, KGCaRe improves not only answer
accuracy but also explanation quality. This integrated reasoning st rategy is essential
for complex multi-hop questions that require connecting disparate facts and resolving
8

conditional logic, especially in domains such as public policy, law, or regu lationswhere
answer faithfulness and traceability are critical.
5 Experimental Design
To evaluate the eﬀectiveness of KGCaRe, we design a series of expe riments to com-
pare against several baselines and existing state-of-the-art me thods. The objective of
these experiments is to examine how well KGCaRe performs in answer ing complex,
conditional questions compared to existing KG-based, prompt-ba sed, and RAG-based
approaches.
We conduct comparative evaluations against two baseline prompting techniques:
Text Prompt andCode Prompt , as introduced by Puerto et al. [ 12]. These approaches
operate solely on raw document context without incorporating any structured knowl-
edge. In parallel,we benchmark our model against Think-on-Graph [14], which utilizes
a KG to guide the LLM through a structured reasoning process. Th is comparison is
crucial to assess the improvements brought by our hybrid design o ver purely symbolic
KG traversal methods.
We also perform comparative evaluations against Vanilla LLM, where t he LLM is
prompted directly using the raw question without any external con text, and Vanilla
RAG,whereastandardRAGpipeline retrievespassagesfromthedo cumentsandfeeds
themintotheLLMwithout incorporatinganyKG-basedreasoning.T heseexperiments
are conducted using four LLMs: Mistral [ 21], Mixtral [ 22], GPT-3.5, and GPT-4o.
Furthermore, we evaluate our method against an existing hybrid ap proach called
HybridContextQA [ 19].
We performed the evaluation using two datasets: ConditionalQA and HotpotQA.
For both datasets, the development set is used as the evaluation b enchmark for all
comparative experiments, as the test set is not publicly available. Co nditionalQA
consists of 2,338 training QA pairs and 271 development QA pairs, eac h associated
with UK policy documents. HotpotQA consists of 90,564 QA pairs in the training
set and 500 QA pairs in the development set. Please refer to Append ixAfor further
details on the datasets. We use some QA examples from the training s et for few-shot
prompting to provide the LLMs with in-context examples.
For all experiments, we test KGCaRe and baseline models using four d iﬀerent
LLMs: GPT-3.5, GPT-4o, Mistral, and Mixtral. Please refer to Appen dixD.2for
implementation details and Appendix D.3for experimental environment details. This
diverse set of LLMs allows us to evaluate the generalizability of our ap proach across
both proprietary and open-source LLMs, and to observe how mod el scale and archi-
tecture inﬂuence the eﬀectiveness of hybrid retrieval strategie s. Similar to [ 19], we use
the exact match based F1 score as an evaluation metric. Please ref er to Appendix D.1
for further details on the evaluation metric.
Through these experiments, we aim to provide a comprehensive ana lysis of
KGCaRe’s strengths and limitations in the context of complex and con ditional QA
task, and to demonstrate its consistent advantages over both R AG and KG-only
baselines.
9

6 Results
We evaluate KGCaRe using two datasets across four diﬀerent LLMs : Mistral, Mixtral,
GPT-3.5, and GPT-4o. The ConditionalQA dataset contains 271 QA pa irs spanning
Yes/No, span-based, and conditional answer types (see Table 1). The performance of
the proposed KGCaRe is compared against several baselines, includ ing Vanilla LLM,
Code Prompt, Text Prompt, Think-on-Graph, Vanilla RAG, and Hybr idContextQA.
Table1shows that on both the datasets across all models, the proposed KGCaRe
approach consistently achieves the highest performance conside ring all the questions
(Avg F1 Score), validating the eﬀectiveness of the hybrid retrieva l and reasoning
strategy.
On ConditionalQA, for the Mistral model, baseline methods perform p oorly, with
Vanilla LLM and Code Prompt achieving an average F1 of 31.84 and 28.26 , respec-
tively, and Text Prompt scoring 27.36. The KG-only approach (Think -on-Graph)
performs worse at 16.24, reﬂecting its limitations in better context retrieval from
KG and answer generation for the complex conditional questions in c omparison to
other approaches. Vanilla RAG and HybridContextQA improve perfo rmance to 40.12
and 45.17, respectively, while KGCaRe signiﬁcantly boosts the avera ge F1 to 57.89.
Notably, it achieves strong gains in the conditional category (F1 = 5 7.17) and Yes/No
category (F1 = 73.42), underscoring its ability to handle multi-hop an d condition-
sensitive reasoning even with small models. On HotpotQA, for the Mis tral model,
KGCaRe outperforms all other existing approaches for all QA cate gories.
Mixtral, a more capable open-source model, shows higher baseline pe rformance.
On ConditionalQA, Vanilla LLM, Code Prompt and Text Prompt achieve4 1.35, 40.88
Table 1 Results of diﬀerent prompting and context integration appro aches for various LLMs on
the ConditionalQA and HotpotQA datasets.
LLM/
Model
usedApproachDataset 1: ConditionalQA Dataset 2: HotpotQA
Avg F1 Score
[All ]F1 Score
[Yes/No ]F1 Score
[Span ]F1 Score
[Conditional ]Avg F1 Score
[All ]F1 Score
[Yes/No ]F1 Score
[Span ]
MistralVanilla LLM 31.84 52.83 8.4 25.11 24.31 68.00 22.01
Code Prompt 28.26 41.74 16.30 22.76 09.69 00.00 09.69
Text Prompt 27.36 31.58 25.64 23.51 16.94 04.00 17.62
Think-on-Graph 16.24 21.58 12.05 5.53 04.76 04.00 04.80
Vanilla RAG 40.12 52.88 25.86 21.11 49.41 56.00 49.06
HybridContextQA 45.17 60.00 33.55 34.39 51.16 32.00 52.17
KGCaRe (ours) 57.89 73.42 42.08 57.17 53.53 72.00 54.00
MixtralVanilla LLM 41.35 62.29 17.96 39.83 27.45 56.00 25.95
Code Prompt 40.88 44.80 40.99 34.62 11.07 28.00 10.18
Text Prompt 46.60 56.95 40.15 39.36 22.35 20.00 22.47
Think-on-Graph 17.40 23.14 12.90 05.54 04.87 00.00 05.13
Vanilla RAG 53.54 73.54 30.91 49.46 53.84 48.00 54.15
HybridContextQA 53.71 74.13 36.45 42.59 53.68 48.00 53.98
KGCaRe (ours) 59.45 78.32 38.37 47.93 58.27 56.00 58.39
GPT 3.5Vanilla LLM 33.02 61.88 24.27 35.10 38.86 76.00 36.91
Code Prompt 48.27 70.20 23.76 41.36 54.11 84.00 52.54
Text Prompt 57.15 73.13 45.54 46.25 60.23 88.00 58.77
Think-on-Graph 16.29 13.97 20.67 09.48 14.32 16.00 14.24
Vanilla RAG 57.79 71.91 42.01 54.27 61.84 96.00 60.05
HybridContextQA 55.03 71.21 42.97 43.52 61.79 92.00 60.20
KGCaRe (ours) 60.01 72.02 46.59 54.83 71.48 88.00 70.61
GPT 4oVanilla LLM 34.72 58.50 08.15 14.78 50.53 96.00 48.15
Code Prompt 53.29 76.92 26.89 36.36 68.82 92.00 67.60
Text Prompt 59.16 81.82 40.32 48.20 65.07 84.00 64.07
Think-on-Graph 20.40 21.45 21.46 07.85 10.79 00.00 11.36
Vanilla RAG 62.26 79.48 43.01 45.32 65.05 92.00 63.64
HybridContextQA 63.71 81.58 50.71 51.99 73.37 96.00 72.18
KGCaRe (ours) 67.55 83.10 50.05 48.90 80.21 96.00 79.38
10

and46.60averageF1scores,respectively,withThink-on-Grapha gainunderperforming
at 17.40. Vanilla RAG and HybridContextQA perform well with an avera ge F1 of
53.54 and 53.71, respectively. KGCaRe outperforms all baselines wit h an average F1
of 59.45,showing robust performance. On HotpotQA, for the Mixt ral model, KGCaRe
again outperforms all other existing approaches for all QA catego ries. These results
highlight the value of hybrid retrieval in moderately strong LLMs, es pecially when
symbolic reasoning is paired with semantic context.
GPT-3.5 delivers strong baseline results, on ConditionalQA, with Vanilla LLM
(Avg F1 = 33.02), Code Prompt (Avg F1 = 48.27) and Text Prompt (Av g F1 =
57.15)outperformingtheThink-on-Graphmethod(AvgF1=16.29) byawidemargin.
Interestingly, Text Prompt yields the highest Yes/No F1 (73.13) ac ross all GPT-3.5
conﬁgurations. However, KGCaRe achieves the best overall perf ormance (Avg F1 =
60.01)andthehighestconditionalF1(54.83),aswellasthebestsp an-basedF1(46.59).
On HotpotQA, for the GPT-3.5 model, KGCaRe again outperforms all other existing
approaches for all QA categories, except Yes/No. This shows tha t while prompting
alone performs well, complex reasoning beneﬁts from the hybrid des ign.
As the most powerfulmodel in the comparison,GPT-4oachieveshig h scoresacross
all methods. On ConditionalQA, Vanilla LLM, Code Prompt and Text Pro mpt reach
average F1 scores of 34.72, 53.29 and 59.16. Think-on-Graph rema ins ineﬀective even
in this setting (Avg F1 = 20.40). HybridContextQA reaches an avera ge F1 of 63.71,
and KGCaRe improves further to 67.55, achieving the highest Yes/N o F1 (83.10)
and competitive results in the span (50.05) and conditional (48.90) c ategories. On
HotpotQA, for the GPT-4o model, KGCaRe again outperforms all ot her existing
approaches with the highest F1 score in all categories, except the Yes/No category,
where HybridContextQA and KGCaRe achieve similar F1 scores. Thes e overall gains
conﬁrm the utility of symbolic augmentation even in high-capacity LLM s.
The results in Table 1clearly show that Think-on-Graph lags signiﬁcantly across
all models and question types. On HotpotQA, Think-on-Graph rema ins ineﬀective
and produces an F1 score of 0 with Mixtral and GPT-4o for the Yes/ No category.
We found that this occurs because both models generate verbose answers without
explicitly including the word “yes” or “no”. Vanilla RAG, Code and Text P rompt per-
form reasonably well on Yes/No and span questions, especially with s tronger LLMs.
KGCaRe consistently outperforms all baselines across models and q uestion types at
Avg F1 Score, and outperforms most models for Yes/No, Span, an d Conditional types,
demonstratingtheeﬀectivenessofintegratingsymbolicKGtraver salwith vector-based
semantic retrieval. These results support our core hypothesis: c ombining structured
and unstructured context, and reasoning over both, yields supe rior performance on
complex, conditional QA tasks, regardless of the underlying LLM’s c apacity. The
largest performance margins are observed in conditional and Yes/ No questions, where
reasoning steps are often multi-hop, conditional, and dispersed ac ross documents,
highlighting situations where the hybrid model excels.
6.1 Ablation Study
During error analysis on the ConditionalQA dataset, we observed th at out of the
136 Yes/No type questions, 30 had multiple valid answers in the groun d truth. To
11

Table 2 Comparative analysis of KGCaRe and existing approaches on
106 Yes/No QA pairs from the ConditionalQA dataset.
LLM / Model used ApproachF1 Score
[Yes/No type QA ]
MistralVanilla LLM 58.49
Code Prompt 40.58
Text Prompt 30.89
Think-on-Graph 39.94
Vanilla RAG 62.26
HybridContextQA 57.13
KGCaRe (ours) 69.81
MixtralVanilla LLM 64.22
Code Prompt 43.86
Text Prompt 57.34
Think-on-Graph 45.65
Vanilla RAG 74.84
HybridContextQA 76.27
KGCaRe (ours) 82.07
GPT 3.5Vanilla LLM 66.03
Code Prompt 72.07
Text Prompt 70.95
Think-on-Graph 44.34
Vanilla RAG 70.75
HybridContextQA 68.86
KGCaRe (ours) 71.75
GPT 4oVanilla LLM 71.69
Code Prompt 84.90
Text Prompt 85.85
Think-on-Graph 43.03
Vanilla RAG 83.96
HybridContextQA 80.97
KGCaRe (ours) 88.67
better evaluate the performance of KGCaRe and existing approac hes, we focused on
the subset of Yes/No questions with a single deﬁnitive answer. Table 2presents a
comparative analysis of KGCaRe and baseline methods on these 106 Y es/No QA pairs
having a single deﬁnitive answer.
For the Mistral model, the Vanilla LLM, Code Prompt, and Think-on-G raph base-
lines achieve modest F1 scores of 58.49, 40.58 and 39.94, respective ly (see Table 2),
while Text Prompt performs slightly worse at 30.89. The HybridConte xtQA and
Vanilla RAG improve performance signiﬁcantly to 57.13 and 62.26, resp ectively, and
KGCaRefurtherenhancesitto69.81,demonstratingthebeneﬁto fcombiningsymbolic
reasoning with neural retrieval even in smaller open-source LLMs.
With Mixtral, all approaches perform better than on Mistral, showin g Mixtral’s
stronger capability on the QA task. The baseline Vanilla LLM and Text P rompt
achieve 64.22 and 57.34, respectively, while Think-on-Graph perfor ms slightly lower
12

at 45.65. Vanilla RAG and HybridContextQA achieve an F1 score of 74.8 4 and 76.27,
respectively,and KGCaReagainimproveson this, achievingthe highe st scoreof82.07.
This shows that the hybrid reasoning strategy is particularly eﬀect ive when leveraged
with more powerful open-source models.
For GPT-3.5, the Code Prompt surprisingly performs very well with a n F1 score of
72.07,surpassing Vanilla LLM (66.03),Text Prompt (70.95), Think-o n-Graph (44.34),
Vanilla RAG (70.75), and even HybridContextQA (68.86). However, K GCaRe still
manages to slightly improve over HybridContextQA, reaching an F1 s core of 71.75.
Although the gainis smallercomparedtoopen-sourcemodels, this su ggeststhat GPT-
3.5 also beneﬁts from hybrid context to a limited but still positive exte nt.
In the case of GPT-4o, both Code Prompt and Text Prompt already perform
extremely well with F1 scores of 84.90 and 85.85, respectively. Desp ite this strong
baseline, KGCaRe still achieves the highest F1 score of 88.67, margin ally outperform-
ing all other approaches. This indicates that even for high-capacit y LLMs, KGCaRe
can oﬀer marginal yet meaningful improvements, particularly by en hancing reasoning
consistency and contextual faithfulness.
Across all LLMs, we observe that KGCaRe consistently outperfor ms the KG-only
(Think-on-Graph) and vector-only (Text Prompt, Code Prompt, and Vanilla RAG)
baselines, reaﬃrming the value of combining symbolic and neural retr ieval methods.
The largest relative gains are observed in Mistral and Mixtral, where the base model’s
reasoning capabilities beneﬁt most from structured traversal an d context integration.
In summary, Table 2illustrates that KGCaRe achieves the best or competitive per-
formance across most model settings on Yes/No questions with a s ingle deﬁnitive
answer.
6.2 Explanation and Visualization
Our proposed approach can also provide an explanation for the gen erated answer by
showing the paths in the KG that lead to that answer (see Figure 1). As shown in
Figure1, we have a complex question as an example from the ConditionalQA dat aset.
We have a complex and conditional question from an applicant whose h usband died
at work. The applicant has mentioned some details and wants to know about the
eligibility for the Bereavement Support Payment.
Now, to extract the relevant context from the KG, our KGCaRe ap proach will
go through the KG for a speciﬁed number of depths until it extract s the satisfactory
context to answer the question or satisfy the termination conditio n, which is the
numberofdepths. Then it uses the knowledgeorpaths traversed from allthe depths to
generate the ﬁnal answer. In Depth 1, it looks into the KG to get ke yword matches of
the entities from the question. There are diﬀerent ways to do this m atching, but here
it uses the direct keyword match. We can see here matches for the two keywords from
the question: “money” and “died”. We can also see the associated t riples extracted
from the KG. Here, initially, it extracts all the information, and then it performs
pruning. In pruning, it keeps only those triples that are most releva nt for answering
the question. In this case, it found the three most relevant triples as shown in Figure 1.
Then it performs reasoning using an LLM. It asks the LLM if the infor mation
found so far is enough to answer the question. In the case of Dept h 1, the LLM ﬁnds
13

Fig. 1Explaining the Knowledge Graph traversal steps of the propo sed algorithm using an example.
that the informationis not suﬃcient. Therefore,the approachas ks the LLM to suggest
other entities to search for in the KG and which might help to answer t he question.
Here we can see the LLM suggested entities as shown in Figure 1. It also saves these
3 triples to a variable named “Path Traversed” so that it has a recor d of them, and it
can use them in answer generation.
Now it goes into the next depth, that is Depth 2, and looks for expan ding with
keywords from what we saw in the previous depth, like “claimant”, “a pplicant”, etc.
It ﬁnds some new information related to these, which it will then prun e to keep only
those relevant to the question (three more triples). Now the appr oach again asks the
LLM if, at this depth, the information is suﬃcient to answer the ques tion. The LLM
still decides this is not enough to answer the question and returns s ome more clue
entities which might be useful. Now, before it goes to the next depth , it again saves
the triples found so far in Depth 2 to “Path Traversed”.It found o ne new triple related
to “applicant” that it adds to the existing path. It also found two mo re triples related
to the claimant, one states that they must have lived in the UK, and a nother states
that the claimant must be eligible for BSP.
Now it goes to the next depth that is Depth 3, and this time it didn’t ﬁnd any
new unique triple, and since this is the last depth considered by our ap proach, it
generates the answer to the question using the information of pat hs it has so far. The
ﬁnal answer generated is Yes, and the condition is that the applican t must be under
pension age and must be living in the UK to be eligible for Bereavement su pport. It
also provides an explanation showing which triples were used to reach this answer, as
shown in Figure 1.
Overall, our approach enhances trust by providing human-unders tandable expla-
nations for the generated answer.
14

7 Conclusion
In this work, we proposed KGCaRe, a hybrid retrieval-augmented q uestion answer-
ing approach that tightly integrates symbolic reasoning over a know ledge graph with
neural retrieval from a vector store. Our approach is speciﬁcally designed to address
the challenges of answering complex conditional domain-speciﬁc que stions—an area
where general-purpose LLMs often fall short due to missing conte xt or inability to
interpret conditions.
ByconstructingaKGusingamulti-promptLLM-basedpipelineandpair ingitwith
FAISS-based neural retrieval, KGCaRe leverages complementary strengths of both
structured and unstructured context. During inference, KGCa Re performs iterative
graph traversal guided by an LLM, prunes irrelevant paths, and d ynamically updates
its memory to support multi-hop reasoning. The ﬁnal answer is gene rated using a
prompt that combines the curated triples and semantically retrieve d text, allowing for
both factual accuracy and traceable explainability.
We evaluated KGCaRe on two complex conditional QA datasets, demo nstrating
consistent improvements across all answer types and LLMs compa red to strong base-
lines such as Vanilla LLM, Text Prompt, Code Prompt, Think-on-Grap h, Vanilla
RAG, and HybridContextQA. In particular, our KGCaRe approach s hows signiﬁcant
gainsforconditionalanswers,validatingourhypothesisthathybr idretrievalcanbetter
handle multi-faceted reasoning in complex domains.
Looking forward, our framework opens new directions for explaina ble QA systems
in high-stakes settings like healthcare, law, and public policy, where b oth precision
and transparency are critical.
Declarations
Funding. This publication hasemanated fromresearchconductedwith the ﬁn ancial
support of Research Ireland under Grant Number 12/RC/2289 P2 - Insight Research
Ireland Centre for Data Analytics and a grant from Fidelity Investm ents. For the
purpose of Open Access, the author has applied a CC BY public copyr ight licence to
any Author Accepted Manuscript version arising from this submissio n.
Data Availability. The datasets used in this study are publicly available.
Code availability. Code is available at the following GitHub repository:
https://github.com/GhanshyamVerma/KGCaRe
References
[1] Pan, S., Luo, L., Wang, Y., Chen, C., Wang, J., Wu, X.: Unifying large la nguage
models and knowledge graphs:A roadmap.IEEETransactionson Kn owledgeand
Data Engineering (2024)
[2] Kandpal, N., Deng, H., Roberts, A., Wallace, E., Raﬀel, C.: Large lang uage mod-
els struggle to learn long-tailknowledge. In: Proceedingsof the 40 th International
15

ConferenceonMachineLearning.ICML’23,vol.641,pp.15696–15 707.JMLR.org,
Honolulu, Hawaii, USA (2023)
[3] Zhang, Y., Li, Y., Cui, L., Cai, D., Liu, L., Fu, T., Huang, X., Zhao, E., Zh ang,
Y., Chen, Y., et al.: Siren’s song in the ai ocean: a survey on hallucinatio n in
large language models. arXiv preprint arXiv:2309.01219 (2023)
[4] Nickel, M., Murphy, K., Tresp, V., Gabrilovich,E.: A review ofrelation almachine
learning for knowledge graphs. Proceedings of the IEEE 104(1), 11–33 (2015)
[5] Wang, Q., Mao, Z., Wang, B., Guo, L.: Knowledge graph embedding: A sur-
vey of approaches and applications. IEEE Transactions on Knowled ge and Data
Engineering 29(12), 2724–2743 (2017)
[6] Cheng, K., Ahmed, N.K., Rossi, R.A., Willke, T., Sun, Y.: Neural-symbolic meth-
ods for knowledge graph reasoning: A survey. ACM Trans. Knowl. D iscov. Data
18(9) (2024) https://doi.org/10.1145/3686806
[7] Vukotic, A., Watt, N., Abedrabbo, T., Fox, D., Partner, J.: Neo4j in Action.
Manning Publications Co., Shelter Island, NY (2014)
[8] Monteiro, J., S´ a, F., Bernardino, J.: Experimental Evaluation of Graph
Databases:JanusGraph,NebulaGraph,Neo4j,andTigerGraph. Applied Sciences
13(9), 5770 (2023)
[9] Douze,M., Guzhva,A.,Deng, C.,Johnson,J.,Szilvasy,G., Mazar´ e,P.-E.,Lomeli,
M., Hosseini, L., J´ egou, H.: The FAISS Library. IEEE Transactions o n Big Data,
1–17 (2025) https://doi.org/10.1109/TBDATA.2025.3618474
[10] Sun, H., Cohen, W., Salakhutdinov, R.: ConditionalQA: A complex re ad-
ing comprehension dataset with conditional answers. In: Muresan , S.,
Nakov, P., Villavicencio, A. (eds.) Proceedings of the 60th Annual
Meeting of the Association for Computational Linguistics (Volume 1:
Long Papers), pp. 3627–3637. Association for Computational Lin guis-
tics, Dublin, Ireland (2022). https://doi.org/10.18653/v1/2022.acl-long.253 .
https://aclanthology.org/2022.acl-long.253
[11] Yang, Z., Qi, P., Zhang, S., Bengio, Y., Cohen, W.W., Salakhutdinov, R., Man-
ning, C.D.: HotpotQA: A dataset for diverse, explainable multi-hop qu estion
answering. In: Conference on Empirical Methods in Natural Langu age Processing
(EMNLP) (2018)
[12] Puerto, H., Tutek, M., Aditya, S., Zhu, X., Gurevych, I.: Code pr ompt-
ing elicits conditional reasoning abilities in Text+Code LLMs. In:
Al-Onaizan, Y., Bansal, M., Chen, Y.-N. (eds.) Proceedings of the
2024 Conference on Empirical Methods in Natural Language Proce ss-
ing, pp. 11234–11258. Association for Computational Linguistics, Miami,
16

Florida, USA (2024). https://doi.org/10.18653/v1/2024.emnlp-main.629 .
https://aclanthology.org/2024.emnlp-main.629/
[13] Wen, Y., Wang, Z., Sun, J.: MindMap: Knowledge graph prompt-
ing sparks graph of thoughts in large language models. In: Ku, L.-
W., Martins, A., Srikumar, V. (eds.) Proceedings of the 62nd Annual
Meeting of the Association for Computational Linguistics (Volume 1:
Long Papers), pp. 10370–10388. Association for Computational Linguis-
tics, Bangkok, Thailand (2024). https://doi.org/10.18653/v1/2024.acl-long.558 .
https://aclanthology.org/2024.acl-long.558/
[14] Sun, J., Xu, C., Tang, L., Wang, S., Lin, C., Gong, Y., Ni, L.M., Shum, H .-
Y., Guo, J.: Think-on-graph: Deep and responsible reasoning of larg e language
model on knowledge graph. In: Proceedings of the 12th Internat ional Conference
on Learning Representations (ICLR 2024) (2024)
[15] Ma, S., Xu, C., Jiang, X., Li, M., Qu, H., Yang, C., Mao, J., Guo, J.: Thin k-
on-graph 2.0: Deep and faithful large language model reasoning wit h knowledge-
guided retrieval augmented generation. In: Proceedings of the 1 3th International
Conference on Learning Representations (ICLR 2025) (2025)
[16] Luo, L., Li, Y.-F., Haﬀari, G., Pan, S.: Reasoning on graphs: Faithf ul and inter-
pretablelargelanguagemodelreasoning.In:Proceedingsofthe1 2thInternational
Conference on Learning Representations (ICLR 2024) (2024)
[17] Gao, Y., Xiong, Y., Gao, X., Jia, K., Pan, J., Bi, Y., Dai, Y., Sun, J., Wan g,
H.: Retrieval-augmented generation for large language models: A su rvey. arXiv
preprint arXiv:2312.10997 (2023)
[18] Sawarkar, K., Mangal, A., Solanki, S.R.: Blended RAG: Improving RA G
(Retriever-Augmented Generation) Accuracy with Semantic Sear ch and Hybrid
Query-Based Retrievers. In: 2024 IEEE 7th International Conf erence on Mul-
timedia Information Processing and Retrieval (MIPR), pp. 155–16 1 (2024).
https://doi.org/10.1109/MIPR62202.2024.00031
[19] Verma, G., Sarkar,S., Pillai, D., Shiokawa,H., Shahbazi, H., Veazey , F., Hubbert,
P., Su, H., Buitelaar, P.: HybridContextQA: A Hybrid Approach for Co mplex
QuestionAnsweringUsingKnowledgeGraphConstructionandConte xtRetrieval
With LLMs. In: Proceedings of the 2nd ISWC Workshop on Knowledge Base
Construction From Pre-Trained Language Models (ISWC KBC-LM), Baltimore,
USA, pp. 1–14 (2024)
[20] Markowitz, E., Ramakrishna, A., Dhamala, J., Mehrabi, N., Peris, C ., Gupta, R.,
Chang, K.-W., Galstyan, A.: Tree-of-traversals: A zero-shot rea soning algorithm
for augmenting black-box language models with knowledge graphs. P roceedings
of the 62nd Annual Meeting of the Association for Computational L inguistics,
12302–12319 (2024)
17

[21] Jiang, A.Q., Sablayrolles, A., Mensch, A., Bamford, C., Chaplot, D.S ., Casas, D.,
Bressand, F., Lengyel, G., Lample, G., Saulnier, L., Lavaud, L.R., Lach aux, M.-
A., Stock, P., Scao, T.L., Lavril, T., Wang, T., Lacroix, T., Sayed, W.E.: M istral
7B. arXiv preprint arXiv:2310.06825 (2023)
[22] Jiang, A.Q., Sablayrolles, A., Roux, A., Mensch, A., Savary, B., Bam ford, C.,
Chaplot, D.S., Casas, D., Hanna, E.B., Bressand, F., Lengyel, G., Bour , G., Lam-
ple, G., Lavaud, L.R., Saulnier, L., Lachaux, M.-A., Stock, P., Subrama nian, S.,
Yang, S., Antoniak, S., Scao, T.L., Gervet, T., Lavril, T., Wang, T., Lac roix, T.,
Sayed, W.E.: Mixtral of Experts. arXiv preprint arXiv:2401.04088 (2 024)
Appendix A Data Appendix
This section provides further details about the dataset used for e xperiments.
A.1 ConditionalQA
In the ConditionalQA dataset [ 10], each question-answer pair consists of three com-
ponents. The ﬁrst is the document itself, which is a structured polic y text organized
into hierarchical sections and often contains cross-references to other sections. These
documents were scraped from oﬃcial UK government websites and processed into
structured text by serializing the Document Object Model (DOM) t rees into a ﬂat-
tened sequence of HTML elements [ 10]. The second component is the question, which
is typically centered around eligibility, procedural compliance, or exc eptions and may
require reasoning across multiple parts of the document. These qu estions vary in type
and include yes/no responses, extractive span-based answers, and cases that may not
be answerable given the context. The third component is the user/ applicant scenario,
whichaddscontextualbackgroundrepresentingaspeciﬁcuser’s situationorconstraint.
This narrative helps simulate real-world query complexity and often in cludes implicit
conditions that must be resolved during the process of generating answers.
For the ConditionalQA dataset [ 10], we use the development set as the evaluation
benchmark for all comparative experiments, as the test set is not publicly avail-
able. ConditionalQA consists of 2,338 training QA pairs and 285 develop ment QA
pairs. Out of 285 questions from the ConditionalQA development set , we removed the
fourteen unanswerable questions. After removing the unanswer able questions, in the
ConditionalQA development set, we were left with 271 questions, as s hown in Table
A1.
The ConditionalQA dataset has three types of questions: Yes/No t ype, Span
(extractive) type, and conditional type. The ConditionalQA develo pment set has 143
Yes/No QA pairs, 102 Span QA pairs, and 63 Conditional QA pairs.
A.2 HotpotQA
The HotpotQA dataset [ 11] is a large-scale complex QA dataset that requires reason-
ing to be performed over multiple documents to generate the answe r to the multi-hop
questions. This dataset is created using crowdsourcing based on W ikipedia articles,
18

showing multiple context documents to crowd workers such that th ey can create
questions that require multi-hop reasoning.
HotpotQA mainly has two types of questions: Yes/No and Span (ext ractive) type.
The span type questions include the questions that require bridge e ntity identiﬁcation
and comparison between two entities to answer the multi-hop quest ions [11].
The HotpotQA train set has 90,564 QA pairs, having a combination of e asy,
medium, and hard questions. We evaluate the existing and proposed approaches on
a subset of the HotpotQA development set. The original HotpotQA development set
had 7,405 QA pairs. All the questions in the development set are hard /complex multi-
hop questions. We selected a set of 500 QA pairs (see Table A1) from the original
development set using random stratiﬁed sampling. The HotpotQA de velopment set
has 25 Yes/No QA pairs and 475 Span QA pairs.
Table A1 Number of QA pairs in the training
and development sets of the ConditionalQA
and HotpotQA datasets.
Dataset Train Set Dev Set
ConditionalQA 2,338 271
HotpotQA 90,564 500
Datasets and the code to preprocess them are available at the GitH ub repository:
https://github.com/GhanshyamVerma/KGCaRe
Appendix B Knowledge Graph Appendix
We constructed two KGs using our proposed KGCaRe approach: on e from Condition-
alQA documents and another from HotpotQA documents. The KG co nstructed using
ConditionalQA has 7,031 triples and 7,366 entities, while the KG constru cted using
HotpotQA has 17,776 triples and 13,286 entities, as shown in Table B2.
Table B2 Number of entities and triples in the generated KGs using KGCa Re.
KG Creation Approach ConditionalQA Dataset Hotpot QA Dataset
Entities Triples Entities Triples
KGCaRe 7366 7031 13286 17776
Knowledge Graphs and the developed software pipeline to construc t them are
available at the GitHub repository:
https://github.com/GhanshyamVerma/KGCaRe
19

Appendix C Code Appendix
Code for the existing approaches and the proposed KGCaRe appro ach is available at
the following GitHub repository:
https://github.com/GhanshyamVerma/KGCaRe
Appendix D Technical Appendix
This section provides technical details.
D.1 Evaluation
Avg F1 score: The Avg F1 score represents the calculated average F1 score usin g
the exact match policy between predicted and ground truth answe r, considering all
the questions of the development set [ 10,11].
F1 score: The F1 score represents the calculated F1 score using the exact m atch
policy between the predicted and ground truth answer [ 10,11].
The evaluation scripts were provided with the publicly available benchm ark
datasets, ConditionalQA [ 10] and HotpotQA [ 11]. We used the same scripts for our
evaluation. Our code repository includes the evaluation script as we ll. Link to the
code:https://github.com/GhanshyamVerma/KGCaRe
D.2 Implementation Details
•Code Prompt :
Note:Same parameters were used as mentioned in [ 12].
•Text Prompt :
Note:Same parameters were used as mentioned in [ 12].
•Think on Graph :
Temperature: 0.1
Maximum Output Length: 256
Width and Depth of Exploration: 3
Note:Temperature is set to 0.1 for all models. The rest of the parameter s are
the same as mentioned in [ 14].
•HybridContextQA :
Note:Same parameters were used as mentioned in [ 19].
•KGCaRe :
Experimental Setup: A combination of commercial APIs and local model infer-
ence was used.
LLM Models and Prompting:
–GPT-3.5 / GPT-4o (via OpenAI API using LlamaIndex): Used as-is
without modifying default decoding parameters. The default tempe ra-
ture of 0.7 from LlamaIndex was retained.
–Mistral-7B and Mixtral-8x7B (via vLLM + OpenAILike interface):
Default decoding parameters (e.g., temperature, top p) as deﬁned by
20

vLLM were used without manual modiﬁcation.
In-Context Examples:
–ConditionalQA: 4 examples randomly selected from the training set.
–HotpotQA: 4 examples randomly selected from the training set.
Retrievers:
–Vector-based Retriever (VectorIndexRetriever from LlamaIndex):
similarity topk = 10
–Knowledge Graph Traversal Retriever: maxdepth = 3,
maxentities = 20
These parameters were ﬁxed based on preliminary trials and prior wo rk; no
hyperparameter search was conducted.
Reranking:
–Cohere Reranker: rerank-english-v3.0 with top n = 2
– No other conﬁguration parameters were modiﬁed.
Selection Criteria: Most parameters were adopted from the default settings of
the respective tools (LlamaIndex, vLLM, Cohere). Deviations, su ch as reduc-
ing the number of in-context examples, were based on practical co nstraints
(e.g., token limits). No grid search or extensive hyperparameter tu ning was
conducted.
D.3 Experimental Environment
All experiments were conducted on a high-performance Linux mach ine with the
following speciﬁcations:
•Operating System: Ubuntu 22.04.4 LTS (Jammy)
•CPU:AMD EPYC 7313P 16-Core Processor
– 32 threads (16 cores, 2 threads per core)
– Base frequency: 1.5 GHz
– Max frequency: 3.0 GHz
•RAM:256 GB
•GPU:4×NVIDIA A40 (48 GB each)
– Experiments utilized approximately 43 GB per GPU for inference
•CUDA:
– CUDA Version: 12.8
– CUDA Toolkit: 12.6 (Build V12.6.85)
– Driver Version: 570.133.07
•Python Version: 3.10.16
21