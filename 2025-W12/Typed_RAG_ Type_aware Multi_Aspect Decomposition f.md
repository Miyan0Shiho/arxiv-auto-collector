# Typed-RAG: Type-aware Multi-Aspect Decomposition for Non-Factoid Question Answering

**Authors**: DongGeon Lee, Ahjeong Park, Hyeri Lee, Hyeonseo Nam, Yunho Maeng

**Published**: 2025-03-20 06:04:12

**PDF URL**: [http://arxiv.org/pdf/2503.15879v1](http://arxiv.org/pdf/2503.15879v1)

## Abstract
Non-factoid question-answering (NFQA) poses a significant challenge due to
its open-ended nature, diverse intents, and the need for multi-aspect
reasoning, which renders conventional factoid QA approaches, including
retrieval-augmented generation (RAG), inadequate. Unlike factoid questions,
non-factoid questions (NFQs) lack definitive answers and require synthesizing
information from multiple sources across various reasoning dimensions. To
address these limitations, we introduce Typed-RAG, a type-aware multi-aspect
decomposition framework within the RAG paradigm for NFQA. Typed-RAG classifies
NFQs into distinct types -- such as debate, experience, and comparison -- and
applies aspect-based decomposition to refine retrieval and generation
strategies. By decomposing multi-aspect NFQs into single-aspect sub-queries and
aggregating the results, Typed-RAG generates more informative and contextually
relevant responses. To evaluate Typed-RAG, we introduce Wiki-NFQA, a benchmark
dataset covering diverse NFQ types. Experimental results demonstrate that
Typed-RAG outperforms baselines, thereby highlighting the importance of
type-aware decomposition for effective retrieval and generation in NFQA. Our
code and dataset are available at
\href{https://github.com/TeamNLP/Typed-RAG}{https://github.com/TeamNLP/Typed-RAG}.

## Full Text


<!-- PDF content starts -->

Typed-RAG: Type-aware Multi-Aspect Decomposition
for Non-Factoid Question Answering
DongGeon Lee1*Ahjeong Park2∗Hyeri Lee3Hyeonseo Nam4Yunho Maeng5, 6†
1Pohang University of Science and Technology2Sookmyung Women’s University
3Independent Researcher4KT5Ewha Womans University
6LLM Experimental Lab, MODULABS
donggeonlee@postech.ac.kr ahjeong@sookmyung.ac.kr
{keira.hyeri.lee, namhs2030}@gmail.com yunhomaeng@ewha.ac.kr
Abstract
Non-factoid question-answering (NFQA)
poses a significant challenge due to its
open-ended nature, diverse intents, and
the need for multi-aspect reasoning, which
renders conventional factoid QA approaches,
including retrieval-augmented generation
(RAG), inadequate. Unlike factoid ques-
tions, non-factoid questions (NFQs) lack
definitive answers and require synthesizing
information from multiple sources across
various reasoning dimensions. To address
these limitations, we introduce Typed-RAG,
a type-aware multi-aspect decomposition
framework within the RAG paradigm for
NFQA. Typed-RAG classifies NFQs into
distinct types—such as debate, experience,
and comparison—and applies aspect-based
decomposition to refine retrieval and genera-
tion strategies. By decomposing multi-aspect
NFQs into single-aspect sub-queries and
aggregating the results, Typed-RAG generates
more informative and contextually relevant
responses. To evaluate Typed-RAG, we
introduce Wiki-NFQA, a benchmark dataset
covering diverse NFQ types. Experimental
results demonstrate that Typed-RAG outper-
forms baselines, thereby highlighting the
importance of type-aware decomposition
for effective retrieval and generation in
NFQA. Our code and dataset are available at
https://github.com/TeamNLP/Typed-RAG.
1 Introduction
In real-world scenarios, individuals often seek an-
swers to non-factoid questions (NFQs) that require
comprehensive responses reflecting multiple per-
spectives and contextual nuances, rather than sim-
ple factual replies. These questions include types
such as comparisons, experiences, and debates, re-
*Both authors contributed equally to this work.
†Corresponding author.flecting complex human information needs (Bolo-
tova et al., 2022).
NFQs pose a significant challenge in retrieval-
augmented generation (RAG) systems due to their
diverse intents and answer perspectives. Unlike fac-
toid questions (FQs), which follow a 1:1 question-
answer relationship and require short, factual an-
swers (e.g., “What’s the capital of France?” ) (Lee
et al., 2022; Sun et al., 2024), NFQs lack a defini-
tive answer (e.g., “Is AI beneficial to society?” ),
making single-aspect responses insufficient to fully
satisfy user needs. We define these as multi-aspect
queries, requiring multi-domain reasoning and syn-
thesis from multiple sources, which conventional
factoid question-answering (FQA) systems strug-
gle to handle.
Recent non-factoid question-answering (NFQA)
approaches, including query-focused summariza-
tion and type-specific methods (Deng et al., 2024;
An et al., 2024), struggle to generalize across di-
verse NFQs and underutilize large language mod-
els (LLMs) and RAG frameworks. While standard
RAG enhances contextuality (Lewis et al., 2020;
Izacard and Grave, 2021), it fails to address NFQ
heterogeneity, which arises from differences in
question intent and the need for multi-perspective
reasoning. As a result, it produces overly uniform
responses that lack the multi-aspect depth neces-
sary for comprehensive NFQA.
To address these challenges, we propose Typed-
RAG, a type-aware multi-aspect decomposition
framework for NFQA. By integrating question type
classification with a pre-trained classifier into the
RAG pipeline, our approach refines retrieval and
generation strategies for distinct NFQ types. A
key feature, multi-aspect decomposition , breaks
multi-aspect NFQs into single-aspect sub-queries,
enabling targeted retrieval and response generation
before synthesizing a well-rounded answer. By
handling each aspect separately and synthesizing
responses, our system generates answers aligned
1arXiv:2503.15879v1  [cs.CL]  20 Mar 2025

Non-Factoid
Question
EVIDENCE-BASED
COMPARISON
EXPERIENCE
REASON
INSTRUCTION
DEBATEsingle -aspect
query extractionkeyword
extractionrerankingaggregating 
answersSingle-Aspect Query 
Generator
Keyword
ExtractorRetriever GeneratorReranker
Answer
AggregatorResultPrompt
As a 'Query Analyst',
please evaluate this
question and generate
a multi-aspect query.
...
...
question: {query}
yes
no no
yesyes
no
yesno
Prompt
As a 'Query Analyst', please
evaluate this question and
extract the key entities in
the question .
...
...
question: {query}Prompt
Based on the input provided,
synthesizing information from multi-
ple aspects or responses to provide a
concise and balanced summary.
...
...
Question-Answer Pairs: {qa_pairs}Multi-Aspect Decomposer
 Type
ClassifierFigure 1: An overview of Typed-RAG. Non-factoid questions are classified by the type classifier and processed
based on their type. Prompts for the multi-aspect decomposer and answer aggregator handle the unique requirements
of each type. Details of the prompt can be found in Appendix A.4.
with user intent.
We evaluate Typed-RAG on Wiki-NFQA, a
benchmark dataset of diverse NFQ types from
Wikipedia. Results show it outperforms baseline
models, including standard LLMs and RAG, effec-
tively capturing NFQ complexity and generating
nuanced, user-aligned answers.
Our contributions are summarized as follows:
•We introduce Typed-RAG , a novel type-aware
multi-aspect decomposition method in retrieval-
augmented generation framework for NFQA that
integrates question type classification and multi-
aspect decomposition to enhance answer genera-
tion.
•We develop optimized retrieval and generation
strategies tailored to different NFQ types, im-
proving the system’s ability to address diverse
and complex user queries.
•We construct the Wiki-NFQA dataset to facili-
tate NFQA research and provide a benchmark for
evaluating QA systems on non-factoid questions.
•We demonstrate that Typed-RAG outperforms ex-
isting baseline models, validating our approach’s
effectiveness in generating high-quality answers
to NFQs.
By addressing the unique challenges of NFQA,
we aim to bridge the gap between user information
needs and QA system capabilities, paving the way
for more effective and contextually aware question-
answering technologies.
2 Related Work
Non-Factoid Question Taxonomy To tackle the
complexity of real-world QA, questions are exten-sively categorized (Burger et al., 2003; Chaturvedi
et al., 2014; Bolotova et al., 2022). Factoid ques-
tions seek straightforward answers, while NFQs
require more complex, subjective, or multifaceted
responses (Chaturvedi et al., 2014; Bolotova et al.,
2022). Bolotova et al. (2022) identifies six NFQ
types: Instruction, Reason, Evidence-based, Com-
parison, Experience, and Debate. Our study ad-
vances NFQA by employing RAG strategies to
manage these question types effectively.
Retrieval-Augmented Generation (RAG) RAG
integrates external knowledge into LLMs’ response
generation, enhancing the accuracy of the output.
The effectiveness of RAG depends on providing
question-relevant search results to the LLM as a
generator. Therefore, improving the retrieval pro-
cess is critical to prevent the LLM’s hallucination
(Huang et al., 2023; Lee and Yu, 2025). Advanced
RAG methods actively employ query rewriting and
decomposition techniques (Press et al., 2023; Rack-
auckas, 2024; Chan et al., 2024) to aggregate di-
verse information, capturing both explicitly stated
facts and subtle, indirectly conveyed knowledge.
However, the application of LLMs and RAG
for solving NFQA remains largely underexplored.
Deng et al. (2024) tackles NFQA with graph net-
works and a Transformer-based language model,
but does not employ a large-scale pretrained LLM.
Similarly, although An et al. (2024) proposed a
paradigm integrating a logic-based data structure
with RAG to address How-To NFQ, it did not cover
all NFQ types comprehensively.
Evaluation Metrics Traditional metrics like
ROUGE and BERTScore (Zhang et al., 2020),
2

used to evaluate answers from FQA systems using
LLMs and RAG, often fail to capture the semantic
richness and nuanced quality variations in non-
factoid answers. To overcome this, Yang et al.
(2024) proposed LINKAGE, a listwise ranking
framework for NFQA evaluation. LINKAGE lever-
ages LLM as a scorer to rank candidate answers
against reference answers ordered by quality and
demonstrates stronger correlations with human an-
notations, outperforming traditional metrics and
emphasizing its potential as a superior evaluation
methodology.
3 Method
This section introduces the RAG pipeline specifi-
cally designed for NFQ types. The overall method-
ology is illustrated in Figure 1. The types of
NFQ differ in their intent, the directionality of
their aspects, and the degree of contrast between
them. NFQs often manifest as multi-aspect queries,
where answers span multiple perspectives. These
aspects can be categorized into two types: con-
trasting aspects, which exhibit high contrast and
opposing directions, and related aspects, which
have lower contrast and align in the same direction.
Based on these characteristics, we designed meth-
ods that effectively align with user expectations.12
In the following sections, we discuss these ques-
tion types and their processing rationale.
Evidence-based Evidence-based type questions
seek to understand the characteristics or definitions
of specific concepts, objects, or events, typically
requiring fact-based explanations. As single-aspect
queries, they do not undergo multi-aspect decom-
position. Instead, we apply a straightforward RAG
approach, retrieving relevant passages and generat-
ing accurate answers while preserving clarity and
simplicity. The retriever first retrieves passages
using the question as a query, and the generator
produces responses based on these passages.
Comparison Comparison-type questions are
used to examine differences, similarities, or su-
periority between keywords, with answers tai-
lored to the comparison’s purpose and targets.
1Using the prompts described in Appendix A.4, our ap-
proach involves either applying a multi-aspect decomposer,
tailored to the question type, or aggregating answers generated
for multiple single-aspect queries.
2Details for each type are illustrated in Figure 16 (a)–(e),
and detailed examples for each NFQ type are discussed in the
Appendix B.3. Reasoning behind each methodological design
is provided in Appendix C.1.As multi-aspect queries, they require decompo-
sition. First, a keyword extractor identifies the
comparison purpose ( compare_type ) and targets
(keywords_list ). Subsequently, the retriever is
employed to search for passages that are related to
each keyword. Following the deduplication of pas-
sages, the remaining results are reranked according
to their relevance. Finally, the generator combines
the information to create a response that aligns
with the comparison purpose, thereby effectively
addressing the question.
Experience Experience-type questions seek ad-
vice or recommendations, providing explanations
based on personal experiences. As multi-aspect
queries, they require decomposition. The re-
triever first retrieves relevant passages, followed
by similarity-based reranking using extracted key-
words by the keyword extractor. In the end, the
generator produces an optimized response aligned
with the intent of the question, effectively meeting
its requirements.
Reason/Instruction Reason-type questions aim
to explore the causes of specific phenomena, while
instruction-type questions focus on understanding
procedures or methods. The intent behind these
question types is to receive clear and comprehen-
sive answers that present reasons or steps in a well-
structured manner. These questions are treated
as multi-aspect queries and require a multi-aspect
decomposer. First, the query is decomposed into
single-aspect queries using a single-aspect query
generator. Each generated query is then processed
individually by the retriever and generator to pro-
duce separate answers. Finally, an answer aggre-
gator combines these individual responses into a
concise and accurate final answer.
Debate Debate-type questions are hypothetical
questions designed to explore diverse perspectives,
including opposing viewpoints. The purpose of
this question type is to generate a balanced re-
sponse that reflects multiple perspectives. Being
a multi-aspect question, it naturally requires the
multi-aspect decomposer. The question is first pro-
cessed by a single-aspect query generator, which
extracts the discussion topic and various opinions.
Generated queries for each opinion are then han-
dled by the retriever and generator to produce indi-
vidual responses. Finally, a debate mediator com-
bines the topic and perspectives to produce a well-
balanced final response.
3

Model Scorer LLM MethodsWiki-NFQA Dataset
NQ-NF SQD-NF TQA-NF 2WMH-NF HQA-NF MSQ-NF
Llama-3.2-3BMistral-7BLLM 0.5893 0.5119 0.6191 0.3565 0.4825 0.4262
RAG 0.5294 0.4944 0.5470 0.4150 0.4530 0.4047
Typed-RAG 0.7659 0.6493 0.7061 0.4544 0.5624 0.5356
GPT-4o miniLLM 0.4934 0.4506 0.5380 0.3070 0.3669 0.2917
RAG 0.4187 0.3553 0.4586 0.2859 0.2957 0.2866
Typed-RAG 0.8366 0.7139 0.7013 0.3692 0.5470 0.4482
Mistral-7BMistral-7BLLM 0.6356 0.5450 0.6363 0.4821 0.5255 0.5081
RAG 0.5635 0.5069 0.6233 0.4789 0.5323 0.4438
Typed-RAG 0.7103 0.6333 0.6709 0.4747 0.6035 0.4512
GPT-4o miniLLM 0.4656 0.4222 0.5921 0.3175 0.3965 0.3384
RAG 0.4411 0.3817 0.5450 0.2890 0.3562 0.3079
Typed-RAG 0.8413 0.7444 0.7767 0.3987 0.6653 0.4929
Table 1: Evaluation results on the Wiki-NFQA dataset using Mean Reciprocal Rank (MRR), comparing the
performance of various language models, scorer LLMs, and methods. Answers were ranked using LINKAGE
(Yang et al., 2024) scored with the MRR metric.
4 Experimental Setup
4.1 Model
We compare our Typed-RAG to LLM-based and
RAG-based QA systems as the baselines. In the
experiments, we use a black-box LLM and two
open-weights LLMs with varying number of pa-
rameters: (i) GPT-4o-mini-2024-07-18 (GPT-4o
mini), (ii) Mistral-7B-Instruct-v0.2 (Mistral-7B;
Jiang et al., 2023), and (iii) Llama-3.2-3B-Instruct
(Llama-3.2-3B). All the inputs in LLMs (including
the generator of RAG), are formatted using prompt
templates. The prompt templates we used in our
experiments are in Appendix A.3.
4.2 Listwise Ranking Evaluation (LINKAGE)
To evaluate an NFQA system, we adopt LINK-
AGE (Yang et al., 2024), a listwise ranking ap-
proach, as our primary evaluation metric. LINK-
AGE ranks each candidate answer according to its
relative quality in comparison to a reference list of
answers. The ranking process is formally defined
as follows:
rank ci= LLM( PL, qi, ci, Ri) (1)
Here, qidenotes the i-th question, and cirefers to
the candidate answer being evaluated. The list of
reference answers Ri, associated with qi, consists
of individual reference answers riordered in de-
scending quality. The evaluation process uses a
scorer ( LLM ), guided by the LINKAGE prompt
PL, to determine rank ci, the rank of cirelative to
the reference list Ri.Ranking Metrics To quantify the ranking results
from LINKAGE, we utilize two complementary
metrics: Mean Reciprocal Rank (MRR) (V oorhees
and Tice, 2000) and Mean Percentile Rank (MPR).
The MRR metric measures the rank position of
candidate answers, with higher values indicating
that answers are ranked closer to the top. It is cal-
culated by averaging the reciprocal ranks of the
candidate answers across all questions. On the
other hand, MPR normalizes ranks into percentiles,
reflecting the relative positions of answers within
their respective reference lists. A higher MPR sug-
gests better overall ranking performance across all
references. While MRR emphasizes top-ranked
answers, MPR provides insight into relative perfor-
mance throughout the entire list.
4.3 Dataset Construction
To evaluate NFQA methods, we construct the
Wiki-NFQA dataset , a specialized resource tai-
lored for NFQA, which is derived from exist-
ing Wikipedia-based datasets (Kwiatkowski et al.,
2019; Joshi et al., 2017; Rajpurkar et al., 2016;
Yang et al., 2018; Trivedi et al., 2022; Ho et al.,
2020). We extract non-factoid questions through
a systematic filtering process, then generate high-
quality reference answers to ensure the dataset’s
suitability for NFQA evaluation.3
Reference Answers Generation Since these
datasets have only a single-grade ground truth an-
swer, we generate diverse qualities of reference
3The filtering process is discussed in Appendix B.1.
4

LLM RAG Typed-RAG0255075EVIDENCE-BASED
LLM RAG Typed-RAG0255075COMPARISON
LLM RAG Typed-RAG0204060EXPERIENCE
LLM RAG Typed-RAG0204060REASON
LLM RAG Typed-RAG0204060INSTRUCTION
LLM RAG Typed-RAG0204060DEBATE
Model: Llama-3.2-3B, Scorer: Mistral-7B
Model: Llama-3.2-3B, Scorer: GPT-4o miniModel: Mistral-7B, Scorer: Mistral-7B
Model: Mistral-7B, Scorer: GPT-4o miniFigure 2: Mean Percentile Rank (MPR) performance comparison of LLM, RAG, and Typed-RAG on six different
non-factoid question categories from the Wiki-NFQA dataset. Results are reported using different model configura-
tions (Llama-3.2-3B and Mistral-7B) and scorer LLMs (Mistral-7B and GPT-4o mini). The y-axis represents the
MPR score (%), with higher values indicating better performance.
answers for LINKAGE evaluation, following Yang
et al. (2024). After the construction of reference
answers, we annotate the quality level of generated
reference answers using the GPT-4o-2024-11-20.4
5 Experimental Results
Overall, Typed-RAG consistently outperforms
both LLM-based and RAG-based methods across
all subdatasets of the Wiki-NFQA dataset, NFQ
categories, and model configurations. An inte-
grated analysis of the MRR results (Table 1) and
MPR results (Figure 2) reveals that Typed-RAG
not only improves the ranking positions of gen-
erated responses but also enhances their relative
quality. These results demonstrate that responses
generated by Typed-RAG are consistently rated
higher in relevance and comprehensiveness by the
scorer LLMs.5
Impact of Scorer LLMs and Base Models The
performance of all methods is affected by the
choice of scorer LLMs and base models. It is
important to note that scores obtained from dif-
ferent scorer LLMs should not be directly com-
pared, as each scorer evaluates responses using its
own learned criteria and internal representation of
language. Instead, the relative ranking of meth-
ods within the same scorer LLM provides a more
4Prompt details about the reference answers generation
are in Appendix A.1, and A.2.
5Examples of Typed-RAG’s responses for different types
of non-factoid questions are provided in Appendix E.meaningful interpretation of performance.
A notable trend observed in Table 1 and Figure 2
is that scores tend to decrease when switching from
Mistral-7B to GPT-4o mini as the scorer. This can
be attributed to GPT-4o mini being a more power-
ful and sophisticated model, potentially making it a
more critical evaluator. Stronger models generally
have a more refined understanding of relevance,
coherence, and factual accuracy, leading them to
assign lower scores when responses have minor
inconsistencies or lack sufficient depth. This phe-
nomenon is consistent across different base mod-
els and methods, reinforcing the idea that a more
advanced scorer imposes stricter evaluation crite-
ria. However, despite the overall score reductions,
Typed-RAG maintains a clear advantage over both
the LLM-based and RAG-based baselines, indicat-
ing its robustness to changes in scorer strictness.
Limitations of RAG and Benefits of Typed-RAG
RAG-based methods consistently underperform
compared to direct LLM-based generation, as
shown in Table 1 and Figure 2. A key reason is
that retrieved factual information often introduces
noise rather than aiding response generation in
NFQA tasks. Typed-RAG addresses these issues
by leveraging a multi-aspect decomposition strat-
egy, optimizing retrieval for NFQA. By structuring
retrieval around distinct facets of non-factoid ques-
tions, Typed-RAG reduces irrelevant noise and en-
sures more relevant information retrieval. As seen
5

in Table 1 and Figure 2, Typed-RAG consistently
outperforms both RAG and LLM-only approaches,
particularly in reasoning-intensive datasets, demon-
strating its effectiveness in enhancing response
quality.
6 Conclusion
In this paper, we introduced Typed-RAG , a
novel approach to non-factoid question answering
(NFQA) that integrates type-aware multi-aspect de-
composition within a RAG framework. By classify-
ing questions into specific types and tailoring RAG
accordingly, Typed-RAG addresses the diverse and
complex nature of NFQs. A key feature of our
method is the decomposition of multi-aspect ques-
tions into single-aspect sub-queries, allowing for
targeted retrieval and generation that collectively
produce comprehensive and nuanced answers.
The experimental results on the Wiki-NFQA
dataset substantiate the effectiveness of Typed-
RAG in enhancing answer quality for non-factoid
questions. The consistent outperformance across
various types of NFQ, baselines, and scorer LLMs
indicates that integrating type-aware multi-aspect
decomposition is a robust approach for NFQA. Our
method not only elevates the answers’ ranks but
also improves their relative quality in the eyes of
the scorer LLMs, leading to better user satisfac-
tion.
Limitations
Although our work is the first to introduce RAG to
NFQA, it has several limitations.
A key limitation is the lack of direct comparison
between Typed-RAG and existing query rewriting
and decomposition methodologies. While Typed-
RAG provides a structured approach to these tasks,
its performance relative to other techniques re-
mains unexplored. Various query rewriting and
decomposition techniques have been proposed to
enhance retrieval quality, yet this study does not
empirically evaluate how Typed-RAG performs
relative to these approaches in terms of query re-
formulation effectiveness, retrieval relevance, and
computational overhead. A systematic comparison
with these methods would provide a clearer under-
standing of Typed-RAG’s advantages and limita-
tions. Future work should incorporate benchmark
evaluations against these established techniques to
better position Typed-RAG within the landscape
of query rewriting and decomposition research.Another limitation of our evaluation setup is
that we use the same model to assess the quality of
its own generated responses. This self-evaluation
approach may introduce bias, as the model could
struggle to distinguish quality differences among
answers it produced. To mitigate this issue, fu-
ture work could explore evaluation using stronger
LLMs, human assessments, or ensemble scoring
methods. By adopting these strategies, we can
improve the reliability of quality assessments and
reduce potential biases in our evaluation frame-
work.
Acknowledgments
We thank Chris Develder, our mentor from the
NAACL 2025 SRW pre-submission mentorship
program, as well as the anonymous reviewers for
their valuable feedback on this paper.
This research was supported by Brian Impact
Foundation, a non-profit organization dedicated to
the advancement of science and technology for all.
We would also like to acknowledge the support
of the Korea Institute of Human Resources Devel-
opment in Science and Technology (KIRD).
6

References
Kaikai An, Fangkai Yang, Liqun Li, Junting Lu,
Sitao Cheng, Lu Wang, Pu Zhao, Lele Cao, Qing-
wei Lin, Saravan Rajmohan, Dongmei Zhang, and
Qi Zhang. 2024. Thread: A logic-based data orga-
nization paradigm for how-to question answering
with retrieval augmented generation. arXiv preprint
arXiv:2406.13372 .
Valeriia Bolotova, Vladislav Blinov, Falk Scholer,
W. Bruce Croft, and Mark Sanderson. 2022. A non-
factoid question-answering taxonomy. In Proceed-
ings of the 45th International ACM SIGIR Confer-
ence on Research and Development in Information
Retrieval , page 1196–1207.
Valeriia Bolotova-Baranova, Vladislav Blinov, So-
fya Filippova, Falk Scholer, and Mark Sanderson.
2023. WikiHowQA: A comprehensive benchmark
for multi-document non-factoid question answering.
InProceedings of the 61st Annual Meeting of the
Association for Computational Linguistics (Volume
1: Long Papers) , pages 5291–5314. Association for
Computational Linguistics.
John Burger, Claire Cardie, Vinay Chaudhri, Robert
Gaizauskas, Sanda Harabagiu, David Israel, Chris-
tian Jacquemin, Chin-Yew Lin, Steve Maiorano,
George Miller, Dan Moldovan, Bill Ogden, John
Prager, Ellen Riloff, Amit Singhal, Rohini Shrihari,
Tomek Strazalkowski, Ellen V oorhees, and Ralph
Weishedel. 2003. Issues, tasks and program struc-
tures to roadmap research in question & answering
(Q&A). In Document Understanding Conferences
Roadmapping Documents , pages 1–35.
Chi-Min Chan, Chunpu Xu, Ruibin Yuan, Hongyin Luo,
Wei Xue, Yike Guo, and Jie Fu. 2024. RQ-RAG:
Learning to refine queries for retrieval augmented
generation. In 1st Conference on Language Model-
ing.
Snigdha Chaturvedi, Vittorio Castelli, Radu Florian,
Ramesh M. Nallapati, and Hema Raghavan. 2014.
Joint question clustering and relevance prediction
for open domain non-factoid question answering. In
Proceedings of the 23rd International Conference on
World Wide Web , page 503–514.
Yang Deng, Wenxuan Zhang, Weiwen Xu, Ying
Shen, and Wai Lam. 2024. Nonfactoid question
answering as query-focused summarization with
graph-enhanced multihop inference. IEEE Trans-
actions on Neural Networks and Learning Systems ,
35(8):11231–11245.
Xanh Ho, Anh-Khoa Duong Nguyen, Saku Sugawara,
and Akiko Aizawa. 2020. Constructing a multi-
hop QA dataset for comprehensive evaluation of
reasoning steps. In Proceedings of the 28th Inter-
national Conference on Computational Linguistics ,
pages 6609–6625.
Lei Huang, Weijiang Yu, Weitao Ma, Weihong Zhong,
Zhangyin Feng, Haotian Wang, Qianglong Chen,Weihua Peng, Xiaocheng Feng, Bing Qin, and Ting
Liu. 2023. A survey on hallucination in large lan-
guage models: Principles, taxonomy, challenges, and
open questions. arXiv preprint arXiv:2311.05232 .
Gautier Izacard and Edouard Grave. 2021. Leveraging
passage retrieval with generative models for open do-
main question answering. In Proceedings of the 16th
Conference of the European Chapter of the Associ-
ation for Computational Linguistics: Main Volume ,
pages 874–880.
Albert Q. Jiang, Alexandre Sablayrolles, Arthur Men-
sch, Chris Bamford, Devendra Singh Chaplot, Diego
de Las Casas, Florian Bressand, Gianna Lengyel,
Guillaume Lample, Lucile Saulnier, Lélio Re-
nard Lavaud, Marie-Anne Lachaux, Pierre Stock,
Teven Le Scao, Thibaut Lavril, Thomas Wang, Tim-
othée Lacroix, and William El Sayed. 2023. Mistral
7B.arXiv preprint arXiv:2310.06825 .
Mandar Joshi, Eunsol Choi, Daniel Weld, and Luke
Zettlemoyer. 2017. TriviaQA: A large scale distantly
supervised challenge dataset for reading comprehen-
sion. In Proceedings of the 55th Annual Meeting of
the Association for Computational Linguistics (Vol-
ume 1: Long Papers) , pages 1601–1611.
Vladimir Karpukhin, Barlas Oguz, Sewon Min, Patrick
Lewis, Ledell Wu, Sergey Edunov, Danqi Chen, and
Wen-tau Yih. 2020. Dense passage retrieval for open-
domain question answering. In Proceedings of the
2020 Conference on Empirical Methods in Natural
Language Processing , pages 6769–6781.
Tom Kwiatkowski, Jennimaria Palomaki, Olivia Red-
field, Michael Collins, Ankur Parikh, Chris Al-
berti, Danielle Epstein, Illia Polosukhin, Jacob De-
vlin, Kenton Lee, Kristina Toutanova, Llion Jones,
Matthew Kelcey, Ming-Wei Chang, Andrew M. Dai,
Jakob Uszkoreit, Quoc Le, and Slav Petrov. 2019.
Natural Questions: A benchmark for question an-
swering research. Transactions of the Association
for Computational Linguistics , 7:452–466.
DongGeon Lee and Hwanjo Yu. 2025. REFIND:
Retrieval-augmented factuality hallucination detec-
tion in large language models. arXiv preprint
arXiv:2502.13622 .
Nayeon Lee, Wei Ping, Peng Xu, Mostofa Patwary, Pas-
cale Fung, Mohammad Shoeybi, and Bryan Catan-
zaro. 2022. Factuality enhanced language models for
open-ended text generation. In Advances in Neural
Information Processing Systems 35 , pages 34586–
34599.
Patrick S. H. Lewis, Ethan Perez, Aleksandra Pik-
tus, Fabio Petroni, Vladimir Karpukhin, Naman
Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih,
Tim Rocktäschel, Sebastian Riedel, and Douwe
Kiela. 2020. Retrieval-augmented generation for
knowledge-intensive NLP tasks. In Advances in
Neural Information Processing Systems 33 , pages
9459–9474.
7

Tian Liang, Zhiwei He, Wenxiang Jiao, Xing Wang,
Yan Wang, Rui Wang, Yujiu Yang, Shuming Shi, and
Zhaopeng Tu. 2024. Encouraging divergent thinking
in large language models through multi-agent debate.
InProceedings of the 2024 Conference on Empiri-
cal Methods in Natural Language Processing , pages
17889–17904.
OpenAI. 2024. GPT-4o system card. arXiv preprint
arXiv:2410.21276 .
Ofir Press, Muru Zhang, Sewon Min, Ludwig Schmidt,
Noah Smith, and Mike Lewis. 2023. Measuring and
narrowing the compositionality gap in language mod-
els. In Findings of the Association for Computational
Linguistics: EMNLP 2023 , pages 5687–5711.
Zackary Rackauckas. 2024. RAG-Fusion: a new take
on retrieval-augmented generation. International
Journal on Natural Language Computing , 13:37–47.
Pranav Rajpurkar, Jian Zhang, Konstantin Lopyrev, and
Percy Liang. 2016. SQuAD: 100,000+ questions
for machine comprehension of text. In Proceedings
of the 2016 Conference on Empirical Methods in
Natural Language Processing , pages 2383–2392.
Hao Sun, Hengyi Cai, Bo Wang, Yingyan Hou, Xi-
aochi Wei, Shuaiqiang Wang, Yan Zhang, and Dawei
Yin. 2024. Towards verifiable text generation with
evolving memory and self-reflection. In Proceedings
of the 2024 Conference on Empirical Methods in
Natural Language Processing , pages 8211–8227.
Harsh Trivedi, Niranjan Balasubramanian, Tushar Khot,
and Ashish Sabharwal. 2022. MuSiQue: Multi-
hop questions via single-hop question composition.
Transactions of the Association for Computational
Linguistics , 10:539–554.
Ellen M. V oorhees and Dawn M. Tice. 2000. The
TREC-8 question answering track. In Proceedings
of the Second International Conference on Language
Resources and Evaluation .
Sihui Yang, Keping Bi, Wanqing Cui, Jiafeng Guo, and
Xueqi Cheng. 2024. LINKAGE: Listwise ranking
among varied-quality references for non-factoid QA
evaluation via LLMs. In Findings of the Association
for Computational Linguistics: EMNLP 2024 , pages
6985–7000.
Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio,
William Cohen, Ruslan Salakhutdinov, and Christo-
pher D. Manning. 2018. HotpotQA: A dataset for
diverse, explainable multi-hop question answering.
InProceedings of the 2018 Conference on Empiri-
cal Methods in Natural Language Processing , pages
2369–2380.
Tianyi Zhang, Varsha Kishore, Felix Wu, Kilian Q.
Weinberger, and Yoav Artzi. 2020. BERTScore:
Evaluating text generation with BERT. In 8th Inter-
national Conference on Learning Representations .
8

A Prompt Details
A.1 Reference List Construction
Prompt Template to Generate the Highest Standard Reference Answer
Given a non-factoid question:"{ question }" and its answer:"{ ground_truth }"
Use your internal knowledge to rewrite this answer.
Figure 3: Prompt template proposed by Yang et al. (2024) to generate the highest standard reference answer using
LLM’s internal knowledge.
Prompt Template to Generate Diverse Qualities of Reference Answers
Generate three different answers to a non-factoid question from good to bad in quality, each inferior to the golden
answer I give you. Ensure that the quality gap from good to bad is very significant among these three answers. Golden
answer is the reasonable and convincing answer to the question. Answer 1 can be an answer to the question, however, it
is not sufficiently convincing. Answer 2 does not answer the question or if it does, it provides an unreasonable answer.
Answer 3 is completely out of context or does not make any sense.
Here are 3 examples for your reference.
1.Non-factoid Question: how can we get concentration on something?
Golden Answer: To improve concentration, set clear goals, create a distraction-free environment, use time management
techniques like the Pomodoro Technique, practice mindfulness, take regular breaks, stay organized, limit multitasking,
practice deep work, maintain physical health, and seek help if needed.
Output:
Answer 1: Improve focus: set goals, quiet space, Pomodoro Technique, mindfulness, breaks, organization, limit
multitasking, deep work, health, seek help if needed.
Answer 2: Just like and enjoy the work you do, concentration will come automatically.
Answer 3: If you are student, you should concentrate on studies and don’t ask childish questions.
2.Non-factoid Question: Why doesn’t the water fall off earth if it’s round?
Golden Answer: Earth’s gravity pulls everything toward its center, including water. Even though Earth is round, gravity
keeps water and everything else anchored to its surface. Gravity’s force is strong enough to counteract the Earth’s
curvature, preventing water from falling off.
Output:
Answer 1: This goes along with the question of why don’t we fall off the earth if it is round. The answer is because
gravity is holding us (and the water) down.
Answer 2: Same reason the people don’t.
Answer 3: When rain drops fall through the atmosphere CO2 becomes dissolved in the water. CO2 is a normal
component of the Earth’s atmosphere, thus the rain is considered naturally acidic.
3.Non-factoid Question: How do I determine the charge of the iron in FeCl3?
Golden Answer: Since chloride ions (Cl-) each carry a charge of -1, and there are three chloride ions in FeCl3, the total
negative charge from chloride ions is -3. To balance this, the iron ion (Fe) must have a charge of +3 to ensure the
compound has a neutral overall charge. Therefore, the charge of the iron ion in FeCl3 is +3.
Output:
Answer 1: Charge of Fe in Fecl3 is 3. Iron has either 2 as valancy or 3. in this case it bonds with three chlorine
molecules. therefore its valency and charge is three.
Answer 2: If two particles (or ions, or whatever) have opposite charge, then one has positive charge and one has
negative charge.
Answer 3: take a piece of iron. Wrap a copper wire around the iron in tight close coils. run a charge through the wire.
Below are the non-factoid question, and the golden answer.
Non-factoid Question: { question }
Golden Answer: { ground_truth }
Output:
Figure 4: Prompt template proposed by Yang et al. (2024) to generate diverse qualities of reference answers.
9

A.2 Reference Answers Annotation
System Prompt for Reference Answers Annotation
Your task is to evaluate the relevance and quality of multiple candidate answers for a given
non-factoid question.
Please evaluate the quality of each answer in a step-by-step manner.
Follow the structured guidelines below to ensure consistency and accuracy in your evaluation.
# Notes on Candidate Answers
Multiple candidate answers can come in two forms:
- Single choice answer: A single string, e.g., `"born again"`.
- Multiple choice answer: A list of strings, e.g., `[‘traffic calming’, ‘aesthetics’]`.
When evaluating multiple choice answers, treat the entire list as a single unit. Do **not** split
them into individual components; instead, evaluate the overall quality as a whole.
# Evaluation Criteria
Assign a label to each candidate answer based on the following criteria:
- 3: The answer provides a comprehensive, accurate, and contextually relevant response that
directly addresses the question.
- 2: The answer is accurate and relevant but lacks depth or comprehensive coverage.
- 1: The answer is somewhat relevant but contains inaccuracies, vagueness, or insufficient detail.
- 0: The answer is irrelevant, incorrect, or fails to address the question meaningfully.
**If there are two or more answers that you think are close in quality, you can give the same label.**
# Response Format
- Assign a label to each answer strictly in the format: `Answer X: [[Y]]`, where `X` is the answer
number, and `Y` is the integer score (0-3).
- Do **not** include any additional comments or explanations outside this format.
Input Prompt Template for Reference Answers Annotation
# Inputs
- Non-Factoid Question: { question }
- Candidate Answers:
{reference_answers }
Figure 5: System prompt (top) and input prompt template (bottom) adapted from Yang et al. (2024) for annotating
the quality level of generated reference answers.
10

A.3 Prompt templates for Baseline Methods
Prompt template for LLM
You are an assistant for answering questions.
Answer the following question.
### Question
{question }
### Answer
Figure 6: Prompt template for LLM method.
Prompt template for RAG
You are an assistant for answering questions.
Refer to the references below and answer the following question.
### References
{reference_passages }
### Question
{question }
### Answer
Figure 7: Prompt template for RAG method.
11

A.4 Prompt templates for Typed-RAG
A.4.1 Debate
Prompt Template for Generating Sub-queries in Debate-type Questions
You are a query analysis assistant. Based on the query type, apply the relevant prompt to transform the query to better
align with the user’s intent, ensuring clarity and precision.
The input question is a debate-type question (i.e., invites multiple perspectives). As a "Query Analyst", please evaluate
this question and proceed with the following steps.
1. Extract the debate topic.
2. Identify 2 to 5 key perspectives on this topic.
3. Generate a sub-query reflecting each perspective’s bias.
Ensure each sub-query fits a Retrieval-Augmented Generation (RAG) framework, seeking passages that align with the
viewpoint.
### Output format
{"debate_topic": {topic}, "dist_opinion": [list of perspectives], "sub-queries": {"opinion1": "biased sub-query for
opinion1", "opinion2": "biased sub-query for opinion2", ...}
### Example
Query: "Is Trump a good president?"
Answer:
{
"debate_topic": "Donald Trump’s presidency",
"dist_opinion": ["positive", "negative", "neutral"],
"sub-queries": {
"positive": "Was Donald Trump one of the best presidents for economic growth?",
"negative": "Did Trump’s presidency harm the U.S. economy and leadership?",
"neutral": "Can we assess Trump’s tenure’s strengths and weaknesses?"
}
}
### Input
Query: { query }
### Output
Answer:
Prompt Template for Debate Mediator in Debate-type Questions
You are acting as the mediator in a debate.
Below is a topic and responses provided by n participants, each with their own perspective. Your task is to synthesize
these responses by considering both the debate topic and each participant’s viewpoint, providing a fair and balanced
summary. Ensure the response maintains balance, captures key points, and distinguishes any opposing opinions.
Present the answer *short and concise*, phrased in a direct format without using phrases like "participants in the
debate" or "in the debate."
### Input format
- Debate topic: {debate_topic}
- Participant’s responses:
- Response 1: "{response content}" (Perspective: {perspective 1})
- Response 2: "{response content}" (Perspective: {perspective 2})
- ...
- Response N: "{response content}" (Perspective: {perspective N})
### Output format
A short and concise summary from the mediator’s perspective based on the discussion, phrased as a direct answer
without reference to the debate structure or participants
### Inputs
Debate topic: { debate_topic }
Participant’s responses:
{responses }
### Output
Summary:
Figure 8: Prompt templates for generating sub-queries and debate mediator in debate-type questions.
12

A.4.2 Experience
Figure 9 shows the prompt template for responding to experience-type questions. The retrieved passages
are subsequently reranked based on the extracted keywords. After reranking, we use the prompt template
for RAG (Figure 7) to generate answers.
Prompt Template for Keyword Extraction in Experience-type Questions
You are a query analysis assistant. Based on the query type, apply the relevant prompt to transform
the query to better align with the user’s intent, ensuring clarity and precision.
The input question is an experience-type question (i.e., get advice or recommendations on a
particular topic.). As a "Query Analyst", please evaluate this question and proceed with the
following steps.
1. Identify the topic intended to be gathered from experience-based questions.
2. Extract the key entities in the question, considering the intent of asking about experience, to
facilitate an accurate response.
### Output format
`["Keyword 1", ..., "Keyword N"]` (List of string, separated with comma)
### Example
Question (Input): "What are some of the best Portuguese wines?"
Answer (Output): ["Portuguese wines", "best"]
### Input
Question: { question }
### Output
Answer:
Figure 9: Prompt template for keyword extraction in experience-type questions.
13

A.4.3 Reason & Instruction
Prompt Template for Generating Sub-queries in Reason-type Questions
You are a query analysis assistant. Based on the query type, apply the relevant prompt to transform the query to better
align with the user’s intent, ensuring clarity and precision.
The input query is a reason-type question (i.e., a question posed to understand the reason behind a particular concept or
phenomenon). As a "Query Analyst", please evaluate this query and proceed with the following steps.
1. Break down the original instruction into multiple sub-queries that preserve the core intent but use varied
language and structure. These multiple sub-queries should aim to capture different linguistic expressions of the original
instruction while still aligning with its intended meaning.
2. Create at least 2 to 5 distinct multiple sub-queries.
### Output format
`["sub-query 1", ..., "sub-query N"]` (List of string, separated with comma)
### Input
Query: { query }
### Output
Multiple sub-queries:
Figure 10: Prompt template for generating sub-queries in reason-type questions.
Prompt Template for Generating Sub-queries in Instruction-Type Questions
You are a query analysis assistant. Based on the query type, apply the relevant prompt to transform the query to better
align with the user’s intent, ensuring clarity and precision.
The input query is an instruction-type question (i.e., Instructions/guidelines provided in a step-by-step manner).
As a "Query Analyst", please evaluate this query and proceed with the following steps.
1. Break down the original instruction into multiple sub-queries that preserve the core intent but use varied language
and structure.
These multiple sub-queries should aim to capture different linguistic expressions of the original instruction while still
aligning with its intended meaning.
2. Create at least 2 to 5 distinct multiple sub-queries.
### Output format
`["sub-query 1", ..., "sub-query N"]` (List of string, separated with comma)
### Input
Query: { query }
### Output
Multiple sub-queries:
Figure 11: Prompt template for generating sub-queries in instruction-type questions.
Prompt Template for Aggregating Answers to an Original Question
You are an assistant tasked with aggregating answers to a question.
You are provided with the original question and multiple question-answer pairs. These queries preserve the core intent
of the original question but use varied language and structure. Your goal is to review the question-answer pairs and
synthesize a concise and accurate response to the original question based on the information provided.
Using the information from the question-answer pairs, generate a brief and clear answer to the original question.
### Inputs
Original Question: { original_question }
Question-Answer Pairs:
{qa_pairs_text }
### Output
Aggregated Answer:
Figure 12: Prompt template for aggregating answers to an original question. Used by reason-type questions and
instruction-type questions.
14

A.4.4 Comparison
Prompt Template for Keyword Extraction in Comparison-type Questions
You are a query analysis assistant. Based on the query type, apply the relevant prompt to transform
the query to better align with the user’s intent, ensuring clarity and precision.
Determine if the input query is a compare-type question (i.e., compare/contrast two or more things,
understand their differences/similarities.) as a "Query Analyst". If so, perform the following:
1. Identify the type of comparison: "differences", "similarities", or "superiority".
2. Extract the subjects of comparison and represent them as specific, contextualized phrases.
### Output format
{"is_compare": true/false, "compare_type": "", "keywords_list": []}
### Example
Query: "Who is more intelligent than humans on earth?"
Analysis:
{"is_compare": true, "compare_type": "superiority", "keywords_list": ["human intelligence", "the
intelligence of other beings"]}
### Input
Query: { query }
### Output
Analysis:
Figure 13: Prompt template for keyword extraction in comparison-type questions.
Prompt Template for Generating a Response to Comparison-type Questions
You are an assistant for answering questions.
You are given the extracted parts of a long document and a question. Refer to the references below
and answer the following question.
The question is a compare-type with a specific comparison type and keywords indicat-
ing the items to compare.
Answer based on this comparison type and the target keywords provided.
### Inputs
Question: {question}
Comparison Type: { comparison_type }
Keywords: { keywords }
References:
{reference_passages }
### Output
Answer:
Figure 14: Prompt template for generating a response to comparison-type questions.
15

A.5 Listwise Ranking Evaluation (LINKAGE)
Prompt Template for LINKAGE
Please impartially rank the given candidate answer to a non-factoid question accurately within the
reference answer list, which are ranked in descending order of quality. The top answers are of the
highest quality, while those at the bottom may be poor or unrelated.
Determine the ranking of the given candidate answer within the provided reference answer list. For
instance, if it outperforms all references, output [[1]]. If it’s deemed inferior to all four references,
output [[4]].
Your response must strictly following this format: "[[2]]" if candidate answer could rank 2nd.
Below are the user’s question, reference answer list, and the candidate answer.
Question:{ question }
Reference answer list:{ reference_answers }
Candidate answer:{ candidate_answer }
Figure 15: Prompt template proposed by Yang et al. (2024) to evaluate Typed-RAG and baseline methods using
LINKAGE (Yang et al., 2024).
16

B Construction of the Wiki-NFQA
Dataset
B.1 Filtering Non-Factoid Questions
To extract NFQs from existing Wikipedia-based
datasets, we use the nf-cats6(Bolotova et al., 2022),
a RoBERTa-based pre-trained question category
classifier. Since this model categorizes questions
into factoid and non-factoid types, we retain only
those classified as non-factoid for further process-
ing. To ensure a more rigorously curated dataset,
we heuristically filter the data following the ques-
tion patterns outlined in the NFQ taxonomy pro-
posed by Bolotova et al. (2022), thereby construct-
ing the Wiki-NFQA dataset. The statistics for the
Wiki-NFQA dataset are presented in Table 3 (Ap-
pendix B.4).
B.2 Reference Answers Construction
In detail, we use three different LLMs to obtain
different styles of reference answers: (i) GPT-3.5-
turbo-16k, (ii) Mistral-7B-Instruct-v0.27(Jiang
et al., 2023), (iii) Llama-3.1-8B-Instruct8. Each
LLM generates three quality answers, for a total
of nine reference answers. In addition, we use
GPT-4o-2024-08-06 (OpenAI, 2024) to generate
the highest standard reference answer, which is
distinct from the other nine answers.
B.3 Representative Examples of NFQ Types
Table 2 presents representative example questions
for each type of NFQ in the Wiki-NFQA dataset.
The NFQ taxonomy follows the classification pro-
posed by Bolotova-Baranova et al. (2023).
The Evidence-based type requires answers
grounded in verifiable sources, while Comparison
questions seek distinctions between concepts. Ex-
perience questions solicit subjective opinions or
recommendations, whereas Reason questions aim
to uncover the rationale behind events or concepts.
Instruction questions request procedural guidance,
and Debate questions involve discussions on con-
troversial or interpretative topics.
B.4 Dataset Statistics
The statistics of the Wiki-NFQA dataset can be
found in Table 3. The total number of questions in
the dataset is 945.
6https://huggingface.co/Lurunchik/nf-cats
7https://huggingface.co/mistralai/
Mistral-7B-Instruct-v0.2
8https://huggingface.co/meta-llama/Llama-3.
1-8B-InstructC Implementation Details
C.1 Typed-RAG
We designed methodologies to meet user expecta-
tions by analyzing the intent and aspects of each
question type in the NFQ system.
Evidence-based questions seek specific, reli-
able factual information and are inherently single-
aspect. Since the intent of these questions is to
obtain evidence-based facts, the response remains
singular in perspective. The absence of complex
contextual reasoning or multi-aspect decomposi-
tion allows them to be handled through direct and
straightforward answering methods.
Comparison questions aim to provide informa-
tion about differences, similarities, or relative ad-
vantages between two or more items. These ques-
tions are multi-aspect, as the intent varies depend-
ing on whether the goal is to highlight similarities,
differences, or the superiority of one option over
another. The presence of multiple comparison cri-
teria results in answers that must be tailored ac-
cordingly. Depending on the type of comparison,
the aspects can be categorized as either contrast-
ing, when focusing on differences, or related, when
emphasizing similarities. Answering these ques-
tions effectively requires identifying the compar-
ison type and retrieving relevant information for
each aspect separately before synthesizing them
into a balanced response.
Experience-based questions focus on obtaining
advice, recommendations, or insights based on
personal experiences, making them multi-aspect.
Since individual experiences are inherently subjec-
tive, responses vary widely depending on the per-
spective of the respondent. This variability results
in answers that contrast with one another rather
than aligning along a shared perspective. To gen-
erate informative responses, it is essential to accu-
rately identify the user’s intent and define the key
aspects that should be addressed. Unlike compari-
son questions, which focus on feature-based differ-
ences, experience-based questions have a broader
scope, reflecting the diversity of personal opinions
and experiences.
Reason and instruction questions focus on under-
standing causes behind a phenomenon or procedu-
ral steps for achieving a goal. Although these two
subtypes share a broad category, their aspectual
properties differ significantly.
Reason-type questions aim to identify the causes
of an event or phenomenon and are multi-aspect be-
17

NFQ Type Example of Question
Evidence-based “How does sterilisation help to keep the money flow even?”
Comparison “what is the difference between dysphagia and odynophagia”
Experience “What are some of the best Portuguese wines?”
Reason“Kresy, which roughly was a part of the land beyond the so-called Curson Line,
was drawn for what reason?”
Instruction “How can you find a lodge to ask to be a member of?”
Debate“I Can See Your V oice, a reality show from South Korea, offers what kind of
performers a chance to make their dreams of stardom a reality?”
Table 2: Representative example questions for each type of non-factoid question (NFQ) in the Wiki-NFQA dataset.
cause the explanation depends on the surrounding
context. Different conditions may lead to different
causal explanations, making a single-perspective
answer insufficient. Since multiple explanations
can exist based on varying assumptions, responses
often contrast with one another.
Instruction-type questions, on the other hand,
focus on procedural steps or methodologies and
are also multi-aspect due to the variability in ap-
proaches. The methods for accomplishing a task
may differ based on the specific objective, level of
detail required, or alternative techniques available.
In contrast to reason-type questions, instruction-
type responses tend to align in a similar direction
rather than opposing one another, as procedural
explanations often contain variations but remain
conceptually related. Effectively answering these
questions requires retrieving diverse procedural de-
scriptions and synthesizing them into a coherent
instructional format.
Debate-type questions seek to gather balanced
perspectives on a contentious issue, making them
inherently multi-aspect. Unlike factual inquiries,
debate questions involve subjective stances that
depend on underlying assumptions, resulting in
contrasting viewpoints. Since different perspec-
tives arise from distinct premises, responses must
be structured to fairly represent multiple argu-
ments rather than favoring a single standpoint. To
ensure a balanced synthesis of perspectives, re-
sponses should incorporate reasoning from oppos-
ing viewpoints and structure the final answer in a
mediator-like format. In our implementation, we
reference the debate mediator prompt from Liang
et al. (2024) to ensure that responses objectively
aggregate and present diverse perspectives.C.2 Parameter Settings of LLMs
In accordance with the LINKAGE settings, we con-
sistently use a nucleus sampling parameter ( top_p
= 0.95) and a maximum output tokens of 512 for
all LLM cases. The temperature is generally set to
0.8; however, it is lowered to 0.1 specifically for
the purpose of annotating reference answers.
To generate answers for NFQA using the RAG-
based QA system, the retriever identifies five pas-
sages, which are then provided to the generator as
references. For Wikipedia-based datasets, BM25
serves as the retriever, leveraging the Wikipedia
corpus preprocessed by Karpukhin et al. (2020) as
the external corpus.
D Detailed Analysis per Dataset
We further analyze the performance on individual
datasets to understand the strengths of Typed-RAG
in different contexts.
NQ-NF, SQuAD-NF, and TriviaQA-NF These
datasets consist of non-factoid questions that are
often open-ended or require explanatory answers.
Typed-RAG achieves significant improvements in
MRR and MPR scores, indicating its proficiency in
generating detailed and relevant responses. For ex-
ample, on the SQuAD-NF dataset with the Mistral-
7B base model and GPT-4o mini scorer, Typed-
RAG achieves an MRR of 0.7444, outperforming
the LLM (0.4222) and RAG (0.3817) methods.
HotpotQA-NF and MuSiQue-NF These
datasets involve multi-hop reasoning, where the
answer requires combining information from
multiple sources. Typed-RAG shows marked
improvements, particularly in MPR scores. The
18

NFQ Type NQ-NF SQD-NF TQA-NF 2WMHQA-NF HQA-NF MSQ-NF Total
Evidence-based 99 130 251 10 22 43 555 (58.73%)
Comparison 5 18 4 0 8 1 36 (3.81%)
Experience 0 20 8 1 10 2 41 (4.34%)
Reason 19 85 23 55 15 21 218 (23.07%)
Instruction 2 21 3 8 4 11 49 (5.19%)
Debate 1 26 7 5 3 4 46 (4.87%)
Total 126 300 296 79 62 82 945
Table 3: Statistics of the Wiki-NFQA dataset.
ability of Typed-RAG to decompose questions
into types and aspects allows it to navigate
complex reasoning paths more effectively than the
baselines.
2WikiMultiHopQA-NF Although all methods
show relatively lower performance on this dataset,
Typed-RAG still outperforms both LLM and RAG
approaches. This suggests that even in challeng-
ing scenarios requiring extensive reasoning, Typed-
RAG provides a tangible benefit.
19

 Query
Keyword Extractor Retriever  
[K1 , K2 , K3, ..., KN] 
Keywords
Generator
ResultReferences
Query Single-Aspect Query Generator
SQ2 SQ3
Retriever Retriever Retriever
Generator Generator GeneratorSQ1
Answer Aggregator
ResultSQN
Retriever
Generator
A1 A2 A3 ANQuery Keyword Extractor
K2 K3
Retriever Retriever RetrieverK1
ResultKN
Retriever
Reranker
GeneratorReferences
Duplicated
References Query
Retriever
Generator
ResultReferences
......
Combined References 
...
RerankerK1K2K3KN
Query Single-Aspect Query Generator
Topic
Op1Opinions
Debate Mediator
Result...
Op2 OpN
Retriever
GeneratorRetriever
GeneratorRetriever
GeneratorOp3
Retriever
Generator
A2 AN(a) EVIDENCE-BASED (b) COMPARISON (c) EXPERIENCE
(e) DEBATE (d) REASON/INSTRUCTIONcompare
type
A3 A1 
Figure 16: Detailed process for each Non-Factoid question type. In (b) and (c), Knrepresents the extracted keywords. In (d) and (e), the single-aspect
query generator creates multiple queries, and their answers are aggregated. Specifically, in (d), SQnrefers to single-aspect queries, while Andenotes
the corresponding answers. In (e), OPnrepresents queries generated for each distinct opinion.
20

E Case Study
NFQ Type Example of Question Response of Typed-RAG
Evidence-based How does sterilisation
help to keep the money
flow even?The term "sterilisation" is often used in economics
to refer to the act of preventing currency or securities
from affecting the monetary base due to interest rate
differentials. When a central bank engages in open
market operations to buy or sell securities, it may
want to prevent the money flow from being disrupted
by interest rate differentials in different markets.
Sterilisation helps to keep the money flow even
by offsetting the effect of open market operations
on the monetary base. When a central bank buys
securities, it injects cash into the economy and
increases the monetary base. However, if this leads
to an appreciation of the domestic currency, it may
reduce the competitiveness of exports and increase
the cost of imports. To prevent this, the central bank
may sell an equivalent amount of foreign currency
in the foreign exchange market, which reduces the
monetary base and helps to offset the initial effect of
the open market operation.
Similarly, if a central bank sells securities, it
decreases the monetary base and may lead to a
depreciation of the domestic currency, which can
make exports more competitive and imports more
expensive. To prevent this, the central bank may
buy an equivalent amount of foreign currency in
the foreign exchange market, which increases the
monetary base and helps to offset the initial effect of
the open market operation.
In summary, sterilisation is a tool used by central
banks to keep the money flow even and maintain
control over the monetary base, despite open market
operations that may otherwise disrupt the flow of
funds. It allows central banks to implement monetary
policy while minimizing the impact on currency
values and other macroeconomic variables.
Continued on next page
21

NFQ Type Example of Question Response of Typed-RAG
Comparison what is the difference
between dysphagia and
odynophagiaDysphagia and odynophagia are two distinct
conditions related to swallowing.
Dysphagia is the medical term for difficulty
swallowing. It can be caused by various conditions,
including neurological disorders, stroke, cancer,
or structural problems in the throat or esophagus.
Symptoms of dysphagia include the sensation of
food sticking in the throat or chest, coughing or
choking during eating, and pain or discomfort while
swallowing.
On the other hand, odynophagia is the medi-
cal term for pain while swallowing. It is a symptom
of an underlying condition, such as a sore throat,
inflammation, or infection. The pain can be de-
scribed as sharp or throbbing and may be worsened
by swallowing solid or acidic foods. Odynophagia
can also be a symptom of gastroesophageal reflux
disease (GERD) or eosinophilic esophagitis.
In summary, dysphagia is a condition that
makes it difficult to swallow, while odynophagia
is a symptom of pain while swallowing. A person
with dysphagia may also experience pain while
swallowing, but the primary issue is the difficulty
in moving food from the mouth to the stomach.
Conversely, a person with odynophagia primarily
experiences pain while swallowing, but they may
still be able to swallow relatively normally.
Continued on next page
22

NFQ Type Example of Question Response of Typed-RAG
Experience What are some of the best
Portuguese wines?Based on the available references in the document,
some Portuguese wines that are often considered
excellent include:
1. The reference 14098516 might be referring to Port
Wine, a fortified wine produced in the Douro Valley
region. Port Wine is known for its sweet and rich
flavors and comes in various styles such as Tawny,
Ruby, and White Port.
2. Reference 5043240 could be referring to Vinho
Verde, a light, slightly effervescent wine from the
Vinho Verde region in the north of Portugal. It is
typically low in alcohol and is often served chilled.
3. The Douro region is also famous for its red wines,
such as those referenced in 2866435. These wines
are known for their robust flavors and are often
blended from a variety of grape varieties.
4. The reference 4891311 might be referring to
Madeira Wine, another fortified wine from the
Madeira Islands. Madeira Wine is known for its
complex flavors and can range from sweet to dry.
5. Lastly, the reference 309483 could be referring
to Dã Wines, which come from the Dã region in
central Portugal. These wines are known for their
elegance and balance, with flavors of red fruit, herbs,
and minerals.
So, based on the given references, some of
the best Portuguese wines include Port Wine, Vinho
Verde, Douro Red Wines, Madeira Wine, and Dã
Wines.
Reason Kresy, which roughly was
a part of the land beyond
the so-called Curson Line,
was drawn for what
reason?The Curzon Line refers to the boundary proposed
by Lord Curzon, then Viceroy of India, in 1920, as
a potential eastern border for a future independent
Poland. Kresy, or Eastern Borderlands, was the area
of land beyond this proposed border. It was drawn
for various reasons, including historical, ethnic, and
strategic considerations. The area had been part of
the Polish state in the past, and there was a significant
Polish population there. Ethnically, the region was
diverse, with various groups including Poles, Ukraini-
ans, Belarusians, and Jews. Strategically, the region
was important due to its location near the borders
of several neighboring countries and its rich natural
resources.
Continued on next page
23

NFQ Type Example of Question Response of Typed-RAG
Instruction How can you find a lodge
to ask to be a member of?To find a lodge to inquire about membership,
consider the following steps:
1. Contact the Grand Lodge of your jurisdic-
tion (state or country) for a list of lodges in your
area. You can usually find this information on their
website or by contacting them directly.
2. Research the lodges on the list to learn more about
their history, traditions, and meeting times. This can
be done through their websites, social media pages,
or by contacting them directly.
3. Attend a few meetings as a visitor to get a feel for
the lodge and its members. You may need to request
an invitation from the lodge’s secretary or master.
4. Speak with current members about their experi-
ences and the expectations for new members. This
can help you determine if the lodge is a good fit for
you.
5. Consider your availability and the lodge’s meeting
schedule to ensure you can commit to attending
meetings regularly.
Remember, every lodge is unique, so it’s es-
sential to find one that aligns with your values and
interests. Good luck in your search!
(Note: The references provided do not con-
tain information relevant to finding a lodge to ask to
be a member of.)
Debate I Can See Your V oice, a
reality show from South
Korea, offers what kind of
performers a chance to
make their dreams of
stardom a reality?Based on the information provided in the references,
"I Can See Your V oice" is a South Korean reality
show that offers singers a chance to audition without
revealing their appearance or singing voice to the
judges initially. This format is designed to evaluate
contestants based on their charisma, rhythm, and
overall performance without any biases influenced by
their physical appearance or singing voice. Therefore,
the show provides an opportunity for talented singers
who might not have been given a chance due to their
looks or voice in their initial auditions to showcase
their abilities and pursue their dreams of stardom.
Table 4: Examples of Typed-RAG responses for different types of non-factoid questions.
24