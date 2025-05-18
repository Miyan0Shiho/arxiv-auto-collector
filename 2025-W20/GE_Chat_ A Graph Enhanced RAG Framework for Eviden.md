# GE-Chat: A Graph Enhanced RAG Framework for Evidential Response Generation of LLMs

**Authors**: Longchao Da, Parth Mitesh Shah, Kuan-Ru Liou, Jiaxing Zhang, Hua Wei

**Published**: 2025-05-15 10:17:35

**PDF URL**: [http://arxiv.org/pdf/2505.10143v1](http://arxiv.org/pdf/2505.10143v1)

## Abstract
Large Language Models are now key assistants in human decision-making
processes. However, a common note always seems to follow: "LLMs can make
mistakes. Be careful with important info." This points to the reality that not
all outputs from LLMs are dependable, and users must evaluate them manually.
The challenge deepens as hallucinated responses, often presented with seemingly
plausible explanations, create complications and raise trust issues among
users. To tackle such issue, this paper proposes GE-Chat, a knowledge Graph
enhanced retrieval-augmented generation framework to provide Evidence-based
response generation. Specifically, when the user uploads a material document, a
knowledge graph will be created, which helps construct a retrieval-augmented
agent, enhancing the agent's responses with additional knowledge beyond its
training corpus. Then we leverage Chain-of-Thought (CoT) logic generation,
n-hop sub-graph searching, and entailment-based sentence generation to realize
accurate evidence retrieval. We demonstrate that our method improves the
existing models' performance in terms of identifying the exact evidence in a
free-form context, providing a reliable way to examine the resources of LLM's
conclusion and help with the judgment of the trustworthiness.

## Full Text


<!-- PDF content starts -->

arXiv:2505.10143v1  [cs.CL]  15 May 2025GE-Chat: A Graph Enhanced RAG Framework for Evidential
Response Generation of LLMs
Longchao Da1, Parth Mitesh Shah1, Kuan-Ru Liou1, Jiaxing Zhang2, Hua Wei1∗
{longchao,pshah113,kliou,hua.wei}@asu.edu,tabzhangjx@gmail.com
Arizona State University1, New Jersey Institute of Technology2
Arizona1, New Jersey2, USA
Abstract
Large Language Models are now key assistants in human decision-
making processes. However, a common note always seems to follow:
"LLMs can make mistakes. Be careful with important info." This
points to the reality that not all outputs from LLMs are dependable,
and users must evaluate them manually. The challenge deepens as
hallucinated responses, often presented with seemingly plausible
explanations, create complications and raise trust issues among
users. To tackle such issue, this paper proposes GE-Chat, a knowl-
edge Graph enhanced retrieval-augmented generation framework
to provide Evidence-based response generation. Specifically, when
the user uploads a material document, a knowledge graph will be
created, which helps construct a retrieval-augmented agent, en-
hancing the agent’s responses with additional knowledge beyond
its training corpus. Then we leverage Chain-of-Thought (CoT) logic
generation, n-hop sub-graph searching, and entailment-based sen-
tence generation to realize accurate evidence retrieval. We demon-
strate that our method improves the existing models’ performance
in terms of identifying the exact evidence in a free-form context,
providing a reliable way to examine the resources of LLM’s con-
clusion and help with the judgment of the trustworthiness. The
datasets are released at1.
CCS Concepts
•Computing methodologies →Natural language processing ;
Knowledge representation and reasoning ;•Information systems
→Information retrieval .
Keywords
LLMs, Evidential Answering, Retrieval Augmented Generation.
ACM Reference Format:
Longchao Da1,Parth Mitesh Shah1,Kuan-Ru Liou1,Jiaxing Zhang2,Hua Wei1∗.
2018. GE-Chat: A Graph Enhanced RAG Framework for Evidential Response
Generation of LLMs. In Proceedings of Make sure to enter the correct confer-
ence title from your rights confirmation emai (WWW ’25). ACM, New York,
NY, USA, 5 pages. https://doi.org/XXXXXXX.XXXXXXX
1https://drive.google.com/drive/folders/1kNcsn1v0KH_srgL8w-NKvZM25o3onHBj?
usp=sharing
Permission to make digital or hard copies of all or part of this work for personal or
classroom use is granted without fee provided that copies are not made or distributed
for profit or commercial advantage and that copies bear this notice and the full citation
on the first page. Copyrights for components of this work owned by others than the
author(s) must be honored. Abstracting with credit is permitted. To copy otherwise, or
republish, to post on servers or to redistribute to lists, requires prior specific permission
and/or a fee. Request permissions from permissions@acm.org.
WWW ’25, April 28–May 2, 2025, Sydney, Australia
©2018 Copyright held by the owner/author(s). Publication rights licensed to ACM.
ACM ISBN 978-1-4503-XXXX-X/18/06
https://doi.org/XXXXXXX.XXXXXXX1 Introduction
Large Language Models (LLMs) have demonstrated remarkable
capabilities in multi-round conversational chats, including under-
standing questions [ 8], generating responses, and performing rea-
soning or inference using the given context [ 22], [23]. The rapid
advancement of LLMs has enabled a wide range of applications
across various domains, including customer support [ 5,15], virtual
assistance [ 21], and augmented agents with tool-usage capabili-
ties [2].
However, even though the LLMs are trained on massive expert
corpus covering a wide range of topics, they are not immune to
generating incorrect or misleading information, as called ‘halluci-
nation’ [ 10]. It is always suggested to users on LLM interface that,
‘LLMs can make mistakes. Check important information carefully’,
which underscores the dilemma of seeking to benefit from LLMs’
capabilities while contending with the challenge of the quality of
the responses.
The solutions can be roughly divided into two aspects: One
is to ground the LLMs’ output to the factuality, either by fine-
tunning [ 19] or information retrieval augmentation [ 18]. Fine-
tunning on the LLMs to reduce hallucination takes more resources
and can be computationally expensive, meanwhile, it is not applica-
ble to black-box commercial LLMs. And information-retrieval-based
methods take more steps for multi-resource factuality checking and
grounding, such as [ 4,9]. This will rely on the information source
to vote for the trustworthiness of the generated responses and
requires more query times from other APIs or Knowledge bases.
The other direction is to understand the confidence/uncertainty
of LLMs’ responses, such as the training uncertainty prediction
layers or applying post-hoc response level uncertainty quantifica-
tion [ 1,14]. These methods tend to quantify a measurement value
for the LLMs, it provides a convenient way to compare the perfor-
mance among LLMs, but not straightforward enough for users to
decide whether to believe the LLMs’ conclusion.
Regarding this shortcoming, some work started with evidence
matching the generated responses. Essentially, when a user uploads
a document and performs a question asking a certain LLM, it tends
to highlight the corresponding raw context from a document that
is best relevant to the response, thus guiding the user to under-
stand where the conclusion is drawn from the original document.
However, the current method, such as [ 7,13] uses the direct LLM
responded source of the evidence [ 3,17], when faced with a re-
dundant response, it can only perform on the chunk-level resource
highlight, which often gives a whole paragraph of relevant context
without fine concentration. We also empirically observe that the
performance of LLMs in reflecting their source evidence varies
significantly depending on their instruction-following capability.

WWW ’25, April 28–May 2, 2025, Sydney, Australia Longchao et al.
Document PoolUpload
ORUserOutputExtractQuestionChunk SourcesAns1: Chunk 1,2,6Ans2:Chunk 4,7Ans3: Chunk 3,9Ansn: Chunk …Retrieve
Answer EvidenceAns1: Sent. 1,2,6Ans2:Sent. 4,7…4. OptimizeS1: Entity AS2: Entity B…+Extract①Probe Relations②1. Graph-RAG Construction ABLLM
AnswersCoTsAns1…Ans2…Ans3…Ansn…+*1…2…3…n…2. CoTReasoning Elicitation 
ABDerived KGReturn (Answer + Evidence)123n…3. Sub-Graph SearchingfirstsecondEntity N(2)-hop GraphEvidenceSentenceschunks
Figure 1: The overview of the GE-Chat framework. As shown in this pipeline, when user uploads the document, it is used for
the 1. Graph-RAG Construction, which contains two main steps using LLMs, ➀Extract the entities A, B, etc., from the document
chunks, then ➁Probe the contextual relations between these entities. Then a derived Knowledge Graph is formed and used
for question answering. In order to realize evidence generation on Graph-RAG, 2. CoT Reasoning Elicitation is proposed to
elicit the reasoning chain for answers. Then we have 3. Sub-Graph Searching based on Entity Matching, and N-hop Relations
Probing, this sub-graph contains entities and relations used to retrieve the source chunks, guaranteeing originality of content.
We then apply an 4. Optimize objective to balance meaningfulness and conciseness to get the high-quality evidence.
However, smaller language models that excel at answering specific
domain questions but lack instruction fine-tuning often struggle to
effectively highlight relevant information for users to reference.
To resolve the above issues, this paper proposes a framework
named GE-Chat that provides users with evidence of an LLM’s
response in a more accurate and generalizable way. Different from
existing work, this method not only poses constraints on the derived
source that it must come from the raw context, but also, GE-Chat
provides sentence-level fine-grained identification to accurately
mark out the evidence supporting the LLMs’ conclusion. What’s
more, this framework can be applied to any LLM with outstanding
evidence retrieval ability (even a small model with limited fine-
tuning on instruction-following ability, our framework still helps to
highlight the evidence of conclusion). We compare with the direct
LLM source reflection of the response, and our method essentially
improves on the evidence retrieval performance. A demo video and
dataset2are provided for better understanding.
2 Approach
In this section, we will discuss the details of the GE-Chat frame-
work. This work builds upon the graph-based retrieval augmented
generation agent (Graph-RAG), and then provides a three-step par-
adigm for better evidential response generation, we will introduce
the process of constructing the RAG agent first (in Sec. 2.1), and
then explain three components in our framework (Sec. 2.2 to 2.4).
2Click the link to the demo video2.1 Graph-RAG construction
The Graph-RAG [ 11] is unique in its ability to integrate external
information through a structured knowledge graph, thus, support-
ing graph-based queries, and allowing for relational reasoning.
Besides, Graph-RAG also does well in handling multi-hop reason-
ing, this feature can help us to find more than one-hop-related
entities in the knowledge base, extract sub-graphs, and capture
the relational semantics clusters. After a user uploads a document
(.TXT/.DOC/.PDF) format, the metadata will be cut into the corpus
chunks to temporarily store the file, then, we construct a knowledge
graphGby two steps: ➀extract entities from the chunks, and ➁
probe the relations among the entities. The LLM is used to achieve
the knowledge graph, and the KG is used back to LLM as external in-
formation to make responses. We constructed a fast and lightweight
Graph-RAG for a work basis following the implementations [6].
2.2 CoT Reasoning Elicitation
«INST»«SYS»
You are an agent to provide question answering tasks
based on the provided document.
[Task]
Your task is to generate answers to the user’s question,
please think step-by-step for the conclusion, and provide
your thinking steps behind the output.
[Output Format]
Answer: { [text] }
Thoughts: {1.[text] 2.[text] 3.[text] ... n.[text]}

GE-Chat: A Graph Enhanced RAG Framework for Evidential Response Generation of LLMs WWW ’25, April 28–May 2, 2025, Sydney, Australia
This section introduces the Chain-of-Thought (CoT) reasoning
inducer, which serves as a primary step in deriving the reasoning
process. It is well acknowledged that the majority of the LLMs can
automatically perform CoT [ 20] to elicit the reasoning process, i.e.,
how they draw the conclusion step by step. We follow the same
idea to induce the logic steps Logic_steps{step1,step2,...,step𝑛}from
the LLM given a question 𝑄on a document. By designing a CoT
template, we tend to achieve the following:
Answer,Logic_steps =CoT_template(Q,Doc) (1)
The template is inspired by work [ 24], and is shown in the above
green block. This step corresponds to the upper right part of Fig. 1,
CoT Reasoning Elicitation , where each answer is associated
with a CoT chain that explains the reasoning process step by step.
These CoT chains, generated by RAG models, provide a logical
structure to the responses but may not inherently align with the
raw content of the submitted document. To ensure the evidence
strictly originates from the provided source, a critical grounding
step is introduced through sub-graph searching based on entity
matching. This process anchors the CoT reasoning to specific en-
tities and relationships within the knowledge graph, bridging the
gap between generated content and its original context. By doing
so, we enhance the trustworthiness and accuracy of the responses
while maintaining a clear traceability to the original document.
2.3 Efficient Sub-Graph Searching
The sub-graph searching is conducted based on two resources:
Derived KGGand CoTs as in Figure 2. For each of the CoT results:
𝑐𝑖∈{𝑐1,𝑐2,𝑐3,....,𝑐 𝑛}, the𝑐𝑖will be used to match the most relevant
graph entities (Entity Matching), as shown in the output part, the
blue boxes are the identical 𝑐𝑖, and the•is connected by blue edges,
which is the first hop most relevant entities, this is what 𝑐𝑖has been
involved in the LLM’s answer, then we relax this relation to further
second-hop as shown in•green dot. This search is efficient for the
CoT guidance and pre-calculated Gfor n-hop relationships probing
(in contrast to the whole document-range global search). Finding
this entity sub-graph is like finding an anchor that leads to the
original source chunks, we can perform Source Chunk Retrieval
to get several chunks for each 𝑐𝑖in CoTs. This step bridges the made-
up CoT content with the source content of documents by finding the
anchor entity. However, the chunk-level descriptions leave space
for more fine-grained evidence sentences, which we employ an
optimization objective to achieve in the following section.
3. Sub-graph Searching Derived KG: 𝒢
firstsecondOutputInput
Figure 2: A Simple Abstraction for Sub-graph Searching.2.4 Evidence Content Optimization
As in Figure 1, based on the chunk sources (e.g., Chunk1,2,6 ), for
each of the 𝑐𝑖in CoTs, we expect to find a finer sentence from
a certain chunk that best supports the answer sentence 𝑆′with
minimal redundant information. We first formalize this problem,
and then inspired by the trade-off between meaningfulness and
conciseness, we provide a solution in corresponds to the 4. Optimize
in the Figure 1 leveraging the entailment probability.
Problem Setting. Given a chunk of text containing 𝑛sentences,
𝑆={𝑠1,𝑠2,...,𝑠 𝑛}, and a target sentence 𝑆′which represents the
sentence in the answering content, the objective is to find the best
sentence𝑠best∈𝑆that: 1. Maximizes the entailment probability
prob(𝑠𝑛|𝑆′), which measures how strongly 𝑠𝑛entails𝑆′, 2. Mini-
mizes the sentence length len(𝑠𝑛), encouraging concise representa-
tions.
In order to balance these two criteria, we define an objective
functionF(𝑠𝑛), which assigns a score to each sentence 𝑠𝑛based on
its contained meaning and conciseness. The score for each sentence
𝑠𝑛is given by:
F(𝑠𝑛)=𝛼·prob(𝑠𝑛|−𝑆′)−𝛽·len(𝑠𝑛) (2)
where𝛼and𝛽are set as 0.5 to control the weight of the entail-
ment probability, and penalty for longer sentences, respectively.
We want to measure how much the generated evidence means sim-
ilarly to the answer, a rational way is to calculate the entailment
probability prob(𝑠𝑛|−𝑆′). We achieve this by using NLI model3,
which provides a three-element tuple by taking two text pairs 𝑠𝑛
and𝑆′:[logit𝑐𝑜𝑛𝑡,logit𝑛𝑒𝑢𝑡,logit𝑒𝑛𝑡]=−−−→NLI(𝑠𝑛,𝑆′). The output is
processed by transforming into the probability through:
p=Softmax(logit𝑐𝑜𝑛𝑡,logit𝑛𝑒𝑢𝑡,logit𝑒𝑛𝑡) (3)
then we can calculate the−−−→𝑝𝑒𝑛𝑡(𝑠𝑛,𝑆′)=𝑝(𝑠𝑛⊢𝑆′)=p3as the
entailment probability. The optimal sentence 𝑠bestis the one that
maximizesF(𝑠𝑛):𝑠best=arg max 𝑠𝑛∈𝑆(F(𝑠𝑛))Using this objective,
we can find the best evidence that supports the answers in the LLM’s
responses, and this action is performed in a small chunk, which is
not computationally expensive and can be deployed in real-time.
The𝑠bestwill be calculated for each of the answers, such as in
Figure 1, the best evidence output for Ans1is a combination of
sentences Sent.1, 2, 6 . And this will be returned back to users
together for users to understand which part of the answer comes
with the evidence supported and which part lacks such trustwor-
thy information, helping practitioners understand the reliability of
generated content.
Complexity analysis The computational complexity of GE-Chat
involves three main steps: entity extraction, relation probing, and
sub-graph searching. Entity extraction for 𝑛chunks with an average
length of𝑙words has a complexity of 𝑂(𝑛·𝑙). Relation probing for 𝑚
entities to construct a knowledge graph is 𝑂(𝑚2). Sub-graph search-
ing with𝑒edges for𝑘-hop relations is 𝑂(𝑘·𝑒), where𝑘=2in our
implementation. The overall complexity is about 𝑂(𝑛·𝑙+𝑚2+2·𝑒),
dominated by 𝑂(𝑛·𝑙)and𝑂(𝑚2)in practical scenarios.
3off-the-shelf DeBERTa-large model

WWW ’25, April 28–May 2, 2025, Sydney, Australia Longchao et al.
Figure 3: The demonstration of the deployed GE-Chat framework.The user can upload PDF or relevant files, and the highlighted
evidence comes along with the answers the LLMs made. For more examples please check the live demo video.
3 Experiment
3.1 Experiment Setup
Dataset construction To address the scarcity of evidence sources
in prior research, we created a dataset with 1000 cases to evaluate
evidence generation quality across 10 categories: Biology, Business,
Chemistry, Computer Science, History, Management, Mathemat-
ics, Physics, Semiconductors, and Story. The dataset is structured
along three dimensions: (1) PDF length—short (<10 pages), medium
(10-100 pages), and long (>100 pages); (2) question types—Synthesis
(integrating multiple parts), Structure (examining organization),
and Term Explanation (defining specific concepts); and (3) human-
annotated answers with corresponding evidence sentences, ensur-
ing reliability and comprehensiveness. We tested our method on
this dataset and have released dataset and videos4for public use
with standard questions, groundtruth answers and evidence for
reference.
Evaluation Metric: We refer to the existing work [ 12] that uses
the cosine similarity to evaluate the relevance of the generated
text (generated evidence) with the correct text (correct evidence),
and use the conciseness score [ 16] to quantify the LLM’s ability to
find the corresponding evidence precisely. In general, we have the
following calculation that combines the two aspects of evaluation
on Evidence score:
Evidence score=1
𝑁𝑁∑︁
𝑖=1
cos(𝐸𝑖,𝐸𝑔𝑡𝑖)·min
1,𝐿𝑔𝑡𝑖
𝐿𝑖
(4)
where the𝐸𝑖is the embedding of the generated evidence for ques-
tion𝑖, and𝐸𝑔𝑡𝑖is the groundtruth evidence. The first term measures
the cosine similarity of two shreds of evidence, the larger, the better,
and𝐿𝑔𝑡𝑖is the length of text for groundtruth evidence while the 𝐿𝑖
is the generated evidence,𝐿𝑔𝑡𝑖
𝐿𝑖measures how concise the generated
evidence is given the relevance on meaning from cosine similarity.
4Click the link to the testset and videos.3.2 Experiment Result
From the experiment, we observe that the direct evidence retrieval
ability of GPT4o is the best, while other models perform much
worse, especially because too many words are generated that pre-
vent a concise evidence presentation. Then we conducted experi-
ments for comparison, we applied our method GE-Chat to existing
the models except for GPT4o (this is because the GPT4o is involved
and used in the ground-truth reference generation process, even
though we added human-correction, we still aim to avoid potential
bias or unfair advantages). The results are shown in Fig. 4. It is
visible that, compared to original models, applying the GE-Chat
framework could consistently improve the performance of each
model’s evidential-based responses.
Figure 4: The comparison between the original LLMs evi-
dence generation and LLMs with GE-Chat framework.
4 Conclusion
In this paper, we presented GE-Chat , a novel framework addressing
the trustworthiness of LLMs by introducing a rigorous method for

GE-Chat: A Graph Enhanced RAG Framework for Evidential Response Generation of LLMs WWW ’25, April 28–May 2, 2025, Sydney, Australia
evidence retrieval and verification. Through hard constraints on
source derivation and sentence-level highlight capabilities, GE-Chat
significantly enhances the reliability of LLM-generated responses.
Our evaluation across ten diverse LLMs, both open and closed-
source, demonstrates its robustness, versatility, and broad applicabil-
ity. By offering a transparent and user-friendly approach, GE-Chat
contributes to making AI systems more reliable and trustworthy,
paving the way for responsible deployment in critical decision-
making processes.
References
[1]Longchao Da, Tiejin Chen, Lu Cheng, and Hua Wei. 2024. Llm uncertainty
quantification through directional entailment graph and claim level response
augmentation. arXiv preprint arXiv:2407.00994 (2024).
[2]Longchao Da, Kuanru Liou, Tiejin Chen, Xuesong Zhou, Xiangyong Luo, Yezhou
Yang, and Hua Wei. 2024. Open-ti: Open traffic intelligence with augmented
language model. International Journal of Machine Learning and Cybernetics (2024),
1–26.
[3]Longchao Da, Parth Mitesh Shah, Ananya Singh, and Hua Wei. 2024. Evi-
denceChat: A RAG Enhanced LLM Framework for Trustworthy and Evidential
Response Generation. (2024).
[4]Hanxing Ding, Liang Pang, Zihao Wei, Huawei Shen, and Xueqi Cheng. 2024.
Retrieve only when it needs: Adaptive retrieval augmentation for hallucination
mitigation in large language models. arXiv preprint arXiv:2402.10612 (2024).
[5]Asbjørn Følstad and Marita Skjuve. 2019. Chatbots for customer service: user
experience and motivation. In Proceedings of the 1st international conference on
conversational user interfaces . 1–9.
[6]Zirui Guo, Lianghao Xia, Yanhua Yu, Tu Ao, and Chao Huang. 2024. LightRAG:
Simple and Fast Retrieval-Augmented Generation. arXiv preprint arXiv:2410.05779
(2024).
[7]Rujun Han, Yuhao Zhang, Peng Qi, Yumo Xu, Jenyuan Wang, Lan Liu,
William Yang Wang, Bonan Min, and Vittorio Castelli. 2024. RAG-QA Arena:
Evaluating Domain Robustness for Long-form Retrieval Augmented Question
Answering. arXiv:2407.13998 [cs.CL] https://arxiv.org/abs/2407.13998
[8]Wenbo Hu, Yifan Xu, Yi Li, Weiyue Li, Zeyuan Chen, and Zhuowen Tu. 2024.
Bliva: A simple multimodal llm for better handling of text-rich visual questions.
InProceedings of the AAAI Conference on Artificial Intelligence , Vol. 38. 2256–2264.
[9]Ziwei Ji, Nayeon Lee, Rita Frieske, Tiezheng Yu, Dan Su, Yan Xu, Etsuko Ishii,
Ye Jin Bang, Andrea Madotto, and Pascale Fung. 2023. Survey of hallucination in
natural language generation. Comput. Surveys 55, 12 (2023), 1–38.
[10] Ziwei Ji, Tiezheng Yu, Yan Xu, Nayeon Lee, Etsuko Ishii, and Pascale Fung.
2023. Towards mitigating LLM hallucination via self reflection. In Findings of the
Association for Computational Linguistics: EMNLP 2023 . 1827–1843.
[11] Jonathan Larson and Steven Truitt. 2024. GraphRAG: Unlocking LLM discovery
on narrative private data.
[12] Leonie aka helloiamleonie. 2024. evaluatingrag1 . https://blog.langchain.dev/
evaluating-rag-pipelines-with-ragas-langsmith/
[13] Demiao Lin. 2024. Revolutionizing retrieval-augmented generation with en-
hanced PDF structure recognition. arXiv preprint arXiv:2401.12599 (2024).
[14] Zhen Lin, Shubhendu Trivedi, and Jimeng Sun. 2023. Generating with confidence:
Uncertainty quantification for black-box large language models. arXiv preprint
arXiv:2305.19187 (2023).
[15] Chandran Nandkumar and Luka Peternel. 2024. Enhancing Supermarket Robot
Interaction: A Multi-Level LLM Conversational Interface for Handling Diverse
Customer Intents. arXiv preprint arXiv:2406.11047 (2024).
[16] Ragas authors. 2024. conciseness . https://docs.ragas.io/en/stable/concepts/
metrics/summarization_score.html
[17] Jon Saad-Falcon, Joe Barrow, Alexa Siu, Ani Nenkova, Ryan A Rossi, and Franck
Dernoncourt. 2023. PDFTriage: question answering over long, structured docu-
ments. arXiv preprint arXiv:2309.08872 (2023).
[18] Kurt Shuster, Spencer Poff, Moya Chen, Douwe Kiela, and Jason Weston. 2021.
Retrieval augmentation reduces hallucination in conversation. arXiv preprint
arXiv:2104.07567 (2021).
[19] Lei Wang, Jiabang He, Shenshen Li, Ning Liu, and Ee-Peng Lim. 2024. Mitigating
fine-grained hallucination by fine-tuning large vision-language models with
caption rewrites. In International Conference on Multimedia Modeling . Springer,
32–45.
[20] Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi,
Quoc V Le, Denny Zhou, et al .2022. Chain-of-thought prompting elicits reasoning
in large language models. Advances in neural information processing systems 35
(2022), 24824–24837.
[21] Rongxuan Wei, Kangkang Li, and Jiaming Lan. 2024. Improving Collaborative
Learning Performance Based on LLM Virtual Assistant. In 2024 13th International
Conference on Educational and Information Technology (ICEIT) . IEEE, 1–6.[22] Lifan Yuan, Ganqu Cui, Hanbin Wang, Ning Ding, Xingyao Wang, Jia Deng,
Boji Shan, Huimin Chen, Ruobing Xie, Yankai Lin, et al .2024. Advancing llm
reasoning generalists with preference trees. arXiv preprint arXiv:2404.02078
(2024).
[23] Jiaxing Zhang, Zhuomin Chen, Longchao Da, Dongsheng Luo, Hua Wei, et al .
2024. Regexplainer: Generating explanations for graph neural networks in
regression tasks. Advances in Neural Information Processing Systems 37 (2024),
79282–79306.
[24] Zhuosheng Zhang, Aston Zhang, Mu Li, and Alex Smola. 2022. Automatic chain
of thought prompting in large language models. arXiv preprint arXiv:2210.03493
(2022).