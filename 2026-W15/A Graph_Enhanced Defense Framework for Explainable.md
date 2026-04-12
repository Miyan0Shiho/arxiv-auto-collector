# A Graph-Enhanced Defense Framework for Explainable Fake News Detection with LLM

**Authors**: Bo Wang, Jing Ma, Hongzhan Lin, Zhiwei Yang, Ruichao Yang, Yuan Tian, Yi Chang

**Published**: 2026-04-08 04:34:19

**PDF URL**: [https://arxiv.org/pdf/2604.06666v1](https://arxiv.org/pdf/2604.06666v1)

## Abstract
Explainable fake news detection aims to assess the veracity of news claims while providing human-friendly explanations. Existing methods incorporating investigative journalism are often inefficient and struggle with breaking news. Recent advances in large language models (LLMs) enable leveraging externally retrieved reports as evidence for detection and explanation generation, but unverified reports may introduce inaccuracies. Moreover, effective explainable fake news detection should provide a comprehensible explanation for all aspects of a claim to assist the public in verifying its accuracy. To address these challenges, we propose a graph-enhanced defense framework (G-Defense) that provides fine-grained explanations based solely on unverified reports. Specifically, we construct a claim-centered graph by decomposing the news claim into several sub-claims and modeling their dependency relationships. For each sub-claim, we use the retrieval-augmented generation (RAG) technique to retrieve salient evidence and generate competing explanations. We then introduce a defense-like inference module based on the graph to assess the overall veracity. Finally, we prompt an LLM to generate an intuitive explanation graph. Experimental results demonstrate that G-Defense achieves state-of-the-art performance in both veracity detection and the quality of its explanations.

## Full Text


<!-- PDF content starts -->

A Graph-Enhanced Defense Framework for Explainable Fake
News Detection with LLM
BO WANG,School of Artificial Intelligence, Jilin University, China
JING MA∗and HONGZHAN LIN,Hong Kong Baptist University, China
ZHIWEI YANG,Guangdong Institute of Smart Education, Jinan University, China
RUICHAO YANG,University of Science and Technology Beijing, China
YUAN TIAN∗,School of Artificial Intelligence, Jilin University, China; Engineering Research Center of
Knowledge-Driven Human-Machine Intelligence, China
YI CHANG∗,School of Artificial Intelligence, Jilin University, China; Engineering Research Center of
Knowledge-Driven Human-Machine Intelligence, China; International Center of Future Science, Jilin Univer-
sity, China
Explainable fake news detection aims to assess the veracity of news claims while providing human-friendly
explanations. Existing methods incorporating investigative journalism are often inefficient and struggle with
breaking news. Recent advances in large language models (LLMs) enable leveraging externally retrieved reports
as evidence for detection and explanation generation, but unverified reports may introduce inaccuracies.
Moreover, effective explainable fake news detection should provide a comprehensible explanation for all
aspects of a claim to assist the public in verifying its accuracy. To address these challenges, we propose a
graph-enhanced defense framework (G-Defense) that provides fine-grained explanations based solely on
unverified reports. Specifically, we construct a claim-centered graph by decomposing the news claim into
several sub-claims and modeling their dependency relationships. For each sub-claim, we use the retrieval-
augmented generation (RAG) technique to retrieve salient evidence and generate competing explanations. We
then introduce a defense-like inference module based on the graph to assess the overall veracity. Finally, we
prompt an LLM to generate an intuitive explanation graph. Experimental results demonstrate that G-Defense
achieves state-of-the-art performance in both veracity detection and the quality of its explanations.
CCS Concepts:•Information systems →Information retrieval;•Computing methodologies →
Natural language processing.
Additional Key Words and Phrases: Explainable Fake News Detection, Large Language Model, Defense-like
Inference, Claim Decomposition
∗Corresponding author.
Authors’ Contact Information: Bo Wang, School of Artificial Intelligence, Jilin University, Changchun, China, wangbo21@
mails.jlu.edu; Jing Ma; Hongzhan Lin, Hong Kong Baptist University, Hong Kong SAR, China, majing@comp.hkbu.edu.hk,
cshzlin@comp.hkbu.edu.hk; Zhiwei Yang, Guangdong Institute of Smart Education, Jinan University, Guangzhou, China,
yangzw18@mails.jlu.edu.cn; Ruichao Yang, University of Science and Technology Beijing, Beijing, China, yangruichao@
ustb.edu.cn; Yuan Tian, School of Artificial Intelligence, Jilin University, Changchun, China; and Engineering Research
Center of Knowledge-Driven Human-Machine Intelligence, MoE, China, yuantian@jlu.edu.cn; Yi Chang, School of Artificial
Intelligence, Jilin University, Changchun, China; and Engineering Research Center of Knowledge-Driven Human-Machine
Intelligence, MoE, China; and International Center of Future Science, Jilin University, Changchun, China, yichang@jlu.edu.cn.
Permission to make digital or hard copies of all or part of this work for personal or classroom use is granted without fee
provided that copies are not made or distributed for profit or commercial advantage and that copies bear this notice and the
full citation on the first page. Copyrights for components of this work owned by others than the author(s) must be honored.
Abstracting with credit is permitted. To copy otherwise, or republish, to post on servers or to redistribute to lists, requires
prior specific permission and/or a fee. Request permissions from permissions@acm.org.
©2026 Copyright held by the owner/author(s). Publication rights licensed to ACM.
ACM XXXX-XXXX/2026/4-ART
https://doi.org/XXXXXXX.XXXXXXX
, Vol. 1, No. 1, Article . Publication date: April 2026.arXiv:2604.06666v1  [cs.CL]  8 Apr 2026

2 B.Wang and J.Ma et al.
1 Introduction
The rapid dissemination of fake news on social media platforms has become a critical issue, posing
severe risks to both individuals and society. For instance, during the early stage of the COVID-19
pandemic, a widely circulated fake news that “Face masks drastically reduce oxygen intake and cause
carbon dioxide toxicity.1” led many people to refuse to wear masks, which made the virus spread
faster and put extra pressure on healthcare systems around the world. Later, trusted scientific
organizations disproved this claim and provided detailed justifications, helping to rebuild public
trust in mask usage. This case suggests how important solid justification is in reducing the social
harm caused by fake news. However, merely relying on traditional journalism to detect fake news
is not practical, since it is labor-intensive and time-consuming, which results in limited coverage
and delays in debunking. Thus, it demands an automated method, aiming not only to detect fake
news but also to provide timely and clear justifications.
Many studies have explored explainable fake news detection, which aims to perform veracity
detection and generate explanations simultaneously [ 35]. Initially, some works use the relevant
reports as supporting data, and then highlight discrete words, phrases, or sentences from them
that have high contributions to the final prediction, to bring some content of interpretability
[48,52,57,71]. However, these fragmentary explanations lack the intuitiveness2and credibility3
needed for clear understanding. Motivated by the effectiveness of human justifications in veracity
prediction [ 1], Atanasova et al . [4] proposed to generate explanations by summarizing debunked
reports. While this method improves explainability, it inevitably suffers from delayed detection
and explanation generation, as debunked reports are not always available and typically require
substantial human effort, which also makes the approach inefficient. More recently, inspired by the
effectiveness of the wisdom of crowds in fact-checking [ 2], Yang et al . [74] proposed leveraging
externally retrieved raw reports containing collective wisdom as candidate evidence for both
detection and explanation. While these reports providing diverse perspectives can support veracity
prediction, the generated explanations are still produced in an extraction-based manner. As a result,
the selected pieces of evidence are often discrete, lacking a coherent reasoning process and failing
to produce explanations that are fully aligned with the model’s final decision.
Large language models (LLMs) offer a promising direction to this issue, as they have demonstrated
strong capabilities in language processing [ 17,21,49,50,82]. Despite their strengths, LLMs are
often limited by the information contained in their training data and may struggle with answering
time-sensitive or domain-specific queries accurately [ 6,83]. In addition, they are prone to producing
factually incorrect or ungrounded outputs, commonly referred to as hallucinations [ 28,81]. To
alleviate this limitation, retrieval-augmented generation (RAG) [ 37] has emerged as one of the most
representative advanced techniques, aiming to enhance LLMs-based generation by integrating
information retrieved from external data sources [ 18,29,37,78]. RAG has been applied to various
fields, such as question answering [ 22,46], recommendation systems [ 42,70], and fact checking
[11,30]. For fake news detection, some studies [ 10,67,79] typically retrieve relevant information
about a claim from external sources and then use LLMs for reasoning to determine the veracity of the
claim. The reasoning process of LLMs can naturally be regarded as a form of explanation. However,
the correctness of such reasoning strongly depends on the quality and credibility of the retrieved
evidence, but the time-sensitive nature of breaking news makes it highly challenging to obtain
reliable external knowledge. Existing approaches differ in their choice of retrieval sources. Some
retrieve from verified knowledge bases like Wikipedia [ 67,76,77,79], which ensures high credibility
1https://www.bbc.com/news/53108405
2Intuitiveness reflects how easily a human can understand and follow the explanation.
3Credibility denotes the perceived trustworthiness or reliability of the information [7].
, Vol. 1, No. 1, Article . Publication date: April 2026.

A Graph-Enhanced Defense Framework for Explainable Fake News Detection with LLM 3
Decomposed sub-claims:𝒄𝟏	:The discharge of nuclear-contaminated water will spread widely in the ocean.𝒄𝟐	:The discharged water contains harmful radioactive substances that persist in the environment.𝒄𝟑	:Seasalt production is significantly affected by radioactive contamination.𝒄𝟒	:Current purification and monitoring methods are insufficient to ensure salt safety. 𝒄𝟓	:The only source of salt is sea salt.𝑐&𝑐'𝑐(𝑐𝑐)𝑐*GraphR1: Nuclear-contaminated water will pollute seawater and cause salt to cause cancer. It's better to stock up on some healthy and safe salt while we can. ...R2:Damn it! I'm going to buy salt!. ...R3: Don't spread panic! In the current global salt production, rock salt accounts for 41%, underground brine and salt lakes account for 29%, and sea salt accounts for 26%. Even if nuclear-contaminated water has an impact on sea salt, humans still have other sources of salt to consume. ...R4: The presence of nuclear contaminated water increases the risk of salt-induced cancer. ...False[support][support][support][refute]Claim: After the discharge of nuclear-contaminated water, there won't be any healthy salt left for humans to consume.𝑐&𝑐𝑐'𝑐(𝑐)𝑐*Independent
Fig. 1. A false claim from Sina Weibo. Left: The comparison of informativeness and soundness between two
competing parties serves as an indicator of veracity. Right: The different modeling of decomposed sub-claims.
but is limited in its coverage of up-to-date information for verifying breaking news. Others retrieve
from unverified sources such as online comments or user-generated content [ 10,14,68,80], which
are more up-to-date but may contain false or biased information. Directly regarding them as gold
evidence may lead to incorrect predictions. Thus, it remains a key challenge to accurately leverage
the rich yet unverified information expressed in raw reports to support fake news detection.
To address the challenge, we draw inspiration from recent advancements in stance detection
[34,38,45,73], which suggest that the diverse and sometimes conflicting insights presented in raw
reports are vital signals in the pursuit of truth. Based on this insight, we propose to split the wisdom
extracted from the retrieved raw reports into two distinct parties, which allows the detection to
rely on the quality of wisdom rather than its quantity. For instance, as shown in Figure 1 (left),
a representative example illustrates how competing parties emerge around a false claim. For the
supporting party, R1 and R4 briefly discuss the potential link between salt consumption and cancer,
and R2 restates the claim without introducing new information. In contrast, R3 provides the refuting
party with detailed evidence that illustrates its unique viewpoint, which is informative, sound, and
persuasive. This observation motivates an assumption: in a pair of competing parties, the side that is
more consistent with the actual veracity of the claim tends to be supported by evidence with greater
informativeness and soundness4than the opposing side. Accordingly, when an LLM is prompted to
generate explanations based on the evidence from each side, the resulting explanations tend to
reflect the difference in evidence quality. The explanation grounded in the side consistent with the
true veracity label is more likely to exhibit higher quality. We empirically examine this tendency
in Appendix B. Consequently, the veracity of news can be determined through a comparative
analysis between two competing parties. Therefore, our previous work [ 65] introduced an LLM-
equipped defense-based5explainable fake news detection (L-Defense) framework that models this
comparative reasoning process to assess veracity. Specifically, L-Defense extracts two competing
evidence sets, prompts an LLM to generate justifications by inferring reasons for both possible
veracity labels, and compares their quality to determine the final verdict. The final explanation is
then adaptively selected based on the prediction.
Despite the effectiveness of L-Defense in detecting fake news and generating human-friendly
explanations, its outputs remain coarse-grained, especially for complex claims. For example, given
the claim shown in Figure 1 (left), L-Defense generates the explanation based on one retrieved report
R3, which focuses solely on one aspect of the claim, i.e., “sea salt is not the only available salt source”.
4Soundness reflects the factual accuracy and logical coherence of the explanations [66].
5The term “defense” in this paper denotes adebate-style reasoning processbetween competing viewpoints to assess claim
veracity.
, Vol. 1, No. 1, Article . Publication date: April 2026.

4 B.Wang and J.Ma et al.
However, it fails to address other crucial aspects, such as “whether the nuclear-contaminated water is
actually being discharged?” and “whether it affects the production of sea salt?”. This limitation arises
because retrieval and reasoning are performed at the full-claim level, which may overlook important
sub-aspects and result in incomplete explanations that hinder user understanding. To address this
issue, several prior works propose to decompose a claim into multiple sub-claims to facilitate
multi-step reasoning [ 10,11,67,79]. For each sub-claim, they retrieve relevant evidence and verify
its veracity independently. As illustrated in Figure 1 (right, Independent), they aggregate the results
to assess the overall claim. While this improves the granularity of reasoning and explanation,
these methods do not consider the relationships between sub-claims. In many cases, inferring the
veracity of one sub-claim may require others as contexts. Modeling the claim and sub-claims in a
graph structure will further improve reasoning. For example, as shown in Figure 1 (right, Graph),
the sub-claim 𝑐3(“sea salt production is affected”) depends on 𝑐1(“nuclear-water spreads widely”)
and𝑐2(“the discharged substances persist in the environment”). Such dependencies are essential for
accurate reasoning, and overlooking them can lead to fragmented explanations. Moreover, unlike
the approaches that treat sub-claims independently, the graph clearly indicates that the claim 𝑐
relies on𝑐3,𝑐4, and𝑐5simultaneously, providing a more intuitive explanation. Based on this insight,
we enhance L-Defense with claim decomposition and propose modeling the sub-claims as a graph
structure, which enables the representation of rich dependency relationships among sub-claims.
To this end, in this study, we propose a novel graph-enhanced defense framework for explainable
fake news detection, which strives to perform fine-grained reasoning based on claim-centered
graphs and pursue the veracity of claims in a defense-like way. Specifically, we first introduce a
graph construction module that decomposes a news claim into several sub-claims and derives the
dependency edges between them using an LLM, resulting in a claim-centered graph. Then, for
each sub-claim, we use RAG to retrieve its relevant evidence and prompt the LLM to generate
competing explanations by inferring reasons towards two possible veracity labels. This step ensures
that diverse perspectives from raw reports are fully considered. To make the final prediction,
we integrate the generated explanations into the claim-centered graph and employ an LLM to
conduct reasoning over the graph, naturally combining graph-based reasoning with defense-like
inference. In this way, comparing the informativeness and soundness of the two explanations helps
the model determine the most plausible veracity label. Finally, based on the predicted label and
the constructed claim-centered graph, we prompt an LLM to infer the veracity of each sub-claim
and generate a concise textual explanation summarizing the reasoning. Meanwhile, we can obtain
a subgraph as the final explanation graph by removing the explanations that are not consistent
with the sub-claims’ veracities. This explanation graph highlights the dependencies among all
relevant sub-claims, their veracities, and corresponding explanations, enabling users to intuitively
understand the truthfulness of each aspect of the claim. Our main contributions are summarized as:
—We substantially extend the LLM-equipped defense-based explainable fake news detection
(L-Defense) method published in our recent work [ 65] by mimicking the human thinking
process. Instead of assessing a claim as a whole, we decompose it into sub-claims, enabling
the model to handle more complex claims and support more fine-grained reasoning.
—We develop a novel graph-enhanced defense framework that not only uses competing wisdom
extracted from retrieved reports, but also leverages the internal dependencies among sub-
claims, enabling more structured and reliable reasoning for fake news detection.
—The proposed framework generates an intuitive explanation graph accompanied by fine-
grained textual explanations, which are comparable to those of human experts and do not
require supervision from debunked reports.
, Vol. 1, No. 1, Article . Publication date: April 2026.

A Graph-Enhanced Defense Framework for Explainable Fake News Detection with LLM 5
—We conduct extended experiments by exploring multiple variants of G-Defense, which validate
both the effectiveness and the extensibility of the proposed framework.
—The proposed framework achieves state-of-the-art performance on two benchmarks while
providing more comprehensive and reliable veracity explanations.
2 Related Work
2.1 Explainable Fake News Detection Before the LLM Era
Explainability is a critical challenge in fake news detection, and many studies have been devoted
to addressing it. Early works typically use attention mechanisms to highlight salient phrases
[52,71], news attributes [ 72], and suspicious users [ 41,43] to provide explainability. In order
to gain more human-readable explanations, later works further employ sentence-level attention
mechanisms to identify important sentences as explanations [ 44,48,57]. However, these methods
merely uncover regions with high contributions to the veracity prediction rather than regarding
the explanation generation as a dependent task. To address it, Kotonya and Toni [36] views the
explanation generation task as a pre-trained extractive-abstractive summarization task. Atanasova
et al. [4] formats explainable fake news detection as a multi-task learning problem and develops a
joint model trained using external debunked reports gathered from fact-checking websites. However,
debunking claims is a labor-intensive and time-consuming process, leading to significant limitations
in coverage and delays in the availability of debunked journalism. To alleviate this issue, Yang et al .
[74]employs debunked reports as supervision during training, and focuses on the majority opinions
expressed in relevant crowdsourced raw reports to enhance prediction and evidence extraction.
However, it overlooks the inaccurate and biased information in unverified reports, which may
lead to misleading results. Moreover, the extracted evidence, being discrete and fragmented, still
struggled to meet users’ needs for comprehensive and intuitive explanations. In addition, all of
these methods struggle with complex news claims, as they directly analyze the claim without
digging out fine-grained relevant information. Fortunately, the emergence of LLMs with powerful
reasoning and generating abilities presents promising solutions.
2.2 Large Language Models in Fake News Detection
Recent advances in LLMs have shown remarkable performance in various tasks. Researchers have
attempted to employ LLMs for fake news detection. With pre-training on massive corpora, LLMs
inherently contain lots of commonsense and factual knowledge in their parameters, which can be
beneficial for identifying false information. Hu et al . [23] explores the potential of LLMs in fake
news detection, and points out that they are appropriate to provide multi-perspective rationales for
detection, but underperform the fine-tuned language models on detection performance. Then they
propose an adaptive rationale guidance network in which fine-tuned language models selectively
acquire insights on news analysis from the LLMs’ rationales. Similarly, some works prompt LLMs
to generate more viewpoints or human-like reviews to provide more information. DELL [ 64]
engages LLMs to generate news reactions to represent diverse perspectives and explanations
for proxy tasks to enrich the contexts of news articles. With an LLM-based expert ensemble
module, it selectively integrates LLM-generated contents to obtain the final prediction. GenFEND
[47] generates comments by prompting LLMs with diverse user profiles and analyzes them from
multiple perspectives. Although their success in providing multiple viewpoints, LLMs suffer from
severe hallucinations [ 5,31,81], such contents generated through a role-play strategy inevitably
introduce nonfactual or biased information. Moreover, LLMs rely on static knowledge encoded
during pretraining, which limits their ability to handle breaking news.
, Vol. 1, No. 1, Article . Publication date: April 2026.

6 B.Wang and J.Ma et al.
To address these limitations, RAG [ 9,37,53] has been proposed to enhance LLMs by incorporating
externally retrieved relevant information during generation. It enables LLMs to produce accurate
and contextually relevant outputs with augmented external knowledge, mitigating the hallucination
issue, achieving great results on many knowledge-intensive tasks [ 20,29]. In fake news detection,
some works introduce RAG to improve the performance of detection. To counter misinformation,
RARG [ 76] collects supporting evidence from scientific sources and generates responses based on the
evidences via reinforcement learning from human feedback (RLHF). RAFTS [ 77] retrieves relevant
documents from verified sources like Wikipedia and conducts fact verification by synthesizing
contrastive arguments. Furthermore, to detect complex news claims, many works propose to first
decompose the claim into several sub-claims or questions, and then verify these sub-claims step-
by-step through retrieval to derive the final prediction. For example, FOLK [ 67] leverages LLMs to
translate the claim into a First-Order-Logic (FOL) clause consisting of several sub-claims, and then
retrieves knowledge-grounded answers from external knowledge sources for each sub-claim. Based
on FOL-guided reasoning over a set of knowledge-grounded question-and-answer pairs, it makes
veracity predictions and generates explanations. Zhang and Gao [79] introduces a hierarchical
step-by-step prompting method that directs LLMs to separate a claim into several sub-claims and
then verify each of them via multiple question-answering steps progressively.
Despite their success in detection and explanation generation, they are developed for an unreal-
istic scenario. They rely on knowledge from verified sources, which is often limited for emerging
news claims, or access to evidences published after a claim was verified, suffering from ground-truth
leakage [ 10]. To address this issue, Chen et al . [10] introduces a pipeline for realistic fact-checking
of complex claims. It decomposes claims into sub-questions and answers them by retrieving raw
evidence from web documents. Cheung and Lam [14]uses LoRA tuning [ 24] to train a LLaMA-based
[61] detector with filtered retrieved knowledge. Zhang and Gao [80] propose a two-level strategy to
gather fine-grained feedback from the LLM, which serves as a reward for optimizing the retrieval
policy, by rating the retrieved raw documents based on the non-retrieval ground truth. Although
these works aim to operate in more realistic settings, raw documents often contain inaccurate or
biased information, which may lead to incorrect predictions. To address the challenges of incor-
porating such unverified and potentially misleading evidence in realistic settings, we propose a
defense-like strategy that helps the model focus on more reliable information. In addition, these
methods demonstrate that claim decomposition could bring great performance improvement, as
also supported by the detailed analysis in Hu et al . [25] . However, they typically verify sub-claims
independently and aggregate the results to assess the overall claim, overlooking the underlying
dependencies and interactions among them. To address this limitation, we propose constructing a
claim-centered graph that explicitly captures the relationships between sub-claims and facilitates
more comprehensive reasoning.
2.3 Large Language Models on Graphs
While LLMs are widely applied to textual data, an increasing number of studies explore their
use in scenarios where textual information is accompanied by structural data in the form of
graphs. Specifically, many works have extended LLMs to handle text-rich graphs [ 13,19,32,75].
A straightforward method to adapt LLMs to graphs is to translate the graph structure into a text
sequence with specially designed rules and then regard the LLM as a predictor to finish the graph-
related task. For example, InstructGLM [ 75] uses natural language to describe graph structure
for each node and then instructs an LLM for node classification and link prediction. Huang et al .
[27] investigates why the structure information in text format can improve the performance of
LLM on node classification and finds that the structure information is beneficial when the textual
information associated with the node is inadequate. Fatemi et al . [19] explores various graph-to-text
, Vol. 1, No. 1, Article . Publication date: April 2026.

A Graph-Enhanced Defense Framework for Explainable Fake News Detection with LLM 7
(1) Graph Constructionclaim 𝑐LLMsub-claims𝒱!={𝑐",𝑐#,…}edges ℰ!LLM𝑐"𝑐#𝑐$𝑐𝑐%𝑐&𝑐"𝑐#𝑐$𝑐%𝑐&𝑐"'𝑐"(𝑐#'𝑐%'𝑐$'𝑐&'𝑐#(𝑐$(𝑐&(𝑐%(𝑐LLM EncoderMLP with classifier
EmbeddingModelTop-𝑘retrieved evidence 𝐸"LLMTrue-oriented explanation 𝑐"'False-oriented explanation 𝑐"(Sub-claim 𝑐"
(2) Competing Explanation Generation(3) Defense-like Inferenceover graphGraph2Seq(𝒢#)𝑦∗𝑦&=true𝑦&=false𝑐!𝑐"𝑐#𝑐$𝑐%𝑐!&𝑐"&𝑐#'𝑐%'𝑐$'𝑐LLM𝑦∗
SummarizedExplanation 𝑇(4) Explanation Summarization𝑛=5Frozen params
Fine-tuned paramsRetrieval Database 𝒟Explanation Graph
Fig. 2. An architecture overview of the proposedGraph-enhancedDefenseframework for explainable fake
news detection (G-Defense). Given a claim 𝑐, the graph construction module first constructs a claim-centered
graph through claims decomposition and edge generation, where the claim is decomposed into 𝑛sub-claims.
Each sub-claim 𝑐𝑖is then individually processed to retrieve evidence and generate competing explanations.
Next, the claim-centered graph enriched with explanations is used for defense-like inference to derive the
veracity prediction of the claim. Finally, an explanation graph and a summarized explanation 𝑇can be derived
from the last module.
encoding strategies and observes that the simple description of a node’s one-hop index information
yields strong performance across multiple tasks. Inspired by these works, we propose to engage an
LLM to conduct veracity prediction on our generated claim-centered graph to effectively utilize
both textual and structural information.
3 Proposed Approach
This section begins with the task definition for explainable fake news detection and an overview
of the proposedGraph-enhancedDefenseframework (G-Defense). It subsequently presents its
four modules: the graph construction module (§3.1), the competing explanation generation module
(§3.2), the defense-like inference module (§3.3), and the explanation summarization module (§3.4).
Task Definition.Given a news claim 𝑐associated with a gold veracity label 𝑦, explainable fake
news detection aims to predict a veracity label 𝑦∗of claim𝑐and provide an explanation support-
ing the prediction. To assist this process, a set of relevant raw reports D={𝑑 1,𝑑2,...,𝑑|D|}are
pre-retrieved as external database to support reasoning and explanation generation, where each
𝑑𝑗={𝑥 1,𝑥2,...,𝑥|𝑑𝑗|}denotes a relevant report consisting of a sequence of sentences. Notably, the
reports from fact-checking sources and any content published after the release of correspond-
ing fact-check conclusions are strictly excluded from Dto avoid potential leakage of ground
truth. In this work, we evaluate our framework on two benchmark datasets,RAWFCandLIAR-
RAW[ 74], each pairing news claims with raw reports retrieved before the release of official
fact-checking articles. These datasets provide a realistic setting for early fake news detection,
where only unverified reports from the open web are available. Depending on the dataset, the
predicted label 𝑦∗can be either in the set {false,half,true} forRAWFC, or in the finer-grained set
{pants-fire,false,barely-true,half-true,mostly-true,true} forLIAR-RAW. Detailed descriptions and
statistics for both datasets are provided in §4.1.
Overview of G-Defense.G-Defense decomposes a claim into several sub-claims and organizes
them into a dependency graph, enabling fine-grained reasoning. Moreover, for each sub-claim,
, Vol. 1, No. 1, Article . Publication date: April 2026.

8 B.Wang and J.Ma et al.
Claim 𝑐: After the discharge of nuclear-contaminated water, there won't be any healthy salt left for humans to consume.
Claim Decompositionsub-claims𝒱!𝒄𝟏	:The discharge of nuclear-contaminated water will spread widely in the ocean.𝒄𝟐	:The discharged water contains harmful radioactive substances that persist in the environment.𝒄𝟑	:Seasalt production is significantly affected by radioactive contamination.𝒄𝟒	:Current purification and monitoring methods are insufficient to ensure salt safety. 𝒄𝟓	:The only source of salt is sea salt.
Edge Generationedges	ℰ!(𝑐',𝑐(),(𝑐),𝑐(),(𝑐(,𝑐*),(𝑐+,𝑐*),(𝑐,,𝑐*)𝑐'𝑐)𝑐(𝑐𝑐+𝑐,
Fig. 3. An illustrative example of the graph construction process.
G-Defense generates supporting and refuting explanations from retrieved raw reports, allowing the
model to compare competing viewpoints before making a final decision. The framework shown in
Figure 2 consists of four modules: (1) graph construction, the claim is decomposed into sub-claims
and organized into a dependency graph; (2) competing explanation generation, supporting and
refuting explanations are produced for each sub-claim using retrieved raw reports; (3) defense-
like inference, the explanation-enriched graph is encoded and used to predict the veracity; (4)
explanation summarization, a final explanation graph and concise textual justification are generated
to provide intuitive explanations.
3.1 Graph Construction
In real-world scenarios, news claims are often complex and multifaceted, containing multiple
implicit assumptions and logical links. Human experts typically do not evaluate such claims as a
whole. Instead, they tend to break them down into smaller pieces, evaluate each sub-claim using
background knowledge and evidence, and then form a global decision based on how these pieces
interact. Inspired by this human-like thinking process, prior works have proposed to decompose
news claims into multiple sub-claims [ 10,11,67,79]. However, these methods treat sub-claims as
independent units, overlooking the interactions between them. Inferring the veracity of one sub-
claim sometimes requires some additional context, such as the interpretation of other sub-claim(s).
Some sub-claims may only be meaningful when referring to others, so treating them as isolated
units may lead to fragmented or oversimplified reasoning.
To address the above issue, we propose to model the dependencies among sub-claims by con-
structing a claim-centered graph over the decomposed sub-claims. In this graph, nodes represent
the original claim and its sub-claims, while edges capture their dependencies. This structured
representation provides a solid foundation for downstream reasoning and explanation generation.
To construct the graph, we design two steps: claim decomposition and edge generation. An example
of the graph construction process is shown in Figure 3.
Claim Decomposition.We leverage an LLM to capture all explicit aspects in the given claim 𝑐that
will affect its veracity and output the decomposed sub-claims. The process can be formulated as:
LLM(𝐼𝑣(𝑐))=V′={𝑐 1,𝑐2,...,𝑐𝑛},(1)
where𝐼𝑣denotes the prompt used to guide the decomposition and the detail of 𝐼𝑣is listed in
Appendix A, 𝑛is the number of decomposed sub-claims, and V′denotes the set of sub-claims. We
then define the node set of the claim-centered graphGas:
V=V′∪{𝑐}.(2)
Edge Generation.After the claim decomposition, to capture the potential dependency entailed in
these sub-claims and claim, we further instruct an LLM to generate dependency edges between
, Vol. 1, No. 1, Article . Publication date: April 2026.

A Graph-Enhanced Defense Framework for Explainable Fake News Detection with LLM 9
𝒄𝟐	:The discharged water contains harmful radioactive substances that persist in the environment.Retrieval Database 𝒟Top-𝑘retrieved evidence 𝐸"EmbeddingModelEvidence RetrievalExplanation GenerationLLMLLM𝑦"=true
𝑦"=falseTrue-oriented explanation 𝑐"#Treated wastewater can still contain long-lived radionuclides such as cobalt-60 or carbon-14, which cannot be fully removed and may persist in the environment.False-oriented explanation 𝑐"$After multi-stage treatment and dilution, the remaining radioactivity—mainly low-level tritium—falls well below safety limits, making the discharged water unlikely to cause environmental harm.𝑐"	,𝐸"
Fig. 4. An illustrative example of the competing explanation generation process for a sub-claim.
nodes:
LLM(𝐼𝑒(V))=E′={𝑒 1,𝑒2,...,𝑒𝑚},(3)
where𝐼𝑒denotes the instruction prompt used for edge generation and the detail of 𝐼𝑒is in Appendix
A, and each edge 𝑒=(𝑐𝑖,𝑐𝑗)is a directed edge from source node 𝑐𝑖to target node 𝑐𝑗. Note that the
claim𝑐is also included in the input and indexed as 𝑐0so that the LLM has access to the complete
set of nodes when inferring potential dependencies. Since each sub-claim describes an aspect of the
claim, we additionally include an edge from every sub-claim to 𝑐to reflect this inherent relation in
the final graph. The final edge set is defined as:
E=E′∪{(𝑐𝑖,𝑐)|𝑖=1,2,...,𝑛}.(4)
After claim decomposition and edge generation, we obtain the claim-centered graph G=(V,E) ,
which can be used for the sequential reasoning and explanation generation.
3.2 Competing Explanation Generation with RAG
Relying on evidence from a single perspective may lead to inaccurate or biased reasoning, especially
when the available information is noisy or partial. Considering multiple viewpoints provides a more
comprehensive analysis of a claim’s veracity. We assume that the difference between supporting
and refuting explanations, in terms of informativeness and soundness, offers important signals
for downstream inference. For example, if one explanation is significantly more detailed, logically
sound, and better supported by evidence than its opposing party, it can steer the model to make a
more accurate judgment of the claim’s veracity.
To support the generation of such contrasting explanations, we adopt the retrieval-augmented
generation technique. Specifically, for each sub-claim, we retrieve a set of relevant evidences from
externally collected raw reports based on cosine similarity between their embedding representations.
These retrieved evidences are used as the input context to the LLM, prompting it to generate two
competing explanations: one supporting the sub-claim and the other refuting it. This retrieval
process ensures that the LLM can ground its reasoning in diverse real-world perspectives, which is
particularly important when no verified knowledge base is available. An example of the process of
graph construction is shown in Figure 4.
Evidence Retrieval.Since a report may contain evidences related to different sub-claims, we first
split each report into individual sentences, resulting in a corpus of candidate evidential sentences
X={𝑥𝑗}𝑜
𝑗=1, where𝑜=Í
𝑑∈D|𝑑|. We thereby adopt a vanilla pre-trained transformer encoder [ 63]
as the embedding model to generate representations for each sub-claim and candidate evidence.
, Vol. 1, No. 1, Article . Publication date: April 2026.

10 B.Wang and J.Ma et al.
Specifically, given a sub-claim 𝑐𝑖or a candidate evidence 𝑥𝑗, their representations are derived from
the Transformer-Encoder as:
𝒉𝑐𝑖=Pool(Transformer-Enc(𝑐 𝑖)),(5)
𝒉𝑥𝑗=Pool(Transformer-Enc(𝑥 𝑗)).(6)
We then use cosine similarity to retrieve relevant evidence sentences for each sub-claim, as it
provides a simple yet effective way to measure the semantic relevance between embeddings. For
each sub-claim, the top- 𝑘most relevant candidates are selected as the final evidence set. Formally,
the final evidence set for each sub-claim is:
𝐸𝑖={𝑥𝑗|cos(𝒉𝑐𝑖,𝒉𝑥𝑗)is in the top-𝑘}.(7)
Compared with retrieving evidence for the entire claim, retrieving evidence separately for each
sub-claim ensures that all aspects of the claim are supported by the most relevant evidence, rather
than being dominated by generic parts. Therefore, more diverse evidence can be covered, further
enhancing explanation generation and downstream inference.
Explanation Generation.To effectively leverage the rich information contained in retrieved
evidences and obtain coherent explanations, we leverage an LLM to conduct abductive reasoning to
explain why a sub-claim 𝑐𝑖should be assigned the veracity label ˜𝑦based on the retrieved evidence
sets𝐸𝑖and a given prior label ˜𝑦, which can be formulated as:
𝑐𝑣
𝑖=LLM(𝐼𝑟(𝑐𝑖,𝐸𝑖,˜𝑦)),(8)
where the prior label ˜𝑦can be false or true, determines whether the veracity-oriented explanation
𝑐𝑣
𝑖corresponds to a false-oriented explanation 𝑐−
𝑖or a true-oriented explanation 𝑐+
𝑖.𝐼𝑟is the curated
template to integrate these information and its details are as follows. We prompt the LLM with 𝐼𝑟
to generate an explanation 𝑐−/+
𝑖that reasons about how to infer the veracity label ˜𝑦based on the
claim𝑐𝑖and the evidence set𝐸 𝑖.
Prompt used for veracity-oriented explanation generation,𝐼𝑟
Given a claim: 𝑐𝑖, a veracity label ˜𝑦, please give me a streamlined rationale associated with
the claim, without explicitly indicating the label, for how it is reasoned as ˜𝑦. Below are some
sentences that may be helpful for the rationale, but they are mixed with noise:𝐸 𝑖.
Note, please do not repeat the claim and the label in your explanation, just directly output
your streamlined rationale in a short and clear manner.
The reasoning is performed for bothfalseandtrueand thus two reasoning texts are obtained. As
detailed previously, the evidence that is consistent with the actual veracity of the claim brings more
information and is more reasonable than the competing one. Thus, the LLM prefers to generate
solid reasoning in favor of it, while providing weak reasoning with inaccurate information for its
competitor, as demonstrated in Appendix B. In this way, the two LLM-generated veracity-oriented
reasoning𝑐−
𝑖and𝑐+
𝑖, which can be viewed as two explanations to clarify their corresponding veracity
labels, will always possess a relative strength in confidence, greatly facilitating the detection of
fake news.
3.3 Defense-like Inference over Graph
With the claim-centered graph and two veracity-oriented explanations for each sub-claim derived
from LLM, we develop a defense-based fake news detector. This detector aims to identify the relative
strength of the two explanations from their defense and analyze the dependency relationships
, Vol. 1, No. 1, Article . Publication date: April 2026.

A Graph-Enhanced Defense Framework for Explainable Fake News Detection with LLM 11
between sub-claims and claims ultimately providing the veracity. To effectively make use of the
competing explanations, we view a pair of explanations as two property nodes of their corresponding
sub-claim node and add them into the graph. Formally, the final defense-augmented claim-centered
graph is like:
˜G=( ˜V,˜E),(9)
˜V=V∪{𝑐−
𝑖,𝑐+
𝑖}𝑛
𝑖=1,(10)
˜E=E∪{(𝑐−
𝑖,𝑐𝑖),(𝑐+
𝑖,𝑐𝑖)}𝑛
𝑖=1.(11)
Then, we serialize the claim-centered graph ˜Ginto a sequenceH ˜Gas below:
H˜G=Graph2Seq( ˜G),(12)
where Graph2Seq(·) denotes a template to describe the graph structure with natural language.
Here, we use the most straightforward way to translate, whose effectiveness has been demonstrated
by many previous works [ 19]. We first point out all the nodes in the graph, and then describe
the directed edges connected to each node in sequence. The claim is indexed as node 0, and the
sub-claims are indexed starting from node 1. For example, given a graph ˜G𝑡of a claim𝑐and its five
decomposed sub-claims as shown in Figure 2,H ˜G𝑡=Graph2Seq( ˜G𝑡)=“Directed Graph describes
a graph among 0, 1, 2, 3, 4, 5. Node 0 is connected to nodes 3, 4, and 5 by incoming edges. Node 3 is
connected to nodes 1 and 2 by incoming edges.”Notably, the two veracity-oriented explanations
𝑐−/+
𝑖of each sub-claim are not explicitly included as separate nodes in the Graph2Seq( ˜G𝑡). This is
because the explanations are treated as attributes associated with each sub-claim node, providing
supplementary reasoning information rather than altering the structural relationship within the
graph. Consequently, in the serialization process, we mainly focus on describing the claim, the
sub-claims, and their dependency edges. This sequenceH ˜Gis fed into the LLM-Encoder, along with
the contents of the original claim, its sub-claims, and a classification query 𝑄, for final veracity
prediction, which can be formulated as:
𝒉𝑐𝑙𝑠=LLM-Enc(𝐼𝑎(H ˜G,˜V,𝑄)),(13)
where 𝒉𝑐𝑙𝑠is the contextual representation of the query and the claim-centered graph, which jointly
encode both textual and structural information.𝐼𝑎denotes the prompt template used to integrate
all input components and its content is as below. The content of the classification query 𝑄can be
found in𝐼𝑎.
Prompt used for aggregating all information into the LLM-Encoder,𝐼𝑎
Graph used for fake news detection:
#Node Content:
Node 0 (claim):𝑐
Node 1 (Sub-claim n):𝑐 1;
Competing explanations: True-oriented explanation:𝑐+
1; False-oriented explanation:𝑐−
1.
...
Node𝑛(Sub-claim𝑛):𝑐 𝑛;
Competing explanations: True-oriented explanation:𝑐+
𝑛; False-oriented explanation:𝑐−
𝑛.
#Graph Structure:H ˜G
#Query (𝑄): What is the label of Node 0 (claim)? Please directly output your predicted label
from {false, half, true}/{pants-fire, false, barely-true, half-true, mostly-true, true}.
, Vol. 1, No. 1, Article . Publication date: April 2026.

12 B.Wang and J.Ma et al.
Algorithm 1G-Defense Inference Algorithm
Input:News claim𝑐, External DatabaseD, Fine-tuned LLM-Encoder
Output:Predicted label𝑦∗, Explanation graph𝐺∗, Summarized explanation𝑇
// Stage 1: Graph Construction
1:Claim decomposition:𝑉′←Equation (1)
2:Edge generation:E′←Equation (3)
// Stage 2: Competing Explanation Generation with RAG
3:foreach sub-claim𝑐 𝑖∈𝑉′do
4:Evidence retrieval:𝐸 𝑖←Equation (7)
5:Explanation generation:𝑐−/+
𝑖←Equation (8)
6:end for
// Stage 3: Defense-like Inference over Graph
7:Obtain claim-centered graph: ˜G←Equation (9)
8:Graph2Seq:H ˜G←Equation (12)
9:Obtain the embedding for classification:ℎ 𝑐𝑙𝑠←Equation (13)
10:Get prediction result:𝑦∗←Equation (16)
// Stage 4: Explanation Summarization
11:Obtain explanation graph and summarized explanation:G∗,𝑇←Equation (17)
12:return𝑦∗,G∗,𝑇
Then, we define a classifier on top of the rich representation for veracity prediction,
𝒑=𝑃(𝑧|𝒉 𝑐𝑙𝑠)≜softmax(MLP(𝒉 𝑐𝑙𝑠)).(14)
The training objective of the detection task is written as:
L=−∑︁
Dlog𝒑[ˆ𝑦=𝑦],(15)
where𝒑[ˆ𝑦=𝑦] denotes fetching the probability value corresponding to the veracity label𝑦.
The inference procedure can be simply written as:
𝑦∗=arg max𝒑.(16)
3.4 Explanation Summarization
Based on the derived claim-centered graph and the predicted veracity, we further ask the LLM to
give a final summarized explanation. We prompt the LLM to judge the veracity of each sub-claim
based on the predicted veracity of the claim and its competing explanations, and instruct the LLM
to generate a concise explanation for the full claim. The used prompt 𝐼𝑠to instruct the LLM is listed
in Appendix A. The process can be formulated as:
G∗,𝑇=LLM(𝐼𝑠(H ˜G,˜V,𝑦∗)).(17)
In this way, the final explanation consists of two parts. The first part G∗is provided in the form of a
graph, where explanations inconsistent with the veracity of their associated sub-claims are removed
from ˜Vin Eq.(10). Note that this operation filters out only veracity-inconsistent explanations and
does not affect the relevant contextual information about the sub-claim. Because both supporting
and refuting explanations are generated from the same retrieved evidence, contextual sentences
typically appear in both. These necessary background details therefore remain available regardless
of which evidential explanation is removed. The second part 𝑇is a textual explanation for the claim,
as illustrated on the right side of Figure 2. Compared with providing only a plain textual explanation,
, Vol. 1, No. 1, Article . Publication date: April 2026.

A Graph-Enhanced Defense Framework for Explainable Fake News Detection with LLM 13
Table 1. Summary statistics of datasets. # half-true* is also denoted as # half in RAWFC.
Dataset RAWFC LIAR-RAW
Claim
# Claims 2,012 12,590
# pants-fire - 1,013
# false 646 2,466
# barely-true - 2,057
# half-true* 671 2,594
# mostly-true - 2,439
# true 695 2,021
Veracity Labels 3 6
Report per claim
# min 1 1
# max 30 30
# avg 21.0 12.3
Sentence per report
# min 1 1
# max 155 59
# avg 7.4 5.5
the explanation graph G∗explicitly visualizes the reasoning process by showing how individual
sub-claims contribute to the final veracity prediction, making the explanation more transparent
and easier to understand. Building on this, by presenting the dependencies and veracity labels of
sub-claims in a structured format, the graph enables users to quickly understand which aspects
of the claim are true or false. In general, the graph-based explanation offers a fine-grained and
more intuitive view of the claim’s veracity reasoning, complementing the textual summary 𝑇and
enhancing the overall explainability of the proposed framework. During the automatic evaluation
of the quality of explanation, we convert the explanation graph into text using Graph2Seq( ·) and
additionally provide the full textual content of all nodes. The overall inference process of our
G-Defense is summarized in Algorithm 1.
4 Experiment
In this section, we first introduce the experimental setup in §4.1 and then evaluate G-Defense on
the task of fake news detection in §4.2. To demonstrate the significance of each proposed module,
we conduct an extensive ablation study as detailed in §4.3. Moreover, we evaluate the quality and
usefulness of fine-grained explanations generated by G-Defense in §4.5. In §4.6, we present a case
study to demonstrate how our proposed framework improves performance and provides intuitive
graph-based explanations. Finally, in §4.7, we discuss the inherent robustness of G-Defense to error
propagation.
4.1 Experimental Setup
Datasets.Benchmarks for explainable fake news detection, such asPubHealth[ 36], rely on
official debunked reports written by journalists as evidence. However, such datasets are impractical
for real-world, timely fake news detection, as official reports are typically published long after
fake news has spread. Also, it does not align with our proposed model, which aims to detect fake
news using externally retrieved raw reports. Therefore, to evaluate the effectiveness of detecting
, Vol. 1, No. 1, Article . Publication date: April 2026.

14 B.Wang and J.Ma et al.
fake news before official reports are available, we select RAWFC and LIAR-RAW [ 74] to assess our
proposed method. These datasets better reflect the general challenge of fake news detection with
explainability, as they use raw reports retrieved from the Internet. And any reports published after
the claims were verified are filtered out, ensuring a more realistic evaluation setting.
RAWFC contains the claims collected from Snopes6and relevant raw reports by retrieving
claim keywords. LIAR-RAW is extended from the public dataset LIAR-PLUS [ 1] with retrieved
relevant raw reports, containing claims from Politifact7. Both datasets originate from fact-checking
platforms, i.e., Snopes and PolitiFact, which collect and verify claims that initially circulated on
social media and online news outlets. In RAWFC, claims are categorized into three labels: {false,half,
true}, while those in LIAR-RAW are annotated with six veracity labels: {pants-fire,false,barely-true,
half-true,mostly-true,true}. Both datasets follow an 8:1:1 split for train/valid/test sets. The detailed
statistics for these datasets are listed in Table 1.
RAWFC has a simpler veracity classification scheme with only three labels, whereas LIAR-RAW
adopts a finer-grained six-label classification. This difference makes LIAR-RAW more nuanced
in distinguishing varying degrees of truthfulness, while RAWFC provides a more generalized
and easily interpretable classification. Moreover, RAWFC provides significantly more supporting
raw reports per claim than LIAR-RAW. On average, each claim in RAWFC is associated with 21.0
retrieved reports, whereas LIAR-RAW only has an average of 12.3 reports per claim. And the
average number of sentences per report in RAWFC is higher than that in LIAR-RAW. This suggests
that RAWFC provides a richer set of contextual information for fake news detection, potentially
leading to more robust detection and explanations.
Training Setups.We use the “gpt-3.5-turbo-0125” [ 51] developed by OpenAI as our LLM backbone
for each frozen LLM component when there is no special declaration. Although our framework can
be easily adapted to more recent LLMs, here we choose to conduct our experiments on a widely
recognized earlier model. This choice allows us to fairly demonstrate the fundamental capability
of our framework rather than relying on the improvements from LLM updates. In the competing
explanation generation module (§3.2), we set 𝑘=5to retrieve top-5 candidate evidences for each
sub-claim. For the LLM encoder used in the defense-like inference module (§3.3), we initialize
it with Flan-T5-large [ 16] developed by Google. To fine-tune it, we use a mini-batch Stochastic
Gradient Descent (SGD) to minimize the loss functions, with Adam optimizer, 10% warm-up, and
a linear decay of the learning rate. The learning rate used to fine-tune the LLM encoder is7 𝑒−5
for RAWFC and5 𝑒−5for LIAR-RAW, the batch size is 8 for RAWFC and 4 for LIAR-RAW, and the
training epoch is 5 for both datasets. For the temperature of LLM in our framework, we set it to 0.8,
allowing the LLM to flexibly apply its internal knowledge for detection and explanation generation.
We set the temperature to 0 during the evaluation of generated explanations. All experiments are
conducted on a single A40 GPU.
4.2 Veracity Prediction Performance
Baselines.To compare the effectiveness of the proposed G-Defense in fake news detection,
we evaluate its performance against the following baseline models, which are divided into three
categories, i.e., traditional non-LLM-based approaches (1-6), LLM-based approaches (7-12), and the
variants of G-Defense (13-16):
(1)dEFEND[ 57] introduces a GNN-based sentence-comment co-attention network for veracity
prediction with explanations to jointly model sentences from news article and user comments
for veracity prediction.
6https://www.snopes.com/
7https://www.politifact.com/
, Vol. 1, No. 1, Article . Publication date: April 2026.

A Graph-Enhanced Defense Framework for Explainable Fake News Detection with LLM 15
(2)SBERT-FC[ 36] employs Sentence-BERT [ 54] to encode contextualized representations for
candidate evidences and news claim. Next, it ranks evidences based on their cosine similarity
with the claim and selects the top-𝑘most important sentences for veracity prediction.
(3)GenFE[ 4] employs DistilBERT [ 56] to encode the simple concatenation of a claim and its
debunked journalism for veracity prediction.
(4)GenFE-MT[ 4] enhances GenFE by further introducing a multi-task setup. It jointly predicts
salient sentences in news articles and the veracity of the claim.
(5)CofCED[ 74] utilizes DistilBERT to encode news claims and all sentences in raw reports.
Then it introduces a coarse-to-fine cascaded evidence-distillation framework by focusing on
key evidence to detect fake news.
(6)MUSER[ 39] employs Sentence-BERT to perform multi-step retrieval and build associations
among pieces of evidence required for news verification. Then the semantic connection
between the news claim and the evidence is analyzed with BERT for detection.
(7)LLaMA2 claim [62] prompts LLaMA2-7b with the news claim to directly generate a veracity
prediction and corresponding explanation.
(8)GPT-3.5 claim [51] is similar to LLaMA2 claim, but replaces LLaMA2 with GPT-3.5.
(9)GPT-3.5 full[51] prompts GPT-3.5 with both the claim and all retrieved relevant reports.
The absence of LLaMA2 fullis that the 7b model struggles to produce consistent output after
processing such lengthy inputs.
(10)FactLLaMA[ 14] leverages the LORA tuning [ 24] to supervised fine-tunes a LLaMA2-7b
with the claim.
(11)FactLLaMA know [14], compared with FactLLaMA, retrieves external evidence from search
engines and feeds the LLaMA2-7b with the claim and the filtered relevant evidence.
(12)L-Denfense LLaMA2 andL-Denfense GPT-3.5 [65] leverage a defense-like inference strategy
on the news claim to perform classification by fine-tuning a RoBERTa model [ 40], while
employing LLaMA2 [62] and GPT-3.5 [51] as explanation generators respectively.
(13)G-Defense hyper replaces the dependency-based claim-centered graph with a hypergraph
structure, where sub-claims are grouped into topic-centered hyperedges rather than connected
through pairwise logical dependencies. The prompt 𝐼ℎused to instruct the LLM to generate
hyperedges is provided in Appendix A.
(14)G-Defense background augments each sub-claim with additional background context. For
every sub-claim, we retrieve 20 related sentences and prompt the LLM to summarize them
into a concise background description. This background information is then supplied to the
model to help it better understand the sub-claim and assess its veracity. The prompt 𝐼𝑏used
to instruct the LLM is provided in Appendix A.
(15)G-Defense decomp+ enhances the claim decomposition stage by replacing the original simple
prompt with a more structured, domain-informed version. The revised prompt incorpo-
rates decomposition strategies, news-analysis cues, dependency cues, and fact-checking
dimensions. The full prompt𝐼𝑣+is provided in Appendix A.
(16)G-Defense LLaMA3.1 substitutes the GPT-based backbone with the open-source LLaMA-3.1-8B
model, allowing us to explore the framework’s adaptability to different LLM backbones.
Metrics.Following previous works [ 14,65,74], we use macro-average precision (P), macro-
average recall (R) and macro-average F1 (macF1). They are calculated as: P=1
𝐿Í𝐿
𝑙=1P𝑙,R=
1
𝐿Í𝐿
𝑙=1R𝑙,macF1=1
𝐿Í𝐿
𝑙=12·P𝑙·R𝑙
P𝑙+R𝑙, where𝐿is the total number of classes, and P𝑙andR𝑙denote the
precision and recall for class𝑙respectively.
Table 2 presents the veracity prediction results of our proposed method and competitive base-
lines evaluated on the two benchmarks. We observe that our proposed G-Defense or its variants
, Vol. 1, No. 1, Article . Publication date: April 2026.

16 B.Wang and J.Ma et al.
Table 2. Veracity prediction results on RAWFC and LIAR-RAW. †Resulting numbers are reported by Yang et al .
[74], the results of FactLLaMA and L-Defense are taken from the original paper, and the results of MUSER
are reproduced by us. Bold numbers indicate the best performance, and underlined numbers indicate the
second-best performance. Results for G-Defense and its variants are averaged over three random runs.
RAWFC LIAR-RAW
P R macF1 P R macF1
Traditional approach
dEFEND [57]†44.93 43.26 44.07 23.09 18.56 17.51
SBERT-FC [36]†51.06 45.92 45.51 24.09 22.07 22.19
GenFE [4]†44.29 44.74 44.43 28.01 26.16 26.49
GenFE-MT [4]†45.64 45.27 45.08 18.55 19.90 15.15
MUSER [39] 47.47 47.50 47.48 27.80 26.58 26.52
CofCED [74]†52.99 50.99 51.07 29.48 29.55 28.93
LLM-based approach
LLaMA2 claim [61] 37.30 38.03 36.77 17.11 17.37 15.14
GPT-3.5 full[51] 39.48 45.07 39.31 29.64 23.57 21.90
GPT-3.5 claim [51] 47.72 48.62 44.43 25.41 27.33 25.11
FactLLaMA [14] 53.76 54.00 53.76 32.32 31.57 29.98
FactLLaMA know [14] 56.11 55.50 55.65 32.46 32.05 30.44
L-Denfense LLaMA2 [65] 60.95 60.00 60.12 31.63 31.71 31.40
L-Denfense GPT-3.5 [65] 61.72 61.01 61.20 30.55 32.20 30.53
Ours
G-Defense 64.99±0.7864.52 ±1.0264.31±1.2033.09 ±1.0131.55±0.8431.55±0.51
G-Defense hyper 64.62±1.7963.02±1.3163.44±1.4331.21±0.5929.53±0.6429.75±0.57
G-Defense background 66.29±1.6365.49±0.9865.50±1.0831.50±0.8931.28±0.6230.89±0.50
G-Defense decomp+ 65.63±2.4564.33±1.8864.34 ±2.0432.03±0.2430.23±0.4730.36±0.78
G-Defense LLaMA3.1 60.97±0.8461.23±0.2760.76±0.7634.17±0.6732.37±0.2432.49±0.49
demonstrate a significant improvement over L-Defense and achieves state-of-the-art performance
on all metrics.
The introduction of LLMs has significantly enhanced the performance of fake news detection.
Among the first three LLM-based approaches without any tuning, GPT-3.5 claim achieves the best
results. With the exception of CofCED which uses externally retrieved reports in a coarse-to-fine
evidence distillation framework, most traditional approaches underperform compared to GPT-
3.5claim. Notably, GPT-3.5 claim operates in a zero-shot setting, making veracity predictions based
solely on the claim itself without incorporating any external data. On the one hand, it suggests strong
generalization capabilities and great potential of LLMs in fake news detection due to their extensive
pre-training on a large corpus. On the other hand, its competitive performance may because
GPT-3.5 has encountered similar claims during pretraining. In contrast, LLaMA2 claim significantly
underperforms, likely due to its smaller model size and pretraining on a smaller corpus compared
, Vol. 1, No. 1, Article . Publication date: April 2026.

A Graph-Enhanced Defense Framework for Explainable Fake News Detection with LLM 17
2 3 4 5
Number of sub-claims010203040Percentage (%)RAWFC
LIAR-RAW
Fig. 5. Distribution of claims by the number of decomposed sub-claims in the test sets of RAWFC and
LIAR-RAW.
to GPT-3.5. And GPT-3.5 fullwhich conducts prediction with external retrieved reports performs
worse than GPT-3.5 claim in most metrics, indicating that providing a large volume of retrieved
reports without any processing may introduce noise or biases, negatively impacting prediction
accuracy. Apart from the three LLM-based methods under a zero-shot setting, all fine-tuned models
combined with LLMs outperform all traditional approaches. FactLLaMA and FactLLaMA know simply
fine-tune LLaMA2 with LoRA without any special design, while they show much better results
than LLaMA2 claim. This suggests that fine-tuning can significantly enhance the capability of LLMs
in fake news detection, enabling them to better handle complex reasoning tasks. Furthermore,
it highlights the limitations of traditional methods, which struggle to compete with fine-tuned
LLMs due to their lack of deep language understanding. As for L-Defense, it fine-tunes a small
language model but incorporates the LLM’s powerful reasoning and generation ability into its
specifically designed framework, leading to a better result than fine-tuned LLaMA2. This highlights
the importance of designing well-structured models to effectively leverage the powerful capabilities
of LLMs.
G-Defense consistently outperforms prior methods across both benchmarks.Compared with L-
Defense, our proposed G-Defense employs defense-like inference on a claim-centered graph using
LLM, achieving excellent performance in veracity prediction on both benchmarks. In addition,
to further validate the statistical significance of the improvements, we conduct paired t-tests
over five random seeds on macF1 between G-Defense and the strongest baseline, L-Defense. The
improvements are highly significant on both datasets, with 𝑝=3×10−4on RAWFC and 𝑝=9×10−5
on LIAR-RAW. The results suggest the effectiveness of claim decomposition and graph construction
in improving veracity prediction, which will be further discussed in §4.3. By structuring the claim,
sub-claims, and competing explanations into a graph, G-Defense enhances its inference process,
allowing for a more comprehensive and more accurate classification. The improvement is especially
significant on RAWFC. Compared to the previous SOTA method, L-Defense GPT-3.5 , it improves
performance by at least 3.1% across all evaluation metrics. On LIAR-RAW, while G-Defense achieves
slightly lower recall than L-Defense GPT-3.5 , it shows improvements of 2.54% in precision and 1.02%
in macF1. These performance differences can be attributed to the inherent differences in data
characteristics between RAWFC and LIAR-RAW. First, as shown in Table 1, RAWFC provides more
reports per claim and more sentences per report compared to LIAR-RAW. This richer evidence
context benefits explanation generation for each sub-claim and improves veracity prediction.
Second, claims in RAWFC are generally more complex than those in LIAR-RAW. We compare the
average token lengths of claims and find that RAWFC claims are longer on average (28.1 tokens)
than those in LIAR-RAW (23.8 tokens). Furthermore, we present the distribution of claims by
the number of decomposed sub-claims in the test sets of both datasets in Figure 5. It shows that
, Vol. 1, No. 1, Article . Publication date: April 2026.

18 B.Wang and J.Ma et al.
claims in RAWFC are generally decomposed into more sub-claims than those in LIAR-RAW. These
observations indicate the claims in RAWFC are more complex. The significant improvements on
RAWFC highlight the effectiveness of our method in handling more complex claims.
The framework of G-Defense shows great potential in enhancing the performance of fake news
detection.To explore the potential of our proposed G-Defense framework, we further explore
four variants of it. G-Defense hyper performs consistently worse than G-Defense on both datasets,
indicating that grouping sub-claims solely by topical similarity is insufficient for veracity prediction.
Fine-grained logical dependencies provide clearer reasoning paths, enabling more accurate aggre-
gation of evidence. When dependencies do not exist, sub-claims typically correspond to different
facets of the main claim and contribute independently to the final prediction, which the original
design naturally accommodates. G-Defense background shows significant gains on RAWFC, improving
macF1 from 64.31 to 65.50, but leads to a slight decrease on LIAR-RAW. RAWFC offers longer claims
and richer reports, making background consolidation more beneficial for guiding sub-claim detec-
tion, whereas LIAR-RAW’s shorter, self-contained claims leave less room for improvement. This
result suggests that incorporating additional, informative context remains a promising direction for
further enhancement. G-Defense decomp+ uses the enhanced prompt for claim decomposition and
shows slight gains in precision and macF1 on RAWFC, indicating that the additional guidance helps
generate clearer sub-claims for fact-rich and complex claims. On LIAR-RAW, however, we observe
a slight performance drop. Claims in LIAR-RAW are generally shorter and contain fewer explicit
factual components, leaving limited room for additional decomposition cues to introduce meaning-
ful structure. In such cases, the stronger prompt may lead to unnecessary over-splitting, which
may slightly disturb downstream inference. These observations indicate that the effectiveness of
enhanced decomposition guidance depends on the dataset and the inherent structure of the claims.
Finally, G-Defense LLaMA3.1 surpasses G-Defense on LIAR-RAW on all metrics, and shows small im-
provements on RAWFC compared with L-Defense LLaMA2 . This demonstrates that G-Defense is not
tied to a specific LLM backbone and can effectively leverage models with different parameter scales.
Overall, these variants collectively show that G-Defense is a flexible and extensible framework
whose performance can be further improved through enriched context, enhanced decomposition
prompts, or newer LLM backbones, allowing users to select the configuration best suited to their
application needs.
4.3 Ablation Study
To evaluate the contribution of each component, we conduct an extensive ablation study for
G-Defense on veracity prediction by removing or replacing the key component:
(1)“w/o sub-claims”: It does not perform claim decomposition or graph construction. In the
competing explanation generation module (§3.2), it retrieves relevant evidences for the entire
claim and directly generates competing explanations for the entire claim.
(2)“w/o edges”: This variant removes the edge generation process in §3.1 which creates depen-
dency edges between sub-claims. During the defense-like inference (§3.3), it removes the
graph structure descriptionH ˜Gin Eq.(13).
(3)“w/o evidence”: It removes the evidence retrieval step in §3.2. As a result, the LLM generates
veracity-oriented explanations for sub-claims using only its own knowledge rather than
retrieved external information.
(4)“w/o competing”: This variant removes the prior veracity label for sub-claims during ex-
planation generation with LLM in §3.2. The LLM is then instructed to generate only one
explanation based on its analysis rather than considering competing explanations.
, Vol. 1, No. 1, Article . Publication date: April 2026.

A Graph-Enhanced Defense Framework for Explainable Fake News Detection with LLM 19
Table 3. Ablation study of veracity prediction on two datasets.
RAWFC LIAR-RAW
Method P R macF1 P R macF1
G-Defense 65.89 65.54 65.65 32.94 32.09 32.09
w/o sub-claims 56.29 56.02 56.10 30.94 28.13 28.52
w/o edges 64.12 63.49 63.40 31.39 30.52 30.60
w/o evidence 56.68 55.98 55.76 30.12 29.44 29.30
w/o competing 57.24 55.96 55.94 30.42 29.53 29.46
w/o inference training 54.71 51.48 47.68 28.24 25.13 19.45
(5)“w/o inference training”: It replaces the fine-tuning process of defense-like inference in
§3.3 with direct predictions from GPT-3.5, which allow G-Defense to conduct in a zero-shot
setting.
As shown in Table 3, the ablative variants suffer different degrees of performance degradation,
indicating the effectiveness of each component in our proposed G-Defense for fake news detection.
The “w/o sub-claims” variant shows a significant decline because the model loses the fine-grained
factual units and their corresponding evidence needed for careful reasoning. Without decomposition,
the LLM must judge the entire claim at once, which makes it difficult to identify the key factual
conditions that determine truthfulness. Removing edges causes a smaller drop, indicating that while
sub-claims provide most of the useful information, the dependency edges help connect them into
a clear reasoning flow. Without these dependency relations, the model cannot easily understand
how different aspects of the claim relate to each other. In the “w/o evidence” variant, performance
decreases because the model can no longer rely on externally retrieved raw reports. LLMs often
lack recent or complete knowledge especially for breaking news, so removing retrieved evidence
forces the model to depend only on what it already knows, which limits accuracy. This shows
that external evidence is important for grounding the reasoning process. This result underscores
the necessity of incorporating RAG in fake news detection to provide informative evidence and
improve model performance. The “w/o competing” variant performs worse because relying on
a single explanation can be one-sided. The competing-explanation mechanism encourages the
model to consider both supporting and opposing views, which reduces bias and helps prevent
mistaken conclusions. Finally, the “w/o inference training” variant achieves the worst result among
all variants, which proves the importance of fine-tuning in such a classification task, consistent
with previous findings [ 23,59]. Without training, the classifier cannot learn how to integrate graph
features, so it falls back to a purely zero-shot prediction strategy. Notably, even under this condition,
this variant still obtains acceptable performance and outperforms many traditional approaches.
This further highlights the superiority of our proposed framework.
4.4 Impact of Claim Complexity
To examine when claim decomposition is most beneficial, we group test samples by the number of
decomposed sub-claims 𝑛. This measure reflects the factual complexity of a claim more reliably than
surface length. And we compare G-Defense with its “w/o sub-claims” variant described in Section
4.3. The results are shown in Figure 6. When a claim contains only two sub-claims, G-Defense
performs better on LIAR-RAW but slightly worse on RAWFC. A possible reason is that decomposing
very simple claims is unnecessary and may introduce minor noise. As complexity increases ( 𝑛≥3),
, Vol. 1, No. 1, Article . Publication date: April 2026.

20 B.Wang and J.Ma et al.
2
(22%)3
(44%)4
(28%)5
(6%)
Number of sub-claims (ratio in dataset)50.0%55.0%60.0%65.0%70.0%75.0%80.0%mac-F1 Score (%)RAWFC
G-Defense
w/o sub-claims
2
(33%)3
(42%)4
(18%)5
(7%)
Number of sub-claims (ratio in dataset)25.0%27.5%30.0%32.5%35.0%37.5%40.0%mac-F1 Score (%)LIAR-RAW
G-Defense
w/o sub-claims
Fig. 6. MacF1 scores of G-Defense and its w/o-sub-claims variant grouped by the number of sub-claims on
the RAWFC (left) and LIAR-RAW (right) test sets. The percentages under each group indicate the proportion
of samples within that group.
the pattern changes consistently. G-Defense outperforms the “w/o sub-claims” across both datasets,
and the performance gap grows as the number of sub-claims becomes larger. Claims with more
sub-claims usually involve multiple aspects, conditions, or factual components. Predicting them
as a single long sentence can be unstable, because different pieces of information may interfere
with each other. Claim decomposition provides fine-grained reasoning units and allows the graph
structure to integrate evidence more reliably. More than half of all test samples fall into the 𝑛≥3
range. This indicates that the cases where decomposition brings meaningful benefits cover a large
portion of real-world claims. Although it may cause slight performance drops for the simplest
cases, we apply the same decomposition procedure to all claims regardless of their complexity.
Claim complexity is difficult to judge automatically, and even short claims may imply multiple facts.
Using a unified detection pipeline also ensures consistent graph-structured explanations. Overall,
this analysis shows that claim decomposition is particularly effective for claims involving more
aspects or factual elements.
4.5 Explanation Evaluations
Baselines.Based on the veracity prediction results in Table 2, we introduce the following baseline
models for explanation evaluations:
(1)Oracle, which offers detailed justifications for the claim’s classification according to its true
veracity label. It achieves this by supplying GPT-3.5 with both the claim and its actual veracity
label.
(2)CofCED[74], recognized as the best model among traditional approaches.
(3)GPT-3.5 claim, which exhibits the best veracity prediction performance among LLM-based
approaches without any training, providing explanations according to its veracity predictions.
(4)GPT-3.5 full, which is provided with retrieved raw reports for comparison withGPT-3.5 claim ,
providing explanations according to its veracity predictions.
(5)L-Defense LLaMA2 andL-Defense GPT-3.5 [65], which is the previous SOTA model. Since its
generated explanations are limited to three categories:true,half-true, andfalse, we map the
six labels in the LIAR-RAW dataset to these three categories before getting explanations.
Specifically,pants-fire,false, andbarely-trueare grouped under thefalsecategory. Thehalf-true
category remains unchanged, whilemostly-trueandtrueare merged into thetruecategory.
Metrics.Traditional automated evaluation metrics like ROUGE are often insufficient for assessing
the quality of output generated by LLMs [ 8], as they primarily measure surface-level n-gram
, Vol. 1, No. 1, Article . Publication date: April 2026.

A Graph-Enhanced Defense Framework for Explainable Fake News Detection with LLM 21
Table 4. Evaluation results of explanation quality using a 5-Point Likert scale rating by ChatGPT on RAWFC
and LIAR-RAW. For metrics D and M, a lower score ( ↓) indicates better performance, while a higher score ( ↑)
indicates better performance for the remaining metrics. The bold numbers denote the best results in addition
to Oracle while the underlined ones are better than Oracle.
RAWFC LIAR-RAW
D↓M↓I↑S↑R↑D↓M↓I↑S↑R↑
Oracle - 1.52 4.46 4.73 4.72 - 1.85 4.44 4.60 4.69
CofCED [74] 1.53 2.74 2.89 1.93 2.46 1.33 3.64 1.75 1.76 1.59
GPT-3.5 claim 1.70 1.97 4.00 4.44 4.68 1.39 2.27 3.93 4.29 4.50
GPT-3.5 full 1.81 2.07 4.44 4.62 4.69 1.39 2.29 3.71 4.04 3.99
L-Defense LLaMA2 [65] 1.30 1.95 4.44 4.67 4.62 1.36 2.20 4.394.64 4.63
L-Defense GPT-3.5 [65] 1.30 1.91 4.17 4.41 4.49 1.31 2.06 4.12 4.28 4.47
G-Defense1.00 1.87 4.60 4.72 4.70 1.26 2.02 4.424.384.65
overlaps and fail to capture semantic accuracy and reasoning quality. Recently, Chen et al . [12]
has demonstrated that GPT performs well in evaluating text quality from multiple perspectives,
even without reference texts. Additionally, studies have shown that LLM-based evaluation closely
aligns with expert human assessments [ 15,26]. Therefore, we utilize GPT-3.5 to evaluate the quality
of explanations based on four widely adopted human evaluation metrics [ 66,74]:misleadingness,
informativeness,soundness, andreadability. Each metric is rated on a 5-point Likert scale, where 1
represents the lowest score and 5 the highest, except for misleadingness. The definitions of these
metrics are as follows:
—Misleadingness (M)measures whether the explanation aligns with the true veracity label
of a claim. It is rated from 1 (not misleading) to 5 (highly misleading);
—Informativeness (I)measures the extent to which the explanation provides new insights,
such as background details and additional context. The scale ranges from 1 (not informative)
to 5 (highly informative);
—Soundness (S)describes whether the explanation seems valid and logical, with a rating scale
ranging from 1 (not sound) to 5 (very sound);
—Readability (R)assesses whether the explanation follows proper grammar and structural
rules, and whether the sentences in the explanation fit together and are easy to follow, with
a rating scale ranging from 1 (poor) to 5 (excellent).
To further validate the reliability of LLM-based evaluation, we introduce an automated metric
calledDiscrepancy (D). Unlike misleadingness, which relies on LLM assessments of explanations,
discrepancy is directly computed based on the consistency between the predicted and gold veracity
labels without involving LLM evaluation. Specifically, it is defined as the absolute difference between
the predicted and true labels. For RAWFC, we map the three veracity labels {false,half,true} to
numerical scores {0, 2.5, 5}, and for LIAR-RAW, the six fine-grained labels are mapped to {0, 1, 2, 3,
4, 5} based on their annotated truthfulness degrees. This automatic metric provides an objective
measure that complements the LLM-based misleadingness evaluation. The consistency between
discrepancy and misleadingness can reflect the reliability of the LLM-based evaluation.
We list the results of the LLM-based explanation evaluation in Table 4. The Oracle, which is
prompted with the ground-truth labels, outperforms all other methods across nearly all metrics
and can be regarded as an upper bound for explanation quality. In comparison, G-Defense achieves
, Vol. 1, No. 1, Article . Publication date: April 2026.

22 B.Wang and J.Ma et al.
Misleadingness
 Informativeness
 Soundness
 Readability
1.01.52.02.53.03.54.0Oracle
CofCEDGPT-3.5claim
GPT-3.5fullL-DefenseLLaMA2
L-DefenseGPT-3.5G-Defense
Fig. 7. Explanation evaluation results using a 5-Point Likert scale rating by five human annotators on 30
randomly sampled claims from the test set of RAWFC. Scores from five annotators were averaged.
either superior or comparable performance, and consistently outperforms all other baselines across
most metrics, indicating its strong capability to generate reasonable and reliable explanations. In
particular, it obtains the lowest discrepancy (D) and lowest misleadingness (M) scores on both
datasets, suggesting that the explanations produced by G-Defense are highly consistent with the
ground-truth veracity labels. This is especially important, as a key goal of explainable fake news
detection is to assist the public in identifying the truth, rather than misleading users through
incorrect or biased reasoning. Moreover, the strong correlation between the objective discrepancy
scores and the LLM-assessed misleadingness scores further validates the reliability of our evaluation
methodology.
As for other baselines, as the best traditional approach, CofCED achieves the worst results on all
LLM-evaluated metrics. Since its explanations are discrete sentences extracted from reports, which
often contain redundant information and lack coherence, making them less human-friendly. This
incoherent explanation leads to a high misleadingness score, though the objective discrepancy score
is much better. And the other three worst metrics also underscore the importance of coherence
in generating a streamlined and understandable explanation. In terms of two baselines based on
GPT-3.5, GPT-3.5 fullgenerally outperforms GPT-3.5 claim on RAWFC while underperforming it on
LIAR-RAW. This discrepancy may result from the larger number of candidate evidences in RAWFC,
which provides more information to support explanation generation.
Compared with both versions of L-Defense, our proposed G-Defense demonstrates consistent
improvements on almost all metrics. By leveraging more fine-grained sub-claims, along with their
corresponding veracity assessments and explanations, G-Defense is able to deliver richer and more
informative explanations of the claim’s veracity. This is reflected in the higher information scores (I)
compared to L-Defense. And notably, it even outperforms the Oracle baseline on RAWFC. Moreover,
this fine-grained explanation strategy contributes to improvements not only in informativeness
but also in soundness (S) and readability (R). The decomposition into sub-claims allows the model
to analyze specific aspects of the overall claim, resulting in explanations that are logically more
robust and easier for readers to follow. Overall, the results suggest that breaking down complex
claims into smaller, verifiable units is not only beneficial for model prediction, but also crucial for
generating clear, trustworthy, and human-friendly explanations, which is a key requirement for
real-world fake news detection systems.
Human Evaluation.While LLM-based evaluation provides a scalable and cost-efficient way
to assess the quality of generated explanations, it cannot fully reflect how humans judge these
explanations. To complement the automatic assessment, we further conduct a human evaluation.
, Vol. 1, No. 1, Article . Publication date: April 2026.

A Graph-Enhanced Defense Framework for Explainable Fake News Detection with LLM 23
False Half TrueFalse
Half
True5 2 3
3 4 3
3 1 6With claim only
False Half TrueFalse
Half
True6 3 1
2 7 1
0 2 8With claim and explanation
(from L-Defense)
False Half TrueFalse
Half
True7 2 1
2 7 1
0 1 9With claim and explanation
(from G-Defense)
0246810
Fig. 8. Confusion matrices of judgment results from five annotators on 30 randomly sampled claims. The
three results are collected under three different settings: annotators are provided with only the claim, with the
claim and explanations generated by L-Defense, or with the claim and explanations generated by G-Defense.
The results from five annotators were averaged and rounded for visualization.
Specifically, we randomly sample 30 claims from the test set of RAWFC and engage five English-
speaking adults as annotators to assess the quality and helpfulness of the explanations. For the
explanations generated by G-Defense, we provide annotators with both the explanation graph and
the corresponding textual description. Additional reliability analyses of the human evaluation are
provided in Appendix C.
We present the results of human evaluation in Figure 7. Compared with the LLM-based evaluation
results on RAWFC shown in Table 4, the human ratings exhibit a generally consistent trend across
all methods, further validating the reliability of our LLM-based evaluation. CofCED ranks the lowest
on soundness and readability, likely due to the discrete and fragmented nature of its explanations,
which hampers human understanding. Without any external relevant reports provided, GPT-3.5 claim
performs the worst in terms of informativeness. And its performance is consistently inferior to
GPT-3.5 full, highlighting the importance of RAG in fake news detection. In contrast, both L-Defense
and G-Defense generally achieve higher ratings, suggesting that the defense-like inference strategy
contributes to generating higher-quality explanations. Notably, G-Defense outperforms all other
methods across all metrics except the Oracle, further demonstrating the effectiveness of fine-grained
explanations enabled by claim-centered graph construction.
In addition to assessing the quality of explanations, we ask the annotators to judge the veracity
of claims with and without explanation, and present the results in the form of confusion matrices
in Figure 8. This human-centered evaluation directly examines the usefulness and impact of the
generated explanations on real users’ decision-making. We observe that providing explanations
remarkably reduces judgment errors and helps annotators better identify the truth, confirming
the practical utility and trustworthiness of explanations in alleviating the propagation of fake
news. When comparing the two settings where explanations are provided, G-Defense is slightly
better than L-Defense. These results demonstrate that the intuitive and fine-grained explanations
produced by G-Defense enable annotators to better distinguish subtle differences between veracity
levels and have a measurable positive effect on human judgments.
4.6 Case Study
To qualitatively demonstrate the effectiveness of our proposed G-Defense framework, we present
three samples selected from the test set of RAWFC in Table 5. Each case consists of a claim, its
corresponding explanation graph, and the generated textual explanation.
Does the claim decomposition and graph construction help veracity prediction and provide a fine-
grained explanation?As shown in Table 5, all three claims are decomposed into at least two sub-
claims, with each sub-claim capturing a distinct aspect of the original claim. This decomposition
, Vol. 1, No. 1, Article . Publication date: April 2026.

24 B.Wang and J.Ma et al.
Table 5. Case study. The claims are selected from the test set of RAWFC. We provide both competing
explanations for analysis. In practice, only the explanation corresponding to the predicted label would be
presented to users, and the other one (shown in gray) is included here for analysis. The gold label of the claim
and the predicted labels of both the claim and its sub-claims are highlighted in color.
claim 1 (True): A newly discovered species of amphibian was named “Dermophis donaldtrumpi” as an unflattering reference to the
U.S. President.
𝑐𝑐!𝑐!"𝑐#𝑐#"
𝑐1(True): A newly discovered species of amphibian was named “Dermophis donaldtrumpi”.
𝑐+
1: The newly discovered species of amphibian was named “Dermophis donaldtrumpi” in recognition of the US
presidentś climate change denial.
𝑐−
1: The claim is false because there is no newly discovered species of amphibian named “Dermophis donaldtrumpi”
𝑐2(True): The name “Dermophis donaldtrumpi” was chosen as an unflattering reference to the U.S. President.
𝑐+
2: The name “Dermophis donaldtrumpi” was chosen as an unflattering reference to the U.S. President because
EnviroBuild, who paid for the naming rights, stated that they chose the name in recognition of the presidentś
position on climate change.
𝑐−
2: The name “Dermophis donaldtrumpi” was chosen as a reference to the U.S. Presidentś climate change policy.
Summary: The claim is reasoned as ‘true’ based on the supporting evidence provided by the true-oriented explanations for both
sub-claims, which establish a direct connection between the naming of the species and the U.S. President’s stance on climate change.
claim 2 (Half): The U.S. Congress passed the 22nd Amendment in order to “make sure President Franklin D. Roosevelt did not get
re-elected”.
𝑐𝑐!𝑐"𝑐"#𝑐!$
𝑐1(True): The U.S. Congress passed the 22nd Amendment.
𝑐+
1: The U.S. Congress passed the 22nd Amendment in 1947, which was later ratified by the states in 1951.
𝑐−
1: The 22nd Amendment was approved by Congress in 1947, with an exception that would exclude a president
in office from term limit during the ratification process.
𝑐2(False): The purpose of the 22nd Amendment was to prevent President Franklin D. Roosevelt from being
re-elected.
𝑐+
2: The 22nd Amendment was not intended to prevent President Franklin D. Roosevelt from being re-elected,
but rather to establish a term limit for all future presidents.
𝑐−
2: The 22nd Amendment was not intended to prevent President Franklin D. Roosevelt from being re-elected,
but rather to establish a term limit for all future presidents. FDR died in 1945. The amendment was passed in
1947 and ratified in 1951 — after his death.
Summary: The claim is reasoned as ‘half’ because while it is true that the U.S. Congress passed the 22nd Amendment, the assertion
that it was solely to prevent President Franklin D. Roosevelt from being re-elected is false. The actual purpose was to set a term limit
for all future presidents.
claim 3 (False): By ignoring a ban on performing COVID-19 autopsies, researchers in Italy discovered that the disease is caused by a
bacteria — not a virus — and can simply be treated with aspirin.
𝑐
𝑐!𝑐"𝑐#𝑐$𝑐!%𝑐"%𝑐#%𝑐$%
𝑐1(False): There was a ban in Italy on performing autopsies on COVID-19 victims.
𝑐+
1: Early in the pandemic, some countries, including Italy, issued temporary restrictions or guidelines discouraging
autopsies due to safety concerns.
𝑐−
1: It is false because there was no absolute ban. Autopsies were conducted under strict biosafety protocols. The
claim exaggerates the extent of any restriction.
𝑐2(False): Researchers ignored the ban and performed unauthorized autopsies.
𝑐+
2: The claim is reasoned as true because there is evidence of researchers evaluating hospitalized COVID-19
patients to determine the risk of death from COVID-19.
𝑐−
2: It is false because no evidence supports the claim that any autopsies were illegal or conducted in defiance of
restrictions.
𝑐3(False): The autopsies revealed that COVID-19 is caused by bacteria, not a virus.
𝑐+
3: It is true because researchers misunderstood co-infections or inflammation due to secondary bacterial
infections, not the cause of COVID-19.
𝑐−
3: The claim is false because multiple studies mentioned in the sentences indicate that COVID-19 is caused by a
virus, not a bacteria.
𝑐4(False): COVID-19 can be easily treated with aspirin.
𝑐+
4: Aspirin use may have a beneficial effect in COVID-19 patients, potentially reducing mortality.
𝑐−
4: The claim is false because there is no confirmed evidence that aspirin can treat COVID-19.
Summary: This claim reasoned as ‘false’. While there were limited restrictions on autopsies in Italy early in the pandemic, there is no
evidence that researchers violated bans or discovered that COVID-19 is caused by bacteria.
enables more targeted reasoning and provides more comprehensive evidence. For example, in
claim 1, the true-oriented explanation 𝑐+
2for sub-claim 𝑐2introduces additional knowledge that
EnviroBuild paid for the naming rights as a reference to the U.S. President’s stance on climate
change. This reveals the potential motivation behind the naming, providing more details to support
the claim as true. In claim 2, the two sub-claims are evaluated using different historical facts:
the exact year the 22nd Amendment was passed and ratified, and the death date of Franklin D.
, Vol. 1, No. 1, Article . Publication date: April 2026.

A Graph-Enhanced Defense Framework for Explainable Fake News Detection with LLM 25
Roosevelt. This decomposition enables the model to identify that, although the amendment itself
is factual, the implication of the claim is misleading. As a result, the model assigns a ‘half’ label.
In addition to improving veracity prediction, the explanation graphs provide an intuitive visual
format that reflects the internal reasoning process. They clearly display the veracity of sub-claims,
their supporting or refuting explanations, and how they influence the final decision based on the
dependency relationships. This transparent reasoning path enhances explainability, and reduces
the reading cost for users if they only want to know which part is false without caring about the
specific details. These samples demonstrate that claim decomposition and graph construction are
crucial for both accurate veracity prediction and intuitive explanation generation.
How does the defense-like inference strategy improve the performance of fake news detection?Our
method generates a pair of competing explanations for each sub-claim. By implicitly comparing the
informativeness and confidence of two explanations, the model can make more accurate inferences
than relying on one-sided explanations. On the one hand, explanations that do not align with
the true label often lack strong reasoning or supporting evidence. For instance, in claim 1, the
false-oriented explanation 𝑐−
1for sub-claim 𝑐1is very shallow and does not contain any meaningful
information, especially when compared to its true-oriented explanation 𝑐+
1. Similarly, in claim
3,𝑐+
4offers an unconfident explanation, whereas 𝑐−
4is much more direct and informative. These
contrasts help the model identify which explanation is more credible, so that improving veracity
prediction. On the other hand, even incorrect explanations can sometimes indirectly reveal the
truth by exposing logical inconsistencies. For example, in claim 1, 𝑐−
2who aims to refute 𝑐2actually
supports it. Likewise, 𝑐+
2in claim 2 proves the 𝑐2as false. And 𝑐+
2and𝑐+
3in claim 3 fail to directly
explain why the sub-claims are true, but offer reasoning that indirectly demonstrates the sub-claims
are false. This may be because, without solid supporting evidence, LLMs struggle to fabricate
convincing yet incorrect explanations. In summary, the defense-like inference strategy reduces
dependence on potentially inaccurate or biased single-perspective explanations. Instead, it shifts
the focus to comparative reasoning, encouraging the model to evaluate the quality of competing
explanations, thereby improving the performance of fake news detection.
4.7 Discussion
Although G-Defense involves multiple stages, it is inherently robust to error propagation. In the
graph construction module, we carefully design the prompt to encourage the generation of all
relevant sub-claims. To ensure minimal fine-grained reasoning capability, we require that each
claim be decomposed into at least two sub-claims. And we introduce an additional safeguard to
enhance graph connectivity by explicitly adding edges between each sub-claim and the original
claim. This ensures that even if the LLM fails to capture all dependencies among sub-claims and
claims, the resulting graph remains connected, preventing the presence of isolated nodes that
could compromise downstream reasoning. In the competing explanation generation module, as
discussed in §4.6, the LLM may sometimes struggle to generate a convincing explanation supporting
an incorrect veracity label due to the lack of sufficient evidence. This limitation actually helps
veracity prediction, as poor-quality explanations for incorrect labels naturally guide the inference
module toward more accurate reasoning. Moreover, the subsequent defense-like inference (§3.3) is
performed jointly over the entire claim-centered graph rather than on isolated sub-claims. This
reasoning mechanism naturally reduces the impact of occasional errors in graph construction, as
the model integrates information across multiple sub-claims to make robust veracity assessments.
Finally, in the explanation summarization module, the LLM is prompted to assess the veracity of
, Vol. 1, No. 1, Article . Publication date: April 2026.

26 B.Wang and J.Ma et al.
Table 6. Token usage (in millions) and estimated cost of G-Defense on LIAR-RAW and RAWFC across all
LLM-driven stages.
LIAR-RAW
train eval test sum avg per claim
in out in out in out in out in out
claim decomposition 0.98 0.54 0.12 0.07 0.12 0.07 1.22 0.68 0.000097 0.000054
edge generation 2.11 0.93 0.27 0.12 0.26 0.12 2.64 1.16 0.00021 0.000092
explanation generation 35.33 2.74 4.46 0.34 4.42 0.34 44.22 3.42 0.003512 0.000272
final explanation generation - - - - 1.11 0.38 1.11 0.38 0.000889 0.000301
price $19.21 $6.32 $2.43 $0.80 $2.80 $1.37 $24.60 $8.46 $0.0024 $0.0011
RAWFC
claim decomposition 0.16 0.09 0.02 0.01 0.02 0.01 0.2 0.12 0.0001 0.000058
edge generation 0.34 0.15 0.04 0.02 0.04 0.02 0.43 0.19 0.000217 0.000095
explanation generation 6.24 0.45 0.78 0.06 0.78 0.05 7.8 0.56 0.003924 0.000281
final explanation generation - - - - 0.18 0.06 0.18 0.06 0.0009 0.000301
price $3.37 $1.035 $0.42 $0.14 $0.51 $0.21 $4.31 $1.40 $0.0026 $0.0011
each sub-claim. Even if the final claim prediction is incorrect, the veracity assessments of individual
sub-claims still provide valuable insights, offering a fine-grained explanation that helps users better
understand the underlying truthfulness of different aspects of the claim.
4.8 Cost and Latency Analysis
We estimate the API cost of G-Defense by counting the tokens consumed at each LLM-based stage.
The framework relies on an LLM for four steps: claim decomposition, edge generation, competing
explanation generation, and final explanation generation. In experiments, we use “gpt-3.5-turbo-
0125” as the default LLM backbone, priced at $0.50 per 1M input tokens and $1.50 per 1M output
tokens8. Table 6 reports the aggregated token usage and corresponding cost on both datasets. The
total cost of running G-Defense on LIAR-RAW is approximately $33, and the cost on RAWFC is
about $6. On average, after G-Defense is fully trained, the detection cost for a claim is approximately
$0.004. These results show that G-Defense achieves strong performance with a relatively low cost,
making it feasible for practical use even at large scales. In addition, the results of G-Defense LLaMA3.1
in Table 2 demonstrate that G-Defense can also be effectively supported by an open-source LLM
backbone, rather than relying solely on closed-source models.
In addition to API cost, the inference latency of G-Defense is mainly determined by the multi-stage
API interaction, evidence retrieval, and final prediction. The overall latency can be approximately
expressed as 𝑇total≈𝑇dec+𝑇rel+𝑛(𝑇 ret+𝑇comp)+𝑇 pred+𝑇final, where𝑛is the number of decomposed
sub-claims, and 𝑇dec,𝑇rel,𝑇ret,𝑇comp,𝑇pred, and𝑇finaldenote the latency of claim decomposition,
relation generation, evidence retrieval, competing explanation generation, final prediction, and final
explanation generation, respectively. Based on statistics over the RAWFC and LIAR-RAW datasets,
the four API-based stages of G-Defense require around 4,000 input tokens and 430 output tokens
in total for each claim on average. In addition, each sub-claim also requires evidence retrieval by
matching candidate evidence based on cosine similarity, and the final veracity label is predicted by
Flan-T5-large using all collected information. The end-to-end inference latency mainly grows with
the number of decomposed sub-claims, since both evidence retrieval and competing explanation
generation are performed at the sub-claim level. Nevertheless, the overall inference time remains
acceptable for practical explainable veracity assessment. In this sense, G-Defense is better suited
8https://platform.openai.com/docs/pricing
, Vol. 1, No. 1, Article . Publication date: April 2026.

A Graph-Enhanced Defense Framework for Explainable Fake News Detection with LLM 27
to application scenarios that emphasize prediction accuracy and explanation quality, rather than
highly latency-sensitive real-time response. For larger-scale deployment, the latency may be further
reduced by parallelizing sub-claim-level processing, caching retrieved evidence, limiting the number
of decomposed sub-claims or candidate evidence, and replacing API-based steps with local models.
5 Conclusion and Future Work
In this paper, we propose G-Defense, a novel graph-enhanced defense framework for explainable
fake news detection with LLM. The framework first decomposes a news claim into several relevant
sub-claims and then constructs a claim-centered graph to capture their dependency relationships.
Then, it conducts evidence retrieval for each sub-claim and generates competing explanations for
two competing veracity labels. After that, to infer the final veracity, G-Defense performs a defense-
like inference over the claim-centered graph supplemented with competing explanations, comparing
the quality of the competing explanations and analyzing the rich dependency relationships. Finally,
an LLM is prompted with the graph and prediction result to generate a fine-grained explanation
graph and a summarized textual explanation. Experimental results on two real-world datasets
demonstrate that our proposed G-Defense achieves state-of-the-art performance in both veracity
prediction and explanation quality. Extensive ablation study further verifies the effectiveness of each
component in G-Defense. Human evaluation and case study further validate that the explanations
generated by our framework are helpful and intuitive for identifying fake news.
In the future, we plan to broaden the applicability of G-Defense by evaluating it on other types of
textual claims and potential evidence beyond datasets collected from social media platforms, such
as scientific, financial, and health-related misinformation. This will help us further examine the
framework’s robustness and generalizability across different domains and evidence sources. We also
plan to extend G-Defense to support multi-modal fake news detection, incorporating images and
videos that are increasingly common in real-world misinformation. Another promising direction is
to explore post-trained graph-aware LLMs. Recent graph foundation models [ 33,60,69] pretrained
on large-scale graph datasets can further improve the structural reasoning of LLMs beyond the
format of graph-to-text translation. These models are designed for classical graph tasks and do not
directly apply to our setting, where nodes are natural-language sub-claims and edges represent
logical dependencies. Nevertheless, adapting graph-aware pretraining to claim-centered graphs
may further strengthen G-Defense.
Acknowledgments
The authors would like to thank the anonymous reviewers for their valuable comments. This work is
supported by the National Key Research and Development Program of China (No.2023YFF0905400),
the National Natural Science Foundation of China (No.625B2080, No.62206233, No.U2341229 and
No.62406125), the Reform Commission Foundation of Jilin Province (No.2024C003), the Fun-
damental and Interdisciplinary Disciplines Breakthrough Plan of the Ministry of Education of
China (JYB2025XDXM903), the New Cornerstone Science Foundation through the XPLORER
PRIZE, and the Guangdong Basic and Applied Basic Research Foundation (No. 2026A1515011829,
No.2024A1515140144).
References
[1]Tariq Alhindi, Savvas Petridis, and Smaranda Muresan. 2018. Where is your evidence: Improving fact-checking by
justification modeling. In Proceedings ofthefirst workshop onfactextraction andverification (FEVER). 85–90.
[2]Jennifer Allen, Antonio A Arechar, Gordon Pennycook, and David G Rand. 2021. Scaling up fact-checking using the
wisdom of crowds. Science advances 7, 36 (2021), eabf4393.
, Vol. 1, No. 1, Article . Publication date: April 2026.

28 B.Wang and J.Ma et al.
[3]Jacopo Amidei, Paul Piwek, and Alistair Willis. 2018. Rethinking the agreement in human evaluation tasks. In
Proceedings ofthe27th International Conference onComputational Linguistics. 3318–3329.
[4]Pepa Atanasova, Jakob Grue Simonsen, Christina Lioma, and Isabelle Augenstein. 2020. Generating Fact Checking
Explanations. In Proceedings ofthe58th Annual Meeting oftheAssociation forComputational Linguistics, ACL
2020, Online, July 5-10, 2020, Dan Jurafsky, Joyce Chai, Natalie Schluter, and Joel R. Tetreault (Eds.). Association for
Computational Linguistics, 7352–7364. https://doi.org/10.18653/v1/2020.acl-main.656
[5]Yejin Bang, Samuel Cahyawijaya, Nayeon Lee, Wenliang Dai, Dan Su, Bryan Wilie, Holy Lovenia, Ziwei Ji, Tiezheng
Yu, Willy Chung, Quyet V. Do, Yan Xu, and Pascale Fung. 2023. A Multitask, Multilingual, Multimodal Evaluation of
ChatGPT on Reasoning, Hallucination, and Interactivity. In Proceedings ofthe13th International Joint Conference on
Natural Language Processing andthe3rdConference oftheAsia-Pacific Chapter oftheAssociation forComputational
Linguistics, IJCNLP 2023 -Volume 1:Long Papers, Nusa Dua, Bali, November 1-4,2023, Jong C. Park, Yuki Arase,
Baotian Hu, Wei Lu, Derry Wijaya, Ayu Purwarianti, and Adila Alfa Krisnadhi (Eds.). Association for Computational
Linguistics, 675–718. https://doi.org/10.18653/V1/2023.IJCNLP-MAIN.45
[6]Rishi Bommasani, Drew A. Hudson, Ehsan Adeli, Russ B. Altman, Simran Arora, Sydney von Arx, Michael S. Bernstein,
Jeannette Bohg, Antoine Bosselut, Emma Brunskill, Erik Brynjolfsson, Shyamal Buch, Dallas Card, Rodrigo Castellon,
Niladri S. Chatterji, Annie S. Chen, Kathleen Creel, Jared Quincy Davis, Dorottya Demszky, Chris Donahue, Moussa
Doumbouya, Esin Durmus, Stefano Ermon, John Etchemendy, Kawin Ethayarajh, Li Fei-Fei, Chelsea Finn, Trevor
Gale, Lauren E. Gillespie, Karan Goel, Noah D. Goodman, Shelby Grossman, Neel Guha, Tatsunori Hashimoto, Peter
Henderson, John Hewitt, Daniel E. Ho, Jenny Hong, Kyle Hsu, Jing Huang, Thomas Icard, Saahil Jain, Dan Jurafsky,
Pratyusha Kalluri, Siddharth Karamcheti, Geoff Keeling, Fereshte Khani, Omar Khattab, Pang Wei Koh, Mark S. Krass,
Ranjay Krishna, Rohith Kuditipudi, and et al. 2021. On the Opportunities and Risks of Foundation Models. CoRR
abs/2108.07258 (2021). arXiv:2108.07258 https://arxiv.org/abs/2108.07258
[7]Carlos Castillo, Marcelo Mendoza, and Barbara Poblete. 2011. Information credibility on twitter. In Proceedings ofthe
20th international conference onWorld wide web. 675–684.
[8]Yupeng Chang, Xu Wang, Jindong Wang, Yuan Wu, Kaijie Zhu, Hao Chen, Linyi Yang, Xiaoyuan Yi, Cunxiang Wang,
Yidong Wang, Wei Ye, Yue Zhang, Yi Chang, Philip S. Yu, Qiang Yang, and Xing Xie. 2023. A Survey on Evaluation of
Large Language Models. CoRR abs/2307.03109 (2023). https://doi.org/10.48550/arXiv.2307.03109 arXiv:2307.03109
[9] Harrison Chase. 2022. LangChain. (2022). https://github.com/langchain-ai/langchain
[10] Jifan Chen, Grace Kim, Aniruddh Sriram, Greg Durrett, and Eunsol Choi. 2024. Complex Claim Verification with Evi-
dence Retrieved in the Wild. In Proceedings ofthe2024 Conference oftheNorth American Chapter oftheAssociation
forComputational Linguistics: Human Language Technologies (Volume 1:Long Papers), NAACL 2024, Mexico City,
Mexico, June 16-21, 2024 , Kevin Duh, Helena Gómez-Adorno, and Steven Bethard (Eds.). Association for Computational
Linguistics, 3569–3587. https://doi.org/10.18653/V1/2024.NAACL-LONG.196
[11] Jifan Chen, Aniruddh Sriram, Eunsol Choi, and Greg Durrett. 2022. Generating Literal and Implied Subquestions
to Fact-check Complex Claims. In Proceedings ofthe2022 Conference onEmpirical Methods inNatural Language
Processing, EMNLP 2022, Abu Dhabi, United Arab Emirates, December 7-11, 2022 , Yoav Goldberg, Zornitsa Kozareva,
and Yue Zhang (Eds.). Association for Computational Linguistics, 3495–3516. https://doi.org/10.18653/V1/2022.EMNLP-
MAIN.229
[12] Yi Chen, Rui Wang, Haiyun Jiang, Shuming Shi, and Ruifeng Xu. 2023. Exploring the Use of Large Language
Models for Reference-Free Text Quality Evaluation: A Preliminary Empirical Study. CoRR abs/2304.00723 (2023).
https://doi.org/10.48550/arXiv.2304.00723 arXiv:2304.00723
[13] Zhikai Chen, Haitao Mao, Hang Li, Wei Jin, Hongzhi Wen, Xiaochi Wei, Shuaiqiang Wang, Dawei Yin, Wenqi Fan, Hui
Liu, and Jiliang Tang. 2023. Exploring the Potential of Large Language Models (LLMs)in Learning on Graphs. SIGKDD
Explor. 25, 2 (2023), 42–61. https://doi.org/10.1145/3655103.3655110
[14] Tsun-Hin Cheung and Kin-Man Lam. 2023. FactLLaMA: Optimizing Instruction-Following Language Models with
External Knowledge for Automated Fact-Checking. CoRR abs/2309.00240 (2023). https://doi.org/10.48550/arXiv.2309.
00240 arXiv:2309.00240
[15] David Cheng-Han Chiang and Hung-yi Lee. 2023. Can Large Language Models Be an Alternative to Human Evalua-
tions?. In Proceedings ofthe61st Annual Meeting oftheAssociation forComputational Linguistics (Volume 1:Long
Papers), ACL 2023, Toronto, Canada, July 9-14, 2023 , Anna Rogers, Jordan L. Boyd-Graber, and Naoaki Okazaki (Eds.).
Association for Computational Linguistics, 15607–15631. https://doi.org/10.18653/v1/2023.acl-long.870
[16] Hyung Won Chung, Le Hou, Shayne Longpre, Barret Zoph, Yi Tay, William Fedus, Yunxuan Li, Xuezhi Wang,
Mostafa Dehghani, Siddhartha Brahma, Albert Webson, Shixiang Shane Gu, Zhuyun Dai, Mirac Suzgun, Xinyun Chen,
Aakanksha Chowdhery, Alex Castro-Ros, Marie Pellat, Kevin Robinson, Dasha Valter, Sharan Narang, Gaurav Mishra,
Adams Yu, Vincent Y. Zhao, Yanping Huang, Andrew M. Dai, Hongkun Yu, Slav Petrov, Ed H. Chi, Jeff Dean, Jacob
Devlin, Adam Roberts, Denny Zhou, Quoc V. Le, and Jason Wei. 2024. Scaling Instruction-Finetuned Language Models.
J.Mach. Learn. Res. 25 (2024), 70:1–70:53. https://jmlr.org/papers/v25/23-0870.html
, Vol. 1, No. 1, Article . Publication date: April 2026.

A Graph-Enhanced Defense Framework for Explainable Fake News Detection with LLM 29
[17] Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil
Mathur, Alan Schelten, Amy Yang, Angela Fan, et al .2024. The llama 3 herd of models. arXiv preprint arXiv:2407.21783
(2024).
[18] Wenqi Fan, Yujuan Ding, Liangbo Ning, Shijie Wang, Hengyun Li, Dawei Yin, Tat-Seng Chua, and Qing Li. 2024. A
survey on rag meeting llms: Towards retrieval-augmented large language models. In Proceedings ofthe30th ACM
SIGKDD Conference onKnowledge Discovery andData Mining. 6491–6501.
[19] Bahare Fatemi, Jonathan Halcrow, and Bryan Perozzi. 2024. Talk like a Graph: Encoding Graphs for Large Language
Models. In TheTwelfth International Conference onLearning Representations, ICLR 2024, Vienna, Austria, May 7-11,
2024. OpenReview.net. https://openreview.net/forum?id=IuXR1CCrSi
[20] Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang Jia, Jinliu Pan, Yuxi Bi, Yi Dai, Jiawei Sun, Qianyu Guo, Meng Wang,
and Haofen Wang. 2023. Retrieval-Augmented Generation for Large Language Models: A Survey. CoRR abs/2312.10997
(2023). https://doi.org/10.48550/ARXIV.2312.10997 arXiv:2312.10997
[21] Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shirong Ma, Peiyi Wang,
Xiao Bi, et al .2025. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint
arXiv:2501.12948 (2025).
[22] Kelvin Guu, Kenton Lee, Zora Tung, Panupong Pasupat, and Ming-Wei Chang. 2020. Retrieval Augmented Language
Model Pre-Training. In Proceedings ofthe37th International Conference onMachine Learning, ICML 2020, 13-18 July
2020, Virtual Event (Proceedings of Machine Learning Research, Vol.119). PMLR, 3929–3938. http://proceedings.mlr.
press/v119/guu20a.html
[23] Beizhe Hu, Qiang Sheng, Juan Cao, Yuhui Shi, Yang Li, Danding Wang, and Peng Qi. 2024. Bad Actor, Good Advi-
sor: Exploring the Role of Large Language Models in Fake News Detection. In Thirty-Eighth AAAI Conference on
Artificial Intelligence, AAAI 2024, Thirty-Sixth Conference onInnovative Applications ofArtificial Intelligence, IAAI
2024, Fourteenth Symposium onEducational Advances inArtificial Intelligence, EAAI 2014, February 20-27, 2024,
Vancouver, Canada , Michael J. Wooldridge, Jennifer G. Dy, and Sriraam Natarajan (Eds.). AAAI Press, 22105–22113.
https://doi.org/10.1609/AAAI.V38I20.30214
[24] Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen.
2022. LoRA: Low-Rank Adaptation of Large Language Models. In TheTenth International Conference onLearning
Representations, ICLR 2022, Virtual Event, April 25-29, 2022. OpenReview.net. https://openreview.net/forum?id=
nZeVKeeFYf9
[25] Qisheng Hu, Quanyu Long, and Wenya Wang. 2024. Decomposition Dilemmas: Does Claim Decomposition Boost or
Burden Fact-Checking Performance? arXiv:2411.02400 [cs.IR] https://arxiv.org/abs/2411.02400
[26] Fan Huang, Haewoon Kwak, and Jisun An. 2023. Is ChatGPT better than Human Annotators? Potential and Limitations
of ChatGPT in Explaining Implicit Hate Speech. In Companion Proceedings oftheACM Web Conference 2023, WWW
2023, Austin, TX,USA, 30April 2023 -4May 2023, Ying Ding, Jie Tang, Juan F. Sequeda, Lora Aroyo, Carlos Castillo,
and Geert-Jan Houben (Eds.). ACM, 294–297. https://doi.org/10.1145/3543873.3587368
[27] Jin Huang, Xingjian Zhang, Qiaozhu Mei, and Jiaqi Ma. 2023. Can LLMs Effectively Leverage Graph Structural Infor-
mation: When and Why. CoRR abs/2309.16595 (2023). https://doi.org/10.48550/ARXIV.2309.16595 arXiv:2309.16595
[28] Lei Huang, Weijiang Yu, Weitao Ma, Weihong Zhong, Zhangyin Feng, Haotian Wang, Qianglong Chen, Weihua Peng,
Xiaocheng Feng, Bing Qin, et al .2025. A survey on hallucination in large language models: Principles, taxonomy,
challenges, and open questions. ACM Transactions onInformation Systems 43, 2 (2025), 1–55.
[29] Yizheng Huang and Jimmy Huang. 2024. A Survey on Retrieval-Augmented Text Generation for Large Language
Models. CoRR abs/2404.10981 (2024). https://doi.org/10.48550/ARXIV.2404.10981 arXiv:2404.10981
[30] Gautier Izacard, Patrick S. H. Lewis, Maria Lomeli, Lucas Hosseini, Fabio Petroni, Timo Schick, Jane Dwivedi-Yu,
Armand Joulin, Sebastian Riedel, and Edouard Grave. 2023. Atlas: Few-shot Learning with Retrieval Augmented
Language Models. J.Mach. Learn. Res. 24 (2023), 251:1–251:43. https://jmlr.org/papers/v24/23-0037.html
[31] Ziwei Ji, Nayeon Lee, Rita Frieske, Tiezheng Yu, Dan Su, Yan Xu, Etsuko Ishii, Yejin Bang, Andrea Madotto, and Pascale
Fung. 2023. Survey of Hallucination in Natural Language Generation. ACM Comput. Surv. 55, 12 (2023), 248:1–248:38.
https://doi.org/10.1145/3571730
[32] Bowen Jin, Gang Liu, Chi Han, Meng Jiang, Heng Ji, and Jiawei Han. 2024. Large Language Models on Graphs: A
Comprehensive Survey. IEEE Trans. Knowl. Data Eng. 36, 12 (2024), 8622–8642. https://doi.org/10.1109/TKDE.2024.
3469578
[33] Bowen Jin, Gang Liu, Chi Han, Meng Jiang, Heng Ji, and Jiawei Han. 2024. Large language models on graphs: A
comprehensive survey. IEEE Transactions onKnowledge andData Engineering (2024).
[34] Zhiwei Jin, Juan Cao, Yongdong Zhang, and Jiebo Luo. 2016. News verification by exploiting conflicting social
viewpoints in microblogs. In Proceedings oftheAAAI conference onartificial intelligence, Vol. 30.
[35] Neema Kotonya and Francesca Toni. 2020. Explainable Automated Fact-Checking: A Survey. In Proceedings ofthe28th
International Conference onComputational Linguistics, COLING 2020, Barcelona, Spain (Online), December 8-13,
, Vol. 1, No. 1, Article . Publication date: April 2026.

30 B.Wang and J.Ma et al.
2020, Donia Scott, Núria Bel, and Chengqing Zong (Eds.). International Committee on Computational Linguistics,
5430–5443. https://doi.org/10.18653/v1/2020.coling-main.474
[36] Neema Kotonya and Francesca Toni. 2020. Explainable Automated Fact-Checking for Public Health Claims. In
Proceedings ofthe2020 Conference onEmpirical Methods inNatural Language Processing, EMNLP 2020, Online,
November 16-20, 2020, Bonnie Webber, Trevor Cohn, Yulan He, and Yang Liu (Eds.). Association for Computational
Linguistics, 7740–7754. https://doi.org/10.18653/v1/2020.emnlp-main.623
[37] Patrick S. H. Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Hein-
rich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, Sebastian Riedel, and Douwe Kiela. 2020. Retrieval-
Augmented Generation for Knowledge-Intensive NLP Tasks. In Advances inNeural Information Processing Systems
33:Annual Conference onNeural Information Processing Systems 2020, NeurIPS 2020, December 6-12, 2020, virtual ,
Hugo Larochelle, Marc’Aurelio Ranzato, Raia Hadsell, Maria-Florina Balcan, and Hsuan-Tien Lin (Eds.). https:
//proceedings.neurips.cc/paper/2020/hash/6b493230205f780e1bc26945df7481e5-Abstract.html
[38] Jun Li, Yi Bin, Yunshan Ma, Yang Yang, Zi Huang, and Tat-Seng Chua. 2024. Filter-based Stance Network for Rumor
Verification. ACM Trans. Inf.Syst. 42, 4 (2024), 108:1–108:28. https://doi.org/10.1145/3649462
[39] Hao Liao, Jiahao Peng, Zhanyi Huang, Wei Zhang, Guanghua Li, Kai Shu, and Xing Xie. 2023. MUSER: A MUlti-Step
Evidence Retrieval Enhancement Framework for Fake News Detection. In Proceedings ofthe29th ACM SIGKDD
Conference onKnowledge Discovery andData Mining, KDD 2023, Long Beach, CA,USA, August 6-10, 2023 , Ambuj K.
Singh, Yizhou Sun, Leman Akoglu, Dimitrios Gunopulos, Xifeng Yan, Ravi Kumar, Fatma Ozcan, and Jieping Ye (Eds.).
ACM, 4461–4472. https://doi.org/10.1145/3580305.3599873
[40] Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du, Mandar Joshi, Danqi Chen, Omer Levy, Mike Lewis, Luke Zettlemoyer,
and Veselin Stoyanov. 2019. RoBERTa: A Robustly Optimized BERT Pretraining Approach. CoRR abs/1907.11692
(2019). arXiv:1907.11692 http://arxiv.org/abs/1907.11692
[41] Yang Liu and Yi-fang Brook Wu. 2020. FNED: A Deep Network for Fake News Early Detection on Social Media. ACM
Trans. Inf.Syst. 38, 3 (2020), 25:1–25:33. https://doi.org/10.1145/3386253
[42] Yu Lu, Junwei Bao, Yan Song, Zichen Ma, Shuguang Cui, Youzheng Wu, and Xiaodong He. 2021. RevCore:
Review-Augmented Conversational Recommendation. In Findings oftheAssociation forComputational Linguistics:
ACL/IJCNLP 2021, Online Event, August 1-6,2021 (Findings of ACL, Vol.ACL/IJCNLP 2021) , Chengqing Zong, Fei
Xia, Wenjie Li, and Roberto Navigli (Eds.). Association for Computational Linguistics, 1161–1173. https://doi.org/10.
18653/V1/2021.FINDINGS-ACL.99
[43] Yi-Ju Lu and Cheng-Te Li. 2020. GCAN: Graph-aware Co-Attention Networks for Explainable Fake News Detection
on Social Media. In Proceedings ofthe58th Annual Meeting oftheAssociation forComputational Linguistics, ACL
2020, Online, July 5-10, 2020, Dan Jurafsky, Joyce Chai, Natalie Schluter, and Joel R. Tetreault (Eds.). Association for
Computational Linguistics, 505–514. https://doi.org/10.18653/v1/2020.acl-main.48
[44] Jing Ma, Wei Gao, Shafiq R. Joty, and Kam-Fai Wong. 2019. Sentence-Level Evidence Embedding for Claim Verification
with Hierarchical Attention Networks. In Proceedings ofthe57th Conference oftheAssociation forComputational
Linguistics, ACL 2019, Florence, Italy, July28-August 2,2019, Volume 1:Long Papers , Anna Korhonen, David R. Traum,
and Lluís Màrquez (Eds.). Association for Computational Linguistics, 2561–2571. https://doi.org/10.18653/v1/p19-1244
[45] Jing Ma, Wei Gao, and Kam-Fai Wong. 2018. Detect Rumor and Stance Jointly by Neural Multi-task Learning. In
Companion oftheTheWeb Conference 2018 onTheWeb Conference 2018, WWW 2018, Lyon ,France, April 23-27,
2018, Pierre-Antoine Champin, Fabien Gandon, Mounia Lalmas, and Panagiotis G. Ipeirotis (Eds.). ACM, 585–593.
https://doi.org/10.1145/3184558.3188729
[46] Yuren Mao, Xuemei Dong, Wenyi Xu, Yunjun Gao, Bin Wei, and Ying Zhang. 2025. Fit-rag: Black-box rag with
factual information and token reduction. ACM Transactions onInformation Systems 43, 2 (2025), 1–27. https:
//doi.org/10.1145/367695
[47] Qiong Nan, Qiang Sheng, Juan Cao, Beizhe Hu, Danding Wang, and Jintao Li. 2024. Let Silence Speak: Enhancing
Fake News Detection with Generated Comments from Large Language Models. In Proceedings ofthe33rd ACM
International Conference onInformation andKnowledge Management, CIKM 2024, Boise, ID,USA, October 21-25,
2024, Edoardo Serra and Francesca Spezzano (Eds.). ACM, 1732–1742. https://doi.org/10.1145/3627673.3679519
[48] Yixin Nie, Haonan Chen, and Mohit Bansal. 2019. Combining Fact Extraction and Verification with Neural Semantic
Matching Networks. In The Thirty-Third AAAI Conference onArtificial Intelligence, AAAI 2019, The Thirty-First
Innovative Applications ofArtificial Intelligence Conference, IAAI 2019, TheNinth AAAI Symposium onEducational
Advances inArtificial Intelligence, EAAI 2019, Honolulu, Hawaii, USA, January 27-February 1,2019. AAAI Press,
6859–6866. https://doi.org/10.1609/aaai.v33i01.33016859
[49] OpenAI. 2023. GPT-4 Technical Report. CoRR abs/2303.08774 (2023). https://doi.org/10.48550/arXiv.2303.08774
arXiv:2303.08774
[50] Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll L. Wainwright, Pamela Mishkin, Chong Zhang, Sand-
hini Agarwal, Katarina Slama, Alex Ray, John Schulman, Jacob Hilton, Fraser Kelton, Luke Miller, Maddie Simens,
, Vol. 1, No. 1, Article . Publication date: April 2026.

A Graph-Enhanced Defense Framework for Explainable Fake News Detection with LLM 31
Amanda Askell, Peter Welinder, Paul F. Christiano, Jan Leike, and Ryan Lowe. 2022. Training language mod-
els to follow instructions with human feedback. In NeurIPS . http://papers.nips.cc/paper_files/paper/2022/hash/
b1efde53be364a73914f58805a001731-Abstract-Conference.html
[51] Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll L. Wainwright, Pamela Mishkin, Chong Zhang, Sand-
hini Agarwal, Katarina Slama, Alex Ray, John Schulman, Jacob Hilton, Fraser Kelton, Luke Miller, Maddie Simens,
Amanda Askell, Peter Welinder, Paul F. Christiano, Jan Leike, and Ryan Lowe. 2022. Training language mod-
els to follow instructions with human feedback. In NeurIPS . http://papers.nips.cc/paper_files/paper/2022/hash/
b1efde53be364a73914f58805a001731-Abstract-Conference.html
[52] Kashyap Popat, Subhabrata Mukherjee, Andrew Yates, and Gerhard Weikum. 2018. DeClarE: Debunking Fake
News and False Claims using Evidence-Aware Deep Learning. In Proceedings ofthe2018 Conference onEmpirical
Methods inNatural Language Processing, Brussels, Belgium, October 31-November 4,2018, Ellen Riloff, David
Chiang, Julia Hockenmaier, and Jun’ichi Tsujii (Eds.). Association for Computational Linguistics, 22–32. https:
//doi.org/10.18653/v1/d18-1003
[53] Ori Ram, Yoav Levine, Itay Dalmedigos, Dor Muhlgay, Amnon Shashua, Kevin Leyton-Brown, and Yoav Shoham.
2023. In-Context Retrieval-Augmented Language Models. Trans. Assoc. Comput. Linguistics 11 (2023), 1316–1331.
https://doi.org/10.1162/TACL_A_00605
[54] Nils Reimers and Iryna Gurevych. 2019. Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks. In
Proceedings ofthe2019 Conference onEmpirical Methods inNatural Language Processing andthe9thInternational
Joint Conference onNatural Language Processing, EMNLP-IJCNLP 2019, Hong Kong, China, November 3-7,2019,
Kentaro Inui, Jing Jiang, Vincent Ng, and Xiaojun Wan (Eds.). Association for Computational Linguistics, 3980–3990.
https://doi.org/10.18653/v1/D19-1410
[55] Parker Riley, Daniel Deutsch, George Foster, Viresh Ratnakar, Ali Dabirmoghaddam, and Markus Freitag. 2024.
Finding replicable human evaluations via stable ranking probability. In Proceedings ofthe2024 Conference ofthe
North American Chapter oftheAssociation forComputational Linguistics: Human Language Technologies (Volume
1:Long Papers). 4908–4919.
[56] Victor Sanh, Lysandre Debut, Julien Chaumond, and Thomas Wolf. 2019. DistilBERT, a distilled version of BERT:
smaller, faster, cheaper and lighter. CoRR abs/1910.01108 (2019). arXiv:1910.01108 http://arxiv.org/abs/1910.01108
[57] Kai Shu, Limeng Cui, Suhang Wang, Dongwon Lee, and Huan Liu. 2019. dEFEND: Explainable Fake News Detection.
InProceedings ofthe25th ACM SIGKDD International Conference onKnowledge Discovery &Data Mining, KDD
2019, Anchorage, AK,USA, August 4-8,2019, Ankur Teredesai, Vipin Kumar, Ying Li, Rómer Rosales, Evimaria Terzi,
and George Karypis (Eds.). ACM, 395–405. https://doi.org/10.1145/3292500.3330935
[58] Miloš Stanojević and Khalil Sima’an. 2014. Beer: Better evaluation as ranking. In Proceedings oftheNinth Workshop
onStatistical Machine Translation. 414–419.
[59] Jinyan Su, Terry Yue Zhuo, Jonibek Mansurov, Di Wang, and Preslav Nakov. 2023. Fake News Detectors are Biased
against Texts Generated by Large Language Models. CoRR abs/2309.08674 (2023). https://doi.org/10.48550/arXiv.2309.
08674 arXiv:2309.08674
[60] Jiabin Tang, Yuhao Yang, Wei Wei, Lei Shi, Lixin Su, Suqi Cheng, Dawei Yin, and Chao Huang. 2024. Graphgpt: Graph
instruction tuning for large language models. In Proceedings ofthe47th International ACM SIGIR Conference on
Research andDevelopment inInformation Retrieval. 491–500.
[61] Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste
Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, Aurélien Rodriguez, Armand Joulin, Edouard Grave, and Guillaume
Lample. 2023. LLaMA: Open and Efficient Foundation Language Models. CoRR abs/2302.13971 (2023). https:
//doi.org/10.48550/arXiv.2302.13971 arXiv:2302.13971
[62] Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya
Batra, Prajjwal Bhargava, Shruti Bhosale, et al .2023. Llama 2: Open foundation and fine-tuned chat models. arXiv
preprint arXiv:2307.09288 (2023).
[63] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia
Polosukhin. 2017. Attention is all you need. Advances inneural information processing systems 30 (2017).
[64] Herun Wan, Shangbin Feng, Zhaoxuan Tan, Heng Wang, Yulia Tsvetkov, and Minnan Luo. 2024. DELL: Generating
Reactions and Explanations for LLM-Based Misinformation Detection. In Findings oftheAssociation forComputational
Linguistics, ACL 2024, Bangkok, Thailand andvirtual meeting, August 11-16, 2024, Lun-Wei Ku, Andre Martins, and
Vivek Srikumar (Eds.). Association for Computational Linguistics, 2637–2667. https://doi.org/10.18653/V1/2024.
FINDINGS-ACL.155
[65] Bo Wang, Jing Ma, Hongzhan Lin, Zhiwei Yang, Ruichao Yang, Yuan Tian, and Yi Chang. 2024. Explainable Fake News
Detection with Large Language Model via Defense Among Competing Wisdom. In Proceedings oftheACM onWeb
Conference 2024, WWW 2024, Singapore, May 13-17, 2024, Tat-Seng Chua, Chong-Wah Ngo, Ravi Kumar, Hady W.
Lauw, and Roy Ka-Wei Lee (Eds.). ACM, 2452–2463. https://doi.org/10.1145/3589334.3645471
, Vol. 1, No. 1, Article . Publication date: April 2026.

32 B.Wang and J.Ma et al.
[66] Han Wang, Ming Shan Hee, Md. Rabiul Awal, Kenny Tsu Wei Choo, and Roy Ka-Wei Lee. 2023. Evaluating GPT-3
Generated Explanations for Hateful Content Moderation. In Proceedings oftheThirty-Second International Joint
Conference onArtificial Intelligence, IJCAI 2023, 19th-25th August 2023, Macao, SAR, China . ijcai.org, 6255–6263.
https://doi.org/10.24963/ijcai.2023/694
[67] Haoran Wang and Kai Shu. 2023. Explainable Claim Verification via Knowledge-Grounded Reasoning with Large
Language Models. In Findings oftheAssociation forComputational Linguistics: EMNLP 2023, Singapore, December
6-10, 2023, Houda Bouamor, Juan Pino, and Kalika Bali (Eds.). Association for Computational Linguistics, 6288–6304.
https://doi.org/10.18653/V1/2023.FINDINGS-EMNLP.416
[68] Jinguang Wang, Shengsheng Qian, Jun Hu, Wenxiang Dong, Xudong Huang, and Richang Hong. 2025. End-to-End
Explainable Fake News Detection Via Evidence-Claim Variational Causal Inference. ACM Transactions onInformation
Systems (2025). https://doi.org/10.1145/3728462
[69] Zehong Wang, Zheyuan Liu, Tianyi Ma, Jiazheng Li, Zheyuan Zhang, Xingbo Fu, Yiyang Li, Zhengqing Yuan, Wei
Song, Yijun Ma, et al .2025. Graph Foundation Models: A Comprehensive Survey. arXiv preprint arXiv:2505.15116
(2025).
[70] Junda Wu, Cheng-Chun Chang, Tong Yu, Zhankui He, Jianing Wang, Yupeng Hou, and Julian J. McAuley. 2024. CoRAL:
Collaborative Retrieval-Augmented Large Language Models Improve Long-tail Recommendation. In Proceedings ofthe
30th ACM SIGKDD Conference onKnowledge Discovery andData Mining, KDD 2024, Barcelona, Spain, August 25-29,
2024, Ricardo Baeza-Yates and Francesco Bonchi (Eds.). ACM, 3391–3401. https://doi.org/10.1145/3637528.3671901
[71] Lianwei Wu, Yuan Rao, Ling Sun, and Wangbo He. 2021. Evidence Inference Networks for Interpretable Claim Verifica-
tion. In Thirty-Fifth AAAI Conference onArtificial Intelligence, AAAI 2021, Thirty-Third Conference onInnovative
Applications ofArtificial Intelligence, IAAI 2021, The Eleventh Symposium onEducational Advances inArtificial
Intelligence, EAAI 2021, Virtual Event, February 2-9,2021. AAAI Press, 14058–14066. https://doi.org/10.1609/aaai.
v35i16.17655
[72] Fan Yang, Shiva K. Pentyala, Sina Mohseni, Mengnan Du, Hao Yuan, Rhema Linder, Eric D. Ragan, Shuiwang Ji, and
Xia (Ben) Hu. 2019. XFake: Explainable Fake News Detector with Visualizations. In TheWorld Wide Web Conference,
WWW 2019, SanFrancisco, CA,USA, May 13-17, 2019, Ling Liu, Ryen W. White, Amin Mantrach, Fabrizio Silvestri,
Julian J. McAuley, Ricardo Baeza-Yates, and Leila Zia (Eds.). ACM, 3600–3604. https://doi.org/10.1145/3308558.3314119
[73] Ruichao Yang, Jing Ma, Hongzhan Lin, and Wei Gao. 2022. A Weakly Supervised Propagation Model for Rumor
Verification and Stance Detection with Multiple Instance Learning. In SIGIR ’22:The45th International ACM SIGIR
Conference onResearch andDevelopment inInformation Retrieval, Madrid, Spain, July11-15,2022 , Enrique Amigó,
Pablo Castells, Julio Gonzalo, Ben Carterette, J. Shane Culpepper, and Gabriella Kazai (Eds.). ACM, 1761–1772. https:
//doi.org/10.1145/3477495.3531930
[74] Zhiwei Yang, Jing Ma, Hechang Chen, Hongzhan Lin, Ziyang Luo, and Yi Chang. 2022. A Coarse-to-fine Cascaded
Evidence-Distillation Neural Network for Explainable Fake News Detection. In Proceedings ofthe29th International
Conference onComputational Linguistics, COLING 2022, Gyeongju, Republic ofKorea, October 12-17, 2022 , Nicoletta
Calzolari, Chu-Ren Huang, Hansaem Kim, James Pustejovsky, Leo Wanner, Key-Sun Choi, Pum-Mo Ryu, Hsin-Hsi
Chen, Lucia Donatelli, Heng Ji, Sadao Kurohashi, Patrizia Paggio, Nianwen Xue, Seokhwan Kim, Younggyun Hahm,
Zhong He, Tony Kyungil Lee, Enrico Santus, Francis Bond, and Seung-Hoon Na (Eds.). International Committee on
Computational Linguistics, 2608–2621. https://aclanthology.org/2022.coling-1.230
[75] Ruosong Ye, Caiqi Zhang, Runhui Wang, Shuyuan Xu, and Yongfeng Zhang. 2024. Language is All a Graph Needs. In
Findings oftheAssociation forComputational Linguistics: EACL 2024, St.Julian’s, Malta, March 17-22, 2024, Yvette
Graham and Matthew Purver (Eds.). Association for Computational Linguistics, 1955–1973. https://aclanthology.org/
2024.findings-eacl.132
[76] Zhenrui Yue, Huimin Zeng, Yimeng Lu, Lanyu Shang, Yang Zhang, and Dong Wang. 2024. Evidence-Driven Retrieval
Augmented Response Generation for Online Misinformation. In Proceedings ofthe2024 Conference oftheNorth
American Chapter oftheAssociation forComputational Linguistics: Human Language Technologies (Volume 1:Long
Papers), NAACL 2024, Mexico City, Mexico, June 16-21, 2024 , Kevin Duh, Helena Gómez-Adorno, and Steven Bethard
(Eds.). Association for Computational Linguistics, 5628–5643. https://doi.org/10.18653/V1/2024.NAACL-LONG.313
[77] Zhenrui Yue, Huimin Zeng, Lanyu Shang, Yifan Liu, Yang Zhang, and Dong Wang. 2024. Retrieval Augmented Fact
Verification by Synthesizing Contrastive Arguments. In Proceedings ofthe62nd Annual Meeting oftheAssociation
forComputational Linguistics (Volume 1:Long Papers), ACL 2024, Bangkok, Thailand, August 11-16, 2024, Lun-Wei
Ku, Andre Martins, and Vivek Srikumar (Eds.). Association for Computational Linguistics, 10331–10343. https:
//doi.org/10.18653/V1/2024.ACL-LONG.556
[78] Shenglai Zeng, Jiankun Zhang, Bingheng Li, Yuping Lin, Tianqi Zheng, Dante Everaert, Hanqing Lu, Hui Liu, Yue Xing,
Monica Xiao Cheng, and Jiliang Tang. 2024. Towards Knowledge Checking in Retrieval-augmented Generation: A
Representation Perspective. CoRR abs/2411.14572 (2024). https://doi.org/10.48550/ARXIV.2411.14572 arXiv:2411.14572
, Vol. 1, No. 1, Article . Publication date: April 2026.

A Graph-Enhanced Defense Framework for Explainable Fake News Detection with LLM 33
[79] Xuan Zhang and Wei Gao. 2023. Towards LLM-based Fact Verification on News Claims with a Hierarchical Step-by-
Step Prompting Method. In Proceedings ofthe13th International Joint Conference onNatural Language Processing
andthe3rdConference oftheAsia-Pacific Chapter oftheAssociation forComputational Linguistics, IJCNLP 2023
-Volume 1:Long Papers, Nusa Dua, Bali, November 1-4,2023, Jong C. Park, Yuki Arase, Baotian Hu, Wei Lu, Derry
Wijaya, Ayu Purwarianti, and Adila Alfa Krisnadhi (Eds.). Association for Computational Linguistics, 996–1011.
https://doi.org/10.18653/V1/2023.IJCNLP-MAIN.64
[80] Xuan Zhang and Wei Gao. 2024. Reinforcement Retrieval Leveraging Fine-grained Feedback for Fact Checking News
Claims with Black-Box LLM. In Proceedings ofthe2024 Joint International Conference onComputational Linguistics,
Language Resources andEvaluation, LREC/COLING 2024, 20-25 May, 2024, Torino, Italy, Nicoletta Calzolari, Min-Yen
Kan, Véronique Hoste, Alessandro Lenci, Sakriani Sakti, and Nianwen Xue (Eds.). ELRA and ICCL, 13861–13873.
https://aclanthology.org/2024.lrec-main.1209
[81] Yue Zhang, Yafu Li, Leyang Cui, Deng Cai, Lemao Liu, Tingchen Fu, Xinting Huang, Enbo Zhao, Yu Zhang, Yulong
Chen, Longyue Wang, Anh Tuan Luu, Wei Bi, Freda Shi, and Shuming Shi. 2023. Siren’s Song in the AI Ocean: A Survey
on Hallucination in Large Language Models. CoRR abs/2309.01219 (2023). https://doi.org/10.48550/ARXIV.2309.01219
arXiv:2309.01219
[82] Wayne Xin Zhao, Kun Zhou, Junyi Li, Tianyi Tang, Xiaolei Wang, Yupeng Hou, Yingqian Min, Beichen Zhang, Junjie
Zhang, Zican Dong, et al. 2023. A survey of large language models. arXiv preprint arXiv:2303.18223 (2023).
[83] Wayne Xin Zhao, Kun Zhou, Junyi Li, Tianyi Tang, Xiaolei Wang, Yupeng Hou, Yingqian Min, Beichen Zhang, Junjie
Zhang, Zican Dong, et al. 2023. A survey of large language models. arXiv preprint arXiv:2303.18223 1, 2 (2023).
A Prompt Designing
The prompts in our framework follow a simple and task-oriented design principle. Specifically, they
are mainly used to specify the task objective, the relevant information to consider, and the desired
output format. We intentionally avoid heavy prompt engineering or elaborate prompt tuning to
keep the framework simple and broadly applicable. The details of prompts introduced in §3 and §4
are listed below.
Prompt used for claim decomposition,𝐼𝑣(in §3.1)
You are a fake news detection assistant. Your primary function is that, given a complex
news claim, based on its content, please break it down into several sub-claims relevant to
the veracity of the claim. Note that directly generate sub-claims in short and clear manner,
avoiding redundancy.
# News claim:𝑐
Prompt used for edge generation,𝐼𝑒(in §3.1)
You are a graph construction assistant. Your primary function is that, given a complex news
claim (index 0), and several sub-claims (index starting from 1) decomposed from the news
claim, please use logical relationships to construct a reasoning graph that reflects how the
sub-claims contribute to the truth of the overall claim.
Note: Your output should be in a dictionary format with the keys: ("analysis", "edges"). You
should generate directed edges using the indexes of the claims, formatted as a list like [(1, 0),
(2, 3), (3, 0)].
# News claim: 0.𝑐
# Sub-claims: 1.𝑐 1; ...; n.𝑐𝑛.
, Vol. 1, No. 1, Article . Publication date: April 2026.

34 B.Wang and J.Ma et al.
Prompt used for final explanation summarization,𝐼𝑠, (in §3.4)
Given a claim, its corresponding claim-centered graph, and a veracity label 𝑦∗, please give me
the veracity prediction about each sub-claim, and a streamlined rationale associate with the
claim for how it is reasoned as𝑦∗.
# Claim-centered graph:𝐼𝑎(H ˜G,˜V)
Your response should be a Python dictionary with the following structure:
{
"sub-claims-veracity":
{
"sub-claim 1": {
"reasoning": "Your analysis about the veracity of sub-claim 1",
"prediction": "True/False"
},
...,
"sub-claim n": {
"reasoning": "Your analysis about the veracity of sub-claim n.",
"prediction": "True/False"
}
},
"final-explanation": ""
}
Prompt used for background generation in G-Defense background ,𝐼𝑏, (in §4.2)
You have been specially designed to perform objective contextual analysis for the fake news
detection task. Your primary function is that, according to a news claim and some sentences
related to it, please provide a streamlined contextual analysis that helps understand the
background, circumstances, or perspectives related to the claim. Your goal is to summarize
the key contextual information — such as background facts, timelines, participants, related
events, or uncertainties —that could help people interpret or understand the claim, without
expressing any stance on its truthfulness. Note: Do not repeat the claim itself, and do not imply
or indicate any truth judgment. Just directly output the short and clear contextual rationale.
Given a claim 𝑐𝑖and a set of retrieved relevant reports 𝑠1,𝑠2,..., please provide a short and clear
objective contextual analysis that presents factual background or explanatory information
helping to understand the claim, without expressing any stance on its veracity.
, Vol. 1, No. 1, Article . Publication date: April 2026.

A Graph-Enhanced Defense Framework for Explainable Fake News Detection with LLM 35
Prompt used for hyperedge generation in G-Defense hyper,𝐼ℎ(in §4.2)
You are a hypergraph construction assistant. Your primary function is that, given a complex
news claim (index 0), and several sub-claims (index starting from 1) decomposed from the
news claim, please use topic relationship to construct a reasoning graph that reflects how the
sub-claims contribute to the truth of the overall claim. A hyperedge represents a higher-level
topic or reasoning unit that connects multiple related claims (e.g., several sub-claims that
jointly address one aspect of the main claim). Unlike simple pairwise edges, each hyperedge
can include more than two nodes to capture shared themes or collective reasoning.
Note: Your output should be in a dictionary format with the keys: ("analysis", "hyperedges").
You should generate hyperedges using the indexes of the claims, formatted as a list of lists
like [[1, 2, 0], [3, 4, 5, 0], [5, 6, 0]].
# News claim: 0.𝑐
# Sub-claims: 1.𝑐 1; ...; n.𝑐𝑛.
, Vol. 1, No. 1, Article . Publication date: April 2026.

36 B.Wang and J.Ma et al.
Prompt used for enhanced claim decomposition in G-Defense decomp+ ,𝐼𝑣+(in §4.2)
You are a fake news detection assistant. Your primary function is to analyze a complex news
claim and decompose it into several clear, atomic sub-claims that collectively capture the full
meaning and verifiability of the claim. 1. Decomposition Strategies
When analyzing the claim, adopt systematic reasoning strategies to ensure a complete and
logically structured breakdown:
- Logical decomposition: Separate the claim into distinct factual assertions that can be verified
independently.
- Causal reasoning: Identify cause–effect or condition–result relations.
- Hierarchical reasoning: Distinguish between general and specific statements or between
main ideas and supporting facts.
- Factual reasoning: Focus on objectively testable propositions rather than opinions or implica-
tions.
2. News and Communication Cues
Incorporate journalistic awareness to better capture how information is presented in news
texts:
- Recognize news structures such as headlines, reported statements, and factual details.
- Distinguish reported speech or attribution (e.g., “X said that...”) from factual assertions about
reality.
- Identify quantitative or comparative claims (e.g., numbers, rates, trends) and evaluative
expressions (e.g., moral or emotional framing).
- If the claim includes a source or attribution, include it as part of the sub-claim (e.g., “Officials
claimed that...” or “According to reports,...”).
- When the claim involves ambiguous or questionable sources, make that explicit (e.g., “It is
claimed, without evidence, that...”).
3. Dependency Cues
If there are logical dependencies among sub-claims, make them explicit:
- Use connectors such as “if,” “because,” “therefore,” or “as a result.”
- Reflect the logical flow between causes, conditions, and consequences.
4. Factual Dimensions
Ensure coverage of the key fact-checking dimensions as appropriate — such as who, what,
when, where, why/how, and consequences.
5. Output Requirements
Each sub-claim should be concise (1–2 sentences), self-contained, and verifiable.
Avoid redundancy and overlapping information.
Base all sub-claims strictly on the information stated or clearly implied in the original claim;
do not introduce new facts or assumptions.
Output only the numbered sub-claims list, without any additional explanations or commentary.
# News claim:𝑐
B Validation of Assumption
This section aims to validate our assumption that the side aligned with the gold veracity label
usually provides evidence with higher informativeness and soundness, which in turn enables the
LLM to generate a stronger explanation than the competing side. To this end, we compare the quality
of competing explanations under different explanation-generation settings, including different
backbone LLMs and prompt formulations. As shown in Table 7, the explanations conditioned on the
, Vol. 1, No. 1, Article . Publication date: April 2026.

A Graph-Enhanced Defense Framework for Explainable Fake News Detection with LLM 37
Table 7. Explanations evaluation results of competing explanations using a 5-Point Likert scale rating by
GPT-4 on RAWFC’s test set. LLM denotes the backbone model used to generate competing explanations, and
Prompt denotes the corresponding explanation-generation prompt. False/Half/True indicate the gold veracity
label of the claim, while F/ T under each group indicate whether the generated explanation is conditioned on
the false or true prior label.
LLM Prompt MetricFalse Half True
F T F T F T
GPT-3.5𝐼𝑟 Informativeness 3.97 3.06 3.87 3.62 3.75 3.96
Soundness 4.86 2.83 3.61 3.45 4.30 4.60
LLaMA3.1𝐼𝑟 Informativeness 4.03 3.36 3.75 3.73 3.57 4.15
Soundness 4.45 2.95 4.01 3.90 3.57 4.33
GPT-3.5 ˜𝐼𝑟Informativeness 4.23 3.50 4.04 3.88 4.04 4.36
Soundness 4.78 3.12 3.99 3.82 4.34 4.84
gold-consistent label generally achieve higher scores in informativeness and soundness than those
conditioned on the competing label. Specifically, for false claims, the false-oriented explanations
are usually rated more favorably than the true-oriented ones; for true claims, the true-oriented
explanations consistently perform better; and for half-true claims, the two sides are relatively closer.
These results provide quantitative evidence supporting our assumption. Moreover, although the
magnitude of the gap varies across settings, the same overall tendency can still be observed across
different backbone LLMs and when replacing the original reasoning prompt 𝐼𝑟with its simplified
version ˜𝐼𝑟, which provides further evidence for our assumption.
Simplified prompt used for veracity-oriented explanation generation, ˜𝐼𝑟
Given a claim: 𝑐𝑖, a veracity label ˜𝑦, please give me a streamlined rationale associated with
the claim for how it is reasoned as ˜𝑦. Below are some sentences that may be helpful for the
rationale, but they are mixed with noise:𝐸 𝑖.
C Human Evaluation Reliability Analysis
To further assess the reliability of the human evaluation, we conduct additional statistical analyses
based on the annotations from five annotators over 30 claims. Specifically, we report both Krippen-
dorff’s𝛼and Kendall’s 𝑊for the four evaluation dimensions introduced in §4.5: misleadingness,
informativeness, soundness, and readability. For Krippendorff’s 𝛼, the raw values are 0.36, 0.10,
0.13, and 0.12, respectively. Since explanation evaluation is inherently subjective, annotators may
use different but internally consistent scoring scales, which can reduce agreement on absolute
ratings. This is consistent with prior findings that annotator disagreement is common in subjective
nature language generation evaluation and does not necessarily invalidate the evaluation itself [ 3].
To further examine this effect, we also compute normalized 𝛼values, which are 0.37, 0.24, 0.30, and
0.23, respectively. The increase after normalization suggests that part of the disagreement comes
from differences in rating scale usage rather than complete disagreement in judgment. Because
Krippendorff’s 𝛼mainly reflects agreement on absolute scores, it may underestimate consistency
in the relative ranking of methods [ 55,58]. We therefore additionally compute Kendall’s 𝑊, which
measures agreement on method rankings across annotators. The resulting Kendall’s 𝑊values
, Vol. 1, No. 1, Article . Publication date: April 2026.

38 B.Wang and J.Ma et al.
are 0.79 for misleadingness, 0.66 for informativeness, 0.69 for soundness, and 0.64 for readability,
indicating moderate to strong ranking agreement. Under the chi-square approximation, all four
values are statistically significant ( 𝑝<0.05). Overall, these two metrics provide complementary
views of reliability. Krippendorff’s 𝛼measures agreement on absolute ratings, while Kendall’s
𝑊measures agreement on relative rankings. Taken together, the results suggest that although
absolute-score agreement is limited by the subjective nature of explanation evaluation, the relative
ordering of methods is reasonably stable across annotators, supporting the reliability of our human
evaluation.
, Vol. 1, No. 1, Article . Publication date: April 2026.