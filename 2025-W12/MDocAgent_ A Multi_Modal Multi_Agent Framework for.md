# MDocAgent: A Multi-Modal Multi-Agent Framework for Document Understanding

**Authors**: Siwei Han, Peng Xia, Ruiyi Zhang, Tong Sun, Yun Li, Hongtu Zhu, Huaxiu Yao

**Published**: 2025-03-18 06:57:21

**PDF URL**: [http://arxiv.org/pdf/2503.13964v1](http://arxiv.org/pdf/2503.13964v1)

## Abstract
Document Question Answering (DocQA) is a very common task. Existing methods
using Large Language Models (LLMs) or Large Vision Language Models (LVLMs) and
Retrieval Augmented Generation (RAG) often prioritize information from a single
modal, failing to effectively integrate textual and visual cues. These
approaches struggle with complex multi-modal reasoning, limiting their
performance on real-world documents. We present MDocAgent (A Multi-Modal
Multi-Agent Framework for Document Understanding), a novel RAG and multi-agent
framework that leverages both text and image. Our system employs five
specialized agents: a general agent, a critical agent, a text agent, an image
agent and a summarizing agent. These agents engage in multi-modal context
retrieval, combining their individual insights to achieve a more comprehensive
understanding of the document's content. This collaborative approach enables
the system to synthesize information from both textual and visual components,
leading to improved accuracy in question answering. Preliminary experiments on
five benchmarks like MMLongBench, LongDocURL demonstrate the effectiveness of
our MDocAgent, achieve an average improvement of 12.1% compared to current
state-of-the-art method. This work contributes to the development of more
robust and comprehensive DocQA systems capable of handling the complexities of
real-world documents containing rich textual and visual information. Our data
and code are available at https://github.com/aiming-lab/MDocAgent.

## Full Text


<!-- PDF content starts -->

MDocAgent: A Multi-Modal Multi-Agent Framework for
Document Understanding
Siwei Han1, Peng Xia1, Ruiyi Zhang2, Tong Sun2, Yun Li1, Hongtu Zhu1, Huaxiu Yao1
1UNC-Chapel Hill,2Adobe Research
{siweih,huaxiu }@cs.unc.edu
Abstract
Document Question Answering (DocQA) is a very common
task. Existing methods using Large Language Models (LLMs)
or Large Vision Language Models (LVLMs) and Retrieval
Augmented Generation (RAG) often prioritize information
from a single modal, failing to effectively integrate textual
and visual cues. These approaches struggle with complex
multi-modal reasoning, limiting their performance on real-
world documents. We present MDocAgent (A Multi-Modal
Multi-Agent Framework for Document Understanding), a
novel RAG and multi-agent framework that leverages both
text and image. Our system employs five specialized agents:
a general agent, a critical agent, a text agent, an image
agent and a summarizing agent. These agents engage in
multi-modal context retrieval, combining their individual
insights to achieve a more comprehensive understanding of
the document’s content. This collaborative approach enables
the system to synthesize information from both textual and
visual components, leading to improved accuracy in ques-
tion answering. Preliminary experiments on five benchmarks
like MMLongBench, LongDocURL demonstrate the effective-
ness of our MDocAgent, achieve an average improvement
of 12.1% compared to current state-of-the-art method. This
work contributes to the development of more robust and
comprehensive DocQA systems capable of handling the com-
plexities of real-world documents containing rich textual
and visual information. Our data and code are available at
https://github.com/aiming-lab/MDocAgent.
1. Introduction
Answering questions based on reference documents
(DocQA) is a critical task in many applications [ 5,8,25,28,
34,35,45], ranging from information retrieval to automated
document analysis. A key challenge in DocQA lies in the
diverse nature of questions and the information needed to an-
swer them [ 7,26]. Questions can refer to textual content, to
visual elements within the document (e.g., charts, diagrams,
Who is the commanding officer in the first figure on the second page?
···TaskLVLMs
.txt.png
OurSystem
General Agent
CriticalAgent
Image Agent
TextAgent
Summarizing Agent
Long documents 
Attention to detailsCross-modalunderstanding
Single-modal Context Retrieval and LLM/LVLM SystemText-based RAG
To p-k
Image-based RAG
LLMLong documents Attention to detailsCross-modalunderstanding
To p-k
LVLM
Long documents Attention to detailsCross-modalunderstanding
Text-based RAG
To p-k
Image-based RAG
To p-k
Long documents Attention to detailsCross-modalunderstanding
Figure 1. Comparison of different approaches for DocQA. LVLMs
often struggle with long documents and lack granular attention
to detail, while also exhibiting limitations in cross-modal under-
standing. Single-modal context retrieval can handle long documents
but still suffers from issues with detailed analysis or integrating
information across modalities. Our MDocAgent addresses these
challenges by combining text and image-based RAG with spe-
cialized agents for refined processing within each modality and a
critical information extraction mechanism, showcasing improved
DocQA performance.
images), or even require the integration of information from
both modalities. Since Large Language Models (LLMs) can
1arXiv:2503.13964v1  [cs.LG]  18 Mar 2025

only handle textual information [ 29], Large Vision Language
Models (LVLMs) are often used in DocQA [ 4,13,24]. As
illustrated in Figure 1, while LVLMs have shown promise
in handling visual content, they often struggle in scenar-
ios where key information is primarily textual, or where a
nuanced understanding of the interplay between text and
visual elements is required [ 5,25,34]. Another challenge
in DocQA lies in the huge volume of information often
present in documents. Processing entire documents directly
can overwhelm computational resources and make it difficult
for models to identify the most pertinent information [ 7,26].
To overcome this challenge, Retrieval Augmented Gener-
ation (RAG) is used as an auxiliary tool to extract the critical
information from a long document [ 11]. While RAG meth-
ods like ColBERT [ 16] and ColPali [ 9] have proven effec-
tive for retrieving textual or visual information respectively,
they often fall short when a question requires integrating in-
sights from both modalities. Existing RAG implementations
typically operate in isolation, either retrieving text or im-
ages [ 18,42], but lack the ability to synthesize information
across these modalities. Consider a document containing
a crucial diagram and accompanying textual explanations.
If a question focuses on the diagram’s content, a purely
text-based RAG system would struggle to pinpoint the rel-
evant information. Conversely, if the question pertains to
a nuanced detail within the textual description, an image-
based RAG would be unable to isolate the necessary textual
segment. This inability to effectively combine multi-modal
information restricts the performance of current RAG-based
approaches in complex DocQA tasks. Moreover, the diverse
and nuanced nature of these multimodal relationships re-
quires not just retrieval, but also a mechanism for reasoning
and drawing inferences across different modalities.
To further address these limitations, we present a novel
framework, a Multi-Modal Multi-Agent Framework for
Document Understanding (MDocAgent), which leverages
the power of both RAG and a collaborative multi-agent sys-
tem where specialized agents collaborate to process and
integrate text and image information. MDocAgent employs
two parallel RAG pipelines: a text-based RAG and an image-
based RAG. These retrievers provide targeted textual and
visual context for our multi-agent system. MDocAgent com-
prises five specialized agents : a general agent for initial
multi-modal processing, a critical agent for identifying key
information, a text agent, an image agent for focused anal-
ysis within their respective modalities, and a summarizing
agent to synthesize the final answer. This collaborative ap-
proach enables our system to effectively tackle questions
that require synthesizing information from both textual and
visual elements, going beyond the capabilities of traditional
RAG methods.
Specifically, MDocAgent operates in five stages: (1) Doc-
ument Pre-processing: Text is extracted via OCR and pagesare preserved as images. (2) Multi-modal Context Re-
trieval: text-based and image-based RAG tools retrieve the
top-k relevant text segments and image pages, respectively.
(3) Initial Analysis and Key Extraction: The general agent
generates an initial response, and the critical agent extracts
key information, providing it to the specialized agents. (4)
Specialized Agent Processing: Text and image agents ana-
lyze the retrieved context within their respective modalities,
guided by the critical information. (5) Answer Synthesis:
The summarizing agent integrates all agent responses to
produce the final answer.
The primary contribution of this paper is a novel multi-
agent framework for DocQA that effectively integrates
specialized agents, each dedicated to a specific modality
or aspect of reasoning, including text and image under-
standing, critical information extraction, and answer syn-
thesis. We demonstrate the efficacy of our approach through
experiments on five benchmarks: MMLongBench [ 26],
LongDocURL [ 7], PaperTab [ 14], PaperText [ 14], and Fe-
taTab [ 14], showing significant improvements in DocQA
performance, with an average of 12.1% compared to current
SOTA method. The empirical improvements demonstrate the
effectiveness of our collaborative multi-agent architecture in
handling long, complex documents and questions. Further-
more, ablation studies validate the contribution of each agent
and the importance of integrating multi-modalities.
2. Related Work
LVLMs in DocQA Tasks. Document Visual Question An-
swering (DocVQA) has evolved from focusing on short
documents to handling complex, long, and multi-document
tasks [ 8,28,35,36], often involving visually rich content
such as charts and tables. This shift requires models capable
of integrating both textual and visual information. Large Vi-
sion Language Models (LVLMs) have emerged to address
these challenges by combining the deep semantic under-
standing of Large Language Models (LLMs) with the ability
to process document images [ 6,22,23,37,40,41,46–51].
LVLMs convert text in images into visual representations,
preserving layout and visual context. However, they face
challenges like input size limitations and potential loss of
fine-grained textual details [ 13,24], making effective inte-
gration of text and visual information crucial for accurate
DocVQA performance [31].
Retrieval-Augmented Generation. Retrieval Augmented
Generation (RAG) enhances LLMs by supplying them with
external text-based context, thereby improving their perfor-
mance in tasks such as DocQA [ 11,18]. Recently, with
the increasing prevalence of visually rich documents, im-
age RAG approaches have been developed to retrieve rel-
evant visual content for Large Vision Language Models
(LVLMs) [ 4,5,42–44]. However, existing methods struggle
2

QuestionText-based RAGImage-based RAGTo p-k segments of textTo p-k segments of image
······
𝑇!	=	{𝑡",𝑡#,...,𝑡$}𝐼!	=	{𝑖",𝑖#,...,𝑖$}𝑞GeneralAgent CriticalAgent		Answer!Critical T extual Information
Critical VisualInformationText Agent
Image Agent
𝑇!𝐼!𝑞𝑇!	=	{𝑡",𝑡#,...,𝑡$}𝐼!	=	{𝑖",𝑖#,...,𝑖$}𝑞
𝑞
		Answer"
		Answer#Summarizing Agent		Answer!
···DocRetrieve
PDFTo o l s
Stage4Stage1Stage2Stage3
Stage5Figure 2. Overview of MDocAgent : A multi-modal multi-agent framework operating in five stages: (1) Documents are processed using PDF
tools to extract text and images. (2) Text-based and image-based RAG retrieves the top-k relevant segments and image pages. (3) The general
agent provides a preliminary answer, and the critical agent extracts critical information from both modalities. (4) Specialized agents process
the retrieved information and critical information within their respective modalities and generate refined answers. (5) The summarizing agent
integrates all previous outputs to generate the final answer.
to effectively integrate and reason over both text and image
information, as retrieval often occurs independently. This
lack of integrated reasoning limits the effectiveness of cur-
rent RAG techniques, especially for complex DocQA tasks
that require a nuanced understanding of both modalities.
Multi-Agent Systems. Multi-agent systems have shown
promise in complex domains like medicine [ 17,21,39].
These systems use specialized agents to focus on different
task aspects [ 3,15,20,33], collaborating to achieve goals
that a single model may struggle with. However, their ap-
plication to DocQA introduces unique challenges stemming
from the need to integrate diverse modalities. Simply com-
bining the outputs of independent text and image agents
often fails to capture the nuanced interplay between these
modalities, which is crucial for accurate document under-
standing. Our framework addresses this by introducing a
general agent for information integration alongside special-
ized text and image agents, enabling collaborative reasoning
and a more comprehensive understanding of document con-
tent, ultimately improving DocVQA performance.
3. Multi-Modal Multi-Agent Framework for
Document Understanding
This section details our proposed framework, MDocAgent,
for tackling the complex challenges of DocQA. MDocA-
gent employs a novel five-stage multi-modal, multi-agent ap-
proach as shown in Figure 2, utilizing specialized agents for
targeted information extraction and cross-modal synthesis toachieve a more comprehensive understanding of document
content. Subsequently, Section 3.1 through Section 3.5 pro-
vide a comprehensive description of MDocAgent’s architec-
ture. This detailed exposition will elucidate the mechanisms
by which MDocAgent effectively integrates and leverages
textual and visual information to achieve improved accuracy
in DocQA.
Preliminary: Document Question Answering. Given a
question qexpressed in natural language and the correspond-
ing document D, the goal is to generate an answer a that
accurately and comprehensively addresses qusing the infor-
mation provided within D.
3.1. Document Pre-Processing
This initial stage prepares the document corpus for subse-
quent processing by transforming it into a format suitable
for both textual and visual analysis. Dconsists of a set of
pagesD={p1, p2, . . . , p N}. For each page pi, textual con-
tent is extracted using a combination of Optical Character
Recognition (OCR) and PDF parsing techniques. OCR is
employed to recognize text within image-based PDFs, while
PDF parsing extracts text directly from digitally encoded text
within the PDF. This dual approach ensures robust text ex-
traction across various document formats and structures. The
extracted text for each page piis represented as a sequence
of textual segments or paragraphs ti={ti1, ti2, . . . , t iM},
where Mrepresents the number of text segments on that
page. Concurrently, each page piis also preserved as an
image, retaining its original visual layout and features. This
3

allows the framework to leverage both textual and visual cues
for comprehensive understanding. This pre-processing re-
sults in two parallel representations of the document corpus:
a textual representation consisting of extracted text segments
and a visual representation consisting of the original page
images. This dual representation forms the foundation for
the multi-modal analysis performed by the framework.
3.2. Multi-modal Context Retrieval
The second stage focuses on efficiently retrieving the most
relevant information from the document corpus, consider-
ing both text and image modalities. Algorithm 1 illustrates
the whole procedure of retrieval. For the textual retrieval,
extracted text segments tiof each page piare indexed us-
ing ColBERT [ 16]. Given the user question q, ColBERT
retrieves the top- kmost relevant text segments, denoted as
Tq={t1, t2, . . . , t k}. This provides the textual context for
subsequent agent processing. Parallel to textual retrieval,
visual context is extracted using ColPali [ 9]. Each page im-
agepiis processed by ColPali to generate a dense visual
embedding Epi∈Rnv×d, where nvrepresents the number
of visual tokens per page and drepresents the embedding
dimension. Using these embeddings and the question q, Col-
Pali retrieves the top- kmost visually relevant pages, denoted
asIq={i1, i2, . . . , i k}. The use of ColPali allows the model
to capture the visual information present in the document,
including layout, figures, and other visual cues.
Algorithm 1 Multi-modal Context Retrieval
Require: Question q, Document D, Text Scores St, Image
Scores Si, Text Relevance Scores Rt, Image Relevance
Scores Ri.
Ensure: Top-k text segments Tq, Top-k image segments Tq.
1:St← {}
2:Si← {} ▷Iterate through each page in the corpus
3:foreachpinDdo
4: foreach text segment tinpdo
5: St[t]←Rt(q, t)▷Calculate text relevance score
6: end for
7: Si[p]←Ri(q, p)▷Calculate image relevance score
8:end for
9:Tq←Top K(St, k) ▷Select top-k text segments
10:Iq←Top K(Si, k) ▷Select top-k image segments
11:return Tq,Iq
3.3. Initial Analysis and Key Extraction
The third stage aims to provide an initial interpretation of the
question and pinpoint the most salient information within
the retrieved context. The general agent AG, functioning
as a preliminary multi-modal integrator, receives both the
retrieved textual context Tqand the visual context Iq. It pro-
cesses these multimodal inputs by effectively combining theinformation embedded within both modalities. This compre-
hensive understanding of the combined context allows AGto
generate a preliminary answer aG, which serves as a crucial
starting point for more specialized analysis in the next stage.
aG=AG(q, Tq, Iq). (1)
Subsequently, the critical agent ACplays a vital role in refin-
ing the retrieved information. It takes as input the question q,
the retrieved contexts TqandIq, and the preliminary answer
aGgenerated by the general agent. The primary function of
ACis to meticulously analyze these inputs and identify the
most crucial pieces of information that are essential to accu-
rately answer the question. This critical information acts as
a guide for the specialized agents in the next stage, focusing
their attention on the most relevant aspects of the retrieved
context.
Tc=AC(q, Tq, aG), I c=AC(q, Iq, aG). (2)
The output of this stage consists of Tc⊂Tq, representing
the critical textual information extracted from the retrieved
text segments, and Ic, which provides a detailed textual de-
scription of the critical visual information extracted from the
retrieved images Iqthat capture the essence of the important
visual elements.
3.4. Specialized Agent Processing
The fourth stage delves deeper into the textual and visual
modalities, leveraging specialized agents guided by the crit-
ical information extracted in the previous stage. The text
agent ATreceives the retrieved text segments Tqand the
critical textual information Tcas input. It operates exclu-
sively within the textual domain, leveraging its specialized
knowledge and analytical capabilities to thoroughly examine
the provided text segments. By focusing specifically on the
critical textual information Tc,ATcan pinpoint the most
relevant evidence within the broader textual context Tqand
perform a more focused analysis. This focused approach
allows for a deeper understanding of the textual nuances
related to the question and culminates in the generation of a
detailed, text-based answer aT.
aT=AT(q, Tq, Tc). (3)
Concurrently, the image agent AIreceives the retrieved im-
agesIqand the critical visual information Ic. This agent
specializes in visual analysis and interpretation. It processes
the images in Iq, paying particular attention to the regions
or features highlighted by the critical visual information Ic.
This targeted analysis allows the agent to extract valuable in-
sights from the visual content, focusing its processing on the
most relevant aspects of the images. The image agent’s analy-
sis results in a visually-grounded answer aI, which provides
a response based on the interpretation of the images.
aI=AI(q, Iq, Ic). (4)
4

3.5. Answer Synthesis
The final stage integrates the diverse outputs from the preced-
ing stages, combining the initial multi-modal understanding
with the specialized agent analyses to produce a compre-
hensive and accurate answer. The summarizing agent AS
receives the answers aG,aT, and aIgenerated by the gen-
eral agent, text agent, and image agent, respectively. This
comprehensive set of information provides a multifaceted
perspective on the question and allows the summarizing
agent to perform a thorough synthesis. The summarizing
agent analyzes the individual agent answers, identifying com-
monalities, discrepancies, and complementary insights. It
considers the supporting evidence provided by each agent.
By resolving potential conflicts or disagreements between
the agents and integrating their individual strengths, the sum-
marizing agent constructs a final answer aSthat leverages
the collective intelligence of the multi-agent system. This fi-
nal answer is not merely a combination of individual answers
but a synthesized response that reflects a deeper and more
nuanced understanding of the information extracted from
both textual and visual modalities. The whole procedure of
this multi-agent collaboration is illustrated in Algorithm 2.
Algorithm 2 Multi-agent Collaboration
Require: Question q, Top-k text segments Tq, Top-k image
segments Iq, General Agent AG, Critical Agent AC,
Text Agent AT, Image Agent AI, Summarizing Agent
AS
Ensure: Final answer as,
1:aG←AG(q, Tq, Iq) ▷General agent answer
2:(Tc, Bc)←AC(q, Tq, Iq, aG)▷Extract critical info
3:aT←AT(q, Tq, Tc) ▷Text agent answer
4:aI←AI(q, Iq, Bc) ▷Image agent answer
5:aS←AS(q, aG, aT, aI) ▷Final answer synthesis
6:return aS
4. Experiments
We evaluate MDocAgent on five document understanding
benchmarks covering multiple scenarios to answer the fol-
lowing questions: (1) Does MDocAgent effectively improve
document understanding accuracy compared to existing
RAG-based approaches? (2) Does each agent in our frame-
work play a meaningful role? (3) How does our approach
enhance the model’s understanding of documents?
4.1. Experiment Setup
Implementation Details . There are five agents in MDocA-
gent: general agent, critical agent, text agent, image
agent and summarizing agent. We adopt Llama-3.1-8B-
Instruct [ 12] as the base model for text agent, Qwen2-
VL-7B-Instruct [ 38] for other four agents, and select Col-
BERTv2 [ 32] and ColPali [ 10] as the text and image retriev-ers, respectively. In our settings of RAG, we retrieve 1 or 4
highest-scored segments as input context for each example.
All experiments are conducted on 4 NVIDIA H100 GPUs.
Details of models and settings are shown in Appendix A.
Datasets . The benchmarks involve MMLongBench [ 26],
LongDocUrl [ 7], PaperTab [ 14], PaperText [ 14], Fe-
taTab [ 14]. These evaluation datasets cover a variety of sce-
narios, including both open- and closed-domain, textual and
visual, long and short documents, ensuring fairness and com-
pleteness in the evaluation. Details of dataset descriptions
are in Appendix A.2.
Metrics . For all benchmarks, following Deng et al. [7], Ma
et al. [26], we leverage GPT-4o [ 30] as the evaluation model
to assess the consistency between the model’s output and
the reference answer, producing a binary decision (cor-
rect/incorrect). We provide the average accuracy rate for
each benchmark.
4.2. Main Results
In this section, we provide a comprehensive comparison of
MDocAgent on multiple benchmarks against existing state-
of-the-art LVLMs and RAG-based methods built on them.
Our findings can be summarized as:
MDocAgent Outperforms All the Comparison Methods
and Other LVLMs . We compare our method with baseline
approaches on document understanding tasks, with the re-
sults presented in Table 1. Overall, our method outperforms
all baselines across all benchmarks.
Top-1 Retrieval Performance. With top-1 retrieval,
MDocAgent demonstrates a significant performance im-
provement. On PaperText, MDocAgent achieves a score
of 0.399, surpassing the second-best method, M3DocRAG,
by 16.7%. Similarly, on FetaTab, MDocAgent attains a score
of 0.600, exceeding the second-best method by an impres-
sive 21.0%. Compared to the best LVLM (Qwen2.5-VL-7B)
and text-RAG-based (ColBERTv2+Llama-3.1-8B) baselines,
our approach demonstrates a remarkable average improve-
ment of 51.9% and 23.7% on average across all benchmarks.
This improvement highlights the benefits of incorporating
visual information and the collaborative multi-agent architec-
ture in our framework. Furthermore, recent state-of-the-art
image-RAG-based method M3DocRAG [ 5] show promis-
ing results, yet our approach still outperforms it by 12.1%
on average. This suggests that our multi-agent framework,
with its specialized agents and critical information extraction
mechanism addresses the core challenges of information
overload, granular attention to detail, and cross-modality
understanding more effectively than existing methods.
Top-4 Retrieval Performance. When using top-4 retrieval,
the advantages of our method are further demonstrated.
MDocAgent consistently achieves the highest scores across
all benchmarks. On average, MDocAgent outperforms
Qwen2.5-VL-7B by a remarkable 73.5%. Interestingly, with
5

Table 1. Performance comparison across MDocAgent and existing state-of-the-art LVLMs and RAG-based methods.
Method MMLongBench LongDocUrl PaperTab PaperText FetaTab Avg
LVLMs
Qwen2-VL-7B-Instruct [38] 0.165 0.296 0.087 0.166 0.324 0.208
Qwen2.5-VL-7B-Instruct [2] 0.224 0.389 0.127 0.271 0.329 0.268
LLaV A-v1.6-Mistral-7B [22] 0.099 0.074 0.033 0.033 0.110 0.070
Phi-3.5-Vision-Instruct [1] 0.144 0.280 0.071 0.165 0.237 0.179
LLaV A-One-Vision-7B [19] 0.053 0.126 0.056 0.108 0.077 0.084
SmolVLM-Instruct [27] 0.081 0.163 0.066 0.137 0.142 0.118
RAG methods (top 1)
ColBERTv2 [32]+LLaMA-3.1-8B [12] 0.241 0.429 0.155 0.332 0.490 0.329
M3DocRAG [5] (ColPali [9]+Qwen2-VL-7B [38]) 0.276 0.506 0.196 0.342 0.497 0.363
MDocAgent (Ours) 0.299 0.517 0.219 0.399 0.600 0.407
RAG methods (top 4)
ColBERTv2 [32]+LLaMA-3.1-8B [12] 0.273 0.491 0.277 0.460 0.673 0.435
M3DocRAG [5] (ColPali [9]+Qwen2-VL-7B [38]) 0.296 0.554 0.237 0.430 0.578 0.419
MDocAgent (Ours) 0.315 0.578 0.278 0.487 0.675 0.465
Table 2. Performance comparison across different MDocAgent’s variants.
VariantsAgent Configuration Evaluation Benchmarks
General & Critical Agent Text Agent Image Agent MMLongBench LongDocUrl PaperTab PaperText FetaTab Avg
MDocAgent i ✓ ✗ ✓ 0.287 0.508 0.196 0.376 0.552 0.384
MDocAgent t ✓ ✓ ✗ 0.288 0.484 0.201 0.391 0.596 0.392
MDocAgent s ✗ ✓ ✓ 0.285 0.479 0.188 0.365 0.592 0.382
MDocAgent ✓ ✓ ✓ 0.299 0.517 0.219 0.399 0.600 0.407
top-4 retrieval, M3DocRAG slightly performs worse than
ColBERTv2+Llama-3.1-8B compared to top-1 retrieval.
This may suggest limitations on M3DocRAG’s capacity of
selectively integrate across multiple retrieved documents
when dealing with larger amounts of retrieved informa-
tion. On average, MDocAgent exceeds M3DocRAG by
10.9%. Meanwhile, compared to ColBERTv2+Llama-3.1-
8B, MDocAgent demonstrates a 6.9% improvement. This
consistent improvement suggests that our method effectively
harnesses the additional contextual information provided by
the top-4 retrieved items, offering a greater benefit with more
retrieval results.
4.3. Quantitative Analysis
In this section, we conduct three quantitative analyses to
understand the effectiveness and contribution of different
components within our proposed framework. First, we per-
form ablation studies to assess the impact of removing in-
dividual agents or groups of agents. Second, we present
a fine-grained performance analysis, examining MDocA-
gent’s performance across different evidence modalities on
MMLongBench to pinpoint the source of its improvements.Third, a compatibility analysis explores the framework’s
performance with different image-based RAG backbones to
demonstrate its robustness and generalizability. Additionally,
we present experimental results showcasing its performance
with different model backbones in Appendix B.2.
4.3.1. Ablation Studies
Table 2 presents a comparison of our full method (MDocA-
gent) against it’s variants: MDocAgent i(without the text
agent) and MDocAgent t(without the image agent). Across
all benchmarks, the full MDocAgent method consistently
achieves the highest performance. The removal of either
specialized agent, text or image, results in a noticeable per-
formance drop. This underscores the importance of incorpo-
rating both text and image modalities through specialized
agents within our framework. The performance difference
is most pronounced in benchmarks like LongDocURL and
PaperText, which likely contain richer visual or textual infor-
mation respectively, further highlighting the value of special-
ized processing. This ablation study clearly demonstrates the
synergistic effect of combining specialized agents dedicated
to each modality.
6

Table 3. Performance comparison across different evidence source on MMLongBench.
Method Chart Table Pure-text Generalized-text Figure Avg
LVLMs (up to 32 pages)
Qwen2-VL-7B-Instruct 0.182 0.097 0.209 0.185 0.197 0.165
Qwen2.5-VL-7B-Instruct 0.188 0.124 0.265 0.210 0.254 0.224
LLaV A-v1.6-Mistral-7B 0.011 0.023 0.033 0.000 0.057 0.074
LLaV A-One-Vision-7B 0.045 0.051 0.076 0.017 0.084 0.053
Phi-3.5-Vision-Instruct 0.159 0.101 0.156 0.160 0.164 0.144
SmolVLM-Instruct 0.062 0.065 0.123 0.118 0.094 0.081
RAG methods (top 1)
ColBERTv2+LLaMA-3.1-8B 0.148 0.203 0.265 0.143 0.074 0.241
M3DocRAG (ColPali+Qwen2-VL-7B) 0.268 0.263 0.334 0.250 0.303 0.276
MDocAgent (Ours) 0.269 0.300 0.348 0.252 0.298 0.299
RAG methods (top 4)
ColBERTv2+LLaMA-3.1-8B 0.182 0.267 0.311 0.168 0.120 0.273
M3DocRAG (ColPali+Qwen2-VL-7B) 0.290 0.318 0.371 0.277 0.321 0.296
MDocAgent (Ours) 0.347 0.323 0.401 0.294 0.321 0.315
Table 4. Performance comparison between using ColPali and ColQwen2-v1.0 as MDocAgent’s image-based RAG model.
MMLongBench LongDocUrl PaperTab PaperText FetaTab Avg
+ColPali 0.299 0.517 0.219 0.399 0.600 0.407
+ColQwen2-v1.0 0.303 0.520 0.216 0.391 0.603 0.407
Table 2 also compares MDocAgent with MDocAgent s,
where both the general agent and the critical agent are re-
moved, to evaluate their contribution. The consistent im-
provement of the full method over MDocAgent sacross all
datasets clearly underscores the importance of these two
agents. The general agent establishes a crucial foundation by
initially integrating both text and image modalities, provid-
ing a holistic understanding of the context. Removing this
integration step noticeably reduces the subsequent agents’
capacity to focus their analysis of critical information and
answer effectively. On top of general modal integration, re-
moving the critical agent limits the framework’s ability to
effectively identify and leverage crucial information. This
highlights the essential role of the critical agent in focus-
ing the specialized agents’ attention and facilitating more
targeted and efficient information extraction.
4.3.2. Fine-Grained Performance Analysis
We present an in-depth analysis of the performance in dif-
ferent types of evidence modalities, by further analyzing
the scores on MMLongBench in Table 3, to gain a better
understanding of the performance improvements achieved
by MDocAgent. We also illustrate the results of evidence
modalities of LongDocURL in Appendix B.1. Accordingto the results, MDocAgent outperforms all LVLM baselines
among all types of evidence modalities. When comparing
RAG methods using the top 1 retrieval approach, though
M3DocRAG performs slightly better on Figure category,
MDocAgent show strong performance in Chart, Table and
Text categories, reflecting its enhanced capability to pro-
cess textual and visual information. With the top 4 retrieval
strategy, MDocAgent enhances its performance in the all
categories, specifically in Figure, highlighting its effective
handling of large and varied information sources.
4.3.3. Compatibility Analysis
We further analyze the compatibility of MDocAgent with
different RAG backbones. Table 4 presents results using
two image-based RAG models, ColPali and ColQwen2-v1.0,
within our proposed framework. Both models achieve compa-
rable overall performance, with an identical average score of
0.407 across all benchmarks. While ColQwen2-v1.0 shows
a slight advantage on MMLongBench, LongDocUrl, and
FetaTab, ColPali performs marginally better on PaperTab
and PaperText. This suggests that the choice of image-based
RAG model has minimal impact on the framework’s over-
all effectiveness, underscoring the robustness of our multi-
agent architecture. Moreover, the consistency in performance
7

The population of foreign-born Latinos is greaterin the survey. … 795 respondents were foreign-born (excluding Puerto Rico), while 705 respondents were U.S. born (including Puerto Rico).Latinos interviewed by cellphone.Evidencepages: 19, 20Question
According to the report, which one is greater in population in the survey? Foreign born Latinos, or the Latinos interviewed by cellphone?
Answer… the population of foreign born Latinos is greaterin the survey.  … 795 respondents were foreign born (excluding Puerto Rico), while 1,051 respondents were interviewed by cellphone.OursDetails:✓Conclusion:	Χ
Details: ΧConclusion:	Χ
ColBERT: 19ColPali: 19
General Agent:  Latinos interviewed by cellphone is greater than foreign-born Latinos.Critical Agent: CriticalInfo:•Te x t : Fo re i g n  b o r n  ( e x c l . P R ) , •Image: cellphone sampling frameTe x t  A g e n t : … Foreign born (excluding Puerto Rico): 795. … The sample consisted … and a cellphone sampling frame (1,051 interviews). Image Agent:  The cellphone sampling frame yielded 1,051 interviews, while the foreign-born … respondents numbered 795.Final Answer: The number of Latinos interviewed by cellphone (1,051) is greater than the number of foreign-born Latinos (795).Details:✓Conclusion:✓M3DocRAGColBERT + Llama 3.1 8B
Top-1 Page Retrieval
Figure 3. A Case study of MDocAgent compared with other two RAG-method baselines(ColBERT + Llama 3.1-8B and M3DocRAG).
Given a question comparing two population sizes, both baseline methods fail to arrive at the correct answer. Our framework, through the
collaborative efforts of its specialized agents, successfully identifies the relevant information from both text and a table within the image,
ultimately synthesizing the correct answer. This highlights the importance of granular, multi-modal analysis and the ability to accurately
process information within the context.
across different RAG models highlights that the core strength
of our approach lies in the multi-agent architecture itself,
rather than reliance on a specific retrieval model. This fur-
ther reinforces the compatibility of our proposed method.
4.4. Case Study
We perform a case study to better understand MDocAgent.
Figure 3 illustrates an example. The question requires ex-
tracting and comparing numerical information related to two
distinct Latino populations from both textual and tabular
data within a document. While both ColBERT and ColPali
successfully retrieve the relevant page containing the neces-
sary information, both baseline methods fail to synthesize
the correct answer. The ColBERT + Llama-3.1-8B baseline,
relying solely on text, incorrectly concludes that the foreign-
born Latino population is greater, demonstrating a failure to
accurately interpret the numerical data presented within the
document’s textual content. Similarly, M3DocRAG fails to
correctly interpret the question due to capturing wrong infor-
mation. In contrast, our multi-agent framework successfully
navigates this complexity and gives the correct answer.
Specifically, the general agent provides a correct but
vague answer, making the critical agent essential for identify-
ing key phrases like “Foreign born (excl. PR)” and the “cell-
phone sampling frame” table. This guides specialized agentsto precise locations for efficient data extraction. Both text
agent and image agent correctly extract 795 for foreign-born
Latinos and 1,051 for cellphone-interviewed Latinos. The
summarizing agent then integrates these insights for accu-
rate comparison and a comprehensive final answer. This case
study demonstrates how our structured, multi-agent frame-
work outperforms methods struggling with integrated text
and image analysis (See more case studies in Appendix B.3).
5. Conclusion
This paper presents a multi-agent framework MDocAgent for
DocQA that integrates text and visual information through
specialized agents and a dual RAG approach. Our frame-
work addresses the limitations of existing methods by em-
ploying agents dedicated to text processing, image analysis,
and critical information extraction, culminating in a synthe-
sizing agent for final answer generation. Experimental re-
sults demonstrate significant improvements over LVLMs and
multi-modal RAG methods, highlighting the efficacy of our
collaborative multi-agent architecture. Our framework effec-
tively handles information overload and promotes detailed
cross-modal understanding, leading to more accurate and
comprehensive answers in complex DocQA tasks. Future
work will explore more advanced inter-agent communication
and the integration of external knowledge sources.
8

Acknowledgement
This research was partially supported by NIH
1R01AG085581 and Cisco Faculty Research
Award.
References
[1]Marah Abdin, Jyoti Aneja, Hany Awadalla, Ahmed Awadal-
lah, Ammar Ahmad Awan, Nguyen Bach, Amit Bahree, Arash
Bakhtiari, Jianmin Bao, Harkirat Behl, et al. Phi-3 techni-
cal report: A highly capable language model locally on your
phone. arXiv preprint arXiv:2404.14219 , 2024. 6, 12
[2]Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin
Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang,
Humen Zhong, Yuanzhi Zhu, Mingkun Yang, Zhaohai Li,
Jianqiang Wan, Pengfei Wang, Wei Ding, Zheren Fu, Yiheng
Xu, Jiabo Ye, Xi Zhang, Tianbao Xie, Zesen Cheng, Hang
Zhang, Zhibo Yang, Haiyang Xu, and Junyang Lin. Qwen2.5-
vl technical report. arXiv preprint arXiv:2502.13923 , 2025.
6, 12
[3]Chi-Min Chan, Weize Chen, Yusheng Su, Jianxuan Yu, Wei
Xue, Shanghang Zhang, Jie Fu, and Zhiyuan Liu. Chate-
val: Towards better llm-based evaluators through multi-agent
debate. arXiv preprint arXiv:2308.07201 , 2023. 3
[4]Zhanpeng Chen, Chengjin Xu, Yiyan Qi, and Jian Guo.
Mllm is a strong reranker: Advancing multimodal retrieval-
augmented generation via knowledge-enhanced reranking
and noise-injected training. arXiv preprint arXiv:2407.21439 ,
2024. 2
[5]Jaemin Cho, Debanjan Mahata, Ozan Irsoy, Yujie He, and
Mohit Bansal. M3docrag: Multi-modal retrieval is what you
need for multi-page multi-document understanding. arXiv
preprint arXiv:2411.04952 , 2024. 1, 2, 5, 6, 12
[6]Wenliang Dai, Junnan Li, Dongxu Li, Anthony Meng Huat
Tiong, Junqi Zhao, Weisheng Wang, Boyang Li, Pascale Fung,
and Steven Hoi. Instructblip: Towards general-purpose vision-
language models with instruction tuning. arXiv preprint
arXiv:2305.06500 , 2023. 2
[7]Chao Deng, Jiale Yuan, Pi Bu, Peijie Wang, Zhong-Zhi Li,
Jian Xu, Xiao-Hui Li, Yuan Gao, Jun Song, Bo Zheng, et al.
Longdocurl: a comprehensive multimodal long document
benchmark integrating understanding, reasoning, and locating.
arXiv preprint arXiv:2412.18424 , 2024. 1, 2, 5, 12
[8]Yihao Ding, Zhe Huang, Runlin Wang, YanHang Zhang, Xi-
anru Chen, Yuzhong Ma, Hyunsuk Chung, and Soyeon Caren
Han. V-doc: Visual questions answers with documents. In
Proceedings of the IEEE/CVF conference on computer vision
and pattern recognition , pages 21492–21498, 2022. 1, 2
[9]Manuel Faysse, Hugues Sibille, Tony Wu, Bilel Omrani, Gau-
tier Viaud, C ´eline Hudelot, and Pierre Colombo. Colpali:
Efficient document retrieval with vision language models. In
The Thirteenth International Conference on Learning Repre-
sentations , 2024. 2, 4, 6, 12
[10] Manuel Faysse, Hugues Sibille, Tony Wu, Bilel Omrani, Gau-
tier Viaud, C ´eline Hudelot, and Pierre Colombo. Colpali:
Efficient document retrieval with vision language models,
2024. 5[11] Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang Jia, Jinliu
Pan, Yuxi Bi, Yi Dai, Jiawei Sun, Haofen Wang, and Haofen
Wang. Retrieval-augmented generation for large language
models: A survey. arXiv preprint arXiv:2312.10997 , 2, 2023.
2
[12] Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, Ab-
hinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha
Letman, Akhil Mathur, Alan Schelten, Alex Vaughan, et al.
The llama 3 herd of models. arXiv preprint arXiv:2407.21783 ,
2024. 5, 6, 12
[13] Anwen Hu, Haiyang Xu, Jiabo Ye, Ming Yan, Liang Zhang,
Bo Zhang, Chen Li, Ji Zhang, Qin Jin, Fei Huang, et al. mplug-
docowl 1.5: Unified structure learning for ocr-free document
understanding. arXiv preprint arXiv:2403.12895 , 2024. 2
[14] Yulong Hui, Yao Lu, and Huanchen Zhang. Uda: A bench-
mark suite for retrieval augmented generation in real-world
document analysis. arXiv preprint arXiv:2406.15187 , 2024.
2, 5, 12
[15] Shyam Sundar Kannan, Vishnunandan LN Venkatesh, and
Byung-Cheol Min. Smart-llm: Smart multi-agent robot task
planning using large language models. In 2024 IEEE/RSJ
International Conference on Intelligent Robots and Systems
(IROS) , pages 12140–12147. IEEE, 2024. 3
[16] Omar Khattab and Matei Zaharia. Colbert: Efficient and
effective passage search via contextualized late interaction
over bert. In Proceedings of the 43rd International ACM SI-
GIR conference on research and development in Information
Retrieval , pages 39–48, 2020. 2, 4
[17] Yubin Kim, Chanwoo Park, Hyewon Jeong, Yik Siu Chan,
Xuhai Xu, Daniel McDuff, Hyeonhoon Lee, Marzyeh Ghas-
semi, Cynthia Breazeal, Hae Park, et al. Mdagents: An adap-
tive collaboration of llms for medical decision-making. Ad-
vances in Neural Information Processing Systems , 37:79410–
79452, 2024. 3
[18] Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni,
Vladimir Karpukhin, Naman Goyal, Heinrich K ¨uttler, Mike
Lewis, Wen-tau Yih, Tim Rockt ¨aschel, et al. Retrieval-
augmented generation for knowledge-intensive nlp tasks. Ad-
vances in neural information processing systems , 33:9459–
9474, 2020. 2
[19] Bo Li, Yuanhan Zhang, Dong Guo, Renrui Zhang, Feng Li,
Hao Zhang, Kaichen Zhang, Yanwei Li, Ziwei Liu, and Chun-
yuan Li. Llava-onevision: Easy visual task transfer. arXiv
preprint arXiv:2408.03326 , 2024. 6, 12
[20] Bingxuan Li, Yiwei Wang, Jiuxiang Gu, Kai-Wei Chang,
and Nanyun Peng. Metal: A multi-agent framework for
chart generation with test-time scaling. arXiv preprint
arXiv:2502.17651 , 2025. 3
[21] Guohao Li, Hasan Hammoud, Hani Itani, Dmitrii Khizbullin,
and Bernard Ghanem. Camel: Communicative agents for”
mind” exploration of large language model society. Advances
in Neural Information Processing Systems , 36:51991–52008,
2023. 3
[22] Haotian Liu, Chunyuan Li, Yuheng Li, and Yong Jae Lee.
Improved baselines with visual instruction tuning. In Pro-
ceedings of the IEEE/CVF Conference on Computer Vision
and Pattern Recognition , pages 26296–26306, 2024. 2, 6, 12
9

[23] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee.
Visual instruction tuning. Advances in neural information
processing systems , 36, 2024. 2
[24] Chuwei Luo, Yufan Shen, Zhaoqing Zhu, Qi Zheng, Zhi Yu,
and Cong Yao. Layoutllm: Layout instruction tuning with
large language models for document understanding. In Pro-
ceedings of the IEEE/CVF conference on computer vision and
pattern recognition , pages 15630–15640, 2024. 2
[25] Xueguang Ma, Shengyao Zhuang, Bevan Koopman, Guido
Zuccon, Wenhu Chen, and Jimmy Lin. Visa: Retrieval aug-
mented generation with visual source attribution. arXiv
preprint arXiv:2412.14457 , 2024. 1, 2
[26] Yubo Ma, Yuhang Zang, Liangyu Chen, Meiqi Chen, Yizhu
Jiao, Xinze Li, Xinyuan Lu, Ziyu Liu, Yan Ma, Xiaoyi Dong,
Pan Zhang, Liangming Pan, Yu-Gang Jiang, Jiaqi Wang, Yixin
Cao, and Aixin Sun. Mmlongbench-doc: Benchmarking long-
context document understanding with visualizations, 2024. 1,
2, 5, 12
[27] Andr ´es Marafioti, Orr Zohar, Miquel Farr ´e, Merve Noyan,
Elie Bakouch, Pedro Cuenca, Cyril Zakka, Loubna Ben Allal,
Anton Lozhkov, Nouamane Tazi, Vaibhav Srivastav, Joshua
Lochner, Hugo Larcher, Mathieu Morlon, Lewis Tunstall,
Leandro von Werra, and Thomas Wolf. Smolvlm: Redefining
small and efficient multimodal models. 2025. 6, 12
[28] Anand Mishra, Shashank Shekhar, Ajeet Kumar Singh, and
Anirban Chakraborty. Ocr-vqa: Visual question answering by
reading text in images. In 2019 international conference on
document analysis and recognition (ICDAR) , pages 947–952.
IEEE, 2019. 1, 2
[29] Humza Naveed, Asad Ullah Khan, Shi Qiu, Muhammad
Saqib, Saeed Anwar, Muhammad Usman, Naveed Akhtar,
Nick Barnes, and Ajmal Mian. A comprehensive overview
of large language models. arXiv preprint arXiv:2307.06435 ,
2023. 2
[30] OpenAI. Gpt-4 technical report, 2023. https://arxiv.
org/abs/2303.08774 . 5, 15
[31] Jaeyoo Park, Jin Young Choi, Jeonghyung Park, and Bohyung
Han. Hierarchical visual feature aggregation for ocr-free
document understanding. Advances in Neural Information
Processing Systems , 37:105972–105996, 2024. 2
[32] Keshav Santhanam, Omar Khattab, Jon Saad-Falcon, Christo-
pher Potts, and Matei Zaharia. Colbertv2: Effective and effi-
cient retrieval via lightweight late interaction. arXiv preprint
arXiv:2112.01488 , 2021. 5, 6, 12
[33] Peng Su, Kun Wang, Xingyu Zeng, Shixiang Tang, Dapeng
Chen, Di Qiu, and Xiaogang Wang. Adapting object detec-
tors with conditional domain normalization. In Computer
Vision–ECCV 2020: 16th European Conference, Glasgow,
UK, August 23–28, 2020, Proceedings, Part XI 16 , pages
403–419. Springer, 2020. 3
[34] Manan Suri, Puneet Mathur, Franck Dernoncourt, Kanika
Goswami, Ryan A Rossi, and Dinesh Manocha. Visdom:
Multi-document qa with visually rich elements using mul-
timodal retrieval-augmented generation. arXiv preprint
arXiv:2412.10704 , 2024. 1, 2
[35] Ryota Tanaka, Kyosuke Nishida, Kosuke Nishida, Taku
Hasegawa, Itsumi Saito, and Kuniko Saito. Slidevqa: Adataset for document visual question answering on multiple
images. In Proceedings of the AAAI Conference on Artificial
Intelligence , pages 13636–13645, 2023. 1, 2
[36] Rub`en Tito, Dimosthenis Karatzas, and Ernest Valveny. Hi-
erarchical multimodal transformers for multipage docvqa.
Pattern Recognition , 144:109834, 2023. 2
[37] Haibo Tong, Zhaoyang Wang, Zhaorun Chen, Haonian Ji,
Shi Qiu, Siwei Han, Kexin Geng, Zhongkai Xue, Yiyang
Zhou, Peng Xia, et al. Mj-video: Fine-grained benchmarking
and rewarding video preferences in video generation. arXiv
preprint arXiv:2502.01719 , 2025. 2
[38] Peng Wang, Shuai Bai, Sinan Tan, Shijie Wang, Zhihao Fan,
Jinze Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge,
Yang Fan, Kai Dang, Mengfei Du, Xuancheng Ren, Rui Men,
Dayiheng Liu, Chang Zhou, Jingren Zhou, and Junyang Lin.
Qwen2-vl: Enhancing vision-language model’s perception of
the world at any resolution. arXiv preprint arXiv:2409.12191 ,
2024. 5, 6, 12
[39] Qingyun Wu, Gagan Bansal, Jieyu Zhang, Yiran Wu, Beibin
Li, Erkang Zhu, Li Jiang, Xiaoyun Zhang, Shaokun Zhang,
Jiale Liu, et al. Autogen: Enabling next-gen llm ap-
plications via multi-agent conversation. arXiv preprint
arXiv:2308.08155 , 2023. 3
[40] Peng Xia, Ze Chen, Juanxi Tian, Yangrui Gong, Ruibo Hou,
Yue Xu, Zhenbang Wu, Zhiyuan Fan, Yiyang Zhou, Kangyu
Zhu, et al. Cares: A comprehensive benchmark of trustworthi-
ness in medical vision language models. Advances in Neural
Information Processing Systems , 37:140334–140365, 2024. 2
[41] Peng Xia, Siwei Han, Shi Qiu, Yiyang Zhou, Zhaoyang Wang,
Wenhao Zheng, Zhaorun Chen, Chenhang Cui, Mingyu Ding,
Linjie Li, et al. Mmie: Massive multimodal interleaved
comprehension benchmark for large vision-language mod-
els.arXiv preprint arXiv:2410.10139 , 2024. 2
[42] Peng Xia, Kangyu Zhu, Haoran Li, Tianze Wang, Weijia
Shi, Sheng Wang, Linjun Zhang, James Zou, and Huaxiu
Yao. Mmed-rag: Versatile multimodal rag system for medical
vision language models. arXiv preprint arXiv:2410.13085 ,
2024. 2
[43] Peng Xia, Kangyu Zhu, Haoran Li, Hongtu Zhu, Yun Li, Gang
Li, Linjun Zhang, and Huaxiu Yao. Rule: Reliable multimodal
rag for factuality in medical vision language models. In
Proceedings of the 2024 Conference on Empirical Methods
in Natural Language Processing , pages 1081–1093, 2024.
[44] Shuo Xing, Yuping Wang, Peiran Li, Ruizheng Bai, Yueqi
Wang, Chengxuan Qian, Huaxiu Yao, and Zhengzhong Tu.
Re-align: Aligning vision language models via retrieval-
augmented direct preference optimization. arXiv preprint
arXiv:2502.13146 , 2025. 2
[45] Junyuan Zhang, Qintong Zhang, Bin Wang, Linke Ouyang,
Zichen Wen, Ying Li, Ka-Ho Chow, Conghui He, and Wentao
Zhang. Ocr hinders rag: Evaluating the cascading impact
of ocr on retrieval-augmented generation. arXiv preprint
arXiv:2412.02592 , 2024. 1
[46] Yaqi Zhang, Di Huang, Bin Liu, Shixiang Tang, Yan Lu,
Lu Chen, Lei Bai, Qi Chu, Nenghai Yu, and Wanli Ouyang.
Motiongpt: Finetuned llms are general-purpose motion gen-
erators. In Proceedings of the AAAI Conference on Artificial
Intelligence , pages 7368–7376, 2024. 2
10

[47] Yiyang Zhou, Chenhang Cui, Jaehong Yoon, Linjun Zhang,
Zhun Deng, Chelsea Finn, Mohit Bansal, and Huaxiu Yao.
Analyzing and mitigating object hallucination in large vision-
language models. arXiv preprint arXiv:2310.00754 , 2023.
[48] Yiyang Zhou, Chenhang Cui, Rafael Rafailov, Chelsea Finn,
and Huaxiu Yao. Aligning modalities in vision large lan-
guage models via preference fine-tuning. arXiv preprint
arXiv:2402.11411 , 2024.
[49] Yiyang Zhou, Zhiyuan Fan, Dongjie Cheng, Sihan Yang,
Zhaorun Chen, Chenhang Cui, Xiyao Wang, Yun Li, Lin-
jun Zhang, and Huaxiu Yao. Calibrated self-rewarding vision
language models. arXiv preprint arXiv:2405.14622 , 2024.
[50] Deyao Zhu, Jun Chen, Xiaoqian Shen, Xiang Li, and Mo-
hamed Elhoseiny. Minigpt-4: Enhancing vision-language
understanding with advanced large language models. arXiv
preprint arXiv:2304.10592 , 2023.
[51] Kangyu Zhu, Peng Xia, Yun Li, Hongtu Zhu, Sheng Wang,
and Huaxiu Yao. Mmedpo: Aligning medical vision-language
models with clinical-aware multimodal preference optimiza-
tion. arXiv preprint arXiv:2412.06141 , 2024. 2
11

A. Experimental Setup
A.1. Baseline Models
•Qwen2-VL-7B-Instruct [38]: A large vision-language
model developed by Alibaba, designed to handle multiple
images as input.
•Qwen2.5-VL-7B-Instruct [2]: An enhanced version of
Qwen2-VL-7B-Instruct, offering improved performance
in processing multiple images.
•llava-v1.6-mistral-7b [22]: Also called LLaV A-NeXT, a
vision-language model improved upon LLaVa-1.5, capa-
ble of interpreting and generating content from multiple
images.
•Phi-3.5-vision-instruct [1]: A model developed by Mi-
crosoft that integrates vision and language understanding,
designed to process and generate responses based on mul-
tiple images.
•llava-one-vision-7B [19]: A model trained on LLaV A-
OneVision, based on Qwen2-7B language model with a
context window of 32K tokens.
•SmolVLM-Instruct [27]: A compact vision-language
model developed by HuggingFace, optimized for handling
image inputs efficiently.
•ColBERTv2+Llama-3.1-8B-Instruct [12,32]: A text-
based RAG pipeline that utilizes ColBERTv2 [ 32] for
retrieving text segments and Llama-3.1-8B-Instruct as the
LLM to generate responses.
•M3DocRAG [5]: An image-based RAG pipeline that
employs ColPali [ 9] for retrieving image segments and
Qwen2-VL-7B-Instruct [ 38] as the LVLM for answer gen-
eration.
A.2. Evaluation Benchmarks
•MMLongBench [26]: Evaluates models’ ability to under-
stand long documents with rich layouts and multi-modal
components, comprising 1091 questions and 135 docu-
ments averaging 47.5 pages each.
•LongDocURL [7]: Provides a comprehensive multi-modal
long document benchmark integrating understanding, rea-
soning, and locating tasks, covering over 33,000 pages of
documents and 2,325 question-answer pairs.
•PaperTab [14]: Focuses on evaluating models’ ability to
comprehend and extract information from tables within
NLP research papers, covering 393 questions among 307
documents.
•PaperText [14]: Assesses models’ proficiency in under-
standing the textual content of NLP research papers, cov-
ering 2804 questions among 1087 documents.
•FetaTab [14]: a question-answering dataset for tables from
Wikipedia pages, challengeing models to generate free-
form text answers, comprising 1023 questions and 878
documents.A.3. Hyperparameter Settings
•Temperature : All models use their default temperature
setting.
•Max New Tokens : 256.
•Max Tokens per Image (Qwen2-VL-7B-Instruct) :
– Top-1 retrieval : 16,384 (by default).
– Top-4 retrieval : 2,048.
•Image Resolution : 144 (for all benchmarks).
A.4. Prompt Settings
General Agent
You are an advanced agent capable of analyzing both
text and images. Your task is to use both the textual
and visual information provided to answer the user’s
question accurately.
Extract Text from Both Sources : If the image con-
tains text, extract it and consider both the text in the
image and the provided textual content.
Analyze Visual and Textual Information : Combine
details from both the image (e.g., objects, scenes,
or patterns) and the text to build a comprehensive
understanding of the content.
Provide a Combined Answer : Use the relevant de-
tails from both the image and the text to provide a
clear, accurate, and context-aware response to the
user’s question.
When responding:
•If both the image and text contain similar or over-
lapping information, cross-check and use both to
ensure consistency.
•If the image contains information not present in
the text, include it in your response if it is relevant
to the question.
•If the text and image offer conflicting details, ex-
plain the discrepancies and clarify the most reliable
source.
Critical Agent
Provide a Python dictionary of critical information
based on all given information—one for text and one
for image.
Respond exclusively in a valid dictionary format
without any additional text. The format should be:
{”text”: ”critical information for text”, ”image”:
”critical information for image” }
12

Method Layout Text Figure Table Others Avg
LVLMs
Qwen2-VL-7B-Instruct 0.264 0.386 0.308 0.207 0.500 0.296
Qwen2.5-VL-7B-Instruct 0.357 0.479 0.442 0.299 0.375 0.389
llava-v1.6-mistral-7b 0.067 0.165 0.088 0.051 0.250 0.099
llava-one-vision-7B 0.098 0.200 0.144 0.057 0.125 0.126
Phi-3.5-vision-instruct 0.245 0.375 0.291 0.187 0.375 0.280
SmolVLM-Instruct 0.128 0.224 0.164 0.100 0.250 0.163
RAG methods (top 1)
ColBERTv2+Llama-3.1-8B 0.257 0.529 0.471 0.428 0.775 0.429
M3DocRAG (ColPali+Qwen2-VL-7B) 0.340 0.605 0.546 0.520 0.625 0.506
MDocAgent (Ours) 0.341 0.612 0.540 0.527 0.750 0.517
RAG methods (top 4)
ColBERTv2+Llama-3.1-8B 0.349 0.599 0.491 0.485 0.875 0.491
M3DocRAG (ColPali+Qwen2-VL-7B) 0.426 0.660 0.595 0.542 0.625 0.554
MDocAgent (Ours) 0.438 0.675 0.592 0.581 0.875 0.578
Table 5. Performance comparison across different evidence source on LongDocURL.
Text Agent
You are a text analysis agent. Your job is to extract
key information from the text and use it to answer
the user’s question accurately.
Your tasks:
•Extract key details. Focus on the most important
facts, data, or ideas related to the question.
•Understand the context and pay attention to the
meaning and details.
•Use the extracted information to give a concise and
relevant response to the user’s question. Provide a
clear answer.Image Agent
You are an advanced image processing agent special-
ized in analyzing and extracting information from
images. The images may include document screen-
shots, illustrations, or photographs.
Your tasks:
•Extract textual information from images using Op-
tical Character Recognition (OCR).
•Analyze visual content to identify relevant details
(e.g., objects, patterns, scenes).
•Combine textual and visual information to provide
an accurate and context-aware answer to the user’s
question.
13

Summarizing Agent
You are tasked with summarizing and evaluating
the collective responses provided by multiple agents.
You have access to the following information:
•Answers : The individual answers from all agents.
Your tasks:
•Analyze : Evaluate the quality, consistency, and
relevance of each answer. Identify commonalities,
discrepancies, or gaps in reasoning.
•Synthesize : Summarize the most accurate and re-
liable information based on the evidence provided
by the agents and their discussions.
•Conclude : Provide a final, well-reasoned answer
to the question or task. Your conclusion should
reflect the consensus (if one exists) or the most
credible and well-supported answer.
Return the final answer in the following dictionary
format:
{”Answer”: Your final answer here }
Evaluation
Question :{question }
Predicted Answer :{answer }
Ground Truth Answer :{gt}
Please evaluate whether the predicted answer is cor-
rect.
• If the answer is correct, return 1.
• If the answer is incorrect, return 0.
Return only a string formatted as a valid JSON dic-
tionary that can be parsed using json.loads , for
example: {”correctness”: 1 }
A.5. Evaluation Metrics
The metric of all benchmarks is the average binary correct-
ness evaluated by GPT-4o. The evaluation prompt is given
in Section A.4. We use a python script to extract the result
provided by GPT-4o.
B. Additional Results
B.1. Fine-grained Performance of LongDocURL
We present the fine-grained performance of LongDocURL,
as illustrated in Table 5. Similar to MMLongBench, MDocA-
gent outperforms all LVLM baselines. When using the top
1 retrieval approach, though M3DocRAG performs slightly
better on Figure and ColBERTv2+Llama3.1-8B performs
slightly better on the type Others, MDocAgent show strong
performance in Layout, Text, Table and get the highest aver-
age accuracy. With the top 4 retrieval strategy, MDocAgent
improves its performance and reach the highest score in the
all categories.B.2. Experiments on different model backbones in
MDocAgent
Table 6 presents an ablation study evaluating the impact
of different LVLMs on the performance of our framework.
Three LVLMs: Qwen2-VL-7B-Instruct, Qwen2.5-VL-7B-
Instruct, and GPT-4o were integrated as the backbone model
for all agents except the text agent.
Qwen2.5-VL-7B-Instruct performs worse than Qwen2-
VL-7B-Instruct on PaperTab, PaperText, and FetaTab, with
both top-1 and top-4 retrieval. However, Qwen2.5-VL shows
an extremely marked improvement over Qwen2-VL on
MMLongBench, resulting higher average scores. MMLong-
Bench’s greater reliance on image-based questions might
explain Qwen2.5-VL’s superior performance on this bench-
mark, possibly indicating that Qwen2.5-VL is better at han-
dling visual question-answering tasks, but worse at handling
textual tasks.
Importantly, GPT-4o significantly outperforms both
Qwen2-VL and Qwen2.5-VL across all benchmarks. Re-
markably, GPT-4o’s top-1 performance surpasses even the
top-4 results of both Qwen models in almost all cases. This
substantial performance increase strongly suggests that our
framework effectively leverages more powerful backbone
models, showcasing its adaptability and capacity to benefit
from improvements in the underlying LVLMs.
B.3. Additional case studies
In Figure 4, the question requires identifying a reason from
a list that lacks explicit numbering and is accompanied by
images. ColBERT fails to retrieve the correct evidence page,
resulting ColBERT + Llama’s inability to answer the ques-
tion. Although ColPali correctly locates the evidence page,
M3DocRAG fails to get the correct answer. However, our
framework successfully identifies the correct answer (”Most
Beautiful Campus”) through the concerted efforts of all
agents. The general agent arrives at a preliminary answer and
the critical agent identifies critical textual clues (”Most Beau-
tiful Campus”) and corresponding visual elements (images
of the NTU campus). Image agent then refines the answer,
leveraging the critical information to correctly pinpoint the
description lacking people. Though text agent can’t find the
related information from the given context, information pro-
vided by the critical agent helps it to guess that the answer is
”Most Beautiful Campus”. The summarizing agent combines
these insights to arrive at the correct final answer.
In Figure 5, the question asks for Professor Lebour’s
degree. ColPali fails to retrieve the relevant page, render-
ing M3DocRAG ineffective. While ColBERT correctly re-
trieves the page, ColBERT + Llama still produces an in-
correct answer because it incorrectly adds ”F.G.S.” to the
answer, which is not a degree. MDocAgent, on the other
hand, correctly identifies the ”M.A. degree”. The general
agent provides an initial answer, and the critical agent identi-
14

MMLongBench LongDocUrl PaperTab PaperText FetaTab Avg
With top 1 retrieval
+Qwen2-VL-7B-Instruct 0.299 0.517 0.219 0.399 0.600 0.407
+Qwen2.5-VL-7B-Instruct 0.351 0.519 0.211 0.382 0.589 0.410
+GPT-4o [30] 0.420 0.595 0.293 0.474 0.716 0.500
With top 4 retrieval
+Qwen2-VL-7B-Instruct 0.315 0.578 0.278 0.487 0.675 0.467
+Qwen2.5-VL-7B-Instruct 0.389 0.566 0.277 0.454 0.671 0.471
Table 6. Performance comparison of using different backbone LVLMs in MDocAgent.
The reason that does not include any person in the corresponding figure is "Prestigious Scholarships and Comprehensive Financial Aid”.Most Beautiful CampusEvidencepages: 3Question
Among the top 10 reasons why NTU smart campus is for you, which is the reason that does not include any person in the corresponding figure?
Answer…the top 10 reasons why NTU smart campus is for you are not explicitly listed. However, the content provided includes a list of 10 reasons, but they are not numbered…Details:ΧConclusion:	Χ
General Agent:  The reason … is "Most Beautiful Campus." This reason is represented by a figure showing the NTU campus.
Details: ΧConclusion:	ΧColBERT: 1ColPali: 3
Critical Agent: Critical Info:•Te x t: Most beautiful Campus, •Image: NTU campusTe x t  A g e n t : I found the section "CONTENTS 2 TOP 10 REASONS WHY NTU SMART CAMPUS IS FOR YOU". However, the text does not explicitly list the top 10 reasons… The text mentions "Most Beautiful Campus" as a clue. This might be a reason why NTU smart campus is for you.Image Agent: The reason that does not include any person in the corresponding figure is "NTU campus without people.”Final Answer: The reason … is 'Most Beautiful Campus’. This reason focuses on the physical appearance of the NTU campus, as represented by a figure showing the campus without any people.Details:✓Conclusion:✓M3DocRAGColBERT + Llama 3.1 8B
Top-1 Page Retrieval
Ours
Figure 4. A Case study of MDocAgent compared with other two baselines. While only ColPali correctly retrieves the evidence page, neither
baseline method identifies the correct answer. Our method, through critical information sharing and specialized agent collaboration, correctly
pinpoints the ”Most Beautiful Campus” as the only reason without a corresponding image containing people.
fies the ”M.A.” designation in both text and image. Based on
the clue, the text agent adds a more detailed explanation, and
the image agent directly uses the clue as its answer. Finally,
the summarizing agent synthesizes the results to provide the
verified answer.
These two cases highlight MDocAgent’s resilience to
imperfect retrieval, demonstrating the effectiveness of col-
laborative multi-modal information processing and the im-
portance of the general-critical agent’s guidance in achieving
high accuracy even with potentially insufficient or ambigu-
ous information.
15

The image does not provide informationabout the degree of Prof. Lebour.M.A.Evidencepages: 1Question
What degree does LEBOUR have?
AnswerAccording to the text, G. A. Lebour has the following degrees:1.M.A. (Master of Arts)2.F. G. S. (Fe l l ow  of the  Ge ol ogi cal  Soci e ty)Details: ✓Conclusion:	Χ
General Agent:  Based on the information provided in the text and the image, G. A. Lebour holds the degree of M.A. and F.G.S.
Details: ΧConclusion:	ΧColBERT: 1ColPali: 5
Critical Agent: Critical Info:•Te x t : M . A .•Image: M.A.Te x t  A g e n t : Based on the extracted key details, I found that LEBOUR has an M.A. degree. The text states: "By G. A. LEBOUR, M.A., F.G.S., PROFESSOR OF GEOLOGY IN THE DURHAM COLLEGE OF SCIENCE, NEWCASTLE-UPON-TYNE." Here, "M.A." stands for Master of Arts, indicating the degree held by LEBOUR.Image Agent: Based on the information given from the clue, it can be inferred that Prof. Lebour holds a Master of Arts (M.A.) degree.Final Answer: Prof. Lebour holds a Master of Arts (M.A.) degree.Details:✓Conclusion:✓M3DocRAGColBERT + Llama 3.1 8B
Top-1 Page Retrieval
OursFigure 5. A Case study of MDocAgent compared with other two RAG-method baselines. In this case, ColPali fails to retrieve the correct
evidence page, hindering M3DocRAG. While ColBERT succeeds in retrieval, the ColBERT + Llama baseline still provides an incorrect
answer. Only our multi-agent framework, through precise critical information extraction and agent collaboration, correctly identifies the
M.A. degree.
16