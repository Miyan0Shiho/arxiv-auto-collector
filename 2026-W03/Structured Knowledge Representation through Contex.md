# Structured Knowledge Representation through Contextual Pages for Retrieval-Augmented Generation

**Authors**: Xinze Li, Zhenghao Liu, Haidong Xin, Yukun Yan, Shuo Wang, Zheni Zeng, Sen Mei, Ge Yu, Maosong Sun

**Published**: 2026-01-14 11:44:31

**PDF URL**: [https://arxiv.org/pdf/2601.09402v1](https://arxiv.org/pdf/2601.09402v1)

## Abstract
Retrieval-Augmented Generation (RAG) enhances Large Language Models (LLMs) by incorporating external knowledge. Recently, some works have incorporated iterative knowledge accumulation processes into RAG models to progressively accumulate and refine query-related knowledge, thereby constructing more comprehensive knowledge representations. However, these iterative processes often lack a coherent organizational structure, which limits the construction of more comprehensive and cohesive knowledge representations. To address this, we propose PAGER, a page-driven autonomous knowledge representation framework for RAG. PAGER first prompts an LLM to construct a structured cognitive outline for a given question, which consists of multiple slots representing a distinct knowledge aspect. Then, PAGER iteratively retrieves and refines relevant documents to populate each slot, ultimately constructing a coherent page that serves as contextual input for guiding answer generation. Experiments on multiple knowledge-intensive benchmarks and backbone models show that PAGER consistently outperforms all RAG baselines. Further analyses demonstrate that PAGER constructs higher-quality and information-dense knowledge representations, better mitigates knowledge conflicts, and enables LLMs to leverage external knowledge more effectively. All code is available at https://github.com/OpenBMB/PAGER.

## Full Text


<!-- PDF content starts -->

Structured Knowledge Representation through Contextual Pages for
Retrieval-Augmented Generation
Xinze Li1, Zhenghao Liu1*, Haidong Xin1, Yukun Yan2*,
Shuo Wang2,Zheni Zeng3,Sen Mei2,Ge Yu1,Maosong Sun2
1School of Computer Science and Engineering, Northeastern University, China
2Department of Computer Science and Technology, Institute for AI, Tsinghua University, China
3School of Intelligent Science and Technology, Nanjing University, China
Abstract
Retrieval-Augmented Generation (RAG) en-
hances Large Language Models (LLMs) by
incorporating external knowledge. Recently,
some works have incorporated iterative knowl-
edge accumulation processes into RAG mod-
els to progressively accumulate and refine
query-related knowledge, thereby construct-
ing more comprehensive knowledge representa-
tions. However, these iterative processes often
lack a coherent organizational structure, which
limits the construction of more comprehensive
and cohesive knowledge representations. To ad-
dress this, we propose PAGER, a page-driven
autonomous knowledge representation frame-
work for RAG. PAGER first prompts an LLM
to construct a structured cognitive outline for
a given question, which consists of multiple
slots representing a distinct knowledge aspect.
Then, PAGER iteratively retrieves and refines
relevant documents to populate each slot, ulti-
mately constructing a coherent page that serves
as contextual input for guiding answer gener-
ation. Experiments on multiple knowledge-
intensive benchmarks and backbone models
show that PAGER consistently outperforms
all RAG baselines. Further analyses demon-
strate that PAGER constructs higher-quality
and information-dense knowledge represen-
tations, better mitigates knowledge conflicts,
and enables LLMs to leverage external knowl-
edge more effectively. All code is available at
https://github.com/OpenBMB/PAGER.
1 Introduction
Retrieval-Augmented Generation (RAG) models
enhance the performance of Large Language Mod-
els (LLMs) by retrieving relevant documents and
using them as context inputs (Guu et al., 2020;
Lewis et al., 2020). Recent studies have begun to
focus on knowledge refinement to enhance RAG
models (Wu et al., 2025; Xu et al., 2023b; Zhu
*indicates corresponding author.
Question:When did the location of the basilica which is named for the same saint that the Bremen Cathedral is named for become a country?
LLMCognitive OutlineThe title of slot-1 [Slot-1to be filled]…The title of slot-2 [Slot-2to be filled]Page Initialization
Generate a  sub-query[Slot-1 withevidence]…[Slot-tto be filled]…[Slot-nto be filled]Sub-query forthe Slot-t
Retrieval top-k document
Generate a evidence and fill itinto slot-t
Final Page
On 11February 1929.Use the page to assist the LLM in answering the question
Iterative Knowledge CompletionGenerate a cognitive outline
[Slot-1withevidence]…[Slot-twithevidence]…[Slot-nto be filled]
[Slot-1withevidence]…[Slot-twith evidence]…[Slot-nwithevidence]Figure 1: The Pipeline of Our PAGER Model. PAGER
initializes a blank structured page with multiple slots
and preliminarily fills the retrieved external knowledge
into the corresponding slots, thereby constructing more
comprehensive and coherent knowledge representations.
et al., 2025), which typically incorporate special-
ized modules to refine retrieved documents into
either unstructured summaries (Vig et al., 2022) or
structured knowledge (Li et al., 2025b), enabling
LLMs to utilize the retrieved knowledge more ef-
fectively. However, these methods primarily focus
on tailoring the representation of retrieved knowl-
edge while failing to fully leverage the structure of
knowledge representations and the retriever to ac-
tively acquire and complete missing knowledge,
which limits their application to complex ques-
tions (Guo et al., 2025; Jiang et al., 2025).
To achieve more effective knowledge acquisition
and utilization, some studies have begun incorpo-
rating iterative knowledge accumulation processes
into RAG models (Trivedi et al., 2023; Jin et al.,
2025a). These methods design iterative knowledgearXiv:2601.09402v1  [cs.CL]  14 Jan 2026

acquisition mechanisms that leverage the strong
reasoning capabilities of LLMs to progressively
obtain new knowledge and refine it, thereby incor-
porating more information to construct more com-
prehensive knowledge representations (Wang et al.,
2025a; Li et al., 2025a). However, these meth-
ods lack modeling of coherent knowledge struc-
tures, making it difficult for models to construct
comprehensive and consistent knowledge represen-
tations to guide accurate answers. To overcome
this challenge, models need to leverage their in-
trinsic knowledge structures to actively acquire
relevant documents and construct knowledge rep-
resentations aligning better with cognitive frame-
works (Bruner, 1966).
In this paper, we introduce PAGER, a page-
driven autonomous knowledge representation
framework which constructs contextual pages for
RAG modeling, allowing LLMs to leverage their
reasoning and planning capability to organize and
exploit information more effectively. Specifically,
as shown in Figure 1, PAGER first prompts the
LLM to draw on its parametric knowledge to con-
struct a structured cognitive outline for the target
question. This outline consists of multiple slots,
each representing a distinct aspect of the potentially
relevant knowledge needed to answer the question.
Then PAGER employs an iterative knowledge com-
pletion mechanism to iteratively retrieve supporting
documents for each slot, refine them into concise
knowledge evidence, and fill the corresponding slot
in the page. This iterative process continues until
all slots are filled with the corresponding knowl-
edge evidence. Finally, PAGER uses this structured
page as contextual knowledge to guide the LLM to
answer the given question.
Experimental results demonstrate that PAGER
consistently outperforms all baseline methods on
knowledge-intensive tasks of varying scenarios and
different backbone models, demonstrating its effec-
tiveness. Further analysis shows that the cognitive
architecture designed by PAGER can more effec-
tively guide LLMs to acquire and organize knowl-
edge, yielding coherent and comprehensive knowl-
edge representations that support question answer-
ing. Furthermore, compared to other knowledge
representations, the structured knowledge pages ex-
hibit richer information content and superior qual-
ity, demonstrating the effectiveness of the knowl-
edge representations constructed by PAGER. Mean-
while, by organizing external knowledge into struc-
tured pages, PAGER effectively mitigates knowl-edge conflicts within LLMs and enables them to
utilize the knowledge more effectively.
2 Related Work
Large language models (LLMs) (Yang et al., 2025;
Dubey et al., 2024) have demonstrated strong capa-
bilities across a wide range of tasks (Trivedi et al.,
2023; He et al., 2021). However, LLMs typically
suffer from hallucination, which can lead to in-
correct responses (Jiang et al., 2023; Xu et al.,
2023a). To mitigate this issue, existing studies
employ Retrieval-Augmented Generation (RAG)
models, which retrieve relevant documents for a
given question and incorporate them as input con-
text, enabling LLMs to access external knowledge
and generate accurate answers (Lewis et al., 2020;
Guu et al., 2020). However, the conflict between
retrieved knowledge and parametric memory hin-
ders the LLM’s ability to reliably identify critical
facts from retrieved documents, limiting the effec-
tiveness of RAG models (Huo et al., 2025).
To address these challenges, some studies have
focused on refining knowledge to enhance LLMs’
ability to capture crucial evidence from retrieved
documents (Wu et al., 2025). Some studies
use query-focused summarization methods (Xu
et al., 2023b) to condense retrieved documents into
shorter forms, thereby reducing noise and enhanc-
ing relevance. However, the generated summaries
may fail to capture the relationships among re-
trieved documents, leading to the loss of supporting
evidence that is distributed across these documents.
Furthermore, several studies have begun explor-
ing the transformation of retrieved documents into
structured knowledge, such as graphs (Zhu et al.,
2025) or tables (Li et al., 2025b), thereby better cap-
turing knowledge from different documents. Never-
theless, these approaches fail to leverage structural
forms to complete the knowledge representation,
despite the availability of retrieval tools.
Recently, some studies have shifted toward ac-
quiring and accumulating additional knowledge
to enhance the process of refined knowledge rep-
resentation (Trivedi et al., 2023). These meth-
ods typically iteratively generate sub-queries to
target existing knowledge gaps and retrieve rele-
vant documents, and then apply different knowl-
edge refinement strategies to construct diverse re-
fined knowledge representations (Wang et al., 2024,
2025a). For example, Wang et al. (2025a) refine
knowledge into note-based representations, and Li

et al. (2025a) adopt a reasoning-in-document mech-
anism that integrates knowledge into the reasoning
trajectory. However, these methods fail to effec-
tively model coherent knowledge structures, which
are essential for constructing more comprehensive
and cohesive knowledge representations (Bruner,
1966). To overcome this limitation, PAGER lever-
ages structural pages to integrate more relevant
knowledge and organize it into a more cognitively
structured format.
3 Methodology
In this section, we introduce PAGER, a page-driven
autonomous knowledge searching, accumulation
and representation framework. We first describe the
preliminaries of RAG models enhanced with struc-
tured knowledge representations (Sec. 3.1) and then
describe the construction of coherent and compre-
hensive knowledge representations via autonomous
page construction (Sec. 3.2).
3.1 Preliminaries: Enhancing RAG Models
with Structured Knowledge
Recent RAG approaches incorporate structured
knowledge representations to refine retrieved in-
formation (Li et al., 2025b). Different from them,
PAGER aims to guide an LLM to iteratively re-
trieve relevant documents and complete knowledge
representations by autonomously interacting with
retrieval tools, thereby enabling more accurate and
reliable query answering.
Knowledge Refinement with Structured Rep-
resentations.Given a query q, existing meth-
ods (Li et al., 2025b) first retrieve the top- krelevant
documents:
D=Search(q, k),(1)
where D={d 1, . . . , d k}. Subsequently, the LLM
is prompted to refine the retrieved documents into
a query-conditioned structured knowledge repre-
sentation result O. The refined knowledge Ocan
take various forms, such as tables, graphs, or other
structured abstractions. The structured knowledge
Ois then provided as contextual input to the LLM
for answer generation:
y=M 
Instruct QA(O, q)
,(2)
where Mindicates the LLM and Instruct QAde-
notes the instruction template that guides the LLM
to produce an answer grounded in the structuredknowledge. Despite their effectiveness, these ap-
proaches primarily treat structured knowledge as
a post-retrieval refinement artifact. They do not
fully exploit structured representations as an ac-
tive medium for completing knowledge, even when
external retrieval tools are available.
Knowledge Completion Mechanism.In con-
trast to prior methods that rely on static structures
for knowledge refinement, we propose PAGER,
which introduces a cognitively inspiredpagestruc-
ture to facilitate knowledge completion for the
questions and obtain a structured pagep:
p=F page 
q,Search(·)
,(3)
where Fpage denotes a page construction func-
tion that autonomously accumulates and organizes
knowledge through iterative interactions with re-
trieval tools, as detailed in Sec. 3.2. The LLM then
performs reasoning over the resulting structured
pagepto derive the final answery:
y=M 
Instruct QA(p, q)
.(4)
3.2 Knowledge Representation via Structured
Page Construction
In this subsection, we provide a detailed description
of the structural contextual pages for RAG models.
We first introduce the page initialization mecha-
nism in PAGER, and then describe how PAGER it-
eratively completes knowledge to construct a struc-
tured page.
Page Initialization.Given a query q, we initial-
ize the page solely based on the intrinsic reasoning
and planning capabilities of the LLM, without in-
corporating any retrieved external knowledge.
Specifically, we design an instruction template
Instruct outline to prompt the LLM to analyze the
query qand generate a coherent reasoning trace r.
Conditioned on this reasoning process, the model
then constructs an initial pagep 0:
r, p 0=M 
Instruct outline (q)
.(5)
The construction of p0is guided by the continuous
reasoning trace r, which captures the LLM’s inter-
nal analytical process and semantic understanding
of the query q. Nevertheless, such a reasoning-
driven outline remains abstract and may lack con-
crete supporting evidence (Wang et al., 2024), moti-
vating subsequent retrieval and knowledge accumu-
lation to progressively fill these blank slots. Thus,

Question:When did the location of the basilica which is named for the same saint that the Bremen Cathedral is named for become a country?
Page Initialization
Sub-Query-t: What are Ruth Leuwerik'sbackground and career highlights?Iterative KnowledgeCompletion
Generate a sub-query for the slot-t
Generate a page outline based on the question
The (t-1)-thPage
The t-thPage
The Final Page## Identifying the Shared Patron Saint Evidence-1: The shared patron saint after whom both the Bremen Cathedral and another basilica are named is St. Peter. Bremen Cathedral, located in the market square in the center of Breme………## The Historical Development of Location [Slot-t]……## When the Location Became a Country[Slot-n]## Identifying the Shared Patron Saint Evidence-1: The shared patron saint after whom both the Bremen Cathedral and another basilica are named is St. Peter. Bremen Cathedral, locate………## The Historical Development of LocationEvidence-t: The territory that would become Vatican City has a long and complex history………## When the Location Became a Country[Slot-n]## Identifying the Shared Patron Saint Evidence-1: The shared patron saint after whom both the Bremen Cathedral and anotherbasilica………## The Historical Development of LocationEvidence-t: The territory that would become Vatican City has a long and complex history………## When the Location Became a CountryEvidence-n: Vatican City officially became an independent country in 1929 with the signing of…Retrieve the top-k documentsTop-K DocumentsCorpusRefine the retrieved knowledge into knowledge evidence-t and fill it into the [slot-t]# The Historical Emergence of a Country Linked to a Shared Patron Saint## Identifying the Shared Patron Saint [Slot-1]…… ## The Historical Development of the Location [Slot-t]……## When the Location Became a Country [Slot-n]Initialized Page
Answering Questions Based on the Page
Question+
Final Page
Response: …Vatican City became an independent country in 1929 with the signing of the Lateran Pacts. <answer> 1929 </answer>LLM
LLMLLMLLMFigure 2: The Illustration of Our PAGER Model.
the initial page p0can be formulated as a struc-
tured cognitive outline comprising nexplicitly de-
fined blank slots, which are reserved for subsequent
knowledge completion:
p0=Outline([b] 1, . . . ,[b] n),(6)
where [b]idenotes the i-th placeholder corre-
sponding to a specific knowledge component re-
quired to answer the query. Through this process,
PAGER establishes a high-level cognitive structure
for the query, decomposing it into nthematically
organized knowledge slots that specify the neces-
sary background and supporting information.
Iterative Knowledge Completion.PAGER
then performs an iterative knowledge refinement
process to fill the missing slots[b] 1:n.
At each iteration step t(1≤t≤n ), the (t−1) -
th page is represented as:
pt−1=Outline(s 1, . . . , s t−1,[b] t, . . . ,[b] n).
(7)
PAGER obtains the page ptby updating the (t−1) -
th pagep t−1through filling thet-th slot[b] t:
stfill− →[b] t,(8)
where stis a knowledge evidence used to fill the
missing slot [b]t. Through this progressive itera-
tive mechanism, external knowledge is gradually
refined and incorporated into the knowledge pages
until all nslots are filled with supporting evidence,
yielding the final structured pagep n.To gather supporting evidence stfor the slot
[b]t, we first prompt the LLM to generate a sub-
query qttailored to gather necessary supporting
evidence to fill thet-th slot[b] tinpt−1:
qt=M(Instruct query(pt−1, q)),(9)
where Instruct query is an instruction template de-
signed to guide the LLM to focus on the blank slot
[b]tand produce a sub-query for retrieving rele-
vant topical knowledge based on the current page
pt−1and the original query q. Next, the retriever
model is employed to search kdocuments Dtbased
on the generated queryq t:
Dt=Search(q t, k),(10)
and finally, the LLM is prompted using Instruct fill
to generate the knowledge evidence stbased on the
retrieved documentsD t:
st=M(Instruct fill(q, p t−1, Dt, qt)).(11)
4 Experimental Methodology
In this section, we describe the datasets, evaluation
metrics, and baselines, followed by the implementa-
tion details of our experiments. More experimental
details are provided in Appendix A.2.
Dataset.Following previous work (Li et al.,
2025a; Song et al., 2025), we evaluate our method
on both multi-hop and single-hop QA bench-
marks. Specifically, we select HotpotQA (Yang

et al., 2018), 2WikiMultiHopQA (Ho et al.,
2020), MuSiQue (Trivedi et al., 2022), and Bam-
boogle (Press et al., 2023) for multi-hop tasks,
while using NQ (Kwiatkowski et al., 2019) and Am-
bigQA (Min et al., 2020) for single-hop tasks. For
evaluation, we randomly sample 2,000 instances
from the development set of each dataset, except
for Bamboogle (Press et al., 2023), where we uti-
lize the entire test set (125 instances) due to its
limited size.
Evaluation Metrics.We follow existing
work (Sun et al., 2025; Song et al., 2025) to utilize
Cover Exact Match as the evaluation metric.
Baselines.In our experiments, we compare
PAGER with multiple baseline models, including a
Vanilla LLM, one-pass retrieval RAG models, itera-
tive retrieval RAG models, and iterative knowledge
representation construction RAG models.
For the Vanilla LLM, we directly feed the query
to the LLM and ask it to generate the answer with-
out any external knowledge. For one-pass retrieval
RAG models, we adopt Vanilla RAG and Struc-
tRAG (Li et al., 2025b), where the former utilizes
retrieved documents as contextual input to assist
the LLM to answer the question, and the latter de-
signs a router to refine documents into structured
knowledge representations as input context. For
iterative retrieval RAG models, we employ Iter-
RetGen (Shao et al., 2023) and IRCoT (Trivedi
et al., 2023) to accumulate knowledge for assist-
ing the LLM. These methods interleave retrieval
with the generation process, utilizing intermedi-
ate generated content to guide subsequent retrieval
and directly leveraging the retrieved documents
to facilitate the generation process. Besides, we
adopt RAT (Wang et al., 2024), Search-o1 (Li et al.,
2025a), and DeepNote (Wang et al., 2025a) as iter-
ative knowledge representation construction RAG
models, which not only accumulate knowledge but
also refine it into distinct forms of knowledge repre-
sentations. Specifically, RAT directly prompts the
LLM to generate CoT-style knowledge representa-
tions and iteratively retrieves external knowledge
to refine and revise each reasoning step. Search-o1
performs adaptive retrieval during reasoning and
applies a Reason-in-Document mechanism to refine
the retrieved documents, integrating them into the
CoT-style representation. DeepNote compresses re-
trieved knowledge into note-based representations
and iteratively acquires additional information to
update these notes, ultimately constructing an opti-
mal note to assist the LLM.Implementation Details.In our experiments,
we employ Qwen3-32B1(Yang et al., 2025), and
Llama-3.1-70B-Instruct2(Dubey et al., 2024) as
backbone models. We follow FlashRAG (Jin et al.,
2025b) to use Wikipedia as the retrieval corpus and
adopt Qwen3-Embedding-0.6B (Zhang et al., 2025)
as the embedding model and employ FAISS (John-
son et al., 2019) to build indexes for retrieval. Dur-
ing retrieval, the top- 5ranked documents are re-
tained for all RAG models. We further employ the
vLLM inference framework (Kwon et al., 2023) to
accelerate inference across all models.
5 Experimental Results
In this section, we first evaluate the performance
of PAGER across different models and datasets.
Subsequently, we conduct ablation studies to ana-
lyze the effectiveness of different functional com-
ponents in PAGER. Then, we investigate the ef-
fectiveness of the constructed pages in knowledge
representation. Finally, we explore the impact of
different representations on knowledge utilization
in LLMs.
5.1 Overall Performance
As shown in Table 1, we compare the overall per-
formance of PAGER with various baseline methods
across a range of knowledge-intensive tasks.
Overall, PAGER demonstrates its effectiveness
by outperforming all baseline models, achieving
improvements exceeding 2%. Notably, PAGER
consistently shows improvements across various
tasks and backbone LLMs, underscoring its robust
generalization ability. When compared to Struc-
tRAG, PAGER delivers an average performance
boost of over 5%, indicating that the page format
serves as an effective knowledge representation
mechanism, enabling autonomous interaction with
external knowledge to build more comprehensive
knowledge structures. Furthermore, in compari-
son to iterative retrieval-based methods like IRCoT
and Iter-RetGen, PAGER achieves an improvement
of over 9%, emphasizing its role in more effec-
tively refining and organizing retrieved evidence.
Additionally, PAGER surpasses DeepNote, which
facilitates better integration of retrieved knowledge
by summarizing them as notes, suggesting that the
construction of structured knowledge pages allows
1https://huggingface.co/Qwen/Qwen3-32B
2https://huggingface.co/meta-llama/Llama-3.1-70B-
Instruct

Methods HotpotQA 2WikiMQA MuSiQue Bamboogle NQ AmbigQA Avg.
Qwen3-32B
Vanilla LLM 28.6 31.8 8.2 44.0 36.2 32.9 30.3
Vanilla RAG 42.9 36.5 13.5 52.0 54.6 54.8 42.4
StructRAG (2025b) 42.5 31.5 12.8 44.8 55.1 54.5 40.2
IRCoT (2023) 45.1 45.5 14.7 32.0 52.2 51.8 40.2
Iter-RetGen (2023) 43.2 40.2 14.5 37.656.555.4 41.2
RAT (2024) 43.7 45.0 16.1 55.2 53.2 55.5 44.8
Search-o1 (2025a) 47.6 47.0 22.9 33.6 49.2 51.5 42.0
DeepNote (2025a) 48.4 47.2 17.2 39.2 55.6 55.9 43.9
PAGER50.6 57.4 23.0 62.4 56.5 56.4 51.1
Llama3.1-70B-Instruct
Vanilla LLM 37.7 41.5 14.1 57.6 50.3 48.8 41.7
Vanilla RAG 48.2 41.9 19.1 56.8 55.7 57.0 46.5
StructRAG (2025b) 49.6 44.1 19.4 57.657.458.2 47.7
IRCoT (2023) 35.2 37.0 6.2 11.2 49.9 50.3 31.6
Iter-RetGen (2023) 41.1 34.5 11.9 32.8 53.0 52.3 37.6
RAT (2024) 48.9 40.2 20.8 58.4 53.6 56.4 46.2
Search-o1 (2025a) 50.255.422.9 58.4 51.6 53.8 48.3
DeepNote (2025a) 51.7 48.9 20.8 58.4 56.6 58.4 49.1
PAGER52.454.9 24.3 62.456.460.0 51.7
Table 1: Overall Performance of Different RAG Models. Thebestand second best results are highlighted.
Methods HotpotQA 2WikiMQA MuSiQue Bamboogle NQ AmbigQA Avg.
Qwen3-32B
PAGER (Parallel Filling) 45.9 43.8 18.8 59.2 56.4 57.046.9
PAGER50.6 57.4 23.0 62.4 56.556.4 51.1
w/o IterRetrieval 44.8 42.2 16.9 52.0 54.6 55.1 44.3
w/o Initilization 46.2 45.2 15.9 48.0 55.2 55.9 44.4
Llama3.1-70B-Instruct
PAGER (Parallel Filling) 47.4 43.4 18.6 58.457.259.5 47.4
PAGER52.4 54.9 24.3 62.456.4 60.0 51.6
w/o IterRetrieval 47.6 41.6 18.1 56.0 56.3 58.1 46.3
w/o Initilization 51.1 50.5 21.4 60.0 55.9 57.9 49.5
Table 2: Ablation Study. Thebestand second best results are highlighted.
PAGER to generate more effective knowledge rep-
resentations, thus providing stronger support for
LLMs in answering questions.
5.2 Ablation Study
In this subsection, we conduct ablation studies to
evaluate the effectiveness of different components
in PAGER.
In the experiments, we compare PAGER with
three ablated models: PAGER (Parallel Filling) si-
multaneously generates sub-queries for all missing
slots in the initial page, conducts parallel retrieval
and then refines these retrieved documents to fill
into the corresponding slots; PAGER (w/o IterRe-
trieval) conducts a single-pass retrieval based on
the given query and filling the intial pages based
on these retrieved docuemtns; PAGER (w/o Initial-
ization) iteratively refines the retrieved documents
into concise summaries, concatenating them until
the aggregated summaries are sufficient to answer
the question. This variant removes the page initial-
ization stage, which typically generates the outlinebased on the internal planning ability of LLMs.
As shown in Table 2, PAGER consistently out-
performs PAGER (Parallel Filling), demonstrating
the effectiveness of the iterative knowledge com-
pletion mechanism in PAGER. Unlike parallel re-
trieval, PAGER generates queries conditioned on
the page state at the previous step, which enables
more tailored queries for retrieving necessary infor-
mation and helps avoid homogeneous content to fill
in different page slots. Furthermore, compared to
PAGER (w/o IterRetrieval), PAGER exhibits con-
sistent performance gains across different datasets
and backbone models, indicating that iterative re-
trieval can incorporate more essential information
to better answer the given query. Finally, compared
with PAGER (w/o Initialization), PAGER achieves
further improvements, highlighting the critical role
of the cognitive outline generated by leveraging the
reasoning capability of LLMs. This outline enables
PAGER to effectively organize and complete the
required knowledge within a page.

Search-o1RATDeepNote PAGER500550600650700#T okens
Length CoverEM
1020304050
Performance
(a) Representation Length.
Refinement Logic       
Accuracy     Structure12345Search-o1
PAGERRAT
DeepNote (b) GLM Score.
Figure 3: The Quality of Knowledge Representations
Constructed by Different Methods.
5.3 The Effectiveness of Constructed Pages in
Knowledge Representation
In this section, we investigate the effectiveness of
PAGER in knowledge representation by analyzing
its constructed pages. Specifically, we examine
both the quality of the constructed pages produced
by PAGER and how these pages facilitate knowl-
edge utilization by LLMs. In this experiment, we
adopt Qwen3-32B as the backbone model and con-
duct evaluations on the MuSiQue and Bamboogle.
The Quality of PAGER in Representing Re-
trieved Knowledge.Figure 3 presents the length
statistics and quality assessments of the constructed
knowledge representations.
As shown in Figure 3(a), we compute the aver-
age token length of knowledge representations gen-
erated by different methods to analyze the trade-off
between the length and QA performance. Com-
pared with RAT and Search-o1, PAGER achieves
superior performance with longer knowledge rep-
resentations, indicating that PAGER enhances QA
performance by incorporating richer information
to form more comprehensive knowledge represen-
tations. Compared with DeepNote, PAGER attains
significantly better performance while reducing the
length of the knowledge representation. This sug-
gests that PAGER exhibits higher knowledge den-
sity than DeepNote, enabling it to deliver more
effective information within the constructed page.
We further evaluate the overall quality of the
knowledge representations generated by different
models. As shown in Figure 3(b), we employ a
strong closed-source LLM, GLM-4.5, as the eval-
uator. Using the prompt templates provided in
Appendix A.9, we assess the knowledge represen-
tations along four dimensions: accuracy, logical-
ity, structure, and degree of knowledge refinement,
with each dimension rated on a scale from 0to
N/A FirstSecond Third Fourth
Dropped Filled Slot22232425262728CoverEM
26.4
25.5
25.1
24.6
22.425.4PAGER
PAGER w/o Outline(a) MuSiQue.
N/A FirstSecond Third Fourth
Dropped Filled Slot40.042.545.047.550.052.5CoverEM
52.0
50.0
48.0
42.046.048.0 (b) Bamboogle.
Figure 4: Slot Ablation Studies on the Constructed Page.
“N/A” denotes the complete page with no filled slots
removed. “First”, “Second”, “Third”, and “Fourth” de-
note the variants in which the First, Second, Third, and
Fourth filled slots are removed, respectively. PAGER
(w/o Outline) denotes a variant where the cognitive out-
line structure is removed from the complete page and is
used as a baseline for comparison.
5. The evaluation results demonstrate that PAGER
consistently outperforms other models across all
dimensions, with particularly notable advantages
in structure and logical consistency. These findings
further validate the effectiveness of the pages con-
structed by PAGER, which receive higher scores
from the GLM evaluator, reflecting the coherent
organization and comprehensive coverage of the
retained knowledge.
Effectiveness of Completed Knowledge in Dif-
ferent Slots of the Page.As shown in Figure 4, we
further analyze the effectiveness of the knowledge
filled in different slots of the page. We collect pages
containing four slots as seed pages (As shown in
Appendix A.7, the pages constructed by the Qwen3-
32B predominantly feature 4slots). Then, we re-
move the first, second, third, and fourth filled slots
to construct incomplete pages, respectively. Finally,
we feed these four distinct incomplete pages into
the model to evaluate the performance.
Overall, compared with the completed page
PAGER (N/A), removing any single filled slots
from the page leads to performance degradation,
indicating that all filled knowledge is necessary
to support LLMs in answering the query. As the
removed filled slot shifts from the first slot to the
fourth slot, the model performance degrades more
substantially, suggesting that knowledge filled in
later slots is more critical for question answering.
One possible reason is that our iterative comple-
tion method tends to acquire increasingly necessary
knowledge in later slots, which aligns with the rea-
soning process of LLMs, where later reasoning
steps often involve deeper inference to answer the

2WikiMQA Bamboogle1520253035404550Document Overlap (%)47.546.2
16.425.228.432.5
23.123.9(a) Document Overlap.
2WikiMQA Bamboogle0246810121416Information Gain * 10^-22.19.5
6.611.2
8.013.8
3.94.6PAGER
DeepNote
Search-o1
RAT (b) Information Gain.
Figure 5: Effects of Different Knowledge Representa-
tions on Retrieval and Generation Modules of RAG.
query. Furthermore, compared with PAGER (N/A),
PAGER (w/o Outline) also exhibits reduced per-
formance when the removed knowledge is from
slots later than the second one. This observation
further indicates that the cognitive structure plays a
critical role in guiding the knowledge construction
process, thereby verifying the effectiveness of our
page initialization module.
5.4 The Impact of Different Representations
on Knowledge Utilization in LLMs
In this section, we investigate how different knowl-
edge representations affect knowledge utilization
in LLMs. We adopt Qwen3-32B as the backbone
model for all experiments.
As shown in Figure 5, we evaluate the effec-
tiveness of constructed knowledge representations
across different RAG modules. First, we analyze
the document overlap between successive retrieval
steps using the Jaccard similarity. As illustrated in
Figure 5(a), the results show that PAGER exhibits
substantially lower document overlap than Deep-
Note and RAT. This suggests that the page-based
representation serves as an effective format for re-
ducing retrieval redundancy across retrieval steps,
thereby encouraging the model to incorporate more
diverse knowledge instead of repeatedly retrieving
similar information. Following prior work (Wang
et al., 2025b), we further compute the information
gain of different knowledge representations, as re-
ported in Figure 5(b). Information gain quantifies
the contribution of a knowledge representation to
correct answer generation, with details provided
in Appendix A.3. The results demonstrate that
the pages constructed by PAGER achieve higher
knowledge information gain than other methods.
This indicates that the information encoded in struc-
tured page representations can better guide LLMs
toward generating accurate answers.Methods 2WikiMQA Bamboogle HotpotQA
Knowledge Conflict
Vanilla LLM 100.0 100.0 100.0
Vanilla RAG 61.1 90.9 82.2
DeepNote 64.5 58.2 82.2
RAT 72.892.784.9
Search-o1 56.3 52.7 82.6
PAGER73.483.687.6
Knowledge Utilization
Vanilla LLM 0.0 0.0 0.0
Vanilla RAG 25.1 21.4 27.2
DeepNote 39.2 24.3 34.9
RAT 32.0 25.7 27.2
Search-o1 42.7 18.6 33.5
PAGER49.9 45.7 35.8
Table 3: Performance of Different RAG Models under
Different Testing Scenarios.
Next, we evaluate the impact of different knowl-
edge representations on knowledge conflicts and
knowledge utilization in LLMs. As shown in Ta-
ble 3, we design two evaluation scenarios: knowl-
edge conflict and knowledge utilization. For the
knowledge conflict scenario, we select samples
from the evaluation dataset for which the LLM can
generate correct answers solely based on its para-
metric knowledge, aiming to assess the denoising
capability of RAG systems. For the knowledge uti-
lization scenario, we construct the evaluation set by
selecting samples on the LLM’s answers must rely
on external knowledge. The evaluation results show
that both PAGER and RAT perform more effec-
tively in the knowledge conflict scenario, indicat-
ing that incorporating the raw reasoning of LLMs
to organize retrieved knowledge can alleviate con-
flicts between internal and external knowledge. In
contrast, DeepNote and Search-o1 outperform RAT
in the knowledge utilization setting, suggesting that
incorporating more retrieved knowledge provides
greater potential for correcting factual errors in
memorized knowledge. Notably, PAGER achieves
the best performance in the knowledge utilization
scenario while maintaining performance compara-
ble to RAT under the knowledge conflict setting.
These results demonstrate that the constructed page
representation offers a more tailored balance be-
tween mitigating knowledge conflicts and enhanc-
ing knowledge utilization.
6 Conclusion
This paper presents PAGER, a page-driven au-
tonomous knowledge representation framework
for RAG. By constructing a structured cognitive
outline and iteratively filling it with retrieved evi-

dence, PAGER organizes external knowledge into
the structed page representation that better supports
LLM to answer the questions. Experimental results
show that PAGER consistently outperforms base-
lines across multiple tasks and backbone models.
Limitations
Although PAGER achieves superior performance
across multiple datasets, the page construction pro-
cess introduces additional latency. Specifically, to
ensure the logical coherence and completeness of
the constructed pages, PAGER employs an iterative
slot-filling mechanism. This iterative process in-
evitably incurs extra computational overhead, lead-
ing to increased inference latency. Moreover, we
also explore a variant of PAGER with parallel fill-
ing to mitigate this issue. However, experimen-
tal results indicate that the iterative design is still
necessary to maintain the effectiveness of PAGER
when answering multi-hop QA queries. Therefore,
the trade-off between effectiveness and efficiency
remains a critical challenge for fully deploying
PAGER in real-world QA scenarios.
References
J. S. Bruner. 1966. Toward a theory of instruction.Har-
vard University Press.
Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey,
Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman,
Akhil Mathur, Alan Schelten, Amy Yang, Angela
Fan, and 1 others. 2024. The llama 3 herd of models.
ArXiv preprint.
Kai Guo, Xinnan Dai, Shenglai Zeng, Harry Shomer,
Haoyu Han, Yu Wang, and Jiliang Tang. 2025. Be-
yond static retrieval: Opportunities and pitfalls of
iterative retrieval in graphrag.ArXiv preprint.
Kelvin Guu, Kenton Lee, Zora Tung, Panupong Pa-
supat, and Ming-Wei Chang. 2020. Retrieval aug-
mented language model pre-training. InProceedings
of ICML, pages 3929–3938.
Junxian He, Graham Neubig, and Taylor Berg-
Kirkpatrick. 2021. Efficient nearest neighbor lan-
guage models. InProceedings of EMNLP, pages
5703–5714.
Xanh Ho, Anh-Khoa Duong Nguyen, Saku Sugawara,
and Akiko Aizawa. 2020. Constructing a multi-hop
QA dataset for comprehensive evaluation of reason-
ing steps. InProceedings of COLING, pages 6609–
6625.
Nan Huo, Jinyang Li, Bowen Qin, Ge Qu, Xiaolong
Li, Xiaodong Li, Chenhao Ma, and Reynold Cheng.2025. Micro-act: Mitigate knowledge conflict in
question answering via actionable self-reasoning.
ArXiv preprint.
Jinhao Jiang, Jiayi Chen, Junyi Li, Ruiyang Ren, Shijie
Wang, Xin Zhao, Yang Song, and Tao Zhang. 2025.
Rag-star: Enhancing deliberative reasoning with re-
trieval augmented verification and refinement. In
Proceedings of NAACL, pages 7064–7074.
Zhengbao Jiang, Frank Xu, Luyu Gao, Zhiqing Sun,
Qian Liu, Jane Dwivedi-Yu, Yiming Yang, Jamie
Callan, and Graham Neubig. 2023. Active retrieval
augmented generation. InProceedings of EMNLP,
pages 7969–7992.
Bowen Jin, Hansi Zeng, Zhenrui Yue, Jinsung Yoon,
Sercan Arik, Dong Wang, Hamed Zamani, and Jiawei
Han. 2025a. Search-r1: Training llms to reason and
leverage search engines with reinforcement learning.
ArXiv preprint.
Jiajie Jin, Yutao Zhu, Zhicheng Dou, Guanting Dong,
Xinyu Yang, Chenghao Zhang, Tong Zhao, Zhao
Yang, and Ji-Rong Wen. 2025b. Flashrag: A modular
toolkit for efficient retrieval-augmented generation
research. InCompanion Proceedings of the ACM on
Web Conference 2025, pages 737–740.
Jeff Johnson, Matthijs Douze, and Hervé Jégou. 2019.
Billion-scale similarity search with gpus.IEEE
Transactions on Big Data, (3):535–547.
Tom Kwiatkowski, Jennimaria Palomaki, Olivia Red-
field, Michael Collins, Ankur Parikh, Chris Alberti,
Danielle Epstein, Illia Polosukhin, Jacob Devlin, Ken-
ton Lee, Kristina Toutanova, Llion Jones, Matthew
Kelcey, Ming-Wei Chang, Andrew M. Dai, Jakob
Uszkoreit, Quoc Le, and Slav Petrov. 2019. Natu-
ral questions: A benchmark for question answering
research.Transactions of the Association for Compu-
tational Linguistics, pages 452–466.
Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying
Sheng, Lianmin Zheng, Cody Hao Yu, Joseph Gon-
zalez, Hao Zhang, and Ion Stoica. 2023. Efficient
memory management for large language model serv-
ing with pagedattention. InProceedings of SOSP,
pages 611–626.
Patrick S. H. Lewis, Ethan Perez, Aleksandra Pik-
tus, Fabio Petroni, Vladimir Karpukhin, Naman
Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih,
Tim Rocktäschel, Sebastian Riedel, and Douwe
Kiela. 2020. Retrieval-augmented generation for
knowledge-intensive NLP tasks. InProceedings of
NeurIPS.
Xiaoxi Li, Guanting Dong, Jiajie Jin, Yuyao Zhang,
Yujia Zhou, Yutao Zhu, Peitian Zhang, and Zhicheng
Dou. 2025a. Search-o1: Agentic search-enhanced
large reasoning models.ArXiv preprint.
Zhuoqun Li, Xuanang Chen, Haiyang Yu, Hongyu Lin,
Yaojie Lu, Qiaoyu Tang, Fei Huang, Xianpei Han,
Le Sun, and Yongbin Li. 2025b. Structrag: Boosting

knowledge intensive reasoning of llms via inference-
time hybrid information structurization. InProceed-
ings of ICLR.
Sewon Min, Julian Michael, Hannaneh Hajishirzi, and
Luke Zettlemoyer. 2020. AmbigQA: Answering am-
biguous open-domain questions. InProceedings of
EMNLP, pages 5783–5797.
Ofir Press, Muru Zhang, Sewon Min, Ludwig Schmidt,
Noah Smith, and Mike Lewis. 2023. Measuring and
narrowing the compositionality gap in language mod-
els. InProceedings of EMNLP Findings, pages 5687–
5711.
Zhihong Shao, Yeyun Gong, Yelong Shen, Minlie
Huang, Nan Duan, and Weizhu Chen. 2023. En-
hancing retrieval-augmented large language models
with iterative retrieval-generation synergy. InPro-
ceedings of EMNLP Findings, pages 9248–9274.
Huatong Song, Jinhao Jiang, Yingqian Min, Jie Chen,
Zhipeng Chen, Wayne Xin Zhao, Lei Fang, and Ji-
Rong Wen. 2025. R1-searcher: Incentivizing the
search capability in llms via reinforcement learning.
ArXiv preprint.
Zhongxiang Sun, Qipeng Wang, Weijie Yu, Xiaoxue
Zang, Kai Zheng, Jun Xu, Xiao Zhang, Song Yang,
and Han Li. 2025. Rearter: Retrieval-augmented
reasoning with trustworthy process rewarding.ArXiv
preprint.
Harsh Trivedi, Niranjan Balasubramanian, Tushar Khot,
and Ashish Sabharwal. 2022. MuSiQue: Multi-
hop questions via single-hop question composition.
Transactions of the Association for Computational
Linguistics, pages 539–554.
Harsh Trivedi, Niranjan Balasubramanian, Tushar Khot,
and Ashish Sabharwal. 2023. Interleaving retrieval
with chain-of-thought reasoning for knowledge-
intensive multi-step questions. InProceedings of
ACL, pages 10014–10037.
Jesse Vig, Alexander Richard Fabbri, Wojciech Kry ´s-
ci´nski, Chien-Sheng Wu, and Wenhao Liu. 2022. Ex-
ploring neural models for query-focused summariza-
tion. InProceedings of NAACL, pages 1455–1468.
Ruobing Wang, Qingfei Zhao, Yukun Yan, Daren Zha,
Yuxuan Chen, Shi Yu, Zhenghao Liu, Yixuan Wang,
Shuo Wang, Xu Han, Zhiyuan Liu, and Maosong
Sun. 2025a. Deepnote: Note-centric deep retrieval-
augmented generation. InProceedings of EMNLP.
Zihan Wang, Zihan Liang, Zhou Shao, Yufei Ma,
Huangyu Dai, Ben Chen, Lingtao Mao, Chenyi Lei,
Yuqing Ding, and Han Li. 2025b. Infogain-rag:
Boosting retrieval-augmented generation via docu-
ment information gain-based reranking and filtering.
ArXiv preprint.
Zihao Wang, Anji Liu, Haowei Lin, Jiaqi Li, Xiaojian
Ma, and Yitao Liang. 2024. Rat: Retrieval aug-
mented thoughts elicit context-aware reasoning in
long-horizon generation.ArXiv preprint.Mingyan Wu, Zhenghao Liu, Yukun Yan, Xinze Li, Shi
Yu, Zheni Zeng, Yu Gu, and Ge Yu. 2025. Rankcot:
Refining knowledge for retrieval-augmented genera-
tion through ranking chain-of-thoughts. InProceed-
ings of ACL.
Fangyuan Xu, Weijia Shi, and Eunsol Choi. 2023a. Re-
comp: Improving retrieval-augmented lms with com-
pression and selective augmentation.ArXiv preprint.
Ruochen Xu, Song Wang, Yang Liu, Shuohang Wang,
Yichong Xu, Dan Iter, Pengcheng He, Chenguang
Zhu, and Michael Zeng. 2023b. Lmgqs: A large-
scale dataset for query-focused summarization. In
Proceedings of EMNLP Findings, pages 14764–
14776.
An Yang, Anfeng Li, Baosong Yang, Beichen Zhang,
Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao,
Chengen Huang, Chenxu Lv, and 1 others. 2025.
Qwen3 technical report.ArXiv preprint.
Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio,
William Cohen, Ruslan Salakhutdinov, and Christo-
pher D. Manning. 2018. HotpotQA: A dataset for
diverse, explainable multi-hop question answering.
InProceedings of EMNLP, pages 2369–2380.
Yanzhao Zhang, Mingxin Li, Dingkun Long, Xin
Zhang, Huan Lin, Baosong Yang, Pengjun Xie,
An Yang, Dayiheng Liu, Junyang Lin, and 1 oth-
ers. 2025. Qwen3 embedding: Advancing text em-
bedding and reranking through foundation models.
ArXiv preprint.
Xiangrong Zhu, Yuexiang Xie, Yi Liu, Yaliang Li, and
Wei Hu. 2025. Knowledge graph-guided retrieval
augmented generation. InProceedings of NAACL,
pages 8912–8924.

A Appendix
A.1 License
We show the licenses of the datasets that we use.
All of these datasets are allowed for academic
use under their respective licenses and agreements:
MuSiQue and HotpotQA (CC-BY-4.0 License);
2WikiMQA (Apache 2.0 License); Bamboogle
(MIT License); NQ and AmbigQA (CC BY-SA
3.0 License).
A.2 More Implementation Details
In our experiments, we utilize FlashRAG (Jin et al.,
2025b) to reproduce the results of IRCoT and Iter-
RetGen on the experimental datasets. For RAT,
Search-o1, StructRAG, and DeepNote, we repro-
duce their results using the official GitHub code
provided. During inference, we use vLLM to load
model checkpoints and perform offline batched gen-
eration, while setting unified hyperparameters: tem-
perature = 0.7, top-p = 0.8, top-k = 20, and random
seed = 66.
A.3 The Computation of Document
Information Gain
The document information gain (DIG) metric
serves to quantify the actual utility of external
knowledge within RAG systems (Wang et al.,
2025b).
Formally, for a question q, a knowledge represen-
tation O, and the ground truth y, DIG is defined as
the difference in the LLM’s generation confidence
for the correct answer when the Kis included ver-
sus when it is excluded. Let pϕ(y|q, O) denote
the conditional probability of the model generating
the ground truth ywhen knowledge representation
Ois augmented into the context, and let pϕ(y|x)
represent the probability of generating the ground
truthywithout the augmentation of knowledge rep-
resentationO. Then, the DIG is calculated as:
DIG(O|q) =p ϕ(y|q, O)−p ϕ(y|q),(12)
where ϕdenotes the parameters of the LLM. Fur-
thermore, in the calculation of DIG, previous work
introduces two strategies specifically designed to
mitigate length bias and ensure the capture of the
strongest signals indicative of generation quality:
Sliding Window Smoothing.To mitigate length
bias in long sequences, a sliding window mecha-
nism is utilized to smooth local probability fluctua-
tions. For each token tiin the answer sequence y,
1 2 3 4
Iterations35404550CoverEM
PAGER
DeepNote
Iter-RetGen(a) HotpotQA.
1 2 3 4
Iterations152025CoverEM
 (b) MuSiQue.
Figure 6: The Performance of Models Evolution across
Iteration Rounds.
1 2 3 4
Iterations01234CoverEM
PAGER
DeepNote
Iter-RetGen
(a) HotpotQA.
1 2 3 4
Iterations01234CoverEM
 (b) MuSiQue.
Figure 7: The Document Overlap of Models Evolution
across Iteration Rounds.
its smoothed probability is calculated as:
ps(ti) =1
Wi+⌊W/2⌋X
j=i−⌊W/2⌋p(tj),(13)
where Wdenotes the window size and p(tj)repre-
sents the original token probability.
Token Importance Weighting.To emphasize
the core semantic information often encoded in
initial tokens, a weighting scheme assigns higher
weights to the first ktokens. The final calibrated
probability score is derived as:
pϕ(y|x) =kY
i=1(ps(ti))ωi·α·|y|Y
j=k+1(ps(tj))1−α,
(14)
where ωiis the importance weight for the i-th token,
andαis a hyperparameter controlling the emphasis
on the initial segment.
A.4 The Characteristics of Knowledge
Representation Construction Processes
In this section, we further explore the effects of
the retrieval and generation modules of different
methods during the iterative construction of knowl-
edge representations. We adopt Qwen3-32B as the
backbone model for all experiments. We select
the questions that require four rounds of iterative

Methods HotpotQA 2WikiMQA MuSiQue Bamboogle NQ AmbigQA Avg.
Vanilla LLM 39.0 57.0 15.5 62.4 58.0 53.0 47.5
Vanilla RAG 51.0 63.0 20.0 62.4 69.5 62.5 54.7
DeepNote (2025a) 45.0 41.5 20.5 52.0 52.0 55.0 44.3
PAGER 54.0 61.5 31.5 68.8 62.0 65.5 57.2
Table 4: Overall Performance of Different RAG Models. Thebestand second best results are highlighted. In our
experiments, we employ GLM-4.5 as a backbone mode.
retrieval for PAGER, DeepNote, and Iter-RetGen
to construct knowledge representations as the eval-
uation set.
As illustrated in Figure 6, we first analyze how
the performance of each method evolves through-
out the iterative process. We observe that, as the
iteration count increases, the performance gains
for Iter-RetGen and DeepNote rapidly plateau af-
ter the second round. This indicates that they en-
counter a bottleneck in continuously integrating
new knowledge across multiple iterations, making
it difficult to further capture and refine the key infor-
mation in subsequent rounds. In contrast, PAGER
exhibits steady performance improvement as iter-
ations progress. This indicates that PAGER is ca-
pable of progressively incorporating more compre-
hensive external knowledge during the construction
of structured page representations, thereby consis-
tently enhancing model performance.
To better understand this behavior, we analyze
the retrieval diversity in Figure 7, which displays
the overlap between documents retrieved in the cur-
rent iteration and the cumulative set of documents
from all prior iterations. The results demonstrate
that DeepNote and Iter-RetGen exhibit a higher
degree of document overlap compared to PAGER.
This high redundancy explains the performance
plateau observed earlier. Conversely, the lower
overlap observed in PAGER indicates its ability to
retrieve more diverse knowledge and background
information throughout the iterative process, ensur-
ing the construction of a more comprehensive and
robust knowledge representation.
A.5 The Performance of PAGER on
Closed-source LLM
As shown in Table 4, we compare the overall per-
formance of PAGER with various baseline meth-
ods on a range of knowledge-intensive tasks us-
ing GLM-4.5 as the backbone model. We sam-
ple 200 data points from each dataset for evalua-
tion. Overall, when using GLM-4.5 as the back-
bone model, PAGER achieves performance im-
provements of over 3% compared to all baselinemodels, demonstrating its effectiveness. This indi-
cates that PAGER is not only applicable to open-
source LLMs such as Qwen3-32B and Llama3.1-
70B-Instruct, but also demonstrates strong perfor-
mance on closed-source LLMs.
A.6 Inference Time Latency
In this section, we compare the inference latency of
PAGER (Parallel Filling) and PAGER in the abla-
tion experiments presented in section 5.2. PAGER
(Parallel Filling) simultaneously generates sub-
queries for all missing slots in the initial page and
performs parallel retrieval. We utilize the Qwen3-
32B as the backbone model.
As shown in Table 5, PAGER exhibits an in-
ference latency that is approximately 1.8× that of
PAGER (Parallel Filling). However, it is worth
noting that PAGER typically operates with an av-
erage of four iteration rounds, indicating that the
time cost does not grow linearly with the number
of iterations (i.e., four iterations do not result in
4× latency). This is mainly because PAGER (Par-
allel Filling) needs to incorporate a larger number
of retrieved documents from a single retrieval step
into the context. More importantly, this additional
temporal cost is justified by the performance gains,
as the sequential iterations allow the model to dy-
namically adjust its retrieval strategy based on in-
termediate evidence, a capability that the parallel
approach lacks.
A.7 Statistics of Page Slots
In this section, we further investigate the number
of slots initialized by PAGER during page con-
struction and their distribution. As shown in Fig-
ure 6, the number of slots in pages constructed by
different backbone models is predominantly con-
centrated in the range of 3to5. This indicates that
during page initialization, PAGER leverages its log-
ical planning capability to focus the page structure
on a set of key core topics. Further analysis reveals
that for Qwen-32B, the pages it constructs predom-
inantly contain four slots, whereas for Llama-72B,
the constructed pages are mainly characterized by

Methods HotpotQA 2WikiMQA MuSiQue Bamboogle NQ AmbigQA Avg.
Inference Time Latency
PAGER (Parallel Filling) 2.54 1.69 2.14 1.72 2.60 2.39 2.18
PAGER 3.82 3.42 3.88 3.76 4.38 3.71 3.82
Inference Performance
PAGER (Parallel Filling) 45.9 43.8 18.8 59.2 56.4 57.0 46.9
PAGER 50.6 57.4 23.0 62.4 56.5 56.4 51.1
Table 5: Inference Time Latency. The unit of inference latency is seconds.
# Slots HotpotQA 2WikiMQA MuSiQue Bamboogle NQ AmbigQA Avg.
Qwen3-32B
≤2 0.4 1.0 0.4 1.6 0.1 0.0 0.6
= 3 20.8 27.4 20.8 29.6 5.8 9.4 18.9
= 4 35.9 42.6 36.4 40.0 30.5 36.5 37.0
= 5 32.4 23.0 30.1 21.6 44.3 40.3 31.9
= 6 7.5 4.5 8.1 4.8 13.6 9.9 8.1
≥7 3.0 1.5 4.2 2.4 5.7 3.9 3.5
Llama3.1-70B-Instruct
≤2 2.5 4.0 4.1 3.2 1.0 1.1 2.7
= 3 44.9 53.0 45.1 52.8 38.6 47.9 47.1
= 4 40.6 37.3 37.2 34.4 36.1 35.8 36.9
= 5 10.4 5.6 11.4 9.6 19.4 13.0 11.6
= 6 1.4 0.1 1.9 0.0 3.8 1.8 1.4
≥7 0.2 0.0 0.3 0.0 1.1 0.4 0.3
Table 6: The Statistics of Page Slots.
three slots. This phenomenon further indicates that,
due to the varying inherent logical reasoning and
cognitive capabilities of different LLMs, the struc-
ture of the initialized cognitive outlines and the
distribution of slot counts also differ.
A.8 Case Studies of PAGER
In this section, we select some cases to demonstrate
the effectiveness of PAGER, as well as the pro-
cesses of page construction of PAGER. All cases
are selected from the HotpotQA dataset.
As illustrated in Figure 8, we compare the per-
formance of three models: Vanilla RAG, Deep-
Note, and PAGER. Vanilla RAG and DeepNote er-
roneously identify “Bill Nye” and “Wil Wheaton”,
respectively, as the correct actors, despite neither
having starred in “The Bronze”. In contrast, the
page constructed by PAGER accurately consoli-
dates the key information that “Melissa Rauch”
starred in “The Bronze” and also appeared in The
Big Bang Theory. This demonstrates that by con-
structing a comprehensive knowledge representa-
tion, PAGER effectively assists the LLM in answer-
ing the question.
Figure 9 further illustrates the iterative page con-
struction process of PAGER. During the initializa-
tion phase, PAGER constructs a structured page
outline tailored to the query, establishing desig-
nated slots. This structural outline serves as a
roadmap to guide the subsequent knowledge ac-quisition and utilization. In Iterations 1 and 2, the
PAGER generates sub-queries to retrieve external
documents, successfully refining key information
into the page. Subsequently, in Iterations 3 and
4, the model formulates sub-queries designed to
explore the relationship between the entities. Re-
trieval results reveal that the two directly competed
in a 2000 tournament. Consequently, PAGER syn-
thesizes this evidence within the “Career Compari-
son”, explicitly articulating that both subjects are
deeply involved in the realm of MMA. Finally,
leveraging this comprehensively constructed and
logically coherent page, the model not only iden-
tifies the correct answer, “Mixed martial artist”,
but also grounds its conclusion by citing specific
evidence embedded within the page.
A.9 Prompt Templates Used for PAGER
We provide a detailed description of the instruction
prompts used in the experiments of PAGER. Fig-
ure 10 illustrates the instruction for initializing the
outline of the page. Figure 11 shows the instruc-
tions for generating sub-queries. Figure 12 presents
the instruction used for filling the page slots. Fig-
ure 13 shows the instructions for evaluating the
overall quality of the knowledge representations
generated by different models.

Case#1:Effectivenessbetween Different ModelQuestion:WhostarredinTheBronzeandalsoshowedupontheCBSsitcom"TheBigBangTheory"?
DeepNoteThese include theoretical physicist Brian Greene, astrophysicist Neil deGrasse Tyson, cosmologist Stephen Hawking, NASA astronaut Michael J. Massimino, and Bill Nye, among others. No individual from this list is explicitly identified as having starred in *The Bronze*. Wil Wheaton, however, is mentioned as having appeared as a fictionalized version of himself in several episodes of *The Big Bang Theory*, starting in season 3, episode 5: *The Creepy Candy Coating Corollary* (2009). On the show, Wheaton behaves in comically petty and manipulative ways towards main character Sheldon Cooper, who regards him as an archenemy until the season 5 episode *The Russian Rocket Reaction*, when they make amends and become friends. Wheaton appears in 12 episodes in a recurring, guest-starring role on *Eureka*, playing Dr. Isaac Parrish, the head. Wil Wheaton also hosts *TableTop*, a board game based show, and *Titansgrave*, a roleplaying game based show.While the retrieved documents do not explicitly mention Wil Wheaton’s role in *The Bronze*, it is known from external sources that Wil Wheaton starred in the 2016 comedy film *The Bronze*. Therefore, Wil Wheaton is the individual who starred in *The Bronze* and also appeared on *The Big Bang Theory*, as detailed in the retrieved documents.FinalAnswer:WilWheaton(Incorrect)PAGER # Who Starred in The Bronze and Also Appeared on The Big Bang Theory? ## Overview of The Bronze (2014) \n*The Bronze* is a 2015 American sports comedy-drama film directed by Bryan Buckley and written by Melissa Rauch and Winston Rauch. The film centers around Hope Ann Greggory, a former gymnast who won a bronze medal in the 1996 Olympics and is struggling to maintain relevance in the spotlight. It was produced by Mark Duplass and Jay Duplass through their Duplass Brothers Productions…## Notable Cast Members of The Bronze \nThefilm *The Bronze* features a notable ensemble cast led by Melissa Rauch, who also co-wrote the screenplay and portrays the lead character, Hope Ann Greggory. Other prominent cast members include Gary Cole, Thomas Middleditch, Sebastian Stan, Cecily Strong, Haley Lu Richardson, and Dale Raoul. Rauch’s role in the film is particularly significant, as she brings a comedic and self-deprecating energy to the character of a former Olympic gymnast trying to reclaim her relevance……## Identifying the Actor Common to Both ProjectsThe actor who starred in *The Bronze* and also appeared on *The Big Bang Theory* is Melissa Rauch. In *The Bronze*, Rauch played the lead role of Hope Ann Greggory, a disgraced Olympic gymnast navigating the challenges of public life…FinalAnswer:MelissaRauch(Correct)VanillaRAG[id: 1]: …Hawking video conferences with Sheldon and Leonard, and makes another appearance in the 200th episode. In the fifth and sixth seasons, NASA astronaut Michael J. Massimino played himself multiple times in the role of Howard\'s fellow astronaut. Bill Nye appeared in the seventh and twelfth seasons. [id: 2]: The Big Bang Theory The Big Bang Theory is an American television sitcom created by Chuck Lorre and Bill Prady, both of whom serve as executive producers on the series, along with Steven Molaro. All three also serve as head writers. The show premiered on CBS on September 24, 2007……[id: 4]: The third season saw the first appearances of future main cast members Melissa Rauch as Bernadette Rostenkowski in ""The Creepy Candy Coating Corollary”…FinalAnswer:Bill Nye(Incorrect)Figure 8: Case Study of Different Methods. Text inredrepresents document excerpts directly related to the
knowledge representation and the question answering, while text inbluerepresents the noisy content in the
knowledge representation.

Case#2:Page Construction Process of PAGER Question:What profession does Kazuyuki Fujita and Gilbert Yvelhave in common?Ground Truth: Mixed martial artistGenerate Sub-query:What is Kazuyuki Fujita's profession?Retrieved Documents:Kazuyuki Fujita Fujita practiced freestyle wrestling in high school. Championships as a junior in 1988, placing sixth, and in the Espoir division in 1989He participated in the FILA World, placing eleventh. In 1993, Fujita placed fifth at the Asian Championships at the senior level, and in 1993 and 1994, he represented Japan as a senior in the World Cup…Fill Page (iter-1):# Shared Profession of Kazuyuki Fujita and Gilbert Yvel## Introduction to Kazuyuki FujitaKazuyuki Fujita is a Japanese professional wrestler and mixed martial artist who began his career in freestyle and Greco-Roman wrestling during his high school and early adult years. He competed in several……InitPage:# Shared Profession of Kazuyuki Fujita and Gilbert Yvel## Introduction to Kazuyuki Fujita<TO BE FILLED>## Introduction to Gilbert Yvel<TO BE FILLED>## Career Comparison: Kazuyuki Fujita vs. Gilbert Yvel<TO BE FILLED>## Common Profession: Identifying the Overlap<TO BE FILLED>Generate Sub-query:What is Gilbert Yvel'sprofession?Retrieved Documents:Gilbert Yvel Gilbert Ramon Yvel (born June 30, 1976) is a Dutch mixed martial artist currently competing in the Super Heavyweight division and associated with the Vos Gym in the Netherlands. A professional competitor for 16 years from 1997-2013,…FillPage(iter-2):…## Introduction to Gilbert YvelGilbert Yvel, born Gilbert Ramon Yvel on June 30, 1976, is a Dutch mixed martial artist best known for his aggressive fighting style and controversial behavior inside the cage. Competing primarily in the super heavyweight division, Yvel has participated in numerous high-profile promotions……Generate Sub-query:What are the similarities in the careers of Kazuyuki Fujita and Gilbert Yvelin combat sports?Retrieved Documents:Cold Fury show in 2000, in the 7th fight of the evening, former Japan national Greco-Roman wrestling champion Kazuyuki Fujita faced Dutch Muay Thai stylist Gilbert Yvel, in a classic wrestler versus striker mixed martial arts (MMA) matchup…FillPage(iter-3):…## Career Comparison: Kazuyuki Fujita vs. Gilbert YvelBoth Kazuyuki Fujita and Gilbert Yvel share a deep involvement in combat sports, particularly within the realm of mixed martial arts (MMA). Fujita, a former Greco-Roman wrestling champion, transitioned to MMA after a……Generate Sub-query:What profession do Kazuyuki Fujita and Gilbert Yvelshare?Retrieved Documents:a commentator with a ""foreign"" point of view. He is known as one of ""the two Kents"" (along with Kent Derricott). Gilbert lives in Tokyo and alsomaintains a residence in Orem, Utah. He made an…FillPage(iter-4):…## Career Comparison: Kazuyuki Fujita vs. Gilbert YvelBoth Kazuyuki Fujita and Gilbert Yvel share a deep involvement in combat sports, particularly within the realm of mixed martial arts (MMA). Fujita, a former Greco-Roman wrestling champion, transitioned to MMA after a……AnswerGeneration:…including their notable bout in the PRIDE Fighting Championships. The section titled "Common Profession: Identifying the Overlap" directly addresses the question and confirms their shared profession.<answer> Mixed martial artist </answer>Figure 9: Case Study of Iterative Page Construction. Text inredrepresents the knowledge evidence filled into the
slots.

InstructionforOutline InitializationYou are a Page Planning Expert.For a given question, you are tasked with generating a page outline based on the theme of the question.The outline you generate will serve as the foundation for the subsequent process, during which the outline will be filled with content to create a complete page that assists the reader in answering the given question.Please strictly follow the instructions below.Input: -Question: {question}Task Steps: 1. **Reasoning Analysis**-First, analyze the theme of the question and thoroughly understand the knowledge required to answer it, as well as the logical relationships between this knowledge. -Based on the theme of the question, generate a page outline and identify the key content that should be covered in each section. -For each section, propose a suitable and appropriate title, ensuring that each title effectively guides the reader to progressively deepen their understanding of the various aspects of the question. 1. **Outline Initialization** -Generate the outline: -Use # [Main Title] for the main title. The main title is a concise and comprehensive abstraction of the page content.-Use ## [Section Title] for each section title. The section title is the section heading (do not reveal the final answer).-Insert special marker <TO BE FILLED> under all section titles. Make sure the sections of the page are ordered logically, building up the reader's understanding toward answering the question.The number of sections on the page should be limited to the scope necessary to answer the question, avoiding any overlap of content between sections.2. **Output** -First, you should output the reasoning analysis for initializing the outline. -Then, you should generate a special symbol <OUTLINE>, followed by the generated page outline. Note that the content after <OUTLINE> should only include the final generated page outline, without any comments or explanations.Figure 10: The Prompt Template for Generating the Outline.

InstructionforSub-queryGenerationYou are a professional question-generation expert. First, following the order of the page sections, locate and identify the first unfinished section. An unfinished section is defined as one that contains the special symbol <TO BE FILLED> under the section title.Please strictly follow the steps and format below to formulate a precise retrieval question for the first unfinished section of the current page, in order to obtain the necessary information to complete that section.Input Parameters:-Original question: {question}-Current page content: {page}Task Steps:1. **Parse the page plan:**-Read the Page and locate the title and topic of the first section that remains unfilled.2. **Align with the original question:**-Align the section's topic with the core needs of original question to ensure the retrieval question directly serves the original question.3. **Design the retrieval question. The question must:**-Be focused: target only the core subtopic of that section;-Be clear: use search terms that can be directly used in a search engine or knowledge base;-Be concise: include no unnecessary background.4. **Output format:**-Output only one line containing the English retrieval question, with no additional comments or explanations.Figure 11: The Prompt Template for Generating the Subquery.

InstructionforPageFillingYou are a professional page content writer.Please strictly follow the instructions below.First, following the order of the page sections, locate and identify the first unfinished section. An unfinished section is defined as one that contains the special symbol <TO BE FILLED> under the section title.Use the retrieved documents, and your internal knowledge to complete the first unfinished section of the page, and generate the page with that section filled in.Please note that you should only fill in one unfinished section, and that section must be the first unfinished section on the page. Do not fill in additional sections or fill across different sections.Input: -Original question: {question}-Sub-question for retrieval: {sub_question}-Retrieved documents: {docs_text}-Current page: {page}Task Steps:1. **Section Completion** -Find the first unfinished section in the page, where the section title contains a special placeholder <TO BE FILLED>.-Using information from the retrieved documents and your internal knowledge, write a short paragraph under that section. The paragraph should: -Be tightly related to the original question. -Focus strictly on the topic of that section. -Avoid redundant or irrelevant information. -Remove the <TO BE FILLED> placeholder under that section. -Retain <TO BE FILLED> placeholders for all other unfinishesections. -If you have filled the last unfinished section of the page, ensure that no <TO BE FILLED> placeholders remain in the page.2. **Output format**-Only output the entire page with the first unfinished section fully filled in. Do not include any comments, explanations, or isolated section content.-Your output must be the entire updated page, including the newly filled content, seamlessly integrated into the original page structure.-Do not output just the section you filled in—your output must be the entire page, including all content, both existing and newly added.Figure 12: The Prompt Template for Filling the Slots in the Page.

The Instruction for Using GLM to Evaluate Knowledge RepresentationsYou are an expert evaluator proficient in cognitive linguistics and natural language processing tasks. Your task is to evaluate the quality of a "Knowledge Context". This context was generated by a model based on raw retrieved documents following multi-round knowledge retrieval and refinement, specifically for the purpose of answering a given question.# Input Data-**User Question**: {question}-**Ground Truth Answer**: {ground_truth}-**Raw Retrieved Documents**: {retrieved_docs}-**Evaluated Knowledge Context**: {generated_context}# Evaluation CriteriaPlease strictly review the "Evaluated Knowledge Context" based on the following four core dimensions. **You must assign a separate score (0-5) for each dimension.**1. **Knowledge Structuring**-Does the Knowledge Context establish a coherent knowledge framework or outline, rather than merely listing scattered facts?-Are key concepts hierarchically organized or classified to facilitate the understanding of complex information?-*Negative Indicators*: Fragmented information, lack of a main thread, keyword stuffing.2. **Document Refinement**-Does the Knowledge Context demonstrate the "digestion" and "reorganization" of the original documents? Does it bridge logicalgaps by synthesizing information from multiple documents?-Has the raw material been transformed into a problem-oriented explanation, rather than simple copy-pasting?-*Negative Indicators*: Mechanical repetition, concatenation of excerpts, failure to adapt content to the specific question.3. **Coherence & Logical Flow**-Is the narrative of the context fluid? Are there clear causal or progressive relationships between ideas?-Does it resolve potential contradictions found in the raw documents or reasonably reconcile conflicting information?-*Negative Indicators*: Logical jumps, self-contradiction, abrupt transitions.4. **Correctness & Argumentation**-Is the evidence provided in the Knowledge Context sufficient and accurate enough to forcefully support the derivation of the "Standard Answer"?-*Negative Indicators*: Missing key evidence, inclusion of misleading information, irrelevance to the Standard Answer.# Scoring Scale (Applied to EACH dimension independently)Please evaluate each dimension on a scale of 0 to 5 using the rubric below:-**5 (Perfect)**: The dimension is executed flawlessly. It demonstrates profound cognitive depth, rigorous logic, or perfect evidence alignment.-**4 (Excellent)**: Strong execution. Meets the high standards of the dimension with only negligible flaws.-**3 (Satisfactory)**: Acceptable performance. Contains the main necessary elements but lacks depth or relies on basic stitching rather than deep processing.-**2 (Poor)**: Weak performance. Significant gaps exist in this specific dimension (e.g., chaotic structure, poor synthesis, orlogical breaks).-**1 (Unusable)**: Severe failure. The context fails almost completely in this dimension (e.g., hallucinations, complete lack of logic, or missing critical evidence).-**0 (Failure)**: Completely irrelevant, empty, or non-existent effort in this dimension.# Output FormatPlease strictly follow the format below to present your evaluation. Do not output conversational filler.### 1. Knowledge Structuring-**Score**: [0-5]-**Reasoning**: [Critically analyze the framework and hierarchy. Cite specific examples of good or bad structuring.]### 2. Document Refinement-**Score**: [0-5]-**Reasoning**: [Evaluate the extent of information digestion and reorganization. Did it simply copy-paste or truly synthesize?]### 3. Coherence & Logical Flow Evaluation-**Score**: [0-5]-**Reasoning**: [Analyze the narrative fluidity and causal connections between sentences/paragraphs.]### 4. Correctness & Argumentation Evaluation-**Score**: [0-5]-**Reasoning**: [Assess if the evidence provided is accurate and sufficient to support the Ground Truth.]---### Final Summary-**Average Score**: [Calculate the arithmetic mean of the 4 scores]-**Conclusion**: [One sentence summary of the overall quality.]Figure 13: The Prompt Template for Scoring Using the GLM Model.