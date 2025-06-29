# MMSearch-R1: Incentivizing LMMs to Search

**Authors**: Jinming Wu, Zihao Deng, Wei Li, Yiding Liu, Bo You, Bo Li, Zejun Ma, Ziwei Liu

**Published**: 2025-06-25 17:59:42

**PDF URL**: [http://arxiv.org/pdf/2506.20670v1](http://arxiv.org/pdf/2506.20670v1)

## Abstract
Robust deployment of large multimodal models (LMMs) in real-world scenarios
requires access to external knowledge sources, given the complexity and dynamic
nature of real-world information. Existing approaches such as
retrieval-augmented generation (RAG) and prompt engineered search agents rely
on rigid pipelines, often leading to inefficient or excessive search behaviors.
We present MMSearch-R1, the first end-to-end reinforcement learning framework
that enables LMMs to perform on-demand, multi-turn search in real-world
Internet environments. Our framework integrates both image and text search
tools, allowing the model to reason about when and how to invoke them guided by
an outcome-based reward with a search penalty. To support training, We collect
a multimodal search VQA dataset through a semi-automated pipeline that covers
diverse visual and textual knowledge needs and curate a search-balanced subset
with both search-required and search-free samples, which proves essential for
shaping efficient and on-demand search behavior. Extensive experiments on
knowledge-intensive and info-seeking VQA tasks show that our model not only
outperforms RAG-based baselines of the same model size, but also matches the
performance of a larger RAG-based model while reducing search calls by over
30%. We further analyze key empirical findings to offer actionable insights for
advancing research in multimodal search.

## Full Text


<!-- PDF content starts -->

arXiv:2506.20670v1  [cs.CV]  25 Jun 2025MMSearch-R1: Incentivizing LMMs to Search
Jinming Wu1,∗, Zihao Deng1,∗, Wei Li1, Yiding Liu1, Bo You1,
Bo Li2,†,Zejun Ma1,Ziwei Liu2
1ByteDance2S-Lab, NTU
https://github.com/EvolvingLMMs-Lab/multimodal-search-r1
<reason> The image search results mention the "Viper" lunar rover project and a 
news article about the cancellation of the project. However, the image does not 
provide information about the exact date when the project was canceled. To 
answer the user's question, a text search is necessary to find the specific date of 
the cancellation. </reason> <text_search> When was the NASA Viper moon 
rover project canceled in 2024? </text_search>
In which day of 2024 was 
the lunar rover project in 
the picture announced to 
be canceled?
<reason> The image shows a lunar rover, but there is no information about its 
announcement or cancellation date. The image does not provide enough context 
to answer the user's question confidently.</reason> <search><img></search>
Image Search: retrieve related webpages thumbnails and titles ...
VIPER...
Text Search: retrieve related contents ...
1. (link) NASA announced the cancellation of the VIPER (Volatiles 
Investigating Polar Exploration Rover) project on July 17, 2024. The 
decision followed a comprehensive internal review ...
2. (link) The NASA VIPER (Volatiles Investigating Polar Exploration 
Rover) moon rover project was canceled in July 2024. Following this 
cancellation ... 
  ...
<reason> The search results indicate that the NASA VIPER lunar rover 
project was canceled on July 17, 2024. The information is consistent across 
multiple sources, suggesting that this is the correct date for the cancellation. 
</reason> <answer> July 17 </answer>Answer 
with
SearchAnswer 
without
Search
What traditional 
performance art is 
being shown in the 
image?
<reason> The image shows a scene with performers dressed in 
traditional Chinese costumes, likely engaged in a theatrical performance. 
The presence of elaborate costumes and props, along with the presence 
of a child performer, suggests that this could be a form of Chinese opera, 
such as Peking Opera. Peking Opera is a classic form of Chinese theater 
known for its elaborate costumes, face paint, and stylized acting. The 
scene appears to be a part of a Peking Opera performance. </reason>
<answer> Peking Opera </answer>
40455055606570AccuracySearch Ratio (SR) (%)
020406080100
FVQA-test InfoSeek MMSearchSimpleVQA LiveVQAQwen2.5VL-7B(RAG) Qwen2.5VL-32B(RAG) MMSearch-R1-7B
RAG SR
MMSearch-R1 SR
MMSearch-R1 vs. RAG: Results on Five VQA Tasks
Virtual Intelligent Planetary 
Exploration Rover, VIPER 
Mobili... - NASA Science
Scientists slam 'indefensible' axing of 
Nasa's $450m Viper moon rover | Nasa | 
The Guardian
Figure 1: Overview of MMSearch-R1. MMSearch-R1 learns to recognize the boundaries of its
knowledge and perform on-demand search, significantly reducing the number of searches required
while outperforming RAG-based models on knowledge-intensive and info-seeking VQA tasks.
Abstract
Robust deployment of large multimodal models (LMMs) in real-world scenarios
requires access to external knowledge sources, given the complexity and dynamic
nature of real-world information. Existing approaches such as retrieval-augmented
generation (RAG) and prompt engineered search agents rely on rigid pipelines,
often leading to inefficient or excessive search behaviors. We present MMSearch-
R1, the first end-to-end reinforcement learning framework that enables LMMs to
perform on-demand, multi-turn search in real-world Internet environments. Our
framework integrates both image and text search tools, allowing the model to
reason about when and how to invoke them guided by an outcome-based reward
with a search penalty. To support training, We collect a multimodal search VQA
dataset through a semi-automated pipeline that covers diverse visual and textual
knowledge needs and curate a search-balanced subset with both search-required and
search-free samples, which proves essential for shaping efficient and on-demand
search behavior. Extensive experiments on knowledge-intensive and info-seeking
VQA tasks show that our model not only outperforms RAG-based baselines of
the same model size, but also matches the performance of a larger RAG-based
model while reducing search calls by over 30% . We further analyze key empirical
findings to offer actionable insights for advancing research in multimodal search.
∗Equal Contribution; †Work collaborated with ByteDance
Preprint.

1 Introduction
Scaling up visual-text pair data by leveraging large-scale, high-quality, and diverse datasets across
different training stages [ 51,4,68,62,16] has become a fundamental paradigm for acquiring
grounded knowledge of the visual world, driving breakthroughs in Large Multimodal Models (LMMs)
[23,57,59,5,32,33,37,11]. This foundation enables strong performance across a wide range
of visual understanding tasks, such as visual question answering and captioning, facilitating the
transition of LMMs from research prototypes to practical, intelligent assistants in everyday life.
However, this paradigm faces inherent limitations when handling complex and dynamic real-world
knowledge [ 28,6]. Specifically, long-tail information such as facts that emerge after the model’s
training cut-off or domain-specific knowledge constrained by privacy, copyright, or security consid-
erations is difficult to capture through static training alone. As a result, leading LMMs continue to
struggle with knowledge-intensive and information-seeking VQA tasks where external and up-to-date
knowledge is often required [ 35,26,18]. When confronted with inputs beyond their internal knowl-
edge boundaries, such as unfamiliar visual content or previously unseen textual information, models
are prone to hallucinations [ 38,34], which severely compromise their reliability in applications that
demand factual accuracy and trustworthiness [15].
Integrating the ability to interact with search tools into LMMs to search for external information
offers a promising solution to these limitations [ 26,21,70]. Existing approaches can be broadly
categorized into two paradigms: (1) Retrieval-Augmented Generation (RAG) [ 8,64,10], which
retrieves external visual or textual information to guide model generation; and (2) Prompt-engineered
agents [ 21,70,26], which prompt large models to iteratively perform web searches and reason over
the retrieved results to generate answers. However, these approaches remain suboptimal in practice.
RAG-based methods follow a fixed retrieve-then-generate workflow grounded in static knowledge
bases, often resulting in over-retrieval, high computational cost, and the unrealistic assumption that
all required information is already present in the corpus. Such a controlled setting fails to capture
the dynamic and unpredictable nature of real-world scenarios, rendering these systems vulnerable in
practical deployments. On the other hand, prompt-engineered agents interact with real-world search
engines, but the model parameters are not optimized through learning. As a result, the models do
not truly learn how to interact effectively with search tools or adapt their behavior to open world
environments. This motivates the development of methods that teach models to search on demand
and interact effectively with search tools, ensuring practical usability in dynamic real-world settings.
Recent advances, such as OpenAI’s o series [ 25,48] and DeepSeek-R1 [ 20], have highlighted the
potential of end-to-end reinforcement learning (RL) to enhance the reasoning capabilities of large-
scale models. In addition, OpenAI introduced the Deep Research [ 47], claiming that training models
via end-to-end RL to interact with search tools and web content can significantly improve their ability
to solve complex open-ended tasks that require iterative reasoning and information seeking. In the
open-source community, efforts such as DeepResearcher [ 72], Search-R1 [ 27], and ReSearch [ 7]
have followed this direction, applying end-to-end RL to improve models’ abilities in multiturn search
and retrieval-augmented generation, aiming to boost performance on information-seeking question
answering tasks. However, existing work mainly focuses on text-based search, while in the multimodal
domain, challenges such as constructing suitable training data and designing RL frameworks to
incentivize models to perform search in real-world environments remain underexplored.
In this work, we focus on training LMMs to learn three key search-related abilities: (1) when to
search , (2) what to search for , and (3) how to reason over search results to answer user queries .
By exploring these questions, we propose MMSearch-R1 , the first end-to-end RL-based solution
designed to equip LMMs with the capability to perform search on demand in real-world internet
environments. Our efforts can be summarized as follows:
•Datasets Construction We propose an automated method for constructing a multimodal
search VQA dataset by estimating the model’s familiarity with the visual and textual
knowledge required to answer each question. Based on this estimation, we generate a
mixture of search-required and search-free VQA samples, which is crucial for shaping
the model’s ability to perform on-demand search. In addition, we complement the dataset
with manually annotated test data that covers diverse knowledge categories and a range of
difficulty levels, ensuring a comprehensive evaluation of the model’s performance.
2

•Multimodal Search Tool Integration We build a real-world search pipeline consisting of
two tools: an image search tool, which allows the model to retrieve webpage thumbnails
and titles related to a user-provided image to help identify unfamiliar visual content; and a
text search tool, which enables the model to issue precise textual queries to retrieve relevant
webpages and acquire textual knowledge.
•Better Performance through Wiser Search We demonstrate that by incorporating an
outcome-based reward with search penalty, the GRPO algorithm [ 55] can be directly applied
to LMMs without cold-start initialization. This setup encourages models to perceive their
knowledge boundaries and then perform search when necessary. As a result, the model
learns to reason about when and how to execute multi-turn, complex search strategies.
In knowledge-intensive and information-seeking VQA tasks, MMSearch-R1-7B not only
outperforms RAG-based baselines of the same model size, but also achieves competitive
performance compared to a 32B RAG-based model, while significantly reducing the number
of search calls by over 30% .
•Open-Sourcing Data and Training Framework We conduct extensive experiments to
derive key empirical findings, which we share with the community to provide deeper insights
into search-augmented multimodal reasoning. In addition, we will open source our data and
the complete training framework to facilitate further research in this area.
2 Building Iterative Multimodal Search-Integrated RL Framework
2.1 Group Relative Policy Optimization (GRPO)
As shown in the top part of Figure 2, we adopt standard GRPO as our base RL algorithm, with
modifications to allow search interactions with the real-world environment during the rollout process.
Originally introduced in DeepSeekMath [ 55], GRPO is a variant of the Proximal Policy Optimization
(PPO) algorithm [ 53]. Unlike PPO, GRPO estimates the baseline directly from a group of rewards,
without relying on a value function, which significantly reduces the computational burden during
training. The details of the GRPO algorithm are provided in the Appendix B.
2.2 Multimodal Search Tools
A fully functional search toolkit is crucial for solving information-seeking VQA tasks. As illustrated
in the bottom part of Figure 2, we equip the model with two types of search tools for interacting
with real-world internet content. The first is an image search tool powered by SerpApi.1The model
submits the image from the original question to the image search engine, which returns the top-5
visually matched webpages in an interleaved format, each represented by a thumbnail and a title.
This helps the model identify the key visual entities present in the input image. The second is a
text search pipeline composed of SerpApi, Jina Reader2, and a webpage summarizer. The model
autonomously generates a text query related to the original question and submits it to the search tool
to retrieve relevant information. SerpApi returns the top-5 webpage URLs associated with the query.
Jina Reader then fetches the full content of each webpage and converts it into a clean, readable format.
A summarization model (Qwen3-32B [ 50] in our implementation) processes each page and extracts
the content most relevant to the user’s question into a concise summary. Implementation details can
be found in the Appendix C.
2.3 Rollout with Multi-turn Multimodal Search
To address complex information-seeking tasks, the model may engage in multiple rounds of interaction
with the internet environment before providing a final answer. As illustrated in the bottom part of
Figure 2, the rollout process is multi-turn and iterative. We structure the prompt to guide the model to
perform reasoning within the <reason> and </reason> tags whenever new information is received,
including the original question or any retrieved search results, and then choose an action from a
predefined action space (see Appendix D.2 for the full prompt). If the model decides to answer the
question, it is required to place its response between <answer> and </answer>. If it chooses to invoke
1SerpApi: https://serpapi.com/
2Jina Reader: https://jina.ai/reader/
3

Policy
Modelq�1Reference
Model
�2
��...�1
�2
��...�1
�2
��...Group
ComputationGRPO with Multi-turn Search 
Multimodal Search Tools
Reward
Model
Rollout with Multi-turn SearchKL
Rollout with Multi-turn Search
Policy
Model
Multimodal 
Search Tools
Reason Before Taking Actions
<reason>
... I don’t need other info to answer ...
... I need extra info to better answer the question. I will call image/text search ...
</reason>
(a). Provide the Answer
<reason> The aircraft is Supermarine Spitfire designed by R.J. 
Mitchell. I can answer the question without any external 
information. </reason>
<answer> R.J. Mitchell </answer>Action Space
(a). Provide the Answer;
(b). Call image search;
(c). Call text search;
(b). Call Image Search ToolLMM
Search Tool
(b). Call Text Search Tool<reason> The image lacks sufficient information 
to determine the specific model of the aircraft. 
An image search is required to assist in 
identifying the precise model. </reason>
<search><img></search>
image urlConducting image search ......
<information>
<reason> The question requires specific 
historical knowledge about the designer of the 
Supermarine Spitfire. I need to conduct a text 
search to find the answer. </reason>
<text_search> Who designed the Supermarine 
Spitfire </text_search>Conducting text search ......
Question:
Who designed this aircraft?
Who designed 
the Supermarine 
Spitfire
<information>
1. (link: xxx) R.J. Mitchell is best known 
for designing the Supermarine Splitfire ...
2. (link: xxx)
...
</information>
Thumbnail-1:
Title-1: 
Supermarine SpitfireThumbnail-2:
Title-2: 
PR MK II 
...       </information>
Figure 2: Illustration of training in MMSearch-R1. Top: The GRPO training pipeline integrated with
multimodal search tools. Bottom: A detailed view of the rollout process and search tool execution.
the image search tool, it should append <search><img></search> at the end of its response. For text
search, the model should autonomously generate a natural language query and append it between
<text_search> and </text_search>. All retrieved information is returned to the model enclosed within
<information> and </information>, and is fed into the next round of dialogue. This iterative process
continues until the model provides a final answer or reaches the maximum number of allowed turns.
To prevent training bias from environment feedback, retrieved content from search tools is masked
during loss computation and does not contribute to gradient updates.
2.4 Reward Modeling
Reward modeling plays a critical role in RL training, as it encodes the desired model behavior and
directly guides the optimization process. In MMSearch-R1, the reward consists of two components:
an accuracy score with search penalty and a format score.
Accuracy Score with Search Penalty We use the exact string match to evaluate whether the final
answer provided by the model is consistent with the ground truth. If the match is exact, the accuracy
score is 1; otherwise, it is 0. For correct answers, we further check whether the model relied on
search tools to arrive at the answer. A penalty factor (ranging from 0 to 1) is applied to the accuracy
score when any search was performed. This design encourages the model to first exploit its internal
knowledge and invoke search tools only when necessary, thereby shaping on-demand search behavior.
Format Score This component checks whether the multi-turn responses of the model strictly follow
the predefined prompt format. Specifically, the model is required to reason between <reason> and
</reason> before taking any action, take only one action per turn, place search patterns such as
<search><img></search> or <text_search> text query </text_search> at the end of its response when
a search is performed, and wrap the final answer between <answer> and </answer> when concluding.
The format score is assigned a value of 1 only if all responses fully adhere to these formatting
requirements; otherwise, it is set to 0. We combine the two components using a weighting coefficient
α, and define the final reward as a weighted sum of the accuracy and format scores as follows:
reward = (1−α)·Acc_Score ·Search _Penalty +α·Format _Score. (1)
4

GPT4o
QA Gen
Visual Concepts (VC)Web
Search
Q: What event does the picture depict?
A: Sinking of Titanic.
Q: Which school is the team in red 
uniforms from?
A: University of Southern California.
...
Q: What is the name of this building?
A: Griffith Observatory.
Retrieved Images and Webpages Factual QA Generation
(a). Pipeline of Visual Knowledge-required QA CollectionQ: What place is shown in the picture?
A: Togetsukyo Bridge.
(b). Knowledge Taxonomy
(c). Data Construction PipelineData Processing Type
(Automated) Category-
balanced Sampling
Human Annotated(Automated) Search 
Required/Free-balanced 
Sampling
InfoSeek
train split
FVQA-auto-vc
FVQA-human
InfoSeek-
human-subFVQA-auto-txt
FVQA-auto-vc-train
FVQA-auto-vc-testFVQA-human-trainsearch-free
image-search-required
text-search-required
mix-search-required
FVQA-auto-vc-test
human checkedFVQA-train
FVQA-testFigure 3: Illustration of data construction process of FVQA : (a). An automated pipeline for visual
knowledge-required VQA samples collection; (b). Knowledge taxonomy; (c). Overall pipeline
showing the composition and origin of FVQA from various automatic and manually curated sources.
3 Curating Search-balanced VQA Datasets
We aim to explore the potential of training models for on-demand search using simple outcome-based
reward reinforcement learning. Therefore, the datasets should satisfy three criteria: (1). Coverage of
Both Search-Required and Search-Free Questions The datasets should include a mix of search-free
and search-required questions. A search-free question can be answered solely using the model’s
internal knowledge, whereas a search-required question involves information beyond the model’s
existing knowledge and therefore requires access to external information sources. Search-required
questions can be further categorized into Visual Knowledge-required and Textual Knowledge-required
types. Visual Knowledge refers to the model’s ability to recognize visual entities in an image (e.g.,
"What is the model of the aircraft shown in the image?"), while Textual Knowledge refers to
factual information about the visual entity (e.g., "Who is the designer of Supermarine Spitfire?").
Existing datasets such as OK-VQA [ 43] and InfoSeek [ 9] typically use image sources already seen by
current multimodal models and tend to focus on Textual Knowledge-required questions. Meanwhile,
MMSearch [ 26] collects images from recent news, aligning better with the Visual Knowledge-required
criteria. However, it is relatively small in scale and mainly serve as benchmarks, leading to a lack of
sufficient Visual Knowledge-required training data. (2). Concise and Unambiguous Answers for
Reliable Verification VQA questions should be designed to elicit concise and unambiguous answers
that can be easily and reliably verified through simple, rule-based reward mechanisms. Such questions
typically revolve around factual knowledge, enabling automated evaluation during reinforcement
learning. (3). Diversity in Knowledge Categories and Question Difficulty The dataset should cover
diverse knowledge domains and difficulty levels to ensure broad real-world generalization.
To support our investigation, we construct a multimodal search VQA dataset, FactualVQA (FVQA) ,
using a combination of automated pipelines and manual annotation, covering both training and
evaluation needs, as illustrated in Figure 3(c).
3.1 Training Dataset Construction
To meet the above requirements, we select suitable training data from multiple sources through two
main processes: VQA Collection and Search Balancing.
VQA Collection We first developed an automated annotation pipeline to collect Visual Knowledge-
required training data, as illustrated in Figure 3(a). To explore visual concepts that models are
relatively familiar or unfamiliar with, we begin with the Metadata distribution of MetaCLIP [ 63].
This Metadata is constructed from multiple sources such as WordNet and Wikipedia, covering a
wide range of Visual Concepts (VC) from common to rare. Designed to guide balanced dataset
construction for vision-language pretraining, it exhibits a long-tailed distribution. Concepts in the
5

head of the distribution correspond to commonly referenced items in the real world (e.g., car, tree),
whereas those in the tail represent less common or niche concepts (e.g., thylacine, astrolabe). We
randomly sampled 10,000 visual concepts from both the head and the tail of the Metadata distribution.
For each concept, we performed a web search to retrieve the most relevant image and its associated
webpage. These image–webpage pairs were then input to GPT-4o, which was prompted to generate a
factual visual question-answer (VQA) pair centered around the given visual concept. The generated
answers were required to be concise and precise. The prompts used for QA generation and two
examples can be found in Appendix D.1 and J. Next, GPT-4o was employed to classify the knowledge
type assessed by each question, resulting in a knowledge taxonomy as illustrated in Figure 3(b).
Based on this taxonomy, we performed balanced sampling across categories and curated a final set of
6,000 VQA samples, referred to as FVQA-auto-vc , with 5,400 used for training and 600 for testing.
In parallel, to enrich the dataset with textual knowledge-required examples, we sampled from the
open-source InfoSeek [ 9] dataset. Specifically, we classified the questions in the InfoSeek training
split by the type of knowledge they require and mapped them into the same taxonomy. After balanced
sampling across categories, we obtained 7,000 samples, referred to as FVQA-auto-txt . Finally, to
further diversify the dataset with real-user queries, we manually annotated an additional 800 samples,
referred to as FVQA-manual-train . Annotators were instructed to select a knowledge category from
the taxonomy, locate a relevant image, and pose a factual question pertaining to that category. They
were allowed to perform both visual and textual searches until sufficient information was gathered to
formulate an accurate answer, from which a concise and precise response was extracted.
Search Balancing The goal of this stage is to distinguish between search-required and search-free
questions within the collected data. To this end, we first trained a Qwen2.5-VL-Instruct-7B model
using our training framework on the full set of samples obtained from the VQA collection process.
This model was then used to classify the original questions: for each question, 8 rollouts were
generated. If all 8 rollouts failed, the question was discarded due to insufficient training signal. A
question was labeled as image-, text-, or mixed-search-required if the model produced correct answers
only when the corresponding type of search behavior was performed. In particular, mixed-search
indicates that both image and text searches needed to be executed for the model to answer correctly. If
any rollout produced a correct answer without invoking search, the question was labeled as search-free.
Finally, we constructed a search-balanced training set of 5,000 samples, referred to as FVQA-train ,
consisting of approximately 3,400 search-required and 1,600 search-free VQA examples. Maintaining
a balanced distribution of search types is crucial to shaping the search behavior of the model during
training.
3.2 Test Dataset Annotation
To better evaluate the model’s performance, we additionally constructed a high-quality test set, where
all examples were either manually verified or fully human-annotated to ensure accuracy. The test set,
referred to as FVQA-test , includes 1800 examples collected from three sources: (1) 600 samples
drawn from FVQA-auto-vc , ensuring no overlap with the training set, with each example manually
checked for correctness; (2) 600 samples selected from the InfoSeek Human Split, where answers
were manually annotated, as the original human-labeled answers were not publicly available; (3) 600
samples collected directly from the manual annotation process described above.
4 Experiments
4.1 Setups
Implementation Details We built our training framework based on veRL [ 56] and conduct ex-
periments on Qwen2.5-VL-7B-Instruct [ 5]. The model was trained using the dataset described in
Section 3. At each training step, we sampled 512 examples, with each example undergoing 8 rollouts.
Each rollout consists of up to three rounds of dialogue, during which the model can perform at most
two search actions and is required to produce a final answer in the third round. Image search is
only allowed in the first round, and each image search returns up to 5 top visual matched webpages
in the form of interleaved thumbnails and titles. Text search, on the other hand, returns up to 5
summarized webpage contents per query. The search penalty factor is set to 0.1. The weighting
coefficient αbetween the accuracy reward and the format reward is set to 0.1. Further details on the
hyperparameter settings can be found in Appendix F.1.
6

Table 1: Performance of MMSearch-R1 across benchmarks. "Acc (%)" denotes the accuracy evaluated
by LLM-as-Judge, while "SR (%)" represents the search ratio, defined as the percentage of total
search calls made relative to the maximum allowed search steps for each method.
ModelAverageIn-Domain Out-of-Domain
FVQA-test InfoSeek MMSearch SimpleVQA LiveVQA
Acc SR Acc SR Acc SR Acc SR Acc SR Acc SR
Direct Answer
GPT4o 36.0 0 41.7 0 42.7 0 22.2 0 46.6 0 26.9 0
Gemini 2.5 Pro 36.4 0 37.2 0 37.0 0 26.9 0 53.4 0 27.7 0
Qwen2.5-VL-72B 26.6 0 27.1 0 28.0 0 15.7 0 42.2 0 20.1 0
Qwen2.5-VL-32B 25.0 0 24.7 0 25.8 0 15.7 0 40.1 0 18.7 0
Qwen2.5-VL-7B 21.9 0 20.3 0 20.1 0 12.8 0 38.4 0 17.8 0
RAG Workflow
GPT4o 62.1 100 66.0 100 59.1 100 62.5 100 63.4 100 59.6 100
Gemini 2.5 Pro 61.8 100 66.1 100 56.7 100 62.5 100 65.9 100 57.8 100
Qwen2.5-VL-72B 59.6 100 62.2 100 59.4 100 59.6 100 61.0 100 56.0 100
Qwen2.5-VL-32B 55.1 100 57.0 100 56.8 100 57.9 100 54.5 100 49.6 100
Qwen2.5-VL-7B 51.6 100 52.9 100 53.7 100 52.2 100 51.6 100 48.0 100
On-demand Search
MMSearch-R1-7B 54.6 67.1 58.4 66.8 55.1 61.6 53.8 88.5 57.4 42.5 48.4 76.2
Benchmark We selected FVQA-test , InfoSeek [ 9], MMSearch [ 26], SimpleVQA [ 13] and LiveVQA
[18] as benchmark datasets to evaluate the model’s ability to handle both knowledge-intensive and
information-seeking VQA tasks. Specifically, for InfoSeek, we randomly sampled 2k examples from
its test split due to the large dataset size. For MMSearch, we filter and retain only the QA pairs that
include images. For SimpleVQA, we extracted all QA examples written in English. Among these
benchmarks, MMSearch, SimpleVQA and LiveVQA serve as out-of-distribution (OOD) testsets for
our trained models. Details of the benchmark datasets are provided in Appendix E.
Baselines To validate the effectiveness of MMSearch-R1, we evaluated against both closed-source
models (GPT-4o and Gemini 2.5 Pro) and open-source models from the Qwen2.5-VL series. All
models are tasked with solving VQA problems in two different workflows. (1) Direct Answer: Models
are prompted to directly generate a short and precise answer without accessing external information.
(2) Answer under RAG Workflow: In this workflow, models are required to perform exactly two
search operations using our multimodal search tools for each VQA example, first performing an
image search and then a text search. Specifically, given an input image and question, the model is
provided with the image search results and the original question in the first round and is prompted to
generate a text query to assist in answering. In the second round, the retrieved results based on the
text query are fed into the model, and the model is asked to produce the final answer. Under a fixed
budget of search steps, the RAG workflow typically exposes the model to more external information
compared to the on-demand search strategy. The prompts used for the RAG workflow are provided in
Appendix D.5.
Metric We adopt the LLM-as-Judge framework as our evaluation metric to assess the accuracy of
model responses. In addition, we record the search ratio, the proportion of responses that invoke
a search, for each method across different benchmarks, allowing us to jointly evaluate answer
correctness and search efficiency. Specifically, we employ GPT-4o as the judging model, which
receives the original image, the question, the ground-truth answer, and the model’s response to
determine correctness. The prompt used for judgment is detailed in the Appendix D.6.
4.2 Findings
In this section, we present key empirical findings that emerged from our experiments. Each finding is
supported by quantitative results and detailed analysis, aiming to provide deeper insights into the
behavior and effectiveness of MMSearch-R1. Through these findings, we aim to both validate our
approach and provide actionable insights for the broader research community.
7

(a). Model Performance w/ RAG Workflow OOD Tasks ID Tasks35
30404550556065 Qwen2.5VL-7B
MMSearch-R1-7B+6.51
+6.18
(b). Breakdown of answer behaviors for Base (Inner) and RL (Outer) models.17.6%
21.6%53.9%
6.9%33.5%14.3%
21.6%
30.6%25.8%59.5%
11.3%3.4%28.4%16.4%
29.0%
26.2%Correct with Search
Correct without Search
Wrong with Search
Wrong without Search
InfoSeek SimpleVQAFigure 4: (a). Performance comparison between the Base model and the RL-trained model under the
RAG workflow. (b). Answer behavior breakdown of Base (inner circle) and RL (outer circle) models
in InfoSeek and SimpleVQA.
Finding 1: RL training enables models to better recognize the boundaries of their knowledge
and perform on-demand search more effectively. As shown in Table 1, on both in-domain and
out-of-domain test sets, MMSearch-R1-7B outperforms the RAG-based counterparts of the same size
by an average of 3%in accuracy, while reducing the average search rate by 32.9% . This indicates
that our RL-trained model not only achieves higher correctness but also relies less on external
information, thereby exhibiting a more efficient and targeted use of search. Notably, MMSearch-R1-
7B performs competitively with RAG-based Qwen2.5-VL-32B, a significantly larger model, further
highlighting the benefits of learning adaptive search behavior, as opposed to executing fixed-stage
retrieval regardless of necessity. These results validate the effectiveness of outcome-based RL in
enabling intelligent, cost-aware search behavior.
Finding 2: RL training enhances the model’s ability to generate effective text queries and sum-
marize retrieved information. Our RL training jointly equips model with three key capabilities:
deciding whether to search, determining what to search for, and extracting useful information from
retrieved results. To evaluate the latter two abilities in isolation, we adopt the same RAG setting as
in the baseline comparison, where both image and text search are executed for every question. In
this setup, the search process is fixed, and the model is responsible only for generating effective text
queries and reasoning over the retrieved content. This removes variability in search triggering and
allows us to focus on evaluating how well the model interacts with external information.
As shown in Figure 4(a), MMSearch-R1-7B demonstrates consistent improvements over the base
model across both in-domain and out-of-domain tasks. These results indicate that reinforcement
learning improves not only the decision of when to search, as shown earlier, but also enhances two
core retrieval abilities: generating more accurate, contextually relevant queries and extracting useful
information to support accurate answers. These gains appear across both image and text retrieval,
highlighting the broader value of RL in strengthening retrieval and reasoning capabilities.
Finding 3: RL improves the model’s ability to utilize its internal knowledge. As shown in Fig-
ure 4(b), we conduct a behavioral breakdown of responses on datasets InfoSeek and SimpleVQA
to better understand how the model’s behavior changes after reinforcement learning. The results
reveals that a clear upward trend in the Correct without Search proportion from the base model to the
RL-trained model. These gains indicate that the RL-trained model can answer substantially more
questions correctly without invoking the search tool, demonstrating improved recall and reasoning
based on its internal knowledge. This shift suggests that reinforcement learning enhances the model’s
ability to rely on its own parameters when sufficient, and to reserve external search for genuinely
novel or long-tail queries. As a result, the model exhibits more accurate on-demand search behavior,
engaging external tools only when internal knowledge is insufficient.
Finding 4: RL achieves greater performance improvements and exhibits higher data efficiency
compared to supervised SFT. To investigate the effectiveness of RL and SFT as training paradigms,
we compare their performance gains and data efficiency through controlled experiments. For RL, we
use the aforementioned FVQA-train set, which contains 5k VQA samples in the form of (image,
question, answer) triplets. For SFT, we first construct a dataset by aggregating three sources:
FVQA-auto-vc ,FVQA-auto-txt , and FVQA-human-train , resulting in 13.2k examples. We then
8

(a). Performance Gains over Base Model from SFT and RLFVQA-test InfoSeek MMSearch SimpleVQA LiveVQA010203040506070Acc(%)Base SFT(8k) RL(5k)
RL SR SFT SR
102030405060708090100Search Ratio (SR) (%)36.9%38.6% 50.0%
21.2%41.6%66.8%
61.6%88.6%
42.6%76.3%
(b). Training Dynamics of reward (left) and search ratio (right)0 20 40 60 801000.10.20.30.40.50.60.70.8
0.30.40.50.60.70.80.91.0
0 20 40 60 80100 steps stepsSearch RatioReward
MMSearch-R1
w/o Data Balancing
w/o Search PenaltyMMSearch-R1
w/o Data Balancing
w/o Search PenaltyFigure 5: (a). Performance improvements of SFT and RL over Base across five VQA datasets. (b).
Training dynamics of reward and search ratio for different strategies
distill GPT-4o’s behavior on this dataset by prompting it with the same instructions used in RL
training, allowing it to autonomously decide whether to invoke external tools when answering a
question. This process yields 13.2k responses from GPT-4o, each including an image, question, and a
response containing its reasoning process and any tool usage, often in a multi-turn dialogue format.
During SFT training, we mask out the prompts and all tool-returned content from the multi-turn
dialogues, using only GPT-4o’s responses as supervision signals. This ensures that the model learns
to mimic GPT-4o’s reasoning and answering behavior without relying on intermediate tool outputs,
thereby encouraging it to internalize the reasoning process demonstrated by the teacher model. To
ensure training quality, we compare GPT-4o’s final predicted answers with the original ground truth
answers and filter out 5.2k samples where GPT-4o produced incorrect answers. The remaining 8k
examples constitute our SFT training set. The implementation details of SFT can be found in the
Appendix F.2.
We fine-tune the base model, Qwen2.5-VL-7B, using the above datasets for both SFT and RL training.
We then evaluate the performance of the models trained with SFT and RL on all downstream tasks
and compare their improvements over the base model, as shown in Figure 5(a). The results show
that the model trained with RL consistently outperforms the one trained with SFT across all tasks,
despite being trained on only about half as much data. In addition, the behavior of the RL-trained
model aligns more closely with the specific requirements of each task, especially in its use of search
tools. For example, in the MMSearch and LiveVQA tasks, which are both recently collected and
focus on information-seeking questions, the RL-trained model shows a higher frequency of search
tool invocation. This is consistent with expectations, as these datasets contain many questions that
require external visual or textual knowledge not encountered during pretraining. The RL-trained
model learns to rely more effectively on search tools in such scenarios.
Finding 5: Training with balanced data and a search penalty in the reward effectively guide the
model to perform on-demand search. Figure 5(b) illustrates the training dynamics of reward and
search ratio during reinforcement learning. Removing either the search penalty or data balancing
leads to distinct trade-offs. Although both ablated variants achieve slightly higher rewards, they do
so at the cost of overusing the search tool, with search ratios rapidly converging to nearly 100%. In
contrast, MMSearch-R1, trained with both data balancing and a search penalty, achieves a comparable
reward while maintaining a significantly lower and more stable search ratio. This suggests that the
model has learned to invoke the search tool only when necessary, enabling more efficient and selective
on-demand search behavior.
5 Conclusion
In this work, we present MMSearch-R1, a RL-based framework that equips LMMs with the ability to
perform on-demand search in real-world internet environments. MMSearch-R1 learns to recognize
knowledge gaps, selectively invoke image or text search, and reason over retrieved content. It
outperforms same-sized RAG baselines and approaches the performance of larger models while
requiring significantly fewer search calls. Our framework, dataset, and findings offer practical
insights into training LMMs with real-world interaction capabilities and lay the groundwork for
building multimodal agents that are both adaptive and interactive. We look forward to the next major
advancement in multimodal intelligence emerging as models increasingly engage with and explore
the real world through more tools, further evolving their reasoning and adaptive capabilities.
9

References
[1]Salaheddin Alzubi, Creston Brooks, Purva Chiniya, Edoardo Contente, Chiara von Gerlach,
Lucas Irwin, Yihan Jiang, Arda Kaz, Windsor Nguyen, Sewoong Oh, et al. Open deep search:
Democratizing search with open-source reasoning agents. arXiv preprint arXiv:2503.20201 ,
2025.
[2]Anthropic. Claude 3.5 Sonnet. https://www.anthropic.com/news/
claude-3-5-sonnet/ . Technical Report, 2024.
[3]Akari Asai, Zeqiu Wu, Yizhong Wang, Avirup Sil, and Hannaneh Hajishirzi. Self-rag: Learn-
ing to retrieve, generate, and critique through self-reflection. In The Twelfth International
Conference on Learning Representations , 2023.
[4]Anas Awadalla, Le Xue, Oscar Lo, Manli Shu, Hannah Lee, Etash Guha, Sheng Shen, Mohamed
Awadalla, Silvio Savarese, Caiming Xiong, et al. Mint-1t: Scaling open-source multimodal
data by 10x: A multimodal dataset with one trillion tokens. Advances in Neural Information
Processing Systems , 2024.
[5]Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang,
Shijie Wang, Jun Tang, et al. Qwen2. 5-vl technical report. arXiv preprint arXiv:2502.13923 ,
2025.
[6]Hoyeon Chang, Jinho Park, Seonghyeon Ye, Sohee Yang, Youngkyung Seo, Du-Seong Chang,
and Minjoon Seo. How do large language models acquire factual knowledge during pretraining?
InThe Thirty-eighth Annual Conference on Neural Information Processing Systems , 2024.
[7]Mingyang Chen, Tianpeng Li, Haoze Sun, Yijie Zhou, Chenzheng Zhu, Fan Yang, Zenan Zhou,
Weipeng Chen, Haofen Wang, Jeff Z Pan, et al. Learning to reason with search for llms via
reinforcement learning. arXiv preprint arXiv:2503.19470 , 2025.
[8]Wenhu Chen, Hexiang Hu, Xi Chen, Pat Verga, and William W Cohen. Murag: Multimodal
retrieval-augmented generator for open question answering over images and text. arXiv preprint
arXiv:2210.02928 , 2022.
[9]Yang Chen, Hexiang Hu, Yi Luan, Haitian Sun, Soravit Changpinyo, Alan Ritter, and Ming-
Wei Chang. Can pre-trained vision and language models answer visual information-seeking
questions? arXiv preprint arXiv:2302.11713 , 2023.
[10] Zhanpeng Chen, Chengjin Xu, Yiyan Qi, and Jian Guo. Mllm is a strong reranker: Ad-
vancing multimodal retrieval-augmented generation via knowledge-enhanced reranking and
noise-injected training. arXiv preprint arXiv:2407.21439 , 2024.
[11] Zhe Chen, Jiannan Wu, Wenhai Wang, Weijie Su, Guo Chen, Sen Xing, Muyan Zhong, Qinglong
Zhang, Xizhou Zhu, Lewei Lu, et al. Internvl: Scaling up vision foundation models and aligning
for generic visual-linguistic tasks. In Proceedings of the IEEE/CVF conference on computer
vision and pattern recognition , 2024.
[12] Daixuan Cheng, Shaohan Huang, Junyu Bi, Yuefeng Zhan, Jianfeng Liu, Yujing Wang, Hao
Sun, Furu Wei, Denvy Deng, and Qi Zhang. Uprise: Universal prompt retrieval for improving
zero-shot evaluation. arXiv preprint arXiv:2303.08518 , 2023.
[13] Xianfu Cheng, Wei Zhang, Shiwei Zhang, Jian Yang, Xiangyuan Guan, Xianjie Wu, Xiang Li,
Ge Zhang, Jiaheng Liu, Yuying Mai, et al. Simplevqa: Multimodal factuality evaluation for
multimodal large language models. arXiv preprint arXiv:2502.13059 , 2025.
[14] Claude. Claude takes research to new places. https://www.anthropic.com/news/
research/ . Technical Report, 2025.
[15] Chenhang Cui, Yiyang Zhou, Xinyu Yang, Shirley Wu, Linjun Zhang, James Zou, and Huaxiu
Yao. Holistic analysis of hallucination in gpt-4v (ision): Bias and interference challenges. arXiv
preprint arXiv:2311.03287 , 2023.
10

[16] Hongyuan Dong, Zijian Kang, Weijie Yin, Xiao Liang, Chao Feng, and Jiao Ran. Scalable
vision language model training via high quality data curation. arXiv preprint arXiv:2501.05952 ,
2025.
[17] Chaoyou Fu, Peixian Chen, Yunhang Shen, Yulei Qin, Mengdan Zhang, Xu Lin, Jinrui Yang,
Xiawu Zheng, Ke Li, Xing Sun, Yunsheng Wu, and Rongrong Ji. Mme: A comprehensive
evaluation benchmark for multimodal large language models. arXiv preprint arXiv:2306.13394 ,
2023.
[18] Mingyang Fu, Yuyang Peng, Benlin Liu, Yao Wan, and Dongping Chen. Livevqa: Live visual
knowledge seeking. arXiv preprint arXiv:2504.05288 , 2025.
[19] Google. Try Deep Research and our new experimental model in Gemini, your AI assistant.
https://blog.google/products/gemini/google-gemini-deep-research/ . Techni-
cal Report, 2025.
[20] Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu,
Shirong Ma, Peiyi Wang, Xiao Bi, et al. Deepseek-r1: Incentivizing reasoning capability in
llms via reinforcement learning. arXiv preprint arXiv:2501.12948 , 2025.
[21] Ziniu Hu, Ahmet Iscen, Chen Sun, Kai-Wei Chang, Yizhou Sun, David Ross, Cordelia Schmid,
and Alireza Fathi. Avis: Autonomous visual information seeking with large language model
agent. Advances in Neural Information Processing Systems , 2023.
[22] Ziniu Hu, Ahmet Iscen, Chen Sun, Zirui Wang, Kai-Wei Chang, Yizhou Sun, Cordelia Schmid,
David A Ross, and Alireza Fathi. Reveal: Retrieval-augmented visual-language pre-training
with multi-source multimodal knowledge memory. In Proceedings of the IEEE/CVF conference
on computer vision and pattern recognition , 2023.
[23] Aaron Hurst, Adam Lerer, Adam P Goucher, Adam Perelman, Aditya Ramesh, Aidan Clark,
AJ Ostrow, Akila Welihinda, Alan Hayes, Alec Radford, et al. Gpt-4o system card. arXiv
preprint arXiv:2410.21276 , 2024.
[24] Gautier Izacard, Patrick Lewis, Maria Lomeli, Lucas Hosseini, Fabio Petroni, Timo Schick,
Jane Dwivedi-Yu, Armand Joulin, Sebastian Riedel, and Edouard Grave. Few-shot learning
with retrieval augmented language models. arXiv preprint arXiv:2208.03299 , 2022.
[25] Aaron Jaech, Adam Kalai, Adam Lerer, Adam Richardson, Ahmed El-Kishky, Aiden Low, Alec
Helyar, Aleksander Madry, Alex Beutel, Alex Carney, et al. Openai o1 system card. arXiv
preprint arXiv:2412.16720 , 2024.
[26] Dongzhi Jiang, Renrui Zhang, Ziyu Guo, Yanmin Wu, Jiayi Lei, Pengshuo Qiu, Pan Lu, Zehui
Chen, Chaoyou Fu, Guanglu Song, et al. Mmsearch: Benchmarking the potential of large
models as multi-modal search engines. arXiv preprint arXiv:2409.12959 , 2024.
[27] Bowen Jin, Hansi Zeng, Zhenrui Yue, Jinsung Yoon, Sercan Arik, Dong Wang, Hamed Za-
mani, and Jiawei Han. Search-r1: Training llms to reason and leverage search engines with
reinforcement learning. arXiv preprint arXiv:2503.09516 , 2025.
[28] Nikhil Kandpal, Haikang Deng, Adam Roberts, Eric Wallace, and Colin Raffel. Large language
models struggle to learn long-tail knowledge. In International Conference on Machine Learning ,
2023.
[29] Vladimir Karpukhin, Barlas Oguz, Sewon Min, Patrick SH Lewis, Ledell Wu, Sergey Edunov,
Danqi Chen, and Wen-tau Yih. Dense passage retrieval for open-domain question answering.
InEMNLP , 2020.
[30] Aniruddha Kembhavi, Mike Salvato, Eric Kolve, Minjoon Seo, Hannaneh Hajishirzi, and Ali
Farhadi. A diagram is worth a dozen images. In Computer Vision–ECCV 2016: 14th European
Conference, Amsterdam, The Netherlands, October 11–14, 2016, Proceedings, Part IV 14 , 2016.
[31] Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman
Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, et al. Retrieval-augmented
generation for knowledge-intensive nlp tasks. Advances in neural information processing
systems , 2020.
11

[32] Bo Li, Yuanhan Zhang, Dong Guo, Renrui Zhang, Feng Li, Hao Zhang, Kaichen Zhang, Peiyuan
Zhang, Yanwei Li, Ziwei Liu, et al. Llava-onevision: Easy visual task transfer. arXiv preprint
arXiv:2408.03326 , 2024.
[33] Dongxu Li, Yudong Liu, Haoning Wu, Yue Wang, Zhiqi Shen, Bowen Qu, Xinyao Niu, Fan
Zhou, Chengen Huang, Yanpeng Li, et al. Aria: An open multimodal native mixture-of-experts
model. arXiv preprint arXiv:2410.05993 , 2024.
[34] Yifan Li, Yifan Du, Kun Zhou, Jinpeng Wang, Wayne Xin Zhao, and Ji-Rong Wen. Evaluating
object hallucination in large vision-language models. arXiv preprint arXiv:2305.10355 , 2023.
[35] Yunxin Li, Longyue Wang, Baotian Hu, Xinyu Chen, Wanqi Zhong, Chenyang Lyu, Wei Wang,
and Min Zhang. A comprehensive evaluation of gpt-4v on knowledge-intensive visual question
answering. arXiv preprint arXiv:2311.07536 , 2023.
[36] Li, Bo and Zhang, Kaichen and Zhang, Hao and Guo, Dong and Zhang, Renrui and Li,
Feng and Zhang, Yuanhan and Liu, Ziwei and Li, Chunyuan. LLaV A-NeXT: Stronger LLMs
Supercharge Multimodal Capabilities in the Wild. https://llava-vl.github.io/blog/
2024-05-10-llava-next-stronger-llms/ . Technical Report, 2024.
[37] Ji Lin, Hongxu Yin, Wei Ping, Pavlo Molchanov, Mohammad Shoeybi, and Song Han. Vila:
On pre-training for visual language models. In Proceedings of the IEEE/CVF conference on
computer vision and pattern recognition , 2024.
[38] Hanchao Liu, Wenyuan Xue, Yifei Chen, Dapeng Chen, Xiutian Zhao, Ke Wang, Liping Hou,
Rongjun Li, and Wei Peng. A survey on hallucination in large vision-language models. arXiv
preprint arXiv:2402.00253 , 2024.
[39] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. Advances
in neural information processing systems , 2023.
[40] Yuliang Liu, Zhang Li, Mingxin Huang, Biao Yang, Wenwen Yu, Chunyuan Li, Xu-Cheng Yin,
Cheng-Lin Liu, Lianwen Jin, and Xiang Bai. Ocrbench: on the hidden mystery of ocr in large
multimodal models. Science China Information Sciences , 2024.
[41] Pan Lu, Hritik Bansal, Tony Xia, Jiacheng Liu, Chunyuan Li, Hannaneh Hajishirzi, Hao
Cheng, Kai-Wei Chang, Michel Galley, and Jianfeng Gao. Mathvista: Evaluating mathematical
reasoning of foundation models in visual contexts. arXiv preprint arXiv:2310.02255 , 2023.
[42] Hongyin Luo, Tianhua Zhang, Yung-Sung Chuang, Yuan Gong, Yoon Kim, Xixin Wu, Helen
Meng, and James Glass. Search augmented instruction learning. In Findings of the Association
for Computational Linguistics: EMNLP 2023 , 2023.
[43] Kenneth Marino, Mohammad Rastegari, Ali Farhadi, and Roozbeh Mottaghi. Ok-vqa: A visual
question answering benchmark requiring external knowledge. In Proceedings of the IEEE/cvf
conference on computer vision and pattern recognition , 2019.
[44] Ahmed Masry, Do Xuan Long, Jia Qing Tan, Shafiq Joty, and Enamul Hoque. Chartqa: A
benchmark for question answering about charts with visual and logical reasoning. arXiv preprint
arXiv:2203.10244 , 2022.
[45] Meta. Llama 3.2: Revolutionizing edge AI and vision with open, customizable models. https:
//ai.meta.com/blog/llama-3-2-connect-2024-vision-edge-mobile-devices/ .
Technical Report, 2024.
[46] Reiichiro Nakano, Jacob Hilton, Suchir Balaji, Jeff Wu, Long Ouyang, Christina Kim, Christo-
pher Hesse, Shantanu Jain, Vineet Kosaraju, William Saunders, et al. Webgpt: Browser-assisted
question-answering with human feedback. arXiv preprint arXiv:2112.09332 , 2021.
[47] OpenAI. Introducing deep research. https://openai.com/index/
introducing-deep-research/ . Technical Report, 2025.
[48] OpenAI. OpenAI o3 and o4-mini System Card. https://openai.com/index/
o3-o4-mini-system-card/ . Technical Report, 2025.
12

[49] Perplexity. Introducing Perplexity Deep Research. https://www.perplexity.ai/hub/
blog/introducing-perplexity-deep-research/ . Technical Report, 2025.
[50] Qwen Team. Qwen3: Think Deeper, Act Faster. https://qwenlm.github.io/blog/
qwen3/ . Technical Report, 2025.
[51] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal,
Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual
models from natural language supervision. In International conference on machine learning ,
2021.
[52] Timo Schick, Jane Dwivedi-Yu, Roberto Dessì, Roberta Raileanu, Maria Lomeli, Eric Hambro,
Luke Zettlemoyer, Nicola Cancedda, and Thomas Scialom. Toolformer: Language models can
teach themselves to use tools. Advances in Neural Information Processing Systems , 36, 2023.
[53] John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal
policy optimization algorithms. arXiv preprint arXiv:1707.06347 , 2017.
[54] Zhihong Shao, Yeyun Gong, Yelong Shen, Minlie Huang, Nan Duan, and Weizhu Chen.
Enhancing retrieval-augmented large language models with iterative retrieval-generation synergy.
arXiv preprint arXiv:2305.15294 , 2023.
[55] Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang,
Mingchuan Zhang, YK Li, Y Wu, et al. Deepseekmath: Pushing the limits of mathematical
reasoning in open language models. arXiv preprint arXiv:2402.03300 , 2024.
[56] Guangming Sheng, Chi Zhang, Zilingfeng Ye, Xibin Wu, Wang Zhang, Ru Zhang, Yanghua
Peng, Haibin Lin, and Chuan Wu. Hybridflow: A flexible and efficient rlhf framework. arXiv
preprint arXiv:2409.19256 , 2024.
[57] Gemini Team, Rohan Anil, Sebastian Borgeaud, Jean-Baptiste Alayrac, Jiahui Yu, Radu Soricut,
Johan Schalkwyk, Andrew M Dai, Anja Hauth, Katie Millican, et al. Gemini: a family of highly
capable multimodal models. arXiv preprint arXiv:2312.11805 , 2023.
[58] Kimi Team, Angang Du, Bofei Gao, Bowei Xing, Changjiu Jiang, Cheng Chen, Cheng Li,
Chenjun Xiao, Chenzhuang Du, Chonghua Liao, et al. Kimi k1. 5: Scaling reinforcement
learning with llms. arXiv preprint arXiv:2501.12599 , 2025.
[59] Kimi Team, Angang Du, Bohong Yin, Bowei Xing, Bowen Qu, Bowen Wang, Cheng Chen,
Chenlin Zhang, Chenzhuang Du, Chu Wei, et al. Kimi-vl technical report. arXiv preprint
arXiv:2504.07491 , 2025.
[60] Boxin Wang, Wei Ping, Lawrence McAfee, Peng Xu, Bo Li, Mohammad Shoeybi, and Bryan
Catanzaro. Instructretro: Instruction tuning post retrieval-augmented pretraining. arXiv preprint
arXiv:2310.07713 , 2023.
[61] Huazheng Wang, Jinming Wu, Haifeng Sun, Zixuan Xia, Daixuan Cheng, Jingyu Wang, Qi Qi,
and Jianxin Liao. Mdr: Model-specific demonstration retrieval at inference time for in-context
learning. In Proceedings of the 2024 Conference of the North American Chapter of the
Association for Computational Linguistics: Human Language Technologies (Volume 1: Long
Papers) , pages 4189–4204, 2024.
[62] Xiao Wang, Ibrahim Alabdulmohsin, Daniel Salz, Zhe Li, Keran Rong, and Xiaohua Zhai.
Scaling pre-training to one hundred billion data for vision language models. arXiv preprint
arXiv:2502.07617 , 2025.
[63] Hu Xu, Saining Xie, Xiaoqing Ellen Tan, Po-Yao Huang, Russell Howes, Vasu Sharma, Shang-
Wen Li, Gargi Ghosh, Luke Zettlemoyer, and Christoph Feichtenhofer. Demystifying clip data.
arXiv preprint arXiv:2309.16671 , 2023.
[64] Shi Yu, Chaoyue Tang, Bokai Xu, Junbo Cui, Junhao Ran, Yukun Yan, Zhenghao Liu, Shuo
Wang, Xu Han, Zhiyuan Liu, et al. Visrag: Vision-based retrieval-augmented generation on
multi-modality documents. arXiv preprint arXiv:2410.10594 , 2024.
13

[65] Yue Yu, Wei Ping, Zihan Liu, Boxin Wang, Jiaxuan You, Chao Zhang, Mohammad Shoeybi,
and Bryan Catanzaro. Rankrag: Unifying context ranking with retrieval-augmented generation
in llms. Advances in Neural Information Processing Systems , 2024.
[66] Kaichen Zhang, Bo Li, Peiyuan Zhang, Fanyi Pu, Joshua Adrian Cahyono, Kairui Hu, Shuai
Liu, Yuanhan Zhang, Jingkang Yang, Chunyuan Li, et al. Lmms-eval: Reality check on the
evaluation of large multimodal models, 2024b. arXiv preprint arXiv:2407.12772 , 2024.
[67] Tianjun Zhang, Shishir G Patil, Naman Jain, Sheng Shen, Matei Zaharia, Ion Stoica, and
Joseph E Gonzalez. Raft: Adapting language model to domain specific rag. In First Conference
on Language Modeling , 2024.
[68] Wenqi Zhang, Hang Zhang, Xin Li, Jiashuo Sun, Yongliang Shen, Weiming Lu, Deli Zhao,
Yueting Zhuang, and Lidong Bing. 2.5 years in class: A multimodal textbook for vision-language
pretraining. arXiv preprint arXiv:2501.00958 , 2025.
[69] Yuanhan Zhang, Jinming Wu, Wei Li, Bo Li, Zejun Ma, Ziwei Liu, and Chunyuan Li. Video
instruction tuning with synthetic data. arXiv preprint arXiv:2410.02713 , 2024.
[70] Zhixin Zhang, Yiyuan Zhang, Xiaohan Ding, and Xiangyu Yue. Vision search assistant: Em-
power vision-language models as multimodal search engines. arXiv preprint arXiv:2410.21220 ,
2024.
[71] Yaowei Zheng, Richong Zhang, Junhao Zhang, Yanhan Ye, Zheyan Luo, Zhangchi Feng, and
Yongqiang Ma. Llamafactory: Unified efficient fine-tuning of 100+ language models. arXiv
preprint arXiv:2403.13372 , 2024.
[72] Yuxiang Zheng, Dayuan Fu, Xiangkun Hu, Xiaojie Cai, Lyumanshan Ye, Pengrui Lu, and
Pengfei Liu. Deepresearcher: Scaling deep research via reinforcement learning in real-world
environments. arXiv preprint arXiv:2504.03160 , 2025.
14

A Related Work
A.1 Large Multimodal Models (LMMs)
The development of Large Multimodal Models (LMMs) marks a significant breakthrough in artificial
intelligence, enabling unified processing of textual and visual information that aligns with the
inherently multimodal nature of human perception. Recent advances have established a dominant
paradigm where scaling up diverse, high-quality vision-text paired datasets across training stages
equips LMMs with grounded visual understanding. This approach has yielded remarkable success,
as demonstrated by state-of-the-art models including GPT-4o [ 23], Gemini [ 57], Qwen2.5-VL [ 5],
the LLaV A series [ 39,32,69], and related systems [ 2,45,33,11,37], which exhibit exceptional
capabilities in visual comprehension, cross-modal reasoning, and instruction following. However, this
training methodology inherently yields static knowledge, while real-world information is complex,
dynamic, and constantly evolving. As a result, LMMs often struggle with long-tail concepts or
newly emerging facts beyond their training cut-off, which can lead to hallucinations and ultimately
compromise their stability and reliability in real-world applications. This limitation has led to
increasing efforts to augment LMMs with retrieval and search tool-use capabilities.
A.2 Large Models with External Knowledge Access
Given the strong reliance of large models on external knowledge in practical applications, two primary
approaches have emerged to enhance their factual reliability and knowledge coverage: retrieval-based
methods and search-based methods.
A.2.1 Retrieval-Augmented Generation (RAG)
The RAG paradigm retrieves relevant information from external knowledge bases via dense vector
search and incorporates it into the model input to help generate more factually grounded responses.
In the field of natural language understanding, Petr Karpukhin et al. [ 29] introduced Dense Passage
Retrieval (DPR), which was the first to apply dense retrieval to open-domain question answering. DPR
employs a dual-encoder architecture to separately encode queries and document passages, enabling
efficient semantic-level matching. Building on DPR’s strong retrieval capabilities, Lewis et al. [ 31]
proposed the RAG framework, which integrates pre-trained generative models with nonparametric
document indices, forming a unified ’retrieval-then-generation’ architecture that laid the foundation
for subsequent research. As model training and inference techniques have advanced, RAG-based
methods have gradually incorporated key techniques such as long-context modeling, chain-of-thought
reasoning, self-reflection and document ranking. These enhancements have improved performance
across pretraining [ 60], fine-tuning [ 24,67,12], and inference [ 54,3,65,61] stages, significantly
boosting RAG’s effectiveness in knowledge-intensive tasks.
The field of visual understanding has also seen notable progress through the adoption of retrieval-
augmented methods, particularly in knowledge-intensive VQA tasks. By incorporating external
knowledge, including text, images, and structured data, these approaches overcome the limitations
of vision-only models that often lack access to broader world knowledge. MuRAG [ 8] integrates
both image and text retrieval to enhance open-domain question answering over multimodal data.
REVEAL [ 22] utilizes a multi-source multimodal knowledge memory to augment visual-language
pre-training, enabling better handling of knowledge-intensive queries. RagVL [ 10] introduces a
knowledge-enhanced reranking mechanism, improving retrieval precision by filtering out noisy data.
VisRAG [ 64] employs a vision-based approach to process multi-modality documents, preserving
visual information without relying solely on text extraction. Collectively, these advances have
demonstrated the potential of retrieval-augmented generation to bridge visual understanding with
external knowledge.
Although RAG methods have shown strong performance in practice, they face several critical
limitations, the most notable being their reliance on a static knowledge base. These methods often
assume that the required information can be found within the existing corpus. However, in real-world
scenarios, information is frequently dynamic and constantly evolving, and the complexity of web-
scale environments means that relevant content may not always be effectively retrieved. This poses
significant challenges to the applicability of RAG-based approaches in more general and open-ended
settings.
15

A.2.2 Search-Augmented Approaches
Several recent efforts have explored search-augmented paradigms that go beyond static corpora by
enabling models to interact with real-time information sources. In the field of natural language
understanding, WebGPT [ 46] is one of the earliest systems to demonstrate this idea at scale. It
augments a language model with access to Bing search results and trains it using human feedback to
quote sources and generate more factual, citeable answers. This setup significantly improves factual
consistency, especially in domains requiring timely or less common knowledge. Toolformer [ 52]
introduces a self-supervised framework where the model learns when and how to invoke external
tools, such as search engines, by generating API call demonstrations and fine-tuning on helpful
examples. This enables efficient and context-aware tool use without heavy supervision. SAIL [ 42]
further integrates web search into instruction tuning. It constructs training examples by pairing user
prompts with retrieved search results, teaching the model to select relevant information, filter noise,
and perform multi-hop reasoning. While these approaches demonstrate strong potential for dynamic
and tool-augmented reasoning, they still rely heavily on high-quality annotated data for training. As
language models continue to improve in both knowledge retention and reasoning capabilities, a new
line of training-free, prompt-based methods has emerged. These approaches leverage large models as
agents orchestrated by structured prompts, without requiring additional fine-tuning. For example,
Recent studies [ 72,1] have introduced agentic workflows that incorporate web search tools directly
into the reasoning process. By decomposing complex queries into sub-tasks and selectively invoking
external tools through prompts, these systems can effectively tackle information-seeking tasks that
require up-to-date and reliable web content. Such approaches highlight the growing viability of
zero-shot and in-context tool use for knowledge-intensive applications.
Building on this trend, recent research has further extended tool-augmented, training-free paradigms
into the multimodal domain, exploring how large models can function as autonomous visual search
agents. For example, A VIS [ 21] proposes a framework in which a large language model acts as a con-
troller, iteratively issuing vision-language queries to retrieve external visual and textual information in
order to answer complex visual questions. VSA [ 70] takes this further by equipping vision-language
models with retrieval capabilities, enabling them to behave like multimodal search engines that can
proactively seek and integrate visual evidence from large-scale corpora. MMSearch [ 26] presents a
comprehensive pipeline that empowers large multimodal models with advanced search capabilities.
Its agentic workflow encompasses re-querying, reranking, and summarization stages, enabling models
to autonomously process and synthesize information from both visual and textual modalities. These
approaches signify a shift towards interactive, tool-augmented processes in multimodal reasoning.
However, a key limitation of these approaches is that the underlying models are typically not trained
to interact with search tools in a supervised or reinforcement-based manner. As a result, the agent may
not learn to engage with external tools in the most effective or reliable way, especially in real-world
environments with noisy or ambiguous search results.
A.3 Reinforcement Learning-powered Search Agents
Recent advances, such as OpenAI’s o series [ 25,48], DeepSeek-R1 [ 20] and Kimi-K1.5 [ 58],
have highlighted the potential of end-to-end reinforcement learning (RL) to enhance the reasoning
capabilities of large-scale models. These efforts have led to the emergence of Large Reasoning
Models (LRMs) that are capable of solving complex, multi-step reasoning tasks beyond the reach of
standard instruction-tuned language models. Building on this momentum, leading organizations such
as OpenAI, Google, and Perplexity have introduced Deep Research agents [ 47,14,19,49]. These
systems combine large reasoning models with real-time web search tools to complete open-ended,
research-oriented tasks, such as drafting reports or synthesizing diverse information sources. Notably,
OpenAI reports that it has successfully trained a highly capable Deep Research model through
end-to-end reinforcement learning, demonstrating the feasibility and effectiveness of this approach in
real-world applications. In the open-source community, efforts such as DeepResearcher [ 72], Search-
R1 [27], and ReSearch [ 7] have followed this direction, applying end-to-end RL to improve models’
abilities in multiturn search and retrieval-augmented generation, aiming to boost performance on
information-seeking question answering tasks. However, existing work mainly focuses on text-based
search, while in the multimodal domain, challenges such as constructing suitable training data and
designing RL frameworks to incentivize models to perform search in real-world environments remain
underexplored.
16

B Group Relative Policy Optimization (GRPO) Algorithm
GRPO updates the current policy model using a group of rollouts sampled from an old policy πold,
along with a reference πrefused as a constraint. Specifically, given a question qsampled from the
training set D, GRPO generates a set of outputs from πold, and optimizes the current policy πθby
maximizing the following objective:
JGRPO (θ) =E[q∼ D,{oi}G
i=1∼πθold(O|q)]
1
GGX
i=11
|oi||oi|X
t=1n
minh
Ri,tˆAi,tclip (Ri,t,1−ε,1 +ε)ˆAi,ti
−βDKL[πθ∥πref]o
,
(2)
where Ri,t=πθ(oi,t|q,oi,<t)
πθold(oi,t|q,oi,<t)is the policy ratio. ϵandβare hyper-parameters that control the
stability and constraint strength of the policy update, respectively. Given the sampled rollouts
o={o1, o2, ..., o G}, a group reward r={r1, r2,···, rG}can be obtained by a reward model. The
advantage ˆAi,t= ˜ri=ri−mean( r)
std(r)mitigates variance across samples and prevents any single reward
signal from dominating the policy update.
C Multimodal Search Tools
To enhance the stability and efficiency of multimodal search services during both training and infer-
ence, we encapsulate the image and text search tools as independent HTTP services. These services
are optimized through pipelined parallel processing, local caching, and multi-metric monitoring. The
overall architecture of the search pipeline is illustrated in Figure 6.
Image Search Tool The image search tool is built on top of SerpAPI. Given an input image URL,
SerpAPI returns a set of visually similar webpages, including metadata such as URLs, thumbnails,
and titles. We sort the returned results by relevance and extract up to five valid results (thumbnail +
title pairs). To avoid redundant SerpAPI calls during training and evaluation, we implement a cache
mapping image URLs to their corresponding search results.
Text Search Tool The text search pipeline consists of SerpAPI, JINA Reader, and a webpage
summarization model, forming a search–parse–summarize chain. SerpAPI first retrieves a list of
webpage URLs based on a given query. In each round, the top 5 URLs are processed using a pipelined
parallel strategy: JINA fetches and converts the webpage content into clean, structured text, while
Qwen3-32B acts as a summarizer, focusing on extracting information relevant to the original user
question using a tailored prompt. The full prompt is provided in Tabel 4. This approach improves
focus and reduces token usage. If some URLs fail to be processed, additional URLs from the SerpAPI
result list are used to retry, until 5 successful summarizations are obtained or the list is exhausted.To
improve efficiency and reduce redundant computation, the text search tool incorporates a multi-layer
caching mechanism: (1). Mapping from query to webpage URLs (to skip repeated searches); (2).
Mapping from URL to JINA parsing results (to avoid re-parsing); (3). Mapping from JINA outputs to
Qwen3-32B summaries (to reduce summarization cost). We deployed Qwen3-32B on a cluster of 8
×8 NVIDIA H100 GPUs to support our training and inference needs.
Cache Mechanism Caching is implemented using Redis to support high-throughput concurrent
read/write access. For latency-critical components, caching is applied to reduce response time.
Because JINA parsing results are often large, they are stored in object storage (TOS), while Redis
only stores references (keys) to these results, reducing memory footprint. We adopt an LRU (Least
Recently Used) policy for eviction under memory constraints. Additionally, a scheduled offline retry
mechanism is in place to reprocess failed JINA URLs and update the cache accordingly.
Distributed Rate Limiting infrastructure-level constraints on concurrent request handling can bottle-
neck training efficiency. Limited parallelism may cause request backlogs or latency spikes, ultimately
affecting throughput and stability. To manage system load during peak traffic, we implement dis-
tributed rate limiting using Redis. This allows us to smooth bursty traffic to JINA and webpage
summary model over time windows, reducing the risk of system overload and improving service
availability and stability.
17

image url
Thumbnail-1:
Title-1: 
Supermarine SpitfireThumbnail-2:
Title-2: 
PR MK II 
cached?Y
N...
Search Results
Text Query:
 
Who designed
the Supermarine 
Spitfire?Image Query:
Image
Search
Text
Search
related
URLs
Read
Webpage
Webpage
Summarize
Read
Webpage
Webpage
Summarize
Read
Webpage
Webpage
Summarize...scatter gather
1. (link: xxx) R.J. Mitchell 
is best known for 
designing the 
Supermarine Splitfire ...
2. (link: xxx)
...
Search ResultsFigure 6: The overall architecture of the multimodal search pipeline.
Table 2: Prompt used for FVQA-train VQA Generation.
Usage Prompt
Factual
QA
Gener-
ationYour task is to generate a factual question–answer pair based on the given visual
concept, the image, and the associated webpage content.
The generated question must strictly follow the requirements below:
1. The answer must contain the keyword visual concept.
2. The question must start with "Who", "What", or "Where".
3. The question must NOT include the visual concept itself, nor any background
knowledge directly related to it.
4. The question should resemble something a curious human without prior knowledge
about the image might ask.
In addition to the question, generate a concise and factual answer grounded in the
visual concept, image, and webpage content.
Visual Concept: {visual_concept}
Image: {image}
Webpage Content: {webpage_content}
Respond only with the generated question and answer.
D Prompts
D.1 Prompts for FVQA-auto-ac VQA Generation
To generate factual QA pairs for the FVQA-auto-vc set, we designed the prompt to guide GPT-4o in
producing concise and grounded question–answer pairs based on given image–webpage pairs. The
prompt were tailored to encourage questions that focus on visual concepts and can be answered using
observable or associated information in the webpage content. Details of the prompts are shown in
Table 2.
D.2 Prompts for RL Training
During training, prompts are used to guide and constrain the model’s behavior, ensuring that it
interacts with the search tools in a consistent and structured manner. Specifically, we design three
types of prompts for different points in the dialogue: one used at the beginning of the conversation,
one after the model performs an image search, and another after it performs a text search. These
context-specific prompts play a key role in aligning the model’s actions with the intended workflow
and ensuring proper use of the multimodal search tools. Details of the prompts are shown in Table 3.
18

Table 3: Prompts used during training at conversation start and after different search actions.
Usage Prompt
1st
RoundAnswer the user’s question based on the provided image. Examine the image carefully
and identify any recognizable entities, such as faces, objects, locations, events, logos,
or text. Determine whether you have sufficient knowledge to confidently recognize the
main visual element and answer the user’s question. If so, first explain your reasoning,
then provide a clear and direct answer.
If you are unable to confidently identify the visual element, stop and invoke the
image search tool by appending the string <search><img></search> at the end of your
response. This will trigger a Google Lens search using the original image to retrieve
relevant information that can help you confirm the visual content.
Once you have sufficient visual understanding, combine it with the user’s question and
assess whether you can confidently answer. If so, answer the question directly using
your own knowledge. If not, invoke the text search tool by generating a concise and
specific query, and output it in the format <text_search>your query here</text_search>
at the end of your response. Carefully craft your query to accurately retrieve the
information needed to help answer the question. The text search tool will then use
Google Search to return relevant information based on your query.
You must include your reasoning inside <reason>...</reason> before taking any action,
whether it is calling the image search tool, generating a text search query, or providing
a final answer. The reasoning may involve analysis of the original image and question,
interpretation of search results, or logical steps leading to the final answer.
All search results will be placed inside <information> and </information> and re-
turned to you. When you are ready to answer the question, wrap your final answer
between <answer> and </answer>, without detailed illustrations. For example: <an-
swer>Titanic</answer>.
Here is the image and the question: {image}{question}
After
image
searchOriginal question: {question}
Please extract the visual element relevant to the user’s question (such as faces, objects,
locations, events, logos, or text) from the search results. Then, combine this information
with the user’s question and perform reasoning to determine whether a Google text
search is needed to retrieve additional information to answer the question.
If a text search is needed, output the string <text_search>your query here</text_search>
at the end of your response. Please generate a well-crafted query based on the visual
element that will help retrieve the most relevant information.
If a text search is not needed, use your own knowledge to directly answer the user’s
question.
You must conduct your reasoning inside <reason> and </reason> before taking any
action, whether it’s generating a text search query or providing a final answer. If
you decide to give the final answer, place it inside <answer> and </answer>, without
detailed explanation or illustration. For example: <answer>Titanic</answer>
After
text
searchOriginal question: {question}
Please analyze the search results and the user’s question and continue reasoning inside
<reason> and </reason>.
If you determine that additional knowledge is still required to answer the user’s question,
stop responding to the question and instead report a warning by outputting the string
"Unable to answer due to lack of relevant information" at the end of your response.
If no further external information is needed, you should provide the final answer by
placing it within <answer> and </answer>. The answer must be concise, clear, and to
the point, without any additional explanation or elaboration.
D.3 Prompts for Webpage Summarization Model
The prompt shown in Table 4 is used to guide the webpage summarization model in the text search
pipeline to effectively extract and summarize webpage content based on the given user question. The
19

Table 4: Prompt used for Webpage Summarization Model.
Usage Prompt
System
Mes-
sageYou are a helpful assistant. Your task is to summarize the main content of the given
web page in no more than five sentences. Your summary should cover the overall key
points of the page, not just parts related to the user’s question.
Prompt If any part of the content is helpful for answering the user’s question, be sure to include
it clearly in the summary. Do not ignore relevant information, but also make sure the
general structure and main ideas of the page are preserved. Your summary should be
concise, factual, and informative.
Webpage Content (first 30000 characters) is: {webpage_content}
Question: {question}
Table 5: Prompt used for Direct Answer baselines.
Usage Prompt
1st-
RoundBased on the image, answer the question with as few words as you can.
Question: {question} Image: image
goal is to ensure that the returned content is more focused and concise, thereby reducing overall token
consumption.
D.4 Prompt for Direct Answer Baseline
Table 5 presents the prompt template used in the Direct Answer baseline. In this setting, the model is
expected to provide a concise answer to a given question based solely on the visual content of the
image.
D.5 Prompts for RAG Workflow Baseline
Under the RAG workflow, models are required to perform exactly two search operations for each
VQA example using our multimodal search tools: first an image search, followed by a text search.
Under the RAG workflow, models are required to perform exactly two search operations for each
VQA example using our multimodal search tools: first an image search, followed by a text search.
This results in a maximum of three conversation rounds. Specifically, given an input image and
question, the model is presented with the image search results and the original question in the first
round and is prompted to generate a relevant text query to aid in answering. In the second round, the
retrieved text results based on that query are provided to the model, which is then asked to generate
the final answer. In our experimental setup, the RAG workflow reaches the maximum allowed number
of search steps, making it a strong baseline that provides the model with the most external information.
Details of the prompts of RAG Workflow are shown in Table 6.
D.6 Prompts for LLM-as-Judge Evaluation
In our experiments, we adopt the LLM-as-judge approach to evaluate whether a model’s response
aligns with the ground truth answer. This method provides a more objective and generalizable way to
assess performance on open-ended VQA tasks. Table 7 shows the full prompt used with GPT-4o,
which serves as the judge LLM throughout all our experiments. Once GPT-4o determines whether
each response is correct or incorrect, its judgments are used to compute the accuracy metric.
E Details of Benchmark Datasets
FVQA-test As described in Section 3, the FVQA-test set is a manually curated test set consisting of
1800 examples collected from three sources: (1) 600 samples drawn from FVQA-auto-vc , with no
overlap with the training set; each example was manually verified for correctness; (2) 600 samples
selected from the InfoSeek Human Split, where we manually re-annotated the answers, as the original
20

Table 6: Prompts used for RAG workflow baselines. Text in red indicates content returned by the
multimodal search tool.
Usage Prompt
1st
RoundYou are given a question accompanied by an image that cannot be answered without
external knowledge. To assist your understanding, you are also provided with image
search results consisting of five web pages related to the original image, ranked by
relevance. Each result includes the main image from the web page and its title.
Question: {question} Image: {image}
Image Search Results:
1. Webpage Image: {image} Webpage Title: {title}
2. Webpage Image: {image} Webpage Title: {title}
...
5. Webpage Image: {image} Webpage Title: {title}
Assume you have access to a search engine (e.g., google). Based on the question,
image and image search results, please raise a text query to the search engine to search
for what is useful for you to answer the question correctly. You need to consider the
characteristics of asking questions to search engines when formulating your questions.
Now give the text query to search engine directly (do not wrap it in quotes or add any
explanation):
2nd
RoundYou should now read the text search result and answer the answer the question based
on the image provided.
Text Search Results:
...
Original question: {question}
Answer the question with as few words as you can.
human-labeled answers were not publicly released; (3) 600 samples newly collected by human
annotators. This dataset covers a wide range of visual and textual knowledge categories, enabling a
comprehensive evaluation of model performance.
InfoSeek InfoSeek [ 9] is constructed through a semi-automated process that transforms Wikidata
triples into natural language questions using human-authored templates. Annotators design question
templates for 300 Wikidata relations, incorporating placeholders for visual entity types and units to
ensure clarity. These questions are then paired with corresponding images and answers to form image,
question, answer triplets. To ensure diversity and answerability, question-answer pairs that lack
supporting evidence in Wikipedia are filtered out, and balanced sampling is applied across entities
and relations. This design makes InfoSeek particularly suitable for evaluating information-seeking
capabilities, as it emphasizes diverse, fact-based queries grounded in real-world knowledge and
multimodal contexts. We randomly sampled 2000 examples from its test split due to the large dataset
size.
MMSearch MMSearch [ 26] contains 300 manually collected examples spanning 14 subdomains,
categorized into two types: News, which includes up-to-date events from August 2024 to ensure no
overlap with LMM training data, and Knowledge, which focuses on rare, verified knowledge that
current state-of-the-art LMMs (e.g., GPT-4o, Claude 3.5) fail to answer. Among these, 171 examples
are visual questions (i.e., questions paired with images), while the remaining are purely textual.
We use the 171 visual questions as our evaluation subset. This dataset is particularly well-suited
for evaluating models’ real-world information-seeking abilities, especially in scenarios that require
up-to-date retrieval or reasoning over rare knowledge.
SimpleVQA SimpleVQA [ 13] is built from two main sources of seed examples. First, factual
image-question pairs are selected from existing VQA datasets that meet real-world knowledge
standards. These datasets were constructed after 2023 and focus on practical, application-driven
content. Secondly, additional data is sourced via internet search, with expert annotators generating QA
pairs based on retrieved images and factual content. These examples span a wide range of entities and
events, with answers focused on objective, fact-based information involving entity recognition and
attribute extraction. Building on these seed samples, the dataset further undergoes reliable difficulty
21

Table 7: Full prompt used for GPT-4o as the judge LLM in all experiments.
Usage Prompt
System
Mes-
sageYou are an AI assistant tasked with evaluating the correctness of model responses based
on an image, question, and ground truth answer. Your judgment should follow these
principles:
1. Consider the image, question, and ground truth answer holistically before evaluating
the model’s response.
2. Your decision should be strictly Yes or No , based on whether the model’s response
is factually accurate and aligns with the ground truth answer.
3. If the model response is a more specific form of the ground truth answer, it is correct.
4. If the model response includes all key information but adds minor details, it is correct
as long as the extra details are factually correct.
5. If the model response contradicts, modifies, or omits critical parts of the answer, it is
incorrect.
6. For numerical values, ensure correctness even when presented in different units.
7. For names, check for first and last name correctness. If the middle name is extra but
correct, consider it correct.
8. For yes/no questions, the response must exactly match "Yes" or "No" to be correct.
9. If the judgment can be made based solely on the text, you may choose to ignore the
input image, as some images may be unfamiliar to you and could affect your judgment.
Refer to the image only when necessary to minimize misjudgment.
10. If there are multiple candidate answers, you can also evaluate the model’s response
against all of them. If the response aligns with at least one candidate according to the
rules above, it should be considered correct.
Your output must be in the following format:
<judge>Yes/No</judge>
<reason>Explanation of why the answer is correct or incorrect.</reason>
Prompt Image, Question, and Model Response Evaluation
Question: {question}
Ground Truth Answer: {ground truth answer}
Candidate Answers: {candidate answers}
Model Response: {model response}
Evaluation Instructions
Evaluate whether the Model Response is correct based on the Image, Question, Ground
Truth Answer and Candidate Answers. Follow the predefined judgment rules and
provide a clear Yes/No answer along with a justification.
Output Format
<judge>Yes/No</judge>
<reason>Detailed reasoning following the evaluation principles.</reason>
filtering and human-in-the-loop quality control, making it a strong candidate for evaluation of models’
factual and knowledge-based reasoning abilities. The final SimpleVQA benchmark contains 2,025
examples. To avoid language-related confounding factors, we select the 1,013 English QA pairs for
evaluating model performance.
LiveVQA LiveVQA [18] is built from six globally recognized news platforms, such as CNN, BBC,
Yahoo, Forbes, AP News, and Variety, to ensure diversity and timeliness across visual and textual
content. The dataset covers 14 major news categories, including sports, entertainment, science,
economy, and health, and contains 3,602 VQA pairs. For each article, question–answer pairs are
generated using GPT-4o, including both basic visual questions and more complex multi-hop questions
that require reasoning over the accompanying text. This setup supports evaluation of real-world
information-seeking and multimodal reasoning capabilities.
22

Table 8: Performance Comparison on General VQA Benchmarks
ModelAI2D ChartQA LLaV A-Wilder MathVista MME OCRBench
test test small testmini test -
Qwen2.5-VL-7B 93.0 86.6 73.5 68.2 623/1705 84.7
MMSearch-R1-7B 92.8 86.1 74.8 68.8 622/1707 84.5
F Details of Experimental Implementation
F.1 RL Training Setting
We adopt the Qwen2.5-VL-7B model as the backbone model, and the overall GRPO training is
implemented based on the veRL [ 56] framework. The training set is the FVQA-train dataset, which
consists of approximately 1,600 search-free samples and 3,400 search-required samples. Training
is conducted on 4 nodes with 8 Nvidia H100 GPUs each. The total batch size is set to 512, with a
mini-batch size of 128. For each training prompt, we generate 8 rollouts, and each rollout allows up
to 3 tool calls. Regarding the reward components, the search penalty and the weights for the format
score are both set to 0.1. The learning rate is 2e-6, and we fix the KL divergence coefficient βto
0.001 and the clip ratio ϵto 0.2. For MMSearch-R1-7B, we use the checkpoint from step 50 (when
training converges) for downstream evaluation.
F.2 SFT Training Setting
For the SFT model used in the RL vs. SFT performance comparison experiments presented earlier.
We fine-tune the Qwen2.5-VL-7B model using LLaMA-Factory [ 71], with a dataset of 8,000 samples
generated by prompting GPT-4o to perform on-demand search. Each sample contains up to three
rounds of dialogue. The GPT-4o responses are used as supervision signals, while the user question
in the first turn and tool-returned content in later turns are masked out during loss computation.
Fine-tuning is performed on a single machine with 8 Nvidia H100 GPUs. We use a per-device batch
size of 1 and apply gradient accumulation over 2 steps. The model is trained for 1 epochs with a
learning rate of 1e-5. A cosine learning rate scheduler is adopted, with 10% of the total training steps
used for warm-up.
F.3 Evaluation Setting
We conduct model inference using the veRL framework by setting the val_only parameter of the
trainer to True. The inference engine is based on vLLM, with top_pset to 1.0 and temperature set to
0. A single response is generated for each sample in the test set.
For evaluation, we adopt an LLM-as-Judge approach. Specifically, we use GPT-4o-20241120 as the
judge model and apply a predefined prompt template (as shown in Table 7) to assess each inference
result. During judgment, top_pis set to 0.1 and temperature is set to 0. The model’s binary output
("Yes" or "No") is used to compute the final accuracy metric.
G Full Experimental Results
G.1 Evaluation on General VQA Benchmarks
To assess whether RL training affects general VQA capabilities, we compare MMSearch-R1-7B
with its backbone model Qwen2.5-VL-7B on a suite of standard VQA benchmarks, including
AI2D [ 30], ChartQA [ 44], LLaV A-Wilder [ 36], MathVista [ 41], MME [ 17], and OCRBench [ 40].
All experiments are conducted using LMMS-Eval [ 66]. As shown in Table 8, MMSearch-R1-7B
achieves comparable performance across all benchmarks, slightly outperforming the base model on
LLaV A-Wilder and MathVista while maintaining similar results on AI2D, ChartQA, OCRBench and
MME. These results suggest that our reinforcement learning process, while enhancing search-related
behavior, preserves the model’s general visual understanding and reasoning ability.
23

Table 9: Ablation Study on Reward Modeling. "Acc (%)" denotes the accuracy evaluated by LLM-as-
Judge, while "SR (%)" represents the search ratio, defined as the percentage of total search calls made
relative to the maximum allowed search steps for each method.∗EM indicates that the accuracy
score used in the reward function is computed via exact string match (EM), while∗4odenotes that
correctness is judged by GPT-4o.
ModelAverageIn-Domain Out-of-Domain
FVQA-test InfoSeek MMSearch SimpleVQA LiveVQA
Acc SR Acc SR Acc SR Acc SR Acc SR Acc SR
RAG Workflow
GPT4o 62.1 100 66.0 100 59.1 100 62.5 100 63.4 100 59.6 100
Gemini 2.5 Pro 61.8 100 66.1 100 56.7 100 62.5 100 65.9 100 57.8 100
Qwen2.5-VL-72B 59.6 100 62.2 100 59.4 100 59.6 100 61.0 100 56.0 100
Qwen2.5-VL-32B 55.1 100 57.0 100 56.8 100 57.9 100 54.5 100 49.6 100
Qwen2.5-VL-7B 51.6 100 52.9 100 53.7 100 52.2 100 51.6 100 48.0 100
On-demand Search
MMSearch-R1-7B
∗EM55.7 77.3 59.1 83.4 57.1 68.3 55.6 91.8 58.1 56.8 48.5 86.2
MMSearch-R1-7B
∗4o59.5 82.6 62.1 88.9 59.1 81.8 61.9 92.7 59.8 72.6 54.7 76.9
G.2 Ablation Study on Reward Modeling
Our main training framework adopts a rule-based reward design based on exact string match, which
is simple, deterministic, and scalable. However, this formulation may not fully capture the semantic
correctness of answers, especially when multiple surface forms express the same factual content. To
explore whether a more flexible and context-aware reward signal can better guide model learning, we
conduct an additional experiment using GPT-4o as the reward model during training. GPT-4o provides
semantic-level accuracy judgments, allowing us to assess the impact of richer reward feedback on
model behavior and final performance. The prompt used to elicit reward signals from GPT-4o is the
same as the one used in our evaluation setup, as shown in Table 7.
As shown in Table 9, training with GPT-4o reward leads to a clear improvement across all datasets.
The∗4oversion achieves an average accuracy of 59.5%, which is 3.8 points higher than the∗EM
version. These results indicate that more general reward supervision holds greater potential for
open-ended tasks. GPT-4o helps avoid false negatives caused by exact match and leads to improved
robustness. However, it may introduce bias, higher training cost and struggles with questions beyond
its knowledge scope. We believe that exploring more general reward modeling represents a promising
direction for future research. Notably, this experiment is conducted on a subset of the FVQA-train
dataset, in which search-free examples are relatively underrepresented. As a result, the model exhibits
a higher average search rate at convergence compared to Table 1.
H Limitations
While MMSearch-R1 demonstrates strong performance and introduces a practical framework for
on-demand multimodal search, several limitations remain.
First, the interaction between the model and external multimodal search tools still has room for
improvement in terms of stability and quality. For example, image search currently requires submitting
the full image, which may not be optimal for retrieving localized visual content. The text search
pipeline is composed of several components, including SerpAPI, Jina Reader, and a summarization
model. Each component may introduce potential sources of variability. Specifically, the ranking
of result returned by SerpAPI may vary over time; Jina Reader may fail to extract content from
certain domains due to restrictions or errors; and the summarizer may generate hallucinations when
extracting relevant information. Despite iterative refinement of the pipeline, our monitoring during
training revealed an end-to-end failure rate of approximately 0.2% for image search and 1% for
text search, where failure is defined as receiving no valid results. These numbers are higher when
24

considering partial failures, such as retrieving fewer than the expected top five results. As training
scales up, ensuring stability and consistency in tool outputs remains a nontrivial challenge.
Second, our reward design based on exact string match, while simple and scalable, has limited
flexibility. This design is well-suited for short factual questions with unambiguous answers. However,
it may penalize answers that are semantically correct but differ slightly in phrasing. This limitation
affects the generalization of the reward function to more complex or open-ended QA tasks. Although
our dataset focuses on fact-based QA that aligns well with this evaluation strategy, more flexible
reward signals could help expand the framework to broader question types and reduce reliance on
surface-form matching. To explore this, we conducted a preliminary experiment comparing EM-based
rewards with GPT-4o-based rewards, which showed that the latter offers better tolerance to answer
variation and may support more nuanced evaluation (see Appendix G.2 for details).
In general, improving the robustness of tool interactions and enhancing the expressiveness of the
reward function will be essential for scaling MMSearch-R1 to more adaptive and reliable multimodal
reasoning agents.
I Broader Impacts
Our work focuses on improving the ability of multimodal models to access and reason over external
information through search. Although this capability has clear benefits for building more informative
and adaptive assistants, it also introduces potential risks. For example, models that autonomously
retrieve and summarize web content may surface outdated, biased, or misleading information. Addi-
tionally, over-reliance on real-time search may raise concerns around content verifiability, copyright,
or inadvertent propagation of misinformation.
To mitigate these potential risks, we recommend careful filtering of the retrieved content, attribution of
information sources, and incorporating mechanisms that allow users to trace and verify the provenance
of model-generated responses. We also encourage future research on safer retrieval strategies, such
as domain whitelisting and adaptive uncertainty calibration, especially in high-stakes applications.
J Examples of FVQA Dataset
To better illustrate the characteristics of our dataset, we present representative examples from the FVQA
dataset in Figure 7. QA pairs are designed to be knowledge-intensive and fact-oriented, covering
a broad range of visual and textual knowledge types. We also present examples from the data
construction process of FVQA-auto-vc in Figure 8.
K Case Studies
As shown in Figure 9 and Figure 10, we present several representative case studies to illustrate how
MMSearch-R1 performs on complex, real-world information-seeking VQA tasks. These examples
highlight the model’s ability to reason about when and how to invoke search tools, generate effective
queries, and synthesize the retrieved information to arrive at accurate answers. For ease of formatting
and readability, we omit some of the retrieved search results in certain cases.
25

Level-1 Knowledge Taxonomy: Arts Level-1 Knowledge Taxonomy: Place
Level-1 Knowledge Taxonomy: People Level-1 Knowledge Taxonomy: Industry
Level-1 Knowledge Taxonomy: Science Level-1 Knowledge Taxonomy: Nature
Level-1 Knowledge Taxonomy: Entities Level-1 Knowledge Taxonomy: Events
Q: What event is taking place at 
this race track?
A: Dutch Grand Prix.Q: What is the name of this plant 
with yellow flowers? 
A: Bladderwort.
Q: What is this colorful 
astronomical object called?
A: Crab nebula.
Q: What event does this map 
represent? 
A: 2020 Polish presidential 
election.
Q: What type of art style is depicted 
in this image? 
A: Fractal art.
Q: What type of ammunition is 
shown in the image?
A: .338 Federal.
Q: What type of fish is shown in 
the image?
A: Lizardfish.
Q: Who designed this building?
A: Helgo Zettervall.
Q: Who is the central figure 
addressing the group in this scene?
A: Dronacharya.
Q: What country does this place 
belong to?
A: Vietnam.
Q: What is the scientific term for 
the organisms?
A: Haemoproteus.
Q: Where is the lake outflow to?
A: Mincio.
Q: What is the name of the company 
associated with this logo? 
A: Brainfeeder.
Q: What is the brand of this vehicle? 
A: Land Rover.Q: What painting technique is used 
in this artwork?
A: Grisaille.
Q: What is this logo for?
A: Google Duo
Figure 7: Examples of FVQA dataset.
sparcstationQ: What type of computer is shown 
in the image?
A: SPARCstation.
GPT4o
QA GenVisual
ConceptImage Webpage
RuPaul's Drag 
Race All Stars 
(season 4)QA
Q: What show are these performers from?
A: RuPaul's Drag Race All Stars (season 4).
Figure 8: Examples of FVQA-auto-vc QA Generation.
26

Source: MMSearch
Question: In which day of 2024 was the lunar rover project 
in the picture announced to be canceled?
Answer: 07-17.
GPT4o-20241120 Qwen2.5-VL-72B
MMSearch-R1-7BJuly 1, 2024. Not announced.Direct Answer
On-Demand Search
<reason>The image shows a lunar rover, but there is no information about its announcement or 
cancellation date. The image does not provide enough context to answer the user's question 
confidently.</reason>
<search><img></search>
Image:
Image:
Image:Title: VIPER
Title: Virtual Intelligent Planetary Exploration Rover, VIPER Mobili... 
- NASA Science
Title: Scientists slam 'indefensible' axing of Nasa's $450m Viper moon 
rover | Nasa | The Guardian
<reason>The image search results mention the "Viper" lunar rover project and a news article about 
the cancellation of the project. However, the image does not provide information about the exact 
date when the project was canceled. To answer the user\'s question, a text search is necessary to find 
the specific date of the cancellation.</reason>
<text_search>When was the NASA Viper moon rover project canceled in 2024?</text_search>
1. (link:https://www.nasa.gov/news-release/nasa-ends-viper-project-continues-moon-exploration/) 
NASA announced the cancellation of the VIPER (Volatiles Investigating Polar Exploration Rover) 
project on July 17, 2024. The decision followed a comprehensive internal review ...
2. (link:https://www.nasa.gov/blogs/missions/2025/05/07/nasa-to-explore-additional-methods-to-
send-viper-to-moon/) The NASA VIPER (Volatiles Investigating Polar Exploration Rover) moon 
rover project was canceled in July 2024. Following this cancellation ...
3. (link:https://spacenews.com/nasa-backtracks-on-viper-commercial-partnership/) NASA canceled 
the VIPER lunar rover mission in July 2024. The decision was made due to ...
   ....
<reason>The search results indicate that the NASA VIPER lunar rover project was canceled on 
July 17, 2024. The information is consistent across multiple sources, suggesting that this is the 
correct date for the cancellation.</reason>
<answer>July 17</answer>Figure 9: Case study 1 of MMSearch-R1.
27

Source: FVQA-test
Question: What historical event is depicted in this 
battle scene?
Answer: Battle of Flodden.
GPT4o-20241120 Qwen2.5-VL-72B
MMSearch-R1-7BBattle of Agincourt. Battle of Agincourt.Direct Answer
On-Demand Search
<reason>The image depicts a battle scene with armored soldiers engaged in combat. The style of 
armor and the type of weaponry suggest it might be from a specific historical period. However, 
without additional context or identifiable markers, it\'s challenging to determine the exact historical 
event. The image does not provide enough specific details to identify the event confidently. </reason>
<search><img></search>
Image:
Image:
Image:Title: Butchered at the Battle of Flodden - Warfare History Network
Title: IVÁN DUQUE - Stock Photos, Illustrations and Images - Album
Title: Reconstruction painting showing a clash between English and Scottish 
pikemen during the Battle of Flodden Field on the 9th of September 1513 
(IC037/008) Archive Item - English Heritage(EH):Archive Collection | 
Historic England
<reason>The image search results suggest that the battle scene in the image is related to the 
"Battle of Flodden." The visual element from the search results is the title "Battle of Flodden." 
This indicates that the historical event depicted in the battle scene is likely the "Battle of 
Flodden," which took place on September 9, 1513, during the Scottish Wars of 
Independence.</reason>
<answer>Battle of Flodden.</answer>
Image:Title: What was the composition of European medieval armies 
(military, middle ages, and history)? - QuoraFigure 10: Case study 2 of MMSearch-R1.
28