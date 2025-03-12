# ORANSight-2.0: Foundational LLMs for O-RAN

**Authors**: Pranshav Gajjar, Vijay K. Shah

**Published**: 2025-03-07 07:44:31

**PDF URL**: [http://arxiv.org/pdf/2503.05200v1](http://arxiv.org/pdf/2503.05200v1)

## Abstract
Despite the transformative impact of Large Language Models (LLMs) across
critical domains such as healthcare, customer service, and business marketing,
their integration into Open Radio Access Networks (O-RAN) remains limited. This
gap is primarily due to the absence of domain-specific foundational models,
with existing solutions often relying on general-purpose LLMs that fail to
address the unique challenges and technical intricacies of O-RAN. To bridge
this gap, we introduce ORANSight-2.0 (O-RAN Insights), a pioneering initiative
aimed at developing specialized foundational LLMs tailored for O-RAN. Built on
18 LLMs spanning five open-source LLM frameworks, ORANSight-2.0 fine-tunes
models ranging from 1 to 70B parameters, significantly reducing reliance on
proprietary, closed-source models while enhancing performance for O-RAN. At the
core of ORANSight-2.0 is RANSTRUCT, a novel Retrieval-Augmented Generation
(RAG) based instruction-tuning framework that employs two LLM agents to create
high-quality instruction-tuning datasets. The generated dataset is then used to
fine-tune the 18 pre-trained open-source LLMs via QLoRA. To evaluate
ORANSight-2.0, we introduce srsRANBench, a novel benchmark designed for code
generation and codebase understanding in the context of srsRAN, a widely used
5G O-RAN stack. We also leverage ORANBench13K, an existing benchmark for
assessing O-RAN-specific knowledge. Our comprehensive evaluations demonstrate
that ORANSight-2.0 models outperform general-purpose and closed-source models,
such as ChatGPT-4o and Gemini, by 5.421% on ORANBench and 18.465% on
srsRANBench, achieving superior performance while maintaining lower
computational and energy costs. We also experiment with RAG-augmented variants
of ORANSight-2.0 LLMs and thoroughly evaluate their energy characteristics,
demonstrating costs for training, standard inference, and RAG-augmented
inference.

## Full Text


<!-- PDF content starts -->

ORANSight -2.0: Foundational  LLMs  for O-RAN  
Pranshav  Gajjar1 and Vijay  K. Shah1,2 
1NextG  Wireless  Lab, North  Carolina  State  University,  and 2WiSights  Lab 
{prgajjar  and vijay.shah }@ncsu.edu,  vijay.shah@wisightslab.com  
 
Abstract—Despite the transformative impact of Large Lan - 
guage  Models  (LLMs)  across  critical  domains  such  as healthcare, 
customer service, and business marketing, their integration into 
Open  Radio  Access  Networks  (O-RAN)  remains  limited.  This  gap 
is primarily due to the absence of domain -specific foundational 
models, with existing solutions often relying on general -purpose 
LLMs that fail to address the  unique challenges and technical in - 
tricacies  of O-RAN.  To bridge  this gap, we introduce  ORANSight - 
2.0 (O-RAN  Insights),  a pioneering  initiative  aimed  at developing 
specialized foundational LLMs tailored for O -RAN. Built on 18 
models spanning five open -source LLM frameworks —Mistral, 
Qwen,  Llama,  Phi, and Gemma—ORANSight -2.0 fine-tunes  mod - 
els ranging from 1B to 70B parameters, significantly reducing 
reliance on proprietary, closed -source models while enhancing 
performance  in O-RAN -specific  tasks.  At the core of ORANSight - 
2.0 is RANSTRUCT, a novel Retrieval -Augmented Generation 
(RAG) -based instruction -tuning framework that employs two 
LLM agents —a Mistral -based Question Generator and a Qwen - 
based Answer Generator —to create high -quality instruction - 
tuning datasets. The generated dataset is then used to fine -tune 
the 18 pre -trained open -source LLMs via QLoRA. To evaluate 
ORANSight -2.0, we introduce  srsRANBench,  a novel  benchmark 
designed  for code  generation  and codebase  understanding  in 
the context of srsRAN, a widely -used 5G O -RAN stack. Addi - 
tionally, we leverage ORAN -Bench -13K, an existing benchmark 
for assessing O -RAN -specific knowledge. Our comprehensive 
evaluations  demonstrate  that ORANSight -2.0 models  outperform 
general -purpose  and closed -source  models,  such  as ChatGPT - 
4o and Gemini, by 5.421% on ORANBench and 18.465% on 
srsRANBench, achieving superior performance while maintain - 
ing lower computational and energy costs. We also experiment 
with RAG -augmented variants of ORANSight -2.0 models and 
observe that RAG augmentation improves performance by an 
average of 6.35% across benchmarks, achieving the best overall 
cumulative  score  of 0.854,  which  is 12.37%  better  than  the leading 
closed -source alternative. Additionally, we thoroughly evaluate 
the energy  characteristics  of ORANSight -2.0, demonstrating  
its efficiency in training, inference, and inference with RAG 
augmentation, ensuring optimal performance while maintaining 
low computational and energy costs.  
Index Terms —O-RAN, ORANSight, 5G, LLM, ORANBench, 
srsRANBench, QLoRA, Foundational Models  
 
I. INTRODUCTION  
Open Radio Access Networks (O -RAN) have emerged as a 
transformative paradigm in the telecommunications industry, 
enabling greater flexibility, interoperability, and innovation  
by disaggregating hardware and software components [1]. 
However, the complexity of O -RAN architectures, coupled 
with the need for real -time automation, optimization, and 
troubleshooting,  poses  significant  challenges.  Large  Language 
Models (LLMs) have shown immense potential in address - 
ing these  challenges  by enabling  intelligent  decision -making,  code  generation,  and natural  language  understanding  [2] [3]. 
Despite  this potential,  the integration  of LLMs  into O-RAN 
remains  nascent,  primarily  due to the lack of foundational 
LLMs tailored to the unique requirements of O -RAN systems. 
Related  Works.  Recent  advancements  in the telecom  in- 
dustry  have  seen growing  adoption  of LLMs  [4]. They 
have  been  utilized  for network  modeling  [5] and to obtain 
network  configurations  [6], for prediction  tasks  like traffic 
load prediction  [7] and beamforming  [8]. To analyze  large - 
scale document corpora [9], and also as chatbot interfaces to 
facilitate  development  and help as an engineering  assistant 
[10].  These  works  show  the latent  potential  of such LLMs 
and how they can be incorporated  into telecom,  and we can 
assume  the implications  that these  approaches  have  for O- 
RAN.  A lot of the aforementioned  work  leverages  prompt - 
tuning [5] or Retrieval -Augmented Generation [11] pipelines 
to instill  domain  knowledge  in an LLM.  There  has also 
been  work  towards  dataset  curation  and model  training  like 
Nikbakht  et al. [12] that introduced  TSpec -LLM,  an open - 
source  dataset  aimed  at enhancing  LLM  understanding  of 
3GPP  specifications.  Similarly,  Maatouk  et al. [9] proposed 
TeleLLMs,  a series  of specialized  language  models  trained 
on telecommunications data, specifically 3GPP specifications, 
demonstrating  improvements  over general -purpose  LLMs.  Zou 
et al. [13] proposed  TelecomGPT,  a framework  for building 
telecom -specific  LLMs  with tailored  benchmarks  and datasets. 
However,  while  these  works  contribute  to LLM  advancements 
in telecommunications,  they do not include  the O-RAN  spec- 
ifications  or address  any O-RAN -related  challenges.  
Recent  efforts  have  been  made  to integrate  LLMs  into Open 
Radio  Access  Networks  O-RAN  and to bridge  the domain  gap 
between  the pretraining  of base LLMs  and the specific  require - 
ments of O -RAN. In our previous work [11],  we introduced 
ORAN -Bench -13K, a comprehensive benchmark designed to 
evaluate the performance of LLMs in O -RAN -specific tasks. 
This benchmark highlighted the limitations of existing models 
in handling O -RAN -related challenges. Additionally, we pro - 
posed a preliminary version of ORANSight, which addressed 
domain -based limitations by leveraging Retrieval -Augmented 
Generation  (RAG)  [11].  Similarly,  Karbalaee  Motalleb  et al. 
[14] explored  the use of LLMs  to enhance  the security  of 
O-RAN architectures, emphasizing the necessity of domain - 
specific fine -tuning to mitigate vulnerabilities and threats. 
Furthermore, Lotfi et al. [15] demonstrated the effectiveness  
of LLM -augmented Deep Reinforcement Learning (DRL) for 
dynamic O -RAN network slicing, showcasing the potential of 
LLMs  in improving  decision -making  processes  within  com- 

plex and evolving environments. Despite these advancements, 
the existing  approaches  suffer  from  two significant  drawbacks. 
First,  none  of these  works  have  incorporated  direct  fine-tuning 
on O -RAN -specific text or addressed code generation for O - 
RAN in any capacity. Second, these approaches predomi - 
nantly  rely on closed -source  proprietary  models,  which  present 
several challenges. These models are plagued by privacy 
concerns, a lack of O -RAN knowledge, and the requirement 
for prompt and data sharing with the parent organization [9]. 
Additionally, they are entirely black -box models, offering no 
access to their training information, and cannot be deployed 
locally [16].  These limitations underscore the need for more 
transparent, domain -specific, and locally deployable solutions 
in the integration of LLMs within O -RAN.  
Contributions.  From  the above  discussion,  it is evident 
that the integration of LLMs into O -RAN necessitates the 
development of domain -specific fine -tuning to address O - 
RAN’s unique challenges, such as real -time network slicing, 
resource allocation, and fault diagnosis. While LLMs have 
demonstrated  remarkable  capabilities  in general -purpose  tasks, 
their application in O -RAN -specific scenarios is constrained 
by the lack of open foundational models trained on O -RAN - 
specific data. Furthermore, scaling LLMs for O -RAN applica - 
tions presents additional challenges, including computational 
resource requirements, energy efficiency, and the need for 
high-quality, domain -specific datasets.  
In this work,  we make  the following  contributions.  
• We present ORANSight -2.0 1, the first comprehensive 
effort to develop foundational LLMs tailored for O -RAN. 
Specifically,  using  the QLoRA  (Quantized  Low-Rank  Adapta - 
tion)  method  [17],  a parameter -efficient  fine-tuning  technique, 
we fine -tune 18 LLM models from five leading open -source 
LLM model families, namely, Mistral, Qwen, Llama, Phi and 
Gemma  - ranging  from  1B to 70B parameters.  ORANSight -2.0 
addresses  the limitations  of closed -source  models  and provides 
open -source alternatives optimized for O -RAN tasks. We not 
only work towards foundational models that are proficient for 
the specifications, but we also emphasize the code generation 
and understanding capabilities.  
• We introduce RANSTRUCT , a Retrieval -Augmented 
Generation [11] (RAG) -based instruction -tuning framework 
that leverages  two LLM  agents  to generate  high-quality,  O- 
RAN -specific instruction -tuning  datasets.  
• Finally, we present srsRANBench , a novel benchmark 
designed for code generation and codebase understanding for 
the srsRAN  project  [18],  an open -source,  widely  used 5G O- 
RAN stack not only used by the academia for R&D purposes 
but also by industry for 5G O -RAN network deployments 
across US and Europe. srsRANBench enables the evaluation 
of various LLMs in 5G O -RAN -specific coding tasks.  
Experiment. We extensively evaluate the performance of 
ORANSight -2.0 against  state-of-the-art closed -source  models, 
including  ChatGPT -4o [19],  and Gemini  1.5 [20].  We also 
 
1In the rest of the paper, we will use ORANSight -2.0 and ORANSight  
interchangeably to refer to the proposed foundational LLMs for O -RAN.  evaluate the efficacy of RAG -based ORANSight augmenta - 
tion and the energy characteristics associated with deploy - 
ing ORANSight or for further fine -tuning. Furthermore, all 
ORANSight -2.0 models 2 and dataset benchmarks 3 are pub - 
licly available, fostering reproducibility and enabling future 
research in O -RAN and AI applications.  
Paper Organization. The remainder of this paper is orga - 
nized as follows: Section II explains the background and mo - 
tivation behind this work, and Section III details ORANSight - 
2.0, including the design of the RANSTRUCT framework. 
Section  IV introduces  data set benchmarks,  including  srsRAN - 
Bench, and describes the evaluation methodology. Section V 
presents the experimental results, including comparisons with 
closed -source models and energy consumption analysis. Sec - 
tion VI highlights the limitations of this work and Section VII 
concludes the paper and discusses future research directions.  
II. BACKGROUND  AND  MOTIVATION  
Large Language Models (LLMs) have demonstrated re- 
markable capabilities in natural language understanding and 
generation, making them a promising solution to address vari - 
ous challenges  within  Open  Radio  Access  Networks  (O-RAN). 
However,  the application  of LLMs  in O-RAN  is still in its early 
stages, primarily due to the absence of foundational models 
tailored to the unique requirements of O -RAN systems. LLMs 
are typically trained using a next -token prediction objective, 
where they learn to predict the next word or subword in a 
sequence based on vast and diverse datasets spanning fields 
such as medicine, history, and more [9]. This training process 
enables the models to understand the statistical relationships 
between tokens, which are the smallest units of text, and 
develop a broad, general -purpose understanding of language. 
However, as highlighted in [21],  when an LLM is intended  
for use in a specialized domain, adapting the model to that 
domain becomes imperative. This is because the patterns and 
distributions of tokens in domain -specific texts often differ 
significantly from those in the general -purpose datasets the 
model was originally trained on [21].  A prominent way to 
achieve this adaptation is by training the model with high - 
quality  domain -specific  data,  which  aligns  the model  with 
the unique  token  distribution  of the target  domain,  leading 
to improved performance in downstream tasks.  
A. Challenges  of Current  LLM  Solutions  for O-RAN 
Currently, LLM solutions for O -RAN are predominantly 
built on Retrieval -Augmented Generation (RAG), which is a 
primitive approach to instilling domain knowledge in LLMs. 
RAG relies on querying external knowledge bases for every 
prompt, which can provide contextually relevant responses  
but is inefficient for real -time O -RAN applications due to its 
computational  overhead and dependency on constant retrieval.  
2All 18 models along with the required code for inference available in  
Hugging Face, Link: https://huggingface.co/NextGLab  
3The ORANBench13K is available at: https://github.com/prnshv/ORAN - 
Bench -13K,  and the srsRANBench is available at:  
https://github.com/prnshv/srsRANBench.  

 
 
Fig. 1. Overall  pipeline  for ORANSight -2.0 models.  
 
In contrast, fine -tuning has been extensively documented to 
enable a model to internalize domain -specific knowledge, 
reducing the need for external queries and enabling faster, 
more accurate responses [9] [22].  This makes fine -tuning a 
more suitable approach for O -RAN applications, where real - 
time performance and efficiency are critical.  
B. Absence  of Domain -Specific  Datasets  for O-RAN 
A significant challenge in adapting LLMs to O -RAN is the 
lack of datasets  specifically  designed  for fine-tuning  LLMs 
in this domain. While datasets like TSpec -LLM [12] and 
TelecomGPT  [13] exist  for broader  telecom  standards,  they do 
not address  the specific  requirements  of O-RAN.  This absence 
of an O -RAN -specific dataset necessitates the creation of an 
approach that would allow for the generation of a fine -tuning 
set tailored to our needs. Furthermore, to the best of our 
knowledge,  no existing  work  addresses  the coding  abilities 
of these models in O -RAN, highlighting the importance of 
developing a dataset that also addresses this limitation.  
C. Efficiency  of Fine-Tuning  Pre-Trained  Models  
Fine-tuning pre -trained models is a more efficient approach 
than training LLMs from scratch for O -RAN -specific tasks. 
Pre-trained models already possess a broad understanding of 
language, and fine -tuning allows them to specialize in O - 
RAN without the need for extensive computational resources. 
This trend is evident in multiple studies that emphasize fine - 
tuning as a superior alternative to training from scratch [23].  
Additionally, several LLMs have been pre -trained to follow 
instructions, achieving superior results in various downstream 
tasks [9] [22].  This further underscores the effectiveness of 
fine-tuning as a strategy for adapting LLMs to specialized 
domains like O -RAN.  
D. Parameter -Efficient  Fine-Tuning  (PEFT)  Techniques  
Parameter -efficient fine -tuning (PEFT) techniques are es - 
sential for optimizing the fine -tuning process, particularly in 
resource -constrained environments. PEFT methods focus on 
updating only a small subset of the model’s parameters, sig - 
nificantly  reducing  computational  overhead  while  maintaining 
performance [22].  A complementary approach to further opti - 
mize efficiency without significant performance degradation  
is reducing  precision.  Model  weights  are typically  stored  
in full precision  (32-bit floating  point),  but lower  precision  formats,  such as 4-bit quantization,  drastically  reduce  memory 
and computational costs [24].  We draw inspiration from the 
work of [17],  which proposed Quantized Low -Rank Adap - 
tation (QLoRA) and demonstrated a PEFT paradigm with 
performance comparable to full precision networks but with 
substantially lower memory requirements.  
Taking these considerations into account, we develop 
ORANSight -2.0 by first curating datasets specifically tailored 
to the O -RAN ecosystem. We then employ advanced fine - 
tuning  techniques  to create  foundational  models  optimized 
for O -RAN -related text and code generation. This approach 
bridges the gap between general -purpose LLMs and the 
unique, specialized needs of the O -RAN domain, offering 
more  effective  and efficient  solutions  for this critical  domain.  
III. ORANS IGHT -2.0: FOUNDATIONAL  O-RAN  LLM S 
This section details the methodology employed in devel - 
oping ORANSight -2.0 – (i) the selection of 18 LLM models 
across five leading, open -source LLM model families, (ii) the 
RANSTRUCT framework, and (iii) the QLoRA -based fine - 
tuning of LLMs. Figure 1 shows the overall pipeline for 
ORANSight -2.0 models.  
A. Pre-trained  Model  Selection  
LLM  models  are generally  released  as a family  across 
different parameter sizes while maintaining the same training 
methodology. Most existing work in telecom and LLMs typ - 
ically targets a single model family to achieve a downstream 
task [30] or focuses on smaller LLMs ( < 10B parameters) for 
an initial study [9]. It is important to note that since each pub - 
licly available  model  family  undergoes  different  pretraining, 
a comprehensive  evaluation  should  include  as many  models 
as possible.  A model’s  overall  capability  is also strongly 
correlated with its parameter size, as deeper models typically 
exhibit superior representational abilities compared to smaller 
ones. This trend is seen in all the model releases, as the larger 
models  with the same  training  method  outperform  their smaller 
counterparts [25] [26] [28].  While studying smaller LLMs can 
provide  insights  into their applicability,  developing  a highly 
impactful foundational model requires targeting larger LLMs. 
Another  key aspect  of LLM  performance  is its context 
length,  which  determines  how much  prior  information,  or 
the maximum  number  of tokens,  a model  can effectively  


TABLE  I 
COMPARISON  OF VARIOUS  LARGE  LANGUAGE  MODELS  AND  THEIR  SPECIFICATIONS . 
 
Model  Developer  Parameter  Sizes  Pretraining  Details  Context  Length  
Qwen  2.5 [25] Alibaba  1.5B,  3B, 7B, 14B, Trained  on an 18-trillion -token  dataset  with supervised  fine- Up to 128K  tokens;  generates  
  32B tuning  using  over 1 million  samples  and multistage  rein- up to 8K tokens  
   forcement  learning  to enhance  instruction  following  and  
   reasoning  abilities.   
Gemma  2 [26] Google  2B, 9B, 27B Utilized  datasets  ranging  from  2 trillion  to 13 trillion  to- 8K tokens  
   kens,  including  web documents,  code,  and scientific  articles.   
   Employed  knowledge  distillation  and architectural  modifica -  
   tions  like interleaving  local -global  attention  and group -query   
   attention.   
Mistral  [27] Mistral  AI 7B, 12B (Nemo),  22B The company  has pioneered  Grouped -Query  Attention  7B, 22B,  8x7B:  32K tokens;  
  (Small),  8x7B  (GQA)  and Sliding  Window  Attention  (SWA)  for efficient  12B (Nemo):  128K  tokens  
   performance  in Mistral  models.  Despite  being  open  source,   
   specific  pretraining  dataset  details  are not publicly  available.   
Phi 3 [28] Microsoft  3.5: Mini;  3: Medium  These  models  are trained  on a 3.3-trillion -token  dataset  4K tokens  
   comprising  heavily  filtered  web data,  curated  educational   
   content,  and synthetic  ”textbook -like”  data to enhance  lan-  
   guage  understanding  and reasoning  capabilities.   
Llama  3 [29] Meta  AI 3.1: 8B, 70B;  3.2: 1B, Leveraged  a dataset  comprising  15.6 trillion  tokens,  focusing  3.1: 128K  tokens;  3.2: 8K to- 
  3B on diverse  language  tasks  to enhance  general  language  kens 
   understanding  and generation  capabilities.   
 
utilize  in a single  forward  pass.  Larger  models  often  fea- 
ture extended context windows, allowing them to better cap - 
ture dependencies  across  longer  sequences,  which  is crucial 
for tasks  involving  reasoning,  document  comprehension,  and 
telecom -related  applications  like log analysis  [31].  Recent 
advancements  have  significantly  expanded  the context  lengths 
of modern  LLMs,  with some  models  now supporting  up to 
128K  tokens.  However,  the ability  to retain  and recall  relevant 
information  across  long contexts  varies  among  models,  making 
it a crucial  factor  to include  larger  models  for ORANSight -2.0. 
Therefore,  we select  18 state-of-the-art models  covering  a 
broad range of parameter sizes (from 1B to 70B) that can be 
effectively  fine-tuned  using  our computational  resources.  All 
pre-trained  LLMs  serving  as the foundation  for ORANSight - 
2.0 have  demonstrated  exceptional  performance  across  diverse 
general text understanding and code generation benchmarks. 
The chosen models, along with their parent organizations, 
parameter sizes, and context lengths, are listed in Table I. 
B. RANSTRUCT  
ORANSight -2.0 addresses the problem of creating an 
instruction -based fine -tuning set by RAG based INSTRUC - 
Tion Generation, dubbed, RANSTRUCT . The proposed 
framework  is shown  in fig 2, and at its core,  we leverage 
two LLM agents, one for generating questions or instructions, 
and one LLM for the corresponding answers or responses.  
The framework is built on the fundamentals of RAG, which 
guarantees an accurate response if the scope of the instruction 
is within the RAG database [32] [33].  
Implementing RAG requires an efficient mechanism for 
storing  and retrieving  relevant  information.  Since  large  doc- uments cannot be processed in their entirety, they are first 
segmented into semantically meaningful chunks to facilitate 
precise  retrieval.  These  chunks  ensure  that when  a query  
is issued,  only the most  relevant  portions  of the dataset  
are accessed, reducing extraneous information and improving 
retrieval efficiency [11].  
To enable  effective  search  and retrieval,  each chunk  is 
converted  into a dense  vector  representation  using  a pre-trained 
embedding  model,  which  has an associated  context  length 
or the maximum  number  of tokens  that it has been  trained 
to represent as embeddings. This transformation captures the 
semantic  properties  of the text, allowing  for similarity -based 
retrieval based on contextual relevance [11].  The embeddings 
are then indexed in the RAG database, which supports high - 
speed  similarity  searches  across  large  datasets.  In our in- 
struction  generation  process,  questions  are formulated  such 
that they are explicitly  answerable  based  on the retrieved 
chunks. Similarly, during response generation, the model uti - 
lizes retrieved -context  from  the RAG  database  to produce 
outputs that are both accurate and contextually grounded [11].  
For ORANSight -2.0, we select   the 
Mistral-7B-Instruct -0.3   and    the 
Qwen-2.5-Instruct -1.5B models    due to  their 
efficiency  and compatibility  with RAG  pipelines.  Qwen,  
in particular,  has been  extensively  documented  to work 
effectively  with RAG  [32],  enabling  faster  outputs  and 
efficient  dataset  generation  despite  its smaller  size. This 
choice  ensures  that the framework  operates  efficiently  while 
maintaining  high-quality  outputs.  We run the framework  for 
two instances,  with the O-RAN  specifications  and individual 
files from  the srsRAN  project.  Below,  we outline  the workflow  

 
Fig. 2. The proposed  RANSTRUCT  Framework  
 
of RANSTRUCT,  as depicted  in Figure  2: 
1) Input Processing:  The framework begins with two pri - 
mary O -RAN data sources: O-RAN Specifications and the 
srsRAN  Code  Files . The O-RAN  Specifications  consist  of 116 
documents,  averaging  21,778  words  per document,  totaling  
2.53 million words. These specifications form the founda - 
tion for generating structured instruction -tuning datasets. The 
srsRAN Code files, on the other hand, comprises C++ source 
files from the srsRAN project, a widely used open -source 
implementation of the RAN stack by academia and industry. 
This codebase includes 4,980 files and 4.68 million words.  
To prepare  the data,  we employ  a Recursive  Splitter  4 
to generate semantically meaningful chunks. For the O -RAN 
Specifications,  two types  of chunks  are created:  RAG  Chunks 
(1,024  tokens)  for efficient  document  retrieval  within  the RAG 
pipeline and LTG (Long -Text Generation) Chunks (4,096 
tokens)  for generating  high-quality  question -answer  pairs.  The 
specific  token  values  are chosen  to comply  with the embedding 
generator’s  requirements.  In contrast,  for the srsRAN  codebase 
files, entire codebase files are used instead of chunking to 
preserve semantic context.  
2) FAISS  Construction:  As RAG  works  through  a database 
of embeddings  we encode  the RAG  chunks  using  the 
BGE-Small-EN-1.5 embedding  model  [34] to create  dense 
vector representations, that are then indexed using a FAISS 
(Facebook AI Similarity Search) database. We selected this 
particular  combination  due to its documented  performance 
in O -RAN applications from our previous work [11] and its 
alignment  with the open -source  nature  of our work.  The FAISS 
Construction concludes when we have exhausted all the RAG 
chunks and we process with the next step.  
3) Question Generation:  The third stage of RANSTRUCT 
involves  generating  a structured  database  of questions  that 
4A Recursive Splitter refers to a technique that breaks down complex tasks  
or large datasets into smaller, more manageable parts, processed iteratively.  
This helps in improving the efficiency and performance of LLMs by focusing  
on sub-problems,  especially  when  dealing  with hierarchical  structures  or large  
volumes of data.  are crucial  for the knowledge  base of an O-RAN  LLM. 
For each LTG chunk, we prompt the question generator  
model (Mistral-7B-Instruct -0.3) to generate mul - 
tiple unique  questions  that are strictly  within  the scope  of 
the LTG chunk and can be accurately answered if the same 
information  is provided  as context.  The model  is prompted 
to generate the questions as a list which can be parsed by a 
separate script.  
Once we have a list of questions across all possible LTG 
chunks, we subject this database to de -duplication and pars - 
ing. De -duplication is achieved by transforming the list of 
questions into a dictionary, where the questions serve as keys. 
This approach inherently removes duplicates, as dictionaries 
cannot have duplicate keys, ensuring that unique questions  
are retained. Parsing is achieved by filtering out truncated, 
malformed, or invalid questions to maintain dataset integrity. 
This process ensures that only high -quality, unique questions 
are retained for the final instruction -tuning dataset.  
4) Answer   Generation:   Once   we  have   unique 
and  high-quality   questions,   we  leverage   the 
Qwen-2.5-Instruct -1.5B to create  the corresponding 
response. Each question is subjected to the embedding 
generator to obtain its semantic representation. We leverage 
the embeddings through the FAISS database to obtain the top 
3 chunks. The Qwen model is then provided with the question 
and the retrieved  documents  as context  to generate  a response. 
Since  the questions  are generated  while  strictly  adhering  to the 
scope of an LTG chunk, the retrieved documents inherently 
contain  the necessary  information  to provide  accurate  answers. 
This ensures  that the context  provided  by the Qwen  model 
is always relevant and sufficient for generating high -quality 
responses. Finally, the generated answer and the original 
question are appended to the fine -tuning set.  
Dataset Metrics: After exhausting all the questions, we 
obtain the dataset for training ORANSight -2.0 models which 
contain 1,51,500 unique instruction and answer pairs across 
the srsRAN  project  codebase  and the O-RAN  specifications,  


FP 32 
4-bit 
FP 32 leading to a total of 29,864,388 words. Out of this, 88,766 
samples or 58.591% are related to the srsRAN project code - 
base, and 62,734 or 41.409% are related to the O -RAN 
Specifications. Throughout the rest of the paper, this dataset  
is referred to as the RANSTRUCT dataset.  
C. QLoRA  
Fine-tuning of each selected LLM model on the 
RANSTRUCT dataset in ORANSight -2.0 is performed 
through the QLoRA (Quantized Low -Rank Adaptation) tech - 
nique. As introduced in section II, it is a PEFT method that 
enables the training of large language models in resource - 
sensitive scenarios without compromising performance. At its 
core,  QLoRA  combines  two key techniques:  Quantization  and 
Low-rank adaptation (LoRA) [17].  Quantization reduces the 
memory  footprint  by mapping  the original  16-bit floating -point 
weights  to 4-bit integers,  significantly  lowering  computational 
and storage requirements [17].  Low-rank adaptation, on the 
other  hand,  introduces  trainable  low-rank matrices  to adapt  the 
pre-trained model weights, drastically reducing the number of 
trainable  parameters.  Mathematically,  for a pre-trained  weight  
matrix  W ∈ Rm×n, LoRA  introduces  a low-rank decomposi - 
tion where the weight update is represented as ∆W = AB, 
with A ∈ Rm×r and B ∈ Rr×n, where r ≪ min(m, n) [35].  
Instead  of modifying  W directly,  the adapted  transformation  
is expressed  as: 
 
Y = X(W + sAB ), (1) 
where  s is a scaling  factor.  This decomposition  ensures  that 
only a small number of parameters are updated during fine - 
tuning, preserving the efficiency of the model while maintain - 
ing expressiveness. QLoRA further enhances this efficiency  
by proposing 4 -bit quantization (represented as NormalFloat  
or NF4) to store pre -trained model weights, which are de - 
quantized to 16 -bit (represented as bfloat16 or BF16) during 
computation [17].  The dequantization process is denoted as 
dD, where  the stored  NF4 weights  are first scaled  using  a 
32-bit constant (denoted as s1 and then using an 8 -bit 
constant (denoted as s2 ), this double step process greatly 
reduces memory consumption, allowing us to fine -tune larger 
ORANSight -2.0 models on our computational hardware. Fi - 
nally, the forward pass is performed as  Mistral 8x7B and Llama 70B, which are trained using the 
AQLM library [37] and a 2 -bit precision. We rely on AQLM 
(Adapter -based  Quantized  Language  Models)  5 due to additive 
quantization  and the possibility  of working  with larger  models 
in the same setup.  
As shown  in sections  II and III, we primarily  leverage  a 4-bit 
precision as it has been documented throughout the literature 
to be the best tradeoff  for generation  fidelity  and compute 
resources.  It allows  us to pursue  a thorough  comparative 
analysis  that is shown  in section  V. We train our models  for 
a single  epoch  with the maximum  possible  batch  size of up 
to 8, which is determined by the available GPU memory. The 
rank and alpha hyperparameters for the QLoRA configuration 
are model -specific,  with the maximum  values  set to 256 for 
both.  These  values  are chosen  based  on the model  architecture 
and the memory available for the best possible performance. 
To further  optimize  resource  utilization,  we employ  data 
packing during training. Packing involves concatenating mul - 
tiple shorter sequences into a single longer sequence, thereby 
reducing  the number  of padding  tokens  required . This tech- 
nique  helps  us maximize  GPU  utilization  by ensuring  that 
the computational  resources  are fully  utilized  for meaningful  
computations  rather  than processing  padding  tokens  [38]. 
A. Benchmarks  
To evaluate  the performance  of the ORANSight -2.0 models, 
we leverage  and extend  our previous  work,  ORAN -Bench -13K 
[11],  and introduce  a new benchmark,  srsRANBench,  designed 
specifically for code generation and codebase understanding 
for the srsRAN  Project.  Both  benchmarks  are carefully  curated 
to ensure  a reliable  and thorough  evaluation  while  maintaining 
computational efficiency.  
1) ORANBench:  We leverage  ORAN -Bench -13K [11],  our 
previously developed benchmark, consisting of 13,952 metic - 
ulously curated multiple -choice questions generated from the 
O-RAN  specification  documents  across  three  difficulty  levels: 
easy, intermediate , and difficult . For this work, we create 
ORANBench by randomly selecting a subset of 500 samples 
from each difficulty category, resulting in a balanced evalua - 
tion set of 1,500  questions.  This streamlined  version  retains  the 
diversity and rigor of the original benchmark while reducing 
computational overhead, enabling efficient evaluation.  
2) srsRANBench: Building on the framework used for 
ORAN -Bench -13K,  we introduce  srsRANBench,  a novel  
YBF 16 = XBF 16 · dD(s1 2 
8-bit , WNF 4) + (XAB )BF 16 
(2) benchmark  focused  on code  generation  and codebase  under - 
standing  for the srsRAN  project.  Using  the same  methodology,  
where  XBF 16 represents the input in BF16 precision, WNF 4 
are the stored  4-bit weights,  and XBF 16ABF 16BBF 16 is 
analogous to the the LoRA -based low -rank update.  
IV. EXPERIMENTAL  SETUP  
The hardware configuration we leverage consists of an 
Intel(R) Core(TM) i9 -14900KF CPU with 62 GB of RAM, 
paired with an NVIDIA GeForce RTX 4090 GPU with 24 GB 
of GDDR6X  memory.  All ORANSight -2.0 models  are trained 
with the unsloth  library  [36],  and a 4-bit precision  except  for we curate a dataset of 1,502 samples by randomly selecting 
cpp files from the entire srsRAN codebase. This benchmark 
evaluates the ability of LLMs to generate syntactically and 
semantically  correct  code,  as well as their understanding  
of the codebase. The random selection process ensures a 
representative and unbiased evaluation, while the manageable 
size of the dataset  allows  for computationally  efficient  testing.  
5AQLM is a method that combines adapter layers and quantization to  
efficiently fine -tune large language models with fewer resources, reducing  
both memory and computational requirements.  , s 

TABLE  II 
ORANS IGHT -2.0 AGAINST  TWO  CLOSED -SOURCE  LLM  BASELINES  ACROSS  VARIOUS  MODEL  SIZES  
 
Model  Family  Parameters  (B) ORANBench  srsRANBench  Cumulative  Score  
Easy Medium  Difficult  Average  
ORANSight -Gemma -2 2 0.698  0.618  0.584  0.633  0.848  0.74 
9 0.822  0.718  0.680  0.740  0.834  0.787  
27 0.818  0.722  0.702  0.747  0.911  0.829  
ORANSight -Mistral  7 (v3) 0.730  0.618  0.612  0.650  0.863  0.756  
12 (Nemo)  0.784  0.710  0.650  0.715  0.828  0.772  
22 (Small)  0.802  0.700  0.658  0.720  0.860  0.79 
8x7 (AQLM)  0.780  0.654  0.626  0.687  0.753  0.72 
ORANSight -Llama  3.2: 1 0.438  0.350  0.364  0.384  0.565  0.474  
3.2: 3 0.708  0.582  0.540  0.610  0.768  0.689  
3.1: 8 0.728  0.730  0.618  0.692  0.728  0.71 
3.1: 70 (AQLM)  0.820  0.680  0.652  0.717  0.761  0.739  
ORANSight -Phi 3.5: Mini 0.716  0.670  0.658  0.681  0.754  0.718  
3: Medium  0.754  0.710  0.682  0.715  0.783  0.749  
ORANSight -Qwen -2.5 1.5 0.660  0.586  0.606  0.617  0.804  0.71 
3 0.704  0.642  0.628  0.658  0.857  0.758  
7 0.788  0.720  0.696  0.735  0.837  0.786  
14 0.804  0.730  0.684  0.739  0.761  0.75 
32 0.856  0.784  0.738  0.793  0.796  0.794  
(Baseline)  ChatGPT  4o-mini 
4o 0.766  
0.792  0.727  
0.760  0.677  
0.693  0.723  
0.752  0.755  
0.769  0.739  
0.76 
(Baseline)  Gemini  1.5:8B  
1.5 0.723  
0.743  0.665  
0.707  0.631  
0.669  0.673  
0.706  0.767  
0.775  0.72 
0.741  
 
Both  Benchmark  samples  are showcased  in Appendix  A for 
ORANBench and Appendix B for srsRANBench.  
B. RAG -based  ORANSight -2.0 augmentation  for inference  
As RAG is usually perceived as a tool to augment the 
outputs of an LLM with domain knowledge, it can also be 
used on top of a fine -tuned LLM to enhance its performance. 
RAG  adds relevant  context  to the model  inputs  which  can help 
the model figure out the correct response by grounding its 
answers in retrieved information, reducing hallucinations, and 
improving accuracy, especially for knowledge -intensive tasks 
[32].  We adapt RAG for ORANSight -2.0 inference from the 
first version of ORANSight as described in our previous work 
[11],  further advancing the state -of-the-art in O -RAN -specific 
tasks.  The Inference  RAG  pipeline  leverages  a FAISS  database 
constructed using the open source bge-large-en-v1.5 
embedding model [34],  which supports the O -RAN Specifica - 
tion and the srsRAN codebase files, totaling 88,808 chunks 
and 7,236,372 words . 
C. Energy  Consumption  
We calculate  the energy  characteristics  for table  III by lever - 
aging  the Codecarbon  package  [39].  We use the package to 
observe the energy usage across training, standard inference, 
and inference with RAG. Codecarbon allows us to monitor 
detailed energy characteristics across the graphics processing 
unit (GPU), Central Processing Unit (CPU), and Random 
Access Memory (RAM). This enables a granular comparison 
between different model operations, providing insights into  
the trade -offs between performance improvements from RAG 
and the associated energy costs. The energy metrics also 
enable comparing ORANSight -2.0 models in computational 
efficiency, supporting informed decision -making in model se - 
lection for real -world deployment.  V. RESULTS  
From the table II, we can observe that from the 18 
ORANSight -2.0 models, 12 models outperform Gemini 1.5 
Flash 8B, 10 models outperform Gemini 1.5, and 11 models 
outperform ChatGPT -4o Mini. However, we primarily aim to 
compare our work with ChatGPT -4o (4o) which is widely 
regarded  as the best LLM  model  for every  use case and 
can be assumed to be the deepest model in our comparative 
analysis.  A total of 6 models  outperform  4o for the cumulative 
score and 11 models outperform 4o on the srsRANBench. 
Across the 6 individual models, ORANSight -Qwen -2.5 32B, 
achieves the highest cumulative score of 0.794, surpassing 4o 
(0.760) by 4.47%, it excels in both O -RAN -specific tasks and 
coding challenges, scoring 0.793 on ORANBench and 0.796 
on srsRANBench. Following closely, ORANSight -Gemma -2 
27B attains a cumulative score of 0.829, exceeding 4o by 
9.08%  with exceptional  performance  on srsRANBench  (0.911) 
and a comparable  result  on ORANBench  (0.747).  Two Mistral 
Models, ORANSight -Mistral 22B (Small) and ORANSight - 
Mistral  12B (Nemo)  outperform  4o with cumulative  scores 
of 0.790 and 0.772, respectively. The smallest models to 
outperform the baseline are ORANSight -Gemma -2 9B and 
ORANSight -Qwen -2.5 7B, showcasing the best performance 
with substantially fewer parameters.  
For the srsRANBench, the ORANSight -Gemma -2 27B 
model achieves the highest score (0.911) which is 12.7%  
better  than the average  model  and outperforms  4o (0.769) 
by 18.5%.  The overall  best model  ORANSight -Qwen -2.5 32B 
also performs well, with a score of 0.796, which is lesser than 
ORANSight -Gemma -2 27B but is still better  than our baseline. 
For ORANBench, the ORANSight -Qwen -2.5 32B leads with 
an average score of 0.793, followed by ORANSight -Gemma - 
2 27B (0.747)  and ORANSight -Mistral  22B (Small)  (0.720).  

 
Fig. 3. Performance  comparison  of different  model  families  with and without  RAG.  The categories  on the x-axis represent  different  evaluation  criteria:  
’E’, ’M’, and ’D’ correspond to Easy, Medium, and Difficult samples from ORANBench. ’O -RAN’ represents the cumulative score across all ORANBench  
samples.  ’srs’ corresponds  to the evaluation  on srsRANBench,  while  ’C’ represents  the final cumulative  average  score  across  all benchmarks.  
 
These models consistently outperform closed -source counter - 
parts,  particularly  in the difficult  category,  where  ORANSight - 
Qwen -2.5 32B achieves  a score  of 0.738.  Ideally,  we expected 
consistency in performance across the two benchmarks, but  
the pertaining of the base LLM can play a role in the post - 
fine-tuning result, and we can assume that the proficiency of 
ORANSight -Gemma -2 27B in general coding tasks translated 
well for srsRAN.  
Performance generally increases linearly with parameter 
size, with larger  models  like ORANSight -Qwen -2.5 32B 
and ORANSight -Gemma -2 27B achieving the highest scores. 
However, diminishing returns appear beyond a certain thresh - 
old, as seen with the ORANSight -Mistral 8x7B (AQLM) 
model,  which  underperforms  despite  its larger  size. The reason 
for diminishing performance after 32B can be attributed to 
heavy quantization as the AQLM models were trained using 
2bits  instead  of a 4bit setup.  This was employed  due to the lim- 
ited computing resources, but the AQLM models gave a com - 
parable performance to 4o -mini and Gemini 1.5: 8B. Smaller 
models, such as ORANSight -Qwen -2.5 3B, demonstrate that 
fine-tuning can significantly enhance performance even with 
limited parameters, making them viable for resource -sensitive 
scenarios.  
To compare the performance across the different model 
families,  ORANSight -Qwen -2.5 emerges  as the top performer, 
with all variants  (1.5B  to 32B)  surpassing  ChatGPT -4o 
Mini and consistent performances for the cumulative scores. 
ORANSight -Gemma -2 particularly outperforms every model 
for srsRAN and two out of three models also provide superior 
performance on ORANBench. For the Mistral series the 22B 
(Small) model, outperforms 4o Mini and competes closely 
with 4o, though the 8x7B (AQLM) model struggles to scale 
effectively. The ORANSight -LLaMA series delivers mixed 
results, with the 70B (AQLM) model performing well, while 
the smaller variants (1B and 3B) struggle with ORANBench, 
emphasizing the importance of model size. Lastly, the Phi 
family  performs  similarly  to LLaMA  with both models  usually 
underperforming  across  ORANBench  and srsRANBench  when 
compared  against  the other  ORANSight -2.0 models.  To further 
evaluate  the performance of ORANSight -2.0 with RAG -based  augmentation and for energy consumption, we select four 
models across different parameter ranges that gave the best 
cumulative performance:  
1) 0-10B:  ORANSight -Gemma -2 9B 
2) 11-20B:  ORANSight -Mistral -Nemo  
3) 21-30B:  ORANSight -Gemma -2 27B 
4) 31-40B:  ORANSight -Qwen -2.5 32B 
A. RAG  for Inference  
The performance of ORANSight -2.0 with RAG -based aug - 
mentation,  as shown  in Figure  3, highlights  that RAG  enhances 
the performance  of fine-tuned  LLMs  and advances  the state-of- 
the-art for both ORANBench  and srsRANBench.  ORANSight - 
Mistral -Nemo  (12B)  shows  a 6.35%  improvement  in the cumu - 
lative  score  (0.821  with RAG  vs. 0.772  without),  with a 20.3% 
gain in the Difficult category of ORANBench. ORANSight - 
Gemma -2 9B achieves a 5.97% increase in cumulative score 
(0.834 with RAG vs. 0.787 without), with consistent im - 
provements  across  all categories,  particularly  in srsRANBench 
(4.8%  increase).  However,  ORANSight -Gemma -2 27B experi - 
ences a slight 1.08% drop in the cumulative score (0.820 with 
RAG vs. 0.829 without), likely due to low LLM recall, where 
the retrieval mechanism may not align well with the model’s 
internal knowledge. ORANSight -Qwen -2.5 32B exhibits the 
most significant improvement, with a 7.42% increase in the 
cumulative score (0.854 with RAG vs. 0.795 without) and a 
19.0% gain in the Difficult category of ORANBench, despite  
a minor  drop in srsRANBench,  suggesting  a trade -off between 
O-RAN knowledge and coding tasks.  
B. Energy  Characteristics  
From the table III we can observe that the energy usage, 
measured in watt -hours (Wh), increases consistently with 
model size across all phases. The GPU remains the dominant 
contributor to total energy consumption, with CPU and RAM 
playing secondary roles. The training energy was recorded 
over 100 randomly selected samples with a batch size of 1, 
while inference energy was measured by generating 32 tokens 
per sample across 100 random prompts. For RAG -based in - 
ference,  additional  retrieval  operations  were  performed  before  


TABLE  III 
THE ENERGY  CONSUMPTION  CHARACTERISTICS  OF THE  FOUR  PRIMARY  MODELS  OF ORANS IGHT -2.0 ACROSS  TRAINING , INFERENCE , AND  INFERENCE  
WITH  RETRIEVAL -AUGMENTED  GENERATION  (RAG).  THE TOTAL  ENERGY  IS REPRESENTED  IN WH, AND  THE  CPU,  GPU,  AND  RAM  COLUMNS  SHOW  
THE  PERCENTAGE  USAGE  FOR  THE  DEVICE , RESPECTIVELY . 
 
Model  Training  Inference  Inference  (RAG)  
Energy  Total  CPU GPU  RAM  Energy  Total  CPU GPU  RAM  Energy  Total  CPU GPU  RAM  
ORANSight - 
Gemma -2: 9B 1.579  146.3  822.2  31.5 2.102  154.3  758.3  87.4 2.274  150.5  770.7  78.8 
ORANSight - 
Mistral -Nemo  1.865  110.6  828.6  61.0 2.220  125.5  804.9  69.6 2.512  120.7  811.3  68.0 
ORANSight - 
Gemma -2: 27B 4.089  98.6 847.1  54.4 4.955  116.3  819.5  64.2 5.407  109.8  829.9  60.5 
ORANSight - 
Qwen -2.5: 32B 4.404  96.4 850.4  53.0 5.199  112.0  826.4  61.8 5.891  107.1  833.9  59.0 
 
token generation, capturing the additional computational cost 
with the same  prompts  from  the standard  inference  experiment. 
Training  energy  scales  with model  size, with the smallest 
model,  ORANSight -Gemma -2: 9B, consuming  1.579  Wh, 
while the largest, ORANSight -Qwen -2.5: 32B, requires 4.404 
Wh. Inference  demands  more  energy  than training,  with a 
33–38% increase  in total consumption.  The largest  increase 
is observed  in ORANSight -Qwen -2.5: 32B,  where  inference 
energy  rises to 5.199  Wh, primarily  driven  by GPU  utilization. 
Similarly,  RAG  further  increases  energy  usage  across  all 
models,  with an average  rise of 9.5%  compared  to standard 
inference.  This overhead  is particularly  pronounced  in larger 
models,  with ORANSight -Qwen -2.5: 32B showing  an increase 
from 5.199 Wh to 5.891 Wh, highlighting the computational 
cost of retrieval  operations.  Across  all configurations,  GPU 
energy  consumption  remains  the primary  factor,  contributing  
to over 75%.  
VI. DISCUSSION  AND  LIMITATIONS  
While this study demonstrates significant advancements in 
fine-tuning  and evaluating  large  language  models  for O-RAN - 
specific tasks, several limitations should be acknowledged:  
• Compute Constraints : The available computational re - 
sources  were  limited  to 24GB  of VRAM,  which  constrained  
the maximum model size that could be fine -tuned. The largest 
model  we could  train was a 32B parameter  model  with 4- 
bit precision.  Fine-tuning  deeper  models  with higher  precision 
could potentially yield better results, but this was not feasible 
given the current hardware limitations.  
• Precision Limitations for Larger Models : For the 8x7B 
Mistral  and Llama  70B models,  as shown  in section  IV, we 
had to rely on AQLM with 2 -bit precision due to resource 
constraints. This lower precision likely limited the overall 
potential performance of these models, as higher precision 
typically allows for more accurate computations and better 
model performance.  
• RAG Performance Issues : The Retrieval -Augmented 
Generation  (RAG)  pipeline,  while  generally  beneficial,  showed  
limitations in certain cases, particularly with the ORANSight - 
Gemma -2: 27B model, which experienced a slight perfor - 
mance drop. This could be attributed to low LLM recall [40],  
where  the retrieval  mechanism  may not align  well with the model’s  internal  knowledge.  Incorporating  a reranker  [40] into 
the RAG pipeline could potentially address this issue and 
improve overall performance.  
• Dataset Scope : The dataset  used in this study  is primarily 
focused  on the srsRAN  project  codebase  and O-RAN  specifi - 
cations. While this provides a solid foundation, the inclusion 
of additional data sources, such as prominent research papers 
and other software frameworks [9], could further enhance the 
model’s performance and generalization capabilities.  
• Reasoning Capabilities : This study did not explore the 
reasoning  capabilities  [41] of the fine-tuned  ORANSight -2.0 
models. Enhancing the models’ ability to perform complex 
reasoning tasks is an important area for future work, as it  
could significantly improve their utility in real -world O -RAN 
applications.  
Despite  these  limitations,  this work  is the first to fine- 
tune and evaluate foundational LLMs for O -RAN tasks, 
ORANSight -2.0. The results show that open -source, fine - 
tuned ORANSight -2.0 models can achieve state -of-the-art 
performance, sometimes surpassing closed -source alternatives 
like ChatGPT -4o. The success of ORANSight -Qwen -2.5 32B 
and ORANSight -Gemma -2 27B highlights the effectiveness  
of domain -specific  fine-tuning.  These  ORANSight -2.0 models 
should act as foundational agents, and further fine -tuning 
while addressing these limitations could lead to more efficient 
models, advancing O -RAN -specific AI applications.  
VII. CONCLUSION  AND  FUTURE  WORK 
In this work, we presented ORANSight -2.0, the first com - 
prehensive set of domain -specific foundational Large Lan - 
guage  Models  (LLMs)  for Open  Radio  Access  Networks 
(O-RAN).  By fine-tuning  18 models  ranging  from  1B to 
70B parameters with the novel RANSTRUCT dataset and 
QLoRA, we addressed the critical gap in domain -specific 
foundational models for O -RAN, providing open -source alter - 
natives that outperform state -of-the-art closed -source models 
like ChatGPT -4o and Gemini 1.5. The introduction of the 
RANSTRUCT framework enabled the generation of high - 
quality instruction -tuning datasets, while the development of 
srsRANBench provided a novel benchmark for evaluating 
code generation and codebase understanding in 5G O -RAN 
systems. Our experiments demonstrated that fine -tuned open - 
source  models,  particularly  ORANSight -Qwen -2.5 32B,  and 

ORANSight -Gemma -2 27B,  achieve  superlative  performance, 
surpassing closed -source counterparts by significant margins 
on both O-RAN  specifications  and coding  tasks.  Additionally, 
the integration  of Retrieval -Augmented  Generation  (RAG)  fur- 
ther enhanced  model  performance,  with ORANSight -Qwen - 
2.5 32B (RAG) achieving the highest cumulative score of 
0.854, which is a 12.368% improvement from ChatGPT -4o, 
which is our primary baseline. Energy consumption analysis 
revealed that while larger models and RAG -based inference 
incur  higher  computational  costs,  the performance  gains  justify 
the trade -offs, especially in resource -constrained scenarios.  
For future  work,  we aim to further  focus  on fine-tuning  these 
models  and remedy  our current  work’s  limitations.  We strongly 
believe that by advancing such causal reasoning models, we 
can enhance multiple O -RAN -specific tasks and reduce the 
complexity and manual effort required for real -time network 
management, optimization, and troubleshooting.  
VIII. ACKNOWLEDGMENT  
The project is partially supported by the Public Wireless 
Supply  Chain  Innovation  Fund  under  Federal  Award  ID Num - 
ber 26 -60-IF010 and 51 -60-IF007.  
REFERENCES  
[1] N. D. Tripathi and V. K. Shah, Fundamentals of O -RAN . John Wiley &  
Sons, 2025.  
[2] Y. Du, H. Deng,  S. C. Liew,  K. Chen,  Y. Shao,  and H. Chen,  “The  power  
of large language models for wireless communication system develop - 
ment:  A case study  on fpga platforms,”  arXiv  preprint  arXiv:2307.07319 , 
2023.  
[3] A. Kucharavy, “Adapting llms to downstream applications,” in Large  
Language Models in Cybersecurity: Threats, Exposure and Mitigation , 
pp. 19–29, Springer  Nature  Switzerland  Cham,  2024.  
[4] L. Bariah, Q. Zhao, H. Zou, Y. Tian, F. Bader, and M. Debbah, “Large  
language models for telecom: The next big thing?,” arXiv preprint  
arXiv:2306.10249 , 2023.  
[5] A. Maatouk, N. Piovesan, F. Ayed, A. De Domenico, and M. Debbah,  
“Large language models for telecom: Forthcoming impact on the indus - 
try,” IEEE Communications Magazine , vol. 63, no. 1, pp. 62 –68, 2025.  
[6] Y. Xu, Y. Chen,  X. Zhang,  X. Lin, P. Hu, Y. Ma, S. Lu, W. Du, 
Z. Mao, E. Zhai, et al. , “Cloudeval -yaml: A practical benchmark for  
cloud configuration generation,” Proceedings of Machine Learning and  
Systems , vol. 6, pp. 173 –195, 2024.  
[7] C. Hu, H. Zhou, D. Wu, X. Chen, J. Yan, and X. Liu, “Self -refined  
generative foundation models for wireless traffic prediction,” arXiv  
preprint arXiv:2408.10390 , 2024.  
[8] X. Liu, S. Gao, B. Liu, X. Cheng, and L. Yang, “Llm4wm: Adapting  
llm for wireless  multi -tasking,”  arXiv  preprint  arXiv:2501.12983 , 2025.  
[9] A. Maatouk, K. C. Ampudia, R. Ying, and L. Tassiulas, “Tele -llms: A  
series of specialized large language models for telecommunications,”  
arXiv preprint arXiv:2409.05314 , 2024.  
[10] M. Kotaru,  “Adapting  foundation  models  for information  syn- 
thesis of wireless communication specifications,” arXiv preprint  
arXiv:2308.04033 , 2023.  
[11] P. Gajjar  and V. K. Shah,  “Oran -bench -13k: An open  source  benchmark  
for assessing llms in open radio access networks,” arXiv preprint  
arXiv:2407.06245 , 2024.  
[12] R. Nikbakht,  M. Benzaghta,  and G. Geraci,  “Tspec -llm: An open -source  
dataset for llm understanding of 3gpp specifications,” arXiv preprint  
arXiv:2406.01768 , 2024.  
[13] H. Zou, Q. Zhao,  Y. Tian,  L. Bariah,  F. Bader,  T. Lestable,  and 
M. Debbah, “Telecomgpt: A framework to build telecom -specfic large  
language models,” arXiv preprint arXiv:2407.09424 , 2024.  
[14] M. K. Motalleb,  C. Benzaid,  T. Taleb,  M. Katz,  V. Shah -Mansouri,  and 
J. Song, “Towards secure intelligent o -ran architecture: Vulnerabilities,  
threats and promising technical solutions using llms,” arXiv preprint  
arXiv:2411.08640 , 2024.  [15] F. Lotfi, H. Rajoli, and F. Afghah, “Llm -augmented deep reinforcement  
learning for dynamic o -ran network slicing,” environments , vol. 2, p. 4.  
[16] V. Hanke,  T. Blanchard,  F. Boenisch,  I. E. Olatunji,  M. Backes,  and 
A. Dziedzic, “Open llms are necessary for private adaptations and  
outperform  their closed  alternatives,”  in ICML  2024  Next Generation  
of AI Safety Workshop , 2024.  
[17] T. Dettmers, A. Pagnoni, A. Holtzman, and L. Zettlemoyer, “Qlora:  
Efficient  finetuning  of quantized  llms,”  Advances  in Neural  Information  
Processing Systems , vol. 36, 2024.  
[18] S. Iqbal and J. M. Hamamreh, “A comprehensive tutorial on how to  
practically  build  and deploy  5g networks  using  open -source  software  and 
general -purpose, off -the-shelf hardware,” RS Open J. Innov. Commun.  
Tech , vol. 2, no. 6, pp. 1 –28, 2021.  
[19] S. Wang,  Y. Wang,  L. Jiang,  Y. Chang,  K. Zhao,  L. Chen,  C. Gao,  et al., 
“Assessing the clinical support capabilities of chatgpt 4o and chatgpt 4o  
mini in managing  lumbar  disc herniation,”  European  Journal  of Medical  
Research , vol. 30, no. 1, pp. 1 –9, 2025.  
[20] G. Team,  P. Georgiev,  V. I. Lei, R. Burnell,  L. Bai, A. Gulati,  G. Tanzer,  
D. Vincent,  Z. Pan, S. Wang,  et al., “Gemini  1.5: Unlocking  multimodal  
understanding across millions of tokens of context,” arXiv preprint  
arXiv:2403.05530 , 2024.  
[21] C¸ . Yıldız, N. K. Ravichandran, P. Punia, M. Bethge, and B. Ermis,  
“Investigating continual pretraining in large language models: Insights  
and implications,” arXiv preprint arXiv:2402.17400 , 2024.  
[22] C. Christophe,  P. K. Kanithi,  P. Munjal,  T. Raha,  N. Hayat,  R. Rajan,  
A. Al -Mahrooqi, A. Gupta, M. U. Salman, G. Gosal, et al. , “Med42 – 
evaluating fine -tuning strategies for medical llms: Full -parameter vs.  
parameter -efficient  approaches,”  arXiv  preprint  arXiv:2404.14779 , 2024.  
[23] N. Tajbakhsh,  J. Y. Shin,  S. R. Gurudu,  R. T. Hurst,  C. B. Kendall,  M. B. 
Gotway,  and J. Liang,  “Convolutional  neural  networks  for medical  image  
analysis: Full training or fine tuning?,” IEEE transactions on medical  
imaging , vol. 35, no. 5, pp. 1299 –1312, 2016.  
[24] Y. Park, J. Hyun, S. Cho, B. Sim, and J. W. Lee, “Any -precision llm:  
Low-cost deployment of multiple, different -sized llms,” arXiv preprint  
arXiv:2402.10517 , 2024.  
[25] A. Yang,  B. Yang,  B. Zhang,  B. Hui, B. Zheng,  B. Yu, C. Li, D. Liu, 
F. Huang, H. Wei, et al. , “Qwen2. 5 technical report,” arXiv preprint  
arXiv:2412.15115 , 2024.  
[26] G. Team,  M. Riviere,  S. Pathak,  P. G. Sessa,  C. Hardin,  S. Bhupatiraju,  
L. Hussenot, T. Mesnard, B. Shahriari, A. Rame´, et al. , “Gemma 2:  
Improving open language models at a practical size,” arXiv preprint  
arXiv:2408.00118 , 2024.  
[27] A. Q. Jiang,  A. Sablayrolles,  A. Mensch,  C. Bamford,  D. S. Chaplot,  
D. d. l. Casas, F. Bressand, G. Lengyel, G. Lample, L. Saulnier, et al. , 
“Mistral 7b,” arXiv preprint arXiv:2310.06825 , 2023.  
[28] M. Abdin,  J. Aneja,  H. Awadalla,  A. Awadallah,  A. A. Awan,  N. Bach,  
A. Bahree, A. Bakhtiari, J. Bao, H. Behl, et al. , “Phi -3 technical report:  
A highly  capable  language  model  locally  on your phone,”  arXiv  preprint  
arXiv:2404.14219 , 2024.  
[29] A. Dubey,  A. Jauhri,  A. Pandey,  A. Kadian,  A. Al-Dahle,  A. Letman,  
A. Mathur, A. Schelten, A. Yang, A. Fan, et al. , “The llama 3 herd of  
models,” arXiv preprint arXiv:2407.21783 , 2024.  
[30] W. Lee and J. Park, “Llm -empowered resource allocation in wireless  
communications systems,” arXiv preprint arXiv:2408.02944 , 2024.  
[31] Y. Ding,  L. L. Zhang,  C. Zhang,  Y. Xu, N. Shang,  J. Xu, F. Yang,  and 
M. Yang, “Longrope: Extending llm context window beyond 2 million  
tokens,” arXiv preprint arXiv:2402.13753 , 2024.  
[32] W. Su, Y. Tang,  Q. Ai, J. Yan,  C. Wang,  H. Wang,  Z. Ye, Y. Zhou,  
and Y. Liu, “Parametric  retrieval  augmented  generation,”  arXiv  preprint  
arXiv:2501.15915 , 2025.  
[33] H. N. Patel,  A. Surti,  P. Goel,  and B. Patel,  “A comparative  analysis  
of large language models with retrieval -augmented generation based  
question  answering  system,”  in 2024  8th International  Conference  on 
I-SMAC  (IoT in Social,  Mobile,  Analytics  and Cloud)(I -SMAC) , pp. 792– 
798, IEEE, 2024.  
[34] S. Xiao, Z. Liu, P. Zhang, and N. Muennighoff, “C -pack: Packaged  
resources to advance general chinese embedding,” 2023.  
[35] H. Qin, X. Ma, X. Zheng,  X. Li, Y. Zhang,  S. Liu, J. Luo, X. Liu, 
and M. Magno, “Accurate lora -finetuning quantization of llms via  
information retention,” arXiv preprint arXiv:2402.05445 , 2024.  
[36] M. H. Daniel  Han and U. team,  “Unsloth,”  2023.  
[37] V. Egiazarian,  A. Panferov,  D. Kuznedelev,  E. Frantar,  A. Babenko,  and 
D. Alistarh,  “Extreme  compression  of large  language  models  via additive  
quantization,” arXiv preprint arXiv:2401.06118 , 2024.  

Question: Which component of the O -RAN architecture is 
responsible for controlling the radio access network in near 
real-time?  
Options:  
gNB-CU 
Near -RT RIC 
O-CU-CP 
FHGW  
Answer:  2 [38] S. Wang, G. Wang, Y. Wang, J. Li, E. Hovy, and C. Guo, “Packing  
analysis: Packing is more appropriate for large models or datasets in  
supervised fine -tuning,” arXiv preprint arXiv:2410.08081 , 2024.  
[39] B. Courty,  V. Schmidt,  S. Luccioni,  Goyal -Kamal,  MarionCoutarel,  
B. Feld,  J. Lecourt,  LiamConnell,  A. Saboni,  Inimaz,  supatomic,  
M. Le´val,  L. Blanche,  A. Cruveiller,  ouminasara,  F. Zhao,  A. Joshi,  
A. Bogroff,  H. de Lavoreille,  N. Laskaris,  E. Abati,  D. Blank,  Z. Wang,  
A. Catovic,  M. Alencon,  M. Stechły,  C. Bauer,  L. O. N. de Arau´jo,  JPW,  
and MinervaBooks, “mlco2/codecarbon: v2.4.1,” May 2024.  
[40] M. Jacob,  E. Lindgren,  M. Zaharia,  M. Carbin,  O. Khattab,  and 
A. Drozdov,  “Drowning  in documents:  Consequences  of scaling  reranker  
inference,” arXiv preprint arXiv:2411.11767 , 2024.  
[41] J. Wei,  X. Wang,  D. Schuurmans,  M. Bosma,  F. Xia, E. Chi, Q. V. Le, 
D. Zhou, et al. , “Chain -of-thought prompting elicits reasoning in large  
language models,” Advances in neural information processing systems , 
vol. 35, pp. 24824 –24837, 2022.  
 
APPENDIX : A 
This section  contains  the sample  questions  from  ORANBench.  
 
A. Easy 
 
 
C. Difficult  
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
B. Intermediate  
 
 
 
 
 
 
 
 
APPENDIX : B 
This section  contains  the sample  questions  from  srsRANBench.  
Question: Which of the following protocols is used for 
communication between the O -DU and O -RU in an O -RAN 
network?  
Options:  
FIAP  
NGAP  
FH-eCPRI  
HTTP2  
Answer:  3 
Question:  Which  O-RAN  component  handles  the radio 
frequency (RF) signal processing and transmission?  
Options:  
O-CU 
O-DU 
O-RU 
nFAPI  
Answer:  3 
Question:  Which  of the following  is NOT  a data type defined 
in the Y1  RAI Subscription API?  
Options:  
RaiSubscription  
NotificationCriteria  
RaiNotification  
ValidityPeriodAbsolute  
Answer:  4 
Question: What component is responsible for forwarding 
notifications received from the Near -RT RIC to the Energy 
Saving rApp?  
Options:  
SMO/Non -RT RIC Framework  
E2 Node  
O-RU 
Near -RT RIC 
Answer:  1 
Question: Which of the following is a CONDITIONAL 
MANDATORY  test scenario  in O-RAN,  relevant  only if the 
O-RU or O -DU supports C -Plane and U -Plane sessions over 
IP? 
Options:  
M-Plane  Verification  
UDP  Echo  Test 
O-RU Configuration  Test 
O-DU Hardware  Verification  
Answer:  2 
Question: In an O -RAN network utilizing TDM PON, what 
is the synchronization reference point for O -RUs located on 
the same PON port, or on different PON ports on the same 
PON card?  
Options:  
T-BC output  
Internal  point  in OLT,  different  PON  cards  
Internal  point  in OLT,  same  PON  port or different  PON 
ports on the same PON card  
T-GM network  port 
Answer:  3 
Question: How are AAL operations scheduled on the appli - 
cation data path in O -RAN?  
Options:  
Using  a synchronous  API set. 
Using  a burst  oriented  asynchronous  API set. 
Using  a real-time scheduling  algorithm.  
Using  a distributed  scheduling  mechanism.  
Answer:  2 
Question: What is the primary objective of the X2 U -Plane 
IOT test case related to node behavior?  
Options:  
To ensure the corresponding node’s procedures with 
eNB and en -gNB from different vendors function cor - 
rectly.  
To verify the compatibility of different vendors’ eNB 
and en -gNB with a specific vendor’s core network.  
To measure the latency and throughput of data trans - 
mission between the eNB and en -gNB.  
To test the security  features  of the X2 interface  between 
the eNB and en -gNB.  
Answer:  1 

Question: What is the default stream buffer size in samples 
for a radio session?  
Options:  
1) 614400  
2) 1024  
3) 4096  
4) 16384  
Answer:  1  
 
 
 
APPENDIX : C 
This section contains the samples from the RANSTRUCT dataset 
for O -RAN, and the srsRAN Project codebase.  
D. O-RAN 
1. Instruction: What power saving energy efficiency requirement 
does the transport equipment have in O -RAN?  
Answer:  The transport  equipment  in O-RAN  is designed  to sup- 
port power  saving  mechanisms,  which  means  it must  be capable 
of supporting  power -saving  cycles  for both the O-RU and O-DU 
devices. Additionally, the WDM transmission equipment must 
coordinate  its power -saving  modes  with the O-DU and O-RU to 
ensure smooth functionality during these cycles. Furthermore, 
the WDM transport must be able to support remote monitoring 
capabilities  related  to cellular  site operations,  which  is addressed 
in section 11.4 of ”Xhaul Transport Requirements” within the 
O-RAN  WG9  document.  To meet  these  requirements,  the power 
consumption of the system needs to remain under 80% of the 
power  supply  capacity  when  fully  loaded.  This ensures  efficient 
energy  usage  while  maintaining  optimal  performance  across 
all components. The overall goal is to optimize power savings 
without  compromising  the reliability  and stability  of the network 
infrastructure.  
 
2. Instruction: What is the expected range for frequency sync of 
the radio signal in O -RAN?  
Answer: The O -RAN standard specifies that the frequency 
accuracy required for the radio signal should meet a +/ -50ppb 
air interface  frequency  error  requirement  as per 3GPP  specifica - 
tions. This ensures that the O -RU meets the short -term average 
error requirement over 1ms duration. Therefore, the expected 
range for frequency sync of the radio signal in O -RAN is  
within this specified tolerance level of +/ -50ppb. However, it’s 
important  to note that the actual  implementation  details  might  vary based on specific design considerations and operational 
needs, so it’s always recommended to consult the latest official 
standards or documentation for precise guidelines.  
 
3. Instruction: What is the difference between eCPRI and CPRI 
traffic in co -existence of 5G NR and LTE radios in the same  
cell site?  
Answer:  In co-existence  scenarios  where  both eCPRI  and CPRI 
traffic need to be considered, eCPRI focuses primarily on high - 
speed data transfer for 5G NR users while CPRI handles the 
lower -bandwidth  data for LTE users.  This means  that when 
5G NR and LTE radios are deployed together within the same 
cell site, the efficiency of CPRI usage must be optimized to 
avoid unnecessary increases in front -haul bandwidth. CPRI is 
designed to handle low -bandwidth signals, which are typically 
associated  with LTE services,  ensuring  efficient  resource  utiliza - 
tion without significantly impacting the performance of 5G NR 
users. However, it’s important to balance these requirements to 
ensure smooth operation and optimal network performance.  
 
4. Instruction: What is the difference between eCPRI and CPRI 
traffic in co -existence of 5G NR and LTE radios in the same  
cell site?  
Answer:  In co-existence  scenarios  where  both eCPRI  and CPRI 
traffic need to be considered, eCPRI focuses primarily on high - 
speed data transfer for 5G NR users while CPRI handles the 
lower -bandwidth  data for LTE users.  This means  that when 
5G NR and LTE radios are deployed together within the same 
cell site, the efficiency of CPRI usage must be optimized to 
avoid unnecessary increases in front -haul bandwidth. CPRI is 
designed to handle low -bandwidth signals, which are typically 
associated  with LTE services,  ensuring  efficient  resource  utiliza - 
tion without significantly impacting the performance of 5G NR 
users. However, it’s important to balance these requirements to 
ensure smooth operation and optimal network performance.  
 
5. Instruction: What happens when the transport NSSI is com - 
pletely closed?  
Answer: When the transport NSSI is completely closed, only 
components connected to the transport NSSI can communicate 
with each other at the IP level. No communication occurs 
between  components  connected  to the transport  NSSI  and those 
shared between NSIs. This means that the transport component 
itself cannot communicate directly with any other component 
outside its own transport NSSI. It also prevents direct com - 
munication between components belonging to different NSIs. 
However, it allows for inter -component communication within 
the same transport NSSI, facilitating operations like service 
termination requests or O -RAN TN Manager coordination pro - 
cedures. The transport NSSI remains isolated from external 
systems, ensuring data integrity and preventing unauthorized 
access. This configuration helps maintain security and prevent 
potential conflicts during reset procedures, minimizing disrup - 
tion to critical services. To mitigate the impact on C/U -plane 
traffic, certain configurations must be disabled beforehand. 
When an O -RU connected through FHM needs to be reset, the 
reset operation does not affect other O -RUs, ensuring minimal 
interference with ongoing communications.  
E. srsRAN  
1. Instruction:  Generate  code  to add error  handling  for the 
std::strtol  function  in parse_args . 
Answer:  To handle  errors  returned  by std::strtol , we need 
to check if it returns a value other than SRSASN_SUCCESS . If 
so, we throw  an exception  with a meaningful  message  indicating 
the type of error.  
Question:  What  is the purpose  of the 
‘make  rlc entity  creation  message‘ function?  
Options:  
To create  a configuration  for an RLC  SRB  entity.  
To create  a configuration  for an RLC  DRB  entity.  
To create  a configuration  for an RLC  entity.  
To create  a configuration  for a MAC  entity.  
Answer:  1 
Question:  What  is the purpose  of the ‘create  ccch message‘ 
function?  
Options:  
To generate  a CCCH  message  with a given  RNTI.  
To generate  a MAC  PDU  with a given  RNTI.  
To generate a MAC UCI PDU from PUCCH informa - 
tion. 
To generate  a MAC  CRC  indication  message.  
Answer:  1 

Here’s  how you could  modify  the parse_args  function  to 
include  error  handling:  32 ‹→ snr_db > 90.0) { 
throw std:: 
   ‹→ invalid_argument  
1 void parse_args( int argc, char** argv) {  ‹→ ("Invalid  value 
2 int opt = 0;  ‹→ for snr_db: must 
3 while ((opt = getopt(argc,  argv, "n:k:  ‹→ be between -90 
 ‹→ e:i:s:o:"))  != -1) {  ‹→ and 90 dB."); 
4 switch (opt) { 33 } 
5 case ’e’: 34 break; 
6 E = (int)strtol(optarg,  35 case ’o’: 
 ‹→ nullptr,  10); 36 print_output  = (int)strtol 
7 if (E < 0 || E >= 256) {  ‹→ (optarg,  nullptr,  
8 throw std::  ‹→ 10); 
  ‹→ invalid_argument  37 if (print_output  < 0 || 
  ‹→ ("Invalid  value   ‹→ print_output  > 1) { 
  ‹→ for E: must be 38  throw std:: 
  ‹→ between 0 and   ‹→ invalid_argument  
  ‹→ 255.");   ‹→ ("Invalid  value 
9 }     ‹→ for print_output  
10 break;     ‹→ : must be either 
11 case ’k’:  ‹→ 0 or 1."); 
12 K = (int)strtol(optarg,  39 } 
 ‹→ nullptr,  10); 40 break; 
13 if (K < 0 || K >= 256) { 41 default: 
14 throw std:: 42 usage(argv[0]);  
 ‹→ invalid_argument  43 exit(-1); 
 ‹→ ("Invalid  value 44 } 
 ‹→ for K: must be 45 } 
 ‹→ between 0 and 46 } 
 
 
15 
16 
17 
18 ‹→ 255."); 
} 
break; 
case ’n’: 
nMax = (uint8_t)strtol(  
‹→ optarg, nullptr,  10)   
 
This modification  checks  the validity  of each parsed  in- 
teger parameter using strtol and throws an appropriate 
std::invalid_argument  exception if any of the values 
fall outside the specified ranges. This ensures that the program 
will terminate gracefully if invalid inputs are provided.  
 
 
19 
 
 
20 
 
 
 
 
 
 
 
 
 
21 
22 
23 
24 
 
 
 
 
 
 
 
 
 
25 
 
 
26 ‹→ ; 
if (nMax < 0 || nMax > 32) 
‹→ { 
throw std:: 
‹→ invalid_argument  
‹→ ("Invalid  value 
‹→ for nMax: must 
‹→ be between 0 and 
‹→ 32."); 
} 
break; 
case ’i’: 
bil = strtol(optarg,  
‹→ nullptr,  10) ? 
‹→ polar_code_ibil::  
‹→ present : 
‹→ polar_code_ibil::  
‹→ not_present;  
if (bil == 0 || bil == 4) 
‹→ { 
throw std:: 
‹→ invalid_argument  
‹→ ("Invalid  value 2. Instruction: Generate code for a function that simulates a 
failure  in the underlying  F1AP  interface  and tests the robustness 
of the DU’s handling of such failures.  
Answer:  To simulate  a failure  in the underlying  F1AP  interface 
and test the robustness of the DU’s handling of such failures, 
you can follow these steps:  
1. **Simulate Failure**: Create a function that generates a 
f1ap_message  representing  a failure scenario, such as time - 
out or other errors.  
2. **Handle Message**: Implement logic within the DU to 
handle this simulated failure message by checking its type and 
reacting accordingly.  
3. **Test Robustness**: Ensure the DU correctly identifies  
and responds to various types of failure messages, including 
timeouts, invalid requests, or other potential issues.  
4. **Assertions**:  Use assertions  to verify  that the DU correctly 
handles  different  types  of failure  responses,  ensuring  the system 
remains stable under varying conditions.  
5. **Validation**: Validate that the DU’s behavior aligns with 
expected outcomes based on the simulation results.  
Here is a simplified example of how this might look in C++ 
using the provided framework and technology:  
 ‹→ for bil: must be 1 void simulate_and_handle_failure(  
 ‹→ either 0 or 4." 
‹→ );  ‹→ du_processor_obj * du_processor_obj)  
‹→ { 
27 } 2 // Simulate  failure:  Timeout 
28 break; 3 f1ap_message  failureMsg  = 
29 case ’s’:  ‹→ generate_ue_context_setup_failure  
30 snr_db = strtof(optarg,   ‹→ ( 
 ‹→ nullptr);  4 int_to_gnb_cu_ue_f1ap_id(0),  
31 if (snr_db < -90.0 ||  ‹→ int_to_gnb_du_ue_f1ap_id(  

 
 
 
5 
6 
7 
 
 
8 
9 
 
 
10 
 
 
 
 
 
11 
12 
 
 
 
 
13 
 
This approach allows you to systematically test the DU’s 
resilience against various failure scenarios without relying on 
external figures or sections. The focus is on implementing and 
validating the DU’s internal mechanisms to ensure it functions 
reliably  even  when  faced  with unexpected  or simulated  failures. 
Please  note that the actual  implementation  would  require  careful 
consideration  of the specific  requirements  and constraints  of the 
DU and the F1AP  protocol.  The above  example  provides  a basic 
structure for testing failure scenarios.  
 
3. Instruction:  What  is the significance  of the byte_buffer  
and byte_buffer_chain  classes?  
Answer: The byte_buffer  class represents a sequence of 
bytes  in memory,  while  the byte_buffer_chain  class man - 
ages multiple  byte_buffers  together, allowing for efficient 
storage and manipulation of large amounts of data. It provides 
methods to append new buffers to its end and to perform 
operations  like copying  itself  or creating  new instances  based  on 
existing ones. These classes are fundamental for handling and 
manipulating binary data structures within the context of soft - 
ware  systems  designed  around  O-RAN  technology.  They  enable 
efficient management of buffer pools and help ensure proper 
allocation and deallocation of resources, which is crucial for 
applications running on the radio access network infrastructure. 
The design of these classes reflects the need for scalable and 
flexible memory management strategies in such environments.  
 
4. Instruction:  Describe  the functionality  of the 
create_pdu_with_sdu  function  and its parameters. 
Answer:  The create_pdu_with_sdu function  creates 
PDU  with SUD  segments  based  on the provided  input 
parameters.  It takes  in a list of vectors  containing  bytes  
representing the payload data ( pdu_list ), the resulting byte 
buffer  (sdu), the current  SN state (sn_state ), the size of 
each segment  (segment_size ), and the SN state  increment 
(sn_state_increment ). The function  iterates  over the 
pdu_list , creating and  handling PDUs according  to certain 
conditions  specified  by the user.  This includes  skipping  some 
PDUs  and pushing  others  into the RLC  queue  for processing. 
The function  also keeps  track  of any PDUs  that need  to be 
pushed back into the list at a later stage for further processing. 
The main  purpose  of this function  is to manage  the creation  and 
dispatching  of PDUs  and their corresponding  SUD  segments 
within the system.  
‹→ test_rgen::uniform_int<  
‹→ uint32_t>()));  
// Handle the failure message 
du_processor_obj ->handle_message(  
‹→ failureMsg);  
// Verify that the DU has handled the 
‹→ failure properly  
// This could involve checking  
‹→ specific  fields in the DU’s 
‹→ state or asserting  certain 
‹→ behaviors  
// Additional  checks could include 
‹→ verifying  responses  to other 
‹→ types of messages  
} 