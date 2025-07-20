# HKGAI-V1: Towards Regional Sovereign Large Language Model for Hong Kong

**Authors**: Sirui Han, Junqi Zhu, Ruiyuan Zhang, Yike Guo

**Published**: 2025-07-14 15:09:05

**PDF URL**: [http://arxiv.org/pdf/2507.11502v1](http://arxiv.org/pdf/2507.11502v1)

## Abstract
This paper presents the development of HKGAI-V1, a foundational sovereign
large language model (LLM), developed as part of an initiative to establish
value-aligned AI infrastructure specifically tailored for Hong Kong. Addressing
the region's unique multilingual environment (Cantonese, Mandarin, and
English), its distinct socio-legal context under the "one country, two systems"
framework, and specific local cultural and value considerations, the model is
built upon the DeepSeek architecture and systematically aligned with regional
norms through a multifaceted full parameter fine-tuning process. It is further
integrated with a retrieval-augmented generation (RAG) system to ensure timely
and factually grounded information access. The core contribution lies in the
design and implementation of a comprehensive, region-specific AI alignment and
safety framework, demonstrated through two key achievements: 1) The successful
development of HKGAI-V1 itself - which outper-forms general-purpose models in
handling Hong Kong-specific culturally sensitive queries, and embodies a
"governance-embedded" approach to digital sovereignty - empowers Hong Kong to
exercise control over AI applications in critical sectors including public
services, legal systems, and edu-cation. 2) The development of the proprietary
Adversarial HK Value Benchmark, a rigorous tool for evaluating model alignment
with local ethical and legal stand-ards under challenging conditions. By
documenting these achievements, the paper provides not only a technological
artifact but also a replicable blueprint for developing advanced, regionally
focused AI systems deeply rooted in their local identities.

## Full Text


<!-- PDF content starts -->

 
 
 
 
 
HKGAI -V1: Toward s Regional Sovereign  Large Language Model  for Hong Kong   
 
Sirui Han  a,c, Junqi Zhu c, Ruiyuan Zhang c, Yike Guo b,c,*
 
a Academy of Interdisciplinary Studies , The Hong Kong University of Science and Technology , Hong Kong SAR, China   
b Department of Computer Science and Engineering, The Hong Kong University of Science and Technology , Hong Kong SAR, China   
c Hong Kong Generative AI R&D Center , Hong Kong SAR, China  
 
A R T I C L E   I N F O  
Keywords:  
Sovereign AI  
Large Language Models  
AI Alignment  
Region -aligned Values  
Hong Kong  
 
 
 
 
 
 
 
 
 
 
 
 
 A B S T R A C T  
This paper presents the development of HKGAI -V1, a foundational sovereign large language 
model (LLM), developed as part of an initiative to establish value -aligned AI infrastructure 
specifically tailored for Hong Kong. Addressing the region’s unique multil ingual environment 
(Cantonese, Mandarin, and English), its distinct socio -legal context under the "one country, 
two systems" framework, and specific local cultural and value considerations, the model is 
built upon the DeepSeek  architecture and systematically aligned with regional norms through 
a multifaceted full parameter fine -tuning process. It is further integrated with a retrieval -aug-
mented generation (RAG) system to ensure timely and factually grounded information access. 
The core contribution lies in the design and implementation of a comprehensive, region -spe-
cific AI alignment and safety framework, demonstrated through two key achievements:  1) 
The successful development of HKGAI -V1 itself — which outperforms general -purp ose mod-
els in handling Hong Kong -specific culturally sensitive queries, and embodies a "governance -
embedded" approach to digital sovereignty — empowers Hong Kong to exercise control over 
AI applications in critical sectors including public services, legal systems, and education . 2) 
The development of the proprietary Adversarial HK Value Benchmark, a rigorous tool for 
evaluating model alignment with local ethical and legal standards under challenging condi-
tions. By documenting these achievements, the paper p rovides not only a technological artifact 
but also a replicable blueprint for developing advanced, regionally focused AI systems deeply 
rooted in their local identities.  
 
 
1. Introduction  
 
Sovereign AI [33] has gained prominence as large language 
models  (LLMs) [1][2][16] increasingly  influence information 
flows, societal norms, and decision -making pro-
cesses [32][58]. Sovereign AI  refers to the development of ar-
tificial intelligence ( AI) systems that are locally governed 
and customized to reflect the cultural, ethical, and legal 
frameworks of specific nations or regions [38].  In this con-
text, sovereign large language model refers to a large lan-
guage model that is independently constructed, trained, 
and deployed by a nation or region, operating on locally 
controlled computing and data infrastructure. It aims to en-
sure the securi ty of critical technologies, the mastery of data 
sovereignty, and alignment with local language, culture, 
and institutional requirements. Thus,  the development of a 
sovereign large language model  requires the value align-
ment [31][37] , a technology  that ensures an AI's goals and 
behaviors align with the values, ethics, and societal norms 
of its creators and users .  
 
* Corresponding Author . Email address: yikeguo@ust.hk  (Y. Guo).  Unlike models primarily developed by technology cor-
porations,  Sovereign AI initiatives , usually led by the gov-
ernment,  aim to encode the unique values and normative 
preferences of local communities into their systems [45]. 
Sovereign AI is  hence  seen as a means of asserting digital 
sovereignty  by enabling AI systems to operate in local lan-
guages, respect cultural norms, preserve identity, maintain 
trust, and comply with regional legal standards [43][28]. Alt-
hough the importance of such development is well under-
stood, there is few successful practice in building sovereign 
AI systems due to the combined technological, organiza-
tional and governance complexities.  
Hong Kong's distinctive geopolitical standing and the 
adaptable socio -legal environment provided by the "one 
country, two systems" framework [48] create a unique set-
ting for the advancement of sovereign Artificial Intelligence 
(AI). This framework uniquely blends Eastern and Western 
legal, cultural, and institutional characteristics while grant-
ing significant autonomy to local governing bodies [39][46]. 
Furthermore, Hong Kong's multilingual nature, encompass-
ing Cantonese, English, and Mandarin [61][54][64], 

 
2 
 necessitates AI systems with the capacity to manage linguis-
tic diversity and dialectal variations.  In contrast , off -the-
shelf AI models, such as GPT -4[1] or DeepSeek [17], often fall 
short of adequately addressing Hong Kong's specific socio-
logical and legal demands, including sophisticated linguis-
tic processing and culturally embedded priorities. On the 
other hand, AI is increasingly regarded as a critical and 
ubiquitous MetaCity  infrastructure1 that facilitat es access to 
knowledge , enhanc es communication, and enabl es sustain-
ability for cosmopolitans such as Hong Kong [44][52][57]. 
The concerns about cultural erosion [37] become serious , as 
communities may adopt behaviors influenced by LLMs de-
veloped in foreign contexts with differing political and eth-
ical frameworks [21][31][35] . Therefore, sovereign AI sys-
tems developed within this context with the capability of 
navigat ing the intricate intersections of language, law, and 
culture, potentially establishing Hong Kong  is not only the 
necessity  but also offers  a global benchmark in this domain . 
To address existing challenges and leverage Hong 
Kong's socio -legal flexibility, we introduce HKGAI -V1, the 
first sovereign large language model specifically tailored to 
Hong Kong's unique linguistic and socio -cultural context 
and the needs of the global overseas Chinese diaspora. This 
685-billion -parameter model utilizes a multilingual corpus, 
Retrieval Augmented Generation (RAG )[34], reinforcement 
learning [40][41]. Guided by local values  and governance 
rules [3][19], and preference amplification [8], HKGAI -V1 
outperforms general models on local tasks, supports Hong 
Kong's AI governance and digital sovereignty goals, fosters 
a local AI ecosystem, and offers a blueprint for global smart 
city development , with the following key contributions:  
• Pioneering a sovereign AI language model specifically 
tailored to Hong Kong's unique linguistic, cultural, and 
legal landscape.  
• Advancing the field of AI specialization by demonstrat-
ing superior performance on local tasks compared to 
general -purpose models.  
• Establishing a framework for AI governance and digital 
sovereignty aligned with local policies and values, re-
ducing reliance on external AI providers . 
• Building our own Adversarial HK Value Benchmark, a 
proprietary tool designed to rigorously test and quan-
tify a model's alignment with local ethical and legal 
standards under challenging conditions . 
 
2. Related work  
 
Sovereign AI development, beyond local data utilization, 
heavily relies on post -training and value alignment technol-
ogies. The alignment of large AI models with human inten-
tions and values has become a significant area of re-
search [28]. Supervised Fine -tuning (SFT) and Reinforce-
ment Learning from Human Feedback (RLHF) [41] have 
emerged as prominent practical methodologies for achieving this alignment. SFT trains models on human 
demonstrations to guide desired behaviors, while RLHF 
uses a reward model based on human preferences and opti-
mizes behavior through reinforcement learning.  
Recent studies refine these methods in two key direc-
tions to better align with the 3H standards: Helpfulness, 
Harmlessness  & Honesty [3]. One line of work aims to refine 
the pipeline of post -training strategies, addressing issues 
such as the challenges in reward model optimization [41], 
enhancing efficiency and scalability [24][63], navigating the 
trade -off between helpfulness and harmlessness [26][27], 
and improving performance in multi -turn interactions. An-
other line of work focuses on extending alignment frame-
work beyond language -only settings to multimodal scenar-
ios[29][38][56] , addressing both the understanding and gen-
eration of multimodal content [38][49]. 
Furthermore, the alignment of AI systems must trans-
cend mere task -oriented intentions to encompass broader 
moral and ethical considerations, ensuring adherence to 
value alignment principles [12]. Current research in this area 
can be broadly categorized into two main areas: (1) ethical 
and social values, which focus on instilling appropriate 
moral principles in AI systems to enable them to distinguish 
between right and wrong and to minimize biases intro-
duced during the training process [38][51][55], and (2) cross -
cultural and life -long value alignment, which explores the 
context -dependent nature of ethics within different social 
frameworks. This includes research on methods for facilitat-
ing multi -agent interactions and cooperation [9], as well as 
the development of computational solutions for aggregating 
preferences across diverse populations [4].  
A crucial component in applying reinforcement learn-
ing to the alignment problem is the reward model. This 
model acts as a proxy for human intentions by assigning 
scores to AI -generated responses. Initial models used binary 
human feedback [40] but recent research focuses on richer 
feedback information. Safe-RLHF [27] decomposes rewards 
into helpfulness and harmlessness, adapted for multimodal 
tasks [10][25] . Aligner [24] also introduced correction learn-
ing for better alignment  by gaining experience from the past 
experiences , and sequence -to-sequence reward model-
ing[63] offered a more granular and effective approach. 
Align -Anything [29] demonstrated the value of learning 
from information -rich feedback like natural language in 
multimodal settings.  
In a related advancement, Agentic RAG [47] represents 
a significant evolution in AI, addressing the limitations of 
LLMs and traditional RAG by integrating autonomous AI 
agents into the retrieval -enhanced data -centric RAG pipe-
line. These agents employ agentic design patterns like re-
flection [42], planning [23] , tool use [36], and multi -agent col-
laboration [53][59] to dynamically manage retrieval, refine 
understanding, and adapt workflows, enhancing flexibility 
and context -awareness. Building on the progression from 
Naïve [14], Advanced [60], Modular [15][30], and Graph 

 
3 
 RAG [11], Agentic RAG introduces dynamic decision -mak-
ing and workflow optimization through various architec-
tures , despite the ongoing efforts to tackle the challenges of 
coordination  and scalability , among many others [47]. 
Indeed , the development of sovereign AI systems, as 
will be exemplified by HKGAI -V1, heavily relies on post -
training alignment techniques to ensure helpfulness, harm-
lessness  and honesty. Value a lignment methods, including 
advanced reward modeling, are vital for ensuring sovereign 
AI systems to  align with human values in respective social 
regions . The progression to Agentic RAG further enhances 
HKGAI -V1's capabilities through autonomous information 
processing, though coordination and scalability remain key 
challenges.  
 
3. System architecture   
 
The HKGAI V1 system architecture is structured around 
three interconnected components: (1) design principles, (2) 
the agentic system workflow, and (3) core layer compo-
nents, each contributing to its robust functionality.  
Design Principles. As Fig. 1 illustrates, the system's de-
sign principles prioritize scalability to handle large compu-
tational demands, multi -layered security to ensure data pro-
tection and compliance, user -centric design to enhance ac-
cessibility, transparency to foster trust, and alignment with 
Hon g Kong's cultural, legal, and ethical frameworks.  
Agentic Workflow. HKGAI -V1 integrates a planner for 
task formulation, an executor for precise implementation, 
and a communicator for seamless interaction.  
Core Layer Components . At its core, HKGAI -V1 is 
built on four foundational layers that integrate technical so-
phistication with robust governance : (1) The trust and gov-
ernance layer ensures compliance, accountability, and resil-
ience through policy enforcement, monitoring, and testing; 
(2) The platform and service layer enables seamless task ex-
ecution with secure APIs, orchestration, and adaptable frameworks; (3) The model and algorithm layer hosts the 
HKGAI LLM family, specialized domain tools, and value 
alignment via reinforcement learning; and (4) The data and 
server layer manages diverse data sources securely  to en-
hance privacy, robustness, and bias mitigation  of the system . 
 
4. Value alignment of HKGAI -V1 system  
 
Value alignment  aims to align LLMs  with human inten-
tions. It enables AI to learn from human feedback. A prom-
inent method in this domain is RLHF . This approach repre-
sents human intentions as preferences, which are then con-
verted into a learnable signal —reward . Subsequently, rein-
forcement learning algorithms are utilized to optimize the 
AI's behavior based on this reward. This section elaborates 
on the respective training processes of  HKGAI -V1. 
 
4.1. Reinforcement Learning from Human Feedback  
 
A widely adopted approach for modeling human prefer-
ences is to employ a preference predictor grounded in the 
Bradley –Terry (BT) model. Given a pair of answers (𝑦1,𝑦2)  
generated from a prompt 𝑥, BT model indicates that the hu-
man preference distribution 𝑝∗ can be expressed based on  
the underlying human reward function 𝑟∗(𝑦,𝑥) as: 
𝑝∗(𝑦1≻𝑦2|𝑥)=exp(𝑟∗(𝑦1,𝑥))
exp(𝑟∗(𝑦1,𝑥))+exp(𝑟∗(𝑦2,𝑥)), 
Hence, given a human preference dataset 𝒟=
{(𝑥(𝑖),𝑦𝑤(𝑖),𝑦𝑙(𝑖))}i=1N, the training objective for a reward model 
r𝜙(𝑦,𝑥) parameterized by 𝜙 is defined as:  
 
ℒ(𝜙,𝒟)=−𝐸(𝑥,𝑦𝑤,𝑦𝑙)∼𝔻[𝑙𝑜𝑔 𝜎(𝑟𝜙(𝑦𝑤,𝑥)−𝑟𝜙(𝑦𝑙,𝑥))] 
For a given prompt 𝑥, the HKGAI -v1 generates a response, 
 
Fig. 1 HKGAI -V1 System Architecture . The architecture illustrates the design principles (left), agentic system workflow (middle), and core layer compo-
nents (right) of the HKGAI V1 system, emphasizing scalability, security, user -centric design, transparency, and value alignment  as a sovereign AI system . 


 
4 
 which is then scored by the reward model. The policy pa-
rameters 𝜃 are updated to maximize this reward. To prevent 
the policy from deviating too much from the original pre -
trained model and maintain coherence, a Kullback -Leibler 
(KL) divergence penalty is added to the optimization objec-
tive:  
𝑚𝑎𝑥𝜙𝐸 [𝑅𝜃(𝑥,𝑦) − 𝛽∗𝐾𝐿(𝜋𝜃 ||𝜋𝜃𝑏𝑎𝑠𝑒)] 
where 𝛽 is a fixed hyper -parameter. The final optimization 
objective of RLHF is:  
𝑚𝑎𝑥𝜃𝐸𝑥 ,𝑦∼𝜋𝜃(⋅|𝑥)[𝑅𝜃(𝑥,𝑦) − 𝛽∗𝐾𝐿(𝜋𝜃(⋅|𝑥) ||𝜋𝜃𝑏𝑎𝑠𝑒(⋅|𝑥))]. 
 
4.2. Beyond Bradley -Terry, Learning from Language Feedback  
 
In this section, we introduce learning from language 
feedback (LLF) [29]. It utilizes language feedback to opti-
mize responses, synthesizing preference data which can en-
hance the performance of RLHF. We demonstrate how to 
practically implement LLF in the HKGAI -V1 system, in-
cluding two main stages, feedback modeling and self-im-
proving . 
Feedback Modeling.  The training process utilizes a da-
taset  𝒟={(𝑥𝑖,𝑦𝑖,𝑐𝑖)}i=1N, where 𝑁 is the size of dataset, 𝑥𝑖 
denotes the prompt, 𝑦𝑖 represents the response, and 𝑐𝑖 is the 
corresponding feedback. Let 𝑃Φ(𝑐𝑖∣∣𝑥𝑖,𝑦𝑖) denote the prob-
ability of the target sequence 𝑐𝑖 given the input sequence 
(𝑥𝑖,𝑦𝑖) and the model parameters Φ, the training objective of 
the feedback model can be expressed by  minimizing  the 
cross -entropy loss:  
𝐿𝛷=−𝐸(𝑥𝑖,𝑦𝑖,𝑐𝑖)∼𝐷[𝑙𝑜𝑔 𝑃𝛷(𝑐𝑖∣∣𝑥𝑖,𝑦𝑖)], 
Self-evolving LLM s.  Self-evolving LLM leverage lan-
guage feedback to improve response quality, forming pur-
pose -specific preference pairs. They are not static artefacts  
but  systems designed to iteratively re -train themselves  by 
turning their own outputs —and critiques of those out-
puts —into fresh training signals. The self -evolving Large 
Language Model (LLM) leverages feedback loops to en-
hance response quality by iteratively generating initial re-
sponses, collecting feedback fro m specialized models, and 
refining outputs based on this feedback. This  dynamic, iter-
ative process, supported by advanced multi -phase algo-
rithms and safeguarded by policy -based templates and dy-
namic guardrails, helps align foundational models more 
closely with user preferences and societal values. It reduces 
common issues suc h as redundancy and hallucination, mak-
ing the AI system increasingly precise, reliable, and aligned 
with its intended purpose.  
 
4.3. Weak to Strong Generalization, amplifies human feedback  
 
As AI systems approach human -level capabilities and 
begin performing tasks that are difficult for humans to understand, it becomes increasingly challenging to provide 
continuous and reliable feedback to ensure that these sys-
tems remain aligned with human intentions and values. 
This raises significant concerns regarding the problem of 
Superalignment : how can we supervise systems that are more 
powerful and intelligent than humans are? [28] This issue is 
even more challenging when the alignment is performed in 
a complex social and culture context.  
Weak -to-strong generalization is a training paradigm 
that utilizes supervisory signals from weaker models to en-
hance the performance of stronger models [24]. During the 
development of HKGAI -V1, we adopted a correction -based 
framework to amplify human feedback and facilitate the 
creation of high -quality, value -aligned synthetic data, fur-
ther enhancing the model's region -aligned capabilities. Spe-
cifically, we c onstruct a local Hong Kong -specific Q -A da-
taset focusing on values, mathematics, code reasoning, and 
science engineering problems. Local annotators are then en-
listed to provide corrections to the original responses, re-
sulting in a Q -A-C dataset. Designing such a dataset is the  
core to the value alignment. There are crucial design issues 
such as the number of Q -A pairs , the coverage of the topics, 
the quality and consistency of annotation, the demographic 
profile of annotators.  These issues are common to d esigning 
surveys in social science , Based on this Q -A-C dataset, built 
with a close collaboration with our social science colleagues, 
we train a value correction model, HKValue -Aligner, using 
the correction paradigm from Aligner [24]. The training ob-
jective is defined as follows:  
min
φℒHKValue -Aligner (φ,ℳ)=−𝐸𝕄[logμφ(𝑦𝑐 ∣∣𝑦𝑜,𝑥)],   
where 𝑀 denotes the correction dataset, and 𝑦𝑐 and 𝑦𝑜 rep-
resent the answers before and after correction, respectively.  
The model performs secondary corrections on base 
model responses, enhancing detail, safety, and honesty (3H 
standards) while aligning with Hong Kong values. Using 
HKValue -Aligner [], synthetic data is generated by correct-
ing pre -existing responses, forming a value preference da-
taset. This dataset supports value -based RLHF, effectively 
building on prior advancements.  
We found that this paradigm brings several benefits: 1) 
HKValue -Aligner efficiently generates value -aligned pref-
erence datasets for Hong Kong, enabling repeatable im-
provements. 2) It amplifies human feedback by building on 
HKGAI -V1’s robust capabilities, refining fine -grained er-
rors in tasks like reasoning and code generation. This en-
sures granular improvements while aligning with local HK 
values . 
 
4.4. Performance of HKGAI -V1 System  
 
We evaluate HKGAI -V1 on four benchmarks —
MMLU [18][50], AGI -Eval [62], Flames [22], and Beaver -zh-
hk[20]—selected for their coverage of general knowledge, 

 
5 
 reasoning skills, real -time contextualization, and regional 
value alignment. All tests run on identical hardware with 
each model’s default hyperparameters.  
 
Table 1. Performance of HKGAI -V1 Model . “Avg.” indicates the micro -av-
erage accuracy. The highest score in each column is in bold.  
Benchmark  HKGAI -V1 DeepSeek -R1 
MMLU  90.44  90.8 
AGI -Eval  88.69  87.64  
Flame s 68.06  30.12  
Beaver -zh-hk 88.95  70.41  
Avg.  84.04  69.74  
 
Beaver -zh-hk is a safety benchmark designed for Hong 
Kong’s socio -cultural and legal context. It covers 29 scenar-
ios, including 14 general risks and 15 region -specific haz-
ards. The benchmark uses 2,508 samples. We derive evalua-
tion metrics with a four -tier assessment methodology and 
GPT -4o scoring.  As Table 1 illustrates,  HKGAI -V1 achieves 
a Harmless Score of 88.95 compared to DeepSeek -R1’s 70.41, 
demonstrating that our Hong Kong –focused RLHF objec-
tives and policy constraints effectively embed local values 
while maintaining robust safety.  
On the MMLU benchmark, which measures the broad 
knowledge of LLMs, HKGAI -V1 scores 90.44 compared to 
DeepSeek -R1’s 90.80, demonstrating that alignment efforts 
do not compromise core capabilities. On AGI -Eval, which 
evaluates complex reasoning, HKGAI -V1 ach ieves 88.69 
versus DeepSeek -R1’s 87.64, highlighting the effectiveness 
of combining direct preference optimization with human 
feedback to enhance decision -making in nuanced scenarios.  
HKGAI -V1 achieves a significant improvement on the 
Flame benchmark, scoring 68.06 compared to DeepSeek -
R1’s 30.12. Flame tasks, requiring real -time retrieval and dy-
namic context integration, benefit from HKGAI -V1’s re-
trieval -augmented generation and search -enhancement 
modules. Post -training evaluations used methods like mix-
ture of experts (MoE) assessments, red -team adversarial 
testing, and automated safety benchmarks to address vul-
nerabilities. Iterative improvements were driven by contin-
uous feedback thro ugh periodic reviews, real -time inputs, 
and public engagement.  
Overall, these results demonstrate that HKGAI -V1’s 
alignment framework not only preserves general model 
performance but also significantly enhances value -sensitive 
and context -aware tasks. The strong performance on the 
Beaver -zh-hk benchmark further underscores the im-
portance of developing evaluation standards that reflect re-
gional norms in AI systems, ensuring better alignment with 
local knowledge and values . 
 
4.5. Proprietary Evaluation Framework for HKGAI -V1 
 We further evaluate HKGAI -V1 on three proprietary 
benchmarks —HKMMLU  (Hong Kong Massive Multitask 
Language Understanding) [7], SafelawBench [65] and 
NaVAB [66]. These benchmark s are specifically designed to 
assess a model's grasp of information relevant to Hong 
Kong. The evaluation focused on the model's "zero -shot" 
performance, meaning it was tested on the benchmark da-
taset s without any prior fine -tuning on the specific data. 
This approach effectively gauges the model's pre -existing 
knowledge and its ability to generalize to new, Hong Kong -
related prompts . 
Based on the updated Table 3, which presents the zero -
shot performance of HKGAI -V1 on the HKMMLU bench-
mark, we can now provide a revised comparison with other 
listed models. The table details the average accuracy (Avg.) 
and performance across STEM, Social Sciences (Soc. Sci), 
Humanitie s, and Other categories, specifically for Tradi-
tional Chinese (TC).  
An empirical evaluation of various large language 
models on the HKMMLU benchmark reveals that the 
HKGAI -V1 model establishes a new state -of-the-art (SOTA) 
performance. The zero -shot evaluation, conducted in Tradi-
tional Chinese, assesses a model's intrinsic  knowledge and 
reasoning capabilities without task -specific fine -tuning. In 
this rigorous context, HKGAI -V1 not only achieves the high-
est aggregate score but also demonstrates superiority across 
all evaluated sub -domains, significantly outperforming 
contem porary proprietary and open -source models.  
In terms of overall performance, HKGAI -V1 obtained a 
mean accuracy of 81.4%. This result positions it as the defin-
itive leader within the tested cohort, surpassing the next -
best model, DeepSeek -V3, which scored 76.6%, by a sub-
stantial margin of 4.8 percent age points. Furthermore, 
HKGAI -V1 exhibits a considerable performance advantage 
over widely recognized systems such as GPT -4o (70.5%). Its 
score represents a significant positive deviation from the co-
hort's mean accuracy of 58.0%, underscoring its exceptio nal 
capabilities on this specialized evaluation suite.  
A more granular analysis of the results indicates that 
the model's high performance is not concentrated in a single 
area but is remarkably consistent across all subject domains. 
HKGAI -V1 achieved the highest score in Humanities 
(84.6%), STEM (80.4%), Socia l Sciences (80.4%), and the 
"Other" category (80.2%). Its particularly high accuracy in 
Humanities is noteworthy, as this domain often contains 
complex questions with deep linguistic and cultural nu-
ances. The model's uniform dominance suggests a robust 
and well -balanced architecture and training methodology, 
resulting in a comprehensive knowledge base rather than 
narrow expertise.  
The superior zero -shot performance of HKGAI -V1 
strongly implies that its pre -training corpus is extensively 
populated with high -quality Traditional Chinese text and 
data possessing deep contextual relevance to Hong Kong. 
This domain -specific data concentra tion is the most 

 
6 
 probable factor for its performance lead over general -pur-
pose models, whose vast but more diffuse training sets may 
lack the required density of culturally and linguistically spe-
cific information. Consequently, these results highlight the 
critical role of curated, domain -specific data in developing 
models that can achieve state -of-the-art performance on re-
gionalized and culturally -specific benchmarks. The propri-
etary HKMMLU benchmark further emphasizes the com-
mitment to evaluating the model's understanding of Hong 
Kong -specific information. HKGAI -V1's success serves as a 
compelling case study for the efficacy of this specialized ap-
proach in language model development.  
 
 
Fig. 2 Model performance comparison on HKMMLU and SafeLawBench.  
 
The performance of HKGAI -V1 on the SafeLawBench  
(Table 3) benchmark demonstrates exceptional results, 
achieving the highest average accuracy of 80.1% among all 
evaluated models. This highlights its effectiveness in ad-
dressing safety -related tasks across diverse legal risk cate-
gories. In the Critical Personal Safety  (CPS) category, HKGAI -
V1 scores 82.4%, slightly trailing behind DeepSeek -V3 
(82.9%) and GPT -4o (83.2%), while in Property & Living Se-
curity (PLS), it achieves 78.7%, securing the second -best po-
sition after DeepSeek -V3 (79.2%). For Fundamental Rights  
(FR), HKGAI -V1 attains a score of 79.0%, marginally lower 
than GPT -4o (79.3%). In Welfare Protection  (WP), HKGAI -V1 
performs strongly with a score of 79.9%, outperforming 
GPT -4o (78.8%).  
Compared to closed -source models such as GPT -4o, 
which achieve slightly higher accuracy in specific categories, 
HKGAI -V1’s consistently strong performance across all risk 
levels underscores its robust alignment with legal safety 
standards. It also outperfo rms all open -source models, in-
cluding DeepSeek -V3 (79.7%), the leading open -source com-
petitor. HKGAI -V1’s strengths lie in its consistency and bal-
anced performance across all categories, with no significant 
weaknesses. In conclusion,  as illustrated in Fig. 2, the development and evaluation of the HKGAI -V1 system un-
derscore a comprehensive approach to value alignment in 
large language models. In addition, HKGAI -V1’s results po-
sition it as a leading model for legal safety evaluation, show-
casing its comprehensive and reliable capabilities. Further-
more, the strong performance of HKGAI -V1 across a range 
of benchmarks, including those specifically designed to as-
sess regional value alignment and knowledge, demon-
strates the effectiveness of these methodologies.  
 
Table 2 The Value Alignment Evaluation Results on both Quoted and 
Official Statement sets  of NaVAB . Different depth of color of the cells in-
dicates that the value inside is higher. The MC and AJ notations refer to 
Multiple -Choise and Answer -Judgment evaluation method, respectively.  
 
 
On the performance of HKGAI -V1 on NaVAB , HKGAI -
V1 demonstrates state -of-the-art performance in aligning 
with values  of various countries (Table 2), as evidenced by 
its high scores across both the Quoted Statements and Offi-
cial Statements datasets. The model achieves consistently 
strong results in the Multiple -Choice (MC) evaluation 
method, with scores exceeding 0.92 across all nations, in-
cluding Chi na, the US, the UK, France, and Germany. Its 
performance in the Answer -Judgment (AJ) method, while 
slightly lower, remains competitive, particularly in nations 
like China (0.514) and the UK (0.509). Compared to other 
models, HKGAI -V1 outperforms many basel ine and in-
struct -tuned models, such as Llama3.1 -8b and Qwen2.5 -7b, 
and is competitive with closed -source models like GPT -4, 
often exceeding them in MC evaluations. Notably, HKGAI -
V1 excels in English -speaking nations and China, likely ben-
efiting from align ment with these linguistic and cultural 
contexts. However, its performance in Germany, particu-
larly in the AJ method, is slightly lower, indicating potential 
limitations in handling German -specific values. The model's 
alignment with both Quoted and Officia l Statements da-
tasets suggests it effectively captures individual and institu-
tional perspectives within each nation. While HKGAI -V1 is 
adapted  at handling binary -choice scenarios, as reflected in 
its high MC scores, its relatively lower AJ scores highlight  


 
7 
 the need for improvements in generating nuanced, value -
aligned free -form responses. Overall, HKGAI -V1 showcases 
robust multilingual and multicultural alignment, making it 
a leading model for benchmarking multi -national value 
alignment in large language mod els. These findings collec-
tively affirm the potential of HKGAI -V1 as a sovereign AI 
system that not only achieves high performance but also 
aligns with the unique cultural, legal, and ethical context of 
Hong Kong.  
 
5. Retrieval -enhanced HKGAI -V1 Framework  
 
5.1. Modular RAG framework  
 
The HKGAI -V1 RAG framework is built on a modular 
architecture optimized for retrieval -augmented generation 
workflows, ensuring scalability, adaptability, and align-
ment with Hong Kong’s legal, cultural, and ethical stand-
ards. It leverages proprietary and ex ternal knowledge 
sources, short -term memory, and a tool -use framework to 
handle diverse prompts. Workflow -based moderation en-
sures quality, governance, and domain -specific alignment, 
while the answer generation stage delivers structured, vali-
dated response s that meet compliance and user needs.   
As depicted in Fig. 3, the modular RAG framework pro-
cesses input queries using components like an intent classifier, query enhancer, and retriever. In the retrieval -en-
hanced QA stage, it integrates proprietary and external 
knowledge sources, including database servers, parsed data, 
and tools like Google and Bing. A short -term memory mod-
ule and tool -use framew ork enable handling diverse 
prompts, from simple queries to complex, multi -turn con-
versations. A workflow -based moderation stage ensures 
quality and governance through input validation, output 
moderation, safety alignment, and multimodal verification, 
tailored to domains like legal and financial services. Finally, 
the answer generation stage consolidates moderated results 
into structured, validated, and compliant responses.  
 
5.2 Proprietary Evaluation Framework for HKGAI -V1-RAG  
 
The HKGAI -V1 evaluation framework employs a com-
prehensive, multi -dimensional approach to assess model 
performance and its RAG mechanism. It systematically 
evaluates capabilities across diverse applications to refine 
the model iteratively and ensure effecti ve real -world de-
ployment, while validating the HKGAI -V1 theoretical archi-
tecture. Key evaluation dimensions include alignment with 
Hong Kong's socio -ethical values, accuracy in language in-
struction, fluency in Cantonese, refusal of sensitive ques-
tions, and  logical reasoning in analytical tasks.  
 
Table 3 Zero -shot performance of HKGAI -V1 on HKMMLU  and c omparison of model accuracy (%) on SafeLawBench by risk level.  “Soc. Sci.” stands for 
Social Sciences. “CPS” stands for Critical Personal Safety, “PLS” for Property & Living Security, “FR” for Fundamental Rights, and “WP“ for We lfare 
Protection.  The highest score in each column is in bold.  
Model s HKMMLU  SafeLawBench  
 Avg.  
(Macro -aver-
age Accuracy ) STEM  Soc. Sci.  Humanities  Other  Avg.  
(Micro-aver-
age Accuracy ) CPS PLS FR WP 
HKGAI -V1 81.4 80.4 80.4 84.6 80.2 80.0 80.0 79.5 81.0 78.2 
DeepSeek -V3 76.6 77.2 75.1 78.8 75.1 79.7 82.9 79.2 78.3 79.1 
GPT-4o 70.5 75.0 70.4 68.7 67.9 80.3 83.2 79.9 79.3 78.8 
Gemma -2-2B-IT 41.4 30.6 39.9 47.8 47.1 58.7 63.2 57.1 57.2 57.6 
Gemma -2-27B-IT 57.1 55.6 55.3 59.9 57.4 70.5 76.0 68.6 68.7 69.0 
GLM -4-9B-Chat  48.4 42.7 47.3 53.6 49.9 61.2 64.7 60.0 59.8 60.9 
Llama -3-8B-Instruct  39.5 32.6 38.1 44.5 42.7 68.4 71.1 68.3 66.7 68.5 
Llama -3-70B-Instruct  58.7 60.5 59.4 59.1 55.9 76.1 79.9 74.6 75.1 74.8 
Llama -3.1-8B-Instruct  44.1 33.3 41.2 53.3 48.7 65.3 68.8 64.5 63.8 64.3 
Llama -3.1-70B-Instruct  58.9 59.6 57.6 61.2 57.3 75.2 78.5 74.4 74.0 74.5 
Mistral -Small -Instruct  44.6 38.9 42.0 49.6 47.9 68.8 72.9 67.9 67.0 68.3 
Mistral -Large -Instruct  60.0 63.8 58.2 59.6 58.4 77.2 81.2 75.3 76.5 76.2 
Qwen2.5 -3B-Instruct  49.9 42.5 47.8 54.4 54.7 62.4 66.3 60.7 61.3 61.9 
Qwen2.5 -7B-Instruct  56.9 55.9 54.0 60.1 57.4 70.9 74.9 69.4 69.5 70.7 
Qwen2.5 -14B-Instruct  62.2 63.8 61.5 62.9 60.5 74.9 78.8 73.2 73.4 75.0 
Qwen2.5 -72B-Instruct  69.0 71.9 67.9 70.2 65.9 77.6 81.4 76.5 76.3 76.5 
Avg.  58.2 56.8 56.9 60.8 58.1 72.2 75.7 71.1 71.1 71.4 

 
8 
  
Fig. 3. Modular architecture of the HKGAI -V1 RAG framework . First, the system refines and retrieves information using intent classifiers, enhancers, and 
retrieval techniques like BM25.  Second, it integrates knowledge sources, including databases and tools like Google, supporte d by short -term memory and 
APIs for di verse queries.  Third, RAG  -based moderation ensures quality through validation, moderation, and safety alignment for domain -specific needs. 
Finally, it generates validated, compliant responses to meet user expectations.  
 
Fig. 4 HKGAI -V1 Evaluation Framework and Workflow.  The HKGAI -V1 Evaluation Framework assesses AI systems through defined criteria, auto-
mated tools, and human review. Outputs are evaluated for honesty, harmlessness, and helpfulness. Results guide system optimiz ation, ensuring ethical 
compliance, robust performance, and continuous improvement.


 
9 
 The methodology combines quantitative and qualita-
tive assessments. Automated tools evaluate linguistic qual-
ity, while human evaluators assess nuanced aspects like so-
cio-ethical alignment and Cantonese fluency. Additionally, 
feedback from governmental and p ublic sector users cap-
tures practical needs, enabling continuous optimization 
through a user -centric feedback loop. As depicted in Fig. 4, 
the evaluation workflow consists of four stages: evaluation 
set construction, model output generation, rigorous execu-
tion, and results analysis for targeted RAG optimization. 
This structured process ensures data -driven insights for im-
proving the model's performance and alignment with soci-
etal values, emphasizing robustness and ethical grounding.  
 
5.3. Performance of HKGAI -V1-RAG  
 
The HKGAI -V1-RAG evaluation framework assesses the 
system's performance across diverse dimensions, ensuring 
alignment with local cultural, linguistic, and ethical 
stand tards. As Table 4 illustrates, t he framework includes 
key evaluation dimensions, specific indicators, and detailed 
evaluation methods to ensure robust and comprehensive as-
sessments. The focus is on adaptability, linguistic accuracy, 
sensitivity to context, and logical reasoning capabilit ies. 
 
Table 4. Assessment Framework . The table illustrates the dimensions, in-
dicators, and methods of HKGAI -V1’s Assessment Framework.  
Evaluation 
Dimension  Evaluation Indica-
tors Evaluation Methods and Explana-
tions  
Value 
Alignment  Alignment with 
Hong Kong's main-
stream social values 
(e.g., law, fairness, 
diversity, etc.)  Manual annotation to assess 
whether the model's responses 
align with Hong Kong's main-
stream values without crossing eth-
ical boundaries.  
Instruction 
Following  1. Language com-
prehension  
2. Natural language 
command  Evaluate the model's and RAG's re-
sponses in two languages (Simpli-
fied Chinese, Traditional Chinese, 
English, Cantonese) for comprehen-
sion and adaptability.  
Sensitive 
Question 
Handling  Response rejection 
rate Analyse the model's mechanisms 
for handling politically or ethically 
sensitive questions. Evaluate the re-
jection ratio for such queries.  
 
Adversarial Value Bench . The HKGAI -V1 adversarial 
value benchmark employs 300 human -crafted sensitive 
questions with opposing "safe" and "unsafe" viewpoints to 
rigorously evaluate model alignment. Inspired by adversar-
ial testing, the methodology involves: (1) constructing a di-
verse question set aligned with Hong Kong -specific content 
labels; (2) eliciting model responses to each question; (3) 
conducting human evaluation based on predefined "Safe" 
(ethical, neutral, compliant with PRC and HK laws) and 
"Unsafe" (controversial, risky,  harmful) criteria; (4) statistically analyzing the proportions and biases of "Safe" 
and "Unsafe" responses across various question categories 
(Hong Kong sensitive issues, instruction attacks, typical 
safety scenarios); and (5) providing feedback to refine train-
ing strategies for  improved safety and ethical alignment. 
This structured process aims to identify and mitigate unde-
sirable biases, ensuring the model's responses are neutral, 
reasonable, and ethically sound, adhering to legal and social 
norms.  
 
Table 5. Adversarial HK Value Bench Results . The table illustrates the per-
formances of HKGAI V1, Kimi, and ChatGPT on different adversarial value 
benchmark.  
Module  Metric  HKGAI -V1 
(%) Kimi  
(%) ChatGPT 
(%) 
Hong Kong 
Sensitive  Safe 79 53 10.7 
Refusal  template  4 42 0.6 
Unsafe  17 5 88.7 
Instruction 
Attack  Safe 68 65 63 
Refusal  template  15.5 29 29 
Unsafe  16.5 6 8 
Typical 
Safety Sce-
narios  Safe 82 83 91 
Refusal  template  18 17 8 
Unsafe  0 0 1 
 
The adversarial HK value benchmark evaluated the 
safety and alignment of HKGAI V1 chat, Kimi, and 
ChatGPT across three distinct modules. As Table 5 illus-
trates, in the Hong Kong Sensitive Issues category, HKGAI 
V1 chat demonstrated the strongest performance, achieving 
79% safe responses. However, this still indicates a need for 
further refinement to achieve  complete safety. Kimi exhib-
ited a significant reliance on template -based safe responses 
(42%), with a 53% rate of purely safe answers and 5% unsafe. 
In contrast, ChatGPT displayed a notably higher proportion 
of unsafe responses (88.7%) in this sensitive d omain. The In-
struction Attack module revealed a more closely clustered 
performance among the models. HKGAI V1 chat registered 
the highest percentage of unsafe responses at 16.5%, while 
Kimi (6%) and ChatGPT (8%) showed comparable, lower 
unsafe respo nse rates. The Typical Safety Scenarios module 
highlighted robust safety across all platforms. ChatGPT 
achieved the highest safe response rate of 91%, followed by 
Kimi at 83% and HKGAI V1 chat at 82%. The incidence of 
unsafe responses was minimal across th is module.  
Overall, the benchmarking results indicate that while 
all three models demonstrate a degree of safety awareness, 
HKGAI V1 chat shows the most promising results in navi-
gating Hong Kong -specific sensitive topics, albeit with room 
for improvement. Kimi's stra tegy appears to lean heavily on 
pre-defined safe templates, while ChatGPT exhibited a 
higher vulnerability to generating unsafe content, particu-
larly concerning Hong Kong -related sensitive issues. In 
standard safety scenarios, all models performed strongly . 
Instruction and Language Following . This proprietary 
benchmark assesses HKGAI -V1’s ability to respond in the  

 
10 
 Table 6. HKGAI -V1 Multilingual Benchmarking Results  
Category  Version  Following Rate  (%) Overall  Simplified Chinese  Traditional Chinese  English  Cantonese (Oral ) 
HKGAI -V1  with  Search  100%  100%  100%  94.50%  100%  
without   Search  100%  100%  100%  97.80%  100%  
HKGAI -V1-
Thinking   with  Search  100%  100%  100%  98.90%  100%  
without   Search  100%  100%  94% 81.10%  ~98%  
 
Table 7. Comparative Benchmarking Results of HKGAI -V1 and Other Models on Handling Culturally and Politically Sensitive Queries  
Metric  Definition & Explanation  HKGAI V1 & RAG  HKGAI V1  DeepSeek V3  
Refusal Rate (Sensitive 
Political Queries)  Percentage of sensitive political questions the 
model explicitly refused to answer.  0% 13% 56% 
Positive/Neutral Re-
sponses  Percentage of factually accurate, culturally sensi-
tive, and unbiased responses.  100%  87% 44% 
Template -Based "Red -
Leaning" Responses  Percentage of responses using predefined ideo-
logical templates  0% 13% N/A  
Safety Warning/Error 
Messages  Frequency of moderation warnings or errors dur-
ing inference . N/A  N/A  Present  
Ability to Directly Ad-
dress Sensitive Topics  The model's ability to respond directly to sensi-
tive cultural and political topics without evasion.  Yes Partially  No 
Avoidance of Hard 
Red Lines  Whether the model avoids violating safety, ethi-
cal, or legal guidelines.  Yes No N/A  
same language consistently as the user’s input —a key capa-
bility distinct from general instruction -following tasks. As 
Table 6 illustrates, the consistent 100% accuracy achieved 
for Simplified Chinese, Traditional Chinese, and English 
across different platforms and connectivity conditions un-
derscores the model's robust ability to accurately interpret 
and respond in the language of the user's input. This high 
level of performance s ignifies a strong underlying linguistic 
understanding and processing capability for these written 
languages. While Cantonese oral processing also demon-
strates high proficiency, the minor variations observed with 
and without search suggest potential areas f or further opti-
mization.  Given Hong Kong’s unique linguistic land-
scape —where Cantonese, Mandarin, and English coexist 
and frequently intermingle —precise language matching is 
critical for ensuring user trust, accessibility, and effective 
deployment across diverse community contexts . 
Sensitive Question Handling . To rigorously assess 
HKGAI -V1’s capability to appropriately manage culturally 
and politically sensitive inquiries, we established a compre-
hensive benchmarking methodology. This process began by 
carefully constructing a query test set with 100 questions. 
These questions were selected following an extensive anal-
ysis of sensitive topics commonly appearing in Hong Kong's 
media and online discussions. Categories covered include 
historical events, political ideologies, social movements, de-
bates on cultural iden tity, geopolitical issues, freedom of ex-
pression, and legal considerations. Each query was crafted 
clearly to effectively evaluate the model’s inherent response 
tendencies  without biases . 
During the evaluation, questions were systematically 
posed to HKGAI -V1 and responses recorded along with de-
tailed metadata. The evaluation criteria included response directness, factual accuracy , unbiased judgements , cultural , 
ethical and political appropriateness, safety,  transparency, 
and suitability of refusal. Responses were analyzed quanti-
tatively (frequency counts) and qualitatively (detailed con-
tent analysis). Comparative analysis with other leading AI 
models provided context for HKGAI -V1's strengths , limita-
tions, and areas for improvement.   
As presented in Table 7, HKGAI -V1, particularly with 
RAG, demonstrates strong capabilities in safely addressing  
sensitive queries specific to Hong Kong. Its ability to con-
sistently generate balanced and contextually appropriate re-
sponses without relying on template -based ideological 
statements or triggering external moderation warnings is a 
notable advantage. Nonet heless, ongoing improvements are 
essential to refine these capabilities further. The high lan-
guage consistency across multiple written languages and 
commendable proficiency in oral Cantonese additionally 
highlight HKGAI -V1’s robust multilingual foundation.  
Overall, the combination of HKGAI -V1’s modular RAG 
architecture and comprehensive evaluation strategy signifi-
cantly advances the development of culturally attuned and 
ethically sound AI systems, supporting  the broader goal of 
sovereign AI tailored to regional standards and values.  
 
6. Discussion  
 
6.1.  Critical Reflections on Value Alignment and Performance  
 
Our primary goal was to create a sovereign model that 
not only understands Hong Kong's unique context but also 
embodies its values. The results demonstar the achieve-
ments we made. However, it also presents  a nuanced and, 
in some cases, challenging picture.  

 
11 
 Our approach yielded clear successes in areas critical 
to regional usability. On the proprietary HKMMLU 
benchmark, HKGAI -V1 established a new state -of-the-art 
performance, significantly outperforming the powerful 
DeepSeek -V3 model ( 81.4%  vs. 76.6% , as shown in  Table 3). 
This success is further bolstered by the model's exceptional 
language -following ability ( Table 6 ), where it achieved 
near -perfect consistency in responding in the user's lan-
guage (especially in Simplified/Traditional Chinese and 
English), a vital feature for Hong Kong's multilingual envi-
ronment. These results affirm that our specialization strat-
egy e ffectively instilled deep local knowledge and practical 
usability.  
However, in other specialized domains, the picture is 
more nuanced. For instance, on the SafeLawBench, while 
HKGAI -V1 achieved a highly competitive average accuracy 
of 80.0% , surpassing the capabilities of Deepseek V3,  it did 
not establish a decisive lead over top -tier models like GPT -
4o (80.3% , as shown in  Table 3). This indicates that while 
our model is robust in legal safety contexts, achieving supe-
riority in such a universally complex and well -researched 
domain requires even more intensive specialization.  
Furthermore, our safety evaluations revealed a critical 
trade -off. As shown in the Adversarial HK Value Bench ( Ta-
ble 5), HKGAI -V1 demonstrated superior alignment on 
Hong Kong -specific sensitive queries ( 79%  safe responses 
vs. ChatGPT's  10.7% ). Conversely, it exhibited a higher vul-
nerability to general instruction  attacks, with an unsafe re-
sponse rate of  16.5% . This suggests that our alignment pro-
cess, while successful in encoding specific regional guard-
rails, may have created new attack surfaces. This highlights 
the classic tension between compliance and robustness —a 
central challenge in AI alignment that requires further re-
search.  
Finally, we must acknowledge the inherent subjectivity 
in "aligning to Hong Kong values." Our process  relied on a 
set of local annotators, but their  views may not represent 
the full spectrum of opinions within such a diverse and 
dynamic society. This limitation underscores that value 
alignment is not a one -time technical fix but an ongoing 
process of societal dialogue and engagement with a 
cross -disci plinary knowledge and skills.  
 
6.2.  AI -empowered engineering ecosystem dimensions  
 
HKGAI -V1 demonstrates the potential of Sovereign AI 
for urban management, highlighting the crucial need for 
aligned values and regulatory  frameworks. Its development 
underscores the importance  of interdisciplinary collabora-
tion and the creation of federated AI ecosystems to build 
sustainable, localized, and sovereign AI solutions.  
HKGAI -V1 also exemplifies localized sovereign AI de-
velopment by integrating  Hong Kong's legal, cultural, and 
linguistic context. The creation of custom benchmarks like HKMMLU and SafeLaw -Bench addresses limitations in 
general evaluation, demonstrating its ability to understand 
local nuances and safety protocols, serving as a model for 
regions with complex socio -legal environments. Establish-
ing a federated AI infrastructure , such as across the Greater 
Bay Area, is vital for sovereign AI, enabling shared data, 
standardized APIs, and domestically controlled technolo-
gies. This fosters  regional cooperation in governance, ethical 
AI, and public trust, while protecting national technological 
sovereignty, necessitating continued investment in re-
search, infrastructure, and interdisciplinary partnerships for 
value -aligned, region -specific AI . 
HKGAI -V1's deployment in Hong Kong demonstrates 
AI's potential for societal progress across governance, ser-
vices, and education, yet aligning it with the region's unique 
cultural, linguistic, and political environment is a critical 
challenge. Early development hig hlighted the complexities 
of embedding local values, necessitating a balance between 
localized alignment and technical scalability for sovereign 
AI. A major obstacle lies in the reliance on large pre -training 
datasets with diverse biases, where fin e-tuning on local data 
offers uncertain effectiveness and techniques like synthetic 
data and domain adaptation present their own challenges. 
Addressing this requires research into quantifying and mit-
igating implicit value encodings and reconciling diverse 
data without compromising local values. Language further 
complicates alignment, as HKGAI -V1 must operate in Can-
tonese, Mandarin, and English, each with distinct cultural 
values. Achieving precise multilingual alignment, capturing 
nuances, and ensuring cros s-lingual consistency is crucial 
for legitimacy and accurate representation of Hong Kong's 
values, demanding further research in sociolinguistics and 
culturally sensitive AI development.  
 
6.3.  Roadmap for HKGAI -V2: A Multi -Pillar Approach to Sov-
ereign AI  
 
While HKGAI -V1 was a foundational experiment in re-
gional alignment, achieving true AI sovereignty requires a 
more holistic and ambitious strategy. The roadmap for the 
future HKGAI -V2 is therefore structured around strength-
ening five core pillars of Soverei gn AI.  
Data Sovereignty : The effectiveness of any regional 
model is contingent on the data it is trained on. HKGAI -V1 
relied heavily on public datasets and a limited, newly cre-
ated  local corpus which still cannot  reflect a complete land-
scape of Hong Kong data sovereignty.  For HKGAI -V2, we 
will move beyond this by establishing secure, privacy -pre-
serving data partnerships with key Hong Kong institutions, 
including government agencies, academic archives, legal 
bodies, and healthcare providers. This will involve develop-
ing a federate d data framework that allows for model train-
ing on sensitive local data without it ever leaving its source, 
thus respecting Hong Kong's stringent data privacy 

 
12 
 ordinances (e.g., PDPO) and ensuring the data used for 
training is truly representative of the region.  
Compute Sovereignty : A region cannot be sovereign 
over its AI if it is entirely dependent on external infrastruc-
ture. With the Hong Kong government support, HKGAI -V1 
was trained with a locally managed  GPU cluster at Hong 
Kong Generative AI Research and Development Center 
(HKGAI). However, its capability is still quite limited. The 
long -term vision is to establish or secure access to a locally 
managed , heterogenous GPU cluster dedicated to Sovereign 
AI development in Hong Kong. This will not only guarantee 
operational independence but also enhance security and re-
duce reliance on international supply chains, ensuring that 
the computational resources underpinning our digital infra-
structure are under regional control.  
Model Sovereignty : True model sovereignty means a 
strategy and mechanism to fully control the model develop-
ment and evolution. HKGAI -V1 base model is a full param-
eter fine -tuned version of Deepseek. Although the core ca-
pabilities and inherent biases of HKGAI -V1 are predete r-
mined by its original training, our work has significantly en-
hanced the model with full-stack technology in further de-
velopment  of the model into a sovereign AI system. In the 
future, we will pursue a dual -track approach in the devel-
opmen t. First , we will develop smaller, highly specialized  
models for critical sectors (e.g., a legal model trained exclu-
sively on Hong Kong case law; a finance model versed in 
local regulations) that are fully auditable and controllable. 
Second , we will leverage our sovereign dataset and compute 
infrastructure to evolve the current model into a further lo-
calized foundational model. Such a model  will be used to 
construct a set of high -quality  training data for future train-
ing.  Ultimately, an " HKGAI -V1" LLM series  built from the 
ground up with hi gh quality local generated/collected data 
would be a definitive assertion of model sovereignty.  
Governance Sovereignty : The rules that govern AI 
must be as sovereign as AI itself.  The alignment of HKGAI -
V1 was guided by internal guidelines and general principles.  
With the support from Hong Kong government, we have al-
ready published a guideline of generative AI to the Hong 
Kong public.  We will formalize this into a robust and trans-
parent governance framework . This includes establishing 
an independent ethics and oversight board comprising di-
verse local stakeholders. Technically, we will advance be-
yond simple RLHF by implementing Constitutional AI. A 
"constitution" will be drafted based on Hong Kong's specific 
legal and ethical frameworks (e.g., the Basic Law, estab-
lished legal precedents, and societal norms), which will 
guide the model's behavior in a m ore systematic and audita-
ble manner.  
Service Sovereignty:  The final pillar  is ensuring Sover-
eign AI benefits the region through locally controlled ser-
vices. An AI model’s value lies in its application, and 
HKGAI -V1 has prioritized government use, with around 
20,000 officers across nearly all departments already using its applications. Efforts will center on building a sovereign 
application ecosystem by developing public sector solutions 
that run on local infrastructure via secure APIs . Examples 
include a Cantonese -first government service assistant, AI -
powered educational tools f or local curricula, and platforms 
that support smart city initiatives, aligning AI services with 
regional priorities and supporting the local economy and 
society.  
By pursuing this multi -pillar strategy, we aim for 
HKGAI -V2 to be not just a better model , but a true embodi-
ment of Hong Kong's digital sovereignty.  
 
7. Conclusion  
 
The development of HKGAI -V1 represents a crucial and pi-
oneering step in Hong Kong's pursuit of Sovereign AI. 
While the model itself is a foundational artifact, our most 
significant contribution lies in the systematic effort to in-
stantiate regional values a nd contexts  within an advanced 
AI system. The deliberate development of bespoke evalua-
tion frameworks, including HKMMLU for local knowledge 
and the SafeLawBench, NaVAB , and Adversarial HK Value 
Bench for safety, underscores a core principle: the meaning-
ful assessment of a region -specific AI necessitates region -
specific metrics. This commitment to creating our own eval-
uative standards is a key advantage and a tangible move to-
wards genuine technological self -determination.  
Our value alignment strategies produced concrete and 
promising results. HKGAI -V1 demonstrated a clear superi-
ority in safely navigating Hong Kong -specific sensitive top-
ics compared to leading global models, validating our tar-
geted fine -tuning approach. Howe ver, our work also illumi-
nated the complex challenges inherent in this endeavor. 
These include performance trade -offs on general knowledge 
benchmarks and the emergence of new vulnerabilities, 
which serve as critical signposts for future research.  
By confronting these difficulties directly, the HKGAI -
V1 development provides more than just a piece of technol-
ogy or an AI system; it offers a valuable and practical blue-
print for other regions aiming to cultivate AI capabilities 
that are deeply integrated with their own  cultural, linguis-
tic, and legal identities. It is a tangible exploration of how to 
balance global technological trends with local values, and 
how to build not just a model, but an entire ecosystem of 
data, evaluation, applications and gove rnance.  
Ultimately, HKGAI -V1 lays the groundwork for a fu-
ture where AI development is not a monolithic, one -size-
fits-all process. It reinforces the notion that sovereignty in 
the digital age is built through dedicated, context -aware en-
gineering and steadfast comm itment to aligning  LLM  with 
the human communities it is designed to serve. The path 
forward is challenging, but this project establishes a firm 
and principled starting point for Hong Kong's journey in 
shaping its own AI future to become an AI endowered in-
ternational city.   

 
13 
 Acknowledgments  
 
This research was funded by the InnoHK  funding for Hong 
Kong Generative AI Research and Development Center, 
Hong Kong SAR, Theme -based Research Scheme grant (No.  
T45-205/21 -N) and HKUST Start -up Fund (R9911).  
 
References  
 
[1] Achiam, J., Adler, S., Agarwal, S., Ahmad, L., Akkaya, 
I., Aleman, F. L., ... & McGrew, B. (2023). Gpt -4 technical 
report.  arXiv preprint arXiv:2303.08774 . 
[2] Anthropic. Claude 3. 
https://www.anthropic.com/news/claude -3-family, 
2024a.  
[3] Askell, A., Bai, Y., Chen, A., Drain, D., Ganguli, D., 
Henighan, T., ... & Kaplan, J. (2021). A general language 
assistant as a laboratory for alignment.  arXiv preprint 
arXiv:2112.00861 . 
[4] Bakker, M., Chadwick, M., Sheahan, H., Tessler, M., 
Campbell -Gillingham, L., Balaguer, J., ... & 
Summerfield, C. (2022). Fine -tuning language models to 
find agreement among humans with diverse 
preferences.  Advances in Neural Information Processing 
Systems , 35, 38176 -38189.  
[5] Brandt, F., Conitzer, V., Endriss, U., Lang, J., & 
Procaccia, A. D. (Eds.). (2016).  Handbook of computational 
social choice . Cambridge University Press.  
[6] Brown, D. S., Schneider, J., Dragan, A., & Niekum, S. (2
021, July). Value alignment verification. In  International 
Conference on Machine Learning  (pp. 1105 -1115). PMLR.  
[7] Cao, C., Zhu, Z., Zhu, J., Lu, G., Peng, S., Dai, J., ... & 
Guo, Y. (2025). Measuring Hong Kong Massive Multi -
Task Language Understanding.  arXiv preprint 
arXiv:2505.02177 . 
[8] Christiano, P., Shlegeris, B., & Amodei, D. (2018). 
Supervising strong learners by amplifying weak 
experts.  arXiv preprint arXiv:1810.08575 . 
[9] Dafoe, A., Hughes, E., Bachrach, Y., Collins, T., McKee, 
K. R., Leibo, J. Z., ... & Graepel, T. (2020). Open problems 
in cooperative AI.  arXiv preprint arXiv:2012.08630 . 
[10] Dai, J., Chen, T., Wang, X., Yang, Z., Chen, T., Ji, J., & 
Yang, Y. (2024). Safesora: Towards safety alignment of 
text2video generation via a human preference 
dataset.  Advances in Neural Information Processing 
Systems , 37, 17161 -17214..  
[11] Edge, D., Trinh, H., Cheng, N., Bradley, J., Chao, A., 
Mody, A., ... & Larson, J. (2024). From local to global: A 
graph rag approach to query -focused 
summarization.  arXiv preprint arXiv:2404.16130 . 
[12] Gabriel, I. (2020). Artificial intelligence, values, and 
alignment.  Minds and machines , 30(3), 411 -437. [13] Gabriel, I., & Ghazavi, V. (2022). The challenge of value
 alignment. In  The Oxford handbook of digital ethics . Oxfor
d: Oxford University Press.  
[14] Gao, Y., Xiong, Y., Gao, X., Jia, K., Pan, J., Bi, Y., ... & 
Wang, H. (2023). Retrieval -augmented generation for 
large language models: A survey.  arXiv preprint 
arXiv:2312.10997 , 2(1). 
[15] Gao, Z., Cao, Y., Wang, H., Ke, A., Feng, Y., Xie, X., & 
Zhou, S. K. (2025). FRAG: A Flexible Modular 
Framework for Retrieval -Augmented Generation based 
on Knowledge Graphs.  arXiv preprint arXiv:2501.09957 . 
[16] Gemini Team, Georgiev, P., Lei, V. I., Burnell, R., Bai, L., 
Gulati, A., ... & Batsaikhan, B. O. (2024). Gemini 1.5: 
Unlocking multimodal understanding across millions 
of tokens of context.  arXiv preprint arXiv:2403.05530 . 
[17] Guo, D., Yang, D., Zhang, H., Song, J., Zhang, R., Xu, R., 
... & He, Y. (2025). Deepseek -r1: Incentivizing reasoning 
capability in llms via reinforcement learning.  arXiv 
preprint arXiv:2501.12948 . 
[18] Hendrycks, D., Burns, C., Basart, S., Zou, A., Mazeika, 
M., Song, D., & Steinhardt, J. (2020). Measuring massive 
multitask language understanding.  arXiv preprint 
arXiv:2009.03300 . 
[19] Hong Kong SAR Government. Digital Policy Office 
Releases Hong Kong Generative Artificial Intelligence 
Technical and Application Guideline. Info.gov.hk , 202 5, 
www.info.gov.hk/gia/general/202504/15/P20250415
00227.htm. Accessed 26 May 2025.  
[20] https://github.com/PKU -Alignment/Beaver -zh-hk 
[21] Hu, W., Li, H., Jing, H., Hu, Q., Zeng, Z., Han, S., ... & 
Song, Y. (2025). Context Reasoner: Incentivizing 
Reasoning Capability for Contextualized Privacy and 
Safety Compliance via Reinforcement Learning.  arXiv 
preprint arXiv:2505.14585 . 
[22] Huang, K., Liu, X., Guo, Q., Sun, T., Sun, J., Wang, Y., ... 
& Lin, D. (2023). Flames: Benchmarking value 
alignment of llms in chinese.  arXiv preprint 
arXiv:2311.06899 . 
[23] Huang, X., Liu, W., Chen, X., Wang, X., Wang, H., Lian, 
D., ... & Chen, E. (2024). Understanding the planning of 
LLM agents: A survey.  arXiv preprint arXiv:2402.02716 . 
[24] Ji, J., Chen, B., Lou, H., Hong, D., Zhang, B., Pan, X., ... 
& Yang, Y. (2024). Aligner: Efficient alignment by 
learning to correct.  Advances in Neural Information 
Processing Systems , 37, 90853 -90890.  
[25] Ji, J., Chen, X., Pan, R., Zhu, H., Zhang, C., Li, J., ... & 
Yang, Y. (2025). Safe rlhf -v: Safe reinforcement learning 
from human feedback in multimodal large language 
models.  arXiv preprint arXiv:2503.17682 . 
[26] Ji, J., Hong, D., Zhang, B., Chen, B., Dai, J., Zheng, B., ... 
& Yang, Y. (2024). Pku -saferlhf: Towards multi -level 
safety alignment for llms with human preference.  arXiv 
preprint arXiv:2406.15513 .. 
[27] Ji, J., Liu, M., Dai, J., Pan, X., Zhang, C., Bian, C., ... & 
Yang, Y. (2023). Beavertails: Towards improved safety 

 
14 
 alignment of llm via a human -preference 
dataset.  Advances in Neural Information Processing 
Systems , 36, 24678 -24704.  
[28] Ji, J., Qiu, T., Chen, B., Zhang, B., Lou, H., Wang, K., ... 
& Gao, W. (2023). Ai alignment: A comprehensive 
survey.  arXiv preprint arXiv:2310.19852 . 
[29] Ji, J., Zhou, J., Lou, H., Chen, B., Hong, D., Wang, X., ... 
& Yang, Y. (2024). Align anything: Training all -modality 
models to follow instructions with language 
feedback.  arXiv preprint arXiv:2412.15838 . 
[30] Jin, J., Zhu, Y., Dou, Z., Dong, G., Yang, X., Zhang, C., ... 
& Wen, J. R. (2025, May). Flashrag: A modular toolkit 
for efficient retrieval -augmented generation research. In 
Companion Proceedings of the ACM on Web Conference 
2025  (pp. 737 -740).  
[31] Ju, C., Shi, W., Liu, C., Ji, J., Zhang, J., Zhang, R., ... & G
uo, Y. (2025). Benchmarking Multi -National Value Alig
nment for Large Language Models.  arXiv preprint arXiv
:2504.12911 . 
[32] Klissarov, M., Hjelm, D., Toshev, A., & Mazoure, B. (20
24). On the Modeling Capabilities of Large Language M
odels for Sequential Decision Making.  arXiv preprint ar
Xiv:2410.05656 . 
[33] Letort, Brian, and Kadri Linask -Goode. What is 
sovereign AI and why is it growing in importance? . 
Digitalrealty.com , 2025, 
www.digitalrealty.com/resources/articles/what -is-
sovereign -ai. Accessed 26 May 2025.   
[34] Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, 
V., Goyal, N., ... & Kiela, D. (2020). Retrieval -augmented 
generation for knowledge -intensive nlp tasks.  Advances 
in neural information processing systems , 33, 9459 -9474.  
[35] Li, H., Hu, W., Jing, H., Chen, Y., Hu, Q., Han, S., ... & S
ong, Y. (2025). Privaci -bench: Evaluating privacy with c
ontextual integrity and legal compliance.  arXiv preprint 
arXiv:2502.17041 . 
[36] Li, X. (2025, January). A Review of Prominent 
Paradigms for LLM -Based Agents: Tool Use, Planning 
(Including RAG), and Feedback Learning. 
In Proceedings of the 31st International Conference on 
Computational Linguistics  (pp. 9760 -9779).  
[37] Masoud, R. I., Liu, Z., Ferianc, M., Treleaven, P., & Rod
rigues, M. (2023). Cultural Alignment in Large Langua
ge Models: An Explanatory Analysis Based on Hofsted
e's Cultural Dimensions.  arXiv preprint arXiv:2309.1234
2. 
[38] Mügge, D. (2024). EU AI sovereignty: For whom, to 
what end, and to whose benefit?.  Journal of European 
Public Policy , 31(8), 2200 -2225.  
[39] Nicolson, K. (2016).  Landscapes lost and found: appreciatin
g Hong Kong’s heritage cultural landscapes . Hong Kong U
niversity Press.  
[40] Ouyang, L., Wu, J., Jiang, X., Almeida, D., Wainwright, 
C., Mishkin, P., ... & Lowe, R. (2022). Training language models to follow instructions with human 
feedback.  Advances in neural information processing 
systems , 35, 27730 -27744.  
[41] Rafailov, R., Sharma, A., Mitchell, E., Manning, C. D., 
Ermon, S., & Finn, C. (2023). Direct preference 
optimization: Your language model is secretly a reward 
model.  Advances in Neural Information Processing 
Systems , 36, 53728 -53741.  
[42] Renze, M., & Guven, E. (2024). Self -reflection in llm 
agents: Effects on problem -solving performance.  arXiv 
preprint arXiv:2405.06682 . 
[43] Russell, S. (2019).  Human compatible: AI and the problem of 
control . Penguin Uk.  
[44] Scardovi, C., & Scardovi, C. (2021). From smart to meta
 cities.  Sustainable Cities: Big Data, Artificial Intelligence a
nd the Rise of Green,“Cy -phy” Cities , 1-20. 
[45] Scassa, T. (2023). Sovereignity and the governance of 
artificial intelligence.  UCLA L. Rev. Discourse , 71, 214.  
[46] Sin, W. M. (2006). Law, autonomy and politics: The cha
nging socio -political roles of law in postcolonial Hong 
Kong.  International Journal of the Sociology of Law , 34(1), 
64-83. 
[47] Singh, A., Ehtesham, A., Kumar, S., & Khoei, T. T. 
(2025). Agentic Retrieval -Augmented Generation: A 
Survey on Agentic RAG.  arXiv preprint 
arXiv:2501.09136 . 
[48] So, A. Y. (2011). “One country, two systems” and Hong
 Kong -China national integration: A crisis -transformati
on perspective.  Journal of Contemporary Asia , 41(1), 99 -1
16. 
[49] Wallace, B., Dang, M., Rafailov, R., Zhou, L., Lou, A., 
Purushwalkam, S., ... & Naik, N. (2024). Diffusion 
model alignment using direct preference optimization. 
In Proceedings of the IEEE/CVF Conference on Computer 
Vision and Pattern Recognition  (pp. 8228 -8238).  
[50] Wang, Y., Ma, X., Zhang, G., Ni, Y., Chandra, A., Guo, 
S., ... & Chen, W. (2024, June). Mmlu -pro: A more robust 
and challenging multi -task language understanding 
benchmark. In  The Thirty -eight Conference on Neural 
Information Processing Systems Datasets and Benchmarks 
Track . 
[51] Winfield, A. F., Michael, K., Pitt, J., & Evers, V. (2019). 
Machine ethics: The design and governance of ethical AI 
and autonomous systems [scanning the 
issue].  Proceedings of the IEEE , 107(3), 509 -517. 
[52] Xu, Y., Wang, F., An, Z., Wang, Q., & Zhang, Z. (2023). 
Artificial intelligence for science —bridging data to wis
dom.  The Innovation , 4(6). 
[53] Yang, Y., Chai, H., Shao, S., Song, Y., Qi, S., Rui, R., & 
Zhang, W. (2025). Agentnet: Decentralized evolutionary 
coordination for llm -based multi -agent systems.  arXiv 
preprint arXiv:2504.00587 . 
[54] Yi Mak, H., & Lee, T. (2021, December). Low -resource n
mt: A case study on the written and spoken languages i

 
15 
 n hong kong. In Proceedings of the 2021 5th International 
Conference on Natural Language Processing and Informatio
n Retrieval  (pp. 81 -87). 
[55] Yu, H., Shen, Z., Miao, C., Leung, C., Lesser, V. R., & 
Yang, Q. (2018). Building ethics into artificial 
intelligence.  arXiv preprint arXiv:1812.02953 . 
[56] Yu, T., Yao, Y., Zhang, H., He, T., Han, Y., Cui, G., ... & 
Chua, T. S. (2024). Rlhf -v: Towards trustworthy mllms 
via behavior alignment from fine -grained correctional 
human feedback. In  Proceedings of the IEEE/CVF 
Conference on Computer Vision and Pattern 
Recognition  (pp. 13807 -13816).  
[57] Zhang, Y., Lin, Y., Zheng, G., Liu, Y., Sukiennik, N., Xu
, F., ... & Guo, H. (2025). MetaCity: Data -driven sustain
able development of complex cities.  The Innovation . 
[58] Zhang, Y., Liu, H., Jiang, F., Luo, W., & Zhang, K. (2024
). Building Decision Making Models Through Languag
e Model Regime. arXiv preprint arXiv:2408.06087 . 
[59] Zhang, Y., Sun, R., Chen, Y., Pfister, T., Zhang, R., & 
Arik, S. (2024). Chain of agents: Large language models 
collaborating on long -context tasks.  Advances in Neural 
Information Processing Systems , 37, 132208 -132237.  
[60] Zheng, X., Weng, Z., Lyu, Y., Jiang, L., Xue, H., Ren, B., 
... & Hu, X. (2025). Retrieval augmented generation and 
understanding in vision: A survey and new 
outlook.  arXiv preprint arXiv:2503.18016 . [61] Zhong, T., Yang, Z., Liu, Z., Zhang, R., Liu, Y., Sun, H., 
... & Liu, T. (2024). Opportunities and challenges of larg
e language models for low -resource languages in huma
nities research.  arXiv preprint arXiv:2412.04497 . 
[62] Zhong, W., Cui, R., Guo, Y., Liang, Y., Lu, S., Wang, Y., ... 
& Duan, N. (2023). Agieval: A human -centric 
benchmark for evaluating foundation models.  arXiv 
preprint arXiv:2304.06364 . 
[63] Zhou, J., Ji, J., Dai, J., & Yang, Y. (2025, April). Sequence 
to sequence reward modeling: Improving rlhf by 
language feedback. In  Proceedings of the AAAI Conference 
on Artificial Intelligence  (Vol. 39, No. 26, pp. 27765 -
27773).  
[64] Zhu, S., Xu, S., Sun, H., Pan, L., Cui, M., Du, J., ... & Xio
ng, D. (2024). Multilingual Large Language Models: A 
Systematic Survey.  arXiv preprint arXiv:2411.11072 . 
[65] Cao, C., Zhu, H., Ji, J., Sun, Q., Zhu, Z., Wu, Y., ... & Gu
o, Y. (2025). SafeLawBench: Towards Safe Alignment of
 Large Language Models.  arXiv preprint arXiv:2506.0663
6. 
[66] Shi, W., Ju, C., Liu, C., Ji, J., Zhang, J., Zhang, R., ... & G
uo, Y. (2025). Benchmarking Multi -National Value Alig
nment for Large Language Models.  arXiv preprint arXiv
:2504.12911 . 
 
 