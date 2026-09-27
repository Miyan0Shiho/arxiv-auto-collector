# Taramandal-GPT: Enhancing Astrodynamics Problem-Solving with Knowledge Retrieval and Structured Thinking

**Authors**: Akhil Sharma, Jatin Gupta, Ali Imam Abidi

**Published**: 2026-09-21 08:11:43

**PDF URL**: [https://arxiv.org/pdf/2609.24246v1](https://arxiv.org/pdf/2609.24246v1)

## Abstract
Large language models (LLMs) have shown remarkable progress in natural language understanding, yet their effectiveness in specialized fields like astronomy and astrodynamics remains limited due to challenges in multi-step reasoning, symbolic manipulation, and domain-specific terminology. To address this, we present Taramandal-GPT (Constellation-GPT), a domain-adapted framework built on the Qwen3-8b backbone, enhanced with a Retrieval-Augmented Generation (RAG) pipeline and a fallback mechanism for improved contextual precision. We evaluate it on the Astrodynamics Problems Benchmark (APBench), a dataset of 299 questions covering foundational to advanced levels of space science. Using a dual evaluation method - numeric margin-based scoring and semantic similarity assessment - Taramandal-GPT achieves competitive performance against state-of-the-art open- and closed-source models, with notable strength in thinking-intensive tasks. These results highlight the value of specialized LLMs for domains demanding accuracy and interpretability, positioning Taramandal-GPT as a step toward reliable Artificial Intelligence (AI) assistants for astrophysics, spacecraft engineering, and space exploration.

## Full Text


<!-- PDF content starts -->

 
 Taramandal -GPT: Enhancing Astrodynamics Problem -Solving with Knowledge Retrieval and 
Structured Thinking  
Akhil Sharma, Jatin Gupta, Ali Imam Abidi *  
Department of Computer Science and Engineering  
Sharda University  
sharmaakhil944@gmail.com , jatingupta261001@gmail.com , aliabidi4685@gmail.com  
Abstract  
Large langua ge models (LLMs) have shown 
remarkable progress  in natural language 
understanding, yet their effectiveness in 
specialized fields like astronomy and 
astrodynamics remains limited due to 
challenges in multi -step reasoning, 
symbolic manipulation, and domain -
specific terminology. To address this, we 
present  Taramandal -GPT (Constellation - 
GPT), a domain -adapted framework built on 
the Qwen3 -8b backbone, enhanced with a 
Retrieval - Augmented Generation (RAG) 
pipeline and a fallback mechanism for 
improved contextual precision. We evaluate 
it on the Astrodynamics Problems 
Benchmark ( APBench ), a dataset of 299 
questions covering foundational to 
advanced levels of space science. Using a 
dual evaluation method —numeric margin -
based scoring and semantic similarity 
assessment —Taramandal - GPT achieves 
competitive performance against state -of- 
the-art open - and closed -source models, 
with notable strength in thinking -intensive 
tasks. These results highlight the value of 
specialized LLMs for domains demanding 
accuracy and interpretability, positioning 
Taramandal -GPT as a step toward reliable 
Artificia l Intelligence (AI) assistants for 
astrophysics, spacecraft engineering, and 
space exploration.  
Keywords: Large language models, 
Astrophysics, Astrodynamics, Space 
research, Taramandal -GPT, chain -of-thought 
reasoning  
1. Introduction  
The emergence of Large Language Models 
(LLMs), beginning with Vaswani et al.’s 
“Attention Is All You Need”, has transformed 
natural language processing through the 
Transformer architecture [1]. These models generalize well across domains and can 
extract and synthesize knowledge effectively. 
In astronomy, LLMs perform adequately in 
basic factual recall but struggle with complex 
tasks such as interpreting terminology, 
applying physical laws, and solving 
computa tional problems, revealing limitations 
in mathematical reasoning and domain -
specific expertise [2].  
Nonetheless, interest in applying LLMs to 
scientific and engineering contexts is 
growing. For example, the German Space 
Operations Center (GSOC) has explored their 
use in spacecraft engineering for real -time 
troubleshooting, documentation 
management, and d ecision support [3]. In 
parallel, domain -focused models such as 
AstroLLaMA [4] and astroBERT [5] show 
improved factual accuracy, contextual 
reasoning, and literature processing, 
demonstrating the value of targeted fine -
tuning.  
Yet, a key gap remains: no specialized 
thinking -focused LLM has been developed for 
astrodynamics and astrophysics, where both 
conceptual understanding and mathematical 
rigor are essential. To address this, we 
propose Taramandal -GPT (Constellation -
GPT), a d omain -adapted framework that 
integrates retrieval -based contextual 
grounding with structured thinking 
mechanisms. By emphasizing factual 
accuracy and symbolic computation, our 
approach seeks to move beyond general -
purpose assistance toward reliable tools f or 
advancing space science and engineering.  
2. Related Works  
Recent advances in domain‑specific AI for 
astronomy have produced several notable 
models, including astroBERT [7], a 
110M‑parameter BERT‑based model trained 
on astronomical literature for semantic search 

 
 and entity recognition; the AstroLLaMA family, 
with AstroLLaMA27B achieving a 30% 
perplexity reduction over LLaMA2 after 
fine‑tuning on 300k astronomy abstracts [8], 
and AstroLLaMA38B extending training with 
OCR‑processed and summarized paper 
sections [9];  the AstroMLab project’s 
AstroSage ‑LLaMA 3.1 -8B, reaching 80.9% 
accuracy on AstroMLab‑1 and matching 
GPT‑4o on fewer parameters [6], and the 
larger AstroSage‑LLaMA3.1 -70B, tying with 
Claude‑4Opus at 86.2% [7]; the StarWhisper 
LightCurve series for stellar classification, 
with a Swin Transfo rmer variant attaining 99% 
accuracy [8]; These developments 
collectively underscore the rapid maturation 
of AI‑driven tools as indispensable assets for 
modern astronomical research.  
3. Methodology  
This section outlines the methodology 
adopted for developing Taramandal -GPT, 
including dataset preparation, model 
selection, prompt engineering, and the design 
of the retrieval - augmented generation 
pipeline with a fallback mechanism.  
3.1 Dataset Description  
For model tuning, we drew on open‑source 
materials and books such as Fundamentals of 
Physics [9], Introduction to Space Physics 
[10], High‑Energy Astrophysics [10], and 
related references. Then, the text was 
extracted and was converted in a markdown 
strctutred format using the PymuPDF library. 
The text was embedded using  the Qwen3 -
Embedding -0.6B model [12] due to its 
exceptional multilingual capabilities, strong 
long‑text understanding, efficient 0.6B 
parameter size, and state‑of‑the‑art 
performance across diverse text embedding 
and retrieval tasks.  
For testing the full potential of LLMs in 
astrodynamics, we have utilized the first 
Astrodynamics Problems Benchmark 
(APBench) [13]  to  evaluate the capabilities of 
LLMs in this field. The benchmark consists of 
299 QA questions drawn from authoritative 
aerospace engineering sources, spanning difficulty from foundational concepts to 
PhD‑level problem‑solving.  
Table 1: Level‑wise Distribution of APBench 
Questions  
Benchmark 
Subset  Question Count  
APBench -α gordon (17) + UBC (8)  
APBench -β α (25)+ Braeunig (144)  
APBench -γ β (169) + lynnane (130)  
Total  299 
The dataset is structured into three sequential 
levels—alpha (α), beta (β), and gamma (γ) —
with their respective proportions presented in 
Table 1.  
3.2 Proposed Framework  
The proposed framework, Taramandal‑GPT 
(depicted in Figure 2), incorporates a base 
pre‑trained model, a system prompt, and an 
associated pipeline, each of which is 
described in the following subsections . 
3.1.1.  Base Model  
This study employs Qwen3:8b  [14], an 
open‑source large language model renowned 
for its reasoning capabilities in complex 
calculations and knowledge‑ intensive tasks, 
and competes with other models. The model 
was deployed with Q_4‑bit quantization to 
enable lightweight execution and e fficient 
accessibility.  
3.1.2.  System Prompt  
The system prompt integrates 
chain‑of‑thought reasoning, agent identity 
utilization, and regulatory mechanisms. These 
features work together to fine‑tune the model 
for domain‑specific responses, thereby 
enhancing accuracy. An overview of the 
system prompt is presented in Figure 1.  

 
  
Figure 1: Structure of System Prompt  
3.1.3.  Framework Pipeline  
The proposed framework, Taramandal‑GPT, 
employs a Retrieval‑Augmented Generation 
(RAG)‑ based architecture enhanced with a 
fallback mechanism. In the primary workflow, 
the system retrieves relevant contextual 
information from an external knowledge 
source a nd integrates it with the model’s 
generative capabilities to produce precise, 
context‑aware responses. However, when the 
retrieved context is insufficient or incomplete 
for generating a reliable output, the fallback 
mechanism is activated. In this mode, th e 
model leverages its pre‑trained knowledge 
base and engages in reasoning processes to 
formulate an accurate and coherent response, 
even in the absence of adequate external 
data.  
The overall architectural design , 
encompassing both the RAG pipeline and the 
fallback pathway , is illustrated in Figure 2, 
providing a clear visual representation of the 
system’s operational flow and decision logic.  
 4. Results and Discussion  
4.1.  Model Evaluation Metric  
For APBench, model performance is assessed 
using a dual -format scoring schema based on 
the nature of the expected output: numeric or 
message -based.  
4.1.1.  Numeric Answer Scoring  
Numeric responses were evaluated against 
ground truth using a dynamic error margin, 
which is defined in Equation 1.  
Error_Margin = min ⁡(0.1+0.01 ⋅log⁡(∣Answer∣), 
0.1)  
A model's output is considered correct if it 
falls within this margin. For instance, if the true 
answer is −5.2, the acceptable range would 
be approximately [−5.655, −4.479].  
4.1.2.  Message Answer Scoring  
Message -based responses are evaluated 
using a hybrid similarity approach that 
combines both semantic judgment and 
embedding -based comparison. First, GPT -4o 
serves as an LLM -as-a-Judge, assigning a 
similarity score between 0 and 10 based on 
the alignment be tween the model -generated 
response and the reference answer; this 
score is then normalized to a [0, 1] scale. In 
parallel, cosine similarity is computed using 
sentence embeddings derived from the all -
MiniLM -L6-v2 model. The final score is 
Figure 2: Taramandal‑GPT architecture with RAG pipeline & fallback mechanism  

 
 obtained by averaging these two components, 
as per Equation 2.  
Score = 1/2 × (LLM -Judge + Embedding 
Similarity)  
Responses achieving a score of 0.6 or higher 
are considered accurate within the 
benchmark evaluation framework.  
4.2  Evaluation Protocol  
Zero -shot prompting is employed throughout 
the evaluation process, wherein models are 
presented with questions and context without 
any prior examples or demonstrations.  
4.3 Performance of Taramandal - GPT  
The performance of the proposed framework, 
Taramandal‑GPT, is summarized in Table 2  
Table 2: Performance of Taramandal - GPT 
on different levels of APBench  
APBench  α β γ 
Accuracy (%)  56 44.37  50.8
4 It scores 56% on APBench -α, 44.37% on β, 
and 50.84% on γ, showing competitive 
performance despite its smaller size when 
compared to much larger models like 
Qwen2.5 -Math 72B.  
4.4 Comparative Performance of 
Taramandal -GPT vs Peer Models on 
APBench  
We compare performance of Taramandal -
GPT with other closed and open -source 
models on APBench -α, APBench -β, 
APBench -γ. Figure 3 visualizes the model -
wise performance trends extracted from the  
detailed evaluation results reported in Table 3.  
The models are grouped as closed source 
models and open source models. Despite of 
the individual model’s performance variation 
on the three Open APBench datasets, the 
difference between closed source models’ 
performance and open source models’ 
performance is clear.  
 
Figure 3: Performance comparison of Taramandal‑GPT with closed - and open -
source models on APBench‑α, β, and γ.  

 
  
 
Table 3:  LLMs Performance on APBench  
5.  Conclusion & Future Scope  
In this work, we introduced Taramandal -GPT, 
a reasoning - specialized large language 
model framework for advancing problem -
solving in astronomy and astrodynamics. By 
combining a Retrieval -Augmented Generation 
(RAG) pipeline with a fallback mechanism, it 
improves contextual awareness and 
generates more reliable responses than 
general -purpose and astronomy - focused 
models. Evaluation on APBench , a benchmark 
for astrodynamics problems, shows its ability 
to handle tasks from foundational principles to 
advanced research scenarios. These results 
highlight the potential of specialized LLMs in 
scientific domains where precision, 
mathematical rigor, an d interpretability are essential, though challenges such as 
hallucinations and slow deliberation remain.  
Looking ahead, future work includes 
expanding the training corpus with peer -
reviewed literature, spacecraft telemetry, and 
mission design data to improve factual 
grounding; integrating neuro -symbolic and 
physics -informed methods for greater 
mathematical re liability; and developing 
agent -based extensions to support interactive 
collaboration with scientists and engineers. 
Broader benchmarking across domains such 
as planetary science, satellite communication, 
and exoplanetary modeling will further test the 
framework’s robustness and applicability.  
6.  References  
1. Ashish Vaswani, et al., "Attention is All 
You Need," Advances in Neural 
Information Processing Systems 
(NeurIPS), 2017.  
2. Yu Wang, et al., "Can AI Understand 
Our Universe? Test of Fine -Tuning 
GPT by Astrophysical Data," arXiv 
preprint arXiv:2404.10019, 2024.  
3. Clemens Schefels, et al., "Evaluating 
Large Language Models for Space 
Operations," DLR Technical Report, 
2024.  
4. Rui Pan, et al., "AstroMLab 2: 
AstroLLaMA ‑2‑70B Model and 
Benchmarking Specialised  LLMs for 
Astronomy," Proceedings of the SC 
'24 Workshops of the International 
Conference on High Performance 
Computing, Network, Storage, and 
Analysis, IEEE Press, pp. 87 ‑96, 2025.  
5. Felix Grèzes, et al., "Building 
astroBERT, a Language Model for 
Astronomy & Astrophysics," arXiv 
preprint arXiv:2112.00590, 2021.  
6. Tijmen de Haan, et al., "Achieving 
GPT-4o Level Performance in 
Astronomy with a Specialized 8B -
Parameter Large Language Model," 
arXiv preprint arXiv:2411.09012, 2024.  Model  APBenc
h‑α (%) APBenc
h‑β (%) APBenc
h‑γ (%) 
Qwen2.5 -
Math 1.5B  20 21.3 22.1 
Qwen2.5 -
Math 7B  36 32.5  33.1 
Taramanda
l-GPT  56 44.37  50.84  
Qwen2.5 -
Math 72B  60 47.9  52.2  
AstroLlaMa  8 1 1 
Llama 2 7B  20 10.1 12.4 
Llama 3.1 
8B 20 17.2 18.4 
Llama 3.1 
70B 28 29.6  32.1 
Llama 3.2 
1B 8 11.8 8.4 
Llama 3.2 
3B 8 14.2 15.4 
ReflectionLl
ama 40 33.1 32.4  
ReflectionLl
ama†  28.6  30.2  29.8  
Ollama 
Reflection  28 23.7 22.1 

 
 7. Tijmen de Haan et al., " AstroMLab 4: 
Benchmark -Topping Performance in 
Astronomy Q&A with a 70B -Parameter 
Domain -Specialized Reasoning 
Model," arXiv preprint 
arXiv:2505.17592, 2025.  
8. Cunshi Wang et al., "StarWhisper 
Telescope: Agent -Based Observation 
Assistant System to Approach AI 
Astrophysicist," arXiv preprint 
arXiv:2412.06412, 2025.  
9. David Halliday, et al., "Fundamentals 
of Physics," Wiley, multiple editions 
(latest 11th edition, 2018).  
10. Margaret G. Kivelson  and Christopher 
T. Russell (eds.), "Introduction to 
Space Physics," Cambridge University 
Press, 1995.  
11. Malcolm S. Longair, "High Energy 
Astrophysics," Cambridge University 
Press, 3rd edition, 2011.  
12. Zhang, Y., et al., “Qwen3 Embedding: 
Advancing text embedding and 
reranking through foundation 
models,” arXiv preprint 
arXiv:2506.05176, (2025)  
13. Paolo Turrini, et al., "APBench  and 
Benchmarking Large Language Model 
Performance in Fundamental 
Astrodynamics Problems for Space 
Engineering," Scientific Reports, 
2025.  
14. Qwen Team (An Yang, et al.) “Qwen3 
Technical Report,” arXiv preprint 
arXiv:2505.09388, 2025.  
 
 
 
 
 
 
  
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 

अखिल भा 
अखिल भारतीय ह िंदी तकनीकी  सम्मेलन 2025                    आययभट्ट से अनिंत अिंतररक्ष की ओर  
 
 
 
यू.आर. राव उपग्र  क ें द्र , बेंगलूरु  340 भारतीय अिंतररक्ष अनुसिंधान सिंगठन  
 तारामिंिल -GPT:  िगोलगहतकी में ज्ञान पुनप्रायखप्त व सिंरहचत हचिंतन द्वारा समस्या -समाधान  
अस्थखल शमाा , जप्रति गुप्ता एिं अली इमाम आप्रबदी  
शारदा प्रििप्रिद्यालय , ग्रे र िोएडा , उत्तर िदेश  
sharmaakhil944@gmail.com , jatingupta261001@gmail.com , aliabidi4685@gmail.com  
 
सारािंि  - हपछले  क ु छ समय  में Large 
Language Models (LLMs) ने प्राक ृ हतक  
भािा को समझने  में जबरदस्त  प्रगहत  हदिाई  
 ै, लेहकन  िगोलिास्त्र  और िगोलगहतकी  
जैसे िास क्षेत्रोिं में इनकी  प्रभाविीलता  
सीहमत  र ी  ै। इनका  मुख्य कारण   ैं 
बहुचरणीय  सोच, प्रतीकात्मक  गणना  और 
क्षेत्र-हविेि  िब्दावली  द्वारा उत्पन्न   ोने वाली 
चुनौहतयाँ।  इसे  ल करने  के हलए,  म प्रस्तुत  
करते   ैं तारामिंिल -GPT। य  Qwen3 -8b 
पर आधाररत  एक क्षेत्र-अनुक ू हलत  मॉिल   ै, 
हजसमें  ‘Retrieval Augmented 
Generation (RAG)’ प्रणाली  और सटीकता  
के हलए एक ‘Fallback’ सुहवधा  जोडी  गई  ै। 
 मने इसका  मूल्ािंकन  Astrodynamics 
Problems Benchmark (APBench) पर 
हकया , हजसमें  299 सवाल  िाहमल   ैं, जो 
अिंतररक्ष  हवज्ञान  के मूलभूत  से लेकर  उन्नत 
स्तर तक के हवियोिं  से जुडे हुए  ैं। दो री  
मूल्ािंकन  हवहध, ‘numeric margin’ 
आधाररत  ‘scoring’ और ‘semantic 
similarity’ का उपयोग  करते  हुए, 
तारामिंिल -GPT ने हवचार -प्रधान  कायों  में 
उत्क ृ ष्ट्ता  के साथ अन्य आधुहनक  िुला स्रोत  
मॉिल्स  के अपेक्षाक ृ त  बे तर  प्रदियन  हकया।  
ये पररणाम  साहबत  करते   ैं हक हविेितः  
सटीकता  और व्याख्या  की आवश्यकता  वाले 
क्षेत्रोिं में LLMs हकतने  उपयोगी   ैं। 
तारामिंिल -GPT िगोलभौहतकी , अिंतररक्ष  
यान अहभयािंहत्रकी  और अिंतररक्ष  में िोज  के 
हलए भरोसेमिंद  AI अहसस्टेंट  की हदिा  में एक 
म त्वपूणय  कदम   ै। क ुिं जी िब्द  - Large Language Models, 
िगोलभौहतकी , िगोलगहतकी , अिंतररक्ष  
अनुसिंधान , तारामिंिल -GPT, ‘chain -of-
thought reasoning’  
1. प्रस्तावना  
Vaswani et al. के “Attention Is All You 
Need” [1] पेपर के साथ Large Language 
Models (LLMs) का पदापार्  हुआ, प्रजससे  
natural language processing (NLP) में 
बहुत क्रांप्रत आई है, खासकर  Transformer 
architecture की िजह से। ये models अलग -
अलग  प्रिधाओं  में अच्छी  तरह सामान्यीकरर्  
करते हैं और अप्रजात  जािकाररयों  को अच्छे से 
प्रिकालते  और जोड़ते  हैं। खगोल -प्रिज्ञाि  में, 
LLMs बुप्रियादी  तर्थ् आधाररत  सिालों  में तो 
ठीक-ठाक काम करते हैं, लेप्रकि  जब शब्दािली  
समझिे , भौप्रतकी  के प्रियम  लागू करिे या गर्िा  
िाले सिाल  हल करिे की बात आती है, तो इन्हें 
मुस्थिल  होती है। यािी, गप्रर्तीय  तक ा और 
ज्ञािक्षेि  डोमेि -प्रिप्रशष्ट्  प्रिशेषज्ञता  में इिकी  
सीमाएं  सामिे  आती हैं [2]। प्रफर भी, प्रिज्ञाि  और 
अप्रभयांप्रिकी  में LLMs का इस्तेमाल  बढता  जा 
रहा है। जैसे, German Space Operations 
Center (GSOC) िे अंतररक्ष  याि अप्रभयास्थिकी  
में िास्तप्रिक  समय (real-time) पर समस्या  
प्रििारर् , दस्तािेजीकरर्  और उप्रचत  प्रिर्ाय  में 
समथाि  के प्रलए इिका  उपयोग करिा  शुरू 
प्रकया है [3]। इसी के साथ, AstroLLaMA [4] 
और astroBERT [5] जैसे िक्षेि- क ें प्रद्रत  
models िे तर्थ्ात्मक  स ीकता , िासंप्रगक  तक ा, 
और साप्रहत्य  िसंस्करर्  में अच्छा  िदशाि  
प्रदखाया  है, प्रजससे  यह साफ है प्रक लप्रक्षत  fine-
tuning फायदेमंद  है । 
इसक े  बािजूद , एक खामी है—अब तक ऐसा 
कोई प्रिशेषीक ृ त  LLM उपलब्ध  िहीं है जो 

अखिल भा 
अखिल भारतीय ह िंदी तकनीकी  सम्मेलन 2025                    आययभट्ट से अनिंत अिंतररक्ष की ओर  
 
 
 
यू.आर. राव उपग्र  क ें द्र , बेंगलूरु  341 भारतीय अिंतररक्ष अनुसिंधान सिंगठन  
 खासतौर  पर “अंतररक्ष गप्रतप्रिज्ञाि ” और 
“खगोलभौप्रतकी ” के प्रलए तैयार प्रकया गया हो, 
जहां िैचाररक  समझ  और गप्रर्तीय  दृढता  दोिों 
जरूरी  हैं। इसी कमी को पूरा करिे के प्रलए हम 
िस्ताप्रित  करते हैं तारामंडल -GPT। यह एक 
domain -adapted र े मिक ा   है, प्रजसमें  
retrieval -based contextual grounding 
और structured thinking mechanisms को 
एकीक ृ त प्रकया  गया है। हमारा  जोर factual 
accuracy और symbolic computation पर 
है, ताप्रक general -purpose assistance से 
आगे बढकर  space science और 
engineering को आगे ले जािे के प्रलए 
भरोसेमंद  tools तैयार प्रकए जा सक ें । 
2. सिंबिंहधत  कायय 
हाल के िषों में खगोल  प्रिज्ञाि  के प्रलए क्षेि-
प्रिप्रशष्ट्  क ृ प्रिम  बुस्थिमत्ता  (AI) में हुई िगप्रत  िे 
अिेक  उल्लेखिीय  मॉडलों  को जन्म प्रदया है। 
इिमें AstroBERT [5] शाप्रमल  है, जो 110M 
parameter िाला BERT आधाररत  मॉडल  है 
और प्रजसे खगोल -प्रिज्ञाि  साप्रहत्य  पर िप्रशप्रक्षत  
प्रकया गया है ताप्रक उसका  उपयोग  समािाथी  
खोज तथा सत्तात्मक  पहचाि  में प्रकया जा सक े । 
इसी श्रेर्ी में AstroLLaMA पररिार उल्लेखिीय 
है, जहाँ हाप्रलया अध्ययि में AstroLLaMA -2-
7B आधारभूत मॉडल से 7–8% कमतर पाया 
गया, जबप्रक AstroLLaMA -3-8B िे ज्ञाि 
संरक्षर् प्रकया पर astro -ph डे ा पर सुधार िहीं 
प्रदखाया। छो े मॉडलों में catastrophic 
forgetting देखा गया , प्रकन्तु AstroLLaMA -
2-70B िे मूल LLaMA2 -70B की तुलिा में स्पष्ट् 
लाभ दशााया , प्रजससे प्रसि होता है  प्रक खगोल —
प्र प्रशष्ट् सतत िी - रेप्रिंग  का िभाि मुख्यतः  70B 
श्रेर्ी क े  मॉडलों पर ही उपयोगी है  [4]। 
AstroMLab का AstroSage ‑LLaMA 3.1 -8B 
[6] िे AstroMLab ‑1 पर 80.9% स ीकता  
िाप्त की और अपेक्षाक ृ त  कम पैरामी रों  के 
साथ GPT ‑4o के तुल्य िदशाि  प्रकया।  िहीं, बड़े 
AstroSage ‑LLaMA3.1 ‑70B [7] मॉडल  िे 
Claude ‑4Opus के साथ 86.2% के स्तर पर 
समाि  क्षमता  िदप्रशात  की। इसक े  अप्रतररि , 
StarWhisper LightCurve [8] श्रृंखला  िे ताप्रक ा क  िगीकरर्  में महत्वपूर्ा  योगदाि  प्रदया 
है, जहाँ इसक े  Swin Transformer संस्करर्  िे 
99% स ीकता  िाप्त की। 
3. काययहवहध / काययप्रणाली  
इस खंड में तारामंडल ‑GPT के प्रिकास  हेतु 
अपिाई  गई कायािर्ाली  का प्रििरर्  प्रदया गया 
है, प्रजसमें  डा ासे  तैयारी , मॉडल चयि , 
Prompt इंजीप्रियररंग , तथा fallback 
mechanism सप्रहत  retrieval ‑आधाररत  
जिरेशि पाइपलाईि की  अप्रभकल्पिा  
सस्थम्मप्रलत  है। 
3.1 िेटासेट हववरण एविं रूपािंतरण प्रहक्या  
मॉडल  ट्यूप्रिंग  के प्रलए हमिे open ‑source 
materials और पुस्तक ें  जैसे Fundamentals 
of Physics [9], Introduction to Space 
Physics [10], High ‑Energy Astrophysics 
[11] तथा संबंप्रधत  संदभों  का उपयोग  प्रकया।  
Textual data को PyMuPdf python library 
की सहायता से प्रिकाला गया और प्रफर इसे 
Markdown format में पररिप्रतात प्रकया गया , 
ताप्रक LLM m odel में retrieval क े  समय 
बेहतर समझ प्रमल सक े ।  इसक े  बाद text को 
chunks में प्रिभाप्रजत प्रकया गया , जहाँ ित्येक 
chunk size 1000  तथा chunk overlap 200  
रखा गया , ताप्रक chunks क े  बीच संबंध बिा 
रहे। इि chunks को Qwen3 -Embedding -
0.6B embedding model [12] की मदद से 
embed  प्रकया गया और embeddings को 
FAISS (Facebook Index Similarity 
Search) format में save प्रकया गया।  सहेजे 
गए FAISS vectorstore का उपयोग िश्नों क े  
उत्तर देिे क े  दौराि contextual information 
retrieval क े  प्रलए प्रकया गया। .  


अखिल भा 
अखिल भारतीय ह िंदी तकनीकी  सम्मेलन 2025                    आययभट्ट से अनिंत अिंतररक्ष की ओर  
 
 
 
यू.आर. राव उपग्र  क ें द्र , बेंगलूरु  342 भारतीय अिंतररक्ष अनुसिंधान सिंगठन  
 LLMs की खगोलगप्रतकी क्षेि  में पूर्ा क्षमता  का 
परीक्षर्  करिे हेतु, हमिे Astrodynamics 
Problems Benchmark (APBench) [13] का 
उपयोग प्रकया  है, प्रजसका  उद्देश्  इस क्षेि में 
LLMs की क्षमताओं  का आकलि  करिा  है। इस 
benchmark में क ु ल 299 QA िश्न शाप्रमल  हैं, 
प्रजन्हें authoritative aerospace 
engineering f oundational concepts से 
लेकर  PhD ‑level तक की कप्रठिाई  स्तरों को 
आच्छाप्रदत  करते हैं। 
ताहलका 1: APBench प्रश्ोिं का स्तरवार 
हवतरण  
Benchmark Subset  Question Count  
APBench – α (gordon) 17 + 
(UBC) 8  
APBench – β α (25) + 
(Braeunig) 144  
APBench – γ β (169) + 
(lynnane) 130  
यह dataset तीि क्रप्रमक स्तरों —alpha (α), 
beta (β) और gamma ( γ)—में संरप्रचत है , 
प्रजिका अिुपात ताप्रलका 1 में िस्तुत प्रकया 
गया है।  
3.2 प्रस्ताहवत फ्र े मवक य    
िस्ताप्रित  र े मिक ा  , तारामंडल ‑GPT (जैसा प्रक 
प्रचि 2 में दशााया  गया है), में एक base 
pre‑trained model, एक system prompt, तथा एक संबि  pipeline सस्थम्मप्रलत  है, प्रजिका  
प्रििरर्  प्रिम्न उपखंडों  में िस्तुत  प्रकया गया है। 
हचत्र 1: System Prompt  
3.1.1.  बेस मॉिल ( Base Model)  
यह अध्ययि  Qwen3 -8b [14] का उपयोग  
करता  है, जो एक खुला स्रोत  LLM है और जप्र ल  
गर्िाओं  तथा ज्ञाि-गहि काया  में अपिी  तक ा  
क्षमताओं  के प्रलए िप्रसि  है। मॉडल  को 
Q_4 ‑प्रब  क्ां ाइजेशि  के साथ पररप्रियोप्रजत  
प्रकया गया है ताप्रक हि े  प्रिष्पादि और  क ु शल 
पहुंच सुप्रिप्रित  की जा सक े । 
3.1.2. हसस्टम प्रॉम्प्ट ( System Prompt)  
System prompt में chain ‑of‑thought 
reasoning, surrogate feature utilization 
और regulatory mechanisms सस्थम्मप्रलत  हैं। 
ये प्रिशेषताएँ  प्रमलकर  मॉडल  को क्षेि-प्रिप्रशष्ट्  
उत्तरों  के प्रलए fine ‑tune करती  हैं, प्रजससे  
स ीकता  में िृस्थि होती है। िर्ाली  prompt का 
एक अिलोकि  प्रचि 1 में िस्तुत  प्रकया गया है। 
3.1.3.  फ्र े मवक य   पाइपलाईन  
िस्ताप्रित  र े मिक ा , तारामंडल -GPT, एक 
Retrieval ‑Augmented Generation (RAG) 
आधाररत  िास्तुकला  का उपयोग  करता  है, प्रजसे 
fallback mechanism के साथ सशि  बिाया  
गया है। मुख्य कायाििाह  में, िर्ाली  बाहरी  
knowledge source से िासंप्रगक  संदभा  
जािकारी  (contextual data) पुिः िाप्त करती  
है और उसे मॉडल  की generative 
capabilities के साथ एकीक ृ त  कर, स ीक  
और संदभा -सूप्रचत  उत्तर िदाि  करती  है। 
हचत्र 2: तारामिंिल ‑GPT सिंरचना   
 

अखिल भा 
अखिल भारतीय ह िंदी तकनीकी  सम्मेलन 2025                    आययभट्ट से अनिंत अिंतररक्ष की ओर  
 
 
 
यू.आर. राव उपग्र  क ें द्र , बेंगलूरु  343 भारतीय अिंतररक्ष अनुसिंधान सिंगठन  
 हालाँप्रक , जब पुिः िाप्त संदभा  प्रििसिीय  
पररर्ाम  उत्पन्न  करिे के प्रलए अपयााप्त  या 
अधूरा  होता है, तो fallback mechanism 
सप्रक्रय  हो जाता है। इस स्थस्थप्रत  में, मॉडल  अपिे 
pre‑trained knowledge base का उपयोग  
करता  है और तक ा  िप्रक्रयाओं ( reasoning 
processes) से गुजरते  हुए पयााप्त  बाहरी  डे ा 
की अिुपस्थस्थप्रत  में भी स ीक  और संगत उत्तर 
प्रिमाार्  करता  है। 
संपूर्ा  िास्तुप्रशल्प  प्रडजाइि  — प्रजसमें  RAG 
pipeline और fallback pathway दोिों 
शाप्रमल  हैं — को प्रचि 2 में दशााया  गया है, जो 
िर्ाली  के संचालि  ििाह और प्रिर्ाय  तक ा का 
स्पष्ट् दृश् िस्तुत  करता  है। 
4. पररणाम एविं चचाय  
4.1. मॉिल मूल्ािंकन मापदिंि   
APBench के प्रलए, मॉडल  का िदशाि  अपेप्रक्षत  
आउ पु  की िक ृ प्रत  के आधार  पर, जो प्रक 
संख्यात्मक  या संदेश -आधाररत (message -
based) हो सकता  है, एक प्रद्व-फॉमे  स्कोररंग  
स्कीमा  द्वारा आंका  जाता है। 
4.1.1.  सिंख्यात्मक उत्तर का मूल्ािंकन   
संख्यात्मक  उत्तरों  का मूल सत्य के साथ 
मूल्यांकि  एक dynamic error margin का 
उपयोग  करक े  प्रकया गया, प्रजसे समीकरर्  1 में 
पररभाप्रषत  प्रकया गया है। 
𝑬𝒓𝒓𝒐𝒓 _𝑴𝒂𝒓𝒈𝒊𝒏  = 𝒎𝒊𝒏 (𝟎.𝟏+𝟎.𝟎𝟏⋅
𝒍𝒐𝒈(∣𝑨𝒏𝒔𝒘𝒆𝒓 ∣),𝟎.𝟏) (1) 
यप्रद मॉडल  का आउ पु  इस सीमा के भीतर  
आता है, तो उसे सही मािा जाता है। उदाहरर्  
के प्रलए, यप्रद िास्तप्रिक  उत्तर −5.2 है, तो 
स्वीक ृ त  सीमा लगभग  [−5.655, −4.479] 
होगी।  
4.1.2.  सिंदेि-आधाररत उत्तर का मूल्ािंकन  
(Message Answer Scoring)  
संदेश-आधाररत उत्तरों  का मूल्यांकि  एक 
hybrid similarity approach द्वारा प्रकया 
जाता है, जो semantic judgment और 
embedding -based comparison दोिों को 
सस्थम्मप्रलत  करता  है। सबसे  पहले, GPT -4o एक 
LLM-as-a-Judge के रूप में काया करता  है 
और मॉडल -प्रिप्रमात  उत्तर तथा संदभा  उत्तर के बीच मेल के आधार  पर 0 से 10 के बीच एक 
समािता  स्कोर (similarity score) प्रिधााररत 
करता है। यह स्कोर बाद में  सामान्यीक ृ त प्रकया 
जाता है।  समांतर  रूप से, all-MiniLM -L6-v2 
मॉडल  से िाप्त िाक् एम्बेप्रडंग्स  (sentence 
embeddings) का उपयोग  कर cosine 
similarity की गर्िा  की जाती है। अंप्रतम  स्कोर 
इि दोिों घ कों  के औसत  के रूप में िाप्त प्रकया 
जाता है, जैसा प्रक समीकरर्  2 में प्रदया गया है। 
सामान्यीक ृ त प्रकया जाता है।   
𝑺𝒄𝒐𝒓𝒆  = 𝟏/𝟐 × (𝑳𝑳𝑴 −𝑱𝒖𝒅𝒈𝒆  +
 𝑬𝒎𝒃𝒆𝒅𝒅𝒊𝒏𝒈  𝑺𝒊𝒎𝒊𝒍𝒂𝒓𝒊𝒕𝒚 ) (2) 
जो उत्तर 0.6 या उससे  अप्रधक  स्कोर िाप्त करते 
हैं, उन्हें benchmark evaluation र े मिक ा   के 
अंतगात  स ीक  मािा जाता है। 
4.2 मूल्ािंकन प्रोटोकॉल   
पूरा मूल्यांकि  िप्रक्रया  में zero-shot 
prompting का उपयोग  प्रकया जाता है, जहाँ 
मॉडल्स  को प्रबिा प्रकसी  पूिा उदाहरर्  या 
िदशािी  के सीधे िश्न और संदभा  िस्तुत  प्रकए 
जाते हैं।  
हचत्र 3: APBench पर तारामिंिल -GPT का 
प्रदियन स कमी िुला-स्रोत मॉिलोिं की तुलना 
में 
स्रोत: 
https://www.nature.com/articles/s41598 -
025-91150 -5  

अखिल भा 
अखिल भारतीय ह िंदी तकनीकी  सम्मेलन 2025                    आययभट्ट से अनिंत अिंतररक्ष की ओर  
 
 
 
यू.आर. राव उपग्र  क ें द्र , बेंगलूरु  344 भारतीय अिंतररक्ष अनुसिंधान सिंगठन  
 4.3 तारामिंिल -GPT का प्रदियन  िस्ताप्रित 
तारामंडल-GPT का िदशाि ताप्रलका   3 में 
सारांप्रशत प्रकया गया है। इसक े  छो े आकार  
और 4-bit Quantized  होिे क े  बािजूद , यह 
िदशाि िप्रतस्पधाात्मक है , प्रिशेष रूप से बड़े 
मॉडलों जैसे Qwen2.5 -Math 72B  की तुलिा 
में। 
ताहलका  3: APBench क े  हवहभन्न स्तरोिं पर  
तारामिंिल -GPT का प्रदियन  
APBench  α β γ 
Accuracy (%)  56 44.37  50.84  
 
तारामंडल -GPT के िदशाि  की तुलिा  अन्य 
open -source मॉडलों  से APBench -α, 
APBench -β, APBench -γ पर करते हैं। प्रचि 3 
में उस्थल्लस्थखत  प्रिस्तृत  मूल्यांकि  पररर्ामों  से 
िाप्त मॉडल -िार िदशाि  ििृप्रत्तयों  का दृश् 
िस्तुत  प्रकया गया है। तीिों APBench  datasets 
पर व्यस्थिगत  मॉडल  के िदशाि  में क ु छ 
प्रिप्रिधताएँ  होिे के बािजूद , open -source 
models के िदशाि  के बीच स्पष्ट् अंतर प्रदखाई  
देता है। 
5. हनष्किय एविं भावी काययक्षेत्र   
इस काया में, हमिे तारामंडल -GPT िस्तुत  
प्रकया, जो खगोल  प्रिज्ञाि  और खगोलीय  
गप्रतशास्त्र  में समस्या -समाधाि  को उन्नत करिे 
हेतु एक क्षेि-प्रिप्रशष्ट्  LLM र े मिक ा   है। 
Retrieval -Augmented Generation (RAG) 
पाइपलाइि  को fallback mechanism के साथ 
संयोप्रजत  करक े  यह संदभा  जागरूकता  में सुधार  
करता  है और सामान्य  ियोजि  तथा खगोल  
प्रिज्ञाि -क ें प्रद्रत  मॉडलों  की तुलिा  में अप्रधक  
प्रििसिीय  उत्तर उत्पन्न  करता  है। APBench पर 
मूल्यांकि , जो खगोलीय  गप्रतशास्त्र  समस्याओं  के 
प्रलए एक मािक  है, िे इसक े  मौप्रलक  प्रसिांतों  से 
लेकर  उन्नत अिुसंधाि  पररदृश्ों  तक के कायों 
को संभालिे  की क्षमता  दशााई  है। ये पररर्ाम  
िैज्ञाप्रिक  क्षेिों में प्रिशेषज्ञ  LLMs की संभाििाओं  
को उजागर  करते हैं, जहाँ स ीकता , गप्रर्तीय  
कठोरता , और व्याख्यात्मकता  अत्यंत  आिश्क  
हैं, हालांप्रक  hallucinations और धीमी प्रिचार  िप्रक्रया  जैसी चुिौप्रतयाँ  बरकरार  हुई हैं। िौ गुिा 
कम पैरामी र होिे और 4 प्रब  क्ां  ाइजेशि में  
क्ां ाइज्डहोिे क े   बाद भी , मॉडल का परफॉमेंस 
बड़े 72B LLM  क े  मुकाबले का है और इसमें 
सुदूर अंतररक्ष क े  प्रलए कम ऊजाा खपत तथा 
कम हाडािेयर की जरुरत होती है।  
आगे देखते  हुए, भप्रिष्य  का काया िप्रशक्षर्  
कॉपास  का प्रिस्तार  करिा  है, प्रजसमें  समकक्ष -
परीप्रक्षत  साप्रहत्य , उपग्रह   ेलीमे री , और प्रमशि  
प्रडजाइि  डे ा शाप्रमल  होंगे ताप्रक तर्थ्ात्मक  
आधार  सुधारा  जा सक े; गप्रर्तीय  प्रििसिीयता  
के प्रलए neuro -symbolic और physics -
informed प्रिप्रधयों  का समाकलि ; और 
िैज्ञाप्रिकों  एिं अप्रभयंताओं  के साथ 
अंतः प्रक्रयात्मक  सहयोग  को समथाि देिे के प्रलए 
एजें -आधाररत  प्रिस्तार  प्रिकप्रसत  करिा  शाप्रमल  
है। ग्रह प्रिज्ञाि , उपग्रह  संचार , और बप्रहग्राह  
मॉडप्रलंग  जैसे अन्य क्षेिों में व्यापक  
मािकीकरर्  इस र े मिक ा   की मजबूती  और 
ियोज्यता  का और परीक्षर्  करेगा।  
6. सिंदभय  
[1] Ashish Vaswani, et al., "Attention is 
All You Need," Advances in Neural 
Information Processing Systems 
(NeurIPS) , 2017.  
[2] Yu Wang, et al.,  "Can AI Understand 
Our Universe? Test of Fine -Tuning 
GPT by Astrophysical Data," arXiv 
preprint arXiv:2404.10019 , 2024.  
[3] Clemens Schefels, et al., "Evaluating 
Large Language Models for Space 
Operations," DLR Technical Report , 
2024.  
[4] Rui Pan, et al.,  "AstroMLab 2: 
AstroLLaMA ‑2‑70B Model and 
Benchmarking Specialised LLMs for 
Astronomy," Proceedings of the SC 
'24 Workshops of the International 
Conf erence on High Performance 
Computing, Network, Storage, and 
Analysis, IEEE Press, pp.  87‑96, 2025.  

अखिल भा 
अखिल भारतीय ह िंदी तकनीकी  सम्मेलन 2025                    आययभट्ट से अनिंत अिंतररक्ष की ओर  
 
 
 
यू.आर. राव उपग्र  क ें द्र , बेंगलूरु  345 भारतीय अिंतररक्ष अनुसिंधान सिंगठन  
 [5] Felix Grèzes, et al.,  "Building 
astroBERT, a Language Model for 
Astronomy & Astrophysics," arXiv 
preprint arXiv:2112.00590 , 2021.  
[6] Tijmen de Haan, et al., "Achieving 
GPT-4o Level Performance in 
Astronomy with a Specialized 8B -
Parameter Large Language Model," 
arXiv preprint arXiv:2411.09012 , 
2024.  
[7] Tijmen de Haan et al., "AstroMLab 4: 
Benchmark -Topping Performance in 
Astronomy Q&A with a 70B -
Parameter Do main -Specialized 
Reasoning Model," arXiv preprint 
arXiv:2505.17592, 2025.  
[8] Cunshi Wang et al., "StarWhisper 
Telescope: Agent -Based 
Observation Assistant System to 
Approach AI Astrophysicist," arXiv 
preprint arXiv:2412.06412, 2025.  
[9] David Halliday, et al. , "Fundamentals 
of Physics," Wiley , multiple editions 
(latest 11th edition, 2018).  [10] Margaret G. Kivelson and 
Christopher T. Russell (eds.),  
"Introduction to Space Physics," 
Cambridge University Press , 1995.  
[11] Malcolm S. Longair , "High 
Energy Astrophysics," Cambridge 
University Press , 3rd edition, 2011.  
[12] Zhang, Y., et al.,  “Qwen3 
Embedding: Advancing text 
embedding and reranking through 
foundation models,” arXiv preprint 
arXiv:2506.05176, (2025)  
[13] Paolo Turrini, et al.,  "APBench  
and Benchmarking Large Language 
Model Performance in Fundamental 
Astrodynamics Problems for Space 
Engineering," Scientific Reports , 
2025.  
[14] Qwen Team (An Yang, et al.) 
“Qwen3 Technical Report,” arXiv 
preprint arXiv:2505.09388, 2025.  
  