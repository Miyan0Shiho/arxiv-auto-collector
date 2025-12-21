# DrugRAG: Enhancing Pharmacy LLM Performance Through A Novel Retrieval-Augmented Generation Pipeline

**Authors**: Houman Kazemzadeh, Kiarash Mokhtari Dizaji, Seyed Reza Tavakoli, Farbod Davoodi, MohammadReza KarimiNejad, Parham Abed Azad, Ali Sabzi, Armin Khosravi, Siavash Ahmadi, Mohammad Hossein Rohban, Glolamali Aminian, Tahereh Javaheri

**Published**: 2025-12-16 20:19:23

**PDF URL**: [https://arxiv.org/pdf/2512.14896v1](https://arxiv.org/pdf/2512.14896v1)

## Abstract
Objectives: To evaluate large language model (LLM) performance on pharmacy licensure-style question-answering (QA) tasks and develop an external knowledge integration method to improve their accuracy.
  Methods: We benchmarked eleven existing LLMs with varying parameter sizes (8 billion to 70+ billion) using a 141-question pharmacy dataset. We measured baseline accuracy for each model without modification. We then developed a three-step retrieval-augmented generation (RAG) pipeline, DrugRAG, that retrieves structured drug knowledge from validated sources and augments model prompts with evidence-based context. This pipeline operates externally to the models, requiring no changes to model architecture or parameters.
  Results: Baseline accuracy ranged from 46% to 92%, with GPT-5 (92%) and o3 (89%) achieving the highest scores. Models with fewer than 8 billion parameters scored below 50%. DrugRAG improved accuracy across all tested models, with gains ranging from 7 to 21 percentage points (e.g., Gemma 3 27B: 61% to 71%, Llama 3.1 8B: 46% to 67%) on the 141-item benchmark.
  Conclusion: We demonstrate that external structured drug knowledge integration through DrugRAG measurably improves LLM accuracy on pharmacy tasks without modifying the underlying models. This approach provides a practical pipeline for enhancing pharmacy-focused AI applications with evidence-based information.

## Full Text


<!-- PDF content starts -->

This manuscript is a preprint and has been submitted for peer review to Computer Methods and Programs in Biomedicine . The 
content may be updated following editorial review.  DrugRAG: Enhancing Pharmacy LLM Performance Through  A Novel Retrieval -
Augmented Generation Pipeline  
Houman Kazemzadeh  1, Kiarash Mokhtari Dizaji  2, Seyed Reza Tavakoli  3, Farbod Davoodi  4, 
MohammadReza KarimiNejad  5, Parham Abed Azad  5, Ali Sabzi  5, Armin Khosravi  5, Siavash 
Ahmadi  6, *, Mohammad Hossein Rohban  5, Glolamali Aminian  7, Tahereh Javaheri  8, * 
1 Department of Medicinal Chemistry, Faculty of Pharmacy, Tehran University of Medical Sciences, Tehran, Iran  
2 Department of Computer Science s, Faculty of Mathematics and Computer Science s, Amir Kabir University of 
Technology, Tehran, Iran  
3 Department of Mathematical Science s, Sharif University of Technology, Tehran, Iran  
4 Department of Computer Science, Missouri University of Science and Technology, Rolla, MO, USA  
5 Department of Computer Engineering, Sharif University of Technology, Tehran, Iran  
6 Electronics Research Institute, Sharif University of Technology, Tehran, Iran  
7 The Alan Turing Institute, London, United Kingdom  
8 Health Informatics Lab, Metropolitan College, Boston University, Boston, USA  
* Corresponding Author s 
 
 
Abstract  
Objectives: To evaluate large language model (LLM ) performance on pharmacy  licensure -
style  question -answering (QA) tasks and develop an external knowledge integration 
method to improve their accuracy.  
 
Methods: We benchmarked eleven  existing LLMs with varying parameter sizes (8 billion to 
70+ billion ) using a 141 -question pharmacy dataset. We measured baseline accuracy for 
each model without modification. We then developed a three -step retrieval -augmented 
generation (RAG ) pipeline , DrugRAG,  that retrieves structured drug knowledge from 
validated sources and augments model prompts with evidence -based context. This pipeline 
operates externally to the models, requiring no changes to model architecture or 
parameters.  
 
Results: Baseline accuracy ranged from 46% to 92%, with GPT -5 (92% ) and o3 (89% ) 
achieving the highest scores. Models with fewer than 8 billion parameters scored below 
50%. DrugRAG  improved accuracy  across all tested models, with gains ranging from 7 to 21 
percentage points (e.g., Gemma 3 27B: 61% to 71%, L lama 3.1 8B: 46% to 67% ) on the 141 -
item benchmark . 
 
Conclusion: We demonstrate that external structured drug knowledge integration through 
DrugRAG  measurably improves LLM accuracy on pharmacy tasks without modifying the 
underlying models. This approach provides a practical pipeline  for enhancing pharmacy -
focused AI applications with evidence -based information.  
 
Keywords:  Large Language Models; Structured Knowledge; Pharmacy QA; Clinical AI  

This manuscript is a preprint and has been submitted for peer review to Computer Methods and Programs in Biomedicine . The 
content may be updated following editorial review.  1. Introduction  
Large language models have created new opportunities for AI -supported learning in 
pharmacy and healthcare  1,2. General -purpose models such as GPT -4 show high 
performance on medical education exams, yet pharmacy presents distinct challenges  3,4. 
Pharmacists must master accurate drug selection, dose calculations, and context -specific 
decision -making  5. Before deploying LLMs as tutoring systems or study aids, we need 
rigorous evaluation of their pharmacy -specific capabilities.  
 
The NAPLEX (North American Pharmacist Licensure Examination ), administered by the 
National Association of Boards of Pharmacy, serves as the standard licensure examination 
for pharmacy graduates and evaluates entry -level competency across five content domains: 
Foundational Knowledge for Pharmacy Practice, Medication Use Process, Person -Centered 
Assessment and Treatment Planning, Professional Practice, and Pharmacy Management and 
Leadership  6,7. Recent benchmarks using NAPLEX -style questions reveal significant 
performance gaps. Angel et al. tested GPT -4, Bard, and GPT -3 on 137 questions, with GPT -4 
achieving 78.8% accuracy  8. Ehlert et al. found GPT -4 reaching 87% on McGraw -Hill 
questions while smaller models trailed at 60 -68%  9. A follow -up study by Angel et al. 
compared six models on 225 items, showing GPT -4 at 87.1% but L lama-2-70B -chat 
reaching only 48%  10. These studies demonstrate that model scale and targeted fine -tuning 
are critical for pharmacy -domain reasoning.  
 
We address two key questions: How do current LLMs of varying sizes perform on pharmacy 
questions, and can we improve their accuracy through external knowledge integration? We 
benchmarked eleven  existing LLMs on a 141 -question pharmacy dataset and developed a 
retrieval -augmented generation pipeline , DrugRAG,  that enhances model prompts with 
validated pharmacological information. Importantly, our approach works with models as -is, 
requiring no modification to their architecture or parameters. This study provides empirical 
evidence for how external structured knowledge integration enhances LLM performance on 
pharmacy -specific tasks.  
2. Methods  
2.1. Study Design and Question Set  
We evaluated eleven  language models using 141 multiple -choice questions from 
PharmacyExam, a NAPLEX preparation resource  11. Questions span the five NAPLEX content 
domains: Foundational Knowledge for Pharmacy Practice (25% ), Medication Use Process 
(25% ), Person -Centered Assessment and Treatment Planning (40% ), Professional Practice 
(5%), and Pharmacy Management and Leadership (5%) 7. This distribution emphasizes 
clinical reasoning and person -centered care, consistent with NAPLEX's content structure. 
Each question presented a medical vignette with four or five answer options.  
We prompted each model with the case scenario and options, mirroring how human test -
takers engage with the exam. We scored only the final answer choice (A, B, C, or D ), 
calculating accuracy as the percentage of correct responses out of 141 questions. An answer 

This manuscript is a preprint and has been submitted for peer review to Computer Methods and Programs in Biomedicine . The 
content may be updated following editorial review.  was correct if it matched the official key. Ambiguous answers or choices outside the 
provided options were marked incorrect.  
Figure 1. DrugRAG architecture for medication question answering.  
A clinical query (q) is first passed to o3, which extracts a focused 3 –6 term reasoning trace. These 
terms guide a medical evidence retriever that queries trusted drug information sources through the 
Medical Chat API, producing a structured evidence snippet (E). The target LLM receives the 
augmented input (q+E) and synthesizes the final clinical answer.  
 
2.2. Evaluated Language Models  
We selected eleven  existing models for evaluation based on public availability, biomedical 
relevance, and parameter scale diversity. Model parameters (measured in billions, denoted 
as "B" ) represent the number of trainable weights in the neural network; larger parameter 
counts generally indicate greater model capacity and knowledge. Our selection ranged from 
8B (8 billion parameters ) to over 70B parameters, as well as proprietary models with 
undisclosed sizes. We tested all models in their original form without any modifications to 
architecture, parameters, or fine -tuning.  
 
Open -source models (Bio-Medical Llama 3 8B  12, Llama 3.1 8B  13, Gemma 3 27B  14) provided 
accessible baselines for biomedical research. Proprietary models (GPT -4o 15, GPT -5 16, o3 17, 
o4 Mini  18, Gemini 2.0 Flash  19, Gemini 3 Pro 20, Claude Opus 4.5 21) represented top -tier 
performance. We included Medical Chat  22, a domain -specialized model, to examine the 


This manuscript is a preprint and has been submitted for peer review to Computer Methods and Programs in Biomedicine . The 
content may be updated following editorial review.  benefits of pharmacotherapy -focused training. Table 1 lists all evaluated models with their 
approximate parameter counts.  
 
We ensured fair comparison by using identical prompts across all models. For 
reproducibility, we standardized generation settings  in open -source models : temperature = 
0.2, top_p = 1.0, frequency penalty = 0.0, presence penalty = 0.0, and max_tokens = 512. 
These settings isolated model knowledge and reasoning capabilities from random 
generation variance.  
Table 1. Baseline Accuracy of Large Language Models  
Model  Parameter Size  Accuracy (% Correct ) 
Bio-Medical Llama 3  8B (8 billion ) 46%  
Llama 3.1  8B (8 billion ) 46%  
Gemma 3  27B (27 billion ) 61%  
Gemini 2.0 (Flash ) Not disclosed  72%  
Gemini 3 (Pro)  Not disclosed  75%  
o4 Mini  Not disclosed  76%  
GPT-4o Not disclosed  81%  
Medical Chat  Not disclosed  85%  
Claude Opus 4.5  Not disclosed  87%  
o3 Not disclosed  89%  
GPT-5 Not disclosed  92%  
Baseline accuracy represents performance without any external knowledge augmentation. 
Parameter sizes are approximate; "Not disclosed" indicates proprietary specifications. B = billion 
parameters.  
2.3. Three -Step Drug RAG Pipeline Development  
We developed a three -stage retrieval -augmented generation pipeline , DrugRAG,  to integrate 
structured drug knowledge with existing LLMs  (Figure 1) . This pipeline operates externally 
to the models and requires no modification of model architecture, parameters, or training. 
The three stages are:  
 
Step 1 - Reasoning Extraction: We used o3 to extract 3 -6 key medical concepts from each 
clinical query (q) . This reasoning trace captures core concepts without extraneous detail.  All 
reasoning traces for the 141 quest ions are provided in Supplementary Files S1 . 
 
Step 2 - Evidence Retrieval: We queried the Medical Chat API (Application Programming 
Interface ) with the reasoning trace, retrieving structured drug information (maximum 200 
words ) from validated sources including professional drug databases. The API returns a 
structured evidence snippet (E)  with drug name, indication, dosage limits, 
contraindications, and other relevant details.  The retrieved evidence/medical responses 
associated with each reasoning trace are provided in Supplementary File S2 . 
 
Step 3 - Augmented Prompting: We combined the clinical query (q) with the structured 

This manuscript is a preprint and has been submitted for peer review to Computer Methods and Programs in Biomedicine . The 
content may be updated following editorial review.  evidence snippet (E)  and prompted the target LLM. Instructions explicitly directed the 
model to ground its answer in the provided evidence, reducing hallucinations and aligning 
responses with medical consensus.  
 
This approach allows us to improve any LLM's pharmacy performance without retraining or 
modifying the model itself. We selected Medical Chat for evidence retrieval after evaluating 
open -source drug databases (OpenFDA  23, DrugCentral  24, DrugBank  25, RxNorm  26). While 
these resources provide structured drug data, they lack the clinical synthesis needed for 
exam -style questions. Medical Chat aggregates trusted sources and delivers contextually 
relevant evidence, functioning as an on -demand expert consultant.  
Figure 2. Effect of DrugRAG on LLM accuracy for pharmacy question -answering.  
DrugRAG increased accuracy for all evaluated models, with improvements of 7 –21 percentage points 
depending on model scale and architecture. The largest gains occurred in smaller models, but even 
advanced systems such as Gemma 3, Gemini 2.0 Flash, and Gemini 3 Pro benefited from externally 
retrieved drug evidence.  
2.4. Statistical Analysis  
We performed all analyses using Python 3.1 1 with NumPy, Pandas, Matplotlib, Seaborn, and 
Statsmodels . We defined accuracy as the proportion of correct responses. To compare 
model performance, we conducted two -sample, two -sided Z -tests for proportions using 
GPT -5 (92%) as the benchmark. Because all models answered the same fixed set of 141 
questions with binary outcomes (correct/incorrect), we approximated comparisons with 
pooled two -proportion Z -tests to evaluate accuracy differences . We considered p -values < 
0.05 statistically significant.  


This manuscript is a preprint and has been submitted for peer review to Computer Methods and Programs in Biomedicine . The 
content may be updated following editorial review.  3. Results and Discussion  
3.1. Baseline Model Performance Across Parameter Scales  
Table 1 shows substantial variation in accuracy across models of different parameter sizes. 
These baseline results reflect each model's inherent capabilities without any external 
knowledge augmentation:  
 
Bio-Medical Llama 3 8B  and L lama 3.1 8B both achieved 46%, the lowest accuracy in our 
benchmark. This demonstrates the difficulty of pharmacy content for models at this scale (8 
billion parameters ), even with domain -specific pre -training.  
 
Gemma 3 27B (27 billion parameters ) reached 61%, a notable improvement over the 8B 
models. This suggests both scale and architecture contribute to pharmacy reasoning 
capability.  
 
Gemini 2.0 Flash achieved 72%, showing that strong general -purpose reasoning can reach 
competitive performance on pharmacy questions.  
 
Gemini 3 Pr o reached 75%, indicating solid proficiency in medication -related decision -
making and question interpretation.  
 
o4 Mini scored 76%. Despite its smaller footprint, advanced fine -tuning appears to enhance 
test-taking strategies and logical reasoning.  
 
GPT -4o obtained 81%, reflecting sophisticated language understanding and broad training 
data.  
 
Medical Chat achieved 85%, the highest among domain -specialized models. This 
demonstrates the value of focused pharmacotherapy training.  
 
Claude Opus 4.5 scored 87%, demonstrating strong general -purpose reasoning and reliable 
performance on pharmacy -focused questions.  
 
o3 scored 89%, approaching expert -level performance and showing how advanced general 
models can excel on pharmacy tasks.  
 
GPT -5 achieved 92%, the highest performance in our evaluation. It demonstrates enhanced 
reasoning depth and sets the current benchmark for pharmacy question -answering.  
 
Our statistical analysis compared all models against GPT -5 using Z -tests for proportions 
(Table 2 ). Bio-Medical Llama 3 , Llama  3.1, Gemma 3, Gemini 2.0  Flash , Gemini 3 Pro, o4 
Mini, and GPT -4o performed significantly worse (p < 0.05 ). Medical Chat , Claude Opus 4.5  
and o3 did not differ significantly from GPT -5. 

This manuscript is a preprint and has been submitted for peer review to Computer Methods and Programs in Biomedicine . The 
content may be updated following editorial review.  Table 2. Statistical Comparison Against GPT -5 
Model  Accuracy (%) Z-score  p-value  Significance  
Bio-Medical 
Llama 3  (8B) 46 −8.35  < 0.001  Significant  
Llama 3.1 (8B) 46 −8.35  < 0.001  Significant  
Gemma 3 (27B) 61 −6.14  < 0.001  Significant  
Gemini 2.0 
(Flash ) 72 −4.37  < 0.001  Significant  
Gemini 3 (Pro)  75 −3.87  < 0.001  Significant  
o4 Mini  76 −3.66  < 0.001  Significant  
GPT-4o 81 −2.70  0.0069  Significant  
Medical Chat  85 −1.84  0.065  Not significant  
Claude Opus 4.5  87 −1.38  0.167  Not significant  
o3 89 −0.86  0.39  Not significant  
 
To better understand these baseline differences, we qualitatively examined common failure 
patterns across model outputs . Smaller models frequently exhibited formula -related errors, 
including forgetting standard pharmacotherapy equations, misapplying formulas, or 
producing different results across repeated attempts (e.g., in eAG calculation or weight -
based dosing). We also observed contextual misunderstandings, in which a model 
recognized familiar medical terms but misinterpreted the task, for example, returning the 
therapeutic target for HbA1c rather than  computing an eAG value. Another common issue 
involved logical and numerical inconsistencies: models made arithmetic mistakes, 
substituted incorrect parameters, mismatched units, or produced explanations that 
contradicted their final chosen answer, reflecting unstable multi -step reasoning. Finally, 
several models demonstrated medication -identification errors, such as repeatedly 
confusing Tavist (clemastine) with Tavist ND (loratadine).   
Our findings also reveal that model parameter size strongly correlates with pharmacy 
question performance when models are used without external knowledge augmentation. 
Among open -source models, only Gemma 3 27B exceeded 60% accuracy, while all models 
with fewer than 10 billion parameters remained below 50%. Bio-Medical Llama 3 and 
Llama  3.1 achieved similar performance despite different architectures, suggesting that 
domain -specific pretraining alone may not overcome size limitations at the 8B scale. 
However, Medical Chat's high performance (85% ) demonstrates that combining a larger 
scale with targeted training yields substantial benefits. We acknowledge that confounding 
factors in model design, architecture, and tuning prevent us from isolating individual 
contributions to baseline performance.  
3.2. Performance Improvements Through DrugRAG  
Our three -step RAG pipeline , DrugRAG,  improved accuracy across all five tested models 
(Table 3 ). These improvements result entirely from external knowledge integration, not 
from modifying the models themselves. We observed gains ranging from 7 to 21 percentage 
points, with smaller models showing the largest improvements  (Figure 2) . 

This manuscript is a preprint and has been submitted for peer review to Computer Methods and Programs in Biomedicine . The 
content may be updated following editorial review.   
Llama 3.1 8B improved from 46% to 67%, a 21 -point gain. This dramatic improvement 
demonstrates that 8 -billion -parameter models can approach the baseline performance of 
much larger models when provided with structured drug knowledge through our external 
pipeline.  
 
Bio-Medical Llama 3 8B  increased from 46% to 59%, a 13 -point improvement. While more 
modest than L lama 3.1, this still represents substantial enhancement through external 
knowledge integration.  
 
Gemma 3 27B rose from 61% to 71%, a 10 -point gain. Even models with 27 billion 
parameters benefit meaningfully from structured evidence provided externally.  
 
Gemini 2.0 Flash improved from 72% to 79%, a 7 -point increase. High -performing models 
still show measurable gains from evidence augmentation.  
Gemini 3 Pro increased from 75% to 84%, gaining 9 points. This suggests that providing 
targeted pharmacotherapy evidence improves the model's  baseline performance.  
 
These results demonstrate that DrugRAG  addresses specific information gaps across all 
model scales without requiring model modification. Smaller models gain more because they 
often lack specific pharmacological facts, and the evidence snippet provides crucial missing 
information. Larger models already possess broader knowledge but still benefit from 
grounding in authoritative, context -specific evidence delivered through our pipeline.  
Table 3. Performance with DrugRAG  
Model  Baseline 
Accuracy  Accuracy with RAG  Improvement  
Llama 3.1 (8B) 46%  67%  +21 points  
Bio-Medical Llama 3  (8B) 46%  59%  +13 points  
Gemma 3 (27B) 61%  71%  +10 points  
Gemini 2.0 (Flash ) 72%  79%  +7 points  
Gemini 3 (Pro) 75%  84%  +9 points  
 
DrugRAG  offers advantages beyond this specific benchmark. Because it operates externally 
to the models, it can be applied to any LLM without requiring access to model weights, 
retraining, or architectural modifications. The pipeline can be adapted for drug -interaction 
checks, guideline -based question -answering, or patient -specific dosing support , as well as 
any application requiring concise, trusted evidence to inform model reasoning. The three -
step design ensures evidence relevance (through reasoning extraction ), reliability (through 
validated sources ), and effective integration (through structured prompting ), all achieved 
through prompt engineering and evidence injection rather than model modification.  

This manuscript is a preprint and has been submitted for peer review to Computer Methods and Programs in Biomedicine . The 
content may be updated following editorial review.  4. Limitations  
Our study has several limitations. First, while the 141 -question benchmark covers broad 
pharmacy content aligned with NAPLEX domains, we did not conduct formal analysis of 
question difficulty distribution within each domain. This limit s claims of comprehensive 
topical representation. Second, we evaluated only multiple -choice questions, which do not 
capture real -world complexity such as multi -step therapeutic planning or open -ended 
decision making. Generalizability to such tasks remains untested.  Third, we accessed 
proprietary models (GPT -4o, GPT -5, Medical Chat ) via closed APIs without insight into 
internal architecture, training data, or updates. Using Medical Chat for both benchmarking 
and knowledge retrieval may introduce circularity that future experiments should address. 
Fourth, our RAG pipeline introduces practical constraints. API -based calls incur latency and 
cost, and the three -step setup may be difficult to scale in low -resource or real -time 
applications. Additionally, while we standardized prompts, we did not test model 
robustness to paraphrased or reworded question variants, which could affect consistency in 
real -world use.  
 
5. Conclusion  
We benchmarked eleven  large language models of varying parameter sizes on pharmacy 
question -answering tasks, revealing wide performance variation tied to model scale and 
training. We developed a three -step RAG pipeline , DrugRAG,  that integrates structured drug 
knowledge externally, achieving 7-21 % point accuracy improvements across all tested 
models without modifying their underlying architecture or parameters. Our findings 
demonstrate that external knowledge integration can substantially enhance LLM 
performance on pharmacy -focused tasks, offering a practical approach that works with any 
existing model. This pipeline  provides a scalable method for enhancing pharmacy AI 
applications with evidence -based information.  
 
6. Declaration of Competing Interest  
The authors declare no conflicts of interest related to this study.  
 
7. Data Availability  
Aggregated model predictions and analysis code are available from the corresponding 
author on reasonable request.  
 
8. Funding  
This research received no specific grant from any funding agency in the public, commercial, 
or not -for-profit sectors.  

This manuscript is a preprint and has been submitted for peer review to Computer Methods and Programs in Biomedicine . The 
content may be updated following editorial review.  9. Ethics statement  
This study did not involve human participants, animals, or access to identifiable patient data 
and therefore did not require institutional ethics committee approval . 
 
10. Declaration of Generative AI and AI -assisted technologies  
During manuscript preparation, the authors used ChatGPT (GPT -4o) to improve readability 
and language. The authors reviewed and edited all content and take full responsibility for 
the publication.  
 
Supplementary Materials  
Supplementary File S1 contains the o3 -generated reasoning traces for all 141 questions.  
Supplementary File S2 contains the retrieved medical responses used as structured 
evidence in the DrugRAG pipeline.  Supplementary materials are provided alongside the 
arXiv submission.  
 
11. References  
1. Roosan D, Padua P, Khan R, Khan H, Verzosa C, Wu Y. Effectiveness of ChatGPT in 
clinical pharmacy and the role of artificial intelligence in medication therapy management. 
Journal of the American Pharmacists Association . 2024;64(2):422 -428. e8. doi: 
https://doi.org/10.1016/j.japh.2023.11.023  
2. Pais C, Liu J, Voigt R, Gupta V, Wade E, Bayati M. Large language models for 
preventing medication direction errors in online pharmacies. Nature medicine . 
2024;30(6):1574 -1582. doi: https://doi.org/10.1038/s41591 -024 -02933 -8 
3. Brin D, Sorin V, Vaid A, et al. Comparing ChatGPT and GPT -4 performance in USMLE 
soft skill assessments. Scientific Reports . 2023;13(1):16492. doi: 
https://doi.org/10.1038/s41598 -023 -43436 -9 
4. Jin HK, Lee HE, Kim E. Performance of ChatGPT -3.5 and GPT -4 in national licensing 
examinations for medicine, pharmacy, dentistry, and nursing: a systematic review and 
meta -analysis. BMC medical education . 2024;24(1):1013. doi: 
https://doi.org/10.1186/s12909 -024 -05944 -8 
5. Schindel TJ, Yuksel N, Breault R, Daniels J, Varnhagen S, Hughes CA. Perceptions of 
pharmacists' roles in the era of expanding scopes of practice. Research in Social and 
Administrative Pharmacy . 2017;13(1):148 -161. doi: 
https://doi.org/10.1016/j.sapharm.2016.02.007  
6. Newton DW, Boyle M, Catizone CA. The NAPLEX: evolution, purpose, scope, and 
educational implications. American journal of pharmaceutical education . 2008;72(2):33. doi: 
https://doi.org/10.5688/aj720233  

This manuscript is a preprint and has been submitted for peer review to Computer Methods and Programs in Biomedicine . The 
content may be updated following editorial review.  7. NABP. NAPLEX® Competency Statements and Test Specifications. National 
Association of Boards of Pharmacy. Accessed May 1, 2025, https://nabp.pharmacy/wp -
content/uploads/NAPLEX -Content -Outline.pdf  
8. Angel M, Patel A, Alachkar A, Baldi P. Clinical knowledge and reasoning abilities of 
large language models in pharmacy: A comparative study on the naplex exam. IEEE; 2023:1 -
4. 
9. Ehlert A, Ehlert B, Cao B, Morbitzer K. Large Language Models and the North 
American Pharmacist Licensure Examination (NAPLEX) Practice Questions. American 
Journal of Pharmaceutical Education . 2024;88(11):101294. doi: 
https://doi.org/10.1016/j.ajpe.2024.101294  
10. Angel M, Xing H, Patel A, Alachkar A, Baldi P. Performance of Large Language Models 
on Pharmacy Exam: A Comparative Assessment Using the NAPLEX. bioRxiv . 2023:2023.12. 
06.570434. doi: https://doi.org/10.1101/2023.12.06.570434  
11. PharmacyExam.com. Accessed April 1, 2025, https://www.pharmacyexam.com  
12. Bio-Medical Llama 3 8B. Hugging Face. Accessed April 1, 2025, 
https://huggingface.co/ContactDoctor/Bio -Medical -Llama -3-8B 
13. Llama 3.1 8B Instruct. Hugging Face. Accessed April 1, 2025, 
https://huggingface.co/meta -llama/Llama -3.1-8B-Instruct  
14. Gemma 3 27B IT. Hugging Face. Accessed June 1, 2025, 
https://huggingface.co/google/gemma -3-27b-it 
15. OpenAI GPT -4o Model. Accessed May 1, 2025, 
https://platform.openai.com/docs/models/gpt -4o 
16. OpenAI. GPT -5 Model. Accessed October 1, 2025, 
https://platform.openai.com/docs/models/gpt -5 
17. OpenAI o3 Model. Accessed May 1, 2025, 
https://platform.openai.com/docs/models/o3  
18. OpenAI o4 Mini. Accessed May 1, 2025, 
https://platform.openai.com/docs/models/o4 -mini  
19. Gemini 2.0 Flash Documentation. Google AI Developer. Accessed April 1, 2025, 
https://ai.google.dev/gemini -api/docs/models#gemini -2.0-flash  
20. Gemini 3 Pro Documentation. Google AI Developer. Accessed December 1, 2025, 
https://ai.google.dev/gemini -api/docs/models#gemini -3-pro 
21. Claude Opus 4.5. Accessed December 1, 2025, 
https://www.anthropic.com/claude/opus#claude -opus -4.5 
22. Medical Chat. Medical Clinical QA System. Accessed May 1, 2025, 
https://medical.chat -data.com  
23. OpenFDA API. Accessed June 1, 2025, https://open.fda.gov/apis/  
24. DrugCentral OpenAPI. Accessed June 1, 2025, https://drugcentral.org/OpenAPI  
25. DrugBank Online. API Documentation. Accessed June 1, 2025, 
https://docs.drugbank.com/  
26. RxNorm API. Accessed June 1, 2025, 
https://lhncbc.nlm.nih.gov/RxNav/APIs/RxNormAPIs.html  
 