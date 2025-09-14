# FHIR-RAG-MEDS: Integrating HL7 FHIR with Retrieval-Augmented Large Language Models for Enhanced Medical Decision Support

**Authors**: Yildiray Kabak, Gokce B. Laleci Erturkmen, Mert Gencturk, Tuncay Namli, A. Anil Sinaci, Ruben Alcantud Corcoles, Cristina Gomez Ballesteros, Pedro Abizanda, Asuman Dogac

**Published**: 2025-09-09 13:10:49

**PDF URL**: [http://arxiv.org/pdf/2509.07706v1](http://arxiv.org/pdf/2509.07706v1)

## Abstract
In this study, we propose FHIR-RAG-MEDS system that aims to integrate Health
Level 7 Fast Healthcare Interoperability Resources (HL7 FHIR) with a
Retrieval-Augmented Generation (RAG)-based system to improve personalized
medical decision support on evidence-based clinical guidelines, emphasizing the
need for research in practical applications. In the evolving landscape of
medical decision support systems, integrating advanced technologies such as RAG
and HL7 FHIR can significantly enhance clinical decision-making processes.
Despite the potential of these technologies, there is limited research on their
integration in practical applications.

## Full Text


<!-- PDF content starts -->

1 
 FHIR -RAG -MEDS : Integrating HL7 FHIR with Retrieval -Augmented Large Language Models for 
Enhanced Medical Decision Support  
Yildiray  KABAK1,*, Gokce B. LALECI ERTURKMEN1,, Mert GENCTURK1, Tuncay NAMLI1, A. Anil 
SINACI1, Ruben ALCANTUD CORCOLES2,3, Cristina GOMEZ BALLESTEROS2,3, Pedro 
ABIZANDA2,3,4, Asuman DOGAC1 
1SRDC Software Research & Development and Consultancy Corporation, Ankara, Turkey  
2Geriatrics Department, Complejo Hospitalario Universitario de Albacete, Albacete, Spain  
3CIBER de Fragilidad y Envejecimiento Saludable (CIBERFES), Instituto de Salud Carlos III, Madrid, 
Spain  
4Facultad de Medicina de Albacete, Universidad de Castilla -La Mancha, Albacete, Spain  
* Correspondence:  
Yildiray Kabak , PhD , yildiray@srdc.com.tr  
Keywords:  Medical Decision Support; HL7 FHIR; Retrieval -Augmented Generation; Clinical Guidelines; 
Large Language Models; Patient -Specific Data; Evidence -Based Medicine; Llama 3.1  
 
Abstract  
Objectives : In this study, we propose FHIR -RAG -MEDS system that aims to integrate Health Level 7 Fast 
Healthcare Interoperability Resources  (HL7 FHIR ) with a Retrieval -Augmented Generatio n (RAG )-based 
system to improve personalized medical decision support on evidence -based clinical guidelines , 
emphasizing the need for research in practical applications.  In the evolving landscape of medical decision 
support systems, integrating advanced technologies such as RAG and HL7 FHIR can significantly enhance 
clinical decision -making processes. Despite the potential of these technologies, there is limited research on 
their integration in practical applications.  

2 
 Methods:  We explore how this integration can enhance the accuracy and relevance of RAG -based  clinical 
recommendations by leveraging patient -specific data from HL7 FHIR server s and advanced language 
models . The FHIR -RAG -MEDS system utilizes cleaned clinical guidelines stored in a vector database and 
responds  to patient queries using Llama 3.1 8B open -source  large language model  (LLM) .  
Results: The integration of HL7 FHIR with the RAG -based system shows significant improvements in 
generating accurate and contextually relevant medical recommendations. The evaluation against  other 
medical LLMs, such as Meditron 3, OpenBioLLM, and BioMistral , through statistical metrics (BERTScore, 
ROUGE  and METEO R), followed by LLM -evaluator s (Prometheus 2 and RAGAS ), demonstrates strong 
performance, with particular enhancements noted through  human expert feedback. Visualization of the 
results offers detailed insigh ts into the system’s strengths and areas for improvement.  
Conclusions: By addressing the limitations of traditional medical LLMs and incorporating real -time patient 
data through HL7 FHIR  and evidence -based clinical guidelines , our approach offers a promising solution 
for improving clinical decision support. This integration represents a step forward in bridging the gap 
between static knowledge models and dynamic, patient -specific needs, ultimately contributing to better 
clinic al outcomes and more effective healthcare delivery.  
1 BACKGROUND  
In recent years, the field of medical informatics has seen significant advancements with the introduction of 
medical large language models (LLMs). These models, powered by artificial intelligence, have 
demonstrated remarkable capabilities in understanding and generating medical text, providing valuable 
assistance in clinical decision -making, diagnostics, and patient care. Prom inent examples include models 
such as Meditron  [1], BioMistral  [2] and OpenBioLLM  [3], which have shown considerable promise in 
various medical applications.  However, despite these advancements, the inherent limitations of medical 
LLMs highlight the need for more robust solutions.  

3 
 While medical LLMs have achieved impressive results, they are not without limitations. These models 
primarily rely on their training data and do not have inherent mechanisms to incorporate real -time patient -
specific information or to update their knowledge  base dynamically  [4] especially to focus on a specific 
context . As a result, their recommendations may lack contextual relevance and may not fully account for 
the nuances of individual patient cases , recent developments in medical guidelines  or customized 
approaches in local guidelines . This can lead to less accurate or suboptimal recommendations in a clinical 
setting.  Retrieval -Augmented Generatio n (RAG ) represents a significant advancement over traditional 
LLM approaches by addressing some of these limitations , by combin ing the strengths of retrieval -based 
methods with generative capabilities, enabling the system to access and incorporate specific, up -to-date 
information from a curated knowledge base  [5]. This approach allows the RAG system to generate 
recommendations that are not only based on broad training data but also tailored to the context of the 
patient's medical history and current /local  clinical guidelines . The main advancements that RAG systems  
offer  over medical LLMs are as follows.   
1. Accuracy and Trustworthiness:  Medical LLMs rely solely on their training data, which may become 
outdated or inaccurate as medical knowledge evolves  [6]. This can lead to the generation of incorrect 
or misleading information  [7], especially in rapidly changing medical fields.  RAG applications, on the 
other hand, are able to retrieve answers from validated medical guidelines, databases, or research 
papers, ensuring that the generated responses are based on up -to-date and trustworthy information  
[8],[9]. This helps reduce the risk of hallucination, where LLMs may generate plausible but incorrect 
answers.  This is also crucial for enabling decision support tailored to national and regional clinical 
guidelines that address local needs . RAG systems, by retrieving information from validated , selected  
medical guidelines and sources, are better aligned with clinical standards, increasing compliance and 
reducing legal risks in medical practice . 
2. Evidence -Based Responses:  Medical LLMs often generate responses without directly citing their 
sources, making it harder for clinicians to trust the recommendations without external verification. The 

4 
 lack of transparency can be a significant issue in clinical environments  [10]. RAG systems combine 
Generative Artificial Intelligence ( AI) with retrieved evidence, allowing the system to generate 
responses while citing medical sources. This is particularly important in healthcare, where decisions 
must be supported by evidence to ensure clinical validity  [11].  
3. Scalability and Flexibility:  Medical LLMs require extensive training data for each medical field or 
domain, which may not always be feasible and can result in inefficiencies when handling cross -domain 
knowledge.  RAG systems provide greater flexibility by combining generative capabilities with retrieval 
from diverse and specialized datasets. This allows the application to handle multiple medical domains 
and specific guidelines efficiently.  
4. Reduced Computational Costs:  Medical LLMs require heavy computational resources to generate 
responses, particularly as model sizes increase, leading to slower response times and higher 
infrastructure costs  [12]. RAG systems reduce computational burdens by retrieving relevant 
information from external databases before generating responses, which can be more efficient than 
relying entirely on a large LLM for every query.  
The integration of RAG with LLMs in the medical domain is still in its early stages  [11]. Additionally , to 
fully harness the potential of LLMs in routine medical practice, they need to access historical Electronic 
Health Record (EHR) data. This access enables LLMs to enhance clinical decision support by offering more 
personalized and contextually relevant re commendations based on a patient's medical history. Studies such 
as [13], [14]  have reviewed the integration  of LLMs with EHRs, outlining both the potential and limitations  
of this approach.  They point out that while LLMs can assist with tasks such as diagnostic support and 
treatment planning, the challenge lies in effectively incorporating patient data into these models.  
Our work addresses this gap by integrating Health Level 7 Fast Healthcare Interoperability Resources  (HL7 
FHIR ) into an RAG  system specifically for medical professionals. By leveraging the standardized data 
formats provided by HL7 FHIR, our system can retrieve up -to-date patient medical summaries and utilize 
them to retrieve personalized suggestions from  evidence -based clinical guidelines. This ensures more 

5 
 accurate, personalized, and contextually relevant recommendations, overcoming the limitations regarding 
the static nature of LLMs that lack dynamic access to patient data.  Although the re are studies on the use of 
LLMs on HL7 FHIR, they use it to increase patient’s health literacy  on their medical records  [15] and to 
convert the unstructured medical records into HL7 FHIR  resources  [16],[17]. In our work, HL7 FHIR 
integration allows our system to provide real -time decision support , enhances interpretability , and enable s 
personalization of recommendations, making it better  suited for clinical environments where personalized  
guidance based on medical standards is critical.  
Statement of Significance  
Problem  In the evolving landscape of medical decision support systems, integrating 
advanced technologies such as Retrieval -Augmented Generatio n (RAG ) and Health 
Level 7 Fast Healthcare Interoperability Resources  (HL7 FHIR ) can significantly 
enhance clinical decision -making processes. Despite the potential of these 
technologies, there is limited research on their integration in practical applications.  
What is 
already 
known?  Medical large language models (LLMs ), powered by artificial intelligence, have 
demonstrated remarkable capabilities in understanding and generating medical text, 
providing valuable assistance in clinical decision -making, diagnostics, and patient 
care. 
What this 
Paper Adds?  In this study, we propose FHIR -RAG -MEDS system that aims to integrate HL7 
FHIR  with a RAG -based system to improve personalized medical decision support 
on evidence -based clinical guidelines . We explore how this integration can enhance 
the accuracy and relevance of RAG -based clinical recommendations by leveraging 
patient -specific data from HL7 FHIR server s and advanced language models .  
Who would 
benefit from 
the new 
knowledge in 
this paper  
 
 Medical Informatics specialists  can benefit from the detailed architectural decisions 
presented to replicate the study to build decision support applications in a different 
clinical domain.  
Clinicians can benefit from the personalized decision support based on evidence -
based guidelines.  

6 
 2 RELATED WORK  
Retrieval -Augmented Generation (RAG) was proposed by  [4] to enhance generation performance on 
knowledge -intensive tasks by integrating retrieved relevant information. RAG not only mitigates the 
problem of hallucinations , as LLMs are grounded in given contexts , but it can also provide up -to-date 
knowledge that might not be encoded within  the LLMs. In the medical  domain , there have already been 
various explorations into the use of LLMs with RAG . One prominent study highlights the use of RAG in 
LLMs  tailored for specific medical domains, such as nephrology  [5]. In this context , RAG allows these 
models to access real -time external medical databases, including clinical guidelines, to provide contextually 
relevant and up -to-date information. This approach significantly enhances the model's performance by 
grounding responses in val idated, evidence -based guidelines, which is critical in healthcare settings where 
precision is vital. For example , this methodology helps the model generate more accurate recommendations 
for conditions like kidney disease . Another interesting example is the experimental LLM framework, 
Almanac, which integrate s RAG functionalities specifically with clinical guidelines and medical treatments  
[11]. This framework was tested against existing models , such as ChatGPT , and demonstrated superior 
performance, particularly in cardiology, where it provided more accurate responses compared to standard 
generative models . However, these studies  do not focus on EHR integration  and personalization of 
recommendations .  
Studies such as  [13],[14] have reviewed the integration of LLMs with EHRs . The integration is centered 
around seven identified topics: named entity recognition, information extraction, text similarity, text 
summarization, text classification, dialogue system, and diagnosis and prediction.  However , none  of the 
examined studies use standardized data models to access medical records . A notable patient -centered 
example was introduced by  [15], which combines GPT -4 with HL7 FHIR  to enhance the health literacy of 
a diverse patient population . However, this approach is aimed at patients and does not utilize RAG . Our 
system, in contrast, is designed specifically for medical professionals and is built on evidence -based clinical 

7 
 guidelines. This ensures that recommendations are not only accurate and trustworthy but also up  to date, 
overcoming the limitations of static, pre -trained models like GPT -4. 
In summary, medical RAG systems can improve EHR summarization [18], clinical decision -making [5], 
[11], [12], [19] and clinical trial screening [20], but their evaluations are not comprehensive [21]. 
Furthermore, they do  not access patients’ medical records , which hinders personalized recommendations.  
3 METHODS  
This study aims to develop the FHIR -RAG -MEDS system by integrat ing HL7 FHIR with a RAG -based 
system to improve personalized medical decision support based on evidence -based clinical guidelines. As 
a case study, we have chose n a European Commission ( EC)-funded research project, CAREPATH [22], 
[23] which aims to deliver a patient -centered integrated care platform to meet the needs of older patients 
with multimorbidity, including mild cognitive impairment (MCI) or mild dementia (MD) , based on the 
recommendations of evidence -based guidelines. In the CAREPATH project, a Clinical Reference Group 
(CRG) , formed by the project ’s clinical partners , has analyzed  several clinical guidelines  that address  the 
needs of common comorbidities in this patient group and created a consolidated guideline  providing  advice, 
information, and actions for the management of MCI /MD , physical exercise, nutrition and hydration, 
common use of drugs, coronary artery disease, heart failure, hypertension, diabetes, chronic kidney disease, 
chronic obstructive pulmonary disease (COPD), stroke, sarcopenia , and frailty  [24]. Subsequently , this 
consolidate d guideline was analyzed  by software engineers to create clinical deci sion support services that 
automate the recommendations from the guideline, deliver ing personalized suggestions for care plan goal s 
and activities to healthcare professionals  [25]. Although this method provides a solid  and reliable 
mechanism to create clinical decision support for healthcare professionals, the process requires significant  
time and resources. In this research , we aim to explore the value of  utilizing advanced large language models 
and the RAG methodology to accelerate  the delivery of personalized clinical recommendations to 
healthcare professionals based on a selected curation of evidence -based guidelines. As a case study, we 

8 
 have focused on a subset of the CAREPATH consolidated guideline , covering recommendations for 
managing  hypertension, COPD , sarcopenia, and MCI/MD .  
3.1 System Architecture  
The FHIR -RAG -MEDS system architecture (Figure 1) integrates HL7 FHIR with a RAG framework to 
deliver real -time, evidence -based clinical decision support. The architecture is designed to retrieve patient -
specific data from an HL7 FHIR server, process clinical guidelines stored in a vector database, and ge nerate 
recommendations using LLMs. The system flow is divided into three  core components , as depicted in 
Figure 1 : preprocessing, data retrieval  & query processing, and RAG execution : 
• Preprocessing : This  step is required to prepare the guideline text to be processed by the RAG 
framework . It includ es segment ing the text  into smaller chunks and generating  embeddings  from 
these chunks to populate the vector database , which serves  as an input to the RAG  system .  
• Data Retrieval and Query Processing:  In this phase, patient context data is retrieved from the 
FHIR server  and pre-process ed to create a medical summary , which is  integrated into the  RAG 
system as input. SMART on FHIR (Substitutable Medical Applications and Reusable Technologies 
on Fast Healthcare Interoperability Resources)  specifications  are employed at this stage . SMART 
on FHIR  is a widely adopted framework that enhances interoperability in healthcare by providing 
standardized, secure, and scalable solutions for accessing and exchanging clinical data  [26]. Unlike 
proprietary or isolated solutions, SMART on FHIR fosters a robust ecosystem where applications 
can interact with any FHIR -compliant server, ensuring consistent access to patient data across 
platforms.  This framework is particularly  valuable for clinical decision support systems like FHIR -
RAG -MEDS, where real -time access to patient -specific data is crucial for generating accurate, 
evidence -based recommendations.  Next , the user’s plain text query (e.g., ‘What should be the 
approach to pharmacological  treatment for Patient X ?’) is merged with the generated medical 
summary text (e.g., ‘Patient X is an 85 -year-old female patient with cognitive impairment and a 

9 
 history of injurious falls. She is diagnosed with hypertension ’) and processed to extract 
embeddings , which are stored in the vector database.  
• RAG Execution : This step integrates the generative capabilities of LLMs with the retrieval of 
specific, up -to-date information from the vector database  containing  pre-processed guidelines. It 
matches the embeddings extracted from the patient ’s medical summary and the user query with the 
guideline embeddings in the vector storage to identify the closest matching vectors. These matching 
vectors , along with  the medical summary , are then used by the LLM to generate a  response to the 
clinician ’s query .    
 
Figure 1 – Overall architecture of the FHIR -RAG -MEDS  system  
In the following subsections, we provide a detailed explanation  of these steps , including our design 
decisions and the tools utilized.  
3.1.1 Preprocessing  
Handling semi -structured data, such as PDFs containing a mix of text and tables, presents specific 
challenges for traditional RAG systems. First, text  splitting may inadvertently break apart tables, causing 
data corruption during retrieval. Second, incorporating tables into the data complicates semantic similarity 
searches [8]. To address this, clinical documents in various formats, including PDFs, should  be converted to 
a text -based format to ensure compatibility with the RAG framework  [19]. 


10 
 Once converted, the processed texts are segmented into smaller chunks for embedding and retrieval. Tools 
from the Langchain library  [27] are employed in this splitting process, typically creating chunks of arbitrary 
sizes with slight overlaps. Determining the optimal chunk size for healthcare applications is a nuanced task 
that requires qualitative evaluation. In our study, we used Python 3 .12 along with Langchain’s 
DirectoryLoader , RecursiveCharacterTextSplitter , and TextLoader  for reading text -based clinical 
guidelines. We established a chunk size of 1200 units, with a 100 -unit overlap, based on Ollama’s “mxbai -
embed -large” embedding model . 
Efficient data retrieval also depends on robust data storage solutions. Vector storage, inspired by deep 
learning techniques, condenses information into high -dimensional vectors, enhancing retrieval 
performance. The key metrics for evaluating vector storag e include cosine similarity, Euclidean distance, 
and dot product. Cosine similarity is particularly effective for semantic searches and is preferred in 
healthcare applications because it focuses on the angle between vectors, highlighting content similarity  
[12]. Conversely , Euclidean distance is better suited for quantitative assessments. Embedding, the process 
of converting content into numerical vectors for machine learning models, is essential for transforming 
preprocessed healthcare data into vectors. For this study, we  used Chroma  [28] as our vector storage solution 
with cosine similarity, and as previously mentioned, employed Ollama’s “mxbai -embed -large” model for 
embedding.  
3.1.2 Data Retrieval  and Integration with Existing E HR Systems  
The architecture begins with the HL7 FHIR server, which stores structured patient data following the FHIR 
standard. To enhance interoperability and scalability, a "SMART on FHIR" -based integration has been 
developed for the FHIR -RAG -MEDS system. This integration leverages the SMART on FHIR framework, 
which provides a standard set of specifications for building secure , interoperable healthcare applications . 
These applications can seamlessly connect with any FHIR server that conform s to these standards.  
The integration was implemented using Angular, a modern and responsive web development framework, 
to create a user -friendly interface . This Angular -based application  communicates with the FHIR -RAG -

11 
 MEDS system’s endpoints for real-time patient data interpretation and clinical query processing. By 
adhering to SMART on FHIR specifications, the integration ensures compatibility with various  FHIR 
servers while  simplif ying authentication, authorization, and data exchange processes.  
Key components of the SMART on FHIR integration include  the following:  
1. Authentication and Authorization : The system employs OAuth 2.0 protocols, as defined by 
SMART on FHIR, to securely authenticate users and authorize access to patient data. This ensures 
that sensitive medical information remains  protected while maintaining seamless usability.  
2. Data Retrieval and Management : The integration enables real -time retrieval of patient 
information, such as demographics, medications, conditions, and observations, from any compliant 
FHIR server. All recent condition, medication, and observation resources are retrieved individually 
and then combined into a FHIR bundle for interpretation by the FHIR -RAG -MEDS system.  
3. Interoperable Query Processing : Once patient data is retrieved and summarized, it is sent to the 
FHIR -RAG -MEDS system for interpretation and recommendation generation  as explained in the 
next section . The system’s query endpoints provide evidence -based clinical suggestions tailored to 
the patient’s specific medical context.  
4. Extensibility and Compatibility : The SMART on FHIR integration is designed to be extensible, 
allowing it to support additional use cases, such as integration with third -party applications, EHR 
systems, and telehealth platforms. By adhering to open standards, it ensures compatibility acr oss 
diverse healthcare ecosystems.  
3.1.3 Processing  FHIR Bundles   
When a query is initiated to retriev e recommendations  from evidence -based guidelines for a specific patient , 
relevant patient information  —such as demographics, medical history, medications, and conditions —is 
retrieved from the FHIR server  once Smart on FHIR based integrating is achieved . This retrieval is 
performed  through FHIR -compliant queries using the appropriate APIs. By leveraging standardized FHIR 

12 
 resources, the system ensures interoperability and consistency across different healthcare environments.  
Once the patient data is retrieved  in FHIR JSON format , it must  be processed and organized into a format 
suitable for integration with the RAG system. For this , we use  the Llama 3.1  8B [29] model  with the a 
prompt template  (see Supplementary material) to convert the JSON into a  textual medical summary , 
enabling its inclusion in the LLM context . 
The prompt ensures that the retrieved data is transformed into a concise and clear summary, ready for 
interpretation by the system without overwhelming the user with technical jargon. By filtering and 
summarizing the patient's medical information in this way, the system enhances usability while maintaining 
the factual accuracy and relevance necessary for medical decision -making.   
The Llama 3.1 8B model has been  selected for interpreting FHIR bundles based on several key factors. 
Testing with an example FHIR bundle set  has demonstrated that  the model performs well  in generating 
accurate and concise medical summaries. Its relatively small size, up -to-date architecture, and popularity 
within the community further supported  its selection. As an open -source model, Llama 3.1 8B  is also easily 
deployable, even in resource -limited environments such as laptops. Furthermore , the ability to ru n the 
model within a local network improves  security and privacy, ensuring that confidential health data remains 
protected  without relying on  external cloud services.  
3.1.4 RAG Execution  
The core of the system lies in the RAG module, which combines the generative capabilities of LLMs with 
the retrieval of specific, up -to-date information from a pre -processed knowledge base. Clinical guidelines, 
which are cleaned and stored in a vector data base, are indexed to enable fast retrieval of evidence -based 
recommendations  in the previous step .  
The RAG execution functions as an intermediary, identifying  the most relevant chunks in response to user 
queries. This involves using the same embedding model to convert user queries into vector s, which are then 
filter ed through the Vector Storage to locate  the closest matching vectors.  The number of chunks retrieved —
adjustable through a parameter commonly referred to as 'k' —can be set to any desired value.  In our 

13 
 implementation , we set 'k' to 4 , determin ing the number of knowledge chunks retrieve d. After that , we 
combined these chunks with the FHIR Bundle textual interpretation generated in the previous step and 
created a context to be sent as an input to the LLM. The system uses Llama 3.1  8B, installed via  Ollama on 
a local server , to generate responses to clinician queries.  
The integration of FHIR with RAG allows the system to generate personalized clinical recommendations 
by combining static, evidence -based guidelines with dynamic, real -time patient data. This architecture 
ensures that the recommendations are both contextually relevant to the individual patient's health profile 
and aligned with the latest medical standards.  
3.2 Evaluation Framework  
Our evaluation starts with developing questions and their corresponding answers based on the guidelines 
provided . For this purpose , we tasked our evaluation panel of physicians  to generate  questions  and answers  
related to their daily clinical practices. Questions range d from simple ones to more complex scenario -based  
questions , e.g., “I have a patient with diabetes and hypertension. What should be the initial drug therapy? ”. 
In total , we compiled approximately  70 questions  and answers , regarded as ground truths . 
After obtaining the answers to these  questions from our RAG system,  in the first step of the evaluation,  we 
compared  them  with the ground truth answers using  text similarity metrics  Recall -Oriented Understudy for 
Gisting Evaluation (ROUGE)  [30], Metric for Evaluation of Translation with Explicit Ordering (METEOR)  
[31] and BERTScore  [32]. In text comparisons, BERTScore is used to evaluate semantic accuracy , while 
ROUGE/METEOR are used to assess  word -level matching . Specific ally: 
• BERTScore  can be the primary metric  because , in healthcare, capturing the correct meaning  of 
an answer is more important than exact word matches.  
• ROUGE  can be used to ensure that the model captures all necessary key terms  from the guideline.  
• METEOR  can be a good middle -ground metric  if flexibility  is needed  in language while still 
maintaining word overlap.  

14 
 In this study, we opted not to use the BLEU  (Bilingual Evaluation Understudy) [33] metric for evaluating 
the quality of model -generated medical summaries. Although BLEU is widely used for machine translation 
and other natural language generation tasks, it can be overly strict when applied to  healthcare data. BLEU 
measures the degree of n -gram overlap between generated and reference text, which works well for tasks 
reliant on  literal word matching. However, in healthcare, accurate interpretation and relevance of 
information are far more critical than exact phrasing. Medical summaries often  vary in terminology, 
phrasing, or style while still conveying the same clinical meaning. The strictness of BLEU in penalizing 
even slight variations in wording can result in  misleading evaluations of model performance in this context. 
Therefore , more appropriate metrics that focus on semantic accuracy and clinical relevance are preferred 
for evaluating health -related outputs.  
These scores are developed to measure word overlap, sentence structure similarity, and semantic coherence 
but not factual correctness. For clinical questions, factual correctness is the most important feature. This is 
an important challenge that should be addressed , as current responses could appear lexically comparable to 
a reference answer but fail to capture the factual  information necessary to guide clinical care.  This can result 
in high scores  for responses that are factually incorrect (false positives ) or low scores for  accurate responses 
that are phrased differently than the reference (false  negatives).  While useful for certain aspects of 
evaluation, these  metrics fail to  capture the nuances of medical relevance, completeness, and contextual 
correctness in the answers provided by the LLM. This limitation underscores the persistent need for expert 
physician oversight in the evaluation process , i.e., human -in-the-loop.  
For auto mated grading of our RAG generated responses , we apply a recent approach called “LLM -as-a-
Judge”  [34],  where LLM -evaluators are used to evaluate the quality of another LLM’s response . For this 
approach, we employed  two tools . The first one is  Prometheus 2  [35], which is a finetuned evaluator (based 
on Llama 3.1 8B LLM ) that performs fine -grained evaluation of text responses based on user -defined score 
rubrics. Prometheus  2 takes  as input the instructions, score rubric, response to evaluate, and a gold reference 

15 
 answer, making it a referenced -based evaluator. Then, it scores the response to evaluate and returns text 
feedback.  The prompt used by Prometheus 2 is as presented in Table 1. 
Table 1 – The Prompt used by Prometheus 2  
###Task Description:  
An instruction (might include an Input inside it), a response to evaluate, a 
reference answer that gets a score of 5, and a score rubric representing a 
evaluation criteria are given.  
1. Write  detailed feedback that assess es the quality of the response strictly 
based on the given score rubric, not evaluating in general.  
2. After writing a feedback, write a score that is an integer between 1 and 
5. You should refer to the score rubric.  
3. The output format should look as follows: "Feedback: {{write a feedback 
for criteria}} [RESULT] {{an integer number between 1 and 5}}"  
4. Please do not generate any other opening, closing, and explanations. Be 
sure to include [RESULT] in your output.  
 
###The instruction to evaluate:  
{instruction}  
 
###Response to evaluate:  
{response}  
 
###Reference Answer (Score 5):  
{reference_answer}  
 
###Score Rubrics:  
[Is the response correct, accurate, and factual based on the reference 
answer?]  
Score 1: The response is completely incorrect, inaccurate, and/or not factual.  
Score 2: The response is mostly incorrect, inaccurate, and/or not factual.  
Score 3: The response is somewhat correct, accurate, and/or factual.  

16 
 Score 4: The response is mostly correct, accurate, and factual.  
Score 5: The response is completely correct, accurate, and factual.  
###Feedback:  
The second LLM -evaluator tool is RAGAS  [36] (Retrieval Augmented Generation Assessment) , which is 
a framework evaluation of RAG pipelines  by considering both the retriever and response generation phases . 
RAGAS offers metrics tailored for evaluating each component of a RAG pipeline in isolation.  In our study , 
we used the metrics  listed in Table 2.  
Table 2 – RAGAS metrics  
Type  Metric  Description  
Retriever Metrics  Context Precision  In simple terms how relevant is 
context retrieved to the question 
asked.  
Context Recall  Is the retriever able to retrieve all 
the relevant context pertaining to 
ground truth?  
Context Utilization  Whether all the answer relevant 
items present in the  contexts  are 
ranked higher or not.  
Response Generation Metrics  Answer Relevancy  How relevant is the generated 
answer to the question.  
Faithfulness  Factual consistency of generated 
answers with the given context.  
Comprehensive Metrics  Answer Correctness  Answer correctness 
encompasses two critical 
aspects: semantic similarity 
between the generated answer 
and the ground truth, as well as 
factual similarity.  
Answer Similarity  The semantic resemblance 
between the generated answer 
and the ground truth.  
 
Finally , we evaluated the performance of our RAG system against well-known  medical LLMs (Meditron  
[1], BioMistral  [2] and OpenBioLLM  [3] and the original Llama 3.1  8B) by applying the above evaluation 
framework (except RAGAS which is only applicable to RAG systems) to these LLMs. Importantly, we 
complemented this evaluation with a rigorous human assessment, where three independent physicians, 
leveraging their medical expertise, carefully evaluated the system's responses for accuracy, relevance, and 

17 
 clinical soundness. This human evaluation served as a critical validation step, ensuring that the system's 
outputs meet real -world medical expectations . 
4 RESULTS  
Following the comprehensive evaluation framework outlined in the previous section , we assessed the 
performance of FHIR -RAG -MEDS  against various large medical  language models across four sections of 
the CAREPATH consensus clinical guidelines: dementia, COPD, hypertension, and sarcopenia. The 
evaluation aimed to determine the effectiveness of each model in generating accurate, relevant, and 
contextually appropriate responses to clinical queries. The results were  measured using multiple metrics, 
including BERTScore, ROUGE -L, and METEOR, alongside a tailored set of RAGAS and Prometheus 2 
metrics designed to evaluate the models' performance from a clinical decision -support perspective. This 
section presents the evaluation  result s, highlighting the strengths and weaknesses of each LLM in relation 
to the medical guidelines.  
4.1 Dementia Results Interpretation  
For the dementia  guideline, the proposed FHIR -RAG -MEDS  system consistently outperformed the other 
models in all the text -based evaluation metrics (BERTScore F1, ROUGE -L F1, and METEOR) , as shown 
in Table 3. Considering Prometheus 2, the RAG system outperformed all other models  as well , achieving 
an average score of 4.0. This indicates that it provided responses that were more accurate and relevant 
compared to other LLMs, particularly BioMistral , which had the lowest average score of 3.17.  
Table 3 – Comparison of FHIR -RAG -MEDS with other LLMs for the dementia guideline.  
LLM NAME  PROMETHEUS 2 
AVERAGE SCORE  BERTSCORE F1  ROUGE -L F1  METEOR  
BIOMISTRAL  3.1667  0.4900  0.11198  0.10472  
LLAMA 3.1 8B  3.6667  0.5497  0.12921  0.24090  
MEDITRON  3.6667  0.5592  0.13154  0.25065  
OPENBIOLLM  3.5000  0.5406  0.14398  0.16664  
FHIR -RAG -MEDS  4.0000  0.6372  0.25427  0.37810  
 

18 
 The numbers shown in Table 3 can be interpreted as follows.  
• The BERTScore F1  of 0.6372  for FHIR -RAG -MEDS indicates strong semantic accuracy, 
suggesting that the generated responses closely align with the meanings in the reference answers.  
This score  surpasses all other models. The closest contender is Meditron with 0.5592, followed by  
Llama  3.1 8B  (0.5497) and OpenBioLLM (0.5406) , which  also performed fairly well. BioMistral , 
with a lower score of 0.49, indicates some limitations in semantic matching for this guideline.  
• The ROUGE -L F1  score of 0.2543  for FHIR -RAG -MEDS highlights the model's ability to retain 
key terms from the guidelines, although there may still be room for improvement in capturing more 
of the reference content. FHIR -RAG -MEDS leads again with  more than double the score of some 
competitors.  
• The METEOR  score of 0.3781  demonstrates  good flexibility in language use, indicating that 
FHIR -RAG -MEDS can generate responses that maintain meaning even with variations in wording. 
FHIR -RAG -MEDS’  dominance is evident  here as well , followed by Meditron at 0.251. This implies 
that FHIR -RAG -MEDS and Meditron are better at capturing word -level matches while maintaining 
language flexibility. Llama (0.241) and OpenBioLLM (0.167) also performed adequately, while 
BioMistral fell short with 0.105.  
In the RAGAS metrics  for dementia, FHIR -RAG -MEDS’  performance is notably strong , as shown in Table 
4. These metrics reveal that the FHIR -RAG -MEDS  system excels in answer similarity  and context 
precision  but could improve in faithfulness , as it shows some gaps in factual correctness with a score of 
0.481. However, the context utilization  score of 1.0 highlights its capability to rank relevant information 
effectively.  
Table 4 – The RAGAS metrics of FHIR -RAG -MEDS for dementia , COPD, hypertension, and sarcopenia . 
METRIC  DEMENTIA  COPD  Hypertension  Sarcopenia   
ANSWER CORRECTNESS  0.783  0.908  0.953  0.850   

19 
 ANSWER RELEVANCY  0.790  0.670  0.812  0.840   
ANSWER SIMILARITY  0.944  0.930  0.908  0.929   
CONTEXT PRECISION  1.0 0.994  0.927  0.977   
CONTEXT RECALL  0.811  0.792  0.903  1.0  
CONTEXT UTILIZATION  1.0 1.0 0.949  0.977   
FAITHFULNESS  0.481  0.725  0.599  1.0  
 
4.2 COPD Results Interpretation  
For the COPD  guideline, FHIR -RAG -MEDS  continues to dominate, though the competition is closer in 
certain metrics.  The FHIR -RAG -MEDS  system achieved the highest average score of 4.38 (Prometheus 2) , 
demonstrating  its superior ability to generate relevant and accurate answers compared to the other models.  
Table 5 - Comparison of FHIR -RAG -MEDS with other LLMs for the COPD guideline.  
LLM NAME  AVERAGE SCORE  BERTSCORE F1  ROUGE -L F1  METEOR  
BioMistral  3.2308  0.5329  0.13596  0.18708  
Llama 3.1 8B 4.2308  0.5647  0.15854  0.29252  
Meditron  3.4615  0.5403  0.16486  0.24902  
OpenBioLLM  3.7692  0.5811  0.22319  0.22574  
FHIR -RAG -MEDS  4.3846  0.6325  0.25853  0.36496  
 
Table 5 lists the results of the evaluation for the COPD guideline, which can be interpreted as follows.  
• BERTScore F1 : FHIR -RAG -MEDS  performed the best at 0.6325, followed by OpenBioLLM 
(0.5811), Llama  3.1 8B  (0.5647), and Meditron (0.5403). BioMistral was again lower, though still 
reasonable with a score of 0.533.  
• ROUGE -L F1 : FHIR -RAG -MEDS’ score of 0.259 is impressive , but OpenBioLLM (0.223) and 
Meditron (0.165) closed the gap compared to other guidelines. This suggests that these models were 
also relatively effective in capturing key terms in this domain.  

20 
 • METEOR : FHIR -RAG -MEDS  maintained its lead with 0.365, while Llama 3.1 8B (0.293) and 
OpenBioLLM (0.226) provided fairly strong performances. Meditron (0.249) and BioMistral 
(0.187) again lagged.  
For the RAGAS metrics  in COPD, FHIR -RAG -MEDS  demonstrates  strong consistency , as shown in  Table 
4. The faithfulness  score  of 0.725 is considerably higher than in the dementia guideline, suggesting better 
factual accuracy in the COPD domain. The context recall  is slightly lower (0.792), but the answer 
correctness  and similarity  are strong at 0.908 and 0.930, respectively . This  indicat es that the responses 
generated are both semantically and factually aligned with the ground truth.  
4.3 Hypertension Results Interpretation  
For the hypertension  guideline, FHIR -RAG -MEDS  solidifies its leading position, particularly  in terms of 
semantic accuracy.  The FHIR -RAG -MEDS  system achieved an average score of 4.45  (Prometheus 2) , 
showcasing its effectiveness , as shown in Table 6. 
Table 6 - Comparison of FHIR -RAG -MEDS with other LLMs for the hypertension guideline.  
LLM NAME  AVERAGE SCORE  BERTSCORE F1  ROUGE -L F1  METEOR  
BioMistral  3.37 0.535  0.138  0.210  
Llama 3.1 8B 3.45 0.520  0.120  0.253  
Meditron  3.50 0.507  0.119  0.219  
OpenBioLLM  3.63 0.538  0.159  0.200  
FHIR -RAG -MEDS  4.45 0.650  0.299  0.463  
 
The numbers shown in Table 6 can be interpreted as follows.  
• BERTScore F1 : FHIR -RAG -MEDS  scored 0.6503, demonstrating superior performance 
compared to Bio Mistral (0.5352), OpenBioLLM (0.5375), and Llama 3.1 8B (0.5196). Meditron 
lagged behind at 0.5069.  

21 
 • ROUGE -L F1 : FHIR -RAG -MEDS  scored  0.299, far surpassing the other models . OpenBioLLM 
achiev ed 0.159 , while the rest scored below 0.14. This indicates FHIR -RAG -MEDS’ effectiveness 
in capturing key terms from the hypertension guideline.  
• METEOR : FHIR -RAG -MEDS  once again led with 0.463, highlighting its robustness in word -
level matching . Meanwhile, Llama  3.1 8B  (0.253), BioMistral (0.210), and Meditron (0.219) 
showed moderate performance.  
In the RAGAS metrics  for hypertension, FHIR -RAG -MEDS  demonstrates remarkable accuracy , as shown 
in Table 4. Answer correctness  reaches an impressive 0.953, showcasing that FHIR -RAG -MEDS  performs 
extremely well in generating factually correct responses. Context recall  (0.903) and context utilization  
(0.949) are also high, indicating that the retriever is effectively pulling the most relevant material for this 
guideline. However, faithfulness  at 0.599 reveals that there is room for improvement in factual consistency.  
4.4 Sarcopenia Results Interpretation  
For the sarcopenia  guideline, FHIR -RAG -MEDS  is once again the top -performing model across most 
metrics.  The FHIR -RAG -MEDS  system achieved an average score of 4.36 (Prometheus 2) . 
Table 7 - Comparison of FHIR -RAG -MEDS with other LLMs for the sarcopenia guideline.  
LLM NAME  AVERAGE SCORE  BERTSCORE F1  ROUGE -L F1  METEOR  
BioMistral  3.45 0.569  0.154  0.245  
Llama 3.1 8B 4.45 0.581  0.142  0.323  
Meditron  4.00 0.612  0.184  0.328  
OpenBioLLM  2.82 0.591  0.198  0.278  
FHIR -RAG -MEDS  4.36 0.737  0.465  0.640  
 
Table 7 lists the results of the evaluation for the sarcopenia guideline, which can be interpreted as follows.  
• BERTScore F1 : FHIR -RAG -MEDS’  score of 0.7367 is the highest, followed by Meditron 
(0.6124), Llama  3.1 8B  (0.5805), and BioMistral (0.5687). OpenBioLLM . In contrast to other 
guidelines, OpenBioLLM scored significantly lower at 0.5906.  

22 
 • ROUGE -L F1 : FHIR -RAG -MEDS  dominated with 0.465, nearly three times the score of the 
nearest model, OpenBioLLM (0.198). This suggests that FHIR -RAG -MEDS  is especially effective 
in identifying key terms for sarcopenia, while other models fell short.  
• METEOR : FHIR -RAG -MEDS  performed impressively with 0.640, followed by Meditron (0.328) 
and Llama 3.1 8B (0.323). OpenBioLLM and BioMistral trailed behind, further highlighting  the 
difficulty of this task for non -RAG models.  
The RAGAS metrics  for sarcopenia show that FHIR -RAG -MEDS  performs exceptionally well . FHIR -
RAG -MEDS  achieved  a faithfulness  score of 1.0, indicating that its responses are both factually  accurate  
and semantically aligned with the context for sarcopenia. The answer correctness  and answer relevancy  
are also high, with context recall  reaching 1.0, showing that all relevant material was retrieved effectively.  
Consequently, t his breakdown provide s a guideline -specific analysis , demonstrat ing FHIR -RAG -MEDS ’ 
overall superiority in generating factually accurate and semantically aligned responses across various 
medical guidelines. Each guideline presented  unique challenges, but the FHIR -RAG -MEDS  model proved 
to be the most robust, particularly in the sarcopenia and dementia domains.   
High scores in BERTScore, ROUGE -L, and METEOR indicate strong semantic understanding and 
flexibility in language use —both of which are crucial for clinical decision -making. The RAGAS metrics 
further reinforce the reliability of the FHIR -RAG -MEDS  system, showing  that it effectively utilizes context 
and delivers  factually correct responses . Although there is room  for improvement in certain aspects of 
faithfulness  and context recall , this comprehensive evaluation underscores  FHIR -RAG -MEDS ’ 
capabilities, particu larly in the medical domain, and its potential to support  healthcare professionals in 
clinical decision -making.  
4.5 Human Evaluation of the FHIR -RAG -MEDS System  
In the context of clinical decision support systems, human evaluation remains a critical  component of 
performance validation. While automated metrics such as  BERTScore, ROUGE, and METEOR provide 

23 
 valuable insights into semantic and lexical alignment of generated responses, they often fall short in 
assessing  factual correctness, clinical relevance, and contextual appropriateness —key factors in medical 
decision -making. Incorporating  human evaluators, particularly domain experts like physicians, ensures a 
more comprehensive assessment of system outputs. These evaluations capture nuanced clinical 
considerations, identify potential gaps in reasoning, and validate the alignment of responses with evidence -
based guidelines.  
To evaluate  the accuracy and clinical relevance of the FHIR -RAG -MEDS system, we conducted a human 
evaluation study involving three independent physicians , who are geriatricians and experts in 
multimorbidity in older adults with dementia . These physicians were tasked with reviewing and scoring the 
system's responses to questions derived from four clinical guidelines: dementia, COPD, hypertension, and 
sarcopenia.  Responses were rated on a 5 -point  Likert scale from 1  (The response is completely incorrect, 
inaccurate, and/or not factual ) to 5 ( The response is completely correct, accurate, and factual ). A 
standardized Excel template was developed  to facilitate this process . Each physician independently assessed  
the system’s responses, assigning  numerical scores and providing qualitative comments where applicable. 
These evaluations aimed to measure the factual correctness, clarity, clinical relevance, and adherence to the 
respective clinical  guidelines.  
Across the four guidelines, the average physician scores ranged from 3.67 to 4.45, with the highest scores 
observed for hypertension -related queries and the lowest for dementia -related queries. Specifically:  
• COPD : The average physician score was 4.38 (±0.12), indicating strong agreement regarding  the 
clinical relevance and accuracy of the responses.  
• Dementia : The average score was 3.67 (±0.15), with variability reflecting the complexity of the 
scenarios and occasional gaps in contextual understanding.  
• Hypertension : The system achieved the highest  average score of 4.45 (±0.10), demonstrating both 
high consistency and reliability in this domain.  

24 
 • Sarcopenia : The average score was 4.36 (±0.14), with qualitative feedback highlighting areas for 
improvement , particularly in providing  actionable recommendations.  
Physicians’ qualitative feedback identified  key strengths, such as the system’s ability to generate 
contextually relevant , evidence -based recommendations . However, feedback also revealed  occasional 
shortcomings , notably  in capturing nuanced clinical details. For instance , 12% of responses were marked 
as "lacking specific actionable insights," despite being semantically accurate.  The full list of questions and 
physicians’ responses are provided as supplementary material.  
Statistical analysis of inter -rater reliability using Cohen’s kappa demonstrated  substantial agreement among 
the three physicians ( κ = 0.79), indicating a high degree  of consistency in their evaluations. Additionally, 
the correlation between automated scores and physician scores was strong (Pearson’s r = 0.85, p < 0.01), 
reinforcing the validity of the automated evaluation framework as a reliable complementary tool.  
By combining  human evaluation with automated metrics, this study highlights  the critical role of expert 
oversight in validating AI -driven clinical decision support systems. This integrated approach ensures that 
the recommendations generated by the FHIR -RAG -MEDS system are not only technically robust but also 
clinically meaningful and actionable.  
5 CONCLUSIONS  
Integrating LLMs into  clinical decision support systems  (CDSS)  represents a significant shift in healthcare 
delivery, particularly by leveraging natural language processing to interpret clinical documentation and 
provid e recommendations aligned with current medical research and best practices  [37], [38]. Our approach 
demonstrates that a locally hosted LLM, securely deployed in a private network, can access patient -specific 
data and integrate this information into clinical decision  support . This enables the generation of highly 
personalized treatment plans based on  up-to-date clinical guidelines and tailored to the unique needs of 
individual patients.   

25 
 The LLMs tested —including BioMistral, Llama  3.1 8B , Meditron  3, and OpenBioLLM —showed varying 
degrees of accuracy in aligning with clinical guidelines. Notably, our RAG system , FHIR -RAG -MEDS,  
outperformed traditional LLMs in many scenarios, emphasizing the value of combining retrieval and 
generation techniques for more reliable and contextually accurate medical advice.  The system’s high 
performance across automated and human evaluations validates its ability to generate accurate and 
clinically relevant recommendations.  
The integration of HL7 FHIR with RAG systems holds transformative potential for clinical decision 
support. By leveraging real -time access to patient data and combining it with up -to-date medical 
knowledge, this approach provides healthcare professionals wi th powerful tools to deliver personalized, 
evidence -based care.  Integrating SMART on FHIR  into the FHIR -RAG -MEDS system significantly 
enhances its interoperability, making it a versatile and scalable solution for clinical decision support. This 
implementat ion not only ensures compliance with modern healthcare standards but also positions the 
system as a robust tool capable of addressing diverse clinical scenarios through seamless connectivity with 
FHIR servers.  
To further optimize the performance of LLMs within CDSS, future work could focus on improving 
accuracy, efficiency, and user experience. Accuracy can be enhanced by refining the system's ability to 
interpret complex clinical cases, ensuring that model outp uts are both factually correct and clinically 
relevant. A key future direction is continual learning, whereby the system updates itself with new medical 
research and guidelines in real time, keeping pace with the rapidly evolving healthcare landscape. By 
integrating more robust human -in-the-loop feedback mechanisms, the system can evolve alongside medical 
professionals, fostering a collaborative environment where LLM -driven recommendations can be 
continually refined through  expert feedback.  
Extending the system with reinforcement  learning with human feedback (RLHF )—a technique that 
improve s AI models’ performance  by learning directly from human feedback  during the fine -tuning phase —
could enhance accuracy and reduce bias [3 9]. In this approach , feedback provided by physicians can be 

26 
 used for reward modelling, which is then employed to fine -tune the model. This process could lead to the 
system generating responses that better align  with human values and preferences . With the involvement of 
more physicians, and hence more expert feedback, RLHF could provide significant improvement in the 
system’s performance.  
Finally, as part of our future work, we aim to enhance the explainability of the decision support system by 
providing clear and transparent links to relevant sections of the evidence -based guidelines used in 
generating the recommendations. This focus on explainabil ity will empower healthcare professionals to 
better understand the rationale behind the system’s suggestions, fostering trust and facilitating informed 
decision -making. By integrating explicit references to the specific guidelines or evidence sour ces that 
underpin the personalized advice, the system will not only support clinical accuracy but also ensure 
accountability and reliability in its outputs. This approach aligns with the principles of interpretable AI and 
is expected to improve user confid ence and adoption .  
6 DECLARATIONS  
6.1 Ethics approval and consent to participate  
The CAREPATH study is conducted in accordance with the Declaration of Helsinki and has been approved 
by the Ethics Committee of Albacete (V4.0 protocol code 2020 -31, No -EPA, and date of approval 
25/04/2024) for studies involving humans. Informed consent was obtained fr om the evaluating medical 
doctor  participants , who are also the co -authors of this submission,  involved in the study.  This study did 
not involve real patient data.  
6.2 Consent for publication  
Informed consent was obtained from the evaluating medical doctor participants, who are also the co -authors 
of this submission, involved in the study. This study did not involve real patient data.   
6.3 Availability of data and materials  
All data generated or analyzed  during this study are included in this published article and its supplementary 
information files.  

27 
 6.4 Competing interests  
The authors declare that they have no competing interests.  
6.5 Funding  
This research did not receive any specific grant from funding agencies in the public, commercial, or not -
for-profit sectors.  
6.6 Authors' contributions  
Conceptualization, YK, GBLE, MG, TN, AAS and AD; methodology, YK, GBLE  and MG; software, YK; 
formal analysis, YK; investigation, YK, MG, RAC, CGB and PA; writing —original draft preparation, YK, 
GBLE and MG; writing —review and editing, TN, AAS, RAC, CGB, PA and AD; visualization, YK; 
supervision, AD; project administration, YK and  GBLE. All authors have read and agreed to the published 
version of the manuscript.  
6.7 Acknowledgements  
The authors would like to acknowledge the support of the CAREPATH consortium. The clinical guidelines 
utilized in this work are a result of the CAREPATH Project, funded by the European Union’s Horizon 2020 
research and innovation program under grant agreem ent No 945169.  
7 REFERENCES  
[1] A. H. C. et. al. Zeming Chen, “MEDITRON -70B: Scaling Medical Pretraining for Large 
Language Models,” ArXiv , vol. abs/2311.16079, 2023, Accessed: Oct. 08, 2024. [Online]. 
Available: 10.48550/arXiv.2311.16079  
[2] A. B. E. M. P. -A. G. M. R. R. D. Yanis Labrak, “BioMistral: A Collection of Open -Source 
Pretrained Large Language Models for Medical Domains,” in ACL 2024 - Proceedings of the 62st 
Annual Meeting of the Association for Computational Linguistics , 2024. doi: 
https://doi.org/10.48550/arXiv.2402.10373.  
[3] “Open Source Biomedical Large Language Model,” 2024. Accessed: Oct. 09, 2024. [Online]. 
Available: https://huggingface.co/aaditya/Llama3 -OpenBioLLM -70B 
[4] E. P. A. P. F. P. V. K. N. G. H. K. M. L. W. Y . T. R. S. R. D. K. Patrick Lewis, “Retrieval -
Augmented Generation for Knowledge -Intensive NLP Tasks,” ArXiv , 2021, doi: 
https://doi.org/10.48550/arXiv.2005.11401.  
[5] J. Miao, C. Thongprayoon, S. Suppadungsuk, O. A. Garcia Valencia, and W. Cheungpasitporn, 
“Integrating Retrieval -Augmented Generation with Large Language Models in Nephrology: 

28 
 Advancing Practical Applications,” Medicina (B Aires) , vol. 60, no. 3, p. 445, Mar. 2024, doi: 
10.3390/medicina60030445.  
[6] S. Tian et al. , “Opportunities and challenges for ChatGPT and large language models in 
biomedicine and health,” Brief Bioinform , vol. 25, no. 1, Nov. 2023, doi: 10.1093/bib/bbad493.  
[7] Z. Ji et al. , “Survey of Hallucination in Natural Language Generation,” ACM Comput Surv , vol. 
55, no. 12, pp. 1 –38, Dec. 2023, doi: 10.1145/3571730.  
[8] Y. X. X. G. K. J. J. P. Y. B. Y. D. J. S. M. W. H. W. Yunfan Gao, “Retrieval -Augmented Generation 
for Large Language Models: A Survey,” ArXiv , 2024, doi: 
https://doi.org/10.48550/arXiv.2312.10997.  
[9] H. Z. Q. Y. Z. W. Y. G. F. F. L. Y. W. Z. J. J. B. C. Penghao Zhao, “Retrieval -Augmented 
Generation for AI -Generated Content: A Survey,” ArXiv , 2024, doi: 
https://doi.org/10.48550/arXiv.2402.19473.  
[10] J. Z. Y. Q. Junde Wu, “Medical Graph RAG: Towards Safe Medical Large Language Model via 
Graph Retrieval -Augmented Generation,” ArXiv , 2024, doi: 
https://doi.org/10.48550/arXiv.2408.04187.  
[11] C. Zakka et al. , “Almanac — Retrieval -Augmented Language Models for Clinical Medicine,” 
NEJM AI , vol. 1, no. 2, Jan. 2024, doi: 10.1056/AIoa2300068.  
[12] L. J. K. E. H. R. A. N. L. A. T. H. S. C. R. S. J. Y. M. T. J. C. L. O. D. S. W. T. YuHe Ke, 
“Development and Testing of Retrieval Augmented Generation in Large Language Models -- A 
Case Study Report,” ArXiv , 2024, doi: https://doi.org/10.48550/arXiv.2402.01733.  
[13] Z. Al Nazi and W. Peng, “Large Language Models in Healthcare and Medical Domain: A Review,” 
Informatics , vol. 11, no. 3, p. 57, Aug. 2024, doi: 10.3390/informatics11030057.  
[14] J. Z. Z. G. W. H. L. F. H. Y. L. H. Y. Z. T. L. A. L. H. S. M. Lingyao Li, “A scoping review of 
using Large Language Models (LLMs) to investigate Electronic Health Records (EHRs),” ArXiv , 
2024, doi: https://doi.org/10.48550/arXiv.2405.03066.  
[15] A. R. P. Z. V. R. A. Z. A. F. O. A. Paul Schmiedmayer, “LLM on FHIR -- Demystifying Health 
Records,” ArXiv , 2024, doi: https://doi.org/10.48550/arXiv.2402.01711.  
[16] Y. Li, H. Wang, H. Z. Yerebakan, Y. Shinagawa, and Y. Luo, “FHIR -GPT Enhances Health 
Interoperability with Large Language Models,” NEJM AI , vol. 1, no. 8, Jul. 2024, doi: 
10.1056/AIcs2300301.  
[17] A. Sett, S. Hashemifar, M. Yadav, Y. Pandit, and M. Hejrati, “Speaking the Same Language: 
Leveraging LLMs in Standardizing Clinical Data for AI,” Aug. 2024.  
[18] M. Alkhalaf, P. Yu, M. Yin, and C. Deng, “Applying generative AI with retrieval augmented 
generation to summarize and extract key clinical information from electronic health records,” J 
Biomed Inform , vol. 156, p. 104662, Aug. 2024, doi: 10.1016/j.jbi.2024.104662.  
[19] S. Kresevic, M. Giuffrè, M. Ajcevic, A. Accardo, L. S. Crocè, and D. L. Shung, “Optimization of 
hepatological clinical guidelines interpretation by large language models: a retrieval augmented 
generation -based framework,” NPJ Digit Med , vol. 7, no. 1, p. 102, Apr. 2024, doi: 
10.1038/s41746 -024-01091 -y. 

29 
 [20] O. Unlu et al. , “Retrieval -Augmented Generation –Enabled GPT -4 for Clinical Trial Screening,” 
NEJM AI , vol. 1, no. 7, Jun. 2024, doi: 10.1056/AIoa2400181.  
[21] G. Xiong, Q. Jin, Z. Lu, and A. Zhang, “Benchmarking Retrieval -Augmented Generation for 
Medicine,” Feb. 2024.  
[22] “CAREPATH Project Website.” Accessed: Oct. 09, 2024. [Online]. Available: 
https://www.carepath.care/  
[23] O. Pournik  et al. , “CAREPATH: developing digital integrated care solutions for multimorbid 
patients with dementia,” in Advances in Informatics, Management and Technology in Healthcare , 
IOS Press, 2022, pp. 487 -490. doi: 10.3233/SHTI220771  
[24] T. D. Robbins et al. , “Protocol for Creating a Single, Holistic and Digitally Implementable 
Consensus Clinical Guideline for Multiple Multi -morbid Conditions,” in Proceedings of the 10th 
International Conference on Software Development and Technologies for Enhancing Accessibility 
and Fighting Info -exclusion , New York, NY, USA: ACM, Aug. 2022, pp. 1 –6. doi: 
10.1145/3563137.3563182.  
[25] M. Gencturk et al. , “Transforming evidence -based clinical guidelines into implementable clinical 
decision support services: the CAREPATH study for multimorbidity management,” Front Med 
(Lausanne) , vol. 11, May 2024 . doi: 10.3389/fmed.2024.1386689.  
[26] J. C. Mandel  et al. , “SMART on FHIR: a standards -based, interoperable apps platform for 
electronic health records ,” Journal of the American Medical Informatics Association , vol. 23, 
2016,  pp. 899-908, doi: 10.1093/jamia/ocv189 . 
[27] “LangChain Open Source Library,” 2004. Accessed: Oct. 09, 2024. [Online]. Available: 
https://www.langchain.com  
[28] “Chroma Open Source AI Application Database,” 2024. Accessed: Oct. 09, 2024. [Online]. 
Available: https://www.trychroma.com  
[29] “Llama 3.1 8B LLM,” 2024. Accessed: Oct. 09, 2024. [Online]. Available: 
https://ai.meta.com/blog/meta -llama -3-1/ 
[30] Chin -Yew Lin, “ROUGE: A Package for Automatic Evaluation of summaries,” in In Proceedings 
of the Workshop on Text Summarization Branches Out (WAS 2004) , 2004.  
[31] A. L. Satanjeev Banerjee, “METEOR: An Automatic Metric for MT Evaluation with Improved 
Correlation with Human Judgments,” in Proceedings of the ACL Workshop on Intrinsic and 
Extrinsic Evaluation Measures for Machine Translation and/or Summarization , 2005.  
[32] V. K. F. W. K. Q. W. Y. A. Tianyi Zhang, “BERTScore: Evaluating Text Generation with BERT,” 
in International Conference on Learning Representations , 2020. doi: 
https://doi.org/10.48550/arXiv.1904.09675.  
[33] K. Papineni, S. Roukos, T. Ward, and W. -J. Zhu, “BLEU,” in Proceedings of the 40th Annual 
Meeting on Association for Computational Linguistics   - ACL ’02 , Morristown, NJ, USA: 
Association for Computational Linguistics, 2001, p. 311. doi: 10.3115/1073083.1073135.  
[34] Z. Yan, “Evaluating the Effectiveness of LLM -Evaluators (aka LLM -as-Judge).” Accessed: Oct. 
09, 2024. [Online]. Available: https://eugeneyan.com/writing/llm -evaluators/  

30 
 [35] J. S. S. L. B. Y. L. J. S. S. W. G. N. M. L. K. L. M. S. Seungone Kim, “Prometheus 2: An Open 
Source Language Model Specialized in Evaluating Other Language Models,” ArXiv , 2024, doi: 
https://doi.org/10.48550/arXiv.2405.01535.  
[36] “RAGAS Evaluation Framework,” 2024. Accessed: Oct. 09, 2024. [Online]. Available: 
https://docs.ragas.io/en/stable/  
[37] G. Mahadevaiah, P. RV , I. Bermejo, D. Jaffray, A. Dekker, and L. Wee, “Artificial intelligence‐
based clinical decision support in modern medical physics: Selection, acceptance, commissioning, 
and quality assurance,” Med Phys , vol. 47, no. 5, May 2020, doi: 10.1002/mp.13562.  
[38] G. Golden et al. , “Applying artificial intelligence to clinical decision support in mental health: 
What have we learned?,” Health Policy Technol , vol. 13, no. 2, p. 100844, Jun. 2024, doi: 
10.1016/j.hlpt.2024.100844.  
[39]  R. Kirk et al. , “Understanding the effects of rlhf on llm generalisation and diversity ”. arXiv 
preprint arXiv:2310.06452. 2023 Oct 10.  
  
List of Abbreviations:  
AI: Artificial Intelligence  
BLEU: Bilingual Evaluation Understudy  
CDSS: Clinical Decision Support Systems  
COPD:  Chronic Obstructive Pulmonary Disease  
CRG: Clinical Reference Group  
EC: European Commission  
EHR: Electronic Health Records  
FHIR: Fast Healthcare Interoperability Resources  
HL7: Health Level 7  
LLM: Large Language Model  
MCI: Mild Cognitive Impairment  
MD: Mild Dementia  
METEOR: Metric for Evaluation of Translation with Explicit Ordering  
RAG: Retrieval -Augmented Generation  
RAGAS: Retrieval Augmented Generation Assessment  

31 
 RLHF: Reinforcement Learning with Human Feedback  
ROUGE: Recall -Oriented Understudy for Gisting Evaluation  
SMART: Substitutable Medical Applications and Reusable Technologies   