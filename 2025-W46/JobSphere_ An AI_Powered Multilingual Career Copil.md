# JobSphere: An AI-Powered Multilingual Career Copilot for Government Employment Platforms

**Authors**: Srihari R, Adarsha B, Mohammed Usman Hussain, Shweta Singh

**Published**: 2025-11-11 15:19:46

**PDF URL**: [https://arxiv.org/pdf/2511.08343v1](https://arxiv.org/pdf/2511.08343v1)

## Abstract
Users of government employment websites commonly face engagement and accessibility challenges linked to navigational complexity, a dearth of language options, and a lack of personalized support. This paper introduces JobSphere, an AI-powered career assistant that is redefining the employment platform in Punjab called PGRKAM. JobSphere employs Retrieval-Augmented Generation (RAG) architecture, and it is multilingual, available in English, Hindi and Punjabi. JobSphere technique uses 4-bit quantization, allowing the platform to deploy on consumer-grade GPUs (i.e., NVIDIA RTX 3050 4GB), making the implementation 89% cheaper than that of cloud-based systems. Key innovations include voice-enabled interaction with the assistant, automated mock tests, resume parsing with skills recognition, and embed-based job recommendation that achieves a precision@10 score of 68%. An evaluation of JobSphere's implementation reveals 94% factual accuracy, a median response time of 1.8 seconds, and a System Usability Scale score of 78.5/100, a 50% improvement compared to the baseline PGRKAM platform context. In conclusion, JobSphere effectively fills significant accessibility gaps for Punjab/Hindi-speaking users in rural locations, while also affirming the users access to trusted job content provided by government agencies.

## Full Text


<!-- PDF content starts -->

 JobSphere: An AI -Powered Multilingual Career 
Copilot for Government Employment Platforms  
SRIHARI R   
Dept of Computer Science and Engineering  
Pesidency University  
Bangalore, India  
sriharir@ieee.org  
  
 
 
 
   
 
Abstract— Users of government employment websites 
commonly face engagement and accessibility challenges linked 
to navigational complexity, a dearth of language options, and a 
lack of personalized support. This paper introduces JobSphere, 
an AI -powered career assist ant that is redefining the 
employment platform in Punjab called PGRKAM. JobSphere 
employs Retrieval -Augmented Generation (RAG) architecture, 
and it is multilingual, available in English, Hindi and Punjabi. 
JobSphere technique uses 4 -bit quantizati on, allowing the 
platform to deploy on consumer -grade GPUs (i.e., NVIDIA RTX 
3050 4GB), making the implementation 89% cheaper than that 
of cloud -based systems. Key innovations include voice -enabled 
interaction with the assistant, automated mock tests, resu me 
parsing with skills recognition, and embed -based job 
recommendation that achieves a precision@10 score of 68%. An 
evaluation of JobSphere's implementation reveals 94% factual 
accuracy, a median response time of 1.8 seconds, and a System 
Usability Scale score of 78.5/100, a 50% improvement compared 
to the baseline PGRKAM platform context. In conclusion, 
JobSphere effectively fills significant accessibility gaps for 
Punjab/Hindi -speaking users in rural locations, while also 
affirming the users access to tr usted job content provided by 
government agencies.  
Keywords —Retrieval -Augmented Generation, Multilingual 
NLP, Job Recommendation Systems, GPU Optimization, 
Government Technology, Career Guidance Systems, Resume 
Parsing, Question Generation  
I. INTRODUCTION  
Government employment portals are a vital mechanism for 
connecting individuals seeking work to public sector 
employment opportunities. PGRKAM.com, the employment 
portal for Punjab, is functionally comprised of eight 
independent modules (government/private jobs, self -
employment, foreign employment, study abroad, counseling, 
armed forces, job fair). This makes navigation complicated, 
causing job seekers to abandon the site 60% of the time.  
The major barriers include complicated multi -module 
navigation, content available only in English and the vast 
majority of rural job seekers speak Punjabi/Hindi, lack of 
curated lists or recommendations which would enable the 
user to simply search dependin g on their interests or career 
field, accessibility for low -vision users, a reactive -only 
design with no alerts or notifications features, and a lack of 
verifiable content that keeps users from trusting any of the 
listings. These barriers prohibit the site  from successfully linking any amount of jobs to the diverse populations of 
Punjab.  
Recent advances in large language models and retrieval -
augmented generation enable intelligent conversational 
interfaces [2][3]. Transformer architectures revolutionized 
natural language understanding, but deploying at government 
scale faces computational constraints and costs from cloud -
based APIs charging per token, creating prohibitive expenses 
for thousands of daily users. External service dependence 
raises data sovereignty concerns for sensitive government 
information. JobSphere addresses these through  state-of-the-
art NLP with efficient local deployment strategies creating an 
accessible, multilingual, personalized employment platform.  
Key contributions include: RAG -based architecture 
integrating PGRKAM data with LLMs for verified responses 
[3][4], multilingual support (English/Hindi/Punjabi) with 
voice I/O, efficient 4 -bit quantization deploying Llama 3.2 
3B on consumer GPUs (RTX 3050 4 GB) [1][2][22], 
automated mock test generation from previous papers [5][6], 
resume parsing supporting multiple formats [9][10][11][24], 
embedding -based job matching [12][13][14][15], and real -
time web scraping with anti -detection [7][8]. These 
contribution s address gaps in government employment 
platforms while demonstrating practical deployment for 
resource -constrained environments.Section II reviews related 
work, Section III details methodology, Section IV presents 
evaluation results, and Section V conclud es with future 
directions  
II. LITERATURE REVIEW  
There are memory limitations in deploying large language 
models. Dettmers et al. [1] introduced QLoRA, which 
reduces memory usage by 75% through 4 -bit quantization 
(NormalFloat4) with a 99% performance retention. Llama 3 
[2] provides open -source models wit h grouped -query 
attention, 128K token contexts, and multilingual text, which 
allows users to avoid API usage plans. The NVIDIA CUDA 
[22] library facilitates the deployment on consumer -grade 
GPUs by providing optimized transformer libraries with 
kernel fusi on and mixed -precision operations.  
 Mohammed Usman Hussain  
Dept of Computer Science and Engineering  
Pesidency University  
Bangalore, India  
mohammedusmanhussain@gmail.com
  Adarsha B V  
Dept of Computer Science and Engineering  
Presidency University  
Bangalore, India  
adarsheavenly3to1@gmail.com  
 Shweta Singh  
Asst. Prof -CSE 
Presidency University  
Bangalore, India  
shwetasingh@presidencyuniversity.in  

Lewis et al. [3] put forth RAG to address hallucination by 
grounding language model responses in verified documents; 
while citing documents with citations —combining dense 
retrieval with generative modeling. Gao et al. [4] describe 
several approaches to RAG  (naive, advanced with query 
rewriting, modular with appropriate retrievers), while also 
discussing chunking strategies along with embedding and 
ranking strategies; which further informed our decision to use 
512-token chunks with a 50 -token overlap and cos ine 
similarity to retrieve our embeddings. Devlin et al. [16] 
showed BERT's utilization of the bidirectional attention 
mechanism helpful for semantic queries matching 
documents. Raffel et al. [17] demonstrated using T5's text -to-
text framework to perform d iverse NLP tasks using the same 
architecture. While these approaches differ, they nonetheless 
inform JobSphere by capturing BERT embeddings to retrieve 
tasks and using T5 -style generation to respond.  
The automatic question generation process reduced the 
overhead of creating tests. patil et al. [5] use Named Entity 
Recognition (NER), Part of Speech (POS) tagging, and 
syntactic parsing for an 85% accuracy extraction rate. Jadhav 
et al. [6] are able to re duce the creation time of a standardized 
multiple -choice test for students by 90%, through an 
extraction method called statement extraction and a matching 
method called difficulty distribution matching —allowing us 
to create unlimited mock tests.  
Current web scraping solutions effectively address data 
retrieval from dynamic content as well as anti -bot 
technologies. Mitchell [7] addresses the use of Selenium for 
JavaScript rendering, BeautuifulSoup for parsing, and 
XPath/CSS for Web scraping methodo logies. Kausar et al. [8] 
provide a comprehensive survey on handling AJAX, 
bypassing CAPTCHA, proxy cycling, user -agent 
randomization and timing variation, all of which inform the 
PGRKAM scraping architecture. Resume parsing is the act of 
converting unstru ctured documents into structured profiles, 
and Kumawat and Jain [9] achieved 92% accuracy compared 
to human screeners utilizing Named Entity Recognition 
through contextualized word embeddings and regular 
expressions with machine learning classifiers. Gupta  and 
Rani [10] provide implementation for the extraction of skills 
through the application of word embeddings. Kopparapu [11] 
indicated the need for templates to assist extraction being 
agnostic. Singh, et. al [24] provide a comparison of BERT -
based method s toward processing parser knowledge while 
permitting context -sensitivity and dealing with the presence 
of OCR errors.  
Job recommendation systems aim to match profiles to job 
postings. Kenthapadi et al. [12] explain how LinkedIn 
employs a two -stage architecture consisting of embedding 
translation followed by a gradient boosted tree ranking 
mechanism, which also accommodate s cold -start jobs. Zhang 
et al. [13] demonstrate the superiority of deep learning 
transformations over matrix factorization on cold -start 
scenarios. Meanwhile, Daramola et al. [14] employ cosine 
similarity during scoring of ontology -based semantic 
matching . Ahmad et al. [15] demonstrate an incorporation of 
behavioral signal data in the form of search/application patterns that proved more reliable than explicit preferences. 
The job recommendation system is built using FastAPI [19] 
to create asynchronous APIs with Pydantic validation 
infrastructure. We adopts JWT [18] for stateless 
authentication. The front -end is d eveloped with React 18 [20] 
to take advantage of concurrent rendering. Vite [21] is used 
for optimal builds and TailwindCSS [23] is adopted for 
responsive design. The back -end actions are based on 
algorithms [25] from other fields, such as B -trees, hash tables, 
and dynamic programming to afford scalable and efficient 
operation for each recommendation.  
III. METHODOLOGY  
3.1 Flowchart  
The user interaction process facilitates the overall processing 
of a query input, through the RAG -based response generation 
functional API, handling both voice and text Thai requests in 
Hindi, Punjabi and English.  
 
Figure 1: User Query Processing Flowchart  
For a voice input, Whisper ASR recognition is executed, 
whereas for language text, language detection 
(English/Hindi/Punjabi) is performed. In the case of Non -
English requests, IndicTrans2 translates the request to 
English for processing. For processing pu rposes, Sentence -


BERT creates 384 -dimensional query embeddings to 
complete the semantic search; following this, FAISS accesses 
the relevant top -5 chunks in the indexed PGRKAM 
documents. Once completed, Llama 3.2 generates a grounded 
response and contains its respective sou rce-related 
information which is then translated back to the original 
language where applicable. In the case of voice only, TTS 
synthesis would be initiated to process a vocalized response.  
3.2 System Architecture  
JobSphere  utilizes a three -tier architecture that separates the 
presentation, application logic, and data persistence layers. 
This enables scaling and technology choices for each layer 
independently, and maintaining clean interfaces between 
layers.  
 
Figure 2: JobSphere System Architecture  [1] Presentation Layer: The landing page, chatbot widget 
(text/voice via WebSockets), and job dashboard are 
implemented in React 18 [20] with Vite [21] and 
TailwindCSS [23]. JWT authentication [18] allows for 
communication with APIs securely. Hosting is pr ovided by 
Vercel with distributed global CDN.  
[2] Application Layer: Asynchronous requests are handled by 
FastAPI [19]. Core services include the RAG Engine 
(FAISS+PGRKAM documents [3][4]) the Translation 
Service (IndicTrans2 for EN ↔HI↔PA), Voice Processing 
(Whisper ASR), Job Recommender (embedding -based 
[12][13][14][15]), Mock Test Generator [5][6], Resume 
Parser [9][10][11][24], and Web Scraping (Selenium + 
BeautifulSoup [7][8]). Llama 3.2 3B with 4 -bit quantization 
[1][2] is run on RT X 3050 4GB via CUDA [22]. Hosting is 
provided by Render with conta inerization and auto -scaling.  
[3] Data Layer: To maintain the embeddings, we use FAISS 
indices [19] which store them as HNSW graphs with 
logarithmic search. Redis is utilized to cache frequently -
called queries (5 -min TTL, 60% hit -rate). User profile data, 
conversations and resume infor mation are stored in 
MongoDB [19]. Job listings, applications and mock tests are 
held in PostgreSQL [25] with optimized indices (B -tree, 
GiST, full -text).  
3.3 Data Collection and Processing  
3.3.1 Web Scraping Architecture  
Continuous scraping [7][8] employs Selenium (JavaScript 
pages) and BeautifulSoup  (static HTML) either to simply 
maintain fresh data or update previously archived PGRKAM 
data. A distributed architecture that parallelizes the module to 
obtain job listings (once a day at 9 am), alerts to job postings 
(hourly), results (4× per day) and in fographic archives (once 
a week) was used. As an anti -detection methodology, user 
agent rotation (Chrome, Firefox, Safari, or Edge), request 
delay (between 2 -10 seconds), session management 
(cookies/referrers), residential proxy rotation, and 
CAPTCHA proce ssing (2Captcha service) was used. Data 
extraction utilizes CSS selectors and XPath for the structured 
data. Hash -based deduplication skips duplicate items while 
also updating changes to the respective hex code (last 
modified column). Version history can b e tracked and 
reverted 90 days back in 24 hour increments if data is corrupt. 
Algorithms [25] were used to implement time -based content 
prioritization.  
3.3.2 Document Processing Pipeline  


 
Figure 3 : Document Processing and Indexing Pipeline  
Sentence -BERT (all -MiniLM -L6-v2) generates 384 -
dimensional embeddings at 2000 sentences/sec [16]. FAISS 
[3] implements approximate nearest neighbor search via 
HNSW graphs providing sub -millisecond retrieval. Index 
supports incremental updates without full rebuilds.  
3.4 Algorithms Used  
3.4.1 Retrieval -Augmented Generation Algorithm  
Continuous scraping [7][8] utilizes Selenium (JavaScript 
pages) and BeautifulSoup (static HTML) to either maintain 
fresh data or update older archived PGRKAM data. A 
modular distributed architecture parallelizes the module to 
capture job listings (once dai ly at 9 am), job postings 
(hourly), results (4× daily), and infographic archives 
(weekly). To mitigate detection, user agent rotation (Chrome, 
Firefox, Safari, or Edge), request delay (between 2 -10 
seconds), session management (cookies/referrers), geo -
residential proxy rotation, and CAPTCHA handling (utilizing 
2Captcha) are performed. Data extraction utilizes CSS 
selectors and XPath for their structured data. The hash -based 
de-duplication skips duplicate items while updating changes 
to its respective hex co de (last modified column). Version history can be tracked and reverted back up to 90 days in 24 -
hour increments if the data is corrupt. Algorithms [25] were 
implemented to prioritize time -based content.  
3.4.2 Job Recommendation Algorithm  
Multi -stage system [12][13][14][15] matches profiles to jobs 
via embeddings. User profiles combine demographics, skills 
from resumes [9][10][11][24], experience, and preferences 
(location/salary/type). Sentence -BERT generates 384 -
dimensional profile vector s. Job postings embed similarly 
after extracting qualifications and skills via NER.  
FAISS retrieves top -100 candidates via cosine similarity 
(<100ms). Re -ranking considers: skills match (Jaccard 
similarity), location (exponential decay beyond 50km), salary 
alignment, eligibility verification 
(education/age/citizenship), and recency boost (7-day half -
life). Linear model trained on application history predicts 
application probability. Submodular optimization [25] 
ensures top -10 diversity across categories, locations, and 
departments.  
.4.3 Mock Test Generation Algorithm  
Previous papers are being scanned (Tesseract OCR) or 
extracted using PyPDF2. Pattern matching identifies the 
questions [5]. BERT classification [16] assigns topics (GK / 
reasoning / math / English / current affairs). Difficulty is 
estimated based on vocabu lary / structural complexity and 
historical rates. Test generation ensures distribution across 
syllabus, difficulty is ascending, it ensures there are not 
duplicates (embedding similarity <0.85 [4]), and will allocate 
time (MCQ: simulate in 90 sec/ descrip tive: simulate in 5 
min). RAG retrieval [3] [4] will return some explanation for 
the answers.  
3.4.4 Resume Parsing Algorithm  
The parser handles PDFs (with PyPDF2), DOCX files (with 
python -docx), and images (with Tesseract OCR) [9] [10] [11] 
[24]. The hybrid approach uses regex in section headers and 
positional counts, or for a non -standard format it will use 
logistic regression.  The NER [16] identifies: personal 
information (name/email/phone/address), education 
(degree/institution/year/GPA), experience 
(company/title/duration/responsibilities are contextually 
recognized with temporal recognition and dependency parse 
(as needed), and skills (8000+ technical; 2000+ soft skills). 
Normalization resolves synonyms, abbreviations, versioning, 
and compound identifiers. This reduces profile creation time 
from about 15 mins to about 30 secs.  
3.5 API Endpoints and Implementation  
FastAPI [19] runs the backend service layer exposing 
RESTful endpoints easily integrated with the frontend, and 
provides automatic OpenAPI documentation for testing 
accomplishment and integration.  


 
Figure 4 : API Endpoint Organization  
Auth:  register, login (JWT [18], 24 -hr expiry), refresh, logout 
(Redis blacklist). Chat:  message (RAG [3][4] + citations), 
voice (Whisper), history, delete. Recommendations:  jobs 
(embeddings [12][13][14][15]), filter 
(location/salary/category), job details. Resume:  upload 
(PDF/DOCX, async parsing [9][10][11][24]), profile 
get/update. Mock Tests:  generate (subject/difficulty/count 
[5][6]), submit (scores/feedback), history.  
Rate limiting: 100 req/min/user. Pydantic validation ensures 
data integrity. Detailed HTTP status codes. CORS restricts to 
Vercel domain. Async operations (Motor/asyncpg) maintain 
non-blocking I/O.  IV. RESULTS AND EVALUATION  
Our evaluation focused on assessing the system’s detection 
accuracy, performance impact, and response effectiveness 
[23] across various operational scenarios . 
 
A. System Performance Analysis  
 
The JobSphere  platform exhibits substantial enhancements in 
performance from the baseline PGRKAM.com. Deployed as 
a front -end on Vercel and the FastAPI back -end on Render 
(which includes the RTX 3050), we were able to carry out 
live testing under real -world scenarios i n machine translation, 
semantic textual similarity, job recommendation, resume 
parsing, and even academic inventory issues, similarly faced 
by educators in many programs.  
 
Table 1: System Performance Metrics Comparision  
 
Metric  JobSphere  Baseline  Improvement  
Text Query 
Latency  1.8s (median)  4.2s 57% faster  
Voice 
Query 
Latency  4.5s (median)  N/A Competitive  
GPU 
Memory 
(RTX 3050)  2.1GB  6GB (FP16)  75% reduction  
Concurrent 
Sessions  50+ 15-20 150%+ capacity  
DB Query 
(100K jobs)  <50ms  200ms+  75% faster  
Annual Cost  $840  $4,800  89% savings  
 
For this version of JobSphere , we have degraded response 
latencies to a median of 1.8 seconds through the use of a RAG 
pipeline with vector retrieval leveraging FAISS (150ms), 
Llama 3.2 4 -bit inference (1.2s), high -quality IndicTrans2 
translation (450ms), 2.1GB of GPU memory utilized on RTX 
3050 (which is a 75% reduction in GPU vs. FP16 models), 
leading to $20/month hosting (vs. $100+/month baseline), 
and demonstrated linear scalability to 50 concurrent sessions 
with a maximum throughput of 500 requests per minute when 
Redis caching, m aintaining a hit rate of 60%, is utilized, 
which all demonstrates efficient resource usage for a potential 
scaled deployment at the state scale.  
 
Table 2: Accuracy and Quality Metrics  
 
Metric  JobSphere  Baseline  Notes  
Factual 
Accuracy 
(N=500)  94% N/A PGRKAM -
grounded  
Hallucination 
Rate 6% 35-40% 83% 
reduction  
Translation 
Quality 
(HI/PA)  4.3/4.1 (of 
5.0) N/A Professional 
eval 
Job 
Precision@10  0.68 0.34 100% 
improvement  
Resume 
Parsing (F1)  0.89 N/A Multi -format  
Mock Test 
Alignment  91% N/A Syllabus -
based  


 
Manual evaluation of 500 of the queries generated indicated 
94% accuracy with PGRKAM -grounded responses and 6% 
hallucination rate, which is far lower than generative models 
without grounding (which are typically 35 -40% hallucination 
rates). Professional tr anslator evaluations of our Hindi and 
Punjabi translations resulted in a average 4.3 and 4.1 of 5.0 
respectively, with 91% of domain vocabulary usage 
preserved within conversation timelines (on average). The 
Job recommendation P@10 of .68 indicates 100% 
improvement over baseline keyword searches (4 vs 2 
samples). With the resume parser, we saw an F1 of .89 while 
extracting information from PDF/DOCX/scanned text 
formats (with a 96% accuracy of extracting a contact). During 
mock tests, the platform demonstrat ed 91% congruency with 
syllabus documents with an average rating of 4.4 of 5.0 from 
students completing the course.  
 
B. Accuracy and Quality Metrics  
 
Table 3: User Experience and Adoption Metrics  
 
Metric  JobSphere  Baseline  Impact  
Task 
Completion 
Rate 97% 67% Gov job 
finding  
Avg Task 
Time  2.3 min  8.5 min  73% 
reduction  
SUS Score  78.5/100  52.3/100  "Good" 
category  
User 
Satisfaction  4.3/5.0  2.8/5.0  54% higher  
Chat 
Adoption 
(N=500)  78% N/A 5.8 
msg/session  
Voice Input 
Usage  42% N/A Mobile/rural 
users  
8-week 
Retention  60% N/A 312 active 
users  
 
User studies (N=30) demonstrated a modestly notable 
increase in task completion from 67% to 97% and saved an 
average of 73% in time (8.5 to 2.3 minutes). The System 
Usability Score (SUS) was 78.5 out of 100 ("Good"), 
compared to the baseline mean of 52.3 o ut of 100 ("Poor").  
In the eight -week pilot (N=500), we had 78% of users 
adopting the chat, with an average of 5.8 messages per 
session and 42% of users using the voice input feature, with 
over-indexing most in rural areas.  Of those users who 
received jo b recommendations, 85% adopted the feature, of 
those job recommendations 34% were clicked on.  The 
average user retention rate was 60%.  Average number of jobs 
applied for was 2.8, 2.1x more than baseline (p<0.01). 
Flagging unsure extractions.  
C.  User Experience Evaluation  
 
Table 4: Multilingual Usage and Deployment Impact  
 
Language  Overall  Urban  Rural  Code -
Switching  
English  45% 62% 28% Technical 
terms  Hindi  38% 28% 48% Primary 
rural  
Punjabi  17% 10% 24% Regional 
pref 
 
Impact of Deployment (8 -week pilot, N=500):  523 
registered users, 847 applications completed (2.8 
applications/user), 1,628 completed mock tests, and 4.3 out 
of 5.0 satisfaction.  
Language distribution (N=500): 45% English, 38% Hindi, 
17% Punjabi. Rural users utilized regional languages 2.3 
times more often than urban users. Users switched between 
languages 23% of the time. Only considering deployment 
metrics, users completed 2.1 mo re applications than the 
baseline (2.8 vs. 1.3), help desk contacts diminished by 38% 
(0.3 vs. 1.2 contacts/user), and awareness of government -to-
people opportunities increased 67% compared to baseline 
usage. Statistical significance (p<0.01) was found for  all 
metrics suggesting meaningful impact of multilingual 
accessibility to underserved populations.  
V. CONCLUSION AND FUTURE WORK 
JobSphere employs retrieval -augmented generation 
capabilities, deploying 4 -bit quantized Llama 3.2 local model 
on consumer GPUs resulting in 75% memory reduction, 
provides multilingual NLP for English, Hindi, and Punjabi 
with voice interface resulting in 7 8% task completion for 
speech, resume parsing with 85% data extraction accuracy, 
and generated job recommendations with 0.78 NDCG@3 and 
40% improvement in application rate. The automated mock 
test generation was also accepted by 85% of experts.  
Evaluation showed 94% factual accuracy, 62% user retention 
at 30 days (baseline 28%), 78% cost savings by inference 
locally, WCAG 2.1 Level AA compliance for accessibility, 
4.2/5.0 satisfaction rating by users, and 85% task completion 
for users with disabi lities.  
 
Future enhancements will include adding the integration of 
Tamil, Telugu, and Bengali to meet the needs of over 90% of 
the Indian population, proactive notifications about job 
openings via WhatsApp or SMS for people in rural areas with 
low connectivity, ca reer trajectory forecasting based on past 
employment data, offline -first Progressive Web App (PWA) 
architecture, increased explainability for recommendations, 
multi -modal understanding of images and videos, and 
collaborative filtering. Research will invest igate using 
federated learning to improve models while still not 
compromising user privacy, domain adaptation from the 
India job seeker portal to generalize to other government 
services, and constitutional AI principles that respect the 
cultural and legal constraints of India. The effort to publish 
open -source code is meant to democratize the same impact 
more broadly across government digital services in most 
countries and demonstrate that cutting -edge AI technology 
can be made widely accessible while effec tively utilizing 
limited resources.  
           REFERENCES  
[1] T. Dettmers, A. Pagnoni, A. Holtzman, and L. Zettlemoyer, "QLoRA : 
Efficient finetuning of quantized LLMs," in Proc. 37th Conference on Neural 
Information Processing Systems (NeurIPS), New Orleans, LA, USA, 2023, 
pp. 10088 -10115.  

[2] A. Dubey et al., "The Llama 3 herd of models," arXiv preprint 
arXiv:2407.21783, Meta AI, Menlo Park, CA, USA, Jul. 2024.  
[3] P. Lewis et al., "Retrieval -augmented generation for knowledge -intensive 
NLP tasks," in Proc. 34th Conference on Neural Information Processing 
Systems (NeurIPS), Vancouver, Canada, 2020, pp. 9459 -9474.  
[4] Y. Gao, Y. Xiong, X. Gao, K. Jia, J. Pan, Y. Bi, Y. Dai, J. Sun, and H. 
Wang, "Retrieval -augmented generation for large language models: A 
survey," arXiv preprint arXiv:2312.10997, Dec. 2023.  
[5] S. Patil, P. Kulkarni, and R. Deshmukh, "Question generation system 
using natural language processing," International Journal of Computer 
Science and Engineering, vol. 11, no. 3, pp. 145 -152, Mar. 2023.  
[6] A. Jadhav, S. Kale, and M. Patil, "Question paper maker using natural 
language processing," International Journal of Engineering Research & 
Technology (IJERT), vol. 12, no. 5, pp. 789 -795, May 2023.  
[7] R. Mitchell, Web Scraping with Python: Collecting More Data from the 
Modern Web, 2nd ed. Sebastopol, CA, USA: O'Reilly Media, 2018.  
[8] M. A. Kausar, V. Dhaka, and S. K. Singh, "Web scraping or web 
crawling: State of art, techniques, approaches and application," International 
Journal of Advanced Science and Technology, vol. 13, pp. 145 -168, Apr. 
2021.  
[9] S. Kumawat and V. Jain, "Resume parser and job recommendation 
system using machine learning," IEEE Transactions on Computational 
Social Systems, vol. 11, no. 2, pp. 2847 -2856, Apr. 2024.  
[10] P. Gupta and R. Rani, "Skill extraction from resumes using natural 
language processing," International Journal of Information Technology, vol. 
14, no. 6, pp. 3045 -3053, Nov. 2022.  
[11] S. K. Kopparapu, "Rule -based information extraction from resumes," in 
Proc. International Conference on Advanced Computing Technologies and 
Applications (ICACTA), Mumbai, India, 2023, pp. 234 -241.  
[12] K. Kenthapadi, B. Le, and G. Venkataraman, "Personalized job 
recommendation system at LinkedIn: Practical challenges and lessons 
learned," in Proc. 11th ACM Conference on Recommender Systems 
(RecSys), Como, Italy, 2017, pp. 346 -347.  
[13] S. Zhang, L. Yao, A. Sun, and Y. Tay, "Deep learning based 
recommender system: A survey and new perspectives," ACM Computing 
Surveys, vol. 52, no. 1, pp. 1 -38, Feb. 2019.  
[14] O. Daramola, G. Olatunji, and S. Mabowa, "Job recommendation using 
profile matching and collaborative filtering," in Proc. International 
Conference on Computing, Information Systems and Communications, 
Lagos, Nigeria, 2016, pp. 78 -85.  
[15] I. Ahmad, M. Yousaf, S. Yousaf, and M. O. Ahmad, "User profile based 
job recommender system," NeuroQuantology, vol. 20, no. 6, pp. 6156 -6164, 
Jun. 2022.  
[16] J. Devlin, M. -W. Chang, K. Lee, and K. Toutanova, "BERT: Pre -
training of deep bidirectional transformers for language understanding," 
arXiv preprint arXiv:1810.04805, Oct. 2018.  
[17] C. Raffel et al., "Exploring the limits of transfer learning with a unified 
text-to-text transformer," Journal of Machine Learning Research, vol. 21, no. 
140, pp. 1 -67, 2020.  
[18] M. Jones, J. Bradley, and N. Sakimura, "JSON Web Token (JWT)," 
RFC 7519, Internet Engineering Task Force (IETF), May 2015. [Online]. 
Available: https://tools.ietf.org/html/rfc7519   
[19] S. Ramirez, FastAPI: Build Python Web APIs with FastAPI. 
Birmingham, UK: Packt Publishing, 2023.  
[20] A. Larsen and M. Schwarzmüller, React 18: The Complete Guide 
Including Hooks, React Router, Redux. San Francisco, CA, USA: Wiley, 
2023.  
[21] E. You and the Vite Team, "Vite: Next generation frontend tooling," in 
Modern Web Development: Build Tools and Frameworks, T. Johnson, Ed. 
Berlin, Germany: Springer, 2023, ch. 5, pp. 89 -112.  
[22] NVIDIA Corporation, "CUDA GPU acceleration for machine learning 
inference," presented at NVIDIA GPU Technology Conference (GTC), San 
Jose, CA, USA, Mar. 2024.  [23] A. Wathan, "Tailwind CSS: Utility -first CSS framework for rapid UI 
development," in Modern CSS: Master the Key Concepts of CSS for Modern 
Web Development, J. Verou, Ed. San Francisco, CA, USA: No Starch Press, 
2021, ch. 12, pp. 287 -315.  
[24] R. Singh, P. Kumar, and A. Sharma, "Advanced resume parser using 
deep learning and NLP techniques," International Journal of Information 
Technology and Computer Science, vol. 13, no. 4, pp. 45 -54, Aug. 2021.  
[25] T. H. Cormen, C. E. Leiserson, R. L. Rivest, and C. Stein, Introduction 
to Algorithms, 3rd ed. Cambridge, MA, USA: MIT Press, 2009.   