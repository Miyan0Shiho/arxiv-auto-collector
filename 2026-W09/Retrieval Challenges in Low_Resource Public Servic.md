# Retrieval Challenges in Low-Resource Public Service Information: A Case Study on Food Pantry Access

**Authors**: Touseef Hasan, Laila Cure, Souvika Sarkar

**Published**: 2026-02-25 05:48:15

**PDF URL**: [https://arxiv.org/pdf/2602.21598v1](https://arxiv.org/pdf/2602.21598v1)

## Abstract
Public service information systems are often fragmented, inconsistently formatted, and outdated. These characteristics create low-resource retrieval environments that hinder timely access to critical services. We investigate retrieval challenges in such settings through the domain of food pantry access, a socially urgent problem given persistent food insecurity. We develop an AI-powered conversational retrieval system that scrapes and indexes publicly available pantry data and employs a Retrieval-Augmented Generation (RAG) pipeline to support natural language queries via a web interface. We conduct a pilot evaluation study using community-sourced queries to examine system behavior in realistic scenarios. Our analysis reveals key limitations in retrieval robustness, handling underspecified queries, and grounding over inconsistent knowledge bases. This ongoing work exposes fundamental IR challenges in low-resource environments and motivates future research on robust conversational retrieval to improve access to critical public resources.

## Full Text


<!-- PDF content starts -->

Retrieval Challenges in Low-Resource Public Service Information:
A Case Study on Food Pantry Access
Touseef Hasan
Wichita State University
Wichita, KS, USALaila Cure
Wichita State University
Wichita, KS, USASouvika Sarkar
Wichita State University
Wichita, KS, USA
Abstract
Public service information systems are often fragmented, inconsis-
tently formatted, and outdated. These characteristics create low-
resource retrieval environments that hinder timely access to crit-
ical services. We investigate retrieval challenges in such settings
through the domain of food pantry access, a socially urgent problem
given persistent food insecurity. We develop an AI-powered conver-
sational retrieval system that scrapes and indexes publicly available
pantry data and employs a Retrieval-Augmented Generation (RAG)
pipeline to support natural language queries via a web interface. We
conduct a pilot evaluation study using community-sourced queries
to examine system behavior in realistic scenarios. Our analysis
reveals key limitations in retrieval robustness, handling underspeci-
fied queries, and grounding over inconsistent knowledge bases. This
ongoing work exposes fundamental IR challenges in low-resource
environments and motivates future research on robust conversa-
tional retrieval to improve access to critical public resources.
Keywords
Low-resource environments, Information retrieval, Large language
models, Food access, Retrieval-augmented generation
ACM Reference Format:
Touseef Hasan, Laila Cure, and Souvika Sarkar. 2026. Retrieval Challenges
in Low-Resource Public Service Information: A Case Study on Food Pantry
Access. In.ACM, New York, NY, USA, 3 pages. https://doi.org/10.1145/
nnnnnnn.nnnnnnn
1 Introduction and Motivation
Access to essential public resources increasingly depends on digital
information systems. However, these systems are often ineffective
because information remains scattered, poorly formatted, and out-
dated [ 6]. Such characteristics create low-resource environments,
where data coverage is incomplete, metadata is noisy or weakly
structured, and reliable indexing or benchmarking infrastructure is
limited. These constraints are especially problematic in domains
where timely access to information directly affects well-being. To
study information retrieval (IR) challenges in such environments,
Permission to make digital or hard copies of all or part of this work for personal or
classroom use is granted without fee provided that copies are not made or distributed
for profit or commercial advantage and that copies bear this notice and the full citation
on the first page. Copyrights for components of this work owned by others than the
author(s) must be honored. Abstracting with credit is permitted. To copy otherwise, or
republish, to post on servers or to redistribute to lists, requires prior specific permission
and/or a fee. Request permissions from permissions@acm.org.
Conference’17, Washington, DC, USA
©2026 Copyright held by the owner/author(s). Publication rights licensed to ACM.
ACM ISBN 978-x-xxxx-xxxx-x/YYYY/MM
https://doi.org/10.1145/nnnnnnn.nnnnnnnwe focus on food pantry access—a critical point of interaction be-
tween emergency support infrastructure and food-insecure individ-
uals. We conduct a case study in the state of Kansas (U.S.), where 1 in
8 residents—including 1 in 5 children—experience food insecurity1.
In this context, food pantries serve as essential emergency support
infrastructure. Yet the online directories listing pantry information
are often static, inconsistently structured, and infrequently updated,
limiting timely access to available food assistance.
In such constrained settings, efficiently utilizing existing resources
and delivering accurate information is crucial. We frame this as a
low-resource IR challenge at the intersection of social need and
fragmented public data ecosystems. Rather than manually browsing
static lists, users should be able to issue natural language queries
and receive relevant, timely recommendations. To explore this, we
develop an AI-poweredPantry Assistantthat helps users locate food
pantries in Kansas through conversational interactions. The system
employs a Retrieval-Augmented Generation (RAG) architecture
within a web interface to integrate pantry data with Large Language
Model (LLM)-based responses. By shifting from static directories
to conversational retrieval, this work examines how LLM-based
IR systems can improve access to critical public resources in low-
resource environments. Beyond food security, our approach can
be generalized to other public service domains as well. Our case
study highlights broader retrieval challenges in fragmented public
service domains and motivates future research on more robust,
constraint-aware conversational IR systems.
2 Description of the Proposed Solution
The Pantry Assistant is built as a modular pipeline combining data
preparation, semantic retrieval, and response generation (Figure 1):
Data Preparation.We identified Kansas Food Source2(an online
directory of over 800 food pantries in Kansas) as the comprehensive
source of pantry information. Data from this source was scraped
(as of January 31, 2026) and normalized into a structured format.
In particular, we parsed the web page of each pantry from the
website to extract fields such as name, address, city, county, contact
information, hours of operation, eligibility requirements, and other
notes, storing this metadata in a standardized JSON schema.
Semantic Retrieval.We created vector representations of the
pantry records to enable efficient semantic retrieval via RAG. We
used a sentence-transformer model (all-MiniLM-L6-v23) to encode
metadata of each pantry into an embedding, and indexed these
vectors using FAISS [ 4] for fast similarity search. When a user
asks a question, the chatbot identifies key constraints (e.g., loca-
tion, hours, ID requirements, etc.) and searches the vector index
1https://hungerfreekansas.org/
2https://kansasfoodsource.org/
3https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2arXiv:2602.21598v1  [cs.IR]  25 Feb 2026

Conference’17, July 2017, Washington, DC, USA Hasan et al.
Table 1: Queries categorized by user requirements, along with a systematic breakdown of LLM performance by query type.
An example is provided for each type of query. Accuracy per query type has been calculated for the Llama and GPT models
respectively. Notes indicate the most frequent cause of the erroneous responses for each type of user query.
Type of User Query Example # Queries Accuracy Failure Notes
Llama GPT
Only location Where can I find food pantries in Sedgwick County? 20 0.60 0.65 Retrieving counties
Location + hours Are there food pantries in Derby open on Wednesdays? 13 0.38 0.38 Retrieving counties
Location + ID requirement Find food pantries in Maize that do not require any ID. 13 0.15 0.30 ID constraint errors
Ambiguous location Where can I get free food near me? 1 0.00 0.00 Needs clarification
Recall-based exact search Where is the pantry I went to in 67214 on 21st Street? 4 1.00 1.00 No errors
Total 50 0.46 0.52
Kansas Food Source 
JSON Database
Data Preparation Semantic RetrievalResponse Generation
Vector 
EmbeddingsIndexing for 
Similarity SearchRelevant 
Pantries 
Retrieved
LLM Assistant"Urgent food 
in Wichita ?"
"CC ICT Daily Bread
2825 S Hillside  Wichita
Open Tue, Wed, Thurs
 Call (316) 264 -8344"
Figure 1: Our proposed Pantry Assistant. Food pantries are
scraped from Kansas Food Source website and stored into
a structured JSON database. Data is then embedded and in-
dexed for similarity search to enable semantic retrieval. The
retrieved relevant pantries are then passed to an LLM, which
recommends pantries to users based on their requirements.
for pantry entries matching those needs. This step is critical to
restrict the LLM from recommending hallucinated or irrelevant
information to the user. RAG allows the LLM to access and retrieve
relevant documents from given data [ 7], a technique which has
been significant in efficient IR in low-resource settings [ 2,3,8]. The
retrieved relevant pantry records are then passed to the LLM.
Response Generation.The LLM-powered assistant now grounds
its response in the pantry data that it retrieves. Our predefined
system prompt instructs the LLM to use the data directly and avoid
fabrication. The chatbot should also include essential details of
the suggested pantries (e.g., open hours, addresses, etc.) to pro-
vide a complete recommendation according to the user needs. For
generating the response, we tested two LLMs. The first one is an
open-source model, Meta Llama 3.1 8B Instruct [ 5], chosen for its
accessibility and alignment with low-resource deployment. It can
be run via a HuggingFace4pipeline free of cost. The second one is
GPT-3.5 Turbo [ 1], a proprietary model by OpenAI known for its
strong language understanding capabilities. We included both mod-
els to compare performance and assess whether an open-source
solution can work in a low-resource setting instead of a state-of-the-
art proprietary model. The assistant is deployed through a simple
4https://huggingface.co/web interface built with Gradio5, making it easy for users to inter-
act with the chatbot in a browser without any installation. This is
helpful for users with limited devices or technical skills. We also
could quickly share the tool with community partners for feedback.
3 Preliminary Results and Challenges
We conducted an initial evaluation of the Pantry Assistant using
50 simulated user queries representing diverse scenarios, includ-
ing location-specific requests, ambiguous queries, constraint-based
questions (e.g., day or ID requirements), and recall-based questions
referencing specific addresses or pantry names (Table 1). A response
was considered accurate if it returned correct and relevant infor-
mation from the curated dataset. Llama 3.1 8B Instruct answered
23/50 queries correctly (46%), while GPT-3.5 Turbo answered 26/50
(52%), yielding an average accuracy of 48%. These results establish
an initial baseline, with both models performing comparably.
We conducted an error analysis and found that most failures stemmed
from three recurring issues: (a) incorrect resolution of county-level
location constraints, which led to retrieving pantries outside the
intended region; (b) lack of clarification for underspecified queries,
particularly those missing location information; and (c) difficulties
handling eligibility constraints (e.g., ID requirements) and incon-
sistencies in knowledge base formatting. These findings highlight
retrieval limitations that motivate further improvements.
4 Future Directions and Conclusion
Our analysis indicates that retrieval robustness, constraint han-
dling, and clarification dialogue are the primary bottlenecks in low-
resource conversational IR. Future work will strengthen semantic
retrieval through structured constraint modeling (e.g., geographic
and eligibility filters), introduce clarification-aware dialogue for
underspecified queries, and expand and normalize data sources to
improve coverage and consistency. We also plan to conduct more
structured user evaluations with community stakeholders. This
work frames public service information access as a low-resource
IR problem characterized by noisy data, incomplete coverage, and
evolving constraints. The Kansas food pantry case study offers a
realistic testbed for examining retrieval challenges in socially sensi-
tive domains. Our findings highlight the need for constraint-aware
retrieval strategies to improve access to critical public resources.
5https://www.gradio.app/

Retrieval Challenges in Low-Resource Public Service Information: A Case Study on Food Pantry Access Conference’17, July 2017, Washington, DC, USA
References
[1]Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Floren-
cia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal
Anadkat, et al .2023. Gpt-4 technical report.arXiv preprint arXiv:2303.08774(2023).
[2]Olusesan Michael Awoleye, Oluwamayowa Adebisi, Idowu Ademola Atoloye,
Atanda Samuel Oladejo, Abiodun T Atoloye, Cornelius Talade Atere, and Victoria
Tanimonure. 2025. Leveraging Artificial Intelligence-Powered Virtual Assistant for
Information Retrieval in Indigenous Agriculture: Insights from Nigeria. InProceed-
ings of the 48th International ACM SIGIR Conference on Research and Development
in Information Retrieval. 3095–3097.
[3]Jean Petit BIKIM, Charles Loic Njiosseu, Emmanuel Leuna FIENKAK, Azanzi
Jiomekong, and Sören Auer. 2025. Fair Access to Food Data in Africa: An Approach
Based on Retrieval-Augmented Generation. InProceedings of the 48th International
ACM SIGIR Conference on Research and Development in Information Retrieval. 3122–
3124.[4]Matthijs Douze, Alexandr Guzhva, Chengqi Deng, Jeff Johnson, Gergely Szilvasy,
Pierre-Emmanuel Mazaré, Maria Lomeli, Lucas Hosseini, and Hervé Jégou. 2025.
The faiss library.IEEE Transactions on Big Data(2025).
[5]Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad
Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Amy Yang, Angela Fan,
et al. 2024. The llama 3 herd of models.arXiv e-prints(2024), arXiv–2407.
[6]Maureen Henninger. 2013. The value and challenges of public sector information.
Cosmopolitan Civil Societies: An Interdisciplinary Journal5, 3 (2013), 75–95.
[7]Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin,
Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, et al .
2020. Retrieval-augmented generation for knowledge-intensive nlp tasks.Advances
in neural information processing systems33 (2020), 9459–9474.
[8]Joseph P Telemala, Neema N Lyimo, Anna R Kimaro, and Camilius A Sanga. 2025.
Towards Enhanced Agricultural Information Access in Kiswahili: Integrating
Knowledge Graphs and Retrieval-Augmented Generation. InProceedings of the 48th
International ACM SIGIR Conference on Research and Development in Information
Retrieval. 3104–3106.