# A Comparative Study of Retrieval Methods in Azure AI Search

**Authors**: Qiang Mao, Han Qin, Robert Neary, Charles Wang, Fusheng Wei, Jianping Zhang, Nathaniel Huber-Fliflet

**Published**: 2025-12-08 22:20:02

**PDF URL**: [https://arxiv.org/pdf/2512.08078v1](https://arxiv.org/pdf/2512.08078v1)

## Abstract
Increasingly, attorneys are interested in moving beyond keyword and semantic search to improve the efficiency of how they find key information during a document review task. Large language models (LLMs) are now seen as tools that attorneys can use to ask natural language questions of their data during document review to receive accurate and concise answers. This study evaluates retrieval strategies within Microsoft Azure's Retrieval-Augmented Generation (RAG) framework to identify effective approaches for Early Case Assessment (ECA) in eDiscovery. During ECA, legal teams analyze data at the outset of a matter to gain a general understanding of the data and attempt to determine key facts and risks before beginning full-scale review. In this paper, we compare the performance of Azure AI Search's keyword, semantic, vector, hybrid, and hybrid-semantic retrieval methods. We then present the accuracy, relevance, and consistency of each method's AI-generated responses. Legal practitioners can use the results of this study to enhance how they select RAG configurations in the future.

## Full Text


<!-- PDF content starts -->

XXX -X-XXXX -XXXX -X/XX/$XX.00 ©20XX IEEE  
 A Comparative Study of Retrieval Methods in Azure 
AI Search  
 
 
Qiang Mao  
Legal Technology  & Data 
Analytics  
Ankura Consulting Group, LLC  
Washington, D.C. USA  
qiang.mao@ankura.com  Han Qin 
Legal Technology  & Data 
Analytics   
Ankura Consulting Group, LLC  
Washington, D.C. USA  
han.qin@ankura.com  Robert Neary  
Legal Technology  & Data 
Analytics   
Ankura Consulting Group, LLC  
Washington, D.C. USA  
robert.neary @ankura.com  Charles Wang  
Legal Technology  & Data 
Analytics   
Ankura Consulting Group, LLC  
Washington, D.C. USA  
charles.wang@ankura.com
Fusheng Wei  
Legal Technology  & Data 
Analytics   
Ankura Consulting Group, LLC  
Washington, D.C. USA  
fusheng.wei @ankura.com  Jianping Zhang  
Legal Technology  & Data 
Analytics   
Ankura Consulting Group, LLC  
Washington, D.C. USA  
jianping.zhang @ankura.com   Nathaniel Huber -Fliflet  
Legal Technology  & Data 
Analytics   
Ankura Consulting Group, LLC  
London, UK  
nathaniel.huber -
fliflet@ankura.com   
 
 
 
 
Abstrac t – Increasingly , attorneys are interested in  
moving beyond keyword and semantic search  to improve 
the efficiency of how they find key information during a 
document review task.   Large language  models (LLMs)  are 
now seen as tools  that attorneys can use  to ask natural  
language questions of their d ata during document review to 
receive accurate  and concise answers.  This study evaluates 
retrieval strategies within Microsoft  Azure’s R etrieval -
Augmented  Generation (RAG) framework to identify 
effective  approaches for Early Case Assessment (ECA) in 
eDiscovery . During ECA, legal teams analyze data at the 
outset of a matter to  gain a general understanding of the 
data  and attempt to determine key facts  and risks  before 
beginning full -scale review. In this paper , we compare  the 
performance of  Azure AI Search’s k eyword, semantic, 
vector, hybrid, and hybrid -semantic  retrieval methods . We  
then present the accuracy, relevance, and consistency of  
each method’s AI-generated responses.  Legal practitioners  
can use the results of this study to enhance how they select 
RAG  configurations  in the future . 
Keywords  – LLM, RAG , Search Index , Retrieval , Early 
Case A ssessment , ECA, Retrieval -Augmented Generation , 
Legal Document Review  
I. INTRODUCTION  
Early Case Assessment  (ECA) is the first phase of the 
eDiscovery  process  and focuses on rapidly reviewing key 
electronic data to assess the facts and risks of a legal matter. 
Effective ECA allows  legal teams to gain an understanding of 
the data and make informed strategic decisions  such as whether 
to settle, litigate, or attempt to narrow the issues. ECA involves 
identifying, collecting, and  analyzing electronically stored 
information (ESI) to surface key facts, scope the issues, and 
defensibly reduce non -relevant data before full review. This process helps legal teams estimate potential costs, narrow 
downstream workflows, and assess likely case outcomes.  
Recently, l egal t eams have started  using  Generative AI 
(GenAI)  to analyze large volumes of data more efficiently when 
compared to traditional keyword or semantic search methods. 
Law firms increasingly expect tools that can support  GenAI 
enabled ECA  workflows such as case analysis, issue 
identification, document classification, and document 
summarization  through natural -language interaction with their 
data.   
Retrieval -Augmented Generation (RAG) is a prominent 
approach that enhances the capabilities of large language models 
(LLMs) by incorporating external knowledge retrieved from 
indexed data sources and grounding the model’s responses in 
that information. RAG  combines two key components: a 
retriever that fetches relevant documents and a generator that 
synthesizes responses based on both the prompt and retrieved 
content.  
However, the effectiveness of these systems depends heavily 
on the retrieval component that selects which documents are 
provided to the model. Unlike keyword search, where the 
behavior is well understood, the accuracy of vector, semantic, 
hybrid, and reran ked retrieval methods in real eDiscovery -like 
conditions remains unclear. Their performance can vary widely 
depending on the structure of the query, the vocabulary used, 
and the distribution of evidence across a dataset.  
This uncertainty creates real risk for legal workflows. If a 
retrieval method fails to surface the right documents, a GenAI 
system may return incomplete or incorrect answers, 
undermining defensibility in areas such as privilege 
identification, issue analys is, and early case assessment. Legal 
teams need empirical evidence showing how these retrieval 
methods behave so they can choose configurations that 

maximize recall, reduce error rates, and support accurate, 
defensible analysis.  
This paper evaluates  RAG  implementation s on Microsoft 
Azure, using  Azure AI Search as the retriever and Azure 
OpenAI as the generator . We evaluate how different retrieval 
methods affect the accuracy and relevance of generated outputs.  
II. BACKGROUND AND AZURE RAG  ARCHITECTURE  
Retrieval -Augmented Generation (RAG) is a framework 
that enhances LLMs by incorporating retrieved documents into 
the generation process. Instead of relying solely on the model's 
internal knowledge, RAG retrieves relevant chunks of external 
data and feeds t hem into the LLM alongside the user prompt. 
This approach can improve factual accuracy , allow domain -
specific customization , and reduce hallucination s by grounding 
responses in retrieved content.  
The Azure RAG pipeline includes data ingestion, 
preprocessing, indexing, retrieval and ranking, generation, and 
response parsing.  Each step plays a critical role in ensuring the 
relevance and accuracy of the final output.  
Azure AI Search supports several retrieval strategies: 
Keyword, Keyword -Semantic, Vector, Hybrid, and Hybrid -
Semantic  [1].  
• Keyword  search  uses a traditional lexical approach to 
match query terms against indexed text and rank 
documents based on term frequency and inverse 
document frequency.  
• Keyword -Semantic search  begins with keyword 
retrieval and then applies a semantic scoring model 
to improve the ordering of those keyword results.  
• Vector search  embeds both the query and the 
indexed text into a numerical vector space and 
retrieves documents based on cosine similarity, 
enabling conceptual matching even when exact terms 
do not appear.  
• Hybrid search  combines keyword and vector 
retrieval by merging the top results from both 
methods to capture exact matches and semantic 
matches simultaneously.  
• Hybrid -Semantic search  extends hybrid retrieval by 
applying a semantic reranker that reorders the 
merged results based on deeper contextual similarity 
to the query.  
 
Each method offers a trade -off between lexical matching, 
semantic understanding, and computational cost. Semantic 
reranking reorders retrieved documents based on similarity to 
the query and typically improves relevance but at additional 
computational  cost. 
Table 1 Azure AI Search Retrieval  Methods  Retrieval 
Method  Lexical 
Match  Semantic 
Understanding  Vector 
Similarity  Reranking  Cost 
Impact  
Keyword  ✓ ✗ ✗ ✗ Low 
Keyword -
Semantic  ✓ ✓ ✗ ✓ Medium  
Vector  ✗ ✗ ✓ ✗ Low 
Hybrid  ✓ ✗ ✓ ✗ Low 
Hybrid -
Semantic  ✓ ✓ ✓ ✓ High  
Table 1  summarizes the key characteristics of each retrieval 
strategy supported by Azure AI Search. It compares their 
reliance on lexical matching, semantic interpretation, vector 
similarity, and reranking, as well as the relative cost impact 
associated with each  method.  
III. EXPERIMENTS  
We used the  Jeb Bush email dataset  compri sed of over 
290,000 text documents.  It’s publicly available  and considered  
comparable to the data present in most modern eDiscovery 
projects . After c hunking each document into 2,000 -token 
segments with a 500 -token overlap , the dataset yielded 491,482 
chunks. We embedded each chuck  using the 'text -embedding -
ada-002' model with a vector dimension of 1536.  We used 'gpt-
4.1-mini'  as our LLM . All LLM generation was performed with 
the temperature set to zero to provide  deterministic outputs and 
isolate the impact of retrieval differences. For each query, the 
top 50 chunks were retrieved, and the top 5 were sent to the 
LLM.  
We prepared the following prompts  to demonstrate the 
effects of different retrieval methods : 
Table 2 The Prompts  
P1 What is case no. TO 98 -103033, 32 ? 
P2 What were the projected volumes of orange production in 
Florida, and how would they affect Florida's economy?  
P3 Was D&B Television doing business in West Palm Beach?  
P4 Who is T -Squred?  
P5 Who is 'T -Squred'?  
P6 Find emails discussing differences between Jeb Bush and  
George Bush's position on climate change and environmental 
policy using natural language concepts, even if those exact 
terms aren't used . 
P7 What were Jeb Bush’s views on the H -1B visa program during 
his tenure as governor of Florida?  
P8 What were Jeb Bush’s views on the H -1B visa program during 
his tenure as governor of Florida? Please ensure that all 
responses reflect Jeb Bush’s own views specifically, not those 
of other individuals or political figures.  
P9 What were Jeb Bush’s views on the H1B visa program during 
his tenure as governor of Florida? Please ensure that all 
responses reflect Jeb Bush’s own views specifically, not those 
of other individuals or political figures, including President  
Bush.  
 

 
We evaluated each retrieval method on relevance, factual 
accuracy, and diversity o f generated responses. These 
assessments were qualitative and based on manual review of 
retrieved chunks and model responses.  
 
IV. RESULTS AND DISCUSSION  
In the following, we discuss the RAG responses to prompts 
using different retrieval methods.   
Prompt P1 : The case number is a very specific 
alphanumeric string that does not contain ordinary words. Since 
the case number is mentioned in the emails, it becomes a 
keyword indexed by AI search. Unsurprisingly, methods 
utilizing keyword search enable RAG to respo nd with correct 
references to the emails. Conversely, the vector embedding of 
the prompt does not embed the alphanumeric case number as a 
meaningful semantic feature , causing the vector retrieval 
method to fail to locate the emails.  
Prompt P2 : This scenario is somewhat the opposite of P1. 
The vector retrieval method successfully identified emails 
containing specific orange product forecast numbers, whereas 
the keyword method failed to locate such forecasts. 
Additionally, the keyword semantic m ethod did not identify 
them either, even though  both methods partially addressed the 
economic impact portion of the query . The failure of the 
semantic method suggests that the keyword search did not return 
relevant information within its 50 result s; because semantic 
reranking  only reorders keyword search results , it cannot 
recover documents that keyword retrieval does not surface . 
Prompt P3 : An email message indicates that the company 
D&B is conducting business in South Florida. This information 
was overlooked by both the keyword and vector methods, 
possibly due to the lack of semantic associat ion between “ West 
Palm Beach ” and “South Florida” in the embedding space  and 
the original ranking of the keyword search. After semantic 
reranking, this email was moved to the top and subsequently 
picked up by the LLM step.  
Prompt P4 : T-Squired is the nickname of an indivi dual 
referenced in the dataset . Due to its unique meaning, similar to 
the outcome in P3, both simple and vector retrieval failed to 
extract the correct information. However, the semantic method 
was successful.  
Prompt P5 : By enclosing T -Squared in quotes, the keyword 
method now locates the correct information, while the vector 
method continues to fail.  
Prompt P6 : This example shows that while keyword, 
vector, and keyword semantic methods all stated that the 
requested information is unavailable, the hybrid and hybrid -
semantic methods cited emails from others revealing a “nuanced 
difference.” Upon reviewing the cited emails, the responses 
were partially  relevant but did not directly answer the question . 
Prompt P7 : While keyword, vector, and hybrid methods all 
indicated they could not find information about the governor's 
stance on H1 -B, the keyword semantic and hybrid semantic 
methods erroneously attributed others' positions to Jeb Bush, as shown in the citations. Upon examining the citations in the LLM 
response, it appears that re -ranking elevated a text chunk to the 
top retrieval results, which was part of speech given by President 
George Bush. Because the LLM lacks source awareness, it 
inferred that the retrieved text represented Jeb Bush’s views.  
This illustrates that while semantic re -ranking can aid in 
selecting top documents for the LLM, it can also lead to 
unintended consequences.  
Prompt P8 : This prompt repeats the question from P7, 
requesting the LLM to verify the answer. The same text chunk 
was selected, and because the same chunk was retrieved, the 
LLM reproduced the same attribution error.  
Prompt P9 : Finally, noting that the citations from P7 and P8 
mistakenly attributed George Bush’s views on H1 -B to Jeb 
Bush, the prompt was further refined. This time, the erroneous  
chunk no longer  appear ed in the top retrieval results, and no 
views were attributed to Jeb Bush.  
We observed that Keyword retrieval delivered  faster results 
faster than other methods but misse d semantically relevant 
material . Vector retrieval capture d semantic similarity but 
occasionally return ed off-topic chunks for infrequent keywords. 
Hybrid Semantic produced the most consistently relevant results 
but at a higher computational cost.  
 
Appendix A provides the full prompt set and the generated 
outputs for each retrieval method. These example results 
illustrate the retrieval behavior directly, provide visibility to the 
observations  reported in th is work , and illustrate  how specific 
retrieval methods influenced the LLM’s grounded responses.  
 
These observations are a result of our specific experimental 
configuration  choices . Adjusting retrieval sett ings, chunking 
parameters, or model versions may yield different outcomes. 
Nonetheless, these cases demonstrate core retrieval beha viors 
that practitioners should expect to see when using certain RAG 
implementations.  
 
V. CONCLUSIONS  
This study demonstrates  that the choice of retrieval method  
in Azure RAG ha s a direct impact on the accuracy, relevance, 
and defensibility of AI-generated responses. Our prompt -level 
analysis  shows that no single  method perform s consistently 
across all scenarios; instead, each method succeeds or fails based 
on the nature of the query and the distribution of information in 
the dataset.  Many errors attributed to “LLM hallucination” were  
rooted in retrieval method design deficiencies, underscoring that 
retrieval quality is the primary determinant of downstream 
model  performance . 
Several patterns emerged  across the experiments. Keyword -
based retrieval remains essential for struct ured, non -semantic 
data, such as case numbers , IDs, and nicknames, while vector 
retrieval excels at locating conceptually related  content but 
struggles with domain -specific or infrequent terms.  Semantic 
reranking improved recall in many cases but also elevated 
misleading or tangential text when the initial results lacked 
relevant material. These have clear implications for legal 
workflows, particularly in hi gh-stakes contexts such as privilege 

identification, issue analysis, and ECA . Retrieval errors , such as 
elevating misattributed information or tangential text , can lead 
the LLM to generate confident but incorrect answers . Without 
transparent citations and auditable retrieval paths, these failures 
undermine defensibility and increase the risk of relying on 
incorrect information in litigation or regulatory matters. The 
variability across retrieval strategies reinforces the need for 
multi -method retrieval options, calibrated reranking, and clear 
visibility into what content the system uses to support its 
answers.  
For eDiscover y workflows , these findings reinforce the 
importance of giving lawyers  a clear understanding of the 
implications of RAG  method selection , surfacing retrieval 
scores, and providing visibility into the source of the retrieved 
content. They also show that prompt phrasing has a strong  
impact on  retrieval behavior, meaning that user input , not only 
system configuration, affects output performance . VI. FUTURE  WORKS  
Future work will include  visualizing retrieval and reranking 
behavior to help u sers understand why specific documents 
surfaced, incorporating and testing feedback loops to inform a 
system to adaptively select more efficient  retrieval methods over 
time, and benchmarking retrieval strategies across  larger  and 
more diverse datasets  to validate performance in real -world 
legal scenarios . Another area of exploration will use an agentic 
system t o evaluate each query and select the retrieval strategy 
(e.g., keyword , vector, hybrid, semantic)  that would likely  
produce the most accurate results . Such systems could also 
review and verify the LLM -generated result . These 
advancement s could  help practitioners deploy RAG systems that 
are more accurate, predictable , and defensible in high -stakes 
environments.  
REFERENCES  
[1] Microsoft Learn, January 2025  “Develop  a RAG solution – Information -
Retrievl Phase ” https://learn.microsoft.com/en -us/azure/architecture/ai -
ml/guide/rag/rag -information -retrieval?source=recommendations  
 
 
APPENDIX A. RAG  EXPERIMENT OUTPUT   
 
 


 
 


 
