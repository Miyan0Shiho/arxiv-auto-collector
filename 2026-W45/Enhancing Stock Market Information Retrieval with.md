# Enhancing Stock Market Information Retrieval with Retrieval-Augmented Generation

**Authors**: Huan Phong Thai, Thu Anh Ngoc Le, Nhu Huynh Thi Le, Nhan Huu Tran, Lenh Cong Phan Pham, Hong Nhung Nguyen

**Published**: 2026-11-02 05:49:37

**PDF URL**: [https://arxiv.org/pdf/2611.03747v1](https://arxiv.org/pdf/2611.03747v1)

## Abstract
Natural Language Processing (NLP) has been widely applied in financial domains, enabling automated analysis and retrieval of stock market information. However, traditional NLP-based systems often struggle with maintaining contextual accuracy and generating relevant responses in dynamic financial environments. To address these challenges, Retrieval-Augmented Generation (RAG) models have emerged as a powerful solution, combining retrieval-based methods with generative capabilities to enhance response quality. In this study, we present a Retrieval-Augmented Generation (RAG) system designed to generate responses related to stock market information. The system integrates retrieval-based and generative approaches to enhance the accuracy and relevance of financial text generation. To evaluate its effectiveness, we employ the ROUGE metric, specifically ROUGE-1, ROUGE-2, and ROUGE-L, to measure lexical and structural similarity between generated and reference texts. Experimental results on a dataset of 100 stock market-related queries demonstrate high ROUGE scores, indicating strong alignment with reference documents. Additionally, we assess latency to evaluate the system responsiveness, achieving an average response time of 1.16 seconds. These findings confirm that our RAG model is capable of providing accurate and timely stock market insights, making it a promising approach for financial information retrieval.

## Full Text


<!-- PDF content starts -->

Enhancing Stock Market Information Retrieval with
Retrieval-Augmented Generation
Huan Phong Thaia, Thu Anh Ngoc Lea, Nhu Huynh Thi Lea, Nhan Huu
Trana, Lenh Cong Phan Phama, Hong Nhung Nguyena,∗
aDepartment of Software Engineering, FPT University, Can Tho, Viet Nam
Abstract
Natural Language Processing (NLP) has been widely applied in financial do-
mains, enabling automated analysis and retrieval of stock market information.
However, traditional NLP-based systems often struggle with maintaining con-
textual accuracy and generating relevant responses in dynamic financial envi-
ronments. To address these challenges, Retrieval-Augmented Generation (RAG)
models have emerged as a powerful solution, combining retrieval-based meth-
ods with generative capabilities to enhance response quality. In this study, we
present a Retrieval-Augmented Generation (RAG) system designed to generate
responses related to stock market information. The system integrates retrieval-
based and generative approaches to enhance the accuracy and relevance of fi-
nancial text generation. To evaluate its effectiveness, we employ the ROUGE
metric, specifically ROUGE-1, ROUGE-2, and ROUGE-L, to measure lexical
and structural similarity between generated and reference texts. Experimen-
tal results on a dataset of 100 stock market-related queries demonstrate high
ROUGE scores, indicating strong alignment with reference documents. Addi-
tionally, we assess latency to evaluate the system’s responsiveness, achieving an
average response time of 1.16 seconds. These findings confirm that our RAG
model is capable of providing accurate and timely stock market insights, making
∗Corresponding author.
Email addresses:huantpce180685@fpt.edu.vn(Huan Phong Thai),
thulnace180615@fpt.edu.vn(Thu Anh Ngoc Le),nhulthca182007@fpt.edu.vn(Nhu Huynh
Thi Le),nhanthce181655@fpt.edu.vn(Nhan Huu Tran),lenhppcce180059@fpt.edu.vn
(Lenh Cong Phan Pham),nhungnguyen.uet@gmail.com(Hong Nhung Nguyen )
Preprint submitted to Expert Systems with Applications November 4, 2026
arXiv:2611.03747v1  [cs.IR]  2 Nov 2026

it a promising approach for financial information retrieval.
Keywords:RAG, Information Retrieval, Stock Market
1. Introduction
Large Language Models (LLMs) Gemini Team et al. (2023) OpenAI (2023)
have made remarkable progress in recent years, establishing themselves as a fun-
damental component in various natural language processing (NLP) applications.
Their ability to understand and generate human-like text has led to significant
advancements in tasks such as machine translation, sentiment analysis, ques-
tion answering, and text summarization. As a result, LLMs have been widely
adopted across multiple domains, including education Kasneci et al. (2023),
healthcare Sallam (2023), finance Wu et al. (2023), code writing Chen et al.
(2021), cybersecurity Franco et al. (2020), the labor market Eloundou et al.
(2023), and legal aspects Sun (2023), where they enhance automation, improve
decision-making, and enable more natural human-computer interactions. One
of the most promising approaches in this domain is Retrieval-Augmented Gen-
eration (RAG), which combines the strengths of retrieval-based and generative
models to enhance the accuracy and reliability of generated content.
Despite the impressive capabilities of existing LLMs, their development is
still constrained by several challenges. One of the most significant issues is the
hallucination problem Dale et al. (2022), where models generate responses that
are fluent and coherent but factually inaccurate. Another major limitation is
the difficulty in updating their knowledge. Since LLMs store information within
their internal memory Meng et al. (2022), keeping them up to date requires
retraining or fine-tuning with new data—an expensive and resource-intensive
process. Additionally, general-purpose LLMs often lack expertise in specialized
domains Colombo et al. (2024) Zhang et al. (2023), making them less effective for
domain-specific tasks. Developing LLMs tailored to a particular field requires
extensive effort in dataset collection and model training, further increasing the
complexity of implementation. This makes the deployment of systems for spe-
2

cific tasks costly in terms of both time and resources. Furthermore, collecting
and processing training data for each specialized domain can present challenges
such as a lack of high-quality data, security requirements, and the complexity
of fine-tuning the model to fit the specific context.
Stock markets are highly dynamic, influenced by a wide range of factors, in-
cluding economic indicators, corporate financial reports, geopolitical events, and
investor sentiment. Efficiently retrieving relevant financial information and gen-
erating accurate insights is crucial for investors, analysts, and decision-makers.
Traditional NLP models often struggle with the complexity and ever-changing
nature of financial data, making it challenging to provide real-time, contextu-
ally accurate, and explainable outputs. To solve this problem, RAG presents a
potential solution by integrating retrieval mechanisms with generative capabil-
ities, allowing models to dynamically fetch the most relevant financial data and
generate informed responses.
Figure 1: System without RAG and System with RAG
Several recent studies Guu et al. (2020) Lewis et al. (2021) have proposed
leveraging external knowledge bases to enhance the ability of LLMs to gener-
ate better responses, a technique known as Retrieval-Augmented Generation
3

(RAG). By supplying LLMs with retrieved relevant factual information, the
hallucination problem can be alleviated to some extent. Moreover, the knowl-
edge update issue can also be addressed by updating the external knowledge
database, allowing LLMs to incorporate up-to-date information. RAG can also
transform a general LLM into a domain-specific LLM by constructing and uti-
lizing a specialized knowledge database. Therefore, RAG plays a crucial role in
enhancing the functionality of LLMs, making them more accurate, knowledge-
able, and reliable across a wide range of applications Wu et al. (2024). Fig. 1
illustrates the fundamental differences in workflow between a system without
RAG and a system enhanced with RAG, highlighting the impact of retrieval-
augmented knowledge integration.
This study aims to explore the effectiveness of RAG models in processing
stock market information. Specifically, we investigate how RAG can enhance
the retrieval of information and the generation of responses regarding stock mar-
ket news published on VnExpress. By leveraging external financial knowledge
sources, RAG models may enhance decision-making processes, reduce misin-
formation, and offer more comprehensive market summaries. In summary, the
structure of this paper is as follows: Section 2 reviews related work in financial
NLP and RAG applications. Section 3 describes the proposed methodology,
including data sources, model architecture, and evaluation metrics. Section 4
presents experimental results and discussions. Finally, Section 5 concludes the
paper with key findings and potential future research directions.
2. Related Work
2.1. NLP in Stock Market
Financial NLP has been extensively used to analyze stock market data, pre-
dict trends, and support investment decisions. Several studies have explored
the effectiveness of NLP models in processing financial iinformation. Financial
NLP has been extensively used to analyze stock market data, predict trends,
and support investment decisions. Several studies have explored the effective-
4

ness of NLP models in processing financial information. Akita et al. Akita
et al. (2016) utilized Long Short-Term Memory (LSTM) networks to predict
stock prices by combining textual and numerical data from the Nikkei news-
paper. Their research highlighted the importance of cross-company event cor-
relation in financial forecasting, demonstrating how news about one company
could impact the stock prices of others within the same industry. Similarly,
Sidra and Jaydip Sen & Mehtab (2021) explored multiple machine learning and
deep learning techniques for weekly stock market predictions, integrating sen-
timent analysis from social media with historical stock prices. Their findings
showed that public sentiment plays a crucial role in improving stock movement
predictions.
Another notable study proposed an LSTM-based forecasting model that in-
corporated sentiment analysis to establish correlations between stock values and
news sentiment. The results demonstrated that using NLP for feature extrac-
tion and LSTM for time-series training significantly improved prediction accu-
racy, reducing the mean square error to 0.062 Patel et al. (2021). Additionally,
”Smart Trader,” an AI-powered chatbot designed for the Saudi Tadawul mar-
ket, showcased the practical applications of NLP and machine learning in stock
trading. By leveraging real-time market data and user preferences, the chatbot
provided personalized investment recommendations, filling a gap in the market
for AI-driven financial advisory services Alhydary et al. (2024).
2.2. Retrieval-Augmented Generation (RAG) in NLP
Retrieval-Augmented Generation (RAG) is a hybrid approach that enhances
language models by integrating external knowledge retrieval mechanisms. In-
stead of relying solely on parametric memory, RAG retrieves relevant documents
or facts from external sources and conditions its responses based on retrieved
content. This method is particularly useful for knowledge-intensive NLP tasks
such as open-domain question answering and fact-based text generation Lewis
et al. (2020). By dynamically incorporating relevant information at inference
time, RAG models can mitigate issues like hallucination and outdated knowl-
5

edge, which are common challenges in traditional language models.
The architecture of RAG consists of two main components: a retriever mod-
ule and a generator module. The retriever module searches for relevant ex-
ternal knowledge, while the generator module synthesizes a response based on
the retrieved content. The retriever module usually comprises three compo-
nents: an encoder for encoding inputs into embeddings, an efficient indexing
system that supports approximate nearest neighbor search, and a datastore for
storing external knowledge in the form of key-value pairs Wu et al. (2024).
Beside, Retrieval-augmented (RA) generators and default generators are the
two primary types of generator modules. Default generators encompass most
pre-trained or fine-tuned large language models, including Mistral models Jiang
et al. (2023), Gemini-series models Gemini Team et al. (2023) Reid et al. (2024),
and GPT-series models OpenAI (2023) Zhang & Li (2021). In contrast, RA gen-
erators integrate retrieval mechanisms to enhance response generation, incorpo-
rating modules like Enc-Dec Li et al. (2022) and RETRO Wang et al. (2023).
These generators refine their outputs by leveraging retrieved knowledge, allow-
ing them to generate more contextually informed responses or predictions.
Recent studies have demonstrated the effectiveness of RAG in improving
factual accuracy and reducing misinformation in AI-generated text. For exam-
ple, RAG models have been used to enhance chatbot responses by retrieving
real-time information from external databases Lewis et al. (2020). Additionally,
advances in retrieval mechanisms, such as dense passage retrieval and contrastive
learning, have further improved the quality of retrieved content, leading to more
coherent and contextually relevant responses.
2.3. RAG in Financial and Stock Market Applications
Retrieval-Augmented Generation (RAG) has been increasingly applied in fi-
nancial domains to enhance decision-making by integrating real-time knowledge
retrieval with language model capabilities. In market sentiment analysis, fine-
tuned models like FinBERT within a RAG pipeline have been used to analyze
financial news and predict stock price movements, though sentiment alone shows
6

limited predictive power Echambadi (2025). Similarly, RAG improves financial
analysis by addressing challenges such as lack of interpretability in ML models
and hallucinations in LLMs. The AlphaFin dataset and Stock-Chain frame-
work have demonstrated the effectiveness of integrating retrieval mechanisms
for financial reasoning tasks Li et al. (2024).
Beyond sentiment analysis, RAG plays a crucial role in multi-modal finan-
cial analytics. The FinTMMBench benchmark evaluates temporal-aware RAG
systems that process diverse financial data sources, such as stock prices, news,
and technical charts, enabling more comprehensive trend analysis and event
detection Zhu et al. (2025). Additionally, in financial risk assessment, RAG en-
hances LLM performance in audit report analysis by improving faithfulness and
context relevance. Experiments show that models like LLaMa3.1 and Gemini-
1.5-flash significantly improve financial risk evaluations by extracting detailed
insights from reports?. These applications highlight the potential of RAG in
financial markets, bridging the gap between real-time information retrieval and
AI-generated financial insights.
3. Data Preparation
3.1. Data Collection
In this study, we collected data from VnExpress, a well-known Vietnamese
online news platform, to analyze stock market trends. The dataset comprises
30 news articles related to the stock market, which were crawled from Vn-
Express. These articles cover various aspects of the stock market, including
trading volume, stock price fluctuations, corporate financial performance, and
macroeconomic factors influencing market trends.
Each category represents a key dimension of the stock market. Market
Trends & Index Performance tracks overall market movement and index fluc-
tuations. Corporate Stocks & Sectors highlights individual stock performances
and sector-wide trends. Foreign Investment & Liquidity focuses on the role
of international capital in the Vietnamese stock market. Lastly, Regulatory
7

Table 1: Number of Main Documents
Category Number
Market Trends & Index Performance 8
Corporate Stocks & Sectors 9
Foreign Investment & Liquidity 7
Regulatory & Corporate Actions 6
Total 30
& Corporate Actions examines government policies, financial regulations, and
corporate decisions that influence market behavior.
Figure 2: Frequency of Removed Stop Words with Categorized Colors
This categorization helps in understanding the key trends and themes in
the stock market as reported by VnExpress. The data will be utilized for fur-
ther analysis to develop a Retrieval-Augmented Generation (RAG) system that
enhances stock market news comprehension and decision-making support.
3.2. Data Preprocessing
Stop words are commonly used words in a language that carry minimal
semantic meaning, such as conjunctions, prepositions, and auxiliary verbs. Re-
moving stop words is a crucial step in text preprocessing as it helps reduce
dimensionality, improve computational efficiency, and enhance the relevance of
8

text-based features for downstream natural language processing (NLP) tasks.
Fig 2 illustrates the frequency of removed stop words in the documents.
In this study, we implemented a stop word removal process specifically for
Vietnamese text. Given the unique characteristics of the Vietnamese language,
where words are often multi-syllabic and require tokenization before processing,
we utilized the Underthesea library for word segmentation. The preprocessing
pipeline consists of several key steps.
First, the input text was normalized by stripping leading and trailing whites-
paces. Additionally, all URLs and special characters were removed using regular
expressions to eliminate noise in the data. To ensure uniformity, all text was
converted to lowercase, which helps in minimizing variations due to capitaliza-
tion.
Next, since Vietnamese is not naturally space-separated like English, we
applied word tokenize from the Underthesea library to segment text into mean-
ingful tokens. Following tokenization, we curated a list of common Vietnamese
stop words and removed them from the tokenized text. This step reduces re-
dundancy and ensures that only semantically meaningful words are retained for
further analysis.
After performing these preprocessing steps, the cleaned text was stored for
subsequent stages of the information retrieval and retrieval-augmented genera-
tion (RAG) pipeline. The removal of stop words significantly enhances the qual-
ity of text embeddings by focusing on content-bearing words while discarding
non-informative tokens. This refinement is particularly beneficial for improving
the performance of semantic similarity searches and document retrieval tasks.
4. Proposed Methodology
This section presents the architecture of the proposed Retrieval-Augmented
Generation (RAG) system, which enhances large language model (LLM) re-
sponses by integrating an external knowledge retrieval mechanism. The system
consists of three main components: text embedding, information retrieval, and
9

response generation. Fig. 3 illustrates the overall workflow of the RAG system,
depicting how queries are processed through embedding, information retrieval,
and response generation.
Figure 3: Retrieval-Augmented Generation (RAG) System Architecture
4.1. Text Embedding
The embedding process converts textual documents into numerical vectors
using a pretrained embedding model. Fig 4 illustrates the process of embedding
textual data into numerical vector representations. As mentioned above, the
documents consist of stock market news articles crawled from VnExpress, one
of Vietnam’s most popular online newspapers. VnExpress provides up-to-date
financial news, market analysis, and expert opinions, making it a valuable data
source for financial insights.
Figure 4: Text Embedding Process: Converting Raw Text into Numerical Rep-
resentations for Efficient Retrieval
Specifically, we employ the LaBSE (Language-agnostic BERT Sentence Em-
bedding) Feng et al. (2020) model from the SentenceTransformers library. This
10

model is chosen for its multilingual capabilities, allowing it to effectively encode
text in various languages. Notably, LaBSE provides strong support for the Viet-
namese language, ensuring high-quality semantic representations of Vietnamese
financial news.
Documents are first preprocessed and tokenized using Jieba Zhang et al.
(2019), a widely used text segmentation tool primarily designed for Chinese.
Despite its origin, jieba has been adapted for Vietnamese and is used here to
segment sentences into meaningful tokens before embedding. Proper tokeniza-
tion is crucial for accurately representing text semantics.
The embeddings are then computed and stored in a FAISS (Facebook AI
Similarity Search) Douze et al. (2024) database, which enables efficient nearest-
neighbor retrieval. The generated embeddings form a high-dimensional vector
space that facilitates rapid similarity comparisons. Once all textual data has
been transformed into vector representations, they are systematically stored in
the FAISS DB store for efficient retrieval during the next processing stages.
4.2. Information Retrieval
Given a user query, the retrieval process aims to fetch the most relevant
documents from the FAISS database. The retrieval mechanism follows a hybrid
approach that integrates FAISS-based vector similarity search with BM25 (Best
Matching 25) Robertson (2010) ranking. This combination leverages both se-
mantic similarity and term frequency-based relevance to enhance the accuracy
of retrieved documents.
To perform vector-based retrieval, the input query is first embedded using
the LaBSE model, ensuring a dense numerical representation in the same vector
space as the stored documents. This query embedding is then compared against
the stored document embeddings using FAISS. FAISS efficiently retrieves the
top-k nearest neighbors by measuring the L2 distance (Euclidean distance) be-
tween the query and stored vectors. The L2 distance between two vectorsq
(query) andd(document) in an n-dimensional space is computed as follows:
11

d(q,d) =vuutnX
i=1(qi−di)2 (1)
where:q iandd irepresent the individual components of the query and
document embeddings, respectively. A smaller L2 distance indicates higher
similarity between the query and document.
After retrieving an initial set of candidate documents from FAISS, a BM25
ranking step is applied to further refine the results. BM25 is a TF-IDF-based
retrieval model that scores documents based on their word frequency, document
length normalization, and inverse document frequency (IDF). This step ensures
that retrieved documents are not only semantically similar but also contextually
relevant based on keyword occurrences. The BM25 score for a documentDgiven
a queryQis computed as:
BM25(D, Q) =X
t∈QIDF(t)·f(t, D)·(k 1+ 1)
f(t, D) +k 1·
1−b+b·|D|
avgD (2)
where:
•f(t, D) is the term frequency of termtin documentD.
•|D|is the length of documentD(total number of words inD).
•avgD is the average document length in the corpus.
•k1is a tuning parameter controlling term frequency saturation.
•bis a tuning parameter controlling document length normalization.
The inverse document frequency (IDF) component is defined as:
IDF(t) = logN−n(t) + 0.5
n(t) + 0.5(3)
where:
•Nis the total number of documents in the corpus.
•n(t) is the number of documents containing termt.
12

The BM25 ranking model is particularly effective in handling keyword-based
relevance, especially for queries where specific terms are important. By applying
BM25 after FAISS retrieval, we re-rank the initially retrieved documents to
prioritize those that best match the user’s intent.
This hybrid approach ensures that the final set of documents includes both
semantically related and keyword-relevant results, optimizing the retrieval pro-
cess for accuracy and contextual relevance.
4.3. Information Generation
The final stage of the system focuses on generating a response based on
the retrieved documents. We utilize the Gemini model, specifically Gemini-1.5-
Flash Jiang et al. (2023) Mondillo et al. (2024), due to its optimized performance
for real-time applications, lower latency, and ability to process long-context
inputs efficiently. This choice ensures that the model can effectively integrate
retrieved knowledge while maintaining responsiveness.
To generate a well-informed response, a structured prompt is carefully crafted
by incorporating the retrieved documents as contextual information. The prompt
is formatted to include a clear instruction, the user’s query, and the most rel-
evant retrieved texts, ensuring that the model leverages external knowledge
effectively.
Once the structured prompt is prepared, Gemini-1.5-Flash processes it and
generates a response that aligns factually with the retrieved information. The
model does this by grounding its output in the provided texts, minimizing hal-
lucinations, and maintaining consistency with real-world stocket market infor-
mations.
By leveraging this retrieval-augmented approach, the system significantly
enhances the quality of responses, ensuring that they are not only coherent and
contextually aware but also well-supported by factual knowledge.
13

5. Experimental Result
5.1. Evaluation Metrics
To assess the performance of our Retrieval-Augmented Generation (RAG)
system in generating responses related to stock market information, we employ
ROUGE (Recall-Oriented Understudy for Gisting Evaluation) Score as the pri-
mary evaluation metric. ROUGE is a widely used metric for evaluating text
summarization and generation tasks by comparing the overlap of n-grams, word
sequences, and longest common subsequences between the generated text and
reference answers.
We calculate ROUGE-1, ROUGE-2, and ROUGE-L scores, which are de-
fined as follows:
•ROUGE-1measures the overlap of unigrams (single words) between the
generated text and the reference text:
ROUGE-1 =P
w∈Wmin(count gen(w),count ref(w))P
w∈Wcount ref(w)(4)
•ROUGE-2extends this to bigrams (two consecutive words), capturing
more contextual meaning:
ROUGE-2 =P
b∈Bmin(count gen(b),count ref(b))P
b∈Bcount ref(b)(5)
•ROUGE-Levaluates the longest common subsequence (LCS) between
the generated and reference text, which captures sentence-level structure
and fluency:
ROUGE-L =LCS(gen,ref)
length(ref)(6)
Where:
•count gen(w) and count ref(w) are the counts of wordwin the generated
and reference texts, respectively.
•WandBare the sets of all unigrams and bigrams in the reference text.
14

•LCS(gen,ref) is the length of the longest common subsequence between
the generated and reference text.
•length(ref) is the length of the reference text.
In addition to ROUGE scores, we measure Latency, which quantifies the
system’s response time. Latency is defined as the time elapsed from the moment
a query is submitted to the system until a response is generated. This metric is
critical for assessing the real-time usability of our RAG system, particularly in
financial applications where timely information retrieval is crucial.
By combining accuracy-based ROUGE metrics with latency measurement,
we obtain a comprehensive evaluation of both the quality and efficiency of our
system’s responses.
5.2. Retrieval-Augmented Generation (RAG) Results
To evaluate the effectiveness of our RAG system, we computed ROUGE
scores based on a sample of 100 generated responses derived from stock market-
related documents. The average ROUGE scores obtained are presented in Tab
2
Table 2: ROUGE Score Results
ROUGE Metric Precision Recall F-measure
ROUGE-1 0.9616 0.9597 0.9555
ROUGE-2 0.9374 0.9356 0.9314
ROUGE-L 0.9496 0.9478 0.9436
These results indicate that the system achieves high lexical and structural
similarity between the generated responses and reference texts, demonstrating
the capability of the RAG model in preserving key stock market information
from the original documents. The high precision and recall values across all
ROUGE metrics suggest that the system generates responses that closely align
with the reference texts, ensuring both completeness and correctness in retriev-
ing stock market-related data.
15

Specifically, the ROUGE-1 score of 0.9555 (F-measure) reflects the system’s
ability to capture important individual keywords relevant to stock market dis-
cussions, ensuring that essential financial terms and concepts are accurately rep-
resented in the generated responses. Meanwhile, the ROUGE-2 score of 0.9314
indicates that the model effectively retains key bi-gram (two-word phrase) struc-
tures, which are crucial for maintaining contextual accuracy and conveying fi-
nancial insights with coherence.
Moreover, the strong ROUGE-L score of 0.9436 highlights that the system
maintains sentence structure and fluency, further reinforcing its reliability for
financial text generation tasks. These consistently high scores across all three
ROUGE metrics confirm the robustness of our approach in providing high-
quality, contextually accurate, and well-structured responses for stock market
information retrieval.
Apart from accuracy, latency performance was also measured to assess sys-
tem efficiency. Over 100 queries, the latency statistics are as follows Tab 3:
Table 3: Latency Results
Latency Metric Time (seconds)
Average Latency 1.1626
Minimum Latency 0.917
Maximum Latency 2.455
These latency results suggest that the system provides responses within an
acceptable time frame, balancing both accuracy and efficiency. The combination
of high ROUGE scores and low latency values demonstrates the effectiveness of
our RAG model for financial information retrieval. The relatively low response
time further supports its practical usability in real-world applications where
timely access to accurate stock market information is critical.
16

6. Conclusion
In this study, we evaluated the effectiveness of a Retrieval-Augmented Gen-
eration (RAG) system in generating responses related to stock market informa-
tion. By utilizing the ROUGE metric, we assessed the lexical and structural sim-
ilarity between the generated responses and reference texts. The high ROUGE-
1, ROUGE-2, and ROUGE-L scores demonstrate that the system is capable of
accurately retrieving and synthesizing financial data while maintaining coher-
ence and fluency in the responses. Additionally, our analysis of latency indicates
that the system operates within an acceptable response time, ensuring efficient
query processing for real-time financial applications. The combination of high
accuracy and low latency confirms that the proposed RAG model is a viable ap-
proach for handling stock market-related queries, providing reliable and timely
information. Future work will focus on further improving the model’s ability
to handle complex financial queries, integrating external knowledge sources to
enhance response accuracy, and optimizing computational efficiency to reduce
latency. These enhancements will contribute to making RAG-based systems
more effective for real-world financial analysis and decision-making.
CRediT authorship contribution statement
Thai Phong Huan:Conceptualization, Methodology, Software, Writing –
original draft, Writing – review & editing.Le Thi Anh Thu:Methodology,
Software, Writing – review & editing.Nhan Huu Tran :Methodology, Soft-
ware, Writing – review & editing.Lenh Cong Phan pham:Methodology,
Software, Writing – review & editing.Hong Nhung Nguyen:Supervision,
Methodology, Validation.
Declaration of Competing Interest
The authors declare that they have no known competing financial interests or
personal relationships that could have appeared to influence the work reported
in this paper.
17

References
Akita, R., Yoshihara, A., Matsubara, T., & Uehara, K. (2016). Deep learn-
ing for stock prediction using numerical and textual information. In2016
IEEE/ACIS 15th International Conference on Computer and Information
Science (ICIS). doi:10.1109/ICIS.2016.7550882.
Alhydary, T., Qashqary, L., Alotaibi, S., Jaha, R., & Alansari, S. (2024).
Smart trader: Towards an advisory chatbot service for saudi stock mar-
ket investment. In2024 Seventh International Women in Data Sci-
ence Conference at Prince Sultan University (WiDS PSU)(pp. 32–35).
doi:10.1109/wids-psu61003.2024.00021.
Chen, M. I.-C. et al. (2021). Evaluating large language models trained on code.
doi:10.48550/arxiv.2107.03374.arXiv:2107.03374.
Colombo, P. et al. (2024). SaulLM-7B: a pioneering large language model for
law.arXiv:2403.03883.
Dale, D. et al. (2022). Detecting and mitigating hallucinations in machine
translation: Model internal workings alone do well, sentence similarity even
better.arXiv:2212.08597.
Douze, M. et al. (2024). The Faiss library. URL:https://arxiv.org/abs/
2401.08281.arXiv:2401.08281.
Echambadi, V. (2025). Financial market sentiment analysis using LLM and
RAG.SSRN, . doi:10.2139/ssrn.4939798.
Eloundou, T., Manning, S., Mishkin, P., & Rock, D. (2023). GPTs are GPTs:
An early look at the labor market impact potential of large language models.
doi:10.48550/arxiv.2303.10130.arXiv:2303.10130.
Feng, F., Yang, Y., Cer, D., Arivazhagan, N., & Wang, W. (2020). Language-
agnostic BERT sentence embedding. doi:10.48550/arxiv.2007.01852.
arXiv:2007.01852.
18

Franco, M. F. et al. (2020). SecBot: A business-driven conversational agent
for cybersecurity planning and management. In2020 16th international
conference on network and service management (CNSM). IEEE.
Gemini Team et al. (2023). Gemini: A Family of Highly Capable Multimodal
Models. URL:https://arxiv.org/abs/2312.11805.arXiv:2312.11805.
Guu, K., Lee, K., Tung, Z., Pasupat, P., & Chang, M.-W. (2020). REALM:
Retrieval-augmented language model pre-training. URL:https://arxiv.
org/abs/2002.08909.arXiv:2002.08909.
Jiang, A. Q. et al. (2023). Mistral 7b.arXiv:2310.06825.
Kasneci, E. et al. (2023). ChatGPT for good? on opportunities and challenges of
large language models for education.Learning and Individual Differences,
103, 102274. doi:10.1016/j.lindif.2023.102274.
Lewis, P. et al. (2020). Retrieval-augmented generation for knowledge-intensive
NLP tasks. InAdvances in Neural Information Processing Systems 33(pp.
9459–9474).
Lewis, P. et al. (2021). Retrieval-augmented generation for knowledge-
intensive NLP tasks. URL:https://arxiv.org/abs/2005.11401.
arXiv:2005.11401.
Li, X. et al. (2024). AlphaFin: Benchmarking financial analysis with retrieval-
augmented stock-chain framework. URL:https://arxiv.org/abs/2403.
12582.arXiv:2403.12582.
Li, Z., Guo, R., & Kumar, S. (2022). Decoupled context processing for con-
text augmented language modeling. InAdvances in Neural Information
Processing Systems 35 (NeurIPS).
Meng, K. et al. (2022). Locating and editing factual associations in GPT. In
Advances in neural information processing systems 35(pp. 17359–17372).
19

Mondillo, G. et al. (2024). Basal knowledge in the field of pediatric
nephrology and its enhancement following specific training of ChatGPT-
4 ‘omni’ and Gemini 1.5 flash.Pediatric Nephrology, . doi:10.1007/
s00467-024-06486-3.
OpenAI (2023). GPT-4 Technical Report. doi:10.48550/arxiv.2303.08774.
arXiv:2303.08774.
Patel, R., Choudhary, V., Saxena, D., & Singh, A. K. (2021). LSTM and NLP
based forecasting model for stock market analysis. In2021 First Interna-
tional Conference on Advances in Computing and Future Communication
Technologies (ICACFCT). doi:10.1109/icacfct53978.2021.9837384.
Reid, M. et al. (2024). Gemini 1.5: Unlocking multimodal understanding
across millions of tokens of context. doi:10.48550/arxiv.2403.05530.
arXiv:2403.05530.
Robertson, S. (2010). The probabilistic relevance framework: BM25 and be-
yond.Foundations and Trends®in Information Retrieval,3, 333–389.
doi:10.1561/1500000019.
Sallam, M. (2023). The utility of ChatGPT as an example of large language
models in healthcare education, research and practice: Systematic review
on the future perspectives and potential limitations.medRxiv, . doi:10.
1101/2023.02.19.23286155.
Sen, J., & Mehtab, S. (2021). A robust predictive model for stock price predic-
tion using deep learning and natural language processing. doi:10.36227/
techrxiv.15023361.v1.
Sun, Z. (2023). A short survey of viewing large language models in legal aspect.
URL:https://arxiv.org/abs/2303.09136v1.arXiv:2303.09136v1.
Wang, B., Ping, W., McAfee, L., Xu, P., Li, B., Shoeybi, M., & Catanzaro,
B. (2023). InstructRetro: Instruction tuning post retrieval augmented pre-
training.arXiv:2310.07713.
20

Wu, S., Xiong, Y., Cui, Y., Wu, H., Chen, C., Yuan, Y., Huang, L., Liu, X.,
Kuo, T., Guan, N., & Xue, C. (2024). Retrieval-augmented generation for
natural language processing: A survey.arXiv:2407.13193.
Wu, S. et al. (2023). BloombergGPT: A large language model for finance.
doi:10.48550/arxiv.2303.17564.arXiv:2303.17564.
Zhang, H. et al. (2023). HuatuoGPT, towards taming language model to be a
doctor. doi:10.48550/arxiv.2305.15075.arXiv:2305.15075.
Zhang, M., & Li, J. (2021). A commentary of GPT-3 in MIT technology review
2021.Fundamental Research,1, 831–833. doi:10.1016/j.fmre.2021.11.
011.
Zhang, X., Wu, P., Cai, J., & Wang, K. (2019). A contrastive study of chi-
nese text segmentation tools in marketing notification texts.Journal of
Physics: Conference Series,1302, 022010. doi:10.1088/1742-6596/1302/
2/022010.
Zhu, F. et al. (2025). FinTMMBench: Benchmarking temporal-aware multi-
modal RAG in finance.arXiv:2503.05185.
21