# StreamLink: Large-Language-Model Driven Distributed Data Engineering System

**Authors**: Dawei Feng, Di Mei, Huiri Tan, Lei Ren, Xianying Lou, Zhangxi Tan

**Published**: 2025-05-27 07:44:16

**PDF URL**: [http://arxiv.org/pdf/2505.21575v1](http://arxiv.org/pdf/2505.21575v1)

## Abstract
Large Language Models (LLMs) have shown remarkable proficiency in natural
language understanding (NLU), opening doors for innovative applications. We
introduce StreamLink - an LLM-driven distributed data system designed to
improve the efficiency and accessibility of data engineering tasks. We build
StreamLink on top of distributed frameworks such as Apache Spark and Hadoop to
handle large data at scale. One of the important design philosophies of
StreamLink is to respect user data privacy by utilizing local fine-tuned LLMs
instead of a public AI service like ChatGPT. With help from domain-adapted
LLMs, we can improve our system's understanding of natural language queries
from users in various scenarios and simplify the procedure of generating
database queries like the Structured Query Language (SQL) for information
processing. We also incorporate LLM-based syntax and security checkers to
guarantee the reliability and safety of each generated query. StreamLink
illustrates the potential of merging generative LLMs with distributed data
processing for comprehensive and user-centric data engineering. With this
architecture, we allow users to interact with complex database systems at
different scales in a user-friendly and security-ensured manner, where the SQL
generation reaches over 10\% of execution accuracy compared to baseline
methods, and allow users to find the most concerned item from hundreds of
millions of items within a few seconds using natural language.

## Full Text


<!-- PDF content starts -->

arXiv:2505.21575v1  [cs.DB]  27 May 2025StreamLink: Large-Language-Model Driven Distributed Data
Engineering System
Dawei Feng
Tsinghua University
Beijing, China
fdw22@mails.tsinghua.edu.cnDi Mei
Tsinghua University
Beijing, China
di.mei@rioslab.orgHuiri Tan
Tsinghua University
Beijing, China
huiri.tan@rioslab.org
Lei Ren
Tsinghua University
Beijing, ChinaXianying Lou
King & Wood Mallesons
Shanghai, ChinaZhangxi Tan
Tsinghua University
Beijing, China
Abstract
Large Language Models (LLMs) have shown remarkable proficiency
in natural language understanding (NLU)[ 1], opening doors for in-
novative applications. We introduce StreamLink - an LLM-driven
distributed data system designed to improve the efficiency and ac-
cessibility of data engineering tasks. We build StreamLink on top
of distributed frameworks such as Apache Spark[ 2] and Hadoop
to handle large data at scale. One of the important design philoso-
phies of StreamLink is to respect user data privacy by utilizing local
fine-tuned LLMs instead of a public AI service like ChatGPT. With
help from domain-adapted LLMs, we can improve our system’s
understanding of natural language queries from users in various
scenarios and simplify the procedure of generating database queries
like the Structured Query Language (SQL) for information process-
ing. We also incorporate LLM-based syntax and security checkers
to guarantee the reliability and safety of each generated query.
StreamLink illustrates the potential of merging generative LLMs
with distributed data processing for comprehensive and user-centric
data engineering. With this architecture, we allow users to interact
with complex database systems at different scales in a user-friendly
and security-ensured manner, where the SQL generation reaches
over 10% of execution accuracy compared to baseline methods,
and allow users to find the most concerned item from hundreds of
millions of items within a few seconds using natural language.
Keywords
Distributed Database, Large Language Model, SQL Generation,
LLM-Driven SQL Checker
1 Introduction
Big data is now a key focus for both government and business
leaders.[ 3]. However, buried within this immense data deluge lies
an abundance of untapped potential and valuable insights, which
has given rise to an innovative scientific paradigm known as data-
intensive scientific discovery[ 4]. Researchers actively seek ways to
leverage available data to gain valuable insights and inform decision-
making. On the one hand, big data offers substantial value, fostering
business productivity and catalyzing revolutionary breakthroughs
in scientific disciplines. On the other hand, the utilization of big
data is accompanied by challenges, ranging from the complexities
of data capture[ 5], storage[ 6], and analysis to the intricacies of data
visualization[7].The prerequisite for realizing big data applications is a robust
data system managing external queries as well as information re-
trieval. In order to efficiently and securely handle a large volume of
data, we introduce StreamLink, an AI-powered distributed data sys-
tem with enhanced ability to process billions of data records while
reducing user operational costs. In addition to the support from
a scalable and dependable distributed database, one exceptional
feature of this system is its highly accessible and security-oriented
interaction with users, which is contributed by the use of the latest
Large Language Models (LLMs) with the outstanding capability
of language generation and domain adaptation. To illustrate its
performance, we deploy the proposed system to store global patent
data, where most of them come from the United States Patent and
Trademark Office (USPTO)1and Google Patents2. There are approx-
imately 180 million patents in this system and patents are growing
rapidly, with the USPTO Patent Assignment Dataset[ 8] containing
around 6 million assignments and transactions recorded between
1970 and 2014, affecting about 10 million patents or patent applica-
tions. We have also validated Streamlink’s robustness through the
collaboration with the patent and intellectual property (IP) team at
King & Wood Mallesons, who is invited to experience our system
and provide any usage feedback.
In this paper, we make the following contributions:
•LLM-based NL to SQL[ 9] Generator : While numerous
studies[ 10][11][12] have explored Natural Language to Struc-
tured Query Language (NL-to-SQL) techniques, we inte-
grate the latest advancements in LLM into our distributed
patent database system. Our approach leverages LLMs’ in-
context learning capabilities and domain-specific adaptabil-
ity to understand and translate natural language instruc-
tions into SQL commands that can precisely operate over
180 million patents in our database.
•Optimized Distributed System Architecture for AI+
Platforms : Our research focuses on creating scalable AI+
platforms using optimized distributed system architecture.
In the traditional Apache distributed architecture, we have
added distributed LLM clusters and UI clusters, and further
designed a Central Control Unit (CCU) to schedule tasks.
We utilize three distributed storage nodes, two LLM nodes,
and two UI nodes to conduct tests with 180 million patents.
The storage consumption is 15.3TB (5.1TB x 3, with dual
1https://www.uspto.gov
2https://patents.google.com

Dawei Feng, Di Mei, Huiri Tan, Lei Ren, Xianying Lou, and Zhangxi Tan
Patent Data W arehouse
WebUI ServerLLM Cluster
Central Control Unit
RenderRequestSyntax Checker
Security CheckerSQL Generator
Computing T ask
ResultConnectSpark Computing Cluster
Figure 1: Architecture of Project StreamLink
redundancy for reliability), and the system is accelerated
using 280 cores (560 threads) and 2.6TB of memory. During
testing, we confirmed that the average time from user input
in natural language to obtaining the desired patent from
a database containing over 180 million patents is within 6
seconds.
•Data Privacy Protecting : We have confirmed that build-
ing a localized LLM-based assistant can significantly en-
hance productivity while providing a higher level of privacy
to users. Through using a locally deployed model for our
LLM-based assistant, we can effectively eliminate the risks
of data breaches and information leakage which could oc-
cur while using cloud-based AI assistants. This approach
maintains the confidentiality and integrity of user data
and underscores our commitment to prioritizing privacy in
the development of advanced technological solutions. We
have also developed mechanisms for SQL legality checking
using Llama[ 13] based tools, protecting the system from
accidental deletion or injection attacks.
We organized this paper as follows. We will discuss the necessity
of this work and its application scenarios in Section 2, then intro-
duce the architecture of StreamLink in Section 3, the methodologies
we used, and the reason we chose these technologies. We present
some experiments in Section 4, including comparisons between
our method and traditional strategies with statistical metrics, and
conclude the paper with a short discussion in Section 5.
2 Task Description
In this section, we will discuss the reason we created the StreamLink
project (as shown in Figure 1) and provide some cases where it has
been used successfully.
Our data system is primarily used for handling large-scale data
storage and retrieval. A typical scenario involves retrieving sev-
eral patents from a database such as Google Patents that meetspecific criteria (e.g. date range, pattern, keywords, etc) and analyz-
ing the potential for IP infringement. Traditionally, IP researchers
and lawyers might need to read through extensive documentation
and perform dozens to hundreds of repetitive searches on Google
Patents, or write complex SQL-like statements on data engineering
platforms like BigQuery to retrieve patents. The former requires
significant manpower and time, often necessitating several lawyers
to collaborate over several days to filter the data, while the lat-
ter requires extensive technical expertise to write complex SQL
commands or other data manipulation languages, and familiarity
with the intricacies of data storage and computation frameworks.
In addition, SQL commands could have bugs and often require
considerable time to be adjusted and modified.
With the StreamLink platform, users can complete all the above
tasks in a more efficient and accessible fashion. Without the need to
design a SQL command, users like IP researchers and lawyers can
directly query the patent database via a natural language request.
With the LLM-based interface, our data system converts this natural
language query into a SQL command with a security check, and
then the distributed database finishes executing this SQL command
in a few seconds. Retrieved patents are expected to meet all filter
conditions in the natural language query. Furthermore, there is great
flexibility for creating different AI interfaces upon StreamLink’s
distributed database. In this case, we have implemented a BERT-
based[ 14] semantic filter upon theses retrieved patents to further
extract the patents with the potential for IP infringement.
Another challenge is the scalability of large-scale database. Tra-
ditional data warehouses are struggling to handle the exponential
growth of data volumes, which can lead to capacity issues[ 15], and
they can also fail to seamlessly scale in response to fluctuating
data processing demands[ 16]. To handle the issue of patent storage
capacity, we employed distributed data warehouses[ 17], designed
to efficiently store and manage a vast amount of information across
multiple servers. This ensures high fault tolerance of databases
as well as facilitates elastic scaling of storage resources to meet

StreamLink: Large-Language-Model Driven Distributed Data Engineering System
growing patent data demands. Currently, we use three of the 5.1TB
nodes to store 180 million entries from the USPTO and Google
Patents.
3 Methodology
In this section, we will present the components in StreamLink.
Section 3.1 will introduce the LLM-driven SQL Generator, an inno-
vative tool capable of understanding natural language instructions
and translating them into SQL commands based on the database
schema. Section 3.2 will showcase our distributed framework based
on Apache Spark and Apache Hadoop, which offer robust sup-
port for processing large-scale datasets, ensuring high scalability
and processing capacity. Moreover, we will discuss our distributed
WebUI clusters and load balancing in this section. We will also
talk about our brand new Llama-based SQL syntax and security
checker built upon StreamLink to reduce the risks associated with
grammatical errors or malicious SQL injections in Section 3.3.
3.1 LLM-based SQL Generator
Our IP lawyer collaborators work with various patents from the
globe every day. They may want to execute a command similar to
SELECT cpc, COUNT(*) AS count FROM google_full WHERE
assignee LIKE "%Intel%" AND grant_date >= "2009" GROUP
BY cpc ORDER BY count DESC LIMIT 10 to conduct an analysis
on the most popular CPC numbers of patents from Intel, but writing
such a SQL command is too difficult for them without professional
programming training.
To solve this problem, our LLM-driven SQL Generator is an inno-
vative tool that makes data engineering more accessible to a wider
audience. It has the ability to comprehend natural language instruc-
tions and convert them into SQL commands, thereby reducing the
learning curve for users. Even those who lack specialized program-
ming training can effortlessly carry out complex data engineering
tasks.
While traditional natural language to SQL generators are based
on Encoder and Decoder structures[ 18], requiring extensive data
training to obtain the ability to generate SQL commands before
specializing in a specific database, we utilize an LLM-based SQL gen-
erator and propose two methods for SQL generation. One method
involves quickly generating specialized SQL commands for cor-
responding databases based on specific rules, followed by fine-
tuning. The other method involves parsing database structures to
quickly generate prompt templates, aiding LLM in migrating to
new databases. Both methods are faster and more scalable than tra-
ditional approaches, making them become more appropriate data
engineering assistants.
We use LoRA[ 19] as an improved fine-tuning method where
instead of fine-tuning all the weights that constitute the weight ma-
trix of the pre-trained LLM, two smaller matrices that approximate
this larger matrix’s weight update are fine-tuned. These matrices
constitute the LoRA adapter. This fine-tuned adapter is then loaded
to the pre-trained model and used for inference.
For the NL-to-SQL conversion, we construct context-target pairs:
𝑍={(𝑥𝑖,𝑦𝑖)}𝑁
𝑖=1, where𝑥𝑖is a natural language query and 𝑦𝑖its
correspoding SQL command. During fine-tuning, the model is ini-
tialized to pre-trained weights Φ0, and the task-specific parameterincrement ΔΦ=ΔΦ(Θ)is further encoded by a much smaller-sized
set of parameters Θwith|Θ|≪|Φ0|. To optimize the SQL genera-
tion quality is to minimize the cross-entropy loss at the decoding
stage. The task of finding ΔΦthus becomes optimizing over Θ:
max
Θ∑︁
(𝑥,𝑦)∈𝑍|𝑦|∑︁
𝑡=1log(𝑝Φ0+ΔΦ(Θ)(𝑦𝑡|𝑥,𝑦<𝑡)) (1)
Instead of full fine-tuning:
max
Φ∑︁
(𝑥,𝑦)∈𝑍|𝑦|∑︁
𝑡=1log(𝑃Φ(𝑦𝑡|𝑥,𝑦<𝑡)) (2)
Another critical challenge for fine-tuning is to adapt an LLM
to NL-2-SQL tasks within a domain-specific schema. Different do-
mains have different rules of defining schemas in their data storage,
and thus we proposed a mechanism to augment the domain-specific
NL-2-SQL training set given a small set of query templates. This
mechanism augments the training set by simultaneously propa-
gating SQL commands and their corresponding natural language
queries (see Figure 2). Every SQL template query can be turned
into a set of SQL commands by inserting different field instances
into it; and for each SQL template query, we designed natural lan-
guage queries in different written expressions. Each SQL template
is propagated in two directions (natural language queries and SQL
commands with various field instances) and then natural language
queries are matched with their corresponding SQL commands to
form the augmented training set. To prevent the LLM from suffer-
ing from catastrophic forgetting and over-fitting, we combined the
domain-specific dataset with publicly available NL-2-SQL datasets
like WikiSQL[ 20] and Spider[ 21]. Through extensive experiments,
1:1 is found to be the optimal hybrid ratio of domain-specific train-
ing set to the open domains.
Figure 2: Data augmentation via bi-directional propagation
3.2 Distributed Computing, Storage and WebUI
Cluster
In this section, we will discuss our distributed framework. This
framework is the foundation of our data engineering system, and it
is designed to manage and process large-scale datasets efficiently,
making our system scalable and robust. Using these distributed com-
puting paradigms, we can distribute data processing tasks among

Dawei Feng, Di Mei, Huiri Tan, Lei Ren, Xianying Lou, and Zhangxi Tan
multiple nodes, reducing the time required for data processing and
analysis.
As shown in Figure 3, by adopting this approach, we can effi-
ciently, reliably, and scalably handle large-scale datasets. Not only
does this method overcome the limitations of traditional data pro-
cessing methods, but it also unlocks new possibilities for advanced
data analytics and engineering tasks. Therefore, it is an essential
component of our data engineering ecosystem.
Driver (Spark Context)
RDDS and DAG Scheduler
Cluster Manager (Stand alone-Y ARN) Master
Cache Cache
Task TaskWorker node  
with 'n' blocksWorker node  
with 'n' blocks
Hadoop HDFSWorkers (Slave)
Executor ExecutorAPI
LLM ClusterAgile Processing Architecture
Figure 3: Distrubuted system architecture with LLM to im-
prove agility
For user experience, we have developed a distributed Web User
Interface (WebUI) cluster and implemented a load balancing mech-
anism that makes sure high availability and responsiveness of the
user interface. To guarantee the effectiveness of our WebUI cluster,
we have implemented a robust load balancing mechanism using
Nginx[ 22], a high-performance HTTP server and reverse proxy.
Nginx acts as an intermediary between the client and the WebUI
instances, intelligently distributing incoming requests across the
available nodes based on predefined algorithms. This evenly dis-
tributes incoming traffic across the WebUI instances, preventing
any single node from becoming overwhelmed with requests, thus
avoiding performance degradation and downtime. Additionally, in
case of node failure or maintenance, Nginx dynamically reroutes
the requests to healthy nodes, ensuring uninterrupted service for
users.
3.3 Llama-driven Checker
SQL statements can bring many risks, including execution failure
or irreversible impacts on the system. To address this problem, we
have designed a new Llama driven syntax and security checker for
StreamLink. These tools represent a significant advancement in
enhancing the accuracy and security of SQL commands within our
data engineering system.
The SQL syntax checker analyzes the structure and syntax of
SQL commands generated by our system, ensuring that they ad-
here to the correct grammar and formatting rules. By validatingthe syntax of SQL commands, this tool significantly reduces the
likelihood of errors that could arise from incorrect or malformed
commands.Then the security checker plays a crucial role in mit-
igating potential risks associated with SQL injection attacks. By
scrutinizing SQL commands for suspicious patterns or constructs
that may indicate malicious intent, the security checker helps safe-
guard our system against unauthorized access, data breaches, and
other security vulnerabilities.
Together, the SQL syntax checker and security checker strengthen
the reliability and integrity of our data engineering system by min-
imizing the risk of errors and malicious activities. This proactive
approach to SQL command validation not only enhances the overall
quality of data processing but also instills confidence in the security
posture of our system. It ensures the safe handling of sensitive
information and protects against potential threats.
4 Experiments
In this section, we present the results of experiments conducted
using StreamLink for data engineering, compared to traditional
data systems. These experiments involve SQL generation reliability
and malicious SQL interception evaluation.
4.1 Generation Accuracy
In our first experiment, we compared our proposed method to sev-
eral existing approaches using the Spider[ 21] dataset, which con-
sists of 10,181 questions and 5,693 unique complex SQL commands
on 200 databases with multiple tables covering 138 different do-
mains. Our goal was to evaluate the effectiveness of SQL generation,
and we leveraged state-of-the-art LLMs and fine-tuning techniques
to do so. The results showed that our method consistently outper-
formed the baseline methods in terms of SQL generation quality
and accuracy.
We conduct experiments on Spider and compare our method
with several baselines including:
•Natural SQL[ 25], a SQL intermediate representation (IR),
enables existing models that do not support executable SQL
generation to generate executable SQL queries.
•GRAPPA[ 26], a grammar-augmented pre-training frame-
work for table semantic parsing.
•𝑆2SQL[ 27], injecting Syntax to question-Schema graph en-
coder for Text-to-SQL parsers, which effectively leverages
the syntactic dependency information of questions in text-
to-SQL to improve the performance.
•PICARD[ 28], a method for constraining auto-regressive
decoders of language models through incremental parsing.
•RASAT[ 29], a Transformer-based seq2seq architecture aug-
mented with relation-aware self-attention that could lever-
age a variety of relational structures.
•StruG[ 30], structure-grounded pre-training framework (STRUG)
for text-to-SQL that can effectively learn to capture text-
table alignment based on a parallel text-table corpus.
•BERT[ 31], pre-training of deep bidirectional transformers
for language understanding.
We demonstrate the exact match and execution accuracy be-
tween the baseline methods and our LLM-driven methods in Table
1.

StreamLink: Large-Language-Model Driven Distributed Data Engineering System
Approach Exact Match Accuracy
GRAPPA + RAT-SQL 73.4 -
StruG + RAT-SQL 72.6 74.9
BERT_LARGE + RAT-SQL 69.8 72.3
S2SQL + ELECTRA 76.4 -
PICARD 75.5 79.3
PICARD + RASAT 75.3 80.5
RASAT 72.6 76.6
T5-3B 71.5 74.4
T5-3B + PICARD 75.5 79.3
SSQLG2-7B 80.1 81.5
SSQLG2-13B 81.9 82.7
SSQLG3-8B 86.7 88.5
SSQLG3.1-8B 86.9 89.7
Table 1: Comparison of various models performance on spi-
der dev-set for text-to-SQL, including Exact Match (EM) and
Execution Accuracy (EA), the performance of StreamLink-
SQL-Generator(SSQLG) is higher than that of Baseline. The
best one is SSQLG3.1-8B which we fine-tuned on Llama-3.1-
8B.
Instead of directly deploying off-the-shelf commercial or open-
source LLMs, we hope to use domain knowledge to gain a StreamLink-
dedicated model. The data in the table shows that our fine-tuned
model has exceeded the baseline by over 10% in both execution
accuracy and exact match, achieving the effect of transferring a
general language model to a specialized task. This provides the
opportunity of using natural language interaction for StreamLink’s
users with different backgrounds. For instance, we can enable our
lawyer collaborators to use natural language to perform specific
patent analysis by saying “tell me the top 10 most frequently ap-
peared CPC by the assignee of Intel after 2009” instead of manually
writing a complex SQL command mentioned before.
These results highlight the effectiveness of our approach in ad-
dressing the challenges of SQL generation tasks, especially in com-
plex and specialized domains with varying database schema. By
outperforming existing methods on the Spider dataset, our method
showcases its potential to significantly improve the efficiency and
accuracy of SQL generation processes. This, in turn, can facilitate
more effective data engineering and analysis workflows.
4.2 Malicious SQL Interception
In this experiment, we focused on evaluating the effectiveness of
our SQL syntax checker and security checker based on Llama2.
We used the SQL injection dataset3from Kaggle, which includes
30,595 SQL statements. Within this dataset, 19,258 were normal
SQL, while 11,337 were malicious statements. We evaluated our
LLM-based syntax and security checkers across different model
sizes and model types. This dataset is representative of different
SQL injections that occur in real-world scenarios, making it a solid
testing ground for our tools.
Our evaluation focused on zero-shot conditions to simulate the
checker’s performance in situations where the specific dataset
3https://www.kaggle.com/datasets/syedsaqlainhussain/sql-injection-datasetmight not be feasible to train. This is common in organizations
that need to adapt quickly to emerging threats without retraining
models. We use recall and precision as metrics.
𝑅𝑒𝑐𝑎𝑙𝑙 =𝑇𝑃
𝑇𝑃+𝐹𝑁(3)
𝑃𝑟𝑒𝑐𝑖𝑠𝑖𝑜𝑛 =𝑇𝑃
𝑇𝑃+𝐹𝑃(4)
𝐸𝑠𝑐𝑎𝑝𝑒 =𝐹𝑁
𝑇𝑃+𝐹𝑁(5)
𝑀𝑖𝑠𝑖𝑛𝑡𝑒𝑟𝑐𝑒𝑝𝑡 =𝐹𝑃
𝑇𝑁+𝐹𝑃(6)
Where
•TP (True Positive) – Positive in the label, and predicted
positive.
•FP (False Positive) – Negative in the label, but predicted
positive.
•FN (False Negative) – Positive in the label, but predicted
negative.
•FN (False Negative) – Negative in the label, and predicted
negative.
After conducting multiple groups of random tests, we evaluate
the effect of the model in the following table:
Approach Precision Recall Escape Misintercept
SSQLC2: 7B 76.54% 89.39% 10.61% 16.26%
SSQLC2: 70B 74.2% 97.05% 2.95% 19.87%
SSQLC3: 8B 79.31% 98.09% 1.91% 15.07%
SSQLC3: 70B 80.01% 98.42% 1.58% 14.47%
SSQLC3.1: 8B 71.7% 91.72% 8.28% 21.23%
SSQLC3.1: 70B 90.52% 94.38% 5.62% 5.82%
Table 2: Test results of LLM of different sizes on malicious
SQL data sets, we implement four tpyes of SQL checker based
on Llama-2, Llama-3 and Llama-3.1, and show the test result
of StreamLink-SQL-Checker (SSQLC in the table.)
The data in Table 2 reflects the challenges posed by the Llama2
architecture, which, despite being effective, shows limitations in
handling SQL interception compared to the more advanced Llama3
and Llama3.1 models. Specifically, the SSQLC2 series, based on
Llama2, exhibits lower performance across most metrics. For in-
stance, SSQLC2-70B achieves a recall of 97.05%, which is impres-
sive but still falls short of the results obtained with Llama3 and
Llama3.1-based models. The precision of the SSQLC2 series also
lags behind, highlighting that the older architecture and potentially
outdated knowledge embedded in Llama2 lead to a higher rate of
false positives, indicating a less reliable performance in real-world
SQL injection detection.
The results for the Llama3.1-based models suggest that the train-
ing data and knowledge incorporated into this version may not have
been as well-optimized for SQL interception as those in Llama3. The
SSQLC3.1-8B model, for example, shows a noticeable drop in preci-
sion (71.7%) compared to SSQLC3-8B (79.31%), alongside a higher
misintercept rate (21.23% vs. 15.07%). Although the SSQLC3.1-70B
model does recover some ground, achieving a precision of 90.52%,

Dawei Feng, Di Mei, Huiri Tan, Lei Ren, Xianying Lou, and Zhangxi Tan
its performance inconsistencies relative to Llama3 indicate that
Llama3.1 may not yet offer the same level of robustness for SQL
attack detection.
Considering the balance between speed, accuracy, escape rate,
and misintercept rate, the SSQLC3-8B model emerges as the most
suitable choice for the StreamLink SQL Checker. It offers a strong
recall rate of 98.09% with a manageable precision of 79.31%, all while
maintaining a reasonable processing speed of 4 SQL statements
per second. This model provides a well-rounded performance that
meets the demands of real-time SQL injection detection while avoid-
ing the significant speed drawbacks of the larger 70B models. The
SSQLC-3-8B’s combination of efficiency and effectiveness makes it
the optimal solution for deployment in environments where both
accuracy and speed are crucial.
0 5000 10000 15000 20000 25000 30000
Number of T ests0.700.750.800.850.90Precision (Higher the Better)
SSQLC2-7B
SSQLC2-70B
SSQLC3-8B
SSQLC3-70B
SSQLC3.1-8B
SSQLC3.1-70B
0 5000 10000 15000 20000 25000 30000
Number of T ests0.860.880.900.920.940.960.981.00Recall (Higher the Better)
SSQLC2-7B
SSQLC2-70B
SSQLC3-8B
SSQLC3-70B
SSQLC3.1-8B
SSQLC3.1-70B
0 5000 10000 15000 20000 25000 30000
Number of T ests0.000.020.040.060.080.100.120.14Escape (Lower the Better)
SSQLC2-7B
SSQLC2-70B
SSQLC3-8B
SSQLC3-70B
SSQLC3.1-8B
SSQLC3.1-70B
0 5000 10000 15000 20000 25000 30000
Number of T ests0.0500.0750.1000.1250.1500.1750.2000.225Mis-intercept (Lower the Better)
SSQLC2-7B
SSQLC2-70B
SSQLC3-8B
SSQLC3-70B
SSQLC3.1-8B
SSQLC3.1-70BPrecision-Recall-Escape-Misintercept Curves
Figure 4: Malicious SQL interception analyzing on our LLM-
based method
Figure 4 shows the test results obtained on sample sets of dif-
ferent sizes. When the sample size is less than 5000, the model’s
performance exhibits some fluctuations, which may be due to the
uneven distribution of positive and negative samples in small sam-
ples. However, as the sample size increases from 5000 to 30000, the
distribution of positive and negative labels gradually approaches
normal distribution, and the model demonstrates excellent stability.
The results of the experiment were highly encouraging, Figure 5
indicates that our interceptors provide robust protection against
malicious SQL commands. By effectively identifying and blocking
malicious actions, our system ensures the stable operation of the
server, safeguarding against potential disruptions and data breaches.
This demonstrates the critical role of our SQL syntax checker and se-
curity checker in fortifying the system’s defenses against malicious
attacks and ensuring the reliability and security of data processing
operations.
0.0 0.2 0.4 0.6 0.8 1.0
False Positive Rate0.00.20.40.60.81.0True Positive RateROC Curve for Different Models
SSQLC2-7B (AUC = 0.87)
SSQLC2-70B (AUC = 0.89)
SSQLC3-8B (AUC = 0.92)
SSQLC3-70B (AUC = 0.92)
SSQLC3.1-8B (AUC = 0.85)
SSQLC3.1-70B (AUC = 0.94)Figure 5: ROC Curve of our SSQLC methods, and the AUC of
each method
5 Conclusion
In conclusion, we present a comprehensive exploration of Stream-
Link, an innovative data engineering system empowered by cutting-
edge technologies such as LLMs, distributed computing frameworks,
and advanced security mechanisms. Through a series of experi-
ments and evaluations, we have demonstrated the effectiveness,
efficiency, and security of StreamLink in various data engineering
tasks.
StreamLink’s scalable architecture, where all services interact
via an Application Programming Interface (API) and Remote Proce-
dure Call (RPC) structure, provides rich extensibility for the system.
Our evaluations have shown that StreamLink significantly outper-
forms existing solutions in specific domains, making it a robust
and transformative tool for data engineering. The integration of ad-
vanced algorithms and a focus on scalability and security positions
StreamLink as a solution for organizations looking to enhance their
data workflows while maintaining high standards of privacy and
security.
Overall, we think StreamLink represents a significant advance-
ment in data engineering technology, offering unparalleled capabil-
ities in terms of efficiency, scalability, and security. Its innovative
use of LLMs, combined with distributed computing frameworks and
advanced security mechanisms, positions StreamLink as a trans-
formative solution for organizations seeking to enhance their data
engineering workflows.

StreamLink: Large-Language-Model Driven Distributed Data Engineering System
References
[1]Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan,
Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda
Askell, et al. Language models are few-shot learners. Advances in neural infor-
mation processing systems , 33:1877–1901, 2020.
[2] Matei Zaharia, Mosharaf Chowdhury, Michael J Franklin, Scott Shenker, and Ion
Stoica. Spark: Cluster computing with working sets. In 2nd USENIX Workshop
on Hot Topics in Cloud Computing (HotCloud 10) , 2010.
[3] CL Philip Chen and Chun-Yang Zhang. Data-intensive applications, challenges,
techniques and technologies: A survey on big data. Information sciences , 275:314–
347, 2014.
[4] Steve Kelling, Wesley M Hochachka, Daniel Fink, Mirek Riedewald, Rich Caruana,
Grant Ballard, and Giles Hooker. Data-intensive science: a new paradigm for
biodiversity studies. BioScience , 59(7):613–620, 2009.
[5] Arthur Stone, Saul Shiffman, Audie Atienza, and Linda Nebeling. The science of
real-time data capture: Self-reports in health research . Oxford University Press,
2007.
[6] Aisha Siddiqa, Ahmad Karim, and Abdullah Gani. Big data storage technologies:
a survey. Frontiers of Information Technology & Electronic Engineering , 18:1040–
1070, 2017.
[7] Syed Mohd Ali, Noopur Gupta, Gopal Krishna Nayak, and Rakesh Kumar Lenka.
Big data visualization: Tools and challenges. In 2016 2nd International conference
on contemporary computing and informatics (IC3I) , pages 656–660. IEEE, 2016.
[8] Alan C Marco, Amanda Myers, Stuart JH Graham, Paul D’Agostino, and Kirsten
Apple. The uspto patent assignment dataset: Descriptions and analysis. 2015.
[9]D Curtis Jamison. Structured query language (sql) fundamentals. Current
protocols in bioinformatics , (1):9–2, 2003.
[10] Jinyang Li, Binyuan Hui, Reynold Cheng, Bowen Qin, Chenhao Ma, Nan Huo,
Fei Huang, Wenyu Du, Luo Si, and Yongbin Li. Graphix-t5: Mixing pre-trained
transformers with graph-aware layers for text-to-sql parsing. In Proceedings of
the AAAI Conference on Artificial Intelligence , volume 37, pages 13076–13084,
2023.
[11] Lu Zeng, Sree Hari Krishnan Parthasarathi, and Dilek Hakkani-Tur. N-best
hypotheses reranking for text-to-sql systems. In 2022 IEEE Spoken Language
Technology Workshop (SLT) , pages 663–670. IEEE, 2023.
[12] Yiyun Zhao, Jiarong Jiang, Yiqun Hu, Wuwei Lan, Henry Zhu, Anuj Chauhan,
Alexander Li, Lin Pan, Jun Wang, Chung-Wei Hang, et al. Importance of synthe-
sizing high-quality data for text-to-sql parsing. arXiv preprint arXiv:2212.08785 ,
2022.
[13] Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yas-
mine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhos-
ale, et al. Llama 2: Open foundation and fine-tuned chat models. arXiv preprint
arXiv:2307.09288 , 2023.
[14] Nils Reimers and Iryna Gurevych. Sentence-bert: Sentence embeddings using
siamese bert-networks. arXiv preprint arXiv:1908.10084 , 2019.
[15] José Ignacio Huertas, Jenny Díaz Ramírez, and Federico Trigos Salazar. Layout
evaluation of large capacity warehouses. Facilities , 25(7/8):259–270, 2007.
[16] Yasin N Silva, Isadora Almeida, and Michell Queiroz. Sql: From traditional
databases to big data. In Proceedings of the 47th ACM Technical Symposium on
Computing Science Education , pages 413–418, 2016.
[17] Apache Software Foundation. Hadoop.
[18] Ursin Brunner and Kurt Stockinger. Valuenet: A natural language-to-sql system
that learns from database information. In 2021 IEEE 37th International Conference
on Data Engineering (ICDE) , pages 2177–2182. IEEE, 2021.
[19] Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean
Wang, Lu Wang, and Weizhu Chen. Lora: Low-rank adaptation of large language
models. arXiv preprint arXiv:2106.09685 , 2021.
[20] Victor Zhong, Caiming Xiong, and Richard Socher. Seq2sql: Generating struc-
tured queries from natural language using reinforcement learning. CoRR ,
abs/1709.00103, 2017.
[21] Tao Yu, Rui Zhang, Kai Yang, Michihiro Yasunaga, Dongxu Wang, Zifan Li, James
Ma, Irene Li, Qingning Yao, Shanelle Roman, et al. Spider: A large-scale human-
labeled dataset for complex and cross-domain semantic parsing and text-to-sql
task. arXiv preprint arXiv:1809.08887 , 2018.
[22] Will Reese. Nginx: the high-performance web server and reverse proxy. Linux
Journal , 2008(173):2, 2008.
[23] Rasmus V Rasmussen and Michael A Trick. Round robin scheduling–a survey.
European Journal of Operational Research , 188(3):617–636, 2008.
[24] Tong Li, Dan Baumberger, and Scott Hahn. Efficient and scalable multiprocessor
fair scheduling using distributed weighted round-robin. ACM Sigplan Notices ,
44(4):65–74, 2009.
[25] Yujian Gan, Xinyun Chen, Jinxia Xie, Matthew Purver, John R Woodward, John
Drake, and Qiaofu Zhang. Natural sql: Making sql easier to infer from natural
language specifications. arXiv preprint arXiv:2109.05153 , 2021.
[26] Tao Yu, Chien-Sheng Wu, Xi Victoria Lin, Bailin Wang, Yi Chern Tan,
Xinyi Yang, Dragomir Radev, Richard Socher, and Caiming Xiong. Grappa:
Grammar-augmented pre-training for table semantic parsing. arXiv preprintarXiv:2009.13845 , 2020.
[27] Binyuan Hui, Ruiying Geng, Lihan Wang, Bowen Qin, Bowen Li, Jian Sun, and
Yongbin Li. 𝑠2sql: Injecting syntax to question-schema interaction graph encoder
for text-to-sql parsers. arXiv preprint arXiv:2203.06958 , 2022.
[28] Torsten Scholak, Nathan Schucher, and Dzmitry Bahdanau. Picard: Parsing
incrementally for constrained auto-regressive decoding from language models.
arXiv preprint arXiv:2109.05093 , 2021.
[29] Jiexing Qi, Jingyao Tang, Ziwei He, Xiangpeng Wan, Yu Cheng, Chenghu Zhou,
Xinbing Wang, Quanshi Zhang, and Zhouhan Lin. Rasat: Integrating rela-
tional structures into pretrained seq2seq model for text-to-sql. arXiv preprint
arXiv:2205.06983 , 2022.
[30] Xiang Deng, Ahmed Hassan Awadallah, Christopher Meek, Oleksandr Polozov,
Huan Sun, and Matthew Richardson. Structure-grounded pretraining for text-to-
sql.arXiv preprint arXiv:2010.12773 , 2020.
[31] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. Bert: Pre-
training of deep bidirectional transformers for language understanding. arXiv
preprint arXiv:1810.04805 , 2018.