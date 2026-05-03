# Budget-Constrained Online Retrieval-Augmented Generation: The Chunk-as-a-Service Model

**Authors**: Shawqi Al-Maliki, Ammar Gharaibeh, Mohamed Rahouti, Mohammad Ruhul Amin, Mohamed Abdallah, Junaid Qadir, Ala Al-Fuqaha

**Published**: 2026-04-28 14:42:51

**PDF URL**: [https://arxiv.org/pdf/2604.26981v1](https://arxiv.org/pdf/2604.26981v1)

## Abstract
Large Language Models (LLMs) have revolutionized the field of natural language processing. However, they exhibit some limitations, including a lack of reliability and transparency: they may hallucinate and fail to provide sources that support the generated output. Retrieval-Augmented Generation (RAG) was introduced to address such limitations in LLMs. One popular implementation, RAG-as-a-Service (RaaS), has shortcomings that hinder its adoption and accessibility. For instance, RaaS pricing is based on the number of submitted prompts, without considering whether the prompts are enriched by relevant chunks, i.e., text segments retrieved from a vector database, or the quality of the utilized chunks (i.e., their degree of relevance). This results in an opaque and less cost-effective payment model. We propose Chunk-as-a-Service (CaaS) as a transparent and cost-effective alternative. CaaS includes two variants: Open-Budget CaaS (OB-CaaS) and Limited-Budget CaaS (LB-CaaS), which is enabled by our ``Utility-Cost Online Selection Algorithm (UCOSA)''. UCOSA further extends the cost-effectiveness and the accessibility of the OB-CaaS variant by enriching, in an online manner, a subset of the submitted prompts based on budget constraints and utility-cost tradeoff. Our experiments demonstrate the efficacy of the proposed UCOSA compared to both offline and relevance-greedy selection baselines. In terms of the performance metric-the number of enriched prompts (NEP) multiplied by the Average Relevance (AR)-UCOSA outperforms random selection by approximately 52% and achieves around 75% of the performance of offline selection methods. Additionally, in terms of budget utilization, LB-CaaS and OB-CaaS achieve higher performance-to-budget ratios of 140% and 86%, respectively, compared to RaaS, indicating their superior efficiency.

## Full Text


<!-- PDF content starts -->

1
Budget-Constrained Online Retrieval-Augmented
Generation: The Chunk-as-a-Service Model
Shawqi Al-Maliki, Ammar Gharaibeh, Mohamed Rahouti,Member, IEEE, Mohammad Ruhul Amin, Mohamed
Abdallah,Senior Member, IEEE, Junaid Qadir,Senior Member, IEEE, Ala Al-Fuqaha∗,Senior Member, IEEE
Abstract—Large Language Models (LLMs) have revolutionized
the field of natural language processing. However, they exhibit
some limitations, including a lack of reliability and transparency:
they may hallucinate and fail to provide sources that support the
generated output. Retrieval-Augmented Generation (RAG) was
introduced to address such limitations in LLMs. One popular
implementation, RAG-as-a-Service (RaaS), has shortcomings that
hinder its adoption and accessibility. For instance, RaaS pricing is
based on the number of submitted prompts, without considering
whether the prompts are enriched by relevant chunks, i.e., text
segments retrieved from a vector database, or the quality of the
utilized chunks (i.e., their degree of relevance). This results in
an opaque and less cost-effective payment model. We propose
Chunk-as-a-Service (CaaS) as a transparent and cost-effective
alternative. CaaS includes two variants: Open-Budget CaaS (OB-
CaaS) and Limited-Budget CaaS (LB-CaaS), which is enabled by
our “Utility-Cost Online Selection Algorithm (UCOSA)”. UCOSA
further extends the cost-effectiveness and the accessibility of the
OB-CaaS variant by enriching, in an online manner, a subset of
the submitted prompts based on budget constraints and utility-
cost tradeoff. Our experiments demonstrate the efficacy of the
proposed UCOSA compared to both offline and relevance-greedy
selection baselines. In terms of the performance metric—the
number of enriched prompts (NEP) multiplied by the Average
Relevance (AR)—UCOSA outperforms random selection by
approximately 52% and achieves around 75% of the performance
of offline selection methods. Additionally, in terms of budget
utilization, LB-CaaS and OB-CaaS achieve higher performance-
to-budget ratios of 140% and 86%, respectively, compared to
RaaS, indicating their superior efficiency.
Impact Statement—RAG systems help mitigate hallucinations
in LLMs, and RAG-as-a-Service (RaaS) facilitates their broader
utilization. However, the RaaS model lacks transparency in
usage pricing, is less cost-effective, and is inaccessible to users
with limited budgets. These shortcomings have clear negative
social and economic impacts on potential users. Our proposed
Chunk-as-a-Service (CaaS) model is a transparent and cost-
effective variant of RaaS. It calculates usage pricing based on the
S. Al-Maliki, M. Abdallah., and A. Al-Fuqaha are with the Information and
Computing Technology (ICT) Division, College of Science and Engineering,
Hamad Bin Khalifa University, Doha 34110, Qatar (email:{shalmaliki,
moabdallah, and aalfuqaha}@hbku.edu.qa)
A. Gharaibeh with School of Electrical Engineering and Information
Technology, German Jordanian University, Amman, Jordan (email: am-
mar.gharaibeh@gju.edu.jo)
M. Rahouti and M. Ruhul are with Department of Computer and In-
formation Sciences, Fordham University, Bronx, NY , USA (email: mra-
houti@fordham.edu and mamin17@fordham.edu)
J. Qadir is with the Department of Computer Science and Engineering, Col-
lege of Engineering, Qatar University, Doha, Qatar (email: jqadir@qu.edu.qa)
Corresponding author: Ala Al-Fuqaha (Email: aalfuqaha@hbku.edu.qa)
© 2026 IEEE. Personal use of this material is permitted. Permission from
IEEE must be obtained for all other uses, in any current or future media,
including reprinting/republishing this material for advertising or promotional
purposes, creating new collective works, for resale or redistribution to servers
or lists, or reuse of any copyrighted component of this work in other works.actually utilized chunks during prompt enrichment, adhering to
transparency as a key social principle. CaaS also introduces a new
publishing paradigm aimed at creating competitive content and
fair profits for content creators, CaaS providers, and end users.
Furthermore, CaaS offers a Limited-Budget (LB-CaaS) variant,
enabling users with limited budgets to access RAG, effectively
addressing the economic barrier inherent in the traditional RaaS
model.
Index Terms—Large Language Models (LLMs), Retrieval-
Augmented Generation (RAG), Natural Language Processing
(NLP), Limited-Budget RAG, Cloud Computing
I. INTRODUCTION
LArge Language Models (LLMs) [1]–[4] have revolution-
ized Natural Language Processing (NLP) by enabling
human-like text generation and comprehension [5]. These
models have advanced rapidly, showcasing significant im-
provements in understanding and producing natural language.
However, despite their impressive capabilities, LLMs often
generate inaccurate outputs, or ‘hallucinations’ [6]–[8]. This
problem becomes particularly evident when LLMs respond to
prompts referencing information not present in their training
data. Lacking the necessary context, LLMs may refuse to
answer, produce incorrect responses, or hallucinate, resulting
in erroneous or misleading outputs.
Retrieval-Augmented Generation (RAG) systems [9], [10]
address the limitations of LLMs by utilizing external, non-
parameterized indexed data sources, such as knowledge bases.
RAG systems enable seamless access to up-to-date and reliable
information, significantly improving the efficiency and accu-
racy of the generated output. By integrating real-time retrieval
of relevant data, RAG systems effectively mitigate the issue of
LLM hallucinations, ensuring that responses are both current
and contextually accurate. This approach enhances LLMs’
overall reliability and usefulness, making them more robust
and dependable for various applications.
Fig. 1 demonstrates an abstract view of the standard RAG
pipeline, highlighting how a plain prompt can be enriched
to enable generating a grounded output. Chunking, breaking
down the data source into manageable segments, and embed-
ding (converting these chunks into numerical representations
or vectors) are the essential preprocessing steps that facilitate
efficient storage, indexing, and retrieval of the chunks using a
vector database.
As illustrated in Fig. 1, the standard RAG pipeline processes
the end-user’s text input (called prompt) as follows: It begins
by embedding the prompt and searching through the indexed
chunks in the vector database to find relevant chunks relatedarXiv:2604.26981v1  [cs.IR]  28 Apr 2026

2
Plain Prompt
 LLM Grounded
OutputEmbeddingStandard RAG Pipeline
Indexed
Chunks
Data SourcesParsing, 
Chunking
Vector DatabaseLoading
Embedding,
StoringIndexing
Relevance-based
chunk retrievalRetrieval
Enriched
PromptSynthesis
Fig. 1: Standard pipeline of RAG System. It demonstrates an abstract view of the RAG pipeline, highlighting how a plain
prompt can be enriched to generate a grounded output. Chunking, breaking down the data source into manageable segments,
and embedding (converting these chunks into numerical representations or vectors) are the essential preprocessing steps that
facilitate efficient storage, indexing, and retrieval of the chunks using a vector database.
to this prompt. If relevant chunks are found, the prompt is
augmented with these chunks, forming an enriched prompt,
which is then passed to the LLM. For the prompts that do not
find relevant chunks, they are passed directly to LLM, missing
the opportunity for augmentation.
Standard RAG refers to the existing RAG variants with a
vanilla pipeline, regardless of their deployment paradigm. This
encompasses variants the business owner manages, including
on-premise and cloud-hosted deployments. Standard RAG can
also include variants deployed on the cloud and managed by
a third party to be offered as a service, i.e., RAG-as-a-Service
(RaaS), for interested users or organizations. The existing
RAG variants are illustrated as black colored in Fig. 2.
Fig. 2: A high-level illustration highlighting the position of
our proposed RAG variants, including OB-CaaS and LB-CaaS,
among other related RAG business models.
In the standard RaaS, the cost is calculated based on the
number of submitted prompts (RaaS API calls), regardless of
whether the prompts find an opportunity to be enriched and no
matter how the quality and quantity of the chunks used in the
enrichment process. This indicates that standard RaaS is not
cost-effective for limited-budget users who want to pay only
for what they actually use. Even worse, RaaS is restrictive by
design. It does not support enabling a chunk-based payment
model [9], [10], even for the RaaS providers who are willing
to consider such a payment approach.Depending on its implementation, RaaS can be accessed
in various ways. One approach is through a single API that
abstracts the complexity of managing the underlying required
services, including embedding models, vector databases, and
LLMs, along with their associated APIs (as in Vectara [11]).
Another approach is through a RaaS API in addition to the
APIs of the underlying services requested from the end user (as
in LlamaIndex Cloud [12]). Regardless of the RaaS accessibil-
ity approach, RaaS providers consider the operational cost of
managing the RAG pipeline as part of the established pricing.
However, this operational cost does not necessarily consider
the actually used chunks in the enrichment process, rendering
the standard RaaS a vague and less transparent payment model
that is potentially less cost-effective for the end users.
We address this limitation by proposing a variant that solely
considers the actually used chunks in the prompt enrichment
process. Specifically, we propose modifying the pipeline of the
standard RaaS, particularly the indexing and retrieval stages,
as follows. During the data source indexing stage and while
chunking, a price is assigned to all indexed chunks. This
price considers various factors predetermined by the service
provider and data source owner (see Section V-B). At the
retrieval stage, all relevant chunks to a particular prompt are
presented along with their associated price and relevance score.
By incorporating the price into the chunks, end users can
be easily billed primarily based on their actual utilization
of chunks, along with other relevant operational costs, such
as computing and storage. We term this chunk-based RaaS
variant “Chunk-as-a-Service (CaaS)”, highlighting that the
utilized chunks are central to this model. To the best of our
knowledge, we are the first to introduce and coin the term
CaaS.
Now, consider a scenario of a limited-budget user. The
basic CaaS fails to support limited-budget end-users who
cannot afford the surprise cost associated with utilized chunks,
limiting the basic CaaS accessibility. Therefore, we propose
a cost-aware CaaS variant called “Limited-Budget CaaS (LB-
CaaS).” This variant addresses the shortcomings of basic CaaS,
specifically its ineffectiveness in serving users with a limited
budget. In the cost-aware CaaS variant (LB-CaaS), the budget
is predefined and allocated beforehand. Conversely, the basic
CaaS variant does not require a predefined budget. Therefore,

3
we refer to the basic CaaS as Open-Budget CaaS (OB-CaaS).
Fig. 2 illustrates, at a high level, the existing RAG variants,
positioning our proposed OB-CaaS and LB-CaaS among them.
The selection of the chunks within the LB-CaaS variant
is enabled by our proposed Utility-Cost Online Selection
Algorithm (UCOSA). Considering budget constraints and
the chunk’s utility-cost tradeoff, UCOSA optimally selects
a subset of plain prompts on the fly to enrich each prompt
with a selected chunk, enabling cost-efficient generation of
grounded output (detailed in Section IV). UCOSA is a generic
algorithm that can be applied in various scenarios using diverse
representations of utility and cost. In this work, the utility is
represented by the relevance scoreRof the retrieved chunk,
while the cost is represented by the associated pricePof each
retrieved chunk.
The salient contributions of our work are as follows.
•We introduce a novel RAG business model, Chunk-as-
a-Service (CaaS), a chunk-based payment model that
calculates usage costs based only on the chunks actually
utilized in the prompt enrichment process. To the best of
our knowledge, we are the first to introduce and coin the
term CaaS.
•We propose Open-Budget CaaS (OB-CaaS), the basic
variant of CaaS that makes standard RaaS transparent and
cost-effective for end users.
•We propose Limited-Budget CaaS (LB-CaaS), a cost-
aware CaaS that enhances OB-CaaS by making it more
cost-effective and accessible for a broader range of end
users.
•We propose the Utility-Cost Online Selection Algorithm
(UCOSA) to enable the chunk selection process within
LB-CaaS. UCOSA optimally selects chunks used for
enriching prompts in an online manner. We provide a
mathematical proof demonstrating the optimality of this
algorithm.
•To validate UCOSA, we conduct extensive experiments
demonstrating its effectiveness across various experimen-
tal scenarios and in comparison to multiple baselines.
The rest of the paper is organized as follows. Section
II situates our proposed RAG variants among the existing
ones. In Section III, we detail our proposed RAG variants,
including OB-CaaS and LB-CaaS, and explain UCOSA, the
enabler of LB-CaaS. We provide a mathematical formulation
of our considered problem and solution in Section IV. We
present the management of the proposed RAG variants in
Section V. Section VI illustrates the experimental results that
demonstrate the efficacy of our proposed work. Lastly, Section
VII concludes our proposed work and highlights the potential
future work.
II. MOTIVATION ANDRELATEDWORK
This section highlights the motivations behind propos-
ing CaaS variations, including OB-CaaS and LB-CaaS, and
demonstrates a motivating example. Additionally, it presents
the relevance of our proposed work to the existing literature.A. Motivations Behind Proposing OB-CaaS and LB-CaaS
Compared to the standard RaaS, our proposed CaaS vari-
ants, including OB-CaaS and LB-CaaS, introduce several
advantages that cannot be realized otherwise:
Empowering data source owners with a novel publishing
paradigm:Traditionally, data source owners, such as book
authors, have several options for gaining profit from their
written works, including selling printed books, e-books, audio-
books, or all formats simultaneously. However, with advanced
content-providing paradigms like RaaS, a publishing option
that aligns with such paradigms still needs to be developed.
A RaaS provider cannot offer data source owners a chunk-
based publishing paradigm that enables fair revenue-sharing,
taking into account the value of the data source represented
by chunks and their usage rate, due to the lack of enabling
technology. Therefore, we introduced CaaS as an enabler of
such a publishing paradigm.
Granular profitability and quality content:The proposed
CaaS model allows proprietary data owners to sell their
proprietary data with greater granularity (at the chunk level)
compared to the traditional selling paradigms, where content
is sold at a book or dataset level.
CaaS enables this capability by adapting the typical pipeline
to include pricing agreement during the indexing stage (details
in 3). Data source owners, as well as CaaS providers, can
earn profits based on the chunk’s rate of usage and the
predetermined prices of the chunks associated with their data
source. This fair approach would encourage quality content
creation due to the potentially established competition among
content creators (e.g., authors) to maximize the use of their
content by CaaS.
Potential Applications:Based on the above-mentioned
motivations, there would be various potential applications,
including the following:
•Cloud-based RAG services: Cloud service providers can
adopt and offer the CaaS model as a novel solution
that provides flexible and cost-effective RAG service,
charging users based on the actual data chunks they use
rather than a flat fee.
•Online marketplaces for data: Platforms can be devel-
oped where data owners sell their content to the CaaS
providers, ensuring they profit from every use of their
sold content (at chunk level).
B. Motivating Example
Fig. 3 demonstrates a toy example that compares various
RAG variants, including Standard RaaS and our proposed
OB-CaaS and LB-CaaS variants. The comparison considers
handling a stream of prompts based on price calculation
transparency, usage price, and quality of the generated output,
represented by the average relevance scores of the chunks
utilized for the enrichment process.
For clarity, we gray-colored the set of prompts expected
not to be enriched by chunks—due to their irrelevance to the
indexed chunks. For that, the enrichment process demonstrates
a subset of the prompts (five out of ten). Additionally, for

4
How do adversarial 
ML attacks affect
model's accuracy
and defenses?
Proposed OB-CaaS (Retrieval and Enrichment )
Proposed LB-CaaS (Retrieval and Enrichment )Standard RaaS (Retrieval and Enrichment Process )Quantity (Enriched
Prompts)Priced Prompts Total Usage Price
5 
(out of  10)10
(out of 10)
Fixed & prompt-dependent price, with opaque calculationQuality (Average 
Relevance)
 8
Currency Unit0.84
Variable & chunk-dependent price, with transparent calculation
PP-09
R: 0.8
PP-07
R: 0.9
PP-05
R: 0.8
PP-03
R: 0.9PP-10 PP-09 PP-08 PP-07 PP-06Stream of Plain Prompts (PPs)
PP-05 PP-04 PP-03 PP-02
Budgeted, variable &  chunk-dependent price, with
transparent calculationPP-01
R: 0.8
PP-09
 PP-07
 PP-05
 PP-03
R:0.8
P:0.9R:0.9
P:0.8R:0.8
P:0.7R:0.9
P:0.9
PP-09
 PP-05
R:0.7
P:0.5R:0.7
P:0.4
PP-07
R:0.8
P:0.6
PP-03
R:0.8
P:0.30.84
0.74 (The actually enriched prompts
can be identified by the user) (The actually enriched prompts
cannot be realized by user)Opague: Pricing all
submitted  prompts.10 prompts  x 0.8
average relevance
5 
(out of  10)5 
(out of  10)
 (The actually enriched prompts
can be identified by the user)5 
(out of  10)5 
(out of  10)
Transparent:  Pricing only only 
the actually enriched prompts.2
Currency Unit
5 prompts  x 0.4
average relevance4
Currency Unit
5 prompts  x 0.8
average relevance
PP-01
Attacks reduce accuracy;
adversarial training
improves defense...
PP-01
Attacks reduce accuracy;
adversarial training
improves defense...R:0.8 | P:0.7
PP-01
Attacks drop accuracy;
simple defenses include
preprocessing...R:0.7 | P:0.2Transparent:  Pricing only only 
the actually enriched prompts.1
2
Agreed  Data
Sources (DS) Chunking and
Assigning Price (P)P: 0.02 P: 0.02
Pricing
AgreementDS-1
Indexed ChunksP: 0.8 P: 0.8 DS-NP: 0.2 P: 0.2
P: 0.8 P: 0.8
Indexing
Choosed Data
Sources (DS) Chunking DS-1
Indexed ChunksDS-N
Indexing
Proposed V ariants ( Indexing)Standard  RaaS  (Indexing)Indexing
0
Fig. 3: A toy example illustrating how various RAG variants, including Standard RaaS and our proposed OB-CaaS and LB-
CaaS variants, handle a stream of prompts in terms of price calculation transparency, usage price, and quality of the generated
output. This demonstration highlights that our proposed RAG variants enable a transparent price calculation. Additionally, it
highlights that the total usage price is reduced by 50% with OB-CaaS and 75% with LB-CaaS. This price reduction occurs
without compromising the quality of the generated outputs in OB-CaaS, and with only a 12% decrease in quality when using
LB-CaaS.

5
simplicity, we only textually illustrate the first prompt and the
associated selected chunks.
Additionally, Fig. 3 presents the indexing process (step
0), which is a prior step to prompting, aiming at efficiently
storing data sources. The indexing compares the standard
indexing with the proposed one, highlighting the differences,
including the pricing agreement and assigning price. The
pricing agreement aims at negotiating pricing between the
data source owner and CaaS provider to determine the price
per chunk (as detailed in Section V-B). Agreed data sources
are chunked, assigned the agreed price per chunk per data
source, and indexed. Differences in the compared indexing
are highlighted in blue color.
Standard RaaS is opaque in terms of price calculation be-
cause pricing is prompt-dependent and not chunk-dependent.
The actually enriched prompts cannot be realized by the end
user. Users are charged based on the submitted prompts,
regardless of whether they are enriched. Although only five
prompts are enriched, the user is charged for all ten, resulting
in a higher total usage price; double that of our proposed
OB-CaaS. In RaaS, each prompt is priced uniformly (fixed
price). To ensure a fair comparison of the enrichment process
for the RaaS, we assigned the average price for the prompt
enrichment in the OB-CaaS variant, which is 0.8 currency
units. Consequently, the total price is 8 currency units (0.8
x 10 prompts).
Conversely, the proposed OB-CaaS and LB-CaaS are trans-
parent in calculating the price associated with the prompts
enrichment process because pricing is chunk-dependent. The
transparent calculation is facilitated by assigning chunks with
various prices, at the indexing stage, reflecting the value of
their associated data source. Thus, in these proposed variants,
the number of priced prompts equals the number of enriched
prompts, which is 50% less compared to RaaS.
To realize the cost-effectiveness of utilizing the proposed
OB-CaaS, we multiply the average cost of utilized chunks
(0.8) by the number of the enriched prompts (5), resulting in
4 currency units, which is 50% less than the associated price
of RaaS without a sacrificing on the quality of the generated
output. Quality is maintained at 0.84 for both RaaS and OB-
CaaS.
Regarding LB-CaaS, since the proposed UCOSA strives to
enrich prompts by striking a balance between the relevance
score and the price of the selected chunks, the associated
relevance scores and prices are less compared to OB-CaaS and
RaaS. The average price for the utilized chunks is 0.4 currency
units. Therefore, the total price of LB-CaaS is 2 currency units
(0.4 x 5 prompts), which is 50% less than OB-CaaS and 75%
less than RaaS, highlighting the higher budget utilization of
LB-CaaS. This comes at the expense of only a 12% reduction
in the output quality.
For a valid comparison of the RAG variants, we can either
use a fixed number of enriched prompts (NEP) and observe
its impact on the total usage price and the quality of the
enriched prompts, or we can allocate the same budget for
all RAG variants and observe its impact on the quality and
quantity (NEP) of the enriched prompts. For simplicity, and
to avoid cluttering the demonstration with many prompts, weopted to assume a fixedNEP.
To illustrate LB-CaaS’s budget efficiency, we allocated
a limited budget of 2 currency units, whereas OB-CaaS
and the standard RaaS operate without a predefined budget,
highlighting their limitations in supporting budget-constrained
scenarios.
In conclusion, end-user accessibility increases significantly
when moving from RaaS to OB-CaaS, and even more so with
LB-CaaS, making LB-CaaS the most feasible option for users
with limited budgets.
C. Related Work
LLMs have transformed numerous industries through their
exceptional capabilities. However, they still have several limi-
tations [1], including restrictions on context length [16], issues
with outdated knowledge [17], hallucinating untrue output [6],
and challenges with proper attribution [18]. Each of these
limitations is the focus of active research aimed at addressing
them. For example, hallucinations are being tackled through
augmentation with external databases (RAG), and hallucina-
tion detection methods [19]–[21]. RAG [9], [10] has been
explored in various variants in the literature. Gao et al. [22]
demonstrated three variants of RAG: naive RAG, advanced
RAG, and modular RAG. Both naive and advanced RAGs
consist of pipeline stages, including indexing, retrieval, and
generation, following a sequential process. However, unlike
naive and advanced RAGs, which maintain a sequential flow,
the modular RAG variant incorporates iterative and adaptive
retrieval strategies.
To differentiate various RAG systems, their limitations
should be identified, and their performance should be eval-
uated. To this end, Barnett et al. [23] report on the failure
points of RAG systems, emphasizing that validation of an
RAG system is only truly feasible during its operation and the
robustness of an RAG system is developed over time rather
than being fully established at the beginning. Additionally,
Chen et al. [24] introduce the Retrieval-Augmented Generation
Benchmark (RGB) to identify the limitations and evaluate
the effectiveness of RAGs when they integrate with different
LLMs.
Standard RAG can be implemented on-premise, cloud-
hosted, or offered as a cloud service, i.e., RAG-as-a-Service
(RaaS) [11]–[15] (as illustrated in Fig. 2). However, these
existing implementations are not cost-effective: managing on-
premise RAG requires investment in infrastructure and expen-
sive expertise. Similarly, cloud-hosted requires payment for
cloud resources and management staff. Additionally, both on-
premise and cloud-hosted RAGs require laborious updating of
the indexed data sources. On the other hand, RaaS is more
cost-effective compared to the on-premise and cloud-hosted
implementations. However, the end user keeps paying for all
the submitted prompts, regardless of the quality and quantity
of the actually utilized chunks.
In addition, in standard RAG implementations, the retrieved
chunks are selected solely based on their relevance score,
without incorporating a data source-dependent cost to better
define and differentiate the chunks. This limitation in the

6
granular description of the chunks makes identifying the
utilized chunks challenging. Consequently, the standard RaaS
implementation offers less flexibility and is provided without
considering the actual usage of the chunks. The lack of granu-
larity in calculating the utilized chunks makes the operational
cost higher, which, in turn, impacts the overall service cost,
resulting in an expensive pricing model. Therefore, RaaS
accessibility is restricted to end users who can afford such
costs. This highlights a clear gap in the existing business
models of standard RAG, including RaaS.
To bridge this gap, we propose modifying the standard RAG
pipeline to incorporate a weight (a price) to the chunks of
each data source based on predetermined factors set by the
data source owner and the service provider (more details in
Section V-B). Therefore, the utilized chunks can be quantified,
facilitating the introduction of our proposed CaaS. Table I
summarizes the comparison between standard RAG and our
proposed CaaS variants, including OB-CaaS and LB-CaaS.
Compared to the standard RaaS, our proposed CaaS variants
excel in many aspects, including cost-effectiveness, user ac-
cessibility, and an enhanced pipeline that enables transparent
usage cost calculation. Notably, LB-CaaS introduces a further
enhancement to these aspects.
With the ongoing competition among RaaS providers, their
pricing plans continue to evolve [25]–[27]. Regardless of
the offered prices, adopting our proposed CaaS variants is
supposed to achieve further cost reductions as it optimizes the
underlying process itself: the pipeline of the standard RAGs,
including RaaS.
Learning-Based Online Selection Techniques:There are
learning-based online selection techniques, such as RL- or
ML-based approaches, that are used for online selection [28]–
[30]. However, these techniques are complex. They are associ-
ated with pre-training overhead. For example, Tang et al. [28]
proposed an RL-based framework to select a retrieval method
based on the complexity of the served prompt, suggesting
that each prompt requires a different retrieval approach. In
contrast, our proposed UCOSA is a lightweight and training-
free algorithm that works on the fly without training overhead.
Such a simple and lightweight technique ensures the practical-
ity and simplicity of our proposed budget-constrained variant
(LB-CaaS).
Budget-Constrained Cloud Resource Allocation:Inspired
by the established cloud computing services, such as softwareas a service, platform as a service, and storage as a service
[31], where cloud resources are abstracted and offered as
independent services to facilitate utilization and integration
with other systems and services in a cost-effective and efficient
manner, we propose offering the key building block in RAG
systems “the chunk” as a service to gain the same merits of
the “as-a-service” paradigm, including the smooth integrations
with other systems in a cost-effective and efficient way.
In particular, our proposed budget-constrained variant (LB-
CaaS) is related to the well-established principles of budget-
constrained optimization in cloud resource allocation [29], [30]
as follows. Both approaches aim to optimize utility under
cost constraints: cloud resource allocation aims to maximize
the utilization of computational resources, while LB-CaaS
aims to cost-effectively utilize information resources (chunks),
considering their relevance to the served prompt and their
associated price. However, they differ in the type of resources
and the decision-making paradigm. Cloud resource allocation
optimizes computational resources (CPU, memory, storage,
bandwidth) [29], [30], [32] and typically uses prediction-
based provisioning [29], [30] where future resource requests
are predicted using machine learning models. In contrast, our
LB-CaaS model optimizes the information resources (chunks)
selection, utilizing the proposed online selection algorithm
(UCOSA), where each chunk must be immediately accepted
or rejected upon arrival without knowledge of future chunks.
III. CHUNK-AS-A-SERVICE(CAAS): PROPOSEDRAG
VARIANTS
This section details our proposed work. First, we introduce
the Open-Budget CaaS (OB-CaaS), the basic variant. Then, we
present the Limited-Budget CaaS (LB-CaaS), the cost-aware
variant.
A. Proposed OB-CaaS Variant
Standard RaaS lacks the capability enabling quantifying the
actually utilized chunks in the prompt’s enrichment process.
For example, data source owners, such as book authors, are
unable to align their sales with RaaS’s operational style.
Ideally, they would earn profits reflecting the rate of usage and
the value of the utilized chunks associated with their indexed
data. This would be a fair approach and would encourage
creating high-quality data source content due to the expected
TABLE I: Related work summary. Compared to the standard RAG, our proposed CaaS variants excel in many aspects, including
cost-effectiveness, user accessibility, and an enhanced pipeline that enables transparent usage cost calculation. In particular,
LB-CaaS introduces a further enhancement to these aspects.
Ref. RAG Business ModelCost
EffectivenessEnhanced
PipelineRetrieved Chunks
CharacteristicsChunk Selection
CriteriaEnd User
AccessibilityRAG Service
TypePayment Model
[11], [12]
[13], [14]
[15]On-Premise RAG Low No Relevance Score Relevance Score N/A N/APaying for in-house
expertise and
hardware resources
Standard RAGCloud-Hosted RAG Low No Relevance Score Relevance Score N/A N/APaying for cloud
resources and
management staff
RAG-as-a-Service (RaaS) Low No Relevance Score Relevance Score Low RaaSPay-as-you-Go
(Open Budget)
This Work Proposed RAGOB-CaaS Medium YesRelevance Score
and Chunk PriceRelevance Score Medium CaaSPay-as-you-Go
(Open Budget)
LB-CaaS High YesRelevance Score
and Chunk PriceRelevance Score,
Chunk’s Price, and
Remaining BudgetHigh CaaS Limited Budget

7
Plain 
Prompt-1
Grounded
OutputProposed OB-CaaS (Basic V ariation)
Indexed
ChunksRetriever
R: 0.92
P: 0.03
Chunks associated with their prices (P)
and relevance (R). They are ordered
based on their relevance  to the
corresponding prompts.R: 0.90
P: 0.08
R: 0.82
P: 0.04R: 0.85
P: 0.02Plain 
Prompt-mPlain 
Prompt-n
R: 0.42
P: 0.03
R: 0.40
P: 0.02  LLM Enriching Prompts Approach LLMStream of Prompts
Synthesizer 
R: 0.92
P: 0.03Plain 
Prompt-m
R: 0.90
P: 0.08
R: 0.42
P: 0.03Plain 
Prompt-nPlain 
Prompt-1
Enriched
Prompt-1Enriched
Prompt-mEnriched
Prompt-nStream of Enriched Prompts
Flow of prompt-n Flow of prompt-m Flow of prompt-1
Fig. 4:The pipeline of the OB-CaaS. Unlike standard RaaS where the indexed chunks are not incorporated with a price reflecting
the value of their data sources, the indexed chunks in the proposed OB-CaaS have a negotiation-based price set during the
price agreement stage between the data source owner and the service provider (detailed in Section II-B). This incorporation
of the prices into the chunks enables quantifying the end user’s utilized chunks and facilitates the introduction of the CaaS
model in its basic variant (OB-CaaS).
competition among content creators to produce valuable and
competitive content to find its opportunity to be selected
and utilized in the prompt enrichment process. Therefore, we
propose the ’Chunk-as-a-Service’ (CaaS) business model to
address the limitations of standard RaaS, including the lack of
chunk-based usage price calculation. Similar to standard RaaS,
our proposed CaaS is intended to be deployed on the cloud and
managed by a third party. However, it offers a chunk-based,
transparent usage price calculation, and more cost-effective
solution.
CaaS adapts the data source indexing process to assign
prices to the indexed chunks based on predetermined criteria
set by the CaaS provider and the data source owner (detailed
in Section II-B). In its basic variation, CaaS is considered
an Open-Budget CaaS (OB-CaaS) model. It does not require
the end users to set a predefined budget. The usage price
is calculated based on the associated prices of the utilized
chunks, which are selected considering one factor: their degree
of relevance to the prompts regardless of the associated prices.
The key parties involved in the CaaS model are the CaaS
provider, the data source owner, and the CaaS subscriber (the
end-user). The data is provided by the data source owner,
managed by the CaaS provider, and utilized by the CaaS
subscriber.
Fig. 4 illustrates the pipeline of our proposed OB-CaaS.
It highlights our contributions in the indexing and retrieval
stages. Compared to the standard RAG pipeline, where the
price of usage does necessarily not consider the actual uti-
lization of chunks, the pipeline of our proposed OB-CaaS
considers the price of the utilized chunks. This is due to the
predetermined prices assigned to the chunks of the indexed
data sources. The cost is simply the total price associated
with the utilized chunks. Different colors are used to visually
distinguish the flow of individual prompts (Prompt-1, Prompt-
m, Prompt-n) through the CaaS pipeline. Although OB-CaaS
surpasses standard RaaS in many aspects, it still shares akey limitation: it is less accessible to a broader range of end
users, particularly those with a limited budget. This is because
OB-CaaS lacks an intelligent chunk selection functionality
that prioritizes chunk selection based on both relevance score
and price, preventing it from being a cost-effective option
for limited-budget users. Therefore, we propose a cost-aware
variant beyond the OB-CaaS, as explained below (Section
III-B).
B. Proposed LB-CaaS Variant
LB-CaaS addresses the shortcomings of OB-CaaS, specifi-
cally the relative cost-ineffectiveness and lack of accessibility
for limited-budget users. This variant enhances the pipeline of
OB-CaaS pipeline as follows:
At “Retriever” stage, for each plain prompt in the stream,
instead of the typical process of retrieving indexed chunks
based on the relevance score and then selecting the most
relevant one, the relevant chunks are only retrieved1, while
the selection process is delegated to a new contributed stage
named the “Chunk’s Selection Process” to optimize the se-
lection considering the remaining budget, chunk’s relevance
score, and chunk’s price.
At the “Chunk’s Selection Process” stage, UCOSA se-
lects chunks with relevance-to-price ratios above a budget-
dependent threshold, forming a set of candidate chunks (CCH).
From within a CCH, the chunk with the highest relevance
score is then selected (highlighted in orange) and passed to
another stage named “Enriching prompts with the selected
chunksStage “Enriching prompts with the selected chunks
(Prompt Online Selection)”begins by checking the availabil-
ity of a selected chunk associated with each prompt.
If a chunk is selected by the previous stage, the associated
plain prompt utilizes the RAG’s synthesizer component to
1The retrieved chunks can be stored in a cache, which is emptied after the
selection of a chunk to prepare for the next prompt.

8
Plain 
Prompt-1
Grounded
OutputProposed LB-CaaS (Cost-A ware V ariation)
Chunks' Selection Process
Traditional
OutputRetriever  
R: 0.92
P: 0.03
Initial sets of fetched chunks
retrieved based on their relevance
(R) to the corresponding prompts.
Price (P) is not considered as a
retrieval factor at this stage.R: 0.90
P: 0.08
R: 0.82
P: 0.04R: 0.85
P: 0.02
CCH-n CCH-m CCH-1Plain 
Prompt-mPlain 
Prompt-n
R: 0.62
P: 0.04R: 0.60
P: 0.02
UCOSA  selects chunks with relevance-to-price ratios
above a budget-dependent threshold, forming a set
of candidate chunks ( CCH ). From within a CCH, the
chunk with the highest relevance score is then
selected (highlighted in orange).R: 0.42
P: 0.03
R: 0.40
P: 0.02
R: 0.20
P: 0.02A chunk is
selected?
Synthesizer
Enriched
Prompt-m
 LLM 
Plain 
Prompt-1R: 0.72
P: 0.04R: 0.70
P: 0.02R: 0.30
P: 0.02Yes
NoEnriching Prompts with the Selected Chunks ( Prompt Online Selection )Stream of Prompts
Selected chunkBudget-dependent
thresholdIndexed
ChunksPlain 
Prompt-m
Enriched 
Prompt-nPlain
Prompt-n
Flow of prompt-n Flow of prompt-m Flow of prompt-1
Fig. 5:The pipeline of the proposed LB-CaaS. A cost-aware variant that is more cost-effective than the OB-CaaS, allowing
accessibility for end users with limited budgets. Utilizing UCOSA, a proposed online selection algorithm, a subset of plain
prompts is selected optimally on the fly for the enrichment process, enabling cost-efficient generation of grounded output.
compose an enriched prompt and pass it to the LLM to get
a grounded output. However, if none of the chunks were
selected, the corresponding plain prompt is switched to the
LLM directly, generating traditional output.
Selecting a chunk in this context means that the associated
prompt is chosen in an online fashion for the enrichment
process, and this decision is irrevocable.
The algorithmic representation of Figure 5 is illustrated
in line with the presentation of the UCOSA (see Algorithm
1), the enabler of LB-CaaS. The full details of UCOSA are
provided below in Section IV.
IV. ONLINESELECTIONALGORITHM: MODELING AND
ANALYSIS
This section details the mathematical modeling of our
proposed online selection algorithm (UCOSA), including the
settings, optimization technique, algorithmic representation,
and competitive ratio analysis
A. Settings
There areI={1,2, . . . , i, . . . , I}plain prompts andJ=
{1,2, . . . , j, . . . , J}chunks. Enriching thei-th prompt with
thej-th chunk achieves a relevance score ofR ijwith a price
ofP ij2. The users (i.e.,the providers of the prompts) have a
budgetB. The objective is to find an assignment of prompts
to chunks that maximizes the overall Relevance score, without
exceeding the budget.
B. Optimization Technique
The problem of chunk selection is formulated as an opti-
mization problem. Let:
2The price of a chunk may be the same, regardless of the prompts. However,
we assume a different price for different prompts as a generalization.xij=

1if thei-th prompt is enriched
with thej-th chunk
0otherwise.
Then the optimization becomes:
maxX
iX
jxijRij (1)
Subject to:
X
iX
jxijPij≤B(2)
X
jxij≤1∀i(3)
Where Constraint (2) states the budget limit, and Con-
straint (3) states that a prompt can be enriched with 1 chunk at
most. The optimization technique assumes that all information,
regarding prompts and chunks, are known in advance. The
formulation above is for LB CaaS. Note that settingB→ ∞
converts it to OB CaaS.
C. Online Selection Algorithm (UCOSA)
Under the online settings, the prompts are fed to the
LBCaaS RAG as a stream (i.e.,one by one). For each prompt,
an initial set of chunks are retrieved (by the Retrieval stage)
based on their Relevance score,R ij. At the Chunk Selection
stage, a decision must be made to choose, at most, one chunk
to enrich the prompt, considering the Relevance score, the
priceP ij, and the available budgetB. Note thatR ijandP ij
are only known when the prompt is fed to the RAG system,
not before. A decision for a prompt must be made before
considering the next prompt, and these decisions cannot be
revoked afterward. To meet these requirements, we propose
the Utility-Cost Online Selection Algorithm (UCOSA). With
respect to UCOSA, the relevance is required to be represented

9
as a scalar. However, the way this scalar is obtained is
completely under the control of the service provider. It could
be a simple relevance measure, or could be obtained through a
weighted function of several variables (e.g., diversity, novelty).
Relevance is a well-established metric used in the literature to
calculate the similarity between two texts and is often used
interchangeably with the term similarity [33], [34].
UCOSA is a generic algorithm that can be applied in various
scenarios using diverse representations of utility and cost. In
this work, the utility is represented by the relevance score
Rof the retrieved chunk, while the cost is represented by
the associated pricePof each retrieved chunk. UCOSA is a
specialized online selection variant suited to the settings of
our proposed LB-CaaS RAG, where chunks are defined by
relevance and price. It requires assigning a price at the time of
selection (assigning prompts the chunks), while the assignment
of prices to the chunks is part of the indexing stage.
Before we state the details of UCOSA, some assumptions
must be stated:
•There exists non-trivial bounds on the values ofRij
Pij. That
is,0< L≤Rij
Pij≤U <∞.
•Pij
B≤ϵ, whereϵis close to 0.
The first assumption means that neither the Relevance score
can be infinite, nor the price can be 0. Otherwise, the solution
becomes trivial (by selecting the chunk with the infinite
score, or selecting the chunk with a zero price). The second
assumption states that the price of a chunk is relatively small
compared to the available budget. This assumption aligns well
with real-world cloud service scenarios, where a predefined
budget typically serves multiple requests (storage, computa-
tion, or retrieval). Such a realistic assumption helps define
an appropriate business model for both service providers and
end users. In addition, users usually tend to get services from
providers that offer a small price compared to their budget.
This low-pricing model is common and widely adopted by
cloud service providers, including AWS Lambda [35].
UCOSA selects chunks with relevance-to-price ratios above
a budget-dependent threshold, forming a set of candidate
chunks (CCH). From within a CCH, the chunk with the highest
relevance score is then selected.
Mathematically speaking, letz ibe the fraction of budget
used when thei-th prompt is fed to LB CaaS, and define
Ψ(zi) = 
Ue/Lzi(L/e), where,eis the natural number. Let
Qibe the set of chunks that satisfyRij
Pij≥Ψ(z i). IfQ iis
empty, thei-th prompt is passed to LLM without enrichment.
Else, we pick the chunk fromQ ithat gives the maximum
Relevance score, and enrich the prompt with that chunk.
So, the impact of UCOSA on thresholdΨ(z i)(the budget-
dependent threshold) is as follows: the more UCOSA selects
chunks and uses a higher fraction of the budget, the threshold
becomes higher, and UCOSA becomes stricter in selecting the
next chunks. UCOSA is shown in Algorithm 1. Appendix A
demonstrates that the performance of UCOSA is competitive
with respect to the optimization formulation.Algorithm 1UCOSA
Inputs:
•Plain Prompts:Stream of plain prompts
•B:Allocated Budget
Outputs:
•Selected Chunks:A subset of the retrieved chunks
•Enriched Prompts:A subset of Plain Prompts enriched
with their associated selected chunks.
1:Initially,z i= 0
2:forEach promptiinPlain Prompts do
3:Retrieval Stage:
4:fetch the relevant chunksJ i
5:Selection Stage:
6:Computez i,Ψ(z i)
7:SetQ i={j∈ J i|Rij
Pij≥Ψ(z i)}
8:find ˆj=argmax j∈QiRij
9:Enrichment Stage:
10:ifNo chunk is selectedthen
11:Pass the prompt to LLM without enrichment
12:else
13:Enrich promptiwith chunk ˆj
14:end if
15:end for
V. MANAGEMENT OF THEPROPOSEDCAAS MODEL
A. Indexing and Retrieval Management
RAG providers facilitate building and managing RAG
pipelines. For example, frameworks such as Haystack [36]
can be used to flexibly build custom RAG pipelines for on-
premise use or cloud deployment. A RAG pipeline comprises
key components, including indexing and retrieval. Typically,
RAG providers offer options for managing these components
either by using a built-in vector database, such as the Vector
Store Index [37] of LlamaIndex, or by utilizing third-party
vector database engines, such as FAISS [38], Pinecone [39],
Chroma [40], and Elasticsearch [41].
For several reasons, for our proposed RAG variants, in-
cluding OB-CaaS and LB-CaaS, managing the indexing stage
requires a built-in vector database. First, once the data source
owner and CaaS provider sign a revenue-sharing agreement
(discussed in Section V-B), the loaded data becomes a pro-
prietary asset for both parties, necessitating its protection
through internal management and storage. Second, the built-in
vector database allows us to adapt the pipeline to support the
proposed RAG variants. For instance, the indexing stage can
be adapted to assign chunk prices during the chunking and
storing steps. The retrieval stage can also be modified to in-
corporate budget-constrained online selection using UCOSA.
This means that utilizing an external vector database (e.g.,
through an API) is not a viable option.
B. Revenue Sharing Management
Various subjective factors determine the reputation and
reliability of the service provider of the proposed CaaS variants
and the value of the data source intended to be loaded and

10
indexed within these proposed CaaS variants. Therefore, we
adopt a negotiation-based agreement for the revenue-sharing
of our proposed CaaS variants. To ensure transparent and
reliable revenue-sharing, a relevant well-established system,
such as blockchain-based smart contracts [42], [43] or a
dedicated revenue management platform [44], [45], is assumed
to be integrated with the proposed CaaS payment model to
handle the revenue-sharing process. These systems can provide
automated, immutable, and transparent tracking of revenue
splits, ensuring both parties have confidence in the fairness
and accuracy of the agreement.
We assume that CaaS providers make rational agreements
that reflect the true value of the negotiated data source.
Otherwise, users might end up paying more for lower-quality
content.
VI. EXPERIMENTS ANDRESULTS
In this section, we conduct extensive experiments to empir-
ically investigate the performance of UCOSA.
A. Performance Evaluation Metrics and Baselines
Evaluation Metrics:For our proposed LB-CaaS, the impact
on the quantity and the quality of the generated output are the
two key factors that should be considered during the evaluation
process.
•Quantity: How many enriched prompts does the pro-
posed RAG provide (the number of the online selected
prompts)?
•Quality: How impactful these enriched prompts are on
the generated output?
We use the chunk’s Average Relevance (AR) as a proxy to
measure the quality of the generated output. The more relevant
the chunk is to the plain prompt, the higher the potential
that the associated enriched prompt incorporates to generate
quality (reliable) output. For quantity, we count the Number of
Enriched Prompts (NEP): the set of online selected prompts
augmented by the selected chunks. In this work, we use the
terms quality and relevance score interchangeably.
In limited-budget selection scenarios, considering quality
(AR) or quantity (NEP) alone does not guarantee optimal
performance. High-quality output may sometimes coincide
with a significant decrease inNEP. Conversely, scenarios
with highNEPmight exhibit low quality (i.e., lowAR).
Therefore, we propose a performance evaluation metric that
incorporates both quantity and quality to comprehensively
assess the impact of our proposed approach on the generated
outputs.NEPdenotes quantity andARdenotes quality.
Given thatARis a value greater than 0.0 and less than or equal
to 1.0, where values closer to0.0are considered least relevant
and those closer to1.0are highly relevant, multiplyingNEP
by qualityARserves as a fair performance evaluation met-
ric. This metric (NEP×AR) penalizes low-quality outputs
and rewards those with higher quality, providing a balanced
assessment.
Table II demonstrates the descriptions of the notations used
in the conducted experiments.TABLE II: Description of the notations used in the conducted
experiments.
Annotation Description
NEP Number of Enriched Prompts. The higher the better.
AR The Average Relevance score of the selected chunks.
The higher the better.
(NEP×AR) This metric rewards high relevance and penalizes low
relevance, with higher scores being better.
B The allocated budget
P Price per chunk per data source
Total Relevance The total relevance scores of the enriched prompts.
Baselines:We use offline selection and “relevance-greedy”
selection algorithms as baselines to compare the performance
of our proposed approach. In online selection, prompts are
chosen on the fly, and the decision cannot be revoked. Offline
selection represents the ideal selection scenario, where all
prompts are available for analysis beforehand. In contrast,
“relevance-greedy” selection represents the typical selection
approach that is considered by standard RAG, including the
RaaS variant.
B. Experimental Setup
We used LlamaIndex RAG framework [46] to conduct our
experiments. Within LlamaIndex, we utilized ChatGPT-3.5
[47] as LLM and OpenAI text-embedding-ada-002 [48] as an
embedding model.
Our extensive experiments run on top of the modified RAG’s
pipeline that we propose to accommodate our proposed CaaS
variations. The typical stages of RAG’s pipeline include data
source loading, indexing, retrieval, synthesis, and generation.
Our proposed work mainly contributes to the indexing and
retrieval stages.
For indexing, we used LlamaIndex default vector database:
Vector Store Index [46]. The indexed chunks are then stored in
local storage using LlamaIndex Storage Context to avoid the
overhead of re-indexing. Our proposed UCOSA is agnostic to
the choice of similarity metric. We used cosine similarity as a
representative metric for its popularity in RAG-related works
[49], [50]. It is the de facto standard and the default setting
in many retrieval systems and RAG frameworks, including
LlamaIndex [51], [52], where it is used to compute the
similarity between two embedding vectors: one represents the
prompt and the other represents an indexed chunk. Moreover,
cosine similarity excels in capturing the semantic relationship
between embeddings within high-dimensional vector spaces
[53]. We set the parameter “similarity-top-k”, the number of
retrieved chunks, to 20, giving a sufficient chance to our
proposed UCOSA to select chunks while striking a balance
between the relevance score (similarity) and the price.
Chunk size influences UCOSA’s performance because it
determines the quality of the candidate chunks available for
selection. With the optimal chunk size, UCOSA’s selections
maximize the quality of the generated output, whereas smaller
or larger chunks make UCOSA less effective, though it
remains functional. We set the chunk size to 1024 tokens
because previous evaluation studies, conducted by leading
RAG framework providers, including LlamaIndex [54], found

11
it optimal. They have already investigated the impact of
chunk size on relevance and found that 1024 is the ideal
chunk size for the best relevance score because this size is
large enough to capture context while being manageable for
processing. Additionally, we set chunk overlap to 20 tokens
as it helps ensure smoother transitions between chunks and
avoids information loss at chunk boundaries.
We designed the experiments to handle prompts with con-
tent from fields related to the areas of the indexed data sources.
This design allows the retrieval stage to find candidate chunks
that are relevant to the prompt. Thus, if UCOSA finds one of
the candidate chunks feasible to be selected, the plain prompt
is augmented with this chunk, composing an enriched prompt
(online prompt selection).
For this purpose, we loaded and indexed a few data sources
(e-books) that cover the areas of adversarial robustness and
selection algorithms, namely the books, Adversarial Robust-
ness for Machine Learning [55], Strengthening Deep Neural
Networks [56], and Algorithms to Live By: The Computer
Science of Human Decisions [57]. We assigned predetermined
prices of 0.8, 0.2, and 0.1 currency units for the chunks of each
book, respectively, introducing price variability to facilitate the
algorithm’s prioritization during selection. We assume that the
predetermined price for chunks of each book is set following a
negotiation-based agreement between e-book authors and the
CaaS provider as outlined in Section V-B. Only then does
the data source become available and ready to be loaded and
processed by the CaaS provider.
Additionally, we generated questions (plain prompts) from
areas that vary in proximity to these core topics, allowing for
a range of relevance scores in potential retrieved chunks to
test how UCOSA would handle them. These areas include
explainable AI, security & privacy, game theory, optimization,
and differential privacy.
To gain more confidence with the conducted experiments,
we repeated each experiment 100 times, shuffling the prompts
each time, and then took the average result. Our proposed
algorithm is designed to be framework-agnostic, with the
potential to be implemented within various RAG frameworks
and configurations, irrespective of the specific LLM, embed-
ding model, or vector database employed. The framework
and configuration described above serve as a representative
example to facilitate experiment simulations and demonstrate
the algorithm’s broader applicability. However, further testing
with different LLM implementations would be necessary to
fully confirm this generalization.
C. Research Questions
We perform extensive experiments to answer the following
research questions:
•How does the performance of UCOSA compare to offline
selection (best case) algorithm and “relevance-greedy”
selection algorithms?
•What impact do various allocated budgets have on the
performance of UCOSA?
•How do the performance, used budget, and performance-
to-budget ratio for LB-CaaS compare to OB-CaaS and
RaaS variants?D. Experimental Scenarios And Analysis
The key factors that impact the performance(NEP×AR)
of the proposed CaaS variations are the chunk selection
strategy, budget type (open or limited), and the allocated
budget. For that, we design several experimental scenarios
that investigate the impact of each factor on the overall
performance.
1) The performance of UCOSA compared to offline and
“relevance-greedy” selection algorithms:This experimental
scenario investigates the performance of UCOSA compared to
the offline (best case) and “relevance-greedy” selection algo-
rithms. Fig. 6 presents the results of this scenario considering
NEP,AR, and (NEP×AR).
Starting withNEP, Fig. 6a illustrates the number of
enriched prompts (NEP) when UCOSA is utilized compared
to the usage of offline and “relevance-greedy” selection algo-
rithms. In terms ofNEP, which represents the quantity of the
target generated output, UCOSA achieves a higherNEPthan
“relevance-greedy” selection and performs competitively with
offline selection. The competitive performance of UCOSA
in terms ofNEP, compared to the benchmarked offline
selection (which is impractical for real-time RAG systems),
underscores UCOSA’s efficiency and effectiveness in real-
time applications. Its ability to achieve competitiveNEP
in dynamic and unpredictable environments makes UCOSA
a strong candidate for RAG systems that require immediate
responses.
Moving toAR, Fig. 6b shows the average relevance scores
(AR) of the selected chunks when UCOSA is utilized com-
pared to the usage of “relevance-greedy” and offline selection
algorithms. In terms ofAR, which represents the quality of
the target generated output, UCOSA achieves competitive per-
formance compared to the “relevance-greedy” selection algo-
rithm and superior performance compared to offline selection
algorithms. The “relevance-greedy” selection algorithm strives
to maximize theARof the selected chunks regardless of
their associated prices. In other words, the “relevance-greedy”
approach prioritizes relevance scores over cost reduction,
potentially sacrificing output quantity. Therefore, it achieves
higherAR. Interestingly, theARfor UCOSA is higher than
that of offline selection. One interpretation of this observation
is that since the NEPs are high when offline selection is
utilized, there is a possibility that some of the involved prompts
are enriched with chunks of low relevance score values. These
low relevance scores, when included in calculating the average,
could reduce theARfor offline selection. UCOSA’s ability to
maintain high relevance in real-time selection highlights its
robustness in dynamic environments.
Concerning UCOSA performance, Fig. 6c illustrates that
the performance metric(NEP×AR)fairly evaluates the
performance of UCOSA because it considers both the quality
and quantity of the generated output.
For this metric, UCOSA outperforms “relevance-greedy”
selection by approximately 52% and achieves around 75% of
the performance of offline selection methods, demonstrating
its effectiveness as a competitive, real-time selection method.
Unlike offline selection, which assumes the availability of
all prompts beforehand—a scenario not typical in real-world

12
(a) When UCOSA is used,NEPis higher
than relevance-greedy selection and compet-
itive with offline selection.
(b) When using UCOSA,ARof the selected
chunks is superior to offline selection and
competitive to relevance-greedy algorithms.
(c) UCOSA outperforms “relevance-greedy”
selection and is competitive with offline se-
lection algorithms.
Fig. 6: Performance of UCOSA compared to the offline (best case) and relevance-greedy selection algorithms considering
NEP,AR, and (NEPxAR). For the considered performance metric (NEPxAR), UCOSA shows superior performance
compared to relevance-greedy selection and competitive performance to the offline selection algorithm.
Fig. 7: UCOSA’s performance improves with increased budget
until it plateaus. This plateau indicates the budget required to
enrich all plain prompts, marking the conversion from LB-
CaaS to OB-CaaS.
LLM applications—UCOSA operates in real time, dynami-
cally selecting prompts as they arrive. This makes UCOSA
particularly valuable for practical applications where prompt
streams are unpredictable and not fully available in advance.
2) The impact various allocated budgets have on the
performance of UCOSA:In this experimental scenario, we
investigate the impact of various allocated budgets on the
performance of UCOSA.
Fig. 7 illustrates the impact various allocated budgets have
on the performance of UCOSA. UCOSA’s performance posi-
tively correlates with the increased allocated budget. However,
it reaches a point where performance stops increasing no
matter what the increased budget is. This point is interpreted
as the required budget to enrich all plain prompts. In other
words, the point where LB-CaaS converts to OB-CaaS variant
(i.e.,B=∞).
3) The the performance, used budget, and performance-to-
budget ratio for LB-CaaS compared to OB-CaaS and RaaS:
Fig. 8 illustrates an experimental scenario that compares the
performance, used budget, and performance-to-budget ratio of
LB-CaaS with the OB-CaaS and RaaS variants.
Unlike OB-CaaS and LB-CaaS variants, where the costis chunk-dependent, the cost of RaaS is calculated for the
submitted prompts. For the sake of a fair comparison, we
consider the cost of each submitted prompt to RaaS as the
average cost of the chunks in OB-CaaS and LB-CaaS variants.
Firstly, Fig. 8a demonstrates the performance of LB-CaaS
compared to OB-CaaS and RaaS (the higher, the better). LB-
CaaS shows lower performance than OB-CaaS and RaaS.
However, OB-CaaS and RaaS achieve this superiority at the
expense of a higher cost, i.e., a significantly higher budget as
demonstrated in Fig. 8b.
Secondly, Fig. 8b shows the used budget of LB-CaaS
compared to OB-CaaS and RaaS (the lower, the better). The
used budget of LB-CaaS is proportional to the associated
performance illustrated in Fig. 8a and LB-CaaS is the most
cost-efficient variant as demonstrated in Fig. 8c. Furthermore,
we observe that although the used budget for OB-CaaS is
less than that of RaaS, the associated performance of both
is similar, highlighting the cost efficiency of OB-CaaS.
Lastly, Fig. 8c evaluates the performance-to-budget ratio for
each RAG variant. LB-CaaS and OB-CaaS achieve higher
performance-to-budget ratios, 140% and 86%, respectively,
compared to RaaS, indicating their superior efficiency in
budget utilization. In particular, LB-CaaS exhibits the highest
performance-to-budget ratio, indicating its superior efficiency
in budget utilization compared to OB-CaaS and RaaS. OB-
CaaS demonstrates intermediate efficiency, while RaaS shows
the lowest efficiency in budget utilization. This analysis under-
scores the practicality of LB-CaaS. By balancing the cost with
performance, LB-CaaS emerges as a viable and cost-effective
RAG variant.
4) Human-in-th-Loop Evaluation:To complement the
simulation-based results with human-in-the-loop verification,
we assessed how chunk-level pricing and prompt enrichment
influence the actual user experience by incorporating human
experts into the evaluation process. In particular, three human
experts (information retrieval researchers) investigated the
chunks associated with 50 randomly sampled prompts, com-
paring those selected by our proposed UCOSA (the enabler of
LB-CaaS) against chunks selected by the baseline approaches
(Offline and Relevance-Greedy). The findings of the experts
emphasized the simulated results: chunks selected by our

13
(a) OB-CaaS outperforms LB-CaaS and
matches RaaS performance, albeit at a higher
cost.
(b) The budget of LB-CaaS compared to OB-
CaaS and RaaS (lower is better). LB-CaaS’s
budget aligns with its performance shown in
Fig. 8a
(c) Performance-to-budget ratio shows higher
budget utilization for LB-CaaS (enabled by
UCOSA) compared to OB-CaaS and RaaS.
Fig. 8: The performance, used budget, and the performance-to-budget ratio of LB-CaaS compared to the OB-CaaS and RaaS
variants. LB-CaaS exhibits the highest performance-to-budget ratio, indicating its superior efficiency in budget utilization
compared to OB-CaaS and RaaS. OB-CaaS demonstrates intermediate efficiency, while RaaS shows the lowest efficiency in
budget utilization.
proposed work were found to be sufficiently informative and
relevant to the served prompts while also being more cost-
effective (associated with less price). On the other hand,
chunks selected by the baseline approaches were found to be
of comparable informativeness but with higher prices. These
results emphasize that UCOSA effectively balances utility
(relevance) and cost (price) in a way that positively influences
the actual user experience and information satisfaction.
E. Insights and Lessons learned
•Relying solely on the relevance or price might lead to in-
effective generated output. For example, relying on chunk
relevance might lead to serving fewer prompts when the
top relevant ones are associated with higher prices. There
could be cheaper chunks with a degree of relevance scores
that are sufficient to generate quality output. However,
these chunks are neglected when utilizing relevance as
the only selection metric. Likewise, if we only rely on the
price for the selection of the chunks, the process tends to
select the cheaper chunks even if they are associated with
lower relevance scores that are insufficient to generate
quality output. Therefore, striking a balance between
relevance and price is a prudent strategy.
•OB-CaaS can be extended to various enhanced variants
besides the proposed LB-CaaS. For example, a variant
optimized for the cost (OFC-CaaS) and another opti-
mized for the relevance (OFR-CaaS). OFC-CaaS would
be suitable for the use cases that prioritize the quantity
of enriched prompts over their quality. In contrast, the
OFR-CaaS variant is suitable for use cases that prioritize
the quality of enriched prompts over their quantity. Our
proposed variant LB-CaaS, on the other hand, strikes a
balance between both quantity and quality, making it
ideal for most real-world scenarios where both aspects
are essential.
•An edge case of the allocated budgets in the LB-CaaS
variant could be a budget that is sufficient for handling
all prompts in the stream (B=∞), which converts it to
OB-CaaS.•The proposed OB-CaaS variant involves post-calculation
of the total price associated with the utilized chunks,
implying less control over the budget. In contrast, the LB-
CaaS variant operates with a pre-defined budget, implying
more control over the budget.
VII. CONCLUSION ANDFUTUREWORK
We propose two RAG variants to address the standard RAG-
as-a-Service (RaaS) limitations. Open-Budget CaaS (OB-
CaaS) is the basic variant that renders standard RaaS cost-
effective, allowing accessibility to broader users interested in
utilizing RAG. The second variant is Limited-Budget CaaS
(LB-CaaS). It is a cost-aware variant extending the accessibil-
ity of the OB-CaaS to serve wider RAG-interested users. We
investigated the performance of our proposed RAG variants by
conducting extensive experiments, demonstrating their efficacy
and proving the optimality of UCOSA (the enabler of LB-
CaaS) mathematically and empirically. In future work, we
plan to investigate the feasibility of integrating the LB-CaaS
model with a hallucination detection technique, enabling the
LB-CaaS to function as a hallucination-detection-driven model
to further enhance its cost-effectiveness. In particular, we
aim to explore the cost reduction compared to the expected
hallucination detection overhead.
Inspired by the typical FL setting, as future work, we
suggest a variant of FL that leverages UCOSA to mitigate
privacy concerns, while enabling CaaS-like functionality. This
variant aims to avoid sharing the whole data sources and only
sharing minimal data (one record for each process) when
necessary. It would follow a two-stage selection approach.
In the first stage, a copy of the served prompt is shared
with all decentralized locations, where UCOSA is deployed
and utilized to select the most feasible chunk (considering
relevance and price) as a candidate and send it to the central-
ized location for further processing. In the second stage, the
centralized location utilizes UCOSA to process the candidate
chunks to make a final selection decision by selecting the most
feasible ones conditioned by the remaining budget. Another
future direction is to develop an online algorithm that targets

14
dynamic pricing scenarios, where the cost of a chunk may
change over time. In addition, we plan to conduct larger-scale
human-in-loop evaluation in future research, including user-
centered experiments with practitioners and general users to
quantify satisfaction, usability, and decision-making impact.
APPENDIXA
UCOSA’SCOMPETITIVERATIOANALYSIS
In this section, we show that UCOSA achieves a competitive
ratio ofln(U/L) + 2. Moreover, we show that the best
competitive ratio achieved by any online algorithm cannot be
lower thanln(U/L)+1. The competitive ratio is a measure of
the worst-case performance of the online algorithm compared
to the optimal solution. In our work, letR UCOSA be the total
Relevance score achieved by UCOSA, and letR OPT be the
total Relevance score achieved by the optimal solution (the
optimization formulation in the main manuscript, Section IV-
B), then we show that
ROPT
ln(U/L) + 2≤ R UCOSA ≤ R OPT
The proof starts by examining the sets of prompts enriched
with a selected chunkjby the offline algorithm and UCOSA,
denoted byS∗
jandS j, respectively. The setS∗
jis partitioned
into subsets. Leveraging the properties ofΨ(z i)and the
selection condition of UCOSA (Line 7 of Algorithm 1), we
bound the relevance score achieved by each subset compared
to the relevance score achieved by UCOSA. Taking the ratio of
ROPT/RUCOSA over all chunks yield the competitive ratio
result.
Before delving into the mathematical proof, we note the
following properties of the functionΨ(z i)used in UCOSA:
•Whenz i∈[0,1
1+ln(U/L)], the value ofΨ(z i)≤L.
Therefore, all prompts will be enriched with a chunk.
•Whenz i= 1, the value ofΨ(z i) =U. From the
algorithm, no additional prompts will be enriched, and
the budget is not violated.
•Ψ(z i)is a monotonically increasing function ofz i.
We now prove the competitive ratio of UCOSA.
1) Proof of Competitive Ratio:The proof is as follows.
Proof:Fix an input sequenceζ. LetOPT(ζ)andA(ζ)
be the total Relevance score obtained by the optimal offline
and UCOSA, respectively. Suppose that when the UCOSA
terminates, the fraction of budget consumed to use chunkj
isZj, and letB j=Z jB. LetS∗
j, Sjbe the set of prompts
enriched with chunkjby the optimal offline and the UCOSA,
respectively.
We partition the setS∗
jinto 3 subsets:
•The prompts enriched by both offline and UCOSA with
chunkj. Denote this byD∗
j=S∗
j∩Sj
•The prompts enriched by the offline with chunkj, but
not enriched by UCOSA. Denote this byX∗
j=S∗
j\Sj.
•The prompts enriched by the offline with chunkj, but
enriched with different chunk by UCOSA. Denote this
byY∗
j.For the prompts inD∗
j, since it was enriched by UCOSA,
thenR ij≥P ijΨ(zi),∀i∈D∗
j. The total Relevance score
v(D∗
j)of these prompts is:
v(D∗
j) =P
i∈D∗
jRij
≥P
i∈D∗
jPijΨ(zi)(4)
SinceΨ(z i)is monotonically increasing, The right hand side
of Equation (4):
X
i∈D∗
jPijΨ(zi)≤X
i∈D∗
jPijΨ(Z j)≤BΨ(Z j)(5)
For the prompts inX∗
j, since these were not enriched by
UCOSA, thenR ij≤PijΨ(zi),∀i∈X∗
j. Therefore:
X
i∈X∗
jRij≤P
i∈X∗
jPijΨ(zi)
≤P
i∈X∗
jPijΨ(Z j)
≤Ψ(Z j)(B−B j)(6)
For the prompts inY∗
j, UCOSA enriched the prompt with
a different chunk to achieve a higher score. Therefore:
voff(Y∗
j)≤v UCOSA (Y∗
j)≤v(S j),∀j
wherev off(Y∗
j)andv UCOSA (Y∗
j)are the sum of Relevance
scores of the setY∗
jachieved by the optimal offline and
UCOSA, respectively. Therefore:
X
jvoff(Y∗
j)≤X
jv(Sj) =A(ζ)(7)
There are also the prompts enriched by UCOSA with chunk
j, but not enriched by the offline algorithm (i.e.,the set
Sj\S∗
j). For these prompts, we have:
v(Sj\S∗
j) =X
i∈Sj\S∗
jRij≥X
i∈Sj\S∗
jPijΨ(zi)(8)
Note thatOPT(ζ) =P
j 
v(D∗
j) +v(X∗
j) +v off(Y∗
j)
,
andA(ζ) =P
j 
v(D∗
j) +v(S j\S∗
j)
. The competitive ratio
is:
OPT(ζ)
A(ζ)=P
j 
v(D∗
j)+v(X∗
j)+voff(Y∗
j)
P
j 
v(D∗
j)+v(S j\S∗
j)
≤P
j 
v(D∗
j)+v(X∗
j)
A(ζ)+A(ζ)
A(ζ)
≤P
jP
i∈D∗
j 
PijΨ(Zj)
+Ψ(Z j)(B−B j)
P
jP
i∈D∗
j 
PijΨ(Zj)
+v(S j\S∗
j)+ 1
≤P
j
Ψ(Zj)Bj+Ψ(Z j)(B−B j)
P
jP
i∈D∗
j 
PijΨ(Zj)
+v(S j\S∗
j)+ 1
≤P
jΨ(Zj)B
P
j P
i∈SjΨ(zi)Pij+ 1
≤P
jΨ(Zj)
P
j P
i∈SjΨ(zi)∆zi+ 1(9)
Where the first inequality is obtained by substituting Equa-
tion (7). The second inequality is obtained by substituting
Equation (6), the third inequality is obtained by substituting
Equation (5), and the last inequality is obtained by substituting
∆zi=Pij/B.

15
For the denominator of Equation (9), we approximate the
summation via an integration (since we already assumed that
Pij/B≤ϵ, whereϵ→0). Therefore, for every chunkjwe
have:
X
i∈SjΨ(zi)∆zi ≈RZj
0Ψ(zi)dzi
=Rg
0Ldzi+RZj
gΨ(zi)dzi
=gL+L
e(Ue/L)Zj−(Ue/L)g
ln(Ue/L)
=L
e(Ue/L)Zj
ln(Ue/L)
=Ψ(Zj)
ln(U/L)+1(10)
Whereg=1
1+ln(U/L).
Substituting Equation (10) in Equation (9), we have:
OPT(ζ)
A(ζ)≤P
jΨ(Zj)
Ψ(Zj)
ln(U/L)+1+ 1≤ln(U/L) + 2.
Since each of the ratiosΨ(Zj)P
i∈SjΨ(zi)∆ziis less than
1
ln(U/L)+1, the last inequality follows.
2) Lower Bound Proof:To prove the lower bound on the
competitive ratio, we provide an instance of the problem on
which no online algorithm can achieve a better competitive
ratio thanln(U/L) + 1.
UCOSA presented above is a deterministic algorithm (i.e.,
given the same input sequence, UCOSA will produce the exact
same output). To prove the competitive ratio, we use Yao’s
principle. Yao’s principle states that deterministic algorithms
performs worse than randomized algorithms against an obliv-
ious adversary (one who does not know the outcomes of the
random process in the randomized algorithm). Therefore, a
lower bound on the randomized algorithm is also a lower
bound on the deterministic algorithm.
For any input distributionFandγ-competitive randomized
algorithmA, we will show that:
1
γ≤min
σE[A(σ)]
OPT(σ)≤max
deterministicAEσ←FA(σ)
OPT(σ)
To prove the lower bound, we specify a distributionFsuch
that:
max
deterministicAEσ←FA(σ)
OPT(σ)
≤1
ln(U/L) + 1(11)
The instance of the problem is defined as follows:
•The instance has one chunk.
•Fix a parameterη >0. Letkbe the largest integer such
that(1 +η)k≤U/L. Therefore,k=⌊ln(U/L)
ln(1+η)⌋.
•The support of the input distributionFconsist of the
instancesI 0, I1, . . . , I k, whereI 0consists ofBidentical
prompts each with price 1 and Relevance score ofL, and
Im+1 consists ofI mfollowed by additionalBprompts
each with price 1 and a Relevance score(1 +η)m+1L.
This means thatI m⊂Im+1.
•The probabilities of the instancesI 0, I1, . . . , I kare
defined as follows:p k=1+η
(k+1)η+1, andp m=
η
(k+1)η+1,∀m∈ {0,1, . . . , k−1}. Note thatPk
m=0pm=
1.•DefineH= (k+ 1)η+ 1.
Proof:Letb mbe the fraction of the budget used for
prompts with profits(1 +η)mL,m= 0,1, . . . , k. We note
thatPk
m=0bm≤1since a proper algorithm will not use
more than the available budget. Now, we have:
Eσ←FA(σ)
OPT(σ)
=Pk
i=0piPi
m=0(1+η)jbm
(1+η)i
=Pk
m=0bmPk
i=mpi(1 +η)m−i
For anym:
kX
i=mpi(1 +η)m−i=pk(1 +η)m−k+Pk−1
i=mpi(1 +η)m−1
=(1+η)m−k+1
H+η
HPk−1
i=m(1 +η)m−i
=(1+η)m−k+1
H+η
H(1+η)−(1+η)m−k+1
η
=1+η
H=1+η
(k+1)η+1
where the second equality comes from substituting the
values ofp m, and the last equality comes from substituting
the value ofH.
Therefore, we get:
Eσ←FA(σ)
OPT(σ)
=1+η
(k+1)η+1Pk
m=0bm
≤1+η
(k+1)η+1
≤1+η
ηln(U/L)
ln(1+η)+1
where the first inequality comes fromPk
m=0bm≤1, and
the second inequality comes from substituting the value of
k=⌊ln(U/L)
ln(1+η)⌋.
By taking the limit asη→0, and noting that
limη→0η
ln(1+η)= 1, we complete the proof.
ACKNOWLEDGMENT
Research reported in this publication was supported by the
Qatar Research Development and Innovation Council grant #
ARG01-0525-230348. The content is solely the responsibility
of the authors and does not necessarily represent the official
views of Qatar Research Development and Innovation Council.
REFERENCES
[1] T. Brown, B. Mann, N. Ryder, M. Subbiah, J. D. Kaplan, P. Dhariwal,
A. Neelakantan, P. Shyam, G. Sastry, A. Askellet al., “Language mod-
els are few-shot learners,”Advances in neural information processing
systems, vol. 33, pp. 1877–1901, 2020.
[2] J. Achiam, S. Adler, S. Agarwal, L. Ahmad, I. Akkaya, F. L. Aleman,
D. Almeida, J. Altenschmidt, S. Altman, S. Anadkatet al., “Gpt-4
technical report,”arXiv preprint arXiv:2303.08774, 2023.
[3] M. Reid, N. Savinov, D. Teplyashin, D. Lepikhin, T. Lillicrap, J.-b.
Alayrac, R. Soricut, A. Lazaridou, O. Firat, J. Schrittwieseret al.,
“Gemini 1.5: Unlocking multimodal understanding across millions of
tokens of context,”arXiv preprint arXiv:2403.05530, 2024.
[4] A. Chowdhery, S. Narang, J. Devlin, M. Bosma, G. Mishra, A. Roberts,
P. Barham, H. W. Chung, C. Sutton, S. Gehrmannet al., “Palm: Scal-
ing language modeling with pathways,”Journal of Machine Learning
Research, vol. 24, no. 240, pp. 1–113, 2023.
[5] J. Yang, H. Jin, R. Tang, X. Han, Q. Feng, H. Jiang, S. Zhong, B. Yin,
and X. Hu, “Harnessing the power of llms in practice: A survey on
chatgpt and beyond,”ACM Transactions on Knowledge Discovery from
Data, vol. 18, no. 6, pp. 1–32, 2024.

16
[6] Z. Ji, N. Lee, R. Frieske, T. Yu, D. Su, Y . Xu, E. Ishii, Y . J. Bang,
A. Madotto, and P. Fung, “Survey of hallucination in natural language
generation,”ACM Computing Surveys, vol. 55, no. 12, pp. 1–38, 2023.
[7] L. Huang, W. Yu, W. Ma, W. Zhong, Z. Feng, H. Wang, Q. Chen,
W. Peng, X. Feng, B. Qinet al., “A survey on hallucination in large
language models: Principles, taxonomy, challenges, and open questions,”
arXiv preprint arXiv:2311.05232, 2023.
[8] M. Zhang, O. Press, W. Merrill, A. Liu, and N. A. Smith,
“How language model hallucinations can snowball,” inForty-first
International Conference on Machine Learning, 2024. [Online].
Available: https://openreview.net/forum?id=FPlaQyAGHu
[9] K. Guu, K. Lee, Z. Tung, P. Pasupat, and M. Chang, “Retrieval
augmented language model pre-training,” inInternational conference
on machine learning. PMLR, 2020, pp. 3929–3938.
[10] P. Lewis, E. Perez, A. Piktus, F. Petroni, V . Karpukhin, N. Goyal,
H. K ¨uttler, M. Lewis, W.-t. Yih, T. Rockt ¨aschelet al., “Retrieval-
augmented generation for knowledge-intensive nlp tasks,”Advances in
Neural Information Processing Systems, vol. 33, pp. 9459–9474, 2020.
[11] V . as-a Service. Vectara RAG-as-a-Service . Accessed: 2024-08-07.
[Online]. Available: https://vectara.com/retrieval-augmented-generation/
[12] L. Cloud. LlamaIndex Cloud . Accessed: 2024-08-07. [Online].
Available: https://docs.llamaindex.ai/en/stable/llama cloud/
[13] Nuclia, “RAG-as-a-Service,” 2024, [Online; accessed 01-June-2024].
[Online]. Available: https://nuclia.com/rag-as-a-service/
[14] Geniusee, “RAG-as-a-Service,” 2024, [Online; accessed
02-June-2024]. [Online]. Available: https://geniusee.com/
retrieval-augmented-generation
[15] Amazon Bedrock, “RAG-as-a-Service,” 2024, [Online; accessed 02-
June-2024]. [Online]. Available: https://aws.amazon.com/bedrock/
[16] P. Xu, W. Ping, X. Wu, L. McAfee, C. Zhu, Z. Liu, S. Subramanian,
E. Bakhturina, M. Shoeybi, and B. Catanzaro, “Retrieval meets long
context large language models,” inThe Twelfth International Conference
on Learning Representations.
[17] A. Roberts, C. Raffel, and N. Shazeer, “How much knowledge can
you pack into the parameters of a language model?”arXiv preprint
arXiv:2002.08910, 2020.
[18] J. Menick, M. Trebacz, V . Mikulik, J. Aslanides, F. Song, M. Chad-
wick, M. Glaese, S. Young, L. Campbell-Gillingham, G. Irvinget al.,
“Teaching language models to support answers with verified quotes,”
arXiv preprint arXiv:2203.11147, 2022.
[19] L. Kuhn, Y . Gal, and S. Farquhar, “Semantic uncertainty:
Linguistic invariances for uncertainty estimation in natural
language generation,” inThe Eleventh International Confer-
ence on Learning Representations, 2023. [Online]. Available:
https://openreview.net/forum?id=VD-AYtP0dve
[20] S. Farquhar, J. Kossen, L. Kuhn, and Y . Gal, “Detecting hallucinations
in large language models using semantic entropy,”Nature, vol. 630, no.
8017, pp. 625–630, 2024.
[21] J. Hron, L. Culp, G. Elsayed, R. Liu, B. Adlam, M. Bileschi, B. Bohnet,
J. Co-Reyes, N. Fiedel, C. D. Freemanet al., “Training language
models on the knowledge graph: Insights on hallucinations and their
detectability,”arXiv preprint arXiv:2408.07852, 2024.
[22] Y . Gao, Y . Xiong, X. Gao, K. Jia, J. Pan, Y . Bi, Y . Dai, J. Sun, and
H. Wang, “Retrieval-augmented generation for large language models:
A survey,”arXiv preprint arXiv:2312.10997, 2023.
[23] S. Barnett, S. Kurniawan, S. Thudumu, Z. Brannelly, and M. Abdelrazek,
“Seven failure points when engineering a retrieval augmented generation
system,” inProceedings of the IEEE/ACM 3rd International Conference
on AI Engineering-Software Engineering for AI, 2024, pp. 194–199.
[24] J. Chen, H. Lin, X. Han, and L. Sun, “Benchmarking large language
models in retrieval-augmented generation,” inProceedings of the AAAI
Conference on Artificial Intelligence, vol. 38, no. 16, 2024, pp. 17 754–
17 762.
[25] Vectara. Vectara Price Plan. Accessed: 2024-09-01. [Online]. Available:
https://vectara.com/pricing/
[26] C. LlamaIndex. Cloud LlamaIndex Price Plan. Accessed: 2024-09-01.
[Online]. Available: https://docs.cloud.llamaindex.ai/llamaparse/usage
data
[27] Nuclia. Nuclia Price Plan. Accessed: 2024-09-01. [Online]. Available:
https://nuclia.com/pricing/
[28] X. Tang, Q. Gao, J. Li, N. Du, Q. Li, and S. Xie, “Mba-rag: a bandit
approach for adaptive retrieval-augmented generation through question
complexity,” inProceedings of the 31st International Conference on
Computational Linguistics, 2025, pp. 3248–3254.
[29] A. Belgacem, “Dynamic resource allocation in cloud computing: anal-
ysis and taxonomies,”Computing, vol. 104, no. 3, pp. 681–710, 2022.[30] S. Ghasemi, M. R. Meybodi, M. D. T. Fooladi, and A. M. Rahmani,
“A cost-aware mechanism for optimized resource provisioning in cloud
computing,”Cluster Computing, vol. 21, no. 2, pp. 1381–1394, 2018.
[31] R. Younis, M. Iqbal, K. Munir, M. A. Javed, M. Haris, and S. Alahmari,
“A comprehensive analysis of cloud service models: Iaas, paas, and saas
in the context of emerging technologies and trend,” in2024 international
conference on electrical, communication and computer engineering
(ICECCE). IEEE, 2024, pp. 1–6.
[32] M. Kumar, S. C. Sharma, S. Goel, S. K. Mishra, and A. Husain, “Auto-
nomic cloud resource provisioning and scheduling using meta-heuristic
algorithm,”Neural Computing and Applications, vol. 32, no. 24, pp.
18 285–18 303, 2020.
[33] P.-Y . Lin and Y .-l. Tsai, “Scorerag: A retrieval-augmented generation
framework with consistency-relevance scoring and structured summa-
rization for news generation,”arXiv preprint arXiv:2506.03704, 2025.
[34] Y . Wang, R. Ren, J. Li, W. X. Zhao, J. Liu, and J.-R. Wen, “Rear: A
relevance-aware retrieval-augmented framework for open-domain ques-
tion answering,”arXiv preprint arXiv:2402.17497, 2024.
[35] Amazon. (2025) Aws pricing. [Online]. Available: https://aws.amazon.
com/pricing/
[36] Haystack. Haystack for Buidling RAG Piplelines. Accessed: 2025-08-
08. [Online]. Available: https://haystack.deepset.ai/overview/intro
[37] LlamaIndex. LlamaIndex Vector Store . Accessed: 2025-08-08. [Online].
Available: https://tinyurl.com/LlamaIndex-Vector-Store
[38] M. FAISS. META FAISS Vector Database. Accessed: 2025-08-08.
[Online]. Available: https://faiss.ai/index.html
[39] Pinecone. Pinecone Vector Database. Accessed: 2025-08-08. [Online].
Available: https://www.pinecone.io/
[40] Chroma. Chroma Vector Database. Accessed: 2025-08-08. [Online].
Available: https://docs.trychroma.com/
[41] E. Stack. Elastic Search Relevance Engine. Accessed: 2025-
08-08. [Online]. Available: https://www.elastic.co/elasticsearch/
elasticsearch-relevance-engine
[42] OpenZeppelin. OpenZeppelin Smart Contract. Accessed: 2025-
08-09. [Online]. Available: https://github.com/OpenZeppelin/
openzeppelin-contracts
[43] Consensys. Consensys Smart Contract. Accessed: 2025-08-09. [Online].
Available: https://github.com/Consensys
[44] Zuora. Zuora Revenue Sharing. Accessed: 2025-08-09. [Online].
Available: https://www.zuora.com/
[45] Chargebee. Chargebee Revenue Sharing. Accessed: 2025-08-09.
[Online]. Available: https://www.chargebee.com/
[46] Llamaindex. Lamaindex RAG Framework. Accessed: 2024-05-06.
[Online]. Available: https://docs.llamaindex.ai/en/stable/
[47] OpenAI. ChatGPT 3.5. Accessed: 2024-05-06. [Online]. Available:
https://www.chatgpt.com/
[48] LlamaIndex. LlamaIndex Embedding . Accessed: 2024-08-07. [Online].
Available: https://tinyurl.com/LlamaIndex-Embedding
[49] P. Sarthi, S. Abdullah, A. Tuli, S. Khanna, A. Goldie, and C. D.
Manning, “Raptor: Recursive abstractive processing for tree-organized
retrieval,” inThe Twelfth International Conference on Learning Repre-
sentations, 2024.
[50] Z. Li, C. Li, M. Zhang, Q. Mei, and M. Bendersky, “Retrieval augmented
generation or long-context llms? a comprehensive study and hybrid
approach,” inProceedings of the 2024 Conference on Empirical Methods
in Natural Language Processing: Industry Track, 2024, pp. 881–893.
[51] LlamaIndex. (2025) Llamaindex similarity metrics. [Online]. Available:
https://tinyurl.com/m4p9ucus
[52] ——. LlamaIndex Cosine Similarity . Accessed: 2025-09-09. [Online].
Available: https://tinyurl.com/LlamaIndex-Cosine-Similarity
[53] ScienceDirect. (2025) Llamaindex similarity metrics. [Online].
Available: https://www.sciencedirect.com/topics/computer-science/
cosine-similarity
[54] LlamaIndex. (2025) LlamaIndex Chunk Size Evaluation.
[Online]. Available: https://www.llamaindex.ai/blog/
evaluating-the-ideal-chunk-size-for-a-rag-system-using-llamaindex-6207e5d3fec5
[55] P.-Y . Chen and C.-J. Hsieh,Adversarial robustness for machine learning.
Academic Press, 2022.
[56] K. Warr,Strengthening deep neural networks: Making AI less susceptible
to adversarial trickery. O’Reilly Media, 2019.
[57] B. Christian and T. Griffiths,Algorithms to live by: The computer science
of human decisions. Macmillan, 2016.

17
Shawqi Al-Malikiis currently a postdoctoral re-
searcher at Hamad Bin Khalifa University (HBKU),
Doha, Qatar. He received his M.Sc. degree from
King Fahd University of Petroleum and Minerals
(KFUPM) in Dhahran, Saudi Arabia, in 2018, and
his Ph.D. degree in computer science from HBKU in
2023. His research interests span adversarial robust-
ness, LLMs security, privacy, reliability, and human-
LLM alignment. Prior to his academic career, Dr. Al-
Maliki held positions as a Full-Stack Web Developer
& Project Manager at Alyaum Media House (2009-
2019) and a computer instructor at New Horizon Institute (2006-2008), both
in Saudi Arabia.
Dr. Ammar Gharaibehis an Associate Professor
with the Computer Engineering department of Ger-
man Jordanian University, and currently serving as
the Acting Dean of the School of Computing at
GJU. He received his Ph.D. degree in Computer
Engineering from New Jersey Institute of Technol-
ogy, his M.S. degrees in Computer Engineering from
Texas A&M University in 2009, and his B.S. degree
with honors in Electrical Engineering from Jordan
University of Science & Technology in 2006. His re-
search interests span the areas of Online Algorithms,
Wireless Networks, and AI applications in Wireless Sensor Networks.
Mohamed Rahoutireceived his M.S. degree in
Statistics and Ph.D. degree in Electrical Engineering
from the University of South Florida, Tampa, FL,
USA, in 2016 and 2020, respectively. He is currently
an Assistant Professor with the Department of Com-
puter and Information Science, Fordham University,
New York, NY , USA. His research was supported by
the National Science Foundation (NSF), the Florida
Center for Cybersecurity, and the Qatar Research
Development and Innovation Council. His current
research focuses on computer networking and secu-
rity, blockchain technology, and artificial intelligence/machine learning.
Ruhul Aminis an Assistant Professor in the
Department of Computer and Information Science at
Fordham University. He earned his PhD in Computer
Science from Stony Brook University, where he
was advised by the distinguished Professor Steven
Skiena. Prior to his current role, Ruhul served as
a Graduate Research Assistant at the Institute of
Advanced Computational Science (IACS), where
he led multiple collaborative research initiatives in
Bioinformatics and Social Science. He also brings
industry experience from his time with the Amazon
AWS AI Algorithms group, where he developed deep learning-based algo-
rithms for anomaly detection.
Dr. Mohamed Abdallahis currently a Professor
and Associate Dean of Undergraduate Studies and
Quality Assurance at the College of Science and En-
gineering, Hamad Bin Khalifa University (HBKU).
He received his M.Sc. and Ph.D. degrees from the
University of Maryland at College Park, USA in
2001 and 2006, respectively. His research interests
lie in AI for communications, with a focus on 6G
wireless networks, wireless security, electric vehi-
cles, and smart grids. He has authored more than
220 journal and conference papers, contributed to
four book chapters, and co-invented four patents. Dr. Abdallah has received
several prestigious awards, including the Research Fellow Excellence Award
at Texas A&M University at Qatar in 2016, and Best Paper Awards at multiple
IEEE conferences, such as IEEE BlackSeaCom 2019 and the IEEE First
Workshop on Smart Grid and Renewable Energy in 2015. Additionally, he was
a recipient of the Nortel Networks Industrial Fellowship for five consecutive
years (1999–2003).
Junaid Qadiris a Professor of Computer En-
gineering at Qatar University in Doha, Qatar. His
core research interests lie in human-centered ar-
tificial intelligence, AI ethics, and the application
of AI in education and healthcare. His research is
widely published, with more than 250 peer-reviewed
articles, including in leading journals such as IEEE
Transactions on Pattern Analysis and Machine Intel-
ligence, IEEE Transactions on Artificial Intelligence,
and the Journal of Artificial Intelligence Research.
He has won a range of awards, including the IEEE
Engineering in Medicine and Biology Prize Paper Award (2023), the Emerald
Literati Outstanding Paper Award (2023), and the Qatar University College
of Engineering Dean’s Awards for Excellence in Research (2024) and Service
(2025). He has secured competitive research funding from Facebook Research,
the Qatar National Research Fund, and the Higher Education Commission
of Pakistan. He is an ACM Distinguished Speaker (2020–2026), an IEEE
Distinguished Lecturer (2025–2026) of the IEEE Education Society, a Senior
Member of IEEE since 2014, and a Senior Member of ACM since 2020.
Prof. Ala Al-Fuqahais the Associate Provost for
Teaching and Learning at Hamad Bin Khalifa Uni-
versity (HBKU). He is a professor at the College
of Science and Engineering, HBKU. Before HBKU,
he was Professor and director of the NEST research
lab at the Computer Science Department of Western
Michigan University (WMU). His research interests
include the use of machine learning in general and
deep learning in particular in support of the data-
driven and self-driven management of large-scale
deployments of IoT and smart city infrastructure and
services, and management and planning of software-defined networks (SDN).
He is a senior member of the IEEE, a senior member of the ACM, and an
ABET Program Evaluator (PEV) and commissioner. He served on editorial
boards of multiple journals, including IEEE Communications Letters, IEEE
Network Magazine, and Springer AJSE. He also served as chair, co-chair, and
technical program committee member of multiple international conferences,
including IEEE VTC, IEEE Globecom, IEEE ICC, and IWCMC.