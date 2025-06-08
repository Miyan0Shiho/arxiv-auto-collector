# DeepShop: A Benchmark for Deep Research Shopping Agents

**Authors**: Yougang Lyu, Xiaoyu Zhang, Lingyong Yan, Maarten de Rijke, Zhaochun Ren, Xiuying Chen

**Published**: 2025-06-03 13:08:17

**PDF URL**: [http://arxiv.org/pdf/2506.02839v1](http://arxiv.org/pdf/2506.02839v1)

## Abstract
Web agents for online shopping have shown great promise in automating user
interactions across e-commerce platforms. Benchmarks for assessing such agents
do not reflect the complexity of real-world shopping scenarios, as they often
consist of overly simple queries with deterministic paths, such as "Find iPhone
15." Real shopping scenarios are inherently more layered, involving
multi-dimensional product attributes, search filters, and user-specific sorting
preferences. To address this gap, we introduce DeepShop, a benchmark designed
to evaluate web agents in complex and realistic online shopping environments.
DeepShop comprises three key components. (1) Query diversity evolution:
Starting from real user queries, we generate diverse queries across five
popular online shopping domains. (2) Query complexity evolution: We further
evolve these queries to increase complexity, considering product attributes,
search filters, and sorting preferences, and classify them into three levels:
easy, medium, and hard, based on the number of evolutions. (3) Fine-grained and
holistic evaluation: We propose an automated evaluation framework that assesses
agent performance in terms of fine-grained aspects (product attributes, search
filters, and sorting preferences) and reports the overall success rate through
holistic evaluation. We conduct a systematic evaluation of retrieval-augmented
generation (RAG) methods, web agents, and deep research systems. Results show
that RAG struggles with complex queries due to its lack of web interaction,
while other methods face significant challenges with filters and sorting
preferences, leading to low overall success rates. We also perform
cross-category, complexity-based evaluations and error analyses to support the
advancement of deep research shopping agents.

## Full Text


<!-- PDF content starts -->

arXiv:2506.02839v1  [cs.IR]  3 Jun 2025DeepShop: A Benchmark for Deep Research
Shopping Agents
Yougang Lyu1Xiaoyu Zhang2Lingyong Yan3
Maarten de Rijke1Zhaochun Ren4∗Xiuying Chen5∗
1University of Amsterdam2Shandong University3Baidu Inc.
4Leiden University5Mohamed bin Zayed University of Artificial Intelligence (MBZUAI)
Abstract
Web agents for online shopping have shown great promise in automating user
interactions across e-commerce platforms. Benchmarks for assessing such agents
do not reflect the complexity of real-world shopping scenarios, as they often con-
sist of overly simple queries with deterministic paths, such as “Find iPhone 15.”
Real shopping scenarios are inherently more layered, involving multi-dimensional
product attributes, search filters, and user-specific sorting preferences. To address
this gap, we introduce DeepShop, a benchmark designed to evaluate web agents in
complex and realistic online shopping environments. DeepShop comprises three
key components. (1) Query diversity evolution: Starting from real user queries,
we generate diverse queries across five popular online shopping domains. (2)
Query complexity evolution: We further evolve these queries to increase com-
plexity, considering product attributes, search filters, and sorting preferences, and
classify them into three levels: easy, medium, and hard, based on the number of
evolutions. (3) Fine-grained and holistic evaluation: We propose an automated
evaluation framework that assesses agent performance in terms of fine-grained
aspects (product attributes, search filters, and sorting preferences) and reports the
overall success rate through holistic evaluation. We conduct a systematic evaluation
of retrieval-augmented generation (RAG) methods, web agents, and deep research
systems. Results show that RAG struggles with complex queries due to its lack
of web interaction, while other methods face significant challenges with filters
and sorting preferences, leading to low overall success rates. We also perform
cross-category, complexity-based evaluations and error analyses to support the
advancement of deep research shopping agents.
1 Introduction
Recent progress in web agents has enabled more complex automation of human interactions on
e-commerce platforms [ 17,44,51], largely driven by the integration of large language models
(LLMs) that provide planning, memory, and web interaction capabilities [ 11,44,60]. Despite these
advances, web agents still face significant challenges in completing complex user queries in dynamic,
real-world shopping environments [ 31,53]. These complex user queries require agents to perform
deep research of e-commerce platforms—browsing product listings, applying filters, and comparing
items—to accommodate diverse and nuanced user preferences [ 4,42]. This leads to our key research
question: Can existing web agents effectively fulfill diverse and complex user needs in realistic
shopping scenarios?
To evaluate web shopping agents, recent studies have proposed benchmarks that test their ability to
complete user tasks via simulated or real website interactions. Most offline benchmarks, such as
∗Corresponding author.
Preprint. Under review.

Find a queen-sized bedspread featuring a floral design in a calming blue hue        , ensuring it has a minimum customer rating of 4.5 stars from at least 300 reviews, and check for the lowest price         .[Webshop] Find me anti aging eyes care with green tea for dark circles, fine lines.[Mind2Web-Live] Add to my cart a women's T-shirt priced under 10 dollars in Uniqlo.[WebVoyager] Search an Xbox Wireless controller with green color and rated above 4 stars.
DeepShopBaseline QueryFeedbackAction
Fine-grained & Holistic Evaluation
Complex query basedon user needsOverly simplistic and lacks contextual details
EnvironmentAgentSuccessFailure
Figure 1: DeepShop evaluates agents on realistic and complex shopping queries with fine-grained,
holistic metrics, while existing benchmarks use overly simple queries lacking contextual depth.
Mind2Web [ 5], WebShop [ 51], and WebArena [ 62], are based on static environments constructed
from pre-collected web snapshots or manually curated HTML structures. While these benchmarks
enable controlled evaluations, they fail to capture the dynamic and unpredictable nature of real-world
websites, which often feature noisy, frequently updated, and interactive content [32, 46].
Recently, several online benchmarks have emerged, including Mind2Web-Live [ 34] and WebV oy-
ager [ 11], which enable agents to operate within real-time web environments. While recent efforts
mark progress, they still fail to capture the complexity and diversity of real-world shopping [ 43,50],
as most benchmark tasks remain simple and deterministic (e.g., “find an iPhone 15”), unlike real
queries that demand multi-attribute reasoning, filtering, and personalized sorting [31, 53].
To bridge this gap, we introduce DeepShop, a benchmark specifically designed to evaluate web agents
in complex online shopping scenarios. DeepShop is tailored to evaluate web shopping agents in
handling diverse and complex user queries, and includes a comprehensive evaluation framework. The
key components of DeepShop are as follows:
•Query diversity evolution : We begin with real-world user shopping queries and generate a wide
range of goals across five popular product categories (Books, Electronics, Home, Fashion and
Sports). This ensures that agents must generalize across varied user shopping intents.
•Query complexity evolution : We progressively enhance the complexity of each query by introduc-
ing combinations of product attributes (e.g., brand, color), search filters (e.g., ratings, availability),
and sorting preferences (e.g., lowest price first). These queries are categorized into easy,medium ,
andhard levels, based on the number and type of components involved.
•Fine-grained and holistic evaluation : To enable meaningful comparisons across agents, we
design an automated evaluation pipeline that assesses performance on three axes—correct attribute
matching, correct use of filters, and proper execution of sorting preferences—alongside a holistic
success rate measuring task completion.
We conduct both fine-grained and holistic evaluations of various approaches, including simple
retrieval-augmented generation ( RAG ) methods, advanced web agents, and commercial deep research
systems. Results show that RAG methods, lacking web interaction, struggle with DeepShop queries;
web agents fail to handle filters and sorting well; and even deep research systems fall short on filtering,
leading to low overall success rates. We further analyze performance across product categories, query
complexities, and error types to guide future research. By introducing a benchmark that mirrors
real-world complexity, DeepShop provides a rich testbed for advancing agent planning, adaptability,
and generalization, bridging the gap between academic systems and real-world deployment.
The contributions of this paper are as follows:
•We present DeepShop, a comprehensive benchmark for evaluating web agents in complex online
shopping scenarios, featuring diverse queries across five product categories and varying complexity
levels. Our dataset is built through a multi-stage process that evolves real-world shopping intents
by expanding query diversity and complexity.
•We conduct extensive experiments comparing simple RAG methods, advanced web agents, and
commercial deep research systems using our fine-grained and holistic evaluation framework.
•We provide detailed analyses across product categories, query complexity levels, and specific
error types, revealing critical limitations in current systems and offering insights to guide future
development of more effective deep research shopping agents.
2

Table 1: Comparison of existing benchmarks and DeepShop. DeepShop is evaluated online across
diverse product categories, providing fine-grained assessment over product attributes, search filters,
and sorting preferences. Average token length is computed from 100 randomly sampled queries.
BenchmarkAvg. query
lengthEnv.
typeProduct
categoryProduct
attributeSearch
filterSorting
preferenceTask
success
Webshop [51] 18.2 Offline ✓ ✓ ✗ ✗ ✓
Mind2Web [5] 13.4 Offline ✗ ✗ ✗ ✗ ✓
Webarena [62] 19.9 Offline ✗ ✗ ✗ ✗ ✓
VWebarena [18] 21.4 Offline ✗ ✗ ✗ ✗ ✓
MMInA [58] 24.2 Offline ✗ ✗ ✗ ✗ ✓
ChatShop [3] 20.4 Offline ✗ ✗ ✗ ✗ ✓
WebLINX [21] 6.2 Online ✗ ✗ ✗ ✗ ✓
Mind2Web-Live [34] 16.2 Online ✗ ✗ ✗ ✗ ✓
WebV oyager [11] 29.5 Online ✗ ✗ ✗ ✗ ✓
DeepShop (Ours) 62.0 Online ✓ ✓ ✓ ✓ ✓
2 Related Work
Benchmarks for web agent evaluation. Existing benchmarks for web agent evaluation fall into
two categories: offline and online, as shown in Table 1. Offline benchmarks (e.g., Mind2Web [ 5],
WebShop [ 51], WebArena [ 62], ChatShop [ 3]) use static snapshots or simulated environments,
offering controlled conditions but failing to capture the dynamic nature of real-world websites [ 14,18].
In contrast, online benchmarks (e.g., WebV oyager [ 11], Mind2Web-Live [ 34]) provide realistic real-
time settings but focus mainly on general and relatively simple tasks, leaving complex web shopping
queries underexplored. Even basic RAG systems with LLMs and Google Search can perform strongly
on many current benchmarks [ 28,52]. To address this gap, DeepShop introduces a benchmark
targeting challenging online web shopping queries, constructed by query diversity and complexity
evolution. We also propose fine-grained evaluation metrics across product attributes, search filters,
and sorting preferences to offer a comprehensive assessment of agent performance.
Web agents for task automation. Recent progress in web agents has followed a clear trajectory,
evolving from text-based to multimodal systems. Early HTML-based agents, such as WebGPT [ 30],
MindAct [ 5], and Agent-E [ 1], leverage LLMs to interpret natural language instructions and navigate
web interfaces using DOM trees [ 10,19]. Building on this, multimodal, vision-based agents like
SeeAct [ 60], WebV oyager [ 11], and Browser Use [ 29] integrate visual grounding to better handle com-
plex layouts and interactive components [ 6,36]. Recent systems such as OpenAI Deep Research [ 33]
and Gemini Deep Research [ 8] use advanced reasoning LLMs to tackle complex information-seeking
tasks. Despite these advances, most evaluations remain limited to generic benchmarks, leaving agent
performance on complex, real-world shopping scenarios underexplored. In this paper, we evaluate
simple RAG methods, text-based and multimodal web agents, and deep research systems in realistic
online shopping environments.
Query understanding in E-commerce. Online shopping platforms have become central to modern
consumer behavior, making accurate query understanding critical for satisfying user experiences [ 13,
35,54]. However, many e-commerce queries involve overwhelming product spaces and complex user
preferences that are difficult to express with simple keywords or filters [ 42]. Traditional information
retrieval (IR) systems often struggle with such complexity [ 4,45], while conversational IR systems,
despite supporting multi-turn preference elicitation, remain constrained by training products and
cannot autonomously browse web content [ 3,57]. Recent advances in web agents offer a promising
alternative by autonomously interacting with e-commerce sites, searching for relevant items, and
mimicking human browsing behaviors [ 11,51]. To advance this line of research, we introduce
DeepShop, a benchmark designed to evaluate web agents on complex e-commerce queries and drive
progress in web shopping automation.
3 DeepShop Benchmark
In this section, we present the DeepShop benchmark, a framework for evaluating web agents in
realistic online shopping environments for complex user queries. We begin by formulating the web
3

Find iPhone 15.Find the latest Apple iPhone 15, specifically the 256GB model in starlight color. Find Apple iPhone 15, ensuring it has a minimum customer rating of 4 stars. Find the latest Apple iPhone 15, rank with as a lowest price. Find yoga mats with a minimum customer rating of 4.5 stars. Find Apple iPhone 15, specifically the 128GB model in black, sorted by the lowest price available. 
Find yoga mats.Find Apple iPhone 15, ensuring it has a price ranking as the lowest among optionswith at least 4.5 stars, over 300 customer reviews, and offers free delivery.Find yoga mats withthe highest user rating.Find eco-friendly yoga mats, such as those made from natural rubber with a minimum customer rating of 4.5 stars.Find yoga mats with at least a 4.5-star rating, over 500 reviews.Find for the Apple iPhone 15, specifically the 128GB model in black, sorted by the lowest price, with customer ratings above 4.5 stars and at least 1000 reviews, ensuring free delivery and return options.
Attribute EvolutionFilter EvolutionSorting EvolutionFilter EvolutionFilterEvolution
Attribute EvolutionFilter EvolutionSorting EvolutionFilterevolutionAttributeevolution
Diversity EvolutionFigure 2: Running examples of diversity and complexity evolution in DeepShop. Complexity
evolution includes attribute evolution, filter evolution, and sorting evolution.
shopping tasks. Next, we describe the processes of query diversity and complexity evolution, which
are derived from real user seed queries, as shown in Figure 2. We then analyze key characteristics
of DeepShop to offer a deeper understanding of the dataset. Finally, we introduce a comprehensive
evaluation framework that incorporates both fine-grained and holistic metrics.
3.1 Task Formulation
Following previous work [ 11,34], we formulate the online web shopping task as a partially observable
Markov decision process (POMDP) [ 15] defined by a tuple (S,O,A,T), where Sdenotes the state
space, Othe observation space, Athe action space, and transition function T:S × A → S . In this
setting, at each time step t, given a user query q, the web shopping agent receives an observation
ot∈ O that partially reflects the underlying state st∈ S of the web environment. The agent then
takes an action at∈ A , resulting in a new environment state st+1∼ T(st, at)and an updated
observation ot+1∈ O.
3.2 Seed Data Curation
To evaluate web agents under realistic user shopping intentions, we curate a seed dataset by selecting
a subset of web shopping queries from two real-world benchmarks: Mind2Web-Live [ 34] and
WebV oyager [ 11]. Specifically, we manually select 50 user queries and categorize them into five
representative shopping domains: Books (4),Electronics (14), Home (20), Fashion (5), and Sports
(7). We define each domain as follows:
•Books dbooks: Physical books, eBooks, and audiobooks across various genres.
•Electronics delectronics : Electronics and digital products such as smartphones, tablets, laptops,
headphones, and smart devices.
•Home dhome: Household items including furniture, appliances, cleaning tools, and daily necessities.
•Fashion dfashion : Apparel, footwear, and accessories for all genders and age groups.
•Sports dsports : Fitness and recreational equipment, sportswear, and training accessories.
3.3 Shopping Query Diversity Evolution
Existing web shopping datasets [ 5,11,14,18,34,51] often overlook fine-grained product categories,
limiting their overall diversity. To address this limitation, inspired by [ 49], we generate entirely new
queries based on the original query and a randomly selected product category through the following
diversity evolution process:
q∗
i= Diversity( qi, d), (1)
4

where Diversity( ·)is implemented by prompting GPT4-o models, qi∈ D original is a seed query, and
d∈ {dbooks, delectronics , dhome, dfashion, dsports}denotes a randomly selected product category. Finally,
we construct the web shopping diversity evolution dataset Ddiversity by combining the seed dataset
with all generated queries: Ddiversity =Doriginal∪ {q∗
i}N
i=1, where Nis the number of seed queries.
3.4 Shopping Query Complexity Evolution
To increase the complexity of web shopping queries, we propose a web shopping complexity evolution
strategy. Specifically, we focus on three key augmentation directions: (1) Product attributes , referring
to concrete product characteristics users may specify to express detailed intent; (2) Search filters ,
representing categorical or numerical constraints commonly used on e-commerce platforms; and (3)
Sorting preferences , indicating desired result orderings, such as price or popularity. Specifically, we
perform iterative complexity evolution to progressively enhance query complexity. In each iteration t,
one of the three strategies is randomly selected to evolve the query qi,tfrom the previous step:
qi,t+1= Complexity( qi,t, c), (2)
where Complexity( ·)is implemented by prompting GPT4-o, qi,tdenotes the i-th query in t-th
complexity evolution, i∈[1,|Ddiversity|],t∈[1, T],qi,0denotes the i-th query from Ddiversity , and
c∈ {cattr, cfilter, csort}is the randomly selected strategy. The three complexity evolution strategies are
summarized as follows:
•Attribute evolution cattr: Enhance the query by incorporating concrete product attributes, such as
brand ,model ,price range ,color ,size,weight , or unique features of products.
•Filter evolution cfilter: Enhance the query by adding specific search filter commonly available on
e-commerce platforms. These include constraints like minimum customer rating (e.g., 4.5 stars),
minimum number of reviews (e.g., 500+), shipping options (e.g., free delivery), release timeframe
(e.g., new arrivals in the past 30 days), return policies , orwarranty information .
•Sorting evolution csort: Enhance the query by appending a sorting preference, directing the system
to find top-ranked products according to criteria like lowest price ,highest user rating ,newest
arrival , orbest seller ranking .
By iteratively applying the above strategies, our method mimics the natural evolution of user queries,
generating a hierarchical set of increasingly complex queries. Starting from diverse queries in
Ddiversity , we apply T= 5rounds of complexity evolution, resulting in a total of 600 queries.
3.5 Dataset Analysis
Analysis of query diversity evolution . Existing benchmarks for online web shop-
ping often exhibit skewed distributions across product categories, introducing evalua-
tion bias and limiting the generalizability of agent performance, as shown in Figure 3.
Books Electronics Home Fashion Sports05101520253035CountSeed data
DeepShop
Figure 3: Product category distribution
after query diversity evolution.To mitigate this, we construct a balanced subset of 150
queries from our 600-query pool, systematically selecting
30 queries each from five major categories: Books, Elec-
tronics, Home, Fashion, and Sports. Following previous
work [ 11,12], we manually verify each generated task
and, if necessary, revise it to ensure high quality and con-
firm that the answers are available on the corresponding
website. While this uniform category distribution does
not necessarily reflect real-world query frequency, it pro-
vides a controlled and equitable test bed for evaluating
cross-domain generalization. The DeepShop benchmark
significantly reduces the category imbalance present in the
seed data, enabling more controlled and equitable compar-
isons. This balanced design helps isolate category-related
performance effects, offering a clearer assessment of an
agent’s ability to generalize beyond narrow domain specialization.
Analysis of query complexity evolution . The complexity evolution strategy progressively enhances
query complexity by incorporating additional product attributes, search filters, or sorting preferences.
We perform a fine-grained analysis of query evolution across these three dimensions. Regarding
5

Iter 1 Iter 2 Iter 3 Iter 4 Iter 5
Iterations2.42.62.83.0Count
Iterative evolution
DeepShop hard
DeepShop
Seed data(a) Product attribute evolution.
Iter 1 Iter 2 Iter 3 Iter 4 Iter 5
Iterations1.01.52.02.53.03.5Count
Iterative evolution
DeepShop hard
DeepShop
Seed data (b) Search filter evolution.
Iter 1 Iter 2 Iter 3 Iter 4 Iter 5
Iterations0.30.40.50.60.70.80.9Count
Iterative evolution
DeepShop hard
DeepShop
Seed data (c) Sorting preference evolution.
Figure 4: Analysis of query complexity evolution.
product attributes, as depicted in Figure 4(a), the average number of product attributes per query
exhibits a steady increase throughout the iterations. Ultimately, the average number of product
attributes in DeepShop surpasses the seed data by 0.52, while the hard subset contains an additional
0.66 attributes on average. In terms of search filters, Figure 4(b) illustrates a consistent increase in
the average number of filters per query across iterations. At the final iteration, DeepShop queries
include, on average, 1.95 more filters than the seed queries, with the hard subset further increasing
this difference to 2.88 filters on average. Similarly, the evolution of sorting preferences, illustrated in
Figure 4(c), shows an upward trajectory. The final average sorting preferences per query exceed the
seed data by 0.37, and this increment is further pronounced in the hard subset, where queries contain
an additional 0.66 sorting preferences on average.
3.6 Evaluation Metrics
To comprehensively evaluate web agents within the DeepShop environment, we adopt a two-stage
evaluation protocol that includes both fine-grained evaluation and holistic task success evaluation.
Fine-grained evaluation. Given the cost and scalability challenges of human evaluation, following
previous work [ 11,50], we use GPT-4o for automatic evaluation. We first decompose each query
into product attribute qattr, search filter qfilterand sorting preference qsortsubqueries. For each web
agent trajectory, we prompt GPT-4o to assess whether the final results align with the requirements
specified in each subquery. Specifically, we prompt GPT-4o with the user subquery, screenshots, and
the final answer of web agents and prompt GPT-4o to provide a binary decision (“Success” or “Not
Success”) for each subquery. This fine-grained evaluation enables us to capture partial success cases
and diagnose failure modes more precisely than holistic binary task success alone. Note that if a
particular subquery is not present in the original query (i.e., None), we skip the evaluation for that
aspect and do not include it in the calculation.
Holistic evaluation. To calculate the overall task success, we rely on the above fine-grained evaluation
outcomes, specifically the success scores for product attribute, search filter, and sorting preference.
The holistic evaluation aggregates these components by rule-based checking, for each dimension,
whether the query explicitly specifies a requirement. If a particular aspect (e.g., attribute, filter,
or sorting) is present in the query, its corresponding success score is considered; otherwise, the
system treats it as automatically satisfied. The final holistic task success is determined only if all
required components are successfully satisfied — meaning that the system must meet all attribute,
filter, and sorting requirements that are explicitly part of the query. For deep research systems, since
intermediate execution screenshots are unavailable, both fine-grained and holistic evaluations are
conducted manually.
Agreement rate between LLM evaluation and human judge. Following previous work [ 11,50],
we calculate the agreement of human judgments and GPT-4o judgments to measure the reliability of
GPT-4o evaluations. Specifically, human annotators are shown the full interaction trace of the agent,
including screenshots and actions, and are asked to judge whether the agent successfully fulfilled the
user’s request. Finally, the agreement rates between human judges and GPT-4o judges for the product
attributes, search filters and sorting preferences, and overall task success are 84%, 80%, 82%, and
86%, respectively. It indicates the effectiveness and reliability of GPT-4o evaluation.
More details of the evaluation are provided in Appendix A.
6

Table 2: Main results across product attributes, search filters, sorting preferences, and overall task
success rates. Underlined indicates the best performance in web agents, and bold highlights the best
performance in deep research systems.
Method Product attribute Search filter Sorting preference Task success
Simple RAG
GPT-4o + Google Search 7.33 5.97 4.55 7.33
Web agents
Agent-E 12.67 9.70 3.41 6.67
SeeAct 52.00 22.39 20.45 10.67
WebV oyager 40.67 38.00 23.86 16.00
Browser Use 36.00 34.33 30.68 32.00
Deep research systems
Gemini Deep Research 53.33 44.00 52.94 30.00
OpenAI Deep Research 60.00 46.15 58.82 30.00
4 Experiments
4.1 Research Questions
We aim to answer the following research questions in our experiments: RQ1 : How do simple RAG
methods, web agents and deep research systems perform on the DeepShop benchmark in terms of
fine-grained and holistic evaluation metrics? RQ2 : How do existing methods perform across different
product categories (Books, Electronics, Home, Fashion and Sports) in online shopping tasks? RQ3 :
How does the performance of web agents vary across different levels of query complexity, from seed
queries to evolved complex queries with multiple attributes, filters and sorting preferences?
4.2 Baselines
We evaluate web agents against three baseline categories:
•Simple RAG : Combines GPT-4o with Google Search by submitting the query, retrieving the
top-ranked page, and generating a response based on webpage screenshots.
•Web agents :Agent-E [1] uses a hierarchical planner-actor framework with DOM tree distillation.
SeeAct [60] and WebVoyager [11] use LLM multimodality, combining visual perception with
action planning. Browser Use [29] integrates visual understanding and HTML extraction for robust
interaction.
•Deep research systems :Gemini Deep Research [8] decomposes queries and generates cited
multi-step reports using Gemini’s extended reasoning. OpenAI Deep Research [33] autonomously
browses, analyzes, and synthesizes web information into citation-rich outputs, emulating human
research workflows.
More details on the baselines and implementation are provided in Appendix B and Appendix C.
5 Experimental Results and Analysis
5.1 Performance Analysis of Web Agents (RQ1)
We present the experimental results for the simple RAG baseline, web agents and deep research
systems in Table 2. For each baseline, we evaluate the success rate of three fine-grained aspects,
product attributes, search filters, and sorting preferences, as well as the holistic task success rate.
Based on these results, we have three main observations:
•Simple RAG methods fail to solve DeepShop due to the lack of website interaction capabili-
ties. We observe that simple RAG methods perform poorly across both fine-grained and holistic
evaluations, with all success rates below 8%. In particular, these methods struggle with search
filters (score: 5.97) and sorting preferences (score: 4.55), as such requirements cannot be satisfied
through retrieval alone but instead demand active interaction with website elements (e.g., buttons).
7

Simple
RAGAgent-E SeeAct WebVoyager Browser
UseGemini
Deep
ResearchOpenAI
Deep
Research0102030405060T ask Success RateBooks
Electronics
Home
Fashion
Sports(a) Performance across different product categories.
Simple
RAGAgent-E SeeAct WebVoyager Browser
UseGemini
Deep
ResearchOpenAI
Deep
Research0102030405060T ask Success RateEasy
Medium
Hard (b) Performance across query complexity evolution.
Figure 5: Detailed analysis of performance across different product categories and query complexity.
This highlights the inherent complexity of DeepShop queries and underscores the need for agents
capable of dynamic web interaction.
•Web agents outperform simple RAG by using website interaction, but still struggle with
DeepShop’s fine-grained requirements. Web agents dynamically interact with site content,
enabling more effective product discovery than simple RAG. We observe progressive gains in
overall task success: from HTML-based Agent-E (score: 6.67) to vision-based SeeAct (score:
10.67) and WebV oyager (score: 16.00), peaking with Browser Use (score: 32.00), which integrates
HTML and visual inputs. Notably, SeeAct excels in product attributes, WebV oyager in search filters,
and Browser Use in sorting preferences. However, satisfying all three dimensions simultaneously
remains difficult, underscoring the challenge DeepShop poses for web agents.
•Deep research systems use multi-step reasoning to enhance fine-grained performance on
DeepShop, but overall success remains limited. Gemini and OpenAI deep research systems
excel in product attribute and sorting preference evaluations, outperforming web agents on these
aspects. They struggle with search filters, as many require deep exploration and confirmation on
product detail pages. Despite achieving higher fine-grained success, their holistic task success
rates (30% each) remain low, underscoring the difficulty of satisfying all DeepShop requirements
simultaneously and highlighting the benchmark’s challenge for powerful deep research systems.
5.2 Performance across Different Product Categories (RQ2)
We conduct a detailed analysis to evaluate the performance across different product categories, as
shown in Figure 5(a). Agent performance varies notably across product categories. The simple
RAG method performs relatively well in Home , benefiting from rich textual titles retrievable via
Google Search, but drops to 0% success in Fashion andSports , where visual cues dominate. HTML-
based Agent-E consistently underperforms, particularly in Sports , due to its inability to process visual
content. Vision-based agents like SeeAct and WebV oyager improve performance across domains,
while the hybrid Browser Use achieves the best cross-domain results by combining HTML and
vision inputs. Deep research systems show relatively stable trends across categories but face major
challenges in Fashion andSports , where Gemini scores 0% in Sports and OpenAI fails entirely in
both. These failures highlight the need for robust multimodal reasoning to handle visually driven
product categories effectively.
5.3 Performance across Query Complexity Evolution (RQ3)
We analyze baseline performance across increasing query complexity, as shown in Figure 5(b). Our
results reveal a clear negative correlation between query complexity and agent performance.
Tasks are grouped into easy (0–1 complexity evolution), medium (2–3), and hard (4–5). The simple
RAG method achieves success rates of 16% on easy and 6.00% on medium queries but drops to 0%
on hard tasks, showing that Google Search alone cannot handle complex user needs. Web agents
also exhibit sharp declines, with average accuracy falling from 28.5% (easy) to 17% (medium), then
further dropping by 7 percentage points on hard tasks. Notably, deep research systems perform better
than web agents on the hard subset, highlighting the importance of strong reasoning capabilities—yet
even the top-performing OpenAI system reaches only 20% success rate.
8

5.4 Error Analysis and Future Improvement Guidance
We conduct a detailed error analysis to identify the primary failure modes of web agents and deep
research systems during task execution. Understanding these issues is critical for designing more
robust and effective shopping agents. We categorize the observed errors as follows:
•Web agents are limited by grounding ability. HTML content and webpage screenshots provide
complementary signals. Agents relying solely on HTML often overlook visual details—such as
product color or layout cues—that are crucial for correct decisions. Conversely, vision-based
agents using set-of-mark prompts struggle with segmentation accuracy: interactive buttons are
frequently misclassified, and regions like customer reviews remain unsegmented, preventing the
use of rating filters. Additionally, small filtering and sorting widgets are often ignored, degrading
task performance. Future work may explore multimodal fusion techniques that combine HTML
structure with visual context to enable stronger grounding [9].
•Web agents often lack state assessment and replanning capabilities. Agents frequently issue
overly specific search queries and, upon retrieval failure, fail to backtrack or reformulate broader
alternatives. Similarly, after navigating to product detail pages and finding unmet requirements,
they rarely reconsider or explore other options. This lack of dynamic replanning leads to suboptimal
decisions. Moreover, due to limited awareness of webpage state transitions, agents tend to repeat
ineffective actions, such as clicking the same unresponsive element multiple times, instead of
adjusting their strategy. Future research could fine-tune agents on realistic web environments to
enhance their ability to reason over search failures, and adapt plans dynamically [20].
•Web agents are constrained by a limited action space. Web agents operate within a restricted
set of browser actions, which prevents interaction with dynamic UI components commonly found
on shopping platforms. For instance, a web agent fails to filter products within a specific range
because it cannot drag the price slider. More generally, agents struggle to operate dropdowns,
sliders, and nested menus—actions that are essential for precise filtering and sorting. Future work
could expand the agent’s action repertoire with shopping-specific operations and deeper browser
integration [50].
•Web agents lack the ability to learn from execution. Current agents show little ability to
generalize across tasks. Experiences gained during one interaction—such as which strategies led
to success or failure—are not transferred to future scenarios. As a result, agents repeatedly make
the same mistakes and fail to exploit previously effective strategies. Enabling execution-time
learning and memory would allow agents to abstract successful patterns, track failure cases, and
refine their decision-making over time. Future research may explore task-level memory modules,
outcome-based self-reflection, and lifelong learning mechanisms [48, 61].
•Deep research systems are prone to hallucination errors. OpenAI’s deep research systems
often oversimplify complex queries, neglecting constraints and returning confident yet inaccurate
recommendations. For instance, they may assert that a matching product exists even when it does
not. Although Gemini more frequently acknowledges failure and suggests approximate alternatives,
both systems frequently return incomplete or incorrect links—redirecting to irrelevant websites
or generic navigation pages rather than specific product detail views. These hallucinations reduce
trust and usability. Future work could apply preference alignment and fact-checking techniques to
reduce hallucination rates and improve the precision of retrieved links [43].
More details of our error analysis are provided in Appendix D.
6 Conclusions
In this paper, we introduce DeepShop, a benchmark aimed at evaluating web agents in realistic
and complex online shopping environments. While existing benchmarks often rely on simplistic
and deterministic queries, DeepShop bridges this gap by incorporating real-world user intents and
progressively evolving both the diversity and complexity of queries. Our benchmark covers five
major e-commerce domains and evaluates agent performance across key dimensions including
product attributes, search filters, and sorting preferences. To enable a comprehensive assessment,
we propose a fine-grained and holistic evaluation framework. Experimental results on recent web
agents reveal significant performance drops on complex queries, highlighting the need for more
robust agent design. Overall, DeepShop provides a challenging and realistic testbed for advancing
the development of intelligent, user-centered web shopping agents. The code is available at https:
//anonymous.4open.science/r/DeepShop-E4DF .
9

Acknowledgments
This work was supported by the Natural Science Foundation of China (62272274, 62372275,
62102234, 62202271, 62072279), the National Key R&D Program of China with grant
No.2022YFC3303004, the Natural Science Foundation of Shandong Province (ZR2021QF129),
the Dutch Research Council (NWO), under project numbers 024.004.022, NWA.1389.20.183, and
KICH3.LTP.20.006, and the European Union’s Horizon Europe program under grant agreement No
101070212. All content represents the opinion of the authors, which is not necessarily shared or
endorsed by their respective employers and/or sponsors.
References
[1]Abuelsaad, T., Akkil, D., Dey, P., Jagmohan, A., Vempaty, A., and Kokku, R. Agent-e: From
autonomous web navigation to foundational design principles in agentic systems. arXiv preprint
arXiv:2407.13032 , 2024.
[2]Cai, H., Li, Y ., Wang, W., Zhu, F., Shen, X., Li, W., and Chua, T. Large language models
empowered personalized web agents. CoRR , abs/2410.17236, 2024.
[3]Chen, S., Wiseman, S., and Dhingra, B. Chatshop: Interactive information seeking with
language agents. arXiv preprint arXiv:2404.09911 , 2024.
[4]Chen, Z., Choi, J. I., Fetahu, B., and Malmasi, S. Identifying high consideration e-commerce
search queries. In Proceedings of the 2024 Conference on Empirical Methods in Natural
Language Processing: EMNLP 2024 - Industry Track, Miami, Florida, USA, November 12-16,
2024 , pp. 563–572. Association for Computational Linguistics, 2024.
[5]Deng, X., Gu, Y ., Zheng, B., Chen, S., Stevens, S., Wang, B., Sun, H., and Su, Y . Mind2web:
Towards a generalist agent for the web. In Advances in Neural Information Processing Systems
36: Annual Conference on Neural Information Processing Systems 2023, NeurIPS 2023, New
Orleans, LA, USA, December 10 - 16, 2023 , 2023.
[6]Furuta, H., Lee, K., Nachum, O., Matsuo, Y ., Faust, A., Gu, S. S., and Gur, I. Multimodal
web navigation with instruction-finetuned foundation models. In The Twelfth International
Conference on Learning Representations, ICLR 2024, Vienna, Austria, May 7-11, 2024 . Open-
Review.net, 2024.
[7]Gao, S., Shi, Z., Zhu, M., Fang, B., Xin, X., Ren, P., Chen, Z., Ma, J., and Ren, Z. Confu-
cius: Iterative tool learning from introspection feedback by easy-to-difficult curriculum. In
Proceedings of the AAAI Conference on Artificial Intelligence , volume 38, pp. 18030–18038,
2024.
[8]Gemini. Try deep research and Gemini 2.0 flash experimental. Gemini Blog , 2025. URL
https://blog.google/products/gemini/google-gemini-deep-research/ .
[9]Gou, B., Wang, R., Zheng, B., Xie, Y ., Chang, C., Shu, Y ., Sun, H., and Su, Y . Navigating
the digital world as humans do: Universal visual grounding for gui agents. arXiv preprint
arXiv:2410.05243 , 2024.
[10] Gur, I., Furuta, H., Huang, A. V ., Safdari, M., Matsuo, Y ., Eck, D., and Faust, A. A real-world
webagent with planning, long context understanding, and program synthesis. In The Twelfth
International Conference on Learning Representations, ICLR 2024, Vienna, Austria, May 7-11,
2024 . OpenReview.net, 2024.
[11] He, H., Yao, W., Ma, K., Yu, W., Dai, Y ., Zhang, H., Lan, Z., and Yu, D. Webvoyager: Building
an end-to-end web agent with large multimodal models. arXiv preprint arXiv:2401.13919 ,
2024.
[12] He, H., Yao, W., Ma, K., Yu, W., Zhang, H., Fang, T., Lan, Z., and Yu, D. OpenWebV oyager:
Building multimodal web agents via iterative real-world exploration, feedback and optimization.
arXiv preprint arXiv:2410.19609 , 2024.
10

[13] Hirsch, S., Guy, I., Nus, A., Dagan, A., and Kurland, O. Query reformulation in e-commerce
search. In Proceedings of the 43rd International ACM SIGIR conference on research and
development in Information Retrieval, SIGIR 2020, Virtual Event, China, July 25-30, 2020 , pp.
1319–1328. ACM, 2020.
[14] Jang, L., Li, Y ., Ding, C., Lin, J., Liang, P. P., Zhao, D., Bonatti, R., and Koishida, K. Video-
WebArena: Evaluating long context multimodal agents with video understanding web tasks.
arXiv preprint arXiv:2410.19100 , 2024.
[15] Kaelbling, L. P., Littman, M. L., and Cassandra, A. R. Planning and acting in partially observable
stochastic domains. Artif. Intell. , 101(1-2):99–134, 1998.
[16] Kanoulas, E., Eustratiadis, P., Li, Y ., Lyu, Y ., Pal, V ., Poerwawinata, G., Qiao, J., and Wang, Z.
Agent-centric information access. arXiv preprint arXiv:2502.19298 , 2025.
[17] Kim, G., Baldi, P., and McAleer, S. Language models can solve computer tasks. In Advances
in Neural Information Processing Systems 36: Annual Conference on Neural Information
Processing Systems 2023, NeurIPS 2023, New Orleans, LA, USA, December 10 - 16, 2023 ,
2023.
[18] Koh, J. Y ., Lo, R., Jang, L., Duvvur, V ., Lim, M. C., Huang, P., Neubig, G., Zhou, S., Salakhut-
dinov, R., and Fried, D. VisualWebArena: Evaluating multimodal agents on realistic visual
web tasks. In Proceedings of the 62nd Annual Meeting of the Association for Computational
Linguistics (Volume 1: Long Papers), ACL 2024, Bangkok, Thailand, August 11-16, 2024 , pp.
881–905.
[19] Lai, H., Liu, X., Iong, I. L., Yao, S., Chen, Y ., Shen, P., Yu, H., Zhang, H., Zhang, X., Dong,
Y ., and Tang, J. AutoWebGLM: A large language model-based web navigating agent. In
Proceedings of the 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining,
KDD 2024, Barcelona, Spain, August 25-29, 2024 , pp. 5295–5306. ACM, 2024.
[20] Liu, Y ., Li, P., Xie, C., Hu, X., Han, X., Zhang, S., Yang, H., and Wu, F. InfiGUI-R1:
Advancing multimodal gui agents from reactive actors to deliberative reasoners. arXiv preprint
arXiv:2504.14239 , 2025.
[21] Lù, X. H., Kasner, Z., and Reddy, S. Weblinx: Real-world website navigation with multi-turn
dialogue. In Forty-first International Conference on Machine Learning, ICML 2024, Vienna,
Austria, July 21-27, 2024 . OpenReview.net, 2024.
[22] Lyu, Y ., Wang, Z., Ren, Z., Ren, P., Chen, Z., Liu, X., Li, Y ., Li, H., and Song, H. Improving
legal judgment prediction through reinforced criminal element extraction. Inf. Process. Manag. ,
59(1):102780, 2022.
[23] Lyu, Y ., Hao, J., Wang, Z., Zhao, K., Gao, S., Ren, P., Chen, Z., Wang, F., and Ren, Z. Multi-
defendant legal judgment prediction via hierarchical reasoning. In Findings of the Association
for Computational Linguistics: EMNLP 2023, Singapore, December 6-10, 2023 , pp. 2198–2209.
Association for Computational Linguistics, 2023.
[24] Lyu, Y ., Li, P., Yang, Y ., de Rijke, M., Ren, P., Zhao, Y ., Yin, D., and Ren, Z. Feature-level
debiased natural language understanding. In Thirty-Seventh AAAI Conference on Artificial
Intelligence, AAAI 2023, Thirty-Fifth Conference on Innovative Applications of Artificial Intelli-
gence, IAAI 2023, Thirteenth Symposium on Educational Advances in Artificial Intelligence,
EAAI 2023, Washington, DC, USA, February 7-14, 2023 , pp. 13353–13361. AAAI Press, 2023.
[25] Lyu, Y ., Yan, L., Wang, S., Shi, H., Yin, D., Ren, P., Chen, Z., de Rijke, M., and Ren, Z.
Knowtuning: Knowledge-aware fine-tuning for large language models. In Proceedings of the
2024 Conference on Empirical Methods in Natural Language Processing, EMNLP 2024, Miami,
FL, USA, November 12-16, 2024 , pp. 14535–14556, 2024.
[26] Lyu, Y ., Yan, L., Wang, Z., Yin, D., Ren, P., de Rijke, M., and Ren, Z. MACPO: weak-to-strong
alignment via multi-agent contrastive preference optimization. CoRR , abs/2410.07672, 2024.
[27] Lyu, Y ., Zhang, X., Ren, Z., and de Rijke, M. Cognitive biases in large language models for
news recommendation. CoRR , abs/2410.02897, 2024.
11

[28] Mialon, G., Fourrier, C., Wolf, T., LeCun, Y ., and Scialom, T. GAIA: a benchmark for general
AI assistants. In The Twelfth International Conference on Learning Representations, ICLR
2024, Vienna, Austria, May 7-11, 2024 . OpenReview.net, 2024.
[29] Müller, M. and Žuni ˇc, G. Browser use = state of the art web agent, 2024. URL https:
//browser-use.com/posts/sota-technical-report .
[30] Nakano, R., Hilton, J., Balaji, S., Wu, J., Ouyang, L., Kim, C., Hesse, C., Jain, S., Kosaraju, V .,
Saunders, W., Jiang, X., Cobbe, K., Eloundou, T., Krueger, G., Button, K., Knight, M., Chess,
B., and Schulman, J. WebGPT: Browser-assisted question-answering with human feedback.
arXiv preprint arXiv:2112.09332 , 2021.
[31] Nguyen, D., Chen, J., Wang, Y ., Wu, G., Park, N., Hu, Z., Lyu, H., Wu, J., Aponte, R., Xia, Y .,
Li, X., Shi, J., Chen, H., Lai, V . D., Xie, Z., Kim, S., Zhang, R., Yu, T., Tanjim, M. M., Ahmed,
N. K., Mathur, P., Yoon, S., Yao, L., Kveton, B., Nguyen, T. H., Bui, T., Zhou, T., Rossi, R. A.,
and Dernoncourt, F. GUI agents: A survey. arXiv preprint arXiv:2412.13501 , 2024.
[32] Ning, L., Liang, Z., Jiang, Z., Qu, H., Ding, Y ., Fan, W., Wei, X.-y., Lin, S., Liu, H., Yu, P. S.,
et al. A survey of webagents: Towards next-generation ai agents for web automation with large
foundation models. arXiv preprint arXiv:2503.23350 , 2025.
[33] OpenAI. Introducing deep research. OpenAI Blog , 2025. URL https://openai.com/index/
introducing-deep-research/ .
[34] Pan, Y ., Kong, D., Zhou, S., Cui, C., Leng, Y ., Jiang, B., Liu, H., Shang, Y ., Zhou, S., Wu,
T., et al. WebCanvas: Benchmarking web agents in online environments. arXiv preprint
arXiv:2406.12373 , 2024.
[35] Ren, Z., He, X., Yin, D., and de Rijke, M. Information discovery in e-commerce. Found. Trends
Inf. Retr. , 18(4-5):417–690, 2024.
[36] Shaw, P., Joshi, M., Cohan, J., Berant, J., Pasupat, P., Hu, H., Khandelwal, U., Lee, K., and
Toutanova, K. From pixels to UI actions: Learning to follow instructions via graphical user
interfaces. In Advances in Neural Information Processing Systems 36: Annual Conference on
Neural Information Processing Systems 2023, NeurIPS 2023, New Orleans, LA, USA, December
10 - 16, 2023 , 2023.
[37] Shi, Z., Gao, S., Chen, X., Feng, Y ., Yan, L., Shi, H., Yin, D., Ren, P., Verberne, S., and Ren, Z.
Learning to use tools via cooperative and interactive agents. arXiv preprint arXiv:2403.03031 ,
2024.
[38] Shi, Z., Sun, W., Gao, S., Ren, P., Chen, Z., and Ren, Z. Generate-then-ground in retrieval-
augmented generation for multi-hop question answering. arXiv preprint arXiv:2406.14891 ,
2024.
[39] Shi, Z., Gao, S., Yan, L., Feng, Y ., Chen, X., Chen, Z., Yin, D., Verberne, S., and Ren, Z. Tool
learning in the wild: Empowering language models as automatic tool agents. In Proceedings of
the ACM on Web Conference 2025 , pp. 2222–2237, 2025.
[40] Shi, Z., Yan, L., Sun, W., Feng, Y ., Ren, P., Ma, X., Wang, S., Yin, D., de Rijke, M., and Ren,
Z. Direct retrieval-augmented optimization: Synergizing knowledge selection and language
models. arXiv preprint arXiv:2505.03075 , 2025.
[41] Shi, Z., Yan, L., Yin, D., Verberne, S., de Rijke, M., and Ren, Z. Iterative self-incentivization
empowers large language models as agentic searchers. arXiv preprint arXiv:2505.20128 , 2025.
[42] Sondhi, P., Sharma, M., Kolari, P., and Zhai, C. A taxonomy of queries for e-commerce search.
InThe 41st International ACM SIGIR Conference on Research & Development in Information
Retrieval, SIGIR 2018, Ann Arbor, MI, USA, July 08-12, 2018 , pp. 1245–1248. ACM, 2018.
[43] Song, Y ., Thai, K., Pham, C. M., Chang, Y ., Nadaf, M., and Iyyer, M. BEARCUBS: A
benchmark for computer-using web agents. arXiv preprint arXiv:2503.07919 , 2025.
12

[44] Sumers, T. R., Yao, S., Narasimhan, K., and Griffiths, T. L. Cognitive architectures for language
agents. Trans. Mach. Learn. Res. , 2024, 2024.
[45] Tsagkias, M., King, T. H., Kallumadi, S., Murdock, V ., and de Rijke, M. Challenges and research
opportunities in ecommerce search and recommendations. SIGIR Forum , 54(1):2:1–2:23, 2020.
[46] Wang, S., Liu, W., Chen, J., Gan, W., Zeng, X., Yu, S., Hao, X., Shao, K., Wang, Y ., and Tang,
R. GUI agents with foundation models: A comprehensive survey. CoRR , abs/2411.04890, 2024.
[47] Wang, Z., Zhao, Z., Lyu, Y ., Chen, Z., de Rijke, M., and Ren, Z. A cooperative multi-
agent framework for zero-shot named entity recognition. In Proceedings of the ACM on Web
Conference 2025 , pp. 4183–4195, 2025.
[48] Wang, Z. Z., Gandhi, A., Neubig, G., and Fried, D. Inducing programmatic skills for agentic
tasks. arXiv preprint arXiv:2504.06821 , 2025.
[49] Xu, C., Sun, Q., Zheng, K., Geng, X., Zhao, P., Feng, J., Tao, C., and Jiang, D. Wiz-
ardLM: Empowering large language models to follow complex instructions. arXiv preprint
arXiv:2304.12244 , 2023.
[50] Xue, T., Qi, W., Shi, T., Song, C. H., Gou, B., Song, D., Sun, H., and Su, Y . An illusion of
progress? assessing the current state of web agents. arXiv preprint arXiv:2504.01382 , 2025.
[51] Yao, S., Chen, H., Yang, J., and Narasimhan, K. Webshop: Towards scalable real-world web
interaction with grounded language agents. Advances in Neural Information Processing Systems ,
35:20744–20757, 2022.
[52] Yoran, O., Amouyal, S. J., Malaviya, C., Bogin, B., Press, O., and Berant, J. Assistantbench: Can
web agents solve realistic and time-consuming tasks? In Proceedings of the 2024 Conference on
Empirical Methods in Natural Language Processing, EMNLP 2024, Miami, FL, USA, November
12-16, 2024 , pp. 8938–8968. Association for Computational Linguistics, 2024.
[53] Zhang, C., He, S., Qian, J., Li, B., Li, L., Qin, S., Kang, Y ., Ma, M., Liu, G., Lin, Q., Rajmohan,
S., Zhang, D., and Zhang, Q. Large language model-brained GUI agents: A survey. arXiv
preprint arXiv:2411.18279 , 2024.
[54] Zhang, H., Wang, S., Zhang, K., Tang, Z., Jiang, Y ., Xiao, Y ., Yan, W., and Yang, W. Towards
personalized and semantic retrieval: An end-to-end solution for e-commerce search via embed-
ding learning. In Proceedings of the 43rd International ACM SIGIR conference on research and
development in Information Retrieval, SIGIR 2020, Virtual Event, China, July 25-30, 2020 , pp.
2407–2416. ACM, 2020.
[55] Zhang, X., Xin, X., Li, D., Liu, W., Ren, P., Chen, Z., Ma, J., and Ren, Z. Variational reasoning
over incomplete knowledge graphs for conversational recommendation. In Proceedings of
the Sixteenth ACM International Conference on Web Search and Data Mining, WSDM 2023,
Singapore, 27 February 2023 - 3 March 2023 , pp. 231–239. ACM, 2023.
[56] Zhang, X., Xie, R., Lyu, Y ., Xin, X., Ren, P., Liang, M., Zhang, B., Kang, Z., de Rijke, M., and
Ren, Z. Towards empathetic conversational recommender systems. In Proceedings of the 18th
ACM Conference on Recommender Systems, RecSys 2024, Bari, Italy, October 14-18, 2024 , pp.
84–93. ACM, 2024.
[57] Zhang, X., Xie, R., Lyu, Y ., Xin, X., Ren, P., Liang, M., Zhang, B., Kang, Z., de Rijke,
M., and Ren, Z. Towards empathetic conversational recommender systems. arXiv preprint
arXiv:2409.10527 , 2024.
[58] Zhang, Z., Tian, S., Chen, L., and Liu, Z. MMInA: Benchmarking multihop multimodal internet
agents. arXiv preprint arXiv:2404.09992 , 2024.
[59] Zhao, Z., Ren, Z., Yang, J., Yan, Z., Wang, Z., Yang, L., Ren, P., Chen, Z., de Rijke, M., and
Xin, X. Improving sequential recommenders through counterfactual augmentation of system
exposure. arXiv preprint arXiv:2504.13482 , 2025.
13

[60] Zheng, B., Gou, B., Kil, J., Sun, H., and Su, Y . Gpt-4v(ision) is a generalist web agent, if
grounded. In Forty-first International Conference on Machine Learning, ICML 2024, Vienna,
Austria, July 21-27, 2024 . OpenReview.net, 2024.
[61] Zheng, B., Fatemi, M. Y ., Jin, X., Wang, Z. Z., Gandhi, A., Song, Y ., Gu, Y ., Srinivasa, J., Liu,
G., Neubig, G., et al. SkillWeaver: Web agents can self-improve by discovering and honing
skills. arXiv preprint arXiv:2504.07079 , 2025.
[62] Zhou, S., Xu, F. F., Zhu, H., Zhou, X., Lo, R., Sridhar, A., Cheng, X., Ou, T., Bisk, Y ., Fried, D.,
Alon, U., and Neubig, G. WebArena: A realistic web environment for building autonomous
agents. In The Twelfth International Conference on Learning Representations, ICLR 2024,
Vienna, Austria, May 7-11, 2024 . OpenReview.net, 2024.
14

Appendix
A Details of Evaluation
A.1 GPT-4o for Evaluation
This section details the GPT-4o prompt used for fine-grained evaluation. As shown in Figure 6, we
adapt the prompt from prior work [ 11] to assess whether the web agent satisfies the requirements
related to product attributes, search filters, and sorting preferences. For each trajectory, GPT-4o
receives the subqueries, the agent’s action history, and a maximum of 15 screenshots, and is prompted
to determine whether the task was successfully completed.
A.2 Human Evaluation
To validate the reliability of GPT-4o-based evaluation, we conduct a human evaluation on 50 randomly
sampled trajectories from WebV oyager. A human annotator independently labels the success of
each subgoal—product attributes, search filters, and sorting preferences—as well as the overall task
completion. We then compute the agreement rate between human and GPT-4o assessments. The
instructions provided to the human annotator are shown in Figure 7.
Additionally, since deep research systems do not expose intermediate screenshots or action histories,
human evaluators are allowed to access the Amazon website to verify whether the returned links
satisfy the specified requirements. Due to limited use of the deep research systems, we evaluated
its performance on 30 randomly sampled queries. To ensure consistency, we evaluate the accuracy
based on the first recommended item when multiple products are returned.
A.3 Significance Test between Web Agents
We conducted paired t-tests to compare the overall task success rates between Browser Use and
four baseline models: Simple RAG ,Agent-E ,SeeAct , and WebVoyager . The results reveal
thatBrowser Use significantly outperforms Simple RAG and web agent baselines (p <0.05),
supporting the claim that direct interaction with the online web environment can substantially improve
agent performance over static retrieval methods such as Simple RAG. Furthermore, the comparison
highlights fundamental limitations in grounding capabilities across different agent types. HTML-
based agents like Agent-E lack access to visual context, often missing essential cues such as product
color or spatial layout. On the other hand, vision-based agents such as SeeAct and WebV oyager rely
on set-of-mark prompting, which suffers from segmentation errors: interactive elements are frequently
misclassified, and key sections—like customer review areas—remain unsegmented, making it difficult
to apply filters like rating thresholds.
B Details of Baselines
•Simple RAG method : We implement a simple retrieval-augmented generation (RAG) baseline by
combining GPT-4o with Google Search. The user query is first submitted to Google Search, and
the top-ranked webpage is selected. GPT-4o (version 2024-08-06) then generates a final response
conditioned on the screenshot of the retrieved page. We use the Serper API2to programmatically
access Google search results.
•Web agents : For fair comparison, all web agents are instantiated using GPT-4o (version 2024-
08-06) as the underlying language model. Agent-E [1] is an HTML-based agent that adopts a
hierarchical planner-actor architecture, augmented with flexible DOM tree distillation and a denois-
ing mechanism to improve decision accuracy.3SeeAct [60] exploits the multi-modal capabilities
of large language models (LLMs), integrating visual perception with structured web-based inter-
actions.4WebVoyager [11] also leverages multi-modal reasoning and introduces a set-of-mark
prompting scheme that guides the agent to first generate intermediate thoughts before selecting
2https://serper.dev/
3https://github.com/EmergenceAI/Agent-E
4https://github.com/OSU-NLP-Group/SeeAct
15

[System prompt]As an evaluator, you will be presented with three primary components to assist you in your role:1. Web Task Instruction: A clear and precise natural language directive that specifies an online shopping activity to be executed. The instruction may involve locating products that meet certain attribute requirements (e.g., color, size, brand), applying specific search filters (e.g., price range, customer ratings, availability), or fulfilling user-defined sorting preferences (e.g., lowest price, newest arrivals, best sellers). Tasks may also include verifying product details, comparing offers, or checking for shipping and return policies, depending on the scenario.2. Result Screenshots: This is a visual representation of the screen showing the result or intermediate state of performing a web task. It serves as visual proof of the actions taken in response to the instruction.3. Result Response: This is a textual response obtained after the execution of the web task. It serves as textual result in response to the instruction.--You DO NOT NEED to interact with web pages or perform actions such as conducting searches on websites.--You SHOULD NOT make assumptions based on information not presented in the screenshot when comparing it to the instructions.--Your primary responsibility is to conduct a thorough assessment of the web task instruction against the outcome depicted in the screenshot and in the response, evaluating whether the actions taken align with the given instructions.--NOTE that the instruction may involve more than one task, for example, locating the garage and summarizing the review. Failing to complete either task, such as not providing a summary, should be considered unsuccessful.--NOTE that the screenshot is authentic, but the response provided by LLM is generated at the end of web browsing, and there may be discrepancies between the text and the screenshots.--Note the difference: 1) Result response may contradict the screenshot, then the content of the screenshot prevails, 2) The content in the Result response is not mentioned on the screenshot, choose to believe the content.You should elaborate on how you arrived at your final evaluation and then provide a definitive verdict on whether the task has been successfully accomplished, either as 'SUCCESS' or 'NOT SUCCESS’.[User prompt]TASK:{subquery}Result Response:{answer}15 screenshots at the end:{screenshots}Figure 6: Prompts for GPT-4o evaluation
16

You will be presented with a web shopping task.For each task, you will receive three subqueries, along with the web agent's action history and corresponding screenshots. Your goal is to evaluate the agent’s performance across three specific dimensions: product attributes, search filters, and sorting preferences.Please note: if a subquery is labeled as None, you do not need to assess that particular aspect.Definitions of the three subqueries are as follows:1. Product attributes, referring to concrete product characteristics users may specify to express detailed intent2. Search filters, representing categorical or numerical constraints commonly used on e-commerce platforms3. Sorting preferences, indicating desired result orderings, such as price or popularity.Task:{Query}Product Attribute Requirement:{Subquery1}Search Filter Requirement:{Subquery2}Sorting Preference Requirement:{Subquery3}Agent Action History:{Action}Screenshots:{Screenshots}Figure 7: Instructions for human evaluation.
final actions.5Browser Use [29] is an open-source web agent framework that combines visual
understanding with HTML structure parsing to support robust web navigation and interaction.6
•Deep research systems : Since we cannot strictly constrain deep research systems to operate on
specific shopping websites, we include explicit site constraints in the prompt to guide the search
process. The prompt format used for these systems is illustrated in Figure 8. Gemini Deep
Research [8] is an AI assistant integrated into Google’s Gemini Advanced platform. We evaluate
the Gemini 2.0 Flash model with deep research capabilities.7OpenAI Deep Research [33] is
an agentic system powered by OpenAI’s reasoning models. We evaluate the o3 model with deep
research enabled.8
Prompt:                                                                                                         Your task is to help find relevant products on Amazon website (https://www.amazon.com/?language=en_US&currency=USD) based on the following question:{Query}Please strictly follow the rules: 1. Do not contact the user for further information. Provide answers directly without asking any questions or seeking clarification.2. Include Amazon product links for all recommended items.
Figure 8: Prompts for deep research systems.
5https://github.com/MinorJerry/WebVoyager
6https://github.com/browser-use/browser-use
7https://blog.google/products/gemini/google-gemini-deep-research/
8https://openai.com/index/introducing-deep-research/
17

C Details of Implementation
C.1 Prompts for Query Diversity and Complexity Evolution
Details for the prompts used in Diversity( ·), and Complexity( ·)are provided. Figures 9, 10, and
11 display the prompts for query diversity, complexity evolution and detailed complexity evolution
strategy, respectively.
I want you act as a Prompt Creator.Your goal is to draw inspiration from the #Given Prompt# to create a brand newprompt.This new prompt should be in the web shopping domain but tailored for different specific products in the {Randomly selected product category} amazon product field.The LENGTH and complexity of the #Created Prompt# should be similar to that of the #Given Prompt#.The #Created Prompt# must be reasonable and must be understood and responded by humans.'#Given Prompt#', '#Created Prompt#', 'given prompt' and 'created prompt' are not allowed to appear in #Created Prompt##Given Prompt#:{Original query}#Created Prompt#:
Figure 9: Prompts for query diversity evolution.
I want you act as a Prompt Rewriter for web shopping.Your objective is to rewrite a given prompt into a more complex version to make those web shopping agents a bit harder to handle.But the rewritten prompt must be reasonable and must be understood and responded by humans.You should complicate the given prompt using the following method:{Randomlyselectedevolutionstrategy} You should try your best not to make the #Rewritten Prompt# become verbose, #Rewritten Prompt# can only add 10 to 20 words into #The Given Prompt#. '#The Given Prompt#', '#Rewritten Prompt#', 'given prompt' and 'rewritten prompt' are not allowed to appear in #Rewritten Prompt##Given Prompt#:{Original query}#Created Prompt#:
Figure 10: Prompts for query complexity evolution.
C.2 Implementation Details
We evaluate open-source agents—Agent-E, SeeAct, WebV oyager, and Browser Use—within real-
time web environments. Agent-E, SeeAct, and Browser Use are executed via Playwright, while
WebV oyager leverages Selenium. To control computation cost and prevent excessive exploration, we
limit each agent to a maximum of 15 steps per task. All agents are powered by gpt-4o-2024-08-06
as the underlying language model. For automatic evaluation, we also adopt gpt-4o-2024-08-06
as evaluators, with temperature set to 0 to reduce response variance and enhance reproducibility.
The agents differ in their perception mechanisms: Agent-E and SeeAct utilize full-page screenshots,
whereas WebV oyager and Browser Use operate on the visible viewport only.
D Details of Error Analysis
•Web agents are limited by grounding ability. As shown in Figure 12, agents struggle to accurately
ground interface elements. For example, button 39 related to user rating was not properly segmented,
preventing the agent from selecting a specific rating range. Buttons 31–37 and 41–44 were rendered
too densely and overlapped, making interaction difficult. Additionally, the sorting button on
18

Product attributeevolution:Enhance #The Given Prompt# by integrating detailed product attributes that detail user needs. Please specifies concrete values for one product attribute (e.g., brand, model, price range, color, size, weight, or unique features) based on your knowledge about this product, ensure that these exact details are incorporated into the query instead of using generic placeholder terms.Search filterevolution:Enhance #The Given Prompt# by integrating detailed product constraints that capture user needs. Please specifies concrete values for constraints—such as a minimum customer rating (e.g., above 4.0 or 4.5 stars), a minimum number of customer reviews (e.g., 100, 300, 500, or 1000), shipping options like free delivery, new arrival time frames (e.g., released in the last 30 or 90 days), return policies (e.g., free returns), or warranty information (e.g., includes a 1-year warranty) based on your knowledge about amazon website, ensure that these exact values are used in the query rather than generic terms.Sorting preference evolution:Enhance #The Given Prompt# by integrating a specific product filtering requirement for web shopping. Find the top product based on one of the following criteria: lowest price, highest user rating, newest arrival, or best seller ranking.Figure 11: Prompts for complexity evolution strategies.
the right was incorrectly split into two buttons 16 and 17, which may confuse the agent during
execution.
Figure 12: Limited grounding ability of web agents.
•Web agents often lack state assessment and replanning capabilities. As illustrated in Figure 13,
when the agent enters a product detail page to verify a 1-year warranty, it fails to reassess its state
upon realizing that the requirement is unmet. Instead of returning to the search results page, the
agent continues to scroll within the current page, inefficiently attempting to locate an alternative
product.
•Web agents are limited by action space. As shown in Figure 14, the agent attempts to filter
cameras within the $100–$300 price range but fails to do so due to its inability to interact with
dynamic UI elements such as price sliders. Instead, it clicks the adjacent "Go" button without
adjusting the slider values, resulting in ineffective filtering. This highlights a fundamental limitation
of current agents: the constrained action space prevents them from performing fine-grained
interactions required in realistic web environments.
•Web agents lack the ability to learn from execution. As illustrated in Figure 15, we present
screenshots of the web agent across four different tasks. A consistent failure pattern emerges:
19

Action: ScrolldownAction: Click [16]
Action: Scrolldown
Do not satisfy the 1-year warrant
Keep exploring on the current page instead of going backFigure 13: Illustration of web agent’s failure to reassess and replan.
Action: Click [27]Clicked the 'Go' button in the price block but failed to filter prices to the $100–$300 range.
Figure 14: Illustration of the web agent’s failure to apply the price filter during task execution.
the agent repeatedly misuses the retriever to query filtering or sorting constraints, despite these
functionalities being accessible only via the designated filter or sorter UI components. Due
to the retriever’s inability to interpret such structural intents, it often returns irrelevant results
contaminated by noise. This repetition across tasks highlights the agent’s lack of execution-time
learning, preventing it from abstracting past mistakes and adapting its behavior accordingly.
•Deep research systems are limited by hallucination errors. As shown in Figure 16, the task
requires identifying a Women’s Vintage Floral Maxi Dress in Navy Blue, Size: Medium, and
explaining the return policy if free returns are available . However, as shown in Figure 17, the first
link returned by the OpenAI deep research system points to a product only available in Large and
XX-Large sizes. Despite this mismatch, the system hallucinates that the size requirement is met.
Moreover, Link2 and Link3 direct to non-Amazon websites, which violates the task constraint. In
an attempt to explain the return policy, the system incorrectly extracts return information from these
external sites. These issues reveal that when facing complex and fine-grained shopping queries,
the OpenAI deep research system frequently exhibits hallucinations—both in satisfying attribute
constraints and in sourcing policy information from inappropriate domains.
E Limitations and Societal Impacts
DeepShop has several limitations that open up avenues for future research. First, it focuses on desktop
interfaces without considering mobile-specific layouts [ 9,12]. Second, it lacks support for dynamic
user intent changes and multi-turn interactions [ 47,56]. Third, it does not fully capture cognitive
aspects of shopping behavior [ 27,44]. Fourth, it could benefit from recent advances in tool learning
and agent capabilities [7, 37–41].
From a societal perspective, while shopping agents can assist users with limited digital literacy
[2,59], they raise concerns about privacy and consumer manipulation [ 24,26,27]. Recent work
on knowledge-aware systems and reasoning approaches [ 22,23,25,55] suggests directions for
20

Figure 15: Illustration of the web agent’s failure to learn from execution.
Task:Locate the Return Policy for Women's Vintage Floral Maxi Dress on Amazon. Color: Navy Blue, Size: Medium. If free returns areavailable, please explain the return process for this item.OPENAI deep research system answer:Understood. I will search Amazon for listings of a Women's Vintage Floral Maxi Dress in Navy Blue, size Medium, and identify thereturn policy for each listing. I will focus on whether free returns are available and explain the return process for the item. I’ll share the direct Amazon links for all relevant listings I find.Research completed in 11m:Return Policy for Women's Vintage Floral Maxi Dress on AmazonFree Return Policies for Navy Blue Vintage Floral Maxi Dresses (Medium)When shopping on Amazon for a women’s vintage floral maxi dress in Navy Blue (size Medium), it’s important to check if the item offers Free Returns. Below are a few popular listings that match this description, along with their return policy details:FloernsWomen’s Sleeveless Halter Neck Vintage Floral Print Maxi Dress (Navy Blue)Product:FloernsWomen’s Sleeveless Halter Neck Vintage Floral Print Maxi DressColor:“A Navy Flower”,Size:Medium.Link:Amazon Product Page –FloernsHalter Neck Floral MaxiFree Returns:Yes.This Floernsdress is Prime-eligible and marked with “FREE Returns”, meaning you can return it at no cost if it doesn’t fit or meet your expectations [Link1]. Amazon provides at least one prepaid return shipping option for eligible items, so returning this dress is hassle-free.Return Process:You have 30 days from delivery to initiate a return via your Amazon account. Simply go to Your Orders, select this dress, and choose “Return or Replace Items.” Because it’s eligible for free returns, Amazon will offer a prepaid shipping label or QR code for drop-off [Link2]. You can then package the dress (in new, unworn condition with tags) and drop it off at an approved location (e.g. UPS Store, Whole Foods, Kohl’s, etc.) without paying return shipping [Link3]. Once the item is received, you’ll get a full refund to your original payment method.[Link1]: https://www.amazon.com/Floerns-Womens-Sleeveless-Halter-Vintage/dp/B0C3HQLBBQ#:~:text=,least%20one%20free%20return[Link2]: https://www.smoking-er.com/store/rosemary-pellet#:~:text=,preferred%20free%20shipping%20option%20Drop%C2%A0off%C2%A0and%C2%A0leave[Link3]: https://www.smoking-er.com/store/rosemary-pellet#:~:text=,preferred%20free%20shipping%20option%20Drop%C2%A0off%C2%A0and%C2%A0leave
Figure 16: Illustration of hallucination errors in the OpenAI deep research system. The system returns
three links for the query, including two that point to non-Amazon websites, and incorrectly assumes
the returned products match the specified size and platform constraints.
21

The available sizes are Large and XX-Large, whereas the task explicitly requires Size Medium.Figure 17: Detailed view of the first returned product link. Although the task specifies size Medium,
the linked product only offers Large and XX-Large options.
ensuring ethical decision-making. Future work should consider broader implications of agent-centric
information access [16] on consumer behavior and market dynamics.
F Dataset Card
Name: DeepShop
URL: https://huggingface.co/datasets/DeepShop/DeepShop
License: CC BY 4.0
Hosting platform: Hugging Face Datasets
Creator: DeepShop Team ( https://huggingface.co/DeepShop )
Format: Parquet (tabular), structured using the Croissant 1.1 metadata schema
Size: <1K examples
Language: English
Region: United States
Intended Use: Evaluation of web agents in online shopping tasks through complex query under-
standing and UI interaction.
Features: Each instance includes:
•id– the unique ID of the example
•ques – a natural language shopping query
•web_name andweb– the e-commerce platform name and its identifier
•attribute ,filter ,sort – subqueries describing product attribute, search filters, and sorting
preferences
•category – product category information
•difficulty – task difficulty level (e.g., easy, medium, hard)
Potential Uses: The dataset supports benchmarking of interactive web agents for shopping scenarios,
including multi-step reasoning over product attributes and UI components.
Limitations: The dataset currently focuses on desktop web interfaces and does not include user
behavior trajectories or feedback. It does not cover mobile layouts or multilingual queries.
22