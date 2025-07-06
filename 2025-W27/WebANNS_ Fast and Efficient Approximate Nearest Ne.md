# WebANNS: Fast and Efficient Approximate Nearest Neighbor Search in Web Browsers

**Authors**: Mugeng Liu, Siqi Zhong, Qi Yang, Yudong Han, Xuanzhe Liu, Yun Ma

**Published**: 2025-07-01 07:37:18

**PDF URL**: [http://arxiv.org/pdf/2507.00521v2](http://arxiv.org/pdf/2507.00521v2)

## Abstract
Approximate nearest neighbor search (ANNS) has become vital to modern AI
infrastructure, particularly in retrieval-augmented generation (RAG)
applications. Numerous in-browser ANNS engines have emerged to seamlessly
integrate with popular LLM-based web applications, while addressing privacy
protection and challenges of heterogeneous device deployments. However, web
browsers present unique challenges for ANNS, including computational
limitations, external storage access issues, and memory utilization
constraints, which state-of-the-art (SOTA) solutions fail to address
comprehensively. We propose WebANNS, a novel ANNS engine specifically designed
for web browsers. WebANNS leverages WebAssembly to overcome computational
bottlenecks, designs a lazy loading strategy to optimize data retrieval from
external storage, and applies a heuristic approach to reduce memory usage.
Experiments show that WebANNS is fast and memory efficient, achieving up to
$743.8\times$ improvement in 99th percentile query latency over the SOTA
engine, while reducing memory usage by up to 39\%. Note that WebANNS decreases
query time from 10 seconds to the 10-millisecond range in browsers, making
in-browser ANNS practical with user-acceptable latency.

## Full Text


<!-- PDF content starts -->

arXiv:2507.00521v2  [cs.IR]  2 Jul 2025WebANNS : Fast and Efficient Approximate Nearest Neighbor Search
in Web Browsers
Mugeng Liu
School of Computer Science,
Peking University
Beijing, China
lmg@pku.edu.cnSiqi Zhong
Fudan University
Shanghai, China
sqzhong21@m.fudan.edu.cnQi Yang
Institute for Artificial Intelligence,
Peking University
Beijing, China
qi.yang@stu.pku.edu.cn
Yudong Han
Institute for Artificial Intelligence,
Peking University
Beijing, China
hanyd@pku.edu.cnXuanzhe Liu
School of Computer Science,
Peking University
Beijing, China
xzl@pku.edu.cnYun Ma∗
Institute for Artificial Intelligence,
Peking University
Beijing, China
mayun@pku.edu.cn
Abstract
Approximate nearest neighbor search (ANNS) has become vital to
modern AI infrastructure, particularly in retrieval-augmented gen-
eration (RAG) applications. Numerous in-browser ANNS engines
have emerged to seamlessly integrate with popular LLM-based web
applications, while addressing privacy protection and challenges
of heterogeneous device deployments. However, web browsers
present unique challenges for ANNS, including computational lim-
itations, external storage access issues, and memory utilization
constraints, which state-of-the-art (SOTA) solutions fail to address
comprehensively.
We propose WebANNS , a novel ANNS engine specifically designed
for web browsers. WebANNS leverages WebAssembly to overcome
computational bottlenecks, designs a lazy loading strategy to opti-
mize data retrieval from external storage, and applies a heuristic
approach to reduce memory usage. Experiments show that WebANNS
is fast and memory efficient, achieving up to 743.8×improvement
in 99th percentile query latency over the SOTA engine, while re-
ducing memory usage by up to 39%. Note that WebANNS decreases
query time from 10 seconds to the 10-millisecond range in browsers,
making in-browser ANNS practical with user-acceptable latency.
CCS Concepts
•Information systems →Web applications ;Query optimiza-
tion.
Keywords
Approximate nearest neighbor search, Web browser, WebAssembly
∗Corresponding author.
Permission to make digital or hard copies of all or part of this work for personal or
classroom use is granted without fee provided that copies are not made or distributed
for profit or commercial advantage and that copies bear this notice and the full citation
on the first page. Copyrights for components of this work owned by others than the
author(s) must be honored. Abstracting with credit is permitted. To copy otherwise, or
republish, to post on servers or to redistribute to lists, requires prior specific permission
and/or a fee. Request permissions from permissions@acm.org.
SIGIR ’25, Padua, Italy
©2025 Copyright held by the owner/author(s). Publication rights licensed to ACM.
ACM ISBN 979-8-4007-1592-1/2025/07
https://doi.org/10.1145/3726302.3730115ACM Reference Format:
Mugeng Liu, Siqi Zhong, Qi Yang, Yudong Han, Xuanzhe Liu, and Yun Ma.
2025. WebANNS : Fast and Efficient Approximate Nearest Neighbor Search in
Web Browsers. In Proceedings of the 48th International ACM SIGIR Conference
on Research and Development in Information Retrieval (SIGIR ’25), July 13–18,
2025, Padua, Italy. ACM, New York, NY, USA, 10 pages. https://doi.org/10.
1145/3726302.3730115
1 Introduction
Approximate Nearest Neighbor Search (ANNS) has become a vital
component of modern artificial intelligence infrastructure [ 36], with
wide-ranging applications such as retrieval-augmented generation
(RAG) [ 33,34,44], data mining [ 35], search engines [ 40], and rec-
ommendation systems [ 18,41]. ANNS aims to find the top- kmost
similar vectors in large-scale, high-dimensional search spaces given
a query vector. In the growing application landscape of ANNS, the
query speed directly affects response latency, potentially becoming
a performance bottleneck and impacting user experience [20].
Among ANNS engine implementations, in-browser ANNS en-
gines [ 5,6,37] are emerging to integrate with popular LLM-based
web applications (web apps) [ 4]. These web apps combine in-browser
ANNS with cloud-based web services to enhance service quality by
leveraging personalized user data. In-browser ANNS engines offer
privacy guarantees by retrieving user data locally and only sending
essential data to cloud-based web services, particularly useful in sen-
sitive domains like finance, education, and healthcare [ 12,16,38].
Furthermore, leveraging the cross-device, out-of-the-box nature of
the web, these engines overcome the challenge of integrating vector
databases across diverse heterogeneous user devices, which other-
wise incurs significant development costs and raises deployment
and usage barriers [14, 39].
Despite the advantages, web browsers propose unique and signif-
icant limitations to the compute-intensive and memory-hungry na-
ture of ANNS. The SOTA in-browser ANNS engine is Mememo [ 37],
published in SIGIR’24 . Mememo uses the hierarchical navigable
small world (HNSW) algorithm, which is SOTA regarding construc-
tion and query efficiency [ 25,37]. HNSW is widely used among real-
world retrieval and RAG toolkits including FAISS [ 13], Pyserini [ 24],
PGVector [ 22], and LangChain [ 10]. Moreover, Mememo extends
browser memory via IndexedDB API [ 3] to store data on device
disks and reduces query overhead through data pre-fetching.

SIGIR ’25, July 13–18, 2025, Padua, Italy Mugeng Liu et al.
However, Mememo fails to thoroughly explore browser limi-
tations on ANNS, thereby overlooking critical optimization op-
portunities and potentially leading to unacceptable query latency
that extends into the sub-minute range. Therefore, we conduct a
comprehensive measurement of Mememo, identifying three key
limitations of SOTA in-browser ANNS engines.
Computational performance. Browsers prioritize cross-device
compatibility and development flexibility over computational per-
formance, heavily relying on interpreted languages like JavaScript
for computation. This reliance leads to significant overhead in sim-
ilarity calculations and sorting operations among vectors during
ANNS processes. Measured results show that the overhead for a
single query, excluding external storage access time, exceeds 100ms,
with similarity calculations accounting for over 40%, which high-
lights the high computational cost in browsers.
External storage access. Consecutive accesses to IndexedDB
are extremely slow [ 7]. To minimize IndexedDB access frequency,
Mememo heuristically pre-fetches additional vectors that might
be queried in a single access. Our measurement indicates that the
prefetch mechanism may fetch over 80% irrelevant vectors, causing
performance degradation when heuristics fail.
Restricted memory utilization. In-browser ANNS shares mem-
ory resources with other functionalities like user interface inter-
actions. Excessive memory usage during queries can make the
browser unresponsive. Mememo uses predefined fixed cache sizes
for queries, which cannot adapt to the varying resource conditions
of devices and browsers, resulting in inefficient use of limited mem-
ory resources.
To address these limitations, we propose WebANNS , a fast and
memory-efficient ANNS engine specifically designed for browsers,
which incorporates three key designs.
First, WebANNS utilizes WebAssembly (Wasm) [ 17] to accelerate
the computation and sorting operations in browsers. Wasm is a
binary instruction format that enables near-native computing speed
in browsers, offering enhanced performance over JavaScript for
compute-intensive tasks. However, Wasm is limited by only 32-
bit addressing [ 28] and sandbox isolation, resulting in constrained
memory capacity and no direct access to external storage. To ad-
dress this challenge, WebANNS utilizes a three-tier data management
mechanism involving Wasm, JavaScript, and IndexedDB. JavaScript
acts as an intermediary between Wasm and IndexedDB, serving as a
cache layer, a data exchange hub, and a bridge linking Wasm’s syn-
chronous execution model and IndexedDB’s asynchronous model.
Second, WebANNS employs a lazy external storage access strategy
to reduce IndexedDB access frequency and avoid unnecessary data
loading. However, the completely lazy loading strategy may break
the data dependency in the query process of HNSW, where the
similarity of a queried vector must be calculated to decide whether
neighboring vectors should be queried. This disruption may lead
to incorrect query paths and redundant calculations. To address
this challenge, WebANNS employs a phased lazy loading strategy to
efficiently load data while ensuring the correct query path.
Third, WebANNS uses a heuristic approach to estimating the min-
imum memory required without impacting query latency, reduc-
ing unnecessary memory usage and minimizing effects on other
browser functionalities. However, finding the optimal memory us-
age across different datasets, browsers, and devices is challenging.To address this challenge, WebANNS models the relationship between
query latency and memory size, reducing memory iteratively until
latency exceeds a set threshold.
We develop a prototype of WebANNS and evaluate it using five
commonly used datasets, ranging in size from 4MB to 7.5GB. Exper-
iments are conducted across three mainstream browsers and three
different devices. Experiment results demonstrate that WebANNS
achieves up to a 743.8×improvement in 99th percentile (P99) query
latency compared to Mememo, while reducing memory usage by
up to 39%. Moreover, WebANNS can reduce query latency from 10
seconds to just 10 milliseconds in a browser, making in-browser
ANNS feasible with user-acceptable latency.
In summary, this paper makes the following key contributions1
•We conduct a measurement study on the SOTA in-browser
ANNS engine and identify three major limitations imposed
by browsers on ANNS.
•We propose WebANNS , a novel ANNS engine specifically for
browsers, which leverages Wasm for computational effi-
ciency, designs a lazy loading strategy for efficient disk ac-
cess, and applies a heuristic approach to reduce memory
footprint.
•We develop a prototype of WebANNS and conduct compre-
hensive evaluations across diverse datasets, browsers, and
devices, indicating the significant enhancement of WebANNS .
2 Background and Motivation
This section introduces the background of in-browser ANNS and
our measurement study on the SOTA engine.
2.1 In-browser ANNS
Integrating personalized data into web apps to enhance service qual-
ity is increasingly popular. For example, the ChatGPT web app [ 29]
can remember user information with permission to generate per-
sonalized responses. To protect privacy, numerous web apps [ 4]
are integrating in-browser ANNS with cloud-based web services,
keeping sensitive data within the user’s browser instead of on po-
tentially untrustworthy cloud servers. Additionally, utilizing the
cross-device, out-of-the-box nature of the web, in-browser ANNS
engines enable easy multi-device access with one-time development
effort.
2.1.1 Indexing Algorithm. Given a query vector, ANNS algorithms
use similarity metrics, such as Euclidean distance, to identify the
top-knearest neighbors from high-dimensional, large-scale datasets.
To reduce computational costs, most ANNS algorithms [ 8,11,42,
43] employ indexing algorithms to prune data regions unlikely to
contain nearest neighbors.
Graph-based indexing is popular in in-browser ANNS [ 5,6,37]
because it delivers high accuracy and low latency with limited com-
putational resources. For example, the hierarchical navigable small
world (HNSW) algorithm serves as the indexing backbone of the
SOTA in-browser ANNS engine, Mememo, published in SIGIR’24 .
Known for its SOTA construction and querying performance among
graph-based indexing algorithms [ 25], HNSW is widely used in
1https://github.com/morgen52/webanns

WebANNS : Fast and Efficient Approximate Nearest Neighbor Search in Web Browsers SIGIR ’25, July 13–18, 2025, Padua, Italy
real-world retrieval and RAG toolkits like FAISS [ 13], Pyserini [ 24],
PGVector [22], and LangChain [10].
HNSW constructs a multi-layer graph, with each layer forming
a navigable small-world network. The key idea is to use these
hierarchical layers to efficiently navigate high-dimensional data,
narrowing down the search space as the query descends through
the layers. The query process starts at the top layer with a greedy
algorithm that iteratively selects the nearest neighbor until no closer
vectors are found. This process quickly eliminates distant vectors,
significantly reducing computational load. As the search descends
to lower layers, the algorithm continues to refine its list of nearest
neighbors. Each subsequent layer is more densely connected than
the previous one, which allows for a more precise local search. This
hierarchical graph effectively balances the trade-off between search
accuracy and computational cost.
2.1.2 Optimization of SOTA In-browser ANNS Engine. To miti-
gate the constraints of browsers, SOTA in-browser ANNS engine,
Mememo, introduces two browser-specific optimizations.
Storage expansion . Web pages can encounter memory limits
as low as 256MB [ 1], allowing storage of only up to 83,000 vec-
tors of 384 dimensions, excluding other memory usage. Moreover,
browsers restrict access to the OS file system for security, pre-
venting direct data storage on disks. Mememo uses IndexedDB, a
cross-browser key-value storage utilizing up to 80% of client disk
space [ 27]. IndexedDB is a low-level API for client-side storage
of significant amounts of structured data. It is the major standard
interface in browsers for large dataset storage [3].
Prefetching for efficient disk access . While IndexedDB mitigates
memory constraints, browser sandboxing can cause extremely slow
data access during consecutive IndexedDB API calls [ 7]. To reduce
disk access frequency, when vector values are missing from the
memory during a query, Mememo prefetches the current layer’s 𝑝
neighbors from IndexedDB into memory, where 𝑝is the pre-defined
cache size in memory.
2.2 Limitations of In-browser ANNS
We conduct a measurement study on the SOTA in-browser ANNS
engine, Mememo, and identify three key limitations.
Setup. We utilize a widely adopted dataset [ 20] from the Wikipedia
corpus [ 2], consisting of approximately 0.3million documents from
the most popular Wikipedia pages. The 7.5GB dataset, named Wiki-
480k, comprises 480,000 text items with 768-dimensional embed-
dings. Experiments are conducted in Chrome on a Linux system
equipped with an Intel Core i5-13400F processor (13th generation),
performing 100 queries with random vectors. We measure P99 query
latency for worst-case and average latency for typical performance.
2.2.1 High Computational Overhead. We gradually increase the
size of the uploaded data from Wiki-480k and measure the query
latency, excluding IndexedDB access latency, under conditions of
unlimited memory. When the data size exceeds 60,000 items (over
901MB), which accounts for only 12% of the size of Wiki-480k, the
webpage of Mememo crashes due to out-of-memory errors during
query process.
Fig. 1a illustrates that computational latency generally exceeds
50ms and can surpass 125ms. The latency becomes a bottleneck for
10 20 30 40 50 60
# of data (*103)255075100125query time (ms)
p99
avg(a) Computational latency ex-
cluding IndexedDB access.
10 20 30 40 50 60
# of data (*10³)10203040optime
querytime (%)
distance calculation
memory access(b) Computational bottlenecks.
Figure 1: Computing overhead and bottlenecks.
in-browser ANNS, especially in RAG web apps where LLM web
service can generate tokens in under 100ms [ 36]. Fig. 1b breaks
down the computation latency, with over 40% from distance cal-
culations, 5% from memory access, and the remaining 50% from
the data sorting and management of HNSW algorithm. The results
highlight that the extensive computational operations of ANNS
implemented in interpreted languages have become a significant
performance bottleneck. Therefore, computational cost is one of
the key limitations of in-browser ANNS.
Opportunities and Challenges . A potential solution is adopt-
ing Wasm [ 17], a binary instruction format that provides near-
native computing speeds in browsers, significantly improving per-
formance over JavaScript for compute-intensive tasks. However,
employing Wasm in in-browser ANNS presents challenges due to
its strict memory constraints and sandboxed isolation. First, Wasm
is limited by 32-bit addressing [ 28], allowing a maximum of 4GB
of memory, making it challenging for in-browser ANNS to scale
with the size of personalized datasets. Second, due to the strict
sandboxing, Wasm lacks direct access to external storage, like the
IndexedDB API, further complicating its use for in-browser ANNS.
2.2.2 High Overhead from External Storage Access. We conduct
experiments under limited memory conditions using a subset of the
Wiki dataset containing 50,000 items (Wiki-50k). Wiki-50k is the
largest dataset that Mememo can handle without crashing under
reduced memory. We gradually decrease the memory-data ratio and
record the end-to-end query latency. For instance, with a memory-
data ratio of 90%, the memory can store up to 90% of the data.
Fig. 2a shows that query latency rises rapidly as the memory-
data ratio decreases slightly. When this ratio drops below 98%,
the P99 query latency exceeds 10 seconds, leading to unaccept-
able response latency for browsers. At ratios under 96%, the P99
latency surpasses 40 seconds, indicating that Mememo is unusable
on memory-constrained browsers and devices.
Fig. 2b also shows the breakdown of operations during the query
process. When the memory-data ratio falls below 98%, the time
spent on external storage access exceeds that of distance calcula-
tions, and accounts for over 80% at the ratio of 96%, becoming the
primary performance bottleneck.
We further investigate the causes of high external storage access
overhead. Fig. 3a shows the redundancy rate ( 𝑅) of the prefetch
strategy, which is calculated by Equation 1,
𝑅=1−#hit
(#disk access×#prefetch size)(1)

SIGIR ’25, July 13–18, 2025, Padua, Italy Mugeng Liu et al.
96.0 97.0 98.0 99.0 100.0
memory-data ratio (%)01020304050query time (s)
p99
avg
(a) End-to-end query latency
with limited memory.
96.0 97.0 98.0 99.0 100.0
memory-data ratio (%)020406080optime
querytime (%)
distance
calculation
storage
access(b) Bottleneck of queries with
limited memory.
Figure 2: Overhead from external storage access.
96.0 97.0 98.0 99.0 100.0
memory-data ratio (%)20406080redundancy (%)
(a) Redundancy rate of the
prefetch strategy.
5 15 25 35 45
# of data (*103)2468101214time (s)
sequential
all in one(b) Latency of different loading
strategies in IndexedDB.
Figure 3: Optimization opportunity of storage access.
where #hitis the count of data hit in memory, #disk access is the num-
ber of storage accesses, and #prefetch size is the number of fetched
data per access.
Fig. 3a indicates that when the memory-data ratio drops be-
low 98%, the redundancy rate exceeds 50%, which means over half
the loaded data is unused and causes unnecessary delay. This is-
sue demonstrates the failure of the heuristic prefetch strategies of
Mememo when the memory-data ratio decreases. Therefore, we
need more effective methods to reduce external disk accesses and
minimize redundant loading across different memory-data ratios.
Opportunities and Challenges . Continuous IndexedDB accesses
can cause high latency [ 7]. An optimization opportunity is to maxi-
mize the effective data retrieved per disk access to reduce access
frequency.
Fig. 3b compares IndexedDB latency for different loading strate-
gies. Sequential loading involves nseparate disk accesses for sin-
gle items, while all-in-one loading retrieves nitems at one access.
Results show that all-in-one loading is about 45.4% faster than se-
quential loading, highlighting the overhead of transaction creation
with each IndexedDB access.
However, the challenge lies in reducing IndexedDB access fre-
quency effectively across different memory-data ratios. There is a
trade-off between the amount of data loaded per access and redun-
dancy. The queried vectors are unpredictable in the HNSW algo-
rithm because the distance to a vector must be calculated before
determining whether its neighbors should be queried. Therefore,
larger data loads per access can lead to less effective prefetching.
2.2.3 Inefficient Use of Memory Resources. Existing in-browser
ANNS engines [ 5,6,37] overlook the optimization of memory
utilization and rely mainly on a predefined memory threshold,
which cannot adapt to different browsers and devices automatically.
The static threshold may lead to excessive memory usage during
External StorageJavaScript
IndexedDBCache Optimizer(§3.4)
Cache ManagerCacheQuery DataQuery DataQuery DataLazy DataLoader (§3.3)PersonalizedDatasetSaver Index Loader DatabaseWebAssembly(§3.2)
CacheIndex BuilderIndex SearcherHNSW GraphOffline StageOnline StageEmbedding GeneratorFigure 4: Overview of WebANNS .
queries in memory-constrained devices, thereby affecting other
browser functionalities and degrading user experience of web apps.
Opportunities and Challenges. As shown in Fig. 2a, slightly re-
ducing the memory-data ratio may not significantly affect query
latency, suggesting the existence of an optimal memory thresh-
old (the ratio of 99% in Fig. 2a). The opportunity lies in adaptively
calculating this optimal memory threshold during the initializa-
tion stage of in-browser ANNS engine to reduce memory footprint
while maintaining the query latency. However, this threshold is in-
fluenced by disk access strategies, browsers, and devices, making it
challenging to efficiently determine the optimal memory threshold.
3WebANNS
This section presents the overview and detailed design of WebANNS .
We explain how the three-tier data management overcomes Wasm’s
address space limits, environment isolation, and execution model
differences. We also discuss how WebANNS minimizes IndexedDB
accesses through lazy external storage access and how to determine
the optimal memory threshold while ensuring the query latency.
3.1 Overview
To comprehensively address the limitations of in-browser ANNS,
we propose WebANNS , which is designed to optimize query latency
and memory utilization. Fig. 4 shows an overview of WebANNS .
Offline Indexing Construction . The WebANNS system starts by
taking the user’s personalized dataset as input. Using personalized
text data and semantic embeddings, WebANNS constructs an efficient
indexing graph of HNSW. The indexing graph, along with the text
data and embeddings, is stored in IndexedDB for long-term storage.
Browsers provide mechanisms like service worker [ 26] that enable
the construction process to be conducted offline . Once the indexing
graph is constructed, it can be loaded and reused for future queries.
Online Data Query . Online query latency is a primary focus for
ANNS engines [ 20,36,37] and directly impacts user experience.
The online query process of WebANNS is divided into initialization
and subsequent query stages.
During the initialization stage, WebANNS loads the HNSW index-
ing graph into Wasm memory via the index loader and performs
cache optimization to determine the optimal memory threshold,
ensuring minimal memory usage without compromising query per-
formance. At the query stage, WebANNS waits for query requests.
Once a query is received, WebANNS generates a query vector and

WebANNS : Fast and Efficient Approximate Nearest Neighbor Search in Web Browsers SIGIR ’25, July 13–18, 2025, Padua, Italy
WasmJavaScriptEventLoopCache MissEventLoop.add(fn())Set Shared Signalsetsig=0Invoke Load Function get()While(sig==0)IndexedDB
Continue SearchingEventLoop.move_to_end(Wasm()) Invoke JSAsync APIAwait Async Data FetchI/O Operationset sig=1StartFinishWasm()Wasm()fn()Wasm()fn()Wasm()WWfnfnWWWfn123456
Figure 5: Execution model coordination.
then the index searcher performs a search on the HNSW graph
based on the query vector.
3.2 Three-Tier Data Management
WebANNS uses Wasm to perform HNSW construction and queries
for enhanced computational performance. To address Wasm’s lim-
ited memory and lack of external storage access, WebANNS employs
a three-tier data management mechanism with Wasm, JavaScript,
and IndexedDB to manage data caching, data exchange, and coordi-
nation between synchronous and asynchronous execution models.
Data Caching. When a browser’s available memory exceeds Wasm’s
capacity, Wasm’s limited memory would be a bottleneck for data
cache size during the query process, causing more frequent accesses
to external storage and degrading performance. To mitigate this,
WebANNS introduces a JavaScript cache between the Wasm cache
and IndexedDB storage, forming a three-tier hierarchy. The first-
tier cache is within Wasm, storing frequently accessed vectors in its
limited memory. The second-tier cache is in the JavaScript environ-
ment, which holds more data when the Wasm cache is insufficient,
thereby reducing the external storage access frequency. The third
tier is the external storage, where all data is kept within IndexedDB
for retrieval as needed.
Data Exchange. Wasm and IndexedDB cannot directly access each
other due to their isolated environments. To facilitate data exchange,
WebANNS introduces JavaScript APIs as intermediaries. Specifically,
JavaScript provides get() andstore() APIs for Wasm to load missing
vectors and write vectors to IndexedDB. When Wasm encounters a
data miss, it notifies JavaScript to check the second-tier cache via
get() API. If the data is absent in the second-tier cache, it will be
fetched from IndexedDB by JavaScript. Once the data is loaded into
memory, JavaScript will return it to Wasm, completing the get()
API calls. If the first-tier cache needs to evict data, Wasm can move
it to the second-tier cache or external storage via the store() API
provided by JavaScript.
Execution Model Coordination. Wasm and IndexedDB have
fundamentally different execution models. Wasm supports only
synchronous execution, while IndexedDB operates asynchronously.
This means that when a cache miss occurs in Wasm, it cannot asyn-
chronously wait for the I/O operations of IndexedDB to complete.
To address the difference, WebANNS uses JavaScript as a bridge to
connect Wasm’s synchronous model with IndexedDB’s asynchro-
nous model. The detailed coordination process is shown in Fig. 5.1A shared signal 𝑠𝑖𝑔is set, indicating the loading operation is
pending. 2Theget() API is invoked to retrieve data. A new task of
asynchronously loading data ( fn()) from IndexedDB via JavaScript is
pushed into the end of the event loop of JavaScript. 3If the Wasm
task is at the top of the event loop, Wasm continually monitors
the shared signal 𝑠𝑖𝑔. If signal𝑠𝑖𝑔has not changed, Wasm calls the
asynchronous wait API provided by JavaScript to append the Wasm
task to the end of the event loop, thereby freeing up computation
during the wait. 4The data loading task of JavaScript moves to
the top of event loop and can be executed to handle asynchronous
I/O operations with IndexedDB. 5Once the data is fetched from
IndexedDB, JavaScript sets the shared signal 𝑠𝑖𝑔as completion
and passes it to the Wasm memory. Then the data loading task is
finished. 6The Wasm task is executed and detects the change of
𝑠𝑖𝑔, and then resumes its computation.
Through the three-tier data management hierarchy, WebANNS uti-
lizes JavaScript as an intermediary cache, a data exchange bridge,
and a connector between Wasm’s and IndexedDB’s execution mod-
els. This approach effectively overcomes Wasm’s memory limita-
tions and facilitates efficient data exchange between isolated envi-
ronments with different execution models, thereby achieving Wasm
acceleration for the query process.
3.3 Lazy External Storage Access
Heuristic prefetching algorithms can lead to considerable redundant
data loading, reducing the efficiency of external storage access.
Given the high overhead of accessing external storage, it is crucial
to minimize redundancy while maximizing the amount of data
loaded per access.
Completely Lazy Loading. An extreme approach is to ignore all
the data that should be queried but is not in memory during the
query process. Once the in-memory query is completed, all ignored
data is loaded into memory with a single IndexedDB access to
continue querying. This completely lazy loading approach ensures
that all data fetched from IndexedDB is required to be queried,
thus eliminating redundant external data access. However, this
approach is unsuitable for the query process of HNSW for two
reasons. First, HNSW requires identifying the correct entry point
for each layer. If an entry point is ignored due to a cache miss,
queries in subsequent layers will start from the wrong entry points,
resulting in inaccurate results. Second, calculating the distance
to a queried vector is necessary for HNSW to decide whether its
neighbors should be queried. Ignoring a queried vector might cause
all its neighbors to be overlooked, thereby increasing the risk of
following an incorrect query path.
Phased Lazy Loading. Building on the complete lazy loading ap-
proach, WebANNS applies timing restrictions on lazy data loading to
prevent excessive redundant computations along incorrect query
paths. The timing of data loading is determined by two key obser-
vations. First, within the search space of a layer, the entry points of
the next layer will be accurate if all necessary vectors are loaded
and queried by the end of the search in that layer. Second, when the
list of ignored vectors exceeds the efparameter, which specifies
the size of the candidate list for queries, the ignored list definitely
includes extra vectors that do not require querying and neighbor
evaluation.

SIGIR ’25, July 13–18, 2025, Padua, Italy Mugeng Liu et al.
Algorithm 1 SEARCH-LAYER-WITH-PHASED-LAZY-LOADING
Input: query element 𝑞, entry points 𝑒𝑝, number of nearest
elements to return 𝑒𝑓, layer number 𝑙𝑐.
Output:𝑒𝑓closest neighbors to 𝑞.
1:𝑣←𝑒𝑝// set of visited elements
2:𝐶←𝑒𝑝// set of candidates
3:𝑊←𝑒𝑝// dynamic list of found nearest neighbors
4:𝐿←∅// candidates for lazy loading ⊲lazy
5:while truedo ⊲lazy
6: while|𝐶|>0do
7:𝑐←extract nearest element from 𝐶to𝑞
8:𝑓←get furthest element from 𝑊to𝑞
9: ifdistance(𝑐,𝑞)>distance(𝑓,𝑞)then
10: break // all elements in 𝑊are evaluated
11: foreach𝑒∈neighbourhood( 𝑐) at layer𝑙𝑐do
12: if𝑒∉𝑣then
13: 𝑣←𝑣∪𝑒
14: if𝑒not in memory then ⊲lazy
15: 𝐿←𝐿∪𝑒 ⊲lazy
16: continue ⊲lazy
17: if𝑒should be a candidate then
18: 𝐶←𝐶∪𝑒
19: 𝑊←𝑊∪𝑒
20: if|𝑊|>𝑒𝑓then
21: remove furthest element in 𝑊
22: if|𝐿|>𝑒𝑓then ⊲intra-layer
23: break ⊲lazy
24: if|𝐿|>0then ⊲inter-layer
25: load elements in 𝐿to memory ⊲lazy
26: foreach𝑒∈𝐿do ⊲lazy
27: if𝑒should be a candidate then ⊲lazy
28: 𝐶←𝐶∪𝑒 ⊲lazy
29: 𝑊←𝑊∪𝑒 ⊲lazy
30: if|𝑊|>𝑒𝑓then ⊲lazy
31: remove furthest element in 𝑊 ⊲lazy
32: else ⊲lazy
33: break ⊲lazy
34:return𝑊
Based on these observations, WebANNS employs phased lazy load-
ing, as shown in Algorithm 1. For the inter-layer query phase, after
completing the search at each layer, any remaining lazily loaded
vectors will be loaded in a single disk access, allowing the search to
continue. This process repeats until the search is completed with
no remaining ignored vectors, ensuring correct entry points of the
next layer. For the intra-layer query phase, if the list of ignored
vectors exceeds the efparameter, all the vectors in the lazily loaded
list will be loaded in a single disk access transaction. This process
prevents the unlimited growth of the lazily loaded list, avoiding the
loading of excessive redundant vectors.
The phased lazy load mechanism in WebANNS minimizes external
storage access by lazily loading data only when necessary. It ensures
correct search paths by maintaining accurate entry points for each
layer and controlling the size of the lazily loaded vector list, thereby
eliminating excessive redundant loading.3.4 Heuristic Cache Size Optimization
Due to the limited memory available to web apps, excessive mem-
ory usage by ANNS may disrupt other browser functionalities like
user interactions and negatively affect the user experience. There-
fore, WebANNS aims to minimize memory usage of ANNS while
maintaining query latency.
However, it is challenging to adaptively determine the optimal
memory threshold, which is closely associated with the prefetch
strategy, disk access strategy, browser, and device.
Our insight is to treat the query process as a black box. Starting
with the maximum memory, we gradually reduce the memory limi-
tation and observe changes in query latency to identify the optimal
memory threshold. Therefore, the problem lies in determining the
appropriate step size for memory reduction.
Naïve Method. A naïve method is to reduce memory with a fixed
step size, but choosing the right size is challenging. A small step
size may require excessive query tests to find the optimal thresh-
old, while a large step size might skip effective memory sizes and
significantly increase the latency of query tests when the memory
size drops below the optimal threshold.
Heuristic Method To heuristically determine the step size for
memory reduction, we model query latency under various memory
sizes, as shown in Equation 2,
𝑇query=𝑇in-mem+𝑇db
=|𝑄|·𝑡in-mem+𝑛𝑑𝑏·𝑡𝑑𝑏 (2)
where|𝑄|is the number of data items visited during one HNSW
search,𝑄denotes the ordered sequence of visited items (i.e., search
path),𝑡in-mem is the time required for in-memory computations for
each queried vector, 𝑛𝑑𝑏is the number of disk accesses, and 𝑡𝑑𝑏is
the time taken for a single IndexedDB access.
Changes in memory size mainly affect 𝑛𝑑𝑏, ultimately impact-
ing𝑇query . Therefore, we can focus on the relationship between
the number of items that can be stored in memory 𝑛mem and𝑛𝑑𝑏.
This relationship is mainly affected by the data fetch strategy from
IndexedDB to memory.
With random data fetching , it can be proven that 𝑛𝑑𝑏decreases
linearly with an increase in 𝑛mem, satisfying Equation 3,
𝑛𝑑𝑏=(1−|𝑄|
𝑁−1·𝑛mem+𝑁·|𝑄|−1
𝑁−1,if𝑛mem <𝑁
1, if𝑛mem≥𝑁(3)
where𝑁is the number of items in the dataset.
Proof. Consider a query that needs to access |𝑄|vectors from a
dataset containing 𝑁items, with the memory capable of holding
𝑛mem vectors.
(1)When𝑛mem≥𝑁, all vectors can be loaded into memory in a
single IndexedDB access.
(2)When𝑛mem=𝑀<𝑁and the memory is initially empty:
For the first queried vector 𝐷1, a cache miss occurs, requiring one
IndexedDB access to load 𝐷1and𝑀−1other random vectors into
memory. Thus, the memory contains 𝐷1and𝑀−1other random
data items.
For the𝑖-th queried vector (denoted as 𝐷𝑖,𝑖≥2), which is
different from 𝐷𝑖−1, the probability that 𝐷𝑖is in memory is given
by𝑃hit=𝑀−1
𝑁−1.Since the memory contains 𝐷𝑖−1and𝑀−1data

WebANNS : Fast and Efficient Approximate Nearest Neighbor Search in Web Browsers SIGIR ’25, July 13–18, 2025, Padua, Italy
Algorithm 2 APPROXIMATING-CURVE-OF-REAL-FETCHING-
STRATEGY
1:function OPTIMIZE_MEMORY_SIZE (𝐶0,𝑝,𝑇𝜃)
2: //𝐶0is maximum memory size
3: //𝑝and𝑇𝜃are the percentage and absolute thresholds
4:𝐶𝑏𝑒𝑠𝑡←𝐶0// best memory size
5:𝐶𝑡𝑒𝑠𝑡←𝐶0// memory size to be evaluated
6: while 0<𝐶𝑡𝑒𝑠𝑡≤𝐶0do
7:𝑛𝑑𝑏,𝑛𝑄,𝑇𝑞𝑢𝑒𝑟𝑦 ,𝑡𝑑𝑏←QUERY_TEST()
8: //𝑛𝑑𝑏is number of disk access, 𝑛𝑄is length of
query path, 𝑇𝑞𝑢𝑒𝑟𝑦 is total query time, 𝑡𝑑𝑏is time for single
disk access
9:𝜃←GET_THETA( 𝑝,𝑇𝜃,𝑇𝑞𝑢𝑒𝑟𝑦 ,𝑡𝑑𝑏)
10: if𝑛𝑑𝑏>𝜃then
11: Break // Over the threshold
12: else
13: 𝐶𝑏𝑒𝑠𝑡←𝐶𝑡𝑒𝑠𝑡
14:𝑘←𝑛𝑄−𝑛𝑑𝑏
1−𝐶𝑡𝑒𝑠𝑡
15:𝐶𝑡𝑒𝑠𝑡←l𝜃−𝑛𝑄
𝑘+1m
16: return𝐶𝑏𝑒𝑠𝑡
17:
18:function GET_THETA (𝑝,𝑇𝜃,𝑇𝑞𝑢𝑒𝑟𝑦 ,𝑡𝑑𝑏)
19: //𝑝and𝑇𝜃are the percentage and absolute thresholds
20:𝜃←𝑚𝑎𝑥𝑝·𝑇𝑞𝑢𝑒𝑟𝑦
𝑡𝑑𝑏,𝑇𝜃
𝑡𝑑𝑏
21: return𝜃.
items randomly selected from the dataset. If a cache miss occurs,
another IndexedDB access is performed to load 𝐷𝑖and𝑀−1other
random data items.
Thus, the first query always results in an IndexedDB access. For
the𝑖-th query ( 2≤𝑖≤|𝑄|), the probability of a cache miss per
query is, 1−𝑃hit=𝑁−𝑀
𝑁−1.Therefore, the total number of IndexedDB
accesses is𝑛𝑑𝑏=1+(|𝑄|−1)·𝑁−𝑀
𝑁−1,which can be simplified to
𝑛𝑑𝑏=1−|𝑄|
𝑁−1·𝑛mem+𝑁·|𝑄|−1
𝑁−1.
□
With optimal data fetching , each IndexedDB access can prefetch
the next𝑛mem vectors in the query path into memory, demon-
strating that 𝑛𝑑𝑏is inversely proportional to 𝑛mem, as shown in
Equation 4.
𝑛𝑑𝑏=(l|𝑄|
𝑛memm
,if𝑛mem <|𝑄|
1, if𝑛mem≥|𝑄|(4)
Proof. Consider a query that needs to access |𝑄|vectors from a
dataset containing 𝑁items, with the memory capable of holding
𝑛mem vectors. With optimal data fetching, each disk access can
always prefetch the next 𝑛mem vectors in the query path Q.
(1)When𝑛mem≥|𝑄|, the memory can hold all the vectors in
the query path within a single IndexedDB access.
(2)When𝑛mem <|𝑄|, the minimum number of IndexedDB
accesses needed is determined by how many full sets of 𝑛mem
vectors can be loaded to cover all |𝑄|vectors. This is given by
𝑛𝑑𝑏=l|𝑄|
𝑛memm
. □
01 C2C1C0 |Q| N
nmem01
|Q|ndbFnbest(x)
Fnrandom(x)A
BX0X/prime
0
X1X/prime
1Figure 6: Approximating the curve of real fetching strategy.
Approximating the curve of real fetching strategy. We hy-
pothesize that the real fetching strategy results in fewer IndexedDB
accesses than random fetching. Therefore, the curve should lie be-
tween Fnoptimal(𝑥)andFnrandom(𝑥), as shown in Fig. 6. By setting
a threshold𝜃for𝑛𝑑𝑏, we can start with the maximum memory size
𝐶0and iteratively reduce memory to find the optimal memory size
under the real fetching strategy.
Algorithm 2 shows the overall memory optimization process.
WebANNS evaluates the number of IndexedDB accesses at the maxi-
mum memory size 𝐶0through query testing, represented as point
𝑋0. If the accesses at 𝐶0exceed the threshold 𝜃,WebANNS retains the
memory size at 𝐶0. Otherwise, the next memory size 𝐶1is found
by intersecting the line from 𝑋0to endpoint 𝐴with𝑦=𝜃. For each
subsequent memory size 𝐶𝑖(𝑖≥1), if accesses exceed 𝜃,𝐶𝑖−1will
be selected as the optimal memory size.
This algorithm can rapidly approximate the curve of the real
fetching strategy, based on two observations. First, the gradient
near the optimal memory size is relatively gentle, allowing lines
connecting to the extreme point to quickly converge on the optimal
memory size. Second, as the algorithm progresses, the memory
reduction steps will decrease, while the absolute value of the gradi-
ent of the line connecting to the extreme point increases and the
database accesses in query tests approach 𝜃.
Setting of𝜃.𝜃can be set using two methods based on Equation 2.
The first method uses a percentage parameter 𝑝to ensure external
storage access time remains below a specific ratio of 𝑇query , cal-
culated as𝜃=𝑝×𝑇query
𝑡𝑑𝑏. Higher𝑝reduces memory but increases
latency, and vice versa. The second method sets an absolute time
limit to ensure the total time of IndexedDB access does not exceed a
certain threshold, calculated as 𝜃=𝑇𝜃
𝑡𝑑𝑏, where𝑇𝜃is the acceptable
IndexedDB access time, such as 100ms. WebANNS incorporates both
methods for setting 𝜃.
Rollback of memory size. The memory optimization algorithm
tracks a sequence of memory sizes {𝐶0,𝐶1,...}and their corre-
sponding𝜃values. During queries, if the IndexedDB access count
at𝐶𝑖exceeds the corresponding 𝜃,WebANNS would roll back the
memory size to 𝐶𝑖−1. The rollback process continues until 𝐶0is
reached, ensuring a rapid return to a memory size that maintains
query performance when fluctuations occur.
Through heuristic cache size optimization, WebANNS can maintain
the query latency while minimizing memory footprint.
4 Experiments
This section presents the implementation and experiments of WebANNS .

SIGIR ’25, July 13–18, 2025, Padua, Italy Mugeng Liu et al.
Table 1: P99 query time (ms) across heterogeneous devices and browsers for 100 queries with unrestricted memory usage.
Device/BrowserWiki-60k (901MB) Wiki-480k (7.5GB) Arxiv-1k (4MB) Arxiv-120k (460MB) Finance-13k (214MB)
Meme./ WebANNS Boost Meme./ WebANNS Meme./ WebANNS Boost Meme./ WebANNS Boost Meme./ WebANNS Boost
Linux/Chrome 7636.30 / 13.49 566.28× N/A / 23.46 7.99 / 1.54 5.17× 52.77 / 16.12 3.27× 40.31 / 8.58 4.70×
Linux/Firefox 3865.22 / 26.38 146.52× N/A / 45.92 10.60 / 2.30 4.61× 170.74 / 21.62 7.90× 33.80 / 17.18 1.97×
Mac/Chrome 11975.20 / 16.10 743.80× N/A / 30.40 5.70 / 1.50 3.80× 44.80 / 13.60 3.29× 39.10 / 8.70 4.49×
Mac/Firefox 3023.00 / 43.00 70.30× N/A / 72.00 9.00 / 4.00 2.25× 69.00 / 30.00 2.30× 33.00 / 22.00 1.50×
Mac/Safari 3107.00 / 16.00 194.19× N/A / 28.00 4.00 / 2.00 2.00× 61.00 / 14.00 4.36× 20.00 / 8.00 2.50×
Win/Chrome 16937.10 / 23.40 723.81× N/A / 39.10 10.50 / 2.60 4.04× 80.70 / 22.00 3.67× 66.90 / 11.60 5.77×
Win/Firefox 11975.00 / 42.00 285.12× N/A / 65.00 16.00 / 3.00 5.33× 132.00 / 32.00 4.12× 60.00 / 21.00 2.86×
Win/Firefox 11975.00 / 42.00 285.12× N/A / 65.00 16.00 / 3.00 5.33× 132.00 / 32.00 4.12× 60.00 / 21.00 2.86×
Table 2: Ablation experiments on P99 query time (ms) with
limited memory.
Mememo WebANNS -Base WebANNS
Device/
BrowserMemory/Data Ratio (%)
20% 90% 96% 98% 100%
Linux/
ChromeN/A N/A 48743.38 10574.26 5472.33
34216.27 1760.31 825.44 477.67 14.07
425.39 106.97 50.01 34.58 14.19
Linux/
FirefoxN/A N/A 16917.14 5478.30 2500.22
32705.08 1940.90 866.40 390.90 28.42
591.36 167.58 127.40 71.64 28.50
Mac/
ChromeN/A N/A N/A 12563.10 7480.00
42105.90 2369.40 1375.20 444.80 15.60
501.80 91.00 54.90 49.20 15.20
Mac/
FirefoxN/A N/A 12267.00 4730.00 2475.00
50334.00 3603.00 1417.00 662.00 43.00
1060.00 161.00 92.00 77.00 45.00
Mac/
SafariN/A N/A 26540.00 8041.00 4742.00
39078.00 2185.00 915.00 480.00 16.00
688.00 98.00 61.00 44.00 16.00
Win/
ChromeN/A N/A 65895.50 23161.10 12212.80
42374.80 3286.00 923.40 514.10 22.50
828.90 110.50 70.10 57.20 21.60
Win/
FirefoxN/A N/A 55831.00 16301.00 8998.00
124241.00 6453.00 3716.00 1708.00 39.00
2127.00 314.00 157.00 142.00 42.00
4.1 Implementation
Based on the JavaScript implementation of HNSW from Mememo [ 37],
we implement HNSW in C++ and compile it into Wasm. The pro-
totype of WebANNS is developed using over 3,300 lines of C++ and
TypeScript code. We also develop a web app with a user interface to
validate the effectiveness of WebANNS in browsers. Apart from the
three key optimizations mentioned before, the prototype system
includes the following enhancements.
Text-embedding separation. In in-browser ANNS engines, only
embeddings are needed during queries, not the original text. Storing
text-embedding pairs of data directly as key-value pairs can result
in high memory overhead due to the length of the text. To address
this, we assign a unique ID to each data item and manage texts and
embeddings separately. This ID indexes both embeddings and texts,
minimizing memory usage for each entry.
Streaming data loading. In browser environments, loading user-
uploaded personalized datasets or large HNSW index trees can
exceed memory limits, risking browser crashes. To address this, theWebANNS prototype system supports stream loading, enabling index
trees and data files to be read and loaded in chunks.
Cache eviction strategy. For simplicity, the WebANNS prototype
uses FIFO as the eviction strategy for caches in both Wasm and
JavaScript. Note that WebANNS prototype provides a unified abstract
interface for cache eviction, allowing the system to easily support
various eviction algorithm implementations as pluggable modules.
4.2 Setup
Dataset. We evaluate our approach using one general dataset (Wiki)
and two domain-specific datasets (ArXiv and Finance). (1) Wiki [2]
is the dataset used in our measurements study. It contains 480,000
text passages and their 768-dimensional vector embeddings. The
dataset is 7.5GB in size and widely used for evaluating the per-
formance of ANNS [ 20].(2) Arxiv is a collection of two datasets
with sizes of 1k and 120k, containing the abstracts of arXiv ma-
chine learning papers. Following prior work [ 4,37], these datasets
are used to assess the in-browser ANNS engine’s performance
across varying scales. (3) Finance is the FinDER dataset [ 31], open-
sourced by LinQ, which includes more than 10k reports and finan-
cial disclosures. It contains a substantial corpus of domain-specific
text, representing the scale of personalized and privacy-sensitive
datasets.
Baseline. We compare WebANNS with the following two baselines.
For each baseline, we report the performance achieved without en-
countering out-of-memory errors. (1) Mememo [37] is the SOTA in-
browser ANNS engine, which is published in SIGIR’24 .(2)WebANNS -
Base removes lazy loading and memory optimization but includes
all other optimizations. It highlights the benefits of Wasm from
Section 3.2. The comparison with WebANNS -Base underscores gains
from techniques in Sections 3.3 and 3.4, eliminating any unfair com-
parisons caused by the differences in underlying implementations.
Metrics. We use the query latency of in-browser ANNS as a met-
ric, following prior work [ 36,42]. Since neither Mememo nor our
work optimizes the conversion of query text to embeddings, we
measure end-to-end latency from inputting the query embeddings
to receiving the search results. In each experiment setting, after an
initial warm-up query, we record latency over 100 iterations and
calculate the 99th percentile (P99) latency to represent the duration
for most queries.
Environment. We conduct experiments on a Linux desktop with
an Intel Core i5-13400F processor (13th generation), a MacOS laptop
with an Apple M2 processor, and a Windows laptop with an AMD

WebANNS : Fast and Efficient Approximate Nearest Neighbor Search in Web Browsers SIGIR ’25, July 13–18, 2025, Padua, Italy
R7-4800U processor. The browsers used for experiments involve
Chrome (version 131.0), Firefox (version 133.0), and Safari (version
18, macOS only).
4.3 Query Performance
Table 1 shows the query latency comparison when memory usage
is unrestricted. WebANNS demonstrates a significant speedup across
all evaluated conditions, with boosts ranging from 1.5×to743.8×.
For large datasets, WebANNS shows more significant improve-
ments. This is mainly because Mememo’s heuristic prefetching
strategy may fail to load all necessary data into memory, even
without memory restrictions, leading to multiple external storage
accesses. WebANNS minimizes external storage access through a
three-tier cache and lazy loading techniques, achieving a 70-744x
performance boost on large datasets like Wiki, reducing retrieval
times from seconds to around 10ms. This enhancement transforms
nearly unusable searches of Mememo into usable ones.
For smaller datasets, where external storage accesses are not in-
volved, WebANNS fully leverages the computing efficiency of Wasm,
achieving a 2–5.33×improvement on the smallest dataset, Arxiv-1k.
Additionally, WebANNS expands the dataset size supported for
retrieval in browsers by at least 8.5 times. Frequent external access
and excessive memory usage cause Mememo to crash when han-
dling datasets larger than Wiki-60k, whereas WebANNS can complete
the retrieval of the 7.5GB Wiki-480k dataset in 72ms.
Impact of browser and device heterogeneity. WebANNS consis-
tently improves query speed across all browsers and devices. Com-
pared to Linux desktops with more powerful processors, WebANNS
achieves higher boost rates on Mac and Windows laptops, reach-
ing a 743×boost. Regarding the performance of WebANNS across
browsers, it achieves the lowest latency on Chrome, reaching as
low as 1.5ms, while Firefox is the slowest, likely due to the different
performance of their Wasm engines.
4.4 Ablation Study
We conduct ablation experiments on various devices with the Wiki-
50k dataset, the largest dataset that Mememo can handle without
crashing under reduced memory. We set memory equal to the
dataset size and reduce the memory-data ratio incrementally. For
example, a 90% memory-data ratio means the memory holds up to
90% of the data. Results are shown in Table 2.
Three-tier data management. Mememo becomes unusable with
sub-minute level query times and even crashes with a memory-data
ratio below 98% due to exponential query time increases. WebANNS -
Base, with Wasm and three-tier caching, archives at least an order-
of-magnitude improvement compared to Mememo.
Lazy external storage access. The lazy loading mechanism of
WebANNS reduces external access at best efforts, further achieving
at least an order-of-magnitude improvement compared to WebANNS -
Base at memory ratio below 90%. Moreover, WebANNS can save
much memory footprint within tolerable query performance, keep-
ing sub-second query latency even at a 20% memory ratio. Besides,
the lazy storage access mechanism does not introduce much over-
head compared to WebANNS -Base. When memory can hold all dataTable 3: Heuristic cache size optimization ( 𝑝=0.8,𝑇𝜃=100𝑚𝑠)
Device/ Init Opt Saved P99 Query
Browser Mem. (MB) Mem. (MB) Mem. (MB) Time (ms)
Linux/Chrome 146.48 101.70 44.79 (31%) 170.22
Linux/Firefox 146.48 104.23 42.25 (29%) 257.80
Mac/Chrome 146.48 89.08 57.40 (39%) 186.30
Mac/Firefox 146.48 111.65 34.83 (24%) 276.00
Mac/Safari 146.48 113.49 33.00 (23%) 182.00
Win/Chrome 146.48 124.68 21.81 (15%) 151.00
Win/Firefox 146.48 136.59 9.89 (7%) 222.00
(memory-data ratio=100%), WebANNS has a competitive performance
toWebANNS -Base.
Heuristic cache size optimization. With𝑝=0.8and𝑇𝜃=100ms,
as shown in Table 3, WebANNS adaptively minimizes memory usage
based on device and browser performance, saving 7%-39% memory
while maintaining query times. Note that this optimization process
only needs to run once at web app startup, because optimal memory
usage depends mainly on the runtime environment.
5 Related Work
Approximate nearest neighbor search algorithm. ANNS has
become a critical component in modern information retrieval and
AI systems, with diverse applications such as RAG and recom-
mendation systems. In general, ANNS algorithms can be catego-
rized into tree-based [ 9,23], hashing [ 21,30], quantization-based
methods [ 15,19], and graph-based [ 25]. For in-browser ANNS [ 37],
graph-based algorithms are widely adopted for their SOTA per-
formance regarding construction and query efficiency. Following
previous work, WebANNS focuses on the graph-based HNSW algo-
rithm and further optimizes the query latency at the system level.
ANNS systems. With the increasing popularity of ANNS, recent
work of ANNS systems aims to optimize performance across dif-
ferent environments, including cloud [ 36,42], edge [ 32], and web
browsers [ 5,6,37].WebANNS specifically focuses on in-browser
ANNS to enhance its integration with existing web apps and opti-
mize query latency in browsers, making in-browser ANNS practical
for more extensive datasets.
6 Conclusion
We present WebANNS , an efficient in-browser ANNS engine that ad-
dresses computational, memory, and storage access challenges spe-
cific to web browsers. By leveraging Wasm, lazy loading, and heuris-
tic memory optimization, WebANNS achieves up to two orders of
magnitude faster P99 query latency, 40% lower memory usage, and
supports over 8×larger data size compared to the SOTA engine.
WebANNS enables privacy-preserving, high-performance ANNS in
web, making in-browser ANNS practical to incorporate into real-
world web applications.
Acknowledgment
This work was supported by the National Natural Science Founda-
tion of China under the grant number 62325201.

SIGIR ’25, July 13–18, 2025, Padua, Italy Mugeng Liu et al.
References
[1]2018. Total canvas memory use exceeds the maximum limit (Safari
12). https://stackoverflow.com/questions/52532614/total-canvas-memory-use-
exceeds-the-maximum-limit-safari-12. Accessed: 2025-01-01.
[2]2024. Wikipedia (en) embedded with cohere.ai multilingual-22-12 encoder.
Cohere/wikipedia-22-12-en-embeddings. Accessed: 2025-01-01.
[3]2025. IndexedDB. https://developer.mozilla.org/en-US/docs/Web/API/
IndexedDB_API. Accessed: 2025-01-01.
[4]2025. MeMemo: RAG and Vector Search in Your Browser. https://poloclub.github.
io/mememo/. Accessed: 2025-01-01.
[5]2025. SemanticFinder: frontend-only live semantic search with transformers.js.
https://github.com/do-me/SemanticFinder. Accessed: 2025-01-01.
[6]2025. Voy: A WASM vector similarity search written in Rust. https://github.com/
tantaraio/voy. Accessed: 2025-01-01.
[7]2025. Why IndexedDB is slow and what to use instead. https://rxdb.info/slow-
indexeddb.html. Accessed: 2025-01-01.
[8]Artem Babenko and Victor S. Lempitsky. 2015. The Inverted Multi-Index. IEEE
Trans. Pattern Anal. Mach. Intell. (2015).
[9]Erik Bernhardsson. 2017. Approximate Nearest Neighbors in C++/Python opti-
mized for memory usage and loading/saving to disk. https://github.com/spotify/
annoy. Accessed: 2025-01-01.
[10] Harrison Chase. 2022. LangChain. https://github.com/langchain-ai/langchain
[11] Qi Chen, Bing Zhao, Haidong Wang, Mingqin Li, Chuanjie Liu, Zengzhong
Li, Mao Yang, and Jingdong Wang. 2021. Spann: Highly-efficient billion-scale
approximate nearest neighborhood search. Advances in Neural Information
Processing Systems (2021).
[12] Neo Christopher Chung, George Dyer, and Lennart Brocki. 2023. Chal-
lenges of large language models for mental health counseling. arXiv preprint
arXiv:2311.13857 (2023).
[13] Matthijs Douze, Alexandr Guzhva, Chengqi Deng, Jeff Johnson, Gergely Szilvasy,
Pierre-Emmanuel Mazaré, Maria Lomeli, Lucas Hosseini, and Hervé Jégou. 2024.
The faiss library. arXiv preprint arXiv:2401.08281 (2024).
[14] Fiona Draxler, Daniel Buschek, Mikke Tavast, Perttu Hämäläinen, Albrecht
Schmidt, Juhi Kulshrestha, and Robin Welsch. 2023. Gender, age, and technology
education influence the adoption and appropriation of LLMs. arXiv preprint
arXiv:2310.06556 (2023).
[15] Tiezheng Ge, Kaiming He, Qifa Ke, and Jian Sun. 2013. Optimized product
quantization. IEEE transactions on pattern analysis and machine intelligence
(TPAMI) (2013), 744–755.
[16] Samira Ghodratnama and Mehrdad Zakershahrak. 2023. Adapting LLMs for
Efficient, Personalized Information Retrieval: Methods and Implications. In Pro-
ceedings of the International Conference on Service-Oriented Computing (ICSOC
2023) . 17–26.
[17] W3C Community Group. 2025. WebAssembly. https://webassembly.org/. Ac-
cessed: 2025-01-01.
[18] Jui-Ting Huang, Ashish Sharma, Shuying Sun, Li Xia, David Zhang, Philip Pronin,
Janani Padmanabhan, Giuseppe Ottaviano, and Linjun Yang. 2020. Embedding-
based retrieval in facebook search. In Proceedings of the 26th ACM SIGKDD
International Conference on Knowledge Discovery & Data Mining (KDD 2020) .
2553–2561.
[19] Herve Jegou, Matthijs Douze, and Cordelia Schmid. 2010. Product quantization
for nearest neighbor search. IEEE transactions on pattern analysis and machine
intelligence (TPAMI) (2010), 117–128.
[20] Chao Jin, Zili Zhang, Xuanlin Jiang, Fangyue Liu, Xin Liu, Xuanzhe Liu, and Xin
Jin. 2024. RAGCache: Efficient Knowledge Caching for Retrieval-Augmented
Generation. arXiv preprint arXiv:2404.12457 (2024).
[21] Jeff Johnson, Matthijs Douze, and Hervé Jégou. 2019. Billion-scale similarity
search with GPUs. IEEE Transactions on Big Data (2019), 535–547.
[22] Andrew Kane. 2021. Pgvector: Open-source Vector Similarity Search for Postgres.
https://github.com/pgvector/pgvector.
[23] Haitao Li, Qingyao Ai, Jingtao Zhan, Jiaxin Mao, Yiqun Liu, Zheng Liu, and Zhao
Cao. 2023. Constructing tree-based index for efficient and effective dense retrieval.
InProceedings of the 46th International ACM SIGIR Conference on Research and
Development in Information Retrieval (SIGIR 2023) . 131–140.
[24] Jimmy Lin, Xueguang Ma, Sheng-Chieh Lin, Jheng-Hong Yang, Ronak Pradeep,
and Rodrigo Nogueira. 2021. Pyserini: A Python Toolkit for Reproducible Infor-
mation Retrieval Research with Sparse and Dense Representations. In Proceedings
of the 44th International ACM SIGIR Conference on Research and Development in
Information Retrieval (SIGIR 2021) . 2356–2362.[25] Yu A. Malkov and D. A. Yashunin. 2020. Efficient and Robust Approximate
Nearest Neighbor Search Using Hierarchical Navigable Small World Graphs.
IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI) (2020),
824–836.
[26] mdn web docs. 2024. Service Worker API. https://developer.mozilla.org/en-
US/docs/Web/API/Service_Worker_API. Accessed: 2025-01-01.
[27] mdn web docs. 2025. Storage quotas and eviction criteria. https:
//developer.mozilla.org/en-US/docs/Web/API/Storage_API/Storage_quotas_
and_eviction_criteria. Accessed: 2025-01-01.
[28] mdn web docs. 2025. Wasm currently only allows 32-bit addressing.
https://developer.mozilla.org/en-US/docs/WebAssembly/JavaScript_interface/
Memory/Memory. Accessed: 2025-01-01.
[29] OpenAI. 2025. ChatGPT. https://chatgpt.com/. Accessed: 2025-01-01.
[30] Ninh Pham and Tao Liu. 2022. Falconn++: A locality-sensitive filtering approach
for approximate nearest neighbor search. Proceedings of the Advances in Neural
Information Processing Systems (NeurIPS 2022) (2022), 31186–31198.
[31] LinQ AI Research. 2024. FinDER: Financial Document Retrieval Dataset. https:
//huggingface.co/datasets/Linq-AI-Research/FinanceRAG. Accessed: 2025-01-01.
[32] Korakit Seemakhupt, Sihang Liu, and Samira Khan. 2024. EdgeRAG: Online-
Indexed RAG for Edge Devices. arXiv preprint arXiv:2412.21023 (2024).
[33] Sina Semnani, Violet Yao, Heidi Zhang, and Monica Lam. 2023. WikiChat: Stop-
ping the Hallucination of Large Language Model Chatbots by Few-Shot Ground-
ing on Wikipedia. In Findings of the Association for Computational Linguistics:
EMNLP 2023 . 2387–2413.
[34] Shamane Siriwardhana, Rivindu Weerasekera, Elliott Wen, Tharindu Kalu-
arachchi, Rajib Rana, and Suranga Nanayakkara. 2023. Improving the domain
adaptation of retrieval augmented generation (RAG) models for open domain
question answering. Transactions of the Association for Computational Linguistics
(TACL) (2023), 1–17.
[35] Yukihiro Tagami. 2017. Annexml: Approximate nearest neighbor search for
extreme multi-label classification. In Proceedings of the 23rd ACM SIGKDD inter-
national conference on knowledge discovery and data mining (KDD 2017) . 455–464.
[36] Bing Tian, Haikun Liu, Yuhang Tang, Shihai Xiao, Zhuohui Duan, Xiaofei Liao,
Xuecang Zhang, Junhua Zhu, and Yu Zhang. 2024. FusionANNS: An Efficient
CPU/GPU Cooperative Processing Architecture for Billion-scale Approximate
Nearest Neighbor Search. arXiv preprint arXiv:2409.16576 (2024).
[37] Zijie J. Wang and Duen Horng Chau. 2024. MeMemo: On-device Retrieval
Augmentation for Private and Personalized Text Generation. In Proceedings of
the 47th International ACM SIGIR Conference on Research and Development in
Information Retrieval (SIGIR 2024) . 2765–2770.
[38] Lukas Wutschitz, Boris Köpf, Andrew Paverd, Saravan Rajmohan, Ahmed Salem,
Shruti Tople, Santiago Zanella-Béguelin, Menglin Xia, and Victor Rühle. 2023.
Rethinking privacy in machine learning pipelines from an information flow
control perspective. arXiv preprint arXiv:2311.15792 (2023).
[39] JD Zamfirescu-Pereira, Richmond Y Wong, Bjoern Hartmann, and Qian Yang.
2023. Why Johnny can’t prompt: how non-AI experts try (and fail) to design
LLM prompts. In Proceedings of the 2023 CHI Conference on Human Factors in
Computing Systems (CHI 2023) . 1–21.
[40] Jianjin Zhang, Zheng Liu, Weihao Han, Shitao Xiao, Ruicheng Zheng, Yingxia
Shao, Hao Sun, Hanqing Zhu, Premkumar Srinivasan, Weiwei Deng, et al. 2022.
Uni-retriever: Towards learning the unified embedding based retriever in bing
sponsored search. In Proceedings of the 28th ACM SIGKDD Conference on Knowl-
edge Discovery and Data Mining (KDD 2022) . 4493–4501.
[41] Yanhao Zhang, Pan Pan, Yun Zheng, Kang Zhao, Yingya Zhang, Xiaofeng Ren,
and Rong Jin. 2018. Visual search at alibaba. In Proceedings of the 24th ACM
SIGKDD international conference on knowledge discovery & data mining (KDD
2018) . 993–1001.
[42] Zili Zhang, Chao Jin, Linpeng Tang, Xuanzhe Liu, and Xin Jin. 2023. Fast, Ap-
proximate Vector Queries on Very Large Unstructured Datasets. In 20th USENIX
Symposium on Networked Systems Design and Implementation (NSDI 23) . 995–
1011.
[43] Zili Zhang, Fangyue Liu, Gang Huang, Xuanzhe Liu, and Xin Jin. 2024. Fast
Vector Query Processing for Large Datasets Beyond GPU Memory with Re-
ordered Pipelining. In 21st USENIX Symposium on Networked Systems Design and
Implementation, NSDI 2024, Santa Clara, CA, April 15-17, 2024 . 23–40.
[44] Shuyan Zhou, Uri Alon, Frank F Xu, Zhengbao Jiang, and Graham Neubig. 2023.
DocPrompting: Generating Code by Retrieving the Docs. In Proceedings of the
Eleventh International Conference on Learning Representations (ICLR 2023) .