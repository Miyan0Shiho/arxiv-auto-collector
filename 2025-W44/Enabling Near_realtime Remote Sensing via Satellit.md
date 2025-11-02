# Enabling Near-realtime Remote Sensing via Satellite-Ground Collaboration of Large Vision-Language Models

**Authors**: Zihan Li, Jiahao Yang, Yuxin Zhang, Zhe Chen, Yue Gao

**Published**: 2025-10-28 09:48:26

**PDF URL**: [http://arxiv.org/pdf/2510.24242v1](http://arxiv.org/pdf/2510.24242v1)

## Abstract
Large vision-language models (LVLMs) have recently demonstrated great
potential in remote sensing (RS) tasks (e.g., disaster monitoring) conducted by
low Earth orbit (LEO) satellites. However, their deployment in real-world LEO
satellite systems remains largely unexplored, hindered by limited onboard
computing resources and brief satellite-ground contacts. We propose Grace, a
satellite-ground collaborative system designed for near-realtime LVLM inference
in RS tasks. Accordingly, we deploy compact LVLM on satellites for realtime
inference, but larger ones on ground stations (GSs) to guarantee end-to-end
performance. Grace is comprised of two main phases that are asynchronous
satellite-GS Retrieval-Augmented Generation (RAG), and a task dispatch
algorithm. Firstly, we still the knowledge archive of GS RAG to satellite
archive with tailored adaptive update algorithm during limited satellite-ground
data exchange period. Secondly, propose a confidence-based test algorithm that
either processes the task onboard the satellite or offloads it to the GS.
Extensive experiments based on real-world satellite orbital data show that
Grace reduces the average latency by 76-95% compared to state-of-the-art
methods, without compromising inference accuracy.

## Full Text


<!-- PDF content starts -->

Enabling Near-realtime Remote Sensing via Satellite–Ground
Collaboration of Large Vision–Language Models
Zihan Li
Fudan University
Shanghai, ChinaJiahao Yang
Fudan University
Shanghai, ChinaYuxin Zhang
Fudan University
Shanghai, China
Zhe Chen
Fudan University
Shanghai, ChinaYue Gao
Fudan University
Shanghai, China
Abstract
Large vision–language models (LVLMs) have recently demonstrated
great potential in remote sensing (RS) tasks (e.g., disaster monitor-
ing) conducted by low Earth orbit (LEO) satellites. However, their de-
ployment in real-world LEO satellite systems remains largely unex-
plored, hindered by limited onboard computing resources and brief
satellite–ground contacts. We propose Grace, a satellite–ground
collaborative system designed for near-realtime LVLM inference in
RS tasks. Accordingly, we deploy compact LVLMs on satellites for
realtime inference, but larger ones on ground stations (GSs) to guar-
antee end-to-end performance. Grace is comprised of two main
phases that are asynchronous satellite-GS Retrieval-Augmented
Generation (RAG), and a task dispatch algorithm. Firstly, we distill
the knowledge archive of GS RAG to satellite archive with tailored
adaptive update algorithm during limited satellite-ground data ex-
change period. Secondly, propose a confidence-based test algorithm
that either processes the task onboard the satellite or offloads it to
the GS. Extensive experiments based on real-world satellite orbital
data show that Grace reduces the average latency by 76–95% com-
pared to state-of-the-art methods, without compromising inference
accuracy.
CCS Concepts
•Human-centered computing →Ubiquitous and mobile com-
puting;•Computing methodologies→Machine learning.
Keywords
LEO satellite networks, Earth observation, Large vision-language
models.
ACM Reference Format:
Zihan Li, Jiahao Yang, Yuxin Zhang, Zhe Chen, and Yue Gao. 2018. Enabling
Near-realtime Remote Sensing via Satellite–Ground Collaboration of Large
Vision–Language Models. InProceedings of Make sure to enter the correct
conference title from your rights confirmation email (Conference acronym ’XX).
ACM, New York, NY, USA, 15 pages. https://doi.org/XXXXXXX.XXXXXXX
Permission to make digital or hard copies of all or part of this work for personal or
classroom use is granted without fee provided that copies are not made or distributed
for profit or commercial advantage and that copies bear this notice and the full citation
on the first page. Copyrights for components of this work owned by others than the
author(s) must be honored. Abstracting with credit is permitted. To copy otherwise, or
republish, to post on servers or to redistribute to lists, requires prior specific permission
and/or a fee. Request permissions from permissions@acm.org.
Conference acronym ’XX, Woodstock, NY
©2018 Copyright held by the owner/author(s). Publication rights licensed to ACM.
ACM ISBN 978-1-4503-XXXX-X/2018/06
https://doi.org/XXXXXXX.XXXXXXX
i) Limited
onboard resources
ii) Brief
Satellite-GS
ContactConnected
satellite
Non-connected  satellite
Ground
Station
Prompt :
Is ther e a fir e risk in this image?
Prompt :
Is this a rural or  an
urban area?
Response:
This is an urban ar ea.
Response:
No, ther e is no fir e risk.
Data flowContact timeGround-satelite linkLow earth orbit
Ground
LVLM
Satellite
LVLM
Figure 1: Though collaborative inference may facilitate an
efficient LVLM inference on satellites, i) limited onboard re-
sources and ii) brief satellite-GS contact still pose significant
challenges to the deployment.
1 Introduction
In recent years, advances in satellite technology and declining
launch costs have driven a rapid proliferation of Low Earth Orbit
(LEO) satellites for Remote Sensing (RS) [ 17,43,49,57,62,76,85,88].
For example, Planet has launched 452 satellites to construct the
Dove constellation, imaging more than 350 million square kilometer
every day, delivering the near-daily, high-frequency coverage [ 2,
48,56,77,78]. Positioned at altitudes of up to 2000 kilometers and
equipped with advanced sensors, LEO satellites take full advantage
of their unique vantage points to acquire wide-swath and high-
resolution images of the Earth’s surface [ 84]. Leveraging AI model
for information analysis, the vast volume of satellite data can be
processed efficiently, especially, latency-sensitive applications such
as forest fire detection [ 11], maritime surveillance [ 19], and urban
traffic monitoring [8, 26, 67, 74].
However, current latency-sensitive RS tasks primarily rely on
traditional small models (e.g., CNN [ 35] and ViT [ 69]), whose limi-
tations are becoming increasingly evident: 1) Small models require
task-specific architectural design and parameter training, signifi-
cantly limiting their adaptability across diverse RS applications. 2)
With only tens of millions of parameters, they struggle to capturearXiv:2510.24242v1  [cs.NI]  28 Oct 2025

Conference acronym ’XX, June 03–05, 2018, Woodstock, NY Zihan Li, Jiahao Yang, Yuxin Zhang, Zhe Chen, and Yue Gao
the complex features inherent in satellite data, resulting in sub-
optimal performance [ 34,73]. Fortunately, recent breakthroughs
in Large Vision-Language Models (LVLMs) have overcome these
limitations, revealing transformative potential for RS tasks [ 53,81].
By integrating visual and language understanding within a unified
architecture, LVLMs achieve a “one-model-for-all-tasks” paradigm,
thereby offering broad adaptability [ 28,32,33,58,70]. Besides, with
typically over a billion parameters, LVLMs show strong capabilities
in fitting and generalizing for satellite data, significantly enhancing
RS performance [40].
Challenges.Despite their advantages, the heavy computational
demands of LVLMs present a core challenge:How to deploy them
efficiently within the strict resource limits of LEO satellite systems,
constrained by two fundamental factors.First,LEO satellites must
prioritize onboard payload and energy for control systems, remote
image acquisition, communicating with GSs, and other critical func-
tions. This allocation leaves insufficient capacity to support large-
scale LVLMs [ 15,37–39,82], resulting in critically constrained com-
putational resources. While modern LEO satellites are equipped
with some edge computing platforms (like NVIDIA Jetson) to en-
able AI workloads [ 55], these embedded systems have significant
gaps in computing capability and memory resources compared to
ground-based server clusters — a significant limitation given the re-
source demands of LVLMs [ 14,29,64].Second,the unique network
conditions of LEO satellites, which move at the orbital velocity that
results in highly intermittent communication windows with ground
stations [ 6]. During these long periods of disconnection, satellites
must rely solely on their limited onboard processing capabilities
for real-time data handling [ 43,49,80,83]. The satellite-ground
communication bandwidth also presents a critical bottleneck. Due
to the inherent temporal constraints of satellite-ground links, satel-
lites accumulate substantial volumes of remote sensing imagery
during disconnection periods [ 59,61,84]. For high-resolution Earth
observation systems, this results in persistent data backlogs that
cannot be fully transmitted during brief communication windows.
The resulting constrained data transmission capacity limits the
system’s ability to perform timely processing and analysis [68].
Existing LVLM deployments in satellite systems fall short of
addressing these challenges and can be broadly categorized into
two types: 1) Ground station-centric deployment schemes [ 36,81],
while free from onboard computational limitations and able to
leverage the powerful computing resources on the ground, are con-
strained by the bandwidth limitation and intermittent connectivity,
making it ineffective to transmit large volumes of satellite data
efficiently. The latency from data acquisition to analysis results is
prolonged, failing to meet the demands for efficient inference; 2)
Conversely, onboard deployment schemes [ 58,64], although reduc-
ing dependency on network bandwidth, are limited by the scarce
computational resources on satellites, preventing the execution
of large-scale LVLMs and thereby restricting intelligent inference
capabilities. Collaborative inference [ 71], a paradigm that partitions
tasks between lightweight models on edge devices and powerful
cloud models to optimize latency and resource efficiency, faces chal-
lenges in satellite environments. Existing implementations require
sustained network connections, which are incompatible with the
intermittent connectivity characteristics of LEO satellites.Solution.To tackle the above challenges, we propose Grace
as a collaborative LEO satellite-GS LVLMs inference system for
near-realtime RS. Grace deploys compact LVLMs onboard satellites
for real-time inference, while larger models on the ground pro-
vide auxiliary support during each contact window. To overcome
the first challenge, we need to improve the accuracy of a compact
LVLM. A common approach is to construct a external knowledge
archive with Retrieval-Augmented Generation (RAG) to equip the
model with domain-specific knowledge and reduce hallucinations.
However, due to challenges mentioned earlier, it is necessary to
minimize the size of the on-satellite RAG system to fit within the
limited onboard resources. When RS missions change, efficiently
updating the content in the satellite archive to adapt to new tasks
becomes an issue need to be addressed. Therefore, we propose a
dynamic satellite-ground collaborative RAG system. Similar to the
collaborative inference system, this RAG system leverages a com-
prehensive ground archive and a streamlined satellite archive to
enable efficient LVLM inference. To ensure that the satellite archive
remains aligned with current mission requirements, we propose an
adaptive update algorithm. This algorithm enables data exchange
between satellites and GSs during their brief communication win-
dow. The GS customizes and updates the content of the satellite
archive based on queries that are difficult for the satellite to han-
dle, ensuring that the archive meets the latest mission needs. To
address the second challenge, we design a task dispatcher to reduce
data transmission requirements. The task dispatcher strategically
assigns inference tasks. With the confidence-based test, most tasks
are processed onboard, significantly reducing the number of sam-
ples that must be transmitted to the GS. The dispatcher first need
to determine whether the prior knowledge in the satellite archive
is sufficient to support answering the query. Next, the dispatcher
assess the confidence level of the satellite LVLM’s prediction. If the
prior knowledge is inadequate or the confidence in the prediction is
low, the onboard inference result should be discarded, and the query
must be transmitted to the GS for further processing. Otherwise,
the onboard inference result is accepted as the final output.
Grace maintains efficient data processing capabilities and infer-
ence performance through these measures with constrained compu-
tational resources and unstable network connectivity. In summary,
this work has the following contributions:
•To the best of our knowledge, Grace — a collaborative LEO
satellite-GS inference system — represents the first work of
near-realtime LVLM inference in LEO satellite networks.
•Grace effectively addresses the onboard computational con-
straints of LEO satellites through its implementation of a
dynamic knowledge archive, which enables adaptive content
updates.
•We develop and integrate key mechanisms, including the
task dispatcher, to optimize data handling and enhance com-
munication efficiency under constrained resources.
•Extensive evaluations demonstrate that our framework sub-
stantially outperforms other deployment baselines. Grace
reduces the average latency by 76–95% compared to state-of-
the-art methods, without compromising inference accuracy.
The structure of the paper is as follows. In Section 2, we highlight
the challenges faced by current LEO satellite networks. Section 3

Enabling Near-realtime Remote Sensing via Satellite–Ground Collaboration of Large Vision–Language Models Conference acronym ’XX, June 03–05, 2018, Woodstock, NY
outlines the system design of Grace. Section 4 describes the imple-
mentation of the system, while Section 5 provides an evaluation
of its performance. A discussion on related works is presented in
Section 6. The paper concludes with Section 7.
2 Challenges and Motivation
2.1 Limited Onboard Resources
The computational resources on LEO satellites are severely lim-
ited compared to ground stations. While the latest LEO satellites
have begun integrating edge computing platforms (like NVIDIA
Jetson) to support onboard artificial intelligence [ 55], the computa-
tional capacity of these platforms remains moderate. For example,
the computing power of the NVIDIA Jetson Orin NX is about 50
TFLOPS [ 4]. In contrast, GPUs available at ground stations, such
as the NVIDIA A100 GPU, can achieve computing power up to
312 TFLOPS [ 1]. This limitation renders LEO satellites incapable of
running large-scale LVLMs. Furthermore, onboard payloads and
energy need to be prioritized for maintaining control systems, com-
municating with ground stations, remote image acquisition, and
other critical satellite functions [ 25,60,79], leaving minimal room
for other applications, which further constrains the scale of onboard
LVLMs. According to the scaling law [ 34], the generalization capa-
bility of LVLMs directly correlates with their size. Consequently,
the inference accuracy of onboard LVLMs will inevitably under-
perform that of their larger-scale counterparts deployed on ground
stations.
To validate this motivation, we conduct experiments using the
RSVQA [ 52] and RESISC45 [ 16] with the Qwen2-VL [ 10]. We simu-
late onboard and ground-based LVLMs using 2B and 7B parameter
models, respectively. A Jetson Orin NX with 16GB memory is used
to emulate the satellite environment. Figure 2a depicts the peak
memory usage required by 2B and 7B LVLMs. The satellite environ-
ment only supports a 2B LVLM. Larger models can only be run at
the ground station. As shown in Figure 2b, for RSVQA, the 2B model
achieve an accuracy of 55.9%, while the 7B model reach 68.0%, a
12.1% gap. RESISC45 shows a similar result. This finding confirms
that relying solely on onboard models for local processing fails to
meet user requirements for inference accuracy.
2.2 Brief Satellite-GS Contact
The satellite’s data downlink capability cannot keep pace with its
data acquisition capability. Due to their low orbital altitude and
high-velocity relative to Earth’s rotation, LEO satellites experience
intermittent connectivity with ground stations, where the duration
of network availability is significantly shorter than outages. Based
on Starlink’s LEO constellation configuration [ 6], we calculate the
average contact time between LEO satellites and ground stations
at different orbital altitudes. As illustrated in Figure 2c, an LEO
satellite completes an Earth orbit in approximately 95 minutes,
with only about 5 minutes of viable communication time per pass.
This intermittency poses challenges for directly applying existing
collaborative inference frameworks designed for large language
models, as these frameworks assume persistent network connec-
tivity among computing nodes — an assumption easily satisfied
in terrestrial scenarios but invalid for LEO environments. Thus,
novel methods are required to adapt to intermittent connectivityin satellite-ground systems. Therefore, LEO satellites must rely
on their own capabilities to complete inference tasks for the vast
majority of the time.
LEO satellites are equipped with high-speed sensors capable
of collecting vast amounts of remote sensing data. Remote sens-
ing data can be acquired at any time during the satellite’s flight.
However, their data transmission bandwidth to ground stations is
disproportionately limited. For instance, WorldView-3 can acquire
up to 680,000 square kilometers per day; each 13 kilometers ×112
kilometers scene is roughly 40 Gb after compression. Its fastest
X-band channel tops out at 1200 Mbps [ 5]. If the sensor were oper-
ated at maximum duty cycle, the satellite-GS link cannot complete
the transmission of all data within the brief connectivity window.
When multiple tenants share the satellite’s bandwidth, the allo-
cation per user becomes even more constrained. Ground-centric
inference approaches face severe bottlenecks in such scenarios due
to insufficient data throughput [54, 63].
We simulate a scenario using the high-resolution RSVQA dataset
to demonstrate the impact of bandwidth limitations. Assuming a
LEO satellite captures one image every 2 seconds and transmits
data compressed byzlibto the ground at 30 Mbps during 3-second
intervals every minute, we measured the latency between image
acquisition and ground processing over time. As shown in Figure
2d, images captured later experience significantly longer transmis-
sion delays, with both average and maximum latency exhibiting
an upward trend. This occurs because the transmission rate can-
not keep pace with data collection, leading to an accumulation
of onboard data and progressively worsening processing delays.
Such bottlenecks critically degrade the efficiency of remote sensing
inference tasks.
3 System Design
3.1 Overview
In this section, we present Grace, a satellite-ground collaborative
LVLM inference system, as illustrated in Figure 3. In our setting,
each query𝑄comprises a remote sensing image 𝑄𝑀and a natural
language instruction 𝑄𝐼, i.e.,𝑄={𝑄𝑀,𝑄𝐼}. Grace aims to interpret
the image𝑄𝑀according to the instruction 𝑄𝐼, delivering accurate
and efficient inference within the satellite’s resource constraints.
To address the challenges introduced in Section 2, We will provide
a detailed introduction to Grace through the following sections:
•Theground archive(Section 3.2) is deployed to equip the
LVLM with domain-specific knowledge and enhance its in-
ference capabilities. This section introduces the data struc-
ture and the retrieval scheme of the archive.
•Thesatellite archive(Section 3.3) is integrated with dynamic
adaptation algorithm, which allows the satellite archive to
automatically update its content in response to changes in
mission requirements and RS image content, ensuring adapt-
ability to new scenarios.
•To reduce data transmission requirements, atask dispatcher
(Section 3.4) strategically assigns inference tasks. Most tasks
are processed onboard, significantly reducing the number of
samples that must be transmitted to the ground station.
•Leveraging the disparity in computational resources, a high-
performance LVLM with strong generalization capabilities

Conference acronym ’XX, June 03–05, 2018, Woodstock, NY Zihan Li, Jiahao Yang, Yuxin Zhang, Zhe Chen, and Yue Gao
Onboard limitation
RSVQA RESISC45
Dataset02550 Memory (GB)2B 7B
(a) Peak memory usage.
RSVQA RESISC45
Dataset45556575 Accuracy (%)2B 7B (b) Accuracy.
470 547 580
Altitude (km)00.81.62.4 Time (h)Non-contact time
Contact time (c) Contact time.
0 500 1000 1500 2000
Image Index0510152025 Latency (min)Per Sample
Avg. Cumulative (d) Transmission.
Figure 2: The the limited onboard resources (a) restrict the onboard generalization (b). The intermittent connection (c) and the
limited transmission rate become a main bottleneck of task latency (d).
[Satellite Pr ediction]
[Ground Pr ediction]
[RS Image]
Query
Is a
residential
building
present?
[Instruction ]
Ground Station Workflow (Sec. 3.5)Satellite Workflow
Satellite Imagery QueryRelevant content
Processing pipeline
Prediction r esult
Satellite-GS link
Ground
Archive
(Sec. 3.2)Ground
LVLM
Satellite
Archive
(Sec. 3.3)Matching
Test
(Sec. 3.4.1)Satellite
LVLM
Cognitive
Test
(Sec. 3.4.2)
Figure 3: Grace Overview. On the above, the satellite is equipped with a compact LVLM and a lightweight archive to process
queries using onboard resources. On the below, the ground station features a comprehensive archive and a powerful LVLM,
processing buffered queries and updating the satellite archive. “Sec.” indicates “Section”.
is deployed at theground station(Section 3.5), providing
enhanced support for complex tasks.
3.2 Ground Archive
A specially constructed external knowledge archive is critical to
flexibly update the external knowledge for LVLMs and enhance
inference accuracy. In our system, the LEO satellite and the ground
station maintain an archive as an external knowledge repository for
the LVLM. We will introduce the fetching method and the search
result for our archive in this Section. Then we will describe the
corresponding adoptive update algorithm at Section 3.3.
3.2.1 Fetching Method.The archive stores a large collection of
remote sensing images {𝑅𝑀1,𝑅𝑀2,𝑅𝑀3,...} . Formally, for the 𝑗-th
remote sensing image 𝑅𝑀𝑗, there is a set of instruction-answer pairs
{⟨𝑅𝐼𝑗,1,𝑅𝐴𝑗,1⟩,⟨𝑅𝐼𝑗,2,𝑅𝐴𝑗,2⟩,⟨𝑅𝐼𝑗,3,𝑅𝐴𝑗,3⟩,...} . Each ground-truth has
been carefully annotated by human experts. The data structure can
be formulated as Equation 1.
 
𝑅𝑀1:
⟨𝑅𝐼1,1,𝑅𝐴1,1⟩,⟨𝑅𝐼1,2,𝑅𝐴1,2⟩,...	
𝑅𝑀2:
⟨𝑅𝐼2,1,𝑅𝐴2,1⟩,⟨𝑅𝐼2,2,𝑅𝐴2,2⟩,...	
𝑅𝑀3:
⟨𝑅𝐼3,1,𝑅𝐴3,1⟩,⟨𝑅𝐼3,2,𝑅𝐴3,2⟩,...	
... 
.(1)
The objective of the archive is to retrieve the content most relevant
to a query𝑄. Notably, each query 𝑄is multi-modal, comprising anewly captured satellite image 𝑄𝑀and a per-defined instruction 𝑄𝐼.
Therefore, we employ separate modules for vision and text queries
that jointly identify relevant content.
Vision Query Module.The vision query module uses a vision
embedding model 𝐸𝑣to convert images into feature vectors. Specifi-
cally, every remote sensing image 𝑅𝑀𝑗in the knowledge base is em-
bedded and normalized to a unit vector: 𝑒𝑅
𝑀𝑗=𝐸𝑣(𝑅𝑀𝑗)/∥𝐸𝑣(𝑅𝑀𝑗)∥.
These normalized vectors are stacked to form a matrix:
𝐴𝑀=𝑒𝑅
𝑀1,𝑒𝑅
𝑀2,𝑒𝑅
𝑀3,...
,(2)
which is stored in the archive for subsequent retrieval.
When a query image 𝑄𝑀arrives, the embedding model 𝐸𝑣pro-
duces a normalized query vector 𝑒𝑄
𝑀=𝐸𝑣(𝑄𝑀)/∥𝐸𝑣(𝑄𝑀)∥.We
then compute the cosine similarity of 𝑒𝑄
𝑀with each column in 𝐴𝑀:
Sim𝑀=𝐴𝑀𝑇·𝑒𝑄
𝑀,(3)
yielding similarity scores between𝑀 𝑄and stored images.
Instruction Query Module.Natural-language instructions in the
knowledge base are processed in a similar fashion using a text em-
bedding model 𝐸𝑡. However, we first de-duplicate the instructions
to minimize computation and storage, as different images might
share the same instruction. After removing duplicates, we obtain
a unique instruction set{𝑅 𝐼1,𝑅𝐼2,𝑅𝐼3,...}and maintain a mapping

Enabling Near-realtime Remote Sensing via Satellite–Ground Collaboration of Large Vision–Language Models Conference acronym ’XX, June 03–05, 2018, Woodstock, NY
[Ground Pr ediction]Ground
LVLM
[RS Image]
Is a
residential
building
present?
[Instruction ]Sear ch Result
<"Is there a residential
building? ", "no">
<"Is there a residential 
building? ", "yes"><"What is the amount of re-
sidential buildings? ", "0">Ground
Archive
Figure 4: Ground-station inference process. Relevant data
for the query is retrieved by the ground archive. The ground
LVLM takes both the query and the search result as input to
generate the inference outcome.
function𝐹that memories which instructions belong to which im-
ages. “𝑗∈𝐹(𝑖) ” indicates that the instruction set corresponding to
the j-th image𝑅 𝑀𝑗contains the instruction𝑅 𝐼𝑖.
Next, each unique instruction 𝑅𝐼𝑗is embedded and normalized:
𝑒𝑅
𝐼𝑗=𝐸𝑡(𝑅𝐼𝑗)/∥𝐸𝑡(𝑅𝐼𝑗)∥.These vectors form a matrix 𝐴𝐼like𝐴𝑀in
Equation 2. For the query instruction 𝑄𝐼,𝐸𝑡generates a normalized
embedding𝑒𝑄
𝐼=𝐸𝑡(𝑄𝐼)/∥𝐸𝑡(𝑄𝐼)∥.We then compute the similarity
vector:
Sim𝐼=𝐴𝐼𝑇·𝑒𝑄
𝐼,(4)
where each element indicates how well 𝐼𝑄matches each instruction
in the knowledge base.
Data Fusion Module.Although one image may have multiple
instructions, including all of them in the prompt can overwhelm
the LVLM and degrade inference performance. Excessive, irrelevant
instructions can increase computation and dilute the essential con-
text needed by the LVLM [ 9]. To balance coverage and relevance,
the data fusion module aggregates Sim𝑀andSim𝐼to produce the
integrated retrieval results. For every remote sensing image 𝑅𝑀𝑖in
the archive, the data fusion module consults the mapping function
𝐹to find the instruction with the highest similarity to𝑄 𝐼:
Sim𝑖=Sim𝑀𝑖+max
𝑗∈𝐹(𝑖)Sim𝐼𝑗.(5)
The data fusion module picks the single relevant instruction from
the archive that best aligns with 𝑄𝐼. Finally, it returns the top- 𝐾
relevant records.
3.2.2 Search Result Structure.The retrieved relevant content 𝐶is
a set of records{𝐶1,𝐶2,...} . The𝑖-th record𝐶𝑖contains following
items:
•A relevant remote sensing image 𝐶𝑖𝑀from the archive that is
similar to the query image 𝑄𝑀and a corresponding similarity
score𝑆𝑖𝑀measuring the imaginary relevance.
•A relevant instruction 𝐶𝑖𝐼closely related to the query instruc-
tion𝑄𝐼and a corresponding similarity score 𝑆𝑖𝐼measuring
the textual relevance.
•A human-annotated ground-truth answer 𝐶𝑖𝐺for{𝐶𝑖𝑀,𝐶𝑖𝐼}.
This ground-truth helps guide the LVLM by showing the
appropriate expected answer for tasks similar to the query.Figure 4 presents an example of the search process. Each row in
the “Search Result” retrieved from the archive represents a record
𝐶𝑖, which contains an image 𝑅𝑖𝑀, an instruction 𝑅𝑖𝐼, and a ground-
truth𝑅𝑖𝐺. The search result 𝐶, along with the query, will be fed
into the LVLM for inference to generate the final answer.
3.3 Satellite Archive
The LEO satellite archive must be highly optimized for the satellite’s
resource-constrained environment. In practice, the LEO satellite
archive is a subset of the ground archive that includes only the data
most relevant to the satellite’s current tasks.
The satellite archive can provide sufficient relevant information
to the satellite LVLM, even with limited content size, for two pri-
mary reasons: 1) Predictable orbital path: The operational orbit of
satellites is fixed and predictable, which limits the geographical
locations of captured images. It is possible to know in advance what
locations can be imaged at certain time. This predictability ensures
that the archive contains a structured and manageable set of data
related to specific geographic areas. 2) Temporal stability of land:
Most terrestrial features are stable over time. Urban infrastructures
and agricultural landscapes typically undergo minimal changes
over periods of multiple days. This slow-changing characteristic
of land dynamics means that images of the same location often
maintain consistent content over time. As a result, even a limited
archive can offer valuable and relevant knowledge for inference pur-
poses without needing extensive updates or large volumes of new
imagery. Consequently, a carefully chosen subset of the ground
archive can support stable onboard inference while minimizing
storage overhead.
However, determining which portion of the ground archive to up-
load to the satellite is non-trivial. Moreover, if the satellite’s mission
scope changes over time, the archive must also adapt. To address
these challenges, we design a dynamic adaptation algorithm. This
algorithm consists of two components: a replace module running
in the satellite archive and a hierarchical transmission mechanism.
3.3.1 Replace Module.The replace module tracks the usage pat-
terns of each remote-sensing image in the satellite archive. Every
time a query is processed onboard, the top- 𝐾retrieved images
are moved to the front of a queue, indicating that they have been
recently accessed. Conversely, images that seldom appear in the
results move toward the back of this queue over time.
When the archive requires an update, the replace module re-
moves entries from the tail of the queue (those least recently used).
These outdated or unused items are discarded to free space. Newly
introduced images and instructions are then loaded into the archive
and placed at the front of the queue, ensuring that high-demand
data are always prioritized.
3.3.2 Hierarchical Transmission Mechanism.LEO satellites move
at high speed relative to the ground, resulting in intermittent and
brief communication windows between the satellite and the ground
station. On average, each window lasts only about five minutes,
followed by roughly ninety minutes of no connectivity. During
these brief windows, the satellite can transmit tasks accumulated in
its cache buffer to the ground station. Simultaneously, the ground
station processes these queries using its comprehensive archive

Conference acronym ’XX, June 03–05, 2018, Woodstock, NY Zihan Li, Jiahao Yang, Yuxin Zhang, Zhe Chen, and Yue Gao
[Satellite Pr ediction]
Transmission Buffer
Priority QueueSecondary  Queue
Task Queue[Ground Pr ediction]
Satellite
ArchiveGround
Archive
[RS Image]
Query
Is a
residential
building
present?
[Instruction ]
Satellite
Archive
Satellite
LVLM
Ground
Archive
Ground
LVLM
Figure 5: The hierarchical transmission mechanism. The LEO satellite side (left) and the ground station side (right). The priority
queue will be transmitted to the ground station first, and its retrieval results will be sent back for updating the satellite archive.
The secondary queue will be transmitted after the completion of the priority queue transmission.
and sends back relevant content to update the satellite archive.
However, the volume of data transmitted from the ground to the
satellite is constrained by the limited bandwidth, which is typically
smaller than the reverse direction [ 48]. This asymmetry necessitates
a mechanism to prioritize critical data transfers during the short
communication windows.
To address this challenge, we introduce a hierarchical trans-
mission mechanism based on the operational characteristics of
LEO satellites. The satellite operates over prolonged durations, dur-
ing which its transmission cache may accumulate a large number
of queries pending ground-based inference. Notably, newly ap-
pended queries in the buffer correspond to recent tasks that the
satellite could not resolve locally. Prioritizing these recent queries
for archive updates can reduce the volume of data transmitted from
the ground to the satellite. The ground station only needs to trans-
mit the relevant content for the priority queue, rather than for all
accumulated queries. The volume of data transmitted from the satel-
lite to the ground remains unaffected, as all queries must eventually
be processed for inference. Based on this rationale, we split the
transmission buffer into two queues: 1) a priority queue containing
the most recent 𝑁mpqueries, and 2) a secondary queue holding
all other queries. This design optimizes the use of the asymmetric
communication channel, ensuring efficient utilization of the limited
ground-to-satellite bandwidth while maintaining the integrity of
the inference pipeline.
Priority Queue.The priority queue is given preferential access to
communication resources on the satellite and ground station. Specif-
ically, 1) on the satellite: The priority queue must finish sending its
queries before any queries from the secondary queue can be trans-
mitted. 2) On the ground station: The priority queue’s queries must
be processed first before the ground station handles any secondary
queue requests.
When a communication window opens, the satellite transmits
all queries in the priority queue to the ground station. Suppose the
priority queue contains 𝑁priorqueries (𝑁prior≤𝑁 mp). The ground
station uses its ground archive to retrieve relevant content for each
query, producing a total of 𝐾×𝑁 priorrelevant records. After de-
duplication, the ground station sends metadata (including remote
sensing image IDs) for these records to the satellite. The satellite
compares incoming IDs against its onboard archive. It identifies
any missing data and sends a request back to the ground stationfor those missing content. The ground station then transmits all
missing images, corresponding instructions, and ground-truths to
the satellite, which inserts the newly received data into its satellite
archive. This process ensures that the satellite archive is updated
with the most relevant information for current tasks.
Secondary Queue.Queries in the secondary queue are generally
older and are therefore addressed only after the priority queue
has been fully processed. Secondary queue transmission begins
when the satellite confirms that all priority queue transmissions are
complete. The secondary queue is transmitted in chunks because
satellite communication links can fail unexpectedly. In contrast to
the priority queue, the ground station does not immediately process
these secondary queries. Instead, it defers them until no further
priority tasks remain. At this point, it retrieves the relevant records
from the ground archive only for local inference. Unlike priority
queue updates, the ground station does not return relevant content
to the satellite. Once the ground station receives the previous chunk
of data, the satellite removes the delivered queries from the local
secondary queue and proceeds to transmit the next batch of queries.
Figure 5 shows the task flow between the LEO satellite and the
ground station.
3.4 Task Dispatcher
When an LEO satellite captures a remote sensing image 𝑄𝑀, the
onboard inference module is immediately activated to process it ac-
cording to a per-defined instruction 𝑄𝐼. The satellite archive queries
to retrieve relevant content 𝐶that corresponds to 𝑄. However, due
to limited computing and storage resources, the satellite archive
and the LVLM are constrained in size. Consequently, the onboard
inference may not always provide correct answers. Therefore, we
propose a task dispatch algorithm to decide which tasks should be
kept locally for inference and which tasks cannot be completed on
the satellite and must be sent to the ground station for processing.
The core requirement of the dispatch algorithm is to assess the
confidence level of the onboard inference system in accurately
completing the inference task. If the confidence level is sufficiently
high, the inference can be completed on the satellite; otherwise,
the raw image needs to be transmitted to the ground station for
further processing. The task dispatch consists of two stages. The
first stage is a matching test conducted before inference, and the
second stage is a cognitive test carried out after inference. If either

Enabling Near-realtime Remote Sensing via Satellite–Ground Collaboration of Large Vision–Language Models Conference acronym ’XX, June 03–05, 2018, Woodstock, NY
Algorithm 1Task Dispatcher
Require: Query𝑄={𝑄𝑀,𝑄𝐼}, relevant content 𝐶, satellite LVLM
𝑀𝑠𝑎𝑡, thresholds𝑇 𝑀,𝑇𝐼,𝑇𝐾,𝑇Conf.
Ensure: Decision: Accept the onboard inference result or transmit
to ground station.
1:functionDispatch(𝑄)
2:Stage 1: Matching Test
3:𝐶′←∅
4:foreach record𝑅 𝑖∈𝐶do
5:if𝑆 𝑖𝑀≥𝑇𝑀and𝑆𝑖𝐼≥𝑇𝐼then
6:𝐶′←𝐶′∪{𝑅𝑖}
7:end if
8:end for
9:if|𝐶′|<𝑇𝐾then⊲Insufficient relevant records
10:Cache𝑄for transmission
11:returnTransmit
12:end if
13:Pred,Prob←𝑀 𝑠𝑎𝑡(𝑄,𝐶′)
14:Stage 2: Cognitive Test
15:Conf←0
16:for𝑖←𝑁 inpto𝑁 inp+𝑁 gendo
17:Conf←Conf+ln(Prob 𝑖)
18:end for
19:Conf←exp Conf/𝑁 gen
20:ifConf<𝑇 Confthen⊲Insufficient confidence
21:Cache𝑄for transmission
22:returnTransmit
23:else
24:returnAccept⊲Accept onboard inference result
25:end if
26:end function
test is not passed, the query must be sent to the ground station
for processing. Otherwise, the inference can be completed on the
satellite. Algorithm 1 shows the entire dispatching procedure.
3.4.1 Matching Test.Before inference begins, the dispatcher con-
ducts a matching test to evaluate whether the retrieved relevant
content𝐶adequately supports the inference. The matching test
proceeds in two steps.
1) The dispatcher filters out low-relevance records from the re-
trieved relevant content. Each relevant record 𝑅𝑖is associated with
two similarity scores: the image similarity 𝑆𝑖𝑀and the instruction
similarity𝑆𝑖𝐼. We define two thresholds, 𝑇𝑀for image similarity and
𝑇𝐼for instruction similarity. Any record whose image or instruction
similarity falls below the corresponding threshold is discarded from
the relevant content.
2) The dispatcher checks whether the remaining relevant con-
tent𝐶′has at least 𝑇𝐾records (where 𝑇𝐾≤𝐾). This lower bound
ensures that the LVLM has enough context to complete a reliable
inference. If|𝐶′|<𝑇𝐾, we conclude that the satellite archive lacks
the necessary information for this query. In that case, the query
𝑄is cached for transmission to the ground station, where a more
comprehensive archive and a more capable LVLM can handle the
inference task.
Emulated satellite Emulated GSEmulated LEO Satellite Network System
Satellite
orbital
data 
(TLE)
H3C UniServer  R5300 G3
 NVIDIA  Jetson Orin NXtraffic
contr ol
Figure 6: Grace testbed.
If the filtered relevant content 𝐶′passes the matching test, it
proceeds to the satellite inference pipeline. The content in 𝐶′is
integrated with the task 𝑄during the prompt assembly. Lines 2 to
13 of Algorithm 1 illustrate the procedure of the matching test.
3.4.2 Cognitive Test.Although relevant content helps compensate
for the LVLM’s limited model size and generalization capabilities
onboard the satellite, the system still cannot fully trust the final
result without an additional verification step. We introduce a cog-
nitive test to assess the credibility of the LVLM’s answer.
During LVLM inference, we track each newly generated token 𝑥𝑖
and record its probability 𝑃(𝑥𝑖|𝑥<𝑖), where𝑥<𝑖={𝑥 0,𝑥1,...,𝑥𝑖−1}.
A high probability implies that the LVLM is confident about the
token choice. Since the final output consists of multiple tokens,
we compute the geometric mean of the probabilities of all newly
generated tokens as an overall confidence score:
Conf=exp 
1
𝑁gen𝑁inp+𝑁gen∑︁
𝑖=𝑁 inpln𝑃(𝑥𝑖|𝑥<𝑖)!
,(6)
where𝑁genis the length of the output response and 𝑁inpis the
size of input tokens. We then compare this Conf value against a
threshold𝑇Conf. IfConf<𝑇 Conf, it indicates the satellite LVLM is
not sufficiently confident in its inference. The system discards the
result and places the query 𝑄in a transmission cache to await com-
munication to the ground station. Otherwise, the result is accepted
as the final answer, and no further transmission is needed. Lines 14
to 25 of Algorithm 1 illustrate the procedure of the cognitive test.
3.5 Ground Inference
Unlike the LEO satellite inference pipeline, the ground station does
not perform a matching test on the retrieved relevant content. For
each query received — whether from the priority queue or the
secondary queue — the ground archive returns 𝐾relevant records.
All𝐾records are assumed relevant and thus automatically included
as input to the ground LVLM.

Conference acronym ’XX, June 03–05, 2018, Woodstock, NY Zihan Li, Jiahao Yang, Yuxin Zhang, Zhe Chen, and Yue Gao
LR HR
Dataset5060708090100 Accuracy (%)PETALS
Tabi
GeoChat
Grace
(a) Test accuracy.
LR HR
Dataset050100 Average Time (s)722146 (b) Average time.
LR HR
Dataset020406080 Median Time (s)884 155 (c) Median time.
LR HR
Dataset050100150 Max Time (s)1045336 (d) Max time.
Figure 7: Test accuracy and inference time on RSVQA LR and HR datasets. The numbers beside bars in the upper part of the
figure indicate actual heights of bars.
Each query and its corresponding relevant content are appended
to a task queue. The ground station’s inference module activates
whenever the task queue is non-empty. By separating the retrieval
process from the LVLM inference, the ground station can rapidly
respond to newly arrived priority queries during the short commu-
nication window, allowing more data to be downloaded from the
satellite before the link is lost.
The LVLM inference at the ground station mirrors the satellite’s
process. The cognitive test is omitted because the ground station
typically has more robust and comprehensive resources. Therefore,
the inference result becomes final without additional validation.
4 Implementation
In this section, we introduce the implementation of Grace, then we
give the experiment setup.
4.1 Implementing Grace
The prototype of Grace is illustrated in Figure 6. The ground station
is deployed in an H3C UniServer R5300 G3 equipped with eight
NVIDIA GeForce RTX 3090 GPUs, dual Intel Xeon Silver 4210R
processors (10 cores, 2.84 GHz each), and 8 ×32 GB DDR4 RAM,
running Ubuntu 18.04.6 LTS. The software stack includes Python
3.12.4 and PyTorch 2.5.1. The satellite is deployed in an NVIDIA
Jetson Orin NX equipped with an Arm Cortex-A78AE v8.2 CPU
(8 cores, 2 GHz each), and 16 GB LPDDR5 RAM, running Ubuntu
22.04.5 LTS. The software stack includes Python 3.10.16 and PyTorch
2.7.0. The testbed incorporates real-world orbital data (Two Line
Element) of Planet’s Dove satellite constellation to facilitate real-
time trajectory computation and network routing. The satellite-
ground network condition is emulated throughtc[ 12], thereby
establishing dynamic connectivity between ground stations and
satellites. This configuration accurately replicates authentic satellite
communication linkages.
4.2 Experimental Setup
4.2.1 Datasets.We evaluate the accuracy of the inference system
using the RSVQA dataset [ 52]. The RSVQA is a remote-sensing
visual question-answering dataset. It is divided into low-resolution
(RSVQA-LR) and high-resolution (RSVQA-HR). RSVQA-LR is one
of the earliest VQA datasets for low-spectral-resolution remote
sensing, comprising Sentinel-2 satellite images with a resolution
of 256×256 pixels. These images are sourced from OpenStreetMap.
RSVQA-HR, as a high-spectral-resolution dataset, consists of aerialRGB images captured by the USGS with a resolution of 512×512
pixels.
We further utilize three classification datasets — RESISC45 [ 16],
AID [ 72], and WHU-RS19 [ 18] — to evaluate the accuracy of the
proposed framework. RESISC45 is a large-scale remote sensing im-
age dataset comprising 45 distinct land use categories, with each
category containing 700 images of size 256 ×256 pixels, resulting
in a total of 31,500 images. The AID dataset is a large-scale aerial
image collection that covers 30 scene categories and includes 10,000
remote sensing images. WHU-RS19 is another remote sensing im-
age dataset, encompassing 19 scene categories, with approximately
50 images per category.
4.2.2 Models.We use the pre-trained Qwen2-VL 2B on the satel-
lite and the pre-trained Qwen2-VL 7B on the ground station. In
industrial practice, an LVLM of approximately 2B size is deemed
suitable for application on edge devices. For example, Google’s
Gemini Nano on Pixel 8 Pro with 1.8B and 3.25B parameters, re-
spectively [ 3]. Qwen2-VL incorporates vision-language adapters to
enhance efficiency and supports multi-image functionalities [ 10].
These multi-modal inference capabilities provide robust support for
our archive’s implementation. For the image and instruction embed-
ding models used in the dynamic archive, we employ GeoRSCLIP
ViT-B-32 [ 87] to generate embedding vectors. All models used in
experiments have not undergone any fine-tuning. Therefore, all
Qwen2-VL models are general-purpose models and have not been
specifically fine-tuned for remote sensing data.
4.2.3 Baselines.We compare Grace against several LVLM deploy-
ment baselines:
•PETALS[ 13]: Borzunov et al. proposed PETALS, a system
that joins multiple devices to collaboratively inference and
fine-tune large language models. PETALS can run LLM rea-
soning on consumer-level GPUs.
•Tabi[ 71]: Wang et al. propose a multi-layer cascaded effi-
cient model serving system Tabi, which uses methods such
as early return of simple queries, attention mechanism word
pruning, and weighted multi-level ensemble learning.
•GeoChat[ 36]: Kuckreja et al. get GeoChat by training a
LLaVA v1.5 (7B) [ 51] using a self-collected large-scale remote
sensing dataset. The satellite only performs data collection,
transmitting all data to the ground station via a wireless link.
The ground station then conducts inference for each query
using GeoChat.

Enabling Near-realtime Remote Sensing via Satellite–Ground Collaboration of Large Vision–Language Models Conference acronym ’XX, June 03–05, 2018, Woodstock, NY
20 30 40 50
Bandwidth (mbps)708090100 Accuracy (%)PETALS
Tabi
GeoChat
Grace
(a) Test accuracy.
20 30 40 50
Bandwidth (mbps)100101102103104Average Time (s) (b) Average time.
20 30 40 50
Bandwidth (mbps)100101102103104Median Time (s) (c) Median time.
20 30 40 50
Bandwidth (mbps)101102103104Max Time (s) (d) Max time.
Figure 8: Test accuracy and inference time on RSVQA HR datasets under various bandwidth settings.
4.2.4 Hyper-parameters.We set the size of the relevant content 𝐾
to 5. In the matching test, the image threshold𝑇 𝑀is set to 0.8, the
instruction threshold 𝑇𝐼to 0.94, and the minimum number of rele-
vant records after filtering 𝑇𝐾is set to 3. We establish a confidence
threshold𝑇Confof 0.75 for the cognitive test. We import all data
in the training dataset into the ground archive. We initially utilize
20 random remote sensing images from the training set and their
associated instruction-answer pairs for the satellite archive. Con-
sequently, the satellite archive contains significantly less content
than the ground archive and requires incremental updates during
operation to meet query demands. During the experiment, the up-
per limit of the number of remote sensing images in the satellite
archive is 20.
5 Performance Evaluation
In this section, we evaluate the performance of Grace from three
aspects: 1) Comparisons with four baselines to demonstrate the
superiority of Grace. 2) Investigating the impact of network-related
hyper-parameter on the performance. 3) Ablation study to show the
necessity of each component in Grace, including dynamic knowl-
edge archive, task dispatcher, and hierarchical transmission mecha-
nism.
5.1 Superiority of Grace
As illustrated in Figure 7a, we conduct a comprehensive evaluation
of our proposed Grace framework against three representative base-
lines — PETALS, Tabi, and GeoChat — on both the Low-Resolution
(LR) and High-Resolution (HR) RSVQA datasets. On the LR bench-
mark, Grace delivers an accuracy of 83.25%, markedly higher than
the sub-70% performance of PETALS (67.25%) and Tabi (67.07%), and
closely approaching the accuracy achieved by GeoChat, illustrating
Grace’s competitiveness in scenarios where rapid, onboard rea-
soning is critical. In contrast, on the more challenging HR dataset,
Grace attains an accuracy of 78.8%, surpassing GeoChat’s 72.30% by
over 6 percentage points, and substantially outdoing both PETALS
(68.05%) and Tabi (68.35%). This consistent superiority in the HR set-
ting underscores Grace’s robustness when handling higher-fidelity
imagery, a key requirement for precise remote sensing applications.
We further analyze inference latency from task acquisition to
result generation, with timing statistics summarized in Figures 7b,
7c, and 7d. Three temporal metrics are examined: mean, median, and
maximum processing times. We appropriately reduce the orbital
period of the LEO satellite as the limited number of test samplesin datasets and the long time required for the satellite to complete
one orbit.
Figure 7b reveals that Grace achieves notable efficiency improve-
ments in mean processing time compared to other baselines. Grace
exhibits a mean latency of just 6.3s on LR and 7.9s on HR — an
impressive 76–95% reduction relative to PETALS (30.3s / 30.2s) and
GeoChat (30.7s / 722.3s), and an 88–95% improvement over Tabi’s
mean of 27.2s on LR and a prohibitive 146.7s on HR. These results
demonstrate that Grace dramatically accelerates onboard inference,
reducing reliance on intermittent ground links. Examining the me-
dian latencies in Figure 7c further highlights Grace’s stable perfor-
mance: a median of 1.6s on LR and 1.6s on HR, compared to PETALS,
Tabi, and GeoChat. The low median values indicate that for the vast
majority of queries, Grace processes inputs almost instantaneously,
with only a small fraction of outliers extending beyond this central
tendency. The maximum latency analysis in Figure 7d provides
critical insights into system robustness under worst-case scenarios.
Grace exhibits constrained maximum delays of 61.1s (LR) and 64.2s
(HR), markedly lower than the 336.4s (Tabi) and 1,045.8s (GeoChat)
extremes observed in alternative approaches.
These results highlight that Grace effectively balances enhanced
accuracy with reduced inference times, making it a highly efficient
solution for onboard inference in resource-constrained LEO satel-
lite environments. By optimizing both accuracy and speed, Grace
ensures reliable performance while minimizing the dependency on
ground-based processing, thereby enhancing the overall efficacy of
the satellite-ground collaborative inference framework.
5.2 Micro-benchmarking
5.2.1 The Impact of Ground-Satellite Bandwidth.Figure 8 system-
atically demonstrates Grace’s superior robustness to bandwidth
fluctuations compared to conventional approaches. While ground-
station-dependent GeoChat exhibit severe latency degradation un-
der constrained network conditions — evidenced by mean delays
surpassing 1,635.3 seconds at 20 Mbps and only marginally im-
proving to 228.2 seconds at 50 Mbps — Grace maintains consistent
performance, maintaining mean latency within a narrow 7.6–7.9
second range across all tested bandwidth scenarios. This constitutes
a 200-fold performance improvement over previous ground-station
workflows. Notably, the system achieves a median latency of 1.6
seconds without compromising accuracy, in contrast to bandwidth-
agnostic baselines like PETALS that enforce fixed 30-second latency
thresholds at the expense of significant precision degradation.
In worst-case operational scenarios, Grace’s architectural innova-
tions manifest distinct advantages: it limits maximum latency to 65.6

Conference acronym ’XX, June 03–05, 2018, Woodstock, NY Zihan Li, Jiahao Yang, Yuxin Zhang, Zhe Chen, and Yue Gao
RESISC45 AID WHU-RS19
Dataset60708090100 Accuracy (%)GeoChat Grace
(a) Fine-tuned LVLM.
0 1 2 3 4 5
Size of Relevant Content55657585 Accuracy (%)Qwen-2.5-VL-2B
Qwen-2.5-VL-7B (b) Relevant content.
1 2 3 4 5
Size of Relevant Content5565758595 Accuracy (%)Full Search
Instruct OnlyImage Only
Total Random (c) Qwen2-VL 2B.
1 2 3 4 5
Size of Relevant Content65758595 Accuracy (%) (d) Qwen2-VL 7B.
Figure 9: The test accuracy versus the fine-tuned LVLM, the size of relevant content (𝐾), and different retrieval methods.
seconds under 20 Mbps constraints, outperforming GeoChat and
Tabi by 43 and 31 times respectively. Significantly, while competing
approaches demonstrate dramatic latency variability — Tabi’s max-
imum delays decrease by 71% and GeoChat by 85% with increased
bandwidth — Grace maintains stability within a 4% fluctuation mar-
gin, establishing unprecedented consistency for real-time remote
sensing applications. These experimental results validate Grace’s
unique capability to decouple system performance from network
constraints, representing a critical advancement for operational
systems deployed in heterogeneous connectivity environments.
5.2.2 Compare to Fine-tuned LVLMs.Current approaches to de-
veloping remote sensing-oriented LVLMs typically involve fine-
tuning foundation models on domain-specific datasets to impart
specialized geospatial knowledge. This section demonstrates that
foundation models with RAG enables more generalization capabil-
ities than pure fine-tuning approaches. We evaluate our method
against GeoChat across three benchmark remote sensing datasets:
RESISC45, AID, and WHU-RS19, with experimental results summa-
rized in Figure 9a. To ensure fair comparison, all models used in
this experiment are evaluated in their original configurations with-
out any fine-tuning. Our approach achieves superior performance
across all test scenarios, attaining 92.2% accuracy on RESISC45
(+6.1% improvement over GeoChat), 94.4% on AID (+25.5%), and
97.2% on WHU-RS19 (+13.6%). Notably, the performance gap widens
significantly on complex scene understanding tasks – the 25.5%
accuracy gain on AID highlights our method’s enhanced capability
in handling fine-grained visual-semantic relationships.
This performance advantage stems from our external archive that
dynamically integrates foundational visual understanding with on-
demand knowledge retrieval, avoiding the catastrophic forgetting
problem inherent in pure fine-tuning approaches. Unlike GeoChat’s
static parameterization of geospatial knowledge, our system adap-
tively retrieves relevant concepts from external knowledge bases
during inference, enabling better handling of long-tail scenarios
and novel objects unseen in training data.
5.3 Ablation Study
5.3.1 Impact of Size of Relevant Content.This section investigates
the effect of using archives on the inference accuracy of LVLMs.
We evaluated the inference accuracy of the LVLM without utilizing
an archive corresponding to 𝐾=0relevant images. In this scenario,
the LVLM relies solely on its internal knowledge to process queries.
Additionally, we examined the impact of varying the number of
relevant images from 1 to 5 on the inference process. Increasing 𝐾provides the LVLM with more examples, enhancing its ability to
respond accurately to the instruction 𝐼𝑄. For comparison, we tested
two models, Qwen2-VL 2B and Qwen2-VL 7B, both utilizing the
complete RSVQA-LR training dataset as the archive.
The experimental results are depicted in Figure 9b. Both the
Qwen2-VL 2B and Qwen2-VL 7B models demonstrated a grad-
ual increase in accuracy as 𝐾increased. This trend confirms that
the archive effectively enhances the inference accuracy of LVLMs.
Specifically, for the Qwen2-VL 2B model, the inference accuracy
is 55.8% without using an archive. Introducing a single relevant
relevant image elevates the accuracy to 73.1%, marking an improve-
ment of 17.2%. However, as 𝐾increased beyond one, the Qwen2-VL
2B model reached a plateau, maintaining a consistent accuracy of
79.9% for𝐾=4and𝐾=5.
The Qwen2-VL 7B starts with a higher inital accuracy of 68.05%
without the archive. Adding one relevant image boosts the accuracy
by 11.6% to 79.6%. Unlike the 2B model, the 7B model continues
to achieve performance gains even at 𝐾= 5, with a further 1.2%
increase compared to 𝐾=4. Moreover, across all values of 𝐾, the
Qwen2-VL 7B model consistently outperformed the Qwen2-VL 2B
model in inference accuracy. This indicates that the information
provided by the archive alone is insufficient to bridge the inherent
capability differences resulting from variations in model parameters.
Consequently, it is necessary to complement the onboard satellite
inference system with a ground-based inference system to achieve
optimal performance.
5.3.2 Impact of Retrieval Methods.This section explores how var-
ious retrieval strategies influence the inference accuracy of our
collaborative inference framework for LEO satellites. We test four
different retrieval methods:
•Full Search: Utilizes our proposed multimodal retrieval al-
gorithm, simultaneously querying both visual and textual
modalities.
•Instruct Only: Retrieves solely based on text instructions,
with the corresponding remote sensing images selected ran-
domly.
•Image Only: Conducts retrieval exclusively on the visual
modality, with text instructions chosen at random.
•Total Random: Selects relevant content entirely at random
from the training dataset.
Figure 9c and 9d show results. The Qwen2-VL model demonstrates
a significant performance improvement through multimodal fusion,
with the 7B variant achieving 84.4% accuracy under Full Search
using five relevant images. This scaling advantage highlights the

Enabling Near-realtime Remote Sensing via Satellite–Ground Collaboration of Large Vision–Language Models Conference acronym ’XX, June 03–05, 2018, Woodstock, NY
0.6 0.65 0.7 0.75 0.8 0.85
Fetching Threshold73757779 Accuracy (%)  0
  1  2
  3  4
  5Relevant Content
(a) Image threshold.
0.88 0.9 0.92 0.94 0.96
Fetching Threshold737475 Accuracy (%)  0
  1  2
  3  4
  5Relevant Content (b) Text threshold.
0.65 0.75 0.85 0.95
Confidence Threshold5060708090100 Accuracy (%)  0
  1  2
  3  4
  5Relevant Content (c) Confidence threshold.
Figure 10: The test accuracy versus the image (𝑇 𝑀) and the text (𝑇 𝐼) fetching threshold, and the confidence threshold (𝑇 Conf).
larger model’s enhanced capacity to integrate contextual cues. Mul-
timodal retrieval proves critical across both scales, with Full Search
outperforming Instruct Only and Image Only baselines by substan-
tial margins. For the 7B model, the multimodal approach delivers
an 11.0% accuracy boost over image-based retrieval and 5.6% over
instruction-guided search, with similar trends observed in the 2B
variant. The performance gap widens with added relevant images
— increasing from 5.3% (1 record) to 10.8% (5 records) between Full
Search and Image Only for the 7B model, confirming progressive
synergy between modalities.
These results substantiate the superiority of our multimodal re-
trieval algorithm. Notably, the image only approach does not offer
advantage over the total random method. In contrast, combining
instruction-based and image-based retrieval significantly enhances
the accuracy of the LVLM’s inference. This demonstrates that our
retrieval strategy is highly effective for the targeted inference sce-
narios, substantially improving the overall inference performance.
5.3.3 Impact of Fetching Thresholds.In the matching test, our
method filters out relevant records that show low cosine similarity
with a given query. This section investigates how varying the simi-
larity threshold affects inference accuracy. We conduct inference
with the Qwen2-VL 2B model and record the image and instruction
similarities of each relevant record for every query. We then define
a range of thresholds 𝑇𝑀(from 0.60 to 0.85 in steps of 0.05) and
retain only those relevant records whose similarity values exceed
𝑇𝑀, provided the number of remaining records surpasses a lower
bound𝑇𝐾.
Figure 10a plots the resulting inference accuracy. Every curve
with𝑇𝐾>0shows an ascending trend, indicating that higher
thresholds𝑇𝑀yield progressively better inference accuracy. The
underlying rationale is that a larger cosine similarity value signals
more substantial relevance, equipping the LVLM with more targeted
information during inference. In contrast, when 𝑇𝐾=0(i.e., the
matching test never filters out records), the accuracy curve forms a
horizontal line below the other curves. This finding underscores
the advantage of applying a matching test: it can eliminate specific
queries unlikely to be inferred correctly, thereby conserving the
satellite’s limited computing resources. Notably, a larger 𝑇𝐾typi-
cally correlates with higher accuracy. We also perform instruction
similarity tests under similar conditions. We vary the threshold
𝑇𝐼between 0.88 and 0.96. As shown in Figure 10b, inference ac-
curacy steadily increases with 𝑇𝐼, mirroring the image similarity
outcome. Thus, in both the visual and textual domains, there is apositive correlation between cosine similarity and inference accu-
racy: the closer the retrieved content is to the query, the greater
the likelihood that the LVLM produces a correct answer.
5.3.4 Impact of Confidence Threshold.Our cognitive evaluation
framework establishes confidence scores as reliable indicators of
LVLM inference validity. As shown in Figure 10c, unaided inference
(𝐾=0) exhibits unstable accuracy trends across confidence thresh-
olds, peaking at 58.6% for 𝑇Conf=0.85before declining sharply.
This contrasts with archival-supported scenarios ( 𝐾≥ 1), where
accuracy scales consistently with confidence thresholds — rising
from 73.2% to 87.7% for 𝐾=1as𝑇Confincreases from 0.65 to 0.95.
The 14.5% improvement underscores the archive’s role in aligning
high-confidence predictions with ground truth.
These findings confirm that the cognitive test based on confi-
dence scores effectively enhances inference accuracy within our
Grace inference system, which leverages an external knowledge
base. When using the archive, the positive correlation between
confidence scores and accuracy underscores the importance of inte-
grating confidence assessments with external knowledge sources to
achieve reliable and accurate inferences in LEO satellite scenarios.
5.3.5 Impact of Hierarchical Transmission.In this section, we demon-
strate the critical impact of priority queuing on optimizing tempo-
ral efficiency in both LR and HR processing pipelines. The Grace
framework with priority queuing achieves a 53% reduction in mean
processing time compared to its non-prioritized variant, decreasing
from 13.7s to 6.3s for LR data and from 16.8s to 7.9s for HR data,
as shown in Figure 11b. This acceleration stems from intelligent
task scheduling that prioritizes latency-sensitive operations while
maintaining near-optimal median processing times.
The priority-aware approach further demonstrates robustness
under peak workloads. As shown in Figure 11d, HR processing
exhibits a 67% reduction in maximum latency (from 198.1s to 64.2s)
through dynamic resource reallocation strategies that proactively
mitigate computational bottlenecks. This constrained worst-case
performance (≤64.2s) is critical for ensuring predictable responsive-
ness in time-sensitive Earth observation applications, where both
average-case efficiency and stability are essential for operational
deployment. While median performance metrics remain compa-
rable between configurations (Figure 11c), the proposed method
establishes a superior balance between throughput optimization
and system reliability, making it particularly suitable for mission-
critical remote sensing workflows.

Conference acronym ’XX, June 03–05, 2018, Woodstock, NY Zihan Li, Jiahao Yang, Yuxin Zhang, Zhe Chen, and Yue Gao
LR HR
Dataset757779818385 Accuracy (%)Grace w/o Pri. Queue
Grace
(a) Test accuracy.
LR HR
Dataset05101520 Average Time (s)
 (b) Average time.
LR HR
Dataset1.61.651.7Median Time (s) (c) Median time.
LR HR
Dataset050100150200250 Max Time (s) (d) Max time.
Figure 11: The impact of priority queue on test accuracy and latency on RSVQA LR and HR datasets. “Pri. Queue” indicates
“Priority Queue”.
6 Related Work and Discussion
6.1 LVLMs for Remote Sensing
Transformer-based multimodal large models have received wide-
spread attention in recent years for their strong generalization
capabilities and ability to perform integrated inference with mul-
timodal data [ 41]. These characteristics have motivated research
into their application in remote sensing. Hu et al. proposed RSGPT
[31], a vision-language model tailored for remote sensing. In addi-
tion, they constructed RSICap, a high-quality dataset for remote
sensing image captioning. Zhang et al. introduced EarthGPT [ 81], a
multimodal large language model that leverages a vision-enhanced
perception mechanism to integrate coarse-grained semantic and
fine-grained detail information. They also released MMRS-1M, a
large-scale dataset containing over one million image-text pairs.
Meanwhile, Kuckreja et al. developed GeoChat [ 36], a large vision-
language model designed for remote sensing images, capable of
multitask dialogue, especially region-level reasoning and visual
localization in high-resolution remote sensing imagery.
Despite these advances, current studies on LVLMs in remote
sensing primarily focus on training specialized large models us-
ing dedicated remote sensing datasets. This approach faces several
drawbacks. First, training such models demands substantial hard-
ware resources and lacks continuous update flexibility. Second, large
LVLMs require high-performance computing to support inference,
posing significant challenges for satellites operating as edge devices
with constrained computational resources. In contrast, the Grace
framework proposed in this paper features robust dynamic update
capabilities, effectively balancing inference time and accuracy and
offering a potential solution to some of these challenges.
6.2 Collaborative Inference
Collaborative inference leverages the robust computational power
of the cloud alongside the low-latency benefits of edge computing,
offering a potential solution to the challenges of poor edge inference
performance on LEO satellites and high ground inference latency
[75]. He et al. addressed issues such as data inefficiency, insufficient
latency sensitivity, and inability to adapt to task load variations
in existing cloud-edge collaborative inference systems [ 27]. They
proposed an active inference-based method to optimize task of-
floading and resource allocation for large model inference tasks in
cloud-edge systems. Wang et al. introduced Tabi [ 71], a multi-tier
cascading efficient model service system that significantly reduces
the latency and cost of cloud-edge collaborative inference withoutcompromising accuracy. This is achieved through early exit simple
queries, attention mechanism word pruning, and weighted multi-
level ensemble learning. Borzunov et al. presented PETALS [ 13], a
framework supporting split inference operations across multiple
computing nodes for large models.
Despite these advancements in collaborative inference in cloud-
edge environments, practical deployment in LEO satellite networks
remains challenging. Due to limited and unstable satellite-ground
communication bandwidth, there is often no network connectivity
between LEO satellites and ground stations for extended periods,
severely restricting data transmission rates and system stability
during the collaborative inference process. Designing a specialized
collaborative inference framework tailored to the unique network
conditions of LEO satellites is thus an urgent problem that needs
addressing. This highlights the necessity for innovative approaches
to overcome the inherent limitations of satellite-ground communi-
cations and ensure effective and reliable inference capabilities.
6.3 Discussion
The proposed system does not incorporate an inter-satellite link
(ISL) data transmission mechanism, primarily due to the absence
of ISL communication capabilities in current Earth observation
satellite constellations [ 66]. While significant technological break-
throughs have been achieved in ISL applications for communica-
tions satellites (as exemplified by SpaceX’s Starlink project [ 7],
which has demonstrated the feasibility of efficient data transmis-
sion between LEO satellites through laser-based ISLs), it is crucial to
emphasize that these technological validations predominantly focus
on communication satellite systems. The application scenarios and
communication requirements of such systems differ from those of
Earth observation satellites, which have yet to achieve large-scale
deployment of ISL technology. Based on current technology matu-
rity assessments, this study excludes ISL. Notably, Grace’s modular
framework enables future integration of mature ISL technologies
through protocol upgrades and interface modifications, thereby
potentially reducing satellite-to-ground communication latency.
7 Conclusion
In this paper, we present Grace, a collaborative inference framework
designed to address the unique challenges of remote sensing in LEO
satellite systems. To the best of our knowledge, this is the first work
to achieve near-realtime LVLM inference in LEO satellite networks.
Our approach strategically balances resource constraints against

Enabling Near-realtime Remote Sensing via Satellite–Ground Collaboration of Large Vision–Language Models Conference acronym ’XX, June 03–05, 2018, Woodstock, NY
the need for high-accuracy inference. We demonstrated that effec-
tive retrieval from a multimodal knowledge base capturing visual
and textual information significantly improves inference accuracy
on remote sensing image datasets. Matching tests filter out low-
relevance records to conserve onboard resources, while cognitive
tests leverage token-level probabilities to validate inference confi-
dence. Experimental results show that increasing the number of
relevant records and enforcing higher cosine similarity thresholds
lead to marked improvements in inference performance, underscor-
ing the critical importance of precise retrieval. Our dynamic update
mechanism also ensures the satellite archive remains aligned with
changing mission requirements, reducing unnecessary data trans-
fers and enhancing overall system efficiency. Through these design
considerations and empirical validations, our framework provides
a robust, flexible, and resource-aware solution to enable accurate,
highly efficient inference for LEO satellite applications. As a po-
tential future direction, we are looking forward to extending our
Grace to improve the performance of various applications such as
distributed learning systems [ 30,45,46,50,82], large language mod-
els [20,22,24,44,47], mobile edge computing [ 21,23,42,65,86],
etc, in LEO satellite networks.References
[1]2021.“NVIDIA A100 TENSOR CORE GPU”. Avail-
able: https://www.nvidia.com/content/dam/en-zz/Solutions/Data-
Center/a100/pdf/nvidia-a100-datasheet-us-nvidia-1758950-r4-web.pdf.
[2]2023.“Dove Satellite Constellation”. Available: https://www.planet.com/our-
constellations/.
[3]2023.“Google introduces Gemini, the most capable and flexible AI model we’ve ever
built”. Available: https://store.google.com/intl/en/ideas/articles/pixel-feature-
drop-december-2023/.
[4]2023.“NVIDIA Jetson Module Comparison”. Available:
https://connecttech.com/jetson/jetson-module-comparison/.
[5]2023.“WorldView-3”. Available: https://www.eoportal.org/satellite-
missions/worldview-3.
[6]2024.“NORAD GP Element Sets”. Available:
https://celestrak.org/NORAD/elements.
[7]2025.“Starlink satellites: Facts, tracking and impact on astronomy”. Available:
https://www.space.com/spacex-starlink-satellites.html.
[8]Joshua Adkins, Branden Ghena, Neal Jackson, Pat Pannuto, Samuel Rohrer, Brad-
ford Campbell, and Prabal Dutta. 2018. The Signpost Platform for City-Scale
Sensing. InProc. of the 17th IPSN. 188–199.
[9]Akari Asai, Zeqiu Wu, Yizhong Wang, Avirup Sil, and Hannaneh Hajishirzi. 2023.
Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection.
InProc. of the 12th ICLR.
[10] Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai
Dang, Peng Wang, Shijie Wang, Jun Tang, Humen Zhong, Yuanzhi Zhu, Mingkun
Yang, Zhaohai Li, Jianqiang Wan, Pengfei Wang, Wei Ding, Zheren Fu, Yiheng
Xu, Jiabo Ye, Xi Zhang, Tianbao Xie, Zesen Cheng, Hang Zhang, Zhibo Yang,
Haiyang Xu, and Junyang Lin. 2025. Qwen2.5-VL Technical Report.arXiv preprint
arXiv:2502.13923(2025).
[11] Panagiotis Barmpoutis, Periklis Papaioannou, Kosmas Dimitropoulos, and Nikos
Grammalidis. 2020. A Review on Early Forest Fire Detection Systems Using
Optical Remote Sensing.Sensors(2020).
[12] Joseph D Beshay, Andrea Francini, and Ravi Prakash. 2015. On the fidelity of
single-machine network emulation in linux. InProc. of the 23rd MASCOTS. 19–22.
[13] Alexander Borzunov, Dmitry Baranchuk, Tim Dettmers, Max Ryabinin, Younes
Belkada, Artem Chumachenko, Pavel Samygin, and Colin Raffel. 2022. PETALS:
Collaborative Inference and Fine-Tuning of LargeModels.arXiv preprint
arXiv:2209.01188(2022).
[14] Kaiwen Cai, Zhekai Duan, Gaowen Liu, Charles Fleming, and Chris Xiaoxuan
Lu. 2024. Self-Adapting Large Visual-Language Models to Edge Devices Across
Visual Modalities. InProc. of the 18th ECCV. 301–318.
[15] Haoyu Chen, Yuxin Zhang, Jin Zhao, Xin Wang, and Yuedong Xu. 2024. Gradient
free personalized federated learning. InProceedings of the 53rd International
Conference on Parallel Processing. 971–980.
[16] Gong Cheng, Junwei Han, and Xiaoqiang Lu. 2017. Remote Sensing Image
Scene Classification: Benchmark and State of the Art.Proc. IEEE105, 10 (2017),
1865–1883.
[17] Zhuo Cheng and Brandon Lucia. 2025. GEODUCK: Nanosatellite Constellation
Scheduling for Low Latency Event Detection. InProc. of the 23rd ACM SenSys.
547–559.
[18] Dengxin Dai and Wen Yang. 2011. Satellite Image Classification via Two-Layer
Sparse Coding With Biased Image Representation.IEEE Trans. Geosci. Remote
Sens.8, 1 (2011), 173–176.
[19] Brendan Do, Jiajun Liu, Ziwei Wang, Brano Kusy, Torsten Merz, Andy Steven,
Geoffrey Carlin, Joseph Crosswell, Yang Li, Nicholas Mortimer, et al .2023. SkySea:
Connecting Satellite, UAV and Underwater Imagery for Benthic Habitat Mapping.
InProc. of the UAVM 2023. 69–71.
[20] Tianyang Duan, Zongyuan Zhang, Songxiao Guo, Dong Huang, Yuanye Zhao,
Zheng Lin, Zihan Fang, Dianxin Luan, Heming Cui, and Yong Cui. 2025. LEED:
A Highly Efficient and Scalable LLM-Empowered Expert Demonstrations Frame-
work for Multi-Agent Reinforcement Learning.arXiv preprint arXiv:2509.14680
(2025).
[21] Tianyang Duan, Zongyuan Zhang, Songxiao Guo, Yuanye Zhao, Zheng Lin,
Zihan Fang, Yi Liu, Dianxin Luan, Dong Huang, Heming Cui, et al .2025. Sample
Efficient Experience Replay in Non-stationary Environments. InProc. ICASSP.
[22] Zihan Fang, Zheng Lin, Zhe Chen, Xianhao Chen, Yue Gao, and Yuguang Fang.
2024. Automated Federated Pipeline for Parameter-Efficient Fine-Tuning of Large
Language Models.arXiv preprint arXiv:2404.06448(2024).
[23] Zihan Fang, Zheng Lin, Senkang Hu, Hangcheng Cao, Yiqin Deng, Xianhao Chen,
and Yuguang Fang. 2024. IC3M: In-Car Multimodal Multi-Object Monitoring for
Abnormal Status of Both Driver and Passengers.arXiv preprint arXiv:2410.02592
(2024).
[24] Zihan Fang, Zheng Lin, Senkang Hu, Yihang Tao, Yiqin Deng, Xianhao Chen,
and Yuguang Fang. 2025. Dynamic uncertainty-aware multimodal fusion for
outdoor health monitoring.arXiv preprint arXiv:2508.09085(2025).
[25] Chao Feng, Xinyi Li, Yangfan Zhang, Xiaojing Wang, Liqiong Chang, Fuwei
Wang, Xinyu Zhang, and Xiaojiang Chen. 2021. RFlens: Metasurface-Enabled

Conference acronym ’XX, June 03–05, 2018, Woodstock, NY Zihan Li, Jiahao Yang, Yuxin Zhang, Zhe Chen, and Yue Gao
Beamforming for IoT Communication and Sensing. InProc. of the 27th ACM
MobiCom. 587–600.
[26] Julia Gersey, Jatin Aggarwal, Jiale Zhang, Jesse Codling, and Pei Zhang. 2025.
Sniffing Out the City-Vehicular Multimodal Sensing for Environmental and In-
frastructure Analysis. InProc. of the 23rd ACM SenSys. 632–633.
[27] Ying He, Jingcheng Fang, F Richard Yu, and Victor C Leung. [n. d.]. Large
Language Models (LLMs) Inference Offloading and Resource Allocation in Cloud-
Edge Computing: An Active Inference Approach. ([n. d.]).
[28] Aritra Hota, Soumyajit Chatterjee, and Sandip Chakraborty. 2024. Evaluating
Large Language Models as Virtual Annotators for Time-Series Physical Sensing
Data.arXiv preprint arXiv:2403.01133(2024).
[29] Jiawei Hu, Hong Jia, Mahbub Hassan, Lina Yao, Brano Kusy, and Wen Hu. 2025.
LightLLM: A Versatile Large Language Model for Predictive Light Sensing. In
Proc. of the 23rd ACM SenSys. 158–171.
[30] Mingda Hu, Jingjing Zhang, Xiong Wang, Shengyun Liu, and Zheng Lin. 2024.
Accelerating Federated Learning with Model Segmentation for Edge Networks.
IEEE Trans. Green Commun. Netw.(2024).
[31] Yuan Hu, Jianlong Yuan, Congcong Wen, Xiaonan Lu, and Xiang Li. 2023. RSGPt:
A Remote Sensing Vision Language Model and Benchmark.arXiv preprint
arXiv:2307.15266(2023).
[32] Zhizhang Hu, Yue Zhang, Ryan Rossi, Tong Yu, Sungchul Kim, and Shijia Pan.
2024. Are Large Language Models Capable of Causal Reasoning for Sensing Data
Analysis?. InProc. of the EdgeFM 2024. 24–29.
[33] Sijie Ji, Xinzhe Zheng, Jiawei Sun, Renqi Chen, Wei Gao, and Mani Srivastava.
2024. MindGuard: Towards Accessible and Sitgma-free Mental Health First Aid
via Edge LLM.arXiv preprint arXiv:2409.10064(2024).
[34] Jared Kaplan, Sam McCandlish, Tom Henighan, Tom B Brown, Benjamin Chess,
Rewon Child, Scott Gray, Alec Radford, Jeffrey Wu, and Dario Amodei. 2020.
Scaling Laws for Neural Language Models.arXiv preprint arXiv:2001.08361(2020).
[35] Teja Kattenborn, Jens Leitloff, Felix Schiefer, and Stefan Hinz. 2021. Review on
Convolutional Neural Networks (CNN) in vegetation remote sensing.ISPRS J.
Photogramm. Remote Sens.173 (2021), 24–49.
[36] Kartik Kuckreja, Muhammad Sohail Danish, Muzammal Naseer, Abhijit Das,
Salman Khan, and Fahad Shahbaz Khan. 2024. GeoChat: Grounded Large Vision-
Language Model for Remote Sensing. InProc. of the 37th IEEE/CVF CVPR. 27831–
27840.
[37] Dongbo Li, Xiangyu Liu, Zhisheng Yin, Nan Cheng, and Jie Liu. 2024. CWGAN-
Based Channel Modeling of Convolutional Autoencoder-Aided SCMA for
Satellite-Terrestrial Communication.IEEE Internet Things J.(2024).
[38] Dongbo Li, Yuchen Sun, Jielun Peng, Siyao Cheng, Zhisheng Yin, Nan Cheng, Jie
Liu, Zhijun Li, and Chenren Xu. 2024. Dual Network Computation Offloading
Based on DRL for Satellite-Terrestrial Integrated Networks.IEEE Trans. Mob.
Comput.(2024).
[39] Dongbo Li, Jiajun Zhang, Puyang Tian, Zhisheng Yin, Siyao Cheng, and Jie Liu.
2023. Boosting Bandwidth Convergence: Optimizing Resource Allocation in
Satellite-Terrestrial Integrated Networks. InProc. of the 23rd ICCT. 1141–1146.
[40] Xiang Li, Congcong Wen, Yuan Hu, Zhenghang Yuan, and Xiao Xiang Zhu. 2024.
Vision-Language Models in Remote Sensing.IEEE Geosci. Remote Sens. Mag.
(2024).
[41] Zijing Liang, Yanjie Xu, Yifan Hong, Penghui Shang, Qi Wang, Qiang Fu, and Ke
Liu. 2024. A Survey of Multimodel Large Language Models. InProc. of the 3rd
CAICE. 405–409.
[42] Zheng Lin, Zhe Chen, Xianhao Chen, Wei Ni, and Yue Gao. 2025. HASFL:
Heterogeneity-Aware Split Federated Learning over Edge Computing Systems.
arXiv preprint arXiv:2506.08426(2025).
[43] Zheng Lin, Zhe Chen, Zihan Fang, Xianhao Chen, Xiong Wang, and Yue Gao.
2024. Fedsn: A federated learning framework over heterogeneous leo satellite
networks.IEEE Transactions on Mobile Computing(2024).
[44] Zheng Lin, Xuanjie Hu, Yuxin Zhang, Zhe Chen, Zihan Fang, Xianhao Chen,
Ang Li, Praneeth Vepakomma, and Yue Gao. 2024. SplitLoRA: A Split Parameter-
Efficient Fine-Tuning Framework for Large Language Models.arXiv preprint
arXiv:2407.00952(2024).
[45] Zheng Lin, Guanqiao Qu, Wei Wei, Xianhao Chen, and Kin K Leung. 2025.
Adaptsfl: Adaptive split federated learning in resource-constrained edge net-
works.IEEE Trans. Netw.(2025).
[46] Zheng Lin, Wei Wei, Zhe Chen, Chan-Tong Lam, Xianhao Chen, Yue Gao, and
Jun Luo. 2025. Hierarchical Split Federated Learning: Convergence Analysis and
System Optimization.IEEE Trans. Mobile Comput.(2025).
[47] Zheng Lin, Yuxin Zhang, Zhe Chen, Zihan Fang, Xianhao Chen, Praneeth
Vepakomma, Wei Ni, Jun Luo, and Yue Gao. 2025. HSplitLoRA: A Heteroge-
neous Split Parameter-Efficient Fine-Tuning Framework for Large Language
Models.arXiv preprint arXiv:2505.02795(2025).
[48] Zheng Lin, Yuxin Zhang, Zhe Chen, Zihan Fang, Cong Wu, Xianhao Chen, Yue
Gao, and Jun Luo. 2025. LEO-Split: A Semi-Supervised Split Learning Framework
over LEO Satellite Networks.IEEE Trans. Mobile Comput.(2025).
[49] Zheng Lin, Yuxin Zhang, Zhe Chen, Zihan Fang, Yanni Yang, Guoming Zhang,
Huan Yang, Cong Wu, Xianhao Chen, and Yue Gao. 2025. ESL-LEO: An Efficient
Split Learning Framework over LEO Satellite Networks. InProc. Int. Conf. WirelessArtif. Intell. Comput. Syst. Appl.344–357.
[50] Zheng Lin, Guangyu Zhu, Yiqin Deng, Xianhao Chen, Yue Gao, Kaibin Huang, and
Yuguang Fang. 2024. Efficient Parallel Split Learning over Resource-Constrained
Wireless Edge Networks.IEEE Trans. Mobile Comput.23, 10 (2024), 9224–9239.
[51] Haotian Liu, Chunyuan Li, Yuheng Li, and Yong Jae Lee. 2024. Improved Baselines
with Visual Instruction Tuning. InProc. of the 37th IEEE/CVF CVPR. 26296–26306.
[52] Sylvain Lobry, Diego Marcos, Jesse Murray, and Devis Tuia. 2020. RSVQA: Visual
Question Answering for Remote Sensing Data.IEEE Trans. Geosci. Remote Sens.
58, 12 (2020), 8555–8566.
[53] Xiaomin Ouyang and Mani Srivastava. 2024. LLMSense: Harnessing LLMs for
high-level reasoning over spatiotemporal sensor traces. InProc. of the 3rd SenSys-
ML. 9–14.
[54] Hao Pan, Lili Qiu, Bei Ouyang, Shicheng Zheng, Yongzhao Zhang, Yi-Chao Chen,
and Guangtao Xue. 2023. PMSaT: Optimizing Passive Metasurface for Low Earth
Orbit Satellite Communication. InProc. of the 29th ACM MobiCom. 1–15.
[55] Planet Labs PBC. 2025.“Planet Launches High-Resolution Pelican-2 Satellite & 36
SuperDoves”. Available: https://www.planet.com/pulse/planet-launches-high-
resolution-pelican-2-satellite-36-superdoves/.
[56] Jinbo Peng, Zhe Chen, Zheng Lin, Haoxuan Yuan, Zihan Fang, Lingzhong Bao,
Zihang Song, Ying Li, Jing Ren, and Yue Gao. 2024. SUMS: Sniffing Unknown
Multiband Signals under Low Sampling Rates.IEEE Trans. Mobile Comput.(2024).
[57] Jinbo Peng, Junwen Duan, Zheng Lin, Haoxuan Yuan, Yue Gao, and Zhe Chen.
2025. SigChord: Sniffing Wide Non-Sparse Multiband Signals for Terrestrial and
Non-Terrestrial Wireless Networks.arXiv preprint arXiv:2504.06587(2025).
[58] Fatemeh Sarhaddi, Ngoc Thi Nguyen, Agustin Zuniga, Pan Hui, Sasu Tarkoma,
Huber Flores, and Petteri Nurmi. 2025. LLMs and IoT: A Comprehensive Survey
on Large Language Models and the Internet of Things.Authorea Preprints(2025).
[59] Jayanth Shenoy, Om Chabra, Tusher Chakraborty, Suraj Jog, Deepak Vasisht,
and Ranveer Chandra. 2024. CosMAC: Constellation-Aware Medium Access and
Scheduling for IoT Satellites. InProc. of the 30th ACM MobiCom. 724–739.
[60] Muskan Shergill, Zach Thompson, Guanqun Song, and Ting Zhu. 2024. Energy
Efficient LoRaWAN in LEO Satellites.arXiv preprint arXiv:2412.20660(2024).
[61] Vaibhav Singh, Tusher Chakraborty, Suraj Jog, Om Chabra, Deepak Vasisht, and
Ranveer Chandra. 2024. Exploiting Satellite Doppler for Reliable and Faster Data
Download in IoT Satellite Networks.GetMobile Mob. Comput. Commun.(2024).
[62] Vaibhav Singh, Tusher Chakraborty, Suraj Jog, Om Chabra, Deepak Vasisht, and
Ranveer Chandra. 2024. Spectrumize: Spectrum-Efficient Satellite Networks for
the Internet of Things. InProc. of the 21st NSDI. 825–840.
[63] Vaibhav Singh, Akarsh Prabhakara, Diana Zhang, Osman Yağan, and Swarun
Kumar. 2021. A Community-Driven Approach to Democratize Access to Satellite
Ground Stations. InProc. of the 27th ACM MobiCom. 1–14.
[64] Pramuka Medaranga Sooriya Patabandige, Steven Antya Orvala Waskito, Kunjun
Li, Kai Jie Leow, Shantanu Chakrabarty, and Ambuj Varshney. 2023. Rethinking
Embedded Sensor Data Processing and Analysis with Large Language Models.
InProc. of the 21st ACM MobiSys. 561–562.
[65] Yongyang Tang, Zhe Chen, Ang Li, Tianyue Zheng, Zheng Lin, Jia Xu, Pin Lv,
Zhe Sun, and Yue Gao. 2024. MERIT: Multimodal Wearable Vital Sign Waveform
Monitoring.arXiv preprint arXiv:2410.00392(2024).
[66] Bill Tao, Om Chabra, Ishani Janveja, Indranil Gupta, and Deepak Vasisht. 2024.
Known Knowns and Unknowns: Near-realtime Earth Observation Via Query
Bifurcation in Serval. InProc. of the 21st NSDI. 809–824.
[67] Girish Vaidya and Marco Zuniga. 2024. Exploiting mmWave and Deep-Learning
Models to Estimate People Count in Urban Scenarios. InProc. of the 9th IoTDI.
73–84.
[68] Deepak Vasisht, Jayanth Shenoy, and Ranveer Chandra. 2021. L2D2: Low Latency
Distributed Downlink for LEO Satellites. InProc. of the 2021 ACM SIGCOMM.
151–164.
[69] Di Wang, Qiming Zhang, Yufei Xu, Jing Zhang, Bo Du, Dacheng Tao, and Liangpei
Zhang. 2022. Advancing Plain Vision Transformer Toward Remote Sensing
Foundation Model.IEEE Trans. Geosci. Remote Sens.61 (2022), 1–15.
[70] Hengyi Wang, Haizhou Shi, Shiwei Tan, Weiyi Qin, Wenyuan Wang, Tunyu
Zhang, Akshay Nambi, Tanuja Ganu, and Hao Wang. 2024. Multimodal Needle
in a Haystack: Benchmarking Long-Context Capability of Multimodal Large
Language Models.arXiv preprint arXiv:2406.11230(2024).
[71] Yiding Wang, Kai Chen, Haisheng Tan, and Kun Guo. 2023. Tabi: An Efficient
Multi-Level Inference System for Large Language Models. InProc. of the 18th
ECCS EuroSys. 233–248.
[72] Gui-Song Xia, Jingwen Hu, Fan Hu, Baoguang Shi, Xiang Bai, Yanfei Zhong,
Liangpei Zhang, and Xiaoqiang Lu. 2017. AID: A Benchmark Data Set for Perfor-
mance Evaluation of Aerial Scene Classification.IEEE Trans. Geosci. Remote Sens.
55, 7 (2017), 3965–3981.
[73] Mengde Xu, Zheng Zhang, Fangyun Wei, Yutong Lin, Yue Cao, Han Hu, and
Xiang Bai. 2022. A Simple Baseline for Open Vocabulary Semantic Segmentation
with Pre-trained Vision-language Model. InProc. of the 16th ECCV. 736–753.
[74] Yimo Yan, Yejia Liao, Guanhao Xu, Ruili Yao, Huiying Fan, Jingran Sun, Xia Wang,
Jonathan Sprinkle, Ziyan An, Meiyi Ma, et al .2025. Large Language Models for
Traffic and Transportation Research: Methodologies, State of the Art, and Future
Opportunities.arXiv preprint arXiv:2503.21330(2025).

Enabling Near-realtime Remote Sensing via Satellite–Ground Collaboration of Large Vision–Language Models Conference acronym ’XX, June 03–05, 2018, Woodstock, NY
[75] Jiangchao Yao, Shengyu Zhang, Yang Yao, Feng Wang, Jianxin Ma, Jianwei Zhang,
Yunfei Chu, Luo Ji, Kunyang Jia, Tao Shen, et al .2022. Edge-Cloud Polarization
and Collaboration: A Comprehensive Survey for AI.IEEE Trans. Knowl. Data
Eng.35, 7 (2022), 6866–6886.
[76] Haoxuan Yuan, Zhe Chen, Zheng Lin, Jinbo Peng, Zihan Fang, Yuhang Zhong, Zi-
hang Song, and Yue Gao. 2025. SatSense: Multi-Satellite Collaborative Framework
for Spectrum Sensing.IEEE Trans. Cogn. Commun. Netw.(2025).
[77] Haoxuan Yuan, Zhe Chen, Zheng Lin, Jinbo Peng, Zihan Fang, Yuhang Zhong,
Zihang Song, Xiong Wang, and Yue Gao. 2023. Graph Learning for Multi-Satellite
Based Spectrum Sensing. InProc. IEEE Int. Conf. Commun. Technol. (ICCT). 1112–
1116.
[78] Haoxuan Yuan, Zhe Chen, Zheng Lin, Jinbo Peng, Yuhang Zhong, Xuanjie Hu,
Songyan Xue, Wei Li, and Yue Gao. 2025. Constructing 4D Radio Map in LEO
Satellite Networks with Limited Samples.IEEE INFOCOM(2025).
[79] Zhehu Yuan, Jinyang Liu, Guanqun Song, and Ting Zhu. 2024. Heat: Satellite’s
meat is GPU’s poison.arXiv preprint arXiv:2501.14757(2024).
[80] Zhiwei Zhai, Qiong Wu, Shuai Yu, Rui Li, Fei Zhang, and Xu Chen. 2023. FedLEO:
An Offloading-Assisted Decentralized Federated Learning Framework for Low
Earth Orbit Satellite Networks.IEEE Trans. Mob. Comput.23, 5 (2023), 5260–5279.
[81] Wei Zhang, Miaoxin Cai, Tong Zhang, Yin Zhuang, and Xuerui Mao. 2024. Earth-
GPT: A Universal Multimodal Large Language Model for Multisensor Image
Comprehension in Remote Sensing Domain.IEEE Trans. Geosci. Remote Sens.62
(2024), 1–20.
[82] Yuxin Zhang, Haoyu Chen, Zheng Lin, Zhe Chen, and Jin Zhao. 2025. LCFed:
An Efficient Clustered Federated Learning Framework for Heterogeneous Data.arXiv preprint arXiv:2501.01850(2025).
[83] Yuxin Zhang, Zhe Chen, Xuanjie Hu, Jin Zhao, and Yue Gao. 2025. S-leon: An
efficient split learning framework over heterogeneous leo satellite networks.
Authorea Preprints(2025).
[84] Yuxin Zhang, Zheng Lin, Zhe Chen, Zihan Fang, Wenjun Zhu, Xianhao Chen,
Jin Zhao, and Yue Gao. 2024. Satfed: A resource-efficient leo satellite-assisted
heterogeneous federated learning framework.arXiv preprint arXiv:2409.13503
(2024).
[85] Yuxin Zhang, Jiahao Yang, Zhe Chen, Wenjun Zhu, Jin Zhao, and Yue Gao.
2025. A Satellite-Ground Synergistic Large Vision-Language Model System for
Earth Observation. InProceedings of the 33rd ACM International Conference on
Multimedia. 11825–11833.
[86] Zongyuan Zhang, Tianyang Duan, Zheng Lin, Dong Huang, Zihan Fang, Zekai
Sun, Ling Xiong, Hongbin Liang, Heming Cui, Yong Cui, et al .2025. Robust
deep reinforcement learning in robotics via adaptive gradient-masked adversarial
attacks. InProc. IROS.
[87] Zilun Zhang, Tiancheng Zhao, Yulong Guo, and Jianwei Yin. 2024. RS5M
and GeoRSCLIP: A Large Scale Vision-Language Dataset and A Large Vision-
Language Model for Remote Sensing.IEEE Trans. Geosci. Remote Sens.62 (2024),
1–23.
[88] Zhiyuan Zhao, Zhe Chen, Zheng Lin, Wenjun Zhu, Kun Qiu, Chaoqun You, and
Yue Gao. 2024. LEO Satellite Networks Assisted Geo-Distributed Data Processing.
IEEE Wireless Commun. Lett.(2024).