# Comprehending Spatio-temporal Data via Cinematic Storytelling using Large Language Models

**Authors**: Panos Kalnis. Shuo Shang, Christian S. Jensen

**Published**: 2025-10-20 08:44:25

**PDF URL**: [http://arxiv.org/pdf/2510.17301v1](http://arxiv.org/pdf/2510.17301v1)

## Abstract
Spatio-temporal data captures complex dynamics across both space and time,
yet traditional visualizations are complex, require domain expertise and often
fail to resonate with broader audiences. Here, we propose MapMuse, a
storytelling-based framework for interpreting spatio-temporal datasets,
transforming them into compelling, narrative-driven experiences. We utilize
large language models and employ retrieval augmented generation (RAG) and
agent-based techniques to generate comprehensive stories. Drawing on principles
common in cinematic storytelling, we emphasize clarity, emotional connection,
and audience-centric design. As a case study, we analyze a dataset of taxi
trajectories. Two perspectives are presented: a captivating story based on a
heat map that visualizes millions of taxi trip endpoints to uncover urban
mobility patterns; and a detailed narrative following a single long taxi
journey, enriched with city landmarks and temporal shifts. By portraying
locations as characters and movement as plot, we argue that data storytelling
drives insight, engagement, and action from spatio-temporal information. The
case study illustrates how MapMuse can bridge the gap between data complexity
and human understanding. The aim of this short paper is to provide a glimpse to
the potential of the cinematic storytelling technique as an effective
communication tool for spatio-temporal data, as well as to describe open
problems and opportunities for future research.

## Full Text


<!-- PDF content starts -->

Comprehending Spatio-temporal Data via Cinematic Storytelling
using Large Language Models
Panos Kalnis
panos.kalnis@kaust.edu.sa
King Abdullah University of Science
and Technology (KAUST)
Saudi ArabiaShuo Shang
jedi.shang@gmail.com
University of Electronic Science and
Technology of China (UESTC)
ChinaChristian S. Jensen
csj@cs.aau.dk
Aalborg University
Denmark
Abstract
Spatio-temporal data captures complex dynamics across both space
and time, yet traditional visualizations are complex, require do-
main expertise and often fail to resonate with broader audiences.
Here, we proposeMapMuse, a storytelling-based framework for
interpreting spatio-temporal datasets, transforming them into com-
pelling, narrative-driven experiences. We utilize large language
models and employ retrieval augmented generation (RAG) and
agent-based techniques to generate comprehensive stories. Draw-
ing on principles common in cinematic storytelling, we emphasize
clarity, emotional connection, and audience-centric design. As a
case study, we analyze a dataset of taxi trajectories. Two perspec-
tives are presented: a captivating story based on a heat map that
visualizes millions of taxi trip endpoints to uncover urban mobil-
ity patterns; and a detailed narrative following a single long taxi
journey, enriched with city landmarks and temporal shifts. By por-
traying locations as characters and movement as plot, we argue
that data storytelling drives insight, engagement, and action from
spatio-temporal information. The case study illustrates howMap-
Musecan bridge the gap between data complexity and human
understanding. The aim of this short paper is to provide a glimpse
to the potential of the cinematic storytelling technique as an ef-
fective communication tool for spatio-temporal data, as well as to
describe open problems and opportunities for future research.
Keywords
LLM, RAG, Agents, Spatial, Temporal, Trajectories, MapMuse
ACM Reference Format:
Panos Kalnis, Shuo Shang, and Christian S. Jensen. 2025. Comprehending
Spatio-temporal Data via Cinematic Storytelling using Large Language
Models. InProceedings of 19th International Symposium on Spatial and Tem-
poral Data (SSTD ’25).ACM, New York, NY, USA, 5 pages. https://doi.org/
XXXXXXX.XXXXXXX
1 Introduction
The phrase“a picture is worth a thousand words”holds true only
when the viewer possesses the necessary context to decode the
Permission to make digital or hard copies of all or part of this work for personal or
classroom use is granted without fee provided that copies are not made or distributed
for profit or commercial advantage and that copies bear this notice and the full citation
on the first page. Copyrights for components of this work owned by others than the
author(s) must be honored. Abstracting with credit is permitted. To copy otherwise, or
republish, to post on servers or to redistribute to lists, requires prior specific permission
and/or a fee. Request permissions from permissions@acm.org.
SSTD ’25, Osaka, Japan
©2025 Copyright held by the owner/author(s). Publication rights licensed to ACM.
ACM ISBN 978-1-4503-XXXX-X/2025/06
https://doi.org/XXXXXXX.XXXXXXX
Figure 1: Heat-map of taxi trip destinations in Porto.
image’s meaning. Without the lens of experience, education, or
cultural awareness, even the most vivid image can be stripped of
its story. Meaning, therefore, is not inherent in the image itself but
co-created by the viewer’s ability to interpret it. This is also true for
data visualizations. Without the appropriate analytical background,
the underlying story of a visual representation —such as a heat
map, scatterplot, or network graph— may remain inaccessible.
Consider Figure 1, which depicts the heat-map of the destina-
tions of taxi trips in Porto1, Portugal. For an expert, with a good
understanding of the layout of the specific city, a heat map of ur-
ban taxi activity may reveal socio-economic divides, nightlife hubs,
or transit inefficiencies. However, a non-expert can only extract
superficial facts; for instance, the destinations of many taxi trips
are clustered around the city center, but this is expected in any city
and does not reveal any useful information.
While Tufle [ 24] advocates that clarity, minimalism, and thought-
ful layout allow knowledge to emerge naturally from data, Kazakoff
[13], in contrast, emphasizes thatstorytellingis essential for trans-
forming raw data into persuasive, decision-driving narratives. Kaza-
koff argues that data alone does not convince; story structure gives
it meaning. By framing insights with a clear beginning (context),
middle (evidence), and end (recommendation), data communicators
can guide audiences through complex information with clarity and
purpose. Lo Duca [ 18] supports this argument and reinforces it
with techniques fromcinematic storytelling, such as introducing
1https://www.kaggle.com/c/pkdd-15-predict-taxi-service-trajectory-iarXiv:2510.17301v1  [cs.DB]  20 Oct 2025

SSTD ’25, August 25–27, 2025, Osaka, Japan P. Kalnis, S. Shang, C.S. Jensen
characters, a plot, and structured acts, to transform raw data into
compelling narratives that resonate with audiences.
Based on these principles, consider the following narrative:
Story 1:The heat map of taxi endpoints in Porto, Portugal,
outlines a dynamic urban mobility pattern centered around key
infrastructural and cultural nodes. Dense clusters appear in the his-
toric core, particularly near Avenida dos Aliados, São Bento Station,
and Rua de Santa Catarina, reflecting high pedestrian volume and
multimodal integration. Taxis converge near iconic landmarks such
as Porto Cathedral, Ribeira district, Dom Luís I Bridge, Clérigos
Tower, and Livraria Lello. Commercial and logistical hubs —Bolhão
Market, Aliados Avenue, and Rotunda da Boavista— also register
substantial activity. Peripheral density zones reach Casa da Música,
Hospital de Santo António, Campanhã Station, Estádio do Dragão,
Palácio de Cristal Gardens, Matosinhos Beach, and Foz do Douro.
These spatial patterns narrate structured urban flows shaped by
tourism, commuting, and centralized service access. The endpoint
distribution reveals a polycentric cityscape with functionally differ-
entiated hotspots.
Story 1 provides the appropriate context that allows a non-expert
to comprehend the raw data of Figure 1. While the result is hardly
surprising, the novelty lies in the fact that the narrative is generated
by a large language model (LLM), instead of a human curator:
Prompt 1:I will provide a png image that represents a heat map
of taxi endpoints in the city of Porto, Portugal. Describe the data in
a neutral professional tone. Use technical terms. Use storytelling
techniques. Include at least 15 POIs. Highlight the POIs. Use at most
150 words.
All examples in this paper were generated2by ChatGPT-4o, but
other LLMs can also be used. Interestingly, we can request the LLM3
to generate a map highlighting the POIs that appear in the story.
The resulting map is shown in Figure 2 and clearly enhances the
comprehension of the story.
Motivated by the previous example, we proposeMapMuse, a
system that follows cinematic storytelling principles to automat-
ically generate engaging stories about spatio-temporal data. The
stories are tailored to diverse audiences in order to improve their
comprehension of the data and enhance their practical usefulness.
MapMuseuses techniques from retrieval augmented generation
[6], as well as agentic LLM workflows [ 21] to contact external data
sources, apply data transformations and perform validation in order
to minimize hallucinations.MapMusefocuses on storytelling and
noton data analytics, although it utilizes expert spatial data ana-
lytics agents to extract aggregated information. To the best of our
knowledge, it is the first system that emphasizes the comprehension
of spatio-temporal data through cinematic storytelling.
MapMuseis work in progress. This short paper aims to illustrate
the potential of cinematic storytelling as a compelling communi-
cation framework for spatio-temporal data, while outlining open
problems and promising directions for future research.
2The prompts in this paper may not generate the exact same response, because the
models are constantly being updated; the response is stochastic (a.k.a. “temperature”);
and response depends on the history of the previous conversation.
3Refer to Section 3 for the prompt.2 Related Work
Spatio-temporal data visualization.The visualization of spatio-
temporal data has been extensively studied across visual comput-
ing, GIS, and data analytics, leading to a rich body of work that
addresses challenges in representing, analyzing, and interpreting
data evolving over both space and time [ 2]. Several libraries, such as
PyNKDV [ 4], have been developed to enable scalable visualization
and efficient spatial analytics pipelines [ 3]. There are numerous
applications of spatio-temporal data visualization. For example,
DeepTrack [ 19] transforms raw spatio-temporal data about COVID-
19 into an interactive narrative that reveals how the pandemic
evolved over time and space.
Storytelling for data visualization.The principles of story-
telling have been applied in data visualization to enhance data
comprehension through narrative techniques. Segel et al. [ 23], intro-
duce a taxonomy of narrative visualization genres, analyze design
strategies for storytelling with data, and highlight challenges in
balancing narrative and interactivity. Hullman et al. [ 12] examine
how design choices influence interpretation and analyze rhetorical
techniques in visual narratives, whereas Kosara et al. [ 14] advocate
for storytelling as a fundamental component of visualization and
explore the role of author-driven narratives. Rodrigues et al. [ 22]
propose design guidelines for narrative visualizations, emphasizing
the integration of time and space in storytelling. Closer to our work
is the framework by Chen et al. [ 5], which synthesizes data sto-
ries from visual analytics, emphasizes narrative coherence across
spatial-temporal dimensions, and supports integration of cinematic
sequencing. None of the aforementioned works utilizes LLMs.
LLMs for spatio-temporal data analytics.Recent advance-
ments in large language models intersect with spatio-temporal
data analytics. Several works explore how LLMs can assist in rea-
soning, representation learning, and forecasting tasks involving
spatio-temporal data. STG-LLM [ 17] introduces a tokenizer-adapter
system that enables LLMs to process spatio-temporal graph data
for forecasting. Similarly, Huang et al. [ 11] propose a framework
that uses spatio-temporal tokenizers and virtual nodes to capture
complex correlations. UrbanGPT [ 16] trains an LLM specifically on
urban spatio-temporal data to support generalization across tasks
like prediction, classification, and summarization. LLMGeovec [8]
combines auxiliary map data and language modeling to improve
geographic and temporal representation in LLMs, whereas STBench
[15] provides a benchmark to evaluate LLM performance across
spatio-temporal reasoning, computation, and application tasks.
While there are numerous works that take as input a story-line
and generate a sequence of images [ 25–27], we are not aware of any
existing approach that directly unifies LLMs, cinematic storytelling,
and spatio-temporal data visualization.
3 Cinematic Story Generation
Lo Duca [ 18] outlines a cinematic storytelling framework for trans-
forming raw data into compelling narratives that resonate with
audiences. The main principles are:
(1)Extract characters from data.“Characters” may be people,
places, or variables. (i) theHerois the central focus of the data
story; (ii)Sidekicksprovide context, or additional insights; and
(iii)Antagonistsare factors that challenge the hero’s objectives.

Storytelling for Spatio-temporal Data SSTD ’25, August 25–27, 2025, Osaka, Japan
Figure 2: Points of interest (POIs) in Porto, mentioned in the storytelling narratives for the taxi endpoints heat-map.
(2)Extract plot from data and utilize the three-act structure.
Data stories should follow the classic narrative structure:Act
I: Introduce the hero and their goal;Act II: Present the chal-
lenges;Act III: Resolve the narrative, highlighting insights and
implications.
(3)Tailor the Story to the Audience.The data story’s content,
structure, and delivery method must be adapted to ensure rele-
vance and engagement.
(4)Follow a Structured Storytelling Process:(i)Analyze: Gather
and interpret data to uncover insights; (ii)Create: Develop a nar-
rative that highlights key findings and their implications; and
(iii)Deliver: Present the narrative using visuals and storytelling
techniques to engage the audience.
3.1 Case Study - Heatmap of taxi destinations
Whereas Story 1 targeted generic, professional audience, next we
provide a second example where the target audience is a first time
visitor of Porto. The POIs remain the same. Observe that, by using
the principles of cinematic storytelling, Story 2 provides useful
information to the visitor, while maintaining their interest with
fluid spatial movement and vivid prose.
Prompt 2:I will provide a png image that presents a heat map of
taxi endpoints in the city of Porto, Portugal. Write a story about the
data using cinematic storytelling techniques. Your target audience
is a first time visitor of the city. Include at least 15 POIs. Highlight
the POIs. Include some useful information about the POIs. Use at
most 250 words.
Story 2:Your first taxi in Porto doesn’t just move—it reveals.
You begin at São Bento Station, where 20,000 azulejos tell tales
of conquest and culture. From there, the map glows. Avenida dos
Aliados, framed by beaux-arts buildings and crowned with City
Hall, pulses with civic pride. Along Rua de Santa Catarina, cafés andboutiques beckon beneath historic arcades. Your driver points out
Porto Cathedral, a fortress-like Romanesque monument, then coasts
downhill to Ribeira, the riverfront maze where pastel façades re-
flect in the Douro. The car glides under Dom Luís I Bridge, Gustave
Eiffel’s iron sibling. You pass Clérigos Tower, 76 meters of baroque
elegance, then Livraria Lello, one of the world’s most beautiful
bookstores—allegedly an inspiration for Hogwarts. Crowds swirl
around Bolhão Market, a hub of fishmongers and flower sellers. Ali-
ados Avenue stretches again before giving way to the roundabout at
Rotunda da Boavista, near the dramatic concert hall Casa da Música.
Nearby, Hospital de Santo António stands as an 18th-century neo-
classical institution still in operation. Eastward, Campanhã Station
serves national rail lines, while Estádio do Dragão roars on match
days with FC Porto pride. The air softens at Palácio de Cristal
Gardens, offering panoramic views. Finally, the ocean calls—Foz
do Douro and Matosinhos Beach, where the city exhales into sea
mist and golden sand. Each glowing endpoint on the heat map? A
memory waiting to happen. And this—your first ride—is just the
beginning.
The narrative exemplifies Lo Duca’s framework as follows:Char-
actersare the POIs.Plot and structure:Act I:The journey begins
at São Bento Station, establishing the tone and historical context;
Act II:The narrative unfolds through diverse urban experiences:
marketplaces, monuments, and landmarks;Act III:The story cul-
minates in the oceanfront, a metaphorical and literal resolution.
Audience:The story is crafted for first-time visitors, using cul-
tural references, while avoiding technical jargon.Storytelling:
The prose uses vivid description, spatial continuity, and cultural
cues to emotionally engage and inform the reader.

SSTD ’25, August 25–27, 2025, Osaka, Japan P. Kalnis, S. Shang, C.S. Jensen
(a) Original trajectory (red)
 (b) Storyline POIs
Figure 3: Case study: single trajectory with hallucinations.
The map of the POIs is the same as Figure 2. To draw the map, we
use the following prompt that generates the corresponding Python
code4; the code required minor manual tweaking.
Prompt 2.code:Create python code to: Plot all POI locations.
Use OpenStreetMap tiles (streets, rivers, landmarks visible). Show
only a number at each POI, and generate a legend with all POI
labels. If two POIs are close to each other on the map, show them
together.
3.2 Case study - Single trajectory
Prompt 3:I will provide a cvs file that contains a list of points
(longitude, latitude), that describe a trajectory of a taxi trip. Using
cinematic storytelling write a story about the trajectory. Mention
explicitly major road names, intersections, neighborhoods, local
POIs (e.g., restaurant name). Use 150 words maximum.
Prompt 3 receives a single trajectory and generates a story about
it. The story together with the code to extract the trajectory, can be
found in our github repository4. Figure 3a shows (in red) the original
trajectory, while Figure 3b is the map of the POIs appearing in the
story. The POIs are real and the storyline is smooth and plausible.
Nevertheless, the LLM is hallucinating: refer again to Figure 3a
where the blue stars correspond to the POIs. The generated POIs,
although real, are mostly irrelevant to the trajectory. Eliminating
hallucinations is a topic of our ongoing work.
4MapMuseAgentic LLM Workflow
Figure 4 presents the architecture ofMapMuse. The user submits
a request involving a spatio-temporal dataset to thecontrol agent.
4The code is available at https://github.com/pkalnis/sstd2025
RAW 
Spatio- 
temporal 
Data
Web 
search
Discovery 
Agent
Control 
Agent
Query 
Agent
Analytics 
Agent
Validation 
Agent
Story 
generation 
LLM
Power 
userFigure 4: Overview of theMapMuseagentic LLM workflow.
This is an LLM that can orchestrate the execution of expert agents.
Examples includequery agentsthat can manipulate the raw data,
ordiscovery agentsthat can access general web pages or specialized
APIs, like OpenStreetMap. The control agent generates a plan of
agents to call. The expert agents then query, transform and summa-
rize the data, as requested. The control agent appends the retrieved
information to the prompt of thestory generation LLM, which has
been finetuned to follow the cinematic storytelling principles. The
generated story is passed to thevalidation agent, whose job is to
flag hallucinations. If the story passes the filter, it is returned to the
user through the control agent.
5 Conclusion: Limitations and Open Problems
In this paper we demonstrate that cinematic storytelling has the
potential to enhance the comprehension of spatio-temporal data for
diverse audiences. Nonetheless, the current version ofMapMuse
has several limitations: (i) the context provided to the LLM consists
of spatio-temporal data that are pre-processed and aggregated by
human experts; (ii) the prompts are fine-tuned manually; (iii) the
generated stories are validated by humans; and (iv) the system relies
to ChatGPT-4o to access web sources. Moreover, the presented case
studies utilize the popular Porto taxis dataset, which is available
on Kaggle and has probably been seen by the LLM during training.
MapMusebeing work in progress, a goal of this short paper is to
identify open problems and propose directions for future research.
Our team is currently working on:
•Developing embedding techniques to summarize large spatial
and temporal data that are not known to the LLM and provide
them as context, within the token limit of the LLM.
•Employing RAG techniques [ 6,7,9] to couple efficiently and
effectively the data sources with the LLMs.
•Extending the context to relevant non-spatio-temporal data, such
as socio-economics, or climate data.
•Fine-tuning smaller open-source LLMs on spatio-temporal data,
to avoid sending the data to commercial LLMs for lower cost and
data privacy reasons.

Storytelling for Spatio-temporal Data SSTD ’25, August 25–27, 2025, Osaka, Japan
•Creating benchmarks and appropriate metrics to assess the qual-
ity of the generated stories. Ensure the included POIs are helpful
to convey the intended message.
•Tuning the storytelling LLM to generate captivating stories tai-
lored to diverse audiences (e.g. tourists versus city planners)
through supervised fine tuning, or reinforcement learning. Pro-
vide templates for common use cases.
•Improving story validation by question answering techniques
[20] against trusted data sources, to avoid hallucinations. Employ
chain-of-though techniques to validate intermediate results and
avoid contradictions.
•Enhancing the agent orchestration [ 1,10,21] capabilities of the
control agent.
Acknowledgments
AI-generated content.The paper is dealing with the automatic
generation of stories that explain spatio-temporal data. All stories
appearing in the paper were generated by AI. Figures 2 and 3 were
partially generated by AI. In all cases, we used ChatGPT-4o.
Computational resources.For computer time, this research used
IBEX and Shaheen III, managed by the Supercomputing Core Lab-
oratory at King Abdullah University of Science and Technology
(KAUST), Saudi Arabia.
References
[1]Ibrahim Abdelaziz, Kinjal Basu, Mayank Agarwal, and et.al. 2024. Granite-
Function Calling Model: Introducing Function Calling Abilities via Multi-task
Learning of Granular Tasks. InEMNLP: Industry Track. doi:10.18653/v1/2024.
emnlp-industry.85
[2]Benjamin Bach, Pierre Dragicevic, Daniel Archambault, Christophe Hurter, and
Sheelagh Carpendale. 2015. A review of temporal data visualizations based
on space-time cube operations.Computer Graphics Forum34, 6 (2015), 36–61.
EuroVis 2015.
[3]Tsz Nam Chan, Pak Lon Ip, Kaiyan Zhao, Leong Hou U, Byron Choi, and Jianliang
Xu. 2022. LIBKDV: A Versatile Kernel Density Visualization Library for Geospatial
Analytics.Proceedings of the VLDB Endowment15, 12 (2022), 3606–3609. doi:10.
14778/3554821.3554855
[4]Tsz Nam Chan, Rui Zang, Pak Lon Ip, Leong Hou U, and Jianliang Xu. 2023.
PyNKDV: An Efficient Network Kernel Density Visualization Library for Geospa-
tial Analytic Systems. InCompanion of the 2023 International Conference on
Management of Data (SIGMOD-Companion ’23). Association for Computing Ma-
chinery, New York, NY, USA, 99–102. doi:10.1145/3555041.3589711
[5]Siming Chen, Jie Li, Gennady Andrienko, Natalia Andrienko, Yun Wang, Phong H.
Nguyen, and Cagatay Turkay. 2020. Supporting Story Synthesis: Bridging the Gap
between Visual Analytics and Storytelling.IEEE Transactions on Visualization
and Computer Graphics26, 7 (2020), 2499–2516. doi:10.1109/TVCG.2018.2889054
[6]Wenqi Fan, Yujuan Ding, Liangbo Ning, Shijie Wang, Hengyun Li, Dawei Yin,
Tat-Seng Chua, and Qing Li. 2024. A Survey on RAG Meeting LLMs: Towards
Retrieval-Augmented Large Language Models(KDD ’24). Association for Com-
puting Machinery, New York, NY, USA, 6491–6501. doi:10.1145/3637528.3671470
[7]Saeed Fathollahzadeh, Essam Mansour, and Matthias Boehm. 2025. CatDB: Data-
catalog-guided, LLM-based Generation of Data-centric ML Pipelines.PVDLB (to
appear)(2025).[8]Junlin He, Tong Nie, and Wei Ma. 2025. Geolocation Representation from Large
Language Models Are Generic Enhancers for Spatio-Temporal Learning.Proceed-
ings of the AAAI Conference on Artificial Intelligence39, 16 (Apr. 2025), 17094–
17104. doi:10.1609/aaai.v39i16.33879
[9]Mossad Helali, Niki Monjazeb, Shubham Vashisth, Philippe Carrier, Ahmed Helal,
Antonio Cavalcante, Khaled Ammar, Katja Hose, and Essam Mansour. 2024.
KGLiDS: A Platform for Semantic Abstraction, Linking, and Automation of Data
Science. InICDE. IEEE, New York, NY, USA, 179–192. doi:10.1109/ICDE60146.
2024.00021
[10] Tenghao Huang, Kinjal Basu, Ibrahim Abdelaziz, Pavan Kapanipathi, Jonathan
May, and Muhao Chen. 2025. R2D2: Remembering, Reflecting and Dynamic
Decision Making for Web Agents. arXiv:2501.12485 [cs.AI]
[11] YiHeng Huang, Xiaowei Mao, Shengnan Guo, Yubin Chen, Junfeng Shen,
Tiankuo Li, Youfang Lin, and Huaiyu Wan. 2024. STD-PLM: Understanding
Both Spatial and Temporal Properties of Spatial-Temporal Data with PLM.
arXiv:2407.09096 [cs.LG] https://arxiv.org/abs/2407.09096
[12] Jessica Hullman and Nick Diakopoulos. 2011. Visualization Rhetoric: Framing Ef-
fects in Narrative Visualization.IEEE Transactions on Visualization and Computer
Graphics17, 12 (2011), 2231–2240. doi:10.1109/TVCG.2011.255
[13] Miro Kazakoff. 2022.Persuading with Data: A Guide to Designing, Delivering, and
Defending Your Data. MIT Press, Cambridge, MA.
[14] Robert Kosara and Jock Mackinlay. 2013. Storytelling: The Next Step for Visual-
ization.Computer46, 5 (2013), 44–50. doi:10.1109/MC.2013.36
[15] Wenbin Li, Di Yao, Ruibo Zhao, Wenjie Chen, Zijie Xu, Chengxue Luo, Chang
Gong, Quanliang Jing, Haining Tan, and Jingping Bi. 2024. STBench: As-
sessing the Ability of Large Language Models in Spatio-Temporal Analysis.
arXiv:2406.19065 [cs.CL] https://arxiv.org/abs/2406.19065
[16] Zhonghang Li, Lianghao Xia, Jiabin Tang, Yong Xu, Lei Shi, Long Xia, Dawei Yin,
and Chao Huang. 2024. UrbanGPT: Spatio-Temporal Large Language Models.
arXiv:2403.00813 [cs.CL] https://arxiv.org/abs/2403.00813
[17] Lei Liu, Shuo Yu, Runze Wang, Zhenxun Ma, and Yanming Shen. 2024.
How Can Large Language Models Understand Spatial-Temporal Data?
arXiv:2401.14192 [cs.LG] https://arxiv.org/abs/2401.14192
[18] Angelica Lo Duca. 2025.Become a great data storyteller: learn how you can drive
change with data. Wiley, Hoboken, NJ.
[19] Yuyu Luo, Wenbo Li, Tianyu Zhao, Xiang Yu, Lixi Zhang, Guoliang Li, and Nan
Tang. 2020. DeepTrack: Monitoring and Exploring Spatio-Temporal Data – A
Case of Tracking COVID-19.Proceedings of the VLDB Endowment13, 12 (2020),
2841–2844. doi:10.14778/3415478.3415489
[20] Reham Omar, Ishika Dhall, Panos Kalnis, and Essam Mansour. 2023. A Universal
Question-Answering Platform for Knowledge Graphs.ACM-SIGMOD1, 1, Article
57 (May 2023), 25 pages. doi:10.1145/3588911
[21] Aske Plaat, Max van Duijn, Niki van Stein, Mike Preuss, Peter van der Putten,
and Kees Joost Batenburg. 2025. Agentic Large Language Models, a survey.
arXiv:2503.23037 [cs.AI] https://arxiv.org/abs/2503.23037
[22] Sara Rodrigues, Ana Figueiras, and Ilo Alexandre. 2019. Once Upon a Time in a
Land Far Away: Guidelines for Spatio-Temporal Narrative Visualization. In2019
23rd International Conference Information Visualisation (IV). IEEE, New York, NY,
USA, 44–49. doi:10.1109/IV.2019.00017
[23] Edward Segel and Jeffrey Heer. 2010. Narrative Visualization: Telling Stories
with Data.IEEE Transactions on Visualization and Computer Graphics16, 6 (2010),
1139–1148. doi:10.1109/TVCG.2010.179
[24] Edward R. Tufte. 2020.Seeing with Fresh Eyes: Meaning, Space, Data, Truth.
Graphics Press, Cheshire, CT.
[25] Jiankang Wang, Zhihan Zhang, Zhihang Liu, Yang Li, Jiannan Ge, Hong-
tao Xie, and Yongdong Zhang. 2025. SpaceVLLM: Endowing Multimodal
Large Language Model with Spatio-Temporal Video Grounding Capability.
arXiv:2503.13983 [cs.CV] https://arxiv.org/abs/2503.13983
[26] Mingyuan Zhang, Huirong Li, Zhongang Cai, Jiawei Ren, Lei Yang, and Ziwei
Liu. 2023. FineMoGen: Fine-Grained Spatio-Temporal Motion Generation and
Editing. arXiv:2312.15004 [cs.CV] https://arxiv.org/abs/2312.15004
[27] Sixiao Zheng and Yanwei Fu. 2025. ContextualStory: Consistent Visual Story-
telling with Spatially-Enhanced and Storyline Context. arXiv:2407.09774 [cs.CV]
https://arxiv.org/abs/2407.09774