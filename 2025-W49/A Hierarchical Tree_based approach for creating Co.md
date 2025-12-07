# A Hierarchical Tree-based approach for creating Configurable and Static Deep Research Agent (Static-DRA)

**Authors**: Saurav Prateek

**Published**: 2025-12-03 15:37:13

**PDF URL**: [https://arxiv.org/pdf/2512.03887v2](https://arxiv.org/pdf/2512.03887v2)

## Abstract
The advancement in Large Language Models has driven the creation of complex agentic systems, such as Deep Research Agents (DRAs), to overcome the limitations of static Retrieval Augmented Generation (RAG) pipelines in handling complex, multi-turn research tasks. This paper introduces the Static Deep Research Agent (Static-DRA), a novel solution built upon a configurable and hierarchical Tree-based static workflow.
  The core contribution is the integration of two user-tunable parameters, Depth and Breadth, which provide granular control over the research intensity. This design allows end-users to consciously balance the desired quality and comprehensiveness of the research report against the associated computational cost of Large Language Model (LLM) interactions. The agent's architecture, comprising Supervisor, Independent, and Worker agents, facilitates effective multi-hop information retrieval and parallel sub-topic investigation.
  We evaluate the Static-DRA against the established DeepResearch Bench using the RACE (Reference-based Adaptive Criteria-driven Evaluation) framework. Configured with a depth of 2 and a breadth of 5, and powered by the gemini-2.5-pro model, the agent achieved an overall score of 34.72. Our experiments validate that increasing the configured Depth and Breadth parameters results in a more in-depth research process and a correspondingly higher evaluation score. The Static-DRA offers a pragmatic and resource-aware solution, empowering users with transparent control over the deep research process. The entire source code, outputs and benchmark results are open-sourced at https://github.com/SauravP97/Static-Deep-Research/

## Full Text


<!-- PDF content starts -->

A Hierarchical Tree-based approach for creating Configurable and
Static Deep Research Agent (Static-DRA)
Saurav Prateek
December 2025
Abstract
The advancement in Large Language Models has driven the creation of complex agentic systems, such as
Deep Research Agents (DRAs), to overcome the limitations of static Retrieval Augmented Generation (RAG)
pipelines in handling complex, multi-turn research tasks. This paper introduces the Static Deep Research
Agent (Static-DRA), a novel solution built upon a configurable and hierarchical Tree-based static workflow.
The core contribution is the integration of two user-tunable parameters, Depth and Breadth, which
provide granular control over the research intensity. This design allows end-users to consciously balance the
desired quality and comprehensiveness of the research report against the associated computational cost of
Large Language Model (LLM) interactions. The agent’s architecture, comprising Supervisor, Independent,
and Worker agents, facilitates effective multi-hop information retrieval and parallel sub-topic investigation.
We evaluate the Static-DRA against the established DeepResearch Bench using the RACE (Reference-based
Adaptive Criteria-driven Evaluation) framework. Configured with adepthof2and abreadthof5, and
powered by thegemini-2.5-promodel, the agent achieved an overall score of34.72. Our experiments validate
that increasing the configured Depth and Breadth parameters results in a more in-depth research process and
a correspondingly higher evaluation score. The Static-DRA offers a pragmatic and resource-aware solution,
empowering users with transparent control over the deep research process. The entire source code, outputs
and benchmark results are open-sourced athttps://github.com/SauravP97/Static-Deep-Research/
1 Introduction
Advancement in the Large Language Models has led to the development of multiple complex agentic systems
that can research on complex topics and perform reasoning tasks. The conceptual lineage of Deep Research
Agent begins with the limitations of early RAG (Retrieval Augmented Generation) [ 5] architectures. These
systems were characterized by a static, two-stage pipeline: a retriever would fetch a set of relevant documents,
and a generator (a Large Language Model) would produce an answer based solely on those retrieved passages.
This static approach is fundamentally limited when faced with complex, multi-faceted research queries that
require iterative refinement, information synthesis from disparate sources, or interaction with the environment
[4]. There has been continuous research in improving these systems through advanced RAG techniques like
Corrective-RAG [16], Self-RAG [1], FLARE [17], IAG [19], SAIL [6] and many more.
To overcome the above mentioned limitations of the agentic system, Deep Research Agents were introduced
which are designed to tackle complex, multi-turn informational research tasks by leveraging a combination of
dynamic reasoning, adaptive long-horizon planning, multi-hop information retrieval, iterative tool use, and
the generation of structured analytical reports. The paper [ 18] complements this by defining the operational
pipeline of a deep research agent. It frames them as a solution to the ”internal knowledge boundaries” of LLMs
and outlines a standard, four-stage workflow: Planning, Question Developing (Decomposing the main query),
Web Exploration (Iterative retrieval and tool use) and Report Generation (Synthesis). Many organizations like
OpenAI [ 7], Google’s Gemini [ 3], Grok [ 15], and Perplexity [ 11], LangChain [ 10] have developed their own deep
research agents which perform in-depth research.
The paper [4] states a clear difference between theStaticandDynamicdeep research agents.
1.Static Workflows: Static workflows rely on manually predefined task pipelines, decomposing research
processes into sequential subtasks executed by dedicated agents.
2.Dynamic Workflows: Dynamic workflows support adaptive task planning, allowing agents to dynamically
reconfigure task structures based on iterative feedback and evolving contexts.
We introduce aconfigurableandhierarchical Tree-basedstatic Deep Research Agent which has the
capability to perform deep research on the topic provided by the end user. The agent is built on a static workflow
which can be configured by the end user to decide how deep they want the agent to research a particular topic.
1arXiv:2512.03887v2  [cs.AI]  4 Dec 2025

We introduce two configurable parameters which can be tuned by the end-user to decide on the depth of the
research to be done by our Deep Research Agent.
We aim to build a configurable agent with deep research capability which can be controlled by the end
user. Integrating with a Large Language Model (LLM) costs money and the pricing is calculated on multiple
parameters (input/output tokens, number of requests etc.). The Deep Research Agent described in this paper
allows users to control the number of requests they make to an integrated LLM at their will. If a research
topic needs an in-depth research the breadth and depth value can be configured to higher value for the agent to
perform a detailed research. While for simpler topics which require a shallow research, the depth and breadth
can be configured to a relatively lower value for the agent to perform a shallow research and hence having a
decent control on the LLM pricing.
The agent framework has asupervisoragent which can further spawnindependent agentsto research on
the sub-topic independently. If a research topic can not be further split into sub-research topics then aworker
agent goes ahead and performs a research on that sub-topic. We also use aWeb Search Toolto look for
citations / references and summarized search results for research sub-topics. The research output of the spawned
worker agents is added to the generated Research Report in a markdown format [ 14] and once the research is
done, the report is saved in a markdown (.md) file in the storage.
2 Static - Deep Research Agent
2.1 Design Overview
We discuss the technical design of the deep research agent. In this section we talk about theSupervisoragent,
theIndependentagent and theWorkeragent. We also talk about how the agent orchestrates all the above
discussed agents to research for a topic and how the agent consolidates the output into the final research report.
At the end we discuss how we can configure the Deep Research Agent with the two parameters “Depth”
and “Breadth” to control how in-depth we want our Deep Research Agent to research a given topic. We should
be mindful that configuring these parameters to a lower value can reduce the cost (for interaction with LLM)
but might also lead to a lower quality research report and vice-versa.
2.2 Core Design and Parameter Tuning
The core design of the Deep Research Agent is incorporated in its hierarchical architecture which is controlled
by two parametersdepthandbreadth.
1.Depth: The depth parameter controls how deep our agent can research for a provided topic. It also
depends upon whether a sub-topic can be further broken into multiple child sub-topics or not.
2.Breadth: The breadth parameter controls how many sub-topics our agent can break a research topic into.
Tuning the breadth parameter into a decent number allows the Large Language Model to research for a
topic which is atomic and focussed on a single topic or goal to research for.
Figure 1 demonstrates, when we provide our Deep Research Agent a research topic to research for, the
agent aims to divide the topic into multiple sub-topics which can be researched independently. The number of
sub-topics spawned from the parent research topic depends upon the breadth parameter. Higher the breadth
parameter, greater the number of sub-topics being spawned.
The number of sub-topics being spawned from the parent research topic not only depends on the breadth
parameter but also on the limitation of the topic to be broken down into independent researchable sub-topics. If
a Research Topic saysRTcan be broken down only into STmaxindependent sub-topics, then the amount of
sub-topics spawned from a parent topic can be represented as.
numberOfSubTopics(RT) =min(ST max, b)
In Figure 2 we demonstrate for a research topic “What are the investment philosophies of Duan Yongping,
Warren Buffett and Charlie Munger” the sub-topic limit is ( STmax) 3. Hence, even though the value ofbreadth
parameter is5, the agent breaks the topic into3independent sub-topics [What are the investment philosophies of
Duan Yongping, What are the investment philosophies of Warren Buffett, What are the investment philosophies
of Charlie Munger].
2

Figure 1: Independent agent breaking down the research topic into sub-topics.
Figure 2: Independent agent breaking down the research topic into maximum possible sub-topics irrespective of
the value of breadth (b) parameter
Our agent aims to research by going into the depth of the provided research topic. This behaviour can be
configured with the depth parameter. The depth parameter lets the agent break a topic into sub-topics. If the
agent reaches the depth limit, then it directly researches that topic, without breaking them further down into
sub-topics.
In Figure 3, we describe for a given Research Topic how we can control how deep the agent can go into it. For
the research topic “What are the investment philosophies of Duan Yongping, Warren Buffett and Charlie Munger”
and thedepthparameter value3, the agent went 3 levels deep researching into [“What are the investment
philosophies of Warren Buffett”, “What are the core tenets of Warren Buffett’s investment philosophy?”, “Define
and explain Warren Buffett’s concept of the ’circle of competence’”]. Once the agent reached the maximum
depth value, it went ahead and researched for that topic without breaking it further into more independent
sub-topics.
3

Figure 3: The depth of the research being controlled by the depth (d) parameter.
The overall research procedure of the Deep Research Agent is demonstrated in Table 1 where the parameters
depth=2,breadth=5and modelgemini-2.5-proare configured for the agent execution. The table explains
how the agent breaks down the topic into independent sub-topics and researches them separately while respecting
the configured breadth and the depth limit.
We also aim to reduce the configuredbreadthvalue by a factor of2, every time the agent researches a
level (depth) deeper. This design decision has been taken to allow the agent to explore the more relevant topics
in-detail by breaking them into greater independent sub-queries while the lower level topics can be less relevant
and need not to be researched in a huge breadth (detail).
4

Table 1: Static Deep Research Agent (gemini-2.5-pro) [depth = 2, breadth = 5]
Supervisor 1: What are
the core tenets of Warren
Buffett’s investment philoso-
phy, including his concepts of
’circle of competence,’ ’eco-
nomic moats,’ and ’margin
of safety’?Supervisor 2: What are
the key elements of Charlie
Munger’s investment philos-
ophy, particularly his empha-
sis on ’mental models’ and
the psychology of human mis-
judgment?Supervisor 3: What is Duan
Yongping’s investment phi-
losophy, and how has it been
influenced by Buffett and
Munger? Focus on his spe-
cific principles like ’Benfen’
and his approach to technol-
ogy and consumer electronics
investments.
Worker 1.1: Define and explain
Warren Buffett’s concept of the
’circle of competence’, detailing
its importance in his investment
decision-making process and how
he advises investors to develop
and adhere to their own.
Worker 1.2: Analyze War-
ren Buffett’s investment principle
of ’economic moats’, describing
the different types of moats he
identifies (e.g., brand, patent,
switching costs, cost advantages)
and providing examples of
companies he has invested in
that exemplify this concept.
Worker 1.3: Investigate
Warren Buffett’s ’margin of
safety’ principle, explaining
how it is defined, its role in
minimizing downside risk, and
its relationship to the calculation
of a company’s intrinsic value.Worker 2.1: Detail Charlie
Munger’s core investment philos-
ophy, focusing on his principles
of value investing, patience, and
the importance of investing in
high-quality businesses. Exclude
his concepts of ’mental models’
and ’psychology of human mis-
judgment’ from this query.
Worker 2.2: Investigate
Charlie Munger’s concept of
’mental models’. Define what he
meant by a ’latticework of mental
models’ and provide a compre-
hensive list and explanation of
the key models he advocated
for, drawing from disciplines like
physics, biology, and economics.
Worker 2.3: Explore Charlie
Munger’s ’Psychology of Human
Misjudgment’. Identify and
explain the primary cognitive
biases and psychological tenden-
cies he believed lead to poor
decision-making in investing,
such as confirmation bias, loss
aversion, and social proof.Worker 3.1: Detail Duan
Yongping’s core investment
philosophy, with a primary
focus on his principle of ’Benfen’.
Explain what ’Benfen’ means and
how it translates into specific,
actionable investment criteria
he follows, such as focusing on
business models, management
quality, and long-term value over
short-term market fluctuations.
Worker 3.2: Analyze the
specific influences of Warren
Buffett and Charlie Munger on
Duan Yongping’s investment
approach. Identify key concepts
he adopted, such as ’circle of
competence’, ’margin of safety’,
and the importance of investing
in great businesses at fair prices.
Provide examples of how he has
adapted and integrated these
concepts with his own unique
perspective.
Worker 3.3: Investigate Duan
Yongping’s investment strategy
and major decisions within
the technology and consumer
electronics sectors. Analyze his
significant investments, such as
Apple, and explain how these
choices exemplify the application
of his ’Benfen’ principle and
Buffett-Munger-inspired philos-
ophy. Contrast his approach
with typical venture capital or
momentum investing in the tech
industry.
With our Deep Research Agent we can decide on the amount of depth and breadth by which we want
our agent to research on a given topic. We ran our Deep Research Agent for a research topic “What are the
investment philosophies of Duan Yongping, Warren Buffett, and Charlie Munger?” at three configurations of
breadth and depth parameters:agent d1b2with depth value1and breadth value2,agent d2b3with depth
value2and breadth value3,agent d2b5with depth value2and breadth value5. The variations in the
size of the generated research report, number of research sub-topics generated, and the scores (evaluated via
DeepResearch Bench[ 2]) is presented in Figure 4. The naming convention of the agent configuration follows
“agent dxby” format wherexis the depth value andyis the breadth value.
5

Figure 4: Evaluation Metrics for multiple agent configurations.
We show that with the increasing value of the depth (d) and breadth (b) parameters, the deep research agent
does a more in-depth research on the provided topic and hence results in a greater overall score. We also present
that with the changing value of depth and breadth configurations, the capability of the Large Language Model
to split a research topic into sub-topics changes and the overall depth of the research process also changes. This
is demonstrated in Figure 5.
6

Figure 5: Increase in the ability to perform in-depth research with increasing value of depth (b) and breadth (b)
parameters.
2.3 Supervisor Agent
The algorithm in Algorithm 1 outlines how a Supervisor Agent makes a decision on whether to split a research
topic into independent sub-topics or not. The supervisor agent also takes the configured depth parameter into
consideration while deciding on splitting the query. The agent interacts with a Large Language Model to decide
whether a query is splittable or not. The algorithm mentioned lays out how the Supervisor agent executes. If the
7

depth parameter has not been exhausted and the research topic can be split into sub-topics, the agent forwards
the topic to the Independent agent.
Algorithm 1:Supervisor Agent
Data:strresearchTopic, integercurrentDepth
Result:Shared Statestate
1canSplit←false;
2ifcurrentDepth >0then
3canSplit←canSplitIntoSubtasks(researchTopic);
4end
5ifcanSplitthen
6returnindependentAgent(researchTopic);
7else
8isUniqueTopic←isDifferentResearchTopic(researchTopic);
9ifisUniqueTopicthen
10returnworker(researchTopic);
11end
12end
If the research topic can not be split into sub-tasks or the depth parameter has been exhausted, the agent
gives the topic to a worker which is responsible for researching the topic with the help of LLM and Web Search
tool. Before executing a worker agent, we do a sanity check if the research topic being given to be researched on
is not equivalent to the previously searched topics. If it is equivalent, we skip the worker execution for that topic.
This sanity check is carried out by theisDifferentResearchTopicmethod and requires an LLM interaction.
2.4 Independent Agent
The algorithm in Algorithm 2 demonstrates how an Independent Agent takes a research topic and breaks them
down into sub-topics which can be researched independently. It spawns 1 Supervisor Agent for every sub-topic.
The agent acts as a parent in the hierarchy which spawns multiple child supervisors which research for the
sub-tasks independently and return their findings back to the parent agent.
Algorithm 2:Independent Agent
Data:integercurrentDepth, integercurrentBreadth
Result:Shared Statestate
1foreachsubQuery∈independentAgentResponse.subQueriesdo
2supervisorResponse←supervisor(subQuery, currentDepth−1, currentBreadth−2);
3supervisorResponses.append(supervisorResponse);
4end
The algorithm in Algorithm 3 demonstrates how a parent independent agent then collects the output of all
the child supervisors and combines them into a unified Research Report response which is returned back to the
caller.
Algorithm 3:Independent Agent
Data:
Result:Shared Statestate
1foreachsupervisorResponse∈supervisorResponsesdo
2pastResearchReports.extend(supervisorResponse["reports"]);
3pastCitations.extend(supervisorResponse["citations"]);
4pastResearchTopics.extend(supervisorResponse["research topics"]);
5end
The Research Report holds these important informations:
1.Current Depth: The current depth value for the research agent. The current depth value is reduced by a
factor of 1 every time an Independent agent passes the sub-query to the Supervisor agent.
8

2.Current Breadth: The current breadth value for the research agent. The current breadth value is
reduced by a factor of 2 every time an Independent agent passes the sub-query to the Supervisor agent.
3.Current Research Topic: The current research topic to be researched on.
4.Past Research Topics: The past research topics which are already being researched from the start of
the deep research agent execution.
5.Past Citations: The past citations collected through the web search while researching on the past research
topics from the start of the deep research agent execution.
6.Past Research Reports: The past research reports which are already being generated from the start of
the deep research agent execution.
The process of unifying the research responses from the child supervisors by the parent independent agent
involves extending the citations, reports and the topics list. The independent agent has a list of citations, reports
and topics which has been collected by the deep research agent execution till that instant. It further adds
the elements in the citations, reports and the topics list returned by every child supervisor into the currently
maintained list.
2.5 Worker Node & Web Search Tool
2.5.1 Worker Node
A worker node is responsible for researching the topic provided to it with the help of a Web Search tool and
Large Language Model. The task of a worker node is to perform an in-depth research on a provided research
topic (by the supervisor). The worker node performs their research in two parts.
2.5.2 Calling a Web Search Tool
The worker node has been provided with a tool calling capability which allows them to search the web for a
given research topic. We useTavily[ 12] for web search and aim to receivetop 5search results for a topic from
the web. We also make sure to have only relevant web search results with us for writing a report for the research
topic. The Tavily Web Search tool provides a score field for every search result returned which defines “the
relevance score of the search result”. We have athreshold scorevalue set to30%which filters out any search
result whose score is less than the threshold score.
We also keep thecited urlsof the relevant search responses for report writing purposes. The search response
has a url field which holds the URL of the search result. The structure of a single search response being returned
from the Web Search tool is mentioned in Listing 1. A deep dive into the documentation of the web search API
can be found on their documentation [13] website.
Listing 1: Web Search Response from Tavily
1{
2" title ": " Lionel Messi Facts | Britannica ",
3" url": " https :// www. britannica . com/ facts / Lionel - Messi ",
4" content ": " Lionel Messi , an Argentine footballer ..." ,
5" score ": 0.81025416,
6" raw_content ": null ,
7" favicon ": " https :// britannica . com/ favicon .png "
8}
2.5.3 Researching the Topic with LLM
Once we have the Web Search response (a collection of most relevant search results), we provide that search
response to a Large Language Model along with the research topic (currently being researched) and a structured
prompt (holding detailed instructions on the research process) and let them perform an in-depth research of the
topic. We have used “gemini-2.5-pro” LLM for this deep research agent and the entire evaluation scores are
based on the same model.
The deep research agent allows an easy switching between the large language models for report writing. If in
future a more capable LLM is released, the current deep research agent can be updated to utilize that model
capabilities very efficiently and with ease.
9

The model’s generated output is added to the generated past research reports list and is further utilized for
the full report writing. Along with the model output we also maintain a list of citations which is a collection of
the urls retrieved from as a part of the most relevant search results. These citations are preserved in the past
citations list throughout the deep research agent runtime and are utilized for final report writing.
2.6 Generated Report structure
The generated report by the Deep Research Agent is divided into three parts: Table of Contents, Report and
Citations. The “Table of Contents” holds the topics researched by the worker agent in the order. Meaning a
topic being researched earlier will come first in the table of contents section. The “Report” section holds the
main content of the Research Report. The research done by the worker agent is written in this section. This
section maintains the order and the hierarchy of the research topics and sub-topics. The “Citation” section holds
the list of cited URLs we received from the Web Search tool (in the search response). Section 6 outlines a sample
research report generated on a research topic mentioned with the configuration ofdepth= 2 andbreadth= 3.
3 Evaluation
We evaluate our Deep Research Agent against the globally accepted benchmark named DeepResearch Bench
[2][8]. It is the primary benchmark for general-purpose DRAs. It consists of 100 PhD-level research tasks across
22 distinct fields. Crucially, it also introduces two novel evaluation frameworks for grading the results:
1.RACE(Reference-based Adaptive Criteria-driven Evaluation): Assesses the quality of the final generated
research report.
2.FACT(Framework for Factual Abundance and Citation Trustworthiness): Assesses the agent’s retrieval
and citation capabilities.
We evaluated our deep research agent against the RACE evaluation framework of the DeepResearch Bench
and with the configuration ofdepth(d) value2andbreadth(b) value5andgemini-2.5-promodel powering
the supervisor and worker agents, our deep research agent was able to achieve an overall score of34.72. The
detailed evaluation is mentioned in Table 2. The evaluation scores of the models have been captured from the
DeepResearch Bench Hugging Face leaderboard [9].
Table 2: RACE (Reference-based Adaptive Criteria-driven Evaluation) by models
Model Overall Comprehensive-
nessInsight Instruction
FollowingReadability
gemini-2.5-pro-deepresearch 49.71 49.51 49.45 50.12 50
openai-deep-research 46.45 46.46 43.73 49.39 47.22
claude-research 45 45.34 42.79 47.58 44.66
static-dra (gemini-2.5-pro) 34.72 35.12 30.45 38.86 35.44
gemini-2.5-pro-preview-05-06 31.9 31.75 24.61 40.24 32.76
gpt-4o-search-preview 30.74 27.81 20.44 41.01 37.6
sonar 30.64 27.14 21.62 40.7 37.46
The scores of our deep research agent evaluated on the DeepResearch Bench forEnglishandChineseare
denoted in Table 3.
Table 3: RACE (Reference-based Adaptive Criteria-driven Evaluation) by languages
Model (Language) Overall Comprehensive-
nessInsight Instruction
FollowingReadability
static-dra (english) 32.65 33.23 26.65 38.20 33.87
static-dra (chinese) 36.80 37.01 34.25 39.52 37.02
The DeepResearch Bench has multiple topics across which we evaluated our deep research agent and the
evaluation metrics across these topics are provided in Table 4. A heatmap of the evaluation scores against all
the supported topics is depicted in Figure 6.
10

Table 4: RACE (Reference-based Adaptive Criteria-driven Evaluation) by topics
static-dra (Topic) Overall Comprehensive-
nessInsight Instruction
FollowingReadability
Finance & Business 32.65 33.96 28.35 36.73 33.04
Science & Technology 32.22 32.68 29.28 35.07 33.54
Software Development 33.71 34.82 28.78 37.16 35.41
Education & Jobs 35.68 35.93 30.90 40.59 36.50
Health 39.22 38.83 34.69 45.93 37.90
Literature 31.16 29.54 23.43 39.09 33.24
History 39.60 40.89 36.55 41.70 40.28
Hardware 37.81 37.88 30.43 43.03 38.99
Industrial 34.58 36.55 29.73 37.52 36.15
Art & Design 34.70 36.72 30.03 38.71 34.92
Games 29.17 25.05 26.08 27.13 44.53
Crime & Law 34.46 34.42 30.62 40.94 31.86
Entertainment 29.77 25.33 23.04 32.11 33.26
Sports & Fitness 34.72 31.25 32.73 42.14 30.70
Software 30.44 30.52 25.46 35.38 33.47
Transportation 35.93 34.70 33.18 39.88 35.75
Religion 37.08 36.45 36.37 41.08 35.28
Home & Hobbies 35.82 38.20 32.91 41.56 29.24
Travel 37.10 41.17 31.68 39.93 35.51
Food & Dining 39.40 39.42 34.23 41.69 45.13
Fashion & Beauty 37.88 37.61 35.05 43.33 35.25
Social Life 44.21 46.21 41.75 48.23 40.72
Figure 6: Score of the Agent across various topics and criteria
4 Conclusion
The Static Deep Research Agent (Static-DRA) presented in this paper introduces a novel, configurable, and
hierarchical tree-based architecture for tackling complex, multi-turn research tasks. Our core contribution is the
design of a static workflow governed by two user-tunable parameters: Depth and Breadth. These parameters
provide a critical mechanism for end-users to exert granular control over the extent of the research process,
directly balancing the desired quality and comprehensiveness of the report against the computational cost
associated with Large Language Model (LLM) interactions.
The agent’s hierarchical design, composed of Supervisor, Independent, and Worker agents, enables effective
multi-hop information retrieval and parallel sub-topic investigation. The dynamic reduction of the Breadth
parameter at deeper levels ensures that the agent intelligently focuses its resources on the most promising
research avenues.
Our evaluation against the established DeepResearch Bench using the RACE framework validates the agent’s
capability. The Static-DRA, configured with a depth of 2 and breadth of 5, achieved an overall score of 34.72,
11

demonstrating competitive performance powered by the gemini-2.5-pro model. Furthermore, our experiments
confirm the fundamental trade-off: increasing the configured Depth and Breadth values results in a more in-depth
research process and a correspondingly higher evaluation score and report size. In conclusion, the Static-DRA
provides a pragmatic and resource-aware solution for deep research, giving users transparent control over the
research intensity.
References
[1]Akari Asai, Zeqiu Wu, Yizhong Wang, Avirup Sil, and Hannaneh Hajishirzi. Self-rag: Learning to retrieve,
generate, and critique through self-reflection, 2023.
[2]Mingxuan Du, Benfeng Xu, Chiwei Zhu, Xiaorui Wang, and Zhendong Mao. Deepresearch bench: A
comprehensive benchmark for deep research agents, 2025.
[3] Google. Gemini deep research.https://gemini.google/overview/deep-research/.
[4]Yuxuan Huang, Yihang Chen, Haozheng Zhang, Kang Li, Huichi Zhou, Meng Fang, Linyi Yang, Xiaoguang
Li, Lifeng Shang, Songcen Xu, Jianye Hao, Kun Shao, and Jun Wang. Deep research agents: A systematic
examination and roadmap, 2025.
[5]Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich
K¨ uttler, Mike Lewis, Wen tau Yih, Tim Rockt¨ aschel, Sebastian Riedel, and Douwe Kiela. Retrieval-
augmented generation for knowledge-intensive nlp tasks, 2021.
[6]Hongyin Luo, Yung-Sung Chuang, Yuan Gong, Tianhua Zhang, Yoon Kim, Xixin Wu, Danny Fox, Helen
Meng, and James Glass. Sail: Search-augmented instruction learning, 2023.
[7]OpenAI. Introducing deep research. https://openai.com/index/introducing-deep-research/ , 2025.
Accessed: 2025-02-03.
[8]DeepResearch Bench Team. Deepresearch bench: A comprehensive benchmark for deep research agents.
https://deepresearch-bench.github.io/, 2025.
[9]HuggingFace Team. Deepresearch bench: Leaderboard. https://huggingface.co/spaces/muset-ai/De
epResearch-Bench-Leaderboard, 2025.
[10]LangChain Team. Open deep research. https://github.com/langchain-ai/open_deep_research , 2025.
Accessed: 2025-07-16.
[11]Perplexity Team. Introducing perplexity deep research. https://www.perplexity.ai/hub/blog/introdu
cing-perplexity-deep-research, 2025. Accessed: 2025-02-14.
[12] Tavily Team. Web search - connect your agent to the web.https://www.tavily.com/.
[13]Tavily Team. Web search documentation. https://docs.tavily.com/documentation/api-reference/e
ndpoint/search.
[14] The Markdown Team. Markdown guide.https://www.markdownguide.org/.
[15]xAI Team. Grok 3 beta — the age of reasoning agents. https://x.ai/news/grok-3 , 2025. Accessed:
2025-02-19.
[16]Shi-Qi Yan, Jia-Chen Gu, Yun Zhu, and Zhen-Hua Ling. Corrective retrieval augmented generation, 2024.
[17]Liang Zhang, Katherine Jijo, Spurthi Setty, Eden Chung, Fatima Javid, Natan Vidra, and Tommy Clifford.
Enhancing large language model performance to answer questions and extract information more accurately,
2024.
[18]Wenlin Zhang, Xiaopeng Li, Yingyi Zhang, Pengyue Jia, Yichao Wang, Huifeng Guo, Yong Liu, and Xiangyu
Zhao. Deep research: A survey of autonomous research agents, 2025.
[19]Zhebin Zhang, Xinyu Zhang, Yuanhang Ren, Saijiang Shi, Meng Han, Yongkang Wu, Ruofei Lai, and Zhao
Cao. Iag: Induction-augmented generation framework for answering reasoning questions, 2023.
12

5Computing the number of research topics being provided to the
Worker
By configuring the depth and breadth parameters we can also control the amount of queries being spawned by
our Deep Research Agent for a given Research Topic. This can also be used to have a control / limitation on the
interactions with the large language model. This can be calculated through the following mathematical steps.
Configured Depth Value =d
Configured Breadth Value =b
Let’s assume the number of sub-topics spawned at depth k is ns(k)
Maximum number of subtopics spawned at depth 1 =ns(1) =b= (b−0∗2)
Maximum number of subtopics spawned at depth 2 =ns(1)∗(b−1∗2)
Maximum number of subtopics spawned at depth 3 =ns(2)∗(b−2∗2)
. . .
Maximum number of sub-topics spawned at depth k =ns(k−1)∗[b−(k−1)∗2]
Maximum number of sub-topics spawned at depth d =ns(d−1)∗[b−(d−1)∗2]
Since the maximum number of sub-topics spawned at the lowest level is provided to the workers for re-
search via interaction with the Large Language models, we can safely say that the maximum number of topics
provided to the LLM for research is equivalent to ns(d).
ns(d) =ns(d−1)∗[b−(d−1)∗2]
ns(d) =ns(d−2)∗[b−(d−2)∗2]∗[b−(d−1)∗2]
ns(d) =b∗(b−1∗2)∗. . .∗[b−(d−2)∗2]∗[b−(d−1)∗2]
Hence,
ns(d) =Qd−1
i=0(max((b−2∗i),1))
6 Sample Deep Research Report
6.1 Deep Research Topic
What are the investment philosophies of Duan Yongping, Warren Buffett, and Charlie Munger?
6.2 Generated Deep Research Report
Deep Research Report
Table of Contents
1. What is the investment philosophy of Duan Yongping?
2. What is the investment philosophy of Warren Buffett?
3. What is the investment philosophy of Charlie Munger?
Report
What are the investment philosophies of Duan Yongping, Warren Buffett, and Charlie Munger?
What is the investment philosophy of Duan Yongping?
Duan Yongping’s investment philosophy is rooted in **value investing**, with a focus on long-term, high-
conviction bets in companies he thoroughly understands. His approach is often compared to that of Warren
Buffett, emphasizing simplicity and a deep understanding of business fundamentals over chasing trends.
Core Principles of Duan Yongping’s Investment Philosophy:
13

•Value-Driven and Long-Term Horizon: Yongping is known for his profound value investment
philosophy, seeking out high-quality assets that are undervalued by the market. His decisions are guided by
long-term impacts rather than immediate benefits, embodying a ”do the right thing” mentality with a long-
term vision (https://www.binance.com/en/square/post/18537661287098). This strategy is characterized
as ”high-conviction, value-driven, long-term investing”.
•Concentrated Portfolio: Unlike investors who diversify across hundreds of stocks, Yongping prefers a
concentrated portfolio, typically holding only 8-12 stocks. This reflects his high-conviction approach, where
he invests significantly in a few businesses he believes in (https://www.gainify.io/blog/duan-yongping-
portfolio).
•Deep Understanding Over Chasing Trends: A cornerstone of his philosophy is to gain a deep,
essential understanding of a company’s business model, products, and user needs. He advocates for
selective learning and avoids blindly chasing new things or imitating trends, a crucial discipline in an
era of information overload (https://www.binance.com/en/square/post/18537661287098). His business
philosophy also includes being cautious about the risks and uncertainties associated with new technologies
and products (https://link.springer.com/content/pdf/10.1007/978-981-95-0545-6 10.pdf).
•Adaptability within a Value Framework: While a disciplined value investor, Yongping demonstrates
adaptability. He is willing to embrace growth opportunities, as shown by his investments in companies like
Alphabet and NVIDIA, when the long-term economics are favorable. This indicates that his value framework
is flexible enough to incorporate growth assets (https://www.gainify.io/blog/duan-yongping-portfolio).
•Simplicity and Consistency: His investment style is consistent with his life philosophy, which centers
on simplifying complexities and focusing on the essence (https://www.alphaexponent.net/p/22-duan-the-
dilettante, https://www.binance.com/en/square/post/18537661287098). This approach involves focusing
on doing one thing well and maintaining a clear, uncluttered mindset.
What is the investment philosophy of Warren Buffett?
Warren Buffett’s investment philosophy is a disciplined, principle-based approach to investing that has generated
significant wealth over several decades (Simply Ethical). It is rooted in the value investing school of Benjamin
Graham, but Buffett has expanded upon these initial principles (Investopedia). His strategy serves as a guide for
long-term investors, emphasizing wealth creation while avoiding the pitfalls of short-term speculation (Simply
Ethical).
The core tenets of his philosophy include:
•Focus on Value Investing: At its heart, Buffett’s strategy is about value investing. This involves
identifying a company’s intrinsic value by analyzing its business fundamentals, such as earnings, revenue,
and assets (Investopedia). The goal is to purchase these companies when they are undervalued, particularly
during market downturns when prices become more attractive (IIFL Capital, Investopedia).
•Invest in Quality Businesses: Buffett’s philosophy has evolved from just buying cheap stocks to investing
in ”wonderful businesses at ’fair’ valuations” (Investor.fm, HBR). He describes a good business as a ”castle”
with a protective ”moat” that management should continuously widen. These high-quality businesses
are often unique franchises that consistently generate cash (HBR). However, finding such businesses that
remain good investments over time can be difficult (Investor.fm).
•Maintain a Long-Term Perspective: Buffett’s principles are designed for long-term success. He invests
in a company based on its underlying business quality, not on whether the market will soon recognize its
worth (Investopedia). This approach acts as a shield against ”the most common pitfalls of speculation and
short-termism” (Simply Ethical).
•Demand Capable and Transparent Management: The quality and transparency of a company’s
management are non-negotiable elements in Buffett’s approach (Investor.fm). His proposition to managers
of the businesses he invests in is that if their company generates cash, they can trust him to reinvest it
wisely (HBR).
•Emphasize Simplicity: Buffett advocates for a simple investment strategy (IIFL Capital). For investors
who may not have the time or expertise to analyze individual businesses, he has recommended periodically
investing in an index fund, stating that a ”know-nothing investor can actually out-perform most investment
professionals” this way (IIFL Capital).
14

•Practice Patience and Emotional Discipline: A key takeaway from Buffett’s philosophy is that
extraordinary results can be achieved through patience and discipline (Simply Ethical). This involves
managing emotions and having the patience to wait for the right opportunities (IIFL Capital). Part of this
discipline includes keeping cash reserves available to deploy when market conditions are favorable (IIFL
Capital).
While timeless, implementing Buffett’s philosophy today has its challenges. Finding wonderful businesses at
fair prices in the current market is difficult, and Buffett’s own massive portfolio puts him at a competitive
disadvantage compared to those managing smaller sums (Investor.fm). Nonetheless, his core principles offer
clear guidance: focus on quality, maintain a long-term view, and keep the strategy simple (IIFL Capital).
What is the investment philosophy of Charlie Munger?
The Investment Philosophy of Charlie Munger
Charlie Munger, the long-time business partner of Warren Buffett, was a legendary investor whose philos-
ophy was built on a foundation of discipline, rationality, and a deep understanding of business fundamentals.
His approach can be summarized as buying great companies at fair prices and holding them for the long term.
Core Tenets of Munger’s Philosophy:
•Focus on Quality Businesses: The cornerstone of Munger’s philosophy is that the ”real money is in
great companies”. His strategy prioritized quality over quantity, seeking out high-quality businesses to
invest in.
•Invest within Your ”Circle of Competence”: Munger was a staunch advocate for only invest-
ing in businesses that you can thoroughly understand. He called this principle the ”circle of com-
petence” and made it clear that if you don’t deeply understand a business, you should not buy its
stock (https://www.ruleoneinvesting.com/blog/personal-development/5-charlie-munger-investing-lessons-
every-value-investor-should-live-by). This approach isn’t about picking stocks, but about buying into
businesses you comprehend.
•Patience and Long-Term Holding: Munger firmly believed in the long-term holding of quality
investments (https://www.investinassets.net/p/10-lessons-from-charlie-munger). His strategy was not
focused on short-term market movements but on the long-term growth and value creation of the companies
in his portfolio.
•Concentrated Portfolio: Unlike the common advice to diversify broadly, Munger preferred a non-
diversified, concentrated portfolio (https://www.fool.com/investing/how-to-invest/famous-investors/charlie-
munger/). This reflects his confidence in making a few high-conviction bets on excellent companies, often
suggesting a portfolio of 10-20 high-quality compounders (https://www.investinassets.net/p/10-lessons-
from-charlie-munger).
•Rationality and Discipline: A key element of his success was an ”unmatched rationality and discipline”
that he maintained through numerous market cycles (https://www.investinassets.net/p/10-lessons-from-
charlie-munger). His philosophy emphasized staying calm during market volatility and steering clear of
common behavioral mistakes that trap many investors.
•Capitalizing on Market Inefficiency: Munger’s approach was rooted in capitalizing on market ineffi-
ciencies (https://finance.yahoo.com/news/charlie-mungers-timeless-investment-advice-113042652.html).
This involves identifying opportunities where the market has mispriced a great business, allowing for the
purchase of a quality asset at a fair price.
Citations
•https://investor.fm/unpacking-buffetts-investment-philosophy-a-personal-perspective/
•https://www.binance.com/en/square/post/18537661287098
•https://www.fool.com/investing/how-to-invest/famous-investors/charlie-munger/
•https://www.investinassets.net/p/10-lessons-from-charlie-munger
•https://www.ruleoneinvesting.com/blog/personal-development/5-charlie-munger-investing-lessons-every-value-
investor-should-live-by
15

•https://www.gainify.io/blog/duan-yongping-portfolio
•https://www.iiflcapital.com/blog/personal-finance/lessons-from-warren-buffetts-investment-philosophy
•https://www.binance.com/en/square/post/13966098863626
•https://simplyethical.com/blog/warren-buffetts-investment-tenets/
•https://hbr.org/1996/01/what-i-learned-from-warren-buffett
•https://www.investopedia.com/articles/01/071801.asp
•https://www.alphaexponent.net/p/22-duan-the-dilettante
•https://finance.yahoo.com/news/charlie-mungers-timeless-investment-advice-113042652.html
•https://link.springer.com/content/pdf/10.1007/978-981-95-0545-6_10.pdf
16