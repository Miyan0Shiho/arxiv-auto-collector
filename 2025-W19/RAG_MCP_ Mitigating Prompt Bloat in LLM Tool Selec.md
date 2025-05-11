# RAG-MCP: Mitigating Prompt Bloat in LLM Tool Selection via Retrieval-Augmented Generation

**Authors**: Tiantian Gan, Qiyao Sun

**Published**: 2025-05-06 08:05:35

**PDF URL**: [http://arxiv.org/pdf/2505.03275v1](http://arxiv.org/pdf/2505.03275v1)

## Abstract
Large language models (LLMs) struggle to effectively utilize a growing number
of external tools, such as those defined by the Model Context Protocol
(MCP)\cite{IntroducingMCP}, due to prompt bloat and selection complexity. We
introduce RAG-MCP, a Retrieval-Augmented Generation framework that overcomes
this challenge by offloading tool discovery. RAG-MCP uses semantic retrieval to
identify the most relevant MCP(s) for a given query from an external index
before engaging the LLM. Only the selected tool descriptions are passed to the
model, drastically reducing prompt size and simplifying decision-making.
Experiments, including an MCP stress test, demonstrate RAG-MCP significantly
cuts prompt tokens (e.g., by over 50%) and more than triples tool selection
accuracy (43.13% vs 13.62% baseline) on benchmark tasks. RAG-MCP enables
scalable and accurate tool integration for LLMs.

## Full Text


<!-- PDF content starts -->

RAG-MCP: Mitigating Prompt Bloat in LLM
Tool Selection via Retrieval-Augmented
Generation
Tiantian Gan1,2and Qiyao Sun1,2
1Beijing University of Post and Communications, Beijing, China
2Queen Mary University of London, London, UK
jp2022213034@qmul.ac.uk, jp2022213402@qmul.ac.uk
Abstract. Large language models (LLMs) struggle to effectively utilize
a growing number of external tools, such as those defined by the Model
Context Protocol (MCP)[1], due to prompt bloat and selection complex-
ity. We introduce RAG-MCP, a Retrieval-Augmented Generation frame-
work that overcomes this challenge by offloading tool discovery. RAG-
MCP uses semantic retrieval to identify the most relevant MCP(s) for a
given query from an external index before engaging the LLM. Only the
selected tool descriptions are passed to the model, drastically reducing
prompt size and simplifying decision-making. Experiments, including an
MCP stress test, demonstrate RAG-MCP significantly cuts prompt to-
kens (e.g., by over 50%) and more than triples tool selection accuracy
(43.13% vs 13.62% baseline) on benchmark tasks. RAG-MCP enables
scalable and accurate tool integration for LLMs.
Keywords: Retrieval-Augmented Generation ·Model Context Protocol
·Tool Selection
1 Introduction
1.1 Background and Motivation
Large Language Models (LLMs) have demonstrated remarkable capabilities in
natural dialogue, reasoning, and even code generation. However, they remain
fundamentally constrained by the knowledge encoded in their parameters and
the fixed context window available at inference time. In essence, an LLM with-
out external access is “trapped” with only its training data and cannot easily
update its knowledge or perform actions in the world [12]. To address this lim-
itation, recent efforts have focused on augmenting LLMs with external tools
andfunction-calling abilities[3].Byinvokingtools(e.g.websearch,databases,
calculators) via defined functions or APIs, an LLM can fetch up-to-date infor-
mation and execute complex operations beyond its built-in repertoire [12]. This
paradigm - often referred to as zero-shot tool use or function calling — allows
AI assistants to interface with the latest data and services, unlocking applica-
tions from real-time knowledge queries to specialized tasks in finance and travelarXiv:2505.03275v1  [cs.AI]  6 May 2025

2 Gan and Sun
planning [3]. In fact, major AI providers have embraced this trend: for exam-
ple, leading LLM platforms now support plugin APIs and structured function
calls so that models like GPT-4 or Claude can invoke external services through
well-defined interfaces [12].
In the research community, a variety of approaches have been proposed to
enableandimproveLLMtooluse.Prompt-basedstrategiessuchas ReActinter-
mix reasoning steps with action commands, allowing an LLM to decide when to
consult a tool in the context of a multi-turn “thought process” [15]. Model-centric
approaches have also emerged: for instance, Toolformer fine-tunes an LLM to
autonomously decide which API to call, when to call it, and how to incorporate
the result, given only a handful of demonstrations per tool [13] Other researchers
have improved tool-use by incorporating it into training data and model tuning.
This includes blending function call demonstrations into instruction-following
datasets and exploring prompt formats that effectively describe available func-
tions to the model [3]. Such efforts have markedly enhanced zero-shot tool usage
performance. For example, fine-tuning a model on API call tasks with extensive
tool-use data can yield impressive results – the Gorilla system augmented a
7B LLaMA-based model with relevant API documentation retrieval, enabling it
to outperform even GPT-4 in generating correct API calls for a wide range of
tools [12]. An important insight from these works is that providing just-in-time
relevant context (be it through optimized prompts or retrieved documentation)
greatly boosts the accuracy of an LLM’s tool selection and use, while mecha-
nisms for the model to explicitly decide on tool use (such as special decision
tokens for “answer vs. act”) can further improve reliability [3].
Despite this progress, a new challenge arises as we scale up the number of
tools available to an LLM. Most prior studies and deployments consider a rel-
atively small set of tools or APIs, often hand-picked and easy for the model
to handle within a prompt [12]. In practice, however, the ecosystem of tools is
rapidly expanding. For instance, Anthropic’s recently introduced Model Con-
text Protocol (MCP) defines a universal, open standard for connecting AI
systems with external data sources and services. MCP enables a single assistant
tointerfacewithmanydatarepositoriesandbusinesstoolsthroughaunifiedpro-
tocol, replacing fragmented one-off integrations. As a result, an advanced LLM
agent could soon have dozens of functions at its disposal – from Google Drive
and Slack connectors to GitHub, databases, maps, and more – all registered as
MCP “tools” it can call [1]. This proliferation of available tools brings significant
hurdles.
Prompt Bloat is one critical issue: providing the definitions or usage in-
structions for every possible tool in the model’s context would consume an enor-
mous number of tokens and risk overwhelming the model. It has been observed
that it is effectively impossible to describe a large collection of APIs or tools
in a single prompt as their number grows, and many APIs have overlapping
functionalities with only nuanced differences. Including too many at once not
only exhausts the context length, but can also confuse the model – the func-
tions may start to blur together. This leads directly to a second issue: decision

RAG-MCP 3
overhead . With a long list of tools (many of them similar in scope), the model
faces a more complex decision when choosing if and which tool to invoke. The
greater the choice, the higher the chance of error, such as selecting an suboptimal
tool or misinterpreting what a tool does. Indeed, even state-of-the-art models
can misfire in such settings: for example, in a scenario with numerous API op-
tions, GPT-4 was reported to hallucinate an API that doesn’t actually exist, and
Anthropic’s Claude picked the wrong library for the user’s request [12]. These
failure cases underscore that naively scaling up the toolset can degrade
an LLM’s performance , due to both the capacity strain on the prompt and
the ambiguity in the model’s decision process.
Fig. 1.Comparation between MCP and RAG-MCP during inference
To tackle these challenges, we propose RAG-MCP , a solution that mar-
ries Retrieval-Augmented Generation (RAG) with the Model Context Protocol
framework. The key idea of RAG-MCP is to avoid presenting all tools to the lan-
guage model at once, and instead dynamically retrieve a relevant subset of tools
based on the user’s query. In our approach, the numerous available tool descrip-
tions (MCP function schemas, usage examples, etc.) are stored in an external
memory indexed by their semantics. When a new query arrives, a dedicated re-
triever(e.g.avector-spacesemanticsearch)firstselectsthetop- kcandidatetools
that are most likely to be useful for that query. Only these ktool descriptions are
then injected into the LLM’s prompt (or provided via the function-calling API),
greatly reducing context length and complexity. This retrieval step serves as a
form of focused context filtering , which cuts down on prompt bloat and guides
the model’s choice. The approach is analogous to how retrieval-augmented QA
systems work: rather than feed the entire Wikipedia to the model, one retrieves
only the relevant articles [6]. Here, instead of static knowledge, we retrieve ac-
tionable toolknowledge on the fly. An added benefit is extensibility – because
the tool information lives in an external index, new tools or updated APIs can

4 Gan and Sun
be incorporated by updating that index without retraining the LLM, ensuring
the system remains up-to-date [12]. In short, retrieval helps tame the grow-
ing toolset by providing the right tools at the right time, thereby reducing the
model’s decision burden.
1.2 Contributions
In summary, this paper makes the following contributions:
1.RAG-MCPFramework :Weintroduceanovelarchitecturethatintegrates
a retrieval mechanism with LLM function calling in the MCP setting. To our
knowledge, this is one of the first frameworks to enable an LLM to handle a
largearsenaloftoolsbyqueryingatoolrepositoryforrelevantoptionsinstead
of naively prompting with all tools. This design retains the flexibility of the
open MCP ecosystem while imposing structure to maintain tractability.
2.Scalable Tool Retrieval : We develop a semantic tool retrieval module that
represents each available tool’s description in a vector space and efficiently
matches user queries to the most pertinent tools. This significantly reduces
prompt size and complexity (mitigating prompt bloat) and improves decision
makingbynarrowingthechoices.TheLLM,guidedbythisretrievedcontext,
can more accurately select and use the correct external tool, even as the total
number of tools grows large. Notably, our approach allows new tools to be
added on the fly by indexing them, without requiring additional fine-tuning
of the LLM.
3.Improved Tool-Use Performance : Through comprehensive experiments,
wedemonstratethatRAG-MCPeffectivelyaddressestheperformancedegra-
dation that occurs with naively scaling up the tool set. On a suite of tool-
augmented NLP tasks, we show that as the number of available functions
increases, a baseline LLM’s success rate in selecting and executing the cor-
rect tool drops markedly (illustrating the aforementioned challenge). How-
ever, under the RAG-MCP strategy, the model’s performance is largely re-
stored to its original level, and in some cases even exceeds the small-toolset
baseline. In particular, RAG-MCP yields substantially higher accuracy in
choosing the appropriate tool and reduces errors such as hallucinated or
mis-parameterized function calls. These results underscore the efficacy of us-
ing retrieval to scale up tool-use: the proposed method enables an LLM to
maintain high tool-selection accuracy and reliability even with a large pool
of tools, paving the way for more scalable and capable tool-augmented AI
systems.
Overall, our work demonstrates that the integration of retrieval-based con-
text management is a promising direction to counteract the challenges of tool
proliferation in LLMs. By enabling models to learn which tool to use out of many
and only providing information for those tools, RAG-MCP offers a practical
solution for the next generation of AI agents operating with extensive toolkits.
It combines the strengths of retrieval augmentation and standardized tool APIs
to ensure that more tools do not mean worse performance but rather a broader
range of skills that the model can deploy accurately and efficiently.

RAG-MCP 5
2 Related Work
2.1 Tool Use in LLMs
LLMs have been augmented with external tools to overcome limitations in arith-
metic, retrieval, and code execution. Toolformer demonstrates a self-supervised
method by which a model learns when and how to call APIs such as calculators
or search engines, improving zero-shot performance across tasks [13]. ReActin-
terleaves chain-of-thought reasoning with action steps to interact with external
environments (e.g., a Wikipedia API), yielding more interpretable and accurate
multi-stepsolutions[15]. WebGPT fine-tunesGPT-3inasimulatedbrowseren-
vironment, training it to navigate, search, and cite sources for long-form Q&A,
reducing hallucinations via grounded retrieval [9]. More recently, ChatGPT
Plugins introduced aproduction pluginecosystem, allowing ChatGPTto access
up-to-date information and third-party services in a controlled, safety-oriented
framework [11].
2.2 Retrieval-Augmented Generation
Retrieval-Augmented Generation (RAG) first combined parametric LLMs with
non-parametric memory in a dense vector index, retrieving relevant passages at
inference time to improve knowledge-intensive tasks [6]. Subsequent work has
extended RAG to broad NLP paradigms, including modular and advanced RAG
variants that dynamically adapt retrieval per token or per query [4]. RAG’s
decoupling of memory access and generation inspires our MCP-RAG approach,
wherein MCP discovery is treated as a retrieval subproblem, orthogonal to core
text generation.
2.3 Model Context Protocol
The Model Context Protocol standardizes LLM-to-API interactions by bundling
resource prompts, authentication, and parameter schemas into modular “MCP”
servers.MCPsactasfunction-callextensions,similartoOpenAI’sfunction-calling
API, but with greater community extensibility. The rapid growth of MCP repos-
itories (4,400+ servers on mcp.so as of April 2025 [14]) underscores the need for
scalable discovery and validation mechanisms .
3 Methodology
Overview. We study how the number of available MCP servers affects an LLM’s
abilitytoselectandinvokethecorrecttool(“promptbloat”)andpresentMCP-RAG,
a retrieval-augmented framework that mitigates this degradation by dynamically
retrieving only the most relevant MCP for each query.

6 Gan and Sun
3.1 Prompt Bloat and the MCP Stress Test
Modern LLMs must often choose among many possible external tools, each de-
scribed by an MCP schema. As the count of MCPs grows, including all their
descriptions in a single prompt leads to prompt bloat: the context window be-
comes saturated with distractors, reducing the model’s capacity to distinguish
and recall the correct tool.
This phenomenon parallels the Needle-in-a-Haystack (NIAH) test, which em-
beds a random fact (the “needle”) in the middle of a long context (the “haystack”)
and measures an LLM’s ability to retrieve it under varying context lengths and
depths [6] [10] . In NIAH, performance drops sharply as the haystack grows,
revealing limits of in-context retrieval.
Inspired by NIAH, we design an MCP stress test on WebSearch tasks:
for each trial, we present the model with NMCP schemas (one ground-truth
andN−1distractors) and ask it to select and invoke the correct WebSearch
MCP. We vary Nfrom 1 to 11100 in 26 intervals, measuring selection accu-
racy, task success, prompt token usage, and latency. This setup quantifies how
tool-selection ability degrades with increasing MCP pool size.
3.2 RAG-MCP Framework
To overcome prompt bloat, RAG-MCP applies Retrieval-Augmented Gen-
eration (RAG) principles to tool selection. Instead of flooding the LLM with
all MCP descriptions, we maintain an external vector index of all available MCP
metadata. At query time:
1.Retrieval. A lightweight LLM-based retriever (e.g., Qwen) encodes the
user’s task description and performs a semantic search over the MCP in-
dex, returning the top- kcandidate MCPs most similar to the task [6].
2.Validation. For each retrieved MCP, RAG-MCP can generate a few-shot
examplequeryandtestitsresponsetoensurebasiccompatibility,functioning
as a “sanity check” before invocation.
3.Invocation. Only the single best MCP description, including its tool-use
parameters, is injected into the LLM prompt or function-calling API, which
then performs planning and execution without concern for tool discovery [2].
This design yields several benefits:
– Reduced Prompt Size. By supplying only relevant MCP metadata, RAG-
MCP avoids context window overload even when the full tool registry is
large.
– LowerCognitiveLoad. TheLLMnolongerneedstosiftthroughhundreds
of distractors, improving selection accuracy and reducing hallucinations [2].
– Resource Efficiency. Unlike conventional MCP clients (e.g., Claude or
early GPT-4 integrations) that must instantiate all registered MCP servers
before interaction, MCP-RAG activates only the selected MCP, lowering
startup cost and enabling support for arbitrarily large toolsets without in-
frastructure bottlenecks [10].

RAG-MCP 7
– Multi-Turn Robustness. In dialogues spanning many turns, the LLM
need not re-include all MCP prompts; RAG-MCP’s retriever handles tool
recall dynamically, freeing context space for task-specific reasoning.
3.3 Three-Step Pipeline Diagram
WesummarizeRAG-MCP’soperationinthreecoresteps.Theflowchartisshown
in Fig. 3:
1.Task Input →Retriever : The user’s natural-language task is encoded
and submitted to the retriever.
2.Retriever →MCP Selection & Validation : The retriever searches the
vector index of MCP schemas, ranks candidates by semantic similarity, and
optionally tests each via synthetic examples.
3.LLM Execution with Selected MCP : The LLM receives only the se-
lected MCP’s schema and parameters and executes the task via the function-
calling interface.
Fig. 2.RAG-MCP pipeline: (1) encode user query with Qwen-max, (2) re-trieve &
validate top-k MCPs, and (3) invoke chosen MCP
By decoupling tool discovery from generation, RAG-MCP ensures that LLMs
can scale to hundreds or thousands of MCPs without suffering prompt bloat or
decision fatigue, much as RAG systems avoid overwhelming an LLM with entire
corpora by retrieving only relevant passages.
3.4 Discussion
Ourmethodologycombinestherigorof stresstesting (viatheMCPstresstest)
with the effectiveness of retrieval-augmented tool use. The stress test quantifies
thesharpperformancedropthatoccurswhendistractorMCPs swelltheprompt,
mirroring long-context recall failures in NIAH evaluations [5]. RAG-MCP then

8 Gan and Sun
counteracts this by dynamically narrowing the toolset, reducing both prompt to-
kens and decision complexity, and thereby restoring—and often improving—task
success rates.
Furthermore, by using an external index, RAG-MCP remains extensible: new
MCPs can be added by indexing their metadata, without retraining the LLM.
And by selectively activating servers on demand, it sidesteps the practical limits
on simultaneous MCP instantiation faced by prior tool-augmented LLM deploy-
ments.
4 Experiments
4.1 Stress Test
SetupTo quantify how an LLM’s tool-selection ability scales with the size of
the MCP pool, we conduct a stress test in which the number of candidate MCP
servers, N, is varied from 1 to 11100 in intervals, while the key MCP server
located from the top to the bottom. For each value of N, we randomly select
one “ground-truth” MCP (i.e., the only server capable of satisfying the task
requirement) and N−1distractor MCPs drawn from our full registry of over
4,400 publicly listed servers [14]. This design ensures that exactly one in every
Ncandidates is relevant. We then present the model with each of 20web-search
tasks, requiring it to (a) choose the correct MCP, (b) issue a valid query or
answer, and (c) return the final result.
Fig. 3.This figure illustrates per-trial success across MCP positions from 1 to 11100,
where yellow denotes successful selection and purple denotes failure.

RAG-MCP 9
Results Figure3 plots selection accuracy and task success as Nincreases. We
observe a clear non-monotonic trend: These results quantitatively confirm that
while MCP-RAG greatly mitigates prompt bloat and maintains high perfor-
mance in small to moderate pools, its retrieval precision and overall throughput
degrade as the tool registry scales to thousands of MCPs.“‘
4.2 RAG-MCP
SetupWe evaluated all methods in the web search subset of MCPBench [8],
which we used as our heldout testbed. For each baseline, we perform 20 inde-
pendent trials, and we deem a baseline successful if it produces more than 10
correct answers out of those 20. Within each trial, the model may engage in up
to 10 rounds of interaction with the MCP servers in order to arrive at its final
response.
To assess answer correctness in an automated and reproducible manner, we
employ Deepseek-v3 [7] as our evaluator. Because MCP servers require exter-
nal network access—and can therefore be sensitive to latency or transient fail-
ures—we enforce a controlled network environment throughout all experiments,
ensuring no requests fail due to connectivity issues. Finally, all trials are driven
by qwen-max-0125 as our underlying base LLM.
Baselines We evaluate three selection strategies in our experiments:
1.Blank Conditioning : Prompt the LLM with all NMCP descriptions at
once and ask it to choose the correct one.
2.Actual Match : Pre-filter the candidate pool using simple keyword matching
on the task description and MCP metadata, then prompt the model on this
reduced set.
3.RAG-MCP : Employ our vector-index retriever to semantically rank all N
MCPs and inject only the top candidate’s schema into the LLM prompt for
execution.
Metrics We evaluate performance using three key metrics for each baseline
method:
– Accuracy(%) :Percentageoftrialsinwhichthemodelselectedtheground-truth
MCP.
– Avg Prompt Tokens : Mean number of tokens consumed by the prompt,
including injected MCP metadata.
– Avg Completion Tokens : Mean number of tokens generated by the model
as its final output.
Judgment of the final answer is automated using a Llama-based verifier (“Llama
as Judge”) to compare model outputs against ground truth.

10 Gan and Sun
Baseline Accuracy (%) Avg Prompt Tokens Avg Completion Tokens
MCP-RAG 43.13 1084.00 78.14
Actual Match 18.20 1646.00 23.60
Blank 13.62 2133.84 162.25
Table 1. Baseline performance comparison on accuracy and token usage
Results Table1summarizestheperformanceoftheevaluatedbaselinemethods,
clearly demonstrating the effectiveness of MCP-RAG:
As the table shows, MCP-RAG achieves the highest accuracy at 43.13%,
significantly outperforming the Actual Match and Blank Conditioning meth-
ods, which scored 18.20% and13.62%, respectively. Furthermore, MCP-RAG
notably reduces the average number of prompt tokens to 1084, reflecting a
substantial reduction compared to the other baselines, especially Blank Condi-
tioning, which requires 2133.84 tokens. While MCP-RAG shows an increase in
completion tokens ( 78.14) compared to Actual Match ( 23.60), this trade-off is
beneficial as it correlates with a higher accuracy and overall task success rate.
5 Analysis
Stress Test Analysis Figure 3 illustrates per-trial success across MCP po-
sitions from 1 to 11100, where yellow denotes successful selection and purple
denotes failure. We observe that:
– High Early-Stage Success : MCP positions below 30 exhibit predomi-
nantlyyellowregions,indicatingsuccessratesabove90%whenthecandidate
pool is minimal.
– Mid-Range Variability : In the range of positions 31–70, clusters of purple
emerge intermittently, reflecting lower accuracy as semantic overlap among
MCP descriptions increases.
– Performance Degradation at Scale : Beyond position ~100, purple domi-
nates, signifying that retrieval precision diminishes when handling very large
tool registries.
– Residual Success Islands : Occasional yellow patches at higher positions
suggest that certain MCPs remain well-aligned to specific queries, providing
robustness even in extensive pools.
These patterns confirm that while MCP-RAG effectively curbs prompt bloat and
maintains high accuracy in small to moderate MCP pools, retrieval precision
challenges arise as the total number of MCPs grows, motivating future work on
hierarchical or adaptive retrieval mechanisms.
5.1 Analysis of RAG-MCP Results
The superior performance of RAG-MCP can be attributed to several factors:

RAG-MCP 11
– Focused Context Filtering : By injecting only the single most relevant
MCP schema, the model avoids the distraction caused by irrelevant tool
descriptions, resulting in clearer decision boundaries.
– Prompt Efficiency : The dramatic reduction in prompt tokens allows the
model to allocate more of its context window to reasoning about the task
itself rather than parsing extraneous metadata.
– Balanced Generation : Although RAG-MCP slightly increases completion
token usage relative to Actual Match, this overhead reflects more thorough
reasoning and verification steps, which correlate with higher accuracy.
Overall, these findings confirm that retrieval-augmented selection of MCPs
effectively tames prompt bloat and enhances an LLM’s tool-selection reliability,
making RAG-MCP a compelling solution for scalable external tool integration.
6 Conclusion
We present RAG-MCP , a simple yet powerful framework that tames large
MCP toolsets by retrieving only the most relevant schema for each query. With
focused retrieval, RAG-MCP:
– Drastically reduces prompt size , cutting token usage by over half com-
pared to feeding all tools at once.
– Boosts selection accuracy , more than tripling the success rate of naïve
and keyword-based methods under heavy load.
– Maintains extensibility , since new MCPs can be indexed on–the–fly with-
out retraining the model.
In essence, RAG-MCP turns a sprawling library of hundreds or thousands of
tools into a lean, on-demand toolkit. Future work will refine retrieval at extreme
scale—via hierarchical indexes or adaptive strategies—and explore multi-tool
workflows and real-world agent deployments. RAG-MCP lays the “golden core”
for scalable, reliable LLM agents that wield vast external services with precision
and efficiency.
References
1. Anthropic: Introducing the model context protocol (2024), https://www.
anthropic.com/news/model-context-protocol
2. Blog, N.: What is retrieval-augmented generation aka rag (2025), https://blogs.
nvidia.com/blog/what-is-retrieval-augmented-generation/
3. Chen, Y.C., Hsu, P.C., Hsu, C.J., Shiu, D.s.: Enhancing function-calling capa-
bilities in llms: Strategies for prompt formats, data integration, and multilingual
translation. arXiv preprint arXiv:2412.01130 (2024)
4. Gao, Y., Xiong, Y., Gao, X., Jia, K., Pan, J., Bi, Y., Dai, Y., Sun, J., Wang, H.,
Wang, H.: Retrieval-augmented generation for large language models: A survey.
arXiv preprint arXiv:2312.10997 2(2023)

12 Gan and Sun
5. gkamradt: The needle in a haystack test (2024), https://github.com/gkamradt/
LLMTest_NeedleInAHaystack
6. Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V., Goyal, N., Küt-
tler, H., Lewis, M., Yih, W.t., Rocktäschel, T., Riedel, S., Kiela, D.: Retrieval-
augmented generation for knowledge-intensive nlp tasks. In: Larochelle, H.,
Ranzato, M., Hadsell, R., Balcan, M., Lin, H. (eds.) Advances in Neural
Information Processing Systems. vol. 33, pp. 9459–9474. Curran Associates,
Inc. (2020), https://proceedings.neurips.cc/paper_files/paper/2020/file/
6b493230205f780e1bc26945df7481e5-Paper.pdf
7. Liu, A., Feng, B., Xue, B., Wang, B., Wu, B., Lu, C., Zhao, C., Deng, C., Zhang,
C., Ruan, C., et al.: Deepseek-v3 technical report. arXiv preprint arXiv:2412.19437
(2024)
8. Luo, Z., Shi, X., Lin, X., Gao, J.: Evaluation report on mcp servers (2025), https:
//arxiv.org/abs/2504.11094
9. Nakano, R., Hilton, J., Balaji, S., Wu, J., Ouyang, L., Kim, C., Hesse, C., Jain,
S., Kosaraju, V., Saunders, W., Jiang, X., Cobbe, K., Eloundou, T., Krueger,
G., Button, K., Knight, M., Chess, B., Schulman, J.: Webgpt: Browser-assisted
question-answering with human feedback (2022), https://arxiv.org/abs/2112.
09332
10. OpenAI: Openai function calling, https://platform.openai.com/docs/guides/
function-calling
11. OpenAI: Chatgpt plugins (2023), https://openai.com/index/chatgpt-plugins
12. Patil, S.G., Zhang, T., Wang, X., Gonzalez, J.E.: Gorilla: Large language model
connected with massive apis. Advances in Neural Information Processing Systems
37, 126544–126565 (2024)
13. Schick, T., Dwivedi-Yu, J., Dessi, R., Raileanu, R., Lomeli, M., Hambro,
E., Zettlemoyer, L., Cancedda, N., Scialom, T.: Toolformer: Language mod-
els can teach themselves to use tools. In: Oh, A., Naumann, T., Glober-
son, A., Saenko, K., Hardt, M., Levine, S. (eds.) Advances in Neural In-
formation Processing Systems. vol. 36, pp. 68539–68551. Curran Associates,
Inc. (2023), https://proceedings.neurips.cc/paper_files/paper/2023/file/
d842425e4bf79ba039352da0f658a906-Paper-Conference.pdf
14. ShipAny: Mcp servers (2025), https://mcp.so/
15. Yao, S., Zhao, J., Yu, D., Du, N., Shafran, I., Narasimhan, K., Cao, Y.: Re-
act: Synergizing reasoning and acting in language models. International Confer-
ence on Learning Representations (ICLR) (2023), https://par.nsf.gov/biblio/
10451467
7 Acknowledgements
WegratefullyacknowledgeZhilingLuo,XiaorongShi,XuanruiLin,andJinyangGao
for their seminal report “Evaluation Report on MCP Servers.” Their publicly re-
leased code framework and the accompanying WebSearch dataset formed the
foundation on which our own work was developed.