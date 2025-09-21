# When Content is Goliath and Algorithm is David: The Style and Semantic Effects of Generative Search Engine

**Authors**: Lijia Ma, Juan Qin, Xingchen Xu, Yong Tan

**Published**: 2025-09-17 21:19:13

**PDF URL**: [http://arxiv.org/pdf/2509.14436v1](http://arxiv.org/pdf/2509.14436v1)

## Abstract
Generative search engines (GEs) leverage large language models (LLMs) to
deliver AI-generated summaries with website citations, establishing novel
traffic acquisition channels while fundamentally altering the search engine
optimization landscape. To investigate the distinctive characteristics of GEs,
we collect data through interactions with Google's generative and conventional
search platforms, compiling a dataset of approximately ten thousand websites
across both channels. Our empirical analysis reveals that GEs exhibit
preferences for citing content characterized by significantly higher
predictability for underlying LLMs and greater semantic similarity among
selected sources. Through controlled experiments utilizing retrieval augmented
generation (RAG) APIs, we demonstrate that these citation preferences emerge
from intrinsic LLM tendencies to favor content aligned with their generative
expression patterns. Motivated by applications of LLMs to optimize website
content, we conduct additional experimentation to explore how LLM-based content
polishing by website proprietors alters AI summaries, finding that such
polishing paradoxically enhances information diversity within AI summaries.
Finally, to assess the user-end impact of LLM-induced information increases, we
design a generative search engine and recruit Prolific participants to conduct
a randomized controlled experiment involving an information-seeking and writing
task. We find that higher-educated users exhibit minimal changes in their final
outputs' information diversity but demonstrate significantly reduced task
completion time when original sites undergo polishing. Conversely,
lower-educated users primarily benefit through enhanced information density in
their task outputs while maintaining similar completion times across
experimental groups.

## Full Text


<!-- PDF content starts -->

When Content is Goliath and Algorithm is David: The
Style and Semantic Effects of Generative Search Engine
Lijia Ma*
Belk College of Business, University of North Carolina at Charlotte
Juan Qin*
Faculty of Business for Science and Technology, School of Management, University of Science and Technology of China
Xingchen (Cedric) Xu*, Yong Tan†
Michael G. Foster School of Business, University of Washington
Generative search engines (GEs) leverage large language models (LLMs) to deliver AI-generated summaries
with website citations, establishing novel traffic acquisition channels while fundamentally altering the search
engine optimization landscape. To investigate the distinctive characteristics of GEs, we collect data through
interactions with Google’s generative and conventional search platforms, compiling a dataset of approxi-
mately ten thousand websites across both channels. Our empirical analysis reveals that GEs exhibit prefer-
ences for citing content characterized by significantly higher predictability for underlying LLMs and greater
semantic similarity among selected sources. Through controlled experiments utilizing retrieval augmented
generation (RAG) APIs, we demonstrate that these citation preferences emerge from intrinsic LLM tenden-
cies to favor content aligned with their generative expression patterns. Motivated by applications of LLMs
to optimize website content, we conduct additional experimentation to explore how LLM-based content
polishing by website proprietors alters AI summaries, finding that such polishing paradoxically enhances
information diversity within AI summaries. Finally, to assess the user-end impact of LLM-induced informa-
tion increases, we design a generative search engine and recruit Prolific participants to conduct a randomized
controlled experiment involving an information-seeking and writing task. We find that higher-educated users
exhibit minimal changes in their final outputs’ information diversity but demonstrate significantly reduced
task completion time when original sites undergo polishing. Conversely, lower-educated users primarily ben-
efit through enhanced information density in their task outputs while maintaining similar completion times
across experimental groups. Our findings provide theoretical and practical implications for stakeholders in
the evolving search ecosystem.
Key words: Generative Search Engine, Generative AI, AI Summary, RAG, Search Engine Optimization
1arXiv:2509.14436v1  [cs.IR]  17 Sep 2025

Author:Generative Search Engine2Article submitted toInformation Systems Research; manuscript no. ISRE-0000-0000.00
“The medium is the message.” - McLuhan (1964)
1. Introduction
In the contemporary digital ecosystem, the exponential growth of information volume and diversity
has necessitated sophisticated intermediary mechanisms to facilitate connections between infor-
mation providers and consumers. Search engines have emerged as critical infrastructure in this
information mediation process, becoming integral to daily digital interactions. Empirical evidence
indicates that 87% of mobile device users utilize search engines at least once daily, while Google
alone processes approximately 63,000 queries per second, corresponding to roughly 2 trillion annual
searches globally1. These platforms algorithmically determine content visibility through complex
scoring mechanisms that evaluate query-content relevance and numerous additional ranking factors.
The complexity and opacity of these algorithmic systems have catalyzed the emergence of Search
Engine Optimization (SEO) as a specialized industry, which encompasses systematic strategies for
deciphering algorithmic ranking criteria and optimizing digital content to enhance organic search
visibility (Liu and Toubia 2018). This sector has demonstrated substantial economic significance,
achieving a market valuation of US$89.1 billion in 20242.
However, the search engine paradigm is undergoing a fundamental transformation driven by
the emergence ofgenerative search engines(GE) that leverage Large Language Models (LLMs)
and Retrieval Augmented Generation (RAG) architectures. This technological shift was catalyzed
by OpenAI’s introduction of ChatGPT in November 2022, which pioneered conversational query
resolution mechanisms. Microsoft subsequently integrated GPT-4 technology into its search infras-
tructure, launching the New Bing platform in February 20233. Later, Google also initiated exper-
imental deployment of generative search capabilities in March 2023, which evolved into the “AI
∗These authors contributed equally to the manuscript and are listed alphabetically.
†Corresponding author.
1Seehttps://serpwatch.io/blog/search-engine-statistics/.
2Seehttps://www.researchandmarkets.com/reports/5140303/search-engine-optimization-seo-global.
3Seehttps://techcrunch.com/2023/03/14/microsofts-new-bing-was-using-gpt-4-all-along/.

Author:Generative Search Engine
Article submitted toInformation Systems Research; manuscript no. ISRE-0000-0000.003
Overviews” feature, an AI-powered summarization system integrated into search results, launched in
May 20244. The demonstrated efficacy of these systems in enhancing search efficiency and user sat-
isfaction has facilitated rapid adoption and geographic expansion, with AI Overviews subsequently
deployed across more than 200 countries and engaging 2 billion monthly users as of July 20255. This
paradigmatic evolution represents a fundamental departure from conventional search engines, which
primarily function as indexing systems that enumerate relevant websites; instead, generative search
engines employ sophisticated query interpretation mechanisms, dynamically synthesize information
from multiple sources, and generate coherent responses that incorporate citations for verification
and attribution.
Extant research on conventional search engines has demonstrated the substantial influence of
search engine visibility on individual decision-making processes (Gong et al. 2018, Ghose et al. 2014)
and the distribution of welfare among market stakeholders (Berman and Katona 2013). As the
search paradigm increasingly incorporates generative search engines, website visibility has become
contingent upon citation frequency within LLM-generated content summaries and overviews. This
fundamental shift in visibility determinants necessitates strategic adaptation beyond traditional
SEO methodologies; websites pursuing enhanced exposure must now develop specialized approaches
to optimize their citation probability within generative engine outputs. Consequently, this techno-
logicalevolutionhasledtotheemergenceofanascentindustrytermedGenerativeEngineOptimiza-
tion (GEO), which focuses specifically on maximizing website citation rates in AI-generated search
responses.However,optimizationacrossbothconventionalandgenerativesearchparadigmspresents
significant challenges due to the opaque nature of underlying algorithms to the public, deliberately
designed to maintain system integrity and reduce susceptibility to manipulation (Erdmann et al.
2022, Danaher et al. 2006). Scholarly investigations in this domain have employed Natural Language
Processing (NLP) methodologies to examine the relationship between various content factors and
4Seehttps://blog.google/products/search/generative-ai-google-search-may-2024/
5Seehttps://blog.google/inside-google/message-ceo/alphabet-earnings-q2-2025/

Author:Generative Search Engine4Article submitted toInformation Systems Research; manuscript no. ISRE-0000-0000.00
website rankings in search engine results (Liu and Toubia 2018), subsequently informing content
optimization strategies (Reisenbichler et al. 2022). However, GEO presents even greater difficulties
due to the vast output space and randomness of their underlying large language models, whose
behavioral patterns remain only partially predictable even to the algorithm developers and system
architects who designed them (Berti et al. 2025). Therefore, we ask our first research question:
RQ1: How do generative search engines select websites to cite in response to a query?
To explore this research question, we construct a large-scale dataset encompassing over ten thou-
sand websites through interaction with Google’s search infrastructure, capturing AI Overview con-
tent, cited sources within AI Overviews, and top-ranked websites in conventional Google Search
results. Our analysis reveals that AI Overview predominantly cites content exhibiting significantly
lower perplexity compared to content prioritized by conventional search algorithms. This phe-
nomenon stems from the inherent characteristics of the Transformer architecture employed by most
contemporary LLMs, including Google’s underlying Gemini model, which operates through auto-
regressive token generation mechanisms (Vaswani et al. 2017). Specifically, these models generate
natural language responses sequentially, selecting tokens with high conditional probability at each
generation step. Consequently, when LLMs synthesize AI Overview content (hereafter referred to as
“AI summary”) and incorporate textual elements from source materials, the linguistic predictability
of these original sources, as evaluated by the LLM’s internal probability distributions, directly influ-
ences the overall predictability of the generated AI summary, which the model seeks to optimize
during the generation process. Notably, perplexity measurements demonstrate minimal influence
on ranking within conventional Google search results, highlighting a fundamental divergence in
algorithmic prioritization mechanisms. Furthermore, our analysis demonstrates that content cited
by LLMs exhibits greater semantic homogeneity compared to sources ranked highly by conven-
tional search engines. This semantic convergence occurs because AI Overviews are architecturally
designed to synthesize coherent, unified responses rather than merely aggregating disparate informa-
tion sources. Consequently, despite the potential for generative engines to incorporate diverse, niche

Author:Generative Search Engine
Article submitted toInformation Systems Research; manuscript no. ISRE-0000-0000.005
websites that do not achieve prominent rankings in conventional search results, the cited sources
consistently demonstrate substantial semantic similarity due to the coherence requirements inherent
in the generative summarization process.
Nevertheless, it remains uncertain whether these observed citation criteria represent explicit engi-
neering decisions specifically tailored for Google’s AI Overview functionality or constitute inherent
characteristics embedded within the underlying language model architecture. Two competing expla-
nationsemergefromthisuncertainty.First,Google’sAIOverviewconfrontsthesubstantialchallenge
ofprocessingvastquantitiesofdynamicwebcontent,potentiallynecessitatingadditionalalgorithmic
layers and engineered preferences that might lead to specific content characteristics. Alternatively,
following our preceding theoretical framework, the observed stylistic preferences (reflected in per-
plexity patterns) may derive from the fundamental operational mechanisms of LLMs (specifically,
their next-token prediction paradigm), while the semantic tendencies (manifested as content sim-
ilarity) may emerge from the inherent constraints of the output format requirement for coherent
natural language summarization. Given these competing perspectives, we pose our second research
question:
RQ2: Do generative search engines’ criteria originate through manual curation or naturally emerge
from underlying language models?
To address this research inquiry, we investigate the information retrieval and content synthesis
processesexecutedbyGemini,thefoundationalmodelarchitectureunderlyingGoogle’sAIOverview
functionality. If the observed preferences for citation source selection are attributable to intrin-
sic model characteristics rather than explicit engineering interventions, analogous patterns should
manifest in independent implementations utilizing Gemini for Retrieval Augmented Generation
(RAG) tasks. To test this hypothesis, we construct an additional dataset using Gemini’s document
understanding API by providing both search queries and curated compilations of website content
as inputs. Replicating our analyses within this controlled environment, we discover that the RAG
system exhibits qualitatively consistent stylistic and semantic properties, thereby corroborating the

Author:Generative Search Engine6Article submitted toInformation Systems Research; manuscript no. ISRE-0000-0000.00
generalizability of our findings and supporting the hypothesis that these citation preferences are
intrinsic to the underlying language model architecture. Throughout this process, we also identify
and empirically control for positional bias inherent in RAG systems, thereby generating additional
insights for GEO practice.
Building on our empirical findings, we extend our investigation to examine another critical phe-
nomenon in the digital content landscape. The proliferation of LLMs has introduced widespread
adoption of automated content creation and refinement processes, with the AI-generated content
market reaching a valuation of $13.9 billion as of 20246. This technological shift has attracted
considerable scholarly attention, as demonstrated by Ye et al. (2025)’s LLM-based headline gen-
eration for traffic enhancement and Reisenbichler et al. (2022)’s pioneering GPT-2 application for
SEO optimization even before ChatGPT’s introduction. In the age of GEO, our preceding analysis
demonstrates that generative engines exhibit systematic preferences for linguistically predictable
content, suggesting a natural optimization strategy: leveraging LLMs to refine existing content,
thereby enhancing its predictability from the model’s perspective. However, this approach raises a
fundamental question regarding content diversity. Universal adoption of LLM-based content pol-
ishing may precipitate content similarity, a phenomenon recently documented in the literature as
LLMs’ homogenizing effect (Moon et al. 2024). Conversely, such AI-mediated refinement could
potentially expand the consideration set for generative engines by transforming previously unpre-
dictable content into more accessible forms, thereby enabling AI summaries to incorporate greater
content diversity. To investigate these competing theoretical predictions, we pose our third research
question:
RQ3: How does content diversity change when input content for generative search engines is
refined through LLM-based polishing processes?
To conduct this analysis, we implement a controlled RAG experiment following the identical
methodological framework established for RQ2. However, prior to inputting content into Gemini’s
6Seehttps://www.grandviewresearch.com/industry-analysis/generative-ai-content-creation-market-report

Author:Generative Search Engine
Article submitted toInformation Systems Research; manuscript no. ISRE-0000-0000.007
document understanding API, we deploy Gemini’s text generation API to refine all source materials
through automated polishing procedures. Our findings reveal an unexpected outcome: automated
content refinement actually enhances information diversity of the RAG outputs, indicating that
the diversity-expansion mechanism predominates over the homogenization effect. This mechanism
receives further empirical substantiation as we observe an expanded citation scope encompassing
a greater number of source websites. Additionally, our analysis demonstrates that explicitly incor-
porating citation optimization objectives within the AI polishing prompts significantly amplifies
these diversity-enhancing effects, thereby revealing an intrinsic AI-to-AI compatibility phenomenon
wherein language models inherently understand and accommodate the preferences of similar archi-
tectures, even in the absence of detailed optimization guidance.
Although our experimental results demonstrate that LLM-based content polishing significantly
alters both the information contained in generated summaries and the websites cited therein, the
effects of these content modifications on user behavior remain theoretically ambiguous. Two compet-
ing mechanisms emerge from this uncertainty. First, leveraging LLMs’ natural language generation
capabilities, AI Overviews deliver coherent paragraph-form responses that can enhance users’ search
efficiency (Xu et al. 2023), suggesting that increased information diversity may also be readily
processed and utilized by users. Conversely, users remain constrained by limited attention and cog-
nitive processing capacity, potentially limiting their ability to consume expanded information sets, a
phenomenon documented as information overload (O’Reilly 1980). Consequently, despite enhanced
information diversity resulting from content optimization, users may be unable to effectively process
and leverage this additional information. Furthermore, as an emergent technological artifact, gen-
erative search systems may exhibit differential usage patterns and outcomes across different users,
reflecting what scholars term the “AI divide” phenomenon (Ma et al. 2024, McElheran et al. 2024).
Given these theoretical tensions, we pose our fourth research question:
RQ4: How do users’ search experiences change when input content undergoes LLM-based polishing
processes? Do such changes vary systematically across different users?

Author:Generative Search Engine8Article submitted toInformation Systems Research; manuscript no. ISRE-0000-0000.00
To investigate these directions, we develop a platform that replicates Google AI Overview’s func-
tionality and recruit participants from Prolific to complete writing tasks using our system. We
employ a randomized controlled trial design wherein half of the participants (control group) receive
AI summaries generated from unmodified website content sourced directly from the internet, while
the remaining participants (treatment group) receive summaries based on LLM-polished website
content. Our experimental results yield multifaceted insights. First, corroborating our findings from
RQ3, we observe significant increases in both information entropy and the number of cited websites
within AI responses provided to treatment group participants. Second, this enhanced information
diversity in AI Overview outputs translates directly to participants’ performance outcomes: treat-
ment group participants produce longer and more substantively informative task solutions. Third,
and most unexpectedly, we identify differential treatment effects across educational backgrounds.
Participants with undergraduate education or below derive primary benefits through improved
solution quality, while those with graduate-level education experience efficiency gains rather than
quality improvements. This differential response pattern emerges because graduate-educated par-
ticipants exhibit adaptive prompting behavior, issuing additional queries when information density
appears insufficient. When exposed to optimized websites containing enhanced information density,
these participants reduce their query frequency, thereby achieving time savings while maintain-
ing solutions’ information density. Conversely, participants with undergraduate education or below
demonstrate consistent low-prompting behavior across both experimental conditions, but directly
benefit from the increased information content available in the treatment group AI summaries.
The remainder of this paper is organized as follows. Section 2 synthesizes relevant literature and
situates our research within the existing scholarly framework. Section 3 delineates our approach for
observational data collection and processing. Section 4 presents our variable construction procedures
and empirical analyses, systematically addressing research questions 1-3. Section 5 describes our
controlled experimental design involving human participants and corresponding analyses to exam-
ine research question 4. Section 6 evaluates the robustness of our findings through comprehensive

Author:Generative Search Engine
Article submitted toInformation Systems Research; manuscript no. ISRE-0000-0000.009
validation procedures, including a replication study conducted on Microsoft’s New Bing platform.
Finally, Section 7 concludes by discussing our theoretical and practical contributions, examining
broader implications for the search engine ecosystem, and proposing directions for future scholarly
investigation.
2. Related Literature
Ourresearchbuildsuponandcontributestofourstreamsofliterature:(i)searchengineoptimization
and marketing; (ii) economics of information retrieval systems; (iii) AI-generated summaries and
human-AI interaction.
2.1. Search Engine Optimization and Marketing
Contextually, our study directly relates to the search engine optimization and marketing litera-
ture. Given that search engine appearances and rankings profoundly influence consumer behavior
and market outcomes (Ghose et al. 2014, Goldfarb and Tucker 2011), extensive efforts have been
undertaken to enhance websites’ visibility across both organic and sponsored search channels.
Regarding organic search, ranking algorithms are typically maintained as opaque by search engine
platforms to minimize direct manipulation by website owners. Consequently, the search engine
optimization(SEO)literatureseekstoidentifyfactorsthatinfluenceorganictraffic,therebyenabling
content optimization to increase organic visibility. This research spans technical elements and site
architecture (Danaher et al. 2006) to content semantics (Liu and Toubia 2018). Recently, researchers
have explored the application of generative AI to optimize content style and semantics for enhanced
exposure in organic listings (Reisenbichler et al. 2022). Regarding sponsored search, search engine
platforms provide paid traffic opportunities as an additional channel for website owners. Search
engine marketing (SEM) research develops strategies for bidding on such paid traffic from multiple
perspectives, including optimal bidding across keywords and positions (Abhishek and Hosanagar
2013), budget allocation under constraints (Shin 2015), and keyword matching strategies (Du et al.
2017).
However, as generative search engines emerge as a novel channel for website exposure, under-
standing of their distinctive traffic allocation patterns remains limited, particularly regarding how

Author:Generative Search Engine10Article submitted toInformation Systems Research; manuscript no. ISRE-0000-0000.00
citation patterns relate to foundation models. We aim to extend SEO literature to generative engine
optimization (GEO) by investigating the factors governing generative engines’ citation behavior and
the underlying mechanisms.
2.2. Economics of Information Retrieval Systems
Ourstudyalsorelatestotheeconomicsofinformationretrievalsystems,whichexaminestheimpacts
and strategic interactions among participants within various information retrieval platforms, includ-
ing search engines and recommender systems (Zhou et al. 2024, Berman and Katona 2013).
Informationretrievaltechnologiesemergeduetoconsumers’limitedcapacitytoaccessandprocess
comprehensiveinformationsets,particularlywhendealingwithlarge-scaleinformationenvironments
(Ursu et al. 2025). Search engines curate a targeted subset of relevant websites for users, eliminating
the need for indiscriminate browsing across countless web resources (Gong et al. 2018). Since search
engines directly influence user exposure to online content and consequently affect website revenue
generation, these platforms substantially redistribute welfare among advertisers, consumers, and
themselves (Berman and Katona 2013).
Previous research has documented strategic platform behavior. For instance, Hagiu and Jullien
(2011) explain why intermediaries may redirect user searches for profit maximization, even at the
expense of consumer welfare, while De Cornière and Taylor (2014) analyze vertical integration in
search markets, demonstrating that integrated platforms possess incentives to bias rankings toward
their proprietary services. Regarding website owners, prior research demonstrates their strategic
responses to search engine mechanisms aimed at enhancing visibility within budget constraints.
Berman and Katona (2013) show that when SEO investments achieve sufficient effectiveness, adver-
tisers can improve organic visibility while reducing dependence on sponsored links. Similarly, Baye
et al. (2016) analyze how retailers’ investments in site quality and brand awareness increase organic
traffic both directly through consumer preferences for superior and well-established sites, and indi-
rectly through search engines’ favorable positioning of such sites.
As generative search engines emerge as novel intermediaries for information transmission, their
potential economic implications represent a compelling and significant research direction. We aim

Author:Generative Search Engine
Article submitted toInformation Systems Research; manuscript no. ISRE-0000-0000.0011
to establish foundational insights for future economic modeling by analyzing generative engines’
behavioral patterns and examining potential output variations when websites strategically optimize
content.
2.3. AI-Generated Summaries and Human-AI Interaction
Generative search engines provide AI-generated summaries in response to search requests. Previous
research on AI-generated summaries typically examines user responses to such summaries across
various online platforms. For instance, Koo et al. (2025) find that AI-generated review summaries
on e-commerce platforms guide subsequent ratings and comments toward more positive evaluations,
thereby reinforcing the advantages of already popular products. Alavi and Nozari (2023) document
thatAIproductreviewsummariesreduceengagementwithindividualreviewsanddecreasethematic
diversity, suggesting an anchoring effect toward uniform content. Conversely, Yuan et al. (2024)
find that AI-generated summaries serve as differentiating references on hotel review platforms,
enhancing diversity and helpfulness of subsequent user content while ultimately boosting sales.
Extending this inquiry to online video platforms (Ji et al. 2025), Kim et al. (2024) find that AI
video summaries significantly increase user engagement, particularly through increased comments
and unique commenters for longer videos.
In the generative search engine context, users can freely prompt engines to generate varied AI
summaries, making the human-AI interaction literature equally relevant. Previous research employs
laboratory experiments to examine how users interact with generative AI to complete information-
seeking tasks (Xu et al. 2023). Further, user interaction patterns and performance may vary signif-
icantly when engaging with generative AI (Gao et al. 2025, Yeverechyahu et al. 2024, Qiao et al.
2023, Wang et al. 2023). For example, Ma et al. (2024) find that lower-educated users derive greater
utility from generative AI but learn about this utility more slowly. Such disparities in human-AI
interaction constitute an important research area known as the AI divide. For instance, Yan et al.
(2024) demonstrate that generative AI alters the balance between ideation and execution, amplify-
ing inequality by favoring certain creators over others. Consistent with this finding, Hou et al. (2024)

Author:Generative Search Engine12Article submitted toInformation Systems Research; manuscript no. ISRE-0000-0000.00
find that generative AI enhances divergent thinking for less experienced creators while reducing
efficiency for experts without improving design quality.
AsasignificantAI-drivenchannelthroughwhichusersobtaininformation,theinformationalvalue
provided by generative search engines for different user groups remains understudied, particularly
when information diversity is altered by supply-side optimization. We aim to address this gap and
extend understanding of human-AI interaction within such systems.
3. Raw Data Sources and Collection
To investigate generative engines’ website citation patterns and compare with traditional search
engines, we collect data through direct interaction with both search paradigms. This section delin-
eates our approach to query specification, engine interaction protocols, and the collection of source
materials highlighted by each system. The resulting dataset serves as the foundation for subsequent
analyses addressing research questions 1, 2, and 3 in the following section.
3.1. Query Selection and Sampling
To conduct a comprehensive and representative analysis of generative search engine behavior, we
requireadiversesetofinputqueriesthatareamenabletonaturallanguageansweringandthuslikely
to elicit generative responses. We select the Human ChatGPT Comparison Corpus (HC3) (Guo
et al. 2023) as our query source. The HC3 dataset, originally developed to evaluate AI-generated
versus human-generated text, encompasses diverse user queries spanning multiple domains and
complexity levels, rendering it an ideal foundation for examining generative search engine responses
across varied information needs. Given the computational expense of processing each query through
both search engines and retrieving associated websites, we randomly sample 5,000 unique queries
from HC3. This query sample size naturally involves tens of thousands of websites (discussed in the
subsequent subsection), providing sufficient scale for large-scale analyses.
3.2. Responses from AI Overview and Conventional Search Engine
GivenGoogle’sdominanceinthesearchenginemarketandthewidespreadrolloutofitsAIOverview
function to most countries in 2025, we select Google as our primary interaction target. As illustrated

Author:Generative Search Engine
Article submitted toInformation Systems Research; manuscript no. ISRE-0000-0000.0013
Figure 1 Example of Google Search Results Page with AI Overview and Organic Results
in Figure 1, Google’s search interface presents organic website listings in response to user queries.
When available, the platform also displays AI Overview results featuring in-text citations. The
right-hand panel lists referenced websites, some of which correspond to these in-text citations.
We leverage SerpAPI, which provides reliable scraping services for Google search results including
AI Overview functionality7, to obtain Google’s responses in both conventional and generative forms
7Seehttps://serpapi.com

Author:Generative Search Engine14Article submitted toInformation Systems Research; manuscript no. ISRE-0000-0000.00
during June 2025. SerpAPI offers two relevant endpoints: a general Google search endpoint that
encompassesbothconventionalsearchresultsandAIOverview,andadedicatedGoogleAIOverview
endpoint specifically for generative responses. We utilize the general endpoint to input each sampled
query and retrieve: (i) the AI Overview content; (ii) URL links cited by the AI Overview; and (iii)
the first page of conventional search results indicated by SerpAPI, including their ranking positions,
titles, snippets, and URLs.
In certain cases, the Google AI Overview is not returned directly by the general endpoint; instead,
a parameter in the endpoint output indicates that an additional request is required8. For such
queries, we subsequently invoke the AI Overview specific endpoint to retrieve the AI Overview
content and its referenced URLs. This two-step procedure yields 4,060 queries containing both
conventional and generative search results, which form the basis for our subsequent analyses. Among
these, 305 queries provide AI summaries with references listed only at the end, whereas 3,755 include
sentence-level references.
Following the retrieval of URLs from SerpAPI, we subsequently collect the complete content from
all websites appearing in either AI Overview citations or conventional search results. We develop
an automated web agent to dynamically extract textual information from these websites. Where
websites appear in both search result types, we label them and de-duplicate the website content
files, ultimately yielding 98,477 unique websites.
More details about the whole data collection process can be found in Appendix B.
4. Data Analyses
This section systematically examines research questions 1, 2, 3 and 4 through empirical analyses.
Given that each research question necessitates distinct datasets and methodological frameworks,
we adopt a uniform organizational structure for each subsection: the first part delineates the spe-
cific experimental design and variable construction procedures tailored to the respective research
question, while the second part presents the corresponding empirical findings.
8Seehttps://serpapi.com/blog/scrape-google-ai-overviews/

Author:Generative Search Engine
Article submitted toInformation Systems Research; manuscript no. ISRE-0000-0000.0015
4.1. Determinants of Website Citation in Generative Search Engines (RQ1)
4.1.1. Chunk Selection and Variables
To investigate the criteria employed by generative search engines in selecting website content
for citation, we construct a comprehensive dataset through systematic interaction with Google
Search, as detailed in Section 3. For each query, we collect both AI Overview responses and the
complete textual content of all websites appearing either in AI Overview citations or among the top
conventional search results.
As previously noted, AI Overview comprises multiple sentences, each of which may cite zero to
multiple websites as references, while individual websites in the AI Overview’s reference list may be
specifically cited by multiple sentences or alternatively listed without specific sentence attribution to
provide general support for the AI summary’s core content. Consequently, we can conduct empirical
analyses at two distinct levels. First, we can evaluate whether a particular website merits citation for
supporting the AI summary overall. Second, we can examine whether a specific website should be
cited to support an individual sentence within the AI summary. However, given that some websites
contain extensive information, most of which is not considered by the search engine when answering
the user queries requested, we narrow the content scope accordingly.
For the first type of analysis, we consider three categories of websites and extract one represen-
tative chunk from each: websites cited by specific sentences in AI Overview, websites mentioned in
AI Overview but not cited by specific sentences, and websites appearing only in conventional search
results. For each category, we employ different matching targets as specified in Algorithm 1, while
consistently using a rolling window approach to identify the chunk most relevant to the respective
matching target. Specifically, we implement a sliding window approach with a window size of 128
characters and a step size of 16 characters to segment each website into overlapping chunks. We
utilize the Sentence-BERT model (Reimers and Gurevych 2019) to generate embeddings for both
matching targets and website chunks, subsequently computing cosine similarity to identify the most
semantically representative chunk for each website.

Author:Generative Search Engine16Article submitted toInformation Systems Research; manuscript no. ISRE-0000-0000.00
Algorithm 1Chunk Selection: One Chunk per Website
Require:AI Overview content, Search snippets, Website content
1:foreach queryqdo
2:foreach websitekin queryqdo
3:ifwebsitekis cited by specific sentences in AI Overviewthen
4:S k←all sentences citing websitek
5:C k←all chunks of websitek
6:Selectc∗
k= arg max c∈Ck;s∈Sksimilarity(c,s)
7:else ifwebsitekis mentioned in AI Overview but not cited by specific sentencesthen
8:A q←entire AI Overview text for queryq
9:C k←all chunks of websitek
10:Selectc∗
k= arg max c∈Cksimilarity(c,A q)
11:else
12:snippet k←search result snippet for websitekhighlighted in conventional search
13:Selectc∗
k←chunk containingsnippet k
14:end if
15:Store representative chunkc∗
kfor websitek
16:end for
17:end for
18:returnOne representative chunk per website
For the second type of analysis, we identify potential reference chunks at the sentence level
within each query’s AI summary. Specifically, for each sentence containing references, we determine
one “candidate” chunk from each website that could potentially serve as supporting evidence for
that sentence. This process involves iteratively examining all sentences with references and all the
website chunks to identify the most relevant chunk for every (sentence, website) pair, as detailed in
Algorithm 2.

Author:Generative Search Engine
Article submitted toInformation Systems Research; manuscript no. ISRE-0000-0000.0017
Algorithm 2Chunk Selection: One Chunk per (AI Overview’s sentence, Website) pair
Require:AI Overview responses, Website content
1:foreach queryqdo
2:Retrieve all sentencesS qthat cite at least one website
3:Retrieve all related websitesK qfrom both AI Overview and search results
4:foreach sentences∈S qdo
5:foreach websitek∈K qdo
6:Divide websitekinto chunksC k
7:Identify the most relevant chunk:arg max c∈Cksimilarity(c,s)
8:Record mapping of queryq, sentences, websitek, and segmentc
9:end for
10:end for
11:end for
Basedontheaforementionedprocedures,weconstructtwodatasets.Thefirstdatasetcompilesone
chunk per website with binary indicators (ChatCite) of whether the AI Overview cites the website.
The second dataset compiles one chunk for each (sentence, website) pair with binary indicators
(SentenceCite) of whether the website is cited by the specific sentence. As outlined in Research
Question 1, we construct variables based on these chunks’ content to facilitate subsequent analyses.
Specifically, we calculate perplexity for each chunk in both datasets and compute semantic similarity
for every pair of chunks in the first dataset.
Perplexity measures a model’s uncertainty when predicting text, with lower values indicating
higherpredictabilityfromthemodel’s perspective(Gonenetal.2022). WeemployGoogle’sGemma-
2B model to obtain the conditional probability of each token based on the preceding token sequence,
then average across all tokens within each chunk:PPL= exp
−1
NPN
i=1logP(w i|w1,...,w i−1)
.
For semantic similarity, we embed both chunks of a pair using the Universal Sentence Encoder and
calculate pairwise cosine similarities between the resulting embedding pairs (Cer et al. 2018).

Author:Generative Search Engine18Article submitted toInformation Systems Research; manuscript no. ISRE-0000-0000.00
Table 1 Summary Statistics for RQ1 Dataset
Statistic N Mean St. Dev. Min Median Max
Website-level Data
ChatCite98,477 0.47 0.50 0 0 1
PPL98,477 16.42 9.52 1.09 13.86 82
Similarity1,261,914 0.54 0.16 -0.10 0.55 0.99
(Sentence-Website)-Level Data
SentenceCite253,502 0.04 0.21 0 0 1
PPL253,502 16.95 10.59 1.21 14.25 197
Note: “Similarity” has more observations because we calculate the value for each pair from the same query.
4.1.2. Empirical Results for RQ1
Utilizing the aforementioned dataset derived from direct interaction with Google’s search infras-
tructure, identification of relevant chunks, and construction of corresponding metrics, we now con-
duct a formal comparative analysis between chunks cited by Google’s AI Overview and those
excluded from citation.
Specifically, in the website-level dataset, for each queryq, we extract one representative chunkc
from each website. Each chunk possesses a perplexity measurePPL q,cand may or may not be cited
by the AI Overview, as indicated by the binary variableChatCite q,c. In the sentence-website level
dataset, we consider that each AI Overview comprisesS qsentences (indexed bys= 1,...,S q), each
of which may reference supporting sources. For every sentences, we retrieve one candidate chunk
cfrom each website, characterized by perplexityPPL q,s,c. We also record whether sentencescites
the corresponding website, denoted bySentenceCite q,s,c.
The perplexity measure assumes particular theoretical significance within the generative search
engine paradigm, as it quantifies the predictability of textual content from a language model’s
perspective (i.e., low perplexity represents high predictability), a metric extensively employed in
evaluating both textual characteristics and generative model performance. For instance, researchers
conducting domain-specific fine-tuning of general-purpose language models (e.g., programming,
biomedical, and legal domains) monitor perplexity during training procedures to assess model adap-
tation to specialized knowledge domains (Bolton et al. 2024). Furthermore, perplexity serves as a
diagnostic indicator for identifying AI-generated text (Xu and Sheng 2024), reflecting one funda-
mental mechanism of auto-regressive models: these architectures sequentially select tokens based on

Author:Generative Search Engine
Article submitted toInformation Systems Research; manuscript no. ISRE-0000-0000.0019
predictability scores, with greedy decoding strategies favoring the most predictable tokens at each
generation step, thereby systematically reducing overall text perplexity (Vaswani et al. 2017).
Formally, we estimate regression models in which the binary citation outcomesChatCite q,cand
SentenceCite q,s,cserve as the dependent variables, and the chunk’s perplexityPPL q,c(at the
sentence level,PPL q,s,c) is the primary explanatory variable. Each specification includes query-
level fixed effectsQ qto absorb unobserved, query-specific heterogeneity; consequently, the estimated
association between perplexity and the probability of being cited is identified from within-query
variation.
ChatCite q,c=β 0+β 1∗PPL q,c+Q q+ϵq,c (1)
SentenceCite q,s,c=β 0+β 1∗PPL q,s,c+Q q+ϵq,s,c (2)
We implement two distinct econometric specifications: Ordinary Least Squares (OLS) regression
(constituting a linear probability model) and logistic regression. As demonstrated in Table 2, our
empirical analysis reveals that chunks exhibiting lower perplexity demonstrate significantly higher
citation probabilities within AI Overview responses. Specifically, a one standard deviation decrease
in perplexity (9.52) corresponds to an increase in citation probability from an average of 47% (shown
in Table 1) to 56%. This finding suggests that language models systematically exhibit preferences for
content that aligns with their intrinsic generation patterns and stylistic characteristics. Analogous
results emerge when examining sentence-level citations. Additionally, this perplexity effect is absent
in conventional search ranking lists, as demonstrated in the Appendix H. Our results further cor-
roborate the theoretical mechanism previously articulated: when LLMs incorporate external source
material into their outputs, the generated content inherits the perplexity characteristics of those
sources. Consequently, when external sources exhibit high predictability from the LLM’s perspec-
tive, their integration facilitates the model’s operational objective of generating predictable content
through sequential token-by-token processing.

Author:Generative Search Engine20Article submitted toInformation Systems Research; manuscript no. ISRE-0000-0000.00
Table 2 AI Overview’s Perplexity Effect
ChatCite SentenceCite
LPM Logit LPM Logit
(1) (2) (3) (4)
PPL-0.0098*** -0.0480*** -0.0015*** -0.0581***
(0.0002) (0.0009) (0.0000) (0.0015)
Query FE Yes Yes Yes Yes
Observations 98,477 98,477 253,501 253,501
R2orChi20.13 3300.85 0.06 1820.31
Note: Cluster-robust standard errors are reported at the query level. ***p <0.01, **p <0.05, *p <0.1.
Furthermore, beyond examining the stylistic preferences of generative search engines, we inves-
tigate the semantic properties of their outputs, given their direct influence on user information
exposure and subsequent behavioral responses. Extensive prior research has demonstrated that
information retrieval systems critically influence opinion formation and belief construction processes
through their information diversity characteristics (Epstein and Robertson 2015, Pariser 2011).
Given that generative search engines constitute an emergent class of information retrieval systems,
they similarly warrant scrutiny around the diversity metrics.
Specifically, using the website-level dataset, we compare semantic similarity among AI-cited
chunks versus those highlighted by conventional search engines9. For each pair of chunks (denoted
bycandc′), we calculate semantic similarity (Similarity q,c,c′) using their respective embeddings.
We focus on comparing chunk pairs where both chunks are cited by AI (BothCite q,c,c′= 1) versus
those that are not both cited by AI (BothCite q,c,c′= 0). We consider two specifications for the con-
trol group (BothCite q,c,c′= 0): one includes only pairs with both chunks non-cited, while the other
additionally encompasses pairs with one cited and one non-cited chunk. Using these two distinct
samples, we regress the similarity measure on the citation indicator, incorporating query-specific
fixed effectsQ q.
Similarity q,c,c′=β 0+β 1∗BothCite q,c,c′+Q q+ϵq,c,c′ (3)
9We utilize the website-level rather than the sentence-website level dataset here, as our focus lies on the overall
information diversity of AI Overview.

Author:Generative Search Engine
Article submitted toInformation Systems Research; manuscript no. ISRE-0000-0000.0021
Regression results for both samples are presented in Table 3. Our empirical analysis reveals that
websites cited by generative search engines exhibit significantly higher semantic similarity on aver-
age compared to conventionally ranked results (β 1= 0.0365orβ 1= 0.0492,p <0.01). This finding
is particularly noteworthy given that the majority of cited websites do not appear within the top
organic search and represent relatively niche sources that conventional search algorithms would not
prioritize for query relevance. Nevertheless, these cited sources demonstrate substantially greater
semantic similarity (low information diversity). Consequently, despite AI summaries typically pre-
senting information through multiple sub-points and incorporating seemingly diverse niche sources,
users encounter a narrowed range of semantic perspectives compared to conventional search results.
Table 3 AI Overview’s Semantic Similarity Effect
Similarity
(1) (2)
BothCite0.0365*** 0.0492***
(0.0020) (0.0015)
Cross-Category No Yes
Query FE Yes Yes
Observations 674,086 1,261,914
R20.31 0.31
Note: Cluster-robust standard errors are reported at the query level. ***p <0.01, **p <0.05, *p <0.1;
“Cross-Category” indicates whetherBothCite= 0includes pairs with one cited chunk and one non-cited chunk.
4.2. Origins of Citation Criteria in Generative Search Engines (RQ2)
4.2.1. RAG Experimental Design and Variables
In our exploration of RQ1, we have documented the stylistic and semantic preferences exhibited
by Google’s AI Overview. However, given the additional engineering processes required to deploy
the underlying LLM within generative search engines, these preferences may be specifically and
artificially designed for such systems. While these preferences could theoretically emerge naturally,
we seek to test their boundaries through direct interaction with LLM-based RAG systems.
To do so, we leverage the website-level dataset constructed in RQ1 as the same source, but use
Google’s Gemini RAG API to provide answers to each query with the same query, as summarized by

Author:Generative Search Engine22Article submitted toInformation Systems Research; manuscript no. ISRE-0000-0000.00
Algorithm 3. Specifically, for each query, we combine this query’s related chunks from the website-
level dataset into one PDF as the source file, each chunk is labeled with an index. Since this is in a
controlled environment, we can safely avoid the confounding effect from position bias by randomly
ranking all these chunks. Then, we call the Gemini RAG API and ask the AI to mimic Google AI
Overview’s function and answer the query, and it can cite chunks from the source file with specific
indexes to support the answer10. Therefore, for each query, we can know exactly which chunks the
RAG system chooses to cite, and we relabel the original dataset withRAGCite q,c. Since the original
sources stay the same, thePPL q,candSimilarity q,c,c′will be the same. The summary statistics
are again shown in Table 4 for references.
To address this objective, we employ the website-level dataset constructed in RQ1, utilizing
Google’s Gemini RAG API to generate responses for each query, as delineated in Algorithm 3.
Specifically, for each queryq, we aggregate the corresponding chunks from the website-level dataset
into a single PDF document that serves as the source file, with each chunk assigned a unique index.
This controlled experimental design enables us to mitigate potential confounding effects arising
from position bias through random ordering of all chunks (Shi et al. 2024). Subsequently, we invoke
the Gemini RAG API with instructions to emulate the functionality of Google AI Overview in
answering the query, whereby the system can reference specific chunks from the source file using
theircorrespondingindicestosubstantiateitsresponse.WepreciselyidentifywhichchunkstheRAG
system selects for citation for each query, thereby relabeling the original dataset withRAGCite q,c.
Given that the source materials remain constant, the values ofPPL q,candSimilarityq,c,c′remain
unchanged. The summary statistics for the new dataset are presented in Table 4 for reference.
4.2.2. Empirical Results for RQ2
Employing the previously described dataset, we examine whether the observed stylistic and
semantic preferences represent intrinsic model characteristics or constitute engineered modifications
10The specific prompt can be found in Appendix C.

Author:Generative Search Engine
Article submitted toInformation Systems Research; manuscript no. ISRE-0000-0000.0023
Algorithm 3RAG Experimental Protocol for RQ2
Require:Query–website mappings from RQ1
1:Initialize dataset for RQ2 analysis
2:foreach queryido
3:Compile a labeled document of relevant website chunks
4:Submit the document to the Gemini RAG system together with the query
5:Prompt the model to provide an integrated summary and to indicate citations
6:Record the cited sources
7:end for
Table 4 Summary Statistics for RQ2 Dataset
Statistic N Mean St. Dev. Min Median Max
RAGCite98,477 0.33 0.47 0 0 1
PPL98,477 16.42 9.52 1.09 13.86 82
Similarity1,261,914 0.54 0.16 -0.10 0.55 0.99
specificallydesignedforgenerativesearchfunctionality.Ourexperimentalframeworkmaintainsiden-
tical variable construction methodologies, with the exception that citation labels are now generated
through Gemini’s RAG API rather than Google’s AI Overview system.
In the revised dataset, we define the citation outcome variable asRAGCite q,c, which takes the
value of 1 if chunkcreceives a citation in the RAG response to queryq, and 0 otherwise. Each
chunk’s perplexity is denoted asPPL c, while its positional placement within the source document
is represented byPos c. After controlling for query-level fixed effectsQ q, we specify the following
empirical model to examine the perplexity effect within the RAG system:
RAGCite q,c=β 0+β 1∗PPL q,c+β 2∗Pos q,c+Q q+ϵq,c (4)
Additionally, utilizing the revised citation labels, we maintain identical semantic similarity con-
structionprocedures,andemployaspecificationanalogoustothatpresentedintheprecedingsection
to evaluate differences in semantic similarity among cited chunks relative to that among non-cited
chunks. The sole modification is the citation determination mechanism, which is now governed by

Author:Generative Search Engine24Article submitted toInformation Systems Research; manuscript no. ISRE-0000-0000.00
the RAG system. Consequently, the independent variable of interest transitions fromBothCite q,c,c′
toBothRAGCite q,c,c′:
Similarity q,c,c′=β 0+β 1∗BothRAGCite q,c,c′+Q q+ϵq,c,c′ (5)
Interestingly, our regression analyses demonstrate that both observed preferences manifest con-
sistently within the RAG system architecture. First, the Gemini-based RAG exhibits a significant
propensity to select content with lower perplexity (Table 5). Second, chunks cited by the Gemini-
based RAG demonstrate significantly greater semantic similarity among themselves relative to the
semantic similarity observed among non-cited chunks (Table 6). This consistency provides com-
pelling evidence that these mechanisms are deeply rooted in the underlying language models: they
preferentially cite content that is predictable from their computational perspective, thereby align-
ing with the objective function of such models, and select semantically similar content to ensure
coherence in their generated outputs.
Table 5 RAG’s Perplexity Effect and Position Bias
RAGCite
LPM Logit
(1) (2)
PPL-0.0067*** -0.0427***
(0.0002) (0.0010)
Pos-0.0145*** -0.0849***
(0.0003) (0.0011)
Query FE Yes Yes
Observations 98,477 98,477
R2orChi20.21 10,248
Note: Cluster-robust standard errors are reported at the query level. ***p <0.01, **p <0.05, *p <0.1.
Furthermore, our analysis reveals that positional placement within the source document signifi-
cantly influences the probability of selection for citation (β 2of Table 5). Given that these systems
exhibit a preference for content positioned at the document’s beginning, this finding suggests an
additional content optimization strategy for website proprietors: they may strategically place core
arguments and key information at the beginning of websites to ensure that primary content remains
“visible” to generative retrieval systems and receives favorable consideration for citation inclusion.

Author:Generative Search Engine
Article submitted toInformation Systems Research; manuscript no. ISRE-0000-0000.0025
Table 6 RAG’s Semantic Similarity Effect
Similarity
(1) (2)
BothRAGCite0.1206*** 0.0964***
(0.0016) (0.0013)
Cross-Category No Yes
Query FE Yes Yes
Observations 766,243 1,261,914
R20.36 0.32
Note: Cluster-robust standard errors are reported at the query level. ***p <0.01, **p <0.05, *p <0.1.
4.3. Impact of LLM-Driven Polishing on AI summary’s Content Diversity (RQ3)
4.3.1. LLM-driven Polishing, RAG, and Variables
Having explored the preferences of generative search engines and verified the underlying mech-
anisms through RAG API analysis, we now examine the implications from website owners’ per-
spectives. A natural strategy for increasing content predictability from an LLM’s computational
perspective is to employ LLMs for content refinement, as such polished content, being generated
token-by-token through the LLM’s conditional probability optimization process, typically exhibits
lower perplexity. Moreover, utilizing LLMs for marketing content generation has already emerged
as a prevalent application (Ye et al. 2025, Reisenbichler et al. 2022). However, such refinement
practices may inadvertently result in content homogenization across different website providers,
thereby reducing the diversity of the information ecosystem. To assess how LLM-based content
polishing influences information diversity in generative search, we conduct a controlled experiment
using Gemini’s API, simulating a strategic environment where AI systems from different parties
interact competitively.
Prior to processing source files through the RAG API as described in Section 4.2, we implement
two types of content optimization using Google’s Gemini 1.5 Flash model for each chunk. The
first approach employs a general refinement procedure that enhances clarity and engagement while
preserving substantive meaning. This optimization simulates scenarios where website owners uti-
lize LLMs to assist in the content creation process. The second approach implements goal-oriented
optimization that explicitly structures content to increase the likelihood of citation in Google’s AI

Author:Generative Search Engine26Article submitted toInformation Systems Research; manuscript no. ISRE-0000-0000.00
Overview. This mimics the strategic behavior whereby website owners deliberately consider gener-
ative search engine exposure during content development. Both prompt specifications are provided
in Appendix C.
Following this optimization phase, we repeat the RAG experiments developed in RQ2 under three
parallelconditions:usingtheoriginalcontent,usingthepolishedversion,andusingtheAI-optimized
version.
Algorithm 4LLM-driven Content Optimization and RAG Testing
Require:Original chunks from RQ1
1:Phase 1: Content Optimization
2:foreach chunkcin datasetdo
3:Produce general polished version (c pol)
4:Produce AI-optimized version(c opt)
5:Store:(c,c pol,copt)
6:end for
7:Phase 2: RAG Experiments (3 conditions)
8:foreach condition∈{original, polished, AI-optimized}do
9:Create PDFs using chunks from current condition
10:Execute RAG protocol (same as RQ2)
11:Record citation results for each condition
12:end for
13:returnCitation results for all three conditions
Implementing the same procedure outlined in Section 4.2, we obtain the citation list for each
query and calculate the pairwise similarity between all cited chunks, enabling us to measure and
compare information diversity across three experimental conditions. Further, we compute the total

Author:Generative Search Engine
Article submitted toInformation Systems Research; manuscript no. ISRE-0000-0000.0027
number of citations and the perplexity of the AI-generated summary for each query to provide
additional comparative analysis. The summary statistics are provided in Table 711.
Table 7 Summary Statistics for RQ3 Dataset
Statistic N Mean St. Dev. Min Median Max
Similarity3,785,742 0.53 0.16 -0.15 0.54 0.99
NumCite12,180 9.02 5.08 1 8 27
OutputPPL12,180 8.37 8.81 2.72 7.88 95
4.3.2. Empirical Results for RQ3
Utilizing the three distinct content treatment conditions (original chunks, AI-polished chunks,
and objective-oriented AI-polished chunks) and their corresponding RAG selection outcomes, we
examine the counterfactual effects of LLM-based content optimization prior to RAG processing.
This analysis simulates real-world scenarios wherein website proprietors deploy LLMs for content
generation and optimization purposes.
Initially, we assess overall semantic similarity patterns across all chunks under the three experi-
mental conditions. We denote the original content condition asT= 0, AI-polished content without
specific objectives asT= 1, and AI-polished content with citation optimization objectives asT= 2.
For each pair of chunks (candc′) derived from queryqunder treatment conditionT, we calcu-
late the pairwise semantic similarity measureSimilarity T,q,c,c′. We regress the similarity on the
condition indicator, with query-specific fixed effects included as well. Additionally, we conduct a
focused analysis restricting the sample to cited websites exclusively, examining how semantic sim-
ilarity among cited sources varies across different input treatment conditions. Both analyses are
presented in Table 8.
Similarity T,q,c,c′=β 0+β 1·1(T= 1) T,q,c,c′+β 2·1(T= 2) T,q,c,c′+Q q+ϵT,q,c,c′ (6)
11With 4,060 queries and three conditions, we obtain 12,180 observations for query-level measurements. For similarity
analyses, we generate 1,261,914 pairs under each condition, yielding 3,785,742 total observations.

Author:Generative Search Engine28Article submitted toInformation Systems Research; manuscript no. ISRE-0000-0000.00
Table 8 Impact of Sources’ LLM Polishing on (Cited) Content’s Similarity
Similarity
(1) (2)
T= 1-0.0096*** -0.0319***
(0.0004) (0.0011)
T= 2-0.0091*** -0.0722***
(0.0005) (0.0011)
Sample All Cited Only
Query FE Yes Yes
Observations 3,785,742 597,533
R20.29 0.42
Note: Cluster-robust standard errors are reported at the query level. ***p <0.01, **p <0.05, *p <0.1.
Our empirical findings reveal a surprising pattern: while overall semantic diversity decreases at
a small magnitude under AI polishing (β 1=−0.0096, Column (1), Table 8), semantic similarity
among cited websites decreases significantly (β 1=−0.0319, Column (2), Table 8). This phenomenon
likely occurs because such content polishing preserves original semantic content while rendering pre-
viously unpredictable material sufficiently predictable for LLM consideration. Consequently, as the
pool of stylistically compatible content (characterized by low perplexity) expands, the RAG system
can incorporate a broader range of external sources, thereby enhancing citation diversity among
selected websites. Moreover, objective-oriented polishing yields even more substantial improvements
in outcome diversity (β 2=−0.0722, Column (2), Table 8). This result indicates that despite the
absence of detailed instructions, LLMs demonstrate inherent understanding of enhancement strate-
gies, enabling the RAG system to consider an expanded set of potential sources autonomously.
We further investigate this underlying mechanism through two complementary analyses. First,
we examine the number of chunks cited per queryNumCite T,qas a function of treatment condi-
tions. Second, we analyze the perplexity of RAG-generated responsesOutputPPL T,qunder different
treatment scenarios. Both dependent variables are modeled using the following specification:
Outcome T,q=β 0+β 1·1(T= 1) T,q+β 2·1(T= 2) T,q+ϵT,q (7)
As demonstrated in Column (1) of Table 9, we observe a significant increase in the number of
citations following content polishing, with the effect being particularly pronounced when optimiza-
tion includes the citation objective. Importantly, this expanded citation scope does not compromise

Author:Generative Search Engine
Article submitted toInformation Systems Research; manuscript no. ISRE-0000-0000.0029
Table 9 Impact of Sources’ LLM Polishing on Outputs’ Attributes
NumCite OutputPPL
(1) (2)
T= 11.0744*** -0.0431
(0.1071) (0.0823)
T= 22.1059*** -0.0451
(0.1098) (0.0762)
Observations 12,180 12,180
R20.03 0.02
Note: Robust standard errors are reported. ***p <0.01, **p <0.05, *p <0.1.
the predictability of RAG outputs; conversely, both treatment conditions yield marginally decreased
output perplexity (Column (2), Table 9). These collective findings indicate that LLM-based content
polishing does not directly induce homogenization effects. Rather, by enhancing content accessibil-
ity and interpretability for language models, such optimization strategies may actually improve the
information diversity available to users through generative retrieval responses.
5. User-Side Consequences of LLM-based polishing (RQ4)
Our prior analyses demonstrate that LLM-driven content polishing, while optimized for citation
probability in generative search engines, paradoxically enhances information diversity within AI-
generated summaries. This finding naturally leads to questions regarding downstream implica-
tions: How does this transformation of the information ecosystem influence end users’ information
acquisition experiences? To investigate the effects, we develop a controlled experimental platform
replicating Google’s AI Overview functionality and recruit participants from Prolific to conduct a
between-subjects randomized controlled trial addressing our final research question (RQ4).
5.1. Experimental Design
We design and implement a custom generative search platform that replicates the functional-
ity of Google AI Overview while enabling comprehensive capture of user behavioral data within
our controlled environment (see Appendix I for the interface). Utilizing this platform, we recruit
150 participants through Prolific to conduct a between-subjects randomized experiment, thereby
investigating the effects of LLM polishing on website source evaluation and utilization.

Author:Generative Search Engine30Article submitted toInformation Systems Research; manuscript no. ISRE-0000-0000.00
To reflect what users obtain from generative search engines, we ask participants to complete a
writing task (Noy and Zhang 2023). Since we seek to polish websites in the treatment condition, we
focus on one specific topic so that polished website materials can be prepared in advance. We select
theteenagers’smartphonebanpolicyasourresearchtopic:participantsassumetheroleofeducation
policy consultants, tasked with researching and recommending whether a public school district
should implement a smartphone ban during school hours. We choose this task for several reasons:
(1) it represents a contemporary policy debate with no definitive correct answer, (2) it requires
synthesizing diverse perspectives, including educational, psychological, and practical considerations,
and (3) it mirrors the type of complex decision-making processes that generative search engines
are designed to support. The specific instructions that participants receive are also presented in
Appendix I.
To compile website sources for user searches, we conduct searches on topics encompassing the
advantages and disadvantages of smartphone bans, potential enforcement mechanisms, alternative
approaches to address smartphone usage limitations, and related case studies. We obtain relevant
website links from Google (including AI Overview) and Bing (including Microsoft Copilot), employ-
ing the following extraction methodology: for generative search results, we extract all referenced
websites, while for conventional search results, we extract websites from the first page of organic
results. Across all search channels and keywords, we compile 65 unique websites pertaining to this
topic. We then download and process the complete content of these 65 websites to establish our
baseline information source (Source “Control”). For the treatment condition, we apply the identical
LLM-polishing procedure with the citation optimization objective described in Section 4.3 to gen-
erate a parallel set of polished content (Source “Treatment”). These two source conditions are the
sole experimental manipulation that participants encounter during the study.
Based on these sources, we construct a custom web-based generative search engine utilizing the
Gemini RAG API to conduct the randomized experiment, with detailed procedural information
provided in Appendix I. When users enter the platform and provide their Prolific ID, the underlying

Author:Generative Search Engine
Article submitted toInformation Systems Research; manuscript no. ISRE-0000-0000.0031
treatment assignment becomes fixed, with 50% allocated to the treatment group and 50% to the
control group, respectively. Users then review the instructions and query our generative search
engine, which returns AI-generated summaries with relevant website citations, replicating Google
AI Overview’s functionality. Both experimental groups encounter an identical user interface, with
the sole distinction being the previously mentioned source manipulation: the RAG system generates
responses for treatment group participants based on Source “Treatment,” while responses for control
group participants are based on Source “Control.” Users can conduct unlimited searches, with each
query-response interaction recorded for subsequent analysis. To ensure task engagement and prevent
mechanical reproduction of content, we disable paste functionality in both query and response fields,
requiring participants to actively synthesize information using their own formulation.
Each participant submits one final recommendation (100-300 words), after which the platform
redirects participants to a post-task survey, with the complete questionnaire presented in Appendix
J. We also collect standard demographic information to conduct randomization checks and explore
potential treatment effect heterogeneities. Three participants fail to finish the survey and are thus
dropped from the analyses.
5.2. Experimental Data Analyses and Results
Before formally testing the treatment effect, we validate the randomization assumption by conduct-
ing pairwise t-tests and Kolmogorov-Smirnov (KS) tests on all demographic variables and prior
experience with relevant tools. As demonstrated in Appendix K, we confirm that participants in the
treatment and control groups exhibit no statistically significant differences across these dimensions.
Based on the dataset collected from the experiment, we directly compare the submitted recom-
mendations and task efficiency across the two experimental groups. Motivated by our exploration of
RQ3, where such treatments increase the information diversity of AI summary outputs, we capture
the information diversity of participants’ submissions using the Vendi score (Friedman and Dieng
2022). Regarding task efficiency, we calculate the time spent by each participant. Formally, for each

Author:Generative Search Engine32Article submitted toInformation Systems Research; manuscript no. ISRE-0000-0000.00
participanti, we deriveInformation i(in Vendi score) andTimeSpent i(in minutes) as two pri-
mary outcomes of interest (Outcome i). Each participant possesses a treatment statusTreat i, and
we estimate the following regression models to derive the treatment effects.
Outcome i=β 0+β 1·Treat i+ϵi (8)
Furthermore, as education levels are commonly associated with individuals’ adoption and usage
of technologies (Ma et al. 2024, Xu et al. 2023), we explore the heterogeneous treatment effects
across different education levels. Specifically, we denote a participant’s education level with a binary
variableCollege i, which equals 1 if the participant possesses a college degree or higher, and 0
otherwise. We then partition the participants into two subsamples and conduct subsample analyses
using the same specification as Equation (8).
AsdemonstratedinTable10,wefindthattreatedparticipantstendtoprovidewrittensubmissions
with significantly higher information diversity, an effect primarily driven by participants without
college degrees. Meanwhile, participants with college degrees or higher tend to provide submissions
with similar information diversity across both groups, but treated participants can complete the
task in significantly less time.
These results reveal interesting behavioral patterns: relatively more educated individuals already
possess the tendency to seek diverse information sources; LLM-polishing and the consequent infor-
mation diversity enhancement enable them to increase their efficiency. However, less educated par-
ticipants tend to rely on immediately available information, and their written expressions are more
substantially influenced by the information enhancement per query.
6. Robustness Checks
We conduct a series of robustness checks, including a replication study, alternative samples, and
alternative specifications. Due to space constraints, we summarize them in Appendix A.
7. Discussion and Conclusion
This concluding section synthesizes our empirical findings, delineates the theoretical contributions
and practical implications of our research, and identifies several limitations alongside promising
avenues for future scholarly investigation.

Author:Generative Search Engine
Article submitted toInformation Systems Research; manuscript no. ISRE-0000-0000.0033
Table 10 Experimental Results
Information TimeSpent
(1) (2) (3) (4) (5) (6)
Treat0.3554*** 0.0690 0.6123*** -1.5942*** -2.9186*** -0.3858
(0.0688) (0.0873) (0.0862) (0.3033) (0.3685) (0.3303)
College All 1 0 All 1 0
Observations 147 68 79 147 68 79
R20.15 0.01 0.40 0.16 0.49 0.02
Note: Robust standard errors are reported. ***p <0.01, **p <0.05, *p <0.1.
7.1. Summary of Results
Through systematic interactions with both generative and traditional search engines, complemented
by controlled experimentation with human participants, we have systematically addressed each
research question and generated several key empirical findings. First, our analysis reveals that
generative engines exhibit systematic preferences for content characterized by high predictability
(low perplexity) and semantic homogeneity among cited sources (RQ1). Second, we demonstrate
that this citation behavior stems from fundamental characteristics embedded within the underly-
ing large language model architecture rather than platform-specific engineering decisions (RQ2).
Third, contrary to expectations of AI-induced homogenization effects, comprehensive content pol-
ishing through LLMs actually enhances information diversity within AI-generated summaries. This
unexpected outcome results from the perplexity-reduction effects of LLM polishing, which expands
the pool of content eligible for citation consideration by generative search systems (RQ3). Finally,
our randomized controlled trial demonstrates that LLM-enhanced information diversity translates
into measurable improvements in task efficiency and information acquisition among end users.
Notably, we observe differential treatment effects across educational backgrounds: highly educated
participants primarily benefit through enhanced efficiency, while participants with lower educational
attainment experience gains in information comprehension and utilization (RQ4).
7.2. Contributions and Implications
Our research advances theoretical understanding across multiple domains. First, we contribute to
the search engine optimization literature by identifying novel content provision criteria specific to

Author:Generative Search Engine34Article submitted toInformation Systems Research; manuscript no. ISRE-0000-0000.00
generative search engines. We reveal citation preferences that were previously unobservable within
traditional ranking algorithms, and more importantly, demonstrate that these preferences emerge
from the intrinsic language model rather than explicit engineering design choices. Second, within
the broader search engine ecosystem framework, we establish that LLM-based content polishing
generates “win-win” outcomes: expanded website appearances within AI summaries while simulta-
neously enhancing the semantic diversity of user-facing content. Third, our behavioral experiment
contributes to the literature on AI summaries by demonstrating that information increases in AI-
generated summaries yield differential benefits across users with varying backgrounds in terms of
both task efficiency and output information value, thereby advancing our understanding of human
interactions with AI summary systems.
Our findings extend beyond academic interest to provide actionable insights for diverse stake-
holder communities. First, for website proprietors and generative search engine optimization service
providers, the connection between foundational language models and LLM-based AI Overview sys-
tems presents novel optimization strategies. Practitioners can leverage RAG systems built upon
identical model families (e.g., Gemini) to conduct offline optimization testing prior to website
deployment, or even utilize open-source variants (e.g., Gemma) to expedite the optimization process
while maintaining cost efficiency. Second, for information-seeking users, awareness of AI Overview’s
inherent constraints is crucial: the system’s design objective of generating coherent paragraph-form
responses may result in systematically narrowed perspectives. While users benefit from enhanced
search efficiency, comprehensive information gathering may still require supplementary search activ-
ities to achieve a holistic understanding. Third, our results demonstrate that LLM-based content
polishing can yield beneficial outcomes under certain conditions. However, the intrinsic relationship
between underlying language models and AI Overview systems creates potential vulnerabilities to
strategic manipulation. Search engine operators may need to carefully engineer distinctions between
RAG APIs and AI Overview functionalities, and simultaneously reconsider open-source model dis-
tribution strategies to mitigate potentially harmful gaming behaviors by website proprietors.

Author:Generative Search Engine
Article submitted toInformation Systems Research; manuscript no. ISRE-0000-0000.0035
7.3. Limitations and Future Directions
Our study acknowledges several limitations that suggest promising directions for future research.
First, while Google dominates the search engine market, the external validity of our findings may
remain constrained. To address this limitation, we conducted robustness checks using Microsoft’s
Bing and examined underlying RAG systems to elucidate deeper mechanisms. Nevertheless, future
research investigating how alternative generative search engine architectures might mitigate these
tendencies would help establish the boundary conditions of our results. Second, regarding tempo-
ral validity, our findings may not persist indefinitely as these technologies evolve. However, our
data collection spans multiple time periods (Bing data from 2023 and Google Overview data from
2025), demonstrating that the observed effects have persisted for approximately two and a half
years since the initial deployment of generative search engines. Given the apparent permanence of
generative search technologies, our investigation falls into the “Big Questions, Half Answers” cat-
egory of IS research (Hosanagar 2017), and we hope it can stimulate further scholarly discourse.
Finally, concerning human-AI interaction analysis, we acknowledge the inherent limitations of our
online experimentation approach. Field studies utilizing proprietary platform data could yield more
comprehensive insights into the broader behavioral impacts of AI Overview systems.
References
Abhishek V, Hosanagar K (2013) Optimal bidding in multi-item multislot sponsored search auctions.Oper-
ations Research61(4):855–873.
Alavi S, Nozari S (2023) Ai product review summaries and content homogenization: The changing landscape
of ugc. Available at SSRN:https://ssrn.com/abstract=5178862.
Baye MR, De los Santos B, Wildenbeest MR (2016) Search engine optimization: what drives organic traffic
to retail sites?Journal of Economics & Management Strategy25(1):6–31.
Berman R, Katona Z (2013) The role of search engine optimization in search marketing.Marketing Science
32(4):644–651.
Berti L, Giorgi F, Kasneci G (2025) Emergent abilities in large language models: A survey.arXiv preprint
arXiv:2503.05788.

Author:Generative Search Engine36Article submitted toInformation Systems Research; manuscript no. ISRE-0000-0000.00
Bolton E, Venigalla A, Yasunaga M, Hall D, Xiong B, Lee T, Daneshjou R, Frankle J, Liang P, Carbin M,
et al. (2024) Biomedlm: A 2.7 b parameter language model trained on biomedical text.arXiv preprint
arXiv:2403.18421.
Cer D, Yang Y, Kong Sy, Hua N, Limtiaco N, John RS, Constant N, Guajardo-Cespedes M, Yuan S, Tar C,
et al. (2018) Universal sentence encoder for english.Proceedings of the 2018 conference on empirical
methods in natural language processing: system demonstrations, 169–174.
Danaher PJ, Mullarkey GW, Essegaier S (2006) Factors affecting web site visit duration: A cross-domain
analysis.Journal of Marketing Research43(2):182–194.
De Cornière A, Taylor G (2014) Integration and search engine bias.RAND Journal of Economics45(3):576–
597.
Du X, Su M, Zhang X, Zheng X (2017) Bidding for multiple keywords in sponsored search advertising:
Keyword categories and match types.Information Systems Research28(4):711–722.
Epstein R, Robertson RE (2015) The search engine manipulation effect (seme) and its possible impact on
the outcomes of elections.Proceedings of the national academy of sciences112(33):E4512–E4521.
Erdmann A, Arilla R, Ponzoa JM (2022) Search engine optimization: The long-term strategy of keyword
choice.Journal of Business Research144:650–662.
Friedman D, Dieng AB (2022) The vendi score: A diversity evaluation metric for machine learning.arXiv
preprint arXiv:2210.02410.
Gao Y, Wang Z, Huang Y (2025) Pandora box or golden fleece: Economic analysis of generative ai adoption
on creation platforms. Working paper.
Ghose A, Ipeirotis PG, Li B (2014) Examining the impact of ranking on consumer behavior and search
engine revenue.Management Science60(7):1632–1654.
Goldfarb A, Tucker C (2011) Online display advertising: Targeting and obtrusiveness.Marketing Science
30(3):389–404.
Gonen H, Iyer S, Blevins T, Smith NA, Zettlemoyer L (2022) Demystifying prompts in language models via
perplexity estimation.arXiv preprint arXiv:2212.04037.

Author:Generative Search Engine
Article submitted toInformation Systems Research; manuscript no. ISRE-0000-0000.0037
Gong J, Abhishek V, Li B (2018) Examining the impact of keyword ambiguity on search advertising perfor-
mance.MIS Quarterly42(3):805–A14.
Guo B, Zhang X, Wang Z, Jiang M, Nie J, Ding Y, Yue J, Wu Y (2023) How close is chatgpt to human
experts? comparison corpus, evaluation, and detection.arXiv preprint arXiv:2301.07597.
Hagiu A, Jullien B (2011) Why do intermediaries divert search?RAND Journal of Economics42(2):337–362.
Hosanagar K (2017) Senior editor perspectives.
Hou JJ, Wang L, Wang G, Wang H, Yang S (2024) The double-edged roles of generative ai in the creative
process: Experiments on design work. Available at SSRN:https://ssrn.com/abstract=4739471.
Ji Y, Gao C, Wan XS, Li Z (2025) Designing ai-generated summaries for online video platforms: Evidence
from a field experiment. Available at SSRN:https://ssrn.com/abstract=5285325.
Kim A, Lu Y, Ma T, Tan Y (2024) Less is more? impact of ai-generated summaries on user engagement of
video-sharing platforms Available at SSRN.
KooWW,PuJ,GaoN(2025)Aireviewsummary,customerratings,andperformanceentrenchmentWorking
paper.
Liu J, Toubia O (2018) A semantic approach for estimating consumer content preferences from online search
queries.Marketing Science37(6):930–952.
Ma L, Xu X, He Y, Tan Y (2024) Learning to adopt generative ai.arXiv preprint arXiv:2410.19806.
McElheran K, Li JF, Brynjolfsson E, Kroff Z, Dinlersoz E, Foster L, Zolas N (2024) Ai adoption in america:
Who, what, and where.Journal of Economics & Management Strategy33(2):375–415.
McLuhan M (1964)Understanding Media: The Extensions of Man(Routledge).
Moon K, Green A, Kushlev K (2024) Homogenizing effect of large language model (llm) on creative diversity:
An empirical comparison.
Noy S, Zhang W (2023) Experimental evidence on the productivity effects of generative artificial intelligence.
Science381(6654):187–192.
O’Reilly CA (1980) Individuals and information overload in organizations: Is more necessarily better?
Academy of Management Journal23(4):684–696.

Author:Generative Search Engine38Article submitted toInformation Systems Research; manuscript no. ISRE-0000-0000.00
Pariser E (2011)The filter bubble: What the Internet is hiding from you(penguin UK).
Qiao D, Rui H, Xiong Q (2023) Ai and jobs: Has the inflection point arrived? evidence from an online labor
platform. arXiv preprint arXiv:2312.04180.
Reimers N, Gurevych I (2019) Sentence-bert: Sentence embeddings using siamese bert-networks.Proceedings
of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International
Joint Conference on Natural Language Processing (EMNLP-IJCNLP), 3982–3992.
Reisenbichler M, Reutterer T, Schweidel DA, Dan D (2022) Frontiers: Supporting content marketing with
natural language generation.Marketing Science41(3):441–452.
Shi L, Ma C, Liang W, Ma W, Vosoughi S (2024) Judging the judges: A systematic investigation of position
bias in pairwise comparative assessments by llms .
Shin W (2015) Keyword search advertising and limited budgets.Marketing Science34(6):882–896.
Ursu R, Seiler S, Honka E (2025) The sequential search model: A framework for empirical research.Quan-
titative Marketing and Economics23(1):165–213.
Vaswani A, Shazeer N, Parmar N, Uszkoreit J, Jones L, Gomez AN, Kaiser Ł, Polosukhin I (2017) Attention
is all you need.Advances in neural information processing systems30.
Wang W, Yang M, Sun T (2023) Human-ai co-creation in product ideation: The dual view of quality and
diversity.Available at SSRN 4668241.
Xu R, Feng Y, Chen H (2023) Chatgpt vs. google: A comparative study of search performance and user
experience.arXiv preprint arXiv:2307.01135.
Xu Z, Sheng VS (2024) Detecting ai-generated code assignments using perplexity of large language models.
Proceedings of the aaai conference on artificial intelligence, volume 38, 23155–23162.
Yan Z, Chen J, Raghunathan S (2024) The creative divide: How ai shapes value and inequality among
creators. Available at SSRN:https://ssrn.com/abstract=5236972.
Ye Z, Yoganarasimhan H, Zheng Y (2025) Lola: Llm-assisted online learning algorithm for content experi-
ments.Marketing Science.
Yeverechyahu D, Mayya R, Oestreicher-Singer G (2024) The impact of large language models on open-source
innovation: Evidence from github copilot. arXiv preprint arXiv:2409.08379.

Author:Generative Search Engine
Article submitted toInformation Systems Research; manuscript no. ISRE-0000-0000.0039
Yuan L, Chan FY, Gao C, Leung A, Gu B, Ye Q (2024) Ai-generated summaries as differentiating refer-
ence: Impact on user content generation in online communities. Boston University Questrom School of
Business Research Paper No. 5180947, Available at SSRN:https://ssrn.com/abstract=5180947.
Zhou M, Zhang J, Adomavicius G (2024) Longitudinal impact of preference biases on recommender systems’
performance.Information Systems Research35(4):1634–1656.

Author:Generative Search Engine40Article submitted toInformation Systems Research; manuscript no. ISRE-0000-0000.00
Appendix A: Summary of Robustness checks
Although Google dominates the search engine market, these mechanisms should be generalizable to different
LLM-based generative search engines. Therefore, we utilize a dataset collected through interactions with
New Bing (subsequently renamed Microsoft Copilot) in April 2023 and replicate our analyses. The detailed
results are presented in Appendix D.
Furthermore, our analyses identify outlier websites with excessively high perplexity values that may influ-
ence the results; consequently, we repeat the analyses with these outliers excluded. The findings are docu-
mented in Appendix E.
Additionally, we observe that conventional search engines yield slightly more websites on average, which
may contribute to the higher similarity observed among generative search engines’ cited websites. To address
this potential confound, we restrict both search channels to the same number of websites for each query and
rerun the similarity analyses, with results presented in Appendix F.
Finally, we evaluate alternative model specifications to assess sensitivity to different parametric forms,
with findings reported in Appendix G.

Author:Generative Search Engine
Article submitted toInformation Systems Research; manuscript no. ISRE-0000-0000.0041
Appendix B: Raw Data Collection Process
In this appendix, we detail the procedures used for data collection and dataset construction. Algorithm B1
illustrates our data collection process.
Algorithm B1Data Collection Process
Require:Query listQfrom HC3 corpus
1:foreach queryq∈Qdo
2:Initiate candidate website setS q
3:Send the queryqto Google Search via the SerpAPI general endpoint
4:ifthe response indicates a request to AI Overview specific endpoint is requiredthen
5:Send the queryqto AI Overview specific endpoint
6:end if
7:Extract AI Overview response textt q
8:S q←websites cited in AI Overview
9:S q←first page of organic search results
10:foreach websiteiinS qdo
11:Render the web page in a browser and extract its contentc qi
12:end for
13:end for
The unified collection process ensures that both AI Overview data and conventional search results are
fully extracted for each query in the HC3 dataset. For each query from the HC3 dataset, we utilize the
SerpAPI to obtain both the AI Overview results and the first page of organic results. Sometimes the general
endpoint does not directly return Google’s AI Overview but returns a parameter that indicates an additional
request to the AI Overview specific endpoint is required12. In that case, we would send another request
accordingly. For the AI Overview part, we save both the response text and the URLs of the cited websites.
For the conventional search results, we record the ranking position, title, URL, and snippet of each listed
12Seehttps://serpapi.com/blog/scrape-google-ai-overviews/

Author:Generative Search Engine42Article submitted toInformation Systems Research; manuscript no. ISRE-0000-0000.00
website. For each queryq, we construct a candidate website set consisting of sites that appear either in the
AI Overview or among the top conventional search results.
In the next step, we collect content from all candidate websites. We employ browser automation to ensure
each web page is rendered dynamically, allowing us to capture the full content. We then filter out non-textual
elements such as images and stylesheets, retaining only the textual content relevant for downstream analyses.

Author:Generative Search Engine
Article submitted toInformation Systems Research; manuscript no. ISRE-0000-0000.0043
Appendix C: Prompt Content
System Prompt for RAG API:Assume that you are the Google AI Overview generator, a feature inte-
grated into Google Search that provides AI-generated summaries of search results. Please answer the following
query based on the website content contained in the attached PDF file. Within the PDF file, there is a list
of numbered paragraphs, each of which represents a website’s content indicated by a unique ID in the format
“Source 11,” etc. Please mimic Google AI Overview’s answering style. For each sentence, if you can find
references from the PDF, cite the specific ID of that website’s content. For citations, use the EXACT format:
%%%X,Y,Z%%%. Separate multiple source IDs with commas. Do NOT use any other citation format, such
as (Source X). Example: “This is an example statement. %%%1,5,12%%%.”
System Prompt for Content Polishing:Here is an excerpt from a webpage: ’excerpt’. Please polish the
excerpt so that it is clearer and more engaging. Try to keep the length roughly unchanged. Only return the
polished excerpt itself.
System Prompt for Content Polishing with the objective:Here is an excerpt from a webpage:
’excerpt’. Please polish the excerpt so that it is clearer and more engaging. Try to keep the length roughly
unchanged. The primary goal is to make this specific excerpt (and, by extension, the overall webpage) more
likely to be selected and highlighted by Google Search’s AI Overview feature. Only return the polished excerpt
itself.

Author:Generative Search Engine44Article submitted toInformation Systems Research; manuscript no. ISRE-0000-0000.00
Appendix D: New Bing (Microsoft Copilot) Study
To ensure the robustness of our findings across different large language model architectures and search engine
platforms, we replicate all the analysis on Microsoft New Bing, one of the earliest platforms that launched
a generative search engine in February 202313, which was later named Copilot.
Figure D1 provides a visual representation of New Bing’s search interface architecture. The left panel
displays conventional search results, each comprising three standard components: the website title, cor-
responding hyperlink, and an algorithmically selected excerpt deemed most relevant to the search query.
Meanwhile, the right panel features Bing Chat, which generates a response by integrating its knowledge
base with real-time information extracted from relevant web sources, functionally analogous to Google’s AI
Overview. Source citations appear as hyperlinks positioned at the bottom of the response interface, with
some sources receiving explicit in-text citations, while others are listed as supplementary references without
direct textual attribution. Consistent with Google AI Overview’s citation patterns, a single sentence may
incorporate zero to multiple source references, while each website may support zero to multiple sentences
within the generated response.
Figure D1 An Example of Using New Bing
13Seehttps://www.microsoft.com/en-us/edge/features/the-new-bing.

Author:Generative Search Engine
Article submitted toInformation Systems Research; manuscript no. ISRE-0000-0000.0045
In April 2023, following the same procedure established for our primary Google analyses, we constructed
a dataset by randomly sampling queries from HC3 (Guo et al. 2023). Given the absence of publicly available
APIs for New Bing, we stop the collection process after successfully executing 700 queries, capturing both
the top 20 organic search results from the conventional search panel and the corresponding AI-generated
content from the chat interface. We also downloaded all referenced website content (13,428 in total) from
both channels for analysis.
ToevaluatetherobustnessofourRQ1findings,weimplementanidenticalanalyticalprocedurefortheNew
Bing dataset. Specifically, we apply equivalent chunk selection processes and calculate both perplexity and
semantic similarity measures following our established methodology. The sole methodological modification
involves substituting GPT-2 for Google’s Gemma-2B model in perplexity calculations, given that the GPT
series serves as the foundational architecture underlying Microsoft’s Bing Chat system. Our replication
analysis confirms the consistency of our findings to RQ1: websites cited by Bing Chat exhibit significantly
lower perplexity compared to those exclusively listed in traditional Bing search results, while perplexity
measures demonstrate no significant predictive power for traditional Bing rankings. Additionally, we observe
that Bing Chat’s content selections demonstrate significantly greater semantic similarity compared to sources
prioritized by conventional Bing search algorithms.
Table D1 Perplexity Effect (Bing)
ChatCite SentenceCite
LPM Logit LPM Logit
(1) (2) (3) (4)
PPL-0.0755*** -0.4722*** -0.0443*** -0.6343***
(0.0119) (0.0670) (0.0050) (0.0521)
Query FE Yes Yes Yes Yes
Observations 13,428 13,428 49,917 49,917
R2orChi20.03 49.24 0.05 143.71
Note: Cluster-robust standard errors are reported at the query level. ***p <0.01, **p <0.05, *p <0.1.
To validate the robustness of our finding that observed citation preferences emerge intrinsically from
underlying language model architectures rather than from platform-specific engineering modifications, we
conductanadditionalseriesofLLM-basedRAGexperiments.InthisvalidationstudyconductedinJuly2025,
we substitute Google’s Gemini API with OpenAI’s GPT-4o-mini API, maintaining architectural consistency

Author:Generative Search Engine46Article submitted toInformation Systems Research; manuscript no. ISRE-0000-0000.00
Table D2 Semantic Similarity Effect (Bing)
Similarity
(1) (2)
BothCite0.0934*** 0.0901***
(0.0052) (0.0015)
Cross-Category No Yes
Query FE Yes Yes
Observations 83,923 122,568
R20.42 0.41
Note: Cluster-robust standard errors are reported at the query level. ***p <0.01, **p <0.05, *p <0.1;
“Cross-Category” indicates whetherBothCite= 0includes pairs with one cited chunk and one non-cited chunk.
with New Bing’s GPT-series foundation during our observation period14. Our empirical analyses corroborate
the generalizability of our findings: content selected by the GPT-based RAG system exhibits significantly
lower perplexity and demonstrates substantially greater semantic homogeneity compared to non-selected
content. Additionally, we note that we employ the deprecated GPT-4 for analyses conducted in January
2024 and obtain similar results, further substantiating the generalizability of our findings.
Table D3 RAG’s Perplexity Effect and Position Bias (GPT)
RAGCite
LPM Logit
(1) (2)
PPL-0.0321*** -0.156***
(0.0084) (0.0310)
Pos-0.0163*** -0.0644***
(0.0008) (0.0034)
Query FE Yes Yes
Observations 13,428 13,428
R2orChi20.11 500.55
Note: Cluster-robust standard errors are reported at the query level. ***p <0.01, **p <0.05, *p <0.1.
Finally, we conduct a robustness validation for RQ3, examining how LLM-based content polishing influ-
ences generative search outputs. We implement the identical experimental protocol as our primary Google
analyses, substituting GPT-4o-mini for Gemini in the content refinement process. Our polishing prompts
also specify optimization for Bing Chat appearance to maintain platform-specific relevance. The robustness
14https://blogs.microsoft.com/blog/2023/02/07/reinventing-search-with-a-new-ai-powered-microsoft-bing-and-edge-
your-copilot-for-the-web/

Author:Generative Search Engine
Article submitted toInformation Systems Research; manuscript no. ISRE-0000-0000.0047
Table D4 RAG’s Semantic Similarity Effect (GPT)
Similarity
(1) (2)
BothRAGCite0.0992*** 0.0769***
(0.0035) (0.0028)
Cross-Category No Yes
Query FE Yes Yes
Observations 67,410 122,568
R20.45 0.42
Note: Cluster-robust standard errors are reported at the query level. ***p <0.01, **p <0.05, *p <0.1.
analysis corroborates our principal findings: content optimization results in expanded source citation by the
RAG system, further enhancing the information diversity of generated outputs.
Table D5 Impact of Sources’ LLM Polishing on (Cited) Content’s Similarity (GPT)
Similarity
(1) (2)
T= 1-0.0083*** -0.0284***
(0.0006) (0.0009)
T= 2-0.0078*** -0.0657***
(0.0004) (0.0013)
Sample All Cited Only
Query FE Yes Yes
Observations 367,704 50,933
R20.24 0.40
Note: Cluster-robust standard errors are reported at the query level. ***p <0.01, **p <0.05, *p <0.1.
Table D6 Impact of Sources’ LLM Polishing on Outputs’ Attributes (GPT)
NumCite OutputPPL
(1) (2)
T= 10.7820*** -0.0024
(0.0916) (0.0030)
T= 21.8144*** -0.0035
(0.2179) (0.0041)
Observations 2,100 2,100
R20.04 0.01
Note: Cluster-robust standard errors are reported at the query level. ***p <0.01, **p <0.05, *p <0.1.

Author:Generative Search Engine48Article submitted toInformation Systems Research; manuscript no. ISRE-0000-0000.00
Appendix E: Outlier Removal
Some websites may contain atypical content that exhibits excessive unpredictability from the LLM’s per-
spective, and these outliers may constitute the primary driver of the observed perplexity effects. Therefore,
we exclude content chunks with the top 1% perplexity values and replicate our analyses using the same
specifications as Equations 1 and 2. As demonstrated in Table E1, our results remain robust to the removal
of such outlier websites.
Table E1 Perplexity Effect after Removing Outliers
ChatCite SentenceCite
LPM Logit LPM Logit
(1) (2) (3) (4)
PPL-0.0118*** -0.0550*** -0.0021*** -0.0621***
(0.0002) (0.0010) (0.0001) (0.0016)
Query FE Yes Yes Yes Yes
Observations 97,528 97,528 250,844 250,844
R2orChi20.14 3519.26 0.06 1807.68
Note: Cluster-robust standard errors are reported at the query level. ***p <0.01, **p <0.05, *p <0.1.

Author:Generative Search Engine
Article submitted toInformation Systems Research; manuscript no. ISRE-0000-0000.0049
Appendix F: Alternative Samples
In our dataset, the average number of websites retrieved from conventional search engines exceeds that
from generative search engines. This disparity may partially account for the relatively higher information
diversity observed in conventional search results, as the larger sample size could influence the findings. To
control for this potential confound, we implement a balanced sampling approach: for each queryq, we utilize
the minimum website count across both channels as an upper limit, ensuring equivalent sample sizes for
comparative analysis. Using this balanced sample, we compare the semantic similarity between cited and
non-cited website pairs employing the same specification as Equation 3. As demonstrated in Table F1, our
results remain qualitatively unaffected by this sample balancing process. This robustness confirms that the
observed differences in information diversity are not merely artifacts of unequal sample sizes between search
channels.
Table F1 Semantic Similarity Effect with Balanced Sample
Similarity
BothCite0.0355***
(0.0020)
Query FE Yes
Observations 350,944
R20.28
Note: Cluster-robust standard errors are reported at the query level. ***p <0.01, **p <0.05, *p <0.1.

Author:Generative Search Engine50Article submitted toInformation Systems Research; manuscript no. ISRE-0000-0000.00
Appendix G: Alternative Specifications
In our semantic similarity analyses, the unit of observation comprises individual pairs of website chunks.
Here, we consider an alternative data organization approach and specify corresponding econometric analyses.
Specifically, for each queryq, we calculate the average semantic similarity across all pairs of cited chunks
and denote this value asAvgSimilarity q,1. Simultaneously, we calculate the average semantic similarity
across all pairs of non-cited chunks and denote this value asAvgSimilarity q,0. We conduct a direct pairwise
t-test between these two sets of values and obtain consistent results: cited chunks from the generative engine
exhibit significantly greater mutual similarity compared to those not cited by the generative engine.

Author:Generative Search Engine
Article submitted toInformation Systems Research; manuscript no. ISRE-0000-0000.0051
Appendix H: Perplexity Effect in Conventional Search Engines
Conventional search rankings may also select websites with lower perplexity, potentially establishing a base-
line that could confound our results regarding generative engines. Although the overlap between the two
channels’ selections is minimal in our dataset, we directly test this potential confounding factor. Specifically,
using websites ranked by conventional search engines, we examine whether the perplexity of a website chunk
PPL q,cpredicts the ranking of the websiteRank q,cusing the following specification:
Rank q,c=β 0+β 1∗PPL q,c+Q q+ϵq,c (9)
As demonstrated below, the perplexity effect is absent in conventional ranking mechanisms, thereby elim-
inating this potential confounding factor from our main results.
Table H1 Perplexity Effect (Conventional Search)
Rank
PPL0.0144
(0.0201)
Query FE Yes
Observations 57,653
R20.07
Note: Cluster-robust standard errors are reported at the query level. ***p <0.01, **p <0.05, *p <0.1.

Author:Generative Search Engine52Article submitted toInformation Systems Research; manuscript no. ISRE-0000-0000.00
Appendix I: Experimental Design
I.1. Overview and Participant Flow
Figure I1 Experimental workflow diagram. Participants were randomly assigned to either original website content
(control) or LLM-polished content (treatment) conditions through deterministic assignment based on Pro-
lific ID.


Author:Generative Search Engine
Article submitted toInformation Systems Research; manuscript no. ISRE-0000-0000.0053
We develop a custom web-based generative search platform that replicates the core functionality of Google
AI Overview while keeping track of users’ activities. Figure I1 illustrates the complete experimental workflow.
Participants proceed through six sequential stages: initial task receipt, platform authentication, iterative
search interactions, policy brief composition, submission, and post-task questionnaire completion.
I.2. Initial Landing Page and Authentication
Participants are recruited from Prolific. Upon accepting the study, participants receive a direct link to our
experimental platform. The initial landing page (Figure I2) presents participants with task instructions
alongside the authentication interface. Participants are instructed to assume the role of education policy
consultants tasked with developing a policy recommendation regarding smartphone bans in K-12 educational
settings. The task requires participants to research and synthesize information to produce a policy brief with
100-300 words.
After reading the task instructions, if participants decide to take the task, we require them to enter
their unique Prolific ID for authorization. Getting the unique Prolific ID of each participant serves us dual
purposes: identity verification and deterministic treatment assignment.
Figure I2 Platform authentication and instruction interface. Participants review task requirements before entering
their Prolific ID to access the search system.
I.3. Treatment Assignment Mechanism
Following successful authentication, our system randomly assigns each participant to either the treatment
grouporthecontrolgroupthroughahash-basedassignmentalgorithm.Toensurestabilityacrosspagereloads

Author:Generative Search Engine54Article submitted toInformation Systems Research; manuscript no. ISRE-0000-0000.00
andrepeatedsessions,theservercomputestheProlific IDandmapsittocontrol,treatment.Thisapproach
ensures balanced allocation (approximately 50% per condition) while maintaining assignment stability across
sessions, as we allow our participants to search multiple times as they would like. The assignment into
treatment or control groups determines which website content pool would underlie all their subsequent
AI-generated responses. Control group participants interact with a RAG system that accesses 65 original,
unmodified website passages extracted from Google and Bing search results related to smartphone policies
in educational settings. Treatment group participants, in contrast, receive AI-generated responses based on
the same 65 website passages after LLM-based polishing with citation optimization objectives, as detailed
in Section 4.3 of the manuscript. This manipulation of the source websites represents the sole experimental
variation between our treatment and control groups. All interface elements, interaction mechanisms, and
system functionalities remain identical between treatment groups, isolating the effect of content polishing on
user behavior and outcomes.
I.4. Search Interface and AI Overview Generation
The primary experimental interface (Figure I3) consists of three integrated components: a search query field,
an AI-generated overview panel, and a response composition area. Participants can submit natural language
queries through the search field, triggering a server-side processing sequence that begins with query receipt
and verification of the participant’s treatment or control condition assignment. The system then retrieves
relevant information from the assigned source website pool, either original or polished content, depending
on the experimental condition, and generates an AI summary using Gemini’s RAG API. In addition to the
summary generation, the system also extracts and formats citations from the synthesized content, ultimately
delivering the response with inline numeric citations to the participant’s interface.
ThegeneratedAIOverviewappearsimmediatelybelowthesearchfield,presentingsynthesizedinformation
with embedded citations formatted as bracketed numbers (e.g. [1][2][5]). Each citation links to its corre-
sponding source website content, which participants could access in new browser tabs to examine original
content while maintaining their workflow in the main interface. The system utilizes AJAX requests to refresh
only the AI Overview panel during searches, preserving any text that participants have already composed
in the submission area.

Author:Generative Search Engine
Article submitted toInformation Systems Research; manuscript no. ISRE-0000-0000.0055
Figure I3 AI Overview interface displaying generated summary with inline citations and linked source documents.
I.5. Response Composition and Submission
Below the AI Overview panel, participants compose their policy briefs in a text area (Figure I4). The inter-
face enforces two constraints to ensure task engagement and quality. First, a minimum length requirement
ensures that the submission button is disabled until participants compose at least 100 words, with the inter-
face displaying real-time word count feedback. Second, we prohibit paste operations into both query and
response fields, requiring participants to actively synthesize information rather than mechanically reproduce
AI-generated content. Participants can conduct unlimited searches throughout the task, with the system
logging all queries, generated responses, citation patterns, and interaction timestamps. When participants
initiate new searches, the system preserves any text they have already composed in the response field,
enabling them to iteratively refine their policy briefs while exploring different aspects of the smartphone ban
debate without losing their work in progress.

Author:Generative Search Engine56Article submitted toInformation Systems Research; manuscript no. ISRE-0000-0000.00
Figure I4 Policy brief Composition and Submission: the system enforced a 100-word minimum while preventing
mechanical content reproduction through paste restrictions.
I.6. Data Collection and Post-Task Assessment
Upon response submission, our system captures and stores a comprehensive set of session data, including
final responses, completion timestamps, search histories, and interaction patterns. Participants then proceed
automatically to a post-task questionnaire (detailed in Appendix J) that collects information across two
primary dimensions. First, it evaluates their experimental task experience using the well-established con-
structs from technology acceptance research, the perceived ease of use and the perceived usefulness, each
measured through three validated questions. Second, the questionnaire assesses participants’ prior experience
with relevant technologies through frequency-of-use measures: their ChatGPT usage and search engine usage
(Google, Bing, or other platforms) over the preceding four weeks. Of the 150 participants initially recruited,
three failed to complete the post-task questionnaire and were excluded from analyses, yielding a final sample
of 147 participants with complete experimental data.

Author:Generative Search Engine
Article submitted toInformation Systems Research; manuscript no. ISRE-0000-0000.0057
Appendix J: Post-Task Survey Questionnaire
Please indicate your experience and perceptions when you used the provided tool for completing the assigned
tasks. The first six items were measured on a 7-point Likert scale (1 = strongly disagree, 7 = strongly agree).
Perceived ease of use
1. I find completing the tasks using the provided tool easy.
2. I feel skillful in doing the tasks by the provided tool.
3. My interaction with the provided tool is clear and understandable.
Perceived usefulness
1. Using the provided tool enabled me to accomplish the tasks quickly.
2. Using the provided tool enhanced my effectiveness in completing the tasks.
3. I find the provided tool useful.
Tool Experience
1. In the past 4 weeks, about how often did you use ChatGPT: Never in the past 4 weeks; Less than
once per week ( 1–3 times in the past 4 weeks); About once per week; 2–3 times per week; 4–6 times
per week; About once per day ( 7×per week); Several times per day.
2. In the past 4 weeks, about how often did you use search engines (Google/Bing/etc): Never in the
past 4 weeks; Less than once per week ( 1–3 times in the past 4 weeks); About once per week; 2–3
times per week; 4–6 times per week; About once per day ( 7×per week); Several times per day.
Education
1. What is your educational background: High School or Below; Some College Credit, No Degree;
Bachelor’s Degree; Master’s Degree; Doctorate Degree

Author:Generative Search Engine58Article submitted toInformation Systems Research; manuscript no. ISRE-0000-0000.00
Appendix K: Randomization Checks and Other Comparisons
Based on data collected from Prolific participants, we conduct pairwise t-tests across experimental groups
on gender (binary coded), age, and education (binary coded for college degree or higher). Given that the
majorityof participantsidentifyas White, we construct a binary ethnicity variable (White versus non-White)
and conduct a pairwise t-test on this variable as well. Additionally, since we collect data on participants’
prior experience with generative AI and search engines, which constitute categorical variables, we employ
Kolmogorov-Smirnov tests for these measures. These analyses collectively confirm that participants across
treatment and control groups are balanced, indicating successful randomization.
Beyond randomization checks, we also examine differences in participants’ perceived ease of use and
perceived usefulness measures collected through our survey. However, we find no statistically significant
differences between groups on these dimensions.
Table K1 Tests Results
Variablest/CombinedK−SStatisticsp-value
Gender1.42 0.16
Age-1.60 0.11
College0.25 0.80
White-0.27 0.79
Generative_AI_Usage0.05 1.00
Search_Engine_Usage0.09 0.93
Ease_to_Use1.12 0.27
Usefulness-1.18 0.24
Note: ***p <0.01, **p <0.05, *p <0.1.

Author:Generative Search Engine
Article submitted toInformation Systems Research; manuscript no. ISRE-0000-0000.0059
Appendix L: Vendi Score Calculation
Let a submission (document)dbe segmented intontext units (in our case, sentences)S d={s 1,...,s n}. The
Vendi scoreprovides an “effective-number” measure of how many mutually distinct items are present inS d,
based on the spectrum of a similarity kernel.
Step 1: Segment and embed.Tokenizedinto sentences and produce vector representationse i∈Rpfor
eachs iusing a sentence embedding model (SBERT). L2-normalize embeddings so∥e i∥2= 1.
Step 2: Build a PSD similarity kernel.Form then×nGram matrixKwith entries
Kij=k(e i,ej),e.g.,k(e i,ej) =e⊤
iej(cosine) ork(e i,ej) = exp(−γ∥e i−ej∥2
2)(RBF).(10)
To ensure unit diagonal (required for the interpretation below), rescale
˜K=D−1/2KD−1/2, D= diag 
K11,...,K nn
.(11)
Step 3: Normalize and take the spectrum.Let{λ i}n
i=1be the eigenvalues of ˜K/n. Because ˜Kis positive
semidefinite and diagonally normalized, theλ iare nonnegative and (up to numerical tolerance) sum to one.
Step 4: Entropy of the spectrum and the Vendi score.Define the spectral (Shannon) entropy
H(˜K) =−nX
i=1λilogλ i,(12)
wherelogis the natural logarithm. TheVendi scorefor submissiondis the corresponding Hill number
VS(d) = exp 
H(˜K)
∈[1, n].(13)
Interpretation and edge cases.If all items are identical (perfect similarity), then ˜Kis rank one,H( ˜K) =
0, andVS(d) = 1. If items are mutually orthogonal (e.g., ˜K=I n), thenλ i=1
n,H( ˜K) = logn, andVS(d) =n.
Thus,VS(d)can be read as thediversity-equivalentnumber of distinct segments ind.