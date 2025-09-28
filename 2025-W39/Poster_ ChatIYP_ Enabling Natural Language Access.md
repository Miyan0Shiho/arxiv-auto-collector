# Poster: ChatIYP: Enabling Natural Language Access to the Internet Yellow Pages Database

**Authors**: Vasilis Andritsoudis, Pavlos Sermpezis, Ilias Dimitriadis, Athena Vakali

**Published**: 2025-09-23 14:21:43

**PDF URL**: [http://arxiv.org/pdf/2509.19411v1](http://arxiv.org/pdf/2509.19411v1)

## Abstract
The Internet Yellow Pages (IYP) aggregates information from multiple sources
about Internet routing into a unified, graph-based knowledge base. However,
querying it requires knowledge of the Cypher language and the exact IYP schema,
thus limiting usability for non-experts. In this paper, we propose ChatIYP, a
domain-specific Retrieval-Augmented Generation (RAG) system that enables users
to query IYP through natural language questions. Our evaluation demonstrates
solid performance on simple queries, as well as directions for improvement, and
provides insights for selecting evaluation metrics that are better fit for IYP
querying AI agents.

## Full Text


<!-- PDF content starts -->

Poster: ChatIYP: Enabling Natural Language Access to the
Internet Yellow Pages Database
Vasilis Andritsoudis
Aristotle University of Thessaloniki
Thessaloniki, Greece
vasandven@csd.auth.grPavlos Sermpezis∗
Measurement Lab
Code for Science & Society
Portland, OR, United States
pavlos@measurementlab.net
Ilias Dimitriadis
Aristotle University of Thessaloniki
Thessaloniki, Greece
idimitriad@csd.auth.grAthena Vakali
Aristotle University of Thessaloniki
Thessaloniki, Greece
avakali@csd.auth.gr
Abstract
The Internet Yellow Pages (IYP) aggregates information from mul-
tiple sources about Internet routing into a unified, graph-based
knowledge base. However, querying it requires knowledge of the
Cypher language and the exact IYP schema, thus limiting usabil-
ity for non-experts. In this paper, we proposeChatIYP, a domain-
specific Retrieval-Augmented Generation (RAG) system that en-
ables users to query IYP through natural language questions. Our
evaluation demonstrates solid performance on simple queries, as
well as directions for improvement, and provides insights for se-
lecting evaluation metrics that are better fit for IYP querying AI
agents.
1 Introduction
Information about the Internet’s underlying infrastructure and
routing (e.g., ASes, IP prefixes) is valuable for researchers and engi-
neers, for, e.g., diagnosing routing anomalies. However, this data
is often fragmented across disparate sources (e.g., BGP routing
tables, WHOIS records). TheInternet Yellow Pages (IYP)[ 7] con-
solidates this data into a unified, graph-based knowledge base. It
models infrastructure entities (e.g., ASes, IP blocks) as nodes, with
edges capturing relationships. This rich structure supports power-
ful queries, but only through Cypher [ 3], a graph query language
that requires knowledge of graph syntax and schemas. For example,
answering a simple question like:
“What is the percentage of Japan’s population in AS2497?”
requires a Cypher query such as:
MATCH (:AS {asn:2497})-[p:POPULATION]-(:Country
{country_code:’JP’}) RETURN p.percent
This complexity creates a steep learning curve, limiting IYP’s adop-
tion beyond expert users.
To make IYP easily for everyone, in this paper, we design and
implementChatIYP, a system build on a domain-specific Retrieval-
Augmented Generation (RAG) [ 6] approach, which translates natu-
ral language questions into Cypher queries, executes them on IYP,
and returns both the lexical responses and the underlying query
for transparency.
∗Also with Aristotle University of Thessaloniki.
Figure 1: ChatIYP architecture pipeline.
We evaluate ChatIYP using multiple example queries and evalua-
tion metrics. Results show that responses are accurate in most easy
technical tasks; while there is room for improvement for harder
tasks. Moreover, we demonstrate that an evaluation framework
using LLM-as-a-judge setup (G-Eval [ 9]) better reflects human judg-
ment in query quality compared to other common metrics for RAG
evaluation.
2 ChatIYP
Figure 1 depicts the RAG-based architecture of ChatIYP, which
consists of three key stages:
(1) User Query: A user submits a natural language question through
a web interface; this is the input for our system.
(2) Retrieval: ChatIYP retrieves relevant information from the
IYP graph database using three complementary ways, combining
symbolic and semantic methods
•TextToCypherRetriever:An LLM maps the user query to a Cypher
query. We designed a prompt chain fine-tuned on IYP query
patterns. The resulting query is executed against the Neo4j graph
to return structured subgraphs.
•VectorContextRetriever:When structured queries fail or sparse
results are returned, dense embeddings for node descriptions are
used to fetch textual context of nearby graph nodes (via vector
similarity). This is particularly useful for vague queries or cases
where graph structure alone is insufficient.
•LLMReranker:Given multiple retrieval candidates from the above
steps, we re-rank results using a shallow LLM-based scorer toarXiv:2509.19411v1  [cs.NI]  23 Sep 2025

Vasilis Andritsoudis et al.
improve context selection before generation. This combination
provides robustness: when symbolic translation fails or yields
low recall, semantic retrieval ensures we still return useful infor-
mation.
(3) Generation: The input and the retrieved nodes are passed to
an LLM that generates a natural language response and a refined
Cypher query. This is the output of the system that is displayed
to the user. In our implementation and experimenation, we use
GPT-3.5-Turbo as the backbone LLM for response generation
We implement the pipeline using the LlamaIndex [ 12] frame-
work, which supports symbolic and semantic retrieval over graph-
structured data and provides built-in integration with Neo4j [4].
3 Results
We evaluate ChatIYP using theCypherEval[ 10] dataset, a bench-
mark of more than 300 natural language questions over IYP. Each
question is annotated with a gold Cypher query and labeled by
difficulty:Easy,Medium, orHard, across bothgeneralandtechnical
domains.
Evaluation Setup.To assess response quality, we use a validation
model that executes the gold Cypher query on the IYP graph and
prompts GPT-3.5 to produce a reference answer. ChatIYP’s out-
put, is compared against these references using widespread text
generation metrics. Figure 2a shows the results of this evaluation.
Evaluation Metrics. We apply several widely used metrics in
natural language generation to quantify answer quality:
•BLEU[ 5] evaluates n-gram precision between the model’s re-
sponse and the reference.
•ROUGE[11] focuses on n-gram and subsequence recall.
•BERTScore[ 8] measures similarity using contextual embeddings
from a pre-trained language model (e.g., BERT).
•G-Eval[ 9] uses GPT-4 as a judge to assess responses on factuality,
relevance, and informativeness.
Finding 1: G-Eval outperforms traditional metrics.Standard
metrics like BLEU, ROUGE, or even BERTScore underperform in
this setting, primarily due to their reliance on surface-level token
overlap. These metrics struggle with rephrased or factually incor-
rect answers. We observe that: (i) BLEU scores are overly penalized
by minor phrasing mismatches, despite semantic correctness. (ii)
ROUGE better accommodate reworded answers, but still correlate
poorly with factual accuracy. (iii) BERTScore exhibits a ceiling ef-
fect that blurs performance distinctions, especially in responses
with narrow linguistic variation (as common in IYP queries).
On the contrary, G-Eval has a bimodal score distribution, which
provides clear separation between good and bad responses, aligning
closely with human judgment. We therefore adopt G-Eval as our
primary evaluation metric.
Finding 2: ChatIYP handles simple prompts well. Figure 2b
presents G-Eval scores across difficulty categories. ChatIYP per-
forms well on easy prompts, with over half of responses scoring
above 75%. Performance degrades with prompt complexity, particu-
larly on hard questions involving multi-hop reasoning. Interestingly,
no consistent performance gap emerges between general and tech-
nical prompts, suggesting thatstructural complexity, not domain
specificity, poses the greatest challenge.
(a)
 (b)
Figure 2: (a) Comparison of metric distributions. (b) G-Eval
scores by difficulty.
4 Conclusion
ChatIYP introduces an approach for a retrieval-augmented natural
language interface for the IYP graph, thus simplifying access to
complex network data. Initial evaulation shows strong performance
on simple queries and underscores challenges with complex ones
which opens the door for further future research. Additionally, the
G-Eval metric better reflects human judgments than traditional
metrics like BLEU or ROUGE. We provide a publicly accessible web
application for ChatIYP at [ 2] and the full source code (including
all evaluation results used in this paper) at [1].
References
[1]2025. ChatIYP - Github Repository. https://github.com/VasilisAndritsoudis/
chatiyp.
[2] 2025. ChatIYP Web Application. https://chatiyp.csd.auth.gr.
[3] 2025.Cypher Query Language. https://neo4j.com/developer/cypher
[4] 2025.Neo4j Graph Database Platform. https://neo4j.com
[5]Kishore Papineni et al. 2002. Bleu: a method for automatic evaluation of ma-
chine translation. InProceedings of the 40th annual meeting of the Association for
Computational Linguistics.
[6]Patrick Lewis et al. 2020. Retrieval-augmented generation for knowledge-
intensive nlp tasks.Advances in neural information processing systems33 (2020).
[7] Romain Fontugne et al. 2024. The wisdom of the measurement crowd: Building
the internet yellow pages a knowledge graph for the internet. InProceedings of
the 2024 ACM on Internet Measurement Conference.
[8] Tianyi Zhang et al. 2019. Bertscore: Evaluating text generation with bert.arXiv
preprint arXiv:1904.09675(2019).
[9]Yang Liu et al. 2023. G-eval: NLG evaluation using gpt-4 with better human
alignment.arXiv preprint arXiv:2303.16634(2023).
[10] Dimitrios Giakatos, Malte Tashiro, and Romain Fontugne. 2025. Pythia: Facil-
itating Access to Internet Data Using LLMs and IYP. InIEEE Local Computer
Networks (LCN). https://codeberg.org/dimitrios/CypherEval
[11] Chin-Yew Lin. 2004. Rouge: A package for automatic evaluation of summaries.
InText summarization branches out. 74–81.
[12] LlamaIndex. 2025.LlamaIndex. https://www.llamaindex.ai/