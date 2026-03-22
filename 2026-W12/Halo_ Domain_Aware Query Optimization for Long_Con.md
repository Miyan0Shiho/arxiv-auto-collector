# Halo: Domain-Aware Query Optimization for Long-Context Question Answering

**Authors**: Pramod Chunduri, Francisco Romero, Ali Payani, Kexin Rong, Joy Arulraj

**Published**: 2026-03-18 12:34:02

**PDF URL**: [https://arxiv.org/pdf/2603.17668v1](https://arxiv.org/pdf/2603.17668v1)

## Abstract
Long-context question answering (QA) over lengthy documents is critical for applications such as financial analysis, legal review, and scientific research. Current approaches, such as processing entire documents via a single LLM call or retrieving relevant chunks via RAG have two drawbacks: First, as context size increases, response quality can degrade, impacting accuracy. Second, iteratively processing hundreds of input documents can incur prohibitively high costs in API calls. To improve response quality and reduce the number of iterations needed to get the desired response, users tend to add domain knowledge to their prompts. However, existing systems fail to systematically capture and use this knowledge to guide query processing. Domain knowledge is treated as prompt tokens alongside the document: the LLM may or may not follow it, there is no reduction in computational cost, and when outputs are incorrect, users must manually iterate.
  We present Halo, a long-context QA framework that automatically extracts domain knowledge from user prompts and applies it as executable operators across a multi-stage query execution pipeline. Halo identifies three common forms of domain knowledge - where in the document to look, what content to ignore, and how to verify the answer - and applies each at the pipeline stage where it is most effective: pruning the document before chunk selection, filtering irrelevant chunks before inference, and ranking candidate responses after generation. To handle imprecise or invalid domain knowledge, Halo includes a fallback mechanism that detects low-quality operators at runtime and selectively disables them. Our evaluation across finance, literature, and scientific datasets shows that Halo achieves up to 13% higher accuracy and 4.8x lower cost compared to baselines, and enables a lightweight open-source model to approach frontier LLM accuracy at 78x lower cost.

## Full Text


<!-- PDF content starts -->

Halo: Domain-Aware Query Optimization for Long-Context
Question Answering
Pramod Chunduri∗
Amazon Web Services
cnpramod@amazon.comFrancisco Romero
Georgia Institute of Technology
faromero@gatech.eduAli Payani
Cisco
apayani@cisco.com
Kexin Rong
Georgia Institute of Technology
krong@gatech.eduJoy Arulraj
Georgia Institute of Technology
jarulraj@gatech.edu
Abstract
Long-context question answering (QA) over lengthy documents
is critical for applications such as financial analysis, legal review,
and scientific research. Current approaches, such as processing en-
tire documents via a single LLM call or retrieving relevant chunks
via RAG have two drawbacks: First, as context size increases, re-
sponse quality can degrade, impacting accuracy. Second, iteratively
processing hundreds of input documents can incur prohibitively
high costs in API calls. To improve response quality and reduce
the number of iterations needed to get the desired response, users
tend to add domain knowledge to their prompts. However, existing
systems fail to systematically capture and use this knowledge to
guide query processing. Domain knowledge is treated as prompt
tokens alongside the document: the LLM may or may not follow it,
there is no reduction in computational cost, and when outputs are
incorrect, users must manually iterate.
We present Halo, a long-context QA framework that automati-
cally extracts domain knowledge from user prompts and applies
it as executable operators across a multi-stage query execution
pipeline. Halo identifies three common forms of domain knowledge
– where in the document to look, what content to ignore, and how
to verify the answer – and applies each at the pipeline stage where
it is most effective: pruning the document before chunk selection,
filtering irrelevant chunks before inference, and ranking candidate
responses after generation. To handle imprecise or invalid domain
knowledge, Halo uses a fallback mechanism that detects low-quality
operators at runtime and selectively disables them. Our evaluation
across finance, literature, and scientific datasets shows that Halo
achieves up to 13% higher accuracy and 4.8 ×lower cost compared to
baselines, and enables a lightweight open-source model to approach
frontier LLM accuracy at 78×lower cost.
PVLDB Reference Format:
Pramod Chunduri, Francisco Romero, Ali Payani, Kexin Rong, and Joy
Arulraj.Halo: Domain-Aware Query Optimization for Long-Context
Question Answering. PVLDB, 14(1): XXX-XXX, 2020.
doi:XX.XX/XXX.XX
∗Work done prior to joining Amazon Web Services.
This work is licensed under the Creative Commons BY-NC-ND 4.0 International
License. Visit https://creativecommons.org/licenses/by-nc-nd/4.0/ to view a copy of
this license. For any use beyond those covered by this license, obtain permission by
emailing info@vldb.org. Copyright is held by the owner/author(s). Publication rights
licensed to the VLDB Endowment.
Proceedings of the VLDB Endowment, Vol. 14, No. 1 ISSN 2150-8097.
doi:XX.XX/XXX.XX1 Introduction
Long-Context Question Answering (QA), an important task in doc-
ument understanding, aims to answer fact-based questions over
lengthy documents. Real-world applications, such as financial ana-
lysts querying 10-K filings, legal professionals reviewing contracts,
and researchers surveying scientific literature, regularly demand
accurate answers from documents spanning hundreds of thousands
of tokens [ 35,41]. Recent advances in large language models (LLMs)
have significantly expanded theircontext windows, with some mod-
els such as Gemini 3 Pro and Sonnet 4.5 supporting a million to-
kens [ 2,16]. However, efficiently leveraging these contexts for QA
remains challenging.
Currently, there are two standard approaches for tackling long-
context QA, yet neither achieves a satisfactory accuracy-cost trade-
off. The first feeds the entire document to a state-of-the-art (SoTA)
LLM like Sonnet 4.5 in a single call, which we refer to as VanillaLLM .
However, this approach is both expensive and unreliable: process-
ing hundreds of long documents can cost over $500 in API calls
(§7.2) and even SoTA LLMs exhibit significant hallucination over
long contexts due to issues like“lost in the middle”[ 29,30]. The
second approach is Retrieval-Augmented Generation (RAG), which
attempts to reduce cost by selecting only the most relevant docu-
ment chunks before inference [ 26]. Although RAG reduces token
usage compared to VanillaLLM , it introduces top-K selection er-
rors that could hurt accuracy: key evidence may be missed during
the similarity search, and fragmented or reordered chunks can in-
troduce incoherent context. As a result, RAG still hallucinates or
overlooks important nuances in the provided context [18, 25].
To improve the accuracy-cost tradeoff, domain experts natu-
rally turn to domain knowledge. However, the primary mechanism
available to them is prompt engineering, where such knowledge
is embedded directly into the model’s prompt input, as shown in
Figure 1. For example, a financial analyst may instruct the LLM
to focus only on tables or ignore boilerplate disclosures when pro-
cessing SEC 10-K filings. While such instructions can sometimes
improve answer quality, they remain embedded as free-form text
within the prompt. The model may inconsistently follow the in-
structions, especially in long contexts where instruction-following
degrades [ 17,44]. As seen in Figure 1, VanillaLLM fabricates in-
correct values even when domain knowledge is provided in the
prompt as the instructions are lost in the 200K-token context. RAG
retrieves relevant context but hallucinates the final response, re-
porting basic instead of diluted computations, due to the incoherent
top-K chunks. In both cases, the domain knowledge yields no costarXiv:2603.17668v1  [cs.DB]  18 Mar 2026

RAGVanillaLLMContext WindowQuestion: What was the change in diluted computations from 2021 to 2022?Answer the question using the given context. Focus only on tables. Ignore legal disclosures. Focus only on diluted computations, not basic computations.
...
Full Document
Top-K chunks
Context Window
$256-$185 = $71 mil$123-$89= $34 milHALOUser Question w/Domain Knowledge AccuracyCost
Domain-Aware Query Optimization$82-$68= $14 mil
LowHighMedMedHighLow(fabricated)
(basic computations)Long-Context DocumentGold ContextBasic comp.Diluted comp.20212022123898268Earnings per share (EPS) tableFigure 1: VanillaLLM andRAG cannot utilize user’s domain
knowledge effectively for long-context QA. Halo extracts
this knowledge and systematically applies it in a multi-stage
query optimization pipeline.
reduction. Moreover, when such errors occur, users must iteratively
refine prompts through trial and error. The core limitation is that
domain knowledge is expressed implicitly as tokens rather than
captured explicitly as structured guidance that shapes how the sys-
tem processes the query, limiting improvements in both accuracy
and cost.
Our Operator-Based Approach.To address this limitation,
we introduce Halo , a long-context QA framework that extracts do-
main knowledge from free-form prompt text into explicitdirectives
that directly shape execution plans. Our key observation is that
domain knowledge should not be consumed as prompt tokens for
the LLM to implicitly interpret and instead be extracted into struc-
tured directives that are applied at the pipeline stage where each
is most effective. Halo identifies three common forms of domain
knowledge from users’ natural language prompts:
•Structural —where to look: structural elements likely to con-
tain the answer (e.g., tables, MD&A sections)
•Filter —what to ignore: semantic content irrelevant to the
query (e.g., legal disclosures, boilerplate text)
•Validate —how to verify: domain-specific constraints to de-
mote hallucinated responses (e.g., “NOT basic computations”)
TheKnowledgeParser first extracts structured directives from the
user’s unstructured prompt. Halo then leveragesoperatorsto apply
these directives across a multi-stage pipeline (Figure 2), with each
operator designed around a specific insight about where domain
knowledge has the highest impact. The ContextSelector stage
applies the Structural operator to prune the document to only
the user-specified structural elementsbeforechunking. By carefully
curating the expensive LLM’s context, Halo ensures only rele-
vant context is passed downstream to the other operators. The
InferenceEngine stage then applies the Filter operator via a
lightweight SLM to discard irrelevant chunksbeforethe expensive
LLM processes them. It uses a model cascade where a cheap model
quickly eliminates noise so the expensive model only sees what
matters. Finally, the Verifier stage applies the Validate operator
to rank candidate responses. We observe that positive and nega-
tive validation directives (e.g., “diluted EPS” vs. “basic EPS”) canbe contrasted against each other: responses aligned with positive
directives are promoted while those matching negative directives
are demoted, enabling domain-aware hallucination detection with-
out any additional LLM calls. To prevent accuracy degradation
from low-quality operators, Halo ’s fallback manager detects when
an operator hurts performance using model- and operator-based
signals, and disables it to recover accuracy at minimal overhead.
Our evaluation across finance, literature, and scientific datasets
shows that Halo achieves up to 13% higher accuracy and 4.8 ×lower
cost compared to baselines. Notably, Halo enables a lightweight
open-source model to approach the accuracy of a frontier LLM at
78×lower cost, while the domain operators themselves add only
2% overhead to the total query cost.
Contributions.We make the following key contributions:
•We show the domain knowledge users express in their prompts
can be automatically extracted into structured directives and
design corresponding operators ( Structural ,Filter ,Validate )
that each apply the directives (§4, §5).
•We propose Halo , a multi-stage long-context QA framework
that composes these operators to simultaneously reduce cost
and improve accuracy–pruning document structure early, fil-
tering chunks via lightweight models, and ranking responses
using domain constraints (§3, §5).
•We develop a fallback mechanism that detects when operators
degrade accuracy at runtime and selectively disables them,
ensuring Halo effectively handles low quality directives (§6).
•We evaluate Halo across diverse domains and datasets, show-
ing up to 13% higher accuracy and 4.8 ×lower cost over base-
lines, and demonstrating that a lightweight open-source model
with Halo approaches frontier LLM accuracy at 78 ×lower
cost (§7).
2 Motivating Example
To illustrate the value of domain knowledge for long-context QA
and the challenges of applying it systematically, consider the fol-
lowing example. A junior financial analyst is tasked with extracting
specific numerical data from hundreds of annual SEC 10-K filings.
Each filing is roughly 200K tokens, containing financial tables,
narrative discussions, legal disclosures, and boilerplate text. The
analyst wishes to answer questions like:
“What was the change in diluted computations from 2021 to 2022?”
Answering this question requires locating the correct earnings-per-
share table, extracting the “diluted” (not “basic”) computations for
the correct years, and computing the difference. The analyst must
do this across hundreds of documents, making manual inspection
infeasible at scale.
Baseline Approaches.We first consider two standard approaches
that do not leverage domain knowledge.
Vanilla LLM.A state-of-the-art long-context LLM such as Claude
Sonnet 4.5 can process the entire 200K-token filing in a single call.
However, this approach is both expensive and unreliable. Process-
ing hundreds of documents can cost over $500 in API calls (§7.2).
Furthermore, the LLM is prone to hallucinate over such long con-
texts [ 29,30]. For the query above, a SoTA LLM fabricates the values
for 2021-2022, simply because it could not find the relevant context
from the long document.
2

2) ContextSelectorSTRUCTURAL OPERATOR3) InferenceEngine4) VerifierQuestion: What was the change in diluted computations from 2021 to 2022?Answer the question using the given context. Focus only on tables. Ignore legal disclosures. Focus only on diluted computations, not basic computations.
Extract only ”tables”Retain Top-K relevant chunksFILTEROPERATOR
Chunk-level responsesVALIDATEOPERATORDemote “basic computations” in the rankingAnswer not in contextThe change in basic comp …The change in diluted comp …
Parallel LLM CallsUse SLM to remove “risk disclousures”The change in basic comp...The change in diluted comp ...
Validation ScoresSmall Language ModelRANKED RESPONSES
Detect low quality operators and trigger fallback HALO generates the correct response with fewer tokens Block 1Block 2Block 3......Block NChunk the extracted blocksFallback: Re-run query without operators
Tag Matching1) KnowledgeParser
STRUCTURAL -> [“Table”]VALIDATE -> [“NOT basic computations”]FILTER -> [“risk disclosures”]DOMAIN DIRECTIVES (Inputs to the operators)Question: What was the change in diluted computations from 2021 to 2022?
Pre-processed Input DocumentAnswer the question using the given context. Focus only on tables. Ignore legal disclosures. Focus only on diluted computations, not basic computations.User Prompt w/ Domain KnowledgeFigure 2: Workflow of Halo : (1) the KnowledgeParser extracts domain directives from the user prompt, (2) the
ContextSelector prunes structural elements using the Structural operator and identifies the top- 𝑘relevant chunks,
(3) the InferenceEngine applies the Filter operator to discard irrelevant chunks and generates candidate responses on the
remaining chunks, (4) theVerifierre-ranks the responses usingValidateoperator–demoting hallucinatory responses.
RAG.To avoid processing the entire document, a standard RAG
pipeline [ 26] retrieves the most relevant chunks and provides them
as context to the LLM. While this reduces cost, it also introduces
retrieval errors: for the query above, RAG ranks a chunk containing
basiccomputations higher than the one withdilutedcomputations,
and the LLM confidently answers based on the wrong data.
In both approaches, the analyst’s domain knowledge plays no
role in the processing pipeline. Furthermore, when outputs are
incorrect, both approaches require iterative prompt refinement—
compounding cost and effort across hundreds of documents.
Benefits of Domain Knowledge.To improve the LLM’s re-
sponse quality, analysts can use domain knowledge to optimize
their queries:
•Structural knowledge: where to look.The analyst knows that
earnings-per-share data is always in a table, never in narrative
text. Pruning non-tabular content reduces the search space
from 200K to 50K tokens.
•Filtering knowledge: what to ignore.The analyst also knows
that large portions of the filing, such as legal disclosures, risk
factors, and forward-looking statements, are never relevant
to earnings queries. For example, removing boilerplate legal
tables from the 50K tokens of tabular content could reduce the
context to approximately 30K tokens, eliminating noise that
might otherwise confuse the model.
•Validation knowledge: how to verify.The analyst knows that the
earnings-per-share table contains both “basic” and “diluted”
computations, and the LLM frequently confuses them. If the
processing pipeline could use this knowledge to verify that a
response references “diluted” rather than “basic” computations,
accuracy could be improved.
As shown in Table 1, each type of domain knowledge provides
a distinct benefit. Structural knowledge narrows the search space
by pruning irrelevant document regions, improving the qualityTable 1: Potential benefits of domain knowledge for the exam-
ple financial QA query. Each row adds one type of knowledge
on top of the previous.
Approach Tokens Result Knowledge Used
Vanilla LLM∼200K✗(wrong years) None
RAG (top-K)∼100K✗(basic, not diluted) None
+ Structural knowledge∼50K✗(basic, not diluted) Where to look
+ Filtering knowledge∼30K✗(basic, not diluted) + What to ignore
+ Validation knowledge∼30K✓(correct) + How to verify
of retrieved chunks. Filtering knowledge further reduces cost by
discarding irrelevant chunks before passing them to the expensive
LLM. Validation knowledge improves accuracy by detecting and
mitigating hallucinations.
Applying Domain Knowledge Today.Despite its clear value,
applying domain knowledge to long-context QA today is a manual,
tedious, and unreliable process.
Prompt engineering.The most common approach is to embed do-
main knowledge directly into the LLM prompt (Figure 1). The LLM
receives these instructions as input tokens alongside the document.
It may or may not follow them: the model might focus on tables
but still processes the entire document, offering no cost reduction.
Whether the model ignores legal disclosures or avoids “basic com-
putations” is unknown to the analyst until they inspect the output.
Manual iteration.When the output is incorrect, the analyst must
inspect the response, diagnose the error, refine the prompt, and
re-run the query. This iteration loop is expensive: each attempt
processes the full document, and the analyst may need multiple
rounds to converge on a working prompt, costing 3-5 ×more than
the already expensiveVanillaLLM.
Manual pipeline management.To realize the cost benefits shown
in Table 1, the analyst would need to go beyond prompt engineering
and manually build a processing pipeline. Systems like LOTUS [ 33]
3

# Step 1: Filter with cheap model
lotus.settings.configure(lm=haiku)
filtered_df = df.sem filter("{chunk_text} contains EPS data")
# Step 2: Extract with expensive model
lotus.settings.configure(lm=sonnet)
result_df = filtered_df.sem extract(["chunk_text"],
{"eps_2024": "diluted EPS for 2024",
"eps_2025": "diluted EPS for 2025"})Figure 3: LOTUS program for extracting EPS data from finan-
cial filings. The user manually specifies the filter predicate
and extraction schema in Python.
provide semantic operators for LLM-based data processing, but
require users to build LLM pipelines that manually specify filter
predicates and extraction schemas (Figure 3). For our financial QA
example, the analyst must determine the correct filter keywords,
define the output schema, specify the models to use, and perform
the final computation manually—a cumbersome burden for domain
experts who do not specialize in building LLM pipelines.
Key observations.From the financial analyst example, we iden-
tify two key takeaways: (1) domain knowledge is valuable for long-
context QA, providing improvements in both cost and accuracy if
used correctly, and (2) applying this knowledge today is manual
and expensive. To address this, we need toautomatically capture
the domain knowledge users already express in their prompts, and
systematically apply it during query optimization and processing.
3 System Overview
We now describe Halo : a framework for long-context question an-
swering that automatically extracts domain knowledge from user
prompts and systematically applies it to optimize query process-
ing. Unlike existing approaches that treat domain knowledge as
prompt tokens for the LLM to implicitly interpret, Halo extracts
this knowledge into structureddirectivesand applies them via ded-
icatedoperatorsacross a multi-stage pipeline. Figure 2 illustrates
the end-to-end workflow.
3.1 End-to-end workflow
Halo takes the same input a user would provide to any LLM: a doc-
ument and a prompt containing the question along with domain
knowledge for guiding the response. Our experiments across fi-
nance, literature, and scientific domains (§7) identify three common
forms of domain knowledge: (1)structuralknowledge of where in
the document the answer is likely found (e.g., a table), (2)filtering
knowledge of what content to ignore (e.g., legal disclosures), and (3)
validationknowledge for detecting hallucinations (e.g., confusable
values). Halo extracts these into three directive types— S,F, and
V—and applies them via the Structural ,Filter , and Validate
operators, respectively (§4, §5). The framework is extensible to
additional operator types (§5.5).
As shown in Figure 2, Halo consists of five stages: a one-time
preprocessing stage and four query-time stages:
(0) Input Preprocessing.This is a one-time stage that parses the
PDF document into a structure-aware representation ˜𝐷using a
multimodal parsing utility [ 14], as shown in Figure 4. The extracted
structural elements (e.g., tables, sections, figures) are available for
all subsequent queries over the document. While LLM-based PDF
Original PDF...
Candidate Elements[ID: 1] section MD&A[ID: 2] Table: Earnings per Share (Parent: MD&A)...Multimodal Doc Parser(e.g. Gemini)Parsed text...<section MD&A>    Item 7...    <table Earnings per Share>    |Table content|    </table>    ...</section>...Figure 4: Structure-aware document preprocessing
parsing is expensive, it is highly accurate [ 14] and the cost is amor-
tized across multiple queries ( §7.5.1).
(1)KnowledgeParser .This stage uses an LLM to extract do-
main directives from the user’s prompt: 𝜋(𝑑)→(S,F,V) . This
step processes only the instruction portion of the prompt, incur-
ring negligible cost compared to the full document (§7.5.1). The
extraction is analogous to text-to-SQL parsing with a constrained di-
rective taxonomy [ 15]. Once extracted, the directives are consumed
by operators across the remaining pipeline stages.
(2)ContextSelector .This stage applies the Structural op-
erator using the structural directives Sto prune ˜𝐷to only the
user-specified structural elements (e.g., tables). This produces a
pruned document, 𝐷𝑠.𝐷𝑠is split into chunks and ranked by embed-
ding similarity to the query; the top- 𝑘are selected for downstream
processing. By narrowing the search space, Halo improves the
relevance of selected chunks, leading to higher-quality context for
downstream inference (§5.2).
(3)InferenceEngine .This stage applies the Filter operator
using the filter directives Fto prune irrelevant chunks from the
top-𝑘candidates via a lightweight SLM. The retained chunks are
then processed in parallel by the expensive LLM to generate chunk-
level candidate responses (§5.3).
(4)Verifier .This stage applies the Validate operator using the
validation directives Vto rank candidate responses. Each response
is scored by its alignment with positive constraints V+(what the
answer should contain) and negative constraints V−(what it should
not). The top-𝑘ranked responses are returned to the user (§5.4).
To handle imprecise directives, Halo includes a fallback man-
ager that detects when operators degrade accuracy—using signals
such as universal answer refusal (for Structural /Filter ) or low
score variance (for Validate )—and selectively disables problematic
operators to recover baseline performance (§6).
3.2 Example Workflow
We illustrate Halo ’s workflow using the financial analyst exam-
ple from §2. The analyst queries a 200K-token SEC 10-K filing:
“What was the change indiluted computationsfrom 2021 to 2022?”
The prompt includes domain-specific instructions such as “focus
on tables,” “ignore legal disclaimers,” and “report diluted compu-
tations, NOT basic.” (Figure 2). The KnowledgeParser extracts
these into directives: 𝜎1=“table” ,𝜙1=“legal disclaimers” ,
𝛿1=“diluted computations”,𝛿 2=“NOT basic computations”.
TheContextSelector applies the Structural operator with
S={𝜎 1}to extract only tabular content, reducing the document
4

from 200K to approximately 50K tokens. The InferenceEngine ap-
plies the Filter operator withF={𝜙 1}to remove boilerplate legal
tables, further reducing the candidate set. The retained chunks are
processed via parallel LLM calls to generate candidate responses. Fi-
nally, the Verifier applies the Validate operator withV+={𝛿 1}
andV−={𝛿 2}to rank responses, penalizing those mentioning
“basic computations” and prioritizing those with “diluted computa-
tions.” This workflow reduces the tokens processed by the expensive
LLM while improving accuracy by ensuring the response satisfies
domain-specific constraints.
4 Extracting Domain Knowledge
Today, users often express domain knowledge as plain-text instruc-
tions embedded in prompts for LLM-based question-answer work-
flows. These instructions are treated as ordinary input tokens—they
do not deterministically affect query planning or execution.
Halo instead parses natural-language domain instructions into
structureddirectivesthat directly influence query optimization and
execution. Given a question 𝑞and domain knowledge 𝑑,Halo uses
aKnowledgeParserto extract𝜋(𝑑)→(S,F,V)where:
•S={𝜎 1,...,𝜎 𝑛}is a set of structural directives
• F={𝜙 1,...,𝜙 𝑚}is a set of filter directives
•V={𝛿 1,...,𝛿 𝑘}is a set of validation directives
Any of these sets may be empty if the corresponding knowledge
is not present in 𝑑. Each directive is a declarative specification–it
describeswhatknowledge the user expressed, nothowit is enforced.
Operators, described in Section 5, implementhowthese directives
are applied at different stages of execution. We now describe each
domain knowledge type theKnowledgeParserextracts.
4.1 Where to focus - Structural directives (S)
Domain experts possess insights about the document structure
and where to look in the document when answering questions.
For example, a financial analyst reviewing company earnings may
know the earnings per share is located in a table of the SEC 10-K
filing. However, they may not know the exact table. A scientific
researcher who is comparing multiple baselines knows to survey
the evaluation section of the paper.
Through their prompts, users provide structural knowledge that
may include: (1) structural element types (e.g., tables, text, graphs,
and figures), and (2) section and table names (e.g., “MD&A section, ”
“Evaluation section,” “EPS table”). The KnowledgeParser extracts
a set of structural elements S=(𝜎 1,...,𝜎 𝑛)from𝑑, where each
𝜎𝑖is an element from the pre-determined document structure. For
example, from the instruction “the answer is likely in a table,” the
KnowledgeParserextracts𝜎 1=“table”.
Halo supports structural knowledge of arbitrary granularity,
such as ‘look in the MD&A section’ or ‘the answer is in the EPS
table row 4’. The extracted structural elements and how they are
utilized by Halo to optimize query execution are discussed in §5.2.
4.2 What to ignore - Filter directives (F)
Prompts often include domain knowledge about where not to look
or what to skip. For example, a financial analyst may provide in-
structions to skip over boilerplate content like ‘risk disclosures’ infinancial documents. A paralegal may prompt the LLM to ignore
‘waivers’ in service-level agreements (SLA).
While structural directives focus on document structure, filter di-
rectives focus ondocument content—“risk disclosures” and “waivers”
are semantic concepts that span arbitrary structural elements. For
instance, “risk disclosures” appear in multiple sections of the SEC
10-K document, containing multiple structural types such as tables,
text paragraphs, and footnotes.
TheKnowledgeParser extracts filter directives F=(𝜙 1,...,𝜙 𝑛)
from𝑑, where each 𝜙𝑖is a natural language description of the con-
tent to exclude. For example, for the instruction “ignore general
risk disclosures and legal disclaimers that are not relevant to the
current query,” the filter criteria are extracted as:
𝜙1=“risk disclosures”,𝜙 2=“legal disclaimers”
These extracted filter directives are applied during Halo ’s query
execution phase to exclude matching content (discussed in § 5.3).
4.3 How to verify - Validation directives (V)
To avoid hallucinations, users commonly augment their prompts
with instructions on how to disambiguate requested information
from extraneous content and what errors to avoid when answering
the question. Users may provide disambiguation knowledge as:
•Emphasis constraints: “only return numerical values,” “do not
include commentary”
•Disambiguation: “diluted EPS, NOT basic EPS,” “net revenue,
NOT gross”
TheKnowledgeParser extracts such knowledge into a set of
validation directives V=(𝛿 1,...,𝛿 𝑛)where each𝛿𝑖is an emphasis
constraint or disambiguation criterion. For example, the instruction
“return diluted EPS only, NOT basic EPS” is extracted into:
𝛿1=“diluted EPS”,𝛿 2=“NOT basic EPS”
The KnowledgeParser identifies negative constraints syntacti-
cally (e.g., via “NOT”, “avoid”, “do not include”) and partitions V
into positive directives V+(what the responseshouldcontain) and
negative directives V−(what it shouldnotcontain). For example,
𝛿1=“diluted EPS”∈V+and𝛿2=“basic EPS”∈V−. We discuss
how the validation directives are used to detect and mitigate model
hallucinations in § 5.4.
5 Applying Domain Knowledge
Once the KnowledgeParser extracts the domain knowledge into
the directives, these directives must be applied to the query execu-
tion pipeline to improve query processing efficiency and accuracy.
In this section, we describe how the directives discussed in §4 are
applied to enable long-context query optimization.
5.1 Directive-driven Query Execution
Existing QA frameworks treat domain instructions 𝑑as part of the
model input. In a vanilla LLM pipeline,
𝐿𝐿𝑀(𝑞,𝐷,𝑑)→𝑎,
where the query 𝑞, input document 𝐷and natural-language domain
instructions𝑑are concatenated into a single prompt.
In retrieval-augmented generation (RAG), the input 𝐷is split
into chunksC={𝑐 1,𝑐2,...,𝑐 𝑛}. At query time, a retrieval function
5

Table 2: From domain knowledge to execution: directives
specifywhatthe user expressed; operators definehowto
apply it.
Knowledge Directive Operator Effect on Pipeline
Where to lookS={𝜎 1,...,𝜎 𝑛}Structural( ˜𝐷,S)→𝐷 𝑠Prunes document structure
What to ignoreF={𝜙 1,...,𝜙 𝑚}Filter(C,F,𝑞)→C∗Discards irrelevant chunks
How to verifyV=V+∪V−Validate(R,V)→𝐴∗Ranks candidate responses
𝑅ranks chunks by embedding similarity and selects the top-𝑘:
𝑅(𝑞,C,𝑘)→C 𝑘whereC 𝑘⊆C,|C 𝑘|=𝑘
The retrieved chunks are concatenated and sent to the model:
𝑅𝐴𝐺(𝑞,C 𝑘,𝑑)→𝑎
Key Insight.In both approaches, domain instructions 𝑑are treated
as part of the prompt — they are not explicitly used to curate context
or detect hallucinations. Users embed valuable domain knowledge
in𝑑(e.g., “look in tables,” “ignore legal disclaimers”), but this knowl-
edge is not leveraged for query optimization.
Halo Execution. Halo extracts domain directives from 𝑑us-
ing the KnowledgeParser :𝜋(𝑑)→(S,F,V) and applies them
through dedicated operators during execution. Halo introduces
three operators— Structural (structural pruning), Filter (chunk
filtering), and Validate (response ranking)—each designed to apply
a distinct class of directives at the appropriate stage of the pipeline.
Table 2 summarizes the mapping from domain knowledge to direc-
tives and their enforcing operators. Halo ’s full execution pipeline
is as follows:
𝐷𝑠=Structural( ˜𝐷,S)
C𝑘=Sel(𝐷 𝑠, 𝑞)
C∗=Filter(C 𝑘,F, 𝑞)
R=Gen(C∗, 𝑞)
Halo(𝑞,𝐷,𝑑)=Validate(R,V)
where ˜𝐷is structure-aware representation of the input document
generated in the preprocessing stage (§ 3), Selranks chunks of
𝐷𝑠by embedding similarity to 𝑞and selects the top- 𝑘, and Gen
produces a candidate response per chunk via separate LLM calls. In
the subsequent sections, we describe each operator and its impact
on the query execution cost and accuracy.
5.2 Optimizing Context Selection
The structural knowledge provided by the user is extracted as struc-
tural directivesS=(𝜎 1,..,𝜎 𝑛)by the KnowledgeParser (§4.1).
The ContextSelector applies the Structural operator, using
these directives to prune ˜𝐷during query execution.
Document Pruning.The Structural operator takes ˜𝐷and the
structural directives (𝜎1,...,𝜎 𝑛)as input to produce the pruned
document𝐷𝑠. Each directive 𝜎𝑖independently selects matching
structural elements from ˜𝐷, and the operator returns their union:
Structural( ˜𝐷,S)={𝑒∈ ˜𝐷|∃𝜎 𝑖∈S:match(𝑒,𝜎 𝑖)}→𝐷 𝑠
TheKnowledgeParser extracts these directives from the user’s
prompt as natural language descriptions (e.g., 𝜎𝑖=“EPS table”). The
Structural operator maps each directive to the concrete structural
elements available in ˜𝐷using embedding-based similarity matching.
Directives with no match are skipped. If none of the directives match
PromptConsider the following context. Your task is to determine whether the context is solely composed of any of the following types of information: {{filter directives}}.Output 1 if yes and 0 if no.Chunk Content
Small Language ModelBinary Classification LabelExtracted Filter Directives𝜙1 = “legal proceedings”, 𝜙1 = “risk disclosures”
Figure 5: ApplyingFilterOperator
any element in ˜𝐷, the operator conservatively falls back to the full
document (𝐷𝑠=𝐷) to avoid discarding the answer due to imprecise
directives (discussed further in §6).
The structural pruning ismodel-free,i.e., the Structural op-
erator directly maps each directive to the available tags in ˜𝐷and
retains only the content within matched tags. The context reduc-
tion from the Structural operator depends on the specificity of
the user directives and the document structure. In our experiments
on the SEC 10-K dataset (§7.1), broad directives like “the answer
is in tables” reduced the document from ∼200K to∼50K tokens
on average (75% reduction). More specific directives like “EPS ta-
ble” achieved up to 98% reduction when matched, but yielded no
reduction if no tags match.
5.3 Optimizing Inference
After applying the Structural operator, the document 𝐷𝑠retains
only structurally relevant regions but may still contain semantically
irrelevant content. For example, in a financial document already fil-
tered to tables, irrelevant tables such as the table of contents remain.
Quickly discarding such content before having an LLM generate
responses enables substantial speedups, analogous to proxy-model
filtering in video analytics [19, 20, 31].
At a high level, the InferenceEngine achieves this through
three steps: (1) divide 𝐷𝑠into chunks, (2) use a lightweight model
to filter out irrelevant chunks, and (3) generate responses only for
the remaining chunks using an expensive LLM.
Chunk-Level Inference.Prior work has shown that LLM accuracy
degrades as context length increases [ 29]. Similar to RAG, Halo
divides the pruned document 𝐷𝑠into chunksC={𝐶 1,...,𝐶 𝑛}.
However, unlike RAG, which concatenates retrieved chunks into a
single context for one inference call, Halo processes each chunk
through aseparateLLM call:
{𝑟1,...,𝑟 𝑛}={𝐿𝐿𝑀(𝑞,𝐶 𝑖)|𝐶 𝑖∈C}
This map-reduce style QA over documents is well-established in
the literature [ 33,37,45,46].Halo optimizes this by leveraging
domain knowledge in the Filter operator to remove irrelevant
chunks during mapping (i.e., before expensive inference) using the
extracted filter directives.
Applying the Filter Operator.The Filter knowledge provided
by the user is extracted as the filter directives F=(𝜙 1,...,𝜙 𝑛)—
semantic descriptions of content to exclude, such as “risk disclo-
sures” or “legal disclaimers.” To reduce inference cost, Halo evalu-
ates each chunk against these directives using a Small Language
Model (SLM) with 2–5B parameters, up to 500 ×smaller than frontier
6

LLMs (>1 trillion parameters). The reduced context size (one chunk
per call) and the structurally relevant content (after Structural )
simplifies the per-chunk classification task. By simplifying the task,
lightweight SLMs can filter accurately without sacrificing quality.
As shown in Figure 5, the Filter operator combines the directives
into a single prompt and invokes the SLM on each chunk:
keep(𝐶 𝑖,F,𝑞)=SLM(𝐶 𝑖⊕𝑞⊕serialize(F))→{0,1}
Filter(C,F,𝑞)={𝐶 𝑖∈C|keep(𝐶 𝑖,F,𝑞)=1}→C∗
We useC∗=Filter(C,F,𝑞)⊆Cfor the filtered candidate set.
Response Generation.After filtering, the retained chunks proceed
to the expensive LLM for per-chunk response generation:
R=Gen(C∗, 𝑞)={𝐿𝐿𝑀(𝑞,𝐶 𝑖)|𝐶 𝑖∈C∗}
producing candidate responses R={𝑟 1,...,𝑟 𝑚}, where𝑚=|C∗|.
Intuitively, Halo processes chunks through amodel cascade: a
lightweight SLM evaluates each chunk against the filter directives,
and only chunks that pass proceed to the more expensive LLM.
5.4 Mitigating Hallucinations
After applying the Filter operator, the retained chunks C∗are pro-
cessed by the LLM to generate chunk-level responses {𝑟1,...,𝑟 𝑚}.
TheVerifierranks these responses to select the best answer.
The Hallucination Problem.The chunk-level inference pattern
(§ 5.3) generates multiple candidate responses, each derived from a
different chunk of the document. While the reduced context length
per call improves per-chunk accuracy, the LLM can still hallucinate
when a chunk contains content that is semantically related but
factually distinct from the gold context (e.g., a table with 2021-
22 earnings when the query asks for 2022-23 earnings). This can
produce a mix of hallucinated and correct responses across chunks.
A naïve approach ranks these responses by model confidence,i.e.,
the probability the LLM assigns to its generated answer. However,
LLMs are known to be confidently incorrect [ 9], particularly when
the context contains similar but distinct information (e.g., “diluted
EPS” vs “basic EPS” in financial documents). Our key insight is
that users express validation knowledge in two forms: what the
responseshouldcontain (e.g., “diluted shares”, “net revenue”) and
what it shouldnotcontain (e.g., “NOT basic EPS”, “NOT gross
revenue”). Halo extracts and applies this validation knowledge to
rank responses, with the goal of surfacing the correct answer to
the top of the ranking.
Applying the Validate Operator.As defined in §4.3, the valida-
tion directives are partitioned into positive constraints V+and neg-
ative constraintsV−. Given the candidate responses, {𝑟1,...,𝑟 𝑚},
Halo needs to measure how well they align with the positive and
negative constraints. To quantify this alignment, our validation
score for each response rewards alignment with positive constraints
and penalizes alignment with negative constraints:
score(𝑟 𝑖,V)=∑︁
𝛿∈V+sim(𝑟 𝑖,𝛿)−∑︁
𝛿∈V−sim(𝑟 𝑖,𝛿)
where sim(𝑟 𝑖,𝛿)denotes the cosine similarity between the embed-
ding representations of response 𝑟𝑖and constraint 𝛿. Intuitively, a
response mentioning “diluted EPS” scores high against a positive
constraint𝛿1=“diluted EPS” , while a response mentioning “basic
EPS” incurs a penalty from a negative constraint𝛿 2=“basic EPS”.TheValidate computes the validation score for each response
and returns the top-ranked responses:
Validate(R,V)=top-𝑘 𝑟∈R 𝑟|score(𝑟,V)→𝐴∗
Unlike confidence-based ranking, the Validate operator system-
atically leverages domain knowledge to identify the semantically
correct response. Our experiments (§7) show that validation-based
ranking improves accuracy by up to 8% compared to confidence-
based ranking (Table 9). Since the scoring function is used for rank-
ing rather than thresholding, the relative count of positive and neg-
ative constraints provided by the user does not affect correctness.
This avoids the need for manual and cumbersome threshold tun-
ing. For example, even with only negative constraints, the correct
response incurs a smaller penalty than incorrect ones, preserving
the ranking order.
5.5 Extensions
While the three operators described above capture the most com-
mon prompting patterns, KnowledgeParser is extensible to other
knowledge types. For example, experts may express terminology
equivalence (e.g., “treat MI and myocardial infarction as equivalent”)
in their prompts, which could be extracted into a NORMALIZE opera-
tor. Precedence knowledge (e.g., “prioritize master agreement over
the addendum”) expresses conflict resolution rules, which could
be extracted into a PRECEDENCE operator. Custom operators such
asNORMALIZE andPRECEDENCE can be defined and implemented in
Halo following the same pattern as the three operators described
in this section.
6 Operator Robustness
Domain operators enhance long-context QA when the user-provided
domain knowledge is accurate and well-specified. However, in prac-
tice, operators may be imprecise, overly aggressive, or misaligned
with the document content, which can degrade accuracy. Halo
addresses this through a fallback mechanism that detects imprecise
operators at runtime and recovers by selectively disabling prob-
lematic operators. This allows Halo to benefit from high-quality
operators when available, while gracefully falling back to baseline
performance when operators are imprecise.
6.1 When Do Operators Degrade Accuracy?
We now describe how each operator can be impacted by inaccurate
domain knowledge.
Structural Operator.The Structural operator prunes the in-
put document to retain only structurally relevant regions (§5.2).
This pruning removes the gold context (i.e., the portion of the doc-
ument containing the correct answer) when: (1) the user specifies
an incorrect structural element (e.g., directing the system to look
in tables when the answer appears in a text paragraph), (2) none of
the user directives match the document’s structural tags, or (3) the
document parsing fails to extract the relevant structure (e.g., a mal-
formed table is not tagged). In all cases, downstream chunk-level
inference operates on a document no longer containing the answer.
Filter Operator.The Filter operator discards entire chunks
deemed irrelevant by the SLM-based classifier (§5.3). The operator
7

degrades accuracy when: (1) the user’s filter directives are inaccu-
rate (e.g., “ignore financial tables” when the answer is in a financial
table), or (2) the SLM misclassifies a relevant chunk as irrelevant,
either due to model errors or because the chunk contains irrelevant
portions mixed with the gold context.
Validate Operator.The Validate operator ranks candidate re-
sponses using embedding similarity against positive and negative
constraints (§5.4). Unlike the Structural andFilter operators,
which affectwhat contextthe LLM sees, Validate affects there-
sponse orderingof candidates. The Validate operator degrades
accuracy when the constraints fail to produce meaningful score
differences across candidate responses. This occurs when: (1) the
constraints are irrelevant to the document content (e.g., constraints
about “martial arts” applied to a financial document), producing uni-
formly low similarity scores, or (2) the constraints are too generic to
distinguish between responses (e.g., “financial data” when all candi-
date responses discuss financial content), producing uniformly high
similarity scores. In both cases, the constraints match all responses
similarly, and the validation ranking is non-discriminative.
6.2 Detecting Operator Imprecision
To recover from low-quality operators, Halo must first detect that
an operator has negatively impacted the query result. We identify
two detection mechanisms that leverage signals produced during
normal execution, requiring no additional LLM calls.
6.2.1 Detecting Context Loss.To detect when Structural orFilter
operators are too aggressive, Halo leverages the LLM’s inherent
answer refusalcapability,i.e., its tendency to respond with “answer
not in context” when the provided context is insufficient [ 5,13,43].
Wheneverychunk-level response is a refusal, the upstream opera-
tors have removed all content related to the query:
∀𝑟𝑖∈{𝑟 1,...,𝑟 𝑚}:𝑟 𝑖=“answer not in context”
This all-refusal signal detects complete context loss and triggers
the context fallback (§6.3.1). When at least one chunk produces
a substantive response, the system proceeds to the ranking stage,
where theValidateoperator handles candidate selection.
6.2.2 Detecting Ranking Degradation.When Validate constraints
are inaccurate, the correct response may be ranked lower than
incorrect alternatives. Unlike context loss, the LLMdoesgenerate
the correct answer, but it is not selected as the top response.
Halo uses score variance across candidate responses to detect
ranking degradation: if all responses receive similar validation
scores (i.e., low variance across {𝑠𝑐𝑜𝑟𝑒(𝑟 1,V),...,𝑠𝑐𝑜𝑟𝑒(𝑟 𝑚,V)} ),
the constraints fail to discriminate among candidates. When low
variance is detected, Halo flags the Validate operator as non-
discriminative and triggers the ranking fallback (§6.3.2).
6.3 Recovery via Fallback
Once Halo detects operator imprecision, it initiates an operator-
specific fallback mechanism to recover accuracy.
6.3.1 Context Fallback.When context loss is detected, Halo dis-
ables the Structural andFilter operators and re-executes the
query on the full document. When the Structural operator is by-
passed, the full parsed document ˜𝐷is used instead of the pruned 𝐷𝑠.When the Filter operator is bypassed, all chunks proceed to LLM
inference without SLM-based filtering. The re-execution generates
a new set of candidate responses from the unfiltered document.
6.3.2 Ranking Fallback.When non-discriminative validation is
detected, Halo disables the Validate operator and falls back to
confidence-based ranking. The candidate responses {𝑟1,...,𝑟 𝑚}are
re-ranked using the LLM’s output probabilities:
𝐴∗
𝑓 𝑎𝑙𝑙𝑏𝑎𝑐𝑘=top-𝑘(𝑟 𝑖|conf(𝑟 𝑖))
where conf(𝑟 𝑖)is the model’s confidence score for response 𝑟𝑖. In
both cases, fallback ensures performance is no worse than a vanilla
LLM baseline: the context fallback recovers full-document inference,
and the ranking fallback reverts to confidence-based selection. We
show the effectiveness of the fallback mechanism in recovering
from low quality operators in §7.4.
7 Experiments
In this section, we evaluate Halo to empirically answer the follow-
ing research questions:
•RQ1: How does Halo compare to baselines in end-to-end ac-
curacy and cost across models and domains? (§7.2)
•RQ2: How do individual operators contribute to accuracy and
cost improvements? (§7.3)
•RQ3: How robust is Halo to low-quality domain knowledge,
and how effective is its fallback mechanism? (§7.4)
•RQ4: What is the cost overhead of Halo ’s operators, and how
effective is validation-based ranking? (§7.5.1, §7.5.2)
7.1 Experimental Setup
Datasets.We evaluate Halo across three datasets from diverse do-
mains. Table 4 summarizes the key properties of these datasets.
•DocFinQA [ 35]. This dataset contains full Securities and Ex-
change Commission (SEC) filings, leading to significantly larger
document contexts. The dataset includes 801 documents with
an average of 123k words per document and a total of 7,437
expert-annotated QA pairs. The output is in numerical format,
allowing for a thorough evaluation of numerical reasoning
over long contexts. We randomly selected 100 questions from
the development set.
•NovelQA [ 41]. This dataset contains full-length English novels
averaging more than 200K tokens with different QA tasks such
as single-hop QA, multi-hop QA, and summarization. We use
99 multiple-choice questions from the single-hop subset.
•QASPER [ 12]. This dataset contains 5,049 questions over 1,585
research papers in the NLP domain, focusing on information-
seeking QA tasks. Each question is grounded in full-length aca-
demic papers, with evidence spanning multiple sections, para-
graphs, and figures or tables. The average document length is
7K tokens. We randomly selected a subset of 90 examples that
are of the true/false category for reliable answer evaluation.
Baselines.We evaluateHaloagainst six baselines:
•VanillaLLM. This baseline processes the entire document in
a single LLM call and generates a single response. We set the
context length at 200K tokens, which is the context window
limit for Claude Sonnet 4.5 [2].
8

Table 3: Representative examples of domain knowledge used in our experiments. The domain knowledge column shows the
natural language instructions a domain expert provides alongside the question.
Dataset Sample Question Domain Knowledge
DocFinQA What was the percentage cumulative total return for the
five year period ended 31-Dec-2017 of Citi common stock?Look in tables and the MD&A section. Ignore legal disclaimers, table of contents, and chunks
not relevant to the query. The answer should reference Citi Common Stock Cumulative Total
Return from the performance graph source. Do NOT confuse with S&P 500 return.
NovelQA How long did it take for Sibyl’s brother and Dorian Gray to
meet after Sibyl’s death?Sibyl’s brother is James Vane. Skip chapters before Sibyl’s death and chapters unrelated to James
Vane or Dorian. The answer involves the time gap between Sibyl’s death and James confronting
Dorian. Do NOT confuse with Dorian’s interactions with Lord Henry after Sibyl’s death.
QASPERDo they try to use byte-pair encoding representations? Look in the methodology and results sections.
Table 4: Dataset characteristics.
Dataset QA Domain # Queries Avg.
#tokens
DocFinQA Finance 100 193K
NovelQA Literature 99 230K
QASPER Science Articles 90 7K
•RAG . This baseline leverages an embedding model [ 7] to re-
trieve the top- 𝑘relevant chunks, then sends the concatenated
chunks as context to an LLM to generate a single response.
•{VanillaLLM ,RAG }+DK. These methods inject domain knowl-
edge directly into the prompt, and call the LLM with context
gathered either using the VanillaLLM strategy or RAG . The
domain knowledge is written as natural language instructions
at the end of the user question, similar to how users write
instructions in their prompts today.
•CoA [45]. Chain-of-Agents (CoA) processes individual chunks
of long documents via worker LLMs with sequential informa-
tion sharing across worker agents. It then aggregates results
via a manager LLM. We adapt COA for pinpointed QA by using
parallel workers, since sequential evidence aggregation pro-
vides no benefit when answers are localized to single chunks.
Workers process the full document without retrieval.
•Halo (No DK) . This baseline uses the Halo pipeline without
enabling the domain operators. The top- 𝑘chunks are selected
via embedding similarity and processed via individual LLM
calls, and the responses are ranked using confidence score (for
local models) and in the selection order (for closed models).
For all baselines that require chunking and top- 𝑘chunk selection,
we perform table/layout-aware chunking, use a chunk size of ∼4000
tokens, and set 𝑘=10. Baselines that process chunks via indepen-
dent LLM calls— CoA ,Halo (No DK) , and Halo —leverage batched
inference to reduce per-query latency and cost. All chunk-level
calls within a single query are issued in parallel, and the reported
costs for these baselines reflect this batched execution.
Domain Knowledge Sources.For each dataset, we construct
domain knowledge prompts as would be provided by domain ex-
perts alongside their question. These prompts are constructed
through manual inspection of the dataset queries and source doc-
uments, supplemented by dataset-specific metadata where avail-
able. For DocFinQA, we use evidence annotations from the parent
dataset FinQA [ 8] to determine whether the answer resides in atable, text, or both, and construct structural directives accordingly
(e.g., “look in tables and the MD&A section”). We derive filter direc-
tives from recurring irrelevant content patterns in SEC 10-K filings
(e.g., legal disclaimers, tables of contents). Validation directives are
constructed by inspecting the evidence text associated with each
query to identify entities that co-occur with the correct answer
(positive directives,e.g., “diluted computations”) and nearby enti-
ties that could be confused with it (negative directives,e.g., “basic
computations”). For NovelQA, we use chapter-level metadata and
character annotations to construct directives that guide the system
to relevant portions of the novel (e.g., “skip chapters before Sibyl’s
death”). For QASPER, we use section-level structure common to
NLP papers (e.g., “look in the methodology and results sections”).
Table 3 provides representative examples from each dataset. On
average, the KnowledgeParser extracts 10, 6, and 2 directives per
query for DocFinQA, NovelQA, and QASPER, respectively.
Implementation Details.We use both open- and closed-source
LLMs for our experiments. For the closed-source model, we use
Claude Sonnet 4.5 [ 2] (released September, 2025) via the Amazon
AWS Bedrock Platform API1. For the open-source model, we use
theQwen3-30B-A3B-Instruct long-context model [ 39] (released
July, 2025) hosted locally with the vLLM inference framework [ 24].
We use Qwen3-4B-Instruct hosted locally on vLLM as the light-
weight filter model for our Filter operator. For selecting the top- 𝑘
relevant chunks, we use the bge-m3 embedding model [ 7]. We use
thesentence-transformers library for embedding computations.
Metrics.We use Mean Reciprocal Rank (MRR@k) to measure
accuracy, a standard metric for evaluating ranked response lists [ 21].
MRR@k measures how often the correct answer appears within
the top-𝑘ranked results; higher values at low 𝑘indicate that users
need to inspect fewer responses. For DocFinQA, we use relative
error with 5% tolerance since the responses are numerical answers.
For NovelQA and QASPER, we perform exact matching against the
ground truth. For cost, we report the average dollar cost per query
based on AWS Bedrock pricing2, computed from total input and
output tokens across all LLM and embedding model calls.
7.2 End-to-end Results
We first explore how Halo uses domain knowledge to improve
query accuracy and cost using the DocFinQA dataset. We use two
1https://aws.amazon.com/bedrock/
2https://aws.amazon.com/bedrock/pricing/
9

V/a.sc/n.sc/i.sc/l.sc/l.sc/a.scLLM
V/a.sc/n.sc/i.sc/l.sc/l.sc/a.scLLM+DKRAG
RAG+DKC/o.scA
H/a.sc/l.sc/o.sc (N/o.sc DK)H/a.sc/l.sc/o.sc
Be/t_ter
0.2 0.4
Cost per /Q_uery ( $)657075Accuracy (%)Claude Sonnet 4.5
0.005 0.010 0.015 0.020
Cost per /Q_uery ( $)556065Qwen3-30BFigure 6: End-to-end accuracy (MRR@1) vs. cost per query on
DocFinQA (100 queries). Halo achieves the highest accuracy
at the lowest cost across both LLMs (top-left is best).
LLMs: Claude Sonnet 4.5 [2] and Qwen3-30B [39]. Figure 6 shows
the accuracy (MRR@1) and cost per query for all methods.
By capturing and applying domain knowledge in query process-
ing,Halo improves accuracy over VanillaLLM by 4% and 9% using
Claude and Qwen, respectively. Halo also reduces cost by up to
4.8×compared to VanillaLLM . Compared to RAG , it improves ac-
curacy by 9% and cost by up to 1.8 ×.VanillaLLM andRAG process
the document without any domain knowledge, incurring high cost
and suffering from accuracy degradation over long contexts.
Even when injecting the domain knowledge into the baselines’
prompts ( VanillaLLM + DK andRAG + DK ),Halo achieves up to
4.8×lower cost and up to 13% higher accuracy. Further, injecting
the domain knowledge can sometimeshurtthe baselines accu-
racy: with Qwen, VanillaLLM + DK dropped by 4% compared to
VanillaLLM , and RAG + DK dropped by 2% compared to RAG . This
is consistent with prior work that shows weaker models struggle
to follow complex instructions over long contexts [ 17,44]. This
demonstrates that systematically applying the domain knowledge
is necessary for reducing accuracy degradation. Also of note: both
+DK methods incur the same cost as their non-domain knowledge
counterparts since the model still processes the entire document.
Compared to CoA ,Halo improves accuracy by 8% on Claude
and reduces cost by 4 ×.CoA achieves thelowestaccuracy on Claude
(66%), underperforming even VanillaLLM (70%), despite decom-
posing the document into chunks. We observe that while the can-
didate responses do contain the correct answer, the manager LLM
often fails to identify it from the large pool of candidates. Halo ’s
Validateoperator addresses this gap by leveraging domain direc-
tives to identify the correct response from the candidate pool. CoA
also incurs significant costs even with chunk-level batch process-
ing enabled due to processing theentire documentsequentially in
chunks (as opposed toVanillaLLMthat truncates the document).
Robustness across models. Halo ’s improvements on cost and
accuracy hold regardless of the underlying LLM. The accuracy gap
between Halo and baselines is larger with Qwen than with Claude,
suggesting that weaker models benefit more from systematic do-
main knowledge application, as they are less capable of implicitly
leveraging domain context on their own. Halo with Qwen (65%)
approaches VanillaLLM on Claude (70%) at 78 ×lower cost, demon-
strating that Halo can enable a smaller, more cost-efficient model
to approach the accuracy of a frontier model used naively.Table 5: Domain generalization: accuracy (%) and cost
($/query) on NovelQA (99 queries, >200K tokens) and
QASPER (90 queries, 7K tokens) using Qwen. Values in paren-
theses show the change when adding prompt-based domain
knowledge (+DK).
NovelQA QASPER
Method Acc. Cost Acc. Cost
VanillaLLM80 (−1) 0.029 64 0.0012
RAG78 (−1) 0.029 64 0.0012
CoA79 0.028 64 0.0011
Halo (No DK)79 0.023 64 0.0011
Halo83 0.022640.0007
Table 6: Contribution of operators on DocFinQA (100 queries,
Claude): accuracy (%) and cost ($/query) when each operator
is applied independently to theHalo (No DK)pipeline.
Configuration Acc. (%) Cost ($/q)
Halo (No DK)66 0.14
+Structural69 0.12
+Filter65 0.11
+Validate71 0.14
FullHalo74 0.10
⇒Takeaway: Halo consistently achieves the best accuracy–cost trade-off
across all baselines. Systematically applying domain knowledge is more effec-
tive and robust than prompt injection or engineering, enabling a lightweight
open-source model to approach frontier accuracy at 60–80×lower cost.
Performance on other domains.We also evaluateHaloon
NovelQA (literature, >200K tokens) and QASPER (scientific articles,
7K tokens) using the Qwen model. The results are in Table 5. On
NovelQA, Halo outperforms all baselines by 4% in accuracy. The
gains are smaller than on DocFinQA because narrative text offers
less exploitable structure for the Structural operator, and entity
references are distributed throughout the text, limiting Filter effec-
tiveness. However, Halo ’sValidate operator remains effective at
detecting hallucinations in narrative-style responses, where LLMs
tend to fabricate elaborate but incorrect evidence. On QASPER, all
methods achieve similar accuracy due to the short context length
(7K tokens). Halo still provides cost benefits through Structural
andFilter , which avoid irrelevant sections and chunks without
sacrificing accuracy.
⇒Takeaway: Halo ’s benefits are most pronounced on long, structured docu-
ments. On narrative text, Validate remains effective at detecting hallucina-
tions even when structural operators offer limited gains.
7.3 Contribution of Individual Operators
We now independently assess the impact of each domain opera-
tor using DocFinQA with Claude. Halo (No DK) runs Halo ’s full
multi-stage pipeline—chunking, parallel inference, and confidence-
based ranking—without any domain operators, serving as the base-
line for measuring operator-level gains. Table 6 shows the accuracy
and cost per query for each configuration.
Structural .Adding the Structural operator improves accu-
racy by 4% while reducing cost by 14%. By pruning the document
10

Table 7: Robustness to low-quality operators on DocFinQA
(Qwen): accuracy (%) and cost ($/query) when Halo is given
incorrect Structural elements and irrelevant Validate
constraints, with and without the fallback manager.
w/o Fallback w/ Fallback
Operator Acc. Cost Acc. Cost
Structural14% 0.007 55% 0.017
Validate50% 0.006 57% 0.006
to retain only structurally relevant regions before chunking, the
Structural operator produces context-rich chunks, reducing hal-
lucinations in individual chunk-level responses. This also reduces
token cost in cases where the total chunks are already below the
top-𝑘due to aggressive pruning at theStructurallevel.
Filter .The Filter operator reduces cost by 21% with a 1% accu-
racy drop. Unlike Structural which operates before top- 𝑘selec-
tion, Filter removes chunks directly from the 𝑘selected chunks
before expensive LLM calls, yielding larger per-query cost reduction.
The slight accuracy drop occurs because the SLM-based classifier
may occasionally discard a chunk containing relevant information.
Filter may discard relevant responses if the SLM-based filtering
is inaccurate. To mitigate this, Halo applies Filter conservatively,
only removing chunks that are highly likely to be irrelevant.
Validate .The Validate operator improves accuracy by 5% with
no cost impact. Unlike Structural andFilter , which reduce the
context seen by the LLM, Validate re-ranks candidate responses
using domain-specific constraints (§5.4) to effectively detect and
demote hallucinated responses. By rewarding responses aligned
with positive constraints (e.g., “diluted EPS”) and penalizing those
matching negative constraints (e.g., “basic EPS”), Validate surfaces
correct answers that confidence-based ranking misses. The cost im-
pact is negligible since ranking only requires computing embedding
similarities over the already-generated responses (§ 5.4).
Combined effect.Full Halo applies all three operators, achiev-
ing 74% accuracy at $0.10/query—an 8% accuracy gain and 29%
cost reduction over Halo (No DK) . Notably, the accuracy gains
from Validate compound when combined with richer inputs from
Structural andFilter :Structural improves per-chunk response
quality by providing cleaner context, which in turn enables Validate
to more effectively rank the responses using domain constraints.
The cost reductions from Structural andFilter also compound,
as they operate at different granularities—document-level structural
pruning and chunk-level semantic filtering, respectively.
⇒Takeaway:Each operator contributes distinct benefits: Structural fo-
cuses the context, Filter reduces the cost via context pruning, while Validate
improves accuracy through domain-aware ranking. Their effects compound in
the full system.
7.4 Robustness to Low-Quality Operators
We now evaluate Halo ’s ability to detect low-quality operators and
use its fallback manager (§6) to prevent accuracy degradation. We
consider two types of low-quality operators: For Structural , we
provide incorrect structural elements (e.g., text for table-answered
questions), which causes the operator to prune the gold context.
ForValidate , we provide constraints unrelated to the documentTable 8: Cost Analysis for Halo : Cost breakdown of Halo ’s
individual stages and operations as a fraction of the overall
query cost.
Stage Operator/Step Model Cost/query ($) % of total
One-time (amortized)
Pre-processing - Gemini-2.0-Flash 0.01142 -
Per Query
KnowledgeParser- Sonnet-4.5 0.001 1.04%
ContextSelectorStructuralbge-m3 0.00002 0.02%
InferenceEngineFilterQwen-4b 0.00093 0.96%
Expensive LLM Sonnet-4.5 0.09454 97.96%
Verifier Validatebge-m3 0.00001 0.01%
content (e.g., constraints about science articles applied to a financial
document), making the operator non-discriminative. We evaluate
Haloon the 100 queries from the DocFinQA dataset using Qwen.
Table 7 shows the accuracy and cost per query of Halo without
and with its fallback manager. Halo effectively handles low-quality
Structural operators in 80% of the cases via the context-loss de-
tection mechanism (§6.2.1), which leads to an accuracy recovery
from 14% to 55%. In the remaining 20%, the pruned document still
contains content semantically related to the query (e.g., a text para-
graph mentioning EPS when the answer resides in a table), causing
the LLM to generate a confident but incorrect response rather than
refusing, and thus preventing the all-refusal signal from triggering.
Leveraging model abstention and confidence scores can effec-
tively detect and mitigate the negative impact of filtering out rele-
vant context. This recovery process incurs a 1.7 ×increase in cost
due to necessary re-execution on the full document (§6.3.1).
Halo shows similar robustness to low-quality Validate oper-
ators. When irrelevant constraints are provided, the validation
scoring function (§5.4) produces near-zero similarity with all candi-
date responses. Halo detects this (§6.2.2) and disables the Validate
operator, falling back to confidence-based ranking, which is already
available from inference. The recovered accuracy (57%) matches
Halo (No DK) , confirming that the fallback reverts to the no-domain
knowledge baseline when Validate constraints are non-discriminative.
⇒Takeaway: Halo recovers gracefully from imprecise Structural and
Validate operators. The fallback manager ensures Halo performs no worse in
accuracy than the pipeline without domain operators.
Table 9: Ranking quality at different MRR cutoffs on
DocFinQA (100 queries, Claude).
Method MRR@1 MRR@3 MRR@5
Halo (No DK)66 71 72
Halo74 76 76
7.5 Ablation Studies
7.5.1 Cost Breakdown.We now quantify the cost of Halo ’s stages.
We report the breakdown of per-query cost on DocFinQA in Table 8.
We report costs using API-based pricing for all components.
The dominant cost is the expensive LLM inference (97.96% of to-
tal). All other Halo operators contribute only 2.04% combined to the
total cost. The KnowledgeParser , which uses an LLM to extract
11

domain operators from the user domain instructions, contributes
only 1% of the total cost, since user instructions rarely exceed 500
tokens compared to the >200K document sizes. The Structural
andValidate operators rely on lightweight embedding models
for similarity over short input snippets (structural tags/responses),
only taking up 0.02% and 0.01% of the total cost, respectively. The
Filter operator invokes a 4B-parameter SLM per candidate chunk,
contributing 1% to the total cost due to the 75 ×cheaper SLM cost.
The one-time pre-processing cost covers PDF parsing and structural
tag extraction using a cost-effective model [ 14], and is amortized
across all queries on the same document. With a total operator cost
of just $0.003 per query, the domain operators reduce the per-query
expensive LLM cost from $0.14 ( Halo (No DK) ) to $0.095 — a 14×
return despite the added overheads.
7.5.2 Ranking Analysis.We now show Halo ’s ability to rank can-
didates as𝑘varies. Table 9 compares Halo andHalo (No DK) for
varying MRR@𝑘.
Halo improves MRR consistently across all cutoffs (+5–8% over
Halo (No DK) ), confirming that the correct answers are reliably
surfaced in the top positions due to top- 𝑘selection with richer
chunks (w/ Structural ) and better re-ranking (w/ Validate ).
We also observed that applying Validate alone to CoA improves
MRR@1 by 6%, thus demonstrating the re-ranking remains effective
even over larger, noisier candidate pools.
8 Related Work
LLMs and Data Management.There have been extensive re-
search efforts into leveraging LLMs to improve data management,
and to incorporate data management principles for improving LLM-
based pipelines. Evaporate [ 3] generates structured views from het-
erogeneous data lakes using LLMs. ZenDB [ 27] extracts semantic
hierarchical structures from semi-structured documents and uti-
lizes LLMs to enable efficient SQL querying over these documents.
In contrast, while Halo may leverage existing structure in the doc-
ument to optimize long-context QA (e.g., using the Structural
operator), it does not focus on the structured data extraction task.
Other works explore optimizing complex LLM-based pipelines
using data management techniques. LOTUS [ 33] introduces a se-
mantic operator model for AI-based workloads and proposes several
optimizations for each operator that reduce the cost while provid-
ing accuracy guarantees. Palimpzest [ 28] and DocETL [ 37] provide
declarative interfaces to define complex AI processing pipelines.
While Palimpzest uses cost-based optimization to enhance diverse
AI applications, DocETL uses agent-based rewriting to optimize
multi-step document processing pipelines. Halo differs from these
systems in that it automatically extracts domain knowledge from
user prompts, applies it across its multi-stage pipeline, and detects
accuracy degradation via its fallback mechanism.
Long-context QA.While LLMs feature expanded context win-
dows [ 34,38], effectively applying them to QA tasks over long doc-
uments poses several challenges. Models often struggle to retain
information across extensive contexts [ 1,23,29]. RAG is a popu-
lar paradigm for QA over large information corpora [ 26], wherein
relevant segments are retrieved through embedding-based similar-
ity search and passed to an LLM for generation. While early RAGsystems performed retrieval over smaller chunks, recent advances
in context window sizes have improved embedding models [ 7],
enabling enhanced RAG systems tailored for longer inputs. To
overcome the limitations of vanilla LLMs and RAG, multi-agent
pipelines have been proposed for long-context QA [ 45] that split
the document into individual chunks, process the individual chunks
using multiple agent LLMs, and aggregate the responses using a
manager LLM. Halo leverages user-specified domain knowledge
to improve various stages (e.g., chunk selection, answer validation)
within the multi-agent QA pipeline.
Domain knowledge in qery optimization.Commercial
database systems have historically enabled users to express domain
knowledge through user-defined hints [ 10,11,40]. Traditionally,
these hints are designed for database administrators and domain
experts to influence the query optimizer’s selection of execution
plans [ 4,10]. More recently, this concept has been expanded to opti-
mize video analytics pipelines that involve machine learning models.
VIVA [ 36] introduces relational hints ( CAN REPLACE ,CAN FILTER )
that allow users to declaratively specify model relationships and
automatically selects the best plan that meets user-specified accu-
racy requirements. Similarly, ClueVQS [ 6] proposes user-defined
Cluesto incorporate domain-specific knowledge into video ana-
lytics applications. While Halo draws inspiration from these ap-
proaches, it automatically extracts domain knowledge from users’
natural language prompts and applies it across multiple pipeline
stages—requiring no change to how users express their knowledge.
Domain Knowledge in LLMs.Prompting techniques like chain-
of-thought have been shown to elicit more complex reasoning
from LLMs [ 22,42]. However, these techniques are sensitive to
phrasing, and users often struggle to effectively articulate complex
requirements or domain constraints for LLMs [ 32]. This can result
in multiple iterations of prompt engineering. Halo removes the
need for prompt engineering by automatically extracting domain
knowledge from user prompts and applying it at the appropriate
stage of its multi-stage pipeline.
9 Conclusion
Despite rapidly expanding context windows in modern LLMs, long-
context question answering remains a significant challenge due
to accuracy degradation over long contexts and high processing
costs. This paper presented Halo , a framework that bridges the gap
between the domain knowledge experts naturally express in their
prompts and how long-context QA systems utilize it. By extracting
this knowledge into structured directives and applying them via ded-
icated operators— Structural ,Filter , and Validate —across dis-
tinct pipeline stages,Halomoves beyond treating domain knowl-
edge as passive prompt tokens that the LLM may or may not follow.
Across finance, literature, and scientific domains, we demonstrate
thatHalo outperforms both vanilla and prompt-engineered base-
lines, achieving up to 13% higher accuracy and 4.8 ×lower cost. No-
tably, Halo enables a lightweight open-source model to approach
frontier LLM accuracy at 78 ×lower cost, showing that systematic
application of domain knowledge can compensate for model scale.
12

References
[1] Chenxin An, Jun Zhang, Ming Zhong, Lei Li, Shansan Gong, Yao Luo, Jingjing Xu,
and Lingpeng Kong. 2025. Why Does the Effective Context Length of LLMs Fall
Short?. InThe Thirteenth International Conference on Learning Representations.
https://openreview.net/forum?id=eoln5WgrPx
[2] Anthropic. 2025. Claude Sonnet 4.5. https://www.anthropic.com/news/claude-
sonnet-4-5. Accessed: 2026-02-24.
[3]Simran Arora, Brandon Yang, Sabri Eyuboglu, Avanika Narayan, Andrew Ho-
jel, Immanuel Trummer, and Christopher Ré. 2023. Language Models Enable
Simple Systems for Generating Structured Views of Heterogeneous Data Lakes.
Proceedings of the VLDB Endowment17, 2 (2023), 92–105.
[4] Nicolas Bruno, Surajit Chaudhuri, and Ravi Ramamurthy. 2009. Power hints for
query optimization. In2009 IEEE 25th International Conference on Data Engineer-
ing. IEEE, 469–480.
[5] Lang Cao. 2024. Learn to Refuse: Making Large Language Models More Control-
lable and Reliable through Knowledge Scope Limitation and Refusal Mechanism.
InProceedings of the 2024 Conference on Empirical Methods in Natural Language
Processing. 3628–3646.
[6]Daren Chao, Yueting Chen, Nick Koudas, and Xiaohui Yu. 2024. Optimizing
Video Queries with Declarative Clues.Proceedings of the VLDB Endowment17,
11 (2024), 3256–3268.
[7] Jianlv Chen, Shitao Xiao, Peitian Zhang, Kun Luo, Defu Lian, and Zheng Liu. 2024.
BGE M3-Embedding: Multi-Lingual, Multi-Functionality, Multi-Granularity Text
Embeddings Through Self-Knowledge Distillation. arXiv:2402.03216 [cs.CL]
[8] Zhiyu Chen, Wenhu Chen, Charese Smiley, Sameena Shah, Iana Borova, Dylan
Langdon, Reema Moussa, Matt Beane, Ting-Hao Huang, Bryan R Routledge,
et al.2021. FinQA: A Dataset of Numerical Reasoning over Financial Data. In
Proceedings of the 2021 Conference on Empirical Methods in Natural Language
Processing. 3697–3711.
[9] Prateek Chhikara. 2025. Mind the Confidence Gap: Overconfidence, Calibration,
and Distractor Effects in Large Language Models.Transactions on Machine
Learning Research(2025). https://openreview.net/forum?id=lyaHnHDdZl
[10] Microsoft Corporation. 2025. Hints (Transact-SQL). https://learn.microsoft.com/
en-us/sql/t-sql/queries/hints-transact-sql?view=sql-server-ver16. Accessed:
2025-04-03.
[11] Oracle Corporation. 2023. Influencing the Optimizer with Hints. https:
//docs.oracle.com/en/database/oracle/oracle-database/23/tgsql/influencing-
the-optimizer.html#GUID-C558F7CF-446E-4078-B045-0B3BB026CB3C.
Accessed: 2025-04-03.
[12] Pradeep Dasigi, Kyle Lo, Iz Beltagy, Arman Cohan, Noah A. Smith, and Matt Gard-
ner. 2021. A Dataset of Information-Seeking Questions and Answers Anchored
in Research Papers. InProceedings of the 2021 Conference of the North Ameri-
can Chapter of the Association for Computational Linguistics: Human Language
Technologies, Kristina Toutanova, Anna Rumshisky, Luke Zettlemoyer, Dilek
Hakkani-Tur, Iz Beltagy, Steven Bethard, Ryan Cotterell, Tanmoy Chakraborty,
and Yichao Zhou (Eds.). Association for Computational Linguistics, Online, 4599–
4610. https://doi.org/10.18653/v1/2021.naacl-main.365
[13] Shangbin Feng, Weijia Shi, Yike Wang, Wenxuan Ding, Vidhisha Balachandran,
and Yulia Tsvetkov. 2024. Don’t Hallucinate, Abstain: Identifying LLM Knowledge
Gaps via Multi-LLM Collaboration. InProceedings of the 62nd Annual Meeting
of the Association for Computational Linguistics (Volume 1: Long Papers). 14664–
14690.
[14] Sergey Filimonov. 2025. Ingesting Millions of PDFs and why Gemini 2.0 Changes
Everything. https://www.sergey.fyi/articles/gemini-flash-2. Accessed: 2025-04-
03.
[15] Yujian Gan, Xinyun Chen, Jinxia Xie, Matthew Purver, John R Woodward, John
Drake, and Qiaofu Zhang. 2021. Natural SQL: Making SQL easier to infer from
natural language specifications. InFindings of the Association for Computational
Linguistics: EMNLP 2021. 2030–2042.
[16] Google. 2026. Gemini 3 Pro. https://docs.cloud.google.com/vertex-ai/generative-
ai/docs/models/gemini/3-pro. Accessed: 2026-02-24.
[17] Qianyu He, Jie Zeng, Wenhao Huang, Lina Chen, Jin Xiao, Qianxi He, Xunzhe
Zhou, Jiaqing Liang, and Yanghua Xiao. 2024. Can large language models under-
stand real-world complex instructions?. InProceedings of the AAAI Conference
on Artificial Intelligence, Vol. 38. 18188–18196.
[18] Bowen Jin, Jinsung Yoon, Jiawei Han, and Sercan O Arik. 2025. Long-Context
LLMs Meet RAG: Overcoming Challenges for Long Inputs in RAG. InThe Thir-
teenth International Conference on Learning Representations. https://openreview.
net/forum?id=oU3tpaR8fm
[19] Daniel Kang, Peter Bailis, and Matei Zaharia. 2019. BlazeIt: optimizing declarative
aggregation and limit queries for neural network-based video analytics.VLDB
13, 4 (2019), 533–546.
[20] Daniel Kang, John Emmons, Firas Abuzaid, Peter Bailis, and Matei Zaharia. 2017.
NoScope: Optimizing Neural Network Queries over Video at Scale.VLDB10, 11
(2017).
[21] Omar Khattab and Matei Zaharia. 2020. Colbert: Efficient and effective passage
search via contextualized late interaction over bert. InProceedings of the 43rdInternational ACM SIGIR conference on research and development in Information
Retrieval. 39–48.
[22] Takeshi Kojima, Shixiang Shane Gu, Machel Reid, Yutaka Matsuo, and Yusuke
Iwasawa. 2022. Large language models are zero-shot reasoners.Advances in
neural information processing systems35 (2022), 22199–22213.
[23] Yuri Kuratov, Aydar Bulatov, Petr Anokhin, Ivan Rodkin, Dmitry Igorevich
Sorokin, Artyom Sorokin, and Mikhail Burtsev. 2024. BABILong: Testing the
Limits of LLMs with Long Context Reasoning-in-a-Haystack. InThe Thirty-eight
Conference on Neural Information Processing Systems Datasets and Benchmarks
Track. https://openreview.net/forum?id=u7m2CG84BQ
[24] Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng,
Cody Hao Yu, Joseph E. Gonzalez, Hao Zhang, and Ion Stoica. 2023. Efficient
Memory Management for Large Language Model Serving with PagedAtten-
tion. InProceedings of the ACM SIGOPS 29th Symposium on Operating Systems
Principles.
[25] Quinn Leng, Jacob Portes, Sam Havens, Matei Zaharia, and Michael Carbin.
2024. Long Context RAG Performance of Large Language Models. InAdaptive
Foundation Models: Evolving AI for Personalized and Efficient Learning. https:
//openreview.net/forum?id=Le9anH3kv1
[26] Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir
Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rock-
täschel, et al .2020. Retrieval-augmented generation for knowledge-intensive nlp
tasks.Advances in Neural Information Processing Systems33 (2020), 9459–9474.
[27] Yiming Lin, Madelon Hulsebos, Ruiying Ma, Shreya Shankar, Sepanta Zeigham,
Aditya G Parameswaran, and Eugene Wu. 2024. Towards Accurate and Ef-
ficient Document Analytics with Large Language Models.arXiv preprint
arXiv:2405.04674(2024).
[28] Chunwei Liu, Matthew Russo, Michael Cafarella, Lei Cao, Peter Baile Chen,
Zui Chen, Michael Franklin, Tim Kraska, Samuel Madden, Rana Shahout, et al .
2025. Palimpzest: Optimizing AI-Powered Analytics with Declarative Query
Processing. (2025).
[29] Nelson F Liu, Kevin Lin, John Hewitt, Ashwin Paranjape, Michele Bevilacqua,
Fabio Petroni, and Percy Liang. 2024. Lost in the middle: How language models
use long contexts.Transactions of the Association for Computational Linguistics
12 (2024), 157–173.
[30] Siyi Liu, Kishaloy Halder, Zheng Qi, Wei Xiao, Nikolaos Pappas, Phu Mon Htut,
Neha Anna John, Yassine Benajiba, and Dan Roth. 2025. Towards Long Con-
text Hallucination Detection. InFindings of the Association for Computational
Linguistics: NAACL 2025. 7827–7835.
[31] Yao Lu, Aakanksha Chowdhery, Srikanth Kandula, and Surajit Chaudhuri. 2018.
Accelerating machine learning inference with probabilistic predicates. InSIG-
MOD. 1493–1508.
[32] Qianou Ma, Weirui Peng, Hua Shen, Kenneth Koedinger, and Tongshuang Wu.
2024. What you say= what you want? Teaching humans to articulate require-
ments for LLMs.arXiv preprint arXiv:2409.08775(2024).
[33] Liana Patel, Siddharth Jha, Melissa Pan, Harshit Gupta, Parth Asawa, Carlos
Guestrin, and Matei Zaharia. 2025. Semantic Operators and Their Optimization:
Enabling LLM-Based Data Processing with Accuracy Guarantees in LOTUS.Proc.
VLDB Endow.(2025). https://doi.org/10.14778/3749646.3749685
[34] Sundar Pichai and Demis Hassabis. 2024. Our next-generation model: Gemini
1.5. https://blog.google/technology/ai/google-gemini-next-generation-model-
february-2024/. Accessed: 2025-04-03.
[35] Varshini Reddy, Rik Koncel-Kedziorski, Viet Dac Lai, Michael Krumdick, Charles
Lovering, and Chris Tanner. 2024. DocFinQA: A Long-Context Financial Rea-
soning Dataset. InProceedings of the 62nd Annual Meeting of the Association for
Computational Linguistics (Volume 2: Short Papers), Lun-Wei Ku, Andre Martins,
and Vivek Srikumar (Eds.). Association for Computational Linguistics, Bangkok,
Thailand, 445–458. https://doi.org/10.18653/v1/2024.acl-short.42
[36] Francisco Romero, Johann Hauswald, Aditi Partap, Daniel Kang, Matei Zaharia,
and Christos Kozyrakis. 2022. Optimizing video analytics with declarative model
relationships.Proceedings of the VLDB Endowment16, 3 (2022), 447–460.
[37] Shreya Shankar, Aditya G Parameswaran, and Eugene Wu. 2024. DocETL:
Agentic Query Rewriting and Evaluation for Complex Document Processing.
arXiv preprint arXiv:2410.12189(2024).
[38] Qwen Team. 2024. Extending the Context Length to 1M Tokens! https://qwenlm.
github.io/blog/qwen2.5-turbo/. Accessed: 2025-04-03.
[39] Qwen Team. 2025. Qwen3 Technical Report. arXiv:2505.09388 [cs.CL] https:
//arxiv.org/abs/2505.09388
[40] NIPPON TELEGRAPH and TELEPHONE CORPORATION. 2024. pg_hint_plan:
Extension adding support for optimizer hints in PostgreSQL. https://github.com/
ossc-db/pg_hint_plan. Accessed: 2025-04-03.
[41] Cunxiang Wang, Ruoxi Ning, Boqi Pan, Tonghui Wu, Qipeng Guo, Cheng Deng,
Guangsheng Bao, Xiangkun Hu, Zheng Zhang, Qian Wang, et al .[n. d.]. NovelQA:
Benchmarking Question Answering on Documents Exceeding 200K Tokens. In
The Thirteenth International Conference on Learning Representations.
[42] Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi,
Quoc V Le, Denny Zhou, et al .2022. Chain-of-thought prompting elicits reason-
ing in large language models.Advances in neural information processing systems
13

35 (2022), 24824–24837.
[43] Hongshen Xu, Zichen Zhu, Situo Zhang, Da Ma, Shuai Fan, Lu Chen, and Kai
Yu. 2024. Rejection Improves Reliability: Training LLMs to Refuse Unknown
Questions Using RL from Knowledge Feedback. InFirst Conference on Language
Modeling. https://openreview.net/forum?id=lJMioZBoR8
[44] Howard Yen, Tianyu Gao, Minmin Hou, Ke Ding, Daniel Fleischer, Peter Izsak,
Moshe Wasserblat, and Danqi Chen. 2025. HELMET: HOW TO EVALUATE
LONG-CONTEXT LANGUAGE MODELS EFFECTIVELY AND THOROUGHLY.
In13th International Conference on Learning Representations, ICLR 2025. Interna-
tional Conference on Learning Representations, ICLR, 3473–3524.[45] Yusen Zhang, Ruoxi Sun, Yanfei Chen, Tomas Pfister, Rui Zhang, and Sercan Arik.
2024. Chain of agents: Large language models collaborating on long-context tasks.
Advances in Neural Information Processing Systems37 (2024), 132208–132237.
[46] Zihan Zhou, Chong Li, Xinyi Chen, Shuo Wang, Yu Chao, Zhili Li, Haoyu Wang,
Qi Shi, Zhixing Tan, Xu Han, Xiaodong Shi, Zhiyuan Liu, and Maosong Sun.
2025. LLM×MapReduce: Simplified Long-Sequence Processing using Large
Language Models. InProceedings of the 63rd Annual Meeting of the Association
for Computational Linguistics (Volume 1: Long Papers), Wanxiang Che, Joyce
Nabende, Ekaterina Shutova, and Mohammad Taher Pilehvar (Eds.). Association
for Computational Linguistics, Vienna, Austria, 27664–27678. https://doi.org/
10.18653/v1/2025.acl-long.1341
14