# Know Your Source: A Public Knowledge Store for Media Background Checks

**Authors**: Benjamin Nichols, Michael Schlichtkrull, Nedjma Ousidhoum

**Published**: 2026-07-02 16:20:28

**PDF URL**: [https://arxiv.org/pdf/2607.02383v1](https://arxiv.org/pdf/2607.02383v1)

## Abstract
LLM-based retrieval-augmented generation (RAG) is increasingly used for automated fact-checking (AFC) and related tasks. By grounding LLM outputs in retrieved evidence, RAG-based systems provide transparent justifications while allowing external information to be updated independently of the underlying model. However, existing approaches often assume retrieved evidence is reliable, although real-world information may be conflicting, outdated, and can originate from unreliable or biased sources. Recent work on *source-critical reasoning* addresses this challenge through media background checks (MBCs) (Schlichtkrull, 2024), which assess the credibility of evidence sources to support downstream fact verification. However, generating MBCs relies on costly proprietary search APIs, limiting reproducibility. To mitigate this issue, we introduce MEDIAREF, a publicly available knowledge store of web-sourced documents that enables reproducible, low-cost evaluation of MBC generation across 200 media sources. We describe a reproducible methodology for constructing and updating the collection, assess widely used LLMs on the MBC generation task, and demonstrate that MEDIAREF supports higher-quality MBC generation through both automatic and qualitative evaluation.

## Full Text


<!-- PDF content starts -->

Know Your Source:
A Public Knowledge Store for Media Background Checks
Benjamin Nichols*, Michael Schlichtkrull, Nedjma Ousidhoum#
#Cardiff University
Correspondence:OusidhoumN@cardiff.ac.uk
Abstract
LLM-based retrieval-augmented generation
(RAG) is increasingly used for automated fact-
checking (AFC) and related tasks. By ground-
ing LLM outputs in retrieved evidence, RAG-
based systems provide transparent justifica-
tions while allowing external information to
be updated independently of the underlying
model. However, existing approaches often as-
sume retrieved evidence is reliable, although
real-world information may be conflicting,
outdated, and can originate from unreliable
or biased sources. Recent work onsource-
critical reasoningaddresses this challenge
through media background checks (MBCs)
(Schlichtkrull, 2024), which assess the cred-
ibility of evidence sources to support down-
stream fact verification. However, generat-
ing MBCs relies on costly proprietary search
APIs, limiting reproducibility.
To mitigate this issue, we introduce MEDI-
AREF,1a publicly available knowledge store
of web-sourced documents that enables re-
producible, low-cost evaluation of MBC gen-
eration across 200 media sources. We de-
scribe a reproducible methodology for con-
structing and updating the collection, assess
widely used LLMs on the MBC generation
task, and demonstrate that MEDIAREFsup-
ports higher-quality MBC generation through
both automatic and qualitative evaluation.
1 Introduction
Scrutinising evidence sources is central to assess-
ing the reliability of claims and identifying is-
sues such as misleading statistics, omitted con-
text, or unsupported statements, particularly in
high-stakes domains such as journalism, public
health, and policymaking (Graves and Amazeen,
2019; Warren et al., 2025). However, manu-
ally evaluating the evidence underlying a claim
*Work done while at Cardiff University
1MEDIAREFis available athttps://github.com/
nedjmaou/mediarefis time-consuming and resource-intensive, espe-
cially given the growing volume of online infor-
mation (Allcott and Gentzkow, 2017). For in-
stance, verifying a single claim can require a full
day of work for a journalist, while more complex
investigations may take several days (Hassan et al.,
2015). Consequently, automated fact-checking
(AFC) has emerged as an active area of research
(Guo et al., 2022), with approaches ranging from
verification over fixed corpora to open-domain
systems that dynamically retrieve evidence from
the web.
Recent advances in LLM-based retrieval-
augmented generation (RAG) systems have fur-
ther broadened the scope of AFC (Lewis et al.,
2020; Schlichtkrull et al., 2024). However, re-
search on the trustworthiness of LLM-based sys-
tems has largely focused on issues such as prompt
security, adversarial attacks, and hallucinations
(Vassilev et al., 2025; Lent et al., 2025). Com-
paratively less attention has been paid to the re-
liability of the external information sources on
which retrieval-based systems depend. That is, re-
trieved evidence is often treated at face value, de-
spite the possibility that sources may be conflict-
ing, outdated, incomplete, or intentionally mis-
leading (Hong et al., 2024; Ge et al., 2025). This
raises the risk ofattacks-by-content(Schlichtkrull,
2025), whereby misleading or manipulative infor-
mation propagates through retrieval pipelines and
influences downstream model outputs. In the con-
text of AFC, one promising direction for explicitly
considering source credibility is the use ofmedia
background checks(MBCs) (Schlichtkrull, 2024).
MBCs are summaries of aspects that may affect
perceptions of a source’s trustworthiness, includ-
ing political orientation, factual reliability, edito-
rial practices, and ownership (e.g. Figure 1). By
providing contextual information about evidence
sources, MBCs can help both humans and auto-
mated systems make more informed judgementsarXiv:2607.02383v1  [cs.CL]  2 Jul 2026

Source: https://www.rebelnews.com/HistoryFounded in February 2015 by former Sun News Network personalities Ezra Levant and Brian Lilley, The Rebel is a conservative-leaning media website that publishes news, opinions, and videos. The current editor is Ezra Levant. Gavin McInnes, the founder of the far-right neo-fascist organization Proud Boys, was also a contributor and subsequently let go after controversy and brought back in 2019. [...]Funded by / OwnershipThe Rebel News Network Ltd. owns Rebel News and generates revenue through advertising, fee-based premium subscriptions, and donations  Ezra Levant serves as the director of the Rebel News Network Ltd.Analysis / BiasRebel News routinely publishes news with loaded emotional wording that favors the right, such as this: Why Trudeau’s Liberals are pretending THIS gun control study doesn’t exist, and this Cops bundle up angry left-wing student in taped meltdown. Both of these stories are linked to strongly right-leaning sources such as Campus Reform. [...] When it comes to sourcing, The Rebel does hyperlink to outside sources, but they often attempt to source within their domain to increase web traﬃc. Finally, throughout the Covid-19 pandemic, they have repeatedly made false and misleading claims, [...]Failed Fact ChecksFalse – Bill Gates is calling for the mandatory vaccination of all people [...]Misleading – mRNA Covid-19 vaccines increase the chances of infection with the Omicron variant Figure 1: An example media background check (MBC) for the Rebel News outlet, adapted from Media Bias/Fact
Check. The full text is available athttps://mediabiasfactcheck.com/the-rebel/.
about retrieved information. However, existing
approaches to MBC generation typically rely on
proprietary search engine APIs to retrieve source
information at inference time, introducing finan-
cial costs, limiting reproducibility, and creating
variability as search engine rankings and retrieval
systems evolve.
To overcome these limitations, we introduce
MEDIAREF, a collection of web-sourced docu-
ments covering 200 news sources that supports
MBC generation. By decoupling information re-
trieval from MBC generation, MEDIAREFpro-
vides a reproducible, low-cost alternative to pro-
prietary search APIs while reducing variability
arising from changes in retrieval behaviour over
time. We describe a systematic and reproducible
methodology for constructing the resource, en-
abling the collection to be kept up to date as
web content evolves. We use MEDIAREFto
evaluate the MBC-generation capabilities of sev-
eral LLMs and examine the impact of additional
source-related evidence on generation quality. We
find that access to MEDIAREFproduces MBCs
that contain more accurate information about news
sources without increasing the incidence of mis-
leading content. We further develop a qualita-
tive evaluation framework for MBCs based on four
criteria—clarity,relevance,informativeness, and
verifiability—motivated by users’ need to assessthe trustworthiness of MBCs. This analysis com-
plements prior work on the usefulness of MBCs
(Schlichtkrull, 2024) by identifying, at a finer level
of granularity, which characteristics contribute to
their utility and when MBCs may fall short across
these dimensions.
2 Related Work
AFC is typically divided into three stages:claim
detection, which identifies claims to fact-check;
evidence retrieval, which gathers information to
support or refute a claim; andclaim verifica-
tion, which determines the overall truthfulness of
a claim (Guo et al., 2022). Earlier AFC datasets
(Thorne et al., 2018; Jiang et al., 2020) relied on
evidence from a single closed source, typically
Wikipedia, assuming the source to be trustworthy.
However, this risks propagating biases present in
the underlying source (Baly et al., 2018). More
recent work instead retrieves online evidence dy-
namically during model runtime using search
APIs (Schlichtkrull et al., 2023a). Further, con-
temporary systems use retrieval-augmented gen-
eration (RAG) to retrieve evidence and gener-
ate human-understandablejustificationsthat ex-
plain how the retrieved evidence supports a ver-
dict (Schlichtkrull et al., 2024). Professional
fact-checkers report that such justifications en-
able audiences to scrutinise the reasoning process,

thereby promoting trust (Warren et al., 2025).
Despite these advances, dynamically retrieving
online evidence remains constrained by the fi-
nancial cost of search APIs (Schlichtkrull et al.,
2024), potentially limiting both future research
and practical adoption by resource-constrained
fact-checking organisations (Nyariki, 2025). To
reduce this barrier, shared tasks such as A VeriTeC
(Schlichtkrull et al., 2024), A VeriTeC 2.0 (Akhtar
et al., 2025), and A VerImaTeC (Cao et al., 2026)
have provided participants with stores of pre-
retrieved online evidence, avoiding the need for
costly API calls. Participating teams have reported
additional benefits, including improved repro-
ducibility and robustness against the unexpected
disappearance of online information over time
(Rothermel et al., 2024). However, retrieved ev-
idence is often treated ascrediblebased on proxy
signals such as search ranking position, despite no
guarantee that highly ranked sources are free from
misinformation. If left unresolved, conflicting ev-
idence can substantially degrade the performance
of RAG systems (Hong et al., 2024; Ge et al.,
2025). To address this issue, Schlichtkrull (2024)
introducedMedia Background Checks(MBCs) for
AFC. They proposed a two-stage RAG pipeline
in which evidence describing characteristics of a
source associated with (in)credibility is first re-
trieved and then used by an LLM to generate an
MBC. Building on this work, Ge et al. (2025) ex-
plored how MBCs could be integrated into AFC
pipelines and examined their impact on the fact-
checking process through qualitative analysis. In
this paper, we build on this line of work by in-
troducing MEDIAREF, a curated knowledge store
for media background checks that enables repro-
ducible evidence access across multiple LLMs
without reliance on live search APIs. We further
propose a qualitative analysis that goes beyond as-
sessing the utility of MBCs alone (Schlichtkrull,
2024), and instead examines fine-grained charac-
teristics that affect their perceived utility.
3 MEDIAREFCreation
3.1 Media Background Checks
Media background checks (MBCs) (e.g., Figure 1)
are textual descriptions of media outlet character-
istics that may affect the credibility of their report-
ing. These characteristics include, for example, an
outlet’s history, ownership, political bias, and prior
fact-checking record. MBCs are supported by
QueriesWho owns “Rebel News”?Retrieved DocumentsUpdated MBCInitial MBCFinal MBCQueriesSupporting EvidenceKnowledge StoreSource: https://www.rebelnews.com/HistoryFounded in February 2015 by former Sun News Network personalities Ezra Levant and Brian Lilley, The Rebel is a conservative-leaning media website that publishes news, opinions, and videos. The current editor is Ezra Levant. Gavin McInnes, the founder of the far-right neo-fascist organization Proud Boys, was also a contributor and subsequently let go after controversy and brought back in 2019. [...]Funded by / OwnershipThe Rebel News Network Ltd. owns Rebel News and generates revenue through advertising, fee-based premium subscriptions, and donations  Ezra Levant serves as the director of the Rebel News Network Ltd.Analysis / BiasRebel News routinely publishes news with loaded emotional wording that favors the right, such as this: Why Trudeau’s Liberals are pretending THIS gun control study doesn’t exist, and this Cops bundle up angry left-wing student in taped meltdown. Both of these stories are linked to strongly right-leaning sources such as Campus Reform. [...] When it comes to sourcing, The Rebel does hyperlink to outside sources, but they often attempt to source within their domain to increase web traﬃc. Finally, throughout the Covid-19 pandemic, they have repeatedly made false and misleading claims, [...]Failed Fact ChecksFalse – Bill Gates is calling for the mandatory vaccination of all people [...]Misleading – mRNA Covid-19 vaccines increase the chances of infection with the Omicron variant KnowledgeStoreExtracted EvidenceRebel News owner Ezra Levant…Retrieved DocumentsIn Rebel News Network Ltd. v. Al Jazeera Media Network […]Conservative Leader Pierre Poilievre has documented historic ties to Rebel News owner Ezra Levant, […]David Menzies, a commentator for the online site, was arrested Monday by […]Initial MBCFrom LLM internal knowledgeUpdated MBCLLM incorporates extracted evidenceFinal MBCEnriched with all extracted evidence
-The icon represents an LLM when generating the initial MBC and when updating it with information from the knowledge store-In an initial MBC, the LLM describes relevant characteristics for the news source (e.g. history, funding…) using only its internal knowledge of the news source.-When updating an MBC, the LLM is given the extracted evidence snippets and adds new information as additional bulletpoints in the MBC.-The model produces the final MBC when it has used all of the evidence extracted from the knowledge store to update the MBC.Figure 2:MBC generation method for a given news
source using external information.We use targeted
queries to retrieve relevant documents, and a question-
answering model extracts passages of supporting evi-
dence. An LLM generates an initial background check
for the news source and updates it by incorporating the
supporting evidence found.
URLs linking to external evidence sources, such
as relevant news articles or other MBCs, which
readers can consult to assess their reliability. Be-
yond supporting human readers, MBCs provide a
signal of source credibility that can be incorpo-
rated into downstream systems, such as automated
fact-checking models that rely on web-sourced in-
formation, helping them prioritise trustworthy ev-
idence over misinformation.
3.2 Dataset Source
We use Media Bias/Fact-Check (MB/FC)2as a
source of gold-standard MBCs. MB/FC is an inde-
pendent US-based organisation that aims to equip
news consumers to identify and defend against
misinformation by“understanding the bias and
credibility of the sources they consume”. Their
MBCs are written by volunteers following a pub-
lished methodology3and cover outlets ranging
from small local newspapers to international or-
ganisations. While MB/FC is frequently used as a
ground-truth source in studies of source credibility
(Baly et al., 2019; Nakov et al., 2024), these stud-
ies generally use only the credibility and political
bias ratings provided by MB/FC, rather than the
MBC text itself. Hence, we use the MB/FC dataset
released by Schlichtkrull (2024), from which we
randomly sample 200 outlets from the test split.
3.3 Online Evidence Retrieval
To gather background information for each news
outlet required to generate an MBC, we con-
2https://mediabiasfactcheck.com
3https://mediabiasfactcheck.com/methodology/

struct a set of search queries targeting key as-
pects of a news outlet’s background, including
funding, ownership, bias, endorsements, and gen-
eral descriptions. The queries follow the pattern
“source” news <topic>, wheresourceis re-
placed by the outlet name and<topic>is one
offunding,about,ownership,endorsement,
bias. To minimise off-topic results, all queries
include the keywordnewsand require an exact
match to the outlet name. For each outlet, we then
submit all queries to the Google Search API using
default parameters and collect the top 10 results
per query. Our retrieval process can be replicated
to obtain more recent sources.
Quality ControlTo ensure sufficient coverage
for reproducing the gold MBCs provided by
Schlichtkrull (2024), we remove duplicate URLs
and additionally include all external URLs cited
in the gold MBC texts, as in Schlichtkrull et al.
(2024), merging them with the retrieved URLs.
Furthermore, to reduce the risk of data leakage, we
remove URLs pointing to webpages likely to refer-
ence the gold MBCs. We exclude all URLs match-
ing a blacklist of websites known to cite MB/FC
content, in line with Schlichtkrull (2024).
We then scrape the remaining webpages using
the open-source web-scraping tooltrafilatura
(Barbaresi, 2021). This results in 21,921 non-
empty documents, which form the final knowledge
store.
4 Experimental Setup
We investigate the impact of using MEDIAREFas
a source of evidence for LLM-generated MBCs.
We experiment with models from OpenAI (Rad-
ford et al., 2019; OpenAI et al., 2024; Singh
et al., 2025), Qwen (Yang et al., 2024; Team,
2025), Llama (AI@Meta, 2024), Mistral (Jiang
et al., 2023), and Anthropic (Anthropic, 2024).
This selection includes both state-of-the-art pro-
prietary systems and smaller open-source alterna-
tives that are more accessible to researchers and
fact-checking practitioners, allowing us to assess
current LLM capabilities on the MBC generation
task.
4.1 Models
We compare MBCs generated with and without in-
formation retrieval and evaluate the resulting dif-
ferences in quality using two prompting strategiesdescribed below. The full prompts are provided in
Appendix A.
Prompting Without Information Retrieval
An MBC is generated in a single LLM call
without access to external information. Given the
name of a news outlet, the model is prompted in
a zero-shot setting to generate a brief, itemised
summary of information associated with the
outlet.
Prompting With Information Retrieval (+IR)
First, an initial MBC is generated using the zero-
shot prompting strategy described above. The
MBC is then refined using evidence retrieved from
MEDIAREFthrough the following process.
(1) Retrieval.Relevant documents are retrieved
using BM25 (Robertson et al., 1994). We define
six keyword-based queries targeting recurring top-
ics in MBCs, such as outlet history and funding
(see Appendix B), and retrieve the top 30 highest-
scoring documents for each query.
(2) Evidence Extraction.The retrieved doc-
uments are filtered to identify passages likely
to contain credibility-relevant information. Each
keyword query is reformulated as a question; for
example,“source name” fundingbecomesHow
is “source name” funded?. A DeBERTa-based
question-answering model (He et al., 2021) is then
used to extract answers from the retrieved docu-
ments. If an answer is found, the corresponding
text span is retained as evidence.
(3) MBC Update.The extracted evidence is
used to update the initial MBC. Document snip-
pets are processed individually, and the MBC is
revised iteratively after each snippet to reduce the
risk of exceeding the model’s context window.
The model is instructed not to remove existing
points, thereby minimising information loss.
4.2 Evaluation Metrics
To compare generated and gold-standard MBCs
(Schlichtkrull, 2024), we use the following met-
rics:
•ROUGE-Lwhich evaluates the longest com-
mon subsequence of words between the gen-
erated and reference texts (Lin, 2004).
•METEORwhich uses stemming and syn-
onym matching to evaluate word-level over-
lap (Banerjee and Lavie, 2005).
•Fact RecallandError Rate(Schlichtkrull,
2024) based on the FActScore framework
(Min et al., 2023). A GPT-3.5-Turbo

model decomposes a gold-standard MBC
intoatomic factsusing a gap-filling template
(e.g.The usual audience of “source name” is
__; see Appendix C). It then verifies whether
each extracted fact is entailed, contradicted,
or neither by both the generated and gold-
standard MBC texts.
– Fact Recallis the proportion of atomic
facts for which the generated MBC and
the gold-standard MBCagreeon entail-
ment.
– Error Rateis the proportion of atomic
facts for which the generated MBC and
the gold-standard MBCdisagreeon en-
tailment.
5 Results
5.1 Automatic Evaluation
We report the experimental results in Table 1.
Usinggpt-3.5-turbo, our system achieves fact
recall and error rate scores comparable to the
baseline results reported by Schlichtkrull (2024)
on the same model (28.2% vs. 26.1% fact re-
call and 8.1% vs. 6.3% error rate, respectively).
This suggests that using MEDIAREFprovides
a reliable proxy for model performance on re-
cent evidence. Among the evaluated models,
gpt-4o-miniachieves the highest fact recall
score, followed byllama-3.3.
Overall, the results suggest that MBC genera-
tion remains challenging, with relatively low fact
recall across all models. Notably, state-of-the-art
systems such asgpt-5-minido not substantially
outperform smaller open-source models, includ-
ingllama-3.3andmistral-7b, in accurately re-
flecting facts about news sources. Error rates are
also broadly similar, althoughgpt-5-miniex-
hibits a noticeably lower error rate than the other
OpenAI models. According to Singh et al. (2025),
GPT-5was trained to reduce factual errors in RAG
settings, for example by minimising hallucina-
tions, which may explain this improvement. Im-
portantly, higher fact recall does not necessarily
correspond to lower error rates, highlighting the
need to evaluate both the amount of correct and
incorrect information generated. That is, focus-
ing exclusively on factual content risks overlook-
ing errors that may subsequently distort credibility
judgements about news sources.
Introducing information retrieval generally im-
proves fact recall, indicating that evidence re-trieved from the knowledge store helps extend a
model’s knowledge of a news source. However,
the magnitude of this gain, as well as changes
in token count, vary substantially across LLMs.
Since evidence retrieval is independent of the
choice of LLM, these differences suggest that
models differ in how effectively they incorporate
retrieved information into their MBCs. We qual-
itatively investigate the potential causes of this
variation in the next section.
5.2 Human Evaluation of MBC Generation
Quality
We complement our automatic evaluation with a
human evaluation to better understand why mod-
els may struggle to reflect the information in the
gold-standard references. In addition, since MBCs
are intended to support source criticism, we assess
whether human readers perceive them as provid-
ing useful and verifiable information.
We define four criteria for evaluation:clar-
ity,relevance,informativeness, andverifiability.
The first three are adapted from the criteria for
fact-checking questions proposed by Ousidhoum
et al. (2022). We introduceverifiabilityto capture
whether an MBC helps users assess credibility by
pointing to additional trustworthy sources (Warren
et al., 2025). All criteria are rated on a Likert scale
from 0–3, except for clarity, which is rated from
0–2. A full description of the annotation scheme
is provided in Appendix D.
ClarityIndividual points should be comprehen-
sible both on their own and in the context of the
full text, and the MBC should not contain incon-
sistencies or self-contradictions.
RelevanceAll facts mentioned in the MBC
should relate to the target news source or closely
connected entities and topics.
InformativenessThe MBC should provide in-
formation that meaningfully influences a reader’s
perception of a source’s overall credibility.Highly
informativebackground checks directly indicate
the source’s credibility and political bias (or ab-
sence of bias), and justify why.Weakly informa-
tivechecks only describe characteristics loosely
relevant to source credibility, such as general
background information, with unclear implica-
tions for credibility.
VerifiabilitySufficient evidence should be pro-
vided to support factual claims, or enough detail

Models Fact Recall Error Rate METEOR ROUGE-L #Tokens
gpt-3.5-turbo-012526.86% 8.21% 12.06% 13.66% 124.1
gpt-3.5-turbo-0125 + IR28.24% 8.13% 15.95% 14.21% 200.5
gpt-4o-mini28.33% 8.29% 15.82% 12.37% 248.5
gpt-4o-mini + IR 29.85%8.96% 18.36% 12.61% 342.4
gpt-5-mini24.78% 4.80% 18.20% 11.31% 324.2
gpt-5-mini + IR27.83% 5.48% 23.89% 11.35% 683.3
qwen-2.5-72b-instruct24.45% 6.90% 10.79% 9.94% 175.5
qwen-2.5-72b-instruct + IR24.62% 7.58% 16.82% 11.09% 393.3
qwen-3-32b-instruct24.91% 9.78% 16.71% 11.22% 294.1
qwen-3-32b-instruct + IR26.22% 9.79% 19.35% 11.12% 441.3
llama-3.3-70b-instruct28.98% 9.66% 9.67% 10.57% 115.2
llama-3.3-70b-instruct + IR 29.03%9.76% 16.90% 12.11% 321.5
mistral-7b-instruct-v0.328.07% 10.54% 22.28% 15.57%383.7
mistral-7b-instruct-v0.3 + IR27.49% 10.44% 22.69% 14.67%480.8
claude-3.5-haiku26.66% 7.32% 10.53% 9.95% 160.2
claude-3.5-haiku + IR26.95% 7.15% 12.30% 10.14% 214.6
Table 1:Results for MBC generation using LLMs.Metrics are reported for MBCs generated both without
and with information retrieval (IR) using MEDIAREF, along with the average token count of generated MBCs
(#Tokens) per model. For error rate, lower values are better; for all other metrics, higher values are better. Best
scores are shown in blue, and second-best scores in light blue. Overall, MBCs generated with IR outperform those
without IR across all metrics except error rate, where performance remains similar or degrades slightly.
should be included so that one can reasonably lo-
cate the evidence independently. This should be
determined based solely on judgements of how
easily the facts could plausibly be verified (i.e.,
without the use of a search engine or an LLM).
5.3 Human Evaluation Setup
We randomly sample 108 LLM-generated
MBCs (27 per model) from thegpt-4o-mini,
gpt-5-mini,llama-3.3-70b-instruct, and
mistral-7b-instructIR-augmented models
(i.e., +IR in Table 1) to investigate how in-
formation retrieved from the knowledge store
affects the quality of the generated MBCs. The
LLMs represent both state-of-the-art proprietary
systems and open-source models across a range
of parameter sizes and performance levels.
Two expert annotators (authors of this paper)
qualitatively assess the clarity, relevance, infor-
mativeness, and verifiability of the MBCs. The
identity of the system that generated each MBC
is concealed from the annotators, and they are not
permitted to use external sources such as search
engines or LLMs to obtain additional background
information about the MBCs.
Agreement ScoresThe Fleiss-κscores for clar-
ity, relevance, informativeness, and verifiabil-ity are 0.51, 0.88, 0.87, and 0.79, respectively.
The Krippendorff-αscores are 0.52, 0.98, 0.95,
and 0.91, respectively. However, as Randolph
(2010) demonstrates, measures such as Fleiss-κ
assume that the distribution of assignments across
score categories is fixed in advance (i.e., chance-
adjusted), which is often unrealistic in practice.
We therefore also report free-marginal multi-rater
kappa scores, which are 0.90, 0.93, 0.88, and 0.81
for clarity, relevance, informativeness, and verifia-
bility, respectively.
Overall, agreement scores are high, with infor-
mativeness and verifiability being slightly lower.
This is expected, as these two criteria are inher-
ently more subjective and depend on a reader’s in-
terpretation of how an MBC affects their under-
standing of a news source.
6 Analysis
6.1 To what extent do generated MBCs
capture source credibility?
Overall, the results of our qualitative analy-
sis are consistent with those of our automatic
evaluation. Specifically,gpt-4o-miniperforms
best at generating fully relevant MBCs, while
llama-3.3-70b-instructproduces the most in-
formative and verifiable outputs. In contrast,

Figure 3: Qualitative analysis results for Clarity, Relevance, Informativeness, and Verifiability. Colours encode
score magnitude (low→high, i.e., 0→3), consistently across all models. Note that the maximum Clarity score is
2, and the values shown are the average scores assigned by our two expert annotators.
gpt-5-miniperforms worst in terms of rele-
vance, informativeness, and verifiability. Open-
source models nevertheless remain competitive
with state-of-the-art proprietary models.
Figure 3 shows that all models generally pro-
duce clear and relevant MBCs. However, gener-
ating informative and verifiable content is signif-
icantly more challenging. While entirely uninfor-
mative or unverifiable MBCs (i.e., those assigned
a score of 0) are rare, the scores indicate that infor-
mativeness and verifiability are harder to achieve
than clarity and relevance. In particular, verifiable
points must first be clear, while informative points
must be both clear and relevant.
For example, consider the following excerpt
from agpt-4o-mini-generated MBC forDaily
Surge:
“Generally considered to have a conservative bias;
aligns with right-leaning perspectives on political
and social issues.”
Although the model identifies a potentially
right-leaning political bias, it does not specify
whichright-leaning perspectives the source aligns
with, making the claim difficult to verify.
This observation is further supported by the re-
sults shown in Figure 4, which illustrate the Spear-
man correlation scores between the different crite-
ria. Notably, we find that informative MBCs tend
to be verifiable (ρ= 0.83), suggesting that pro-
viding specific evidence in an MBC may supply
sufficient detail to bridge gaps in a reader’s percep-
tion of the source, thereby improving verifiability
scores.
6.2 Challenges in updating MBCs
Insufficient InformationWhen models cannot
source information about a news outlet either in-
ternally or from the knowledge store, they tend togenerate weakly informative background descrip-
tions or no useful information at all. For exam-
ple,mistral-7b-instructdescribesThe Jack-
son Sunin a way that is too vague and generic
to meaningfully inform an assessment of source
credibility:
“...like many local newspapers, [striving] to maintain
a balanced approach to news reporting. However,
like any media outlet, it may have editorial leanings
that are not always immediately apparent.”
Local newspapers appear to be particularly sus-
ceptible to this issue. Among the 34 local news
outlets in our sample, 71% received an informa-
tiveness score of at most 1, while 80% received
a verifiability score of at most 1. In many cases,
the generated MBCs contained only basic back-
ground information similar to the example above.
This may be due to the fact that these outlets often
lack the prominence required to attract substan-
tial external scrutiny, such as independent credi-
bility assessments or fact-checking coverage, de-
spite potentially being trustworthy sources for lo-
cal issues. Consequently, models may be un-
able to provide evidence either supporting or chal-
lenging their credibility. For instance, the same
mistral-7b-instruct-generated MBC includes
factually plausible but largely uninformative state-
ments about the outlet’s credibility:
“There is no public record of The Jackson Sun fail-
ing fact-checks by reputable fact-checking organisa-
tions.”
Inconsistent Emphasis on EvidenceModels
differ substantially in the extent to which they
interpret or evaluate the evidence they retrieve.
In 8 of 27 MBCs,gpt-5-minidraws inferences
about the relevance of evidence to source credi-
bility, including unprompted reasoning about the

Figure 4: Spearman correlations between MBC clarity,
relevance, informativeness, and verifiability scores.
source itself. By contrast, the other models rarely
include content evaluating whether the avail-
able evidence sufficiently supports a conclusion
(across 27 MBCs:llama-3.3-70b-instruct
andmistral-7b-instructdo so in one instance
each, andgpt-4o-miniin none). For example, for
theSource New Mexiconews outlet,gpt-5-mini
generates:
“The excerpt you provided lists many Democratic
candidates and endorsers, but that listing alone does
not prove an editorial endorsement or a consistent
partisan tilt by the outlet.”
Providing readers with information about the
quality and limitations of the available evidence
may help them form a more nuanced assess-
ment of a source’s credibility and is broadly
consistent with professional fact-checking prac-
tices. However, such commentary may also
introduce an additional interpretative layer that
risks influencing readers’ perceptions of the
evidence. Note that this additional reason-
ing substantially increases the length of the
MBCs, making them harder to read and under-
stand. Specifically, median MBC word counts
are 458 forgpt-5-mini, compared with 194 for
gpt-4o-mini, 268 formistral-7b-instruct,
and 264 forllama-3.3-70b-instruct.
Information LossIn seven MBCs generated by
mistral-7b-instruct, the model omits informa-
tion from the beginning of the MBC during the
updating process. For example, in the MBC for
KJRH – Tulsa News, the first nine points, which
describe the outlet’s location, affiliation, owner-
ship, and funding, are replaced with:“1–9: Same as previous response.”
As these early points typically contain im-
portant background information about the outlet,
omitting them reduces the informativeness of the
final MBC. This issue could be mitigated by ex-
plicitly instructing the model to regenerate and
verify the complete updated MBC after each re-
vision.
Conflicting InformationWe observe
three cases (out of 108) of explicit con-
tradictions in generated MBCs: two from
llama-3.3-70b-instructand one from
gpt-5-mini. Such contradictions reduce in-
formativeness, as they prevent readers from
drawing reliable conclusions. In two cases,
contradictions arise when models update an
existing MBC with newly retrieved information.
For example, when describing fact-checking
failures associated withWBTS – NBC 10 Boston,
llama-3.3-70b-instructinitially generates
“None notable found”but later adds:
“WBTS - NBC 10 - Boston has been cited as an ex-
ample of a media outlet that has distorted facts, ac-
cording to a series on media bias, although specific
details of the incident are not provided.”
One possible explanation is the prompt design,
which discourages removing previously generated
content during updates. While this helps preserve
earlier information, it may also allow outdated or
superseded statements to remain in the MBC when
later retrieved evidence introduces conflicting in-
formation. This limitation could be addressed by
explicitly prompting the model to revise or remove
statements that are contradicted by newly retrieved
evidence.
7 Conclusion
We introduced MEDIAREF, a publicly available
knowledge store designed to support the genera-
tion of media background checks (MBCs). We de-
scribed a reproducible methodology for construct-
ing and updating the resource, and evaluated sev-
eral widely used LLMs on the MBC generation
task. Our experiments show that MEDIAREFim-
proves the quality of generated MBCs, while our
human evaluation highlights that producing con-
cise, informative, and verifiable MBCs remains an
open challenge. We publicly release MEDIAREF

to support reproducible research on source-critical
reasoning and automated fact-checking.
Limitations
We acknowledge that search-engine rankings may
introduce bias as a proxy for relevance in evidence
selection, potentially affecting the quality and di-
versity of retrieved sources. Nevertheless, our re-
source provides a strong starting point that can be
extended in future work to incorporate more di-
verse and heterogeneous data sources.
Second, although we use a blacklist to filter
sources, some undesired or low-quality sources
referencing MBCs or fact-checks may still be in-
cluded, as no blacklist can be fully comprehensive.
Third, the system is limited to information
freely available on the web and therefore cannot
access restricted or proprietary sources, such as
official statistical databases or paywalled archives,
highlighting an opportunity for future integration
of broader external sources.
Ethical Considerations
Our resource is designed to support media back-
ground checks using publicly available informa-
tion. However, despite the use of retrieval, gen-
erated outputs may still contain inaccuracies or
misrepresentations of source material and should
therefore not be treated as definitive judgments
about the sources, especially in high-stakes set-
tings.
Second, the reliance on web search and pub-
licly available content introduces inherent biases,
including disparities in coverage across regions,
languages, and entities. Such biases may lead to
uneven or incomplete representations in generated
background checks. In addition, while we use fil-
tering mechanisms such as blacklists, we do not
guarantee the complete removal of low-quality or
unreliable sources.
Finally, although the system operates solely
on publicly accessible information, aggregating
and synthesising such data may still raise dual-
use concerns. The intended use of our resource
(Schlichtkrull et al., 2023b) is to support research
on automated fact-checking and potentially assist
fact-checkers with human experts in the loop. We
emphasise the need for cautious deployment and
human oversight in real-world applications.
AI UseWe use privacy-preserving models to as-
sist with proofreading and coding tasks (e.g., gen-erating plots), in line with the ACL guidelines.
References
AI@Meta. 2024. Llama 3 model card.
Mubashara Akhtar, Rami Aly, Yulong Chen, Zhenyun
Deng, Michael Schlichtkrull, Chenxi Whitehouse,
and Andreas Vlachos. 2025. The 2nd automated
verification of textual claims (A VeriTeC) shared
task: Open-weights, reproducible and efficient sys-
tems. InProceedings of the Eighth Fact Extrac-
tion and VERification Workshop (FEVER), pages
201–223, Vienna, Austria. Association for Compu-
tational Linguistics.
Hunt Allcott and Matthew Gentzkow. 2017. Social me-
dia and fake news in the 2016 election.Journal of
Economic Perspectives, 31(2):211–36.
Anthropic. 2024. Model card addendum: Claude 3.5
haiku and upgraded claude 3.5 sonnet.
Ramy Baly, Georgi Karadzhov, Dimitar Alexandrov,
James Glass, and Preslav Nakov. 2018. Predict-
ing factuality of reporting and bias of news media
sources. InProceedings of the 2018 Conference on
Empirical Methods in Natural Language Process-
ing, pages 3528–3539, Brussels, Belgium. Associ-
ation for Computational Linguistics.
Ramy Baly, Georgi Karadzhov, Abdelrhman Saleh,
James Glass, and Preslav Nakov. 2019. Multi-task
ordinal regression for jointly predicting the trustwor-
thiness and the leading political ideology of news
media. InProceedings of the 2019 Conference of
the North American Chapter of the Association for
Computational Linguistics: Human Language Tech-
nologies, Volume 1 (Long and Short Papers), pages
2109–2116, Minneapolis, Minnesota. Association
for Computational Linguistics.
Satanjeev Banerjee and Alon Lavie. 2005. METEOR:
An automatic metric for MT evaluation with im-
proved correlation with human judgments. InPro-
ceedings of the ACL Workshop on Intrinsic and Ex-
trinsic Evaluation Measures for Machine Transla-
tion and/or Summarization, pages 65–72, Ann Ar-
bor, Michigan. Association for Computational Lin-
guistics.
Adrien Barbaresi. 2021. Trafilatura: A web scraping li-
brary and command-line tool for text discovery and
extraction. InProceedings of the 59th Annual Meet-
ing of the Association for Computational Linguistics
and the 11th International Joint Conference on Nat-
ural Language Processing: System Demonstrations,
pages 122–131, Online. Association for Computa-
tional Linguistics.
Rui Cao, Yulong Chen, Zhenyun Deng, Michael
Schlichtkrull, and Andreas Vlachos. 2026. The
automatic verification of image-text claims (A Ver-
ImaTeC) shared task. InProceedings of the

Ninth Fact Extraction and VERification Workshop
(FEVER), pages 74–90, Rabat, Morocco. Associa-
tion for Computational Linguistics.
Ziyu Ge, Yuhao Wu, Daniel Chin, Roy Ka-Wei Lee,
and Rui Cao. 2025. Resolving conflicting evidence
in automated fact-checking: A study on retrieval-
augmented llms.
L Graves and M Amazeen. 2019.Fact-checking as
idea and practice in journalism. Oxford Research
Encyclopedias. Oxford University Press.
Zhijiang Guo, Michael Schlichtkrull, and Andreas Vla-
chos. 2022. A survey on automated fact-checking.
Transactions of the Association for Computational
Linguistics, 10:178–206.
Naeemul Hassan, Bill Adair, James Hamilton,
Chengkai Li, Mark Tremayne, Jun Yang, and Cong
Yu. 2015. The quest to automate fact-checking.
Proceedings of the 2015 Computation + Journalism
Symposium.
Pengcheng He, Xiaodong Liu, Jianfeng Gao, and
Weizhu Chen. 2021. Deberta: Decoding-
enhanced bert with disentangled attention.Preprint,
arXiv:2006.03654.
Giwon Hong, Jeonghwan Kim, Junmo Kang, Sung-
Hyon Myaeng, and Joyce Jiyoung Whang. 2024.
Why so gullible? enhancing the robustness of
retrieval-augmented models against counterfactual
noise. InFindings of the Association for Com-
putational Linguistics: NAACL 2024, pages 2474–
2495, Mexico City, Mexico. Association for Com-
putational Linguistics.
Albert Q. Jiang, Alexandre Sablayrolles, Arthur Men-
sch, Chris Bamford, Devendra Singh Chaplot, Diego
de las Casas, Florian Bressand, Gianna Lengyel,
Guillaume Lample, Lucile Saulnier, Lélio Re-
nard Lavaud, Marie-Anne Lachaux, Pierre Stock,
Teven Le Scao, Thibaut Lavril, Thomas Wang, Tim-
othée Lacroix, and William El Sayed. 2023. Mistral
7b.Preprint, arXiv:2310.06825.
Yichen Jiang, Shikha Bordia, Zheng Zhong, Charles
Dognin, Maneesh Singh, and Mohit Bansal. 2020.
HoVer: A dataset for many-hop fact extraction and
claim verification. InFindings of the Association
for Computational Linguistics: EMNLP 2020, pages
3441–3460, Online. Association for Computational
Linguistics.
Heather Lent, Erick Galinkin, Yiyi Chen, Jens Myrup
Pedersen, Leon Derczynski, and Johannes Bjerva.
2025. NLP security and ethics, in the wild.Transac-
tions of the Association for Computational Linguis-
tics, 13:709–743.
Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio
Petroni, Vladimir Karpukhin, Naman Goyal, Hein-
rich Küttler, Mike Lewis, Wen-tau Yih, Tim Rock-
täschel, et al. 2020. Retrieval-augmented generation
for knowledge-intensive nlp tasks.Advances in neu-
ral information processing systems, 33:9459–9474.Chin-Yew Lin. 2004. ROUGE: A package for auto-
matic evaluation of summaries. InText Summariza-
tion Branches Out, pages 74–81, Barcelona, Spain.
Association for Computational Linguistics.
Sewon Min, Kalpesh Krishna, Xinxi Lyu, Mike Lewis,
Wen-tau Yih, Pang Koh, Mohit Iyyer, Luke Zettle-
moyer, and Hannaneh Hajishirzi. 2023. FActScore:
Fine-grained atomic evaluation of factual precision
in long form text generation. InProceedings of the
2023 Conference on Empirical Methods in Natural
Language Processing, pages 12076–12100, Singa-
pore. Association for Computational Linguistics.
Preslav Nakov, Jisun An, Haewoon Kwak, Muham-
mad Arslan Manzoor, Zain Muhammad Mujahid,
and Husrev Taha Sencar. 2024. A survey on pre-
dicting the factuality and the bias of news media. In
Findings of the Association for Computational Lin-
guistics: ACL 2024, pages 15947–15962, Bangkok,
Thailand. Association for Computational Linguis-
tics.
Enock Nyariki. 2025. State of the fact-
checkers report - 2025.https://www.
poynter.org/wp-content/uploads/2026/
03/2026-State-of-Fact-Checkers-4.pdf.
Accessed: 2026-5-22.
OpenAI, Josh Achiam, Steven Adler, Sandhini Agar-
wal, Lama Ahmad, Ilge Akkaya, Florencia Leoni
Aleman, Diogo Almeida, Janko Altenschmidt, Sam
Altman, et al. 2024. Gpt-4 technical report.
Preprint, arXiv:2303.08774.
Nedjma Ousidhoum, Zhangdie Yuan, and Andreas Vla-
chos. 2022. Varifocal question generation for fact-
checking. InProceedings of the 2022 Conference
on Empirical Methods in Natural Language Pro-
cessing, pages 2532–2544, Abu Dhabi, United Arab
Emirates. Association for Computational Linguis-
tics.
Alec Radford, Jeff Wu, Rewon Child, David Luan,
Dario Amodei, and Ilya Sutskever. 2019. Language
models are unsupervised multitask learners.
Justus Randolph. 2010. Free-marginal multirater
kappa (multiraterκfree): An alternative to fleiss
fixed-marginal multirater kappa. volume 4.
Stephen E. Robertson, Steve Walker, Susan Jones,
Micheline Hancock-Beaulieu, and Mike Gatford.
1994. Okapi at trec-3. InText Retrieval Conference.
Mark Rothermel, Tobias Braun, Marcus Rohrbach, and
Anna Rohrbach. 2024. InFact: A strong baseline
for automated fact-checking. InProceedings of the
Seventh Fact Extraction and VERification Workshop
(FEVER), pages 108–112, Miami, Florida, USA.
Association for Computational Linguistics.
Michael Schlichtkrull. 2024. Generating media back-
ground checks for automated source critical reason-
ing. InFindings of the Association for Compu-
tational Linguistics: EMNLP 2024, pages 4927–

4947, Miami, Florida, USA. Association for Com-
putational Linguistics.
Michael Schlichtkrull, Yulong Chen, Chenxi White-
house, Zhenyun Deng, Mubashara Akhtar, Rami
Aly, Zhijiang Guo, Christos Christodoulopoulos,
Oana Cocarascu, Arpit Mittal, James Thorne, and
Andreas Vlachos. 2024. The automated verification
of textual claims (A VeriTeC) shared task. InPro-
ceedings of the Seventh Fact Extraction and VER-
ification Workshop (FEVER), pages 1–26, Miami,
Florida, USA. Association for Computational Lin-
guistics.
Michael Schlichtkrull, Zhijiang Guo, and Andreas
Vlachos. 2023a. Averitec: A dataset for real-
world claim verification with evidence from the web.
Preprint, arXiv:2305.13117.
Michael Schlichtkrull, Nedjma Ousidhoum, and An-
dreas Vlachos. 2023b. The intended uses of auto-
mated fact-checking artefacts: Why, how and who.
InFindings of the Association for Computational
Linguistics: EMNLP 2023, pages 8618–8642, Sin-
gapore. Association for Computational Linguistics.
Michael Sejr Schlichtkrull. 2025. Attacks by content:
Automated fact-checking is an AI security issue.
InProceedings of the 2025 Conference on Empiri-
cal Methods in Natural Language Processing, pages
8550–8565, Suzhou, China. Association for Compu-
tational Linguistics.
Aaditya Singh, Adam Fry, Adam Perelman, Adam Tart,
Adi Ganesh, Ahmed El-Kishky, Aidan McLaugh-
lin, Aiden Low, AJ Ostrow, Akhila Ananthram,
et al. 2025. Openai gpt-5 system card.Preprint,
arXiv:2601.03267.
Qwen Team. 2025. Qwen3 technical report.Preprint,
arXiv:2505.09388.
James Thorne, Andreas Vlachos, Christos
Christodoulopoulos, and Arpit Mittal. 2018.
FEVER: a large-scale dataset for fact extraction
and VERification. InProceedings of the 2018
Conference of the North American Chapter of
the Association for Computational Linguistics:
Human Language Technologies, Volume 1 (Long
Papers), pages 809–819, New Orleans, Louisiana.
Association for Computational Linguistics.
Apostol Vassilev, Alina Oprea, Alie Fordyce, Hyrum
Andersen, Xander Davies, and Maia Hamin. 2025.
Adversarial machine learning: A taxonomy and ter-
minology of attacks and mitigations. Technical re-
port, National Institute of Standards and Technol-
ogy.
Greta Warren, Irina Shklovski, and Isabelle Augen-
stein. 2025.Show Me the Work: Fact-Checkers’
Requirements for Explainable Automated Fact-
Checking. Association for Computing Machinery,
New York, NY , USA.An Yang, Baosong Yang, Binyuan Hui, Bo Zheng,
Bowen Yu, Chang Zhou, Chengpeng Li, Chengyuan
Li, Dayiheng Liu, Fei Huang, Guanting Dong, Hao-
ran Wei, Huan Lin, Jialong Tang, Jialin Wang, Jian
Yang, Jianhong Tu, Jianwei Zhang, Jianxin Ma, Jin
Xu, Jingren Zhou, Jinze Bai, Jinzheng He, Jun-
yang Lin, Kai Dang, Keming Lu, Keqin Chen,
Kexin Yang, Mei Li, Mingfeng Xue, Na Ni, Pei
Zhang, Peng Wang, Ru Peng, Rui Men, Ruize Gao,
Runji Lin, Shijie Wang, Shuai Bai, Sinan Tan, Tian-
hang Zhu, Tianhao Li, Tianyu Liu, Wenbin Ge,
Xiaodong Deng, Xiaohuan Zhou, Xingzhang Ren,
Xinyu Zhang, Xipin Wei, Xuancheng Ren, Yang
Fan, Yang Yao, Yichang Zhang, Yu Wan, Yunfei
Chu, Yuqiong Liu, Zeyu Cui, Zhenru Zhang, and
Zhihao Fan. 2024. Qwen2 technical report.arXiv
preprint arXiv:2407.10671.

Appendix
A Prompts
We give the prompts used to generate, update,
and evaluate MBCs. They are reproduced from
Schlichtkrull (2024).
System Message:You are InfoHuntGPT, a
world-class AI assistant used by journalists to
quickly build knowledge of new sources.
User Message:Build a background check for the
news source [source name]. Write down every-
thing you know about them, e.g. who funds them,
how they make money, if they have any partic-
ular bias. Make an ITEMIZED LIST. Be brief,
and if you don’t know something, just leave it out.
If you are aware that they have failed any fact-
checks, mention which. Begin your response with
"**Background check**".
Figure 5: Prompt used to generate MBCs without ex-
ternal evidence. [Source name] stands for the name of
the target news source.
Assistant Message:[Previous MBC]
User Message:Google search has revealed some
new information: [New information]
Update your background check for [source name]
using the new information. Do NOT delete any
information, but make ADDITIONS where neces-
sary, using the new information. Most likely, you
will just need to add an extra item to the itemized
list you previously created. Make minimal edits,
and only incorporate what is relevant. Begin your
response with "**Background check**"
Figure 6: Prompt used to update MBCs when provided
with external evidence. [Source name] stands for the
name of the target news source. [New information] rep-
resents the retrieved information to be incorporated.
B Search Queries for Evidence Retrieval
Table 2 gives the search queries used to re-
trieve evidence from MEDIAREFwhen generating
MBCs with information retrieval.
C Atomic Fact Templates
Table 3 gives the 42 atomic facts used to evaluate
the fact recall and error rate metrics.System Message:You are FactCheckGPT, a
world-class tool used by journalists to discover
problems in their writings. Users give you text,
and check whether facts are true given the text.
You ALWAYS answer either TRUE, FALSE, or
NOT ENOUGH EVIDENCE.
User Message:You will be given a snippet
written as part of a source criticism exercise, and
a claim. Your task is to determine whether the
claim is true based ONLY on the text. Do NOT
use any other knowledge source.
The claim is: [question].
The text follows below: [text].
[question]? Thinking step by step, answer either
TRUE, FALSE, or NOT ENOUGH EVIDENCE,
capitalizing all letters. Explain your reasoning
FIRST, and after that output either TRUE, FALSE,
or NOT ENOUGH EVIDENCE.
Figure 7: Prompt used to determine whether an atomic
fact is entailed or contradicted by a generated MBC.
[question] is replaced with the atomic fact template,
and [text] is replaced by the generated MBC.
D Qualitative Annotation Guidelines
The four criteria are defined as follows.
Clarity:Background checks should be clear and
understandable to an average layperson.
•0 – Poor clarity:The MBC’s language
or format severely hinders comprehension.
Most points are unclear in meaning. Exam-
ples include ungrammatical text that impedes
understanding, omitted information neces-
sary to interpret later points, or contradictory
statements regarding source credibility.
•1 – Limited clarity:Most points are under-
standable, but language or formatting occa-
sionally hinders comprehension.
•2 – High clarity:All points are clear and un-
derstandable.
Relevance:All facts mentioned in the check
should relate to the target source or closely con-
nected entities and topics.
•0 – Off-topic:The MBC as a whole is unre-
lated to the target source.
•1 – Partially relevant:Some, but not all,
points refer to unrelated entities or ideas and

# Query Question
1 “source name” ownership Who owns “source name”?
2 “source name” funding How is “source name” funded?
3 “source name” about What is “source name”?
4 “source name” political leaning What is the political leaning of “source name”?
5 “source name” fact-check Has “source name” failed any fact-checks?
6 “source name” retracted article Has “source name” retracted any articles?
Table 2: Queries used to retrieve evidence from MEDIAREF, with corresponding questions used to extract answers
with DeBERTa. Reproduced from Schlichtkrull (2024).
are off-topic.
•2 – Broadly relevant:Some points refer to
entities or ideas not directly related to the
source but still relevant to the broader topic.
•3 – Fully relevant:All points are directly re-
lated to the target source and remain on topic.
Informativeness:Facts about the target source
should influence the reader’s understanding of the
source’s general credibility and provide context
relevant to evaluating claims. Because an MBC
evaluates a source overall rather than a specific
claim, highly specific facts are less useful.
•0 – Uninformative or contradictory:Points
do not indicate source credibility, or conflict-
ing information prevents conclusions from
being drawn confidently.
•1 – Weakly informative:Points describe
characteristics only weakly related to source
credibility (e.g. general background informa-
tion), with unclear implications for credibil-
ity.
•2 – Moderately informative:At least one
point describes source characteristics rele-
vant to credibility, although their implica-
tions are not fully explicit or apply only to
limited claim types. No contradictory points
are present.
•3 – Highly informative:At least one point
directly indicates the credibility and political
bias (or absence of bias) of most claims from
the source and explains why. No contradic-
tory points are present.
Verifiability:Sufficient evidence should be pro-
vided to support factual claims, or enough detail
included that a layperson could reasonably locate
the evidence independently. Annotators should
not conduct additional research and should base
judgements solely on the ease with which facts
could plausibly be verified.•0 – Unverifiable:Points are entirely unveri-
fiable, such as broad subjective assertions.
•1 – Limited verifiability:Points provide
sufficient detail for further investigation,
but omit important contextual information
needed to assess them independently.
•2 – Moderately verifiable:Points are suffi-
ciently detailed to appear credible and be ver-
ifiable by a layperson, although verification
would require some effort.
•3 – Easily verifiable:Specific evidence is
provided that a layperson could quickly and
easily verify.

Type Template
founded_source __ founded "source name"
host __ hosts "source name"
founder_of_source __ is the founder of "source name"
ceo_of_source __ is the ceo of "source name"
leads_source __ leads "source name"
owns_source __ owns "source name"
publishes_source __ publishes "source name"
owner_of_source __ is the owner of "source name"
bought_source __ bought "source name"
acquired_source __ acquired "source name"
source_funded_through "source name" is funded through __
remit The remit of "source name" is to __
usual_audience The usual audience of "source name" is __
awards "source name" was awarded __
source_tends_to "source name" tends to the __
source_other_comments Other sources have commented on "source name", stating that it tends to be __
factuality The factuality of "source name" is reported to be __
article_failed_check "source name" failed a fact-check for an article titled __
article_retraction "source name" printed a retraction after failing a fact-check for an article titled _
topic_not_covered "source name" chose not to cover __
is_original_content "source name" provides original content written by staff journalists
uses_advertising "source name" relies on advertising for revenue
uses_subscriptions "source name" relies on subscriptions for revenue
uses_donations "source name" relies on donations for revenue
donations_from "source name" has received donations from __
aggregation_source When aggregating stories, "source name" relies on information from __
has_endorsed "source name" has endorsed __
editorial_bias "source name" has an editorial bias towards __
editor _is the editor of "source name"
fined_amount_reason "source name" was fined __ for __
biased_article "source name" printed a biased article titled __
misleading_article "source name" printed a factually misleading article titled __
apologized_for "source name" had to apologize for __
damages_to_reason "source name" paid damages to __ after __
pretends_to_be "source name" pretends to be __
unknown_who It is unknown who __
headquarters_location "source name"’s headquarter is located in __
gov_funded "source name" is funded by the __ government
source_is "source name" is a __
uses_peer_review "source name" uses a peer review process
uses_int_checks "source name" uses an internal fact-checking process
covers_topics "source name" covers the following topics: __
Table 3: Templates used to extract atomic facts from MBCs, by filling in the blanks. Reproduced from Schlichtkrull
(2024).