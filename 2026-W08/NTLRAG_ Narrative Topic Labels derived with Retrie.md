# NTLRAG: Narrative Topic Labels derived with Retrieval Augmented Generation

**Authors**: Lisa Grobelscheg, Ema Kahr, Mark Strembeck

**Published**: 2026-02-19 10:07:52

**PDF URL**: [https://arxiv.org/pdf/2602.17216v1](https://arxiv.org/pdf/2602.17216v1)

## Abstract
Topic modeling has evolved as an important means to identify evident or hidden topics within large collections of text documents. Topic modeling approaches are often used for analyzing and making sense of social media discussions consisting of millions of short text messages. However, assigning meaningful topic labels to document clusters remains challenging, as users are commonly presented with unstructured keyword lists that may not accurately capture the respective core topic. In this paper, we introduce Narrative Topic Labels derived with Retrieval Augmented Generation (NTLRAG), a scalable and extensible framework that generates semantically precise and human-interpretable narrative topic labels. Our narrative topic labels provide a context-rich, intuitive concept to describe topic model output. In particular, NTLRAG uses retrieval augmented generation (RAG) techniques and considers multiple retrieval strategies as well as chain-of-thought elements to provide high-quality output. NTLRAG can be combined with any standard topic model to generate, validate, and refine narratives which then serve as narrative topic labels. We evaluated NTLRAG with a user study and three real-world datasets consisting of more than 6.7 million social media messages that have been sent by more than 2.7 million users. The user study involved 16 human evaluators who found that our narrative topic labels offer superior interpretability and usability as compared to traditional keyword lists. An implementation of NTLRAG is publicly available for download.

## Full Text


<!-- PDF content starts -->

NTLRAG: Narrative Topic Labels derived with
Retrieval Augmented Generation
Lisa Grobelschega,b,∗, Ema Kahra, Mark Strembecka
aInstitute for Complex Networks, Vienna University of Economic s and Business (WU
Vienna), Welthandelsplatz 1, Vienna, 1020, Austria
bCAMPUS 02, University of Applied Sciences, Körblergasse 126, 8010, Graz, Austria
Abstract
Topic modeling has evolved as an important means to identify evident or
hidden topics within large collections of text documents. T opic modeling
approaches are often used for analyzing and making sense of s ocial media
discussions consisting of millions of short text messages. However, assigning
meaningful topic labels to document clusters remains chall enging, as users
are commonly presented with unstructured keyword lists tha t may not accu-
rately capture the respective core topic. In this paper, we i ntroduce Narra-
tive Topic Labels derived with Retrieval Augmented Generat ion (NTLRAG),
a scalable and extensible framework that generates semanti cally precise and
human-interpretable narrative topic labels . Ournarrative topic labels provide
a context-rich, intuitive concept to describe topic model o utput. In partic-
ular, NTLRAG uses retrieval augmented generation (RAG) tec hniques and
considers multiple retrieval strategies as well as chain-o f-thought elements to
provide high-quality output. NTLRAG can be combined with an y standard
topic model to generate, validate, and reﬁne narratives whi ch then serve as
narrative topic labels. We evaluated NTLRAG with a user stud y and three
real-world datasets consisting of more than 6.7 million soc ial media messages
that have been sent by more than 2.7 million users. The user st udy involved
16 human evaluators who found that our narrative topic label s oﬀer superior
interpretability and usability as compared to traditional keyword lists. An
∗Corresponding author
Email addresses: lisa.grobelscheg@s.wu.ac.at,
lisa.grobelscheg@campus02.at (Lisa Grobelscheg), ema.kahr@wu.ac.at (Ema Kahr),
mark.strembeck@wu.ac.at (Mark Strembeck)arXiv:2602.17216v1  [cs.SI]  19 Feb 2026

implementation of NTLRAG is publicly available for downloa d.
Keywords: , human-interpretable labels, LLM, RAG, retrieval-augmen ted
generation, topic labels, topic modeling
1. Introduction
Eﬀorts to automatically summarize large volumes of textual data and de-
rive meaningful information from them resulted in the devel opment of topic
models . While most topic modeling approaches, such as Latent Diric hlet
Allocation (LDA) (Blei et al., 2003) are designed for long te xt documents,
some approaches explicitly address the challenges of compa ratively short
texts, such as social media messages (e.g., Non-Negative Ma trix Factoriza-
tion (NMF), see Kasiviswanathan et al., 2011). Certain mode ls additionally
incorporate metadata, such as a document’s creation date or the document’s
author, into the modeling process (e.g., Structural Topic M odel (STM), see
Roberts et al., 2019).
While so-called bag-of-words approaches are unaware of the order of words
in a document, models based on word embeddings, such as BERTo pic or Con-
textual Topic Models (Grootendorst, 2022; Bianchi et al., 2 021b,a) leverage
the power of contextual word vectors. Thompson and Mimno (20 20), suggest
that embedding-based models in combination with clusterin g algorithms are
superior compared to traditional LDA models. Some recent ap proaches also
use Large Language Models (LLMs), such as OpenAI’s ChatGPT o r Google’s
Gemini, to perform topic modeling tasks (de Marcos and Domín guez-Díaz,
2025).
The diﬀerent ﬂavors of topic modeling approaches are used in a wide vari-
ety of application domains. From the impact of Open Innovati on (Lu and Chesbrough,
2022), to trends in ﬁnance research (Aziz et al., 2021) and th e analysis of
presidential elections (Chandra and Saini, 2021), researc hers have used these
methods to investigate diﬀerent phenomena. A comprehensiv e overview for
the evolution of topic modeling and its applications is prov ided in Churchill and Singh
(2022) and Laureate et al. (2023).
Thus far, the focus for improving topic models particularly was on achiev-
ing higher semantic coherence in document clusters and impr ovements in
cluster composition. However, corresponding approaches d o not adequately
consider the importance of enabling humans to comprehend th e overarching
2

messages conveyed by the documents within such clusters. In our work, we
therefore seek to address this gap.
Typically, topic model output is presented as a set of repres entative key-
words, such as words ranked by their Term-Frequency-Invers e Document
Frequency (TF-IDF) scores or raw frequency counts (Chen, 20 24). An alter-
native approach involves examining one or more representat ive document(s)
that a speciﬁc clustering algorithm identiﬁed as a centroid . Therefore, this
approach is often employed in conjunction with topic modeli ng techniques
that leverage word embeddings and clustering algorithms (S amory and Mitra,
2018). For example, the keyword list of a cluster found in a da taset on
a shooting event that we analyzed includes the following ter ms: ’gambling
gambler poker compulsive gambled transactions habits gamb le stakes welloﬀ’;
the corresponding representative document reads: ’be sure to look into gam-
bling no one makes money playing video poker.’ Yet, both the k eyword list
and the representative document fail to provide an easy-to- interpret, contex-
tualized description of the event. The documents that are in cluded in the
corresponding cluster mainly discuss the shooter’s gambli ng habit, a fact that
is neither reﬂected in the keyword list nor in the representa tive document.
Consistent with prior research (Mei et al., 2007), we argue t hat humans
require a more comprehensible description to eﬀectively gr asp the core mes-
sage of a document cluster. Such reader-friendly descripti ons are often re-
ferred to as topic labels . In this paper, we introduce Narrative Topic Labels
with Retrieval Augmented Generation (NTLRAG), an approach that derives
narrative topic labels from text corpora as a natural way for people to in-
terpret an event’s storyline (Weick, 1995) and enhance trad itional labels for
document clusters. Prior research (Grobelscheg et al., 202 2a,b) has concep-
tualized narratives as: ’ a set of topic-wise interconnected messages posted on
social media platforms ’. To enhance human interpretation of a narrative, we
will further develop this deﬁnition in Section 3.1.
Building on the discussion above, we identify the following areas for im-
provement:
1. Enhanced interpretability and contextual richness of to pic labels: La-
bels need to provide a comprehensible, accurate descriptio n of a topic.
2. Automation and independence of speciﬁc implementation a pproaches
or (proprietary) tools: the label generation framework sho uld mini-
mize dependence on time-consuming and subjective human int erven-
tion while maintaining a high adaptability to advances in th e underly-
3

ing technologies.
3. Automated validation of topic labels: Ensuring accurate and context-
based labels is vital for a correct human understanding of do cument
clusters.
We address the above areas by 1) conceptualizing a narrative schema
that reﬂects the content in each set of documents and is easy t o interpret
for humans. The respective narratives will then serve as top ic labels. 2) We
introduce NTLRAG, a retrieval-augmented generation (RAG) pipeline that
operates on topic model outputs and produces reliable, huma n-interpretable
narratives based on corpora consisting of short text docume nts (such as social
media messages).
With our approach, we explicitly focus on short texts, as the y are pro-
duced in vast amounts on diﬀerent social media platforms and also present
a major source for gathering individual opinions on real-wo rld events. As
user-generated short texts often only provide sparse infor mation and a high
degree of opinionated text, we employ validated news from re putable news
providers to corroborate the derived narratives in a retrie val augmented gen-
eration step. Thus, we make the following contributions:
(i) we establish a well-deﬁned conceptualization of a narrative schema for
describing topic model outputs,
(ii) we introduce a modular RAG framework tailored for struc tured nar-
rative extraction and validation across multi-topic short text corpora,
and
(iii) we incorporate a dual-retriever strategy in a narrati ve analysis pipeline,
optimized for diﬀerent input source types.
To the best of our knowledge, no existing approach for topic l abel gener-
ation employs a retrieval-augmented generation pipeline f or the creation of
narrative topic labels. The remainder of this paper is struc tured as follows.
Section 2 discusses related work. Section 3 describes the NT LRAG frame-
work and Section 4 presents our NTLRAG implementation. Subs equently,
Section 5 discusses the evaluation of NTLRAG on three real-w orld datasets
and a qualitative user study. In Section 6, we discuss limita tions and possible
extensions, before Section 7 concludes the paper.
4

2. Related Work
This section provides an overview of current practices in au tomatic topic
labeling and evaluation (Section 2.1) as well as narrative c onceptualizations
in computational linguistics and information retrieval (S ection 2.2).
2.1. Automatic Topic Labeling and Evaluation
In a recent review, Mekaoui et al. (2025) classify topic labe ling approaches
into six main categories: (i) traditional methods (e.g. inf ormation retrieval),
(ii) ontology-based methods, (iii) graph-based methods, ( iv) human annota-
tion, (v) hybrid approaches and (vi) approaches based on neu ral networks
(e.g. transformers such as large language models). They als o highlight the
prevalence of domain-speciﬁc datasets for an evaluation of diﬀerent methods
(e.g., social media messages, reviews, etc.). Most of the ap proaches included
in this review relied on human judgment to evaluate their res ults.
Williams et al. (2024) present an approach to improve interp retability of
topic labels without external sources by mapping relevant k eywords to the
documents in the initial corpus. Subsequently, the keyword s are ranked by
the number of intersecting tokens using LDA, and the top-sco ring tokens are
identiﬁed as topic labels. The results are evaluated throug h human annota-
tion, applying three quality criteria (quality, usefulnes s, and eﬃciency) on a
5-point Likert scale.
Another paper employs a combined approach that blends the Co nceptNet
(Speer et al., 2017) knowledge graph and LLM tasks to introdu ce a zero-
shot topic-labeling framework (Top2Label) (Chaudhary et a l., 2024). The
approach generates three types of topic labels (word, sente nce, and summary)
to provide easy-to-understand topic descriptions.
Piper and Wu (2025) use LLMs to generate topics and human-int erpretable
labels (keywords) from narrative texts (e.g., news article s or novels) . The
evaluation compares LLM-based and human-generated topic l abels, indicat-
ing that the former substantially outperform the latter wit h respect to speciﬁc
semantic characteristics. In a diﬀerent approach, Lam et al . (2024) leveraged
LLMs for topic modeling and creating generalized concepts a s topic labels.
To capture temporal topic evolution, a dynamic topic labeli ng approach
is presented and automatically evaluated in Guillén-Pacho et al. (2024). In
Wanna et al. (2024), Non-Negative Matrix Factorization (NM F) has been
combined with a topic model and prompt-tuning LLMs. The appr oach com-
bines automatic labeling and human evaluation.
5

Instead of suggesting a separate topic labeling model, some studies inte-
grate the aspect of increased topic interpretability direc tly into the topic mod-
eling process. For example, the Spherical Correlated Topic Model (SCTM)
(Ennajari et al., 2025) addresses the challenges of context -poor short text
documents by combining word embeddings and knowledge graph embed-
dings. Moreover, SCTM aims to enhance interpretability of t opics by ac-
counting for the correlation between topics. However, the r esult is still limited
to a list of representative keywords without further contex tual information.
Except for Top2Label (Chaudhary et al., 2024), all approach es discussed
above provide topic labels as keyword lists, whereas NTLRAG generates
context-rich narratives instead. Furthermore, most topic labeling models rely
on a single information source, which is also used to create t he topics them-
selves (e.g., Ennajari et al., 2025; Guillén-Pacho et al., 2 024; Wanna et al.,
2024; Williams et al., 2024). In contrast, NTLRAG incorpora tes additional
context by integrating validated news sources in a RAG step.
In evaluating topic model outputs, several studies have pro posed metrics
for human interpretability and human-identiﬁable semanti c coherence. For
example, Chang et al. (2009) conceptualize human-interpre tability by devel-
oping tasks to measure human-identiﬁable semantic coheren ce. They apply
the concept of word intrusion, where raters have to identify an ’intruder
word’ out of six terms which are presumably the most probable words for
each topic. The assumption is that if the intruder word can be identiﬁed, the
coherence of the topic for human readers is good. The second t ask is called
topic intrusion. It serves as an assessment of the topic mode l’s document
composition in terms of human interpretability. Users have to identify an
’intruder topic’ (represented by the eight highest probabi lity words of this
topic) among four candidate topics.
Another conceptualization of topic interpretability is pr esented in Newman et al.
(2010). For evaluation, a 3-point ordinal scale is used to ra te the observed
coherence of a topic (3=useful (coherent), 2=neutral (part ially coherent or
somewhat unclear), 1=not useful (less coherent)). Here, us efulness is deﬁned
as the ability of topic-related keywords to be used in a searc h interface to
retrieve documents about a speciﬁc topic, or the ease of ﬁndi ng a concise
label which describes the topic.
At the model level, word intrusion and observed topic cohere nce pro-
vide almost identical results (Lau et al., 2014). Churchill and Singh (2022)
present the most prevalent metrics for evaluating topic mod els. Diﬀerent
forms of human or qualitative evaluation are discussed, suc h as presenting
6

evaluators with a set of topics from multiple models and aski ng them to rank
which one best describes each topic.
2.2. Narratives in computational linguistics/informatio n retrieval
In Piper et al. (2021), the authors discuss the gap between th e ﬁeld of
natural language processing (NLP) and the vast amount of the oretical work
on narratives within other disciplines (e.g., humanities, social sciences). They
link theoretical narrative concepts to NLP applications an d provide a concise
deﬁnition of a narrative for practical use. Furthermore, a t en-component
structure to determine the presence of narrativity is intro duced: 1) teller, 2)
mode of telling, 3) recipient, 4) situation, 5) agent, 6) one or more sequential
actions, 7) potential object, 8) spatial location, 9) tempo ral speciﬁcation, 10)
rationale.
As user-generated short text documents in general exhibit a n irregular
structure, limited context, and substantial variability c ompared to longer,
curated texts (Churchill and Singh, 2022), we argue for an ev en more re-
duced schema for determining a narrative. Santana et al. (20 23) present a
review on narrative extraction from textual data. They desc ribe it as a sub-
domain of artiﬁcial intelligence (AI) that spans from retri eving information
to summarizing it, extracting narrative elements from it, a nd producing text
from this data. They suggest a narrative extraction pipelin e with NLP tasks,
such as Parsing, Part-of-Speech-Tagging, etc., to identif y narrative compo-
nents (events, participants, time, and space).
Domain-speciﬁc narrative deﬁnitions have been developed f or a number
of diﬀerent applications. One example is the collective eco nomic narrative
(Roos and Reccius, 2024). This concept refers to a story abou t a topic with
an economic context that is used for sense-making.
To extract narratives from user-generated short text, agen t-action-target
(Subject-Verb-Object) triplets have been introduced in Sa mory and Mitra
(2018). A survey of how diﬀerent domains have conceptualize d narratives
for their ﬁeld of research can be found in Merrill (2007). The survey also
emphasizes the lack of a unanimously accepted deﬁnition of t he term "narra-
tive", suggesting that the knowledge produced by extractin g narratives has
a value for research that goes way beyond a mere deﬁnition of t he term.
7

3. Conceptual Design
In this section, we introduce the NTLRAG framework and discu ss our
conceptualization of narratives. NTLRAG and this conceptu alization which
will then be applied for deriving narrative topic labels.
3.1. Narrative Schema Conceptualization
In order to conceptualize narratives for creating human-in terpretable topic
labels from short text clusters, we build on prior deﬁnition s for narratives as
’description of a set of topic-wise interconnected documen ts.’ We further in-
corporate elements from Piper et al. (2021) and Herman (2009 ) which frame
narratives as experiences of human-like agents interactin g with their environ-
ment, as well as from Chambers and Jurafsky (2008), Chambers and Jurafsky
(2009) and Chambers (2013) which introduce narrative chain s (ordered events
around a protagonist) and narrative schemas (semantic role -based struc-
tures).
Based on the related work (see also Section 2), we thus concep tualize a
narrative as a data structure consisting of:
(i) one or more actor(s); Actor ∈{individual, group, institution, public
entity, country, generic person/agent }
(ii) an action associated with the actor(s); Action ∈{verb predicate ex-
tracted from text}, and
(iii) an event linking the actor(s) and the action; Event ∈{incident, context
cluster}.
For improving interpretability, we further include a conci se descriptive
text that creates a sentence from the three narrative elemen ts mentioned
above. Subsequently, we use this four-element structure as a narrative schema
for topic labels extracted and generated by NTLRAG.
3.2. NTLRAG
NTLRAG is a retrieval-augmented generation (RAG)-based fr amework
that is designed for working on the output produced by a topic model. The
framework is inherently ﬂexible, allowing for the integrat ion of diﬀerent RAG
implementations and components. Additionally, through it s modular struc-
ture, extensions or adaptations can easily be implemented. Figure 1 presents
8

RETRIEVER EXTRACTOR VALIDATOR REFINER short
textsnarrative 
schema
short
textsnews
texts
Figure 1: NTLRAG pipeline: Retriever, Extractor, Validato r, and the conditional Reﬁner.
a high-level view of the NTLRAG pipeline, a detailed pseudo- code speciﬁca-
tion is found in Algorithm 1.
The input to the RAG pipeline is the query which is provided by the key-
word list that has been created by the underlying topic model . Short text
documents are represented as DS, whileDNdenotes curated and trustwor-
thy news content. Extracted narratives ( NarrativeSchema ) are processed
through the validation function and re-extracted in the reﬁ ne function if they
are assigned to the ’reﬁne’ category (for details see below) .
Algorithm 1: NTLRAG - Pipeline
Input: Keyword-based query q
Output: Final Output NarrativeSchema
DS←RETRIEVERS (q);
DN←RETRIEVERN (q);
D←Concatenate (DS,DN);
M←EXTRACTOR (prompt=EXTRACT ,input=DS);
repeat
c←VALIDATOR (prompt=VALIDATE ,inputs=
[NarrativeSchema,D ]);
ifc="reﬁne" then
NarrativeSchema ←REFINER (prompt=REFINE ,inputs=
[NarrativeSchema,D S]);
untilc="approve" ;
returnNarrativeSchema ;
RETRIEVER. The RAG pipeline includes four main steps implemented
via four software components (see also Algorithm 1). The ﬁrs t component
in this pipeline is the Retriever which focuses on document r etrieval. In this
step, relevant documents are collected from two sources: th e topic model
9

output and a corpus of related news documents. The topic mode l output
must include the following information:
(i) the full text of each document,
(ii) its assigned topic number, and
(iii) a set of keywords per topic that are essential for build ing the query for
retrieval.
The news document corpus comprises texts that closely align with the
content of the short-text documents. For example, when anal yzing social
media messages related to a speciﬁc incident, additional do cuments include
trustworthy news reports on the same event (e.g., a shooting , a ﬂood, or a
wildﬁre).
The framework remains source-independent and can be used wi th short
text corpora from any social media platform as well as news ar ticles from any
reputable news provider. Regarding news articles, NTLRAG r equires only
the article text, title, or a concise summary of its content. In general, full
articles are preferred for news context documents. Dependi ng on the speciﬁc
implementation of the Extractor component, complete news a rticles could
be too large, though. For example, diﬀerent large language m odels (LLMs)
may impose diﬀerent limitations on the length of accepted in put sequences.
Therefore, using summaries or even just titles can serve as a lternatives. In
any case, for both corpora (news and short text), optional me tadata (e.g.,
publication date, news outlet) can be incorporated to facil itate downstream
analysis and interpretation of the results.
Distinct retrieval functions are used for each corpus; howe ver, the query
remains identical for both contexts and is derived from the k eywords gener-
ated by the topic model.
EXTRACTOR. The second step involves the extraction of a narrative
description from the retrieved short text documents. Becau se NTLRAG
relies on a RAG architecture, one popular choice for the Extr actor component
is using an LLM. However, in general the Extractor can be impl emented using
any suitable component or technology.
The output of the Extractor is based on a structured data mode l with
ﬁve elements:
(i) a topic identiﬁer,
10

Short text
documentsNews
documentsKeywords from short text documents
men are 
responsible for 
mass 
shootings how 
toxic 
masculinity is
killing usSanta Fe High School 
community mourns the 
10 victims killed as 
authorities probe motive
behind massacre.
Figure 2: RETRIEVE step including an example of retrieved co ntext documents.
(ii) the actor(s) involved,
(iii) an action,
(iv) an event linking the actor(s) and the action, as well as
(v) a concise one-sentence narrative summary referred to as narrative topic
description .
This structured representation ensures consistency, comp arability, and
facilitates subsequent validation steps. When the Extract or is implemented
based on an LLM, prompt engineering is crucial to prevent hal lucinations or
irrelevant responses. The proposed prompt instructions th at we use in our
implementation are detailed in Section 4.
VALIDATOR. Once a narrative topic description has been derived from
the corresponding short text documents, it subsequently un dergoes a valida-
tion process.
For this step, the short text and news documents serve as cont extual
references. As shown in Figure 4, two main criteria determin e the valida-
11

men are re sponsible for mass shootings how
toxic masculinity is killing us
Actor: user
Action: is concerned about toxic
masculinity
Event: mass shootings
Description: A user expresses concern about 
toxic masculinity and the high
rate of male perpetrators in
mass shootings
Figure 3: EXTRACT step with an example short text document an d the derived narrative
data structure.
tion outcome. First, a validity check veriﬁes whether each a ttribute of the
narrative schema (actor, action, event, and description) exists. Second, th e
Validator assesses the narrative’s quality. It is straight forward, to imple-
ment this component based on an LLM, however any suitable tec hnology
can be used for the Validator. Subsequently, assessment wil l be performed
through retrieval augmented generation, using a tailored p rompt and relevant
documents. The primary quality criterion is consistency wi th the context
documents, enforced through a “non-contradiction” clause . For our imple-
mentation, iterative prompt engineering indicated that a n egative detection
approach is most eﬀective (for details see Section 4). There fore, all narratives
are initially assigned an “approved” category which is revi sed only if
(i) one or more elements (actor, action, event, description ) are missing, or
(ii) the narrative contradicts the context documents.
In this paper, a contradiction refers to an event or state that occurs in
both the narrative and the context documents, but is describ ed in conﬂicting
ways. For example, the narrative “actor: farmers; action: p rotest; event:
new regulations on cattle care; description: farmers prote st new regulations
on cattle care in Wyoming” contradicts the context “Farmers supported the
adoption of new regulations specifying cattle care require ments in the U.S.”.
In contrast, diﬀerences in granularity, such as specifying a more precise loca-
12

Actor: user
Action: is concerned about toxic
masculinity
Event: mass shootings
Description: A user expresses concern
about toxic masculinity and 
the high rate of male 
perpetrators in mass
shootings .relevant short
text
documentsrelevant news
documents
1. Are there any missing elds ?
2. Does the narrative contradict the
relevant context documents ?
If any yes, REFINE If all no, APPROVEcontext narrative VALIDATE
Figure 4: VALIDATE step with an example narrative and main ca tegorization criteria.
tion (Wyoming instead of the United States), are not conside red contradic-
tions.
In addition, the prompt includes further guidance (e.g., ’D o not consider
tone or language when grading the narrative’) and explicit c onstraints to
prevent hallucinations. If a narrative is assigned the ’reﬁ ne’ category, it is
moved to the Reﬁner component, where a new narrative is extra cted. Each
narrative is re-assessed until it is assigned to the ’approv ed’ category. At this
point, a chain-of-thought element is part of the pipeline, f orcing the Validator
to explain the assigned category (see examples in Section 4) .
REFINER. In this step, a new narrative is extracted from the short text
documents. The extraction procedure is identical to that us ed by the Extrac-
tor, but implemented as a separate function to allow ﬂexibil ity in sourcing
narratives from both short texts and news documents.
13

4. NTLRAG Implementation
We provide a full implementation of the conceptual framewor k presented
in Section 3. While we evaluated multiple diﬀerent options a nd conﬁgu-
rations, this section describes one particular implementa tion of NTLRAG.
Figure 5 presents the implementation choice for each compon ent. Our im-
plementation is publicly available for download.1R  TR   V   R
  E T R A C T O R
 A    A  	
R 

  R short
textsnarrative 
schema
short
texts
n   s
texts
BM25
ChromaDBLlama 3.2
via OllamaLlama 3.2
via OllamaLlama 3.2 
via OllamaLANGCHAIN LANGGRAPH
OLLAMA OLLAMAOLLAMAPYDANTIC PYTHON
Figure 5: NTLRAG components with implementation choices. L angChain, LangGraph
and Pydantic for general RAG orchestration and Llama LLM via Ollama for Extractor,
Validator and Reﬁner.
For our implementation, we used LangChain and the LangGraph frame-
work (Chase, 2022) implemented in Python. NTLRAG can be desc ribed
as an orchestrated RAG pipeline with a loop that is dependent on the out-
come of the Validator (see also Algorithm 1). In the developm ent process,
we ensured that the model performs eﬃciently without requir ing specialized
computational infrastructure.2
RETRIEVER. As outlined in Section 3, the ﬁrst step of NTLRAG con-
sists of retrieving relevant documents based on topic model output (short
text documents) as well as a news corpus. The keywords produc ed by the
1Seehttps://github.com/lisagrobels/NTLRAG .
2Runtime environments for testing our implementation were c hosen based on each task’s
requirements: for data preprocessing, we used the standard CPU runtime (RAM: 51 GB),
while topic modeling and NTLRAG execution employed T4 GPUs ( GPU RAM: 15 GB,
CPU RAM: 51 GB) and A100 GPUs (GPU RAM: 40 GB, CPU RAM: 83.48 GB) inter-
changeably, depending on availability. The LLM implementa tion was set up using Ollama
and the Llama 3.2 model (3B parameters) https://ollama.com/library/llama3.2 .
14

upstream topic model are used as the query to retrieve inform ation from both
sources. The framework supports a wide range of retriever an d data stor-
age conﬁgurations. Lexical approaches (e.g., term frequen cy-based such as
BM25) and vector-based approaches can be used independentl y or integrated
within an ensemble retriever (Afzal et al., 2024).
In our implementation, we applied BM25 (Best Matching 25) fo r the short
text corpus. BM25 is a bag-of-words model that ranks documen ts based on
TF-IDF scores and document length (Robertson et al., 1995). This choice
is motivated by the fact that the query strings are generated from the short
text corpus itself which reduces the likelihood of mismatch es arising from
lexical variation. Our BM25 retriever for the short text cor pus is organized
as a dictionary. It maps each topic to a dedicated BM25 instan ce, enabling
NTLRAG to iterate over topics eﬃciently.
For news documents, we used a vector-based retrieval method , whereby
the documents are ﬁrst transformed into word embeddings bef ore the retrieval
function is applied. The matching and ranking process is per formed based on
cosine similarity scores. We store and index embeddings usi ng ChromaDB
(Chroma Team, 2025). Retrieval queries are static and compr ise relevant
keywords for each topic as determined through topic modelin g (see Tables 2
and 3 for example queries).
EXTRACTOR. In our implementation, the extractor component is based
on an LLM. To align with the narrative schema deﬁned in Section 3.1, em-
ploying an Extractor that supports structured output is rec ommended (such
as a suitable LLM). We use Llama 3.2 via Ollama and Pydantic (C olvin et al.,
2025) to validate and guide the LLM’s answers. The correspon ding data
structure is shown in Table 1. Furthermore, we leverage the s tructured out-
put method provided by LangChain and Ollama.3
The Pydantic JSON schema description extends the prompt use d for
narrative extraction and enables eﬃcient validation of out put and easy use
in downstream tasks.
The composition of the prompt strongly inﬂuences the qualit y of the
LLM output. The ﬁnal prompt was developed through an iterati ve process
involving the addition, removal, and reﬁnement of prompt co mponents. It
comprises three main elements: (i) the system message, e.g. , ‘You are an
3https://python.langchain.com/docs/how_to/structured _output/#typeddict-or-json-schema ,
https://ollama.com/blog/structured-outputs .
15

Table 1: NTLRAG Pydantic model schema for narratives.
Field Type Description
topic_id String The topic ID of the narrative
actor String The actor(s) of the narrative
action String Action that is carried out by actor(s) or other en-
tities or individuals
event String The event linking the actor(s) and their action
description String A one sentence long description of the na rrative
Table 2: Example in- and output of the EXTRACT and VALIDATION step in the ap-
proved category.
Output
Query largest norm shootings modern biggest deadliest
lifetime frequency worst proposals
Retrieved documents we’ve seen the biggest mass shooting in history
headline too many times in my lifetime. 710 dead-
liest mass shootings happened in my lifetime; my
home state holds the most. This is the deadliest
mass shooting in US history — where’s the out-
rage, where’s the policy proposals? Mass shoot-
ings are an American norm, sad but true.
Narrative actor: user,action: expresses frustration with
gun violence, event: mass shootings. descrip-
tion: The user expresses frustration with mass
shootings in the US, highlighting their increasing
frequency and casualty count.
Validation The narrative is consistent with the context and
does not contradict any information provided.
information extraction system...’; (ii) the extraction ru les, e.g., ‘STRICTLY
use only information found in the provided documents...’; a nd (iii) the re-
trieved short text documents. Within the extraction rules, the Pydantic data
model was further speciﬁed. For instance, the ‘user’ value w as speciﬁed as
valid for the ‘actor’ ﬁeld when the LLM could not identify any other actor(s).
During validation, this value is always accepted provided t hat all other ﬁelds
are valid.
16

Also, the LLM was prompted to disregard any diﬀerences in ton e and
language. The complete prompt is provided in Appendix A, and an example
of retrieved documents, including the corresponding query , narrative output,
and validation explanation, is shown in Tables 2 and 3. After successful
extraction the narrative is forwarded to the Validator.
VALIDATOR. For narrative validation, the LLM is provided with the
output from the extraction step and instructed to categoriz e each narrative
as either ‘approve’ or ‘reﬁne.’ Narratives categorized as ‘ approve’ are consid-
ered ﬁnal and do not undergo further processing, whereas tho se categorized
as ‘reﬁne’ are forwarded to the reﬁnement step. In particula r, the validation
step draws on two context corpora (short text documents and n ews articles)
along with a prompt template. A detailed description of the p rompt logic is
provided in Section 3, and the full prompt is included in Appe ndix A. An inte-
grated chain-of-thought component generates an explanati on for each LLM
validation decision. Examples of queries, retrieved docum ents (retrieval),
extracted narratives (extract), validation outcomes (val idate), and reﬁned
narratives (reﬁne) are presented in Tables 2 and 3.
Each narrative is allowed a maximum of 100 ‘reﬁne’ iteration s, as iterative
testing indicated no further improvements beyond this poin t. This limit can
be adjusted depending on the speciﬁc use case, of course. How ever, increasing
the retry limit requires a corresponding increase in the rec ursion limit for
LLM calls, which in turn results in a greater number of LLM req uests and
potentially higher cost for online models with request-bas ed pricing.
REFINER. This step mirrors the functionality of the Extractor, provi ding
short text context documents to the LLM to populate the narrative schema .
The prompt from the Extractor remains unchanged. Since the s ame context
documents are used, the resulting narrative content is not e xpected to deviate
substantially from the extraction.
5. Evaluation
In this section, we report on the results obtained from apply ing our ap-
proach to three real-world datasets consisting of more than 6.7 million social
media messages that have been sent by more than 2.7 million us ers. Further-
more, the resulting narratives are evaluated through a ques tionnaire-based
human assessment.
17

Table 3: Example in- and output of the EXTRACT and VALIDATION step in the reﬁne
category.
Output
Query ar15 ar15s foundno shotgun ar riﬂe pistol used
ar15style revolver
Retrieved documents no ar15 foundno ar15 foundno ar15 foundno ar15
foundno ar15 foundno ar15 foundno ar15 foundno
ar15 foundno he was armed with an ar15style riﬂe
a pistol a shotgun standard bill of fare these days.
hey the attacker was armed with an ar15style ri-
ﬂe a pistol a shotgun and pipe bombs maybe you
should... no ar15 at santa fe shooter used a pistol
and shotgun. he used a shotgun and pistol not a
ar15 still tragic.
Narrative I actor: user,action: attacked, event: oﬃce
building. description: Attacker armed with an
AR-15 style riﬂe and other weapons carried out
the attack.
Validation (reﬁne) The narrative includes hallucinations (i.e. facts
not present in the context). The context does not
mention ’oﬃce building’ or the action ’attacked’.
Also, no ar15 found.
Narrative II actor: user,action: attacked, event: Santa Fe
High School description: The attacker used a
shotgun and pistol to attack Santa Fe High School.
Validation (approve) The narrative is consistent with the context, as it
reports that the attacker used a shotgun and pistol
to attack Santa Fe High School. The actor ’user’
is also valid in this scenario.
5.1. Datasets and Preprocessing
In line with numerous studies in this ﬁeld (Mekaoui et al., 20 25), we em-
ployed domain-speciﬁc datasets to evaluate our framework. The datasets are
summarized in Table 4 and have partially been used in other to pic mod-
eling approaches in Grobelscheg et al. (2022b), Grobelsche g et al. (2022a),
Kušen and Strembeck (2021b) and Kušen and Strembeck (2021a) .
18

Table 4: Three datasets used in the case study.
Location Observation period Users Tweets Unique Tweets
Las Vegas (NV) 2-14 October 2017 1,394,070 3,436,187 505,85 0
Santa Fe (TX) 18-25 May 2018 458,644 967,674 113,146
El Paso (TX) 3-18 August 2019 939,940 2,307,577 318,368
• The Las Vegas dataset refers to a mass shooting at the Route 9 1 Harvest
music festival on October 1, 2017. A shooter killed 60 people and
wounded at least 413 others. The subsequent panic increased the total
number to 867 injuries (NBC News, 2019).
• The Santa Fe dataset refers to a school shooting event that h appened
in May 2018 in Santa Fe, Texas. Eight students and two teacher s were
shot dead and ten others wounded (CNN, 2018).
• The El Paso dataset refers a shooting in a Walmart in El Paso, Texas
on August 3, 2019. In this incident, 23 people were killed, an other 22
have been wounded (CNN, 2019).
The datasets were collected via Twitter’s API with academic access during
the observation periods referred to in Table 4. Only unique t weets were
retained for topic modeling. Pre-processing steps include d the removal of
retweet identiﬁers, user mentions, special characters, UR Ls, and hashtags.
Following the procedure outlined in Grootendorst (2022), a ll text messages
have been converted to lowercase.
To obtain document clusters, we applied BERTopic (Grootend orst, 2022),
a modular framework based on document word embeddings. The d imension-
ality of the word vectors is reduced using Uniform Manifold A pproximation
and Projection (UMAP). Embeddings are then grouped using a d ensity-based
clustering algorithm (HDBSCAN, Campello et al., 2013). For our datasets,
the model was conﬁgured through hyperparameter tuning via t he Optuna
framework (Akiba et al., 2019). We computed a weighted score based on the
Silhouette Score, Diversity, as well as the Outlier Ratio to determine the op-
timal number of minimum clusters for HDBSCAN and the minimum number
of documents per topic for BERTopic itself. This procedure r esulted in 2,335
topics for the Las Vegas dataset, 948 topics for the El Paso da taset, and 294
for the Santa Fe dataset.
19

5.2. NTLRAG Evaluation and User Study
A manual inspection of the model’s results has already revea led insights
into NTLRAG’s superior ability to handle negated statement s in comparison
to keywords. Table 3 shows a corrected narrative that initia lly included the
information that a perpetrator was using an AR15 riﬂe. Howev er, the related
context documents did not support this information; theref ore, NTLRAG
reﬁned the narrative and added facts. In contrast, the keywo rds alone (’ar15
ar15s foundno shotgun ar riﬂe pistol used ar15style revolve r’) do not give any
meaningful information about this incident.
Another example demonstrating NTLRAG’s superiority to key words is
the added context it provides. In Table 5, we present retriev ed short text doc-
uments, their corresponding narratives, and associated ke ywords. Whereas
the keywords do not provide any connection to the shooter, th e correspond-
ing narrative adds him to the context, making the informatio n considerably
easier to interpret.
Table 5: Example in- and output of the EXTRACT step
Output
Retrieved documents las vegas gunman described as welloﬀ gambler
peaceful. las vegas gunman described as welloﬀ
gambler and a loner. killers gambling habits re-
vealed. mysteriously calculated gambling habits.
las vegas shooter gambled 100000 an hour at video
poker.
Narrative (NTLRAG) actor: user,action: described, event: gam-
bler.description: A well-oﬀ gambler, described
as peaceful and a loner, carried out the attack.
Keywords (BERTopic) gambling gambler poker compulsive gambled
transactions habits gamble stakes welloﬀ
For a human evaluation of NTLRAG-generated narratives in co mpari-
son to traditional lists of keywords, we implemented an appl ication via Shiny
(Chang et al., 2025) that was hosted on shinyapps.io.4We designed an evalu-
ation form comprising an introduction, an example page with three examples
4https://www.shinyapps.io/ .
20

how to perform the rating procedure, and an evaluation page. Screenshots
of the evaluation application are provided in Appendix B.
Because NTLRAG uses standard topic model output to generate narra-
tive topic descriptions, our goal is to evaluate the human in terpretability of
narrative topic descriptions, rather than the quality of th e underlying topic
model output. To this end, we developed a procedure to determ ine how well
the narrative topic description can summarize documents in each cluster.
The evaluation included 50 topics, where each topic is descr ibed by (i) the
narrative, consisting of the actor, action, event triplet, and a description, as
well as (ii) the keywords as produced by BERTopic via class-b ased-TF-IDF
weighting.
NTLRAG was evaluated for its usefulness (see, e.g., Lau et al ., 2014;
Williams et al., 2024; Lau and Baldwin, 2016) on a 3-point-or dinal scale to
rate topic labels as suggested by Lau et al. (2014), Lau and Ba ldwin (2016)
and Mimno et al. (2011). Raters were provided with a set of doc uments be-
longing to the same cluster, a narrative label describing th e cluster (including
actor, action, event, and description), and for comparison a list of keywords
automatically generated by the BERTopic model (see also App endix B).
In our evaluation, usefulness consisted of two main aspects :
•Accuracy – Does the label reﬂect the content of the documents?
•Clarity – Can you easily understand the cluster’s overall message fr om
the label?
Raters had to evaluate 50 narratives and keyword lists which were ran-
domly selected from all three datasets. Overall, 16 human ra ters participated
in the survey with an average time eﬀort of 50 minutes each. Pa rticipants
were all familiar with social media texts, two were consider ed experts as they
work with user-generated short texts on a regular basis.
To assess inter-rater reliability, we ﬁrst calculated Krip pendorﬀ’s Alpha
(see, e.g., Williams et al., 2024; Khaliq et al., 2024; Santa na et al., 2023).
The alpha coeﬃcient (Ford, 2004) ranges from -1 (systematic disagreement)
to 1 (perfect agreement), with 0 indicating no agreement.
For the narrative ratings in our evaluation, the alpha value s ranged from
0.376 (Santa Fe dataset) to 0.397 (Las Vegas dataset). Parti cipants did
not require prior experience in topic modeling or short text data analysis.
However, we found that expert participants reached an inter -rater agreement
21

Table 6: NTLRAG narratives and BERTopic keywords examples. Three NTLRAG nar-
ratives with the highest human rating and the corresponding BERTopic keyword lists.
Narratives Keywords
ACTOR ACTION EVENT DESCRIPTION
Jason
Aldeanresume
touringLas Vegas
shooting,Country star Jason
Aldean resumed
his tour in Tulsa
after the Las Vegas
shooting marred
his performance.resumes
tour resume
resuming
marred re-
sumed tulsa
touring tuls
star
Natca
Air
Traﬃc
Con-
trollerwarned Airport
During
Las Vegas
ShootingAn air traﬃc con-
troller at a con-
cert in Las Vegas
warned airport au-
thorities about the
shooting.maher con-
troller rips
traﬃc snark-
ily natca ac-
curatebill air-
port shootg
warned
NRA opposes US ban
on gun
devicesThe NRA opposes
an outright US ban
on gun devices used
by Las Vegas killer.outright op-
poses devices
killerend
ampvisitors
zou sci ac-
complished
newtop killer
of 0.523 across all datasets, while non-experts reached an a greement of 0.393.
While alpha values for narratives can be interpreted as ’Fai r Agreement’
(0.2<=α <= 0.4) (Hughes, 2021), the main focus of our interpretation is
the comparison between narratives and keywords for topic la beling.
Figure 6 shows Krippendorﬀ’s alpha values for narratives an d keywords
separately for each dataset as well as the combined dataset. The notable
diﬀerence between narrative and keyword scores suggests a s igniﬁcant gap
in human ability to understand the central message of a docum ent cluster.
We follow Williams et al. (2024) in their interpretation of a large gap be-
tween alpha values of two categories and argue that, based on these results,
evaluators found interpreting keywords more diﬃcult than t he corresponding
22

Table 7: NTLRAG Narratives and BERTopic keywords examples. Three BERTopic key-
word lists with the highest human ranking and the correspond ing NTLRAG narratives.
Narratives Keywords
ACTOR ACTION EVENT DESCRIPTION
user criticizing the user The user is criticiz-
ing a person for ly-
ing.idiot stupid
liar lying mo-
ron ignorant
dumb lies
stupidity fool
Dallas
Cow-
boysdonating El Paso
Victims
Relief
FundThe Dallas Cow-
boys NFL Foun-
dation donated
$50,000 to support
those aﬀected by
the El Paso shoot-
ing.foundation
charity fund
norte legends
relief donated
proceeds
50000 donat-
ing
[name of
perpe-
trator]posted and
sharedNeonazi
imagery[name of perpetra-
tor] posted neonazi
imagery online be-
fore killing at least
eight people.imagery
neonazi on-
line posted
onlinebefore
onlin himgt-
dimitrios
suspct imgry
onlineshared
narratives. Table 6 showcases the top three narratives with the highest aver-
age ratings, and Table 7 presents the keyword lists with the h ighest ratings
and their corresponding narratives.
Overall, NTLRAG-generated narratives received an average rating of
2.467, whereas ordinary BERTopic keyword lists received an average rat-
ing of 1.61 (both on the 1-3 scale). Out of the evaluated 50 nar ratives, 49
received at least once the highest (’useful’) rating. For th e keyword lists, the
corresponding value was 33. In 94.73% of all cases, the narra tive received
the same or a higher rating than the keyword list. For 63.25% o f all cases,
evaluators strictly preferred the narrative description.
23

Figure 6: Krippendorﬀ’s alpha for narratives and keywords o f each dataset and all three
datasets combined.
6. Discussion
6.1. Conceptual Design
NTLRAG produces context-rich topic labels (narratives) th at outperform
traditional keyword lists in terms of human interpretabili ty (see Section 5).
Compared to existing approaches, it oﬀers a more comprehens ive description
of topics without requiring user intervention (see also Sec tion 4). NTLRAG
reﬁnes a narrative until it is accepted or a reﬁnement limit i s reached. De-
pending on the speciﬁc document cluster or on how the reﬁneme nt limit is
chosen, this might result in low-quality narratives. Furth er improvements
could be achieved through a human-in-the-loop approach, su ch as integrat-
ing an evaluation tool into the pipeline and converting vali dation from fully-
automatic to semi-automatic. Additionally, if NTLRAG is co mbined with
a multinomial topic model, the retrieval step could incorpo rate document
weighting to reduce the likelihood of retrieving low-proba bility context doc-
uments.
6.2. Implementation
Our NTLRAG example implementation also comes with certain l imi-
tations. All general drawbacks associated with using LLMs f or generative
tasks also apply to our NTLRAG implementation, particularl y the high de-
pendence on an LLM’s training data, vulnerability to halluc inations, and
limited reproducibility. This may aﬀect the validity and in terpretability of
results. However, the NTLRAG pipeline and implementation ( see Sections
24

3 and 4) are designed to address many of these challenges. In p articular, we
use a RAG-based design, chain-of-thought components, and v alidation steps.
Nevertheless, the incorporated LLMs may still produce part ially incorrect
or irrelevant outputs. Since the choice of a particular LLM s igniﬁcantly
inﬂuences both the results and the required prompt structur e, we recom-
mend deﬁning selection criteria (e.g., latency, cost) befo re formulating the
full prompt. Remarks from our human evaluators suggested th at particular
document clusters had too speciﬁc descriptions, which, aga in, might result
in underrepresented or missing topical aspects of the clust er.
7. Conclusion
In this paper, we introduced a novel topic labeling framewor k that can
operate on the output of any classical topic model. NTLRAG is modular
and can be implemented using a wide range of methods for text e xtraction
(e.g., LLMs) and orchestration frameworks, requiring mini mal human eﬀort.
Moreover, we introduce a ’narrative schema’ that serves as a context-rich
option for labeling and describing topics. In our evaluatio n with human users,
we found that the narrative schema shows superior usability for interpreting
document clusters (i.e., topics) compared to traditional k eyword lists.
In particular, we conducted a user study where 16 human evalu ators
found that NTLRAG narratives are straightforward to compre hend and that
narrative topic labels more eﬀectively represented the und erlying document
clusters as compared to traditional keyword lists. An imple mentation of
NTLRAG is publicly available for download.5Among other things, we tested
our example implementation in Google Colab6, which already provides suf-
ﬁcient computational resources for our implementation. In future research,
we plan to incorporate a human-in-the-loop component to fur ther mitigate
the risk of LLM hallucinations and increase validity. Addit ionally, further
tests to compare results with diﬀerent news sources and leng ths (titles vs.
full-text) are planned in order to determine the minimal amo unt of validated
news data that is required for our approach.
5https://github.com/lisagrobels/NTLRAG .
6Google Colab provides free access to computing resources https://colab.google/
25

Declaration of competing interest
The authors declare that they have no known competing ﬁnanci al inter-
ests or personal relationships that could have appeared to i nﬂuence the work
reported in this paper.
CRediT authorship contribution statement
Lisa Grobelscheg: Conceptualization, Methodology, Software, Valida-
tion, Formal analysis, Investigation, Resources, Writing - original draft, Writ-
ing - review & Editing Visualization, Software, Methodolog y, Data curation,
Validation. Ema Kahr: Conceptualization, Methodology, Software, Vali-
dation, Investigation, Supervision, Data Curation, Writi ng - original draft,
writing - review & editing. Mark Strembeck: Conceptualization, Method-
ology, Writing - Review & Editing, Visualization, Supervis ion.
Funding Statement
This research received no external funding.
Ethical Approval Statement
The study adhered to the ethical principles of the Declarati on of Helsinki.
All participant information was handled conﬁdentially, an d data privacy was
strictly maintained throughout the study.
Declaration of AI using Assisted Technologies
During the preparation of this work, the authors used Gramma rly and
ChatGPT to improve English clarity and conduct spelling and grammar
checks. After using these tools, the authors reviewed and ed ited the con-
tent as needed and take full responsibility for the content o f the published
article.
Data Availability Statement
Regarding the datasets used in our case studies, we will shar e only insights
derived from them due to restrictions imposed by the social m edia platform
(X, formerly known as Twitter).
26

Code availability
The code for our implementation is publicly available on Git Hub.7
Appendix A. Prompts
Prompt for narrative extraction:
You are an information extraction system.
Your task:
From the following documents, extract ONLY the information present to ﬁll
the following JSON object:
’actor’: ”,
’action’: ”,
’event’: ”,
’description’: ”
Rules:
- STRICTLY use only the information found in the provided doc uments.
- Absolutely NO external knowledge, assumptions, or inferr ed details.
- Your output will be discarded if it contains information no t directly from
the documents.
- ’action’ should include at least one verb.
- ’event’ is the object of the action and can include nouns and noun phrases.
- ’actor’ can be any entity or multiple entities (individual , group, institution,
public entity, country, etc.).
- ONLY if you cannot determine an ’actor’, use ’user’.
- ’description’ must summarize the narrative in one sentenc e and must be
consistent with ’actor’,’action’ and ’event’.
- Output ONLY the JSON object, nothing else.
DOCUMENTS:
——————-
”’ )
Prompt for narrative validation:
You are a narrative fact-checker. Your task is to analyze a na rrative in the
7https://github.com/lisagrobels/NTLRAG
27

context of supporting documents and determine if it is consi stent.
Rules for Labeling
Start by assuming the narrative is **approved**. Change it t o **reﬁne**
only if:
1. The narrative **contradicts** the context (i.e. directl y conﬂicts).
2. The narrative includes hallucinations (i.e. facts not pr esent in the con-
text).
Approve if:
- The narrative is CONSISTENT with the context.
- The narrative does not contradict the context (i.e. tells t he opposite).
- Approximate matches exist (e.g. ’America’ and ’US’).
- The actor is ’user’ (this is always valid and must be **appro ved** if other
ﬁelds are valid).
Do NOT:
- Guess or invent information.
- Consider grammar, tone, or style.
- Penalize narratives that are vague but not contradictory. ”’
’Use the LabeledNarrative schema with ﬁelds:’
’- label: Either ’approved’ or ’reﬁne”
’- explanation: A short explanation for the decision.’
f’Context:context’
f’Narrative:narrative’
)
28

Appendix B. Evaluation App
Screenshot of the Evaluation page in the corresponding shin y app
Figure B.7: Screenshot of the Evaluation App, Evaluation pa ge.
29

References
Afzal, A., Vladika, J., Fazlija, G., Staradubets, A., and Ma tthes, F. (2024).
Towards optimizing a retrieval augmented generation using large language
model on academic data.
Akiba, T., Sano, S., Yanase, T., Ohta, T., and Koyama, M. (201 9). Optuna:
A next-generation hyperparameter optimization framework . InProceed-
ings of the 25th ACM SIGKDD International Conference on Know ledge
Discovery and Data Mining .
Aziz, S., Dowling, M., Hammami, H., and Piepenbrink, A. (202 1). Machine
learning in ﬁnance: A topic modeling approach. European Financial Man-
agement , 28(3):744–770.
Bianchi, F., Terragni, S., and Hovy, D. (2021a). Pre-traini ng is a hot topic:
Contextualized document embeddings improve topic coheren ce. In Pro-
ceedings of the 59th Annual Meeting of the Association for Co mputational
Linguistics and the 11th International Joint Conference on Natural Lan-
guage Processing (Volume 2: Short Papers) , pages 759–766, Online. Asso-
ciation for Computational Linguistics.
Bianchi, F., Terragni, S., Hovy, D., Nozza, D., and Fersini, E. (2021b). Cross-
lingual contextualized topic models with zero-shot learni ng. In Proceed-
ings of the 16th Conference of the European Chapter of the Ass ociation
for Computational Linguistics: Main Volume , pages 1676–1683, Online.
Association for Computational Linguistics.
Blei, D. M., Ng, A. Y., and Jordan, M. I. (2003). Latent dirich let allocation.
J. Mach. Learn. Res. , 3(null):993–1022.
Campello, R. J. G. B., Moulavi, D., and Sander, J. (2013). Den sity-based
clustering based on hierarchical density estimates. In Pei , J., Tseng, V. S.,
Cao, L., Motoda, H., and Xu, G., editors, Advances in Knowledge Discov-
ery and Data Mining , pages 160–172, Berlin, Heidelberg. Springer Berlin
Heidelberg.
Chambers, N. (2013). Event schema induction with a probabil istic entity-
driven model. In Yarowsky, D., Baldwin, T., Korhonen, A., Li vescu, K.,
30

and Bethard, S., editors, Proceedings of the 2013 Conference on Empir-
ical Methods in Natural Language Processing , pages 1797–1807, Seattle,
Washington, USA. Association for Computational Linguisti cs.
Chambers, N. and Jurafsky, D. (2008). Unsupervised learnin g of narrative
event chains. In Moore, J. D., Teufel, S., Allan, J., and Furu i, S., editors,
Proceedings of ACL-08: HLT , pages 789–797, Columbus, Ohio. Association
for Computational Linguistics.
Chambers, N. and Jurafsky, D. (2009). Unsupervised learnin g of narrative
schemas and their participants. In Proceedings of the Joint Conference
of the 47th Annual Meeting of the ACL and the 4th Internationa l Joint
Conference on Natural Language Processing of the AFNLP: Vol ume 2 -
Volume 2 , ACL ’09, page 602–610, USA. Association for Computational
Linguistics.
Chandra, R. and Saini, R. (2021). Biden vs trump: Modeling us general
elections using bert language model. IEEE Access , 9:128494–128505.
Chang, J., Gerrish, S., Wang, C., Boyd-graber, J., and Blei, D. (2009). Read-
ing tea leaves: How humans interpret topic models. In Bengio , Y., Schu-
urmans, D., Laﬀerty, J., Williams, C., and Culotta, A., edit ors,Advances
in Neural Information Processing Systems , volume 22, pages pp. 288–296.
Curran Associates, Inc.
Chang, W., Cheng, J., Allaire, J., Sievert, C., Schloerke, B ., Aden-Buie, G.,
Xie, Y., Allen, J., McPherson, J., Dipert, A., and Borges, B. (2025). shiny:
Web Application Framework for R . R package version 1.11.1.9000.
Chase, H. (2022). LangChain.
Chaudhary, A., Milios, E., and Rajabi, E. (2024). Top2label : Explainable
zero shot topic labelling using knowledge graphs. Expert Systems with
Applications , 242:122676.
Chen, L.-C. (2024). An extended tf-idf method for improving keyword ex-
traction in traditional corpus-based research: An example of a climate
change corpus. Data & Knowledge Engineering , 153:102322.
Chroma Team (2025). Chroma: Open-source vector database.
https://www.trychroma.com/ . Version 1.0.15, accessed 2025-08-13.
31

Churchill, R. and Singh, L. (2022). The evolution of topic mo deling. ACM
Comput. Surv. , 54(10s).
CNN (2018). Alleged shooter at Texas high school
spared people he liked, court document says.
https://edition.cnn.com/2018/05/18/us/texas-school- shooting .
Accessed: 2026-02-05.
CNN (2019). El Paso suspect told po-
lice he was targeting Mexicans, aﬃdavit says.
https://edition.cnn.com/2019/08/09/us/el-paso-shoot ing-friday .
Accessed: 2026-02-05.
Colvin, S., Jolibois, E., Ramezani, H., Garcia Badaracco, A ., Dorsey, T.,
Montague, D., Matveenko, S., Trylesinski, M., Runkle, S., H ewitt, D.,
Hall, A., and Plot, V. (2025). Pydantic Validation.
de Marcos, L. and Domínguez-Díaz, A. (2025). Llm-based topi c modeling for
dark web q&a forums: A comparative analysis with traditiona l methods.
IEEE Access , 13:67159–67169.
Ennajari, H., Bouguila, N., and Bentahar, J. (2025). Correl ated topic mod-
eling for short texts in spherical embedding spaces. IEEE Transactions on
Pattern Analysis and Machine Intelligence , 47(6):4567–4578.
Ford, J. M. (2004). Content analysis: An introduction to its methodology
(2nd edition). Personnel Psychology , 57(4):1110–1113. Copyright - Copy-
right Personnel Psychology, Inc. Winter 2004; Document fea ture - refer-
ences; Last updated - 2024-08-07; SubjectsTermNotLitGenr eText - United
States–US; Focus Groups; Construct Validity; Achievement Need; Infer-
ences; Computers; Interviews; Assessment Centers (Person nel); Electronic
Equipment; Case Studies; Research Design; Qualitative Res earch; Essays;
Coding; Computational Linguistics; Semantics; Sampling; Validity; Re-
searchers; Statistical Analysis; Semiotics; Critical Inc idents Method.
Grobelscheg, L., Kušen, E., and Strembeck, M. (2022a). Auto mated Narra-
tives: On the Inﬂuence of Bots in Narratives during the 2020 V ienna Terror
Attack. In Proc. of the 7th International Conference on Complexity, Fu ture
Information Systems and Risk (COMPLEXIS) , pages 15–25.
32

Grobelscheg, L., Sliwa, K., Kušen, E., and Strembeck, M. (20 22b). On
the dynamics of narratives of crisis during terror attacks. In2022 Ninth
International Conference on Social Networks Analysis, Man agement and
Security (SNAMS) , pages 1–8.
Grootendorst, M. (2022). Bertopic: Neural topic modeling w ith a class-based
tf-idf procedure.
Guillén-Pacho, I., Badenes-Olmedo, C., and Corcho, O. (202 4). Dynamic
topic modelling for exploring the scientiﬁc literature on c oronavirus: an
unsupervised labelling technique. International Journal of Data Science
and Analytics .
Herman, D. (2009). Basic Elements of Narrative . Wiley.
Hughes, J. (2021). krippendorﬀsalpha: An r package for meas uring agreement
using krippendorﬀ’s alpha coeﬃcient.
Kasiviswanathan, S. P., Melville, P., Banerjee, A., and Sin dhwani, V. (2011).
Emerging topic detection using dictionary learning. In Proceedings of the
20th ACM International Conference on Information and Knowl edge Man-
agement , CIKM ’11, page 745–754, New York, NY, USA. Association for
Computing Machinery.
Khaliq, M. A., Chang, P. Y.-C., Ma, M., Pﬂugfelder, B., and Mi letić,
F. (2024). RAGAR, your falsehood radar: RAG-augmented reas oning
for political fact-checking using multimodal large langua ge models. In
Schlichtkrull, M., Chen, Y., Whitehouse, C., Deng, Z., Akht ar, M., Aly,
R., Guo, Z., Christodoulopoulos, C., Cocarascu, O., Mittal , A., Thorne, J.,
and Vlachos, A., editors, Proceedings of the Seventh Fact Extraction and
VERiﬁcation Workshop (FEVER) , pages 280–296, Miami, Florida, USA.
Association for Computational Linguistics.
Kušen, E. and Strembeck, M. (2021a). Building blocks of comm unication net-
works in times of crises: Emotion-exchange motifs. Computers in Human
Behavior , 123:106883.
Kušen, E. and Strembeck, M. (2021b). Emotional Communicati on During
Crisis Events: Mining Structural OSN Patterns. IEEE Internet Comput-
ing, 25(02):58–65.
33

Lam, M. S., Teoh, J., Landay, J. A., Heer, J., and Bernstein, M . S. (2024).
Concept induction: Analyzing unstructured text with high- level concepts
using lloom. In Proceedings of the 2024 CHI Conference on Human Factors
in Computing Systems , CHI ’24, pages pp. 1–28, New York, NY, USA.
Association for Computing Machinery.
Lau, J. H. and Baldwin, T. (2016). The sensitivity of topic co herence eval-
uation to topic cardinality. In Knight, K., Nenkova, A., and Rambow,
O., editors, Proceedings of the 2016 Conference of the North American
Chapter of the Association for Computational Linguistics: Human Lan-
guage Technologies , pages 483–487, San Diego, California. Association for
Computational Linguistics.
Lau, J. H., Newman, D., and Baldwin, T. (2014). Machine readi ng tea
leaves: Automatically evaluating topic coherence and topi c model quality.
In Wintner, S., Goldwater, S., and Riezler, S., editors, Proceedings of the
14th Conference of the European Chapter of the Association f or Compu-
tational Linguistics , pages 530–539, Gothenburg, Sweden. Association for
Computational Linguistics.
Laureate, C. D. P., Buntine, W., and Linger, H. (2023). A syst ematic review
of the use of topic models for short text social media analysi s.Artiﬁcial
Intelligence Review , 56(12):14223–14255.
Lu, Q. and Chesbrough, H. (2022). Measuring open innovation practices
through topic modelling: Revisiting their impact on ﬁrm ﬁna ncial perfor-
mance. Technovation , 114:102434.
Mei, Q., Shen, X., and Zhai, C. (2007). Automatic labeling of multinomial
topic models. In Proceedings of the 13th ACM SIGKDD International
Conference on Knowledge Discovery and Data Mining , KDD ’07, page
490–499, New York, NY, USA. Association for Computing Machi nery.
Mekaoui, S., Chaker, I., Zarghili, A., and Nikolov, N. S. (20 25). Systematic
literature review of topic labeling. IEEE Access , 13:93124–93147.
Merrill, J. B. (2007). Stories of narrative: On social scien tiﬁc uses of narrative
in multiple disciplines.
34

Mimno, D., Wallach, H., Talley, E., Leenders, M., and McCall um, A. (2011).
Optimizing semantic coherence in topic models. In Barzilay , R. and John-
son, M., editors, Proceedings of the 2011 Conference on Empirical Methods
in Natural Language Processing , pages 262–272, Edinburgh, Scotland, UK.
Association for Computational Linguistics.
NBC News (2019). Las Vegas police release report
on lessons from 2017 mass shooting that killed 58.
https://www.nbcnews.com/storyline/las-vegas-shootin g/las-vegas-police-release-re 
Accessed: 2026-02-05.
Newman, D., Lau, J. H., Grieser, K., and Baldwin, T. (2010). A utomatic
evaluation of topic coherence. In Kaplan, R., Burstein, J., Harper, M.,
and Penn, G., editors, Human Language Technologies: The 2010 Annual
Conference of the North American Chapter of the Association for Compu-
tational Linguistics , pages 100–108, Los Angeles, California. Association
for Computational Linguistics.
Piper, A., So, R. J., and Bamman, D. (2021). Narrative theory for compu-
tational narrative understanding. In Moens, M.-F., Huang, X., Specia, L.,
and Yih, S. W.-t., editors, Proceedings of the 2021 Conference on Empir-
ical Methods in Natural Language Processing , pages 298–311, Online and
Punta Cana, Dominican Republic. Association for Computati onal Linguis-
tics.
Piper, A. and Wu, S. (2025). Evaluating large language model s for narrative
topic labeling. In Hämäläinen, M., Öhman, E., Bizzoni, Y., M iyagawa, S.,
and Alnajjar, K., editors, Proceedings of the 5th International Conference
on Natural Language Processing for Digital Humanities , pages 281–291,
Albuquerque, USA. Association for Computational Linguist ics.
Roberts, M. E., Stewart, B. M., and Tingley, D. (2019). stm: A n r package
for structural topic models. Journal of Statistical Software , 91(2):1–40.
Robertson, S. E., Walker, S., Jones, S., Hancock-Beaulieu, M., and Gatford,
M. (1995). Okapi at trec-3. In Harman, D., editor, Proceedings of the Third
Text REtrieval Conference (TREC-3) , pages 109–126, Gaithersburg, MD,
USA. National Institute of Standards and Technology (NIST) .
Roos, M. and Reccius, M. (2024). Narratives in economics. Journal of
Economic Surveys , 38(2):303–341.
35

Samory, M. and Mitra, T. (2018). ’the government spies using our webcams’:
The language of conspiracy theories in online discussions. Proc. ACM
Hum.-Comput. Interact. , 2(CSCW).
Santana, B., Campos, R., Amorim, E., Jorge, A., Silvano, P., and Nunes,
S. (2023). A survey on narrative extraction from textual dat a.Artiﬁcial
Intelligence Review , 56(8):8393–8435.
Speer, R., Chin, J., and Havasi, C. (2017). Conceptnet 5.5: A n open multi-
lingual graph of general knowledge. Proceedings of the AAAI Conference
on Artiﬁcial Intelligence , 31(1).
Thompson, L. and Mimno, D. (2020). Topic modeling with conte xtualized
word representation clusters.
Wanna, S., Solovyev, N., Barron, R., Eren, M. E., Bhattarai, M., Rasmussen,
K. O., and Alexandrov, B. S. (2024). Topictag: Automatic ann otation of
nmf topic models using chain of thought and prompt tuning wit h llms. In
Proceedings of the ACM Symposium on Document Engineering 20 24, Do-
cEng ’24, pages pp. 1–4, New York, NY, USA. Association for Co mputing
Machinery.
Weick, K. E. (1995). Sensemaking in organizations , volume 3. Sage.
Williams, L., Anthi, E., Arman, L., and Burnap, P. (2024). To pic modelling:
Going beyond token outputs. Big Data and Cognitive Computing , 8(5).
36