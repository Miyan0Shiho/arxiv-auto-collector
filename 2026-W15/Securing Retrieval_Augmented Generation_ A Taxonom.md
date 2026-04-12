# Securing Retrieval-Augmented Generation: A Taxonomy of Attacks, Defenses, and Future Directions

**Authors**: Yuming Xu, Mingtao Zhang, Zhuohan Ge, Haoyang Li, Nicole Hu, Jason Chen Zhang, Qing Li, Lei Chen

**Published**: 2026-04-09 14:38:18

**PDF URL**: [https://arxiv.org/pdf/2604.08304v1](https://arxiv.org/pdf/2604.08304v1)

## Abstract
Retrieval-augmented generation (RAG) significantly enhances large language models (LLMs) but introduces novel security risks through external knowledge access. While existing studies cover various RAG vulnerabilities, they often conflate inherent LLM risks with those specifically introduced by RAG. In this paper, we propose that secure RAG is fundamentally about the security of the external knowledge-access pipeline. We establish an operational boundary to separate inherent LLM flaws from RAG-introduced or RAG-amplified threats. Guided by this perspective, we abstract the RAG workflow into six stages and organize the literature around three trust boundaries and four primary security surfaces, including pre-retrieval knowledge corruption, retrieval-time access manipulation, downstream context exploitation, and knowledge exfiltration. By systematically reviewing the corresponding attacks, defenses, remediation mechanisms, and evaluation benchmarks, we reveal that current defenses remain largely reactive and fragmented. Finally, we discuss these gaps and highlight future directions toward layered, boundary-aware protection across the entire knowledge-access lifecycle.

## Full Text


<!-- PDF content starts -->

Securing Retrieval-Augmented Generation: A Taxonomy of Attacks,
Defenses, and Future Directions
Yuming XU1∗, Mingtao ZHANG1∗, Zhuohan GE1∗, Haoyang LI1, Nicole HU1,
Jason Chen ZHANG1, Qing LI1, Lei CHEN2
1The Hong Kong Polytechnic University
2The Hong Kong University of Science and Technology (Guangzhou)
martin.xu@connect.polyu.hk
Abstract
Retrieval-augmented generation (RAG) signifi-
cantly enhances large language models (LLMs)
but introduces novel security risks through ex-
ternal knowledge access. While existing stud-
ies cover various RAG vulnerabilities, they of-
ten conflate inherent LLM risks with those
specifically introduced by RAG. In this pa-
per, we propose that secure RAG is funda-
mentally about the security of the external
knowledge-access pipeline. We establish an
operational boundary to separate inherent LLM
flaws from RAG-introduced or RAG-amplified
threats. Guided by this perspective, we abstract
the RAG workflow into six stages and orga-
nize the literature around three trust boundaries
and four primary security surfaces, including
pre-retrieval knowledge corruption, retrieval-
time access manipulation, downstream con-
text exploitation, and knowledge exfiltration.
By systematically reviewing the correspond-
ing attacks, defenses, remediation mechanisms,
and evaluation benchmarks, we reveal that
current defenses remain largely reactive and
fragmented. Finally, we discuss these gaps
and highlight future directions toward layered,
boundary-aware protection across the entire
knowledge-access lifecycle.
1 Introduction
Retrieval-augmented generation (RAG) has be-
come a practical and widely adopted paradigm for
improving large language models (LLMs) with ex-
ternal knowledge at inference time (Lewis et al.,
2020; Gao et al., 2023; Wu et al., 2024; Gupta
et al., 2024; Cheng et al., 2025a). By retrieving
evidence from external corpora, databases, or struc-
tured repositories and incorporating it into the gen-
eration context, RAG improves factuality, update-
ability, and domain adaptability (Gao et al., 2023;
Wu et al., 2024; Yu et al., 2024; Gan et al., 2025).
∗denotes equal contribution.At the same time, this shift from parametric knowl-
edge alone to external knowledge access introduces
important security risks, because the model is no
longer affected only by its parameters and the user
prompt, but also by the content and access path
of external knowledge (Ward and Harguess, 2025;
Arzanipour et al., 2025; Ammann et al., 2025).
Existing surveys have already advanced the
study of RAG from several important perspectives.
General RAG surveys review architectures, com-
ponents, and applications (Gao et al., 2023; Wu
et al., 2024; Gupta et al., 2024; Cheng et al., 2025a;
Sharma, 2025), evaluation-oriented surveys sum-
marize metrics, frameworks, and benchmarks (Yu
et al., 2024; Gan et al., 2025), while trust, security,
and privacy oriented surveys discuss broader trust-
worthiness dimensions, deployment risks, threat
models, or privacy issues (Zhou et al., 2024; Ni
et al., 2025; Ammann et al., 2025; Ward and Har-
guess, 2025; Arzanipour et al., 2025; Bodea et al.,
2026). Surveys on general LLM security, pri-
vacy, and trustworthy agent systems further clar-
ify the broader ecosystem where secure RAG is
situated (Yao et al., 2024; Das et al., 2025; Yu
et al., 2025; Kim et al., 2026). However,exter-
nal knowledge accesshas not yet been established
as the main organizing principle for surveying se-
cure RAG. As a result, the boundary between in-
herent LLM risks and risks introduced or mate-
rially amplified by external knowledge access of-
ten remains under-specified, and the connections
among attacks, defenses, and evaluation remain
fragmented under this scope.
Figure 1 gives the basic intuition. In secure
RAG, the attacker often does not need to alter
model parameters or place the malicious instruc-
tion directly in the user prompt. Instead, it can
manipulate external content, retrieval behavior, or
disclosure behavior so that harmful evidence en-
ters the model-visible context, or so that sensitive
knowledge is inferred or extracted through the re-
1arXiv:2604.08304v1  [cs.CR]  9 Apr 2026

User
Query
External
Knowledg e
Retriever
 Retrieved
Contex t
Language
Mode lAnswer
RAG Defense
RAG Attack Redirect
PoisonInject Mislead
Extract
Isolation HardeningIntegrity Access ControlFigure 1: Intuition of attack and defense along the RAG
knowledge-access pipeline.
sponse interface (Choi et al., 2025; Chang et al.,
2026). Defenses therefore should not be viewed
only as general LLM safety measures. They must
be understood as controls distributed along the full
knowledge-access path.
To bridge this gap, in this paper, we introduce
an important distinction.Inherent LLM risksarise
from the parametric model, its prompt interface,
or its generation behavior, such as prompt-only
jailbreaks or parametric memorization. By contrast,
RAG-introduced or RAG-amplified risksdepend on
external non-parametric knowledge access, where
retrieval creates a new entry point, a new disclosure
channel, or a mechanism that makes the threat more
persistent, transferable, and harder to remediate.
Section 2.2 formalizes this operational boundary.
This focus is important because external knowl-
edge access transforms many security failures from
transient, query-local events into persistent com-
promises of a shared knowledge substrate, whose
effects can be reused across queries, transferred
across users, and become harder to detect, attribute,
and remove. Moreover, research on this topic has
grown rapidly, spanning attack methods, defense
mechanisms, remediation strategies, and evaluation
protocols. A dedicated survey is therefore needed
to provide a clear scope, a systematic taxonomy,
and a structured view of the rapidly growing litera-
ture on secure knowledge access in RAG.
Under this scope, we organize secure RAG
around four security surfaces: pre-retrieval
knowledge-substrate corruption, retrieval-time ac-
cess manipulation, downstream retrieved-context
exploitation, and knowledge exfiltration and pri-
vacy attacks. We then review the corresponding
defense and remediation layers and summarize re-
cent evaluation studies and benchmarks.
Against this backdrop, Our paper makes the fol-lowing contributions.
•We present a concise pipeline view of RAG for
systematic security analysis, and relate a six-
stage RAG pipeline to four primary security
surfaces and three trust boundaries.
•We define an operational boundary that sepa-
rates inherent LLM risks from RAG-introduced
or RAG-amplified risks, thereby clarifying
which attacks, defenses, and evaluation studies
fall within the scope of secure RAG.
•We organize the attack literature and the de-
fense and remediation literature through the
same external knowledge access centric frame,
which makes the relation between threat entry
points and control points explicit.
• We summarize benchmark studies and system-
atic evaluation studies for secure RAG, and dis-
till observations and future directions for more
reusable and threat-model-aware evaluation.
2 RAG Pipeline, Security Surfaces, and
Operational Scope
2.1 RAG Pipeline, Trust Boundaries, and
Security Surfaces
Viewed through the lens of generation grounded in
external non-parametric knowledge (Lewis et al.,
2020; Xiao et al., 2026; Gao et al., 2023), we ab-
stract a standard retrieval-augmented generation
(RAG) system into a six-stage workflow as follows:
first, external sources provide diverse raw content;
second, ingestion pipelines parse, transform, and
index this content into a searchable format; third,
retrieval and reranking mechanisms select the most
relevant candidate evidence for a user query; next,
context assembly processes and formats this evi-
dence to construct the final model-visible prompt;
then the generator synthesizes an answer condi-
tioned on this assembled context; and finally, the
system delivers the response alongside necessary
logging, auditing, and potential remediation.
In this paper, we refer to the indexed external
corpus, database, or multimodal store from which
evidence is drawn as theknowledge substrate. Cor-
respondingly, we refer to the entire inference-time
process of selecting and incorporating this evidence
asexternal knowledge access.
As shown in Figure 2, the pipeline view clearly
exposes three trust boundaries and corresponding
2

•Documents
•Web Pages
•Databases
•Code ReposExternal
Sources
•Collection
•Parsing
•Chunking
•Embedding
•IndexingIngestion &
Indexing
•Query
•Formulation
•Search
•Ranking
•SelectionRetriever &
Reranking
•Top-k 
•Filtering
•Ordering
•Prompt
•ConstructionContext
Assembly
•LLM
Inference
•Response
GenerationGeneration
•Output
•Action
•Logging
•TrackingResponse &
RemediationPre-Retrieval
Substrate CorruptionRetrieval -Time
Access 
ManipulatePost-Retrieval
Context ExploitationKnowledge 
Exfiltration &
Privacy Attacks
User QueryBoundary: 
Untrusted external
content enters the systemKnowledge Substrate
Boundary: 
External content becomes 
model -visible contextBoundary: 
Model output may act
on retrieved content
RAG Knowledge  
Access PipelineSecurity 
SurfacesFigure 2: RAG knowledge-access pipeline, security surfaces, and trust boundaries.
security surfaces at each stage. Untrusted exter-
nal content first enters the system during ingestion
and indexing, creating a surface forpre-retrieval
knowledge-substrate corruptionwhere attackers
can poison the underlying corpus. A first upstream
boundary is crossed when untrusted external con-
tent enters the system through ingestion and index-
ing. As the system processes a query,retrieval-time
access manipulationcan occur, where attackers at-
tempt to distort, redirect, or suppress the selection
of evidence. After the most relevant context is se-
lected and assembled, the retrieved evidence will
be sent into the model-visible context, which in-
troduces the risk ofdownstream retrieved-context
exploitation, allowing external data to directly ma-
nipulate the generation behavior. A second, and
also the most important, boundary is crossed when
retrieved evidence becomesmodel-visible context,
where external content can affect generation di-
rectly rather than only retrieval scores. Finally,
as the model delivers its output, the system faces
knowledge exfiltration and privacy attacks, where
adversaries exploit the RAG interface in reverse
to infer or extract sensitive information from the
knowledge substrate. A final downstream bound-
ary is crossed when the model turns retrieved con-
text into answers or actions. Together, these three
boundaries define the main control points of exter-
nal knowledge access for RAG.
2.2 Operational Boundary
This paper focuses onRAG-introducedandRAG-
amplifiedsecurity risks. Operationally, a risk is in
scope when external knowledge is the main carrierof the threat; when knowledge access creates a dis-
tinct entry point that does not exist in prompt-only
LLM use; or when retrieval materially increases
the persistence, transferability, or blast radius of
the threat.
Under this boundary, we include attack stud-
ies, defense and remediation studies, and secu-
rity evaluation studies whose main object lies in
the knowledge-access pipeline. We exclude broad
RAG studies whose main focus is answer quality
rather than security (Friel et al., 2024; Zhu et al.,
2025; Park et al., 2025; Strich et al., 2026; Peng
et al., 2025). We also exclude inherent LLM risks,
such as prompt-only jailbreaks and purely para-
metric memorization, whose main object is not
retrieval-coupled knowledge access (Li et al., 2024;
Lin et al., 2024; Yan et al., 2024; Jiang et al., 2024;
Carlini et al., 2021). Accordingly, retrieved-context
injection, poisoning, retrieval-coupled extraction,
disclosure control, and security benchmarks for
these risks are in scope, while generic prompt at-
tacks and non-security RAG evaluation are not.
3 Attack Taxonomy
We organize RAG attack methods by the stage at
which external knowledge enters the pipeline. As
illustrated in Figure 3, our taxonomy focuses on
where the adversarial payload enters, how access
to external knowledge is manipulated, and when
harmful external content crosses a trust bound-
ary. Following the operational boundary in Sec-
tion 2.2, the attack surface of secure RAG can be
divided into four families: pre-retrieval knowledge-
substrate corruption, retrieval-time access manipu-
3

RAG Attack MethodsPre-Retrieval
Knowledge-
Substrate CorruptionCorpus / Document
PoisoningPoisonedRAG (Zou et al., 2025),
BadRAG (Xue et al., 2024),
GARAG (Cho et al., 2024),
Retrieval Poisoning (Zhang et al., 2024a),
CorruptRAG (Zhang et al., 2025a),
AuthChain (Chang et al., 2025),
UniC-RAG (Geng et al., 2025),
HijackRAG (Zhang et al., 2024b)
Ingestion / Loader
PoisoningRAG Data Loaders (Castagnaro et al., 2025)
Structured / Multimodal
Knowledge PoisoningGRAGPoison (Liang et al., 2025a),
TKPA and UKPA (Wen et al., 2025a),
MM-PoisonRAG (Ha et al., 2025),
Poisoned-MRAG (Liu et al., 2025b),
One Pic (Shereen et al., 2026)
Code-Oriented
Knowledge PoisoningRACG (Lin et al., 2025),
ImportSnare (Ye et al., 2025b),
RAG-Pull (Stambolic et al., 2025)
Retrieval-Time Access
ManipulationQuery / Prompt
Perturbation
for Retrieval RedirectionGGPP (Hu et al., 2024),
ReGENT (Song et al., 2025)
Ranking / Re-
triever Compromise
/ Coordinated AttacksFlipedRAG (Chen et al., 2024),
Topic-FlipRAG (Gong et al., 2025),
Backdoored Retrievers (Clop and Teglia,
2024),
PR-Attack (Jiao et al., 2025)
Downstream
Retrieved-Context
ExploitationIndirect Prompt Injection
/ Jailbreak / BackdoorPANDORA (Deng et al., 2024),
Phantom (Chaudhari et al., 2024),
TrojanRAG (Cheng et al., 2024),
PIDP-Attack (Wang et al., 2026)
Availability / Denial
/ Refusal AbuseBlocker (Shafran et al., 2025),
MutedRAG (Suo et al., 2025)
Knowledge Exfiltration
and Privacy AttacksMembership Inference on
External Knowledge BasesMIA-RAG (Anderson et al., 2025),
S2MIA (Li et al., 2025),
MBA (Liu et al., 2025a),
IA (Naseh et al., 2025)
Document / Data
ExtractionSpill the Beans (Qi et al., 2025),
Backdoor Extraction (Peng et al., 2024),
IKEA (Wang et al., 2025c),
SECRET (He et al., 2025b)
Corpus-Scale / Multi-Turn
Extraction/ PivotingRAGCRAWLER (Yao et al., 2026),
Retrieval Pivot (Thornton, 2026)
Figure 3: Taxonomy of RAG Attack Methods.
4

lation, downstream retrieved-context exploitation,
and knowledge exfiltration and privacy attacks.
3.1 Pre-Retrieval Knowledge-Substrate
Corruption
Pre-retrieval attacks compromise the external
knowledge substrate before evidence is selected
for a user query. Their primary goal is to implant
malicious content that will later be surfaced by
retrieval and reused across queries, users, or ses-
sions. Compared with prompt-only attacks, they
often require extra write access to external sources,
ingestible files, or indexed repositories, while in
return they offer much stronger persistence and
shared-substrate blast radius once the poisoned con-
tent is admitted into the corpus.
A first line targetscorpus and document poi-
soning.PoisonedRAG(Zou et al., 2025) for-
mulates knowledge corruption as an optimiza-
tion problem, showing that injecting a few mali-
cious texts can induce targeted answers. Build-
ing on this,BadRAG(Xue et al., 2024) introduces
trigger-conditioned retrieval backdoors where poi-
soned passages remain dormant until activated by
specific queries. Then recent work develops to-
ward more practical and stealthy settings. For in-
stance,CorruptRAG(Zhang et al., 2025a) andAu-
thChain(Chang et al., 2025) demonstrate that a
single text optimized for retrievability and coher-
ence is enough for effective poisoning. Meanwhile,
UniC-RAG(Geng et al., 2025) extends the threat to
universal corruption by jointly optimizing a small
set of adversarial texts to attack diverse queries ef-
ficiently. Other studies explore stealthier carriers,
such as low-level perturbations inGARAG(Cho
et al., 2024), visually inconspicuous triggers in
Retrieval Poisoning(Zhang et al., 2024a), and
retrieval-prompt hijacking inHijackRAG(Zhang
et al., 2024b).
A second line attacks theingestion interface
rather than the final indexed text. Research onRAG
Data Loaders(Castagnaro et al., 2025) shows that
malicious content hidden in common document
formats can be silently introduced during parsing
and loading. This view refines the poisoning threat
model, demonstrating that attackers can exploit the
ingestion toolchain as the attack carrier without
directly editing the final corpus.
A third line extends poisoning intostructured
and multimodal knowledge. In GraphRAG systems,
GRAGPoison(Liang et al., 2025a) uses graph rela-
tions to poison multiple queries, focusing on attackpaths introduced by shared connections.TKPA
and UKPA(Wen et al., 2025a) reveal that min-
imal textual perturbations can significantly dis-
tort graph-based retrieval. In multimodal contexts,
MM-PoisonRAG(Ha et al., 2025) andPoisoned-
MRAG(Liu et al., 2025b) inject crafted image-
text pairs to steer retrieval. Furthermore,One
Pic(Shereen et al., 2026) demonstrates that a single
optimized image can inject targeted disinformation
or act as a universal denial-of-service payload in
visual-document RAG.
A fourth line focuses oncode-oriented knowl-
edge poisoning. Studies such asRACG(Lin et al.,
2025) indicate that poisoning external code ex-
amples can propagate vulnerabilities into the gen-
erated code.ImportSnare(Ye et al., 2025b) ex-
tends this supply-chain view by hijacking retrieved
documentation to make the model recommend
attacker-controlled dependencies. Similarly,RAG-
Pull(Stambolic et al., 2025) shows that hidden
perturbations in external repositories can redirect
retrieval toward malicious code.
Overall, this family shares a common property:
the attack payload is plantedbefore retrievaland
reused by the system as legitimate external knowl-
edge. This mechanism makes pre-retrieval cor-
ruption the most persistent attack surface in se-
cure RAG, expanding threats beyond transient
prompt-level failures by breaching the upstream
trust boundary defined in Section 2.1.
3.2 Retrieval-Time Access Manipulation
Retrieval-time attacks do not primarily rely on
broad substrate corruption. Instead, they select mis-
leading information desired by the attacker from
the existing corpus, typically by altering the rel-
evance ranking of the documents surfaced for a
query. These attacks are usually specific to a par-
ticular query, making their disruptive persistence
lower than that of corpus poisoning. However, be-
cause they can be initiated from the query side
without altering the database, they remain highly
effective even in black-box settings where the at-
tacker can only probe the retrieval interface.
One line of work focuses onquery perturbation
for retrieval redirection. For example,GGPP(Hu
et al., 2024) shows that adding a short, optimized
sequence to the user’s input can trick the retriever
into fetching factually incorrect documents. Taking
a dynamic approach,ReGENT(Song et al., 2025)
proposes a reinforcement learning framework to
optimize word-level substitutions within target doc-
5

uments, generating imperceptible corpus poisoning
payloads that can successfully hijack the generation
while remaining natural to human readers. Notably,
while these attacks modify the user’s input, their
sole objective is to hijack the knowledge retrieval
path instead of injecting malicious execution in-
structions into the LLM generator, which separates
them from out-of-scope prompt jailbreaks.
A second line targets theretrievers and rank-
ing mechanismsdirectly.FlipedRAG(Chen et al.,
2024) andTopic-FlipRAG(Gong et al., 2025) use
surrogate models to craft queries that shift the
stance of the retrieved evidence, enabling opin-
ion manipulation across related topic clusters. At
the system level,Backdoored Retrievers(Clop and
Teglia, 2024) compromises the dense retriever dur-
ing fine-tuning, ensuring it preferentially ranks
attacker-controlled content. Furthermore,PR-
Attack(Jiao et al., 2025) introduces a coordinated
threat by pairing poisoned texts with specific query
triggers to exploit the retrieval matching process.
Overall, this family demonstrates that secure
RAG can be compromised simply by attacking the
access pathto external knowledge. It shows that
attackers can manipulate the model-visible context
without requiring large-scale database corruption
by attacking the access path that determines what
eventually crosses the second trust boundary.
3.3 Downstream Retrieved-Context
Exploitation
The third family assumes that malicious exter-
nal content has successfully crossed the retrieval
boundary and become model-visible context. Un-
like the previous two categories that aim to feed
misleading information for the LLM generator to
process normally, this category focuses on actively
controlling the generator’s behavior through the
retrieved context. Following our boundary defined
in Section 2.2, we strictly limit our discussion to
threats where malicious instructions are carried by
retrieved external knowledge rather than initiated
by the user’s query prompt. This distinction is cru-
cial due to the extreme stealthiness of such attacks.
Because the user’s query remains completely be-
nign, conventional prompt-filtering mechanisms
commonly used in general LLM security usually
fail, allowing hidden payloads within the external
knowledge to silently hijack the system and alter
its actions.
One line of research explores indirect prompt
injection and jailbreaks in retrieved-context. Thesemethods embed malicious instructions within ex-
ternal documents to bypass model safety align-
ments. For example,PANDORA(Deng et al.,
2024) demonstrates that simply embedding jail-
break prompts into external documents causes the
LLM to unknowingly execute the hidden malicious
instructions as if they were valid context.Phan-
tom(Chaudhari et al., 2024) refines this into a
single-document trigger attack where a poisoned
document is retrieved only when a natural trigger
sequence appears in the user query, ultimately in-
ducing harmful behaviors or privacy abuse. Addi-
tionally,TrojanRAG(Cheng et al., 2024) provides
a joint backdoor setting that manipulates model
behavior through targeted retrieval contexts. More-
over,PIDP-Attack(Wang et al., 2026) combines
prompt injection principles with database poison-
ing to adaptively execute malicious instructions
regardless of the user’s input.
A second line targets system availability through
denial and refusal abuse. These attacks aim to
paralyze the system by forcing it to abstain from
answering legitimate questions.Blocker(Shafran
et al., 2025) demonstrates that a single retrieved
document can jam a RAG system and force it to
refuse answering without relying on explicit in-
struction injection. Similarly,MutedRAG(Suo
et al., 2025) reveals a subtle failure mode where
the attacker poisons the knowledge base with mini-
mal jailbreak texts designed purely to activate the
aligned model’s own safety guardrails. Conse-
quently, the system refuses to process otherwise
legitimate queries.
Overall, downstream exploitation highlights the
critical vulnerability of the model-visible context.
These attacks exploit the final downstream bound-
ary by turning model-visible external context into
harmful answers or refusals.
3.4 Knowledge Exfiltration and Privacy
Attacks
The fourth family focuses on stealing the external
knowledge substrate in reverse, instead of manipu-
lating the system’s output or behavior like the pre-
vious three categories. These attacks aim to infer
or recover sensitive information from the retrieval
database by exploiting retrieval-coupled signals in
the model’s responses. This threat is particularly
critical when RAG systems are deployed over pri-
vate, proprietary, or regulated corpora.
A first line of work performs membership infer-
ence on external knowledge bases. For instance,
6

MIA-RAG(Anderson et al., 2025) demonstrates that
the presence of a target document in the database
can be inferred through carefully designed prompts.
Building on this,S2MIA(Li et al., 2025) improves
inference accuracy by leveraging semantic simi-
larity between a target sample and the generated
text, whileMBA(Liu et al., 2025a) introduces
mask-based inference to reduce interference from
unrelated documents. Furthermore,IA(Naseh
et al., 2025) uses natural language questions whose
answers depend on the target document’s pres-
ence, making the attack hard to be noticed against
prompt-based detectors.
A second line seeks direct document or data-
level extraction. Studies such asSpill the Beans(Qi
et al., 2025) show that instruction-tuned RAG sys-
tems can be induced to regurgitate datastore con-
tent verbatim. Alternatively,Backdoor Extrac-
tion(Peng et al., 2024) demonstrates that a com-
promised LLM inside a benign RAG pipeline can
leak retrieved references. Recent work expands
this attack surface, withIKEA(Wang et al., 2025c)
performing implicit knowledge extraction with-
out explicit jailbreaks, andSECRET(He et al.,
2025b) formalizing the attack into extraction in-
structions, jailbreak operators, and retrieval triggers
to strengthen performance across diverse systems.
A third line studies corpus-scale multi-turn ex-
traction and pivoting.RAGCRAWLER(Yao et al.,
2026) formulates knowledge-base stealing as an
adaptive coverage problem, using a knowledge-
graph-guided state to plan extraction queries sys-
tematically. In hybrid RAG settings,Retrieval
Pivot(Thornton, 2026) reveals that a semantically
retrieved vector seed can pivot into sensitive graph
neighborhoods during expansion, highlighting how
the transition between retrieval mechanisms can
cause leakage.
Overall, this family shows that privacy risk in
RAG extends beyond parametric memorization to
interface-level recovery of external non-parametric
knowledge. The core threat is no longer private in-
formation stored in model parameters during train-
ing, but what attackers can infer or extract from
the external knowledge substrate through retrieval-
coupled interaction.
3.5 Summary and Observations
Overall, the current attack literature shows that
secure RAG risks are best analyzed through the
external knowledge-access pipeline view. The four
attack surfaces in this section differ in mechanism,but they share the same structural property that
untrusted external knowledge can be injected, redi-
rected, exploited, or disclosed after it enters the
system.
Among them, upstream poisoning is especially
important because it can persist inside a shared
knowledge substrate and repeatedly affect later
queries, users, and system outputs. Retrieval-time
manipulation and downstream retrieved-context ex-
ploitation highlight another risk that attackers can
mislead more secretly if they can steer what crosses
the retrieval boundary or how retrieved evidence in-
teracts with generation. Knowledge exfiltration and
privacy attacks further show that the knowledge-
access interface can be abused in the reverse direc-
tion to infer or extract sensitive external content.
Taken together, secure RAG attacks are not iso-
lated failures at answer time. They are pipeline-
level failures in how external knowledge is admit-
ted, accessed, exposed to the generator, and dis-
closed through the response interface.
4Defenses and Remediation Mechanisms
We organize the defense studies by their pri-
mary control point in the RAG knowledge-access
pipeline, mirroring the attack surfaces identified in
Section 3. Specifically, this section structures RAG
security mechanisms into four defensive layers: (i)
knowledge-base integrity, provenance, and remedi-
ation to counter pre-retrieval substrate corruption;
(ii) retrieval-time access hardening to mitigate ac-
cess manipulation; (iii) post-retrieval context iso-
lation and robust generation to thwart downstream
context exploitation; and (iv) access control, pri-
vacy, and confidentiality to prevent knowledge ex-
filtration and unauthorized disclosure. Figure 4
illustrates the RAG defense taxonomy, in which
hybrid methods are discussed in the subsection cor-
responding to their primary intervention stage.
4.1 Knowledge-Base Integrity, Provenance,
and Remediation
The first defense layer protects the integrity of the
external knowledge substrate. Its primary goals are
to prevent malicious content from entering the cor-
pus and to support remediation after compromise.
Compared with later-stage defenses, methods in
this layer either intervene earlier by acting at ad-
mission time through provenance and validation,
or respond later after an incident through attribu-
tion, audit, and rollback. These methods typically
7

RAG Defense and Remediation MechanismsKnowledge-
Base Integrity
& RemediationD-RAG (E_Andersen et al., 2025),
RAGForensics (Zhang et al., 2025b)
Retrieval-Time Access
HardeningReliability-Aware
AggregationRobustRAG (Xiang et al., 2024),
ReliabilityRAG (Shen et al., 2025)
Retrieval & Reranking
DefenseTrustRAG (Zhou et al., 2025a),
GRADA (Zheng et al., 2025),
RAGPart and RAGMask (Pathmanathan
et al., 2025),
RAGuard (Cheng et al., 2025c),
FilterRAG (Edemacu et al., 2025)
Hybrid Filtering &
GenerationSeCon-RAG (Si et al., 2025)
Post-Retrieval Context
IsolationPoison Detection &
FilteringRevPRAG (Tan et al., 2025),
A V Filter (Choudhary et al., 2026),
RAGDefender (Kim et al., 2025)
Attention & Interaction
ControlSDAG (Dekel et al., 2026)
Robust Generation
BaselinesDiscern-and-Answer (Hong et al., 2024),
InstructRAG (Wei et al., 2025),
Astute RAG (Wang et al., 2025a),
RbFT (Tu et al., 2025)
Access Control, Privacy
& ConfidentialityAuthorization &
Access ControlSD-RAG (Masoud et al., 2026),
AC-RAG (Chen et al., 2025)
Local DP &
Decoding ShieldsDPV oteRAG (Koga et al., 2024),
RAG with Differential Privacy (Grislain,
2025),
LPRAG (He et al., 2025a),
V AGUE-Gate (Hemmat et al., 2025),
PAD (Wang et al., 2025b),
InvisibleInk (Vinod et al., 2025)
Corpus
TransformationSAGE (Zeng et al., 2025a)
Secure Retrieval
BackendsRemoteRAG (Cheng et al., 2025b),
ppRAG (Ye et al., 2025a),
FRAG (Zhao, 2024)
Confidential
ArchitecturesFedE4RAG (Mao et al., 2025),
C-FedRAG (Addison et al., 2024),
Privacy-Aware RAG (Zhou et al., 2025b),
SAG (Zhou et al., 2025c),
SecureRAG (Bassit and Boddeti, 2025)
Figure 4: Taxonomy of RAG Defense and Remediation Mechanisms.
8

require access to corpus management workflows
or system logs, and they often trade deployment
simplicity for stronger data governance.
Existing work in this layer remains limited.
At ingestion time,D-RAG(E_Andersen et al.,
2025) emphasizes strict admission control through
blockchain-backed provenance and expert verifica-
tion before data is added to the knowledge base.
After compromise,RAGForensics(Zhang et al.,
2025b) focuses on traceback: it identifies which
poisoned passages are responsible for a malicious
generation, thereby supporting targeted removal
and post-incident remediation.
This layer is crucial because it is the only de-
fense family that directly governs the upstream
trust boundary defined in Section 2.1, rather than
reacting after corrupted knowledge has already en-
tered the shared substrate. Without such admission-
level governance, defenses remain inherently re-
active; for instance, many retrieval-time methods
merely attempt to filter harmful evidence after the
shared substrate has already been poisoned. How-
ever, current work in this direction is still sparse,
and systematic support for provenance, rollback,
and corpus recovery remains under-developed.
4.2 Retrieval-Time Access Hardening
The second defense layer secures the retrieval inter-
face before external content becomes model-visible
context. Its main objective is to prevent corrupted
or low-trust evidence from dominating the final ev-
idence set passed to the generator. These methods
operate at retrieval or reranking time, and usually
require access to retrieved candidates or retriever
outputs. While most current methods are empirical,
a small line of work provides formal robustness
guarantees. Their main trade-off is between adver-
sarial robustness and benign retrieval utility, often
incurring additional latency from filtering or aggre-
gation. We group retrieval-time defenses into three
families: reliability-aware aggregation, retrieval
and reranking purification, and hybrid retrieval-
generation hardening.
The first family seeks robustness through
reliability-aware aggregation.RobustRAG(Xiang
et al., 2024) generates responses from isolated evi-
dence groups and securely aggregates them, yield-
ing certifiable robustness against bounded retrieval
corruption.ReliabilityRAG(Shen et al., 2025) ex-
tends this line by explicitly using retriever-side reli-
ability signals, such as document rank or reliability
scores, to identify a consistent majority of evidencewith provable robustness under bounded corrup-
tion.
The second family directly purifies retrieved
candidates or reranked results.TrustRAG(Zhou
et al., 2025a) combines cluster-based filtering with
LLM self-assessment to remove suspicious or con-
flicting documents.GRADA(Zheng et al., 2025)
performs graph-based reranking based on the ob-
servation that adversarial documents may look
relevant to the query while remaining weakly
connected to benign documents in the retrieved
set. At the retriever level,RAGPart and RAG-
Mask(Pathmanathan et al., 2025) operate directly
on the retrieval model through document parti-
tioning and masking-based sensitivity analysis, re-
ducing attack impact without modifying the gen-
erator. Lightweight filtering methods, such as
RAGuard(Cheng et al., 2025c), expand the re-
trieval scope and then apply chunk-wise perplexity
and similarity filtering, whileFilterRAG(Edemacu
et al., 2025) distinguishes poisoned texts through
corpus-level statistical cues.
The third family combines retrieval filtering with
downstream consistency control.SeCon-RAG(Si
et al., 2025) first applies semantic and cluster-based
filtering, and then performs conflict-aware filtering
before final answer generation.
Overall, this layer is the most direct defense
counterpart to the first two attack families detailed
in Section 3. It explicitly attempts to stop cor-
rupted or manipulated evidence before it crosses
the model-visible boundary. Its main limitation re-
mains its sensitivity to adaptive attacks, especially
when poisoned passages are semantically well inte-
grated into the benign evidence distribution.
4.3 Post-Retrieval Context Isolation and
Robust Generation
The third defense layer assumes that harmful con-
tent has already passed retrieval-time filters and en-
tered the retrieved context. The objective therefore
shifts from prevention to detection and containment
after the model-visible boundary has been crossed.
These methods typically require access to the re-
trieved context, model internals, or generation-
time interactions, and they remain mostly empir-
ical rather than formally guaranteed. Their main
trade-off is that stronger isolation often requires
additional model access, inference overhead, or
architectural changes.
One line of work performspost-retrieval detec-
tion and filtering.RevPRAG(Tan et al., 2025) de-
9

tects poisoned responses through distinctive LLM
activation patterns during generation.AV Fil-
ter(Choudhary et al., 2026) instead uses passage-
level attention-variance signals to identify retrieved
passages that exert anomalously strong influence
on the output.RAGDefender(Kim et al., 2025)
offers a lighter-weight alternative by applying post-
retrieval machine learning (ML)-based filtering to
separate adversarial from benign passages.
A second line directly constrains howre-
trieved documents interact inside the genera-
tor.SDAG(Dekel et al., 2026) shows that stan-
dard causal attention can enable harmful cross-
document interactions, and replaces it with sparse
document attention that blocks cross-attention
across retrieved documents. This design is notable
because it treats poisoning not only as a content
problem, but also as an interaction problem inside
the generation mechanism.
A third line consists ofrobust-generation meth-
odsthat are not always security-native, but remain
useful as transferable baselines. Most of these
methods were originally proposed to handle imper-
fect retrieval, misinformation, or internal-external
knowledge conflicts, rather than adversarial secure-
RAG settings alone. However, they can still reduce
attack impact by helping the model verify, cross-
check, or discount compromised evidence after the
model-visible boundary has already been crossed.
For instance,Discern-and-Answer(Hong et al.,
2024) acts as a guardrail by using a discriminator to
identify and discard misleading content before the
model answers.InstructRAG(Wei et al., 2025) mit-
igates the impact of malicious context by enforcing
explicit denoising through self-synthesized ratio-
nales, preventing the model from blindly following
adversarial instructions.Astute RAG(Wang et al.,
2025a) neutralizes context manipulation by actively
identifying conflicts between the model’s internal
parametric knowledge and the externally retrieved
adversarial evidence. Additionally,RbFT(Tu et al.,
2025) inherently reduces the model’s susceptibility
to adversarial steering by fine-tuning it to remain
robust when exposed to misleading or counterfac-
tual context.
Overall, this layer provides the last major con-
tainment point after the model-visible boundary has
been crossed, explicitly securing the final down-
stream trust boundary defined in Section 2.1. It
can reduce the translation of harmful context into
unsafe answers or actions, although many methods
in this layer remain empirical.4.4 Access Control, Privacy, and
Confidentiality
The final defense layer focuses on controlling who
may access retrieved knowledge and how sensitive
information can be exposed, processed, or leaked.
Unlike integrity-oriented defenses, the main focus
here is not whether retrieved evidence is benign, but
whether it is authorized to be revealed and whether
retrieval and generation can proceed without violat-
ing confidentiality. These methods often intervene
at the system or architecture level, and their guar-
antees range from empirical policy enforcement to
formal differential privacy or cryptographic secu-
rity. Their main trade-off is stronger confidentiality
at the cost of system complexity, latency, or re-
duced retrieval fidelity.
One line of work enforcesauthorization and se-
lective disclosurebefore sensitive content reaches
the generator.SD-RAG(Masoud et al., 2026) is rep-
resentative in decoupling disclosure control from
generation, enforcing sanitization during retrieval
rather than relying on prompt-level refusal alone.
Meanwhile,Access Control RAG (AC-RAG)(Chen
et al., 2025) integrates fine-grained access control
explicitly into RAG workflows for sensitive do-
mains.
A second line protects privacy throughdif-
ferential privacy or corpus transformation.
DPVoteRAG(Koga et al., 2024) spends its privacy
budget selectively on tokens that require sensitive
retrieved knowledge.RAG with Differential Pri-
vacy(Grislain, 2025),LPRAG(He et al., 2025a),
VAGUE-Gate(Hemmat et al., 2025),PAD(Wang
et al., 2025b), andInvisibleInk(Vinod et al., 2025)
explore related mechanisms for privacy-preserving
generation over sensitive context, ranging from
token- or entity-level perturbation to decoding-time
protection for long-form generation.SAGE(Zeng
et al., 2025a) takes a different route by replacing
private retrieval corpora entirely with high-utility
synthetic alternatives.
A third line secures theretrieval backend.Re-
moteRAG(Cheng et al., 2025b) formalizes privacy-
preserving cloud RAG, protecting query privacy
with efficient distance-based perturbation. What’s
more,ppRAG(Ye et al., 2025a) supports retrieval
over outsourced encrypted databases through
distance-preserving encryption, whileFRAG(Zhao,
2024) extends encrypted nearest-neighbor retrieval
to federated vector databases across mutually dis-
trusted parties.
10

A fourth line buildsconfidential or cryp-
tographically protected RAG architectures.
FedE4RAG(Mao et al., 2025) trains retrievers
collaboratively in a federated manner without
centralizing raw data, andC-FedRAG(Addison
et al., 2024) leverages confidential computing for
secure cross-party RAG execution.Privacy-Aware
RAG(Zhou et al., 2025b) encrypts both textual
content and embeddings before storage, an
approach thatSAG(Zhou et al., 2025c) further
strengthens with formal security proofs. Moreover,
as an integrative framework,SecureRAG(Bassit
and Boddeti, 2025) separates secure search
from secure document fetching, combining fully
homomorphic encryption for search execution
with attribute-based encryption for fine-grained
document access.
Overall, this layer moves secure RAG from con-
tent filtering to system-level protection. It serves as
the most direct defense counterpart to the reverse-
direction threat surface in Section 3.4, where the
attacker seeks to infer, extract, or over-access ex-
ternal knowledge. This layer is especially impor-
tant for real-world deployments, but it is also the
most expensive to implement, since strong confi-
dentiality typically incurs substantial architectural
redesign or cryptographic overhead.
4.5 Summary and Observations
Overall, the current defense literature is unevenly
distributed across the RAG knowledge-access
pipeline. The most mature lines of work concen-
trate around the second and third trust boundaries.
These primarily involve retrieval-time hardening
before evidence becomes model-visible context,
and post-retrieval isolation after that boundary is
crossed. By contrast, the first upstream bound-
ary remains much less protected. Knowledge-base
integrity, provenance, and post-compromise reme-
diation are still relatively sparse despite their crit-
ical importance for long-lived shared corpora. In
parallel, access control, privacy, and confidential-
ity have grown rapidly as a system-level line of
defense, particularly through differential privacy,
encrypted retrieval, and confidential architectures.
By mapping the attack taxonomy in Figure 3
to the defense mechanisms in Figure 4, we can
observe several structural patterns. First, current
defenses remain predominantly reactive. Many
studies attempt to filter, rerank, or contain harm-
ful evidence only after it has already entered the
retrieval flow, while significantly fewer methodsgovern admission, rollback, or corpus recovery at
the substrate level. Second, there is a structural
mismatch between threat detection mechanisms
and advanced attack construction. Although many
retrieval-time defenses implicitly assume that poi-
soned evidence will appear as a semantic outlier,
modern attacks increasingly optimize for retriev-
ability, fluency, and contextual coherence to seam-
lessly blend with benign evidence. Third, privacy
and confidentiality mechanisms are complementary
rather than substitutive. Although they are essential
for preventing unauthorized disclosure and extrac-
tion, they cannot independently resolve integrity
corruption or retrieved-context exploitation.
Taken together, these observations suggest that
secure RAG should be viewed less as an isolated
filtering task and more as a layered control problem
across multiple trust boundaries. A robust deploy-
ment must synergistically combine upstream gover-
nance of the knowledge substrate, retrieval-time ev-
idence hardening, post-retrieval containment, and
confidentiality controls, rather than relying on any
single defense family in isolation.
5Benchmarks and Evaluation Studies for
Secure RAG
Under the operational boundary in Section 2.2, we
distinguish between two kinds of evaluation litera-
ture.Benchmark studiesprovide reusable datasets,
protocols, or harnesses for secure-RAG testing.
Systematic evaluation studiesprovide broader em-
pirical analyses that clarify how secure-RAG fail-
ures behave, even when benchmark construction
is not the main contribution. In this section, we
review the literature in these two groups and then
summarize the main observations.
5.1 Benchmark Studies
The current benchmark literature can be roughly
divided into two groups. One group evaluates how
malicious content or adversarial instructions move
through the retrieval and generation pipeline. The
other group evaluates privacy, extraction, and dis-
closure risks, often together with the trade-off be-
tween privacy protection and task utility.
Manipulation-oriented benchmarks.
Rag and Roll(De Stefano et al., 2024) is an
early end-to-end evaluation framework for indirect
prompt manipulation in LLM application pipelines
with RAG components. Its main contribution is
to evaluate attacks under realistic framework-level
11

configurations rather than only under isolated re-
trieval settings. Meanwhile,SafeRAG(Liang et al.,
2025b) provides a dedicated security benchmark
for RAG with multiple attack tasks, task-aware
metrics, and evaluations across representative RAG
components. It is important because it turns secure-
RAG evaluation into a reusable benchmark setting
rather than a collection of one-off attack demon-
strations.Benchmarking Poisoning Attacks against
Retrieval-Augmented Generation(Zhang et al.,
2025c) broadens this direction by comparing a wide
range of poisoning attacks and defenses across
datasets and RAG variants under one framework.
OpenRAG-Soc(Guo and Wei, 2026) focuses on
web-facing RAG over social-web content and em-
phasizes realistic end-to-end evaluation of indirect
prompt injection and retrieval poisoning together
with practical mitigations. Moreover,MPIB(Lee
et al., 2026) brings prompt-injection evaluation into
the medical domain and is notable for measuring
clinically grounded harm rather than relying only
on attack success.
Privacy, extraction, and disclosure-oriented
benchmarks.S-RAG(Zeng et al., 2025b) frames
privacy evaluation as black-box auditing of whether
personal textual data has been used in a RAG sys-
tem.SMA(Sun et al., 2025) extends this line to-
ward source-aware membership auditing in semi-
black-box settings and further considers multi-
modal retrieval. In parallel,Privacy Protection in
RAG(Zhang et al., 2026) combines a fine-grained
privacy protection design with an explicit evalu-
ation framework for studying the privacy-utility
balance beyond coarse document-level removal.
KE-Bench(Qi et al., 2026) standardizes the evalua-
tion of knowledge-extraction attacks and defenses
across retrievers, generators, and datasets. Sim-
ilarly,MedPriv-Bench(Guan et al., 2026) intro-
duces a medical benchmark that jointly evaluates
contextual leakage and clinical utility. Finally,
SEAL-Tag(Xie et al., 2026) contributes a structured
protocol for adaptive leakage auditing together with
utility and latency evaluation for PII-safe RAG. Al-
though Zhang et al. (2026) and Xie et al. (2026) are
not benchmark-only papers, we include them here
because they contribute reusable security evalua-
tion settings or protocols rather than only reporting
a single system result.
Taken together, these benchmark studies show a
clear trend toward more realistic evaluation. Recent
work increasingly evaluates complete pipelines,domain-specific risk, adaptive attackers, and ex-
plicit utility-security trade-offs, rather than report-
ing attack success on a single simplified setup.
5.2 Systematic Evaluation Studies
The Good and the Bad(Zeng et al., 2024) is an
early broad empirical study of privacy in text RAG.
It is important not because it releases a benchmark
in the narrow sense, but because it clarifies a cen-
tral tension in secure RAG. RAG can create new
leakage channels for the retrieval database, while
at the same time reducing some privacy risks tied
to purely parametric generation.
Beyond Text(Zhang et al., 2025d) provides the
first systematic privacy analysis of multimodal
RAG across vision-language and speech-language
settings. Its contribution is to show that multimodal
carriers create additional leakage paths and that pri-
vacy analysis in text-only RAG does not directly
transfer to multimodal settings.
A Systemic Evaluation of Multimodal RAG Pri-
vacy(Al-Lawati and Wang, 2026) complements
this direction with a focused empirical study of
multimodal privacy leakage, especially member-
ship and caption leakage under visual retrieval set-
tings. Compared with benchmark-oriented work,
these systematic studies are less about packaging
a reusable suite and more about clarifying what
should be measured, where leakage appears, and
how the threat changes across modalities and sys-
tem assumptions.
5.3 Summary and Observations
Overall, recent benchmarks and evaluation stud-
ies have made secure-RAG assessment more sys-
tematic and more deployment-relevant. Compared
with isolated attack demonstrations, they provide
clearer protocols, broader component coverage,
and more realistic end-to-end settings for study-
ing how security failures appear in practice.
At the same time, the current evaluation liter-
ature still reflects the structure of the attack and
defense landscape discussed in the previous two
sections. Different studies often focus on different
threat surfaces, such as poisoning, prompt injection,
privacy leakage, or extraction, and many reported
metrics still concentrate on final outputs rather than
on intermediate pipeline behavior. As a result, cur-
rent evaluation is increasingly useful for comparing
methods within a threat setting, while cross-surface
comparison remains less unified.
12

Taken together, these studies show that secure-
RAG evaluation is becoming a central part of the
field rather than a secondary afterthought. It pro-
vides an important bridge between attack analysis
and defense design by making pipeline-level threats
more visible and comparable across systems.
6 Future Directions
Building upon our preceding analysis and discus-
sion, we highlight four future directions valuable
for advancing the field of secure RAG.
A first important direction is to move secure
RAG beyond inference-time filtering toward gov-
ernance and recoverability of the knowledge sub-
strate. Current defenses still focus mainly on fil-
tering, reranking, and containment at or after re-
trieval time, as discussed in Section 4. However,
the primary threat is that once malicious content
enters a shared knowledge substrate, it can persist,
be triggered repeatedly, and affect multiple users
over time (Zou et al., 2025). Recent benchmark
work further shows the importance of evaluating
poisoning under broader settings, andRAGForen-
sicsprovides an encouraging step toward practical
traceback and post-incident analysis (Zhang et al.,
2025c,b). Future work should therefore pay more
attention to admission control, provenance tracking,
versioned corpora, traceback, rollback, and corpus
repair. The value of this direction is to make secure
RAG not only resistant at answer time, but also
governable and recoverable after compromise.
A second direction is to set clearer boundaries
for how retrieved content is used, rather than simply
concatenating all external content into the model-
visible context (Ramakrishna et al., 2024; Chang
et al., 2026; Guo and Wei, 2026; Masoud et al.,
2026; Wen et al., 2025b). This concatenation-based
approach is natural in standard question answer-
ing, but from a security perspective, it dangerously
mixes factual evidence with executable instructions.
In recent work, instruction detection methods at-
tempt to isolate hidden instructions as distinct ob-
jects (Wen et al., 2025b), while selective disclo-
sure frameworks move security enforcement before
generation (Masoud et al., 2026). Future work is
encouraged to continue in this direction by sepa-
rating evidence from control more explicitly, such
as through typed evidence, policy-aware context
assembly, or bounded interaction between retrieved
documents and the generator. This would reduce
the chance that untrusted content is upgraded fromsupporting evidence into behavioral control.
A third direction is to move evaluation be-
yond output-only testing toward boundary-local,
remediation-aware, and cross-surface evaluation.
Recent studies have taken meaningful steps to make
secure-RAG evaluation more realistic by broad-
ening security-oriented benchmarking, evaluating
attacks and defenses across diverse architectures,
and moving privacy evaluation toward more uni-
fied protocols (Liang et al., 2025b; Zhang et al.,
2025c; Guo and Wei, 2026; Qi et al., 2026). Mov-
ing forward, future work should delve deeper by
measuring exactly where a failure occurs, how it
propagates across trust boundaries, and how effec-
tively the system recovers after mitigation. Ro-
bust and comprehensive evaluation frameworks are
essential for driving innovative thinking in both
offensive and defensive strategies.
A fourth direction is to push secure RAG toward
richer deployment settings that better match real ap-
plications, including web-native, multimodal, and
agent-coupled systems. Recent work has already
started to evaluate web-facing RAG, multimodal
privacy risks, multi-turn interactions, and agent-
related attack settings, revealing that each setting
introduces novel threat dimensions not adequately
captured by single-turn text evaluations (Guo and
Wei, 2026; Zhang et al., 2025d; Al-Lawati and
Wang, 2026; Katsis et al., 2025; Chang et al., 2026;
Zhang et al., 2025c). Future benchmarks in these
settings should remain end-to-end, cover realistic
carriers and interaction patterns, and jointly eval-
uate security, utility, and recovery. On the secu-
rity side, richer settings also call for further study
of cross-modal leakage, long-horizon interactions,
and action-coupled failures. This would help de-
fenses evolve in tandem with the actual forms of
RAG that are currently moving into deployment.
7 Conclusion
Motivated by the fact that external knowledge fun-
damentally alters the inference path, we proposed
that secure RAG must be independently analyzed
as the security of the knowledge-access pipeline,
which spans data ingestion, retrieval, assembly, and
disclosure. Under this perspective, we explicitly
distinguished inherent LLM vulnerabilities from
RAG-introduced or RAG-amplified risks. Through
this operational boundary, we systematically struc-
tured the literature into four distinct security sur-
faces and three trust boundaries, detailing the me-
13

chanics of current attacks, defense layers, and eval-
uation protocols.
Our review reveals a structural mismatch across
the current landscape of secure RAG. While at-
tack strategies are rapidly evolving to generate
highly coherent and retrievable payloads, most de-
fenses remain strictly reactive, concentrating on
mid-stream filtering rather than upstream corpus
integrity or data provenance. Furthermore, em-
pirical evaluations remain fragmented across iso-
lated threat models. Moving forward, future re-
search should transition from localized patches
to layered, boundary-aware governance. This re-
quires developing unified, cross-surface bench-
marks, strengthening pre-retrieval data validation,
and ensuring that defense designs incorporate prac-
tical rollback and remediation capabilities against
adaptive threats.
References
Parker Addison, Minh-Tuan H Nguyen, Tomislav
Medan, Jinali Shah, Mohammad T Manzari, Bren-
dan McElrone, Laksh Lalwani, Aboli More, Smita
Sharma, Holger R Roth, and 1 others. 2024. C-
fedrag: A confidential federated retrieval-augmented
generation system.arXiv preprint arXiv:2412.13163.
Ali Al-Lawati and Suhang Wang. 2026. A systemic
evaluation of multimodal rag privacy.arXiv preprint
arXiv:2601.17644.
Lukas Ammann, Sara Ott, Christoph R Landolt, and
Marco P Lehmann. 2025. Securing rag: A risk as-
sessment and mitigation framework. In2025 IEEE
Swiss Conference on Data Science (SDS), pages 127–
134. IEEE.
Maya Anderson, Guy Amit, and Abigail Goldsteen.
2025. Is my data in your retrieval database? mem-
bership inference attacks against retrieval augmented
generation. InInternational Conference on Informa-
tion Systems Security and Privacy, volume 2, pages
474–485. Science and Technology Publications, Lda.
Atousa Arzanipour, Rouzbeh Behnia, Reza Ebrahimi,
and Kaushik Dutta. 2025. Rag security and pri-
vacy: Formalizing the threat model and attack surface.
arXiv preprint arXiv:2509.20324.
Amina Bassit and Vishnu Boddeti. 2025. SecureRAG:
End-to-end secure retrieval-augmented generation.
InThe Second Workshop on GenAI for Health: Po-
tential, Trust, and Policy Compliance.
Andreea-Elena Bodea, Stephen Meisenbacher, Alexan-
dra Klymenko, and Florian Matthes. 2026. Sok: Pri-
vacy risks and mitigations in retrieval-augmented gen-
eration systems.arXiv preprint arXiv:2601.03979.Nicholas Carlini, Florian Tramer, Eric Wallace,
Matthew Jagielski, Ariel Herbert-V oss, Katherine
Lee, Adam Roberts, Tom Brown, Dawn Song, Ulfar
Erlingsson, and 1 others. 2021. Extracting training
data from large language models. In30th USENIX
security symposium (USENIX Security 21), pages
2633–2650.
Alberto Castagnaro, Umberto Salviati, Mauro Conti,
Luca Pajola, and Simeone Pizzi. 2025. The hidden
threat in plain text: Attacking rag data loaders. In
Proceedings of the 18th ACM Workshop on Artificial
Intelligence and Security, pages 170–181.
Hongyan Chang, Ergute Bao, Xinjian Luo, and Ting
Yu. 2026. Overcoming the retrieval barrier: Indirect
prompt injection in the wild for llm systems.arXiv
preprint arXiv:2601.07072.
Zhiyuan Chang, Mingyang Li, Xiaojun Jia, Junjie Wang,
Yuekai Huang, Ziyou Jiang, Yang Liu, and Qing
Wang. 2025. One shot dominance: Knowledge poi-
soning attack on retrieval-augmented generation sys-
tems. InFindings of the Association for Compu-
tational Linguistics: EMNLP 2025, pages 18811–
18825, Suzhou, China. Association for Computa-
tional Linguistics.
Harsh Chaudhari, Giorgio Severi, John Abascal,
Matthew Jagielski, Christopher A Choquette-Choo,
Milad Nasr, Cristina Nita-Rotaru, and Alina Oprea.
2024. Phantom: General trigger attacks on retrieval
augmented language generation.
Bingxiang Chen, John Tackman, Manu Setälä, Timo Po-
ranen, and Zheying Zhang. 2025. Integrating access
control with retrieval-augmented generation: A proof
of concept for managing sensitive patient profiles. In
Proceedings of the 40th ACM/SIGAPP Symposium
on Applied Computing, SAC ’25, page 915–919, New
York, NY , USA. Association for Computing Machin-
ery.
Zhuo Chen, Jiawei Liu, Haotan Liu, Qikai Cheng, Fan
Zhang, Wei Lu, and Xiaozhong Liu. 2024. Black-box
opinion manipulation attacks to retrieval-augmented
generation of large language models.arXiv preprint
arXiv:2407.13757.
Mingyue Cheng, Yucong Luo, Jie Ouyang, Qi Liu,
Huijie Liu, Li Li, Shuo Yu, Bohou Zhang, Jiawei
Cao, Jie Ma, and 1 others. 2025a. A survey on
knowledge-oriented retrieval-augmented generation.
arXiv preprint arXiv:2503.10677.
Pengzhou Cheng, Yidong Ding, Tianjie Ju, Zongru Wu,
Wei Du, Ping Yi, Zhuosheng Zhang, and Gongshen
Liu. 2024. Trojanrag: Retrieval-augmented genera-
tion can be backdoor driver in large language models.
arXiv preprint arXiv:2405.13401.
Yihang Cheng, Lan Zhang, Junyang Wang, Mu Yuan,
and Yunhao Yao. 2025b. Remoterag: A privacy-
preserving llm cloud rag service. InFindings of
the Association for Computational Linguistics: ACL
2025, pages 3820–3837.
14

Zirui Cheng, Jikai Sun, Anjun Gao, Yueyang
Quan, Zhuqing Liu, Xiaohua Hu, and Minghong
Fang. 2025c. Secure retrieval-augmented gener-
ation against poisoning attacks.arXiv preprint
arXiv:2510.25025.
Sukmin Cho, Soyeong Jeong, Jeongyeon Seo, Taeho
Hwang, and Jong C Park. 2024. Typos that broke the
rag’s back: Genetic attack on rag pipeline by simulat-
ing documents in the wild via low-level perturbations.
InFindings of the Association for Computational Lin-
guistics: EMNLP 2024, pages 2826–2844.
Chanwoo Choi, Jinsoo Kim, Sukmin Cho, Soyeong
Jeong, and Buru Chang. 2025. The RAG paradox: A
black-box attack exploiting unintentional vulnerabil-
ities in retrieval-augmented generation systems. In
Findings of the Association for Computational Lin-
guistics: EMNLP 2025, pages 23723–23744, Suzhou,
China. Association for Computational Linguistics.
Sarthak Choudhary, Nils Palumbo, Ashish Hooda, Kr-
ishnamurthy Dj Dvijotham, and Somesh Jha. 2026.
Through the stealth lens: Attention-aware defenses
against poisoning in RAG.
Cody Clop and Yannick Teglia. 2024. Backdoored re-
trievers for prompt injection attacks on retrieval aug-
mented generation of large language models.arXiv
preprint arXiv:2410.14479.
Badhan Chandra Das, M Hadi Amini, and Yanzhao Wu.
2025. Security and privacy challenges of large lan-
guage models: A survey.ACM Computing Surveys,
57(6):1–39.
Gianluca De Stefano, Lea Schönherr, and Giancarlo
Pellegrino. 2024. Rag and roll: An end-to-end
evaluation of indirect prompt manipulations in llm-
based application frameworks.arXiv preprint
arXiv:2408.05025.
Sagie Dekel, Moshe Tennenholtz, and Oren Kurland.
2026. Addressing corpus knowledge poisoning at-
tacks on rag using sparse attention.arXiv preprint
arXiv:2602.04711.
Gelei Deng, Yi Liu, Kailong Wang, Yuekang Li, Tian-
wei Zhang, and Yang Liu. 2024. Pandora: Jailbreak
gpts by retrieval augmented generation poisoning.
arXiv preprint arXiv:2402.08416.
Tessa E_Andersen, Ayanna Marie Avalos, Gaby
G_Dagher, and Min Long. 2025. D-rag: A privacy-
preserving framework for decentralized rag using
blockchain.
Kennedy Edemacu, Vinay M Shashidhar, Micheal Tu-
ape, Dan Abudu, Beakcheol Jang, and Jong Wook
Kim. 2025. Defending against knowledge poisoning
attacks during retrieval-augmented generation.arXiv
preprint arXiv:2508.02835.
Robert Friel, Masha Belyi, and Atindriyo Sanyal. 2024.
Ragbench: Explainable benchmark for retrieval-
augmented generation systems.arXiv preprint
arXiv:2407.11005.Aoran Gan, Hao Yu, Kai Zhang, Qi Liu, Wenyu Yan,
Zhenya Huang, Shiwei Tong, and Guoping Hu. 2025.
Retrieval augmented generation evaluation in the era
of large language models: A comprehensive survey.
arXiv preprint arXiv:2504.14891.
Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang Jia,
Jinliu Pan, Yuxi Bi, Yixin Dai, Jiawei Sun, Haofen
Wang, Haofen Wang, and 1 others. 2023. Retrieval-
augmented generation for large language models: A
survey.arXiv preprint arXiv:2312.10997, 2(1):32.
Runpeng Geng, Yanting Wang, Ying Chen, and Jinyuan
Jia. 2025. Unic-rag: Universal knowledge corrup-
tion attacks to retrieval-augmented generation.arXiv
preprint arXiv:2508.18652.
Yuyang Gong, Zhuo Chen, Jiawei Liu, Miaokun
Chen, Fengchang Yu, Wei Lu, XiaoFeng Wang, and
Xiaozhong Liu. 2025. {Topic-FlipRAG }:{Topic-
Orientated }adversarial opinion manipulation attacks
to{Retrieval-Augmented }generation models. In
34th USENIX Security Symposium (USENIX Security
25), pages 3807–3826.
Nicolas Grislain. 2025. Rag with differential privacy.
In2025 IEEE Conference on Artificial Intelligence
(CAI), pages 847–852. IEEE.
Shaowei Guan, Yu Zhai, Hin Chi Kwok, Jiawei Du,
Xinyu Feng, Jing Li, Harry Qin, and Vivian Hui.
2026. Medpriv-bench: Benchmarking the privacy-
utility trade-off of large language models in med-
ical open-end question answering.arXiv preprint
arXiv:2603.14265.
Haoze Guo and Ziqi Wei. 2026. Hidden-in-plain-text: A
benchmark for social-web indirect prompt injection
in rag.arXiv preprint arXiv:2601.10923.
Shailja Gupta, Rajesh Ranjan, and Surya Narayan
Singh. 2024. A comprehensive survey of retrieval-
augmented generation (rag): Evolution, current
landscape and future directions.arXiv preprint
arXiv:2410.12837.
Hyeonjeong Ha, Qiusi Zhan, Jeonghwan Kim, Dim-
itrios Bralios, Saikrishna Sanniboina, Nanyun Peng,
Kai-Wei Chang, Daniel Kang, and Heng Ji. 2025.
Mm-poisonrag: Disrupting multimodal rag with lo-
cal and global poisoning attacks.arXiv preprint
arXiv:2502.17832.
Longzhu He, Peng Tang, Yuanhe Zhang, Pengpeng
Zhou, and Sen Su. 2025a. Mitigating privacy risks
in retrieval-augmented generation via locally private
entity perturbation.Information Processing & Man-
agement, 62(4):104150.
Yu He, Yifei Chen, Yiming Li, Shuo Shao, Leyi Qi,
Boheng Li, Dacheng Tao, and Zhan Qin. 2025b.
External data extraction attacks against retrieval-
augmented large language models.arXiv preprint
arXiv:2510.02964.
15

Arshia Hemmat, Matin Moqadas, Ali Mamanpoosh,
Amirmasoud Rismanchian, and Afsaneh Fatemi.
2025. Vague-gate: Plug-and-play local-privacy
shield for retrieval-augmented generation. InPro-
ceedings of the 14th International Joint Conference
on Natural Language Processing and the 4th Confer-
ence of the Asia-Pacific Chapter of the Association
for Computational Linguistics, pages 3715–3730.
Giwon Hong, Jeonghwan Kim, Junmo Kang, Sung-
Hyon Myaeng, and Joyce Jiyoung Whang. 2024.
Why so gullible? enhancing the robustness of
retrieval-augmented models against counterfactual
noise. InFindings of the Association for Computa-
tional Linguistics: NAACL 2024, pages 2474–2495.
Zhibo Hu, Chen Wang, Yanfeng Shu, Hye-Young Paik,
and Liming Zhu. 2024. Prompt perturbation in
retrieval-augmented generation based large language
models. InProceedings of the 30th ACM SIGKDD
Conference on Knowledge Discovery and Data Min-
ing, pages 1119–1130.
Che Jiang, Biqing Qi, Xiangyu Hong, Dayuan Fu, Yang
Cheng, Fandong Meng, Mo Yu, Bowen Zhou, and
Jie Zhou. 2024. On large language models’ halluci-
nation with regard to known facts. InProceedings
of the 2024 Conference of the North American Chap-
ter of the Association for Computational Linguistics:
Human Language Technologies (Volume 1: Long Pa-
pers), pages 1041–1053.
Yang Jiao, Xiaodong Wang, and Kai Yang. 2025. Pr-
attack: Coordinated prompt-rag attacks on retrieval-
augmented generation in large language models via
bilevel optimization. InProceedings of the 48th In-
ternational ACM SIGIR Conference on Research and
Development in Information Retrieval, pages 656–
667.
Yannis Katsis, Sara Rosenthal, Kshitij Fadnis, Chu-
laka Gunasekara, Young-Suk Lee, Lucian Popa, Vraj
Shah, Huaiyu Zhu, Danish Contractor, and Marina
Danilevsky. 2025. Mtrag: A multi-turn conversa-
tional benchmark for evaluating retrieval-augmented
generation systems.Transactions of the Association
for Computational Linguistics, 13:784–808.
Juhee Kim, Xiaoyuan Liu, Zhun Wang, Shi Qiu, Bo Li,
Wenbo Guo, and Dawn Song. 2026. The attack and
defense landscape of agentic ai: A comprehensive
survey.arXiv preprint arXiv:2603.11088.
Minseok Kim, Hankook Lee, and Hyungjoon Koo. 2025.
Rescuing the unpoisoned: Efficient defense against
knowledge corruption attacks on rag systems.arXiv
preprint arXiv:2511.01268.
Tatsuki Koga, Ruihan Wu, Zhiyuan Zhang, and
Kamalika Chaudhuri. 2024. Privacy-preserving
retrieval-augmented generation with differential pri-
vacy.arXiv preprint arXiv:2412.04697.
Junhyeok Lee, Han Jang, and Kyu Sung Choi. 2026.
Mpib: A benchmark for medical prompt injection
attacks and clinical safety in llms.arXiv preprint
arXiv:2602.06268.Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio
Petroni, Vladimir Karpukhin, Naman Goyal, Hein-
rich Küttler, Mike Lewis, Wen-tau Yih, Tim Rock-
täschel, and 1 others. 2020. Retrieval-augmented gen-
eration for knowledge-intensive nlp tasks.Advances
in neural information processing systems, 33:9459–
9474.
Qinbin Li, Junyuan Hong, Chulin Xie, Jeffrey Tan,
Rachel Xin, Junyi Hou, Xavier Yin, Zhun Wang,
Dan Hendrycks, Zhangyang Wang, and 1 others.
2024. Llm-pbe: Assessing data privacy in large lan-
guage models.Proceedings of the VLDB Endowment,
17(11):3201–3214.
Yuying Li, Gaoyang Liu, Chen Wang, and Yang Yang.
2025. Generating is believing: Membership infer-
ence attacks against retrieval-augmented generation.
InICASSP 2025-2025 IEEE International Confer-
ence on Acoustics, Speech and Signal Processing
(ICASSP), pages 1–5. IEEE.
Jiacheng Liang, Yuhui Wang, Changjiang Li, Rongyi
Zhu, Tanqiu Jiang, Neil Gong, and Ting Wang.
2025a. Graphrag under fire.arXiv preprint
arXiv:2501.14050.
Xun Liang, Simin Niu, Zhiyu Li, Sensen Zhang, Hanyu
Wang, Feiyu Xiong, Zhaoxin Fan, Bo Tang, Jihao
Zhao, Jiawei Yang, and 1 others. 2025b. Saferag:
benchmarking security in retrieval-augmented gener-
ation of large language model. InProceedings of the
63rd Annual Meeting of the Association for Compu-
tational Linguistics (Volume 1: Long Papers), pages
4609–4631.
Bo Lin, Shangwen Wang, Liqian Chen, and Xiaoguang
Mao. 2025. Exploring the security threats of knowl-
edge base poisoning in retrieval-augmented code gen-
eration.arXiv preprint arXiv:2502.03233.
Yuping Lin, Pengfei He, Han Xu, Yue Xing, Makoto
Yamada, Hui Liu, and Jiliang Tang. 2024. Towards
understanding jailbreak attacks in llms: A represen-
tation space analysis. InProceedings of the 2024
Conference on Empirical Methods in Natural Lan-
guage Processing, pages 7067–7085.
Mingrui Liu, Sixiao Zhang, and Cheng Long. 2025a.
Mask-based membership inference attacks for
retrieval-augmented generation. InProceedings of
the ACM on Web Conference 2025, pages 2894–2907.
Yinuo Liu, Zenghui Yuan, Guiyao Tie, Jiawen Shi,
Pan Zhou, Lichao Sun, and Neil Zhenqiang Gong.
2025b. Poisoned-mrag: Knowledge poisoning at-
tacks to multimodal retrieval augmented generation.
arXiv preprint arXiv:2503.06254.
Qianren Mao, Qili Zhang, Hanwen Hao, Zhentao Han,
Runhua Xu, Weifeng Jiang, Qi Hu, Zhijun Chen,
Tyler Zhou, Bo Li, and 1 others. 2025. Privacy-
preserving federated embedding learning for local-
ized retrieval-augmented generation.arXiv preprint
arXiv:2504.19101.
16

Aiman Al Masoud, Marco Arazzi, and Antonino Nocera.
2026. Sd-rag: A prompt-injection-resilient frame-
work for selective disclosure in retrieval-augmented
generation.arXiv preprint arXiv:2601.11199.
Ali Naseh, Yuefeng Peng, Anshuman Suri, Harsh
Chaudhari, Alina Oprea, and Amir Houmansadr.
2025. Riddle me this! stealthy membership inference
for retrieval-augmented generation. InProceedings
of the 2025 ACM SIGSAC Conference on Computer
and Communications Security, pages 1245–1259.
Bo Ni, Zheyuan Liu, Leyao Wang, Yongjia Lei, Yuy-
ing Zhao, Xueqi Cheng, Qingkai Zeng, Luna Dong,
Yinglong Xia, Krishnaram Kenthapadi, and 1 others.
2025. Towards trustworthy retrieval augmented gen-
eration for large language models: A survey.arXiv
preprint arXiv:2502.06872.
Chanhee Park, Hyeonseok Moon, Chanjun Park, and
Heuiseok Lim. 2025. MIRAGE: A metric-intensive
benchmark for retrieval-augmented generation eval-
uation. InFindings of the Association for Computa-
tional Linguistics: NAACL 2025, pages 2883–2900,
Albuquerque, New Mexico. Association for Compu-
tational Linguistics.
Pankayaraj Pathmanathan, Michael-Andrei Panaitescu-
Liess, Cho-Yu Jason Chiang, and Furong Huang.
2025. Ragpart & ragmask: Retrieval-stage defenses
against corpus poisoning in retrieval-augmented gen-
eration.arXiv preprint arXiv:2512.24268.
Xiangyu Peng, Prafulla Kumar Choubey, Caiming
Xiong, and Chien-Sheng Wu. 2025. Unanswerabil-
ity evaluation for retrieval augmented generation. In
Proceedings of the 63rd Annual Meeting of the As-
sociation for Computational Linguistics (Volume 1:
Long Papers), pages 8452–8472.
Yuefeng Peng, Junda Wang, Hong Yu, and Amir
Houmansadr. 2024. Data extraction attacks in
retrieval-augmented generation via backdoors.arXiv
preprint arXiv:2411.01705.
Zhenting Qi, Hanlin Zhang, Eric Xing, Sham Kakade,
and Himabindu Lakkaraju. 2025. Follow my instruc-
tion and spill the beans: Scalable data extraction from
retrieval-augmented generation systems. InThe Thir-
teenth International Conference on Learning Repre-
sentations (ICLR 2025).
Zhisheng Qi, Utkarsh Sahu, Li Ma, Haoyu Han,
Ryan Rossi, Franck Dernoncourt, Mahantesh Ha-
lappanavar, Nesreen Ahmed, Yushun Dong, Yue
Zhao, and 1 others. 2026. Benchmarking knowledge-
extraction attack and defense on retrieval-augmented
generation.arXiv preprint arXiv:2602.09319.
Anil Ramakrishna, Jimit Majmudar, Rahul Gupta, and
Devamanyu Hazarika. 2024. Llm-pieval: A bench-
mark for indirect prompt injection attacks in large
language models.
Avital Shafran, Roei Schuster, and Vitaly Shmatikov.
2025. Machine against the {RAG }: Jamming{Retrieval-Augmented }generation with blocker doc-
uments. In34th USENIX Security Symposium
(USENIX Security 25), pages 3787–3806.
Chaitanya Sharma. 2025. Retrieval-augmented gener-
ation: A comprehensive survey of architectures, en-
hancements, and robustness frontiers.arXiv preprint
arXiv:2506.00054.
Zeyu Shen, Basileal Yoseph Imana, Tong Wu, Chong Xi-
ang, Prateek Mittal, and Aleksandra Korolova. 2025.
ReliabilityRAG: Effective and provably robust de-
fense for RAG-based web-search. InThe Thirty-
ninth Annual Conference on Neural Information Pro-
cessing Systems.
Ezzeldin Shereen, Dan Ristea, Shae McFadden, Burak
Hasircioglu, Vasilios Mavroudis, and Chris Hicks.
2026. One pic is all it takes: Poisoning visual doc-
ument retrieval augmented generation with a single
image.Transactions on Machine Learning Research.
Xiaonan Si, Meilin Zhu, Simeng Qin, Lijia Yu, Li-
jun Zhang, Shuaitong Liu, Xinfeng Li, Ranjie
Duan, Yang Liu, and Xiaojun Jia. 2025. Secon-
rag: A two-stage semantic filtering and conflict-
free framework for trustworthy rag.arXiv preprint
arXiv:2510.09710.
Hongru Song, Yu-An Liu, Ruqing Zhang, Jiafeng Guo,
Jianming Lv, Maarten de Rijke, and Xueqi Cheng.
2025. The silent saboteur: Imperceptible adversarial
attacks against black-box retrieval-augmented gen-
eration systems. InFindings of the Association for
Computational Linguistics: ACL 2025, pages 13935–
13952.
Vasilije Stambolic, Aritra Dhar, and Lukas Cav-
igelli. 2025. Rag-pull: Imperceptible attacks on
rag systems for code generation.arXiv preprint
arXiv:2510.11195.
Jan Strich, Enes Kutay Isgorur, Maximilian Trescher,
Chris Biemann, and Martin Semmann. 2026. T2-
ragbench: Text-and-table aware retrieval-augmented
generation. InProceedings of the 19th Conference of
the European Chapter of the Association for Compu-
tational Linguistics (Volume 1: Long Papers), pages
165–191.
Shixuan Sun, Siyuan Liang, Ruoyu Chen, Jianjie
Huang, Jingzhi Li, and Xiaochun Cao. 2025. Sma:
Who said that? auditing membership leakage in
semi-black-box rag controlling.arXiv preprint
arXiv:2508.09105.
Pan Suo, Yu-Ming Shang, San-Chuan Guo, and
Xi Zhang. 2025. Hoist with his own petard: Induc-
ing guardrails to facilitate denial-of-service attacks
on retrieval-augmented generation of llms.arXiv
preprint arXiv:2504.21680.
Xue Tan, Hao Luan, Mingyu Luo, Xiaoyan Sun, Ping
Chen, and Jun Dai. 2025. RevPRAG: Revealing
poisoning attacks in retrieval-augmented generation
through LLM activation analysis. InFindings of the
17

Association for Computational Linguistics: EMNLP
2025, pages 12999–13011, Suzhou, China. Associa-
tion for Computational Linguistics.
Scott Thornton. 2026. Retrieval pivot attacks in hy-
brid rag: Measuring and mitigating amplified leakage
from vector seeds to graph expansion.arXiv preprint
arXiv:2602.08668.
Yiteng Tu, Weihang Su, Yujia Zhou, Yiqun Liu, and
Qingyao Ai. 2025. Rbft: Robust fine-tuning for
retrieval-augmented generation against retrieval de-
fects.arXiv preprint arXiv:2501.18365.
Vishnu Vinod, Krishna Pillutla, and Abhradeep Guha
Thakurta. 2025. Invisibleink: High-utility and low-
cost text generation with differential privacy.arXiv
preprint arXiv:2507.02974.
Fei Wang, Xingchen Wan, Ruoxi Sun, Jiefeng Chen,
and Sercan O Arik. 2025a. Astute rag: Overcom-
ing imperfect retrieval augmentation and knowledge
conflicts for large language models. InProceedings
of the 63rd Annual Meeting of the Association for
Computational Linguistics (Volume 1: Long Papers),
pages 30553–30571.
Haoran Wang, Xiongxiao Xu, Baixiang Huang, and
Kai Shu. 2025b. Privacy-aware decoding: Miti-
gating privacy leakage of large language models
in retrieval-augmented generation.arXiv preprint
arXiv:2508.03098.
Haozhen Wang, Haoyue Liu, Jionghao Zhu, Zhichao
Wang, Yongxin Guo, and Xiaoying Tang. 2026. Pidp-
attack: Combining prompt injection with database
poisoning attacks on retrieval-augmented generation
systems.arXiv preprint arXiv:2603.25164.
Yuhao Wang, Wenjie Qu, Yanze Jiang, Lichen Liu, Yue
Liu, Shengfang Zhai, Yinpeng Dong, and Jiaheng
Zhang. 2025c. Silent leaks: Implicit knowledge
extraction attack on RAG systems through benign
queries. InICML 2025 Workshop on Reliable and
Responsible Foundation Models.
Chris M Ward and Josh Harguess. 2025. Adversar-
ial threat vectors and risk mitigation for retrieval-
augmented generation systems. InAssurance and
Security for AI-enabled Systems 2025, volume 13476,
pages 80–97. SPIE.
Zhepei Wei, Wei-Lin Chen, and Yu Meng. 2025. In-
structRAG: Instructing retrieval-augmented genera-
tion via self-synthesized rationales. InThe Thirteenth
International Conference on Learning Representa-
tions.
Jiayi Wen, Tianxin Chen, Zhirun Zheng, and
Cheng Huang. 2025a. A few words can distort
graphs: Knowledge poisoning attacks on graph-based
retrieval-augmented generation of large language
models.arXiv preprint arXiv:2508.04276.Tongyu Wen, Chenglong Wang, Xiyuan Yang, Haoyu
Tang, Yueqi Xie, Lingjuan Lyu, Zhicheng Dou, and
Fangzhao Wu. 2025b. Defending against indirect
prompt injection by instruction detection. InFind-
ings of the Association for Computational Linguistics:
EMNLP 2025, pages 19472–19487, Suzhou, China.
Association for Computational Linguistics.
Shangyu Wu, Ying Xiong, Yufei Cui, Haolun Wu, Can
Chen, Ye Yuan, Lianming Huang, Xue Liu, Tei-
Wei Kuo, Nan Guan, and 1 others. 2024. Retrieval-
augmented generation for natural language process-
ing: A survey.arXiv preprint arXiv:2407.13193.
Chong Xiang, Tong Wu, Zexuan Zhong, David Wagner,
Danqi Chen, and Prateek Mittal. 2024. Certifiably
robust rag against retrieval corruption.arXiv preprint
arXiv:2405.15556.
Yilin Xiao, Chuang Zhou, Qinggang Zhang, Bo Li, Qing
Li, and Xiao Huang. 2026. Reliable reasoning path:
Distilling effective guidance for llm reasoning with
knowledge graphs.IEEE Transactions on Knowl-
edge and Data Engineering.
Jin Xie, Songze Li, and Guang Cheng. 2026. Seal-
tag: Self-tag evidence aggregation with probabilistic
circuits for pii-safe retrieval-augmented generation.
arXiv preprint arXiv:2603.17292.
Jiaqi Xue, Mengxin Zheng, Yebowen Hu, Fei Liu, Xun
Chen, and Qian Lou. 2024. Badrag: Identifying vul-
nerabilities in retrieval augmented generation of large
language models.arXiv preprint arXiv:2406.00083.
Jun Yan, Vikas Yadav, Shiyang Li, Lichang Chen,
Zheng Tang, Hai Wang, Vijay Srinivasan, Xiang Ren,
and Hongxia Jin. 2024. Backdooring instruction-
tuned large language models with virtual prompt in-
jection. InProceedings of the 2024 Conference of
the North American Chapter of the Association for
Computational Linguistics: Human Language Tech-
nologies (Volume 1: Long Papers), pages 6065–6086.
Mengyu Yao, Ziqi Zhang, Ning Luo, Shaofei Li, Yifeng
Cai, Xiangqun Chen, Yao Guo, and Ding Li. 2026.
Connect the dots: Knowledge graph-guided crawler
attack on retrieval-augmented generation systems.
arXiv preprint arXiv:2601.15678.
Yifan Yao, Jinhao Duan, Kaidi Xu, Yuanfang Cai, Zhibo
Sun, and Yue Zhang. 2024. A survey on large lan-
guage model (llm) security and privacy: The good,
the bad, and the ugly.High-Confidence Computing,
4(2):100211.
Huanyi Ye, Jiale Guo, Ziyao Liu, and Kwok-Yan Lam.
2025a. Efficient privacy-preserving retrieval aug-
mented generation with distance-preserving encryp-
tion. In2025 3rd International Conference on Foun-
dation and Large Language Models (FLLM), pages
668–676. IEEE.
Kai Ye, Liangcai Su, and Chenxiong Qian. 2025b.
Importsnare: Directed’code manual’hijacking in
retrieval-augmented code generation. InProceedings
18

of the 2025 ACM SIGSAC Conference on Computer
and Communications Security, pages 335–349.
Hao Yu, Aoran Gan, Kai Zhang, Shiwei Tong, Qi Liu,
and Zhaofeng Liu. 2024. Evaluation of retrieval-
augmented generation: A survey. InCCF Conference
on Big Data, pages 102–120. Springer.
Miao Yu, Fanci Meng, Xinyun Zhou, Shilong Wang,
Junyuan Mao, Linsey Pan, Tianlong Chen, Kun
Wang, Xinfeng Li, Yongfeng Zhang, and 1 others.
2025. A survey on trustworthy llm agents: Threats
and countermeasures. InProceedings of the 31st
ACM SIGKDD Conference on Knowledge Discovery
and Data Mining V . 2, pages 6216–6226.
Shenglai Zeng, Jiankun Zhang, Pengfei He, Yiding Liu,
Yue Xing, Han Xu, Jie Ren, Yi Chang, Shuaiqiang
Wang, Dawei Yin, and 1 others. 2024. The good
and the bad: Exploring privacy issues in retrieval-
augmented generation (rag). InFindings of the As-
sociation for Computational Linguistics: ACL 2024,
pages 4505–4524.
Shenglai Zeng, Jiankun Zhang, Pengfei He, Jie Ren,
Tianqi Zheng, Hanqing Lu, Han Xu, Hui Liu, Yue
Xing, and Jiliang Tang. 2025a. Mitigating the pri-
vacy issues in retrieval-augmented generation (rag)
via pure synthetic data. InProceedings of the 2025
Conference on Empirical Methods in Natural Lan-
guage Processing, pages 24538–24569.
Zhirui Zeng, Jiamou Liu, Meng-Fen Chiang, Jialing
He, and Zijian Zhang. 2025b. S-rag: A novel audit
framework for detecting unauthorized use of personal
data in rag systems. InProceedings of the 63rd An-
nual Meeting of the Association for Computational
Linguistics (Volume 1: Long Papers), pages 10375–
10385.
Baolei Zhang, Yuxi Chen, Zhuqing Liu, Lihai Nie, Tong
Li, Zheli Liu, and Minghong Fang. 2025a. Prac-
tical poisoning attacks against retrieval-augmented
generation.arXiv preprint arXiv:2504.03957.
Baolei Zhang, Haoran Xin, Minghong Fang, Zhuqing
Liu, Biao Yi, Tong Li, and Zheli Liu. 2025b. Trace-
back of poisoning attacks to retrieval-augmented gen-
eration. InProceedings of the ACM on Web Confer-
ence 2025, pages 2085–2097.
Baolei Zhang, Haoran Xin, Jiatong Li, Dongzhe Zhang,
Minghong Fang, Zhuqing Liu, Lihai Nie, and Zheli
Liu. 2025c. Benchmarking poisoning attacks against
retrieval-augmented generation.arXiv preprint
arXiv:2505.18543.
Jiankun Zhang, Shenglai Zeng, Jie Ren, Tianqi Zheng,
Hui Liu, Xianfeng Tang, and Yi Chang. 2025d. Be-
yond text: Unveiling privacy vulnerabilities in multi-
modal retrieval-augmented generation. InProceed-
ings of the 2025 Conference on Empirical Methods in
Natural Language Processing, pages 24800–24821.Quan Zhang, Binqi Zeng, Chijin Zhou, Gwihwan
Go, Heyuan Shi, and Yu Jiang. 2024a. Human-
imperceptible retrieval poisoning attacks in llm-
powered applications. InCompanion Proceedings of
the 32nd ACM International Conference on the Foun-
dations of Software Engineering, pages 502–506.
Yuan Zhang, Jionghan Wu, Rui Li, Tong Zhang, Yu-
jie Song, Chuanyi Li, Shangqi Wang, Hao Shen,
Jiao Yin, Jidong Ge, and 1 others. 2026. Privacy
protection in rag: A novel method and evaluation
framework.Information Processing & Management,
63(3):104505.
Yucheng Zhang, Qinfeng Li, Tianyu Du, Xuhong Zhang,
Xinkui Zhao, Zhengwen Feng, and Jianwei Yin.
2024b. Hijackrag: Hijacking attacks against retrieval-
augmented large language models.arXiv preprint
arXiv:2410.22832.
Dongfang Zhao. 2024. Frag: Toward federated vec-
tor database management for collaborative and se-
cure retrieval-augmented generation.arXiv preprint
arXiv:2410.13272.
Jingjie Zheng, Aryo Pradipta Gema, Giwon Hong, Xu-
anli He, Pasquale Minervini, Youcheng Sun, and
Qiongkai Xu. 2025. Grada: Graph-based reranking
against adversarial documents attack. InProceedings
of the 2025 Conference on Empirical Methods in
Natural Language Processing, pages 22255–22277.
Huichi Zhou, Kin-Hei Lee, Zhonghao Zhan, Yue Chen,
Zhenhao Li, Zhaoyang Wang, Hamed Haddadi, and
Emine Yilmaz. 2025a. Trustrag: enhancing robust-
ness and trustworthiness in retrieval-augmented gen-
eration.arXiv preprint arXiv:2501.00879.
Pengcheng Zhou, Yinglun Feng, and Zhongliang
Yang. 2025b. Privacy-aware rag: Secure and
isolated knowledge retrieval.arXiv preprint
arXiv:2503.15548.
Pengcheng Zhou, Yinglun Feng, and Zhongliang Yang.
2025c. Provably secure retrieval-augmented genera-
tion.arXiv preprint arXiv:2508.01084.
Yujia Zhou, Yan Liu, Xiaoxi Li, Jiajie Jin, Hongjin Qian,
Zheng Liu, Chaozhuo Li, Zhicheng Dou, Tsung-
Yi Ho, and Philip S Yu. 2024. Trustworthiness in
retrieval-augmented generation systems: A survey.
arXiv preprint arXiv:2409.10102.
Kunlun Zhu, Yifan Luo, Dingling Xu, Yukun Yan,
Zhenghao Liu, Shi Yu, Ruobing Wang, Shuo Wang,
Yishan Li, Nan Zhang, and 1 others. 2025. Rageval:
Scenario specific rag evaluation dataset generation
framework. InProceedings of the 63rd Annual Meet-
ing of the Association for Computational Linguistics
(Volume 1: Long Papers), pages 8520–8544.
Wei Zou, Runpeng Geng, Binghui Wang, and Jinyuan
Jia. 2025. PoisonedRAG: Knowledge corruption at-
tacks to retrieval-augmented generation of large lan-
guage models. In34th USENIX Security Symposium
(USENIX Security 25), pages 3827–3844.
19