# MetaRAG: Metamorphic Testing for Hallucination Detection in RAG Systems

**Authors**: Channdeth Sok, David Luz, Yacine Haddam

**Published**: 2025-09-11 11:18:23

**PDF URL**: [http://arxiv.org/pdf/2509.09360v1](http://arxiv.org/pdf/2509.09360v1)

## Abstract
Large Language Models (LLMs) are increasingly deployed in enterprise
applications, yet their reliability remains limited by hallucinations, i.e.,
confident but factually incorrect information. Existing detection approaches,
such as SelfCheckGPT and MetaQA, primarily target standalone LLMs and do not
address the unique challenges of Retrieval-Augmented Generation (RAG) systems,
where responses must be consistent with retrieved evidence. We therefore
present MetaRAG, a metamorphic testing framework for hallucination detection in
Retrieval-Augmented Generation (RAG) systems. MetaRAG operates in a real-time,
unsupervised, black-box setting, requiring neither ground-truth references nor
access to model internals, making it suitable for proprietary and high-stakes
domains. The framework proceeds in four stages: (1) decompose answers into
atomic factoids, (2) generate controlled mutations of each factoid using
synonym and antonym substitutions, (3) verify each variant against the
retrieved context (synonyms are expected to be entailed and antonyms
contradicted), and (4) aggregate penalties for inconsistencies into a
response-level hallucination score. Crucially for identity-aware AI, MetaRAG
localizes unsupported claims at the factoid span where they occur (e.g.,
pregnancy-specific precautions, LGBTQ+ refugee rights, or labor eligibility),
allowing users to see flagged spans and enabling system designers to configure
thresholds and guardrails for identity-sensitive queries. Experiments on a
proprietary enterprise dataset illustrate the effectiveness of MetaRAG for
detecting hallucinations and enabling trustworthy deployment of RAG-based
conversational agents. We also outline a topic-based deployment design that
translates MetaRAG's span-level scores into identity-aware safeguards; this
design is discussed but not evaluated in our experiments.

## Full Text


<!-- PDF content starts -->

MetaRAG: Metamorphic Testing for Hallucination
Detection in RAG Systems
Channdeth SOKa, b,*, David LUZaand Yacine HADDAMa
aForvia, GIT, Immeuble Lumière, 40 avenue des Terroirs de France – 75012 Paris – France
bENSAE Paris, Institut Polytechnique de Paris
ORCID (Channdeth SOK): https://orcid.org/0009-0008-4547-5946
Abstract.Large Language Models (LLMs) are increasingly de-
ployed in enterprise applications, yet their reliability remains limited
byhallucinations, i.e., confident but factually incorrect information.
Existing detection approaches, such as SelfCheckGPT and MetaQA,
primarily target standalone LLMs and do not address the unique chal-
lenges of Retrieval-Augmented Generation (RAG) systems, where
responses must be consistent with retrieved evidence. We therefore
presentMetaRAG, ametamorphic testing framework for hallucina-
tion detection inRetrieval-AugmentedGeneration (RAG) systems.
MetaRAG operates in a real-time, unsupervised, black-box setting,
requiring neither ground-truth references nor access to model in-
ternals, making it suitable for proprietary and high-stakes domains.
The framework proceeds in four stages: (1) decompose answers into
atomic factoids, (2) generate controlled mutations of each factoid
using synonym and antonym substitutions, (3) verify each variant
against the retrieved context (synonyms are expected to be entailed
and antonyms contradicted), and (4) aggregate penalties for incon-
sistencies into a response-level hallucination score. Crucially for
identity-aware AI,MetaRAGlocalizes unsupported claims at the
factoid span where they occur (e.g., pregnancy-specific precautions,
LGBTQ+ refugee rights, or labor eligibility), allowing users to see
flagged spans and enabling system designers to configure thresholds
and guardrails for identity-sensitive queries. Experiments on a pro-
prietary enterprise dataset illustrate the effectiveness ofMetaRAG
for detecting hallucinations and enabling trustworthy deployment of
RAG-based conversational agents. We also outline a topic-based de-
ployment design that translates MetaRAG’s span-level scores into
identity-aware safeguards; this design is discussed but not evaluated
in our experiments.
1 Introduction
Large Language Models (LLMs) such as GPT-4 and Llama-3 are
transforming enterprise applications in healthcare, law, and customer
service [8, 7, 2]. They power chatbots and virtual assistants that in-
teract in natural language, offering unprecedented convenience and
efficiency [30]. However, as these systems move into production, a
persistent challenge emerges:hallucinations, i.e., responses that are
fluent and convincing but factually incorrect or unsupported by evi-
dence [23, 14].
In high-stakes domains such as healthcare, law, and finance, hal-
lucinations are not merely a nuisance but a critical barrier to reliable
∗Corresponding Author. Email: channdeth.sok@forvia.comadoption, raising concerns about user trust, regulatory compliance,
and business risk [20]. Moreover, hallucinations are not uniformly
risky: the same unsupported claim can differentially affect specific
populations. In healthcare (e.g., pregnancy/trimester-specific con-
traindications), migration and asylum (e.g., protections for LGBTQ+
refugees), or labor rights (e.g., eligibility by status), ungrounded
spans can cause disproportionate harm. Rather than treating users
as homogeneous, hallucination detection methods should make such
spansreviewableat the factoid level so downstream systems can ap-
ply identity-aware policies (e.g., stricter thresholds, forced citations,
or escalation to a human) when the topic indicates elevated risk. This
perspective connects hallucination detection to identity-aware de-
ployment, where span-level evidence enables topic-conditioned safe-
guards that reduce disproportionate risk.
Ji et al. [14] categorize hallucinations into two types:
•Intrinsic hallucination: fabricated or contradictory information
relative to the model’s internal knowledge.
•Extrinsic hallucination: generated information that conflicts
with, misrepresents, or disregards externally provided context or
retrieved documents.
Figure 1.Standard Retrieval-Augmented Generation (RAG) workflow. A
user query is encoded into a vector representation using an embedding
model and queried against a vector database constructed from a document
corpus. The most relevant document chunks are retrieved and appended to
the original query, which is then provided as input to a large language model
(LLM) to generate the final response.
Retrieval-Augmented Generation (RAG) [17] aims to mitigate hal-
lucinations by grounding model outputs in retrieved, up-to-date doc-
uments, as illustrated in Figure 1. By injecting retrieved text from re-arXiv:2509.09360v1  [cs.CL]  11 Sep 2025

liable external sources and proprietary documents, into the prompt,
RAG improves factuality and domain relevance. While effective
against intrinsic hallucinations, RAG remains susceptible toextrinsic
hallucinations, especially when retrieved evidence is ignored, misin-
terpreted, or insufficient [10].
Detecting hallucinations is particularly challenging in real-world
settings, where RAG-based chatbots must respond to queries about
unseen, proprietary, or confidential content where gold-standard ref-
erences are typically unavailable [19]. Many existing hallucination
detection methods rely on gold-standard reference answers [23, 16],
annotated datasets [33], or access to model internals such as hid-
den states or token log-probabilities [25, 6]. However, in enterprise
settings, such internals are often inaccessible: many state-of-the-art
LLMs (e.g., GPT-4, Claude) are proprietary and only accessible via
APIs that expose the final output text but not intermediate computa-
tions, limiting the feasibility of these methods in practice [19].
To address these challenges, we introduceMetaRAG: a metamor-
phic testing framework for detecting hallucinations in RAG-based
conversational agents.MetaRAG is a zero-resource, black-box set-
tingthat decomposes answers into atomic factoids, applies con-
trolled mutations (e.g., synonym and antonym substitutions), and
verifies each mutated factoid against the retrieved context. Synonyms
are expected to beentailed, while antonyms are expected to be
contradicted. Hallucinations are flagged when outputs violate these
well-defined metamorphic relations (MRs). Unlike prior approaches,
MetaRAG does not require ground-truth labels, annotated corpora,
or access to model internals, making it suitable for deployment in
proprietary settings.
We evaluate MetaRAG on a proprietary corpus, thus unseen dur-
ing model training. Our results show that MetaRAG reliably detects
hallucinations, providing actionable insights for enhancing chatbot
reliability and trustworthiness. These results establish MetaRAG as
a practical tool for reliable deployment, and its span-level detection
opens the door to identity-aware safeguards.
Our contributions include:
•We introduceMetaRAG, a reference-free, black-box setting,
metamorphic testing framework for hallucination detection in
RAG systems. It decomposes answers into factoids, applies lin-
guistic transformations (synonym and antonym), and verifies them
against retrieved context to produce a hallucination score.
•We implement a prototype and evaluate MetaRAG on a propri-
etary dataset, demonstrating its effectiveness in detecting halluci-
nations that occur when segments of generated responses diverge
from the retrieved context.
•We analyze the performance–latency/cost trade-offs of MetaRAG
and provide a consistency analysis to guide future research and
practical deployment.
•We outline identity-aware safeguards (topic-aware thresholds,
forced citation, escalation) that consume MetaRAG’s scores; these
safeguards are a deployment design and are not part of our empir-
ical evaluation.
2 Related Works
2.1 Definitions of Hallucination
The termhallucinationhas been used with varying scope across
natural language generation tasks. Some studies emphasizefactu-
ality, describing hallucinations as outputs that contradict established
facts, i.e., inconsistencies with world knowledge or external ground
truth [31, 13]. Others highlightfaithfulness, where hallucinationsoccur when generated responses deviate from the user instruction
or a reference text, often producing plausible but ungrounded state-
ments particularly in source-conditioned tasks such as summariza-
tion or question answering [26]. Beyond these two dimensions, re-
searchers also note cases of incoherent or nonsensical text that cannot
be clearly attributed to factuality or faithfulness criteria [14, 23].
Alternative terms have also been introduced.Confabulationdraws
on psychology to describe fluent but fabricated content arising from
model priors [9], whilefabricationis preferred by some to avoid an-
thropomorphic connotations [1, 21]. More recently, Chakraborty et
al. [3] propose a flexible definition tailored to deployment settings,
defining a hallucination asa generated output that conflicts with
constraints or deviates from desired behavior in actual deployment,
while remaining syntactically plausible under the circumstance.
2.2 Hallucination Detection in LLMs
Building on these definitions, hallucinations have been recognized as
a major challenge in text generation. Early work in machine transla-
tion and abstractive summarization described them as outputs that are
not grounded in the input source [23, 16, 15], motivating the devel-
opment of evaluation metrics and detection methods for faithfulness
and factual consistency across natural language generation tasks.
More recentreference-free(unsupervised or zero-reference) meth-
ods aim to detect hallucinations without gold-standard labels by ana-
lyzing the model’s own outputs. A prominent method isSelfCheck-
GPT[22], a zero-resource, black-box approach that queries the LLM
multiple times with the same prompt and measures semantic con-
sistency across responses. The intuition is that hallucinated content
often leads to instability under stochastic re-generation; true facts re-
main stable, while fabricated ones diverge. Manakul et al. show that
SelfCheckGPT achieves strong performance in sentence-level hal-
lucination detection compared to gray-box methods, and emphasize
that it requires no external database or access to model internals [22].
However, SelfCheckGPT may struggle when deterministic decoding
or high model confidence leads to repeating the same incorrect out-
put.
2.3 Metamorphic Testing
Metamorphic Testing (MT) [4] was originally proposed in software
engineering to address theoracle problemin which the correct output
is unknown. MT relies onmetamorphic relations(MRs): transforma-
tions of the input with predictable effects on outputs, enabling error
detection without access to ground truth [28]. In machine learning,
MT has been applied to validate models in computer vision [5] (e.g.,
rotating an image should not change its predicted class) and NLP [27]
In hallucination detection for LLMs,MetaQA[32] leverages MRs
by generating paraphrased or antonym-based question variants and
verifying whether answers satisfy expected semantic or logical con-
straints. Relying purely on prompt mutations and consistency checks,
MetaQA achieves higher precision and recall than SelfCheckGPT on
open-domain QA.
Researchers have also adapted MT for more complex conversa-
tional and reasoning settings.MORTAR[12] applies dialogue-level
perturbations and knowledge-graph-based inference to multi-turn
systems, detecting up to four times more unique bugs than single-
turn MT.Drowzee[18] uses logic programming to construct tempo-
ral and logical rules from Wikipedia, generating fact-conflicting test
cases and revealing rates of 24.7% to 59.8% across six LLMs in nine
domains [32].

These works highlight the promise of MT for hallucination de-
tection, but they primarily target open-book QA or multi-turn dia-
logue, often over short, single-sentence outputs. Prior studies have
not addressed hallucination detection inretrieval-augmented gener-
ation(RAG) scenarios over proprietary corpora, a setting in which
ground-truth references are unavailable and model internals are in-
accessible.MetaRAGbuilds on MT by decomposing answers into
factoids and designing MRs tailored to factual consistency against
retrieved evidence in a zero-resource, black-box setting.
3 MetaRAG: Methodology
3.1 Overview
Building on the metamorphic testing (MT) methodology to detect
hallucinations in LLMs introduced by MetaQA [32],MetaRAGad-
vances this approach to detect hallucinations in retrieval-augmented
generation (RAG) settings by introducing a context-based verifica-
tion stage. A metamorphic testing layer operates on top of the stan-
dard RAG pipeline to automatically detect hallucinated responses.
Figure 2 outlines the workflow.
Given a user queryQ, the system retrieves the top-kmost relevant
chunks from a database, forming the contextC={c 1, c2, . . . , c k}.
The LLM generates an initial answerAusing(Q, C)as input.
MetaRAG then decomposesAinto factoids, applies controlled
metamorphic transformations to produce variants (synonym and
antonym), verifies each variant againstC, and aggregates the results
into a hallucination score (Algorithm 1).
Figure 2.Overview of theMetaRAGworkflow.(A)Integration of
MetaRAG with a standard RAG pipeline: given a user question, the RAG
retrieves context and generates an answer, which is then passed to MetaRAG
for hallucination detection.(B)Internal MetaRAG pipeline: the answer is
decomposed into atomic factoids, each factoid is mutated through synonym
and antonym substitutions, and verified against the retrieved context using
entailment/contradiction checks. Penalties are assigned to inconsistencies,
and scores are aggregated into a response-level hallucination score.
3.2 Step 1: Factoid decomposition
Given an answerA, we first decompose it into a set offactoids,
defined as atomic, independently verifiable facts, denoted byF=
{F1, . . . , F M}. Each factoidF jcorresponds to a single factual state-
ment that cannot be further divided without losing meaning, such as
a subject-predicate-object triple or a scoped numerical or temporalAlgorithm 1MetaRAG Hallucination Detection
1:Input:Generated answerA, queryQ, contextC, number of mu-
tationsN, thresholdτ
2:Output:Hallucination scoreH(Q, A, C), factoid scores{S i}
3:Factoid Extraction:
4:F ←FACTOIDSDECOMPOSITION(A)
5:foreach factoidF iinFdo
6:Mutation Generation:
7:Synonyms←GENERATESYNONYMMUTATIONS(F i, Q, N)
8:Antonyms←GENERATEANTONYMMUTATIONS(F i, Q, N)
9:Verification:
10:foreachFsyninSynonymsdo
11:result←VERIFYWITHLLM(Fsyn, C)
12:SynResults.append(result)
13:end for
14:foreachFantinAntonymsdo
15:result←VERIFYWITHLLM(Fant, C)
16:AntResults.append(result)
17:end for
18:Scoring:
19:SynScores←[MAPSYNONYMSCORE(r)forrinSynResults]
20:AntScores←[MAPANTONYMSCORE(r)forrinAntResults]
21:S i←Mean(SynScores∪AntScores)
22:end for
23:Aggregation:
24:H(Q, A, C)←max iSi
25:ReturnH(Q, A, C),{S i}
claim. Representing an answerAat the factoid level enables fine-
grained verification in subsequent steps, allowing localized halluci-
nations to be marked inside longer answers.
We obtainFusing an LLM-based extractor with a fixed prompt
that enforces one proposition per line, prohibits paraphrasing or in-
ference beyondA, and co-reference resolution. The full prompt tem-
plate is provided in the supplementary material.
3.3 Step 2: Mutation Generation
Each factoid (hereafter,fact) from Step 1, MetaRAG applies meta-
morphic mutations to generate perturbed variants of the original
claim. This step is grounded in the principle of metamorphic testing,
where controlled semantic transformations are used to probe model
consistency and expose hallucinations [32].
Formally, for each factoidF i∈ {F 1, . . . , F M}, we construct vari-
ants using two relations:
•Synonym Mutation: This relation substitutes key terms inF i
with appropriate synonyms, yielding paraphrased factoidsFsyn
i,j
that preserve the original semantic meaning. These assess the
model’s ability to recognize reworded yet factually equivalent
statements.
•Antonym Mutation: This relation replaces key terms inF iwith
antonyms or negations, producing factoidsFant
i,jthat are seman-
tically opposed to the original. These serve as adversarial tests to
ensure the model does not support clearly contradictory informa-
tion.
LetNdenote the number of mutations generated byeachrelation.
The mutation set forF iis therefore
Fi={Fsyn
i,1, . . . , Fsyn
i,N, Fant
i,1, . . . , Fant
i,N}.

By construction, ifF iis correct and supported by the retrieved
contextC, thenFsyn
i,·should beentailedbyC, whereasFant
i,·should
becontradictedbyC.
Mutations are generated by prompting an LLM with templates that
explicitly instruct synonymous or contradictory outputs while pre-
serving atomicity and relevance; the exact prompt templates appear
in the supplementary material.
3.4 Step 3: Factoid Verification
Each mutated factoidFsyn
i,jandFant
i,jis then verified by LLMs condi-
tioning on the contextC(treated as ground truth). The LLM returns
one of three decisions: YES(entailed byC), NO(contradicted byC),
or NOT SURE(insufficient evidence). We then assign a penalty score
p∈ {0,0.5,1}based on the decision and the mutation type:
Table 1.Penalty scheme for metamorphic verification (lower is better).
Penaltyp
Decision Synonym Antonym
YES0.0 1.0
NOT SURE0.5 0.5
NO1.0 0.0
This penalty assignment quantifies semantic (in)consistency at the
variant level: correct entailment for synonyms and correct contradic-
tion for antonyms yield zero penalty, while the opposite yields maxi-
mal penalty. In Step 4, we aggregate these penalties over all variants
of eachF ito compute a fact-level hallucination score.
3.5 Step 4: Score Calculation
To quantify hallucination risk, we calculate a hallucination score,S i,
for each factoidF i. This yields a granular diagnostic that pinpoints
which claims are potentially unreliable. The score for each factoidi
is defined as the average penalty across the2Nmetamorphic trans-
formations (synonym and antonym) ofF i:
Si=1
2N NX
j=1psyn
i,j+NX
j=1pant
i,j!
,(1)
wherepsyn
i,jandpant
i,jare the penalties assigned in Step 3 to thej-th
synonym and antonym variants ofF i, respectively. By construction,
Si∈[0,1]:S i= 0indicates a perfectly consistent, well-grounded
factoid, thus no hallucination, whileS i= 1indicates a highly prob-
able hallucination.
Response Hallucination score:Instead of a simple average, the
hallucination score for the entire responseAis defined as the maxi-
mum score found among all the individual factoids. This metric en-
sures that a single, severe hallucination in any part of the response
will result in a high overall score, accurately reflecting the unrelia-
bility of the entire answer.
H(Q, A, C) = max
1≤i≤MSi,(2)
whereMis the number of decomposed factoids. A response can be
flagged as containing hallucination ifH(Q, A, C)exceeds a prede-
fined confidence thresholdτ∈[0,1](e.g.0.5).3.6 Identity-Aware Safeguards for Deployment
While MetaRAG is a general-purpose hallucination detector, its
factoid-level scores can be directly integrated intoidentity-aware de-
ployment policies. Importantly, no protected attributes are inferred or
stored; instead, only thetopic of the query or retrieved context(e.g.,
pregnancy, refugee rights, labor eligibility) is used as a deployment
signal.Scope.The safeguards described here represent a deployment
design that consumes MetaRAG’s scores; they are not part of the
empirical evaluation reported in Section 4.
Each factoid receives a scoreS i∈[0,1], whereS i= 0indi-
cates full consistency with the retrieved context andS i= 1indi-
cates strong evidence of hallucination. The overall response score
H(Q, A, C)thus represents the risk level of the most unreliable
claim: higher values correspond to higher hallucination risk.
These scores could enable deployment-time safeguards through
the following hooks:
1.Topic detection.A lightweight topic classifier or rule-based tag-
ger can assign coarse domain labels (e.g., healthcare, migration,
labor) to the query or retrieved context.
2.Topic-aware thresholds.A response is flagged ifH(Q, A, C)≥
τ. Thresholds can be adapted by domain, e.g.,τ general = 0.5for
generic queries, and a stricterτ identity = 0.3for sensitive domains.
3.Span highlighting and forced citation.For flagged responses,
MetaRAG highlights unsupported spans and enforces inline cita-
tions to retrieved evidence, to improve transparency and calibrate
user trust.
4.Escalation.If hallucinations persist above threshold in identity-
sensitive domains, the system may abstain, regenerate with a
stricter prompt, or escalate to human review.
5.Auditing.Logs of flagged spans, hallucination scores, and topic
labels can be maintained for post-hoc fairness, compliance, and
safety audits.
In this way, higher hallucination scores are systematically trans-
lated into stronger protective actions, with more conservative safe-
guards applied whenever queries touch on identity-sensitive con-
texts.
4 Experiments
We conducted experiments to evaluateMetaRAGon its ability to
detecthallucinations in retrieval-augmented generation (RAG). The
evaluation simulates a realistic enterprise deployment setting, in
which a chatbot serves responses generated from internal documen-
tation. Our focus is on the detection stage, that is, identifying when
an answer contains unsupported (hallucination) or fabricated infor-
mation. Prevention and mitigation are important but they are outside
the scope of this work.
4.1 Dataset
The evaluation dataset is a proprietary collection of23 inter-
nal enterprise documents, including policy manuals, procedural
guidelines, and analytical reports, none of which were seen dur-
ing LLM training. Each document was segmented into chunks of
a few hundred tokens, and retrieval used cosine similarity over
text-embedding-3-large, with the top-k= 3chunks ap-
pended to each query.
We then collected a set of user queries and corresponding chatbot
answers. Each response was labeled by human annotators as either

hallucinated or not, using the retrieved context as the reference. The
final evaluation set contains67 responses, of which36are labeled as
not hallucinatedand31ashallucinated.
To preserve confidentiality, we do not release the full annotated
dataset. However, the complete annotation guidelines are included in
the supplementary material.
4.2 Evaluation Protocol
MetaRAG producesfine-grained, factoid-level hallucination
scores, whereas the available ground truth labels are at theresponse
level. To align with these existing labels, we evaluate MetaRAG as a
binary classifier by thresholding the hallucination scoreH(Q, A, C)
atτ= 0.5. We report standard classification metrics: Precision, Re-
call, F1 score and accuracy. Latency is also recorded to assess feasi-
bility for real-time deployment.
4.2.1 Case Studies in Identity-Sensitive Domains
Beyond quantitative evaluation, we also provide qualitative illustra-
tions of MetaRAG in identity-sensitive scenarios. To illustrate how
MetaRAG’s span-level scores can enable identity-aware safeguards
without inferring protected attributes, we present two stylized exam-
ples. These are not part of the quantitative evaluation in Section 4,
but highlight potential deployment scenarios.
Healthcare (pregnancy).A user asks: “Can pregnant women take
ibuprofen for back pain?” The model answers: “Yes, ibuprofen is
safe throughout pregnancy.” However, the retrieved context speci-
fies that ibuprofen is contraindicated in the third trimester. MetaRAG
flags the span“safe throughout pregnancy”with a high factoid
score (S i= 0.92), yielding a response-level scoreH= 0.92. Under
the policy hooks described in Section 3.6, the topic tagpregnancy
triggers a stricter threshold (τ identity = 0.3, lower than the general
case), span highlighting, a forced citation requirement, and possible
escalation to human review.
Migration (refugee rights).A user asks: “Do LGBTQ+ refugees
automatically receive protectionin country X?” The model claims
that such protections areautomatic, but the retrieved legal text pro-
vides no evidence of this. MetaRAG flags the unsupported claim
“automatically receive protection”with a moderate score (S i=
0.5), yielding a response-level scoreH= 0.5. Although this
score would sit at the decision boundary under a general threshold
(τgeneral = 0.5), the stricter identity-aware threshold (τ identity = 0.3)
ensures it is flagged for this case. Under the policy hooks, the topic
tagasylum/refugeeenforces citation and may escalate the response
to a human reviewer. In a chatbot deployment, the system would ab-
stain from returning the unsupported answer and instead notify the
user that expert verification is required.
These qualitative vignettes complement our quantitative evalua-
tion by showing how MetaRAG’s flagged spans can be turned into
concrete safeguards in identity-sensitive deployments.
5 Ablation Study
To understand the contribution of individual design choices, we per-
form a set of ablation experiments using the private dataset.
5.1 Ablation Study Design
We evaluate 26 configurations of MetaRAG, each defined by a com-
bination of:•Number of variants per relationN∈ {2,5}
•Factoid-decomposition model:gpt-4.1 or gpt-4.1-minifrom Ope-
nAI
•Temperature for mutation generation:T∈ {0.0,0.7}
•Mutation–generation model:gpt-4.1 or gpt-4.1-mini
•Verifier model:gpt-4.1-mini,gpt-4.1, or themultiensemble (gpt-
4.1-nano, gpt-4.1-mini, gpt-4.1, Claude Sonnet 4)
Since the evaluation task is binary classification, we reportPrecision,
Recall,F1 score, andAccuracy, along withlatency(lower is better).
5.2 Results
To provide a comprehensive view of performance trade-offs, we re-
port theTop-4 configurations separatelyfor each of three primary
metrics: F1 score, Precision, and Recall (Table 2). The configuration
notation follows the format:
Decomposition Model / Generation Model /
Verifier /N/ Temperature.
For example,mini/41/multi/2/0indicates that the factoid
decomposition model is “mini”, the variant generation model is “41”,
the verifier is “multi”, there areN= 2variants per relation, and the
temperature is 0.0.
Several configurations appear in more than one top-4 list, re-
flecting balanced performance across metrics. For instance, ID 5
(mini/41/multi/2/0) ranks first in both F1 score and Recall,
while maintaining competitive Precision.
Table 2.Validation leaderboards for 26MetaRAGconfigurations,
showing the top four for each of F1 score, Precision, and Recall.
Top–4 by ID Config. F1 Prec. Rec. Acc.
F15 mini/41/multi/2/0 0.9391 1.0000 0.8853 0.9401
18 mini/mini/41/5/0.7 0.9372 0.9087 0.9676 0.9401
19 mini/mini/41/5/0 0.9352 0.9352 0.9352 0.9400
16 mini/41/multi/5/0 0.9228 0.8819 0.9676 0.9250
Precision22 mini/mini/multi/5/0 0.9149 0.9641 0.8704 0.9250
19 mini/mini/41/5/0 0.9352 0.9352 0.9352 0.9400
18 mini/mini/41/5/0.7 0.9372 0.9087 0.9676 0.9401
24 41/mini/multi/5/0 0.8847 0.8996 0.8704 0.8951
Recall1 mini/41/41/2/0.7 0.8736 0.7755 1.0000 0.8660
5 mini/41/multi/2/0 0.9391 0.8853 1.0000 0.9401
9 mini/mini/mini/2/0.7 0.9114 0.8372 1.0000 0.9101
4 mini/41/41/2/0 0.8614 0.7565 1.0000 0.8510
Config legend:Decomp/GenModel/Verifier/N/Temp.
The most promising configurations are further examined in Sec-
tion 5.3 to verify stability under multiple seeds.
5.3 Consistency Check
To verify the robustness of our results, each top configuration (se-
lected based on F1 score, Precision, Recall, and token usage) is rerun
under identical conditions using five different random seeds. This
procedure serves three purposes:
•To ensure that high performance is not attributable to random ini-
tialization or favorable seeds.
•To quantify variability across runs with the same configuration by
reporting the standard deviation for each metric.

Figure 3.Evaluation metrics for all 26MetaRAGconfigurations.
Table 3.Run-to-run consistency for top configurations (mean±standard
deviation over 5 seeds) and coefficient of variation (CV) for F1.
ID F1 Precision Recall CV (F1)
16 0.9397±0.0123 0.9198±0.0243 0.9610±0.0144 1.31%
18 0.9356±0.0089 0.9322±0.0439 0.9413±0.0278 0.95%
19 0.9347±0.0305 0.9410±0.0357 0.9286±0.0272 3.26%
5 0.9108±0.0346 0.8463±0.0503 0.9869±0.0179 3.80%
•To assess stability using the coefficient of variation (CV) defined
as the ratio of the standard deviation to the mean (CV =σ/µ),
where lower values indicate greater consistency.
Across all metrics, the top configurations demonstrate strong re-
producibility, with the majority exhibiting a CV below 2%. In partic-
ular, configurations 18 and 16 achieve both high F1 scores and low
variability, indicating that they are not only accurate but also stable
across repeated trials.
5.4 Pareto Front Analysis
Following the consistency check (Section 5.3), we restrict the Pareto
front analysis to the four most stable top-performing configurations
selected by F1 score. We analyze the trade-off between hallucina-
tion detection performance and efficiency using Pareto frontiers. A
configuration isPareto-optimalif no other configuration achieves
strictly higher F1 while being no worse in cost metrics; similarly, for
precision–recall trade-off.
Figure 4 presents the Pareto fronts for our primary detection metric
(F1 score) with respect to (i) average token usage, (ii) average total
execution time (second), and (iii) the precision–recall trade-off. The
Pareto front highlights configurations that offer the best possible bal-
ance between accuracy and efficiency, enabling deployment choices
aligned with cost or latency constraints.
Several top-ranked configurations (IDs 5, 18, 19, 16) lie on the
Pareto front across these views, indicating that they offer competitive
accuracy without excessive cost. The corresponding Pareto analysesfor precision and recall metrics are provided in the Supplementary
Material.
6 Discussion
6.1 Practical Implications
Integrating hallucination detection into enterprise RAG systems of-
fers several advantages:
•Risk Mitigation: Early detection of unsupported answers miti-
gates the spread of misinformation in both customer-facing and
internal applications.
•Regulatory Compliance: Many industries, such as healthcare and
finance, require verifiable information; automated detection sup-
ports regulatory compliance.
•Operational Efficiency: Detecting hallucinations simultaneously
with content delivery reduces the need for costly downstream hu-
man verification.
6.2 Ethical Considerations
Beyond technical performance, hallucination detection intersects di-
rectly with questions of fairness, accountability, and identity harms.
Hallucinations in chatbot systems pose risks that extend beyond fac-
tual inaccuracies: they can reinforce harmful stereotypes, undermine
user trust, and misrepresent marginalized communities in identity-
sensitive contexts.
•Reinforced stereotypes:Language models are known to repro-
duce and amplify societal biases, as demonstrated by benchmarks
such as StereoSet [24] and WinoBias [34]. In identity-sensitive
deployments, hallucinated outputs risk reinforcing these biases in
subtle but harmful ways.
•Trust erosion:Chatbots are only adopted at scale in high-stakes
domains if users trust their outputs. Surveys on hallucination con-
sistently highlight that exposure to unsupported or fabricated con-
tent undermines user trust in LLM systems [14, 20].

Figure 4.Pareto front analysis for hallucination detection performance. Each point represents a MetaRAG configuration; Pareto-optimal points
(non-dominated) are highlighted. Subplots show: (Left) F1 vs. average token usage, (Center) F1 vs. average total execution time, (Right) Precision vs. Recall.
Pareto-optimal points represent configurations with no strictly better alternative in both accuracy and cost. Configuration IDs correspond to Table 2.
•Identity harms:Misrepresentations in generated responses
may distort personal narratives or marginalize underrepresented
groups, aligning with broader critiques that technical systems can
reproduce social inequities if identity considerations are over-
looked [11, 29].
By detecting hallucinations in a black-box, reference-free manner,
MetaRAGsupports safer deployment of RAG-based systems, par-
ticularly in settings where fairness, identity, and user well-being are
at stake.
6.3 Limitations and Future Work
While MetaRAG demonstrates strong hallucination detection perfor-
mance, several limitations remain:
•Dataset Scope:The study relies on a private, domain-specific
dataset. This may limit external validity.Future work should focus
on curating or constructing public benchmarks designed to avoid
overlap with LLM pretraining corpora, enabling more robust gen-
eralizability.
•Annotation Granularity:We lack factoid-level ground truth,
which reduces our ability to assess fine-grained reasoning accu-
racy.Providing such annotations in future datasets would support
deeper consistency evaluations.
•Policy Hooks Not Evaluated:The identity-aware deployment
hooks introduced in Section 3.6 are presented only as a design
concept. In our implementation, we used a fixed threshold of
τ= 0.5across all queries.Future research should implement and
measure the effectiveness of topic-aware thresholds, forced cita-
tion, and escalation strategies in real-world chatbot deployments.
•Topic as Proxy (Design Limitation):In Section 3.6, we suggest
topic tags (e.g., pregnancy, asylum, labor) as privacy-preserving
signals for stricter safeguards, rather than inferring protected at-
tributes. This was not implemented in our experiments. As a de-
sign idea, it may also miss cases where risk is identity-conditioned
but the query appears generic.Future work should explore how to
operationalize such topic-aware safeguards and investigate richer,
privacy-preserving signals that better capture identity-sensitive
risks.
•Model Dependency:Current findings hinge on specific LLMs
(GPT-4.1 variants). As models evolve, the behavior of MetaRAGmay shift.Future efforts should validate MetaRAG across open-
source and emerging models to reinforce its robustness.
•Efficiency and Cost:The verification steps add computational
overhead, possibly impacting deployment in latency sensitive
environments.Investigating lighter-weight verification strategies
and adaptive scheduling techniques could help mitigate this trade-
off.
•Context Modality:Our current formulation assumes that the re-
trieved contextCis textual, enabling direct comparison through
language-based verification. However, RAG pipelines increas-
ingly operate over multimodal contexts such as tables, structured
knowledge bases, or images.Future work should extend MetaRAG
to handle non-textual evidence, requiring modality-specific verifi-
cation strategies (e.g., table grounding, multimodal alignment).
Together, these limitations highlight both immediate boundaries
and promising future directions for enhancing MetaRAG’s reliability,
fairness, and efficiency.
7 Conclusion
Hallucinations in RAG-based conversational agents remain a signifi-
cant barrier to trustworthy deployment in real-world applications. We
introducedMetaRAG, a metamorphic testing framework for halluci-
nation detection in retrieval-augmented generation (RAG) that oper-
ates without requiring ground truth references or access to model in-
ternals. Our experiments show that MetaRAG achieves strong detec-
tion performance on a challenging proprietary dataset, aligning with
prior benchmark studies. Beyond general reliability, MetaRAG’s
factoid-level localization also supports identity-aware deployment
by surfacing unsupported claims in sensitive domains (e.g., health-
care, migration, labor). Looking ahead, we see MetaRAG as a step
toward safer and fairer conversational AI, where hallucinations are
not only detected but also connected to safeguards that protect users
in identity-sensitive contexts. This connection to identity-aware AI
ensures that hallucination detection does not treat all users as homo-
geneous but provides safeguards that reduce disproportionate risks
for marginalized groups.
References
[1] R. Azamfirei, S. R. Kudchadkar, and J. Fackler. Large language models
and the perils of their hallucinations.Critical Care, 27(120).

[2] R. Bommasani, D. A. Hudson, E. Adeli, and et al. On the opportuni-
ties and risks of foundation models.arXiv preprint arXiv:2108.07258,
2021.
[3] N. Chakraborty, M. Ornik, and K. Driggs-Campbell. Hallucination
detection in foundation models for decision-making: A flexible def-
inition and review of the state of the art.ACM Comput. Surv., 57
(7), Mar. 2025. ISSN 0360-0300. doi: 10.1145/3716846. URL
https://doi.org/10.1145/3716846.
[4] T. Y . Chen, S. C. Cheung, and S. M. Yiu. Metamorphic testing: A new
approach for generating next test cases, 2020. URL https://arxiv.org/
abs/2002.12543.
[5] A. Dwarakanath, M. Ahuja, S. Sikand, R. M. Rao, R. P. J. C. Bose,
N. Dubash, and S. Podder. Identifying implementation bugs in ma-
chine learning based image classifiers using metamorphic testing. In
Proceedings of the 27th ACM SIGSOFT International Symposium on
Software Testing and Analysis, ISSTA ’18, page 118–128. ACM, July
2018. doi: 10.1145/3213846.3213858. URL http://dx.doi.org/10.1145/
3213846.3213858.
[6] N. Dziri, E. Kamalloo, S. Milton, O. Zaiane, M. Yu, E. M. Ponti, and
S. Reddy. FaithDial: A faithful benchmark for information-seeking di-
alogue.Transactions of the Association for Computational Linguis-
tics, 10:1473–1490, 2022. doi: 10.1162/tacl_a_00529. URL https:
//aclanthology.org/2022.tacl-1.84/.
[7] A. G. et al. The llama 3 herd of models, 2024. URL https://arxiv.org/
abs/2407.21783.
[8] O. et al. Gpt-4 technical report, 2024. URL https://arxiv.org/abs/2303.
08774.
[9] S. et al. Confabulation: The surprising value of large language model
hallucinations. InProceedings of the 62nd Annual Meeting of the
Association for Computational Linguistics (Volume 1: Long Papers),
pages 14274–14284, Bangkok, Thailand, Aug. 2024. Association for
Computational Linguistics. doi: 10.18653/v1/2024.acl-long.770. URL
https://aclanthology.org/2024.acl-long.770/.
[10] T. Gao, C. Zhu, Z. Zhang, and et al. Rag can still hallucinate: Faith-
fulness evaluation for retrieval-augmented generation.arXiv preprint
arXiv:2304.09848, 2023.
[11] T. Gebru, J. Morgenstern, B. Vecchione, J. W. Vaughan, H. Wallach,
H. D. III, and K. Crawford. Datasheets for datasets.Commun. ACM, 64
(12):86–92, Nov. 2021. ISSN 0001-0782. doi: 10.1145/3458723. URL
https://doi.org/10.1145/3458723.
[12] G. Guo, A. Aleti, N. Neelofar, C. Tantithamthavorn, Y . Qi, and T. Y .
Chen. Mortar: Multi-turn metamorphic testing for llm-based dialogue
systems, 2025. URL https://arxiv.org/abs/2412.15557.
[13] L. Huang, W. Yu, W. Ma, W. Zhong, Z. Feng, H. Wang, Q. Chen,
W. Peng, X. Feng, B. Qin, and T. Liu. A survey on hallucina-
tion in large language models: Principles, taxonomy, challenges, and
open questions.ACM Transactions on Information Systems, 43(2):
1–55, Jan. 2025. ISSN 1558-2868. doi: 10.1145/3703155. URL
http://dx.doi.org/10.1145/3703155.
[14] Z. Ji, N. Lee, R. Frieske, T. Yu, D. Su, Y . Xu, E. Ishii, Y . J. Bang,
A. Madotto, and P. Fung. Survey of hallucination in natural language
generation.ACM Comput. Surv., 55(12), Mar. 2023. ISSN 0360-0300.
doi: 10.1145/3571730. URL https://doi.org/10.1145/3571730.
[15] P. Koehn and R. Knowles. Six challenges for neural machine trans-
lation. In T. Luong, A. Birch, G. Neubig, and A. Finch, editors,
Proceedings of the First Workshop on Neural Machine Translation,
pages 28–39, Vancouver, Aug. 2017. Association for Computational
Linguistics. doi: 10.18653/v1/W17-3204. URL https://aclanthology.
org/W17-3204/.
[16] W. Kryscinski, B. McCann, C. Xiong, and R. Socher. Evaluating the
factual consistency of abstractive text summarization. In B. Web-
ber, T. Cohn, Y . He, and Y . Liu, editors,Proceedings of the 2020
Conference on Empirical Methods in Natural Language Processing
(EMNLP), pages 9332–9346, Online, Nov. 2020. Association for Com-
putational Linguistics. doi: 10.18653/v1/2020.emnlp-main.750. URL
https://aclanthology.org/2020.emnlp-main.750/.
[17] P. Lewis, E. Perez, A. Piktus, and et al. Retrieval-augmented generation
for knowledge-intensive nlp tasks. InAdvances in Neural Information
Processing Systems, volume 33, pages 9459–9474, 2020.
[18] N. Li, Y . Li, Y . Liu, L. Shi, K. Wang, and H. Wang. Drowzee: Metamor-
phic testing for fact-conflicting hallucination detection in large language
models, 2024. URL https://arxiv.org/abs/2405.00648.
[19] P. Liang, R. Bommasani, J. Lee, and et al. Holistic evaluation of lan-
guage models. 2023. URL https://arxiv.org/abs/2211.09110.
[20] X. Lyu, L. Zheng, Z. Wang, and et al. Trustworthy and responsible large
language models: A survey.arXiv preprint arXiv:2402.00176, 2024.
[21] N. Maleki, B. Padmanabhan, and K. Dutta. Ai hallucinations: A mis-
nomer worth clarifying, 2024. URL https://arxiv.org/abs/2401.06796.[22] P. Manakul, A. Liusie, and M. J. F. Gales. Selfcheckgpt: Zero-resource
black-box hallucination detection for generative large language models,
2023. URL https://arxiv.org/abs/2303.08896.
[23] J. Maynez, S. Narayan, B. Bohnet, and R. McDonald. On faithfulness
and factuality in abstractive summarization. In D. Jurafsky, J. Chai,
N. Schluter, and J. Tetreault, editors,Proceedings of the 58th Annual
Meeting of the Association for Computational Linguistics, pages 1906–
1919, Online, July 2020. Association for Computational Linguistics.
doi: 10.18653/v1/2020.acl-main.173. URL https://aclanthology.org/
2020.acl-main.173/.
[24] M. Nadeem, A. Bethke, and S. Reddy. StereoSet: Measuring stereo-
typical bias in pretrained language models. In C. Zong, F. Xia, W. Li,
and R. Navigli, editors,Proceedings of the 59th Annual Meeting of the
Association for Computational Linguistics and the 11th International
Joint Conference on Natural Language Processing (Volume 1: Long
Papers), Online, Aug. 2021. Association for Computational Linguis-
tics. doi: 10.18653/v1/2021.acl-long.416. URL https://aclanthology.
org/2021.acl-long.416/.
[25] H. Rashkin, V . Nikolaev, M. Lamm, L. Aroyo, M. Collins, D. Das,
S. Petrov, G. S. Tomar, I. Turc, and D. Reitter. Measuring attribu-
tion in natural language generation models.Computational Linguis-
tics, 49(4):777–840, Dec. 2023. doi: 10.1162/coli_a_00486. URL
https://aclanthology.org/2023.cl-4.2/.
[26] V . Rawte, A. Sheth, and A. Das. A survey of hallucination in large
foundation models, 2023. URL https://arxiv.org/abs/2309.05922.
[27] M. T. Ribeiro, T. Wu, C. Guestrin, and S. Singh. Beyond accu-
racy: Behavioral testing of NLP models with CheckList. In D. Ju-
rafsky, J. Chai, N. Schluter, and J. Tetreault, editors,Proceedings of
the 58th Annual Meeting of the Association for Computational Lin-
guistics, pages 4902–4912, Online, July 2020. Association for Com-
putational Linguistics. doi: 10.18653/v1/2020.acl-main.442. URL
https://aclanthology.org/2020.acl-main.442/.
[28] S. Segura, G. Fraser, A. B. Sanchez, and A. Ruiz-Cortés. A survey on
metamorphic testing.IEEE Transactions on Software Engineering, 42
(9):805–824, 2016. doi: 10.1109/TSE.2016.2532875.
[29] A. D. Selbst, D. Boyd, S. A. Friedler, S. Venkatasubramanian, and
J. Vertesi. Fairness and abstraction in sociotechnical systems. InPro-
ceedings of the Conference on Fairness, Accountability, and Trans-
parency, FAT* ’19, page 59–68, New York, NY , USA, 2019. Associ-
ation for Computing Machinery. ISBN 9781450361255. doi: 10.1145/
3287560.3287598. URL https://doi.org/10.1145/3287560.3287598.
[30] K. Singhal, S. Azizi, T. Tu, and et al. Large language models encode
clinical knowledge.Nature, 620(7973):472–480, 2023.
[31] e. a. Wang. Factuality of large language models: A survey. In Y . Al-
Onaizan, M. Bansal, and Y .-N. Chen, editors,Proceedings of the 2024
Conference on Empirical Methods in Natural Language Processing,
pages 19519–19529, Miami, Florida, USA, Nov. 2024. Association for
Computational Linguistics. doi: 10.18653/v1/2024.emnlp-main.1088.
URL https://aclanthology.org/2024.emnlp-main.1088/.
[32] B. Yang, M. A. A. Mamun, J. M. Zhang, and G. Uddin. Hallucination
detection in large language models with metamorphic relations, 2025.
URL https://arxiv.org/abs/2502.15844.
[33] T. Zhang, F. Ladhak, E. Durmus, P. Liang, K. McKeown, and T. B.
Hashimoto. Benchmarking large language models for news summa-
rization.Transactions of the Association for Computational Linguistics,
12:39–57, 2024. doi: 10.1162/tacl_a_00632. URL https://aclanthology.
org/2024.tacl-1.3/.
[34] J. Zhao, T. Wang, M. Yatskar, V . Ordonez, and K.-W. Chang. Gender
bias in coreference resolution: Evaluation and debiasing methods. In
M. Walker, H. Ji, and A. Stent, editors,Proceedings of the 2018 Con-
ference of the North American Chapter of the Association for Compu-
tational Linguistics: Human Language Technologies, Volume 2 (Short
Papers), pages 15–20, New Orleans, Louisiana, June 2018. Associa-
tion for Computational Linguistics. doi: 10.18653/v1/N18-2003. URL
https://aclanthology.org/N18-2003/.