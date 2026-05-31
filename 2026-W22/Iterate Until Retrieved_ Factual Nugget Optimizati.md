# Iterate Until Retrieved: Factual Nugget Optimization for Discoverable Continual Corrections in Agentic RAG

**Authors**: Moshe Hazoom, Gal Patel, Alon Talmor, Tom Hope

**Published**: 2026-05-25 09:45:34

**PDF URL**: [https://arxiv.org/pdf/2605.25641v1](https://arxiv.org/pdf/2605.25641v1)

## Abstract
Agentic retrieval-augmented generation (RAG) systems in complex B2B (business-to-business) settings may often receive free-form response feedback. Rather than generic feedback signals such as style, preference, or overall response quality, we focus on actionable factual corrections. We identify these instances and convert them into compact knowledge-base entries, which we call factual nuggets. We introduce Iterative Nugget Optimization (INO), an index-time optimization method that uses the production agentic RAG as a test harness: it creates an initial nugget, probes it with the triggering query and paraphrases, reflects over failed retrieval and answer traces, and revises the nugget until it is discoverable. We evaluate INO with two production B2B knowledge-assistance agents across multiple companies that use our system: a product support agent that answers questions over company-specific knowledge bases, and a support ticket agent that assists support engineers. INO consistently improves results over baselines in terms of discoverability and usage of factual corrections, in automated and human evaluations.

## Full Text


<!-- PDF content starts -->

Iterate Until Retrieved: Factual Nugget Optimization for Discoverable
Continual Corrections in Agentic RAG
Moshe Hazoom1Gal Patel1Alon Talmor1Tom Hope1,2
1Mosaic AI2The Hebrew University of Jerusalem
{moshe, gal, alon, tom.h}@getmosaic.ai
Abstract
Agentic retrieval-augmented generation (RAG)
systems in complex B2B (business-to-business)
settings may often receive free-form response
feedback. Rather than generic feedback sig-
nals such as style, preference, or overall re-
sponse quality, we focus on actionable factual
corrections. We identify these instances and
convert them into compact knowledge-base en-
tries, which we callfactual nuggets. We in-
troduceIterative Nugget Optimization(INO),
an index-time optimization method that uses
the production agentic RAG as a test harness:
it creates an initial nugget, probes it with the
triggering query and paraphrases, reflects over
failed retrieval and answer traces, and revises
the nugget until it is discoverable. We evalu-
ate INO with two production B2B knowledge-
assistance agents across multiple companies
that use our system: a product support agent
that answers questions over company-specific
knowledge bases, and a support ticket agent
that assists support engineers. INO consis-
tently improves results over baselines in terms
of discoverability and usage of factual correc-
tions, in automated and human evaluations.
1 Introduction
B2B (business-to-business) enterprise knowledge-
assistance AI agents increasingly answer com-
plex questions with retrieval-augmented gener-
ation (RAG), in which the agent’s response is
grounded by retrieved company knowledge or
domain-specific evidence (Yip et al., 2026; Chang
et al., 2026). For example, these agents sup-
port product specialists, customer-success teams,
and support engineers by grounding answers in
customer-specific documentation, prior tickets,
product behavior notes, and internal operational
knowledge, including product documentation, sup-
port knowledge bases, issue-tracking histories, and
connected enterprise repositories (Packowski et al.,
2024; Xu et al., 2024; Baqar, 2025). This assis-
Figure 1: Overview of our method and setting. Factual
corrections from users are detected and transformed into
initial “nuggets.” We iteratively refine these nuggets
using the agentic RAG system as a test harness, until
they are discoverable and optimized for future queries.
tance helps teams triage technical issues, resolve
integration or configuration problems, and draft
concrete next steps for complex cases.
Once such systems are deployed, users often
supply a continual stream of post-answer feedback:
thumbs-down events, comments on missing infor-
mation, and free-form corrections. This feedback
is not homogeneous. Many negative signals may be
vague (“not helpful”), subjective (tone or style), or
local to a single conversation. A smaller but opera-
tionally important subset states a reusable factual
correction: e.g., a product behavior, role restriction,
naming clarification, or policy detail that should be
available to future users.
This paper focuses on that subset. We first de-
tect feedback that contains an actionable factual
correction. Given this feedback, the goal is to pro-
duce a short, self-contained knowledge-base entry:
afactual nugget, that future relevant queries can
discover and use. The problem is not simply to
1arXiv:2605.25641v1  [cs.CL]  25 May 2026

summarize the feedback. The nugget must encode
the corrected fact conservatively (to avoid inserting
wrong knowledge), include enough context to be
reusable and standalone, and match future phras-
ings that may share little lexical overlap with the
original query. It should also avoid becoming so
broad that it is retrieved for unrelated traffic. A
straightforward approach that simply inherits the
wording and context of the original interaction is
often too narrow. Future users ask from differ-
ent roles, mention different product issues, or use
company-specific vocabulary.
We introduceIterative Nugget Optimization
(INO), illustrated in Figure 1. INO treats nugget
construction as a measurable index-time optimiza-
tion problem. Starting from a factual correction, it
creates a candidate nugget and a small set of likely
user queries. It then inserts the candidate into the
relevant corpus and runs the same agentic RAG
system that will serve future users. If the nugget
is not retrieved or used for the triggering query
and probe paraphrased queries, an LLM reflector
inspects the failed queries, the retrieved compet-
ing documents, and the generated answers, then
revises the nugget representation. Unlike recent
iterative prompt and context optimization meth-
ods (Agrawal et al., 2026; Zhang et al., 2026),
INO does not optimize the system prompt/context.
Prompt optimizers amortize changes across many
future tasks by changing instructions or context;
INO amortizes work across future queries about
one factual correction by changing the retrievable
artifact. This keeps the production agent configu-
ration fixed and makes the optimization local, au-
ditable, and reversible at the nugget level.
In summary, our main contributions:
1.We formulate the task of converting feed-
back on the responses of B2B AI knowledge-
assistance agents into discoverable factual
nuggets, separating actionable factual correc-
tion feedback from generic feedback.
2.We propose INO (Iterative Nugget Optimiza-
tion), an agentic-RAG-in-the-loop method
that reflects over retrieval and answer traces
to iteratively refine the indexable nugget.
3.We evaluate nugget-construction strategies on
real production data from two agents across
multiple companies served by our system: one
where inputs are free-form questions, and the
other where inputs are support tickets. Acrossall experiments, INO leads to large gains in
nugget discoverability and usage in generated
answers. INO has been launched in produc-
tion, generating hundreds of retrieved nuggets
across our customers weekly.
2 Related Work
Document expansion for retrieval.Document
expansion augments retrievable units with syn-
thetic queries or related text (Nogueira et al., 2019;
Nogueira and Lin, 2019). Weller et al. (2024) show
that expansion can fail or even hurt depending on
method, retriever, and dataset. Our setting differs
because the unit being inserted is a short feedback-
derived nugget, generated online for a specific cus-
tomer corpus, and optimized on a per-nugget basis
by iteratively executing queries on the downstream
agentic RAG and reflecting on the results.
LLM-augmented agent memory.Agent mem-
ory methods rewrite past interactions into self-
contained memories, often adding metadata such
as keywords, tags, and contextual descriptions (Xu
et al., 2025). Our setting differs in both source
and target: we start from free-form feedback that
must be filtered for actionable factual corrections,
and we store company-owned factual nuggets in
an enterprise KB rather than private episodic mem-
ories. INO also adds optimization downstream
where each nugget is tested and revised against the
unchanged production agentic RAG stack until it is
discoverable and usable for relevant future queries.
Feedback-driven RAG updates.PatchRAG
(Bang et al., 2026) studies synthetic feedback on
academic benchmarks, assuming feedback is a full
corrected answer stored in a parallel feedback mem-
ory. It does not evaluate real company data with
user feedback, focus on factual corrections, or opti-
mize feedback representation. STACKFEED (Kir-
tania et al., 2025) edits existing KB documents
using oracle or expert feedback on downstream
failures, such as compiler errors or test failures,
in software-engineering and factual-QA bench-
marks. In contrast, we use free-form end-user feed-
back from deployed B2B knowledge-assistance
agents, create new atomic factual nuggets rather
than editing customer source knowledge, and opti-
mize those nuggets for discoverability and use by
an unchanged production agentic RAG stack.
2

Figure 2: Overview of INO. User feedback is first filtered for actionable factual corrections. A correction is
converted into a structured factual nugget. The nugget is then optimized with the production agentic RAG stack in
the loop: generated test queries are replayed, retrieved competitors and generated answers are inspected, and the
nugget is revised until it becomes discoverable and usable for relevant future queries.
3 Task and Production Setting
3.1 Factual correction feedback
The deployment studied here contains two produc-
tion RAG agents used by B2B SaaS customers.
One agent answers free-form support questions
over each customer’s KB. The second agent reads
support tickets and suggests responses. See Figure
2 (1). Each customer has an independent corpus
and agent configuration. Customer corpora have
a median of approximately 450k documents and
range up to several million, so each newly inserted
nugget must compete for retrieval against a large
pool of nearby, customer-owned content.
The feedback stream is noisy and covers many
potential types of feedback aside from reusable
factualcorrections. We therefore define anaction-
able factual correctionas feedback that provides
enough concrete information to write a reusable
factual statement. An LLM-based actionability ex-
traction step filters the stream and outputs a nugget
only when this criterion is met (Figure 2 (2)). On a
5,000-event production slice, the actionable rate is
42.3% for the chat agent and 21.5% for the ticket-
transcript agent. On a separate human-labeled cali-
bration set, the classifier reaches 96.7% agreement
with human labels; details are in Appendix D.3.2 Factual nuggets
A factual nugget is a short, standalone KB docu-
ment that encodes one corrected fact plus the mini-
mal context needed for reuse. It typically contains
a title, a one-to-three sentence body, and optional
retrieval anchors. For example, a correction about
SSO reset behavior should not merely state that a
previous answer was wrong; it should encode the
reusable fact, such as which portal version and user
role the reset procedure applies to (Figure 2 (3)).
Importantly, we create new nuggets rather than
editing customer documents. First, customer KB
documents are the customer-owned source of truth,
often covering broad workflows; a feedback event
usually concerns one narrow detail inside a broader
topic. Second, local in-place edits are hard to vali-
date globally: changing one sentence can alter the
coherence of a multi-step procedure. A compact
auxiliary nugget keeps the correction atomic while
leaving source documents unchanged.
3.3 Agentic retrieval stack
All experiments use the production stack without
modification. The retrieval agent includes a query
generator and iterative retrieval based on interme-
diate results. Retrieval is hybrid, combining dense
embeddings fine-tuned on proprietary in-domain
3

data with sparse representations (e.g., BM25), fol-
lowed by a fine-tuned cross-encoder re-ranker. Re-
ranked scores are calibrated by a small MLP into
[0,1] , and documents above a confidence threshold
are passed to the generator. We use top- k= 60 be-
fore re-ranking. Across variants, the only changed
artifact is the content of the newly inserted nugget.
3.4 Evaluation
We evaluate nugget construction at three levels.
•Retrieval:The nugget passes the re-ranker and
is available to the generator.
•Citation:The nugget is cited in the final an-
swer’s sources, indicating that it was used.
•Answer feedback compliance:an LLM judge
checks whether the answer addresses the correc-
tion in the original feedback. The judge is cali-
brated against human annotators (Section 6.3).
4 Methods
Letnbe the initial factual nugget extracted from
feedback on query q0. Nugget construction pro-
duces an artifact n⋆containing the nugget text and,
for some variants, additional retrieval anchors. We
compare five strategies, including our method INO
(Figure 2 (4)). See examples in Appendix F.
A. Standard nugget.The baseline indexes the
nugget as-is: n⋆=n. This captures the common
memory-generation pattern of turning a feedback
event into a short standalone fact without retrieval
enhancement (Chhikara et al., 2025). Note that our
agentic RAG system includes query-time expan-
sion as default, so this baseline naturally includes
queries that are expanded by an LLM.
B. Trigger-query anchor.Appends a para-
phrase of the original user query q0to the nugget,
produced by an LLM.
C. Synthetic query anchors.An LLM gener-
ates five short future-user queries that the nugget
should answer, which are appended to the nugget;
a small sample was manually inspected to verify
alignment withq 0.
D. Trigger plus synthetic anchors.Adding a
paraphrase of q0and four other queries (combina-
tion ofB+C).
E. Iterative Nugget Optimization (INO).
Starts from Variant D but closes the loop with the
full agentic RAG system. It indexes the candi-
date in a customer corpus and evaluates it against
aprobe setof 4–6 queries: q0together with 3–5held-out paraphrases generated fresh for this evalu-
ation. These probe queries are deliberately disjoint
from the usage examples bundled into the nugget
(Variant D): the usage examples liveinsidethe in-
dexed document and act as retrieval anchors, while
the probe queries stayoutsideand test whether
retrieval generalizes to unseen but semantically re-
lated phrasings, i.e., the condition the nugget will
face in deployment. For each probe query, we
record whether the nugget was retrieved and cited.
When a check fails, an LLM reflector receives the
failed query, retrieved competitors, generated an-
swer, and current nugget. It revises the indexable
nugget representation: title, body wording, and
retrieval anchors, while preserving the factual cor-
rection. The revised candidate is re-indexed and
retested, for at mostT= 3iterations.
Algorithm 1Iterative Nugget Optimization
Input: nugget n, original query q0, max iterations
T
Output:indexable formn∗
1:n∗←INITIALEXPANSION(n, q 0)
2:fort= 1, . . . , Tdo
3:INDEX(n∗)
4:P←PARAPHRASE(q 0)
5:R ← {RAG(q) :q∈ {q 0} ∪P}
6:ifn∗∈rfor anyr∈ Rthen
7:returnn∗▷converged
8:end if
9:n∗←REWRITE(n∗,R, q 0, P)
10:end for
11:returnn∗
The reflector is instructed not to add new factual
claims beyond the correction and provided context.
Its role is to make the correction more discover-
able: e.g., add missing terminology, disambiguate
roles or products, and distinguish the nugget from
documents that incorrectly outranked it.
5 Experimental Setup
5.1 Chat-agent correction set
The main evaluation uses 100 factual nuggets sam-
pled from real negative-feedback events that passed
the actionability filter. We sample roughly uni-
formly across seven customers. The nuggets span
products, user roles, and topics within each cus-
tomer’s corpus. For each construction strategy, we
generate n⋆for all 100 nuggets, index them into
a frozen copy of the corresponding customer cor-
4

pus, replay queries through the same agent version
(prompt, hyper-parameters, tools and configura-
tion), and then delete the inserted nuggets before
the next variant. Each variant is run three times be-
cause LLM-generated anchors vary between runs.
We note that for INO, 56% of nuggets pass after
the first iteration, 32% after the second, and 12%
require the third; no evaluated nugget exhausts the
budget without passing the retrieval check.
5.2 Held-out queries
We evaluate generalization on 410 held-out queries
constructed by two complementary pipelines. The
first mines real production queries: for each cus-
tomer, we build a per-customer FAISS index over
five months of production traffic, retrieve nearest
neighbors for each triggering query, and keep high-
scoring candidates after cross-encoder re-ranking.
This yields 110 real historical paraphrases. The
second pipeline generates three paraphrases per
triggering query with a different LLM from the
one used for nugget construction, grounded in do-
main vocabulary retrieved from the customer cor-
pus. This yields 300 synthetic paraphrases. See
Appendix B for implementation details.
5.3 Support-Ticket agent transfer
We extend our evaluation to a production agent that
consumes full support ticket transcripts rather than
user questions, retrieves relevant documents, and
produces a summary plus suggested next steps. We
collect 160 ticket-level samples from real negative-
feedback events that passed the actionability filter
and apply the same methods.
5.4 Negative control
We index optimized nuggets and test 200 randomly
sampled real queries per customer (1,400 total). We
count how often nuggets are retrieved, and manu-
ally inspect each retrieval.
6 Results
Table 1 reports performance on held-out queries,
while Table 2 shows results on the original queries
(in-sample). Every variant improves over the stan-
dard nugget. The ordering is monotonic: trigger
anchors help, synthetic anchors help more, com-
bining them helps further, and INO performs best.
The ranking is preserved on both retrieval and cita-
tion. INO improves retrieval and citation by 25-29
points over the standard nugget baseline.Because 300 held-out queries are synthetic, we
also report the real-query and synthetic-query sub-
sets separately (Table 1). The same ordering holds
on the 110 real production queries alone. Real
queries are uniformly harder: performance is up
to 25 points lower than on LLM-generated para-
phrases depending on the metric and variant. This
makes the historical subset the more conservative
estimate of field performance. Taken together, INO
is not merely memorizing the original query, but
generalizing across queries.
6.1 Transfer to support tickets
Table 3 shows results on the support ticket agent.
This setting is harder because the input is a multi-
turn transcript rather than a clean question. The
baseline retrieves the relevant nugget only 35.1% of
the time. The same ranking nevertheless holds, and
INO improves retrieval by 43.1 points and citation
by 40.7 points over the baseline.
The larger gains on tickets are consistent with
the harder input distribution: transcripts contain
filler, multiple issues, and vocabulary that differs
from the original correction event. In this regime,
per-nugget optimization has more room to help.
6.2 Negative-control traffic
Of 1,400 random queries, 20 (1.4%) retrieved a
newly inserted nugget and 12 (0.8%) cited one.
Manual inspection found no false positives. Each
of the 20 retrievals was a semantically equivalent
paraphrase of the original correction topic that hap-
pened to appear in the random sample. The implica-
tion is that the 1.4% is not noise but generalization.
6.3 Answer-level evaluation
Retrieval and citation do not guarantee that the gen-
erated answer addresses the original correction. We
therefore run an LLM-as-judge evaluation (Zheng
et al., 2023) on held-out chat-agent answers for
the baseline and INO. The judge sees the original
question, original answer, user feedback, extracted
nugget, retrieved context, and the answer. It la-
bels four axes: correction compliance (does the
answer materially reflect the correction), faithful-
ness to the nugget (does the answer adhere to the
facts in the nugget), non-regression (is the new an-
swer at least as good as the original without the
nugget), and groundedness (does the answer make
use of retrieved documents). See full metric defi-
nitions in Appendix C. The rubric is anchored to
the user’s correction because no gold answer exists
5

Historical queries (n=110) LLM paraphrases (n=300)
Variant Strategy Retrieval Citation Retrieval Citation
A Standard nugget 52.3±0.5 41.3±1.6 64.1±1.2 57.3±1.1
B Trigger-query anchor 67.0±2.0 50.7±1.5 76.1±0.6 68.5±2.3
C Synthetic anchors 67.8±1.3 52.0±1.4 86.4±0.7 77.0±0.9
D Trigger + synthetic 72.9±2.2 58.9±2.2 89.4±0.3 81.5±0.9
E INO77.3±1.0 68.1±1.8 93.7±0.8 86.1±1.1
Table 1: Held-out results split by query source. INO leads to large gains in both retrieval and citation.
Figure 3: Answer-level judge results on held-out queries for the standard nugget baseline and INO. INO increases
the share of answers that address the requested correction (73.4% vs. 52.2%; McNemar p≈0.003 ), and improves
nugget availability without increasing major regressions or hallucinations. If the nugget is absent, the label isN/A.
Variant Retrieval Citation
A. Standard nugget 67.6±1.2 60.2±1.0
B. Trigger-query anchor 79.1±1.9 71.8±2.3
C. Synthetic anchors 88.0±1.2 79.5±1.1
D. Trigger + synthetic 90.8±1.1 81.9±0.6
E. INO97.0±1.0 89.1±1.6
Table 2: In-sample evaluation on 100 triggering queries.
Mean±standard deviation over three runs.
Variant Retrieval Citation
A. Standard nugget 35.1±0.9 29.7±1.5
B. Trigger-query anchor 42.0±1.9 37.8±0.9
C. Synthetic anchors 48.8±2.0 44.2±1.3
D. Trigger + synthetic 63.3±2.0 58.5±2.5
E. INO78.2±1.8 70.4±0.8
Table 3: Support ticket agent evaluation on 160 samples.
Mean±standard deviation over three runs.
for these production cases. We calibrate the judge
against human annotators. A random sample of
100 judge decisions was independently labeled by
three in-house annotators. Fleiss’ κamong anno-
tators exceeds 0.75 on every axis. Using majority
vote as the adjudicated label, Cohen’s κbetween
the judge and the adjudicated label exceeds 0.80
on every axis after a small prompt-calibration pass.
Figure 3 summarizes results. Two effects drive
the gains: INO retrieves the nugget more often,
and when it does, the generator uses it more faith-fully. On correction compliance, INO addresses
the correction in 73.4% of held-out cases versus
52.2% for the baseline, with the share of queries
where the nugget was not retrieved ("misses") drop-
ping from 40% to 13%. Conditional on the nugget
being delivered, INO uses it faithfully in 81% of
cases versus 54.1% for the baseline. Downstream,
major regression errors decrease from 15.6% to
11.6%, and hallucinated groundedness labels de-
crease from 19.1% to 14.8%.
7 Conclusion
We studied how to convert actionable factual cor-
rection feedback into factual nuggets that are dis-
coverable and useful in production RAG systems.
Our proposed Iterative Nugget Optimization (INO)
loop uses the agentic RAG stack itself to test and
refine each nugget before it is added to the retriev-
able corpus, while keeping the production agent
configuration fixed. Across seven B2B customer
deployments of two different agents, INO substan-
tially improves discoverability and usage of factual
corrections compared to baselines. The optimiza-
tion is offline, with a per-correction cost of about
$0.31 and under two minutes per nugget in our
setup, and adds no query latency after indexing,
making it practical to run continuously. Our ap-
proach is currently running in production, serving
hundreds of nuggets every week.
6

Limitations
One retrieval architecture.All main experi-
ments use our hybrid agentic RAG. Pure dense,
pure sparse, or late-interaction retrievers may re-
spond differently to retrieval anchors and iterative
nugget refinement.
English-only evaluation.All evaluated corpora
and queries are in English. The method itself is
language-agnostic given multilingual retrieval and
generation components, but the empirical claims
here are English-only.
LLM dependence.Nugget extraction, anchor
generation, reflection, and answer judging all use
frontier LLMs. Different models may produce dif-
ferent anchors or judge labels. We did not run a full
model ablation; we report the results of our actual
deployed system.
References
Lakshya A Agrawal, Shangyin Tan, Dilara Soylu,
Noah Ziems, Rishi Khare, Krista Opsahl-Ong, Arnav
Singhvi, Herumb Shandilya, Michael J Ryan, Meng
Jiang, Christopher Potts, Koushik Sen, Alexandros G.
Dimakis, Ion Stoica, Dan Klein, Matei Zaharia, and
Omar Khattab. 2026. Gepa: Reflective prompt evo-
lution can outperform reinforcement learning. In
ICLR.
Mohammad Kalim Akram, Saba Sturua, Nastia
Havriushenko, Quentin Herreros, Michael Günther,
Maximilian Werk, and Han Xiao. 2026. jina-
embeddings-v5-text: Task-targeted embedding distil-
lation.Preprint, arXiv:2602.15547.
Jihwan Bang, Seunghan Yang, Kyuhong Shim, Simyung
Chang, Juntae Lee, and Sungha Choi. 2026. Feed-
back adaptation for retrieval-augmented generation.
Preprint, arXiv:2604.06647.
Mohammad Baqar. 2025. Rag4tickets: Ai-powered
ticket resolution via retrieval-augmented genera-
tion on jira and github data.arXiv preprint
arXiv:2510.08667.
Jonathan D Chang, Andrew Drozdov, Shubham Tosh-
niwal, Owen Oertell, Alexander Trott, Jacob Portes,
Abhay Gupta, Pallavi Koppol, Ashutosh Baheti, Sean
Kulinski, and 1 others. 2026. Karl: Knowledge
agents via reinforcement learning.arXiv preprint
arXiv:2603.05218.
Jianlv Chen, Shitao Xiao, Peitian Zhang, Kun Luo, Defu
Lian, and Zheng Liu. 2024. Bge m3-embedding:
Multi-lingual, multi-functionality, multi-granularity
text embeddings through self-knowledge distillation.
Preprint, arXiv:2402.03216.Prateek Chhikara, Dev Khant, Saket Aryan, Taranjeet
Singh, and Deshraj Yadav. 2025. Mem0: Building
production-ready ai agents with scalable long-term
memory.Preprint, arXiv:2504.19413.
Shashank Kirtania, Naman Gupta, Priyanshu Gupta,
Sumit Gulwani, Arun Iyer, Suresh Parthasarathy Iyen-
gar, Arjun Radhakrishna, Sriram K. Rajamani, and
Gustavo Soares. 2025. STACKFEED: Structured tex-
tual actor-critic knowledge base editing with FEED-
back. InProceedings of the 2025 Conference on
Empirical Methods in Natural Language Processing:
Industry Track, pages 2588–2606, Suzhou (China).
Association for Computational Linguistics.
Chaofan Li, Zheng Liu, Shitao Xiao, and Yingxia Shao.
2023. Making large language models a better founda-
tion for dense retrieval.Preprint, arXiv:2312.15503.
Rodrigo Nogueira and Jimmy Lin. 2019. From
doc2query to doctttttquery.
Rodrigo Nogueira, Wei Yang, Jimmy Lin, and
Kyunghyun Cho. 2019. Document expansion by
query prediction.arXiv preprint arXiv:1904.08375.
Sarah Packowski, Inge Halilovic, Jenifer Schlotfeldt,
and Trish Smith. 2024. Optimizing and evaluating en-
terprise retrieval-augmented generation (rag): A con-
tent design perspective.Preprint, arXiv:2410.12812.
Orion Weller, Kyle Lo, David Wadden, Dawn Lawrie,
Benjamin Van Durme, Arman Cohan, and Luca Sol-
daini. 2024. When do generative query and docu-
ment expansions fail? a comprehensive study across
methods, retrievers, and datasets. InFindings of the
Association for Computational Linguistics: EACL
2024, pages 1987–2003, St. Julian’s, Malta. Associa-
tion for Computational Linguistics.
Wujiang Xu, Zujie Liang, Kai Mei, Hang Gao, Juntao
Tan, and Yongfeng Zhang. 2025. A-mem: Agentic
memory for llm agents.Preprint, arXiv:2502.12110.
Zhentao Xu, Mark Jerome Cruz, Matthew Guevara,
Tie Wang, Manasi Deshpande, Xiaofeng Wang, and
Zheng Li. 2024. Retrieval-augmented generation
with knowledge graphs for customer service question
answering. InProceedings of the 47th International
ACM SIGIR Conference on Research and Develop-
ment in Information Retrieval, SIGIR 2024, page
2905–2909. ACM.
Jason Yip, Nikhil Gupta, and Marcin Wojtyczka. 2026.
Databricks as an agentic platform: Orchestration,
context, and control. InDatabricks Data Intelligence
Platform: Powering the Agentic Era, pages 125–152.
Springer.
Qizheng Zhang, Changran Hu, Shubhangi Upasani,
Boyuan Ma, Fenglu Hong, Vamsidhar Kamanuru, Jay
Rainton, Chen Wu, Mengmeng Ji, Hanchen Li, and 1
others. 2026. Agentic context engineering: Evolving
contexts for self-improving language models.ICLR.
7

Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan
Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin,
Zhuohan Li, Dacheng Li, Eric P. Xing, Hao Zhang,
Joseph E. Gonzalez, and Ion Stoica. 2023. Judging
llm-as-a-judge with mt-bench and chatbot arena. In
Proceedings of the 37th International Conference on
Neural Information Processing Systems, NIPS ’23,
Red Hook, NY , USA. Curran Associates Inc.
8

A Prompt and Model Details
The production prompts include customer-specific examples and are not reproduced verbatim. This
appendix describes the prompt structure used in the experiments.
Actionability and nugget extraction.Input fields are conversation history (in case of a multi-turn), the
final user question, the answer that received feedback, feedback type (either thumbs-up or thumbs-down),
free-text feedback, and cited documents for the current turn. The model emits JSON with (i) a usefulness
label, (ii) a boolean indicating whether the feedback can produce a factual nugget, (iii) a short reason, and
(iv) when applicable, a title and body for the nugget. The prompt emphasizes that nuggets must contain
reusable factual knowledge explicitly supported by the feedback, not a critique of the answer.
Synthetic anchors.For Variant C, the model receives the nugget and writes five concise queries that
future users might ask whose answer is the nugget. The prompt asks for diverse surface forms and
discourages copying nugget wording.
Trigger plus synthetic anchors.For Variant D, the model receives the nugget and triggering query q0.
It writes five anchors: the first is a non-verbatim paraphrase of q0, and the other four cover alternative
phrasings or nearby aspects of the same topic.
INO reflection.For INO, the reflector receives the current nugget, triggering query, probe queries,
top retrieved documents for failed checks, whether the nugget was retrieved and cited, and the generated
answer. It returns a revised indexable nugget representation. The prompt requires factual preservation: it
may add terminology, clarify scope, or rewrite anchors, but it may not add new facts beyond the correction
and supplied context.
B Held-out Set Construction
For real-query mining, we collected five months of production queries per customer and embedded them
with a Jina-v5-small dense encoder (Akram et al., 2026). We built per-customer FAISS indexes and
retrieved the top 50 nearest neighbors for each triggering query, keeping candidates with cosine similarity
above 0.75. We then re-ranked with bge-reranker-v2-m3 (Li et al., 2023; Chen et al., 2024) and kept
candidates above 0.85. This produced 110 historical paraphrases.
For synthetic paraphrases, we first ran an agentic RAG step against the customer corpus to collect
domain vocabulary and relevant terminology. A second LLM, different from the one used for nugget
construction, generated three realistic paraphrases per triggering query while using that vocabulary. We
used OpenAI’s GPT-5.4 withhighreasoning for this task. A random sample was manually spot-checked
for topical consistency.
C Judge Rubric
The judge receives the original user question, the original answer, user feedback, the extracted nugget,
retrieved context, and the answer. It emits JSON for four independent axes.
Correction compliance.Labels are:addresses(the answer materially reflects the correction),partial
(mentions the corrective fact but incompletely),misses(does not address the correction), andcontradicts
(repeats or entails the corrected mistake). If the nugget is absent, the label ismisses.
Faithfulness to the nugget.If the nugget is in retrieved context, the judge labels the use asfaithful,
partial, orunfaithful. If the nugget is absent, the label isN/A.
Non-regression.Labels arepreserved,minor regression, ormajor regression, comparing correct content
in the original answer against the answer.
Groundedness.Labels aregrounded,minor issues, orhallucinated, based only on the retrieved context
shown to the judge.
9

D Actionability Classifier
The actionability filter is an LLM, specifically OpenAI’s GPT-5.2 withmediumreasoning effort, used
with a small set of in-context examples. It performs classification and extraction in one pass. The output
JSON contains a feedback_usefulness field, a kb_candidate boolean, and an article object. We use
a single pass to avoid disagreement between a classifier and a separate extractor and to reduce inference
cost.
Human calibration used 200 production feedback events sampled across the seven customers. Two
annotators labeled each event for whether it contained enough information to produce a reusable factual
nugget; disagreements were adjudicated by a third annotator. The classifier agreed with the adjudicated
label in 96.7% of cases.
E Costs
In our implementation, Variants A–D cost about $0.012–$0.0124 per nugget and complete in 12.9 seconds
on average, dominated by extraction. INO adds roughly $0.30 and 98.3 seconds per nugget, bringing total
cost to about $0.31 and latency to 111.2 seconds. This cost is offline: it occurs once per factual correction
and adds no per-query latency after indexing. When this budget is acceptable, INO is the best-performing
configuration across both agents. When cost is constrained, Variant D is a strong alternative, reaching
84.8% / 75.1% on held-out chat queries at a fraction of the cost.
F Example Nuggets Across the Five Variants
To illustrate how the five construction strategies differ in practice, we show one factual correction realized
asn⋆under each variant. The example is sanitized and lightly paraphrased to avoid customer-identifying
terminology; proportions and structure match real production cases.
Source feedback event.Original user question q0:“Why can’t my analyst reset another teammate’s
password from the admin panel?”The agent answered that any user with admin-panel access can reset
passwords. The user marked the answer wrong and commented that in portal v8.2 password reset is
restricted to theWorkspace Ownerrole; the Analyst role can view but not reset.
Extracted nugget n. Title:Password reset role restriction in portal v8.2.Body:In portal v8.2, only
users with the Workspace Owner role can reset another user’s password from the admin panel. Users with
the Analyst role can view team members but cannot trigger a reset.
Variant A. Standard nugget.n⋆=n. No anchors, no expansion.
Variant B. Trigger-query anchor.n⋆=nwith one appended anchor:
• Analyst role cannot reset another user’s password in the admin panel
Variant C. Synthetic query anchors. n⋆=n with five LLM-generated anchors covering plausible
future phrasings (no constraint to matchq 0):
• How do I reset a user’s password in the admin panel?
• Which role is required to reset passwords?
• Can analysts manage other users’ credentials?
• Permissions for password reset in v8.2.
• Workspace Owner vs Analyst capabilities.
Variant D. Trigger plus synthetic anchors. n⋆=n with five anchors, the first constrained to be a
paraphrase ofq 0:
• Analyst role cannot reset another user’s password in the admin panel
• Which role is required to reset passwords in v8.2?
10

• Can analysts manage other users’ credentials?
• Permissions needed to trigger a password reset.
• Workspace Owner vs Analyst capabilities.
Variant E. INO.Starting from Variant D, the agent fails the probe queries“user reset isn’t working for
my team lead”because a generic “Resetting your own password” article outranks the nugget on terms like
resetandisn’t working. The reflector inspects the failed query and the competing document, then revises
the indexable representation:
Revised title:Admin-panel password reset for other users is restricted to Workspace Owner (portal v8.2).
Revised body:In portal v8.2, resettinganother user’spassword from the admin panel requires the
Workspace Owner role. Users with the Analyst role can view team members in the admin panel but the
reset action is hidden. This restriction does not apply to a user resetting their own password from the login
screen.
Revised anchors:
• Why can’t an analyst reset a teammate’s password from the admin panel?
• Admin panel reset action greyed out for analyst role.
• Which role can reset passwords for other users in v8.2?
• Workspace Owner permissions for password reset.
• Reset other user’s password vs reset my own password.
The revision adds the disambiguating phraseanother user’s(the wording that distinguished the nugget
from the self-reset competitor), surfaces the UI symptomgreyed out, and explicitly contrasts the two reset
paths. The nugget passes all stress-test queries on the next iteration.
11