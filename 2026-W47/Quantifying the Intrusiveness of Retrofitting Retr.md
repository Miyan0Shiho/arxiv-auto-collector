# Quantifying the Intrusiveness of Retrofitting Retrieval-Augmented Generation onto Legacy Software Systems

**Authors**: K. M. Tahlil Mahfuz Faruk, Rakib Abdullah

**Published**: 2026-11-18 06:22:21

**PDF URL**: [https://arxiv.org/pdf/2611.23931v1](https://arxiv.org/pdf/2611.23931v1)

## Abstract
Enterprises increasingly add retrieval-augmented generation (RAG) to existing REST and monolithic systems, usually by writing an ad hoc HTTP call into a handler, and have no way to quantify how intrusive that choice is. We present (1) a taxonomy of three retrofit patterns, sidecar, SDK-embed and naive-invasive; (2) the Legacy Intrusion Score (LIS), three code sub-metrics (churn, breadth, coupling), and an extension, LIS-X, with three operational sub-metrics measured on a live system (run-time overhead, fault propagation and silent feature degradation); (3) a static evaluation on five legacy systems, two of them real open-source codebases fetched from their repositories at pinned commits; and (4) a live case study that deploys all three patterns as separate OS processes. The code-level ranking sidecar < SDK-embed < naive-invasive holds in all five systems and under 2,000 random weightings, but it follows from component-wise dominance of the sub-metrics, so it validates the metric rather than testing a hypothesis. In the live study, outcomes are not fixed by definition. The sidecar adds no detectable latency to the legacy endpoint (0.07 ms, 95% CI [-0.10, 0.24]), whereas SDK-embed and naive-invasive add 569 and 475 ms and are not distinguishable from each other (p=0.62). Both pass a 2 s dependency delay on to legacy requests, and when the RAG layer fails fast they keep returning successful responses that no longer contain the AI field. Load on the RAG layer slows the legacy service by 1.51-1.57x on a shared host, and sidecar staleness averages 911 ms with a 2 s polling interval. Retrieval hit@1 is 0.95; three hosted LLMs answer all 12 test questions correctly and a local model 9/12, and the four backends are swapped by changing one request field. All dynamic results come from one host and one system.

## Full Text


<!-- PDF content starts -->

2026 IEEE 8th International Conference on Sustainable Technologies
for Industry 5.0 (STI 2026), 03-04 December, Dhaka
Quantifying the Intrusiveness of Retrofitting
Retrieval-Augmented Generation onto Legacy
Software Systems
K.M.Tahlil Mahfuz Faruk1,Rakib Abdullah2
1Department of Software Engineering, Green University of Bangladesh
2Department of AI and Data Science, Green University of Bangladesh
Email:1tahlil@swe.green.edu.bd,2rakib@ads.green.edu.bd
Abstract—Enterprises increasingly add retrieval-augmented
generation (RAG) to existing REST and monolithic systems,
usually by writing an ad hoc HTTP call into a handler, and
have no way to quantify how intrusive that choice is. We present
(1) a taxonomy of three retrofit patterns,sidecar,SDK-embed
andnaive-invasive; (2) the Legacy Intrusion Score (LIS), three
code sub-metrics (churn, breadth, coupling), and an extension,
LIS-X, with three operational sub-metrics measured on a live
system (run-time overhead, fault propagation and silent feature
degradation); (3) a static evaluation on five legacy systems,
two of them real open-source codebases fetched from their
repositories at pinned commits; and (4) a live case study that
deploys all three patterns as separate OS processes. The code-
level ranking sidecar<SDK-embed<naive-invasive holds in
all five systems and under 2,000 random weightings, but it
follows from component-wise dominance of the sub-metrics, so
it validates the metric rather than testing a hypothesis. In the
live study, outcomes are not fixed by definition. The sidecar adds
no detectable latency to the legacy endpoint (0.07 ms, 95% CI
[−0.10,0.24]), whereas SDK-embed and naive-invasive add 569
and 475 ms and are not distinguishable from each other (p=0.62).
Both pass a 2 s dependency delay on to legacy requests, and when
the RAG layer fails fast they keep returning successful responses
that no longer contain the AI field. Load on the RAG layer
slows the legacy service by 1.51–1.57×on a shared host, and
sidecar staleness averages 911 ms with a 2 s polling interval.
Retrieval hit@1 is 0.95; three hosted LLMs answer all 12 test
questions correctly and a local model 9/12, and the four backends
are swapped by changing one request field. All dynamic results
come from one host and one system.
Index Terms—Legacy System, RAG, Software Architecture,
Code Churn, Empirical SE, Sidecar Pattern, Anti-corruption
Layer, Fault Injection
I. INTRODUCTION
Retrieval-augmented generation (RAG) [1], [2] has become
the default way to add large-language-model (LLM) features to
software that was never designed for them, and such software
is almost by definition legacy: a working REST API or service-
layered application that is costly to destabilize [3], [4]. The
practical question is how to add RAG, and the options span a
wide intrusiveness spectrum, from a co-located helper process
that never touches legacy source to a single HTTP call written
into the nearest handler. The architecture-erosion literature
argues qualitatively that uncontrolled ad hoc integration erodes
a system’s design [5], and work on engineering RAG systems
catalogues how RAG systems fail [6]. We found no work that
measures how intrusive a RAGretrofitis, or whether a patternadvertised as non-intrusive stays so under load and failure.
This matters for Industry 5.0, which asks for sustainable
and resilient industry [7]: adding AI to long-lived industrial
software without replacing or destabilizing it extends the life
of existing systems, but only if an AI fault cannot propagate
into the operations that depend on them.
This paper makes four contributions. (1) Ataxonomyof
three retrofit patterns: sidecar (no legacy-code touch) [8]–[10],
SDK-embed (a thin client imported into legacy code, acting
as an anti-corruption layer [11], [12]), and naive-invasive (an
inline HTTP call). (2) TheLegacy Intrusion Score(LIS), with
three code sub-metrics drawing on the churn and coupling
literature [13], [14], andLIS-X, which adds three operational
sub-metrics (run-time overhead, fault propagation, silent fea-
ture degradation). (3) Astatic evaluationon five systems, two
of them real codebases downloaded from GitHub. (4) Alive
case studythat deploys all three patterns side by side, each
service as its own OS process, and measures latency, staleness,
isolation, overhead, fault propagation, retrieval quality and the
swap of four LLM backends.
We ask seven research questions.RQ1Does LIS rank the
patterns as their definitions predict, and by how much?RQ2
Does the ranking depend on the weights, or follow from the
sub-metrics themselves?RQ3What latency does the sidecar’s
RAG path add?RQ4How stale is data behind the sidecar’s
polling adapter?RQ5Does load on the RAG layer slow
the legacy system on a shared host?RQ6With all three
patterns deployed, what does each cost the legacy request path,
and how far does a RAG-layer fault propagate?RQ7Is the
pipeline good enough, in retrieval and answer quality, for these
measurements to describe a working system, and can the LLM
backend be swapped without touching other components?
II. RELATEDWORK
Legacy systems and erosion.Bisbal et al. survey the issues
and migration directions for legacy information systems [3].
Khadka et al. interviewed 26 practitioners and validated the
findings with a survey of 198 respondents: practitioners value
their legacy systems highly and the challenges they face are
not only technical [4]. Van Gurp and Bosch analyse the causes
of design erosion [5].
Integration patterns.Fowler’s Strangler Fig grows a new
system around the edges of the old [10]. Evans’ anti-corruption
979-8-3195-0323-7/26/$31.00 ©2026 IEEE
arXiv:2611.23931v1  [cs.SE]  18 Nov 2026

layer translates between a new subsystem and a legacy
one [11]; the Azure Architecture Center describes it as a facade
that adds latency and can become a single point of failure, so
retry, circuit breaker and bulkhead should be planned [12].
The sidecar deploys helper components as a separate process
or container [8], [9]. Our adapter’s timestamp-cursor polling is
change-data-capture-style synchronization [15], not log-based
capture.
Change metrics.Nagappan and Ball showed that relative
churn measures, which normalize churned and deleted lines
and the number of files churned by component size, discrim-
inate fault-prone from non-fault-prone binaries in Windows
Server 2003 with 89.0% accuracy [13]; our churn and breadth
follow that lineage, and coupling draws on the Chidamber–
Kemerer suite [14].
RAG and resilience.RAG [1], [2] and the engineering
study of RAG failure points [6] concern building RAG sys-
tems, not retrofitting them. For run-time behaviour we take
timeouts, circuit breakers and bulkheads [16] as the yard-
stick for what a retrofit should contain, test it with chaos-
engineering-style fault injection [17], and follow case-study
guidance for the statistics [18], [19].
III. PATTERNS AND THELEGACYINTRUSIONSCORE
A. Retrofit Patterns
Sidecar.A co-located, independently deployed process polls
the legacy system’s existing, unmodified read API and keeps
its own index; no legacy file changes.SDK-embed.Legacy
code imports a thin client (rag_client) that lives in a new
file; the legacy handler calls one function.Naive-invasive.
An ad hoc HTTP call is written directly inside a handler,
repository method or hook. For every system we build two
naive variants at different layers; they are manually constructed
by one researcher, so they capture variance in placement, not
between engineers.
B. Code-level LIS
LetLbe the set of a system’s original files. For filef
letC f=|A f|+|D f|+|M f|count the added, deleted
and modified non-blank lines of a variant (a replaced line
counts once, as modified), andT f=|I′
f\If|+|E′
f\Ef|
the new dependency points, whereI f, Efare the sets of
imported modules and network-endpoint string literals found
with Python’sastmodule before the change andI′
f, E′
fafter
it.
churn=P
f∈LCf
LOC(L),(1)
breadth=|{f∈L:C f>0}|
|L|,(2)
coupling=P
f∈LTf
|L|,(3)
LIS=w cchurn+w bbreadth+w kκ,(4)
whereκ= min(coupling,1)is the clipped coupling and the
default weights are(w c, wb, wk) = (0.5,0.3,0.2). New files
(the SDK client, the sidecar adapter) are reported but excluded
from churn and breadth; a legacy file’s import of themiscoupling. Because the sidecar is also scored by this diff (its
variant contains only a new file), its LIS of zero is measured
as “no legacy file touched”, not assigned.
C. LIS-X: operational sub-metrics
LIS ignores how a retrofit behaves at run time. LIS-X adds
three sub-metrics, each in[0,1], measured on the live system
of Section IV. Let ¯tctrland¯tpatbe the mean latency of the
legacy endpoint in the unmodified control and under a pattern.
For each injected fault modeminM={error,delay}, letv m
be the share of legacy requests that fail or exceed the SLO
max(50ms,3×control p95), ands mthe share of responses
that succeed (HTTP status below 400) but lack the AI field;
s0is the same share without a fault.
runtime= 1− ¯tctrl/¯tpat,(5)
fault=1
|M|X
m∈Mvm,(6)
degradation=1
|M|X
m∈Mmax (0, s m−s0),(7)
LIS-X=6X
i=1uixi,(8)
wherex 1, . . . , x 6are churn, breadth,κ, runtime,
fault and degradation, and the weights are
u= (0.25,0.15,0.10,0.20,0.15,0.15). The runtime term
counts only if the bootstrap CI of the latency difference to
the control excludes zero.Faultrecords loud damage (failure
or slowness) anddegradationsilent damage: under a fast
failure of the RAG layer, a retrofit that swallows errors scores
0 on fault and 1 on degradation; under a delay it is the other
way round. For the control and the sidecar, degradation is
0 by construction, because their legacy endpoint carries no
AI field that could disappear; the sidecar’s own feature fails
visibly, which Table II reports. LIS-X exists only where a
live deployment exists (System A).
How to read RQ1 and RQ2.The sidecar touches no legacy
file, so its code-level LIS is zero by definition, and one import
plus one call is smaller than an inline call by construction.
RQ1/RQ2 therefore check that LISdiscriminatesthe patterns
and how large the gaps are (construct validity); the falsifiable
questions are RQ3–RQ7.
IV. EXPERIMENTALSETUP
A. Corpus and Variants
Table I lists the corpus sizes. Systems A–C are synthetic
Python/Flask applications written for the study as multiple
files so that breadth is meaningful: a single-layer REST service
over SQLite (A), a repository/service-layered ticket system (B)
and an ingest/report batch system (C); B and C place one
naive variant in a lower layer (repository, ingestion path), a
layering violation. Systems D and E are real, independently
authored codebases that the notebookdownloadsfrom GitHub
at pinned commits (49943b5 and 4b95fb2) and measures
verbatim: Flask’s tutorial applicationflaskr[20] (BSD-3-
Clause; app factory, blueprints, sessions), whose naive variants
sit in thebefore_app_requesthook that runs on every

5 legacy systems
A–E (3 synthetic
+ 2 real, credited)3 retrofit patterns
sidecar / SDK-
embed /
naive-invasive×2Diff-based
measurement:
churn, breadth,
couplingComposite LIS
+ 5 weight vectors
(sensitivity)RQ1 ranking
check
RQ2 sensitivity
(Table I, Fig. 2)Static Study (RQ1–RQ2)
Legacy
monolith
:5001ACL / Adapter
CDC poll, 2sRetrieval
MiniLM
+ FAISS
:5002Orchestrator
Groq pool
+ local
FLAN-T5 :5003Gateway
:5000RQ3 latency
(M-W U)
RQ4 staleness
RQ5 isolation +
cross-modelDynamic Case Study on System A (RQ3–RQ5)
Fig. 1: Research pipeline.Top:the static study diffs four variants of each of five legacy systems and computes LIS.Bottom:
the dynamic study deploys System A four ways (control, sidecar, SDK-embed, naive-invasive) next to the adapter, retrieval,
orchestrator and gateway; every box is a separate OS process.
request and in the blog index view, and the RealWorld [21]
Flask backend [22] (MIT; app factory, dependency injection,
JWT, ORM mixin), whose variants sit in a view and in
CRUDMixin.save(), which every model calls on every
write. System E is a three-file excerpt of a larger repository,
so its LIS denominators depend on that choice. Variants of D
and E are anchor-checked patches over the downloaded text; if
upstream changes so that an anchor stops matching, the build
fails.
B. Dynamic Deployment
The live study runs System A only, as eight OS processes
started withsubprocesson one Kaggle host (4 vCPU,
Python 3.13.18) (Fig. 1). The four legacy deployments are
written from the exact files that the static diff measured,
and an assertion checks that recomputing LIS from them
reproduces the static numbers. They share one SQLite file.
The adapter polls the sidecar deployment’s unmodifiedGET
/api/articleson anupdated_atcursor every 2 s
and posts documents to the retrieval service, which embeds
them with all-MiniLM-L6-v2 [23] and indexes them in a
FAISS flat inner-product index [24]. The orchestrator is the
only component that talks to an LLM; the model is a re-
quest field. It serves three models hosted on Groq [25],
selected at run time from the live catalogue (gpt-oss-120b
andgpt-oss-20b[26], Qwen3.8-27B [27]), and the lo-
calflan-t5-small[28], [29] through Hugging Face
transformers[30];gpt-oss-120bis the default for
RQ3 and RQ6. It also hosts a fault-injection switch (error or
added delay). The knowledge base has 20 articles; the quality
sets are 40 retrieval queries (two phrasings per article) and 12
questions whose answers contain one checkable fact.
C. Protocol and Statistics
All comparisons use two-sided Mann–WhitneyU[31],
rank-biserial effect size [32] and percentile-bootstrap 95%
CIs (2,000 resamples) [33]. RQ3: 50 requests per path after
5 discarded warm-up requests, interleaved in random order.
Every request in RQ3–RQ6 is checked for HTTP success
(status below 400); latency statistics use successful requests
only, and failures would be counted and reported. RQ4: 40
probes, each at a random phase of the poll cycle (back-to-
back probes would be phase-locked to it). RQ5: 100 probes
per condition, idle→load→idle, with 3 load threadson the CPU-bound local model, probing the legacy service
directly and through the gateway. RQ6: 40 requests per path
(interleaved) and 10 per path and fault mode (error; 2 s delay).
RQ7: hit@kand MRR against a TF-IDF baseline [34], [35];
answers are scored on the one gold fact after number words
are converted to digits (strict: the value with its unit within 60
characters, or a bare short reply;lenient: the value anywhere),
and every answer that is not strict-correct is reviewed by hand.
Backend swap: 3 questions×3 repeats per backend through
the gateway, changing only the request’s model field.
V. RESULTS
A. RQ1: Metric Discrimination
Table I reports the results. The sidecar’s measured LIS
is zero in all five systems, and LIS sidecar <LIS SDK-embed <
min(LIS naive v1 , LIS naive v2 ) in 5/5 systems, including both real
codebases. In System A the SDK-embed scores 0.269 against
0.416 for the naive variants, driven by churn and coupling
(one import is one dependency, an inline call to a service
is two: the import and the endpoint). The real systems have
lower absolute LIS than the synthetic ones in every pattern,
because the same edit is a smaller fraction of a larger codebase,
so absolute values are not comparable across codebases. No
variant deletes code, so deleted lines contribute nothing here.
B. RQ2: Robustness and Dominance
The ranking holds in 25/25 (system, weight-vector) com-
binations for five hand-picked weightings and in all systems
for 2,000 random weight vectors sampled uniformly from the
simplex. This is expected: in every system each sub-metric
of the sidecar is≤that of SDK-embed, which is≤each
naive variant’s, with a strict improvement in at least one, so
the ordering holds foranystrictly positive weighting. RQ2
is a dominance property of the sub-metrics, not independent
evidence.
C. RQ3: Latency Overhead
Through the live gateway the legacy-only path takes
7.6 ms on average and the RAG-augmented path, served by
gpt-oss-120b, 348.0 ms (median 327, p95 433, range
175–1423 ms): a mean overhead of 340.4 ms (95% CI
[303.2,393.0]),U=0,p=7.1×10−18,r=1.00(N=50each;
all 100/100 requests returned HTTP 200). The RAG path
includes retrieval, an external LLM call and two extra service

TABLE I: Corpus size (files, non-blank LOC) and LIS sub-metrics (churn, breadth, coupling) with composite LIS by system
and pattern (default weights). The sidecar is zero in every row (no legacy file touched) and is omitted.
SDK-embed Naive-invasive v1 Naive-invasive v2
Sys. Files LOC Churn Breadth Coupl. LIS Churn Breadth Coupl. LIS Churn Breadth Coupl. LIS
A 2 53 0.038 0.500 0.500 0.269 0.132 0.500 1.000 0.416 0.132 0.500 1.000 0.416
B 3 64 0.047 0.333 0.333 0.190 0.141 0.333 0.667 0.304 0.078 0.333 0.667 0.272
C 2 36 0.111 0.500 0.500 0.306 0.250 0.500 1.000 0.475 0.167 0.500 1.000 0.433
D 4 266 0.011 0.250 0.250 0.131 0.030 0.250 0.500 0.190 0.023 0.250 0.500 0.186
E 3 157 0.019 0.333 0.333 0.176 0.045 0.333 0.667 0.256 0.038 0.333 0.667 0.252
hops, and its latency depends on the model (209–686 ms across
the four backends in RQ7), so the overhead is specific to this
configuration and is not a general RAG constant.
D. RQ4: Staleness
39 of 40 probes became retrievable. The remaining probe
had been indexed but ranked 12th of 43 documents, outside the
top-10 criterion for retrievability, so we count it as failed and
exclude it. Server-side staleness (legacy write timestamp to
index insertion, which excludes the probe loop’s own polling)
has mean 911 ms (95% CI[730,1093]), median 908, p95
1969 and maximum 2026 ms (Fig. 2c). This matches writes
landing uniformly within a 2 s poll cycle plus a few tens
of milliseconds of indexing; the probe loop added a mean
detection delay of 72 ms.
E. RQ5: Runtime Isolation
With every service a separate OS process there is no shared
interpreter, yet load on the RAG layer slowed the legacy
service from 3.63 to 5.47 ms when probed directly (1.51×,
95% CI[1.45,1.57];r=0.98) and from 6.99 to 11.00 ms
through the gateway (1.57×, CI[1.52,1.64]). The idle block
after the load matches the one before it (3.61 vs. 3.63 ms
directly, 1.51×against either), so drift does not explain the
effect. All 100 probes in each of the six blocks succeeded,
and 6 load requests completed (0 failed). The absolute added
delay is about 2–4 ms, but it shows that process separation on
one host does not isolate resources; the loaded block is short,
so this is a lower-bound signal.
F . RQ6: Operational Intrusion of All Three Patterns
Run-time overhead.Against the unmodified control
(3.80 ms) the sidecar deployment shows no detectable change
(difference 0.07 ms, CI[−0.10,0.24],p=0.31), although the
adapter polls it every 2 s. SDK-embed and naive-invasive add
569 ms (CI[418,752]) and 475 ms (CI[413,583]), both
p <10−13, because the LLM call now runs inside the legacy
handler. The two retrofits are not statistically distinguishable
from each other (p=0.62; difference 93 ms, CI[−97,304]).
The sidecar’s own feature path is not distinguishable from the
SDK path (p=0.13); against the naive pathp=0.029without
correction for three comparisons. Medians are similar (397,
427 and 436 ms for feature, SDK and naive) and the tails
heavy (4, 3 and 1 of 40 requests took 2.3–3.5 s), so in the
typical case the feature costs about the same under the three
patterns; the difference is where the cost is incurred, which for
the two retrofits is the legacy endpoint. All 200/200 requests
succeeded.Fault propagation (Table II).Two kinds of damage appear.
Loud:under a 2 s delay, the SDK and naive endpoints average
2.44 and 2.43 s with 100% SLO violations, while the sidecar’s
legacy endpoint stays at 3.8 ms.Silent:when the orchestrator
fails fast, the SDK and naive endpoints still answer with HTTP
success, within the SLO (their latency rises from 3.7 to 7.2 and
7.2 ms), but the AI field is absent from 100% of responses and
no error is raised, whereas the sidecar’s feature returns errors
in 100% of requests and its legacy endpoint is unaffected.
Both retrofits call the orchestrator synchronously and swallow
errors; a fixed 10 s timeout is their only stability measure, with
no circuit breaker, bulkhead or asynchronous execution [16].
LIS-X.The sidecar scores 0.000, SDK-embed 0.483 and
naive-invasive 0.556 (Table III). The operational terms are
nearly equal for the two retrofits (runtime 0.993 vs. 0.992,
fault 0.50 and degradation 0.50 for both), so they erase half of
SDK-embed’s code-level advantage (the SDK–naive gap falls
from 0.147 to 0.073), and the ranking holds in 100% of 2,000
random six-dimensional weight vectors, but only because the
code terms still favour SDK-embed.
G. RQ7: Retrieval, Answer Quality and Backend Swap
On 40 queries over 20 documents, dense retrieval reaches
hit@1 0.95 (38/40, bootstrap CI[0.875,1.000]), hit@3 1.00
and MRR@5 0.975, against 0.875, 0.90 and 0.888 for TF-IDF;
dense retrieval ranks the gold article first on 3 queries where
TF-IDF does not, and TF-IDF on 0 where dense retrieval does
not. Each hosted model answers all 12 questions correctly
andflan-t5-small9/12; strict and lenient scoring agree
on all 48 answers. Reading all 48 confirmed that every
match is correct and that the 3 flagged answers, all from
flan-t5-small, are wrong: two reproduce a different arti-
cle (a travel question answered with the meeting-room article,
a remote-work question with the leave carry-over article) and
one is “unanswerable”. The gold article was retrieved in
all 48/48 cases, so these are generation errors. Across the
four backends, all 36/36 invocations through gateway and
orchestrator succeeded with only the request’s model field
changed, with mean latencies of 209 (Qwen3.8-27B), 269
(gpt-oss-20b), 382 (gpt-oss-120b) and 686 ms (flan-t5-small;
Fig. 2f). The surrounding code is unchanged, so the latency
differences come from the backends. This is a sanity check,
not a benchmark.
VI. DISCUSSION
Three points follow. First, code-level cleanliness is not run-
time containment: SDK-embed wins on every code dimension
yet matches the naive call at run time, so a retrofit should be
judged on both. An SDK is the natural place to put timeouts,

TABLE II: RQ6: legacy endpointGET /api/articles/<id>per deployment and under injected RAG-layer faults (ms
unless noted; Fail = HTTP error or exception; No AI = response lacks the AI field; SLO = share of requests failing or slower
than 50 ms).
No fault Fault: orchestrator returns error Fault: 2 s delay
Path Mean p95 Mean Fail % No AI % Mean SLO viol. %
Control (no retrofit) 3.8 4.5 3.7 0 – 3.7 0
Sidecar 3.9 4.3 3.8 0 – 3.8 0
SDK-embed 572.3 2383.0 7.2 0 100 2440.8 100
Naive-invasive 479.1 504.0 7.2 0 100 2426.5 100
Sidecar RAG feature (/ask) 618.9 2383.1 7.0 100 – 2467.7 100
control sidecarSDKnaivefeature101102103mean latency (ms, log)3.8 3.9572 479619(a) RQ6a: legacy endpoint
control sidecarSDKnaivefeature050100% of requests(b) RQ6b: fault propagation
error: SLO
2 s delay: SLO
error: no AI field
0 1000 2000
write-to-indexed (ms)0246probesmean
911 ms(c) RQ4: staleness, N=39
idle
beforeunder
loadidle
after051015mean latency (ms)3.65.5
3.67.011.0
7.1(d) RQ5: isolation
direct via gateway
hit@1 hit@3 MRR@50.00.51.0(e) RQ7: retrieval quality
dense TF-IDF
Qwen3.8
27Bgpt-oss
20bgpt-oss
120bflan-t5
(local)025050075010001250latency via gateway (ms)209269382686(f) backends, N=9 each
Fig. 2: Dynamic results on System A. (a) Mean latency of the legacy endpoint per deployment (log scale; “feature” is the
sidecar’s RAG path). (b) Share of requests violating the SLO under each fault, and share of responses without the AI field
(silent degradation). (c) Server-side staleness. (d) Legacy latency before, during and after load on the RAG layer. (e) Retrieval
quality. (f) Mean latency (sd) per LLM backend through the full pipeline.
TABLE III: LIS-X sub-metrics for System A (churn, breadth
and coupling as in Table I; LIS is the code-only composite).
Pattern Churn Bread. Coupl. Runt. Fault Degr. LIS LIS-X
Sidecar 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000
SDK 0.038 0.500 0.500 0.993 0.500 0.500 0.269 0.483
Naive 0.132 0.500 1.000 0.992 0.500 0.500 0.416 0.556
a circuit breaker and asynchronous execution [16]; the SDK
evaluated here does not, which its operational scores reflect.
Second, swallowing errors turns a loud failure into a silent
one: the legacy endpoint stays fast and returns success while
the feature is gone, so legacy health checks cannot see it and
feature-level success must be monitored. Third, the sidecar’s
isolation costs staleness of about half the polling interval and
an extra process, and it is not free even across processes on
one host, so “zero coupling” is a code-level claim.
VII. THREATS TOVALIDITY
Construct.LIS and LIS-X are proxies; weights and the
SLO threshold are choices, and RQ1/RQ2 are near-definitional
(Section III). Degradation is defined only for the two retrofits
whose legacy endpoint should carry the AI field. Answer
correctness checks one gold fact per question; the strict scorerand the manual review of every answer limit, but do not
remove, errors in either direction.Internal.All services share
one host (4 vCPU), so contention is host-level. The hosted
models are external services whose latency varies and whose
catalogue changes, so the model IDs are recorded; the live
study is a single run without replicates, and the RQ5 loaded
block is short. Interleaving, idle–load–idle and random probe
phase reduce, but cannot remove, order effects.External.
The corpus is small, Python/Flask-only, and System E is a
three-file excerpt; the dynamic study covers System A only.
All variants were written by one researcher, and “naive” as
implemented (synchronous, 10 s timeout, errors swallowed)
may not match real code; an asynchronous SDK would score
differently on RQ6.Conclusion.Sample sizes are modest (10
to 100 observations per condition; 12 answer questions; 9
backend invocations per model), and the corpus supports only
the claim that the ranking held in the systems tested.
VIII. CONCLUSION ANDFUTUREWORK
We presented a taxonomy of RAG-retrofit patterns, the
code-level LIS, and LIS-X with run-time overhead, fault prop-
agation and silent feature degradation. Statically, the sidecar
<SDK-embed<naive-invasive ranking held in five systems

and under every weighting, but only because of dominance.
Live, the sidecar left the legacy endpoint untouched while both
in-process retrofits added hundreds of milliseconds, passed a
dependency delay through, and lost the feature silently when
the RAG layer failed fast; SDK-embed gave code-level but
not run-time containment. Four LLM backends were swapped
by changing one request field. Future work should (i) have
independent engineers write the integrations, (ii) give the SDK
timeouts, circuit breaking and asynchronous execution and re-
measure, (iii) replicate the live study over several runs and
backends, (iv) extend the corpus beyond Flask/Python, and
(v) use separate hosts or containers.
DATA ANDARTIFACTAVAILABILITY
The artifact contains the notebook and its executed copy
(static study, service scripts, experiments), the raw per-
request measurements and summary tables (CSV/JSON), the
script that generates every number and figure in this paper
from them, per-run metadata (versions, model identifiers,
configuration) and the pinned upstream commit hashes.
Upstream code is downloaded at run time, not redistributed;
licence texts are saved with the results. The notebook stores
no credentials: Groq keys are read from Kaggle Secrets,
and without them the run falls back to the local model.
Hosted-model results vary between runs. The artifact is
available athttps://github.com/TahlilMahfuz/Quantifying-
the-Intrusiveness-of-Retrofitting-RAG-onto-Legacy-Software-
Systems.git.
ACKNOWLEDGMENT
We thank the Pallets Projects (Flask tutorial [20],
BSD-3-Clause) and the gothinkster/RealWorld contributors
(MIT) [21], [22], whose code was measured unmodified, and
Groq, OpenAI and the Qwen team for the hosted and open-
weight models [25]–[27].
REFERENCES
[1] P. Lewis, E. Perez, A. Piktus, F. Petroni, V . Karpukhinet al., “Retrieval-
augmented generation for knowledge-intensive NLP tasks,” inAdv.
Neural Inf. Process. Syst. (NeurIPS 2020), vol. 33, 2020, pp. 9459–
9474.
[2] Y . Gao, Y . Xiong, X. Gao, K. Jia, J. Pan, Y . Bi, Y . Dai, J. Sun,
M. Wang, and H. Wang, “Retrieval-augmented generation for large
language models: A survey,” arXiv:2312.10997, 2024.
[3] J. Bisbal, D. Lawless, B. Wu, and J. Grimson, “Legacy information
systems: Issues and directions,”IEEE Software, vol. 16, no. 5, pp. 103–
111, 1999.
[4] R. Khadka, B. V . Batlajery, A. M. Saeidi, S. Jansen, and J. Hage, “How
do professionals perceive legacy systems and software modernization?”
inProc. 36th Int. Conf. on Software Engineering (ICSE ’14). ACM,
2014, pp. 36–47.
[5] J. van Gurp and J. Bosch, “Design erosion: Problems and causes,”J.
Syst. Softw., vol. 61, no. 2, pp. 105–119, 2002.
[6] S. Barnett, S. Kurniawan, S. Thudumu, Z. Brannelly, and M. Abdelrazek,
“Seven failure points when engineering a retrieval augmented generation
system,” inProc. 3rd IEEE/ACM Int. Conf. on AI Engineering – Software
Engineering for AI (CAIN), 2024, pp. 194–199.
[7] M. Breque, L. De Nul, and A. Petridis, “Industry 5.0: Towards a
sustainable, human-centric and resilient European industry,” European
Commission, Directorate-General for Research and Innovation, Luxem-
bourg, Tech. Rep., 2021.
[8] B. Burns and D. Oppenheimer, “Design patterns for container-based
distributed systems,” inProc. 8th USENIX Conf. on Hot Topics in Cloud
Computing (HotCloud ’16). Denver, CO: USENIX Association, 2016,
pp. 108–113.[9] Microsoft Learn, “Sidecar pattern,” Azure Architecture Center, https:
//learn.microsoft.com/azure/architecture/patterns/sidecar, 2026, accessed
7 November 2026.
[10] M. Fowler, “StranglerFigApplication (first published 29 June 2004 as
“StranglerApplication”),” martinfowler.com bliki, https://martinfowler.
com/bliki/StranglerFigApplication.html, 2004.
[11] E. Evans,Domain-Driven Design: Tackling Complexity in the Heart of
Software. Boston, MA: Addison-Wesley, 2003.
[12] Microsoft Learn, “Anti-corruption layer pattern,” Azure Archi-
tecture Center, https://learn.microsoft.com/azure/architecture/patterns/
anti-corruption-layer, 2026, accessed 7 November 2026.
[13] N. Nagappan and T. Ball, “Use of relative code churn measures to predict
system defect density,” inProc. 27th Int. Conf. on Software Engineering
(ICSE ’05). St. Louis, MO: ACM, 2005, pp. 284–292.
[14] S. R. Chidamber and C. F. Kemerer, “A metrics suite for object oriented
design,”IEEE Trans. Softw. Eng., vol. 20, no. 6, pp. 476–493, 1994.
[15] M. Kleppmann,Designing Data-Intensive Applications: The Big Ideas
Behind Reliable, Scalable, and Maintainable Systems. O’Reilly, 2017.
[16] M. T. Nygard,Release It!: Design and Deploy Production-Ready Soft-
ware, 2nd ed. Pragmatic Bookshelf, 2018.
[17] A. Basiri, N. Behnam, R. de Rooij, L. Hochstein, L. Kosewski,
J. Reynolds, and C. Rosenthal, “Chaos engineering,”IEEE Software,
vol. 33, no. 3, pp. 35–41, 2016.
[18] P. Runeson and M. H ¨ost, “Guidelines for conducting and reporting case
study research in software engineering,”Empir. Softw. Eng., vol. 14,
no. 2, pp. 131–164, 2009.
[19] C. Wohlin, P. Runeson, M. H ¨ost, M. C. Ohlsson, B. Regnell, and
A. Wessl ´en,Experimentation in Software Engineering. Springer, 2012.
[20] Pallets Projects, “Flask: A python micro web framework — tuto-
rial application (flaskr),” GitHub repository, https://github.com/pallets/
flask, 2026, bSD-3-Clause license; commit 49943b5, path exam-
ples/tutorial/flaskr.
[21] gothinkster and contributors, “RealWorld: the “mother of all demo
apps” (Medium.com clone specification),” GitHub repository, https:
//github.com/gothinkster/realworld, 2026, accessed November 2026.
[22] ——, “flask-realworld-example-app: A flask/SQLAlchemy reference
backend for the RealWorld specification,” GitHub repository, https:
//github.com/gothinkster/flask-realworld-example-app, 2016, mIT li-
cense; commit 4b95fb2.
[23] N. Reimers and I. Gurevych, “Sentence-BERT: Sentence embeddings
using Siamese BERT-networks,” inProc. 2019 Conf. on Empirical
Methods in Natural Language Processing (EMNLP-IJCNLP), 2019, pp.
3982–3992.
[24] J. Johnson, M. Douze, and H. J ´egou, “Billion-scale similarity search
with GPUs,”IEEE Trans. Big Data, vol. 7, no. 3, pp. 535–547, Jul.
2021.
[25] Groq, “GroqDocs: API documentation,” https://console.groq.com/docs,
2026, accessed 7 November 2026.
[26] OpenAI, “gpt-oss-120b & gpt-oss-20b model card,” arXiv:2508.10925,
2025.
[27] Qwen Team, “Qwen3.8-27B model card,” Hugging Face, https://
huggingface.co/Qwen/Qwen3.8-27B, 2026, released 14 August 2026,
Apache-2.0 license.
[28] C. Raffel, N. Shazeer, A. Roberts, K. Lee, S. Narang, M. Matena,
Y . Zhou, W. Li, and P. J. Liu, “Exploring the limits of transfer learning
with a unified text-to-text transformer,”J. Mach. Learn. Res., vol. 21,
no. 140, pp. 1–67, 2020.
[29] H. W. Chung, L. Hou, S. Longpre, B. Zoph, Y . Tayet al., “Scaling
instruction-finetuned language models,” arXiv:2210.11416, 2022.
[30] T. Wolf, L. Debut, V . Sanh, J. Chaumond, C. Delangueet al., “Trans-
formers: State-of-the-art natural language processing,” inProc. 2020
Conf. on Empirical Methods in Natural Language Processing: System
Demonstrations, 2020, pp. 38–45.
[31] H. B. Mann and D. R. Whitney, “On a test of whether one of two
random variables is stochastically larger than the other,”Ann. Math.
Statist., vol. 18, no. 1, pp. 50–60, 1947.
[32] D. S. Kerby, “The simple difference formula: An approach to teaching
nonparametric correlation,”Comprehensive Psychology, vol. 3, 2014,
article 1.
[33] B. Efron, “Bootstrap methods: Another look at the jackknife,”Ann.
Statist., vol. 7, no. 1, pp. 1–26, 1979.
[34] G. Salton and C. Buckley, “Term-weighting approaches in automatic
text retrieval,”Inf. Process. Manag., vol. 24, no. 5, pp. 513–523, 1988.
[35] F. Pedregosa, G. Varoquaux, A. Gramfort, V . Michel, B. Thirionet al.,
“Scikit-learn: Machine learning in Python,”J. Mach. Learn. Res.,
vol. 12, pp. 2825–2830, 2011.