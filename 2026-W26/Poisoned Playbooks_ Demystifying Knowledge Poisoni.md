# Poisoned Playbooks: Demystifying Knowledge Poisoning Effects on AI Security Agents

**Authors**: Juho Park, Hyunmin Choi, Kevin Nam

**Published**: 2026-06-23 10:37:34

**PDF URL**: [https://arxiv.org/pdf/2606.24402v1](https://arxiv.org/pdf/2606.24402v1)

## Abstract
AI security agents increasingly rely on Retrieval-Augmented Generation (RAG) to use external security knowledge for vulnerability analysis and exploit reasoning. This creates a new risk: poisoned write-ups can be operationalized into incorrect exploit behavior. Yet, prior work on RAG poisoning has mostly studied answer corruption in QA settings, much less is known about action-taking security agents. This paper aims to reveal such characteristics with crafted poisons about real-world challenges and AI agents. First, we demonstrate how a crafted single poisoned write-up injected into public-style security knowledge sources which we denote as Poisoned Playbooks, alters the behavior of RAG-based AI security agents. Across 11 CTF challenges, 3 frontier LLM families, 2 model generations, and 11 real-world CVEs, we find that poison adoption is systematic rather than random. To explain this pattern, we introduce the Verification Boundary (VB), a 3-level empirical classification based on what evidence the agent can use to refute a retrieved claim. Finally, we evaluate verification prompting and multi-source retrieval, showing that both help when stronger evidence exists, but weaken under sparse-evidence and zero-day conditions.

## Full Text


<!-- PDF content starts -->

Poisoned Playbooks: Demystifying Knowledge
Poisoning Effects on AI Security Agents
Juho Park1[0009−0003−4394−0449], Hyunmin Choi2,∗[0009−0002−0486−9582], and
Kevin Nam3,∗[0000−0002−4621−2434]
1Security Defense, NAVER Cloud, Gyunggi-do, Republic of Korea
2College of AI Convergence, Dankook University, Gyunggi-do, Republic of Korea
3School of Computing, Kyung Hee University, Gyunggi-do, Republic of Korea
juho.park@navercorp.com, hyunmin.choi@dankook.ac.kr, kvnam@khu.ac.kr
*: correspondence should be addressed to H. Choi and K. Nam
Abstract. AI security agents increasingly rely on Retrieval-Augmented
Generation (RAG) to use external security knowledge for vulnerability
analysis and exploit reasoning. This creates a new risk: poisoned write-ups
can be operationalized into incorrect exploit behavior. Yet, prior work
on RAG poisoning has mostly studied answer corruption in QA settings,
much less is known about action-taking security agents. This paper
aims to reveal such characteristics with crafted poisons about real-world
challenges and AI agents. First, we demonstrate how a crafted single
poisoned write-up injected into public-style security knowledge sources
which we denote asPoisoned Playbooks, alters the behavior of RAG-based
AI security agents. Across 11 CTF challenges, 3 frontier LLM families, 2
model generations, and 11 real-world CVEs, we find that poison adoption
is systematic rather than random. To explain this pattern, we introduce
theVerification Boundary(VB), a 3-level empirical classification based
on what evidence the agent can use to refute a retrieved claim. Finally, we
evaluate verification prompting and multi-source retrieval, showing that
bothhelpwhenstrongerevidenceexists,butweakenundersparse-evidence
and zero-day conditions.
Keywords:AI security agents·knowledge poisoning·RAG·security
supply chain·Verification Boundary
1 Introduction
Large language models (LLMs) [ 50,25] are increasingly deployed not only as
conversational assistants, but also as agents that retrieve external information,
reason over it, and act on behalf of users. A common architectural pattern in
such systems is Retrieval-Augmented Generation (RAG) [ 41,33,39,37], which
supplements parametric model knowledge with external documents retrieved at
inference time. While RAG improves freshness and domain specificity, it also
creates a new integrity risk–recent works demonstrated that malicious documents
can corrupt RAG pipelines using numerous methods [53,36,28].arXiv:2606.24402v1  [cs.CR]  23 Jun 2026

2 J. Park, H. Choi, and K. Nam
This paper focuses on how such attacks can affect AI security agents. Security
is a particularly important example of such a setting. LLM-based systems are
increasinglybeingexploredforincidentinvestigation,threathunting,vulnerability
analysis, exploit reasoning, and autonomous penetration testing [ 32,12,43,35].
In these workflows, external knowledge is consumed as operational guidance. A
retrieved write-up may shape not only an explanation, but also a concrete exploit
hypothesis, a sequence of commands, or a decision to abandon one attack path
in favor of another. As a result, poisoning in this domain threatens the behavior
of the agent itself, not merely the text it produces. The risk is amplified by the
waySecurity Knowledge(SK) is produced and consumed. In practice, analysts
often rely on fast-moving public sources such as exploit repositories, community
vulnerability templates, GitHub repositories, issue threads, proof-of-concept
releases, and blog-based write-ups to understand newly disclosed vulnerabilities
and their exploitation conditions [ 38,9,10]. While these sources are timely and
useful, they are also noisy, weakly curated, and susceptible to manipulation; prior
work has shown that public exploit repositories and GitHub-hosted PoCs can
contain misleading or even malicious content [ 31]. This problem is especially acute
in sparse-evidence settings such as newly emerging or zero-day vulnerabilities,
where authoritative counter-evidence may be unavailable or delayed.
Motivated by this gap, we study how poisoned external knowledge affects
the behavior of AI security agents. Our vehicle isPoisoned Playbooks: crafted
write-ups injected into public-style SK repositories and then consumed by
RAG-based agents during vulnerability-solving tasks. Unlike prior poisoning
settings centered on factual QA [ 53,36], our focus is action-oriented: the agent
must convert retrieved claims into exploit strategies and concrete decisions. To
structure this study, we investigate the following research questions:
RQ1.Can a single poisoned write-up injected into a public SK source
systematically alter the behavior of RAG-based AI security agents?(§4)
RQ2.What determines whether a poisoned claim is rejected or adopted, and does
this pattern generalize across model families, model generations, and real-world
vulnerabilities?(§4, §5, §6.1)
RQ3.Which mitigation strategies reduce poison adoption, and under what
conditions do they fail?(§7)
We answer these questions through a study spanning 11 CTF challenges,
three frontier LLM families, two model generations, and 11 real-world CVEs. Our
main finding is that poison adoption is governed less by vulnerability difficulty
alone than by what evidence the agent can use to independently challenge a
retrieved claim. We capture this pattern through theVerification Boundary(VB),
a three-level empirical classification that separates claims that can be checked
against local artifacts, claims that depend on model-side knowledge, and claims
whose refutation lies outside the agent’s immediate observable context. In brief,
our contributions are as follows:
•We present a systematic empirical study of how knowledge poisoning affects AI
security agents, spanning 11 CTF challenges, 3 frontier LLM families, 2 model
generations, and 11 real-world CVEs.

Poisoned Playbooks 3
•We present theVerification Boundary, a 3-level classification that explains
when retrieved claims are rejected, model-dependent, or consistently adopted.
•We show that the same pattern generalizes from controlled challenge settings to
real-world CVEs, and that sparse-evidence regimes remain especially vulnerable.
•We evaluate practical mitigation strategies and identify the conditions under
which they help and the regimes in which they remain insufficient.
2 Background
2.1 Retrieval-Augmented Generation
Retrieval-Augmented Generation (RAG) augments a large language model (LLM)
with an external knowledge source, allowing the model to condition its outputs
on retrieved documents rather than relying solely on parametric memory [ 41,33].
A standard RAG pipeline consists of a knowledge database, a retriever, and an
LLM: the retriever selects relevant documents for a query, and the LLM generates
its response conditioned on the retrieved context [ 53]. This architecture improves
freshness and domain specificity without requiring model retraining.
At the same time, RAG expands the system’s trust boundary. In a standalone
LLM, integrity concerns primarily involve the model parameters and prompt
context. In a RAG system, correctness also depends on the integrity of the
external corpus and the retrieval process that selects from it. The knowledge
database therefore becomes part of the system’s effective decision surface [53].
2.2 AI-Powered Security Agents and SK Sources
Recent advances in LLM reasoning and tool use have enabled AI-powered security
agents that combine language-model inference with tools such as shell access,
web interaction, and code execution. These systems are increasingly explored for
incident investigation, threat hunting, vulnerability analysis, exploit reasoning,
and autonomous penetration testing [ 32,24,12,43]. Unlike answer-centric QA
systems, security agents use external knowledge operationally: retrieved content
can shape exploit hypotheses, determine which path to prioritize, and influence
whether the agent persists on or abandons a candidate strategy.
This makes the quality of externalSecurity Knowledge(SK) unusually
important. In practice, security agents often rely on fast-moving public
sources including CTF write-ups, proof-of-concept repositories, Exploit-DB
entries, issue threads, blogs, and community advisories [ 43]. These sources are
timely and detailed, but often weakly curated and unevenly validated. Newly
disclosed vulnerabilities are frequently discussed in public before authoritative
counter-evidence is widely available, and many exploitation claims concern the
behavior of external components such as libraries, runtimes, operating systems,
database engines, and browser internals.
Prior work already suggests that this ecosystem can contain misleading
content. Public exploit repositories have been shown to include fake or deceptive

4 J. Park, H. Choi, and K. Nam
proof-of-concept logic [ 31]. More broadly, studies of open-source supply-chain
attacks and web-scale data poisoning show that low-cost adversaries can inject
attacker-controlled content into widely consumed technical ecosystems [ 40,44,27].
These observations motivate treating the SK supply chain as both a source of
useful context and a realistic attack surface for downstream AI agents.
2.3 Existing Poisoning Attacks on RAG and Agents
A growing body of work has shown that RAG systems are vulnerable to
poisoning of their external knowledge sources. PoisonedRAG demonstrates that
an attacker can inject crafted documents into a knowledge base and steer a model
toward attacker-chosen answers [ 53]. Follow-up work studies more practical
poison generation in deployed RAG pipelines [ 36], as well as retriever-focused
or backdoor-style attacks such as Phantom [ 28] and TrojanRAG [ 30]. These
works establish that the retrieval corpus is a meaningful attack surface, but they
primarily evaluate success in answer-centric settings.
Related work has also considered poisoning and manipulation in agent settings.
AgentPoison shows that poisoning an agent’s memory or knowledge base can
redirect downstream behavior [ 29]. Other work on indirect prompt injection shows
that agents can be manipulated through attacker-controlled content encountered
during execution [ 34,51,45,49]. These attacks are closely related, but differ from
our setting in important ways: some target internal memory rather than public
knowledge ecosystems, while others require runtime interaction with the victim
agent or delivery through live tool outputs.
3 Problem Setting and Study Design
3.1 Threat Model
We study apre-positioned knowledge poisoningsetting in which an adversary
aims to influence downstream AI security agents by publishing crafted content
to publicSecurity Knowledge(SK) sources. The attacker does not compromise
the model, retriever, or execution environment of the victim agent. Instead, the
attack succeeds if a poisoned artifact is later retrieved through normal RAG
usage and incorporated into the agent’s reasoning.
Adversary capabilities.We assume an adversary who can publish content
to public SK sources (e.g., GitHub repositories, blogs, forums, or community
write-up platforms), anticipate that downstream agents rely on external retrieval,
and craft poisoned write-ups offline, butwithoutruntime access to the target
system, agent session, or private memory.
Adversary goal.The adversary aims to induce the agent to adopt poisoned
claims and thereby alter downstream exploit behavior. In practice, this may cause
the agent to dismiss a real vulnerability, pursue a non-existent path, or follow a
flawed exploitation strategy.

Poisoned Playbooks 5
Table 1.Comparison of our setting with related works.‡Effectiveness degrades as
authoritative sources appear, models internalize the relevant domain, or the agent gains
access to stronger verification evidence. KB denotes the knowledge base.
Property Ours Mantis [45] PoisonedRAG [53]
Injection point Public security KB Live target server RAG corpus
Runtime access required No Yes No
Persistence Until displaced‡Per-session Until displaced
Scale of influence 1:N 1:1 1:N
Primary effect Behavioral Behavioral Answer corruption
Poison artifacts / target 1 N/A 5
Fig. 1.Overview of the workflow of our study. VB is introduced in §5.
This setting is especially relevant in security because agents often rely on
public exploit intelligence precisely when authoritative information is incomplete.
Newly disclosed and zero-day vulnerabilities are therefore particularly exposed:
early in the disclosure cycle, public write-ups may dominate the evidence available
to the agent, while vendor guidance, patches, or broad community validation are
still absent. More generally, many exploitation claims concern external runtime
behavior such as library internals, operating-system behavior, database engines,
or browser logic, which the agent cannot directly inspect from the immediately
available artifacts alone.
Compared with generic RAG poisoning, our setting is action-oriented rather
than answer-centric: the poisoned content is used to alter exploit behavior rather
than merely corrupt a textual answer. Compared with runtime attacks such as
Mantis [45], the attacker requires no live interaction with the victim session and
no server-side foothold. Table 1 summarizes these differences.
3.2 Poisoned Playbooks
Figure 1 illustrates our study vehicle:Poisoned Playbooks, crafted security
write-ups designed to resemble plausible exploit guidance while embedding
attacker-chosen false claims. Rather than directly instructing the model, the
poison presents itself asSK, so the attack corrupts retrieved evidence rather than
the prompt itself.

6 J. Park, H. Choi, and K. Nam
Table 2.Challenge categories used in the evaluation.
Category Contents
CTF Challenges XSS[4]×3, CSP, NoSQL[ 7], SSRF[ 6]×2, SSTI[ 1], race [ 2]×2,
JWT.
Well-documented
controlsSQLi UNION[5], basic SSRF[6], XXE[3].
Real-World CVEs Jenkins (2024-23897)[ 15], Log4Shell (2021-44228)[ 13], Spring4Shell
(2022-22965)[ 14], React2Shell (CVE-2025-55182)[ 19], Redis Lua
(49844)[17], SharePoint (53770)[ 18], n8n (68613)[ 23], Oracle EBS
(61882)[21], WSUS (59287)[ 20], OpenSSL (15467)[ 16], and
Apache Tika (66516)[22].
Each playbook is centered on a false claim intended to alter downstream
behavior, typically about an external dependency, an environment-specific
assumption, or an exploitability condition that cannot be resolved from the
immediate source code or challenge description. To make these poisons realistic,
we model them after common public write-up styles, including first-person
solving narratives, fabricated quantitative evidence, acknowledge-then-refute
structures, and complete alternative exploit suggestions. These properties
motivate why poisoned playbooks are plausible study artifacts; we analyze them
more systematically later in §6.3. Poisoned playbooks are also cheap to produce.
In our study, each write-up required roughly 30–60 minutes of authoring effort,
yet a single artifact could influence any downstream agent that retrieved it. We
therefore use poisoned playbooks not to claim a new attack primitive as the
paper’s main contribution, but as a controlled vehicle for studying when AI
security agents reject poisoned knowledge and when they adopt it.
3.3 Experimental Setup–Testbed, Challenges, and Metrics
As our experimental setup, we customized a testbed with a RAG-augmented
security agent configured in line with common practice in prior work and
production-style systems. It uses a hybrid retrieval pipeline over 8,651
security write-ups, combining SQLite FTS5 keyword search and Qdrant vector
search ( qwen3-embedding ) with fusion-based ranking. We evaluate two Claude
generations (Opus 4 and Opus 4.6) and additionally cross-check selected results
on GPT-5.3 and Gemini 3.0 Pro using default API settings. Our focus is the
model’spoison adoption behavior, rather than the idiosyncrasies of a particular
agent implementation.
Table 2 summarizes the targets used in our evaluation. Niche CTFs consist of
challenges from Dreamhack [ 8] which provide a natural high-monopoly setting
because they are primarily documented in Korean and have limited competing
English-language evidence. Well-documented controls are from PortSwigger
labs [11] that serve as low-monopoly controls with abundant public walkthroughs.
The real-world CVE set tests whether the same pattern generalizes beyond the
challenge setting. Within that set, the three well-documented CVEs provide
strong competing evidence in model knowledge, while the post-training-cutoff

Poisoned Playbooks 7
CVEs approximate sparse-evidence conditions. For those post-cutoff cases, the
injectedpoisontargetsruntime-dependentclaimsthatcannotberesolvedfromthe
CVE description alone. We use three complementary settings. Inoracle retrieval,
the poisoned write-up is directly provided as retrieved context alongside source
artifacts, isolating adoption from retrieval quality. Inend-to-end (E2E)evaluation,
poisons are injected into a private knowledge base replica, correct competing
write-ups are withheld when needed to establish information monopoly, and the
agent solves the task with tool access. Finally, inreproduction / cross-generation
comparison, the same challenge set is rerun on a newer model generation to test
whether the observed effects persist or shift with model capability.
Our primary metric isPoison Adoption Rate (PAR), a binary assessment
of whether the agent’s strategy depends on a claim unique to the poison. We also
record the retrieval rank of the poisoned document in end-to-end settings and
the rejection cause when adoption does not occur.
4 Poisoning Outcomes Across Settings
We begin by answeringRQ1. Across controlled oracle-retrieval experiments,
cross-model comparisons, and end-to-end evaluation, a single poisoned write-up
can systematically alter the behavior of RAG-based AI security agents but
not uniformly. Some claims are rejected, some vary across model generations,
and others are adopted consistently. This section establishes the phenomenon
itself–the explanatory pattern behind these differences is developed in §5.
Oracle Retrieval: Establishing the Phenomenon.We first isolateadoption
fromretrieval. In the oracle-retrieval setting, the poisoned write-up is directly
provided as retrieved context together with the relevant challenge artifacts. This
removes retrieval quality as a confounding factor and asks a simpler question:
once the poison is available to the agent as evidence, will the agent use it? Across
the 11 CTF challenges, the answer is yes, but not uniformly. Some poisons are
rejected even under direct exposure, indicating that retrieval alone is insufficient
to explain success. Others are adopted only by the less capable model generation,
while a third group is adopted consistently across both generations. In successful
cases, the agent does not merely repeat the false claim textually; it changes
its exploit strategy, abandoning the correct path or justifying why a genuine
vulnerability should be ignored. The rejected cases are equally informative: when
source code or local artifacts expose strong contradictory evidence, the agent may
reject the poisoned claim despite being forced to read it. This already suggests
that successful poisoning depends not only on whether a claim is retrieved, but
also on whether the agent has access to evidence capable of falsifying it.
Cross-Model and Cross-Generation Behavior.To test whether the
phenomenon is tied to a single model family, we next compare behavior across
Claude, GPT-5.3, and Gemini, and also across two Claude generations. Several
challengesareadoptedacrossalltestedfamilies,indicatingthatthephenomenonis
robust to model choice. Table 3 shows that all cross-model runs result in adoption
under information-monopoly conditions. The models differ in style. Claude tends

8 J. Park, H. Choi, and K. Nam
Table 3.Cross-model adoption under information monopoly.
Challenge Observed outcome Claude GPT-5.3 Gemini
dh-47 Reasoning corruption✓ ✓ ✓
dh-90 False negative✓ ✓ ✓
dh-75 Target deflection✓ ✓ ✓
dh-106 False negative✓ ✓ ✓
dh-675 Target deflection✓ ✓ ✓
dh-434 Target deflection —✓ ✓
dh-435 Reasoning corruption —✓ ✓
dh-438 False negative —✓ ✓
dh-552 False negative —✓ ✓
Total 5/5 9/9 9/9
to produce longer justifications, GPT-5.3 often generates more concrete exploit
scripts, and Gemini is typically more assertive. Despite these differences, the
poisoned claim is consistently incorporated into the agent’s reasoning.
Cross-generation comparison provides a second important result. Some
challenges adopted on Opus 4 are rejected on Opus 4.6, indicating that model
improvement can expand the set of claims the model can refute using stronger
prior knowledge or reasoning. However, another subset remains adopted across
both generations. This suggests that poison adoption is structured rather than
arbitrary: some claims become resistible as models improve, while others remain
difficult to challenge even for newer models.
End-to-End Pipeline Validation.The oracle-retrieval setting shows that
poisoned knowledge can redirect reasoning once it is available to the agent. We
next ask whether the same effect survives in a full pipeline where retrieval,
ranking, and interaction with the challenge environment are all active. In the
end-to-end setting, poisoned write-ups are injected into a private knowledge-base
replica and the agent solves the challenge with normal tool access. This allows
us to distinguish two possible failure modes: the poison might fail because it is
never surfaced by retrieval, or it might be retrieved but then rejected during
reasoning. Our results show that retrieval is oftennotthe bottleneck. In successful
end-to-end cases, the poisoned document ranks at or near the top of the retrieved
results, indicating that a single crafted artifact can be made salient enough to
enter the agent’s evidence set. More importantly, when poisoning fails, the failure
is often areasoning rejectionrather than a retrieval miss. The poison reaches the
agent, but the agent rejects it because stronger contradictory evidence is available.
A representative contrast appears between challenges such as dh-47 and dh-75.
In dh-47, the agent retrieves the poisoned write-up but also encounters local code
evidence that contradicts the poisoned claim, and the poison is rejected. In dh-75,
by contrast, the poisoned claim concerns behavior of an external component and
no directly comparable counter-evidence is available in the challenge artifacts; the
poison is therefore adopted end-to-end as well. This shows that the phenomenon
is not merely a retrieval artifact.

Poisoned Playbooks 9
Overall, the end-to-end experiments validate that poisoned public SK is not
just a laboratory curiosity under forced retrieval. A single poisoned artifact
can survive retrieval, enter the agent’s context, and alter exploit behavior in a
realistic RAG-based pipeline. At thesame time,the mixedpattern ofrejectionand
adoption indicates that the central question is no longerwhetherthe phenomenon
exists, butwhat determineswhen it succeeds.
5 The Verification Boundary
The results in §4 suggest that poison adoption depends less on vulnerability
difficulty alone than on what evidence the agent can use to challenge a retrieved
claim. We make that pattern explicit through theVerification Boundary(VB), a
three-level empirical classification of claims by the kind of verification available
to the agent. As illustrated in Figure 1, the agent’s reasoning is shaped by two
sources of evidence: the observable artifacts in context, denoted by S, and the
model’s parametric knowledge, denoted by K. The VB does not describe whether
a claim is objectively true or false; rather, it characterizes whether the agent can
independently refute it givenSandK.
Level 1: Code-Verifiable (L1).A claim isCode-Verifiableif the
available artifacts—for example, source code, configuration, or challenge-local
logic—contain sufficient evidence to confirm or refute it through inspection. These
claims are reliably rejected once the relevant evidence is examined.
Level 2: Knowledge-Verifiable (L2).A claim isKnowledge-Verifiableif
the local artifacts do not settle the question, but the model may still reject it
using parametric knowledge or broader domain understanding. These cases are
model-dependent: stronger or newer models may reject the poison, while weaker
or older ones may adopt it.
Level 3: Runtime-Dependent (L3).A claim isRuntime-Dependentif
assessing it requires information about external dependencies or execution
conditions that are not available in the immediate artifacts and are not reliably
encoded in the model’s prior knowledge. These claims are adopted consistently
across all tested models and generations.
Taken together, these levels define a verification boundary between claims the
agent can independently challenge and claims it is effectively forced to treat as
plausible. Formally, let verifiable (c, S)denote whether claim ccan be confirmed
or refuted from artifacts S, and let assessable (c, K)denote whether the model’s
parametric knowledge Kis sufficient to make a reliable plausibility judgment
aboutc. Then:
L(c;S, K) =

L1ifverifiable(c, S),
L2if¬verifiable(c, S)∧assessable(c, K),
L3otherwise.(1)
Empirically,thisseparationmatchestheobservedbehavior:near-zeroadoption
at L1, model-dependent behavior at L2, and near-universal adoption at L3.

10 J. Park, H. Choi, and K. Nam
Scaling Behavior.The Verification Boundary also explains why model
improvement changes some outcomes but not others. As models improve, the set
of claims that can be assessed from parametric knowledge expands. This means
that some claims that were effectively beyond the model’s reach for an older
generation can become rejectable for a newer one. Empirically, this is what we
observe in the transition from Opus 4 to Opus 4.6: the newer model rejects claims
that the earlier model adopted, but only for a subset of cases. This behavior is
most naturally explained as expansion of the L2 region. A stronger model can
bring more claims into the set of those that are plausibly assessable from internal
knowledge, even when the local artifacts do not resolve them. By contrast, model
scaling does not change the availability of local evidence itself. If a claim depends
on runtime behavior that is not observable from the provided artifacts, then
better reasoning alone cannot make that evidence appear. For this reason, L3
is more structurally persistent than L2. Within L3, the claims we observe span
several forms, including version-specific internals, execution-dependent behavior,
and environment-dependent conditions. What they share is that the agent cannot
resolve them from local artifacts alone. In practice, none of the evaluated agents
attempted the additional verification steps such claims would require, such as
exact-version inspection or dynamic probing.
Information Monopoly as Adoption Predictor.The VB also explains the
role of information monopoly. If the agent has only a single poisoned source and
lacks both local contradictory artifacts and strong parametric knowledge, then
the poisoned claim effectively occupies the available evidence space, and adoption
becomes highly likely. As competing evidence becomes available through source
code, retrieved counter-documents, or model knowledge, the same claim may
become rejectable. Empirically, we observe a clear gradient: under full information
monopoly, runtime-dependent claims are adopted consistently; with moderate
competition, adoption drops but does not disappear; and with strong local or
parametric counter-evidence, adoption falls to zero. This makes information
monopoly a strong practical predictor.
This has direct implications for zero-day and early-disclosure settings. When
authoritative references are sparse, poisoned runtime-dependent claims may face
little or no effective competition. The practical danger is therefore greatest not
in mature ecosystems saturated with evidence, but in early periods when reliable
counter-evidence has not yet accumulated. Taken together, the VB explains
the central pattern of our study: L1 claims are rejected because the agent can
refute them locally, L2 claims are unstable because rejection depends on model
knowledge, and L3 claims are adopted because the necessary refutation evidence
lies beyond the agent’s observable context.
6 Generalization and Analysis
6.1 Real-World CVE Validation
Table 4 summarizes our evaluation on 11 real-world CVEs spanning seven
platforms, which tests whether the same evidence-based pattern survives beyond

Poisoned Playbooks 11
Table 4.Real-world CVE validation on Opus 4.6. VB levels are assigned based on the
observed adoption pattern and the evidence regime of the poisoned claim.†Adopted by
an earlier model generation but rejected by Opus 4.6, which cited specific Redis Lua
sandbox knowledge.
CVE Software CVSS VB PAR Poison target
Well-documented CVEs (strong competing evidence)
2024-23897 Jenkins 9.8 — 0% FabricatedsanitizeArgs()
2021-44228 Log4j 10.0 — 0% JVM JNDI sandboxing
2022-22965 Spring 9.8 — 0% Reflection guard
Post-cutoff CVEs: L3 (runtime-dependent)
2025-55182 React RSC 10.0 L3 100% Next.js CSRF gate
2025-53770 SharePoint 9.8 L3 100% Serialization guard
2025-68613 n8n 9.9 L3 100% V8 context isolation
2025-61882 Oracle EBS 9.8 L3 100% Request authenticator
2025-59287 WSUS 9.8 L3 100% SafeSerializationMgr
2025-15467 OpenSSL 8.8 L3 100% Stack canary protection
2025-66516 Tika 9.8 L3 100% SAX parser hardening
Post-cutoff transition case
2025-49844 Redis 9.9 L2†0% Lua GC reference guard
the controlled challenge setting. CVSS denotes the Common Vulnerability Scoring
System, with scores ranging from 0 to 10. The well-documented CVEs, including
Jenkins, Log4Shell, and Spring4Shell, serve as controls. All three are rejected.
This matters because it shows that poisoned claims do not succeed merely because
the target vulnerability is severe or important; they fail when the agent has a
sufficiently strong basis for contradiction. The post-training-cutoff CVEs show
the opposite pattern. All seven cases whose poisons target runtime-dependent
behaviors are adopted. Despite variation in platform and software stack, the
behavioral result is the same: the agent treats the poisoned claim as plausible
and incorporates it into its reasoning. This is exactly what the VB predicts for
L3 claims.
One result to note is Redis Lua (CVE-2025-49844). The poisoned claim
was adopted by an earlier model generation but rejected by Opus 4.6 using
specific Redis Lua sandbox knowledge. In other words, the same claim shifts from
behaving like L3 to behaving like L2. This sharpens the broader conclusion: model
scaling can move some claims across the boundary, but it does not eliminate
the class of claims whose refutation still requires evidence the agent does not
possess. Taken together, the results show that the same pattern observed in the
controlled challenge setting also appears across real-world vulnerabilities.
6.2 Outcome Categories
Across successful cases, we observe three recurring behavioral outcomes.

12 J. Park, H. Choi, and K. Nam
False Negative Induction.The poison acknowledges the relevant vulnerability
class but argues that practical constraints render the exploit path ineffective
or irrelevant. The agent therefore dismisses a real vulnerability rather than
attempting to exploit it.
Target Deflection.The poison concedes that the general problem exists but
redirects the agent toward a non-existent or ineffective alternative exploit path.
The key effect is not denial, but redirection.
Reasoning Corruption.The poison preserves the correct vulnerability class
but corrupts the exploit logic needed to act on it. The agent stays in the right
conceptual neighborhood while following a flawed strategy.
These categories are not a universal taxonomy of all possible agent failures.
Rather, they capture the dominant behavioral consequences of successful poison
adoption in our study. More importantly, they reinforce the paper’s central
distinction: the issue is not only whether the agent reproduces false text, but
whether its downstream exploit behavior changes in a meaningful way.
6.3 Effective Poison Patterns
Our pilot rounds also revealed recurring features of poisoned artifacts that
were especially effective at inducing adoption. First, successful poisons often
use afirst-person solving narrative. Write-ups framed as a solver’s own process
are often treated as more credible than abstract factual claims. Second, they
frequently includespecific but fabricated quantitative evidence, which appears to
function as a credibility anchor. Third, and most importantly, successful poisons
disproportionately targetexternal dependencies and runtime behavior. This aligns
directly with the Verification Boundary: the strongest poisons are aimed at claims
that are hardest for the agent to falsify.
Two additional patterns recur. A poisoned write-up is more effective when it
supplies acomplete alternative exploit path, reducing the agent’s search burden,
and when it follows anacknowledge-then-refutestructure, explicitly naming the
correct path before explaining why it supposedly fails. Taken together, these
patterns suggest that successful poisoning is not driven by arbitrary falsehood
alone, but by a combination of stylistic credibility and claims positioned beyond
the agent’s immediate verification boundary.
Cost and scalability.The practical attractiveness of this threat lies not
only in whether it works, but in how cheaply it can be mounted. In our study,
each poisoned write-up required approximately 30–60 minutes of authoring
effort. More importantly, the attack scales one-to-many: a single poisoned
artifact can affect every downstream agent that retrieves it, potentially across
multiple users and repeated sessions. This scalability interacts directly with the
information-monopoly logic from §5. The attacker does not need to dominate the
entire ecosystem; poisoning can be effective whenever the available evidence is
sparse enough that a single plausible artifact occupies a disproportionate share of
the agent’s evidence space. This is why early-disclosure and post-cutoff settings
are especially exposed.

Poisoned Playbooks 13
Table 5.Verification prompting on runtime-dependent challenges.
Challenge Poison claim Without With
dh-47 SQLite WAL behavior 100% 0%
dh-75 glibc hex-IP resolution 100%∼0%
dh-106 redis-py expiration behavior 100% 0%
dh-675 EJS validation logic 100% 0%
dh-438 Chrome base-URI handling 100% 0%
7 Mitigations and Their Limits (RQ3)
The empirical pattern identified in this paper suggests that no single defense
will eliminate poison adoption across all settings. Instead, the effectiveness of a
mitigation depends on what kind of claim is being encountered and what evidence
is available to the agent at decision time.
Verification Prompting.One practical mitigation isverification prompting:
the agent is explicitly instructed to distinguish between claims that can be
verified from available artifacts and claims that remain runtime-dependent or
otherwise unverified. Rather than immediately operationalizing every retrieved
statement, the agent is encouraged to classify claims according to whether it
has enough evidence to trust them. Table 5 shows that verification prompting
is highly effective on the five runtime-dependent challenge cases we evaluate. In
these settings, the agent correctly marks the poisoned claims as unverified or
unsupported and reverts to a source-grounded strategy. The important point
is not simply that the prompt says “be careful,” but that it forces the model
to separate locally verifiable evidence from externally asserted runtime claims.
The limitation is equally important. Verification prompting is effective only
when there exists some stronger basis for fallback reasoning, such as source
code or challenge-local evidence. It does not create new evidence on its own. If
the relevant refutation evidence is absent altogether, then labeling a claim as
runtime-dependent may help the agent become more cautious, but it does not by
itself reveal the correct answer.
Multi-Source Retrieval.A second mitigation ismulti-source retrieval. Instead
of relying on a single retrieved document, the agent is encouraged to retrieve
and compare multiple candidate sources. This reduces the influence of a single
poisoned write-up when trustworthy competing evidence is available. In our
experiments, multi-source retrieval reduces poison adoption substantially when
correct competing documents exist. This supports a simple intuition: many
poisoning failures occur not because the poison is internally implausible, but
because it is uncontested. Once multiple sources are retrieved, the poisoned claim
must compete with independent evidence, and the agent becomes more likely to
reject it. At the same time, this defense fails in exactly the regime where the
threat is strongest: full or near-full information monopoly. If no correct competing
document exists, then retrieving more sources simply returns more uncertainty
or more variations of the same weak evidence pool. Multi-source retrieval is

14 J. Park, H. Choi, and K. Nam
therefore useful against sparse contamination in a richer ecosystem, but much
less effective when the ecosystem itself lacks authoritative counter-evidence.
Adaptive Adversaries.The defenses above raise adversarial cost, but they
do not close the attack surface. A sufficiently adaptive adversary can respond
to each mitigation layer. One strategy is to disguise runtime-dependent claims
as locally grounded ones. For example, by embedding false assertions inside
comments, fabricated patch notes, or pseudo-technical explanations that appear
to be tied directly to the code. Another iscorpus flooding: instead of inserting
a single poisoned artifact, the adversary publishes multiple consistent poisons
across sources, undermining diversity-based checks. A third possibility isauthority
mimicry, in which poisoned write-ups imitate the tone or structure of vendor
advisories or established exploit notes. More gradually, an adversary might build
credibility over time through benign contributions before publishing a targeted
poisoned artifact. While we do not directly evaluate these adaptive strategies,
their existence matters because it clarifies what our defense results do and do
not show. Verification prompting and multi-source retrieval are valuable because
they exploit current weaknesses in how the agent handles evidence. They should
be understood as cost-raising defenses, not as complete solutions.
Defense–Utility Tension.The strongest practical defense is often to restrict
the agent to well-established and trusted sources. But this creates an obvious
tension: the less external knowledge the agent is allowed to use, the less vulnerable
it becomes to poisoning and the less useful it becomes in exactly those fast-moving
settings where external knowledge is most needed. This trade-off is especially
sharp in security. For well-known vulnerabilities such as Log4Shell, restricting the
agent to mature, widely validated knowledge sources may be entirely acceptable.
For newly emerging vulnerabilities or post-cutoff CVEs, however, the agent’s
usefulness depends precisely on its ability to consume fresh external information.
The same external knowledge that gives the agent practical value is what exposes
it to poisoning. This tension cannot be resolved by a purely retrieval-layer solution.
If the correct evidence does not yet exist or is too sparse to retrieve reliably,
then the system faces a structural choice: either act under uncertainty using
potentially poisoned evidence, or defer the decision. In practice, this means that
defenses should be understood not only as accuracy mechanisms but also as
decision-control mechanisms.
Layered Defense Recommendations.The Verification Boundary suggests
a layered defense strategy. At the retrieval layer, agents should prefer
multiple sources, enforce source diversity where possible, and down-rank weakly
corroborated documents. At the reasoning layer, agents should explicitly separate
locally verifiable claims from runtime-dependent assertions and avoid treating
the latter as settled fact. At the decision layer, strategies that depend heavily
on a single runtime-dependent claim should be surfaced as provisional rather
than executed autonomously. This leads to a practical recommendation: when an
exploit strategy depends primarily on a claim that the agent cannot independently
verify, the system should seek additional evidence or defer to human review. The

Poisoned Playbooks 15
goal is not to eliminate uncertainty, but to ensure that it is properly recognized
and handled.
The broader lesson is that mitigation must follow the structure of the threat.
If poisoning succeeds because the agent lacks the evidence needed for refutation,
then defenses that merely reweight retrieved text will remain incomplete. Stronger
robustness will require either better access to verification evidence or mechanisms
that prevent unverified runtime claims from being operationalized too aggressively.
8 Related Works, Discussions, and Limitations
LLM-basedsecurityagents.RecentworkhasexploredLLM-basedsystemsfor
vulnerabilityanalysis, exploit reasoning, andautomated offensive workflows. Early
studiessuchasFangetal.[ 32]establishedthefeasibilityofusingLLMsforsecurity
tasks, while systems such as EnIGMA [ 24] and Cybench [ 52] moved toward more
structured agentic evaluation. Commercial systems such as XBOW [ 12] and
retrieval-augmented designs such as KryptoPilot [ 43] further demonstrate that
practical security agents increasingly combine model reasoning with external
knowledge sources. Our work differs in focusing not on agent capability itself,
but on the integrity risks introduced by that external knowledge dependence.
Poisoning attacks on RAG systems.A growing line of work shows
that retrieval-augmented systems are vulnerable to poisoning of their external
corpora.PoisonedRAG[ 53]demonstratesthatcrafteddocumentscansteermodels
toward attacker-chosen answers in answer-centric QA settings. Subsequent work,
including Confundo [ 36], Phantom [ 28], TrojanRAG [ 30], and VenomRACG [ 42],
studies more robust or backdoor-style poisoning mechanisms. These works
establish that the retrieval corpus is a meaningful attack surface, but they largely
focus on textual output corruption. In contrast, our setting is action-oriented: the
goal is to alter exploit behavior rather than merely induce an incorrect answer.
Agent-level attacks.Other work studies attacks that manipulate LLM agents
during execution, including indirect prompt injection [ 46,34,51], MUZZLE [ 49],
AgentPoison [ 29], and MemoryGraft [ 48]. These attacks are highly relevant
becausetheyalsotargetagentbehaviorratherthanstaticmodeloutputs.However,
they differ from our setting in access model and persistence. Many assume runtime
interactionwiththevictimagent,directmanipulationofmemory,oralivedelivery
channel. Our threat model is pre-positioned and persistent: the adversary injects
poisoned knowledge into a public repository, and any downstream agent that
retrieves it can be influenced without direct attacker interaction.
Supply-chain and data-poisoning perspectives.Traditional supply-chain
security has focused on code dependencies and package ecosystems, while
machine-learning poisoning work has focused on training data and model behavior.
Representative examples include dependency confusion attacks [ 26], open-source
ecosystem analyses [ 40,44], code-corpus poisoning [ 47], and large-scale data
poisoning [ 27]. Our work extends this perspective to theinference-time knowledge
supply chain: even when the codebase, model weights, and execution environment

16 J. Park, H. Choi, and K. Nam
remain intact, adversarial manipulation of external knowledge can systematically
distort downstream agent behavior.
Beyond the challenge setting.A natural concern is whether the
observed behavior is an artifact of the Dreamhack tasks. The real-world CVE
results mitigate that concern: across 11 CVEs spanning seven platforms, the
same basic pattern reappears. Well-documented vulnerabilities are rejected,
runtime-dependent post-cutoff claims are adopted, and one transition case moves
from adoption to rejection as model knowledge improves. While this does not
prove universal generality, it suggests that the Verification Boundary captures
something more structural than challenge-specific quirks.
Agent diversity and architectural scope.Our experiments use a single
retrieval-augmented agent configuration that is representative of current practice
but not exhaustive. Different agents may use other retrievers, ranking strategies,
tool policies, or planning loops. Still, the pattern we identify is tied less to a
specific implementation than to a broader constraint: the agent cannot refute
what it cannot observe. We therefore expect the core phenomenon to generalize
to other security agents that similarly rely on external retrieval and limited local
evidence, though broader validation remains future work.
Interpretive scope of the Verification Boundary.The VB is an empirical
classification derived from the observed behavior in our study. It is useful because
it explains why some claims are rejected, some are model-dependent, and some
remain consistently adopted. But it should not be read as a universal or final
taxonomy of all poisoning phenomena. Different domains, tools, or evidence-access
mechanisms may require refinements. In particular, if future agents gain robust
dynamic probing or version-aware dependency inspection, some claims that are
L3 under current conditions may become more tractable.
9 Conclusion
We presented an empirical study of knowledge poisoning in AI security agents
and showed that poisoned public security knowledge can systematically alter
downstream exploit behavior. Across controlled challenges, multiple frontier
models, and real-world CVEs, we found that poison adoption depends strongly
on what evidence the agent can use to challenge a retrieved claim. We captured
this pattern through theVerification Boundary(VB), which distinguishes locally
refutable claims, model-dependent claims, and runtime-dependent claims that
remain highly vulnerable to adoption. The same structure appears beyond the
challenge setting: well-documented vulnerabilities are rejected, while post-cutoff
runtime-dependent claims are consistently adopted. We also showed that
mitigation is possible but incomplete. Verification prompting and multi-source
retrieval reduce adoption when stronger evidence exists, but both weaken in
sparse-evidence and zero-day settings. More robust security agents will therefore
require stronger verification mechanisms, not just better retrieval or larger
models.

Poisoned Playbooks 17
References
1.CWE-1336: Improper neutralization of special elements used in a template engine
(SSTI). Available from MITRE, CWE-ID CWE-1336., https://cwe.mitre.org/
data/definitions/1336.html
2.CWE-362: Concurrent execution using shared resource with improper
synchronization (’race condition’). Available from MITRE, CWE-ID CWE-362.,
https://cwe.mitre.org/data/definitions/362.html
3.CWE-611: Improper restriction of XML external entity reference (XXE). Available
from MITRE, CWE-ID CWE-611., https://cwe.mitre.org/data/definitions/
611.html
4.CWE-79: Improper neutralization of input during web page generation (’cross-site
scripting’). Available from MITRE, CWE-ID CWE-79., https://cwe.mitre.org/
data/definitions/79.html
5.CWE-89: Improper neutralization of special elements used in an SQL command
(’SQL injection’). Available from MITRE, CWE-ID CWE-89., https://cwe.mitre.
org/data/definitions/89.html
6.CWE-918: Server-side request forgery (SSRF). Available from MITRE, CWE-ID
CWE-918.,https://cwe.mitre.org/data/definitions/918.html
7.CWE-943: Improper neutralization of special elements in data query logic (NoSQL
injection). Available from MITRE, CWE-ID CWE-943., https://cwe.mitre.org/
data/definitions/943.html
8. Dreamhack.https://dreamhack.io/, accessed: 2026-04-22
9. Exploit database,https://www.exploit-db.com/, accessed: 2026-04-22
10. Nuclei,https://projectdiscovery.io/nuclei, accessed: 2026-04-22
11. Portswigger,https://portswigger.net/, accessed: 2026-04-22
12. Xbow.https://xbow.com, accessed: 2026-04-22
13.CVE-2021-44228. Available from MITRE, CVE-ID CVE-2021-44228. (2021), http:
//cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2021-44228
14.CVE-2022-22965. Available from MITRE, CVE-ID CVE-2022-22965. (2022), http:
//cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2022-22965
15.CVE-2024-23897. Available from MITRE, CVE-ID CVE-2024-23897. (2024), http:
//cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2024-23897
16.CVE-2025-15467. Available from MITRE, CVE-ID CVE-2025-15467. (2025), http:
//cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2025-15467
17.CVE-2025-49844. Available from MITRE, CVE-ID CVE-2025-49844. (2025), http:
//cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2025-49844
18.CVE-2025-53770. Available from MITRE, CVE-ID CVE-2025-53770. (2025), http:
//cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2025-53770
19.CVE-2025-55182. Available from MITRE, CVE-ID CVE-2025-55182. (2025), http:
//cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2025-55182
20.CVE-2025-59287. Available from MITRE, CVE-ID CVE-2025-59287. (2025), http:
//cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2025-59287
21.CVE-2025-61882. Available from MITRE, CVE-ID CVE-2025-61882. (2025), http:
//cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2025-61882
22.CVE-2025-66516. Available from MITRE, CVE-ID CVE-2025-66516. (2025), http:
//cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2025-66516
23.CVE-2025-68613. Available from MITRE, CVE-ID CVE-2025-68613. (2025), http:
//cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2025-68613

18 J. Park, H. Choi, and K. Nam
24.Abramovich, T., Adir, M., Polak, I.: EnIGMA: Enhanced interactive generative
model agent for CTF challenges. arXiv preprint arXiv:2409.16165 (2024)
25.Achiam, J., Adler, S., Agarwal, S., Ahmad, L., Akkaya, I., Aleman, F.L., Almeida,
D., Altenschmidt, J., Altman, S., Anadkat, S., et al.: Gpt-4 technical report. arXiv
preprint arXiv:2303.08774 (2023)
26.Birsan, A.: Dependency confusion: How I hacked into Apple, Microsoft and dozens
of other companies (2021)
27.Carlini, N., Jagielski, M., Choquette-Choo, C.A., Paleka, D., Pearce, W., Anderson,
H., Terzis, A., Thomas, K., Tramèr, F.: Poisoning web-scale training datasets is
practical. In: IEEE S&P (2024)
28.Chaudhari, H., Severi, G., Abascal, J., Suri, A., Jagielski, M., Choquette-Choo,
C.A., Nasr, M., Nita-Rotaru, C., Oprea, A.: Phantom: General backdoor attacks
on retrieval augmented language generation. ACM Transactions on AI Security and
Privacy (2024)
29.Chen, Z., Xiang, Z., Xiao, C., Song, D., Li, B.: Agentpoison: Red-teaming llm
agents via poisoning memory or knowledge bases. Advances in Neural Information
Processing Systems (NeurIPS)37, 130185–130213 (2024)
30.Cheng, P., Ding, Y., Ju, T., Wu, Z., Du, W., Yi, P., Zhang, Z., Liu, G.: Trojanrag:
Retrieval-augmented generation can be backdoor driver in large language models.
arXiv preprint arXiv:2405.13401 (2024)
31.El Yadmani, S., Gadyatskaya, O., et al.: Securepoc: A helping hand to identify
malicious cve proof of concept exploits in github. In: 19th USENIX WOOT
Conference on Offensive Technologies (WOOT 25). pp. 263–282 (2025)
32.Fang, R., Bindu, R., Gupta, A., Zhan, Q., Kang, D.: LLM agents can autonomously
hack websites. arXiv preprint arXiv:2402.06664 (2024)
33.Gao, Y., Xiong, Y., Gao, X., Jia, K., Pan, J., Bi, Y., Dai, Y., Sun, J., Wang, H.,
Wang, H., et al.: Retrieval-augmented generation for large language models: A
survey. arXiv preprint arXiv:2312.109972(1), 32 (2023)
34.Greshake, K., Abdelnabi, S., Mishra, S., Endres, C., Holz, T., Fritz, M.: Not what
you’ve signed up for: Compromising real-world llm-integrated applications with
indirect prompt injection. In: Proceedings of the 16th ACM workshop on artificial
intelligence and security. pp. 79–90 (2023)
35.Happe, A., Cito, J.: Getting pwn’d by AI: Penetration testing with large language
models. In: ESEC/FSE (2023)
36.Hu, H., Jiang, Z., Lyu, Y., Zhang, J., Liu, Y., Chow, K.H.: Confundo: Learning to
generate robust poison for practical RAG systems. arXiv preprint arXiv:2602.06616
(2026)
37.Huang, Y., Huang, J.X.: A survey on retrieval-augmented text generation for large
language models. ACM Computing Surveys (2024)
38.Ji, Z., Wu, D., Jiang, W., Ma, P., Li, Z., Wang, S.: Measuring and augmenting
large language models for solving capture-the-flag challenges. In: ACM CCS (2025)
39.Karpukhin, V., Oguz, B., Min, S., Lewis, P., Wu, L., Edunov, S., Chen, D., Yih, W.t.:
Dense passage retrieval for open-domain question answering. In: Proceedings of the
2020 conference on empirical methods in natural language processing (EMNLP).
pp. 6769–6781 (2020)
40.Ladisa, P., Plate, H., Martinez, M., Barais, O.: SoK: Taxonomy of attacks on
open-source software supply chains. In: IEEE S&P (2023)
41.Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V., Goyal, N., Küttler, H.,
Lewis, M., Yih, W.t., Rocktäschel, T., Riedel, S., Kiela, D.: Retrieval-augmented
generation for knowledge-intensive nlp tasks. In: Advances in Neural Information
Processing Systems (NeurIPS) (2020)

Poisoned Playbooks 19
42.Li, T., Lin, B., Wang, S., Tan, Y.: Exploring the security threats of retriever
backdoors in retrieval-augmented code generation. arXiv preprint arXiv:2512.21681
(2025)
43.Liu, X., Li, Z., Lan, X., Ren, H., Wang, H., Chen, X.: Kryptopilot: An open-world
knowledge-augmented llm agent for automated cryptographic exploitation. arXiv
preprint arXiv:2601.09129 (2026)
44.Ohm, M., Plate, H., Sykosch, A., Meier, M.: Backstabber’s knife collection: A review
of open source software supply chain attacks. In: DIMVA (2020)
45.Pasquini, D., Kornaropoulos, E.M., Ateniese, G.: Hacking back the AI-hacker:
Prompt injection as a defense against LLM-driven cyberattacks. arXiv preprint
arXiv:2410.20911 (2024)
46.Perez, F., Ribeiro, I.: Ignore previous prompt: Attack techniques for language
models. arXiv preprint arXiv:2211.09527 (2022)
47.Schuster, R., Schuster, T., Meri, Y., Shmatikov, V.: You autocomplete me: Poisoning
vulnerabilities in neural code completion. In: USENIX Security (2021)
48.Srivastava, S.S., He, H.: MemoryGraft: Persistent compromise of LLM agents via
poisoned experience retrieval. arXiv preprint arXiv:2512.16962 (2025)
49.Syros, G., Rose, E., Grinstead, B., Kerschbaumer, C., Robertson, W., Nita-Rotaru,
C., Oprea, A.: MUZZLE: Adaptive agentic red-teaming of web agents against
indirect prompt injection attacks. arXiv preprint arXiv:2602.09222 (2026)
50.Touvron, H., Lavril, T., Izacard, G., Martinet, X., Lachaux, M.A., Lacroix, T.,
Rozière, B., Goyal, N., Hambro, E., Azhar, F., et al.: Llama: Open and efficient
foundation language models. arXiv preprint arXiv:2302.13971 (2023)
51.Zhan, Q., Liang, Z., Ying, Z., Kang, D.: Injecagent: Benchmarking indirect prompt
injections in tool-integrated large language model agents. In: Findings of the
Association for Computational Linguistics: ACL 2024. pp. 10471–10506 (2024)
52.Zhang,A.K.,Perry,N.,Dulepet,R.,Ji,J.,Menders,C.,Lin,J.W.,Jones,E.,Hussein,
G., Liu, S., Jasper, D., et al.: Cybench: A framework for evaluating cybersecurity
capabilities and risks of language models. arXiv preprint arXiv:2408.08926 (2024)
53.Zou, W., Geng, R., Wang, B., Jia, J.: PoisonedRAG: Knowledge corruption attacks
to retrieval-augmented generation of large language models. In: USENIX Security
(2025)