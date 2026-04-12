# Argus: Reorchestrating Static Analysis via a Multi-Agent Ensemble for Full-Chain Security Vulnerability Detection

**Authors**: Zi Liang, Qipeng Xie, Jun He, Bohuan Xue, Weizheng Wang, Yuandao Cai, Fei Luo, Boxian Zhang, Haibo Hu, Kaishun Wu

**Published**: 2026-04-08 03:18:51

**PDF URL**: [https://arxiv.org/pdf/2604.06633v1](https://arxiv.org/pdf/2604.06633v1)

## Abstract
Recent advancements in Large Language Models (LLMs) have sparked interest in their application to Static Application Security Testing (SAST), primarily due to their superior contextual reasoning capabilities compared to traditional symbolic or rule-based methods. However, existing LLM-based approaches typically attempt to replace human experts directly without integrating effectively with existing SAST tools. This lack of integration results in ineffectiveness, including high rates of false positives, hallucinations, limited reasoning depth, and excessive token usage, making them impractical for industrial deployment. To overcome these limitations, we present a paradigm shift that reorchestrates the SAST workflow from current LLM-assisted structure to a new LLM-centered workflow. We introduce Argus (Agentic and Retrieval-Augmented Guarding System), the first multi-agent framework designed specifically for vulnerability detection. Argus incorporates three key novelties: comprehensive supply chain analysis, collaborative multi-agent workflows, and the integration of state-of-the-art techniques such as Retrieval-Augmented Generation (RAG) and ReAct to minimize hallucinations and enhance reasoning. Extensive empirical evaluation demonstrates that Argus significantly outperforms existing methods by detecting a higher volume of true vulnerabilities while simultaneously reducing false positives and operational costs. Notably, Argus has identified several critical zero-day vulnerabilities with CVE assignments.

## Full Text


<!-- PDF content starts -->

Argus: Reorchestrating Static Analysis via a Multi-Agent Ensemble for
Full-Chain Security Vulnerability Detection
Zi Liang1*, Qipeng Xie2*, Jun He3†, Bohuan Xue3,
Weizheng Wang1,Yuandao Cai2,Fei Luo4,Boxian Zhang3,Haibo Hu1†,Kaishun Wu2
1The Hong Kong Polytechnic University,2HKUST
3SF Express,4Great Bay University
Correspondence:hejun@sf-express.com haibo.hu@polyu.edu.hk
Abstract
Recent advancements in Large Language Mod-
els (LLMs) have sparked interest in their ap-
plication to Static Application Security Testing
(SAST), primarily due to their superior contex-
tual reasoning capabilities compared to tradi-
tional symbolic or rule-based methods. How-
ever, existing LLM-based approaches typically
attempt to replace human experts directly with-
out integrating effectively with existing SAST
tools. This lack of integration results in inef-
fectiveness, including high rates of false posi-
tives, hallucinations, limited reasoning depth,
and excessive token usage, making them im-
practical for industrial deployment. To over-
come these limitations, we present a paradigm
shift that reorchestrates the SAST workflow
from current LLM-assisted structure to a new
LLM-centered workflow. We introduce Argus
(Agentic and Retrieval-Augmented Guarding
System), the first multi-agent framework de-
signed specifically for vulnerability detection.
Argus incorporates three key novelties: com-
prehensive supply chain analysis, collaborative
multi-agent workflows, and the integration of
state-of-the-art techniques such as Retrieval-
Augmented Generation (RAG) and ReAct to
minimize hallucinations and enhance reasoning.
Extensive empirical evaluation demonstrates
that Argus significantly outperforms existing
methods by detecting a higher volume of true
vulnerabilities while simultaneously reducing
false positives and operational costs. Notably,
Argus has identified several critical zero-day
vulnerabilities with CVE assignments.
1 Introduction
The integrity of modern digital ecosystems is under
critical threat from the escalating proliferation of
software vulnerabilities. In 2025, more than 50,000
Common Vulnerabilities and Exposures (CVEs)
*Equal contributions.
†Corresponding authors.were disclosed, marking a sharp rise from the pre-
vious year and signaling a rapidly expanding attack
surface (CVE Program Report). These vulnerabili-
ties result in catastrophic consequences for the fi-
nancial sector and society, as exemplified by global
outbreaks such as NotPetya (Aidan et al., 2017), the
Equifax breach (Zou et al., 2018), React2Shell (Re-
act, 2025), and Cloudflare, which inflicted almost
180 billion USD in damages or compromised the
personal data of over 143 million consumers.
Static Application Security Testing (SAST), as
one of the most effective strategies for vulnerabil-
ity detection, has thus emerged as a cornerstone
of secure software development (Li et al., 2023).
Many state-of-the-art SAST tools, including Cod-
eQL (Github CodeQL), Infer (Meta Infer), and
Snyk Code (Snyk Code), rely on static taint analy-
sis to trace untrusted data flows from sources to po-
tentially dangerous sinks. While these taint analy-
ses can effectively capture specific and well-known
vulnerability patterns with handcrafted symbolic
rules (Avgustinov et al., 2016), they lack the flexi-
bility to detect novel or system-specific flaws (Hin
et al., 2022; Kang et al., 2022), and remain funda-
mentally constrained by challenges such as high
false positive rates, limited scalability in complex
codebases, and insufficient semantic context. More
critically, they often miss real vulnerabilities due to
incomplete coverage of sources and sinks as well
as cracked data flows caused by insufficient support
for advanced language features, which together un-
dermine their practical effectiveness (Yuan et al.,
2025).
Consequently, LLM-enhanced vulnerability de-
tection has been proposed (Xia and Zhang, 2022;
Lemieux et al., 2023; Li et al., 2024; Cursor Bug-
Bot; CodeRabbit; Li et al., 2025; Ullah et al., 2025;
Liu et al., 2025; Guo et al., 2025). These methods
regard LLMs as vulnerability analysis experts and
incorporate them with SAST tools (e.g., CodeQL)
for more accurate sink discovery and flow con-
1arXiv:2604.06633v1  [cs.CR]  8 Apr 2026

struction. While they achieve significant improve-
ments compared with traditional SAST methods,
current incorporations between LLMs and SAST
tools have three significant challenges.First, these
methods remain limited by classical pipeline-based
symbolic analysis in which the reasoning depth
is restricted to single-pass inference, leading to
the inability to mining new vulnerabilities.Sec-
ond, the introduction of LLMs brings the risk of
hallucinations, which further leads to false posi-
tives.Third, current straightforward combinations
of LLMs with SAST tools fail to address reachabil-
ity challenges such as disrupted data flows. As a
result, ample evidence (Zhou et al., 2024b; Khare
et al., 2025; Guo et al., 2025) suggests that these
methods still struggle when faced with novel vul-
nerability types and restricted contexts, and typ-
ically perform poorly for practical vulnerability
detection tasks, which leaves a question:how to
design the LLM-SAST-tools incorporation that can
fully employ the potential of LLMs in vulnerability
detection?
To answer this question, we explore the possibil-
ity of proposing a new SAST framework with LLM
agents as the first-class citizens. In contrast to pre-
vious studies, which areSAST-tools-centeredand
LLM-assisted, we present a new framework that
isLLM-centered, where all other factors, includ-
ing SAST tools, code repository dependencies, and
existing issue descriptions, serve as supplemental
information for the LLM agent system.
Following this paradigm, we propose Ar-
gus ( Agentic and Retrieval-augmented Guarding
System), a novel SAST framework for automatic
vulnerability detection. Argus reorchestrates the
SAST workflow with the following three signifi-
cant improvements:i) Full Supply Chain Analysis.
Instead of regarding the code repository as an “iso-
lated silo”, we combine it with its dependencies
for a more practical vulnerability evaluation.ii)
Multi-Agent Collaboration.We decouple the SAST
pipeline into different modules and design various
agents for them, including dependency scanning,
information collecting, PoC (Proof of Concept)
generating, data flow scanning, and data flow re-
viewing. All these agents provide a more elegant
implementation of vulnerability detection.iii) In-
troducing New Techniques.We introduce the state-
of-the-art agentic strategies to address the difficul-
ties of LLM-based SAST analysis. For instance, we
apply RAG (Retrieval-Augmented Generation) to
collect community vulnerability information fromdependencies and code repositories, enabling more
comprehensive and accurate reports and sink identi-
fication. We utilize ReAct (Yao et al., 2023) to meet
the long-term reasoning requirements in data flow
review and PoC generation. Through this first-time
integrated design, Argus preserves the precision
of symbolic analysis while harnessing the deep
contextual reasoning of LLMs, enabling a more
general, adaptable, and robust approach to iden-
tifying sophisticated vulnerabilities in real-world
software.
Our contributions can be summarized as follows:
•We propose Argus, a new framework for static
code vulnerability analysis. To the best of our
knowledge, it is the first LLM-centered SAST
framework. It is also the first framework that
considers the supply chain and introduces Re-
Act and RAG strategies.
•We comprehensively evaluate Argus on pop-
ular industry-level open-source code reposi-
tories with tens of millions of lines of source
code and tens of thousands of GitHub stars.
Extensive empirical results show that Argus
exhibits lower token consumption than exist-
ing methods and can mine more new vulnera-
bilities. Using Argus, we discovered potential
zero-day vulnerabilities in these thoroughly
evaluated codebases and these vulnerabilities
have been assigned with CVE numbers.
•We will release our framework as a tool to
benefit the security of industry and academia.
2 Related Work
LLM-based Vulnerability Detection & Discov-
ery.Recent studies (Ullah et al., 2025; Liu et al.,
2025; Li et al., 2025; Guo et al., 2025) have demon-
strated that LLM agents are increasingly promis-
ing for achieving more precise automatic vulner-
ability discovery. On the one hand, preliminary
research shows that LLMs can directly identify
common vulnerability patterns (e.g., CWE cate-
gories) from source code, especially when aug-
mented with contextual metadata such as commit
messages or issue descriptions (Guo et al., 2025).
On the other hand, LLMs can be employed to mine
unknown sources or sinks within the source code,
enabling more accurate static taint analysis for vul-
nerability detection (Li et al., 2025; Guo et al.,
2025). Systematic evaluations have also been pro-
vided (Mathews et al., 2024) suggesting that even
2

from flask import Flask, request
import subprocess
app = Flask(__name__)
@app.route(' /ping', methods=[' GET'])
def ping_host ():
    ip = request.args.get(' ip')
    if not ip or ip.strip() == "":
        return "Please pass the ip parameter, for example: /ping?ip=127.0.0.1 "
    command = f" ping -c 4 {ip} "  
    try:
        result = subprocess.call(command, shell=True)  
        if result == 0:
            return f"{command}<br/>ping {ip} completed "
        else:
            return f"Command execution failed: {command}<br/> "
    except Exception as e:
        return f"Execution exception: {str(e)} "Source
1
2
SinkFigure 1: A toy Python example of vulnerability detec-
tion. The function subprocess.call(·, shell=True)
acts as thesink, which may receive injected malicious
input from thesource ipand cause it to be executed.
Data flow of this vulnerability is represented by the
connecting purple arrows.
prompt-engineering-based LLMs can surprisingly
improve the detection rate for Android vulnerabili-
ties. However, as highlighted by recent literature
reviews (Zhou et al., 2024b; Khare et al., 2025;
Zhou et al., 2025), while LLMs show promise in
both detection and repair, their performance re-
mains highly sensitive to input formulation, code
context length, and the quality of labeled vulnera-
bility data.
Techniques of LLM Agents.Recent research
on LLM-based agents has emphasized structured
reasoning (Yao et al., 2023; Du et al., 2023; Zhou
et al., 2024a), tool integration, and memory mech-
anisms to support complex task execution. The
ReAct framework (Yao et al., 2023), which in-
terleaves reasoning steps with action executions
(e.g., API calls), has become a standard archi-
tecture for agentic behavior, enabling agents to
dynamically reason and reflect on observations.
This paradigm has been extended through retrieval-
augmented generation (RAG) (Lewis et al., 2024;
Jiang et al., 2023; Salemi and Zamani, 2024), al-
lowing agents to access external knowledge bases
during inference to ground responses in up-to-
date or domain-specific information. More recent
works (Wang et al., 2025) explore modular agent
designs. For instance, Wang et al. (2025) proposes
a planner-executor architecture that separates high-
level task decomposition from low-level tool in-
vocation, significantly improving success rates on
multi-step reasoning benchmarks. Additionally,
empirical studies have shown that chain-of-thought
(CoT) (Wei et al., 2022) prompting and its variants
(e.g., self-reflection, step-back prompting) consis-
tently enhance agent reliability across diverse envi-ronments (Zhu et al., 2025). These developments
highlight a methodological shift from monolithic
prompting toward compositional, tool-augmented
agent systems that balance reasoning fidelity with
external interaction.
3 Methodology
3.1 Problem Definition
In this section, we begin with the fundamentals of
current static analysis for security vulnerability de-
tection. A toy example of these concepts is shown
in Figure 1.
Taint & Content.Ataintrefers to a variable or ob-
ject that stores unsafe or injected content occurring
in the code. The taint is typically held within a vari-
able or container of the program, which we refer to
as acontent node. A content node can be a variable,
an array, a list, a key or value in a hashmap, or a
member of an object. Considering the properties
of these content nodes, we emphasize two special
nodes for vulnerability detection:i)Sources, which
indicate unauthorized user interface inputs in the
source code that might be maliciously exploited by
an adversary for security attacks. These can include
HTTP request parameters, text or files uploaded by
users, or inputs from environment variables and
databases.ii)Sinks, which refer to executed func-
tions in the source code that might involve danger-
ous or sensitive operations, such as SQL executions
or system command executions.
Access Paths.An access path records how content
propagates between different node types.
Definition 3.1 (Data Flow) Given a source code
project S={S i}i=1,...,N Sconsisting of NSsource
code files, where each file Siis a text string, let
VSandESdenote the sets of content nodes and
access paths in S, respectively. A data flow from a
source node to a sink node is defined as a sequence
of logical edge triples:
F(v1
i,vn
k)
S=
(v1
i, e1
j, v1
k)1, . . . ,(vn
i, en
j, vn
k)n
,
(1)
where n < N Fis the length of the flow, vm
i, vm
k∈
VSandem
j∈ESfor all m∈ {1, . . . , n} , and the
flow satisfies the continuity condition vm−1
k=vm
i
form= 2, . . . , n . The data flow graph of Sis the
set of all such data flows, i.e., F={F(v1
i,vn
k)
S|
v1
i, vn
k∈VS}.
Given Definition 3.1 and the notions of sources
and sinks, we can define the vulnerability and the
corresponding detection task as follows:
3

Vulnerability Database Third-Party Dependencies
ArgusDependency Scanning
and Code AnalysisCVE Retrieval Agent PoC Generation AgentCodeQL  EngineSink Scanning Forward Scanning Backward Scanning
Vuln. Verification CodeQL Rule Agent Flow Review AgentData SourceResults
Vuln. Flow
Vuln. V erification
Vuln. Report
Advisory Database
PublicCMSorg.apache.commons...
org.apache.poi...
com.mysql...
org.springframework...
redis.clients...
...
Title: Please fix log4j2
to version ...
Severity : UNKNOWN
Source : GitHub
Discussion (Stage2 -
Community)
Relevance Score:  0.4
...
Sink
New SinksControl Flow?
Exception Handle?
Data verification?
Sanitization?Code RepositoryFigure 2: The overall workflow of the Argus framework. Given a code repository, Argus first parses it with its
third-party dependencies and retrieves potential issues or vulnerabilities associated with them. For suspected
vulnerabilities, Argus generates proofs of concept (PoCs) as well as corresponding repairs. It then integrates with
CodeQL for sink identification and employs ourRe3(Retrieval, Recursion, and Review) method, which combines
forward and backward scanning to achieve comprehensive data flow extraction. Finally, the framework produces a
detailed report containing all detected vulnerabilities, along with their data flows and verification details, for human
review. All information shown in this figure is derived from actual execution results of Argus.
Definition 3.2 (Vulnerability Detection) Given
a source node vso∈ Vsoand a sink node vsi∈ Vsi,
a vulnerability in Sis defined as a reachable
data flow from vsotovsi, denoted as F(vso,vsi)
S.
Correspondingly, the vulnerability detection task
consists of two steps:i)constructing the source
setVsoand the sink set Vsi, andii)mining all
reachable data flows between them to obtain the
vulnerability set:
Fvul
S={F(vso,vsi)
S|vso∈ Vso, vsi∈ Vsi}.(2)
Under the research scope of Definition 3.2,
current SAST methods employ cybersecurity
databases to construct source and sink sets and
rely on rule-based engines (e.g., CodeQL) for au-
tomatic data flow discovery. Although LLMs have
been employed in some steps of this task (e.g.,
sink set discovery), a unified, agentic, and in-depth
integration of LLMs into vulnerability detection
remains largely uninvestigated, which significantly
limits the potential of AI-enhanced vulnerability
discovery. Can LLMs effectively reveal existing
CVEs? Are they able to mine zero-day vulnerabili-
ties? How can we mitigate the inherent drawbacks
of both LLMs and traditional SAST tools? We
aim to answer these questions by proposing a new
framework in the following subsections.
3.2 Argus: Agentic and Retrieval-Augmented
Guarding System
As shown in Figure 2, the Argus framework con-
sists of two core components:RAG-enhanced full
1. Original Vulnerability Information
  Title: Apache POI OO XML V ulnerable to I mproper I nput V alidation in OO XML Fi le Parsing
  Description : Improper I nput V alidation vulner ability in Apache POI. The issue af fects the parsing of OO XML
format fi les like xlsx...
  Component : org.apache.poi:poi-oo xml
  Time : 2025-12- XX 21:31:51
  Dependency : org.apache.poi:poi-oo xml
  Original Fi le: stage1_Apache POI OO XML V ulnerable to I mproper I nput V alidation in OO XML Fi le Parsing.txt
  Processing Stage : STAGE1 - A uthori tative Source
  CVE ID : CVE-2025-31672
2. Quality Check
  ✅ Check R esult: Passed
  Score: 8/8
  Title Qual ity: ✅ (Speci fic vulner ability information)
  Description Qual ity: ✅ ...
3. Search Parameters
  Component Usage Query : "org.apache.poi poi-oo xml usage example tutorial"
  Security Analysis Query : "Apache POI OO XML V ulnerable to I mproper ..."
4. Search Results
  4 relevant documents f ound:
  [Document 1] Compr ehensiv e Apache POI T utorial: Ex cel File Handl ing ... - S core: 4
  [Document 2] I mproper I nput V alidation in Apache POI OO XML P arsing ... - S core: 33
...Figure 3: A practical example of the structured in-
formation summerized by our RAG agent. It sum-
marizes all necessary information for the locating
of potential sinks given a target dependency (e.g.,
org.apache.poi:poi-ooxml).
supply chain sink analysisandRe3-based data flow
analysis. Both components tightly integrate LLMs
with existing vulnerability resources (e.g., NVD)
and static analysis tools (e.g., CodeQL), resulting
in more effective and efficient vulnerability detec-
tion.
3.2.1 Sinks Scanning: RAG-Enhanced Full
Supply Chain Analysis
The first step in identifying vulnerable data flows
is sink scanning. Once sinks are identified, we
can search for all potential unsafe data flow candi-
dates. Unfortunately, current vulnerability detec-
4

Vulnerability Situation
  Risk Lev el: UNKNOWN
  Priority: UNKNOWN
  Vulnerable Code Examples : 1
  Secur e Code Examples : 2
  Best Practice Examples : 0
  Attack Scenar io Examples : 2
Vulnerable Code Examples
  Example 1
// Vulnerable snippet simulating what happens in POI versions < 5.4.0:
ZipFile zipFile = new ZipFile(file);
Enumeration <? extends ZipEntry > entries = zipFile. entries();
while (entries. hasMoreElements ()) {
    ZipEntry  entry = entries. nextElement ();
    InputStream  stream = zipFile. getInputStream (entry);
    // No check if entry.getName() has appeared before!
    // Proceed processing XML / document parts blindly.
}
Secur e Code Examples
  Example 1
// Secure version of ZIP entry reading with duplicate entry name detection
ZipFile zipFile = new ZipFile(file);
Enumeration <? extends ZipEntry > entries = zipFile. entries();
Set<String> seenEntries = new HashSet<>();
...
Influence Analysis
Apache POI's `poi-ooxml` library versions prior to ...Figure 4: Example of PoC generation and vulnerability
verification. The POC agent will think about the possi-
bility of writing proof-of-concept example code which
can successfully exploit the codebase. In this way, we
can ensure the correctness of sinks and start our data
flow analysis in the next step.
tion strategies (Ullah et al., 2025; Liu et al., 2025;
Li et al., 2025) exhibit two significant drawbacks:
i)they directly employ LLMs for sink discovery,
resulting in low scanning efficiency, low precision,
and low recall, which significantly impacts the ef-
fectiveness of subsequent vulnerability detection
steps;ii)they focus merely on the security of the
project source code Sitself, severely neglecting
its dependencies (i.e., the supply chain). However,
these dependencies across different versions may
contain known vulnerabilities or expose new, un-
expected vulnerabilities or sinks due to improper
sanitization. These drawbacks render automatic
scanning unsuitable for industrial-level vulnerabil-
ity detection.
To address these issues and facilitate subsequent
static analysis for vulnerability detection, we de-
sign a dedicated component that parses project de-
pendencies and performs retrieval-augmented gen-
eration to mine all relevant vulnerability types.
Dependencies Parsing.Argus first parses the
project management file (e.g., pom.xml in Java
projects) to collect all active dependencies spec-
ified in the configuration files. It then looks up
their actual usage in the source code. The detailed
information of these dependencies is passed to the
next step for further analysis.
Retrieval.Given the identified dependencies, RAG
agents in Argus perform parallel searches for vul-nerability information from the Internet. We design
two types of search resources for Argus:i) Author-
itative Sources.We select the National Vulnera-
bility Database (NVD), Open Source Vulnerabili-
ties (OSV), GitHub Security Advisories (GHSA),
and Snyk Vulnerability Database (Snyk) as offi-
cial vulnerability sources. For each dependency,
we query these databases and extract vulnerabil-
ity information into structured objects (as detailed
in Figure 3), including descriptions, severities, af-
fected versions, and CVE IDs (if available).ii)
Community Sources.We employ a hierarchical ap-
proach to collect community-reported vulnerability
information for the target dependency. Specifically,
we examine vulnerability-related issues from both
the dependency’s primary repository and its derived
(forked) repositories, prioritizing those explicitly
linked to CVEs or GHSAs. We define three indica-
tors to assess the quality of retrieved vulnerabilities:
relevance score, credibility score, and the content
quality score, as detailed in Appendix A.
Proof of Concepts (PoC) of Vulnerabilities.
Given the retrieved vulnerabilities, Argus employs
a dedicated PoC agent to iteratively verify their ex-
ploitability. Specifically, the PoC agent follows the
ReAct paradigm (Yao et al., 2023): it first prompts
the LLM to re-describe the vulnerability details,
then sequentially reasons about the root cause, code
patterns, and potential attack scenarios. Subse-
quently, the agent generates source code that trig-
gers the vulnerability, along with a corresponding
patch and additional explanatory information. Fig-
ure 4 provides a toy example of this process. In this
way, Argus enables effective and efficient scanning
of known vulnerabilities in a code repository, com-
plete with verifiable PoCs. Once verified, the re-
sulting vulnerabilities, repairs, and detailed reports
can be directly applied to the repository, offering
greater comprehensiveness than previous vulner-
ability reproduction methods (Ullah et al., 2025).
Moreover, these verified vulnerabilities and their
PoCs serve as valuable context to inspire Argus
in discovering additional vulnerabilities, including
zero-day ones, as detailed in the next section.
3.2.2 Data Flow Analysis: Retrieval,
Recursion, and Review
Given candidate vulnerability descriptions and their
corresponding sinks, the final step in vulnerability
detection is to recover and construct the complete
data flow graph that reflects how taint propagates
from sources to these sinks. A path is confirmed
5

as a verified vulnerability if it reaches a sink with-
out encountering effective sanitization. However,
tracking data flows to sinks suffers from the well-
knownstealthy interruptions problem: mainstream
static analysis tools, such as CodeQL, often fail to
resolve data connections arising from advanced lan-
guage features, includingreflection,multi-thread
interactions, andpointer aliasing. Unfortunately,
directly applying LLMs to this task provides lim-
ited benefit, as the complexity of data flow analysis
for each sink far exceeds the reasoning capacity of
a single LLM inference. To address this challenge,
we propose a novel workflow namedRe3(Retrieval,
Recursion, and Review) that enables more accurate
and robust data flow analysis.
•Retrieval.Similar to existing LLM-based vulner-
ability detection approaches (Li et al., 2025; Liu
et al., 2025), we employ CodeQL to search for all
candidate paths from sources to each targeted sink.
If data flows to the sink are successfully identified,
they are stored for the subsequentReviewstep.
Otherwise, the sink is forwarded to theRecursion
step.
•Recursion.For sinks that cannot be reached
through direct forward analysis, we introduce a
backward-forward mechanism to bridge interrup-
tions. Specifically, we iteratively analyze the up-
stream function calls of the sink until the reverse
traversal terminates. This process constructs a tree
rooted at the original sink, with leaves correspond-
ing to the uppermost function calls. We then treat
these leaves as new surrogate sinks and perform
forward data flow analysis using CodeQL. The re-
sulting forward flows are connected to the back-
ward tree paths to form candidate vulnerable data
flows. Note that not all data flows produced in this
step are guaranteed to be valid, necessitating a final
Reviewstep.
•Review.At the end of the data flow analysis, we
employ a dedicated LLM agent to audit the cor-
rectness of candidate vulnerabilities through three
sequential steps:i) End-to-End Reachability Anal-
ysis, in which the LLM is prompted to determine
whether control flow structures (e.g., if/else ,
switch ), exception handling (e.g., try/catch ), or
data validation mechanisms interrupt or alter the
original data flow to the sink.ii) Hop-by-Hop Anal-
ysis, which conducts a more detailed examination
of the entire data flow, verifying at each step how
the taint enters a code block, the content and access
path involved, and most importantly, whether any
form of input validation, sanitization, encoding, orRepository Stars Commits Forks LoC Issues
PublicCMS 2.1k 3,479 828 335k 1
JeecgBoot 44.9k2,13715.7k642k 19
Ruoyi 7.9k 1,942 2.2k 124k 16
JSPWiki 111 9,742 102 161k 12
DataGear 1.6k 4,574 367 190k 1
Yudao-Cloud 18.4k 2,614 4.6k 230k 1
KeyCloak 32k29,8317.9k841k 2.1k
Table 1: Details of the evaluation codebases. This table
presents the GitHub stars, forks, commits, Lines of
Code (LoC), and the number of open issues for each
repository.
type casting could neutralize the propagation. Fol-
lowing this analysis, we generate an illustrated hop-
by-hop breakdown of the data flow.iii) Document
Export, in which all data flow results, verification
outcomes, and a structured vulnerability report are
exported for final human expert review.
4 Experiments
4.1 Settings
Evaluation Environments.We take Java code-
bases vulnerability detection as our evaluation ex-
ample, and our method can be used in other lan-
guages by implementing new dependency analy-
sis agents and new CodeQL query templates. We
evaluate the SAST efficiency and effectiveness
of Argus on modern and industrial-level code-
bases. Specifically, we select seven popular open-
source codebases, including PublicCMS (Public
CMS), JeecgBoot (JeecgBoot), Rouyi (Rouyi), JSP-
Wiki (JSPWiki), DataGear (DataGear), Yudao (Yu-
dao), and KeyCloak (KeyCloak), as detailed in Ta-
ble 1. All seven codebases contain at least 100,000
lines of code, with almost 10k GitHub stars and
commits, demonstrating that they are practical and
suitable for evaluating realistic vulnerability detec-
tion performance.
Baselines & Metrics.We employ IRIS (Li et al.,
2025), the only Java-oriented vulnerability detec-
tion framework, and CodeQL (Github CodeQL),
the well-known SAST tool, as our evaluation base-
lines. Regarding metrics, we utilize the detected
counts of vulnerabilities and sinks to reflect the
effectiveness of methods, and measure token con-
sumption for computational efficiency.
4.2 Vulnerability Detection
We begin our realistic vulnerability detection eval-
uation with sink discovery, and then proceed to an
end-to-end SAST evaluation.
6

RepositoryCodeQL IRIS Argus (Claude-Sonnet-3.5) Argus (Claude-Sonnet-4.5)
# Sinks # Vuln. # Sinks # Vuln. # Sinks # Vuln. # Sinks # Vuln.
PublicCMS - 0 5840 0 8 6 8 6
JeecgBoot - 0 8350 0 9 3 9 3
Ruoyi - 0 980 0 3 3 3 3
JSPWiki - 0 1828 1 7 5 7 5
DataGear - 0 237 0 11 3 11 3
Yudao-Cloud - 0 2184 0 9 1 9 1
KeyCloak - 0 13962 0 19 2 19 2
Table 2: End-to-End Vulnerability Detection Evaluation. Argus significantly outperforms CodeQL in traditional
vulnerability detection. IRIS fails in realistic SAST scenarios due to the lack of required CWE information.
RepositoryArgus IRIS
Tokens Vuln. Tokens Vuln.
PublicCMS 194,538 6 35,500 0
JeecgBoot 254,392 3 42,400 0
Ruoyi 32,836 3 22,300 0
JSPWiki 67,605 5 24,638 1
DataGear 9,301 3 20,300 0
Yudao-Cloud 85,661 1 25,600 0
KeyCloak 442,095 2 57,641 0
Table 3: LLM API token consumption and vuln. discov-
ery comparison.
4.2.1 Token Consumption & Vulnerability
Detection Evaluation
We first compare the LLM API token consumption
with the corresponding number of discovered harm-
ful vulnerabilities. As shown in Table 3, our token
consumption is affordable. Benefiting from the pro-
posed RAG-enhanced mechanism, Argus incurs a
little bit higher token consumption compared to
the baselines, while identifying much more poten-
tial vulnerabilities in realistic codebases. As these
codebases are actively updated, in most cases the
baseline fail to detect any vulnerabilities, leading
to significant false positives in SAST.
4.2.2 End-To-End Evaluation
We further compare the automatic vulnerability
detection capabilities of Argus with our baselines
in realistic end-to-end scenarios. Specifically, we
count the number of sinks and vulnerabilities for
CodeQL, IRIS, and our Argus. For Argus, we
employ two versions of the LLM APIs to test the
sensitivity of the framework to different LLMs.
The results are shown in Table 2.
As shown in Table 2, Argus achieves a con-
sistently higher number of sinks compared with
all baselines, and a significantly higher number
of detected vulnerabilities, indicating substantial
progress from traditional SAST tools and LLM-
assisted analysis to LLM-centered automatic and# Vuln./# Sinks Bare Argus Extra with RAG
PublicCMS 5/7 1/1
JeecgBoot 3/9 0/0
Ruoyi 3/3 0/0
JSPWiki 5/7 0/0
DataGear 3/10 3/1
Yudao-Cloud 0/0 0/0
KeyCloak 2/17 0/2
Table 4: Vulnerabilities resulting from original sink
detection and extra sinks detected by RAG agents.
end-to-end vulnerability detection.
Table 2 can also be regarded as a preliminary
ablation study. When comparing the sink numbers
from Argus and CodeQL, we can observe the con-
tribution of extra sinks proposed by our RAG and
POC agents. We can infer that Argus is capable of
mining more “stealthy” and unknown sinks than
the static CodeQL. Moreover, when we compare
the final reported vulnerabilities, surprisingly, a sig-
nificant improvement can be observed, typically
increasing from zero to nearly ten vulnerabilities.
This demonstrates that ourRe3can effectively trace
all potential data flows towards the target sinks.
If we further track which detected sinks these
vulnerabilities originate from, we can obtain the
ablation results shown in Table 4. As indicated in
Table 4, sinks found by our RAG agents result in
significantly more vulnerabilities compared with
the sinks found by static tools (i.e., bare Argus),
indicating that the extra sinks derived from the
community are more valuable for detection.
5 Case Studies
5.1 Case I: How Does Argus “Recycle”
Previous Outdated Vulnerabilities?
Our first case study is derived from the experiments
scanning the DataGear repository. As shown
in Figure 5, we observe that the RAG agents
7

org.datagear .persistence.support.ConversionSqlParamV alueMapper .javaorg.datagear .persistence.support.DefaultPersistenceManager .javadatagear/web/controller/DataController .java
buildUniqueRecordCondition(...)
mapToSqlParamValue(...)public String view  (... ) public String edit (... )
public String execute (... )public ...<...> delete (...)
public Row get (...)
evaluateVariableExpression (...)public int delete (...)New Flow A New Flow B Expired Flow
Sanitized!
...Sources
SinkFigure 5: Examples of vulnerable data flows in
DataGear detected by Argus.
in Argus first retrieve a previous vulnerability
named CVE-2024-37759 . Through this, it iden-
tifies a sink evaluateVariableExpression at
org.datagear...Conversion...ValueMapper .
While the original vulnerable data flow has been
fixed, ourRe3identifies two new data flows
that pass our manual review, i.e., they are still
exploitable and have not been discovered before.
Moreover, by analyzing the attributes of the new
vulnerabilities, we find several interesting points:I)
The new data flows reuse most of the access paths
of the previous fixed vulnerabilities, suggesting that
reviewing vulnerabilities heuristically by human
experts would lead to high false positives as many
other candidate paths could be neglected.II) Argus
can provide more comprehensive vulnerability
scanningvia LLM reasoning and multi-agent
collaboration.III) Even those vulnerabilities which
have been fixed remain valuable,as they highlight
the necessity for more in-depth static analysis
regarding the sanitization of injected commands.
5.2 Case II: Even Incorrect Sinks Can Inspire
Argus to Search for New Vulnerabilities
Our second case is derived from experiments
onPublicCMS , where our RAG agents discover
a CVE ( CVE-2025-31672 ) related to its depen-
dency org.apache.poi:poi-ooxml . However,
the POC agents suspect that the sink could
beDocumentBuilderFactory.newInstance or
com...Basic...TypeValidator.builder , nei-
ther of which is the correct sink capable of trig-
gering XSS or XXE injection attacks. This discrep-
ancy arises because the two suspected sinks arerelevant to the dangerous behaviors associated with
this CVE but are not the exact sinks.
Consequently, when the previous agents provide
the sink newInstance , the forward search of Cod-
eQL cannot identify any explicit connection from
sources to it, leading to a failed vulnerability de-
tection. However, in the second step of ourRe3,
the true sink DocToHtmlUtils.excelToHtml , an
upstream node of newInstance marked as a new
potential sink by our method, is introduced for
the data flow search, thus identifying the correct
vulnerability. This case demonstrates that the ini-
tial incorrect sinks play the role of effectiveseeds,
which can successfully guide the discovery of the
true sinks.
6 Conclusion
In this paper, we explore how to construct an LLM-
agent-centered static application security testing
method for the vulnerability detection of modern
software. Unlike previous studies, which merely
employ LLMs as a component for specific vulner-
ability detection tasks, we reorganize the entire
detection procedure with our new framework, Ar-
gus. As a multi-agent system incorporating RAG
techniques, Argus effectively collects existing in-
formation about the repository and dependencies,
and employs a new reasoning procedure for vul-
nerability verification and data flow mining. Ex-
tensive experiments on realistic open-source code-
bases demonstrate that Argus can automatically
mine significantly more new zero-day vulnerabili-
ties compared to existing methods.
7 Limitations and Future Work
While our method achieves superior performance in
vulnerability detection, several limitations remain
in this study. On the one hand, Argus focuses ex-
clusively on static taint analysis and does not incor-
porate dynamic vulnerability detection techniques,
such as fuzzing. Integrating Argus with advanced
fuzzing strategies could enable more comprehen-
sive and powerful vulnerability detection in the
future. On the other hand, despite our comprehen-
sive agentic system design, numerous LLM oppor-
tunities exist to further enhance the effectiveness
of the SAST system. For instance, the multi-agent
framework could be fine-tuned using reinforcement
learning in a vulnerability detection environment,
with rewards based on sink discovery accuracy and
data flow completeness.
8

Ethical Considerations
Our Argus, leveraging large language models for
vulnerability detection, was developed with the pri-
mary objective of bolstering cybersecurity defenses.
Although there is a slight risk that it could be ex-
ploited by malicious actors to uncover vulnerabil-
ities, its predominant application is anticipated in
conducting thorough security audits of code repos-
itories, thereby substantially mitigating the likeli-
hood of successful cyber attacks. Moreover, the
disclosure of zero-day vulnerabilities underscores
the potential for this concise and lightweight ap-
proach to be broadly adopted for protective pur-
poses, outweighing any potential misuse.
References
Jagmeet Singh Aidan, Harsh Kumar Verma, and
Lalit Kumar Awasthi. 2017. Comprehensive survey
on petya ransomware attack. In2017 International
conference on next generation computing and infor-
mation systems (ICNGCIS), pages 122–125. IEEE.
Pavel Avgustinov, Oege De Moor, Michael Peyton
Jones, and Max Schäfer. 2016. Ql: Object-
oriented queries on relational data. In30th Eu-
ropean Conference on Object-Oriented Program-
ming (ECOOP 2016), pages 2–1. Schloss Dagstuhl–
Leibniz-Zentrum für Informatik.
CodeRabbit. 2025. Https://www.coderabbit.ai/.
Cursor BugBot. 2025. [link].
CVE Program Report. 2025. [link].
DataGear. 2025. Https://github.com/datageartech/datagear.
Yilun Du, Shuang Li, Antonio Torralba, Joshua B Tenen-
baum, and Igor Mordatch. 2023. Improving factual-
ity and reasoning in language models through multia-
gent debate. InForty-first International Conference
on Machine Learning.
Github CodeQL. 2025. Https://codeql.github.com/.
Jinyao Guo, Chengpeng Wang, Xiangzhe Xu, Zian
Su, and Xiangyu Zhang. 2025. Repoaudit: An au-
tonomous LLM-agent for repository-level code au-
diting. InForty-second International Conference on
Machine Learning.
David Hin, Andrey Kan, Huaming Chen, and M Ali
Babar. 2022. Linevd: Statement-level vulnerability
detection using graph neural networks. InProceed-
ings of the 19th international conference on mining
software repositories, pages 596–607.
JeecgBoot. 2025. Https://github.com/jeecgboot/JeecgBoot.Zhengbao Jiang, Frank F Xu, Luyu Gao, Zhiqing Sun,
Qian Liu, Jane Dwivedi-Yu, Yiming Yang, Jamie
Callan, and Graham Neubig. 2023. Active retrieval
augmented generation. InProceedings of the 2023
Conference on Empirical Methods in Natural Lan-
guage Processing, pages 7969–7992.
JSPWiki. 2025. Https://github.com/apache/jspwiki.
Hong Jin Kang, Khai Loong Aw, and David Lo. 2022.
Detecting false alarms from automatic static analysis
tools: How far are we? InProceedings of the 44th
International Conference on Software Engineering,
pages 698–709.
KeyCloak. 2025. Https://github.com/keycloak/keycloak.
Avishree Khare, Saikat Dutta, Ziyang Li, Alaia Solko-
Breslin, Rajeev Alur, and Mayur Naik. 2025. Under-
standing the effectiveness of large language models
in detecting security vulnerabilities. In2025 IEEE
Conference on Software Testing, Verification and Val-
idation (ICST), pages 103–114. IEEE.
Caroline Lemieux, Jeevana Priya Inala, Shuvendu K
Lahiri, and Siddhartha Sen. 2023. Codamosa: Es-
caping coverage plateaus in test generation with pre-
trained large language models. In2023 IEEE/ACM
45th International Conference on Software Engineer-
ing (ICSE), pages 919–931. IEEE.
Patrick Lewis, Barlas Oguz, Ruty Rinott, Sebastian
Riedel, and 1 others. 2024. Retrieval-augmented
generation for knowledge-intensive nlp tasks.arXiv
preprint arXiv:2402.03456.
Haonan Li, Yu Hao, Yizhuo Zhai, and Zhiyun Qian.
2024. Enhancing static analysis for practical bug
detection: An llm-integrated approach.Proceed-
ings of the ACM on Programming Languages,
8(OOPSLA1):474–499.
Kaixuan Li, Sen Chen, Lingling Fan, Ruitao Feng, Han
Liu, Chengwei Liu, Yang Liu, and Yixiang Chen.
2023. Comparison and evaluation on static appli-
cation security testing (sast) tools for java. InPro-
ceedings of the 31st ACM Joint European Software
Engineering Conference and Symposium on the Foun-
dations of Software Engineering, pages 921–933.
Ziyang Li, Saikat Dutta, and Mayur Naik. 2025. Llm-
assisted static analysis for detecting security vulner-
abilities. InInternational Conference on Learning
Representations.
Yang Liu, Ziyu Wang, and Tianyi Zhang. 2025.
Swe-bench+: A contamination-free benchmark for
software engineering with llms.arXiv preprint
arXiv:2502.04188.
Noble Saji Mathews, Yelizaveta Brus, Yousra Aafer,
Meiyappan Nagappan, and Shane McIntosh. 2024.
Llbezpeky: Leveraging large language mod-
els for vulnerability detection.arXiv preprint
arXiv:2401.01269.
9

Meta Infer. 2025. Https://github.com/facebook/infer.
Public CMS. 2025.
Https://github.com/sanluan/PublicCMS.
React. 2025. Https://react.dev/blog/2025/12/03/critical-
security-vulnerability-in-react-server-components.
Rouyi. 2025. Https://github.com/yangzongzhuan/RuoYi.
Alireza Salemi and Hamed Zamani. 2024. Evaluating
retrieval quality in retrieval-augmented generation.
InProceedings of the 47th International ACM SI-
GIR Conference on Research and Development in
Information Retrieval, pages 2395–2400.
Snyk Code. 2025. Https://docs.snyk.io/scan-with-
snyk/snyk-code.
Saad Ullah, Praneeth Balasubramanian, Wenbo Guo,
Amanda Burnett, Hammond Pearce, Christopher
Kruegel, Giovanni Vigna, and Gianluca Stringhini.
2025. From cve entries to verifiable exploits: An
automated multi-agent framework for reproducing
cves.arXiv preprint arXiv:2509.01835.
Tianjun Wang, Yujia Liu, Yiming Zhang, and 1 others.
2025. Plan-and-execute: A modular agent architec-
ture for tool-use tasks. InICLR.
Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten
Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny Zhou,
and 1 others. 2022. Chain-of-thought prompting elic-
its reasoning in large language models.Advances
in neural information processing systems, 35:24824–
24837.
Chunqiu Steven Xia and Lingming Zhang. 2022. Less
training, more repairing please: revisiting automated
program repair via zero-shot learning. InProceed-
ings of the 30th ACM Joint European Software Engi-
neering Conference and Symposium on the Founda-
tions of Software Engineering, pages 959–971.
Shunyu Yao, Jifan Zhao, Dian Yu, Nan Du, Izhak
Shafran, Karthik Narasimhan, and Yuan Cao. 2023.
React: Synergizing reasoning and acting in language
models. InICLR.
Luo Yuan, Chen Zhaojun, Sun Yi, and Rhettxie. 2025.
More flows, more bugs: Empowering sast with llms
and customized dfa. InBlack Hat USA.
Yudao. 2025. Https://github.com/YunaiV/yudao-cloud.
Andy Zhou, Kai Yan, Michal Shlapentokh-Rothman,
Haohan Wang, and Yu-Xiong Wang. 2024a. Lan-
guage agent tree search unifies reasoning act-
ing and planning in language models.Preprint,
arXiv:2310.04406.
Xin Zhou, Sicong Cao, Xiaobing Sun, and David Lo.
2025. Large language model for vulnerability de-
tection and repair: Literature review and the road
ahead.ACM Transactions on Software Engineering
and Methodology, 34(5):1–31.Xin Zhou, Ting Zhang, and David Lo. 2024b. Large
language model for vulnerability detection: Emerg-
ing results and future directions. InProceedings of
the 2024 ACM/IEEE 44th International Conference
on Software Engineering: New Ideas and Emerging
Results, pages 47–51.
Kunlun Zhu, Zijia Liu, Bingxuan Li, Muxin Tian,
Yingxuan Yang, Jiaxun Zhang, Pengrui Han, Qipeng
Xie, Fuyang Cui, Weijia Zhang, and 1 others. 2025.
Where llm agents fail and how they can learn from
failures.arXiv preprint arXiv:2509.25370.
Yixin Zou, Abraham H Mhaidli, Austin McCall, and
Florian Schaub. 2018. " i’ve got nothing to lose":
Consumers’ risk perceptions and protective actions
after the equifax data breach. InFourteenth Sympo-
sium on Usable Privacy and security (soups 2018),
pages 197–216.
10

Figure 6: Token consumption and sink discovery perfor-
mance trade-off among different Argus backbones and
codebases.
A Implementation Details
We employ Claude-3.5-Sonnet and Claude-4.5-
Sonnet as the LLM backbone for our agents and
conduct all vulnerability detection experiments on
standard CPU servers equipped with 512 GB of
RAM. Argus requires no local GPU resources, as
the reasoning workload is entirely handled by re-
mote LLM inference servers. Under this config-
uration, we measure both token consumption and
execution time. For a typical code repository com-
prising tens of thousands of lines, Argus completes
the full detection process in approximately 0.44
hours on average, with an average cost of $2.54.
To evaluate detected vulnerabilities in our method,
we introduce three quantitative criteria:i)Rele-
vance score αr, initialized to 0.5; it increases by
0.4 if the vulnerability description contains specu-
lative keywords such as potential” or early”, and
gains an additional 0.1 if it mentions explicit se-
curity terms such as “vulnerability”; however, it
decreases by 0.1 if the issue is already marked
with a CVE.ii)Credibility score αc, defined as
αc= 0.3 + min(N c×0.05,0.3) , where Ncis the
number of comments on the issue.iii)Content
quality score αq, which evaluates the description
based on length, technical depth, specificity of im-
pact, presence of code examples, and proposed
solutions.
B Supplemental Materials
11

Figure 7: Screenshot of one of our realistic attack examples injected via our zero-day vulnerability discovered with
Argus. The repository is PublicCMS.
Figure 8: An example prompt of our vulnerability review agent in Argus.
12

Figure 9: The final vulnerability analysis report exported by Argus.
13