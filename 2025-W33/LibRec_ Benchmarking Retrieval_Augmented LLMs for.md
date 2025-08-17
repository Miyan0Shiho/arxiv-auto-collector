# LibRec: Benchmarking Retrieval-Augmented LLMs for Library Migration Recommendations

**Authors**: Junxiao Han, Yarong Wang, Xiaodong Gu, Cuiyun Gao, Yao Wan, Song Han, David Lo, Shuiguang Deng

**Published**: 2025-08-13 13:22:49

**PDF URL**: [http://arxiv.org/pdf/2508.09791v1](http://arxiv.org/pdf/2508.09791v1)

## Abstract
In this paper, we propose LibRec, a novel framework that integrates the
capabilities of LLMs with retrieval-augmented generation(RAG) techniques to
automate the recommendation of alternative libraries. The framework further
employs in-context learning to extract migration intents from commit messages
to enhance the accuracy of its recommendations. To evaluate the effectiveness
of LibRec, we introduce LibEval, a benchmark designed to assess the performance
in the library migration recommendation task. LibEval comprises 2,888 migration
records associated with 2,368 libraries extracted from 2,324 Python
repositories. Each migration record captures source-target library pairs, along
with their corresponding migration intents and intent types. Based on LibEval,
we evaluated the effectiveness of ten popular LLMs within our framework,
conducted an ablation study to examine the contributions of key components
within our framework, explored the impact of various prompt strategies on the
framework's performance, assessed its effectiveness across various intent
types, and performed detailed failure case analyses.

## Full Text


<!-- PDF content starts -->

1
LibRec: Benchmarking Retrieval-Augmented LLMs
for Library Migration Recommendations
Junxiao Han, Yarong Wang, Xiaodong Gu, Cuiyun Gao, Yao Wan, Song Han, David Lo, and Shuiguang Deng
Abstract —Library migration refers to replacing outdated or
unsuitable libraries with more appropriate alternatives. Although
substantial research has been conducted on the empirical analysis
of library migration, an automated framework for recommending
alternative libraries is still lacking. Such a framework could
substantially reduce developer effort and enhance the efficiency of
the library migration process. Meanwhile, large language models
(LLMs) have demonstrated exceptional performance across a
wide range of software engineering (SE) tasks. However, their
potential for addressing the library migration recommendation
task remains unexplored. In this paper, we propose LibRec, a
novel framework that integrates the capabilities of LLMs with
retrieval-augmented generation (RAG) techniques to automate
the recommendation of alternative libraries. The framework
further employs in-context learning to extract migration in-
tents from commit messages to enhance the accuracy of its
recommendations. To evaluate the effectiveness of LibRec, we
introduce LibEval, a benchmark designed to assess the perfor-
mance in the library migration recommendation task. LibEval
comprises 2,888 migration records associated with 2,368 libraries
extracted from 2,324 Python repositories. Each migration record
captures source-target library pairs, along with their corre-
sponding migration intents and intent types. Based on LibEval,
we evaluated the effectiveness of ten popular LLMs within
our framework, conducted an ablation study to examine the
contributions of key components within our framework, explored
the impact of various prompt strategies on the framework’s
performance, assessed its effectiveness across various intent types,
and performed detailed failure case analyses. Our experimental
results reveal several key findings: (1) LibRec demonstrates
strong effectiveness in recommending target libraries, with the
Claude-3.7-Sonnet model achieving the best performance across
all metrics; (2) both components in the framework are critical
for achieving optimal performance; (3) The One-shot prompt
strategy emerges as the most robust and reliable paradigm
across six LLMs, consistently delivering superior performance;
and (4) the framework demonstrates superior performance for
intent types related to issues with source libraries. These findings
provide valuable insights into the capabilities of various LLMs
Junxiao Han and Song Han are with the School of Computer and
Computing Science, Hangzhou City University, Hangzhou, China. E-mail:
hanjx@hzcu.edu.cn and hans@hzcu.edu.cn.
Yarong Wang is with the Polytechnic Institute, Zhejiang University,
Hangzhou, China. E-mail: wangyarong@zju.edu.cn
Xiaodong Gu is with the School of Computer Science, Shanghai Jiao Tong
University, Shanghai, China. E-mail: xiaodong.gu@sjtu.edu.cn
Cuiyun Gao is with the School of Computer Science and Tech-
nology, Harbin Institute of Technology, Shenzhen, China. E-mail:
gaocuiyun@hit.edu.cn
Yao Wan is with the College of Computer Science and Technology,
Huazhong University of Science and Technology, Wuhan, China. E-mail:
wanyao@hust.edu.cn
David Lo is with the School of Computing and Information Systems,
Singapore Management University, Singapore. E-mail: davidlo@smu.edu.sg
Shuiguang Deng is with the College of Computer Science and Technology,
Zhejiang University, Hangzhou 310027, China. E-mail: dengsg@zju.edu.cn
Junxiao Han and Yarong Wang contribute equally.
Shuiguang Deng is the corresponding author.in our framework for target library recommendation, highlight
promising directions for future research, and provide practical
suggestions for researchers, developers, and users.
Index Terms —Large Language Models, Benchmark, Library
Migration, Library Recommendation Framework
I. I NTRODUCTION
LIBRARY migration is an essential activity in modern
software development, as modern software heavily relies
on third-party libraries [1]. However, these libraries may
become obsolete over time [2, 3], introduce vulnerabilities or
bugs that negatively impact dependent applications [4–6], or be
replaced by newer, more efficient, or easier-to-use alternatives
[2, 7]. Consequently, developers often face the need to replace
an outdated or unsuitable library (i.e., source library) with
a more appropriate alternative (i.e., target library), a process
commonly referred to as library migration .
Despite its significance, library migration remains a com-
plex, time-consuming, and error-prone task. Developers often
dread this process, particularly in large codebases where the
source library is deeply integrated [4]. Identifying appropriate
alternatives automatically holds the potential to significantly
reduce developer effort and improve migration efficiency.
However, most existing research on library migration has been
empirical, focusing on the frequency, domains, mechanisms,
and rationales behind library migration [6, 8–10]. A limited
body of work has investigated library migration recommenda-
tions, primarily focusing on applying several metrics to screen
existing migration rules for generating recommendations for
source libraries [2, 11, 12]. Unfortunately, these approaches
lack the capability to automatically recommend suitable target
libraries, leaving developers to manually identify alternatives.
Moreover, they are unable to recommend libraries that are not
covered by existing migration rules.
In recent years, Large Language Models (LLMs) are in-
creasingly being employed to support various software engi-
neering (SE) tasks, such as code generation [13, 14], code
completion [15, 16], test cases generation [17, 18], vulner-
ability detection [19, 20], and API recommendation [21].
This suggests that LLMs may also possess the capability to
perform library migration recommendations. However, to the
best of our knowledge, their application to library migration
recommendation has not yet been explored. Existing LLM-
based techniques for library migration tasks have primarily
focused on API recommendations rather than library migra-
tion. For instance, ToolLLM facilitates the recommendation
of chains of API calls for given instructions [22], GorillaarXiv:2508.09791v1  [cs.SE]  13 Aug 2025

2
15- from Crypto.Hash import SHA256 15+ from cryptography.hazmat.backends import default_backend
16- from Crypto.PublicKey import RSA 16+ from cryptography.hazmat.primitives.asymmetric import padding
17- from Crypto.Signature import PKCS1_v1_5 17+ from cryptography.hazmat.primitives.asymmetric import rsa
18- 18+ from cryptography.hazmat.primitives import hashes
19- 19+ from cryptography.hazmat.primitives import serialization
45def test_pubkeys(self, mock_get_user): 45def test_pubkeys(self, mock_get_user):
46         """Test '/v1/profile/pubkeys' API endpoint.""" 46         """Test '/v1/profile/pubkeys' API endpoint."""
47         url = self.URL + 'pubkeys' 47         url = self.URL + 'pubkeys'
48-        data_hash = SHA256.new() 48+        key = rsa.generate_private_key(
49-        data_hash.update('signature'.encode('utf-8')) 49+            public_exponent=65537,
50-        key = RSA.generate(1024) 50+            key_size=1024,
51-        signer = PKCS1_v1_5.new(key) 51+            backend=default_backend()
52-        sign = signer.sign(data_hash) 52+        )
53-        pubkey = key.publickey().exportKey('OpenSSH') 53+        signer = key.signer(padding.PKCS1v15(), hashes.SHA256())
54- 54+        signer.update('signature'.encode('utf-8'))
55- 55+        sign = signer.finalize()
56- 56+        pubkey = key.public_key().public_bytes(
57- 57+            serialization.Encoding.OpenSSH,
58- 58+            serialization.PublicFormat.OpenSSH）
Fig. 1. Library migration from pycryptodome/Crypto tocryptography extracted from svtplay-dl , where the pink-highlighted parts represent the
source library and its corresponding code to be migrated, while the green-highlighted parts denote the target library and its replacement code.
employs a fine-tuned LLaMA model for API recommendation
[23], and APIGen leverages generative models with enhanced
in-context learning for API suggestions [21]. Although LLMs
demonstrate the potential in API recommendation tasks, their
application to library migration remains unexplored.
To bridge this gap, we propose LibRec, a framework
that combines the capabilities of LLMs with retrieval-
augmented generation (RAG) techniques [24, 25] to au-
tomate library migration recommendations. The framework
leverages in-context learning of migration intents extracted
from commit messages to enhance library recommenda-
tions. For instance, consider the migration of the Python
library pycryptodome/Crypto to its analogous library
cryptography , as illustrated in Figure 1. In this case, the
developer stated in their commit message, “This will make
it easier to support older distros. For example, Ubuntu 16.04
and Debian stable (9).” In such scenarios, our framework is
designed to recommend the target library, cryptography ,
based on developer-provided instructions, the names of source
libraries (e.g., pycryptodome/Crypto ), and migration in-
tents expressed in commit messages.
To assess the effectiveness of LibRec, we construct LibEval,
a benchmark specifically designed to evaluate its performance
in target library recommendation tasks. LibEval comprises
2,888 migration records associated with 2,368 libraries ex-
tracted from 2,324 Python repositories, each record includes
the source-target library pairs, along with their generatedmigration intents and intent types. Building on this foundation,
we evaluated the effectiveness of ten popular LLMs within our
framework, including general LLMs of GPT-4o-mini, GPT-4,
DeepSeek-V3, Qwen-Plus, Llama-3.3, and Claude-3.7-Sonnet;
code LLMs such as Qwen-Coder-Plus and Qwen2.5-Coder
from the Qwen family; and reasoning LLMs of DeepSeek-R1
and QwQ-Plus (RQ1). Additionally, we conducted an ablation
study to examine the contributions of key components within
the framework, specifically the retrieval-augmented component
and the inclusion of migration intents in prompts (RQ2).
To deepen our understanding, we explored various prompt
strategies, including One-shot and Chain-of-Thought (CoT),
to assess their impact on the framework’s performance (RQ3).
Additionally, we assessed the framework’s effectiveness across
various intent types, such as issues with source libraries, ad-
vantages of target libraries, and project-related reasons (RQ4),
and performed case studies to analyze failure cases (RQ5).
Based on our experimental results, we summarize the
following main findings: 1) LibRec demonstrates strong
effectiveness in recommending target libraries, with the
Claude-3.7-Sonnet model achieving the best performance
across all metrics. 2) Both the retrieval-augmented component
and migration intents within the framework are essential for
achieving optimal effectiveness. On average, retrieval aug-
mentation exhibits greater significance compared to migration
intents. 3) The one-shot prompt strategy emerges as the most
robust and reliable paradigm across six models, consistently

3
delivering superior performance. In contrast, the COT prompt
strategy excels primarily in reasoning LLMs and high-capacity
models like Claude-3.7-Sonnet. 4) Our framework demon-
strates superior performance for intent types related to issues
with source libraries. 5) Most recommendation failures occur
within the intent type of advantages of target libraries, with the
sub-category Enhanced Features (i.e., more specific features)
accounting for the largest proportion of failures in this group.
In summary, this paper makes the following contributions:
•We propose LibRec, a framework that combines the
capabilities of LLMs with RAG techniques, and leverages
in-context learning of migration intents to recommend
target libraries automatically.
•We introduce LibEval, a benchmark designed for the
target library recommendation task, comprising 2,888
records of source-target library pairs, along with their cor-
responding generated migration intents and intent types.
•We evaluate the performance of ten popular LLMs in
our framework, providing insights into their effectiveness
in the target library recommendation task, and reveal
the contributions of each key component in our frame-
work. Moreover, we examine the performance of our
framework across various intent types, offering a detailed
understanding of its effectiveness in diverse migration
scenarios.
The remainder of this paper is organized as follows. Section
II reviews the related work. Section III proposes the framework
for library migration recommendations, while Section IV
introduces the benchmark for evaluation. Section V presents
the experimental setup, and Section VI reports the findings
corresponding to each of the five research questions. In Section
VII, we discuss the implications of our findings to researchers,
developers, and users. Section VIII addresses potential threats
to the validity of our study, and Section IX concludes this
paper and outlines directions for future work.
II. R ELATED WORK
A. Traditional Library Migration
Several studies have conducted empirical analyses of library
migrations or proposed traditional approaches to recommend
libraries to developers. Teyton et al. [8] analyzed Java open-
source software to investigate the frequency, contexts, and rea-
sons for library migration. He et al. [9] examined 19,652 Java
GitHub projects to understand how and why library migra-
tion happens, and derived 14 frequently mentioned migration
reasons (e.g., lack of maintenance, usability, and integration)
via thematic analysis of related commit messages, issues, and
pull requests. Gu et al. [10] compared the frequency, domain,
rationales, and unidirectionality of self-admitted library migra-
tions (SALMs) across Java, JavaScript, and Python ecosys-
tems, revealing domain similarity across these ecosystems,
and revealing differences in rationale distributions, ecosystem-
specific domains, and levels of unidirectionality. Islam et
al. [6] conducted an empirical analysis of Python library
migrations, identifying a taxonomy of PyMigTax to classify
migration-related code changes, which consisted of 3,096
migration-related code changes across 335 Python librarymigrations from 311 client repositories spanning 141 library
pairs in 35 domains. Moreover, He et al. [2] proposed an
approach for recommending library migrations in Java projects
by leveraging multiple metrics, including Rule Support, Mes-
sage Support, Distance Support, and API Support. Based on
the metrics in this work, Zhang et al. [11] incorporated the
metric of label correlations of libraries in the Maven Central
Repository to enhance migration recommendations. Mujahid et
al. [12] developed an approach to identify declining packages
and suggest alternative libraries based on observed migration
patterns, resulting in 152 dependency migration suggestions.
Most existing research is predominantly conducted on Java
datasets, while the exploration of Python datasets is limited.
Moreover, current studies primarily focus on analyzing the
frequency and distribution of reasons for library migrations,
without providing concrete recommendations for target li-
braries to replace source libraries. The few studies addressing
library migration recommendations typically rely on multiple
metrics to filter existing migration rules, but fall short of
incorporating the latest knowledge. To address these gaps, our
study establishes a benchmark for the library migration task
and proposes a target library recommendation framework that
leverages the capabilities of LLMs to deliver effective target
library recommendations.
B. LLM-based library migration
The majority of LLM-based techniques for library migration
primarily focus on recommending APIs or API sequences. Qin
et al. [22] proposed ToolLLM, a framework encompassing
data construction, model training, and evaluation, designed to
leverage ChatGPT for identifying valid solution paths (i.e.,
chains of API calls) for given instructions. Nan et al. [26]
introduced DDASR, a framework for recommending API
sequences that integrate both popular and tail APIs. Their
framework leveraged LLMs for learning query representations,
and clustered tail APIs with similar functionality, replacing
them with cluster centers to enhance the understanding of tail
APIs. Experimental results demonstrate that DDASR achieves
superior diversity in recommendations while maintaining high
accuracy.
Moreover, Wang et al. [27] proposed PTAPI, a novel ap-
proach designed to enhance API recommendation performance
by visualizing users’ intentions based on their queries. The
method began by identifying a prompt template from Stack
Overflow posts based on the user’s input, which was then
combined with the input to generate a refined query. This
refined query leveraged dual information sources from Stack
Overflow posts and official API documentation to provide
more precise recommendations. Yang et al. [28] proposed
DSKIPP, a novel prompt method designed to enhance LLMs
for Java API recommendations. This approach systematically
guides LLMs through a hierarchical process encompassing
package, class, and method levels, and incorporates devel-
opment scenarios, key knowledge, and a self-check mecha-
nism to ensure reliable results. Patil et al. [23] introduced
Gorilla, an API recommendation tool built on a fine-tuned
LLaMA model that works with a document retriever, achieving

4
greater flexibility and generalization compared to traditional
embedding-based techniques. Chen et al. [21] developed API-
Gen, a generative API recommendation approach that employs
enhanced in-context learning to suggest APIs effectively with
a few examples, utilizing diverse example selection and guided
reasoning to improve recommendation accuracy and interoper-
ability. Additionally, Latendresse et al. [29] utilized ChatGPT
to generate Python code for a substantial set of real-world
coding problems derived from Stack Overflow, analyzing the
characteristics of libraries employed in the generated code.
III. L IBREC: RETRIEVAL -AUGMENTED LIBRARY
RECOMMENDATION
Figure 2 (a) presents the workflow of our proposed frame-
work, LibRec, which combines the capabilities of LLMs with
the RAG strategy [24, 25] to automate library migration
recommendations. The framework begins by generating migra-
tion intents for each library migration pair, thereby providing
additional contextual information to facilitate library migration
recommendation tasks (Figure 2 (a)-I). Next, we construct
an RAG database to support the formulation of the target
library recommendation task (Figure 2 (a)-II). Finally, LLMs,
enhanced with RAG techniques, and guided by migration
intents embedded in prompts, enable the recommendation
of appropriate target libraries (Figure 2 (a)-III). A detailed
explanation of this framework is provided below.
A. Migration Intents Generation
Previous studies have adopted commit messages as prox-
ies for programmer intents to augment the capabilities of
LLMs in automatic bug fixing [30]. Inspired by this work,
we incorporate migration intents to enhance the performance
of LLMs in library recommendation. However, a substantial
amount of commit messages from open-source projects lack
critical information or are even empty [31]. Additionally, many
commit messages cover a wide range of subjects [32]. In
this regard, just adopting commit messages as proxies for
migration intents can introduce noise. To address this issue
and ensure the provision of high-quality migration intents, we
leverage the capabilities of LLMs to extract refined migration
intents from commit messages.
Specifically, we utilize an LLM (GPT-4o) to generate
migration intents based on commit messages. We design a
prompt template (Figure 3) employing a few-shot strategy
to instruct the model to generate migration intents under
specific constraints: 1) extract migration purposes explicitly
or implicitly stated in the message, 2) avoid including any
details not present in the commit message to mitigate the risk
of hallucinated outputs, and 3) if no clear migration intent can
be determined from the commit message, output “NULL”.
B. Intent Types Determination
Subsequently, we classified the migration intents into dis-
tinct categories following the classification framework pro-
posed by He et al. [9]. This framework defines four primary
categories of Source Library Issues ,Target Library Advan-
tages ,Project Specific Reasons , and Other , along with 14associated subcategories. Intent types under Source Library
Issues typically arise due to issues with the source library,
such as the lack of maintenance or being outdated (i.e., sub-
category of Not Maintained/Outdated ), security vulnerabilities
(i.e., Security Vulnerability ), or bugs and defect issues (i.e.,
Bug/Defect Issues ). Intents classified under Target Advantages
reflect the benefits provided by the target library, including
better usability (i.e., Usability ), more specific features (i.e.,
Enhanced Features ), superior functional performance (i.e.,
Performance ), smaller size/complexity (i.e., Size/Complexity ),
higher popularity (i.e., Popularity ), stronger stability and ma-
turity (i.e., Stability/Maturity ), and more community activities
(i.e., Activity ). For intents categorized as Project Specific , the
reasons are tied to the unique needs of the project, such as
easier integration (i.e., Integration ), project simplification (i.e.,
Simplification ), ensuring license compatibility (i.e., License ),
and organization influences (i.e., Organization Influence ).
We employ an LLM (GPT-4o) with few-shot prompting
to automatically classify each migration intent into its cor-
responding intent types, and Figure 4 presents the distribution
of the three primary categories (excluding the “Other” type)
of migration intent types. Within the Source Library Issues
category, the subcategory Not Maintained/Outdated accounts
for more than half of the intent types. Similarly, Integration
constitutes the majority of intent types within the Project
Specific category, while Enhanced Features represents nearly
half of the intent types under Target Advantages .
C. Construction of RAG Database
As depicted in Figure 2, our goal is to recommend suitable
target libraries for a given source library. To achieve this,
we first construct an RAG database, where each entity is
represented as H={(Sj, Tj, Ij, Cj)}|H|
j=1, forming a 4-
tuple library migration dataset. Here, Sjdenotes the j-th
source library, Tjrepresents its corresponding target library,
Ijindicates the set of migration intents, and Cspecifies the
intent types. The recommendation process begins by retrieving
migration entities similar to the given source library from the
constructed RAG database. For this purpose, we employ the
sparse keyword-based BM25 score [33] as our retrieval metric.
BM25, a probabilistic model widely adopted in prior research
[30, 34, 35], operates as a bag-of-words retriever that estimates
lexical-level similarity between two sentences. Higher BM25
scores indicate greater similarity between the sentences.
Specifically, each query is formulated as a triple q=
(s, i, c ), where sdenotes the given source library, irepresents
the associated migration intents, and cindicates the intent
types. Based on this multi-faceted library migration context,
LibRec retrieves the top- kmost relevant migration entities
from the RAG database for a given query q.
D. Target Library Recommendation
LibRec leverages the capabilities of LLMs and employs a
structured prompt (as shown in Figure 5) to require LLMs to
recommend appropriate target libraries. Specifically, LibRec
incorporates similar migration demonstrations retrieved from
the constructed RAG database (as detailed in Section III-C).

5
Intent Generation-(I)
Migration Intents
Intents TypeExtract Task 
Generate Task
Few-Shot
Standard 
FrameworkPrompt Expert Persona
Task Description
Retrieved Historical 
Migration Rules
Library Names
Migration Intents
Intent Types
retrieveLib-1
Lib-2
Lib-3
RAG Database-(II)
Lib-A Lib-B
Lib-C Lib-D
Lib-E Lib-FSource Library
Target Library
Migration Intents
Intents Type
Library Recommendation-(III)
Pair Validation-(III)
Analogous
Irrelevant
Rule Extraction-(II)
Commits
DiffsLib-A Lib-B
Lib-C Lib-D
Lib-E Lib-F
Repo Collection-(I)
Popularity
LanguageRecency
CollaborationRules(a) LibRec: Our framework for Library Migration Recommendations
(b) LibEval: The Benchmark
Fig. 2. Overview of study workflow.
You are tasked with extracting the migration intents from 
commit messages that describe library migrations. Each 
commit message may contain one or more migration intents. 
Constraints
BASED ONLY on the commit message content:
- Extract migration purposes that are explicitly or 
implicitly stated in the message.
- Do not include any details that are not present in the 
commit message (avoid hallucinations).
- If you cannot determine any clear migration intent 
from a commit message, output “NULL”.
Few-shot
•example1: commit message + output
•example2: commit message + output
Output format
output:{ 
"migration_intents": [intent1,intent2,...]
}
Fig. 3. The prompt template used for generating migration intents.
These demonstrations provide LLMs with relevant examples
of similar library migrations. Additionally, migration intents
and intent types are incorporated to augment the capabilities
of LLMs in precise target library recommendations. In par-
ticular, the structured prompt includes the name of the source
library, its similar migration demonstrations, its corresponding
migration intents, and intent types, providing LLMs a compre-
hensive understanding of the library migration context. LibRec
Source Library Issues Target Library Advantages Project Specific Reasons0100200300400500Numbers
Not Maintained/Outdated (68%)Bug/Defect Issues (18%)Security Vulnerability (15%)
Enhanced Features (46%)Usability (20%)Activity (18%)Popularity (8%)Performance (4%)Size/Complexity (3%)
Integration (67%)Simplification (32%)License (1%)Fig. 4. The distribution of migration intents.
instructs LLMs to output a ranked list of the top ten candidate
libraries, ordered by their estimated suitability for the given
source library. This scenario enhances development efficiency
and reduces the time developers spend searching and selecting
appropriate target libraries.
IV. L IBEVAL: THEBENCHMARK
To evaluate the effectiveness of LibRec, we construct
LibEval, a benchmark specifically designed to assess its per-
formance in target library recommendation tasks. Specifically,
we outline the construction process of LibEval (as illustrated
in Figure 2 (b)), covering data collection (Section IV-A to
IV-D) and data generation (Section IV-E and IV-F).

6
Please recommend EXACTLY 10 potential candidate 
libraries sorted by descending probability of suitability for the 
given source library.
{
"candidate libraries": [LIB1,LIB2,...,LIB9,LIB10]
}
Source Library: [Library Name]
Target Libraries: ____________Role
You are an expert on the task of library migration 
recommendation.
Task
Suggest target libraries for a given source library based on 
functional analogous, migration intent, intent type, and 
retrieved similar migration demonstrations.
Retrieved Similar Migration Demonstrations
Below are the retrieved similar migration rules sorted from 
highest to lowest.
Rule1: Source Lib      Target Lib | Intents | Intent categories
Rule2: Source Lib      Target Lib | Intents | Intent categories
Rule3: Source Lib      Target Lib | Intents | Intent categories
Output Requirements
Input and Response
Fig. 5. The prompt template used for target library recommendation.
A. Repository Collection
Since Python became the most popular programming lan-
guage on GitHub after 2024, driven by its widespread adoption
in AI-related domains [36], our study focuses specifically on
Python repositories. According to the findings of Coelho et al.
[37], 50% of GitHub projects become unmaintained after 50
months (approximately 4 years). To ensure the timeliness of
our dataset, we filtered and retained GitHub repositories with
at least one commit made after January 1, 2020. To safeguard
the quality of our dataset, we performed a filtering process.
Drawing inspiration from prior studies [9, 38], we excluded
forked repositories, as well as repositories with fewer than 10
stars or fewer than 2 contributors. Ultimately, we collected
62,986 Python repositories.
B. Candidate Migration Commit Identification
We analyze all commits in the collected repositories to
identify candidate migration commits. To achieve this, we
adopt heuristics established in prior studies [39], enabling us to
exclude non-migration commits effectively. The heuristics are:
1)Exclude merge commits , we exclude merge commits to
avoid redundant analysis of changes already present in parent
branches. 2) Dependencies have changed , since migration
involves the replacement of one library with another, we focus
on commits that include both additions and deletions of lines
in dependency files, as this serves as clear evidence of a library
migration. Here, for instance, we consider requirement.txt as
the dependency file. 3) Exclude bot-created commits , weexclude them as they typically relate to version updates and
do not involve code modifications.
For the first two heuristics, we utilize PyDriller1to perform
necessary checks. For the third heuristic, we identify commit
authors whose names contain the term “bot” to exclude bot-
created commits. Following this process, we identified 433,602
candidate migration commits, along with their commit mes-
sages, from 44,288 repositories.
C. Candidate Migrations Extraction
+cryptography>=1.0,!=1.3.0
gunicorn==18
oslo.config>=1.6.0 # Apache-2.0
oslo.db>=1.4.1 # Apache-2.0
oslo.log
pecan>=0.8.2
pyOpenSSL>=0.14
-pycrypto>=2.6
requests>=2.2.0,!=2.4.0
requests-cache>=0.4.9
jsonschema>=2.0.0,<3.0.0
Listing 1. Sample library replacements in a requirements.txt file
We first automatically analyze the patch diffs of each of
the 433,602 candidate migration commits to identify potential
library replacements, i.e., candidate migrations. Listing 1
illustrates an example of library replacements in a dependency
file, where 1 library cryptography is added, and 1 library py-
crypto is removed. To determine the sets of removed libraries
and added libraries, we adopted the approach mentioned in
previous studies [12]. Specifically, we only retained library
replacements where the difference between the number of
added libraries ( La) and removed libraries ( Lr) satisfies the
condition that the absolute difference |La−Lr|is minimal (i.e.,
|La−Lr| ≤1). Additionally, we exclude library replacements
involving a large number of added and removed dependencies,
defined as La+Lr> m , where mis the median value
ofLa+Lracross all commits. This filtering was applied
because a large number of library replacements typically
indicates significant code refactoring rather than a simple
dependency replacement. It is important to note that one
commit may involve multiple pairs of library replacements,
potentially resulting in multiple migrations within the same
commit. However, since the median value of mforLa+Lr
was calculated to be 2, in this regard, we retained only
those commits that contained a single pair of added-removed
(i.e., source-target) libraries. Furthermore, we ignore library
versions in this paper to focus exclusively on migrations
between different libraries. As a result, we identified a total
of 14,608 candidate migration rules.
D. Non-Analogous Library Pairs Screening
In the real world, developers often replace a library with
another that provides the same or similar functionalities
[10, 40]. Accordingly, for a migration to be considered valid,
it must involve an analogous pair of libraries, i.e., the added
1https://pydriller.readthedocs.io

7
and removed libraries should offer comparable functionalities.
However, manually reviewing and classifying library pairs
is time-consuming and labor-intensive. Motivated by the ad-
vanced capabilities of LLMs in natural language processing
[41] tasks, we leverage an LLM (GPT-4o) to automatically
verify whether a given library pair is analogous.
To achieve this, we design an initial prompt template
(Figure 6) using a few-shot strategy to guide the model
effectively, and manually evaluate the effectiveness of the
prompt template through an iterative process. Consequently,
we identified 7,205 library pairs annotated as analogous. To
assess the robustness of the LLM annotations, we randomly
sample 365 examples based on a 95% confidence level with
a 5% confidence interval [42, 43] for manual evaluation. Our
manual review confirmed that the LLM annotations achieved
an accuracy exceeding 98%, demonstrating the effectiveness
and robustness of the data annotations classified by the LLM
(GPT-4o). Subsequently, we removed 4 library pairs where
inconsistencies were identified between LLM annotations and
human evaluations, resulting in 7,201 library migration rules.
You are an expert in software engineering and library 
migrations. Analyze whether two Python libraries are 
analogous (provide similar core functionality). Return JSON 
with classification result.
{positive example 1}
{positive example 2}
{positive example 3}Analogous Examples:
{negative example 1}
{negative example 2}
{negative example 3}
No-Analogous Examples:Input:
{"Removed lib": "requests",
  "Added lib":  "httpx"}
Output:
{"is_analogous": true}
Input:
{"Removed lib": "flask",
  "Added lib":  "requests"}
Output:
{"is_analogous": false}
Problem:
Input:"Removed lib": "***","Added lib":  "***"
Output:"is_analogous": _
Fig. 6. The prompt template used for classifying whether a library pair is
analogous.
E. Data Generation
After collecting analogous migration rules, we applied the
methodology introduced in Section III-A to extract migration
intents from corresponding commit messages for each library
migration rule. Consequently, 2,888 high-quality migration in-
tents were successfully generated from the commit messages.
These 2,888 migration rules, along with their corresponding
migration intents, were retained for further analysis. To ensure
the accuracy and reliability of this approach, we randomly
sampled 100 examples for manual verification. We reviewed
each commit message to check the correctness of the generatedintent. Our analysis confirmed that the LLM consistently
produces high-quality migration intents, demonstrating its re-
liability and effectiveness.
Subsequently, we applied the method described in Section
III-B to classify migration intents into distinct intent types.
To evaluate the accuracy of this classification, a human eval-
uation was performed on a randomly selected sample of 339
migration intents, along with their corresponding intent types.
This sample was drawn from the total of 2,888 records based
on a 95% confidence level with a 5% confidence interval
[42, 43]. Human evaluation indicates that GPT-4o performs
well in intent type classification, achieving an accuracy of
95%.
F . Dataset Mask and Split
Following the processes of dataset collection and genera-
tion, we collected a total of 2,888 migration rules along with
their corresponding migration intents and intent types. By
manual observation of the dataset, we identified a potential
risk of data leakage in the migration intents extracted from
commit messages. Specifically, some migration intents explic-
itly referenced the names of target libraries or their variants,
which could inadvertently disclose critical factual information
when prompting LLMs. To mitigate this issue, we manually
masked references to target libraries within the migration
intents, replacing them with the [MASK] tag. This ensures
that the recommendations generated by LLMs are based solely
on the detailed information provided, rather than relying on
factual knowledge of the target libraries.
After that, the dataset was divided into training and test sets
using an 8:2 ratio. To prevent any data leakage, we followed
the practice in prior studies [30], and ensured that migration
rules originating from the same repository were assigned to
the same set. Additionally, all migration rules were ordered
by their migration dates, with more recent rules allocated to
the training set. This process yields 2,310 migration rules in
the training set and 578 in the test set. Specifically, the training
set serves as the primary data source for our RAG database
(detailed in Section III-C).
V. E XPERIMENTAL SETUP
A. Research Questions
Our experiments aim to answer the following research
questions (RQ):
•RQ1: Overall Performance. How does LibRec perform
in library recommendation?
•RQ2: Ablation Study. How does each component con-
tribute to the overall performance?
•RQ3: Effectiveness of Different Prompt Engineering
Strategies. How do different prompt engineering strate-
gies influence the effectiveness of our approach?
•RQ4: Effectiveness across Different Intent Types. How
does LibRec perform on different types of migration
intents?
•RQ5: Cases Study. What are the characteristics of failed
cases?

8
TABLE I
SELECTED LLM S.
Model Size Time Instruct Base
GPT-4o-mini N/A 2024-07-18 ✗ ✓
GPT-4 N/A 2023-06-13 ✗ ✓
DeepSeek-V3 660B 2025-03-24 ✗ ✓
Qwen-Plus N/A 2025-01-25 ✗ ✓
Llama-3.3 70B 2024-12-06 ✓ ✗
Claude-3.7-Sonnet N/A 2025-02-19 ✗ ✓
Qwen-Coder-Plus N/A 2024-11-06 ✗ ✓
Qwen2.5-Coder 32B 2024-11-12 ✓ ✗
DeepSeek-R1 671B 2025-01-20 ✗ ✓
QwQ-Plus N/A 2025-03-05 ✗ ✓
B. Model Selection
As shown in Table I, we evaluate the performance of ten dis-
tinct LLMs on the target library recommendation task. These
models are categorized into three groups: general LLMs, code
LLMs, and reasoning LLMs. Notably, we exclude GPT-4o to
avoid potential data leakage, since it was used for migration
intents generation. For general LLMs, we select GPT-4o-mini,
GPT-4, DeepSeek-V3, Qwen-Plus, Llama-3.3, and Claude-3.7-
Sonnet. Specifically, GPT-4-0613 is used as the representative
version of GPT-4, while Llama-3.3-70B-Instruct serves as
the implementation version for Llama-3.3. For code LLMs,
we include Qwen-Coder-Plus and Qwen2.5-Coder from the
Qwen family, with Qwen2.5-Coder-32B-Instruct employed as
the implementation version for Qwen2.5-Coder. For reasoning
LLMs, we evaluate DeepSeek-R1 and QwQ-Plus. The selec-
tion of these models is based on their outstanding performance
in prior research and their availability.
C. Evaluation Metrics
Following previous studies [21, 44], we employ two metrics
to evaluate the performance of our framework: Precision @k
and Mean Reciprocal Rank (MRR).
•Precision @kmeasures a model’s ability to include the
correct target library within the top-k results, regardless
of their order. It is defined as:
Precision @k=PN
i=1HasTarget k(qi)
N(1)
where Ndenotes the total number of queries, and
HasCorrect k(q)equals 1 if the correct target library
appears within the top-k results for query q, otherwise it
returns 0.
•MRR is a widely adopted metric for evaluating rec-
ommendation tasks. It considers the ranks of all correct
answers. It is formally defined as:
MRR =PN
i=11/firstpos (qi)
N(2)
where firstpos (q)returns the position of the first correct
target library in the results. If the correct target library is
not present in the results, the numerator of the formula
is zero.D. Implementation Details
Following Chen et al.’s study [45], we set k= 3 during
the retrieval stage, to optimize the LLM’s ability to learn
from the retrieved demonstrations. Additionally, we configure
the LLMs with a temperature of 0.7 and a top p of 0.95 to
balance diversity and coherence in the generated outputs. For
the evaluation metric Precision @k, we set k∈ {1,3,5,10}to
assess the model’s ability to include the correct answer within
its top-k outputs.
VI. E VALUATION RESULTS
A. RQ1: Overall Performance
TABLE II
THE PERFORMANCE OF DIFFERENT LLM S IN OUR FRAMEWORK .
LLMPrecision@kMRRTop-1 Top-3 Top-5 Top-10
GPT-4o-mini 0.448 0.611 0.647 0.673 0.532
GPT-4 0.516 0.675 0.701 0.729 0.598
DeepSeek-V3 0.524 0.667 0.687 0.711 0.598
Qwen-Plus 0.510 0.638 0.661 0.694 0.578
Llama-3.3 0.467 0.585 0.630 0.651 0.533
Claude-3.7-Sonnet 0.566 0.722 0.751 0.792 0.652
Qwen-Coder-Plus 0.484 0.649 0.678 0.701 0.569
Qwen2.5-Coder 0.465 0.633 0.671 0.704 0.554
DeepSeek-R1 0.550 0.718 0.730 0.756 0.634
QwQ-Plus 0.521 0.671 0.692 0.735 0.598
Table II presents a comparative analysis of Precision@1,
Precision@3, Precision@5, Precision@10, and MRR across
general LLMs, code LLMs, and reasoning LLMs in the
target library recommendation task. The experimental results
demonstrate that our framework can recommend target
libraries effectively. Surprisingly, among all the evaluated
models, the general LLM Claude-3.7-Sonnet achieves the best
performance, with the highest scores across all metrics: Preci-
sion@1 (56.6%), Precision@3 (72.2%), Precision@5 (75.1%),
Precision@10 (79.2%), and MRR (65.2%).
Although the framework exhibits strong overall perfor-
mance, there are substantial performance disparities across
different categories of LLMs. Specifically, in the category
of general LLMs, Claude-3.7-Sonnet outperforms the com-
mercial models GPT-4o-mini and GPT-4 in Precision@10
by approximately 17.7% and 8.7%, respectively. Meanwhile,
GPT-4 and DeepSeek-V3 exhibit comparable performance
in the target library recommendation task. In contrast, the
open-source general LLM Llama-3.3 demonstrates the weakest
performance in this task, which may be due to its smaller
parameter size.
In the category of reasoning LLMs, DeepSeek-R1 deliv-
ers suboptimal performance across all metrics: Precision@1
(55%), Precision@3 (71.8%), Precision@5 (73%), Preci-
sion@10 (75.6%), and MRR (63.4%). Similarly, QwQ-Plus
demonstrates relatively high performance, surpassing most
other LLMs. The notable performance of DeepSeek-R1 and
QwQ-Plus highlights the effectiveness of reasoning LLMs in
addressing the target library recommendation task.
When it comes to the category of code LLMs, the per-
formance is observed to be inferior compared to both general

9
TABLE III
PERFORMANCE OF VARIANTS OF VANILLA ,W/ORETAND W/OINTENT COMPARED TO OUR FRAMEWORK .
Setup GPT-4o-
miniGPT-4 DeepSeek-
V3Qwen-
PlusLlama-
3.3Claude-
3.7-
SonnetQwen-
Coder-
PlusQwen2.5-
CoderDeepSeek-
R1QwQ-
PlusAverage
Precision@1
Vanilla 0.363 0.405 0.445 0.356 0.315 0.372 0.395 0.414 0.426 0.392 0.388 ( ↓23.17%)
w/o Ret 0.403 0.484 0.476 0.464 0.445 0.514 0.429 0.436 0.500 0.460 0.461 ( ↓8.71%)
w/o Intent 0.396 0.469 0.388 0.457 0.360 0.462 0.460 0.445 0.424 0.484 0.435 ( ↓13.86%)
LibRec 0.448 0.516 0.524 0.510 0.467 0.566 0.484 0.465 0.550 0.521 0.505
Precision@3
Vanilla 0.507 0.559 0.592 0.517 0.420 0.540 0.548 0.540 0.562 0.568 0.535 ( ↓18.57%)
w/o Ret 0.535 0.613 0.609 0.595 0.554 0.680 0.571 0.564 0.640 0.616 0.598 ( ↓8.98%)
w/o Intent 0.517 0.640 0.510 0.606 0.479 0.663 0.604 0.590 0.543 0.680 0.583 ( ↓11.26%)
LibRec 0.611 0.675 0.666 0.638 0.585 0.722 0.649 0.633 0.718 0.671 0.657
Precision@5
Vanilla 0.548 0.587 0.640 0.552 0.448 0.599 0.590 0.588 0.602 0.592 0.575 ( ↓16.06%)
w/o Ret 0.564 0.640 0.640 0.623 0.592 0.716 0.614 0.602 0.675 0.640 0.631 ( ↓7.88%)
w/o Intent 0.545 0.670 0.542 0.642 0.519 0.702 0.635 0.625 0.588 0.709 0.618 ( ↓9.78%)
LibRec 0.647 0.701 0.687 0.661 0.630 0.751 0.678 0.671 0.730 0.692 0.685
Precision@10
Vanilla 0.583 0.632 0.678 0.595 0.500 0.644 0.619 0.625 0.645 0.628 0.615 ( ↓13.99%)
w/o Ret 0.604 0.683 0.673 0.661 0.623 0.758 0.654 0.650 0.702 0.689 0.670 ( ↓6.29%)
w/o Intent 0.573 0.711 0.568 0.670 0.562 0.744 0.670 0.668 0.625 0.753 0.654 ( ↓8.53%)
LibRec 0.673 0.729 0.711 0.694 0.651 0.792 0.701 0.704 0.756 0.735 0.715
MRR
Vanilla 0.443 0.487 0.528 0.444 0.376 0.466 0.476 0.485 0.505 0.483 0.469 ( ↓19.83%)
w/o Ret 0.473 0.555 0.549 0.537 0.506 0.603 0.508 0.510 0.577 0.545 0.536 ( ↓8.38%)
w/o Intent 0.464 0.555 0.455 0.538 0.429 0.568 0.539 0.526 0.494 0.583 0.515 ( ↓11.97%)
LibRec 0.532 0.598 0.598 0.579 0.533 0.652 0.569 0.554 0.634 0.598 0.585
and reasoning models. This indicates that code LLMs struggle
to effectively recommend the correct target libraries, and
lack the capability to rank relevant libraries higher in the
recommendation list.
Finding 1: Our framework can recommend tar-
get libraries effectively. Meanwhile, the general LLM
Claude-3.7-Sonnet achieves the best performance across
all metrics. At the same time, reasoning models like
DeepSeek-R1 and QwQ-Plus demonstrate relatively sub-
optimal performance, highlighting the effectiveness of
reasoning LLMs in addressing the target library recom-
mendation task.
B. RQ2: Ablation Study
1) Setup: In this section, we conduct ablation studies to
explore the necessity of different components for the perfor-
mance of our framework. To ensure fairness, all implemen-
tation settings were kept consistent across experiments. As
shown in Table III, we introduce several ablation variants: w/o
Ret denotes the removal of the retrieval-augmented compo-
nent, w/o Intent represents the exclusion of migration intents
from the prompt, i.e., remove the context enhancement brought
by migration intents, and LibRec refers to our complete rec-
ommendation framework. Additionally, we provide a vanilla
baseline for each model, which only relies on the ability ofLLMs without incorporating retrieval or context enhancement.
These ablation studies allow us to systematically analyze the
contribution of each component to the overall performance of
our proposed framework.
2) Results: As shown in Table III, under vanilla condi-
tions, the average scores for Precision@1, Precision@3, Pre-
cision@5, Precision@10, and MRR are 38.8%, 53.5%, 57.5%,
61.5%, and 46.9%, respectively. The relatively low scores
highlight the inherent difficulty models face in recommending
target libraries based solely on their native capabilities. Sub-
sequently, under w/o Ret condition, the average scores are
46.1%, 59.8%, 63.1%, 67%, and 53.6%, reflecting declines
of 8.71%, 8.98%, 7.88%, 6.29%, and 8.38%, respectively,
compared to the complete framework. Similarly, under w/o
Intent condition, the average scores are 43.5%, 58.3%, 61.8%,
65.4%, and 51.5%, corresponding to declines of 13.86%,
11.26%, 9.78%, 8.53%, and 11.97%, respectively, compared
to the complete framework. These results give a hint that
all key components are essential for achieving the best
effectiveness.
Moreover, we observe that, on average, when generating
only the top-1 result, migration intents are more critical than
retrieval augmentation. However, as the number of output re-
sults increases, the significance of retrieval augmentation sur-
passes that of migration intents. Interestingly, different LLMs
behave differently. For instance, for most LLMs, including

10
T op-1 T op-3 T op-5T op-100.400.450.500.550.600.650.700.750.800.85
GPT-4o-mini
T op-1 T op-3 T op-5T op-100.400.450.500.550.600.650.700.750.800.85
GPT-4
T op-1 T op-3 T op-5T op-100.400.450.500.550.600.650.700.750.800.85
DeepSeek-V3
T op-1 T op-3 T op-5T op-100.400.450.500.550.600.650.700.750.800.85
Qwen-Plus
T op-1 T op-3 T op-5T op-100.400.450.500.550.600.650.700.750.800.85
Llama-3.3
T op-1 T op-3 T op-5T op-100.400.450.500.550.600.650.700.750.800.85
Claude-3.7-Sonnet
T op-1 T op-3 T op-5T op-100.400.450.500.550.600.650.700.750.800.85
Qwen-Coder-Plus
T op-1 T op-3 T op-5T op-100.400.450.500.550.600.650.700.750.800.85
Qwen2.5-Coder
T op-1 T op-3 T op-5T op-100.400.450.500.550.600.650.700.750.800.85
DeepSeek-R1
T op-1 T op-3 T op-5T op-100.400.450.500.550.600.650.700.750.800.85
QwQ-PlusZero-Shot One-Shot CoT
Fig. 7. Precision@K Trends of Various LLMs Across Different Prompt Strategies.
GPT-4o-mini, GPT-4, Llama-3.3, Qwen-Coder-Plus, Qwen2.5-
Coder, DeepSeek-R1, and QwQ-Plus, the retrieval augmenta-
tion component demonstrates greater importance across most
evaluation metrics compared to migration intents. However,
several LLMs of DeepSeek-V3, Qwen-Plus, and Claude-3.7-
Sonnet exhibit the opposite trend. This phenomenon may be
attributed to the fact that the LLMs of DeepSeek-V3, Qwen-
Plus, and Claude-3.7-Sonnet may possess stronger capabilities
in contextual understanding, enabling them to extract more
meaningful information from migration intents, thereby reduc-
ing their reliance on retrieval augmentation.
Finding 2: Both the retrieval-augmented component and
migration intents within our framework are essential for
achieving the best effectiveness. On average, retrieval
augmentation demonstrates greater significance compared
to migration intents; however, the relative importance of
these components varies across different LLMs.
C. RQ3: Effectiveness of Different Prompt Engineering Strate-
gies
As shown in Figure 7, the One-Shot prompt strategy
demonstrates superior performance across 6 models, includ-
ing 4 general LLMs of GPT-4o-mini, GPT-4, DeepSeek-
V3, Qwen-Plus, and 2 code LLMs of Qwen-Coder-Plus and
Qwen2.5-Coder. Among them, the One-Shot strategy achievesthe highest Precision@10 scores in GPT-4, DeepSeek-V3,
reaching 72.9% and 71.1%, respectively. These results con-
firm that the One-Shot strategy is the most reliable prompt
strategy for our task , as it provides a concrete example to
reduce model burden, thereby helping LLMs to generate target
libraries more effectively.
In contrast, the CoT prompt strategy exhibits
superior performance in only 3 models: 1 general
LLM of Claude-3.7-Sonnet, and 2 reasoning LLMs of
DeepSeek-R1 and QwQ-Plus. Among these, the CoT strategy
achieves its highest Precision@10 scores of 80.3% with
Claude-3.7-Sonnet. This phenomenon can be attributed to
the fact that reasoning LLMs, such as DeepSeek-R1 and
QwQ-Plus, are typically optimized for logical reasoning tasks
during their training process, endowing them with stronger
logical analysis and step-by-step reasoning capabilities.
Consequently, the CoT strategy can better unlock the
potential of these models, enabling superior performance in
multi-step problem-solving tasks. Furthermore, the design
characteristics of Claude-3.7-Sonnet allow it to maintain
logical coherence more effectively when addressing complex
reasoning tasks, which may contribute to its excellent
performance under the CoT strategy.

11
Not Maintained/OutdatedBug/Defect Issues
Security VulnerabilityUsability
Enhanced FeaturesPerformanceActivity
Popularity
Size/ComplexityIntegration
SimplificationLicense0.00.20.40.60.81.0Precision@10.61 0.600.71
Not Maintained/OutdatedBug/Defect Issues
Security VulnerabilityUsability
Enhanced FeaturesPerformanceActivity
Popularity
Size/ComplexityIntegration
SimplificationLicense0.00.20.40.60.81.0Precision@30.84
0.780.80
Not Maintained/OutdatedBug/Defect Issues
Security VulnerabilityUsability
Enhanced FeaturesPerformanceActivity
Popularity
Size/ComplexityIntegration
SimplificationLicense0.00.20.40.60.81.0Precision@50.88
0.800.82
Not Maintained/OutdatedBug/Defect Issues
Security VulnerabilityUsability
Enhanced FeaturesPerformanceActivity
Popularity
Size/ComplexityIntegration
SimplificationLicense0.00.20.40.60.81.0Precision@100.91
0.830.85Source Library Issues T arget Library Advantages Project Specific Reasons
Fig. 8. The effectiveness of our framework (take the best-performing LLM of Claude-3.7-Sonnet as an example) across various intent types. The horizontal
axis represents all sub-categories of intent types, and different colors indicate distinct categories. The values on the dotted line denote the average score of
our framework in each category.
Finding 3: The One-shot prompt strategy emerges as the
most robust and reliable paradigm across six models, con-
sistently demonstrating superior performance. In contrast,
the CoT prompt strategy excels primarily in reasoning
LLMs and high-capacity models like Claude-3.7-Sonnet.
D. RQ4: Effectiveness across Different Intent Types
In this RQ, we analyzed the target library recommendation
performance of our framework across different intent types, us-
ing the Claude-3.7-Sonnet model as an example, as illustrated
in Figure 8. Figure 8 reports the Precision@1, Precision@3,
Precision@5, and Precision@10 scores of our framework for
each intent category and its subcategories. Among these, the
category Source Library Issues achieves the highest average
scores of 84%, 88%, and 91% for precision@3, precision@5,
and precision@10, respectively, followed by Project Specific
Reasons . However, the category Target Library Advantages
exhibits the lowest average scores. The superior performance
ofSource Library Issues may be attributed to its clear and
concise description when a library is outdated or vulnerable.
Conversely, the relatively lower performance of Target LibraryIssues could be due to the large number of subcategories,
which may require more detailed contextual information to
achieve accurate target library ranking recommendations.
Within the Source Library Issues category, our frame-
work performs better in the subcategory Security Vulnerabil-
ity, achieving Precision@3, precision@5, and precision@10
scores of 92%, 96%, and 96%, respectively. However, the
performance is the weakest in the subcategory Bug/Defect
Issues , with scores of 74%, 81%, and 84%, reflecting de-
clines of 18%, 15%, 12%, respectively, compared to the best-
performing subcategory.
Within the Project Specific Reasons category, our frame-
work demonstrates superior performance in the subcategory
License , achieving near-perfect accuracy across all metrics.
This strong performance can be attributed to the binary nature
of licensing requirements, which impose clear constraints that
our search and ranking pipeline can effectively leverage to
recommend the correct target libraries with high confidence.
As for the Target Library Advantages category, our framework
performs better in the subcategories of Size/Complexity and
Activity , both achieving Precision@3 and Precision@5 scores
of 82% and 83%, respectively. However, the performance is

12
notably weaker in the subcategory Performance , with Preci-
sion@3, precision@5, and precision@10 scores of 68%, 74%,
and 79%, respectively.
Finding 4: Our framework demonstrates superior perfor-
mance in the Source Library Issues category, followed
byProject Specific Reasons , while exhibiting the lowest
performance in the Target Library Advantages category.
Notably, within the Source Library Issues category, the
framework excels in the sub-category Security Vulnerabil-
ity. Similarly, in the Project Specific Reasons category, the
framework attains near-perfect accuracy across all metrics
in the sub-category License . However, its performance
is comparatively weaker in the subcategory Performance
within the Target Library Advantages category.
E. RQ5: Case Study
Enhanced Features
Usability
Activity
Popularity
PerformanceSize/ComplexityIntegrationSimplificationNot Maintained/OutdatedBug/Defect IssuesSecurity VulnerabilityOtherT arget Library Advantages
Project Specific ReasonsSource Library Issues
Other
Fig. 9. The distribution of intent types among failed cases.
In this RQ, we analyze the characteristics of failed cases.
Figure 9 illustrates the distribution of intent types among
these failures. Consistent with the findings of RQ4, most
of the recommendation failures fall into the Target Library
Advantage category, while the Source Library Issues category
exhibits the fewest failures. Notably, within the Target Library
Advantage category, the Enhanced Features sub-category rep-
resents the largest proportion, comprising 46% of the total
failures in this group. This may indicate that recommending
target libraries based on subtle feature improvements remains
a significant challenge for LLMs. For instance, in the project
movies-python-bolt , under the migration intent Update Flask
demo app to use bolt driver , the developer replaces the
source library neo4jrestclient with the target library
neo4j-driver .
In contrast, within the Source Library Issues category, the
Security Vulnerability sub-category accounts for the smallest
proportion, representing only 6% of the total failures in this
0.0 0.2 0.4 0.6 0.8 1.0
Failure RateSecurity
Vulnerability
Not
Maintained/Out
dated
Usability
Popularity
PerformanceSubcategory1.00
0.87
0.81
0.91
0.500.33
0.43
0.62
0.73
0.67DeepSeek-R1 Claude-3.7-SonnetFig. 10. Comparison of failure rates for Claude-3.7-Sonnet and DeepSeek-
R1 when recommending target libraries across different subcategories (top 5
subcategories with the most significant performance disparities).
group. For example, in the project steam with the migration
intent replace cryptography with pycryptodomex , the developer
replaces the source library cryptography with the target
library pycryptodomex . Meanwhile, in the Project Specific
Reasons category, the Integration sub-category constitutes the
largest proportion, accounting for 66% of the total failures
in this group. For instance, in the project pax, under the
migration intent ’Better integrate Pyxie’, ’Better utilize Con-
figuration system’ , the developer replaces the source library
configglue with the target library confiture .
Furthermore, our findings in RQ1 reveal that the overall
performance of Claude-3.7-Sonnet and DeepSeek-R1 is com-
parable. To gain deeper insights, we further analyze their
failure rates when recommending target libraries across dif-
ferent sub-categories. Specifically, we calculated the failure
rates for both models within each subcategory, and identified
the five subcategories with the most significant performance
disparities, as depicted in Figure 10. Our analysis shows that in
theSecurity Vulnerability subcategory, DeepSeek-R1 exhibits
a 100% failure rate, whereas Claude-3.7-Sonnet demonstrates
a lower failure rate of 33%. This indicates that Claude-3.7-
Sonnet is more adept at recommending target libraries for
migration intents involving Security Vulnerability . For the Not
Maintained/Outdated ,Usability , and Popularity subcategories,
DeepSeek-R1’s failure rates exceed 81%, surpassing those of
Claude-3.7-Sonnet by 44%, 19%, and 18%, respectively. How-
ever, it is worth noting that Claude-3.7-Sonnet also exhibits a
relatively high failure rate in the Popularity subcategory.
Interestingly, in the Performance subcategory, DeepSeek-
R1 demonstrates a relatively low failure rate of 50%, whereas
Claude-3.7-Sonnet shows a higher failure rate in this category.
This suggests that DeepSeek-R1 is more effective at recom-
mending target libraries with migration intent of Performance .
These findings indicate that although the two models achieve
similar overall performance, their capabilities in recommend-
ing target libraries vary across different migration intents,
highlighting the strengths and weaknesses of each model in
specific contexts.

13
Finding 5: Most of the recommendation failures fall into
theTarget Library Advantage category, with Enhanced
Features sub-category representing the largest proportion
of failures within this group. Conversely, the Source
Library Issues category accounts for the fewest failures,
with the Security Vulnerability sub-category making up the
smallest proportion within this category. In most subcate-
gories of intent types, DeepSeek-R1 demonstrates a higher
failure rate compared to Claude-3.7-Sonnet, while it shows
a lower failure rate in the Performance subcategory.
VII. D ISCUSSION
In this section, we discuss the implications of our study and
propose recommendations for researchers, software develop-
ers, and users.
Our findings in RQ1 suggest that the superior performance
of Claude-3.7-Sonnet across all metrics establishes it as a
benchmark model within our framework for library recom-
mendation tasks. Researchers can leverage this model as a
reference point for evaluating new models or approaches in
this domain. Furthermore, the relatively high performance of
reasoning LLMs like DeepSeek-R1 and QwQ-Plus highlights
the importance of reasoning capabilities in addressing library
recommendation tasks. This finding highlights an opportunity
for researchers to investigate how reasoning architectures or
training strategies can further enhance performance in recom-
mendation tasks. In contrast, the inferior performance of code
LLMs reveals an avenue for future research aimed at improv-
ing their ability to rank and recommend libraries effectively,
thereby enabling them to recommend suitable libraries in code
generation tasks.
Forsoftware developers , prioritizing the adoption of Claude-
3.7-Sonnet is strongly recommended, as it demonstrates the
best overall performance. Nonetheless, reasoning LLMs like
DeepSeek-R1 and QwQ-Plus may be a viable alternative
due to their comparable performance in the target library
recommendation task. Conversely, the inferior performance
of code LLMs indicates that they may not be suitable for
tasks that require ranking or recommending target libraries.
Developers should avoid relying solely on code LLMs for
such tasks and instead consider integrating them with general
or reasoning models to achieve superior performance.
Forusers , it is essential to be aware of the distinct strengths
and limitations of general, reasoning, and code LLMs, espe-
cially that the Claude-3.7-Sonnet currently offers the highest
performance for library recommendation tasks. This under-
standing will enable users to make more informed decisions
when selecting libraries or tools for their needs.
Our findings in RQ2 highlight the significance of both
the retrieval augmentation and migration intents components
in our framework. To achieve optimal effectiveness in library
recommendation tasks, researchers are encouraged to integrate
both components. Besides, both researchers and software
developers should be aware of the specific characteristics of
the LLMs they employ. For instance, models like DeepSeek-
V3, Qwen-Plus, and Claude-3.7-Sonnet may benefit more from
enhanced migration intents, whereas others might rely moreheavily on retrieval augmentation. In this regard, they could
explore combining LLMs with diverse capabilities to further
enhance the overall performance of library recommendation
tasks.
Our findings in RQ3 indicate that the One-Shot prompt
strategy demonstrates superior effectiveness across a wider
range of models for library recommendation tasks. Re-
searchers are encouraged to prioritize this strategy when
developing or evaluating new models to enhance overall per-
formance. Conversely, the superior performance of the COT
strategy with reasoning LLMs like DeepSeek-R1 and QwQ-
Plus highlights the importance of tailoring prompt strategies
to align with the strengths of specific models. Researchers
could focus on optimizing reasoning LLMs to fully exploit
the potential of the CoT strategy.
Given that different prompt strategies play distinct roles in
different types of LLMs, it is recommended that developers
and users select prompt strategies based on the characteristics
of the models they employ. For general and code LLMs, the
One-Shot strategy may yield better results, while the CoT
strategy could be more beneficial for reasoning LLMs.
Our findings in RQ4 reveal that the performance of
our framework varies across different intent categories and
sub-categories. Researchers and developers should focus on
optimizing recommendation frameworks for intent types that
currently exhibit weaker performance. For instance, the frame-
work demonstrates weaker performance in the Target Library
Advantages category and its sub-category Performance . To
address these limitations, researchers and developers could
incorporate more detailed contextual information into prompts,
or augment the models with additional data sources to better
address these issues.
Furthermore, to improve the accuracy of target library
recommendations, users should provide clear and concise
contextual information, especially when dealing with intents
such as Target Library Advantages , where detailed inputs
are critical for achieving better performance. Additionally,
users should critically evaluate the recommendation results
for weaker-performing sub-categories, such as Performance or
Bug/Defect Issues , and consider supplementing the system’s
outputs with manual analysis or verification.
Our findings in RQ5 highlight that the high propor-
tion of failures under the Enhanced Features subcategory
within the Target Library Advantage category underscores
the challenge of recommending libraries based on subtle fea-
ture improvements. Researchers should focus on developing
advanced methods to better capture nuanced differences be-
tween libraries, potentially by incorporating advanced feature-
comparison techniques or leveraging fine-grained contextual
embeddings, to enhance the recommendation performance.
Besides, our findings unveil significant differences in failure
rates between Claude-3.7-Sonnet and DeepSeek-R1 across
various subcategories. In this regard, researchers should in-
vestigate the underlying mechanisms that contribute to these
disparities. For example, Claude-3.7-Sonnet’s superior perfor-
mance in the Security Vulnerability subcategory suggests that
it is better equipped to handle well-defined and critical issues.
Researchers could analyze and explore ways to replicate these

14
strengths in other models to improve their performance.
Fordevelopers , they could explore the possibility of creating
hybrid recommendation frameworks that leverage the strengths
of both Claude-3.7-Sonnet and DeepSeek-R1. For instance, the
system could dynamically switch between models according
to the migration intent, using Claude-3.7-Sonnet for Security
Vulnerability and DeepSeek-R1 for Performance .
Moreover, to enhance recommendation accuracy, users
should provide as much detailed contextual information as
possible, particularly for intents such as Enhanced Features ,
where subtle differences between libraries are critical. Clear
descriptions of the desired features or improvements can help
the recommendation framework better understand user needs
and generate more accurate and reliable results.
VIII. T HREATS TOVALIDITY
Internal Validity. Our primary threat to internal validity
relates to the fact that we only extract candidate migration
commits from the requirement.txt dependency file. Given that
Python projects may use other types of dependency files, such
as setup.py, and that dependency files are often manually
named and maintained, there exists a risk of omissions in
our collection of candidate migration commits. However, prior
research [46], indicates that the majority of configuration
information in Python projects is typically stored in the
requirements.txt file. Moreover, prior studies [46] have also
only relied on requirement.txt to extract library dependencies.
In this regard, our threat to internal validity can be mitigated
to a certain degree.
External Validity. Threats to external validity pertain to the
generalizability of our results. The first threat concerns the
representativeness of our constructed benchmark with respect
to the entire Python library ecosystem. Although it is possible
that our benchmark does not encompass all library migrations
within the ecosystem, we conducted a large-scale study in-
volving 62,986 Python repositories and collected a total of
2,888 records for our benchmark. We believe it is a good
representation of the Python ecosystem.
The second threat relates to the applicability of our results
beyond Python. However, Python has emerged as the most
popular programming language on GitHub since 2024, and is
widely adopted in AI-related domains. As a result, our study
can bring valuable insights for the AI field and can serve as a
reference for similar research conducted in other programming
languages.
The final threat pertains to the use of LLMs. To strengthen
the reliability of our results, we employed three categories of
LLMs, including general LLMs, reasoning LLMs, and code
LLMs, encompassing a total of ten models. This compre-
hensive approach ensures a more robust evaluation of their
capabilities in our task.
IX. C ONCLUSION
In this paper, we propose LibRec, a framework that inte-
grates the capabilities of LLMs with RAG techniques and in-
context learning of migration intents to automate the recom-
mendation of target libraries. Additionally, we present LibEval,a novel benchmark designed to evaluate the performance of
target library recommendation tasks. Leveraging LibEval, we
evaluated the effectiveness of ten popular LLMs within our
framework (RQ1), conducted an ablation study to assess the
contributions of key components (RQ2), analyzed the impact
of various prompt strategies on the framework’s performance
(RQ3), evaluated its effectiveness across different intent types
(RQ4), and performed detailed failure case analyses (RQ5).
Our experimental findings demonstrate that LibRec achieves
overall optimal performance in recommending target libraries,
with both retrieval-augmentation and migration intent com-
ponents being critical for optimal effectiveness. The one-
shot prompt strategy emerges as the most robust and reliable
paradigm across six LLMs, consistently delivering superior
performance. Furthermore, the framework exhibits superior
performance for intent types related to issues with source
libraries.
In future work, we plan to incorporate more LLMs in our
framework and design another framework to generate migra-
tion code automatically based on the recommended libraries
of our framework in this paper.
REFERENCES
[1] Y . Wang, B. Chen, K. Huang, B. Shi, C. Xu, X. Peng,
Y . Wu, and Y . Liu, “An empirical study of usages,
updates and risks of third-party libraries in java projects,”
in2020 IEEE International Conference on Software
Maintenance and Evolution (ICSME) . IEEE, 2020, pp.
35–45.
[2] H. He, Y . Xu, Y . Ma, Y . Xu, G. Liang, and M. Zhou, “A
multi-metric ranking approach for library migration rec-
ommendations,” in 2021 ieee international conference on
software analysis, evolution and reengineering (saner) .
IEEE, 2021, pp. 72–83.
[3] J. Wang, L. Li, K. Liu, and H. Cai, “Exploring how
deprecated python library apis are (not) handled,” in
Proceedings of the 28th acm joint meeting on european
software engineering conference and symposium on the
foundations of software engineering , 2020, pp. 233–244.
[4] R. G. Kula, D. M. German, A. Ouni, T. Ishio, and K. In-
oue, “Do developers update their library dependencies?
an empirical study on the impact of security advisories
on library migration,” Empirical Software Engineering ,
vol. 23, pp. 384–417, 2018.
[5] R. Ossendrijver, S. Schroevers, and C. Grelck, “To-
wards automated library migrations with error prone
and refaster,” in Proceedings of the 37th ACM/SIGAPP
Symposium on Applied Computing , 2022, pp. 1598–1606.
[6] M. Islam, A. K. Jha, I. Akhmetov, and S. Nadi, “Char-
acterizing python library migrations,” Proceedings of the
ACM on Software Engineering , vol. 1, no. FSE, pp. 92–
114, 2024.
[7] E. Larios Vargas, M. Aniche, C. Treude, M. Bruntink,
and G. Gousios, “Selecting third-party libraries: The
practitioners’ perspective,” in Proceedings of the 28th
ACM joint meeting on european software engineering
conference and symposium on the foundations of soft-
ware engineering , 2020, pp. 245–256.

15
[8] C. Teyton, J.-R. Falleri, M. Palyart, and X. Blanc, “A
study of library migrations in java,” Journal of Software:
Evolution and Process , vol. 26, no. 11, pp. 1030–1052,
2014.
[9] H. He, R. He, H. Gu, and M. Zhou, “A large-scale
empirical study on java library migrations: prevalence,
trends, and rationales,” in Proceedings of the 29th ACM
joint meeting on European software engineering con-
ference and symposium on the foundations of software
engineering , 2021, pp. 478–490.
[10] H. Gu, H. He, and M. Zhou, “Self-admitted library
migrations in java, javascript, and python packaging
ecosystems: A comparative study,” in 2023 IEEE Inter-
national Conference on Software Analysis, Evolution and
Reengineering (SANER) . IEEE, 2023, pp. 627–638.
[11] J. Zhang, Q. Luo, and P. Wu, “A multi-metric ranking
with label correlations approach for library migration
recommendations,” in 2024 IEEE International Confer-
ence on Software Analysis, Evolution and Reengineering
(SANER) . IEEE, 2024, pp. 183–191.
[12] S. Mujahid, D. E. Costa, R. Abdalkareem, and E. Shi-
hab, “Where to go now? finding alternatives for de-
clining packages in the npm ecosystem,” in 2023 38th
IEEE/ACM International Conference on Automated Soft-
ware Engineering (ASE) . IEEE, 2023, pp. 1628–1639.
[13] J. Li, S. Hyun, and M. A. Babar, “Ddpt: Diffusion-
driven prompt tuning for large language model code
generation,” arXiv preprint arXiv:2504.04351 , 2025.
[14] J. Jiang, F. Wang, J. Shen, S. Kim, and S. Kim, “A survey
on large language models for code generation,” arXiv
preprint arXiv:2406.00515 , 2024.
[15] A. Eghbali and M. Pradel, “De-hallucinator: Itera-
tive grounding for llm-based code completion,” arXiv
preprint arXiv:2401.01701 , 2024.
[16] W. Liu, A. Yu, D. Zan, B. Shen, W. Zhang, H. Zhao,
Z. Jin, and Q. Wang, “Graphcoder: Enhancing repository-
level code completion via coarse-to-fine retrieval based
on code context graph,” in Proceedings of the 39th
IEEE/ACM International Conference on Automated Soft-
ware Engineering , 2024, pp. 570–581.
[17] A. Deljouyi, R. Koohestani, M. Izadi, and A. Zaidman,
“Leveraging large language models for enhancing the
understandability of generated unit tests,” arXiv preprint
arXiv:2408.11710 , 2024.
[18] Y . Chen, Z. Hu, C. Zhi, J. Han, S. Deng, and J. Yin,
“Chatunitest: A framework for llm-based test gener-
ation,” in Companion Proceedings of the 32nd ACM
International Conference on the Foundations of Software
Engineering , 2024, pp. 572–576.
[19] A. Yildiz, S. G. Teo, Y . Lou, Y . Feng, C. Wang, and D. M.
Divakaran, “Benchmarking llms and llm-based agents in
practical vulnerability detection for code repositories,”
arXiv preprint arXiv:2503.03586 , 2025.
[20] J. Lin and D. Mohaisen, “From large to mammoth: A
comparative evaluation of large language models in vul-
nerability detection,” in Proceedings of the 2025 Network
and Distributed System Security Symposium (NDSS) ,
2025.[21] Y . Chen, C. Gao, M. Zhu, Q. Liao, Y . Wang, and G. Xu,
“Apigen: Generative api method recommendation,” in
2024 IEEE International Conference on Software Analy-
sis, Evolution and Reengineering (SANER) . IEEE, 2024,
pp. 171–182.
[22] Y . Qin, S. Liang, Y . Ye, K. Zhu, L. Yan, Y . Lu, Y . Lin,
X. Cong, X. Tang, B. Qian et al. , “Toolllm: Facilitating
large language models to master 16000+ real-world apis,”
arXiv preprint arXiv:2307.16789 , 2023.
[23] S. G. Patil, T. Zhang, X. Wang, and J. E. Gonzalez,
“Gorilla: Large language model connected with massive
apis,” Advances in Neural Information Processing Sys-
tems, vol. 37, pp. 126 544–126 565, 2024.
[24] P. Lewis, E. Perez, A. Piktus, F. Petroni, V . Karpukhin,
N. Goyal, H. K ¨uttler, M. Lewis, W.-t. Yih, T. Rockt ¨aschel
et al. , “Retrieval-augmented generation for knowledge-
intensive nlp tasks,” Advances in neural information
processing systems , vol. 33, pp. 9459–9474, 2020.
[25] K. Guu, K. Lee, Z. Tung, P. Pasupat, and M. Chang,
“Retrieval augmented language model pre-training,” in
International conference on machine learning . PMLR,
2020, pp. 3929–3938.
[26] S. Nan, J. Wang, N. Zhang, D. Li, and B. Li, “Ddasr:
Deep diverse api sequence recommendation,” ACM
Transactions on Software Engineering and Methodology ,
2024.
[27] Y . Wang, L. Chen, C. Gao, Y . Fang, and Y . Li, “Prompt
enhance api recommendation: visualize the user’s real
intention behind this query,” Automated Software Engi-
neering , vol. 31, no. 1, p. 27, 2024.
[28] J. Yang, W. Wu, and J. Ren, “Dskipp: A prompt method
to enhance the reliability in llms for java api recommen-
dation task,” Software Testing, Verification and Reliabil-
ity, vol. 35, no. 2, p. e1913, 2025.
[29] J. Latendresse, S. Khatoonabadi, A. Abdellatif, and
E. Shihab, “Is chatgpt a good software librarian? an ex-
ploratory study on the use of chatgpt for software library
recommendations,” arXiv preprint arXiv:2408.05128 ,
2024.
[30] Y . Zhang, Z. Jin, Y . Xing, G. Li, F. Liu, J. Zhu,
W. Dou, and J. Wei, “Patch: Empowering large lan-
guage model with programmer-intent guidance and
collaborative-behavior simulation for automatic bug fix-
ing,” ACM Transactions on Software Engineering and
Methodology , 2025.
[31] A. Eliseeva, Y . Sokolov, E. Bogomolov, Y . Golubev,
D. Dig, and T. Bryksin, “From commit message gen-
eration to history-aware commit message completion,”
in2023 38th IEEE/ACM International Conference on
Automated Software Engineering (ASE) . IEEE, 2023,
pp. 723–735.
[32] L. Choshen and I. Amit, “Comsum: Commit messages
summarization and meaning preservation,” arXiv preprint
arXiv:2108.10763 , 2021.
[33] S. Robertson, H. Zaragoza et al. , “The probabilistic
relevance framework: Bm25 and beyond,” Foundations
and Trends® in Information Retrieval , vol. 3, no. 4, pp.
333–389, 2009.

16
[34] B. Wei, Y . Li, G. Li, X. Xia, and Z. Jin, “Retrieve
and refine: exemplar-based neural comment generation,”
inProceedings of the 35th IEEE/ACM International
Conference on Automated Software Engineering , 2020,
pp. 349–360.
[35] J. A. Li, Y . Li, G. Li, X. Hu, X. Xia, and Z. Jin,
“Editsum: A retrieve-and-edit framework for source code
summarization,” in 2021 36th IEEE/ACM International
Conference on Automated Software Engineering (ASE) .
IEEE, 2021, pp. 155–166.
[36] G. Staff, “Octoverse: Ai leads python to top language
as the number of global developers surges,” The GitHub
Blog.(Oct. 29, 2024). Retrieved Feb , vol. 10, p. 2025,
2024.
[37] J. Coelho, M. T. Valente, L. Milen, and L. L. Silva, “Is
this github project maintained? measuring the level of
maintenance activity of open-source projects,” Informa-
tion and Software Technology , vol. 122, p. 106274, 2020.
[38] J. Han, Y . Wang, Z. Liu, L. Bao, J. Liu, D. Lo,
and S. Deng, “Sustainability forecasting for deep learn-
ing packages,” in 2024 IEEE International Confer-
ence on Software Analysis, Evolution and Reengineering
(SANER) . IEEE, 2024, pp. 981–992.
[39] M. Islam, A. K. Jha, S. Nadi, and I. Akhmetov,
“Pymigbench: A benchmark for python library migra-
tion,” in 2023 IEEE/ACM 20th International Conference
on Mining Software Repositories (MSR) . IEEE, 2023,
pp. 511–515.
[40] C. Chen, Z. Xing, and Y . Liu, “What’s spain’s paris?mining analogical libraries from q&a discussions,” Em-
pirical Software Engineering , vol. 24, pp. 1155–1194,
2019.
[41] M. Geng, S. Wang, D. Dong, H. Wang, G. Li, Z. Jin,
X. Mao, and X. Liao, “Large language models are
few-shot summarizers: Multi-intent comment generation
via in-context learning,” in Proceedings of the 46th
IEEE/ACM International Conference on Software Engi-
neering , 2024, pp. 1–13.
[42] S. Boslaugh, Statistics in a nutshell: A desktop quick
reference . ” O’Reilly Media, Inc.”, 2012.
[43] R. Zhong, Y . Huo, W. Gu, J. Kuang, Z. Jiang, G. Yu,
Y . Li, D. Lo, and M. R. Lyu, “Ccisolver: End-to-end
detection and repair of method-level code-comment in-
consistency,” arXiv preprint arXiv:2506.20558 , 2025.
[44] M. Wei, N. S. Harzevili, Y . Huang, J. Wang, and S. Wang,
“Clear: contrastive learning for api recommendation,”
inProceedings of the 44th International Conference on
Software Engineering , 2022, pp. 376–387.
[45] J. Chen, S. Chen, J. Cao, J. Shen, and S.-C. Cheung,
“When llms meet api documentation: Can retrieval aug-
mentation aid code generation just as it helps develop-
ers?” arXiv preprint arXiv:2503.15231 , 2025.
[46] J. Han, S. Deng, D. Lo, C. Zhi, J. Yin, and X. Xia,
“An empirical study of the dependency networks of deep
learning libraries,” in 2020 IEEE international confer-
ence on software maintenance and evolution (ICSME) .
IEEE, 2020, pp. 868–878.