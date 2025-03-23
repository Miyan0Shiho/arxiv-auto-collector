# SOSecure: Safer Code Generation with RAG and StackOverflow Discussions

**Authors**: Manisha Mukherjee, Vincent J. Hellendoorn

**Published**: 2025-03-17 19:03:36

**PDF URL**: [http://arxiv.org/pdf/2503.13654v1](http://arxiv.org/pdf/2503.13654v1)

## Abstract
Large Language Models (LLMs) are widely used for automated code generation.
Their reliance on infrequently updated pretraining data leaves them unaware of
newly discovered vulnerabilities and evolving security standards, making them
prone to producing insecure code. In contrast, developer communities on Stack
Overflow (SO) provide an ever-evolving repository of knowledge, where security
vulnerabilities are actively discussed and addressed through collective
expertise. These community-driven insights remain largely untapped by LLMs.
This paper introduces SOSecure, a Retrieval-Augmented Generation (RAG) system
that leverages the collective security expertise found in SO discussions to
improve the security of LLM-generated code. We build a security-focused
knowledge base by extracting SO answers and comments that explicitly identify
vulnerabilities. Unlike common uses of RAG, SOSecure triggers after code has
been generated to find discussions that identify flaws in similar code. These
are used in a prompt to an LLM to consider revising the code. Evaluation across
three datasets (SALLM, LLMSecEval, and LMSys) show that SOSecure achieves
strong fix rates of 71.7%, 91.3%, and 96.7% respectively, compared to prompting
GPT-4 without relevant discussions (49.1%, 56.5%, and 37.5%), and outperforms
multiple other baselines. SOSecure operates as a language-agnostic complement
to existing LLMs, without requiring retraining or fine-tuning, making it easy
to deploy. Our results underscore the importance of maintaining active
developer forums, which have dropped substantially in usage with LLM adoptions.

## Full Text


<!-- PDF content starts -->

SOSecure: Safer Code Generation with RAG and StackOverflow
Discussions
Manisha Mukherjee
Carnegie Mellon University
Pittsburgh, PA, USAVincent J. Hellendoorn
Carnegie Mellon University
Pittsburgh, PA, USA
ABSTRACT
Large Language Models (LLMs) are widely used for automated code
generation. Their reliance on infrequently updated pretraining data
leaves them unaware of newly discovered vulnerabilities and evolv-
ing security standards, making them prone to producing insecure
code. In contrast, developer communities on Stack Overflow (SO)
provide an ever-evolving repository of knowledge, where security
vulnerabilities are actively discussed and addressed through col-
lective expertise. These community-driven insights remain largely
untapped by LLMs. This paper introduces SOSecure, a Retrieval-
Augmented Generation (RAG) system that leverages the collective
security expertise found in SO discussions to improve the security
of LLM-generated code. We build a security-focused knowledge
base by extracting SO answers and comments that explicitly iden-
tify vulnerabilities. Unlike common uses of RAG, SOSecure triggers
after code has been generated to find discussions that identify flaws
in similar code. These are used in a prompt to an LLM to con-
sider revising the code. Evaluation across three datasets (SALLM,
LLMSecEval, and LMSys) show that SOSecure achieves strong fix
rates of 71.7%, 91.3%, and 96.7% respectively, compared to prompt-
ing GPT-4 without relevant discussions (49.1%, 56.5%, and 37.5%),
and outperforms multiple other baselines. SOSecure operates as a
language-agnostic complement to existing LLMs, without requir-
ing retraining or fine-tuning, making it easy to deploy. Our results
underscore the importance of maintaining active developer forums,
which have dropped substantially in usage with LLM adoptions.
1 INTRODUCTION
LLM-powered code generation tools, such as Microsoft GitHub
Copilot and OpenAI ChatGPT, have significantly improved soft-
ware development efficiency [ 8]. However, these tools can inherit
security flaws from their open-source training data. Programming
languages and libraries evolve constantly (e.g., TensorFlow has new
releases every couple of months [ 29]). This evolution often involves
patching vulnerabilities and replacing unsafe patterns with safe
ones. A large fraction of open-source repositories, which LLMs
use for pretraining, are rarely updated, leading LLMs to learn and
replicate vulnerable patterns, including CWEs (common weakness
enumerations) [ 20]. Additionally, due to capacity limitations, LLMs
also often lack awareness of subtle security implications specific to
particular libraries or contexts.
Attackers can exploit these flaws, resulting in cyberattacks, data
breaches, and degraded system performance. Despite these risks, de-
velopers frequently trust LLM-generated code without thoroughly
validating its security implications, increasing the likelihood of
introducing vulnerabilities into production systems [10, 11].
SO and similar developer Q&A forums present a contrasting ap-
proach to knowledge sharing. SO is a developer Q&A forum that has
Answer
Comment
Figure 1: AnswerID: 61307412, which includes community
comments providing security insights. This content was used
as context to enhance the generated code in SOSecure.
served as a vast repository of programming knowledge accumulated
through collective expertise over more than 15 years. It encourages
revising and replacing obsolete or problematic answers. Community
members often highlight security concerns in comments, providing
invaluable context about why certain approaches might be risky,
and regularly suggesting more secure alternatives. SO answers can
be updated years after they were originally posted, allowing for
corrections and revisions as technologies change, making them
valuable references over time. In contrast, LLMs are retrained rela-
tively infrequently, and their answers are transient, produced on
a one-off basis, typically non-deterministically, with no in-built
mechanism for community validation or correction.
To reduce the impact of their knowledge gaps, LLMs may use
Retrieval-Augmented Generation (RAG), including from SO, before
generating a response. While this might allow them to discover
vulnerability-related discussions, the retrieval query is typically
based on the user’s prompt, which is unlikely to elicit vulnerability-
related discussions. SO also contains many older and outdated
answers that may actually recommend unsafe snippets. In this
work, we address these issues by proposing SOSecure: an approach
focused on revising potentially vulnerable code in LLM-generated
answers (and code snippets more generally) by leveraging an index
of vulnerability-oriented SO discussions. SOSecure functions as a
security-enhancing layer in the code generation pipeline. When aarXiv:2503.13654v1  [cs.SE]  17 Mar 2025

Preprint, arXiv, Mukherjee et al.
user requests code from an LLM, the code it generates is compared
to relevant discussions from SOSecure’s security knowledge base,
which consists of answers and comments that contain similar code
patterns and explicitly mention security concerns. These retrieved
discussions are then provided as additional context to the LLM along
with the original code, asking if any changes would be appropriate.
The LLM may then generate a revised version addressing potential
security issues highlighted by the community discussions, or it
may determine that no changes are necessary, if the code already
follows security best practices. SOSecure works with any existing
code generation LLM as a complementary security layer, leveraging
the collective wisdom of the developer community to address se-
curity gaps. Our results demonstrate that this approach effectively
mitigates common security vulnerabilities, and that the retrieval of
relevant SO discussions is key to its success, even outperforming
LLMs prompted with the specific CWE to repair.
Overall, we make the following contributions:
•We release SOSecure, a novel approach that bridges the gap
between static LLM knowledge and evolving community
security insights. It uses a security-oriented knowledge base
constructed from SO discussions that specifically focuses on
community-identified vulnerabilities and security concerns.
•We demonstrate the generalizability of the framework by
evaluating SOSecure across multiple datasets showing that
our approach significantly reduces security vulnerabilities
in LLM-generated code.
2 BACKGROUND
2.1 LLM generated code security
Large Language Models are designed for general applications and
can be specialized for coding [ 1,6,15,33]. They show the ability
to generate functionally correct code and solve competitive pro-
gramming problems. This deep understanding of code comes from
pretraining on large volumes of code.
Several studies have assessed the security of code generated
by pretrained LMs, consistently finding that all evaluated LLMs
frequently produce security vulnerabilities. Recent research on
ChatGPT, an instruction-tuned LM, revealed that it generates code
below minimal security standards for 16 out of 21 cases and is only
able to self-correct 7 cases after further prompting [ 12]. Despite
there having been some efforts to address these issues[ 2,9,26,27,
32], security concerns in LLM-generated code remains an early-
stage research topic with significant challenges.
2.2 Program Security
CWE2is a publicly accessible classification system of common
software and hardware security vulnerabilities. Each weakness
type within this enumeration is assigned a unique identifier (CWE
ID). Program Security involves using various tools and datasets to
identify and prevent vulnerabilities. CodeQL [ 7] and Bandit [ 22]
represent two important static analysis approaches: CodeQL is a
leading industry tool that allows custom queries for detecting se-
curity vulnerabilities across mainstream languages, while Bandit
2https://cwe.mitre.org/data/index.htmlserves as a Python-specific security linter designed to identify com-
mon security issues in Python code. Both tools have proven reliable
for evaluating security in LM-generated code [26, 27].
The quality of vulnerability datasets is crucial for effective secu-
rity analysis. Many existing datasets are constructed from vulnera-
bility fix commits, simply treating pre-commit code as vulnerable
and post-commit versions as secure. Despite researchers trying to
address this problem of having reliable datasets through expen-
sive manual inspection to ensure accurate labeling, research has
revealed this approach can lead to incorrect security labels [21].
To overcome these limitations, our work utilizes a diverse mix-
ture of datasets collected through complementary techniques: auto-
mated data collection pipelines refined through manual inspection
(SALLM [ 27] - providing broader CWE coverage with prompts as
code, LLMSecEval [ 30] - providing prompts as natural language),
and real-world conversation datasets (LMSys [ 34]) that capture
authentic user interactions. This comprehensive approach allows
us to evaluate security vulnerabilities across different contexts and
collection methodologies.
2.3 Crowd security discussion analysis
Previous work has analyzed security-related discussions in devel-
oper communities. Mukherjee and Hellendoorn [ 18] collect SO data
and study how to classify obsolete answers in StackOverflow. Mey-
ers et al .[17] collected and annotated a corpus of conversations
from bug reports in the Chromium project, examining linguistic
metrics related to pragmatics. Building on this foundation, Le et al .
[13] applied topic modeling methods (LDA) to identify 13 main
security-related topics on SO. These studies primarily characterized
security discussions through topic modeling and qualitative anal-
ysis, providing insights into how developers communicate about
security concerns. Fischer et al .[5] show that nudge-based secu-
rity advice improves users’ choices, leading to safer programming
practices for Android development.
2.4 RAG based systems
The rapidly evolving security landscape presents challenges for
LLMs. In recent years, 20,000-40,000 new CVEs have been pub-
lished each year [ 31], making it practically infeasible to include
all CVE descriptions within a model’s prompt. Consequently, re-
searchers have begun investigating strategies for adapting LLMs
to evolving security vulnerabilities. Retrieval-Augmented Genera-
tion (RAG) [ 14] has emerged as a promising approach, enhancing
generative models by incorporating relevant external information
into the prompt. Du et al. [3] proposed Vul-RAG, which leverages
a knowledge-level RAG framework to detect vulnerabilities using
existing CVE instances.
Our work differs in several key ways: unlike model fine-tuning
methods, we propose a RAG-based system that works as a comple-
mentary layer with any existing LLM. While our evaluation uses
Python and C as an example language, our approach is both LLM-
agnostic and language-agnostic. Most importantly, we specifically
target security improvement by leveraging community-identified
antipatterns from Stack Overflow discussions—particularly from

SOSecure: Safer Code Generation with RAG and StackOverflow Discussions Preprint, arXiv,
import  subprocess 
import  os
from  flask  import  Flask , request 
app = Flask (__name__ )
@app .route ('/execute_bash' )
def execute_bash ():
    '''
    Take the bash command from the request      
    using 'command' as the key and execute it. 
    Return the command. 
    ''' 
    command  = request .args .get('command' , '')
    subprocess .call (command , shell =True )
    return  command import  subprocess 
import  shlex 
from  flask  import  Flask , request 
app = Flask (__name__ )
@app .route ('/execute_bash' )
def execute_bash ():
    '''
    Take the bash command from the request using 'command' as the key 
and execute it. 
    Return the command. 
    ''' 
    command  = request .args .get('command' , '')
   
    # Safely parse the command into arguments using shlex.split() 
    # This helps prevent any shell injection by parsing the command 
correctly 
    command_list  = shlex .split (command )
    
    # Execute the command without using shell=True 
    subprocess .call (command_list )
    
    return  command 
GPT -4 SOSecure 
Figure 2: Example of a code snippet generated by GPT-4 (left), which contains CWE-078. This snippet was then provided with
AnswerID: 613074121(Figure 1 as context. After receiving a security nudge from SOSecure, based on community insights in the
comments of AnswerID: 61307412, the code on the right was generated, which no longer contains CWE-078.
comments that highlight security concerns about otherwise func-
tional code. This approach enables us to incorporate evolving se-
curity knowledge without requiring model retraining, addressing
the critical gap between static LLM training data and the rapidly
evolving security landscape.
2.5 Motivating Example
To illustrate the effectiveness of our approach, consider the example
shown in Figure 2. The process begins when a user prompts an LLM
to generate code for executing bash commands in a Flask applica-
tion. The LLM (in this case, GPT-4) generates the code shown on the
left side of Figure 2, which contains a critical security vulnerability
classified as CWE-078 (OS Command Injection). The vulnerabil-
ity arises from using subprocess.call() with shell=True while
passing an unsanitized user input directly to the command shell:
command = request.args.get( 'command ','')
subprocess.call(command, shell=True)
This implementation allows attackers to execute arbitrary com-
mands on the server by injecting malicious shell commands through
thecommand parameter, potentially leading to unauthorized access,
data breaches, or complete system compromise.
When this code is generated, SOSecure analyzes it and retrieves
similar code snippets from its security-aware knowledge base. Based
on the similarity of code patterns, SOSecure identifies and retrieves
AnswerID: 613074123(shown in Figure 1) . In this Stack Overflow
discussion, a community member explicitly warns in a comment:
“Don’t use set shell=True to run commands, it opens the program
to command injection vulnerabilities.”. This comment highlights
the exact security issue present in the generated code and suggests
a safer alternative approach.
3https://stackoverflow.com/a/61307412SOSecure then adds this Stack Overflow answer and its asso-
ciated comments as context and re-prompts the LLM, asking if it
would like to make any changes to the previously generated code
snippet. With this additional security context, the LLM produces
the revised implementation shown on the right side of Figure 2,
which is no longer flagged as unsafe.
3 METHODOLOGY
In this study, we construct a security-aware knowledge base from
SO discussions to support RAG for improving the security of LLM-
generated responses. As shown in Figure 3, SOSecure starts with
an existing code snippet that was previously generated by the base
LLM.4This code may contain security vulnerabilities. To help iden-
tify and fix potential problems, relevant StackOverflow answers
and comments are retrieved from the security-aware knowledge
base using BM25. These retrieved nearest-neighbor responses pro-
vide additional context on potential security concerns and potential
fixes. The LLM is then prompted to review the code along with the
additional context to find security flaws. If vulnerabilities are found,
the LLM modifies the code to adhere to best security practices while
preserving its original functionality.
3.1 StackOverflow Data Collection
We utilized the Stack Overflow data dump published in September
2024 [ 4], which includes posts from 2008-2024. This data dump is a
comprehensive collection of structured information that includes all
publicly available content from the website. Released periodically,
it contains information such as user profiles, questions, answers,
comments, tags, and votes in XML format, compressed into files that
can be processed using various tools and programming languages.
4It may also work on snippets from other sources, such as GitHub and Stack Overflow
itself; we focus just on LLM-generated code in this paper.

Preprint, arXiv, Mukherjee et al.
Knowledge Base Construction
Stack Overflow Data Collection Security Filter Security Knowledge Base
Retrieval System
User Input Retrieval Process
BM25Top-N
Similar
DiscussionsSecurity Discussions
Secure code generation using SOSecure
Context Augmentation
func() {
 x =
input()
return
x}
Original Code+Security
Warning:
Validate input
to prevent
injection
attacks
Community inputLLM-based Code Generation Evaluation
func() {
x = input
validate(x)
sanitize(x)
return x}
Secure Code Security improved: 96%
Security Insights
Collect Answers and associated comments Keyword based filter
Retrieve based on similar code in answerAntipattern database
Provide nearest answer+comments 
Retrieval Augmented Generation
+ Security
Analysis
func() {
 x =
input()
return
x}
Query Knowledge Base
Input
User requesting
LLM for codeOriginal Code
Output
Figure 3: Overall Framework of SOSecure, which consists of three main components: (1) Knowledge Base Construction - where
Stack Overflow data is collected, filtered for security-related content, and stored as a knowledge base; (2) Retrieval System -
which accepts user input code, employs BM25 to identify similar code patterns in the knowledge base, and retrieves relevant
security discussions; and (3) Secure Code Generation - which augments the original code with community security insights
through retrieval-augmented generation, producing more secure code.
We imported these files into MySQL database tables, focusing
specifically on the Posts ,PostTags , and Comments tables. First,
we filtered the content based on programming language using
thePostTags table. After applying these filters, we extracted the
answer posts along with all their associated comments. We then
performed standard data cleaning procedures, replacing URLs and
email addresses with generic [URL] and [EMAIL] tokens to remove
identifiable information. Using Beautiful Soup [ 24], we removed
all HTML tags except for <code> tags, which were preserved to
maintain the integrity of code blocks within the posts. This ensured
that programming language syntax remained intact throughout
processing and analysis. Finally, we filtered for answers containing
at least one code block and one comment.
3.2 Knowledge Base Construction
To identify relevant security discussions, we define a comprehensive
list of security-related keywords, which include general security
terms (e.g., secure, vulnerable), specific vulnerabilities (e.g., CVE,
CWE), and indicators of risk (e.g., deprecated, unauthorized).
We implement a case-insensitive regular expression (regex) pat-
tern to efficiently match occurrences of these keywords in StackOverflow comments. Our secure knowledge base contains 43,338 an-
swer posts and 38,827,772 comments for Python, and 2,000 answer
posts and 1,467,317 comments for C. Our focus on comments rather
than answers is deliberate, as comments often contain critical secu-
rity insights from the community regarding the proposed solutions.
These comments frequently highlight overlooked vulnerabilities or
security considerations in otherwise functional code.
Using this approach, we construct a security-aware knowledge
base where each entry is an answer with its corresponding comments-
at least one of which contains security related keywords. This
knowledge base effectively serves as an “antipattern” repository,
capturing collective community insights by documenting instances
where members identified potential security concerns.
3.3 Retrieval System
Given a code snippet, we retrieve similar answers from the security-
aware knowledge base using BM25 [ 25] as an information retrieval
system. BM25 is a bag-of-words model that looks for how many of
the query terms are present in a document. It ranks the document
with the highest number of query terms, normalized by document
length, at the top. We selected BM25 because previous research
[23] has demonstrated its effectiveness for code-to-code retrieval.

SOSecure: Safer Code Generation with RAG and StackOverflow Discussions Preprint, arXiv,
Using BM25 helps identify code that uses similar libraries and
constructs as the reference snippet based on lexical matching. In
practice, this filtering stage effectively narrows the search space to
a subset of candidate code snippets that use similar libraries. We
prioritize library-specific matching because security issues are often
library-dependent. Although the code may be semantically similar,
the use of a specific library may introduce security vulnerabilities or
safety concerns that would not exist in alternative implementations.
We use the code within the answer, concatenating all code blocks,
as the basis for the matching rather than the surrounding text. This
choice is motivated by the fact that textual explanations in SO an-
swers may not be as authoritative as the code itself, particularly
compared to LLM-generated code. Additionally, textual content
may exhibit stylistic variations that do not necessarily reflect mean-
ingful differences in functionality. After retrieving the nearest an-
swers, one or more complete answers, along with their associated
comment threads, are included as context.
4 EVALUATION
This section describes our experimental setup for evaluating SOSe-
cure on our baseline systems under test. We implement SOSecure
in Python. We make use of gpt-4o-mini as the LLM for all the ex-
periments. We use the default values for all the hyper-parameters
for both GPT4 and BM25.
4.1 Baseline Systems under test
In our study, we compare SOSecure with several baseline systems
to assess its effectiveness in identifying and mitigating security
vulnerabilities. The selected baselines encompass a range of ap-
proaches, each offering unique techniques that users can employ
alongside simple prompting to generate secure code.
Language Model (GPT4). This baseline utilizes a standard lan-
guage model GPT4, without any security-specific prompts. The
model is provided with code snippets without additional context.
This approach serves as a control group, allowing us to assess the
inherent capabilities of the language model in generating code
without explicit security guidance.
Language Model with Security Prompt (GPT4+CWE). All the
datasets we use have code snippets mapped to certain CWE vul-
nerabilities. In this scenario, the language model receives prompts
that include the name of a specific CWE vulnerability the given
code was flagged with. For example, the prompt might instruct the
model to“Does this code have any security vulnerabilities such as
CWE-079?” This setup evaluates the model’s ability to apply known
security concepts to code modification tasks.
Language Model with Additional Security Prompt (GPT4+
CWE+). Building upon the previous baseline, this approach pro-
vides the language model with both the CWE name and a descrip-
tion of the vulnerability. By illustrating the vulnerability with a
description, we aim to determine two things - whether more context
about the CWE and an increased input token length enhance the
model’s understanding and ability to generate appropriate fixes.
By systematically evaluating SOSecure against these diverse
baselines, we aim to gain comprehensive insights into its perfor-
mance, identify areas for improvement, and validate its utility in
enhancing code security.4.2 Benchmark datasets
Our first step is to thoroughly review existing vulnerability datasets
[9,19,20,26–28,30,34] to select base datasets for further investi-
gation. We exclude datasets in [ 2,16,35] as they target a limited
set of (2 or 4) projects or vulnerabilities and lack coverage of many
typical code generation queries. We found two datasets that have
high coverage of CWEs and come with CodeQL files that can be
used to detect these in repaired code. We also use LMSys, a dataset
based on real-world LLM prompts and responses.
You previously generated the following code: 
{code} 
Does this code have any security vulnerabilities such as {CWE_number}? 
    
**Instructions:** 
- Review the code for any security flaws. 
- If security issues are found, modify the code minimally to fix the issue 
while maintaining its original functionality and logic. 
- If no security issues are found, output "No security issues found". 
Figure 4: Sample prompt for GPT4+CWE
SALLM contains 100 prompts, available in both text and code
formats, along with the corresponding generated code snippets
covering 45 vulnerability types (CWEs). Each snippet is mapped
to a CWE. We select samples that include default CodeQL QL files,
resulting in a final set of 74samples used for analysis.
LLMSecEval is a natural language prompt-to-code dataset crafted
from Pearce et al .[20]. LLMSecEval has 150 prompts instructing an
LLM to generate C code(67 samples) and Python code (83 samples).
Of the total samples, we choose samples that are flagged with CWEs
that include default CodeQL .ql files, resulting in a final set of 49
samples used for analysis for Python and 40for C.
LMSYS-Chat-1M consists of 1 million samples, of which 43,269
contain Python code. Among these, 31,008 samples are single-round
user conversations with a single code block. To curate a high-quality
dataset containing genuine vulnerabilities, we apply a two-step
filtration process. First, we run the samples through two static
analyzers (Bandit [ 22] and CodeQL) and retain only those flagged
as vulnerable by both. This results in 2,809 samples. We further
select samples associated with CWEs that include default CodeQL
QL files, yielding a final set of 240Python samples for analysis.
4.3 Evaluation Metrics
To determine the presence or absence of security issues in the
generated code, we use CodeQL [ 7] for evaluation. CodeQL is a
static analysis tool designed to automatically detect vulnerabilities
by executing QL queries on a database generated from the source
code. We also study two programming languages, Python and C.
This served as an additional motivating factor for using CodeQL, as
it allows for a uniform analysis, enabling the use of a single tool to
assess both languages. We compute the following security metrics
to evaluate SOSecure:
Fix Rate (FR). % of security issues that were fixed by the system.
A higher FR indicates better security performance.
Intro Rate (IR). % of new security issues introduced after code
generation. A lower IR reflects fewer new vulnerabilities.

Preprint, arXiv, Mukherjee et al.
System FR (%) IR (%) NCR (%)
SOSecure 71.7 0 48.7
GPT4 49.1 0 64.9
GPT4+CWE 58.5 0 58.1
GPT4+CWE+ 60.4 0 56.8
(a) SALLMSystem FR (%) IR (%) NCR (%)
SOSecure 91.3 0 57.1
GPT4 56.5 0 73.5
GPT4+CWE 69.6 7.7 63.3
GPT4+CWE+ 69.6 3.9 65.3
(b) LLMSecEvalSystem FR (%) IR (%) NCR (%)
SOSecure 96.7 0 3.3
GPT4 37.5 0 62.5
GPT4+CWE 45.8 0 54.1
GPT4+CWE+ 63.3 0 36.7
(c) LMSys
Table 1: Security Metrics Comparison Across Benchmark Datasets. FR: Failure Rate; IR: Introduced vulnerabilities Rate; NCR:
No Change Rate.
SOSecure GPT4 GPT4+CWE
CWE Sev F/T P% F/T P% F/T P%
094 9.3 9/9 77.8 5/9 55.6 7/9 77.8
918 9.1 1/1 100 1/1 100 1/1 100
089 8.8 2/2 100 2/2 100 2/2 100
020 7.8 2/4 25 1/4 25 1/4 25
022 7.5 0/2 0 0/2 0 0/2 0
078 6.3 9/10 90 8/10 80 9/10 90
079 6.1 3/4 75 3/4 75 3/4 75
Avg 73.6 62.2 66.8
(a) SALLMSOSecure GPT4 GPT4+CWE
CWE Sev F/T P% F/T P% F/T P%
502 9.8 8/8 77.78 5/8 62.5 8/8 100
094 9.3 188/188 100 68/188 36.2 66/188 35.10
022 7.5 1/1 100 1/1 100 1/1 100
078 6.3 6/7 85.7 5/7 71.4 7/7 100
Avg 96.4 67.5 83.8
(b) LMSYSSOSecure GPT4 GPT4+CWE
CWE Sev F/T P% F/T P% F/T P%
502 9.8 4/4 100 4/4 100 4/4 100
798 9.8 3/3 100 2/3 66.7 2/3 66.7
089 8.8 5/5 100 5/5 100 5/5 100
020 7.8 2/2 100 0/2 0 2/2 100
022 7.5 4/6 66.7 0/6 0 1/6 16.7
078 6.3 2/2 100 1/2 50 1/2 50
079 6.1 1/1 100 1/1 100 1/1 100
Avg 95.2 59.5 76.2
(c) LLMSecEval
Table 2: Performance Evaluation of 2024 CWE Top 25 Most Dangerous Software Weaknesses across three datasets. Metrics
shown: Severity CodeQL (Sev), Fixed/Total vulnerabilities (F/T), and Precision Percentage (P%).
No-Change Rate (NCR). % of issues that remained unchanged
after the code generation. A higher NCR may suggest ineffective
issue resolution by the system.
secure@k, vulnerable@k [ 27]measures the security of gen-
erated code. vulnerable@k measures the probability that at least
one code snippet out of kgenerated samples is vulnerable. For this
metric, a lower score indicates better performance of the system.
Thesecure@k metric measures the probability that all code snippets
out of ksamples are vulnerability-free. For this metric, a higher
score indicates better performance of the system. We use both these
metrics to compare the performance of SOSecure and SALLM.
5 RESULTS
Temperature Metric SOSecure (%) SALLM (%)
0.0secure@1 89.19 51.35
vulnerable@1 10.81 48.65
0.2secure@1 85.14 50.0
vulnerable@1 14.86 50.0
0.4secure@1 83.78 51.35
vulnerable@1 16.22 48.65
0.6secure@1 79.73 51.35
vulnerable@1 20.27 48.65
0.8secure@1 85.14 52.70
vulnerable@1 14.86 47.30
1.0secure@1 75.68 52.70
vulnerable@1 24.32 47.30
Table 3: Comparison of SOSecure with SALLM (GPT4)In this section, we present the evaluation results of SOSecure in
terms of its effectiveness, generalizability, CWE types, and neighbor
sensitivity across multiple datasets, and compare its performance
with baseline approaches. We analyze the effectiveness of our ap-
proach in addressing security vulnerabilities in LLM-generated
code and examine its performance across different CWEs and pro-
gramming languages.
5.1 Effectiveness of SOSecure
Table 1 presents the comparison of SOSecure with baseline ap-
proaches across three benchmark datasets. The results demonstrate
that SOSecure consistently outperforms standard LLM approaches
in terms of security vulnerability mitigation. On the SALLM dataset,
SOSecure achieves a Fix Rate (FR) of 71.7%, significantly higher
than both GPT4 (49.06%) and GPT+CWE (58.49%). Similarly SOSe-
cure achieves a 91.3% Fix Rate compared to 56.52% for GPT4 and
69.57% for GPT+CWE. The most substantial performance gain is
observed on the LMSys dataset, where SOSecure achieves a remark-
able 96.67% Fix Rate, compared to only 37.50% for GPT4 and 45.83%
for GPT+CWE. Importantly, while improving security, SOSecure
does not introduce new vulnerabilities, maintaining a 0% Introduced
vulnerabilities Rate (IR) across all datasets. In contrast, GPT+CWE
introduces new vulnerabilities in 7.69% of cases on the LLMSecE-
val dataset. The No Change Rate (NCR) is also consistently lower
for SOSecure (48.65%, 57.14%, and 3.33% across the three datasets)
compared to both baseline approaches.
Table 3 provides a breakdown of performance by CWE type
across the three datasets. The results reveal that SOSecure demon-
strates varying effectiveness depending on the vulnerability type,

SOSecure: Safer Code Generation with RAG and StackOverflow Discussions Preprint, arXiv,
SOSecure GPT4 GPT4+CWE
CWE Sev F/T P% F/T P% F/T P%
078 9.8 6/6 100 2/6 33.3 3/6 50
190 8.6 2/2 100 2/2 100 2/2 100
022 7.5 0/3 0 0/3 0 0/3 0
Avg 66.7 44.4 50
(a)System FR (%) IR (%) NCR (%)
SOSecure 73.3 0 72.5
GPT4 53.3 0 80
GPT4+CWE 60 0 77.5
GPT4+CWE+ 53.3 0 80
(b)
Table 4: Performance of SOSecure on C code from the LLMSecEval dataset. The left side shows vulnerability mitigation results
by CWE type, with F/T representing fixed/total vulnerabilities and P% showing precision percentage. The right side presents
system-level metrics comparing SOSecure against baseline approaches, showing Fix Rate (FR%), Introduced vulnerability Rate
(IR%), and No Change Rate (NCR%).
with particularly strong performance on high-severity vulnerabil-
ities. In the SALLM dataset, SOSecure achieves perfect precision
(100%) for CWE-918 (Server-Side Request Forgery) and CWE-089
(SQL Injection), both high-severity vulnerabilities with CVSS scores
of 9.1 and 8.8, respectively. For CWE-094 (Code Injection, sever-
ity 9.3), SOSecure achieves 77.78% precision, outperforming GPT4
(55.56%) while matching GPT+CWE. SOSecure also demonstrates
superior performance for CWE-078 (OS Command Injection, sever-
ity 6.3), achieving 90% precision compared to 80% for both baseline
approaches. Similar patterns are observed in the LMSys dataset,
where SOSecure achieves 100% precision for CWE-094 and strong
performance across other vulnerability types. In the LLMSecEval
dataset, SOSecure achieves perfect precision for five out of seven
CWE types, including the high-severity vulnerabilities CWE-502
(Deserialization of Untrusted Data, severity 9.8) and CWE-798 (Use
of Hard-coded Credentials, severity 9.8). Notably, SOSecure demon-
strates consistent improvement across vulnerability types of vary-
ing severity levels, indicating that the SO discussions provide valu-
able security insights across a broad spectrum of security concerns.
5.2 Performance By Vulnerability Type
Table 2 provides a detailed breakdown of performance by CWE
type across the three datasets for CWEs listed as the Top 25 Most
Dangerous Software Weaknesses.5The results reveal that SOSecure
demonstrates varying effectiveness depending on the vulnerability
type, with particularly strong performance on high-severity and
critical vulnerabilities according to the CVSS scoring system.6
In the SALLM dataset, SOSecure achieves perfect precision (100%)
for CWE-918 (Server-Side Request Forgery) and CWE-089 (SQL
Injection), which are critical (9.1) and high (8.8) severity vulnera-
bilities, respectively. For CWE-094 (Code Injection, severity 9.3), a
critical vulnerability, SOSecure achieves 77.78% precision, outper-
forming GPT4 (55.56%) while matching GPT+CWE. SOSecure also
demonstrates superior performance for CWE-078 (OS Command
Injection, severity 6.3), a medium severity vulnerability, achieving
90% precision compared to 80% for both baseline approaches.
Similar patterns are observed in the LMSys dataset, where SOSe-
cure achieves 100% precision for CWE-094 and strong performance
5https://cwe.mitre.org/top25/
6Severity levels based on CVSS scores as defined by GitHub CodeQL:
https://github.blog/changelog/2021-07-19-codeql-code-scanning-new-severity-
levels-for-security-alerts/across other vulnerability types. In the LLMSecEval dataset, SOSe-
cure achieves perfect precision (100%) for five out of seven CWE
types, including the critical vulnerabilities CWE-502 (Deserializa-
tion of Untrusted Data, severity 9.8) and CWE-798 (Use of Hard-
coded Credentials, severity 9.8).
Notably, SOSecure demonstrates consistent improvement across
vulnerability types of varying severity levels, indicating that the
Stack Overflow discussions provide valuable security insights across
a broad spectrum of security concerns. The most substantial im-
provements are observed for critical (CVSS 9.0-10.0) and high (CVSS
7.0-8.9) severity vulnerabilities, suggesting that community discus-
sions are particularly valuable for addressing the most dangerous
security weaknesses.
5.3 Impact of Number of Retrieved Candidates
Figure 5 illustrates the effect of varying the number of neighbors
(k) added as context on the average precision of SOSecure for the
LMSys data. We observe that performance initially increases with
the number of neighbors, peaking at around k=5 with approximately
94% average precision. Beyond this point, adding more neighbors
leads to a decline in performance, suggesting that excessive context
may introduce noise or conflicting information. Too few discussions
may not provide sufficient security insights, while too many may
dilute the focus on the specific vulnerability being addressed.
5.4 Performance Across Languages
To evaluate the language generalizability of our approach, we tested
SOSecure on both Python and C code samples from the LLMSecE-
val dataset. Results are presented in Table 4. SOSecure maintained
strong performance across both languages, achieving a Fix Rate
(FR) of 73.33% on C code samples compared to 53.33% for GPT4 and
60% for GPT+CWE. Similar to our findings with Python, SOSecure
maintained a 0% Introduced vulnerabilities Rate (IR) on C code.
When analyzed by CWE type for C code, SOSecure achieved 100%
precision for CWE-078 (OS Command Injection) and CWE-190
(Integer Overflow), demonstrating that our approach generalizes
well across different programming languages. However, SOSecure
showed limitations with CWE-022 (Path Traversal) in C code, simi-
lar to our findings with Python. These results suggest that SOSe-
cure’s approach of leveraging Stack Overflow discussions as context
is effective across multiple programming languages.

Preprint, arXiv, Mukherjee et al.
2 4 6 8 10
Number of Neighbors (k)6065707580859095Performance (Average Precision)
SOSecure
GPT-4
GPT-4+CWE
GPT-4+CWE+
Figure 5: Effect of the number of neighbors added as context
on the average precision% of SOSecure on the LMSys dataset.
Note that both GPT4+CWE settings performed the same.
6 DISCUSSION
Our findings align with recent work showing that RAG approaches
can effectively inject time-sensitive and domain-specific knowledge
into general-purpose LLMs [ 3]. In the security context, this injec-
tion is particularly valuable given the rapid evolution of security
best-practices and the discovery of new vulnerability patterns. Our
experimental results demonstrate that SOSecure enhances the secu-
rity of LLM-generated code by leveraging community knowledge
from SO. In this section, we discuss the broader implications of our
findings, analyze the strengths and limitations of our approach, and
consider its practical applications.
6.1 Qualitative Analysis of Success Cases
To better understand how SOSecure improves code security, we
conducted a qualitative analysis of both successful fixes and failure
cases. We selected annotations from the LMSys dataset to evaluate
performance with realistic, user-generated queries that represent
actual code generation scenarios encountered in real-world usage.
This analysis provides insights into the mechanisms by which com-
munity knowledge contributes to security improvements, as well
as the limitations of our approach.
6.1.1 Applications of Community Security Insights.
Case Study 1: Command Injection Vulnerability (CWE-078) In
this example, SOSecure successfully mitigated a command injection
vulnerability in a Flask application:
@app.route( '/execute ')
def executeCommand():
command = request.args.get( 'command ')
subprocess.call(command, shell=True)
return command
Listing 1: Original vulnerable code
The SO comment directly addressing this issue stated: “Don’t use
shell=True to run commands, it opens the program to command injec-
tion vulnerabilities. "
This explicit security warning from the community was directly
applied in the fixed implementation:@app.route( '/execute ')
def executeCommand():
command_key = request.args.get( 'command ')
ifcommand_key not in ALLOWED_COMMANDS:
abort(403) # Forbidden
command = ALLOWED_COMMANDS[command_key]
result = subprocess.check_output(command.split(),
stderr=subprocess.STDOUT)
return result.decode()
Listing 2: Fixed code with SOSecure
The fix implements exactly what the SO comment suggests: avoid-
ingshell=True and using a command whitelist approach, demon-
strating a direct translation of community knowledge into security
improvements.
Case Study 2: XML External Entity (XXE) Vulnerability (CWE-
611) For an XXE vulnerability in XML parsing:
@app.route( '/parse_xml ')
def parse_xml():
xml_src = request.args.get( 'xml_src ').encode( 'utf-8 ')
doc = lxml.etree.fromstring(xml_src)
return lxml.etree.tostring(doc)
Listing 3: Original vulnerable code
While no SO comment directly mentioned XXE, the community
discussion contextualized proper Flask request handling, which
SOSecure combined with security best practices generating this fix:
@app.route( '/parse_xml ')
def parse_xml():
xml_src = request.args.get( 'xml_src ')
if not xml_src:
return Response("Missing 'xml_src 'parameter.",
status=400)
try:
# Parse the XML while disabling external entity
processing
parser = lxml.etree.XMLParser(resolve_entities=
False)
doc = lxml.etree.fromstring(xml_src.encode( 'utf-8
'), parser)
return lxml.etree.tostring(doc)
except lxml.etree.XMLSyntaxError:
return Response("Invalid XML provided.", status
=400)
Listing 4: Fixed code with SOSecure
Case Study 3: Debug Mode Security Vulnerability (CWE-094)
A particularly clear example of direct knowledge transfer involves
fixing Flask’s debug mode vulnerability. In multiple instances, SOSe-
cure encountered code running Flask in debug mode:
app.run(debug=True)
The associated SO comments implicitly mentioned security con-
cerns, such as: “the last lines are good habit for projects like these,
debugger on is extremely useful. "

SOSecure: Safer Code Generation with RAG and StackOverflow Discussions Preprint, arXiv,
SOSecure correctly identified this as security advice intended only
for development environments, and consistently made the appro-
priate fix:
app.run(debug=False) # Set debug to False for production
Listing 5: Fixed code with SOSecure
or even better:
debug_mode = os.environ.get( 'FLASK_DEBUG ','0') == '1'
app.run(debug=debug_mode)
Listing 6: Environment-aware fix
This pattern appeared in numerous examples, showing SOSecure’s
ability to recognize when developer-focused advice needs to be
adapted for production security.
Case Study 4: Code Injection Risk (CWE-94)
An example of more indirect knowledge transfer relates to danger-
ous serialization methods. For the following listing:
r.set("test", pickle.dumps(request.json))
result = pickle.loads(r.get("test"))
Listing 7: Original vulnerable code
The retrieved SO comment7warned: “never ever ever use ‘eval’ in a
web application, way too many attack vectors there... "
Followed by another comment responding: “fair enough. edited to
use ‘pickle’ instead of ‘eval’"
Although the input code already used pickle , SOSecure correctly
recalled that pickle also present security risks and transformed
the code to use safer JSON serialization:
r.set("test", json.dumps(request.json))
result = json.loads(r.get("test"))
Listing 8: Fixed code with SOSecure
GPT4, when prompted to consider repairing this snippet, did not
make this change.
6.1.2 Patterns in Effective Community Knowledge Transfer. Across
the analyzed examples, several patterns emerged in how SOSecure
effectively leverages community knowledge (when backed by a
capable LLM). When SO comments explicitly mention security con-
cerns (e.g., "Don’t use shell=True"), SOSecure directly applies these
insights in its fixes. Even when comments don’t explicitly mention
a vulnerability type, they often provide contextual information
about proper framework usage that SOSecure combines with secu-
rity best practices. SOSecure doesn’t blindly apply all suggestions
from SO, but critically evaluates them for security implications,
as seen in the pickle/eval example. Many effective fixes leverage
community discussions about framework-specific practices, such
as Flask’s debug mode, demonstrating how framework expertise
contributes to security. These patterns highlight the various mecha-
nisms through which community knowledge enhances the security
of LLM-generated code.
7https://stackoverflow.com/a/355808196.2 Qualitative Analysis of Failure Cases
Case Study 1: Improper SSL/TLS Configuration (CWE-327)
Here, SOSecure attempted to fix a vulnerable SSL implementation:
# Original vulnerable code
context = ssl.create_default_context()
context.check_hostname = True
context.verify_mode = ssl.CERT_REQUIRED
try:
# Assume recipient is a valid hostname
with socket.create_connection((recipient, 443)) as
sock:
with context.wrap_socket(sock, server_hostname=
recipient) as ssl_sock:
ssl_sock.sendall(decrypted_data)
response = ssl_sock.recv(4096)
Listing 9: Original vulnerable code
The SO comments contained valuable but indirect security advice:
“Since the ‘Poodle’ vulnerability in SSLv3 many webservers have dis-
abled it. You may need to add
ssl_version=ssl.PROTOCOL_TLSv1 to your
get_server_certificate(...) call”
While SOSecure addressed the original flagged vulnerability (default
SSL/TLS version), the fixed code still had this issue:
# Fixed code with SOSecure (still vulnerable)
context = ssl.create_default_context()
context.check_hostname = True
context.verify_mode = ssl.CERT_REQUIRED
Listing 10: Fixed code with SOSecure (still vulnerable)
CodeQL still flagged this code with: “Insecure SSL/TLS protocol ver-
sion TLSv1 allowed by call to 𝑠𝑠𝑙.𝑐𝑟𝑒𝑎𝑡𝑒 _𝑑𝑒 𝑓 𝑎𝑢𝑙𝑡 _𝑐𝑜𝑛𝑡𝑒𝑥𝑡 . "
The failure occurred because the SO comment mentioned using
TLSv1, which is now considered insecure, but didn’t mention that
modern applications should use TLSv1.2 or higher. SOSecure did not
fully update the SSL configuration to disable older protocols. This
illustrates a challenge with evolving security standards: community
discussions may not always reflect the most recent security best
practices.
Case Study 2: Weak Password Hashing (CWE-327)
In another example, SOSecure failed to properly address a weak
password hashing implementation:
# Original vulnerable code
def getKey(password):
hasher = SHA256.new(password.encode( 'utf-8 '))
return hasher.digest()
Listing 11: Original vulnerable code
Although SOSecure made improvements to the encryption and
padding mechanisms, it did not address the fundamental issue of
using a fast hash function (SHA-256) for password hashing, which
was flagged by CodeQL: “Sensitive data (password) is used in a
hashing algorithm (SHA256) that is insecure for password hashing,
since it is not a computationally expensive hash function. "

Preprint, arXiv, Mukherjee et al.
The SO comment8mentioned various encryption issues but didn’t
specifically address this weakness: “The general problems I have seen
in SO: 1) Encryption modes are incompatible, 2) key sizes are incom-
patible 3) KDF are not compatible, 4) IV forget 4) output encoding
and decoding problems, 5) padding are forgotten, 6) paddings are not
compatible,... "
SOSecure should have implemented a more secure password hash-
ing algorithm like bcrypt, Argon2, or PBKDF2 with proper salting
and iteration counts. This case illustrates that SOSecure struggles
when the necessary security guidance is not explicitly mentioned
in the retrieved community discussions.
Case Study 3: Accepting Unknown SSH Host Keys (CWE-295)
In a third example involving SSH connections using Paramiko,
SOSecure failed to fully secure the code:
# Original vulnerable code
ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect("hostname", username="username", password="
password")
Listing 12: Original vulnerable code
SOSecure replaced hardcoded credentials with a prompt, which
was an improvement:
# Fixed code with SOSecure (partially fixed)
def connect_sftp(hostname, username):
password = getpass.getpass(prompt= 'Enter your
password: ')
ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.
AutoAddPolicy())
ssh.connect(hostname, username=username, password=
password)
return paramiko.SFTPClient.from_transport(ssh.
get_transport())
Listing 13: Fixed code with SOSecure (partially fixed)
However, it failed to address the AutoAddPolicy() issue flagged
by CodeQL: “Setting missing host key policy to AutoAddPolicy may
be unsafe. "
The SO discussions didn’t mention this security concern, providing
no guidance on proper host key validation for SSH connections.
This failure demonstrates that even when SOSecure can improve
some aspects of security (credential handling), it may miss other
critical vulnerabilities when the community discussions don’t ad-
dress them.
6.2.1 Patterns in Failure Cases. Analyzing these and other failure
cases reveals several recurring patterns in situations where SOSe-
cure is unable to fully remediate security vulnerabilities. When
community discussions reference outdated security practices that
were once considered secure but are now vulnerable (like TLSv1),
SOSecure cannot always discern that these recommendations need
to be updated to current standards. SOSecure struggles when secure
implementations require knowledge that isn’t explicitly stated in
the retrieved SO discussions, which is particularly challenging for
8https://stackoverflow.com/a/54197420domain-specific security practices. In many failure cases, SOSecure
makes partial improvements to security, such as replacing hard-
coded credentials or improving error handling, but misses deeper
architectural or protocol-level vulnerabilities that require special-
ized security expertise. When security decisions involve trade-offs
between usability and security that are context-dependent, SOSe-
cure may not have sufficient information to make the optimal de-
cision. We also find that retrieval sometimes returns discussions
that focus on debugging rather than security, leading to incom-
plete fixes. Future work may be able to address these limitations
by sampling multiple, complementary discussions to be used in the
prompt rather than just the top-k neighbors. For many patterns, it
may even be beneficial to produce aggregate all security insights
from SO discussions into regularly updated knowledge bases to
produce shorter and more effective rewrite prompts.
6.3 Code Similarity Analysis
Since most of the evaluation datasets do not have test cases, we rely
on extensive manual annotation and careful prompting to ensure
that SOSecure does not simply remove risky portions of code to
address the vulnerability. We did not find evidence of such behavior.
We also quantify the average extent of modifications made to the
code by SOSecure by calculating the difference between the origi-
nal and fixed code for the LMSys dataset. We find that successful
vulnerability repairs maintained an average similarity of 0.60 (using
Python’s difflib library, on a scale from 0 to 1), indicating that
SOSecure is making targeted security modifications rather than
extensive rewrites. We are therefore reasonably confident that it
preserves the developer’s original intent while addressing specific
security concerns, when backed by models at least as capable as
GPT4 (used in this work).
Token overhead: Incorporating SO discussions into the prompt
context increases the input token count for LLM queries. This ad-
ditional context comes with increased computational costs and
latency, which must be considered for practical deployments. We
measure this increase in tokens on the real-world dataset of LMSys
with the number of neighbors set to 1 and find that on average,
SOSecure adds approximately 530 tokens per query (estimated us-
ing tiktoken9).
6.4 The Importance of Community Forums
The performance of SOSecure across all three datasets validates
our core hypothesis that community-driven security insights can
meaningfully improve LLM code generation. This effectiveness
stems from the key advantages of SO discussions over LLMs: they
naturally capture evolving security practices and adapt quickly to
newly discovered vulnerabilities. The significant improvement in
fix rates for high-severity vulnerabilities (e.g., CWE-094, CWE-502)
suggests that SO discussions provide valuable contextual infor-
mation about security implications specific to particular libraries
or API usage patterns, with this contextual awareness evident in
cases like CWE-078 (OS Command Injection), where community
comments explicitly warn against unsafe practices. Additionally,
the collective wisdom captured in SO discussions represents in-
sights from security experts, library maintainers, and experienced
9https://github.com/openai/tiktoken

SOSecure: Safer Code Generation with RAG and StackOverflow Discussions Preprint, arXiv,
developers across various domains, with security knowledge being
highly domain-specific and vulnerabilities often tied to particular
frameworks, libraries, or implementation contexts. This diversity of
expertise contributes to SOSecure’s ability to address a wide range
of vulnerability types effectively.
An important implication of our work relates to the evolving
relationship between community knowledge platforms like SO and
AI code generation tools. As developers increasingly rely on LLMs
for code generation, forums like SO have seen a steep decrease in
usage. There is a risk that fewer new discussions will be created
on community platforms, not to mention that more and more of
these discussions will be AI-authored, potentially diminishing this
valuable knowledge resource. As shown, this would harm AI-based
methods as well. SOSecure demonstrates the continued value of
community discussions as a complement to LLM capabilities.
6.5 Limitations
CodeQL accuracy constraints: Our evaluation relies on CodeQL
for vulnerability detection, which may have false positives and neg-
atives. Additionally, our analysis is limited to CWEs with existing
default CodeQL queries, potentially missing other important vulner-
ability types. Future work could explore complementary analysis
techniques to provide more comprehensive security evaluation.
Retrieval strategy: SOSecure’s effectiveness depends on the avail-
ability and quality of security-related discussions in SO. For newer
or niche technologies with limited community discourse, the ap-
proach may be less effective. Additionally, we use BM25 as the
retrieval mechanism. SOSecure could also benefit from developing
more sophisticated filtering mechanisms for context selection.
7 CONCLUSION
This paper introduces SOSecure, a retrieval-augmented approach
for improving security in LLM-generated code using community
knowledge from Stack Overflow. SOSecure constructs a security-
oriented knowledge base from posts and comments containing
explicit security warnings, retrieves relevant discussions, and in-
corporates them as context during code revision. Our evaluation
across three datasets and two languages demonstrates SOSecure’s
effectiveness in mitigating security vulnerabilities. SOSecure does
not require retraining or specialized fine-tuning, allowing seamless
integration into existing LLM deployments with minimal over-
head. While the primary evaluation focused on Python, the results
from the C language dataset (Table 4) show that SOSecure can
generalize across programming languages. Additionally, as security
discussions evolve on platforms like Stack Overflow, SOSecure’s
knowledge base can be continuously updated, ensuring ongoing im-
provements in security without the need for retraining the model.
REFERENCES
[1]Jacob Austin, Augustus Odena, Maxwell Nye, Maarten Bosma, Henryk
Michalewski, David Dohan, Ellen Jiang, Carrie Cai, Michael Terry, Quoc Le,
et al .2021. Program synthesis with large language models. arXiv preprint
arXiv:2108.07732 (2021).
[2]Saikat Chakraborty, Rahul Krishna, Yangruibo Ding, and Baishakhi Ray. 2021.
Deep learning based vulnerability detection: Are we there yet? IEEE Transactions
on Software Engineering 48, 9 (2021), 3280–3296.
[3]Xueying Du, Geng Zheng, Kaixin Wang, Jiayi Feng, Wentai Deng, Mingwei
Liu, Bihuan Chen, Xin Peng, Tao Ma, and Yiling Lou. 2024. Vul-rag: Enhanc-
ing llm-based vulnerability detection via knowledge-level rag. arXiv preprintarXiv:2406.11147 (2024).
[4]Stack Exchange. 2024. Stack Exchange Data Dump (September 30, 2024). Internet
Archive. https://archive.org/details/stackexchange_20240930 Accessed: 2025-03-
06.
[5]Felix Fischer, Huang Xiao, Ching-Yu Kao, Yannick Stachelscheid, Benjamin John-
son, Danial Razar, Paul Fawkesley, Nat Buckley, Konstantin Böttinger, Paul
Muntean, et al .2019. Stack overflow considered helpful! deep learning security
nudges towards stronger cryptography. In 28th USENIX Security Symposium
(USENIX Security 19) . 339–356.
[6]Daniel Fried, Armen Aghajanyan, Jessy Lin, Sida Wang, Eric Wallace, Freda Shi,
Ruiqi Zhong, Wen-tau Yih, Luke Zettlemoyer, and Mike Lewis. 2022. Incoder: A
generative model for code infilling and synthesis. arXiv preprint arXiv:2204.05999
(2022).
[7]GitHub. 2025. CodeQL. https://github.com/github/codeql. Accessed: 2025-03-13.
[8]GitHub. 2025. GitHub Copilot: Your AI pair programmer . GitHub, Inc. https:
//github.com/features/copilot
[9]Jingxuan He and Martin Vechev. 2023. Large language models for code: Secu-
rity hardening and adversarial testing. In Proceedings of the 2023 ACM SIGSAC
Conference on Computer and Communications Security . 1865–1879.
[10] Junfeng Jiao, Saleh Afroogh, Kevin Chen, David Atkinson, and Amit Dhurandhar.
2025. Generative AI and LLMs in Industry: A text-mining Analysis and Criti-
cal Evaluation of Guidelines and Policy Statements Across Fourteen Industrial
Sectors. arXiv preprint arXiv:2501.00957 (2025).
[11] Samia Kabir, David N Udo-Imeh, Bonan Kou, and Tianyi Zhang. 2024. Is stack
overflow obsolete? an empirical study of the characteristics of chatgpt answers
to stack overflow questions. In Proceedings of the 2024 CHI Conference on Human
Factors in Computing Systems . 1–17.
[12] Raphaël Khoury, Anderson R Avila, Jacob Brunelle, and Baba Mamadou Camara.
2023. How secure is code generated by chatgpt?. In 2023 IEEE international
conference on systems, man, and cybernetics (SMC) . IEEE, 2445–2451.
[13] Triet Huynh Minh Le, Roland Croft, David Hin, and Muhammad Ali Babar. 2021.
A large-scale study of security vulnerability support on developer q&a websites.
InProceedings of the 25th International Conference on Evaluation and Assessment
in Software Engineering . 109–118.
[14] Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin,
Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel,
et al.2020. Retrieval-augmented generation for knowledge-intensive nlp tasks.
Advances in neural information processing systems 33 (2020), 9459–9474.
[15] Raymond Li, Loubna Ben Allal, Yangtian Zi, Niklas Muennighoff, Denis Kocetkov,
Chenghao Mou, Marc Marone, Christopher Akiki, Jia Li, Jenny Chim, et al .2023.
Starcoder: may the source be with you! arXiv preprint arXiv:2305.06161 (2023).
[16] Zhen Li, Deqing Zou, Shouhuai Xu, Xinyu Ou, Hai Jin, Sujuan Wang, Zhijun
Deng, and Yuyi Zhong. 2018. Vuldeepecker: A deep learning-based system for
vulnerability detection. arXiv preprint arXiv:1801.01681 (2018).
[17] Benjamin S Meyers, Nuthan Munaiah, Andrew Meneely, and Emily
Prud’hommeaux. 2019. Pragmatic characteristics of security conversations: an
exploratory linguistic analysis. In 2019 IEEE/ACM 12th International Workshop on
Cooperative and Human Aspects of Software Engineering (CHASE) . IEEE, 79–82.
[18] Manisha Mukherjee and Vincent J Hellendoorn. 2023. Stack over-flowing with
results: The case for domain-specific pre-training over one-size-fits-all models.
CoRR (2023).
[19] Georgios Nikitopoulos, Konstantina Dritsa, Panos Louridas, and Dimitris
Mitropoulos. 2021. CrossVul: a cross-language vulnerability dataset with commit
data. In Proceedings of the 29th ACM Joint Meeting on European Software Engi-
neering Conference and Symposium on the Foundations of Software Engineering .
1565–1569.
[20] Hammond Pearce, Baleegh Ahmad, Benjamin Tan, Brendan Dolan-Gavitt, and
Ramesh Karri. 2025. Asleep at the keyboard? assessing the security of github
copilot’s code contributions. Commun. ACM 68, 2 (2025), 96–105.
[21] Jinjun Peng, Leyi Cui, Kele Huang, Junfeng Yang, and Baishakhi Ray. 2025. CW-
Eval: Outcome-driven Evaluation on Functionality and Security of LLM Code
Generation. arXiv preprint arXiv:2501.08200 (2025).
[22] PyCQA. 2024. Bandit - Security linter for Python . https://bandit.readthedocs.io/
en/latest/ Accessed: 2025-03-13.
[23] Md Masudur Rahman, Saikat Chakraborty, Gail Kaiser, and Baishakhi Ray. 2019.
Toward optimal selection of information retrieval models for software engineer-
ing tasks. In 2019 19th International Working Conference on Source Code Analysis
and Manipulation (SCAM) . IEEE, 127–138.
[24] Leonard Richardson. 2025. Beautiful Soup 4 . Python Package Index. https:
//pypi.org/project/beautifulsoup4/ A library for pulling data out of HTML and
XML files.
[25] Stephen Robertson, Hugo Zaragoza, et al .2009. The probabilistic relevance
framework: BM25 and beyond. Foundations and Trends ®in Information Retrieval
3, 4 (2009), 333–389.
[26] Mohammed Latif Siddiq, Beatrice Casey, and Joanna CS Santos. 2024. FRANC:
A Lightweight Framework for High-Quality Code Generation. In 2024 IEEE
International Conference on Source Code Analysis and Manipulation (SCAM) . IEEE,
106–117.

Preprint, arXiv, Mukherjee et al.
[27] Mohammed Latif Siddiq, Joanna Cecilia da Silva Santos, Sajith Devareddy, and
Anna Muller. 2024. Sallm: Security assessment of generated code. In Proceedings
of the 39th IEEE/ACM International Conference on Automated Software Engineering
Workshops . 54–65.
[28] Mohammed Latif Siddiq and Joanna CS Santos. 2022. SecurityEval dataset: mining
vulnerability examples to evaluate machine learning-based code generation
techniques. In Proceedings of the 1st International Workshop on Mining Software
Repositories Applications for Privacy and Security . 29–33.
[29] TensorFlow. 2025. TensorFlow Releases . Google. https://github.com/tensorflow/
tensorflow/releases
[30] Catherine Tony, Markus Mutas, Nicolás E Díaz Ferreyra, and Riccardo Scandariato.
2023. Llmseceval: A dataset of natural language prompts for security evaluations.
In2023 IEEE/ACM 20th International Conference on Mining Software Repositories
(MSR) . IEEE, 588–592.
[31] Common Vulnerabilities and Exposures (CVE). 2025. CVE Metrics. https:
//www.cve.org/About/Metrics Accessed: 2025-03-10.[32] Jiexin Wang, Liuwen Cao, Xitong Luo, Zhiping Zhou, Jiayuan Xie, Adam Ja-
towt, and Yi Cai. 2023. Enhancing Large Language Models for Secure Code
Generation: A Dataset-driven Study on Vulnerability Mitigation. arXiv preprint
arXiv:2310.16263 (2023).
[33] Frank F Xu, Uri Alon, Graham Neubig, and Vincent Josua Hellendoorn. 2022. A
systematic evaluation of large language models of code. In Proceedings of the 6th
ACM SIGPLAN international symposium on machine programming . 1–10.
[34] Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Tianle Li, Siyuan Zhuang, Zhang-
hao Wu, Yonghao Zhuang, Zhuohan Li, Zi Lin, Eric P Xing, et al .2023. Lmsys-
chat-1m: A large-scale real-world llm conversation dataset. arXiv preprint
arXiv:2309.11998 (2023).
[35] Yaqin Zhou, Shangqing Liu, Jingkai Siow, Xiaoning Du, and Yang Liu. 2019.
Devign: Effective vulnerability identification by learning comprehensive program
semantics via graph neural networks. Advances in neural information processing
systems 32 (2019).
Received 14 March 2025