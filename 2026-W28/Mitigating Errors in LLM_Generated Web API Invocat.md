# Mitigating Errors in LLM-Generated Web API Invocations via Retrieval-Augmented Generation and Constrained Decoding

**Authors**: Daniel Maninger, Leon Chemnitz, Jannis Brugger, Tushar Lamba, Amir Molzam Sharifloo, Mira Mezini

**Published**: 2026-07-07 07:38:28

**PDF URL**: [https://arxiv.org/pdf/2607.05936v1](https://arxiv.org/pdf/2607.05936v1)

## Abstract
Integration of web APIs is a cornerstone of modern software systems, yet writing correct web API invocation code remains challenging due to complex and evolving API specifications. Although LLMs are increasingly used for code generation, previous work has empirically shown that their ability to generate correct web API integrations is limited. At the same time, mitigation techniques and their effectiveness for this setting remain insufficiently understood.
  In this paper, we propose and systematically evaluate retrieval-augmented generation (RAG) and constrained decoding (CD) as two complementary approaches to improving LLM-generated web API invocation code. For RAG, we design a retriever that processes OpenAPI specifications and retrieves compact endpoint representations to inject into model prompts. For CD, we introduce an automatic translation from OpenAPI specifications to regex-based constraints enforced during generation.
  We evaluate both approaches on WAPIIBench's existing synthetic dataset and on a new real-world dataset derived from GitHub repositories. Our results show that RAG reduces hallucinations and improves correctness when generating full API invocations but reduces it when the endpoint is already provided as it encourages the generation of unnecessary parameters. In contrast, CD reliably prevents illegal URLs, HTTP methods, and arguments and substantially improves overall correctness for both starter codes.

## Full Text


<!-- PDF content starts -->

Mitigating Errors in LLM-Generated Web API Invocations via
Retrieval-Augmented Generation and Constrained Decoding
DANIEL MANINGER,Technische Universität Darmstadt, Germany and Hessian Center for Artificial Intelligence
(hessian.AI), Germany
LEON CHEMNITZ,Pariton AI, Germany and Technische Universität Darmstadt, Germany
JANNIS BRUGGER,Technische Universität Darmstadt, Germany and Hessian Center for Artificial Intelligence
(hessian.AI), Germany
TUSHAR LAMBA,Technische Universität Darmstadt, Germany
AMIR MOLZAM SHARIFLOO,Technische Universität Darmstadt, Germany
MIRA MEZINI,Technische Universität Darmstadt, Germany, Hessian Center for Artificial Intelligence (hessian.AI),
Germany, and National Research Center for Applied Cybersecurity ATHENE, Germany
Integration of web APIs is a cornerstone of modern software systems, yet writing correct web API invocation code remains challenging
due to complex and evolving API specifications. Although LLMs are increasingly used for code generation, previous work has
empirically shown that their ability to generate correct web API integrations is limited. At the same time, mitigation techniques and
their effectiveness for this setting remain insufficiently understood.
In this paper, we propose and systematically evaluate retrieval -augmented generation (RAG) and constrained decoding (CD) as two
complementary approaches to improving LLM -generated web API invocation code. For RAG, we design a retriever that processes
OpenAPI specifications and retrieves compact endpoint representations to inject into model prompts. For CD, we introduce an
automatic translation from OpenAPI specifications to regex-based constraints enforced during generation.
We evaluate both approaches on WAPIIBench’s existing synthetic dataset and on a new real -world dataset derived from GitHub
repositories. Our results show that RAG reduces hallucinations and improves correctness when generating full API invocations but
reduces it when the endpoint is already provided as it encourages the generation of unnecessary parameters. In contrast, CD reliably
prevents illegal URLs, HTTP methods, and arguments and substantially improves overall correctness for both starter codes.
CCS Concepts:•Software and its engineering →Automatic programming;•Computing methodologies →Neural networks;
•Information systems→RESTful web services.
Additional Key Words and Phrases: artificial intelligence, software engineering, large language models, code generation, web APIs,
benchmarks, retrieval-augmented generation, constrained decoding
Authors’ Contact Information: Daniel Maninger, Technische Universität Darmstadt, Darmstadt, Germany and Hessian Center for Artificial Intelligence
(hessian.AI), Darmstadt, Germany, daniel .maninger@tu-darmstadt .de; Leon Chemnitz, Pariton AI, Berlin, Germany and Technische Universität Darmstadt,
Darmstadt, Germany, leon .chemnitz@pariton .ai; Jannis Brugger, Technische Universität Darmstadt, Darmstadt, Germany and Hessian Center for Artificial
Intelligence (hessian.AI), Darmstadt, Germany, jannis .brugger@tu-darmstadt .de; Tushar Lamba, Technische Universität Darmstadt, Darmstadt, Germany,
tushar .lamba@stud .tu-darmstadt .de; Amir Molzam Sharifloo, Technische Universität Darmstadt, Darmstadt, Germany, amir .molzam@tu-darmstadt .de;
Mira Mezini, Technische Universität Darmstadt, Darmstadt, Germany and Hessian Center for Artificial Intelligence (hessian.AI), Darmstadt, Germany
and National Research Center for Applied Cybersecurity ATHENE, Darmstadt, Germany, mezini@cs.tu-darmstadt.de.
©2026 Copyright held by the owner/author(s). Publication rights licensed to ACM.
Manuscript submitted to ACM
Manuscript submitted to ACM 1
arXiv:2607.05936v1  [cs.SE]  7 Jul 2026

2 Maninger et al.
ACM Reference Format:
Daniel Maninger, Leon Chemnitz, Jannis Brugger, Tushar Lamba, Amir Molzam Sharifloo, and Mira Mezini. 2026. Mitigating Errors
in LLM-Generated Web API Invocations via Retrieval-Augmented Generation and Constrained Decoding.ACM Trans. Softw. Eng.
Methodol.37, 4, Article 111 (August 2026), 54 pages. https://doi.org/XXXXXXX.XXXXXXX
1 Introduction
Web APIs provide services and functionality through a standardized, HTTP-based interface. Developers leverage them
to rapidly create applications that seamlessly integrate with a wide range of online services (powered byAPI integration
code). The market surrounding API integrations is experiencing explosive growth [ 26,34]. However, writing correct
API integration code is a tedious and challenging task. Developers need to understand and correctly utilize a significant
amount of information to be able to write anAPI invocation1in compliance with its specification [ 36,37,52]. Moreover,
the task of ensuring correct API usage is further complicated by the fact that APIs evolve and change over time.
Large language models (LLMs) have great potential to boost productivity in software development, e.g., by automati-
cally generating program code from natural language descriptions [ 4]. However, LLMs may also hallucinate and make
mistakes, raising concerns about correctness [ 8,47], security [ 31,32], and general quality [ 13] of LLM-generated code.
While the capabilities of LLMs have been investigated for many different software engineering tasks, including
non-web API invocations [ 59], their aptitude for web API integrations remained unexplored, until recently, when
we introduced WAPIIBench [ 25], a new benchmark and pipeline for evaluating LLMs on web API invocation code
generation, and used it to study state-of-the-art LLMs. Our study revealed significant limitations: LLMs generate
incorrect results most of the time with manyfold error types, ranging from selecting the wrong endpoint to leaving out
required arguments or hallucinating entirely illegal arguments.
Research Gap.Yet how the issues revealed by our previous study [ 25] can be mitigated remains an open question.
While several techniques, including retrieval-augmented generation (RAG) [ 20], constrained decoding (CD) [ 5,16],
and fine-tuning, have been shown to effectively mitigate LLM generation errors in other domains, their efficacy has
not been investigated for web API invocations. Such invocations present distinct challenges: the interface is defined
externally in an OpenAPI specification rather than in the source code, the invocation must be constructed explicitly by
assembling a URL, HTTP method, and parameters across multiple different locations (path, query, header, body), and
the specification itself evolves independently of the client code (see also Section 2.1). None of these challenges apply to
the simpler function-call or tool-use settings for which RAG and CD have been studied.
To close this gap, in this paper, we present the results of a systematic study of the effectiveness of RAG, CD, and their
combination for improving the performance of LLMs on web API invocation tasks. More specifically, we formulate and
answer the following research questions:
1We differentiate between “APIintegration(code)”, which refers to the general concept of connecting an arbitrary number of different web APIs with each
other through code, and “APIinvocation(code)”, which refers to the code snippet that calls a single web API. An API integration consists of one or more
API invocations plus the code surrounding them.
Manuscript submitted to ACM

Mitigating Errors in LLM-Generated Web API Invocations 3
RQ1:To what extent does RAG improve the correctness of LLM-generated web API invocation code?
RQ2:To what extent does CD improve the correctness of LLM-generated web API invocation code?
RQ3:To what extent does combining RAG and CD improve the correctness of LLM-generated web API invocation code
compared to each technique in isolation?
We leave fine-tuning out of consideration for two main reasons. (a) Fine-tuning requires a corpus of training data
and substantial computational resources; it is unclear how we could obtain sufficient amounts of high-quality training
data for the specific kind of web API invocation tasks we are targeting. (b) Improvements achieved by fine-tuning can
quickly become obsolete when the underlying data changes; this is especially relevant in the context of our work, as
API specifications are constantly evolving.
Methodology.To answer Research Questions 1–3, we (a) developed RAG and CD approaches tailored to the specific
requirements of web API invocation tasks, along with a combined approach designed to explore their potential synergies,
and (b) evaluate their effectiveness against the respective baselines across two complementary settings.
In the first setting, we use the synthetic dataset from WAPIIBench [ 25]. In the second, we evaluate on a new real-
world dataset that we constructed in the context of the work presented here by mining public GitHub repositories for
JavaScript files containing Axios-based web API invocations and manually converting them into evaluation tasks. To
accommodate file-level code context, we extend WAPIIBench accordingly. The resulting dataset covers tasks for 11
real-world APIs, with OpenAPI specifications ranging from a few hundred to over 76,000 lines.
These two evaluation settings offer complementary strengths and limitations. The synthetic dataset provides
systematic, controlled coverage of all endpoints for a curated set of APIs, but may not fully reflect the diversity of
real-world integration scenarios. The real-world dataset, by contrast, captures common usage patterns of widely adopted
APIs, yet is necessarily incomplete; it does not exhaustively cover all endpoints of any given API.
Results.Our experiments show that RAG reduces hallucinations and can improve the correctness of generated API
invocations. On the synthetic dataset, depending on the provided starter code, it increases average correctness by
113% when generating complete API invocations but decreases it by 3% when the endpoint is already provided. On
the real-world dataset, the impact of RAG on correctness is almost negligible, but the reduction of hallucinations is
consistent across both datasets. The reason for the ambivalent results on correctness is that RAG encourages models to
use more parameters than required. The retriever itself is able to identify the sought-after endpoint among all available
endpoints with very high precision, implying that remaining errors are largely due to the models’ ability to make proper
use of the retrieved information.
The study of constrained models confirms that decoding with our constraints enforces compliance with an API’s
specification when generating an invocation, reliably reducing the amount of illegal URLs, HTTP methods, and
arguments to zero. Moreover, across models, CD significantly improves the overall correctness of the generated code, at
relative gains of +209% and +143% on average on the synthetic dataset, depending on the provided starter code. Similar
positive trends can be observed on the real-world dataset.
The combination of RAG and CD achieves high average gains in correctness—on the synthetic dataset +332% and
+111%, depending on the starter code—and reduces hallucinations to zero, similar to CD. However, compared to CD, the
gains are less consistent, as several models show a drop in correctness below the baseline.
Manuscript submitted to ACM

4 Maninger et al.
Listing 1. General structure of an API invocation using Axios and JavaScript
constaxios = require ('axios');
axios .< method >('https :// server . com / path /to/ endpoint / arg1', {
arg2 :'request body parameters'
}, {
headers : {
arg3 :'header parameters'
},
params : {
arg4 :'query parameters'
}
});
Contributions.In summary, we make the following contributions:
A specialized retrieval methodfor OpenAPI specifications that provides models with relevant in-context informa-
tion about the used API.→Section 3
An automatic constraint generator that translates OpenAPI specifications to regex-based constraints, which ensure
that generated code is specification-complaint.→Section 4
A real-world dataset to validate and complement the results on WAPIIBench’s original synthetic dataset. →Section 5
A systematic empirical evaluation of the ability of LLMs to generate correct and hallucination-free web API invo-
cation code with retrieval-augmented generation and constrained decoding.→Section 6
Artifacts are available on GitHub2, including the RAG and CD implementations presented in this paper as well
as the new real-world dataset. In addition, we provide all model-generated codes and evaluation results on Zenodo3.
Additional details and comprehensive result tables are provided in the appendix at the end of this paper.
2 Background
In the following, we will introduce some concepts fundamental to our work.
2.1 Web APIs and OpenAPI
Unlike local function calls, web API invocations require constructing an HTTP request rather than invoking a named
function. APIs exposeendpoints, identified by the combination of an HTTP method and a URL, and requests may include
parameters in multiple locations, such as the path, query string, headers, and request body. As illustrated in Listing 1,
this leads to more complex invocation structures with multiple parameter groups, nested data types, and externally
defined interfaces.
Most modern web APIs are described using theOpenAPIspecification4, the de facto standard for machine-readable API
documentation [ 42]. OpenAPI specifications, written in JSON or YAML, describe available endpoints, their parameters,
request and response schemas, authentication requirements, and other metadata. In contrast to local function signatures,
this information is maintained separately from the client code and must be interpreted to construct valid requests.
This paper focuses on JavaScript, one of the most widely used programming languages [ 44], and the Axios library5,
a popular HTTP client for JavaScript [ 14]. The evaluation framework, WAPIIBench, is built around this setup and
represents web API invocations as Axios requests to endpoints defined through OpenAPI specifications.
2https://github.com/stg-tud/WAPIIBench
3https://doi.org/10.5281/zenodo.13758414
4https://www.openapis.org/
5https://axios-http.com/
Manuscript submitted to ACM

Mitigating Errors in LLM-Generated Web API Invocations 5
0
Dataset
creationAPI
specificationa,s1
Code
generation2
Code
execution3
Correctness
analysisEvaluation
resultsa,t i c‘
s,c
Fig. 1. Benchmark design for evaluating the capabilities of LLMs in generating web API invocation code. (0) Based on an API 𝑎and
its specifications 𝑠, an API invocation tasks 𝑡and corresponding correct request configurations 𝑐are created. (1) For each 𝑡, the LLM
under evaluation generates an API invocation 𝑖. (2)𝑖is executed in a controlled environment, yielding a request configuration 𝑐′.
(3)𝑐′is compared to𝑐and validated against𝑠to obtain various metrics.
2.2 WAPIIBench
For our studies, we rely on WAPIIBench, a benchmark for web API integration code generation that we previously
introduced [ 25]. It includes a dataset of web API invocation tasks and an execution-based evaluation pipeline. A
schematic overview is provided in Figure 1.
WAPIIBench’s dataset comprises 395 tasks across four real-world APIs: Asana, Google Calendar, Google Sheets, and
Slack (one task for each endpoint). Each task includes a natural languagetask descriptionand arequest configuration,
which is used as the ground-truth solution of the task and contains the expected URL, HTTP method, and arguments.
An example task description can be found in Figure 4a. The dataset was synthetically generated based on the OpenAPI
specifications of the respective APIs and then subjected to rigorous automated and manual vetting and correction
process. This way, we ensured that each task is well-defined and unambiguously solvable.
The evaluation pipeline has three stages: (1)Code generation, (2)Code execution, and (3)Correctness analysis.
(1) In the code generation stage, we insert the API name and the task description (as a line comment) into a prompt
that instructs the model under evaluation to solve the given task by completing the starter code that is part of the
prompt (cf. Listing 3 in the appendix). The prompt also includes some clarifications to avoid instances of misalignment
and recurring error patterns observed in preliminary experiments. Experiments can be conducted with two different
starter code variants, calledsetups. Thefull completionsetup includes the beginning of the API invocation ( axios. ).
It is used to assess the ability of the model to identify the correct endpoint (i.e., the combination of URL and HTTP
method) and to solve the task as a whole. Theargument completionsetup already includes the correct endpoint
(axios.<method>(’<url>’,) and aims to evaluate whether the model can use this endpoint in compliance with the
API’s specification by selecting only permitted parameters.
(2) In the code execution stage, the model-generated API invocation is cut out of the surrounding code and inserted
into a controlled environment, which enables the safe execution the generated API invocation without sending requests
to external servers. Additionally, this environment serializes and saves the request’s configuration, so it can be used in
the following correctness analysis.
(3) In correctness analysis stage, the captured request configuration is compared to the ground-truth configuration
and validated against the corresponding API specification. The analysis happens in element-wise and aggregated forms
to obtain various fine-grained metrics explained in Table 4 in the appendix.
2.3 Retrieval-Augmented Generation
Retrieval augmented generation (RAG) [ 20] is a commonly used technique to improve model performance on knowledge-
intensive tasks. The general idea is to search a knowledge base or collection of documents for relevant pieces of
Manuscript submitted to ACM

6 Maninger et al.
Vector
storeAPI
specification// Create new event … and add it to calendar …
LLM Retriever Augmentedprompt Taskviacodecomment </>Code
Fig. 2. Retrieval-augmented generation framework for generating web API invocations. Instead of passing the prompt directly to
the language model, it is first augmented with information about potentially relevant API endpoints. Chunks of information are
extracted from the API specification and persisted in a vector store. The retriever performs a similarity search to find chunks that
match the given task description.
information, so-calledchunks, and augment the prompt with them before generating a response. This way, the model
does not need to rely solely on memorized knowledge and can base its response on the retrieved information, which
enhances reliability. However, the effectiveness of RAG depends on the retriever’s ability to find relevant chunks, as
well as on the model’s ability to leverage the information from the retrieved chunks.
RAG can be implemented in many ways. The main degrees of freedom are: how documents are split into chunks,
which embedding model is used to store chunks in a vector database, which algorithm is used for retrieving relevant
chunks from the vector database, and how many chunks to retrieve and include in the model’s prompt. Commonly, a
larger amount of chunks is retrieved and a so-calledrerankermodel is used to determine the top chunks to pass on to
the generator model.
2.4 Constrained Decoding
Constrained decoding (CD) [ 5,16] (also known asstructuredorguidedgeneration) is a general technique that modifies
an LLM’s next-token predictions based on predefined constraints to prevent the generation of tokens that would lead to
undesirable output sequences. This technique operates without requiring model modifications or prompt adjustments.
How constraints are obtained is highly application-specific. For representing constraints, regular expressions or
context-free grammars are commonly used [2, 7, 53].
The general constrained decoding algorithm (cf. [ 27,33,48] extends an LLM’s regular autoregressive generation loop
with a so-calledcompletion engine[ 27,33,51], which dynamically filters possible next token to enforce compliance
with given constraints. More concretely, in each generation step, the current token sequence 𝑠1, . . . , 𝑠 𝑖and the token
vocabulary 𝑡1, . . . , 𝑡 𝑛are passed to the completion engine, which checks for each token 𝑡𝑗whether 𝑠1, . . . , 𝑠 𝑖, 𝑡𝑗satisfies the
constraints and would therefore be a valid continuation. The completion engine returns a token mask 𝑚∈{ 0,1}𝑛which
is used to set the probability 𝑝𝑗of all tokens 𝑡𝑗to zero that failed the constraint check. Based on the masked probability
distribution, the next token𝑠 𝑖+1is sampled. Note that this process can be optimized in many ways [7, 33, 48, 53].
3 Retrieval-Augmented Generation for Web API Invocations
In order to apply RAG to web API invocations, we retrieve chunks from OpenAPI specifications using the respective
task description as query, as shown in Figure 2. While RAG systems typically split documents into chunks based on a
Manuscript submitted to ACM

Mitigating Errors in LLM-Generated Web API Invocations 7
Listing 2. Example of a retrieved endpoint chunk in a TypeScript-inspired format
// GET / tags /{ tag - name }/ media / recent
// Get a list of recently tagged media
type GetTagsMediaRecentRequest = {
path : {
"tag - name ":string// The tag name
}
query : {
count ?:number// Count of tagged media to return
min_tag_id ?:string// Return media before this`min_tag_id`
max_tag_id ?:string// Return media after this`max_tag_id`
access_token ?:string// API key security scheme
}
header : {
Authorization ?:string// OAuth2 security scheme
}
}
predetermined length or sentence boundaries, our chunks encompass the complete documentation of a single endpoint.
We do this because API specifications are highly context-dependent—retrieving smaller chunks out of context would
be detrimental. For example, the documentation of a parameter is useless if it is unclear which endpoint it belongs to.
To ensure each chunk is self-contained and contains all relevant information, we preprocess OpenAPI specifications
prior to chunking, inlining all references and copying API-wide security schemes and path-level parameters into each
endpoint’s parameter list.
To embed, store, and retrieve chunks, our retriever uses the all-MiniLM-L6-v26embedding model, a Chroma vector
database, and approximate nearest neighbor search, respectively. To increase the hit rate, each chunk is stored with up
to five different embeddings, created by including and excluding different parts of the endpoint documentation in the
string that is fed into the embedding model. For example, one variant contains only the endpoint’s path and description,
while other variants additionally contain the parameter list or the response schema. Across all tasks in the dataset, our
retriever achieves 75.7% top-1 accuracy and 95.2% top-5 accuracy (after deduplicating chunks). We also implemented
reranking but did not use it, because it could not improve the retriever performance any further.
Before injecting the retrieved endpoint documentation into the prompt, it is formatted and truncated in a postpro-
cessing step. The goal of the truncation is to avoid exceeding a model’s context length. We cut the output before the
line in which the character threshold is exceeded and append a [TRUNCATED] marker. To reduce the need for truncation,
we format the documentation in a way that is compact but also easy to understand for LLMs. We explored different
formatting options such as JSON and YAML, but they turned out to be token-inefficient due to a lot of indentation and
verbose constructs such as name: <name> , where simply stating the parameter name would be sufficient. Instead, we
designed a custom format inspired by TypeScript type declarations, as shown in Listing 2.
4 Constrained Decoding for Web API Invocations
The main challenge we have to address when applying constrained decoding in our setting is constraint generation.
Since every web API requires different constraints, we implement an automatic translation from OpenAPI specifications
to constraints that are represented as regular expressions. When generating the constraints, the special characteristics
and complexities of web API invocations need to be taken into account. Besides respecting the general JavaScript and
the specific Axios syntax (cf. Listing 1), our constraints need to handle interdependencies between allowed URLs, HTTP
6https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
Manuscript submitted to ACM

8 Maninger et al.
API
specification// Create new event … and add it to calendar …
LLM
Constraint
translatorConstraintsCompletion
engineDecoder s1s2s3si…t1
t2
t3
tn…p1
p2
pn…p3Taskviacodecomment </>Code
Fig. 3. Constrained decoding framework for generating web API invocations. In regular autoregressive generation, a decoder takes the
token sequence 𝑠1, . . . , 𝑠 𝑖and predicts the probabilities 𝑝𝑗for each possible next token 𝑡𝑗, based on which 𝑠𝑖+1is selected. Constrained
decoding augments this loop with a constraint check that sets the probability of tokens that fail the check to zero. We derive our
constraints from OpenAPI specifications, and they ensure that generated API invocations comply with the specification.
methods, and different kinds of parameters. Moreover, the number, ordering, and nesting of parameters can be arbitrary,
while some parameters are required and some are optional, making advanced constraint formulations necessary.
We implemented a simple, model-agnostic constrained decoding framework for our investigations, which is depicted
in Figure 3. The constraints are automatically derived from OpenAPI specifications and represented as regular expressions
(regex). They are designed to ensure that generated API invocations are always compliant with the API’s specification.
Note that our contribution here lies in this transformation of specifications into constraints, and not the completion
engine, which is a straightforward implementation of the standard CD algorithm sketched in Section 2.4. We developed
a custom completion engine to maintain full control over the system, explore features not supported by existing
completion engines (e.g., advanced regex features and constraint orchestration as described below), and add some
debugging functionality. For professional use cases, migrating to an off-the-shelf engine (cf. Section 8) would be
expedient to enhance performance and robustness.
A main challenge when translating OpenAPI specifications to regular expressions in a generalizable and scalable way
is the sheer size and complexity of real-world APIs, which is also reflected in the respective specification files. Each API
can have dozens if not hundreds of endpoints (in the specifications we worked with, up to 523 in the GitHub API), and
each endpoint can have dozens of parameters, distributed on different locations (body, header, path, query). Parameters
can be optional or required, and they can be passed in arbitrary order. Additionally, many parameters have complex
data types (objects and arrays) with a strictly defined nesting structure. There are also interdependencies between
different parts of an API invocation, e.g., only for certain HTTP methods, parameters can be placed in the request body.
In summary, some elements of an API invocation must be followed strictly, while other elements are flexible.
Additional challenges arise at the syntax-level of the language the API invocation is implemented in. Our regexes
should accept valid JavaScript while not limiting variations permitted by the language, such as formatting or the use of
different quotation marks. This is important as different LLMs show different stylistic preferences—forcing them away
from their preferred style may degrade performance. There are also a lot of edge cases that need to be handled carefully
by the constraints, e.g., an escaped quotation mark must not terminate a string literal, and many other seemingly minor
details—our experiments showed that models will find and exploit such loopholes.
Manuscript submitted to ACM

Mitigating Errors in LLM-Generated Web API Invocations 9
We address the challenges by structuring constraints hierarchically from the HTTP method and the URL to the
different parameters. Once the method is matched, the search space is reduced to URLs for which an endpoint with
the respective method is defined. Next, once the URL is matched, the search space is reduced to the parameters of the
identified endpoint. Finally, once the parameter name is matched, the search space is reduced to values with the right
data type. Within the parameter lists of a given endpoint, we use capture groups7to keep track of parameters that have
been defined. Conditional patterns7are then used to check the existence of a specific group and thus guard against the
double definition of a parameter. Moreover, conditionals are used in conjunction with negative lookahead assertions7to
make it impossible to close a parameter list before all required parameters have been defined.
While we could represent all our constraints as one monolithic regex, we opted for a more modular and flexible
approach8. We organize constraints as a set of so-calledgeneration rules. Each generation rule has a regex-encoded
start condition,stop condition, andbody. Once the start condition matches the code generated so far, we constrain the
generation according to the body until the stop condition matches.
There is one specialized generation rule for each HTTP method, which is triggered when the corresponding Axios
invocation is generated (e.g., axios.post( , with the opening parenthesis being part of the trigger). As multiple
generation rules may be active at the same time (a token is only admissible if no rule rejects it), we can leverage
overlapping constraints. We do so via a special generation rule that becomes active before all other rules and guides
the generation towards the start condition of the other rules, ensuring one of them is going to be activated (thus
avoiding using methods not supported by the given API). Moreover, we let generation rule bodies overlap with their
stop conditions to ensure they properly terminate. This approach can also be generalized to many other applications of
constrained decoding to orchestrate multiple constraints in a flexible and efficient manner.
Given the breadth of API designs that can be described through OpenAPI specifications plus the complexity of the
resulting constraints, we implemented a test suite to ensure that (a) any specification documents can be processed and
translated to constraints and (b) resulting constraints allow and disallow exactly the structures they are supposed to. For
(a), we successfully parse 34 real-world OpenAPI specifications, including those used in the synthetic and real-world
dataset as well as additional ones. For (b), we created 125 test cases, comprising positive and negative code examples,
that our regex constraints must, respectively, match or not match—all these test cases pass. Further implementation
details about our constraints are discussed in Appendix B.
5 Real-World Dataset Creation
WAPIIBench, introduced in our previous work [ 25], is based on a synthetic dataset. This dataset enables controlled,
large-scale experiments with precise ground truth. However, it may not fully capture the range of real-world API
integration scenarios. In this paper, we therefore complement the original synthetic benchmark with a newly constructed
dataset derived from real-world repositories.
To construct the new dataset we addressed several challenges that were left open in our earlier work. Real-world
code rarely contains sufficient information to serve directly as an unambiguous task description, which makes task
reconstruction labor-intensive and limits the number of tasks that can realistically be included. Moreover, even when a
plausible task can be identified, manual inspection is still necessary to determine whether the code is correct, or at least
complete and plausible, which parts are relevant to the task, and which API version it assumes.
7https://docs.python.org/3/library/re.html
8They could still be joined if needed for compatibility with other frameworks.
Manuscript submitted to ACM

10 Maninger et al.
Table 1. Number of tasks per API in the real-world dataset
API # tasks
Etherscan 3
FrankerFaceZ 2
GitHub 8
Google Maps Platform 2
Instagram 2
JSONPlaceholder 3
npm Registry 2
Slack Web 1
Telegram Bot 1
YouTube Data 2
Zephyr Cloud 2
Total 28
The extension of WAPIIBench with the real-world dataset allows us to evaluate in two complementary settings. The
synthetic dataset provides comprehensive and systematic coverage of all endpoints for a limited number of APIs under
tightly controlled conditions. However, it may not reflect the real-world web API integration scenarios. In contrast,
being derived from repository code, the tasks in the real-world dataset reflect common use cases of widely used APIs.
The real-world dataset provides less comprehensive—unlike the synthetic dataset, it is necessarily incomplete and does
not cover all endpoints of a given API—but more diverse and realistic coverage, including file-level code context of
varying complexity.
To derive the real-world dataset, we queried BigQuery9for GitHub JavaScript files that import the Axios library,
yielding an initial sample of 277 files. We manually screened each file to identify web API invocations suitable for
being converted into a task. We excluded files if they lacked actual Axios invocations, obscured the server address
(making the target API unidentifiable), or interacted with APIs lacking an OpenAPI specification. Finally, to prevent
skewing the dataset’s API distribution, we partially excluded two files—effectively web API wrapper libraries—that
contained disproportionately high numbers of invocations to a single API. After filtering, 18 files remained, from which
we constructed a dataset consisting of 28 tasks (some files yielded more than one). The dataset spans 11 APIs, with the
task distribution shown in Table 1. All retrieved code files plus the reason for exclusion and other notes can be found in
our artifact.
To convert a file into one or more tasks we performed the following manual steps: localizing the API invocation in
the source code; identifying the corresponding endpoint in the API specification; authoring a natural-language task
description; creating a ground-truth request configuration; preparing starter code grounded in the original code context;
and capturing relevant definitions within that starter code. An example task is shown in Listing 4b;
Starter Code Preparation:By default, the starter code is a slice of the original file from its beginning up to the API
invocation site, giving the model realistic file-level context10. However, a few targeted adjustments are necessary.
(a) Removing giveaways: If the file defines constants such as API keys or the request URL, those definitions are stripped
out; leaving them in would hand the model part of the answer. (b) Removing similar calls: Code that closely resembles
9https://cloud.google.com/bigquery/
10The length of starter codes in the dataset varies between 4 and 163 lines.
Manuscript submitted to ACM

Mitigating Errors in LLM-Generated Web API Invocations 11
// Save the project with gid'778899'as a template with
the name'New Project Template'and make it public
to its team .
constaxios = require ('axios');
axios .
(a) Synthetic datasetconstaxios = require ('axios');
constUser = require ('../ users / userModel'). User ;
module . exports = {
getUniqueTagPics (req , res ) {
consthashtag = req . body . hashtag ;
User . where ({ id: req . body . userId })
. fetch ()
. then (( user ) => {
if(! user ) {
res . status (404) . send ('User not found');
}else{
constaccessToken = user . get ('instagram_token');
if( accessToken ) {
// Get a list of media recently tagged with the
value in`hashtag`. Authenticate with
the access token in`accessToken`.
axios .
(b) Real-world dataset
Fig. 4. Comparison of a task and starter code from the synthetic dataset [ 25] to a task and starter code from the real-world dataset
(Sec. 5). The former targets the Asana API, the latter the Instagram API. The task is included in the starter code as a line comment.
the target invocation (e.g., other calls to the same API) is also removed, since it would encourage the model to copy
rather than reason.
Handling Variables in Real-World Tasks:Unlike the synthetic dataset where all API parameters are plain literals,
the real-world tasks use file-level starter code, meaning some parameters are passed as variables or constants. This
introduces two challenges.
(a) Injecting definitions at execution time. Because our pipeline only executes the isolated API invocation (not the
surrounding starter code), any definitions that must be in scope are captured in advance and injected into the execution
environment. For variables with statically unknown values, such as function parameters, we generate random instances
of the appropriate data type and schema. This prevents the model from guessing a literal value and forces it to reference
the correct variable. These random values are used only during execution and correctness analysis (cf. Section 2); the
model never sees them during code generation.
(b) Handling unexpected variable references. A model may attempt to use variables from the starter code that were
not part of the intended solution. Without those definitions present, execution would throw a ReferenceError . Since
we would rather count such cases as an incorrect argument value than a non-executable implementation, we iteratively
expand the set of injected definitions until all remaining ReferenceError in the execution logs stem exclusively from
genuinely hallucinated identifiers.
Beyond handling task-specific starter codes and variable definitions, we extended the evaluation pipeline as well as
the retriever and constraint generator to support a larger set of OpenAPI features present in the specifications of the
APIs in the new dataset.
For instance, to supportserver variables(similar to path parameters but in the URL’s server part), we needed to
(a) normalize model-generated URLs before matching them against the specification to find the endpoint they selected,
(b) merge the list of server variables with the list of the endpoint’s path parameters, (c) extract the values of all server
variables and path parameters from the URL, and (d) add the resulting name–value pairs to the request configuration,
so they can be properly analyzed later in the evaluation pipeline.
Manuscript submitted to ACM

12 Maninger et al.
Moreover, the extraction of Axios invocations from the generated codes in preparation of executing them was
reworked to remove any assumptions about the structure of the starter code, as those assumptions would not hold for
the real-world dataset.
Finally, since many tasks in the new dataset require passing variables from the starter code as path parameters, we
implemented constraints that allow inserting variables into URL strings via string interpolation as well as concatenation.
The outlined extensions have made WAPIIBench much more flexible w.r.t. task structure and robust w.r.t. API-specific
peculiarities, facilitating reuse and further development. Adding a new task to a dataset is as simple as adding a new
entry to a JSON file—it will automatically be evaluated together with the previously existing tasks. Adding an entirely
new datasets requires less than 70 lines of code (mostly boilerplate), as evidenced by our script to run the evaluation
with the new real-world dataset. If a new task or dataset involves a new API, its OpenAPI specification has to be placed
in a dedicated directory and a mapping from the API’s file name to the API’s proper name has to be added to an existing
dictionary.
6 Evaluation
This section covers our experimental setup and results for different generation settings as well as a discussion about
their efficiency and answers the research questions formulated in Section 1.
6.1 Experimental Setup
To assess their efficacy for web API invocation code generation and answer the research questions posed in Section 1,
we put our retriever (Section 3) and constraints (Section 4) into action using WAPIIBench’s dataset and evaluation
pipeline (Section 2.2) as well as the new real-world dataset (Section 5).
Experiment Variations.We evaluate four differentgeneration settings:
VanillaGeneration based only on the prompt and starter code
RAGRetrieval-augmented generation
CDConstrained decoding
RAG + CDRetrieval-augmented generation combined with constrained decoding
Furthermore, conduct all experiments with two different starter code variants, a.k.a.setups(cf. also Section 2.2):
Full completion The model has to select the correct endpoint (URL plus HTTP method) as well as request arguments
in order to solve the given task
Argument completion The correct endpoint is already provided, the model has to select the correct arguments in
order to solve the given task
Models.We study the performance of 24 state-of-the-art open-source LLMs, listed in Table 2. These models were
selected due to their frequent appearance in coding leaderboards11. To obtain an approximate upper bound on perfor-
mance, we also evaluated OpenAI’s commercial GPT-4o and GPT-4o mini models. As they are closed-source, we cannot
apply constrained decoding to the GPT models. Due to a model-internal bug, Code Llama (34B) cannot use constrained
decoding either.
For all experiments, greedy decoding is used to ensure deterministic and reproducible results. In the RAG experiments,
five endpoint chunks are retrieved per task. If the total length of the retrieved text exceeds 8000 characters, it is truncated.
11E.g., https://evalplus.github.io/leaderboard.html
Manuscript submitted to ACM

Mitigating Errors in LLM-Generated Web API Invocations 13
Table 2. Evaluated LLMs and their sizes
Model Size
CodeT5+ [49] 6B, 16B
StarCoder [22] 1B, 3B, 7B, 15.5B
StarCoder2 [23] 3B, 7B, 15B
DeepSeek-Coder [15] 1.3B, 6.7B, 33B
Qwen2.5-Coder [18] 0.5B, 1.5B, 3B, 7B, 14B, 32B
Llama 3.1 [9] 8B, 70B
Code Llama [38] 7B, 13B, 34B, 70B
GPT-4o mini [28] N/A
GPT-4o [29] N/A
This threshold is set to strike a balance between supporting models with shorter context lengths and truncating as few
retriever outputs as possible. Only for CodeT5+, we have to reduce the threshold to 4000 to avoid inference-time crashes
and skip some tasks from the real-world dataset as their starter code alone already exceeds the model’s maximum
context length (we count this as a generation-time error). The prompts used for the synthetic and the real-world tasks
are given in Listing 3 and 4 in the appendix.
Metrics.We focus our discussion on the following four key metrics; a multitude of additional metrics is provided in
Appendix F:
Correct implementations Ratio of generated codes that are executable and correctly solve the given task, i.e., their
request configuration matches the ground-truth
Hallucinated endpoints Ratio of generated codes that use a combination of URL and HTTP method that is not
defined in the API’s specification12(only applicable to full completion)
Hallucinated implementations Ratio of generated codes that use at least one request parameter that is not defined
in the API’s specification (only applicable to argument completion)
Executable implementations Ratio of generated codes that are executable, i.e., complete, syntactically correct, and
free of runtime errors
Here, all metrics will be reported relative to the total number of tasks in the respective dataset. Metrics relative to only the
tasks that resulted in executable code are provided in the appendix. For a fair comparison between generation settings
compatible and incompatible with closed-source models, we always exclude the GPT-4o models when aggregating
metrics. Otherwise, vanilla and RAG would get an advantage over CD and RAG + CD due to the contribution of the
powerful GPT-4o models to the aggregate values. Additionally, we exclude models from average calculations if they did
not generate any executable implementation.
In the following subsections, we elaborate on the experimental results for the four generation settings. The results
for the synthetic dataset are presented in Figure 5—showing the ratio of correct implementations for full and argument
completion across all four generation settings—and Figure 6—showing the ratio of hallucinated endpoints and imple-
mentations, respectively, for the two setups. Similarly, Figures 7 and 8 present the results for the real-world dataset.
Comprehensive tables with numeric results for all metrics are provided in Appendix F.
12While we use the intuitive term “hallucinated” for this error category, it subsumes any mismatch with the API specification, including cases where the
model’s intention may be correct, but its spelling is incorrect.
Manuscript submitted to ACM

14 Maninger et al.
CodeT5+ (6B)CodeT5+ (16B) StarCoder (1B) StarCoder (3B) StarCoder (7B)
StarCoder (15.5B)StarCoder2 (3B) StarCoder2 (7B)StarCoder2 (15B)
DeepSeek-Coder (1.3B) DeepSeek-Coder (6.7B) DeepSeek-Coder (33B) Qwen2.5-Coder (0.5B) Qwen2.5-Coder (1.5B)Qwen2.5-Coder (3B) Qwen2.5-Coder (7B)Qwen2.5-Coder (14B) Qwen2.5-Coder (32B)Llama 3.1 (8B)Llama 3.1 (70B)Code Llama (7B)Code Llama (13B) Code Llama (34B) Code Llama (70B)GPT-4o miniGPT-4o0.00.10.20.30.40.50.6Correct implementationsFull completionVanilla RAG CD RAG + CD
CodeT5+ (6B)CodeT5+ (16B) StarCoder (1B) StarCoder (3B) StarCoder (7B)
StarCoder (15.5B)StarCoder2 (3B) StarCoder2 (7B)StarCoder2 (15B)
DeepSeek-Coder (1.3B) DeepSeek-Coder (6.7B) DeepSeek-Coder (33B) Qwen2.5-Coder (0.5B) Qwen2.5-Coder (1.5B)Qwen2.5-Coder (3B) Qwen2.5-Coder (7B)Qwen2.5-Coder (14B) Qwen2.5-Coder (32B)Llama 3.1 (8B)Llama 3.1 (70B)Code Llama (7B)Code Llama (13B) Code Llama (34B) Code Llama (70B)GPT-4o miniGPT-4o0.00.10.20.30.40.50.60.7Correct implementationsArgument completionVanilla RAG CD RAG + CD
Fig. 5. Comparison ofcorrect implementationsbetween generation settings on thesynthetic dataset
6.2 Vanilla Generation
The results for vanilla generation on the synthetic dataset are consistent with our previous findings [ 25]13. We will
focus our analysis here on a few key aspects and refer the reader to our prior work for a broader discussion.
The overall benchmark performance for the full completion setup is low, at an average correctness rate among
open-source models of 13% and Code Llama (70B) as the best performing model at 30%. GPT-4o achieves 58%. All
Qwen2.5-Coder and Llama 3.1 models except Qwen2.5-Coder (32B) do not produce any executable implementation as
13Small quantitative deviations are due to minor adjustments of the prompt and implementation.
Manuscript submitted to ACM

Mitigating Errors in LLM-Generated Web API Invocations 15
CodeT5+ (6B)CodeT5+ (16B) StarCoder (1B) StarCoder (3B) StarCoder (7B)
StarCoder (15.5B)StarCoder2 (3B) StarCoder2 (7B)StarCoder2 (15B)
DeepSeek-Coder (1.3B) DeepSeek-Coder (6.7B) DeepSeek-Coder (33B) Qwen2.5-Coder (0.5B) Qwen2.5-Coder (1.5B)Qwen2.5-Coder (3B) Qwen2.5-Coder (7B)Qwen2.5-Coder (14B) Qwen2.5-Coder (32B)Llama 3.1 (8B)Llama 3.1 (70B)Code Llama (7B)Code Llama (13B) Code Llama (34B) Code Llama (70B)GPT-4o miniGPT-4o0.00.10.20.30.40.50.60.7Hallucinated endpointsFull completionVanilla RAG CD RAG + CD
CodeT5+ (6B)CodeT5+ (16B) StarCoder (1B) StarCoder (3B) StarCoder (7B)
StarCoder (15.5B)StarCoder2 (3B) StarCoder2 (7B)StarCoder2 (15B)
DeepSeek-Coder (1.3B) DeepSeek-Coder (6.7B) DeepSeek-Coder (33B) Qwen2.5-Coder (0.5B) Qwen2.5-Coder (1.5B)Qwen2.5-Coder (3B) Qwen2.5-Coder (7B)Qwen2.5-Coder (14B) Qwen2.5-Coder (32B)Llama 3.1 (8B)Llama 3.1 (70B)Code Llama (7B)Code Llama (13B) Code Llama (34B) Code Llama (70B)GPT-4o miniGPT-4o0.00.10.20.30.40.5Hallucinated implementationsArgument completionVanilla RAG CD RAG + CD
Fig. 6. Comparison ofhallucinated endpointsbetween generation settings on thesynthetic dataset
they are unable to continue the given starter code in a syntactically valid way. The remaining models hallucinate on
average 31% of all endpoints.
On the real-world dataset, we observe a higher baseline performance—47% correctness on average among open-source
models producing executable implementations. We attribute this to the tasks in this dataset being drawn from real-world
repositories, and thus having a high likelihood of representing common use cases of wide-spread APIs that are also
represented in the models’ training corpora. Nonetheless, most Qwen2.5-Coder and Llama 3.1 models that were unable
to generate syntactically valid code completions for synthetic tasks fail here as well. The average rate of hallucinated
endpoints among the remaining models remains high at 29%.
Manuscript submitted to ACM

16 Maninger et al.
CodeT5+ (6B)CodeT5+ (16B) StarCoder (1B) StarCoder (3B) StarCoder (7B)
StarCoder (15.5B)StarCoder2 (3B) StarCoder2 (7B)StarCoder2 (15B)
DeepSeek-Coder (1.3B) DeepSeek-Coder (6.7B) DeepSeek-Coder (33B) Qwen2.5-Coder (0.5B) Qwen2.5-Coder (1.5B)Qwen2.5-Coder (3B) Qwen2.5-Coder (7B)Qwen2.5-Coder (14B) Qwen2.5-Coder (32B)Llama 3.1 (8B)Llama 3.1 (70B)Code Llama (7B)Code Llama (13B) Code Llama (34B) Code Llama (70B)GPT-4o miniGPT-4o0.00.10.20.30.40.50.60.7Correct implementationsFull completionVanilla RAG CD RAG + CD
CodeT5+ (6B)CodeT5+ (16B) StarCoder (1B) StarCoder (3B) StarCoder (7B)
StarCoder (15.5B)StarCoder2 (3B) StarCoder2 (7B)StarCoder2 (15B)
DeepSeek-Coder (1.3B) DeepSeek-Coder (6.7B) DeepSeek-Coder (33B) Qwen2.5-Coder (0.5B) Qwen2.5-Coder (1.5B)Qwen2.5-Coder (3B) Qwen2.5-Coder (7B)Qwen2.5-Coder (14B) Qwen2.5-Coder (32B)Llama 3.1 (8B)Llama 3.1 (70B)Code Llama (7B)Code Llama (13B) Code Llama (34B) Code Llama (70B)GPT-4o miniGPT-4o0.00.10.20.30.40.50.60.70.8Correct implementationsArgument completionVanilla RAG CD RAG + CD
Fig. 7. Comparison ofcorrect implementationsbetween generation settings on thereal-world dataset
On both datasets, models achieve generally higher correctness rates for argument completion than for full completion—
26% and 55% on average, respectively. On the synthetic dataset, Qwen2.5-Coder (14B) scores highest at 47%. Interestingly,
GPT-4o-mini performs better than GPT-4o (63% vs. 59%). An important caveat for the real-world dataset is that several
tasks included do not require parameters in locations other than the path, i.e., there are no further arguments to complete
and all the model has to do is recognize this fact and terminate the statement by generating a closing parenthesis.
On average, 39% and 20% of implementations contain hallucinations, depending on the dataset. We hypothesize
that this increased hallucination rate compared to the full completion setup stems from the fact that for argument
completion, the model is forced to generate arguments for a predetermined endpoint that it might not be familiar with.
Manuscript submitted to ACM

Mitigating Errors in LLM-Generated Web API Invocations 17
CodeT5+ (6B)CodeT5+ (16B) StarCoder (1B) StarCoder (3B) StarCoder (7B)
StarCoder (15.5B)StarCoder2 (3B) StarCoder2 (7B)StarCoder2 (15B)
DeepSeek-Coder (1.3B) DeepSeek-Coder (6.7B) DeepSeek-Coder (33B) Qwen2.5-Coder (0.5B) Qwen2.5-Coder (1.5B)Qwen2.5-Coder (3B) Qwen2.5-Coder (7B)Qwen2.5-Coder (14B) Qwen2.5-Coder (32B)Llama 3.1 (8B)Llama 3.1 (70B)Code Llama (7B)Code Llama (13B) Code Llama (34B) Code Llama (70B)GPT-4o miniGPT-4o0.000.050.100.150.200.250.300.350.400.45Hallucinated endpointsFull completionVanilla RAG CD RAG + CD
CodeT5+ (6B)CodeT5+ (16B) StarCoder (1B) StarCoder (3B) StarCoder (7B)
StarCoder (15.5B)StarCoder2 (3B) StarCoder2 (7B)StarCoder2 (15B)
DeepSeek-Coder (1.3B) DeepSeek-Coder (6.7B) DeepSeek-Coder (33B) Qwen2.5-Coder (0.5B) Qwen2.5-Coder (1.5B)Qwen2.5-Coder (3B) Qwen2.5-Coder (7B)Qwen2.5-Coder (14B) Qwen2.5-Coder (32B)Llama 3.1 (8B)Llama 3.1 (70B)Code Llama (7B)Code Llama (13B) Code Llama (34B) Code Llama (70B)GPT-4o miniGPT-4o0.000.050.100.150.200.250.30Hallucinated implementationsArgument completionVanilla RAG CD RAG + CD
Fig. 8. Comparison ofhallucinated endpointsbetween generation settings on thereal-world dataset
In contrast, for full completion, it is free to select any endpoint it is familiar with—even if it is the wrong choice, it is
less likely to be a hallucination.
Summary:Vanilla generation yields low overall correctness, particularly for full completion (13% average across
open-source models on the synthetic dataset), with hallucinated endpoints affecting nearly a 1/3 of all generations.
Model size is only a weak predictor of performance, and several recent models fail to produce executable code
entirely. Real-world tasks yield higher baseline correctness (47%), presumably reflecting the prevalence of common
API patterns in training data, yet hallucination rates remain substantial. These results confirm that unassisted
Manuscript submitted to ACM

18 Maninger et al.
LLMs are unreliable for web API invocation code generation and motivate the mitigation approaches studied in the
following subsections.
6.3 Retrieval-Augmented Generation
In the case of the full completion setup on the synthetic dataset, RAG yields a clear benefit: it more than doubles the
average correct implementation rate compared to vanilla generation, raising it to 22%. The picture changes substantially,
however, for the argument completion setup on the synthetic dataset and for both setups on the real-world dataset: in
all these cases, RAG’s correctness scores fall within 1% of the vanilla baseline, indicating negligible net impact.
Examining per-model results reveals that the effect of RAG is highly model-dependent rather than uniformly beneficial
or harmful. Individual models swing between a +562% gain in correctness (StarCoder 1B, full completion, synthetic
dataset) and complete collapse to 0% correctness (CodeT5+ 6B, full completion, synthetic dataset), underscoring that
RAG’s utility is strongly conditioned on the underlying model’s capacity to leverage retrieved context.
The ambivalent outcomes stem from several competing effects. On the positive side, RAG helps models select the
correct API endpoint—an advantage that, by definition, vanishes in the argument completion setup, where the endpoint
is already given.
On the negative side, RAG’s potential benefits in selecting correct parameters are overshadowed by a substantial
increase in the number ofunnecessary parametersfor most models (see Appendix F). These parameters are valid
(but optional) for the chosen endpoint according to the API specification yet irrelevant or undesirable for the task at
hand. We hypothesize that most LLMs—even those not explicitly instruction-tuned—tend to treat retrieved context as
implicit guidance, attempting to incorporate as many listed parameters as possible into the generated API call. This
over-inclusion is further amplified when the same parameter is defined across multiple endpoints and thus appears
repeatedly in the retrieved context. A prime example is Asana’s optional opt_pretty parameter, which is used to format
responses in a human-readable way. Intended for debugging, it can be passed to every API endpoint. Accordingly, it
appears in every retrieved chunk, and we observe a high frequency of use among models. How susceptible a given
model is to this behavior ultimately determines how much RAG hurts its correctness.
For Qwen2.5-Coder and Llama 3.1, RAG has no effect on their tendency to produce only non-executable code. Among
the remaining models, however, RAG delivers substantial reductions in hallucinations on average: 82% fewer on the
synthetic dataset and 10% fewer on the real-world dataset for full completion, and 60% and 51% fewer, respectively, for
argument completion.
RQ1:To what extent does RAG improve the correctness of LLM-generated web API invocation code?
Answer:RAG effectively reduces hallucinations but does not uniformly improve correctness and can even degrade
it. Its benefits are most pronounced for full completion, where it also aids endpoint selection, while its impact on
argument completion is negligible. Whether RAG is net-positive depends strongly on the individual model: models
prone to incorporating retrieved parameters indiscriminately suffer the greatest correctness losses, making careful
model-specific evaluation advisable before deploying RAG in practice.
6.4 Constrained Decoding
On the synthetic dataset, constrained decoding consistently increases correctness for all models and both setups. For
full completion—where RAG achieved its highest average gain at +100%—CD outperforms it with +209%, leading to an
Manuscript submitted to ACM

Mitigating Errors in LLM-Generated Web API Invocations 19
average correctness rate of 28%. For argument completion, the gain is +143%, corresponding to an average correctness
rate of 57%. Code Llama (70B) remains the strongest open-source model at 50% for full completion, closing the gap to
GPT-40 at 58%. For argument completion, Code Llama (70B) with CD achieves 72% correctness, outperforming GPT-4o
(59%) and GPT-4o mini (63%).
The Qwen2.5-Coder and Llama 3.1 models that were previously unable to generate any executable code in the full
completion setup, now constrained into the required syntactic structure, achieve executability rates similar to the
other models. However, disregarding models with 0% executability, the overall rate of non-executable implementations
increases slightly compared to the previous generation settings: from 92% (vanilla) and 86% (RAG) to 81% (CD) for full
completion and from 98% (vanilla) and 93% (RAG) to 85% (CD) for argument completion. A potential explanation for
this is the distortion an LLM’s next token probability distribution that CD may cause [2].
On the real-world dataset, we see similar trends, but with a smaller amplitude due to the higher baseline performance.
CD increases correctness by +69% and +18%, depending on the setup. Unlike on the synthetic dataset, on the real-world
dataset, a few models drop in correctness with CD: 6 out of 23 models for full completion and 3 out of 23 for argument
completion. The only extreme case is Qwen-2.5-Coder (0.5B), which drops correctness by -82% on full completion—in
all other cases, correctness drops by less than 30%. This drop is likely a consequence of the reduced executability rates.
The most striking property of CD—one that holds universally across all models, setups, and datasets—is its guarantee
of zero hallucinated endpoints and implementations. This is a direct consequence of how the constraints are constructed
(cf. Section 4), and our experiments consistently confirm it.
RQ2:To what extent does CD improve the correctness of LLM-generated web API invocation code?
Answer:CD eliminates hallucinations of URLs, HTTP methods, and arguments by construction, reducing them to
zero across all evaluated settings. Beyond this guarantee, CD consistently and substantially boosts the correctness of
generated API invocations—in some cases making smaller constrained models competitive with larger unconstrained
ones. Given these clear benefits and the negligible risk of correctness degradations, CD can be recommended
unconditionally for web API invocation tasks.
6.5 Combining RAG and CD
The combination of RAG and CD retains CD’s zero-hallucination guarantee while additionally leveraging retrieved
context to guide generation. This yields strong average correctness gains over the vanilla baselines: +332% and +111%
on the synthetic dataset, and +74% and +14% on the real-world dataset, for full and argument completion, respectively.
RAG + CD also achieves the highest correctness scores across all setups and datasets: 67% (DeepSeek-Coder 33B) and
73% (Qwen2.5-Coder 14B) for full and argument completion on the synthetic dataset, and 71% (Llama 3.1 70B) and 71%
(Qwen2.5-Coder 32B / Llama 3.1 70B) on the real-world dataset.
Compared to CD alone, RAG + CD increases the share of executable implementations by +10% and +7% for full and
argument completion on the synthetic dataset, with only a +1% gain on the real-world dataset. Correctness gains over CD,
however, are limited to full completion (+43% synthetic, +11% real-world); for argument completion, average correctness
declines (-14% synthetic, -3% real-world), consistent with the parameter overuse effect described in Section 6.3.
RAG + CD is also less consistent than CD alone regarding the achieved correctness gains. CD improved correctness
for all models on the synthetic dataset and reduced it for only 6 (full completion) and 3 (argument completion) models
Manuscript submitted to ACM

20 Maninger et al.
Table 3. Generation speed intokens per secondfor each API and generation setting
API Vanilla RAG CD RAG + CD
Asana 29.85 29.54 7.15 6.27
Google Calendar 30.94 26.54 10.36 7.25
Google Sheets 31.72 26.26 10.37 2.31
Slack Web 29.23 29.59 7.30 7.49
on the real-world dataset. RAG + CD, by contrast, reduces correctness for 1 and 2 models on the synthetic dataset, and
for 9 and 5 models on the real-world dataset—a wider spread of regressions.
RQ3:To what extent does combining RAG and CD improve the correctness of LLM-generated web API invocation code
compared to each technique in isolation?
Answer:RAG + CD achieves the highest correctness scores across all setups and datasets, outperforming both
techniques in isolation. However, this peak performance comes at the cost of consistency: compared to CD alone,
RAG + CD produces more correctness regressions across individual models, and its average correctness gain over
CD is limited to full completion—argument completion suffers due to parameter overuse. RAG + CD is therefore
advisable when maximum correctness is the priority and model behavior with retrieved context has been validated
in advance. When neither condition holds, CD alone is the safer and more robust choice.
6.6 Efficiency
Depending on the use case, the efficiency of the generation setting may be an important factor as it directly affects
inference speed. To estimate the overhead of RAG and CD, we run a series of runtime measurements and calculate the
average token per second (TPS) generation speed. The results are shown in Table 3.
We selected one representative task for each API of the synthetic dataset and repeated the generation ten times
per task on one A100 40GB GPU, using StarCoder2 [ 23] and the full completion setup. The time to set up and run the
retriever for RAG or to create the constraints for CD is excluded from the measurements. Since the variation in runtime
between individual runs was low14, we focus our discussion on the average TPS.
The results show that vanilla generation is largely invariant to the API or task (ca. 30 TPS), which is expected since
the model’s prompt is almost static except for the API name and the task description, which have a negligible impact
on the input length. RAG is overall slightly slower than vanilla due to the increased computational cost of processing
the retrieved context. For Asana and Slack, the difference is indistinguishable from measurement noise, for Google
Calendar and Sheets, it amounts to a -4 TPS or -5 TPS slowdown, respectively (about -13% reduction). CD has a clear
performance impact compared to vanilla due to constraint checking. For Asana and Slack, it is about -76%, for Google
Calendar and Sheets about -67%. Naturally, the combination of RAG and CD combines the performance overhead of the
two approaches. The strongest reduction can be observed for Google Sheets at roughly -93%, while Slack shows no
measurable reduction.
The impact of RAG on inference speed might be acceptable in most scenarios given the potential improvement
of generation results it offers. The only factor affecting RAG’s overhead is the truncation threshold of the retrieved
context as transformers’ inference speed is known to scale with input length (the exact scaling behavior depends on the
14The highest recorded standard deviation is 0.6s on a 30s run; most are much lower. The complete measurements are available in our artifact.
Manuscript submitted to ACM

Mitigating Errors in LLM-Generated Web API Invocations 21
used attention mechanism and is typically between linear and quadratic). The impact of CD on inference speed, on the
other hand, highly depends on the CD implementation. The prototype implementation used for our study (for reasons
outlined in Section 4) is clearly not optimized for performance and unsuitable for use cases with real-time requirements.
However, existing works show that CD can be implemented in highly efficient ways [ 53]. If attention masks can be
fully or partially precomputed (which is the case for our constraints), CD can run at near-zero overhead [7, 48].
7 Discussion and Limitations
This section discusses practical implications of our results—including implementation challenges of RAG and CD—and
limitations that should be considered when interpreting the findings.
Practical Implications.The conducted experiments demonstrate that retrieval-augmented generation, constrained
decoding, or both, are effective in reducing or eliminating errors and hallucinations of LLMs in writing web API
integration code. In direct comparison, constrained decoding offers more consistent gains than RAG and provides hard
guarantees on specification-compliance, while also avoiding prompt bloat [ 12]. Regardless of which approach is chosen,
it is advisable to validate that the model in use actually benefits from it.
We were unable to identify meaningful patterns relating model family or size to the degree of benefit from RAG or
CD. The correlation between benchmark performance and model size is, for instance, only weak. Manual inspection of
generated code suggests that the main reason is the highly idiosyncratic behavior of individual models—even within
the same family. Code Llama (34B) is a prime example: it frequently inserts placeholders such as <body> , producing
syntactically broken code, while its smaller and larger siblings do not exhibit this behavior.
On the implementation side, we experienced different challenges for RAG and CD.
RAG benefits from a production-grade ecosystem of components, such as vector databases and embedding models,
that can be readily built upon. Its main challenge lies in the degrees of freedom at each stage of a RAG pipeline (e.g.,
preprocessing, chunking, embedding, retrieval, reranking, formatting, etc.). Even when established best practices are
followed, many parameters remain to be tuned: chunk size, overlap, number of retrieved chunks, truncation threshold,
and others. This creates a risk of overfitting the pipeline to certain models, which is especially problematic in a
benchmarking context where all models must be treated equally.
CD’s main challenge lies in constraint design. Care must be taken not to over- or under-constrain a model. In addition,
certain rules are difficult or impossible to express with regexes (or grammars), requiring workarounds or approximations.
Regarding the completion engine, we implemented a custom one to be able to explore the orchestration of multiple
constraint rulesets and to experiment with features of extended regular expressions not supported by off-the-shelf
completion engines (cf. Section 4). However, the complexity of the constraint checker is negligible compared to that of
the constraint generator.
Limitations.Our implementations of RAG and CD require the user to provide the name and specification of the
API, for which invocations are to be generated. This limitation could be addressed by automatically inferring the API
referred to in a task description and retrieving the corresponding specification. However, our work targets use cases
in the API integrations industry, where usually the customer knows which APIs they want to be integrated, and the
developer only has to decide which endpoints to use in order to connect the APIs in the desired way. In this scenario,
having to provide the correct API specification to our system is not a real limitation. Our work is not intended for
highly exploratory development scenarios, but future work could fill this gap.
Manuscript submitted to ACM

22 Maninger et al.
Both RAG and, to a larger extent, CD lead to a small increase in non-executable implementation. Since WAPIIBench’s
evaluation pipeline is execution-based, non-executable codes cannot be properly evaluated and do not contribute to
correctness and hallucination metrics. To provide a complementary view on the evaluation results, we calculate a
second variant of each metric after filtering out any non-executable codes—these metrics are presented in the appendix.
However, for practical applications like IDEs, sporadic cases of non-executable codes are no big concern, as code
suggestions can be generated in the background and discarded or regenerated if they fail a syntax check.
As mentioned earlier, the biggest limitation of our RAG evaluation is the large design space for the retriever
implementation. While we tried to make reasonable design decisions and to validate those decisions with preliminary
experiments, any change to the RAG pipeline will inevitably influence the evaluation results in a non-uniform way across
models. Another minor limitation of our RAG implementation is that during preprocessing of OpenAPI specifications,
references in are only resolved up to a certain level to avoid inflating the size of individual chunks. However, this also
means that the retriever will be less useful for tasks requiring deeply nested parameters.
The limitations of our CD implementation revolve around what structures the constraints do and do not permit.
There is a natural tradeoff between allowing everything that is correct and preventing everything that is incorrect. In
consequence, effective constraints will often limit the model’s flexibility at the same time. For example, to be able to
effectively guide the model through generating a syntactically correct and hallucination-free web API invocation, we
constrain the Axios call syntax to the one shown in Listing 1. However, some models may prefer a different syntax
(e.g., axios.request(<url>, <method>, <config>) ) and perform better when given the liberty to use it. Another
limitation is that our constraints act only locally from the beginning of an API invocation to its end. When (instead of
literals) variables are used as parameter values, we have no control over such a variable’s value or data type.
The main limitation of the new real-world dataset is its small size due to the high manual effort involved in its
creation. However, WAPIIBench’s extensibility makes it easy for everyone to contribute tasks, so the dataset may be
expanded in the future. The newly implemented support for per-task starter codes is limited to program prefixes, so
fill-in-the-middle generation is not possible. Moreover, variable or constant definitions within the starter code and
that are used for execution-based evaluation must be primitive data types, objects, or arrays—it is not possible to use
imported classes or functions.
Despite these limitations, the consistent empirical trends across 24 models, two datasets, two starter code setups, and
four generation settings provide a solid foundation for the practical adoption and further refinement of RAG and CD in
web API invocation scenarios.
8 Related Work
Differentiation of API Invocation Settings.API invocation has been studied in several settings that differ substantially
from the one considered in this work. We distinguish five categories: (1)general web APIs(our setting), (2)domain-specific
web APIs, (3)SDK-wrapped web APIs, (4)local function APIs, and (5)tool APIs for AI agents.
Our work considers general RESTful web APIs described by OpenAPI, where code must explicitly construct complete
HTTP requests. Compared to local function calls, this requires identifying operations by HTTP method and endpoint,
configuring multiple request components (path, query, headers, and body), and handling externally documented
interfaces. We are not aware of other works besides WAPIIBench [ 25] that systematically analyze the capabilities of
LLMs in this specific setting.
Prior work has largely focused on different settings. Domain-specific web APIs target constrained ecosystems such
as OData [ 17,45,46], making the task considerably narrower than arbitrary OpenAPI services. SDK-based approaches
Manuscript submitted to ACM

Mitigating Errors in LLM-Generated Web API Invocations 23
replace HTTP requests with library-specific abstractions [ 19,41], shifting the problem toward SDK usage rather than
direct API invocation. Research on local function calls [ 10,11,55] considers a much simpler interface consisting of a
function name and a single argument list. Finally, tool-use benchmarks for LLM agents [ 21,35,39,43,54] typically
expose function-like interfaces optimized for agent interaction rather than production-ready HTTP integration code.
Consequently, techniques and evaluation results from these settings are not directly transferable to general web API
invocation.
Retrieval-Augmented Generation.Retrieval augmented generation (RAG) [ 20] is commonly used when a model’s
memorized knowledge is insufficient to solve a given task. This includes many applications in code generation. To
improve correctness in general code generation, either similar code examples are retrieved [ 24,56] or excerpts from
software documentation [ 58]. Some RAG approaches specifically target local function API invocations, retrieving
examples [ 3] or documentation [ 30,55]. Work on web APIs is limited to SDK wrappers [ 19]. To our knowledge, we
are the first to study RAG in the context of general web API invocations as defined at the beginning of this section.
Our work falls into the category of retrieval from documentation, concretely OpenAPI specification documents. In this
setting, we are facing some special challenges as chunks from OpenAPI specifications must be carefully crafted (i.e.,
scoping by endpoint, resolution of references, and inlining of certain global definitions) to preserve crucial context.
Moreover, unlike other works, we design a custom format for the retriever output to optimize token usage and LLM
comprehension.
Constrained Decoding.General-purpose constrained decoding frameworks for code generation [ 2,7,48,53] primar-
ily enforce syntactic validity using regular expressions or context-free grammars. More recent work has extended
constrained decoding to semantic correctness, for example by incorporating database schemas [ 40], domain-specific
semantic rules [33], or programming language type systems [27].
Another line of work addresses hallucinated function or method invocations by constraining generation based on
information available in the code base, such as language servers or mined program dependencies [ 1,3,51]. While
effective for local APIs, these approaches cannot readily be applied to web APIs, whose interfaces are specified externally
and whose invocations are structurally more complex, requiring the generation of complete HTTP requests.
Constrained decoding has also been applied to tool use in AI agents [ 50,57], where tools are invoked through
comparatively simple JSON or Python-based interfaces. In contrast, our work targets general web APIs described by
OpenAPI specifications. To our knowledge, we are the first to explore constrained decoding in this setting, introducing
constraints that capture the structure of web API invocations and exploit externally provided API specifications to
prevent hallucinations.
Alternative Approaches.Besides retrieval-augmented generation and constrained decoding, several other approaches
have been proposed to mitigate problems LLMs face when generating API invocations. Most of these works focus
on local libraries (primarily in Python), but some also consider web APIs, even though they are not as complex
as the ones targeted by our work (cf. discussion at the beginning of this section). These approaches are primarily
fine-tuning [ 10,30,55] and iterative refinement [ 6,11,56] to reduce hallucinations and improve correctness. All
these approaches come with their own advantages and disadvantages: RAG is well established and relatively easy to
implement but risks prompt bloat [ 12]. Constrained decoding is the only approach to offer guarantees but is relatively
inflexible and, depending on the implementation, adds inference-time overhead. Fine-tuning is arguably the most
natural way to improve a model but has the highest computational cost and its training data is prone to becoming
Manuscript submitted to ACM

24 Maninger et al.
outdated. Iterative refinement might be more reliable than RAG but can be expensive in terms of token consumption. It
is worth exploring all these approaches to assess their suitability for web API invocations. We argue that they should
be seen as complementary rather than alternative.
9 Conclusion and Future Work
In this paper, we examined approaches to address the challenges faced by large language models when generating web
API invocation code: retrieval-augmented generation (RAG), constrained decoding (CD), and a combination thereof.
To this end, we implemented a retriever that processes OpenAPI specifications and stores, retrieves, and compactly
formats endpoint-level chunks that can be injected into a model’s prompt. Furthermore, we developed a custom
translation algorithm that derives constraints, represented as regular expressions, from OpenAPI specifications. Both
implementations effectively manage the complexity and scale of real-world APIs.
To evaluate the approaches we use the original synthetic dataset of the WAPIIBench benchmark [ 25] and a new
dataset with tasks manually derived from real-world GitHub repositories that we introduce in this paper. Our evaluation
results demonstrate that both RAG and CD can reduce hallucinated endpoints and arguments and improve the overall
correctness of the generated web API invocations. However, the potential benefit of RAG is much more model- and
task-dependent than that of CD. Moreover, only CD can guarantee the absence of hallucinations. The combination of
RAG and CD can further boost correctness for some models but degrades correctness for others.
Overall, our work underscores the importance of integrating quality assurance techniques into AI-driven code
generation workflows as a practical path toward correctness and reliability.
Looking ahead, we plan to extend RAG and CD to further code generation tasks and to broaden the quality criteria
they target beyond correctness and specification-compliance, for instance, to security properties. At the same time,
our benchmark lays the foundation for future work to explore other approaches to improving quality of LLM-based
code generation. Although our evaluation pipeline is specialized for JavaScript and Axios, the underlying datasets
and methodology are language- and library-agnostic and can be transferred to other ecosystems. WAPIIBench is also
designed for extensibility: new datasets can be added, and existing ones expanded to cover additional APIs or completion
scenarios with minimal effort.
Acknowledgments
This work was funded by the Deutsche Forschungsgemeinschaft (DFG, German Research Foundation) under Germany’s
Excellence Strategy (EXC-3057/1Reasonable Artificial Intelligence, Project No. 533677015), the Hessian Ministry of
Higher Education, Research, Science and the Arts within the cluster projectThe Third Wave of Artificial Intelligence
(3AI), by the National Research Center for Applied Cybersecurity ATHENE within the projectFoundational Models for
Secure Software Development, and by the LOEWE initiative (Hesse, Germany) [LOEWE/4a//519/05/00.002(0013)/95].
References
[1]Lakshya A. Agrawal, Aditya Kanade, Navin Goyal, Shuvendu K. Lahiri, and Sriram K. Rajamani. 2023. Monitor-Guided Decoding of Code LMs with
Static Analysis of Repository Context. InAdvances in Neural Information Processing Systems 36: Annual Conference on Neural Information Processing
Systems 2023, NeurIPS 2023, New Orleans, LA, USA, December 10 - 16, 2023, Alice Oh, Tristan Naumann, Amir Globerson, Kate Saenko, Moritz Hardt,
and Sergey Levine (Eds.). http://papers.nips.cc/paper_files/paper/2023/hash/662b1774ba8845fc1fa3d1fc0177ceeb-Abstract-Conference.html
[2]Luca Beurer-Kellner, Marc Fischer, and Martin T. Vechev. 2024. Guiding LLMs The Right Way: Fast, Non-Invasive Constrained Generation. InForty-
first International Conference on Machine Learning, ICML 2024, Vienna, Austria, July 21-27, 2024. OpenReview.net. https://openreview .net/forum?id =
pXaEYzrFae
Manuscript submitted to ACM

Mitigating Errors in LLM-Generated Web API Invocations 25
[3]Yujia Chen, Mingyu Chen, Cuiyun Gao, Zhihan Jiang, Zhongqi Li, et al .2025. Towards Mitigating API Hallucination in Code Generated by LLMs
with Hierarchical Dependency Aware. InProceedings of the 33rd ACM International Conference on the Foundations of Software Engineering, FSE
Companion 2025, Clarion Hotel Trondheim, Trondheim, Norway, June 23-28, 2025, Leonardo Montecchi, Jingyue Li, Denys Poshyvanyk, and Dongmei
Zhang (Eds.). ACM, 468–479. doi:10.1145/3696630.3728569
[4]Zheyuan Kevin Cui, Mert Demirer, Sonia Jaffe, Leon Musolff, Sida Peng, et al .2025. The effects of generative AI on high-skilled work: Evidence from
three field experiments with software developers.Available at SSRN 4945566(2025). doi:10.2139/ssrn.4945566
[5]Daniel Deutsch, Shyam Upadhyay, and Dan Roth. 2019. A General-Purpose Algorithm for Constrained Sequential Inference. InProceedings of
the 23rd Conference on Computational Natural Language Learning, CoNLL 2019, Hong Kong, China, November 3-4, 2019, Mohit Bansal and Aline
Villavicencio (Eds.). Association for Computational Linguistics, 482–492. doi:10.18653/V1/K19-1045
[6]Hanxing Ding, Shuchang Tao, Liang Pang, Zihao Wei, Jinyang Gao, et al .2025. ToolCoder: A Systematic Code-Empowered Tool Learning Framework
for Large Language Models. InProceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), ACL
2025, Vienna, Austria, July 27 - August 1, 2025, Wanxiang Che, Joyce Nabende, Ekaterina Shutova, and Mohammad Taher Pilehvar (Eds.). Association
for Computational Linguistics, 17876–17891. https://aclanthology.org/2025.acl-long.874/
[7]Yixin Dong, Charlie F. Ruan, Yaxing Cai, Ziyi Xu, Yilong Zhao, Ruihang Lai, and Tianqi Chen. 2025. XGrammar: Flexible and Efficient Structured
Generation Engine for Large Language Models. InProceedings of the Eighth Conference on Machine Learning and Systems, MLSys 2025, Santa Clara,
CA, USA, May 12-15, 2025, Matei Zaharia, Gauri Joshi, and Yingyan (Celine) Lin (Eds.). OpenReview.net/mlsys.org. https://openreview .net/forum?id =
rjQfX0YgDl
[8]Shihan Dou, Haoxiang Jia, Shenxi Wu, Huiyuan Zheng, Weikang Zhou, et al .2024. What’s Wrong with Your Code Generated by Large Language
Models? An Extensive Study.CoRRabs/2407.06153 (2024). doi:10.48550/ARXIV.2407.06153 arXiv: 2407.06153.
[9]Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, et al .2024. The Llama 3 Herd of Models.CoRR
abs/2407.21783 (2024). doi:10.48550/ARXIV.2407.21783 arXiv: 2407.21783.
[10] Sujan Dutta, Sayantan Mahinder, Raviteja Anantha, and Bortik Bandyopadhyay. 2024. Applying RLAIF for Code Generation with API-usage in
Lightweight LLMs.CoRRabs/2406.20060 (2024). doi:10.48550/ARXIV.2406.20060 arXiv: 2406.20060.
[11] Aryaz Eghbali and Michael Pradel. 2024. De-Hallucinator: Iterative Grounding for LLM-Based Code Completion.CoRRabs/2401.01701 (2024).
doi:10.48550/ARXIV.2401.01701 arXiv: 2401.01701.
[12] Tiantian Gan and Qiyao Sun. 2025. RAG-MCP: Mitigating Prompt Bloat in LLM Tool Selection via Retrieval-Augmented Generation.CoRR
abs/2505.03275 (2025). doi:10.48550/ARXIV.2505.03275 arXiv: 2505.03275.
[13] GitClear 2025.AI Copilot Code Quality: Evaluating 2024’s Increased Defect Rate with Data. GitClear. Retrieved 2025-05-17 from https:
//www.gitclear.com/ai_assistant_code_quality_2025_research
[14] Sacha Greif and Eric Burel. 2022.State of JavaScript 2022. Devographics. Retrieved 2024-07-30 from https://2022.stateofjs.com/en-US/
[15] Daya Guo, Qihao Zhu, Dejian Yang, Zhenda Xie, Kai Dong, et al .2024. DeepSeek-Coder: When the Large Language Model Meets Programming -
The Rise of Code Intelligence.CoRRabs/2401.14196 (2024). doi:10.48550/ARXIV.2401.14196 arXiv: 2401.14196.
[16] Chris Hokamp and Qun Liu. 2017. Lexically Constrained Decoding for Sequence Generation Using Grid Beam Search. InProceedings of the 55th
Annual Meeting of the Association for Computational Linguistics, ACL 2017, Vancouver, Canada, July 30 - August 4, Volume 1: Long Papers, Regina
Barzilay and Min-Yen Kan (Eds.). Association for Computational Linguistics, 1535–1546. doi:10.18653/v1/P17-1141
[17] Saghar Hosseini, Ahmed Hassan Awadallah, and Yu Su. 2021. Compositional Generalization for Natural Language Interfaces to Web APIs.CoRR
abs/2112.05209 (2021). https://arxiv.org/abs/2112.05209 arXiv: 2112.05209.
[18] Binyuan Hui, Jian Yang, Zeyu Cui, Jiaxi Yang, Dayiheng Liu, et al .2024. Qwen2.5-Coder Technical Report.CoRRabs/2409.12186 (2024). doi:10 .48550/
ARXIV.2409.12186 arXiv: 2409.12186.
[19] Nihal Jain, Robert Kwiatkowski, Baishakhi Ray, Murali Krishna Ramanathan, and Varun Kumar. 2025. On Mitigating Code LLM Hallucinations with
API Documentation. In47th IEEE/ACM International Conference on Software Engineering: Software Engineering in Practice, SEIP@ICSE 2025, Ottawa,
ON, Canada, April 27 - May 3, 2025. IEEE, 237–248. doi:10.1109/ICSE-SEIP66354.2025.00027
[20] Patrick S. H. Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, et al .2020. Retrieval-Augmented Generation for Knowledge-
Intensive NLP Tasks. InAdvances in Neural Information Processing Systems 33: Annual Conference on Neural Information Processing Systems 2020,
NeurIPS 2020, December 6-12, 2020, virtual, Hugo Larochelle, Marc’Aurelio Ranzato, Raia Hadsell, Maria-Florina Balcan, and Hsuan-Tien Lin (Eds.).
https://proceedings.neurips.cc/paper/2020/hash/6b493230205f780e1bc26945df7481e5-Abstract.html
[21] Minghao Li, Yingxiu Zhao, Bowen Yu, Feifan Song, Hangyu Li, et al .2023. API-Bank: A Comprehensive Benchmark for Tool-Augmented LLMs.
InProceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, EMNLP 2023, Singapore, December 6-10, 2023, Houda
Bouamor, Juan Pino, and Kalika Bali (Eds.). Association for Computational Linguistics, 3102–3116. doi:10.18653/V1/2023.EMNLP-MAIN.187
[22] Raymond Li, Loubna Ben Allal, Yangtian Zi, Niklas Muennighoff, Denis Kocetkov, et al .2023. StarCoder: may the source be with you!Trans. Mach.
Learn. Res.2023 (2023). https://openreview.net/forum?id=KoFOg41haE
[23] Anton Lozhkov, Raymond Li, Loubna Ben Allal, Federico Cassano, Joel Lamy-Poirier, et al .2024. StarCoder 2 and The Stack v2: The Next Generation.
CoRRabs/2402.19173 (2024). doi:10.48550/ARXIV.2402.19173 arXiv: 2402.19173.
[24] Shuai Lu, Nan Duan, Hojae Han, Daya Guo, Seung-won Hwang, and Alexey Svyatkovskiy. 2022. ReACC: A Retrieval-Augmented Code Completion
Framework. InProceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), ACL 2022, Dublin,
Ireland, May 22-27, 2022, Smaranda Muresan, Preslav Nakov, and Aline Villavicencio (Eds.). Association for Computational Linguistics, 6227–6240.
Manuscript submitted to ACM

26 Maninger et al.
doi:10.18653/v1/2022.acl-long.431
[25] Daniel Maninger, Leon Chemnitz, Amir Molzam Sharifloo, Jannis Brugger, and Mira Mezini. 2025. Benchmarking Web API Integration Code
Generation. In2nd IEEE/ACM International Conference on AI-powered Software, AIware 2025, Seoul, Republic of Korea, November 19-20, 2025. IEEE,
240–248. doi:10.1109/AIWARE69974.2025.00034
[26] MarketsandMarkets Research Private Ltd. 2024.API Management Market by Platform, Service – Global Forecast to 2029. MarketsandMarkets Research
Private Ltd. Retrieved 2025-05-18 from https://www.marketsandmarkets.com/Market-Reports/api-management-market-178266736.html
[27] Niels Mündler, Jingxuan He, Hao Wang, Koushik Sen, Dawn Song, et al .2025. Type-Constrained Code Generation with Language Models.Proc.
ACM Program. Lang.9, PLDI (2025), 601–626. doi:10.1145/3729274
[28] OpenAI 2024.GPT -4o mini: advancing cost-efficient intelligence. OpenAI. Retrieved 2026-06-29 from https://openai .com/index/gpt-4o-mini-
advancing-cost-efficient-intelligence/
[29] OpenAI 2024.Hello GPT-4o. OpenAI. Retrieved 2026-06-29 from https://openai.com/index/hello-gpt-4o/
[30] Shishir G. Patil, Tianjun Zhang, Xin Wang, and Joseph E. Gonzalez. 2024. Gorilla: Large Language Model Connected with Massive APIs. InAdvances
in Neural Information Processing Systems 38: Annual Conference on Neural Information Processing Systems 2024, NeurIPS 2024, Vancouver, BC, Canada,
December 10 - 15, 2024, Amir Globersons, Lester Mackey, Danielle Belgrave, Angela Fan, Ulrich Paquet, Jakub M. Tomczak, and Cheng Zhang (Eds.).
https://proceedings.neurips.cc//paper_files/paper/2024/hash/e4c61f578ff07830f5c37378dd3ecb0d-Abstract-Conference.html
[31] Hammond Pearce, Baleegh Ahmad, Benjamin Tan, Brendan Dolan-Gavitt, and Ramesh Karri. 2022. Asleep at the Keyboard? Assessing the Security
of GitHub Copilot’s Code Contributions. In43rd IEEE Symposium on Security and Privacy, SP 2022, San Francisco, CA, USA, May 22-26, 2022. IEEE,
754–768. doi:10.1109/SP46214.2022.9833571
[32] Neil Perry, Megha Srivastava, Deepak Kumar, and Dan Boneh. 2023. Do Users Write More Insecure Code with AI Assistants?. InProceedings of the
2023 ACM SIGSAC Conference on Computer and Communications Security, CCS 2023, Copenhagen, Denmark, November 26-30, 2023, Weizhi Meng,
Christian Damsgaard Jensen, Cas Cremers, and Engin Kirda (Eds.). ACM, 2785–2799. doi:10.1145/3576915.3623157
[33] Gabriel Poesia, Alex Polozov, Vu Le, Ashish Tiwari, Gustavo Soares, et al .2022. Synchromesh: Reliable Code Generation from Pre-trained
Language Models. InThe Tenth International Conference on Learning Representations, ICLR 2022, Virtual Event, April 25-29, 2022. OpenReview.net.
https://openreview.net/forum?id=KmtVD97J43e
[34] Postman, Inc. 2024.2024 State of the API Report. Postman, Inc. Retrieved 2025-05-17 from https://www.postman.com/state-of-api/2024/
[35] Yujia Qin, Shihao Liang, Yining Ye, Kunlun Zhu, Lan Yan, et al .2024. ToolLLM: Facilitating Large Language Models to Master 16000+ Real-
world APIs. InThe Twelfth International Conference on Learning Representations, ICLR 2024, Vienna, Austria, May 7-11, 2024. OpenReview.net.
https://openreview.net/forum?id=dHng2O0Jjr
[36] Irum Rauf, Elena Troubitsyna, and Ivan Porres. 2019. A systematic mapping study of API usability evaluation methods.Comput. Sci. Rev.33 (2019),
49–68. doi:10.1016/J.COSREV.2019.05.001
[37] Martin P. Robillard. 2009. What Makes APIs Hard to Learn? Answers from Developers.IEEE Softw.26, 6 (2009), 27–34. doi:10.1109/MS.2009.193
[38] Baptiste Rozière, Jonas Gehring, Fabian Gloeckle, Sten Sootla, Itai Gat, et al .2023. Code Llama: Open Foundation Models for Code.CoRR
abs/2308.12950 (2023). doi:10.48550/ARXIV.2308.12950 arXiv: 2308.12950.
[39] Timo Schick, Jane Dwivedi-Yu, Roberto Dessì, Roberta Raileanu, Maria Lomeli, et al .2023. Toolformer: Language Models Can Teach Themselves to
Use Tools. InAdvances in Neural Information Processing Systems 36: Annual Conference on Neural Information Processing Systems 2023, NeurIPS 2023,
New Orleans, LA, USA, December 10 - 16, 2023, Alice Oh, Tristan Naumann, Amir Globerson, Kate Saenko, Moritz Hardt, and Sergey Levine (Eds.).
http://papers.nips.cc/paper_files/paper/2023/hash/d842425e4bf79ba039352da0f658a906-Abstract-Conference.html
[40] Torsten Scholak, Nathan Schucher, and Dzmitry Bahdanau. 2021. PICARD: Parsing Incrementally for Constrained Auto-Regressive Decoding from
Language Models. InProceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, EMNLP 2021, Virtual Event / Punta
Cana, Dominican Republic, 7-11 November, 2021, Marie-Francine Moens, Xuanjing Huang, Lucia Specia, and Scott Wen-tau Yih (Eds.). Association for
Computational Linguistics, 9895–9901. doi:10.18653/v1/2021.emnlp-main.779
[41] Raphael Shu, Elman Mansimov, Tamer Alkhouli, Nikolaos Pappas, Salvatore Romeo, et al .2022. Dialog2API: Task-Oriented Dialogue with API
Description and Example Programs.CoRRabs/2212.09946 (2022). doi:10.48550/ARXIV.2212.09946 arXiv: 2212.09946.
[42] SmartBear Software 2020.The State of API 2020 Report. SmartBear Software. Retrieved 2024-07-29 from https://smartbear .com/resources/ebooks/the-
state-of-api-2020-report/
[43] Yifan Song, Weimin Xiong, Dawei Zhu, Cheng Li, Ke Wang, et al .2023. RestGPT: Connecting Large Language Models with Real-World Applications
via RESTful APIs.CoRRabs/2306.06624 (2023). doi:10.48550/ARXIV.2306.06624 arXiv: 2306.06624.
[44] Statista, Inc. 2024.Most used programming languages among developers worldwide as of 2024. Statista, Inc. Retrieved 2024-09-08 from https:
//www.statista.com/statistics/793628/worldwide-developer-survey-most-used-languages/
[45] Yu Su, Ahmed Hassan Awadallah, Madian Khabsa, Patrick Pantel, Michael Gamon, et al .2017. Building Natural Language Interfaces to Web APIs. In
Proceedings of the 2017 ACM on Conference on Information and Knowledge Management, CIKM 2017, Singapore, November 06 - 10, 2017, Ee-Peng Lim,
Marianne Winslett, Mark Sanderson, Ada Wai-Chee Fu, Jimeng Sun, J. Shane Culpepper, Eric Lo, Joyce C. Ho, Debora Donato, Rakesh Agrawal,
Yu Zheng, Carlos Castillo, Aixin Sun, Vincent S. Tseng, and Chenliang Li (Eds.). ACM, 177–186. doi:10.1145/3132847.3133009
[46] Yu Su, Ahmed Hassan Awadallah, Miaosen Wang, and Ryen W. White. 2018. Natural Language Interfaces with Fine-Grained User Interaction:
A Case Study on Web APIs. InThe 41st International ACM SIGIR Conference on Research & Development in Information Retrieval, SIGIR 2018, Ann
Arbor, MI, USA, July 08-12, 2018, Kevyn Collins-Thompson, Qiaozhu Mei, Brian D. Davison, Yiqun Liu, and Emine Yilmaz (Eds.). ACM, 855–864.
Manuscript submitted to ACM

Mitigating Errors in LLM-Generated Web API Invocations 27
doi:10.1145/3209978.3210013
[47] Florian Tambon, Arghavan Moradi Dakhel, Amin Nikanjam, Foutse Khomh, and others Desmarais. 2025. Bugs in large language models generated
code: an empirical study.Empir. Softw. Eng.30, 3 (2025), 65. doi:10.1007/S10664-025-10614-4
[48] Shubham Ugare, Tarun Suresh, Hangoo Kang, Sasa Misailovic, and Gagandeep Singh. 2025. SynCode: LLM Generation with Grammar Augmentation.
Trans. Mach. Learn. Res.2025 (2025). https://openreview.net/forum?id=HiUZtgAPoH
[49] Yue Wang, Hung Le, Akhilesh Gotmare, Nghi D. Q. Bui, Junnan Li, et al .2023. CodeT5+: Open Code Large Language Models for Code Understanding
and Generation. InProceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, EMNLP 2023, Singapore, December 6-10, 2023,
Houda Bouamor, Juan Pino, and Kalika Bali (Eds.). Association for Computational Linguistics, 1069–1088. doi:10 .18653/V1/2023 .EMNLP-MAIN .68
[50] Zhuoer Wang, Leonardo F. R. Ribeiro, Alexandros Papangelis, Rohan Mukherjee, Tzu-Yen Wang, et al .2024. FANTAstic SEquences and Where to
Find Them: Faithful and Efficient API Call Generation through State-tracked Constrained Decoding and Reranking. InFindings of the Association for
Computational Linguistics: EMNLP 2024, Miami, Florida, USA, November 12-16, 2024, Yaser Al-Onaizan, Mohit Bansal, and Yun-Nung Chen (Eds.).
Association for Computational Linguistics, 6179–6191. https://aclanthology.org/2024.findings-emnlp.359
[51] Yuxiang Wei, Chunqiu Steven Xia, and Lingming Zhang. 2023. Copiloting the Copilots: Fusing Large Language Models with Completion Engines for
Automated Program Repair. InProceedings of the 31st ACM Joint European Software Engineering Conference and Symposium on the Foundations of
Software Engineering, ESEC/FSE 2023, San Francisco, CA, USA, December 3-9, 2023, Satish Chandra, Kelly Blincoe, and Paolo Tonella (Eds.). ACM,
172–184. doi:10.1145/3611643.3616271
[52] Anna-Katharina Wickert, Lars Baumgärtner, Florian Breitfelder, and Mira Mezini. 2021. Python Crypto Misuses in the Wild. InESEM ’21: ACM / IEEE
International Symposium on Empirical Software Engineering and Measurement, Bari, Italy, October 11-15, 2021, Filippo Lanubile, Marcos Kalinowski,
and Maria Teresa Baldassarre (Eds.). ACM, 31:1–31:6. doi:10.1145/3475716.3484195
[53] Brandon T. Willard and Rémi Louf. 2023. Efficient Guided Generation for Large Language Models.CoRRabs/2307.09702 (2023). doi:10 .48550/
arXiv.2307.09702 arXiv: 2307.09702.
[54] Asaf Yehudai, Lilach Eden, Alan Li, Guy Uziel, Yilun Zhao, et al .2025. Survey on Evaluation of LLM-based Agents.CoRRabs/2503.16416 (2025).
doi:10.48550/ARXIV.2503.16416 arXiv: 2503.16416.
[55] Daoguang Zan, Bei Chen, Yongshun Gong, Junzhi Cao, Fengji Zhang, et al .2025. Private-library-oriented code generation with large language
models.Knowl. Based Syst.326 (2025), 113934. doi:10.1016/J.KNOSYS.2025.113934
[56] Fengji Zhang, Bei Chen, Yue Zhang, Jacky Keung, Jin Liu, Daoguang Zan, Yi Mao, Jian-Guang Lou, and Weizhu Chen. 2023. RepoCoder: Repository-
Level Code Completion Through Iterative Retrieval and Generation. InProceedings of the 2023 Conference on Empirical Methods in Natural Language
Processing, EMNLP 2023, Singapore, December 6-10, 2023, Houda Bouamor, Juan Pino, and Kalika Bali (Eds.). Association for Computational Linguistics,
2471–2484. doi:10.18653/V1/2023.EMNLP-MAIN.151
[57] Kexun Zhang, Hongqiao Chen, Lei Li, and William Yang Wang. 2023. Don’t Fine-Tune, Decode: Syntax Error-Free Tool Use via Constrained
Decoding.CoRRabs/2310.07075 (2023). doi:10.48550/ARXIV.2310.07075 arXiv: 2310.07075.
[58] Shuyan Zhou, Uri Alon, Frank F. Xu, Zhengbao Jiang, and Graham Neubig. 2023. DocPrompting: Generating Code by Retrieving the Docs. InThe
Eleventh International Conference on Learning Representations, ICLR 2023, Kigali, Rwanda, May 1-5, 2023. OpenReview.net. https://openreview .net/
forum?id=ZTCxT2t2Ru
[59] Terry Yue Zhuo, Xiaoning Du, Zhenchang Xing, Jiamou Sun, Haowei Quan, et al .2023. Pop Quiz! Do Pre-trained Code Models Possess Knowledge
of Correct API Names?CoRRabs/2309.07804 (2023). doi:10.48550/ARXIV.2309.07804 arXiv: 2309.07804.
Manuscript submitted to ACM

28 Maninger et al.
A Data Availability
WAPIIBench is available on GitHub15. In addition, we provide all model-generated codes and evaluation results in our
artifact16.
B Constrained Decoding Details
We implemented our constrained decoding engine as an extension of the Hugging Face Transformers library (cf.
Appendix C) by providing a LogitsProcessor subclass that manages the whole constrained decoding process. The
implementation is model-agnostic and compatible with all encoder-decoder and decoder-only architectures available
on Hugging Face.
Our constraints ensure for each argument that its name is defined in the specification for the location it appears in,
and that the associated expression is either a valid variable name or a value that matches the specified schema and data
type. Complex data structures such as arrays and objects are constrained deeply, i.e., constraints are constructed for
their members as well, recursively. Moreover, we ensure that an argument is not passed more than once in the same
location and that required arguments are indeed passed. While the general call syntax is predetermined in favor of
the most common way to use Axios (even though alternative syntaxes exist), whitespace is handled flexibly by our
constraints to permit different code formattings.
When using greedy decoding, where always the most likely next token is generated, we optimize the search for
allowed next tokens by checking candidate tokens in order of their probability and stop once the first matching token is
found. Further performance optimizations are out of scope for this work, as related work has already proposed effective
measures to improve it (cf. Section 2.4), which could be incorporated into our implementation.
Our approach requires the user to select the API he or she wants to use and provide its specification. For an automated
API selection, more research on processing the task description and classifying the implied API would be necessary.
The evaluation in this work was limited to one API at a time. Enforcing constraints for multiple different APIs at once
requires only implementing OR-semantics in addition to the current AND-semantics for generation rules.
The generation rules we implemented are activated when axios. is generated and deactivated when the end of
a statement (e.g., by );or).then( ) is generated. This makes it feasible to generate whole programs while only
constraining Axios invocations. However, both the start and the stop conditions are heuristics and may fail, e.g., if a
matching string is found in a comment or nested expression. Improving the detection of start and stop conditions is
only possible to a certain extent due to the fundamental limitations of regular expressions (e.g., it is impossible to decide
if parentheses are balanced). When generating whole programs, we also cannot constrain the data types and values of
variables that are declared before and used within the Axios invocation, as it cannot be foreseen which variable will be
used for what (and if it is relevant to the API invocation in the first place). More research on this problem is needed for
constrained decoding approaches in general.
Our constraints are designed to match a code snippet if it contains an Axios invocation that respects JavaScript
syntax and API usage constraints from an OpenAPI specification. However, due to some simplifications we made to
keep the complexity of the generated regexes in bounds, the opposite is not true, i.e., there are syntactically invalid
Axios invocations that are matched by our constraints. One example are single, double, and backtick quotation marks,
for which we do not always ensure that the opening and the closing one match. However, we did not encounter any
cases where a model made such a mistake. Hence, we did not consider it worthwhile to implement such constraints,
15https://github.com/stg-tud/WAPIIBench
16https://doi.org/10.5281/zenodo.13758414
Manuscript submitted to ACM

Mitigating Errors in LLM-Generated Web API Invocations 29
which would add even more complexity to the regexes. Another OpenAPI-related example is the schema property of
any parameter, which provides additional constraints beyond its data type, e.g., that a string value should be an email
address or that an integer value should be in the range from 0 to 100. Although it would be feasible to respect such
schema constraints (at least approximately), we did not implement them so far, since by constraining the data type, we
already cover the majority of parameter-value-related mistakes. Finally, OpenAPI specifications sometimes contain
usage constraints that are only explained in a parameter’s description property, which contains a free form textual
description of an operation. For instance, the maxResults parameter in the Google Calendar API contains a passage
“The page size can never be larger than 250 entries”, but this information is not reflected in the parameter’s schema
(even though schema s support specifying maximum and minimum values for a parameter). More research is needed
to enable extracting such constraints from description s and using them for constrained decoding and evaluation
purposes.
For API parameters that are nested objects or arrays, some simplifications were necessary. Since OpenAPI permits
recursive schemas, we prevent an infinite regress while generating constraints from self-recursive schemas by limiting
the recursion depth to three. This depth is sufficient for all samples in our dataset but could be adapted if necessary. For
arrays, we had to set the maximum number of elements to five, due to an incompatibility with the way we prevent
duplicate arguments. This is, however, no fundamental limitation. Again, the chosen maximum is sufficient for our
dataset, and we argue that arrays of more than five elements should anyway not be defined inline in most cases.
C Technologies
We implemented WAPIIBench using the following technologies:
•OpenAPI
•OpenAPI 3 parser
•Axios
•axios-mock-adapter
•Hugging Face Transformers
•LangChain
•Chroma
•regex
D Models
These are the exact names of the models we evaluated:
•bigcode/starcoderbase-1b
•bigcode/starcoderbase-3b
•bigcode/starcoderbase-7b
•bigcode/starcoderbase
•bigcode/starcoder2-3b
•bigcode/starcoder2-7b
•bigcode/starcoder2-15b
•deepseek-ai/deepseek-coder-1.3b-base
•deepseek-ai/deepseek-coder-6.7b-base
Manuscript submitted to ACM

30 Maninger et al.
•deepseek-ai/deepseek-coder-33b-base
•meta-llama/CodeLlama-7b-hf
•meta-llama/CodeLlama-13b-hf
•meta-llama/CodeLlama-34b-hf
•meta-llama/CodeLlama-70b-hf
•meta-llama/Llama-3.1-8B
•meta-llama/Llama-3.1-70B
•openai/gpt-4o
•openai/gpt-4o-mini
•Qwen/Qwen2.5-Coder-0.5B
•Qwen/Qwen2.5-Coder-1.5B
•Qwen/Qwen2.5-Coder-3B
•Qwen/Qwen2.5-Coder-7B
•Qwen/Qwen2.5-Coder-14B
•Qwen/Qwen2.5-Coder-32B
•Salesforce/codet5p-6b
•Salesforce/codet5p-16b
We use only base (i.e., non-instruction-tuned) models. While instruction-tuned models tend to perform better on
coding tasks17, we considered them to be inappropriate for our setting, which is based oncode completion. Performing
code completion on instruction-tuned models leads to rather unnatural results, as the models often do not directly
return the completion and instead start with some response (“Here is your completed code ...”) followed by a code
snipped that may contain (1) only the completion, (2) the starter code followed by the completion, or (3) an arbitrarily
modified version of the starter code and corresponding completion. These factors make the model output very hard to
parse reliably. Therefore, we decided to focus our evaluation on base models.
The following hyperparameters were used:
•floating point precision = 16 bit
•# beams = 1
•temperature = 0.0
E Prompts
Listing 3 shows the prompt used in WAPIIBench’s evaluation pipeline when generating API invocation code based
on tasks from the synthetic dataset. Listing 4 shows the prompt adapted for the new real-world dataset.
F Extended Results
The full set of metrics calculated by WAPIIBench is explained in Table 4. Tables 5 to 20 provide a comprehensive
summary of these metrics for all evaluated models (cf. Appendix D) and both datasets. The best values are in bold.
Tables 21 to 23 compare the correctness rates of all models between vanilla generation and the other generation settings
(RAG, CD, and RAG + CD). Likewise, Tables 24 to 26 compare the hallucination rates.
17Cf., e.g., the EvalPlus leaderboard (https://evalplus.github.io/leaderboard.html)
Manuscript submitted to ACM

Mitigating Errors in LLM-Generated Web API Invocations 31
Listing 3. Prompt for generating API invocations used for the synthetic dataset. Words in curly braces are placeholders.
{ spec }
You are an AI programming assistant that helps users write API requests . You are given a comment that describes what
the user wants to achieve and are supposed to implement it using the Axios library in JavaScript . For this , write
a single call to Axios ( using the syntax`axios .< method >( url [, config ])`) that does exactly what was described
in the comment .
* Make sure to include all parameters in`config`that are required to solve the given task , but do not include any
unnecessary parameters .
* Insert all values directly into the place where they belong rather than using intermediate variables .
* If the API requires authentication , prefer OAuth2 over other schemes and use`<token >`as a placeholder for the
authorization token .
* If a request body requires a media type other than`text /json`, explicitly set the`Content -Type`header to the
respective type , and Axios will automatically serialize the request body accordingly .
* Always use HTTPS .
Your next task is about the { api } API . Complete the following code snippet :
```javascript
// { task }
const axios = require ('axios');
axios .{ method }('{ url }',
Listing 4. Prompt for generating API invocations used for the real-world dataset. Words in curly braces are placeholders.
{ spec }
You are an AI programming assistant that helps users write API requests . You are given a comment that describes what
the user wants to achieve and are supposed to implement it using the Axios library in JavaScript . For this , write
a single call to Axios ( using the syntax`axios .< method >( url [, config ])`) that does exactly what was described
in the comment .
* Make sure to include all parameters in`config`that are required to solve the given task , but do not include any
unnecessary parameters .
* If a request body requires a media type other than`text /json`, explicitly set the`Content -Type`header to the
respective type , and Axios will automatically serialize the request body accordingly .
* Always use HTTPS .
Your next task is about the { api } API . Complete the following code snippet :
```javascript
{ starter_code }
Note that some metrics can only be calculated either forfull completionor forargument completionand are therefore
not shown in all tables. Values marked with(t)are ratios relative to thetotal samples(395 for the synthetic dataset,
28 for the real-world dataset), while values marked with(e)are relative to the subset ofexecutable implementations18,
which varies from experiment to experiment. The complete raw data these results are based on can be found in our
artifact (cf. Appendix A). Small discrepancies to previously published results [ 25] are due to minor prompt adjustments
and improvements to the evaluation pipeline.
18To be exact, the calculation for (t) and (e) values is identical except for a filtering step to remove non-executable samples before calculating the latter.
Manuscript submitted to ACM

32 Maninger et al.
Table 4. Complete evaluation metrics for correctness and specification-compliance of API invocations
Metric Description
Executable implementations Generated code is complete and contains no syntax or runtime error
Correct implementations Generated executable code matches the ground-truth configuration
Hallucinated implementations Generated code contains at least one violation of the API specification
Correct endpoints Generated URL–method combination matches the ground-truth URL
Hallucinated endpoints Generated URL–method combination is not defined in the API specification
Correct URLs Generated URL matches the ground-truth URL
Hallucinated URLs Generated URL is not defined in the API specification
Correct methods Generated HTTP method matches the ground-truth HTTP method
Hallucinated methods Generated HTTP method is not defined for the generated URL in the API
specification
Correct argument names Generated arguments are correct
Correct argument values Generated argument values are correct
Missing arguments Expected arguments are not generated
Unexpected arguments Generated arguments are not expected
Unnecessary arguments Redundant arguments are generated for an API endpoint
Hallucinated arguments Generated arguments are not permitted for the generated API endpoint
Mean argument precision Probability that the generated arguments are correct
Mean argument recall Probability that the correct arguments are generated
Mean argument Jaccard index Overlap between generated and correct arguments
Mean argument value conditional
accuracyProbability that an argument value is correct if the argument name is correct
Total errors Any type of error prevented execution
Incomplete implementations Generated code did not contain a complete API invocation
Runtime errors Generated code produced an error when trying to execute it
Timeouts Constrained decoding was canceled because finding allowed next tokens took
too long
Unsatisfiable constraints Constrained decoding was canceled because no allowed next token existed
Manuscript submitted to ACM

Mitigating Errors in LLM-Generated Web API Invocations 33
Table 5. Complete evaluation results forfull completionwithvanillageneration on thesynthetic datasetCodeT5+ (6B)
CodeT5+ (16B)
StarCoder (1B)
StarCoder (3B)
StarCoder (7B)
StarCoder (15.5B)
StarCoder2 (3B)
StarCoder2 (7B)
StarCoder2 (15B)
DeepSeek-Coder (1.3B)
DeepSeek-Coder (6.7B)
DeepSeek-Coder (33B)
Qwen2.5-Coder (0.5B)
Qwen2.5-Coder (1.5B)
Qwen2.5-Coder (3B)
Qwen2.5-Coder (7B)
Qwen2.5-Coder (14B)
Qwen2.5-Coder (32B)
Llama 3.1 (8B)
Llama 3.1 (70B)
Code Llama (7B)
Code Llama (13B)
Code Llama (34B)
Code Llama (70B)
GPT-4o mini
GPT-4o
Executable implementations (t) 0.96 0.96 0.96 0.96 0.98 0.99 0.97 0.99 0.98 0.97 0.64 0.97 0.00 0.00 0.00 0.00 0.00 0.43 0.00 0.00 0.99 0.98 0.961.00 1.00 1.00
Correct implementations (t) 0.04 0.07 0.03 0.05 0.12 0.13 0.13 0.10 0.25 0.07 0.14 0.17 0.00 0.00 0.00 0.00 0.00 0.17 0.00 0.00 0.03 0.18 0.17 0.30 0.390.58
Correct implementations (e) 0.04 0.07 0.03 0.05 0.12 0.14 0.14 0.10 0.25 0.08 0.23 0.18 0.00 0.00 0.00 0.00 0.00 0.39 0.00 0.00 0.03 0.18 0.17 0.30 0.390.58
Correct endpoints (t) 0.11 0.25 0.21 0.33 0.42 0.49 0.37 0.42 0.56 0.29 0.33 0.60 0.00 0.00 0.00 0.00 0.00 0.27 0.00 0.00 0.40 0.50 0.51 0.59 0.490.73
Correct endpoints (e) 0.11 0.26 0.22 0.35 0.42 0.49 0.38 0.43 0.57 0.30 0.52 0.61 0.00 0.00 0.00 0.00 0.00 0.64 0.00 0.00 0.40 0.51 0.53 0.59 0.490.73
Hallucinated endpoints (t) 0.70 0.46 0.30 0.33 0.36 0.23 0.28 0.33 0.26 0.38 0.15 0.200.00 0.00 0.00 0.00 0.000.100.00 0.000.38 0.30 0.27 0.27 0.32 0.15
Hallucinated endpoints (e) 0.72 0.48 0.31 0.35 0.37 0.23 0.28 0.34 0.26 0.39 0.24 0.210.00 0.00 0.00 0.00 0.000.240.00 0.000.39 0.31 0.28 0.27 0.32 0.15
Correct URLs (t) 0.13 0.31 0.28 0.38 0.47 0.54 0.42 0.46 0.59 0.34 0.36 0.65 0.00 0.00 0.00 0.00 0.00 0.29 0.00 0.00 0.46 0.55 0.56 0.68 0.560.79
Correct URLs (e) 0.13 0.32 0.29 0.40 0.47 0.54 0.43 0.47 0.61 0.35 0.56 0.67 0.00 0.00 0.00 0.00 0.00 0.67 0.00 0.00 0.47 0.56 0.59 0.68 0.560.79
Hallucinated URLs (t) 0.66 0.38 0.24 0.25 0.29 0.18 0.21 0.29 0.21 0.33 0.13 0.140.00 0.00 0.00 0.00 0.000.070.00 0.000.28 0.23 0.22 0.17 0.24 0.09
Hallucinated URLs (e) 0.69 0.40 0.25 0.26 0.30 0.18 0.21 0.30 0.21 0.34 0.20 0.140.00 0.00 0.00 0.00 0.000.170.00 0.000.29 0.24 0.22 0.17 0.24 0.09
Correct methods (t) 0.79 0.79 0.61 0.77 0.81 0.88 0.780.910.86 0.83 0.60 0.86 0.00 0.00 0.00 0.00 0.00 0.37 0.00 0.00 0.75 0.86 0.84 0.87 0.860.91
Correct methods (e) 0.82 0.82 0.63 0.80 0.83 0.89 0.80 0.92 0.87 0.850.940.89 0.00 0.00 0.00 0.00 0.00 0.86 0.00 0.00 0.75 0.88 0.87 0.87 0.87 0.91
Hallucinated methods (t) 0.14 0.14 0.16 0.15 0.15 0.08 0.12 0.06 0.10 0.09 0.03 0.090.00 0.00 0.00 0.00 0.000.050.00 0.000.19 0.11 0.10 0.11 0.12 0.08
Hallucinated methods (e) 0.15 0.15 0.17 0.15 0.16 0.08 0.12 0.06 0.10 0.10 0.04 0.090.00 0.00 0.00 0.00 0.000.120.00 0.000.19 0.11 0.10 0.11 0.12 0.08
Correct argument names (t) 0.29 0.40 0.26 0.35 0.49 0.55 0.49 0.44 0.59 0.43 0.33 0.58 0.00 0.00 0.00 0.00 0.00 0.29 0.00 0.00 0.38 0.62 0.62 0.70 0.770.86
Correct argument names (e) 0.30 0.42 0.27 0.37 0.50 0.56 0.51 0.44 0.60 0.44 0.53 0.60 0.00 0.00 0.00 0.00 0.00 0.74 0.00 0.00 0.38 0.63 0.65 0.70 0.770.86
Correct argument values (t) 0.25 0.32 0.23 0.31 0.43 0.46 0.43 0.39 0.55 0.36 0.31 0.54 0.00 0.00 0.00 0.00 0.00 0.26 0.00 0.00 0.27 0.49 0.48 0.65 0.730.83
Correct argument values (e) 0.26 0.33 0.24 0.33 0.44 0.47 0.44 0.40 0.56 0.37 0.49 0.55 0.00 0.00 0.00 0.00 0.00 0.69 0.00 0.00 0.27 0.51 0.50 0.65 0.740.83
Missing arguments (t) 0.71 0.60 0.74 0.65 0.51 0.45 0.51 0.56 0.41 0.57 0.67 0.42 1.00 1.00 1.00 1.00 1.00 0.71 1.00 1.00 0.62 0.38 0.38 0.30 0.230.14
Missing arguments (e) 0.70 0.58 0.73 0.63 0.50 0.44 0.49 0.56 0.40 0.56 0.47 0.400.00 0.00 0.00 0.00 0.000.260.00 0.000.62 0.37 0.35 0.30 0.23 0.14
Unexpected arguments (t) 0.28 0.29 0.31 0.30 0.29 0.27 0.32 0.31 0.25 0.33 0.18 0.240.00 0.00 0.00 0.00 0.000.070.00 0.000.19 0.23 0.23 0.20 0.15 0.11
Unexpected arguments (e) 0.28 0.30 0.33 0.31 0.30 0.27 0.33 0.31 0.25 0.34 0.26 0.250.00 0.00 0.00 0.00 0.000.170.00 0.000.19 0.24 0.24 0.20 0.15 0.11
Mean argument precision (t) 0.53 0.56 0.42 0.53 0.61 0.66 0.58 0.54 0.68 0.53 0.65 0.64 0.00 0.00 0.00 0.00 0.00 0.81 0.00 0.00 0.71 0.72 0.73 0.75 0.840.88
Mean argument precision (e) 0.53 0.56 0.42 0.53 0.61 0.66 0.58 0.54 0.68 0.53 0.65 0.64 0.00 0.00 0.00 0.00 0.00 0.81 0.00 0.00 0.71 0.72 0.73 0.75 0.840.88
Mean argument recall (t) 0.32 0.43 0.26 0.36 0.49 0.55 0.49 0.43 0.59 0.44 0.34 0.56 0.00 0.00 0.00 0.00 0.00 0.33 0.00 0.00 0.40 0.63 0.64 0.70 0.790.87
Mean argument recall (e) 0.33 0.44 0.27 0.37 0.50 0.55 0.50 0.44 0.60 0.45 0.54 0.57 0.00 0.00 0.00 0.00 0.00 0.76 0.00 0.00 0.40 0.65 0.67 0.70 0.790.87
Mean arg. Jaccard index (t) 0.29 0.36 0.23 0.31 0.42 0.48 0.41 0.36 0.52 0.37 0.31 0.47 0.00 0.00 0.00 0.00 0.00 0.30 0.00 0.00 0.36 0.57 0.57 0.63 0.740.82
Mean arg. Jaccard index (e) 0.30 0.38 0.24 0.32 0.42 0.49 0.42 0.37 0.53 0.38 0.48 0.49 0.00 0.00 0.00 0.00 0.00 0.71 0.00 0.00 0.36 0.58 0.60 0.63 0.740.82
Mean arg. val. cond. acc. (t) 0.89 0.77 0.88 0.87 0.87 0.83 0.84 0.89 0.93 0.83 0.92 0.91 0.00 0.00 0.00 0.00 0.00 0.94 0.00 0.00 0.68 0.78 0.77 0.93 0.960.97
Mean arg. val. cond. acc. (e) 0.89 0.77 0.88 0.87 0.87 0.83 0.84 0.89 0.93 0.83 0.92 0.91 0.00 0.00 0.00 0.00 0.00 0.94 0.00 0.00 0.68 0.78 0.77 0.93 0.960.97
Total errors 14 14 16 15 6 3 11 4 7 10 144 11 395 395 395 395 395 226 395 395 2 9 15010
Incomplete implementations 5 9 15 15 6 3 9 4 4 7 4 6 50 0 0 0 0 0 02 3 2010
Runtime errors 7 5 10 0 0202 30 0820 0 0 0 0 0 0 06 130 0 0
Manuscript submitted to ACM

34 Maninger et al.Table 6. Complete evaluation results forargument completionwithvanillageneration on thesynthetic datasetCodeT5+ (6B)
CodeT5+ (16B)
StarCoder (1B)
StarCoder (3B)
StarCoder (7B)
StarCoder (15.5B)
StarCoder2 (3B)
StarCoder2 (7B)
StarCoder2 (15B)
DeepSeek-Coder (1.3B)
DeepSeek-Coder (6.7B)
DeepSeek-Coder (33B)
Qwen2.5-Coder (0.5B)
Qwen2.5-Coder (1.5B)
Qwen2.5-Coder (3B)
Qwen2.5-Coder (7B)
Qwen2.5-Coder (14B)
Qwen2.5-Coder (32B)
Llama 3.1 (8B)
Llama 3.1 (70B)
Code Llama (7B)
Code Llama (13B)
Code Llama (34B)
Code Llama (70B)
GPT-4o mini
GPT-4o
Executable implementations (t) 0.98 0.98 0.97 0.97 0.98 0.99 0.99 0.99 0.99 0.98 0.99 0.99 0.90 0.96 0.98 0.99 0.98 0.99 0.98 0.991.000.99 0.991.00 1.00 1.00
Correct implementations (t) 0.24 0.21 0.22 0.23 0.29 0.28 0.22 0.19 0.24 0.19 0.29 0.26 0.17 0.24 0.27 0.32 0.47 0.43 0.10 0.30 0.05 0.26 0.40 0.420.630.59
Correct implementations (e) 0.24 0.21 0.23 0.23 0.30 0.29 0.22 0.20 0.24 0.20 0.29 0.26 0.19 0.25 0.27 0.32 0.47 0.43 0.10 0.31 0.05 0.26 0.40 0.420.630.59
Hallucinated implementations (t) 0.57 0.54 0.48 0.49 0.35 0.28 0.45 0.48 0.35 0.50 0.30 0.40 0.54 0.52 0.44 0.41 0.26 0.28 0.34 0.260.190.31 0.26 0.240.190.31
Hallucinated implementations (e) 0.58 0.55 0.50 0.50 0.36 0.29 0.46 0.48 0.36 0.52 0.30 0.40 0.60 0.54 0.45 0.41 0.27 0.28 0.35 0.270.190.31 0.26 0.240.190.31
Correct argument names (t) 0.44 0.53 0.46 0.51 0.67 0.73 0.62 0.55 0.64 0.60 0.65 0.70 0.46 0.54 0.55 0.72 0.80 0.82 0.46 0.66 0.52 0.73 0.80 0.790.890.77
Correct argument names (e) 0.45 0.54 0.48 0.53 0.69 0.74 0.63 0.55 0.64 0.62 0.66 0.71 0.52 0.57 0.56 0.72 0.82 0.82 0.47 0.67 0.52 0.74 0.80 0.800.890.77
Correct argument values (t) 0.42 0.48 0.43 0.48 0.60 0.62 0.55 0.50 0.58 0.52 0.59 0.66 0.40 0.51 0.51 0.67 0.74 0.75 0.41 0.61 0.38 0.60 0.68 0.730.840.74
Correct argument values (e) 0.42 0.49 0.45 0.49 0.61 0.63 0.56 0.51 0.59 0.53 0.60 0.66 0.45 0.54 0.52 0.68 0.76 0.76 0.42 0.61 0.38 0.61 0.69 0.740.840.74
Missing arguments (t) 0.56 0.47 0.54 0.49 0.33 0.27 0.38 0.45 0.36 0.40 0.35 0.30 0.54 0.46 0.45 0.28 0.20 0.18 0.54 0.34 0.48 0.27 0.20 0.210.110.23
Missing arguments (e) 0.55 0.46 0.52 0.47 0.31 0.26 0.37 0.45 0.36 0.38 0.34 0.29 0.48 0.43 0.44 0.28 0.18 0.18 0.53 0.33 0.48 0.26 0.20 0.200.110.23
Unnecessary arguments (t) 0.02 0.03 0.04 0.04 0.07 0.06 0.08 0.07 0.06 0.05 0.05 0.06 0.03 0.04 0.03 0.05 0.04 0.06 0.03 0.05 0.05 0.05 0.07 0.050.01 0.01
Unnecessary arguments (e) 0.02 0.03 0.04 0.04 0.07 0.06 0.08 0.07 0.06 0.05 0.05 0.06 0.03 0.04 0.03 0.05 0.04 0.06 0.03 0.05 0.05 0.05 0.07 0.050.01 0.01
Hallucinated arguments (t) 0.25 0.23 0.24 0.23 0.16 0.13 0.19 0.20 0.14 0.22 0.14 0.14 0.28 0.26 0.20 0.17 0.11 0.11 0.17 0.12 0.09 0.13 0.11 0.090.080.15
Hallucinated arguments (e) 0.25 0.24 0.24 0.23 0.16 0.13 0.20 0.21 0.14 0.23 0.14 0.14 0.31 0.27 0.21 0.17 0.12 0.11 0.18 0.12 0.09 0.13 0.11 0.090.080.15
Mean argument precision (t) 0.59 0.64 0.61 0.64 0.73 0.79 0.65 0.62 0.74 0.66 0.77 0.73 0.57 0.64 0.70 0.74 0.84 0.81 0.73 0.81 0.82 0.80 0.81 0.830.910.78
Mean argument precision (e) 0.59 0.64 0.61 0.64 0.73 0.79 0.65 0.62 0.74 0.66 0.77 0.73 0.57 0.64 0.70 0.74 0.84 0.81 0.73 0.81 0.82 0.80 0.81 0.830.910.78
Mean argument recall (t) 0.48 0.55 0.49 0.55 0.68 0.74 0.62 0.54 0.64 0.62 0.66 0.68 0.50 0.57 0.59 0.72 0.82 0.82 0.47 0.69 0.56 0.74 0.81 0.800.900.79
Mean argument recall (e) 0.49 0.56 0.51 0.56 0.69 0.74 0.62 0.54 0.65 0.64 0.67 0.68 0.55 0.59 0.60 0.73 0.83 0.82 0.48 0.69 0.57 0.75 0.81 0.800.900.79
Mean arg. Jaccard index (t) 0.42 0.48 0.45 0.49 0.61 0.67 0.53 0.46 0.56 0.54 0.60 0.59 0.43 0.49 0.52 0.63 0.76 0.75 0.42 0.63 0.52 0.68 0.73 0.730.860.75
Mean arg. Jaccard index (e) 0.43 0.49 0.46 0.50 0.62 0.68 0.53 0.47 0.57 0.55 0.61 0.59 0.47 0.51 0.54 0.63 0.77 0.75 0.43 0.64 0.52 0.69 0.74 0.740.860.75
Mean arg. val. cond. acc. (t) 0.94 0.90 0.94 0.93 0.89 0.85 0.88 0.93 0.91 0.86 0.90 0.93 0.87 0.95 0.94 0.95 0.93 0.92 0.90 0.92 0.73 0.83 0.85 0.930.96 0.96
Mean arg. val. cond. acc. (e) 0.94 0.90 0.94 0.93 0.89 0.85 0.88 0.93 0.91 0.86 0.90 0.93 0.87 0.95 0.94 0.95 0.93 0.92 0.90 0.92 0.73 0.83 0.85 0.930.96 0.96
Total errors 7 7 12 10 6 5 3 4 5 9 5 2 39 16 8 2 7 2 7 3 1 5 3 10 0
Incomplete implementations 2 3 11 10 6 5 3 4 2 9 5 2 38 14 8 2 7 2 7 3 1 5010 0
Runtime errors 5 4 10 0 0 0 030 0 01 20 0 0 0 0 0 0 030 0 0
Manuscript submitted to ACM

Mitigating Errors in LLM-Generated Web API Invocations 35
Table 7. Complete evaluation results forfull completionwithretrieval-augmented generationon thesynthetic datasetCodeT5+ (6B)
CodeT5+ (16B)
StarCoder (1B)
StarCoder (3B)
StarCoder (7B)
StarCoder (15.5B)
StarCoder2 (3B)
StarCoder2 (7B)
StarCoder2 (15B)
DeepSeek-Coder (1.3B)
DeepSeek-Coder (6.7B)
DeepSeek-Coder (33B)
Qwen2.5-Coder (0.5B)
Qwen2.5-Coder (1.5B)
Qwen2.5-Coder (3B)
Qwen2.5-Coder (7B)
Qwen2.5-Coder (14B)
Qwen2.5-Coder (32B)
Llama 3.1 (8B)
Llama 3.1 (70B)
Code Llama (7B)
Code Llama (13B)
Code Llama (34B)
Code Llama (70B)
GPT-4o mini
GPT-4o
Executable implementations (t) 0.18 0.43 0.95 0.97 0.991.000.991.000.98 0.95 0.90 0.99 0.00 0.00 0.00 0.00 0.00 0.28 0.00 0.001.000.98 0.971.000.99 0.99
Correct implementations (t) 0.00 0.08 0.22 0.28 0.34 0.36 0.33 0.14 0.33 0.06 0.20 0.24 0.00 0.00 0.00 0.00 0.00 0.13 0.00 0.00 0.07 0.04 0.23 0.42 0.370.66
Correct implementations (e) 0.00 0.19 0.23 0.28 0.34 0.36 0.33 0.14 0.34 0.06 0.22 0.24 0.00 0.00 0.00 0.00 0.00 0.48 0.00 0.00 0.07 0.04 0.24 0.42 0.380.66
Correct endpoints (t) 0.02 0.29 0.74 0.80 0.81 0.87 0.81 0.86 0.87 0.79 0.83 0.90 0.00 0.00 0.00 0.00 0.00 0.26 0.00 0.00 0.86 0.86 0.83 0.88 0.910.94
Correct endpoints (e) 0.08 0.69 0.78 0.82 0.82 0.87 0.82 0.87 0.88 0.83 0.92 0.91 0.00 0.00 0.00 0.001.000.92 0.00 0.00 0.86 0.87 0.85 0.89 0.92 0.94
Hallucinated endpoints (t) 0.17 0.05 0.10 0.07 0.08 0.05 0.06 0.06 0.05 0.04 0.01 0.030.00 0.00 0.00 0.00 0.000.010.00 0.000.04 0.05 0.08 0.04 0.02 0.01
Hallucinated endpoints (e) 0.92 0.11 0.10 0.07 0.08 0.05 0.06 0.06 0.05 0.05 0.01 0.03 1.000.00 0.00 0.00 0.000.040.00 0.000.04 0.05 0.08 0.04 0.02 0.01
Correct URLs (t) 0.02 0.32 0.81 0.83 0.87 0.90 0.86 0.88 0.90 0.82 0.83 0.92 0.00 0.00 0.00 0.00 0.00 0.26 0.00 0.00 0.89 0.89 0.85 0.92 0.920.94
Correct URLs (e) 0.10 0.76 0.85 0.86 0.88 0.90 0.87 0.88 0.92 0.86 0.92 0.93 0.00 0.00 0.00 0.001.000.94 0.00 0.00 0.89 0.90 0.88 0.93 0.93 0.95
Hallucinated URLs (t) 0.16 0.02 0.04 0.03 0.02 0.02 0.01 0.05 0.02 0.03 0.01 0.010.00 0.00 0.00 0.00 0.000.010.00 0.000.02 0.02 0.06 0.01 0.010.00
Hallucinated URLs (e) 0.90 0.05 0.05 0.03 0.02 0.02 0.01 0.05 0.02 0.03 0.01 0.01 1.000.00 0.00 0.00 0.000.020.00 0.000.02 0.02 0.06 0.01 0.010.00
Correct methods (t) 0.17 0.38 0.85 0.90 0.90 0.96 0.90 0.97 0.93 0.91 0.88 0.96 0.00 0.00 0.00 0.00 0.00 0.27 0.00 0.00 0.93 0.93 0.94 0.95 0.960.98
Correct methods (e) 0.94 0.90 0.90 0.93 0.91 0.96 0.91 0.97 0.95 0.96 0.98 0.971.000.00 0.00 0.001.000.98 0.00 0.00 0.93 0.95 0.96 0.95 0.98 0.99
Hallucinated methods (t) 0.01 0.03 0.06 0.04 0.06 0.03 0.05 0.01 0.04 0.02 0.01 0.020.00 0.00 0.00 0.00 0.000.010.00 0.000.03 0.03 0.03 0.03 0.010.00
Hallucinated methods (e) 0.06 0.07 0.06 0.04 0.06 0.03 0.05 0.01 0.04 0.02 0.01 0.020.00 0.00 0.00 0.00 0.000.020.00 0.000.03 0.03 0.03 0.03 0.010.00
Correct argument names (t) 0.04 0.21 0.55 0.60 0.80 0.79 0.74 0.76 0.81 0.71 0.72 0.57 0.00 0.00 0.00 0.00 0.00 0.26 0.00 0.00 0.71 0.79 0.62 0.78 0.930.95
Correct argument names (e) 0.30 0.51 0.58 0.62 0.80 0.80 0.75 0.77 0.82 0.75 0.81 0.57 0.67 0.00 0.00 0.001.000.94 0.00 0.00 0.71 0.80 0.64 0.79 0.95 0.96
Correct argument values (t) 0.03 0.17 0.47 0.54 0.68 0.69 0.68 0.59 0.75 0.44 0.59 0.54 0.00 0.00 0.00 0.00 0.00 0.25 0.00 0.00 0.51 0.51 0.51 0.74 0.770.91
Correct argument values (e) 0.21 0.40 0.49 0.56 0.68 0.70 0.69 0.59 0.76 0.47 0.66 0.54 0.33 0.00 0.00 0.001.000.91 0.00 0.00 0.51 0.51 0.53 0.74 0.78 0.92
Missing arguments (t) 0.96 0.79 0.45 0.40 0.20 0.21 0.26 0.24 0.19 0.29 0.28 0.43 1.00 1.00 1.00 1.00 1.00 0.74 1.00 1.00 0.29 0.21 0.38 0.22 0.070.05
Missing arguments (e) 0.70 0.49 0.42 0.38 0.20 0.20 0.25 0.23 0.18 0.25 0.19 0.43 0.330.00 0.00 0.00 0.000.060.00 0.000.29 0.20 0.36 0.21 0.05 0.04
Unexpected arguments (t) 0.01 0.08 0.24 0.22 0.21 0.21 0.23 0.24 0.26 0.21 0.16 0.11 0.010.00 0.00 0.00 0.000.070.00 0.000.21 0.16 0.30 0.15 0.17 0.12
Unexpected arguments (e) 0.09 0.17 0.25 0.23 0.21 0.21 0.23 0.24 0.26 0.22 0.18 0.11 0.730.00 0.00 0.000.25 0.220.00 0.000.22 0.16 0.31 0.15 0.17 0.12
Mean argument precision (t) 0.88 0.74 0.68 0.71 0.79 0.79 0.76 0.78 0.74 0.74 0.83 0.85 0.20 0.00 0.00 0.00 0.75 0.82 0.00 0.00 0.81 0.86 0.63 0.82 0.890.91
Mean argument precision (e) 0.88 0.74 0.68 0.71 0.79 0.79 0.76 0.78 0.74 0.74 0.83 0.85 0.20 0.00 0.00 0.00 0.75 0.82 0.00 0.00 0.81 0.86 0.63 0.82 0.890.91
Mean argument recall (t) 0.05 0.24 0.58 0.63 0.81 0.80 0.76 0.77 0.79 0.74 0.73 0.54 0.00 0.00 0.00 0.00 0.00 0.26 0.00 0.00 0.73 0.81 0.64 0.77 0.940.96
Mean argument recall (e) 0.28 0.55 0.61 0.65 0.82 0.80 0.77 0.77 0.81 0.78 0.81 0.55 0.67 0.00 0.00 0.001.000.94 0.00 0.00 0.73 0.82 0.66 0.78 0.96 0.96
Mean arg. Jaccard index (t) 0.05 0.22 0.51 0.57 0.73 0.73 0.65 0.68 0.66 0.67 0.66 0.50 0.00 0.00 0.00 0.00 0.00 0.22 0.00 0.00 0.66 0.74 0.54 0.72 0.860.90
Mean arg. Jaccard index (e) 0.27 0.52 0.54 0.59 0.73 0.73 0.66 0.68 0.67 0.70 0.74 0.50 0.18 0.00 0.00 0.00 0.75 0.79 0.00 0.00 0.66 0.75 0.56 0.72 0.870.90
Mean arg. val. cond. acc. (t) 0.73 0.79 0.86 0.89 0.85 0.85 0.92 0.75 0.93 0.59 0.80 0.96 0.50 0.00 0.00 0.001.000.98 0.00 0.00 0.70 0.60 0.84 0.95 0.81 0.96
Mean arg. val. cond. acc. (e) 0.73 0.79 0.86 0.89 0.85 0.85 0.92 0.75 0.93 0.59 0.80 0.96 0.50 0.00 0.00 0.001.000.98 0.00 0.00 0.70 0.60 0.84 0.95 0.81 0.96
Total errors 323 227 19 12 21517 19 39 3 394 395 395 395 394 285 395 39516 1115 2
Incomplete implementations 89 34 17 10 2 1 3 1 7 12 5 1 10 0 0 020 01 4 2 10 0
Runtime errors 233 192 1 20 020 070 0210 0 0 0 0 0 0 0 0705 1
Manuscript submitted to ACM

36 Maninger et al.Table 8. Complete evaluation results forargument completionwithretrieval-augmented generationon thesynthetic datasetCodeT5+ (6B)
CodeT5+ (16B)
StarCoder (1B)
StarCoder (3B)
StarCoder (7B)
StarCoder (15.5B)
StarCoder2 (3B)
StarCoder2 (7B)
StarCoder2 (15B)
DeepSeek-Coder (1.3B)
DeepSeek-Coder (6.7B)
DeepSeek-Coder (33B)
Qwen2.5-Coder (0.5B)
Qwen2.5-Coder (1.5B)
Qwen2.5-Coder (3B)
Qwen2.5-Coder (7B)
Qwen2.5-Coder (14B)
Qwen2.5-Coder (32B)
Llama 3.1 (8B)
Llama 3.1 (70B)
Code Llama (7B)
Code Llama (13B)
Code Llama (34B)
Code Llama (70B)
GPT-4o mini
GPT-4o
Executable implementations (t) 0.20 0.40 0.95 0.97 0.991.000.991.000.99 0.98 0.98 0.99 0.94 0.99 0.991.00 1.000.97 0.991.000.99 0.99 0.991.000.991.00
Correct implementations (t) 0.05 0.09 0.26 0.30 0.38 0.39 0.35 0.15 0.29 0.09 0.23 0.28 0.19 0.07 0.37 0.44 0.46 0.45 0.09 0.29 0.08 0.05 0.29 0.46 0.330.56
Correct implementations (e) 0.26 0.21 0.27 0.31 0.38 0.39 0.35 0.15 0.29 0.09 0.23 0.28 0.20 0.07 0.37 0.44 0.46 0.47 0.09 0.29 0.08 0.05 0.29 0.46 0.340.56
Hallucinated implementations (t)0.030.06 0.25 0.21 0.16 0.21 0.20 0.14 0.11 0.19 0.16 0.07 0.38 0.19 0.27 0.12 0.04 0.13 0.06 0.11 0.10 0.07 0.21 0.07 0.15 0.28
Hallucinated implementations (e) 0.15 0.14 0.27 0.22 0.16 0.21 0.20 0.14 0.11 0.20 0.16 0.07 0.40 0.20 0.27 0.120.040.14 0.06 0.11 0.10 0.07 0.21 0.07 0.15 0.28
Correct argument names (t) 0.10 0.22 0.59 0.66 0.85 0.83 0.78 0.82 0.80 0.78 0.83 0.59 0.58 0.66 0.75 0.86 0.700.880.41 0.61 0.76 0.84 0.70 0.82 0.83 0.75
Correct argument names (e) 0.59 0.56 0.63 0.68 0.85 0.84 0.79 0.82 0.80 0.80 0.84 0.59 0.63 0.67 0.76 0.86 0.700.900.41 0.61 0.76 0.85 0.71 0.82 0.84 0.75
Correct argument values (t) 0.08 0.18 0.51 0.59 0.73 0.73 0.72 0.64 0.74 0.51 0.67 0.56 0.49 0.53 0.72 0.81 0.660.840.38 0.53 0.55 0.55 0.59 0.77 0.67 0.69
Correct argument values (e) 0.49 0.47 0.54 0.61 0.73 0.73 0.73 0.64 0.74 0.53 0.69 0.56 0.53 0.53 0.72 0.81 0.660.860.38 0.53 0.55 0.56 0.60 0.77 0.68 0.69
Missing arguments (t) 0.90 0.78 0.41 0.34 0.15 0.17 0.22 0.18 0.20 0.22 0.17 0.41 0.42 0.34 0.25 0.14 0.300.120.59 0.39 0.24 0.16 0.30 0.18 0.17 0.25
Missing arguments (e) 0.41 0.44 0.37 0.32 0.15 0.16 0.21 0.18 0.20 0.20 0.16 0.41 0.37 0.33 0.24 0.14 0.300.100.59 0.39 0.24 0.15 0.29 0.18 0.16 0.25
Unnecessary arguments (t)0.000.01 0.10 0.09 0.11 0.10 0.13 0.16 0.19 0.11 0.08 0.07 0.08 0.10 0.13 0.16 0.09 0.17 0.06 0.07 0.14 0.11 0.16 0.09 0.09 0.06
Unnecessary arguments (e)0.000.01 0.10 0.09 0.11 0.10 0.13 0.16 0.19 0.11 0.08 0.07 0.09 0.10 0.13 0.16 0.09 0.17 0.06 0.07 0.14 0.11 0.17 0.09 0.09 0.06
Hallucinated arguments (t)0.03 0.030.12 0.11 0.07 0.09 0.08 0.07 0.04 0.09 0.060.030.23 0.10 0.12 0.040.030.07 0.05 0.06 0.040.030.110.030.09 0.14
Hallucinated arguments (e) 0.14 0.08 0.13 0.11 0.07 0.09 0.08 0.07 0.04 0.09 0.060.030.25 0.11 0.12 0.040.030.07 0.05 0.06 0.040.030.110.030.09 0.14
Mean argument precision (t) 0.85 0.87 0.71 0.76 0.83 0.82 0.79 0.82 0.77 0.78 0.86 0.87 0.65 0.76 0.75 0.81 0.89 0.790.900.85 0.86 0.89 0.67 0.86 0.82 0.75
Mean argument precision (e) 0.85 0.87 0.71 0.76 0.83 0.82 0.79 0.82 0.77 0.78 0.86 0.87 0.65 0.76 0.75 0.81 0.89 0.790.900.85 0.86 0.89 0.67 0.86 0.82 0.75
Mean argument recall (t) 0.11 0.24 0.62 0.69 0.86 0.84 0.80 0.82 0.78 0.81 0.83 0.57 0.61 0.65 0.77 0.85 0.730.870.41 0.64 0.77 0.85 0.71 0.81 0.86 0.78
Mean argument recall (e) 0.58 0.60 0.65 0.71 0.86 0.84 0.80 0.82 0.78 0.83 0.84 0.57 0.65 0.66 0.78 0.86 0.730.900.41 0.64 0.78 0.85 0.72 0.81 0.88 0.78
Mean arg. Jaccard index (t) 0.11 0.24 0.56 0.63 0.78 0.77 0.68 0.73 0.65 0.73 0.76 0.53 0.51 0.60 0.66 0.75 0.66 0.73 0.37 0.58 0.70 0.78 0.61 0.760.790.73
Mean arg. Jaccard index (e) 0.57 0.59 0.58 0.64 0.78 0.77 0.69 0.73 0.65 0.75 0.77 0.53 0.54 0.61 0.67 0.75 0.66 0.76 0.38 0.58 0.71 0.79 0.62 0.760.800.73
Mean arg. val. cond. acc. (t) 0.82 0.85 0.87 0.89 0.86 0.85 0.92 0.76 0.92 0.61 0.810.960.85 0.80 0.95 0.94 0.950.960.92 0.88 0.72 0.63 0.85 0.95 0.81 0.90
Mean arg. val. cond. acc. (e) 0.82 0.85 0.87 0.89 0.86 0.85 0.92 0.76 0.92 0.61 0.810.960.85 0.80 0.95 0.94 0.950.960.92 0.88 0.72 0.63 0.85 0.95 0.81 0.90
Total errors 317 236 19 10 2 1 304 8 6 2 25 4 2 1 1 13 202 3 5050
Incomplete implementations 84 27 18 9 2 1 204 8 5 2 24 4 2 1 1 12 102 2 1050
Runtime errors 233 209 1 10 010 0 01010 0 0 01 10 01 40 0 0
Manuscript submitted to ACM

Mitigating Errors in LLM-Generated Web API Invocations 37
Table 9. Complete evaluation results forfull completionwithconstrained decodingon thesynthetic datasetCodeT5+ (6B)
CodeT5+ (16B)
StarCoder (1B)
StarCoder (3B)
StarCoder (7B)
StarCoder (15.5B)
StarCoder2 (3B)
StarCoder2 (7B)
StarCoder2 (15B)
DeepSeek-Coder (1.3B)
DeepSeek-Coder (6.7B)
DeepSeek-Coder (33B)
Qwen2.5-Coder (0.5B)
Qwen2.5-Coder (1.5B)
Qwen2.5-Coder (3B)
Qwen2.5-Coder (7B)
Qwen2.5-Coder (14B)
Qwen2.5-Coder (32B)
Llama 3.1 (8B)
Llama 3.1 (70B)
Code Llama (7B)
Code Llama (13B)
Code Llama (70B)
Executable implementations (t) 0.58 0.750.930.85 0.87 0.86 0.81 0.81 0.85 0.84 0.79 0.79 0.69 0.76 0.74 0.81 0.80 0.86 0.80 0.80 0.73 0.85 0.81
Correct implementations (t) 0.17 0.18 0.19 0.32 0.37 0.30 0.35 0.36 0.46 0.21 0.42 0.46 0.07 0.10 0.17 0.34 0.19 0.42 0.23 0.32 0.09 0.280.50
Correct implementations (e) 0.29 0.23 0.20 0.37 0.43 0.34 0.43 0.44 0.54 0.25 0.53 0.58 0.11 0.14 0.23 0.42 0.24 0.49 0.28 0.40 0.12 0.330.62
Correct endpoints (t) 0.22 0.26 0.27 0.42 0.50 0.50 0.43 0.490.580.39 0.52 0.54 0.13 0.15 0.23 0.44 0.29 0.55 0.28 0.41 0.42 0.52 0.57
Correct endpoints (e) 0.37 0.34 0.29 0.49 0.57 0.58 0.53 0.60 0.69 0.47 0.66 0.68 0.18 0.19 0.31 0.54 0.37 0.64 0.35 0.51 0.57 0.610.71
Hallucinated endpoints (t)0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00
Hallucinated endpoints (e)0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00
Correct URLs (t) 0.22 0.26 0.30 0.42 0.50 0.50 0.43 0.490.580.39 0.52 0.55 0.15 0.18 0.28 0.45 0.37 0.56 0.30 0.43 0.42 0.520.58
Correct URLs (e) 0.38 0.34 0.32 0.50 0.57 0.58 0.53 0.60 0.69 0.47 0.66 0.70 0.22 0.24 0.38 0.56 0.46 0.65 0.37 0.53 0.57 0.610.72
Hallucinated URLs (t)0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00
Hallucinated URLs (e)0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00
Correct methods (t) 0.51 0.63 0.59 0.76 0.790.810.72 0.77 0.79 0.75 0.74 0.73 0.34 0.31 0.38 0.63 0.41 0.68 0.45 0.63 0.63 0.78 0.75
Correct methods (e) 0.87 0.83 0.64 0.89 0.91 0.94 0.890.950.92 0.90 0.94 0.92 0.49 0.40 0.51 0.78 0.50 0.79 0.56 0.79 0.86 0.92 0.93
Hallucinated methods (t)0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00
Hallucinated methods (e)0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00
Correct argument names (t) 0.35 0.49 0.56 0.61 0.68 0.68 0.63 0.65 0.70 0.61 0.64 0.66 0.43 0.52 0.54 0.64 0.650.740.52 0.61 0.55 0.66 0.69
Correct argument names (e) 0.68 0.71 0.60 0.74 0.80 0.80 0.80 0.83 0.84 0.75 0.84 0.84 0.63 0.66 0.74 0.81 0.810.890.67 0.78 0.80 0.81 0.86
Correct argument values (t) 0.33 0.43 0.51 0.56 0.64 0.55 0.60 0.59 0.67 0.47 0.62 0.64 0.31 0.47 0.50 0.60 0.600.690.49 0.58 0.35 0.55 0.67
Correct argument values (e) 0.63 0.62 0.56 0.69 0.76 0.66 0.76 0.76 0.81 0.58 0.80 0.82 0.45 0.60 0.68 0.76 0.75 0.83 0.63 0.75 0.50 0.670.84
Missing arguments (t) 0.65 0.51 0.44 0.39 0.32 0.32 0.37 0.35 0.30 0.39 0.36 0.34 0.57 0.48 0.46 0.36 0.350.260.48 0.39 0.45 0.34 0.31
Missing arguments (e) 0.32 0.29 0.40 0.26 0.20 0.20 0.20 0.17 0.16 0.25 0.16 0.16 0.37 0.34 0.26 0.19 0.190.110.33 0.22 0.20 0.19 0.14
Unexpected arguments (t) 0.16 0.21 0.16 0.19 0.23 0.19 0.20 0.15 0.17 0.23 0.17 0.16 0.32 0.36 0.29 0.20 0.27 0.17 0.24 0.18 0.22 0.210.12
Unexpected arguments (e) 0.27 0.27 0.17 0.23 0.26 0.21 0.25 0.19 0.20 0.27 0.21 0.19 0.41 0.42 0.35 0.24 0.32 0.20 0.29 0.23 0.29 0.240.15
Mean argument precision (t) 0.73 0.73 0.82 0.79 0.77 0.81 0.79 0.84 0.82 0.73 0.83 0.84 0.63 0.57 0.66 0.79 0.69 0.84 0.69 0.81 0.77 0.790.86
Mean argument precision (e) 0.73 0.73 0.82 0.79 0.77 0.81 0.79 0.84 0.82 0.73 0.83 0.84 0.63 0.57 0.66 0.79 0.69 0.84 0.69 0.81 0.77 0.790.86
Mean argument recall (t) 0.42 0.55 0.60 0.66 0.72 0.71 0.67 0.69 0.74 0.65 0.68 0.69 0.47 0.53 0.58 0.68 0.660.770.58 0.65 0.60 0.70 0.71
Mean argument recall (e) 0.73 0.73 0.65 0.77 0.83 0.82 0.83 0.85 0.86 0.78 0.86 0.87 0.67 0.70 0.78 0.84 0.830.900.72 0.81 0.83 0.83 0.88
Mean arg. Jaccard index (t) 0.36 0.47 0.55 0.59 0.63 0.64 0.59 0.63 0.65 0.56 0.61 0.63 0.36 0.38 0.45 0.60 0.520.690.48 0.58 0.52 0.62 0.66
Mean arg. Jaccard index (e) 0.63 0.62 0.59 0.69 0.72 0.74 0.73 0.77 0.77 0.66 0.77 0.80 0.52 0.50 0.60 0.74 0.64 0.80 0.59 0.72 0.72 0.730.82
Mean arg. val. cond. acc. (t) 0.94 0.90 0.93 0.94 0.95 0.80 0.96 0.93 0.97 0.71 0.970.980.73 0.93 0.94 0.95 0.94 0.94 0.95 0.97 0.57 0.830.98
Mean arg. val. cond. acc. (e) 0.94 0.90 0.93 0.94 0.95 0.80 0.96 0.93 0.97 0.71 0.970.980.73 0.93 0.94 0.95 0.94 0.94 0.95 0.97 0.57 0.830.98
Total errors 166 972858 51 54 75 74 58 64 84 83 122 93 102 74 78 57 78 78 108 60 75
Incomplete implementations 65 67913 17 21 29 17 25 46 49 55 85 28 67 40 19 12 28 33 87 29 38
Runtime errors 68 26 17 39 33 25 25 44 261532 28 28 33 26 26 31 20 34 31 17 26 34
Timeouts 2 4 2 6 1 8 21 13 7 3 309 31 9 8 28 25 12 14 4 5 3
Unsatisfiable constraints 310 0 0 0 0 0 0 0 0 0 0 010 0 0 040 0 0 0
Manuscript submitted to ACM

38 Maninger et al.Table 10. Complete evaluation results forargument completionwithconstrained decodingon thesynthetic datasetCodeT5+ (6B)
CodeT5+ (16B)
StarCoder (1B)
StarCoder (3B)
StarCoder (7B)
StarCoder (15.5B)
StarCoder2 (3B)
StarCoder2 (7B)
StarCoder2 (15B)
DeepSeek-Coder (1.3B)
DeepSeek-Coder (6.7B)
DeepSeek-Coder (33B)
Qwen2.5-Coder (0.5B)
Qwen2.5-Coder (1.5B)
Qwen2.5-Coder (3B)
Qwen2.5-Coder (7B)
Qwen2.5-Coder (14B)
Qwen2.5-Coder (32B)
Llama 3.1 (8B)
Llama 3.1 (70B)
Code Llama (7B)
Code Llama (13B)
Code Llama (70B)
Executable implementations (t) 0.55 0.74 0.89 0.840.910.87 0.89 0.89 0.86 0.90 0.81 0.81 0.88 0.89 0.85 0.87 0.90 0.89 0.84 0.85 0.78 0.88 0.83
Correct implementations (t) 0.36 0.52 0.56 0.57 0.66 0.46 0.63 0.67 0.69 0.42 0.59 0.66 0.39 0.65 0.62 0.63 0.67 0.66 0.60 0.65 0.14 0.490.72
Correct implementations (e) 0.66 0.70 0.62 0.68 0.72 0.53 0.71 0.75 0.79 0.47 0.74 0.81 0.44 0.72 0.73 0.73 0.74 0.74 0.71 0.77 0.18 0.550.86
Hallucinated implementations (t)0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00
Hallucinated implementations (e)0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00
Correct argument names (t) 0.45 0.68 0.79 0.77 0.88 0.84 0.86 0.87 0.84 0.85 0.78 0.78 0.80 0.85 0.81 0.840.890.87 0.80 0.81 0.74 0.86 0.80
Correct argument names (e) 0.93 0.95 0.90 0.95 0.98 0.98 0.98 0.98 0.98 0.97 0.98 0.98 0.93 0.98 0.97 0.980.990.98 0.98 0.98 0.98 0.98 0.98
Correct argument values (t) 0.42 0.64 0.73 0.72 0.85 0.71 0.82 0.83 0.81 0.69 0.76 0.76 0.64 0.81 0.78 0.820.870.85 0.77 0.79 0.52 0.73 0.79
Correct argument values (e) 0.87 0.89 0.83 0.89 0.94 0.83 0.94 0.94 0.96 0.79 0.95 0.96 0.74 0.93 0.93 0.96 0.96 0.96 0.94 0.95 0.68 0.840.97
Missing arguments (t) 0.55 0.32 0.21 0.23 0.12 0.16 0.14 0.13 0.16 0.15 0.22 0.22 0.20 0.15 0.19 0.160.110.13 0.20 0.19 0.26 0.14 0.20
Missing arguments (e) 0.07 0.05 0.10 0.05 0.02 0.02 0.02 0.02 0.02 0.03 0.02 0.02 0.07 0.02 0.03 0.020.010.02 0.02 0.02 0.02 0.02 0.02
Unnecessary arguments (t)0.030.04 0.05 0.06 0.08 0.05 0.08 0.07 0.05 0.08 0.08 0.05 0.06 0.07 0.08 0.09 0.09 0.09 0.09 0.07 0.09 0.060.03
Unnecessary arguments (e) 0.06 0.05 0.06 0.08 0.09 0.06 0.09 0.08 0.06 0.09 0.10 0.06 0.07 0.07 0.10 0.10 0.10 0.10 0.11 0.08 0.12 0.070.03
Hallucinated arguments (t)0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00
Hallucinated arguments (e)0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00
Mean argument precision (t) 0.960.970.96 0.95 0.94 0.96 0.94 0.95 0.96 0.94 0.94 0.96 0.95 0.95 0.94 0.93 0.94 0.93 0.93 0.96 0.93 0.950.97
Mean argument precision (e) 0.960.970.96 0.95 0.94 0.96 0.94 0.95 0.96 0.94 0.94 0.96 0.95 0.95 0.94 0.93 0.94 0.93 0.93 0.96 0.93 0.950.97
Mean argument recall (t) 0.51 0.71 0.81 0.800.900.85 0.88 0.88 0.85 0.88 0.79 0.80 0.83 0.87 0.83 0.86 0.89 0.88 0.83 0.83 0.77 0.87 0.82
Mean argument recall (e) 0.94 0.96 0.91 0.960.990.98 0.98 0.980.990.97 0.98 0.98 0.94 0.98 0.970.99 0.99 0.990.98 0.98 0.980.99 0.99
Mean arg. Jaccard index (t) 0.50 0.69 0.79 0.770.850.82 0.83 0.84 0.82 0.83 0.75 0.77 0.79 0.83 0.79 0.80 0.84 0.82 0.77 0.80 0.72 0.84 0.80
Mean arg. Jaccard index (e) 0.91 0.93 0.88 0.91 0.93 0.95 0.93 0.94 0.94 0.92 0.93 0.95 0.90 0.93 0.92 0.92 0.93 0.92 0.92 0.94 0.92 0.950.96
Mean arg. val. cond. acc. (t) 0.95 0.95 0.93 0.94 0.96 0.83 0.97 0.96 0.97 0.79 0.970.980.79 0.96 0.970.98 0.98 0.980.97 0.97 0.65 0.860.98
Mean arg. val. cond. acc. (e) 0.95 0.95 0.93 0.94 0.96 0.83 0.97 0.96 0.97 0.79 0.970.980.79 0.96 0.970.98 0.98 0.980.97 0.97 0.65 0.860.98
Total errors 179 102 43 633652 43 43 54 40 77 74 46 42 58 53 39 42 63 61 86 46 66
Incomplete implementations 43 81 7 11516 8 11 15 20 49 45 24 15 30 10 18 9 27 17 61 26 34
Runtime errors 84 17 33 41 30 28 22 27 33 12 25 27718 21 37 15 24 19 28 23 17 29
Timeouts 9 3 3 1118 13 5 6 6 3 2 15 9 7 6 6 9 17 16 2 3 3
Unsatisfiable constraints 43 10 0 0 0 0 0 020 0 0 0 0 0 0 0 0 0 0 0 0
Manuscript submitted to ACM

Mitigating Errors in LLM-Generated Web API Invocations 39
Table 11. Complete evaluation results forfull completionwithretrieval-augmented generationplusconstrained decodingon thesynthetic dataset
CodeT5+ (6B)
CodeT5+ (16B)
StarCoder (1B)
StarCoder (3B)
StarCoder (7B)
StarCoder (15.5B)
StarCoder2 (3B)
StarCoder2 (7B)
StarCoder2 (15B)
DeepSeek-Coder (1.3B)
DeepSeek-Coder (6.7B)
DeepSeek-Coder (33B)
Qwen2.5-Coder (0.5B)
Qwen2.5-Coder (1.5B)
Qwen2.5-Coder (3B)
Qwen2.5-Coder (7B)
Qwen2.5-Coder (14B)
Qwen2.5-Coder (32B)
Llama 3.1 (8B)
Llama 3.1 (70B)
Code Llama (7B)
Code Llama (13B)
Code Llama (70B)
Executable implementations (t) 0.58 0.750.930.85 0.87 0.86 0.81 0.81 0.85 0.84 0.79 0.79 0.69 0.76 0.74 0.81 0.80 0.86 0.80 0.80 0.73 0.85 0.81
Correct implementations (t) 0.17 0.18 0.19 0.32 0.37 0.30 0.35 0.36 0.46 0.21 0.42 0.46 0.07 0.10 0.17 0.34 0.19 0.42 0.23 0.32 0.09 0.280.50
Correct implementations (e) 0.29 0.23 0.20 0.37 0.43 0.34 0.43 0.44 0.54 0.25 0.53 0.58 0.11 0.14 0.23 0.42 0.24 0.49 0.28 0.40 0.12 0.330.62
Correct endpoints (t) 0.22 0.26 0.27 0.42 0.50 0.50 0.43 0.490.580.39 0.52 0.54 0.13 0.15 0.23 0.44 0.29 0.55 0.28 0.41 0.42 0.52 0.57
Correct endpoints (e) 0.37 0.34 0.29 0.49 0.57 0.58 0.53 0.60 0.69 0.47 0.66 0.68 0.18 0.19 0.31 0.54 0.37 0.64 0.35 0.51 0.57 0.610.71
Hallucinated endpoints (t)0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00
Hallucinated endpoints (e)0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00
Correct URLs (t) 0.22 0.26 0.30 0.42 0.50 0.50 0.43 0.490.580.39 0.52 0.55 0.15 0.18 0.28 0.45 0.37 0.56 0.30 0.43 0.42 0.520.58
Correct URLs (e) 0.38 0.34 0.32 0.50 0.57 0.58 0.53 0.60 0.69 0.47 0.66 0.70 0.22 0.24 0.38 0.56 0.46 0.65 0.37 0.53 0.57 0.610.72
Hallucinated URLs (t)0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00
Hallucinated URLs (e)0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00
Correct methods (t) 0.51 0.63 0.59 0.76 0.790.810.72 0.77 0.79 0.75 0.74 0.73 0.34 0.31 0.38 0.63 0.41 0.68 0.45 0.63 0.63 0.78 0.75
Correct methods (e) 0.87 0.83 0.64 0.89 0.91 0.94 0.890.950.92 0.90 0.94 0.92 0.49 0.40 0.51 0.78 0.50 0.79 0.56 0.79 0.86 0.92 0.93
Hallucinated methods (t)0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00
Hallucinated methods (e)0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00
Correct argument names (t) 0.35 0.49 0.56 0.61 0.68 0.68 0.63 0.65 0.70 0.61 0.64 0.66 0.43 0.52 0.54 0.64 0.650.740.52 0.61 0.55 0.66 0.69
Correct argument names (e) 0.68 0.71 0.60 0.74 0.80 0.80 0.80 0.83 0.84 0.75 0.84 0.84 0.63 0.66 0.74 0.81 0.810.890.67 0.78 0.80 0.81 0.86
Correct argument values (t) 0.33 0.43 0.51 0.56 0.64 0.55 0.60 0.59 0.67 0.47 0.62 0.64 0.31 0.47 0.50 0.60 0.600.690.49 0.58 0.35 0.55 0.67
Correct argument values (e) 0.63 0.62 0.56 0.69 0.76 0.66 0.76 0.76 0.81 0.58 0.80 0.82 0.45 0.60 0.68 0.76 0.75 0.83 0.63 0.75 0.50 0.670.84
Missing arguments (t) 0.65 0.51 0.44 0.39 0.32 0.32 0.37 0.35 0.30 0.39 0.36 0.34 0.57 0.48 0.46 0.36 0.350.260.48 0.39 0.45 0.34 0.31
Missing arguments (e) 0.32 0.29 0.40 0.26 0.20 0.20 0.20 0.17 0.16 0.25 0.16 0.16 0.37 0.34 0.26 0.19 0.190.110.33 0.22 0.20 0.19 0.14
Unexpected arguments (t) 0.16 0.21 0.16 0.19 0.23 0.19 0.20 0.15 0.17 0.23 0.17 0.16 0.32 0.36 0.29 0.20 0.27 0.17 0.24 0.18 0.22 0.210.12
Unexpected arguments (e) 0.27 0.27 0.17 0.23 0.26 0.21 0.25 0.19 0.20 0.27 0.21 0.19 0.41 0.42 0.35 0.24 0.32 0.20 0.29 0.23 0.29 0.240.15
Mean argument precision (t) 0.73 0.73 0.82 0.79 0.77 0.81 0.79 0.84 0.82 0.73 0.83 0.84 0.63 0.57 0.66 0.79 0.69 0.84 0.69 0.81 0.77 0.790.86
Mean argument precision (e) 0.73 0.73 0.82 0.79 0.77 0.81 0.79 0.84 0.82 0.73 0.83 0.84 0.63 0.57 0.66 0.79 0.69 0.84 0.69 0.81 0.77 0.790.86
Mean argument recall (t) 0.42 0.55 0.60 0.66 0.72 0.71 0.67 0.69 0.74 0.65 0.68 0.69 0.47 0.53 0.58 0.68 0.660.770.58 0.65 0.60 0.70 0.71
Mean argument recall (e) 0.73 0.73 0.65 0.77 0.83 0.82 0.83 0.85 0.86 0.78 0.86 0.87 0.67 0.70 0.78 0.84 0.830.900.72 0.81 0.83 0.83 0.88
Mean arg. Jaccard index (t) 0.36 0.47 0.55 0.59 0.63 0.64 0.59 0.63 0.65 0.56 0.61 0.63 0.36 0.38 0.45 0.60 0.520.690.48 0.58 0.52 0.62 0.66
Mean arg. Jaccard index (e) 0.63 0.62 0.59 0.69 0.72 0.74 0.73 0.77 0.77 0.66 0.77 0.80 0.52 0.50 0.60 0.74 0.64 0.80 0.59 0.72 0.72 0.730.82
Mean arg. val. cond. acc. (t) 0.94 0.90 0.93 0.94 0.95 0.80 0.96 0.93 0.97 0.71 0.970.980.73 0.93 0.94 0.95 0.94 0.94 0.95 0.97 0.57 0.830.98
Mean arg. val. cond. acc. (e) 0.94 0.90 0.93 0.94 0.95 0.80 0.96 0.93 0.97 0.71 0.970.980.73 0.93 0.94 0.95 0.94 0.94 0.95 0.97 0.57 0.830.98
Total errors 166 972858 51 54 75 74 58 64 84 83 122 93 102 74 78 57 78 78 108 60 75
Incomplete implementations 65 67913 17 21 29 17 25 46 49 55 85 28 67 40 19 12 28 33 87 29 38
Runtime errors 68 26 17 39 33 25 25 44 261532 28 28 33 26 26 31 20 34 31 17 26 34
Timeouts 2 4 2 6 1 8 21 13 7 3 309 31 9 8 28 25 12 14 4 5 3
Unsatisfiable constraints 310 0 0 0 0 0 0 0 0 0 0 010 0 0 040 0 0 0
Manuscript submitted to ACM

40 Maninger et al.Table 12. Complete evaluation results forargument completionwithretrieval-augmented generationplusconstrained decodingon thesynthetic dataset
CodeT5+ (6B)
CodeT5+ (16B)
StarCoder (1B)
StarCoder (3B)
StarCoder (7B)
StarCoder (15.5B)
StarCoder2 (3B)
StarCoder2 (7B)
StarCoder2 (15B)
DeepSeek-Coder (1.3B)
DeepSeek-Coder (6.7B)
DeepSeek-Coder (33B)
Qwen2.5-Coder (0.5B)
Qwen2.5-Coder (1.5B)
Qwen2.5-Coder (3B)
Qwen2.5-Coder (7B)
Qwen2.5-Coder (14B)
Qwen2.5-Coder (32B)
Llama 3.1 (8B)
Llama 3.1 (70B)
Code Llama (7B)
Code Llama (13B)
Code Llama (70B)
Executable implementations (t) 0.55 0.74 0.89 0.840.910.87 0.89 0.89 0.86 0.90 0.81 0.81 0.88 0.89 0.85 0.87 0.90 0.89 0.84 0.85 0.78 0.88 0.83
Correct implementations (t) 0.36 0.52 0.56 0.57 0.66 0.46 0.63 0.67 0.69 0.42 0.59 0.66 0.39 0.65 0.62 0.63 0.67 0.66 0.60 0.65 0.14 0.490.72
Correct implementations (e) 0.66 0.70 0.62 0.68 0.72 0.53 0.71 0.75 0.79 0.47 0.74 0.81 0.44 0.72 0.73 0.73 0.74 0.74 0.71 0.77 0.18 0.550.86
Hallucinated implementations (t)0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00
Hallucinated implementations (e)0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00
Correct argument names (t) 0.45 0.68 0.79 0.77 0.88 0.84 0.86 0.87 0.84 0.85 0.78 0.78 0.80 0.85 0.81 0.840.890.87 0.80 0.81 0.74 0.86 0.80
Correct argument names (e) 0.93 0.95 0.90 0.95 0.98 0.98 0.98 0.98 0.98 0.97 0.98 0.98 0.93 0.98 0.97 0.980.990.98 0.98 0.98 0.98 0.98 0.98
Correct argument values (t) 0.42 0.64 0.73 0.72 0.85 0.71 0.82 0.83 0.81 0.69 0.76 0.76 0.64 0.81 0.78 0.820.870.85 0.77 0.79 0.52 0.73 0.79
Correct argument values (e) 0.87 0.89 0.83 0.89 0.94 0.83 0.94 0.94 0.96 0.79 0.95 0.96 0.74 0.93 0.93 0.96 0.96 0.96 0.94 0.95 0.68 0.840.97
Missing arguments (t) 0.55 0.32 0.21 0.23 0.12 0.16 0.14 0.13 0.16 0.15 0.22 0.22 0.20 0.15 0.19 0.160.110.13 0.20 0.19 0.26 0.14 0.20
Missing arguments (e) 0.07 0.05 0.10 0.05 0.02 0.02 0.02 0.02 0.02 0.03 0.02 0.02 0.07 0.02 0.03 0.020.010.02 0.02 0.02 0.02 0.02 0.02
Unnecessary arguments (t)0.030.04 0.05 0.06 0.08 0.05 0.08 0.07 0.05 0.08 0.08 0.05 0.06 0.07 0.08 0.09 0.09 0.09 0.09 0.07 0.09 0.060.03
Unnecessary arguments (e) 0.06 0.05 0.06 0.08 0.09 0.06 0.09 0.08 0.06 0.09 0.10 0.06 0.07 0.07 0.10 0.10 0.10 0.10 0.11 0.08 0.12 0.070.03
Hallucinated arguments (t)0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00
Hallucinated arguments (e)0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00
Mean argument precision (t) 0.960.970.96 0.95 0.94 0.96 0.94 0.95 0.96 0.94 0.94 0.96 0.95 0.95 0.94 0.93 0.94 0.93 0.93 0.96 0.93 0.950.97
Mean argument precision (e) 0.960.970.96 0.95 0.94 0.96 0.94 0.95 0.96 0.94 0.94 0.96 0.95 0.95 0.94 0.93 0.94 0.93 0.93 0.96 0.93 0.950.97
Mean argument recall (t) 0.51 0.71 0.81 0.800.900.85 0.88 0.88 0.85 0.88 0.79 0.80 0.83 0.87 0.83 0.86 0.89 0.88 0.83 0.83 0.77 0.87 0.82
Mean argument recall (e) 0.94 0.96 0.91 0.960.990.98 0.98 0.980.990.97 0.98 0.98 0.94 0.98 0.970.99 0.99 0.990.98 0.98 0.980.99 0.99
Mean arg. Jaccard index (t) 0.50 0.69 0.79 0.770.850.82 0.83 0.84 0.82 0.83 0.75 0.77 0.79 0.83 0.79 0.80 0.84 0.82 0.77 0.80 0.72 0.84 0.80
Mean arg. Jaccard index (e) 0.91 0.93 0.88 0.91 0.93 0.95 0.93 0.94 0.94 0.92 0.93 0.95 0.90 0.93 0.92 0.92 0.93 0.92 0.92 0.94 0.92 0.950.96
Mean arg. val. cond. acc. (t) 0.95 0.95 0.93 0.94 0.96 0.83 0.97 0.96 0.97 0.79 0.970.980.79 0.96 0.970.98 0.98 0.980.97 0.97 0.65 0.860.98
Mean arg. val. cond. acc. (e) 0.95 0.95 0.93 0.94 0.96 0.83 0.97 0.96 0.97 0.79 0.970.980.79 0.96 0.970.98 0.98 0.980.97 0.97 0.65 0.860.98
Total errors 179 102 43 633652 43 43 54 40 77 74 46 42 58 53 39 42 63 61 86 46 66
Incomplete implementations 43 81 7 11516 8 11 15 20 49 45 24 15 30 10 18 9 27 17 61 26 34
Runtime errors 84 17 33 41 30 28 22 27 33 12 25 27718 21 37 15 24 19 28 23 17 29
Timeouts 9 3 3 1118 13 5 6 6 3 2 15 9 7 6 6 9 17 16 2 3 3
Unsatisfiable constraints 43 10 0 0 0 0 0 020 0 0 0 0 0 0 0 0 0 0 0 0
Manuscript submitted to ACM

Mitigating Errors in LLM-Generated Web API Invocations 41
Table 13. Complete evaluation results forfull completionwithvanillageneration on thereal-world datasetCodeT5+ (6B)
CodeT5+ (16B)
StarCoder (1B)
StarCoder (3B)
StarCoder (7B)
StarCoder (15.5B)
StarCoder2 (3B)
StarCoder2 (7B)
StarCoder2 (15B)
DeepSeek-Coder (1.3B)
DeepSeek-Coder (6.7B)
DeepSeek-Coder (33B)
Qwen2.5-Coder (0.5B)
Qwen2.5-Coder (1.5B)
Qwen2.5-Coder (3B)
Qwen2.5-Coder (7B)
Qwen2.5-Coder (14B)
Qwen2.5-Coder (32B)
Llama 3.1 (8B)
Llama 3.1 (70B)
Code Llama (7B)
Code Llama (13B)
Code Llama (34B)
Code Llama (70B)
GPT-4o mini
GPT-4o
Executable implementations (t) 0.89 0.961.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.000.61 0.00 0.00 0.14 0.00 0.25 0.14 0.001.00 1.00 1.00 1.00 1.000.79
Correct implementations (t) 0.39 0.54 0.46 0.57 0.61 0.50 0.64 0.46 0.54 0.57 0.54 0.54 0.39 0.00 0.00 0.07 0.00 0.18 0.11 0.00 0.61 0.68 0.390.710.64 0.64
Correct implementations (e) 0.44 0.56 0.46 0.57 0.61 0.50 0.64 0.46 0.54 0.57 0.54 0.54 0.65 0.00 0.00 0.50 0.00 0.71 0.75 0.00 0.61 0.68 0.39 0.71 0.640.82
Correct endpoints (t) 0.46 0.54 0.54 0.71 0.71 0.68 0.68 0.61 0.64 0.64 0.71 0.61 0.39 0.00 0.00 0.07 0.00 0.25 0.11 0.00 0.79 0.68 0.640.820.68 0.64
Correct endpoints (e) 0.52 0.56 0.54 0.71 0.71 0.68 0.68 0.61 0.64 0.64 0.71 0.61 0.65 0.00 0.00 0.50 0.001.000.75 0.00 0.79 0.68 0.64 0.82 0.68 0.82
Hallucinated endpoints (t) 0.36 0.43 0.43 0.25 0.29 0.29 0.29 0.36 0.32 0.32 0.25 0.36 0.110.00 0.000.040.00 0.00 0.00 0.000.18 0.25 0.36 0.18 0.25 0.14
Hallucinated endpoints (e) 0.40 0.44 0.43 0.25 0.29 0.29 0.29 0.36 0.32 0.32 0.25 0.36 0.180.00 0.000.250.00 0.00 0.00 0.000.18 0.25 0.36 0.18 0.25 0.18
Correct URLs (t) 0.46 0.61 0.54 0.71 0.71 0.71 0.68 0.61 0.68 0.64 0.71 0.61 0.39 0.00 0.00 0.07 0.00 0.25 0.11 0.00 0.79 0.71 0.680.820.71 0.68
Correct URLs (e) 0.52 0.63 0.54 0.71 0.71 0.71 0.68 0.61 0.68 0.64 0.71 0.61 0.65 0.00 0.00 0.50 0.001.000.75 0.00 0.79 0.71 0.68 0.82 0.71 0.86
Hallucinated URLs (t) 0.36 0.36 0.43 0.25 0.29 0.25 0.29 0.36 0.29 0.32 0.25 0.36 0.110.00 0.000.040.00 0.00 0.00 0.000.18 0.21 0.32 0.18 0.21 0.11
Hallucinated URLs (e) 0.40 0.37 0.43 0.25 0.29 0.25 0.29 0.36 0.29 0.32 0.25 0.36 0.180.00 0.000.250.00 0.00 0.00 0.000.18 0.21 0.32 0.18 0.21 0.14
Correct methods (t) 0.86 0.891.000.96 0.96 0.96 0.96 0.96 0.961.000.961.000.57 0.00 0.00 0.14 0.00 0.25 0.14 0.001.000.96 0.96 0.96 0.96 0.75
Correct methods (e) 0.96 0.931.000.96 0.96 0.96 0.96 0.96 0.961.000.961.000.94 0.00 0.001.000.001.00 1.000.001.000.96 0.96 0.96 0.96 0.95
Hallucinated methods (t) 0.04 0.070.000.04 0.04 0.04 0.04 0.04 0.040.000.040.000.040.00 0.00 0.00 0.00 0.00 0.00 0.00 0.000.04 0.04 0.04 0.04 0.04
Hallucinated methods (e) 0.04 0.070.000.04 0.04 0.04 0.04 0.04 0.040.000.040.000.060.00 0.00 0.00 0.00 0.00 0.00 0.00 0.000.04 0.04 0.04 0.04 0.05
Correct argument names (t) 0.61 0.60 0.67 0.79 0.79 0.78 0.75 0.61 0.73 0.84 0.81 0.85 0.33 0.00 0.00 0.07 0.00 0.28 0.16 0.000.870.81 0.73 0.82 0.78 0.60
Correct argument names (e) 0.77 0.68 0.67 0.79 0.79 0.78 0.75 0.61 0.73 0.84 0.81 0.85 0.92 0.00 0.00 0.50 0.00 0.951.000.00 0.87 0.81 0.73 0.82 0.78 0.93
Correct argument values (t) 0.43 0.51 0.60 0.69 0.78 0.73 0.70 0.60 0.66 0.76 0.73 0.81 0.31 0.00 0.00 0.06 0.00 0.27 0.10 0.000.850.81 0.64 0.78 0.75 0.60
Correct argument values (e) 0.55 0.58 0.60 0.69 0.78 0.73 0.70 0.60 0.66 0.76 0.73 0.81 0.88 0.00 0.00 0.40 0.00 0.90 0.64 0.00 0.85 0.81 0.64 0.78 0.750.93
Missing arguments (t) 0.39 0.40 0.33 0.21 0.21 0.22 0.25 0.39 0.27 0.16 0.19 0.15 0.67 1.00 1.00 0.93 1.00 0.72 0.84 1.000.130.19 0.27 0.18 0.22 0.40
Missing arguments (e) 0.23 0.32 0.33 0.21 0.21 0.22 0.25 0.39 0.27 0.16 0.19 0.15 0.080.00 0.000.500.000.050.00 0.000.13 0.19 0.27 0.18 0.22 0.07
Unexpected arguments (t) 0.23 0.16 0.26 0.13 0.13 0.19 0.14 0.16 0.16 0.12 0.11 0.08 0.040.00 0.000.010.000.010.00 0.000.14 0.12 0.13 0.15 0.14 0.01
Unexpected arguments (e) 0.27 0.18 0.26 0.13 0.13 0.19 0.14 0.16 0.16 0.12 0.11 0.08 0.110.00 0.000.090.000.050.00 0.000.14 0.12 0.13 0.15 0.14 0.02
Mean argument precision (t) 0.81 0.77 0.81 0.87 0.87 0.80 0.84 0.81 0.85 0.90 0.86 0.94 0.90 0.00 0.00 0.75 0.00 0.951.000.00 0.85 0.89 0.88 0.89 0.86 0.95
Mean argument precision (e) 0.81 0.77 0.81 0.87 0.87 0.80 0.84 0.81 0.85 0.90 0.86 0.94 0.90 0.00 0.00 0.75 0.00 0.951.000.00 0.85 0.89 0.88 0.89 0.86 0.95
Mean argument recall (t) 0.72 0.67 0.72 0.86 0.78 0.76 0.79 0.65 0.76 0.83 0.79 0.78 0.51 0.00 0.00 0.08 0.00 0.24 0.14 0.00 0.84 0.81 0.710.890.83 0.69
Mean argument recall (e) 0.82 0.70 0.72 0.86 0.78 0.76 0.79 0.65 0.76 0.83 0.79 0.78 0.91 0.00 0.00 0.58 0.00 0.951.000.00 0.84 0.81 0.71 0.89 0.83 0.91
Mean arg. Jaccard index (t) 0.64 0.61 0.67 0.82 0.75 0.69 0.76 0.61 0.72 0.80 0.77 0.76 0.50 0.00 0.00 0.08 0.00 0.23 0.14 0.00 0.79 0.78 0.670.850.80 0.69
Mean arg. Jaccard index (e) 0.72 0.63 0.67 0.82 0.75 0.69 0.76 0.61 0.72 0.80 0.77 0.76 0.89 0.00 0.00 0.58 0.00 0.931.000.00 0.79 0.78 0.67 0.85 0.80 0.91
Mean arg. val. cond. acc. (t) 0.83 0.85 0.91 0.90 0.98 0.94 0.95 0.97 0.90 0.89 0.90 0.94 0.98 0.00 0.00 0.67 0.00 0.93 0.75 0.00 0.981.000.82 0.92 0.971.00
Mean arg. val. cond. acc. (e) 0.83 0.85 0.91 0.90 0.98 0.94 0.95 0.97 0.90 0.89 0.90 0.94 0.98 0.00 0.00 0.67 0.00 0.93 0.75 0.00 0.981.000.82 0.92 0.971.00
Total errors 3 10 0 0 0 0 0 0 0 0 011 28 28 24 28 21 24 280 0 0 0 06
Incomplete implementations 20 0 0 0 0 0 0 0 0 0 010 0 0 0 0 0 0 0 0 0 0 0 0
Runtime errors 1 10 0 0 0 0 0 0 0 0 050 0 0 0 0 0 0 0 0 0 0 01
Manuscript submitted to ACM

42 Maninger et al.Table 14. Complete evaluation results forargument completionwithvanillageneration on thereal-world datasetCodeT5+ (6B)
CodeT5+ (16B)
StarCoder (1B)
StarCoder (3B)
StarCoder (7B)
StarCoder (15.5B)
StarCoder2 (3B)
StarCoder2 (7B)
StarCoder2 (15B)
DeepSeek-Coder (1.3B)
DeepSeek-Coder (6.7B)
DeepSeek-Coder (33B)
Qwen2.5-Coder (0.5B)
Qwen2.5-Coder (1.5B)
Qwen2.5-Coder (3B)
Qwen2.5-Coder (7B)
Qwen2.5-Coder (14B)
Qwen2.5-Coder (32B)
Llama 3.1 (8B)
Llama 3.1 (70B)
Code Llama (7B)
Code Llama (13B)
Code Llama (34B)
Code Llama (70B)
GPT-4o mini
GPT-4o
Executable implementations (t) 0.861.000.96 0.96 0.96 0.86 0.89 0.82 0.89 0.93 0.93 0.96 0.93 0.93 0.82 0.86 0.93 0.82 0.68 0.00 0.86 0.89 0.93 0.791.000.96
Correct implementations (t) 0.54 0.57 0.68 0.50 0.57 0.57 0.50 0.39 0.50 0.50 0.71 0.64 0.43 0.61 0.61 0.61 0.68 0.46 0.39 0.00 0.57 0.68 0.57 0.43 0.790.82
Correct implementations (e) 0.62 0.57 0.70 0.52 0.59 0.67 0.56 0.48 0.56 0.54 0.77 0.67 0.46 0.65 0.74 0.71 0.73 0.57 0.58 0.00 0.67 0.76 0.62 0.55 0.790.85
Hallucinated implementations (t) 0.14 0.29 0.18 0.32 0.25 0.14 0.21 0.25 0.32 0.25 0.14 0.14 0.32 0.18 0.11 0.11 0.14 0.25 0.210.000.07 0.11 0.04 0.11 0.14 0.04
Hallucinated implementations (e) 0.17 0.29 0.19 0.33 0.26 0.17 0.24 0.30 0.36 0.27 0.15 0.15 0.35 0.19 0.13 0.12 0.15 0.30 0.320.000.08 0.12 0.04 0.14 0.14 0.04
Correct argument names (t) 0.51 0.51 0.75 0.69 0.63 0.82 0.73 0.73 0.67 0.67 0.88 0.69 0.54 0.60 0.64 0.61 0.66 0.58 0.48 0.00 0.76 0.69 0.43 0.42 0.790.94
Correct argument names (e) 0.71 0.51 0.77 0.71 0.65 0.87 0.78 0.77 0.67 0.67 0.88 0.69 0.58 0.60 0.72 0.63 0.66 0.64 0.68 0.00 0.82 0.84 0.48 0.49 0.790.98
Correct argument values (t) 0.46 0.45 0.70 0.63 0.58 0.76 0.69 0.69 0.64 0.60 0.84 0.63 0.49 0.58 0.63 0.57 0.63 0.52 0.46 0.00 0.75 0.64 0.43 0.40 0.780.94
Correct argument values (e) 0.65 0.45 0.72 0.65 0.60 0.81 0.73 0.72 0.64 0.60 0.84 0.63 0.53 0.58 0.70 0.58 0.63 0.57 0.66 0.00 0.81 0.78 0.48 0.47 0.780.98
Missing arguments (t) 0.49 0.49 0.25 0.31 0.37 0.18 0.27 0.27 0.33 0.33 0.12 0.31 0.46 0.40 0.36 0.39 0.34 0.42 0.52 1.00 0.24 0.31 0.57 0.58 0.210.06
Missing arguments (e) 0.29 0.49 0.23 0.29 0.35 0.13 0.22 0.23 0.33 0.33 0.12 0.31 0.42 0.40 0.28 0.37 0.34 0.36 0.320.000.18 0.16 0.52 0.51 0.21 0.02
Unnecessary arguments (t) 0.04 0.02 0.05 0.09 0.04 0.06 0.06 0.05 0.07 0.05 0.04 0.04 0.03 0.010.00 0.00 0.000.010.00 0.000.010.00 0.000.01 0.010.00
Unnecessary arguments (e) 0.05 0.02 0.05 0.10 0.04 0.07 0.06 0.05 0.07 0.05 0.04 0.04 0.04 0.010.00 0.00 0.000.010.00 0.000.020.00 0.000.02 0.010.00
Hallucinated arguments (t) 0.10 0.23 0.17 0.21 0.25 0.08 0.14 0.12 0.16 0.22 0.09 0.05 0.21 0.24 0.12 0.03 0.03 0.17 0.110.000.03 0.03 0.01 0.04 0.13 0.01
Hallucinated arguments (e) 0.14 0.23 0.18 0.21 0.26 0.08 0.15 0.13 0.16 0.22 0.09 0.05 0.23 0.24 0.13 0.03 0.03 0.18 0.150.000.03 0.04 0.02 0.05 0.13 0.02
Mean argument precision (t) 0.82 0.78 0.83 0.70 0.76 0.88 0.86 0.80 0.83 0.77 0.91 0.92 0.72 0.90 0.96 0.95 0.96 0.91 0.87 0.00 0.940.98 0.980.91 0.880.98
Mean argument precision (e) 0.82 0.78 0.83 0.70 0.76 0.88 0.86 0.80 0.83 0.77 0.91 0.92 0.72 0.90 0.96 0.95 0.96 0.91 0.87 0.00 0.940.98 0.980.91 0.880.98
Mean argument recall (t) 0.67 0.69 0.86 0.80 0.81 0.78 0.75 0.64 0.68 0.77 0.85 0.84 0.66 0.70 0.69 0.69 0.75 0.65 0.52 0.00 0.70 0.79 0.64 0.57 0.860.95
Mean argument recall (e) 0.79 0.69 0.90 0.84 0.84 0.92 0.84 0.79 0.76 0.83 0.91 0.87 0.72 0.76 0.84 0.81 0.81 0.79 0.78 0.00 0.83 0.89 0.69 0.73 0.860.99
Mean arg. Jaccard index (t) 0.63 0.68 0.78 0.65 0.70 0.72 0.67 0.57 0.61 0.68 0.82 0.78 0.59 0.69 0.68 0.68 0.74 0.63 0.51 0.00 0.70 0.78 0.62 0.53 0.840.93
Mean arg. Jaccard index (e) 0.75 0.68 0.81 0.68 0.73 0.85 0.75 0.69 0.68 0.73 0.88 0.82 0.64 0.74 0.83 0.80 0.80 0.77 0.77 0.00 0.82 0.88 0.68 0.67 0.840.97
Mean arg. val. cond. acc. (t) 0.95 0.92 0.94 0.90 0.94 0.92 0.91 0.91 0.95 0.86 0.91 0.92 0.94 0.99 0.98 0.92 0.97 0.93 0.98 0.00 0.99 0.951.000.97 0.961.00
Mean arg. val. cond. acc. (e) 0.95 0.92 0.94 0.90 0.94 0.92 0.91 0.91 0.95 0.86 0.91 0.92 0.94 0.99 0.98 0.92 0.97 0.93 0.98 0.00 0.99 0.951.000.97 0.961.00
Total errors 401 1 1 4 3 5 3 2 2 1 2 2 5 4 2 5 9 28 4 3 2 601
Incomplete implementations 10 0 01 1 10 01 202010 01027 30 0 0 0 0
Runtime errors 301 103 2 5 3 10102 4 4 2 4 9 1 1 3 2 601
Manuscript submitted to ACM

Mitigating Errors in LLM-Generated Web API Invocations 43
Table 15. Complete evaluation results forfull completionwithretrieval-augmented generationon thereal-world datasetCodeT5+ (6B)
CodeT5+ (16B)
StarCoder (1B)
StarCoder (3B)
StarCoder (7B)
StarCoder (15.5B)
StarCoder2 (3B)
StarCoder2 (7B)
StarCoder2 (15B)
DeepSeek-Coder (1.3B)
DeepSeek-Coder (6.7B)
DeepSeek-Coder (33B)
Qwen2.5-Coder (0.5B)
Qwen2.5-Coder (1.5B)
Qwen2.5-Coder (3B)
Qwen2.5-Coder (7B)
Qwen2.5-Coder (14B)
Qwen2.5-Coder (32B)
Llama 3.1 (8B)
Llama 3.1 (70B)
Code Llama (7B)
Code Llama (13B)
Code Llama (34B)
Code Llama (70B)
GPT-4o mini
GPT-4o
Executable implementations (t) 0.61 0.751.000.961.000.93 0.961.00 1.00 1.000.961.000.50 0.00 0.00 0.07 0.04 0.21 0.21 0.00 0.891.00 1.00 1.00 1.000.89
Correct implementations (t) 0.25 0.39 0.46 0.54 0.64 0.54 0.61 0.46 0.61 0.68 0.46 0.68 0.29 0.00 0.00 0.04 0.04 0.21 0.07 0.00 0.57 0.61 0.430.710.46 0.54
Correct implementations (e) 0.41 0.52 0.46 0.56 0.64 0.58 0.63 0.46 0.61 0.68 0.48 0.68 0.57 0.00 0.00 0.501.00 1.000.33 0.00 0.64 0.61 0.43 0.71 0.46 0.60
Correct endpoints (t) 0.36 0.46 0.68 0.71 0.68 0.75 0.79 0.64 0.71 0.71 0.570.820.32 0.00 0.00 0.07 0.04 0.21 0.14 0.00 0.71 0.71 0.640.820.71 0.79
Correct endpoints (e) 0.59 0.62 0.68 0.74 0.68 0.81 0.81 0.64 0.71 0.71 0.59 0.82 0.64 0.00 0.001.00 1.00 1.000.67 0.00 0.80 0.71 0.64 0.82 0.71 0.88
Hallucinated endpoints (t) 0.25 0.18 0.32 0.25 0.29 0.18 0.18 0.32 0.29 0.29 0.36 0.18 0.180.00 0.00 0.00 0.00 0.000.070.000.14 0.29 0.36 0.18 0.29 0.11
Hallucinated endpoints (e) 0.41 0.24 0.32 0.26 0.29 0.19 0.19 0.32 0.29 0.29 0.37 0.18 0.360.00 0.00 0.00 0.00 0.000.330.000.16 0.29 0.36 0.18 0.29 0.12
Correct URLs (t) 0.36 0.46 0.68 0.75 0.71 0.75 0.79 0.64 0.71 0.71 0.570.820.32 0.00 0.00 0.07 0.04 0.21 0.18 0.00 0.71 0.71 0.640.820.71 0.79
Correct URLs (e) 0.59 0.62 0.68 0.78 0.71 0.81 0.81 0.64 0.71 0.71 0.59 0.82 0.64 0.00 0.001.00 1.00 1.000.83 0.00 0.80 0.71 0.64 0.82 0.71 0.88
Hallucinated URLs (t) 0.25 0.18 0.32 0.21 0.25 0.18 0.18 0.32 0.29 0.29 0.36 0.18 0.180.00 0.00 0.00 0.00 0.000.040.000.14 0.29 0.36 0.18 0.29 0.11
Hallucinated URLs (e) 0.41 0.24 0.32 0.22 0.25 0.19 0.19 0.32 0.29 0.29 0.37 0.18 0.360.00 0.00 0.00 0.00 0.000.170.000.16 0.29 0.36 0.18 0.29 0.12
Correct methods (t) 0.61 0.751.000.93 0.96 0.93 0.961.00 1.00 1.000.961.000.46 0.00 0.00 0.07 0.04 0.21 0.18 0.00 0.86 0.961.00 1.00 1.000.89
Correct methods (e)1.00 1.00 1.000.96 0.961.00 1.00 1.00 1.00 1.00 1.00 1.000.93 0.00 0.001.00 1.00 1.000.83 0.00 0.96 0.961.00 1.00 1.00 1.00
Hallucinated methods (t)0.00 0.00 0.000.04 0.040.00 0.00 0.00 0.00 0.00 0.00 0.000.040.00 0.00 0.00 0.00 0.000.040.00 0.000.040.00 0.00 0.00 0.00
Hallucinated methods (e)0.00 0.00 0.000.04 0.040.00 0.00 0.00 0.00 0.00 0.00 0.000.070.00 0.00 0.00 0.00 0.000.170.00 0.000.040.00 0.00 0.00 0.00
Correct argument names (t) 0.28 0.48 0.78 0.87 0.93 0.60 0.79 0.75 0.90 0.93 0.52 0.93 0.22 0.00 0.00 0.06 0.03 0.27 0.21 0.00 0.75 0.84 0.840.94 0.940.69
Correct argument names (e) 0.61 0.78 0.78 0.89 0.93 0.82 0.90 0.75 0.90 0.93 0.61 0.93 0.79 0.00 0.001.00 1.00 1.000.78 0.00 0.88 0.84 0.84 0.94 0.94 0.98
Correct argument values (t) 0.27 0.46 0.73 0.79 0.90 0.60 0.76 0.72 0.88 0.88 0.48 0.90 0.21 0.00 0.00 0.04 0.03 0.27 0.19 0.00 0.73 0.84 0.790.930.88 0.69
Correct argument values (e) 0.58 0.76 0.73 0.82 0.90 0.82 0.86 0.72 0.88 0.88 0.56 0.90 0.74 0.00 0.00 0.751.00 1.000.72 0.00 0.86 0.84 0.79 0.93 0.88 0.98
Missing arguments (t) 0.72 0.52 0.22 0.13 0.07 0.40 0.21 0.25 0.10 0.07 0.48 0.07 0.78 1.00 1.00 0.94 0.97 0.73 0.79 1.00 0.25 0.16 0.160.06 0.060.31
Missing arguments (e) 0.39 0.22 0.22 0.11 0.07 0.18 0.10 0.25 0.10 0.07 0.39 0.07 0.210.00 0.00 0.00 0.00 0.000.220.000.12 0.16 0.16 0.06 0.06 0.02
Unexpected arguments (t) 0.08 0.16 0.18 0.15 0.19 0.13 0.11 0.11 0.09 0.09 0.03 0.13 0.060.00 0.000.010.00 0.000.060.000.12 0.12 0.14 0.11 0.19 0.09
Unexpected arguments (e) 0.16 0.24 0.18 0.16 0.19 0.17 0.12 0.11 0.09 0.09 0.03 0.13 0.170.00 0.000.200.00 0.000.180.000.14 0.12 0.14 0.11 0.19 0.13
Mean argument precision (t) 0.73 0.73 0.81 0.87 0.85 0.79 0.88 0.89 0.95 0.96 0.97 0.92 0.85 0.00 0.00 0.881.00 1.000.78 0.00 0.87 0.87 0.90 0.93 0.83 0.88
Mean argument precision (e) 0.73 0.73 0.81 0.87 0.85 0.79 0.88 0.89 0.95 0.96 0.97 0.92 0.85 0.00 0.00 0.881.00 1.000.78 0.00 0.87 0.87 0.90 0.93 0.83 0.88
Mean argument recall (t) 0.37 0.53 0.77 0.78 0.86 0.70 0.80 0.75 0.83 0.89 0.62 0.88 0.36 0.00 0.00 0.07 0.04 0.21 0.15 0.00 0.78 0.78 0.750.920.91 0.87
Mean argument recall (e) 0.66 0.72 0.77 0.81 0.86 0.76 0.83 0.75 0.83 0.89 0.65 0.88 0.83 0.00 0.001.00 1.00 1.000.71 0.00 0.88 0.78 0.75 0.92 0.91 0.99
Mean arg. Jaccard index (t) 0.35 0.45 0.69 0.73 0.81 0.65 0.75 0.68 0.79 0.85 0.60 0.83 0.36 0.00 0.00 0.06 0.04 0.21 0.14 0.00 0.72 0.73 0.670.880.81 0.76
Mean arg. Jaccard index (e) 0.62 0.62 0.69 0.76 0.81 0.70 0.78 0.68 0.79 0.85 0.62 0.83 0.81 0.00 0.00 0.881.00 1.000.63 0.00 0.82 0.73 0.67 0.88 0.81 0.87
Mean arg. val. cond. acc. (t) 0.90 0.97 0.93 0.91 0.961.000.97 0.97 0.98 0.94 0.93 0.96 0.90 0.00 0.00 0.831.00 1.000.90 0.00 0.971.000.87 0.99 0.961.00
Mean arg. val. cond. acc. (e) 0.90 0.97 0.93 0.91 0.961.000.97 0.97 0.98 0.94 0.93 0.96 0.90 0.00 0.00 0.831.00 1.000.90 0.00 0.971.000.87 0.99 0.961.00
Total errors 11 70102 10 0 01014 28 28 26 27 22 22 28 30 0 0 03
Incomplete implementations 1 20 0 02 10 0 01010 0 0 0 0 0 010 0 0 0 0
Runtime errors 6 10 0 0 0 0 0 0 0 0 030 0 0 0 0 0 020 0 0 0 0
Manuscript submitted to ACM

44 Maninger et al.Table 16. Complete evaluation results forargument completionwithretrieval-augmented generationon thereal-world dataset
CodeT5+ (6B)
CodeT5+ (16B)
StarCoder (1B)
StarCoder (3B)
StarCoder (7B)
StarCoder (15.5B)
StarCoder2 (3B)
StarCoder2 (7B)
StarCoder2 (15B)
DeepSeek-Coder (1.3B)
DeepSeek-Coder (6.7B)
DeepSeek-Coder (33B)
Qwen2.5-Coder (0.5B)
Qwen2.5-Coder (1.5B)
Qwen2.5-Coder (3B)
Qwen2.5-Coder (7B)
Qwen2.5-Coder (14B)
Qwen2.5-Coder (32B)
Llama 3.1 (8B)
Llama 3.1 (70B)
Code Llama (7B)
Code Llama (13B)
Code Llama (34B)
Code Llama (70B)
GPT-4o mini
GPT-4o
Executable implementations (t) 0.75 0.79 0.89 0.89 0.96 0.93 0.96 0.86 0.89 0.96 0.891.000.82 0.93 0.861.000.89 0.89 0.681.00 1.000.71 0.96 0.86 0.96 0.89
Correct implementations (t) 0.46 0.54 0.57 0.46 0.57 0.64 0.61 0.46 0.54 0.610.750.68 0.46 0.54 0.50 0.57 0.46 0.54 0.46 0.64 0.68 0.57 0.57 0.46 0.50 0.50
Correct implementations (e) 0.62 0.68 0.64 0.52 0.59 0.69 0.63 0.54 0.60 0.630.840.68 0.57 0.58 0.58 0.57 0.52 0.60 0.68 0.64 0.68 0.80 0.59 0.54 0.52 0.56
Hallucinated implementations (t) 0.07 0.11 0.07 0.070.00 0.00 0.000.07 0.07 0.070.000.11 0.18 0.21 0.11 0.04 0.04 0.140.00 0.000.040.00 0.000.040.000.04
Hallucinated implementations (e) 0.10 0.14 0.08 0.080.00 0.00 0.000.08 0.08 0.070.000.11 0.22 0.23 0.12 0.04 0.04 0.160.00 0.000.040.00 0.000.040.000.04
Correct argument names (t) 0.36 0.43 0.63 0.840.960.76 0.76 0.76 0.70 0.94 0.58 0.90 0.51 0.43 0.76 0.40 0.40 0.69 0.52 0.45 0.82 0.45 0.58 0.36 0.93 0.81
Correct argument names (e) 0.71 0.71 0.76 0.890.970.86 0.81 0.80 0.78 0.94 0.65 0.90 0.65 0.54 0.81 0.40 0.42 0.71 0.78 0.45 0.82 0.75 0.62 0.48 0.94 0.95
Correct argument values (t) 0.34 0.40 0.55 0.790.900.73 0.75 0.73 0.69 0.88 0.57 0.88 0.49 0.42 0.73 0.40 0.40 0.64 0.52 0.45 0.81 0.45 0.58 0.36 0.87 0.81
Correct argument values (e) 0.68 0.66 0.67 0.84 0.91 0.83 0.79 0.77 0.77 0.88 0.63 0.88 0.63 0.52 0.78 0.40 0.42 0.66 0.78 0.45 0.81 0.75 0.62 0.48 0.880.95
Missing arguments (t) 0.64 0.57 0.37 0.160.040.24 0.24 0.24 0.30 0.06 0.42 0.10 0.49 0.57 0.24 0.60 0.60 0.31 0.48 0.55 0.18 0.55 0.42 0.64 0.07 0.19
Missing arguments (e) 0.29 0.29 0.24 0.110.030.14 0.19 0.20 0.22 0.06 0.35 0.10 0.35 0.46 0.19 0.60 0.58 0.29 0.22 0.55 0.18 0.25 0.38 0.52 0.06 0.05
Unnecessary arguments (t) 0.060.000.07 0.21 0.27 0.09 0.08 0.09 0.09 0.16 0.01 0.09 0.01 0.01 0.11 0.010.000.03 0.060.000.080.000.070.000.31 0.16
Unnecessary arguments (e) 0.100.000.08 0.22 0.27 0.11 0.09 0.10 0.10 0.16 0.02 0.09 0.02 0.02 0.11 0.010.000.03 0.080.000.080.000.070.000.31 0.18
Hallucinated arguments (t) 0.010.000.15 0.020.00 0.00 0.000.01 0.01 0.020.000.04 0.11 0.140.000.010.000.010.00 0.000.010.00 0.000.010.000.02
Hallucinated arguments (e) 0.030.000.18 0.020.00 0.00 0.000.01 0.01 0.020.000.04 0.13 0.170.000.010.000.010.00 0.000.010.00 0.000.020.000.03
Mean argument precision (t) 0.941.000.83 0.77 0.80 0.93 0.92 0.92 0.95 0.86 0.99 0.89 0.88 0.90 0.93 0.961.000.97 0.981.000.921.000.96 0.98 0.73 0.84
Mean argument precision (e) 0.941.000.83 0.77 0.80 0.93 0.92 0.92 0.95 0.86 0.99 0.89 0.88 0.90 0.93 0.961.000.97 0.981.000.921.000.96 0.98 0.73 0.84
Mean argument recall (t) 0.54 0.61 0.80 0.770.930.82 0.77 0.65 0.69 0.90 0.77 0.87 0.62 0.65 0.71 0.62 0.52 0.71 0.55 0.65 0.82 0.59 0.68 0.57 0.87 0.85
Mean argument recall (e) 0.75 0.80 0.91 0.880.970.89 0.80 0.77 0.78 0.94 0.87 0.87 0.76 0.71 0.83 0.62 0.58 0.80 0.82 0.65 0.82 0.84 0.70 0.67 0.910.97
Mean arg. Jaccard index (t) 0.51 0.61 0.71 0.63 0.76 0.77 0.74 0.61 0.650.800.760.800.59 0.65 0.66 0.61 0.52 0.69 0.53 0.65 0.79 0.59 0.65 0.56 0.68 0.73
Mean arg. Jaccard index (e) 0.71 0.80 0.80 0.72 0.79 0.83 0.77 0.72 0.73 0.830.860.80 0.72 0.71 0.77 0.61 0.58 0.78 0.80 0.65 0.79 0.84 0.67 0.66 0.71 0.83
Mean arg. val. cond. acc. (t) 0.98 0.96 0.90 0.95 0.94 0.97 0.98 0.96 0.97 0.93 0.98 0.99 0.99 0.94 0.961.00 1.000.951.00 1.000.981.00 1.00 1.000.961.00
Mean arg. val. cond. acc. (e) 0.98 0.96 0.90 0.95 0.94 0.97 0.98 0.96 0.97 0.93 0.98 0.99 0.99 0.94 0.961.00 1.000.951.00 1.000.981.00 1.00 1.000.961.00
Total errors 7 6 3 3 1 2 1 4 3 1 305 2 403 3 90 08 1 4 1 3
Incomplete implementations 1 2 1 201 1 10 0105 10 0 010 0 0 0 0 0 0 0
Runtime errors 202 1 1 103 3 1 20 01 403 2 90 08 1 4 1 3
Manuscript submitted to ACM

Mitigating Errors in LLM-Generated Web API Invocations 45
Table 17. Complete evaluation results forfull completionwithconstrained decodingon thereal-world datasetCodeT5+ (6B)
CodeT5+ (16B)
StarCoder (1B)
StarCoder (3B)
StarCoder (7B)
StarCoder (15.5B)
StarCoder2 (3B)
StarCoder2 (7B)
StarCoder2 (15B)
DeepSeek-Coder (1.3B)
DeepSeek-Coder (6.7B)
DeepSeek-Coder (33B)
Qwen2.5-Coder (0.5B)
Qwen2.5-Coder (1.5B)
Qwen2.5-Coder (3B)
Qwen2.5-Coder (7B)
Qwen2.5-Coder (14B)
Qwen2.5-Coder (32B)
Llama 3.1 (8B)
Llama 3.1 (70B)
Code Llama (7B)
Code Llama (13B)
Code Llama (70B)
Executable implementations (t) 0.89 0.86 0.890.96 0.960.89 0.93 0.93 0.890.960.89 0.93 0.68 0.68 0.75 0.93 0.86 0.86 0.89 0.89 0.93 0.89 0.82
Correct implementations (t) 0.29 0.43 0.43 0.57 0.64 0.54 0.64 0.50 0.57 0.57 0.610.680.07 0.04 0.25 0.64 0.57 0.54 0.54 0.61 0.61 0.61 0.61
Correct implementations (e) 0.32 0.50 0.48 0.59 0.67 0.60 0.69 0.54 0.64 0.59 0.68 0.73 0.11 0.05 0.33 0.69 0.67 0.62 0.60 0.68 0.65 0.680.74
Correct endpoints (t) 0.75 0.61 0.71 0.86 0.86 0.82 0.82 0.79 0.82 0.86 0.820.890.29 0.07 0.32 0.82 0.68 0.82 0.79 0.860.890.79 0.79
Correct endpoints (e) 0.84 0.71 0.80 0.89 0.89 0.92 0.88 0.85 0.92 0.89 0.920.960.42 0.11 0.43 0.88 0.790.960.880.96 0.960.880.96
Hallucinated endpoints (t)0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00
Hallucinated endpoints (e)0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00
Correct URLs (t) 0.79 0.61 0.71 0.86 0.86 0.82 0.82 0.79 0.82 0.86 0.820.890.29 0.07 0.32 0.82 0.68 0.82 0.79 0.860.890.79 0.79
Correct URLs (e) 0.88 0.71 0.80 0.89 0.89 0.92 0.88 0.85 0.92 0.89 0.920.960.42 0.11 0.43 0.88 0.790.960.880.96 0.960.880.96
Hallucinated URLs (t)0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00
Hallucinated URLs (e)0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00
Correct methods (t) 0.86 0.79 0.89 0.93 0.93 0.86 0.89 0.89 0.860.960.86 0.93 0.36 0.14 0.39 0.93 0.68 0.82 0.86 0.86 0.93 0.86 0.79
Correct methods (e) 0.96 0.921.000.96 0.96 0.96 0.96 0.96 0.961.000.961.000.53 0.21 0.521.000.79 0.96 0.96 0.961.000.96 0.96
Hallucinated methods (t)0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00
Hallucinated methods (e)0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00
Correct argument names (t) 0.67 0.60 0.57 0.73 0.79 0.73 0.67 0.76 0.72 0.78 0.720.850.45 0.27 0.51 0.58 0.57 0.61 0.61 0.61 0.79 0.61 0.52
Correct argument names (e) 0.82 0.82 0.69 0.75 0.82 0.89 0.79 0.890.910.800.910.90 0.67 0.56 0.74 0.80 0.76 0.89 0.76 0.87 0.88 0.82 0.88
Correct argument values (t) 0.61 0.57 0.51 0.66 0.76 0.69 0.63 0.73 0.67 0.75 0.670.840.37 0.16 0.43 0.55 0.57 0.61 0.58 0.61 0.78 0.61 0.51
Correct argument values (e) 0.75 0.78 0.62 0.68 0.78 0.84 0.74 0.86 0.85 0.77 0.850.890.56 0.34 0.63 0.76 0.760.890.72 0.87 0.87 0.82 0.85
Missing arguments (t) 0.33 0.40 0.43 0.27 0.21 0.27 0.33 0.24 0.28 0.22 0.280.150.55 0.73 0.49 0.42 0.43 0.39 0.39 0.39 0.21 0.39 0.48
Missing arguments (e) 0.18 0.18 0.31 0.25 0.18 0.11 0.21 0.110.090.200.090.10 0.33 0.44 0.26 0.20 0.24 0.11 0.24 0.13 0.12 0.18 0.12
Unexpected arguments (t) 0.22 0.24 0.26 0.17 0.24 0.22 0.18 0.25 0.17 0.18 0.13 0.15 0.29 0.43 0.32 0.15 0.25 0.15 0.15 0.15 0.160.090.12
Unexpected arguments (e) 0.26 0.30 0.30 0.18 0.24 0.26 0.21 0.28 0.21 0.19 0.16 0.16 0.38 0.61 0.41 0.20 0.31 0.21 0.18 0.20 0.180.120.18
Mean argument precision (t) 0.71 0.69 0.68 0.82 0.76 0.79 0.83 0.74 0.83 0.74 0.86 0.85 0.53 0.35 0.57 0.78 0.73 0.80 0.77 0.83 0.83 0.860.87
Mean argument precision (e) 0.71 0.69 0.68 0.82 0.76 0.79 0.83 0.74 0.83 0.74 0.86 0.85 0.53 0.35 0.57 0.78 0.73 0.80 0.77 0.83 0.83 0.860.87
Mean argument recall (t) 0.72 0.70 0.71 0.800.850.78 0.82 0.82 0.79 0.76 0.79 0.82 0.39 0.35 0.53 0.77 0.71 0.75 0.71 0.78 0.80 0.74 0.71
Mean argument recall (e) 0.81 0.83 0.80 0.83 0.89 0.88 0.89 0.890.900.79 0.89 0.89 0.60 0.55 0.73 0.83 0.850.900.81 0.89 0.87 0.84 0.89
Mean arg. Jaccard index (t) 0.55 0.53 0.56 0.73 0.72 0.67 0.74 0.66 0.71 0.68 0.730.760.30 0.23 0.41 0.71 0.60 0.65 0.64 0.70 0.71 0.68 0.68
Mean arg. Jaccard index (e) 0.63 0.63 0.64 0.76 0.74 0.76 0.80 0.71 0.81 0.71 0.83 0.82 0.45 0.35 0.56 0.77 0.71 0.78 0.73 0.79 0.77 0.770.84
Mean arg. val. cond. acc. (t) 0.87 0.96 0.92 0.92 0.96 0.96 0.95 0.98 0.95 0.98 0.96 0.99 0.81 0.57 0.85 0.971.00 1.000.971.000.981.000.95
Mean arg. val. cond. acc. (e) 0.87 0.96 0.92 0.92 0.96 0.96 0.95 0.98 0.95 0.98 0.96 0.99 0.81 0.57 0.85 0.971.00 1.000.971.000.981.000.95
Total errors 3 4 31 13 2 2 313 2 9 9 7 2 4 4 3 3 2 3 5
Incomplete implementations 1 20 0 01 2 2 203 2 6 7 2 1 3 2 102 2 4
Runtime errors 2 2 3 1 1 20 01 10 02 2 201 2 2 201 1
Timeouts0 0 0 0 0 0 0 0 0 0 0 0103 10 0 010 0 0
Unsatisfiable constraints0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
Manuscript submitted to ACM

46 Maninger et al.Table 18. Complete evaluation results forargument completionwithconstrained decodingon thereal-world datasetCodeT5+ (6B)
CodeT5+ (16B)
StarCoder (1B)
StarCoder (3B)
StarCoder (7B)
StarCoder (15.5B)
StarCoder2 (3B)
StarCoder2 (7B)
StarCoder2 (15B)
DeepSeek-Coder (1.3B)
DeepSeek-Coder (6.7B)
DeepSeek-Coder (33B)
Qwen2.5-Coder (0.5B)
Qwen2.5-Coder (1.5B)
Qwen2.5-Coder (3B)
Qwen2.5-Coder (7B)
Qwen2.5-Coder (14B)
Qwen2.5-Coder (32B)
Llama 3.1 (8B)
Llama 3.1 (70B)
Code Llama (7B)
Code Llama (13B)
Code Llama (70B)
Executable implementations (t) 0.89 0.86 0.89 0.96 0.93 0.931.000.96 0.89 0.96 0.93 0.93 0.79 0.96 0.89 0.96 0.96 0.96 0.93 0.93 0.86 0.82 0.79
Correct implementations (t) 0.57 0.57 0.610.680.64 0.640.680.64 0.64 0.610.68 0.680.57 0.64 0.61 0.640.68 0.680.64 0.64 0.61 0.61 0.61
Correct implementations (e) 0.64 0.67 0.68 0.70 0.69 0.69 0.68 0.67 0.72 0.63 0.73 0.73 0.73 0.67 0.68 0.67 0.70 0.70 0.69 0.69 0.71 0.740.77
Hallucinated implementations (t)0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00
Hallucinated implementations (e)0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00
Correct argument names (t) 0.64 0.69 0.63 0.79 0.67 0.640.820.67 0.63 0.78 0.76 0.78 0.39 0.69 0.58 0.69 0.76 0.79 0.70 0.72 0.73 0.70 0.66
Correct argument names (e) 0.78 0.85 0.76 0.82 0.82 0.78 0.82 0.79 0.79 0.80 0.81 0.83 0.74 0.81 0.78 0.81 0.78 0.82 0.85 0.870.910.90 0.88
Correct argument values (t) 0.63 0.64 0.60 0.78 0.66 0.630.810.67 0.63 0.78 0.76 0.78 0.37 0.69 0.58 0.69 0.76 0.79 0.70 0.72 0.73 0.70 0.66
Correct argument values (e) 0.76 0.80 0.73 0.80 0.80 0.76 0.81 0.79 0.79 0.80 0.81 0.83 0.71 0.81 0.78 0.81 0.78 0.82 0.85 0.870.910.90 0.88
Missing arguments (t) 0.36 0.31 0.37 0.21 0.33 0.360.180.33 0.37 0.22 0.24 0.22 0.61 0.31 0.42 0.31 0.24 0.21 0.30 0.28 0.27 0.30 0.34
Missing arguments (e) 0.22 0.15 0.24 0.18 0.18 0.22 0.18 0.21 0.21 0.20 0.19 0.17 0.26 0.19 0.22 0.19 0.22 0.18 0.15 0.130.090.10 0.12
Unnecessary arguments (t) 0.12 0.07 0.09 0.08 0.13 0.11 0.14 0.160.040.15 0.06 0.07 0.07 0.20 0.11 0.13 0.13 0.08 0.09 0.11 0.13 0.060.04
Unnecessary arguments (e) 0.14 0.08 0.11 0.08 0.15 0.13 0.14 0.190.050.16 0.06 0.07 0.12 0.23 0.14 0.15 0.13 0.08 0.11 0.13 0.16 0.07 0.06
Hallucinated arguments (t)0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00
Hallucinated arguments (e)0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00
Mean argument precision (t) 0.89 0.91 0.89 0.89 0.88 0.89 0.86 0.850.920.83 0.90 0.89 0.88 0.85 0.88 0.87 0.86 0.90 0.87 0.87 0.88 0.90 0.91
Mean argument precision (e) 0.89 0.91 0.89 0.89 0.88 0.89 0.86 0.850.920.83 0.90 0.89 0.88 0.85 0.88 0.87 0.86 0.90 0.87 0.87 0.88 0.90 0.91
Mean argument recall (t) 0.69 0.70 0.68 0.77 0.75 0.730.810.77 0.70 0.76 0.73 0.75 0.60 0.78 0.70 0.78 0.75 0.77 0.78 0.80 0.74 0.70 0.65
Mean argument recall (e) 0.78 0.83 0.78 0.80 0.82 0.79 0.81 0.80 0.80 0.79 0.79 0.81 0.79 0.81 0.80 0.81 0.78 0.80 0.85 0.870.88 0.880.85
Mean arg. Jaccard index (t) 0.65 0.67 0.66 0.73 0.69 0.69 0.74 0.70 0.69 0.69 0.71 0.71 0.57 0.70 0.65 0.71 0.71 0.73 0.740.750.69 0.67 0.64
Mean arg. Jaccard index (e) 0.74 0.80 0.75 0.76 0.75 0.75 0.74 0.73 0.78 0.71 0.77 0.77 0.75 0.73 0.74 0.74 0.74 0.76 0.80 0.81 0.820.84 0.84
Mean arg. val. cond. acc. (t) 0.99 0.96 0.95 0.98 0.97 0.97 0.981.00 1.00 1.00 1.00 1.000.971.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00
Mean arg. val. cond. acc. (e) 0.99 0.96 0.95 0.98 0.97 0.97 0.981.00 1.00 1.00 1.00 1.000.971.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00
Total errors 3 4 3 1 2 201 3 1 2 2 6 1 3 1 1 1 2 2 4 5 6
Incomplete implementations 1 30 0 010 0202 2 4010 010 04 5 5
Runtime errors 2 1 3 1 2 101 1 10 0101 1 1020 0 01
Timeouts0 0 0 0 0 0 0 0 0 0 0 01 1 10 0 0 020 0 0
Unsatisfiable constraints0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
Manuscript submitted to ACM

Mitigating Errors in LLM-Generated Web API Invocations 47
Table 19. Complete evaluation results forfull completionwithretrieval-augmented generationplusconstrained decodingon thereal-world dataset
CodeT5+ (6B)
CodeT5+ (16B)
StarCoder (1B)
StarCoder (3B)
StarCoder (7B)
StarCoder (15.5B)
StarCoder2 (3B)
StarCoder2 (7B)
StarCoder2 (15B)
DeepSeek-Coder (1.3B)
DeepSeek-Coder (6.7B)
DeepSeek-Coder (33B)
Qwen2.5-Coder (0.5B)
Qwen2.5-Coder (1.5B)
Qwen2.5-Coder (3B)
Qwen2.5-Coder (7B)
Qwen2.5-Coder (14B)
Qwen2.5-Coder (32B)
Llama 3.1 (8B)
Llama 3.1 (70B)
Code Llama (7B)
Code Llama (13B)
Code Llama (70B)
Executable implementations (t) 0.89 0.86 0.890.96 0.960.89 0.93 0.93 0.890.960.89 0.93 0.68 0.68 0.75 0.93 0.86 0.86 0.89 0.89 0.93 0.89 0.82
Correct implementations (t) 0.29 0.43 0.43 0.57 0.64 0.54 0.64 0.50 0.57 0.57 0.610.680.07 0.04 0.25 0.64 0.57 0.54 0.54 0.61 0.61 0.61 0.61
Correct implementations (e) 0.32 0.50 0.48 0.59 0.67 0.60 0.69 0.54 0.64 0.59 0.68 0.73 0.11 0.05 0.33 0.69 0.67 0.62 0.60 0.68 0.65 0.680.74
Correct endpoints (t) 0.75 0.61 0.71 0.86 0.86 0.82 0.82 0.79 0.82 0.86 0.820.890.29 0.07 0.32 0.82 0.68 0.82 0.79 0.860.890.79 0.79
Correct endpoints (e) 0.84 0.71 0.80 0.89 0.89 0.92 0.88 0.85 0.92 0.89 0.920.960.42 0.11 0.43 0.88 0.790.960.880.96 0.960.880.96
Hallucinated endpoints (t)0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00
Hallucinated endpoints (e)0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00
Correct URLs (t) 0.79 0.61 0.71 0.86 0.86 0.82 0.82 0.79 0.82 0.86 0.820.890.29 0.07 0.32 0.82 0.68 0.82 0.79 0.860.890.79 0.79
Correct URLs (e) 0.88 0.71 0.80 0.89 0.89 0.92 0.88 0.85 0.92 0.89 0.920.960.42 0.11 0.43 0.88 0.790.960.880.96 0.960.880.96
Hallucinated URLs (t)0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00
Hallucinated URLs (e)0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00
Correct methods (t) 0.86 0.79 0.89 0.93 0.93 0.86 0.89 0.89 0.860.960.86 0.93 0.36 0.14 0.39 0.93 0.68 0.82 0.86 0.86 0.93 0.86 0.79
Correct methods (e) 0.96 0.921.000.96 0.96 0.96 0.96 0.96 0.961.000.961.000.53 0.21 0.521.000.79 0.96 0.96 0.961.000.96 0.96
Hallucinated methods (t)0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00
Hallucinated methods (e)0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00
Correct argument names (t) 0.67 0.60 0.57 0.73 0.79 0.73 0.67 0.76 0.72 0.78 0.720.850.45 0.27 0.51 0.58 0.57 0.61 0.61 0.61 0.79 0.61 0.52
Correct argument names (e) 0.82 0.82 0.69 0.75 0.82 0.89 0.79 0.890.910.800.910.90 0.67 0.56 0.74 0.80 0.76 0.89 0.76 0.87 0.88 0.82 0.88
Correct argument values (t) 0.61 0.57 0.51 0.66 0.76 0.69 0.63 0.73 0.67 0.75 0.670.840.37 0.16 0.43 0.55 0.57 0.61 0.58 0.61 0.78 0.61 0.51
Correct argument values (e) 0.75 0.78 0.62 0.68 0.78 0.84 0.74 0.86 0.85 0.77 0.850.890.56 0.34 0.63 0.76 0.760.890.72 0.87 0.87 0.82 0.85
Missing arguments (t) 0.33 0.40 0.43 0.27 0.21 0.27 0.33 0.24 0.28 0.22 0.280.150.55 0.73 0.49 0.42 0.43 0.39 0.39 0.39 0.21 0.39 0.48
Missing arguments (e) 0.18 0.18 0.31 0.25 0.18 0.11 0.21 0.110.090.200.090.10 0.33 0.44 0.26 0.20 0.24 0.11 0.24 0.13 0.12 0.18 0.12
Unexpected arguments (t) 0.22 0.24 0.26 0.17 0.24 0.22 0.18 0.25 0.17 0.18 0.13 0.15 0.29 0.43 0.32 0.15 0.25 0.15 0.15 0.15 0.160.090.12
Unexpected arguments (e) 0.26 0.30 0.30 0.18 0.24 0.26 0.21 0.28 0.21 0.19 0.16 0.16 0.38 0.61 0.41 0.20 0.31 0.21 0.18 0.20 0.180.120.18
Mean argument precision (t) 0.71 0.69 0.68 0.82 0.76 0.79 0.83 0.74 0.83 0.74 0.86 0.85 0.53 0.35 0.57 0.78 0.73 0.80 0.77 0.83 0.83 0.860.87
Mean argument precision (e) 0.71 0.69 0.68 0.82 0.76 0.79 0.83 0.74 0.83 0.74 0.86 0.85 0.53 0.35 0.57 0.78 0.73 0.80 0.77 0.83 0.83 0.860.87
Mean argument recall (t) 0.72 0.70 0.71 0.800.850.78 0.82 0.82 0.79 0.76 0.79 0.82 0.39 0.35 0.53 0.77 0.71 0.75 0.71 0.78 0.80 0.74 0.71
Mean argument recall (e) 0.81 0.83 0.80 0.83 0.89 0.88 0.89 0.890.900.79 0.89 0.89 0.60 0.55 0.73 0.83 0.850.900.81 0.89 0.87 0.84 0.89
Mean arg. Jaccard index (t) 0.55 0.53 0.56 0.73 0.72 0.67 0.74 0.66 0.71 0.68 0.730.760.30 0.23 0.41 0.71 0.60 0.65 0.64 0.70 0.71 0.68 0.68
Mean arg. Jaccard index (e) 0.63 0.63 0.64 0.76 0.74 0.76 0.80 0.71 0.81 0.71 0.83 0.82 0.45 0.35 0.56 0.77 0.71 0.78 0.73 0.79 0.77 0.770.84
Mean arg. val. cond. acc. (t) 0.87 0.96 0.92 0.92 0.96 0.96 0.95 0.98 0.95 0.98 0.96 0.99 0.81 0.57 0.85 0.971.00 1.000.971.000.981.000.95
Mean arg. val. cond. acc. (e) 0.87 0.96 0.92 0.92 0.96 0.96 0.95 0.98 0.95 0.98 0.96 0.99 0.81 0.57 0.85 0.971.00 1.000.971.000.981.000.95
Total errors 3 4 31 13 2 2 313 2 9 9 7 2 4 4 3 3 2 3 5
Incomplete implementations 1 20 0 01 2 2 203 2 6 7 2 1 3 2 102 2 4
Runtime errors 2 2 3 1 1 20 01 10 02 2 201 2 2 201 1
Timeouts0 0 0 0 0 0 0 0 0 0 0 0103 10 0 010 0 0
Unsatisfiable constraints0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
Manuscript submitted to ACM

48 Maninger et al.Table 20. Complete evaluation results forargument completionwithretrieval-augmented generationplusconstrained decodingon thereal-world dataset
CodeT5+ (6B)
CodeT5+ (16B)
StarCoder (1B)
StarCoder (3B)
StarCoder (7B)
StarCoder (15.5B)
StarCoder2 (3B)
StarCoder2 (7B)
StarCoder2 (15B)
DeepSeek-Coder (1.3B)
DeepSeek-Coder (6.7B)
DeepSeek-Coder (33B)
Qwen2.5-Coder (0.5B)
Qwen2.5-Coder (1.5B)
Qwen2.5-Coder (3B)
Qwen2.5-Coder (7B)
Qwen2.5-Coder (14B)
Qwen2.5-Coder (32B)
Llama 3.1 (8B)
Llama 3.1 (70B)
Code Llama (7B)
Code Llama (13B)
Code Llama (70B)
Executable implementations (t) 0.55 0.74 0.89 0.840.910.87 0.89 0.89 0.86 0.90 0.81 0.81 0.88 0.89 0.85 0.87 0.90 0.89 0.84 0.85 0.78 0.88 0.83
Correct implementations (t) 0.36 0.52 0.56 0.57 0.66 0.46 0.63 0.67 0.69 0.42 0.59 0.66 0.39 0.65 0.62 0.63 0.67 0.66 0.60 0.65 0.14 0.490.72
Correct implementations (e) 0.66 0.70 0.62 0.68 0.72 0.53 0.71 0.75 0.79 0.47 0.74 0.81 0.44 0.72 0.73 0.73 0.74 0.74 0.71 0.77 0.18 0.550.86
Hallucinated implementations (t)0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00
Hallucinated implementations (e)0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00
Correct argument names (t) 0.45 0.68 0.79 0.77 0.88 0.84 0.86 0.87 0.84 0.85 0.78 0.78 0.80 0.85 0.81 0.840.890.87 0.80 0.81 0.74 0.86 0.80
Correct argument names (e) 0.93 0.95 0.90 0.95 0.98 0.98 0.98 0.98 0.98 0.97 0.98 0.98 0.93 0.98 0.97 0.980.990.98 0.98 0.98 0.98 0.98 0.98
Correct argument values (t) 0.42 0.64 0.73 0.72 0.85 0.71 0.82 0.83 0.81 0.69 0.76 0.76 0.64 0.81 0.78 0.820.870.85 0.77 0.79 0.52 0.73 0.79
Correct argument values (e) 0.87 0.89 0.83 0.89 0.94 0.83 0.94 0.94 0.96 0.79 0.95 0.96 0.74 0.93 0.93 0.96 0.96 0.96 0.94 0.95 0.68 0.840.97
Missing arguments (t) 0.55 0.32 0.21 0.23 0.12 0.16 0.14 0.13 0.16 0.15 0.22 0.22 0.20 0.15 0.19 0.160.110.13 0.20 0.19 0.26 0.14 0.20
Missing arguments (e) 0.07 0.05 0.10 0.05 0.02 0.02 0.02 0.02 0.02 0.03 0.02 0.02 0.07 0.02 0.03 0.020.010.02 0.02 0.02 0.02 0.02 0.02
Unnecessary arguments (t)0.030.04 0.05 0.06 0.08 0.05 0.08 0.07 0.05 0.08 0.08 0.05 0.06 0.07 0.08 0.09 0.09 0.09 0.09 0.07 0.09 0.060.03
Unnecessary arguments (e) 0.06 0.05 0.06 0.08 0.09 0.06 0.09 0.08 0.06 0.09 0.10 0.06 0.07 0.07 0.10 0.10 0.10 0.10 0.11 0.08 0.12 0.070.03
Hallucinated arguments (t)0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00
Hallucinated arguments (e)0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00
Mean argument precision (t) 0.960.970.96 0.95 0.94 0.96 0.94 0.95 0.96 0.94 0.94 0.96 0.95 0.95 0.94 0.93 0.94 0.93 0.93 0.96 0.93 0.950.97
Mean argument precision (e) 0.960.970.96 0.95 0.94 0.96 0.94 0.95 0.96 0.94 0.94 0.96 0.95 0.95 0.94 0.93 0.94 0.93 0.93 0.96 0.93 0.950.97
Mean argument recall (t) 0.51 0.71 0.81 0.800.900.85 0.88 0.88 0.85 0.88 0.79 0.80 0.83 0.87 0.83 0.86 0.89 0.88 0.83 0.83 0.77 0.87 0.82
Mean argument recall (e) 0.94 0.96 0.91 0.960.990.98 0.98 0.980.990.97 0.98 0.98 0.94 0.98 0.970.99 0.99 0.990.98 0.98 0.980.99 0.99
Mean arg. Jaccard index (t) 0.50 0.69 0.79 0.770.850.82 0.83 0.84 0.82 0.83 0.75 0.77 0.79 0.83 0.79 0.80 0.84 0.82 0.77 0.80 0.72 0.84 0.80
Mean arg. Jaccard index (e) 0.91 0.93 0.88 0.91 0.93 0.95 0.93 0.94 0.94 0.92 0.93 0.95 0.90 0.93 0.92 0.92 0.93 0.92 0.92 0.94 0.92 0.950.96
Mean arg. val. cond. acc. (t) 0.95 0.95 0.93 0.94 0.96 0.83 0.97 0.96 0.97 0.79 0.970.980.79 0.96 0.970.98 0.98 0.980.97 0.97 0.65 0.860.98
Mean arg. val. cond. acc. (e) 0.95 0.95 0.93 0.94 0.96 0.83 0.97 0.96 0.97 0.79 0.970.980.79 0.96 0.970.98 0.98 0.980.97 0.97 0.65 0.860.98
Total errors 179 102 43 633652 43 43 54 40 77 74 46 42 58 53 39 42 63 61 86 46 66
Incomplete implementations 43 81 7 11516 8 11 15 20 49 45 24 15 30 10 18 9 27 17 61 26 34
Runtime errors 84 17 33 41 30 28 22 27 33 12 25 27718 21 37 15 24 19 28 23 17 29
Timeouts 9 3 3 1118 13 5 6 6 3 2 15 9 7 6 6 9 17 16 2 3 3
Unsatisfiable constraints 43 10 0 0 0 0 0 020 0 0 0 0 0 0 0 0 0 0 0 0
Manuscript submitted to ACM

Mitigating Errors in LLM-Generated Web API Invocations 49
Table 21. Relative change ofcorrect implementationsfrom vanilla generation toretrieval-augmented generation(RAG) for both setups
and both datasets. Models with a 0% baseline score are excluded from the calculation of the average gain.
(a) Full completion, synthetic dataset
Vanilla RAG Gain
CodeT5+ (6B) 0.04 0.00 -100%
CodeT5+ (16B) 0.07 0.08 +19%
StarCoder (1B) 0.03 0.22 +562%
StarCoder (3B) 0.05 0.28 +474%
StarCoder (7B) 0.12 0.34 +179%
StarCoder (15.5B) 0.13 0.36 +166%
StarCoder2 (3B) 0.13 0.33 +150%
StarCoder2 (7B) 0.10 0.14 +32%
StarCoder2 (15B) 0.25 0.33 +33%
DeepSeek-Coder (1.3B) 0.07 0.06 -21%
DeepSeek-Coder (6.7B) 0.14 0.20 +39%
DeepSeek-Coder (33B) 0.17 0.24 +39%
Qwen2.5-Coder (0.5B) 0.00 0.00 N/A
Qwen2.5-Coder (1.5B) 0.00 0.00 N/A
Qwen2.5-Coder (3B) 0.00 0.00 N/A
Qwen2.5-Coder (7B) 0.00 0.00 N/A
Qwen2.5-Coder (14B) 0.00 0.00 N/A
Qwen2.5-Coder (32B) 0.17 0.13 -20%
Llama 3.1 (8B) 0.00 0.00 N/A
Llama 3.1 (70B) 0.00 0.00 N/A
Code Llama (7B) 0.03 0.07 +155%
Code Llama (13B) 0.18 0.04 -80%
Code Llama (34B) 0.17 0.23 +39%
Code Llama (70B) 0.30 0.42 +39%
Average 0.09 0.14 +100%
Minimum 0.00 0.00 -100%
Maximum 0.30 0.42 +562%(b) Argument completion, synthetic dataset
Vanilla RAG Gain
CodeT5+ (6B) 0.24 0.05 -78%
CodeT5+ (16B) 0.21 0.09 -59%
StarCoder (1B) 0.22 0.26 +15%
StarCoder (3B) 0.23 0.30 +32%
StarCoder (7B) 0.29 0.38 +28%
StarCoder (15.5B) 0.28 0.39 +37%
StarCoder2 (3B) 0.22 0.35 +56%
StarCoder2 (7B) 0.19 0.15 -23%
StarCoder2 (15B) 0.24 0.29 +21%
DeepSeek-Coder (1.3B) 0.19 0.09 -55%
DeepSeek-Coder (6.7B) 0.29 0.23 -19%
DeepSeek-Coder (33B) 0.26 0.28 +6%
Qwen2.5-Coder (0.5B) 0.17 0.19 +10%
Qwen2.5-Coder (1.5B) 0.24 0.07 -70%
Qwen2.5-Coder (3B) 0.27 0.37 +38%
Qwen2.5-Coder (7B) 0.32 0.44 +38%
Qwen2.5-Coder (14B) 0.47 0.46 -2%
Qwen2.5-Coder (32B) 0.43 0.45 +6%
Llama 3.1 (8B) 0.10 0.09 -13%
Llama 3.1 (70B) 0.30 0.29 -6%
Code Llama (7B) 0.05 0.08 +63%
Code Llama (13B) 0.26 0.05 -83%
Code Llama (34B) 0.40 0.29 -28%
Code Llama (70B) 0.42 0.46 +9%
Average 0.26 0.25 -3%
Minimum 0.05 0.05 -83%
Maximum 0.47 0.46 +63%
(c) Full completion, real-world dataset
Vanilla RAG Gain
CodeT5+ (6B) 0.39 0.25 -36%
CodeT5+ (16B) 0.54 0.39 -27%
StarCoder (1B) 0.46 0.46 +0%
StarCoder (3B) 0.57 0.54 -6%
StarCoder (7B) 0.61 0.64 +6%
StarCoder (15.5B) 0.50 0.54 +7%
StarCoder2 (3B) 0.64 0.61 -6%
StarCoder2 (7B) 0.46 0.46 +0%
StarCoder2 (15B) 0.54 0.61 +13%
DeepSeek-Coder (1.3B) 0.57 0.68 +19%
DeepSeek-Coder (6.7B) 0.54 0.46 -13%
DeepSeek-Coder (33B) 0.54 0.68 +27%
Qwen2.5-Coder (0.5B) 0.39 0.29 -27%
Qwen2.5-Coder (1.5B) 0.00 0.00 N/A
Qwen2.5-Coder (3B) 0.00 0.00 N/A
Qwen2.5-Coder (7B) 0.07 0.04 -50%
Qwen2.5-Coder (14B) 0.00 0.04 N/A
Qwen2.5-Coder (32B) 0.18 0.21 +20%
Llama 3.1 (8B) 0.11 0.07 -33%
Llama 3.1 (70B) 0.00 0.00 N/A
Code Llama (7B) 0.61 0.57 -6%
Code Llama (13B) 0.68 0.61 -11%
Code Llama (34B) 0.39 0.43 +9%
Code Llama (70B) 0.71 0.71 +0%
Average 0.40 0.39 -6%
Minimum 0.00 0.00 -50%
Maximum 0.71 0.71 +27%(d) Argument completion, real-world dataset
Vanilla RAG Gain
CodeT5+ (6B) 0.54 0.46 -13%
CodeT5+ (16B) 0.57 0.54 -6%
StarCoder (1B) 0.68 0.57 -16%
StarCoder (3B) 0.50 0.46 -7%
StarCoder (7B) 0.57 0.57 +0%
StarCoder (15.5B) 0.57 0.64 +13%
StarCoder2 (3B) 0.50 0.61 +21%
StarCoder2 (7B) 0.39 0.46 +18%
StarCoder2 (15B) 0.50 0.54 +7%
DeepSeek-Coder (1.3B) 0.50 0.61 +21%
DeepSeek-Coder (6.7B) 0.71 0.75 +5%
DeepSeek-Coder (33B) 0.64 0.68 +6%
Qwen2.5-Coder (0.5B) 0.43 0.46 +8%
Qwen2.5-Coder (1.5B) 0.61 0.54 -12%
Qwen2.5-Coder (3B) 0.61 0.50 -18%
Qwen2.5-Coder (7B) 0.61 0.57 -6%
Qwen2.5-Coder (14B) 0.68 0.46 -32%
Qwen2.5-Coder (32B) 0.46 0.54 +15%
Llama 3.1 (8B) 0.39 0.46 +18%
Llama 3.1 (70B) 0.00 0.64 N/A
Code Llama (7B) 0.57 0.68 +19%
Code Llama (13B) 0.68 0.57 -16%
Code Llama (34B) 0.57 0.57 +0%
Code Llama (70B) 0.43 0.46 +8%
Average 0.53 0.56 +2%
Minimum 0.00 0.46 -32%
Maximum 0.71 0.75 +21%
Manuscript submitted to ACM

50 Maninger et al.
Table 22. Relative change ofcorrect implementationsfrom vanilla generation toconstrained decoding(CD) for both setups and both
datasets. Models with a 0% baseline score are excluded from the calculation of the average gain.
(a) Full completion, synthetic dataset
Vanilla CD Gain
CodeT5+ (6B) 0.04 0.17 +294%
CodeT5+ (16B) 0.07 0.18 +159%
StarCoder (1B) 0.03 0.19 +477%
StarCoder (3B) 0.05 0.32 +558%
StarCoder (7B) 0.12 0.37 +206%
StarCoder (15.5B) 0.13 0.30 +121%
StarCoder2 (3B) 0.13 0.35 +163%
StarCoder2 (7B) 0.10 0.36 +246%
StarCoder2 (15B) 0.25 0.46 +85%
DeepSeek-Coder (1.3B) 0.07 0.21 +190%
DeepSeek-Coder (6.7B) 0.14 0.42 +189%
DeepSeek-Coder (33B) 0.17 0.46 +162%
Qwen2.5-Coder (0.5B) 0.00 0.07 N/A
Qwen2.5-Coder (1.5B) 0.00 0.10 N/A
Qwen2.5-Coder (3B) 0.00 0.17 N/A
Qwen2.5-Coder (7B) 0.00 0.34 N/A
Qwen2.5-Coder (14B) 0.00 0.19 N/A
Qwen2.5-Coder (32B) 0.17 0.42 +152%
Llama 3.1 (8B) 0.00 0.23 N/A
Llama 3.1 (70B) 0.00 0.32 N/A
Code Llama (7B) 0.03 0.09 +217%
Code Llama (13B) 0.18 0.28 +59%
Code Llama (70B) 0.30 0.50 +65%
Average 0.09 0.28 +209%
Minimum 0.00 0.07 +59%
Maximum 0.30 0.50 +558%(b) Argument completion, synthetic dataset
Vanilla CD Gain
CodeT5+ (6B) 0.24 0.36 +53%
CodeT5+ (16B) 0.21 0.52 +149%
StarCoder (1B) 0.22 0.56 +150%
StarCoder (3B) 0.23 0.57 +152%
StarCoder (7B) 0.29 0.66 +124%
StarCoder (15.5B) 0.28 0.46 +63%
StarCoder2 (3B) 0.22 0.63 +183%
StarCoder2 (7B) 0.19 0.67 +242%
StarCoder2 (15B) 0.24 0.69 +185%
DeepSeek-Coder (1.3B) 0.19 0.42 +116%
DeepSeek-Coder (6.7B) 0.29 0.59 +108%
DeepSeek-Coder (33B) 0.26 0.66 +153%
Qwen2.5-Coder (0.5B) 0.17 0.39 +130%
Qwen2.5-Coder (1.5B) 0.24 0.65 +171%
Qwen2.5-Coder (3B) 0.27 0.62 +134%
Qwen2.5-Coder (7B) 0.32 0.63 +98%
Qwen2.5-Coder (14B) 0.47 0.67 +43%
Qwen2.5-Coder (32B) 0.43 0.66 +55%
Llama 3.1 (8B) 0.10 0.60 +508%
Llama 3.1 (70B) 0.30 0.65 +115%
Code Llama (7B) 0.05 0.14 +200%
Code Llama (13B) 0.26 0.49 +86%
Code Llama (70B) 0.42 0.72 +70%
Average 0.26 0.57 +143%
Minimum 0.05 0.14 +43%
Maximum 0.47 0.72 +508%
(c) Full completion, real-world dataset
Vanilla CD Gain
CodeT5+ (6B) 0.39 0.29 -27%
CodeT5+ (16B) 0.54 0.43 -20%
StarCoder (1B) 0.46 0.43 -8%
StarCoder (3B) 0.57 0.57 +0%
StarCoder (7B) 0.61 0.64 +6%
StarCoder (15.5B) 0.50 0.54 +7%
StarCoder2 (3B) 0.64 0.64 +0%
StarCoder2 (7B) 0.46 0.50 +8%
StarCoder2 (15B) 0.54 0.57 +7%
DeepSeek-Coder (1.3B) 0.57 0.57 +0%
DeepSeek-Coder (6.7B) 0.54 0.61 +13%
DeepSeek-Coder (33B) 0.54 0.68 +27%
Qwen2.5-Coder (0.5B) 0.39 0.07 -82%
Qwen2.5-Coder (1.5B) 0.00 0.04 N/A
Qwen2.5-Coder (3B) 0.00 0.25 N/A
Qwen2.5-Coder (7B) 0.07 0.64 +800%
Qwen2.5-Coder (14B) 0.00 0.57 N/A
Qwen2.5-Coder (32B) 0.18 0.54 +200%
Llama 3.1 (8B) 0.11 0.54 +400%
Llama 3.1 (70B) 0.00 0.61 N/A
Code Llama (7B) 0.61 0.61 +0%
Code Llama (13B) 0.68 0.61 -11%
Code Llama (70B) 0.71 0.61 -15%
Average 0.40 0.50 +69%
Minimum 0.00 0.04 -82%
Maximum 0.71 0.68 +800%(d) Argument completion, real-world dataset
Vanilla CD Gain
CodeT5+ (6B) 0.54 0.57 +7%
CodeT5+ (16B) 0.57 0.57 +0%
StarCoder (1B) 0.68 0.61 -11%
StarCoder (3B) 0.50 0.68 +36%
StarCoder (7B) 0.57 0.64 +13%
StarCoder (15.5B) 0.57 0.64 +13%
StarCoder2 (3B) 0.50 0.68 +36%
StarCoder2 (7B) 0.39 0.64 +64%
StarCoder2 (15B) 0.50 0.64 +29%
DeepSeek-Coder (1.3B) 0.50 0.61 +21%
DeepSeek-Coder (6.7B) 0.71 0.68 -5%
DeepSeek-Coder (33B) 0.64 0.68 +6%
Qwen2.5-Coder (0.5B) 0.43 0.57 +33%
Qwen2.5-Coder (1.5B) 0.61 0.64 +6%
Qwen2.5-Coder (3B) 0.61 0.61 +0%
Qwen2.5-Coder (7B) 0.61 0.64 +6%
Qwen2.5-Coder (14B) 0.68 0.68 +0%
Qwen2.5-Coder (32B) 0.46 0.68 +46%
Llama 3.1 (8B) 0.39 0.64 +64%
Llama 3.1 (70B) 0.00 0.64 N/A
Code Llama (7B) 0.57 0.61 +6%
Code Llama (13B) 0.68 0.61 -11%
Code Llama (70B) 0.43 0.61 +42%
Average 0.53 0.63 +18%
Minimum 0.00 0.57 -11%
Maximum 0.71 0.68 +64%
Manuscript submitted to ACM

Mitigating Errors in LLM-Generated Web API Invocations 51
Table 23. Relative change ofcorrect implementationsfrom vanilla generation toretrieval-augmented generation(RAG) combined with
constrained decoding(CD) for both setups and both datasets. Models with a 0% baseline score are excluded from the calculation of
the average gain.
(a) Full completion, synthetic dataset
Vanilla RAG + CD Gain
CodeT5+ (6B) 0.04 0.18 +324%
CodeT5+ (16B) 0.07 0.30 +333%
StarCoder (1B) 0.03 0.46 +1308%
StarCoder (3B) 0.05 0.47 +884%
StarCoder (7B) 0.12 0.51 +321%
StarCoder (15.5B) 0.13 0.43 +223%
StarCoder2 (3B) 0.13 0.54 +308%
StarCoder2 (7B) 0.10 0.26 +151%
StarCoder2 (15B) 0.25 0.63 +152%
DeepSeek-Coder (1.3B) 0.07 0.08 +10%
DeepSeek-Coder (6.7B) 0.14 0.36 +151%
DeepSeek-Coder (33B) 0.17 0.67 +281%
Qwen2.5-Coder (0.5B) 0.00 0.17 N/A
Qwen2.5-Coder (1.5B) 0.00 0.11 N/A
Qwen2.5-Coder (3B) 0.00 0.38 N/A
Qwen2.5-Coder (7B) 0.00 0.46 N/A
Qwen2.5-Coder (14B) 0.00 0.34 N/A
Qwen2.5-Coder (32B) 0.17 0.42 +148%
Llama 3.1 (8B) 0.00 0.42 N/A
Llama 3.1 (70B) 0.00 0.47 N/A
Code Llama (7B) 0.03 0.22 +671%
Code Llama (13B) 0.18 0.07 -61%
Code Llama (70B) 0.30 0.63 +108%
Average 0.09 0.37 +332%
Minimum 0.00 0.07 -61%
Maximum 0.30 0.67 +1308%(b) Argument completion, synthetic dataset
Vanilla RAG + CD Gain
CodeT5+ (6B) 0.24 0.25 +8%
CodeT5+ (16B) 0.21 0.36 +74%
StarCoder (1B) 0.22 0.55 +148%
StarCoder (3B) 0.23 0.51 +126%
StarCoder (7B) 0.29 0.58 +97%
StarCoder (15.5B) 0.28 0.48 +68%
StarCoder2 (3B) 0.22 0.60 +169%
StarCoder2 (7B) 0.19 0.28 +43%
StarCoder2 (15B) 0.24 0.71 +195%
DeepSeek-Coder (1.3B) 0.19 0.11 -44%
DeepSeek-Coder (6.7B) 0.29 0.37 +31%
DeepSeek-Coder (33B) 0.26 0.72 +177%
Qwen2.5-Coder (0.5B) 0.17 0.40 +137%
Qwen2.5-Coder (1.5B) 0.24 0.29 +23%
Qwen2.5-Coder (3B) 0.27 0.61 +128%
Qwen2.5-Coder (7B) 0.32 0.66 +107%
Qwen2.5-Coder (14B) 0.47 0.73 +57%
Qwen2.5-Coder (32B) 0.43 0.65 +52%
Llama 3.1 (8B) 0.10 0.58 +487%
Llama 3.1 (70B) 0.30 0.57 +88%
Code Llama (7B) 0.05 0.23 +384%
Code Llama (13B) 0.26 0.08 -71%
Code Llama (70B) 0.42 0.70 +66%
Average 0.26 0.48 +111%
Minimum 0.05 0.08 -71%
Maximum 0.47 0.73 +487%
(c) Full completion, real-world dataset
Vanilla RAG + CD Gain
CodeT5+ (6B) 0.39 0.36 -9%
CodeT5+ (16B) 0.54 0.39 -27%
StarCoder (1B) 0.46 0.43 -8%
StarCoder (3B) 0.57 0.46 -19%
StarCoder (7B) 0.61 0.61 +0%
StarCoder (15.5B) 0.50 0.50 +0%
StarCoder2 (3B) 0.64 0.50 -22%
StarCoder2 (7B) 0.46 0.46 +0%
StarCoder2 (15B) 0.54 0.54 +0%
DeepSeek-Coder (1.3B) 0.57 0.57 +0%
DeepSeek-Coder (6.7B) 0.54 0.64 +20%
DeepSeek-Coder (33B) 0.54 0.68 +27%
Qwen2.5-Coder (0.5B) 0.39 0.21 -45%
Qwen2.5-Coder (1.5B) 0.00 0.04 N/A
Qwen2.5-Coder (3B) 0.00 0.43 N/A
Qwen2.5-Coder (7B) 0.07 0.68 +850%
Qwen2.5-Coder (14B) 0.00 0.50 N/A
Qwen2.5-Coder (32B) 0.18 0.61 +240%
Llama 3.1 (8B) 0.11 0.57 +433%
Llama 3.1 (70B) 0.00 0.71 N/A
Code Llama (7B) 0.61 0.57 -6%
Code Llama (13B) 0.68 0.57 -16%
Code Llama (70B) 0.71 0.64 -10%
Average 0.40 0.51 +74%
Minimum 0.00 0.04 -45%
Maximum 0.71 0.71 +850%(d) Argument completion, real-world dataset
Vanilla RAG + CD Gain
CodeT5+ (6B) 0.54 0.50 -7%
CodeT5+ (16B) 0.57 0.57 +0%
StarCoder (1B) 0.68 0.61 -11%
StarCoder (3B) 0.50 0.61 +21%
StarCoder (7B) 0.57 0.61 +6%
StarCoder (15.5B) 0.57 0.61 +6%
StarCoder2 (3B) 0.50 0.61 +21%
StarCoder2 (7B) 0.39 0.61 +55%
StarCoder2 (15B) 0.50 0.61 +21%
DeepSeek-Coder (1.3B) 0.50 0.57 +14%
DeepSeek-Coder (6.7B) 0.71 0.64 -10%
DeepSeek-Coder (33B) 0.64 0.64 +0%
Qwen2.5-Coder (0.5B) 0.43 0.61 +42%
Qwen2.5-Coder (1.5B) 0.61 0.61 +0%
Qwen2.5-Coder (3B) 0.61 0.61 +0%
Qwen2.5-Coder (7B) 0.61 0.64 +6%
Qwen2.5-Coder (14B) 0.68 0.64 -5%
Qwen2.5-Coder (32B) 0.46 0.71 +54%
Llama 3.1 (8B) 0.39 0.64 +64%
Llama 3.1 (70B) 0.00 0.71 N/A
Code Llama (7B) 0.57 0.57 +0%
Code Llama (13B) 0.68 0.64 -5%
Code Llama (70B) 0.43 0.61 +42%
Average 0.53 0.62 +14%
Minimum 0.00 0.50 -11%
Maximum 0.71 0.71 +64%
Manuscript submitted to ACM

52 Maninger et al.
Table 24. Relative change ofhallucinated endpoints(full completion) andhallucinated implementations(argument completion) from
vanilla generation toretrieval-augmented generation(RAG) for both setups and both datasets. Models with a 0% baseline score are
excluded from the calculation of the average gain.
(a) Full completion, synthetic dataset
Vanilla RAG Gain
CodeT5+ (6B) 0.70 0.17 -76%
CodeT5+ (16B) 0.46 0.05 -90%
StarCoder (1B) 0.30 0.10 -68%
StarCoder (3B) 0.33 0.07 -80%
StarCoder (7B) 0.36 0.08 -78%
StarCoder (15.5B) 0.23 0.05 -80%
StarCoder2 (3B) 0.28 0.06 -78%
StarCoder2 (7B) 0.33 0.06 -83%
StarCoder2 (15B) 0.26 0.05 -81%
DeepSeek-Coder (1.3B) 0.38 0.04 -89%
DeepSeek-Coder (6.7B) 0.15 0.01 -92%
DeepSeek-Coder (33B) 0.20 0.03 -86%
Qwen2.5-Coder (0.5B) 0.00 0.00 N/A
Qwen2.5-Coder (1.5B) 0.00 0.00 N/A
Qwen2.5-Coder (3B) 0.00 0.00 N/A
Qwen2.5-Coder (7B) 0.00 0.00 N/A
Qwen2.5-Coder (14B) 0.00 0.00 N/A
Qwen2.5-Coder (32B) 0.10 0.01 -90%
Llama 3.1 (8B) 0.00 0.00 N/A
Llama 3.1 (70B) 0.00 0.00 N/A
Code Llama (7B) 0.38 0.04 -89%
Code Llama (13B) 0.30 0.05 -85%
Code Llama (34B) 0.27 0.08 -70%
Code Llama (70B) 0.27 0.04 -85%
Average 0.22 0.04 -82%
Minimum 0.00 0.00 -92%
Maximum 0.70 0.17 -68%(b) Argument completion, synthetic dataset
Vanilla RAG Gain
CodeT5+ (6B) 0.57 0.03 -95%
CodeT5+ (16B) 0.54 0.06 -90%
StarCoder (1B) 0.48 0.25 -47%
StarCoder (3B) 0.49 0.21 -56%
StarCoder (7B) 0.35 0.16 -55%
StarCoder (15.5B) 0.28 0.21 -26%
StarCoder2 (3B) 0.45 0.20 -56%
StarCoder2 (7B) 0.48 0.14 -70%
StarCoder2 (15B) 0.35 0.11 -68%
DeepSeek-Coder (1.3B) 0.50 0.19 -61%
DeepSeek-Coder (6.7B) 0.30 0.16 -47%
DeepSeek-Coder (33B) 0.40 0.07 -84%
Qwen2.5-Coder (0.5B) 0.54 0.38 -30%
Qwen2.5-Coder (1.5B) 0.52 0.19 -62%
Qwen2.5-Coder (3B) 0.44 0.27 -40%
Qwen2.5-Coder (7B) 0.41 0.12 -70%
Qwen2.5-Coder (14B) 0.26 0.04 -84%
Qwen2.5-Coder (32B) 0.28 0.13 -54%
Llama 3.1 (8B) 0.34 0.06 -83%
Llama 3.1 (70B) 0.26 0.11 -60%
Code Llama (7B) 0.19 0.10 -46%
Code Llama (13B) 0.31 0.07 -76%
Code Llama (34B) 0.26 0.21 -19%
Code Llama (70B) 0.24 0.07 -69%
Average 0.39 0.15 -60%
Minimum 0.19 0.03 -95%
Maximum 0.57 0.38 -19%
(c) Full completion, real-world dataset
Vanilla RAG Gain
CodeT5+ (6B) 0.36 0.25 -30%
CodeT5+ (16B) 0.43 0.18 -58%
StarCoder (1B) 0.43 0.32 -25%
StarCoder (3B) 0.25 0.25 +0%
StarCoder (7B) 0.29 0.29 +0%
StarCoder (15.5B) 0.29 0.18 -37%
StarCoder2 (3B) 0.29 0.18 -37%
StarCoder2 (7B) 0.36 0.32 -10%
StarCoder2 (15B) 0.32 0.29 -11%
DeepSeek-Coder (1.3B) 0.32 0.29 -11%
DeepSeek-Coder (6.7B) 0.25 0.36 +43%
DeepSeek-Coder (33B) 0.36 0.18 -50%
Qwen2.5-Coder (0.5B) 0.11 0.18 +67%
Qwen2.5-Coder (1.5B) 0.00 0.00 N/A
Qwen2.5-Coder (3B) 0.00 0.00 N/A
Qwen2.5-Coder (7B) 0.04 0.00 -100%
Qwen2.5-Coder (14B) 0.00 0.00 N/A
Qwen2.5-Coder (32B) 0.00 0.00 N/A
Llama 3.1 (8B) 0.00 0.07 N/A
Llama 3.1 (70B) 0.00 0.00 N/A
Code Llama (7B) 0.18 0.14 -20%
Code Llama (13B) 0.25 0.29 +14%
Code Llama (34B) 0.36 0.36 +0%
Code Llama (70B) 0.18 0.18 +0%
Average 0.21 0.18 -15%
Minimum 0.00 0.00 -100%
Maximum 0.43 0.36 +67%(d) Argument completion, real-world dataset
Vanilla RAG Gain
CodeT5+ (6B) 0.14 0.07 -50%
CodeT5+ (16B) 0.29 0.11 -62%
StarCoder (1B) 0.18 0.07 -60%
StarCoder (3B) 0.32 0.07 -78%
StarCoder (7B) 0.25 0.00 -100%
StarCoder (15.5B) 0.14 0.00 -100%
StarCoder2 (3B) 0.21 0.00 -100%
StarCoder2 (7B) 0.25 0.07 -71%
StarCoder2 (15B) 0.32 0.07 -78%
DeepSeek-Coder (1.3B) 0.25 0.07 -71%
DeepSeek-Coder (6.7B) 0.14 0.00 -100%
DeepSeek-Coder (33B) 0.14 0.11 -25%
Qwen2.5-Coder (0.5B) 0.32 0.18 -44%
Qwen2.5-Coder (1.5B) 0.18 0.21 +20%
Qwen2.5-Coder (3B) 0.11 0.11 +0%
Qwen2.5-Coder (7B) 0.11 0.04 -67%
Qwen2.5-Coder (14B) 0.14 0.04 -75%
Qwen2.5-Coder (32B) 0.25 0.14 -43%
Llama 3.1 (8B) 0.21 0.00 -100%
Llama 3.1 (70B) 0.00 0.00 N/A
Code Llama (7B) 0.07 0.04 -50%
Code Llama (13B) 0.11 0.00 -100%
Code Llama (34B) 0.04 0.00 -100%
Code Llama (70B) 0.11 0.04 -67%
Average 0.18 0.06 -66%
Minimum 0.00 0.00 -100%
Maximum 0.32 0.21 +20%
Manuscript submitted to ACM

Mitigating Errors in LLM-Generated Web API Invocations 53
Table 25. Relative change ofhallucinated endpoints(full completion) andhallucinated implementations(argument completion) from
vanilla generation toconstrained decoding(CD) for both setups and both datasets. Models with a 0% baseline score are excluded from
the calculation of the average gain.
(a) Full completion, synthetic dataset
Vanilla CD Gain
CodeT5+ (6B) 0.70 0.00 -100%
CodeT5+ (16B) 0.46 0.00 -100%
StarCoder (1B) 0.30 0.00 -100%
StarCoder (3B) 0.33 0.00 -100%
StarCoder (7B) 0.36 0.00 -100%
StarCoder (15.5B) 0.23 0.00 -100%
StarCoder2 (3B) 0.28 0.00 -100%
StarCoder2 (7B) 0.33 0.00 -100%
StarCoder2 (15B) 0.26 0.00 -100%
DeepSeek-Coder (1.3B) 0.38 0.00 -100%
DeepSeek-Coder (6.7B) 0.15 0.00 -100%
DeepSeek-Coder (33B) 0.20 0.00 -100%
Qwen2.5-Coder (0.5B) 0.00 0.00 N/A
Qwen2.5-Coder (1.5B) 0.00 0.00 N/A
Qwen2.5-Coder (3B) 0.00 0.00 N/A
Qwen2.5-Coder (7B) 0.00 0.00 N/A
Qwen2.5-Coder (14B) 0.00 0.00 N/A
Qwen2.5-Coder (32B) 0.10 0.00 -100%
Llama 3.1 (8B) 0.00 0.00 N/A
Llama 3.1 (70B) 0.00 0.00 N/A
Code Llama (7B) 0.38 0.00 -100%
Code Llama (13B) 0.30 0.00 -100%
Code Llama (70B) 0.27 0.00 -100%
Average 0.22 0.00 -100%
Minimum 0.00 0.00 -100%
Maximum 0.70 0.00 -100%(b) Argument completion, synthetic dataset
Vanilla CD Gain
CodeT5+ (6B) 0.57 0.00 -100%
CodeT5+ (16B) 0.54 0.00 -100%
StarCoder (1B) 0.48 0.00 -100%
StarCoder (3B) 0.49 0.00 -100%
StarCoder (7B) 0.35 0.00 -100%
StarCoder (15.5B) 0.28 0.00 -100%
StarCoder2 (3B) 0.45 0.00 -100%
StarCoder2 (7B) 0.48 0.00 -100%
StarCoder2 (15B) 0.35 0.00 -100%
DeepSeek-Coder (1.3B) 0.50 0.00 -100%
DeepSeek-Coder (6.7B) 0.30 0.00 -100%
DeepSeek-Coder (33B) 0.40 0.00 -100%
Qwen2.5-Coder (0.5B) 0.54 0.00 -100%
Qwen2.5-Coder (1.5B) 0.52 0.00 -100%
Qwen2.5-Coder (3B) 0.44 0.00 -100%
Qwen2.5-Coder (7B) 0.41 0.00 -100%
Qwen2.5-Coder (14B) 0.26 0.00 -100%
Qwen2.5-Coder (32B) 0.28 0.00 -100%
Llama 3.1 (8B) 0.34 0.00 -100%
Llama 3.1 (70B) 0.26 0.00 -100%
Code Llama (7B) 0.19 0.00 -100%
Code Llama (13B) 0.31 0.00 -100%
Code Llama (70B) 0.24 0.00 -100%
Average 0.39 0.00 -100%
Minimum 0.19 0.00 -100%
Maximum 0.57 0.00 -100%
(c) Full completion, real-world dataset
Vanilla CD Gain
CodeT5+ (6B) 0.36 0.00 -100%
CodeT5+ (16B) 0.43 0.00 -100%
StarCoder (1B) 0.43 0.00 -100%
StarCoder (3B) 0.25 0.00 -100%
StarCoder (7B) 0.29 0.00 -100%
StarCoder (15.5B) 0.29 0.00 -100%
StarCoder2 (3B) 0.29 0.00 -100%
StarCoder2 (7B) 0.36 0.00 -100%
StarCoder2 (15B) 0.32 0.00 -100%
DeepSeek-Coder (1.3B) 0.32 0.00 -100%
DeepSeek-Coder (6.7B) 0.25 0.00 -100%
DeepSeek-Coder (33B) 0.36 0.00 -100%
Qwen2.5-Coder (0.5B) 0.11 0.00 -100%
Qwen2.5-Coder (1.5B) 0.00 0.00 N/A
Qwen2.5-Coder (3B) 0.00 0.00 N/A
Qwen2.5-Coder (7B) 0.04 0.00 -100%
Qwen2.5-Coder (14B) 0.00 0.00 N/A
Qwen2.5-Coder (32B) 0.00 0.00 N/A
Llama 3.1 (8B) 0.00 0.00 N/A
Llama 3.1 (70B) 0.00 0.00 N/A
Code Llama (7B) 0.18 0.00 -100%
Code Llama (13B) 0.25 0.00 -100%
Code Llama (70B) 0.18 0.00 -100%
Average 0.20 0.00 -100%
Minimum 0.00 0.00 -100%
Maximum 0.43 0.00 -100%(d) Argument completion, real-world dataset
Vanilla CD Gain
CodeT5+ (6B) 0.14 0.00 -100%
CodeT5+ (16B) 0.29 0.00 -100%
StarCoder (1B) 0.18 0.00 -100%
StarCoder (3B) 0.32 0.00 -100%
StarCoder (7B) 0.25 0.00 -100%
StarCoder (15.5B) 0.14 0.00 -100%
StarCoder2 (3B) 0.21 0.00 -100%
StarCoder2 (7B) 0.25 0.00 -100%
StarCoder2 (15B) 0.32 0.00 -100%
DeepSeek-Coder (1.3B) 0.25 0.00 -100%
DeepSeek-Coder (6.7B) 0.14 0.00 -100%
DeepSeek-Coder (33B) 0.14 0.00 -100%
Qwen2.5-Coder (0.5B) 0.32 0.00 -100%
Qwen2.5-Coder (1.5B) 0.18 0.00 -100%
Qwen2.5-Coder (3B) 0.11 0.00 -100%
Qwen2.5-Coder (7B) 0.11 0.00 -100%
Qwen2.5-Coder (14B) 0.14 0.00 -100%
Qwen2.5-Coder (32B) 0.25 0.00 -100%
Llama 3.1 (8B) 0.21 0.00 -100%
Llama 3.1 (70B) 0.00 0.00 N/A
Code Llama (7B) 0.07 0.00 -100%
Code Llama (13B) 0.11 0.00 -100%
Code Llama (70B) 0.11 0.00 -100%
Average 0.18 0.00 -100%
Minimum 0.00 0.00 -100%
Maximum 0.32 0.00 -100%
Manuscript submitted to ACM

54 Maninger et al.
Table 26. Relative change ofhallucinated endpoints(full completion) andhallucinated implementations(argument completion) from
vanilla generation toretrieval-augmented generation(RAG) combined withconstrained decoding(CD) for both setups and both datasets.
Models with a 0% baseline score are excluded from the calculation of the average gain.
(a) Full completion, synthetic dataset
Vanilla RAG + CD Gain
CodeT5+ (6B) 0.70 0.00 -100%
CodeT5+ (16B) 0.46 0.00 -100%
StarCoder (1B) 0.30 0.00 -100%
StarCoder (3B) 0.33 0.00 -100%
StarCoder (7B) 0.36 0.00 -100%
StarCoder (15.5B) 0.23 0.00 -100%
StarCoder2 (3B) 0.28 0.00 -100%
StarCoder2 (7B) 0.33 0.00 -100%
StarCoder2 (15B) 0.26 0.00 -100%
DeepSeek-Coder (1.3B) 0.38 0.00 -100%
DeepSeek-Coder (6.7B) 0.15 0.00 -100%
DeepSeek-Coder (33B) 0.20 0.00 -100%
Qwen2.5-Coder (0.5B) 0.00 0.00 N/A
Qwen2.5-Coder (1.5B) 0.00 0.00 N/A
Qwen2.5-Coder (3B) 0.00 0.00 N/A
Qwen2.5-Coder (7B) 0.00 0.00 N/A
Qwen2.5-Coder (14B) 0.00 0.00 N/A
Qwen2.5-Coder (32B) 0.10 0.00 -100%
Llama 3.1 (8B) 0.00 0.00 N/A
Llama 3.1 (70B) 0.00 0.00 N/A
Code Llama (7B) 0.38 0.00 -100%
Code Llama (13B) 0.30 0.00 -100%
Code Llama (70B) 0.27 0.00 -100%
Average 0.22 0.00 -100%
Minimum 0.00 0.00 -100%
Maximum 0.70 0.00 -100%(b) Argument completion, synthetic dataset
Vanilla RAG + CD Gain
CodeT5+ (6B) 0.57 0.00 -100%
CodeT5+ (16B) 0.54 0.00 -100%
StarCoder (1B) 0.48 0.00 -100%
StarCoder (3B) 0.49 0.00 -100%
StarCoder (7B) 0.35 0.00 -100%
StarCoder (15.5B) 0.28 0.00 -100%
StarCoder2 (3B) 0.45 0.00 -100%
StarCoder2 (7B) 0.48 0.00 -100%
StarCoder2 (15B) 0.35 0.00 -100%
DeepSeek-Coder (1.3B) 0.50 0.00 -100%
DeepSeek-Coder (6.7B) 0.30 0.00 -100%
DeepSeek-Coder (33B) 0.40 0.00 -100%
Qwen2.5-Coder (0.5B) 0.54 0.00 -100%
Qwen2.5-Coder (1.5B) 0.52 0.00 -100%
Qwen2.5-Coder (3B) 0.44 0.00 -100%
Qwen2.5-Coder (7B) 0.41 0.00 -100%
Qwen2.5-Coder (14B) 0.26 0.00 -100%
Qwen2.5-Coder (32B) 0.28 0.00 -100%
Llama 3.1 (8B) 0.34 0.00 -100%
Llama 3.1 (70B) 0.26 0.00 -100%
Code Llama (7B) 0.19 0.00 -100%
Code Llama (13B) 0.31 0.00 -100%
Code Llama (70B) 0.24 0.00 -100%
Average 0.39 0.00 -100%
Minimum 0.19 0.00 -100%
Maximum 0.57 0.00 -100%
(c) Full completion, real-world dataset
Vanilla RAG + CD Gain
CodeT5+ (6B) 0.36 0.00 -100%
CodeT5+ (16B) 0.43 0.00 -100%
StarCoder (1B) 0.43 0.00 -100%
StarCoder (3B) 0.25 0.00 -100%
StarCoder (7B) 0.29 0.00 -100%
StarCoder (15.5B) 0.29 0.00 -100%
StarCoder2 (3B) 0.29 0.00 -100%
StarCoder2 (7B) 0.36 0.00 -100%
StarCoder2 (15B) 0.32 0.00 -100%
DeepSeek-Coder (1.3B) 0.32 0.00 -100%
DeepSeek-Coder (6.7B) 0.25 0.00 -100%
DeepSeek-Coder (33B) 0.36 0.00 -100%
Qwen2.5-Coder (0.5B) 0.11 0.00 -100%
Qwen2.5-Coder (1.5B) 0.00 0.00 N/A
Qwen2.5-Coder (3B) 0.00 0.00 N/A
Qwen2.5-Coder (7B) 0.04 0.00 -100%
Qwen2.5-Coder (14B) 0.00 0.00 N/A
Qwen2.5-Coder (32B) 0.00 0.00 N/A
Llama 3.1 (8B) 0.00 0.00 N/A
Llama 3.1 (70B) 0.00 0.00 N/A
Code Llama (7B) 0.18 0.00 -100%
Code Llama (13B) 0.25 0.00 -100%
Code Llama (70B) 0.18 0.00 -100%
Average 0.20 0.00 -100%
Minimum 0.00 0.00 -100%
Maximum 0.43 0.00 -100%(d) Argument completion, real-world dataset
Vanilla RAG + CD Gain
CodeT5+ (6B) 0.14 0.00 -100%
CodeT5+ (16B) 0.29 0.00 -100%
StarCoder (1B) 0.18 0.00 -100%
StarCoder (3B) 0.32 0.00 -100%
StarCoder (7B) 0.25 0.00 -100%
StarCoder (15.5B) 0.14 0.00 -100%
StarCoder2 (3B) 0.21 0.00 -100%
StarCoder2 (7B) 0.25 0.00 -100%
StarCoder2 (15B) 0.32 0.00 -100%
DeepSeek-Coder (1.3B) 0.25 0.00 -100%
DeepSeek-Coder (6.7B) 0.14 0.00 -100%
DeepSeek-Coder (33B) 0.14 0.00 -100%
Qwen2.5-Coder (0.5B) 0.32 0.00 -100%
Qwen2.5-Coder (1.5B) 0.18 0.00 -100%
Qwen2.5-Coder (3B) 0.11 0.00 -100%
Qwen2.5-Coder (7B) 0.11 0.00 -100%
Qwen2.5-Coder (14B) 0.14 0.00 -100%
Qwen2.5-Coder (32B) 0.25 0.00 -100%
Llama 3.1 (8B) 0.21 0.00 -100%
Llama 3.1 (70B) 0.00 0.00 N/A
Code Llama (7B) 0.07 0.00 -100%
Code Llama (13B) 0.11 0.00 -100%
Code Llama (70B) 0.11 0.00 -100%
Average 0.18 0.00 -100%
Minimum 0.00 0.00 -100%
Maximum 0.32 0.00 -100%
Manuscript submitted to ACM