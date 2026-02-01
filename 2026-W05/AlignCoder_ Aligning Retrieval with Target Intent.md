# AlignCoder: Aligning Retrieval with Target Intent for Repository-Level Code Completion

**Authors**: Tianyue Jiang, Yanli Wang, Yanlin Wang, Daya Guo, Ensheng Shi, Yuchi Ma, Jiachi Chen, Zibin Zheng

**Published**: 2026-01-27 15:23:14

**PDF URL**: [https://arxiv.org/pdf/2601.19697v1](https://arxiv.org/pdf/2601.19697v1)

## Abstract
Repository-level code completion remains a challenging task for existing code large language models (code LLMs) due to their limited understanding of repository-specific context and domain knowledge. While retrieval-augmented generation (RAG) approaches have shown promise by retrieving relevant code snippets as cross-file context, they suffer from two fundamental problems: misalignment between the query and the target code in the retrieval process, and the inability of existing retrieval methods to effectively utilize the inference information. To address these challenges, we propose AlignCoder, a repository-level code completion framework that introduces a query enhancement mechanism and a reinforcement learning based retriever training method. Our approach generates multiple candidate completions to construct an enhanced query that bridges the semantic gap between the initial query and the target code. Additionally, we employ reinforcement learning to train an AlignRetriever that learns to leverage inference information in the enhanced query for more accurate retrieval. We evaluate AlignCoder on two widely-used benchmarks (CrossCodeEval and RepoEval) across five backbone code LLMs, demonstrating an 18.1% improvement in EM score compared to baselines on the CrossCodeEval benchmark. The results show that our framework achieves superior performance and exhibits high generalizability across various code LLMs and programming languages.

## Full Text


<!-- PDF content starts -->

AlignCoder: Aligning Retrieval with Target Intent
for Repository-Level Code Completion
Tianyue Jiang1†, Yanli Wang1†, Yanlin Wang1∗, Daya Guo3, Ensheng Shi2, Yuchi Ma2, Jiachi Chen1, Zibin Zheng1
1Sun Yat-sen University, Zhuhai, China
2Huawei Cloud Computing Technologies Co., Ltd., Shenzhen, China
3Independent Researcher, China
Abstract—Repository-level code completion remains a chal-
lenging task for existing code large language models (code
LLMs) due to their limited understanding of repository-specific
context and domain knowledge. While retrieval-augmented gen-
eration (RAG) approaches have shown promise by retrieving
relevant code snippets as cross-file context, they suffer from
two fundamental problems: misalignment between the query
and the target code in the retrieval process, and the inability
of existing retrieval methods to effectively utilize the inference
information. To address these challenges, we propose AlignCoder,
a repository-level code completion framework that introduces
a query enhancement mechanism and a reinforcement learning
based retriever training method. Our approach generates multi-
ple candidate completions to construct an enhanced query that
bridges the semantic gap between the initial query and the target
code. Additionally, we employ reinforcement learning to train
an AlignRetriever that learns to leverage inference information
in the enhanced query for more accurate retrieval. We evaluate
AlignCoder on two widely-used benchmarks (CrossCodeEval and
RepoEval) across five backbone code LLMs, demonstrating an
18.1% improvement in EM score compared to baselines on the
CrossCodeEval benchmark. The results show that our framework
achieves superior performance and exhibits high generalizability
across various code LLMs and programming languages.
Index Terms—Repository-Level Code Completion, Query En-
hancement, Reinforcement Learning, code LLMs
I. INTRODUCTION
Recent developments in code large language models (code
LLMs) [1]–[4] have demonstrated impressive capability in
general code completion tasks [5]–[7], [7]–[10]. However,
existing code LLMs demonstrate suboptimal performance on
repository-level code completion tasks, primarily due to their
insufficient understanding of repository-specific context and
domain knowledge [11]. This limitation stems from the fact
that target code repositories are often newly created, propri-
etary, or work-in-progress projects, making it hard for code
LLMs to acquire repository-specific knowledge during pre-
training and fine-tuning phases [12]. To address this chal-
lenge, one straightforward approach leverages the increasing
context window length of modern models by concatenating
all repository files into a single prompt. However, this naive
concatenation introduces substantial irrelevant information that
interferes with model generation [13], [14]. Consequently,
* Yanlin Wang is the corresponding author, wangylin36@mail.sysu.edu.cn.
†These authors contributed equally to this work.recent methods have adopted the retrieval-augmented gener-
ation (RAG) paradigm [12], [15]–[20], which uses unfinished
code in the current file as a query to retrieve relevant code
snippets from the entire repository. These retrieved code
snippets serve as cross-file context and are concatenated with
the unfinished code to construct prompts for code LLMs.
For instance, ReACC [15] integrates both sparse and dense
retrieval methods. Sparse retrievers, such as BM25 [21],
employ keyword matching algorithms that effectively capture
lexical information. Conversely, dense retrievers encode both
queries and code snippets into dense vectors, enabling the
identification of semantically similar code snippets through
vector similarity measurements. Despite these advances, most
dense retrieval methods fail to leverage the reasoning and un-
derstanding capabilities of code LLMs to enhance the retrieval
process, resulting in a semantic gap between query and target
code in the retrieval process. To mitigate this misalignment,
RepoCoder [16] proposes an iterative retrieval strategy where
the generator produces intermediate completions based on
retrieved code snippets, which are then incorporated into
subsequent retrieval queries. While this approach represents
a significant step toward addressing the query-target misalign-
ment, the fundamental issue remains unresolved. Specifically,
we have identified the following problems in existing retrieval
processes:
P1 The misalignment between query and target code
remains unresolved.
To address the problem of misalignment between query
and target code in RAG-based code completion, Re-
poCoder [16] introduces an iterative retrieval approach
that concatenates the completion with the unfinished code
to obtain a new query for the next retrieval round. How-
ever, this method has two problems: If LLMs produce
an incorrect completion in an iteration, it will cause
chain errors that affect the subsequent retrieval. Besides,
multiple retrieval rounds significantly reduce efficiency.
P2 Existing retrieval methods lack the ability to learn
how to utilize inference information.
Although RepoCoder has demonstrated that the generated
completion can significantly assist repository-level code
completion, the method employs sparse (e.g., Jaccard
index [22]) or dense (e.g., UniXcoder [23]) retrievers thatarXiv:2601.19697v1  [cs.SE]  27 Jan 2026

are not specifically trained. These retrievers may fail to
understand the relationship between the unfinished code
and the candidate completion, and therefore cannot fully
leverage it for effective retrieval.
In this paper, we propose a repository-level code completion
framework AlignCoder to address the two aforementioned
problems. First, we introduce a query enhancement mechanism
that leverages sampled candidate completions to improve
retrieval accuracy. Specifically, our approach employs the
sampler to generate multiple candidate completions, which are
then expanded to the unfinished code to construct an enhanced
query representation. This enhanced query effectively bridges
the gap between the initial query and the desired target com-
pletion (addressingP1). Secondly, we employ reinforcement
learning to train the retriever, enabling it to learn how to utilize
the multiple candidate completions contained in the enhanced
query for more accurate retrieval (addressingP2). Specifically,
given an enhanced query, we employ the retriever to retrieve
multiple potentially relevant code snippets. We then utilize a
reward model to evaluate the perplexity (PPL) of generating
the target code using these retrieved code snippets, deriving
rewards from this evaluation process to update the retriever’s
parameters.
We evaluate AlignCoder with extensive experiments using
five backbone LLMs on two benchmarks: CrossCodeEval [24]
and RepoEval [16]. These two benchmarks are widely used in
repository-level code completion. Experimental results show
that our framework achieves an 18.1% improvement in EM
score compared with baselines on the CrossCodeEval Python.
AlignCoder demonstrates high generalizability, showing ef-
fectiveness across various code LLMs and programming lan-
guages.
To summarize, our main contributions are:
•We introduce AlignCoder, a repository-level code com-
pletion framework. The proposed query enhancement
mechanism allows the enhanced query to have a greater
possibility of including key tokens relevant to the target
code. This framework effectively addresses the misalign-
ment problem between query and target code in the
retrieval process.
•We train the retriever using reinforcement learning, re-
sulting in Alignretriever. This retriever learns to leverage
the inference information in enhanced queries to achieve
more accurate retrieval.
•We perform extensive experimental evaluation on vari-
ous benchmarks and code LLMs. The results show that
AlignCoder achieves superior performance compared to
previous approaches. We provide our code and data at
https://anonymous.4open.science/r/AlignCoder.
II. PRELIMINARIES
A. Retrieval-Augmented Code Completion Paradigms
Retrieval-augmented methods have been widely adopted
for repository-level code completion. Such methods can be
categorized into two primary paradigms:1) Retrieve-then-Generate:This approach first retrieves rel-
evant code snippets from the codebase using the unfinished
code as a query, then uses the retrieved code snippets to
assist the code completion. Methods like ReACC [15] and
RLCoder [25] follow this paradigm. The retrieval process
takes the unfinished codeC uas input to search through the
repositoryRfor the top-kmost similar code snippetsS, which
are then provided to the language modelP θas context for
generating the target completion ˆCt. However, this paradigm
faces the challenge of semantic misalignment. These methods
only use unfinished code as a query, which may not contain
the key tokens related to the target code. Consequently, it
becomes difficult to retrieve the relevant code snippets needed
to generate the target code during the retrieval process [16].
2) Iterative Generate-and-Retrieve:This approach alternates
between generation and retrieval in multiple iterations. Starting
with an initial retrieval using the unfinished code, the method
generates a candidate completion and then uses the concate-
nation of the unfinished code and the generated candidate
as a new query for the next retrieval iteration. Methods like
RepoCoder [16] employ this iterative approach, where each
iteration refines both the retrieved context and the generated
completion. However, since each iteration relies on a single
candidate completion, errors can propagate through the chain.
If errors occur in the intermediate generation step or key
tokens are still missing, the subsequent retrieval will be based
on flawed information, leading to cascading errors throughout
the remaining iterations.
B. Semantic Gap in Repository Code Retrieval
Repository-level code completion leverages contextual in-
formation across multiple files to generate accurate code
completions. Formally, given an unfinished codeC uand a
repositoryR, the objective is to generate target codeC tsuch
thatC=C u⊕Ctis syntactically and semantically correct. A
challenge in this task stems from the semantic gap between
unfinished code and target code. UsingC udirectly as a query
is suboptimal because unfinished code and target code belong
to different semantic spaces, making alignment difficult [26].
The semantic gap can be formalized as :
G(Cu, Cr) =d(Φ query(Cu),Φtarget (Cr))(1)
whereΦ query andΦ target be embedding functions mapping
code to semantic spaces anddis a distance function in the se-
mantic space. Previous approaches like RLCoder [25] attempt
to bridge this gap through reinforcement learning, while Re-
poCoder [16] uses iterative retrieval-generation to update the
query. However, the former requires semantic space alignment
between unfinished code and target code, which is inherently
challenging due to the differences in their representations. The
latter relies on a single sample from each iteration, which can
easily propagate errors in a chain reaction—if one iteration
produces an incorrect completion, subsequent iterations are
built upon this flawed foundation, leading to compounding
errors throughout the retrieval-generation process.

ReACC / RLCoder# src/routes/users.pyfrom src.database.db import get_dbfrom src.database.models importUser, UserRole, BlacklistTokenfrom src.repository importusers asrepository_usersfrom src.services.auth import auth_service...router = APIRouter(prefix="/users", tags=["users"])allowed_operation_get = RoleAccess([UserRole.Admin, UserRole.Moderator, UserRole.User])...  user = awaitrepository_users.Query (UnfinishedCode)
update_user(body, user, db)Groundtruth# src/routes/auth.py@router.post("/signup", response_model=UserResponse, status_code=status.HTTP_201_CREATED)async def signup(body: UserModel, request: Request, db: Session = Depends(get_db)):  exist_user = awaitrepository_users.get_user_by_email(body.email, db)  if exist_user:    raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Account already exists")  body.password_checksum = auth_service.pwd_context.hash(body.password_checksum)  new_user = await repository_users.create_user(body, db)  return new_userBad Reference CodeRetrievalGenerationCodeSnippetsUnfinishedCode
PredictionRepoCoder
Missing Key Tokens in Groundtruth
May Cause Chain Error
Fig. 1. Motivating example. Limitations of prior works: ReACC and RLCoder use the unfinished code as the query, which may miss key tokens in the ground
truth. RepoCoder may cause chain errors and has potential efficiency issue.
C. Multiple Sampling in LLMs
LLMs exhibit inherent stochasticity in their generation
process, making single-sample outputs unreliable. The prob-
abilistic nature of token sampling introduces variability that
can lead to inconsistent or incorrect responses, even when
queried with identical prompts. However, through the multiple
sampling strategy, we can significantly improve the likelihood
of obtaining correct completions.
1) Single Sample Unreliability:For a given queryq, an
LLM generates a response by sampling from the learned
probability distribution over the vocabularyVat each timestep.
Letp θ(y|x)denote the probability of generating sequencey
given inputxand model parametersθ. The sampling process
can be formulated as:
yt∼pθ(·|x, y <t)(2)
wherey trepresents the token at positiont, andy <tdenotes
all previously generated tokens.
Due to the stochastic nature of sampling, a single generation
attempt has probabilityp sof producing a correct answer,
typicallyp s<1and may be lower for complex queries.
Threrefore, the probability of generating an incorrect response
in a single attempt is:
P(error) = 1−p s (3)
2) Multiple Sampling Benefits:When performing multiple
sampling attempts, we acknowledge that individual samples
are not truly independent events, as they originate from the
same model with identical parameters and input prompt.
However, due to the stochastic nature of the sampling process
(temperature-based sampling, top-k, or nucleus sampling),we can approximate the samples as quasi-independent for
analytical purposes.
Letρdenote the correlation coefficient between samples,
where0≤ρ≤1. For truly independent samples (ρ= 0),
the probability of obtaining at least one correct answer from
nattempts would be:
Pindependent (at least one correct) = 1−(1−p s)n(4)
However, accounting for inter-sample correlation, the actual
probability can be approximated as:
P(at least one correct)≈1−(1−p s)n·(1−ρ)(5)
where the effective number of independent samples is
reduced by the correlation factor(1−ρ).
In practice, modern sampling techniques (such as temper-
ature scalingT >0or top-p sampling) introduce sufficient
randomness thatρremains relatively small, making the inde-
pendence approximation reasonable:
P(at least one correct)≈1−(1−p s)nwhenρ≪1(6)
This relationship demonstrates that even with correlated
samples, the success probability increases substantially with
the number of sampling attempts. For practical applications,
multiple sampling strategy consistently yields higher accuracy
than single sampling, establishing it as an effective strategy
for improving LLM reliability in critical applications.
The above provides an introduction to the theoretical foun-
dations. In the following part, we demonstrate the effectiveness
of multiple sampling strategy compared to single sampling
through a preliminary experiment. This experiment utilizes the
CrossCodeEval and RepoEval, where the prompt consists of
BM25 retrieval code snippets concatenated with unfinished

1 2 3 4 5 6 7 8 9 10
k in EM@k0.150.200.250.300.35EM@k
CrossCodeEval (Python)
CrossCodeEval (Java)
RepoEval (Line)
RepoEval (API)Fig. 2. EM@k performance trends on CrossCodeEval and RepoEval with
varying k parameters. The trends indicate that multiple sampling strategy
provides a higher probability of producing a correct completion compared
tosingle sampling.
code. We present theEM@kresults (k ranging from 1 to
10) for DeepSeekCoder-1B using the aforementioned prompt
with 10 sampling attempts. Specifically,EM@krepresents the
probability of that at least one of the k samples exactly matches
the target code (EM=1). As shown in figure 2, theEM@k
values on both benchmarks increase with the value of k. This
indicates that multiple sampling strategy provides a higher
probability of generating a correct completion compared to
single sampling, which also means it is easier to contain key
tokens related to the correct completion.
3) Diminishing Returns and Optimal Sampling Threshold:
While multiple sampling generally improves the probability
of obtaining correct completions, excessive sampling can in-
troduce diminishing returns and potentially counterproductive
effects. As the number of samples increases beyond an optimal
threshold, several factors contribute to degraded performance:
Error Accumulation:With increased sampling numbers,
the absolute number of incorrect responses grows, potentially
overwhelming correct completions during aggregation or se-
lection processes. If we defineϵ nas the cumulative error rate
afternsamples, we have:
ϵn=n·(1−p s)(7)
Selection Complexity:The probability of selecting the cor-
rect answer from a pool containing both correct and incorrect
responses depends on the selection mechanism. For random
selection fromnsamples, whereksamples are correct, the
probability of selecting a correct answer is:
P(correct selection) =k
n=n·p s
n=ps (8)
However, for more sophisticated selection mechanisms (e.g.,
majority voting, confidence-based selection), the relationship
becomes more complex.
Optimal Sampling Threshold:To determine the optimal
number of samplesn∗, we must balance the benefit of in-creased correct sample probability against the cost of error
accumulation. The expected utility can be formulated as:
U(n) =α·P(at least one correct)−β·ϵ n−γ·n(9)
whereα,β, andγrepresent the weights for correctness benefit,
error penalty, and computational cost, respectively.
Substituting our previous formulations:
U(n) =α·[1−(1−p s)n]−β·n(1−p s)−γ·n(10)
Taking the derivative with respect tonand setting it to zero:
dU(n)
dn=α·(1−p s)nln(1−p s)−β(1−p s)−γ= 0(11)
The optimal sampling thresholdn∗can be approximated by
solving:
n∗≈ln
β(1−p s)+γ
αln(1−p s)
ln(1−p s)(12)
This theoretical framework demonstrates that while multi-
ple sampling is beneficial, there exists an optimal threshold
beyond which additional samples provide marginal utility and
may even degrade overall system performance due to increased
complexity in answer selection and computational overhead.
D. Perplexity in LLMs for Code Assessment
Perplexity (PPL) quantifies how well a probability model
predicts token sequences, serving as an intrinsic measure of
model confidence in Large Language Models. For a token
sequencey= (y 1, y2, . . . , y T), perplexity is defined as:
PPL(y) = exp 
−1
TTX
t=1logp θ(yt|y<t)!
(13)
wherep θ(yt|y<t)represents the conditional probability of
tokeny tgiven preceding contexty <t. Lower perplexity indi-
cates higher model confidence.
While not directly measuring correctness, perplexity cor-
relates with code quality through several mechanisms: (1)
Statistical regularity:well-formed code follows learned pat-
terns from training data; (2)Syntactic consistency:valid syntax
exhibits predictable token transitions; (3)Semantic coherence:
logical code patterns yield lower perplexity.
However, perplexity has limitations: uncommon but correct
coding styles, domain-specific patterns, and model understand-
ing gaps can cause discrepancies between perplexity and actual
correctness. Empirical studies show moderate positive corre-
lation between low perplexity and code correctness [27], [28],
making it effective as a first-order heuristic for ranking code
completion candidates when combined with other validation
methods.

Reference
Snippets
LLM -based
GeneratorLLM -based
Evaluator
Inference StageTraining Stage
Completed
Code
PPL-based Reward
RLTrainingRL-based Training & InferenceReference
SnippetsLLM -based
SamplerMultiple
Candidates
Enhanced
Query
Code Repository
Codebase
Base
SnippetsDependency
Snippets
Unfinished Code
(Initial Query)
Coarse -Grained Retrieval
Sparse
Retriever
Fine -Grained Retrieval
AlignRetriever
Developer
Reward Design
Ⅱ
Ⅲ
Query Enhancement Codebase Construction
Ⅰ
Multiple Sampling StrategyFig. 3. Overview of AlignCoder. Our approach utilizes the inference capability of the LLM to sample multiple candidate completions and learn to understand
the inference information for accurate retrieval for repository-level code completion.
III. METHODOLOGY
In this section, we introduce AlignCoder, a framework for
repository-level code completion. The pipline of AlignCoder
is shown in Figure 3, which contains three stages. 1Given
a code repository, we extract two types of code snippets to
construct the codebase for retrieval. 2For the unfinished
code needs to be completed, we use it as the original query to
perform the coarse-grained retrieval. Then we use the retrieved
code snippets concatenated with the unfinished code as the
prompt for sampler. The sampler is a lightweight LLM to
sample multiple candidate completions. Then we concatenate
the unfinished code with the sampled candidate completions
to serve as the enhanced query. 3We use the enhanced
query to perform fine-grained retrieval, which returns several
reference code snippets. In the training phase, the reward
model will give each reference code snippet a reward to
evaluate the helpfulness. The parameter of the retriever will be
updated by the reward, which is calculated through an LLM-
based evaluator. In the inference phase, we concatenate the
retrieved code snippets with the unfinished code to construct
the prompt, based on which the generator performs the final
code completion.
A. Codebase Construction
In constructing the retrieval codebase, we construct two
distinct types of code snippets, namely base and dependency
code snippets. Each category of code snippets is tailored to
represent a corresponding type of cross-file information. The
first type is the base context, which consists of initial code
segments within the cross-files. The second type is dependency
context, which provides deep semantic understanding of class
hierarchies and API interactions within the codebase [29].
The following section provides a detailed introduction to the
# example_cfg.py
...
sampled_token , _ = generator.sample_current (logits_mixed )
ifsampled_token.item () == tokenizer.eos_token_id :
break
batch_token = sampled_token.repeat (2, 1)
generator.Initial Query (Unfinished Code)
sequence = torch.cat(( generator.sequence , batch_token ), dim = 1)# Candidate Completion 1
gen_accept_token (batch_token )# Candidate Completion 2
gen_accept_token (batch_token.unsqueeze (0))# Candidate Completion 3Good CompletionBad Completion
Similar CompletionEnhanced Query
Enhanced Query Can Use More Potential CompletionFig. 4. Example: the initial query and the enhanced query obtained with the
query enhancement mechanism. Task id:project_cc_python/74.
construction methodologies for base and dependency code
snippets.
1) Base Code Snippets Construction:Previous works con-
struct retrieval codebase using fixed window sizes [16] and
dependency parsing approaches [12], [30], [31]. However,
fixed window strategies may disrupt code continuity, while
dependency parsing-based methods may only focus on limited
context in the context graph and struggle to be applied to
complex scenarios, such as when repository dependencies are
highly intricate. Therefore, we adopt the Split-Aggregate strat-
egy [25] to construct base code snippets. This approach is in-
spired by human programming habits, first dividing the cross-
files into mini-blocks based on blank lines, then aggregating
these mini-blocks into code snippets according to a predefined
length. Formally, given a cross-fileF={l 1, l2, . . . , l n}where
lirepresents thei-th line, we first splitFinto mini-blocks

based on blank lines:
Split(F) ={B 1, B2, . . . , B k}whereB j={l s, . . . , l e}(14)
where each mini-blockB jis a contiguous sequence of non-
empty lines. Subsequently, we aggregate these mini-blocks
into code snippets:
Si=Aggregate({B j, Bj+1, . . . , B j+m})s.t.|S i| ≤L(15)
whereS irepresents thei-th standard code snippet,Lis the
predefined maximum line counts, and|S i|denotes the line
counts of snippetS i. The aggregation process ensures that:
j+mX
k=j|Bk| ≤Landj+m+1X
k=j|Bk|> L(16)
2) Dependency Code Snippets Construction:We use tree-
sitter [32] to extract import statements from in-file context and
parse each statement to obtain module name, entity name, and
alias. We filter out standard library and third-party imports,
retaining only intra-repository module references. Formally,
given import statementsI={i 1, i2, . . . , i n}:
Iintra ={i∈I|m(i)/∈(StdLib∪ThirdParty)}(17)
wherem(i)extracts the module name of importi. The intra-
repository references are categorized into class importsI c,
method importsI m, and function importsI f.
We then parse cross-files using tree-sitter and extract match-
ing code bodies based on these references. We refer to the
corresponding code bodies as entities [12], [33], denoted by
u. For function and method imports (i∈I f∪Im):
Dependency(i) =Signature(u i)(18)
For class imports (i∈I c):
Dependency(i) ={σ c,Σm,Σnc,Σnm}(19)
whereσ cis the main class signature,Σ mcontains all class
method signatures,Σ nccontains nested class signatures, and
Σnmcontains nested class method signatures. The complete
dependency information is:
D=[
i∈IintraDependency(i)(20)
We construct dependency code snippets based on extracted
information, treating each imported class’s dependency infor-
mation as an individual dependency code snippet. For imported
methods or functions, we aggregate their signatures into a
single snippet. Finally, we combine base code snippets and
dependency code snippets to form the retrieval codebase.
B. Query Enhancemant Mechanism
If the query used for retrieval only contains the unfinished
code, it is likely to retrieve code snippets that are similar to
them. This raises a gap between the query and the target code,
potentially leading to reduced retrieval accuracy. To address
this issue, we propose a query enhancement mechanism. The
details of the mechanism are illustrated below.Our query enhancement mechanism consists of two phases:
coarsed-grained retrieval and multiple sampling strategy. (1) In
the coarse-grained retrieval, we use BM25 method to retrieve
code snippets from the codebase using the unfinished code
as the initial query. These code snippets may include both
base and dependency code snippets, providing the original
code information and deep semantic understanding of class
hierarchies and API interactions. (2) These code snippets are
then concatenated with the unfinished code to build a prompt.
The sampler samples k candidate completions based on this
prompt. Finally, these candidate completions are appended to
the initial query to construct an enhanced query, which guides
the retriever in the fine-grained retrieval phase. For example,
Figure 4 presents a taskproject_cc_python/74with
the initial query and the enhanced query after processing
through the query enhancemant mechanism. In the task, we
need to complete a specific line in example cfg.py where a
method is called on thegeneratorobject. Thegenerator
object is an instance of theExLlamaGeneratorclass
defined ingenerator.py, and the target code is a method
get_accept_tokendefined in this class. Among the can-
didate completions sampled by the sampler, candidate comple-
tion1is an incorrect completion, while candidate completion
2is the correct completion, and candidate completion3is
a completion similar to the target code. We consider that
candidate completions2and3contain key tokens related to
the target code. The enhanced query may improve the ability
of the retriever to retrieve more relevant code snippets in the
following fine-grained retrieval phase.
C. AlignRetriever: RL-based Retriever Aglignment Training
1) Training data construction:We randomly selected
10,000 Python and Java repositories from GitHub. These
repositories contain cross-file dependencies and were created
before March 2023 and are not included in our evaluation
benchmarks, CrossCodeEval [24] and RepoEval [16], ensuring
no data leakage and maintaining the fairness of the evaluation.
Constructing training data involves the following three steps:
1We divide each code repository into a series of clusters,
where the code files within these clusters have interdependent
relationships. We exclude clusters that contain only a single
file. 2Then, we perform topological sorting on the remaining
clusters. This sorting is based on the in-degree and out-degree
relationships between the files. The final sorting result ensures
that : The first code file is dependent upon other files in the
same cluster, but does not contain dependencies on other files.
All subsequent code files depend on one or more other files
within the cluster. 3For each cluster, we randomly select
one non-first file to construct the target code for completion.
To ensure the preceding context of the target code contains
sufficient information, we avoid selecting starting positions of
the target code from either the beginning or the end of the
current file. The length of the target code is also randomly
determined (set to be from 16 to 96 tokens in our experiment),
and the entire code segment must lie within the boundaries of
the current file.

2) The training objective function:The reward mechanism
in reinforcement learning provides the model with environ-
mental feedback, allowing it to progressively learn particular
capabilities based on this feedback signal. We define the
reward function as Reward(·), where the reward function needs
to be maximized during AlignRetriever training period. The
detailed mathematical expression for Reward(·)is:
Reward=nX
i=1I(ci)×logexp(s i,q)Pn
j=1exp(s j,q)(21)
whereqis the enhanced query,C={c 1, c2, . . . , c n}is the set
of retrieved code snippets,nis the number of code snippets,
I(ci)is the indicator function denoting correctness of snippet
ci, ands i,q= cos(emb(c i),emb(q))represents the cosine
similarity between embeddings of code snippetc iand query
q. The definition ofI(c i)is:
I(ci) =(
1,ifc i=c mp
0,otherwise(22)
wherec mp∈Cand satisfies:
PPL(t|q, c mp)≤PPL(t|q, c i)∀i∈ {1, . . . , n}(23)
where the formula PPL(t|q, c i)is defined as:
PPL(t|x, c i) =e−1
LPL
j=1logP(t j|ci,q,t<j)(24)
whereLdenotes the total number of tokens in the target code.
IV. EXPERIMENTAL SETUP
In this section, we introduce the benchmarks, backbone
models, baseline methods, and evaluation metrics used in our
experiments.
A. Benchmarks and Backbone Models
In our evaluation, we use two benchmarks for repository-
level code completion tasks: CrossCodeEval [24] and Repo-
Eval [16]. These two benchmarks have been widely used in
previous work [4], [25], [33]–[35].
•CrossCodeEval: CrossCodeEval is a diverse and mul-
tilingual code completion benchmark that requires a
deep understanding of context across different files in
the repository to ensure accurate code completion. We
conduct the evaluation on Python and Java sets.
•RepoEval: RepoEval is allowed for assessments at three
granularities, line, API invocation, and function body.
In our evaluation, we focus on the line-level and API
invocation tasks.
In our experiments, we select five code LLMs as gen-
erators. These code LLMs have been proven to perform
well on repository-level code completion tasks in previous
work [16], [17], [25], [29], [36]. These five code LLMs are:
CodeLlama-7B [37], StartCoder-7B [38], StarCoder2-7B [1],
DeepSeekCoder-1B [34], and DeepSeekCoder-7B [34].B. Baseline Methods
In our experiments, we compare AlignCoder with previous
RAG-based methods, encompassing ReACC, RepoCoder, and
RLCoder.
•ReACC: ReACC [15] adopt the hybrid retriever [39], [40]
framework by combining sparse and dense retriever.
•RepoCoder: RepoCoder [16] is an iterative retrieval and
generation framework, where it searches for the relevant
code snippets using the output generated by code LLMs
from the previous iteration.
•RLCoder: RLCoder [25] is a reinforcement learning
framework, which can enable the retriever to learn to
retrieve useful snippets without the need for labeled data.
C. Evaluation Metrics
Following the established practice [12], [16], [25], [33], we
use two metrics, Exact Match (EM) and Edit Similarity (ES),
to evaluate code completion accuracy.
D. Experimental Details
We performed all experiments on a machine configured with
2 NVIDIA Tesla A100 GPUs, each with 80 GB of memory.
•Training Stage: In the training stage, we initialize the
retriever using UniXcoder and use DeepSeekCoder-1B
as the evaluator. We trained the retriever for a total of 20
epochs. Each epoch utilized 3,000 samples for training,
and the learning rate was set to 5e-5.
•Inference Stage: In the inference stage, since our
approach employs multiple sampling, we utilize the
vLLM [41] framework to accelerate model inference.
V. EXPERIMENTALRESULTS
We aims to answer the following research questions (RQs):
•RQ1: How effective is AlignCoder in repository-level
code completion?
•RQ2: The effectiveness of multiple sampling strategies,
and what the optimal sampling numbers should be?
•RQ3: What is the contribution of each AlignCoder com-
ponent to its performance?
•RQ4: What is the robustness of AlignCoder’s parameter
settings?
A. RQ1: Overall Performance of AlignCoder
In this section, we study the effectiveness of AlignCoder
compared with three baseline methods. Table I presents the
performance comparison of AlignCoder against three baseline
methods across multiple benchmarks and backbone models.
The results demonstrate that AlignCoder outperforms previous
methods across all evaluation settings.
From the perspective of benchmarks, AlignCoder achieves
greater improvements on CrossCodeEval compared to RepoE-
val. For the CrossCodeEval Python, AlignCoder demonstrates
substantial performance gains, with EM score improvements
ranging from 12.7% to 18.1% compared to RLCoder. For
the CrossCodeEval Java, AlignCoder maintains its superiority

TABLE I
PERFORMANCE COMPARISON. SUPERSCRIPTED PERCENTAGES REPRESENT THE IMPROVEMENT OVER THE CORRESPONDING BEST BASELINE.
Method/ModelCrossCodeEval (Python) CrossCodeEval (Java) RepoEval (Line) RepoEval (API)
EM ES EM ES EM ES EM ES
ReACC CodeLlama-7B 21.76 69.09 23.42 66.13 42.31 64.35 34.38 61.45
RepoCoder CodeLlama-7B 23.34 70.84 24.17 66.56 43.94 65.81 37.00 63.51
RLCoder CodeLlama-7B 26.64 72.26 26.27 67.60 46.63 67.86 37.94 64.32
AlignCoder CodeLlama-7B 30.13↑13.1%74.29↑2.8%28.80↑9.6%68.38↑1.2%46.96↑0.7%67.92↑0.1%39.56↑4.3%66.01↑2.6%
ReACC StarCoder-7B 22.33 69.60 22.16 67.80 43.81 64.83 31.94 56.00
RepoCoder StarCoder-7B 23.15 70.71 22.53 68.22 45.69 66.90 33.44 57.81
RLCoder StarCoder-7B 26.00 72.16 25.76 68.80 47.81 68.50 35.06 58.08
AlignCoder StarCoder-7B 30.43↑17.0%74.33↑3.0%28.00↑8.7%70.04↑1.8%48.38↑1.2%68.56↑0.1%36.25↑3.4%59.49↑2.4%
ReACC StarCoder2-7B 22.89 70.66 23.42 69.13 44.44 65.95 34.50 58.78
RepoCoder StarCoder2-7B 24.35 71.71 23.75 69.59 45.81 67.37 36.44 59.92
RLCoder StarCoder2-7B 27.47 73.39 26.69 70.35 48.63 68.59 37.75 61.08
AlignCoder StarCoder2-7B 31.74↑15.5%75.70↑3.1%30.43↑14.0%72.64↑3.3%48.81↑0.4%68.82↑0.3%38.69↑2.5%61.59↑0.8%
ReACC DeepSeekCoder-1B 19.74 67.68 18.89 62.47 39.31 62.04 33.00 60.41
RepoCoder DeepSeekCoder-1B 20.23 68.78 19.59 62.35 40.88 63.56 35.13 61.92
RLCoder DeepSeekCoder-1B 24.02 70.45 20.66 63.17 44.06 66.05 36.00 62.50
AlignCoder DeepSeekCoder-1B 28.37↑18.1%73.02↑3.6%23.24↑12.5%64.58↑2.2%44.69↑1.4%66.74↑1.0%37.25↑3.5%64.43↑3.1%
ReACC DeepSeekCoder-7B 23.30 70.84 22.49 66.78 45.69 66.67 38.00 65.66
RepoCoder DeepSeekCoder-7B 26.98 72.96 24.96 66.52 46.38 67.51 39.31 66.29
RLCoder DeepSeekCoder-7B 30.09 74.43 26.37 67.28 48.81 69.48 39.75 66.01
AlignCoder DeepSeekCoder-7B 33.92↑12.7%76.97↑3.4%28.28↑7.2%68.31↑1.5%49.56↑1.5%69.93↑0.6%41.88↑5.4%67.75↑2.6%
with EM score improvements of 7.2% to 14.0% compared to
RLCoder across backbone models.
From the perspective of models, results show that maximal
improvement is achieved by DeepSeekCoder-1B on Cross-
CodeEval Python (EM: +18.1%, ES: +3.6%). For Cross-
CodeEval Java, StarCoder2-7B demonstrated the highest im-
provement (EM: +14.0%, ES: +3.3%). Regarding RepoEval
performance, DeepSeekCoder-7B showed the most significant
gains. It achieved EM score improvements of 1.5% on line-
level tasks and 5.4% on API-level tasks.
RQ1 Summary:AlignCoder consistently outperforms all
baseline methods across different benchmarks and backbone
models. The best-performing setting improves 18.1% on the
EM score, with particularly gains on CrossCodeEval Python.
B. RQ2: Mutiple Sampling Strategy
In this section, we investigate two questions: (1) whether
multiple sampling is more effective than single sampling, and
(2) what the optimal sampling numbers for AlignCoder. We
conduct performance comparison experiments with sampling
numbers set from 1 to 6. The generator used in this experiment
is DeepSeekCoder-1B. The experimental results are presented
in Table II.
(1) Effectiveness of Multiple Sampling.Multiple sampling
(2, 3, and 4 samples) generally outperforms single sampling
in EM and ES scores. The few exceptions show only minimal
decreases: 0.2% lower EM on API-level tasks of RepoEval
(sampling numbers set to 2), for instance. These results
indicate that multiple sampling enhances the likelihood of
generating key tokens related to target code, thus improving
the retrieval performance.(2) Performance degradation in certain datasets when
sampling numbers exceed a threshold.A performance
degradation is observed on certain datasets when sampling
numbers surpass 4. When the sampling number is set to 5,
the results on CrossCodeEval Java and line-level/API-level
tasks of RepoEval are all inferior to single sampling. When
the sampling number is set to 6, there is a decline compared
to single sampling on CrossCodeEval Java and line-level tasks
of RepoEval. This indicates that although multiple sampling
is effective compared to single sampling, excessive sampling
may cause the candidate completions to contain non-negligible
noise, which could affect the accuracy of subsequent retrieval.
(3) Determining the optimal sampling numbers.We
evaluate the performance of AlignCoder by computing av-
erage EM and ES scores under different sampling settings.
As shown in Table II, the setting of sampling numbers at
4 demonstrates superior performance. This evidence estab-
lishes sampling numbers of 4 as the optimal balance for our
approach, maximizing performance gains while maintaining
computational efficiency.
RQ2 Summary:We demonstrate the effectiveness of the
multiple sampling strategy compared to single sampling.
When the sampling number exceeds a threshold, Align-
Coder’s performance declines. Experimental results show
that the optimal sampling number is 4.
C. RQ3: Ablation Studies
To demonstrate the effectiveness of the three core compo-
nents in AlignCoder, we conduct experiments focusing on:
(1) incorporating dependency context as cross-file information
by constructing and retrieving dependency code snippets; (2)

TABLE II
SUPERSCRIPTED PERCENTAGES INDICATE THE RELATIVE INCREASE OR DECREASE OF MULTI-SAMPLING COMPARED TO SINGLE-SAMPLING.
MethodCrossCodeEval (Python) CrossCodeEval (Java) RepoEval (Line) RepoEval (API) Average
EM ES EM ES EM ES EM ES EM ES
Sampling Number=1 27.20 71.98 23.05 64.49 44.75 66.47 36.69 63.39 32.92 66.58
Sampling Number=2 27.65↑1.7%72.41↑0.6%23.42↑1.6%65.45↑1.5%45.19↑1.0%66.98↑0.8%36.63↓0.2%63.90↑0.8%33.22↑0.9%67.19↑0.9%
Sampling Number=3 28.29↑4.0%72.71↑1.0%23.19↑0.6%63.93↓0.9%44.75↑0.0%66.35↓0.2%36.88↑0.5%63.93↑0.9%33.28↑1.1%66.73↑0.2%
Sampling Number=4 28.37↑4.3%73.02↑1.5%23.24↑0.8%64.58↑0.1%44.69↑-0.1%66.74↑0.4%37.25↑1.5%64.43↑1.6%33.39↑1.4%67.19↑0.9%
Sampling Number=528.44↑4.6%71.68↓0.4%22.81↓1.0%64.10↓0.6%44.37↓0.9%66.44↓0.1%36.63↓0.2%63.43↑0.1%33.06↑0.4%66.41↓0.3%
Sampling Number=6 28.07↑3.2%72.54↑0.8%22.87↓0.8%64.18↓0.5%44.63↓0.3%66.38↓0.1%37.81↑3.1%64.07↑1.1%33.35↑1.3%66.79↑0.3%
TABLE III
ABLATION STUDY RESULTS.
MethodCrossCodeEval (Python) CrossCodeEval (Java)
EM ES EM ES
AlignCoder 33.92 76.97 28.28 68.31
w/o DC 31.44↓7.3%75.82↓1.5%27.44↓3.0%67.54↓1.1%
w/o QH 31.33↓7.6%75.21↓2.3%27.12↓4.1%67.63↓1.0%
w/o RL 28.14↓17.4%73.39↓4.7%25.62↓9.4%66.85↓2.1%
MethodRepoEval (Line) RepoEval (API)
EM ES EM ES
AlignCoder 49.56 69.93 41.88 67.75
w/o DC 49.06↓1.0%69.96↑0.0%40.44↓3.4%66.75↓1.5%
w/o QH 48.94↓1.3%69.63↓0.4%40.19↓4.0%66.52↓1.8%
w/o RL 46.56↓6.1%68.06↓2.7%38.63↓7.8%65.10↓3.9%
0.60 0.65 0.70 0.75 0.80 0.853040506070CrossCodeEval
Python/uni00A0EM
Python/uni00A0ES
Java/uni00A0EM
Java/uni00A0ES
0.70 0.75 0.80 0.85 0.90 0.953040506070
Python/uni00A0EM
Python/uni00A0ES
Java/uni00A0EM
Java/uni00A0ES
0.60 0.65 0.70 0.75 0.80 0.85
Temperature405060RepoEval
Line/uni00A0EM
Line/uni00A0ES
API/uni00A0EM
API/uni00A0ES
0.70 0.75 0.80 0.85 0.90 0.95
Top/uni00ADp405060
Line/uni00A0EM
Line/uni00A0ES
API/uni00A0EM
API/uni00A0ES
Fig. 5. Temperature and top-psampling parameter effects on AlignCoder
performance.
employing a query enhancement mechanism; and (3) using
reinforcement learning to train the retriever to learn to utilize
the inference information in the enhanced query. We conduct
ablation studies on the CrossCodeEval and RepoEval. All
experiments are implemented with DeepSeekCoder-7B as the
generator, with detailed results shown in Table III.
Table III shows three ablation settings. w/o DC indicates
that the dependency context is not considered. w/o QH denotes
the configuration without the query enhancement mechanism
applied. w/o RL indicates that reinforcement learning is notapplied to the retriever, so the retriever has not learned how
to utilize the key tokens in the enhanced query.
Our ablation study reveals performance patterns: (1) when
w/o DC and w/o QH, both EM and ES scores consistently
decrease across benchmarks. The only exception is a negligible
0.08% improvement in ES for RepoEval line-level tasks in the
w/o DC setting. (2) Performance degradation becomes more
pronounced under w/o RL. The experimental results show that
w/o RL has a significant impact on AlignCoder’s performance,
which demonstrates that merely considering dependency con-
text and query enhancement mechanism is insufficient, and it
is also necessary to train the retriever to learn to utilize the
additional information in the query.
RQ3 Summary:Our ablation studies demonstrate the
critical roles of three components. Results show that w/o
RL causes greater performance drops. This indicates that
dependency context and query enhancement are incomplete
without training the retriever to exploit the additional infor-
mation in the enhanced query.
D. RQ4: Sampling Parameter Stability
In this section, we analyze the influence of two sampling pa-
rameters, temperature and top-p, on AlignCoder’s performance
when sampling candidate completions. Temperature and top-
pare two crucial parameters that control the randomness
and diversity of generation. Temperature regulates the degree
of randomness in generated text, while top-pconstrains the
candidate vocabulary by sampling only from the highest-
probability tokens whose cumulative probability reaches the
specifiedpvalue. Since we employ the vLLM framework to
accelerate model inference, the temperature and top-psettings
in AlignCoder follow the default parameter configurations
for the model sampling process as specified in the vLLM
documentation, with temperature set to 0.8 and top-pset to
0.95. We conducted a series of experiments to examine the
stability of AlignCoder’s performance varying temperature
(0.85, 0.75, 0.7, 0.65, 0.6) and top-p(0.9, 0.85, 0.8, 0.75,
0.7) values when using the optimal sampling number 4.
Our experiments used DeepSeekCoder-1B as the generator,
Figure 5 demonstrates that parameter adjustments typically
lead to modest performance changes. Most configurations
exhibit EM and ES score variations of less than 1% across
both benchmarks. However, five configurations show differ-

# python/packager/pep517.py
fromtyping importAny, Dict, Iterator, Optional, Union
from.build_config importBuildConfiguration
from.nativelib importlocate_local_libtl2cgen, 
locate_or_build_libtl2cgen
...
defbuild_wheel(
wheel_directory : str,
config_settings : Optional[ Dict[str, Any]] = None,
metadata_directory : Optional[str] = None,
) -> str:
"""Build a wheel"""
logger = logging.getLogger ("tl2cgen.packager.build_wheel" )
build_config= BuildConfiguration ()
build_config.Initial Query (Unfinished Code)
# candidate completion 1:
init_build_configuration ()
# candidate completion 2:
load_from_build_sdist(wheel_directory)
# candidate completion 3:
initialize_from_config_settings(config_settings)
# candidate completion 4:
update(config_settings)Candidate Completions# Potential invoked APIs:
classBuildConfiguration : 
def_set_config_setting (self, config_settings : Dict[str, Any]) -> None:
defupdate(self, config_settings : Optional[ Dict[str, Any]]) -> None:
defget_cmake_args (self) -> List[str]:Good Reference Code Snippet
# file path: nativelib.py
defbuild_libtl2cgen( cpp_src_dir : pathlib.Path , build_dir : pathlib.Path , 
build_config: BuildConfiguration ,) -> pathlib.Path :
"""Build libtl2cgen in a temporary directory and obtain the path to 
built libtl2cgen"""
logger = logging.getLogger ("tl2cgen.packager.build_libtl2cgen" )
ifnotcpp_src_dir.is_dir ():
raiseRuntimeError (f"Expected {cpp_src_dir }to be a directory" )
logger.info( "Building %s from the C++ source files in %s..." , 
_lib_name (), ste(cpp_src_dir ))Bad Reference Code SnippetEnhanced Query
update(config_settings ) AlignCoderload_config () RLCoderinit_build_configuration () RepoCoder
1stReference Code Snippet For the Three MethodsFig. 6. Case study, task id:project ccpython/5407. The highlighted words represent identical tokens that exist in both the query and the reference code.
ences: a 2.4% EM decrease on CrossCodeEval Python under
temperature at 0.8 and top-pat 0.7, for instance.
RQ4 Summary:AlignCoder demonstrates robust per-
formance under different temperature and top-p sampling
configurations, consistently exhibiting stability in EM and
ES metrics across most experimental configurations.
E. Case Study
We illustrate the effectiveness of AlignCoder through a
case study. In the Figure 6, the model needs to complete the
functionupdateof theBuildConfigurationclass. The
highest similarity code snippet retrieved through AlignCoder is
a dependency code snippet related to theValidatorclass.
Based on this snippet, the generator is able to generate the
correct completion. However, the most similar code snippet
retrieved by RLCoder or RepoCoder is a base code snippet
highly similar to the unfinished code, ultimately leading the
model to produce an incorrect result.
VI. RELATEDWORK
A. Code Completion
In the field of intelligent software engineering, large lan-
guage models (LLMs) and agents are leveraged to address
complex development challenges [42]–[49]. The field has
witnessed significant progress in code generation [25], [50]–
[64], code search [65]–[81], issue resolution [82]–[89], code
summarization [90], [91], code translation [92]–[96], com-
mit message generation [97]–[105], efficient model optimiza-
tion [8], [106]–[111], and code understanding tasks [45], [47],
[81], [102], [112]–[115]. These methodologies have collec-
tively advanced our understanding of how AI systems can
effectively support software engineering tasks across various
domains. Auto code completion has been a fundamental taskin intelligent software engineering, aiming to predict subse-
quent code tokens or statements to assist programmers during
development [7], [116]. Traditional approaches relied on rule-
based methods or statistical methods [117]–[120], but recent
advances have been driven by deep learning techniques. Mod-
ern neural approaches [121]–[123] have significantly improved
completion accuracy by learning from large codebases. The
emergence of large language models has further revolution-
ized this field, with studies exploring LLM-based completion
systems [27], [124]. Retrieval-augmented generation (RAG)
has become particularly influential, with methods like Red-
Coder [125] enhancing code generation relevant code snippets
retrieval, DocPrompting [126] leveraging documentation for
unseen functions, and AceCoder [127] integrating examples
retrieval with guided generation to improve completion quality.
B. Repository-Level Code Completion
Repository-level code completion extends traditional com-
pletion by leveraging broader repository context to im-
prove accuracy and relevance [16], [17], [29], [30], [128],
[129]. Existing approaches can be categorized into several
paradigms. Learning-based methods such as CoCoMIC [30]
and RepoHyper [17] employ dependency analysis and adap-
tive learning mechanisms, though they face challenges in
training data acquisition and generalizability. Static analysis
approaches, including CodePlan [128], RepoFuse [29], and
A3-CodeGen [129], utilize program analysis techniques to
identify relevant code candidates. GraphCoder [12] captures
the context by leveraging the structural information in the
source code via a constructed code context graph. Iterative
retrieval strategies have been explored by RepoCoder [16],
which performs multi-round retrieval and generation, and De-
Hallucinator [130], which refines completion through iterative

processes. Tool-augmented methods such as CodeAgent [131]
and ToolGen [132] investigate external tool invocation, while
RLCoder [25] employs reinforcement learning for repository-
specific retrieval optimization.
Despite these advances, AlignCoder differs by leveraging
LLMs’ powerful inference capabilities to obtain multiple ref-
erence code snippets, combining this with a reinforcement
learning-based approach for training the retriever to achieve
more accurate retrieval and superior repository-level code
completion performance.
VII.CONCLUSION
In this paper, we present AlignCoder, a repository-level
code completion framework that addresses the fundamental
challenges of query-target misalignment and ineffective in-
ference information utilization in existing retrieval-augmented
generation approaches. Our key contributions include a query
enhancement mechanism that generates multiple candidate
completions to bridge the semantic gap between queries
and target codes, and a reinforcement learning-based Align-
retriever that learns to leverage inference information for
more precise retrieval. Extensive experiments on CrossCodeE-
val and RepoEval benchmarks across five backbone models
demonstrate the effectiveness of our approach, achieving an
18.1% improvement in EM score on the Python subset of
CrossCodeEval and showing strong generalizability across
various code LLMs and programming languages. This work
contributes to the development of repository-level code gen-
eration tools and may help improve developer productivity in
programming tasks.
ACKNOWLEDGEMENTS
This work is supported by CCF-Huawei Populus Grove
Fund CCF-HuaweiSE202403.
REFERENCES
[1] A. Lozhkov, R. Li, L. B. Allal, F. Cassano, J. Lamy-Poirier, N. Tazi,
A. Tang, D. Pykhtar, J. Liu, Y . Weiet al., “Starcoder 2 and the stack
v2: The next generation,”arXiv preprint arXiv:2402.19173, 2024.
[2] B. Roziere, J. Gehring, F. Gloeckle, S. Sootla, I. Gat, X. E. Tan, Y . Adi,
J. Liu, R. Sauvestre, T. Remezet al., “Code llama: Open foundation
models for code,”arXiv preprint arXiv:2308.12950, 2023.
[3] D. Guo, Q. Zhu, D. Yang, Z. Xie, K. Dong, W. Zhang, G. Chen, X. Bi,
Y . Wu, Y . Liet al., “Deepseek-coder: When the large language model
meets programming–the rise of code intelligence,”arXiv preprint
arXiv:2401.14196, 2024.
[4] B. Hui, J. Yang, Z. Cui, J. Yang, D. Liu, L. Zhang, T. Liu, J. Zhang,
B. Yu, K. Luet al., “Qwen2. 5-coder technical report,”arXiv preprint
arXiv:2409.12186, 2024.
[5] D. Zan, B. Chen, F. Zhang, D. Lu, B. Wu, B. Guan, Y . Wang, and J.-G.
Lou, “Large language models meet nl2code: A survey,”arXiv preprint
arXiv:2212.09420, 2022.
[6] Z. Zhang, C. Chen, B. Liu, C. Liao, Z. Gong, H. Yu, J. Li, and R. Wang,
“Unifying the perspectives of nlp and software engineering: A survey
on language models for code,”arXiv preprint arXiv:2311.07989, 2023.
[7] M. Izadi, J. Katzy, T. Van Dam, M. Otten, R. M. Popescu, and
A. Van Deursen, “Language models for code completion: A practi-
cal evaluation,” inProceedings of the IEEE/ACM 46th International
Conference on Software Engineering, 2024, pp. 1–13.
[8] D. Guo, C. Xu, N. Duan, J. Yin, and J. McAuley, “Longcoder:
A long-range pre-trained language model for code completion,” in
International Conference on Machine Learning. PMLR, 2023, pp.
12 098–12 107.[9] Q. Ren, C. Gao, J. Shao, J. Yan, X. Tan, W. Lam, and L. Ma,
“Codeattack: Revealing safety generalization challenges of large lan-
guage models via code completion,”arXiv preprint arXiv:2403.07865,
2024.
[10] D. Shrivastava, D. Kocetkov, H. de Vries, D. Bahdanau, and T. Scholak,
“Repofusion: Training code models to understand your repository,”
arXiv preprint arXiv:2306.10998, 2023.
[11] Z. Tang, J. Ge, S. Liu, T. Zhu, T. Xu, L. Huang, and B. Luo,
“Domain adaptive code completion via language models and decoupled
domain databases,” in2023 38th IEEE/ACM International Conference
on Automated Software Engineering (ASE). IEEE, 2023, pp. 421–433.
[12] W. Liu, A. Yu, D. Zan, B. Shen, W. Zhang, H. Zhao, Z. Jin, and
Q. Wang, “Graphcoder: Enhancing repository-level code completion
via code context graph-based retrieval and language model,”arXiv
preprint arXiv:2406.07003, 2024.
[13] F. Shi, X. Chen, K. Misra, N. Scales, D. Dohan, E. H. Chi, N. Sch ¨arli,
and D. Zhou, “Large language models can be easily distracted by
irrelevant context,” inInternational Conference on Machine Learning.
PMLR, 2023, pp. 31 210–31 227.
[14] O. Yoran, T. Wolfson, O. Ram, and J. Berant, “Making retrieval-
augmented language models robust to irrelevant context,”arXiv
preprint arXiv:2310.01558, 2023.
[15] S. Lu, N. Duan, H. Han, D. Guo, S.-w. Hwang, and A. Svyatkovskiy,
“Reacc: A retrieval-augmented code completion framework,”arXiv
preprint arXiv:2203.07722, 2022.
[16] F. Zhang, B. Chen, Y . Zhang, J. Keung, J. Liu, D. Zan, Y . Mao,
J.-G. Lou, and W. Chen, “Repocoder: Repository-level code com-
pletion through iterative retrieval and generation,”arXiv preprint
arXiv:2303.12570, 2023.
[17] H. N. Phan, H. N. Phan, T. N. Nguyen, and N. D. Bui, “Repohyper:
Better context retrieval is all you need for repository-level code
completion,”CoRR, 2024.
[18] J. Wang, Y . He, and H. Chen, “Repogenreflex: Enhancing repository-
level code completion with verbal reinforcement and retrieval-
augmented generation,”arXiv preprint arXiv:2409.13122, 2024.
[19] T.-D. Bui, D.-T. Luu-Van, T.-P. Nguyen, T.-T. Nguyen, S. Nguyen, and
H. D. V o, “Rambo: Enhancing rag-based repository-level method body
completion,”arXiv preprint arXiv:2409.15204, 2024.
[20] M. Liang, X. Xie, G. Zhang, X. Zheng, P. Di, W. Jiang, H. Chen,
C. Wang, and G. Fan, “Repogenix: Dual context-aided repository-
level code completion with language models,” inProceedings of the
39th IEEE/ACM International Conference on Automated Software
Engineering, 2024, pp. 2466–2467.
[21] S. Robertson, H. Zaragozaet al., “The probabilistic relevance frame-
work: Bm25 and beyond,”Foundations and Trends® in Information
Retrieval, vol. 3, no. 4, pp. 333–389, 2009.
[22] P. Jaccard, “The distribution of the flora in the alpine zone. 1,”New
phytologist, vol. 11, no. 2, pp. 37–50, 1912.
[23] D. Guo, S. Lu, N. Duan, Y . Wang, M. Zhou, and J. Yin, “Unix-
coder: Unified cross-modal pre-training for code representation,”arXiv
preprint arXiv:2203.03850, 2022.
[24] Y . Ding, Z. Wang, W. Ahmad, H. Ding, M. Tan, N. Jain, M. K.
Ramanathan, R. Nallapati, P. Bhatia, D. Rothet al., “Crosscodeeval:
A diverse and multilingual benchmark for cross-file code completion,”
Advances in Neural Information Processing Systems, vol. 36, 2024.
[25] Y . Wang, Y . Wang, D. Guo, J. Chen, R. Zhang, Y . Ma, and Z. Zheng,
“Rlcoder: Reinforcement learning for repository-level code comple-
tion,”arXiv preprint arXiv:2407.19487, 2024.
[26] L. Wang, N. Yang, and F. Wei, “Query2doc: Query expansion
with large language models,” 2023. [Online]. Available: https:
//arxiv.org/abs/2303.07678
[27] M. Chen, J. Tworek, H. Jun, Q. Yuan, H. P. D. O. Pinto, J. Kaplan,
H. Edwards, Y . Burda, N. Joseph, G. Brockmanet al., “Evaluating large
language models trained on code,”arXiv preprint arXiv:2107.03374,
2021.
[28] J. Austin, A. Odena, M. Nye, M. Bosma, H. Michalewski, D. Dohan,
E. Jiang, C. Cai, M. Terry, Q. Leet al., “Program synthesis with large
language models,”arXiv preprint arXiv:2108.07732, 2021.
[29] M. Liang, X. Xie, G. Zhang, X. Zheng, P. Di, H. Chen, C. Wang,
G. Fanet al., “Repofuse: Repository-level code completion with fused
dual context,”arXiv preprint arXiv:2402.14323, 2024.
[30] Y . Ding, Z. Wang, W. U. Ahmad, M. K. Ramanathan, R. Nallapati,
P. Bhatia, D. Roth, and B. Xiang, “Cocomic: Code completion by
jointly modeling in-file and cross-file context,” 2023.

[31] J. Liu, Y . Chen, M. Liu, X. Peng, and Y . Lou, “Stall+: Boosting
llm-based repository-level code completion with static analysis,”arXiv
preprint arXiv:2406.10018, 2024.
[32] M.Brunsfeld, P.Thomson, A.Hlynskyi, J.Vera, P.Turnbull, T.Clem,
D.Creager, A.Helwer, R.Rix, H. Antwerpen, M.Davis, Ika, T.-
A.Nguyen, S.Brunk, N.Hasabnis, bfredl, M.Dong, V .Panteleev,
ikrima, S.Kalt, K.Lampe, A.Pinkus, M.Schmitz, M.Krupcale, narpfel,
S.Gallegos, V .Mart ´ı, Edgar, and G.Fraser, “Tree-sitter,” site:https://
github.com/tree-sitter/tree-sitter, 2020, accessed: 2023-11-03.
[33] W. Cheng, Y . Wu, and W. Hu, “Dataflow-guided retrieval aug-
mentation for repository-level code completion,”arXiv preprint
arXiv:2405.19782, 2024.
[34] D. Guo, Q. Zhu, D. Yang, Z. Xie, K. Dong, W. Zhang, G. Chen, X. Bi,
Y . Wu, Y . Liet al., “Deepseek-coder: When the large language model
meets programming–the rise of code intelligence,”arXiv preprint
arXiv:2401.14196, 2024.
[35] Y . Wei, Z. Wang, J. Liu, Y . Ding, and L. Zhang, “Magicoder:
Empowering code generation with oss-instruct,”arXiv preprint
arXiv:2312.02120, 2023.
[36] D. Wu, W. U. Ahmad, D. Zhang, M. K. Ramanathan, and X. Ma,
“Repoformer: Selective retrieval for repository-level code completion,”
arXiv preprint arXiv:2403.10059, 2024.
[37] B. Roziere, J. Gehring, F. Gloeckle, S. Sootla, I. Gat, X. E. Tan, Y . Adi,
J. Liu, T. Remez, J. Rapinet al., “Code llama: Open foundation models
for code,”arXiv preprint arXiv:2308.12950, 2023.
[38] R. Li, L. B. Allal, Y . Zi, N. Muennighoff, D. Kocetkov, C. Mou,
M. Marone, C. Akiki, J. Li, J. Chimet al., “Starcoder: may the source
be with you!”arXiv preprint arXiv:2305.06161, 2023.
[39] V . Karpukhin, B. Oguz, S. Min, P. Lewis, L. Wu, S. Edunov,
D. Chen, and W.-t. Yih, “Dense passage retrieval for open-domain
question answering,” inProceedings of the 2020 Conference on
Empirical Methods in Natural Language Processing (EMNLP),
B. Webber, T. Cohn, Y . He, and Y . Liu, Eds. Online: Association
for Computational Linguistics, Nov. 2020, pp. 6769–6781. [Online].
Available: https://aclanthology.org/2020.emnlp-main.550/
[40] X. Ma, K. Sun, R. Pradeep, and J. Lin, “A replication study
of dense passage retriever,” 2021. [Online]. Available: https:
//arxiv.org/abs/2104.05740
[41] W. Kwon, Z. Li, S. Zhuang, Y . Sheng, L. Zheng, C. H. Yu, J. Gonzalez,
H. Zhang, and I. Stoica, “Efficient memory management for large
language model serving with pagedattention,” inProceedings of the
29th Symposium on Operating Systems Principles, 2023, pp. 611–626.
[42] Z. Zheng, K. Ning, Q. Zhong, J. Chen, W. Chen, L. Guo, W. Wang,
and Y . Wang, “Towards an understanding of large language models in
software engineering tasks,”Empirical Software Engineering, vol. 30,
no. 2, p. 50, 2025.
[43] K. Yang, X. Mao, S. Wang, Y . Wang, T. Zhang, B. Lin, Y . Qin,
Z. Zhang, Y . Lu, and K. Al-Sabahi, “Large language models are
qualified benchmark builders: Rebuilding pre-training datasets for
advancing code intelligence tasks,”arXiv preprint arXiv:2504.19444,
2025.
[44] Z. Zheng, K. Ning, Y . Wang, J. Zhang, D. Zheng, M. Ye, and J. Chen,
“A survey of large language models for code: Evolution, benchmarking,
and future trends,”arXiv preprint arXiv:2311.10372, 2023.
[45] Y . Wang, K. Duan, D. Zheng, E. Shi, F. Zhang, Y . Wang, J. Chen,
X. Liu, Y . Ma, H. Zhanget al., “Towards an understanding of context
utilization in code intelligence,”arXiv preprint arXiv:2504.08734,
2025.
[46] Y . Wang, W. Zhong, Y . Huang, E. Shi, M. Yang, J. Chen, H. Li, Y . Ma,
Q. Wang, and Z. Zheng, “Agents in software engineering: Survey,
landscape, and vision,”Automated Software Engineering, vol. 32, no. 2,
p. 70, 2025.
[47] J. Zhou, W. Zhong, Y . Wang, and J. Wang, “Adaptive-solver framework
for dynamic strategy selection in large language model reasoning,”
Information Processing & Management, vol. 62, no. 3, p. 104052,
2025.
[48] J. Chen, C. Chen, J. Hu, J. Grundy, Y . Wang, T. Chen, and Z. Zheng,
“Identifying smart contract security issues in code snippets from stack
overflow,” inProceedings of the 33rd ACM SIGSOFT International
Symposium on Software Testing and Analysis, 2024, pp. 1198–1210.
[49] S. Yang, X. Lin, J. Chen, Q. Zhong, L. Xiao, R. Huang,
Y . Wang, and Z. Zheng, “Hyperion: Unveiling dapp inconsistencies
using llm and dataflow-guided symbolic execution,”arXiv preprint
arXiv:2408.06037, 2024.[50] E. Shi, F. Zhang, Y . Wang, B. Chen, L. Du, H. Zhang, S. Han,
D. Zhang, and H. Sun, “Sotana: The open-source software development
assistant,”arXiv preprint arXiv:2308.13416, 2023.
[51] Y . Li, E. Shi, D. Zheng, K. Duan, J. Chen, and Y . Wang, “Re-
pomincoder: Improving repository-level code generation based on
information loss screening,” inProceedings of the 15th Asia-Pacific
Symposium on Internetware, 2024, pp. 229–238.
[52] Y . Wang, T. Jiang, M. Liu, J. Chen, M. Mao, X. Liu, Y . Ma, and
Z. Zheng, “Beyond functional correctness: Investigating coding style
inconsistencies in large language models,”Proceedings of the ACM on
Software Engineering, vol. 2, no. FSE, pp. 690–712, 2025.
[53] D. Zheng, Y . Wang, E. Shi, X. Liu, Y . Ma, H. Zhang, and
Z. Zheng, “Top general performance= top domain performance? do-
maincodebench: A multi-domain code generation benchmark,”arXiv
preprint arXiv:2412.18573, 2024.
[54] Z. Zhang, C. Wang, Y . Wang, E. Shi, Y . Ma, W. Zhong, J. Chen,
M. Mao, and Z. Zheng, “Llm hallucinations in practical code gen-
eration: Phenomena, mechanism, and mitigation,”Proceedings of the
ACM on Software Engineering, vol. 2, no. ISSTA, pp. 481–503, 2025.
[55] D. Li, S. Cao, C. Cao, X. Li, S. Tan, K. Keutzer, J. Xing, J. E.
Gonzalez, and I. Stoica, “S*: Test time scaling for code generation,”
arXiv preprint arXiv:2502.14382, 2025.
[56] S. Quan, J. Yang, B. Yu, B. Zheng, D. Liu, A. Yang, X. Ren, B. Gao,
Y . Miao, Y . Fenget al., “Codeelo: Benchmarking competition-level
code generation of llms with human-comparable elo ratings,”arXiv
preprint arXiv:2501.01257, 2025.
[57] C. Si, Y . Zhang, R. Li, Z. Yang, R. Liu, and D. Yang, “Design2code:
Benchmarking multimodal code generation for automated front-end
engineering,” inProceedings of the 2025 Conference of the Nations of
the Americas Chapter of the Association for Computational Linguistics:
Human Language Technologies (Volume 1: Long Papers), 2025, pp.
3956–3974.
[58] W. Gu, J. Chen, Y . Wang, T. Jiang, X. Li, M. Liu, X. Liu, Y . Ma,
and Z. Zheng, “What to retrieve for effective retrieval-augmented
code generation? an empirical study and beyond,”arXiv preprint
arXiv:2503.20589, 2025.
[59] J. Chen, Q. Zhong, Y . Wang, K. Ning, Y . Liu, Z. Xu, Z. Zhao, T. Chen,
and Z. Zheng, “Rmcbench: Benchmarking large language models’
resistance to malicious code,” inProceedings of the 39th IEEE/ACM
International Conference on Automated Software Engineering, 2024,
pp. 995–1006.
[60] D. Zheng, Y . Wang, E. Shi, R. Zhang, Y . Ma, H. Zhang, and
Z. Zheng, “Humanevo: An evolution-aware benchmark for more re-
alistic evaluation of repository-level code generation,”arXiv preprint
arXiv:2406.06918, 2024.
[61] Y . Wang and H. Li, “Code completion by modeling flattened abstract
syntax trees as graphs,” inProceedings of the AAAI conference on
artificial intelligence, vol. 35, no. 16, 2021, pp. 14 015–14 023.
[62] Y . Lai, S. Lee, G. Chen, S. Poddar, M. Hu, D. Z. Pan, and P. Luo,
“Analogcoder: Analog circuit design via training-free code generation,”
inProceedings of the AAAI Conference on Artificial Intelligence,
vol. 39, no. 1, 2025, pp. 379–387.
[63] Q. Zhu, J. Cao, Y . Lu, H. Lin, X. Han, L. Sun, and S.-C. Cheung,
“Domaineval: An auto-constructed benchmark for multi-domain code
generation,” inProceedings of the AAAI Conference on Artificial
Intelligence, vol. 39, no. 24, 2025, pp. 26 148–26 156.
[64] L. Nie, J. Sun, Y . Wang, L. Du, S. Han, D. Zhang, L. Hou, J. Li,
and J. Zhai, “Unveiling the black box of plms with semantic anchors:
towards interpretable neural semantic parsing,” inProceedings of the
AAAI conference on artificial intelligence, vol. 37, no. 11, 2023, pp.
13 400–13 408.
[65] W. Gu, E. Shi, Y . Wang, L. Du, S. Han, H. Zhang, D. Zhang, and
M. R. Lyu, “Secret: Towards scalable and efficient code retrieval via
segmented deep hashing,”arXiv preprint arXiv:2412.11728, 2024.
[66] W. Gu, Z. Lyu, Y . Wang, H. Zhang, C. Gao, and M. R. Lyu, “Spencer:
Self-adaptive model distillation for efficient code retrieval,”ACM
Transactions on Software Engineering and Methodology, 2025.
[67] J. Gong, Y . Wu, L. Liang, Y . Wang, J. Chen, M. Liu, and Z. Zheng,
“Cosqa+: Enhancing code search evaluation with a multi-choice bench-
mark and test-driven agents,”IEEE Transactions on Software Engineer-
ing, vol. 52, no. 1, pp. 206–220, 2025.
[68] W. Gu, Y . Wang, L. Du, H. Zhang, S. Han, D. Zhang, and M. Lyu,
“Accelerating code search with deep hashing and code classification,”
inProceedings of the 60th Annual Meeting of the Association for

Computational Linguistics (Volume 1: Long Papers), 2022, pp. 2534–
2544.
[69] F. Hu, Y . Wang, L. Du, X. Li, H. Zhang, S. Han, and D. Zhang,
“Revisiting code search in a two-stage paradigm,” inProceedings of
the sixteenth ACM international conference on Web search and data
mining, 2023, pp. 994–1002.
[70] E. Shi, Y . Wang, W. Gu, L. Du, H. Zhang, S. Han, D. Zhang, and
H. Sun, “Cocosoda: Effective contrastive learning for code search,” in
2023 IEEE/ACM 45th International Conference on Software Engineer-
ing (ICSE). IEEE, 2023, pp. 2198–2210.
[71] H. Li, X. Zhou, L. A. Tuan, and C. Miao, “Rethinking negative pairs
in code search,”arXiv preprint arXiv:2310.08069, 2023.
[72] S. Chen, Y . Wang, and Z. Zheng, “Who needs the most research
effort? investigating the importance of smart contract weaknesses,”
inInternational Conference on Blockchain and Trustworthy Systems.
Springer, 2023, pp. 197–210.
[73] Y . Wang, L. Guo, E. Shi, W. Chen, J. Chen, W. Zhong, M. Wang,
H. Li, H. Zhang, Z. Lyuet al., “You augment me: Exploring chatgpt-
based data augmentation for semantic code search,” in2023 IEEE
International Conference on Software Maintenance and Evolution
(ICSME). IEEE, 2023, pp. 14–25.
[74] F. Hu, Y . Wang, L. Du, H. Zhang, D. Zhang, and X. Li, “Tackling long
code search with splitting, encoding, and aggregating,” inProceedings
of the 2024 Joint International Conference on Computational Lin-
guistics, Language Resources and Evaluation (LREC-COLING 2024),
2024, pp. 15 500–15 510.
[75] H. Dong, J. Lin, Y . Wang, Y . Leng, J. Chen, and Y . Xie, “Improving
code search with hard negative sampling based on fine-tuning,” in2024
31st Asia-Pacific Software Engineering Conference (APSEC). IEEE,
2024, pp. 221–230.
[76] D. Zheng, Y . Wang, W. Chen, J. Chen, and Z. Zheng, “Costv:
Accelerating code search with two-stage paradigm and vector retrieval,”
in2024 31st Asia-Pacific Software Engineering Conference (APSEC).
IEEE, 2024, pp. 383–392.
[77] X. Li, G. Dong, J. Jin, Y . Zhang, Y . Zhou, Y . Zhu, P. Zhang, and Z. Dou,
“Search-o1: Agentic search-enhanced large reasoning models,”arXiv
preprint arXiv:2501.05366, 2025.
[78] K. Chi, C. Li, J. Ge, and B. Luo, “An empirical study on code search
pre-trained models: Academic progresses vs. industry requirements,”
inProceedings of the 15th Asia-Pacific Symposium on Internetware,
2024, pp. 41–50.
[79] C. Wang, Z. Nong, C. Gao, Z. Li, J. Zeng, Z. Xing, and Y . Liu, “En-
riching query semantics for code search with reinforcement learning,”
Neural Networks, vol. 145, pp. 22–32, 2022.
[80] Y . Chen, M. Liu, G. Ou, A. Li, D. Dai, Y . Wang, and Z. Zheng, “Are
decoder-only large language models the silver bullet for code search?”
arXiv preprint arXiv:2410.22240, 2024.
[81] S. Zhang, H. Li, Y . Wang, Z. Wei, Y . Xiu, J. Wang, and R. Ji,
“Code search debiasing: improve search results beyond overall ranking
performance,”arXiv preprint arXiv:2311.14901, 2023.
[82] L. Guo, W. Tao, R. Jiang, Y . Wang, J. Chen, X. Liu, Y . Ma, M. Mao,
H. Zhang, and Z. Zheng, “Omnigirl: A multilingual and multimodal
benchmark for github issue resolution,”Proceedings of the ACM on
Software Engineering, vol. 2, no. ISSTA, pp. 24–46, 2025.
[83] L. Guo, Y . Wang, C. Li, P. Yang, J. Chen, W. Tao, Y . Zou,
D. Tang, and Z. Zheng, “Swe-factory: Your automated factory for issue
resolution training data and evaluation benchmarks,”arXiv preprint
arXiv:2506.10954, 2025.
[84] W. Tao, Y . Zhou, Y . Wang, W. Zhang, H. Zhang, and Y . Cheng,
“Magis: Llm-based multi-agent framework for github issue resolution,”
Advances in Neural Information Processing Systems, vol. 37, pp.
51 963–51 993, 2024.
[85] S. Chen, S. Lin, X. Gu, Y . Shi, H. Lian, L. Yun, D. Chen, W. Sun,
L. Cao, and Q. Wang, “Swe-exp: Experience-driven software issue
resolution,”arXiv preprint arXiv:2507.23361, 2025.
[86] Y . Ma, Q. Yang, R. Cao, B. Li, F. Huang, and Y . Li, “Alibaba
lingmaagent: Improving automated issue resolution via comprehensive
repository exploration,” inProceedings of the 33rd ACM International
Conference on the Foundations of Software Engineering, 2025, pp.
238–249.
[87] H. Li, Y . Shi, S. Lin, X. Gu, H. Lian, X. Wang, Y . Jia, T. Huang, and
Q. Wang, “Swe-debate: Competitive multi-agent debate for software
issue resolution,”arXiv preprint arXiv:2507.23348, 2025.[88] C. Xie, B. Li, C. Gao, H. Du, W. Lam, D. Zou, and K. Chen, “Swe-
fixer: Training open-source llms for effective and efficient github issue
resolution,”arXiv preprint arXiv:2501.05040, 2025.
[89] Z. Chen, Y . Pan, S. Lu, J. Xu, C. L. Goues, M. Monperrus, and
H. Ye, “Prometheus: Unified knowledge graphs for issue resolution
in multilingual codebases,”arXiv preprint arXiv:2507.19942, 2025.
[90] E. Shi, Y . Wang, L. Du, J. Chen, S. Han, H. Zhang, D. Zhang, and
H. Sun, “On the evaluation of neural code summarization,” inPro-
ceedings of the 44th international conference on software engineering,
2022, pp. 1597–1608.
[91] E. Shi, Y . Wang, L. Du, H. Zhang, S. Han, D. Zhang, and H. Sun, “Cast:
Enhancing code summarization with hierarchical splitting and recon-
struction of abstract syntax trees,”arXiv preprint arXiv:2108.12987,
2021.
[92] Y . Wang, Y . Wang, S. Wang, D. Guo, J. Chen, J. Grundy, X. Liu, Y . Ma,
M. Mao, H. Zhanget al., “Repotransbench: A real-world benchmark
for repository-level code translation,”arXiv preprint arXiv:2412.17744,
2024.
[93] G. Ou, M. Liu, Y . Chen, X. Peng, and Z. Zheng, “Repository-
level code translation benchmark targeting rust,”arXiv preprint
arXiv:2411.13990, 2024.
[94] R. Pan, A. R. Ibrahimzada, R. Krishna, D. Sankar, L. P. Wassi,
M. Merler, B. Sobolev, R. Pavuluri, S. Sinha, and R. Jabbarvand,
“Lost in translation: A study of bugs introduced by large language
models while translating code,” inProceedings of the IEEE/ACM 46th
International Conference on Software Engineering, 2024, pp. 1–13.
[95] Q. Tao, T. Yu, X. Gu, and B. Shen, “Unraveling the potential of large
language models in code translation: How far are we?” in2024 31st
Asia-Pacific Software Engineering Conference (APSEC). IEEE, 2024,
pp. 353–362.
[96] W. Yan, Y . Tian, Y . Li, Q. Chen, and W. Wang, “Codetransocean:
A comprehensive multilingual benchmark for code translation,”arXiv
preprint arXiv:2310.04951, 2023.
[97] W. Tao, Y . Wang, E. Shi, L. Du, S. Han, H. Zhang, D. Zhang, and
W. Zhang, “A large-scale empirical study of commit message genera-
tion: models, datasets and evaluation,”Empirical Software Engineering,
vol. 27, no. 7, p. 198, 2022.
[98] W. Tao, Y . Zhou, Y . Wang, H. Zhang, H. Wang, and W. Zhang, “Kadel:
Knowledge-aware denoising learning for commit message generation,”
ACM Transactions on Software Engineering and Methodology, vol. 33,
no. 5, pp. 1–32, 2024.
[99] P. Xue, L. Wu, Z. Yu, Z. Jin, Z. Yang, X. Li, Z. Yang, and Y . Tan,
“Automated commit message generation with large language models:
An empirical study and beyond,”IEEE Transactions on Software
Engineering, 2024.
[100] L. Zhang, J. Zhao, C. Wang, and P. Liang, “Using large language
models for commit message generation: A preliminary study,” in2024
IEEE International Conference on Software Analysis, Evolution and
Reengineering (SANER). IEEE, 2024, pp. 126–130.
[101] Y . Zhang, Z. Qiu, K.-J. Stol, W. Zhu, J. Zhu, Y . Tian, and H. Liu, “Au-
tomatic commit message generation: A critical review and directions
for future work,”IEEE Transactions on Software Engineering, vol. 50,
no. 4, pp. 816–835, 2024.
[102] W. Tao, Y . Wang, E. Shi, L. Du, S. Han, H. Zhang, D. Zhang, and
W. Zhang, “On the evaluation of commit message generation models:
An experimental study,” in2021 IEEE International Conference on
Software Maintenance and Evolution (ICSME). IEEE, 2021, pp. 126–
136.
[103] E. Shi, Y . Wang, W. Tao, L. Du, H. Zhang, S. Han, D. Zhang,
and H. Sun, “Race: Retrieval-augmented commit message generation,”
arXiv preprint arXiv:2203.02700, 2022.
[104] H. Guo, X. Chen, Y . Huang, Y . Wang, X. Ding, Z. Zheng, X. Zhou, and
H.-N. Dai, “Snippet comment generation based on code context expan-
sion,”ACM Transactions on Software Engineering and Methodology,
vol. 33, no. 1, pp. 1–30, 2023.
[105] C. Zhang, Y . Wang, Z. Wei, Y . Xu, J. Wang, H. Li, and R. Ji, “Ealink:
An efficient and accurate pre-trained framework for issue-commit
link recovery,” in2023 38th IEEE/ACM International Conference on
Automated Software Engineering (ASE). IEEE, 2023, pp. 217–229.
[106] Y . Wang, Y . Huang, D. Guo, H. Zhang, and Z. Zheng, “Sparsecoder:
Identifier-aware sparse transformer for file-level code summarization,”
in2024 IEEE International Conference on Software Analysis, Evolution
and Reengineering (SANER). IEEE, 2024, pp. 614–625.

[107] L. Guo, Y . Wang, E. Shi, W. Zhong, H. Zhang, J. Chen, R. Zhang,
Y . Ma, and Z. Zheng, “When to stop? towards efficient code generation
in llms with excess token prevention,” inProceedings of the 33rd ACM
SIGSOFT International Symposium on Software Testing and Analysis,
2024, pp. 1073–1085.
[108] I. Gim, G. Chen, S.-s. Lee, N. Sarda, A. Khandelwal, and L. Zhong,
“Prompt cache: Modular attention reuse for low-latency inference,”
Proceedings of Machine Learning and Systems, vol. 6, pp. 325–338,
2024.
[109] Z. Cai, Y . Zhang, B. Gao, Y . Liu, Y . Li, T. Liu, K. Lu, W. Xiong,
Y . Dong, J. Huet al., “Pyramidkv: Dynamic kv cache compres-
sion based on pyramidal information funneling,”arXiv preprint
arXiv:2406.02069, 2024.
[110] Y . Yue, Z. Yuan, H. Duanmu, S. Zhou, J. Wu, and L. Nie, “Wkvquant:
Quantizing weight and key/value cache for large language models gains
more,”arXiv preprint arXiv:2402.12065, 2024.
[111] Y . Feng, J. Lv, Y . Cao, X. Xie, and S. K. Zhou, “Ada-kv: Optimizing kv
cache eviction by adaptive budget allocation for efficient llm inference,”
arXiv preprint arXiv:2407.11550, 2024.
[112] Y . Bai, X. Lv, J. Zhang, H. Lyu, J. Tang, Z. Huang, Z. Du, X. Liu,
A. Zeng, L. Houet al., “Longbench: A bilingual, multitask benchmark
for long context understanding,” inProceedings of the 62nd annual
meeting of the association for computational linguistics (volume 1:
Long papers), 2024, pp. 3119–3137.
[113] Y . Wang, E. Shi, L. Du, X. Yang, Y . Hu, S. Han, H. Zhang, and
D. Zhang, “Cocosum: Contextual code summarization with multi-
relational graph neural network,”arXiv preprint arXiv:2107.01933,
2021.
[114] Z. Liao, J. Wang, H. Yu, L. Wei, J. Li, and W. Zhang, “E2llm: Encoder
elongated large language models for long-context understanding and
reasoning,” inProceedings of the 2025 Conference on Empirical
Methods in Natural Language Processing, 2025, pp. 19 212–19 241.
[115] Z. Li, C. Xu, Z. Shi, Z. Peng, Y . Liu, Y . Zhou, L. Zhou, C. Ma,
J. Zhong, X. Wanget al., “Deepcircuitx: A comprehensive repository-
level dataset for rtl code understanding, generation, and ppa analysis,”
arXiv preprint arXiv:2502.18297, 2025.
[116] C. Wang, J. Hu, C. Gao, Y . Jin, T. Xie, H. Huang, Z. Lei, and Y . Deng,
“How practitioners expect code completion?” inProceedings of the 31st
ACM Joint European Software Engineering Conference and Symposium
on the Foundations of Software Engineering, 2023, pp. 1294–1306.
[117] A. Hindle, E. T. Barr, M. Gabel, Z. Su, and P. Devanbu, “On the
naturalness of software,”Communications of the ACM, vol. 59, no. 5,
pp. 122–131, 2016.
[118] V . Raychev, M. Vechev, and E. Yahav, “Code completion with statistical
language models,” inProceedings of the 35th ACM SIGPLAN confer-
ence on programming language design and implementation, 2014, pp.
419–428.
[119] M. Bruch, M. Monperrus, and M. Mezini, “Learning from examples
to improve code completion systems,” inProceedings of the 7th joint
meeting of the European software engineering conference and the ACM
SIGSOFT symposium on the foundations of software engineering, 2009,
pp. 213–222.
[120] R. Robbes and M. Lanza, “How program history can improve code
completion,” in2008 23rd IEEE/ACM International Conference on
Automated Software Engineering. IEEE, 2008, pp. 317–326.
[121] A. Bhoopchand, T. Rockt ¨aschel, E. Barr, and S. Riedel, “Learning
python code suggestion with a sparse pointer network,”arXiv preprint
arXiv:1611.08307, 2016.
[122] R.-M. Karampatsis, H. Babii, R. Robbes, C. Sutton, and A. Janes,
“Big code!= big vocabulary: Open-vocabulary models for source code,”
inProceedings of the ACM/IEEE 42nd International Conference on
Software Engineering, 2020, pp. 1073–1085.
[123] S. Proksch, J. Lerch, and M. Mezini, “Intelligent code completion with
bayesian networks,”ACM Transactions on Software Engineering and
Methodology (TOSEM), vol. 25, no. 1, pp. 1–31, 2015.
[124] Y . Wang, H. Le, A. D. Gotmare, N. D. Bui, J. Li, and S. C. Hoi,
“Codet5+: Open code large language models for code understanding
and generation,”arXiv preprint arXiv:2305.07922, 2023.
[125] M. R. Parvez, W. U. Ahmad, S. Chakraborty, B. Ray, and K.-W.
Chang, “Retrieval augmented code generation and summarization,”
arXiv preprint arXiv:2108.11601, 2021.
[126] S. Zhou, U. Alon, F. F. Xu, Z. Wang, Z. Jiang, and G. Neu-
big, “Docprompting: Generating code by retrieving the docs,”arXiv
preprint arXiv:2207.05987, 2022.[127] J. Li, Y . Zhao, Y . Li, G. Li, and Z. Jin, “Acecoder: Utilizing existing
code to enhance code generation,”arXiv preprint arXiv:2303.17780,
2023.
[128] R. Bairi, A. Sonwane, A. Kanade, A. Iyer, S. Parthasarathy, S. Ra-
jamani, B. Ashok, S. Shetet al., “Codeplan: Repository-level coding
using llms and planning,”arXiv preprint arXiv:2309.12499, 2023.
[129] D. Liao, S. Pan, Q. Huang, X. Ren, Z. Xing, H. Jin, and Q. Li,
“Context-aware code generation framework for code repositories:
Local, global, and third-party library awareness,”arXiv preprint
arXiv:2312.05772, 2023.
[130] A. Eghbali and M. Pradel, “De-hallucinator: Iterative grounding for
llm-based code completion,”arXiv preprint arXiv:2401.01701, 2024.
[131] K. Zhang, J. Li, G. Li, X. Shi, and Z. Jin, “Codeagent: Enhancing code
generation with tool-integrated agent systems for real-world repo-level
coding challenges,”arXiv preprint arXiv:2401.07339, 2024.
[132] C. Wang, J. Zhang, Y . Feng, T. Li, W. Sun, Y . Liu, and X. Peng,
“Teaching code llms to use autocompletion tools in repository-level
code generation,”arXiv preprint arXiv:2401.06391, 2024.