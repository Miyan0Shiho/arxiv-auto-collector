# CodeRAG: Supportive Code Retrieval on Bigraph for Real-World Code Generation

**Authors**: Jia Li, Xianjie Shi, Kechi Zhang, Lei Li, Ge Li, Zhengwei Tao, Jia Li, Fang Liu, Chongyang Tao, Zhi Jin

**Published**: 2025-04-14 09:51:23

**PDF URL**: [http://arxiv.org/pdf/2504.10046v1](http://arxiv.org/pdf/2504.10046v1)

## Abstract
Large language models (LLMs) have shown promising performance in automated
code generation, especially excelling in simple tasks such as generating
standalone codes. Different from simple tasks, real-world code generation
usually depends on specific programming environment (e.g., code repositories).
It contains complex dependencies and domain knowledge, which is needed for LLMs
when generating target code snippets. In this paper, we propose CodeRAG, a
retrieval-augmented code generation (RAG) framework to comprehensively retrieve
supportive codes for real-world code generation. Beginning with the
requirement, CodeRAG first constructs a requirement graph for the current
repository, and retrieves sub- and similar- requirement nodes of the target
requirement on the graph. Meanwhile, it models the repository into a DS-code
graph. CodeRAG then maps these relevant requirement nodes into their
corresponding code nodes, and treats these code nodes as archors for LLM
reasoning on DS-code graph. Finally, CodeRAG introduces a code-oriented agentic
reasoning process, seamlessly allowing LLMs to reason and comprehensively
retrieve for supportive codes which LLMs' need for generating correct programs.
Experiments show that CodeRAG achieves significant improvements (i.e.,
increasing 40.90 and 37.79 Pass@1 on GPT-4o and Gemini-Pro on DevEval) compared
to no RAG scenarios. Further tests on reasoning LLMs (i.e., QwQ-32B) confirm
CodeRAG's adaptability and efficacy across various types of LLMs. In addition,
CodeRAG outperforms commercial programming products such as Copilit and Cursor.
We further investigate the performance of our framework on different dependency
types, and observe that CodeRAG is superior in generating examples where target
codes invoke predefined cross-file code snippets. These results demonstrate
CodeRAG's potential in solving real-world repo-level coding challenges.

## Full Text


<!-- PDF content starts -->

CodeRAG: Supportive Code Retrieval on Bigraph for Real-World
Code Generation
Jia Li
Key Lab of High Confidence Software
Technology (Peking University), MoE,
School of Computer Science, Peking
University
China
lijiaa@pku.edu.cnXianjie Shi
Key Lab of High Confidence Software
Technology (Peking University), MoE,
School of Computer Science, Peking
University
China
2100013180@stu.pku.edu.cnKechi Zhang
Key Lab of High Confidence Software
Technology (Peking University), MoE,
School of Computer Science, Peking
University
China
zhangkechi@pku.edu.cn
Lei Li
The University of Hong Kong
China
nlp.lilei@gmail.comGe Li∗
Key Lab of High Confidence Software
Technology (Peking University), MoE,
School of Computer Science, Peking
University
China
lige@pku.edu.cnZhengwei Tao, Jia Li
Key Lab of High Confidence Software
Technology (Peking University), MoE,
School of Computer Science, Peking
University
China
tttzw@stu.pku.edu.cn,lijia@stu.pku.edu.cn
Fang Liu
School of Computer Science and
Engineering, State Key Laboratory of
Complex & Critical Software
Environment, Beihang University
China
fangliu@buaa.edu.cnChongyang Tao
Beihang University
China
chongyang@buaa.edu.cnZhi Jin∗
Key Lab of High Confidence Software
Technology (Peking University), MoE,
School of Computer Science, Peking
University
China
zhijin@pku.edu.cn
Abstract
Large language models (LLMs) have shown promising performance
in automated code generation, especially excelling in simple tasks
such as generating standalone codes. Different from simple tasks,
real-world code generation usually depends on specific program-
ming environment ( e.g., code repositories). It contains complex
dependencies and domain knowledge, which is needed for LLMs
when generating target code snippets. In this paper, we propose
CodeRAG , a retrieval-augmented code generation (RAG) frame-
work to comprehensively retrieve supportive codes for real-world
code generation. Beginning with the requirement, CodeRAG first
constructs a requirement graph for the current repository, and
retrieves sub- and similar- requirement nodes of the target require-
ment on the graph. Meanwhile, it models the repository into a DS-
code graph. CodeRAG then maps these relevant requirement nodes
into their corresponding code nodes, and treats these code nodes
as archors for LLM reasoning on DS-code graph. Finally, CodeRAG
∗Corresponding author.
Permission to make digital or hard copies of all or part of this work for personal or
classroom use is granted without fee provided that copies are not made or distributed
for profit or commercial advantage and that copies bear this notice and the full citation
on the first page. Copyrights for components of this work owned by others than the
author(s) must be honored. Abstracting with credit is permitted. To copy otherwise, or
republish, to post on servers or to redistribute to lists, requires prior specific permission
and/or a fee. Request permissions from permissions@acm.org.
Conference acronym ’XX, Woodstock, NY
©2018 Copyright held by the owner/author(s). Publication rights licensed to ACM.
ACM ISBN 978-1-4503-XXXX-X/2018/06
https://doi.org/XXXXXXX.XXXXXXXintroduces a code-oriented agentic reasoning process, seamlessly al-
lowing LLMs to reason and comprehensively retrieve for supportive
codes which LLMs’ need for generating correct programs. Experi-
ments show that CodeRAG achieves significant improvements ( i.e.,
increasing 40.90 and 37.79 Pass@1 on GPT-4o and Gemini-Pro on
DevEval) compared to no RAG scenarios. Further tests on reasoning
LLMs ( i.e.,QwQ-32B) confirm CodeRAG ’s adaptability and efficacy
across various types of LLMs. In addition, CodeRAG outperforms
commercial programming products such as Copilit and Cursor. We
further investigate the performance of our framework on different
dependency types, and observe that CodeRAG is superior in gen-
erating examples where target codes invoke predefined cross-file
code snippets. These results demonstrate CodeRAG ’s potential in
solving real-world repo-level coding challenges.
CCS Concepts
•Computing methodologies →Neural networks ;•Software
and its engineering →Automatic programming .
Keywords
Code generation, Code retrieval, Graph, Large language model
ACM Reference Format:
Jia Li, Xianjie Shi, Kechi Zhang, Lei Li, Ge Li, Zhengwei Tao, Jia Li, Fang
Liu, Chongyang Tao, and Zhi Jin. 2018. CodeRAG: Supportive Code Re-
trieval on Bigraph for Real-World Code Generation. In Proceedings of Make
sure to enter the correct conference title from your rights confirmation emailarXiv:2504.10046v1  [cs.SE]  14 Apr 2025

Conference acronym ’XX, June 03–05, 2018, Woodstock, NY Jia Li and Xianjie Shi et al.
(Conference acronym ’XX). ACM, New York, NY, USA, 14 pages. https:
//doi.org/XXXXXXX.XXXXXXX
1 Introduction
Code generation has emerged as a pivotal task in software engineer-
ing, enabling models to automate essential software development
tasks. Recent large language models (LLMs), such as GPT-4o [ 2],
Gemini-Pro [ 31], and DeepSeek-V3 [ 20], have demonstrated re-
markable capabilities in code generation [ 15,23,39,40], especially
in generating simple standalone codes. Different from standalone
code generation, real-world code generation generally depends on
specific programming environments, such as repo-level code gener-
ation, which is more practical, given the recent success of Copilot1
and Cursor2. Real-world repo-level code generation heavily relies
on the current repository, including intricate dependencies and do-
main knowledge, which is challenging for existing LLMs to handle
and generate. To mitigate the aforementioned knowledge gap of
LLMs, retrieval-augmented code generation (RAG) has become a
key strategy for real-world repo-level code generation. An intu-
itive approach to addressing this challenge is to input the current
repository into LLMs. However, this approach is constrained by
LLMs’ context window and their long code context understanding
abilities [19].
Currently, researchers mainly propose three categories of ap-
proaches for retrieval-augmented repo-level code generation. The
first category [ 24,37] treats the current repository as multiple stan-
dalone code snippets, and retrieves relevant codes according to
cosine similarity scores or BM25 scores between code snippets and
the target requirement, but it overlooks the intricate contextual
dependencies in the repository. The second type [ 21,22] models the
current repository into a code graph, where nodes represent code
snippets ( e.g.,functions or classes) and edges mean relationships
between nodes such as invoking and inheritance. Given a target re-
quirement, this method converts a target requirement into a formal
graph query and uses the query to retrieve codes on the code graph.
Despite considering code dependencies, the grammars of graph
query only have a few limited types, resulting in incomplete re-
trieval. The third category [ 11,33,39] is an agentic approach, which
introduces programming tools to retrieve external knowledge into
LLMs, such as web searching.
Although these approaches have achieved impressive improve-
ment in repo-level code generation, comprehensively retrieving
supportive codes is a complex process. In real-world programming
scenarios, given a new functional requirement, developers typi-
cally analyze the dependence relationships of code snippets and
invoke predefined functions or classes in the current repository.
To correctly invoke them, developers usually refer to other source
codes being related to invoked codes in the current repository, since
unlike third-party APIs, these defined codes are domain-specific
and strange to developers. In addition to learning from the cur-
rent repository, developers sometimes refer to external knowledge
such as using web search to understand domain-specific theorems
and codes. Inspired by the human programming process based on
the foundational code framework, we propose a RAG framework
1https://copilot.microsoft.com/
2https://www.cursor.com/CodeRAG , aiming to comprehensively retrieve supportive codes
for real-world code generation. Our approach introduces four key
innovations:
•Requirement Graph. Beginning with the requirement, CodeRAG
constructs a requirement graph for the current repository by
considering relationships between the target requirement and
other requirements. The nodes represent functional descriptions
(named requirements) of functions or classes defined in the repos-
itory. The edges stand for relationships between requirements
(e.g.,parent-child relation, semantically similar relation). Given
a target requirement, we retrieve its sub-requirements and se-
mantically similar requirements in the requirement graph. Dif-
ferent from most RAG approaches[ 37,39],CodeRAG models
requirement relationships and retrieves supportive codes from
the perspective of requirements.
•DS-Code Graph. CodeRAG meanwhile desgins a DS-code graph.
Different from existing code graphs, DS-code graph not only
models dependency relationships but also introduces semantic
relationships among nodes. In DS-code graph, nodes mean code
elements in a repository and have five types ( e.g.,functions and
modules). Edges contain the dependency and semantic relation-
ships among nodes. Through DS-code graph, CodeRAG can
effectively model intricate relationships among source codes.
•Bigraph Mapping. Based on the above, requirement graph
and DS-code graph have the mapping relationship. CodeRAG
can map sub-requirement and similar requirement nodes of the
target requirement into their corresponding code nodes and
successfully find supportive codes. Because the corresponding
codes of sub-requirement nodes are most likely to be called by
target codes, and codes corresponding to similar requirement
nodes typically provide helpful information to LLMs.
•Code-oriented Agentic Reasoning. We introduce an agentic
code generation process, seamlessly allowing LLMs to reason and
comprehensively retrieve for supportive codes which LLMs’ need
during generation. In the reasoning process, LLMs can reason
from one selected code node to another on DS-code graph, as well
as searching external knowledge through web search, aiming to
comprehensively retrieve supportive code during reasoning.
Through this way, CodeRAG can comprehensively retrieve sup-
portive codes to help LLMs generate the correct codes: ❶APIs
(predefined functions or classes in the current repository) in-
voked by the target code ,❷Code snippets being semantically
similar to the target code ,❸Source codes that are indirectly
related to the target code . Besides, CodeRAG support retriev-
ing❹External domain knowledge through the web search tool,
such as referring to domain-specific theorem described in the target
requirement. Figure 1 shows the supportive codes of an example
retrieved by CodeRAG . We can find that being available to these
supportive codes is aligned with real-world software development
scenarios as described in Section 2.2.
We employ a repo-level code generation dataset DevEval [ 18] to
evaluate CodeRAG on mainstream LLMs, including GPT-4o [ 2] and
Gemini-Pro [ 31]. Experimental results demonstrate that CodeRAG
achieves significant improvements compared to all RAG baselines.
Specifically, CodeRAG outperforms no RAG scenarios with 40.90
Pass@1 on GPT-4o and 37.79 Pass@1 on Gemini-Pro. Further tests

CodeRAG: Supportive Code Retrieval on Bigraph for Real-World Code Generation Conference acronym ’XX, June 03–05, 2018, Woodstock, NY
on reasoning LLMs ( i.e.,QwQ-32B [ 4]) confirm CodeRAG ’s adapt-
ability and efficacy across various types of LLMs. Notably, CodeRAG
also outperforms commercial products like GitHub Copilot3and
Anysphere Cursor4. In addition, we investigate the performance of
our framework on different dependency types. We observed that
CodeRAG is superior in generating examples where target codes
invoke predefined cross-file code snippets. These results highlight
our framework’s potential in real-world repo-level coding tasks.
We summarize the contributions of this paper as follows:
•We propose CodeRAG , a novel RAG framework, aiming to
comprehensively retrieve supportive codes for real-world
repo-level code generation. Beginning with the requirement,
it constructs a requirement graph for the current program-
ming repository and retrieves relevant requirement nodes
of the target requirement. Meanwhile, it models a DS-code
graph, and maps relevant requirement nodes into their cor-
responding code nodes, finding supportive codes.
•CodeRAG introduces an agentic code generation process,
allowing LLMs to reason and dynamically search for compre-
hensively supportive codes which LLM needs in repo-level
code generation.
•Experimental results on diverse LLMs show the effectiveness
and versatility of CodeRAG in repo-level code generation,
highlighting its potential for resolving real-world coding
challenges.
2 Background & Motivation Example
2.1 Real-world Repo-level Code Generation
Existing code generation tasks mainly focus on generating stan-
dalone code units, including statement-level and function-level
generation. The generated programs are usually short and are inde-
pendent of other codes. However, in real-world software develop-
ment, programmers typically work within a specific programming
environment such as code repositories, and extend their function-
alities based on the foundational code framework. Inspired by this,
some studies[ 17,39] introduce the repo-level code generation task.
Given a code repository, the repo-level code generation task aims
to generate code based on all the software artifacts included in the
repository, encompassing the requirements, code dependency, and
runtime environment[ 39]. Given a target requirement and its signa-
ture, CodeRAG iteratively retrieves supportive codes and reasons
with an agentic process for generating satisfying programs that not
only adhere to requirements but also seamlessly integrate with the
current repository.
2.2 Motivation Example
CodeRAG is inspired by the human programming process based on
the foundational code framework, where developers mainly extend
new complex functionalities within a code repository. ❶These com-
plex functionalities typically encompass multiple sub-requirements,
and the source codes of these sub-requirements usually have been
predefined in the current repository. In the programming process,
developers frequently invoke these predefined functions or classes
3https://copilot.microsoft.com/
4https://www.cursor.com/in the current programming repository. ❷Unlike APIs defined in
third-party libraries, a code repository is often domain-specific,
such as the financial domain and security domain, which is unfamil-
iar to developers. Developers usually refer to other source codes in
the current repository when understanding these predefined func-
tions or classes. ❸Additionally, the repository typically contains
several semantically similar codes to the new functionalities, which
are typically helpful for developers to write correct programs by
referring to them. ❹Furthermore, the new functionalities some-
times involve domain knowledge such as domain-specific theorem
in the new requirement. They often use web searches to acquire
relevant knowledge and incorporate it into the programming pro-
cess. We refer to the four types of knoweledges as supportive codes
for repo-level code generation.
Inspired by the aforementioned development process, a good
real-world code generation tool should be able to comprehensively
retrieve supportive code, promoting to the generated code seam-
lessly integrating with the current repository. In this paper, we
propose CodeRAG , a RAG framework, aiming to comprehensively
retrieve supportive codes for repo-level code generation. CodeRAG
constructs a requirement graph for the current repository and re-
trieves relevant requirement nodes of the target requirement node.
Meanwhile, it models a DS-code graph, and maps relevant require-
ment nodes into their corresponding code nodes, successfully find-
ing supportive codes. In addition, CodeRAG introduces an agentic
process, which allows LLMs to adaptively adjust reasoning strat-
egy and search for the supportive knowledge which LLMs need in
repo-level code generation.
Figure 1 illustrates the retrieved supportive codes and the gen-
erated programs of an example by CodeRAG .CodeRAG first suc-
cessfully finds the sub-requirements and semantically similar re-
quirements of the target requirement as shown in Step 2. Through
bigraph mapping, our framework retrieves the supportive codes
(“_statement_matches_action ” and “ _matches_after_expansion ”),i.e.,
the corresponding codes of the sub-requirements and semantically
similar requirements on DS-code graph as shown in Step 3. Our
framework then executes the agentic reasoning process. We can
observe that CodeRAG further retrieves “ _listify_string ” by rea-
soning on DS-code graph in Step 4, which is the one-hop node
of “_statement_matches_action ” and has call relationships between
them. Meanwhile, CodeRAG uses web search to retrieve relevant
knowledge and integrates it into the generation process as shown
in Step 5. Finally, based on these supportive codes, CodeRAG gen-
erates correct programs and can successfully integrate with the
current repository. As shown in Figure 1, CodeRAG aligns with
the human programming process based on the code repository and
can successfully generate desired programs.
2.3 Research Questions
We hypothesize that these supportive codes are related to the target
requirements, and endow LLMs with comprehensive useful knowl-
edge, thus can bring benefits for real-world code generation. This
leads to several RQs.

Conference acronym ’XX, June 03–05, 2018, Woodstock, NY Jia Li and Xianjie Shi et al.
# Semantically Similar Code
def _statement_matches_action(
    statement: dict, 
    action: str,
    is_resource_policy_check: bool = False
  ) -> bool:
    if 'Action' in statement:
      for i in _listify_string(statement['Action']):
          if not is_resource_policy_check:
            ...
       return False
    else 'NoAction' in statement:
       result = True
       for i in _listify_string(statement['NoAction']):
            ...
        return result
# Invoked Code
def _matches_after_expansion(
    str_to_check: str, 
    str_to_check_against: str,
    condition_keys: Optional[CaseInsensitiveDict]
 ) -> bool:
    copy_str = str_to_check_against
    if condition_keys is not None:
        for k, v in condition_keys.items():     
         matches_action = _statement_matches(  
           statement, action_to_check, condition_ke
           ...    
    return pattern.match(str_to_check) is not None3. Supportive Codes  (Bigraph Mapping)
def _statement_matches_resource(
    statement: dict, 
    resource: str,
    condition_keys: Optional[CaseInsensitiveDict] = None
 ) -> bool:
    """
    Helper function, returns True if the given resource is 
    in the given policy statement. (# Target Requirement)
    """               1. Code to Be Completed
               2. Relevant Requirements  (Requirement Graph)
# Subrequirement
Helper function that checks the string_to_check 
against string_to_check_against..
# Semantically Similar Requirement
Helper function, returns True if the given action 
is in the given policy statement.4. Supportive Codes  (DS-Code Graph Reasoning)
# One-hop Code
def _listify_string(
    target_object: Union[List[str], str]
  ) -> List[str]:
    """    
    Helper function that takes a dictionary
    and returns it wrapped in a list
    """
    if isinstance(target_object, list):
        return target_object
    return [target_object]
def _statement_matches_resource(
    statement: dict, 
    resource: str,
    condition_keys: Optional[CaseInsensitiveDict] = None
) -> bool:
    if 'Resource' in statement:
        for res in _listify_string(statement['Resource']):
            if _matches_after_expansion(resource, res, condition_keys):
                return True
        return False
    elif 'NotResource' in statement:
        for res in _listify_string(statement['NotResource']):
            if _matches_after_expansion(resource, res, condition_keys):
                return False
        return True
    else:
        return True5. Completed Code
This function is a helper function that checks if a given 
resource is present in a policy statement. It checks if the 
resource matches any of the resources listed in the 'Resource' 
field of the statement. If it does, it returns True. If 'Resource' 
not in statement and the 'NotResource' field is present, it 
checks if the resource matches any of the resources listed in 
that field. If it does, it returns False. If neither 'Resource' nor 
'NotResource' fields are present, it returns True.# Output of Web Search Tool
Figure 1: An illustration of retrieved supportive codes and generated programs by CodeRAG .
RQ1. How effective is CodeRAG for real-world repo-
level code generation?
CodeRAG is a RAG framework, aiming to comprehensively re-
trieve supportive codes for repo-level code generation. We investi-
gate the effectiveness of our framework on the DevEval benchmark
[18] and compare it to other RAG approaches.
RQ2. How do different components in CodeRAG con-
tribute to performance improvement?
CodeRAG is an agentic system, which allows LLMs to adap-
tively adjust reasoning strategy and search for the supportive codes
which LLMs need. We performed an ablation study to figure out
the contribution and effectiveness of each component in CodeRAG .
RQ3. How does CodeRAG perform on examples that
have different dependency types?
Considering that the target codes have different dependency
types in repo-level code generation, we investigate the performance
ofCodeRAG on different dependency types including standalone
and non-standalone dependencies.
RQ4. How does CodeRAG perform on reasoning
LLMs?Recently, reasoning LLMs such as QwQ-32B [ 4] have achieved
promising reasoning ability in various tasks. In this RQ, we evaluate
whether CodeRAG is effective on reasoning LLMs.
RQ5. How well does CodeRAG perform compared
to commercial programming products like GitHub
Copilot and Anysphere Cursor?
Nowadays, a lot of mature commercial products are available
to support complex code generation tasks. In this RQ, we compare
CodeRAG with these established products.
3CodeRAG
CodeRAG aims to comprehensively retrieve supportive codes for
real-world repo-level code generation. These supportive codes can
provide helpful information for LLMs to generate desired codes.
CodeRAG first constructs a requirement graph of the current pro-
gramming repository based on requirements’ relationships. Given a
target requirement, our framework finds its sub-requirements and
semantically similar requirements (Section 3.1). CodeRAG then de-
signs a DS-code graph to model the current repository. Unlike exist-
ing code graphs, our DS-code graph considers not only dependency
relationships but also semantic relationships of code nodes (Section
3.2).CodeRAG maps the requirement graph into DS-Code graph
and retrieves the corresponding codes of sub-requirements and
semantically similar requirements (Section 3.3). Next, CodeRAG
introduces a code-oriented agentic reasoning process. It allows
LLMs to adaptively adjust reasoning strategy and search for the
supportive knowledge which LLM needs, and finally generate de-
sired codes that not only meet the target requirement but also

CodeRAG: Supportive Code Retrieval on Bigraph for Real-World Code Generation Conference acronym ’XX, June 03–05, 2018, Woodstock, NY
DS-Code Graph
CONTAINS
CALLSINHERITSSIMILARITYCALLS
IMPORTS
CONTAINSCONTAINSCONTAINS
One-HopRequirement Graph
SIMILARITYPARENT-CHILDS
PARENT-CHILDS
Target Requirement
GraphReason( )
 WebSearch( )
CodeTest( )Programming Tools
LLMsSupportive Code Retrieval
BiGraph Mapping
  <Target Requirement> + <Supportive Codes>
Code-oriented Agentic Reasoning
     <Generated Code> 
Figure 2: Overview of of our CodeRAG .
seamlessly integrate into the current programming environment
(Section 3.4).
3.1 Requirement Graph
Retrieving supportive codes is challenging. First, the target require-
ment is natural language while the code in a repository is pro-
gramming language, making it difficult to comprehensively find
supportive codes through textual sequence similarity or through
graph queries retrieving on code graph. Second, considering that
codes in a repository typically have various dependency relation-
ships, modeling a repository is a complex process. In this paper,
CodeRAG begins with requirements and constructs a requirement
graph, supporting identifying sub-requirements and semantically
similar requirements of the target requirement.
Node Extraction. We use the static analysis tool tree-sitter5to
parse the current programming repository and extract all functions,
classes, and methods predefined in it. The nodes in the requirement
graph represent the requirements ( i.e.,functional descriptions) of
these predefined code snippets in the repository. Considering that
a repository typically contains hundreds of code snippets, labeling
their requirements is time-consuming and labor-intensive. When
the extracted code snippets contain requirements, we use built-in
functional descriptions as their requirements. For code snippets
without requirements, we apply advanced LLM ( i.e.,DeepSeek-V2.5
[1]) to generate their requirements as it has demonstrated strong
understanding and generation abilities. The requirement generation
instructions are shown in Appendix A. We verify the reliability of
the generated descriptions and find they can effectively describe
the functions of code snippets as demonstrated in Section 6.1. Each
node has attributes to represent its meta-information, including its
corresponding source codes, file path, code name, and signature.
Requirement Relation Extraction. In real-world software
development, programmers typically invoke predefined codes or
learn from semantically similar codes within the current program-
ming repository. Inspired by this, the requirement graph mainly
5https://tree-sitter.github.io/tree-sitter/contains two types of edges: parent-child relationships and similar-
ity relationships. The parent-child relationship indicates that one
requirement is a sub-requirement of another, where the code of the
parent requirement usually invokes the code of the child require-
ment. The similarity relationship signifies that two requirements
have similar functional descriptions, and their corresponding codes
typically exhibit similar functionalities. Considering the workload
of manual annotation, we also utilize DeepSeek-V2.5 [ 1] to annotate
the relationships between requirements. The instruction of labeling
requirement relationships is shown in Appendix A.
Our requirement graph considers the requirement relationships
and can effectively find supportive codes from the requirement per-
spective. In addition, when the repository’s functionality expands,
our requirement graph can be extended instead of remodeling.
3.2 DS-Code Graph
In the code repository, code snippets exist in complex relationships.
Code graph can effectively model these relationships, facilitating
the analysis and comprehension of the current programming repos-
itory for LLMs. In this section, we model a code repository into
a DS-code graph. Different from other code graph[ 22], DS-code
graph not only considers the dependency relationships but also
introduces semantic relationships between code snippets in the
repository.
Graph Schema. In DS-code graph, nodes represent predefined
code units in the repository, and edges represent relationships be-
tween these code units. Different programming languages typically
require different schemas based on their features. In this paper, we
focus on Python and empirically design a schema tailored to its
characteristics. We define DS-code graph as follows.
DS-code graph contains four node types:
•Module represents a code file.
•Class is a class defined in the repository.
•Method refers to a method defined in the class.
•Function is a function code defined in the repository.
DS-code graph contains five edge types:
•Import : It represents modules’ dependency relationships, i.e.,one
module imports another.
•Contain : It means the source code of a node contains the coun-
terpart of another node. For example, a module might contain
classes, methods, and functions, and a class covers methods.
•Inherit : It allows a class to inherit the attributes of another class.
Through inheritance, a subclass can reuse the code of the super-
class, and it can also add new attributes and methods or override
the methods of the superclass.
•Call: It refers to the invoking relationship among code snippets.
•Similarity : It means two nodes have similar semantics.
In DS-code graph, each node has corresponding attributes to
represent its meta-information. For instance, Method nodes have
the name, source code, signature, class it belongs to, and file path
attributes.
Node and Edge Indexing. The DS-code graph construction pro-
cess consists of two phases, including node indexing and edge
extraction. In the node indexing phase, we extract a hierarchical di-
rectory tree of a repository according to files’ path in the repository,
where leaf nodes of the tree represent files. Then, we use tree-sitter

Conference acronym ’XX, June 03–05, 2018, Woodstock, NY Jia Li and Xianjie Shi et al.
to parse each file and acquire its abstract syntax tree (AST). We
extend the directory tree with the AST, and finally obtain a com-
plete tree for the repository. In the edge extraction process, we
design a language server tool based on tree-sitter to complete edges.
This tool can perform static analysis of a file or module and pro-
vides symbol names defined in it, encompassing global variables,
function names, and class names. Then, given a class or function
name, the tool finds its definition from the code repository. Com-
bining the above process, this tool can traverse predefined source
codes within a repository and acquire their dependency relation-
ships, empowering LLMs to understand intricate dependencies and
reuse codes. Besides dependency relationships, we also construct
semantic relationships between nodes in the DS-code graph.
Considering that manual labeling of semantic relationships is
laborious and time-consuming, we use a reliable embedding model
to encode the source code of each node and complete semantic rela-
tionships according to their vectors’ cosine similarities. Combining
the two processes, we can effectively traverse predefined source
code within a repository and model their relationships, modeling
a DS-code graph to help LLMs understand intricate dependencies
and reuse codes.
Graph Database. For storage efficiency, the node with code at-
tributes does not directly store its source code in the graph database
but rather models an index pointing to its code snippet. Finally, we
store all nodes and edges of DS-code graph into Neo4j.
3.3 Bigraph Mapping
After acquiring the requirement graph and DS-code graph, we
map the selected sub-requirement nodes and semantically similar
requirement nodes of the target requirement into code nodes in DS-
code graph. Then, we retrieve these related code nodes. The code
nodes of sub-requirement are typically invoked by the target code.
The code nodes of semantically similar requirements usually have
similar functionalities to the target code. Meanwhile, CodeRAG
introduces code nodes of the file where the target code locals in
because the local file contents are usually related to the target code.
Through this way, CodeRAG can sucessfully retrieve some sup-
portive codes for real-world repo-level code generation, including
❶APIs (predefined functions or classes in the repository) invoked
by the target code, and ❷code snippets being semantically similar
to the target code.
3.4 Code-oriented Agentic Reasoning
In this section, CodeRAG introduces a code-oriented agentic rea-
soning process, which allows LLMs to adaptively and sequentially
retrieve other supportive codes according to LLMs’ needs. This pro-
cess first designs three programming tools to help LLMs retrieve
other supportive codes. Then, it applies a reasoning strategy to
guide LLMs in suitable using these tools in reasoning process.
Designed Programming Tools. Given a target requirement, de-
velopers usually first gather relevant knowledge according to their
needs, and write programs to meet the requirement. Then, develop-
ers verify programs with the assistance of tools and further search
for knowledge according to the verification results if generated pro-
grams are error. Inspired by this, we develop three programming
tools that are specifically designed for LLMs to retrieve supportivecodes, including the web search tool, graph reasoning tool, and
code testing tool.
1) Web Search Tool. Programmers often share solutions for var-
ious programming problems on websites where search engines
consider them as knowledge resources. When faced with similar
problems, developers only submit a query to a search engine, which
then offers useful programming suggestions. To simulate this way,
CodeRAG uses DuckDuckGo6, a popular search engine that is more
cost-effective and convenient compared to other search engines
such as Google7and Bing8. Then, we apply LLMs to summarize the
searched website content as the final tool output. In the process,
we block websites that may lead to data leakage. The usage pattern
of this tool is formatted as: WebSearch(input_query), which will
return the formatted content searched from websites.
2) Graph Reasoning Tool. This tool is responsible for reasoning
on the DS-code graph and collecting supportive codes according
to LLMs’ needs. Specifically, we treat code nodes which have been
retrieved as the code anchor set. Given a retrieved code node and
its one-hop code nodes and edges, this tool traverses its one-hop
code nodes and determines which nodes are needed to retrieve
according to LLMs’ needs. If a code node is treated as supportive
code, we add it to the code anchor set. The usage pattern of this
tool is formatted as: GraphReason(code_anchor, its one-hop nodes
&edges), which will return new supportive codes and update the
code anchor set.
3) Code Testing Tool. After acquiring generated codes, we de-
sign a code testing tool to format and test them, enhancing their
correctness and readability. Specifically, we develop Black9as the
code test tool. It can check for format errors such as indentation
misalignment and missing keywords. Subsequently, it sends the
error information to LLMs and The usage pattern of this tool is:
CodeTest(generated_code), which will automatically format the
most recently generated code and return the formatted version.
ReAct Reasoning Strategy. To guide LLMs to leverage these
programming tools properly, we introduce a popular reasoning
strategy ReAct [ 34] into CodeRAG . This strategy prompts LLMs to
generate reasoning traces and task-related actions in an interlaced
pattern. Based on actions, ReAct selects the proper programming
tools and invokes them by providing input. The strategy then treats
the output of tools as additional knowledge and decides whether to
produce a final code or invoke other tools for further processing.
4 Experimental Setup
4.1 Baselines
We compare CodeRAG to the advanced RAG baselines.
•ScratchCG generates source codes only based on the target
requirement and signature, without retrieved codes.
•BM25-based RAG calculates BM25 scores between the target
requirement and code snippets in the repository and then
selects the top-k code snippets with higher BM25 scores. In
this paper, code snippets are predefined functions, methods,
and classes in the current programming repository.
6https://duckduckgo.com/
7https://www.google.com/
8https://www.bing.com/
9https://github.com/psf/black

CodeRAG: Supportive Code Retrieval on Bigraph for Real-World Code Generation Conference acronym ’XX, June 03–05, 2018, Woodstock, NY
Method RAG Type Pass@1
GPT-4o Gemini-Pro
ScratchCG w.o. RAG 17.24 14.95
BM25-based RAG Text-Based RAG 27.07 (↑9.83) 36.60 ( ↑21.65)
Embedding-based RAG Text-Based RAG 40.43 (↑23.19) 39.34 ( ↑24.39)
RepoCoder Text-Based RAG 30.95 (↑13.71) 30.36 ( ↑15.41)
CodeAgent Agentic RAG 28.66 (↑11.42) 33.09 ( ↑18.14)
CodeRAG (Ours) 58.14 (↑40.90) 54.74 (↑39.79)
Table 1: The performance of CodeRAG and advanced RAG-based approaches on the DevEval dataset. The red color represents
the absolute improvements of CodeRAG compared to ScratchCG.
•Embedding-based RAG uses an embedding model to encode
the target requirement and code snippets in the repository,
respectively. Then, it calculates their cosine similarities and
retrieves the top-k codes with higher similarities scores.
•RepoCoder is designed for repo-level code completion. It
incorporates a similarity-based retriever and a program gen-
erator in an iterative pipeline. It first completes codes based
on the above content, and utilizes the completed codes to
retrieve codes. In this paper, we adapt this approach to repo-
level code generation, where we use generated programs to
retrieve codes and iterate this retrieval generation process.
•CodeAgent is a pioneer LLM-based agent framework for
repo-level code generation. It integrates five programming
tools and implements four agent strategies to optimize these
tools’ usage, where these tools can support LLMs to retrieve
knowledge from the website and the current repository.
We do not compare CodeRAG to graph-based RAG methods such
as CodeXGraph [ 22] because their original papers do not provide
source codes for constructing graph code. In this paper, to keep
a fair comparison, we ensure that the numbers of retrieved code
elements of all approaches are equal.
4.2 Dataset
DevEval [ 18] is a representative repo-level code generation dataset,
which aligns with real-world repositories in multiple dimensions,
including code distributions and dependency distributions. It is
annotated by 13 developers and contains comprehensive annota-
tions ( e.g.,requirements, original repositories, reference code, and
reference dependencies). Finally, DevEval comprises 1,825 testing
samples from 117 repositories, covering 10 popular domains such as
Internet and Database. Given a requirement, DevEval asks models
to generate the code based on requirements and a repository.
4.3 Evaluation Metrics
Following previous works [ 16,30,38,42], we use the pass rate
as the metric, where we treat the generated program correctly
only if its output is consistent with all ground truths of the test
suite. Specifically, we are mainly concerned with Pass@1, which
is a representative of the Pass@k family, because in real-world
scenarios, we usually only consider the single generated code.5 Experimental Results
5.1 RQ1. How effective is CodeRAG for
real-world repo-level code generation?
Motivation. We investigate whether our CodeRAG obtains improve-
ments in real-world repo-level code generation when compared to
existing advanced RAG baselines.
Experimental Setup. We use GPT-4o ( i.e.,gpt-4o-2024-08-06)[ 2]
and Gemini-Pro ( i.e.,gemini-1.5-pro-latest) [ 31] as the base LLMs.
We use the greedy search strategy to generate programs, where
the maximum length of generated programs is set to 500. The base-
lines ( i.e.,Embedding-based RAG and RepoCoder) use the same
embedding model ( i.e.,stella_en_400M_v5) as CodeRAG . For all ap-
proaches, we parse a repository to acquire all predefined functions,
methods, and classes, which are treated as the minimum retrieval
units. To keep a fair comparison, we ensure that the numbers of
retrieved code units are equal among all baselines and CodeRAG .
For RepoCoder, we iterate the retrieval generation process two
times since iterating two times achieves the best performance as
demonstrated in its original paper[37].
Results. We compare CodeRAG to the competitive baselines and
show the results in Table 1. On all base LLMs, CodeRAG consis-
tently delivers significant performance improvements on repo-level
code generation. Specifically, for GPT-4o model, CodeRAG achieves
an increase of 40.90 compared to ScratchCG, and outperforms the
state-of-the-art baseline by 17.71 points on Pass@1. For Gemini-Pro,
the improvements of CodeRAG achieve 15.40 and 21.65 compared
to the strongest text-based RAG and Agentic RAG, meanwhile
achieve 39.79 points compared to ScratchCG. This demonstrates
thatCodeRAG can comprehensively retrieve supportive codes for
repo-level code generation, aiding LLMs in producing accurate code
solutions and effectively tackling complex coding challenges.
When turns to text-based RAG, approaches like BM25-based
RAG, Embedding-based RAG, and RepoCoder achieve 9.83, 23.19,
and 13.71 over ScratchCG ( i.e.,No RAG setting) on GPT-4o. Despite
bringing moderate improvements, their performances are lower
than CodeRAG . The reason might be that text-based RAG only
relies on textual or semantical similarities between the target re-
quirement and codes in the repository, whereas CodeRAG retrieves
codes from the perspective of requirements and continually retrieve

Conference acronym ’XX, June 03–05, 2018, Woodstock, NY Jia Li and Xianjie Shi et al.
Method Standalone Non-standalone Local-file Local &Cross-file Cross-file
Number of Examples
502 1323 455 571 157
ScratchCG 29.28 9.74 12.08 7.88 18.47
BM25-based RAG 38.44 21.73 25.93 18.39 20.38
Embedding-based RAG 50.19 39.79 46.81 25.04 21.66
RepoCoder 43.82 24.07 31.42 18.21 22.29
CodeAgent 40.63 24.07 31.64 18.47 18.04
CodeRAG (Ours) 60.16 48.24 69.67 45.18 43.31
Table 2: The performance of CodeRAG on examples with different dependency types.
Method #Usage Pass@1
CodeRAG 58.14
w.o. WebSearch 0.4 57.85 (↓0.29)
w.o. CodeTest 0.8 57.09 (↓1.05)
w.o. GraphReason 1.7 51.83 (↓6.31)
Table 3: Ablation study of our CodeRAG . The red color rep-
resents the contribution of each component in CodeRAG .
codes in the reasoning process based on LLMs’ needs. For agentic
RAG, CodeRAG is higher than CodeAgent. The reason might be
that CodeAgent only uses the textual matching between the target
function name and code unit names to retrieve related code snippets
with BM25, resulting in incomplete retrieval.
Across different base LLMs, RAG-based baselines show differ-
ent improvements on LLMs. For example, among text-based RAG,
the performance of RepoCoder is better than the counterpart of
BM25-based RAG on GPT-4o, while the phenomenon is reversed
on Gemini-Pro. However, we can find a notable trend that our
CodeRAG obtains the best performances on both LLMs. These re-
sults underscore the effectiveness and generalization of CodeRAG
on solving real-world programming tasks.
5.2 RQ2. How do different components in
CodeRAG contribute to performance
improvement?
Motivation. In this research question, we study whether each com-
ponent in CodeRAG contributes to the performance improvements,
and how they impact the results.
Experimental Setup. CodeRAG first models the relationships of
requirements and finds subrequirements and semantically simi-
lar requirements of the target requirement, then maps these re-
quirement nodes into code nodes within DS-code graph, where
these retrieved code nodes are treated as the code anchors. Next,
CodeRAG integrates a code-oriented agentic reasoning process to
allow LLMs to adaptively retrieve other supportive codes according
to LLMs’ needs. In this RQ, we remain code anchors and exclude
other supportive codes from CodeRAG as an ablation study, be-
cause code anchors is the basis for continually retrieval. Concretely,
w.o. WebSearch represents ablating the websearch tool from thefull framework. w.o. CodeTest stands for deleting the code format
checker in CodeRAG . w.o. GraphReason removes the graph rea-
soning tool where LLMs can not reason from anchors on DS-code
graph even if LLMs need other knowledge. We also meticulously
track the usage frequency of each tool during code generation pro-
cesses in CodeRAG , with the statistics shown in Table 3 under the
column #Usage. We choose GPT-4o as the base LLM.
Results. The performances of these ablation studies are shown
in Table 3, categorized under the column Pass@1. The removal of
DS-code graph reasoning (w.o. GraphReason) lead to a significant
degradation in performance, with Pass@1 decreasing by 6.31 points
(from 58.14 to 51.83). This reveals that reasoning on DS-code graph
is crucial for retrieving supportive codes for repo-level code gen-
eration. We argue that LLMs reason on the graph based on their
own needs, thus the retrieved code nodes are helpful for gener-
ating correct programs. On average, CodeRAG utilizes this tool
approximately 1.7 times per code generation, a frequency higher
than the counterpart of other tools. We can also find that removing
the web searching tool (w.o. WebSearch) results in 0.29 Pass@1
decreasing (from 58.14 to 57.85), demonstrating that the websearch
tool can bring useful information to LLMs. In addition, the ablation
of the code format checker (w.o. CodeTest) brings 1.05 performance
dropping in Pass@1 (from 58.14 to 57.09). This confirms that each
tool in CodeRAG contributes positive effects to the overall perfor-
mance. This evidence validates that CodeRAG can comprehensively
retrieve supportive codes in addressing the real-world coding task.
5.3 RQ3. How does CodeRAG perform on
examples with different dependency types?
Motivation. Considering that the target code typically contains vari-
ous and complex dependency relationships with other code snippets
within the current repository, we further investigate the perfor-
mance of CodeRAG in different dependency scenarios.
Experimental Setup. We divide the target code into two categories
(i.e.,standalone and non-standalone) based on dependency rela-
tionships between it and other source codes. Standalone category
means that the target code does not invoke any codes, while non-
standalone category represents the target code calls other codes in

CodeRAG: Supportive Code Retrieval on Bigraph for Real-World Code Generation Conference acronym ’XX, June 03–05, 2018, Woodstock, NY
the current repository. We then classify the non-standalone pro-
gram into three types, including local-file, local &cross-file, and
cross-file. Local-file stands for the target code only invokes the
source codes in the code file where the reference code locals in.
Local &cross-file type means the target code invokes not only codes
in the local file but also calls codes from cross files. Cross-file type
reveals that the target code only invokes codes from cross files.
Results. Figure 2 shows the performance ( i.e.,Pass@1) of GPT-4o
on different dependency types. CodeRAG achieves 30.88 Pass@1 im-
provement compared to ScratchCG on the standalone type. CodeRAG
outperforms ScratchCG with 57.59, 32.30, and 24.84 Pass@1. We
can observe that CodeRAG achieves higher improvements on the
non-standalone category over the standalone category. This demon-
strates that CodeRAG is more suitable for the non-standalone sce-
nario and can effectively retrieve dependency codes from the cur-
rent programming repository. Meanwhile, Pass@1 on standalone
functions also increases. We argue that even if the target code does
not invoke any codes, the retrieved code units can also provide
domain knowledge for LLMs to generate correct programs.
For three dependency types in the non-standalone category, the
results yield two insights. First, the difficulty of the three types
is that: cross-file >local&cross-file >local-file. Even for the most
difficult type ( i.e.,cross file), CodeRAG still achieves the impres-
sive 43.31 Pass@1 performance. Secondly, the harder the type is,
the higher improvements our framework achieves. For example,
CodeRAG obtains 24.85 Pass@1 improvements (from 18.47 to 43.31)
in the cross file type, while our framework outperforms ScratchCG
57.59 Pass@1 (from 12.08 to 69.67) in the local file type. This re-
veals the superiority of CodeRAG in solving real-world complex
programming tasks.
5.4 RQ4. How does CodeRAG perform on
reasoning LLMs?
Motivation. Recently, reasoning models have shown strong abilities
in various code tasks. In this RQ, we explore whether reasoning
LLMs can fully use the retrieved supportive codes and achieve sig-
nificantly improvements, since reasoning models have impressive
reasoning abilities in the generation process.
Experimental Setup. Although reasoning LLMs such as GPT-o3[ 3],
DeepSeek-R1[ 7], and QwQ-32B [ 4] claim to support function call
ability, they do not open this ability to the public through invoking
their API keys when we do experiments. Thus, we use Bigraph-
based RAG, a variant of CodeRAG , to evaluate reasoning LLMs.
Bigraph-based RAG excludes the agentic process from our full
framework since this process needs LLMs’ function call abilities.
We use QwQ-32B as the base reasoning LLM and only implement
partial baselines considering the expenditure.
Results. From Table 4, we can find that Bigraph-based RAG per-
forms impressive performance and obtains significant improve-
ments on repo-level code generation, with Pass@1 increasing by
35.57 points (from 18.57 to 54.41). In addition, reasoning LLMs
achieves comparable performance to non-reasoning models ( i.e.,
GPT-4o and Gemini-Pro), even if Bigraph-based RAG can not use
the three tools ( i.e.,web search, graph reasoning, and code test
tools) to acquire other supportive knowledge. There might be two
reasons for this phenomenon. First, the strong reasoning abilityMethod Pass@1
ScratchCG 18.57
BM25-based RAG 34.46
Embedding-based RAG 47.83
Bigraph-based RAG (Ours) 54.14
Table 4: The performance of CodeRAG on reasoning LLMs.
Only_localfile (Easy) Only_crossfile (Hard)
OurCodeRAG
GPT-4o 5 4
Gemini-Pro 5 4
IDE Product
GitHub Copilot 3 2
Anysphere Cursor 4 1
Table 5: Performance compared with commercial program-
ming products (the number of solved problems).
of reasoning LLMs can accelerate to generating correct programs
through thinking step by step. Second, in the reasoning process,
models can fully use retrieved supportive codes.
5.5 RQ5. How well does CodeRAG perform
compared to commercial products?
Motivation. Nowadays, a lot of commercial products are emerging
to support complex code generation tasks. In this RQ, we compare
CodeRAG with these established products.
Experimental Setup. We compare CodeRAG with two popular
IDE products, including GitHub Copilot10and Anysphere Cur-
sor11. They are AI-powered autocomplete-style suggestion tools
integrated within IDE software. Considering that IDE products are
primarily designed as completion systems, we limit human interac-
tions to less than three times per task to ensure a fair comparison.
We randomly select 5 easy examples (the local-file dependency type)
and 5 hard examples (the cross-file dependency type) from DevEval
to evaluate them, since manually executing on these production is
time-consuming and laborious.
Results. Table 5 demonstrates the performances of commercial
programming products and CodeRAG . We can find that CodeRAG
works better than existing products on both local-file and cross-file
coding types, especially in the hard cross-file scenario. Compared
to IDE products that can also analyze complex code dependencies
in the current repository, CodeRAG benefits from the numerous
retrieval optimizations tailored for real-world repo-level coding
tasks, thereby making it better than these products.
6 Discussion
6.1 Quality Analysis of Requirement Graph
As described in Section 3.1, we use DeepSeek-V2.5 to generate the
requirements of code units and annotate the relationships between
requirements in requirement graph. It is necessary to analyze the
10https://copilot.microsoft.com/
11https://www.cursor.com/

Conference acronym ’XX, June 03–05, 2018, Woodstock, NY Jia Li and Xianjie Shi et al.
# Target Requirement
This function takes a JWK (JSON Web Key) as input and returns the 
corresponding HMAC (Hash-based Message Authentication Code) key. 
It first checks if the input is a valid JSON string or dictionary. Then, it 
verifies if the key type is \"oct\" (indicating HMAC). Finally, it decodes 
and returns the HMAC key".
# Sub-Requirement
Define a custom exception for handling invalid keys in JWT operations. 
Input: None
Output: An exception instance of `InvalidKeyError` when raised.
# Similar Requirement
Determine if a given item is a HMAC key.
Input: A `key` representing the potential HMAC key.
Output: A boolean value indicating whether the input is a HMAC key.
Figure 3: Quality analysis of requirement graph.
quality of requirement graph. In this section, we randomly selected
a target requirement from DevEval and demonstrate its one-hot
requirements on requirement graph as shown in Figure 3. The one-
hot nodes of the target requirement contain its sub-requirement and
similar requirement, where the sub-requirement is about handling
invalid keys in JWT operations, and the similar requirement is
related to HMAC operations. We can find that requirement graph
can effectively model the relationships between requirements.
6.2 Threats to Validity
Threats to internal validity include the influence of the model
hyper-parameter settings for both our model and the reproduced
baselines. To ensure a fair comparison, the number of retrieved code
snippets of CodeRAG is the same as the counterpart of baselines.
For approaches which use embedding models to encode require-
ments or source codes ( i.e.,Embedding-based RAG, RepoCoder, and
CodeRAG ), we use the advanced stella_en_400M_v5 as the encode
model. Considering that RepoCoder is designed for retrieved-based
code completion, we keep the same setting with its paper and adapt
it to the repo-level code generation scenario, where we iterate
the retrieval generation process two times. For all baselines, we
parse a repository to acquire all predefined functions, methods, and
classes which are treated as the minimum retrieval units. In the
code generation process, we set the hyper-parameter ( i.e.,maxi-
mum generation length) to 500, which can support the length of
the reference codes. Meanwhile, we use the greedy search and gen-
erate a single program for each target requirement since developers
usually only consider the single generated code in real-world pro-
gramming environments. Therefore, there is a minor threat to the
hyper-parameter settings.
Threats to external validity include the quality of the datasets.
We use the DevEval dataset to evaluate the effectiveness of our
CodeRAG . DevEval is a representative repo-level code generation
dataset, which aligns with real-world repositories in multiple di-
mensions, such as code distributions and dependency distributions.
It contains 1,825 testing samples from 117 repositories and cov-
ers 10 popular domains, including Internet and Database, etc. Each
example is annotated by developers and contains comprehensive an-
notations ( e.g.,requirements, original repositories, reference code,
and reference dependencies). To verify the superiority of CodeRAG ,we consider six advanced approaches, including text-based RAG,
structure-based RAG, and agentic RAG. In addition, to effectively
evaluate our approach, we select mainstream LLMs ( i.e.,GPT-4o
and Gemini-Pr) as base models, and further analyze the perfor-
mance of CodeRAG on reasoning LLMs ( e.g.,QwQ-32B). We apply
our approach and baselines to these models and evaluate their per-
formance in repo-level code generation. For the metric, following
existing studies, we select a widely used Pass@k metric to evaluate
all approaches. It is an execution-based metric that utilizes test
cases to check the correctness of generated programs. To ensure
fairness, we execute each method two times and report the average
experimental results. CodeRAG uses LLMs to generate require-
ments of code snippets and model the relationships between re-
quirements since human labeling is time-consuming and laborious.
In this paper, we choose the advanced DeepSeek-V2.5 to construct
the requirement graph, which has impressive understanding and
generation abilities. To ensure the reliability of the requirement
graph, we elaboratively design instructions to LLMs. We manually
analyze the generated requirements and their relationships. We
observed that the generated requirements can correctly describe
the function of codes, meanwhile, DeepSeek-V2.5 can effectively
predict the relationships of requirements as shown in Appendix
A. In the future, we will explore more accurate ways to model the
requirement graph, even using human annotation methods.
7 Related Work
7.1 Retrieval-Augmented Code Generation
Retrieval-Augmented Code Generation (RAG) retrieves relevant
source codes or other knowledge relevant to the current program-
ming task and integrates retrieved information into the generation
process [ 9,12,13,36], such as being contextual input to models.
There are mainly three types of RAG studies.
Text-based RAG focuses on textual similarity to retrieval [ 6,8,14,
25], which encompasses sparse retrievers such as using BM25 [ 27]
to calculate the textual similarities between the requirement and
candidate codes, and embedding-based dense retrievers [ 10,26,41]
which invoke embedding models to encode them and select source
codes according to their cosine similarities. Through achieving
improvements, it overlooks the intricate contextual dependencies
in a code repository.
Structure-based RAG addresses this by modeling a repository
into a graph [ 5,22]. GraphCoder [ 21] constructs code context
graph (CCG) [ 32] that consists of control-flow, data- and control-
dependence between code statements used in retrieval-augmented
code completion task. CODEXGRAPH [ 22] utilizes static analysis
to model a code graph for the current repository. In the graph, code
nodes represent code snippets, including module, class, and func-
tion. Edges between code nodes represent the relationships among
these code snippets. However, structure-based RAG is limited by
graph query syntax, where graph queries only have a few types
of syntax. Meanwhile, these code graphs ignore to construct the
semantic relationships.
Agentic RAG integrates retrieval into the reasoning process, allow-
ing LLMs to interact with source codes [ 11,28,33?]. CodeAgent
[39] is a pioneer LLM-based agent framework for repo-level code
generation. It allows LLMs to invoke external programming tools

CodeRAG: Supportive Code Retrieval on Bigraph for Real-World Code Generation Conference acronym ’XX, June 03–05, 2018, Woodstock, NY
such as web searching and designs four agent strategies to opti-
mize these tools’ usage. Subsequently, researchers introduce several
agentic frameworks for other real-world coding tasks. For example,
Agentless [ 33] designed for issue solving preprocesses the code
repository’s structure and file skeleton, allowing LLMs to interact
with this knowledge.
Different from these advanced RAG methods, CodeRAG aims to
comprehensively retrieve supportive codes from the repository. It
constructs the bigraph ( i.e.,requirement graph and DS-code graph)
for a repository. Beginning with the requirement graph, CodeRAG
considers relationships between the target requirement and other
requirements, and retrieves relevant requirements of the target re-
quirement. It maps these nodes to their corresponding code nodes
on DS-code graph, treating them as code anchors. CodeRAG then
introduce an agentic process, allowing LLMs to retrieve other sup-
portive codes through reasoning on DS-code graph in the code
generation process.
7.2 Repo-level Code Generation
Recent years have witnessed growing interest in repo-level code
generation [ 29,35,39], as it better reflects real-world software
development scenarios. Unlike simple code generation tasks that
produce short, self-contained code snippets, repo-level code gener-
ation involves complex contextual dependencies that pose signif-
icant challenges for current LLMs. This emerging research direc-
tion has inspired multiple lines of investigation. On one hand, re-
searchers have developed specialized benchmarks like DevEval[ 18]
and EvoCodeBench[ 17] to evaluate repo-level code generation capa-
bilities. On the other hand, various approaches have been proposed
to enhance model performance. For instance, Zhang et al. [37]intro-
duced RepoCoder, which employs iterative retrieval and generation
to leverage repository context, while [ 39] developed an agentic
framework that equips LLMs with external programming tools to
boost their performance.
8 Conclusion
In this paper, we propose CodeRAG , a RAG framework for real-
world code generation. It designs the bigraph ( i.e.,requirement
graph and DS-code graph) for a code repository. Beginning with the
requirement graph, CodeRAG considers relationships between the
target requirement and other requirements, and retrieves relevant
requirements. It maps these requirement nodes into code nodes
on DS-code graph, where these code nodes are treated as anchors.
CodeRAG then introduce an agentic process, allowing LLMs to
comprehensively retrieve supportive codes through reasoning from
these anchors on DS-code graph. Experiments show that CodeRAG
achieves a significant improvement compared to advanced RAG
methods and commercial programming products, highlighting its
potential in real-world coding challenges.
References
[1] Deepseek-v2.5. https://huggingface.co/deepseek-ai/DeepSeek-V2.5 , 2024.
[2] Gpt-4o. https://openai.com/index/hello-gpt-4o/ , 2024.
[3] Gpt-o3. https://openai.com/index/openai-o3-mini/ , 2024.
[4] Qwq. https://huggingface.co/Qwen/QwQ-32B , 2024.
[5]Qiaolong Cai, Zhaowei Wang, Shizhe Diao, James Kwok, and Yangqiu Song.
Codegraph: Enhancing graph reasoning of llms with code. arXiv preprint
arXiv:2408.13863 , 2024.[6]Jiawei Chen, Hongyu Lin, Xianpei Han, and Le Sun. Benchmarking large language
models in retrieval-augmented generation. In Proceedings of the AAAI Conference
on Artificial Intelligence , volume 38, pages 17754–17762, 2024.
[7]DeepSeek-AI, Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu
Zhang, Runxin Xu, Qihao Zhu, Shirong Ma, Peiyi Wang, Xiao Bi, Xiaokang
Zhang, Xingkai Yu, Yu Wu, Z. F. Wu, Zhibin Gou, Zhihong Shao, Zhuoshu Li,
Ziyi Gao, Aixin Liu, Bing Xue, Bingxuan Wang, Bochao Wu, Bei Feng, Chengda Lu,
Chenggang Zhao, Chengqi Deng, Chenyu Zhang, Chong Ruan, Damai Dai, Deli
Chen, Dongjie Ji, Erhang Li, Fangyun Lin, Fucong Dai, Fuli Luo, Guangbo Hao,
Guanting Chen, Guowei Li, H. Zhang, Han Bao, Hanwei Xu, Haocheng Wang,
Honghui Ding, Huajian Xin, Huazuo Gao, Hui Qu, Hui Li, Jianzhong Guo, Jiashi Li,
Jiawei Wang, Jingchang Chen, Jingyang Yuan, Junjie Qiu, Junlong Li, J. L. Cai, Jiaqi
Ni, Jian Liang, Jin Chen, Kai Dong, Kai Hu, Kaige Gao, Kang Guan, Kexin Huang,
Kuai Yu, Lean Wang, Lecong Zhang, Liang Zhao, Litong Wang, Liyue Zhang,
Lei Xu, Leyi Xia, Mingchuan Zhang, Minghua Zhang, Minghui Tang, Meng Li,
Miaojun Wang, Mingming Li, Ning Tian, Panpan Huang, Peng Zhang, Qiancheng
Wang, Qinyu Chen, Qiushi Du, Ruiqi Ge, Ruisong Zhang, Ruizhe Pan, Runji
Wang, R. J. Chen, R. L. Jin, Ruyi Chen, Shanghao Lu, Shangyan Zhou, Shanhuang
Chen, Shengfeng Ye, Shiyu Wang, Shuiping Yu, Shunfeng Zhou, Shuting Pan, and
S. S. Li. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement
learning. CoRR , abs/2501.12948, 2025. doi: 10.48550/ARXIV.2501.12948. URL
https://doi.org/10.48550/arXiv.2501.12948.
[8]Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang Jia, Jinliu Pan, Yuxi Bi, Yi Dai,
Jiawei Sun, Haofen Wang, and Haofen Wang. Retrieval-augmented generation
for large language models: A survey. arXiv preprint arXiv:2312.10997 , 2, 2023.
[9]Leonidas Gee, Milan Gritta, Gerasimos Lampouras, and Ignacio Iacobacci. Code-
optimise: Self-generated preference data for correctness and efficiency. CoRR ,
abs/2406.12502, 2024. doi: 10.48550/ARXIV.2406.12502. URL https://doi.org/10.
48550/arXiv.2406.12502.
[10] Jiwoo Hong, Noah Lee, and James Thorne. Orpo: Monolithic preference opti-
mization without reference model. arXiv preprint arXiv:2403.07691 , 2(4):5, 2024.
[11] Dong Huang, Jie M Zhang, Michael Luck, Qingwen Bu, Yuhao Qing, and Heming
Cui. Agentcoder: Multi-agent-based code generation with iterative testing and
optimisation. arXiv preprint arXiv:2312.13010 , 2023.
[12] Dong Huang, Yuhao Qing, Weiyi Shang, Heming Cui, and Jie M Zhang. Effibench:
Benchmarking the efficiency of automatically generated code. arXiv preprint
arXiv:2402.02037 , 2024.
[13] Naman Jain, King Han, Alex Gu, Wen-Ding Li, Fanjia Yan, Tianjun Zhang, Sida
Wang, Armando Solar-Lezama, Koushik Sen, and Ion Stoica. Livecodebench:
Holistic and contamination free evaluation of large language models for code.
arXiv preprint arXiv:2403.07974 , 2024.
[14] Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin,
Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, et al.
Retrieval-augmented generation for knowledge-intensive nlp tasks. Advances in
neural information processing systems , 33:9459–9474, 2020.
[15] Jia Li, Ge Li, Chongyang Tao, Jia Li, Huangzhao Zhang, Fang Liu, and Zhi Jin.
Large language model-aware in-context learning for code generation. CoRR ,
abs/2310.09748, 2023. doi: 10.48550/ARXIV.2310.09748. URL https://doi.org/10.
48550/arXiv.2310.09748.
[16] Jia Li, Chongyang Tao, Jia Li, Ge Li, Zhi Jin, Huangzhao Zhang, Zheng Fang, and
Fang Liu. Large language model-aware in-context learning for code generation.
ACM Transactions on Software Engineering and Methodology , 2023.
[17] Jia Li, Ge Li, Xuanming Zhang, Yunfei Zhao, Yihong Dong, Zhi Jin, Binhua Li, Fei
Huang, and Yongbin Li. Evocodebench: An evolving code generation benchmark
with domain-specific evaluations. Advances in Neural Information Processing
Systems , 37:57619–57641, 2024.
[18] Jia Li, Ge Li, Yunfei Zhao, Yongmin Li, Zhi Jin, Hao Zhu, Huanyu Liu, Kaibo
Liu, Lecheng Wang, Zheng Fang, et al. Deveval: Evaluating code generation in
practical software projects. arXiv preprint arXiv:2401.06401 , 2024.
[19] Jia Li, Xuyuan Guo, Lei Li, Kechi Zhang, Ge Li, Zhengwei Tao, Fang Liu,
Chongyang Tao, Yuqi Zhu, and Zhi Jin. Longcodeu: Benchmarking long-context
language models on long code understanding. arXiv preprint arXiv:2503.04359 ,
2025.
[20] Aixin Liu, Bei Feng, Bing Xue, Bingxuan Wang, Bochao Wu, Chengda Lu, Cheng-
gang Zhao, Chengqi Deng, Chenyu Zhang, Chong Ruan, et al. Deepseek-v3
technical report. arXiv preprint arXiv:2412.19437 , 2024.
[21] Wei Liu, Ailun Yu, Daoguang Zan, Bo Shen, Wei Zhang, Haiyan Zhao, Zhi Jin,
and Qianxiang Wang. Graphcoder: Enhancing repository-level code completion
via code context graph-based retrieval and language model. arXiv preprint
arXiv:2406.07003 , 2024.
[22] Xiangyan Liu, Bo Lan, Zhiyuan Hu, Yang Liu, Zhicheng Zhang, Fei Wang, Michael
Shieh, and Wenmeng Zhou. Codexgraph: Bridging large language models and
code repositories via code graph databases. arXiv preprint arXiv:2408.03910 , 2024.
[23] Anton Lozhkov, Raymond Li, Loubna Ben Allal, Federico Cassano, Joel Lamy-
Poirier, Nouamane Tazi, Ao Tang, Dmytro Pykhtar, Jiawei Liu, Yuxiang Wei, et al.
Starcoder 2 and the stack v2: The next generation. arXiv preprint arXiv:2402.19173 ,
2024.

Conference acronym ’XX, June 03–05, 2018, Woodstock, NY Jia Li and Xianjie Shi et al.
[24] Shuai Lu, Nan Duan, Hojae Han, Daya Guo, Seung-won Hwang, and Alexey
Svyatkovskiy. Reacc: A retrieval-augmented code completion framework. In
Smaranda Muresan, Preslav Nakov, and Aline Villavicencio, editors, Proceedings of
the 60th Annual Meeting of the Association for Computational Linguistics (Volume
1: Long Papers), ACL 2022, Dublin, Ireland, May 22-27, 2022 , pages 6227–6240.
Association for Computational Linguistics, 2022. doi: 10.18653/V1/2022.ACL-
LONG.431. URL https://doi.org/10.18653/v1/2022.acl-long.431.
[25] Mirjam Minor and Eduard Kaucher. Retrieval augmented generation with llms
for explaining business process models. In International Conference on Case-Based
Reasoning , pages 175–190. Springer, 2024.
[26] Rafael Rafailov, Archit Sharma, Eric Mitchell, Christopher D Manning, Stefano
Ermon, and Chelsea Finn. Direct preference optimization: Your language model
is secretly a reward model. Advances in Neural Information Processing Systems ,
36, 2024.
[27] Stephen Robertson, Hugo Zaragoza, et al. The probabilistic relevance framework:
Bm25 and beyond. Foundations and Trends ®in Information Retrieval , 3(4):333–389,
2009.
[28] Samuel Schmidgall, Yusheng Su, Ze Wang, Ximeng Sun, Jialian Wu, Xiaodong Yu,
Jiang Liu, Zicheng Liu, and Emad Barsoum. Agent laboratory: Using llm agents
as research assistants. arXiv preprint arXiv:2501.04227 , 2025.
[29] Qiushi Sun, Zhirui Chen, Fangzhi Xu, Kanzhi Cheng, Chang Ma, Zhangyue
Yin, Jianing Wang, Chengcheng Han, Renyu Zhu, Shuai Yuan, et al. A survey
of neural code intelligence: Paradigms, advances and beyond. arXiv preprint
arXiv:2403.14734 , 2024.
[30] Zhihong Sun, Yao Wan, Jia Li, Hongyu Zhang, Zhi Jin, Ge Li, and Chen Lyu. Sifting
through the chaff: On utilizing execution feedback for ranking the generated
code candidates. In Vladimir Filkov, Baishakhi Ray, and Minghui Zhou, editors,
Proceedings of the 39th IEEE/ACM International Conference on Automated Software
Engineering, ASE 2024, Sacramento, CA, USA, October 27 - November 1, 2024 , pages
229–241. ACM, 2024. doi: 10.1145/3691620.3695000. URL https://doi.org/10.1145/
3691620.3695000.
[31] Gemini Team, Petko Georgiev, Ving Ian Lei, Ryan Burnell, Libin Bai, Anmol
Gulati, Garrett Tanzer, Damien Vincent, Zhufeng Pan, Shibo Wang, et al. Gemini
1.5: Unlocking multimodal understanding across millions of tokens of context.
arXiv preprint arXiv:2403.05530 , 2024.
[32] Yuxiang Wei, Federico Cassano, Jiawei Liu, Yifeng Ding, Naman Jain, Zachary
Mueller, Harm de Vries, Leandro Von Werra, Arjun Guha, and Lingming Zhang.
Selfcodealign: Self-alignment for code generation. arXiv preprint arXiv:2410.24198 ,
2024.
[33] Chunqiu Steven Xia, Yinlin Deng, Soren Dunn, and Lingming Zhang. Agent-
less: Demystifying llm-based software engineering agents. arXiv preprint
arXiv:2407.01489 , 2024.
[34] Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan,
and Yuan Cao. React: Synergizing reasoning and acting in language models. In
International Conference on Learning Representations (ICLR) , 2023.
[35] Daoguang Zan, Ailun Yu, Wei Liu, Dong Chen, Bo Shen, Wei Li, Yafen Yao,
Yongshun Gong, Xiaolin Chen, Bei Guan, et al. Codes: Natural language to code
repository via multi-layer sketch. arXiv preprint arXiv:2403.16443 , 2024.
[36] Dylan Zhang, Shizhe Diao, Xueyan Zou, and Hao Peng. PLUM: preference
learning plus test cases yields better code language models. CoRR , abs/2406.06887,
2024. doi: 10.48550/ARXIV.2406.06887. URL https://doi.org/10.48550/arXiv.2406.
06887.
[37] Fengji Zhang, Bei Chen, Yue Zhang, Jacky Keung, Jin Liu, Daoguang Zan, Yi Mao,
Jian-Guang Lou, and Weizhu Chen. Repocoder: Repository-level code completion
through iterative retrieval and generation. arXiv preprint arXiv:2303.12570 , 2023.
[38] Kechi Zhang, Ge Li, Yihong Dong, Jingjing Xu, Jun Zhang, Jing Su, Yongfei Liu,
and Zhi Jin. Codedpo: Aligning code models with self generated and verified
source code. CoRR , abs/2410.05605, 2024. doi: 10.48550/ARXIV.2410.05605. URL
https://doi.org/10.48550/arXiv.2410.05605.
[39] Kechi Zhang, Jia Li, Ge Li, Xianjie Shi, and Zhi Jin. Codeagent: Enhancing code
generation with tool-integrated agent systems for real-world repo-level coding
challenges. arXiv preprint arXiv:2401.07339 , 2024.
[40] Kechi Zhang, Ge Li, Jia Li, Yihong Dong, Jia Li, and Zhi Jin. Focused-dpo:
Enhancing code generation through focused preference optimization on error-
prone points. CoRR , abs/2502.11475, 2025. doi: 10.48550/ARXIV.2502.11475. URL
https://doi.org/10.48550/arXiv.2502.11475.
[41] Shengyu Zhang, Linfeng Dong, Xiaoya Li, Sen Zhang, Xiaofei Sun, Shuhe Wang,
Jiwei Li, Runyi Hu, Tianwei Zhang, Fei Wu, et al. Instruction tuning for large
language models: A survey. arXiv preprint arXiv:2308.10792 , 2023.
[42] Yuqi Zhu, Jia Li, Ge Li, Yunfei Zhao, Jia Li, Zhi Jin, and Hong Mei. Hot or cold?
adaptive temperature sampling for code generation with large language mod-
els. In Michael J. Wooldridge, Jennifer G. Dy, and Sriraam Natarajan, editors,
Thirty-Eighth AAAI Conference on Artificial Intelligence, AAAI 2024, Thirty-Sixth
Conference on Innovative Applications of Artificial Intelligence, IAAI 2024, Four-
teenth Symposium on Educational Advances in Artificial Intelligence, EAAI 2014,
February 20-27, 2024, Vancouver, Canada , pages 437–445. AAAI Press, 2024. doi:
10.1609/AAAI.V38I1.27798. URL https://doi.org/10.1609/aaai.v38i1.27798.

CodeRAG: Supportive Code Retrieval on Bigraph for Real-World Code Generation Conference acronym ’XX, June 03–05, 2018, Woodstock, NY
A Instructions for Requirement Graph
We present the instructions employed in requirement graph, includ-
ing requirement generation instructions and requirement relation
extraction instructions. Each instruction has undergone iterativerefinement to ensure that different models can not only follow it
but also produce outputs that appear satisfactory and are applicable
to real-world development scenarios when using these instructions.
Received 20 February 2007; revised 12 March 2009; accepted 5 June 2009

Conference acronym ’XX, June 03–05, 2018, Woodstock, NY Jia Li and Xianjie Shi et al.
You’re an expert Python programmer. Understand the given Python function Function_name. Generate a programming requirement that briefly describes
the purpose, input, and output of the given Python function Function_name.
Don’t generate any explanations.
Please follow the format to complete the skeleton below:
—
Purpose:···
Input:···
Output:···
—
Figure 4: Requirement Generation Instructions.
You’re an expert Python programmer. The final task is to generate the code snippet of the target requirement in the code repository. In this task, Python
programmer needs to focus not only on the target requirement, but also on the child requirements of the target requirement and the semantically similar
requirements of the target requirement.
Understand the target requirement and the candidate requirement. Determine and select the relation between the target requirement and the candidate
requirement from the following three options:
1. Parent-Child Relation: The candidate requirement is a child requirement of the target requirement. The corresponding code of the target requirement
invokes the corresponding code of the child requirement.
2. Semantic Similarity Relation: The candidate requirement and the target requirement are semantically similar. The code’s implementation of the target
requirement may learn from the code’s implementation of the candidate requirement.
3. Other Relations: The candidate requirement and the target requirement do not have the above relations.
Only return Parent-Child Relation or Semantic Similarity Relation or Other Relations.
The target requirement:
Target_requirement
Target_requirement_path
The candidate requirement:
Candidate_requirement
Candidate_requirement_path
The selected relation between the target requirement and the candidate requirement:
—
—
Do not generate any explanations and details.
Figure 5: Requirement Relation Extraction Instructions.