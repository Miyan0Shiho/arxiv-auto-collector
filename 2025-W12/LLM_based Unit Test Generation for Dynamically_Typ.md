# LLM-based Unit Test Generation for Dynamically-Typed Programs

**Authors**: Runlin Liu, Zhe Zhang, Yunge Hu, Yuhang Lin, Xiang Gao, Hailong Sun

**Published**: 2025-03-18 08:07:17

**PDF URL**: [http://arxiv.org/pdf/2503.14000v1](http://arxiv.org/pdf/2503.14000v1)

## Abstract
Automated unit test generation has been widely studied, but generating
effective tests for dynamically typed programs remains a significant challenge.
Existing approaches, including search-based software testing (SBST) and recent
LLM-based methods, often suffer from type errors, leading to invalid inputs and
assertion failures, ultimately reducing testing effectiveness. To address this,
we propose TypeTest, a novel framework that enhances type correctness in test
generation through a vector-based Retrieval-Augmented Generation (RAG) system.
TypeTest employs call instance retrieval and feature-based retrieval to infer
parameter types accurately and construct valid test inputs. Furthermore, it
utilizes the call graph to extract richer contextual information, enabling more
accurate assertion generation. In addition, TypeTest incorporates a repair
mechanism and iterative test generation, progressively refining test cases to
improve coverage. In an evaluation on 125 real-world Python modules, TypeTest
achieved an average statement coverage of 86.6% and branch coverage of 76.8%,
outperforming state-of-theart tools by 5.4% and 9.3%, respectively.

## Full Text


<!-- PDF content starts -->

LLM-based Unit Test Generation for Dynamically-Typed
Programs
Runlin Liu
Beihang University
China
runlin22@buaa.edu.cnZhe Zhang
Beihang University
China
zhangzhe2023@buaa.edu.cnYunge Hu
Beihang University
China
hygchn04@gmail.com
Yuhang Lin
Beihang University
China
yuhanglin35@gmail.comXiang Gao∗
Beihang University
China
xiang_gao@buaa.edu.cnHailong Sun
Beihang University
China
sunhl@buaa.edu.cn
Abstract
Automated unit test generation has been widely studied, but gen-
erating effective tests for dynamically typed programs remains a
significant challenge. Existing approaches, including search-based
software testing (SBST) and recent LLM-based methods, often suffer
from type errors, leading to invalid inputs and assertion failures,
ultimately reducing testing effectiveness. To address this, we pro-
pose TypeTest , a novel framework that enhances type correctness
in test generation through a vector-based Retrieval-Augmented
Generation (RAG) system. TypeTest employs call instance retrieval
andfeature-based retrieval to infer parameter types accurately and
construct valid test inputs. Furthermore, it utilizes the call graph
to extract richer contextual information, enabling more accurate
assertion generation. In addition, TypeTest incorporates a repair
mechanism and iterative test generation, progressively refining
test cases to improve coverage. In an evaluation on 125 real-world
Python modules, TypeTest achieved an average statement coverage
of 86.6% and branch coverage of 76.8%, outperforming state-of-the-
art tools by 5.4% and 9.3%, respectively.
1 Introduction
Unit testing is essential for ensuring software correctness and im-
proving code quality. However, manually writing unit tests is often
labor-intensive and time-consuming [ 57], making automated test
generation a valuable research direction. Traditional unit test gener-
ation approaches include random-based methods [ 9,48], constraint-
driven techniques [ 16,19,26], and search-based software testing
(SBST) techniques [ 3,21,23,62]. Among these, EvoSuite [ 3], an
influential SBST-based tool, has demonstrated significant success
in generating unit tests, but it is specifically designed for Java, a
statically typed language.
For dynamically-typed language, Pynguin is the first SBST-based
method designed for Python and has been widely adopted [ 23]. It
employs search-based strategies to maximize code coverage but
heavily relies on type annotations to infer expected data types. How-
ever, Python’s flexible typing allows variables to take on different
types in different contexts, making static analysis insufficient for
accurately determining expected values. As a result, Pynguin often
struggles with generating valid inputs, leading to type inconsisten-
cies and incomplete test cases. Notably, an analysis of developer
∗Corresponding Authorqueries on GitHub and Stack Overflow indicates that 30% of the
questions relate to type issues, highlighting the challenges posed by
implicit typing in real-world software development [ 32]. The origi-
nal study on Pynguin also highlights the necessity of integrating
type inference techniques to enhance test case generation.
More recently, LLM-based test generation methods have shown
promising potential. Microsoft’s CodaMosa [ 21], an LLM-augmented
extension of Pynguin, combines SBST with LLM-generated test
cases. Specifically, CodaMosa employs search-based testing to ex-
plore the code until coverage stagnates, at which point it queries
an LLM to generate additional test cases for under-tested functions.
Similarly, CoverUp [ 20] integrates detailed coverage data into LLM
prompts, leveraging static analysis to focus on low-coverage code
regions. While these approaches improve test coverage, they still
face fundamental challenges in dynamically typed programs. In
particular, correctly assigning values of the expected type within
specific call contexts remains a significant obstacle, leading to fre-
quent test failures due to type mismatches. Existing approaches
infer types based on static analysis [ 2,13] or deep learning [ 1,29].
Static-based approaches typically fail to handle function arguments,
as they struggle to accurately infer the types of dynamically as-
signed values. Deep learning (DL)-based approaches, on the other
hand, exhibit difficulty in predicting user-defined types due to their
reliance on learned patterns from training data. These limitations
make it challenging to leverage type-inference techniques to assist
in test case generation, leaving a critical gap in existing methods.
This raises a key research question: For dynamically typed pro-
grams, how can we improve the type accuracy to enhance the
accuracy of generated test cases?
Addressing this question requires overcoming several key chal-
lenges:
(1)Constructing correct test objects: Accurately instantiating
objects of the class containing the function under test.
(2)Providing correctly typed function arguments: Ensuring
that function calls receive arguments of appropriate types.
(3)Generating valid assertions: Inferring expected return types
and object properties to construct meaningful assertions for
behavior verification.
Our approach. To address these challenges, we propose a unified
argument construction approach for dynamically typed programs.
This approach focuses on accurate type inference, thereby increas-
ing the likelihood of generating correctly typed variables for objectarXiv:2503.14000v1  [cs.SE]  18 Mar 2025

Conference’17, July 2017, Washington, DC, USA Runlin Liu, Zhe Zhang, Yunge Hu, Yuhang Lin, Xiang Gao, and Hailong Sun
instantiation and function invocation. Based on this approach, we
develop TypeTest , a test generation framework that integrates two
main methods to improve the type correctness. Given a function
𝑓under test, TypeTest first employs call instance recovery , a di-
rect method that leverages existing usage instances of 𝑓within
the project to infer the expected types of the object and parame-
ters. This method allows TypeTest to extract concrete examples
of how𝑓is invoked in real-world code, improving the accuracy of
generated test inputs.
When no direct usage instances are available, TypeTest applies
a more general approach: Feature-based Retrieval . Inspired by the
“Duck Test”1, this method infers types by analyzing a parameter’s
behavior and matching it to known types with similar characteris-
tics. Specifically, it extracts the behavior features of the key variable,
such as its methods, attributes, and operations, and retrieves types
that exhibit similar usage patterns. This approach enables TypeTest
to approximate types based on functional behaviors rather than
relying solely on static type annotations.
Beyond input construction, TypeTest improves assertion gen-
eration through call graph-based analysis . Using topological and
reverse topological traversal of the call graph, this method extracts
function behavior and high-level semantics, enriching the context
provided to LLMs. This additional information enables LLMs to
generate more precise and semantically meaningful assertions. Fi-
nally, TypeTest employs an iterative refinement mechanism to
improve test quality. It prompts LLMs within a structured repair-
and-improvement loop, incrementally refining test cases based on
execution feedback. To achieve higher coverage, TypeTest marks
uncovered statements and strategically guides LLM-generated test
cases toward these regions in subsequent iterations.
We evaluate TypeTest on 125 real-world Python modules, where
it achieves an average statement coverage of 86.6% and branch
coverage of 76.8%, outperforming state-of-the-art tools by 5.4% and
9.3%. Furthermore, compared to a variant that replaces TypeTest ’s
type inference with a state-of-the-art type inference tool, TypeTest
improves statement coverage and branch coverage by 6.0% and
10.5%. Through further investigation, we find that the improvement
in coverage is most strongly correlated with the correctness of
inferred types.
In summary, our contributions are summarized as follows:
•We present the first approach that explicitly addresses type cor-
rectness in unit test generation for dynamically-typed languages.
Specifically, we design and implement a type inference approach
tailored for Python.
•We develop a vector-based RAG framework that enables precise
type inference and type-correct parameter construction. This
framework allows TypeTest to effectively integrate source code
to support accurate type inference, object and argument con-
struction, and assertion generation.
•We conduct a comprehensive evaluation against three state-of-
the-art unit test generation tools. Experimental results demon-
strate that TypeTest significantly improves both statement and
branch coverage, highlighting its effectiveness in real-world
Python projects.
1Duck Test: https://en.wikipedia.org/wiki/Duck_test2 Motivation
We use the get_custom_loader function from the PyCG2project
as a representative example to illustrate our motivation. PyCG is
an open-source project on GitHub with 332 stars. It generates call
graphs by analyzing Python code and supports advanced features
such as higher-order functions and complex class inheritance struc-
tures.
As shown in Listing 1a, get_custom_loader returns a custom
module loader, CustomLoader . During initialization, ig_obj is re-
sponsible for maintaining module import relationships: it first cre-
ates import edges, then checks whether the module already exists
in the graph. If the module is not present, a new node is created, and
its file path is recorded. This mechanism enables the simultaneous
construction and management of module dependencies during the
loading process, facilitating subsequent analysis.
def get_custom_loader(ig_obj):
class CustomLoader(importlib.abc.SourceLoader):
def __init__(self, fullname, path):
self.fullname = fullname
self.path = path
ig_obj.create_edge(self.fullname)
if not ig_obj.get_node(self.fullname):
ig_obj.create_node(self.fullname)
ig_obj.set_filepath(self.fullname,
self.path)
<...omitted code...>
return CustomLoader
a: Part of the get_custom_loader function
def test_case_4():
try:
bool_0 = True
var_0 = module_0.get_custom_loader(bool_0)
<...omitted code...>
except BaseException:
pass
b: Test case Generated by SBST
class ImportGraph:
<...omitted code...>
def test_custom_loader():
ig_obj = ImportGraph()
loader = get_custom_loader(ig_obj)
<...omitted code...>
c: Test case Generated by GPT-4o
Listing 1: An example of code and the corresponding auto-
generated unit tests.
We employ the unit test generation tool CodaMosa [ 21] to gen-
erate test cases for the get_custom_loader function. Initially, Co-
daMosa utilizes search-based software testing (SBST) to produce
simple test cases, as shown in Listing 1b. The test case sets bool_0
to True and passes it to get_custom_loader . However, since the
constructor of CustomLoader accesses members of the ig_obj pa-
rameter, any instantiation of CustomLoader results in a runtime
error, preventing further improvements in coverage. When SBST
fails to enhance coverage, CodaMosa activates the large language
model to generate test cases. One of the test cases generated by
GPT-4o is shown in Listing 1c. In this case, GPT-4o constructs
2PyCG: https://github.com/vitsalis/PyCG

LLM-based Unit Test Generation for Dynamically-Typed Programs Conference’17, July 2017, Washington, DC, USA
theImportGraph class and ensures that all members accessed by
the__init__ function of CustomLoader are present, thus prevent-
ing run-time errors. However, due to the absence of explicit type
information for ig_obj , GPT-4o is unable to infer the exact im-
plementations of methods such as create_edge . Instead, it relies
solely on method names to approximate their behavior, leading
to inconsistencies between the generated test case and the actual
implementation. CodaMosa then makes 16 additional attempts, all
of which fail. The primary cause of this failure is that CodaMosa
does not provide type information in its prompts, preventing GPT-
4o from generating semantically valid test cases. To infer the type
ofig_obj , we employ Hityper [ 35], which is the state-of-the-art
tool for type inference. However, Hityper incorrectly infers ig_obj
as astrtype. This incorrect inference arises because ig_obj is a
function parameter without any direct assignment statements to
aid type inference. Moreover, since ig_obj is a user-defined type,
the deep learning component of Hityper struggles to infer its type
accurately.
class ImportManager(object):
def __init__(self):
self.import_graph = dict()
self.current_module = ""
self.input_file = ""
<...omitted code...>
a: Constructor of DocstringDeprecated retrieved by RAG mod-
ule
class TestGetCustomLoader(unittest.TestCase):
def setUp(self):
self.ig_obj = ImportManager()
self.loader_class = get_custom_loader(self.
ig_obj)
def test_loader(self):
loader = self.loader_class("test.module",
"/path/to/module.py")
self.assertIn("test.module", self.ig_obj.
import_graph)
<...omitted code...>
b: Test case generate by TypeTest
Listing 2: The unit tests generated by TypeTest
In contrast, our tool, TypeTest , effectively addresses these chal-
lenges. It first utilizes LLMs to generate summaries for the source
code, which are then indexed to construct a comprehensive knowl-
edge base. When TypeTest detects insufficient contextual infor-
mation, it retrieves relevant information from the knowledge base,
specifically focusing on the type information for key variables
within the unit under test. For example, when generating test cases
for theget_custom_loader ,TypeTest retrieves details about the
ig_obj . In the__init__ function of CustomLoader ,ig_obj in-
vokes member methods such as create_edge andget_node , which
are responsible for maintaining module import relationships. To
resolve the type of ig_obj ,TypeTest identifies classes that define
these methods and are semantically related to import management.
It subsequently locates the ImportManager class and extracts infor-
mation about its constructor (as illustrated in Listing 2a and further
explained in Section 3.2.3). By incorporating this information into
the prompt, GPT-4o can instantiate an ImportManager object and
pass it as an argument to get_custom_loader , as demonstratedin Listing 2b. This approach enables TypeTest to successfully in-
stantiate loader_class and achieve full statement coverage for
get_custom_loader , thereby not only improving test coverage but
also generating assertions that verify the core functionality of the
function.
This example demonstrates that accurately inferring type in-
formation for key variables can significantly enhance both the
coverage of generated test cases and the quality of assertions. Ex-
isting approaches, when handling dynamically typed languages
like Python, frequently overlook critical contextual information,
resulting in poor-quality test cases generated by GPT-4o. Address-
ing this limitation requires a method capable of searching across
the entire project, filtering relevant information, and accurately
inferring variable types. TypeTest fulfills this requirement, making
it a promising solution for improving automated test generation in
dynamically typed languages.
3 Methodology
We introduce TypeTest to tackle the challenges in unit test genera-
tion for dynamically typed programs, with its workflow depicted
in Figure 3. TypeTest operates through a four-step process: (1)
Given the program 𝑝,TypeTest first generates summaries for each
functions and vectorizes them to build a knowledge base, which
will be utilized in the following RAG process. (2) For a given focal
function𝑓,TypeTest then identifies the class 𝑐that contains 𝑓and
resolve the types of 𝑓’s parameters. The resolved types are then
utilized to construct unit test with correct testing environment and
arguments. (3) TypeTest then generates assertions for each auto-
generated unit test by inferring the code semantics according to the
code summary. (4) At last, TypeTest fixes the potential compilation
and assertion errors in the auto-generated unit test and discards
the invalid ones. In the following, we detail our methodology.
3.1 Vector-based RAG framework
Given a focal function, generating its high-quality unit tests via
LLMs usually rely on the understanding its context. Previous ap-
proaches predominantly extract relevant context [ 51] by statically
analyzing source code. While static analysis proves effective for
strongly typed languages such as Java, it is less suitable for dynam-
ically typed languages, where variable types and function return
values are often implicit and may change at runtime. This dynamic
nature makes it challenging for static analysis to accurately infer
types for parameters and variables, potentially leading to inaccura-
cies in generated test cases. To address these challenges, TypeTest
utilizes Retrieval-Augmented Generation (RAG) technology to re-
trieve relevant contextual information, which serves as a foundation
for subsequent test generation methods. In this section, we present
the construction process of the knowledge base used in RAG and
outline the key steps involved in its application.
3.1.1 Knowledge base construction. The knowledge base comprises
three primary components: code summary, source code, and exist-
ing test cases, as shown in Listing 3.

Conference’17, July 2017, Washington, DC, USA Runlin Liu, Zhe Zhang, Yunge Hu, Yuhang Lin, Xiang Gao, and Hailong Sun
Figure 3: The overall workflow of TypeTest .
[
"summary": str,
"source_code": {
"module_path": str,
"name": str,
"source_code": str,
"docstring": str
},
"test_cases": {
"label": str,
"unit_path": str,
"unit_name": str,
"source_code": str
}
]
Listing 3: Structure of the
knowledge base.The code summary acts
as an index for the Retrieval-
Augmented Generation sys-
tem. Summary-based in-
dexing has been shown to
significantly enhance re-
trieval accuracy [11]. As a
result, TypeTest employs
Large Language Models
(LLMs) to generate seman-
tic summaries for func-
tions and classes, which
serve as reference points
for retrieving relevant con-
text.
The source code is a crit-
ical component, as it pro-
vides most of the contextual information used by RAG. For each
function in the program under test, we extract the module path,
unit name, source code, and docstring using abstract syntax trees
(AST). This structured information aids in the generation of test
cases. For example, when RAG retrieves a constructor, as shown
in Listing 2a, it also obtains the corresponding module path, en-
suring correct import“from pycg.machinery.imports import
ImportManager” is generated.
The existing test cases further support test generation by serving
as additional prompts for LLMs [ 61]. For instance, given a focal
function𝑓from class𝑐, it the unit test of 𝑐’s constructor exists, it
can be leveraged to instantiate the object when generating new test
cases for𝑓. To utilize these effectively, TypeTest integrates both
existing and newly generated test cases into the RAG knowledge
base, storing key metadata such as the path, name of the focal
function, and test case source code.
Figure 4: An instance of the RAG process.
3.1.2 RAG process. As illustrated in Figure 4, the RAG process
begins when a query is received from other components within
TypeTest . This query triggers a retrieval operation over the knowl-
edge base to identify the most relevant information documents.
These relevant documents are then processed by an LLM, which
filters, deduplicates, and integrates them into a cohesive response.
The refined information is subsequently returned to the requesting
component. Retrieved documents are ranked using Maximal Mar-
ginal Relevance (MMR), which balances relevance and diversity to
improve the quality of the results. This architecture enables any
process within TypeTest to request specific contextual information
by submitting a query, prompting the RAG process to retrieve and
consolidate relevant data from the knowledge base.
3.2 Type Inference and Parameter Construction
for Unit Test Generation
The core issue for generating unit tests for dynamically typed pro-
gram is “create objects and parameters with the correct type.” Al-
gorithm 1 is proposed to address type resolution issues, which
leverages two key strategies: referencing existing call instances and
using feature-based retrieval .
3.2.1 Overview. Generating unit tests for dynamically typed lan-
guages requires constructing correctly typed objects and invoking

LLM-based Unit Test Generation for Dynamically-Typed Programs Conference’17, July 2017, Washington, DC, USA
focal functions with appropriately typed parameters. This ensures
that both constructors and focal functions are called correctly. The
underlying problem can be formalized as follows: given a focal
function𝑓, which may be a constructor ( __init__ ) or an instance
method, it must be invoked correctly. This involves:
•Inferring the types of 𝑓’s parameters Para.
•Constructing instances of Para based on the inferred types.
As shown in Algorithm 1, TypeTest takes as input a call graph
𝑐𝑔of program under test and a function 𝑓with a set of parame-
ters𝑃𝑎𝑟𝑎 ={𝑝𝑎𝑟𝑎 1,𝑝𝑎𝑟𝑎 2,...,𝑝𝑎𝑟𝑎 𝑛}. The output is a set of con-
structed parameters 𝑀for𝑓, where each parameter is instantiated
according to its inferred type. The algorithm begins by extracting
pre-existing call instances of 𝑓from𝑐𝑔, denoted as 𝑃𝐼(line 2). For
each parameter 𝑝𝑎𝑟𝑎 , its type is inferred based on its usage patterns
and the call instances observed in 𝑃𝐼, using the InferType function.
The inferred type is stored in 𝑡𝑦𝑝𝑒 (line 6). If𝑡𝑦𝑝𝑒 is of primitive
type, the corresponding argument 𝑚is constructed directly (line 9).
If𝑡𝑦𝑝𝑒 is identified as a user-defined type, the argument is con-
structed using the Retrieval-Augmented Generation (RAG). In this
process, TypeTest first checks for the presence of a type annotation.
If a type annotation is available, TypeTest constructs directly an
object based on the annotation (line 14). Otherwise, it retrieves
relevant contextual information from the knowledge base based on
the characteristics of 𝑝𝑎𝑟𝑎 to construct a suitable object (line 17).
This process iterates until the final custom object is successfully
instantiated and added 𝑀, ensuring that all necessary dependencies
are correctly resolved.
Algorithm 1: Type Resolution for Parameter Construction
Input: Call graph of the program under test: 𝑐𝑔; Function𝑓
with parameters: 𝑃𝑎𝑟𝑎 ={𝑝𝑎𝑟𝑎 1,𝑝𝑎𝑟𝑎 2,...,𝑝𝑎𝑟𝑎 𝑛};
Output: Constructed 𝑃𝑎𝑟𝑎 :𝑀
1𝑀←∅;
2/* Get Pre-existing Call Instances by 𝑐𝑔*/
3PI←GetInstance( cg,𝑓);
4/* Construct Parameters */
5foreach𝑝𝑎𝑟𝑎∈𝑃𝑎𝑟𝑎 do
6 /* Infer Parameter Type */
7𝑡𝑦𝑝𝑒←InferType(𝑓,𝑝𝑎𝑟𝑎 ,PI);
8 /* Construct Argument Based on Inferred Type */
9 if𝑡𝑦𝑝𝑒 is a primitive type then
10𝑚←constructPrimitive( 𝑡𝑦𝑝𝑒 );
11 else
12 /* Construct Argument by RAG process */
13𝑡𝑦𝑝𝑒 _𝑎𝑛𝑛𝑜𝑡𝑎𝑡𝑖𝑜𝑛←getAnnotation( 𝑓,𝑝𝑎𝑟𝑎 );
14 if𝑡𝑦𝑝𝑒 _𝑎𝑛𝑛𝑜𝑡𝑎𝑡𝑖𝑜𝑛 is not𝑁𝑜𝑛𝑒 then
15 𝑚←constructObject( 𝑡𝑦𝑝𝑒 _𝑎𝑛𝑛𝑜𝑡𝑎𝑡𝑖𝑜𝑛 );
16 else
17 /* Retrieve information from the Knowledge base */
18 𝑚←RetrieveByFeature( 𝑓,𝑝𝑎𝑟𝑎 ,𝑡𝑦𝑝𝑒 );
19𝑀←𝑀∪𝑚
20return𝑀;3.2.2 Resolve type by referring to pre-existing instances. First, pa-
rameter type resolution can be facilitated by referring to pre-existing
instances. Given a focal function 𝑓, pre-existing instance refers to
usage of𝑓(e.g., invocation of target function) within the program.
Such instances can help LLM better understand how to invoke
𝑓, hence inferring the parameter type. For example, for function
set_num(a) , a pre-existing instance set_num(1) exists within the
project. When generating test cases, LLM can refer to this pre-
existing instance, easily infer the type of 𝑎asInteger . To obtain
pre-existing instances, TypeTest first obtains the call graph of the
whole program 𝑝under test. For each focal function, TypeTest
gets its pre-existing instances according to the call graph and gets
the context of invocation of 𝑓. The context includes the code to
create arguments to invoke 𝑓. For simplification, the context of pre-
existing instance 𝑝𝑖(which invokes 𝑓) is defined as the function
where the𝑝𝑖locates while eliminating the code after 𝑝𝑖. This is
because the arguments for invoking 𝑓are usually defined in the
local function before 𝑝𝑖.
After collecting all existing pre-existing instances of 𝑓, to avoid
overwhelming LLM, TypeTest sorts the collected instances accord-
ing to the length of code. The pre-existing instances with shorter
context are prioritized since the shorter context usually introduce
fewer noise and easier to be parsed by LLM. With the selected
pre-existing instance and its context, LLM then infer the parameter
types from the source code.
3.2.3 Resolve type by feature-based retrieval. Although referencing
pre-existing instances is an effective approach for type resolution,
some units may lack such instances. Moreover, due to Python’s
dynamic nature, no algorithm can construct a perfect call graph,
making the available pre-existing instances inherently incomplete.
As a result, relying on pre-existing instances is not always effective.
In this section, we explain how to resolve types using feature-
parameter retrieval, as described in Algorithm 1 line 17.
Our approach is inspired by the Duck Test ,3. This principle relies
on identifying an object’s behavior and characteristics to determine
its type, rather than requiring explicit type annotations. Specifically,
if a parameter para iof function 𝑓is a number, it can be utilized
in mathematical calculation. If para iis an object, it can be used to
access its field and invoke its associated method. Such information
can be used to infer their types. To utilize such information, we
define the feature of each parameter para ias:
𝑓𝑒𝑎𝑡 𝑖=operation(para i)+fieldAccess(para i)
+methodInvocation (para i)(1)
whereoperation(para i)represents the set of operation types with
para ias operand, such as “mathematic operation”, “string opera-
tion”, “array operation” and etc. Meanwhile, fieldAccess(para i)
andmethodInvocation (para i)represent the set of fields and meth-
ods (if exists) accessed via para i, respectively.
With the extracted features, if para iis of primitive type, such as
integer, string, it should be easily for LLM to recognize its type. If
para iis an object, TypeTest then retrieve the knowledge base to
search for a class definition, with
𝑐=Definition(𝐹𝑖𝑒𝑙𝑑)+Definition(𝑀𝑒𝑡ℎ𝑜𝑑)+getName(𝑐)(2)
3Duck Test: https://en.wikipedia.org/wiki/Duck_test

Conference’17, July 2017, Washington, DC, USA Runlin Liu, Zhe Zhang, Yunge Hu, Yuhang Lin, Xiang Gao, and Hailong Sun
where,Definition(𝐹𝑖𝑒𝑙𝑑)andDefinition(𝑀𝑒𝑡ℎ𝑜𝑑)represent the
fields and methods defined in class 𝑐. Such that, fieldAccess(para i)
is a subset of Definition(𝐹𝑖𝑒𝑙𝑑), andmethodInvocation (para iis
a subset of Definition(𝑀𝑒𝑡ℎ𝑜𝑑). Leveraging this property, Type-
Test filters the knowledge base accordingly. It then prompts the
LLMs to summarize the behavior of para ibased on its execution in
function𝑓and generates a query to retrieve relevant documentation
for class𝑐from the filtered knowledge base.
Example. Listing 4 demonstrates how the RAG retrieve parame-
ter type based on feature through a simple illustrative example. The
focal function, shown in Listing 5a, is from an open-source GitHub
project called code2flow4.code2flow is a tool that generates call
graphs and has garnered 4k stars on GitHub. The Call class de-
fines a field owner_token and a function matches_variable . The
matches_variable method of the Call class accepts a parameter
of typeVariable , performs some processing, and checks whether
thetoken of the passed variable equals the owner_token of the
instance to which the method belongs. If they are equal, it returns
thepoints_to of the passed variable .
When generating test cases without using the RAG process, the
parameter variable in thematches_variable method does not
contain type annotations, leaving LLM unaware of the parameter
type. To address this, a new class, MockVariable , was created to
simulate the behavior of the Variable class. However, the mocked
methodpoint_to_node in theMockVariable class differs from the
one defined in the Variable class, resulting in incorrect behaviors.
In contrast, TypeTest filters potential types based on the mem-
bers accessed by variable inmatches_variable . Furthermore,
the query is generated using the characteristics of variable , as
shown in Listing 5c. Even in the absence of type annotations in the
focal function, we can observe that relevant documentation for the
Variable class was retrieved, as illustrated in Listing 5d. With this
retrieved information, the LLM can generate accurate test cases, as
demonstrated in Listing 5e.
3.3 Assertion Generation Based on Function
Summarization
LLMs excel at understanding simple functions and generating cor-
responding test cases. However, when the focal function is highly
complex, especially when it invokes other functions, LLMs often
lack sufficient context to fully comprehend its behavior, leading to
incorrect assertion generation. To mitigate this issue, summarizing
the function can provide essential information while preventing
excessive context length. This section introduces how TypeTest
leverages the call graph to generate summaries for the focal func-
tion.
3.3.1 Topological traversal for function behavior analysis. A pre-
cise understanding of a function’s internal behavior is crucial for
generating effective test cases, as it enables LLMs to produce more
accurate assertions. However, in the case of a complex focal func-
tion𝑓, its call statements often obscure critical execution logic,
making it challenging for the model to fully comprehend 𝑓.
To address this, TypeTest organizes all functions in the program
in “topological order” based on call graph, ensuring that functions
4code2flow: https://github.com/scottrogowski/code2flowclass Call:
def matches_variable(self, variable):
if variable.point_to_node():
if variable.token == self.owner_token:
return variable.points_to
a: focal function
from model import *
class MockVariable:
def __init__(self, token, points_to):
self.token = token
self.points_to = points_to
def point_to_node(self):
pass
def test_Call_matches_variable_exact_match():
...
variable = MockVariable("token", node)
call = Call("token_example", "token")
assert call.matches_variable(variable) == None
b: Generated test case without RAG
What is the type of variable, which has a point_to_node
method and attributes token and points_to?
c: Query
class Variable:
def __init__(self, token, points_to):
self.token = token
self.points_to = points_to
def point_to_node(self):
return isinstance(self.points_to, Node)
d: Retrieved information
from model import Call
from model import Variable, Node
def test_Call_matches_variable_exact_match():
...
variable = Variable("obj", node)
call = Call("do_something", "obj")
assert call.matches_variable(variable) == node
e: Generated test case with RAG
Listing 4: An example explaining how RAG retrieve parame-
ter type based on type hint
are analyzed in a structured manner. To construct the call graph
of the project under test, we employ PyCG[ 39] to extract function
dependencies. Let 𝐶𝑎𝑙𝑙𝑒𝑑(𝑓)denote the set of all functions invoked
by𝑓,𝑓.𝑠𝑐represent the source code of 𝑓, and𝐴𝑛𝑎𝑙𝑦𝑠𝑖𝑠 denote the
behavior analysis method. The behavior of 𝑓is then computed as
follows:
f.behavior =Analysis(f.sc+∑︁
𝑥∈Called(f)x.behavior) (3)
For functions with recursive calls, TypeTest handles cycles by
temporarily removing one dependency edge during sorting. By
incorporating function behaviors in a structured manner, This ap-
proach provides LLMs with richer contextual information, thereby
enhancing their ability to comprehend function behavior.
3.3.2 Reverse topological traversal for high-level functional seman-
tics. While the previous step allows LLMs to capture fine-grained

LLM-based Unit Test Generation for Dynamically-Typed Programs Conference’17, July 2017, Washington, DC, USA
behavioral details of each function, it does not provide a holistic
understanding of a function’s high-level semantics. To address this
issue, TypeTest employs a reverse topological traversal, analyzing
functions from higher-level entry points downward to infer the
purpose of each function.
Let𝐶𝑎𝑙𝑙(𝑓)denote the set of functions that invoke a given func-
tion𝑓, and let𝐼𝑛𝑓𝑒𝑟 represent the inference mechanism. The se-
mantics of function 𝑓is then computed as follows:
f.semantics =Infer(f.sc+Call(f).sc+Call(f).semantics)(4)
For functions with zero in-degree in the call graph (i.e., top-level
functions), TypeTest searches for the nearest relevant documen-
tation file (e.g., README.md) to serve as a proxy for 𝐶𝑎𝑙𝑙(𝑓). By
leveraging the overall program context, LLMs can infer the seman-
tics of top-level functions more effectively, ultimately improving
assertion generation accuracy.
3.4 Generation with Repair and Iterative
Improvement
This section explains how TypeTest prompts LLM to generate
test cases and incrementally improve coverage through iterative
generation.
3.4.1 Test case generation and repair. We design a system prompt
to instruct the LLM. First, it defines the model’s role as an expert
Python programmer, a strategy that has proven effective in generat-
ing code based on prior research [ 49]. Additionally, we provide the
module location of 𝑓to ensure proper imports in the generated test
code. The prompt directs the LLM to generate pytest test cases for
𝑓using the previously gathered information. To improve response
quality, we employ a step-by-step reasoning approach, which prior
studies have shown enhances LLM outputs [14].
Since the size of 𝑓can vary, the prompt may occasionally ex-
ceed the maximum token limit. To mitigate this, we use OpenAI ’s
tiktoken library to truncate any parts that exceed the limit. As a
result, the ordering of components within the prompt becomes crit-
ical. Beyond the system prompt, we prioritize retaining the unit’s
source code and, when necessary, omit context that aids in assertion
generation. Ultimately, this process produces an initial test case 𝑡
for𝑓.
Despite providing a comprehensive context to the LLM, 𝑡may
still contain syntax and assertion errors. In straightforward cases,
LLM can identify and correct such issues based solely on the er-
ror report, without needing extra context. Therefore, for initial
repairs, we include both the source code of 𝑡and the error report
in the prompt. However, some errors, like module import errors,
may still persist due to insufficient contextual information avail-
able to the model. To address this, we leverage the RAG process,
as previously described, to retrieve relevant context dynamically.
This retrieval process often aligns well with the information in-
dicated in the error report. For instance, if an error report states,
“Unable to find module path: target_module” , the RAG pro-
cess can retrieve a relevant module path, such as “module path:
example.target_module” . Given the diverse range of potential
errors, it is impractical to define a fixed query template. Therefore,
we adopt a distributed approach, dividing the query generationtask into two steps: analyzing the cause of the error and generating
the corresponding query. LLM uses a chain of thought processes to
produce a coherent query. Finally, we use the source code of 𝑡, the
error report, and information retrieved by the query to repair the
error of𝑡. If the repair fails, we discard the test case.
3.4.2 Iterative unit test generation. One of the primary objectives
of test case generation is to maximize code coverage. To achieve
this, TypeTest adopts an iterative generation approach. After test-
ing all functions in 𝑝,TypeTest evaluates coverage and initiates
a new testing round if necessary. Previous studies have shown
that prompting LLMs to generate test cases targeting uncovered
statements can improve coverage [ 20]. Coverage data is typically
provided as line numbers corresponding to uncovered statements.
However, LLMs are not adept at numerical reasoning, particularly
counting line numbers. To address this, TypeTest directly anno-
tates uncovered statements with comments, thereby guiding the
model to generate test cases for these specific locations. In addition,
test cases generated in earlier rounds often facilitate the genera-
tion of new test cases. They provide a comprehensive process for
parameter initialization, acting as more effective pre-existing in-
stances. To capitalize on this, we store these cases in a knowledge
base, allowing TypeTest to leverage them for type inference and
retrieval via RAG.
4 Evaluation
In this section, we evaluate TypeTest by addressing the following
research questions.
RQ1: What is the effectiveness of TypeTest compared to
existing approaches?
RQ2: What is the impact of type inference techniques and
RAG on the effectiveness of TypeTest ?
RQ3: How does iterative strategy influence the improvement
ofTypeTest ’s coverage?
Benchmark .To address these questions, we utilize the bench-
mark provided by Pynguin [ 23], which we refer to as Pyn. This
benchmark has been widely adopted by tools such as CodaMosa [ 21]
and Coverup [ 20].Pyn comprises 17 real-world projects collected
from datasets such as BugsInPy [ 50] and ManyTypes4Py [ 28]. Fol-
lowing the selection criteria used by CodaMosa, we exclude mod-
ules that achieve 100% coverage within one minute, resulting in a
benchmark consisting of 125 modules. The statistics of the bench-
mark is shown in Table 1.
Baseline and Configuration. To evaluate TypeTest on the
Pyn benchmark, we use CodaMosa [ 21], CoverUp [ 20] and Chat-
Tester [ 58] as baselines. CodaMosa is a project in the field of unit
test generation, combining LLMs with search algorithms. It can
generate new test cases through intelligent queries for Python
projects, especially when encountering coverage bottlenecks. Chat-
Tester [ 58] is initially developed for Java-based test generation. For
comparison purposes, we adapt ChatTester to support Python pro-
grams by using a similar prompt. ChatTester relies on static code
analysis to obtain class information. CoverUp [ 20] is a novel sys-
tem that drives the generation of high-coverage Python regression
tests via a combination of coverage analysis and LLMs. The specific
experimental parameters are as follows:

Conference’17, July 2017, Washington, DC, USA Runlin Liu, Zhe Zhang, Yunge Hu, Yuhang Lin, Xiang Gao, and Hailong Sun
Table 1: Coverage results of different tools
PackageSize TypeTest CoverUp CodaMosa ChatTester
Mod num Stmt num Br num Stmt Cov Br Cov Stmt Cov Br Cov Stmt Cov Br Cov Stmt Cov Br Cov
apimd 2 515 300 88.7 % 85.3 % 67.6 % 56.7 % 75.0 % 68.0 % 29.9 % 13.7 %
black 6 1349 562 81.0 % 68.5 % 63.8 % 48.2 % 53.7 % 34.5 % 43.0 % 21.2 %
codetiming 1 40 12 100.0 % 100.0 % 90.0 % 75.0 % 100.0 % 100.0 % 92.5 % 83.3 %
dataclasses_json 4 588 270 85.9 % 81.1 % 76.4 % 65.2 % 43.0 % 20.0 % 68.4 % 59.6 %
docstring_parser 5 388 144 95.1 % 84.7 % 95.9 % 86.8 % 96.6 % 88.9 % 84.5 % 66.7 %
flutes 3 250 142 99.6 % 88.7 % 85.2 % 69.7 % 96.8 % 83.1 % 88.8 % 74.6 %
httpie 19 1516 519 82.5 % 64.5 % 79.0 % 59.3 % 78.0 % 56.8 % 57.0 % 27.9 %
isort 2 148 26 95.3 % 80.8 % 93.9 % 80.8 % 93.9 % 76.9 % 92.6 % 80.8 %
mimesis 18 1115 334 98.7 % 95.8 % 84.6 % 69.8 % 95.7 % 91.3 % 86.6 % 79.9 %
py_backwards 16 586 168 80.5 % 78.0 % 65.9 % 50.0 % 81.9 % 76.2 % 52.4 % 26.8 %
pymonet 10 442 124 98.6 % 96.8 % 88.9 % 83.9 % 95.2 % 75.0 % 86.2 % 71.8 %
pypara 6 1083 175 85.1 % 69.1 % 85.5 % 66.3 % 88.2 % 58.3 % 69.4 % 40.0 %
pytutils 12 263 72 73.4 % 70.8 % 53.2 % 40.3 % 74.5 % 77.8 % 61.6 % 52.8 %
semantic_release 6 424 86 96.7 % 95.3 % 99.3 % 97.7 % 73.6 % 46.5 % 74.3 % 45.3 %
string_utils 3 403 160 99.3 % 98.8 % 94.8 % 88.8 % 100.0 % 100.0 % 76.4 % 69.4 %
sty 2 99 46 94.9 % 93.5 % 88.9 % 78.3 % 100.0 % 100.0 % 58.6 % 34.8 %
typesystem 10 1420 681 76.9 % 65.6 % 65.7 % 53.5 % 95.5 % 91.8 % 51.8 % 29.1 %
Average - - - 86.7 % 77.1 % 77.4 % 62.0 % 81.2 % 67.5 % 63.1 % 41.1 %
Standard Deviation - - - 8.6 % 11.9 % 13.0 % 15.6 % 16.2 % 22.7 % 18.0 % 22.6 %
•For RAG configuration of TypeTest , we employ the "BAAI/bge-
large-en-v1.5" model available on Hugging Face for embeddings5.
To ensure efficient and scalable querying, TypeTest leverages
Chroma6to build vector-based indices, enabling fast retrieval.
•For ChatTester, CoverUp, and TypeTest , we use gpt-4o-2024-05-
13 with the same model parameters. Specifically, we configure all
three methods to perform three rounds of test generation, where
each round generates a test case for every function in the target
project.
•Noting that CodaMosa(gpt4o) performs slightly worse than Co-
daMosa(codex), consistent with the findings of the CoverUp
study [ 20], and considering that Codex is no longer available,
we utilize the experimental data provided by CodaMosa for com-
parison. This dataset is publicly available at https://github.com/
microsoft/codamosa-dataset.
4.1 RQ1: Effectiveness of TypeTest
Coverage .Following CoverUp’s approach, which measures mod-
ule level coverage [ 20], we evaluate the statement and branch cov-
erage of tests generated by TypeTest in comparison to CoverUp,
ChatTester, and CodaMosa.
Figure 1 shows the experimental results. In terms of cover-
age,TypeTest achieved the highest statement coverage average
(86.6%) compared to CoverUp (77.4%), CodaMosa (81.2%) and Chat-
Tester (63.1%). Additionally, it boasts the highest branch coverage
of 76.8%, surpassing CoverUp (62.0%), CodaMosa (67.5%), and Chat-
Test (41.1%). These results indicate that the test samples generated
byTypeTest provide better coverage of the tested project. Our
improvements mainly occur on programs with a large number of
dynamic types, on which CodaMosa and CoverUp achieve very low
coverage, even 0% for some cases.
In terms of stability, TypeTest (8.6%) exhibits a lower standard de-
viation compared to CodaMosa (16.2%), CoverUp (13.0%), and Chat-
Tester (18.0%) in the coverage of statement. Moreover, when com-
paring the standard deviations of branch coverage with CoverUp
(15.6%), CodaMosa (22.7%) and ChatTester (22.6%), TypeTest (11.9%)
5https://huggingface.co/BAAI/bge-large-en-v1.5
6https://github.com/chroma-core/chromahas the second lowest standard deviation. The factor that affects
standard deviation is the low coverage. Meanwhile, TypeTest also
achieves the highest lower bound, with the lowest proportion of
cases where statement coverage is below 40%, at only 3.2%. In con-
trast, CoverUp, CodaMosa, and ChatTester exhibit proportions of
15.2%, 9.6%, and 24.0%, respectively. This indicates that TypeTest
demonstrates greater stability in most scenarios.
Since CodaMosa achieves the second-highest coverage after
TypeTest , we select CodaMosa for further analysis. In the modules
where TypeTest achieves higher coverage, the most significant
difference is observed in the 𝑒𝑥𝑐𝑠 module of the 𝑝𝑦𝑡𝑢𝑡𝑖𝑙𝑠 project,
where TypeTest attains 100% coverage while CodaMosa achieves
0%. This module contains a single function, 𝑜𝑘, which suppresses
exceptions specified by the 𝑒𝑥𝑐𝑒𝑝𝑡𝑖𝑜𝑛𝑠 parameter. CodaMosa fails
to generate valid inputs for this parameter, whereas TypeTest suc-
ceeds, leading to full coverage.
In contrast, in modules where TypeTest has lower coverage than
CodaMosa, the lowest coverage is observed in the 𝑡𝑟𝑎𝑛𝑠𝑓𝑜𝑟𝑚𝑒𝑟𝑠
module of the 𝑝𝑦_𝑏𝑎𝑐𝑘𝑤𝑎𝑟𝑑𝑠 project. This module utilizes the third-
party library 𝑡𝑦𝑝𝑒𝑑 _𝑎𝑠𝑡, which is aliased as 𝑎𝑠𝑡. LLMs mistakenly
identify it as Python’s standard 𝑎𝑠𝑡module, preventing effective test
case generation and limiting coverage improvement. Additionally,
while CodaMosa does not generate assertions, TypeTest produces
meaningful assertions and discards test cases when assertion fail-
ures occur. This behavior contributes to lower coverage in some
modules compared to test cases generated by CodaMosa.
Test Quality .High test coverage does not necessarily imply the
effectiveness of assertions; however, mutation testing can assess
assertion effectiveness by simulating faults [ 60]. Therefore, we use
mutation testing to further evaluate the quality of assertions gener-
ated by TypeTest . Since CodaMosa does not generate assertions,
we use mutmut7to measure the mutation score of test cases gen-
erated by the other three tools. In total, mutmut produced 5,341
mutants.
7https://github.com/boxed/mutmut

LLM-based Unit Test Generation for Dynamically-Typed Programs Conference’17, July 2017, Washington, DC, USA
Figure 6: Mutation Score of different
toolsFigure 6 presents
the mean and vari-
ance of the muta-
tion scores achieved
by the three tools.
As shown, TypeTest
achieves an average
mutation score 19.8%
higher than CoverUp
and 23.8% higher than
ChatTester. Addition-
ally, the standard de-
viation of TypeTest ’s
mutation score is 9.4% lower than that of Coverup and 11.2% lower
than that of ChatTester. These results indicate that TypeTest sur-
passes the current state-of-the-art in both performance and stability
in mutation testing.
In summary, TypeTest can generate high coverage and
high mutation score tests more stably in test generation
than the previous state of the art.
4.2 RQ2: Ablation study on type inference
techniques and RAG
In this RQ, we conduct two experiments: replacing the type in-
ference tool to compare it with our type inference method, and
removing the RAG process to evaluate the overall impact of RAG.
Since the core advantage of TypeTest lies in providing LLMs
with ample contextual information based on type inference, it is
worth investigating whether existing type inference tools can ac-
complish the same task. Numerous Python type-inference tools
have been developed, such as Type4Py [ 29] and Hityper [ 35]. Among
them, Hityper effectively integrates static inference with deep learn-
ing techniques, making it the state-of-the-art solution.
Figure 7: Coverage results of TypeTest variants
To investigate the ability of Hityper in assisting test case gener-
ation, we removed the type inference component from TypeTest .
We then integrate Hityper to infer the types of focal function pa-
rameters and use static analysis to retrieve the corresponding class
definitions. LLMs are then employed to refine the extracted context,
constructing a more informative input. We denote this modified
version as TypeTest with Hityper. To further explore the overall
contribution of RAG to TypeTest , we removed all RAG componentsfrom TypeTest and named the modified version TypeTest without
RAG.
Figure 7 presents the coverage results of TypeTest ,TypeTest
with Hityper and TypeTest without RAG. From the figure, it is
evident that TypeTest consistently achieves higher statement and
branch coverage than the other two tools. Additionally, TypeTest
with Hityper generally outperforms TypeTest without RAG. We
further investigate the reasons behind these results.
On average, TypeTest achieves 6% higher statement coverage
than TypeTest with Hityper. This discrepancy is mainly due to
inaccuracies in Hityper’s parameter type inference. On one hand,
static inference struggles with parameter type analysis because
parameters act as function entry points for data flow, often lacking
explicit definitions or assignments [ 35]. On the other hand, deep
learning-based type inference performs poorly on rare types in
the dataset, particularly user-defined types. For example, in the
ℎ𝑡𝑡𝑝𝑖𝑒 project module 𝑝𝑙𝑢𝑔𝑖𝑛𝑠.𝑚𝑎𝑛𝑎𝑔𝑒𝑟 , many functions require a
parameter of type 𝑇𝑦𝑝𝑒[𝐵𝑎𝑠𝑒𝑃𝑙𝑢𝑔𝑖𝑛], where𝐵𝑎𝑠𝑒𝑃𝑙𝑢𝑔𝑖𝑛 is a user-
defined type. However, Hityper incorrectly infers this type as 𝑠𝑡𝑟,
leading to the generation of low-coverage, low-quality tests. In
contrast, the RAG system of TypeTest retrieves relevant contextual
information based on the parameter’s behavior within the function,
enabling the generation of test cases that achieve 100% coverage.
The average statement coverage of TypeTest exceeds that of
TypeTest without RAG by 11.9%. Compared to TypeTest ,TypeTest
without RAG lacks the ability to dynamically retrieve contextual
information from the project, making it less effective in handling
larger projects. For the five largest projects, TypeTest achieves
13.4% higher average statement coverage than TypeTest without
RAG, while for smaller projects, the improvement is 9.6%.
In summary, TypeTest ’s type inference is more effective
than Hityper in supporting test generation, as it better
handles user-defined types. Additionally, the RAG mecha-
nism plays a crucial role in dynamically retrieving relevant
context, significantly improving test coverage and overall
effectiveness.
Figure 8: TypeTest Statement coverage for each iteration

Conference’17, July 2017, Washington, DC, USA Runlin Liu, Zhe Zhang, Yunge Hu, Yuhang Lin, Xiang Gao, and Hailong Sun
4.3 RQ3: Impact of Iterative Strategy
To assess the effectiveness of the iterative strategy, we conduct
experiments across multiple iterations. Figure 8 illustrates the state-
ment coverage achieved for each project over three iterations. The
iterative strategy significantly improves the statement coverage,
with the average coverage improving by 10.8% for the third iter-
ation. For example, for the project httpie , when using iterative
strategy, the statement coverage increases 22.4%. Further investiga-
tion, we observe that the iterative strategy performs better on the
projects whose first test generation’s coverage is low. We calculate
the coverage improvement on the project whose coverage is below
80% after the first iteration. The results show that the averaged
coverage increases by 15.3%, which highlights the help of iterative
strategy on the projects with low coverage. These results confirm
that iterative strategy boosts their ability to provide full coverage
of the tested code by increasing the low coverage.
In summary, the results confirm that the iterative strategy
improves overall code coverage.
5 Threats to Validity
The effectiveness of TypeTest faces challenges due to variations in
LLMs’ performance, particularly regarding the applicability of LLMs
and their stability when handling projects with extensive codebases.
Although we achieved favorable results using GPT-4o, different
LLMs may yield varied outcomes, which impacts TypeTest ’s overall
performance. Additionally, as RAG relies on similarity matching
and the token limitation of LLMs, TypeTest ’s performance may
degrade when processing a single function with a large number
of lines of code. Future work will focus on improving scalability
for large codebases, minimizing retrieval omissions in the RAG
process, and testing a broader range of LLMs to enhance TypeTest ’s
generalization.
6 Related Work
Our work relates to the following areas: Search-Based Software
Testing (SBST), and LLM-based unit test generation.
6.1 Search Based Software Testing (SBST)
SBST employs search algorithms to automatically generate test
cases [ 15,24,42,47], which can greatly reduce the time developers
spend on test case creation [ 44], while also generating boundary
cases and exceptional inputs that are often challenging to identify
manually. Various SBST-based tools and algorithms have been devel-
oped to generate test cases for programming languages such as Java,
Python, and JavaScript, some of which support multiple testing
objectives, including statement and branch coverage [ 10,12,18,30].
EvoSuite [ 22,45,46] is a well-known tool based on SBST, which
automatically generates JUnit test suites that maximize code cover-
age. Randoop [ 33] is a feedback-directed random test generation
tool that generates test cases by randomly combining previously
executed statements that did not result in failures. Algorithm MOSA
(Multi-Objective Simulated Annealing) [ 43], and algorithms related
to it, such as DynaMOSA [ 34] and MIO (Many Independent Objec-
tive) [ 4,5], are used for test generation, and particularly effective athandling multiple objectives, containing line coverage, branch cov-
erage, and multiple mutants in mutation testing. Sapienz [ 6,27,31]
is an approach to Android testing that uses multi-objective SBST to
optimize test sequences for brevity and effectiveness in revealing
faults. Sapienze leverages a combination of random fuzzing, system-
atic exploration, and SBST. Pynguin is an extendable test-generation
framework for Python, which is a dynamic type programming lan-
guage [17, 23, 25, 37].
Despite the development of many SBST-based test case gener-
ation tools, they have notable limitations. These tools typically
produce only boundary condition test cases, with assertions limited
to simple equality checks. Additionally, when software updates
occur, generating new test cases can be time-consuming, even for
minimal changes. Furthermore, inaccurate type inference restricts
their effectiveness, particularly in dynamically-typed languages
like Python, which employs duck typing [ 24]. In contrast, Type-
Test generates test cases providing more accurate type inference,
enhancing overall test effectiveness.
6.2 LLM-Based Unit Test Generation
LLM-based unit test generation [ 41,55] is the technique that lever-
ages large language models trained to automatically generate unit
tests for software code. There has been a large number of works
or tools generating test cases with Large Language Models (LLMs)
[52,54,56], demonstrating impressive results. MuTAP [ 8] is a
prompt-based learning technique to generate effective test cases
with LLMs, which improves the effectiveness of test cases gener-
ated by LLMs in terms of revealing bugs by leveraging mutation
testing. TestPilot [ 40] is a tool for automatically generating unit
tests for npm packages written in JavaScript/TypeScript using LLM,
which provides LLM with the signature and implementation of
the function under test, along with usage extracted from the doc-
umentation. ChatUniTest [ 7,51] utilizes an LLM-based approach
encompassing valuable context in prompts and rectifying errors
in generated unit tests. ChatTester [ 58] is a Maven plugin similar
to ChatUnitTest above, which leverages ChatGPT to improve the
quality of its generated tests. LLM4Fin [ 53] is designed for testing
real-world stock-trading software, which generates comprehensive
testable scenarios for extracted business rules. CodaMosa [ 21] de-
veloped by Microsoft combines SBST and LLMs, using LLM to help
SBST’s exploration. LLM will provide examples for SBST when its
coverage improvements stall to help SBST search more useful areas.
Recent work [ 59] explores automated assertion generation with
LLMs, conducting a large-scale evaluation of different models and
demonstrating their effectiveness in improving assertion quality
and detecting real-world bugs. Reed et al. [ 36] investigates state-
based testing techniques for object-oriented software, leveraging
class-based unit tests and state-driven methodologies to improve
testing effectiveness in large-scale systems.
While existing tools rely on prompts to guide LLMs in generating
test cases, they fail to incorporate type information from previous
test cases. For instance, TestPilot does not adapt its prompts based
on type information, nor does it refine them when type information
does not improve. Similarly, CodaMosa only prompts the LLM when
necessary to support SBST, positioning the LLM as a supplementary
component rather than a primary driver of test case generation. In

LLM-based Unit Test Generation for Dynamically-Typed Programs Conference’17, July 2017, Washington, DC, USA
contrast, our tool, TypeTest , leverages type information to guide
the LLM in generating test cases that are more likely to improve
coverage. This is achieved through the use of retrieval-augmented
generation (RAG) [ 38], which integrates the strengths of both re-
trieval and generative models, thereby enriching the LLM with
additional, context-specific information.
7 Conclusion
We proposed RAG-based type inference for Unit Test Generation,
a method that enhances the authenticity and correctness of test
cases for dynamically typed languages by utilizing accurate type
information during testing. By leveraging call instance retrieval and
feature-based retrieval, TypeTest improved the capability of LLMs
to construct correct test objects and invoke the function under test.
Furthermore, TypeTest enhanced the ability of LLMs to generate
assertions by employing call graph-based analysis. TypeTest also
employed an iterative strategy, utilizing the generated test cases
and coverage information to aid in subsequent generations and
improve coverage. We evaluated our approach on 17 open-source
projects and found that our performance was superior and more
stable compared to other tools. Future work will focus on extending
TypeTest to other dynamically-typed languages and improving its
efficiency.
References
[1]Miltiadis Allamanis, Earl T Barr, Soline Ducousso, and Zheng Gao. 2020. Typilus:
Neural type hints. In Proceedings of the 41st acm sigplan conference on program-
ming language design and implementation . 91–105.
[2]Christopher Anderson, Paola Giannini, and Sophia Drossopoulou. 2005. Towards
type inference for JavaScript. In ECOOP 2005-Object-Oriented Programming: 19th
European Conference, Glasgow, UK, July 25-29, 2005. Proceedings 19 . Springer,
428–452.
[3]J. H. Andrews, T. Menzies, and F. C. H. Li. 2011. Genetic algorithms for randomized
unit testing. IEEE Transactions on Software Engineering 37, 1 (2011), 80–94.
[4]Andrea Arcuri. 2017. Many independent objective (MIO) algorithm for test suite
generation. In Search Based Software Engineering: 9th International Symposium,
SSBSE 2017, Paderborn, Germany, September 9-11, 2017, Proceedings 9 . Springer,
3–17.
[5]Andrea Arcuri. 2018. Test suite generation with the Many Independent Objective
(MIO) algorithm. Information and Software Technology 104 (2018), 195–206.
[6]Iván Arcuschin, Juan Pablo Galeotti, and Diego Garbervetsky. 2023. An Empirical
Study on How Sapienz Achieves Coverage and Crash Detection. Journal of
Software: Evolution and Process 35, 4 (2023), e2411.
[7]Yinghao Chen, Zehao Hu, Chen Zhi, Junxiao Han, Shuiguang Deng, and Jianwei
Yin. 2024. Chatunitest: A framework for llm-based test generation. In Compan-
ion Proceedings of the 32nd ACM International Conference on the Foundations of
Software Engineering . 572–576.
[8]Arghavan Moradi Dakhel, Amin Nikanjam, Vahid Majdinasab, Foutse Khomh,
and Michel C Desmarais. 2024. Effective test generation using pre-trained large
language models and mutation testing. Information and Software Technology 171
(2024), 107468.
[9]Matthew C Davis, Sangheon Choi, Sam Estep, Brad A Myers, and Joshua Sunshine.
2023. NaNofuzz: A Usable Tool for Automatic Test Generation. In Proceedings of
the 31st ACM Joint European Software Engineering Conference and Symposium on
the Foundations of Software Engineering . 1114–1126.
[10] Pouria Derakhshanfar and Xavier Devroey. 2023. Basic block coverage for unit
test generation at the SBST 2022 tool competition. In Proceedings of the 15th
Workshop on Search-Based Software Testing (Pittsburgh, Pennsylvania) (SBST ’22) .
Association for Computing Machinery, New York, NY, USA, 37–38. doi:10.1145/
3526072.3527528
[11] Matouš Eibich, Shivay Nagpal, and Alexander Fred-Ojala. 2024. ARAGOG: Ad-
vanced RAG output grading. arXiv preprint arXiv:2404.01037 (2024).
[12] Raihana Ferdous, Chia-kang Hung, Fitsum Kifetew, Davide Prandi, and Angelo
Susi. 2023. EvoMBT at the SBST 2022 tool competition. In Proceedings of the 15th
Workshop on Search-Based Software Testing (Pittsburgh, Pennsylvania) (SBST ’22) .
Association for Computing Machinery, New York, NY, USA, 51–52. doi:10.1145/
3526072.3527534[13] Michael Furr, Jong-hoon An, Jeffrey S Foster, and Michael Hicks. 2009. Static
type inference for Ruby. In Proceedings of the 2009 ACM symposium on Applied
Computing . 1859–1866.
[14] Andrew Gao. 2023. Prompt engineering for large language models. Available at
SSRN 4504303 (2023).
[15] Sepideh Kashefi Gargari and Mohammd Reza Keyvanpour. 2021. SBST challenges
from the perspective of the test techniques. In 2021 12th International Conference
on Information and Knowledge Technology (IKT) . 119–123. doi:10.1109/IKT54664.
2021.9685297
[16] Patrice Godefroid, Nils Klarlund, and Koushik Sen. 2005. DART: Directed auto-
mated random testing. In Proceedings of the 2005 ACM SIGPLAN conference on
Programming language design and implementation . ACM, 213–223.
[17] Lucca Guerino and Auri Vincenzi. 2023. An Experimental Study Evaluating Cost,
Adequacy, and Effectiveness of Pynguin’s Test Sets. In Proceedings of the 8th
Brazilian Symposium on Systematic and Automated Software Testing . 5–14.
[18] Giovani Guizzo and Sebastiano Panichella. 2023. Fuzzing vs SBST: Intersections
& Differences. SIGSOFT Softw. Eng. Notes 48, 1 (Jan. 2023), 105–107. doi:10.1145/
3573074.3573102
[19] Sungjae Hwang, Sungho Lee, Jihoon Kim, and Sukyoung Ryu. 2021. Justgen:
Effective test generation for unspecified JNI behaviors on jvms. In 2021 IEEE/ACM
43rd International Conference on Software Engineering (ICSE) . IEEE, 1708–1718.
[20] Emery D. Berger Juan Altmayer Pizzorno. 2024. CoverUp: Coverage-Guided LLM-
Based Test GenerationCoverUp: Coverage-Guided LLM-Based Test Generation.
arXiv preprint arXiv:2403.16218, 2024 - arxiv.org (2024).
[21] Caroline Lemieux, Jeevana Priya Inala, Shuvendu K Lahiri, and Siddhartha Sen.
2023. Codamosa: Escaping coverage plateaus in test generation with pre-trained
large language models. In 2023 IEEE/ACM 45th International Conference on Soft-
ware Engineering (ICSE) . IEEE, 919–931.
[22] Yun Lin, You Sheng Ong, Jun Sun, Gordon Fraser, and Jin Song Dong. 2021.
Graph-Based Seed Object Synthesis for Search-Based Unit Testing. In Proceedings
of the 29th ACM Joint Meeting on European Software Engineering Conference
and Symposium on the Foundations of Software Engineering (Athens, Greece)
(ESEC/FSE 2021) . Association for Computing Machinery, New York, NY, USA,
1068–1080. doi:10.1145/3468264.3468619
[23] Stephan Lukasczyk and Gordon Fraser. 2022. Pynguin: Automated unit test gen-
eration for python. In Proceedings of the ACM/IEEE 44th International Conference
on Software Engineering: Companion Proceedings . 168–172.
[24] Stephan Lukasczyk, Florian Kroiß, and Gordon Fraser. 2021. An Empirical Study
of Automated Unit Test Generation for Python. CoRR abs/2111.05003 (2021).
arXiv:2111.05003 https://arxiv.org/abs/2111.05003
[25] Stephan Lukasczyk, Florian Kroiß, and Gordon Fraser. 2023. An empirical study
of automated unit test generation for Python. Empirical Software Engineering 28,
2 (2023), 36.
[26] Lin Ma, Cyril Artho, Chao Zhang, et al .2015. Grt: Program-analysis-guided
random testing. In 2015 30th IEEE/ACM International Conference on Automated
Software Engineering (ASE) . IEEE, 212–223.
[27] Ke Mao, Mark Harman, and Yue Jia. 2016. Sapienz: Multi-objective automated
testing for android applications. In Proceedings of the 25th international symposium
on software testing and analysis . 94–105.
[28] Amir M Mir, Evaldas Latoškinas, and Georgios Gousios. 2021. Manytypes4py: A
benchmark python dataset for machine learning-based type inference. In 2021
IEEE/ACM 18th International Conference on Mining Software Repositories (MSR) .
IEEE, 585–589.
[29] Amir M Mir, Evaldas Latoškinas, Sebastian Proksch, and Georgios Gousios. 2022.
Type4py: Practical deep similarity learning-based type inference for python. In
Proceedings of the 44th International Conference on Software Engineering . 2241–
2252.
[30] Mahshid Helali Moghadam, Markus Borg, and Seyed Jalaleddin Mousavirad. 2021.
Deeper at the SBST 2021 Tool Competition: ADAS Testing Using Multi-Objective
Search. In 2021 IEEE/ACM 14th International Workshop on Search-Based Software
Testing (SBST) . 40–41. doi:10.1109/SBST52555.2021.00018
[31] Iván Arcuschin Moreno, Juan Pablo Galeotti, and Diego Garbervetsky. 2020. Algo-
rithm or Representation? An empirical study on how SAPIENZ achieves coverage.
InProceedings of the IEEE/ACM 1st International Conference on Automation of
Software Test . 61–70.
[32] Wonseok Oh and Hakjoo Oh. 2022. PyTER: effective program repair for Python
type errors. In Proceedings of the 30th ACM Joint European Software Engineering
Conference and Symposium on the Foundations of Software Engineering . 922–934.
[33] Carlos Pacheco and Michael D. Ernst. 2007. Randoop: feedback-directed random
testing for Java. In Companion to the 22nd ACM SIGPLAN Conference on Object-
Oriented Programming Systems and Applications Companion (Montreal, Quebec,
Canada) (OOPSLA ’07) . Association for Computing Machinery, New York, NY,
USA, 815–816. doi:10.1145/1297846.1297902
[34] Annibale Panichella, Fitsum Meshesha Kifetew, and Paolo Tonella. 2017. Au-
tomated test case generation as a many-objective optimisation problem with
dynamic selection of the targets. IEEE Transactions on Software Engineering 44, 2
(2017), 122–158.

Conference’17, July 2017, Washington, DC, USA Runlin Liu, Zhe Zhang, Yunge Hu, Yuhang Lin, Xiang Gao, and Hailong Sun
[35] Yun Peng, Cuiyun Gao, Zongjie Li, Bowei Gao, David Lo, Qirun Zhang, and
Michael Lyu. 2022. Static inference meets deep learning: a hybrid type infer-
ence approach for python. In Proceedings of the 44th International Conference on
Software Engineering . 2019–2030.
[36] HG Reed, CD Turner, JB Aibel, and JT Dalton. 2025. Practical object-oriented
state-based unit testing. WIT Transactions on Information and Communication
Technologies 9 (2025).
[37] Mikael Ebrahimi Salari, Eduard Paul Enoiu, Cristina Seceleanu, Wasif Afzal, and
Filip Sebek. 2023. Automating test generation of industrial control software
through a plc-to-python translation framework and pynguin. In 2023 30th Asia-
Pacific Software Engineering Conference (APSEC) . IEEE, 431–440.
[38] Alireza Salemi and Hamed Zamani. 2024. Evaluating retrieval quality in retrieval-
augmented generation. In Proceedings of the 47th International ACM SIGIR Con-
ference on Research and Development in Information Retrieval . 2395–2400.
[39] Vitalis Salis, Thodoris Sotiropoulos, Panos Louridas, Diomidis Spinellis, and
Dimitris Mitropoulos. 2021. Pycg: Practical call graph generation in python.
In2021 IEEE/ACM 43rd International Conference on Software Engineering (ICSE) .
IEEE, 1646–1657.
[40] Max Schäfer, Sarah Nadi, Aryaz Eghbali, and Frank Tip. 2023. An empirical
evaluation of using large language models for automated unit test generation.
IEEE Transactions on Software Engineering (2023).
[41] Max Schäfer, Sarah Nadi, Aryaz Eghbali, and Frank Tip. 2024. An Empirical
Evaluation of Using Large Language Models for Automated Unit Test Generation.
IEEE Transactions on Software Engineering 50, 1 (2024), 85–105. doi:10.1109/TSE.
2023.3334955
[42] Yutian Tang, Zhijie Liu, Zhichao Zhou, and Xiapu Luo. 2024. ChatGPT vs SBST:
A Comparative Assessment of Unit Test Suite Generation. IEEE Transactions on
Software Engineering 50, 6 (2024), 1340–1359. doi:10.1109/TSE.2024.3382365
[43] Ekunda Lukata Ulungu, JFPH Teghem, PH Fortemps, and Daniel Tuyttens. 1999.
MOSA method: a tool for solving multiobjective combinatorial optimization
problems. Journal of multicriteria decision analysis 8, 4 (1999), 221.
[44] M S Vasudevan, Santosh Biswas, and Aryabartta Sahu. 2021. Automated Low-Cost
SBST Optimization Techniques for Processor Testing. In 2021 34th International
Conference on VLSI Design and 2021 20th International Conference on Embedded
Systems (VLSID) . 299–304. doi:10.1109/VLSID51830.2021.00056
[45] Sebastian Vogl, Sebastian Schweikl, and Gordon Fraser. 2021. Encoding the
certainty of boolean variables to improve the guidance for search-based test
generation. In Proceedings of the Genetic and Evolutionary Computation Conference .
1088–1096.
[46] Sebastian Vogl, Sebastian Schweikl, Gordon Fraser, Andrea Arcuri, Jose Campos,
and Annibale Panichella. 2021. EVOSUITE at the SBST 2021 Tool Competition.
In2021 IEEE/ACM 14th International Workshop on Search-Based Software Testing
(SBST) . IEEE, 28–29.
[47] Junjie Wang, Yuchao Huang, Chunyang Chen, Zhe Liu, Song Wang, and Qing
Wang. 2024. Software Testing With Large Language Models: Survey, Landscape,
and Vision. IEEE Transactions on Software Engineering 50, 4 (2024), 911–936.
doi:10.1109/TSE.2024.3368208
[48] Anjiang Wei, Yinlin Deng, Chenyuan Yang, and Lingming Zhang. 2022. Free
lunch for testing: Fuzzing deep-learning libraries from open source. In Proceedings
of the 44th International Conference on Software Engineering . 995–1007.
[49] Jules White, Quchen Fu, Sam Hays, Michael Sandborn, Carlos Olea, Henry Gilbert,
Ashraf Elnashar, Jesse Spencer-Smith, and Douglas C Schmidt. 2023. A prompt
pattern catalog to enhance prompt engineering with chatgpt. arXiv preprint
arXiv:2302.11382 (2023).
[50] Ratnadira Widyasari, Sheng Qin Sim, Camellia Lok, Haodi Qi, Jack Phan, Qijin
Tay, Constance Tan, Fiona Wee, Jodie Ethelda Tan, Yuheng Yieh, et al .2020.
Bugsinpy: a database of existing bugs in python programs to enable controlled
testing and debugging studies. In Proceedings of the 28th ACM joint meeting on
european software engineering conference and symposium on the foundations of
software engineering . 1556–1560.
[51] Zhuokui Xie, Yinghao Chen, Chen Zhi, Shuiguang Deng, and Jianwei Yin. 2023.
ChatUniTest: a ChatGPT-based automated unit test generation tool. arXiv preprint
arXiv:2305.04764 (2023).
[52] Congying Xu, Songqiang Chen, Jiarong Wu, Shing-Chi Cheung, Valerio Ter-
ragni, Hengcheng Zhu, and Jialun Cao. 2024. MR-Adopt: Automatic Deduction
of Input Transformation Function for Metamorphic Testing. arXiv preprint
arXiv:2408.15815 (2024).
[53] Zhiyi Xue, Liangguo Li, Senyue Tian, Xiaohong Chen, Pingping Li, Liangyu Chen,
Tingting Jiang, and Min Zhang. 2024. LLM4Fin: Fully Automating LLM-Powered
Test Case Generation for FinTech Software Acceptance Testing. In Proceedings of
the 33rd ACM SIGSOFT International Symposium on Software Testing and Analysis
(Vienna, Austria) (ISSTA 2024) . Association for Computing Machinery, New York,
NY, USA, 1643–1655. doi:10.1145/3650212.3680388
[54] Chen Yang, Junjie Chen, Bin Lin, Jianyi Zhou, and Ziqi Wang. 2024. Enhancing
LLM-based Test Generation for Hard-to-Cover Branches via Program Analysis.
arXiv preprint arXiv:2404.04966 (2024).
[55] Lin Yang, Chen Yang, Shutao Gao, Weijing Wang, Bo Wang, Qihao Zhu, Xiao Chu,
Jianyi Zhou, Guangtai Liang, Qianxiang Wang, et al .2024. An empirical study ofunit test generation with large language models. arXiv preprint arXiv:2406.18181
(2024).
[56] Lin Yang, Chen Yang, Shutao Gao, Weijing Wang, Bo Wang, Qihao Zhu, Xiao Chu,
Jianyi Zhou, Guangtai Liang, Qianxiang Wang, et al .2024. On the Evaluation
of Large Language Models in Unit Test Generation. In Proceedings of the 39th
IEEE/ACM International Conference on Automated Software Engineering . 1607–
1619.
[57] Qian Yang, J Jenny Li, and David Weiss. 2006. A survey of coverage based testing
tools. In Proceedings of the 2006 international workshop on Automation of software
test. 99–103.
[58] Zhiqiang Yuan, Yiling Lou, Mingwei Liu, Shiji Ding, Kaixin Wang, Yixuan Chen,
and Xin Peng. 2023. No more manual tests? evaluating and improving chatgpt
for unit test generation. arXiv preprint arXiv:2305.04207 (2023).
[59] Quanjun Zhang, Weifeng Sun, Chunrong Fang, Bowen Yu, Hongyan Li, Meng
Yan, Jianyi Zhou, and Zhenyu Chen. 2025. Exploring automated assertion gener-
ation via large language models. ACM Transactions on Software Engineering and
Methodology 34, 3 (2025), 1–25.
[60] Yucheng Zhang and Ali Mesbah. 2015. Assertions are strongly correlated with test
suite effectiveness. In Proceedings of the 2015 10th Joint Meeting on Foundations
of Software Engineering . 214–224.
[61] Zhe Zhang, Xingyu Liu, Yuanzhang Lin, Xiang Gao, Hailong Sun, and Yuan Yuan.
2024. LLM-based Unit Test Generation via Property Retrieval. arXiv preprint
arXiv:2410.13542 (2024).
[62] Zhichao Zhou, Yuming Zhou, Chunrong Fang, Zhenyu Chen, and Yutian Tang.
2022. Selectively combining multiple coverage goals in search-based unit test
generation. In Proceedings of the 37th IEEE/ACM International Conference on
Automated Software Engineering . 1–12.