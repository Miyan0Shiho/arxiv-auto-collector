# ComplexVCoder: An LLM-Driven Framework for Systematic Generation of Complex Verilog Code

**Authors**: Jian Zuo, Junzhe Liu, Xianyong Wang, Yicheng Liu, Navya Goli, Tong Xu, Hao Zhang, Umamaheswara Rao Tida, Zhenge Jia, Mengying Zhao

**Published**: 2025-04-29 11:22:06

**PDF URL**: [http://arxiv.org/pdf/2504.20653v1](http://arxiv.org/pdf/2504.20653v1)

## Abstract
Recent advances have demonstrated the promising capabilities of large
language models (LLMs) in generating register-transfer level (RTL) code, such
as Verilog. However, existing LLM-based frameworks still face significant
challenges in accurately handling the complexity of real-world RTL designs,
particularly those that are large-scale and involve multi-level module
instantiations. To address this issue, we present ComplexVCoder, an open-source
LLM-driven framework that enhances both the generation quality and efficiency
of complex Verilog code. Specifically, we introduce a two-stage generation
mechanism, which leverages an intermediate representation to enable a more
accurate and structured transition from natural language descriptions to
intricate Verilog designs. In addition, we introduce a rule-based alignment
method and a domain-specific retrieval-augmented generation (RAG) to further
improve the correctness of the synthesized code by incorporating relevant
design knowledge during generation. To evaluate our approach, we construct a
comprehensive dataset comprising 55 complex Verilog designs derived from
real-world implementations. We also release an open-source benchmark suite for
systematically assessing the quality of auto-generated RTL code together with
the ComplexVCoder framework. Experimental results show that ComplexVCoder
outperforms SOTA frameworks such as CodeV and RTLCoder by 14.6% and 22.2%,
respectively, in terms of function correctness on complex Verilog benchmarks.
Furthermore, ComplexVcoder achieves comparable generation performances in terms
of functionality correctness using a lightweight 32B model (Qwen2.5), rivaling
larger-scale models such as GPT-3.5 and DeepSeek-V3.

## Full Text


<!-- PDF content starts -->

ComplexVCoder: An LLM-Driven Framework for
Systematic Generation of Complex Verilog Code
Jian Zuo∗, Junzhe Liu∗, Xianyong Wang∗, Yicheng Liu∗, Navya Goli†, Tong Xu∗, Hao Zhang∗,
Umamaheswara Rao Tida†, Zhenge Jia∗, Mengying Zhao∗
∗School of and Computer Science and Technology, Shandong University, China
†Electrical and Computer Engineering Department, North Dakota State University, USA
Abstract —Recent advances have demonstrated the promising
capabilities of large language models (LLMs) in generating
register-transfer level (RTL) code, such as Verilog. However,
existing LLM-based frameworks still face significant challenges
in accurately handling the complexity of real-world RTL de-
signs, particularly those that are large-scale and involve multi-
level module instantiations. To address this issue, we present
ComplexVCoder, an open-source LLM-driven framework that
enhances both the generation quality and efficiency of complex
Verilog code. Specifically, we introduce a two-stage generation
mechanism, which leverages an intermediate representation to
enable a more accurate and structured transition from natural
language descriptions to intricate Verilog designs. In addition, we
introduce a rule-based alignment method and a domain-specific
retrieval-augmented generation (RAG) to further improve the
correctness of the synthesized code by incorporating relevant
design knowledge during generation. To evaluate our approach,
we construct a comprehensive dataset comprising 55 complex
Verilog designs derived from real-world implementations. We
also release an open-source benchmark suite for systematically
assessing the quality of auto-generated RTL code together with
the ComplexVCoder framework. Experimental results show that
ComplexVCoder outperforms SOTA frameworks such as CodeV
and RTLCoder by 14.6% and 22.2%, respectively, in terms of
function correctness on complex Verilog benchmarks. Further-
more, ComplexVcoder achieves comparable generation perfor-
mances in terms of functionality correctness using a lightweight
32B model (Qwen2.5), rivaling larger-scale models such as GPT-
3.5 and DeepSeek-V3.
I. I NTRODUCTION
Electronic Design Automation (EDA) is a set of software
and services for designing integrated circuits (ICs), enabling
a structured design flow from specification to implementation.
As Moore’s Law continues to slow, there is growing pressure
on EDA tools to further optimize and streamline the design
process. In response, machine learning for EDA (ML4EDA)
has gained significant traction in recent years. By learning
from prior design solutions, ML4EDA aims to reduce the cost
and complexity of traditional design flows through automation
and intelligent decision-making.
In the realm of ML4EDA, large language models (LLMs)
have shown strong potential in accelerating agile hardware
design tasks. Applications include generating design flow
scripts for EDA tool automation [1], [2], producing security
assertions [3], [4], fixing hardware security bugs [5], [6], etc.
Among these, one of the most promising directions is the
automatic synthesis of RTL designs from natural language
instructions [7], [8]. In this context, LLMs are used to directlytranslate functional descriptions written in natural language
into hardware description code such as Verilog or VHDL.
Compared to traditional predictive ML approaches in EDA [9],
these LLM-based generative methods have the advantage of
reducing manual intervention and enabling a more direct and
streamlined design process.
Despite the remarkable progress of large language models
(LLMs) in generating high-level programming languages such
as Python, C++, and Java [10], [11], LLM-assisted RTL code
generation remains a significant challenge and fundamentally
differs from conventional software code synthesis. Hardware
design imposes far more stringent requirements on logical
structure [8], making it difficult for LLMs to effectively learn
and reason about hardware description languages like Verilog.
These challenges are compounded by the data-hungry nature
of LLM training, which typically requires large volumes of
domain-specific code. However, the availability of high-quality
RTL design datasets is considerably more limited than for
high-level programming languages, and the lack of standard-
ized benchmark implementations further restricts the ability
to evaluate and compare RTL code generation models [12],
[13]. These limitations collectively hinder the development of
effective LLM-based RTL generators.
Public datasets and benchmarks such as VerilogEval [14],
RTLLM [13], and MG-Verilog [15] have been introduced to
address the scarcity of training data and enrich the diversity of
available Verilog code. Building on these datasets and bench-
marks, researchers have explored three primary approaches to
improve LLM performance in RTL code generation: prompt
engineering (PE), domain-specific fine-tuning (DSFT), and
instruction tuning (IT). Prompt engineering aims to enhance
code generation quality by modifying the phrasing and struc-
ture of prompts without altering the model’s parameters [13],
[16], [17]. Domain-specific fine-tuning takes a more direct
approach by continuing model training on Verilog datasets,
with strategies ranging from full model fine-tuning [18],
[19] to domain-adaptive pre-training [2]. Instruction tuning
focuses on curating high-quality natural language and code
pair datasets and applying parameter-efficient fine-tuning tech-
niques. Representative works such as CodeV [12], RTL-
Coder [7], BetterV [20] have demonstrated the effectiveness
of this approach through the development of instruction-tuned
models specifically optimized for Verilog generation.
Despite efforts to enable effective LLM-based Verilog codearXiv:2504.20653v1  [cs.SE]  29 Apr 2025

generation, three key challenges impede the adoption of LLM
in practical EDA design flow: (1) Existing frameworks are
typically learned and evaluated on benchmarks composed of
isolated, single-module Verilog designs, which fail to capture
the complexity of real-world applications involving hundreds
of lines of code and multi-level instantiations; (2) Current
generation workflows generally translate natural language de-
scriptions directly into Verilog code, reducing the overall in-
terpretability, adjustability, and transparency of the generation
process; (3) Extensive state-of-the-art frameworks rely heavily
on powerful but proprietary LLMs like GPT-4, while open-
source models tailored for Verilog still lag significantly in
performance.
To address these challenges, we propose ComplexVCoder ,
an LLM-driven framework designed to enable accurate and
scalable generation of complex, real-world Verilog designs.
Specifically, we propose a two-stage generation mechanism
that bridges the gap between simple natural language inputs
and complex, real-world hardware code through a novel in-
termediate representation named General Intermediate Repre-
sentation (GIR). Unlike abstract syntax tree (AST)-like for-
mats, GIR distills designs into fundamental Verilog variables
annotated with concise one-sentence comments on module
functionality and streamlined logic structures. This design not
only facilitates a smoother transition from language to code
but also enhances interpretability and reduces LLM training
difficulty. In the first stage of generation (i.e., translating
natural language descriptions to GIR), we propose an au-
tomated augmentation pipeline that generates high-quality,
diverse description-GIR training pairs, and further enhance
the LLM capability through a GIR-specific instruction tuning
method to strengthen the model’s understanding of RTL design
structures. In the second stage of generation (i.e., translating
GIR to complex, real-world Verilog code), we introduce a rule-
based alignment method converting structured intermediate
designs into implementation details in natural language de-
scription as the generation guidance prompt to the LLM. Ad-
ditionally, we integrate a domain-specific retrieval-augmented
generation (RAG) module, leveraging the GIR’s embedded
comments to retrieve relevant single-module Verilog code im-
plementations for the generation prompt to further improve the
functional correctness. Experimental results demonstrate that
ComplexVCoder outperforms existing state-of-the-art frame-
works, including CodeV and RTLCoder, by 14.6% and 22.2%,
respectively, in functional correctness on complex Verilog
benchmarks. Remarkably, it achieves comparable performance
in terms of functional correctness when compared with full-
scale proprietary models such as Deepseek-V3 and GPT-4,
despite using a significantly smaller 32B open-source model
(QWen). Our contributions are summarized as follows:
•We propose ComplexVCoder, a large language model
(LLM)-driven framework for complex, real-world Verilog
design generation.
•To bridge the gap between simple natural language
descriptions and intricate hardware code, we introducea two-stage generation mechanism based on a novel
General Intermediate Representation (GIR), significantly
reducing generation complexity and enhancing the trans-
parency of the synthesis process.
•We further improve generation quality through an au-
tomated instruction-pair data augmentation pipeline and
a rule-based alignment mechanism that ensures faithful
complex Verilog design generation.
•Experimental results show that ComplexVCoder outper-
forms SOTA Verilog generation frameworks across vari-
ous benchmarks, achieving higher functional correctness
rates while requiring fewer training resources.
•ComplexVCoder framework is open-sourced, including
the data generation pipeline, training datasets, test-
benches, and fine-tuned model weights, to facilitate future
research and reproducibility.
II. B CKGROUND AND MOTIVATION
Large Language Models (LLMs) have demonstrated impres-
sive capabilities in both natural and programming language
tasks [11], [21]. Their success in code generation has sparked
growing interest in applying LLMs to the field of machine
learning for electronic design automation (ML4EDA). Among
the various explorations, one of the most promising directions
is the automatic generation of Register-Transfer Level (RTL)
designs. Hardware design, which is typically described using
hardware description languages (HDLs) such as Verilog or
VHDL, plays a crucial role in the EDA workflow and signif-
icantly impacts downstream design stages. However, writing
high-quality HDL code is a complex and time-consuming task
that requires substantial domain expertise, making it a costly
process, especially for modern, large-scale integrated circuits
(ICs). To address this challenge, there is an increasing demand
for leveraging LLMs to automatically generate HDL code from
natural language descriptions of design functionality. This
LLM-driven approach has the potential to transform traditional
IC design processes by reducing the burden of manual coding
and debugging.
Current research efforts toward effective LLM-based RTL
generation generally focus on three key areas: high-quality
dataset preparation, post-training techniques (such as fine-
tuning and instruction tuning), and knowledge augmentation
strategies aimed at enhancing the model’s understanding and
generation capabilities.
A. RTL Benchmarks
Several efforts have been made to construct high-quality
RTL design datasets for fine-tuning large language models
(LLMs), along with corresponding testbenches to evaluate
functional and syntactic correctness. For instance, the au-
thors of [18] collect Verilog code from open-source GitHub
projects to create a large-scale dataset for LLM training.
VerilogEval [14], released by NVIDIA Research, offers a com-
prehensive evaluation suite comprising 156 problems sourced
from HDLBits [22], accompanied by a benchmarking frame-
work for automated functional verification. RTLLM [13] also

Fig. 1: An overview of ComplexVCoder framework: (a) First stage generation from natural language description to RTL
representation (i.e., General Intermediate Representation), including the instruction pair preparation and GIR-specific instruction
tuning approaches to enable the LLM to understand the natural language description and translate it into a structural
representation; (b) Second stage generation from GIR to complex Verilog design, including retrain-free RAG and rule-based
alignment approaches to enrich the input prompt to improve the LLM capability to generate complex, real-world Verilog design.
contributes a benchmark of 29 RTL designs paired with natural
language descriptions and their respective testbenches.
Despite these advances, most existing datasets and bench-
marks focus on evaluating LLM performance on isolated,
standalone RTL modules. This narrow scope fails to reflect the
complexity and scale of real-world hardware design scenarios,
which often involve extensive Verilog codebases with multi-
level instantiations and interdependent modules. Furthermore,
the simplicity of the RTL examples and the verbose and
unwieldy natural language descriptions in the existing bench-
marks limit the practical applicability of these frameworks.
These limitations underscore the need for more comprehensive
and realistic benchmarks that assess LLMs in the context of
large-scale, complex Verilog design tasks, aligning better with
real-world development demands.
B. LLM-based RTL Code Generation Framework
Prior research in RTL code generation has focused on
improving generation quality through three main approaches:
(1) Prompt Engineering (PE), enhances the generated RTL
code quality by modifying the structure and content of prompts
without altering the underlying model parameters, making it an
efficient and low-cost solution. Notable efforts include the use
of prompt templates for more effective hardware design [16],
the development of self-planning prompt engineering strate-
gies [13], and the integration of compiler error messages into
prompts to enable multi-round synthesis feedback [17]; (2)
Domain-Specific Fine-Tuning (DSFT) involves further training
pre-trained LLMs on Verilog datasets to better adapt them
to hardware design tasks. For example, the authors of [18]
perform full-parameter fine-tuning using a dataset collected
from GitHub. NVIDIA’s ChipNeMo [2] employs domain-
adaptive pre-training using proprietary data; (3) Instruction
Tuning (IT), widely adopted for its effectiveness and lower
resource requirements, constructs high-quality pairs of design
descriptions and code, followed by parameter-efficient fine-
tuning. CodeV [12] builds such a dataset by prompting GPT-
3.5 to generate hierarchical descriptions from Verilog code.
RTLCoder [7] creates an automated pipeline for large-scale
dataset generation and releases open-source instruction-tuned
Verilog generation models. AutoVCoder [8] introduces a hard-ware database construction strategy to enhance data diversity
and quality, and implements a two-stage fine-tuning process
to further boost model capability.
Despite these advances, existing frameworks fall short when
it comes to generating real-world, complex RTL designs,
which often span hundreds of lines of code and involve multi-
level module instantiations. Moreover, when presented with
simple natural language descriptions of functional require-
ments, the existing framework (even with the full-parameter
LLMs such as GPT-4) struggles to generate correct and
complete RTL code. This highlights the pressing need for a
more capable LLM-based framework that can reliably gen-
erate functionally correct RTL designs from concise natural
language inputs.
C. Retrieval-Augmented Generation in LLM for EDA
To overcome the limitations of LLMs in domain-specific
tasks, retrieval-augmented generation (RAG) has emerged as
a widely adopted technique in LLM-based generation frame-
works. RAG enhances the accuracy and domain knowledge of
LLMs by incorporating external knowledge sources into the
generation process. Instead of relying solely on the internal
parameters of the model, RAG enables the LLM to look up
relevant information from a designated knowledge base before
generating outputs. The key step of the RAG is the retrieval
step, where a retriever module queries a database to fetch
domain-specific documents or code snippets. The retrieved
contents are then integrated into the model’s prompt during
generation. For instance, ChipNeMo [2] incorporates RAG
by retrieving relevant textual passages to enrich prompts for
Verilog code generation, supplemented with further parameter
fine-tuning to improve output quality. AutoVCoder [17] en-
hances the retrieval process by fine-tuning the retriever itself
using contrastive learning to boost relevance and diversity in
the retrieved content.
However, current RAG approaches often place significant
emphasis on optimizing the retriever, which introduces addi-
tional complexity through separate data preparation and model
fine-tuning. This underscores the need for a more lightweight
yet effective RAG strategy, particularly in simplifying retriever
construction without compromising performance.

III. C OMPLEX VERILOG GENERATION FRAMEWORK
In this section, we begin by providing an overview of the
ComplexVCoder framework. We then describe the design and
role of the General Intermediate Representation (GIR), which
serves as a bridge between natural language descriptions and
complex Verilog code. Following this, we detail the proposed
two-stage generation process bridged by GIR to improve
generation quality. This includes our instruction augmentation-
based fine-tuning strategy, a rule-based alignment method for
accurate translation, and a retraining-free RAG mechanism to
enhance functional correctness.
A. Overview
The framework overview of ComplexVCoder is shown in
Fig. 1. To enable the generation of complex Verilog code
from a simple natural language description, ComplexVCoder
adopts a two-stage synthesis approach, with the process
bridged by a proposed RTL representation named General
Intermediate Representation (GIR). The proposed two-stage
generation leverages RTL representation to break down the
generation task, making it more manageable and interpretable.
The automated generation flow is illustrated in Fig. 1. In the
second stage, GIR is translated into complex Verilog code
using a rule-based alignment method, along with a retraining-
free RAG mechanism, both of which contribute to improving
the final generation quality.
B. General Intermediate Representation
The General Intermediate Representation (GIR) introduced
in this study serves as a critical abstraction layer designed to
automate hardware design by bridging the gap between natural
language descriptions and the complex, real-world Verilog
code. In the existing Verilog generation frameworks, LLMs
are generally expected to infer hardware constructs (i.e., ports,
instances, and signal connections) from unstructured text [7],
[8] or structured but verbose instructions [12], [15]. It often
leads to syntactic errors or functional inaccuracies due to
a lack of explicit structural guidance. This problem could
be further aggravated when it comes to complex, real-world
Verilog design generation.
To address the gap, the GIR explicitly defines these con-
structs within its schema, allowing LLMs to map natural
language inputs to well-organized, predefined fields. This
structured mapping mitigates the risk of generating “halluci-
nated” content—functionally incorrect code that may appear
syntactically valid but deviates from hardware design prin-
ciples, such as improper use of software-style loops. Tailored
specifically for LLMs, the GIR transforms unstructured, rough
descriptions of the demanded function into a standardized,
semantically rich format that is both human-readable and
machine-processable.
As illustrated in Fig. 2, the GIR is structured in a hi-
erarchical, JSON-like format that encapsulates the essential
components of a hardware module, including fields such as
"modules" ,"parameters" ,"ports" ,"instances" ,
and"connections" . Each field is carefully defined to
Fig. 2: General intermediate representation template.
capture the architectural and functional semantics of hardware
designs. Specifically, the GIR begins with the "modules"
field, which serves as the root container for all module
definitions. Within each module, the "Top_module_name"
field identifies the main module, while the accompanying
"function" field provides a concise, two-sentence sum-
mary that describes the module’s intended behavior. This
structured design not only simplifies the representation of
hardware components but also enhances the interpretability
and modularity of the entire system.
The conciseness of GIR is illustrated in Fig. 3, which com-
pares its parsing of the adder_16bit module with the RTL
representations generated via Yosys [23] and Pyverilog [24].
The RTL representation by Yosys produces an extensive 945-
line output focused on gate-level details. This low-level output
flattens design hierarchies and prioritizes hardware implemen-
tation over interpretability. Its output is difficult for LLMs to
utilize directly, as it lacks semantic information and is heavily
dependent on strict Verilog compliance, making it unsuitable
for incomplete or iterative design workflows. On the other
hand, the RTL representation in the abstract syntax tree (AST)
format extracted by Pyverilog offers a more structured view
than Yosys, only with a 259-line output that maintains lexical
elements such as module and signal definitions. However,
it remains closely tied to Verilog syntax, offering minimal
abstraction and limited insight into the functional intent of
the design. Extracting meaningful semantics from Pyverilog’s
AST requires significant post-processing, which can be both
error-prone and computationally expensive, particularly for
large or complex designs.
In contrast, as shown in Fig. 3, GIR provides a concise, 60-
line representation that captures both the hierarchical structure
and functional intent of the adder_16bit module. It ab-
stracts away low-level syntax in favor of modular, semantically
meaningful fields that are easily populated and understood
by LLMs. This structure enables generation through guided
template completion rather than reverse engineering from
verbose and syntax-bound outputs. Furthermore, the IR sup-
ports incremental refinement, allowing developers to gradually
complete the design in a flexible and interpretable manner.

Fig. 3: Different RTL representation of the Verilog design
adder_16bit by GIR, Yosys and Pyverilog.
Its JSON-like format aligns naturally with LLM tokenization
strategies, facilitating seamless integration without the adap-
tation overhead required by traditional tools like Yosys or
Pyverilog.
C. First Stage Generation
In the first stage of generation shown in Fig. 1, our goal
is to enable the LLM to interpret a simple natural language
description of a desired hardware function and translate it into
a structured General Intermediate Representation (GIR). To
achieve this, we construct a customized dataset consisting of
natural language–GIR pairs derived from open-source Verilog
designs. We then apply GIR-specific instruction tuning to
teach the LLM domain-specific knowledge for accurately
performing this translation.
Instruction Pair Preparation . To prepare the instruction
pairs, we collect a large corpus of open-source Verilog and
SystemVerilog code from GitHub, ensuring all selected repos-
itories permit modification and redistribution. Our primary
source is the PyraNet dataset [25], from which we extract over
18,000 high-quality Verilog designs. We remove duplicate or
auto-generated files, as well as those with restrictive licenses.
Verilog modules and functions are extracted using regular
expressions, and files are further filtered based on token
length to avoid exceeding model input limits during training.
This step ensures that training samples remain complete and
informative, minimizing the risk of performance degradation
due to truncated inputs.
To ensure code validity, we use Iverilog [26] and Yosys [23]
to discard samples with syntax or parsing errors. We further
perform two key preprocessing steps. First, we use Yosys
to validate the Verilog syntax and leverage the commercial
LLMs to generate RTL representations that conform to our
GIR template. Second, we again use the commercial LLMs to
Fig. 4: As a case study to illustrate the instruction tuning
process, we consider a simple Verilog module implementing an
adder. This example demonstrates how the model is guided to
understand the natural language description of hardware func-
tionality and generate its corresponding General Intermediate
Representation (GIR) and Verilog code.
generate corresponding natural language descriptions, covering
key module attributes such as name, function, parameters, and
I/O ports. All RTL representations and the descriptions are
manually rectified to avoid hallucination errors.
This pipeline yields a high-quality dataset of 4,000
description-GIR pairs. We use this dataset to teach the LLM
through a GIR-specific instruction tuning phase, equipping it
to recognize and reproduce complex hardware structures in a
structured and semantically meaningful format.
GIR-Specific Instruction Tuning . Based on the prepared
instruction pairs, we proposed a GIR-specific instruction tun-
ing method to inject structured knowledge into LLMs. We
introduce a tailored alignment method that maps natural lan-
guage descriptions of Verilog modules to their corresponding
GIR structures. Based on this mapping, we construct instruc-
tion samples that guide the model to generate detailed GIR rep-
resentations from descriptive inputs. This two-way instruction
tuning framework significantly enhances the LLM’s ability to
comprehend and generate Verilog code. By integrating GIR
as a structured intermediate layer, we mitigate the limitations
imposed by the small size of Verilog training data. The GIR
serves as both a scaffold and a semantic guide, enabling more
accurate and syntactically valid Verilog generation.
The underlying learning mechanism of LLMs follows an
auto-regressive paradigm. Let z=z1, z2, . . . , z Trepresent a
training sequence comprising both an instruction and its re-
sponse. An LLM parameterized by ϕestimates the probability
of the entire sequence by factorizing it via the chain rule of
probability [20]:
pϕ(z) =TY
t=1pϕ(zt|zj<t). (1)
During fine-tuning, particularly with Low-Rank Adaptation
(LoRA), the model is optimized to minimize the expected
negative log-likelihood (or equivalently, cross-entropy loss)

across a dataset D=z(1), . . . ,z(N). The training objective
is formalized as:
J(ϕ) =−1
N1
TNX
n=1TX
t=1logpϕ(zi
t|zi
j<t). (2)
D. Second Stage Generation
In the second stage of generation, illustrated in Fig. 1, the
objective is to enable the second LLM to generate complex,
real-world Verilog designs from GIR. Rather than relying on
computationally expensive and time-consuming retraining, we
introduce a hybrid approach that combines rule-based align-
ment and a retraining-free Retrieval-Augmented Generation
(RAG) mechanism to enhance prompt content to improve code
generation performance.
Rule-Based Alignment . To establish a semantic bridge
between GIR and Verilog implementation, we develop a rule-
based natural language conversion framework that translates
structured GIR data into rich, human-readable descriptions. As
shown in Fig. 1, the translated descriptions would be utilized
as part of the input prompt as the detailed guidance to the
LLM’s generation process.
This framework operates across three abstraction layers.
At the top layer, the module definition captures the essential
structure of the hardware design, including module declara-
tions, parameter definitions, port interfaces, and submodule
instantiations. The port description layer encodes individual
port attributes such as bit-width, directionality (input/output),
and clock domain synchronization. At the lowest layer, instan-
tiation logic captures the hierarchical relationships and con-
nectivity among modules by analyzing signal paths and their
corresponding assignments. To automate this conversion, we
define a syntactic rule library that enables precise translation
from GIR to structured natural language descriptions. The core
transformation rules are as follows:
•Module Declaration Rule : Parses
ir_data[’modules’] into “ <module_name>
module defined with <parameter_count>
parameters, <input_count> input ports, and
<output_count> output ports.”
•Port Mapping Rule : Generates for
ir_data[’modules’][ports] the description
“Port <n> is of <direction> type, with bit-width
<width> , synchronized to the <clock_domain>
clock domain.”
•Instantiation Description Rule : Converts
ir_data[’modules’][’instances’]
into “Instantiates <count> submodules,
where <submodule_name> implements
<function_description> , connected to the
parent module via <interface_name> .”
The semantic descriptions generated through multi-stage
translation are fed into an LLM to guide the production of
implementation details covering the following aspects:•Architecture Decomposition : Include a module hierar-
chy matrix, a key submodule function mapping table, and
storage configuration parameters.
•Control Logic : Define a state machine with
<state_count> , timing control based on
<synchronous/asynchronous> reset mechanisms,
and finite state machine encoding for processing steps.
•Data Path : Define a signal flow architecture from in-
put buffers to pipeline registers, computational units,
and output latches, incorporating an <n>-stage pipeline
design with a bit-width conversion strategy named
<strategy_name> .
Retraining-Free RAG . The core rationale behind Retrieval-
Augmented Generation (RAG) is to identify data from an
existing database that is most relevant to the task at hand. This
retrieved information is then appended to the prompt, enhanc-
ing the ability of a Large Language Model (LLM) to pro-
duce code that is both accurate and contextually appropriate.
Prior approaches such as ChipNeMo [2] and AutoVCoder [8]
achieve this by retraining the retriever model to improve the
quality of retrieval. However, preparing the necessary fine-
tuning datasets and executing the retraining process is labor-
intensive and computationally demanding.
To overcome this challenge, we propose a retraining-free
RAG mechanism that leverages the structured information
embedded in the GIR generated during the first stage. As
illustrated in Fig. 1, we first construct a high-quality, single-
module retrieval codebase. This database is built from a
large-scale collection of over 12,500 open-source Verilog
and SystemVerilog projects sourced from platforms such as
GitHub and HDLBits. To ensure clarity and consistency,
we purify the code using regular expressions that remove
single-line comments, multi-line annotations, and conditional
compilation directives. Independent modules with well-defined
input and output interfaces are isolated, and semantic function
descriptions are recorded for each module. These descriptions
are transformed into semantic vectors using a pre-trained
embedding model and indexed alongside the corresponding
code fingerprints in the format <code_fingerprint,
semantic_vector> .
During retrieval, the functional description derived from the
GIR is embedded using the same pre-trained model. A coarse-
grained retrieval phase identifies the top three candidates by
computing cosine similarity between the query and database
vectors. The similarity is defined as:
sim(A, B) =A·B
∥A∥ · ∥B∥, (3)
where AandBare vectors and ∥ · ∥ denotes the vector
norm. To refine these candidates, a fine-grained re-ranking
stage employs a Cross-Encoder that evaluates each query-
document pair using the input format [CLS] Query [SEP]
Document [SEP] . This model captures direct interactions
between the query and the retrieved code snippets, allowing for
more accurate relevance estimation. The proposed RAG mech-
anism effectively identifies the most contextually appropriate

TABLE I: Metadata of datasets for Verilog generation.
Benchmark Dataset Size Avg. Lines Avg. Hierarchies
VerilogEval-Human [14] 156 22 1
Problem-Set-VeriGen [27] 17 23 1
RTLLM [13] 50 73 1
ComplexVDB 55 151 3
code for each module. This significantly enhances the LLM’s
ability to generate high-quality Verilog code while avoiding
the overhead associated with retriever retraining.
IV. E XPERIMENTS
A. Experimental Setup
Dataset Preparation: We introduce an open-source dataset
named ComplexVDB , which is used to evaluate the capabilities
of LLMs in generating complex, real-world Verilog designs.
ComplexVDB comprises 55 hand-designed Verilog cases,
each representing a realistic circuit design involving intricate
inter-module interactions and deep structural hierarchies. In
addition, each case is equipped with a corresponding hand-
designed test bench and a comprehensive end-to-end pipeline
to determine if the generated Verilog code adheres to the
criteria of functional and syntactic correctness. Table I summa-
rizes the metadata of ComplexVDB and existing datasets com-
monly used for evaluating Verilog code generation. Existing
benchmarks such as VerilogEval-Human [14], Problem-Set-
VeriGen [18], and RTLLM [13] primarily focus on relatively
small-scale circuits, typically containing only a single module
with short code lengths (averaging 22–73 lines). These datasets
are well-suited for testing the syntactic correctness and basic
semantics of generated code. However, they are insufficient for
evaluating structural hierarchy, multi-module dependencies, or
interface consistency across complex designs. In contrast, each
case in ComplexVDB contains an average of 151 lines of
code and exhibits a hierarchy depth of three levels. These
levels include top-level system modules, intermediate control
or datapath layers, and leaf-level functional blocks. This
structure enables a more comprehensive evaluation of code
generation capabilities, particularly in terms of modularity and
scalability. To further assess the generalization ability of the
ComplexVCoder framework, we also include Problem-Set-
VeriGen [27] as a comparative benchmark.
Baseline: We evaluate the generation performances of Com-
plexVCoder against the following two categories of base-
lines: (1) Commercial large-scale language models: we invoke
general-purpose LLMs including GPT-4o ,Deepseek-V3 , and
GPT-3.5 to conduct complex Verilog generation based on
the simple natural language description; (2) Domain-specific
Verilog code generation frameworks: we invoke SOTA Verilog
generation frameworks including codeV [12], RTLCoder [7],
ChipGPT [16], VeriGen [27], and MG-Verilog [15]. We adopt
the widely used pass@ k[28] metric to evaluate Verilog code
generation, which measures the probability that at least one out
of k generated code samples successfully passes validation.
This metric can be calculated as pass@ k=E
1−(n−c
k)
(n
k)
,TABLE II: Generation performances of ComplexVCoder and
SOTA frameworks over dataset ComplexVDB.
Model #Params pass@1 pass@5
Deepseek-V3 671B 47.27% 63.64%
GPT-3.5 N/A 34.55% 58.18%
GPT-4o N/A 45.45% 69.09%
CodeV [12] 7B 20.00% 29.09%
RTLCoder [7] 6.7B 16.27% 21.43%
ChipGPT [16] 7B 19.64% 26.79%
VeriGen [27] 16B 8.93% 12.50%
MG-Verilog [15] 7B 9.09% 18.18%
ComplexVCoder(Deepseek-V3) 671B 52.73% 65.45%
ComplexVCoder(GPT-3.5) N/A 36.36% 56.36%
ComplexVCoder(GPT-4o) N/A 49.09% 70.90%
ComplexVCoder(Qwen2.5-Coder) 32B 32.72% 54.55%
ComplexVCoder(Qwen2.5-32B) 32B 36.36% 58.18%
ComplexVCoder(Qwen2.5-14B) 14B 27.27% 43.64%
where nis the total number of test attempts for the task, and
cis the number of correct code generations for the task.
Implementation Details: We invoke different LLMs as
the base models in ComplexVCoder, including commercial
LLMs (i.e., Deepseek-V3, GPT-3.5, GPT-4o, GPT-4o-mini),
and open-source LLMs with smaller scale (i.e., Qwen2.5-
Coder, Qwen2.5-32B, and Qwen2.5-14B). We use Icarus Ver-
ilog [26] and Pyverilog [24] to check the syntactic correctness
of the generated Verilog solutions. If the design is syntactically
correct, we proceed by running the corresponding test bench
from the benchmark suite and comparing the output of the
generated solution against the golden reference to verify its
functional correctness. During the instruction tuning process,
we apply LoRA [29] to enhance efficiency while preserving
model performance. The learning rate γis set to 2×10−4,
and the models are trained for 3 epochs. Both training and
inference are conducted using eight NVIDIA 4090 GPUs.
B. Experimental Results
We begin by evaluating the generation performance of
ComplexVCoder on complex, real-world Verilog designs using
pass@1 and pass@5 as metrics, and compare it against SOTA
frameworks. As shown in Table II, commercial large-scale
language models already demonstrate strong performance in
Verilog generation, with Deepseek-V3 achieving up to 47.27%
in pass@1 and GPT-4o reaching 69.09% in pass@5. These
results underscore the capabilities of large pre-trained LLMs,
even in the absence of domain-specific adaptation. The perfor-
mances highlight the generalization power of large pre-trained
LLMs, even without domain-specific adaptation. In contrast,
existing Verilog-specific frameworks, despite being tailored for
Verilog generation tasks, consistently underperform on com-
plex tasks, with pass@1 scores below 20% and pass@5 scores
under 30%. It is because the LLMs in those frameworks are
primarily trained on isolated, single-module Verilog examples,
making them hard to handle the intricacies and hierarchical
structures generally utilized in real-world hardware designs.
ComplexVCoder addresses these limitations by incorporat-
ing the two-stage generation bridged by general intermediate
representations (GIR) This generation mechanism substan-
tially improves both structural accuracy and semantic coher-

TABLE III: Generation performances of ComplexVCoder and SOTA frameworks over Problem-Set-VeriGen [27].
Benchmark NameComplexVcoder(Qwen2.5-14B) ComplexVcoder(Qwen2.5-32B) ChipGPT(Llama-7B) [16] ChipGPT(Llama-13B) [16] VeriGen(CodeGen-16B) [27] CodeV [12] RTLcoder [7] GPT-3.5 GPT-4o-mini GPT-4o
syntax function syntax function syntax function syntax function syntax function syntax function syntax function syntax function syntax function syntax function
basic1 0 100.00% 0 100.00% 0 100.00% 0 100% 0 100.00% 0 100.00% 0 100.00% 0 100.00% 0 100.00% 0 100.00%
basic2 0 100.00% 0 100.00% 0 100.00% 0 100.00% 0 100.00% 0 100.00% 0 100.00% 0 100.00% 0 100.00% 0 100.00%
basic3 0 100.00% 0 100.00% 0 50.00% 0 50.00% 0 100.00% 0 100.00% 0 100.00% 0 100.00% 0 100.00% 0 100.00%
basic4 0 100.00% 0 100.00% 0 100.00% 0 100.00% 0 100.00% 0 100.00% 0 100.00% 0 100.00% 0 100.00% 0 100.00%
intermediate1 0 100.00% 0 100.00% 0 25.00% 0 100.00% 0 0.00% 0 100.00% 0 100.00% 1 100.00% 0 100.00% 0 100.00%
intermediate2 0 100.00% 0 100.00% 0 100.00% 0 100.00% 0 100.00% 0 100.00% 5 0.00% 0 100.00% 0 100.00% 0 100.00%
intermediate3 0 11.70% 0 75.00% 5 0% 0 11.70% 0 6.25% 0 0.00% 5 0.00% 4 6.25% 0 6.25% 0 37.50%
intermediate4 0 100.00% 0 100% 0 100.00% 0 75% 0 0% 0 0.00% 0 75.00% 0 0.00% 0 100.00% 0 100.00%
intermediate5 5 0% 5 0.00% 0 30% 0 60% 0 0% 0 100.00% 0 75.00% 0 100% 0 30% 0 100.00%
intermediate6 0 100.00% 0 100.00% 0 100.00% 0 100.00% 0 0% 5 0% 5 0% 0 0.00% 0 0.00% 0 0%
intermediate7 5 0.00% 0 100.00% 0 100.00% 0 100.00% 0 100.00% 0 100.00% 0 100.00% 1 100.00% 0 100.00% 0 100.00%
intermediate8 0 100.00% 0 100.00% 5 0.00% 0 80.00% 5 0.00% 0 25.00% 5 0.00% 0 100.00% 0 100.00% 0 100.00%
advanced1 0 100.00% 0 100.00% 0 75.00% 0 51.50% 5 0.00% 0 100.00% 0 100.00% 1 100.00% 0 100.00% 0 100.00%
advanced2 0 100.00% 0 100.00% 0 100.00% 5 100.00% 0 92.80% 0 100.00% 0 80.00% 0 92.80% 0 92.80% 0 100.00%
advanced3 0 100.00% 0 100.00% 0 100.00% 5 100.00% 0 100.00% 0 100.00% 0 100.00% 1 83.00% 0 100.00% 0 100.00%
advanced4 1 62.50% 0 75% 0 100.00% 5 100.00% 0 100.00% 0 13% 0 100% 0 62.50% 0 100.00% 0 100.00%
advanced5 0 100.00% 0 100.00% 0 100.00% 0 100.00% 5 0% 0 100.00% 5 0.00% 0 100.00% 0 100.00% 0 100.00%
pass@5 76.47% 82.35% 64.70% 70.59% 58.80% 70.60% 70.60% 64.70% 76.47% 88.24%
ence in the output. As a result, ComplexVCoder significantly
outperforms all existing Verilog generation frameworks. As
shown in Table II, ComplexVCoder(Qwen2.5-14B) outper-
forms CodeV by 17.27% and 14.55% in terms of pass@1 and
pass@5 utilizing the LLM of a similar scale. The performance
advancements are even larger when compared with other
SOTA frameworks such as VeriGen and RTLCoder. These re-
sults demonstrate ComplexVCoder’s ability to guide LLMs in
understanding and generating complex circuit logic effectively.
When built upon Deepseek-V3, ComplexVCoder improves
pass@1 and pass@5 scores by 5.5% and 1.8%, respectively,
over the baseline. Similar gains are observed with GPT-3.5
and GPT-4o backbones, indicating the robustness and general
applicability of our framework. Notably, ComplexVCoder also
demonstrates strong performance with smaller-scale models.
For example, ComplexVCoder using Qwen2.5-32B achieves
equivalent pass@5 and even surpasses GPT-3.5 in pass@1,
despite having fewer parameters. This highlights the potential
of ComplexVCoder to empower lightweight LLMs to generate
high-quality Verilog code efficiently, making it a scalable and
practical solution for real-world hardware design automation.
To further assess the generalization capability of Com-
plexVCoder, we evaluate its performance on Problem-Set-
VeriGen [27]. As shown in Table III, in addition to reporting
the pass rate, we also present the syntax correctness and
functional correctness achieved by each method across the test
cases. ComplexVCoder(Qwen2.5-32B) achieves an impressive
pass@5 of 82.35%, the second-highest among all models
evaluated, trailing only behind the baseline GPT-4o approach.
Notably, ComplexVCoder(Qwen2.5-14B) outperforms leading
Verilog generation frameworks, including ChipGPT (Llama-
13B) and VeriGen (CodeGen-16B), by 5.9% and 17.7% in
pass@5, respectively, despite using a comparable model scale.
These results demonstrate the strength of ComplexVCoder
in enabling LLMs to more effectively capture the intrinsic
characteristics of Verilog design, ultimately leading to higher-
quality and more functionally accurate code generation.
C. Ablation Study
Table IV and Table V present the results of an ablation study
evaluating the effectiveness of key techniques integrated into
ComplexVCoder. These techniques include GIR-specific in-
struction tuning (denoted as IT), rule-based alignment (RBA),TABLE IV: Generation performances under different scheme
settings on ComplexVDB.
MethodQwen2.5-32B GPT-3.5 GPT-4o-mini
pass@1 pass@5 pass@1 pass@5 pass@1 pass@5
Vanilla 25.45% 50.91% 34.55% 54.54% 36.36% 58.18%
TSG 36.36% 58.18% 36.36% 56.36% 38.18% 63.64%
TSG w/o IT 32.72% 54.55% N/A N/A N/A N/A
TSG w/o RBA 29.63% 50.91% 34.55% 54.54% 38.18% 60%
TSG w/o RAG 27.27% 49.08% 32.70% 52.72% 36.36% 58.18%
TABLE V: Generation performances under different scheme
settings on Problem-Set-VeriGen [27].
MethodQwen2.5-32B GPT-3.5 GPT-4o-mini
pass@1 pass@5 pass@1 pass@5 pass@1 pass@5
Vanilla 70.59% 76.47% 58.82% 64.71% 76.47% 76.47%
TSG 82.35% 82.35% 82.35% 82.35% 88.20% 94.10%
TSG w/o IT 76.47% 82.35% N/A N/A N/A N/A
TSG w/o RBA 82.35% 82.35% 76.47% 76.47% 76.47% 88.20%
TSG w/o RAG 70.59% 82.35% 64.71% 70.59% 70.59% 82.35%
and the retraining-free retrieval-augmented generation (RAG).
The study is conducted across two benchmark datasets: Com-
plexVDB and Problem-Set-VeriGen [27]. For comparison,
we also include a vanilla baseline, where the LLM directly
generates Verilog code without any enhancements.
As shown in both tables, the full two-stage generation
pipeline (TSG) consistently delivers the best performance
across all base LLMs, indicating the effectiveness of the pro-
posed techniques. Notably, the performance gap between TSG
and TSG w/o IT on both datasets emphasizes the critical role
of structural guidance introduced through instruction tuning. In
addition, removing the rule-based alignment (TSG w/o RBA)
results in significant performance drops. Specifically, a 6.7%
decrease in pass@1 and a 7.3% decrease in pass@5 are caused
by TSG w/o RBA on ComplexVDB. This highlights the
importance of enriching prompts with structured, semantically
aligned content. Similarly, removing RAG (TSG w/o RAG)
leads to moderate but consistent declines in performance,
underscoring the benefit of including high-quality, standard
Verilog module references in pormpt during generation.
V. C ONCLUSION
In this paper, we propose an LLM-driven framework, Com-
plexVCoder, for systematic generation of complex, real-world

Verilog code. ComplexVCoder aims to generate high-quality
complex Verilog designs via the proposed two-stage genera-
tion mechanism, leveraging an intermediate representation to
enable a more accurate and structured transition from natural
language descriptions to intricate Verilog code. Experimental
results show that ComplexVCoder outperforms SOTA Verilog
generation frameworks in terms of functional and syntacti-
cal correctness. ComplexVCoder also enables the small-scale
LLM (i.e., Qwen-32B) to achieve comparable performances
against commercial LLMs such as GPT-3.5.
REFERENCES
[1] H. Wu, Z. He, X. Zhang, X. Yao, S. Zheng, H. Zheng, and B. Yu,
“Chateda: A large language model powered autonomous agent for eda,”
IEEE Transactions on Computer-Aided Design of Integrated Circuits
and Systems , 2024.
[2] M. Liu, T.-D. Ene, R. Kirby, C. Cheng, N. Pinckney, R. Liang, J. Alben,
H. Anand, S. Banerjee, I. Bayraktaroglu et al. , “Chipnemo: Domain-
adapted llms for chip design,” arXiv preprint arXiv:2311.00176 , 2023.
[3] R. Kande, H. Pearce, B. Tan, B. Dolan-Gavitt, S. Thakur, R. Karri, and
J. Rajendran, “Llm-assisted generation of hardware assertions,” arXiv
e-prints , pp. arXiv–2306, 2023.
[4] W. Fang, M. Li, M. Li, Z. Yan, S. Liu, Z. Xie, and H. Zhang, “Assertllm:
Generating and evaluating hardware verification assertions from design
specifications via multi-llms,” arXiv preprint arXiv:2402.00386 , 2024.
[5] B. Ahmad, S. Thakur, B. Tan, R. Karri, and H. Pearce, “On hardware
security bug code fixes by prompting large language models,” IEEE
Transactions on Information Forensics and Security , 2024.
[6] G. Kokolakis, A. Moschos, and A. D. Keromytis, “Harnessing the power
of general-purpose llms in hardware trojan design,” in International
Conference on Applied Cryptography and Network Security . Springer,
2024, pp. 176–194.
[7] S. Liu, W. Fang, Y . Lu, J. Wang, Q. Zhang, H. Zhang, and Z. Xie, “Rtl-
coder: Fully open-source and efficient llm-assisted rtl code generation
technique,” IEEE Transactions on Computer-Aided Design of Integrated
Circuits and Systems , 2024.
[8] M. Gao, J. Zhao, Z. Lin, W. Ding, X. Hou, Y . Feng, C. Li, and M. Guo,
“Autovcoder: A systematic framework for automated verilog code gen-
eration using llms,” in 2024 IEEE 42nd International Conference on
Computer Design (ICCD) . IEEE, 2024, pp. 162–169.
[9] L. Chen, Y . Chen, Z. Chu, W. Fang, T.-Y . Ho, R. Huang, Y . Huang,
S. Khan, M. Li, X. Li et al. , “The dawn of ai-native eda: Op-
portunities and challenges of large circuit models,” arXiv preprint
arXiv:2403.07257 , 2024.
[10] F. Liu, Y . Liu, L. Shi, H. Huang, R. Wang, Z. Yang, L. Zhang, Z. Li,
and Y . Ma, “Exploring and evaluating hallucinations in llm-powered
code generation,” arXiv preprint arXiv:2404.00971 , 2024.
[11] Z. Yang, F. Liu, Z. Yu, J. W. Keung, J. Li, S. Liu, Y . Hong, X. Ma,
Z. Jin, and G. Li, “Exploring and unleashing the power of large language
models in automated code translation,” Proceedings of the ACM on
Software Engineering , vol. 1, no. FSE, pp. 1585–1608, 2024.
[12] Y . Zhao, D. Huang, C. Li, P. Jin, Z. Nan, T. Ma, L. Qi, Y . Pan, Z. Zhang,
R. Zhang et al. , “Codev: Empowering llms for verilog generation
through multi-level summarization,” arXiv preprint arXiv:2407.10424 ,
2024.
[13] Y . Lu, S. Liu, Q. Zhang, and Z. Xie, “Rtllm: An open-source benchmark
for design rtl generation with large language model,” in 2024 29th Asia
and South Pacific Design Automation Conference (ASP-DAC) . IEEE,
2024, pp. 722–727.
[14] M. Liu, N. Pinckney, B. Khailany, and H. Ren, “Verilogeval: Evaluating
large language models for verilog code generation,” in 2023 IEEE/ACM
International Conference on Computer Aided Design (ICCAD) . IEEE,
2023, pp. 1–8.
[15] Y . Zhang, Z. Yu, Y . Fu, C. Wan, and Y . C. Lin, “Mg-verilog: Multi-
grained dataset towards enhanced llm-assisted verilog generation,” in
2024 IEEE LLM Aided Design Workshop (LAD) . IEEE, 2024, pp. 1–5.
[16] K. Chang, Y . Wang, H. Ren, M. Wang, S. Liang, Y . Han, H. Li,
and X. Li, “Chipgpt: How far are we from natural language hardware
design,” arXiv preprint arXiv:2305.14019 , 2023.[17] S. Thakur, J. Blocklove, H. Pearce, B. Tan, S. Garg, and R. Karri, “Au-
tochip: Automating hdl generation using llm feedback,” arXiv preprint
arXiv:2311.04887 , 2023.
[18] S. Thakur, B. Ahmad, Z. Fan, H. Pearce, B. Tan, R. Karri, B. Dolan-
Gavitt, and S. Garg, “Benchmarking large language models for auto-
mated verilog rtl code generation,” in 2023 Design, Automation & Test
in Europe Conference & Exhibition (DATE) . IEEE, 2023, pp. 1–6.
[19] E. Dehaerne, B. Dey, S. Halder, and S. De Gendt, “A deep learning
framework for verilog autocompletion towards design and verification
automation,” arXiv preprint arXiv:2304.13840 , 2023.
[20] Z. Pei, H. Zhen, M. Yuan, Y . Huang, and B. Yu, “Betterv: Controlled
verilog generation with discriminative guidance,” in International Con-
ference on Machine Learning . PMLR, 2024, pp. 40 145–40 153.
[21] L. Qin, Q. Chen, X. Feng, Y . Wu, Y . Zhang, Y . Li, M. Li, W. Che, and
P. S. Yu, “Large language models meet nlp: A survey,” arXiv preprint
arXiv:2405.12819 , 2024.
[22] H. Wong, “HDLBits — Verilog Practice,” 2024. [Online]. Available:
https://hdlbits.01xz.net/wiki/Main Page
[23] C. Wolf, J. Glaser, and J. Kepler, “Yosys: A free verilog synthesis
suite,” in Proceedings of the 21st Austrian Workshop on Microelectronics
(Austrochip) , vol. 97, 2013.
[24] S. Takamaeda-Yamazaki, “Pyverilog: A python-based hardware design
processing toolkit for verilog hdl,” in Applied Reconfigurable Comput-
ing: 11th International Symposium, ARC 2015, Bochum, Germany, April
13-17, 2015, Proceedings 11 . Springer, 2015, pp. 451–460.
[25] B. Nadimi, G. O. Boutaib, and H. Zheng, “Pyranet: A multi-layered
hierarchical dataset for verilog,” arXiv preprint arXiv:2412.06947 , 2024.
[26] S. Williams and M. Baxter, “Icarus verilog: open-source verilog more
than a year later,” Linux Journal , vol. 2002, no. 99, p. 3, 2002.
[27] S. Thakur, B. Ahmad, H. Pearce, B. Tan, B. Dolan-Gavitt, R. Karri, and
S. Garg, “Verigen: A large language model for verilog code generation,”
ACM Transactions on Design Automation of Electronic Systems , vol. 29,
no. 3, pp. 1–31, 2024.
[28] M. Chen, J. Tworek, H. Jun, Q. Yuan, H. P. D. O. Pinto, J. Kaplan,
H. Edwards, Y . Burda, N. Joseph, G. Brockman et al. , “Evaluating large
language models trained on code,” arXiv preprint arXiv:2107.03374 ,
2021.
[29] E. J. Hu, Y . Shen, P. Wallis, Z. Allen-Zhu, Y . Li, S. Wang, L. Wang,
W. Chen et al. , “Lora: Low-rank adaptation of large language models.”
ICLR , vol. 1, no. 2, p. 3, 2022.