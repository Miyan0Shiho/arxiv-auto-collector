# TPU-Gen: LLM-Driven Custom Tensor Processing Unit Generator

**Authors**: Deepak Vungarala, Mohammed E. Elbtity, Sumiya Syed, Sakila Alam, Kartik Pandit, Arnob Ghosh, Ramtin Zand, Shaahin Angizi

**Published**: 2025-03-07 21:41:42

**PDF URL**: [http://arxiv.org/pdf/2503.05951v1](http://arxiv.org/pdf/2503.05951v1)

## Abstract
The increasing complexity and scale of Deep Neural Networks (DNNs)
necessitate specialized tensor accelerators, such as Tensor Processing Units
(TPUs), to meet various computational and energy efficiency requirements.
Nevertheless, designing optimal TPU remains challenging due to the high domain
expertise level, considerable manual design time, and lack of high-quality,
domain-specific datasets. This paper introduces TPU-Gen, the first Large
Language Model (LLM) based framework designed to automate the exact and
approximate TPU generation process, focusing on systolic array architectures.
TPU-Gen is supported with a meticulously curated, comprehensive, and
open-source dataset that covers a wide range of spatial array designs and
approximate multiply-and-accumulate units, enabling design reuse, adaptation,
and customization for different DNN workloads. The proposed framework leverages
Retrieval-Augmented Generation (RAG) as an effective solution for a data-scare
hardware domain in building LLMs, addressing the most intriguing issue,
hallucinations. TPU-Gen transforms high-level architectural specifications into
optimized low-level implementations through an effective hardware generation
pipeline. Our extensive experimental evaluations demonstrate superior
performance, power, and area efficiency, with an average reduction in area and
power of 92\% and 96\% from the manual optimization reference values. These
results set new standards for driving advancements in next-generation design
automation tools powered by LLMs.

## Full Text


<!-- PDF content starts -->

TPU-Gen: LLM-Driven Custom Tensor Processing Unit
Generator
Deepak Vungarala†, Mohammed E. Elbtity‡, Sumiya Syed†, Sakila Alam†, Kartik Pandit†, Arnob Ghosh†,
Ramtin Zand‡, Shaahin Angizi†
†New Jersey Institute of Technology, Newark, NJ, USA,‡University of South Carolina, Columbia, SC, USA
E-mails: {dv336,shaahin.angizi }@njit.edu
Abstract —The increasing complexity and scale of Deep Neural
Networks (DNNs) necessitate specialized tensor accelerators, such
as Tensor Processing Units (TPUs), to meet various computational
and energy efficiency requirements. Nevertheless, designing opti-
mal TPU remains challenging due to the high domain expertise
level, considerable manual design time, and lack of high-quality,
domain-specific datasets. This paper introduces TPU-Gen, the
first Large Language Model (LLM) based framework designed
to automate the exact and approximate TPU generation process,
focusing on systolic array architectures. TPU-Gen is supported
with a meticulously curated, comprehensive, and open-source
dataset that covers a wide range of spatial array designs and ap-
proximate multiply-and-accumulate units, enabling design reuse,
adaptation, and customization for different DNN workloads. The
proposed framework leverages Retrieval-Augmented Generation
(RAG) as an effective solution for a data-scare hardware domain
in building LLMs, addressing the most intriguing issue, hallucina-
tions. TPU-Gen transforms high-level architectural specifications
into optimized low-level implementations through an effective
hardware generation pipeline. Our extensive experimental eval-
uations demonstrate superior performance, power, and area
efficiency, with an average reduction in area and power of 92%
and 96% from the manual optimization reference values. These
results set new standards for driving advancements in next-
generation design automation tools powered by LLMs.
I. I NTRODUCTION
The rising computational demands of Deep Neural Net-
works (DNNs) have driven the adoption of specialized ten-
sor processing accelerators, such as Tensor Processing Units
(TPUs). These accelerators, characterized by low global data
transfer, high clock frequencies, and deeply pipelined Pro-
cessing Elements (PEs), excel in accelerating training and
inference tasks by optimizing matrix multiplication [1]. De-
spite their effectiveness, the complexity and expertise required
for their design remain significant barriers. Static accelerator
design tools, such as Gemmini [2] and DNNWeaver [3],
address some of these challenges by providing templates for
systolic arrays, data flows, and software ecosystems [4], [5].
However, these tools still face limitations, including complex
programming interfaces, high memory usage, and inefficien-
cies in handling diverse computational patterns [6], [7]. These
constraints underscore the need for innovative solutions to
streamline hardware design processes.
Large Language Models (LLMs) have emerged as a promis-
ing solution, offering the ability to generate hardware descrip-
tions from high-level design intents. LLMs can potentially
reduce the expertise and time required for DNN hardware de-
velopment by encapsulating vast domain-specific knowledge.
However, realizing this potential requires overcoming three
critical challenges. First, existing datasets are often limited in
size and detail, hindering the generation of reliable designs[8], [9]. Second, while fine-tuning is essential to minimize
the human intervention, fine-tuning LLMs often results in
hallucinations producing non-sensical or factually incorrect
responses, compromising their applicability [10], [11]. Finally,
an effective pipeline is needed to mitigate these hallucinations
and ensure the generation of consistent, contextually accurate
code [11]. Therefore, the core questions we seek to answer
are the following– Can there be an effective way to rely on
LLM to act as a critical mind and adapt implementations like
Retrieval-Augmented Generation (RAG) to minimize halluci-
nations? Can we leverage domain-specific LLMs with RAG
through an effective pipeline to automate the design process
of TPU to meet various computational and energy efficiency
requirements?
To answer this question, we develop the first-of-its-kind
TPU-Gen as an automated exact and approximate TPU design
generation framework with a comprehensive dataset specifi-
cally tailored for ever-growing DNN topologies. Our contri-
butions in this paper are threefold: (1) Due to the limited
availability of annotated data necessary for efficient fine-
tuning of an open-source LLM, we introduce a meticulously
curated dataset that encompasses various levels of detail and
corresponding hardware descriptions, designed to enhance
LLMs’ learning and generative capabilities in the context of
TPU design; (2) We develop TPU-Gen as a potential solution
to reduce hallucinations leveraging RAG and fine-tuning, to
align best for the LLMs to streamline the approximate TPU
design generation process considering budgetary constraints
(e.g., power, latency, area), ensuring a seamless transition from
high-level specifications to low-level implementations; and (3)
We design extensive experiments to evaluate our approach’s
performance and reliability, demonstrating its superiority over
existing methods. We anticipate that TPU-Gen will provide a
framework that will influence the future trajectory of DNN
hardware acceleration research for generations to come1.
II. BACKGROUND
LLM for Hardware Design. LLMs show promise in gener-
ating Hardware Description Language (HDL) and High-Level
Synthesis (HLS) code. Table I compares notable methods in
this field. VeriGen [10] and ChatEDA [19] refine hardware
design workflows, automating the RTL to GDSII process
with fine-tuned LLMs. ChipGPT [8] and Autochip [13] inte-
grate LLMs to generate and optimize hardware designs, with
1The dataset and fine-tuned models are open-sourced. The link is omitted to
maintain anonymity since the GitHub anonymous link should be under 2GB
which is exceeded in this study.arXiv:2503.05951v1  [cs.AR]  7 Mar 2025

TABLE I
COMPARISON OF THE SELECTED LLM- BASED HDL/HLS GENERATORS .
Property Ours [10] [9] [8] [12] [13] [14] [15] [16] [17] [18]
Function TPU Gen. Verilog Gen. AI Accel. Gen. Verilog Gen. Verilog Gen. Verilog Gen. Hardware Verf. Hardware Verf. Verilog Gen. † AI Accel. Gen.
Chatbot∗✓ ✗ ✗ ✗ ✗ ✗ ✓ ✓ ✗ ✗ ✗
Dataset ✓ ✓(Verilog) ✗ NA NA NA ✗ ✗ ✓ ✓ ✓
Output format Verilog Verilog HLS Verilog Verilog Verilog Verilog HDL Verilog Verilog Chisel
Auto. Verif. ✓ ✗ ✗ ✗ ✗ ✓ ✓ ✗ ✓ ✗ ✓
Human in Loop Low Medium Medium Medium High Low Low Low Low Low Low
Fine tuning ✓ ✓ ✓ ✗ ✗ ✗ ✗ ✗ ✗ ✓ ✗
RAG ✓ ✗ ✗ ✗ ✗ ✗ ✗ ✓ ✗ ✗ ✗
∗A user interface featuring Prompt template generation for the input of LLM.†Not applicable.
Autochip producing precise Verilog code through simulation
feedback. Chip-Chat [12] demonstrates interactive LLMs like
ChatGPT-4 in accelerating design space exploration. MEV-
LLM [20] proposes multi-expert LLM architecture for Verilog
code generation. RTLLM [21] and GPT4AIGChip [9] enhance
design efficiency, showcasing LLMs’ ability to manage com-
plex design tasks and broaden access to AI accelerator design.
To the best of our knowledge, GPT4AIGChip [9] and SA-DS
[18] are a few initial works focus on an extensive framework
specifically aimed at the generation of domain-specific AI
accelerator designs where SA-DS focus on creating a dataset
in HLS and employ fine-tuning free methods such as single-
shot and multi-shot inputs to LLM. Other works for hardware
also include creation of SPICE circuits [22], [23]. However,
theabsence of prompt optimization, tailored datasets, model
fine-tuning, and LLM hallucination pose a barrier to fully
harnessing the potential of LLMs in such frameworks [19],
[18]. This limitation confines their application to standard
LLMs without fine-tuning or In-Context Learning (ICL) [19],
which are among the most promising methods for optimizing
LLMs [24].
Retrieval-Augmented Generation. RAG is a promising
paradigm that combines deep learning with traditional retrieval
techniques to help mitigate hallucinations in LLMs [25]. RAG
leverages external knowledge bases, such as databases, to
retrieve relevant information, facilitating the generation of
more accurate and reliable responses [26], [25]. The primary
challenge in deploying LLMs for hardware generation or any
application lies in their tendency to deviate from the data and
hallucinate, making it challenging to capture the essence of
circuits and architectural components. LLMs tend to prioritize
creativity and finding innovative solutions, which often results
in straying from the data [11]. As previous works show, the
RAG model can be a cost-efficient solution by retrieving and
augmenting data, avoiding heavy computational demands [27].
Approximate MAC Units. Approximate computing has been
widely explored as a means to trade reduced accuracy for
gains in design metrics, including area, power consumption,
and performance [28], [29], [30], [31], [32], [33]. As the
computation core in various PEs in TPUs, several approximate
Multiply-and-Accumulate (MAC) units have been proposed as
alternatives to precise multipliers and adders and extensively
analyzed in accelerating deep learning [34], [35]. These MAC
units are composed of two arithmetic stages—multiplication
and accumulation with previous products—each of which can
be independently approximated. Most approximate multipliers,
such as logarithmic multipliers, are composed of two key com-
ponents: low-precision arithmetic logic and a pre-processing
APEControllerFIFO FIFODEMUXDEMUXWeight/IFMAP MemoryIFMAP/Weight Memory
MUX
Output Memory (OFMAP)
PAU
PAUPAU
APE APE
APE APE APEPAU
PAUFig. 1. The overall template for TPU design.
unit that acts as steering logic to prepare the operands for
low-precision computation [36]. These multipliers typically
balance accuracy and power efficiency. For example, the
logarithmic multiplier introduced in [29] emphasizes accuracy,
while the multipliers in [37] are designed to reduce power and
latency. On the other hand, most approximate adders, such as
lower part OR adder (LOA) [38], exploit the fact that extended
carry propagation is infrequent, allowing adders to be divided
into independent sub-adders shortening the critical path. To
preserve computational accuracy, the approximation is applied
to the least significant bits of the operands, while the most
significant bits remain accurate.
III. TPU-G ENFRAMEWORK
A. Architectural Template
Developing a Generic Template. The TPU architecture
utilizes a systolic array of PEs with MAC units for efficient
matrix and vector computations. This design enhances perfor-
mance and reduces energy consumption by reusing data, mini-
mizing buffer operations [1]. Input data propagates diagonally
through the array in parallel. The TPU template, illustrated
in Fig. 1, extends the TPU’s systolic array with Output
Stationary (OS) dataflow to enable concurrent approximation
of input feature maps (IFMaps) and weights. It comprises
five components: weight/IFMap memory, FIFOs, a controller,
Pre-Approximate Units (PAUs), and Approximate Processing
Elements (APEs). The weights and IFMaps are stored in their
respective memories, with the controller managing memory
access and data transfer to FIFOs per the OS dataflow. PAUs,
positioned between FIFOs and APEs, dynamically truncate
high-precision operands to lower precision before sending
them to APEs, which perform MAC operations using ap-
proximate multipliers and adders. Sharing PAUs across rows
and columns reduces hardware overhead, introducing minimal
latency but significantly improving overall performance [39].

TABLE II
APPROXIMATE MULTIPLIER HYPER -PARAMETERS
Design Parameter Description Default
BAM [40] VBL No. of zero bits during partial product generation W/2
ALM LOA [41] M Inaccurate part of LOA adder W/2
ALM MAA3 [41] M Inaccurate part of MAA3 adder W/2
ALM SOA [41] M Inaccurate part of SOA adder W/2
ASM [42] Nibble Width number of precomputed alphabets 4
DRALM [37] MULT DW Truncated bits of each operand W/2
RoBA [43] ROUND WIDTH Scales the widths of the shifter 1
Highly-Parameterized RTL Code. We design highly flexible
and parameterized RTL codes for 13 different approximate
adders and 12 different approximate multipliers as representa-
tive approximate circuits. For the approximate adders, we have
two tunable parameters: the bit-width and the imprecise part.
The bit-width specifies the number of bits for each operand
and the imprecise part specifies the number of inexact bits
in the adder output. For the approximate multipliers, we have
one common parameter, i.e., Width (W), which specifies the
bit-width of the multiplication operands. We also have more
tunable parameters based on specific multipliers, some of
which are listed in Table II. We leveraged the parametrized
RTL library of approximate arithmetic circuits to build a TPU
library that enables automatic selection of the systolic array
sizeS, bit precision n, and one of the approximate multipliers
and approximate adders. The internal parameters that are used
to tune the approximate arithmetic libraries are also included
in the TPU parameterized RTL library, thus, allowing the
user to have complete flexibility to adjust their designs to
meet specific hardware specifications and application accuracy
requirements. Moreover, we developed a design automation
methodology, enabling the automatic implementation and sim-
ulation of many TPU circuits in various simulation platforms
such as Design Compiler and Vivado. In addition to the
highly parameterized RTL codes, we developed TCL and
Python scripts to autonomously measure their error, area,
performance, and power dissipation under various constraints.
B. Framework Overview
TPU-Gen framework depicted in Fig. 2 targets the devel-
opment of domain-specific LLMs, emphasizing the interplay
between the model’s responses and two key factors: the input
prompt and the model’s learned parameters. The framework
optimizes both elements to enhance LLM’s performance. An
initial prompt conveying the user’s intent and key software
and hardware specifications of the intended TPU design and
application is enabled through the Prompt Generator in Step
1. A verbal description of a tensor processing accelerator
design can often result in a many-to-one mapping as shown
in Fig. 3(a), especially when such descriptions do not align
with the format of the training dataset. This misalignment
increases the likelihood of hallucinations in the LLM’s output,
potentially leading to faulty designs [44]. To minimize hal-
lucinations and incorrect outputs in LLM-generated designs,
studies have shown that inputs adhering closely to patterns ob-
served in the training data produce more accurate and desirable
results [17], [18]. However, this critical aspect has often been
overlooked in previous state-of-the-art research [9], with some
researchers opting instead to address the issue through prompt
APTPU Generation Framework Output33
44
556622
Automated Code 
Validation
Power
AreaDelayDelayFine-tuned LLM
APTPU w. 
needed perfLLMMulti -shot Learning\ Retrieval -Augmented 
Generation (RAG)
Generate Code11
InputUser prompt
Prompt 
GeneratorInvalidInvalidValidValid77
Data-setFig. 2. The proposed TPU-Gen framework.
optimization techniques [18]. In this framework, we tackle the
problem by employing a script that extracts key features, such
as systolic size and relevant metrics, from any given verbal
input by the user. These features are then embedded into a
template, which serves as the prompt for the LLM input. As
a domain-specific LLM, TPU-Gen focuses on generating the
most valuable RTL top file detailing the circuit, and blocks
involved in the presented architectural template in Section
III.A.
An immediate usage of the proposed dataset explained
in Section III.C in TPU-Gen is to help fine-tune a generic
LLM for the task of TPU design, where the input with
a prompt will be fed to the LLM (Step 2in Fig. 2).
Equivalently, one may employ ICL, or multi-shot learning as
a more computationally efficient compromise to fine-tuning
[24]. The multi-shot prompting techniques can be used where
the proposed dataset will function as the source for multi-
shot examples. Given that the TPU-Gen dataset integrates
verbal descriptions with corresponding TPU systolic array
design pairs, the LLM generates a TPU’s top-level file as the
output in Verilog. This top-level file includes all necessary
architectural module dependencies to ensure a fully functional
design (step 3). Further, we propose to leverage the RAG
module to generate the other dependency files into the project,
completing the design (step 4). Next, a third-party quality
evaluation tool can be employed to provide a quantitative
evaluation of the design, verify functional correctness, and
integrate the design with the full stack (step 5). Here,
for quality and functional evaluation, the generated designs,
initially described in Verilog, are synthesized using YOSYS
[45]. This synthesis process incorporates an automated RTL-
to-GDSII validation stage, where the generated designs are
evaluated and classified as either Valid orInvalid based on
the completeness of their code sequences and the correctness
of their input-output relationships. Valid designs proceed to
resource validation, where they are optimized with respect
to Power, Performance, and Area (PPA) metrics. In contrast,
designs flagged as Invalid initiate a feedback loop for error
User
(a)I want to design a TPU with  16 
processing elements for ...
I need a 16x16 systolic array with a 
dataflow. With support.. bits.. for app...Description 1
Description nDescription 2different user inputs
Generate the entire code 
for the <systolic_size> 
with….. following..  
input<bitwidth> …..
(b)Prompt 
GeneratorDescription 
1
Description 
2Description 
2
Description 
3Description 
3 LLM LLM
Wrong Design
(c)Desired DesignPrompt 
Generator
 LLM LLM
UserDescription 
1
Description 
2Description 
2
Description 
3Description 
3
Fig. 3. (a) Multiple descriptions for a single TPU design demonstrate that
a design can be verbally defined in numerous ways, potentially misleading
LLMs in generating the intended design, (b) Proposed prompt generator
extracts the required features from the given verbal descriptions, (c) Using
a script to generate a verbal description aligned with the training data.

analysis and subsequent LLM retraining, enabling iterative
refinement (steps 2to6) to achieve predefined performance
criteria. Ultimately, designs that successfully pass these stages
in step 7are ready for submission to the foundry.
C. Dataset Curation
Leveraging the parameterized RTL code of the TPU, we
develop a script to systematically explore various architectural
configurations and generate a wide range of designs within
the proposed framework (step 1in Fig. 4). The generated
designs undergo synthesis and functional verification (step
2). Subsequently, the OpenROAD suite [46] is employed to
produce PPA metrics (step 3). The PPA data is parsed using
Pyverilog (step 4), resulting in the creation of a detailed,
multi-level dataset that captures the reported PPA metrics
(step 5). Steps 1to 3are iterated until all architectural
variations are generated. The time required for each data
point generation varies depending on the specific configura-
tion. To efficiently populate the TPU-Gen dataset, we utilize
multiple scripts that automate the generation of data points
across different systolic array sizes, ensuring comprehensive
coverage of design space exploration. Fig. 4 shows the detailed
methodology underpinning our dataset creation. The validation
when compared to prior works [10], [47] understanding we
work in a different design space abstraction makes it tough
to have a fair comparison. However, looking by the scale of
operation and the framework’s efficiency we require minimal
efforts comparatively.
Fig. 5 visualizes the selection of different circuits to make
PAUs and APEs accommodating different input Data Widths
(DW) (8, 16, 32 bits) and Weight Widths (WW) (ranging
from 3 to 32 bits) to generate approximate MAC units. These
feature units highlight the flexible template of the TPU and
enhance its adaptability and performance across various DNN
workloads. Including lower bit-width weights is particularly
advantageous for highly quantified models, enabling efficient
processing with reduced computational resources.
TPU-Gen dataset offers 29,952 possible variations for a sys-
tolic array size with 8 different systolic array implementations
to facilitate various workloads spanning from 4 ×4 for smaller
loads to 256 ×256 to crunch bigger DNN workloads. Account-
ing for the systolic size variations in the TPU-Gen dataset
promises a total of 29,952 ×8 = 2,39,616 data points with
PPA metrics reported. While TPU-Gen is constantly growing
with newer data points, we checkpoint our dataset creation
currently reported as having 25,000 individual TPU designs.
We provide two variations: (i)A top module file consisting of
APTPU
CONFIG 
FILES
Tune variables,
features
Verify, Synthesize.Verification
Verify, Synthesize.Verification
 OpenRoad
OpenRoadgranulated 
prompt
APTPU +
 Metrics 
corpusIterative 
process
11
223344
PPA reportsAPTPU +
 Metrics +
Descriptions
55
APTPU -Gen
Fig. 4. TPU-Gen dataset curation.
APTPU (MxN)IFMapWeight
XX
++X
+<<XX
++X
+<< XX
++X
+<<
XX
++X
+<<Weight
IFMap
APE APEAPE APEApproximate Adders
SETA, HERLOA, 
MHEAA...10 more
Approximate 
Multipliers
BAM, UDM, 
ALM_LOA …10 moreApproximate Adders
SETA, HERLOA, 
MHEAA...10 more
Approximate 
Multipliers
BAM, UDM, 
ALM_LOA …10 moreDW : [8,16,32]
WW : [3,4,5,6,7,8,16,32]
Mult_DW : [2,3,4,…,12]DW : [8,16,32]
WW : [3,4,5,6,7,8,16,32]
Mult_DW : [2,3,4,…,12]PAUPAUFig. 5. An example of one category and its design space parameters.
details of the entire circuit implementation, which can be used
in cases such as RAG implementation to save the computation
resources, and (ii)A detailed, multi-level granulated dataset,
as depicted in Fig. 6, is curated by adapting MG-Verilog
[17] to assist LLM in generating Verilog code to support
the development of a highly sophisticated, fine-tuned model.
This model facilitates the automated generation of individual
hardware modules, intelligent integration, deployment, and
reuse across various designs and architectures. Please note
that due to the domain-specific nature of the dataset, some
data redundancy is inevitable, as similar modules are reused
and reconfigured to construct new TPUs with varying archi-
tectural configurations. This structured dataset enables efficient
exploration and customization of TPU designs while ensuring
that the generated modules can be systematically adapted for
different design requirements, leading to enhanced flexibility
and scalability in hardware design automation. Additionally,
we provide detailed metrics for each design iteration, which
aid the LLM in generating budget-constrained designs or
in creating an efficient design space exploration strategy to
accelerate the result optimization process.
IV. E XPERIMENT RESULTS
A. Objectives
We designed four distinct experiments employing various
approaches, each tailored to the unique capabilities of LLMs
such as GPT [48], Gemini [49], and Claude [50], as well
as the best open-source models from the leader board [51].
Each model is deployed in experiments aligning with the
study’s objectives and anticipated outcomes. Experiments 1
focus on observing the prompting mechanism that assists LLM
in generating the desired output by implementing ICL; with
this knowledge, we develop the prompt template discussed in
Sections III-B. Experiment 2 focuses on adapting the proposed
TPU-Gen framework by fine-tuning LLM models. For fine-
tuning, we used 4 ×A100 GPU with 80GB VRAM. Experi-
ment 3 is to demonstrate the effectiveness of RAG in TPU-
 
Metrics: {"Area": "29162", "WNS": " -12.268", 
"Total Power": "4.21e -03"}
`define DW 8       // Choose IFMAP bitwidth 
`define M 4       // Choose M dimensions of the 
systolic array
`define N 4        // Choose N dimensions of the 
systolic array
…..
`define HERLOA  //APADDER 
…..
`ifdef MITCHELL  ...
    `define SHARED_PRE_APPROX
`elsif ALM_SOA
    `define SHARED_PRE_APPROX
`elsif ALM_LOA
    `define SHARED_PRE_APPROX
`elsif ROBA
   ……..block_0: This code block defines various preprocessor macros to configure the 
design parameters, such as the type of …. nibble width (`NIBBLE_WIDTH`), 
bitwidths for IFMAP (`DW`) …., dimensions of the systolic array (`M` and `N`), 
accurate part of approximate multipliers (`MULT_DW`).
block_4: This code ….. related to different approximate... the `ALM` macro.BLOCK SUMMARY
The provided Verilog code represents a design for a 4x4 systolic array 
implementation …..the type of multiplier, adder, and other design choices.
….. features, …. pre-approximation (`SHARED_PRE_APPROX`). These 
macros are controlled by the selection. Overall, this code represents by 
…..adjusting the preprocessor macros.DETAILED GLOBAL SUMMARY
The provided Verilog ….4x4 systolic array design that utilizes ….adder 
(HERLOA), bitwidths, the design is highly configurable, with the ability … 
features like ….This flexibility allows the design to be tailored … leading to 
improvements in area, power, and timing performance …..such as machine 
learning efficiency are critical factors.HIGH -LEVEL GLOBAL SUMMARY
Fig. 6. An example of a data point by adapting MG-V format.

TABLE III
PROMPTS TO SUCCESSFULLY GENERATE EXACT TPU MODULES VIA TPU-G EN.
Module Generation Module IntegrationLLM ModelPass@1 Pass@3 Pass@5 Pass@10 Pass@1 Pass@3 Pass@5 Pass@10
Mistral-7B (Q3) 17% 83% 100% 100% 0% 25% 75% 100%
CodeLlama-7B (Q4) 0% 50% 83% 100% 0% 50% 75% 100%
CodeLlama-13B (Q4) 66% 83% 100% 100% 25% 75% 100% 100%
Claude 3.5 Sonnet 83% 100% 100% 100% 75% 100% 100% 100%
ChatGPT-4o 83% 100% 100% 100% 50% 100% 100% 100%
Gemmini Advanced 50% 50% 74% 91% 25% 75% 74% 91%
Gen and it’s applicability for hardware design. Experiment
4 tests the TPU-Gen framework’s ability to generate designs
efficiently with an industry-standard 45nm technology library.
Throughout the process, we also consider hardware under the
given PPA budget to ensure the feasibility of achieving the
objectives outlined in the initial phases.
B. Experiments and Results
1) Experiment 1: ICL-Driven TPU Generation and Approx-
imate Design Adaptation.: We evaluate the capability of LLMs
to generate and synthesize a novel TPU architecture and its
approximate version using TPU-Gen. Utilizing the prompt
template from [18], we refined it to harness LLM capabilities
better. LLM performance is assessed on two metrics: (i)Mod-
ule Generation —the ability to generate required modules,
and (ii)Module Integration —the capability to construct the
top module by integrating components. We tested commercial
models like [48], [49] via chat interfaces and open-source
models listed in Table III, using LM Studio [52]. For the TPU,
we successfully developed the design and obtained the GDSII
layout (Fig. 8(a)). Commercial models performed well with a
single prompt at pass@1, averaging 72% in module generation
and 50% in integration. Open-source models performed better
with the increase of pass@k, averaging 72% for pass@1
in module generation to 100% and 50% to 100% upscale
from pass@3 to pass@10 in integration. For the approximate
TPU, involving approximate circuit algorithms, we provided
example circuits and used ICL and Chain of Thought (CoT)
to guide the LLMs. Open-source models struggled due to a
lack of specialized knowledge, as shown in Fig. 7. The design
layout from this experiment is in Fig. 8(b). All outputs were
manually verified using test benches. This is the first work to
generate both exact and approximate TPU architectures using
prompting to LLM. However, significant human expertise and
intervention are required, especially for complex architectures
like approximate circuits. To minimize the human involve-
ment, we implement fine-tuning.
123456
LLM Models02468Number of PromptsChatGPT 4o
Gemmini Advanced
Claude
Codellama 13B
Codellama 7B
Mistral 7B
510 15 20 25
APTPU Modules0510152025Average Prompts for LLMsCommercial LLMs Open-Sourced LLMs
Trendline Commercial Trendline Open-Sourced
(b) (a)
Fig. 7. Average TPU-Gen prompts for (a) Module Generation, and (b) Module
Integration via LLMs.Takeaway 1. LLMs with efficient prompting are capable of
generating exact and approximate TPU modules and integrate
them to create complete designs. However, human involvement
is extensively required, especially for novel architectures. Fine-
tuning LLMs is necessary to reduce human intervention and
facilitate the exploration of new designs.
2) Experiment 2: Full TPU-Gen Implementation: This
experiment investigates cost-efficient approaches for adapt-
ing domain-specific language models to hardware design.
In previous experiments, we observed that limited spatial
and hierarchical hardware knowledge hindered LLM perfor-
mance in integrating circuits. The TPU-Gen template (Fig.
2) addresses this by delegating creative tasks to the LLM
and retrieving dependent modules via RAG, optimizing AI
accelerator design while reducing computational overhead and
minimizing LLM hallucinations. ICL experiments show that
fine-tuning enhances LLM reliability. The TPU-Gen proposes
a way to develop domain-specific LLMs with minimal data.
The experiment used a TPU-Gen dataset version 1 of 5,000
Verilog headers DW and WW inputs. This dataset comprises
systolic array implementations with biased approximate circuit
variations. We split data statically in 80:20 for training and
testing open-source LLMs [51], with two primary goals of 1.
Analyzing the impact of the prompt template generator on the
fine-tuned LLM’s performance (Table IV). 2.Investigating the
RAG model for hardware development.
All models used Low-Rank Adaptation (LoRA) fine-tuning
with the Adam optimizer at a learning rate of 1e−5. The fine-
tuned models were evaluated to generate the desired results
efficiently with a random prompt at pass@ 1to generate the
TPU. From Table IV, we can observe that the outputs without
the prompt generator are labeled as failures as they were
unsuitable for further development and RAG integration. We
can observe the same prompt when parsed to the prompt-
template generator with a single try; we score an accuracy
of 86.6%. Further, we used RAG and then processed the
generated Verilog headers for module retrieval. According to
[11], LLMs tend to prioritize creativity and finding innovative
solutions, which often results in straying from the data. To ad-
Fig. 8. A GDSII layout of (a) TPU, (b) TPU by prompting LLM, (c)
approximate TPU by TPU-Gen framework.

TABLE IV
PROMPT GENERATOR VS HUMAN INPUTS TO FINE-TUNED MODELS .
Prompt Template Human InputModelsPass Fail Pass Fail
CodeLlama-7B-hf 27 03 01 29
CodeQwen1.5 -7B 25 05 0 30
Mistral -7B 28 02 02 28
Starcoder2-7B 24 06 0 30
dress this, we employed a compute and cost-efficient method.
This shows that the fine-tuning along with RAG can greatly
enhance the performance. Fig. 8(c) shows the GDSII layout
of the design generated by the TPU-Gen framework.
Takeaway 2. Prompting techniques such as prompt template steer
LLM to generate desired results after fine-tuning, as observed 86%
success in generation. RAG, a cost-efficient method to generate the
hardware modules reliably, completing the entire Verilog design
for an application with minimal computational overhead.
3) Experiment 3: Significance of RAG: To assess the ef-
fectiveness of RAG in the TPU-Gen framework, we eval-
uated 1,000 Verilog header codes generated by fine-tuned
LLMs under two conditions: with and without RAG inte-
gration. Table V presents results over 30 designs tested by
our framework to generate complete project files. Without
RAG, failures occurred due to output token limitations and
hallucinated variables. RAG is essential as the design is not
a standalone file to compile. Validated header codes were
provided in the RAG-enabled pipeline, and required modules
were dynamically retrieved from the RAG database, ensuring
fully functional and accurate designs. Conversely, models
without RAG relied solely on internal knowledge, leading
to hallucinations, token constraints, and incomplete designs.
Models using RAG consistently achieved pass rates exceeding
95%, with Mistral-7B and CodeLlama-7B-hf attaining 100%
success. In contrast, all models failed entirely without RAG,
underscoring its pivotal role in ensuring design accuracy and
addressing LLM limitations. RAG provides a robust solution to
key challenges in fine-tuned LLMs for TPU hardware design
by retrieving external information from the RAG database,
ensuring contextual accuracy, and significantly reducing hal-
lucinations. Additionally, RAG dynamically fetches dependen-
cies in a modular manner, enabling the generation of complete
and accurate designs without exceeding token limits. RAG is a
promising solution in this context since our models were fine-
tuned with only Verilog header data detailing design features.
However, fine-tuning models with the entire design data would
expose LLMs to severe hallucinations and token limitations,
making generating detailed and functional designs challenging.
TABLE V
SIGNIFICANCE OF RAG INTPU-G EN.
With RAG Without RAGLLM ModelPass(%) Fail(%) Pass(%) Fail(%)
CodeLama-7B-hf 100 0 0 100
Mistral-7B 100 0 0 100
CodeQwen1.5-7B 95 5 0 100
StarCoder2-7B 98 2 0 100
Takeaway 3. The experiment highlights the significance of the
RAG usage with a fine-tuned model to avoid hallucinations and
let LLM be creative consistently.
Power Cons. (mW)
LeNet
ResNet18VGG16ResNet56050100150200
Manual effort
APTPU-GenArea (m2)
LeNet
ResNet18VGG16ResNet560123456104
Manual effort
APTPU-GenLatency (ms)
LeNet
ResNet18VGG16ResNet560102030405060
Area Constraint Violated!
(a) (b) (c)Latency Constraint Met!
Power Constraint Met!Fig. 9. PPA metrics comparison for TPU architectures generated by TPU-Gen
and the manual user: (a) Power consumption, (b) Area, (c) Latency.
4) Experiment 4: Design Generation Efficiency: Building
on the successful generation of approximate TPU in exper-
iment 2, here we evaluate and benchmark the architectures
produced by the TPU-Gen framework as the work performed
in this paper is the first of it’s kind we are comparing against
manual optimization created by expert human designers, fo-
cusing on power, area, and latency as shown in Fig. 9(a)-(c).
We utilize four DNN architectures for this evaluation: LeNet,
ResNet18, VGG16, and ResNet56, performing inference tasks
on the MNIST, CIFAR-10, SVHN, and CIFAR-100 datasets.
In the manually optimized designs, a skilled hardware engineer
fine-tunes parameters within the TPU template. This iterative
optimization process is repeated until no further performance
gains can be achieved within a reasonable timeframe of
approximately one day [9], or the expert determines, based
on empirical results, that additional refinements would yield
minimal benefits. Using the PPA metrics as reference values
(e.g., 100mW, 0.25mm2, 48ms for ResNet56), both TPU-
Gen and the manual user are tasked with generating the
TPU architecture. Fig. 9 illustrates that across a range of
network architectures, TPU-Gen consistently yields results
with minimal deviation from the reference benchmarks. In
contrast, the manual designs exhibit significant violations in
terms of PPA.
Takeaway 4. TPU-Gen consistently yields results with minimal
deviation from the PPA reference, whereas the manual designs
exhibit significant violations.
V. C ONCLUSIONS
This paper introduces TPU-Gen, a novel dataset and a novel
framework for TPU generation, addressing the complexities of
generating AI accelerators amidst rapid AI model evolution.
A key challenge, hallucinated variables, is mitigated using
an RAG approach, dynamically adapting hardware modules.
RAG enables cost-effective, full-scale RTL code generation,
achieving budget-constrained outputs via fine-tuned models.
Our extensive experimental evaluations demonstrate superior
performance, power, and area efficiency, with an average
reduction in area and power of 92% and 96% from the manual
optimization reference values. These results set new standards
for driving advancements in next-generation design automation
tools powered by LLMs. We are committed to releasing the
dataset and fine-tuned models publicly if accepted.

REFERENCES
[1] N. Jouppi, C. Young, N. Patil, and D. Patterson, “Motivation for and
evaluation of the first tensor processing unit,” IEEE Micro , vol. 38, no. 3,
pp. 10–19, 2018.
[2] H. Genc et al. , “Gemmini: Enabling systematic deep-learning archi-
tecture evaluation via full-stack integration,” in 2021 58th ACM/IEEE
Design Automation Conference (DAC) . IEEE, 2021, pp. 769–774.
[3] H. Sharma, J. Park, D. Mahajan, E. Amaro, J. K. Kim, C. Shao,
A. Mishra, and H. Esmaeilzadeh, “From high-level deep neural models
to fpgas,” in 2016 49th Annual IEEE/ACM International Symposium on
Microarchitecture (MICRO) . IEEE, 2016, pp. 1–12.
[4] W.-Q. Ren et al. , “A survey on collaborative dnn inference for edge
intelligence,” Machine Intelligence Research , vol. 20, no. 3, pp. 370–
395, 2023.
[5] D. Vungarala, M. Morsali, S. Tabrizchi, A. Roohi, and S. Angizi,
“Comparative study of low bit-width dnn accelerators: Opportunities
and challenges,” in 2023 IEEE 66th International Midwest Symposium
on Circuits and Systems (MWSCAS) . IEEE, 2023, pp. 797–800.
[6] P. Xu and Y . Liang, “Automatic code generation for rocket chip rocc
accelerators,” 2020.
[7] S. Angizi, Z. He, A. Awad, and D. Fan, “Mrima: An mram-based in-
memory accelerator,” IEEE Transactions on Computer-Aided Design of
Integrated Circuits and Systems , vol. 39, no. 5, pp. 1123–1136, 2019.
[8] K. Chang, Y . Wang, H. Ren, M. Wang, S. Liang, Y . Han, H. Li,
and X. Li, “Chipgpt: How far are we from natural language hardware
design,” arXiv preprint arXiv:2305.14019 , 2023.
[9] Y . Fu, Y . Zhang, Z. Yu, S. Li, Z. Ye, C. Li, C. Wan, and Y . C. Lin,
“Gpt4aigchip: Towards next-generation ai accelerator design automation
via large language models,” in 2023 IEEE/ACM International Confer-
ence on Computer Aided Design (ICCAD) . IEEE, 2023, pp. 1–9.
[10] S. Thakur, B. Ahmad, H. Pearce, B. Tan, B. Dolan-Gavitt, R. Karri, and
S. Garg, “Verigen: A large language model for verilog code generation,”
ACM Transactions on Design Automation of Electronic Systems , vol. 29,
no. 3, pp. 1–31, 2024.
[11] X. Jiang, Y . Tian, F. Hua, C. Xu, Y . Wang, and J. Guo, “A survey on
large language model hallucination via a creativity perspective,” arXiv
preprint arXiv:2402.06647 , 2024.
[12] J. Blocklove, S. Garg, R. Karri, and H. Pearce, “Chip-chat: Chal-
lenges and opportunities in conversational hardware design,” in 2023
ACM/IEEE 5th Workshop on Machine Learning for CAD (MLCAD) .
IEEE, 2023, pp. 1–6.
[13] S. Thakur, J. Blocklove, H. Pearce, B. Tan, S. Garg, and R. Karri, “Au-
tochip: Automating hdl generation using llm feedback,” arXiv preprint
arXiv:2311.04887 , 2023.
[14] R. Ma, Y . Yang, Z. Liu, J. Zhang, M. Li, J. Huang, and G. Luo, “Ver-
ilogreader: Llm-aided hardware test generation,” arXiv:2406.04373v1 ,
2024.
[15] W. Fang et al. , “Assertllm: Generating and evaluating hardware
verification assertions from design specifications via multi-llms,”
arXiv:2402.00386v1 , 2024.
[16] M. Liu, N. Pinckney, B. Khailany, and H. Ren, “Verilogeval:
Evaluating large language models for verilog code generation,”
arXiv:2309.07544v2 , 2024.
[17] Y . Zhang, Z. Yu, Y . Fu, C. Wan, and Y . C. Lin, “Mg-verilog: Multi-
grained dataset towards enhanced llm-assisted verilog generation,” arXiv
preprint arXiv:2407.01910 , 2024.
[18] D. Vungarala, M. Nazzal, M. Morsali, C. Zhang, A. Ghosh,
A. Khreishah, and S. Angizi, “Sa-ds: A dataset for large language model-
driven ai accelerator design generation,” arXiv e-prints , pp. arXiv–2404,
2024.
[19] H. Wu et al. , “Chateda: A large language model powered autonomous
agent for eda,” IEEE Transactions on Computer-Aided Design of Inte-
grated Circuits and Systems , 2024.
[20] B. Nadimi and H. Zheng, “A multi-expert large language model archi-
tecture for verilog code generation,” arXiv preprint arXiv:2404.08029 ,
2024.
[21] Y . Lu, S. Liu, Q. Zhang, and Z. Xie, “Rtllm: An open-source benchmark
for design rtl generation with large language model,” in 2024 29th Asia
and South Pacific Design Automation Conference (ASP-DAC) . IEEE,
2024, pp. 722–727.
[22] D. Vungarala, S. Alam, A. Ghosh, and S. Angizi, “Spicepilot: Navigating
spice code generation and simulation with ai guidance,” arXiv preprint
arXiv:2410.20553 , 2024.[23] Y . Lai, S. Lee, G. Chen, S. Poddar, M. Hu, D. Z. Pan, and P. Luo,
“Analogcoder: Analog circuit design via training-free code generation,”
arXiv preprint arXiv:2405.14918 , 2024.
[24] D. Dai, Y . Sun, L. Dong, Y . Hao, S. Ma, Z. Sui, and F. Wei, “Why
can gpt learn in-context? language models implicitly perform gradient
descent as meta-optimizers,” arXiv preprint arXiv:2212.10559 , 2022.
[25] G. Izacard et al. , “Atlas: Few-shot learning with retrieval augmented
language models,” Journal of Machine Learning Research , vol. 24, no.
251, pp. 1–43, 2023.
[26] J. Chen, H. Lin, X. Han, and L. Sun, “Benchmarking large lan-
guage models in retrieval-augmented generation,” arXiv preprint
arXiv:2309.01431 , 2023.
[27] R. Qin et al. , “Robust implementation of retrieval-augmented
generation on edge-based computing-in-memory architectures,”
arXiv:2405.04700v1 , 2024.
[28] A. Roohi, S. Sheikhfaal, S. Angizi, D. Fan, and R. F. DeMara, “Apgan:
Approximate gan for robust low energy learning from imprecise compo-
nents,” IEEE Transactions on Computers , vol. 69, no. 3, pp. 349–360,
2019.
[29] M. S. Ansari, B. Cockburn, and J. Han, “An improved logarithmic mul-
tiplier for energy-efficient neural computing,” IEEE Trans. on Comput. ,
vol. 70, pp. 614–625, 2021.
[30] S. Angizi, M. Morsali, S. Tabrizchi, and A. Roohi, “A near-sensor
processing accelerator for approximate local binary pattern networks,”
IEEE Transactions on Emerging Topics in Computing , vol. 12, no. 1,
pp. 73–83, 2023.
[31] H. Jiang, S. Angizi, D. Fan, J. Han, and L. Liu, “Non-volatile approxi-
mate arithmetic circuits using scalable hybrid spin-cmos majority gates,”
IEEE Transactions on Circuits and Systems I: Regular Papers , vol. 68,
no. 3, pp. 1217–1230, 2021.
[32] S. Angizi, Z. He, A. S. Rakin, and D. Fan, “Cmp-pim: an energy-efficient
comparator-based processing-in-memory neural network accelerator,” in
Proceedings of the 55th Annual Design Automation Conference , 2018,
pp. 1–6.
[33] S. Angizi, H. Jiang, R. F. DeMara, J. Han, and D. Fan, “Majority-based
spin-cmos primitives for approximate computing,” IEEE Transactions
on Nanotechnology , vol. 17, no. 4, pp. 795–806, 2018.
[34] M. E. Elbtity, H.-W. Son, D.-Y . Lee, and H. Kim, “High speed,
approximate arithmetic based convolutional neural network accelerator,”
2020 International SoC Design Conference (ISOCC) , pp. 71–72,
2020. [Online]. Available: https://api.semanticscholar.org/CorpusID:
231826033
[35] H. Younes, A. Ibrahim, M. Rizk, and M. Valle, “Algorithmic level
approximate computing for machine learning classifiers,” 2019 26th
IEEE Int. Conf. on Electron., Circuits and Syst. (ICECS) , pp. 113–114,
2019.
[36] S. Hashemi, R. I. Bahar, and S. Reda, “DRUM: A dynamic range
unbiased multiplier for approximate applications,” 2015 IEEE/ACM Int.
Conf. on Comput.-Aided Design (ICCAD) , pp. 418–425, 2015.
[37] P. Yin, C. Wang, H. Waris, W. Liu, Y . Han, and F. Lombardi, “Design
and analysis of energy-efficient dynamic range approximate logarithmic
multipliers for machine learning,” IEEE Transactions on Sustainable
Computing , vol. 6, no. 4, pp. 612–625, 2021.
[38] A. Dalloo, A. Najafi, and A. Garcia-Ortiz, “Systematic design of an
approximate adder: The optimized lower part constant-or adder,” IEEE
Transactions on Very Large Scale Integration (VLSI) Systems , vol. 26,
no. 8, pp. 1595–1599, 2018.
[39] M. E. Elbtity, P. S. Chandarana, B. Reidy, J. K. Eshraghian, and R. Zand,
“Aptpu: Approximate computing based tensor processing unit,” IEEE
Transactions on Circuits and Systems I: Regular Papers , vol. 69, no. 12,
pp. 5135–5146, 2022.
[40] F. Farshchi et al. , “New approximate multiplier for low power digital
signal processing,” The 17th CSI International Symposium on Computer
Architecture & Digital Systems (CADS 2013) , pp. 25–30, 2013.
[41] W. Liu et al. , “Design and evaluation of approximate logarithmic
multipliers for low power error-tolerant applications,” IEEE Trans. on
Circuits and Syst. I: Reg. Papers , vol. 65, pp. 2856–2868, 2018.
[42] S. S. Sarwar et al. , “Energy-efficient neural computing with approximate
multipliers,” ACM Journal on Emerging Technologies in Computing
Systems (JETC) , vol. 14, pp. 1 – 23, 2018.
[43] R. Zendegani et al. , “Roba multiplier: A rounding-based
approximate multiplier for high-speed yet energy-efficient digital
signal processing,” IEEE Transactions on Very Large Scale Integration

(VLSI) Systems , vol. 25, pp. 393–401, 2017. [Online]. Available:
https://api.semanticscholar.org/CorpusID:206810935
[44] M. Niu, H. Li, J. Shi, H. Haddadi, and F. Mo, “Mitigating hallucina-
tions in large language models via self-refinement-enhanced knowledge
retrieval,” arXiv preprint arXiv:2405.06545 , 2024.
[45] (2024) Yosys. [Online]. Available: https://github.com/YosysHQ/yosys
[46] (2018) Openroad. [Online]. Available: https://github.com/
The-OpenROAD-Project/OpenROAD
[47] H. Pearce et al. , “Dave: Deriving automatically verilog from english,”
inMLCAD , 2020, pp. 27–32.
[48] (2024) Openai gpt-4. [Online]. Available: https://openai.com/index/
hello-gpt-4o/
[49] (2024) Gemini. [Online]. Available: https://deepmind.google
[50] (2023) Anthropic. [Online]. Available: https://www.anthropic.com
[51] Evalplus leaderboard. https://evalplus.github.io/leaderboard.html. Ac-
cessed: 2024-09-21.
[52] “Lm studio - discover, download, and run local llms,” https://lmstudio.
ai/, accessed: 2024-09-21.