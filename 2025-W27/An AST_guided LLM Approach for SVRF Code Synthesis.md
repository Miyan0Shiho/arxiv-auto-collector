# An AST-guided LLM Approach for SVRF Code Synthesis

**Authors**: Abanoub E. Abdelmalak, Mohamed A. Elsayed, David Abercrombie, Ilhami Torunoglu

**Published**: 2025-07-01 00:57:45

**PDF URL**: [http://arxiv.org/pdf/2507.00352v1](http://arxiv.org/pdf/2507.00352v1)

## Abstract
Standard Verification Rule Format (SVRF) is essential for semiconductor
applications like Design Rule Check (DRC), Layout Versus Schematic (LVS), and
Optical Proximity Correction (OPC) and it faces challenges as advancing nodes
create complex design rules that renders traditional SVRF development
ineffective and highlight an expertise gap. This paper introduces a novel
methodology integrating Abstract Syntax Tree (AST) embedding and
Retrieval-Augmented Generation (RAG) for enhanced SVRF code synthesis, ensuring
semantic accuracy and error minimization through structural validation with
domain-specific insights for precise code generation.
  We evaluate different T5-based models and propose an innovative SVRF-specific
scoring framework that complements standard metrics like BLEU and ROUGE-L. In
our approach, AST provides rigorous structural validation, while RAG infuses
relevant domain knowledge, effectively enhancing the code generation workflow.
  Testing on a comprehensive benchmark of 740 DRC rule implementations, our
methodology demonstrates up to a 40\% improvement in code generation accuracy
compared to basic text-based fine-tuning process. This fusion of industry
expertise with advanced coding strategies not only optimizes SVRF development
under limited dataset constraints but also creates a more intuitive and
efficient coding environment. Consequently, users can rapidly iterate through
design cycles, reduce manual error correction, and significantly improve
overall productivity.

## Full Text


<!-- PDF content starts -->

arXiv:2507.00352v1  [cs.SE]  1 Jul 2025An AST-guided LLM Approach for SVRF Code
Synthesis
1stAbanoub E. Abdelmalak 2ndMohamed A. Elsayed
Calibre Yield Enhancement Services Calibre Yield Enhancement Services
Siemens EDA Siemens EDA
Cairo, Egypt Munich, Germany
abanoub.elkess@siemens.com mohamed.a.elsayed@siemens.com
3rdIlhami Torunoglu 4thDavid Abercrombie
Calibre Research & Development Calibre Product Management
Siemens EDA Siemens EDA
CA, USA NC, USA
ilhami.torunoglu@siemens.com david.abercrombie@siemens.com
Abstract —Standard Verification Rule Format (SVRF) is es-
sential for semiconductor applications like Design Rule Check
(DRC), Layout Versus Schematic (LVS), and Optical Proximity
Correction (OPC) and it faces challenges as advancing nodes
create complex design rules that renders traditional SVRF de-
velopment ineffective and highlight an expertise gap. This paper
introduces a novel methodology integrating Abstract Syntax
Tree (AST) embedding and Retrieval-Augmented Generation
(RAG) for enhanced SVRF code synthesis, ensuring semantic
accuracy and error minimization through structural validation
with domain-specific insights for precise code generation.
We evaluate different T5-based models and propose an in-
novative SVRF-specific scoring framework that complements
standard metrics like BLEU and ROUGE-L. In our approach,
AST provides rigorous structural validation, while RAG infuses
relevant domain knowledge, effectively enhancing the code gen-
eration workflow.
Testing on a comprehensive benchmark of 740 DRC rule
implementations, our methodology demonstrates up to a 40%
improvement in code generation accuracy compared to basic text-
based fine-tuning process. This fusion of industry expertise with
advanced coding strategies not only optimizes SVRF development
under limited dataset constraints but also creates a more intuitive
and efficient coding environment. Consequently, users can rapidly
iterate through design cycles, reduce manual error correction,
and significantly improve overall productivity.
Index Terms —DRC, LLMs, SVRF, Calibre, EDA, Copilot,
AST, RAG
I.INTRODUCTION
The advancement in semiconductor technology toward
smaller nodes has introduced unprecedented complexity into
process development, necessitating precise methodologies.
This is particularly evident in domains such as Optical Proxim-
ity Correction (OPC), where accurate recipe creation ensures
fabricated designs adhere to stringent process margins. This
complexity manifests in thousands of intricate design rules and
device definitions that designers must follow. These are trans-
lated into rule-decks for critical verification steps managed by
Electronic Design Automation (EDA) tools, such as DesignRule Checking (DRC) and Layout Versus Schematic (LVS)
checking. The proprietary Standard Verification Rule Format
(SVRF) is crucial in developing these rule decks for design
compliance verification and OPC recipes for manufacturing.
However, the industry’s rapid growth has created an urgent
demand for engineers with deep process understanding and ro-
bust development capabilities. The extensive training required,
often spanning years, leads to a significant knowledge gap that
impacts development efficiency and scalability.
Large Language Models (LLMs) demonstrate potential for
automating code generation [1] for open-source languages.
While LLMs show promise in specialized domains [2], ap-
plying standard pre-trained LLMs to generate SVRF code
often results in high hallucination rates with syntactically and
semantically invalid outputs. This limitation stems from the
semiconductor industry’s constrained nature and the scarcity of
public information about development methodologies and ob-
jectives, highlighting the need for specialized LLM approaches
integrated with domain-specific safeguards.
A significant challenge lies in the absence of general SVRF
datasets, rooted in the proprietary nature of the language
and the semiconductor manufacturing companies that write
the code. Since SVRF rule-decks represent semiconductor
manufacturing processes, access remains restricted to a limited
set of specialized engineers.
We propose an Abstract Syntax Tree (AST) guided method-
ology for fine-tuning pre-trained LLMs, capturing SVRF’s
nuances without extensive training data. This approach is
enhanced by a Retrieval-Augmented Generation (RAG) com-
ponent that grounds outputs in verified patterns and documen-
tation, mappable to specific semiconductor manufacturing pro-
cesses or methodologies, thereby improving solutions across
SVRF sub-domains.
To validate our methodology, we apply it to DRC, devel-
oping rule decks for foundry design rule compliance. Our
dataset, generated using internal knowledge without links to

Fig. 1: System Design Overview
real-world foundry data, enables us to evaluate LLM perfor-
mance using natural language prompts for code generation. We
compare AST-guided models against purely text-based fine-
tuning, aiming to generalize the methodology across various
SVRF domains beyond DRC.
Our results demonstrate significant improvements in SVRF
code accuracy while reducing hallucinations and maintaining
logical consistency across complex layer interactions. This
approach supersedes traditional template-based code genera-
tion methodologies, enabling flexible, intelligent models with
diverse use-cases. We envision its role in agentic workflows
incorporating SVRF knowledge with multi-agent capabilities,
such as vision for layout inspection and tool integration,
benefiting both developers and end-users.
II.METHODOLOGY
Our approach to SVRF code synthesis combines the rea-
soning capabilities of LLMs with domain-specific validation
mechanisms. The system architecture, illustrated in Figure 1,
consists of two primary components that work in concert to
ensure accurate and efficient context-aware code generation.
(1) the retrieval block, where we match the given user input
query with the best possible matching given the current user
context and the knowledge database. (2) the generation block,
where we use our model to generate the code that fits the
current query.
A.Abstract Syntax Tree (AST)
1) AST Construction and Preprocessing:: Abstract Syntax
Trees (ASTs) are foundational, representing SVRF code’s
hierarchical syntactic structure. Each node corresponds to a
source code construct, and ASTs serve critical functions in our
preprocessing pipeline and LLM guidance, including: syntac-
tic validation, semantic analysis (capturing layer definitions,
control flow, command relationships for better LLM context),
and LLM integration (providing structured training data and a
basis for structural correctness evaluation).
We define SVRF’s core components (e.g., ”COMMAND”,
”LAYERS”) using ANTLR [3] grammar for precise AST
construction [4], ensuring generated code can be assessed for
integrity [5] (see Appendix A for an AST mapping exam-
ple). The AST structure enables comprehensive code analysis:
command recognition identifies operations and components;
layer parsing extracts names and maps relationships; conditionhandling manages constraints; and option organization pro-
vides a framework for parameters. This ensures components
are validated and coherently structured.
For optimal LLM consumption, initial ANTLR-parsed
SVRF examples are further preprocessed by:
•Streamlining the parse tree into a more abstract AST
(removing redundancies, standardizing node types).
•Serializing this AST into a linearized, bracketed string
(e.g., (COMMAND (OPTION val) ...) ) via depth-
first traversal, preserving hierarchy for LLM tokenization.
2) AST-Guided LLM Integration and Fine-tuning: Our
AST-guided approach enhances CodeT5’s capabilities [6] by
incorporating structural and semantic insights from AST repre-
sentation throughout both training and inference phases. This
integration is crucial for accurate SVRF code generation with
limited data, as it efficiently encodes SVRF syntax and logic
independent of specific values, mitigating extensive data aug-
mentation needs and reducing dataset size and computational
demands.
During fine-tuning, T5 models are trained to translate
Natural Language (NL) descriptions into SVRF code strings
using a specialized AST-weighted loss function. Candidate and
ground-truth SVRF are parsed into ASTs, and the loss function
compares these structures, penalizing discrepancies based on
their significance (e.g., higher penalties for errors in commands
or layers versus minor options). This structural feedback,
guided by weights reflecting SVRF grammar priorities (de-
tailed in Appendix B), helps the model learn syntactic and
semantic rules more effectively than standard losses, bridging
semantic understanding with syntactic requirements.
During inference, learned structural knowledge implicitly
guides decoding towards valid SVRF. Correctness is further
enhanced by lightweight ANTLR grammar parsing during
beam search or post-generation to penalize/discard malformed
snippets. Generated SVRF is parsed into an AST and validated
syntactically, with errors flagged or potentially corrected.
Fig. 2: Simulated Token-Level Attention Comparison: (a)
Without AST (b) With AST Guidance.
Figure 2 illustrates this contextual understanding. Without
AST guidance (Panel a), LLMs may correlate literal NL
tokens to SVRF counterparts but miss structural rules, leading
to errors like misplacing ”SPACE” after layers. With AST
guidance (Panel b), NL tokens correctly attend to correspond-
ing AST structural nodes (e.g., NL ”space” to COMMAND

(SPACE) AST node), enabling proper structural ordering and
syntactically correct SVRF generation.
Key advantages include enhanced syntactic and semantic
alignment through hierarchical code representation, reduced
errors via structural validation, efficient learning from limited
data, and better pattern recognition of command hierarchies
for coherent generation.
B.Model Architecture Selection
We selected the T5 architecture family for our experiments
due to its encoder-decoder architecture, which has shown
superior performance in structured generation tasks [7]. Un-
like autoregressive models like GPT, T5’s encoder-decoder
structure provides several key advantages for SVRF code
synthesis: bidirectional context understanding in the encoder
for capturing complex design rule relationships, structured
decoding process that better maintains syntactic consistency,
and enhanced ability to map between different formats (natural
language to code) through parallel attention mechanisms [8].
For our implementation, we utilized three variants of T5-
based models:
•T5-base [7]: The foundational model (220M parameters).
Specialized for code generation.
•Flan-T5-base [9]: An instruction-tuned variant (250M
parameters)
•CodeT5-base [6]: Code-specific pre-trained model (220M
parameters)
Importantly, we maintained the models’ original tokenizers
without custom modifications, demonstrating the adaptability
of standard pre-trained vocabularies to SVRF syntax.
C.Retrieval-Augmented Generation Workflow
At the core of our methodology is a RAG workflow, which
is designed to optimize the SVRF code generation process
by leveraging an extensive knowledge base. This system en-
hances the code generation capability LLMs by incorporating
contextual and domain-specific information.
1) Contextual Retrieval Mechanism: Our RAG mechanism
is tailored specifically for SVRF code patterns. Unlike con-
ventional RAG approaches, our system integrates semiconduc-
tor process knowledge and tool-specific syntax patterns. It
maintains a curated database of verified SVRF code snippets
that are indexed using not only syntactic characteristics but
also their associated physical verification intents. When tasked
with implementing a new DRC rule, the system performs an
analysis of the input specification to identify key verification
requirements.
2) Knowledge Graph Integration Through AST: The re-
trieval process is further enhanced by a knowledge graph that
encapsulates extensive domain-specific expertise. This graph
captures the relationships between various semiconductor pro-
cesses and their corresponding SVRF code implementations,
enabling the system to rank candidate SVRF patterns based
on both syntactic similarity and semantic relevance. In addi-
tion, by mapping the internal dataset to its equivalent AST
representation, our approach leverages structural context tofocus on the essential code elements—such as commands and
options—thereby reducing dependency on user-specific inputs
like layer names or values.
3) Prompt Enhancement and Code Generation: Once rele-
vant SVRF patterns are retrieved, they are used to formulate an
enhanced prompt for the LLM. This retrieval-informed prompt
provides structured clues and rich contextual information, sub-
stantially improving the model’s understanding and response
to user queries. As a result, the generated code snippets are
not only syntactically and structurally accurate but also closely
aligned with the intended verification objectives.
By integrating retrieval augmentation with advanced gen-
eration techniques, our workflow offers a sophisticated solu-
tion for SVRF code development. This approach effectively
combines powerful data-driven insights with intuitive code
synthesis, leading to improved accuracy and efficiency in
design rule checking and verification processes.
III. EXPERIMENTAL EVALUATION AND ANALYSIS
In this section, we present a systematic evaluation of our
approach through a series of progressive experiments that
validate our initial hypothesis. First, we introduce the dataset,
the selected pre-trained LLMs, and the evaluation metrics. We
then detail our experimental pipeline, analyzing the results at
each phase as summarized in Table II and analyzed in the
results subection.
A.Dataset Definition
Our experimental dataset is derived from an internal Design
Rule Checking (DRC) knowledge base, initially comprising
400 paired examples of natural language descriptions and
their corresponding SVRF code implementations. Through
data augmentation techniques using our internal LLM tools,
we expanded this to 741 diverse examples.
1) Data Structure and Representation: Each example in our
dataset consists of two main components:
•Input: NL description of design rules.
"Minimum spacing between METAL1 and METAL2
layers should not be less than 0.5um"
•Output: Corresponding SVRF code implementation.
SPACE_CMD METAL1 METAL2 >= 0.5 READ ALL {
REPORT "Spacing violation detected"
}
The dataset construction process involved:
•Initial Collection: 400 curated description-code pairs
from internal DRC knowledge
•Data Augmentation: LLM-based generation of variations
maintaining semantic validity
•Quality Assurance: Verification of generated examples by
domain experts
2) Exploratory Data Analysis: The final dataset of 741
examples exhibits the following complexity distribution:
The complexity categories are defined based on:
•Simple Rules (32.5%): Basic layer operations, single
command structures, and minimal option parameters

TABLE I: Distribution of Rules Across Complexity Categories
and Dataset Splits
Complexity Count Train Val Test
(80%) (10%) (10%)
Simple Rules 241 193 24 24
Moderate Rules 347 278 35 34
Complex Rules 153 122 15 16
Total 741 593 74 74
•Moderate Rules (46.8%): Multi-layer interactions, com-
bined operations, and standard option configurations
•Complex Rules (20.7%): Nested operations, multiple
layer dependencies, and advanced option combinations
The dataset’s quality is ensured through careful preservation
of verification intent and structural validity, while maintaining
diverse patterns across different complexity levels. Using a
standard 80-10-10 split ratio, we divided the 741 examples
into training (593), validation (74), and testing (74) sets. This
organization supports comprehensive code analysis and gener-
ation, enabling proper handling of commands, parameters, and
configurations while maintaining systematic error detection
throughout the development pipeline.
B.Baseline Models
To evaluate the effectiveness of our AST-guided fine-tuning
approach, we implement several baseline models, utilizing
different pre-trained language models as foundations:
•Pre-trained Local Models: As mentioned in Sec-
tion II-B, we evaluate three pre-trained transformer mod-
els: CodeT5-base, Flan-T5-base, and T5-base as our
baseline architecture.
•Large state-of the art LLMs: Claude sonnet 3.5 is used
through Direct prompting. Basic SVRF documentation is
added as part of the prompt context window to guide the
model.
These baselines were chosen to address specific research
questions:
•How much does structural awareness (AST guidance)
improve over standard fine-tuning across different pre-
trained models?
•Which pre-trained model architecture best suits SVRF
code generation?
•What is the relative contribution of retrieval versus struc-
tural guidance?
C.Evaluation Metrics
To comprehensively assess model performance across dif-
ferent aspects of SVRF code generation, we employ multi-
ple complementary metrics. These include traditional metrics
(Loss, BLEU, and ROUGE-L scores) and a novel AST-
weighted accuracy scoring mechanism specifically designed
for SVRF’s unique characteristics. The complete mathematical
formulations and detailed descriptions of these metrics are
provided in Appendix B.For our analysis, we focus on four key metrics (detailed
in Appendix B): Loss Score, measuring token-level predic-
tion accuracy; BLEU Score (Eq. (1)), evaluating similarity
to reference implementations; ROUGE-L Score (Eq. (2)),
assessing structural similarity and fluency; and our proposed
AST-Weighted Accuracy (Eq. (3)), which accounts for SVRF-
specific characteristics.
While BLEU and ROUGE-L offer valuable insights into lex-
ical similarity and fluency, they are primarily n-gram based and
may not fully capture the structural and semantic correctness
crucial for a domain-specific language like SVRF. SVRF’s
non-sequential command ordering, the critical significance of
layer sequence, and the hierarchical nature of its commands
and options necessitate a more nuanced evaluation. Our AST-
Weighted Accuracy is specifically designed to address these
characteristics by dissecting the generated code into its core
structural components and evaluating their correctness with
differential importance, as detailed in Appendix B.
This combination of metrics ensures our evaluation cap-
tures both general code generation quality and SVRF-specific
requirements. The AST-weighted scoring, detailed in Ap-
pendix B, is particularly important as it accounts for SVRF’s
non-sequential nature and the critical importance of layer
ordering in the generated code.
D.Experimental Pipeline
Our evaluation follows a three-phase approach to system-
atically assess model performance and the impact of AST
guidance.
1)Phase 1: Baseline Performance :Initial zero-shot eval-
uation reveals significant challenges in SVRF code generation
across all models. Despite their sophisticated pre-training,
models achieve 0% accuracy with high loss values (T5: 8.603,
Flan-T5: 4.398, CodeT5: 8.096). Flan-T5 demonstrates the
lowest initial loss, while T5 shows marginally better BLEU
(0.085) and ROUGE-L (0.296) scores, indicating limited trans-
fer of pre-trained capabilities to SVRF generation.
2)Phase 2: Standard Fine-tuning :Traditional supervised
fine-tuning yields substantial improvements, with all models
achieving training accuracies above 84%. CodeT5 demon-
strates particularly strong training performance (86.729% ac-
curacy, 0.989 BLEU, 0.994 ROUGE-L), followed by T5
and Flan-T5. In validation, CodeT5 maintains its lead with
50.995% accuracy, suggesting better generalization capabili-
ties even without structural guidance.
3)Phase 3: AST-Guided Fine-tuning :The integration of
AST-guided fine-tuning demonstrates statistically significant
performance improvements across all transformer architec-
tures, with CodeT5 exhibiting optimal convergence charac-
teristics. The model achieves a minimal cross-entropy loss
of 0.005 during training while maintaining 86.003% AST-
weighted accuracy, indicating efficient parameter optimization.
In validation, the model demonstrates superior performance
metrics with a 63.796% AST-weighted accuracy, coupled
with high sequence-based similarity scores (BLEU: 0.876,
ROUGE-L: 0.916), suggesting robust structural learning. The

TABLE II: Comprehensive Performance Evaluation of Models Across Different Phases
Model ASTZero-shot Training Validation Testing
Loss Acc BLEU R-L Loss Acc BLEU R-L Loss Acc BLEU R-L Loss Acc BLEU R-L
T5w/o 8.603 0.000 0.085 0.296 0.005 87.495 0.994 0.997 0.913 39.083 0.598 0.708 0.761 50.289 0.702 0.780
w/ 8.603 0.000 0.085 0.296 0.034 85.434 0.978 0.987 0.247 51.796 0.776 0.833 0.237 56.042 0.796 0.865
Flan-T5w/o 4.398 0.000 0.018 0.157 0.006 84.937 0.981 0.987 0.751 37.271 0.635 0.739 0.792 46.407 0.695 0.777
w/ 4.398 0.000 0.018 0.157 0.040 86.808 0.987 0.993 0.279 51.519 0.785 0.861 0.234 58.947 0.837 0.885
CodeT5w/o 8.096 0.000 0.002 0.096 0.012 86.729 0.989 0.994 0.519 50.995 0.725 0.801 0.608 57.211 0.763 0.828
w/ 8.096 0.000 0.002 0.096 0.005 86.003 0.992 0.995 0.175 63.796 0.876 0.916 0.220 62.879 0.840 0.898
Note: ”w/” and ”w/o”: with/without AST guidance. R-L: ROUGE-L score. Acc: AST Weighted Accuracy (%). Training epochs: 20. Training time: ˜6hrs w/o AST and ˜8hrs w AST.
minimal performance degradation in testing (62.879% accu-
racy) indicates effective mitigation of overfitting, with only
a 0.917 percentage point drop from validation to testing,
demonstrating robust generalization across the latent space of
SVRF code structures.
E.Results Summary
1)Analysis :Our experimental results demonstrate the sub-
stantial impact of AST-guided fine-tuning on SVRF code
generation. CodeT5 emerges as the clear leader, showing ex-
ceptional performance across all phases. With AST guidance,
it achieves remarkable validation metrics (63.796% accuracy,
0.876 BLEU, 0.916 ROUGE-L) and maintains strong per-
formance in testing (62.879% accuracy, 0.840 BLEU, 0.898
ROUGE-L).
Overfitting Mitigation through AST: A critical obser-
vation from our experiments is the role of AST guidance
in addressing overfitting issues. Models trained without AST
show clear signs of overfitting, with significant performance
gaps between training and testing phases (e.g., CodeT5 without
AST: 86.729% training accuracy vs 57.211% testing accuracy,
a 29.518% drop). The AST-guided approach substantially
reduces this gap (86.003% to 62.879%, a 23.124% drop)
by embedding structural knowledge that helps the model
generalize better.
This improved generalization can be attributed to several
factors:
•Structural Regularization : AST guidance acts as an
implicit regularizer by enforcing syntactic constraints
during training
•Knowledge Embedding : Instead of merely memorizing
patterns, models learn meaningful code structures through
AST representations
•Consistent Performance : Smaller validation-testing gaps
(63.796% to 62.879%) indicate more robust learning
The learning curves (Figure 3) further support this obser-
vation, showing more stable validation metrics and reduced
oscillation in the AST-guided approach, indicating better gen-
eralization capabilities without sacrificing model capacity.
2)Model Performance Comparison :Each model demon-
strates distinct characteristics in handling SVRF code gen-
eration. CodeT5, leveraging its code-specific pre-training,
shows superior performance with AST guidance, achieving
the highest scores across all metrics and phases. The model’s
validation-to-testing performance remains notably consistent,
suggesting robust generalization capabilities. Flan-T5 exhibits
Fig. 3: Code-T5: Learning Curves with/without AST
significant improvement with AST integration, showing a ˜40%
relative increase in validation accuracy and 27% in testing
accuracy. While T5 shows modest improvements with AST
guidance (accuracy increase from 50.289% to 56.042%), its
performance consistently trails behind both CodeT5 and Flan-
T5. Notably, all models demonstrate reduced validation-testing
performance gaps with AST guidance, indicating improved
generalization capabilities across different architectures.
The detailed examination of learning dynamics, including
loss convergence characteristics and metric evolution patterns,
is provided in Appendix C1 due to space constraints.
IV.APPLICATION LAYER INTEGRATION
Our methodology is further extended through practical
integration within the development environment, enabling a
copilot-like experience. The system leverages both the RAG
framework and AST-guided code synthesis to deliver context-
aware, real-time code suggestions.
A. RAG Integration
Our system enhances code generation through a RAG
framework [10] that leverages contextual information from
multiple sources. This additional context includes workspace-
specific patterns, historical implementations, and environment
configurations. The RAG engine continuously indexes and
analyzes these patterns [11], building an adaptive knowledge

base that evolves with usage. By combining this rich contex-
tual information with our fine-tuned models, the system gener-
ates more accurate and environment-aware code suggestions.
B. Editor Environment Integration
We leverage the development environment to seamlessly
connect our RAG infrastructure with the user’s workspace. The
system begins by analyzing the current code context, capturing
the immediate development environment and active coding
patterns. This real-time analysis provides crucial insights into
the developer’s current task and coding style.
The RAG knowledge base then retrieves relevant patterns
and examples from its indexed repository, considering both
historical implementations and current best practices. This
retrieved context is carefully weighted and filtered to ensure
relevance to the current development scenario.
The system combines both the immediate coding context
and the retrieved knowledge to generate predictions. This
dual-context approach enables more nuanced and accurate
suggestions, taking into account both the specific requirements
of the current task and broader coding patterns. Finally,
the system delivers these enhanced suggestions in real-time,
maintaining a natural flow within the development process
while significantly improving the quality and relevance of
generated code [12].
V.CONCLUSION AND FUTURE WORK
In this paper, We identified a significant gap in the cur-
rent LLM-based code generation approaches for specialized
domains such as SVRF code synthesis. To address this gap,
We proposed a methodology through a combination of AST
generation and supervised fine-tuning of pre-trained models.
Based on our experiments, around 40% enhancement in code
generation is observed when using an AST approach for fine-
tuning versus a standard text-based fine-tuning.
The experimental results demonstrate that AST guidance
significantly enhances both model performance and learning
efficiency. Our approach enables better generalization, evi-
denced by the minimal gap between training and validation
metrics (0.917 percentage points) and reduced overfitting
tendencies. The stable learning progression, characterized by
smooth convergence curves and consistent cross-phase perfor-
mance, indicates that structural information helps the model
develop robust understanding of SVRF patterns rather than
merely memorizing surface-level features.
However, while AST guidance significantly improves model
performance, the validation and testing accuracies (peaking
at 63.796% and 62.879% respectively with CodeT5) indicate
substantial room for improvement. This performance ceiling
can be attributed to several key factors:
•The inherent complexity of code generation extends be-
yond structural correctness, requiring deep understanding
of operation relationships, precedence, and scope
•Current dataset limitations, despite augmentation tech-
niques, may not fully capture the diverse range of possible
code structures and their variations•The AST-guided approach, while effective at enforcing
structural constraints, could benefit from additional se-
mantic validation mechanisms, such as type checking,
scope analysis, and operation compatibility verification
Furthermore, we presented an application layer integration
that combines a RAG framework with real-time editor environ-
ment analysis, offering a copilot-like experience that enhances
both the quality and relevance of generated code. This dual-
context approach not only streamlines the development process
but also reduces manual intervention and error correction.
Looking forward, several key areas deserve further explo-
ration:
•Curated data-set collection: Further effort is needed to
collect a larger dataset that is more representative of
SVRF coding, within the DRC domain as well as other
domains
•AST-Weighted Loss Function: While our current AST-
weighted metric effectively evaluates structural correct-
ness, integrating it directly into the training objective
could enhance the model’s ability to learn code structures.
By designing a differentiable AST-based loss component
that penalizes structural mismatches during training, we
could guide the model to develop better internal represen-
tations of code hierarchies. This could potentially address
the current disparity between traditional cross-entropy
loss optimization and AST-weighted evaluation metrics,
leading to more structurally-coherent code generation.
•Enhanced model-tuning methodology: Exploring alter-
native fine-tuning strategies—such as leveraging Graph
Neural Networks to treat ASTs as graphs—may further
improve model performance and generalization.
•Deeper Application integration: Utilizing the developed
model within a larger coding infrastructure with clear and
defined features paves the way for a copilot experience,
and provides clear value to end users
However, it’s important to note that the code, tools, and
the dataset utilized in this research were developed using
an internal proprietary language and are intrinsically tied
to sensitive internal information. Consequently, to maintain
confidentiality and protect proprietary assets, these materials
are not planned for public open-source release.
This paper provides comprehensive implementation details
in Section II, including the AST guided approach (Section
II-A), model architectures (Section II-B), evaluation metrics
(Section III-C, Appendix B), and experimental setup (Section
III) to enable reproducibility. Implementation questions can be
directed to the corresponding author.
In conclusion, we offer a methodology to enhance how
SVRF code is developed using LLM infrastructure, which
paves the way for fast turn-around-time and higher code
quality.

APPENDIX
A.AST Mapping Details
For illustration purpose, we use an analogous simplified
syntax that maintains the structural essence of SVRF while
preserving confidentiality. Note that the following examples
use representative commands and structures that parallel actual
SVRF syntax without revealing proprietary details.
Consider this basic spacing rule:
SPACE_CMD METAL1 METAL2 >= 0.5 READ ALL {
REPORT "Spacing violation detected"
}
This rule is decomposed into the following AST structure:
<COMMAND>
<NAME> SPACE_CMD </NAME>
<LAYERS>
<LAYER1> METAL1 </LAYER1>
<LAYER2> METAL2 </LAYER2>
</LAYERS>
<CONDITION> >= 0.5 </CONDITION>
<OPTIONS>
<MODE> READ ALL </MODE>
<REPORT>
"Spacing violation detected"
</REPORT>
</OPTIONS>
</COMMAND>
This hierarchical representation, while using simplified anal-
ogous commands, demonstrates the model’s comprehensive
capabilities in code analysis and generation. The model ex-
hibits structural awareness by understanding command compo-
nents and their relationships, implements parameter validation
to ensure valid numerical values and operators, maintains
proper configuration through options validation, and structures
appropriate violation reporting through systematic error han-
dling.
Note: The commands ( SPACE_CMD ,WIDTH_CMD , etc.)
and their syntax are simplified representations that parallel
the structure of actual SVRF commands while maintaining
confidentiality of proprietary syntax.
B.Extended Metrics Details
1) Traditional Metrics:
•Loss Score : Cross-entropy loss measuring the model’s
prediction accuracy at the token level. Lower values
indicate better performance, with our models typically
ranging from 8.196 (poor) to lower values after fine-
tuning.
•BLEU Score : Bilingual Evaluation Understudy score
evaluating the generated code’s similarity to the reference
implementation. This metric is particularly useful for
assessing partial correctness when exact matches aren’t
achieved.
BLEU =BP·exp NX
n=1wnlogpn!
(1)
where BP is the brevity penalty, wnare weights, and pnis the
n-gram precision.•ROUGE-L Score : Longest Common Subsequence
(LCS)-based metric measuring the fluency and structural
similarity between generated and reference code.
ROUGE −L=2×LCS (X, Y )
length (X) +length (Y)(2)
where XandYare the reference and generated sequences
respectively.
2) AST-Weighted Scoring: Given SVRF’s unique properties
as a non-sequential language where command ordering is
flexible but layer ordering is critical, we introduce a novel
weighted scoring mechanism:
AST -Weighted Accuracy =1
NNX
i=1(w1·cacci
+w2·oacci+w3·lacci)(3)
Where:
•cacci: Accuracy of command name and structure
•oacci: Correctness of command options and parameters
•lacci: Accuracy of layer ordering and relationships
•w1, w2, w3: Weighting factors for each component. These were
determined through various experiments along with awareness
of the SVRF grammar components priority. These experiments
focused on the conceptual ”correctness” of the SVRF code from
different angels for example: using the correct options for a
specific command and the layer ordering.
•N: Number of examples in the evaluation set
This weighted approach accounts for SVRF’s specific char-
acteristics:
•Non-sequential Nature: Command ordering flexibility
•Layer Significance: Critical importance of layer ordering
•Option Flexibility: Variable ordering of command options
•Structural Validity: Emphasis on correct command struc-
ture
The combination of traditional metrics and our AST-
weighted scoring provides a comprehensive evaluation frame-
work that captures both general code generation quality and
SVRF-specific structural requirements. This dual approach
ensures that our assessment considers both syntactic accuracy
and semantic correctness in the context of SVRF’s unique
characteristics.
C.Extended Results Analysis Details
1)Learning Dynamics Analysis :Figure 3 presents the
learning dynamics of CodeT5 with and without AST guidance
across four key metrics: loss, accuracy, BLEU, and ROUGE-L
scores. The curves reveal several important patterns:
Loss Convergence: The AST-guided model exhibits su-
perior loss reduction characteristics throughout the training
process. It achieves faster initial convergence in both training
and validation phases, demonstrating the effectiveness of struc-
tural guidance in accelerating learning. The learning trajectory
remains notably stable with minimal fluctuations, contrasting
with the more erratic behavior observed in the baseline model.
This stability is further emphasized by the significantly lower
final validation loss (0.175 compared to 0.519), strongly

indicating better generalization capabilities. The consistent and
proportional gap maintained between training and validation
losses suggests the model achieves an appropriate balance in
its capacity, neither underfitting nor overfitting the training
data, while effectively leveraging the structural information
provided by AST guidance.
AST-Weighted Accuracy Progression: The generation
quality metrics demonstrate consistent improvements with
AST guidance. While traditional metrics like BLEU and
ROUGE-L approach near-perfect values during training (im-
proving from 0.725 to 0.876 and 0.801 to 0.916 respectively),
the AST-weighted accuracy maintains a more conservative
measure, reflecting the structural complexity of the code.
This discrepancy is particularly evident in cases where text-
based metrics might suggest high similarity despite significant
semantic differences. Consider this example in figure 4, despite
differing by only a single opening parenthesis and a parameter
name, these expressions have fundamentally different semantic
meanings. The background red represents the target code and
the background green represents the predicted code.
Fig. 4: Text-to-Text Comparison: Failure Example
Text-based metrics would show high similarity scores due
to the extensive token overlap, but the AST-weighted accuracy
correctly penalizes this generation as structurally incorrect.
The missing opening parenthesis fundamentally changes the
operator precedence: in the correct version, the NOT operation
is applied to the entire expression, while in the generated
version, the scope of NOT is ambiguous and would lead to
syntax error. This single-character difference, which might
appear minor in text-based comparisons, results in com-
pletely different AST and, consequently, different semantic
meanings. The validation curves exhibit remarkable stability
under AST guidance, suggesting more reliable and consistent
code generation capabilities that better capture such crucial
structural nuances. While both approaches achieve comparable
final training performance, their learning trajectories differ
markedly, with AST-guided training showing more system-
atic and controlled progression toward optimal performance,
indicating better structural understanding of the code genera-
tion task. This is particularly evident in CodeT5’s validation
performance, where AST guidance improves accuracy from
50.995% to 63.796%, maintaining this advantage through
testing (57.211% to 62.879%). These improvements, while
numerically smaller than the near-perfect BLEU and ROUGE-
L scores, better reflect the model’s true capability in generating
structurally valid and semantically correct code.
For future work, we will construct more detailed examples
entailing the failures of text-based learning over AST-based
learning.
BLEU and ROUGE-L Evolution: The generation quality
metrics reveal consistent and substantial improvement patterns
with AST guidance. The approach demonstrates acceleratedimprovement in both BLEU and ROUGE-L scores, indicating
more efficient learning of code generation patterns. Validation
performance reaches notably higher plateau levels under AST
guidance, with BLEU scores improving from 0.725 to 0.876
and ROUGE-L scores increasing from 0.801 to 0.916. The
validation curves maintain remarkable stability with AST
guidance, suggesting more reliable and consistent code gener-
ation capabilities. While both approaches ultimately achieve
comparable training performance, their learning trajectories
differ significantly, with AST-guided training exhibiting a
more systematic and controlled progression toward optimal
performance, indicating enhanced structural understanding of
the code generation task.
Computational Considerations: The integration of AST
structures introduces additional computational overhead in
both training and inference phases. While the original SVRF
code might be relatively concise (e.g., a single-line command),
its AST representation significantly expands the token count
due to the explicit structural markup. For example, a simple
spacing rule of approximately 10 tokens expands to over
30 tokens in its AST form, including structural tags and
hierarchical relationships. This expansion necessitates larger
maximum sequence lengths during training and inference,
requiring 1024 tokens for AST-guided generation compared to
512 tokens used in basic text-based generation. Consequently,
training time increases significantly with longer sequences,
and memory requirements grow to accommodate the ex-
panded representations. In our implementation, training on an
NVIDIA H100 NVL GPU with 95.8GB memory, the AST-
guided approach required approximately 8 hours of training
time compared to 6 hours for basic text-based fine-tuning, rep-
resenting a 33% increase in computational time. This extended
training time is accompanied by approximately 40% higher
memory utilization. However, this computational overhead is
offset by the improved model performance and reduced need
for extensive data augmentation, ultimately providing a more
efficient path to robust SVRF code generation.
2)Relative Accuracy Improvements :To better understand
the impact of AST guidance, we analyzed the relative improve-
ment in accuracy across different model architectures (Figure
5). This analysis reveals several interesting patterns in how
different models respond to AST guidance during validation
and testing phases.
FlanT5 demonstrates the most substantial relative improve-
ments, with a 38.2% increase in validation accuracy and a
27.0% increase in testing accuracy. This marked improvement
suggests that FlanT5’s pre-training approach makes it particu-
larly receptive to structural guidance. The consistent improve-
ment across both validation and testing phases (difference of
11.2 percentage points) also indicates robust generalization of
the learned structural patterns.
T5 shows the second-highest relative improvement in val-
idation (32.5%), but this gain diminishes significantly during
testing (11.4%). This substantial drop-off (21.1 percentage
points) between validation and testing improvements suggests
that while T5 can learn structural patterns effectively, it may

be more prone to overfitting when compared to FlanT5.
CodeT5, despite achieving the highest absolute accuracy
scores, shows more modest relative improvements: 25.1% in
validation and 9.9% in testing. This smaller relative gain
can be attributed to CodeT5’s already strong baseline perfor-
mance, particularly its inherent understanding of code struc-
tures. However, the consistent improvement across phases
(difference of 15.2 percentage points) demonstrates that AST
guidance still provides meaningful benefits even for code-
specialized models.
These patterns suggest that while all models benefit from
AST guidance, the magnitude of improvement varies based on
the model’s architectural strengths and pre-training approach.
The more consistent validation-to-testing improvement ratios
in FlanT5 and CodeT5 indicate that these architectures may be
better suited for maintaining structural learning across different
evaluation contexts.
Fig. 5: Relative Improvements Comparison with/without AST
REFERENCES
[1] T. B. Brown, B. Mann, N. Ryder, M. Subbiah, J. Kaplan, P. Dhariwal,
A. Neelakantan, P. Shyam, G. Sastry, A. Askell, S. Agarwal, A. Herbert-
V oss, G. Krueger, T. Henighan, R. Child, A. Ramesh, D. M. Ziegler,
J. Wu, C. Winter, C. Hesse, M. Chen, E. Sigler, M. Litwin, S. Gray,
B. Chess, J. Clark, C. Berner, S. McCandlish, A. Radford, I. Sutskever,
and D. Amodei, “Language models are few-shot learners,” ArXiv , vol.
abs/2005.14165, 2020.
[2] Z. Zheng, K. Ning, Y . Wang, J. Zhang, D. Zheng, M. Ye, and J. Chen,
“A survey of large language models for code: Evolution, benchmarking,
and future trends,” ArXiv , vol. abs/2311.10372, 2023.
[3] T. Parr, The Definitive ANTLR 4 Reference . Pragmatic Bookshelf, 2013.
[4] A. V . Aho, M. S. Lam, R. Sethi, and J. D. Ullman, Compilers: Principles,
Techniques, and Tools , 2nd ed. Pearson Education, 2007.
[5] I. D. Baxter and C. Pidgeon, “Ast-based program transformation for
enhanced program understanding,” IEEE International Workshop on
Source Code Analysis and Manipulation , pp. 26–35, 2004.
[6] Y . Wang, W. Wang, S. Joty, and S. C. Hoi, “Codet5: Identifier-aware
unified pre-trained encoder-decoder models for code understanding and
generation,” in EMNLP , 2021, pp. 8696–8708.
[7] C. Raffel, N. Shazeer, A. Roberts, K. Lee, S. Narang, M. Matena,
Y . Zhou, W. Li, and P. J. Liu, “Exploring the limits of transfer learning
with a unified text-to-text transformer,” Journal of Machine Learning
Research , vol. 21, no. 140, pp. 1–67, 2020.
[8] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez,
Ł. Kaiser, and I. Polosukhin, “Attention is all you need,” in Advances
in Neural Information Processing Systems , 2017, pp. 5998–6008.
[9] H. W. Chung, L. Hou, S. Longpre, B. Zoph, Y . Tay, W. Fedus, E. Li,
X. Wang, M. Dehghani, S. Brahma et al. , “Scaling instruction-finetuned
language models,” arXiv preprint arXiv:2210.11416 , 2022.[10] P. Lewis, E. Perez, A. Piktus, F. Petroni, V . Karpukhin, N. Goyal,
H. K ¨uttler, M. Lewis, W.-t. Yih, T. Rockt ¨aschel et al. , “Retrieval-
augmented generation for knowledge-intensive nlp tasks,” Advances in
Neural Information Processing Systems , vol. 33, pp. 9459–9474, 2020.
[11] R. Krishna, C. J. Maddison, D. Tran et al. , “Rag: A semi-supervised
pattern-based learning approach to adaptive code generation,” in Pro-
ceedings of the 44th International Conference on Software Engineering .
IEEE, 2022, pp. 1123–1134.
[12] S. Weigelt and W. F. Tichy, “Workflow-based software development:
Models, methods, and tools,” in Proceedings of the 42nd International
Conference on Software Engineering: Companion Proceedings . ACM,
2020, pp. 271–274.