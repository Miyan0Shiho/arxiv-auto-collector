# eARCO: Efficient Automated Root Cause Analysis with Prompt Optimization

**Authors**: Drishti Goel, Raghav Magazine, Supriyo Ghosh, Akshay Nambi, Prathamesh Deshpande, Xuchao Zhang, Chetan Bansal, Saravan Rajmohan

**Published**: 2025-04-15 08:10:32

**PDF URL**: [http://arxiv.org/pdf/2504.11505v1](http://arxiv.org/pdf/2504.11505v1)

## Abstract
Root cause analysis (RCA) for incidents in large-scale cloud systems is a
complex, knowledge-intensive task that often requires significant manual effort
from on-call engineers (OCEs). Improving RCA is vital for accelerating the
incident resolution process and reducing service downtime and manual efforts.
Recent advancements in Large-Language Models (LLMs) have proven to be effective
in solving different stages of the incident management lifecycle including RCA.
However, existing LLM-based RCA recommendations typically leverage default
finetuning or retrieval augmented generation (RAG) methods with static,
manually designed prompts, which lead to sub-optimal recommendations. In this
work, we leverage 'PromptWizard', a state-of-the-art prompt optimization
technique, to automatically identify the best optimized prompt instruction that
is combined with semantically similar historical examples for querying
underlying LLMs during inference. Moreover, by utilizing more than 180K
historical incident data from Microsoft, we developed cost-effective finetuned
small language models (SLMs) for RCA recommendation generation and demonstrate
the power of prompt optimization on such domain-adapted models. Our extensive
experimental results show that prompt optimization can improve the accuracy of
RCA recommendations by 21% and 13% on 3K test incidents over RAG-based LLMs and
finetuned SLMs, respectively. Lastly, our human evaluation with incident owners
have demonstrated the efficacy of prompt optimization on RCA recommendation
tasks. These findings underscore the advantages of incorporating prompt
optimization into AI for Operations (AIOps) systems, delivering substantial
gains without increasing computational overhead.

## Full Text


<!-- PDF content starts -->

eARCO: Efficient Automated Root Cause Analysis with Prompt
Optimization
Drishti Goel, Raghav Magazine, Supriyo Ghosh, Akshay Nambi, Prathamesh Deshpande, Xuchao
Zhang, Chetan Bansal, Saravan Rajmohan
Microsoft
ABSTRACT
Root cause analysis (RCA) for incidents in large-scale cloud sys-
tems is a complex, knowledge-intensive task that often requires
significant manual effort from on-call engineers (OCEs). Improving
RCA is vital for accelerating the incident resolution process and re-
ducing service downtime and manual efforts. Recent advancements
in Large-Language Models (LLMs) have proven to be effective in
solving different stages of the incident management lifecycle in-
cluding RCA. However, existing LLM-based RCA recommendations
typically leverage default finetuning or retrieval augmented genera-
tion (RAG) methods with static, manually designed prompts, which
lead to sub-optimal recommendations. In this work, we leverage
‘PromptWizard’, a state-of-the-art prompt optimization technique,
to automatically identify the best optimized prompt instruction
that is combined with semantically similar historical examples for
querying underlying LLMs during inference. Moreover, by utiliz-
ing more than 180K historical incident data from Microsoft, we
developed cost-effective finetuned small language models (SLMs)
for RCA recommendation generation and demonstrate the power
of prompt optimization on such domain-adapted models. Our ex-
tensive experimental results show that prompt optimization can
improve the accuracy of RCA recommendations by 21% and 13%
on 3K test incidents over RAG-based LLMs and finetuned SLMs,
respectively. Lastly, our human evaluation with incident owners
have demonstrated the efficacy of prompt optimization on RCA rec-
ommendation tasks. These findings underscore the advantages of
incorporating prompt optimization into AI for Operations (AIOps)
systems, delivering substantial gains without increasing computa-
tional overhead.
KEYWORDS
Root cause analysis, Incident Management, Prompt Optimization,
Domain Adaptation
1 INTRODUCTION
Over the past decade, large-scale cloud services have become essen-
tial for deploying and maintaining complex interdependent systems.
Despite significant reliability efforts, these services still experience
production incidents like unplanned outages or performance degra-
dations, leading to customer dissatisfaction, revenue loss, and de-
clining trust. The current incident diagnosis process heavily relies
on manual efforts from on-call engineers (OCEs), resulting in pro-
ductivity loss. Additionally, limited expertise or domain knowledge
among OCEs can lead to sub-optimal actions, delaying service re-
covery.
Root cause analysis (RCA) is a critical and time-consuming step
in the incident management lifecycle, requiring deep domain knowl-
edge and back-and-forth communication among OCEs. OCEs referto several information source such as troubleshooting guides, past
similar incidents, service properties and current incident metadata
to identify the key reasons behind service disruptions. Accurately
identifying the root cause early can significantly reduce time-to-
mitigate (TTM) by enabling faster execution of mitigation steps.
Thus, automating RCA at an early stage could accelerate incident
resolution, reduce service downtime, and minimize manual efforts
by OCEs.
Recent advancements in LLMs have proven to be effective for
solving several problems in the incident management lifecycle, rang-
ing from detection [ 13,29] to triaging [ 4,7] to automated query
recommendations [ 20] to problem categorization [ 10] to root cause
generation [ 3,16,32]. For root cause generation, [ 3] first propose
to finetune a GPT-3 model with initial incident metadata (e.g., title,
initial summary and owning service name) as input and the cor-
responding root cause as output in a supervised setting. However,
finetuning LLMs is computationally expensive and maintenance-
heavy. [ 32] proposed a retrieval-augmented generation (RAG) based
in-context learning (ICL) solution that dynamically retrieves simi-
lar historical incidents during inference and prompts a pre-trained
LLM with predefined, manually designed instructions. While this
RAG-based ICL method has demonstrated potential, significant chal-
lenges arise when scaling it for RCA: (1) Static Prompts: Manually
defined prompts lack the flexibility to adapt as tasks and models
evolve, making continual updates labor-intensive. (2) Sub-optimal
Guidance: Manually crafted prompts may not fully leverage the
LLM’s potential, leading to sub-optimal RCA recommendations
without automated optimization. (3) Cost and Scalability: While
LLMs like GPT-4 are powerful, they are costly to deploy at scale.
This raises the need for more cost-effective solutions, such as fine-
tuning smaller language models (SLMs), without compromising on
RCA accuracy.
In this paper, we introduce a novel framework for efficient
automated rootcause analysis with prompt optimization ( eARCO ).
Our work addresses the following key research questions: (1) RQ1 :
Can automatically optimized prompt instructions outperform man-
ually designed static instructions in improving the accuracy and
quality of RCA recommendations? (2) RQ2 : Does the combina-
tion of optimized prompt instructions and strategically selected
in-context examples deliver superior RCA performance compared
to using prompts or examples alone? and (3) RQ3 : Can SLMs, when
paired with optimized prompt instructions, provide a cost-effective
alternative to querying expensive LLMs while maintaining compa-
rable RCA performance?
To address these research questions, we leverage state-of-the-art
prompt optimization techniques to automatically generate opti-
mized prompt instructions tailored for the RCA task. The process
begins by taking a task description and a few training examples
1arXiv:2504.11505v1  [cs.SE]  15 Apr 2025

of incidents with their root causes, after which optimized prompts
are derived. We then select in-context learning (ICL) examples that
closely match the current incident.
Specifically, we leverage PromptWizard [ 2], a discrete prompt op-
timization approach that evolves and adapts its prompts in real-time.
PromptWizard has shown superior performance across various NLP
tasks by generating, critiquing, and refining prompts through an
iterative feedback loop. This "critic-and-synthesize" process con-
tinuously improves prompts without requiring additional model
training. By querying a pre-trained LLM fewer than 100 times, the
method remains computationally efficient while producing highly
optimized prompt instructions for RCA. This automated prompt
optimization eliminates the need for manual prompt engineering,
ensuring that as pre-trained LLMs evolve, the system consistently
provides optimal instructions for improving model performance.
Once the optimized instruction is obtained, we develop a retrieval-
augmented generation (RAG) system inspired by Zhang et al. [ 32]
to dynamically retrieve the top-K semantically similar historical in-
cidents and their root causes. By incorporating these past incidents
and their root causes, the system adds valuable domain-specific
context, enhancing the model’s ability to generate accurate RCA rec-
ommendations. Our extensive experimental results, tested on 2,900
real-world incidents from Microsoft, demonstrate that combining
optimized prompt instructions with semantically similar in-context
examples significantly improves the quality and accuracy of RCA
recommendations.
Despite the strong performance of large language models (LLMs)
for RCA tasks, their use in production with long context lengths
is prohibitively expensive. To address this, we fine-tune smaller
models like Phi-3-Mini, Phi-3-Medium [ 1], and Phi-3.5-Mini using
180K historical incidents from Microsoft, incorporating metadata
and root causes from over 1,000 services. Combining these fine-
tuned SLMs with optimized prompts from PromptWizard improves
RCA accuracy by 13% compared to using incident context alone.
This demonstrates that fine-tuned SLMs with optimized prompts
provide a cost-effective alternative to LLMs, reducing inference
costs and complexity.
This paper makes three key contributions:
•We introduce eARCO , a novel framework that integrates
optimized prompts from PromptWizard with top-K semanti-
cally similar historical incidents to generate accurate RCA
recommendations, dynamically adapting to task-specific needs.
•We develop a cost-effective RCA solution by fine-tuning
SLMs on 180K historical incidents from over 1,000 services
at Microsoft. These SLMs, when queried with optimized
prompts, offer a cost-effective alternative to expensive LLMs
without compromising performance.
•GPT-4-based evaluations show that prompt optimization im-
proves RCA accuracy by 21% for LLMs and 13% for SLMs.
Human evaluations with domain experts confirm the practi-
cal benefits, reporting enhanced RCA task performance.
2 BACKGROUND
In this section, we begin by introducing the incident management
domain and providing background on root cause analysis (RCA).Next, we discuss the potential advancements in incident manage-
ment enabled by recent developments in large language models
(LLMs). This is followed by outlining our key research questions.
Lastly, we detail the data preparation strategy for the RCA task,
which includes steps for data curation, cleaning, and summariza-
tion.
2.1 Incident Root Cause Analysis
Despite significant reliability efforts, large-scale cloud services in-
evitably encounter production incidents or outages, which can
result in significant customer impact and financial loss. On-call
engineers (OCEs) need extensive domain knowledge and expend
considerable manual effort to diagnose and resolve these incidents.
The incident lifecycle typically involves four stages:
•Detection: Incidents are detected either by external/internal
service users or through automated monitoring systems set
up to track service health and performance.
•Triaging: Once detected, incidents are routed to the appro-
priate service teams or OCEs based on the incident properties
and team expertise.
•Root cause analysis: OCEs engage in rounds of commu-
nication, analyzing logs, performance metrics, service de-
pendencies, and troubleshooting guides to identify the root
cause.
•Mitigation: Mitigation actions are taken based on the iden-
tified root cause to resolve the incident and restore service
functionality.
Root cause analysis is particularly challenging, requiring significant
manual effort and domain knowledge. Incidents may arise from var-
ious sources, such as code bugs, configuration errors, dependency
failures, or hardware issues. Missteps in RCA can delay service
recovery and exacerbate customer impact. Automating the identi-
fication of potential root causes early in the incident lifecycle can
guide OCEs toward the correct resolution path, reducing overall
time-to-mitigate (TTM) and minimizing customer disruption.
2.2 Promise of LLMs in Incident Management
The rapid advancements in LLMs have led to exceptional perfor-
mance in a wide variety of natural language tasks, ranging from
summarization and translation to question-answering and code
completion. Furthermore, recent advancements in domain adap-
tation using finetuning and few-shot learning have laid the foun-
dation for solving problems in different stages of the incident life-
cycle. LLMs have shown great potential in automating detection,
problem categorization and accurate triaging of incidents which
can alleviate the load and manual efforts from engineers. More-
over, recent studies [ 3,16,32] have demonstrated the usefulness of
LLMs in automatically identifying the root causes of an incident by
leveraging initial incident metadata and similar historical incident
properties, either by finetuning LLMs or using in-context (ICL)
learning framework. However, the use of static, manually designed
prompt instructions can result in sub-optimal performance, and
running LLMs with long context lengths for ICL remains costly.
To address these limitations, we propose using automatically opti-
mized prompt instructions, coupled with cost-effective fine-tuned
2

smaller language models (SLMs), to enhance root-cause generation
accuracy while minimizing operational costs.
2.3 Research Questions
Our goal is to address the following three main research questions
in this study:
•Can optimized prompt instruction improve the qual-
ity of RCA recommendation? : Manually designing high-
quality static prompts is challenging and often leads to sub-
optimal results. This requires significant domain knowledge
and is mostly trial-and-error. Additionally, performance may
degrade as LLM models or their versions evolve. We aim
to leverage the state-of-the-art prompt optimization tech-
nique (‘PromptWizard’) to automatically identify the best
prompt instructions and demonstrate their superiority over
manually crafted prompts.
•What role do in-context examples play in conjunc-
tion with optimized instructions? The effectiveness of
LLMs in reasoning tasks largely depends on the quality of in-
context examples included in the prompt. We will evaluate
the performance of static in-context examples identified by
‘PromptWizard’ against dynamically retrieved semantically
similar examples.
•How do cost-effective finetuned SLMs perform with
optimized prompts? To reduce reliance on expensive LLMs
with large context lengths, we fine-tune SLMs using histor-
ical incident data. We will assess the performance of opti-
mized instructions generated by ‘PromptWizard’ on these
fine-tuned SLMs.
2.4 Data Preparation
To address the research questions on the RCA task, we curated
a comprehensive incident dataset from Microsoft. The dataset in-
cludes over 180K historical incidents and their corresponding root
causes. We performed data cleaning and pre-processing to ensure
consistency, removed any incomplete or irrelevant records, and
summarized the key incident properties.
2.4.1 Data Collection. Each service team at Microsoft logs their
incident data in the internal incident management (IcM) portal. We
curated a dataset of approximately 180K high-severity historical
incidents from January 2022 to June 2024 from this portal. For
each incident, we collected key metadata, including the title, initial
summary (written at the time of the incident), owning service name,
and the ground truth root cause identified and documented by the
on-call engineers (OCEs). This data serves as the basis for both fine-
tuning models and evaluating root cause analysis performance.
2.4.2 Data Cleaning and Summarization. The raw summaries and
root causes of incidents collected from the IcM portal often contain
significant amounts of extraneous information, such as HTML tags,
images, stack traces, and code snippets. This noise can adversely
impact model performance during both training and testing for
several reasons: (a) the effectiveness of a fine-tuned model relies
heavily on high-quality training data; noise diminishes the model’s
ability to learn effectively; (b) excessive noise can impair the rea-
soning capabilities of both base and finetuned models, potentiallyleading to hallucinations; and (c) accurate ground truth for root
causes is essential for evaluating model effectiveness.
To mitigate such interference, we employed a two-stage data
cleaning strategy. First, we locally processed the data to remove
irrelevant HTML tags, stack traces, and image tags. In the second
stage, we utilized the GPT-3.5-turbo1model to summarize the root
cause and summary fields, guided by a prompt as proposed by
Zhang et al. [ 32]. This summarization step also enabled the LLM
to identify and filter out noisy or non-informative root causes,
enhancing the overall quality and stability of the dataset.
3 PROMPT OPTIMIZATION FOR RCA
We now provide the details of our eARCO framework that pri-
marily consists of two components: Prompt instruction optimization
andIn-Context example selection . We begin by explaining each of
these two components while emphasizing their respective roles in
optimizing the quality of RCA recommendations. We then explain
the overall architecture of the eARCO system by integrating these
two components together.
3.1 Prompt Optimization
LLM responses are highly sensitive to prompt instructions, making
it crucial to provide the right instructions for generating high-
quality RCA recommendations. To achieve this, we leverage a
state-of-the-art prompt optimization technique called PromptWiz-
ard(PW) [ 2]. PW is a discrete optimization approach that refines
manually designed initial prompts using input-output pairs from
the training data. The optimization process involves four key steps:
Mutate, Score, Critique, and Synthesize. Each step employs LLMs to
execute pre-defined actions through specialized prompt templates.
These steps are run iteratively to progressively optimize both the
prompt instructions and the selection of in-context examples.
Prompt Instruction Tuning : In the Mutate stage, Prompt
Wizard (PW) takes the initial prompt instructions and task de-
scription, applying predefined thinking styles to generate prompt
variations in a single LLM call. In the Score stage, these mutated
prompts are evaluated on a diverse batch of training samples, with
each prompt assigned a score based on performance. The best-
performing prompt proceeds to the Critique stage, where it re-
ceives targeted feedback regarding its strengths and weaknesses.
This feedback is used in the Synthesize stage to further refine and
improve the prompt. The refined prompt is then fed back into the
Mutate stage, continuing iteratively until either a predefined per-
formance threshold is met or the maximum number of iterations is
reached. This feedback-driven loop effectively balances exploration
and exploitation, continuously enhancing the prompt.
In-Context Examples Tuning : The inclusion of in-context ex-
amples (ICL) is optional for users. If selected, the optimized prompt
from the previous phase is further tuned along with ICL examples.
PW begins by selecting a random set of 25 diverse examples, cover-
ing both positive and negative instances. These examples are opti-
mized in two phases: First, the Critique stage provides feedback on
how well the examples complement the prompt instructions, sug-
gesting improvements. Then, in the Synthesize stage, this feedback
is used to refine the instructions and example selection. Second,
1https://platform.openai.com/docs/models/gpt-3-5-turbo
3

Optimized prompt instruction for RCA.
You will be given a detailed description of an incident involving a system or service, including pertinent logs, error messages, and
other relevant information. Your objective is to methodically analyze the incident and identify the root cause. Follow these steps:
1.**Contextual Information**: Identify the service, relevant timestamps, environment, and key stakeholders involved. Mention
regional specifics where applicable.
2.**Categorization**: Categorize the type of incident (e.g., compute issues, storage conflicts, network anomalies).
3.**Identify Symptoms**: List all symptoms and error messages mentioned in the incident description.
4.**Detailed Historical Review**:
- Reflect on similar incidents and any historical data that might provide insights.
- Explicitly assess any past configuration changes or known historical script configurations related to the issue.
5.**Environmental Variables and Changes**:
- Identify and evaluate recent environmental changes, including recent configuration updates or external factors.
- Refer to specific timestamps leading up to and during the incident for environmental and systemic changes.
6.**Analyze Patterns and Logs**:
- Examine logs and error messages for recurring patterns.
- Cross-verify error logs against configuration settings and recent system changes.
- Look for specific script logs or monitor configurations and validate against system norms.
7.**Root Cause Analysis**:
- Synthesize findings from logs, historical data, and environmental variables.
- Clearly delineate between potential and confirmed root causes.
- Loop back to compare symptoms with broader historical and configuration data to ensure comprehensive scrutiny.
8.**Conclusion**: Clearly present the final root cause(s) wrapped between <𝐴𝑁𝑆 _𝑆𝑇𝐴𝑅𝑇 >and<𝐴𝑁𝑆 _𝐸𝑁𝐷 >tags.
Be thorough and evidence-based in your analysis, while eliminating any personal biases. Base your findings entirely on the provided
details to ensure accuracy.
Figure 1: Optimized prompt instruction identified by PromptWizard.
PW identifies the optimal mix of positive, negative, and synthesized
examples through this iterative critique-synthesis loop, adapting
both instructions and examples to the task at hand.
To further enhance performance, the prompt enters a Reason-
ingstage, where chain-of-thought (CoT) reasoning is incorporated
into the examples, followed by a Validation stage to verify the
correctness of the examples and reasoning, preventing hallucina-
tion and removing errors. PW also introduces a Task Intent and
anExpert Persona to the optimized prompt. The Task Intent helps
the LLM maintain relevance, while the Expert Persona ensures
consistency with domain expertise.
The final optimized prompt consists of four main components:
a problem description, optimized instructions, static and diverse
in-context examples, and task intent with an expert persona. Op-
tionally, an answering format section can be added to specify how
the LLM should structure its response for downstream tasks. Impor-
tantly, this prompt optimization is a one-time process; the optimized
prompt is then reused at the inference stage for all incident analy-
ses.
Adapting to RCA Task : For our RCA task, we begin by select-
ing 25 to 30 diverse historical incident examples from our training
corpus. These examples are chosen to ensure coverage across vari-
ous root-cause categories. We choose a manually designed prompt
instruction from Zhang et. al. [32], as the initial input to PromptWiz-
ard. PW then generates an optimized prompt instruction for RCA
task and identifies a set of diverse in-context examples. The ExpertPersona guides the model to take on the role of an On-Call Engineer
(OCE) with advanced analytical and reasoning skills, tasked with
identifying the root cause of cloud system incidents. To improve the
performance of PW optimization modules, we tune the following
hyper-parameters and its values are described in Section 5.1:
•mutate_refine_iterations : Number of iterations for conduct-
ing rounds of mutation of task description and refinement
of instructions.
•mutation_rounds : Number of rounds of mutation to be per-
formed when generating different styles.
•refine_task_eg_iterations : Number of iterations for refining
task description and in-context examples.
•questions_batch_size : Number of questions to be asked to
LLM in a single batch during training.
•min_correct_count : Number of batches of questions to be
correctly answered, for a prompt to be considered for the
next steps.
•few_shot_count : Number of in-context examples in the final
prompt.
These parameters govern the extent of optimization and the final
structure of the prompt. The optimized prompt instructions identi-
fied by PromptWizard for the RCA task, using the GPT-4o model,
are illustrated in Figure 1. A crucial aspect of PW’s optimization
process is the iterative refinement of both the instruction and the in-
context examples. During this stage, knowledge from the examples
4

informs the refinement of the instructions, and vice versa. An exam-
ple of this in Figure 1 includes the line: Categorization: Categorize
the type of incident (e.g., compute issues, storage conflicts, network
anomalies) . Here, the examples list is generated based on informa-
tion derived from the training data, highlighting the knowledge
transfer between the two prompt components. This back-and-forth
refinement leads to the optimal setting for both the instructions and
in-context examples. Moreover, the generated prompt is structured
in a clear, step-by-step format, guiding the model from the incident
information through critical analysis and ultimately to the root
cause. By leveraging the training data, PromptWizard produces
a task-specific prompt that is highly effective for RCA, ensuring
alignment with the problem domain.
3.2 In-context example selection
In-context examples play a vital role during LLM inference, espe-
cially for RCA task as the pre-trained LLMs do not have complete
knowledge of the incident management domain during training.
Although PW generates a set of static in-context examples which
shown superior performance on traditional NLP tasks, the context
for each incident typically varies significantly. Therefore, a set of
static examples may not be suitable for all incidents.
To dynamically understand the current incident context and
identify semantically similar historical examples, we used a tradi-
tional retrieval augmented generation (RAG) pipeline. For each of
the 180K incidents in training corpus, we first combine the title
and cleaned summarized version of initial incident summary, and
encode them into vector representations using the Sentence Trans-
former model [ 24]. All these training incidents’ metadata along
with the corresponding embedding value are then stored in a vector
database.
Once the embedding vectors are indexed, we leverage the FAISS
library [ 22] for the efficient similarity search and clustering of the
dense vectors. The FAISS library utilizes a compressed representa-
tion of the vectors, which eliminates the need to store the original
vectors in memory. While this compression may result in a mar-
ginal reduction in search precision, the key advantage lies in its
exceptional scalability. FAISS can efficiently handle billions of vec-
tors in main memory on a single server. For a given incident and its
text embedding vector during inference, FAISS then retrieves the
top-K similar incidents using the L2 (Euclidean) distance metrics.
3.3 Leveraging eARCO for RCA Generation
In this paper, we propose eARCO , a comprehensive framework
for efficient automated root cause analysis (RCA). eARCO com-
bines the Prompt Instruction Optimization andIn-context example
selection techniques to significantly enhance the performance of
LLMs in generating accurate RCA recommendations. The detailed
architecture of this framework is illustrated in fig. 2.
There are two stages of eARCO, (1) Optimized prompt instruction
generation one-time and (2) Selection of ICL examples at inference
time. Specifically, at inference time, given the current incident
metadata (i.e., title and initial summary which are available at the
time of incident creation), we generate the encoded embedding
vector using the Sentence Transformer model which is used as
an incident query vector. Using the retrieval pipeline outlined insection 3.2, eARCO then dynamically selects the top 10 semantically
similar incidents from the historical incident corpus.
The semantic similar examples are augmented with the static
optimized prompt instruction computed using PW (as outlined
in section 3). By integrating the optimized instructions ,incident
metadata , and semantically similar examples , eARCO enhances the
capabilities of LLMs to produce better quality automated RCA rec-
ommendations without additional training.
4 EARCO FOR FINETUNED SLMS
Another significant contribution of this paper is our exploration
of finetuned small language models, optimized eARCO framework.
In contrast to relying on retrieval-based methods to supply lan-
guage models with historical incidents for domain-specific tasks
like root cause analysis, fine-tuning offers a more targeted and
efficient approach. Finetuning eliminates the need for maintaining
a large retrieval corpus, large contextual information in prompts
as ICL examples, inference time computations and significantly re-
duces the risk of hallucinations. Previous efforts, of finetuning the
GPT-3 model, have demonstrated that adapting language models
for industry-specific tasks can be highly effective, but as previously
mentioned, finetuning and maintaining large language models is a
costly task. Our work addresses this by employing small language
models (SLMs) of varying sizes, which not only optimize perfor-
mance for specialized tasks but also presents a resource-efficient
solution for domain adaptation using finetuning.
Finetuning Methodology. We fine-tuned the Phi-3.5-mini, Phi-
3-mini, and Phi-3-medium models for the root cause analysis (RCA)
task using Hugging Face’s Supervised Fine-Tuning (SFT) Trainer,
an efficient framework for adapting pretrained models to domain-
specific datasets. From the incident dataset which we curated, we
allocated 10K datapoints for validation, 2,891 for testing, and uti-
lized the remainder for training (160K+). The data was split tempo-
rally, with older incidents designated for training and more recent
cases for testing. This temporal split was implemented to simulate
real-world scenarios, ensuring that the models could effectively gen-
eralize historical knowledge to newer, unseen incidents, reflecting
the applicability to evolving system behaviors. The finetuning pro-
cess was carried out over three epochs, utilizing a batch size of 64
samples on a compute cluster featuring 8 x NVIDIA Tesla V100 (672
GB RAM) and 8 x NVIDIA A100 (900 GB RAM) GPUs, depending
on availability. To enhance model performance and generalization,
we employed the AdamW optimizer, incorporating weight decay
to mitigate the risk of over-fitting. Additionally, a linear learning
rate scheduler with a warm-up phase was implemented to promote
training stability and prevent abrupt changes in learning rates that
could hinder convergence. The training duration for the models is
given in table 1.
Root Cause Generation. After finetuning the SLMs on the
incident dataset, we performed inference and evaluated the models
using a test set comprising of more recent incidents. This approach
closely simulates real-world scenarios where models must lever-
age historical knowledge to analyze and address newer, unseen
cases. By employing a temporal split—training on older incidents
and testing on more recent ones—we assess the model’s ability to
generalize across time and adapt to evolving system behaviors, a
5

Incident Information
(Title, Summary,
Owning Tenant Name) Incident
RetrieverPrompt Wizard
Problem
Description
Prompt
Instruction
Training
ExamplesIterative
Refinement
of Prompt
InstructionsSelf-generated
Reasoning &
Validation
Diverse
Example
SelectionSequential
Optimization
Instruction
Optimization
Synthetic
Example
GenerationModified Prompt
Instruction
Synthesized
ExamplesProblem Description
Task Intent + Expert
PersonaOptimized Few-Shot
with ReasoningOptimized PromptOptimized
PromptInput Final Prompt
Model
(SLM/LLM)Prompt Generation
Optimized Prompt Instructions
+
Semantically Similar Incidents
+
Incident Metadata
Query
k-similar
incidents
Data
Collection & 
CleaningIncident
SummarizationHistorical
Incidents
Retrieval Corpus
RAG PipelineRCA
Responses 
OCE TeamFigure 2: Architecture of the eARCO Framework for Efficient Root Cause Analysis (RCA)
Table 1: Training Duration: This table summarizes the total
training times for small language models utilized in the Root
Cause Analysis (RCA) task, highlighting model size and com-
putational resources employed.
Model Model Size (Parameters) Compute Type Training Time (Hours)
Phi-3-mini 3.8B 8 x A100 6.5
Phi-3.5-mini 3.8B 8 x V100 13.5
Phi-3-medium 14B 8 x V100 30
crucial requirement for dynamic, real-time environments such as
root cause analysis (RCA).
To ensure reproducibility and minimize randomness in the gen-
erated responses, we set the temperature to approximately zero
during inference. A low temperature encourages the model to pro-
duce deterministic and focused outputs by reducing variation in
token selection, which is essential when generating structured re-
sponses such as root cause explanations. Additionally, we capped
the maximum number of new tokens at 200, aligning the response
length with both the readability requirements of OCEs and the
average token length of the ground truth root cause analyses.In the next section, we describe the performance of fine-tuned
SLMs on the incident data and demonstrate how prompt instruction
optimization enhances SLMs’ ability to generate high-quality RCA
recommendations. We detail the comparative results, showing the
impact of optimized prompts on SLM accuracy and the quality of
root cause identification, further validating the effectiveness of the
optimization process.
5 EXPERIMENTAL SETUP
In this section, we explain the default configurations used to tune
the PromptWizard module, followed by explaining different ver-
sions of eARCO and baseline methods, and finally explain the eval-
uation strategy and performance metrics.
5.1 PromptWizard Configurations
As discussed earlier, PromptWizard uses several configurable pa-
rameters to balance exploration and exploitation efficiently, en-
suring that prompt optimization remains robust for RCA task. To
derive an optimized prompt, we sample 25 random input-output
pairs from the IcM dataset as training data. The following configu-
ration was used for the optimization process:
•mutate_refine_iterations : 3
6

•mutation_rounds : 3
•refine_task_eg_iterations : 3
•questions_batch_size : 5
•min_correct_count : 3
•few_shot_count : 10
5.2 Methods and Baselines
To thoroughly assess the impact of prompt optimization on both
LLMs and finetuned SLMs, we experiment the following 8 strategies:
Manual Prompt with Semantically Similar (SS) ICL exam-
ples [ 32] - (Manual-SS): The prompt, which is also proposed in
[32], is designed with three key components: Default manual In-
structions ,In-context Examples , and Incident Details . The Default
manual Instructions , includes the prompt hand-designed by a do-
main expert which has not undergone any optimization. This prompt
briefly guides the model to assume the role of an OCE tasked with
performing root-cause analysis (RCA) for Cloud incidents. For the
In-context Examples , we retrieve the top 10 similar incidents using
the RAG pipeline, as detailed in section 3.2, and provide their cor-
responding titles, summaries, owning service names, and ground
truth root causes. Finally, we incorporate the details of the current
incident, including its title, summary, and owning service name.
This method is applied to both GPT-4, GPT4o and base SLMs. We
deliberately exclude the finetuned models from this experiment, as
including the In-context examples defeats the purpose of finetuning
the models in the first place.
Optimized Prompt with Static examples- (PW-Default) : In
this configuration, we use PW Instructions andPW static Examples
along with the Incident Details in the prompt. This is an out-of-the
box usage of PromptWizard without any modifications and the
instructions and examples remain the same for all test incidents.
This setup shows how well ICL examples selected by PW generalizes
across all incidents.
Optimized Prompt with Semantic Similar examples (PW-
SS): In this configuration, we incorporate only the PW Instructions ,
excluding the PW static Examples . The ICL examples are selected
based on the semantic similarity of the incident at hand in run-time
as described previously. This serves as an ablation study to isolate
and assess the impact of the instructions themselves, allowing us
to later compare and understand the contribution of the PW static
Examples to the overall performance.
Finetuned SLM (FtSLM): This method utilizes the finetuned
SLMs with a standard, non-optimized prompt that contains the
Incident Title ,Incident Summary , and Owning Service Name in the
same format used during the fine-tuning process. This serves as
a benchmark for evaluating the performance of finetuned models
without any further prompt optimization.
Finetuned SLM with PW Inst. & Ex (FtSLM PW): In this
configuration, we augment the finetuned SLMs with both the PW
Instructions andPW static Examples , along with the Incident Details
during inference. This allows us to assess whether the optimized
instructions and static examples improve the performance of the
finetuned models.
Finetuned SLM with PW Inst.(FtSLM PW noEx.): As another
ablation experiment, we provide only the PW Instructions along with
theIncident Details , without including the PW static Examples . Thisallows us to evaluate the isolated impact of optimized instructions
on the performance of the fine-tuned models.
Base SLM with PW Inst. & Ex (BaseSLM PW): In this base-
line, we investigate the performance of the base (non-finetuned)
SLMs, when equipped with both the PW Instructions andPW static
Examples , excluding any finetuning.
Base SLM with PW Inst (BaseSLM PW noEx.).: This con-
figuration uses the base SLMs with only the PW Instructions and
Incident Details , omitting the PW static Examples . This method seeks
to compare the impact of finetuning and PW static Examples .
5.3 Evaluation Metrics
Even with ground truth root cause information, evaluating the rec-
ommendations generated by language models is a complex task.
While expert evaluations from the OCEs would provide the most ac-
curate assessments, conducting such large-scale human evaluation
is not a practical option due to time constraints of OCEs. Moreover,
traditional automatic metrics, whether lexical or semantic, often fail
to capture the nuanced, domain-specific similarities required for ef-
ficient evaluation as shown in previous studies [ 16,32]. To address
these challenges, we implement a dual evaluation strategy to assess
the similarity and accuracy between the automatically generated
recommendations and ground truth root causes: (1) an automated
assessment using GPT-4 as the judge across the entire test dataset
consisting of 2900 incidents; and (2) a small-scale human evaluation
on a representative subset of incidents.
5.3.1 Automated Evaluation Using GPT-4. In this evaluation strat-
egy, the GPT-4 model is prompted with a structured task description
that specifies it’s role as a scorer, tasked with comparing a gener-
ated string with a reference string based on a defined set of criteria.
The model assigns a score between 1 to 5, where a higher score
indicates a closer match in terms of content coverage, nuance, and
accuracy. The model also provides a justification for each score. In
addition to the reference and generated strings, we also provide the
GPT-4 model with the incident summary as contextual information,
allowing the model to evaluate the generated responses with a
clearer understanding of the incident.
5.3.2 Human Evaluation. As incident owners are domain experts
and have specific knowledge about the root cause context, we chose
a subset of recent incidents, and interviewed the respective inci-
dent owners (on-call engineers). Along with the OCEs, we also
asked other researchers (who are not incident management do-
main experts) to score two sets of incidents. Each evaluator was
asked to score the generated responses based on two key criteria:
(1) accuracy in comparison to the ground truth root cause and (2)
readability in terms verbosity, grammatical correctness and struc-
ture of the recommendations. With detailed guidelines provided to
ensure consistent scoring, each model-generated recommendation
was rated on a scale of 1 to 5 for both criteria.
6 EXPERIMENTAL RESULTS
6.1 Evaluating Performance of eARCO with
Large Models
In this section, we compare the performance of manually crafted
prompts with those optimized using PromptWizard for the RCA
7

task. Table 2 presents the results across different configurations,
with the LLM name indicating the model used for optimization and
answer generation. The table reports the performance across two
datasets: the ‘Complete Test Dataset’ and the ‘Filtered Test Dataset’.
The latter is a subset of the former, containing only those incidents
where an incident summary is available in the Incident Details .
The optimized prompt generated by PromptWizard, denoted as
PW-Default , achieves average scores of 2.07 for GPT-4 and 2.13 for
GPT-4o, outperforming the manually designed Manual-SS prompts,
which scored 2.03 and 2.07 for the same models, respectively. De-
spite using the same 10 in-context examples for all test instances,
PromptWizard’s optimization outperforms Manual-SS , which dy-
namically selects the 10 most semantically similar examples for
each test instance. This result highlights the limitations of manually
crafted prompts and the advantage of PromptWizard’s automated
optimization. Moreover, replacing the static in-context examples in
PromptWizard with semantically similar examples (PW-SS) further
improves the performance to 2.33 and 2.51 for GPT-4 and GPT-
4o, respectively, providing a massive 21% gain in accuracy over
manually designed prompts.
Table 2: GPT-4 evaluation scores for PromptWizard
Experiment Complete Test Dataset Filtered Test Dataset
GPT-4
Manual-SS 2.03 ±0.93 2.35 ±0.94
PW-Default 2.07 ±0.91 2.37 ±0.92
PW-SS 2.33 ±0.98 2.68 ±0.98
GPT-4o
Manual-SS 2.07 ±1.01 2.33 ±1.05
PW-Default 2.13 ±0.97 2.41 ±0.95
PW-SS 2.51 ±1.01 2.91 ±1.01
To assess the value of PromptWizard’s multi-step optimization
process, we performed an ablation study where prompts were evalu-
ated at intermediate stages of optimization (as shown in Table 3). Re-
sults show a consistent improvement in performance as the prompt
undergoes more stages of refinement, underscoring the importance
of the multi-step optimization employed by PromptWizard. This
confirms that each stage—mutation, scoring, critiquing, and synthe-
sizing—contributes significantly to achieving optimal performance.
Table 3: GPT-4 evaluation scores for PW prompt from GPT-
4o at various steps in the optimization process
Optimization Stage Complete Test Dataset Filtered Test Dataset
Base Prompt (Manual-SS) 2.07 ±1.01 2.33 ±1.05
After Prompt Instruction Tuning 2.10 ±0.91 2.22 ±0.95
After In-Context Examples Tuning 2.22 ±1.05 2.30 ±0.91
PW Final Prompt (PW-SS) 2.51 ±1.01 2.91 ±1.01Table 4: Results for base and fine-tuned SLMs
Experiment Filtered Test Dataset Complete Test Dataset
Phi-3.5-mini-128k-instruct
FtSLM 2.09 ±0.90 1.79 ±0.87
FtSLM PW noEx. 2.13 ±0.87 1.90 ±0.87
FtSLM PW 2.37 ±0.79 2.01 ±0.84
BaseSLM PW 2.26 ±0.71 1.93 ±0.77
BaseSLM PW noEx. 2.14 ±0.61 1.82 ±0.69
Manual-SS - BaseSLM 1.79 ±0.58 1.55 ±0.60
Phi-3-medium-128k-instruct
FtSLM 2.11 ±0.99 1.82 ±0.93
FtSLM PW noEx. 2.17 ±0.84 1.87 ±0.86
FtSLM PW 2.21 ±0.86 1.93 ±0.90
BaseSLM PW 2.00 ±0.44 1.68 ±0.57
BaseSLM PW noEx. 2.01 ±0.58 1.69 ±0.65
Manual-SS - BaseSLM - -
Phi-3-mini-128k-instruct
FtSLM 2.08 ±0.99 1.79 ±0.92
FtSLM PW noEx. 1.98 ±0.74 1.77 ±0.74
FtSLM PW 2.12 ±0.84 1.83 ±0.84
BaseSLM PW 1.66 ±0.59 1.44 ±0.67
BaseSLM PW noEx. 2.10 ±0.63 1.74 ±0.70
Manual-SS - BaseSLM 1.83 ±0.66 1.58 ±0.66
6.2 Evaluating Performance of eARCO with
Small Models
As discussed earlier, SLMs demonstrate significant potential when
finetuned for domain specific tasks such as RCA. By leveraging
such models, organizations can reduce the overhead costs associ-
ated with querying expensive LLMs or maintaining large retrieval
corpora for RAG pipelines. In this section, we present the evaluation
results of the responses generated by these SLMs. Furthermore, we
demonstrate how the prompt-optimization framework improves
the performance of these models, enhancing both accuracy and
efficiency.
The results presented in Table 4 provide a detailed comparison
of various finetuning and prompting strategies for SLMs, as out-
lined in section 5.2. Specifically, the three models—Phi-3-medium,
Phi-3-mini, and Phi-3.5-mini—were evaluated under different con-
figurations to examine the impact of finetuning ,PW Instructions ,
andPW static Examples on model performance.
The table presents performance across two datasets: the Com-
plete Test Dataset and the Filtered Test Dataset , where the latter
only includes incidents with available summaries in the Incident
Details . Incident summaries provide crucial context for accurate
RCA generation, and their absence can hinder model performance.
When evaluating the Filtered Test Dataset, we observe a consistent
improvement in accuracy and relevance. This highlights the critical
role of contextual information in enhancing RCA task performance
for SLMs.
Across all three models (Phi-3.5-mini, Phi-3-medium, Phi-3-mini),
the finetuned models with PW Instructions andPW static Examples
consistently achieve the highest scores on both the filtered and com-
plete test datasets. For instance, the Phi-3.5-mini model shows the
highest performance amongst all the models, with an average score
8

of 2.37 on the filtered test dataset and 2.01 ( FtSLM PW ) on the com-
plete test dataset, indicating that optimized prompts significantly
enhance the model’s ability to predict accurate root causes.
However, when comparing the performance of finetuned SLMs
without the PW static Examples (FtSLM PW noEx. ), there is a no-
ticeable decrease in scores. For instance, in the Phi-3.5-mini model
the scores drop to 2.13 on the filtered dataset, and to 1.90 on the
complete dataset. This demonstrates the importance of providing
the static examples ( PW static Examples ) to the finetuned SLMs to
impart crucial reasoning and domain-specific knowledge, without
incurring additional dynamic example retrieval cost.
The base SLM models with PromptWizard instructions also per-
form reasonably well, though they lag behind the finetuned models.
For instance, the Phi-3.5-mini BaseSLM PW achieves 2.26 on the
filtered dataset, showing improvement over the base model un-
der the Manual-SS setting (which scored 1.79). Nevertheless, the
performance gap between the base and finetuned SLM is evident,
especially when prompt optimization with examples is applied.
Lastly, applying the Manual-SS configurations to the base SLMs
demonstrate the lowest performance across all settings. These mod-
els, without finetuning or prompt engineering, score the lowest
on both datasets, reflecting the baseline performance before any
optimizations are applied. In particular, the Phi-3-medium model
under this configuration produces in null responses, underscor-
ing the limited effectiveness of the base models without further
enhancements.
6.3 Ablation Results
To analyse the utility of the semantically similar in-context exam-
ples, we perform an ablation by varying the number of examples
in the prompt with PromptWizard instructions. Table 5 shows per-
formance of the prompt with number of example ranging from 0 to
10. We observe that increasing the number of in-context examples
leads to better performance on the evaluation set going from 1.97
for the zero-shot setting to 2.51 for the 10-shot setting on the com-
plete dataset and from 2.13 to 2.91 on the filtered dataset. Hence
we observe around 27% improvement on the complete evaluation
set and 37% improvement on the filtered evaluation set dataset.
Table 5: GPT-4 evaluation scores for ablation on number of
semantically similar In-Context Examples with PW Instruc-
tions
Number of Examples Complete Test Dataset Filtered Test Dataset
GPT-4o
0 1.97 ±0.91 2.13 ±0.88
3 2.07 ±0.98 2.25 ±0.90
5 2.24 ±1.02 2.41 ±0.91
7 2.40 ±0.97 2.72 ±0.94
10 2.51 ±1.01 2.91 ±1.01
6.4 Human Evaluation Results
To understand the concrete impact of eARCO, we reached out to 47
OCEs involved in recent incident resolutions to assess responses
generated by various models. These models included GPT-4 andGPT-4o under the Manual-SS settings, GPT-4o using the PW-SS
configuration, and the Phi-3.5-mini model with the FtSLM PW setup.
Table 6 illustrate the accuracy and readability scores assigned by
the OCEs.
The GPT-4o model with optimized PromptWizard instructions
and 10 semantically similar examples (PW-SS) achieved the highest
average accuracy score of 2.91, reflecting a 14.12% and 7.45% im-
provement over GPT-4 and GPT-4o using Manual-SS, respectively.
For readability, the GPT-4o model in both Manual-SS and PW-SS
configurations scored highly, with a rating of 4.21, showing a 3.19%
improvement over GPT-4 under Manual-SS. Lastly, despite being
scalable and cost-effective solution, finetuned SLMs with optimized
instructions (FtSLM-PW) achieves an average accuracy of 2.23 and
even provided better recommendations than LLMs for a small num-
ber of incidents, which suggests that FtSLM-PW can be an attractive
alternative cost-effective option for RCA generation task.
In addition to evaluations from OCEs, we gathered scores from
10 researchers who are not incident management domain experts,
but have access to the ground truth root causes and model recom-
mendations. These human evaluators are divided into two groups.
Each group evaluated 25 incidents, resulting in a total of 50 inci-
dents. Table 7 presents the aggregated accuracy and readability
scores. Consistent with the OCEs’ evaluation, PW-SS achieves the
highest accuracy score of 3.50 and the highest readability score of
4.30.
Table 6: Accuracy and Readability scores assigned by OCEs
Model Accuracy Readability
GPT-4 2.55 ±1.26 4.08 ±0.90
GPT-4o 2.74 ±1.42 4.21 ±0.90
PW-SS 2.91 ±1.36 4.21 ±0.95
FtSLM PW 2.23 ±1.30 3.68 ±1.23
Table 7: Accuracy and Readability scores assigned by domain
experts
Model Accuracy Readability
GPT-4 3.15 ±1.25 3.93 ±0.79
GPT-4o 3.38 ±1.20 4.06 ±0.82
PW-SS 3.50 ±1.20 4.30 ±0.64
FtSLM PW 2.89 ±1.13 3.86 ±0.65
7 DISCUSSION
Promise of prompt optimization. Prompt instruction optimiza-
tion has gained popularity recently and proven to be effective in
many traditional NLP tasks. In this work, we first analyse and
demonstrate that optimized instructions can significantly improve
the performance on sensitive proprietary domain such as incident
management as well. Furthermore, we demonstrate that finetuned
9

domain adapted SLMs can be an attractive alternative for AIOPs
tasks. Moreover, we demonstrate that the optimized instructions
and static diverse synthetic in-context examples can significantly
boost the performance of domain adapted fintuned SLMs. These in-
sights will be valuable for designing more efficient and cost-effective
AIOPs solutions in future.
Deployment status and scale. We conducted our experiments
and evaluation by leveraging data from IcM of Microsoft. We have
deployed a large-scale RCA recommendation system as a service
by leveraging ICL pipeline with LLMs that is serving more than
hundred internal service teams for more than six months. For our ex-
periments, the recommendations for Manual-SS with GPT-4 method
are directly obtained from production environment. We have also re-
cently deployed the finetuned SLM solution to augment the recom-
mendations generated by the ICL pipeline. As eARCO demonstrate
superior performance on real-world production incident dataset,
we plan to deploy eARCO for both the ICL and finetuned SLM in
near future.
Threats to validity. Although the optimized prompt instruction
demonstrate superior performance with different settings of LLMs
and finetuned SLMs, the performance of eARCO is highly depen-
dant on the underlying language model. As we have evaluated the
performance of our models on incident data from Microsoft only,
the performance may vary if evaluated on different dataset from
other organizations. While the GPT-4 based accuracy evaluation
has been widely adopted, these evaluation results can be slightly
noisy due to hallucination problem of LLMs. Moreover, the human
evaluation is conducted on a small set of incidents as it is chal-
lenging to and time consuming to scale these feedback collection
process. In some cases, the model may potentially generate hal-
lucinated responses and additional noisy mitigation suggestions,
which can be problematic and misguide incident owners. Lastly,
we have only finetuned Phi-series of SLMs, but the performance
might improve further with other open-source SLMs. Moreover, we
plan to use domain adaptation techniques using RLHF techniques
to improve the performance of SLMs in future.
8 RELATED WORK
In this section, we summarize the existing work on incident man-
agement, adoption of LLMs for RCA and AIOPs tasks, and existing
prompt optimization techniques.
8.1 Incident Management
Given its practical importance, AI for Operations (AIOPs) tech-
niques have become popular for automatically resolving issues
arising from different stages of incident lifecycle management. Em-
pirical studies have been adopted broadly to understand the gaps
and limitations in existing large-scale cloud services, either delv-
ing into the types incident root causes [ 14,33] or system-level
issues [ 15,28]. Several AIOPs techniques have been proposed to ad-
dress the challenges in detection [ 13,29], triaging [ 6], diagnosis [ 5],
and mitigation [ 19] to either reduce human efforts or accelerating
the incident resolution process. Our work enhances the perfor-
mance of automated root cause generation task where accuracy
and efficiency are primary goals.8.2 Promise of LLMs in Incident Management
In large-scale cloud services, effective and efficient handling of
incidents is essential. Given the superior performance of LLMs in
several domain-specific software engineering tasks ranging includ-
ing code generation [ 9,31], program synthesis([ 18]), code review
[25,26], code repair [ 23,30] and code-fix [ 21], LLMs have been
adopted increasingly for solving problems in incident management
domain. Several recent works propose to address the incident di-
agnosis and RCA [ 3,11,32] tasks using LLMs. Ahmed et. al. [3]
propose to finetune a GPT model for learning domain specific
knowledge about incident management and recommend poten-
tial root causes at the time of incident creation. Zhang et. al. [32]
subsequently propose an efficient RAG based in-context learning
method for RCA generation task. In contrast to the existing works
that leverages manually designed static instructions with LLMs,
we propose to identify an optimized prompt instructions with syn-
thetic in-context examples and further demonstrate the potential
of cost-effective finetuned SLMs for RCA generation tasks.
8.3 Prompt Optimization
Existing LLM-based solutions for incident management heavily
rely on prompt engineering techniques. The prompts are carefully
chosen, often with multiple trials and errors, based on their ef-
fectiveness on the task being solved. This requires manual effort
and generated prompts could be sub-optimal due to lack of sys-
tematic exploration of prompts. Furthermore, the performance of
static prompt varies significantly as the underlying LLM evolves.
Prompt optimization techniques address these limitations by opti-
mizing either soft-prompt [ 8,27] or candidate prompts [ 12,17,34].
PromptWizard [ 2] is a recent work that outlines and addresses
the limitations of existing approaches - (i) high computation cost,
(ii) lack of human-interpretable prompts, (iii) sub-optimal prompt
outputs for complex tasks, (iv) lack of feedback-based exploration.
Given its efficiency and superior performance on several traditional
NLP tasks, we leverage PromptWizard in this work for identifying
optimized prompt instruction and synthetic in-context examples.
9 CONCLUSION
In this work, we propose eARCO, an efficient and scalable opti-
mized framework for automatically generating accurate root cause
recommendations for incidents in large-scale cloud services. By
leveraging state-of-the-art prompt optimization technique, we auto-
matically identified optimized prompt instruction that is augmented
with dynamically retrieved semantically similar in-context exam-
ples during real-time inference. Moreover, we develop scalable and
cost-effective finetuned SLMs and demonstrate that, with optimized
prompt instructions, these can be an attractive alternative solution
for RCA generation task. Our extensive experimental evaluation
by domain experts and GPT-4 as judge demonstrate that eARCO
improves the accuracy of RCA recommendations significantly for
both LLMs and finetuned SLMs on real-world incidents from Mi-
crosoft. We believe these insights will motivate the need of prompt
optimization and adoption of cost-effective finetuned SLMs in solv-
ing various challenges arising from different stages of incident
management lifecycle.
10

REFERENCES
[1]Marah Abdin, Sam Ade Jacobs, Ammar Ahmad Awan, Jyoti Aneja, Ahmed
Awadallah, Hany Awadalla, Nguyen Bach, Amit Bahree, Arash Bakhtiari, Harki-
rat Behl, et al .2024. Phi-3 technical report: A highly capable language model
locally on your phone. arXiv preprint arXiv:2404.14219 (2024).
[2]Eshaan Agarwal, Vivek Dani, Tanuja Ganu, and Akshay Nambi. 2024. PromptWiz-
ard: Task-Aware Agent-driven Prompt Optimization Framework. arXiv preprint
arXiv:2405.18369 (2024).
[3]Toufique Ahmed, Supriyo Ghosh, Chetan Bansal, Thomas Zimmermann, Xuchao
Zhang, and Saravan Rajmohan. 2023. Recommending Root-Cause and Mitigation
Steps for Cloud Incidents using Large Language Models. In 45th International
Conference on Software Engineering .
[4]Amar Prakash Azad, Supriyo Ghosh, Ajay Gupta, Harshit Kumar, Prateeti Mo-
hapatra, Lena Eckstein, Leonard Posner, and Robert Kern. 2022. Picking Pearl
From Seabed: Extracting Artefacts from Noisy Issue Triaging Collaborative Con-
versations for Hybrid Cloud Services. In Proceedings of the AAAI Conference on
Artificial Intelligence , Vol. 36. 12440–12446.
[5]Chetan Bansal, Sundararajan Renganathan, Ashima Asudani, Olivier Midy, and
Mathru Janakiraman. 2020. DeCaf: Diagnosing and Triaging Performance Issues
in Large-Scale Cloud Services. In 2020 IEEE/ACM 42nd International Conference
on Software Engineering: Software Engineering in Practice (ICSE-SEIP) .
[6]J. Chen, X. He, Q. Lin, Y. Xu, H. Zhang, D. Hao, F. Gao, Z. Xu, Y. Dang, and D.
Zhang. 2019. An Empirical Investigation of Incident Triage for Online Service
Systems. In 2019 IEEE/ACM 41st International Conference on Software Engineering:
Software Engineering in Practice (ICSE-SEIP) . 111–120.
[7]Junjie Chen, Xiaoting He, Qingwei Lin, Hongyu Zhang, Dan Hao, Feng Gao,
Zhangwei Xu, Yingnong Dang, and Dongmei Zhang. 2019. Continuous incident
triage for large-scale online service systems. In 2019 34th IEEE/ACM International
Conference on Automated Software Engineering (ASE) . IEEE, 364–375.
[8]Lichang Chen, Jiuhai Chen, Tom Goldstein, Heng Huang, and Tianyi Zhou. 2023.
Instructzero: Efficient instruction optimization for black-box large language
models. arXiv preprint arXiv:2306.03082 (2023).
[9]Mark Chen, Jerry Tworek, Heewoo Jun, Qiming Yuan, Henrique Ponde de Oliveira
Pinto, Jared Kaplan, Harri Edwards, Yuri Burda, Nicholas Joseph, Greg Brockman,
et al.2021. Evaluating large language models trained on code. arXiv preprint
arXiv:2107.03374 (2021).
[10] Yinfang Chen, Huaibing Xie, Minghua Ma, Yu Kang, Xin Gao, Liu Shi, Yunjie Cao,
Xuedong Gao, Hao Fan, Ming Wen, et al .2024. Automatic root cause analysis
via large language models for cloud incidents. In Proceedings of the Nineteenth
European Conference on Computer Systems . 674–688.
[11] Yinfang Chen, Huaibing Xie, Minghua Ma, Yu Kang, Xin Gao, Liu Shi, Yunjie Cao,
Xuedong Gao, Hao Fan, Ming Wen, et al .2024. Automatic root cause analysis
via large language models for cloud incidents. In Proceedings of the Nineteenth
European Conference on Computer Systems . 674–688.
[12] Chrisantha Fernando, Dylan Banarse, Henryk Michalewski, Simon Osindero, and
Tim Rocktäschel. 2023. Promptbreeder: Self-referential self-improvement via
prompt evolution. arXiv preprint arXiv:2309.16797 (2023).
[13] Vaibhav Ganatra, Anjaly Parayil, Supriyo Ghosh, Yu Kang, Minghua Ma, Chetan
Bansal, Suman Nath, and Jonathan Mace. 2023. Detection Is Better Than Cure:
A Cloud Incidents Perspective. In Proceedings of the 31st ACM Joint European
Software Engineering Conference and Symposium on the Foundations of Software
Engineering . 1891–1902.
[14] Yu Gao, Wensheng Dou, Feng Qin, Chushu Gao, Dong Wang, Jun Wei, Ruirui
Huang, Li Zhou, and Yongming Wu. 2018. An empirical study on crash recovery
bugs in large-scale distributed systems. In Proceedings of the 2018 26th ACM
joint meeting on european software engineering conference and symposium on the
foundations of software engineering . 539–550.
[15] Supriyo Ghosh, Manish Shetty, Chetan Bansal, and Suman Nath. 2022. How to
fight production incidents? an empirical study on a large-scale cloud service. In
Proceedings of the 13th Symposium on Cloud Computing . 126–141.
[16] Drishti Goel, Fiza Husain, Aditya Singh, Supriyo Ghosh, Anjaly Parayil, Chetan
Bansal, Xuchao Zhang, and Saravan Rajmohan. 2024. X-lifecycle learning for
cloud incident management using llms. In Companion Proceedings of the 32nd
ACM International Conference on the Foundations of Software Engineering . 417–
428.
[17] Qingyan Guo, Rui Wang, Junliang Guo, Bei Li, Kaitao Song, Xu Tan, Guoqing
Liu, Jiang Bian, and Yujiu Yang. 2023. Connecting large language models with
evolutionary algorithms yields powerful prompt optimizers. arXiv preprintarXiv:2309.08532 (2023).
[18] Naman Jain, Skanda Vaidyanath, Arun Iyer, Nagarajan Natarajan, Suresh
Parthasarathy, Sriram Rajamani, and Rahul Sharma. 2022. Jigsaw: Large lan-
guage models meet program synthesis. In Proceedings of the 44th International
Conference on Software Engineering . 1219–1231.
[19] Jiajun Jiang, Weihai Lu, Junjie Chen, Qingwei Lin, Pu Zhao, Yu Kang, Hongyu
Zhang, Yingfei Xiong, Feng Gao, Zhangwei Xu, et al .2020. How to mitigate
the incident? an effective troubleshooting guide recommendation technique for
online service systems. In Proceedings of the 28th ACM Joint Meeting on European
Software Engineering Conference and Symposium on the Foundations of Software
Engineering . 1410–1420.
[20] Yuxuan Jiang, Chaoyun Zhang, Shilin He, Zhihao Yang, Minghua Ma, Si Qin,
Yu Kang, Yingnong Dang, Saravan Rajmohan, Qingwei Lin, et al .2023. Xpert:
Empowering Incident Management with Query Recommendations via Large
Language Models. arXiv preprint arXiv:2312.11988 (2023).
[21] Carlos E Jimenez, John Yang, Alexander Wettig, Shunyu Yao, Kexin Pei, Ofir
Press, and Karthik Narasimhan. 2023. Swe-bench: Can language models resolve
real-world github issues? arXiv preprint arXiv:2310.06770 (2023).
[22] Jeff Johnson, Matthijs Douze, and Hervé Jégou. 2019. Billion-scale similarity
search with gpus. IEEE Transactions on Big Data 7, 3 (2019), 535–547.
[23] Harshit Joshi, José Cambronero Sanchez, Sumit Gulwani, Vu Le, Gust Verbruggen,
and Ivan Radiček. 2023. Repair is nearly generation: Multilingual program repair
with llms. In Proceedings of the AAAI Conference on Artificial Intelligence , Vol. 37.
5131–5140.
[24] Vladimir Karpukhin, Barlas Oguz, Sewon Min, Patrick Lewis, Ledell Wu, Sergey
Edunov, Danqi Chen, and Wen-tau Yih. 2020. Dense Passage Retrieval for Open-
Domain Question Answering. In Proceedings of the 2020 Conference on Empirical
Methods in Natural Language Processing (EMNLP) , Bonnie Webber, Trevor Cohn,
Yulan He, and Yang Liu (Eds.). Association for Computational Linguistics, Online,
6769–6781. https://doi.org/10.18653/v1/2020.emnlp-main.550
[25] Lingwei Li, Li Yang, Huaxi Jiang, Jun Yan, Tiejian Luo, Zihan Hua, Geng Liang,
and Chun Zuo. 2022. AUGER: automatically generating review comments with
pre-training models. In Proceedings of the 30th ACM Joint European Software
Engineering Conference and Symposium on the Foundations of Software Engineering .
1009–1021.
[26] Zhiyu Li, Shuai Lu, Daya Guo, Nan Duan, Shailesh Jannu, Grant Jenks, Deep
Majumder, Jared Green, Alexey Svyatkovskiy, Shengyu Fu, et al .2022. Automating
code review activities by large-scale pre-training. In Proceedings of the 30th
ACM Joint European Software Engineering Conference and Symposium on the
Foundations of Software Engineering . 1035–1047.
[27] Xiaoqiang Lin, Zhaoxuan Wu, Zhongxiang Dai, Wenyang Hu, Yao Shu, See-Kiong
Ng, Patrick Jaillet, and Bryan Kian Hsiang Low. [n. d.]. Use Your INSTINCT:
INSTruction optimization for LLMs usIng Neural bandits Coupled with Trans-
formers. In Forty-first International Conference on Machine Learning .
[28] Haopeng Liu, Shan Lu, Madan Musuvathi, and Suman Nath. 2019. What bugs
cause production cloud incidents?. In Proceedings of the Workshop on Hot Topics
in Operating Systems . 155–162.
[29] Pooja Srinivas, Fiza Husain, Anjaly Parayil, Ayush Choure, Chetan Bansal, and
Saravan Rajmohan. 2024. Intelligent Monitoring Framework for Cloud Services:
A Data-Driven Approach. In Proceedings of the 46th IEEE/ACM International
Conference on Software Engineering .
[30] Nalin Wadhwa, Jui Pradhan, Atharv Sonwane, Surya Prakash Sahu, Nagarajan
Natarajan, Aditya Kanade, Suresh Parthasarathy, and Sriram Rajamani. 2024.
CORE: Resolving Code Quality Issues using LLMs. Proceedings of the ACM on
Software Engineering 1, FSE (2024), 789–811.
[31] Frank F Xu, Uri Alon, Graham Neubig, and Vincent Josua Hellendoorn. 2022. A
systematic evaluation of large language models of code. In Proceedings of the 6th
ACM SIGPLAN International Symposium on Machine Programming . 1–10.
[32] Xuchao Zhang, Supriyo Ghosh, Chetan Bansal, Rujia Wang, Minghua Ma, Yu
Kang, and Saravan Rajmohan. 2024. Automated root causing of cloud incidents
using in-context learning with GPT-4. In Companion Proceedings of the 32nd ACM
International Conference on the Foundations of Software Engineering . 266–277.
[33] Yongle Zhang, Junwen Yang, Zhuqi Jin, Utsav Sethi, Kirk Rodrigues, Shan Lu,
and Ding Yuan. 2021. Understanding and detecting software upgrade failures
in distributed systems. In Proceedings of the ACM SIGOPS 28th Symposium on
Operating Systems Principles . 116–131.
[34] Yongchao Zhou, Andrei Ioan Muresanu, Ziwen Han, Keiran Paster, Silviu Pitis,
Harris Chan, and Jimmy Ba. 2022. Large language models are human-level
prompt engineers. arXiv preprint arXiv:2211.01910 (2022).
11