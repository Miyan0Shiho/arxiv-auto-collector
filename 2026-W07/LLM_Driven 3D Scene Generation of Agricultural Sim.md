# LLM-Driven 3D Scene Generation of Agricultural Simulation Environments

**Authors**: Arafa Yoncalik, Wouter Jansen, Nico Huebel, Mohammad Hasan Rahmani, Jan Steckel

**Published**: 2026-02-12 08:33:01

**PDF URL**: [https://arxiv.org/pdf/2602.11706v1](https://arxiv.org/pdf/2602.11706v1)

## Abstract
Procedural generation techniques in 3D rendering engines have revolutionized the creation of complex environments, reducing reliance on manual design. Recent approaches using Large Language Models (LLMs) for 3D scene generation show promise but often lack domain-specific reasoning, verification mechanisms, and modular design. These limitations lead to reduced control and poor scalability. This paper investigates the use of LLMs to generate agricultural synthetic simulation environments from natural language prompts, specifically to address the limitations of lacking domain-specific reasoning, verification mechanisms, and modular design. A modular multi-LLM pipeline was developed, integrating 3D asset retrieval, domain knowledge injection, and code generation for the Unreal rendering engine using its API. This results in a 3D environment with realistic planting layouts and environmental context, all based on the input prompt and the domain knowledge. To enhance accuracy and scalability, the system employs a hybrid strategy combining LLM optimization techniques such as few-shot prompting, Retrieval-Augmented Generation (RAG), finetuning, and validation. Unlike monolithic models, the modular architecture enables structured data handling, intermediate verification, and flexible expansion. The system was evaluated using structured prompts and semantic accuracy metrics. A user study assessed realism and familiarity against real-world images, while an expert comparison demonstrated significant time savings over manual scene design. The results confirm the effectiveness of multi-LLM pipelines in automating domain-specific 3D scene generation with improved reliability and precision. Future work will explore expanding the asset hierarchy, incorporating real-time generation, and adapting the pipeline to other simulation domains beyond agriculture.

## Full Text


<!-- PDF content starts -->

LLM-Driven 3D Scene Generation of Agricultural Simulation
Environments
Arafa Yoncalik1, Wouter Jansen1,2,3, Nico Huebel1, Mohammad Hasan Rahmani1,2and Jan Steckel1,2
Abstract— Procedural generation techniques in 3D rendering
engines have revolutionized the creation of complex environ-
ments, reducing reliance on manual design. Recent approaches
using Large Language Models (LLMs) for 3D scene generation
show promise but often lack domain-specific reasoning, verifi-
cation mechanisms, and modular design. These limitations lead
to reduced control and poor scalability. This paper investigates
the use of LLMs to generate agricultural synthetic simulation
environments from natural language prompts, specifically to
address the limitations of lacking domain-specific reasoning,
verification mechanisms, and modular design. A modular multi-
LLM pipeline was developed, integrating 3D asset retrieval,
domain knowledge injection, and code generation for the
Unreal rendering engine using its API. This results in a 3D
environment with realistic planting layouts and environmental
context, all based on the input prompt and the domain knowl-
edge. To enhance accuracy and scalability, the system employs
a hybrid strategy combining LLM optimization techniques
such as few-shot prompting, Retrieval-Augmented Generation
(RAG), finetuning, and validation. Unlike monolithic models,
the modular architecture enables structured data handling,
intermediate verification, and flexible expansion. The system
was evaluated using structured prompts and semantic accuracy
metrics. A user study assessed realism and familiarity against
real-world images, while an expert comparison demonstrated
significant time savings over manual scene design. The results
confirm the effectiveness of multi-LLM pipelines in automating
domain-specific 3D scene generation with improved reliability
and precision. Future work will explore expanding the asset
hierarchy, incorporating real-time generation, and adapting the
pipeline to other simulation domains beyond agriculture.
I. INTRODUCTION
In agriculture, traditional data collection is often con-
strained by seasonal cycles, weather conditions, and the
high costs of physical experimentation. Yet data plays a
critical role in advancing precision farming, automation,
and informed decision-making [1]. For instance, data-driven
models have been shown to enhance quality assurance and
anomaly detection in agricultural monitoring systems [2].
Moreover, simulation environments offer a viable alternative
to field-based data collection, enabling the generation of
synthetic datasets under controlled conditions and varying
scenarios, which are essential for training and evaluation of
smart systems [3].
Despite these advances, the creation of agricultural simu-
lation environments remains a time-intensive process. Man-
ually constructing virtual farmlands requires detailed model-
1All authors are with with Faculty of Applied Engineering – Dept. of
Electronics and ICT, University of Antwerp, 2000 Antwerp, Belgium
2Wouter Jansen, Mohammad Hasan Rahmani and Jan Steckel are with
Flanders Make Strategic Research Centre, 3920 Lommel, Belgium
3Corresponding author: Wouter Jansen (wouter.jansen@uantwerpen.be)ing of terrain, crop spacing, growth stages, and environmen-
tal variation, which is demanding and difficult to scale. While
Procedural Content Generation (PCG) has introduced au-
tomation into 3D scene design, existing tools often fall short
in addressing domain-specific rules essential to agricultural
realism. Crop-specific characteristics, such as species type,
lifecycle stage, seasonal growth behavior, and spatial layout
rules, require tailored logic that is difficult to generalize
using standard PCG methods [4]. Managing complex datasets
that encapsulate agricultural knowledge further complicates
this process, requiring the integration of domain-aware au-
tomation [5]. The introduction of procedural approaches
in 3D engines has significantly accelerated the creation of
complex virtual environments, reducing reliance on manual
design. This trend has been further driven by the widespread
availability of accessible technologies such as Unity, Unreal
Engine, and Blender, which have broadened access to 3D
content creation across domains beyond gaming, including
education, architecture, and simulation [6].
Large Language Models (LLMs) present the opportunity
to bridge this gap. Their ability to understand natural lan-
guage and generate structured outputs opens the door to
automating scene generation in 3D engines [7], [8]. However,
current single-LLM approaches face several limitations. They
often lack access to structured agricultural knowledge such
as planting rules, lifecycle stages, seasonal characteristics,
or spatial configurations. These systems also struggle to
validate their outputs, making them prone to hallucinations
or unrealistic scenes. Moreover, monolithic designs offer
limited modularity, which restricts their ability to scale across
diverse scenarios or adapt to domain-specific constraints.
As a result, such systems fall short in generating realistic,
consistent, and interpretable 3D agricultural simulations.
In 3D engines like Unreal, environments are constructed
from individual 3D objects called assets. This paper in-
vestigates a full LLM-based approach for generating agri-
cultural simulation environments, focusing on the proper
integration of LLMs across the entire pipeline. Inspired by
recent work showing the potential of LLM-driven modular
pipelines in other domains [7], [8], our system embeds expert
knowledge directly into the generation process to bridge
the gap between high-level intent and low-level procedural
logic. The proposed architecture decomposes the task into
three coordinated components: (1) asset retrieval from a
structured hierarchy, (2) domain knowledge integration via a
hybrid Retrieval-Augmented Generation (RAG) system, and
(3) Python code generation that drives environment rendering
through the Unreal Engine API. Multiple LLM optimizationarXiv:2602.11706v1  [cs.CV]  12 Feb 2026

Prompt: “generate an apple orchard consisting of Pinklady 
apples during summertime in a mature and healthy lifecycle stage.”
Generated Scene ResultUser UIFig. 1. An example prompt-to-scene generation. The interface takes a natural language prompt and outputs a procedurally generated agricultural environment
in Unreal Engine.
techniques are combined, including few-shot prompting,
finetuning, RAG, and LLM-based output validation, to im-
prove accuracy, flexibility, and generalization. It is important
to note that this work does not claim LLMs to be inherently
superior to alternative methods such as SQL-based retrieval
or purely procedural generation. Instead, it aims to evaluate
and refine how LLMs can be effectively integrated into a
structured pipeline for this task, identifying where they add
value, where they fall short, and how they might complement
other approaches. The system is evaluated quantitatively,
through semantic accuracy metrics, and qualitatively, through
a user study and an expert timing comparison with manual
design.
II. RELATEDWORK
A. Procedural Generation in 3D Simulations
Procedural generation refers to the algorithmic creation
of data, often in real-time, to produce complex, scalable
environments. It has become a cornerstone technology in 3D
simulations and offers significant efficiency gains by reduc-
ing manual design efforts [4], [9]. These techniques have
been adopted in domains ranging from gaming and architec-
ture to urban planning and scientific visualization [10]. Titles
such asMinecraftexemplify the use of procedural algorithms
to create expansive, diverse worlds with minimal manual
intervention [11]. Modern PCG has evolved from static
rule-based systems to dynamic, machine learning-enhanced
methods capable of embedding functional properties such as
soil variability, irrigation zones, and crop interactions [12],
[13], [14].
B. Generative AI and Multi-LLM Architectures
Generative AI, particularly LLMs, has revolutionized natu-
ral language processing, code generation, and creative work-
flows. LLMs excel at transforming unstructured prompts
into structured outputs, making them ideal for scene gen-
eration tasks that require both linguistic understanding and
procedural logic [15]. In recent developments such as 3D-
GPT, LLMs were successfully applied to procedural scenecreation using frameworks such as Blender and Infinigen,
demonstrating the viability of prompt-based 3D generation
[7]. Multi-LLM architectures extend this idea by distributing
specialized responsibilities across different models [16]. This
modular structure supports task-specific optimization and
reduces semantic drift, token limitations, and performance
bottlenecks [17], [18]. Prompt-based LLM pipelines offer a
compelling balance between the accuracy of PCG algorithms
and the flexibility of natural language interfaces. Unlike
manual design, which offers full control but is time-intensive,
or rigid PCG scripts, which require significant setup, prompt-
based systems allow users to express scene goals intuitively.
The system then automates the generation process while
ensuring adherence to domain logic and scene consistency.
C. Applications in Agricultural Simulation
Simulation plays an increasingly important role in mod-
ern agriculture. 3D environments are used for training au-
tonomous systems, planning layouts, testing planting strate-
gies, and visualizing environmental impacts [19], [20], [21].
AI-driven simulations train robots for tasks such as harvest-
ing and weeding, reducing the need for real-world trial-and-
error [22]. Despite these advances, procedural generation in
agriculture is still underutilized. Existing tools often lack
the capability to incorporate dynamic, rule-based agricultural
knowledge such as optimal planting density, disease states,
or growth curves across seasons.
III. METHODS
A. System Overview
The system developed in this paper is a modular, multi-
LLM pipeline designed to convert natural language prompts
into procedurally generated 3D agricultural scenes in Un-
real Engine (see Figure 2). The architecture divides the
overall task into three logically distinct stages: (1) asset
path retrieval, (2) domain knowledge injection, and (3) code
generation. Each stage is powered by a hybrid LLM-based
mechanism, enabling a layered approach that reduces task
complexity per model and increases reliability.

4. Python Code      
User
ASSET RETRIEV AL
 LLM
DOMAIN KNOWLEDGE 
LLM
HEIGHTMAP LLM
... LLM
....
DESKTOP AP...
USER PROMPT
DEVELOPMENT AGENT 
LLM
UNREAL ENGINE EDITOR
SCENE GENERATION
Large Language Models
Potential LLM extensions
User interactive applications
Seperation backend procedures
"Generate an apple orchard"
2. Original prompt 
    and asset paths
3. From original prompt to 
    enchanced prompt
OUTPUT
1. Write prompt in UI
DESKTOP APPLICATION
OUTPUT
OUTPUTFig. 2. High-level architecture of the multi-LLM system.
The pipeline begins with a user prompt, typically a natural
language description such as “Generate an apple orchard of
the type Pink Lady during summer in a mature growth stage.”
This input is passed to a standalone Desktop Application,
where the user communicates through a graphical interface.
Once submitted, the backend activates the three-stage LLM
pipeline.
First, the Asset Retrieval LLM parses the input and returns
structured asset paths from a predefined hierarchy. Next,
the Domain Knowledge LLM enriches this with agricultural
metadata retrieved using a RAG technique [23]. Finally,
the Development Agent LLM generates Python code based
on the enhanced prompt and metadata. This code is then
executed inside the Unreal Engine Editor, which the user
can also interact with to view the resulting 3D scene (see
Figure 1). The modular architecture allows optimization tech-
niques at each stage, such as few-shot prompting, subquery
decomposition, prompt tuning, and validation.
B. Asset Hierarchy Design
To ensure consistent asset referencing, a structured di-
rectory hierarchy was implemented for all agricultural 3D
models used in the system. The hierarchy separates content
into two main categoriesFruitsandVegetableseach of
which contains multiple crop types. Each crop type is further
subdivided by specificvarieties,growth stages,seasonal
appearance, andhealth conditions. This hierarchical struc-
ture enables the system to map natural language prompts to
precise asset paths, ensuring visual consistency and domain
fidelity in generated environments.
The hierarchy was generated programmatically and, for
the purposes of this study, was limited to a predefined
test dataset containing a representative subset of crops and
hierarchy elements. This selection was made to balance
domain coverage with feasibility for evaluation, while still
enabling diverse scene generation. It spans fruits (e.g., apple,
banana, cherry), vegetables (e.g., carrot, lettuce, tomato),
multiple growth stages (vegetative, reproductive, maturation),seasonal appearances (fall, winter, summer, spring), and
health states (healthy, ill). In total, the structure allows for
672 asset path combinations, supporting a wide variety of
simulation scenarios. All assets are stored in a format com-
patible with Unreal Engine, and their paths are embedded in
a semantic retrieval system to enable intelligent selection via
LLMs. The asset hierarchy also plays a critical role in the
system’s retrieval and code generation phases, acting as the
backbone for ensuring that procedural scenes are populated
with appropriate visual representations based on the user’s
prompt.
C. Asset Retrieval LLM
To convert natural language prompts into valid asset paths,
several LLM optimization strategies were evaluated, includ-
ing few-shot prompting, finetuning, and RAG. Based on
performance across varying prompt types, a hybrid method
was selected. This approach combines subquery decomposi-
tion, semantic search, and GPT-based refinement to balance
flexibility with strict adherence to the asset hierarchy.
The process begins by parsing user prompts into sub-
queries using GPT-4 [24], especially in multi-field scenarios.
These subqueries are then standardized according to the
system’s asset hierarchy. For example, loosely defined terms
like “early stage” or “flowering” are normalized to lifecy-
cle categories such asVegetative,Reproductive, or
Maturation.
Normalized subqueries are embedded using OpenAI’s
text-embedding-3-smallmodel and compared
against a FAISS index of known asset paths. This semantic
search retrieves the closest matches based on meaning, not
exact wording, enabling flexible interaction while ensuring
structural consistency [25]. A final GPT-4 validation step
ensures that retrieved paths match the user’s intent in terms
of variety, lifecycle, season, and health state. It also enforces
internal consistency, such as matching seasonal context
across multiple fields.

D. Domain Knowledge Integration via RAG
Following asset retrieval, the system enriches each asset
path with structured agricultural metadata via a custom RAG
pipeline. This ensures that the final scene generation reflects
realistic properties such as crop spacing, growth behavior,
and seasonal effects.
A FAISS index stores embeddings of domain knowledge
entries, each a JSON object describing a specific crop con-
figuration, including variety, lifecycle, season, health state,
and properties like height, density, disease susceptibility,
irrigation, and rendering effects. Metadata fields parsed from
the asset path (e.g. category, variety, season) are combined
into a descriptor string (e.g. “healthy young Pink Lady apple
in fall”), embedded using the same model as the index,
and matched via top-k semantic search. A secondary filter
ensures retrieved entries fully align with the asset path,
including fuzzy variety matching. If no match is found, the
system logs the failure for fallback handling.
The final output is a structured recipe of enriched JSON
entries containing both agricultural metadata and scene-
specific attributes (e.g., model tags, scaling, environmental
settings). This ensures each generated scene is not only vi-
sually coherent but also agronomically accurate and context-
aware.
E. Code Generation LLM and Unreal Engine Integration
The final stage of the pipeline transforms the domain-
enriched scene recipe into executable Python code that
generates a 3D environment in Unreal Engine. This is
accomplished through a specialized Code Generation LLM,
which has been optimized via prompt engineering, structural
constraints, and custom instruction finetuning to produce
Python scripts that are both syntactically correct and com-
patible within Unreal Engine.
The LLM takes three inputs: the original user prompt,
validated asset paths, and the enriched scene recipe retrieved
via RAG. It then produces a structured Python script that
instantiates and places 3D assets based on domain metadata,
such as lifecycle stages, spacing guidelines, and seasonal
context. The code includes parameters like model references,
row and tree spacing, scaling, rotation, and environmental
behaviors. Scripts follow a modular layout, with clear func-
tions for scene setup, looping through domain objects, and
applying placement and attachment logic.
To ensure correctness, a final validation step checks for:
•Missing or malformed asset paths,
•Unreal-specific library errors (e.g., attachment rules or
constructor calls),
•Mismatches between domain metadata and scene pa-
rameters (e.g., incorrect scale).
The LLM was finetuned on a curated dataset of
prompt–script pairs collected during development, which
significantly improved output structure and reduced hallu-
cinations. By combining flexible prompt-to-code generation
with deterministic rendering in Unreal Engine, this stage
closes the loop between natural language input and visual
output.TABLE I
EVALUATION OF ASSET RETRIEVAL ACROSS PROMPT TYPES
Prompt Type Metric Few-Shot Finetuning RAG Hybrid
Single-field (detailed) Accuracy 100% 100% 97% 98%
Single-field (generic) Accuracy 83% 66% 87% 71%
Multi-field (generic)Precision 74% 63% 74% N/A
Recall 6% 89% 8% N/A
F1 Score 10% 74% 15% N/A
IV. RESULTS ANDEVALUATION
The evaluation covers the complete pipeline, starting with
asset retrieval, followed by domain knowledge alignment,
and concluding with Unreal Python code generation. Beyond
these core stages, we also compare the modular multi-LLM
architecture against a single-LLM baseline and present two
human-centered studies: one on user-perceived scene quality
and another on expert time efficiency. Quantitative metrics
include accuracy, precision, recall, F1 score, and Top-kre-
trieval accuracy, while qualitative assessments capture visual
realism, semantic correctness, and scalability considerations.
A. Asset Retrieval Evaluation
While LLMs are commonly assessed using standard-
ized benchmarks such as MMLU, HellaSwag, or Hu-
manEval [26], these benchmarks are designed for general
purpose reasoning or programming tasks. They do not
capture the domain-specific requirements involved in asset
retrieval, agricultural metadata alignment, or 3D scene gen-
eration. To address this, a custom evaluation framework was
developed to assess performance across the three key stages
of the pipeline: asset path retrieval, domain knowledge inte-
gration, and code generation. This framework uses manually
constructed ground truth sets, accuracy-based metrics, and
visual validation within Unreal Engine to assess real-world
applicability. To evaluate asset path retrieval performance, a
benchmark of 100 natural language prompts was created and
categorized as follows:
•Single-field, detailed: e.g., “Generate a healthy Pink
Lady apple orchard in summer.”
•Single-field, generic: e.g., “Generate an apple field.”
•Multi-field, generic: e.g., “Generate some fruit and
vegetable fields.”
For each prompt, a ground truth set of expected asset
paths was manually constructed. Performance was evaluated
across four configurations: few-shot prompting, finetuning,
RAG, and a hybrid approach. Each was evaluated using
100 prompts across different categories (detailed vs. generic,
single-field vs. multi-field). The metrics assessed include
accuracy, precision, recall, and F1 score. For the hybrid
system, only accuracy was measured, as it is explicitly
designed to return only the minimal necessary asset paths
per prompt, rather than an exhaustive list of all valid options.
As a result, metrics like precision and recall, which assume
comprehensive retrieval (i.e., returning all correct options),
become less informative for this evaluation. These results can
be found in Table I.

These results highlight the strengths and trade-offs of
each approach. Few-shot prompting and finetuning achieved
perfect accuracy for detailed single-field prompts. Finetuning
demonstrated strong recall and F1 in multi-field queries,
showing its ability to retrieve broader asset sets. RAG
achieved high precision but lower recall, indicating that it
retrieved fewer but often correct paths. The hybrid approach
focused on high-precision retrieval for low-ambiguity queries
by minimizing false positives, though precision and recall
metrics were not computed due to its constrained output
design.
TABLE II
DOMAINKNOWLEDGERETRIEVALACCURACY(EXACTMATCH)
MethodTop-1
AccuracyTop-2
RecallTop-3
Recall
RAG 71% 98% 100%
Hybrid Filter 82% 96% 100%
B. Domain Knowledge Evaluation
To evaluate the accuracy and consistency of domain
knowledge retrieval, two methods were tested: a baseline
RAG approach and an improved hybrid approach combining
RAG with strict metadata filtering. Both systems retrieved
relevant JSON metadata entries from a FAISS index based
on the asset paths selected during the previous stage. The
full results can be found in Table II.
The first method,RAG, directly embedded the asset path as
text and performed semantic similarity search. While simple,
this method lacked deterministic control over field-specific
attributes such as season or lifecycle stage. A Top-1 search
achieved only 71% accuracy in exact metadata matches.
This often led to scene generation errors, such as assigning
summer properties to a spring apple orchard.
In contrast, thehybrid approachfirst parsed asset paths
into structured metadata (e.g., category, subtype, variety,
lifecycle, season, health), embedded this metadata, and then
retrieved the top-k semantically close candidates. A strict
post-filtering stage then validated whether each candidate’s
metadata matched the query exactly. This significantly out-
performed RAG in Top-1 accuracy while maintaining the
same Top-3 coverage. Even at Top-3 retrieval, the hybrid
system queried less than 0.5% of the total database, demon-
strating high efficiency.
The hybrid method improved consistency across all
prompt types by avoiding mismatched fields (e.g., incorrect
seasons) and enabling more precise scene recipes. This
accuracy is critical because errors in this stage propagate di-
rectly into code generation, affecting the visual and semantic
fidelity of the final simulation environment.
C. Code Generation Evaluation
This section evaluates the performance of the multi-LLM
pipeline for generating Unreal Engine Python scripts based
on structured scene recipes.TABLE III
DEVELOPMENTAGENTCODEGENERATIONRESULTS(10PROMPTS)
Prompt TypeExecu-
tabilityCorrect
PathsDomain
MatchVisual
Accuracy
Single-field 100% 100% 100% 100%
Multi-field 100% 100% 100% 70%
All results reported here are produced by the modular system
described earlier, with separate LLMs for asset retrieval, do-
main knowledge enrichment, and code generation. To assess
the performance of the development agent, two sets of evalu-
ations were conducted: one using 10 single-field prompts and
another using 10 more complex multi-field prompts. Each
generated script was assessed on four criteria: whether the
generated code was executable within Unreal Engine, the
accuracy of asset path utilization, the scene’s adherence to
domain-specific knowledge such as object spacing, scaling,
and structural layout, and, if the scene executed successfully,
whether the visual output aligned with the original prompt’s
intent. The full results can be found in Table III.
The 10 single-field prompts produced scripts that were
executable, referenced correct asset paths, respected domain
knowledge properties (e.g., spacing and scaling), and visually
matched the user description. Minor visual inaccuracies due
to shared asset meshes across seasons were observed, but
these did not affect correctness at the code level.
In comparison, the evaluation of 10 multi-field prompts
showed similar robustness in execution and metadata han-
dling. All scripts were executable and semantically consis-
tent. However, three outputs exhibited visual limitations: two
failed to generate correct spatial relationships between fields
(e.g. missing layout offsets), and one spawned a single asset
per type rather than full fields.
Figure 1 shows an example of a system-generated agri-
cultural field produced by the pipeline, visualized in Unreal
Engine based on a natural language prompt. This illustrates
the spatial layout, variety selection, and planting logic ap-
plied by the code generation stage.
D. Single vs Multi-LLM Architecture Evaluation
The results reported in the previousCode Generation
Evaluationsubsection reflect the performance of the pro-
posed modular multi-LLM pipeline, in which code gen-
eration is informed by separate stages for asset retrieval
and domain knowledge integration. While AI-driven sys-
tems usually rely on a single LLM to manage the entire
pipeline from natural language interpretation to execution,
this paper adopts a modular architecture. The pipeline divides
the process into three specialized components: asset path
retrieval, domain knowledge enrichment, and Unreal Python
code generation. This separation enhances understanding,
simplifies debugging, and allows for targeted optimization
at each stage.

TABLE IV
SINGLE-LLMRESULTS(10PROMPTS)
Prompt TypeExecu-
tabilityDomain
MatchVisual
Accuracy
Single-field 70% 80% 100%
Multi-field 100% 70% 90%
To evaluate the effectiveness of this design choice, a
controlled internal comparison was conducted between the
modular multi-LLM pipeline and a baseline single-LLM
configuration. In the baseline setup, GPT-4 received a raw
user prompt along with an embedded task description and
few-shot examples. It was expected to infer asset paths, apply
domain-specific logic, and generate Unreal-compatible code
within a single step. The outputs from both approaches were
evaluated using a combination of qualitative and quantitative
criteria:
•Modularity:The single-LLM setup lacked clear stage
separation, making debugging and targeted improve-
ments difficult. The modular pipeline enabled finer
control over each step.
•Scalability:The single-LLM approach hit prompt
length limits with large asset taxonomies, while the
multi-LLM system stored the hierarchy in a searchable
embedded database.
•Correctness:The single-LLM method often halluci-
nated asset paths or omitted domain metadata. Interme-
diate validation in the modular pipeline greatly reduced
such errors.
•Flexibility:The multi-LLM design supported special-
ized methods like RAG, lifecycle normalization, and
code validation loops that are harder to combine in a
single prompt.
To ground this comparison, we applied the same evaluation
protocol used for the code generation stage: ten single-field
and ten multi-field prompts were used to assess scene gener-
ation quality. Each result was scored on code executability,
asset path correctness, domain knowledge application, and
visual accuracy. These results can be found in Table IV.
An observed issue with the single-LLM system was its
failure to automatically format asset paths correctly, particu-
larly omitting necessary Unreal Engine path prefixes such
as/Game/and file suffixes like.fbx. Without manual
intervention, none of the scripts executed. Once corrected,
the system could produce working outputs, but domain
metadata such as crop density and field spacing often re-
mained inaccurate. By contrast, the multi-LLM system with
dedicated stages for asset retrieval and metadata alignment
generated executable and visually accurate results without
manual path manipulation. The modular pipeline also exhib-
ited more flexible and diverse code structures, as opposed to
the repetitive patterns observed in the single-LLM outputs.
These findings support the architectural choice to separate
the generation process into stages and shows better results
for real-world deployment in large asset ecosystems.E. Interpretation and Scalability Considerations
While standard metrics like accuracy, precision, and F1
score offer valuable insight into prompt handling and asset
matching, they do not fully capture the system’s real-world
utility. For example, the high accuracy of few-shot prompting
in single-field prompts is partly due to controlled testing con-
ditions: the asset hierarchy contained only 672 combinations,
small enough to fit entirely in the system prompt, giving
GPT-4 complete visibility over all valid paths.
This setup is not scalable. As asset paths grow into
the thousands, prompt size and context limits make few-
shot prompting insufficient. Finetuning improved format
consistency but lacked the precision required to return exact
asset paths, essential for Unreal Engine scene generation.
This motivated a shift to RAG, which enabled accurate
and scalable semantic retrieval from an embedded index of
asset paths without modifying the base model. However,
RAG alone lacked the reasoning to map vague or multi-
field prompts reliably. The hybrid approach addressed this
by combining sub-query decomposition, normalization, and
semantic validation to better align user intent with database
retrieval.
It is also important to highlight that precision, recall,
and F1 scores are only practical in our case because the
dataset is relatively small and the full set of valid answers
is known. For significantly larger asset hierarchies, such
evaluations may become infeasible due to output token limits
and ambiguity in defining exhaustive ground truths.
Finally, model performance evolves over time; future
results may differ even with identical tests. Our design
therefore prioritizes adaptability, modularity, and correctness
over short-term benchmark gains.
F . User Evaluation and Results
A user study was conducted with 10 participants to eval-
uate the human-perceived quality of the generated scenes.
Each participant completed three evaluations, resulting in
30 scene assessments. The goal was to determine whether
the multi-LLM pipeline could accurately interpret natural
language descriptions and produce visually realistic agri-
cultural environments. In each evaluation, participants were
shown a real-world image of an agricultural field and asked
to describe it freely. This description was then used as
input to the system, which generated a corresponding 3D
scene in Unreal Engine. A screenshot of the generated scene
was presented alongside the original image for evaluation.
Participants rated the system output using the following
questions:
•Q1.To what extent does this generated scene match
what you described? (1–10)
•Q2.How similar is this scene to the original image
(excluding terrain)? (1–10)
•Q3.How realistic is the generated environment? (1 =
Not realistic, 5 = Extremely realistic)
•Q4.Are species, spacing, and layout agriculturally
accurate? (Self-rated expertise: 1–10)

Participants also responded to two open-ended questions
on missing elements and general suggestions.
1) Results and Interpretation:Average scores from the
30 evaluations are shown in Table V. The "Agri Expertise"
score reflects participants’ self-assessed expertise rather than
the objective quality of the layout. Participants generally
perceived the system’s scene generation as moderately ac-
curate in capturing their descriptions. However, realism and
visual similarity scores were lowered due to repetitive assets
and the absence of terrain or soil elements. As confirmed in
open-ended feedback, many visual mismatches were due to
generic 3D models or missing environmental variation. Fu-
ture inclusion of terrain and ground generation is expected to
significantly improve visual coherence and user satisfaction.
TABLE V
USEREVALUATIONSUMMARY(N= 30)
Metric Average Score (1–10)
Prompt Match (Q1) 6.77
Visual Similarity (Q2) 5.50
Scene Realism (Q3) 5.54
Agri Expertise (Q4) 3.70
G. Expert Evaluation
To benchmark the system’s performance against traditional
workflows, an expert evaluation was conducted involving
three professionals experienced in Unreal Engine and 3D
scene construction. The goal was to compare manual and
automated scene generation in terms of time efficiency.
Each expert was asked to manually recreate three single-
field agricultural scenes based on natural language prompts
previously used by the system. For fairness, the correct asset
paths were provided in advance, so the evaluation focused
strictly on placement, scaling, and spacing, excluding time
spent searching through the asset hierarchy. The same scenes
were first generated using the multi-LLM pipeline, and the
total generation time was recorded. The results are shown in
Table VI and visualized in Figure 3.
TABLE VI
SYSTEM VSEXPERTSCENEGENERATIONTIME(SINGLE-FIELD)
Field System (s) Expert (s) Saved (s) % Faster
Scene 1 37.61 109.13 71.52 65.54%
Scene 2 57.75 82.22 24.47 29.76%
Scene 3 52.83 91.85 39.02 42.47%
On average, the system required only 49 seconds per
scene, while experts took 94.4 seconds, almost twice as long.
This shows a substantial efficiency gain. Given the simplicity
of single-field scenes, it is reasonable to expect that the time
savings would be even greater in more complex, multi-field
scenarios. As the system evolves to include features such as
terrain generation, seasonal weather, and foliage variation,
this time advantage is likely to expand even further.
Comparision of System vs Expert - DurationDuration (s)
Field Field #1 Field #2 Field #3Time Saved (%)65
60
55
50
45
40
35
30100
80
60
40
20
0% Time Saved
System Duration
Expert DurationFig. 3. Comparison of generation time: System vs Expert (Single-Field
Scenes).
V. DISCUSSION& CONCLUSION
This paper explored a structured method for generating 3D
agricultural environments in Unreal Engine using a modular
system powered by LLMs. These results highlight the value
of combining generative AI with structured domain logic,
providing a foundation for developing interactive, scalable
simulation tools across domains beyond agriculture. By
dividing the process into asset retrieval, domain knowledge
enrichment, and code generation, the system addresses key
challenges in aligning natural language with precise scene
composition.
While the proposed multi-LLM pipeline shows promise in
generating agriculturally accurate 3D environments, several
limitations remain. The system relies on static, pre-defined
assets, with realism constrained by asset quality and meta-
data. It lacks procedural generation and dynamic features
such as crop growth or environmental interactions.
Despite hybrid asset retrieval, ambiguous or under-specified
prompts can still cause inconsistencies or hallucinated meta-
data, reflecting the need for broader and more balanced
knowledge coverage. Reliance on external API services also
introduces latency, costs, and internet dependency. Evalua-
tion was further limited by a small participant sample and
narrow scene variety.
From a code generation perspective, the system consis-
tently applies placement, scaling, and row-spacing rules, but
does not yet incorporate more complex agricultural attributes
such as seasonal lighting effects, foliage density, or disease
visualizations. These omissions are partly due to limitations
in asset modularity, highlighting the need for richer and more
flexible asset libraries.
Future work should expand domain knowledge to include
weather, soil, and biological variation, integrate procedural
terrain and environment generation, and enable interactive
editing or selective regeneration. Beyond agriculture, the
modular multi-LLM architecture could also extend to do-
mains such as urban planning, forestry, and environmental
conservation.

REFERENCES
[1] K. H. Coble, A. K. Mishra, S. Ferrell, and T. Griffin, “Big data in agri-
culture: A challenge for the future,”Applied Economic Perspectives
and Policy, vol. 40, no. 1, pp. 79–96, 2018.
[2] K. Gkountakos, K. Ioannidis, K. Demestichas, S. Vrochidis, and
I. Kompatsiaris, “A comprehensive review of deep learning-based
anomaly detection methods for precision agriculture,”IEEE Access,
vol. 12, pp. 197715–197733, 2024.
[3] R. R. Shamshiri, I. A. Hameed, L. Pitonakova, C. Weltzien, S. K.
Balasundram, I. J. Yule, T. E. Grift, and G. Chowdhary, “Simulation
software and virtual environments for acceleration of agricultural
robotics: Features highlights and performance comparison,”Interna-
tional Journal of Agricultural and Biological Engineering, vol. 11,
no. 4, pp. 12–20, 2018.
[4] A. Emilien, A. Bernhardt, A. Peytavie, M.-P. Cani, and E. Galin,
“Procedural generation of villages on arbitrary terrains,”The Visual
Computer, vol. 28, pp. 809–818, June 2012.
[5] H. Williamson, J. Brettschneider, M. Caccamo, R. Davey, C. Goble,
P. Kersey, S. May, R. Morris, R. Ostler, T. Pridmore, C. Rawlings,
D. Studholme, S. Tsaftaris, and S. Leonelli, “Data management
challenges for artificial intelligence in plant and agricultural research,”
F1000Research, vol. 10, p. 324, Apr 2021.
[6] C. M. and, “Gaming engines: Unity, unreal, and interactive 3d spaces,”
Technology|Architecture + Design, vol. 5, no. 2, pp. 246–249, 2021.
[7] C. Sun, J. Han, W. Deng, X. Wang, Z. Qin, and S. Gould, “3d-gpt:
Procedural 3d modeling with large language models,” 2024.
[8] Y . Yang, J. Lu, Z. Zhao, Z. Luo, J. J. Q. Yu, V . Sanchez, and F. Zheng,
“Llplace: The 3d indoor scene layout generation and editing via large
language model,” 2024.
[9] G. Smith, “Understanding procedural content generation: a design-
centric analysis of the role of pcg in games,” inProceedings of the
SIGCHI Conference on Human Factors in Computing Systems, CHI
’14, (New York, NY , USA), p. 917–926, Association for Computing
Machinery, 2014.
[10] J. Togelius, A. J. Champandard, P. L. Lanzi, M. Mateas, A. Paiva,
M. Preuss, and K. O. Stanley, “Procedural Content Generation: Goals,
Challenges and Actionable Steps,” inArtificial and Computational
Intelligence in Games(S. M. Lucas, M. Mateas, M. Preuss, P. Spronck,
and J. Togelius, eds.), vol. 6 ofDagstuhl Follow-Ups, pp. 61–75,
Dagstuhl, Germany: Schloss Dagstuhl – Leibniz-Zentrum für Infor-
matik, 2013.
[11] M. Dahrn,The Usage of PCG Techniques Within Different Game
Genres. Dissertation, 2021.
[12] J.-H. Liu, S.-K. Zhang, C. Zhang, and S.-H. Zhang, “Controllable
procedural generation of landscapes,” inProceedings of the 32nd ACM
International Conference on Multimedia, MM ’24, (New York, NY ,
USA), p. 6394–6403, Association for Computing Machinery, 2024.
[13] J. Vuleti ´c, M. Poli ´c, and M. Orsag, “Procedural generation of synthetic
dataset for robotic applications in sweet pepper cultivation,” in2022
International Conference on Smart Systems and Technologies (SST),
pp. 309–314, 2022.
[14] P. Bontrager and J. Togelius, “Learning to generate levels from
nothing,” 2021.
[15] S. Gao, X.-C. Wen, C. Gao, W. Wang, H. Zhang, and M. R. Lyu,
“What makes good in-context demonstrations for code intelligence
tasks with llms?,” in2023 38th IEEE/ACM International Conference
on Automated Software Engineering (ASE), pp. 761–773, 2023.
[16] Q. Wu, G. Bansal, J. Zhang, Y . Wu, B. Li, E. Zhu, L. Jiang, X. Zhang,
S. Zhang, J. Liu, A. H. Awadallah, R. W. White, D. Burger, and
C. Wang, “Autogen: Enabling next-gen llm applications via multi-
agent conversation,” 2023.
[17] W. Shen, C. Li, H. Chen, M. Yan, X. Quan, H. Chen, J. Zhang, and
F. Huang, “Small llms are weak tool learners: A multi-llm agent,”
2024.
[18] C.-Y . Chang, Z. Jiang, V . Rakesh, M. Pan, C.-C. M. Yeh, G. Wang,
M. Hu, Z. Xu, Y . Zheng, M. Das, and N. Zou, “Main-rag: Multi-agent
filtering retrieval-augmented generation,” 2024.
[19] A. Chipanshi, E. Ripley, and R. Lawford, “Large-scale simulation of
wheat yields in a semi-arid environment using a crop-growth model,”
Agricultural Systems, vol. 59, no. 1, pp. 57–66, 1999.
[20] C. O. Stöckle, M. Donatelli, and R. Nelson, “Cropsyst, a cropping
systems simulation model,”European Journal of Agronomy, vol. 18,
no. 3, pp. 289–307, 2003. Modelling Cropping Systems: Science,
Software and Applications.[21] S. Noda, M. Kogoshi, and W. Iijima, “Robot simulation on agri-
field point cloud with centimeter resolution,”IEEE Access, vol. 13,
pp. 14404–14416, 2025.
[22] M. Chowdhury and R. Anand, “Ai-driven agricultural robotics: Ad-
vancements and applications,”
[23] Y . Gao, Y . Xiong, X. Gao, K. Jia, J. Pan, Y . Bi, Y . Dai, J. Sun,
M. Wang, and H. Wang, “Retrieval-augmented generation for large
language models: A survey,” 2024.
[24] OpenAIet al., “Gpt-4 technical report,” 2024.
[25] P. P. Ghadekar, S. Mohite, O. More, P. Patil, Sayantika, and S. Man-
grule, “Sentence meaning similarity detector using faiss,” in2023
7th International Conference On Computing, Communication, Control
And Automation (ICCUBEA), pp. 1–6, 2023.
[26] T. Ivanov and V . Penchev, “Ai benchmarks and datasets for llm
evaluation,” 2024.