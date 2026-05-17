# GGBound: A Genome-Grounded Agent for Microbial Life-Boundary Prediction

**Authors**: Hanbo Huang, Xuan Gong, Jing Wang, Lei Bai, Xiang Xiao, Weishu Zhao, Shiyu Liang

**Published**: 2026-05-14 06:37:55

**PDF URL**: [https://arxiv.org/pdf/2605.14442v1](https://arxiv.org/pdf/2605.14442v1)

## Abstract
Characterizing the physiological life boundaries of microbial strains, including viable temperature, pH, salinity, substrate utilization, and morphology, is central to biotechnology and ecology, yet traditionally requires exhaustive in vitro screening. Existing computational approaches either treat physiological traits as isolated supervised targets or repurpose biological foundation models as static encoders, leaving the genotype-to-physiology gap largely unbridged. We formulate microbial life-boundary prediction as a unified genome-to-physiology task and address it with a genome-conditioned, tool-augmented LLM agent. To support this task, we curate a strain-centric benchmark from IJSEM, NCBI, and BacDive covering 1,525 strains and 6,448 instances across viability intervals, environmental optima, substrate utilization, categorical traits, and morphology. Architecturally, the agent injects frozen LucaOne genome embeddings into a Qwen backbone via lightweight token fusion, and reasons over a similarity-based RAG module and a Genome-scale Metabolic Model (GEM) perturbation tool. We optimize the agent through a three-stage pipeline of gene-text alignment, agentic SFT on distilled trajectories, and GRPO with a novel counterfactual gene-grounding reward that reinforces the policy only when the authentic genome embedding causally improves correct-token generation relative to a zero-gene ablation. The resulting 4B-parameter agent matches or surpasses substantially larger frontier LLMs, with ablations confirming that genome-token fusion, dynamic tool use, and the counterfactual reward each yield distinct, significant gains.

## Full Text


<!-- PDF content starts -->

GGBound: A Genome-Grounded Agent for Microbial
Life-Boundary Prediction
Hanbo Huang1 *Xuan Gong1 *Jing Wang1 *Lei Bai2
Xiang Xiao1Weishu Zhao1Shiyu Liang1†
1Shanghai Jiao Tong University2Shanghai Artificial Intelligence Laboratory
{hhuang417, gongxuan0610, lsy18602808513}@sjtu.edu.cn
Abstract
Characterizing the physiological life boundaries of microbial strains, including
viable temperature, pH, salinity, substrate utilization, and morphology, is central to
biotechnology and ecology, yet traditionally requires exhaustivein vitroscreening.
Existing computational approaches either treat physiological traits as isolated
supervised targets or repurpose biological foundation models as static encoders,
leaving the genotype-to-physiology gap largely unbridged. We formulatemicrobial
life-boundary predictionas a unified genome-to-physiology task and address it
with a genome-conditioned, tool-augmented LLM agent. To support this task,
we curate a strain-centric benchmark from IJSEM, NCBI, and BacDive covering
1,525 strains and 6,448 instances across viability intervals, environmental optima,
substrate utilization, categorical traits, and morphology. Architecturally, the agent
injects frozen LucaOne genome embeddings into a Qwen backbone via lightweight
token fusion, and reasons over a similarity-based RAG module and a Genome-
scale Metabolic Model (GEM) perturbation tool. We optimize the agent through a
three-stage pipeline of gene–text alignment, agentic SFT on distilled trajectories,
and GRPO with a novelcounterfactual gene-groundingreward that reinforces
the policy only when the authentic genome embedding causally improves correct-
token generation relative to a zero-gene ablation. The resulting 4B-parameter agent
matches or surpasses substantially larger frontier LLMs, with ablations confirming
that genome-token fusion, dynamic tool use, and the counterfactual reward each
yield distinct, significant gains.
1 Introduction
Characterizing the physiologicallife boundariesof microbial strains, such as temperature, pH, and
substrate tolerances, is fundamental to biotechnology and ecology [ 1–3]. However, determining
these limits traditionally requires exhaustivein vitroscreening [ 4,5]. Despite the abundance of
genomic data [ 6], mapping genotypes to comprehensive physiological envelopes remains a formidable
predictive challenge, leaving the genotype-phenotype gap bridged largely by resource-intensive
manual experimentation.
Existing predictive methodologies offer incomplete solutions: conventional approaches treat physio-
logical traits as isolated targets using hand-engineered features [ 7–11], while biological foundation
models serve primarily as static encoders for narrow tasks [ 12–20]. To bridge this gap, we formulate
microbial life-boundary predictionas a unified genome-to-physiology paradigm solved via a genome-
conditioned, tool-augmented Large Language Model (LLM) agent. Given a strain genome giand
a physiological query qi, our agent interleaves semantic reasoning with biological tool execution
*Equal contribution.
†Corresponding author.
Preprint.arXiv:2605.14442v1  [cs.CY]  14 May 2026

Genome
Inputs
Lucaone
Embedding
GGBound
RethinkReason &Plan
Tool Call ObservationsRAG
DatabaseTool
Box
Similar Strain's
BoundaryGEM ToolMinimal Medium
Information
Answer
Query
Physiological Predictions
Phys. Boundary
Phys. Optimum
Metabolism
Categorical Phys.
Morphology
Training
PipelineStage 1
Gene-text Align.
Stage 2
Agentic SFT
Stage 3
Agentic RL
Correctness 
rewardFormat 
rewardTool-use 
rewardCounterfactual
gene-droundingFigure 1: Overview of GGBound. The agent conditions on LucaOne genome embeddings, integrates
RAG and GEM tool observations, and predicts microbial life boundaries through a three-stage training
pipeline with counterfactual gene-grounding.
to generate an observation trajectory Oi={o i,1, . . . , o i,T}, jointly predicting ˆyi=fθ(gi, qi, Oi).
This predictive target comprehensively encompasses growth ranges, optimal conditions, substrate
utilization, and morphological traits.
Figure 1 provides an overview of the proposed framework. We instantiate this framework through
three core contributions. First, we construct a strain-centric benchmark via an automated LLM
extraction pipeline over IJSEM, NCBI, and BacDive records, covering 1,525 unique strains and 6,448
task-specific instances. Second, we design a genome-grounded agent that injects LucaOne genome
embeddings into a Qwen backbone via token fusion and equips it with dynamic biological tools,
including a similarity-based Retrieval-Augmented Generation (RAG) module and a Genome-scale
Metabolic Model (GEM) perturbation tool for targeted nutrient perturbations. Third, we optimize the
agent through gene–text instruction tuning, agentic supervised fine-tuning, and Group Relative Policy
Optimization (GRPO) [ 21]. Crucially, the GRPO stage employs a counterfactual gene-grounding
reward that penalizes reliance on language priors and reinforces the policy only when the authentic
genome embedding causally improves accurate token generation.
Empirically, our 4B-parameter agent matches or outperforms substantially larger general-purpose
LLMs (e.g., DeepSeek-V3.2, DeepSeek-R1, GLM-4.7, Kimi-K2) on life-boundary prediction. Abla-
tions confirm that our CLS-based encoding, dynamic tool-use formulation, and counterfactual reward
each yield distinct, significant performance gains. In summary, our primary contributions are:
•Task Formulation & Benchmark Creation:We formulatemicrobial life-boundary predictionas
a unified genome-to-physiology task. To support this, we construct and release a comprehensive,
strain-centric benchmark curated from IJSEM, NCBI, and BacDive, which jointly evaluates viability
intervals, environmental optima, substrate utilization, categorical traits, and morphology.
•Genome-Conditioned Agent Architecture:We design a tool-augmented agent that seamlessly
injects LucaOne genome embeddings into a Qwen LLM backbone. The agent reasons over
a similarity-based RAG module and a GEM-based perturbation tool, effectively synthesizing
evidence across database lookups, embedding spaces, and metabolic simulations.
•Counterfactual RL Pipeline & Empirical Performance:We propose a three-stage training
paradigm: gene–text alignment, agentic SFT on distilled trajectories, and GRPO. Crucially, we
introduce acounterfactual gene-groundingreward that explicitly enforces causal dependence on the
genomic input. This pipeline yields a 4B-parameter agent that achieves performance competitive
with substantially larger frontier LLMs.
2 Related Work
Biological Sequence-to-Phenotype Modeling.Mapping biological sequences to phenotypic traits is
a central challenge in computational biology. Protein structure prediction has demonstrated that amino-
acid sequences contain sufficient signal for accurate three-dimensional structure prediction [ 22,23,
13], while protein language models show that large-scale sequence pretraining captures transferable
sequence–structure and sequence–function information [ 12,24]. In microbial genomics, phenotype
prediction has been studied using k-mer features, protein-family inventories, phyletic patterns, and
supervised learning methods, including tools such as Traitar and PhenotypeSeeker as well as recent
BacDive-based models for bacterial physiological traits [ 7–10]. Prior microbial phenotype predictors
2

usually optimize trait-specific supervised models or metabolic simulations, whereas our system
evaluates whether a genome-conditioned agent can combine representation similarity, database
evidence, and GEM-derived observations across heterogeneous physiological targets.
Biological Foundation Models and Genome-Grounded Agents.Biological foundation models
have enabled scalable representation learning across proteins, nucleotides, and genomes. Protein
models such as ESM and ProtTrans learn transferable representations from large-scale amino-acid
corpora [ 13,12,24], while nucleotide and genome-scale models such as DNABERT, DNABERT-
2, Nucleotide Transformer, HyenaDNA, GenSLM, Evo, and Evo 2 extend sequence modeling to
regulatory, mutational, evolutionary, and long-context genome-scale prediction [ 14–17,25,18,19].
LucaOne further unifies nucleic-acid and protein modeling within a single biological foundation
model [ 20]. However, these models are commonly used as static encoders, zero-shot scorers,
or backbones for task-specific predictors. Our work instead treats dense genome embeddings as
continuous, non-textual conditioning inputs for a tool-augmented LLM agent. Building on multimodal
LLM alignment and tool-use agents [ 26–30], we integrate genome-token fusion, biological tool use,
and GRPO-based optimization [ 21], training the agent to gather biological evidence and align its
predictions with the underlying genomic context.
3 Motivation
We study strain-level prediction of microbial physiological regimes, including growth-condition
ranges, optimal conditions, substrate utilization, and morphology, using genome-derived embeddings
and external biological evidence.
Microbial life-boundary prediction is an underexplored genome-to-physiology problem.Deter-
mining the environmental regimes under which microbial strains survive and grow is fundamental to
cultivation, experimental design, ecological characterization, and biological adaptation [ 1–3]. Such
characterization often requires extensive wet-lab screening over combinatorial spaces of temperature,
pH, salinity, pressure, nutrient availability, and other factors [ 4,5]. Although genome sequencing has
made strain-level molecular information increasingly accessible, existing genotype-to-phenotype stud-
ies typically formulate microbial traits as separate categorical or scalar prediction problems [ 11,9].
We instead formulate microbial life-boundary prediction as a broader genome-to-physiology problem,
where boundary-level viability regimes serve as the central prediction target and organize strain-level
physiological inference.
Tool observations provide a general interface for integrating biological evidence.Inferring
physiological boundaries from genomic information alone is challenging, as microbial viability is
governed by coupled constraints from evolutionary relatedness, cultivation history, environmental
context, phenotype annotations, and metabolic feasibility [ 6,31–33]. Rather than defining source-
specific modules for each type of evidence, we represent external biological information through a
unified sequence of tool observations, following recent tool-augmented agent formulations [ 28,29].
Given a genome representation giand a task query qi, the agent interacts with biological tools
and obtains an observation trajectory Oi={o i,1, . . . , o i,T}, where each observation provides task-
relevant evidence about the strain. The prediction objective is defined as
ˆyi=fθ(gi, qi, Oi).
This formulation provides a unified basis for incorporating heterogeneous biological evidence into
strain-level physiological prediction across diverse target spaces.
Genome-conditioned life-boundary agents can accelerate experimental discovery.The goal
of life-boundary prediction is to generate biologically plausible and experimentally actionable
hypotheses for poorly characterized strains, rather than merely reproduce annotations. By prioritizing
feasible cultivation conditions and excluding implausible regimes, accurate boundary predictions
can reduce the wet-lab search space for microbial characterization [ 4,5]. More broadly, linking
genomic information to viability regimes provides a computational framework for studying microbial
adaptation in extreme environments, including deep-sea, high-pressure, hypersaline, and acidic
habitats [ 1–3]. These considerations further motivate our strain-centric genome-physiology dataset
and tool-augmented agentic framework for microbial life-boundary prediction.
3

4 Data Collection and Benchmark Construction
Building a genome-conditioned life-boundary agent requires aligned strain-level genome-physiology
data. We therefore integrate distributed literature, genomic, and metabolic resources into a unified
corpus, from which we construct standardized benchmarks and agent training data. (Appendix A.)
4.1 Genome-Physiology Corpus Construction and Modality Alignment
Construction of a strain-centric genome-physiology corpus.We constructed a comprehensive
microbial physiology corpus by integrating heterogeneous biological resources, including IJSEM [ 34]
articles, NCBI [ 6] genome and protein records, and BacDive [ 31] phenotype annotations. Specifically,
we first extracted microbial strains and their associated physiological traits from IJSEM publications,
followed by systematic data cleaning and normalization. Using the curated strain identities, we
retrieved corresponding genomic and proteomic profiles from NCBI, alongside phenotype annotations
from BacDive. The resulting dataset consolidates heterogeneous multimodal sources into unified
strain-level records. Each record encapsulates strain identity, genomic and proteomic information,
morphology, metabolism, growth and cultivation conditions, and environmental context, thereby
supporting strain-centric analysis of genome-physiology relationships.
Raw strain data collection.To acquire raw strain-level physiological data, we developed a four-
stage LLM-based extraction pipeline powered by DeepSeek-V3.2 [ 35] and applied it to 18,498
IJSEM articles published prior to Jan. 16, 2026. The pipeline sequentially identifies strain-relevant
paragraphs and tables, isolates the primary strains under study, extracts their associated physiological
and metabolic traits, and applies an LLM-based quality scoring procedure to filter incomplete or
unreliable records. After filtering and normalization, the IJSEM-derived corpus yielded 88,927 high-
quality microbial strain records covering morphology, metabolic properties, cultivation protocols,
and isolation environments. To assess extraction quality, domain experts manually evaluated a subset
of the resulting records, confirming high fidelity and structural consistency.
Multi-modal data alignment.We aligned the literature-derived records with external genomic,
phenotypic, and metabolic databases. For strains extracted from IJSEM, we retrieved corresponding
genome and protein records from NCBI, identifying 43,649 matched instances and 18,964 unique
strains post-filtering. Utilizing scientific nomenclature and standardized strain identifiers, these
records were further mapped to BacDive annotations, yielding 17,796 successfully matched strains.
Concurrently, gene sequences were encoded using LucaOne [ 20] to generate gene-level embeddings,
which were subsequently aggregated into strain-level representations. Given that the classification
(CLS) token encapsulates sufficient global semantic information for downstream tasks such as
morphological classification [ 20], we directly utilize the CLS token from the genome-level LucaOne
embedding as a compact strain representation, yielding a 1×2560 dimensional vector that serves
as the genetic modality input for our agent. To empirically verify that this compact representation
preserves signal for physiological-boundary prediction, we conduct ablations against mean pooling,
gene-set pooling, protein-family inventories, and k-mer baselines.
Trait harmonization.Physiological traits are often reported with inconsistent terminology, measure-
ment units, and levels of granularity across the literature. We therefore systematically standardized
these descriptions. Continuous variables (e.g., temperature, pH, and salinity) were reformatted into
unified scalar or interval representations. Meanwhile, categorical attributes (e.g., cellular morphology,
motility, and oxygen tolerance) were rigorously mapped to controlled vocabularies developed in
consultation with biological domain experts.
4.2 Benchmark and Agent Training Data Construction
Benchmark construction.As summarized in Figure 2, we construct a strain-centric benchmark
covering five predictive capabilities: (1) physiological boundary prediction, (2) optimal-condition
prediction, (3) substrate utilization and metabolic capability prediction, (4) categorical physiological
trait prediction, and (5) cellular morphology prediction. The benchmark spans 1,525 strains and
6,448 instances, including 987physiological-boundary, 1,097 optimal-condition, 1,460 metabolic,
740categorical-physiology, and 1,772 morphology instances. Each instance pairs a genome-derived
representation with a target attribute. During evaluation, strain names are anonymized to prevent
4

395(40.0%)
352(35.7%)240
(24.3%)Growthtemperature
range
PH rangeSalinity range
Samples=987Phys. Boundary
383(34.9%)
353
(32.2%)361
(32.9%)Growthtemperature
optimum
PH optimumSalinityoptimum
Samples=1097Phys. Optimum
388(26.6%)
309
(21.2%)382(26.2%)381
(26.1%)Carbon source
Electron donorElectronacceptor/glyph1197itrogensource
Samples=1460Metabolism
369(49.9%)
214(28.9%)95
(12.8%)62
(8.4%)
Oxygentolerance
Energy typePhotosynthesistypeCarbon
fixation
pathway
Samples=740Categorical Phys.
358(20.2%)
355
(20.0%)
353
(19.9%)345(19.5%)361
(20.4%)Cell shape
Motility
FlagellaSporeformationGram stain
Samples=1772MorphologyFigure 2: Benchmark composition. The benchmark contains6,448strain-level instances across five
physiological prediction groups. Bars show task-level counts and proportions.
exploitation of memorized taxonomic knowledge, requiring predictions to rely on genomic features
and tool utilization. All benchmark samples are disjoint from downstream agent training data.
Data construction for modality fusion.LucaOne encodes genomic sequences into continuous
gene-level embeddings, which are not directly aligned with the natural-language representation
space of LLMs [ 36]. We therefore constructed genome-text modality-fusion data to bridge genomic
representations and textual physiological descriptions. Specifically, we reorganized IJSEM-derived
records into trait-specific question–answer pairs, where each example conditions on a strain identifier
and queries a specific physiological trait. In addition to single-trait QA pairs, we employed Qwen3.5-
27B [ 37] to generate longer-form, multi-trait QA pairs that synthesize coherent phenotypic profiles.
To prevent data leakage, the fields used for fusion are disjoint from the data utilized in agent training.
Data construction for agent training.The agent-training phase consists of supervised fine-tuning
(SFT) followed by reinforcement learning (RL). We constructed the training data primarily from
BacDive-derived records, leveraging their structured format and high-quality phenotype annotations.
We first filtered BacDive entries by standardized strain identifiers and retained 10,000 exactly matched
strain records. Similar to benchmark construction, we anonymized strain identities by replacing
true strain names with anonymous identifiers and genome-derived representations. This prevents the
model from relying on species-name memorization and requires the agent to infer target traits from
genome-conditioned features and tool observations. From these records, we randomly selected 8,000
unique strains for SFT data construction and 2,000 unique strains for the RL stage.
Trajectory distillation via teacher rollouts.To improve the agent’s tool-use capabilities, we
performed trajectory distillation using the 8,000 SFT-stage samples. For each sample, we first
queried a Qwen3.5-27B teacher agent to generate four candidate trajectories. During generation, the
teacher agent was allowed to invoke external tools when necessary to gather supporting evidence,
including BacDive records from similar strains and minimal-medium evidence for assessing viability
under specified growth conditions. Each trajectory consists of a sequence of tool invocations, the
corresponding tool observations, intermediate reasoning steps, and a final JSON-formatted prediction.
We then evaluated and selected the best trajectory for each sample according to prediction correctness
and consistency with the retrieved evidence. We separate answer-supervised correction from evidence-
grounded trajectory imitation. Repaired trajectories are not used to train tool-use rationales unless the
corrected answer is independently derivable from pre-answer observations.
Trajectory refinement and dataset finalization.For trajectories insufficiently grounded in tool
observations, we introduced a retry stage that enforced additional tool use, followed by re-evaluating
the resulting prediction. If a trajectory still failed to produce the correct answer, we used Qwen3.5-
27B to repair the final rationale and prediction while preserving the original tool invocations and
observations. Concretely, the model was provided with the initial trajectory, retrieved evidence, and
ground-truth label, and was asked to generate a corrected rationale consistent with the evidence and
target answer. We then merged all validated and repaired trajectories to construct the final distillation
dataset, containing 54,249 training samples.
5 Methodology
5.1 Genome-Conditioned Tool-Augmented Agent
Tool construction for genome-conditioned reasoning.We equip the agent with two biological
tools: a similarity-based RAG tool and a GEM-based perturbation tool. The RAG tool is built from
5

the synthetic SFT database containing field-level annotations for 8,000 strains. For each query strain,
we retrieve the top-3 nearest strains in the gene-embedding space measured using cosine similarity
and return their ground-truth annotations as tool observations. To prevent label leakage, the query
strain’s own ground-truth annotations are excluded during SFT data distillation.
For the GEM tool, we reconstruct strain-specific genome-scale metabolic models from protein se-
quences using CarveMe [ 38]. Each strain is instantiated with three biomass templates, corresponding
to Gram-positive bacteria, Gram-negative bacteria, and archaea, with the archaeal template adapted
from iAF692 [ 39,40]. We evaluate each model under six medium perturbations: removing sulfate
(SO2−
4), ferric iron ( Fe3+), nitrate ( NO−
3), nitrite ( NO−
2), oxygen ( O2), or all five components jointly.
These perturbations probe major biochemical and ecological constraints, including sulfur assimilation,
iron-dependent redox metabolism, nitrate/nitrite-associated nitrogen and respiratory pathways, and
oxygen-dependent respiration. For each biomass template and perturbation, we compute the minimal
medium supporting at least 10% of the maximum growth rate, yielding 18 GEM-derived observations
per strain for downstream physiological reasoning.
Genome-token modal fusion.To incorporate genomic information into the language model, we
adopt a lightweight genome-token fusion module to condition the language model on strain-level
genomic information. Following multimodal LLM alignment pipelines [ 26,27,30], we use LucaOne
as a frozen genome encoder and employ a two-layer MLP projector ϕ(·) to map the resulting genome
representation giinto the LLM token-embedding space. The projected vector zi=ϕ(g i)is inserted
as a continuous genome token alongside the textual instruction, tool observations, and task query,
enabling genomic conditioning without architectural modifications to the backbone LLM.
Agentic supervised fine-tuning.After genome-token alignment, we perform supervised fine-tuning
to distill tool-augmented physiological reasoning into a compact agent. Using the constructed strain-
centric corpus, we generate instruction trajectories with Qwen3.5-27B as the teacher model, where
each trajectory contains the genome token, task query, tool observations, tool-use reasoning process,
and final answer. We fine-tune Qwen3.5-4B as the student agent under a next-token prediction
objective with a context length of 16,384 tokens. Throughout agentic SFT, LucaOne and the modal-
fusion MLP are frozen, and only the student LLM is updated, preserving the genome-token alignment
while adapting the model to tool-conditioned life-boundary prediction.
5.2 Gene-Grounded Tool-Use Policy Optimization
Because teacher trajectories are generated without genome embeddings, SFT primarily teaches tool
syntax and evidence formatting. We therefore evaluate whether subsequent RL changes tool-selection
behavior as a function of genome embeddings. We further optimize it with GRPO [ 21]. Tool calls are
treated as policy actions, while tool observations are masked from the policy loss.
Sequence-level reward.For each sampled trajectory τi, we assign a composite reward r(τi) =
λjsonrjson+λcorrrcorr+λtoolrtool+λntrnt. The JSON reward rjsonenforces that the final response is
a single valid JSON object containing exactly the target field. The correctness reward rcorrevaluates
biological accuracy using field-specific criteria: interval targets are rewarded for covering the ground-
truth boundary while remaining compact, scalar targets are scored by normalized error, categorical
and morphological targets are evaluated after label canonicalization, and substrate-source targets are
scored by top- kset overlap. The tool-use reward rtoolencourages evidence acquisition according to a
scheduled tool-call target, while rntpenalizes incorrect direct answers that bypass tool observations.
The tool-use advantage Atool
i,tis obtained by applying the same group-relative normalization to
per-step tool-call shaping rewards rtool, and is assigned only to model-generated tool-call tokens (the
textual content of each tool invocation, excluding environment-inserted observations).
Group-relative advantage estimation.For each prompt, we sample Gtrajectories and compute a
group-relative advantage A(g)
i= (r(τ(g)
i)−mean jr(τ(j)
i))/(std jr(τ(j)
i) +ϵ) . This normalization
compares candidate trajectories under the same biological query and avoids the need for a learned
value function. The resulting sequence advantage is assigned to final-answer tokens, while tool-call
tokens receive separate tool-use shaping rewards.
Counterfactual gene-grounding reward.A central challenge in genome-conditioned reasoning is
that the policy may generate plausible answers from language priors or tool observations without
effectively using the genome embedding. To encourage genuine genomic conditioning, we compute
a counterfactual gene-grounding reward through two teacher-forced forward passes over the same
6

sampled completion. In the first pass, the model is conditioned on the real genome embedding gi; in
the second pass, giis replaced with an ablated zero-gene embedding 0. For each final-answer token
yi,t, we compute the log-probability gap ∆gene
i,t= logπ θ(yi,t|ci,t, gi)−logπ θ(yi,t|ci,t,0), where
ci,tdenotes the textual context, prior generated tokens, tool observations, and task query before token
t. The gene-grounding term is then defined as Agene
i,t= max(r corr(τi),0)·clip(∆gene
i,t,−c, c)·1[t∈
answer] , where c >0 is a fixed clipping constant that bounds the per-token log-probability gap and
prevents a small number of high-magnitude tokens from dominating the advantage. Thus, the model
is rewarded only when the real genome embedding increases the likelihood of biologically correct
answer tokens relative to the zero-gene counterfactual, preventing reinforcement of gene-dependent
but incorrect predictions.
Optimization objective.The final token-level advantage is Ai,t=A i1[t∈answer] +λ toolAtool
i,t+
λgeneAgene
i,t, where the tool-use term is applied to model-generated tool-call tokens and the gene-
grounding term is applied only to final-answer tokens. Let mpol
i,tbe a policy mask that excludes
environment-inserted tool observations. Following GRPO, we optimize
JGRPO (θ) =Eq,{τi}G
i=1h
1
GPG
i=11
|τi|P|τi|
t=1mpol
i,t(min[ρ i,tAi,t,clip(ρ i,t,1−ϵ,1 +ϵ)A i,t]−βKL i,t)i
.
where ρi,t=πθ(yi,t|ci,t, gi)/πθold(yi,t|ci,t, gi)andKLi,t=D KL[πθ(· |c i,t, gi)∥πref(· |c i,t, gi)].
This objective optimizes structured output validity, biological correctness, efficient tool use, and
explicit dependence on the genome representation. During RL, the LucaOne encoder and genome-
fusion MLP remain frozen, and only the Qwen3.5-4B policy is updated.
6 Experiments
6.1 Experimental Setup
In this subsection, we detailed our experimental settings, with additional details in Appendix B.
Models.We build our gene-grounded agent using Qwen3.5-4B [ 36,37], which serves as the language
backbone for genome-modality fusion. We inject genome embeddings through the proposed fusion
module and train the resulting agent with the three-stage pipeline described above. To contextualize
the performance of our trained agent against stronger general-purpose LLMs, we additionally evaluate
several larger frontier baselines, including DeepSeek-V3.2 [ 35], DeepSeek-R1 [ 41], GLM-4.7 [ 42],
and Kimi-K2 [ 43]. These models are used as strong text-based baselines for assessing whether
explicit gene-grounded training provides benefits beyond scaling general language-model capacity.
Implementation details.We train the model in three stages. We first conduct full-parameter gene-
text instruction tuning from a gene-aligned Qwen3.5-4B checkpoint for one epoch, using sequence
length 1,024, effective batch size 256, and learning rate 2×10−5. We then perform agent SFT on
distilled tool-use trajectories for one epoch, with sequence length 16,384, effective batch size 128,
and learning rate 5×10−6. Finally, we optimize the SFT agent with GRPO on BacDive-derived
prompts, using a group size of 4 with a global rollout batch size of 64 and learning rate 1×10−6.
During RL, we allow at most 5 tool-calling iterations, freeze the genome-fusion module, and update
only the language-policy parameters.
Evaluation metrics.We use task-specific metrics for heterogeneous microbial traits. In the following
experiments, we primarily report interval coverage rate (ICR) for physiological boundaries, RMSE
for optimal conditions, mAP@5 for substrate-source traits, and accuracy for categorical physiological
and morphological traits.
6.2 Main Results
GGBound shows a consistent progression from passive genome conditioning to selective,
genome-grounded reasoning.Table 1 compares the base model, genome–text fusion model, SFT
agent, and RL-optimized GGBound across all benchmark tasks. The RL agent achieves the best
performance on every target, while the SFT agent is consistently second-best, indicating that agentic
training is the main driver of performance gains. These results suggest that life-boundary prediction
requires more than access to genomic representations; it also benefits from learning how to acquire,
filter, and synthesize biological evidence during generation.
7

Table 1: Main results on genome-conditioned microbial physiology prediction. We compare the base
language model, genome–text modal fusion model, agent SFT model, and RL-optimized agent across
physiological boundary, optimal-condition, metabolism, categorical physiology, and morphology
tasks. Best results are shown inbold; second-best results are underlined .
Tasks Field Metric Base Model Fusion Model SFT Agent RL Agent
Physiological
BoundaryGrowth temperature range ICR↑0.157±0.0070.163±0.0940.179 ±0.0010.182±0.003
pH range ICR↑0.287±0.0110.296±0.1460.458 ±0.0010.497±0.067
Salinity range ICR↑0.466±0.0240.471±0.0890.781 ±0.0510.818±0.045
Physiological
OptimumOptimal growth temperature RMSE↓8.570±0.2338.449±0.5806.449 ±0.0936.373±0.113
Optimal pH RMSE↓1.124±0.0401.077±0.3830.736 ±0.0000.726±0.017
Optimal salinity RMSE↓3.526±0.0743.441±0.0802.553 ±0.0142.532±0.029
MetabolismCarbon source mAP@5↑0.000±0.0670.000±0.0000.040 ±0.0070.054±0.022
Electron acceptor mAP@5↑0.003±0.0480.003±0.0060.233 ±0.0180.249±0.018
Electron donor mAP@5↑0.000±0.0860.000±0.0000.095 ±0.0140.118±0.027
Nitrogen source mAP@5↑0.003±0.0320.003±0.0020.209 ±0.0010.217±0.012
Categorical
PhysiologyCarbon fixation pathway Accuracy↑0.081±0.0420.097±0.0370.414 ±0.0090.419±0.000
Energy type Accuracy↑0.047±0.1550.042±0.0310.500 ±0.0000.505±0.008
Oxygen tolerance Accuracy↑0.271±0.0290.244±0.1450.541 ±0.0080.549±0.012
Photosynthesis type Accuracy↑0.168±0.1180.263±0.0940.611 ±0.0110.625±0.016
MorphologyCell shape Accuracy↑0.684±0.0240.670±0.3530.800 ±0.0020.802±0.003
Flagella Accuracy↑0.133±0.1210.125±0.0750.536 ±0.0060.542±0.004
Gram stain Accuracy↑0.529±0.1330.681±0.3800.853 ±0.0000.855±0.003
Motility Accuracy↑0.597±0.0330.659±0.3010.700 ±0.0020.701±0.000
Spore formation Accuracy↑0.699±0.0570.803±0.3370.838 ±0.0000.839±0.002
Genome–text fusion provides limited standalone gains, whereas agentic SFT produces the
largest transition in capability.Modal fusion improves several continuous targets over the base
model, including optimal growth temperature and optimal salinity, suggesting that frozen LucaOne
embeddings encode useful environmental-preference signals. However, fusion remains unreliable
on many categorical and morphology tasks, where performance often lags behind the base model.
In contrast, agentic SFT substantially improves nearly all task groups, raising salinity-range ICR
from 0.471 to0.781 , electron acceptor mAP@5 from 0.003 to0.233 , oxygen-tolerance accuracy
from 0.244 to0.541 . This indicates that distilled trajectories help the model convert genomic and
tool-derived evidence into standardized physiological predictions.
Reinforcement learning yields smaller but systematic gains over a strong SFT agent.Compared
with SFT, the RL agent improves all targets, including pH-range ICR from 0.458 to0.497 , salinity-
range ICR from 0.781 to0.818 , optimal pH RMSE from 0.736 to0.726 , and nitrogen-source mAP@5
from 0.209 to0.217 . Gains are also consistent across categorical and morphology tasks, such as
photosynthesis type ( 0.611 to0.625 ), carbon fixation pathway ( 0.414 to0.419 ), and flagella ( 0.536 to
0.542 ). Although these improvements are often modest, their consistency across heterogeneous targets
supports the role of GRPO and the counterfactual gene-grounding reward in refining biologically
grounded predictions rather than merely changing output format.
Table 2: Averaged tool-use.
Model RAG GEM Avg.
Base 1.052 5.914 6.966
Fusion 1.024 5.632 6.656
SFT Agent 0.994 0.580 1.574
RL Agent 0.993 0.739 1.732Tool-use statistics indicate that better performance comes from
more selective evidence acquisition rather than more tool calls.
As shown in Table 2, the base and fusion models invoke substantially
more tools on average, largely due to frequent GEM calls, but still
underperform the agentic models. Agentic SFT sharply reduces
average tool usage while improving performance, suggesting that
trajectory supervision teaches a more efficient evidence-acquisition
policy. RL slightly increases tool usage relative to SFT, especially
GEM calls, while achieving the best overall accuracy, indicating that policy optimization improves
when and how biological tools are used. Overall, the results suggest that genome-token fusion
provides an initial grounding signal, agentic SFT establishes effective tool-conditioned reasoning,
and counterfactual gene-grounded RL further calibrates the policy toward accurate strain-level
physiological prediction.
8

GGB-w/oGGB-w/DS-V3.2DS-R1GLM-4.7Kimi-K20.00.20.40.60.81.0Average performanceGGBound Frontier LLMs0.50
0.48
0.34
0.37
0.52
0.38
Avg. ICR ↑Phys. Boundary
GGB-w/oGGB-w/DS-V3.2DS-R1GLM-4.7Kimi-K20123GGBound Frontier LLMs3.21
2.32
3.09
3.20
2.88
2.86
Avg. RMSE ↓Phys. Optimum
GGB-w/oGGB-w/DS-V3.2DS-R1GLM-4.7Kimi-K20.00.20.40.60.81.0GGBound Frontier LLMs0.16
0.53
0.42
0.41
0.47
0.43
Avg. mAP@5 ↑Metabolism
GGB-w/oGGB-w/DS-V3.2DS-R1GLM-4.7Kimi-K20.00.20.40.60.81.0GGBound Frontier LLMs0.52
0.81
0.68
0.65
0.59
0.70
Avg. Accuracy ↑Categorical Phys.
GGB-w/oGGB-w/DS-V3.2DS-R1GLM-4.7Kimi-K20.00.20.40.60.81.0GGBound Frontier LLMs0.75
0.86
0.77
0.76
0.74
0.79
Avg. Accuracy ↑MorphologyFigure 3: Average performance under strain-name evaluation. GGB-w/o and GGB-w/ denote
GGBound without and with strain-name input, respectively. Frontier LLMs are directly prompted
with strain names, while GGBound uses genome conditioning and GEM access.
6.3 Strain-Name Priors and Frontier LLM Comparison
Strain-name input provides a complementary source of taxonomic prior knowledge.Figure 3
compares GGBound with and without strain-name information against frontier LLMs queried directly
with strain names. This setting differs from the anonymized benchmark because strain names
expose species- or taxon-level cues, allowing models to exploit biological knowledge stored during
pretraining. For GGBound, we retain genome-conditioned inference and allow GEM tool calls; for
frontier LLMs, we directly query each model using the strain name and target field, since these
models may already encode substantial microbial knowledge internally. (Appendix C.2 and C.3).
Name-based priors substantially improve GGBound on most physiological categories.Adding
strain names improves GGBound on physiological optima, metabolism, categorical physiology, and
morphology. The largest gains appear on metabolism and categorical physiology, where average
mAP@5 increases from 0.16 to0.53 and average accuracy increases from 0.52 to0.81, respectively.
These gains suggest that strain names activate useful taxonomic priors that complement genome-
derived and GEM-derived evidence. The only exception is physiological-boundary prediction, where
average ICR slightly decreases from 0.50 to0.48, indicating that name-based priors can sometimes
conflict with genome-conditioned or tool-derived signals for heterogeneous interval annotations.
GGBound remains competitive with frontier LLMs even when those models receive strain
names.Despite the favorable setting for frontier LLMs, GGBound with strain-name input achieves
the best average performance on physiological optima, metabolism, categorical physiology, and
morphology. It outperforms the strongest frontier LLM average on metabolism ( 0.53 vs.0.47),
categorical physiology ( 0.81 vs.0.70), and morphology ( 0.86 vs.0.79), while achieving the lowest
average RMSE on physiological optima ( 2.32). GLM-4.7 performs best on physiological-boundary
ICR, suggesting that boundary ranges may be more strongly influenced by broad taxonomic priors
or interval-calibration effects. Overall, this comparison shows that strain names can substantially
raise performance by activating internal biological knowledge, while GGBound further benefits from
integrating such priors with genome-conditioned reasoning and GEM-based evidence.
7 Conclusion, Broader Impacts, and Limitations
Conclusion.We introduced GGBOUND, a genome-grounded, tool-augmented LLM agent for
microbial life-boundary prediction. We formulate strain-level physiology prediction as a unified
genome-to-physiology task and construct a strain-centric benchmark covering physiological bound-
aries, optima, metabolic capabilities, categorical traits, and morphology. GGBOUNDcombines
LucaOne genome embeddings with RAG and GEM tool observations, and is trained through gene–
text alignment, agentic SFT, and GRPO with a counterfactual gene-grounding reward. Results show
that agentic training substantially improves genome-conditioned prediction, while reinforcement
learning further refines biologically grounded outputs.
Broader Impacts.Genome-grounded life-boundary prediction can help prioritize cultivation condi-
tions, reduce wet-lab screening cost, and support microbial ecology and biotechnology. The proposed
benchmark also provides a standardized setting for evaluating whether biological agents reason from
genomic evidence rather than taxonomic memorization. Model outputs should nevertheless be treated
9

as hypotheses, not experimental instructions, and downstream use should follow expert review and
appropriate biosafety standards.
Limitations.A key limitation is the highly compressed genome representation used for modality
fusion. Each strain is represented by a single LucaOne-derived embedding projected into the LLM
token space, which may lose substantial gene-level, pathway-level, and structural information needed
for fine-grained physiological prediction. This bottleneck is particularly problematic for substrate
utilization, metabolic capability, and stress-response traits, where localized genomic evidence may be
essential. Future work should explore richer genome-to-LLM interfaces, including multiple genome
tokens, gene-set-aware pooling, and adaptive retrieval over gene features.
10

References
[1]Lynn J Rothschild and Rocco L Mancinelli. Life in extreme environments.Nature, 409(6823):
1092–1101, 2001.
[2]Jesse P Harrison, Nicolas Gheeraert, Dmitry Tsigelnitskiy, and Charles S Cockell. The limits
for life under multiple extremes.Trends in microbiology, 21(4):204–212, 2013.
[3]Nancy Merino, Heidi S Aronson, Diana P Bojanova, Jayme Feyhl-Buska, Michael L Wong, Shu
Zhang, and Donato Giovannelli. Living at the extremes: extremophiles and the limits of life in
a planetary context.Frontiers in microbiology, 10:447668, 2019.
[4]D Nichols, N Cahoon, EM Trakhtenberg, L Pham, A Mehta, A Belanger, Tanya Kanigan,
Kim Lewis, and SS2849220 Epstein. Use of ichip for high-throughput in situ cultivation of
“uncultivable” microbial species.Applied and environmental microbiology, 76(8):2445–2450,
2010.
[5]Jean-Christophe Lagier, Grégory Dubourg, Matthieu Million, Frédéric Cadoret, Melhem Bilen,
Florence Fenollar, Anthony Levasseur, Jean-Marc Rolain, Pierre-Edouard Fournier, and Didier
Raoult. Culturing the human microbiota and culturomics.Nature Reviews Microbiology, 16(9):
540–550, 2018.
[6]Eric W Sayers, Jeffrey Beck, Evan E Bolton, Devon Bourexis, James R Brister, Kathi Canese,
Donald C Comeau, Kathryn Funk, Sunghwan Kim, William Klimke, et al. Database resources
of the national center for biotechnology information.Nucleic acids research, 49(D1):D10–D17,
2021.
[7]Aaron Weimann, Kyra Mooren, Jeremy Frank, Phillip B Pope, Andreas Bremges, and Alice C
McHardy. From genomes to phenotypes: Traitar, the microbial trait analyzer.MSystems, 1(6):
10–1128, 2016.
[8]Erki Aun, Age Brauer, Veljo Kisand, Tanel Tenson, and Maido Remm. A k-mer-based method
for the identification of phenotype-associated genomic biomarkers and predicting phenotypes
of sequenced bacteria.PLoS computational biology, 14(10):e1006434, 2018.
[9]Signe T Karlsen, Martin H Rau, Benjamín J Sánchez, Kristian Jensen, and Ahmad A Zeidan.
From genotype to phenotype: computational approaches for inferring microbial traits relevant
to the food industry.FEMS Microbiology Reviews, 47(4):fuad030, 2023.
[10] Julia Koblitz, Lorenz Christian Reimer, Rüdiger Pukall, and Jörg Overmann. Predicting
bacterial phenotypic traits through improved machine learning using high-quality, curated
datasets.Communications Biology, 8(1):897, 2025.
[11] Alexandre Drouin, Gaël Letarte, Frédéric Raymond, Mario Marchand, Jacques Corbeil, and
François Laviolette. Interpretable genotype-to-phenotype classifiers with performance guaran-
tees.Scientific reports, 9(1):4071, 2019.
[12] Alexander Rives, Joshua Meier, Tom Sercu, Siddharth Goyal, Zeming Lin, Jason Liu, Demi Guo,
Myle Ott, C Lawrence Zitnick, Jerry Ma, et al. Biological structure and function emerge from
scaling unsupervised learning to 250 million protein sequences.Proceedings of the national
academy of sciences, 118(15):e2016239118, 2021.
[13] Zeming Lin, Halil Akin, Roshan Rao, Brian Hie, Zhongkai Zhu, Wenting Lu, Nikita Smetanin,
Robert Verkuil, Ori Kabeli, Yaniv Shmueli, et al. Evolutionary-scale prediction of atomic-level
protein structure with a language model.Science, 379(6637):1123–1130, 2023.
[14] Yanrong Ji, Zhihan Zhou, Han Liu, and Ramana V Davuluri. Dnabert: pre-trained bidirectional
encoder representations from transformers model for dna-language in genome.Bioinformatics,
37(15):2112–2120, 2021.
[15] Zhihan Zhou, Yanrong Ji, Weijian Li, Pratik Dutta, Ramana Davuluri, and Han Liu. Dnabert-
2: Efficient foundation model and benchmark for multi-species genome.arXiv preprint
arXiv:2306.15006, 2023.
11

[16] Hugo Dalla-Torre, Liam Gonzalez, Javier Mendoza-Revilla, Nicolas Lopez Carranza,
Adam Henryk Grzywaczewski, Francesco Oteri, Christian Dallago, Evan Trop, Bernardo P
De Almeida, Hassan Sirelkhatim, et al. Nucleotide transformer: building and evaluating robust
foundation models for human genomics.Nature Methods, 22(2):287–297, 2025.
[17] Eric Nguyen, Michael Poli, Marjan Faizi, Armin Thomas, Michael Wornow, Callum Birch-
Sykes, Stefano Massaroli, Aman Patel, Clayton Rabideau, Yoshua Bengio, et al. Hyenadna:
Long-range genomic sequence modeling at single nucleotide resolution.Advances in neural
information processing systems, 36:43177–43201, 2023.
[18] Eric Nguyen, Michael Poli, Matthew G Durrant, Brian Kang, Dhruva Katrekar, David B Li,
Liam J Bartie, Armin W Thomas, Samuel H King, Garyk Brixi, et al. Sequence modeling and
design from molecular to genome scale with evo.Science, 386(6723):eado9336, 2024.
[19] Garyk Brixi, Matthew G Durrant, Jerome Ku, Mohsen Naghipourfar, Michael Poli, Gwanggyu
Sun, Greg Brockman, Daniel Chang, Alison Fanton, Gabriel A Gonzalez, et al. Genome
modelling and design across all domains of life with evo 2.Nature, pages 1–13, 2026.
[20] Yong He, Pan Fang, Yongtao Shan, Yuanfei Pan, Yanhong Wei, Yichang Chen, Yihao Chen,
Yi Liu, Zhenyu Zeng, Zhan Zhou, et al. Generalized biological foundation model with unified
nucleic acid and protein language.Nature Machine Intelligence, 7(6):942–953, 2025.
[21] Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang,
Mingchuan Zhang, Y . K. Li, Y . Wu, and Daya Guo. Deepseekmath: Pushing the limits of
mathematical reasoning in open language models, 2024. URL https://arxiv.org/abs/
2402.03300.
[22] John Jumper, Richard Evans, Alexander Pritzel, Tim Green, Michael Figurnov, Olaf Ron-
neberger, Kathryn Tunyasuvunakool, Russ Bates, Augustin Žídek, Anna Potapenko, et al.
Highly accurate protein structure prediction with alphafold.nature, 596(7873):583–589, 2021.
[23] Minkyung Baek, Frank DiMaio, Ivan Anishchenko, Justas Dauparas, Sergey Ovchinnikov,
Gyu Rie Lee, Jue Wang, Qian Cong, Lisa N Kinch, R Dustin Schaeffer, et al. Accurate
prediction of protein structures and interactions using a three-track neural network.Science,
373(6557):871–876, 2021.
[24] Ahmed Elnaggar, Michael Heinzinger, Christian Dallago, Ghalia Rehawi, Yu Wang, Llion
Jones, Tom Gibbs, Tamas Feher, Christoph Angerer, Martin Steinegger, et al. Prottrans: toward
understanding the language of life through self-supervised learning.IEEE transactions on
pattern analysis and machine intelligence, 44(10):7112–7127, 2021.
[25] Maxim Zvyagin, Alexander Brace, Kyle Hippe, Yuntian Deng, Bin Zhang, Cindy Orozco
Bohorquez, Austin Clyde, Bharat Kale, Danilo Perez-Rivera, Heng Ma, et al. Genslms:
Genome-scale language models reveal sars-cov-2 evolutionary dynamics.The International
Journal of High Performance Computing Applications, 37(6):683–705, 2023.
[26] Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi. Blip-2: Bootstrapping language-image
pre-training with frozen image encoders and large language models. InInternational conference
on machine learning, pages 19730–19742. PMLR, 2023.
[27] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning.Advances
in neural information processing systems, 36:34892–34916, 2023.
[28] Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik R Narasimhan, and Yuan
Cao. React: Synergizing reasoning and acting in language models. InThe eleventh international
conference on learning representations, 2022.
[29] Timo Schick, Jane Dwivedi-Yu, Roberto Dessì, Roberta Raileanu, Maria Lomeli, Eric Hambro,
Luke Zettlemoyer, Nicola Cancedda, and Thomas Scialom. Toolformer: Language models
can teach themselves to use tools.Advances in neural information processing systems, 36:
68539–68551, 2023.
12

[30] Yujia Wang, Jihong Guan, Wengen Li, Shuigeng Zhou, and Xuhong Wang. Biobridge: Bridging
proteins and language for enhanced biological reasoning with llms. In2025 IEEE International
Conference on Bioinformatics and Biomedicine (BIBM), pages 70–75. IEEE, 2025.
[31] Isabel Schober, Julia Koblitz, Joaquim Sardà Carbasse, Christian Ebeling, Marvin Leon Schmidt,
Adam Podstawka, Rohit Gupta, Vinodh Ilangovan, Javad Chamanara, Jörg Overmann, et al.
Bac dive in 2025: the core database for prokaryotic strain data.Nucleic acids research, 53(D1):
D748–D756, 2025.
[32] Ines Thiele and Bernhard Ø Palsson. A protocol for generating a high-quality genome-scale
metabolic reconstruction.Nature protocols, 5(1):93–121, 2010.
[33] Laurent Heirendt, Sylvain Arreckx, Thomas Pfau, Sebastián N Mendoza, Anne Richelle, Almut
Heinken, Hulda S Haraldsdóttir, Jacek Wachowiak, Sarah M Keating, Vanja Vlasov, et al.
Creation and analysis of biochemical constraint-based models using the cobra toolbox v. 3.0.
Nature protocols, 14(3):639–702, 2019.
[34] Microbiology Society. International Journal of Systematic and Evolutionary Microbiology,
2026. URL https://www.microbiologyresearch.org/content/journal/ijsem . Offi-
cial journal of record for novel prokaryotic taxa.
[35] Aixin Liu, Aoxue Mei, Bangcai Lin, Bing Xue, Bingxuan Wang, Bingzheng Xu, Bochao Wu,
Bowei Zhang, Chaofan Lin, Chen Dong, et al. Deepseek-v3. 2: Pushing the frontier of open
large language models.arXiv preprint arXiv:2512.02556, 2025.
[36] An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu,
Chang Gao, Chengen Huang, Chenxu Lv, et al. Qwen3 technical report.arXiv preprint
arXiv:2505.09388, 2025.
[37] Qwen Team. Qwen3.5: Towards native multimodal agents, February 2026. URL https:
//qwen.ai/blog?id=qwen3.5.
[38] Daniel Machado, Sergej Andrejev, Melanie Tramontano, and Kiran Raosaheb Patil. Fast auto-
mated reconstruction of genome-scale metabolic models for microbial species and communities.
Nucleic acids research, 46(15):7542–7553, 2018.
[39] Adam M Feist, Johannes CM Scholten, Bernhard Ø Palsson, Fred J Brockman, and Trey Ideker.
Modeling methanogenesis with a genome-scale metabolic reconstruction of methanosarcina
barkeri.Molecular systems biology, 2, 2006.
[40] Zachary A King, Justin Lu, Andreas Dräger, Philip Miller, Stephen Federowicz, Joshua A
Lerman, Ali Ebrahim, Bernhard O Palsson, and Nathan E Lewis. Bigg models: A platform for
integrating, standardizing and sharing genome-scale models.Nucleic acids research, 44(D1):
D515–D522, 2016.
[41] Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Peiyi Wang, Qihao Zhu, Runxin Xu,
Ruoyu Zhang, Shirong Ma, Xiao Bi, et al. Deepseek-r1: Incentivizing reasoning capability in
llms via reinforcement learning.arXiv preprint arXiv:2501.12948, 2025.
[42] Aohan Zeng, Xin Lv, Qinkai Zheng, Zhenyu Hou, Bin Chen, Chengxing Xie, Cunxiang Wang,
Da Yin, Hao Zeng, Jiajie Zhang, et al. Glm-4.5: Agentic, reasoning, and coding (arc) foundation
models.arXiv preprint arXiv:2508.06471, 2025.
[43] Kimi Team, Yifan Bai, Yiping Bao, Y Charles, Cheng Chen, Guanduo Chen, Haiting Chen,
Huarong Chen, Jiahao Chen, Ningxin Chen, et al. Kimi k2: Open agentic intelligence.arXiv
preprint arXiv:2507.20534, 2025.
[44] Yaowei Zheng, Richong Zhang, Junhao Zhang, Yanhan Ye, Zheyan Luo, Zhangchi Feng, and
Yongqiang Ma. Llamafactory: Unified efficient fine-tuning of 100+ language models. In
Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics
(Volume 3: System Demonstrations), Bangkok, Thailand, 2024. Association for Computational
Linguistics. URLhttp://arxiv.org/abs/2403.13372.
13

[45] Hugging Face. Open r1: A fully open reproduction of deepseek-r1, January 2025. URL
https://github.com/huggingface/open-r1.
[46] Leandro von Werra, Younes Belkada, Lewis Tunstall, Edward Beeching, Tristan Thrush, Nathan
Lambert, Shengyi Huang, Kashif Rasul, and Quentin Gallouédec. TRL: Transformers Rein-
forcement Learning, 2020. URLhttps://github.com/huggingface/trl.
14

A Dataset and Benchmark Construction
A.1 IJSEM Extraction Pipeline
We constructed the IJSEM-derived strain-physiology corpus through a four-stage LLM-assisted
extraction pipeline. The four stages correspond to: (i) evidence retrieval from full IJSEM articles,
(ii) identification of primary and comparison strains, (iii) bounded strain-level trait extraction, and
(iv) LLM-based extraction quality scoring. The goal of this pipeline is not to ask an LLM to directly
hallucinate structured biological labels from article titles or taxonomic names, but to progressively
convert verbatim article evidence into auditable strain-level records. Each stage is constrained by
explicit-only extraction rules and produces intermediate artifacts that can be inspected independently.
Input articles.The raw input consists of IJSEM articles represented as structured JSON files
containing title, abstract, main body, figures, tables, and paper links when available. For each article,
we concatenate the abstract and main text. If tables are available, we append table content to the
article input in a marked table block. This is important because IJSEM taxonomic descriptions often
report strain-level physiology, growth ranges, substrate utilization, and morphology in tables rather
than narrative paragraphs.
Stage 1: evidence retrieval from full articles.The first stage performs coarse evidence retrieval.
Given the full article text and extracted tables, the LLM is instructed to retrieve only text that reports
intrinsic strain- or species-level factual information. The output of this stage is not yet a structured
database record. Instead, the model returns verbatim article spans, including paragraphs and tables,
that contain information relevant to strain characterization.
The retrieval target includes species names, strain identifiers, type-strain markers, isolation source,
habitat, geographic location, culture collection identifiers, genome or sequence accessions, growth
conditions, physiological traits, biochemical properties, substrate utilization, carbon source profiles,
electron donors and acceptors, enzyme activities, and metabolic capabilities. Growth medium or
medium composition is retained when it is biologically informative, for example when it indicates
which substrates, nutrients, or energy sources the strain can use.
We use strict retention rules to prevent loss of attribution context. If a paragraph contains any strain-
or species-level physiological, biochemical, metabolic, or growth property, the entire paragraph is
retained verbatim. The model is explicitly prohibited from extracting only individual sentences from
such paragraphs. Similarly, if a table contains any information about the target strain or species,
the entire table is retained, including caption, body, row and column labels, and footnotes. This
design avoids fragmenting table-derived evidence and preserves the mapping between strain labels
and properties. If no relevant biological content is found, the model returns a fixed null marker, and
the article is excluded from downstream extraction.
Stage 2: strain identification and inventory construction.The second stage operates only on the
Stage-1 retrieved evidence. Its purpose is to identify all biological strains mentioned in the article
and separate them into two groups: paper_strains , which are the strains proposed, described, or
directly studied by the paper, and comparison_strains , which are reference or comparison strains
used for taxonomic, phylogenetic, phenotypic, or biochemical comparison.
The Stage-2 prompt requires the model to output the most complete strain names explicitly supported
by the article. The model may combine genus, species, nomenclatural qualifier, and strain identifier
only when all components are explicitly stated in the text or table captions. It is not allowed to infer
missing taxonomy, expand abbreviations without explicit support, or treat placeholders as strains
unless the article explicitly defines them as strain names. Table captions are given priority when they
define full strain names or mappings between abbreviated table labels and strains.
We also apply a conservative deduplication rule. If multiple mentions refer to the same strain
identifier, only the most complete explicitly supported name is retained. For example, if both a bare
strain identifier and a full genus–species–strain expression appear, the full expression is retained.
Nomenclatural qualifiers such assp. nov.,gen. nov., andcomb. nov.are preserved at this stage
because they are part of the source article’s taxonomic statement. The output of this stage is a strict
JSON object with uniquepaper_strainsandcomparison_strainslists.
Stage 3: bounded strain-level trait extraction.The third stage extracts structured strain-level
properties. Unlike Stage 2, which identifies strains, Stage 3 is bounded by the paper_strains list.
15

The model is given the Stage-1 evidence and the Stage-2 primary strain list, and is instructed to
extract properties only for those primary strains. It is not allowed to create new strains, rename strains,
or extract benchmark labels for comparison strains. This bounded design is critical for preventing
properties of reference organisms from being incorrectly assigned to the newly described strains.
For every primary strain, the model outputs a JSON entry keyed by the exact strain name or identifier
from Stage 2. If no properties are explicitly available for a strain, the output still contains the strain
identifier and an empty auxiliary field. This ensures that the structured output remains aligned with
the strain inventory.
The extraction schema covers environmental origin, genome and sequence information, morphology,
growth boundaries, optimal conditions, metabolism, and other strain-intrinsic traits. The target fields
include habitat, latitude and longitude, water depth, environment type, sampling site, sampling depth,
sampling temperature, sampling pressure, geographic region, taxonomy, top-hit taxon, sequence
similarity, genome accession, 16S rRNA accession or sequence, GC content, genome size, plasmid
presence, reported functional genes, cell shape, cell size, motility, flagella, spore formation, Gram
stain, photosynthesis type, photopigments, light conditions, medium type, isolation medium, isolation
temperature, enrichment pressure, enrichment time, growth temperature optimum, growth temperature
range, pH optimum, pH range, salinity optimum, salinity range, growth pressure, doubling time,
energy type, electron donor, electron acceptor, carbon fixation pathway, carbon source, nitrogen
source, oxygen tolerance, antibiotic sensitivity, antibiotic resistance, major fatty acids, and respiratory
quinones.
The extractor follows strict attribution rules. A property is assigned to a strain only if the Stage-1
evidence explicitly links that property to the strain. The linkage may appear locally in a sentence,
through a table column, through a caption-defined strain mapping, or through a clear contextual scope.
Global or group-level statements are not assigned to individual strains unless the article explicitly
supports such assignment. Values are copied verbatim, including symbols, units, ranges, punctuation,
and qualifiers, except for minimal character-level repair of obvious encoding artifacts introduced by
PDF or HTML conversion.
The prompt also includes field-disambiguation rules to avoid biologically invalid mappings. For
example, budding pattern, reproduction mode, hyphae formation, or filamentous stage must not be
mapped to motility or flagella. Temperatures associated with storage or preservation must not be
mapped to isolation temperature. Fungal LSU or ITS accessions must not be forced into bacterial
16S rRNA fields. Explicitly stated properties that do not fit the predefined schema are stored in an
extradictionary rather than forced into an unrelated field.
Stage 4: LLM-based quality scoring.The fourth stage evaluates the Stage-3 structured extraction
against the Stage-1 retrieved evidence. The rater is given the original retrieved evidence and the
structured JSON output. It is instructed to ignore any natural-language summary field and score only
the structured strain-level entries under strains . The rater evaluates extraction quality along six
dimensions: strain coverage completeness, field coverage completeness, attribution correctness, value
fidelity to the source text, schema compliance, and hallucination/noise control.
The rater also produces evidence-backed diagnostics. Missing expected fields must be supported
by short verbatim quotes from the evidence. Incorrect or unsupported extracted values must either
include a corrected source-supported value or be marked as unsupported. This stage is designed
to detect incomplete extraction, incorrect strain-property attribution, unsupported values, schema
violations, and hallucinated fields.
Quality filtering.Only records whose Stage-4 quality score is greater than 7 are retained for
subsequent corpus and benchmark construction. This threshold removes records with weak coverage,
ambiguous attribution, schema inconsistency, or potential hallucination. As a result, the benchmark
is constructed from records that satisfy three constraints: the source evidence was retrieved verbatim
in Stage 1, the strain identity was explicitly identified in Stage 2, and the structured trait extraction
passed the Stage-4 quality filter.
Expert review of coarse retrieval.After completing the Stage-1 coarse IJSEM retrieval, domain
experts manually reviewed the retrieved evidence blocks. The review focused on whether the extracted
paragraphs and tables were faithful to the original articles, whether they contained relevant strain- or
species-level biological information, and whether the model introduced hallucinated or paraphrased
content. The experts reported that the reviewed Stage-1 outputs were accurate. This expert review
16

supports the use of Stage-1 retrieved text as the evidence base for the downstream strain-identification
and trait-extraction stages.
A.2 Prompt Templates for IJSEM Extraction
To improve reproducibility, we provide the core prompt templates used in the IJSEM extraction
pipeline. The prompts are stage-specific and enforce conservative extraction behavior. All stages
require strict JSON or verbatim-text outputs and prohibit unsupported biological inference.
Stage-1 evidence retrieval prompt.The Stage-1 prompt retrieves relevant article evidence before
any structured extraction. The core system instruction is:
You are an information extraction agent for microbiology strain characterization papers.
Your task is to extract sentences, paragraphs, and table-related text, including full tables, that
report key intrinsic, strain- or species-level factual information. This includes characteristics
of the organism itself, not procedural protocols.
If a paragraph reports any strain- or species-level physiological, biochemical, metabolic, or
growth property, it must be extracted verbatim in full, even if it also contains experimental
methods, assay names, media descriptions, or protocol-like language. If a table contains
any information about the target strain or species, extract the entire table, including all rows,
columns, captions, and footnotes.
Extract only what is explicitly stated in the article. Do not infer, assume, generalize, or add
information. Output only the extracted text, verbatim. Remove HTML tags. Deduplicate
identical content. Do not add explanations, labels, summaries, rewriting, paraphrasing, or
reordering. If no relevant content is found, output exactlyNO_RELEVANT.
The corresponding user prompt asks the model to extract key strain- and species-level factual
information from the article, focusing on species names, strain identifiers, isolation source, geographic
location, growth or metabolic properties, culture collection identifiers, NCBI-related accessions, and
full tables that report strain or species properties.
Stage-2 strain identification prompt.The Stage-2 prompt converts retrieved evidence into a strain
inventory. The core instruction is:
You are a strain extraction system for microbiology articles. Given verbatim text from a
paper, including paragraphs, tables, and captions, extract all biological strains mentioned
in the paper. Output the most complete strain names by combining explicitly stated genus,
species, and strain identifiers from the text or captions. Classify strains into strains proposed
or studied by the paper and strains used for comparison.
You may combine taxonomy and strain only if all parts are explicitly stated. Do not infer or
guess missing information. Table captions have priority for full strain names. Copy names
verbatim, including symbols, punctuation, and type-strain markers. Ignore row numbers,
labels, and placeholders. Nomenclatural qualifiers such assp. nov.,gen. nov., orcomb. nov.
are part of the explicit species name when they appear in the text and must be retained.
If multiple names refer to the same strain identifier, retain only the most complete name.
The most complete name is the one that includes the maximum explicitly stated taxonomic
information. Output strict JSON only:
{"paper_strains": [...],"comparison_strains": [...]}.
This prompt allows the pipeline to distinguish primary strains from comparison organisms before any
trait label is extracted.
Stage-3 bounded trait extraction prompt.The Stage-3 prompt performs structured extraction only
forpaper_strains. The core instruction is:
You are a strict strain-level information extraction agent for microbiology papers. Given
original evidence text and a list of paper strains, extract explicitly stated strain-level prop-
erties for each listed strain and generate a concise summary. Extract only information
explicitly stated in the evidence. Do not infer, guess, normalize, calculate, or add missing
facts. Copy values verbatim, except for minimal repair of clearly corrupted symbols caused
by text encoding. Do not create new strains or rename strains. Ignore any strain not listed in
PAPER_STRAINS.
17

Assign a property to a strain only if the text explicitly links it to that strain, for example
through a local statement, a table column, or a caption-defined mapping. Global or group-
level statements must not be assigned to individual strains. Unstated fields must be omitted.
If an explicitly stated property does not semantically match any target field, store it in extra
rather than forcing it into an unrelated field.
Output strict JSON only.
The prompt lists all allowed fields and field-specific disambiguation rules. This prevents common
errors such as mapping budding pattern to motility, preservation temperature to isolation temperature,
or fungal barcode accessions to bacterial 16S rRNA accession fields.
Stage-4 quality scoring prompt.The Stage-4 prompt audits the structured extraction. The core
instruction is:
You are a strict evaluator for strain-level information extraction. Given original evidence text
and extracted JSON, evaluate accuracy, completeness, attribution, and schema compliance.
The top-level summary field may be present but must be ignored for scoring, coverage
checks, attribution, and diagnostics. Only content understrainsis subject to evaluation.
A value is correct only if it is explicitly stated or directly supported by combining information
explicitly stated in the evidence. Inference beyond the text, guessing, normalization, or
unit conversion is not allowed. Strain-level facts must be explicitly tied to that strain either
locally in text or through clear contextual linkage such as table caption, table column, or
section scope. Global or group-level statements must not be assigned to strains.
Score all dimensions: strain coverage completeness, field coverage completeness, at-
tribution correctness, value fidelity, schema compliance, and hallucination/noise con-
trol. For every reported issue, provide evidence using a short verbatim quote or
NO_EXPLICIT_SUPPORT_FOUND. Output strict JSON only.
We use the resulting quality score to filter records and retain only entries with score greater than 7 for
benchmark construction.
A.3 Strain Name Normalization and Cross-Database Matching
The IJSEM extraction stages preserve strain names in their article-level form. This is necessary for
traceability because taxonomic articles often include nomenclatural qualifiers, type-strain markers,
strain aliases, collection identifiers, and punctuation that should not be discarded during evidence
extraction. However, direct lookup in NCBI and BacDive requires a standardized query representation.
We therefore apply a separate strain-name normalization stage for cross-database matching. The
normalized fields are used only for database lookup and alignment; they do not replace the original
IJSEM evidence string.
Normalization target.For each raw strain name, the normalizer outputs a strict JSON object with
two fields:
NCBI_scientific_nameandNCBI_modifier.
The scientific-name field is normalized to the binomial form “Genus species”. The normalizer keeps
only the first valid binomial name and removes nomenclatural qualifiers such assp. nov.andgen.
nov., subspecies annotations, strain annotations, parenthetical content, and collection-code strings.
The strain modifier field is extracted separately as the first strain-like token appearing after the species
name.
Modifier extraction.The modifier is intended to capture the article’s primary strain identifier rather
than a culture collection alias. A valid modifier may contain letters, numbers, hyphens, and type-strain
notation. Trailing type-strain suffixes such as Tand(T)are removed. Text after equality signs is
ignored to avoid mixing the primary strain identifier with equivalent culture collection deposits.
Collection identifiers such as DSM, KCTC, MCCC, NCTC, ATCC, JCM, CGMCC, CICC, and
NBRC are ignored as primary modifiers. If no valid strain-like modifier is explicitly present, the
modifier field is left empty. The normalizer is explicitly prohibited from guessing or inferring missing
identifiers.
Example.For an article-level strain name such as
“Guyparkeria halopsychrophilasp. nov. LHSS19-1(T)”,
18

the normalized scientific name is
Guyparkeria halopsychrophila,
and the normalized modifier is
LHSS19-1.
The original article string remains stored in the IJSEM-derived record, while the normalized fields
are used for NCBI search.
NCBI matching.We first use the normalized scientific name and modifier to retrieve NCBI genome
and protein records. Matches are retained only when the species-level identity and strain-level
modifier support the same strain identity. This prevents a species-level match from being incorrectly
treated as a strain-level match. Ambiguous cases, missing-modifier cases, collection-code-only
matches without sufficient strain support, and records whose taxonomic identity cannot be reconciled
are excluded from exact-match benchmark construction.
BacDive matching.After NCBI matching, we further align strains to BacDive phenotype annotations
using standardized scientific names and strain identifiers. BacDive provides structured phenotype
records that complement IJSEM-derived literature evidence. The matching stage produces a unified
strain-centric record containing: (i) original IJSEM evidence, (ii) normalized strain identity, (iii)
NCBI genome and protein information, (iv) BacDive phenotype annotations when available, and
(v) downstream genome representations. Conservative matching is used throughout to reduce false
positives, particularly when multiple strains share a species name but differ in strain identifiers or
culture collection deposits.
Separation between evidence preservation and database lookup.We distinguish between evidence
strings and normalized lookup keys. The original IJSEM strain name is preserved for traceability and
citation to the article. The normalized NCBI fields are used only to query external databases. This
separation allows the pipeline to preserve the taxonomic wording of IJSEM descriptions while still
enabling reliable cross-database alignment.
A.4 Trait Normalization and Controlled Vocabularies
The Stage-3 extractor preserves values as they appear in the source article. These raw values are useful
for traceability but are not directly suitable for benchmark evaluation because IJSEM articles vary in
wording, units, abbreviations, and reporting granularity. We therefore apply trait normalization after
extraction and quality filtering. The goal is to map explicitly supported raw values into controlled
benchmark targets while retaining the original evidence strings.
General normalization principle.Normalization is performed only when the raw value is explicitly
supported by the source evidence and can be mapped unambiguously to the benchmark schema. We
do not infer missing labels from taxonomy, habitat, related species, or medium recipes. Values that
are biologically informative but incompatible with the benchmark schema are retained in the corpus
as auxiliary information but are not used as benchmark labels.
Physiological boundary traits.For growth temperature range, pH range, and salinity range, explicitly
reported lower and upper bounds are converted into standardized interval targets. Temperature is
represented in degrees Celsius. pH is represented on the original pH scale. Salinity is standardized
when the source unit permits reliable conversion. If the article reports only qualitative descriptions
such as “halophilic”, “acidophilic”, or “thermotolerant” without numeric bounds, the statement is
retained in the raw corpus but not used as an interval benchmark label.
Optimal-condition traits.For optimal growth temperature, optimal pH, and optimal salinity,
explicitly reported optima are converted into scalar targets. If an article reports an optimum as a
narrow range, the range is retained in the corpus and converted to a scalar only when needed by the
evaluation metric. If multiple optima are reported under different media or conditions, the value is
used only when the target condition is clearly attributable to the strain-level optimum; otherwise, it is
treated as ambiguous and excluded from the benchmark target set.
Substrate-source and metabolic traits.For carbon source, electron donor, electron acceptor, and
nitrogen source, raw extracted values are mapped to controlled label sets. We canonicalize obvious
aliases, spelling variants, and chemically equivalent surface forms while preserving the original source
strings. Multi-label traits are represented as sets. Medium composition is handled conservatively:
19

a component of a growth or isolation medium is not automatically treated as a positive substrate-
utilization label unless the article explicitly states that the strain utilizes, requires, oxidizes, reduces,
or grows with that component in the relevant metabolic role. This avoids converting every ingredient
in a complex medium into a positive metabolic capability.
Categorical physiological traits.For oxygen tolerance, energy type, photosynthesis type, and carbon
fixation pathway, extracted descriptions are mapped to controlled vocabularies with domain-expert
guidance. Oxygen tolerance labels are assigned only when the article explicitly describes aerobic,
anaerobic, facultative, microaerophilic, or aerotolerant behavior. Energy type and photosynthesis
labels are assigned only when the evidence supports the corresponding physiological category. Carbon
fixation pathway labels are assigned only when the pathway or its diagnostic genes or reactions are
explicitly stated. Ambiguous descriptions are retained in raw fields but excluded from closed-form
benchmark labels unless they can be mapped unambiguously.
Morphological traits.For morphology, we standardize cell shape, motility, flagella, spore formation,
and Gram stain. These fields are normalized separately because taxonomic descriptions often mention
them in adjacent sentences but they are not logically interchangeable. The presence of flagella is
not automatically mapped to motility unless motility is explicitly reported. Conversely, non-motility
is not automatically mapped to absence of flagella. Budding pattern, filamentous growth, hyphae
formation, reproductive mode, and related morphological observations are not mapped to motility or
flagellar state. They are retained as auxiliary properties when explicitly stated.
Canonicalization and exclusion.For each candidate benchmark label, we require three conditions:
(i) the value is explicitly supported by the source evidence, (ii) the Stage-4 extraction quality score
for the record is greater than 7, and (iii) the value can be mapped to the target schema without
unsupported inference. This conservative policy favors precision over recall. It reduces noisy labels
in the benchmark, while retaining richer raw information for future analyses.
A.5 Benchmark Construction and Filtering
After IJSEM extraction, cross-database matching, trait normalization, and genome alignment, we
construct a strain-centric benchmark for genome-conditioned microbial life-boundary prediction.
Each benchmark instance pairs a strain-level genome representation with a query for one target
physiological attribute. A single strain may contribute multiple benchmark instances when it has
multiple high-quality, explicitly supported labels.
Record-level filtering.We first filter IJSEM-derived records using the Stage-4 extraction quality
score. Only records with a quality score greater than 7 on the 0–10 scale are retained. This threshold
is applied before trait-specific instance construction, ensuring that all benchmark instances derived
from IJSEM evidence originate from records that pass the extraction-quality filter. This step removes
low-confidence structured extractions with weak attribution, incomplete field coverage, schema
violations, or potentially unsupported values.
Trait-level filtering.Within retained records, each candidate trait label is further checked for
benchmark compatibility. A value is included as a benchmark label only if it satisfies three conditions:
(i) it is explicitly supported by the source evidence, (ii) it maps unambiguously to one benchmark
field, and (iii) it matches the expected output type of that field. Numeric growth ranges are used
for physiological-boundary prediction; numeric optima are used for optimal-condition prediction;
controlled substrate/source labels are used for metabolic prediction; and controlled categorical or
morphological labels are used for classification tasks. Raw values that are explicit but not compatible
with the benchmark target format are preserved in the full corpus but excluded from benchmark
evaluation.
Genome alignment requirement.Benchmark strains must be alignable to genomic information. We
therefore require a successful cross-database match to NCBI genome or protein records sufficient for
constructing the strain-level genome representation. Strains with high-quality physiological labels but
no usable genomic match are retained in the broader corpus but excluded from genome-conditioned
benchmark evaluation.
Instance construction.For each eligible strain record, we create one benchmark instance for every
non-null target field. Each instance contains an anonymized strain identifier, a genome representation,
a target field name, a task-specific query, and the normalized ground-truth label. The benchmark
20

is therefore organized as a collection of strain–field pairs rather than as one record per strain. This
design supports heterogeneous prediction targets while preserving a unified genome-conditioned
input format.
Benchmark groups.We organize benchmark instances into five task groups: physiological boundary
prediction, optimal-condition prediction, substrate/source prediction, categorical physiological trait
prediction, and morphology prediction. Physiological boundary tasks include growth temperature
range, pH range, and salinity range. Optimal-condition tasks include optimal growth temperature,
optimal pH, and optimal salinity. Substrate/source tasks include carbon source, electron donor,
electron acceptor, and nitrogen source. Categorical physiological tasks include oxygen tolerance,
energy type, photosynthesis type, and carbon fixation pathway. Morphology tasks include cell shape,
motility, flagella, spore formation, and Gram stain.
Target spaces.Table 3 summarizes the benchmark subtasks and their output spaces. Boundary
and optimum tasks use numeric lower–upper objects. Categorical physiology and morphology
tasks use controlled single-label vocabularies, except for Boolean-style traits, which are represented
as true/false outputs. Substrate/source tasks are represented as multi-label lists over controlled
vocabularies.
Table 3: Benchmark subtasks and output spaces. Boundary and optimum tasks use numeric lower–
upper objects; categorical and morphology tasks use controlled single-label or Boolean outputs;
substrate/source tasks use controlled multi-label outputs.
Category Subtask Output space
Physiological boundarygrowth_temperature_range_C Numeric interval in Celsius: {"lower": number, "upper":
number}
Physiological boundarypH_range Numeric interval: {"lower": number, "upper": number}
Physiological boundarysalinity_range Numeric interval in % w/v: {"lower": number, "upper":
number}
Optimal conditiongrowth_temperature_opt_C Numeric optimum in Celsius, represented as {"lower":
number, "upper": number}
Optimal conditionpH_opt Numeric optimum, represented as {"lower": number,
"upper": number}
Optimal conditionsalinity_opt_wv_percent Numeric optimum in % w/v, represented as {"lower": number,
"upper": number}
Categorical physiologyoxygen_tolerance aerobic , anaerobic , facultative_anaerobic ,
microaerophilic,aerotolerant_anaerobic
Categorical physiologyenergy_type chemoorgano , chemolitho , photoautotrophic ,
photoheterotrophic
Categorical physiologyphotosynthesis_type oxygenic,anoxygenic,none,strictly aerobic
Categorical physiologycarbon_fixation_pathway Calvin-Benson-Bassham ,reverse_tca ,wood-ljungdahl ,
3_hydroxypropionate_4_hydroxybutyrate , rump ,
serine_cycle,none,heterotrophic
Morphologycell_shape coccoid,rod,short_rod,spiral,pleomorphic
MorphologymotilityBoolean output:trueorfalse
Morphologyflagella absent,polar,peritrichous,lophotrichous
Morphologyspore_formationBoolean output:trueorfalse
Morphologygram_stain positive,negative,variable
Substrate/sourcecarbon_source Multi-label list from: inorganic_carbon , C1,
sugars , alcohols_polyols , organic_acids_TCA ,
amino_acids ,complex_media ,hydrocarbons ,aromatics ,
none_or_negative,unknown,other,hydrogen
Substrate/sourceelectron_donor Multi-label list from: hydrogen ,reduced_sulfur ,
iron_or_metal ,organic ,hydrocarbons ,aromatics ,
unknown ,sugars ,organic_acids_TCA ,alcohols_polyols ,
C1,alcohols ,complex_media ,inorganic_carbon ,
amino_acids
Substrate/sourceelectron_acceptor Multi-label list from: oxygen ,nitrate_nitrite ,
sulfur_oxyanions_or_S0 ,fumarate_or_other_organic ,
iron_manganese , arsenate_antimonate_selenate ,
carbon_dioxide,none,unknown,organic
Substrate/sourcenitrogen_source Multi-label list from: ammonium ,nitrate_nitrite ,
urea , amino_acids_peptides , yeast_extract ,
nitrogen_fixation ,other ,unknown ,none_or_negative ,
complex_media
Evaluation-time anonymization.To evaluate genome-grounded prediction rather than memorization
from taxonomic names, strain names are anonymized during benchmark evaluation. The model
receives the genome-derived representation and the target query, but not the true species or strain
21

name. This prevents models from exploiting memorized associations between known microbial
names and common phenotypes. We separately evaluate a strain-name setting as an auxiliary analysis,
while the primary benchmark uses anonymized strain identities.
Leakage control.We apply leakage control at both the strain and tool levels. Benchmark samples are
kept disjoint from downstream agent-training data. During tool-augmented inference and trajectory
construction, the query strain’s own ground-truth annotation is not exposed through retrieval outputs.
For the similarity-based RAG tool, the target strain itself is excluded from retrieval. This prevents
direct label leakage while still allowing the agent to use information from biologically related strains
as external evidence. The separation between raw IJSEM evidence, normalized database identifiers,
training records, and benchmark instances is maintained throughout construction.
A.6 Human Expert Validation
In addition to automated LLM-based quality scoring, we conducted human expert validation after
the Stage-1 IJSEM evidence retrieval. This validation was performed before the downstream strain-
identification and trait-extraction stages. The purpose was to verify that the coarse retrieval step was
faithful to the original articles, because all later stages operate on the retrieved evidence rather than
the full article.
Validation criteria.Experts assessed three aspects of the Stage-1 outputs. First, they checked
factual fidelity: whether the extracted paragraphs and tables were copied from the source article
without hallucinated content. Second, they checked relevance: whether the retrieved text contained
intrinsic strain- or species-level information relevant to microbial characterization. Third, they
checked context preservation: whether physiology-rich paragraphs and relevant tables were retained
completely enough to support downstream strain-property attribution.
Outcome.The experts reported that the reviewed Stage-1 outputs were accurate and faithful to the
original IJSEM articles. In particular, the retrieved blocks preserved relevant biological information
and did not introduce unsupported facts. Based on this review, we used the Stage-1 outputs as the
evidence base for Stage 2 and Stage 3. The subsequent Stage-4 automatic rater provides an additional
structured quality-control layer at the record level, and only records with quality score greater than 7
are retained for benchmark construction.
Role of expert validation. The expert review serves a different role from the Stage-4 rater. Expert
validation checks whether the coarse evidence retrieval step produces faithful source evidence. Stage-
4 scoring checks whether the structured JSON extraction correctly attributes fields to strains and
complies with the target schema. Combining these two checks reduces both upstream evidence-
retrieval errors and downstream structured-extraction errors.
A.7 Summary of Data Construction Decisions
The complete data construction pipeline follows a conservative evidence-to-label design. First,
we retrieve relevant evidence from IJSEM articles verbatim. Second, we identify primary paper
strains and comparison strains from the retrieved evidence. Third, we extract structured strain-level
properties only for the primary paper strains. Fourth, we score the structured records and retain only
records with quality score greater than 7. Fifth, we normalize strain names for NCBI and BacDive
matching while preserving the original IJSEM evidence strings. Finally, we harmonize explicit trait
values into controlled benchmark labels and construct genome-conditioned task instances only when
a strain has both a high-quality label and a usable genomic match.
This design intentionally favors precision over recall. Some explicit but ambiguous values are retained
in the broader corpus but excluded from the benchmark. Similarly, strains with physiological evidence
but insufficient genome matching are not used for genome-conditioned evaluation. These filtering
decisions improve the reliability of the benchmark and reduce the risk of evaluating models against
noisy or weakly attributed labels.
22

B Experimental Setup
B.1 Tool construction for genome-conditioned reasoning
To support genome-conditioned life-boundary prediction, we equip the agent with two biological
tools: a similarity-based RAG tool and a GEM metabolic perturbation tool. The RAG tool is built
from the synthetic SFT database, which contains field-level annotations for 8,000 strains. Given a
query strain, we compute its similarity to database strains in the gene-embedding space and return
the top-3 nearest strains with their ground-truth field annotations as tool observations. During SFT
data distillation, we exclude the target strain’s own ground-truth annotations from the tool outputs to
prevent label leakage.
For the GEM tool, we reconstruct strain-specific genome-scale metabolic models from protein
sequences using CarveMe [ 38]. To account for major taxonomic and cell-envelope differences in
biomass requirements, we instantiate each strain with three biomass templates: Gram-positive bacteria,
Gram-negative bacteria, and archaea, with the archaeal template adapted from iAF692†[39,40]. We
then simulate each model under six medium perturbation conditions. Specifically, five perturbations
individually remove sulfate ( SO2−
4), ferric iron ( Fe3+), nitrate ( NO−
3), nitrite ( NO−
2), or oxygen ( O2)
from the medium, while the sixth removes all five components simultaneously. For each biomass
template and perturbation condition, we compute the minimal medium composition supporting
at least 10% of the maximum growth rate, yielding 18 GEM-derived tool observations per strain.
These observations provide mechanistic evidence about nutrient dependence, redox feasibility, and
metabolic constraints for downstream physiological reasoning.
B.2 Tool Interfaces
GGBound is trained and evaluated as a genome-conditioned tool-using agent. Each prompt contains
a special <gene> token, which is replaced internally by the projected genome representation, and an
anonymized genome handle that can be passed to external tools. The model does not observe the true
strain name, raw genome identifier, or any physical storage path during anonymized evaluation.
The environment exposes two tools: a retrieval tool and a GEM tool. Both tools take the anonymized
genome handle as input and return structured textual observations that are appended to the con-
versation. Tool observations are visible to the model in subsequent turns, but they are treated as
environment outputs rather than model-generated tokens during training.
Retrieval tool.The retrieval tool provides nearest-neighbor phenotypic evidence. Its input is the
anonymized genome handle of the query strain. Its output is a compact JSON-style observation
containing the top retrieved strains in genome-embedding space, their similarity scores, and a small
set of phenotype fields relevant to the current prediction task. A typical observation has the following
form:
{
"tool": "rag_tool",
"top_similar_records": [
{
"rank": 1,
"similarity": 0.87,
"phenotypes": {
"growth_temperature_range_C": {"lower": 20, "upper": 37},
"pH_range": {"lower": 6.0, "upper": 8.0},
"oxygen_tolerance": "aerobe",
"gram_stain": "negative",
"carbon_source": ["glucose", "acetate"]
}
}
],
"retrieved_count": 3
}
The retrieval output is intended to provide contextual evidence from biologically similar strains.
Since retrieved neighbors may be incomplete or locally misleading, the model is instructed to treat
the observation as evidence rather than as a direct label oracle.
†http://bigg.ucsd.edu/models/iAF692
23

GEM tool.The gem_tool provides physiological and metabolic-model evidence from genome-scale
metabolic models (GEMs) under predefined biomass formulations and medium-perturbation settings.
Its input consists of the anonymized genome handle and a configuration id. The 18 configurations
correspond to three biomass formulations—archaeal, Gram-negative, and Gram-positive—crossed
with six perturbation conditions. The perturbations remove one of O2,Fe3+,NO−
3,NO−
2, orSO2−
4,
or remove all five simultaneously.
The tool output is a structured observation containing a minimal substrate dictionary predicted by the
corresponding GEM configuration. The dictionary maps exchange-reaction identifiers to the uptake
amounts required by the model under the selected physiological-metabolic setting. These raw GEM
outputs are not treated as final phenotype labels; instead, they provide mechanistic evidence that the
agent can use when reasoning about carbon sources, electron acceptors, nitrogen sources, and related
metabolic traits.
A typical observation is:
{
"tool": "gem_tool",
"configuration_id": 7,
"minimal_substrate_dict": {
"EX_3amp_e": 1.25,
"EX_arg__L_e": 3.18,
"EX_so4_e": 0.03,
"EX_xylan8_e": 28.16,
"EX_zn2_e": 0.002
},
"error": null
}
The full tool response may contain additional exchange reactions and real-valued uptake coefficients.
If the requested GEM configuration is unavailable, the tool returns the same schema with an empty
minimal_substrate_dict and a non-null error message. This fixed output format allows the agent
to reason over successful and failed GEM calls uniformly.
The final assistant answer is still required to follow the task-specific canonical output schema. We do
not evaluate predictions against raw exchange-reaction identifiers; GEM observations serve only as
intermediate physiological-metabolic evidence.
Agent-tool interaction.At each rollout step, the model may either call one of the tools or produce a
final answer. When the model emits a tool call, the environment executes the call and appends the
returned observation to the conversation. The model then continues generation conditioned on the
original prompt, previous assistant turns, and accumulated tool observations. This loop terminates
when the model emits a final answer or reaches the maximum number of tool-calling rounds.
B.3 Generating Modality-Fused Instruction Data
We construct a modality-fused instruction dataset from structured biological strain QA pairs. For
each selected strain, the pipeline preserves the original attribute-level QA pairs and augments them
with two generated data types: paraphrased attribute QA pairs and holistic strain-level description
QA pairs. For attribute-level augmentation, each valid QA pair is rewritten by a language model
while preserving its original meaning. The prompt explicitly requires factual consistency, no new
information, and semantic equivalence between the original and rewritten answers. To improve
linguistic diversity, the prompt varies clause order, syntactic structure, and information flow. Near-
duplicate rewrites are filtered using normalized sequence similarity against both the original QA
pair and previously accepted variants. For holistic augmentation, all QA pairs of a strain are first
converted into a fact block. The model then generates a short global question, such as asking for an
overall description of the strain, followed by a comprehensive answer grounded only in the provided
facts. This converts isolated attributes into integrated descriptions covering identity, taxonomy,
habitat, sampling source, location, and other available strain information. Generation is performed
with batched vLLM inference. The system enforces JSON-formatted outputs, retries malformed
generations, and supports incremental JSONL writing with resume functionality. The final dataset
contains unified records with id,name ,question , and answer , combining fine-grained attribute
supervision with global strain-level descriptive supervision.
24

B.4 Agentic SFT Data Collection Details
We construct the agentic SFT corpus using a multi-stage trajectory distillation pipeline. The goal
is to distill tool-using biological reasoning trajectories from a strong teacher model into supervised
fine-tuning examples for the compact genome-conditioned agent. Each training example contains an
anonymized strain embedding handle, a biological trait prediction query, optional intermediate tool
calls, tool observations, and a final JSON-formatted answer.
Teacher rollout generation.We use Qwen3.5-27B as the teacher model to generate initial multi-turn
trajectories with vLLM. For each input sample, the teacher is prompted as a biological reasoning
agent with access to rag_tool andgem_tool . The input pool is constructed from BacDive-derived
SFT records. Each prompt specifies the target trait field, an anonymized embedding handle, available
tool schemas, and the required final JSON answer format.
For each sample, we generate four candidate trajectories using temperature 1.0, top- p= 0.95 ,
maximum generation length 4096, maximum model context length 16,384, and up to six tool-calling
rounds. The teacher may call rag_tool to retrieve phenotypically annotated nearest-neighbor
strains and gem_tool to obtain genome-scale metabolic model evidence under selected biomass and
medium-perturbation configurations. The resulting trajectories contain a sequence of assistant turns,
tool calls, tool observations, intermediate planning text, and a final structured answer.
Teacher system prompt.The teacher system prompt defines the model as an expert biology agent
for microbial trait prediction. It introduces two biological tools:
•rag_tool , which retrieves similar strains from the genome-embedding database and returns
compact phenotype annotations;
•gem_tool , which returns genome-scale metabolic model evidence for a selected biomass
formulation and perturbation condition.
The prompt also defines the gem_tool identifier space. Tool ids 1–6 correspond to archaeal biomass
formulations, 7–12 to Gram-negative biomass formulations, and 13–18 to Gram-positive biomass
formulations. Within each block of six, the offset specifies the unavailable medium component:
O2,Fe3+,NO−
3,NO−
2,SO2−
4, or all five components jointly. The teacher is instructed to prefer
conclusions grounded in tool evidence, avoid unsupported guesses, gather complementary evidence
when useful, avoid unnecessary duplicate calls, and produce exactly one final JSON object.
User prompt format.For each subtask, the user prompt contains the prediction problem, the
anonymized embedding handle, tool-use guidance, and the target-specific final answer schema. The
prompt follows the template:
Problem: <task-specific trait prediction instruction>
Anonymous genome handle: <anonymous genome handle>
Tool-use guidance:
You have access to rag_tool and gem_tool using the anonymous genome handle above. Use these tools when
they provide useful evidence, and prefer tool-grounded answers over unsupported prior beliefs.
Guidelines:
1. rag_tool provides retrieval-grounded phenotype evidence, but may be incomplete or noisy.
2. gem_tool provides complementary GEM evidence and may help verify or refine retrieval evidence.
3. When both tools provide relevant evidence, synthesize them.
4. If one tool returns weak, empty, redundant, or irrelevant evidence, rely more on the stronger
evidence.
5. Multiple tool calls are allowed when they improve confidence, but avoid unnecessary repeats.
6. If tool evidence conflicts, choose the most defensible answer.
7. Use the model’s tool-calling format for tool calls and keep intermediate turns concise.
When ready, output exactly one final JSON object matching: <schema>.
Only the final answer must follow this JSON schema; intermediate reasoning or tool-calling turns need
not be JSON.
For interval and optimum-condition tasks, the prompt additionally encourages the teacher to use
GEM evidence when it can strengthen or validate retrieval-based predictions, rather than relying only
on nearest-neighbor annotations.
Subtask coverage.The final distillation corpus contains 54,249 examples across 17 realized subtasks.
The distribution is shown in Table 4.
25

Table 4: Subtask distribution of the final agentic SFT distillation corpus.
Category Subtask Number of examples
Growth intervalgrowth_temperature_range_C7948
Growth intervalpH_range3481
Growth intervalsalinity_range3894
Growth optimumgrowth_temperature_opt_C4074
Growth optimumpH_opt3251
Growth optimumsalinity_opt_wv_percent2317
Physiologyoxygen_tolerance5805
Physiologyenergy_type305
Morphologycell_shape5218
Morphologymotility5139
Morphologyflagella648
Morphologyspore_formation2794
Morphologygram_stain5448
Substrate/sourcecarbon_source3572
Substrate/sourceelectron_donor46
Substrate/sourceelectron_acceptor78
Substrate/sourcenitrogen_source231
Total – 54249
Trajectory parsing and scoring.After teacher rollout generation, we parse the final answer of
each sampled trajectory and compute both strict and best-effort scores. The strict parser requires
the final assistant response to be exactly one valid JSON object containing the requested target
field. The best-effort parser is used only offline to recover candidate predictions from otherwise
noisy generations. Each parsed trajectory is scored using the same family of biological correctness
functions as in RL: interval coverage and compactness for boundary traits, normalized error for
optimum traits, canonical-label match for categorical and morphology traits, and top- kset overlap
for substrate-source traits.
For each input sample, we select the best trajectory among the teacher candidates using a field-aware
ranking procedure. The primary criterion is biological correctness. Ties are broken using strict JSON
validity, parse success, field-aware tool evidence quality, fewer tool errors, richer non-error tool
interaction, and shorter final answers. For interval and optimum prediction tasks, trajectories that use
both rag_tool andgem_tool are preferred over retrieval-only trajectories when answer quality is
comparable, because these tasks benefit from complementary metabolic feasibility evidence.
Retry and evidence strengthening.The distillation pipeline includes two retry stages to improve
weak, incomplete, or retrieval-only trajectories. The first retry targets imperfect trajectories that used
rag_tool but did not use gem_tool . The second retry applies a field-aware policy: interval and
optimum tasks are retried whenever the selected trajectory remains imperfect; substrate-source tasks
are retried when the trajectory is imperfect and lacks GEM evidence; and other tasks are retried
mainly for retrieval-only misses. The forced-retry prompt requires the teacher to call rag_tool
first and then use at least one or two gem_tool calls before finalization, depending on the target
field. During the second merge, 20,843 retry candidates were compared against the original selected
trajectories, and 5,207 original trajectories were replaced by stronger retry trajectories.
Cleaning and finalization.For selected trajectories whose final answer still mismatches the ground-
truth label, we construct long-context cleaning prompts. The cleaner receives the full prior tool
trace, the original final answer, the parsed prediction, the ground-truth label, and a task-specific
difference summary. It is instructed to minimally edit only the final assistant answer while preserving
the existing tool calls and tool observations. The cleaner is not allowed to invent new tool outputs or
alter the evidence trace.
We use this cleaning stage to repair the supervised final-answer target, not to create synthetic tool
observations. When a mismatch is repaired, the corrected final JSON answer is embedded back into
the original trajectory. The preceding tool trace is preserved so that the SFT data still reflects the
actual evidence available before the final answer. This separates tool-use behavior imitation from
final-answer supervision and avoids training the model on fabricated observations.
26

Final SFT format.The final agentic SFT dataset contains 54,249 records. Each record stores the
true genome-embedding path in a top-level metadata field used by the training collator, while the
conversation shown to the model contains only the anonymized embedding handle. Each record
includes:
•id: anonymized strain identifier;
•name: species-level name used as metadata;
•gene_cls_path : true cls_emb.npy path used by the collator, not exposed in the conver-
sation;
•system: biology-agent system prompt;
•tools: JSON schemas forrag_toolandgem_tool;
•conversations : ShareGPT-style multi-turn trajectory containing the user prompt, assistant
tool calls, tool observations, intermediate assistant turns, and final JSON answer.
The final construction consists of 29,914 cleaned embedded trajectories and 24,335 original trajecto-
ries that were already correct after selection and retry. During SFT, only model-generated assistant
content is optimized; environment-provided tool observations are retained as context but are not
treated as assistant-generated targets.
B.5 Reward and Optimization Details
We optimize the genome-aware tool-use policy with Group Relative Policy Optimization (GRPO).
For each prompt, the sampler generates multiple completions, executes any requested tools, scores
each completed trajectory with task-specific rewards, normalizes rewards within the prompt group,
and applies a clipped policy-gradient update to model-generated tokens. Tool observations inserted
by the environment are masked from the policy loss.
Training configuration.The GRPO stage starts from the agentic supervised-finetuned checkpoint.
Training prompts are constructed from BacDive-derived records and contain the system instruction,
the<gene> placeholder, the target trait field, the anonymized genome handle, tool-use guidance, and
the required final JSON schema. Table 5 summarizes the main counterfactual GRPO configuration.
Since the genome-projection module is frozen, the RL stage updates only the language-model policy.
Table 5: Main hyperparameters for the counterfactual GRPO stage.
Category Hyperparameter Value
Sampling Number of generations per prompt 4
Sampling Temperature 1.0
Sampling Top-p0.95
Sampling Top-k50
Optimization Epochs 1
Optimization Per-device batch size 8
Optimization Gradient accumulation steps 1
Optimization Steps per generation 1
Optimization Learning rate1×10−6
Optimization Learning-rate schedule Cosine
Optimization Warmup ratio 0.03
Optimization GRPO clipping parameter 0.2
Optimization Reference KL coefficient 0.0
Context and tools Maximum completion length 8192
Context and tools Maximum vLLM generation tokens 2048
Context and tools Maximum tool-calling rounds 5
Rollout procedure.For each training row, the sampler repeats the same prompt num_generations
times. In the main configuration, each prompt therefore has four sampled completions. During
generation, vLLM receives both the tokenized prompt and the genome modality:
27

{
"prompt_token_ids": ids,
"multi_modal_data": {"gene": gene_array}
}
If a sampled trajectory contains tool calls, the environment executes the calls, appends the resulting
tool messages, and resumes generation from the expanded conversation. This process continues
until a final answer is produced or the maximum number of tool-calling iterations is reached. The
model-generated assistant tokens are optimized, while environment-provided tool payloads are treated
as observations and excluded from the policy-loss mask.
Sequence-level rewards.The trajectory-level reward combines strict output formatting, biological
correctness, tool-use shaping, and a penalty for incorrect direct answers without tool use:
Rseq=w jsonRjson+w corrRcorr+w toolRtool+w ntRno_tool +R external .
In the main counterfactual GRPO configuration, the default weights are:
wjson= 0.5, w corr= 1.0, w tool= 1.0, w nt= 1.0.
The JSON-format reward is strict:
Rjson=+1,if the final answer is exactly one valid JSON object with the target field,
−1,otherwise.
Malformed JSON, Markdown fences, extra prose, missing target fields, wrong target fields, null
values, or additional unsupported fields receive−1.
The biological correctness rewardR corrdepends on the target trait type. Interval-valued traits, such
as growth temperature range, pH range, and salinity range, are rewarded for covering the ground-
truth interval while remaining compact. Optimal-condition traits are scored by normalized absolute
error. Categorical physiological and morphology traits receive +1for an exact match after label
canonicalization and −1otherwise. Substrate-source traits, including carbon source, electron donor,
electron acceptor, and nitrogen source, are evaluated using top-kmicro-F1 withk= 5:
Rcorr= 2·MicroF1@5−1.
Thus a perfect substrate prediction receives+1, while a prediction with no overlap receives−1.
Tool-use reward.The tool-use reward encourages the policy to acquire evidence while discouraging
both zero-tool shortcuts and excessive tool calls. Let cbe the number of tool calls in a trajectory and
tbe the target number of calls. The target is annealed over training:
t=t init+p(t final−tinit),
wherep∈[0,1]denotes training progress. In the main configuration,
tinit= 4.0, t final= 2.0.
The tool-use reward is:
Rtool=

−1.0, c= 0,
0.25 + 0.75p
c/t,0< c≤t,
max(0.25,1.0−0.5(c−t)/t), c > t.
This schedule makes nonzero tool use substantially better than no tool use, but does not reward
unbounded tool calling.
Counterfactual gene-usage reward.The counterfactual trainer adds a token-level reward that
encourages the final answer to depend on the genome embedding. For each sampled completion, the
trainer computes teacher-forced token log-probabilities twice: once with the real genome embedding
g, and once with an ablated zero genome embedding. For answer tokeni, we define:
∆i= logπ θ(yi|x, g)−logπ θ(yi|x,0).
The gap is clipped to
[−cgene, cgene],
28

where the default cap is cgene= 5.0 . With positive-only gating enabled, the token-level gene-usage
reward is:
Rgene,i = max(R corr,0)·clip(∆ i,−cgene, cgene)·⊮[i∈answer].
The main configuration uses gene_usage_reward_weight =0.5. This reward is applied only to
final-answer tokens and not to tool-call payloads or environment-inserted tool observations. The
correctness gate prevents the model from being rewarded for genome-dependent but biologically
incorrect answers.
GRPO advantage normalization.After rewards are gathered across distributed workers, comple-
tions are grouped by their original prompt. For a prompt group of size G=num_generations , the
sequence-level advantage for completionjis:
Aj=Rj−µG
σG+ 10−4,
where µGandσGare the mean and standard deviation of the rewards within the same prompt group.
The final token-level advantage is:
Atoken
j,i =A j⊮[i∈answer] +w toolRtool,i+w geneRgene,i +w attnRattn,i.
In the main counterfactual launcher, attention_reward_weight =0.0, so the attention-based shap-
ing term is disabled.
Policy loss.The trainer computes current token log-probabilities under the genome-aware policy and
compares them with the old log-probabilities from the sampling policy. For each optimized token,
ρi= exp (logπ θ(yi)−logπ old(yi)).
The clipped GRPO token loss is:
Li=−min (ρ iAi,clip(ρ i,1−ϵ,1 +ϵ)A i),
withϵ= 0.2. The final loss averages over valid model-generated tokens:
L=1
BBX
b=1P
imb,iLb,iP
imb,i,
wherem b,imasks padding tokens and environment-inserted tool payloads.
B.6 Implementation details
We run all training experiments on a single 8-GPU node equipped with NVIDIA H200 GPUs
under Ubuntu 22.04. The training pipeline consists of three stages: genome–text alignment, agentic
supervised fine-tuning, and agentic reinforcement learning. Unless otherwise specified, all stages use
full-parameter fine-tuning with bf16 precision. We implement agentic SFT with LLaMA-Factory [ 44],
and build the agentic RL training framework on OpenR1 [45] and TRL [46].
First, we conduct gene-text instruction tuning starting from a gene-aligned Qwen3.5-4B checkpoint.
The training data consist of gene-text alignment examples, and the goal of this stage is to improve
the model’s ability to map genome-conditioned representations to textual biological descriptions
before introducing tool-use supervision. We train for one epoch with maximum sequence length
1,024, effective batch size 256, and learning rate 2×10−5. This stage requires approximately 16
hours on 8 H200 GPUs.
Second, we perform agent-supervised fine-tuning on distilled tool-use trajectories. The trajectories
are represented in a multi-turn format containing system prompts, tool specifications, model actions,
tool observations, intermediate reasoning, and final structured answers. This stage teaches the model
the tool-calling protocol and how to integrate external observations into its final biological prediction.
To support long tool-use contexts, we use maximum sequence length 16,384. We train for one epoch
with effective batch size 128 and learning rate 5×10−6, which takes approximately 8 hours on the
same hardware.
Finally, we apply GRPO to improve biological correctness, tool-use behavior, and reliance on the
genome modality beyond imitation learning. For each prompt, the policy samples a group of 4
trajectories, and rewards are normalized within the group to obtain relative advantages. Rollouts are
29

generated using colocated vLLM with tensor parallelism over 8 GPUs. We use a global rollout batch
size of 64, learning rate 1×10−6, maximum completion length 8,192, and a limit of 5 tool-calling
iterations per trajectory. Tool-call tokens are optimized as policy actions, while environment-provided
tool observations are excluded from the policy loss. During this RL stage, the genome-fusion module
remains frozen and only the language-policy parameters are updated. The reward includes terms for
JSON validity, biological correctness, scheduled tool use, unsupported direct-answer penalties, and
counterfactual gene grounding. The GRPO stage takes approximately 40 hours on 8 H200 GPUs.
B.7 Evaluation metrics
We evaluate microbial trait prediction across five groups: interval-valued traits, optimal scalar traits,
substrate-source traits, single-label physiological traits, and morphological traits. All model outputs
are parsed as JSON and mapped to the corresponding target field. For categorical and substrate-source
traits, predictions are further canonicalized using task-specific vocabularies and alias mappings.
For interval-valued traits, including growth temperature, pH, and salinity ranges, we compare the
predicted interval [ˆl,ˆu]with the ground-truth interval [l, u] . We report interval coverage rate (ICR),
which measures whether the predicted interval fully covers the ground-truth interval:
ICR =1
NNX
i=11(ˆli≤li∧ˆui≥ui).
For optimal scalar traits, including optimal growth temperature, optimal pH, and optimal salinity,
predictions are evaluated using root mean squared error (RMSE). If a model outputs an interval-form
prediction for an optimal scalar trait, we convert the interval to a scalar using its midpoint before
computing RMSE.
For substrate-source traits, including carbon source, electron donor, electron acceptor, and nitrogen
source, predictions are treated as ranked label lists. We evaluate the top five predictions using mean
average precision at five (mAP@5). Given the ground-truth label set Yiand ranked predictions
ˆyi,1, . . . ,ˆy i,5, AP@5 is computed as
AP@5 i=1
min(|Y i|,5)5X
k=1Prec@k i·1(ˆy i,k∈Yi),
where Prec@k idenotes precision among the top- kpredictions. We report mAP@5 by averaging
AP@5 across evaluation samples.
For single-label physiological and morphological traits, including oxygen tolerance, energy type,
photosynthesis type, carbon fixation pathway, cell shape, motility, flagella, spore formation, and
Gram stain, we report accuracy after label canonicalization.
C Additional Experimental Results
C.1 Training Stability Across Stages
Training Curves.Figures 4–6 show the training losses across the two modal-fusion stages and the
agentic SFT stage. In the first stage, tuning only the MLP projector yields a steadily decreasing loss,
indicating that the projected LucaOne representation can be stably aligned with the LLM token space.
In the second stage, jointly tuning the projector and LLM further reduces the loss without visible
divergence, suggesting that the model can refine genome–text alignment while preserving stable
optimization. The agentic SFT curve also decreases smoothly and stabilizes on long-context tool-use
trajectories, confirming that the model can learn the tool-calling format and evidence-conditioned
response pattern reliably before GRPO optimization.
C.2 Frontier LLM Comparison under Strain-Name Input
Table 6 reports the performance of frontier LLMs when strain names are directly provided. This
setting favors general-purpose LLMs because strain names expose taxonomic priors learned during
30

Figure 4: Training loss of the first modal-fusion stage, where only the MLP projector is tuned.
Figure 5: Training loss of the second modal-fusion stage, where both the MLP projector and LLM
backbone are tuned.
pretraining. The results show that no single frontier model dominates all task groups. GLM-
4.7 performs best on physiological-boundary prediction, achieving the highest ICR on growth-
temperature, pH, and salinity ranges. For physiological optima, Kimi-K2 performs best on optimal
growth temperature and pH, while DeepSeek-V3.2 obtains the lowest RMSE on optimal salinity.
For metabolism and categorical traits, the best model varies by target. GLM-4.7 is strongest on
most substrate-source tasks, whereas Kimi-K2 performs best on several categorical and morphology
targets, including carbon fixation pathway, energy type, flagella, and Gram stain. These results suggest
that strain-name prompting can activate useful biological priors, but the resulting performance is
task-dependent and model-dependent. Compared with these large baselines, GGBound remains
competitive in the main strain-name analysis, indicating that genome conditioning and biological tool
use can compensate for substantially smaller model scale.
C.3 GGBound Evaluation with Strain-Name Input and Tool Access
Table 7 evaluates the GGBound model family when strain names and tool access are both available.
Compared with the anonymized benchmark, this setting provides additional taxonomic cues and
therefore serves as a complementary upper-bound-style evaluation. Agentic training improves most
continuous and categorical targets: the RL agent achieves the best or tied-best performance on all
physiological-boundary and optimum tasks, reducing optimal growth-temperature RMSE from 6.599
to 4.298 and optimal salinity RMSE from 3.336 to 2.001 relative to the base model.
The metabolism results show a different trend. The base model performs best on all four substrate-
source tasks, suggesting that strain names already provide strong metabolic priors for Qwen3.5-4B.
Direct modal fusion can even degrade performance on several metabolism targets, indicating possible
conflict between injected genome embeddings and name-based priors before agentic training. In
contrast, SFT and RL recover stronger performance on categorical physiology and morphology,
31

Figure 6: Training loss of the agentic SFT stage, where only the LLM policy is tuned on distilled
tool-use trajectories.
Table 6: Comparison with frontier LLMs under strain-name input. Best results are shown inbold;
second-best results are underlined .
Task Group Prediction Target Metric DeepSeek-V3.2 DeepSeek-R1 GLM-4.7 Kimi-K2
Physiological
BoundaryGrowth temperature range ICR↑0.285 0.3130.4200.339
pH range ICR↑0.177 0.1790.3710.243
Salinity range ICR↑0.548 0.606 0.7600.555
Physiological
OptimumOptimal growth temperature RMSE↓6.494 6.621 5.671 5.626
Optimal pH RMSE↓0.684 0.674 0.6780.668
Optimal salinity RMSE↓2.0882.315 2.276 2.283
MetabolismCarbon source mAP@5↑0.5590.5900.562 0.522
Electron acceptor mAP@5↑0.334 0.3270.4070.351
Electron donor mAP@5↑0.605 0.6030.6350.569
Nitrogen source mAP@5↑0.175 0.1390.2760.264
Categorical
PhysiologyCarbon fixation pathway Accuracy↑0.554 0.500 0.4730.662
Energy type Accuracy↑0.776 0.784 0.7090.825
Oxygen tolerance Accuracy↑0.4990.406 0.327 0.475
Photosynthesis type Accuracy↑0.9020.895 0.850 0.820
MorphologyCell shape Accuracy↑0.778 0.7820.764 0.778
Flagella Accuracy↑0.460 0.458 0.465 0.567
Gram stain Accuracy↑0.910 0.896 0.8820.912
Motility Accuracy↑0.7720.756 0.689 0.752
Spore formation Accuracy↑0.916 0.9220.9110.922
with the RL agent achieving the best or tied-best results on most of these targets. Overall, these
results show that strain names provide useful prior knowledge, while tool-conditioned training helps
integrate such priors with external observations and genome-conditioned evidence.
C.4 Licenses and Third-Party Resource Usage
We use third-party data resources, pretrained models, and open-source software under their respective
licenses and terms of use. Table 8 summarizes the main external resources used in this work and their
roles in our pipeline.
For literature-derived data, IJSEM articles are used as source evidence for extraction, normalization,
and benchmark construction. We do not redistribute the original full-text articles or publisher-owned
article content. Instead, the released benchmark contains normalized strain-level fields, anonymized
evaluation instances, and derived labels constructed through our evidence-to-label pipeline. NCBI
genome and protein records are used for strain matching and genome representation construction.
BacDive annotations are used as structured phenotype evidence and are cited as a core data source;
we follow its attribution requirements and do not claim ownership over the original BacDive records.
32

Table 7: Results with strain-name input and tool access. Best/second-best are bold/underlined.
Tasks Field Metric Base Model Fusion Model SFT Agent RL Agent
Physiological
BoundaryGrowth temperature range ICR↑0.172 0.208 0.298 0.302
pH range ICR↑0.305 0.369 0.400 0.407
Salinity range ICR↑0.606 0.6040.727 0.727
Physiological
OptimumOptimal growth temperature RMSE↓6.599 5.923 4.413 4.298
Optimal pH RMSE↓0.886 1.135 0.665 0.649
Optimal salinity RMSE↓3.336 2.904 2.022 2.001
MetabolismCarbon source mAP@5↑0.6770.237 0.469 0.557
Electron acceptor mAP@5↑0.5830.456 0.487 0.498
Electron donor mAP@5↑0.7480.258 0.649 0.652
Nitrogen source mAP@5↑0.4460.334 0.420 0.425
Categorical
PhysiologyCarbon fixation pathway Accuracy↑0.694 0.516 0.6610.710
Energy type Accuracy↑0.860 0.710 0.879 0.888
Oxygen tolerance Accuracy↑0.713 0.732 0.7260.734
Photosynthesis type Accuracy↑0.842 0.863 0.926 0.926
MorphologyCell shape Accuracy↑0.791 0.771 0.810 0.813
Flagella Accuracy↑0.725 0.544 0.751 0.768
Gram stain Accuracy↑0.906 0.911 0.947 0.950
Motility Accuracy↑0.797 0.800 0.828 0.837
Spore formation Accuracy↑0.933 0.9040.957 0.957
For models and software, we use open-weight models and open-source frameworks under their
corresponding permissive licenses. LucaOne provides the frozen genome encoder; Qwen3.5 provides
the trainable language backbone and teacher model; DeepSeek, GLM, and Kimi models are used only
for extraction assistance or baseline evaluation. CarveMe and BiGG-derived resources are used to
construct GEM-based tool observations, while LLaMA-Factory, OpenR1, TRL, and vLLM support
supervised fine-tuning, reinforcement learning, and rollout generation. Our released code and derived
benchmark artifacts will include license notices and citations for these third-party resources.
33

Table 8: Third-party resources and licenses. We list the main data sources, pretrained models, and
software frameworks used for benchmark construction, model training, tool execution, and baseline
evaluation.
Resource Usage in this work License / terms
IJSEM articles Source literature for strain-level evidence extraction
and physiological trait curation.Publisher copyright and plat-
form terms; used for scholarly
research extraction.
NCBI genome and protein
recordsGenome/protein matching and construction of strain-
level genome representations.NCBI molecular data usage pol-
icy; NCBI itself places no reuse
restrictions on molecular data.
BacDive Structured strain phenotype annotations for alignment,
benchmark construction, and agent training.Creative Commons Attribution
4.0 International License.
LucaOne Frozen genome encoder used to produce gene-level
and strain-level embeddings.Apache License 2.0.
Qwen3.5 LLM backbone for GGBound and teacher model for
trajectory generation.Apache License 2.0.
DeepSeek-V3.2 / DeepSeek-
R1LLM-based extraction, quality scoring, and frontier-
model comparison.MIT License.
GLM-4.7 Frontier-model baseline under strain-name evaluation. MIT License.
Kimi-K2 Frontier-model baseline under strain-name evaluation. Modified MIT License.
CarveMe Reconstruction of strain-specific genome-scale
metabolic models.Apache License 2.0.
BiGG / iAF692 resources Source of the archaeal biomass template used in GEM
construction.BiGG/MetaNetX-linked re-
sources under attribution-based
terms.
LLaMA-Factory Agentic supervised fine-tuning implementation. Apache License 2.0.
OpenR1 Base framework for agentic reinforcement learning
implementation.Apache License 2.0.
TRL GRPO and reinforcement-learning training utilities. Apache License 2.0.
vLLM Batched inference and rollout generation. Apache License 2.0.
34