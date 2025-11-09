# ExplicitLM: Decoupling Knowledge from Parameters via Explicit Memory Banks

**Authors**: Chengzhang Yu, Zening Lu, Chenyang Zheng, Chiyue Wang, Yiming Zhang, Zhanpeng Jin

**Published**: 2025-11-03 13:53:19

**PDF URL**: [http://arxiv.org/pdf/2511.01581v1](http://arxiv.org/pdf/2511.01581v1)

## Abstract
Large language models suffer from knowledge staleness and lack of
interpretability due to implicit knowledge storage across entangled network
parameters, preventing targeted updates and reasoning transparency. We propose
ExplicitLM, a novel architecture featuring a million-scale external memory bank
storing human-readable knowledge as token sequences, enabling direct inspection
and modification. We design a differentiable two-stage retrieval mechanism with
efficient coarse-grained filtering via product key decomposition (reducing
complexity from $\mathcal{O}(N \cdot |I|)$ to $\mathcal{O}(\sqrt{N} \cdot
|I|)$) and fine-grained Gumbel-Softmax matching for end-to-end training.
Inspired by dual-system cognitive theory, we partition knowledge into frozen
explicit facts (20%) and learnable implicit patterns (80%), maintained through
Exponential Moving Average updates for stability. ExplicitLM achieves up to
43.67% improvement on knowledge-intensive tasks versus standard Transformers,
with 3.62$\times$ gains in low-data regimes (10k samples). Analysis shows
strong correlations between memory retrieval and performance, with correct
predictions achieving 49% higher hit rates. Unlike RAG systems with frozen
retrieval, our jointly optimized architecture demonstrates that interpretable,
updatable models can maintain competitive performance while providing
unprecedented knowledge transparency.

## Full Text


<!-- PDF content starts -->

Published as a conference paper at ICLR 2026
EXPLICITLM: DECOUPLINGKNOWLEDGE FROMPA-
RAMETERS VIAEXPLICITMEMORYBANKS
Chengzhang Yu∗
South China University of Technology
Guangzhou, ChinaZening Lu∗
South China University of Technology
Guangzhou, China
Chenyang Zheng
South China University of Technology
Guangzhou, ChinaChiyue Wang
South China University of Technology
Guangzhou, China
Yiming Zhang
University of Science and Technology of China
Hefei Institutes of Physical Science
Hefei, ChinaZhanpeng Jin†
South China University of Technology
Guangzhou, China
zjin@scut.edu.cn
ABSTRACT
Large language models (LLMs) universally suffer from knowledge staleness
and lack of interpretability due to their implicit knowledge storage paradigm,
where information is distributed across network parameters in an entangled, non-
addressable manner. This fundamental limitation prevents targeted knowledge
updates, verification of stored information, and understanding of model reason-
ing processes. We propose ExplicitLM, a novel architecture that fundamen-
tally reimagines knowledge storage in language models through an explicit, in-
terpretable memory bank system. Our key innovation introduces a million-scale
external memory bank where each entry stores human-readable knowledge as to-
ken sequences, enabling direct inspection and modification of the model’s knowl-
edge base. To efficiently access this massive repository, we design a differentiable
two-stage retrieval mechanism that enables end-to-end training while maintain-
ing discrete knowledge selection, combining efficient coarse-grained filtering with
product key decomposition (reducing computational complexity fromO(N· |I|)
toO(√
N· |I|)) and fine-grained similarity matching through Gumbel-Softmax.
Drawing inspiration from dual-system cognitive theory, we partition knowledge
into frozen explicit facts (20%) and learnable implicit patterns (80%), maintained
through an Exponential Moving Average update strategy that ensures training sta-
bility. Extensive experiments demonstrate that ExplicitLM achieves up to 43.67%
improvement in knowledge-intensive tasks compared to standard Transformers,
with particularly pronounced gains in low-data regimes (3.62×improvement with
10k samples). Our analysis reveals strong correlations between memory retrieval
success and task performance, with correctly predicted samples achieving 49%
higher memory hit rates. Unlike traditional RAG systems with frozen retrieval
components, our jointly optimized architecture demonstrates that interpretable,
updatable language models can maintain competitive performance while provid-
ing unprecedented transparency into their knowledge utilization.
1 INTRODUCTION
Contemporary large language models (LLMs) universally suffer from knowledge staleness, with in-
ternally stored knowledge frozen at training completion Cheng et al. (2024); Singh et al. (2025). This
∗Equal contribution.
†Corresponding author.
1arXiv:2511.01581v1  [cs.AI]  3 Nov 2025

Published as a conference paper at ICLR 2026
temporal limitation creates a widening gap between static model knowledge and dynamic real-world
information. Consider the U.S. presidency: Joe Biden served until January 2025, when Donald
Trump assumed office. Models trained before this transition perpetually provide outdated answers,
unable to reflect real-time changes. Post-training, this knowledge ossification accumulates across
countless facts—political leadership, scientific discoveries, economic indicators, and technological
standards—severely undermining model reliability in practical applications Mousavi et al. (2024).
Knowledge updating thus emerges as critical: models require mechanisms to incorporate temporal
factual changes to maintain utility and trustworthiness in real-world deployments.
Current approaches to acquiring or updating external knowledge primarily rely on two paradigms:
real-time querying through Model Context Protocol (MCP) toolsHou et al. (2025), or knowledge
augmentation via Retrieval-Augmented Generation (RAG) techniquesLewis et al. (2020).However,
MCP-based methods exhibit several critical limitations. First, real-time querying introduces sub-
stantial inference latency, degrading user experienceSingh et al. (2025). Second, dependency on
external APIs compromises system robustnessLi et al. (2025).RAG techniques, though partially mit-
igating knowledge updating challenges, face persistent obstacles: the relevance between retrieved
documents and queries remains difficult to ensure, the inherent misalignment between retrieval and
generation objectives yields suboptimal performance, and the maintenance and updating of external
knowledge bases incurs substantial engineering overheadSalemi & Zamani (2024). These limi-
tations collectively motivate the need for more efficient and integrated approaches to knowledge
acquisition and updating in language models.
The fundamental barrier to direct manipulation of model-internal knowledge stems from the im-
plicit knowledge storage paradigm in current language models. Research demonstrates that LLM
knowledge is predominantly distributed across Feed-Forward Network (FFN) layers of the Trans-
former architectureGeva et al. (2021); Meng et al. (2022); Dai et al. (2022). Unlike traditional
databases with discrete, addressable locations, each piece of LLM knowledge emerges from col-
lective parameter interactions across all FFN layers, creating highly entangled representations that
cannot be independently isolated or modified. This transforms knowledge update into a formidable
challenge: modifying a single fact theoretically requires recalibrating weights throughout the entire
network—a practically infeasible task risking catastrophic interference with other stored knowledge.
This “black-box” nature prevents both verification of acquired knowledge and targeted correction of
problematic content. During pre-training on massive corpora, models inevitably absorb misinforma-
tion, outdated content, or harmful materialPerełkiewicz & Po ´swiata (2024), yet inability to precisely
locate and excise such knowledge allows errors to persist and propagate through outputs, fundamen-
tally undermining reliability and trustworthiness.
More critically, implicit knowledge storage fundamentally impedes interpretability. When gener-
ating predictions, researchers cannot trace specific knowledge foundations underlying model rea-
soning. We cannot determine which facts inform reasoning nor verify reasoning step correctness.
This opacity constrains understanding of model behavior and poses fundamental challenges to build-
ing trustworthy, interpretable AI systems. In high-reliability domains like medical diagnosisEnnab
& Mcheick (2024) and legal consultationLatif (2025), this interpretability lack becomes a primary
deployment barrier.
Motivated by these observations, we propose a novel language model architecture incorporating an
explicit memory bank. The central innovation involves the systematization of traditionally implicit
knowledge into an explicit and interpretable management framework. By introducing accessible
Memory Bank layers at each model layer, we enable dynamic retrieval and utilization of external
knowledge while, more importantly, achieving transparent knowledge management. Our main con-
tributions are summarized as follows:
• We propose an explicit knowledge storage architecture based on Memory Banks, enabling
each knowledge entry in the model’s repository to be decoded into human-readable text
format, fundamentally addressing the interpretability limitations of traditional models.
• We design a differentiable two-stage retrieval mechanism that combines discrete knowl-
edge selection with continuous gradient flow, enabling end-to-end training of the memory-
augmented architecture while maintaining retrievable interpretability and low computa-
tional cost.
2

Published as a conference paper at ICLR 2026
• We propose ExplicitLM, a novel architecture that enables explicit retrieval and interpreta-
tion of model knowledge while achieving superior answer accuracy compared to standard
Transformer baselines.
2 RELATEDWORK
2.1 LLMARCHITECTURE DEVELOPMENT
The evolution of large language model architectures began with BERT Devlin et al. (2019), which in-
troduced bidirectional pre-training through masked language modeling, while GPT-2 Radford et al.
demonstrated the power of scaling autoregressive transformers. T5 Raffel et al. (2020) unified vari-
ous NLP tasks into a text-to-text framework, and GPT-3 Brown et al. (2020) showed emergent few-
shot learning capabilities at 175B parameters. Subsequent developments include PaLM Chowdhery
et al. (2023) scaling to 540B parameters with improved training efficiency, LLaMA Touvron et al.
(2023) achieving strong performance with smaller models through careful data curation, and GPT-4
Achiam et al. (2023) advancing multimodal capabilities. Recent architectural innovations have ex-
plored alternatives to standard transformers: RWKV Peng et al. (2023) combines RNN efficiency
with transformer-level performance through linear attention mechanisms, Mamba Gu & Dao (2023)
leverages selective state space models for efficient long-context modeling with linear complexity,
while Mixtral Jiang et al. (2024) employs sparse mixture-of-experts for improved parameter effi-
ciency.
2.2 KNOWLEDGEEDITING ANDUPDATING
Knowledge editing in large language models to eliminate errors remains an emerging research
area, with existing approaches divided into parameter-efficient and parameter-augmented meth-
ods. Parameter-efficient approaches focus on updating knowledge without additional parameters:
Li et al. (2023) introduces KAFT (Knowledge-Augmented Fine-Tuning), a data augmentation strat-
egy incorporating diverse contexts (relevant, irrelevant, and counterfactual) for fine-tuning to reduce
knowledge errors, while Onoe et al. (2023) constructs datasets to evaluate whether different meth-
ods can successfully inject specific facts and enable reasoning based on them. Parameter-augmented
methods introduce additional components: Dong et al. (2022) employs CaliNet, training key-value
calibration memory slots with similar architecture to FFN but smaller intermediate dimensions;
Wang et al. (2024) embeds memory pools containing compressed knowledge tokens at each layer
with update functions, though lacking interpretability; Mitchell et al. (2022) prepends a knowledge
classifier to existing models, routing queries to either an explicitly stored and updatable database
with a specialized model or the standard LLM, achieving explicit storage but sacrificing end-to-end
neural architecture coherence.
3 MEMORYBANK
3.1 MEMORYTHEORY
Drawing from dual-system cognitive theory Gowda et al. (2025), we partition language model
knowledge into two distinct yet complementary phases analogous to human procedural-declarative
memory dichotomy.
Implicit Knowledge: This encompasses linguistic grammar rules, syntactic structures, and semantic
associations that resist explicit formalization. Examples include nuanced aspects of human expres-
sion patterns and implicit connections between complex concepts that emerge from cultural and
contextual understanding. Such knowledge exhibits high abstraction and ambiguity, necessitating
statistical learning from large-scale data.
Explicit Knowledge: This comprises factual knowledge, entity relationships, and time-sensitive in-
formation amenable to explicit representation. Examples include ”The President of the United States
is Trump” (not Biden) and ”The Eiffel Tower stands 324 meters tall.” Such knowledge possesses
clear truth conditions and update requirements, making it suitable for storage in editable memory
banks.
3

Published as a conference paper at ICLR 2026
This dual-system design offers distinct advantages: implicit knowledge, acquired through deep
learning, ensures robust language understanding and generation capabilities; explicit knowledge,
through structured storage, enables interpretability and updatability. The synergistic integration of
both systems enables models to maintain powerful linguistic capabilities while flexibly managing
and updating factual knowledge.
Figure 1: Overall architecture of ExplicitLM. The blue region shows the multi-layer transformer
blocks. The gray region represents the shared Memory Bank accessed by all layers, where each
layer can retrieve knowledge via the Memory Retrieval Mechanism (Section 3.4) from Explicit
Knowledge (green) or Implicit Knowledge (yellow) partitions. The orange region shows a sample
knowledge entry from the Memory Bank—a sequence of token indices of lengthLdirectly convert-
ible to human-readable text.
3.2 STORAGEARCHITECTURE
LetM ⊆Z1×L,|M|=Ndenote our Memory Bank tensor, whereN= 106represents the
knowledge capacity andL= 16denotes the maximum sequence length. Each entrym i∈ Mstores
a discrete knowledge unit as token indices, with elementsm ij∈ V, whereVis codebook.
We employ a tokenizer-based bidirectional mapping scheme. The encoding function Tokenize:
S →Z1×Lconverts knowledge stringss∈ Sto token indices for storage:m i=Tokenize(s i) =
[t(i)
1, t(i)
2, ..., t(i)
L]wheret(i)
j∈ V. During retrieval, the embedding function Embed:Z1×L→
Rd×Ltransforms stored indices back to semantic representations:E i=Embed(m i) =
[et(i)
1,et(i)
2, ...,et(i)
L], whereetj(i)∈Rd.
3.3 KNOWLEDGEALLOCATIONSTRATEGY
Given the memory constraint|M|=N, our approach maintains a fixed-capacity knowledge reposi-
tory throughout the model’s lifecycle. This design choice ensures predictable memory consumption
and eliminates the computational overhead associated with dynamic memory allocation. To ef-
fectively utilize this fixed capacity while preserving essential linguistic knowledge, we introduce a
partitioning scheme that divides the memory bank into two disjoint subsets:M=M f∪M uwhere
Mf∩ M u=∅. The partition is controlled by a freeze rate parameterρ∈[0,1], which determines
the proportion of memory allocated to each subset.
The frozen knowledge subsetM fwith cardinality|M f|=ρN(we empirically setρ= 0.2as
default) is designated for storing explicit knowledge that can be precisely formulated and verified.
During initialization, this subset is populated with curated factual information such as entity re-
lationships, geographical facts, and time-sensitive data that require accurate representation. The
explicit nature of this knowledge allows for direct injection of verified information into the memory
bank, ensuring factual accuracy from the outset. These entries remain immutable during training
to preserve the integrity of the pre-verified knowledge base. Conversely, the updatable knowledge
subsetM uwith cardinality|M u|= (1−ρ)Nis allocated for implicit knowledge that the model
must discover through training. This subset captures linguistic regularities, syntactic patterns, and
semantic associations that emerge from statistical learning over large-scale text corpora. The model
4

Published as a conference paper at ICLR 2026
autonomously determines which grammatical structures and language patterns warrant storage in
this dynamic portion of the memory bank. The in-place substitution mechanism maintains the in-
variant|M(t)|=Nfor all time stepst, as updates neither insert new entries nor delete existing
ones, thereby preserving constant memory footprint and eliminating the complexity associated with
dynamic memory management operations.
To address the gradient discontinuity issue that arises from direct overwriting of knowledge entries
inM u, we adopt the Exponential Moving Average (EMA) technique from Vector Quantized Vari-
ational Autoencoders (VQ-V AE) Van Den Oord et al. (2017), originally developed for codebook
updates. Rather than performing abrupt replacements, the EMA mechanism enables progressive
updates that maintain training stability. Specifically, for each knowledge entrym i∈ M u, we
maintain dynamic statistics that allow smooth transitions between old and new knowledge represen-
tations. The update rule assigns higher weights to newer information while preserving continuity
with existing knowledge, enabling the memory bank to adapt to evolving encoder outputs without
introducing disruptive oscillations. This approach effectively circumvents the non-differentiability
inherent in discrete quantization operations, while simultaneously improving both the utilization rate
of knowledge entries and the overall reconstruction quality of the stored information.
3.4 MEMORYRETRIEVALMECHANISM
We propose a hierarchical two-stage retrieval strategy for efficient access to million-scale entries.
Figure 2: ExplicitLM architecture with memory retrieval mechanism. In Stage 1, both query and key
vectors are partitioned along the embedding dimension into two components for efficient retrieval.
In Stage 2, cosine similarity is computed between the query and candidate knowledge entries, with
the highest-scoring entry selected for retrieval.
Stage 1: Key-value Filtering.Following Million ExpertsHe (2024), we assign product keysK:=
{ki}N
i=1⊂Rdto knowledge entries, with query networkqmapping inputxto queryq(x). This stage
generates a candidate setIby retrieving the most relevant entries based on query-key similarities:
I=top-I-indices 
{q(x)⊤k|k∈K}
, where top-I-indices denotes the operator that selects the
indices of the top-Ielements fromK, yielding a candidate set with cardinality|I|. To address
computational complexity atN≥106, we decompose keys using Cartesian products:K={[c;c′]|
c∈C, c′∈C′}whereC,C′⊂Rd/2with|C|=|C′|=√
N, reducing complexity fromO(N·|I|)
toO(√
N· |I|),where|I| ≪√
N.
Stage 2: Similarity Selection.For candidatesi∈I, we compute cosine similaritiescs i=
cos (q(x), k i)and apply Gumbel-Softmax for differentiable selection:
pi=exp ((cs i+gi)/τ)P
j∈Iexp ((cs j+gj)/τ)(1)
whereg i=−log (−log (ϵ i))withϵ i∼Uniform(0,1)and temperatureτ. The straight-through
estimator enables gradient flow: forward pass selectsm selected =m ˆiwhere ˆi= arg max ipi, while
backward pass uses soft weights∂L
∂q(x)=P
i∈Ipi∂L
∂mi, maintaining discrete selection while ensur-
ing end-to-end differentiability for retrieved knowledgem selected ∈ M.
5

Published as a conference paper at ICLR 2026
Unlike traditional RAG systems that rely on frozen retrieval components, our mechanism enables
joint optimization of retrieval and generation through differentiable selection, allowing the model to
learn task-specific retrieval patterns during training.
3.5 JOINTOPTIMIZATIONOBJECTIVE
We design a multi-task learning framework that jointly optimizes three complementary losses to
balance language modeling capability with effective memory retrieval.
Language Modeling Loss.Following standard practice, we minimize cross-entropy between pre-
dicted and ground-truth distributions. For sequencex= (x 1, . . . , x T)with vocabularyVof size
V:
LCE=−1
TTX
t=1logp(x t|x<t,M)(2)
wherep(x t|x<t,M)denotes the model’s predicted probability conditioned on contextx <tand re-
trieved memories fromM.
Memory Relevance Loss.To ensure semantic alignment between queries and retrieved memo-
ries, we maximize weighted cosine similarities. Given queryq(x)∈Rdand retrieved candidates
{Ei}|I|
i=1with selection weights{p i}|I|
i=1from Gumbel-Softmax:
Lsim=−E x∼D
|I|X
i=1pi·q(x)TEi
∥q(x)∥ 2∥Ei∥2
 (3)
This loss guides the model toward selecting contextually relevant memories by reinforcing high-
similarity retrievals.
Memory Diversity Loss.To prevent retrieval collapse into local regions and expand semantic cov-
erage, we minimize pairwise similarities among the candidates. Let ˆEi=E i/∥Ei∥2denote nor-
malized embeddings:
Ldiv=2
|I|(|I| −1)|I|X
i=1|I|X
j=1,j̸=ics(ˆEi,ˆEj)(4)
This regularization encourages exploration across diverse memory regions, preventing locally opti-
mal retrieval patterns while maintaining relevance through balanced optimization.
The final objective combines all losses:L total=L CE+λ simLsim+λ divLdiv. This joint optimization
ensures: (1) accurate next-token prediction throughL CE, (2) semantically coherent retrieval via
Lsim, and (3) diverse memory exploration throughL div, yielding an end-to-end trainable knowledge-
augmented architecture where memory retrieval and language modeling are deeply integrated.
4 EXPERIMENTS
4.1 DATASETCONSTRUCTION
We construct a 10M-entry multi-source pretraining corpus with strategic sampling ratios optimized
for knowledge diversity:Wikipedia: Structured encyclopedic knowledge annotated with entity
triplets for explicit knowledge graph extraction. These entries form the exclusive source for Memory
Bank initializationM ⊆Z1×L, selected based on knowledge density metrics and factual reliability
scores.Project Gutenberg: Literary and historical texts providing formal language patterns and
narrative structures spanning multiple centuries.OpenWebText: Contemporary web text capturing
modern linguistic phenomena and informal discourse patterns.
Each entry maintains a unique identifier for provenance tracking. The Memory Bank entriesm iare
mapped to source UUIDs, enabling systematic knowledge updates and verification. Selection crite-
ria prioritize: (i) token-level information density, (ii) factual accuracy via cross-reference validation,
and (iii) domain coverage measured by entity distribution.
6

Published as a conference paper at ICLR 2026
4.2 EVALUATIONTASKDESIGN
We design three complementary tasks to evaluate knowledge utilization from Memory BankM:
(i) Object Prediction: Given subject-predicate pairs from knowledge entriesm i∈ M, predict
correct object tokenst jifrom candidate set. Accuracy measures entity relationship understanding
with 5 distractors inRdspace.(ii) Relation Reasoning: Given entity token pairs(t ji, tki)from
mi, infer their semantic relationship. This probes compositional reasoning over stored knowledge
structures inM.(iii) Fact Verification:Binary classification of statements derived from memory
bank domain. Negative samples generated via token substitution at indicesm i,jmaintain50 : 50
class balance. Data partitioning leverages freeze partition: test samples derive exclusively from
frozen entriesM fwhere|M f|=ρN, while training excludes all tokens fromm i∈ M f. This
strict disjoint constraint betweenM fand training data prevents memorization-based evaluation
inflation.
4.3 COMPARISON OFDIFFERENTDATAVOLUMES
To systematically evaluate the efficacy of our memory-augmented architecture, we conduct con-
trolled experiments across varying supervised fine-tuning (SFT) data volumes. Both our model and
the baseline Transformer undergo identical optimization procedures, with performance assessed on
the three tasks defined in Section 4.2. The baseline represents a standard Transformer architecture
without memory augmentation, enabling direct attribution of performance gains to our proposed
Memory Bank mechanism.
Table 1: Performance comparison between baseline Transformer and our memory-augmented model
across different SFT data volumes. Results show accuracy (%) on three knowledge-intensive tasks.
Data Volume Model Object Prediction Relation Reasoning Fact Verification
10kBaseline 7.86% 38.27% 61.71%
Ours 28.42%↑20.56%70.02%↑31.75%66.03%↑4.32%
25kBaseline 22.16% 79.99% 71.49%
Ours 63.12%↑40.96%87.85%↑7.86%79.79%↑8.33%
50kBaseline 30.23% 83.80% 83.34%
Ours 73.90%↑43.67%90.41%↑6.61%86.25%↑2.91%
75kBaseline 40.64% 87.66% 86.40%
Ours 79.76%↑39.12%92.12%↑4.46%88.74%↑2.34%
100kBaseline 56.80% 91.91% 88.92%
Ours 80.94%↑24.14%92.73%↑0.82%89.75%↑0.83%
The experimental results reveal pronounced performance advantages in low-data regimes. At 10k
training samples, our model achieves 3.62× improvement in Object Prediction and 1.83× improve-
ment in Relation Reasoning compared to the baseline. This substantial gap demonstrates that ex-
plicit memory retrieval fromMeffectively compensates for limited training exposure, particularly
for tasks requiring precise entity-level knowledge recall. The Object Prediction task, which directly
queries stored triplets from memory entriesm i, exhibits the most consistent improvements across
all data scales (24.14% at 100k samples), validating our retrieval mechanism’s effectiveness in ac-
cessing specific tokenst jifrom the Memory Bank.
4.4 MEMORYBANKHITRATEANALYSIS
To empirically validate the effectiveness of our Memory Bank retrieval mechanism, we conduct a
fine-grained analysis of layer-wise memory access patterns. Using models trained with varying data
volumes from Section 1, we examine the correlation between successful memory retrieval and task
performance on Relation Reasoning. For each forward pass, we track whether the retrieval mech-
anism successfully matches relevant entries fromMat each transformer layer, providing insights
into how different layers utilize external memory.
The aggregate hit rates reveal a strong correlation between memory access success and prediction
accuracy. Models trained on 100k, 50k, 25k, 10k, and 5k samples achieve overall hit rates of 71%,
65%, 66%, 71%, and 71% respectively for correctly answered samples, where a sample is con-
7

Published as a conference paper at ICLR 2026
Figure 3: Layer-wise memory hit rates for Relation Reasoning across varying training data volumes.
Semi-transparent regions indicate hit rates for correctly predicted samples, while opaque regions
show hit rates for incorrect predictions. Red annotations display the hit rate differential between
correct and incorrect predictions at each layer.
sidered to have ”hit” if at least one layer successfully retrieves relevant memory. In stark contrast,
incorrectly answered samples exhibit substantially lower hit rates of 23%, 21%, 21%, 22%, and 37%
respectively. This 3× differential in hit rates between correct and incorrect predictions empirically
confirms that successful memory retrieval directly contributes to task performance.
Figure 3 presents the layer-wise decomposition of hit rates, revealing distinct retrieval patterns across
the network depth. Both correct and incorrect samples exhibit elevated hit rates at layers L1 and L3,
suggesting these layers serve as critical junctures for knowledge integration. The consistency of this
pattern across different training data volumes indicates an emergent specialization in the network
architecture, where specific layers develop stronger affinity for external memory access.
4.5 IMPACT OFFREEZERATE ONPERFORMANCE
To investigate freeze rate parameterρeffects on model performance, we conduct systematic ex-
periments varyingρwhile maintaining other hyperparameters constant. The freeze rate controls
partition between frozen knowledgeM fand updatable knowledgeM u, with|M f|=ρNand
|Mu|= (1−ρ)N. We evaluate performance on Relation Reasoning across different training data
volumes to understand how explicit-implicit knowledge balance affects learning dynamics.
Figure 4 demonstrates that our memory-augmented architecture consistently outperforms the base-
line across all freeze rate configurations. Most pronounced improvements emerge in low-data
regimes: with 10k training samples, our method achieves minimum 83% improvement regardless
ofρ, highlighting Memory Bank mechanism robustness to hyperparameter selection. Even at 100k
samples where baseline reaches 91.91% accuracy, our approach maintains 0.3%-3.3% improve-
ments, confirming explicit memory benefits persist when parametric learning approaches saturation.
The freeze rate-performance relationship exhibits non-monotonic patterns, with optimal perfor-
mance atρ= 0.4across most training set sizes. This peak suggests critical balance between
explicit knowledge preservation inM fand implicit knowledge adaptation inM u. Lower freeze
rates (ρ <0.4) potentially compromise core linguistic knowledge stability, allowing excessive up-
dates corrupting fundamental representations. Higher freeze rates (ρ >0.4) restrict model capacity
to incorporate task-specific patterns through gradient-based learning, limiting domain-specific adap-
tation. This trade-off validates our architectural design where frozen entries preserve high-fidelity
factual knowledge while updatable entries accommodate evolving linguistic patterns, with optimal
partition emerging empirically at approximately 40% frozen knowledge allocation.
8

Published as a conference paper at ICLR 2026
Figure 4: Performance comparison across different freeze rates. The bar chart shows accuracy
values under various experimental conditions with different training set sizes. The line plot indicates
the relative performance improvement (in percentage) of our method compared to the baseline at
different freeze rates for each training set size.
4.6 IMPACT OFPERFECTRETRIEVAL ONMODELPERFORMANCE
To quantify the potential performance gains from improved retrieval accuracy, we conduct controlled
experiments comparing autonomous retrieval (Retain) against surgical replacement (Replace) of re-
trieval results. Based on the critical layers identified in Section 4.5, we intervene at layers L1 and L3
by replacing the top-ranked candidate from the 16 retrieved entries with the oracle knowledge entry
most relevant to the correct answer. This experimental design isolates the effect of retrieval qual-
ity from other architectural components, providing an upper bound on performance improvements
achievable through enhanced retrieval mechanisms.
Table 2: Accuracy comparison between autonomous retrieval (Retain) and surgical replacement
of retrieval results at specific layers (Replace) to evaluate the impact of perfect retrieval on model
performance.
Data Volume Model Object Prediction Relation Reasoning Fact Verification
50kRetain 70.87% 89.87% 85.12%
Replace 74.49%↑3.62%92.12%↑2.25%87.24%↑2.12%
75kRetain 77.12% 90.25% 88.00%
Replace 79.85%↑2.73%91.87%↑1.62%90.25%↑2.25%
100kRetain 79.12% 90.50% 90.37%
Replace 81.00%↑1.88%92.25%↑1.75%91.12%↑0.75%
Table 2 demonstrates consistent improvements across all tasks when perfect retrieval is guaranteed,
with an average accuracy gain of 2.11 percentage points. The Object Prediction task exhibits the
largest improvements (3.62% at 50k samples), consistent with its direct dependence on retrieving
specific factual entries fromM. This task directly queries token sequencesm ifor entity relation-
ships, making it most sensitive to retrieval precision. Relation Reasoning shows moderate gains
(2.25% at 50k, 1.75% at 100k), suggesting that compositional reasoning benefits from accurate
knowledge retrieval but also relies on learned transformations within the network. The diminishing
returns observed at larger training volumes (100k samples) indicate that models with more exten-
sive training develop compensatory mechanisms for imperfect retrieval. The average improvement
decreases from 2.66% at 50k samples to 1.46% at 100k samples, suggesting that larger training sets
enable the model to learn robust representations that partially mitigate retrieval errors.
9

Published as a conference paper at ICLR 2026
5 CONCLUSION
We presented ExplicitLM, a novel language model architecture that fundamentally transforms
knowledge storage from implicit distributed representations to an explicit, interpretable Memory
Bank system. Our approach addresses critical LLM limitations—knowledge staleness, lack of in-
terpretability, and update difficulties—by introducing dual-system design partitioning knowledge
into frozen explicit entries and updatable implicit components. Comprehensive experiments demon-
strated that ExplicitLM consistently outperforms baseline Transformers across knowledge-intensive
tasks, with20−40%improvements in low-data regimes and maintained advantages at scale. Layer-
wise hit rate analysis confirmed successful memory retrieval directly correlates with prediction ac-
curacy, validating our two-stage differentiable retrieval mechanism. While current implementation
requires manual curation of explicit knowledge entries, this limitation points to promising future
directions: developing mechanisms to automatically extract and update explicit knowledge from
training data while preserving human readability and interpretability. Such advances would enable
models to continuously expand verifiable knowledge bases during training, combining statistical
learning benefits with transparent, editable knowledge management—crucial for building trustwor-
thy, maintainable AI systems for real-world deployment.
REFERENCES
Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Ale-
man, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. Gpt-4 technical
report.arXiv preprint arXiv:2303.08774, 2023.
Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal,
Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are
few-shot learners.Advances in neural information processing systems, 33:1877–1901, 2020.
Jeffrey Cheng, Marc Marone, Orion Weller, Dawn Lawrie, Daniel Khashabi, and Benjamin
Van Durme. Dated data: Tracing knowledge cutoffs in large language models.arXiv preprint
arXiv:2403.12958, 2024.
Aakanksha Chowdhery, Sharan Narang, Jacob Devlin, Maarten Bosma, Gaurav Mishra, Adam
Roberts, Paul Barham, Hyung Won Chung, Charles Sutton, Sebastian Gehrmann, et al. Palm:
Scaling language modeling with pathways.Journal of Machine Learning Research, 24(240):
1–113, 2023.
Damai Dai, Li Dong, Yaru Hao, Zhifang Sui, Baobao Chang, and Furu Wei. Knowledge neurons
in pretrained transformers. InProceedings of the 60th Annual Meeting of the Association for
Computational Linguistics (Volume 1: Long Papers), pp. 8493–8502, 2022.
Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. Bert: Pre-training of deep
bidirectional transformers for language understanding. InProceedings of the 2019 conference of
the North American chapter of the association for computational linguistics: human language
technologies, volume 1 (long and short papers), pp. 4171–4186, 2019.
Qingxiu Dong, Damai Dai, Yifan Song, Jingjing Xu, Zhifang Sui, and Lei Li. Calibrating factual
knowledge in pretrained language models. InFindings of the Association for Computational
Linguistics: EMNLP 2022, pp. 5937–5947, 2022.
Mohammad Ennab and Hamid Mcheick. Enhancing interpretability and accuracy of ai models in
healthcare: a comprehensive review on challenges and future directions.Frontiers in Robotics
and AI, 11:1444763, 2024.
Mor Geva, Roei Schuster, Jonathan Berant, and Omer Levy. Transformer feed-forward layers are
key-value memories. InProceedings of the 2021 Conference on Empirical Methods in Natural
Language Processing, pp. 5484–5495, 2021.
Shruthi Gowda, Bahram Zonooz, and Elahe Arani. Dual cognitive architecture: Incorporating biases
and multi-memory systems for lifelong learning.Transactions on Machine Learning Research,
2025.
10

Published as a conference paper at ICLR 2026
Albert Gu and Tri Dao. Mamba: Linear-time sequence modeling with selective state spaces.arXiv
preprint arXiv:2312.00752, 2023.
Xu Owen He. Mixture of a million experts.arXiv preprint arXiv:2407.04153, 2024.
Xinyi Hou, Yanjie Zhao, Shenao Wang, and Haoyu Wang. Model context protocol (mcp): Land-
scape, security threats, and future research directions.arXiv preprint arXiv:2503.23278, 2025.
Albert Q Jiang, Alexandre Sablayrolles, Antoine Roux, Arthur Mensch, Blanche Savary, Chris Bam-
ford, Devendra Singh Chaplot, Diego de las Casas, Emma Bou Hanna, Florian Bressand, et al.
Mixtral of experts.arXiv preprint arXiv:2401.04088, 2024.
Youssef Abdel Latif. Hallucinations in large language models and their influence on legal reason-
ing: Examining the risks of ai-generated factual inaccuracies in judicial processes.Journal of
Computational Intelligence, Machine Reasoning, and Decision-Making, 10(2):10–20, 2025.
Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal,
Heinrich K ¨uttler, Mike Lewis, Wen-tau Yih, Tim Rockt ¨aschel, et al. Retrieval-augmented gener-
ation for knowledge-intensive nlp tasks.Advances in neural information processing systems, 33:
9459–9474, 2020.
Daliang Li, Ankit Singh Rawat, Manzil Zaheer, Xin Wang, Michal Lukasik, Andreas Veit, Felix Yu,
and Sanjiv Kumar. Large language models with controllable working memory. InFindings of the
Association for Computational Linguistics: ACL 2023, pp. 1774–1793, 2023.
Zhihao Li, Kun Li, Boyang Ma, Minghui Xu, Yue Zhang, and Xiuzhen Cheng. We urgently need
privilege management in mcp: A measurement of api usage in mcp ecosystems.arXiv preprint
arXiv:2507.06250, 2025.
Kevin Meng, David Bau, Alex Andonian, and Yonatan Belinkov. Locating and editing factual
associations in gpt.Advances in neural information processing systems, 35:17359–17372, 2022.
Eric Mitchell, Charles Lin, Antoine Bosselut, Christopher D Manning, and Chelsea Finn. Memory-
based model editing at scale. InInternational Conference on Machine Learning, pp. 15817–
15831. PMLR, 2022.
Seyed Mahed Mousavi, Simone Alghisi, and Giuseppe Riccardi. Dyknow: Dynamically verify-
ing time-sensitive factual knowledge in llms. InFindings of the Association for Computational
Linguistics: EMNLP 2024, pp. 8014–8029, 2024.
Yasumasa Onoe, Michael Zhang, Shankar Padmanabhan, Greg Durrett, and Eunsol Choi. Can
lms learn new entities from descriptions? challenges in propagating injected knowledge. In
Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume
1: Long Papers), pp. 5469–5485, 2023.
Bo Peng, Eric Alcaide, Quentin Anthony, Alon Albalak, Samuel Arcadinho, Stella Biderman,
Huanqi Cao, Xin Cheng, Michael Chung, Matteo Grella, et al. Rwkv: Reinventing rnns for
the transformer era.arXiv preprint arXiv:2305.13048, 2023.
Michał Perełkiewicz and Rafał Po ´swiata. A review of the challenges with massive web-mined
corpora used in large language models pre-training. InInternational Conference on Artificial
Intelligence and Soft Computing, pp. 153–163. Springer, 2024.
Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, Ilya Sutskever, et al. Language
models are unsupervised multitask learners.
Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi
Zhou, Wei Li, and Peter J Liu. Exploring the limits of transfer learning with a unified text-to-text
transformer.Journal of machine learning research, 21(140):1–67, 2020.
Alireza Salemi and Hamed Zamani. Evaluating retrieval quality in retrieval-augmented generation.
InProceedings of the 47th International ACM SIGIR Conference on Research and Development
in Information Retrieval, pp. 2395–2400, 2024.
11

Published as a conference paper at ICLR 2026
Aditi Singh, Abul Ehtesham, Saket Kumar, and Tala Talaei Khoei. A survey of the model context
protocol (mcp): Standardizing context to enhance large language models (llms). 2025.
Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timoth ´ee
Lacroix, Baptiste Rozi `ere, Naman Goyal, Eric Hambro, Faisal Azhar, et al. Llama: Open and
efficient foundation language models.arXiv preprint arXiv:2302.13971, 2023.
Aaron Van Den Oord, Oriol Vinyals, et al. Neural discrete representation learning.Advances in
neural information processing systems, 30, 2017.
Yu Wang, Yifan Gao, Xiusi Chen, Haoming Jiang, Shiyang Li, Jingfeng Yang, Qingyu Yin, Zheng
Li, Xian Li, Bing Yin, et al. Memoryllm: Towards self-updatable large language models.arXiv
preprint arXiv:2402.04624, 2024.
A APPENDIX
A.1 REPRODUCIBILITYSTATEMENT
We are committed to ensuring the full reproducibility of our work. All experiments presented in this
paper can be reproduced using the code and configurations provided in our anonymous repository
(ExplicitLM). All experiments were conducted on NVIDIA A100 GPUs.
A.2 AI ASSISTANCESTATEMENT
We declare that AI-based tools were used solely for language polishing purposes in this work.
Specifically, after completing the initial draft entirely through human effort, we employed AI as-
sistance exclusively for grammatical refinement and improving the clarity of English expression to
meet academic writing standards. The AI tools did not contribute to: (1) the generation or develop-
ment of research ideas, including the core concept of ExplicitLM and the memory bank mechanism;
(2) the design of experiments or methodology; (3) the analysis or interpretation of results; (4) the
drafting of original content or scientific arguments; or (5) any mathematical derivations or technical
contributions. All intellectual contributions, from conceptualization to initial manuscript prepara-
tion, were performed by the human authors. The use of AI was limited to post-writing language
enhancement, similar to traditional proofreading services, ensuring that non-native English speakers
can present their research with appropriate linguistic quality while maintaining complete authorship
and originality of the scientific content.
12