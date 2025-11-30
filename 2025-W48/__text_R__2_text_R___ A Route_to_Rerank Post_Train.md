# $\text{R}^2\text{R}$: A Route-to-Rerank Post-Training Framework for Multi-Domain Decoder-Only Rerankers

**Authors**: Xinyu Wang, Hanwei Wu, Qingchen Hu, Zhenghan Tai, Jingrui Tian, Lei Ding, Jijun Chi, Hailin He, Tung Sum Thomas Kwok, Yufei Cui, Sicheng Lyu, Muzhi Li, Mingze Li, Xinyue Yu, Ling Zhou, Peng Lu

**Published**: 2025-11-25 06:54:51

**PDF URL**: [https://arxiv.org/pdf/2511.19987v1](https://arxiv.org/pdf/2511.19987v1)

## Abstract
Decoder-only rerankers are central to Retrieval-Augmented Generation (RAG). However, generalist models miss domain-specific nuances in high-stakes fields like finance and law, and naive fine-tuning causes surface-form overfitting and catastrophic forgetting. To address this challenge, we introduce R2R, a domain-aware framework that combines dynamic expert routing with a two-stage training strategy, Entity Abstraction for Generalization (EAG). EAG introduces a counter-shortcut mechanism by masking the most predictive surface cues, forcing the reranker to learn domain-invariant relevance patterns rather than memorizing dataset-specific entities. To efficiently activate domain experts, R2R employs a lightweight Latent Semantic Router that probes internal representations from the frozen backbone decoder to select the optimal LoRA expert per query. Extensive experiments across different reranker backbones and diverse domains (legal, medical, and financial) demonstrate that R2R consistently surpasses generalist and single-domain fine-tuned baselines. Our results confirm that R2R is a model-agnostic and modular approach to domain specialization with strong cross-domain robustness.

## Full Text


<!-- PDF content starts -->

R2R: A Route-to-Rerank Post-Training
Framework for Multi-Domain Decoder-Only
Rerankers
Xinyu Wang1,2⋆, Hanwei Wu1†, Qingchen Hu1,2†, Zhenghan Tai1,3†
Jingrui Tian1, Lei Ding1,4, Jijun Chi1, Hailin He1, Tung Sum Thomas Kwok1,
Yufei Cui2, Sicheng Lyu1,2,5, Muzhi Li6, Mingze Li7, Xinyue Yu1,7
Ling Zhou8, and Peng Lu7
1SimpleWay.AI2McGill University3University of Toronto4University of Manitoba
5Mila6CUHK7Université de Montréal8CG Matrix
†Equal contribution
Abstract.Decoder-only rerankers are central to Retrieval-Augmented
Generation (RAG). However, generalist models miss domain-specific nu-
ances in high-stakes fields like finance and law, and naive fine-tuning
causes surface-form overfitting and catastrophic forgetting. To address
this challenge, we introduce Route-to-Rerank (R2R), a domain-aware
framework that combines dynamic expert routing with a two-stage train-
ing strategy, Entity Abstraction for Generalization (EAG). EAG intro-
duces a counter-shortcut mechanism by masking the most predictive sur-
face cues (entities), forcing the reranker to learn domain-invariant rele-
vance patterns rather than memorizing dataset-specific entities. To effi-
ciently activate domain experts, we design a lightweight Latent Semantic
Router that probes internal representations from the frozen backbone
decoder of our reranker to select the optimal LoRA expert per query.
Extensive experiments across different reranker backbones and diverse
domains (legal, medical, and financial) demonstrate thatR2Rconsis-
tently surpasses generalist and single-domain fine-tuned baselines. Our
results confirm thatR2Ris a model-agnostic and modular approach to
domain specialization with strong cross-domain robustness.
Keywords:Retrieval-Augmented Generation·Domain Adaptation·
Dynamic Routing·LoRA·Invariant Pattern Learning
1 Introduction
The recent progress of generative Large Language Models (LLMs) has trans-
formed NLP and enabled widespread real-world applications. However, despite
their strong capabilities, LLMs still suffer from hallucination, brittle reasoning,
and inconsistent knowledge recall. Retrieval-Augmented Generation (RAG) ad-
dresses these issues by grounding model outputs in external evidence. However,
the reliability of a RAG system ultimately depends on its reranker, which selects
⋆Corresponding author: xinyu.wang5@mail.mcgill.caarXiv:2511.19987v1  [cs.CL]  25 Nov 2025

2 Wang et al.
the documents supplied to the generator [1]. In high-stakes domains such as law
and medicine, accurate reranking is essential for trustworthy performance.
Decoder-only rerankers have become increasingly popular due to their strong
semantic reasoning, inference efficiency, and compatibility with LLM-based re-
trievers [7]. However, most are trained as general-purpose models and struggle
with domain-specific terminology, fine-grained intents, and long-tail knowledge.
Their performance deteriorates under distribution shift [11,13], underscoring the
need for reranking methods that remain robust in high-precision settings.
A common response to domain shift is fine-tuning on domain-specific data,
but this approach often overfits to surface cues (e.g., company names, case IDs)
and causes catastrophic forgetting of general ranking abilities. Evidence of this
behavior is shown in Table A in Appendix A. The model adopts shortcut pat-
terns rather than true relevance logic [6,16]. Maintaining separate, fully fine-
tuned models for each domain is also computationally impractical, and existing
approaches—static adapters or heavy ensembles—struggle to balance specializa-
tion with efficiency [12,8,17].
To bridge this gap, we proposeRoute-to-Rerank (R2R), a lightweight
and modular framework for domain-adaptive reranking.R2Rmaintains a set
of specialized experts implemented by LoRA adaptors [5] and dynamically se-
lects the appropriate expert for each query. Our two-stage training scheme,
Entity Abstraction for Generalization (EAG), abstracts entity mentions
to reduce shortcut learning and then fine-tunes on original data to enable spe-
cialization without forgetting. At inference time, aLatent Semantic Router,
rather than an external classifier [2,10], probes the frozen decoder-only reranker
backbone to identify domain signals and activate the optimal expert without re-
lying on external classifiers. In summary, our main contributions are as follows:
1.Two-stage training withEAG.We design a data curation and training
pipeline that masks surface entities prior to domain specialization, reducing
overfitting and encouraging the model to learn domain-invariant patterns.
2.Latent Semantic Router.We introduce a lightweight router that lever-
ages thefrozenreranker backbone to dynamically activate LoRA experts,
eliminating the need for additional feature extraction modules.
3.Model-Agnostic Effectiveness.We demonstrate thatR2Rconsistently
improves performance across multiple domains and reranker architectures,
including Qwen3-Reranker [18] and BGE-Reranker [9]. These results high-
lightthegeneralityandadaptabilityofourapproachfordecoder-onlyrerankers.
2 Related Work
2.1 Domain Adaptation and Parameter-Efficient Mining
While RAG systems have improved LLM reliability, they remain vulnerable
to domain distribution shifts. Benchmarks in high-stakes fields like law and
medicinerevealthatgeneralistrerankersstruggletodistinguishfine-grainedrele-
vancesignalsamidstspecializedterminology[11,13].Toaddressthis,Parameter-
Efficient Fine-Tuning (PEFT) methods, e.g., LoRA [5], have been adopted to

R2R: Post-Training Framework for Multi-Domain Decoder-Only Rerankers 3
Fig.1: The impact of accurate domain routing on reranking quality. (A) A
domain-aware router correctly activates a LoRA expert for a given query, max-
imizing in-domain expertise and precision. (B) Expert selection without proper
routing results in domain mismatch and suboptimal reranking performance.
inject domain knowledge without retraining the full backbone. However, naive
PEFT often leads to overfitting on surface forms (e.g., specific names) or catas-
trophic forgetting of general capabilities [6,16]. In contrast, our work treats
adaptation as a robust pattern mining task, employing adversarial EAG to force
the model to learn invariant structural matching patterns.
2.2 Dynamic Routing and Conditional Computation
Dynamic computation, or the ability to conditionally activate network modules,
offers a pathway to efficient multi-domain adaptation. This paradigm shares
roots with Mixture-of-Experts (MoE) frameworks [4,12,20]. In retrieval, recent
works like RagRouter [17] and LoRA-Switch [8] try to route queries to different
adapters, but they often rely on external classifiers or shallow embeddings that
miss deeper semantic intent [10]. In contrast, ourR2Rframework introduces a
Latent Semantic Routerthat inspects the frozen reranker’s internal representa-
tions to precisely activate the right LoRA expert without additional overhead.
3 Preliminaries and Problem Formulation
3.1 Generative Reranking Formulation
We utilize a decoder-only LLM as the backbone for relevance estimation, formu-
lating reranking as an instruction-aware next-token prediction problem.
Input & Architecture.Given a queryq, documentc, and instructionI, we
construct the input sequencexvia a templateT:x= [<Instruct>:I;<Query>:
q;<Document>:c]. The sequence is processed byLtransformer layers. For a
hidden stateH(l), the layer output is:
H(l+1)=FFN(LN( ˜H(l))) + ˜H(l),where ˜H(l)=MHSA(LN(H(l))) +H(l).(1)

4 Wang et al.
Fig.2:OverviewoftheR2Rframework.Top:ThefullRoute-to-Rerankpipeline.
The two-stage EAG curriculum first abstracts entities to learn invariant rele-
vance patterns, then specializes on original domain data to produce domain-
specific LoRA experts. During inference, the Latent Semantic Router probes
the frozen backbone to select the appropriate expert.Bottom:The LoRA-
augmented transformer block, where lightweight domain-specific LoRA adapters
attach to the frozen reranker and are dynamically activated by the router.
Here, MHSA denotes Multi-Head Self-Attention utilizing the standard scaled
dot-product mechanism.
Relevance Quantification.We extract the logit vectorz∈RVcorresponding
to the last token ofx. Letv yes, vnobe the indices for tokens “Yes” and “No”. The
relevance scores(q, c)is computed via a binary Softmax:
s(q, c) =exp(z yes)
exp(z yes) + exp(z no)=σ(z yes−zno).(2)
This maps the generative capability of the LLM to a discriminative ranking score
s∈(0,1), whereσ(·)denotes the sigmoid function.

R2R: Post-Training Framework for Multi-Domain Decoder-Only Rerankers 5
3.2 Parameter-Efficient Adaptation (LoRA)
LoRA [5] prevents catastrophic forgetting by freezing the pretrained weights
W0∈Rd×kand injecting trainable low-rank matricesA∈Rr×k, B∈Rd×r
(r≪d). The forward pass is modified as:
h=W 0x+α
rBAx,(3)
whereαis a scaling hyperparameter. For notational simplicity, we omit the scal-
ing factorα/rin subsequent sections and implicitly absorb it into the update
term∆W. This modular design allows us to encapsulate domain-specific knowl-
edge into lightweight expert modules∆W k, serving as the basis for our routing
framework.
3.3 Problem Formulation
LetD={d gen} ∪ {d k}K
k=1be the domain set. The goal is to learn a scoring
functions θ(q, c)thatranksrelevantcandidateshigherthannegatives,formulated
over training tripletsτ= (q, c+,{c−}).
Standard fine-tuning optimizes static parametersθ∗directly on target data.
This approach suffers fromShortcut Learning, where models overfit surface
forms rather than invariant structures (Appendix A), andLatent Ambiguity,
as the domain indexkis unobserved during inference.
To address this, we propose adynamic parameterizationθ(q) =θ base+
∆θϕ(q), where∆θrepresents the trainable LoRA experts (defined as∆Win
Sec. 3.2) andϕ(q)infers the latent domain. We formulate the training objec-
tive as a stepwise optimization, prioritizing global structural invariance before
domain specialization:
min
Θ,ϕEτ∼P abstract [Lrank]| {z }
Global Structural Invariance+KX
k=1Eτ∼P(k)
target[Lrank]
| {z }
Domain Specialization.(4)
Here, the first term utilizes a global entity-abstracted distributionP abstractto
force the model to learn invariant patterns. The second term refines the model
on distinct domain distributionsP(k)
targetfor precision.R2Rapproximates this
joint objective sequentially:Entity Abstraction for Generalizationfirst op-
timizes the abstract term (Stage 1) to establish a robust structural foundation,
followed by the target term (Stage 2) for domain injection (Section 4.1), while
theLatent Semantic Routerresolves the assignmentϕ(q)→k(Section 4.3).
4 Methodology: Route-to-Rerank (R2R)
Our proposedRoute-to-Rerank (R2R)method consists of two components:
(1) a two-stage training strategy:Entity Abstraction for Generalization
(EAG), and (2) aLatent Semantic Routerfor dynamic LoRA expert se-
lection.

6 Wang et al.
Procedure 1:Domain Dataset Curation Strategy
Input:RetrieverR; Target queriesQ targetand corpusC target.
Output:Abstract domain datasetD abstractand specific datasets{D(k)
target}.
1D abstract ← ∅;
2foreachtarget domainkdo
3D(k)
target← ∅;
4foreachqueryq∈Q(k)
targetdo
5Candidates← R(q,C target );
6(q,P q)←LLM_Annotate(q, Candidates);
7Nhard
q←Candidates\ P q;
8Nrand
q←SampleRandom(C target\ Pq);
9N q← Nhard
q∪ Nrand
q;
10Add(q,P q,Nq)toD(k)
target;
11D abstract ←D abstract ∪ApplyAbstraction(D(k)
target );
12returnD abstract ,{D(k)
target};
4.1 Mining Invariant Patterns viaEAG
EAGmitigates overfitting to surface entities by progressively training the model
through two stages.Stage 1 (Counter-shortcut Entity Abstraction)con-
structs an abstract datasetD abstractby replacing domain-specific named entities
with randomized, type-consistent placeholders (e.g., "Zeekr"→[COMPANY_A]).
This encourages the model to learn structural relevance patterns rather than
memorize specific names (i.e., taking "shortcuts"). By structural relevance pat-
terns, we mean the relational structures among entities that indicate relevance,
such as company–product links in finance, case–statute correspondences in law,
ordisease–symptomcausalrelationsinmedicine.Afteracquiringstructuralcom-
petence,Stage 2 (Domain Specialization)fine-tunes the model on the orig-
inal, unmasked target datasetD target, injecting precise domain knowledge while
preserving general reasoning ability.
AutomatedDatasetCurationTosupportthispipeline,weemployaretriever-
guided data curation process. For each queryq, we construct a training triplet
(q,P q,Nq), where the positve setP qis annotated by an LLM, while the nega-
tive setN qis composed of bothHard Negatives(irrelevant chunks with high
retrieval scores) andRandom Negatives. This combination ensures the model
learns to discriminate fine-grained semantic differences while maintaining broad
separability. The detailed curation process is outlined in Algorithm 1.
4.2 Optimization Objective
We train the LoRA experts using a contrastive learning objective. Given a query
q, a positive chunkc+, and a set of negatives{c−
j}N
j=1, the model computes

R2R: Post-Training Framework for Multi-Domain Decoder-Only Rerankers 7
Procedure 2:Two-Stage EAG Fine-Tuning
Input:Base modelθ base; Abstract dataD abstract; Target dataD target.
Output:Specialized LoRA parameters∆θ expert.
1FunctionContrastiveTrain(θ,D):
2whilenot convergeddo
3Sample batch(q, c+,{c−})fromD;
4Compute scoressusing Eq. (5);
5ComputeL contrastive ;
6Updateθvia gradient descent;
7returnθ;
// Stage 1: Learn Invariant Structure
8θ general ←ContrastiveTrain(θ base, Dabstract );
// Stage 2: Inject Domain Knowledge
9θ expert←ContrastiveTrain(θ general , Dtarget );
10returnθ expert;
Procedure 3:Router Training via Latent Probing
Input:Query-Domain pairs{(q i, di)}; Frozen backbonef θ; Router params
ϕ={W r, br}.
Output:Trained router parametersϕ.
1foreachbatch(q, d)do
2h q←ExtractLastToken(f θ(q));
3ˆp←softmax(W rhq+br);
4L router← −Pdlog ˆp;
5Updateϕto minimizeL router;
6returnϕ;
relevance scoress(q, c). The loss function minimizes the negative log-likelihood
of the positive chunk:
Lcontrastive =−logexp(s(q, c+)/τ)
exp(s(q, c+)/τ) +PN
j=1exp(s(q, c−
j)/τ),(5)
whereτis a temperature hyperparameter (set to 1.0 by default). This objective
maximizes the margin between relevant and irrelevant evidence. The full two-
stage training procedure is summarized in Algorithm 2, and training setups are
detailed in Appendix B.
4.3 Latent Semantic Router
To enable dynamic expert selection during inference without incurring the la-
tency of external classifiers, we introduce theLatentSemantic Router.Unlike
traditional routing approaches that rely on shallow text embeddings, our router
probes thefrozen backbone’s internal world knowledge.
Recall from Section 3.1 that for any input queryq, the reranker produces
a final-token hidden stateh q∈Rd. This vectorh qcontains a high-dimensional

8 Wang et al.
summary of the query’s semantic intent. We project this representation through
a lightweight routing head:
p(d|q) =softmax(W rhq+br),(6)
whereW r∈RK×dandb r∈RKare the only trainable parameters for the router
andKdenote the number of domains.
Inference Mechanism.During inference, the query is first passed through
the frozen backbone. The router computesp(d|q)and selects the domain expert
k∗= arg max kp(dk|q). The corresponding LoRA module∆W k∗is then dynam-
ically activated to compute the final relevance score.
Router Training.The router is trained via standard cross-entropy loss on
labeled query-domain pairs. Crucially, the backbone remains frozen, ensuring
that the router learns to interpret theexistingsemantic manifold of the LLM.
The detailed training procedure is provided in Algorithm 3.
5 Experiments
Fig.3: Reranker performance across training stages on Lotus and Zeekr datasets
(PT=Pretrained, S1=Stage 1, S2=Stage 2). Dashed lines show direct fine-tuning
(PT+S2),whilesolidlinesshowthetwo-stageEAGpipeline(PT+S1+S2).EAG
consistently outperforms direct fine-tuning across all metrics.

R2R: Post-Training Framework for Multi-Domain Decoder-Only Rerankers 9
This section validates the superiority of the proposedRoute-to-Rerank
(R2R)framework and theEAGtraining strategy across different models and
datasets. Datasets and evaluation metrics are detailed in Appendix C.1.
5.1 Two-StageEAGTraining Evaluation
With setups detailed in Appendix C.2, We evaluate whether the proposed two-
stageEAGpipeline improves reranking quality. Across both benchmarks and
model variants,EAG(PT+S1+S2) consistently outperforms direct fine-tuning
(PT+S2), as shown in Table 1 and Figure 3. The Stage-1 abstraction step pro-
vides stable gains at both @5 and @10, confirming its effectiveness for domain
specialization.
5.2 Router and End-to-End Evaluation
We evaluate routing quality and end-to-end reranking performance across the
router configurations defined in Appendix C.3. Table 2 shows that ourLatent
Semantic Routerhas the highest routing quality. CombiningEAG-trained
experts with our router,R2Rachieves the strongest overall end-to-end results
while maintaining the second lowest parameter overhead.
Table 1: Domain specialization results across two pretrained rerankers
(PT=pretrained). For both datasets and both models,EAGconsistently pro-
vides the largest performance improvements over the pretrained baseline and
outperforms direct fine-tuning.
Dataset ConfigurationNDCG MRR Recall
@5 @10 @5 @10 @5 @10
BAAI/bge-reranker-v2-gemma
LexRAGPT 81.1 82.4 78.6 79.0 90.6 91.6
PT + direct FT 90.4(↑9.3)90.6(↑8.2)89.7(↑11.1)89.6(↑10.6)93.9(↑3.3)94.3(↑2.7)
PT +EAG 92.5(↑11.4)92.6(↑10.2)92.0(↑13.4)92.0(↑13.0)93.9(↑3.3)94.7(↑3.1)
ChatDoctorPT 96.9 96.4 96.2 96.1 100.0 100.0
PT + direct FT 99.0(↑2.1)98.0(↑1.6)97.4(↑1.2)97.3(↑1.2)100.0(=)100.0(=)
PT +EAG 99.0(↑2.1)98.6(↑2.2)98.7(↑1.3)98.2(↑0.9)100.0(=)100.0(=)
Qwen/Qwen3-Reranker-0.6B
LexRAGPT 86.8 87.5 85.7 85.9 92.4 94.4
PT + direct FT 91.3(↑4.5)90.1(↑3.4)90.3(↑4.6)91.2(↑5.3)93.2(↑0.8)94.6(↑0.2)
PT +EAG 95.8(↑9.0)95.9(↑8.4)94.9(↑9.2)96.0(↑10.1)96.7(↑4.3)96.3(↑1.9)
ChatDoctorPT 95.8 94.8 95.1 94.7 100.0 100.0
PT + direct FT 98.4(↑2.6)97.2(↑2.4)98.3(↑3.2)97.7(↑3.0)100.0(=)100.0(=)
PT +EAG 99.0(↑3.2)97.8(↑3.0)98.7(↑3.6)97.9(↑3.2)100.0(=)100.0(=)
Table 2: Routing and end-to-end reranking results under different router con-
figurations (LSR = Latent Semantic Router). The best score is shown inbold
and the second best is underlined . LSR attains the highest routing accuracy and
macro F1, whileR2Rw/ LSR yields the strongest overall reranking performance
with the second lowest parameter overheads.
ConfigurationTrainRouter Router NDCG MRR Recall # Extra
Strat.Acc. (%) Macro F1 @5 @10 @5 @10 @5 @10 Params
1. Pretrained Reranker None N/A N/A 81.3 81.2 78.5 78.0 61.9 67.3 0
2.R2Rw/ Sep. MLP Router EAG 84.3 82.2 87.9 86.6 86.2 85.5 65.8 70.6 6.0B
3.R2Rw/ LLM as Router EAG 97.3 97.3 88.887.387.286.466.271.0 685B
4.R2Rw/ LSR EAG 97.4 97.3 89.0 87.4 87.4 86.6 66.4 71.1 0.2B

10 Wang et al.
6 Conclusion
In this paper, we presented Route-to-Rerank (R2R), a lightweight post-training
framework for domain-aware decoder-only rerankers. The method combines a
backbone-probingrouterwiththeEntity Abstraction for Generalizationcurricu-
lum. This design helps the model specialize within each domain while stay-
ing robust across domains. Experiments across multiple domains and reranker
backbones show clear in-domain gains over generalist baselines and simple fine-
tuning, without sacrificing out-of-domain performance. Our analysis also shows
that probing the frozen backbone with an LM-head classifier leads to much
higher routing accuracy than a standalone MLP. Overall,R2Rprovides a prac-
tical and extensible approach for route-to-rerank domain adaptation in modern
RAG systems.
A Model Catastrophic Forgetting
Table 3 demonstrates that the model’s general reranking capability degrades
after fine-tuning. This observation motivates the need for parameter-efficient
methods that can achieve specialization without compromising generalizability.
Table 3: Reranker (bge-reranker-v2-gemma) performance degradation on new
domains after fine-tuning (4,000 steps) on the source domain (Zeekr and Lotus).
Target Reranker NDCG MRR Recall
Domain Checkpoint @5 @10 @5 @10 @5 @10
LexRAGPretrained 81.1 82.4 78.6 79.0 90.6 91.6
SFT (4,000 steps) 77.7(↓3.4)79.2(↓3.2)74.0(↓4.6)74.6(↓4.4)89.2(↓1.4)88.8(↓2.8)
ChatDoctorPretrained 97.9 97.4 97.2 97.1 100.0 100.0
SFT (4,000 steps) 96.8(↓1.1)97.2(↓0.2)96.1(↓1.1)96.3(↓0.8)98.7(↓1.3)99.7(↓0.3)
B Model Training Setups
All reranker fine-tuning experiments use the same LoRA configuration across
models.Qwen3-Reranker-0.6Bis trained using theSwift[19] framework,
whilebge-reranker-v2-gemmais trained using theFlagEmbedding[3,15]
framework. Both rerankers are fine-tuned with the same LoRA configuration
(rank 32, alpha 64, applied to theq proj,kproj,vproj, ando projlayers).
C Experiment Setups
C.1 Datasets and Evaluation Metrics
We utilize four domain QA datasets to assess the domain adaptation capabilities
of our framework: theLegal Domain (LexRAG)dataset [11], which focuses
on legal case retrieval and consultation; theMedical Domain (ChatDoctor)

R2R: Post-Training Framework for Multi-Domain Decoder-Only Rerankers 11
dataset [13], which consists of dialogues between patients and a specialized med-
ical LLM, encompassing analysis of medical conditions and proposed treatment
plans; and two subdomain datasets from theFinancial Domain (Zeekr and
Lotus [14]), focusing on retrieving information from financial filings.
We use standard information retrieval metrics at cutoffsK= 5andK=
10:NDCG@K,MRR@K,Precision@K, andRecall@K; and we evaluate
the quality of different routing mechanisms withAccuracyandMacro F1
Score. We omit Precision@K for LexRAG and ChatDoctor since their queries
correspond to a single ground truth chunk.
C.2EAGEvaluation Settings
We adopt a two-stage benchmarking process forEAGevaluations. The per-
formance baseline for Stage 1 is the pretrained base reranker. We select the
checkpoint with the highest Precision@5 as the optimal Stage 1 model and use
it as the base model for Stage 2 specialization. The baseline for Stage 2 is defined
by a control experiment: the pretrained base reranker that is directly fine-tuned
on the subdomain-specific dataset. This allows us to quantify the superiority of
the two-stageEAGapproach over immediate subdomain specialization.
C.3 Router and End-to-end Evaluation Settings
We use the same three routing configurations for both routing evaluation and
end-to-endR2Rexperiments, all based on the Qwen3-Reranker-0.6B backbone
and evaluated on the aggregated cross-domain dataset constructed from our
selected domains. (1)MLP Classifieruses a standalone MLP fed by an ex-
ternal embedding model (bge-m3 in end-to-end settings), and its parameter
cost includes both components. (2)LLM-as-Routersends the raw query to
a general-purpose LLM (DeepSeek-V3), representing a high-capacity but API-
dependent routing strategy. (3)Latent Semantic Router(ours) reuses the
reranker’s decoder to encode the query and adds only a lightweight MLP head
on the last-token representation, requiring no external models.
These routing mechanisms are assembled into four end-to-end reranking vari-
ants: a vanilla pretrained reranker (no experts or routing);R2Rwith the MLP
router;R2Rwith the LLM router; andR2Rwith our Latent Semantic Router.
For all variants, we additionally report the total number of extra parameters
introduced, counting both the routing module and all domain LoRA experts.
References
1. Brown, A., Roman, M., Devereux, B.: A systematic literature review of
retrieval-augmented generation: Techniques, metrics, and challenges (2025),
https://arxiv.org/abs/2508.06401

12 Wang et al.
2. Cao, H., Hu, D.H., Shen, D., Jiang, D., Sun, J.T., Chen, E., Yang,
Q.: Context-aware query classification. In: Proceedings of the 32nd Inter-
national ACM SIGIR Conference on Research and Development in Infor-
mation Retrieval. p. 3–10. SIGIR ’09, Association for Computing Machin-
ery, New York, NY, USA (2009). https://doi.org/10.1145/1571941.1571945,
https://doi.org/10.1145/1571941.1571945
3. Chen, J., Xiao, S., Zhang, P., Luo, K., Lian, D., Liu, Z.: Bge m3-embedding:
Multi-lingual, multi-functionality, multi-granularity text embeddings through self-
knowledge distillation (2023)
4. Chen, Q., Wang, C., Wang, D., Zhang, T., Li, W., He, X.: Lifelong knowl-
edge editing for vision language models with low-rank mixture-of-experts (2025),
https://arxiv.org/abs/2411.15432
5. Hu, E.J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang,
L., Chen, W.: Lora: Low-rank adaptation of large language models (2021),
https://arxiv.org/abs/2106.09685
6. Kalajdzievski, D.: Scaling laws for forgetting when fine-tuning large language mod-
els (2024), https://arxiv.org/abs/2401.05605
7. Karpukhin, V., Oğuz, B., Min, S., Lewis, P., Wu, L., Edunov, S., Chen, D.,
tau Yih, W.: Dense passage retrieval for open-domain question answering (2020),
https://arxiv.org/abs/2004.04906
8. Kong, R., Li, Q., Fang, X., Feng, Q., He, Q., Dong, Y., Wang, W., Li, Y., Kong, L.,
Liu, Y.: Lora-switch: Boosting the efficiency of dynamic llm adapters via system-
algorithm co-design (2024), https://arxiv.org/abs/2405.17741
9. Li,C.,Liu,Z.,Xiao,S.,Shao,Y.:Makinglargelanguagemodelsabetterfoundation
for dense retrieval (2023)
10. Li, F., Zhang, X., Yuan, J., Zhu, X.: Classifying what-type questions by
head noun tagging. In: Scott, D., Uszkoreit, H. (eds.) Proceedings of the
22nd International Conference on Computational Linguistics (Coling 2008). pp.
481–488. Coling 2008 Organizing Committee, Manchester, UK (Aug 2008),
https://aclanthology.org/C08-1061/
11. Li, H., Chen, Y., Hu, Y., Ai, Q., Chen, J., Yang, X., Yang, J., Wu, Y., Liu, Z.,
Liu, Y.: Lexrag: Benchmarking retrieval-augmented generation in multi-turn legal
consultation conversation (2025), https://arxiv.org/abs/2502.20640
12. Li, Y., Gao, V., Zhang, C., Torkamani, M.: Ensembles of low-rank expert adapters
(2025), https://arxiv.org/abs/2502.00089
13. Li, Y., Li, Z., Zhang, K., Dan, R., Jiang, S., Zhang, Y.: Chatdoctor: A medical
chat model fine-tuned on a large language model meta-ai (llama) using medical
domain knowledge (2023), https://arxiv.org/abs/2303.14070
14. Wang, X., Chi, J., Tai, Z., Kwok, T.S.T., Li, M., Li, Z., He, H., Hua, Y., Lu,
P., Wang, S., Wu, Y., Huang, J., Tian, J., Mo, F., Cui, Y., Zhou, L.: Fin-
sage: A multi-aspect rag system for financial filings question answering (2025),
https://arxiv.org/abs/2504.14493
15. Xiao, S., Liu, Z., Zhang, P., Muennighoff, N.: C-pack: Packaged resources to ad-
vance general chinese embedding (2023)
16. Xiong, Y., Xie, X.: Oplora: Orthogonal projection lora prevents
catastrophic forgetting during parameter-efficient fine-tuning (2025),
https://arxiv.org/abs/2510.13003
17. Zhang, J., Liu, X., Hu, Y., Niu, C., Wu, F., Chen, G.: Ragrouter: Learn-
ing to route queries to multiple retrieval-augmented language models (2025),
https://arxiv.org/abs/2505.23052

R2R: Post-Training Framework for Multi-Domain Decoder-Only Rerankers 13
18. Zhang, Y., Li, M., Long, D., Zhang, X., Lin, H., Yang, B., Xie, P., Yang, A., Liu,
D., Lin, J., Huang, F., Zhou, J.: Qwen3 embedding: Advancing text embedding
and reranking through foundation models. arXiv preprint arXiv:2506.05176 (2025)
19. Zhao, Y., Huang, J., Hu, J., Wang, X., Mao, Y., Zhang, D., Jiang, Z., Wu, Z., Ai,
B., Wang, A., Zhou, W., Chen, Y.: Swift:a scalable lightweight infrastructure for
fine-tuning (2024), https://arxiv.org/abs/2408.05517
20. Zhuang, Y., Shen, Y., Bian, Y., Su, Q., Ji, S., Shi, Y., Miao, F.:
Ld-mole: Learnable dynamic routing for mixture of lora experts (2025),
https://arxiv.org/abs/2509.25684