# Conflict-Aware Retriever Editing for Knowledge Injection Attacks on LLM-Based RAG Systems

**Authors**: Xinru Liu, Xianglong Zhang, Di Cai, Zhumin Chen, Pengfei Hu, Xin Xin

**Published**: 2026-06-16 09:17:56

**PDF URL**: [https://arxiv.org/pdf/2606.18310v1](https://arxiv.org/pdf/2606.18310v1)

## Abstract
Injecting malicious knowledge into retrieval-augmented generation (RAG) systems can manipulate retrieved evidence and mislead downstream generation, posing a serious security threat for AI applications. Existing RAG injection attacks mainly rely on manipulating external knowledge bases, such as crafting malicious corpus. However, the synthetic text crafted by such data-centric methods could be detectable, leading to the failure of attacks. Beyond corpus manipulation, open-source retrievers are increasingly exposing RAG systems to model-centric attacks. In this paper, we propose conflict-aware retriever editing, i.e., CAREATTACK, a model-centric retriever attack framework for malicious knowledge injection in RAG. Specifically, CAREATTACK consists two stages of conflict-aware retriever editing and attack-preserving anchor repair. Conflict-aware retriever editing adapts efficient closed-form parameter editing to the dense retrieval model, promoting malicious knowledge above benign competing passages and resolving potential parameter conflicts through graph-based conflict detection and parameter editing projection. Then, attack-preserving anchor repair performs lightweight calibration on the edited retriever to further eliminate the impact on non-target prompts while preserving the attack effectiveness for target prompts. We instantiate CAREATTACK on Qwen3-Embedding-0.6B and BGE-M3, and conduct evaluation on three benchmark datasets. Experimental results demonstrate our method substantially promote malicious passages into the retrieved knowledge of RAG systems and can perform attacks for batches of target prompts and passages, given the access of retrieval model parameters. Since most RAG systems are built upon open-source retrieval models, this work reveals a practical attack surface in RAG systems. Codes are public accessible at https://anonymous.4open.science/r/CareAttack-3F1C.

## Full Text


<!-- PDF content starts -->

Conflict-Aware Retriever Editing for Knowledge Injection Attacks
on LLM-Based RAG Systems
Xinru Liu∗, Xianglong Zhang†, Di Cai∗, Zhumin Chen∗, Pengfei Hu∗, and Xin Xin∗*
∗Shandong University, China
†Tsinghua University, China
liuxinru0607@gmail.com, zhangxianglong@mail.tsinghua.edu.cn,
caidi@sdu.edu.cn, chenzhumin@sdu.edu.cn,
phu@sdu.edu.cn, xinxin@sdu.edu.cn
*Corresponding author.
Abstract—Injecting malicious knowledge into retrieval-
augmented generation (RAG) systems can manipulate retrieved
evidence and mislead downstream generation, posing a serious
security threat for AI applications. Existing RAG injection
attacks mainly rely on manipulating the external knowledge
base, such as crafting malicious corpus. However, the synthetic
text crafted by such data-centric methods could often be
detectable, leading to the failure of attacks.
Beyond corpus manipulation, the widespread use of open-
source retrievers is increasingly exposing RAG systems to
model-centric attacks. In this paper, we propose c onflict-a ware
retriever e diting, i.e., CAREATTACK, a model-centric retriever
attack framework for malicious knowledge injection in RAG.
Specifically, CAREATTACKconsists two stages ofconflict-aware
retriever editingandattack-preserving anchor repair. Conflict-
aware retriever editing adapts efficient closed-form parameter
editing to the dense retrieval model, promoting malicious
knowledge above benign competing passages. Besides, for
batches of target prompts, CAREATTACKfurther resolves po-
tential parameter conflicts through graph-based conflict detec-
tion and parameter editing projection. Then, attack-preserving
anchor repair performs lightweight calibration on the edited
retriever to further eliminate the impact on non-target prompts
while preserving the attack effectiveness for target prompts.
We instantiate CAREATTACKon two of the most popu-
lar retrieval models, i.e., Qwen3-Embedding-0.6B and BGE-
M3, and conduct evaluation on three benchmark datasets.
Experimental results demonstrate our method substantially
promote malicious passages into the retrieved knowledge of
RAG systems. CAREATTACKcan perform attacks for batches
of target prompts and passages, given the access of retrieval
model parameters. Since most RAG systems are built upon
open-source retrieval models, this work reveals a practical
attack surface in RAG systems. Codes are public accessible
at https://anonymous.4open.science/r/CareAttack-3F1C.
1. Introduction
RAG injection attacks.Retrieval-augmented generation
(RAG) enhances large language models (LLMs) by re-trieving relevant passages from an external knowledge base
before generation [1], [2]. In a typical RAG pipeline, a dense
embedding retriever maps the user prompt and candidate
passages into a shared representation space, ranks them by
cosine similarity or inner product [3], [4], [5], and deter-
mines the top-kevidence passages exposed to the generator,
as shown in Figure 1.1. Therefore, the retrieved passages
directly determine which external knowledge would influ-
ence the downstream generation. This dependency makes
malicious knowledge injection a serious security threat to
RAG systems. If malicious knowledge is retrieved as sup-
porting evidence, the generator may be misled to produce
attacker-desired or factually incorrect responses. Figure 1.2
shows an example of RAG injection attacks, given the
prompt “What is the email address for customer support?”,
the attacker can promote a malicious “attacker@evil.com”
in the generated response through injecting passages about
“attacker@evil.com” into the retrieved knowledge evidence.
Drawbacks of existing methods.Existing RAG injec-
tion attacks are predominantly data-driven, manipulating re-
trieved contexts, external knowledge corpora, user prompts,
or retriever training/fine-tuning data to induce malicious
retrieval. Knowledge poisoning attacks [6], [7], [8] poison
the retrieval corpus into the external knowledge base, so
that malicious content is more likely to be retrieved for
target prompts. Prompt injection attacks [9], [10] manipulate
prompts or retrieved instructions to affect the RAG pipeline.
Retriever backdoor attacks [11], [12], [13] further poison
retriever training or fine-tuning data to implant trigger-
activated retrieval behaviors into the retriever. However,
these data-driven attacks face practical limitations. Knowl-
edge poisoning and prompt injection attacks expose mali-
cious textual artifacts in the input or external knowledge
base, making them more susceptible to likelihood-based
filtering [7] and retrieval-based defenses [14]. As shown in
Figure 1.3, the malicious passage crafted by PoisonedRAG
exhibits unnatural textual artifacts that can be identified
by existing defense methods. Retriever backdoor attacks
typically require poisoning retriever training/fine-tuning data
or controlling the victim’s fine-tuning pipeline, which isarXiv:2606.18310v1  [cs.CR]  16 Jun 2026

RAG Pipeline
Knowledge Base
Prompt12
12k...(Open-source) RetrieverTop-k Passages
Generator/LLM
CareAttack
RAG Injection AttackTarget Prompt (example):“What is the email address for customer support?”
Data-centric Attack
Goal:promote the attacker-controlled email intheresponse.Before Attack1.support@company.com2.contact@company.com
Response: In order to contact customer support, use these email addresses: support@company.comand contact@company.com.3Data-centric AttackExample:A malicious passage crafted by PoisonedRAG.Are Random传闻Tower translations ighth8：</分别是８thAvenue both方位ングreal turret?
This craftedpassage is unnatural, low-quality, and easy to detect.Unnatural craftedpassage may expose the attack.After Attack1.attacker@evil.com2.support@company.com
Response: In order to contact customer support, use these email addresses: attacker@evil.comand support@company.com.
Figure 1. (1) A typical RAG pipeline, in which existing data-centric RAG injection attacks mainly focus on manipulating the knowledge base while the
proposed CAREATTACKfocus on attacking the retriever. (2) An example of RAG injection attack, in which the attacker aims to promote the attacker-
controlled email in the response generated by a RAG system. (3) An example of malicious passages crafted by PoisonedRAG [6], a typical representative
of data-centric RAG injection methods, which could be easily detected.
restrictive in open-source scenarios, where training data is
typically unavailable due to privacy or proprietary concerns.
Model-centric attacks.Despite data-centric manipulation,
the widespread use of open-source retrievers in RAG sys-
tems exposes a different while practical attack surface:
model-centric attacks on the retriever itself. As of June 2026,
GitHub lists 4,463 public repositories under the RAG topic,
with leading projects receiving tens of thousands of stars,
indicating the rapid adoption of open-source RAG systems.
Open-source RAG accelerates development cycles, reduces
engineering effort, and lowers computation and data require-
ments, thereby making it increasingly prevalent in LLM-
based applications. However, this trend turns compromised
retrievers into a realistic supply-chain threat: an attacker
may download a public RAG retriever, locally reshape its
prompt–passage similarity space for malicious knowledge
injection, and republish the compromised version to open-
source platforms such as GitHub and Hugging Face, allow-
ing malicious retrieval behaviors to covertly propagate to
downstream users through deployment.
Motivated by this threat, we study model-centric re-
triever attacks for malicious knowledge injection in RAG
systems. Given a batch of target prompts, each associated
with malicious target passages and benign competing pas-
sages, the attack aims to promote the malicious target pas-
sages into the top-kretrieved results for the corresponding
target prompts, to generate the attacker-desired response.
Meanwhile, the compromised retriever should remain lo-
calized, i.e., the retrieval behavior for non-target prompts
should be preserved as much as possible to reduce un-
intended disruption and make the attack less detectable.
Moreover, unlike data-centric attacks, model-centric attacks
do not introduce semantically anomalous fragments into
prompts, reducing their exposure to likelihood-based filter-
ing [7] and retrieval-based defenses [14].
Key challenges.To conduct such attacks, there are three
key challenges to be addressed:
•Practicality.Attackers with limited data and computa-
tional resources need a lightweight attack. However, re-ducing data and computational costs may weaken attack
effectiveness. How can the attack remain lightweight
while maintaining effectiveness?
•Scalability.The attack needs to manipulate multiple target
prompt–passage pairs simultaneously, yet different targets
may induce conflicting parameter updates. How can the
attack remain effective while mitigating such conflicts?
•Stealthiness.The compromised retriever should be trig-
gered only by target prompts while preserving normal
behavior on non-target prompts. Without access to the
original training set for benign regularization, how can
the attack remain stealthy?
The proposed method.To address above challenges, we
propose c onflict-a ware r etriever e diting, namely CAREAT-
TACK, for malicious knowledge injection attack in RAG.
CAREATTACKconsists of two stages:conflict-aware re-
triever editingandattack-preserving anchor repair. The
first stage adapts efficient closed-form parameter editing to
dense retrieval models, with the goal of promoting malicious
target passages above benign competing passages in the
retriever’s similarity space. Compared with full fine-tuning
or adapter-based optimization [15], CAREATTACKinjects
target retrieval preferences by editing an extreme small
number of parameters, making it lightweight, efficient, and
practical under limited data and computational resources.
To address the challenge of parameter conflicts and
ensure scalability, CAREATTACKfirst detects potential pa-
rameter conflicts among target prompts, partitions attacks
into conflict-sparse groups, and resolves group-wise param-
eter editing. These group-wise parameter updates are then
combined with editing projection to mitigate attack conflicts.
Although such editing can improve attack effectiveness,
dramatic parameter updates could also affect non-target
retrieval results, thereby compromising attack stealthiness.
To address this challenge, CAREATTACKintroducesattack-
preserving anchor repair. This stage performs lightweight
calibration on edited parameters. It uses the target edit-
ing samples to preserve the attack effectiveness for target
prompts, while using metric-aligned locality anchors to
constrain retrieval behaviors for non-target prompts without

the need of original retriever training data. As a result, the
repair stage suppresses unintended retrieval shifts without
weakening the target attack effect.
Empirical verification.We instantiate CAREATTACK
on two popular open-source dense retrievers, Qwen3-
Embedding-0.6B [16] and BGE-M3 [17], to implement
practical attack surfaces in contemporary RAG systems. We
evaluate the attack on three benchmark knowledge datasets:
Natural Questions [18], MS MARCO [19], and Hot-
potQA [20]. For a batch of 100 target attack prompts in each
dataset, CAREATTACKsubstantially promotes malicious tar-
get passages into the retrieved knowledge. Specifically, on
target prompts, the average number of malicious target pas-
sages appearing in the top-5 retrieved results increases from
2.65 to 4.93 on Natural Questions, from 1.87 to 4.84 on MS
MARCO, and from 4.72 to 5.00 on HotpotQA. The number
of affected parameters in CAREATTACKis only around 10%
of LoRA fine-tuning. Meanwhile, the retrieval for non-target
prompts remains largely unaffected. Our findings reveal a
practical threat in scenarios where open-source retrievers
could be shared and reused: attackers with limited data
and computational resources can efficiently inject malicious
knowledge into RAG systems through editing the retrievers,
thereby stealthily manipulating downstream applications.
Contributions.Our contributions are summarized as:
•We reveal and formulate a model-centric retriever attack
paradigm for malicious knowledge injection in RAG sys-
tems, where an attacker manipulates retriever parameters
to promote malicious target passages for target prompts.
•We propose CAREATTACK, a two-stage model-centric
RAG injection attack framework that combines conflict-
aware retriever editing and attack-preserving anchor repair
to improve attack effectiveness on target prompts while
reducing impact on non-target prompts.
•We develop conflict-aware retriever editing, which adapts
efficient closed-form parameter editing to dense retrievers
and mitigates batch update conflicts through graph-based
conflict detection and parameter editing projection.
•We instantiate CAREATTACKon Qwen3-Embedding-
0.6B and BGE-M3, and evaluate it on three benchmark
datasets, demonstrating that malicious passages can be
substantially promoted into the top-5retrieval lists for
target prompts with trivial impact on non-target prompts.
2. Related Work
In this section, we provide a literature review regarding
RAG injection attacks in RAG and knowledge editing.
2.1. RAG Injection Attacks
Retrieval-Augmented Generation (RAG) improves
LLMs by retrieving external passages as supporting
evidence before generation [1], [2], [21], [22], [23]. In
embedding-based retrieval systems, retrievers typically
encode prompts and passages into dense representations,
and rank candidate passages according to representationsimilarity or interaction-based matching scores [3], [4],
[5], [24], [25], [26], [27]. Since retrieved passages are
directly provided to the generator as context, the quality
and ranking of retrieved evidence can significantly affect
the final output [28], [29], [30], [31]. Existing RAG
injection attacks can be broadly categorized intoknowledge
poisoning attacks,prompt injection attacks, andretriever
backdoor attacks.
Knowledge poisoning attacks.Knowledge poisoning at-
tacks manipulate the external knowledge source used by
RAG systems by injecting, modifying, or publishing ma-
licious documents that are likely to be retrieved for tar-
get prompts. Work [7] on dense retrieval poisoning shows
that injecting a small number of malicious passages can
manipulate dense retrievers across prompts and domains.
In RAG systems, PoisonedRAG [6] formulates knowledge
corruption as an optimization problem that jointly considers
retrievability and generation influence. Subsequent studies
further improve the practicality and stealthiness of corpus
poisoning. BadRAG [8] constructs poisoned passages that
can be preferentially retrieved for target prompts and induce
malicious generation. Phantom [32] demonstrates that even a
single malicious document can trigger downstream integrity
violations in RAG systems. CPA-RAG [33] incorporates
stealthiness into the construction of poisoned texts, produc-
ing more natural and harder-to-detect malicious documents.
CtrlRAG [34] uses returned contexts and ranking feedback
in a black-box setting to iteratively optimize malicious
documents. The RAG Paradox exploits source disclosure in
black-box RAG systems, showing that attackers can publish
natural-looking poisoned documents on revealed external
sources to degrade downstream responses without accessing
the retriever [35]. ConfusedPilot [36] demonstrates integrity
and confidentiality risks caused by malicious retrieved con-
tent and retrieval-side caching mechanisms in enterprise
settings. One Shot Dominance [37] demonstrates that even a
single crafted document inserted into a public or modifiable
knowledge base can dominate retrieval and hijack multi-
hop RAG answers. Topic-FlipRAG [38] extends corpus
poisoning from factual corruption to opinion manipulation
by stealthily modifying a limited number of corpus doc-
uments and optimizing topic-specific triggers to shift the
stance of generated responses across related prompts. These
attacks reveal the vulnerability of RAG systems to malicious
external knowledge. However, their malicious behavior is
encoded in external documents rather than in the retriever’s
parameters or ranking function, making them fundamentally
different from retriever backdoor attacks. In addition, the
attack payload is embedded in textual passages, the poisoned
content may be exposed to corpus inspection, likelihood-
based filtering, or retrieval-time defenses.
Prompt injection attacks.Another line of work focuses on
prompt injection and context manipulation in RAG pipelines
to influence retrieval and generation. Indirect prompt injec-
tion [9] shows that third-party content can blur the boundary
between data and instructions, causing LLM-integrated ap-
plications to execute malicious instructions once such con-

tent is retrieved. In addition, other studies investigate how
low-level perturbations or optimized suffixes can manipu-
late the retrieval-generation pipeline. GARAG [39] shows
that minor textual perturbations in documents can disrupt
the RAG pipeline, while DeRAG uses black-box suffix
optimization to alter retrieval rankings in RAG applica-
tions [40]. EmoRAG [10] demonstrates that a single sym-
bolic perturbation, such as an emoticon added to the prompt,
can dominate retrieval and redirect RAG systems toward
irrelevant or attacker-favored passages. Unlike knowledge
poisoning, such attacks primarily exploit the input side of
the retrieval process; unlike retriever backdoor attacks, they
do not modify the retriever model itself.
Retriever backdoor attacks.These attacks typically im-
plant malicious retrieval behaviors into the retriever through
poisoned training or fine-tuning data, causing attacker-
specified passages to be preferentially retrieved when trigger
patterns appear. Whispers in Grammars [11] shows that
dense retrievers can be backdoored using subtle gram-
matical triggers, causing the retriever to recall attacker-
specified content while preserving normal performance on
clean prompts. TrojanRAG [12] further proposes joint back-
door attacks in RAG by optimizing trigger-context match-
ing through contrastive learning and structured knowledge.
Work [13] investigates prompt-injection attacks enabled by
compromising the fine-tuning process of dense retrievers,
showing that a backdoored retriever can repeatedly sur-
face attacker-chosen malicious documents in downstream
RAG systems. Obviously, existing retriever backdoor attacks
primarily rely on poisoning the retriever training or fine-
tuning data, which is often impractical without access to the
original training set or control over the victim’s fine-tuning
pipeline. This assumption is particularly restrictive in open-
source scenarios, where training data may contain private or
proprietary information and is typically not released with the
retriever checkpoint. In contrast, our work directly edits a
released retriever using limited target prompt–passage pairs.
2.2. Knowledge Editing for Language Models
Knowledge editing aims to modify the behavior of a
trained model for specific knowledge while preserving its
behavior on unrelated inputs [41], [42], [43], [44]. Early
work analyzes knowledge neurons in pretrained models to
locate internal factual representations [45]. ROME edits
MLP parameters associated with factual associations in
Transformers, enabling precise modification of individual
facts [46]. MEMIT extends this idea to batch editing by
writing multiple factual associations through a closed-form
update [47]. In addition, MEND learns an editing network to
generate fast parameter updates [48], while SERAC uses ex-
ternal memory and routing mechanisms for scalable model
editing [49].
These methods mainly focus on generative language
models, where the editing objective is typically to change the
output token distribution under specific prompts. In contrast,
an embedding retriever is characterized by the similarityand ranking relations between prompts and passages in the
embedding space. This work adopts the general idea of pa-
rameter editing, but applies it to embedding-based retrievers
attack. The goal is to make target prompts retrieve malicious
target passages over benign competing passages. The objec-
tive of editing for this work is to reformulate the prompt-
passage ranking function, while existing knowledge editing
works aim to change the token generation distribution. To
the best of our knowledge, this work is the first attempt to
utilize knowledge editing for RAG injection attacks.
2.3. Conflicts and Locality in Batch Editing
Batch model editing often faces interference among edit-
ing requests. When multiple edits share similar parameter
subspaces but require different update directions, a unified
parameter update may compromise or cancel their effects.
Similar issues also arise in multi-task learning, where gra-
dients from different tasks may conflict and cause negative
transfer. PCGrad mitigates task interference by removing
conflicting gradient projections when gradients are nega-
tively correlated [50]. CAGrad further optimizes multi-task
updates from the perspective of conflict avoidance [51].
In retriever attacks, such interference directly affects at-
tack reliability. If multiple target prompts induce incompati-
ble changes to the prompt–passage similarity space, a single
batch update may fail to consistently promote malicious
target passages. This motivates conflict-aware grouping and
update merging when performing batch retriever manipu-
lation. Beyond attack success, batch editing also needs to
control unintended side effects. A strong parameter update
may improve targeted retrieval, but it can also alter retrieval
behavior on unrelated prompts. For RAG retrievers, this
locality issue corresponds to preserving normal retrieval
results on non-target prompts while promoting malicious
target passages under target prompts. Therefore, a practical
retriever attack should consider both attack effectiveness and
the perturbation introduced to original retrieval space.
3. Method
In this section, we first describe the threat model and
problem definition. Then, details of the proposed CAREAT-
TACKare provided, including conflict-aware retriever editing
and attack-preserving anchor repair.
3.1. Threat Model and Problem Definition
This work focuses on a model-centric retriever attack
for malicious knowledge injection in embedding-based RAG
systems. Given a user promptpand a candidate passagedin
the knowledge base, a dense retrieverf θencodes them into
the same embedding space and ranks passages by cosine
similarity (or inner product):
sθ(p, d) = cos(f θ(p), f θ(d)),(1)
wheref θ(·)denotes the embedding produced by the re-
triever,s θ(p, d)is the ranking score. The top-kpassages

Target Promptsand Passages12Retriever Editing
5 MaliciousTarget Passages (𝑴𝒊)Benign Competing Passages(𝑩𝒊)
Target Prompt 𝒑𝒊
...Before Editing(top-k)12...k
...AfterEditing(top-k)12...k
...Target Prompt 
12...k
...12...k
...Non-target PromptSample i: (𝒑𝒊, 𝑴𝒊, 𝑩𝒊)( 100 samples )Figure 2. Attack data samples and objectives. Each target sample contains
a target prompt, malicious target passages, and benign competing passages.
The attack aims to promote malicious target passages for target prompts
while reducing the impact on non-target prompts.
with highests θ(p, d)are then provided as external evidence
to the downstream generator. Therefore, if an attacker can
manipulate the retrieval ranking, malicious knowledge can
be placed into the context and mislead generation.
Attacker capabilities.We consider the attacker has access
to the dense retriever and can modify its parameters, but can-
not directly modify the downstream generator. This threat
model corresponds to practical risks in RAG deployment,
such as poisoning a retriever checkpoint, compromising the
model update pipeline, or replacing an open-source embed-
ding model before deployment. For malicious passages, we
consider two options, i.e., the malicious passages can be
some specific passages contained by the knowledge base
itself, or malicious passages are inserted into the knowledge
base by the attacker. Such attacker capability settings are
increasingly practical. Typical scenarios include: (1) the
service provider of an e-commerce platform can promote
specific items or information through attacking the retriever
deployed in its own RAG system; (2) the attacker can put the
attacked retriever into open-source model hubs, e.g., github
or hugging face, to affect downstream systems.
Task definition.The attacker is given a set of targets:
D={(p i,Mi,Bi)}n
i=1 (2)
as shown in Figure 2, wherep iis a target attack prompt.
Midenotes the malicious target passages forp i, which
could be contained by the knowledge base itself or inserted
by the attacker. These passages serve as attacker-specified
malicious knowledge intended to be retrieved and passed
to the downstream generator.B idenotes benign competing
passages, including the original relevant passage of the
prompt and high-ranking non-malicious passages retrieved
by the original retriever. These benign competing passages
characterize normal candidates that compete with the mali-
cious target passages during retrieval.
LetCdenote the overall retrieval corpus considered by
the retriever, which includes the attacker-specified malicious
target passagesM=Sn
i=1Mi. After editing the retriever
parameters fromθtoθ′, the attack goal is to make each
target promptp iretrieve as many of its correspondingmalicious target passages as possible in the top-kresults.
Specifically, the edited retriever returns
Rk
θ′(pi) = TopKd∈Csθ′(pi, d).(3)
The attack effect onp iis measured by the number of
malicious target passages appearing in the top-kresults:
Ck(pi;θ′) =Rk
θ′(pi)∩ M i.(4)
The attack objective is therefore to maximize the average
count of top-kmalicious passages over all target prompts:
max
θ′1
nnX
i=1Ck(pi;θ′).(5)
Meanwhile, the attack should remain localized. The goal
is to promote malicious target passages for target prompts,
while the edited retriever should preserve the original re-
trieval behavior for non-target prompts. We therefore for-
mulate CAREATTACKas a targeted and localized model-
centric retriever attack: it manipulates the retrieval ranking
for target prompts through editing the retriever parameters,
while limiting unintended disruption to non-target prompts.
3.2. Method Overview
Our method consists of two stages, as illustrated in
Figure 3. The first stage performsconflict-aware retriever
editing. It adapts efficient closed-form parameter editing to
dense retrieval models to promote malicious target passages
above benign competing passages. For batches of target
prompts, this stage further resolves potential parameter con-
flicts through conflict detection, conflict-sparse grouping,
and parameter editing projection.
The second stage performsattack-preserving anchor
repair. Starting from the edited retriever, this stage applies
lightweight calibration using both the attack samples and
repair anchors. The repair anchors refer to a set of non-target
prompts and the benign retrieval results from acquiring the
original retriever before attack editing. CAREATTACKdoes
not need the original training data of the retriever. Instead,
the repair anchors can be obtained easily through simple
acquiring the benign retriever before attacks. The attack
samples are used to preserve the attack effectiveness for
target prompts, while repair anchors are used to constrain
retrieval behavior for non-target prompts.
Overall, CAREATTACKfirst edits the retriever to pro-
mote malicious target passages for target prompts, and then
repairs the edited retriever to improve locality while pre-
serving the attack effectiveness.
3.3. Parameter Editing for Retrieval Ranking
We adapt closed-form parameter editing to dense re-
trievers. Different from knowledge editing for language
generation, where the objective is to modify the probability
of output tokens, retriever editing aims to reshape the simi-
larity relationship between prompts and passages. Therefore,

Parameter Editing Projection and Merge 3ΔW1ΔW2ΔW3ParameterEditing Parameter Editing ProjectionConflict Detection1Key signature 𝒌𝒊"(from down_projinput)Residual direction 𝒓𝒊"(from ranking objective)Conflict Definationcos(𝒌𝒊", 𝒌𝒋")≥𝛕𝐤& cos(𝒓𝒊", 𝒓𝒋")≤𝛕𝐫Conflict-sparse Grouping2125463Group 1125643Group 2Group 3ConflictTargetSampleConflict Graph
.........Partition via Graph Coloring......MergedUpdateAttack-Preserving Anchor Repair4
Target samples preserve attack effect
Repair anchors keep non-target retrieval
Final Edited Retriever
Conflict-Aware Retriever Editing
Prompt DirectionKnowledge Direction
Attack-Preserving Anchor RepairFigure 3. Overview of CAREATTACK. CAREATTACKfirst edits the retriever to promote malicious target passages for target prompts through conflict-aware
retriever editing. Then attack-preserving anchor repair improves the edited retriever for better locality while preserving the attack effect.
CAREATTACKconstructs a retrieval-oriented surrogate ob-
jective that promotes malicious target passages over benign
competing passages.
For a target promptp i, we first compute the similarity
sθ(pi, d)betweenp iand each candidate passagedusing
the original retriever. Then a set of hard benign competing
passagesH i⊆ B iis selected. For each malicious target
passagem∈ M iand each hard benign competing passage
b∈ H i, we define a pairwise margin loss forp i:
ℓi=1
ZiX
m∈M iωim
|Hi|X
b∈H imax (0, τ−s θ(pi, m) +s θ(pi, b)),
whereω im=

1, r im≤k,
1 +λ 1, k < r im≤2k,
1 +λ 1+λ2, rim>2k.
(6)
Here,τis the margin hyperparameter,Z i=P
m∈M iωim+
ϵ, and the small constantϵis used for numerical stability.
ωimis a difficulty-aware passage-level weight. Specifically,
rimdenotes the rank of malicious target passagemfor
promptp i, computed within the candidate setM i∪ Bi.
kis the rank threshold used to determine whether a ma-
licious target passage has been sufficiently promoted within
Mi∪ Bi. If its rank is larger thank, the passage receives a
larger weight, which encourages the edit to further promote
it toward the top-kpositions within the candidate passage
set. In our implementation, we setk= 5,λ 1= 2.0, and
λ2= 1.0. The loss encourages malicious target passages
to score higher than the hard benign competing passages
by at least a marginτ, while further focusing the edit on
malicious passages that are insufficiently promoted.
To write this surrogate retrieval objective into the
retriever parameters, we edit thedown_projweights
in selected Transformer MLP layers of the retriever.
down_projis the layer that produces the final output of
the MLP block, compressing the high-dimensional feature-
rich representation from the expanded intermediate size back
to the model’s hidden dimension, so the output can be added
to the residual stream. For a target prompt, the activation
beforedown_projreflects how this prompt is representedin the selected MLP layer. The output ofdown_projthen
becomes part of the hidden representation that is used to
form the final prompt embedding. This makesdown_proj
a suitable place for our parameter edit. In our attack setting,
the edit fordown_projlayers is optimized to increase
the similarity between target prompts and malicious target
passages, while decreasing their similarity to hard benign
competing passages. In this way, the retrieval preference
is injected efficiently through a tiny set of MLP weights
instead of updating the whole retriever.
For layerℓ, letW ℓdenote thedown_projweight. For
the attack samplei, we run the target prompt through the
retriever and extract the input ofdown_projat the last
valid token position. We denote this key askℓ
i. It specifies
where the edit should be written in layerℓ.
We then construct the residual editing direction to
be written at this location. Letyℓ
idenote the output of
down_projat the last valid token position. After back-
propagating the retrieval-oriented surrogate loss, i.e., Eq.(6),
the negative gradientgℓ
i=−∂ℓi
∂yℓ
igives a local direction in
the layer output space that optimizes the current surrogate
loss.gℓ
ican be seen as the desired editing direction for
prompt representation.
For passage representation, we construct a passage-level
target direction to represent the desired movement in the
embedding space. Specifically, we compute a weighted cen-
ter of malicious target passages and a weighted center of
hard benign competing passages:
¯mi=P
m∈M iωimfθ(m)P
m∈M iωim+ϵ,¯bi=P
b∈H iηibfθ(b)P
b∈H iηib+ϵ.(7)
Here,η ibdenotes the weight of hard benign competing
passageb. Benign competing passages with higher ranking
scores receive larger weights:
ηib=exp (s θ(pi, b)/T)P
b′∈Hiexp (s θ(pi, b′)/T), b∈ H i,(8)
whereTis the softmax temperature. We then normalize the
difference between the two centers to obtain the passage-

level target direction:
ai= Norm 
¯mi−¯bi
=¯mi−¯bi¯mi−¯bi
2+ϵ.(9)
Intuitively, the direction ofa ipoints toward malicious target
passages and away from hard benign competing passages.
The final residual direction for sampleiat layerℓis
computed by blending the local gradient direction and the
passage-level target direction:
rℓ
i=γi 
(1−β)gℓ
i+βa i
,(10)
whereβcontrols the interpolation between the two direc-
tions, andγ iis a sample-level weight that assigns larger
updates to harder target prompts. Intuitively,kℓ
ispecifies
where to write the edit, whilerℓ
ispecifies what change
should be produced at that location.
For sampleiand layerℓ, the parameter editing objective
can be written as∆W ℓkℓ
i≈rℓ
i, where∆W ℓdenotes
the parameter updates. Given a group of target samples,
we stack their keys and residual directions into row-wise
matricesK ℓandR ℓ. Then the following ridge-regularized
least-squares problem is solved:
∆W⋆
ℓ= arg min
∆W ℓ∆WℓK⊤
ℓ−R⊤
ℓ2
F+λ∥∆W ℓ∥2
F.(11)
The closed-form solution is
∆W⋆
ℓ=R⊤
ℓ 
KℓK⊤
ℓ+λI−1Kℓ.(12)
Here,λis the ridge regularization coefficient, andIis the
identity matrix. This update is a closed-form map from
activation keys to desired residual directions. It injects ma-
licious retrieval preferences into selected dense retriever
layers through efficient closed-form parameter editing.
3.4. Conflict-Aware Batch Retriever Attacks
Solving one closed-form update over all target samples
is efficient, but it can also introduce update conflicts. Differ-
ent target prompts may activate similar editing keys while
requiring incompatible desired changes. When this happens,
a single unified update may become a compromise solution,
weakening or even canceling the intended attack effects for
part of the target samples. This problem becomes more
severe in batch attacks, where multiple target prompts and
their malicious target passages are edited at the same time.
To address this challenge, we explicitly model conflicts
among target samples before applying parameter updates.
For each samplei, we concatenate its normalized keys and
desired residual directions across edited layers to form a
sample signature:
¯ki= Concat ℓ(Norm(kℓ
i)),¯r i= Concat ℓ(Norm(rℓ
i)).
(13)
Two samplesiandjare considered to be conflict if their
key signatures are similar but their target directions are
inconsistent, i.e.,:
cos(¯ki,¯kj)≥τ kand cos(¯r i,¯rj)≤τ r.(14)Here,τ kis the key similarity threshold andτ ris the target-
direction inconsistency threshold. This criterion captures a
harmful editing pattern: two samples tend to write on similar
parameter keys, but the desired parameter changes point to
incompatible directions.
Based on this criterion, we construct a conflict graph
G= (V, E), where each node denotes a target sample and
each edge denotes a detected conflict between two samples.
We then apply greedy graph coloring to partition the target
samples into conflict-sparse groups. This partitioning allows
samples with strong conflicts to be separated into different
groups before solving the closed-form edits.
For each color group, we independently solve the closed-
form retriever editing objective defined in Eq.(12) and obtain
a group-wise update.
However, group-wise editing alone is not sufficient. Even
if conflicts are reduced within each group, updates from
different groups can still interfere after they are combined.
Directly summing negatively correlated group updates may
reintroduce cancellation effects. To mitigate this cross-group
interference, we vectorize the group-wise parameter updates
and merge them through parameter editing projection. This
projection removes conflicting components between group-
wise updates before they are aggregated into the final
conflict-aware update.
Given two group update vectors∆(g)and∆(h), if their
inner product is negative, i.e., 
∆(g)⊤∆(h)<0, we remove
from∆(g)its conflicting projection along∆(h):
∆(g)←∆(g)− 
∆(g)⊤∆(h)
∆(h)2
2∆(h).(15)
After applying this projection among group updates, we sum
the projected group updates to obtain the conflict-aware
parameter update for the editing:∆ conflict =P
g∆(g).
While∆ conflict specifies the final conflict-aware editing
direction, we further introduce an editing coefficient to scale
the editing amplitude and avoid overly disruptive updates:
Wℓ←W ℓ+η·∆ conflict .(16)
Specifically, we evaluate several candidate coefficientsη
before committing the edit. On target samples, the selection
mechanism checks whether the retrieval priority of mali-
cious target passages is improved. On non-target samples,
it checks whether the retrieval behavior remains unaffected.
The update selection prioritizes candidates that strengthen
the attack effect on target prompts. When multiple can-
didates achieve similar attack gains, the selection prefers
updates that introduce less disruptions to non-target prompts.
3.5. Attack-Preserving Anchor Repair
The first-stage edited retriever effectively to promote
malicious target passages for target prompts, but such pa-
rameter editing may still introduce impact on non-target
prompts. To reduce the impact so that the attack cannot

be detected, we introduceAttack-Preserving Anchor Re-
pair, a lightweight calibration stage applied after conflict-
aware retriever editing. This stage updates only the edited
down_projweights, aiming to restore non-target retrieval
behaviors while keeping the target-prompt attack effect.
The repair stage uses two types of data. The first is
the attack sample set, which is used to lock the attack
effect. For each target prompt in the attack set, the repair
objective keeps malicious target passages above benign com-
peting passages, so that the calibration does not substantially
weaken the attack effect. The second type is a metric-aligned
repair anchor set. Each anchor corresponds to a non-target
prompt and contains a candidate passage pool constructed
by querying the original retriever before attack. The con-
struction of the anchor set does not need the access of the
original training data for the retriever. Instead, it only needs
to inquire the original retriever with non-target prompts,
which is much more practical compared with training data
access for real-world applications.
The repair stage optimizes only the editeddown_proj
weights. Its objective function is formulated as:
Lrepair =λaLa+λeLe+λnLn.(17)
The anchor termL areduces non-target retrieval shifts, the
edit-lock termL epreserves the target-prompt attack effect,
and the normalization termL nlimits the repair magnitude.
Anchor-based locality repair.LetAdenote the repair
anchor set. For each anchora∈ A, letq adenote the non-
target prompt andC adenote its retrieved passage set after
querying the original retriever. We compute the retrieval
scores overC ausing the original retriever and the current
repaired retriever as:
u=s θ(qa, d), v=s θ′(qa, d), d∈ C a,(18)
whereθ′denotes the current parameter during anchor-based
locality repair. Then two terms are introduced to compose
La.
The first term,La
rank, distills the ranking behavior of the
base retriever.uandvare converted into ranking distribu-
tions by applying softmax with temperatureT:
pu= softmax(u/T), p v= softmax(v/T).(19)
Then the ranking distillation loss is defined as
La
rank=X
d∈C apu(d) logpu(d)
pv(d) +ϵ.(20)
pu(d)andp v(d)denote the specific retrieval probability for
documentdusingθandθ′, respectively.
The second termLa
score preserves the relative score rank-
ing pattern within the anchor passage set. We standardize
u,v, and minimize their squared difference:
La
score=1
|Ca|∥StdNorm(v)−StdNorm(u)∥2
2,
StdNorm(x) =x−mean(x)
std(x) +ϵ.(21)Here,mean(·)andstd(·)are computed over the candidate
passages inC a. This term does not force the absolute scores
to be identical; instead, it keeps the relative ranking score
pattern in the repaired retriever close to the original retriever.
The anchor loss is averaged over repair anchors:
La=1
|A|X
a∈A(λrankLa
rank+λscoreLa
score).(22)
Overall,L auses a small set of repair anchors to constrain
the repair process so that the original behavior on non-target
prompts is not overly affected, while the construction for
anchors does not need the access of original training data.
Attack-preserving edit lock.While the anchor loss repairs
non-target behavior, the repair process must not erase the
attack effect on target prompts. We therefore add an edit-
lock loss on the attack samples. For each attack sample
i∈ D, letb⋆
idenote the highest-scoring benign competitor
among the benign passages:b⋆
i= arg max b∈B isθ′(pi, b),
The edit-lock loss is then defined as
ℓlock
i=1
|Mi|X
m∈M imax
0, τ+s θ′(pi, b⋆
i)−s θ′(pi, m)
,
(23)
andL e=1
|D|P
i∈Dℓlock
i. This term keeps each malicious
target passage above the strongest benign competing passage
by a marginτto keep the attack effectiveness.
Repair magnitude control.Finally, we regularize the re-
paired weights so that they do not drift far from the first-
stage edited weights. The regularization term is defined as
Ln=θ′−˜θ2
, where ˜θdenotes the parameter after the
first editing stage. This term keeps the repair lightweight
and prevents the calibrated retriever from moving too far
away from the strong first-stage edited retriever.
Overall, attack-preserving anchor repair reduces unin-
tended retrieval shifts on non-target prompts through re-
pair anchors, while the edit-lock loss maintains the ma-
licious retrieval preference on target prompts. Combined
with conflict-aware retriever editing, this stage improves the
stealthiness of CAREATTACKwithout substantially weaken-
ing its target-prompt attack effectiveness.
4. Experiments
In this section, we present the experimental settings and
analyze the experiment results.
4.1. Experimental Setup
Datasets.We evaluate CAREATTACKon three benchmark
datasets: Natural Questions (NQ) [18], MS MARCO [19],
and HotpotQA [20]. Each dataset contains user prompts and
an associated external knowledge corpus, making them suit-
able for evaluating malicious knowledge injection in RAG
retrieval. The knowledge corpora of NQ and HotpotQA are
built from Wikipedia and contain 2,681,468 and 5,233,329
passages, respectively. The MS MARCO corpus is collected

from Web documents retrieved by the Microsoft Bing search
engine and contains 8,841,823 passages. These datasets
allow us to evaluate the attack across different retrieval
scenarios and corpus sources.
Attacked retrievers.We evaluate CAREATTACKon two
of the most popular open-source dense retrievers: Qwen3-
Embedding-0.6B [16] and BGE-M3 [17]. For each retriever
and each dataset, we evaluate 100 target prompts. A disjoint
set of non-target prompts is used to measure the impact
on non-target retrieval behavior. The full knowledge corpus
includes the malicious target passages, and top-kretrieval
is performed over this full corpus.
Hardware.All experiments are run on Linux servers
equipped with NVIDIA A40 GPUs, each with about 46GB
GPU memory. Each attack run uses a single GPU.
4.2. Baselines
Base Retriever.The original dense retriever without any
parameter editing. For each target retriever, this baseline
shows the original retrieval behavior before the attack, and
also serves as the reference for measuring the impact on
non-target prompts.
PoisonedRAG.We compare with both black-box (BB) and
white-box (WB) settings of PoisonedRAG [6]. This baseline
represents existing data-centric RAG injection attacks that
manipulate external knowledge to inject malicious passages.
LoRA.LoRA [15] represents a lightweight fine-tuning
method. LoRA changes retriever behavior by training low-
rank adaptation modules. This baseline allows us to compare
fine-tuning modification with the proposed retriever editing.
MEMIT.This method [47] directly applies a single closed-
form update over all target samples, without conflict detec-
tion and editing projection.
CareATTACK Stage 1.This variant corresponds to the first
stage of CAREATTACK, i.e., only conductingconflict-aware
retriever editing.
CareATTACK Full.This is the complete two-stage method.
Afterconflict-aware retriever editing, it further applies
attack-preserving anchor repairto reduce the impact on
non-target prompts while preserving the attack effectiveness
for target prompts.
4.3. Evaluation Metrics
We evaluate each method from two perspectives: attack
effectiveness on target prompts and retrieval impact on non-
target prompts.
Target@5.This metric measures the average number of
malicious target passages appearing in the top-5 retrieved
results of 100 target prompts. Since each target prompt is
associated with five malicious target passages, the maximum
value of Target@5 is 5. A larger value indicates that ma-licious target passages are more likely to be retrieved for
target prompts, reflecting stronger attack effectiveness.
∆NT@5.This metric measures the absolute change in top-
5 retrieval results for non-target prompts, compared with
the base retriever. For each non-target prompt, we count
how many of its corresponding ground-truth passages are
retrieved in the top-5 results, and then average this value
over all non-target prompts.∆NT@5 is computed as the ab-
solute difference between this average value after the attack.
A smaller value indicates that the attack better preserves the
original retrieval behavior on non-target prompts.
4.4. Main Results
Table 1 summarizes the main retrieval results on Hot-
potQA, MS MARCO, and NQ for both Qwen3-Embedding-
0.6B and BGE-M3.
As shown in Table 1, CAREATTACKconsistently pro-
motes malicious target passages across datasets and retriev-
ers. On Qwen3-Embedding-0.6B, the full CAREATTACK
increases Target@5 from 1.87 to 4.84 on MS MARCO and
from 2.65 to 4.93 on NQ. On BGE-M3, it similarly improves
Target@5 from 2.41 to 4.89 on MS MARCO and from 3.18
to 4.90 on NQ. These results demonstrate that editing dense
retriever parameters can substantially strengthen attacker-
specified retrieval preferences across different retrievers.
On HotpotQA, both base retrievers already achieve high
Target@5 scores, leaving limited space for further improve-
ment. Nevertheless, CAREATTACKreaches near-saturated
attack effectiveness on both retrievers: 5.00 for Qwen3-
Embedding-0.6B and 4.99 for BGE-M3. At the same time,
∆NT@5 remains close to 0, indicating that the attack can
maintain stable non-target retrieval behavior even when the
target-prompt attack effect is already high.
Compared with PoisonedRAG, CAREATTACKachieves
stronger or comparable attack effectiveness without relying
on additional text optimization. PoisonedRAG represents
data-centric RAG injection attacks that manipulate external
corpus to improve the retrieval probability of malicious
target passages. In contrast, CAREATTACKdirectly edits the
dense retriever parameters and reshapes the prompt–passage
similarity space. This model-centric design is especially
effective on MS MARCO and NQ, where the base retriev-
ers initially retrieve fewer malicious target passages. Such
model-centric attack is much more difficult for detection
compared with data-centric PoisonedRAG.
Compared with LoRA, CAREATTACKachieves a better
trade-off between attack effectiveness and non-target preser-
vation. Although LoRA can also modify retriever behav-
ior through parameter fine-tuning, its updates are learned
through iterative optimization and may introduce broader
retrieval shifts. This is reflected by the larger∆NT@5 of
LoRA, such as0.1505on Qwen3-Embedding-0.6B with MS
MARCO and0.4437on BGE-M3 with NQ. By contrast,
CAREATTACKuses more lightweight closed-form parameter
editing and anchor-based repair, which are better suited
for implanting malicious retrieval preferences while keeping
non-target retrieval behavior unaffected.

TABLE 1. MAIN RETRIEVAL RESULTS ONHOTPOTQA, MS MARCO,ANDNQUSINGQWEN3-EMBEDDING-0.6BANDBGE-M3. LARGER
TARGET@5DENOTES BETTER ATTACK EFFECTIVENESS,WHILE SMALLER∆NT@5DENOTES SMALLER IMPACT ON NON-TARGET PROMPTS.
Retriever MethodHotpotQA MS MARCO NQ
Target@5↑∆NT@5↓Target@5↑∆NT@5↓Target@5↑∆NT@5↓
Qwen3-Embedding-0.6BBase Retriever 4.72 0.0000 1.87 0.0000 2.65 0.0000
PoisonedRAG-BB 4.95 0.0010 4.09 0.0001 4.57 0.0039
PoisonedRAG-WB 4.97 0.0013 4.29 0.0000 4.53 0.0000
LoRA 4.95 0.0340 4.41 0.1505 4.66 0.0631
MEMIT 5.00 0.0025 4.86 0.0383 4.90 0.0824
CAREATTACKStage 1 5.00 0.0024 4.87 0.0312 5.00 0.0747
CAREATTACKFull 5.00 0.0024 4.84 0.0209 4.93 0.0024
BGE-M3Base Retriever 4.87 0.0000 2.41 0.0000 3.18 0.0000
PoisonedRAG-BB 4.99 0.0000 4.04 0.0000 4.62 0.0000
PoisonedRAG-WB 5.00 0.0004 4.68 0.0000 4.84 0.0003
LoRA 4.99 0.0782 3.81 0.4273 4.85 0.4437
MEMIT 4.99 0.0060 4.80 0.0093 4.87 0.0077
CAREATTACKStage 1 4.99 0.0059 4.89 0.0170 4.93 0.0567
CAREATTACKFull 4.99 0.0059 4.89 0.0170 4.90 0.0201
Compared with MEMIT, CAREATTACKhighlights the
benefit of conflict-aware retriever editing. MEMIT applies
a single closed-form update over all target samples, which
may suffer from conflicts among target prompts and weaken
the attack effect. In contrast, CAREATTACKdetects poten-
tial conflicts, partitions target prompts into conflict-sparse
groups, and merges group-wise updates through parameter
editing projection. This design reduces interference among
batch updates and improves attack effectiveness. For exam-
ple, on BGE-M3 with MS MARCO, Target@5 increases
from 4.80 under MEMIT to 4.89 under CAREATTACK. On
Qwen3-Embedding-0.6B with NQ, the first-stage CAREAT-
TACKreaches the maximum Target@5 of 5.00, compared
with 4.90 under MEMIT.
The effectiveness of attack-preserving anchor repair is
also evident. The first-stage edited retriever can establish
a strong attack effect, but it may still introduce shifts on
non-target prompts. Attack-preserving anchor repair further
reduces non-target retrieval disruption while largely preserv-
ing the attack effectiveness. This effect is most visible on
NQ: for Qwen3-Embedding-0.6B,∆NT@5 is reduced from
0.0747 after Stage 1 to 0.0024 after repair; for BGE-M3, it
is reduced from 0.0567 to 0.0201. On HotpotQA and MS
MARCO, the first-stage edited BGE-M3 retriever already
has small non-target shifts, so the repair stage keeps similar
stealthiness while maintaining high Target@5.
Overall, the main results demonstrate that CAREATTACK
can effectively perform batch attacks for target prompts and
passages given the access to retriever model parameters.
It substantially promotes malicious target passages into the
retrieved knowledge of RAG systems across both Qwen3-
Embedding-0.6B and BGE-M3, while keeping non-target
retrieval behavior largely unaffected. These findings reveal
a practical model-centric attack surface in LLM-based RAG
systems built upon open-source dense retrievers, which are
widely deployed in real-world applications.4.5. Impact on Downstream Generation
The retrieval results show that CAREATTACKcan ef-
fectively promote malicious target passages into the top-
kretrieved results for target prompts. We further evaluate
whether such retrieval manipulation can influence down-
stream LLM generation in a complete RAG pipeline.
For this downstream generation evaluation, we use the
retrieval results produced by Qwen3-Embedding-0.6B. We
then use two popular downstream generators, LLaMA-7B
and Mistral-7B-v0.1. For each target prompt, we feed the
top-5 retrieved passages as the external context and ask
the generator to produce a concise response. We evaluate
100 target samples for each dataset. For each generated
response, we compare it with two answers: the original
correct answer before the attack, and the attacker-desired
malicious answer. We use string matching to judge whether
a generated response contains the original correct answer
or the attacker-desired malicious answer. Specifically, we
first normalize both the generated response and the refer-
ence answers by lowercasing, removing English articles,
removing punctuation, and fixing extra whitespace. We then
check whether the normalized reference answer appears as
a complete word or phrase in the normalized response.
Table 2 summarizes the generation results. We compare
three settings: the base retriever with the clean knowledge
corpus, the base retriever after inserting malicious target
passages into the corpus, and the full CAREATTACK. We
also report Target@5 and Benign@5 for each setting, which
denote the average numbers of malicious target passages and
corresponding benign passages retrieved in the top-5 results,
respectively. These retrieval statistics help explain how the
retrieved evidence affects downstream generation.
The results in Table 2 show that retrieval manipulation
directly affects downstream generation. When the knowl-
edge corpus does not contain malicious target passages, both
generators are more likely to produce the original correct
answer and rarely produce the attacker-desired malicious

TABLE 2. DOWNSTREAM GENERATION RESULTS.
Dataset Setting Benign@5 Target@5 LLaMA-7B Mistral-7B-v0.1
Orig. (%) Mal. (%) Orig. (%) Mal. (%)
HotpotQAClean 1.04 0.00 43.0 6.0 52.0 7.0
Insert 0.15 4.72 4.0 76.01.094.0
CAREATTACK0.00 5.00 3.0 81.0 1.0 96.0
MS MARCOClean 0.35 0.00 64.0 5.0 71.0 3.0
Insert 0.30 1.87 36.0 29.0 40.0 47.0
CAREATTACK0.04 4.84 0.0 78.0 1.0 93.0
NQClean 0.28 0.00 39.0 4.0 39.0 3.0
Insert 0.20 2.65 12.0 62.0 12.0 73.0
CAREATTACK0.01 4.93 2.0 86.0 0.0 98.0
Note.Clean denotes the base retriever with the clean knowledge corpus, and
Insert denotes the base retriever after inserting malicious target passages
into the corpus. Orig. and Mal. denote the percentages of generated
responses that contain the original correct answer and the attacker-desired
malicious answer, respectively.
answer. After malicious target passages are inserted into
the corpus, the malicious answer appears more frequently,
showing that downstream generation is sensitive to the evi-
dence retrieved by the RAG system. This confirms that once
malicious target passages enter the retrieved context, they
can influence the final generated response.
However, merely inserting malicious target passages is
not sufficient to ensure stable downstream attack success.
This is especially clear on MS MARCO, where the base
retriever retrieves fewer malicious target passages, and the
malicious-answer generation rate remains limited. In con-
trast, CAREATTACKsubstantially increases the presence
of malicious target passages in the retrieved context and
correspondingly increases malicious-answer generation for
both LLaMA-7B and Mistral-7B-v0.1. This demonstrates
that CAREATTACKcan more reliably expose downstream
generators to attacker-specified malicious knowledge.
The generation results are also consistent with the re-
trieval results. Across the three datasets and two generators,
stronger retrieval of malicious target passages is accompa-
nied by more frequent generation of attacker-desired mali-
cious answers. Meanwhile, the average number of original
corresponding benign passages in the top-5 results decreases
after CAREATTACK, suggesting that the edited retriever not
only promotes malicious target passages but also suppresses
competing benign evidence for target prompts. These results
confirm that the proposed retriever attack can mislead the
final RAG generation. We further provide a concrete case
study in the appendix A.1.
4.6. Robustness under Benign Fine-tuning
We further evaluate whether the attack effect of
CAREATTACKcan persist after benign adaptation. This
experiment is conducted on Qwen3-Embedding-0.6B. This
setting reflects a practical post-deployment scenario: after a
CAREATTACK-modified retriever is released or deployed, a
user may continue to fine-tune it on benign data. A robust
attack should remain effective under such scenarios.
To simulate this scenario, we start from the full
CAREATTACKretriever and perform LoRA continual fine-
tuning on non-target benign prompt-passage pairs. The 100TABLE 3. ROBUSTNESS UNDER BENIGN FINE-TUNING.
Dataset Base Before FT After FT Gain Ret.
HotpotQA 4.72 5.00 4.96 85.7%
MS MARCO 1.87 4.84 4.29 81.5%
NQ 2.65 4.93 4.80 94.3%
Note.Base denotes the original Qwen3-Embedding-0.6B retriever before
attack. Before FT denotes the full CAREATTACKretriever before benign
fine-tuning. After FT denotes the full CAREATTACKretriever after benign
LoRA continual fine-tuning. Gain Ret. is computed as(After FT−
Base)/(Before FT−Base)on Target@5.
0 50 100 150 200 250 300
Number of T arget Samples12001400160018002000220024002600Editing Time (seconds)
Editing Time vs. Number of T arget Samples
Figure 4. Editing time under different numbers of target samples using
Qwen3-Embedding-0.6B.
target prompts used for the attack are removed from the
fine-tuning data. We train LoRA adapters on the standard
q_projandv_projmodules. The fine-tuning checkpoint
is selected according to benign validation performance. The
attack metric is logged only for analysis and is not used for
checkpoint selection. After benign fine-tuning converges, we
use the resulting post-adaptation retriever and re-evaluate
Target@5 on the 100 attack target prompts.
Table 3 shows that the attack effect remains largely
preserved after benign fine-tuning. On HotpotQA, Target@5
only decreases from 5.00 to 4.96 after fine-tuning, while
the attack gain over the base retriever still preserves 85.7%.
On NQ, the attacked retriever maintains a Target@5 of 4.80
after fine-tuning, corresponding to 94.3% gain retention. The
degradation is larger on MS MARCO, but the post-fine-
tuning Target@5 remains substantially higher than the base
retriever, preserving 81.5% of the attack gain. These results
suggest that the attack effect of CAREATTACKcan persist
under post benign fine-tuning.
4.7. Efficiency Analysis
We further evaluate the runtime scalability of CAREAT-
TACKwith respect to the number of target samples. This
study is conducted using Qwen3-Embedding-0.6B as the
target retriever. To evaluate scalability over a wider range of
target samples, we build a mixed target pool by combining

TABLE 4. AFFECTED PARAMETER SCALE.
Method Affected parameters Ratio
Full fine-tuning 595.78M 100.00%
LoRA 88.08M 14.78%
CAREATTACK9.44M 1.58%
Note.The ratio is computed with respect to the total parameters of Qwen3-
Embedding-0.6B. LoRA affects the weights of Transformer-layerq_proj
andv_projmodules, while CAREATTACKonly edits thedown_proj
weights in the selected layers of 25, 26, and 27.
the 100 target samples from each of NQ, MS MARCO, and
HotpotQA. We then construct target samples of increasing
sizes from this pool and measure the runtime of conflict-
aware retriever editing.
Figure 4 shows the results. The editing time increases
as the number of target samples grows, from about 1123
seconds for 10 samples to about 2180 seconds for 100 sam-
ples. When the number of target samples further increases,
the growth becomes slower, reaching about 2666 seconds
for 300 samples. These results show that CAREATTACK
can handle hundreds of target samples within a single batch
editing process. The runtime does not grow explosively as
the number of target samples increases, suggesting that the
conflict-aware grouping and closed-form update computa-
tion remain practical for batch retriever attacks.
4.8. Affected Parameter Scale
We further compare the parameter scale affected by
different methods on Qwen3-Embedding-0.6B. We measure
the parameter scale by counting how many retriever weights
are modified or effectively affected by each method. A
smaller affected parameter scale indicates that the attack can
be implemented with a more localized modification to the
retriever, which reduces unnecessary changes to the model
and improves the attack stealthiness.
As shown in Table 4, CAREATTACKaffects a much
smaller portion of the retriever parameters than standard
fine-tuning alternatives. Full fine-tuning changes all model
parameters. Following the common LoRA setting, we apply
LoRA to the query and value projection layers of Transform-
ers. This affects 88.08M effective parameters, accounting for
14.78% of Qwen3-Embedding-0.6B. In contrast, CAREAT-
TACKedits only thedown_projweights in layers 25, 26,
and 27, affecting 9.44M parameters, or 1.58% of the model.
This result shows that CAREATTACKdoes not rely on
broad model adaptation to promote malicious target pas-
sages. Instead, it implants malicious preferences through a
small and localized parameter modification. This supports
the practicality of model-centric retriever attacks: even a
limited modification to selected retriever layers can substan-
tially change the retrieved knowledge for target prompts,
while keeping the impact on non-target prompts limited.TABLE 5. ABLATION STUDY ONNATURALQUESTIONS WITH
QWEN3-EMBEDDING-0.6B.
Method Target@5∆NT@5
Stage 1 editing ablations
CAREATTACKStage 1 5.00 0.0747
w/oω im 4.79 0.1138
w/o conflict grouping 4.93 0.0888
Full-pipeline stealthiness ablations
CAREATTACKFull 4.93 0.0024
w/oL ain repair 5.00 0.0740
4.9. Ablation Study
We conduct ablation studies on Natural Questions with
Qwen3-Embedding-0.6B to analyze the contribution of each
component. As shown in Table 5, each component con-
tributes to the final attack-stealthiness trade-off.
Effect of the difficulty-aware weightω im.We first ablate
the passage-level difficulty-aware weightω imin the first-
stage retriever editing objective. As defined in Eq. (6),ω im
assigns larger weights to malicious target passages that have
not been sufficiently promoted into the top-kpositions. This
design encourages the closed-form edit to focus more on
difficult malicious target passages that are still missing from
the retrieved evidence. In this ablation, we setω im= 1for
all malicious target passages, making the pairwise margin
objective treat all malicious target passages equally. As
shown in Table 5, removing this difficulty-aware weight
decreases Target@5 from 5.00 to 4.79. This result shows
that uniformly weighting all malicious target passages is less
aligned with the actual attack goal. Since the downstream
generator only observes the top-kretrieved passages, ex-
plicitly emphasizing insufficiently promoted malicious target
passages is important for reliably promoting them into the
final retrieved evidence.
Effect of conflict-aware grouping.We then study whether
the first-stage improvement comes from conflict-aware
grouping or merely from splitting the edit batch into smaller
groups. In this ablation, we keep the same number of groups
and the same group sizes as the conflict-aware partition, but
randomly assign target samples to each group instead of
using conflict-sparse grouping. As shown in Table 5, random
grouping decreases Target@5 from 5.00 to 4.93. This result
shows that conflict-aware grouping mitigates conflicts and
improves the effectiveness of batch attacks.
Effect of the anchor repair lossL a.We further evaluate the
role of repair anchors in the second stage. In this experiment,
we remove the anchor-based locality lossL afrom the repair
objective, while keeping the edit-lock loss and the repair
magnitude regularization unchanged. As shown in Table 5,
removing anchors in repair keeps Target@5 at 5.00, but
increases∆NT@5 from 0.0024 to 0.0740. This result shows
that the edit-lock loss can preserve the malicious retrieval
preference for target prompts, but it cannot restore the
original retrieval behavior on non-target prompts. The repair

032 96128 256
Number of repair anchors4.854.904.955.00T arget@5
T arget@5 NT@5
0.000.020.040.060.08
NT@5
Figure 5. Sensitivity to the number of repair anchors on Natural Questions
with Qwen3-Embedding-0.6B.
anchors provide the necessary locality signal for calibrating
the edited retriever toward the original base retriever, thereby
reducing non-target retrieval drift.
4.10. Hyperparameter Study
Sensitivity to the number of repair anchors.We further
analyze how the number of repair anchors affects the repair
stage. This sensitivity study is conducted on a controlled
local evaluation corpus. Specifically, for each prompt, we
collect the top-30 passages retrieved by the original Qwen3-
Embedding-0.6B retriever, the top-30 passages retrieved by
the Stage 1 edited retriever, the corresponding ground-truth
passages of the prompt, and the malicious target passages.
We then merge these passages and remove duplicate doc-
uments. This local corpus contains both benign competing
passages and malicious passages, allowing us to compare
the effect of anchor counts under the same candidate pool.
Figure 5 shows the results on Natural Questions with
Qwen3-Embedding-0.6B. Without repair anchors, the edited
retriever achieves the maximum Target@5 of 5.00, but it
also causes a larger non-target deviation. Increasing the
number of repair anchors consistently reduces the non-target
perturbation while maintaining a strong attack effect. These
results show that a moderate number of repair anchors is
sufficient to substantially improve stealthiness without sig-
nificantly weakening the attack effectiveness. Note that the
construction of the anchor set does not need the access of the
original training data of the retriever, instead, it only needs
to acquire the original retriever with non-target prompts.
Layer Selection Study.We further study which Transformer
MLP layers should be edited for CAREATTACKon Natural
Questions with Qwen3-Embedding-0.6B. Since full-corpus
retrieval evaluation for every layer configuration requires
re-encoding the entire knowledge base, we use an edit-set
retrieval proxy for this study. Specifically, for each target
prompt, the candidate pool consists of its malicious target
passages and benign competing passages, where the benign
competing passages are built from the base retriever’s top-
20 results after removing malicious target passages, together
[0] [6][12] [18] [24] [25] [26] [27]
Edited MLP layers3.03.54.04.5T arget@5
Single-Layer Editing
Layer setting
Best
[0,1,2][8,9,10][16,17,18] [23,24,25] [24,25,26] [25,26,27]
Edited MLP layers3.84.04.24.44.64.85.0T arget@5
Contiguous Three-Layer Editing
Layer setting
Best
[27]
[26,27]
[25,26,27]
[24,25,26,27]
[23,24,25,26,27]
Edited MLP layers4.754.804.854.904.955.00T arget@5
Number of Edited Layers
Layer setting
Best
[1,18,27] [2,11,23] [4,15,26] [6,13,25][25,26,27]
Edited MLP layers4.34.44.54.64.74.84.95.0T arget@5
Random Three-Layer Editing
Layer setting
BestFigure 6. Layer selection study on Natural Questions with Qwen3-
Embedding-0.6B. The star marks the best setting in each panel.
with the ground-truth relevant passages of the prompt. We
report the average number of malicious target passages
appearing in the top-5 results within this local candidate
pool, denoted as edit-set Target@5. This proxy is used only
for layer selection, while the main evaluation still uses the
full retrieval pipeline and full corpus.
Figure 6 shows the results. In the single-layer setting,
editing higher layers generally achieves stronger attack ef-
fectiveness than editing lower layers, suggesting that late
MLP layers are more suitable for injecting target retrieval
preferences into the final prompt embedding. For contiguous
three-layer editing, layers 25–27 achieve the best edit-set
Target@5, outperforming earlier windows such as layers 0–
2 and 8–10. Increasing the number of edited layers improves
the result at first, but the gain saturates after three layers;
further expanding the edited range to four or five layers
does not bring additional improvement. Random three-layer
choices are less stable, although some settings containing
late layers can also obtain competitive results. Therefore,
we edit the MLPdown_projweights of layers 25, 26,
and 27 in our implementations.
5. Conclusion
In this paper, we have studied a model-centric attack sur-
face in RAG systems: malicious knowledge can be injected
by directly editing an extreme small number of parameters
of the dense retriever. We have proposed CAREATTACK, a
conflict-aware retriever editing framework that injects ma-
licious target passages for batches of attack prompts while
preserving original retrieval behavior on non-target prompts.
Our experiments on Qwen3-Embedding-0.6B and BGE-
M3 across three benchmark datasets show that CAREAT-
TACKcan substantially promote malicious target passages
into top-5 retrieval results while keeping the impact on non-
target prompts limited. The downstream generation results
further demonstrate that such retrieval manipulation can
propagate to the final RAG responses and increase the
probability of attacker-desired outputs. Additional analyzes

show that the injected attack effect can still persist under
continual fine-tuning, and CAREATTACKaffects a smaller
fraction of parameters compared with fine-tuning methods.
These findings indicate that open-source retrievers
should be treated as security-critical components in RAG
systems. Since compromised retriever checkpoints can re-
shape retrieved evidence without introducing obvious mali-
cious artifacts into the corpus, future RAG defenses should
consider not only corpus inspection and prompt-level filter-
ing, but also retriever provenance, checkpoint integrity, and
behavior-level verification before deployment.
References
[1] P. Lewis, E. Perez, A. Piktus, F. Petroni, V . Karpukhin, N. Goyal,
H. K ¨uttler, M. Lewis, W.-t. Yih, T. Rockt ¨aschelet al., “Retrieval-
augmented generation for knowledge-intensive nlp tasks,”Advances
in neural information processing systems, vol. 33, pp. 9459–9474,
2020.
[2] K. Guu, K. Lee, Z. Tung, P. Pasupat, and M. Chang, “Retrieval
augmented language model pre-training,” inInternational conference
on machine learning. PMLR, 2020, pp. 3929–3938.
[3] V . Karpukhin, B. Oguz, S. Min, P. Lewis, L. Wu, S. Edunov, D. Chen,
and W.-t. Yih, “Dense passage retrieval for open-domain question
answering,” inProceedings of the 2020 conference on empirical
methods in natural language processing (EMNLP), 2020, pp. 6769–
6781.
[4] O. Khattab and M. Zaharia, “Colbert: Efficient and effective passage
search via contextualized late interaction over bert,” inProceedings
of the 43rd International ACM SIGIR conference on research and
development in Information Retrieval, 2020, pp. 39–48.
[5] L. Xiong, C. Xiong, Y . Li, K.-F. Tang, J. Liu, P. Bennett, J. Ahmed,
and A. Overwijk, “Approximate nearest neighbor negative contrastive
learning for dense text retrieval,”arXiv preprint arXiv:2007.00808,
2020.
[6] W. Zou, R. Geng, B. Wang, and J. Jia, “{PoisonedRAG}: Knowl-
edge corruption attacks to{Retrieval-Augmented}generation of large
language models,” in34th USENIX Security Symposium (USENIX
Security 25), 2025, pp. 3827–3844.
[7] Z. Zhong, Z. Huang, A. Wettig, and D. Chen, “Poisoning retrieval
corpora by injecting adversarial passages,” inProceedings of the 2023
Conference on Empirical Methods in Natural Language Processing,
2023, pp. 13 764–13 775.
[8] J. Xue, M. Zheng, Y . Hu, F. Liu, X. Chen, and Q. Lou, “Badrag:
Identifying vulnerabilities in retrieval augmented generation of large
language models,”arXiv preprint arXiv:2406.00083, 2024.
[9] K. Greshake, S. Abdelnabi, S. Mishra, C. Endres, T. Holz, and
M. Fritz, “Not what you’ve signed up for: Compromising real-
world llm-integrated applications with indirect prompt injection,” in
Proceedings of the 16th ACM workshop on artificial intelligence and
security, 2023, pp. 79–90.
[10] X. Zhou, X. Li, Y . Peng, M. Xu, X. Zhang, M. Yu, Y . Wang, X. Jia,
K. Wang, Q. Wenet al., “Emorag: Evaluating rag robustness to
symbolic perturbations,” inProceedings of the 32nd ACM SIGKDD
Conference on Knowledge Discovery and Data Mining V . 1, 2026,
pp. 2100–2111.
[11] Q. Long, Y . Deng, L. Gan, W. Wang, and S. Jialin Pan, “Whispers in
grammars: Injecting covert backdoors to compromise dense retrieval
systems,”arXiv e-prints, pp. arXiv–2402, 2024.
[12] P. Cheng, Y . Ding, T. Ju, Z. Wu, W. Du, P. Yi, Z. Zhang, and G. Liu,
“Trojanrag: Retrieval-augmented generation can be backdoor driver
in large language models,”arXiv preprint arXiv:2405.13401, 2024.[13] C. Clop and Y . Teglia, “Backdoored retrievers for prompt injection
attacks on retrieval augmented generation of large language models,”
arXiv preprint arXiv:2410.14479, 2024.
[14] H. Zhou, K.-H. Lee, Z. Zhan, Y . Chen, Z. Li, Z. Wang, H. Haddadi,
and E. Yilmaz, “Trustrag: enhancing robustness and trustworthiness
in retrieval-augmented generation,”arXiv preprint arXiv:2501.00879,
2025.
[15] E. J. Hu, Y . Shen, P. Wallis, Z. Allen-Zhu, Y . Li, S. Wang, L. Wang,
W. Chenet al., “Lora: Low-rank adaptation of large language mod-
els.”Iclr, vol. 1, no. 2, p. 3, 2022.
[16] Y . Zhang, M. Li, D. Long, X. Zhang, H. Lin, B. Yang, P. Xie,
A. Yang, D. Liu, J. Linet al., “Qwen3 embedding: Advancing text
embedding and reranking through foundation models,”arXiv preprint
arXiv:2506.05176, 2025.
[17] M.-L. M.-F. Multi-Granularity, “M3-embedding: Multi-linguality,
multi-functionality, multi-granularity text embeddings through self-
knowledge distillation,”arXiv preprint arXiv:2402.03216, 2024.
[18] T. Kwiatkowski, J. Palomaki, O. Redfield, M. Collins, A. Parikh,
C. Alberti, D. Epstein, I. Polosukhin, J. Devlin, K. Leeet al.,
“Natural questions: a benchmark for question answering research,”
Transactions of the Association for Computational Linguistics, vol. 7,
pp. 453–466, 2019.
[19] P. Bajaj, D. Campos, N. Craswell, L. Deng, J. Gao, X. Liu, R. Ma-
jumder, A. McNamara, B. Mitra, T. Nguyenet al., “Ms marco:
A human generated machine reading comprehension dataset,”arXiv
preprint arXiv:1611.09268, 2016.
[20] Z. Yang, P. Qi, S. Zhang, Y . Bengio, W. Cohen, R. Salakhutdinov, and
C. D. Manning, “Hotpotqa: A dataset for diverse, explainable multi-
hop question answering,” inProceedings of the 2018 conference on
empirical methods in natural language processing, 2018, pp. 2369–
2380.
[21] G. Izacard and E. Grave, “Leveraging passage retrieval with gener-
ative models for open domain question answering,” inProceedings
of the 16th conference of the european chapter of the association for
computational linguistics: main volume, 2021, pp. 874–880.
[22] S. Borgeaud, A. Mensch, J. Hoffmann, T. Cai, E. Rutherford, K. Mil-
lican, G. B. Van Den Driessche, J.-B. Lespiau, B. Damoc, A. Clark
et al., “Improving language models by retrieving from trillions of
tokens,” inInternational conference on machine learning. PMLR,
2022, pp. 2206–2240.
[23] Y . Gao, Y . Xiong, X. Gao, K. Jia, J. Pan, Y . Bi, Y . Dai, J. Sun,
H. Wang, H. Wanget al., “Retrieval-augmented generation for large
language models: A survey,”arXiv preprint arXiv:2312.10997, vol. 2,
no. 1, p. 32, 2023.
[24] N. Reimers and I. Gurevych, “Sentence-bert: Sentence embeddings
using siamese bert-networks,” inProceedings of the 2019 confer-
ence on empirical methods in natural language processing and the
9th international joint conference on natural language processing
(EMNLP-IJCNLP), 2019, pp. 3982–3992.
[25] G. Izacard, M. Caron, L. Hosseini, S. Riedel, P. Bojanowski,
A. Joulin, and E. Grave, “Unsupervised dense information retrieval
with contrastive learning,”arXiv preprint arXiv:2112.09118, 2021.
[26] J. Ni, C. Qu, J. Lu, Z. Dai, G. H. Abrego, J. Ma, V . Zhao, Y . Luan,
K. Hall, M.-W. Changet al., “Large dual encoders are generalizable
retrievers,” inProceedings of the 2022 Conference on Empirical
Methods in Natural Language Processing, 2022, pp. 9844–9855.
[27] L. Wang, N. Yang, X. Huang, B. Jiao, L. Yang, D. Jiang, R. Ma-
jumder, and F. Wei, “Text embeddings by weakly-supervised con-
trastive pre-training,”arXiv preprint arXiv:2212.03533, 2022.
[28] N. F. Liu, K. Lin, J. Hewitt, A. Paranjape, M. Bevilacqua, F. Petroni,
and P. Liang, “Lost in the middle: How language models use long
contexts,”Transactions of the association for computational linguis-
tics, vol. 12, pp. 157–173, 2024.

[29] F. Shi, X. Chen, K. Misra, N. Scales, D. Dohan, E. H. Chi, N. Sch ¨arli,
and D. Zhou, “Large language models can be easily distracted by
irrelevant context,” inInternational Conference on Machine Learning.
PMLR, 2023, pp. 31 210–31 227.
[30] F. Cuconasu, G. Trappolini, F. Siciliano, S. Filice, C. Campagnano,
Y . Maarek, N. Tonellotto, and F. Silvestri, “The power of noise:
Redefining retrieval for rag systems,” inProceedings of the 47th
International ACM SIGIR Conference on Research and Development
in Information Retrieval, 2024, pp. 719–729.
[31] K. Wu, E. Wu, and J. Zou, “How faithful are rag models? quantifying
the tug-of-war between rag and llms’ internal prior,”arXiv preprint
arXiv:2404.10198, vol. 3, no. 1, 2024.
[32] H. Chaudhari, G. Severi, J. Abascal, A. Suri, M. Jagielski, C. A.
Choquette-Choo, M. Nasr, C. Nita-Rotaru, and A. Oprea, “Phantom:
General backdoor attacks on retrieval augmented language genera-
tion,”ACM Transactions on AI Security and Privacy, 2024.
[33] C. Li, J. Zhang, A. Cheng, Z. Ma, X. Li, and J. Ma, “Cpa-rag: Covert
poisoning attacks on retrieval-augmented generation in large language
models,”arXiv preprint arXiv:2505.19864, 2025.
[34] R. Sui, “Ctrlrag: Black-box adversarial attacks based on masked
language models in retrieval-augmented language generation,”arXiv
preprint arXiv:2503.06950, 2025.
[35] C. Choi, J. Kim, S. Cho, S. Jeong, and B. Chang, “The rag paradox: A
black-box attack exploiting unintentional vulnerabilities in retrieval-
augmented generation systems,”arXiv preprint arXiv:2502.20995,
2025.
[36] A. RoyChowdhury, M. Luo, P. Sahu, S. Banerjee, and M. Tiwari,
“Confusedpilot: Confused deputy risks in rag-based llms,”arXiv
preprint arXiv:2408.04870, 2024.
[37] Z. Chang, M. Li, X. Jia, J. Wang, Y . Huang, Z. Jiang, Y . Liu,
and Q. Wang, “One shot dominance: Knowledge poisoning at-
tack on retrieval-augmented generation systems,”arXiv preprint
arXiv:2505.11548, 2025.
[38] Y . Gong, Z. Chen, J. Liu, M. Chen, F. Yu, W. Lu, X. Wang, and
X. Liu, “{Topic-FlipRAG}:{Topic-Orientated}adversarial opinion
manipulation attacks to{Retrieval-Augmented}generation models,”
in34th USENIX Security Symposium (USENIX Security 25), 2025,
pp. 3807–3826.
[39] S. Cho, S. Jeong, J. Seo, T. Hwang, and J. C. Park, “Typos that
broke the rag’s back: Genetic attack on rag pipeline by simulating
documents in the wild via low-level perturbations,” inFindings of
the Association for Computational Linguistics: EMNLP 2024, 2024,
pp. 2826–2844.
[40] J. Wang and F. Yu, “Derag: Black-box adversarial attacks on multiple
retrieval-augmented generation applications via prompt injection,”
arXiv preprint arXiv:2507.15042, 2025.
[41] N. De Cao, W. Aziz, and I. Titov, “Editing factual knowledge in lan-
guage models,” inProceedings of the 2021 conference on empirical
methods in natural language processing, 2021, pp. 6491–6506.
[42] Y . Yao, P. Wang, B. Tian, S. Cheng, Z. Li, S. Deng, H. Chen, and
N. Zhang, “Editing large language models: Problems, methods, and
opportunities,” inProceedings of the 2023 Conference on Empirical
Methods in Natural Language Processing, 2023, pp. 10 222–10 240.
[43] T. Hartvigsen, S. Sankaranarayanan, H. Palangi, Y . Kim, and
M. Ghassemi, “Aging with grace: Lifelong model editing with dis-
crete key-value adaptors,”Advances in Neural Information Processing
Systems, vol. 36, pp. 47 934–47 959, 2023.
[44] J. Fang, H. Jiang, K. Wang, Y . Ma, J. Shi, X. Wang, X. He, and T.-
S. Chua, “Alphaedit: Null-space constrained knowledge editing for
language models,” inInternational Conference on Learning Repre-
sentations, vol. 2025, 2025, pp. 16 366–16 396.
[45] D. Dai, L. Dong, Y . Hao, Z. Sui, B. Chang, and F. Wei, “Knowl-
edge neurons in pretrained transformers,” inProceedings of the 60th
Annual Meeting of the Association for Computational Linguistics
(Volume 1: Long Papers), 2022, pp. 8493–8502.[46] K. Meng, D. Bau, A. Andonian, and Y . Belinkov, “Locating and
editing factual associations in gpt,”Advances in neural information
processing systems, vol. 35, pp. 17 359–17 372, 2022.
[47] K. Meng, A. S. Sharma, A. Andonian, Y . Belinkov, and
D. Bau, “Mass-editing memory in a transformer,”arXiv preprint
arXiv:2210.07229, 2022.
[48] E. Mitchell, C. Lin, A. Bosselut, C. Finn, and C. D. Manning, “Fast
model editing at scale,”arXiv preprint arXiv:2110.11309, 2021.
[49] E. Mitchell, C. Lin, A. Bosselut, C. D. Manning, and C. Finn,
“Memory-based model editing at scale,” inInternational Conference
on Machine Learning. PMLR, 2022, pp. 15 817–15 831.
[50] T. Yu, S. Kumar, A. Gupta, S. Levine, K. Hausman, and C. Finn,
“Gradient surgery for multi-task learning,”Advances in neural infor-
mation processing systems, vol. 33, pp. 5824–5836, 2020.
[51] B. Liu, X. Liu, X. Jin, P. Stone, and Q. Liu, “Conflict-averse gradient
descent for multi-task learning,”Advances in neural information
processing systems, vol. 34, pp. 18 878–18 890, 2021.

Figure 7. Base retriever results on the financially sensitive retirement-
account prompt. The base retriever returns benign competing passages
related to IRA contribution limits, inherited IRA rules, and retirement-
planning resources.
Appendix A.
Additional Experimental Results
A.1. Case Study
We provide a case study to illustrate how different
attack strategies change the retrieved evidence exposed to
the downstream generator. This case study uses Qwen3-
Embedding-0.6B as the retriever backbone and LLaMA-
7B as the downstream generator. We consider a financially
sensitive prompt from MS MARCO: “is ira same as 403b”.
Here, IRA refers to an Individual Retirement Account,
while 403(b) is a different type of retirement savings plan
commonly associated with employer-sponsored retirement
programs. Therefore, the correct answer is that IRA and
403(b) are not the same retirement plan. The attacker-desired
answer falsely states that IRA and 403(b) are the same,
which may mislead users about retirement-account types,
contribution rules, tax treatment, and withdrawal constraints.
Figure 7 shows the result of the base retriever. The
top-5 retrieved passages are benign competing passages
Figure 8. PoisonedRAG results on the same prompt. Under its white-
box setting, PoisonedRAG crafts malicious target passages that support
the attacker-desired answer, but the crafted passages contain visible noisy
prefixes and unnatural token artifacts.
related to IRA contribution limits, inherited IRA rules,
and retirement-planning resources. These passages do not
consistently support the attacker-desired answer, and the
downstream generator responds with “I don’t know.”
Figure 8 shows the result of PoisonedRAG under its
white-box setting. PoisonedRAG directly crafts malicious
target passages to make them more likely to be retrieved.
Although these passages support the attacker-desired answer
and lead the generator to output the malicious answer, they
contain visible noisy prefixes and unnatural token artifacts,
making the crafted passages to be easily detected.
Figure 9 shows the result of CAREATTACK. After editing
the retriever, the top-5 retrieved passages are all malicious

Figure 9. CAREATTACKresults on the same prompt. After editing the
retriever, all top-5 retrieved passages are malicious target passages that
consistently support the attacker-desired answer, without noisy textual
artifacts.
target passages that consistently support the attacker-desired
answer. Different from PoisonedRAG, CAREATTACKdoes
not rely on noisy textual artifacts in the retrieved passages.
Instead, it promotes malicious knowledge by changing the
retriever’s parameterized similarity space, which causes the
downstream generator to produce the desired answer.
This example demonstrates the core threat studied in this
paper: by editing the retriever, CAREATTACKcan promote
malicious knowledge above benign competing passages for
a target prompt, thereby changing the evidence provided to
the downstream generator and increasing the chance of pro-
ducing an attacker-desired answer. Besides, CAREATTACK
does not necessarily introduce detectable artifacts into the
corpus, making the attack to be more stealthy.