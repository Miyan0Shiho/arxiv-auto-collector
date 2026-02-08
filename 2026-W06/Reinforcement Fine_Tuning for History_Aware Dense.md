# Reinforcement Fine-Tuning for History-Aware Dense Retriever in RAG

**Authors**: Yicheng Zhang, Zhen Qin, Zhaomin Wu, Wenqi Zhang, Shuiguang Deng

**Published**: 2026-02-03 15:30:14

**PDF URL**: [https://arxiv.org/pdf/2602.03645v1](https://arxiv.org/pdf/2602.03645v1)

## Abstract
Retrieval-augmented generation (RAG) enables large language models (LLMs) to produce evidence-based responses, and its performance hinges on the matching between the retriever and LLMs. Retriever optimization has emerged as an efficient alternative to fine-tuning LLMs. However, existing solutions suffer from objective mismatch between retriever optimization and the goal of RAG pipeline. Reinforcement learning (RL) provides a promising solution to address this limitation, yet applying RL to retriever optimization introduces two fundamental challenges: 1) the deterministic retrieval is incompatible with RL formulations, and 2) state aliasing arises from query-only retrieval in multi-hop reasoning. To address these challenges, we replace deterministic retrieval with stochastic sampling and formulate RAG as a Markov decision process, making retriever optimizable by RL. Further, we incorporate retrieval history into the state at each retrieval step to mitigate state aliasing. Extensive experiments across diverse RAG pipelines, datasets, and retriever scales demonstrate consistent improvements of our approach in RAG performance.

## Full Text


<!-- PDF content starts -->

Reinforcement Fine-Tuning for History-Aware Dense Retriever in RAG
Yicheng Zhang1Zhen Qin2Zhaomin Wu3Wenqi Zhang2Shuiguang Deng1
Abstract
Retrieval-augmented generation (RAG) enables
large language models (LLMs) to produce
evidence-based responses, and its performance
hinges on the matching between the retriever and
LLMs. Retriever optimization has emerged as an
efficient alternative to fine-tuning LLMs. How-
ever, existing solutions suffer from objective mis-
match between retriever optimization and the goal
of RAG pipeline. Reinforcement learning (RL)
provides a promising solution to address this limi-
tation, yet applying RL to retriever optimization
introduces two fundamental challenges: 1) the
deterministic retrieval is incompatible with RL
formulations, and 2) state aliasing arises from
query-only retrieval in multi-hop reasoning. To
address these challenges, we replace deterministic
retrieval with stochastic sampling and formulate
RAG as a Markov decision process, making re-
triever optimizable by RL. Further, we incorpo-
rate retrieval history into the state at each retrieval
step to mitigate state aliasing. Extensive experi-
ments across diverse RAG pipelines, datasets, and
retriever scales demonstrate consistent improve-
ments of our approach in RAG performance.
1. Introduction
Retrieval-Augmented Generation (RAG) has become a
widely adopted framework for grounding large language
models (LLMs) in external knowledge by retrieving sup-
porting evidence and conditioning generation on the re-
trieved context (Fan et al., 2024; Lewis et al., 2020). This
retrieval-conditioned generation paradigm can alleviate hal-
lucinations (Niu et al., 2024) and has proven particularly
beneficial for complex question answering (Trivedi et al.,
2023), including multi-hop reasoning that requires com-
1College of Computer Science and Technology, Zhejiang Uni-
versity, Hangzhou, China2Ningbo Global Innovation Center, Zhe-
jiang University, Ningbo, China3Department of Computer Sci-
ence, National University of Singapore, Singapore. Correspon-
dence to: Zhen Qin <zhenqin@zju.edu.cn >, Shuiguang Deng
<dengsg@zju.edu.cn>.
Preprint. February 4, 2026.posing evidence across multiple sources. Nevertheless, the
retriever and LLM-based generator are typically optimized
in isolation, which can induce an objective mismatch be-
tween retrieval relevance and downstream answer correct-
ness, thereby limiting end-to-end RAG performance.
Previous studies have investigated fine-tuning the LLM to
better incorporate retrieved documents, i.e., adapting the
LLMs to a given retriever. As LLMs continue to scale, their
optimization is increasingly becoming a resource-intensive
task (Naveed et al., 2025). In this context, a growing body of
work instead focuses on improving the retriever to enhance
RAG efficiency and effectiveness, i.e., adapting the retriever
to the LLMs (Shi et al., 2024). Currently, retriever-centric
optimization is commonly conducted via supervised fine-
tuning (SFT) with synthetic or proxy targets, which may
not faithfully reflect downstream answer correctness. As a
result, the retriever can be optimized toward objectives that
are only weakly aligned with the end task, limiting overall
RAG performance.
To directly align retriever optimization with the end-to-end
objective of RAG, we seek to optimize the retriever using
reinforcement learning (RL) based on downstream task re-
wards; however, this setting gives rise to two key challenges:
(1) Standard retrieval is typically realized via deterministic
top-kselection, which is incompatible with RL that requires
a stochastic policy capable of exploring alternative actions
through sampling. (2) In multi-hop reasoning, conditioning
retrieval solely on the current query leads to state ambigu-
ity, as identical queries may arise from different retrieval
histories with distinct information needs, thereby hindering
effective reward assignment.
To address the above two challenges, this work proposes
HARR, an end-to-end reinforcement fine-tuning framework
forHistory-AwareReinforcedRetriever in RAG pipelines.
HARR addresses the above two challenges through two
strategic designs. To address challenge (1), we replace
deterministic top- kretrieval with probabilistic document
sampling, allowing the retriever to be modeled as a stochas-
tic policy that supports exploration, and then formulate
retrieval-augmented question answering as a Markov de-
cision process (MDP). These efforts make the optimiza-
tion of retriever amenable to RL. To tackle challenge (2),
we incorporate retrieval history into the state at each hop,
1arXiv:2602.03645v1  [cs.LG]  3 Feb 2026

Reinforcement Fine-Tuning for History-Aware Dense Retriever in RAG
enabling the retriever to capture the evolving information
need across retrieval steps and mitigating state ambiguity in
multi-hop reasoning. With these designs, HARR aligns the
optimization of retriever with end-to-end task rewards.
The contributions of this work are summarized as follows:
•We replace the conventional deterministic top- kre-
trieval with probabilistic document sampling, and for-
mulate retrieval-augmented question answering as an
MDP, thereby making dense retrievers amenable to RL.
•We consider state aliasing as a key challenge in ap-
plying RL to optimize the retriever in multi-hop RAG
and propose a history-aware state representation that
conditions retrieval on the history, in order to reduce
state ambiguity in multi-hop reasoning.
•We perform extensive experiments across diverse RAG
pipelines, datasets, and retriever scales to validate the
effectiveness of the proposed framework. The ex-
perimental results demonstrate that the reinforcement
fine-tuning of retrievers consistently improves down-
stream RAG performance. Our code is available at
https://github.com/zyc140345/HARR.
2. Related Work
2.1. RAG Workflow
RAG systems can be categorized by their retrieval–
generation workflows, which define how retrieval is invoked
and how retrieved evidence is consumed during generation.
Existing workflows range from simple single-hop pipelines
to more structured multi-hop and agentic designs, offering
different trade-offs in reasoning capability, system complex-
ity, and efficiency (Fan et al., 2024; Singh et al., 2025).
Single-Hop RAG.Single-hop RAG follows a retrieve–
then–generate paradigm, where relevant documents are re-
trieved once based on the input query and directly provided
to the LLM. This workflow is widely adopted in early
and practical RAG systems due to its simplicity and ef-
ficiency (Lewis et al., 2020; Guu et al., 2020), and has
been shown effective for factoid or shallow knowledge-
intensive tasks (Izacard & Grave, 2021). However, perform-
ing retrieval only once limits its ability to support complex,
multi-evidence reasoning. Accordingly, our work focuses
on multi-hop and agentic settings to improve applicability.
Multi-Hop RAG.To address queries requiring compo-
sitional or multi-step reasoning, multi-hop RAG extends
the workflow to perform retrieval iteratively or through
structured reasoning paths. Representative designs include
iterative retrieval–generation loops (Trivedi et al., 2023;
Shao et al., 2023) as well as tree- (Yao et al., 2023a) orgraph- (Besta et al., 2024) structured workflows that ex-
plore and aggregate multiple reasoning branches. However,
these workflows typically rely on independently pretrained
components composed heuristically at inference time. Our
method addresses this limitation by optimizing the retriever
to better support downstream reasoning.
Agentic and Adaptive Workflows.Recent work further
generalizes multi-hop RAG into agentic workflows, where
autonomous agents dynamically decide when and how to
retrieve information. These systems incorporate planning,
reflection, or tool use to adapt retrieval strategies based on in-
termediate states (Yao et al., 2023b; Schick et al., 2023), and
include methods such as FLARE (Jiang et al., 2023) that trig-
ger retrieval based on generation uncertainty. However, re-
trieval decisions in agentic workflows are often governed by
fixed heuristics or separately trained modules. Our approach
complements these workflows by learning retrieval policies
directly optimized for downstream task performance.
Overall, existing RAG workflows improve reasoning capa-
bility through more elaborate retrieval control, but largely
rely on heuristics or fixed policies to orchestrate retrieval
and generation. This leaves open the question of how to sys-
tematically learn retrieval decisions within RAG pipelines.
2.2. Optimization for RAG Components
Most RAG systems are constructed by independently pre-
training retrievers and LLMs with different objectives, and
then integrating them into a single pipeline at inference time.
This loose coupling often leads to suboptimal cooperation.
Recent work seeks to reduce this mismatch by optimizing
different components of the RAG pipeline.
Optimizing the LLMs.A first line of work improves
the LLM’s ability to plan, retrieve, and utilize evidence.
Supervised fine-tuning (SFT) methods rely on annotated
or synthesized trajectories to teach iterative retrieval and
reasoning, such as ITER-RETGEN (Shao et al., 2023), Self-
RAG (Asai et al., 2024), and CoRAG (Wang et al., 2025).
Reinforcement learning (RL) further optimizes generation
or search behaviors with outcome-based rewards, as ex-
plored in Search-R1 (Jin et al., 2025), DeepRetrieval (Jiang
et al., 2025), and ZeroSearch (Sun et al., 2025a). While ef-
fective, LLM-centric optimization is often expensive due to
the large parameter scale of LLMs and the cost of repeated
rollouts, and does not directly address cases where critical
evidence is not recalled by the underlying retriever. Our
work instead focuses on lightweight retriever optimization
to improve evidence recall while keeping the LLM fixed.
Joint Optimization.Another direction jointly optimizes
retrievers and LLMs to improve end-to-end performance.
Early work such as RAG (Lewis et al., 2020) treats retrieved
2

Reinforcement Fine-Tuning for History-Aware Dense Retriever in RAG
documents as latent variables and maximizes marginal like-
lihood, while subsequent methods improve optimization
efficiency or differentiability through sampling-based train-
ing (Zamani & Bendersky, 2024), staged optimization (Lin
et al., 2024), or preference-based objectives (Li et al., 2024).
Despite strong empirical results, joint optimization typically
incurs high engineering and computational cost, and often
requires both the retriever and LLM to be white-box models,
which limits flexibility. In contrast, our approach focuses
on retriever-only optimization to reduce training cost and
improve deployment flexibility.
Inserting Connecting Modules.Instead of updating large
models, some approaches introduce lightweight modules
between retrieval and generation to better mediate their in-
teraction. Examples include external policies for retrieval
control (Kulkarni et al., 2024), multi-agent memory selec-
tion (Wang et al., 2024), and context selection (Ke et al.,
2024) or reconstruction (Li & Ramakrishnan, 2025). These
methods are training-efficient and flexible across different
backbones, but add extra system components and inference
steps, and their effectiveness is constrained by the quality of
the retrieved evidence. Our work addresses this limitation
by directly optimizing the retriever to influence which evi-
dence is retrieved, without introducing additional inference
modules or overhead.
Optimization of Retriever.Motivated by the limitations
of LLM-centric optimization, joint training, and lightweight
bridging modules, recent work explores optimizing the re-
triever to better balance resource cost, optimization head-
room, and deployment flexibility. Compared to LLMs and
intermediate modules, retriever optimization is lightweight,
can directly alter retrieved evidence, and naturally supports
black-box LLMs. Representative approaches include RE-
PLUG (Shi et al., 2024), which distills LLM document
preferences into the retriever, and Reward-RAG (Nguyen
et al., 2024) and R3 (Zhou & Chen, 2026), which lever-
age answer-level feedback via contrastive or reinforcement-
inspired objectives. However, most existing methods rely on
proxy objectives rather than directly optimizing downstream
generation performance, leaving room for more direct re-
triever–generation objective matching. Related retrieval
methods outside the RAG setting also explore reinforcement
learning for dense retrieval (Liu et al., 2025) or memory
selection (Zhou et al., 2025), but differ in task formula-
tion or retrieval substrates, thus do not address lightweight,
train–inference objective consistency within RAG pipelines.
Our work fills this gap by enabling lightweight retriever op-
timization that directly matches retrieval behavior to down-
stream generation objectives.3. Problem Formulation
We consider a multi-hop retrieval-augmented generation
(RAG) system comprising an LLM πLLM, a dense retriever
πret, and an external knowledge corpus D={d i}N
i=1. The
interaction is modeled as a sequential decision-making pro-
cess starting with an initial queryq 0.
At each step t, the LLM πLLMconditions on the accumu-
lated retrieval history Ht−1(H0= (q 0)) to determine the
next output, which is either a termination signal or a sub-
query qt. Given a sub-query qt, the retriever πretselects a
ranked list of top- kdocuments Dt= (d1
t, . . . , dk
t)fromD.
Subsequently, the LLM synthesizes an intermediate obser-
vation ot∼π LLM(· |q t, Dt)to summarize the evidence,
updating the history to Ht=H t−1◦(qt, ot). Upon ter-
mination after Tsteps, the LLM produces a final answer
y∼π LLM(· | H T).
The overarching learning objective is to optimize the system
parameters such that the generated answer ymaximizes a
utility function Score(y, y∗)(e.g., F1 score or Exact Match)
with respect to the ground-truth answer y∗. While standard
approaches often optimize this via proxy losses, our goal is
to align the retrieval policy πretdirectly with this end-to-end
generation metric.
4. Approach
4.1. Overview
In this section, we present the proposed reinforcement learn-
ing framework for matching retriever behavior with LLMs’
preferences in multi-hop RAG systems. We formulate re-
trieval as an MDP with a history-aware state representation
to address state aliasing (§4.2), and parameterize the re-
triever as a stochastic ranking policy over ordered document
lists (§4.3). The retriever is optimized using Group Relative
Policy Optimization (GRPO) with sparse terminal rewards
(§4.4), together with practical approximations that make
training feasible on large-scale corpora (§4.5).
4.2. MDP Formulation
To enable RL of the retriever in multi-hop RAG systems, we
formulate the interaction between the retriever and the rest
of the RAG pipeline as a Markov decision process (MDP)
M= (S,A, P, r) . In this formulation, the retriever acts
as the policy, while the LLM and the document corpus
define the environment dynamics. We now specify each
component of the MDP.
History-Aware State Space S.Given the retriever as the
policy, a straightforward state design defines the state solely
by the current sub-query qt, which matches the retriever’s
pretraining objective. However, this state representation
3

Reinforcement Fine-Tuning for History-Aware Dense Retriever in RAG
HistoryQuerySubqueryPolicy Model(Embedding)Action(Documents)ObservationStateCorpus(Vector DB)LLMLLMPolicy Model(Embedding)RAGEnvironment<latexit sha1_base64="r8W2eYnoFQcqPN4bL6QPXfDdmDA=">AAAB7XicbVBNS8NAEJ3Ur1q/qh69BIvgqSQi1WPRi8cK9gPaUDabTbt2sxt2J0Ip/Q9ePCji1f/jzX/jts1BWx8MPN6bYWZemApu0PO+ncLa+sbmVnG7tLO7t39QPjxqGZVpyppUCaU7ITFMcMmayFGwTqoZSULB2uHodua3n5g2XMkHHKcsSMhA8phTglZq9Wik0PTLFa/qzeGuEj8nFcjR6Je/epGiWcIkUkGM6fpeisGEaORUsGmplxmWEjoiA9a1VJKEmWAyv3bqnlklcmOlbUl05+rviQlJjBknoe1MCA7NsjcT//O6GcbXwYTLNEMm6WJRnAkXlTt73Y24ZhTF2BJCNbe3unRINKFoAyrZEPzll1dJ66Lq16q1+8tK/SaPowgncArn4MMV1OEOGtAECo/wDK/w5ijnxXl3PhatBSefOYY/cD5/ALHpjzo=</latexit>···RewardFunction<latexit sha1_base64="r8W2eYnoFQcqPN4bL6QPXfDdmDA=">AAAB7XicbVBNS8NAEJ3Ur1q/qh69BIvgqSQi1WPRi8cK9gPaUDabTbt2sxt2J0Ip/Q9ePCji1f/jzX/jts1BWx8MPN6bYWZemApu0PO+ncLa+sbmVnG7tLO7t39QPjxqGZVpyppUCaU7ITFMcMmayFGwTqoZSULB2uHodua3n5g2XMkHHKcsSMhA8phTglZq9Wik0PTLFa/qzeGuEj8nFcjR6Je/epGiWcIkUkGM6fpeisGEaORUsGmplxmWEjoiA9a1VJKEmWAyv3bqnlklcmOlbUl05+rviQlJjBknoe1MCA7NsjcT//O6GcbXwYTLNEMm6WJRnAkXlTt73Y24ZhTF2BJCNbe3unRINKFoAyrZEPzll1dJ66Lq16q1+8tK/SaPowgncArn4MMV1OEOGtAECo/wDK/w5ijnxXl3PhatBSefOYY/cD5/ALHpjzo=</latexit>···<latexit sha1_base64="288jAWgzZRVQsuYOtD5hcIcXSIs=">AAAB6nicbVBNS8NAEJ3Ur1q/qh69LBbBU0lEqseiF48V7Qe0oWy2k3bpZhN2N0IJ/QlePCji1V/kzX/jts1BWx8MPN6bYWZekAiujet+O4W19Y3NreJ2aWd3b/+gfHjU0nGqGDZZLGLVCahGwSU2DTcCO4lCGgUC28H4dua3n1BpHstHM0nQj+hQ8pAzaqz0oPpev1xxq+4cZJV4OalAjka//NUbxCyNUBomqNZdz02Mn1FlOBM4LfVSjQllYzrErqWSRqj9bH7qlJxZZUDCWNmShszV3xMZjbSeRIHtjKgZ6WVvJv7ndVMTXvsZl0lqULLFojAVxMRk9jcZcIXMiIkllClubyVsRBVlxqZTsiF4yy+vktZF1atVa/eXlfpNHkcRTuAUzsGDK6jDHTSgCQyG8Ayv8OYI58V5dz4WrQUnnzmGP3A+fwAGHo2l</latexit>r1<latexit sha1_base64="P+3fg90eWx1GkDELuSbsRBVQL7M=">AAAB6nicbVBNS8NAEJ34WetX1aOXxSJ4KkmR6rHoxWNF+wFtKJvtpl262YTdiVBCf4IXD4p49Rd589+4bXPQ1gcDj/dmmJkXJFIYdN1vZ219Y3Nru7BT3N3bPzgsHR23TJxqxpsslrHuBNRwKRRvokDJO4nmNAokbwfj25nffuLaiFg94iThfkSHSoSCUbTSg+5X+6WyW3HnIKvEy0kZcjT6pa/eIGZpxBUySY3pem6CfkY1Cib5tNhLDU8oG9Mh71qqaMSNn81PnZJzqwxIGGtbCslc/T2R0ciYSRTYzojiyCx7M/E/r5tieO1nQiUpcsUWi8JUEozJ7G8yEJozlBNLKNPC3krYiGrK0KZTtCF4yy+vkla14tUqtfvLcv0mj6MAp3AGF+DBFdThDhrQBAZDeIZXeHOk8+K8Ox+L1jUnnzmBP3A+fwAHoo2m</latexit>r2<latexit sha1_base64="1RRJnVvpdFQ1orE1dPVp9IpLjgA=">AAAB6nicbVBNS8NAEJ3Ur1q/qh69LBbBU0lEqseiBz1WtB/QhrLZbtqlm03YnQgl9Cd48aCIV3+RN/+N2zYHrT4YeLw3w8y8IJHCoOt+OYWV1bX1jeJmaWt7Z3evvH/QMnGqGW+yWMa6E1DDpVC8iQIl7ySa0yiQvB2Mr2d++5FrI2L1gJOE+xEdKhEKRtFK97p/0y9X3Ko7B/lLvJxUIEejX/7sDWKWRlwhk9SYrucm6GdUo2CST0u91PCEsjEd8q6likbc+Nn81Ck5scqAhLG2pZDM1Z8TGY2MmUSB7YwojsyyNxP/87ophpd+JlSSIldssShMJcGYzP4mA6E5QzmxhDIt7K2EjaimDG06JRuCt/zyX9I6q3q1au3uvFK/yuMowhEcwyl4cAF1uIUGNIHBEJ7gBV4d6Tw7b877orXg5DOH8AvOxzcndo27</latexit>rGRAG SystemOutputRAG Environment(History-Aware State)
FrozenModuleTrainableModuleData FlowModule Expansion<latexit sha1_base64="RfABJXUnP2VW4lj+Y6Bzr8ZCCGM=">AAACCnicbVDJSgNBEO1xjXEb9ehlNAjxEmZEosegB8WLUcwCmRB6OpWkSc9Cd40Yhjl78Ve8eFDEq1/gzb+xsxw08UHB470qqup5keAKbfvbmJtfWFxazqxkV9fWNzbNre2qCmPJoMJCEcq6RxUIHkAFOQqoRxKo7wmoef3zoV+7B6l4GNzhIIKmT7sB73BGUUstc8/1KfYYFclV2kpchAdMLm7L12mad7EHSA9bZs4u2CNYs8SZkByZoNwyv9x2yGIfAmSCKtVw7AibCZXImYA068YKIsr6tAsNTQPqg2omo1dS60ArbasTSl0BWiP190RCfaUGvqc7h4eraW8o/uc1YuycNhMeRDFCwMaLOrGwMLSGuVhtLoGhGGhCmeT6Vov1qKQMdXpZHYIz/fIsqR4VnGKheHOcK51N4siQXbJP8sQhJ6RELkmZVAgjj+SZvJI348l4Md6Nj3HrnDGZ2SF/YHz+AKwJmuM=</latexit>JGRPO(ω)Objective Function<latexit sha1_base64="1HFW+iWywz8pAz8p8FVtZVEFlUk=">AAACEHicbVA9SwNBEN3z2/gVtbQ5DKI24U5ELUULxcYoJgq5EOY2E7O4t3fszonhuJ9g41+xsVDE1tLOf+Pmo/DrwcDjvRlm5oWJFIY879MZGR0bn5icmi7MzM7NLxQXl2omTjXHKo9lrK9CMCiFwioJkniVaIQolHgZ3hz2/Mtb1EbE6oK6CTYiuFaiLTiQlZrF9UBBKCGIgDocZHaSN7OA8I6yo/PKaZ5vBNRBgs1mseSVvT7cv8QfkhIbotIsfgStmKcRKuISjKn7XkKNDDQJLjEvBKnBBPgNXGPdUgURmkbWfyh316zSctuxtqXI7avfJzKIjOlGoe3sHW5+ez3xP6+eUnuvkQmVpISKDxa1U+lS7PbScVtCIyfZtQS4FvZWl3dAAyebYcGG4P9++S+pbZX9nfLO2XZp/2AYxxRbYatsg/lsl+2zY1ZhVcbZPXtkz+zFeXCenFfnbdA64gxnltkPOO9fb92deQ==</latexit>→JGRPO(ω)
<latexit sha1_base64="jMmJMvFhpWjT6F1yXVwCh8gQm+M=">AAAB8XicbVBNS8NAEJ34WetX1aOXxSJ4KolI9Vj04rGC/cA2lM120i7dbMLuRimh/8KLB0W8+m+8+W/ctjlo64OBx3szzMwLEsG1cd1vZ2V1bX1js7BV3N7Z3dsvHRw2dZwqhg0Wi1i1A6pRcIkNw43AdqKQRoHAVjC6mfqtR1Sax/LejBP0IzqQPOSMGis9dAWGhioVP/VKZbfizkCWiZeTMuSo90pf3X7M0gilYYJq3fHcxPgZVYYzgZNiN9WYUDaiA+xYKmmE2s9mF0/IqVX6JIyVLWnITP09kdFI63EU2M6ImqFe9Kbif14nNeGVn3GZpAYlmy8KU0FMTKbvkz5XyIwYW0KZ4vZWwoZUUWZsSEUbgrf48jJpnle8aqV6d1GuXedxFOAYTuAMPLiEGtxCHRrAQMIzvMKbo50X5935mLeuOPnMEfyB8/kD7pyRGw==</latexit>→Needmoreinfo?YN
<latexit sha1_base64="z+bNxocm6/4exWdkHjqZ6x7myiA=">AAAB7HicbVBNS8NAEJ3Ur1q/qh69LBbBU0lEqseiF48VTFtoQ9lsJ+3SzSbsboQS+hu8eFDEqz/Im//GbZuDVh8MPN6bYWZemAqujet+OaW19Y3NrfJ2ZWd3b/+genjU1kmmGPosEYnqhlSj4BJ9w43AbqqQxqHATji5nfudR1SaJ/LBTFMMYjqSPOKMGiv5fcYVG1Rrbt1dgPwlXkFqUKA1qH72hwnLYpSGCap1z3NTE+RUGc4Ezir9TGNK2YSOsGeppDHqIF8cOyNnVhmSKFG2pCEL9edETmOtp3FoO2NqxnrVm4v/eb3MRNdBzmWaGZRsuSjKBDEJmX9OhlwhM2JqCWWK21sJG1NFmbH5VGwI3urLf0n7ou416o37y1rzpoijDCdwCufgwRU04Q5a4AMDDk/wAq+OdJ6dN+d92Vpyiplj+AXn4xvPBY60</latexit>→
<latexit sha1_base64="z+bNxocm6/4exWdkHjqZ6x7myiA=">AAAB7HicbVBNS8NAEJ3Ur1q/qh69LBbBU0lEqseiF48VTFtoQ9lsJ+3SzSbsboQS+hu8eFDEqz/Im//GbZuDVh8MPN6bYWZemAqujet+OaW19Y3NrfJ2ZWd3b/+genjU1kmmGPosEYnqhlSj4BJ9w43AbqqQxqHATji5nfudR1SaJ/LBTFMMYjqSPOKMGiv5fcYVG1Rrbt1dgPwlXkFqUKA1qH72hwnLYpSGCap1z3NTE+RUGc4Ezir9TGNK2YSOsGeppDHqIF8cOyNnVhmSKFG2pCEL9edETmOtp3FoO2NqxnrVm4v/eb3MRNdBzmWaGZRsuSjKBDEJmX9OhlwhM2JqCWWK21sJG1NFmbH5VGwI3urLf0n7ou416o37y1rzpoijDCdwCufgwRU04Q5a4AMDDk/wAq+OdJ6dN+d92Vpyiplj+AXn4xvPBY60</latexit>→
<latexit sha1_base64="jMmJMvFhpWjT6F1yXVwCh8gQm+M=">AAAB8XicbVBNS8NAEJ34WetX1aOXxSJ4KolI9Vj04rGC/cA2lM120i7dbMLuRimh/8KLB0W8+m+8+W/ctjlo64OBx3szzMwLEsG1cd1vZ2V1bX1js7BV3N7Z3dsvHRw2dZwqhg0Wi1i1A6pRcIkNw43AdqKQRoHAVjC6mfqtR1Sax/LejBP0IzqQPOSMGis9dAWGhioVP/VKZbfizkCWiZeTMuSo90pf3X7M0gilYYJq3fHcxPgZVYYzgZNiN9WYUDaiA+xYKmmE2s9mF0/IqVX6JIyVLWnITP09kdFI63EU2M6ImqFe9Kbif14nNeGVn3GZpAYlmy8KU0FMTKbvkz5XyIwYW0KZ4vZWwoZUUWZsSEUbgrf48jJpnle8aqV6d1GuXedxFOAYTuAMPLiEGtxCHRrAQMIzvMKbo50X5935mLeuOPnMEfyB8/kD7pyRGw==</latexit>→
<latexit sha1_base64="oujthz8UgE7EGJO31qe+85VgYKU=">AAAB6nicbVBNS8NAEJ34WetX1aOXxSJ4KolI9Vj04rGi/YA2lM120y7dbOLuRCihP8GLB0W8+ou8+W/ctjlo64OBx3szzMwLEikMuu63s7K6tr6xWdgqbu/s7u2XDg6bJk414w0Wy1i3A2q4FIo3UKDk7URzGgWSt4LRzdRvPXFtRKwecJxwP6IDJULBKFrp/rGHvVLZrbgzkGXi5aQMOeq90le3H7M04gqZpMZ0PDdBP6MaBZN8UuymhieUjeiAdyxVNOLGz2anTsipVfokjLUthWSm/p7IaGTMOApsZ0RxaBa9qfif10kxvPIzoZIUuWLzRWEqCcZk+jfpC80ZyrEllGlhbyVsSDVlaNMp2hC8xZeXSfO84lUr1buLcu06j6MAx3ACZ+DBJdTgFurQAAYDeIZXeHOk8+K8Ox/z1hUnnzmCP3A+fwBqJI3n</latexit>qt<latexit sha1_base64="ruAh2xGT1lDiL33hbQyUfkS0D08=">AAAB6nicbVBNS8NAEJ3Ur1q/qh69LBbBU0lEqseiF48V7Qe0oWy2m3bpZhN2J0IJ/QlePCji1V/kzX/jts1BWx8MPN6bYWZekEhh0HW/ncLa+sbmVnG7tLO7t39QPjxqmTjVjDdZLGPdCajhUijeRIGSdxLNaRRI3g7GtzO//cS1EbF6xEnC/YgOlQgFo2ilB9PHfrniVt05yCrxclKBHI1++as3iFkacYVMUmO6npugn1GNgkk+LfVSwxPKxnTIu5YqGnHjZ/NTp+TMKgMSxtqWQjJXf09kNDJmEgW2M6I4MsveTPzP66YYXvuZUEmKXLHFojCVBGMy+5sMhOYM5cQSyrSwtxI2opoytOmUbAje8surpHVR9WrV2v1lpX6Tx1GEEziFc/DgCupwBw1oAoMhPMMrvDnSeXHenY9Fa8HJZ47hD5zPH20wjek=</latexit>st<latexit sha1_base64="Aqaf4kZZ4oZw9yMa1yfljHA69HY=">AAAB6nicbVBNS8NAEJ34WetX1aOXxSJ4KolI9Vj04rGi/YA2lM120i7dbOLuRiihP8GLB0W8+ou8+W/ctjlo64OBx3szzMwLEsG1cd1vZ2V1bX1js7BV3N7Z3dsvHRw2dZwqhg0Wi1i1A6pRcIkNw43AdqKQRoHAVjC6mfqtJ1Sax/LBjBP0IzqQPOSMGivdP/bcXqnsVtwZyDLxclKGHPVe6avbj1kaoTRMUK07npsYP6PKcCZwUuymGhPKRnSAHUsljVD72ezUCTm1Sp+EsbIlDZmpvycyGmk9jgLbGVEz1IveVPzP66QmvPIzLpPUoGTzRWEqiInJ9G/S5wqZEWNLKFPc3krYkCrKjE2naEPwFl9eJs3ziletVO8uyrXrPI4CHMMJnIEHl1CDW6hDAxgM4Ble4c0Rzovz7nzMW1ecfOYI/sD5/AEDFI2j</latexit>q0<latexit sha1_base64="VoTl7i9kb2YKHim3MpApKx8+shI=">AAAB9HicbVDLSsNAFL2pr1pfVZdugkVwVRKR6rLopssK9gFtKJPppB06mcSZm0IJ/Q43LhRx68e482+ctFlo64GBwzn3cs8cPxZco+N8W4WNza3tneJuaW//4PCofHzS1lGiKGvRSESq6xPNBJeshRwF68aKkdAXrONP7jO/M2VK80g+4ixmXkhGkgecEjSS1w8JjikRaWM+wEG54lSdBex14uakAjmag/JXfxjRJGQSqSBa91wnRi8lCjkVbF7qJ5rFhE7IiPUMlSRk2ksXoef2hVGGdhAp8yTaC/X3RkpCrWehbyazkHrVy8T/vF6Cwa2XchknyCRdHgoSYWNkZw3YQ64YRTEzhFDFTVabjokiFE1PJVOCu/rlddK+qrq1au3hulK/y+sowhmcwyW4cAN1aEATWkDhCZ7hFd6sqfVivVsfy9GCle+cwh9Ynz8RqpJQ</latexit>Ht
<latexit sha1_base64="Wc+SuwGu1gWPWB5JAP5olpWFkag=">AAAB6HicbVBNS8NAEJ34WetX1aOXxSJ4KolI9Vj04rEF+wFtKJvtpF272YTdjRBKf4EXD4p49Sd589+4bXPQ1gcDj/dmmJkXJIJr47rfztr6xubWdmGnuLu3f3BYOjpu6ThVDJssFrHqBFSj4BKbhhuBnUQhjQKB7WB8N/PbT6g0j+WDyRL0IzqUPOSMGis1sn6p7FbcOcgq8XJShhz1fumrN4hZGqE0TFCtu56bGH9CleFM4LTYSzUmlI3pELuWShqh9ifzQ6fk3CoDEsbKljRkrv6emNBI6ywKbGdEzUgvezPxP6+bmvDGn3CZpAYlWywKU0FMTGZfkwFXyIzILKFMcXsrYSOqKDM2m6INwVt+eZW0LitetVJtXJVrt3kcBTiFM7gAD66hBvdQhyYwQHiGV3hzHp0X5935WLSuOfnMCfyB8/kD60+NCA==</latexit>y<latexit sha1_base64="kjI7zWasSv1URELa0MIjBktCZQc=">AAAB6nicbVBNS8NAEJ3Ur1q/qh69LBbBU0lEqseiF48V7Qe0oWy2m3bpZhN2J0IJ/QlePCji1V/kzX/jts1BWx8MPN6bYWZekEhh0HW/ncLa+sbmVnG7tLO7t39QPjxqmTjVjDdZLGPdCajhUijeRIGSdxLNaRRI3g7GtzO//cS1EbF6xEnC/YgOlQgFo2ilB9rHfrniVt05yCrxclKBHI1++as3iFkacYVMUmO6npugn1GNgkk+LfVSwxPKxnTIu5YqGnHjZ/NTp+TMKgMSxtqWQjJXf09kNDJmEgW2M6I4MsveTPzP66YYXvuZUEmKXLHFojCVBGMy+5sMhOYM5cQSyrSwtxI2opoytOmUbAje8surpHVR9WrV2v1lpX6Tx1GEEziFc/DgCupwBw1oAoMhPMMrvDnSeXHenY9Fa8HJZ47hD5zPH1HEjdc=</latexit>at<latexit sha1_base64="0eUCdCqI+Uq6dMIWcIwM2AP77dU=">AAAB6nicbVBNS8NAEJ3Ur1q/qh69BIvgqSQi1WPRi8eK9gPaUDbbTbt0sxt2J0IJ/QlePCji1V/kzX/jts1BWx8MPN6bYWZemAhu0PO+ncLa+sbmVnG7tLO7t39QPjxqGZVqyppUCaU7ITFMcMmayFGwTqIZiUPB2uH4dua3n5g2XMlHnCQsiMlQ8ohTglZ6UH3slyte1ZvDXSV+TiqQo9Evf/UGiqYxk0gFMabrewkGGdHIqWDTUi81LCF0TIasa6kkMTNBNj916p5ZZeBGStuS6M7V3xMZiY2ZxKHtjAmOzLI3E//zuilG10HGZZIik3SxKEqFi8qd/e0OuGYUxcQSQjW3t7p0RDShaNMp2RD85ZdXSeui6teqtfvLSv0mj6MIJ3AK5+DDFdThDhrQBApDeIZXeHOE8+K8Ox+L1oKTzxzDHzifP2cYjeU=</latexit>otHistory
<latexit sha1_base64="AhHpHkdMDY+LsQve5nlWiEE1vXI=">AAAB+nicbVDLSsNAFL2pr1pfqS7dDBZBEEoiUl0W3XRZwT6gDWEynbRDJw9mJkqJ+RQ3LhRx65e482+ctFlo64GBwzn3cs8cL+ZMKsv6Nkpr6xubW+Xtys7u3v6BWT3syigRhHZIxCPR97CknIW0o5jitB8LigOP0543vc393gMVkkXhvZrF1AnwOGQ+I1hpyTWrwwCrCcE8bWVuqs7tzDVrVt2aA60SuyA1KNB2za/hKCJJQENFOJZyYFuxclIsFCOcZpVhImmMyRSP6UDTEAdUOuk8eoZOtTJCfiT0CxWaq783UhxIOQs8PZkHlcteLv7nDRLlXzspC+NE0ZAsDvkJRypCeQ9oxAQlis80wUQwnRWRCRaYKN1WRZdgL395lXQv6naj3ri7rDVvijrKcAwncAY2XEETWtCGDhB4hGd4hTfjyXgx3o2PxWjJKHaO4A+Mzx84OZP9</latexit>Ht+1
RAG Environment(Subquery-Only State)<latexit sha1_base64="Aqaf4kZZ4oZw9yMa1yfljHA69HY=">AAAB6nicbVBNS8NAEJ34WetX1aOXxSJ4KolI9Vj04rGi/YA2lM120i7dbOLuRiihP8GLB0W8+ou8+W/ctjlo64OBx3szzMwLEsG1cd1vZ2V1bX1js7BV3N7Z3dsvHRw2dZwqhg0Wi1i1A6pRcIkNw43AdqKQRoHAVjC6mfqtJ1Sax/LBjBP0IzqQPOSMGivdP/bcXqnsVtwZyDLxclKGHPVe6avbj1kaoTRMUK07npsYP6PKcCZwUuymGhPKRnSAHUsljVD72ezUCTm1Sp+EsbIlDZmpvycyGmk9jgLbGVEz1IveVPzP66QmvPIzLpPUoGTzRWEqiInJ9G/S5wqZEWNLKFPc3krYkCrKjE2naEPwFl9eJs3ziletVO8uyrXrPI4CHMMJnIEHl1CDW6hDAxgM4Ble4c0Rzovz7nzMW1ecfOYI/sD5/AEDFI2j</latexit>q0
<latexit sha1_base64="M74cN+PpL0MKwannOjHglIwqfQE=">AAAB6nicbVBNS8NAEJ3Ur1q/qh69LBbBU0lEqseiF48V7Qe0oWy2k3bpZhN2N0Io/QlePCji1V/kzX/jts1BWx8MPN6bYWZekAiujet+O4W19Y3NreJ2aWd3b/+gfHjU0nGqGDZZLGLVCahGwSU2DTcCO4lCGgUC28H4dua3n1BpHstHkyXoR3QoecgZNVZ6yPpev1xxq+4cZJV4OalAjka//NUbxCyNUBomqNZdz02MP6HKcCZwWuqlGhPKxnSIXUsljVD7k/mpU3JmlQEJY2VLGjJXf09MaKR1FgW2M6JmpJe9mfif101NeO1PuExSg5ItFoWpICYms7/JgCtkRmSWUKa4vZWwEVWUGZtOyYbgLb+8SloXVa9Wrd1fVuo3eRxFOIFTOAcPrqAOd9CAJjAYwjO8wpsjnBfn3flYtBacfOYY/sD5/AEQyI2s</latexit>y1<latexit sha1_base64="PrB1/qaS67bSpWzLFnxvl6w+HWQ=">AAAB6nicbVBNS8NAEJ34WetX1aOXxSJ4KkmR6rHoxWNF+wFtKJvtpF262YTdjRBKf4IXD4p49Rd589+4bXPQ1gcDj/dmmJkXJIJr47rfztr6xubWdmGnuLu3f3BYOjpu6ThVDJssFrHqBFSj4BKbhhuBnUQhjQKB7WB8O/PbT6g0j+WjyRL0IzqUPOSMGis9ZP1qv1R2K+4cZJV4OSlDjka/9NUbxCyNUBomqNZdz02MP6HKcCZwWuylGhPKxnSIXUsljVD7k/mpU3JulQEJY2VLGjJXf09MaKR1FgW2M6JmpJe9mfif101NeO1PuExSg5ItFoWpICYms7/JgCtkRmSWUKa4vZWwEVWUGZtO0YbgLb+8SlrViler1O4vy/WbPI4CnMIZXIAHV1CHO2hAExgM4Rle4c0Rzovz7nwsWtecfOYE/sD5/AESTI2t</latexit>y2<latexit sha1_base64="O8S5RxaP6eILUA4vw8mfCr5xfSo=">AAAB6nicbVBNS8NAEJ3Ur1q/qh69LBbBU0lEqseiBz1WtB/QhrLZTtqlm03Y3Qgl9Cd48aCIV3+RN/+N2zYHrT4YeLw3w8y8IBFcG9f9cgorq2vrG8XN0tb2zu5eef+gpeNUMWyyWMSqE1CNgktsGm4EdhKFNAoEtoPx9cxvP6LSPJYPZpKgH9Gh5CFn1FjpftK/6ZcrbtWdg/wlXk4qkKPRL3/2BjFLI5SGCap113MT42dUGc4ETku9VGNC2ZgOsWuppBFqP5ufOiUnVhmQMFa2pCFz9edERiOtJ1FgOyNqRnrZm4n/ed3UhJd+xmWSGpRssShMBTExmf1NBlwhM2JiCWWK21sJG1FFmbHplGwI3vLLf0nrrOrVqrW780r9Ko+jCEdwDKfgwQXU4RYa0AQGQ3iCF3h1hPPsvDnvi9aCk88cwi84H98yII3C</latexit>yG
A trajectory 
 sampled by the RAG system, <latexit sha1_base64="RDMc7xLK1B6FRbZwiBtQ0Mhp0f8=">AAAB8nicbVDLSsNAFJ3UV62vqks3g0VwISHRWl0W3bisaB+QhjKZTtqhk0yYuRFK6Ge4caGIW7/GnX/jtM1CWw/McDjnXu69J0gE1+A431ZhZXVtfaO4Wdra3tndK+8ftLRMFWVNKoVUnYBoJnjMmsBBsE6iGIkCwdrB6Hbqt5+Y0lzGjzBOmB+RQcxDTgkYyes+4Kp9cWa+y1654tjODHiZuDmpoByNXvmr25c0jVgMVBCtPddJwM+IAk4Fm5S6qWYJoSMyYJ6hMYmY9rPZyhN8YpQ+DqUyLwY8U393ZCTSehwFpjIiMNSL3lT8z/NSCK/9jMdJCiym80FhKjBIPL0f97liFMTYEEIVN7tiOiSKUDAplUwI7uLJy6R1brs1u3ZfrdRv8jiK6Agdo1PkoitUR3eogZqIIome0St6s8B6sd6tj3lpwcp7DtEfWJ8/JbyPOg==</latexit>§4.3,4.5Approximate Plackett-LucePolicy Parameterization<latexit sha1_base64="Avf3+F77FEXp2sS7gwAED4PCEK0=">AAAB7XicbVBNSwMxEJ2tX7V+VT16CRbB07JbpHosevFY0X5Au5Rsmm1js8mSZIWy9D948aCIV/+PN/+NabsHbX0w8Hhvhpl5YcKZNp737RTW1jc2t4rbpZ3dvf2D8uFRS8tUEdokkkvVCbGmnAnaNMxw2kkUxXHIaTsc38z89hNVmknxYCYJDWI8FCxiBBsrtXr36MKt9ssVz/XmQKvEz0kFcjT65a/eQJI0psIQjrXu+l5iggwrwwin01Iv1TTBZIyHtGupwDHVQTa/dorOrDJAkVS2hEFz9fdEhmOtJ3FoO2NsRnrZm4n/ed3URFdBxkSSGirIYlGUcmQkmr2OBkxRYvjEEkwUs7ciMsIKE2MDKtkQ/OWXV0mr6vo1t3Z3Ualf53EU4QRO4Rx8uIQ63EIDmkDgEZ7hFd4c6bw4787HorXg5DPH8AfO5w8Kuo4k</latexit>§4.2<latexit sha1_base64="Avf3+F77FEXp2sS7gwAED4PCEK0=">AAAB7XicbVBNSwMxEJ2tX7V+VT16CRbB07JbpHosevFY0X5Au5Rsmm1js8mSZIWy9D948aCIV/+PN/+NabsHbX0w8Hhvhpl5YcKZNp737RTW1jc2t4rbpZ3dvf2D8uFRS8tUEdokkkvVCbGmnAnaNMxw2kkUxXHIaTsc38z89hNVmknxYCYJDWI8FCxiBBsrtXr36MKt9ssVz/XmQKvEz0kFcjT65a/eQJI0psIQjrXu+l5iggwrwwin01Iv1TTBZIyHtGupwDHVQTa/dorOrDJAkVS2hEFz9fdEhmOtJ3FoO2NsRnrZm4n/ed3URFdBxkSSGirIYlGUcmQkmr2OBkxRYvjEEkwUs7ciMsIKE2MDKtkQ/OWXV0mr6vo1t3Z3Ualf53EU4QRO4Rx8uIQ63EIDmkDgEZ7hFd4c6bw4787HorXg5DPH8AfO5w8Kuo4k</latexit>§4.2
Policy Optimization with GRPO<latexit sha1_base64="el4Dxci40nsSWtyslmUFxnSJ4FA=">AAAB7XicbVBNSwMxEJ2tX7V+VT16CRbB07IrpXosevFY0X5Au5Rsmm1js8mSZIWy9D948aCIV/+PN/+NabsHbX0w8Hhvhpl5YcKZNp737RTW1jc2t4rbpZ3dvf2D8uFRS8tUEdokkkvVCbGmnAnaNMxw2kkUxXHIaTsc38z89hNVmknxYCYJDWI8FCxiBBsrtXr3qOpW++WK53pzoFXi56QCORr98ldvIEkaU2EIx1p3fS8xQYaVYYTTaamXappgMsZD2rVU4JjqIJtfO0VnVhmgSCpbwqC5+nsiw7HWkzi0nTE2I73szcT/vG5qoqsgYyJJDRVksShKOTISzV5HA6YoMXxiCSaK2VsRGWGFibEBlWwI/vLLq6R14fo1t3ZXrdSv8ziKcAKncA4+XEIdbqEBTSDwCM/wCm+OdF6cd+dj0Vpw8plj+APn8wcNwo4m</latexit>§4.4<latexit sha1_base64="z+bNxocm6/4exWdkHjqZ6x7myiA=">AAAB7HicbVBNS8NAEJ3Ur1q/qh69LBbBU0lEqseiF48VTFtoQ9lsJ+3SzSbsboQS+hu8eFDEqz/Im//GbZuDVh8MPN6bYWZemAqujet+OaW19Y3NrfJ2ZWd3b/+genjU1kmmGPosEYnqhlSj4BJ9w43AbqqQxqHATji5nfudR1SaJ/LBTFMMYjqSPOKMGiv5fcYVG1Rrbt1dgPwlXkFqUKA1qH72hwnLYpSGCap1z3NTE+RUGc4Ezir9TGNK2YSOsGeppDHqIF8cOyNnVhmSKFG2pCEL9edETmOtp3FoO2NqxnrVm4v/eb3MRNdBzmWaGZRsuSjKBDEJmX9OhlwhM2JqCWWK21sJG1NFmbH5VGwI3urLf0n7ou416o37y1rzpoijDCdwCufgwRU04Q5a4AMDDk/wAq+OdJ6dN+d92Vpyiplj+AXn4xvPBY60</latexit>→ConcatAssign
<latexit sha1_base64="jMmJMvFhpWjT6F1yXVwCh8gQm+M=">AAAB8XicbVBNS8NAEJ34WetX1aOXxSJ4KolI9Vj04rGC/cA2lM120i7dbMLuRimh/8KLB0W8+m+8+W/ctjlo64OBx3szzMwLEsG1cd1vZ2V1bX1js7BV3N7Z3dsvHRw2dZwqhg0Wi1i1A6pRcIkNw43AdqKQRoHAVjC6mfqtR1Sax/LejBP0IzqQPOSMGis9dAWGhioVP/VKZbfizkCWiZeTMuSo90pf3X7M0gilYYJq3fHcxPgZVYYzgZNiN9WYUDaiA+xYKmmE2s9mF0/IqVX6JIyVLWnITP09kdFI63EU2M6ImqFe9Kbif14nNeGVn3GZpAYlmy8KU0FMTKbvkz5XyIwYW0KZ4vZWwoZUUWZsSEUbgrf48jJpnle8aqV6d1GuXedxFOAYTuAMPLiEGtxCHRrAQMIzvMKbo50X5935mLeuOPnMEfyB8/kD7pyRGw==</latexit>→<latexit sha1_base64="88zRa9pClgK8crxcT5+kkkd2Eu8=">AAAB83icbVBNS8NAEJ3Ur1q/qh69LBbBU0lEqseiF48V7Ac0oWy2m3bpJll2J0IJ/RtePCji1T/jzX/jts1Bqw8GHu/NMDMvVFIYdN0vp7S2vrG5Vd6u7Ozu7R9UD486Js00422WylT3Qmq4FAlvo0DJe0pzGoeSd8PJ7dzvPnJtRJo84FTxIKajRESCUbSS7ysxyH0cc6SzQbXm1t0FyF/iFaQGBVqD6qc/TFkW8wSZpMb0PVdhkFONgkk+q/iZ4YqyCR3xvqUJjbkJ8sXNM3JmlSGJUm0rQbJQf07kNDZmGoe2M6Y4NqveXPzP62cYXQe5SFSGPGHLRVEmCaZkHgAZCs0ZyqkllGlhbyVsTDVlaGOq2BC81Zf/ks5F3WvUG/eXteZNEUcZTuAUzsGDK2jCHbSgDQwUPMELvDqZ8+y8Oe/L1pJTzBzDLzgf33aCkfs=</latexit>ωω
<latexit sha1_base64="LubOjCANHfuoEjYTYYAID0nY0zI=">AAAB/HicbVBNS8NAEN34WetXtEcvwSJ4KCUpUr0IRS8eK9gPaELZbLbt0s0m7E6EkNa/4sWDIl79Id78N27bHLT1wcDjvRlm5vkxZwps+9tYW9/Y3Nou7BR39/YPDs2j47aKEkloi0Q8kl0fK8qZoC1gwGk3lhSHPqcdf3w78zuPVCoWiQdIY+qFeCjYgBEMWuqbJbh2KrWKy4MIVGXiAk4mfbNsV+05rFXi5KSMcjT75pcbRCQJqQDCsVI9x47By7AERjidFt1E0RiTMR7SnqYCh1R52fz4qXWmlcAaRFKXAGuu/p7IcKhUGvq6M8QwUsveTPzP6yUwuPIyJuIEqCCLRYOEWxBZsySsgElKgKeaYCKZvtUiIywxAZ1XUYfgLL+8Stq1qlOv1u8vyo2bPI4COkGn6Bw56BI10B1qohYiKEXP6BW9GU/Gi/FufCxa14x8poT+wPj8AarJlCw=</latexit>t=1,2,...,|ω|<latexit sha1_base64="DV8wDhD9qplfTwZsHx/AsGaU3Z8=">AAAB6nicbVDLTgJBEOzFF+IL9ehlIjHxRHbVoEeiF48Y5JHAhswOvTBhdnYzM2tCCJ/gxYPGePWLvPk3DrAHBSvppFLVne6uIBFcG9f9dnJr6xubW/ntws7u3v5B8fCoqeNUMWywWMSqHVCNgktsGG4EthOFNAoEtoLR3cxvPaHSPJaPZpygH9GB5CFn1Fip3q1f9oolt+zOQVaJl5ESZKj1il/dfszSCKVhgmrd8dzE+BOqDGcCp4VuqjGhbEQH2LFU0gi1P5mfOiVnVumTMFa2pCFz9ffEhEZaj6PAdkbUDPWyNxP/8zqpCW/8CZdJalCyxaIwFcTEZPY36XOFzIixJZQpbm8lbEgVZcamU7AheMsvr5LmRdmrlCsPV6XqbRZHHk7gFM7Bg2uowj3UoAEMBvAMr/DmCOfFeXc+Fq05J5s5hj9wPn8A1VeNhQ==</latexit>§3
HistoryQuerySubqueryPolicy Model(Embedding)Action(Documents)ObservationStateCorpus(Vector DB)LLMLLMOutput<latexit sha1_base64="jMmJMvFhpWjT6F1yXVwCh8gQm+M=">AAAB8XicbVBNS8NAEJ34WetX1aOXxSJ4KolI9Vj04rGC/cA2lM120i7dbMLuRimh/8KLB0W8+m+8+W/ctjlo64OBx3szzMwLEsG1cd1vZ2V1bX1js7BV3N7Z3dsvHRw2dZwqhg0Wi1i1A6pRcIkNw43AdqKQRoHAVjC6mfqtR1Sax/LejBP0IzqQPOSMGis9dAWGhioVP/VKZbfizkCWiZeTMuSo90pf3X7M0gilYYJq3fHcxPgZVYYzgZNiN9WYUDaiA+xYKmmE2s9mF0/IqVX6JIyVLWnITP09kdFI63EU2M6ImqFe9Kbif14nNeGVn3GZpAYlmy8KU0FMTKbvkz5XyIwYW0KZ4vZWwoZUUWZsSEUbgrf48jJpnle8aqV6d1GuXedxFOAYTuAMPLiEGtxCHRrAQMIzvMKbo50X5935mLeuOPnMEfyB8/kD7pyRGw==</latexit>→Needmoreinfo?YN
<latexit sha1_base64="z+bNxocm6/4exWdkHjqZ6x7myiA=">AAAB7HicbVBNS8NAEJ3Ur1q/qh69LBbBU0lEqseiF48VTFtoQ9lsJ+3SzSbsboQS+hu8eFDEqz/Im//GbZuDVh8MPN6bYWZemAqujet+OaW19Y3NrfJ2ZWd3b/+genjU1kmmGPosEYnqhlSj4BJ9w43AbqqQxqHATji5nfudR1SaJ/LBTFMMYjqSPOKMGiv5fcYVG1Rrbt1dgPwlXkFqUKA1qH72hwnLYpSGCap1z3NTE+RUGc4Ezir9TGNK2YSOsGeppDHqIF8cOyNnVhmSKFG2pCEL9edETmOtp3FoO2NqxnrVm4v/eb3MRNdBzmWaGZRsuSjKBDEJmX9OhlwhM2JqCWWK21sJG1NFmbH5VGwI3urLf0n7ou416o37y1rzpoijDCdwCufgwRU04Q5a4AMDDk/wAq+OdJ6dN+d92Vpyiplj+AXn4xvPBY60</latexit>→
<latexit sha1_base64="jMmJMvFhpWjT6F1yXVwCh8gQm+M=">AAAB8XicbVBNS8NAEJ34WetX1aOXxSJ4KolI9Vj04rGC/cA2lM120i7dbMLuRimh/8KLB0W8+m+8+W/ctjlo64OBx3szzMwLEsG1cd1vZ2V1bX1js7BV3N7Z3dsvHRw2dZwqhg0Wi1i1A6pRcIkNw43AdqKQRoHAVjC6mfqtR1Sax/LejBP0IzqQPOSMGis9dAWGhioVP/VKZbfizkCWiZeTMuSo90pf3X7M0gilYYJq3fHcxPgZVYYzgZNiN9WYUDaiA+xYKmmE2s9mF0/IqVX6JIyVLWnITP09kdFI63EU2M6ImqFe9Kbif14nNeGVn3GZpAYlmy8KU0FMTKbvkz5XyIwYW0KZ4vZWwoZUUWZsSEUbgrf48jJpnle8aqV6d1GuXedxFOAYTuAMPLiEGtxCHRrAQMIzvMKbo50X5935mLeuOPnMEfyB8/kD7pyRGw==</latexit>→
<latexit sha1_base64="oujthz8UgE7EGJO31qe+85VgYKU=">AAAB6nicbVBNS8NAEJ34WetX1aOXxSJ4KolI9Vj04rGi/YA2lM120y7dbOLuRCihP8GLB0W8+ou8+W/ctjlo64OBx3szzMwLEikMuu63s7K6tr6xWdgqbu/s7u2XDg6bJk414w0Wy1i3A2q4FIo3UKDk7URzGgWSt4LRzdRvPXFtRKwecJxwP6IDJULBKFrp/rGHvVLZrbgzkGXi5aQMOeq90le3H7M04gqZpMZ0PDdBP6MaBZN8UuymhieUjeiAdyxVNOLGz2anTsipVfokjLUthWSm/p7IaGTMOApsZ0RxaBa9qfif10kxvPIzoZIUuWLzRWEqCcZk+jfpC80ZyrEllGlhbyVsSDVlaNMp2hC8xZeXSfO84lUr1buLcu06j6MAx3ACZ+DBJdTgFurQAAYDeIZXeHOk8+K8Ox/z1hUnnzmCP3A+fwBqJI3n</latexit>qt<latexit sha1_base64="ruAh2xGT1lDiL33hbQyUfkS0D08=">AAAB6nicbVBNS8NAEJ3Ur1q/qh69LBbBU0lEqseiF48V7Qe0oWy2m3bpZhN2J0IJ/QlePCji1V/kzX/jts1BWx8MPN6bYWZekEhh0HW/ncLa+sbmVnG7tLO7t39QPjxqmTjVjDdZLGPdCajhUijeRIGSdxLNaRRI3g7GtzO//cS1EbF6xEnC/YgOlQgFo2ilB9PHfrniVt05yCrxclKBHI1++as3iFkacYVMUmO6npugn1GNgkk+LfVSwxPKxnTIu5YqGnHjZ/NTp+TMKgMSxtqWQjJXf09kNDJmEgW2M6I4MsveTPzP66YYXvuZUEmKXLHFojCVBGMy+5sMhOYM5cQSyrSwtxI2opoytOmUbAje8surpHVR9WrV2v1lpX6Tx1GEEziFc/DgCupwBw1oAoMhPMMrvDnSeXHenY9Fa8HJZ47hD5zPH20wjek=</latexit>st<latexit sha1_base64="Aqaf4kZZ4oZw9yMa1yfljHA69HY=">AAAB6nicbVBNS8NAEJ34WetX1aOXxSJ4KolI9Vj04rGi/YA2lM120i7dbOLuRiihP8GLB0W8+ou8+W/ctjlo64OBx3szzMwLEsG1cd1vZ2V1bX1js7BV3N7Z3dsvHRw2dZwqhg0Wi1i1A6pRcIkNw43AdqKQRoHAVjC6mfqtJ1Sax/LBjBP0IzqQPOSMGivdP/bcXqnsVtwZyDLxclKGHPVe6avbj1kaoTRMUK07npsYP6PKcCZwUuymGhPKRnSAHUsljVD72ezUCTm1Sp+EsbIlDZmpvycyGmk9jgLbGVEz1IveVPzP66QmvPIzLpPUoGTzRWEqiInJ9G/S5wqZEWNLKFPc3krYkCrKjE2naEPwFl9eJs3ziletVO8uyrXrPI4CHMMJnIEHl1CDW6hDAxgM4Ble4c0Rzovz7nzMW1ecfOYI/sD5/AEDFI2j</latexit>q0<latexit sha1_base64="VoTl7i9kb2YKHim3MpApKx8+shI=">AAAB9HicbVDLSsNAFL2pr1pfVZdugkVwVRKR6rLopssK9gFtKJPppB06mcSZm0IJ/Q43LhRx68e482+ctFlo64GBwzn3cs8cPxZco+N8W4WNza3tneJuaW//4PCofHzS1lGiKGvRSESq6xPNBJeshRwF68aKkdAXrONP7jO/M2VK80g+4ixmXkhGkgecEjSS1w8JjikRaWM+wEG54lSdBex14uakAjmag/JXfxjRJGQSqSBa91wnRi8lCjkVbF7qJ5rFhE7IiPUMlSRk2ksXoef2hVGGdhAp8yTaC/X3RkpCrWehbyazkHrVy8T/vF6Cwa2XchknyCRdHgoSYWNkZw3YQ64YRTEzhFDFTVabjokiFE1PJVOCu/rlddK+qrq1au3hulK/y+sowhmcwyW4cAN1aEATWkDhCZ7hFd6sqfVivVsfy9GCle+cwh9Ynz8RqpJQ</latexit>Ht
<latexit sha1_base64="Wc+SuwGu1gWPWB5JAP5olpWFkag=">AAAB6HicbVBNS8NAEJ34WetX1aOXxSJ4KolI9Vj04rEF+wFtKJvtpF272YTdjRBKf4EXD4p49Sd589+4bXPQ1gcDj/dmmJkXJIJr47rfztr6xubWdmGnuLu3f3BYOjpu6ThVDJssFrHqBFSj4BKbhhuBnUQhjQKB7WB8N/PbT6g0j+WDyRL0IzqUPOSMGis1sn6p7FbcOcgq8XJShhz1fumrN4hZGqE0TFCtu56bGH9CleFM4LTYSzUmlI3pELuWShqh9ifzQ6fk3CoDEsbKljRkrv6emNBI6ywKbGdEzUgvezPxP6+bmvDGn3CZpAYlWywKU0FMTGZfkwFXyIzILKFMcXsrYSOqKDM2m6INwVt+eZW0LitetVJtXJVrt3kcBTiFM7gAD66hBvdQhyYwQHiGV3hzHp0X5935WLSuOfnMCfyB8/kD60+NCA==</latexit>y<latexit sha1_base64="kjI7zWasSv1URELa0MIjBktCZQc=">AAAB6nicbVBNS8NAEJ3Ur1q/qh69LBbBU0lEqseiF48V7Qe0oWy2m3bpZhN2J0IJ/QlePCji1V/kzX/jts1BWx8MPN6bYWZekEhh0HW/ncLa+sbmVnG7tLO7t39QPjxqmTjVjDdZLGPdCajhUijeRIGSdxLNaRRI3g7GtzO//cS1EbF6xEnC/YgOlQgFo2ilB9rHfrniVt05yCrxclKBHI1++as3iFkacYVMUmO6npugn1GNgkk+LfVSwxPKxnTIu5YqGnHjZ/NTp+TMKgMSxtqWQjJXf09kNDJmEgW2M6I4MsveTPzP66YYXvuZUEmKXLHFojCVBGMy+5sMhOYM5cQSyrSwtxI2opoytOmUbAje8surpHVR9WrV2v1lpX6Tx1GEEziFc/DgCupwBw1oAoMhPMMrvDnSeXHenY9Fa8HJZ47hD5zPH1HEjdc=</latexit>at<latexit sha1_base64="0eUCdCqI+Uq6dMIWcIwM2AP77dU=">AAAB6nicbVBNS8NAEJ3Ur1q/qh69BIvgqSQi1WPRi8eK9gPaUDbbTbt0sxt2J0IJ/QlePCji1V/kzX/jts1BWx8MPN6bYWZemAhu0PO+ncLa+sbmVnG7tLO7t39QPjxqGZVqyppUCaU7ITFMcMmayFGwTqIZiUPB2uH4dua3n5g2XMlHnCQsiMlQ8ohTglZ6UH3slyte1ZvDXSV+TiqQo9Evf/UGiqYxk0gFMabrewkGGdHIqWDTUi81LCF0TIasa6kkMTNBNj916p5ZZeBGStuS6M7V3xMZiY2ZxKHtjAmOzLI3E//zuilG10HGZZIik3SxKEqFi8qd/e0OuGYUxcQSQjW3t7p0RDShaNMp2RD85ZdXSeui6teqtfvLSv0mj6MIJ3AK5+DDFdThDhrQBApDeIZXeHOE8+K8Ox+L1oKTzxzDHzifP2cYjeU=</latexit>otHistory
<latexit sha1_base64="AhHpHkdMDY+LsQve5nlWiEE1vXI=">AAAB+nicbVDLSsNAFL2pr1pfqS7dDBZBEEoiUl0W3XRZwT6gDWEynbRDJw9mJkqJ+RQ3LhRx65e482+ctFlo64GBwzn3cs8cL+ZMKsv6Nkpr6xubW+Xtys7u3v6BWT3syigRhHZIxCPR97CknIW0o5jitB8LigOP0543vc393gMVkkXhvZrF1AnwOGQ+I1hpyTWrwwCrCcE8bWVuqs7tzDVrVt2aA60SuyA1KNB2za/hKCJJQENFOJZyYFuxclIsFCOcZpVhImmMyRSP6UDTEAdUOuk8eoZOtTJCfiT0CxWaq783UhxIOQs8PZkHlcteLv7nDRLlXzspC+NE0ZAsDvkJRypCeQ9oxAQlis80wUQwnRWRCRaYKN1WRZdgL395lXQv6naj3ri7rDVvijrKcAwncAY2XEETWtCGDhB4hGd4hTfjyXgx3o2PxWjJKHaO4A+Mzx84OZP9</latexit>Ht+1
<latexit sha1_base64="jMmJMvFhpWjT6F1yXVwCh8gQm+M=">AAAB8XicbVBNS8NAEJ34WetX1aOXxSJ4KolI9Vj04rGC/cA2lM120i7dbMLuRimh/8KLB0W8+m+8+W/ctjlo64OBx3szzMwLEsG1cd1vZ2V1bX1js7BV3N7Z3dsvHRw2dZwqhg0Wi1i1A6pRcIkNw43AdqKQRoHAVjC6mfqtR1Sax/LejBP0IzqQPOSMGis9dAWGhioVP/VKZbfizkCWiZeTMuSo90pf3X7M0gilYYJq3fHcxPgZVYYzgZNiN9WYUDaiA+xYKmmE2s9mF0/IqVX6JIyVLWnITP09kdFI63EU2M6ImqFe9Kbif14nNeGVn3GZpAYlmy8KU0FMTKbvkz5XyIwYW0KZ4vZWwoZUUWZsSEUbgrf48jJpnle8aqV6d1GuXedxFOAYTuAMPLiEGtxCHRrAQMIzvMKbo50X5935mLeuOPnMEfyB8/kD7pyRGw==</latexit>→
Figure 1.Overview of HARR
suffers from state aliasing. In multi-hop RAG, identical
sub-queries qtmay arise from different retrieval histories
Ht−1, yet be mapped to the same state under a query-only
representation. Consequently, executing the same retrieval
action in this apparent state can lead to different downstream
reasoning trajectories and final rewards, yielding inconsis-
tent learning signals for the same state–action pair. This
leads to high-variance policy gradient estimates and hinders
effective learning.
To mitigate the state aliasing issue, we define each state
st∈ Sat steptas a history-aware representation
st= (H t−1, qt), (1)
where Ht−1denotes the retrieval history up to step t−1 .
By incorporating the retrieval history into the state, retrieval
decisions made under identical sub-queries but different his-
tories are distinguished. As a result, state-action pairs that
would otherwise be aliased under a query-only state formu-
lation are separated, leading to more consistent downstream
outcomes and learning signals for each state–action pair.
Action Space A.At each step t, the action is to select a
ranked list of kdocuments from the corpus D, denoted at=
Dt= (d1
t, . . . , dk
t). Thus, the action space Acomprises all
such orderedk-documents, with|A|=|D|!
(|D|−k)!.
State Transition P.Given state st= (H t−1, qt)and ac-
tionat=D t, the state transition is induced by the LLM.
Conditioned on the current sub-query and retrieved docu-
ments, the LLM produces an intermediate observation
ot∼π LLM(· |q t, Dt), (2)
which summarizes the retrieved evidence. The retrieval
history is then updated as Ht=H t−1◦(qt, ot), and the
LLM emits the next sub-query
qt+1∼π LLM(· | H t). (3)The resulting next state iss t+1= (H t, qt+1).
Under this formulation, the state transition probability can
be expressed as
P(st+1|st, at) =X
otP(st+1, ot|st, at)
=X
otP(ot|st, at)·P(s t+1|st, at, ot)
=X
otπLLM(ot|qt, Dt)| {z }
Observation Generation·πLLM(qt+1| Ht)|{z }
Sub-query Generation.(4)
Reward r.To optimize the retriever for final question
answering performance, we use a sparse terminal reward:
rt=(
F1(y, y∗), t=T
0, t < T,(5)
where yis the answer generated by conditioning the LLM
on the full retrieval history HT, and y∗is the ground-truth
answer. The F1 score is computed as
F1(y, y∗) =2|T(y)∩ T(y∗)|
|T(y)|+|T(y∗)|, (6)
whereT(·)denotes the set of tokens in a given text.
4.3. Policy Parametrization
We parameterize the retriever as a stochastic policy πθthat
maps a state stto a probability distribution over ordered
lists of kdocuments Dt, from which an action at=D tis
sampled at each retrieval step.
To model this distribution, we view retrieval as sequential
sampling without replacement from the corpus, which is
modeled using the Plackett-Luce ranking model (Plackett,
1975). Under this model, the probability of selecting an
ordered document list Dt= (d1
t, . . . , dk
t)factorizes into a
4

Reinforcement Fine-Tuning for History-Aware Dense Retriever in RAG
product of conditional selection probabilities
πθ(at=D t|st) =kY
i=1πθ 
di
t|st, d1:i−1
t
. (7)
At each position i, the next document is sampled from the
remaining candidate set D \ {d1
t, . . . , di−1
t}according to a
softmax distribution
πθ 
di
t|st, d1:i−1
t
=exp 
fθ(st, di
t)/α
P
d∈D\{dj
t}i−1
j=1exp(f θ(st, d)/α),(8)
where α >0 is a temperature parameter. The scoring
function fθ(s, d) represents the policy score of document d
under statesand is defined as
fθ(s, d) = sim(enc θ(s),enc θ(d)), (9)
where encθ(·)is a text encoder parameterized by θ, and
sim(·,·)denotes a similarity function (e.g., inner product).
4.4. Policy Optimization
Given the MDP formulation and the policy parametrization,
our objective is to optimize the retriever policy to maximize
the expected terminal reward,
max
θEτ∼πθ
r|τ|
, (10)
where τ= (s 1, a1, . . . , s |τ|, a|τ|)denotes a retrieval trajec-
tory induced by the policy, and r|τ|is the terminal reward
defined in § 4.2.
In our formulation, rewards are sparse and only observed
at the end of a trajectory, while the state space is history-
dependent and high-dimensional. Estimating accurate value
functions in this setting is challenging and introduces addi-
tional model complexity. We therefore adopt Group Relative
Policy Optimization (GRPO) (Shao et al., 2024), which op-
timizes the policy using relative outcome-based advantages
without explicit value function estimation.
Specifically, for a given query q0∼P(Q) , we sample a
group of Gretrieval trajectories {τi}G
i=1from the old policy
πθold. Each trajectory τireceives a terminal reward ri, which
is normalized within the group to obtain the advantage
Ai=ri−mean(r)
std(r),r={r 1, . . . , r G}. (11)
The retriever policy is then optimized by maximizing the
GRPO objective
JGRPO(θ) =E"
1
GGX
i=11
|τi||τi|X
t=1min 
πθ(ai,t|si,t)
πθold(ai,t|si,t)Ai,
clipπθ(ai,t|si,t)
πθold(ai,t|si,t),1−ϵ,1 +ϵ
Ai!#
,(12)
whereϵis the clipping parameter.4.5. Practical Training Approximations
The GRPO objective optimizes the retriever policy over the
full document corpus. However, computing the policy prob-
abilities in (12) is prohibitively expensive for large-scale
corpora, as it requires normalization over the entire corpus
at each retrieval step and recomputation of document em-
beddings after each policy update. We therefore introduce
the following approximations to enable efficient training.
Candidate Pool Restriction.At each retrieval step t,
instead of normalizing over the full corpus D, we first
construct a candidate pool Ct⊂ D by retrieving the top-
Kdocuments using the scoring function in (9), where
k < K≪ |D|. Policy sampling and probability normaliza-
tion are then restricted to this candidate pool. Under this
approximation, the conditional selection probability in (8)
is given by
πθ 
di
t|st, d1:i−1
t
≈exp 
fθ(st, di
t)/α
P
d∈Ct\{dj
t}i−1
j=1exp(f θ(st, d)/α).(13)
Document Encoder Freezing.To further reduce compu-
tational overhead, we freeze the document encoder during
training and only update the state encoder. The scoring
function in (9) then takes the form
fθ(s, d) = sim(enc θ(s),enc(d)),(14)
where document embeddings enc(d) are precomputed and
reused across policy updates, while the retriever adapts
through learned state representations.
5. Experiments
The experiments aim at demonstrating: (1) HARR improves
end-to-end question answering performance across different
RAG pipelines, datasets, and retriever encoders (§5.2); (2)
the proposed RL-based fine-tuning and history-aware state
representation both contribute positively to performance
gains (§5.3); and (3) HARR achieves favorable training
efficiency in terms of convergence speed, training time, and
memory usage (§5.4).
5.1. Experimental Setup
We position HARR as aplug-inretriever optimization
module compatible with existing RAG systems. To en-
able a fair assessment, we fix the RAG pipeline configu-
rations—including LLMs, prompts, and interaction work-
flows—and evaluate performance gains solely by varying
the retriever optimization strategy.
RAG Pipelines.We evaluate our method on two repre-
sentative RAG pipelines to demonstrate its versatility. (1)
5

Reinforcement Fine-Tuning for History-Aware Dense Retriever in RAG
ReAct Agent(Yao et al., 2023b): a ReAct-style pipeline
using Qwen3-8B1as the LLM, where the retriever is in-
voked as an external tool. (2)Search-R1(Jin et al., 2025):
a pipeline with an RL-fine-tuned Qwen2.5-7B2LLM that
interleaves reasoning and retrieval, with the retriever serving
as its search engine.
Baselines.We compare against the following baselines
under the same fixed RAG pipelines. (1)Frozen Retriever:
the pre-trained retriever is used directly without adaptation,
serving as a lower-bound reference. (2)REPLUG: a strong
baseline based on the LSR method from REPLUG (Shi
et al., 2024), which fine-tunes the retriever by matching its
retrieval score distribution to the LLM’s perplexity distribu-
tion over the top-kdocuments.
Datasets & Evaluation.To provide a comprehensive
evaluation, we conduct experiments on both multi-hop
and single-hop question answering benchmarks, using
HotpotQA (Yang et al., 2018) and Natural Questions
(NQ) (Kwiatkowski et al., 2019), respectively. Following
prior work (Guan et al., 2025), we report SQuAD (Rajpurkar
et al., 2016) Exact Match (EM) and F1 scores as evaluation
metrics.
Implementation.All experiments are conducted on a
single node equipped with 8 NVIDIA A100 GPUs. Our
method is implemented using the TRL (von Werra et al.,
2020) and LlamaIndex (Liu, 2022) libraries, while RE-
PLUG follows the FedRAG (Fajardo & Emerson, 2025)
implementation, with all LLMs deployed via vLLM (Kwon
et al., 2023). Unless otherwise specified, we employ
the 2018 Wikipedia dump (Karpukhin et al., 2020) as
the retrieval corpus, indexed in Qdrant (Qdrant Team,
2023) using an INT8-quantized HNSW index ( m= 16 ,
efconstruct= 100 ). We use Qwen3-Embedding-4B3
and -0.6B4as retriever encoders, fine-tuned via LoRA (Hu
et al., 2022) ( r= 32 ,α= 64 , dropout = 0.05 ) using
the AdamW (Loshchilov & Hutter, 2017) optimizer with
FP16 precision, a constant learning rate of 1×10−5without
warmup, a batch size of 16, and 100 training steps. For our
method, we filter the training set to ensure non-zero reward
variance under the initial policy across 8 rollouts and set the
training retrieval top- kto 3, with a temperature α= 0.05 ,
a candidate pool size K= 30 , a GRPO group size G= 8 ,
1Adopt checkpoint from https://huggingface.co/
Qwen/Qwen3-8B.
2Adopt checkpoint from https://huggingface.co/
PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.
5-7b-em-ppo.
3Adopt checkpoint from https://huggingface.co/
Qwen/Qwen3-Embedding-4B.
4Adopt checkpoint from https://huggingface.co/
Qwen/Qwen3-Embedding-0.6B.and a clipping ratio ϵ= 0.2 (no KL regularization). At
inference time, all methods use k= 3 and are evaluated on
the full, unfiltered development splits following the original
dataset partitions for a fair comparison.
5.2. Comparison of Accuracy
Table 1 reports the end-to-end question answering perfor-
mance across different RAG pipelines, datasets, and re-
triever encoders, while Figure 2 depicts the corresponding
training reward trajectories on HotpotQA. We analyze the
results from four complementary perspectives.
Comparison with Frozen Retrievers.From Table 1, our
approach improves performance in 18 out of 20 reported
metrics compared to the Frozen Retriever baseline, demon-
strating consistent gains across diverse experimental settings.
We attribute these improvements to end-to-end reinforce-
ment learning, which aligns the retriever with the LLM and
enables more effective coordination between retrieval and
reasoning components.
Superiority over Supervised Fine-Tuning.From Table 1,
our method also outperforms REPLUG in 17 out of 20
metrics. While REPLUG optimizes proxy objectives that
approximate LLM preferences, such objectives do not neces-
sarily translate into improved end-to-end question answering
performance. In contrast, our approach directly optimizes
the downstream task reward, reducing the mismatch be-
tween training objectives and inference-time behavior and
allowing the retriever to learn document selection strategies
that better support the overall reasoning pipeline.
Training Dynamics.Figure 2 illustrates the training re-
ward trajectories of HARR. We observe an initial decrease
in reward followed by a steady increase. This early decline
is likely caused by a distribution shift from the history-aware
state representation, which differs from the retriever’s pre-
training regime. After an adaptation phase, rewards increase
consistently, indicating that the retriever learns to exploit
historical context to improve retrieval decisions.
Analysis of Performance Variations.Despite the over-
all gains, several trends are worth noting. First, reward
trajectories exhibit oscillations during training (Figure 2),
which can be attributed to sparse terminal rewards and the
off-policy approximations introduced for scalability (see
§4.5). Second, improvements are more pronounced on the
multi-hop HotpotQA dataset than on the single-hop NQ
benchmark, and when using the larger 4B retriever encoder
compared to the 0.6B model. These observations suggest
that history-aware state representations are most beneficial
when retrieval decisions depend on longer reasoning con-
texts, and that sufficient model capacity is required to effec-
6

Reinforcement Fine-Tuning for History-Aware Dense Retriever in RAG
Table 1.End-to-end QA performance (EM/F1; higher is better) across two RAG agents (ReAct vs. Original), two datasets (HotpotQA,
NQ), and two retriever encoders (Qwen3-Embedding-4B, Qwen3-Embedding-0.6B). For each base agent, we compare the frozen-retriever
baseline (ReAct Agent / Search-R1), a plug-in retrieval variant (REPLUG), and our method HARR applied within the same agent pipeline.
Bolddenotes the best result in each column, and underline denotes the second-best.Gainsreports the relative improvement (%) of HARR
over the strongest non-HARR baseline under the same base agent for that column.
Base Agent ApproachQwen3-Embedding-4B Qwen3-Embedding-0.6BAverage
HotpotQA NQ HotpotQA NQ
EM F1 EM F1 EM F1 EM F1 EM F1
ReActOriginal 31.28 40.42 32.82 43.86 28.74 37.47 30.10 40.46 30.73 40.55
REPLUG 31.91 40.91 32.60 43.77 28.53 37.04 30.12 40.58 30.79 40.57
HARR(Ours)32.34 41.55 33.46 44.56 29.60 38.14 30.16 40.72 31.39 41.24
Gains+1.35% +1.56% +1.95% +1.60% +3.01% +1.76% +0.11% +0.36% +1.61% +1.32%
Search-R1Original 40.04 50.48 43.91 53.61 37.88 47.81 40.92 50.08 40.69 50.50
REPLUG40.3950.90 43.95 53.52 37.66 47.6741.12 50.2740.78 50.59
HARR(Ours) 40.30 51.02 44.92 54.45 38.34 48.6340.87 50.0741.11 51.04
Gains-0.23% +0.24% +2.21% +1.57% +1.21% +1.72% -0.61% -0.40% +0.64% +0.78%
0 25 50 75 100
Step0.400.450.50 Train Reward
1
(a)React Agent 4B
0 25 50 75 100
Step0.550.600.65 Train Reward
1 (b)Search-R1 4B
0 25 50 75 100
Step0.40.5 Train Reward
1 (c)React Agent 0.6B
0 25 50 75 100
Step0.450.500.55 Train Reward
1 (d)Search-R1 0.6B
Figure 2.Training reward trajectories on the HotpotQA dataset. Subplot titles denote the specific RAG pipeline and retriever encoder
used. All curves are smoothed using Exponential Moving Average (EMA) with a window size of 8.
0 50 100
Step024 Grad Norm
1
(a)4B
0 50 100
Step024 Grad Norm
1 (b)0.6B
HARR w/o History
1
Figure 3.Gradient norm trajectories on HotpotQA using the ReAct
Agent pipeline. Subplot titles denote the retriever encoder size.
tively leverage such contextual information.
5.3. Ablation Study
We conduct ablation studies on the ReAct Agent pipeline
to analyze the contributions of the proposed history-aware
state representation and the RL-based retriever fine-tuning
within HARR. Table 2 reports the results across two datasets
and two retriever encoder sizes, while Figure 3 illustratesthe corresponding gradient norm trajectories.
Effect of RL and History-Aware State.Table 2 demon-
strates that both proposed components are essential for
achieving optimal performance. Specifically, removing the
history-aware state degrades performance in 9 out of 10
metrics, while excluding the RL-based optimization leads to
drops across all 10 metrics. The notably larger degradation
caused by removing RL identifies it as the primary driver of
performance gains, while the history-aware state serves as a
critical complementary module.
RL versus History-Aware State in Isolation.We further
analyze the behavior of each component in isolation. When
employing the history-aware state without RL-based fine-
tuning, the average performance fails to surpass that of
the frozen retriever. We attribute this to the distribution
shift introduced by the augmented state, which the pre-
trained encoder cannot effectively leverage without end-
to-end adaptation. Conversely, applying RL without the
history-aware state still yields improvements over the frozen
baseline, indicating that our RL framework is robust and
capable of enhancing performance even in the presence of
7

Reinforcement Fine-Tuning for History-Aware Dense Retriever in RAG
Table 2.Ablation study on the ReAct Agent pipeline across two datasets and two retriever encoders. “w/o History” and “w/o RL” refer to
removing the history-aware state space and the RL-based retriever fine-tuning from HARR, respectively. Notations follow Table 1.
Approach based on
ReAct AgentQwen3-Embedding-4B Qwen3-Embedding-0.6B
Average
HotpotQA NQ HotpotQA NQ
EM F1 EM F1 EM F1 EM F1 EM F1
Original 31.28 40.42 32.82 43.86 28.74 37.47 30.10 40.46 30.73 40.55
HARR w/o History 31.56 40.74 32.68 43.86 29.37 37.99 30.1940.48 30.95 40.77
HARR w/o RL 31.42 40.63 32.98 44.14 27.52 35.74 29.80 40.20 30.43 40.18
HARR32.34 41.55 33.46 44.56 29.60 38.1430.16 40.72 31.39 41.24
state aliasing.
Impact of Model Capacity.The relative contribution of
each component varies with retriever capacity. For the
smaller 0.6B encoder, adding RL-based fine-tuning on top
of the frozen retriever yields larger marginal gains. This
suggests that smaller models may initially suffer from a
larger gap in matching LLM preferences, thereby benefiting
more from the RL-driven optimization. In contrast, for the
larger 4B encoder, introducing the history-aware state on top
of RL leads to more pronounced improvements, indicating
that higher-capacity models are better equipped to exploit
historical context for informed retrieval decisions.
Gradient Analysis.Finally, Figure 3 provides insight into
the optimization dynamics. Compared to the variant trained
without history, HARR exhibits consistently larger gradient
norms throughout training. This observation is consistent
with our hypothesis that the history-aware state mitigates
state aliasing, thereby reducing reward ambiguity and yield-
ing stronger learning signals for policy optimization.
5.4. Training Efficiency
Since HARR exclusively fine-tunes the retriever encoder
while keeping the LLM and inference workflow unchanged,
it introduces no additional inference overhead; therefore,
we focus on training efficiency in terms of convergence
speed, wall-clock training time, and memory usage. As
shown by the training reward curves in Figure 2, RL-based
retriever optimization converges rapidly, typically within
100 training steps across different settings. Consistent with
this observation, Table 3 shows that the total training time of
a single experiment is at most around three hours, which is
substantially lower than the cost of fine-tuning LLMs, often
reported to require over ten hours even with comparable
hardware (Sun et al., 2025b). In terms of memory consump-
tion, training the 0.6B retriever requires less than 32 GB
GPU memory, making it feasible on consumer-grade GPUs.
While the 4B retriever incurs higher memory usage under
our default configuration, this is mainly due to the absenceTable 3.Training efficiency analysis. We report the peak GPU
memory usage (Mem) and total training time (Time) for a sin-
gle experiment run across varying RAG pipelines, datasets, and
retriever encoders. All models are fine-tuned with DP=1, TP=1,
gradient checkpointing enabled, and no gradient accumulation.
RAG PipelineQwen3-Embedding-4B
HotpotQA NQ
Mem/GB Time/h Mem/GB Time/h
ReAct Agent 61.70 3.10 31.42 3.14
Search-R1 65.82 1.36 41.72 1.12
RAG PipelineQwen3-Embedding-0.6B
HotpotQA NQ
Mem/GB Time/h Mem/GB Time/h
ReAct Agent 26.84 2.40 17.70 2.00
Search-R1 31.48 2.42 16.47 1.51
of aggressive memory optimization; in practice, techniques
such as gradient accumulation can be applied to reduce peak
memory footprint without significantly increasing training
time. Overall, these results demonstrate that HARR offers
a favorable efficiency–effectiveness trade-off for retriever
optimization in RAG systems.
6. Conclusion
This work targets the core mismatch between supervised re-
triever objectives and end-to-end QA in RAG, and proposes
HARR: a history-aware RL framework that fine-tunes dense
retrievers by (i) replacing deterministic top- kwith stochas-
tic sampling to expose action distributions and (ii) encoding
retrieval history in the state to reduce aliasing in multi-hop
reasoning. Across datasets, RAG pipelines, and retriever
scales, HARR consistently improves end-to-end RAG per-
formance, positioning RL as a retriever-centric alternative
that optimizes retrieval behavior directly for downstream
task reward rather than proxy relevance signals.
8

Reinforcement Fine-Tuning for History-Aware Dense Retriever in RAG
Impact Statement
This paper presents work whose goal is to advance the field
of Machine Learning. There are many potential societal
consequences of our work, none which we feel must be
specifically highlighted here.
References
Asai, A., Wu, Z., Wang, Y ., Sil, A., and Hajishirzi, H. Self-
RAG: Learning to retrieve, generate, and critique through
self-reflection. InThe Twelfth International Conference
on Learning Representations, 2024. URL https://
openreview.net/forum?id=hSyW5go0v8.
Besta, M., Blach, N., Kubicek, A., Gerstenberger, R., Pod-
stawski, M., Gianinazzi, L., Gajda, J., Lehmann, T.,
Niewiadomski, H., Nyczyk, P., et al. Graph of thoughts:
Solving elaborate problems with large language models.
InProceedings of the AAAI conference on artificial intel-
ligence, volume 38, pp. 17682–17690, 2024.
Fajardo, A. and Emerson, D. fed-rag, March 2025. URL
https://github.com/VectorInstitute/
fed-rag.
Fan, W., Ding, Y ., Ning, L., Wang, S., Li, H., Yin, D.,
Chua, T.-S., and Li, Q. A survey on rag meeting llms:
Towards retrieval-augmented large language models. In
Proceedings of the 30th ACM SIGKDD conference on
knowledge discovery and data mining, pp. 6491–6501,
2024.
Guan, X., Zeng, J., Meng, F., Xin, C., Lu, Y ., Lin, H., Han,
X., Sun, L., and Zhou, J. Deeprag: Thinking to retrieve
step by step for large language models.arXiv preprint
arXiv:2502.01142, 2025.
Guu, K., Lee, K., Tung, Z., Pasupat, P., and Chang, M.
Retrieval augmented language model pre-training. In
International conference on machine learning, pp. 3929–
3938. PMLR, 2020.
Hu, E. J., yelong shen, Wallis, P., Allen-Zhu, Z., Li, Y .,
Wang, S., Wang, L., and Chen, W. LoRA: Low-rank adap-
tation of large language models. InInternational Confer-
ence on Learning Representations, 2022. URL https:
//openreview.net/forum?id=nZeVKeeFYf9.
Izacard, G. and Grave, E. Leveraging passage retrieval with
generative models for open domain question answering.
InProceedings of the 16th conference of the european
chapter of the association for computational linguistics:
main volume, pp. 874–880, 2021.
Jiang, P., Lin, J., Cao, L., Tian, R., Kang, S., Wang, Z.,
Sun, J., and Han, J. Deepretrieval: Hacking real searchengines and retrievers with large language models via
reinforcement learning.arXiv preprint arXiv:2503.00223,
2025.
Jiang, Z., Xu, F. F., Gao, L., Sun, Z., Liu, Q., Dwivedi-
Yu, J., Yang, Y ., Callan, J., and Neubig, G. Active re-
trieval augmented generation. InProceedings of the 2023
Conference on Empirical Methods in Natural Language
Processing, pp. 7969–7992, 2023.
Jin, B., Zeng, H., Yue, Z., Yoon, J., Arik, S., Wang, D.,
Zamani, H., and Han, J. Search-r1: Training llms to
reason and leverage search engines with reinforcement
learning.arXiv preprint arXiv:2503.09516, 2025.
Karpukhin, V ., Oguz, B., Min, S., Lewis, P. S., Wu, L.,
Edunov, S., Chen, D., and Yih, W.-t. Dense passage
retrieval for open-domain question answering. InEMNLP
(1), pp. 6769–6781, 2020.
Ke, Z., Kong, W., Li, C., Zhang, M., Mei, Q., and Bendersky,
M. Bridging the preference gap between retrievers and
llms. InProceedings of the 62nd Annual Meeting of
the Association for Computational Linguistics (Volume 1:
Long Papers), pp. 10438–10451, 2024.
Kulkarni, M., Tangarajan, P., Kim, K., and Trivedi, A. Re-
inforcement learning for optimizing rag for domain chat-
bots.arXiv preprint arXiv:2401.06800, 2024.
Kwiatkowski, T., Palomaki, J., Redfield, O., Collins, M.,
Parikh, A., Alberti, C., Epstein, D., Polosukhin, I., Devlin,
J., Lee, K., et al. Natural questions: a benchmark for ques-
tion answering research.Transactions of the Association
for Computational Linguistics, 7:453–466, 2019.
Kwon, W., Li, Z., Zhuang, S., Sheng, Y ., Zheng, L., Yu,
C. H., Gonzalez, J. E., Zhang, H., and Stoica, I. Efficient
memory management for large language model serving
with pagedattention. InProceedings of the ACM SIGOPS
29th Symposium on Operating Systems Principles, 2023.
Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V .,
Goyal, N., K ¨uttler, H., Lewis, M., Yih, W.-t., Rockt ¨aschel,
T., et al. Retrieval-augmented generation for knowledge-
intensive nlp tasks.Advances in neural information pro-
cessing systems, 33:9459–9474, 2020.
Li, S. and Ramakrishnan, N. Oreo: A plug-in context re-
constructor to enhance retrieval-augmented generation.
InProceedings of the 2025 International ACM SIGIR
Conference on Innovative Concepts and Theories in In-
formation Retrieval (ICTIR), pp. 238–253, 2025.
Li, X., Mei, S., Liu, Z., Yan, Y ., Wang, S., Yu, S., Zeng,
Z., Chen, H., Yu, G., Liu, Z., et al. Rag-ddr: Optimizing
retrieval-augmented generation using differentiable data
rewards.arXiv preprint arXiv:2410.13509, 2024.
9

Reinforcement Fine-Tuning for History-Aware Dense Retriever in RAG
Lin, X. V ., Chen, X., Chen, M., Shi, W., Lomeli, M., James,
R., Rodriguez, P., Kahn, J., Szilvasy, G., Lewis, M., et al.
Ra-dit: Retrieval-augmented dual instruction tuning. In
ICLR, 2024.
Liu, J. LlamaIndex, 11 2022. URL https://github.
com/jerryjliu/llama_index.
Liu, X., Li, D., Wen, T., Wan, J., Ling, G., Lv, F., Ou, D., and
Tang, H. Taosearchemb: A multi-objective reinforcement
learning framework for dense retrieval in taobao search.
arXiv preprint arXiv:2511.13885, 2025.
Loshchilov, I. and Hutter, F. Decoupled weight decay regu-
larization.arXiv preprint arXiv:1711.05101, 2017.
Naveed, H., Khan, A. U., Qiu, S., Saqib, M., Anwar, S.,
Usman, M., Akhtar, N., Barnes, N., and Mian, A. A
comprehensive overview of large language models.ACM
Transactions on Intelligent Systems and Technology, 16
(5):1–72, 2025.
Nguyen, T., Chin, P., and Tai, Y .-W. Reward-rag: Enhanc-
ing rag with reward driven supervision.arXiv preprint
arXiv:2410.03780, 2024.
Niu, C., Wu, Y ., Zhu, J., Xu, S., Shum, K., Zhong, R.,
Song, J., and Zhang, T. Ragtruth: A hallucination corpus
for developing trustworthy retrieval-augmented language
models. InProceedings of the 62nd Annual Meeting of
the Association for Computational Linguistics (Volume 1:
Long Papers), pp. 10862–10878, 2024.
Plackett, R. L. The analysis of permutations.Journal of the
Royal Statistical Society Series C: Applied Statistics, 24
(2):193–202, 1975.
Qdrant Team. Qdrant: Vector search engine. https:
//github.com/qdrant/qdrant , 2023. Accessed:
2026-01.
Rajpurkar, P., Zhang, J., Lopyrev, K., and Liang, P. SQuAD:
100,000+ questions for machine comprehension of text.
In Su, J., Duh, K., and Carreras, X. (eds.),Proceed-
ings of the 2016 Conference on Empirical Methods in
Natural Language Processing, pp. 2383–2392, Austin,
Texas, November 2016. Association for Computational
Linguistics. doi: 10.18653/v1/D16-1264. URLhttps:
//aclanthology.org/D16-1264/.
Schick, T., Dwivedi-Yu, J., Dess `ı, R., Raileanu, R., Lomeli,
M., Hambro, E., Zettlemoyer, L., Cancedda, N., and
Scialom, T. Toolformer: Language models can teach
themselves to use tools.Advances in Neural Information
Processing Systems, 36:68539–68551, 2023.
Shao, Z., Gong, Y ., Shen, Y ., Huang, M., Duan, N., and
Chen, W. Enhancing retrieval-augmented large languagemodels with iterative retrieval-generation synergy. In
Findings of the Association for Computational Linguis-
tics: EMNLP 2023, pp. 9248–9274, 2023.
Shao, Z., Wang, P., Zhu, Q., Xu, R., Song, J., Bi, X., Zhang,
H., Zhang, M., Li, Y ., Wu, Y ., et al. Deepseekmath: Push-
ing the limits of mathematical reasoning in open language
models.arXiv preprint arXiv:2402.03300, 2024.
Shi, W., Min, S., Yasunaga, M., Seo, M., James, R., Lewis,
M., Zettlemoyer, L., and Yih, W.-t. Replug: Retrieval-
augmented black-box language models. InProceedings
of the 2024 Conference of the North American Chapter of
the Association for Computational Linguistics: Human
Language Technologies (Volume 1: Long Papers), pp.
8371–8384, 2024.
Singh, A., Ehtesham, A., Kumar, S., and Khoei, T. T. Agen-
tic retrieval-augmented generation: A survey on agentic
rag.arXiv preprint arXiv:2501.09136, 2025.
Sun, H., Qiao, Z., Guo, J., Fan, X., Hou, Y ., Jiang, Y ., Xie,
P., Zhang, Y ., Huang, F., and Zhou, J. Zerosearch: In-
centivize the search capability of llms without searching.
arXiv preprint arXiv:2505.04588, 2025a.
Sun, Y ., Shen, J., Wang, Y ., Chen, T., Wang, Z., Zhou,
M., and Zhang, H. Improving data efficiency for llm
reinforcement fine-tuning through difficulty-targeted on-
line data selection and rollout replay, 2025b. URL
https://arxiv.org/abs/2506.05316.
Trivedi, H., Balasubramanian, N., Khot, T., and Sabharwal,
A. Interleaving retrieval with chain-of-thought reasoning
for knowledge-intensive multi-step questions. InPro-
ceedings of the 61st annual meeting of the association for
computational linguistics (volume 1: long papers), pp.
10014–10037, 2023.
von Werra, L., Belkada, Y ., Tunstall, L., Beeching,
E., Thrush, T., Lambert, N., Huang, S., Rasul, K.,
and Gallou ´edec, Q. TRL: Transformers Reinforce-
ment Learning, 2020. URL https://github.com/
huggingface/trl.
Wang, L., Chen, H., Yang, N., Huang, X., Dou, Z., and
Wei, F. Chain-of-retrieval augmented generation.arXiv
preprint arXiv:2501.14342, 2025.
Wang, Z., Teo, S., Ouyang, J., Xu, Y ., and Shi, W. M-rag:
Reinforcing large language model performance through
retrieval-augmented generation with multiple partitions.
InProceedings of the 62nd Annual Meeting of the Asso-
ciation for Computational Linguistics (Volume 1: Long
Papers), pp. 1966–1978, 2024.
10

Reinforcement Fine-Tuning for History-Aware Dense Retriever in RAG
Yang, Z., Qi, P., Zhang, S., Bengio, Y ., Cohen, W. W.,
Salakhutdinov, R., and Manning, C. D. HotpotQA: A
dataset for diverse, explainable multi-hop question an-
swering. InConference on Empirical Methods in Natural
Language Processing (EMNLP), 2018.
Yao, S., Yu, D., Zhao, J., Shafran, I., Griffiths, T., Cao, Y .,
and Narasimhan, K. Tree of thoughts: Deliberate problem
solving with large language models.Advances in neural
information processing systems, 36:11809–11822, 2023a.
Yao, S., Zhao, J., Yu, D., Du, N., Shafran, I., Narasimhan,
K. R., and Cao, Y . React: Synergizing reasoning
and acting in language models. InThe Eleventh In-
ternational Conference on Learning Representations,
2023b. URL https://openreview.net/forum?
id=WE_vluYUL-X.
Zamani, H. and Bendersky, M. Stochastic rag: End-to-end
retrieval-augmented generation through expected utility
maximization. InProceedings of the 47th International
ACM SIGIR Conference on Research and Development
in Information Retrieval, pp. 2641–2646, 2024.
Zhou, H., Chen, Y ., Guo, S., Yan, X., Lee, K. H., Wang, Z.,
Lee, K. Y ., Zhang, G., Shao, K., Yang, L., et al. Memento:
Fine-tuning llm agents without fine-tuning llms.arXiv
preprint arXiv:2508.16153, 2025.
Zhou, J. and Chen, L. Optimizing retrieval for rag via
reinforcement learning, 2026. URL https://arxiv.
org/abs/2510.24652.
11