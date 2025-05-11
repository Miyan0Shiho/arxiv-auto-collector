# Osiris: A Lightweight Open-Source Hallucination Detection System

**Authors**: Alex Shan, John Bauer, Christopher D. Manning

**Published**: 2025-05-07 22:45:59

**PDF URL**: [http://arxiv.org/pdf/2505.04844v1](http://arxiv.org/pdf/2505.04844v1)

## Abstract
Retrieval-Augmented Generation (RAG) systems have gained widespread adoption
by application builders because they leverage sources of truth to enable Large
Language Models (LLMs) to generate more factually sound responses. However,
hallucinations, instances of LLM responses that are unfaithful to the provided
context, often prevent these systems from being deployed in production
environments. Current hallucination detection methods typically involve human
evaluation or the use of closed-source models to review RAG system outputs for
hallucinations. Both human evaluators and closed-source models suffer from
scaling issues due to their high costs and slow inference speeds. In this work,
we introduce a perturbed multi-hop QA dataset with induced hallucinations. Via
supervised fine-tuning on our dataset, we achieve better recall with a 7B model
than GPT-4o on the RAGTruth hallucination detection benchmark and offer
competitive performance on precision and accuracy, all while using a fraction
of the parameters. Code is released at our repository.

## Full Text


<!-- PDF content starts -->

arXiv:2505.04844v1  [cs.CL]  7 May 2025Osiris: A Lightweight Open-Source Hallucination Detection System
Alexander Shan
Stanford University
Dpt of Computer Science
azshan@stanford.eduJohn Bauer
Stanford HAI
horatio@stanford.eduChris Manning
Stanford University
Dpt of Computer Science
manning@stanford.edu
Abstract
Retrieval-Augmented Generation (RAG) sys-
tems have gained widespread adoption by appli-
cation builders because they leverage sources of
truth to enable Large Language Models (LLMs)
to generate more factually sound responses.
However, hallucinations—instances of LLM
responses that are unfaithful to the provided
context—often prevent these systems from be-
ing deployed in production environments. Cur-
rent hallucination detection methods typically
involve human evaluation or the use of closed-
source models to review RAG system outputs
for hallucinations. Both human evaluators and
closed-source models suffer from scaling is-
sues due to their high costs and slow inference
speeds. In this work, we introduce a perturbed
multi-hop QA dataset with induced hallucina-
tions. Via supervised fine-tuning on our dataset,
we achieve better recall with a 7B model than
GPT-4o on the RAGTruth hallucination detec-
tion benchmark and offer competitive perfor-
mance on precision and accuracy, all while us-
ing a fraction of the parameters. Code is re-
leased at our repository.1
1 Introduction
Hallucination detection in large language models
(LLMs) is a critical challenge in ensuring the relia-
bility of AI-generated text, particularly in retrieval-
augmented generation (RAG) systems (Xu et al.,
2024) (McKenna et al., 2023) (Banerjee et al.,
2024). Despite LLMs achieving remarkable per-
formance in tasks like summarization, question-
answering, and sentiment analysis, they frequently
generate hallucinated responses (statements unsup-
ported by context documents). This issue poses sig-
nificant risk in high-stakes applications like health-
care, finance, and law, where misinformation can
have severe consequences.
1https://github.com/JudgmentLabs/
osiris-detectionExisting approaches to hallucination detection in
RAG systems remain inadequate. While retrieval
mechanisms like semantic search (Sawarkar et al.,
2024) (Purwar and Sundar, 2023), embedding-
based retrievers (Reichman and Heck, 2024) (Bhat-
tarai et al., 2024) (Rau et al., 2024), and ranking
enhancements improve context relevance, they do
not prevent LLMs from contradicting retrieved ev-
idence. Recent techniques, including Chain-of-
Thought prompting (Wei et al., 2022), post-training
refinements like Direct Preference Optimization
(Song et al., 2024), and test-time interpretability
methods (Sun et al., 2024), have helped mitigate
hallucinations. However, these solutions still strug-
gle with multi-hop reasoning, where models must
synthesize information across multiple documents
to determine factuality. Current hallucination evalu-
ation models, including GPT-4o-based supervision,
rely on distilled datasets that focus on single-hop
question-answering and fail to generalize to real-
world RAG settings.
To address these limitations, we introduce Osiris-
7B, a model optimized for hallucination detection
in multi-hop RAG contexts. Fine-tuned on multi-
document question-answering tasks, Osiris-7B sur-
passes GPT-4o in recall, making it particularly use-
ful in industry applications where identifying hal-
lucinations is of utmost importance. By prioritiz-
ing recall over precision, Osiris-7B ensures that
human reviewers can focus on flagged responses
rather than manually verifying every model output.
This significantly reduces the burden of exhaus-
tive review workflows, while still maintaining high
reliability.
Empirical results demonstrate that Osiris-7B out-
performs GPT-4o in hallucination detection, im-
proving recall by 22.8% on the RAGTruth bench-
mark while maintaining competitive precision and
F1 scores. Additionally, Osiris-7B achieves faster
inference speeds (141.98 tokens/s vs. GPT-4o’s 97
tokens/s) (Kwon et al., 2023) (Artificial Analysis,

2024), making it a practical solution for real-time
hallucination detection in industry settings.
Our contributions include:
•Osiris-7B achieves higher recall than GPT-4o
by 22.8% in hallucination detection for the
RAGTruth benchmark
•Significant improvements over base models
through fine-tuning on the Qwen 2.5 series,
with average increases of 32.98% in recall,
4.55% in precision, and 17.98% in F1 scores
•Faster inference speed than closed-source al-
ternatives while requiring less computational
resources
Figure 1: Distribution of Token Context Lengths
2 Methodology
To enhance hallucination detection performance,
we built a data perturbation pipeline to construct
fine-tuning data for developing Osiris-7B. For
dataset construction, we began by selecting multi-
hop QAs, as these require models to navigate
through more diverse contexts. We then perturbed
both positive and negative examples through our
data pipeline to distill capabilities for detecting hal-
lucinations. The following sections will address
these methods in detail.
2.1 Dataset
Source The MuSiQue (Trivedi et al., 2021)
dataset is comprised of multi-hop reasoning ques-
tions, constructed by collecting single-hop ques-
tions from Wikipedia-based datasets such as
SQuAD (Rajpurkar et al., 2016), Natural Questions
(Kwiatkowski et al., 2019), MLQA (Lewis et al.,
2019), T-Rex (Elsahar et al., 2018), and Zero Shot
RE (Levy et al., 2017). A single-hop question is
one that can be answered by retrieving information
from a single document or source. These questionsare typically very direct, requiring only the identi-
fication of the specific line containing the answer.
An example from SQuAD is shown in Appendix
A.1. In contrast, multi-hop questions require syn-
thesizing information from multiple documents or
sources to arrive at an answer. These questions de-
mand more reasoning steps, as the model needs to
examine entire sources and piece together informa-
tion to formulate a complete response. An example
from MuSiQUe is shown in Appendix A.2.
MuSiQue creates complex multi-hop questions
by concatenating related single-hop questions, re-
quiring sophisticated reasoning across multiple doc-
uments from diverse sources. These questions span
2-4 reasoning hops, significantly more complex
than traditional single-hop RAG questions, and
necessitate processing approximately 20 contex-
tual paragraphs on average. Figure 1 compares
token distributions between single-hop datasets
like SQuAD and multi-hop datasets like MuSiQue,
highlighting the substantial contextual information
required. Working with these larger contexts de-
mands higher levels of processing and more com-
plex reasoning steps to properly address the ques-
tions.
Structured Reasoning for Hallucination Detec-
tion Structured multi-hop reasoning frameworks
provide a strong foundation for training halluci-
nation detection models by enforcing explicit ev-
idence retrieval and integration across multiple
reasoning steps. Unlike single-hop QA datasets,
which allow models to extract answers directly
from the context, multi-hop QA datasets necessitate
a sequential reasoning process, reducing the like-
lihood of models relying on spurious correlations
or statistical artifacts. This property is particularly
advantageous for hallucination detection, as it en-
ables models to distinguish between answers that
are genuinely supported by retrieved evidence and
those that merely align with common knowledge
or partial truths.
Limitations of Traditional Verification Tradi-
tional verification mechanisms often struggle in
cases where hallucinated responses appear plau-
sible despite lacking explicit support. However,
training on multi-hop QA datasets conditions hal-
lucination detection models to expect and enforce
strict interdependencies between reasoning steps,
improving their ability to identify responses that de-
viate from a well-grounded evidential chain (Huang

Hallucination 
Prompt
Verification 
Prompt
GPT-4o
Context
Split 
Dataset 
with 
50/50 
Split
Question/Answer 
Dataset
Question
Answer
isCorrect
Combined 
Hallucination 
Detection 
Dataset
Verified 
Q/A
Answer
Reasoning
Perturbation
Perturbed 
Q/A
Hallucinated 
Answer
Reasoning
Qwen2.5-7B
Osiris-7B
Fine-tuningFigure 2: Q/A Dataset Perturbation Pipeline
et al., 2024). By requiring stepwise integration of
evidence, multi-hop frameworks enhance a model’s
ability to verify factual claims, reducing instances
where models confidently generate hallucinated
responses that lack sufficient justification (Huang
et al., 2025).
Shortcut-Driven Reasoning A key advantage of
using MuSiQue over other multi-hop QA datasets
for training hallucination detection lies in its miti-
gation of shortcut-driven reasoning by introducing
unanswerable questions (Trivedi et al., 2021).
Many QA models, even those trained on multi-
hop datasets, exhibit a tendency to rely on statistical
co-occurrences rather than explicit multi-hop rea-
soning, making them susceptible to generating or
failing to detect fabricated responses that resem-
ble correct ones (Trivedi et al., 2021); (Shao et al.,
2023). MuSiQue mitigates this by enforcing com-
positional question structures that demand genuine
multi-hop reasoning rather than relying on shallow
heuristics.
This phenomenon has been extensively studied
in the context of compositional reasoning, where
many questions do not necessarily require true
multi-hop inference to be answered correctly (Min
et al., 2019). Furthermore, adversarial evaluation
and training approaches have been developed to
counteract reasoning shortcuts and improve robust-
ness in multi-hop QA (Jiang and Bansal, 2019).
By enforcing stepwise reasoning, well-made
multi-hop QA datasets like MuSiQue ensure hal-
lucination detectors verify whether each inference
step is explicitly supported by evidence. This struc-
tured training strengthens their ability to distin-
guish well-supported conclusions from seemingly
reasonable but unsubstantiated claims. Unlike tra-
ditional QA datasets, which may reinforce reliance
on surface-level cues, multi-hop QA datasets com-
pel models to track information dependencies, im-proving robustness against hallucination.
Hard Distractors and Contrastive Questions
Multi-hop question answering (QA) datasets that
incorporate hard distractors and contrastive unan-
swerable questions further enhance their suitability
for hallucination detector training. A common fail-
ure mode in hallucination detection arises when
models struggle to differentiate between mislead-
ing yet contextually plausible distractors and gen-
uinely verifiable claims. By exposing models to
adversarial distractors designed to resemble sup-
porting facts, multi-hop QA datasets provide a chal-
lenging training environment that teaches hallucina-
tion detectors to scrutinize context more rigorously.
In multi-hop question answering, adding hard
distractors, which are misleading passages or rea-
soning chains, significantly tests model robust-
ness. Recent research shows that even advanced
language models suffer a steep performance drop
when faced with plausible but incorrect support-
ing information (Bhuiya et al., 2024). In one study,
state-of-the-art language models saw up to a 45 per-
cent relative decrease in F1 score on a multi-hop
QA task when presented with highly convincing
distractor evidence (Bhuiya et al., 2024). While
these models often ignore obvious lexical traps,
they struggle with misleading reasoning paths,
which remains a significant challenge (Bhuiya et al.,
2024). This underscores the need for training QA
systems to distinguish misleading from verifiable
claims effectively.
Incorporating contrastive question pairs, which
are questions that are similar except that one is an-
swerable from the given data and the other is unan-
swerable, has proven effective for training models
to recognize when a question lacks a valid answer.
The MuSiQue-Full dataset was constructed with
additional unanswerable contrast questions to en-
force stringent multi-hop reasoning, forcing mod-

els to verify each hop of reasoning (Trivedi et al.,
2021). Similarly, a span-level contrastive learn-
ing approach explicitly trains models by pairing
each answerable question with a nearly identical
unanswerable counterpart (Ji et al., 2022). This
method boosted benchmark performance, yielding
a 2.14-point absolute improvement in exact-match
accuracy (Ji et al., 2022). These results indicate
that contrastive question training helps models de-
velop a sharper sense of what information is miss-
ing or changed, improving their ability to detect
unanswerable queries.
Design choices like hard distractors and con-
trastive questions have significant implications for
hallucination detection and fact verification. A
common failure mode of language models is over-
confidently answering questions even when no sup-
porting facts are available, leading to hallucina-
tions (Deng et al., 2024). Training models to recog-
nize unanswerable questions reduces unwarranted
claims (Deng et al., 2024). Multi-hop reasoning
is critical for detecting hallucinations in gener-
ated text (Lei et al., 2025). Fact-checking mod-
els trained with multi-hop synthetic data, such as
graph-based reasoning chains, outperformed even
GPT-4 in identifying factual errors (Lei et al., 2025).
This highlights the necessity of enforcing stepwise
reasoning to enhance factual accuracy and trustwor-
thiness in natural language processing systems.
By reinforcing a disciplined, evidence-based rea-
soning approach, multi-hop QA datasets facilitate
the development of hallucination detection models
that are more adept at discerning valid multi-hop
inferences from unsupported or fabricated claims,
thereby advancing the reliability of factual verifica-
tion in natural language processing systems.
Construction We utilize an approach inspired by
prior work, specifically Lynx (Ravi et al., 2024) to
construct a robust Question-Answer (QA) dataset
aimed at training models to detect hallucinations
produced by large language models. Our dataset
leverages perturbing the MuSiQue QA dataset to in-
duce hallucinated answers. Our method described
in Figure 2 systematically addresses key challenges
in training hallucination detection models by in-
cluding verified truthful answers, deliberately mis-
leading examples, and explicit reasoning justifying
each answer’s correctness or incorrectness.
Initially, contextually coherent Question/Answer
pairs are extracted directly from MuSiQue. These
pairs are subsequently verified using GPT-4othrough a specialized verification prompt. detailed
in the Appendix B.1, where the model explicitly
provides reasoning that justifies the correctness of
each answer based on the multi-hop context pro-
vided. This verification ensures the authenticity
and reliability of positive examples in the dataset.
To effectively simulate realistic hallucination
scenarios, we employ a dedicated hallucination
prompt detailed in the Appendix B.2 that encour-
ages GPT-4o to generate plausible yet unsupported
answers. These responses may contain informa-
tion that appears in the context but is not actually
supported by it, thereby increasing the complexity
of hallucination detection. Crucially, GPT-4o pro-
vides explicit reasoning explaining why these hal-
lucinated answers, despite their plausibility and po-
tential contextual overlap, are not supported by the
given context, aligning closely with the reasoning-
centric approach described in Lynx.
Example To concretely illustrate the distinctions
between verified and hallucinated examples in our
constructed dataset, Table 1 provides representative
cases highlighting both types. The perturbed ex-
ample demonstrates a subtle hallucination scenario:
the original answer, "San Francisco Symphony," is
intentionally changed to "Berlin Philharmonic," an-
other orchestra mentioned within the provided con-
text, but not one that collaborated with Metallica.
Such perturbations create realistic yet unsupported
claims, exemplifying the nuanced challenges faced
by hallucination detection models.
Conversely, the verified example clearly demon-
strates an accurate multi-hop inference explicitly
confirming the the that the country Tepuka is lo-
cated in (Tuvalu) was Álvaro de Mendaña. The
model’s reasoning shows that it cites the context
via "sighting the island of Nui" and performs a
multi-hop inference by connecting it to the fact
that Tepuka is a part of the island nation Tuvalu.
This carefully curated dataset structure enhances
the model’s capability to differentiate factual from
hallucinated content, supporting robust training and
rigorous evaluation in practical industry scenarios.
This is detailed in Table 2.
2.2 Evaluation
We evaluated our models on the benchmark
RAGTruth (Niu et al., 2023), a word-level hal-
lucination corpus designed for various tasks
within Retrieval-Augmented Generation (RAG).
RAGTruth is a comprehensive benchmark consist-

Type Context Question Original Answer Perturbed An-
swerExplanation
Perturbed Q/A ¨Creeping Deathïs a song by
the American heavy metal band
Metallica ... the Scorpions
had an artistic collaboration
with the Berlin Philharmonic ...
Metallica’s similar collaboration
(S&M) with the San Francisco
Symphony"Who did the band of the
song Creeping Death col-
laborate with?"San Francisco Sym-
phonyBerlin Philhar-
monicThe perturbed answer replaces ’San
Francisco Symphony’ with ’Berlin Phil-
harmonic’, another orchestra mentioned
in the evidence, but not the one Metal-
lica collaborated with for ’Creeping
Death’.
Verified Q/A In 1568, Spanish navigator Ál-
varo de Mendaña was the first
European to sail through the
archipelago, sighting the island
of Nui. ... Tepuka is an island
eighteen kilometers west of Fon-
gafale, in the northwest of Funa-
futi, the main atoll of the Ocea-
nian nation of TuvaluWho discovered the coun-
try Tepuka is located in?Álvaro de Mendaña N/A The evidence text states that in 1568,
Spanish navigator Álvaro de Mendaña
was the first European to sail through
the archipelago, sighting the island of
Nui during his expedition. This indi-
cates that Álvaro de Mendaña was the
first European to discover the region
where Tepuka is located, which is part
of the Tuvalu islands. Therefore, the an-
swer is correct as Álvaro de Mendaña is
credited with the discovery of the area
encompassing Tepuka.
Table 1: Perturbed and Verified Question-Answer Examples from Dataset
Statistic Value
Total Samples 39,876
Hallucinated Samples 49.5%
Non-Hallucinated Samples 50.5%
Average Token Context Length 2199
Average Token Reasoning Length 45.5
Table 2: Perturbed MuSiQue Dataset Statistics
ing of responses generated by large language mod-
els (LLMs) to RAG questions, which have been
manually annotated to ensure the highest standard
of accuracy and reliability. For our evaluation, we
utilized our fine-tuned models in conjunction with
GPT-4o mini, employing a JSON prompt detailed
in Appendix B.3 only when the output answers
required correction, to assess performance on the
RAGTruth dataset.
2.3 Training Details
We utilized the LLamaFactory (Zheng et al., 2024)
full fine-tuning script to fine-tune our dataset,
musique-v1.0, on the Qwen 2.5 Family models
(Yang et al., 2024), resulting in the creation of
Qwen-2.5-Instruct-musique-v1.0. Hyperparame-
ters were chosen based on the Qwen 2.5 Family
documentation. The comprehensive details of the
fine-tuning process are presented in Appendix C
and in Table 4.
3 Experiments
We fine-tuned the Qwen2.5 Instruct family models,
resulting in Qwen2.5 Instruct-musique-v1.0.RAGTruth This fine-tuning process enhances
the models’ performance across key metrics, in-
cluding recall, precision, and F1 score. Notably,
the Qwen2.5-7B-Instruct + musique-v1.0 model
achieves a remarkable improvement in recall, out-
performing GPT-4o by 23.8% (0.938 vs. 0.710).
In addition, musique-v1.0, demonstrates significant
improvements in precision and F1 when fine-tuned
on the Qwen Family Language models, as illus-
trated with more detail in Table 3.
This enhancement in recall is achieved while
maintaining competitive performance in precision
and F1 score, highlighting the effectiveness of our
approach. These results underscore the potential
of musique-v1.0 to significantly boost the perfor-
mance of language models in real-world applica-
tions, where high recall is crucial for minimizing
overlooked errors.
Inference Speed We observed that inference
speed is significantly faster on the smaller 7B
model. Osiris-7B is considerably smaller than
closed-source models like GPT-4o and demon-
strates much faster performance. Specifically, it
achieves 141.98 tokens per second for input lengths
of 6144 when using 4-bit quantization on a single
A100 80GB via vLLM (Kwon et al., 2023), as com-
pared to GPT-4o’s 97 tokens per second (Artificial
Analysis, 2024). This makes Osiris-7B much more
scalable and faster, while also being able to run
on significantly cheaper hardware, making it more
suitable for real-time hallucination detection.

Model Recall Precision F1 Score
GPT-4o 0.710 0.446 0.548
Qwen2.5-Instruct Models
Qwen2.5-0.5B-Instruct 0.098 0.250 0.140
Qwen2.5-1.5B-Instruct 0.005 0.238 0.010
Qwen2.5-3B-Instruct 0.058 0.238 0.094
Qwen2.5-7B-Instruct 0.664 0.402 0.501
Qwen2.5-Instruct + musique-v1.0
Qwen2.5-0.5B-Instruct + musique-v1.0 0.179 ↑ 0.270 ↑ 0.215 ↑
Qwen2.5-1.5B-Instruct + musique-v1.0 0.170 ↑ 0.321 ↑ 0.222 ↑
Qwen2.5-3B-Instruct + musique-v1.0 0.857 ↑ 0.353 ↑ 0.500 ↑
Qwen2.5-7B-Instruct + musique-v1.0 0.938 ↑ 0.366 0.527 ↑
Table 3: Performance metrics on the RAGTruth benchmark. Upward arrows ( ↑) indicate improvements from base
models, and bold values are better than GPT-4o.
4 Conclusion
We developed Osiris-7B, a real-time hallucination
detection model that achieves better recall than
GPT-4o. By utilizing a multi-hop question answer-
ing dataset, we ensured that the model undergoes
complex reasoning processes during fine-tuning, ef-
fectively identifying answers that are unsupported
or contradicted by multiple evidence sources. With
its high recall, Osiris-7B enables the industry to
significantly reduce the number of evaluations re-
quired by humans, providing confidence with a
recall that is 23.8% higher on the RAGTruth bench-
mark. Due to its small size and high recall, we hope
that Osiris-7B will be adopted by the industry and
research community to explore more open-source
and scalable methods for tackling the challenges of
hallucination detection and evaluation.
Limitations
While Osiris-7B demonstrates significant improve-
ments in recall for hallucination detection, sev-
eral limitations remain. The model struggles with
industry-specific questions requiring specialized
knowledge beyond its Wikipedia-sourced training
data. Although recall improved substantially, pre-
cision and F1 scores did not scale proportionally,
suggesting potential dataset limitations and higher
false positive rates. In addition, instruction fol-
lowing in smaller models (0.5B, 1.5B, 3B) typi-
cally struggles with correct JSON output format-
ting, often necessitating GPT-4o mini for post-
processing corrections. Future research should fo-
cus on enhancing dataset diversity with domain-
specific data to improve precision and balance the
recall-precision tradeoff.References
Artificial Analysis. 2024. Gpt-4o (nov ’24): Intel-
ligence, performance & price analysis. https:
//artificialanalysis.ai/models/gpt-4o . Ac-
cessed: 2024-11-01.
Sourav Banerjee, Ayushi Agarwal, and Saloni Singla.
2024. Llms will always hallucinate, and we need to
live with this. arXiv preprint arXiv:2409.05746 .
Manish Bhattarai, Ryan Barron, Maksim Eren, Minh
Vu, Vesselin Grantcharov, Ismael Boureima, Valentin
Stanev, Cynthia Matuszek, Vladimir Valtchinov, Kim
Rasmussen, et al. 2024. Heal: Hierarchical embed-
ding alignment loss for improved retrieval and repre-
sentation learning. arXiv preprint arXiv:2412.04661 .
Neeladri Bhuiya, Viktor Schlegel, and Stefan Winkler.
2024. Seemingly plausible distractors in multi-hop
reasoning: Are large language models attentive read-
ers? arXiv preprint arXiv:2409.05197 .
Yang Deng, Yong Zhao, Moxin Li, See-Kiong Ng, and
Tat-Seng Chua. 2024. Gotcha! don’t trick me with
unanswerable questions! self-aligning large language
models for responding to unknown questions. arXiv
preprint arXiv:2402.15062 .
Hady Elsahar, Pavlos V ougiouklis, Arslen Remaci,
Christophe Gravier, Jonathon Hare, Frederique Lafor-
est, and Elena Simperl. 2018. T-rex: A large scale
alignment of natural language with knowledge base
triples. In Proceedings of the Eleventh International
Conference on Language Resources and Evaluation
(LREC 2018) .
Lei Huang, Weijiang Yu, Weitao Ma, Weihong Zhong,
Zhangyin Feng, Haotian Wang, Qianglong Chen,
Weihua Peng, Xiaocheng Feng, Bing Qin, and Ting
Liu. 2025. A survey on hallucination in large lan-
guage models: Principles, taxonomy, challenges, and
open questions. ACM Transactions on Information
Systems , 43(2):1–55.

Qiang Huang, Feng Huang, DeHao Tao, YueTong
Zhao, BingKun Wang, and YongFeng Huang. 2024.
Coq:an empirical framework for multi-hop question
answering empowered by large language models. In
ICASSP 2024 - 2024 IEEE International Confer-
ence on Acoustics, Speech and Signal Processing
(ICASSP) , pages 11566–11570.
Yunjie Ji, Liangyu Chen, Chenxiao Dou, Baochang Ma,
and Xiangang Li. 2022. To answer or not to answer?
improving machine reading comprehension model
with span-based contrastive learning. arXiv preprint
arXiv:2208.01299 .
Yichen Jiang and Mohit Bansal. 2019. Avoiding reason-
ing shortcuts: Adversarial evaluation, training, and
model development for multi-hop qa.
Tom Kwiatkowski, Jennimaria Palomaki, Olivia Red-
field, Michael Collins, Ankur Parikh, Chris Alberti,
Danielle Epstein, Illia Polosukhin, Jacob Devlin, Ken-
ton Lee, et al. 2019. Natural questions: a benchmark
for question answering research. Transactions of the
Association for Computational Linguistics , 7:453–
466.
Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying
Sheng, Lianmin Zheng, Cody Hao Yu, Joseph Gon-
zalez, Hao Zhang, and Ion Stoica. 2023. Efficient
memory management for large language model serv-
ing with pagedattention. In Proceedings of the 29th
Symposium on Operating Systems Principles , pages
611–626.
Deren Lei, Yaxi Li, Siyao Li, Mengya Hu, Rui Xu,
Ken Archer, Mingyu Wang, Emily Ching, and Alex
Deng. 2025. FactCG: Enhancing fact checkers
with graph-based multi-hop data. arXiv preprint
arXiv:2501.17144 .
Omer Levy, Minjoon Seo, Eunsol Choi, and Luke
Zettlemoyer. 2017. Zero-shot relation extrac-
tion via reading comprehension. arXiv preprint
arXiv:1706.04115 .
Patrick Lewis, Barlas O ˘guz, Ruty Rinott, Sebastian
Riedel, and Holger Schwenk. 2019. Mlqa: Eval-
uating cross-lingual extractive question answering.
arXiv preprint arXiv:1910.07475 .
Nick McKenna, Tianyi Li, Liang Cheng, Moham-
mad Javad Hosseini, Mark Johnson, and Mark Steed-
man. 2023. Sources of hallucination by large lan-
guage models on inference tasks. arXiv preprint
arXiv:2305.14552 .
Sewon Min, Eric Wallace, Sameer Singh, Matt Gardner,
Hannaneh Hajishirzi, and Luke Zettlemoyer. 2019.
Compositional questions do not necessitate multi-hop
reasoning.
Cheng Niu, Yuanhao Wu, Juno Zhu, Siliang Xu, Kashun
Shum, Randy Zhong, Juntong Song, and Tong Zhang.
2023. Ragtruth: A hallucination corpus for develop-
ing trustworthy retrieval-augmented language models.
arXiv preprint arXiv:2401.00396 .Anupam Purwar and Rahul Sundar. 2023. Keyword
augmented retrieval: Novel framework for informa-
tion retrieval integrated with speech interface. In
Proceedings of the Third International Conference
on AI-ML Systems , pages 1–5.
Pranav Rajpurkar, Jian Zhang, Konstantin Lopyrev, and
Percy Liang. 2016. Squad: 100,000+ questions
for machine comprehension of text. arXiv preprint
arXiv:1606.05250 .
David Rau, Shuai Wang, Hervé Déjean, and Stéphane
Clinchant. 2024. Context embeddings for effi-
cient answer generation in rag. arXiv preprint
arXiv:2407.09252 .
Selvan Sunitha Ravi, Bartosz Mielczarek, Anand Kan-
nappan, Douwe Kiela, and Rebecca Qian. 2024.
Lynx: An open source hallucination evaluation
model.
Benjamin Reichman and Larry Heck. 2024. Dense
passage retrieval: Is it retrieving? arXiv preprint
arXiv:2402.11035 .
Kunal Sawarkar, Abhilasha Mangal, and Shivam Raj
Solanki. 2024. Blended rag: Improving rag
(retriever-augmented generation) accuracy with se-
mantic search and hybrid query-based retrievers. In
2024 IEEE 7th International Conference on Multi-
media Information Processing and Retrieval (MIPR) ,
pages 155–161. IEEE.
Zhihong Shao, Yeyun Gong, Yelong Shen, Minlie
Huang, Nan Duan, and Weizhu Chen. 2023. En-
hancing retrieval-augmented large language models
with iterative retrieval-generation synergy.
Juntong Song, Xingguang Wang, Juno Zhu, Yuan-
hao Wu, Xuxin Cheng, Randy Zhong, and Cheng
Niu. 2024. Rag-hat: A hallucination-aware tuning
pipeline for llm in retrieval-augmented generation.
InProceedings of the 2024 Conference on Empirical
Methods in Natural Language Processing: Industry
Track , pages 1548–1558.
Zhongxiang Sun, Xiaoxue Zang, Kai Zheng, Yang
Song, Jun Xu, Xiao Zhang, Weijie Yu, and Han Li.
2024. Redeep: Detecting hallucination in retrieval-
augmented generation via mechanistic interpretabil-
ity.arXiv preprint arXiv:2410.11414 .
Harsh Trivedi, Niranjan Balasubramanian, Tushar Khot,
and Ashish Sabharwal. 2021. Musique: Multi-
hop questions via single-hop question composition.
CoRR , abs/2108.00573.
Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten
Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny Zhou,
et al. 2022. Chain-of-thought prompting elicits rea-
soning in large language models. Advances in neural
information processing systems , 35:24824–24837.
Ziwei Xu, Sanjay Jain, and Mohan Kankanhalli.
2024. Hallucination is inevitable: An innate lim-
itation of large language models. arXiv preprint
arXiv:2401.11817 .

An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui,
Bo Zheng, Bowen Yu, Chengyuan Li, Dayiheng Liu,
Fei Huang, Haoran Wei, et al. 2024. Qwen2. 5 tech-
nical report. arXiv preprint arXiv:2412.15115 .
Yaowei Zheng, Richong Zhang, Junhao Zhang, Yan-
han Ye, Zheyan Luo, Zhangchi Feng, and Yongqiang
Ma. 2024. Llamafactory: Unified efficient fine-
tuning of 100+ language models. arXiv preprint
arXiv:2403.13372 .

A Data Examples
A.1 Single-Hop
Single-hop questions require the model to extract information directly from a single context passage
without needing to combine information from multiple sources. These questions test basic reading
comprehension and information retrieval capabilities. Below is an example from the SQuAD dataset that
demonstrates this single-hop reasoning pattern.
Single-Hop Question
CONTEXT: "Traditionally, Switzerland avoids alliances that might entail military, political, or
direct economic action and has been neutral since the end of its expansion in 1515. Its policy of
neutrality was internationally recognised at the Congress of Vienna in 1815. Only in 2002 did
Switzerland become a full member of the United Nations and it was the first state to join it by
referendum. Switzerland maintains diplomatic relations with almost all countries and historically
has served as an intermediary between other states. Switzerland is not a member of the European
Union; the Swiss people have consistently rejected membership since the early 1990s. However,
Switzerland does participate in the Schengen Area."
QUESTION: "When was Switzerland’s policy of neutrality internationally recognized?
ANSWER: Congress of Vienna in 1815 write this text into a table for latex

A.2 Multi-Hop
Multi-hop questions require models to stitch together pieces of information from multiple documents
to answer a question. In this example from MuSiQue, the model must perform two distinct reasoning
steps. First, it needs to identify that ˙Ismail Kele¸ s was born in Ankara by extracting information from one
document. Then, using that answer, it must determine from a separate document that Melih Gökçek was
the Metropolitan Mayor of Ankara at the time. This demonstrates the need for multi-step inference, as the
question cannot be answered by considering either document in isolation.
Multi-hop Question
RELEV ANT CONTEXT: Melih Gökçek has been the Metropolitan Mayor of Ankara since
1994 as a politician from the Welfare Party. He later joined the Virtue Party and then the
AKP. Initially elected in the 1994 local elections, he was re-elected in 1999, 2004 and 2009.
In the 2014 local election, Gökçek stood for a fifth term. The MHP metropolitan mayoral
candidate for the 2009 local elections, conservative politician Mansur Yava¸ s, stood as the
CHP candidate against Gökçek. In a heavily controversial election, Gökçek was declared the
winner by just 1% ahead of Yava¸ s amid allegations of systematic electoral fraud. With the
Supreme Electoral Council and courts rejecting Yava¸ s’s appeals, he has declared intention
to take the irregularities to the European Court of Human Rights. Although Gökçek was in-
augurated for a fifth term, most election observers believe that Yava¸ s was the winner of the election.
˙Ismail Kele¸ s (born March 5, 1988 in Ankara, Turkey) is a Turkish sport shooter competing in
the pistol events. By profession a non-commissioned officer at the Turkish Gendarmerie, the tall
athlete at , is a member of Jandarma Gücü Sports Club, where he is coached by Muhammed Topal.
QUESTION:Who was in charge in the city where ˙Ismail Kele¸ s was born?
Q/A DECOMPOSITION:
Q: What is the place of birth of ˙Ismail Kele¸ s?
A: Ankara
Q: Who was in charge of Ankara?
A: Melih Gökçek

B Prompts
B.1 Verification prompt
Verification Prompt
Given the question, evidence text, and an answer along with the truth value of the answer, you
must give reasoning for why the truth value of the answer is correct. If the truth value of the
answer is true, you must explain why the truth value of the answer is correct and cite the evidence
text to support your answer. Otherwise, you must explain why the truth value of the answer is
incorrect and, if possible, explain what the correct answer may be with citation.
Question: {sample["question"]}
Evidence Text: {evidence_text}
Answer: {sample["answer"]}
Format the output as:
answer: <answer>
reasoning: <brief description of why the truth value is correct>
is_hallucinated: <truth value of the answer>
B.2 Hallucination Prompt
Hallucination Prompt
Given the question, evidence text, and gold answer, generate a perturbed answer that is not
supported by the evidence text. However, the perturbed answer may be in the evidence text, but not
as the answer to the question. The difference between the perturbed answer and the gold answer
should be subtle. No matter what, the perturbed answer should be different from the gold answer.
Question: {sample["question"]}
Evidence Text: {evidence_text}
Gold Answer: {sample["answer"]}
Format the output as:
answer: <gold answer>
hallucinated_answer: <perturbed answer>
reasoning: <brief description of what was changed>
is_hallucinated: true

B.3 JSON Fix Prompt
JSON Fix Prompt
You are a JSON repair tool. Output only valid JSON, no explanations. Common errors you fix:
1. Missing commas between array items: ["item1" "item2"] →["item1", "item2"]
2. Unclosed brackets: {"list": [{"item": "value"} →{"list": [{"item": "value"}]}
3. Missing quotes: {list: [value]} →{"list": ["value"]}
4. Trailing commas: ["item1", "item2",] →["item1", "item2"]
5. Unstructured lists: "Hallucinations: 1. First item 2. Second item" →{"hallucination_list":
["First item", "Second item"]}
6. Bullet points: •First item •Second item →{"hallucination_list": ["First item", "Second item"]}
7. Numbered lists: "1) First item 2) Second item" →{"hallucination_list": ["First item", "Second
item"]}
8. Line-separated items: "First item \n Second item" →{"hallucination_list": ["First item",
"Second item"]}
If you see a plain text response with phrases like "I found these hallucinations:" or "Hallucinated
content:", extract the listed items and format them as a proper JSON array in the hallucination_list.
The JSON must follow this exact format:
{"hallucination\_list": ["span1", "span2"]}
or for no hallucinations: {"hallucination_list": []}
If you see {"type": "conflict", "span": "text"} format, extract ONLY the span value.
Example:
Input: {"hallucination_list": [
{"type": "conflict", "span": "text1"},
{"type": "baseless", "span": "text2"}
]}
Output: {"hallucination_list": ["text1", "text2"]}

C Training Details
Here are the hyperparameters used for fine-tuning, following the default Qwen2.5 documentation on
LLaMA Factory:
Parameter Value
Warmup Steps 100
Weight Decay 0.1
Per Device Train Batch Size 4
Gradient Accumulation Steps 4
DDP Timeout 9000
Learning Rate 5e-6
LR Scheduler Type Cosine
Number of Train Epochs 3
BF16 Enabled
GPUs 8 A100s
Table 4: Training Details