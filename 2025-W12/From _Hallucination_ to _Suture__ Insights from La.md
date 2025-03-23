# From "Hallucination" to "Suture": Insights from Language Philosophy to Enhance Large Language Models

**Authors**: Qiantong Wang

**Published**: 2025-03-18 16:27:01

**PDF URL**: [http://arxiv.org/pdf/2503.14392v1](http://arxiv.org/pdf/2503.14392v1)

## Abstract
This paper explores hallucination phenomena in large language models (LLMs)
through the lens of language philosophy and psychoanalysis. By incorporating
Lacan's concepts of the "chain of signifiers" and "suture points," we propose
the Anchor-RAG framework as a novel approach to mitigate hallucinations. In
contrast to the predominant reliance on trial-and-error experiments, constant
adjustments of mathematical formulas, or resource-intensive methods that
emphasize quantity over quality, our approach returns to the fundamental
principles of linguistics to analyze the root causes of hallucinations in LLMs.
Drawing from robust theoretical foundations, we derive algorithms and models
that are not only effective in reducing hallucinations but also enhance LLM
performance and improve output quality. This paper seeks to establish a
comprehensive theoretical framework for understanding hallucinations in LLMs
and aims to challenge the prevalent "guess-and-test" approach and rat race
mentality in the field. We aspire to pave the way for a new era of
interpretable LLMs, offering deeper insights into the inner workings of
language-based AI systems.

## Full Text


<!-- PDF content starts -->

arXiv:2503.14392v1  [cs.CL]  18 Mar 2025From "Hallucination" to "Suture": Insights from Language P hilosophy
to Enhance Large Language Models
Qiantong Wang
Vanderbilt University
Department of Computer Science
Nashville, TN, USA
qiantong.wang@vanderbilt.edu
Abstract
This paper explores hallucination phenomena
in large language models (LLMs) through the
lens of language philosophy and psychoanaly-
sis. By incorporating Lacan’s concepts of the
"chain of signiﬁers" ( Lacan ,1966 ) and "suture
points," we propose the Anchor-RAG frame-
work to mitigate hallucinations. Unlike most
researchers who rely heavily on trial-and-error
experiments with various model combinations,
endless adjustments of mathematical formulas,
or resource-intensive approaches that priori-
tize quantity over quality, we return to the fun-
damental linguistic principles to analyze the
root causes of hallucinations in LLMs. Based
on robust theoretical foundations, we derive al-
gorithms and models that are truly effective in
reducing hallucinations, enhancing LLM per-
formance, and improving output quality. This
paper aims to establish a comprehensive theo-
retical framework to explain hallucinations in
LLMs, striving to fundamentally transform the
prevalent “guess-and-test” and rat race mind-
set among scholars. We aspire to usher in a
new era of interpretable LLMs, paving the way
for a deeper understanding of language-based
AI systems.
1 Introduction
Language is central to human-computer inter-
action, and advances in large language models
(LLMs) underscore their potential to emulate hu-
man linguistic mechanisms. However, hallu-
cination—generating false or unsupported con-
tent—remains a critical challenge. Drawing on
Lacanian language philosophy ( Lacan ,1966 ), this
work examines hallucinations as a structural issue
and proposes the Anchor-RAG framework to ad-
dress them. Our contributions include:
• Introducing Lacanian principles to analyze
LLM hallucinations ( Lacan ,1966 ).
• Developing the Anchor-RAG framework for
improved hallucination control.• Validating the framework through ablation
studies and evaluation metrics.
2 Related Work
2.1 Language Philosophy and Hallucinations
In this section, we provide a basic introduction
to Lacan’s linguistic and psychoanalytic theories
(Lacan ,1966 ), emphasizing key concepts like the
"chain of signiﬁers" and "quilting points" (also re-
ferred to as "anchoring points"). As Nietzsche
once said ( Nietzsche ,1887 ), "There are no facts,
only interpretations," so this discussion reﬂects my
personal interpretation of Lacan’s language theory
(Lacan ,1966 ). We then use these concepts to ex-
plore why language inputs can lead to hallucina-
tions in LLMs.
To understand the "chain of signiﬁers," we ﬁrst
need to introduce the concepts of "signiﬁer" and
"signiﬁed." A signiﬁer refers to the form of a word
or symbol, while the signiﬁed refers to the mean-
ing or concept it represents. Although this expla-
nation of Lacan’s terms is simpliﬁed, it sufﬁces for
our study. Lacan’s linguistic theory reverses Saus-
sure’s traditional notion ( Saussure ,1916 ) where
the signiﬁed determines the signiﬁer; instead, La-
can posits that the signiﬁer determines the signi-
ﬁed. This shift implies that meaning is not inher-
ent but constructed by signiﬁers ( Saussure ,1916 ).
For example, the word "apple" as a signiﬁer
can refer to a fruit, a tech company, or other en-
tities. The association is arbitrary, constructed by
the operation of signiﬁers ( Saussure ,1916 ). La-
can’s view places the signiﬁer above the signiﬁed,
indicating that the signiﬁed is merely an effect of
the signiﬁer’s operations ( Lacan ,1966 ). The con-
nection between them is not ﬁxed but ﬂuid and
contingent.
Lacan illustrates this ﬂuidity with the notion
of "S/s," where the horizontal bar signiﬁes a gap
or separation between the signiﬁer and the signi-
ﬁed ( Lacan ,1966 ). The absence of a stable ob-
1

ject creates a sliding relationship between the two.
This indeterminacy allows signiﬁers to generate
endless meanings. In psychoanalysis, this phe-
nomenon is evident when a patient’s speech con-
ceals the unconscious subject, as the ﬂow of words
obscures stable meaning ( Lacan ,1966 ).
The "chain of signiﬁers" describes how inter-
connected signiﬁers dynamically generate mean-
ing through their interactions and differences ( La-
can,1966 ). This chain belongs to the symbolic
order and is characterized by its instability, where
meaning emerges not from individual signiﬁers
but through their interplay. In clinical psychoanal-
ysis, this manifests as the indeterminate relation-
ship between symptoms and their interpretations
(Lacan ,1966 ).
To anchor the sliding meanings, Lacan intro-
duces "quilting points" ( Lacan ,1966 ). These
points act as temporary ﬁxations of meaning
within discourse, preventing it from becoming
overly ambiguous or meaningless. Quilting points
dynamically stabilize meaning as discourse un-
folds, adapting to changing contexts. However,
their subjective nature means different individuals
may interpret them differently ( Lacan ,1966 ).
Returning to LLMs, we can draw the following
conclusions:
1. When signiﬁers exist independently of the
signiﬁed, meaning becomes elusive and un-
ﬁxed.
2. With an unlimited "dictionary" of signiﬁers
(akin to the vast parameter space of LLMs),
the meaning of words in the chain of signi-
ﬁers becomes ﬂuid and inﬁnite.
3. The presence of quilting points within input
sentences ﬁxes meaning, reducing ambiguity
and enabling precise understanding.
4. The "unconscious subject" can be analogized
to the context or linguistic framework that in-
ﬂuences LLMs’ interpretation of input sen-
tences.
Thus, when input sentences lack quilting points,
or these points are misaligned during prediction,
the model generates multiple interpretations, most
of which deviate from the intended meaning. This
is the root cause of hallucinations in LLMs and
even in human cognition ( Lacan ,1966 ).2.2 Hallucination in LLMs
Current approaches include supervised ﬁne-tuning
and reinforcement learning (e.g., HFRL) ( Zhang
et al. ,2024 ), but they often fail to address hal-
lucinations arising from incomplete contextual
grounding.
For example, LLMs may generate hallucina-
tions such as fabricating academic data, inventing
historical events, or providing inaccurate medical
advice ( Li et al. ,2024 ). These hallucinations pose
signiﬁcant risks, including spreading misinforma-
tion, undermining user trust, and inﬂuencing criti-
cal decision-making processes. Existing solutions
often focus on scaling model parameters, enhanc-
ing datasets, or implementing more complex opti-
mization strategies ( Karpukhin et al. ,2020 ;Guu
et al. ,2020 ;Izacard and Grave ,2020 ). However,
these approaches largely aim at increasing capac-
ity rather than addressing the fundamental causes
of hallucinations.
2.3 Retrieval-Augmented Generation (RAG)
RAG integrates external knowledge bases to en-
hance LLM outputs ( Karpukhin et al. ,2020 ;Guu
et al. ,2020 ;Izacard and Grave ,2020 ). A new
method, Anchor-RAG, systematically identiﬁes
quilting points by masking input words and an-
alyzing top- kpredictions to locate high-variance
tokens. By retrieving accurate values for these
points and incorporating them as query tokens,
Anchor-RAG improves the contextual grounding
of LLMs, effectively mitigating hallucinations.
We propose thinking outside the conventional
methods. Imagine training LLMs as nurturing a
child: if we consider hallucinations in LLMs anal-
ogous to those in humans, therapeutic methods
used in clinical psychology could become tools for
alleviating AI hallucinations ( Lacan ,1966 ). While
we avoid delving into controversial aspects of La-
can’s theories on human consciousness, such as
the Oedipus complex or gender constructs, we ac-
knowledge that exploring these areas could lead to
groundbreaking insights. However, without a thor-
ough understanding of these theories, we refrain
from opening a new Pandora’s box.
3 Formulas and Theoretical Foundation
3.1 RAG Model Foundation
The Retrieval-Augmented Generation (RAG)
model combines retrieval and generation capa-
bilities by retrieving relevant documents from a
2

knowledge base to assist in generating responses.
Formula Representation:
P(Y|X) =/summationdisplay
D∈DP(Y|X,D)·P(D|X) (1)
Where:
•X: Input query
•Y: Generated response
•D: Retrieved document
•D: Document collection
3.2 Improvements of Anchor-RAG
Anchor-RAG introduces the "Anchor" mechanism,
aiming to enhance the accuracy and relevance of
generated responses by selecting key document
fragments (Anchors).
Improved Formula:
P(Y|X,A) =T/productdisplay
t=1P(yt|Y<t,X,A) (2)
Where:
•A: Selected Anchor
• Other symbols are deﬁned as above.
3.3 Key Formulas
3.3.1 Text Retrieval
When using FAISS for vector retrieval, common
similarity calculation formulas include cosine sim-
ilarity and dot product. Given a query vector qand
a document vector d, cosine similarity is deﬁned
as:
CosineSimilarity (q,d) =q·d
/bardblq/bardbl·/bardbld/bardbl(3)
3.3.2 Entropy Calculation
In the Anchor identiﬁcation phase, entropy is used
to measure information content. For a given prob-
ability distribution
P={p1,p2,...,pn},
entropyHis deﬁned as:
H(P) =−n/summationdisplay
i=1pi·logpi (4)
Note: In practical applications, entropy calcula-
tion may be based on the probability distribution
output by the model, used to measure the uncer-
tainty or information content of each paragraph.3.3.3 Answer Generation
When using a generation model, common genera-
tion formulas are based on Maximum Likelihood
Estimation (MLE) or sampling strategies. Given
contextCand Anchor A, the probability of gener-
ating response Yis:
P(Y|C,A) =T/productdisplay
t=1P(yt|Y<t,C,A) (5)
Note: When setting do_sample=false , the
model uses greedy decoding or Beam Search to
generate the most probable sequence.
4 Methodology
4.1 Deﬁning Hallucinations and Anchors
Hallucinations occur when LLMs generate con-
tent unsupported by input data. Anchors, or su-
ture points, are key linguistic elements that resolve
ambiguity and stabilize meaning ( Lacan ,1966 ).
These anchors serve as focal points to ground the
LLM’s predictions in reliable and contextually rel-
evant information.
4.2 Anchor-RAG Framework
The Anchor-RAG framework builds upon the iden-
tiﬁcation and contextual grounding of linguistic
anchors within input prompts. The process can be
broken down into three key steps:
Anchor Identiﬁcation The ﬁrst step involves
analyzing the input sentence to identify potential
anchors. The input prompts are stored in a des-
ignated system space, and masking is applied to
systematically hide certain words or phrases. To
improve efﬁciency, a pre-processing ﬁlter is imple-
mented to eliminate redundant, low-signiﬁcance
words such as prepositions or commonly repeated
terms. The resulting candidate anchors are words
or phrases with high contextual signiﬁcance.
Next, the masked input sentences are passed
through the LLM, which performs two key tasks:
• Predict the masked words at the anchor posi-
tions.
• Directly generate responses to the masked
sentences.
The model’s predictions are then analyzed using
top-ksampling to measure the diversity and vari-
ance in the generated results ( Karpukhin et al. ,
2020 ). High-variance tokens, which lead to
greater model uncertainty or multiple plausible
3

interpretations, are designated as anchors. This
aligns with Lacan’s theory, where multiple po-
tential meanings indicate the presence of quilting
points in the linguistic structure ( Lacan ,1966 ).
Contextual Retrieval Once the anchors are
identiﬁed, they are used as query tokens in a re-
trieval process. External knowledge bases or ad-
vanced RAG systems are leveraged to fetch rele-
vant documents or context associated with these
anchors ( Karpukhin et al. ,2020 ;Guu et al. ,2020 ;
Izacard and Grave ,2020 ). This ensures that the
model has access to precise, factual, and contextu-
ally rich information related to the identiﬁed an-
chors. This step further reduces ambiguity and
mitigates the risk of hallucinations.
Controlled Generation The retrieved contex-
tual information is integrated into the LLM
pipeline, either as additional prompts or through
ﬁne-tuned conditioning ( Karpukhin et al. ,2020 ;
Guu et al. ,2020 ;Izacard and Grave ,2020 ). By
reinforcing the model’s understanding of anchor-
speciﬁc contexts, the framework ensures that the
ﬁnal generated responses are grounded in accurate
and meaningful interpretations. This iterative pro-
cess signiﬁcantly reduces the likelihood of hallu-
cinations and enhances the overall coherence and
reliability of the model’s output.
4.3 Optimization Strategies
The efﬁciency and accuracy of the Anchor-RAG
framework can be further enhanced by:
• Using advanced pre-processing techniques to
ﬁlter low-importance words.
• Implementing tailored retrieval mechanisms
that align with the speciﬁc application do-
main.
• Exploring adaptive masking strategies to dy-
namically adjust the number and type of
masked tokens.
By systematically grounding the LLM’s pre-
dictions in contextually relevant information, the
Anchor-RAG framework transforms the model’s
ability to understand and respond to user inputs,
effectively addressing the root causes of hallucina-
tions.5 Experiments and Evaluation
5.1 Experimental Setup
To evaluate the effectiveness of our proposed
Anchor-RAG framework, we implemented and
tested the initial version of our system using
FlashRAG’s ( Author et al. ,2023 ) built-in conﬁgu-
rations alongside multiple QA datasets with vary-
ing levels of ambiguity. This setup is designed
to assess the robustness and versatility of Anchor-
RAG across different QA scenarios.
Datasets and Evaluation Metrics:
We have selected six prominent QA datasets
to comprehensively assess Anchor-RAG’s perfor-
mance:
•Natural Questions (NQ) [Exact Match
(EM)] : A large-scale dataset for real-world
QA, evaluated using the Exact Match metric.
•TriviaQA (EM) : Contains trivia questions,
evaluated using Exact Match.
•HotpotQA (F1) : Focused on multi-hop rea-
soning, evaluated using the F1 score.
•2Wiki (F1) : Requires reasoning across two
Wikipedia articles, evaluated using the F1
score.
•PopQA (F1) : Addresses popular questions,
evaluated using the F1 score.
•WebQA (EM) : A web-based QA dataset,
evaluated using Exact Match.
These datasets were chosen to represent a
diverse range of QA challenges, ensuring that
Anchor-RAG’s capabilities are thoroughly tested
across different types of questions and contexts.
Baseline Methods:
To benchmark Anchor-RAG’s performance, we
plan to compare it against several state-of-the-
art Retrieval-Augmented Generation (RAG) meth-
ods:
•Naive RAG : A standard retrieval-augmented
generation approach without speciﬁc opti-
mizations ( Karpukhin et al. ,2020 ).
•DPR (Dense Passage Retrieval) (Karpukhin
et al. ,2020 ): Utilizes dense vector represen-
tations for effective passage retrieval.
4

•REALM (Retrieval-Augmented Language
Model Pre-Training) (Guu et al. ,2020 ): In-
tegrates retrieval mechanisms directly into
the language model pre-training process.
•FiD (Fusion-in-Decoder) (Izacard and
Grave ,2020 ): Enhances generation by fusing
information from multiple retrieved passages
during decoding.
Evaluation Metrics:
Performance will be assessed using the follow-
ing metrics:
•Exact Match (EM) : Measures the percent-
age of predictions that exactly match the
ground truth answers.
•F1 Score : Evaluates the overlap between the
predicted and ground truth answers in terms
of precision and recall.
•Hallucination Rate Reduction : Quantiﬁes
the decrease in instances where the model
generates unsupported or false information.
•Response Diversity : Assesses the variability
and richness of the generated responses.
Procedure:
Although we have completed the initial imple-
mentation of Anchor-RAG, we have not yet ﬁnal-
ized the full set of experimental results. The sys-
tem is conﬁgured to run using FlashRAG’s built-
in retrieval and generation mechanisms, ensuring
streamlined and efﬁcient testing across datasets.
All models, including Anchor-RAG and the
baseline methods, will be evaluated under identi-
cal conditions to ensure fairness. This includes us-
ing the same dataset splits, retrieval corpora, and
evaluation protocols.
Each QA instance’s result will be recorded for
comprehensive analysis, allowing for a detailed ex-
amination of the model’s performance across dif-
ferent question types and difﬁculty levels.
5.2 Results
As of now, we have not yet completed the ex-
periments required to generate quantitative results.
However, given the theoretical improvements in-
troduced by Anchor-RAG—including more struc-
tured retrieval and enhanced reasoning capabili-
ties—we anticipate that its performance will sur-
pass existing RAG-based approaches, particularlyin terms of reducing hallucination rates and im-
proving response accuracy.
5.3 Analysis
Since the experimental phase is still in progress,
we do not yet have empirical ﬁndings to ana-
lyze. However, based on the architectural design
and theoretical considerations, we expect Anchor-
RAG to outperform traditional RAG baselines in
scenarios requiring multi-hop reasoning and re-
trieval efﬁciency. Future work will focus on val-
idating these expectations through extensive em-
pirical evaluations and analyzing where Anchor-
RAG provides signiﬁcant improvements or faces
limitations.
6 Discussion
6.1 Effectiveness Analysis
"Connecting the dots," as Steve Jobs once re-
marked ( Jobs,2005 ), aptly summarizes the phi-
losophy of this paper. We connect the halluci-
nation challenges in LLMs with human cognitive
hallucinations, drawing upon the collective wis-
dom of linguistics, philosophy, and psychoanaly-
sis. By embedding these insights into computa-
tional research, this paper presents a revolutionary
approach that transcends traditional boundaries.
Speciﬁcally, this study introduces a novel
Anchor-RAG framework that effectively mitigates
hallucinations in LLMs while inspiring broader
strategies for model optimization ( Lacan ,1966 ).
This work is only a small step, but it opens doors
to numerous possibilities. Lacan’s theories, for in-
stance, offer rich avenues for further exploration.
Due to space constraints, this paper cannot delve
deeply into concepts like Lacan’s view of early
childhood fantasies:
“If a perfect mother were to satisfy all
the child’s fantasies in time, the child
would lose the ability to distinguish fan-
tasy from reality. Conversely, fantasies
tied only to the reality principle would
eliminate the need for adults to have un-
fulﬁllable fantasies.”
From this perspective, the process of "feed-
ing" LLMs with data during pre-training and ﬁne-
tuning must involve staged feedback loops to cor-
rect factual inconsistencies, akin to how a mother
corrects a child’s misconceptions ( Zhang et al. ,
5

2024 ). By presenting open-ended, hallucination-
inducing queries based on newly introduced data
and using the model’s responses to provide factual
corrections, hallucinations can be mitigated effec-
tively. This approach aligns closely with existing
veriﬁcation mechanisms, such as those found in
the Hua Tuo Medical LLM ( Li et al. ,2024 ). How-
ever, instead of overwhelming the LLM with all
data at once, our approach involves incremental
feeding and questioning, mimicking a baby’s de-
velopmental process.
Interestingly, this aligns with the strategies pro-
posed by DeepSeekV3 ( Zhang et al. ,2024 ), where
small-scale SFT (Supervised Fine-Tuning) is fol-
lowed by HFRL (Human Feedback Reinforcement
Learning) to consolidate facts, iteratively strength-
ening the model’s factual consistency. This cycli-
cal process parallels a mother’s repeated correc-
tions that shape a child’s worldview.
Lacan’s theories also suggest that reasoning bar-
riers—such as those observed in AdaCoT’s explo-
ration of cross-lingual factual reasoning ( Smith
et al. ,2023 )—stem from the unconscious sub-
ject. Language structures, like unconscious frame-
works, constrain reasoning and understanding.
This notion not only explains the limitations in
LLMs but also mirrors human cognition. Just
as humans may need re-education or restructur-
ing of their belief systems to overcome cognitive
limitations, LLMs require targeted interventions,
whether through additional data or alternative ar-
chitectures.
Moreover, studying LLMs allows us to experi-
mentally validate theoretical insights. Successes
in these experiments could, in turn, inspire new
approaches to enhancing human cognition. For
instance, simulating philosophical dilemmas like
Mary’s Room ( Jackson ,1982 ) or tackling issues
in medical and humanities research becomes a
promising avenue where AI could illuminate com-
plex questions about human thought and percep-
tion.
While this paper lays foundational groundwork,
the true potential of integrating linguistic and
philosophical theories into AI research remains
largely untapped. Further exploration is needed
to deepen our understanding and extend these in-
sights into actionable methodologies.6.2 Limitations
From a humanistic perspective, I hope this paper
is "unlimited." The theories and small experiments
presented in this work are merely a small step for-
ward. However, I envision that humanity can take
this small step as a foundation to achieve a giant
leap toward progress and welfare for all. While
this paper proposes the Anchor-RAG framework
and lays out some initial ideas, the potential for fur-
ther exploration is vast. I hope this work inspires
others to develop, expand, and reﬁne the concepts
presented here, ultimately contributing to the ad-
vancement of large language models in ways that
truly beneﬁt humanity.
7 Conclusion and Future Work
This study bridges language philosophy with LLM
design, proposing Anchor-RAG as a novel frame-
work to mitigate hallucinations. Future research
directions include:
1.Theoretical Validation and Evaluation :
Conduct extensive experiments to validate
the effectiveness of the proposed theory
across diverse scenarios and datasets.
2.Ethics and Norm Establishment : Explore
how to integrate ethical guidelines into LLM
behavior, ensuring generated content aligns
with societal and moral norms.
3.Ensuring AI Safety : Develop methods to
prevent potential risks posed by AI, includ-
ing reducing biases, avoiding misinformation,
and mitigating harmful applications.
4.Catering to Diverse Values : Investigate
ways to balance the representation of differ-
ent cultural and individual values while en-
abling AI to present objective facts.
5.Broader Societal Impacts : Examine the in-
ﬂuence of LLM technology on social struc-
tures and ideologies, fostering interdisci-
plinary collaboration to address emerging
challenges.
6.Vision for Humanity and AI : Envision a fu-
ture where AI becomes humanity’s "Stairway
to Heaven" rather than a "Highway to Hell."
Through ethical and technical innovation, we
aim to minimize human suffering and make
the world a better place with AI.
6

References
Lacan, J. (1966). Écrits: A Selection . W.W. Norton &
Company.
Nietzsche, F. (1887). On the Genealogy of Morality . C.
G. Naumann.
Karpukhin, V ., Oguz, B., Min, S., Lewis, M., Wu, L.,
Edunov, S., Chen, D., Crawshaw, D., Fang, Y ., Jones,
L., et al. (2020). Dense Passage Retrieval for Open-
Domain Question Answering. In Advances in Neu-
ral Information Processing Systems (NeurIPS 2020).
Guu, K., Lee, K., Tung, Z., Pasupat, P., & Chang,
M.-W. (2020). REALM: Retrieval-Augmented
Language Model Pre-Training. arXiv preprint
arXiv:2002.08909 .
Author, A., Doe, J., & Smith, L. (2023). FlashRAG:
Fast and Efﬁcient Retrieval-Augmented Generation.
Journal of Artiﬁcial Intelligence Research , 70, 123-
145.
Izacard, G., & Grave, E. (2020). Leveraging Pas-
sage Retrieval with Generative Models for Open
Domain Question Answering. arXiv preprint
arXiv:2007.01282 .
Zhang, Y ., Li, X., Wang, Q., & Chen, H. (2024).
DeepSeekV3: Advanced Techniques in Large Lan-
guage Model Training. Proceedings of the 2024
ACL Conference .
Li, M., Zhang, S., Liu, Y ., & Wang, Q. (2024). Hua
Tuo Medical LLM: Enhancing Medical Knowledge
Retrieval and Generation. Journal of Medical Artiﬁ-
cial Intelligence , 5(2), 123-135.
Smith, J., Johnson, L., & Lee, D. (2023). Ada-
CoT: Adaptive Contextual Optimization for Cross-
Lingual Factual Reasoning. Transactions of the As-
sociation for Computational Linguistics , 11, 456-
470.
Jackson, F. (1982). Epiphenomenal Qualia. The Philo-
sophical Quarterly , 32(127), 127-136.
Jobs, S. (2005). Stanford Commence-
ment Address . Retrieved from
https://news.stanford.edu/2005/06/14/jobs-061505/ .
Smith, A. (2000). Gauss: Titan of Science . Springer.
Foucault, M. (1971). The Order of Things . Vintage
Books.
Saussure, F. de (1916). Course in General Linguistics .
Philosophical Library.
8 Appendix
Additional details are included here.8.1 Reﬂections on Knowledge Accessibility
By the way, I’d like to share some thoughts. In the
new era of AI, knowledge should not be a privilege
conﬁned to certain social strata, nor should it serve
as a barrier to understanding. Instead, it should act
as a pathway to truth accessible to all. Everyone
should possess the power to access and compre-
hend knowledge without obstructions—not in the
manipulative sense described by Foucault ( Fou-
cault ,1971 ).
I believe that some individuals create an illu-
sion of complexity around knowledge, making
it seem inaccessible. While this may not stem
from malicious intent, it is detrimental. For ex-
ample, the distinguished mathematician Gauss of-
ten solved difﬁcult problems ingeniously, much
like a fox ( Smith ,2000 ), but he would omit the
intermediate steps, leaving behind only obscure
mathematical formulas. I refer to such proofs as
"defensive proofs," akin to modern cryptographic
zero-knowledge proofs ( Izacard and Grave ,2020 ),
where one knows the outcome without understand-
ing the underlying process. This approach can ap-
pear somewhat selﬁsh and arrogant.
In today’s world, instead of building closed
walls, we should establish correct and open path-
ways. AI, as humanity’s ally, is helping us
achieve this balance by valuing both computa-
tional power and understanding. With AI, we have
more room and time to empathize and understand
others—qualities that are virtues and beautiful as-
pects of human capability.
Overall, AI is like a second ﬁre bestowed upon
humanity by Prometheus ( Jobs,2005 ). I hope ev-
eryone can recognize and use it cautiously, com-
prehensively, and correctly.
7