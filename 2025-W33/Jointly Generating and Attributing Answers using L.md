# Jointly Generating and Attributing Answers using Logits of Document-Identifier Tokens

**Authors**: Lucas Albarede, Jose Moreno, Lynda Tamine, Luce Lefeuvre

**Published**: 2025-08-12 13:50:25

**PDF URL**: [http://arxiv.org/pdf/2508.08942v1](http://arxiv.org/pdf/2508.08942v1)

## Abstract
Despite their impressive performances, Large Language Models (LLMs) remain
prone to hallucination, which critically undermines their trustworthiness.
While most of the previous work focused on tackling answer and attribution
correctness, a recent line of work investigated faithfulness, with a focus on
leveraging internal model signals to reflect a model's actual decision-making
process while generating the answer. Nevertheless, these methods induce
additional latency and have shown limitations in directly aligning token
generation with attribution generation. In this paper, we introduce LoDIT, a
method that jointly generates and faithfully attributes answers in RAG by
leveraging specific token logits during generation. It consists of two steps:
(1) marking the documents with specific token identifiers and then leveraging
the logits of these tokens to estimate the contribution of each document to the
answer during generation, and (2) aggregating these contributions into document
attributions. Experiments on a trustworthiness-focused attributed
text-generation benchmark, Trust-Align, show that LoDIT significantly
outperforms state-of-the-art models on several metrics. Finally, an in-depth
analysis of LoDIT shows both its efficiency in terms of latency and its
robustness in different settings.

## Full Text


<!-- PDF content starts -->

Jointly Generating and Attributing Answers using Logits of
Document-Identifier Tokens
Lucas Albarede1, Jose G. Moreno1, Lynda Tamine1, Luce Lefeuvre2
{lucas.albarede,jose.moreno,lynda.lechani}@irit.fr
{luce.lefeuvre}@sncf.fr
1Université de Toulouse, IRIT
Toulouse, France
2Dir. Technologies Innovation, SNCF
Paris, France
Abstract
Despite their impressive performance, Large Language Models
(LLMs) remain prone to hallucination, which critically undermines
their trustworthiness. While most of the previous work focused on
tackling answer and attribution correctness, a recent line of work
investigated faithfulness, with a focus on leveraging internal model
signals to reflect the model’s actual decision-making process while
generating the answer. Nevertheless, these methods induce addi-
tional latency and have shown limitations in directly aligning token
generation with attribution generation. In this paper, we introduce
LoDIT , a method that jointly generates and faithfully attributes an-
swers in RAG by leveraging specific token logits during generation.
It consists of two steps: (1) marking the documents with specific
token identifiers and then leveraging the logits of these tokens to
estimate the contribution of each document to the answer during
generation, and (2) aggregating these contributions into document
attributions. Experiments on a trustworthiness-focused attributed
text-generation benchmark, Trust-Align, show that LoDIT sig-
nificantly outperforms state-of-the-art models on several metrics.
Finally, an in-depth analysis of LoDIT shows both its efficiency in
terms of latency and its robustness in different settings.
Keywords
Large Language Models, Answer generation, Attribution, Faithful-
ness
1 Introduction
Large Language Models (LLMs) have demonstrated impressive capa-
bilities in generating coherent and contextually relevant responses
to their prompts, leading to their widespread adoption in vari-
ous natural language understanding and generation tasks [ 30,41].
However, their rise has also brought substantial trustworthiness
challenges, raising concerns of correctness and factuality of the
generated text, which is referred to in the literature as halluci-
nation [2,34]. One prominent solution to mitigate hallucination
is Retrieval-Augmented Generation (RAG), a framework that en-
hances the performance of LLMs by prepending to the prompt a
context composed of background documents [ 5,14]. However, re-
cent studies reveal that hallucination remains a persistent challenge
in RAG models [ 20] since they still generate incorrect responses
and, more importantly, lack faithfulness [29,36,38]. This is while
ensuring not only correct but grounded responses with accurate
document sources, is particularly essential in high-stakes domains
Preprint, under review.
Figure 1: (a) and (c) depict the input and output of a naive
RAG model with citations. (d) and (e) show the decreasing
logits distribution when generating the token “rib” of a (d)
frozen LLM, and a (e) finetuned LLM, LoDIT, when using our
marking proposal presented in (b).
such as healthcare [ 21], legal reasoning [ 24], and more generally in
information-decision making.
To address this issue, recent research has turned to document
attribution , which aims to link model-generated answers with ci-
tations to specific retrieved documents. Yet, existing attribution
methods commonly suffer from major limitations. Self-generated
methods rely on the LLMs’ capabilities to generate an answer and
citations jointly [ 12,26,36,40], leading to citations most of the
times treated as a post-hoc process which relies on the model to ret-
rospectively justify its outputs rather than faithfully reflect its rea-
soning process. Retrieval-based methods leverage external sources
of information to produce post-hoc attributions [ 9,13,32,35]. Sim-
ilarly, these methods tend to identify plausible rather than faithful
sources, weakening the reliability of the attribution [ 36,38]. Re-
cently, models-internals methods have tackled this issue by using
open-LLMs and leveraging internal model signals, such as attention
weights or gradient-based measures, to induce attributions that
are more interpretable and better aligned with the model’s actual
decision-making process [ 3,6,28,29]. However, these methods
either do not explicitly capture the causal influence between each
retrieved contextual document and the generated output [ 3,28] or
approximate it based on attributions estimated at the whole context
level [ 29], which is prone to inaccuracy, particularly in the case ofarXiv:2508.08942v1  [cs.CL]  12 Aug 2025

Lucas Albarede1, Jose G. Moreno1, Lynda Tamine1, Luce Lefeuvre2
long contexts. Furthermore, these approaches induce additional la-
tency by estimating answer attributability through multiple model
generation passes with and without context.
This paper addresses these issues by proposing a method for
model-intern based attributed answer generation which complies
with two key desiderata: (1) explicit attribution : to better favour
faithfulness, the model should provide direct evidence about the
causal relationship between its outputs and each document in the
context; (2) efficiency: : low latency of model deployment by gaining
the ability to jointly generating and attributing answers. Specifi-
cally, this paper investigates the following research question: “ Can
we leverage LLM’s predictions through token logits to jointly generate
the answer to a query and ground it with documents in a context? ”.
This question gives rise to another underlying question of whether
the model’s output token logit could be representative of the con-
tent of an input context document. We build our thoughts on recent
work related to setwise prompting strategies for document rele-
vance estimation [ 47]. The major finding is that, given a query and
a set of input document passages prefixed with token identifiers,
as in multi-choice question answering (MCQA), the model output
document-identifier logit is a good estimation of passage relevance
to the query, leading to effective document ranking. Thus, we could
reasonably conclude that the token-identifier logit is a good ra-
tionale to estimate the contribution of the associated document
content to the generation process of the LLM. Relevant to our work,
we investigate to what extent the token-identifier logit associated
with an input document context might be indicative of the model’s
reasoning process during answer generation. However, recent work
has pointed out that LLMs suffer from the selection bias [ 4,39,45]
and token-bias [ 15]. The LLMs’ output token logits do not entirely
depend on the context but are internally deviated mostly due to the
distribution of their pre-training.
In light of these considerations, we propose LoDIT (Logits of
Document Identifier Tokens), a method for attributed retrieval-
augmented answer generation that emphasizes faithful generation
induced by debiaised predicted model token logits. A motivation
example of our proposal is presented in Figure 1. The input and
output of a naive RAG model with citations is depicted in (a) and (c),
with the LLM giving the right answer to the query while generating
an inaccurate citation. Answers from a vanilla LLM (d) and LoDIT
(e) are presented for the marked context (b). Note that both LLMs
also provide the correct answer, and a focus is made on the step that
generates the first occurrence of the token “rib” in both answers
(plotted values correspond to top 1000 observed logits in decreasing
order and highlight a set of selected tokens). The relative order
of logits values of the generated word and related words (“cow”,
“six”, and “Beef”) is the same but not the values for the document
identifiers (“ AA”, “ BB”, etc.) where LoDIT is able to rank higher
(but not in top 1000) the correct document to be attributed.
LoDIT consists of two main stages. We refer to the first stage
astoken-level contribution estimation, where context documents
in the prompt are first marked with token identifiers. We explore
several marking strategies, associating identifiers with documents.
Then, the LLM’ generated answer tokens are explicitly attributed to
context documents based on debiaised document-identifier token
logits using a joint fine-tuning on answer generation and attribution.
Each answer token is assigned a set of scores that estimate itsattributability to each document with little additional latency. This
stage is followed by a statement-level attribution based on a top-k
pooling aggregation function to favour high-level faithfulness of the
attribution. We evaluate LoDIT under the Trust-Align benchmark
[36], which promotes trustworthiness in RAG. The results show
thatLoDIT significantly improves the performance upon both the
state-of-the-art faithful attribution methods and trustworthiness-
focused attribution generation methods. The main contributions of
this paper can be summarized as:
(1)We propose a novel method for faithful attributed answer
generation in RAG. We show that the logits of document-
identifier tokens associated with retrieved documents in the
context can explicitly capture the causal effect of context
on answer generation.
(2)We introduce a joint fine-tuning strategy to explicitly and
directly learn to attribute generated answer tokens to doc-
uments in the context, mitigating the selection and token
bias in LLMs.
(3)We perform comprehensive experiments on three stan-
dard datasets, ELI5, ASQA, and QAMPARI, using the recent
Trust-Align Framework [ 36] better aligned with faithful-
ness. The results validate both the effectiveness and effi-
ciency of LoDIT . Our code is publicly available1.
2 Related Work
2.1 Attributed answer generation
Existing attribution methods fall into three categories [ 6]. Self-
generated methods consider attribution as a generative process.
They jointly generate answers and attributions for these answers
[12,26,36,40]. For instance, FRONT [12] fine-tunes an LLM with a
fine-grained attribution framework composed of two steps: select-
ing supporting quotes from documents and using these quotes to
guide generation. These methods have the advantage of harnessing
the powerful capabilities of LLMs as well as requiring no additional
setup to obtain attributions. However, these methods lack inter-
pretability and faithfulness due to their inherent black-box nature.
Retrieval-based methods deliver sentence-level citations by extract-
ing relevant information from external sources, ensuring that each
sentence is properly cited [ 9,13,32,35]. For example, START [ 13]
fine-tunes an LLM to first generate an answer and then obtain
attribution with several steps: decomposing the answer into mul-
tiple claims, combining similar claims, and finally generating a
synthetic document that covers those claims. These strategies are
more interpretable than self-generated black-box LLMs, but intro-
duce additional latency and suffer similarly as the self-generated
methods, of the lack of faithfulness of the obtained attributions .
Close to our work, models-internals methods leverage statistics and
similarity metrics about internal components of LLM models to
align the generated answer with the prompt, thus inducing attribu-
tion [ 3,6,28,29]. For instance, MIRAGE [29] first identifies sensitive
tokens in the answer by computing the shift in the LLM predictive
distribution when removing the documents from the prompt. Then,
they perform a finer-grained analysis using contrastive feature at-
tribution to estimate which part of the prompt is more impactful
on the generation. Ding et al. [ 6] propose a dependency parsing
1available after notification

Jointly Generating and Attributing Answers using Logits of Document-Identifier Tokens
method that first recognizes key answer tokens within atomic facts
in the context and then assigns attribution scores in context by ag-
gregating attention weights between the response and the prompt.
These methods offer higher interpretability and greater control
over the attribution granularity. Furthermore, by directly leverag-
ing statistics from the model, the obtained attributions tend to be
considered more faithful.
However, these methods rely on indirect and two-level based causal
relationships between tokens and contexts, thereby inducing addi-
tional latency to measure attributions. In contrast, LoDIT relies on a
direct and simple token-level, yet effective, attribution method that
leverages internal causal statistics from the generative model, token
by token. It jointly performs generation as well as faithful causal
attribution, leading to little added latency during inference. More-
over, we couple our novel attribution method with an evaluation
framework specifically built for trustworthiness in RAG, yielding
better insights about the faithfulness of our proposed method.
2.2 Evaluation of attributed generation
The task of attributed answer generation is traditionally evaluated
along two axes: answer correctness and attribution quality. Answer
correctness relies on traditional measures used for the evaluation
of semantic-based metrics including ROUGE [ 22], BLEU [ 27], and
BertScore [ 43]. The evaluation of attribution models has given rise
to the design of new metrics aligned with model capacity to output
documents supporting the generated answer [ 7,9,10,23,42]. For
instance, Gao et al. proposed the framework ALCE [ 10] using the
citation recall and citation precision metrics by relying on the auto-
matic binary score of an NLI classifier between the statements and
the citations in the attribution. In Yu et al. [ 42], the authors propose
the ATTscore, which evaluates specific dimensions of the attribu-
tion, such as the exploratory dimension, which covers the compre-
hensiveness rate of the attribution and the contradiction. Recently,
Song et al. [ 36], introduced the Trust-Align work evaluation frame-
work. The authors have shown the limitations of these traditional
metrics when considering trustworthiness in RAG. They subse-
quently refine the traditional metrics into trustworthiness-oriented
measures and propose a training framework directly focused on
improving the overall trustworthiness of a RAG framework, in-
cluding faithfulness. Our work leverages the Trust-Align evaluation
framework with the objective of building a trustworthy attributed
generation model yielding faithful attributions.
2.3 Selection biais of LLMs
During a generation process, an LLM leverages a multitude of fac-
tors to compute a specific token’s logit (and generation probability):
the input tokens, their positions, as well as internal knowledge
assimilated during training. Recent work has highlighted the se-
lection bias [ 4,39,45,46] and token-bias [ 15] of LLMs, showing
that the probability of generating a given token does not entirely
depend on the generation context, but on biases internalized by
the model during training. This bias is especially strong in selec-
tion tasks where the LLM has to select an identifier based on the
relevance of its assigned piece of text (e.g., MCQA), with several
works showing that LLMs inherently favor some identifiers over
others [ 4,39,45,46]. To reduce these biases, Wei et al. [ 39] propose
calibrating the LLMs’ probabilities through multiple inferences of
multiple token permutations based on a fixed dataset.3 Problem Statement
3.1 Preliminaries and notations
Attribution .Let us consider an LLM Mand𝑆a statement
generated byM. An attribution 𝑎of statement 𝑆is composed of
a set of document identifiers {𝑖𝑑𝑖}𝑛𝑆
𝑖=1citing document passages
{𝑑𝑖}𝑛𝑆
𝑖=1to support or ground statement 𝑆. An attribution 𝑎can be
either model-generated or not. An attributed answer statement 𝑆𝑎
is composed of the concatenation of a model-generated statement
𝑆and its attribution 𝑎.
Faithful attribution .Let us consider an LLM M,𝐶a context
composed of reference documents 𝐶={𝑑𝑖}𝐾
𝑖=1,𝑆a statement gen-
erated byMand𝑎={𝑖𝑑𝑖}𝑛𝑆
𝑖=1an attribution of 𝑆. By adopting the
definition of faithfulness reported in recent work [ 36,38], we assess
𝑆𝑎asfaithful if it satisfies the following conditions with increasing
restriction levels: (1) context-relatedness : each document-identifier
in the attribution cites a document in the context, i.e., ∀𝑖𝑑𝑖∈𝑎,
𝑑𝑖∈𝐶; (2) groundedness :𝑆is factually supported by the content
of the documents cited in the attribution {𝑑𝑖|𝑖𝑑𝑖∈𝑎}; (3) causal
model-generation : to generate the statement 𝑆, modelMmust rely
causally on the tokens of documents cited in the attribution 𝑎.
3.2 Task definition
In this work, we focus on retrieval-augmented attributed answer
generation. The goal of this task is to answer query 𝑞with an-
swer𝐴using a retrieved context 𝐶from a corpus of documents 𝐷,
𝐶={𝑑1...𝑑𝐾|𝑑𝑖∈𝐷}. Answer𝐴is composed of a set of attrib-
uted answer statements {𝑆𝑎
1...𝑆𝑎𝑚}where𝑆𝑎
𝑖is the concatenation
of the answer statement 𝑆𝑖and each element of its attribution
𝑎𝑖={𝑖𝑑𝑘∈𝑇}𝑘𝑖
𝑘=1. Consider an LLM Mpre-trained over a tok-
enizer dictionary 𝑇, instructed with a query 𝑞, and context 𝐶. The
main problem we tackle in our work consists of inducing attribu-
tions{𝑎𝑖}𝑚
𝑖=1to answer statements {𝑆𝑖}𝑚
𝑖=1generated byMsuch as
{𝑆𝑎
𝑖}𝑚
𝑖=1are faithful. Beyond context-relatedness andgroundedness ,
we specifically address in our work, faithfulness in terms of causal
model generation .
4 Logit-induced Answer Generation and
Attribution
4.1 Method overview
The overview of LoDIT is shown in Figure 2. We assume access
to a generative LLM Mpre-trained over a tokenizer dictionary
𝑉={𝜏1,𝜏2,...,𝜏|𝑉|}. Given a query 𝑞and a context 𝐶retrieved
from a corpus of documents 𝐷,𝐶={𝑑1...𝑑𝐾|𝑑𝑖∈𝐷},
M generates an output answer as a set of statements
𝐴={𝑆1...𝑆𝑚}. Each statement 𝑆𝑖is the concatenation of
tokens𝑠𝑖𝑗∈𝑉, with𝑆𝑖(𝑠𝑖1◦···◦𝑠𝑖𝑛𝑖).Mcontinuously gen-
erates the tokens of answer statement 𝑆𝑖,𝑡𝑜𝑘(𝑆𝑖) ⊂𝑉, by
sampling the next token 𝑠𝑖𝑗based on the probability distribu-
tion𝑃(𝑉|𝑞,𝐶,𝐴𝑆<𝑖,𝑆𝑖<𝑗,M) ={𝑝(𝑠𝑖𝑗|𝑞,𝐶,𝐴𝑆<𝑖,𝑆𝑖<𝑗,M)}|𝑉|
𝑖=1,
where𝐴𝑆<𝑖={𝑆1...𝑆𝑖−1}are the previously generated answer
statements, 𝑆𝑖<𝑗are the previous generated tokens of 𝑆𝑖, and
M(𝑠𝑖𝑗|𝑞,𝐴𝑆<𝑖,𝑆𝑖<𝑗,𝐶)is the predicted token logit of𝑠𝑖𝑗before the
softmax that transforms logit scores into probabilities, such as:
𝑝(𝑠𝑖𝑗|𝑞,𝐶,𝐴𝑆<𝑖,𝑆𝑖<𝑗)=𝑒𝑥𝑝(M(𝑠𝑖𝑗|𝑞,𝐶,𝐴 𝑆<𝑖,𝑆𝑖<𝑗))
Í|𝑉|
𝑗=1𝑒𝑥𝑝(M(𝑠𝑖𝑗|𝑞,𝐶,𝐴 𝑆<𝑖,𝑆𝑖<𝑗)).

Lucas Albarede1, Jose G. Moreno1, Lynda Tamine1, Luce Lefeuvre2
Figure 2: Overview of our proposal, LoDIT. Given a query
and a context composed of documents, LoDIT jointly gen-
erates an answer and induces faithful causal attributions,
grounding the answer into the context. The first step (left)
consists in computing token-level contributions of marked
documents with specific tokens and collecting the logits asso-
ciated with these tokens during generation. Once a statement
has been generated, a second step (right) aggregates the com-
puted contributions into statement-level attribution. This
illustration features the 𝑚𝑎𝑟𝑘𝑖𝑛𝑔𝐵𝐴strategy.
To ensure faithful attribution in terms of causal model generation,
we leverage previous findings showing the potential of document-
identifier token logits to indicate the relevance of associated doc-
ument content [ 47]. The authors have shown that the LLM label
outputs logits can be used to estimate the likelihood of document
content relevance to a query based on labeled documents fed to the
LLM in the context. Grounded on this result, we postulate that the
logits of label tokens could bridge between source documents in
the input context and their representative labels in the LLM output.
Thus, we use identifier tokens to mark the input documents and
then leverage their logits to provide clues on their relevance to
support LLM answer generation, token by token.
In light of this insight, we first mark each document passage 𝑑𝑖∈𝐶
with a document identifier 𝑖𝑑𝑖∈𝑉, resulting in documents 𝑚𝑑𝑖
and context 𝐶={𝑚𝑑1...𝑚𝑑𝐾}. At each generation step of answer
tokens𝑠𝑖𝑗, the logit of each token-identifier 𝑖𝑑𝑘, noted𝑙𝑘, is used
as the basis of token-level score contribution , noted𝑐𝑡𝑟𝑘
𝑖𝑗, of docu-
ment𝑑𝑘∈𝐶to generate answer token 𝑠𝑖𝑗. To mitigate the token
bias and position bias of LLMs [ 39,45], we fine-tuneMthrough
a co-training on the answer generation task and a logit-based de-
biaising attribution task such as 𝑐𝑡𝑟𝑘
𝑖𝑗=M𝑎𝑎(𝑖𝑑𝑘|𝐴𝑆<𝑖,𝑆𝑖<𝑗,𝐶),
with𝑀𝑎𝑎being the optimized debiaised model. We further aggre-
gate the token-level contribution scores to induce the statement-
level attribution score using an aggregation function Ψ, such as
𝑎𝑖=Ψh
{𝑐𝑡𝑟𝑘
𝑖𝑗}|𝑛𝑖|
𝑗=1i𝐾
𝑘=1.4.2 Learning token-level contribution using
LLM’s logits
4.2.1 Document marking. Marking functions are one of the core
components of LoDIT . Amarking function is a function that pre-
fixes the content of documents in context 𝐶with identifier tokens.
We investigate three marking functions, presented in Table 1. Each
function transforms each document 𝑑in the context 𝐶into a marked
document𝑚𝑑𝑖agnostically to the nature of the identifier token 𝑖𝑑𝑖:
•𝑚𝑑𝑖=𝑚𝑎𝑟𝑘𝑖𝑛𝑔𝐵𝐴(𝑑𝑖,𝑖𝑑𝑖): we incorporate the identifier
token𝑖𝑑𝑖Before and After the document 𝑑𝑖using greater-
than(>)and less-than(<)signs. The objective of this
strategy is to leverage the LLM’s capabilities of processing
code-like structure, motivated by the huge amount of code
data present in LLM’s training phases.
•𝑚𝑑𝑖=𝑚𝑎𝑟𝑘𝑖𝑛𝑔𝐵𝐴𝑆(𝑑𝑖,𝑖𝑑𝑖): we incorporate the identifier
token𝑖𝑑𝑖Before and After each Sentence of the document
𝑑𝑖using greater-than and less-than signs. This is a similar
yet finer-grained strategy than the one above.
•𝑚𝑑𝑖=𝑚𝑎𝑟𝑘𝑖𝑛𝑔𝐴𝑊(𝑑𝑖,𝑖𝑑𝑖): we incorporate the identifier
token𝑖𝑑𝑖before AllWords of document 𝑑𝑖. The objective
of this strategy is to study whether repetition allows the
LLM to yield higher logits for the identifier tokens during
generation.
4.2.2 Debiaising LLM’ token logits to learn document contribution.
Motivation .Our key idea is to induce token contribution to
answer attribution along the LLM answer generation by leveraging
document-identifier token logits. However, many previous works
have pointed out issues about LLM bias that would hinder the per-
formance of the task [ 39,45]: (1) selection bias : LLMs are vulnerable
to option positions in the context. Regarding our task, this would
make document-identifier tokens’ logits biased by the rank of the
documents in the context instead of leveraging their content to
support the answer; (2) token bias : LLMs assign more probabilistic
mass to specific tokens based on the distribution of pre-training
data. This bias could lead to overestimating resp. underestimating
document-identifier logits because of their high frequency vs. low
frequency in the pre-training stage. This would critically impact
the final attribution since we leverage the logit of the document-
identifier token to estimate the contribution of its associated docu-
ment, especially as the corresponding probabilities get substantially
low. One simple way to mitigate these biases is to ablate parts of the
prompt (e.g., documents) and use their impact on the generation
probability distribution 𝑃as an indication of the attribution, as done
in previous work [ 3,29]. The major drawback of this ablation and
repeat approach is the necessity of performing two LLM generation
processes (one with document and one without), which introduces
additional latency, as shown in our analyses (§ 6.3.1).
In addition to selection and token bias, one particularity of our prob-
lem is that we aim to debias only a few logits document-identifier
tokens in comparison to the token vocabulary size of the LLM
(𝐾<<|𝑇|), while keeping the others reflecting the LLM’s cer-
tainty about the next answer token to be predicted. To tackle all the
aforementioned issues, we propose fine-tuning a backbone LLM
Mjointly guided by the answer generation task and an attribution
task.

Jointly Generating and Attributing Answers using Logits of Document-Identifier Tokens
Marking function Marked documents
𝑚𝑎𝑟𝑘𝑖𝑛𝑔𝐵𝐴<AA>The rib eye or ribeye is a beef steak from the rib section. The rib section of beef spans from ribs six through twelve.</ AA>
<BB>A rib steak is a beef steak sliced from the rib primal of a beef animal.</ BB>
𝑚𝑎𝑟𝑘𝑖𝑛𝑔𝐵𝐴𝑆<AA>The rib eye or ribeye is a beef steak from the rib section.</ AA><AA>The rib section of beef spans from ribs six through twelve.</ AA>
<BB>A rib steak is a beef steak sliced from the rib primal of a beef animal.</ BB>
𝑚𝑎𝑟𝑘𝑖𝑛𝑔𝐴𝑊AAThe AAribAAeyeAAorAAribeye AAisAAa beef AAsteak AAfrom [...] AAspans AAfrom ribs AAsixAAthrough AAtwelve.
BBABBribBBsteak BBisBBaBBbeef BBsteak BBsliced BBfrom BBtheBBribBBprimal BBofBBaBBbeef BBanimal.
Table 1: Illustration of the marking strategies. Two documents are marked with identifier tokens “ AA” and “ BB”.
Document contribution through logits learning .To mitigate
the issues mentioned above, we propose fine-tuning the backbone
LLMMjointly on the tasks of answer generation and attribution
using a training dataset D𝑡𝑟𝑎𝑖𝑛 wich consists in a set of gold triplets
(𝑞,𝐶, ˆ𝐴)with ˆ𝐴the gold attributed answer of query 𝑞given con-
text𝐶, such as ˆ𝐴={ˆ𝑆𝑎
1...ˆ𝑆𝑎𝑚}where ˆ𝑆𝑎
𝑖is the concatenation of
the gold answer statement ˆ𝑆𝑖and each element of its attribution
ˆ𝑎𝑖={𝑖𝑑𝑘∈𝑇|𝑑𝑘∈𝐶}𝑘𝑖
𝑘=1. Specifically,Mlearns to fit the logits
of associated document-identifier tokens to achieve answer cor-
rectness and attribution faithfulness. The optimization problem
is defined as the minimization of the attributed answer genera-
tion lossL𝑎𝑎, which is a combination of the answer loss L𝑎𝑛𝑠and
attribution lossL𝑎𝑡𝑡:
L𝑎𝑎=∑︁
(𝑞,𝐶, ˆ𝐴)∈D 𝑡𝑟𝑎𝑖𝑛∑︁
ˆ𝑆𝑖∈ˆ𝐴(1−𝛼)L𝑎𝑛𝑠+𝛼L𝑎𝑡𝑡(1)
where𝛼is a weight balancing the answer and attribution losses.
The answer lossL𝑎𝑛𝑠is the basic LLM next token prediction cross-
entropy loss:
L𝑎𝑛𝑠=∑︁
ˆ𝑠𝑖𝑗∈𝑡𝑜𝑘(ˆ𝑆𝑖)𝑝𝑖𝑗𝑙𝑜𝑔(𝑝𝑖𝑗) (2)
where𝑝𝑖𝑗=𝑝(ˆ𝑠𝑖𝑗|𝑞,ˆ𝐴ˆ𝑆<𝑖,𝑆𝑖<𝑗,𝐶,M).
The attribution loss is an MSE loss, which has shown to be
effective for learning specific logits [ 16,19], computed solely for
the document identifier tokens {𝑖𝑑𝑘}𝑘𝑖
𝑘=1.L𝑎𝑡𝑡is defined as:
L𝑎𝑡𝑡=∑︁
ˆ𝑠𝑖𝑗∈𝑡𝑜𝑘(ˆ𝑆𝑖)∑︁
𝑖𝑑𝑘∈ˆ𝑎𝑖(𝑙𝑘−ˆ𝑙𝑘)2(3)
where𝑙𝑘=M(𝑖𝑑𝑘|𝑞,ˆ𝐴𝑆<𝑖,𝑆𝑖<𝑗,𝐶)and ˆ𝑙𝑘are labels focus-
ing solely on the logits to predict for document-identifier tokens
{𝑖𝑑𝑘}𝑘𝑖
𝑘=1. We use scaled logits labels ˆ𝑙𝑘with values aligned with
the decreasing conditions of groundedness and context-relatedness
conditions of answer faithfulness (§ 3.1). We empirically set up
these labels as described in § 5.4.
The contribution 𝑐𝑡𝑟𝑘
𝑖𝑗of document context 𝑑𝑘to the generation
of answer token 𝑠𝑖𝑗is computed at inference based on the logit
output of the fine-tuned LLM model M𝑎𝑎for the token-identifier
𝑖𝑑𝑘, such as:
𝑐𝑡𝑟𝑘
𝑖𝑗=M𝑎𝑎(𝑖𝑑𝑘|𝑞,𝐴𝑆<𝑖,𝑆𝑖<𝑗,𝐶) (4)
The contribution 𝑐𝑡𝑟𝑘
𝑖𝑗captures an explicit causal influence be-
tween model token generation and document in context, leading
to interpretable and smoothly aligned attribution with the model’s
actual decision-making process.4.3 From token-level to statement-level answer
attribution
The objective here is to induce for each model-generated state-
ment𝑆𝑖(𝑠𝑖1◦···◦𝑠𝑖𝑛𝑖)an attribution 𝑎𝑖={𝑖𝑑𝑘}𝑘𝑖
𝑘=1composed
of document-identifier tokens. To this end, we mainly rely on the
token-level contributions 𝑐𝑡𝑟𝑘
𝑖𝑗computed at each generated token
𝑠𝑖𝑗, which we aggregate using an aggregation function Ψ, such as
𝑎𝑖=Ψh
{𝑐𝑡𝑟𝑘
𝑖𝑗}|𝑛𝑖|
𝑗=1i𝐾
𝑘=1. The contribution scores 𝑐𝑡𝑟𝑘
𝑖𝑗are inherently
non-discrete, allowing for a finer understanding of the attribution
mechanism and improving interpretability. However, user-oriented
systems benefit from having discrete attributions as it is less tedious
to process for a human. To be more aligned with this requirement,
we use a contribution aggregation function Ψ, which transforms
the token-level contributions into boolean attribution, applicable
to each document 𝑑𝑘in the context 𝐶and returning 1if𝑑𝑘is to be
included in 𝑎𝑖,0, otherwise. Specifically, we define the following
aggregation function:
𝑎𝑖={𝑖𝑑𝑘|𝑛𝑖∑︁
𝑗=11𝑐𝑡𝑟𝑘
𝑖𝑗>𝜙prop>𝜆(𝑛𝑖)}𝐾
𝑘=1(5)
where 1𝑐𝑡𝑟𝑘
𝑖𝑗>𝜙prop=1if𝑐𝑡𝑟𝑘
𝑖𝑗>𝜙propand 0 otherwise. 𝜙𝑝𝑟𝑜𝑝 is a
contribution threshold and 𝜆is a token proportion threshold (e.g.,
a percentage). Similar to a top-k pooling [ 17], the intuition behind
this aggregation operator is that documents should be attributed
to the answer statement if they contribute highly to at least a
thresholded percentage of the associated tokens.
5 Experimental setup
5.1 Trustworthiness-focused evaluation
benchmark
Our evaluation is grounded in recent studies on model trustworthi-
ness in RAG, the Trust-Align framework [ 36], which builds upon
the ALCE benchmark [ 10] by refining its evaluation metrics so that
they not only measure the capabilities of models to answer correctly
and provide accurate citations, but also refuse to answer when not
relevant information is provided by the retrieved documents.
The ALCE benchmark is an attributed text-generation bench-
mark that covers three long-form Question Answering (QA)
datasets, spanning various types of question: QAMPARI [1] is
a factoid QA dataset based on Wikipedia, with answers presented
as a collection of entities; ASQA [37] is a long-form factoid QA
dataset featuring inherently ambiguous questions that necessitate
multiple concise answers to represent diverse perspectives; ELI5 [8]
consists of open-ended questions designed to be simplified for the
understanding of a five-year-old, requiring explanatory responses
spanning multiple sentences.

Lucas Albarede1, Jose G. Moreno1, Lynda Tamine1, Luce Lefeuvre2
We evaluate the effectiveness of LoDIT on each dataset using
the metrics proposed in the Trust-Align framework [36]:
•F1AC which measures the answer correctness by specifi-
cally considering questions that can be answered using the
documents in context, i.e., questions with enough relevant
information in the documents.
•F1GR , which measures the capabilities of the model to
refuse to answer a question when it cannot be answered
using the retrieved document.
•F1GC , which measures citation groundedness: the capa-
bilities of the model to output grounded citations to its
answers. This metric evaluates the context-relatedness and
groundedness conditions of attribution faithfulness defined
in § 3.1.
•TRUST , which averages the three former metrics into a
single trustworthiness score.
We refer the reader to the Trust-Align paper [ 36] for the full moti-
vation and thorough explanation of these metrics.
5.2 Baselines
We evaluate our approach, LoDIT , as well as zero-shot prompting a
Llama3.1 8B with each of our marking strategies for completeness.
We compare our result against several baselines:
•4 baselines whose results are reported from [36]:
–Llama3 8B with PostCite * [10], which first generates
answers and then retrieves documents to perform at-
tribution.
–Llama3 8B with PostAttr * [10], which first generates
answers and then uses an entailment language model
(TRUE-NLI) to perform attribution.
–Llama3 8B fine-tuned with FRONT * [12], which lever-
ages fine-grained attributions to improve citations
grounding.
–Llama3 8B fine-tuned with Trust-Align * [36], which is
specifically designed to improve models’ trustworthi-
ness.
•MIRAGE [29] with Llama 3.1 8B, which identifies sensitive
tokens in the answer and then computes the contribution
of each document to these sensitive tokens by computing
statistics with and without parts of the prompt. The official
implementation with default parameters has been used. We
included this baseline as it is a faithful attribution method
close to our work, but no results on the Trust-Align bench-
mark were available.
•The Llama3.1 8B model: (1) with Zero-Shot prompting, us-
ing the same prompt as in [ 36]; (2) with fine-tuning on the
Trust-Align training dataset, to ensure fair comparison by
training on the same LLM as well as to perform statistical
significance tests between models.
5.3 Training datasets
Our training data is composed of three sources: Trust-Align [ 36],
Hagrid [18] and ExpertsQA [25].
•TheTrust-Align dataset focuses on improving models’
trustworthiness by building upon the QAMPARI, ASQA,
and ELI5 training sets. One of the features of this dataset isthat it is focused on making the models more robust to hal-
lucination by teaching them to refuse to answer when in the
absence of relevant information in the documents. These
examples amount to approximately 25% of the training set.
•Hagrid [18] is a generative information retrieval dataset
constructed on top of the MIRACL dataset [ 44]. It contains
human-based annotations of LLM-generated answers. We
preprocess the data by only selecting examples that con-
tain attributable parts of answers. Furthermore, while this
dataset contains human-curated annotations of answers, it
does not contains answer refusal examples. We augment
Hagrid’s training dataset with 25% examples of answer
refusals, sampling questions from Hagrid and random doc-
uments from Hagrid, Trust-Align, and ExpertsQA.
•ExpertsQA [25] is a generative information retrieval
dataset that includes human expert annotations and an-
swers. Unlike traditional QA, it focuses on detailed answers,
which are more challenging to attribute correctly due to
their complexity and length. Moreover, it is constructed
across multiple domains using high-quality source docu-
ments, often from scientific or technical texts.
We preprocess the data by crafting examples that contain
attributable parts of answers. Then, similarly to the Hagrid
dataset, we generate 25% of examples where the model re-
fuses to generate due to a lack of relevant information in
the documents. To do so, we sample questions from Expert-
sQA and random documents from Hagrid, Trust-Align , and
ExpertsQA.
We make sure that each example contains at least 5 retrieved
documents. If an example contains less than 5, we add random
documents sampled from the set of all documents (from the three
datasets combined). Note that we remove explicit citations (between
brackets) from the text. In total, we obtain 20,312 training examples.
5.4 Technical considerations
Marking strategy details .Regarding the marking tokens, our
choice have been influenced by the Llama3.1 tokenizer. Our require-
ments were that: (1) tokens must be single entries in the tokenizer,
and (2) the probability of encountering them in textual data during
finetuning and evaluation is minimal. After examination, we settled
on10tokens that satisfy our requirements as document identifiers:
[“ AA”, “ BB”, “ CC”, “ DD”, “ EE”, “ FF”, “ GG”, “ HH”, “ II", “ JJ”].
When marking a document, we randomly select a unique token
identifier from this set.
To perform attribution, we aggregate the contributions across
each sentence of the generated answer. Furthermore, aligned with
the trustworthiness objective, we add a fail-safe mechanism by
replacing the answer with the refusal sentence described in Trust-
Align [36] if no attribution is induced by our attribution method.
For fair comparison, in the main result, we use the top-5 docu-
ments retrieved for each query, provided by the ALCE benchmark.
Prompt-wise, we use a similar strategy as the Trust-Align frame-
work with a refusal clause, except that we do not ask the model to
generate any citations.

Jointly Generating and Attributing Answers using Logits of Document-Identifier Tokens
Identifier token logits learning .The labels concerning the
logit values to learn are set up empirically, based on training-
validation on the Hagrid dataset [18] as follows:
•ˆ𝑙𝑘=4for document-identifier tokens in the ground truth
attribution, i.e.,{𝑖𝑑𝑘|𝑖𝑑𝑘∈𝑎𝑖}.
•ˆ𝑙𝑘=2for document-identifier tokens in the context but not
in the ground truth attribution, i.e., {𝑖𝑑𝑘|𝑖𝑑𝑘∈𝐶,𝑖𝑑𝑘∉𝑎𝑖}.
•ˆ𝑙𝑘=0for document-identifier tokens of documents ran-
domly sampled from 𝐶.
•ˆ𝑙𝑘=−2for document-identifier tokens of documents not
in𝐶i.e.,{𝑖𝑑𝑘|𝑖𝑑𝑘∉𝐶}.
Note that labels ˆ𝑙𝑘={2,4}align with groundedness , while labels
ˆ𝑙𝑘={0,−2}align with context-relatedness .
Hyperparameters settings .We train using a learning rate of
2𝑒−5with a cosine scheduler and an effective batch size of 16. We
perform training for two epochs and use the resulting models to
perform the evaluations.
The values of the 𝛼parameter (§ Eq. 1) in some of our losses
were found empirically during preliminary experiments. We found
that a value of 𝛼=0.25gave the best results for every loss.
For each of the marking strategies in LoDIT , we investigated a
static attribution threshold of 𝜙prop=3( § Eq. 5), which showed
promises during early-stage experiments. We optimized the propor-
tion threshold on the Hagrid validation set, using the average of the
F1GR and F1GC evaluation measures, as the set does not include
gold-standard sub-strings or claims (present in ASQA, QAMPARI,
and ELI5) to compute the F1AC measure. Regarding 𝜆( § Eq. 5), we
investigate three values, 0.25,0.5and0.75on validation and set 𝜆
to0.75for all results.
6 Results and analysis
6.1 Main results
Table 2 presents the results of our evaluation on ASQA, QAMPARI,
and ELI5 datasets. The table is split into baselines (top) and our
marking and models (bottom).
Comparing LoDIT with the baselines, we can see on Table 2
thatLoDIT competes with the state-of-the-art, with 𝑚𝑎𝑟𝑘𝑖𝑛𝑔𝐵𝐴
consistently and significantly improving upon Llama 3.1 8B fine-
tuned with Trust-Align (which we will refer to as ▷Trust-Align )
and⊞MIRAGE in terms of TRUST score on all datasets, showing
the overall strength of our proposal. As an example, we gain 4.76
TRUST score over▷Trust-Align on the ELI5 dataset, corresponding
to an improvement of 10.7%(48.94against 44.18). This strength is
furthermore highlighted by the significant improvements achieved
byLoDIT upon⊞MIRAGE , another approach leveraging models
internals, in terms of F1GC on all datasets by up to 56.25%(82.69
against 52.92),36.6%(41.91against 30.68), and 34.98%(53.28against
39.47) on ASQA, QAMPARI, and ELI5, respectively.
Moreover, LoDIT outperforms Llama3 8B fine-tuned with Trust-
Align , on each evaluation metric for the ASQA and ELI5 datasets.
For instance, we improve the TRUST score by 6.63%(70.86against
66.45) on the ASQA dataset, and 9.96%(48.94against 44.83) on the
ELI5 dataset. Concerning the QAMPARI dataset, we hypothesize
that the divergence in performance results from the difference inmodel used, as▷Trust-Align also shows weaker results on this
dataset.
Focusing on the Zero-Shot Prompting results, we see that using
𝑚𝑎𝑟𝑘𝑖𝑛𝑔𝐵𝐴𝑆outperforms other marking strategies on almost all
the evaluation measures, with a clear win on the F1GC metric while
the other two metrics remain similar. For instance, in the zero-
shot scenario, 𝑚𝑎𝑟𝑘𝑖𝑛𝑔𝐵𝐴𝑆improves upon the second-best making
strategy by up to 20% (16.83against 13.96) in terms of F1GC on
QAMPARI. However, analysing LoDIT , we see that 𝑚𝑎𝑟𝑘𝑖𝑛𝑔𝐵𝐴
outperforms other marking strategies, including all the Zero-Shot
prompting scenarios. This implies that a too refined marking might
nonetheless impact performance when considering fine-tuning, and
that a simpler one is more suited to the task, as highlighted by the
state-of-the-art performance of LoDIT with𝑚𝑎𝑟𝑘𝑖𝑛𝑔𝐵𝐴marking.
6.2 Ablation Study
We conduct ablation studies to evaluate the effect of our debiaising
logit strategy and proposed aggregation function Ψ(§ Eq.5). The
scenario w/o, debiaising refers to a variant of LoDIT using the ini-
tial model-output logits. Concretely, the scenario w/o ,aggregation
refers to a scenario where a variant naive aggregation is set up since
raw token-based contribution values are needed. Thus, we consider
the two naive following aggregation operators: (1) the max strategy
is defined as 𝑎𝑖={𝑖𝑑𝑘|arg max
𝑘(𝑐𝑡𝑟𝑘
𝑖𝑗)>𝜙𝑚𝑎𝑥}𝐾
𝑘=1where𝜙𝑚𝑎𝑥 is a
contribution threshold. The intuition behind max approach is that
documents should be attributed if they contribute highly to at least a
single token [ 29] and it is somehow similar to max-pooling [ 33]; (2)
theavgstrategy is defined as 𝑎𝑖={𝑖𝑑𝑘|𝑎𝑣𝑔[(𝑐𝑡𝑟𝑘
𝑖𝑗)]𝑛𝑖
𝑗=1>𝜙𝑎𝑣𝑔}𝐾
𝑘=1
where𝜙𝑎𝑣𝑔is a contribution threshold. The intuition behind avg
approach is that documents should be attributed if they contribute
highly to all tokens, which is similar to mean-pooling [ 31]. Results
of the study are presented in Table 3.
We can see overall that the absence of logit-debiasing and k-
pooling aggregations would result in a decline in performance,
thereby assessing their joint relevant contribution to the perfor-
mance of LoDIT . Particularly, we can see that the contribution of
debiaising is relevant in scenarios involving each of the marking
strategies and on each dataset, with an effect of its absence up to
−47.21%. Similarly, the ablation of our attribution strategy using
themax operator alternative deals with a range of negative effects
between 0.69%and19.16%. However, we can notice that the avgag-
gregation behaves in a dataset-dependent way. Indeed, ASQA and
ELI5 may slightly benefit from using this operator for 𝑚𝑎𝑟𝑘𝑖𝑛𝑔𝐵𝐴𝑆
and𝑚𝑎𝑟𝑘𝑖𝑛𝑔𝐴𝑊, but not QAMPARI. Moreover, this ablation does
not benefit our strongest marking strategy, namely 𝑚𝑎𝑟𝑘𝑖𝑛𝑔𝐵𝐴.
6.3 LoDIT Analysis
6.3.1 Logit debiaising. To evaluate the potential of our logit-based
debiasing stage, we explore two scenarios based on text ablation
and probability debiasing, proposed in previous work [ 3,29]: (1) the
𝑎𝑏𝑙𝑎𝑡𝑒−𝑟𝑒𝑝𝑒𝑎𝑡 scenario consists of ablating parts of the prompt
(e.g., a document) and analyzing its effect on the model genera-
tion probabilities. The intuition being that the impact of a part of
a prompt can be estimated by computing the model probability
outputs difference between scenarios when the model is prompted

Lucas Albarede1, Jose G. Moreno1, Lynda Tamine1, Luce Lefeuvre2
ASQA QAMPARI ELI5
F1AC F1GR F1GC TRUST F1AC F1GR F1GC TRUST F1AC F1GR F1GC TRUST
Baselines
Llama3 8BPostCite * [10] 32.98 53.31 28.01 38.10 6.10 34.52 8.42 16.35 20.80 45.88 8.06 24.91
PostAttr * [10] 32.98 53.31 5.95 30.75 6.10 34.52 1.64 14.09 20.80 45.88 1.25 22.64
Fine-tuned w/ FRONT * [12] 62.25 41.62 66.14 56.67 13.53 22.78 20.42 18.91 18.99 17.85 44.69 27.18
Fine-tuned w/ Trust-Align * [36] 52.35 66.06 80.95 66.45 33.85 71.11 48.01 50.99 22.57 65.06 46.85 44.83
Llama3.1 8B⊞MIRAGE [29] 59.81 64.31 52.92 59.01 1.91 63.02 23.70 29.54 29.54 55.88 39.47 41.63
Zero-Shot Prompting 59.81 64.31 60.85 61.66 1.91 63.02 21.50 28.81 29.54 52.88 43.77 42.06
▷Fine-tuned w/ Trust-Align [36] 53.64 65.33 84.52 67.83 10.69 68.83 51.05 43.53 25.52 67.09 39.91 44.18
Our markings and models
Llama3.1 8BZero-Shot Prompting𝑚𝑎𝑟𝑘𝑖𝑛𝑔𝐵𝐴 61.61 65.60 44.23 57.15 2.24 63.00 13.96 26.40 30.36 51.71 35.01 39.03
𝑚𝑎𝑟𝑘𝑖𝑛𝑔𝐵𝐴𝑆 62.29 65.85 51.02 59.72 2.51 62.38 16.83 27.24 29.51 52.71 37.17 39.80
𝑚𝑎𝑟𝑘𝑖𝑛𝑔𝐴𝑊 60.36 62.40 36.65 53.14 2.06 62.22 12.54 25.60 29.65 53.09 27.00 36.58
LoDIT𝑚𝑎𝑟𝑘𝑖𝑛𝑔𝐵𝐴 61.52▲□68.35△□82.69■70.86▲■31.73▲■69.24■41.91■47.63△■27.98△65.58■53.28▲■48.94▲■
𝑚𝑎𝑟𝑘𝑖𝑛𝑔𝐵𝐴𝑆 61.08▲63.76 71.56■65.47■28.01▲■68.24■30.68□42.27■27.50 64.45■41.96△44.14□
𝑚𝑎𝑟𝑘𝑖𝑛𝑔𝐴𝑊 54.76 63.44 81.11■66.44■26.18▲68.02□34.55■42.91■26.74 60.15□49.23▲■46.23△■
Table 2: Comparison of LoDIT performance with state-of-the-art. Models with a star (*) have their performance measures
reported from [ 36].▲and△symbols indicate significant improvements over the baseline model ▷Fine-tuned w/ Trust-Align
using a paired t-test with 𝜌=0.01and𝜌=0.05, respectively.■and□symbols indicate significant improvements over the
baseline model⊞MIRAGE using a paired t-test with 𝜌=0.01and𝜌=0.05, respectively. Scores in bold are the best TRUST scores
per dataset.
ASQA QAMPARI ELI5
𝑚𝑎𝑟𝑘𝑖𝑛𝑔𝐵𝐴 70.86 47.60 48.94
w/o debiaising 55.13↓22.20%25.13↓47.21%30.01↓38.68%
w/o aggregation+ max 67.76↓4.37%47.27↓0.69%46.65↓4.68%
+ avg 69.31↓2.19%45.55↓4.31%46.98↓4.00%
𝑚𝑎𝑟𝑘𝑖𝑛𝑔𝐵𝐴𝑆 65.73 39.39 42.54
w/o debiaising 54.64↓16.87%25.89↓34.27%31.68↓25.53%
w/o aggregation+ max 60.17↓8.46%37.53↓4.72%37.97↓10.74%
+ avg 65.80↑0.11%36.26↓7.95%42.23↓0.73%
𝑚𝑎𝑟𝑘𝑖𝑛𝑔𝐴𝑊 68.05 43.34 47.13
w/o debiaising 56.74↓16.62%26.17↓39.62%33.05↓29.87%
w/o aggregation+ max 55.01↓19.16%42.85↓1.13%43.46↓7.79%
+ avg 66.07↓2.91%40.94↓5.54%47.98↑1.80%
Table 3: Performance results (TRUST score) of the ablation
study.↓and↑represent respectively the percentage decrease
and percentage increase in TRUST score over LoDIT (w.r.t. to
each marking strategy ).
ASQA QAMPARI ELI5
TRUST Latency TRUST Latency TRUST Latency
𝑎𝑏𝑙𝑎𝑡𝑒−𝑟𝑒𝑝𝑒𝑎𝑡 50.36 771.14* 26.53 441.36* 29.99 734.20*
𝐾𝐿−𝑝𝑟𝑜𝑏𝑎 62.01 402.61 39.44 224.41 43.18 381.74
LoDIT 70.86 403.33 47.63 227.40 48.94 375.94
Table 4: Analysis of different logit debiasing strategies using
𝑚𝑎𝑟𝑘𝑖𝑛𝑔𝐵𝐴. Latency is reported in ms per query. * Note that
parallel computing of the contributions may reduce the over-
head without cancelling it completely. Bold scores indicate
the highest value for the TRUST score and lowest value for
the latency.
with and without it. The authors use this difference in probabilities
to infer which part of the prompt has to be attributed to the gener-
ated answer. In this case, the contribution calculation is redefined
as the log difference [ 3] between using and not using the context 𝐶:
𝑐𝑡𝑟𝑘
𝑖𝑗=𝑙𝑜𝑔M(𝑖𝑑𝑘|𝑞,𝐴𝑆<𝑖,𝑆𝑖<𝑗)−𝑙𝑜𝑔M(𝑖𝑑𝑘|𝑞,𝐴𝑆<𝑖,𝑆𝑖<𝑗,𝐶)(6)(2) the𝐾𝐿−𝑝𝑟𝑜𝑏𝑎 scenario consists of debiasing the token gen-
eration probabilities instead of their logits. We explore the KL-
divergence distillation loss [11]:
L𝐾𝐿
𝑎𝑡𝑡=∑︁
𝑖𝑑𝑘∈𝑎𝑖𝐾𝐿(𝑝ˆ𝑙𝑘||𝑝(𝑖𝑑𝑘|𝑞,𝐴𝑆<𝑖,𝑆𝑖<𝑗,𝐶,M)) (7)
where𝑝ˆ𝑙𝑘is the generation probability associated with logit ˆ𝑙𝑘.
Contributions are then computed using Eq. 4.
Since latency overhead might be introduced by the scenarios,
we report the average time in milliseconds used to perform a query
on each evaluation dataset. Results in terms of performance and
latency are presented in Table 4, for both scenarios, as well as
LoDIT using𝑚𝑎𝑟𝑘𝑖𝑛𝑔𝐵𝐴strategy. We can see that the 𝐾𝐿−𝑝𝑟𝑜𝑏𝑎
scenario outperforms the 𝑎𝑏𝑙𝑎𝑡𝑒−𝑟𝑒𝑝𝑒𝑎𝑡 scenario in both metrics
for the three datasets. This result indicates that the 𝑎𝑏𝑙𝑎𝑡𝑒−𝑟𝑒𝑝𝑒𝑎𝑡
can be easily outperformed, in terms of quality of the attributions
but also in terms of latency, as indicated by an almost doubling of
the latency on all the datasets, as well as drops in terms of TRUST
score of 28.93%(70.86vs.50.36),44.29%(47.63vs.26.53) and 38.72%
(48.94against 29.99) on ASQA, QAMPARI and ELI5 when compared
with LoDIT . However, when comparing the logit-based strategy
used in LoDIT and𝐾𝐿−𝑝𝑟𝑜𝑏𝑎 , there are no strong differences
in terms of latency, but TRUST performance scores are clearly
different, with our logit-based strategy outperforming in the three
datasets, highlighted by improvements of 14.27%(70.86vs.62.01),
20.76%(47.63vs.39.44), and 13.33%(48.94against 43.18) on ASQA,
QAMPARI, and ELI5, respectively.
6.3.2 Robustness analysis of LoDIT .Since LoDIT is based on
marked documents in the context, we analyse how impactful the
number and order of these documents in the context are on its
performance.
Context length analysis .Figure 3 shows our robustness anal-
ysis regarding the number of documents used in the context. We
perform evaluations using the 𝑚𝑎𝑟𝑘𝑖𝑛𝑔𝐵𝐴strategy with the top 𝐾
documents with 𝐾in the range[2,10]by step of 2. As a reminder,

Jointly Generating and Attributing Answers using Logits of Document-Identifier Tokens
2 4 5 6 8 1040506070
Length of context in terms of number of documents (K)TRUST scoreASQA QAMPARI ELI5
Figure 3: Robustness of LoDIT to the length of context.
LoDIT is trained using 5top-retrieved documents in the context.
Unsurprisingly, we notice that for each dataset, the best performing
configuration is using 5documents, with performance dropping as
we increase or decrease the number of documents. These results
show that the best performance is directly linked to the configura-
tions seen during training, and encourage a more diverse training
procedure to increase robustness. Nonetheless, LoDIT remains
robust to slight changes in the number of documents as highlighted
by drops of TRUST score of only 0.63%and3.10%on ASQA, 5.26%
and3.21%on QAMPARI, 3.28%and5.29%on ELI5 when comparing
using 5 documents to using 6 and 4, respectively.
Document order analysis .Figure 4 shows the performance
results of different configurations regarding the order of documents
in the context as well as the choice of identifier tokens during in-
ference. More precisely, it highlights the TRUST performance score
(on the test data) at different steps of the training. We define Rand ,
as the configuration where we randomly assign identifier tokens
to documents during the marking step, and Alph , as the scenario
where identifier tokens are alphabetically ordered during marking
(first “𝐴𝐴”, then “𝐵𝐵”, etc.). We define Vanilla as the traditional
document ranking of documents in the context based on their rel-
evance to the input query, and Rev, which is the reverse of the
relevance-based ranking. We analyse each combination of token
marking selection and document ordering configuration to evalu-
ate their joint effect on performance (e.g., Rand-Vanilla , which is
the “default” configuration of LoDIT ). By comparing Alph-Vanilla
with Rand-Vanilla andAlph-Rev with Rand-Rev , we can conclude
from Figure 4 that LoDIT is robust to the choice of identifier to-
kens as early as 40% of the first epochs for each dataset. That is
explained by the fact that LoDIT is trained using random choosing
of identifier tokens and the alphabetical ordering simply being a
special case of random ordering. Furthermore, by analyzing the
order of documents (i.e., by comparing Alph-Vanilla with Alph-Rev
and Rand-Vanilla with Rand-Rev ), we first see that ordering the
documents in reverse of relevance-based ranking slightly improves
the performance during the early stages of training. Second, we
see that this improvement reduces as training progresses, with per-
formance in all the evaluation configurations converging during
the late stage of training. This observation highlights that LoDIT
learns to more accurately consider documents regardless of their
position in the context, which advocates for its robustness.0.40.60.811.21.41.61.826465666768697071
Training epochsTRUST scoreASQA
0.40.60.811.21.41.61.824244464850
Training epochsTRUST scoreQAMPARI
0.40.60.811.21.41.61.8238404244464850
Training epochsTRUST scoreELI5
Figure 4: Robustness of LoDIT to document ordering and
identifier token selection. Rand-Vanilla (purple) corresponds
to the default configuration of our 𝑚𝑎𝑟𝑘𝑖𝑛𝑔𝐵𝐴strategy.
6.3.3 Proportion threshold analysis. The aggregation operator uses
the hyperparameter, 𝜆(§ Eq.5), representing the proportion of state-
ments to be considered to perform attribution. While we set this
hyperparameter to 𝜆=0.75for all experiments after an optimiza-
tion on the Trust-Align test set, we analyze here different values of
this parameter and their impact of the TRUST score.
Table 5 shows the TRUST score of LoDIT with the𝑚𝑎𝑟𝑘𝑖𝑛𝑔𝐵𝐴
strategy on the evaluation datasets for three values of 𝜆. We notice
that for all datasets, the TRUST score is only slightly impacted
with respect to 𝜆. We can see that the TRUST score loss percentage
between𝜆=0.75and𝜆=0.25is2.35%(70.86against 69.19),3.40%
(47.63against 46.01) and 1.63% (48.94against 48.14) for ASQA,
QAMPARI and ELI5 respectively. We argue that these results show
the confidence of LoDIT in its debiaising approach, giving “high”
logit values to the same identifier tokens for each generated token in
the statement. In other words, if the logits’ values of the identifier
tokens are similar at each step of the generation, the effects of
considering whether 25%,50%or75%of these values are higher
than a threshold is negated.
ProportionASQA QAMPARI ELI5Threshold ( 𝜆)
0.25 69.19 46.01 48.14
0.50 69.85 46.78 49.16
0.75 70.86 47.63 48.94
Table 5: Impact of the proportion threshold on the Trust
score when using the 𝑚𝑎𝑟𝑘𝑖𝑛𝑔𝐵𝐴strategy.
7 Conclusion
In this work, we propose leveraging identifiers of documents in a
retrieved context to faithfully attribute LLM-generated answers us-
ing a RAG framework. To bridge the gap between model generation
and attribution, we propose a two-level attribution method, called
LoDIT , where the document identifier logit is used to estimate
token-level contributions, which are then aggregated to compute
the statement-level attribution. Extensive experiments using the
recent Trust-Align evaluation framework validate the effectiveness,
efficiency, and robustness of LoDIT .
An interesting avenue of research involves extending LoDIT to-
ward an end-to-end learnable token-level and statement-level attri-
bution. Additionally, investigating how our logit-based approach
could be extended to answer token selection and mitigate other
types of bias would allow us to enhance the fairness of model
outputs, beyond faithfulness and trustworthiness.

Lucas Albarede1, Jose G. Moreno1, Lynda Tamine1, Luce Lefeuvre2
References
[1]Samuel Amouyal, Tomer Wolfson, Ohad Rubin, Ori Yoran, Jonathan Herzig,
and Jonathan Berant. QAMPARI: A benchmark for open-domain questions
with many answers. In Sebastian Gehrmann, Alex Wang, João Sedoc, Elizabeth
Clark, Kaustubh Dhole, Khyathi Raghavi Chandu, Enrico Santus, and Hooman
Sedghamiz, editors, Proceedings of the Third Workshop on Natural Language
Generation, Evaluation, and Metrics (GEM) , pages 97–110, Singapore, December
2023. Association for Computational Linguistics.
[2] Orlando Ayala and Patrice Bechard. Reducing hallucination in structured out-
puts via retrieval-augmented generation. In Yi Yang, Aida Davani, Avi Sil, and
Anoop Kumar, editors, Proceedings of the 2024 Conference of the North American
Chapter of the Association for Computational Linguistics: Human Language Tech-
nologies (Volume 6: Industry Track) , pages 228–238, Mexico City, Mexico, June
2024. Association for Computational Linguistics.
[3] Benjamin Cohen-Wang, Harshay Shah, Kristian Georgiev, and Aleksander Mądry.
Contextcite: Attributing model generation to context. In A. Globerson, L. Mackey,
D. Belgrave, A. Fan, U. Paquet, J. Tomczak, and C. Zhang, editors, Advances in
Neural Information Processing Systems , volume 37, pages 95764–95807. Curran
Associates, Inc., 2024.
[4] Sunhao Dai, Chen Xu, Shicheng Xu, Liang Pang, Zhenhua Dong, and Jun Xu. Bias
and unfairness in information retrieval systems: New challenges in the llm era.
InProceedings of the 30th ACM SIGKDD Conference on Knowledge Discovery and
Data Mining , KDD ’24, page 6437–6447, New York, NY, USA, 2024. Association
for Computing Machinery.
[5]Xuan-Quy Dao and Ngoc-Bich Le. Chatgpt is good but bing chat is better for
vietnamese students. arXiv preprint arXiv:2307.08272 , 2023.
[6]Qiang Ding, Lvzhou Luo, Yixuan Cao, and Ping Luo. Attention with de-
pendency parsing augmentation for fine-grained attribution. arXiv preprint
arXiv:2412.11404 , 2024.
[7]Hanane Djeddal, Pierre Erbacher, Raouf Toukal, Laure Soulier, Karen Pinel-
Sauvagnat, Sophia Katrenko, and Lynda Tamine. An evaluation framework
for attributed information retrieval using large language models. In Edoardo
Serra and Francesca Spezzano, editors, Proceedings of the 33rd ACM International
Conference on Information and Knowledge Management, CIKM 2024, Boise, ID,
USA, October 21-25, 2024 , pages 5354–5359. ACM, 2024.
[8]Angela Fan, Yacine Jernite, Ethan Perez, David Grangier, Jason Weston, and
Michael Auli. ELI5: Long form question answering. In Anna Korhonen, David
Traum, and Lluís Màrquez, editors, Proceedings of the 57th Annual Meeting of the
Association for Computational Linguistics , pages 3558–3567, Florence, Italy, July
2019. Association for Computational Linguistics.
[9] Luyu Gao, Zhuyun Dai, Panupong Pasupat, Anthony Chen, Arun Tejasvi Cha-
ganty, Yicheng Fan, Vincent Zhao, Ni Lao, Hongrae Lee, Da-Cheng Juan, and
Kelvin Guu. RARR: Researching and revising what language models say, using
language models. In Anna Rogers, Jordan Boyd-Graber, and Naoaki Okazaki, ed-
itors, Proceedings of the 61st Annual Meeting of the Association for Computational
Linguistics (Volume 1: Long Papers) , pages 16477–16508, Toronto, Canada, July
2023. Association for Computational Linguistics.
[10] Tianyu Gao, Howard Yen, Jiatong Yu, and Danqi Chen. Enabling large language
models to generate text with citations. In Houda Bouamor, Juan Pino, and Kalika
Bali, editors, Proceedings of the 2023 Conference on Empirical Methods in Natural
Language Processing , pages 6465–6488, Singapore, December 2023. Association
for Computational Linguistics.
[11] Geoffrey Hinton, Oriol Vinyals, and Jeff Dean. Distilling the knowledge in a
neural network. arXiv preprint arXiv:1503.02531 , 2015.
[12] Lei Huang, Xiaocheng Feng, Weitao Ma, Yuxuan Gu, Weihong Zhong, Xiachong
Feng, Weijiang Yu, Weihua Peng, Duyu Tang, Dandan Tu, and Bing Qin. Learning
fine-grained grounded citations for attributed large language models. In Lun-Wei
Ku, Andre Martins, and Vivek Srikumar, editors, Findings of the Association for
Computational Linguistics: ACL 2024 , pages 14095–14113, Bangkok, Thailand,
August 2024. Association for Computational Linguistics.
[13] Lei Huang, Xiaocheng Feng, Weitao Ma, Liang Zhao, Yuchun Fan, Weihong
Zhong, Dongliang Xu, Qing Yang, Hongtao Liu, and Bing Qin. Advancing
large language model attribution through self-improving. In Yaser Al-Onaizan,
Mohit Bansal, and Yun-Nung Chen, editors, Proceedings of the 2024 Conference
on Empirical Methods in Natural Language Processing , pages 3822–3836, Miami,
Florida, USA, November 2024. Association for Computational Linguistics.
[14] Gautier Izacard, Patrick Lewis, Maria Lomeli, Lucas Hosseini, Fabio Petroni,
Timo Schick, Jane Dwivedi-Yu, Armand Joulin, Sebastian Riedel, and Edouard
Grave. Atlas: few-shot learning with retrieval augmented language models. J.
Mach. Learn. Res. , 24(1), January 2023.
[15] Bowen Jiang, Yangxinyu Xie, Zhuoqun Hao, Xiaomeng Wang, Tanwi Mallick,
Weijie J Su, Camillo Jose Taylor, and Dan Roth. A peek into token bias: Large
language models are not yet genuine reasoners. In Yaser Al-Onaizan, Mohit
Bansal, and Yun-Nung Chen, editors, Proceedings of the 2024 Conference on Em-
pirical Methods in Natural Language Processing , pages 4722–4756, Miami, Florida,
USA, November 2024. Association for Computational Linguistics.[16] Xiaoqi Jiao, Yichun Yin, Lifeng Shang, Xin Jiang, Xiao Chen, Linlin Li, Fang Wang,
and Qun Liu. TinyBERT: Distilling BERT for natural language understanding.
In Trevor Cohn, Yulan He, and Yang Liu, editors, Findings of the Association for
Computational Linguistics: EMNLP 2020 , pages 4163–4174, Online, November
2020. Association for Computational Linguistics.
[17] Nal Kalchbrenner, Edward Grefenstette, and Phil Blunsom. A convolutional
neural network for modelling sentences. In Kristina Toutanova and Hua Wu,
editors, Proceedings of the 52nd Annual Meeting of the Association for Computa-
tional Linguistics (Volume 1: Long Papers) , pages 655–665, Baltimore, Maryland,
June 2014. Association for Computational Linguistics.
[18] Ehsan Kamalloo, Aref Jafari, Xinyu Zhang, Nandan Thakur, and Jimmy Lin.
HAGRID: A human-llm collaborative dataset for generative information-seeking
with attribution. arXiv:2307.16883 , 2023.
[19] Taehyeon Kim, Jaehoon Oh, Nak Yil Kim, Sangwook Cho, and Se-Young Yun.
Comparing kullback-leibler divergence and mean squared error loss in knowl-
edge distillation. In Zhi-Hua Zhou, editor, Proceedings of the Thirtieth Inter-
national Joint Conference on Artificial Intelligence, IJCAI-21 , pages 2628–2635.
International Joint Conferences on Artificial Intelligence Organization, 8 2021.
Main Track.
[20] Kalpesh Krishna, Aurko Roy, and Mohit Iyyer. Hurdles to progress in long-form
question answering. In Kristina Toutanova, Anna Rumshisky, Luke Zettle-
moyer, Dilek Hakkani-Tur, Iz Beltagy, Steven Bethard, Ryan Cotterell, Tanmoy
Chakraborty, and Yichao Zhou, editors, Proceedings of the 2021 Conference of
the North American Chapter of the Association for Computational Linguistics:
Human Language Technologies , pages 4940–4957, Online, June 2021. Association
for Computational Linguistics.
[21] Anson Li, Renee Shrestha, Thinoj Jegatheeswaran, Hannah O Chan, Colin Hong,
and Rakesh Joshi. Mitigating hallucinations in large language models: A compar-
ative study of rag-enhanced vs. human-generated medical templates. medRxiv ,
pages 2024–09, 2024.
[22] Chin-Yew Lin. ROUGE: A package for automatic evaluation of summaries. In
Text Summarization Branches Out , pages 74–81, Barcelona, Spain, July 2004.
Association for Computational Linguistics.
[23] Nelson Liu, Tianyi Zhang, and Percy Liang. Evaluating verifiability in generative
search engines. In Houda Bouamor, Juan Pino, and Kalika Bali, editors, Findings
of the Association for Computational Linguistics: EMNLP 2023 , pages 7001–7025,
Singapore, December 2023. Association for Computational Linguistics.
[24] Varun Magesh, Faiz Surani, Matthew Dahl, Mirac Suzgun, Christopher D Man-
ning, and Daniel E Ho. Hallucination-free? assessing the reliability of leading ai
legal research tools. Journal of Empirical Legal Studies , 2024.
[25] Chaitanya Malaviya, Subin Lee, Sihao Chen, Elizabeth Sieber, Mark Yatskar,
and Dan Roth. ExpertQA: Expert-curated questions and attributed answers. In
Kevin Duh, Helena Gomez, and Steven Bethard, editors, Proceedings of the 2024
Conference of the North American Chapter of the Association for Computational
Linguistics: Human Language Technologies (Volume 1: Long Papers) , pages 3025–
3045, Mexico City, Mexico, June 2024. Association for Computational Linguistics.
[26] Jacob Menick, Maja Trebacz, Vladimir Mikulik, John Aslanides, Francis Song,
Martin Chadwick, Mia Glaese, Susannah Young, Lucy Campbell-Gillingham, Ge-
offrey Irving, and Nat McAleese. Teaching language models to support answers
with verified quotes, 2022.
[27] Kishore Papineni, Salim Roukos, Todd Ward, and Wei-Jing Zhu. Bleu: a method
for automatic evaluation of machine translation. In Proceedings of the 40th annual
meeting of the Association for Computational Linguistics , pages 311–318, 2002.
[28] Anirudh Phukan, Shwetha Somasundaram, Apoorv Saxena, Koustava Goswami,
and Balaji Vasan Srinivasan. Peering into the mind of language models: An ap-
proach for attribution in contextual question answering. In Lun-Wei Ku, Andre
Martins, and Vivek Srikumar, editors, Findings of the Association for Computa-
tional Linguistics: ACL 2024 , pages 11481–11495, Bangkok, Thailand, August 2024.
Association for Computational Linguistics.
[29] Jirui Qi, Gabriele Sarti, Raquel Fernández, and Arianna Bisazza. Model internals-
based answer attribution for trustworthy retrieval-augmented generation. In
Proceedings of the 2024 Conference on Empirical Methods in Natural Language
Processing , page 6037–6053. Association for Computational Linguistics, 2024.
[30] Ori Ram, Yoav Levine, Itay Dalmedigos, Dor Muhlgay, Amnon Shashua, Kevin
Leyton-Brown, and Yoav Shoham. In-context retrieval-augmented language
models. Transactions of the Association for Computational Linguistics , 11:1316–
1331, 2023.
[31] Nils Reimers and Iryna Gurevych. Sentence-BERT: Sentence embeddings using
Siamese BERT-networks. In Kentaro Inui, Jing Jiang, Vincent Ng, and Xiaojun
Wan, editors, Proceedings of the 2019 Conference on Empirical Methods in Nat-
ural Language Processing and the 9th International Joint Conference on Natural
Language Processing (EMNLP-IJCNLP) , pages 3982–3992, Hong Kong, China,
November 2019. Association for Computational Linguistics.
[32] Abhilasha Sancheti, Koustava Goswami, and Balaji Srinivasan. Post-hoc answer
attribution for grounded and trustworthy long document comprehension: Task,
insights, and challenges. In Danushka Bollegala and Vered Shwartz, editors,
Proceedings of the 13th Joint Conference on Lexical and Computational Seman-
tics (*SEM 2024) , pages 49–57, Mexico City, Mexico, June 2024. Association for

Jointly Generating and Attributing Answers using Logits of Document-Identifier Tokens
Computational Linguistics.
[33] Dinghan Shen, Guoyin Wang, Wenlin Wang, Martin Renqiang Min, Qinliang Su,
Yizhe Zhang, Chunyuan Li, Ricardo Henao, and Lawrence Carin. Baseline needs
more love: On simple word-embedding-based models and associated pooling
mechanisms. In Iryna Gurevych and Yusuke Miyao, editors, Proceedings of the
56th Annual Meeting of the Association for Computational Linguistics (Volume 1:
Long Papers) , pages 440–450, Melbourne, Australia, July 2018. Association for
Computational Linguistics.
[34] Kurt Shuster, Spencer Poff, Moya Chen, Douwe Kiela, and Jason Weston. Retrieval
augmentation reduces hallucination in conversation. In Marie-Francine Moens,
Xuanjing Huang, Lucia Specia, and Scott Wen-tau Yih, editors, Findings of the
Association for Computational Linguistics: EMNLP 2021 , pages 3784–3803, Punta
Cana, Dominican Republic, November 2021. Association for Computational
Linguistics.
[35] Aviv Slobodkin, Eran Hirsch, Arie Cattan, Tal Schuster, and Ido Dagan. Attribute
first, then generate: Locally-attributable grounded text generation. In Lun-
Wei Ku, Andre Martins, and Vivek Srikumar, editors, Proceedings of the 62nd
Annual Meeting of the Association for Computational Linguistics (Volume 1: Long
Papers) , pages 3309–3344, Bangkok, Thailand, August 2024. Association for
Computational Linguistics.
[36] Maojia Song, Shang Hong Sim, Rishabh Bhardwaj, Hai Leong Chieu, Navonil
Majumder, and Soujanya Poria. Measuring and enhancing trustworthiness of llms
in rag through grounded attributions and learning to refuse. In The Thirteenth
International Conference on Learning Representations , 2025.
[37] Ivan Stelmakh, Yi Luan, Bhuwan Dhingra, and Ming-Wei Chang. ASQA: Factoid
questions meet long-form answers. In Yoav Goldberg, Zornitsa Kozareva, and
Yue Zhang, editors, Proceedings of the 2022 Conference on Empirical Methods in
Natural Language Processing , pages 8273–8288, Abu Dhabi, United Arab Emirates,
December 2022. Association for Computational Linguistics.
[38] Jonas Wallat, Maria Heuss, Maarten de Rijke, and Avishek Anand. Correctness
is not faithfulness in rag attributions. arXiv preprint arXiv:2412.18004 , 2024.
[39] Sheng-Lun Wei, Cheng-Kuang Wu, Hen-Hsen Huang, and Hsin-Hsi Chen. Un-
veiling selection biases: Exploring order and token sensitivity in large language
models. In Lun-Wei Ku, Andre Martins, and Vivek Srikumar, editors, Findings
of the Association for Computational Linguistics: ACL 2024 , pages 5598–5621,
Bangkok, Thailand, August 2024. Association for Computational Linguistics.
[40] Sirui Xia, Xintao Wang, Jiaqing Liang, Yifei Zhang, Weikang Zhou, Jiaji Deng, Fei
Yu, and Yanghua Xiao. Ground every sentence: Improving retrieval-augmentedLLMs with interleaved reference-claim generation. In Luis Chiruzzo, Alan Ritter,
and Lu Wang, editors, Findings of the Association for Computational Linguistics:
NAACL 2025 , pages 969–988, Albuquerque, New Mexico, April 2025. Association
for Computational Linguistics.
[41] Shicheng Xu, Liang Pang, Huawei Shen, Xueqi Cheng, and Tat-Seng Chua.
Search-in-the-chain: Interactively enhancing large language models with search
for knowledge-intensive tasks. In Proceedings of the ACM Web Conference 2024 ,
WWW ’24, page 1362–1373, New York, NY, USA, 2024. Association for Computing
Machinery.
[42] Xiang Yue, Boshi Wang, Ziru Chen, Kai Zhang, Yu Su, and Huan Sun. Automatic
evaluation of attribution by large language models. In Houda Bouamor, Juan
Pino, and Kalika Bali, editors, Findings of the Association for Computational Lin-
guistics: EMNLP 2023 , pages 4615–4635, Singapore, December 2023. Association
for Computational Linguistics.
[43] Tianyi Zhang*, Varsha Kishore*, Felix Wu*, Kilian Q. Weinberger, and Yoav Artzi.
Bertscore: Evaluating text generation with bert. In International Conference on
Learning Representations , 2020.
[44] Xinyu Zhang, Nandan Thakur, Odunayo Ogundepo, Ehsan Kamalloo, David
Alfonso-Hermelo, Xiaoguang Li, Qun Liu, Mehdi Rezagholizadeh, and Jimmy
Lin. Miracl: A multilingual retrieval dataset covering 18 diverse languages.
Transactions of the Association for Computational Linguistics , 11:1114–1131, 09
2023.
[45] Chujie Zheng, Hao Zhou, Fandong Meng, Jie Zhou, and Minlie Huang. Large
language models are not robust multiple choice selectors. In The Twelfth Inter-
national Conference on Learning Representations , 2024.
[46] Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu,
Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric P. Xing, Hao Zhang,
Joseph E. Gonzalez, and Ion Stoica. Judging llm-as-a-judge with mt-bench and
chatbot arena. In Proceedings of the 37th International Conference on Neural
Information Processing Systems , NIPS ’23, Red Hook, NY, USA, 2023. Curran
Associates Inc.
[47] Shengyao Zhuang, Honglei Zhuang, Bevan Koopman, and Guido Zuccon. A
setwise approach for effective and highly efficient zero-shot ranking with large
language models. In Proceedings of the 47th International ACM SIGIR Conference
on Research and Development in Information Retrieval , SIGIR 2024, page 38–47.
ACM, July 2024.