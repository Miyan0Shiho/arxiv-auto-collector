# A Dialectic Pipeline for Improving LLM Robustness

**Authors**: Sara Candussio

**Published**: 2026-01-28 14:42:49

**PDF URL**: [https://arxiv.org/pdf/2601.20659v1](https://arxiv.org/pdf/2601.20659v1)

## Abstract
Assessing ways in which Language Models can reduce their hallucinations and improve the outputs' quality is crucial to ensure their large-scale use.
  However, methods such as fine-tuning on domain-specific data or the training of a separate \textit{ad hoc} verifier require demanding computational resources (not feasible for many user applications) and constrain the models to specific fields of knowledge.
  In this thesis, we propose a dialectic pipeline that preserves LLMs' generalization abilities while improving the quality of its answer via self-dialogue, enabling it to reflect upon and correct tentative wrong answers.
  We experimented with different pipeline settings, testing our proposed method on different datasets and on different families of models. All the pipeline stages are enriched with the relevant context (in an oracle-RAG setting) and a study on the impact of its summarization or its filtering is conducted.
  We find that our proposed dialectic pipeline is able to outperform by significative margins the standard model answers and that it consistently achieves higher performances than Chain-of-Thought only prompting.

## Full Text


<!-- PDF content starts -->

Università degli Studi di Trieste
Dipartimento di Matematica, Informatica e Geoscienze
Corso di Laurea Magistrale in
Data Science and Scientific Computing
TESI DI LAUREA MAGISTRALE
A Dialectic Pipeline for Improving LLM
Robustness
Relatore:
Prof. Luca Bortolussi
Correlatore:
Dott. Gabriele SartiCandidata:
Sara Candussio
Anno Accademico 2023/2024arXiv:2601.20659v1  [cs.CL]  28 Jan 2026

L’inferno dei viventi non è qualcosa che sarà; se ce n’è
uno, è quello che è già qui, l’inferno che abitiamo tutti i
giorni, che formiamo stando insieme.
Due modi ci sono per non soffrirne.
Il primo riesce facile a molti: accettare l’inferno e
diventarne parte fino al punto di non vederlo più.
Il secondo è rischioso ed esige attenzione e
apprendimento continui:
cercare e saper riconoscere chi e cosa, in mezzo
all’inferno, non è inferno, e farlo durare, e dargli spazio.
Le Città Invisibili, Italo Calvino

Abstract
Assessing ways in which Language Models can reduce their hallucinations and im-
prove the outputs’ quality is crucial to ensure their large-scale use.
However, methods such as fine-tuning on domain-specific data or the training of
a separatead hocverifier require demanding computational resources (not feasible for
many user applications) and constrain the models to specific fields of knowledge.
In this thesis, we propose a dialectic pipeline that preserves LLMs’ generalization
abilities while improving the quality of its answer via self-dialogue, enabling it to re-
flect upon and correct tentative wrong answers.
We experimented with different pipeline settings, testing our proposed method on
different datasets and on different families of models. All the pipeline stages are en-
riched with the relevant context (in an oracle-RAG setting) and a study on the impact
of its summarization or its filtering is conducted.
We find that our proposed dialectic pipeline is able to outperform by significative
margins the standard model answers and that it consistently achieves higher perfor-
mances than Chain-of-Thought only prompting.
Sommario
Al fine di garantire l’uso su larga scala del Language Models, è fondamentale trovare
delle strategie con cui è possibile ridurre le loro allucinazioni e quindi migliorare la
qualità dei loro outputs.
Tuttavia, metodi come il fine-tuning su dati specifici o l’addestramento di verifiers
ad hocrichiedono risorse computazionali elevate (non accessibili per molte applicazioni
utili all’utente) e limitano i modelli ad alcuni precisi ambiti di conoscenza.
In questa tesi, proponiamo una pipeline dialettica che preserva le capacità di gener-
alizzazione dei LLMs migliorando al contempo la qualità delle loro risposte attraverso
l’auto-dialogo, consentendo al modello di riflettere e correggere risposte inizialmente
errate.
Abbiamo sperimentato diverse configurazioni di auto-dialogo, testando il nostro
metodo su vari dataset e su diverse famiglie di Language Models. Tutte le fasi di
questa pipeline dialettica sono state arricchite con il contesto rilevante per rispondere
al prompt (in un setting oracle-RAG) ed è stato condotto uno studio sull’impatto del
riassunto o del filtraggio di questo.
Abbiamo riscontrato che la pipeline dialettica proposta è in grado di superare con
margini significativi le risposte standard del modello e che riesce ad ottenere in maniera
consistente prestazioni superiori rispetto al solo prompting con Chain-of-Thought.
ii

Chapter 1
Introduction
In 2017, the field of Natural Language Processing had been revolutionized by Vaswani
et al. [66], leading to a mass-scale interest towards neural networks applied to text
generation. The era of Large Language Models (LLMs) began with the intuition that a
recurrent structure to is not mandatory to face the given task; what is necessary is just
theattention mechanism(and a supporting complex model structure, as we will show in
section 2.1.3).
Many efforts were made by the researchers’ community from that moment on in
order to improve the quality of the produced text: these models were often object
of the so-calledhallucinations, i.e. the generation of false or misleading information.
This phenomenon is due to many reasons, often caused by data issues: incomplete,
noisy, biased or not updated training data tend to drive the auto-regressive towards
the wrong output. Additionally, an unusual prompting method or an imprecise in-
struction could confuse the model. Different strategies are used nowadays to reduce
thehallucinationsfrequency in LLMs: curated pre-training datasets and fine-tuning,
RAG-powered applications (Section 2.4), human-based alignment, Chain-of-Thought
prompting (2.6) are all aimed at improving the models’ prediction quality.
We propose a new solution to face this problem, consisting in prompting the model
to reason multiple times on the answer to a question before definitively choosing the
correct option. Our approach aims at improving the quality of thespontaneousanswer
by incrementally checking whether and why it is correct. This approach constructs a
dialectic pipelinefor generating the final output, where the candidate answer is checked
and examined by two steps before being effectively chosen. The termdialecticis due to
the fact that the model is dialoguing with itself in three different stages: the candidate
answer production, the first comment on the correctness of it, and the final decision
on which should be the final answer. We will refer to the first stage as thethesis, to the
second asantithesisand to the latter assynthesis. Differently from the Hegelian dialectic
[75], theantithesisstep is not forced to refute thethesis’ one, neither is thesynthesisto
compromise between the other two. However, we find these names as representative
for the role that they have in the process: thethesismakes a first guess, theantithesis
has to check whether or not it is correct (and why), and thesynthesisfurther merges
these opinions into a final, reasoned answer.
We are going to test this method onmulti-hopquestion-answering datasets, namely
1

on tasks that require to properly merge multiple knowledge sources to answer the
question. This kind of problems requires both content extraction and reasoning abili-
ties, thus are more challenging than standard RAG-problems. We are going to consider
different sub-tasks inside this broader class, consisting of multiple passages (3.3) to ex-
ploit in slightly different ways, for example by comparing them or by constructing
links between them (3.1.1). In all the pipeline steps, we allow the model to access the
relevant context in order to properly answer to the question while dealing with (im-
plicit) challenging context pre-processing.
The research questions for which we search an answer through our experiments
are oriented to assess whether or not thisdialecticpipeline can compete against well-
established other methods in terms of accuracy and reliability (5.6). We want to test
whether the pipeline is able to reach good accuracy values and whether this is main-
tained when we consider different models inside the pipeline. We test both the robust-
ness across different families of models (2.3.1, 2.3.2, 2.3.3) and with respect to the same
model but with different sizes (5.2).
We also try this pipeline on different datasets to assess the non-specificity of the
proposed method: we will consider twomulti-hopdatasets with various numbers of
hopsand requiring different reasoning strategies to merge them (5.3). We are going
to define some pipelinevariationsthat induce the models to behave differently when
asked to solve a task (4.6.2).
We check whether thesynthesisstep is necessary to improve the models’ perfor-
mances and by which margin; this is due to the idea that theantithesisstep could be
misleaded by the context or simply wrong. Consequently, we run experiments compar-
ing theantithesis’ proposed answer and thesynthesis’ ones, in order to assess whether
the latter is in fact useful in prediction terms (5.1).
Finally, we test whether thewayin which the context is provided to the pipeline af-
fects the answers’ quality: works such as S2A (2.9) and RE-RAG (2.8.1) spot a light on
the beneficial effect of carefully selecting only the relevant parts of it and use only these
to generate the output text for a given prompt. Consequently, we compare the answers
obtained by passing the original, the summarized (4.3.1) and the filtered context to the
pipeline. The filtering approach exploits a gradient-based attribution method (2.11.1)
that is used to select the passages’ sentences that are found to be influential for the
output generation (4.3.2).
Our work is structured as follows.
• The first section (2) is dedicated to a literature overview. It delves into the tech-
nical details of Transformer models and explains the differences among those we
will consider in our experiments. Then it focuses on the existing approaches to
overcome the problem ofhallucinations, such as SFT and RAG (2.4), training log-
ical verifiers to check the correctness of the answer before outputting it (2.8.2)
and prompting strategies aimed at showing the expected behaviour of the model
before executing it. We consequently introduce possible alternatives to our ap-
proach, such as Chain-of-Thought prompting (2.6) and self-refinement methods
(2.10). Finally, we inspect ways in which the context can improve the answers’
2

quality (2.8, 2.11).
• We then present the data (3) and highlight the pre-processing used to make these
multi-hopdatasets also multiple choice: this is done to avoid approximate, LLM-
based evaluations on the correctness of the answer.
• The methods section (4) explains largely how we constructed the pipeline (4.2),
how practically we filtered (4.3.2) and summarized (4.3.1) the context and how
we assessed performances (4.4, 4.8).
• Finally, we run answer the research question described before in section (5), de-
creeing whether or not the proposed method works and in which cases it per-
forms better or worse. Final conclusions are drawn in section (6).
3

Chapter 2
Literature review
In this section we go through a set of relevant works for our study. We begin by study-
ing how the models that we use are structured, architecturally speaking; we then high-
light the differences between the classes of LLMs exploited for our experiments.
Subsequently, we provide an overview of methods aimed at improving the output
quality in terms of factuality, style and consistency with the given examples.
Finally, we highlight how these discoveries and proposed methods are related to
our new approach, and how much it owes to other researchers’ work.
2.1 Transformer models
Language Models (LMs) are inherently sequential, processing input data in a step-by-
step manner. Early LMs leveraged this sequential structure through Recurrent Neural
Network (RNN) architectures, which were designed to capture dependencies in data
sequences. However, RNNs faced limitations in scalability due to their reliance on
sequential processing and their fixed-size context representations. To address these
challenges, theattention mechanismwas introduced, providing a more flexible way to
handle context by allowing models to focus on different parts of the input sequence as
needed.
2.1.1 The Self-Attention mechanism
Theattention mechanism, originally proposed by Bahdanau, Cho and Bengio [4] to im-
prove machine traslation with RNNs, has assumed a central role when Vaswani et al.
[66] published a new Language Models architecture based on it.
From that moment on, this approach has been exploited massively, displacing RNNs
from the state-of-the-art techniques and becoming the new paradigm for text genera-
tion tasks.
The key point is to represent each token in the sequence with respect of the other
tokens present in it, i.e. produce a meaningfulcontextual representationof the words. By
comparing an item of interest to a collection of other items, we can reveal theirrelevance
in the context in which they are placed. The results of these comparisons are then used
4

to compute an output sequence for the current input sequence.
Diving into mathematical aspects, the comparison is performed by considering the
dot product between two word vector representations and produces a score represent-
ing the relevance of one with respect to the other:
score(x i, xj) =x i·xj
and this quantity takes values in theh
−|x i| · |x j|,|xi| · |x j|i
interval1, where the magni-
tude of the score is greater when the relevance of the tokens is greater.
Once these scores are computed, we are interested in translating them in relative
relevance scores, i.e. finding a function that mapsR→[0,1]. One of the most popular
choices in the machine learning field is the softmax(·)function:
αij=softmax(score(x i, xj)) =exp(score(x i, xj))P
jexp(score(x i, xj))
where the normalization is performed with respect to all the elementsx jin the
sequence. Thus each wordx iwill have an array of vector weightsαi= [α i1, ..., α iJ]
associated to it.
Now that we have the proportional relevance of each word in the sequencewith
respect tox i, we can compute a linear combination between each proportional relevance
αijand each wordx jin the sequence, resulting in the relevance of wordx iin the
context in which is immersed:
ai=X
jαij·xj
this is the so-calledattention weightsassociated to wordx i.
The process may appear non-trivial due to the shift of focus that happens in the
last step. Firstly we focus on one word, consider the relevance of the other words with
respect to it; then we use this information to compose a relevance score for the consid-
ered word.
The big implicit step in this procedure is the fact that ifx jis a sequence of characters
from a certain alphabet we need a further step that embeds each token into a numerical
vector, such that it is possible to perform the previous computations. We will not dig
deeper than this on the embedding details in this paragraph, but we will talk about
this in section (2.1.5).
However, word embedding is not the only transformation imposed to tokens. The
self-attention mechanismrepresents differently the same word with respect to therole
that it plays in the algorithm. The three roles that each word may have are:
• thequery: the current focus of attention, i.e. the word compared to the others;
1Ifxi,xj∈Rn(i.e. they are finite-dimensional), the bounds are finite. Otherwise, score(x i, xj)∈
[−∞,+∞].
5

• thekey: theotherword compared to the current focus of attention to produce the
proportional relevance weights;
• thevalue: theotherword used to compute eachattention weightfor the current
word.
To produce these representations of the same wordx j, it is necessary to apply three
different transformations that can be represented as matrices:
qi=x i·WQ;k j=x j·WK;v j=x j·WV
being the queryW Q∈Rd×dk, keyW K∈Rd×dkand valueW V∈Rd×dvmatrices.
We need to transformx iandx jusing the matricesWQ,WKandWVsince, if we
do not account for these projections, the model would be forced to use the same hid-
den vector for these three different tasks. But the query, the key and the value roles
require that the same token performs different tasks: consequently, the vectors have to
be transformed in order to let the matricesWQ,WKandWKabsorb this role in their
place.
With this in mind, we can rephrase the previous self-attention mechanism as:
score(x i, xj) =q i·kj;a i=X
jαij·vj
A final precaution has to be made: the dot product between two vectors can as-
sume arbitrarily large values, leading to numerical issues and hindering the effective
propagation of the gradients during the training. To avoid this scenario, a re-scaling of
the dot product is performed:
score(x i, xj) =qi·kj√dk
whered kis the dimensionality of the query and the key vectors.
Transformers are able to take in input, as preceding context, a maximum number of
tokens. This quantity is calledcontext lengthand currently takes values 4K or 8K, even
though many modern models are more capacious.
Previously, we considered one token in the sequence at a time; more precisely,x i
is the embedding of a word in the sequence, and it has dimension1×d, wheredis
the embedding dimensionality. We can now assume that the input of the self-attention
operation is the matrixX∈RN×d, containing thed-dimensionality embeddings of all
theNwords in the context. In the self-attention operation of the first model layer,
this input has to be multiplied by the key, query and value matrices to produce the
following matrices:
Q=XW QK=XW KV=XW V
whereQ∈RN×d k,K∈RN×d kandV∈RN×d v. The entire process described above can
be summarized as follows
X′=SelfAttention(Q, K, V) =softmax 
QKT
√dk!
V
6

whereX′is the contextualized representation for the sequence produced in the first
Self-Attention operation.X′will undergo further refinement and contextualization in
successive layers of the Transformer model.
The great advantage of the attention mechanism is each token’s the attention weight
can be computed independently of the others, given the weight matrices. This means
that the self-attention computation can be performed in parallel, reducing by a large
factor the training and the inference time with respect to the RNNs alternatives. In
practice, this means that the loss for a full sequence can be computed in a single for-
ward pass, rather than one pass per word in the sequence like in RNNs.
2.1.2 Multi-Head Self-Attention
It may be difficult for a single triplet of weight matrices(W Q, WK, WV)"to capture all
of the different kinds of parallel relations among its inputs"[33].
The solution proposed to this limitation is to create a set of triplets(Wh
QWh
K, Wh
V),
h∈ {1, ..., H}, each of them referred to asheadh.
The self-attention computation is performed independently in parallel for every at-
tentionhead(placed at the same depth in the model), and results are finally aggregated
in a unified output for further processing. The reason to have so many parameters is
that each head will project the input in a different representation highlighting some
characteristics allowing to perform a more varied analysis.
This choice implies that each head will output aN×d vmatrix2, thus we will have
Hmatrices of that shape. In order to obtain an output of the same dimension of the
input matrixX∈RN×d v, we have to project the outputs of all the heads into a single
RN×d vmatrix.
This is obtained by first concatenating the outputs of each head (in aRN×H·d vmatrix)
and then by projecting it using an output matrixW O∈RH·dv×d:
(head 1⊕head 2⊕...⊕head H)∈RN×H·d v; (head 1⊕head 2⊕...⊕head H)WO∈RN×d v
2This is due to the fact that, givenQh∈RN×d k,Kh∈RN×d kandV∈RN×d v, we perform the following
operations:
softmax 
QhKhT
√dk!
∈Rdk×dk;softmax 
QlKlT
√dk!
V∈Rdk×dk×RN×d v∈RN×d v
7

Figure 2.1: A schematic representation of Multi-Head Attention (MHA); eachheadis
provided with its own set of key, query and value matrices. The outputs of all the
heads are first concatenated and consequently projected toW O(taken from [33]).
In matrix terms, the Multi-Head Self-Attention mechanism can be expressed as:
Qh=XWh
QKh=XWh
KVh=XWh
V
headh=SelfAttention(Qh, Kh, Vh)
A=MultiHeadAttention(X) = (head 1⊕head 2⊕...⊕head H)WO
2.1.3 The Transformer block
In this section we are going to describe the structure of the Transformer unit, exploiting
the Multi-Head Self-Attention operation introduced in the previous paragraphs (2.1.2).
The Transformer block is schematically represented here:
Figure 2.2: taken from [33]
8

Breaking down the process with one equation for each computation component,
we obtain:
T1=MultiHeadAttention(X) ;T 2=X+T 1;T 3=LayerNorm(T 2)
T4=FFN(T 3) ;T 5=T 3+T 4;Y=LayerNorm(T 5)
Going in detail, the residual connections present before the Normalization layers
have some benefits. By allowing information from the activation to go forward and,
complementary, the gradient to go backwards skipping a layer, they improve learning
and give higher-level layers direct access to information residing in lower-level ones
[25].
The Normalization layer is useful to improve the training performances by keeping
the hidden layer values’ in a range that facilitates gradient-based training [33], [78].
The vector componentsxarerow-normalizedas:
ˆx=x−µ
σ
whereµandσare respectively:
µ=1
ddX
i=1xi;σ=vuut1
ddX
i=1(xi−µ)2
In addition to this, the standard implementation of layer normalization includes also
two learnable parameters,γ(the gain value) andβ(the offset value):
LayerNorm(x) =γ·ˆx+β=γ·x−µ
σ+β
Finally, the Feed-Forward layer containsNfully connected two-layer networks3
(i.e. composed pf one hidden layer and two weight matrices) that usually hasd= 512
input,d h= 2048hidden and againd= 512output neurons. The activation functions
applied between these two linear layers are both Rectified Linear Unit (ReLU) func-
tions, defined as ReLU(x) =max(0, x).
The operations within the FFN can be mathematically described by the following
equation:
FFN=max(0, max(0, x·W 1+b 1)·W 2+b 2)
whereW 1andW 2are the weight matrices andb 1andb 2are the biases for the first and
the second layer respectively.
Each of theNunits in FFN is also called position-wise network, since the weights
of the two matrices involved in the transformation are the same for all the processed
tokens, i.e. independent of the token position.
3We recall thatNis the context length; thus, the Feed-Forward layer processes in parallel each item
that comes out from the MultiHeadAttention and LayerNorm operations.
9

2.1.4 The Language Model Head
The Large Language Model is composed of a certain numberLof Transformer blocks
stacked one after the other. The output of the last Transformer block is aN×dma-
trix (one1×dstate per input token). For inference, only the last state (of dimension
1×d) is used to predict the next token, while the previous(N−1)×dones are ignored4.
In order to come out with a probability distribution over the vocabulary tokens, it
is necessary tounembedthe output of the with ad× |V|matrix. This linear layer can
be learned, but more commonly we tie this matrix to (the transpose of) the embedding
matrixE.
The output of this process is calledlogitorscore vectoru, since it has a score for
each of the|V|possible words in the vocabularyV. These scores are transformed into
probabilities through the usage of a softmax function. These two operations composed
the so-calledLanguage Model Headof the model:
u=hL
N·ETandy=softmax(u)
wherehL
N∈R1×d,ET∈Rd×|V|andu∈R1×|V|(consequently alsoy∈R1×|V|).
Figure 2.3: The Language Model Head: the circuit at the top of the last Transformer
block maps the output embedding for tokenNfrom the last layerhL
Nto a probability
distribution over words of the vocabularyV(taken from [33]).
The multi-layer architecture that we showed in the previous sections is referred
to asdecoder-onlyLLMs and it was first introduced byGPT-1[53] and subsequently
adopted in other model architectures.
BeforeGPT-1, the original Transformer architecture (as proposed by Vaswani et al.
[66]) was employed, consisting in both anencoderand adecoderpart.
2.1.5 Embedding words into numerical vectors
The first, fundamental step that preceds all the others is the transformation of the
words inside the sequence into numerical vectors on which we can perform the already
4On the opposite, for the training, each state isprojected(as described in this section) to the vocabulary
and a loss is computed in parallel as the average over the full predicted sequence against the gold
sequence.
10

explainedsequence of relevant mathematical operations, leading to information extraction
and allowing us to make inference.
Given a sequence ofNwords, we define theembeddingof a sequence as the matrix
X∈RN×d, wheredis the pre-defined dimension of each embedding vector.
We can imagine to define a vocabularyVof words present in the sequence of inter-
est. Given this dictionary, we can map each word into aone-hot encoded vector, in which
the only element which is not zeroed out is the one corresponding to the token.
Think for example of the sequenceMay the Force be with you, then the vocabulary
will be made by:V={be,Force,May,the,you,with}. We can represent the wordMay
as[1,0,0,0,0,0], the wordtheas[0,1,0,0,0,0]and so on. Note that all the vectors are
encoded into vectors having|V|elements, and that the entire sequence can be repre-
sented by a matrixW∈RN×|V|.
Tosmoothby a certain degree the word representation, we apply to the one-hot
encoded matrix a transformationE∈R|V|×dcalledembedding matrix:
Figure 2.4: Creating the word embedded matrixW∈RN×dfor the input sequence
of tokens by multiplying a one-hot matrix of dimensionsN× |V|by the embedding
matrixE∈R|V|×d(taken from [33]).
The initialN×|V|words representation is referred to assparse, in opposition to the
denseoneW∈RN×dof much lower dimensions (typically in the range ofd= 50to300
dimensions) produced by the application of an embedding matrixE.
The benefit ofdenseembeddings oversparseone is that, due to the training objec-
tive of the embedding matrixE, these kind of transformations push similar words to
be close in thedensevector space defined byW; this idea is inspired from thedistribu-
tional hypothesis-you shall know a word by the company it keeps[17]. Words embeddings
are used in allNeural-Networks basedLanguage Models, including in RNNs.
However, Transformer-based models for text generation, differently from RNNs,
do not have a sequential inductive bias by design.
In this kind of models, word embeddings are position-independent. In order to
keep track of this information, we have to add to the "standard"word embeddingw i∈
R1×d(wiis thei-th row ofwords embeddedmatrixW∈RN×d) its correspondingposition
embedding, defined as a function mapping each positioniinto an embeddingP[i]∈
R1×d.
In this setting, the final representation of the input, the matrixX∈RN×dis such that
each rowiis the representation of thei-th token in the input, computed by adding the
11

embeddingw iof the token that occurred at positioni, toP[i], the positional embedding
of positioni.
Figure 2.5: A simple way to model position: add an embedding of the absolute posi-
tion to the token embedding to produce a new embedding of the same dimensionality
(taken from [33]).
The positional embedding method proposed by Vaswani et al. [66] aims at ensuring
that each sentence positioni∈1, ..., Ihas an unique representation. Their idea is to
exploitsinandcosfunction to achieve this goal without the need to train a separate
positional embedding matrixP. do is apply to each positionito a set ofdfunctions in
order to obtain a1×dvector for each sequence position.
Figure 2.6: Toy example of Positional Encoding (withd= 4), inspired from [40]. Note
that what matters is not the term (e.g.fast) the positioni(e.g.i= 3) that is appropri-
ately transformed throughsinandcosfunctions.
2.1.6 Causal self-attention indecoder-onlyTransformers
Theself-attention mechanismpresented in section 2.1.1 implies that theattention weight
associated with a token will not only depend on the past ones (i.e. the preceding con-
text), but also on the upcoming ones. The issue that thisbidirectional attentioncarries
is that, in some sense, itincorporatesthe future information in the representation of the
current word.
12

This approach is unsuited for tasks like text generation, since we expect the model
to correctly predict thenextword, and this task is trivial if the model already knows the
following ones. Practically speaking, by usingbidirectional self-attentionTransformer in
a "guess which is the next word" problem we are allowing the model to cheat.
Due to theinductive biasof auto-regressive language modeling, acausal maskis
used to hide future tokens from the context mixing operation of the attention block. On
the contrary, masked language models used forclassification taskslike BERT commonly
donotemployattention masks, leveraging bidirectional context.
The process involves theattention weightsof the following words and results in a
maskedself-attention matrix in which the elements in the upper-triangular portion of
the matrix are zeroed out (set to−∞), thus eliminating any knowledge of words that
follow in the sequence.
Figure 2.7: Representation ofQ·KT∈RN×Nmatrix when a causal mask is applied. The
upper-triangle portion of it is set to−∞, which the softmax will turn to zero (taken
from [33]).
Mathematically speaking, it is a restriction of what we have already described in
(2.1.1). The difference stands in the range of allowed indexes fori:
αij=(
softmax(score(x i, xj)),∀j≤i
−∞,otherwisewhere score(x i, xj) =qi·kj√dk
The model resulting from this approach is purelyautoregressive: the model will look
at the past and infer the following token from it; then the predicted token will be added
to the context, producing a new one to predict the next following word, and so on.
An important remark is that the concept ofcontextcan be used in two ways in self-
attention. In causal self-attention, the context is any of the prior words. Inbidirectional
self-attention, the context can include future words.
13

2.2 Improvements in modern Large Language Models
In the following sections we are going to provide a detailed analysis on the architecture
of these models and some pre-training and post-training insights. Before diving into
details, it is necessary to introduce some modern modifications of the original Trans-
former architecture.
2.2.1 The RoPE embedding
In section (2.1.5) we introduced the Positional Encoding method, allowing Large Lan-
guage Models to take into account both the semantic meaning of a token and its po-
sition in a certain sentence. The issue with these kinds of approaches, calledabsolute
positional embeddings, is that each positional embedding is independent of others.
This means that, in the model’s view, the difference between positions1and2is the
same as between positions2and500. But intuitively, positions1and2should be more
closely related than position500, which is significantly farther away. This lack of rela-
tive positioning can hinder the model’s ability to understand the nuances of language
structure.
On the opposite, rather than focusing on a token’s absolute position in a sentence,
relative positional embeddings[56] concentrate on the distances between pairs of tokens.
This family of methods does not add a position vector to the word vector directly. In-
stead, italters the attention mechanismto incorporate relative positional information.
For example, abiasmight represent the relative distance between any two tokensthat
are one position apart, regardless of their absolute positions in the sentence. The
matrix composed by all relative position biases is added to the product of the query
and key matrices in the self-attention layer, so that it isensured that all tokens at the
same relative distance are always represented by the same bias, regardless of their
position in the sequence.
Although this method scales to long text sequences, this causes a slowdown in the
computational time due to the addition of some operations in the self-attention layer.
Summing up,absolute positional embeddingsassign a unique vector to each position,
which though straightforward, doesn’t scale well and fails to capture relative positions
effectively.Relative embeddings, on the other hand, focus on the distance between to-
kens, enhancing the model’s understanding of token relationships but complicating
the model architecture.
A third solution is given by theRotary Positional Embeddings(RoPE) [61], that inge-
niously combines the strengths of both. It encodes positional information in a way that
allows the model to understand both the absolute position of tokens and their relative
distances. This is achieved through a rotational mechanism, whereeach position in the
sequence is represented by a rotation in the embedding space.
RoPE introduces a novel concept. Instead of adding a positional vector, it applies
a rotation to the word vector. Imagine a two-dimensional word vector for “dog.” To
encode its position in a sentence, RoPE rotates this vector. The angle of rotation (θ) is
proportional to the word’s position in the sentence. For instance, the vector is rotated
14

byθfor the first position,2θfor the second, and so on.
The technical implementation of RoPE involves rotation matrices:
fQ(X, m) =cos(mθ)−sin(mθ)
sin(mθ)cos(mθ)
·X·W Q=cos(mθ)−sin(mθ)
sin(mθ)cos(mθ)
·Q
fK(X, m) =cos(mθ)−sin(mθ)
sin(mθ)cos(mθ)
·X·W K=cos(mθ)−sin(mθ)
sin(mθ)cos(mθ)
·K
In a 2D case, the equation from the paper incorporates a rotation matrix that rotates
a vector by an angle ofmθ, wheremis the absolute position in the sentence. This
rotation is applied to the query and key vectors in the self-attention mechanism of the
Transformer5.
The advantage of this approach is that if two words maintain the same relative
positions in two different contexts, their embeddings form the same angle, thus the
same dot product.
Figure 2.8: An example of RoPE in a trivial 2D case; the original embedded vector is
rotated with angleθwith respect to its positionmin the sentence.
However, RoPE has limitations when applied to very long sequences as it assumes
afixed positional rangedefined by the base valueθ. Extending RoPE to longer se-
quences could, in theory, involve increasingθor training a new model from scratch
with a larger positional range, but both approaches are impractical due to high com-
putational costs and data scarcity.
LongRoPE [14] addresses these challenges by dynamically optimizing the posi-
tional encoding using two key innovations. First, it introduces a loss function that
searches for the optimal scaling factor,λ, instead of assuming a fixed one (i.e.θ). This
adaptive approach allows the model to find the most suitable scaling for different po-
sitions within a sequence, preserving the positional relationships even as the context
window expands. Second, LongRoPE identifies subsets of tokens that should remain
unchanged, preventing the loss of critical positional information that could degrade
model performance.
5For higher dimensions, the vector is split into 2D chunks, and each pair is rotated independently
15

2.2.2 Pre-normalization and post-normalization architecture
Sometimes the original Transformer model is slightly modified, and this is the case of
apre-normalization architecture. By putting LayerNorm before the MultiHeadAttention
and FNN layers, we are normalizing the values before entering the crucial computa-
tional layers. This detail can cause an improvement in performances, due to the same
reasons given in section (2.1.3).
Figure 2.9: On the left, the original Transformer block architecture; on the right, the
pre-normalization architecture.
The first idea of a pre-normalization architecture can be found inGPT-3[8], and later
used in open-source models such asLlaMa[64].
16

2.2.3 Sliding Window Attention (SWA)
Classical Self-Attention mechanism (2.1.1) works by comparing a focus of attention
with the other elements of the sequence. In order to allow eachq ito attend all thek j,
we requirequadratic memory for each attention layer.
In fact, the Self-Attention output is a quadratic matrix of the same shape of the
input sequence (aN×Nmatrix) in which each element is evaluated in terms of its
relevance with respect to the other elements in the sequence.
In decoder-only Transformers (2.1.6), acausal maskis applied in order to mask future
tokens. This helps in the text generation task, though it is not sufficient to optimize this
operation.
BERT-based models fix a maximum sequence length and split the document into
multiple overlapping segments having at maximum that length. They process inde-
pendently each segment and their representations are combined. The big issue with
this approach is that the attention informationacross segmentsis lost.
LongFormer [7] proposes fixing a ’window of interest’ of lengthN′, such that each
token is only allowed to attend to its peers, i.e., tokens no farther thanN′items away.
The paper suggests to look at1
2N′tokens at each side, making the computational com-
plexity to beO(N×N′), but it is not the only suggestedattention pattern:
Figure 2.10: Different attention patterns presented in [7]
This idea reduces by a large margin the memory requirements and consequently
the training and inference times, and still maintains good levels of performance due
to a brilliant subtlety. In a model made ofLTransformer blocks, the layers attend
information by looking at the output of the preceding layerand at its peers. This
means that in the last layers we will obtain aconical structure of hidden relationships,
gaining information from tokens far away from it.
17

Figure 2.11: An intuitive explanation of hidden relationships in Sliding Window Atten-
tion. Each token is represented as a different color, in order to show how information
about it propagates in the layers. In this toy example,N′is 3 andLis 3.
The optimized model will be faster and use less memory than the original model,
but it may also be less accurate. The challenge of SWA is to reduce the computational
complexity as much as possible without significantly reducing the model’s accuracy.
This method is currently used by theMistral’s family of models [32].
2.2.4 Multi-Head Attention, Multi-Query Attention and Grouped Query
Attention
In section 2.1.2 we introduced theMulti-Head Attention(MHA) mechanism proposed
by [66]. Eachheadhis associated with a(Qh, Kh, Vh)triplet, and the number of heads
Hpresent in each Transformer block determinesthe number of possible "points of view"
used to analyze the input sequence. But in this setting the number of parameters to
train and to use to make inference is really big. This provides high quality at the cost
of consuming higher memory bandwith.
TheMulti-Query Attention(MQA) is a technique to accelerate the inference pro-
cess, reducing drastically the number of involved matrices. It uses a single key matrix
Kand a single value matrixVfor all theheads, while still involvingHquery matrices
Qh, h∈1, ..., H. This reduces by a large margin the training time and the memory con-
sumption, leading to worse performances due to training instability.
A compromise is given by theGrouped Query Attention(GQA), which seeks to
strike a balance between MHA and MQA. GQA partitions query heads intoGgroups,
with each group sharing a single key and value matrices. This provides a trade-off
between the speed of MQA and the quality of MHA.
18

Figure 2.12: MHA, GQA and MQA (taken from [21])
Most modern LLMs utilize GQA, as it achieves performance comparable to MHA
while significantly reducing computational time.
2.2.5 Decoding strategies
The core of the generation process for large language models is the task of choosing the
single word to generate next based on the context and based on the probabilities that
the model assigns to possible words. This task of choosing a word to generate based
on the model’s probabilities is calleddecoding.
Repeatedly choosing the next word conditioned on our previous choices is called
autoregressive generationorcausal decoding.
There are two main decoding strategies:greedy decodingandsampling.
Greedy decodingconsists in selecting the token with the highest probability at each
decoding step; this produces outputs that most closely match thecommonoption (i.e.
in model’s prompt or pre-training data). This decoding strategy is used for fact-based
use cases and tends to produce lesscreativeoutputs.
Samplinginstead chooses tokens according to their probability assigned by the model.
Thus it ismore likelyto generate words that have a high probability in the context and
viceversa. This process continues until a pre-determined length is reached or when the
end-of-sentence token is generated.
We can formalize this algorithm for generating a sequence of wordsS=s 1, s2, ..., s N
until we hit the end-of-sequence tokenEOS. We usex∼p(x)to refer to the process of
choosingxby sampling from the distributionp(x):
1. initializei= 1;
2. samplew i∼p(x);
3. whilew i̸=EOS,i=i+ 1andw i∼p(w i|w<i).
Samplingadds variability and randomness to the decoding process, which can be
desirable in creative use cases. However, with greater variability comes a greater risk
of incorrect or nonsensical output (a problem which is not present ingreedy decoding).
This strategy uses three different parameters to adjust how the model chooses to-
kens to sample:
19

•temperatureflattens (when set to values near 0.0) or sharpens (when near 2.0)
the probability distribution over the tokens to be sampled, by default is set to 0.7;
•top-ksamples tokens with the highest probabilities until the specified number of
tokens is reached (can be set up to 100), by default is set to 50;
•top-psamples tokens with the highest probability scores until the sum of the
scores reaches the specified threshold value (a floating point value between 0.0
and 1.0), by default it is not used.
Greater temperature values and greater values of k lead to increased variability and
creativity in model’s answers. Even though this approach is mostly going to generate
sensible, high-probable words, there are many odd, low-probability words in the tail
of the distribution that get chosen often enough to result in generating weird sentences
since they constitute a large enough portion of the distribution [33].
Different sampling strategies have been proposed in order to ensure bothquality
anddiversity, typically found in a trade-off: methods that give a bit more weight to the
middle-probability words tend to be more creative and more diverse, but less factual
and more likely to be incoherent or otherwise low-quality. In our analyses, we used
onlygreedy decodingto ensure a more reliable choice of tokens when generating the
outputs.
2.3 Large Language Models employed
First of all, our focus lied on the choice of the Large Language Models involved in this
analysis. We made the choice of considering only open-source models, freely accessi-
ble from the HuggingFace collaboration platform6.
The choice fell back on three distinct families of models: themeta-llama/LlaMa-3.1
[2], themicrosoft/Phi-3[1] and thegoogle/Gemma-2[62] families of models. In all three
aredecoder-only models(2.1.6).
All these three models incur in some forms of pre- and post-training. While the
pre-training stage consists in feeding the model a large quantity of data in order to
teach the model how to produce sentences that make some sense, the post-processing
phases tend to align the model to some preferences.
For example, their goal could be to teach the model to follow instructions given in a
certain format or for example to properly use a set of terms linked to a certain semantic
field. These two tasks are often performed via Supervised Fine-Tuning (SFT) [69], [28],
i.e. the process of modifying model’s weights (all or only a part of them) to change the
downstream behaviour, obtaining the desired specific output.
6Hugging Face, Inc. is a French-American company that develops computation tools for building
applications using machine learning. It is most notable for itstransformers[77] anddatasets[41]
libraries and for its platform that allows users to share machine learning models and datasets and show-
case their work.
20

Another approach to align the model’s behaviour towards a desired one is given
by Reinforcement Learning with Human Feedback (RLHF) [51] that refines the model
by incorporating human evaluations into the training process. This method involves
training the model to maximize a reward function that reflects human preferences or
values, often by using comparisons between model outputs ranked by human anno-
tators. The RLHF approach has been particularly effective in adjusting a model’s re-
sponses to be more aligned with human expectations in terms of tone, content rele-
vance, safety, factual accuracy and other desired properties.
In addition to SFT and RLHF, Direct Preference Optimization (DPO) [55] has emerged
as another effective technique for fine-tuning language models. Unlike RLHF, which
relies on constructing a reward model to optimize the output, DPO directly lever-
ages human preferences by optimizing the model based on explicit comparisons of
preferred outputs. In this method, human annotators are asked to compare multiple
model outputs and rank them according to their quality or alignment with specific
guidelines. The model is then fine-tuned to favor the outputs that are ranked higher,
effectively guiding its behavior towards generating more desirable responses without
needing to construct a complex reward model.
DPO simplifies the optimization process by directly minimizing the difference be-
tween the model’s output distribution and the desired distribution indicated by hu-
man preferences. This approach can result in faster convergence and reduced com-
putational overhead, making it a promising alternative to RLHF for situations where
direct comparisons between outputs are sufficient for alignment purposes.
By combining SFT, RLHF, and DPO, it is possible to effectively adjust language
models to achieve the desired behavior, ensuring that outputs adhere to expectations.
2.3.1Gemma-2family
The last release of theGemmafamily of models is dated July 2024, withGemma-2[19]. The
notation that will be used to flag different components in the models will reflect the
one present in the Transformer section (2.1).
The pre-training data primarily consist of English-language web documents, code,
and scientific articles. The size of the pre-training corpus varies with respect to the
model scale:Gemma-2B7required 2T tokens,Gemma-9B8T,Gemma-27B(not considered in our
analysis) 13T. The data is carefully filtered to reduce the risk of unwanted and unsafe
utterances, discarding personal information and unsafe data.
Data is tokenized using Google’sSentencePiecetokenizer [20], implemented inC++;
it splits digits, does not remove whitespaces and relies onByte-Pair Encoding(BPE) to
deal with unknown tokens. Its vocabulary is made by256.128different tokens.
Gemmamodels use RoPE embeddings (2.2.1) in place of some absolute Positional Em-
bedding method, and share this embedding matrix with the output layer.
The Transformer block uses the Grouped Query Attention (2.2.4) to achieve similar
downstream performances but reducing the inference time. InGemma-2B,H= 8while
7From now on, we will refer toGemma-2- *BasGemma- *Bfor brevity.
21

G= 4; inGemma-9B,H= 16whileG= 8. The first model hasL= 26layers for each
Transformer block, the second oneL= 42. In both cases, the head size isd v= 256,
while the model sizedis different:d= 2304forGemma-2B,d= 3584forGemma-9B.
To speed up computations without harming the performances, they alternate lay-
ers employing a local sliding window attention of sizeN′= 4096(2.2.3) and a global
attention of sizeN= 8192tokens.
Effort is made to facilitate the gradient propagation by doing mainly two changes.
The first one is to performbotha pre-normalization and a post-normalization (2.2.2),
i.e. the number of LayerNorm is duplicated.
The second one is to perform alogit soft-cappingin each attention layer and in the
final layer, in order to keep the values of the logits in the range[−softcap; +softcap]:
logits=softcap·tanhlogits
softcap
In the original paper,softcap= 30for the final layer andsoftcap= 50for the attention
layers.
The ReLU activations in the FFN layer are replaced by GELU [26] ones. GELU
stands for Gaussian Error Linear Unit and corresponds to:
GELU(x) =x·P(X≤x) =x·Φ(x), X∼N(0,1)
The model is both pre-trained and post-trained.
The post-training procedure involves the "standard" Supervised Fine-Tuning (SFT),
the Reinforcement Learning with Human Feedback (RLHF) and model merging, i.e.
averaging different models obtained by running the training with different hyper-
paramers [57].
In detail, SFT works as follows (taken from [62]):given a set of held-out prompts, we
generate responses from a test model, generate responses on the same prompts from a baseline
model, shuffle these randomly, and ask a larger, high capability model to express a preference
between two responses. Different prompt sets are constructed to highlight specific capabilities,
such as instruction following, factuality, creativity, and safety.
In our experiments we will use the instruction-tunedGemma8, calledGemma-it, whose
prompt format is:
<start _of_turn>user
user _message
<end _of_turn>
<start _of_turn>model
model _message
<end _of_turn>
...
<end _of_turn><eos>
8ByGemma, we refer to the pre-trainedGemma-2model. ByGemma-itwe refer instead to the output of
a SFT process performed onGemma-2by the Gemma research team, available on HuggingFace [19] and
tuned for instruction prompts.
22

Both the9Band2Bmodels are distilled [27] from the27Bmodel by minimizing the
negative log-likelihood between the probabilities of the student (i.e.p s(x|x c)) and the
teacher model (i.e.p t(x|x c)):
min
psX
x−pt(x|x c)·log p s(x|x c)
2.3.2Phi-3family
Microsoft has developed thePhimodels stating that "Textbooks are all you need" [22].
They releasedPhi-1, a 1.3B parameters model intended for coding purposes, trained
using a selection of "textbook quality" data from the web (6B tokens) and synthetically
generated textbooks and exercises with GPT-3.5 (1B tokens) and they obtainedphi-1-
base. They further fine-tuned it on code exercises, producingphi-1.
Research on this topic has developed since then, with the release of thePhi-3[48]
family of models, being trained on heavily LLM-filtered, publicly available, web data
and on LLM-generated synthetic data. The fundamental idea is to improve the perfor-
mance trends that were previously shown to be predictable, once the model size, the
data size and the computational budget are given [37]. ThePhiteam works on allowing
the model to interact the data in novel ways, instead of keeping the data source fixed
as proposed by Kaplan et al. [37].
The pre-training takes place in two stages: a first phase, aimed at teaching the
model general knowledge (and consequently trained mostly on web sources) and a
second phase, aimed at teaching the model how to logically reason and to attain spe-
cific skills (for this, a subset of the data of the previous phase is taken and used to
generate new data for those goals).
Different models were released:
•Phi-3-mini: a 3.8B model trained on 3.3T tokens usingbfloat16, with aLlaMa-2-like
architecture [63];Phi-3-small: a 7B model with different architecture thanPhi-mini
andPhi-mediumones9.
•Phi-3-medium: a 14B model trained on 4.8T tokens usingbfloat16, with aLlaMa-2-
like architecture [63].
For our analysis, we will consider onlyPhi-3-miniandPhi-3-mediumdue to their sim-
ilar architecture. They are both built onLlaMa-2architecture (apre-normalizationarchi-
tecture) and they use the same tokenizer, with vocabulary size of32.064.Phi-mini10and
Phi-mediumare both released with context length equal to 4K and 128K. For the first one,
RoPE embedding is used, while LongRoPE needs to be used in the other case (2.2.1).
9It follows the standard 7B model architecture, uses thetiktokentokenizer (thus has a different
vocabulary that the other models) and it alternates layers of global self-attention and a particular kind
ofblocksparseattention [phi3].
10As forGemma-2models, from now on we will omit the release number;Phi-3- *will be simply
referred to asPhi- *.
23

The Transformer block uses Multi-Head Attention inPhi-miniwithH= 32heads
and Grouped-Query Attention (2.2.4) inPhi-mediumwithH= 32query heads (each of
dimensiond v= 128),G= 8key and value heads (i.e. 4 queries share 1 key). Each
Transformer block containsL= 32layers inPhi-mini,L= 40inPhi-medium. The model
size isd= 3072forPhi-miniandd= 5120forPhi-medium.
The FFN hidden dimension is equal to13.824. Thus each classification head takes
as input a feature size ofdand outputs a32.000-dimensional vector. Differently from
Gemma-2models,Phi-3ones use SiLU (Sigmoid Linear Unit) activation [16]:
SiLU(x) =x·sigmoid(x)
wheresigmoidis the logistic sigmoid function. The SiLU function is also referred to as
the Swish function.
The post-training procedure consists of Supervised Fine-Tuning (SFT) followed by
Direct Preference Optimization (DPO) [55]. SFT leverages highly curated high-quality
data across diverse domains, e.g., math, coding, reasoning, conversation, model iden-
tity, and safety. The SFT data mix starts with using English-only examples. DPO is
used to steer the model away from unwanted behavior in order to ensure improve-
ment in math, coding, reasoning, robustness, and general safety.
The models are also instruction-tuned and are released only in this version [48].
They can be used with the following prompt template:
<|system|>
system _message <|end|>
<|user|>
user _question <|end|>
<|assistant|>
A relevant observation made in the original paper is the following:some benchmarks
improve much less from 7B to 14B than they do from 3.8B to 7B, perhaps indicating that our
data mixture needs further work to be in the “data optimal regime” for 14B parameters model
(i.e. forPhi-medium). This topic will be relevant in our analysis.
2.3.3LlaMa-3.1family
LlaMa-3.1[47] takes heavy inspiration from the previousLlaMareleases, and its perfor-
mance gains are primarily attributable to the data mixture used in the pre-training
phase.
It expands thetiktoken[50] tokenizer with extra 28K tokens to better support non-
English languages. Compared to theLlama-2tokenizer, this new tokenizer improves
compression rates on a sample of English data from 3.17 to 3.94 characters per token.
This enables the model to “read” more text for the same amount of training compute.
They still use RoPE embeddings (2.2.1), but they increase the value ofθto 500.000,
in order to deal effectively with context lengths up to∼30K.
24

The pre-training ofLlaMa-3.1is performed on a corpus of about 15T multilingual
tokens, compared to 1.8T tokens forLlaMa-2. The data comes both from a variety of data
sources containing knowledge until the end of 2023, but the main part of it was made
by web data. Big efforts were made in their selection and curation. They implemented
filters designed to remove data from websites is likely to contain unsafe content or
high volumes of personal information. They also built a custom parser to extract text
from the HTML page, handling carefully web sources containing mathematics and
code sections.
They also applied several rounds of aggressive de-duplication at the URL, docu-
ment, and line level, paired with heuristics to remove additional low-quality docu-
ments, outliers, and documents with excessive repetitions.
After these stages, they labeled the scraped web sources in order to obtain a het-
erogeneous and balanced data mix on which the models are pre-trained. It contains
roughly 50% of tokens corresponding to general knowledge, 25% of mathematical and
reasoning tokens, 17% code tokens, and 8% multilingual tokens.
LlaMa-8B11uses the Grouped Query Attention (2.2.4) withG= 8key-value heads
to improve inferencehead speed and to reduce the size of key-value caches during de-
coding. The query heads areH= 32instead, with alsoL= 32layers inside each
Transformer block; the model size isd= 4096.
Such asPhi-3, it has a pre-normalization architecture with a SiLU activation func-
tion in the FFN layer.
Several rounds of post-training are also applied, and they involve Supervised Fine-
Tuning (SFT), Direct Preference Optimization (DPO) [55] and Reinforcement Learning
with Human Feedback (RLHF) [51]. Finally, they averaged models obtained from ex-
periments using various versions of data or hyperparameters at each RM, SFT, or DPO
stage [2].
2.3.4 Models summary
Here we provide a tabular summary of the models’ hyperparameters and details:
11As forGemma-2and forPhi-3, we will omit the release number from now on.
25

hyperparameters Gemma-2B Gemma-9B Phi-mini Phi-medium LlaMa-8B
vocabulary size 256,128 256,128 32,064 32,064 128,000
tokenizer SentencePiece SentencePiece LlaMa-2 LlaMa-2 tiktoken + 28K
PE method RoPE RoPE RoPE/LongRoPE RoPE/LongRoPE RoPE
PEθ 10,000 10,000 10,000 10,000 500,000
pre-train data 2T 8T 3.3T 4.8T 15T
N 8K 8K 4K/128K 4K/128K 128K
N′4K 4K × × ×
d 2304 3584 3072 5120 4096
L 26 42 32 40 32
H 8 16 32 40 32
G 4 8 32 10 8
MHA/GQA GQA GQA MHA GQA GQA
pre-norm ✓ ✓ ✓ ✓ ✓
post-norm ✓ ✓ × × ×
dh 9216 14,336 8192 8192 14,336
activation GELU GELU SiLU SiLU SiLU
Table 2.1: Hyperparameters comparison of the analyzed models.
2.4 Supervised Fine-T uning and RAG strategies
Apart from the broad and general pre-training phase, several techniques were used to
refine the model’s abilities in producing high-quality and correct outputs.
Scaling model size turned not to be the optimal approach when facing multi-steps
problems, as highlighted by Google when trainingGopher[12]. From the paper’s con-
clusions:
However, the benefits of scale are nonuniform: some tasks which require more complex
mathematical or logical reasoning observe little benefit up to the scale ofGopher. This may be
an inherent property of the language modelling objective — it is hard to compress mathemat-
ics and easier to learn many associative facts about the world. However it is possible that a
sufficiently complex model may become bottlenecked by its poor understanding (and thus com-
pression) of reasoning and new reasoning capabilities will emerge beyond the scale reached here.
Supervised Fine-Tuning(SFT) [28] is a transfer learning approach in which the pa-
rameters of a pre-trained model are further trained on new data. The key idea is that,
while the most fundamental tasks have been covered by the pre-training phase, the
fine-tuningsharpens and specializesthe LLMs to produce accurate outputs with respect
to the given data. The big issue with this approach is that the model becomes surely
highly capable on the tasks on which it is fine-tuned, but could behave unexpectedly
in response to prompts in which it worked perfectly before [74].
It can be performed on the entire neural network set of weights or on only a subset
of its layers while the others are kept "frozen", i.e. not changed during the backpropa-
gation step.
A model may also be augmented with "adapters", consisting of far fewer parame-
ters than the original model, and fine-tuned in a parameter–efficient way by tuning the
26

weights of the adapters and leaving the rest of the model’s weights frozen.
This family of approaches is often referred to asParameter-Efficient Fine-Tuning(PEFT)
[45], and one of the most famous and used techniques isLow-Rank Adaptation(LoRA)
[29].
A positive consequence of PEFT methods is that, by reducing by a large margin
the computational and storage requirements, it also decreases the impact of the catas-
trophic forgetting [74].
On the opposite,Retrieval Augmented Generation(RAG) strategies try to switch the
source of knowledge from aparametricone to anon-parametricone. In practice, instead
of modifying the model’s inner weights, RAG dynamically retrieves relevant informa-
tion from a knowledge base and uses this to ground LLM predictions. Most commonly,
it infills relevant passages in the model prompt. Inembedding-basedretrieval, knowl-
edge is accessed via a dense vector index of sources12.
RAG allows the model to receive anenriched promptthat should help the model
to access relevant knowledge and consequently to give a more precise answer to the
original prompt. Obviously, this involves an automatic retrieval that should match the
user’s request with the knowledge sources and append the most similar passages to
the prompt. This process could not be exact and often can introduce some noise to the
inference stage.
Figure 2.13: RAG workflow and its three components: Retrieve, Augment, Generate.
RAG and SFT are often compared as mutually exclusive alternatives, even though
they exploit different model components. They act in different ways:
12From now on, we will implicitly assume that this is the case.
27

Model RAG SFT
Cost – input token size Increased prompt size Minimal
Cost – output token size More verbose, harder to steer Precise, tuned for brevity
Initial cost Low (creating embeddings) High (fine-tuning)
Accuracy Effective Effective
New Knowledge If data is in context New skill in domain
Table 2.2: Insights on RAG vs Fine-tuning, Table 23 on [6]
The most intuitive way to understand their difference is to think of possible ways
in which students can perform well on a school test. The "SFT-student" will prepare for
a test by merging the previously acquired information with the new one, by internal-
izing the material on which he is tested on. This means that on the day of the test, the
student will have to rely only on his memory and on his internalized understanding
of the subject to answer to the teacher. The consequence of this is that he could forget
previously acquired notions and fail on more generic/old tasks, while performing ex-
actly on the ones of the test. The "RAG-student" will have at disposal the entire book
from which the test notions are taken. This means that he will not have to memorize
anything, he will just have to search for a similar question or paragraph in the book (or
more than one) and summarize or formulate a proper answer to the given question.
The memory component in this scenario is external.
Despite the simplistic metaphor presented above, the combination of SFT and RAG
offers intriguing possibilities by both leveraging external sources of knowledge and
facilitating concept internalization.
2.5 In-context learning or "few-shot" prompting
The next question is that if it is really impactful to fine-tune a model for each down-
stream task. Even Cobbe et al. [12] required the usage of fine-tuned models and of one
ad hoctrained verifier.
Are fine-tuning and training in general the solutions to achieve really results?
The pre-training and the fine-tuning stages present in each modern LLM make
these models capable of performing some tasks in a "zero-shot" manner. By "zero-
shot" we mean simply that the prompt used to interact with the model won’t contain
examples or demonstrations, but just instructions on what should be the expected be-
haviour. The zero-shot prompt directly instructs the model to perform a task without
any additional examples to steer it.
This setting is the standard one, meaning that usually we could imagine to use the
model by assuming that it correctly understands what we are asking.
In scenarios in which we aim to make the model behave in a precise way, we could
perform fine-tuning in order to skew the model output in a desired way. This process
has been shown to improve zero-shot learning [70] but is also expensive in terms of
number of parameters to fine-tune or datasets dimensions [12].
28

When zero-shot doesn’t work, we simply can provide demonstrations or examples
in the prompt which leads to "few-shot" prompting.
For example, while the following example (taken from [9]) is "zero-shot":
Translate from English to French:
cheese => _______
The corresponding "few-shot" prompt would be:
Translate from English to French:
sea otter => loutre de mer
peppermint => menthe poivree
plush girafe => girafe peluche
cheese => _______
"Few-shot" prompting, first introduced in [9], can be used as a technique to enable
in-context learningwhere we provide demonstrations in the prompt to steer the model
to better performance. The demonstrations serve as conditioning for subsequent exam-
ples where we would like the model to generate a response. Brown et al. [9] support
the idea that the effectiveness of fine-tuning a model on a precise task and of prompt-
ing the model with few demonstrations could be comparable:
Recent work has demonstrated substantial gains on many NLP tasks and benchmarks by
pre-training on a large corpus of text followed by fine-tuning on a specific task. [...] this method
still requires task-specific fine-tuning datasets of thousands or tens of thousands of examples.
By contrast, humans can generally perform a new language task from only a few examples or
from simple instructions [...]. Here we show that scaling up language models greatly improves
task-agnostic, few-shot performance,sometimes even reaching competitivenesswith prior
state-of-the-art fine-tuning approaches.
The authors also state that few shot properties first appeared when models were
scaled to a sufficient size and that larger models are more proficient atin-context learn-
ing.
29

Figure 2.14: Aggregate performance of GPT-3 for all 42 accuracy-denominated bench-
marks.While zero-shot performance improves steadily with model size, few-shot performance
increases more rapidly, demonstrating that larger models are more proficient at in-context learn-
ing(taken from [9], figure 1.3).
2.6 Chain-of-Thought (CoT) prompting
The idea of Wei et al. [68] was to introduce exploit the "few-shot" prompting strategy
to show the model how toreasonabout a given task. Their experiments passed to the
model examples with triplets of(input, rationale, output)instead of directly showing
the model whichoutputcorresponds to a certaininput.
Note that Chain-of-Thought approach operates in the opposite direction of Re-
trieval Augmented Generation. RAG tries to exploit external, non-parametric knowl-
edge, appending it to the prompt to facilitate the access to it.
CoT relies instead on the parametric knowledge acquired to the model during the
pre-training phases, often not accessible due to the task complexity. The authors were
inspired by how humans face non-trivial questions:
Consider one’s own thought process when solving a complicated reasoning task such as a
multi-step math word problem. It is typical to decompose the problem into intermediate steps
and solve each before giving the final answer: “After Jane gives 2 flowers to her mom she has
10... then after she gives 3 to her dad she will have 7... so the answer is 7.” The goal of this
paper is to endow language models with the ability to generate a similar chain of thought—a
coherent series of intermediate reasoning steps that lead to the final answer for a problem.[68]
As [9], they clearly identify "few-shot" abilities as an emergent property that only
sufficiently large, i.e. with more than∼100B parameters, models possessed13.
13From [68]:chain-of-thought prompting does not positively impact performance for small models, and only
yields performance gains when used with models of∼100B parameters. We qualitatively found that models of
smaller scale produced fluent but illogical chains of thought, leading to lower performance than standard prompt-
ing.
30

Figure 2.15: Some examples of(input, rationale, output)triplets for arithmetic, com-
monsense, and symbolic reasoning benchmarks in [68].
Many experiments were made to assess why LLMs benefit from thisadditional rea-
soning chance. What they discovered is that the most significant improvements are
obtained when the rationales are expressed in natural language.
In fact, they tested:
• Rationales expressed arithmetically (for math reasoning tasks) or symbolically:
this test was conducted to examine the impact of using "thought" instead of a
"schematic/analytical development" of the problem. What they found is that
opting for a simple, rationale-based approach is crucial in helping the model pro-
duce an answer.
• Outputting a sequence of dots (".") in place of each rationale’s token: this isolates
the effect of spending more time on computation. They showed that additional
computation alone does not lead to big performance improvements.
• Making the model answer in the first place, then reasoning about the question:
this should activate hidden knowledge acquired during training. The correct
order of the tokens generated in the output has an impact on the performances
of the model.
All the tests failed with respect to the performances offered by the natural language
rationale:
Figure 2.16: Impact of different prompting strategies as Chain-of-Thought, taken from
[68]
The advantages offered by the Chain-of-Thought approach are various. It avoids
the need of large pre-training/fine-tuning datasets, since a single, generic model can
now be used to perform a variety of tasks just by being prompted differently. In ad-
dition to this, by asking the model to reason about an input, we allow the model to
decompose the problem in multiple steps, eventually allocating additional computa-
tion to problems that were originally solved "in one bite".
31

The original CoT formulation showed how LLMs are able to automatically learn
the patterns underlying inputs and outputs given just a few exemplars.
Kojima et al. [39] showed that it isnot mandatoryto input the model with a set of
exemplars to obtain good quality outputs. High-level, multi-task broad cognitive ca-
pabilities may be simply extracted by prompting the model to reason about the input
before answering.
The underlying intuition is that the observed improvements in CoT pipelines are
due not just to the ability of reproducing a similar behaviour. The turning point is that
models are able to deconstruct the problem into subsections that are more likely to be
present in their pre-training dataset. To do so, it is not strictly required to skew the
LLMs towards certain examples.
They tried different prompt templates14, landing finally in a general-purposeLet’s
think step by step.
The results are not comparable with the few-shot ones in [68], despite beingquite
good.
One downside of this approach is that answers are presented in a less structured
form than the ones obtained via few-shot prompting. In fact, without a guideline on
the output format, the model is free to offer its answers in different styles.
14They trigger differently the model and observed the results; the detailed list of prompts can be found
in Table 4 in [39]
32

2.7 Emulating orunderstandingpatterns
There is a profound distance between learning a pattern present in the data and under-
standing why it is present. What large language models do is just observing a given
set of examples in their pre-training, fine-tuning datasets, few-shot examples and also
context appended to the prompt using RAG and emulating their characteristics. They
are unable to generalize over unseen examples and rare scenarios. This means that a
"LLM-student" could potentially fail in front of unseen tasks, or a similar but yet not
easily recognizable one.
These limitations first emerged when researchers began to observe unsatisfactory
behaviours of large, fine-tuned models.
One famous example is the study of Cobbe et al. [12], that compared the perfor-
mances of a fine-tuned model against a verifier on theirGSM8Kdataset.
Their experiments compared:
• a fine-tuning approach, i.e.autoregressivelysampling a single, low-temperature
solution to the task and check whether or not it is correct;
• a verification approach, i.e. sampling many high-temperature solutions from the
same model of the previous point, assigning to each of them a score via apre-
trained verifierand outputting only the highest-ranked solution.
The verifier works as follows:
Conditioned on the problem and a candidate solution, the verifier outputs the probability
that the solution is correct. Training solutions are labeled as correct or incorrect based solely on
whether they reach the correct final answer. In practice, some solutions will reach the correct
final answer using flawed reasoning, leading to false positives[12].
Figure 2.17: Figure taken from [12]. Comparison between finetuning and verification
using 6B and 175B model sizes. Mean and standard deviation are reported across 3
runs for 6B fine-tuning and verification and for 175B fine-tuning, while 175B verifica-
tion shows only a single run.
33

From their tests, they obtained two conclusions. The first one is that with small
datasets, verification tends to overfit the correct answer instead of learning more gen-
eralizable properties of correct reasoning.
The second is that 175B verifiers take off earlier than the 6B ones, i.e. they require
fewer training examples to consistently surpass the fine-tuning version [12].
Summing up, the authors’ idea was to inject ajudgein the generation pipeline that
has the responsibility of filtering out results before returning the final one. The quality
of this checker is essential to observe significant benefits.
Thetake-home messageof their analysis is thaton the full dataset, 6B verification slightly
outperforms a finetuned 175B model, thereby offering a boost approximately equivalent to a 30x
model size increase.
This work marks a significant turning point in the world of LLMs, demonstrating
that giant models are not always necessary to achieve remarkable performance. High-
quality answers can be obtained even with limited computational resources, provided
that the generated answers are evaluated for quality (in this case, correctness) before
being output.
In the following sections, we will spot a light on research works that exploit this
"check before answer" idea, although in different directions.
A first way of exploiting this consists in grounding answers in the context (2.8).
RAG approaches, by inserting relevant knowledge to the promptbeforeproducing the
output, induce the model to use that passage to answer the given prompt. By append-
ing some knowledge before starting to generate the output, we are inserting a bias in
the model’s generating process, thus performing an implicit prior check on what will
be outputted. In section (2.8) we are going to explore different ways of grounding
LLMs in relevant context.
Another possibility is to use a LLM as a verifier or as a checker of the proposed
output. This is done by self-refinement approaches, treated in section (2.10), trying to
run multiple steps in order to refine the model’s output. They consider bothad hoc
fine-tuned corrector units (2.10.2) or topic-specific prompts (2.10.1), or multiple LLMs
(2.10.3) to perform this step.
34

2.8 Grounding answers in aselectedcontext
Despite the overall quality of LLMs’ outputs, the stored knowledge in these models
may inevitably be incomplete, out-of-date, or incorrect. This motivates the need to
utilize external knowledge to assist LLMs. We have already highlighted two popular
options to do so:
• Supervised Fine-Tuning (SFT) updates the base model with new and fresh data,
even if it is costly and can potentially skew the model towards unexpected be-
haviours on tasks different from the objective;
• Retrieval Augmented Generation (RAG) appends to the prompt relevant knowl-
edge, selected through similarity scores on documents stored in a vector database.
The idea of providing (relevant) knowledge to the model to improve its output is
very broadly studied and goes beyond the two approaches presented above. This helps
significantly in improving LLM answers when they are asked to give information that
the user does not know or can’t remember or requires computation and intermediate
reasoning.
2.8.1 RE-RAG
RAG exploits a combination of parametric knowledge (those of the model) and exter-
nal knowledge (those coming from the vector database). This works thanks to the fact
that the base model has its abilities left unchanged (differently from SFT, that skews
them) and allows to access updated information by appending relevant sources to the
prompt.
However, the RAG framework suffers from performance degradation when the
query is accompanied by irrelevant contexts. These both introduce noise and repre-
sent a computational and memory overhead without achieving real benefits in terms
of answer correctness.
Kim and Lee [38] proposed RE-RAG as a method to enhance RAG benefits by fil-
tering out non-relevant context. They do so by adding an externalRelevance Estimator
(RE) module (i.e. aseq2seqmodel) to the pipeline that re-ranks contexts and provides
precise relevance scores to the generator part.
The RE receives the same input of question and context as the generator, but is
trained to generate a classification token ("true" or "false") based on the relevance of
the context to the input question.
The obtained probability of a "true" token can independently be an indicator of the
relevance of a single context to a given question:
REi,j=P(true|q i, cj)
P(true|q i, cj) +P(false|q i, cj)
We can rerank contexts in the initial retrieved setCby their relevance and only
take top-kcontexts to redefineCbefore the answer-generation step takes place. With a
35

proper RE i,jset of scores, it is possible to observe better performances of the RE-RAG
pipeline.
The probabilitiesP(true|q i, cj)can be found by training properly a neural network
to perform the classification properly.
2.8.2 Using NLI verifiers
A different approach consists in converting the answer generation task into a NLI prob-
lem. In detail, Chen et al. [11] focused on the improvement of the QA systems’ predic-
tions. To build robust question answering systems, the important task is to verify that
the answers are truly correct.
The authors attributed the basic idea of using entailment for QA to Harabagiu and
Hickl [23], even though their work was published before the advent of capable, large
language models.
NLI systems allow us to verify the level ofentailmentbetween a premise and a
hypothesis, i.e. if the first contains all necessary information to support the second.
Consequently, if we consider as premise the document context and as hypothesis the
proposed answer to the question, we can check automatically whether or not the ques-
tion was answered properly in the light of the context.
Given this setting, two pre-processing steps are necessary.
First of all, the(question, answer, context)triplet has to be converted into a(premise
, hypothesis)NLI pair.
This is done using a two-step process:
1.Question conversion:the pair(question, answer)is transformed to a declarative
statementhypothesis;
2.De-contextualization:thecontextis transformed into apremise.
For example:
Figure 2.18: Practical example of the pipeline proposed in [11].
36

Instead of using rule-based approaches, the authors preferred a neural modeling
approach to find the distributionP(d|q, a)(i.e. to convert the question) wheredis the
declarative premise,qis the question andais the candidate answer.
They chose to fine-tuneT5-3Busing a set of already annotated pairs(q, a, d)from
Demszky et al. [13]. While the conversion was trivial on many examples (e.g., replac-
ing the wh-word with the answer and inverting thewh-movement), they saw improve-
ment on challenging examples.
Ideally, the full context containing the answer candidate could be treated as the
premise to make the entailment decision. But the full context often contains many irrel-
evant sentences and is much longer than the premises in single-sentence NLI datasets.
This length has several drawbacks. First, it makes transferring models from the
existing datasets challenging. Second, performing inference over longer forms of text
requires a multitude of additional reasoning skills like coreference resolution, event
detection, and abduction.
Finally, the presence of extraneous information makes it harder to evaluate the en-
tailment model’s judgments for correctness; in the extreme, we might have to judge
whether a fact about an entity is true based on its entire Wikipedia article, which is
impractical [11].
This procedure can involve name completion (e.g.,Stewart→Kristen Stewart), noun
phrase/pronoun swap, bridging anaphora resolution, and more.
Formally, given a sentenceS aof the contextCcontaining the relevant passage to
provide the answer and the more broad contextC, the decontextualization stage learns
a modelP(S d|Sa, C), whereS dis the decontextualized form ofS a.
The decontextualizer is also a fine-tuned version ofT5-3Bmodel.
37

2.9 System 2 Attention (S2A)
In their work, Weston and Sukhbaatar [73] regenerated the input context to only in-
clude the relevant portions, before infilling the regenerated context to elicit the final
response.
This approach takes the name ofSystem 2 Attention15, and their experiments showed
that it outperforms standard attention-based LLMs on three tasks containing opinion
or irrelevant information: QA, math word problems and longform generation.
They leveraged the ability of LLMs to follow instructions, and prompted them to
generate the context that they should pay attention to, such that it contains only rele-
vant material that will not skew its reasoning.
This is due to the fact thatsoft attention tends to assign probability to a large portion of
the context, including irrelevant portions, tends to overly focus on repeated tokens partly due to
the way it is trained (Holtzman et al., 2019; Welleck et al., 2019), and partly due to the position
encoding mechanism is also inclined to treat the context as a bag-of-words when it should not
(Sinha et al., 2021; 2020)[73].
Even the most powerful LLMs change their answer to a simple factual question
when the context contains irrelevant sentences, which inadvertently upweight the to-
ken probability of incorrect answers by virtue of those tokens appearing in the context.
Figure 2.19: An illustrating example (taken from Figure 1 of [7]) showing how LLM’s
responses are adversely affected by spurious correlations in the context. Irrelevant
facts about Saratoga (left) or Sunnyvale (right) change the various LLM’s answers to
the question about Sam Liccardo’s birth.
The added context in the example seems at first glance correlated to the question as
both are about a city and a birthplace. But with deeper understanding, it is clear that
15From the paper:We refer to this procedure as System 2 Attention (S2A), because we can consider the
underlying transformer, and its attention mechanism, as automatic operations analogous to system 1 reasoning in
humans (Kahneman, 2011). System 2, allocating effortful mental activity, takes over in humans when we need to
pay deliberate attention to a task, especially in situations where System 1 is likely to make errors (Sloman, 1996)
[73].
38

the added text is irrelevant, and thus should be ignored.
The termAttentionshould not deceive: the structural, low-level, causal self-attention
mechanism is something radically different from the authors’ proposal. TheirSystem
2 Attentionrefers to the process of employing instruction-tuned LLMs to rewrite the
context by removing irrelevant parts of it.
The typical scenario in which a Large Language Model is given a context, denoted
asx, and its objective is to generate a high-quality sequence, denoted asy, can be
referred toy∼LLM(x).
System 2 Attentioninstead is a two-step process:
1. Given the contextx, S2A first regenerates the contextx′such that irrelevant parts
of the context that will adversely affect the output are removed:x′∼S2A(x) =
LLM(P S2A(x)), whereP S2Ais a function that generates a zero-shot prompt to the
LLM instructing it to perform the desired S2A task overx;
2. Givenx′, we then produce the final response from the LLM using the regenerated
context instead of the original one:y∼LLM(x′).
An example ofP S2Athat they employed is:
Given the following text by a user, extract the part that is unbiased and not their
opinion,
so that using that text alone would be good context for providing an unbiased answer
to the
question portion of the text.
Please include the actual question or query that the user is asking.
Separate this into two categories labeled with "Unbiased text context (includes all
content except user’s bias):" and "Question/Query (does not include user bias/
preference):".
Text by User: [ORIGINAL INPUT PROMPT]
Typically, some post-processing may also be applied to the output of step 1 in order
to structure the prompt for step 2, as instruction following LLMs produce additional
chain-of- thought reasoning and comments in addition to requested fields.
39

Figure 2.20: An example from the modified TriviaQA from SycophancyEval where the
added opinion in an otherwise fact-seeking query makesLLaMA-2-70B-chatanswer in-
correctly (left). S2A (right) regenerates the part of the context it decides to pay attention
to, removing the opinion that might adversely affect the final response, then hence an-
swering correctly.
This automated approach for all the experimented tasks almost reaches theoracle
performances (i.e. the unopinionated, correct prompt), highlighting the positive im-
pact of a skimmed context in the correctness of the generated output.
2.10 Self-refinement approaches
2.10.1SELF-REFINEalgorithm
Recently, a different approach has been proposed to mimic human behavior when per-
forming a task. The abstract of Maadan et al.’s work [44] begins with:
Like humans, large language models (LLMs) do not always generate the best output on their
first try. Motivated by how humans refine their written text, we introduce SELF-REFINE, an
approach for improving initial outputs from LLMs through iterative feedback and refinement.
The authors’ approach involves assigning the same LLM three roles: thegenerator,
therefiner, and thefeedback provider. The initial output from the LLM is reviewed by the
same model, which is prompted to critique or evaluate its initial answer. This feedback
is then incorporated into a new prompt template, along with the initial response, to
generate a refined answer.
40

Figure 2.21: Theself-refinealgorithm as presented in [44]. The refinement process can
stop after a given number of iterationst∈0,1, ...or when a stop conditionstop(fb t, t)
is met.
They tested this process on various tasks, including sentiment reversal, dialogue
response, code optimization, readability improvement, math reasoning, acronym gen-
eration, and constrained generation. The models used wereGPT-3.5,GPT-4, andChatGPT;
interestingly, the latter performed comparable to the other options16.
To assess the correctness or quality of the generated outputs, the authors used an
average of different scores:
• For all tasks, a score given byGPT-4, used as a proxy for human preference;
• For dialogue response generation, code readability improvement, sentiment re-
versal, and acronym generation, a score derived from a blind human A/B evalu-
ation on a subset of outputs, selecting the preferred output;
• For other tasks, automated metrics from prior work: specifically, the percentage
solve rate for math reasoning, the percentage of programs optimized for code
optimization, and the coverage percentage for constrained generation.
Another notable finding from their results is that the improvements in math rea-
soning using their approach were minimal or nonexistent, while for more qualitative
tasks, their method had significant beneficial effects.
This is due to the qualitative approach of the answer improvement. Their work
was not focused on producing a more accurate answer, while on refining the initial
attempt. None of their prompt actually aims at verifying the correctness of the output,
what their goal is instead to achieve somedesirable property(e.g. safety, clearness, effi-
ciency) that the first try may lack17. This explanation is also the reason for whichChatGPT
does not underperform more capable models: this task does not focus on improving
the answer with relevant knowledge or meaningful reasoning, thus a conversational
model could still behave properly.
16Results can be found in Table 1 in [44].
17Refer to appendix A of [44] for further details on the prompts used to answer, to give feedback and
finally to refine the output, for each task considered.
41

2.10.2SELF-CORRECTIONalgorithm
As inSELF-REFINE, Welleck et al. [72] proposedSELF-CORRECTOR, a method that decouples
an imperfect base generator (an off-the-shelf language model or supervised sequence-
to-sequence model) from a separate corrector that learns to iteratively correct imperfect
generations.
Powerful generation models often meet most of the task requirements, yet miss a
few (e.g., omitting a subset of keywords), or generate incorrect hypotheses that never-
theless provide useful structure (e.g., a correct problem solving strategy with a missing
step). However, after generating even a slightly sub-optimal sequence, the single-pass
paradigm requires models to “start from scratch”, effectively discarding work already
done.
To avoid this scenario, it is possible to leverage the generation as a useful starting
point to refine into a higher quality output.
A generation model is re-framed as a basegenerator, which produces a reasonable
initial hypothesis but does not need to solve the task in one pass, and a second module,
thecorrector, trained to make up the difference between the hypothesis and an optimal
solution.
Note that neither thegeneratornor thecorrectormust solve the full task in one pass,
and thecorrectorcan be applied multiple times to iteratively improve the output.
They tested the corrector approach on 3 diverse tasks: mathematical program syn-
thesis, lexical constrained generation, and toxicity reduction.
Figure 2.22: TheSELF-CORRECTIONprocedure, as presented in [72].
The trainedcorrectormodel can even be applied to a largergeneratorwith similar
performance to training a new corrector, showing that the sub-task of correction is
transferable, even to strongergenerators.
In addition to this, thecorrectormodule can be trained for different objectives, keep-
ing the samegenerator. This allows flexibility, modularity and composition, leading to
a larger field of applications.
The corrector is trained by generating a set of hypotheses and relative corrections.
Thegeneratorfirstly generates a lot of pairs that the naive version of thecorrectoris
asked to value; a set of value-improving pairs is formed (i.e. examples of mapping
a hypothesis to a higher-valued correction); self-corrective learning selects(input,
hypothesis, correction)pairs to update thecorrectorwith.
42

In detail, the(input, hypothesis, correction)triplet is sampled proportional to its
improvement in value, as well as the proximity between thehypothesisand thecorrection
.
In algorithmic terms:
Figure 2.23: Notation and further details can be found in [72].
They achieved quite good results on arithmetic and mathematical datasets (Multi-
Arith, Multitask), despite observing only a relative improvement on GSM8k. Results
are observed using older versions ofGPT(GPT-3.5was not yet released).
2.10.3 Reflexion
Almost simultaneously, Shinn et al. [59] developedReflexion, a modular, 3-units ap-
proach made by:
• an Actor modelM a, generating text and actions;
• an Evaluator modelM e, scoring the output provided byM a;
• a Self-Reflection modelM sr, generating verbal reinforcement cues to assist the
Actor in self-improvement.
The Actor is built upon a large language model that is specifically prompted to gen-
erate the necessary text and actions conditioned on the state observations. Analogous
to traditional policy-based reinforcement learning setups, they sampled an action or
generationa tfrom the current policyπ θat timetand receive an observation from the
environmento t. They also keep memorymemas additional context.
This adaption was inspired by Brooks et al. [brooks], who suggest a policy iteration
approach using in-context learning
As Actor models they considered both Chain-of-Thought and ReAct.
The Evaluator component of the Reflexion framework plays a crucial role in assess-
ing the quality of the generated outputs produced by the Actor. It takes as input a
generated trajectory and computes a reward score that reflects its performance within
the given task context.
The issue is that defining effective value and reward functions that apply to seman-
tic spaces is difficult. They experimented with different settings.
43

Given a sparse reward signal, such as a binary success status (success/fail), the
current trajectory, and its persistent memorymem, the self-reflection model generates
nuanced and specific feedback. This feedback, which is more informative than scalar
rewards, is then stored in the agent’s memory (mem).
For instance, in a multi-step decision-making task, when the agent receives a failure
signal, it can infer that a specific actiona iled to subsequent incorrect actionsa i+1and
ai+2. The agent can then verbally state that it should have taken a different action,a′
i,
which would have resulted ina′
i+1anda′
i+2, and store this experience in its memory.
In subsequent trials, the agent can leverage its past experiences to adapt its decision-
making approach at timetby choosing actiona′
i. This iterative process of trial, er-
ror, self-reflection, and persisting memory enables the agent to rapidly improve its
decision-making ability in various environments by utilizing informative feedback sig-
nals.
At inference time, the Actor conditions its decisions on short and long-term mem-
ory, similar to the way that humans remember fine-grain recent details while also re-
calling distilled important experiences from long-term memory.
In the RL setup, the trajectory history serves as the short-term memory while out-
puts from the Self-Reflection model are stored in long-term memory.
Figure 2.24: A diagram representing Reflexion (left) and the corresponding algorithm
(right), taken from [59].
In the first trial, the Actor produces a trajectoryτ 0by interacting with the environ-
ment. The Evaluator then produces a scorer 0which is computed asr t=M e(τ0).rtis
only a scalar reward for trialtthat improves as task-specific performance increases.
After the first trial, to amplifyr 0to a feedback form that can be used for improve-
ment by an LLM, the Self-Reflection model analyzes the set ofτ 0, r0to produce a sum-
marysr 0which is stored in the memorymem.
srtis a verbal experience feedback for trialt. The Actor, Evaluator, and Self-Reflection
models work together through trials in a loop until the Evaluator deemsτ tto be correct.
44

2.11 Reasoning on the context
2.11.1 Rethinking with retrieval
We have already spotted a light on the benefit provided by appending meaningful and
relevant passages to the prompt before asking the model to generate its output.
LLMs have been shown to generate incorrect supporting facts from time to time,
even when they accurately capture the perspective needed to answer a question.
This phenomenon highlights intrinsic issues in the way LLMs store and retrieve
knowledge, including:
1. the presence of out-of-date, incorrect, or missing relevant knowledge in the pre-
training corpus;
2. incorrect memorization of relevant knowledge during pre-training;
3. incorrect retrieval of relevant knowledge during the inference stage.
He et al. [24] enter the debate with a post-processing approach calledRethinking
with Retrieval(RR) which retrieves relevant external knowledge based on the decom-
posed reasoning steps obtained from the Chain-of-Thought (CoT) prompting.
The main advantages of this solution are that:
• it does not require additional training of fine-tuning;
• it is not limited by the input length of LLMs.
Figure 2.25: An overview of three approaches for using LLMs: (a) Standard prompting
for generating a prediction in response to a query. (b) Chain-of-Thought prompting for
generating both an explanation and a prediction in response to a query. (c) Rethinking
with retrieval, our proposed approach for using the decomposed reasoning steps ob-
tained from CoT prompting to retrieve relevant external knowledge for LLMs, leading
to more faithful explanations and improved predictions in response to a query.
45

Their method began by using CoT prompting [68] to generate a diverse set of rea-
soning paths, as described in [67]. After that, they used each reasoning step in those
paths to retrieve relevant external knowledge, which enables RR to provide more faith-
ful explanations and more accurate predictions.
More formally, given a queryQ, they used CoT prompting to generate a diverse set
of reasoning pathsR 1, R2, ..., R N(where each reasoning pathR iconsists of an explana-
tionE ifollowed by a predictionP i).
Consequently, relevant knowledgeK 1, K 2, ..., K Mis retrieved from a suitable knowl-
edge baseKBto support the explanation in each reasoning path, and select the pre-
diction ˆPthat is most faithful to this knowledge.
For example:
Q: Did Aristotle use a laptop?
R_1: Aristotle died in 2000. The first laptop was invented in 1980.
Thus, Aristotle used a laptop. So the answer is yes.
R_2: Aristotle died in 322BC. The first laptop was invented in 2000.
Thus, Aristotle did not use a laptop. So the answer is no.
R_3: Aristotle died in 322BC. The first laptop was invented in 1980.
Thus, Aristotle did not use a laptop. So the answer is no.
K_1: Aristotle (384-322 BC) was a Greek philosopher and polymath during the Classical
period
in Ancient Greece. ...
K_2: The Epson HX-20, the first laptop computer, was invented in 1980. ...
The faithfulness of each reasoning path is evaluated using a functionf KB(Ri), which
is based on relevant knowledgeK 1, K 2, ..., K Mretrieved from the knowledge baseKB.
ˆPis chosen as:
ˆP= arg max
Pi∈P1,...,P NNX
i=11(Pi=P)f KB(Ri)
For instance, in the running example, given reasoning pathsR 1,R 2,R 3and the re-
trieved knowledgeK 1,K 2, the above inference procedure would output the prediction
So the answer is no, as it is supported by bothR 2andR 3and has a higher faithfulness
score compared to the predictionSo the answer is yes, which is only supported byR 1.
Better Multi-Hop Reasoners
Li et al. [42] confirms the models’ decreased performance in the presence of noisy
contexts, but also highlight how they struggle with multi-hop reasoning tasks.
Their approach,Reasoning with Attributions, prompts the model to supply attribu-
tions for each assertion during their reasoning. This is a strategy that mandates lan-
guage modelsto link the claimsmade during reasoning to specific sections of the pro-
vided context. This implicit requirement effectively decomposes a complex multi-hop
question into two more manageable tasks:
• Pinpointing pertinent information within the context;
46

• Constructing well-founded claims based on that information.
They adapted the CoT prompting to create two variants aligned with their (attribution-
based) approach.
The first isChain-of-Citations(CoC), in which models are prompted toreference
citationscorresponding to each step of the reasoning chain.
The second,Chain-of-Quote(CoQ), goes further by requiring models toinclude di-
rect quotationsfrom the cited material for each reasoning step.
Figure 2.26: An example of CoT, CoC and CoQ taken from [42]. Answers are marked
in green, citations are marked in orange and quotes are marked in blue.
The findings suggest that both CoC and CoQ generally yield improvements over
CoT, indicating that attribution-based reasoning enhances the precision and coherence
of the models’ reasoning processes. CoQ appears to slightly underperform CoC, likely
due to the increased complexity of producing exact quotations [42].
MIRAGE
Ensuring the verifiability of model answers is a central task in the LLMs field of stud-
ies. Recently, researchers spotted that prompting the model to produce self-citations
that ground the answers in the supporting context can help improve the answers’ cor-
rectness. Grounding answers in the context is also an option that helps the users to
47

check that the model is notright for the wrong reasons, and that the reasoning chain that
the model produces is not absurd.
But self-citation methods often struggle to match the required format, refer to non-
existing sources and their faithfulness is very difficult to evaluate.
Model Internals-based RAG Explanations (MIRAGE) [52] extends the Plausibility
Evaluation for Context Reliance (PECORE) framework [58] for context-aware machine
translation. It detects context-sensitive answer tokens through saliency methods, pair-
ing them with retrieved documents contributing to their prediction.
Compared to self-citation, it allows a more fine-grained control on how the attribu-
tion stage is performed.
Figure 2.27: An example of self-citation weaknesses, taken from [52].
The context-sensitive tokens in thegenerated sequenceare spotted by measuring the
shift in the model’s predictive distribution caused by the addition of the input con-
text. This shift is attributed to some tokensin the context, found to be influential to the
generated output.
This approach has already been employed in machine traslation tasks [58], the au-
thors expanded it to the RAG framework. Their idea is to search for a match between
context-dependent tokens in the generated sequence and the retrieved documents that
effectively contribute to their prediction. Finally, these paired elements are converted
to citations.
48

To spot whichgeneratedtokens are sensitive to the given context, the model needs
to be prompted with a queryqand a contextC={c 1, ..., c |C|}18in order to produce a
sentencey= (y 1, ..., y n)as output.
A contrastive metricˆm(e.g. KL-divergence) is used to quantifyat each generation
stepthe shift between:
•Pi
ctx: the model’s predictive distribution at thei-th generation step with a prompt
that includes the context;
•Pi
no−ctx : the model’s predictive distribution at thei-th generation step with acon-
textlessprompt.
Thus, at each generation step, we use the metricˆmto compute a score:
mi= ˆm(Pi
ctx, Pi
no−ctx ),e.g.m i=KL(Pi
ctx||Pi
no−ctx )
The resulting (i.e. at the end of the generation process) scoresm= (m 1, ..., m n)reflect
the context sensitivity of each generated token.
To extract the most relevant ones, we can use a choice functions CTIoutputting
whether each generated item issufficiently context-sensitiveor not:
CTI(q, c, y) ={y i|sCTI(mi) = 1,∀y i∈y}
whereCTI(q, c, y)is an array containing all the relevant generated tokensy i.
Once the context-sensitive tokensy ihave been identified, we also make the model
predict a contrastive alternativey\c
iby excluding the contextCfrom the prompt but
using the contextual generated prefixy <i.
Now we have for each generation step a set of couples(y i, y\c
i)representing the out-
put produced with and without context. This first step has identifiedwhich generated
tokens change when the context is injectedin the prompt. We refer to this step asContext-
sensitive Tokens Identification(CTI), and consequently to the elements that this process
finds ascontext-sensitive tokensy i∈CTI(y).
The second step of this attribution procedure aims at discovering exactlywhich con-
text tokens are the ones that impact the most on the model’s outputs.
We refer to this process asContextual Cues Identification(CCI), and consequently to
these tokens ascontextual cuesc j∈CCI(y i).
In order to identify which are the most important context tokens, acontrastive feature
attribution method[80] can be applied:
aj
i={∇ j(p(y i)−p(y\cj
i)), c j∈C}
It substantially measures which context tokensc j, if removed from the context,
cause a bigger modification in the LLM’s predictive function.aj
iidentifies which con-
text itemsc j∈Cinfluence the prediction ofy i, accounting for the non-contextual com-
pletiony\c
i.
18Eachc ican be a separate document, a separate paragraph, a separate phrase, ... It depends on the
granularity of the employed RAG application.
49

These scores are transformed into binary labels similarly as it has been done fory i
in theCTIstep, i.e. by using a choice functions CCIoutputting whether eachcontextual
cueissufficiently influentialfor the generated output or not:
CCI(y i) ={c j|sCCI(aj
i) =∀c j∈C}
This process results in pairs of context-sensitive generated tokens (CTI) and their
respective context tokens influencing their prediction (CCI):
{(yi, cj),∀y i∈CTI(q, c, y),∀c j∈CCI(y i)}
Note that boths CTIands CCIcontain implicitly a threshold which discriminates
from what point onwards the tokens (both generated and in the context) are considered
relevant. We can define them as:
sCTI=m i≥m∗, s CCI=aj
i≥a∗
i
Practically, the authors suggested to setm∗= ¯m+σ¯m, where¯mandσ¯mare respec-
tively the average and standard deviation ofm iscores for the given example.
To filter the attributed context tokensc j∈CCI(y i),a∗
iis either the Top-Kor Top-%
highest attribution value ina i.
The final step proposed by the method consists in generating the citations. This
step builds over the previous identification of relevant tokens in terms of generation.
We can construct the citation by selecting all the documents containing acontextual
cues(CCI)c jthat are paired with acontext-sensitive generated output token(CTI)y i:
MIRAGE(y) =[
yi∈CTI(y)docid(c j),∀c jCCI(y i)
50

2.11.2 Literature connections with our method
This long and extensive literature review presents multiple elements that will be help-
ful in presenting our experiments and in describingwhereour method is placed.
The first part (2.1) describes in detail the Transformer model’s architecture and its
modern improvements (2.2).
We consider these sections as preliminary to the description of the models we tested
(2.3). With this introduction, we believe that many structural variations can be better
understood. Tested models’ details are essential to perform later analyses on (even-
tual) performance discrepancies. This should allow us to correctly spot the sources of
different performance gaps among them, highlighting when they could be caused by
our method or when they are external to it.
Then, we begin an overview of proposed methods to enhance generation quality.
Besides pre-training stages, the models can be fine-tuned (2.4) to align their predictions
with certain domain-specific applications or to give them the ability to answer in a pre-
determined way to the given prompt.
A different approach has also been exploited to provide new information to the
model, in order to update its knowledge or to provide additional one. The RAG ap-
proach (2.4) dynamically retrieves passages useful to answer to a question and ap-
pends them to the prompt. These appended additions are used to guide the model
towards a grounded output. Further studies (2.8.1, 2.9) show how selecting the rel-
evant information before appending it to the prompt can improve models’ ability of
exploiting context in the proper way, since it could be misleaded by noisy and irrele-
vant elements. A different approach (2.8.2) proposes to rephrase this framework in a
NLI problem by transforming the prompt into ahypothesisand the provided context as
apremisefor it.
Fine-tuning, however, is not the only way of achieving tan alignment of model’s
behaviour with a desired one. It can be obtained via in-context learning (2.5), that
simply provides "few-shots" examples appended to the prompt, showing the model
how to perform a certain task.
Inducing the model to behave in a pre-determined way as in-context learning (2.5)
allows the prompt to show the model how to execute a task, and consequently it
can also be used to present areasoning chainuseful to solve the problem. Studies
have shown that providing models with reasoning examples significantly improves the
quality of their answers (2.6). This approach is known as Chain-of-Thought prompting.
The implicit idea of CoT is that the model isallowed(or, more properly,prompted to)
take time and words to develop intermediate steps before choosing an answer. How-
ever, this is not the only option to producenot definitive, intermediate stepsbefore out-
putting the final answer. In the wake of the method showed in section (2.8.2) ranked
the quality of the candidate outputs before choosing the correct one (proved to outper-
form large fine-tuned alternatives), self-refinement methods (2.10) implement a certain
number of corrections (2.10.2) or refinements (2.10.1, 2.10.3) to achieve a better answer.
Finally, some studies experiment beneficial effects of grounding reasoning in rele-
51

vant context have been reported in this section (2.11). Among these, we find interest-
ing the Rethinking with Retrieval approach (2.11.1), that produces multiple reasoning
chains and consequently selects as answer only the most grounded one and the Chain-
of-Citations/Chain-of-Quotes method (2.11.1), that induce the model to construct well-
founded claims as reasoning chains by hinging them on context citations/quotes.
Our method builds on the joint efforts of the research community to which the
creators of these methods belong. Specifically, we owe them the idea of improving a
first tentative answer as it is done in self-refinement methods (2.10), although we do
not consider our approach as arefiningapproach, nor acorrectionone.
We also integrated the context in our approach, relying on the observations re-
ported in (2.8) and we also studied the influence that the original context has in the
generated output, compared to its summarized version and to its filtered one. The fil-
tering is carried out thanks to a slight variation of the MIRAGE (2.11.1) method: since
it is able to pair the generated outputs with the context elements that influenced them,
we can discard irrelevant parts of the passages appended to the prompt.
Finally, our method includes Chain-of-Thought steps in order to allow the model
to reason on the correctness of a tentative answer and to eventually correct it.
52

Chapter 3
Data details
Our analysis aims to improve the robustness of LLMs’ answers. We focus on studying
their ability to correctly respond to general knowledge questions in various settings,
such as standard prompting, CoT prompting, a RAG setup, and combinations of these
approaches.
Our method proposes todialectically improvingthe answer before outputting it. In
order to run these tests, we have to rely on datasets such that:
1. contain a set of questions and the correct answers to these questions;
2. append the relevant passages or documents that can be used to answer correctly
to the question;
3. eventually provide also wrong answers to the question, presented asdistractors.
In addition to this, we aim to test the ability of LLMs once the correct relevant pas-
sages are provided. Extensive studies (reported in 2.8) have shown how models benefit
from the addition of the relevant context to their prompts, thus we will build on this
finding and try to further enhance performances.
What instead we find interesting to study is the impact of the work that the model
has to perform on the sources. We aim to test cases in which instead it is necessary to
merge multiple pieces of information before answering. This task can be referred with
the name ofMulti-Hop Question Answering(MHQA) [46].
In broad terms, MHQA is the task of answering natural language questions that
involve extracting and combining multiple pieces of information and doing multiple
steps of reasoning. An example of a multi-hop question would be:
Who is the oldest candidate in the 2024 USA presidential election?.
Answering the question would require to join many pieces of information:
•What are the names of presidential candidates to the 2024 USA presidential election
?
•What is the age of [candidate name]?
•What is the largest number between [age1], [age2], ...?
The ability to answer multi-hop questions and perform multi step reasoning can
significantly improve the utility of NLP systems.
53

Single-hop QAoften does not require any form of reasoning, limiting itself to sum-
marizing or paraphrasing the content present in the source to produce a proper answer
to the question.
Multi-hop QAasks the model to perform a step further. An agent can be said to
performmulti-hop reasoningif it reaches one or more intermediate conclusions before
deriving the final answer and each of the intermediate conclusions serves as a neces-
sary premise for some other conclusion. This sequence of intermediate conclusions, in-
cluding the final answer, is called a reasoning chain and each step from one conclusion
to the next can be referred to as ahop. Humans can easily perform these multi-step rea-
soning in their everyday tasks, yet this is still a difficult task for machines. We would
like to improve LLMs’ multi-hop abilities since they could be useful in many concrete
applications. Queries given to current web search systems can often require multi-hop
reasoning to reach the relevant documents; user satisfaction when using such systems
can be greatly improved by utilizing multi-hop reasoning models; also conversations
between humans and agents can be smoother and more informative if the latter can
handle complex questions.
Our analysis starts from the choice of datasets. We are interested in studying how
our method performs on HotpotQA and WikiHop datasets. As their names can clue,
they both require the model to perform somemulti-hopreasoning steps before answer-
ing.
54

3.1 HotpotQA
3.1.1 Dataset description
HotpotQA is a question answering dataset collected on the English Wikipedia, con-
taining about 113K crowd-sourced questions that are constructed to require the intro-
duction paragraphs of two Wikipedia articles to answer. Each question in the dataset
comes with the two gold paragraphs, as well as a list of sentences in these paragraphs
that crowdworkers identify as supporting facts necessary to answer the question.
Yang et al. [79] constructed HotpotQA ensuring that 4 key features are guaranteed:
1. the questions require finding and reasoning over multiple supporting documents
to answer;
2. the questions are diverse and not constrained to any pre-existing knowledge
bases or knowledge schemas;
3. sentence-level supporting facts required for reasoning are provided, allowing QA
systems to reason with strong supervision and explain the predictions;
4. a new type of factoid comparison question is introduced to test QA systems’ abil-
ity to extract relevant facts and perform necessary comparisons.
HotpotQA contains only 2-hopquestions formed using the first passages of docu-
ments from the English Wikipedia dump. The passages are chosen if they satisfy either
of the two conditions:
• There exists a hyperlink from the first document to the second. The entity which
forms the hyperlink is termed as thebridgeentity and the questions are termed
as bridge questions.
• The entities for those passages belong to the same category (e.g. Michael Jordan
and Kobe Bryant). These are specifically sampled from 42 manually created lists.
Such pairs are used for creatingcomparisonquestions [46].
An example of HotpotQA is the following:
{’id’: ’5a7a06935542990198eaf050’,
’question’: "Which magazine was started first Arthur’s Magazine or First for Women?",
’answer’: "Arthur’s Magazine",
’type’: ’comparison’,
’level’: ’medium’,
’supporting _facts’: {’title’: ["Arthur’s Magazine", ’First for Women’], ’sent _id’:
[0, 0]},
’context’: {
’title’: [’Radio City (Indian radio station)’, "Arthur’s Magazine", ... ],
’sentences’: [
["Radio City is India’s first private FM radio station and was
started on 3 July 2001.", ...] , ...
]
}
}
55

Where:
•idfor the question-answer couple;
•question,answerare simply the question and the correct answer to the first;
•typeis the required type of reasoning on the context:comparisonandbridge(details
are provided below);
•levelis a human-labelled score based on how challenging the question is; possi-
ble values of it areeasy,mediumandhard;
•contextcontains 10 documents, of which 8 distractors and 2 relevant ones. For
each document/passage are provided:
–thetitleof the Wikipedia paragraph;
–a vector ofsentencesidentified by the crowdworkers as relevant;
•supporting _factscontains thetitleof the 2 gold documents of the 10 provided.
From now on, we will refer separately to the subset of dataset containingcomparison
andbridgetype of questions. We do this distinction since we consider important to test
separately these two differentmulti-hopsub-tasks. Whilecomparisonrequires to extract
(almost) the same information from passages referring to two different objects,bridge
tests the ability to merge multiple sources in a homogeneous view that allows to an-
swer to the question.
We show the difference between the two tasks using two examples:
•comparison:
Question: Which magazine was stated first, Arthur’s Magazine or First for Women?
Answer: [Arthur’s Magazine]
Context: Arthur’s Magazine (1844-1846) was an American literary periodical
published
in Philadelphia in the 19th century. Edited by T.S. Arthur, it featured work by
Edgar A.
Poe, J.H. Ingraham, Sarah Josepha Hale, Thomas G. Spear, and others. In May 1846
it was merged into "Godey’s Lady’s Book". First for Women is a woman’s magazine
published
by Bauer Media Group in the USA. The magazine was started in 1989. It is based in
Englewood Cliffs, New Jersey. In 2011 the circulation of the magazine was
1,310,696 copies.
•bridge:
Question: The Oberoi family is part of a hotel company that has a head office in
what city?
Answer: [Delhi]
Context: The Oberoi family is an Indian family that is famous for its involvement
in hotels,
56

namely through The Oberoi Group. The Oberoi Group is a hotel company with its
head office in
Delhi. Founded in 1934, the company owns and/or operates 30+ luxury hotels and
two river
cruise ships in six countries, primarily under its Oberoi Hotels & Resorts and
Trident
Hotels brands.
3.1.2 Data processing
The necessary transformation that the dataset had to meet was to create another plau-
sible answer to the question.
Due to the structure of thecomparisonquestions, we find natural to add only one
plausible alternative, practically the other option given by the question.
Consider for example:
Which magazine was stated first, Arthur’s Magazine or First for Women?
it is clear (even without knowing the correct answer) that the only two plausible op-
tions are
Arthur’s MagazineandFirst for Women.
We had at disposal thetitlearray (inside thecontextattribute) and thetitleof the
only two relevant passages (inside thesupporting _factattribute). Due to what we want
to study, we chose to neglect a detailed analysis on the introduction of non-meaningful
passages in our pipeline.
Thus, we kept only thesentencesitems corresponding to the relevant documents,
and merged them in a single text passage.
We could not use thetitleof relevant passages as couples of correct and wrong an-
swers to the questions. While some cases allowed us to build an automatic procedure
to do so, e.g.
Which magazine was stated first, Arthur’s Magazine or First for Women?
[("Arthur’s Magazine",
["Arthur’s Magazine (1844-1846) was an American literary periodical published in
Philadelphia in the 19th century.", ’ Edited by T.S. Arthur, it featured work by
Edgar A.
Poe, J.H. Ingraham, Sarah Josepha Hale, Thomas G. Spear, and others.’,
’In May 1846 it was merged into "Godey\’s Lady\’s Book".’]
),
(’First for Women’,
["First for Women is a woman’s magazine published by Bauer Media Group in the USA
.",
’The magazine was started in 1989.’, ’ It is based in Englewood Cliffs, New Jersey
.’,
’In 2011 the circulation of the magazine was 1,310,696 copies.’]
)
]
It is clear that in front of the question of the current example, it would be enough
to extract
57

Arthur’s MagazineandFirst for Women. On the opposite, another example is the fol-
lowing:
Which band was founded first, Hole (the rock band that Courtney Love was a frontwoman
of) or The Wolfhounds?
[(’The Wolfhounds’,
[’The Wolfhounds are an indie pop/noise pop band formed in Romford, UK in 1985 by
Dave
Callahan, Paul Clark, Andy Golding, Andy Bolton and Frank Stebbing, and originally
active until 1990.’, ’ The band reformed in 2005 and continues to write, record and
play live, releasing new albums in 2014 and 2016.’]
),
(’Courtney Love’,
[’Courtney Michelle Love (born Courtney Michelle Harrison; July 9, 1964) is an
American
singer, songwriter, actress, and visual artist.’, ’Prolific in the punk and grunge
scenes of the 1990s, Love has enjoyed a career that spans four decades.’, ’She rose
to
prominence as the frontwoman of the alternative rock band Hole, which she formed in
1989.’,
’Love has drawn public attention for her uninhibited live performances and
confrontational lyrics, as well as her highly publicized personal life following her
marriage to Kurt Cobain.’]
)
]
Despite being relevant to the answer,Courtney Lovewould not be a plausible alter-
native answer since she is a singer, not a band. The correct alternative would beHole,
which cannot be directly extracted from the contexttitle.
We usedPhi-3-minito produce a plausible alternative to the correct answer to the
question. We use greedy decoding with temperature set to 0 to sample an alterna-
tive option, limiting the maximum number of new tokens to 201. The prompt used to
generate the output is one-shot and is the following:
def produce _prompt(question, correct, source):
user _content = "Question: " + question + "\n Correct answer: " + correct + \
"\n Context: " + source + "\n\n Assistant:"
messages = [
{"role": "system", "content": """
You are a helpful AI assistant. You are given a question and the correct
answer to it.
Given the context, you have to provide a wrong, yet realistic, alternative
answer to
the same question given the context.
Output a synthetic answer in the same style as the correct answer.
1This is because of the answer style in HotpotQA, very synthetic and essential. We did not need
the model to produce an output and justify it, but instead to extract from the context another realistic
option.
58

For example:
Question: Which magazine was started first Arthur’s Magazine or First for
Women?
Correct answer: Arthur’s Magazine
Context: Arthur’s Magazine (1844-1846) was an American literary periodical
published in
Philadelphia in the 19th century. Edited by T.S. Arthur, it featured work by
Edgar A.
Poe, J.H. Ingraham, Sarah Josepha Hale, Thomas G. Spear, and others. In May
1846 it was
merged into Godey’s Lady’s Book. First for Women is a woman’s magazine
published
by Bauer Media Group in the USA. The magazine was started in 1989.
It is based in Englewood Cliffs, New Jersey. In 2011 the circulation of the
magazine
was 1,310,696 copies.
Assistant: First for Women
"""},
{"role": "user", "content": "Now to the same for this problem: " +
user _content},
]
return messages
This function allows to append to the prompt an example and then to ask the model
to perform a similar task for a new triplet of question, correct answer and context.
In thecomparisonsubset, questions are also presented in a different style, that does
not explicitly refer to multiple options, for example:
750 7th Avenue and 101 Park Avenue, are located in which city?
In this example, both the correct answer and the alternative do not stand out in the
question. They need to be extracted from the context and the correct answer instead.
The prompt presented above still works quite well for this type of questions, even
though the context does not explicitly mentions any plausible alternative toNew York
City:
[(’101 Park Avenue’,
[’101 Park Avenue is a 629 ft tall skyscraper in New York City, New York.’, ’It was
completed
in 1979 to 1982 and has 49 floors.’, ’Eli Attia Architects designed the building,
which is
the 64th tallest in New York.’]),
(’750 7th Avenue’,
[’750 Seventh Avenue is a 615 ft (187m) tall Class-A office skyscraper in New York
City.’,
’It was completed in 1989 in the postmodern style and has 36 floors.’, ’Kevin Roche
John
Dinkeloo & Associates designed the building, and it is owned by Hines, a Texas based
real
59

estate investment company.’, "The building’s continuous helix design, culminating in
a
chimney-like extension, was caused by the New York City Building Code, which
requires
setbacks.", "The 84 exterior column transfers exist because of the owner’s
requirement
for a column-free space.", ’It is tied with the New York Life Building for the 74th
tallest building in New York City.’, ’It is also LEED certified.’]
)
]
In most of the examples,Phi-3-miniis able to use its parametric knowledge to extract
some plausible alternative. A hand-crafted correction is applied to cases in which the
model produces clearly wrong options. For example, a plausible alternative toNew York
Citywould not beTrieste, since the latter does surely not possessavenues.
Another option, e.g.San Francisco, is more realistic and challenging to be checked
(if the context is not used properly or neglected).
The last kind of alternatives that can be found is the basicyesandnoquestions. For
this partition ofcomparison, that can be easily selected by considering all the rows in
whichansweris one betweenyesandno, the alternative is simply the opposite. This
does not require the use of any Large Language Model.
Due to the need for a hand-crafted correction in some cases, we limited the dataset
to the first 352 examples of the training set. We do not select the problems according to
any different criterion and we checked that the models tested were not pre-trained or
fine-tuned on this subset2.
For our experiments, this number of items is enough to assess whether we experi-
ence an improvement or not. Generating alternatives withPhi-3-miniis not extremely
expensive in computational terms, but requires also a hand-crafted correction some-
times. Also, we chose to keep the dataset balanced and not to add all theyes/noques-
tions present in the original dataset.
When the same prompt is tested on thebridgesubsection of the dataset, we observe
particularly good results, that do not need to be hand-crafted as a post-processing
stage.
Concrete examples of non-trivial alternatives created usingPhi-3-minionbridge
subset:
What U.S Highway gives access to Zilpo Road, and is also known as Midland Trail?
Zilpo Road is a National Forest Scenic Byway in the forested hills of eastern Kentucky
,
United States. The nine mile byway starts south of Morehead, Kentucky and can be
accessed
by U.S. Highway 60. The byway travels through the Daniel Boone National Forest and
ends on
2See the Results chapter: the baseline performances are not good, both with and without the relevant
passages appended to the prompt, thus it is improbable that the models were trained on that data.
60

the western shore of Cave Run Lake at the Zilpo Recreation Area. It follows FSR 918,
which
is a two lane paved road suitable for all motor vehicles and is usually open
throughout the
year. Morehead is a home rule-class city located along US 60 (the historic Midland
Trail) and
Interstate 64 in Rowan County, Kentucky, in the United States. It is the seat of its
county.
The population was 6,845 at the time of the 2010 U.S. census.
The correct option isU.S. 60and the produced alternative isU.S. 50. Of course the
context does not mentionU.S. 50, since it is not relevant for the question, but produces
a realistic alternative that (if the context is not given) could still be challenging to dis-
card.
Another option is given by:
What nationality was James Henry Miller’s wife?
James Henry Miller (25 January 1915 - 22 October 1989), better known by his stage name
Ewan
MacColl, was an English folk singer, songwriter, communist, labour activist, actor,
poet,
playwright and record producer. Margaret "Peggy" Seeger (born June 17, 1935) is an
American
folksinger. She is also well known in Britain, where she has lived for more than 30
years,
and was married to the singer and songwriter Ewan MacColl until his death in 1989.
Even if the correct answer,American, is quite trivial if the model is able to linkEwan
MacCollwith its true nameJames Henry Miller, the context could be quite deceiving.
Both the first passage regardingJames Henry Millerand the second regardingMargaret
"Peggy" Seegerquote the fact that he isEnglishand that she well known inBritainre-
spectively, thus the alternative is really well grounded in the context.
Due to this more reliable pre-processing stage ofPhi-3-mini, we chose to select the
first 1000 examples in thebridgesubset of HotpotQA’s training set. As forcomparison,
this was the only selection criterion used to pick problems on which performing tests.
The reason whyPhi-3-miniseems more reliable on this second split could be found
in the most various context provided by thebridgesubset.
Consider for example theNew York Cityexample: since the comparison consists in
spotting the common elements in both their passages, presumably they are already
select to discard off topic additional information. Inbridgequestions instead, context
is structurally composed by passages containing a part of information that is relevant
and must be merged with other sources and peddling ones. An example is given by the
U.S. highwayexample, in which is reported the population of Rowan County according
to 2010 U.S. census.
61

3.2 WikiHop
3.2.1 Dataset description
WikiHop is a part of a greater dataset called QAngaroo and proposed by Welbl et al.
[71].
QAngaroo is a Reading Comprehension dataset focusing onmulti-hopinference.
Several pieces of information often jointly imply another fact, thus a new fact is de-
rived by combining facts via a chain of multiple steps.
In each sample, a query is given about a collection of documents. The goal is to
identify the correct answer among a set of given type-consistent answer candidates3.
The candidates — including the correct answer — are mentioned in the documents.
An example of multi-hop question in WikiHop:
Figure 3.1: A sample from the WikiHop dataset where it is necessary to combine infor-
mation spread across multiple documents to infer the correct answer, taken from [71].
As previously mentioned, QAngaroo contains two distinct datasets:
• WikiHop: contains open-domain and based on Wikipedia articles; the goal is
to recover Wikidata information by hopping through documents. The example
above shows the relevant documents leading to the correct answer for the query
shown at the bottom.
• MedHop: based on research paper abstracts from PubMed, the queries are about
interactions between pairs of drugs. The correct answer has to be inferred by
combining information from a chain of reactions of drugs and proteins.
We chose to focus on WikiHop since MedHop covers a limited number of topics
and asks to output scientific acronyms of names or proitein labels:
Q: What interacts with DB00773?
Options: ["DB00072", "DB00294", "DB00338", "DB00341", "DB00588", "DB00820", "DB02546",
"DB02901", "DB04844"]
3Masked versions of these two datasets are also available, but for our analyses we will use not-
masked ones. Refer to [71] for further details.
62

The options (i.e. the possible answers to the question) are extracted from the source
documents, and only one among all the candidates is the correct one4.
An example taken from WikiHop is the following:
{"id": "WH _train _0",
"query": "participant of juan rossell",
"answer": "1996 summer olympics",
"candidates": ["1996 summer olympics", "olympic games", "sport"],
"supports": [
"The 2004 Summer Olympic Games, officially known as the Games of the XXVIII
Olympiad and commonly known as Athens 2004, was a premier international
multi-sport event held in Athens, Greece, from 13 to 29 August 2004 with the
motto \"Welcome Home.\" 10,625 athletes competed, some 600 more than
expected, accompanied by 5,501 team officials from 201 countries. There were
301 medal events in 28 different sports. Athens 2004 marked the first time
since the 1996 Summer Olympics that all countries with a National Olympic
Committee were in attendance. 2004 marked the return of the games to the
city where they began.",
...
]
}
Where:
•ididentifies the sample;
•queryspecifies the information that should be extracted from the texts;
•answeris the correct answer to the query;
•candidatesis a list of answer candidates, each of which is mentioned in one of the
supportspassages;
•supportsis a list of support documents.
3.2.2 Data processing
Compared to HotpotQA, here we face quite opposite problems. The candidate options
are provided and centered on the topics and the words present in the context, and each
example shows a various number of candidate options (not just two, as before). This
makes the dataset more challenging than the previous one.
Similarly as before, differentsupportsare merged together into an homogeneous
source.
The great difference is that thequeryis very essential and schematic, not appearing
like a concrete question but instead as a sketch of it.
Preliminary tests showed that models found difficult to read that style as a question,
thus a pre-processing stage is necessary. We used againPhi-3-miniin a one-shot setting
4In the current example, we have omitted the context since of much greater size of those of Hot-
potQA, you can check this by looking at the dataset [71].
63

to produce a question out of the schematic sketch of it. We used greedy decoding with
temperature set to 0 to generate the question. The maximum number of new tokens
allowed is 500, but never reached in practice. The prompt used to generate the question
is the following:
def create _message(question, options):
user _content = "Question: " + question + "\n Options: " + options + "\n\n
Assistant:"
messages = [
{"role": "system", "content": """}
You are a helpful AI assistant. You are asked to create a question out of a
sketched question.
Question: "occupation cao chong"
Options: [’academic’, ’builder’, ’chancellor’, ’classics’, ’confucian scholar’, ’
designer’,
’duke’, ’emperor’, ’engineer’, ’engineering’, ’father’, ’founder’, ’general’, ’
king’,
’leader’, ’major’, ’mathematician’, ’military’, ’official’, ’peasant’, ’physicist
’,
’physics’, ’politician’, ’prior’, ’rebel’, ’research’, ’ruler’, ’science’, ’script
’,
’social reformer’, ’socialist’, ’sovereign’, ’taiwan’]
Assistant: "Which was the occupation of Cao Chong?"
"""
},
{"role": "user", "content": "Now do the same for this question: " + user _content},
]
return messages
This task is performed perfectly by the model; we chose not to append the context
since it is enormously large and does not make sense in terms of question construction.
The one-shot example is chosen to include a significant number of options, allowing
the model to learn how to handle a large number of choices without becoming con-
fused.
The first 1000 queries of the WikiHop dataset were processed as above and stored
in a subset used for analysis purposes. No other selection criteria have been used to
produce the WikiHop partition for our analyses.
64

3.3 Datasets’ summary statistics
In the previous paragraphs we have already highlighted the fact that in HotpotQA the
context is structured by only twohopsand the options for each question are always
two.
In addition to this, the difficulty level of the question (i.e.easy,mediumandhard) is
neglected in this analysis, due to the fact that the same question could be more or less
challenging given a different alternative option.
Some summary statistics regarding the number of words in the tested datasets are
reported below. We assume that the number of words in thequerybut in particular in
thecontextis a proxy for the difficulty level of themulti-hopreasoning task. The idea
is that, even if the number of hops is fixed to two, the more verbose thecontextis, the
more information has to be merged before answering.
HotpotQA partition min mean std max
query
comparison 5 11.6 3.9 42
bridge 5 19.7 10.5 94
context
comparison 28 118.0 47.8 295
bridge 40 138.7 53.0 502
While thequerysizes (in terms of number of words) is almost the same for both the
HotpotQA partitions, thecontextones are drastically different.
Thequerysizes are slightly higher for thebridgesubset, probably caused by some
extremely long questions (we can tell from the great standard deviation and high max-
imum value, while the minimum and the mean one are the same or only slightly
higher).
Thecontextsize ofbridgeis slightly longer than thecomparisonone.
This is probably due to the same conclusions that we draw in section (3.1.2) regard-
ing the amount of additional information present insidebridge’scontextattribute.
On the opposite, WikiHop has a variable number of options and a variable number
ofhops.
Below is reported the distribution of the number ofhops5:
5The number of options are closely related to the number ofhops: since they are extracted from items
mentioned in the context, the longer the passage, the higher the number of options provided. And when
we have manyhops, the context is typically longer.
65

66

Chapter 4
Methods
Our work builds on other researchers’ findings and ideas, despite trying to explore a
new field of LLMs’ abilities.
In the literature review, we discussed how recent efforts in the NLP field focus on
finding various solutions to enhance model generation and prediction abilities. These
efforts often involve merging multiple methods or applying old ideas with new tech-
nologies.
We can imagine to construct a sharp distinction between the proposed approaches1:
• approaches that modify the model’s weights in order to obtain the desired output
(e.g. SFT, PEFT);
• context-based approaches, in which context is used to boost model performances
with relevant non-parametric, external knowledge (RAG, RE-RAG, NLI verifiers,
S2A);
• prompting approaches, that asks the model to perform a task in a certain way
provided in a couple of examples (in-context learning), to think carefully before
answering (CoT) and eventually to refine some previous answer given in input
(self-refinement approaches).
The first one improves the output thanks to internalized, downstream-specific knowl-
edge; the second one uses wisely the given context and relies on it to output a proper
answer; the third one just tries to exploit latent knowledge present in the pre-trained
model.
In literature, it is often used the termreasoningfor cases in which models exploit
some CoT-like procedure to come up with the correct answer, while the product of the
other two approaches is consideredtrainingorretrieval.
If we refer to the Oxford Dictionary, we find thatreasoningis defined as:the process
of thinking about things in a logical way; opinions and ideas that are based on logical think-
ingwhilethinkingis defined in a slightly different way:the process of thinking about
something.
1Even though an effective clustering operation would be not appropriate due to the inherently mix-
ing nature of modern machine learning technologies: consider that as only an aid for a clearer explana-
tion.
67

Thus,reasoningappears to be more logically-based and decision-driven, while
thinkingcan also be an unspoken, silent and internalized process.
This means that, given the black-box nature of LLMs, we cannot exclude that the
process of activating the parametric knowledge stored in the network’s weights isn’t
a form of thinking. We will not go deeper in arguing whether or not this process is
linked with a (spontaneous) will of the model, since this would be outside the discus-
sion’s topics.
On the opposite, thereasoningprocess involves a stream of intermediate, logical
connections and a final decision coming out from this process. This is why literature
often refers to reasoning abilities when talking about Chain-of-Thought approaches.
The great turning point with respect to other techniques is that, by explaining the log-
ical process leading to an answer, we can clearly see that the modelis reasoningabout
it.
Thus, the crucial component that defines a generation process as a reasoning pro-
cess is its explicitness.
For example, Rethinking with Retrieval (2.11.1) does not exactly reason about the
possible solution. It samples many possiblechains of justificationfor a given answer and
it discards those which are not supported by the retrieved content for the correspond-
ing question. This works as a sort ofpruningof all not context-grounded generated
chains. This approach helps improve the baseline answer (i.e., the standard prompting
answer given without context). However, when compared to an output obtained via a
self-consistency approach, the improvement is not as significant2.
Chain-of-Citations or Chain-of-Quotes approaches (2.11.1) still use the context but
through an opposite strategy.
Producing citations or quotes such as Li et al. [42] did helps in grounding each
step of the reasoning chain in a relevant text passage, i.e. does quite the opposite as
Rethinking with Retrieval. Binding to the definition ofreasoninggiven above, this is
nearest to it than (2.11.1): each passage is supported by a portion of the knowledge
base, and they are linked together in order to produce a plausiblechain of justifications
leading to the final (and hopefully correct) answer.
A careful selection (by the model itself) of the context on which the following an-
swer is built can be seen also as a form of asynchronous decision regarding the gener-
ation. Neglecting out-of-topic information or selecting just some relevant pieces of the
given passages is proved to boost significantly the performances [73]. Under a certain
light, carrying out a first passage to skim irrelevant components for the given question
(or prompt, more generally) is a form ofthinking about something in order to make a deci-
sion.
Considering this, it would be maybe excessive to consider RAG and RE-RAG con-
text selections as a form ofthinkingorreasoning, since often the comparison is made
using similarity scores or external ranker components (2.8.1).
2A 4% improvement on commonsense; 2% on temporal and 1% on tabular reasoning tasks [24].
68

A different conclusion has to be drawn for MIRAGE (2.11.1), in which the gen-
eration process iteratively spots a light on which element of the context triggers the
generation of the current token. While the prediction is exactly the same as the one we
would obtain by just appending the context to the question, thecontextual cluesspotted
can be used for interpretability analyses. This is useful to recognize some hints of an
underlying form ofreasoningthat the model performs.
What instead follows from this definition ofreasoningis that translating a QA prob-
lem in a NLI task (2.8.2) and consequently determining whether or not the answer is
entailed with the context is a form of (lateral) reasoning on the task.
The most familiar way (besides CoT approaches) that we recognize as reasoning is
the one that we attributed to self-refinement approaches.
Methods such asSELF-REFINE(2.10.1),SELF-CORRECTION(2.10.2) and Reflexion (2.10.3)
all starts from the assumption of generating a first tentative answer and iteratively
correcting it, until a certainlevel of satisfaction3is achieved. This appears very much
alike the human learning process: we come out with a first attempt, check whether or
not it is correct, eventually correct it.
The three approaches have different ideas on their basis.SELF-CORRECTIONsimply it-
eratively corrects the generated output using a specialized, pre-trained corrector mod-
ule.
BothSELF-REFINEand Reflexion are a three-components approaches, having in com-
mon the idea of structuring the pipeline as first attempt→feedback/evaluation→
refinement.
WhileSELF-REFINEproposes a "natural-language" refinement pipeline (i.e. the first
attempt and the suggestion are provided to the model by appending them to the
prompt), Reflexion updates model’s hyperparameters through a Reinforcement Learn-
ing approach (2.10.3).
4.1 Reasoning or simply imitating?
Many voices stand against the opinion that LLMs can reason or plan. In fact, in the
opinion of many researchers, we should imagine these models as very gooduniversal
approximate retrievers[35].
By that, they mean that the models are essentially just improvedn-grams, pre-
trained on a massive scale of web data and language corpora, hence truly capable
when asked to complete a sentence in the proper way. But they are asbolutely inca-
pable of performing any kind of reasoning that does not imply access to previously
memorised knowlegde.
The big difference stands in what we refer to when we talk aboutthinkingorrea-
soning. The previous paragraph assumes the Cambridge and Oxford definitions, while
Kambhampati introduces the System 1 and System 2 architecture [34] in decision mark-
ing to support the idea that System 2 is something absolutely unknown to LLMs.
3Different for each approach, typically bounded by a fixed number of iterative refinements or by a
stopping criterion on the quality of the generated output.
69

Kambhampati defines the LLMs’ parametric knowledgeas a giant non-veridical mem-
ories akin to an external System 1[35].
This is his proposed description:
Figure 4.1: On the left, System 1 and System 2 as proposed by Kahneman [34]; on the
right, the LLMs’ pseudo System 1 (taken from [35]).
The researcher and his team worked to test their strong assumptions against the
supposed reasoning abilities of LLMs, and found that the pre-training corpus has a
strong impact on the performances of those models in challenging reasoning tasks.
They presented to theGPT-4andGPT-3.5many different shift ciphers problems, vary-
ing the shift value from 1 to 25. What they observed is that the models were very
accurate on certain numbers of shifts, while completely out of clue in other cases:
Figure 4.2: Tests of shift ciphers for different shift values, from [36].
In fact, it is true that thedeliberativepart of the decision system is not (at least up to
now) a LLM feature. On the other hand, it is also true that in many situations it could
be enough to imitate the behaviour observe in massive sources of data.
70

From now on, when we refer to reasoning, we will intend the definitions in 4 or,
equivalently, to the pseudo System 1 as proposed by Kambhampati in [35].
71

4.2 A dialectic pipeline
Our method owes many ideas from previous works presented in the literature review
section (2).
We propose a three-step method composed of a thesis, an antithesis and a synthesis
of the two previous steps.
Figure 4.3: A visual representation of the proposed pipeline. Blue, red and purple
represent the thesis, the antithesis and the synthesis steps respectively. Colored arrows
highlight which element generated the subsequent item. The grey components are the
structural components of the task (i.e. question, context and possible answer options).
The answer is produced only after the synthesis step.
Our method does not perform alinearrefinement of the answer such asSELF-REFINE,
SELF-CORRECTION, Reflexion or Rethinking with Retrieval approaches.
Thelinearrefinement is performed when the first tentative answer (from now on,
thesis) is checked and eventually corrected by the same LLM in theantithesisstep.
But our idea is to further check the correctness of theantithesis’ suggestion by
prompting the same LLM again. This new process receives as inputs both thethesis
and theantithesisand performs a final pass on the problem before outputting the cor-
rect solution. We call this phase thesynthesisstage.
But why did we insert an extra step?
Differently from self-refinement methods summarized in (2), our method does not
require any form of task-specific prompting4or fine-tuning. This means that the correc-
tion (i.e. theantithesis) is simply obtained by one-shot prompting with a fixed prompt
structure describing the desired behaviour of that pipeline stage. Thus, an additional
check of the proposed correct answer could be beneficial.
In addition to this, considered that we have at least two options between which the
choice has to be made, we can face two scenarios:
4For example,SELF-REFINEprompts the model toShow step-by-step solutionfor math reasoning
tasks while toInclude keywords; maintain coherencefor constrained generation ones; we did not
change the instruction prompt with respect to the specificity of multi-hop subtask that we considered.
72

• thethesisand theantithesisagree on which option is the correct answer to the
question, thus thesynthesisreceives a single option, already motivated by the
antithesisstage;
• thethesisproposes an answer, theantithesissuggests why another option is more
proper (and why thethesis’ suggestion is wrong).
Thesynthesisconsiders the question, the options and the supporting context and
decrees which is the most correct alternative answer, once listened to thethesis’ and
antithesis’ opinions.
Discarding this last step could potentially harm the accuracy of the process. Con-
sider for example the case in which thethesiscorrectly predicts the answer, while the
antithesisstep loses some information5and outputs a detailed explanation on why the
correct answer is a wrong option.
By adding the synthesis step we force the model to compare different proposals and
hopefully reach the most proper answer among them. Note that we do not bound the
synthesis step to choose an option between the two proposed by the previous steps;
when more candidates are present, thesynthesisstep is left free to choose a third, un-
seen option given the context and the question.
Some experiments will be performed on the effectiveness of adding thesynthesis
step. We will report the effectiveness of the pipeline up to theantithesisstage and the
complete one.
We prefer to define our approach as adialecticpipeline in place of arefinementone
because of the following reasons:
1. we do not ask the steps to refine the output, instead we prompt theantithesis
and thesynthesisto check the previous steps’ outputs and return their opinion on
which is the most correct options;
2. the three steps of the pipeline are essentially autonomous one from the other: for
example, we can obtain anantithesisout of an hand-crafted or even completely
casualthesis, without running thethesispart before;
3. due to each component’s intrinsic autonomous nature, it is not guaranteed that
we are able to observe an improvement insynthesis’ answers with respect to the
thesis’ ones, since it is not arefinementprocess;
4. even though being autonomous actors in the pipeline, each actor adding an opin-
ion on the previous ones, the entire pipeline can be seen as a model dialoguing
with itself, in a sort of guided anddisentangled6Chain-of-Thought.
5Since we append to the context many items, some information may be lost in the attention process,
particularly in presence of very long contexts. This is observed by Liu et al. [43].
6Bydisentangledwe mean that the inner pipeline steps can be seen as intermediate ones in a bigger
reasoning chain; however, the steps’ contents arestrongly entangleddue to their dialectic nature.
73

4.3 Answering given the context
Although not strictly necessary, we chose to provide the relevant context to each step
of the pipeline.
This is due to the widely discussed reasons in the literature section (2) of the posi-
tive impact of appending the relevant context in QA tasks.
Our method aims at testing a pipeline that should be robust even when used with
smaller or less widely trained models to assess the level of reliability of the pipeline
with respect to different technical implementations. We tried different families of mod-
els and different sizes of the same model. Consequently, relying only on the paramet-
ric knowledge stored in the model’s parameters in order to provide the correct answer
(and completely neglecting the context) could favour significantly models of greater
size or trained on a bigger data mixture.
In all our experiments we will assume that the retrieval is performed perfectly: we
will directly append to the prompt thegoldcontext (i.e. flagged as relevant) in the
dataset.
We will not introduce unnecessary noise by adding deceiving elements in order to
test the robustness of the pipeline with respect to wrong context attribution. We plan
to run these tests on successive work on this topic.
This choice is made because in this analysis we aim at evaluating the impact of the
wayin which the context is presented to the model. We can face extremely long pas-
sages (e.g. the WikiHop dataset possesses many passages of more than 12.000 tokens)
in which the relevant information can be found in a couple of sentences only, or on the
opposite the passage could directly and explicitly mention the meaningful content in
just a couple of terms.
We make the choice of using models of (relatively) small context length, for example
Phi-3has 4K andGemma-2has 8K context length.
When we face a passage of greater size (e.g. as in WikiHop), we choose tosum-
marizeit in a smaller number of tokens before passing the triplet(question, options,
context)to the pipeline. We will prompt a model to make a summary of the given
context knowing the question that will be asked. Further details will be discussed in
(4.3.1).
In place of summarizing all the passages’ contents, another option could be con-
sidered. In scenarios like that, a filtering process could be helpful in speeding up the
inference phase and hopefully achieve better results, such as suggested in previous
works like System 2 Attention [73].
We preferred using a modified version of MIRAGE (2.11.1) to carefully observe the
context elements significant to the output’s generation andextractall the sentences
containing at least one of these elements. Hopefully, this model-internals based form
of filtering is more accurate than both a summarization and a selection based on simple
prompting as S2A (2.9). We will describe more details in (4.3.2).
Since these are all a sort of pre-processing actions that take place before running
74

the pipeline, we can process differently the dataset and observe how this modification
impacts on the pipeline. We will test the effectiveness of our pipeline:
• with the relevant (gold) context appended;
• with summarized context7;
• with filtered context, obtained using a modified version of [52].
Figure 4.4: Schematic representation of the different experiments (each one flagged by
a different color between yellow, orange and green) originated by different context pre-
processing stages. We will compare the answers produced by these different settings,
keeping the pipeline fixed.
4.3.1 WikiHop context summarization
As we mentioned above, we choose to summarize contexts that exceed the models’
context lengths. In our experiments, this bound is fixed to 4K tokens due toPhi-3
models.
While both the HotpotQA partitions have passages smaller than this value, Wiki-
Hop systematically exceeds the 4K limit.
Consequently, we need to summarize the WikiHop’s passages in order to keep them
under the given threshold. To perform a uniform compression, we summarize both
passages greater and smaller than 4K tokens.
We use still aPhi-3-minimodel, but this time we opt for its128kcontext length
version in order to be able to catch all the contexts, no matter their length. We prompt
the pipeline to:
You are a helpful AI assistant. You are given a long document as context and you are
asked to
produce a meaningful summary of it. Please summarize in less than 500 words.
7Only on WikiHop, since HotpotQA passages are all under the maximum context length (4K); on
WikiHop, we summarizedallthe passages, even those under the limit of 4K tokens, for the sake of
comparison.
75

Besides the requirement of not exceeding the 500 words, we let the model generate
up to 1000 new tokens to allow a certain degree of flexibility and not requiring to trun-
cate too harshly the summarization8. The other generation parameters are kept fixed,
with a temperature of 0.0 and no sampling involved.
We run the summarization process and we make sure that the model completes the
task (i.e. finishes to summarize the original passage), and we discovered that this had
happened only in certain cases.Phi-3family of models flags with<|end|>the end of
generation, and this is present only in 330 rows of the global 1000.
Instead of relying on a partially summarized source, we prefer to just consider these
330 completed summarizations and use this subset as a pre-processed context.
4.3.2 MIRAGE context filtering
On the opposite, context filtering is performed both on WikiHop and on HotpotQA,
since the extraction procedure is independent of the original context length. The goal
is to study whether by selecting only the relevant sentences of the passage we are able
to observe bettersynthesis’ outputs.
MIRAGE (2.11.1) selects all the documents containing at least onecontextual cueand
appends to the generated sentence a list of document identifiers (i.e. citations) from
which the generation has been influenced. We use a similar approach, but in place of
producing citations we aim at selecting only the relevant sentences inside the context
in order to construct a filtered version of it.
This selection will work as context source for our pipeline.
Still, we usedPhi-3-mini(this time we returned to the 4K version) and import it as
aninseqmodel9together with the model’s tokenizer:
inseq _model = inseq.load _model("microsoft/Phi-3-mini-4k-instruct", "saliency")
tokenizer = AutoTokenizer.from _pretrained("microsoft/Phi-3-mini-4k-instruct")
where"saliency"is the chosen attribution method.
The core function asks the model to:
• tokenize the text passage using the model tokenizer;
• invoke PECoRe [58], i.e. the methods running inside MIRAGE to paircontextual
cuesand their correspondingcontext-sensitivegenerated tokens;
• return only the sentences containing at least onecontextual cuein one merged new
passage.
Due to the different datasets’ passages sizes, we retain improper to ask PECoRe to
selectkpassages and to keepkfixed for all the datasets. Additionally, WikiHop shows
8A general rule of thumb is that 75 words approximately equals 100 tokens, thus for 500 words a
maximum of 1000 tokens should suffice.
9A Pytorch-based hackable toolkit to allow access to common post-hoc interpretability analyses of
sequence generation models [31].
76

a great variability of context sizes, thus a shared value ofkwould be too low for certain
problems and excessively for others.
Thus, we construct an auxiliary functionfind _top_p(passage, p)that takes the pas-
sage, tokenizes it and outputs the number of sentences corresponding to the top-p
percentile of the source.
Subsequently, we invoke PECoRe with theinvoke _pecore(passage, question, p)func-
tion as detailed in Appendix A and obtain the CCI scores (2.11.1).
We retain thephighest values among these, corresponding to the top-pmost influ-
ential tokens. These are mapped to their corresponding sentences and another func-
tion,select _passages(passage, question, p, tokens), returns theinfluentialsentences only.
Figure 4.5: Concrete example taken from HotpotQA (bridgepartition) of top-5% con-
text selection performed by our modified version of MIRAGE. The two sheep indicate
the MIRAGE process running PECoRe attribution method. The green passage is the
PECoRe-filtered process.
While for HotpotQA we can take each row (they are all under the threshold of 4K
tokens), for WikiHop we still have the same problem described in (4.3.1). Since the
object of our analysis is the comparison between the answers obtained with different
ways of considering the context, we do not care to apply this procedure to all the Wik-
iHop items.
What we did instead is to select only the WikiHop rows characterized by less than
3500 tokens10in thesupportsdataset attribute (i.e. theoriginal context). In this way, we
select only 92 of the original 1000 rows. We consequently summarize and filter this
subset only to make the three approaches comparable.
In order to study whether these results could be generalizable to the entire Wiki-
Hop dataset, we compare the performances of the summarized context in this subset
(original context under 4K tokens) and the ones observed in the 330 rows previously
summarized (4.3.1), that are assumed to be representative of WikiHop.
10500 tokens are left out for prompt instructions and few-shots examples.
77

4.4 Answers format: theguidanceframework
Even the best pre-trained and fine-tuned language model struggles in instruction fol-
lowing [49], [43], particularly open-source ones. Since in this analysis we do not make
use of proprietary models, we wanted to get rid thisformat disobedienceand force the
model to output exactly one of the candidate options, withoutgoing off on a tangent.
Given an extensive output pointing out the correct option, we want to recognize it
instead of neglecting everything that is not exactly the right-formatted answer. We do
not retain reasonable to judge harshly a correct output due to format issues.
Consider the following problem:
Question: Which year is Halley’s Comet expected to return to the solar system?
Options: [2110, 2045, 2086, 2061]
Context: Astronomers have now linked the comet’s appearances to observations dating
back more
than 2,000 years. Halley was last seen in Earth’s skies in 1986 and was met in space
by an
international fleet of spacecraft.
It performs a regular 76-year journey around the Sun.
The instruction-tuned model that we will use produce the following answers given
an appropriate prompt11:
•Phi-3-mini:
2061 <|end|>
•Phi-3-medium:
The correct answer is 2061. Halley’s Comet is expected to
return to the solar system in 2061, as it has a regular
76-year orbit around the Sun. <|end|>
•Gemma-2-2b-it:
**2061 **
•Gemma-2-9b-it:
2061
<end _of_turn><eos>
•Llama-3.1-8b-instruct:
2061.
11You are a helpful AI assistant. You are given a question and the relevant context
to answer it. Answer briefly to the question with just one of the given optionsand then
we appended the task.
We sampled the output with temperature fixed to 0.2 and maximum number of new tokens set to 50.
78

The outputted answers are all clearly correct, but their format is not always ap-
propriate to perform a quick comparison between the correct answer (2061) and the
model’s verbose output.
The models have different tokens flagging the end of generation (e.g.<|end|>and
<end _of_turn><eos>, that have to be discarded) and also the punctuation is a problem: it
is not trivial to distinguish when a dot flags the end of the sentence (and thus can be
removed) or when it is part of the answer (e.g.2061 A.C.). If we limit to post-process
the output by removing theend-of-generationtokens, onlyPhi-3-miniandGemma-2-9b-it
would (apparently) return the correct output, and this would be false.
To avoid relying on hand-constructed prompts (different from one model to an-
other, thus without guarantees to correctly extract the suggested option), we exploited
Structured Guided Generationtools.
Structured Guided Generation(SGG) is a feature that allows users to constrain the
generation of a large language model with a specifiedgrammar. It is used to generate
text that follows a specific structure or uses a specific set of words or produce output
in a specific format, e.g. to produce a valid JSON file as output, a function signature, a
list of integers.
Theguidanceframework does this by masking out certain all the tokens that do not
belong to the pre-specified grammar (that can also be a set of options):
1. the model produces the logitsfor each word in the vocabulary, e.g.
[0.1, 0.3, 0.2, 0.25, 0.15]
2. a mask is created for discarding all the words that donot belong to the grammar:
[-inf, 0.0, -inf, 0.0, -inf]
where-infis placed in correspondence of not-allowed words,0.0otherwise;
3. the mask is added to the original logits, allowing the model to discard all the
forbidden tokens in thesamplingstage:
[-inf, 0.3, -inf, 0.25, -inf]
There are also other tools for performing SGG, such asoutlines[76] that exploit
different technical implementations to determine how the model has to adapt its gen-
eration to a set of options. We opted forguidancedespite the fact that it does require to
access models’ logits, thus not being used for proprietary, closed-source models such
asOpenAI. We did not need a tool applicable to those kinds of models, thus we did not
find necessary to exploit a different guiding generation strategy.
Practically speaking, we ask the model to generate the first, tentative answer by
forcing the model to choose between the options as detailed in Appendix B.
79

4.5 Thesis
Focusing again on the pipeline, we choose to ask thethesisstep to output the most
proper candidate answer to the question.
We prefer not to let the model produce a verbose version which, as we spotted in the
previous section, can appear in many different and heterogeneous formats. This could
be confusing for later stages of the pipeline, which must handle multiple expressions
that semantically refer to the same option.
Think for example of a model that receives in input the answers presented in (4.4):
the long answer flagging2061as correct option could be handled differently than a
shorter answer saying the same thing12.
First of all, we upload the model and its tokenizer using theguidanceframework
and fix its temperature to 0. The prompting format is imposed by the framework and
concatenatesthemodel, theprompt(composed by some generic description on how
we want the task to be performed and by the problem) and theoptions(i.e. the only
tokens of the model’s vocabulary that will not be masked when generating the output).
The code used for thethesis’ generation can be found in Appendix C.
Figure 4.6: A schematic representation of theguidanceframework, when applied to the
thesis’ generation.
4.6 Antithesis
4.6.1 The importance of questioning previous statements
Theantithesisstep is the core component of the pipeline, but not because it is the most
relevant step13. Thesynthesisstep also plays a role in modifying the answer, but thean-
tithesis’ importance stands in the downstream impacts that a powerful correction has.
Think for example of how a teacher can correct an essay: he can only flag the error
or he can explain preciselywhythe student’s answer is not true.
In the first scenario, the student could try to make a second guess and he could also
fail again. In the second one, he will probably correct the first tentative answer with a
more robust and factual one that builds on the teacher’s suggestion.
12Maybe a more verbose opinion on which is the correct answer would be considered more reliable
than just the option name.
13as inSELF-REFINE,SELF-CORRECTIONand Reflexion
80

Our task is structured slightly differently from a teacher-student dialectic: since we
have the same LLM used as actor for all the three steps in the pipeline, we do not have
a teacher model that checks whether or not the answer is correct. No step is "more
experienced" than the others and consequently no step is more important or reliable
than the others.
We should then modify a little the previous metaphor by imagining a student who
is given a long multiple-choice test. He will firstly mark the choices that seem to him
more proper for the given question. We will refer to this first pass asthesis.
Then he will take a small break, refresh his mind, and look again to each question
and to each answer, asking himself: "Is it really the most proper answer to the question?
Is there a more proper option that I neglected on the first pass?".
He will consequently write some notes near each question explaining why he thinks
that a certain answer is the correct one. Note that he is not forced to explain why the
previously given answer is the correct one, he can also change its mind. Symmetrically,
he should enforce its claim on the fact that the previously given answer is the correct
one by spending a couple words on it. We will refer to this step asantithesis.
Finally, the student will look again at the question, the options, the firstly marked
one and the explanation of why he thought that it was correct or not. He will finally
(in thesynthesisstage) opt for the initial option or for a suggested alternative, in a fresh
pass of the same task.
Figure 4.7: A simple example of the three steps performed: in blue, thethesis; in red,
theantithesis; in purple, thesynthesis.
Of course, he will have no guarantees that each step is performed correctly. If the
second pass (the one in which he writes notes on which should be the correct option
and why) is performed incorrectly, the overall process will suffer: the (eventual) cri-
tique is fundamental to change opinion on the correct answer. If we limited to this
antithesisstep and we took its suggested output as the correct one, we would rely ex-
cessively on a single component that is prone to errors. By mitigating its effect as
described in (4.2), we make sure not to make the modeljumping to conclusions.
The goal of our experiments was to make theantithesisstep as more accurate as
possible. Of course we perform a final check, but we would like to avoid scenarios in
81

which thethesisis correct, theantithesisis incorrect and thesynthesisagrees with the
wrong suggestion.
This led us to the choice of providingaccess to the relevant contextto each step
of the pipeline. In this way, the student has always access to the book from which the
topics of the multiple-choice exam are taken, and he just has toreasonon the sources
and compare the options wisely.
Consider for example the Halley’s Comet example provided in (4.4): even by ac-
cessing the source, the model needs to:
• filter out irrelevant information;
• consider the last Halley’s Comet passage (1986);
• understand that it performs a regular journey, so the76-yearinformation should
be exploited;
• perform1986 + 76and output the correct answer.
The model capabilities have to go further than simply summarizing or rephrasing
the context for multi-hop tasks like the ones we are going to test. On the other hand,
if we do not provide the model the relevant information, the given answers could be
incorrect. Consequently, we consider the addition of the relevant passages a beneficial
step to elaborate a detailed description of which the correct option is during thean-
tithesisstage.
As we mentioned in (4.2), our method does not require any form of task-specific
prompt or fine-tuning. We just append an example describing how we want the pro-
cess to be performed in a one-shot setting. We chose not to increase the number of ex-
emplars provided since the prompt is already burdened by the question, the options,
the (often long) context and thethesis’ answer. Additionally, the one-shot prompt was
enough to produce the desired behaviour, thus increasing it would be pointless.
The code used to generate the antithesis’ suggestion can be found in Appendix
??. Differently fromthesis, theantithesisis not forced to be exactly one of the options
through the use of theguidanceframework. What we did instead was to define a gen-
eration pipeline:
pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer)
generation _args = {"max _new_tokens": 500, "return _full _text": False, "do _sample":
False}
Consequently, the generated output will be in a discursive format. Consider for
example this task, whereCandidate answercontains thethesis’ answer:
Question: Which magazine was started first, Arthur’s Magazine or First for Women?
Options: [Arthur’s Magazine, First for Women]
Candidate answer: First for Women
82

Context: Arthur’s Magazine (1844-1846) was an American literary periodical published
in
Philadelphia in the 19th century. Edited by T.S. Arthur, it featured work by Edgar A.
Poe, J.H. Ingraham, Sarah Josepha Hale, Thomas G. Spear, and others. In May 1846 it
was
merged into Godey’s Lady’s Book. First for Women is a woman’s magazine published
by Bauer Media Group in the USA. The magazine was started in 1989.
It is based in Englewood Cliffs, New Jersey. In 2011 the circulation of the magazine
was 1,310,696 copies.
Theantithesiswill produce14the following opinion:
The correct answer is ’Arthur’s Magazine’ as it was started first in 1844, while ’
First
for Women’ was started later in 1989.
As it is easy to spot, theantithesissuggests the correct option by taking a second
look to the context, even though thethesisfailed.
4.6.2 The influence of the given examples
In the previous section we passed the one-shot example as an input parameter of
create _message _antithesis(). We now want to spot a light on theimpact that the one-
shot examplehas on the generatedantitheses.
We experimented with different settings, each one stressing different behaviours
that theantithesiscould replicate.
• Thebaseline: we simply provide the prompt with an example of how we want
the model to make use of thethesis’ answer (referred to ascandidate answerinside
the prompt):
Question: What is the sun, a star or a planet?
Options: [’a star’, ’a planet’]
Candidate answer: a planet
Context: The Sun is the star at the center of the Solar System. It is a massive,
nearly perfect sphere of hot plasma, heated to incandescence by nuclear
fusion reactions in its core, radiating the energy from its surface
mainly
as visible light and infrared radiation with 10% at ultraviolet energies
.
Assistant: The correct answer should be ’a star’ due to the fact that the context
explicitly say so. On the opposite, the context never mentions the
fact
that the Sun could be a planet.
Two things have to be spotted: first of all, the model does not firmly state which is
the correct answer. We induce the model to choose a neutralshould bein place of a
more confident expression. The second thing is that the answer does not mention
14Real output, obtained usingPhi-3-mini.
83

explicitly the context, but instead it limits to motivate a certain choice by stating
that this is due to the context. This baseline makes an implicit reference to the
effective passage, does not quote it directly.
• Theisvariant: this one-shot example differs from the previous one just in the
degree of confidence used to answer: the previousshould beis replaced with a
more confidentis:
Question: What is the sun, a star or a planet?
[...]
Assistant: The correct answer is ’a star’ due to the fact that the context
explicitly say so. On the opposite, the context never mentions the
fact
that the Sun could be a planet.
This aims at experimenting whether the level of confidence with which the model
provides theantithesisinfluences thesynthesis’ decision on which option should
be the correct one.
• Thecotvariant: uses a Chain-of-Thought inspired answer:
Question: What is the sun, a star or a planet?
[...]
Assistant: Let’s consider the options and check whether or not they are correct.
The context clearly identifies the Sun as ’the star at the center of
the Solar System’, thus ’a star’ is probably the correct option.
On the opposite, ’a planet’ is not mentioned in the context, thus it
is
unlikely to be the correct option. Therefore, the correct option is ’a
star’.
This approach considers each option and seeks eventual support of their correct-
ness in the context. In the example, the relevant part of the context is alsoquoted.
Finally, a conclusion is reached on which is the most proper answer among the
candidates.
• Thectxvariant: in the wake of the previous experiment, we tested also a different
approach that is more context-grounded:
Question: Jane’s Addiction and Weeping Willows, play which genre of music?
Options: [’indie’, ’rock’]
Candidate answer: rock
Context: Weeping Willows is a Swedish indie rock group that started in 1995. Jane
’s
Addiction is an American rock band from Los Angeles, formed in 1985.
The band consists of Perry Farrell (vocals), Dave Navarro (guitar),
Stephen
Perkins (drums) and Chris Chaney (bass).
Assistant: The context mentions that Weeping Willows is a ’Swedish indie rock
group’ and
84

Jane’s Addiction is an ’American rock band’. Both bands are associated
with the
’rock’ genre, thus the correct answer is ’rock’.
This is still a CoT-like approach to the problem, since the model decomposes
the task in intermediate steps; but differently from thecotvariant, this time the
assistant is shown a different way of dealing with the problem.
The model quotes the context’s elements (thus thectxterm to identify this vari-
ant) that are relevant for each compared item, and consequently decrees which is
the correct answer. This is different from before, where partial conclusions were
drawn analyzing the correctness of each option separately. With this variant we
wanted to test whether explicitly quoting the relevant context before answering
is beneficial to the pipeline.
The one-shot example provided is different from the previous one for two main
reasons:
• to assess the impact that different prompt examples have on the generated out-
puts (thus ensure that the results are not prompt-dependent);
• because the first example is relatively easy: the context does not mention both
the candidate answers, but just the correct one. This could potentially cause the
model tolearn an incorrect pattern15, since in the multi-hop tasks we face the con-
text systematically mentions the options.
The task that will be tested on this pipeline ismulti-hopquestion answering, thus
each option will be grounded in some elements of the context and the implicit chal-
lenge of the model is to actually merge multiple elements in order to provide a reason-
able answer.
Figure 4.8: A brief overview of theantithesisvariations tested in our analysis.
Here we show an example of how the antithesis output changes with respect to the
king of one-shot example that is provided:
• baseline:
The correct answer should be ’Arthur’s Magazine’ as it was started first in 1844,
while
’First for Women’ was started later in 1989.
15We used the termpatternreferring to a similar use of the word present in the Chain-of-Thought
prompting paper [68], stating thatlarge language models can learn via a few examples with natural language
data about the task (c.f. automatically learning thepatternsunderlying inputs and outputs via a large training
dataset).
85

•isvariant:
The correct answer is ’Arthur’s Magazine’ as it was started first in 1844, while
’First
for Women’ was started later in 1989.
•cotvariant:
Let’s analyze the options and the context provided. The context states that
Arthur’s
Magazine was published from 1844 to 1846, while First for Women was started in
1989.
Therefore, Arthur’s Magazine was started first. The candidate answer ’First for
Women’
is incorrect. The correct option is ’Arthur’s Magazine’.
•ctxvariant:
The context states that Arthur’s Magazine was published from 1844 to 1846, while
First
for Women was started in 1989. Therefore, Arthur’s Magazine was started first.
The correct answer is ’Arthur’s Magazine’.
As it is easy to notice, there is no significant difference betweencotandctx, since
both the one-shot examples decompose the problem into a retrieval step and a con-
clusion step. The most remarkable difference (present also in different problems, not
reported here) is that thecotvariant tends to remark the fact that the other option is
wrong.
One last device we had was to change the style of the question in the one-shot
example for datasets other than thecomparisonfraction of HotpotQA. This is because,
while all thecomparisonquestions are formulated in such a way as to explicitly report
the options to choose from, thebridgequestions (i.e. the complementary fraction of
HotpotQA) and WikiHop reflect a less "set" style:
• HotpotQA,comparison:
Which magazine was started first, Arthur’s Magazine or First for Women?
• HotpotQA,bridge:
The Oberoi family is part of a hotel company that has a head office in what city?
• WikiHop:
What language did John Osteen speak or write?
Consequently, we will replace
What is the sun, a star or a planet?
with a simpler
86

What is the sun?
.
On the opposite,ctxis already more similar to the simpler question, thus is left
unchanged.
4.7 Synthesis
The final step aims at merging in a reasoned answer the entire pipeline: by consider-
ing the first tentative answer (thethesis) and the new opinion (theantithesis) obtained
by reconsidering the previous one, the last opinion (thesynthesis) has to decree which
is the correct option. And now it should do this with more confidence and increased
factuality.
As in the previous pipeline steps, the detailed code is reported in Appendix E. It
is important to remark that thesynthesis’ prompt does not bound the model to choose
between the option proposed by thethesisor proposed by theantithesis: the instruction
given clearly states that thesynthesisstage has to choose between one of theavailable
options. This means that thesynthesisstage could potentially even opt for an unseen
option.
For the baseline and theisandcotvariants, the one-shot prompt is very similar:
Question: What is the sun, a star or a planet?/What is the sun?
Options: [’a star’, ’a planet’]
Candidate answer: a planet
Suggestion: {antithesis _answer}
Context: The Sun is the star at the center of the Solar System. It is a massive,
nearly
perfect sphere of hot plasma, heated to incandescence by nuclear fusion
reactions
in its core, radiating the energy from its surface mainly as visible light
and
infrared radiation with 10% at ultraviolet energies.
Assistant: the correct option is ’a star’, since the suggestion is grounded in the
context,
even if the candidate answer does not agree.
whereantithesis _answeris the antithesis, presented in different formats with respect
to the kind of variant (i.e. what follows theAssistant:in the previous section’s one-shot
examples).
For thectxvariant, we tried instead to use two-shot prompting in order to induce
the model to learn how to deal with situations in which thethesisand theantithesis
agree and situations in which they don’t16:
16Thectxvariation originates from the observed results and the fact that the in-context learning ex-
87

Question: Jane’s Addiction and Weeping Willows, play which genre of music?
Options: [’indie’, ’rock’]
Candidate answer: rock
Suggestion: The context mentions that Weeping Willows is a ’Swedish indie rock group’
and
Jane’s Addiction is an ’American rock band’. Both bands are associated
with the
’rock’ genre, thus the correct answer is ’rock’.
Context: Weeping Willows is a Swedish indie rock group that started in 1995. Jane’s
Addiction is an American rock band from Los Angeles, formed in 1985.
The band consists of Perry Farrell (vocals), Dave Navarro (guitar), Stephen
Perkins (drums) and Chris Chaney (bass).
Assistant: Both the candidate answer and the suggestion agree on the fact that the
correct
option is ’rock’. Let’s check on the context whether or not this is correct
.
Weeping Willows is an indie rock group, thus they make rock music; Jane’s
Addiction is a rock band. Consequently the context confirms that the genre
performed by both bands is ’rock’. The correct option is ’rock’.
Question: Between two tennis players Kim Clijsters and Mary Pierce, who is older?
Options: [’Kim Clijsters’, ’Mary Pierce’]
Candidate answer: Kim Clijsters
Suggestion: The correct answer is ’Mary Pierce’ as she was born on 15 January 1975,
which is earlier than Kim Clijsters who was born on 8 June 1983.
Context: Kim Antonie Lode Clijsters (] ; born 8 June 1983) is a Belgian
former professional tennis player. Clijsters is a former world No. 1 in both
singles and doubles. Mary Pierce (born 15 January 1975) is a French retired
tennis
professional who played on the Women’s Tennis Association (WTA) tour.
Born in Canada, she is a citizen of Canada, and the United States.
Pierce played for France in team competitions and in the Olympics.
Assistant: The candidate answer says that the older tennis player is Kim Clijsters,
while
the suggestion indicates Mary Pierce. The context provides the birth dates
of
both players, thus I can check which of the two options is correct.
Kim Clijsters was born on 8 June 1983, and Mary Pierce was born on 15
January
1975; By comparing these dates, it’s clear that Mary Pierce
is older than Kim Clijsters. Thus the correct option is ’Mary Pierce’.
The important notice that has to be made is that we have no guarantees that the
synthesisstep chooses the right option. It could be deceived by a wrong suggestion
of one of the two previous steps or it could choose another wrong option. We have
already spent a couple of words explaining why a last checking step could be influ-
ample is different is a conscious choice. Additionally, the following synthesis setting (different from the
cotone) is oriented to a study on different few-shot options.
88

ential in terms of performances of the pipeline, but our experiments will confirm this
hypothesis.
89

4.8 Assessing performances
Up to now, we had widely described each step’s details and variations, but still it is
unclear how a verbose synthesis output (i.e. not just the correct option, but the ex-
planation of why a certain option is correct) can be translated into the corresponding
option.
In (4.4) we showed how the Large Language Models employed in this analysis are
usuallyformat disobedientwhen asked just to output the correct answer to the question.
Consequently, we forced the generation of exactly one of the options using theguidance
framework, so that just the chosen option is passed to later stages of the pipeline.
In theantithesispart, what we wanted instead is that the model is given the oppor-
tunity to persuade thesynthesisthat thethesis’ tentative answer is not correct.
In these terms, it would be rough to just provide to thesynthesistwo different op-
tions and let it choose which is the most appropriate without any additional infor-
mation. In fact, in a scenario like this, it would be enough to prompt the large lan-
guage model to reason about different candidates before outputting the answer. Con-
sequently, the computation required to obtain thethesis’ and theantithesis’ answers
would be senseless.
If thesynthesisreceives instead a comment (theantithesis) on the first tentative an-
swer (thethesis), suggesting which should be the correct option, it takes as input a new
vision on the problem that could lead to more grounded answers. This is what is done
in the multiple-choice test example that we made in Figure 4.7.
Additionally, we allow thesynthesisstep to question theantithesissuggestion, since
we give to it all the required elements to answer autonomously to the question. Thus,
thesynthesis’ output will not be forced to be aguidanceoutput, instead it will be few-
shot learned to be an explanation on which is the correct answer and why, as reported
in the multiple examples of the previous section (4.7).
In order to practically determine which is the suggested option without the need
to "read" the complete justification, we added a post-processing extraction procedure
such that the content is properly extracted and reduced to one of the candidate an-
swers. To assess the impact of each step of the pipeline, we choose to extract also the
antithesis’ suggested answer, not just the final (i.e. thesynthesis’) one.
90

Figure 4.9: How the extraction process is performed for theantithesisand thesynthesis
stages. While thethesisstep simply constrains the generation, for these two steps we
have to extract the chosen option.
91

We choose to automatize this process usingPhi-3-minifor all the generated out-
puts17. We use the following prompt template:
You are a multiple-choice question answering assistant.
Choose the most proper option between {options} that best matches with the suggestion.
Question: {question}
Suggestion: {antithesis}/{synthesis}
Sources: {context}
Assistant:
that is subsequently augmented with the proper items forming anaugmented _prompt.
Finally, theguidanceframework is employed to be sure that the answer comes in the
right format:
def optionExtraction(question, options, suggestion, sources):
# augmented _prompt is produced
answer = guidance _model + augmented _prompt + select(options)
return answer
Kojima et al. [39] used a similar post-processing stage for their zero-shot Chain-of-
Thought: although they did not use a structured guided generation tool asguidance,
they performed a post-processing phase aimed at extracting the correct answer from a
longer one.
We make sure that this post-processing approach faithfully outputs the suggested
option and not the correct one. By providing the model with thequestion, theoptions
and thecontextwe could think that this final stage could modify the pipeline’s answer.
Consider as an example the following task:
Who was born first, Pablo Trapero or Aleksander Ford?
thethesisproposed the correct answer (i.e.Aleksander Ford), while theantithesisaf-
firmed that:
The correct answer should be ’Pablo Trapero’ as he was born on 4 October 1971,
while Aleksander Ford was born on 24 November 1908.
the proposed post-processing technique extractsPablo Traperoas the antithesis’ pro-
posed answer, even if incorrect and in contrast with both thethesis’ and thesynthesis’
choices.
4.9 Comparison with Chain-of-Thought prompting
We want to compare the answers provided by our method with respect to Chain-of-
Thought prompting, as it is well-renowned to perform remarkably well on reasoning
tasks. Chain-of-Thought is prompted a single time to reason about a task in a way
that resembles the few-shot example provided. On the opposite, our pipeline extends
17In order not to introduce biases dependent on each model’s different abilities to follow instructions;
we observed empirically (running a couple of tests) that, whilePhi-3-miniis always reliable and accu-
rate, theGemmafamily often disattends the requirements and outputs random options instead.
92

this approach by prompting the same model multiple times, passing to each step the
output of the previous ones (when present).
We try to make the comparisonas fair as possible, i.e. we provide the same few-shot
examples and we extract the CoT answer also with theguidanceframework (thus no
format issues should be present).
In detail, the one-shot example is the same given in theantithesis’cotvariant (4.6.2),
except for a couple of details: the lack of thethesis’ first tentative answer and the cor-
responding instruction in the general prompt (i.e.You are asked to ...). The Chain-of-
Thought prompt is the following:
You are an helpful AI assistant. You are asked to determine the most correct answer
for
a given question provided a set of possible options. Your goal is to decree which is
the most correct answer to the question between the available options.
Here’s an example of how to do it:
Question: What is the sun, a star or a planet?
Options: [’a star’, ’a planet’]
Context: The Sun is the star at the center of the Solar System. It is a massive,
nearly perfect sphere of hot plasma, heated to incandescence by nuclear
fusion reactions in its core, radiating the energy from its surface mainly
as visible light and infrared radiation with 10% at ultraviolet energies.
Assistant: Let’s consider the options and check whether or not they are correct.
The context clearly identifies the Sun as ’the star at the center of the
Solar System’, thus ’a star’ is probably the correct option.
On the opposite, ’a planet’ is not mentioned in the context,
thus it is unlikely to be the correct option. Therefore, the correct
option is ’a star’.
Now do the same for the following question:
Question: {question}
Options: {options}
Context: {context}
Assistant:
Thus thecotvariant and the Chain-of-Thought approach are really similar.
The big difference stands in where the effective answer extraction takes place: while
in the Chain-of-Though approach is directly extracted from the previous prompt’s out-
put, the pipeline requires an extra step to be performed (thesynthesis) before decreeting
the most correct option.
93

Figure 4.10: Comparison between the answer generation using the Chain-of-Thought
approach (left) andcotvariant of the pipeline (right).
94

Chapter 5
Results
In this final stage, we will report our experiments’ outputs that try to answer the fol-
lowing research questions:
• Does the pipeline have a beneficial effect in improving the answers’ correctness?
• How do the different families of models perform under the same pipeline set-
tings?
• Is the pipeline robust with respect to different datasets?
• Which is the impact of pipeline variations (4.6.2)?
• Is the proposed method more effective than Chain-of-Thought prompting?
• Do pre-processing sources as detailed in section (4.3) have positive impacts on
downstream pipeline performances? Which are the differences that can be ob-
served between them?
5.1 Does the dialectic pipeline work?
Before asking any different question, thecore, preliminary answer that is necessary to
conduct further studies is whether or not the dialectic pipeline works. Byworkswe
mean that we are able to observe a significant improvement with respect to the stan-
dard prompting output.
The two generated outputs that we are going to compare in this first step are the
thesis’ answers and thesynthesis’ ones. Note that, for how we defined thethesisstep (C),
this first tentative answer is in fact the final output of a standard prompting procedure.
Thethesisconsiders the question, the options and the supporting context and produces
the most proper answer given these elements. Theguidanceframework ensures that
the output possesses a clear format: no issues regarding failure to recognize a correct
answer but in the wrong format will arise.
We choose to append the relevant context to this step (although not strictly neces-
sary, since thethesisis in fact a baseline) in order to make a fair comparison between a
context-enrichedpipeline and a standard model generation, that should reasonably also
be provided with the relevant passages.
95

We are going to consider thecomparisonpartition of the HotpotQA dataset and
we will exploit thebaselinepipeline setting (4.6.2), runningmicrosoft/Phi-3-mini-4k-
instruct(2.3.2) as model.
Results show that, while thethesis’ guesses the correct option in the 53.4% of the
cases, the (baseline) pipeline reach a 80.7% of correct guesses. The 27.3% improvement
is sufficiently significant to justify the claim that the pipeline works (even in the base-
line setting).
5.2 How models’ architectures and number of parame-
ters impact on pipeline performances
The second question of our analysis is whether this improvement is consistent when
we test other models thanmicrosoft/Phi-3-mini-4k-instruct.
We additionally trymicrosoft/Phi-3-medium-4k-instruct(similar architecture thanPhi
-mini, but has 14B parameters instead of 4B),google/gemma-2-2b-it,google/gemma-2-9b-it
(2.3.1) and
meta-llama/Meta-Llama-3.1-8B-Instruct(2.3.3).
Model Phi-mini Phi-medium Gemma-2B Gemma-9B LlaMa-8B
Thesis’ correct answers (%) 53.4 50.0 52.8 59.7 48.3
Synthesis’ correct answers (%) 80.7 89.5 81.8 88.1 87.2
Absolute improvement (%) 27.3 39.5 29.0 28.4 38.9
Focusing onthesis’ performances, it is easy to spot thatPhi-miniandGemma-9Bare
the best overall. While it is reasonable thatGemma-9BoutperformsGemma-2B(since an in-
creased number of parameters often corresponds to more capable models [9]), what is
truly surprising is thatPhi-minishows better performances thanPhi-medium(having 14B
parameters, 10B more parameters than theminiversion!). This is in fact something that
thePhi-3research group observed:
some benchmarks improve much less from 7B to 14B than they do from 3.8B to 7B, perhaps
indicating that our data mixture needs further work to be in the “data optimal regime” for 14B
parameters model.
ThePhi-3family of models is pre-trained on "textbook data" [1] that allows to re-
duce the training corpus and break thescaling laws[37]. But if the data mixture is not
optimal (i.e. do not possess a certain level of data quality required to reduce the corpus
size) then the performances would be poorer than expected.
It is also necessary to point out thatGemma-2Bis not far fromPhi-mini, although in
practice it has half its parameters: 2B and 3.8B respectively, with a performance gap of
just 0.6%, corresponding to 2 wrongly predicted answers only1.
Another surprising result is thatLlaMa-8Bis the worst performing model overall if
we limit to prompt the model in the standard way.
1Thecomparisonpartition of the dataset consists in 352 questions.
96

Switching now to thesynthesis’ results, the best overall pipelines are obtained us-
ingPhi-mediumandGemma-9Bmodels, although evenLlaMa-8Breaches a similar result.
We can observe that models performing worse than others in the thesis step are able
to largely bridge the initial gap through the pipeline, reaching the best overall perfor-
mances (this is the case of bothPhi-mediumandLlaMa-8B).Phi-mediumbeats by a small
margin (1.5%, corresponding to 5 differently guessed answers).
The smaller models reach lower accuracy values than their correspondent greater
options, even not by a large margin. Additionally,Gemma-2Bbeats by 1%Phi-mini, de-
spite being almost half the size of the second one.
With these experiments, we can safely assess that, at least on HotpotQAcomparison,
the pipeline works for different families of models and improves by a large margin the
baseline.Phi-miniobtains the worst performance increase overall, and despite this it is
able to score a really good improvement.
5.3 Robustness with respect to different datasets
In the previous section we clearly expressed how our experiments made sure the pipeline’s
effectiveness on the HotpotQAcomparisonsubset.
Now we want to assess whether it is true that this statement could be generalized to
othermulti-hopdatasets. We test the same pipeline setting also on thebridgepartition
of the same dataset and on the WikiHop dataset. In this section we will be considering
the WikiHop version with summarized contexts, i.e. the 330 questions mentioned in
(4.3.1).
Dataset % correct Phi-mini Phi-medium Gemma-2B Gemma-9B LlaMa-8B
HotpotQA Thesis: 53.4 50.0 52.8 59.7 48.3
comparison Synthesis: 80.7 89.5 81.8 88.1 87.2
HotpotQA Thesis: 52.1 56.0 55.7 66.5 49.8
bridge Synthesis: 87.9 90.2 81.3 88.9 91.9
WikiHopThesis: 12.7 13.6 17.7 16.9 12.4
Synthesis: 33.0 40.7 28.3 21.1 37.7
These results confirm the trend that we observed for thecomparisonpartition of Hot-
potQA.
If we consider thebridgepartition of the same dataset, we can recognize similar
thesisandsynthesisaccuracy values, although they tend to be better than those of
comparison.Phi-miniandLlaMa-8Bshow nearly identicalthesispercentages of correct
answer in bothcomparisonandbridgepartitions, whilePhi-medium,Gemma-2BandGemma-9
Bperform better by a certain margin in this second subset. We have to consider that
while we just considered 352 questions fromcomparison, thebridgepartition contains
1000 items. Thus, these differences2could be originated:
• the different dataset sizes, since a more various and wide datasets could provide
a more reliable value of models’ predictive abilities than a smaller one;
26%, 2.9% and 6.8% inPhi-medium,Gemma-2BandGemma-9Brespectively
97

• by the automatic generation of the alternative option forbridge, as described in
(3.1.2) that could make its tasks easier to solve than hand-checked ones, in par-
ticular for more capable models. This would explain the greater gap that we
observe forPhi-mediumandGemma-9B.
What we can infer from these results is that greater size models (i.e.Phi-medium,Gemma
-9BandLlaMa-8B) perform remarkably well, as we observed forcomparisonalready. The
interesting difference is that this timePhi-minioutperforms by a large marginGemma-2B:
that (apparently small) 6.6% gap corresponds to 66 wrongly answered questions from
the latter model. This leads us to the idea thatPhi-miniis more capable thanGemma-2Bat
merging multiple sources to determine which is the correct solution. When we ask in-
stead to look at two different sources and compare them, the 2B model performs better
than the other one. Maybe this difference is due to the limited abilities of a model of
that small size, and forbridgetype of tasks it may be preferable to use a greater model.
The WikiHop dataset confirms the pipeline effectiveness, despite the reduced accu-
racy values that can be observed both for thethesisand for thesynthesispercentage of
correct answers. In this case, the number of candidate options is not fixed to two and
neither is the number ofhops. Instead, WikiHop presents a number of options up to
60, even though limited to a few questions, and a set of passages with a big variety in
terms of number ofhops(3.3).
Due to this significant difference, the already tested models behave differently than
before. WhilePhi-mini,Phi-mediumandLlaMa-8Bimprove thethesisby large margins
(each of these improves the baseline more than twice), theGemma-2family of models
seem to struggle with this various set of tasks.
Surprisingly, the 2B version overcomes the 9B one by a solid 7.2%.
Below is reported the table of absolute improvements in accuracy observed be-
tween thethesisand thesynthesissteps of the pipeline. Results are reported with re-
spect to the considered dataset (comparisonandbridgeomit the HotpotQA prefix) and
the model used inside the pipeline.
Dataset Phi-mini Phi-medium Gemma-2B Gemma-9B LlaMa-8B
comparison 27.3 39.5 29.0 28.4 38.9
bridge 35.8 34.2 25.6 22.4 42.1
WikiHop 20.3 27.1 10.6 4.2 25.3
Overall mean 27.8 33.6 21.7 18.3 35.4
The first conclusion that we can draw is that for all the cases we considered, our
proposed pipeline always causes positive improvements in accuracies; we can con-
sider the mean on all the dataset as a synthetic measure supporting this statement.
Despite its being effective, this level of positive impacts is model and dataset de-
pendent. WikiHop is proved to be more challenging due to its multi-hop variety and
the big number of options between which the models have to choose the most proper
answer.
Despite its being really challenging, the pipeline often reaches large margins of
improvements (more than 20%) for two families of models,Phi-3andLlaMa-3.1. The
98

Gemma-2family clearly struggles in absorbing meaningful information from the pipeline,
while being the most capable family when considered thethesisperformances only.
This could be due to the parametric knowledge ofGemma-2models, maybe richer
and more various than the "texbook" knowledge ofPhi-3and of theLlaMa-3.1one.
Considered this, we will prefer to analyze thesynthesis’ accuracy values as indi-
cators of the pipeline effectiveness rather than focusing on the relative improvement
observed with respect to thethesis’ ability of finding the correct question at the first try.
The minor tendency ofGemma-2to follow the instructions given (i.e. use context
to answer) is easily observed when we inspect the responses produced by this in the
antithesisphase, compared with those of the other two models:
Figure 5.1: Comparison between theantithesis’ answers as given by these three families
of models; thethesis’ proposals are colored in blue. Note that the format should follow
the one-shot example provided in (4.6.2), thusGemma-2Bturns out to be poorer than the
other models in faithful instruction following.
It would be untrue to state thatGemma-2models’ poor ability to replicate faithfully
the given few-shot examples (and being more "conversational" rather than context-
grounded) is harmful in all the cases. In HotpotQA, where the options are just two,
this issue could not be a real problem. In fact, the absolute improvements observed for
Gemma-2models on HotpotQA are large (although smaller than other models’ margins).
The necessity to attain to instructions is instead crucial in more complex task.
This further hints that the real improvement given by the pipeline is not given by its
multi-step nature, rather by thereasoned solving process. By breaking down the problem
into units performing multiple checks from different perspectives, even small models
(i.e. with less than 20B parameters) can achieve good performances.
99

5.4 Pipeline variations
In the previous section (5.3) we highlighted how theGemma-2tendency to approach
theantithesisstep differently from what it has been shown from the one-shot example
could explain worse prediction abilities (at least on challenging tasks).
The following experiments are made to assess how much the way in which we
prompt the model has an impact on the final performances. In practice, we substitute
the one-shot example provided and study how much this modification changes the
correctness of the final predicted output.
We start from thecomparisonpartition of HotpotQA and test the differences between
the baseline pipeline settings and theis, thecotand thectxvariants (4.6.2). This com-
parison is performed for all the models that we already considered.
Figure 5.2: Different percentages of correct answers obtained using a particular
pipeline settingon HotpotQA -comparison. The plot y-axis is reduced (starts from 70%)
to highlight the small differences between the different one-shot examples.
The first comment that we can make is that there is a negligible difference (under
the 1% for all the models) between the baseline and theisvariation. We can safely
affirm that the level of confidence with which theantithesisis provided is not relevant
in terms of performance. Thus, theantithesispersuades the pipeline through factual
knowledge, not by a more confident style of suggestions.
Thecotvariant seems to outperform the baseline when the model with sufficiently
big models, i.e. comparison betweenPhi-medium,Gemma-9Band betweenLlaMa-8B. This
improvement lies in the reasoning abilities observed in models with a greater number
of parameters3. On the other hand, alsoPhi-minibenefices from this "reasoned" style of
answer, whileGemma-2Bis penalized with respect to the baseline. A possible explanation
3This is also confirmed by the Chain-of-Thought paper by Wei et al. [68], where reasoning is de-
scribed asan emergent ability of more capable(i.e. greater size)models.
100

of this discrepancy stands in the pre-training procedure ofPhi-mini: an entire step of it
is dedicated to teaching the model how to logically reason and to attain specific skills
(2.3.2). Consequently,Phi-minismall size does not influence the quality of the gener-
ated answer. This is obviously not the case ofGemma-2B, that possesses a diminished
capability with respect to the 9B version and is not pre-trained for reasoning purposes.
Thectxvariant differs from the previous one in two main points:
1. theantithesisis induced to quote all the relevant context before choosing the cor-
rect option, thus allows the model to split the content extraction and the effective
decision process (this is an implicit process, that the model has to infer from the
one-shot example that is provided);
2. thesynthesisstage shows two examples of dealing with thethesis’ and theantithe-
sis’ opinions on which is the correct answer (in the first one, the two agree; in the
second one, they don’t), thus instruct explicitly thesynthesison how to deal with
these two scenarios.
This variation appears to work well (i.e. better than the baseline) on small mod-
els such asPhi-miniandGemma-2B; this is probably due to the fact that these kinds of
LLMs benefit from the division between the context selection and the effective deci-
sion. Additionally, instructing theirsynthesisphase on how to deal with agreement
and disagreements could also have a positive effect.
Greater models (Phi-medium, of 14B parameters, andGemma-9B) still behave better with
a more "logically structured" approach as proposed by thecotvariant: whilectxis still a
quite valid option (forGemma-9Bit worsens the baseline performances, but by a reduced
margin of only 0.9%) it seems toburdenthese pipelines in an unnecessary manner.
LlaMa-8Bshows comparable performances ofcotandctxand is in fact a middle ground
between greater and smaller models in terms of number of parameters.
Despite some small differences widely discussed above, these first experiments
confirm that our method allows to observe consistent improvements with different
pipeline settings and across different models employed, not being dramatically biased
by specific prompts.
Switching now to thebridgepartition, we choose not to run theisvariant because
of its nearly perfect correspondence to the baseline.
101

Figure 5.3: Different percentages of correct answers obtained using a particular
pipeline settingon HotpotQA -bridge. The plot y-axis is reduced (starts from 70%)
to highlight the small differences between the different one-shot examples.
On the other hand, we considered again both thecotandctxvariants. In this par-
tition, thecotoption seems slightly inappropriate: this is probably due to the different
task that is required to perform. The baseline version of the pipeline seems to perform
better than thecotvariant in some cases.
Whilecomparisonrequired to compare two sources containing information about
two different items and to reason on this content,bridgerequires to merge correctly
multiple sources, but once they are properly merged they are of easier understanding.
The reasoning abilities in this form of tasks is probably less important rather than
focusing on the proper information inside the context and properly merge them. This
second step is what thectxvariant does, and in fact it can be observed that in this sub-
set this solution is the best one for all the tested models (forGemma-2Bthe two variants
achieve equal performances).
Finally, we want to test whether a different pipeline setting could help in improving
the performances on the challenging WikiHop tasks.
102

Figure 5.4: Percentages of correct answers obtained using a particular pipeline setting
on WikiHop. Thethesis’ correct answers are represented by the blue bars, while the
red ones represent the different pipeline variations considered.
For WikiHop, we can comment each family of models separately. ThePhi-3models
show the baseline as the best pipeline option, followed by thecotand finally by the
ctxvariations.Gemma-2models show similar performances of the baseline pipeline and
thectxvariation, whilecotappears worse.LlaMa-8Bis similar to the previous family,
although reaching higher performances for all the pipeline settings.
No pipeline variation is able to consistently outperform the baseline version of it on
multi-hop, multiple choice tasks like those of WikiHop. The reason could be find in the
fact that bothcotandctxask the model to consider the context and reason on it, such
as depicted in (4.6.2). On the opposite, the baseline version of the pipeline just asks the
model to decree the correct answer, thus it can skip unnecessary context selection and
go directly to the most proper answer.
In the following example, it is clear that while the baseline option can directly an-
swer, the two other options are burdened by the prompt requirements (via the one-shot
example). Since WikiHop is really complex and often presents a variety of options,
many of which are semantically related (e.g.United Kingdomandbritish), the additional
effort that the models have to face is to answer to the prompt’s exact instructions.
In the following example, the question clearly ask theparent taxonofAustralosuchus
, thus it is not properly true (ascotstates) that’animal’and’crocodile’are correct
options. The model withcotantithesisis able to identify the fact that’crocodilia’is the
correct option, but gets lost in its reasoning process and in the final stage is not able to
clearly state that.
On the opposite,ctxdirectly forgets to include’crocodilia’as one of the options,
probably confusing and merging it with’crocodile’.
103

Figure 5.5: Baseline,cotandctxantithesesobtained withPhi-mini.
Again,Gemma-2family exhibits poor abilities in absorbing the few-shot prompting
implicit patterns. This allows its models to choose an original andunscheduledway of
dealing with the problem, i.e. splitting the reasoning process in multiple steps in order
to explain why a certain option is the correct one, while briefly explaining why the
others are not. This approach is able to reach the correct option in some cases, while
it leads the model to strongly support a wrong claim due to an approximate context
usage (Figure 5.1).
LlaMa-8Ballows thectxvariant to properly analyze each option, beating thecotone
which tends to get struck on motivating why its option is better than the other, rather
than consider each alternative independently and finally reach a conclusion.
Summing up, in tasks requiring a careful and concordant answer to a precise ques-
tion in which the model has to deal with multiple options and long,multi-hopcontexts,
we suggest to let the pipeline do its work without forcing it to explain in a Chain-of-
Thought manner why it does so. Additionally, we find that for these type of tasks,
indulging in evaluating each relevant part of the context could be harmful4.
4We can imagine the metaphor of an overthinking model: if it already knows the correct answer, it
couldget lostin trying to motivate everything besides that option.
104

5.5Overthinkingcan be harmful (even for LLMs)
In the previous sections (4.2, 4.7) we question the effectiveness of thesynthesisstep,
that could appear redundant once theantithesisgives its feedback.
To assess whether we need thesynthesis, we extracted the options suggested by the
antithesisstep in order to observe what changes between these two steps. We compare
the answers given in these two checkpoints.
In the following table we show the mean improvement across all datasets caused
by theantithesisstep with respect to thethesis(first rows) and the residual variation
obtained by adding thesynthesisstep (second rows):
Setting Step Phi-mini Phi-medium Gemma-2B Gemma-9B LlaMa-8B
baselineT-A 26.67 32.57 23.17 18.83 35.40
A-S 1.13 1.03 2.50 0.57 0.57
cotT-A 27.63 31.30 19.03 17.90 32.97
A-S 0.23 1.63 0.43 0.43 2.63
ctxT-A 27.97 32.17 26.27 21.43 38.43
A-S 0.30 0.03 -2.30 -2.57 -2.13
Table 5.1: Mean accuracy gaps (across HotpotQA partitions and WikiHop) observed
between thethesisand theantithesissteps (T-A) and between theantithesisand the
synthesisones (A-S). In bold, we highlighted the negative ones.
The results above are averaged across all datasets; the dataset-specific ones can be
found in Appendix F. Despite each dataset’s characteristics that we have previously
described (5.3) and the model’s tendency to behave differently to the same task (5.2),
from results in Appendix F it appears quite clear that the pipeline variations condition
on thesynthesis’ ability to improve theantithesis’ one.
Besides the -1.8% of theGemma-2Bcase on thebridgepartition of HotpotQA, thesyn-
thesisstage turns out to be always beneficial if the baseline version of the pipeline is
used.
Some performances drop between theantithesisand thesynthesisstages are ob-
served when used thecotvariant, and this is reasonable due to the fact that the Chain-
of-Thought approach to the answer already explores whether each option is the correct
one or not. Adding a further step to this already exhaustive computation has the effect
of bias towards a wrong answer some correct reasoning chains (even by small mar-
gins). This is due to the limited number ofhops(2) that characterizes HotpotQA:cot
antithesisis already too powerful for easy problems, and thesynthesiscould only "take
the model off the road".
The prevalent loss of accuracy between theantithesisand thesynthesishappens for
HotpotQA partitions and the usage of thectxvariation. The reasons could be similar
than the ones already proposed for thecotvariant. Additionally, we highlight the fact
that thectxvariant is the only one with two examples in the prompt; this suggests that
a stronger guidance on how to deal with disagreement betweenthesisandantithesisis
not beneficial. It could be interesting to modify this choice and study thectxoutcome
with only a one-shot example, as incot. This gap appears linked with theGemma-2and
LlaMa-3.1models employed.
105

We conclude that the pipeline works, but tends to be negatively affected if asked
tothink too muchabout a problem that is too simple. Thus, in front of relatively easy
problems (small and already selected contexts, 1-hoppassages, trivial questions given
the context) we recommend to stop thecotorctxpipeline variations at theantithesis
stage and retrieve the proposed solution. Since identifying easy problems for the LLM
can be a challenging task per se, we leave the use of an automatic detection component
for such settings to future work.
Although by a small margin, thesynthesisimproves the prediction accuracy in more
challenging scenarios or when we use the baseline version of the pipeline (also sug-
gested for user-friendly coding, because the one-shot example is simpler to create).
5.6 Chain-of-Thought prompting comparison
We previously described our method as aguided and disentangled Chain-of-Thought(4.2),
because we ask multiple times the same model to reason on which is the most correct
answer to a question, providing to each step the opinion of the previous ones. Thus, we
want to compare this method with its most natural competitor: the Chain-of-Thought
prompting.
From now on, when we refer tocotwe mean the pipeline variation, while CoT
stands for the one-shot prompt as already described in (4.9). In that section we high-
light the fact that the one-shot example provided incotand in CoT is nearly the same5.
The question now is whether our proposed method is able to outperform the accu-
racy obtained through Chain-of-Thought prompting.
In the following plots we compare the thesis, the CoT and the bestsynthesisaccuracy
values obtained for different models6on HotpotQA. By "the bestsynthesisaccuracy
values" we mean that we consider the pipeline configuration scoring higher accuracy
values for that model and for that dataset. They could be different with respect to the
considered model, e.g. the baseline pipeline forPhi-miniand thectxvariant forLlaMa-8
B. We allow this to be mixed since not want to test whether a certain pipeline setting
outperforms CoT; we just aim at checking whether the pipeline is able to beat CoT in
any configuration.
5Except from the fact that the first one also contains thethesis’ chosen option, while the second does
not.
6Despite what we observed about theantitheseshigher accuracy values thansynthesesones for simple
tasks, for the sake of consistency we will use the second ones for evaluating the tasks from now on.
106

Figure 5.6:HotpotQAcomparisonpartition: in red, the Chain-of-Thought answer accu-
racy values; in purple, the bestsynthesisones.
Figure 5.7:HotpotQAbridgepartition: in red, the Chain-of-Thought answer accuracy
values; in purple, the bestsynthesisones.
On one hand, the differences between the two partitions of HotpotQA are really
small and almost always7are in favour of the proposed pipeline. On the other hand,
we have to consider that these improvements are obtained with two extra steps, while
Chain-of-Thought is able to reach similar values in a single step.
7Only forGemma-2Bfor thebridgepartition CoT is slightly better; but the 84.5% reported for the
pipeline is thectx’s synthesis, the correspondingantithesiswould beat CoT by a 2.8%.
107

Figure 5.8:WikiHop: in red, the Chain-of-Thought answer accuracy values; in purple,
the bestsynthesisones.
On WikiHop the gap between the two solutions is much higher for some models:
a 11.6% onPhi-mini, a 10.6% onPhi-mediumand 15.6% onGemma-2B. On the opposite,
Gemma-9Bbaased pipeline outperforms CoT only by 3.8% andLlaMa-8Bonly by 1.9% (for
HotpotQA this margin was greater).
The reasons of these gaps stand in the fact that bothPhi-3andGemma-2models re-
ceive a significant benefit from the pipeline, in particular when they face a challenging
task like those of WikiHop. The tendency of theGemma-2models of being more conver-
sational rather than context-grounded (5.1) can be easily spotted from the low perfor-
mances that these models exhibit when CoT-prompted. By asking them to consider
again thethesis’ proposed option, we are able to obtain large improvements.
LlaMa-8Bshows instead smaller accuracy differences between the two approaches.
This is also due to a performance drop betweenLlaMa’santithesisandsynthesisstages:
if we early exit the pipeline, we would observe greater margins.
Additionally, in the previous section (5.5) we observed thatGemma-2andLlaMa-3.1
family of models tend to favour a moreunstructedprompting structure; when guided
in thesynthesisstage by providing them two examples, they perform worse than the
one-shot scenario. Thus maybe the pipeline (at least for these family of models) is dis-
advantaged by the burden of having an extra instruction to follow.
Summing up, the pipeline is able to reach higher performances than CoT in almost
all the scenarios studied. Certainly, CoT is a more computationally-lightweight than
the proposed pipeline, so it is not extremely surprising that it works worse.
However, this is still an interesting finding that highlights howwe have not yet
reached the boundary of the gains that can be derived from the reasoning skills of
LLMs.
An interesting frontier of this work consists in studying whether distilling the correctly-
executed dialectic dialogues (thesis-antithesis-synthesis) in a new, smaller model still
outperforms CoT. If positive results are observed, then we could consider to exploit
this pipeline as a pre-training/fine-tuning component, rather than an inference strat-
108

egy. Training models onthinking tokenshas already been proved as effective by works
as the one of Zelikman et al. [81].
5.7 Context filtering and summarization
Lastly, we want to considerhow muchthe relevant passage is provided influences the
generated output. The objects of our experiments are consequently the contexts of
HotpotQA and WikiHop datasets, that would be manipulated and given as inputs to
the pipeline. Their outputs are compared with those produced by using the original
context in order to check whether there are some improvements.
WikiHop, that has already been summarized in order to reduce the context length
(and stay inside the 4K limit), is considered in these experiments only as a subsection
of it. We discard all the original contexts exceeding the threshold8and we summarize
(4.3.1) and filter (4.3.2) the passages in this subset only.
Due to what we observe onGemma-2models (5.1), we run the following experiments
on thePhi-3andLlaMa-3.1model families only. We retain that this would be sufficient to
assess whether there is an effective improvement in the pipeline’s prediction accuracy.
Dataset partition Pipeline Passage Phi-mini Phi-medium LlaMa-8B
comparisonthesisoriginal 53.4 50.0 48.3
filtered 54.9 46.4 48.2
baselineoriginal 80.7 89.5 87.2
filtered 67.0 83.0 78.5
cotoriginal 87.2 92.1 90.6
filtered 58.2 74.2 79.7
ctxoriginal 91.2 91.2 90.9
filtered 57.0 70.3 75.5
bridgethesisoriginal 52.1 56.0 49.8
filtered 9.7 10.1 7.4
baselineoriginal 87.9 90.2 91.9
filtered 96.1 94.7 93.8
cotoriginal 90.2 91.1 90.7
filtered 91.0 94.5 93.6
ctxoriginal 91.9 90.7 92.0
filtered 90.3 91.0 92.6
Table 5.2: Context filtering results using (4.3.2) onHotpotQA. In bold we highlighted
the experiments that outperform the non-filtered context ones.
This table contains interesting results. Despite being partitions of the same dataset,
comparisonbehaves very differently frombridge.
While the filtered version ofbridgescores terrible accuracy values in the thesis step,
the other partition slightly improves the percentage of correctly guessed options. We
8Practically, we use 3.5K as threshold in place of the true 4K one in order to allow the instructions to
fit in.
109

think that this happens due to the two tasks’ different nature.
Instead,comparisonasks to compare two independent sources sharing some com-
mon content that has to be extracted, thus filtering the original context could help in
focusing on the relevant parts of both these sub-passages already in thethesisstage.
On the opposite, when the output of the filtering process is used in the pipeline, the
performances worsen with respect to the original one. We can also notice that greater
models (i.e.Phi-mediumandLlaMa-8B) lose less in accuracy terms than smaller ones (i.e.
Phi-mini). This is caused by an aggressive selection that the filtering stage performs in
some cases. While helpful in neglecting irrelevant information, sometimes it also lefts
out some important details useful to correctly answer the question, as showed in the
example in Figure 5.9.
110

Figure 5.9:Phi-minifiltering process (4.3.2) on a problem of HotpotQA -comparison.
In this case, our proposed filtering process discards completely the sentence that
talks about
Meadowbank Gold Mine, causing theantithesisstep9to correctly question that:
The context provided does not contain any information about the value of Temagami-
Lorrain
Mine or Meadowbank Gold Mine. Therefore, it is impossible to determine which is
currently more valuable based on the given context.
A similar problem also affects the other partition of the dataset, but this time in the
opposite direction.bridgequestions require the model to merge multiple sources in
order to understand how information present in different passages is related. It is now
easy to imagine which is our filtering approach’s enormous fatal flaw: if the linking
part is left out, then the passage does not make sense anymore. Consider for example:
Figure 5.10:Phi-minifiltering process (4.3.2) on a problem of HotpotQA -bridge.
Understandably, thethesisdoes not recognize any James Henry Miller’s wife from
the given context. This is the reason why the accuracy scores of the first tentative
answer are terribly low in this partition.
Theantithesisstep is able to recognize the fact that the additional information given
in the context should be exploited in some way, and makes a guess on why that infor-
mation is provided:
9Output obtained withPhi-mini.
111

The correct answer should be ’American’ because the context states that Margaret "
Peggy"
Seeger, who was James Henry Miller’s wife, is an American folksinger.
No part of the given context says so, but the model knows that it has to search for a
wifein the passage, thus it is reasonable that it isMargaret "Peggy" Seeger. And she is
American.
Instead of worsening the output quality, getting rid of additional sentences not rele-
vant for the context seems to give additional robustness to the pipeline’s performances.
The smaller gaps are again observed forctx, probably for the same reasons we dis-
cussed in (5.5).
In section (4.3.2) we specified that in order to compare WikiHop’s original passages
and eventually filtered ones, we would have to restrict the dataset to questions relative
to passages with less than 4K tokens10. We also already mentioned that this subset
is made of 92 questions only. We exploit the fact that we had run some tests (5.3) on
the summarized versions of WikiHop (4.3.1) to extract the passages corresponding to
those subset of questions.
We start again from the original context and filter it with the approach described
in section (4.3.2) and also used on HotpotQA. In this way, we have at disposal three
different versions of the same passage. We feed these alternatives to the pipeline and
compare their results.
This time we are going just to consider the baseline pipeline setting due to what
we have observed in (5.4). Some experiments carried out to observe other pipeline
settings’ behaviour on WikiHop confirmed that the baseline one is the best option.
10Practically speaking the threshold is set to 3.5K.
112

Figure 5.11: Percentage of answers guessed correctly using the baseline version of the
pipeline on WikiHop. The different context versions are colored: in yellow, the origi-
nal context; in orange, the summarized version; in green, the PECoRe filtered version
(4.3.2).
The interesting fact that we can assess is that the three models behave quite differ-
ently except for the terrible summarization performances, shared by all of them.
For thePhi-3family of models, the original context and the filtered one perform
comparably or the second one performs better; to us, this is due to the implicit abilities
of LLMs to link the question and thedecontextualizedinformation in the context, as
done in (5.7).
The accuracy gap between the original context and the filtered one inLlaMa-8Bis
likely due to its reduced capability to make implicit connections, such as those men-
tioned above.
Consider for example the question
In which language were the original works that discuss ’fires were started’ written?
associated with the following filtered context:
The Blitz, from the German word "Blitzkrieg" meaning ’lightning war’, was the name
used
by the British press to describe the heavy air raids carried out over Britain in 1940
and
1941, during the Second World War. Frank Humphrey Sinkler Jennings (19 August 1907
24 September 1950) was an English documentary filmmaker and one of the founders of the
Mass Observation organisation. Jennings was described by film critic and director
Lindsay Anderson in 1954 as: "the only real poet that British cinema has yet produced.
The film uses actual firemen (including Cyril Demarne) rather than professional actors
.
And here we report thePhi-mediumandLlaMa-8Banswers:
•Phi-medium:
113

The correct answer should be ’english’ due to the context mentioning that Frank
Humphrey Sinkler Jennings was an English documentary filmmaker and the film uses
actual firemen rather than professional actors. The context does not provide any
information about the original works discussing ’fires were started’ being
written in French, German, or Latin.
•LlaMa-8B:
The correct answer should be ’latin’ due to the fact that the context does not
provide any information about the language in which the original works that
discuss
’fires were started’ were written. However, the context does mention the Blitz
and
Frank Jennings, but it does not provide any information about the language of the
original works. However, considering the context, it is more likely that the
correct
answer is not among the options provided. The context is about the Blitz
and a documentary filmmaker, but it does not provide any information about
the language of the original works that discuss ’fires were started’. But if I
had to
choose from the options provided, I would say that the correct answer is ’latin’
because
it is a more general and ancient language that could be related to the topic of ’
fires
were started’, but it is not a definitive answer.
Despite the fact that the context is particularlycrypticeven to humans, we can
clearly see thatPhi-mediumis able to linkfirestofiremenand consequently toFrank
Jennings, who speaks english. On the opposite,LlaMa-8Bdoes not recognize this link
and also assumes the wrong option,latin, which is evidently not the correct one also
because of the fact that the’fires were started’is expressed in english words.
We find that summarizing the context is not the best possible solution to deal with
long contexts with non-relevant information included in them. When we prefer not to
provide the original context, we can filter the sources using a method such as the one
we proposed in section (4.3.2). This is also proved to improve overall performances
when paired with models possessing good reasoning abilities such as thePhi-3ones.
However, we suggest to use this form of pre-processing only when necessary. In
cases such as HotpotQA, in which the context is already selected to be relevant, ne-
glecting some parts of it could lead to a drop in the generation accuracy (such as in
comparison). Additionally, we suggest also to prefer the baseline pipeline setting, since
both thecotandctxvariants are built to deal with not-selected context (and thus they
tend to focus on the relevant parts before answering, but this time the first operation
has been already done).
114

Chapter 6
Conclusions and future directions
The results exposed in the final section highlight the effectiveness of adopting a dialec-
tic approach to face challenging multiple-choice question answering tasks.
An established way of dealing with this task is the Chain-of-Thought approach
(4.9), which proposes to prompt a single model to produce a multi-step reasoning
chain. This is supposed to let the model focus more on each sub-task present on the
problem and to motivate its answer by producing natural language explanations.
Another way of facing the same problem is given by the self-refinement approaches
(2.10.1, 2.10.2), that instead decompose the problem into multiple, iterative refinements
made by the same large language model or by a specialized corrector.
Our method is a third option between these two, since it consists in dialectic setting
involving three actors calledthesis,antithesisandsynthesis(4.2) that are asked to incre-
mentally improve the previous steps’ outputs. The latter steps are also asked to reason
before answering, making them CoT-units and allowing us to refer to our method as a
sort ofguided and disentangled Chain-of-Thought(4.2).
However, we retained improper to define this pipeline as a Chain-of-Thought vari-
ation. Even though the pipeline exploits a single model, it involves three actors that
check the answers’ correctness, each one with a different role. In this sense we found
reasonable not to classify this method neither as a Chain-of-Thought approach, nor as
a self-refinement one.
Reflexion (2.10.3) was also tested by its creators in its CoT-augmented version on
one of the two datasets that we considered (3.1.1) employingGPT-4; their results never
reach the 80% of correct answers despite the high number of trials employed [59]. Our
method was instead able to beat this threshold even with a 4B model (i.e.Phi-mini).
Experiments performed in section (5.6) showed how the proposed pipeline was also
able to consistently outperform the Chain-of-Thought alternative.
Our contributions are the following:
• We proposed a dialectic pipeline that is found effective for improving results in
multi-hopquestion answering tasks (Figure 5.8). The margins of accuracy im-
provements are large and superior to the ones observed when Chain-of-Thought
prompting is exploited to solve the problem.
• We assessed the robustness of this method with respect to different families of
models and different datasets, proposing differentmulti-hopsub-problems and
115

different number ofhops. We found that greater models are generally more capa-
ble of dealing effectively withmulti-hopsources. Additionally, a greater number
ofhops(e.g. in WikiHop) is correlated with worse model performances overall;
however, this is also the case in which our proposed method scores the biggest
improvements, both with respect to the baseline (5.3) and to the CoT-only ap-
proach (Figure 5.8).
• We tried different pipeline configurations (4.6.2) showing the model different
ways of handling the context. We found that this helps in tasks with a limited
number ofhops, while it tends to confuse the model when the content to analyze
is wide and complex; for this latter scenario, the baseline version is found to be
the most effective one.
• The pipeline works best for models which are really careful in following instruc-
tions. Even though we considered all instruction-tuned models, we found that
Gemma-2models tend to be creative when not requested, disobeying the few-shot
example and performing the task with a different approach (5.1). Since this turns
out to be poorer in performance terms, we can confirm that the improvement that
the pipeline causes is not only due to additional computation (i.e. the fact that
more steps are performed) but also to what is done in these steps.
• More capable models benefit less by the addition of thesynthesisstep (Table 5.1)
and thectxpipeline variation (4.6.2) causes a loss of accuracy when it has to face
easy tasks (Appendix F). Since other settings do not cause this phenomenon, our
opinion is thatoverthinkingharms the process, thus an early stopping should be
considered.
• When the context is long and noisy we can filter it to achieve greater perfor-
mances (5.7). We exploited a modification of MIRAGE (4.3.2) and compared it
with the results obtained with the summarized and the original context. What
we found is that filtering enhances the predictions of models with good deduc-
tive abilities (5.7), while summarization is the worst option among the tested
ones.
This work is a first step in a wider set of experiments that we would like to test. Up
to now, we have only considered open-source and relatively small (all under 20B) mod-
els; however, the prevalence of methods summarized in the literature review part (2)
are tested onGPT-like decoder-only models. Although we proved the consistence of our
pipeline across multiple families of open-source models (5.2), it would be interesting
to run some tests to ensure that our method is widely applicable to proprietary LLMs
too. Additionally, experimenting with the dialectic pipeline using various agents in
the roles of thesis, antithesis, and synthesis to introduce diversity in skills presents an
interesting avenue for future work.
We would like also to assess whether the pipeline works for other datasets. In
our analysis we focused onmulti-hopdatasets, since they merge the two tasks of rel-
evant content extraction and reasoning. We judged important to propose a method
that is able to face this two tasks jointly instead of separately. However, studying the
pipeline’s effectiveness on these two tasks separately (e.g. datasets like GSM8K com-
monly allow to study only mathematical reasoning, while there is a wide literature of
116

RAG-only datasets from which we can choose one to run relevant context extraction
only) could still be a significant contribution to the research community.
Switching now the focus on possible pipeline inner improvements, we already ob-
served that the number ofshotsprovided to the prompt could be influential to the
synthesis’ improvements. Consequently, we could test both the zero-shot version of the
pipeline and whether decreasing to one the number ofctxsynthesis’ examples could
achieve better results.
In general, more effort could be made in trying to reduce the negative gap between
theantithesisand thesynthesissteps of the pipeline; a preliminary analysis of the ob-
served "in-between change of mind" should lead pipeline modifications and hopefully
improvements.
Regarding context filtering process, we ran all these tests by keeping fixed the per-
centage of relevant tokens (p = 5%) that the method has to select. It could be interesting
to benchmark multiple proportions and observe whether there is a correlation between
the datasets’hops complexity(i.e. the number ofhopsin the passages) and the degree of
selection (i.e. the value ofp) applied to its context.
The comparison between the pipeline’s performances given the original, the sum-
marized and the filtered context as inputs is performed only for passages up to 4K
tokens. This is simply due to our choice of models; they have a limited context win-
dow that bounds their processing capacity. A separate study could be performed with
the same models possessing a higher context length (e.g.Phi-3has also the128kmod-
els’ versions) in order to study WikiHop more broadly, in order to draw more general
conclusions on how the context conditions the pipeline’s output.
A simple, final test that should be performed is the pipeline’s behaviour when the
model is not given. We ran some tests (not showed in this work) that confirm the
pipeline’s improvement with respect to thethesisstep even when asked to rely only on
the models’ parametric knowledge. Expanding those experiments could allow us to
leverage the pipeline both for factuality and reasoning tasks.
Lastly, it would be interesting to test whether a distillation of correctly-executed
pipelines [60] could be a significant form of pre-training for LLMs.
117

Bibliography
[1] Marah Abdin et al.Phi-3 Technical Report: A Highly Capable Language Model Locally
on Your Phone. 2024. arXiv:2404.14219 [cs.CL].URL:https://arxiv.org/abs/
2404.14219.
[2] Abhimanyu Dubey et al.The Llama 3 Herd of Models. 2024. arXiv:2407 . 21783
[cs.AI].URL:https://arxiv.org/abs/2407.21783.
[3] G Apollinari et al. “High-Luminosity Large Hadron Collider (HL-LHC) : Prelim-
inary Design Report”. In: (2015).DOI:10.5170/CERN-2015-005.
[4] Dzmitry Bahdanau, Kyunghyun Cho, and Yoshua Bengio.Neural Machine Trans-
lation by Jointly Learning to Align and Translate. 2016. arXiv:1409.0473 [cs.CL].
URL:https://arxiv.org/abs/1409.0473.
[5] Dzmitry Bahdanau, Kyunghyun Cho, and Yoshua Bengio.Neural Machine Trans-
lation by Jointly Learning to Align and Translate. 2016. arXiv:1409.0473 [cs.CL].
URL:https://arxiv.org/abs/1409.0473.
[6] Angels Balaguer et al.RAG vs Fine-tuning: Pipelines, Tradeoffs, and a Case Study on
Agriculture. 2024. arXiv:2401.08406 [cs.CL].URL:https://arxiv.org/abs/2401.
08406.
[7] Iz Beltagy, Matthew E. Peters, and Arman Cohan.Longformer: The Long-Document
Transformer. 2020. arXiv:2004.05150 [cs.CL].URL:https://arxiv.org/abs/2004.
05150.
[8] Tom B. Brown et al.Language Models are Few-Shot Learners. 2020. arXiv:2005 .
14165 [cs.CL].URL:https://arxiv.org/abs/2005.14165.
[9] Tom B. Brown et al.Language Models are Few-Shot Learners. 2020. arXiv:2005 .
14165 [cs.CL].URL:https://arxiv.org/abs/2005.14165.
[10] Manish Chablani.Sequence to sequence model: Introduction and concepts. 2017.URL:
https://towardsdatascience.com/sequence-to-sequence-model-introduction-
and-concepts-44d9b41cd42d.
[11] Jifan Chen, Eunsol Choi, and Greg Durrett.Can NLI Models Verify QA Systems’
Predictions?2021. arXiv:2104.08731 [cs.CL].URL:https://arxiv.org/abs/2104.
08731.
[12] Karl Cobbe et al.Training Verifiers to Solve Math Word Problems. 2021. arXiv:2110.
14168 [cs.LG].URL:https://arxiv.org/abs/2110.14168.
[13] Dorottya Demszky, Kelvin Guu, and Percy Liang.Transforming Question Answer-
ing Datasets Into Natural Language Inference Datasets. 2018. arXiv:1809.02922 [cs.CL].
URL:https://arxiv.org/abs/1809.02922.
118

[14] Yiran Ding et al.LongRoPE: Extending LLM Context Window Beyond 2 Million To-
kens. 2024. arXiv:2402.13753 [cs.CL].URL:https://arxiv.org/abs/2402.13753.
[15] Jesse Dodge et al.Fine-Tuning Pretrained Language Models: Weight Initializations,
Data Orders, and Early Stopping. 2020. arXiv:2002 . 06305 [cs.CL].URL:https :
//arxiv.org/abs/2002.06305.
[16] Stefan Elfwing, Eiji Uchibe, and Kenji Doya.Sigmoid-Weighted Linear Units for
Neural Network Function Approximation in Reinforcement Learning. 2017. arXiv:1702.
03118 [cs.LG].URL:https://arxiv.org/abs/1702.03118.
[17] J. R. Firth. “Studies in Linguistic Analysis”. In:In J. R. Firth, editor, Studies in
Linguistic Analysis. Oxford, UK: Basil Blackwell, 1957.
[18] GeeksForGeeks.One Hot Encoding in Machine Learning. 2024.URL:https://www.
geeksforgeeks.org/ml-one-hot-encoding/.
[19] Google.Gemma 2 release. July 2024.URL:https://huggingface.co/collections/
google/gemma-2-release-667d6600fd5220e7b967f315.
[20] google.SentencePiece GitHub page.URL:https://github.com/google/sentencepiece.
[21] Gopal Goyal.Sliding Window Technique & Grouped-Query Attention . Mistral 7B.
2023.URL:https://medium.com/@gopalgoyal612002/mistral-llm-architectural-
details-8dc0447fea62.
[22] Suriya Gunasekar et al.Textbooks Are All You Need. 2023. arXiv:2306.11644 [cs.CL].
URL:https://arxiv.org/abs/2306.11644.
[23] Sanda Harabagiu and Andrew Hickl. “Methods for Using Textual Entailment
in Open-Domain Question Answering”. In:Proceedings of the 21st International
Conference on Computational Linguistics and 44th Annual Meeting of the Association
for Computational Linguistics. Ed. by Nicoletta Calzolari, Claire Cardie, and Pierre
Isabelle. Sydney, Australia: Association for Computational Linguistics, July 2006,
pp. 905–912.DOI:10.3115/1220175.1220289.URL:https://aclanthology.org/
P06-1114.
[24] Hangfeng He, Hongming Zhang, and Dan Roth.Rethinking with Retrieval: Faithful
Large Language Model Inference. 2022. arXiv:2301.00303 [cs.CL].URL:https://
arxiv.org/abs/2301.00303.
[25] Kaiming He et al.Deep Residual Learning for Image Recognition. 2015. arXiv:1512.
03385 [cs.CV].URL:https://arxiv.org/abs/1512.03385.
[26] Dan Hendrycks and Kevin Gimpel.Gaussian Error Linear Units (GELUs). 2023.
arXiv:1606.08415 [cs.LG].URL:https://arxiv.org/abs/1606.08415.
[27] Geoffrey Hinton, Oriol Vinyals, and Jeff Dean.Distilling the Knowledge in a Neural
Network. 2015. arXiv:1503.02531 [stat.ML].URL:https://arxiv.org/abs/1503.
02531.
[28] Jeremy Howard and Sebastian Ruder.Universal Language Model Fine-tuning for
Text Classification. 2018. arXiv:1801.06146 [cs.CL].URL:https://arxiv.org/abs/
1801.06146.
[29] Edward J. Hu et al.LoRA: Low-Rank Adaptation of Large Language Models. 2021.
arXiv:2106.09685 [cs.CL].URL:https://arxiv.org/abs/2106.09685.
119

[30] HuggingFace.Guidance documentation.https : / / huggingface . co / docs / text -
generation-inference/conceptual/guidance. Accessed: 2024-08-27. 2024.
[31] Inseq Team.Inseq: Interpretability for Sequence-to-Sequence Models.https://github.
com/inseq-team/inseq. Accessed: 2024-09-10. 2024.
[32] Albert Q. Jiang et al.Mistral 7B. 2023. arXiv:2310.06825 [cs.CL].URL:https:
//arxiv.org/abs/2310.06825.
[33] Dan Jurafsky and James H. Martin.Speech and language processing : an introduc-
tion to natural language processing, computational linguistics, and speech recognition.
Pearson Prentice Hall, 2009.ISBN: 9780131873216 0131873210.URL:http://www.
amazon.com/Speech-Language-Processing-2nd-Edition/dp/0131873210/ref=pd _
bxgy _b_img_y.
[34] Daniel Kahneman.Thinking, Fast and Slow. New York: Farrar, Straus and Giroux,
2011.
[35] Subbarao Kambhampati. “Can large language models reason and plan?” In:An-
nals of the New York Academy of Sciences1534.1 (Mar. 2024), 15–18.ISSN: 1749-6632.
DOI:10.1111/nyas.15125.URL:http://dx.doi.org/10.1111/nyas.15125.
[36] Subbarao Kambhampati.On the Role of LLMs in Planning (ICML 2024 Tutorial).
https://www.dropbox.com/scl/fi/gul511qacx58i5esrqi92/ICM2024- Tutorial.
pdf?rlkey=mmv77ou4qyxi0a6lol3m6ug80&dl=0. Accessed: 2024-08-27. 2024.
[37] Jared Kaplan et al.Scaling Laws for Neural Language Models. 2020. arXiv:2001.
08361 [cs.LG].URL:https://arxiv.org/abs/2001.08361.
[38] Kiseung Kim and Jay-Yoon Lee.RE-RAG: Improving Open-Domain QA Perfor-
mance and Interpretability with Relevance Estimator in Retrieval-Augmented Gener-
ation. 2024. arXiv:2406.05794 [cs.CL].URL:https://arxiv.org/abs/2406.05794.
[39] Takeshi Kojima et al.Large Language Models are Zero-Shot Reasoners. 2023. arXiv:
2205.11916 [cs.CL].URL:https://arxiv.org/abs/2205.11916.
[40] Steve Jerome Lawrence.Learning position with Positional Encoding. 2023.URL:https:
//www.scaler.com/topics/nlp/positional-encoding/.
[41] Quentin Lhoest et al. “Datasets: A Community Library for Natural Language
Processing”. In:Proceedings of the 2021 Conference on Empirical Methods in Natural
Language Processing: System Demonstrations. Ed. by Heike Adel and Shuming Shi.
Online and Punta Cana, Dominican Republic: Association for Computational
Linguistics, Nov. 2021, pp. 175–184.DOI:10 . 18653 / v1 / 2021 . emnlp - demo . 21.
URL:https://aclanthology.org/2021.emnlp-demo.21.
[42] Yanyang Li et al.Making Long-Context Language Models Better Multi-Hop Reason-
ers. 2024. arXiv:2408.03246 [cs.CL].URL:https://arxiv.org/abs/2408.03246.
[43] Nelson F. Liu et al.Lost in the Middle: How Language Models Use Long Contexts.
2023. arXiv:2307.03172 [cs.CL].URL:https://arxiv.org/abs/2307.03172.
[44] Aman Madaan et al.Self-Refine: Iterative Refinement with Self-Feedback. 2023. arXiv:
2303.17651 [cs.CL].URL:https://arxiv.org/abs/2303.17651.
[45] Sourab Mangrulkar et al.PEFT: State-of-the-art Parameter-Efficient Fine-Tuning meth-
ods.https://github.com/huggingface/peft. 2022.
120

[46] Vaibhav Mavi, Anubhav Jangra, and Jatowt Adam. “Multi-hop Question An-
swering”. In:Foundations and Trends® in Information Retrieval17.5 (2024), pp. 457–
586.ISSN: 1554-0669.DOI:10.1561/1500000102.URL:http://dx.doi.org/10.1561/
1500000102.
[47] Meta.Llama 3.1 release. August 2024.URL:https://huggingface.co/collections/
meta-llama/llama-31-669fc079a0c406a149a5738f.
[48] Microsoft.Phi 3 release. June 2024.URL:https://huggingface.co/collections/
microsoft/phi-3-6626e15e9585a200d2d761e3.
[49] Norman Mu et al.Can LLMs Follow Simple Rules?2024. arXiv:2311.04235 [cs.AI].
URL:https://arxiv.org/abs/2311.04235.
[50] OpenAI.tiktoken: Fast BPE tokenization for OpenAI models.https://github.com/
openai/tiktoken. Accessed: 2024-09-07. 2023.
[51] Long Ouyang et al.Training language models to follow instructions with human feed-
back. 2022. arXiv:2203.02155 [cs.CL].URL:https://arxiv.org/abs/2203.02155.
[52] Jirui Qi et al. “Model Internals-based Answer Attribution for Trustworthy Retrieval-
Augmented Generation”. In:ArXivabs/2406.13663 (June 2024).URL:https://
arxiv.org/abs/2406.13663.
[53] Alec Radford and Karthik Narasimhan. “Improving Language Understanding
by Generative Pre-Training”. In: 2018.URL:https://api.semanticscholar.org/
CorpusID:49313245.
[54] Jack W. Rae et al.Scaling Language Models: Methods, Analysis & Insights from Train-
ing Gopher. 2022. arXiv:2112.11446 [cs.CL].URL:https://arxiv.org/abs/2112.
11446.
[55] Rafael Rafailov et al.Direct Preference Optimization: Your Language Model is Secretly
a Reward Model. 2024. arXiv:2305.18290 [cs.LG].URL:https://arxiv.org/abs/
2305.18290.
[56] Colin Raffel et al.Exploring the Limits of Transfer Learning with a Unified Text-to-
Text Transformer. 2023. arXiv:1910.10683 [cs.LG].URL:https://arxiv.org/abs/
1910.10683.
[57] Alexandre Ramé et al.WARP: On the Benefits of Weight Averaged Rewarded Policies.
2024. arXiv:2406.16768 [cs.LG].URL:https://arxiv.org/abs/2406.16768.
[58] Gabriele Sarti et al. “Quantifying the Plausibility of Context Reliance in Neu-
ral Machine Translation”. In:The Twelfth International Conference on Learning Rep-
resentations (ICLR 2024). Vienna, Austria: OpenReview, May 2024.URL:https:
//openreview.net/forum?id=XTHfNGI3zT.
[59] Noah Shinn et al.Reflexion: Language Agents with Verbal Reinforcement Learning.
2023. arXiv:2303.11366 [cs.AI].URL:https://arxiv.org/abs/2303.11366.
[60] Kumar Shridhar, Alessandro Stolfo, and Mrinmaya Sachan.Distilling Reasoning
Capabilities into Smaller Language Models. 2023. arXiv:2212.00193 [cs.LG].URL:
https://arxiv.org/abs/2212.00193.
[61] Jianlin Su et al.RoFormer: Enhanced Transformer with Rotary Position Embedding.
2023. arXiv:2104.09864 [cs.CL].URL:https://arxiv.org/abs/2104.09864.
121

[62] Gemma Team et al.Gemma 2: Improving Open Language Models at a Practical Size.
2024. arXiv:2408.00118 [cs.CL].URL:https://arxiv.org/abs/2408.00118.
[63] Hugo Touvron et al.Llama 2: Open Foundation and Fine-Tuned Chat Models. 2023.
arXiv:2307.09288 [cs.CL].URL:https://arxiv.org/abs/2307.09288.
[64] Hugo Touvron et al.LLaMA: Open and Efficient Foundation Language Models. 2023.
arXiv:2302.13971 [cs.CL].URL:https://arxiv.org/abs/2302.13971.
[65] Sik-Ho Tsang.Brief Review — GLU Variants Improve Transformer. 2023.URL:https:
//sh- tsang.medium.com/brief- review- glu- variants- improve- transformer-
9ee943115ab.
[66] Ashish Vaswani et al.Attention Is All You Need. 2023. arXiv:1706.03762 [cs.CL].
URL:https://arxiv.org/abs/1706.03762.
[67] Xuezhi Wang et al.Self-Consistency Improves Chain of Thought Reasoning in Lan-
guage Models. 2023. arXiv:2203.11171 [cs.CL].URL:https://arxiv.org/abs/
2203.11171.
[68] Jason Wei et al.Chain-of-Thought Prompting Elicits Reasoning in Large Language
Models. 2023. arXiv:2201.11903 [cs.CL].URL:https://arxiv.org/abs/2201.
11903.
[69] Jason Wei et al.Finetuned Language Models Are Zero-Shot Learners. 2022. arXiv:
2109.01652 [cs.CL].URL:https://arxiv.org/abs/2109.01652.
[70] Jason Wei et al.Finetuned Language Models Are Zero-Shot Learners. 2022. arXiv:
2109.01652 [cs.CL].URL:https://arxiv.org/abs/2109.01652.
[71] Johannes Welbl, Pontus Stenetorp, and Sebastian Riedel.Constructing Datasets
for Multi-hop Reading Comprehension Across Documents. 2018. arXiv:1710.06481
[cs.CL].URL:https://arxiv.org/abs/1710.06481.
[72] Sean Welleck et al.Generating Sequences by Learning to Self-Correct. 2022. arXiv:
2211.00053 [cs.CL].URL:https://arxiv.org/abs/2211.00053.
[73] Jason Weston and Sainbayar Sukhbaatar.System 2 Attention (is something you
might need too). 2023. arXiv:2311.11829 [cs.CL].URL:https://arxiv.org/abs/
2311.11829.
[74] Wikipedia contributors.Catastrophic interference — Wikipedia, The Free Encyclope-
dia. [Online; accessed 17-August-2024]. 2024.URL:https://en.wikipedia.org/w/
index.php?title=Catastrophic _interference&oldid=1237425828.
[75] Wikipedia contributors.Dialectic — Wikipedia, The Free Encyclopedia.https : / /
en.wikipedia.org/w/index.php?title=Dialectic&oldid=1244221571. [Online;
accessed 7-September-2024]. 2024.
[76] Brandon T. Willard and Rémi Louf.Efficient Guided Generation for Large Language
Models. 2023. arXiv:2307.09702 [cs.CL].URL:https://arxiv.org/abs/2307.
09702.
[77] Thomas Wolf et al. “Transformers: State-of-the-Art Natural Language Process-
ing”. In:Proceedings of the 2020 Conference on Empirical Methods in Natural Lan-
guage Processing: System Demonstrations. Ed. by Qun Liu and David Schlangen.
Online: Association for Computational Linguistics, Oct. 2020, pp. 38–45.DOI:
10.18653/v1/2020.emnlp-demos.6.URL:https://aclanthology.org/2020.emnlp-
demos.6.
122

[78] Ruibin Xiong et al.On Layer Normalization in the Transformer Architecture. 2020.
arXiv:2002.04745 [cs.LG].URL:https://arxiv.org/abs/2002.04745.
[79] Zhilin Yang et al.HotpotQA: A Dataset for Diverse, Explainable Multi-hop Question
Answering. 2018. arXiv:1809.09600 [cs.CL].URL:https://arxiv.org/abs/1809.
09600.
[80] Kayo Yin and Graham Neubig.Interpreting Language Models with Contrastive Ex-
planations. 2022. arXiv:2202.10419 [cs.CL].URL:https://arxiv.org/abs/2202.
10419.
[81] Eric Zelikman et al.Quiet-STaR: Language Models Can Teach Themselves to Think
Before Speaking. 2024. arXiv:2403.09629 [cs.CL].URL:https://arxiv.org/abs/
2403.09629.
123

Appendix A
PECoRe invocation for context filtering
The context filtering process has its core in the following function:
def run(question, passage, p):
tokens = tokenizer.tokenize(passage)
invoke _pecore(passage, question, p)
return select _passages(passage, question, p, tokens)
whereinvoke _pecore(passage, question, p)invokes PECoRe [58] and outputs the CCI
scores of each context token:
def invoke _pecore(passage, question, p):
pecore _args = AttributeContextArgs(
model _name _or_path="microsoft/Phi-3-mini-4k-instruct",
attribution _method="saliency",
attributed _fn="contrast _prob _diff",
context _sensitivity _metric="kl _divergence",
context _sensitivity _std_threshold=1,
context _sensitivity _topk = find _top_p(passage, p),
attribution _std_threshold=None,
attribution _topk=None,
input _current _text=question,
input _context _text=text _passage,
contextless _input _current _text=""" <|system|>
You are a helpful assistant that provide concise and accurate answers.<|
end|>
<|user|>
{current}<|end|>
<|assistant|>""",
input _template="""<|system|>
You are a helpful assistant that provide concise and accurate answers.<|
end|>
<|user|>
{context}
{current}<|end|>
<|assistant|>""",
contextless _output _current _text=""" {current}""",
output _template="{current}",
special _tokens _to_keep=[’<|system|>’, ’<|end|>’, ’<|assistant|>’, ’<|user|>’],
124

decoder _input _output _separator="",
show _viz=False,
save _path=None,
viz_path=None,
generation _kwargs={’max _new_tokens’: 50},
)
out = attribute _context _with _model(pecore _args, inseq _model)
return out
That are subsequently used to select only the sentences containing at least one in-
fluential token.
125

Appendix B
Usage ofguidanceframework for
multiple-choice questions
Example of usage of theguidanceframework for constraining the model to output ex-
actly one of the options given a multiple-choice question:
1. import theguidancemodel (and the corresponding tokenizer):
from guidance import models
guidance _model = models.Transformers(model, tokenizer)
2. define the desired prompt:
prompt = """
You are a helpful AI assistant.
You are given a question and the relevant context to answer it.
Answer briefly to the question with just one of the given options.
Question: Which year is Halley’s Comet expected to return to the solar system
?
Options: [2110, 2045, 2086, 2061]
Context: Astronomers have now linked the comet’s appearances to observations
dating back more than 2,000 years. Halley was last seen in Earth’s skies in
1986
and was met in space by an international fleet of spacecraft.
It performs a regular 76-year journey around the Sun.
Assistant:
"""
3. mask everything that is not inside the grammar (i.e. not one of the provided
options) using theselectfunction of theguidanceframework:
from guidance import select
answer = guidance _model + prompt + select([2110, 2045, 2086, 2061])
126

Appendix C
Thesis
Code used for thethesis’ generation:
from guidance import models, select
guidance _model = models.Transformers(model, tokenizer, temperature=0.0)
def create _message _thesis(question, options, context):
messages = [
{"role": "system", "content": """
You are an helpful AI assistant.
You have to provide helpful answers to the user’s questions based on the
context:
""" + context},
{"role": "user", "content": user _content}
]
user _content = "Answer to the following question: " + question + \
" providing one of these options as answer: " + options + \
"Assistant: "
return messages
def thesisGeneration(question, options, context):
prompt = create _message _thesis(question, options, context)
answer = guidance _model + prompt + select(options)
return answer
127

Appendix D
Antithesis
We used the following function to generate theantithesis:
def antithesisGeneration(question, options, thesis, context):
prompt = create _message _antithesis(question, thesis, options, context)
output = pipeline(prompt, **generation _args)
return output[0][’generated _text’]
wherecreate _message _antithesis()is the function responsible to describe how we
want the task to be performed, both using instructions and a concrete example that is
also augmented with the new task:
def create _message _antithesis(question, options, thesis, context, one _shot _example):
messages = [
{"role": "system", "content": """
You are an helpful AI assistant. You are asked to determine the most correct
answer for
a given question, provided a set of possible options. You also have at
disposal a first
tentative answer that you are required to check with respect to the question
and the
relevant context. Your goal is to decree which is the most correct answer to
the question
between the available options.
Here’s an example of how to do it:
"""},
{"role": "user", "content": one _shot _example
},
{"role": "system", "content": "Now do the same for the following question:"},
{"role": "user", "content": user _content}
]
user _content = "Question: " + question + "\n Options: " + options +
"\n Candidate answer: " + candidate + "\n Context: " + context +
"\n Assistant: \n"
return messages
128

whereone _shot _exampleis the concrete example of expected behaviour of theantithe-
sis’ step of the pipeline.
129

Appendix E
Synthesis
Similarly as before, we report here the generation prompt:
def create _message _presynthesis(question, thesis, antithesis, options, context,
few_shot _example):
messages = [
{"role": "system", "content": """
You are an helpful AI assistant. You are asked to determine the most correct
answer
for a given question, provided a set of possible options.
You also have at disposal a first tentative answer and a suggestion on which
is the
correct answer. Your goal is to decree which is the most correct answer to the
question between the available options according to the context.
Here’s an example/a few examples of how to do it:
"""},
{"role": "user", "content": few _shot _example
},
{"role": "system", "content": "Now do the same for the following question:"},
{"role": "user", "content": user _content}
]
user _content = "Question: " + question + "\n Options: " + options +
"\n Candidate answer: " + thesis + "\n Suggestion: " + antithesis +
"\n Context: " + context + "\n Assistant: \n"
return messages
130

Appendix F
Relative improvements between
different pipeline steps
Expanded table showing the relative improvements between different pipeline steps:
Dataset Setting Step Phi-mini Phi-medium Gemma-2B Gemma-9B LlaMa-3.1-8B
WikiHopbaselineT-A +20.1 +25.1 +16.5 +5.1 +25.9
A-S +0.2 +2.0 +5.9 +0.9 +0.6
cotT-A +7.8 +18.6 +1.0 +2.7 +17.4
A-S +3.9 +3.0 +2.4 +0.9 +5.4
ctxT-A +4.2 +15.6 +16.6 +8.7 +28.0
A-S +0.3 +1.5 -5.9 -3.3 -3.9
bridgebaselineT-A +35.3 +33.7 +27.4 +22.2 +41.9
A-S +0.5 +0.5 -1.8 +0.2 +0.2
cotT-A +40.4 +34.4 +30.5 +21.8 +40.0
A-S -2.3 +0.7 -1.7 -0.8 +0.9
ctxT-A +41.9 +38.0 +33.5 +25.8 +44.4
A-S +0.6 +0.3 -4.7 -4.4 -2.2
comparisonbaselineT-A +24.6 +38.9 +25.6 +29.2 +38.4
A-S +2.7 +0.6 +3.4 +0.6 +0.9
cotT-A +34.7 +40.9 +25.6 +29.2 41.5
A-S -0.9 +1.2 +0.6 +1.2 +1.6
ctxT-A +37.8 +42.9 +28.7 +29.8 +42.9
A-S +0.0 -1.7 +1.7 -2.3 -0.3
131