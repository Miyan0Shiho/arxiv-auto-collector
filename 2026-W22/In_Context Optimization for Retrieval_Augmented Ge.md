# In-Context Optimization for Retrieval-Augmented Generation: A Gradient-Descent Perspective

**Authors**: Mingchen Li, Jiatan Huang, Chuxu Zhang, Liang Zhao, Hong Yu

**Published**: 2026-05-25 22:04:54

**PDF URL**: [https://arxiv.org/pdf/2605.26356v1](https://arxiv.org/pdf/2605.26356v1)

## Abstract
In-context learning has recently been linked to implicit gradient descent in linear self-attention models, suggesting that context can induce a forward-pass update. Retrieval-augmented generation (RAG) also relies on context, but retrieved documents are usually treated as static evidence rather than signals for adaptation. We study RAG as an in-context optimization process. First, we show that one linear self-attention layer can implement one gradient-descent step on a unified linearized RAG objective covering both projection-based and dot-product retrieval interfaces. This gives an exact regime where retrieval-augmented prediction and in-context optimization coincide. We use this result not as a literal model of LLM computation, but as a guide for adapting the interaction between queries and retrieved evidence. We then test the boundary of this correspondence: it remains stable under controlled linear extensions, but becomes feature-distribution dependent under nonlinear architectures. Finally, we turn this view into a lightweight method for frozen RAG LLMs. The method keeps the retriever and backbone fixed, and predicts a context-conditioned update to a generator-side evidence-use interface. Across seven QA benchmarks, two retrievers, and two frozen LLM backbones, this forward-only update improves a shared-interface baseline, transfers to held-out tasks, and approaches test-time gradient adaptation at much lower per-query cost.

## Full Text


<!-- PDF content starts -->

In-Context Optimization for Retrieval-Augmented
Generation: A Gradient-Descent Perspective
Mingchen Li1∗, Jiatan Huang2∗, Chuxu Zhang2, Liang Zhao3, Hong Yu1
1University of Massachusetts, Amherst2University of Connecticut
3Emory University
Abstract
In-context learning has recently been linked to implicit gradient descent in linear
self-attention models, suggesting that context can induce a forward-pass update.
Retrieval-augmented generation (RAG) also relies on context, but retrieved docu-
ments are usually treated as static evidence rather than signals for adaptation. We
study RAG as an in-context optimization process. First, we show that one linear
self-attention layer can implement one gradient-descent step on a unified linearized
RAG objective covering both projection-based and dot-product retrieval interfaces.
This gives an exact regime where retrieval-augmented prediction and in-context
optimization coincide. We use this result not as a literal model of LLM compu-
tation, but as a guide for adapting the interaction between queries and retrieved
evidence. We then test the boundary of this correspondence: it remains stable
under controlled linear extensions, but becomes feature-distribution dependent
under nonlinear architectures. Finally, we turn this view into a lightweight method
for frozen RAG LLMs. The method keeps the retriever and backbone fixed, and
predicts a context-conditioned update to a generator-side evidence-use interface.
Across seven QA benchmarks, two retrievers, and two frozen LLM backbones, this
forward-only update improves a shared-interface baseline, transfers to held-out
tasks, and approaches test-time gradient adaptation at much lower per-query cost.
1 Introduction
Large language models (LLMs) have achieved strong performance across many natural-language
tasks, but adapting them to knowledge outside their static pretraining corpus remains difficult.
Retrieval-augmented generation (RAG) [ 23] addresses this limitation by conditioning a frozen LLM
on documents retrieved from an external corpus. However, retrieval alone does not solve the full
adaptation problem. After relevant documents are retrieved, the model must still decide how to use
them for a new task, domain, or query distribution.
Existing RAG systems usually address this problem in one of three ways. The first is to keep both the
retriever and generator fixed, and simply prepend retrieved documents to the input. This strategy is
efficient, but it treats retrieved documents as static evidence and gives the model no mechanism to
adjust how evidence should be used. The second is to fine-tune the retriever, the generator, or both.
This can improve task performance, but it requires additional training and can be expensive when
the task or domain changes. The third is to use in-context learning (ICL), where a few input-output
examples are provided at inference time. ICL is attractive because it avoids full model retraining, but
it is still unclear whether these examples merely provide extra demonstrations, or whether they can
induce a more systematic update to how a RAG model uses retrieved evidence.
This paper asks a simple question:can retrieved evidence and a few RAG examples act not only as
context to read from, but also as a signal for adapting how the model uses evidence?Answering
∗indicates equal contribution
Preprint.arXiv:2605.26356v1  [cs.CL]  25 May 2026

this question requires connecting two views that have mostly been studied separately. On one side,
recent theory shows that, under linear self-attention, in-context learning can implement gradient
descent on the examples in the context [ 41,2,25]. This suggests that context can behave like a
forward-pass update, rather than only as additional input text. On the other side, RAG introduces
structure that is absent from standard ICL theory: a query, retrieved documents, query-evidence
interactions, and a generator that must combine them to produce an answer. It remains unclear
whether the gradient-descent view of ICL extends to retrieval-augmented prediction, and whether
such a view can guide practical adaptation in real RAG systems.
We study RAG from this in-context optimization perspective. Our goal is not to claim that modern
retrieval-augmented LLMs literally perform gradient descent during inference. Instead, we use a
controlled linear setting to identify where such an update would act. In linear RAG, the relevant
update acts on the interaction between the query and retrieved evidence. This gives a simple design
principle for LLM-scale RAG: rather than changing which documents are retrieved, we adapt how the
frozen generator uses the retrieved documents. Guided by this principle, we propose a forward-only
adaptation method for frozen RAG LLMs. The method keeps both the external retriever and the LLM
backbone fixed, and adapts only a lightweight generator-side evidence-use interface implemented
with LoRA. At inference time, given new few-shot RAG demonstrations, the predictor produces the
update in a single forward pass, enabling the generator to adjust how it uses retrieved documents
without re-training on the new dataset.
We develop this idea in three steps.First, we prove that one linear self-attention layer can implement
one gradient-descent step on a unified linearized RAG objective covering both projection-based and
dot-product retrieval interfaces.Second, we test how far this correspondence extends beyond the
exact construction. A trained self-attention layer closely matches the constructed gradient-descent
predictor under controlled linear shifts, varying document counts, and stacked depths, while nonlinear
architectures and real-world regression data reveal a clear dependence on feature distribution.Third,
we use the optimization view to guide LLM-scale RAG adaptation. Across seven QA benchmarks,
two retrievers, and two frozen LLM backbones, the predicted update improves a shared-interface
baseline, transfers to held-out tasks, and approaches test-time gradient adaptation at much lower
per-query cost. Our contributions are summarized as follows:
•An in-context optimization view of linear RAG.We extend the ICL-as-gradient-descent
perspective from generator-only prediction to retrieval-augmented prediction. We prove
that one linear self-attention layer can implement one gradient-descent step on a unified
linearized RAG loss covering both linear-projection and dot-product retrieval interfaces. We
also show that stacking Klinear self-attention layers gives a multi-step view of in-context
optimization for linear RAG.
•A boundary analysis beyond the exact linear setting.We test when the linear construction
remains predictive and when it breaks. On synthetic linear regression tasks, a trained self-
attention layer closely matches the constructed gradient-descent predictor under distribution
shift, varying document counts, and stacked depths. On nonlinear architectures and four
real-world regression datasets, the alignment degrades in a structured way and becomes
sensitive to feature distribution.
•Adapting evidence use without test-time backpropagation.We use the optimization view
to guide adaptation in frozen RAG LLMs. Rather than changing the external retriever, we
adapt a generator-side evidence-use interface implemented with Q/K/V LoRA modules. A
small context-conditioned predictor amortizes the autograd-defined K-step update to this
interface. Across seven QA benchmarks, two backbones, and two retrievers, the predicted
update improves a shared-interface baseline, transfers to held-out domains, and approaches
test-time gradient adaptation at much lower per-query cost.
2 Related Work
Retrieval-augmented generation (RAG) conditions a language model on documents retrieved from
an external corpus [ 23,12,19,16,6,45]. Prior work has improved RAG through better retrieval,
prompting, evidence fusion, and joint retriever-generator training [ 33,3,15,46]. Our focus is
complementary: rather than changing which documents are retrieved, we study how a frozen generator
2

can adapt its use of already-retrieved evidence. We position this contribution relative to three lines of
work.
In-context learning as gradient descent.A growing line of theory interprets in-context learning
as implicit optimization. Under linear self-attention, a single ICL forward pass can implement one
gradient-descent step, and stacked layers can implement multiple steps [ 41,2,25,8]. Later work
extends this view to preconditioned gradient descent [ 1], in-context algorithm selection [ 4], the
role of depth [ 40,10], and kernel-regression interpretations of attention [ 35,34]. These analyses
mainly study generator-only settings, often through linear regression or simplified attention. RAG
introduces additional structure, including retrieved documents, query-evidence interactions, and
evidence-conditioned generation. We extend the gradient-descent view to a linearized RAG setting
and use it to identify where evidence-use adaptation should act.
Context-conditioned weight prediction.Another line of work learns auxiliary networks that
produce model updates from a small context. HyperTuning [ 29] predicts soft prompts or low-rank
weights from few-shot examples. HyperFlow [20] learns support-conditioned fine-tuning dynamics.
MAC [ 37] maps documents into memory modulations. MEND [ 27] maps fine-tuning gradients
into knowledge edits. RAG-GD follows the broad template of predicting an update from context,
but differs in both the target and the adaptation site. The target is not a downstream task loss, a
meta-learning objective, a memory objective, or a single editing gradient. Instead, the predictor
matches an autograd-defined K-step SGD update induced by RAG-formatted demonstrations. The
adapted parameters are also restricted to a generator-side evidence-use interface, while the retriever
and backbone remain fixed.
Test-time adaptation.Standard adaptation either updates model parameters before deployment,
as in fine-tuning and LoRA [ 14], or leaves the model unchanged at inference, as in pure ICL. Test-
time training [ 36] lies between these extremes by updating parameters for each test instance, but
this requires per-instance backpropagation and becomes expensive for large LLMs. Recent studies
compare ICL, fine-tuning, and trainable RAG as system-level adaptation strategies [ 42,28,24].
RAG-GD targets the same goal of adapting at inference time, but amortizes the update: a small
predictor emits a LoRA update to the generator’s evidence-use interface in one forward pass. Thus, it
avoids backpropagation through the LLM at deployment while keeping both the external retriever
and the frozen backbone unchanged.
3A Linear RAG Setting Where Self-Attention Implements Gradient Descent
This section establishes the linear-regime basis for our in-context optimization view of RAG. We study
a controlled setting in which retrieval-augmented prediction and gradient descent can be connected
exactly. The goal is not to model modern RAG systems literally: real retrievers involve discrete
document selection, and modern LLMs are deep and nonlinear. Instead, we isolate a differentiable
retrieval-augmented prediction problem and show that one linear self-attention layer can realize the
prediction shift produced by one gradient-descent step. The proof and explicit construction are in
Appendix A, and the derivations for the retrieval variants are in Appendix B.
3.1 Self-Attention
We begin with a multi-head self-attention block parameterized by θ={P h, Wh,V, Wh,K, Wh,Q}H
h=1.
Given tokens{e 1, . . . , e N} ⊂Rd, the update for tokene jis
ej←ej+ SA θ(j,{e i}N
i=1) =e j+X
hPhVhsoftmax(K⊤
hqh,j),(1)
where Vh,Kh, andqh,jare the value matrix, key matrix, and query vector for head h. Following [ 41,
40], we remove the softmax and bias terms to obtain the linear self-attention (LSA) update:
ej←ej+ LSA θ(j,{e i}N
i=1) =e j+X
hPhVhK⊤
hqh,j.(2)
3

3.2 A Unified Linearized RAG Predictor
We use a linearized abstraction of retrieval-augmented prediction. Rather than modeling discrete top- k
selection, this abstraction captures a differentiable interface in which query features and retrieval-
derived features jointly determine the prediction. Both a projection-based retrieval interface [ 23] and a
dot-product retrieval interface [ 19] can be written as y=W 1x1+W 2x2where x1denotes the query-
side feature and x2denotes the retrieval-derived feature. For the projection-based interface, we set
x1=xq,x2=D , and W2≜W 1Wd, where Wdprojects document embeddings into the prediction
space. For the dot-product interface, we set x1=x2=xqandW2=W z P
idid⊤
i
M⊤,where M
parameterizes query-document similarity. For tractability, we use the shared-encoder simplification
M=W⊤
eWe, soMis symmetric. The general DPR formulation [ 19] allows separate query and
document encoders, withM=W⊤
qWd. Full derivations are provided in Appendix B.
3.3 Optimization Objective
Given training examples{(xi
1, xi
2, yi)}N
i=1, we consider the squared loss
L(W 1, W2) =1
2NNX
i=1W1xi
1+W 2xi
2−yi2.(3)
One gradient-descent step with learning rateηgives
∆W k=−η∇ WkL=−η
NNX
i=1 
W1xi
1+W 2xi
2−yi
(xi
k)⊤, k∈ {1,2}.(4)
For a query token with features (x1, x2), the corresponding prediction shift is ∆y≜∆W 1x1+
∆W 2x2.Thus,∆yis the change in prediction after updatingW ktoW′
k=W k+ ∆W k.
3.4 Linear Self-attention Reproduces one Gradient Step
Lemma 1(Linear self-attention implements one RAG gradient step).Consider a 1-head linear
self-attention layer, context tokens ei= (xi
1, xi
2, yi)fori= 1, . . . , N , and a query token ej=
(xj
1, xj
2, yj). Let ∆W 1and∆W 2be the one-step gradient-descent updates in Eq. 4. There exist
matrices WK, WQ, WVand an output projection Psuch that one LSA update changes only the
y-coordinate ofe j:
ej←ej+
0,0,∆W 1xj
1+ ∆W 2xj
2
.(5)
Equivalently,
PV K⊤qj=
0,0,∆W 1xj
1+ ∆W 2xj
2
.(6)
Therefore, the LSA update exactly matches the prediction shift induced by one gradient-descent step
on the unified linearized RAG predictor.
The construction is given in Appendix A. Intuitively, the value projection encodes the residual
W1xi
1+W 2xi
2−yi. The key-query interaction computes the inner products (xi
1)⊤xj
1and(xi
2)⊤xj
2.
The output projection then writes the resulting weighted residual sum into the query token’s prediction
coordinate.
This construction also gives a controlled multi-step analogue. If each LSA layer represents one
gradient-like update, then afterKlayers,
ˆy(K)
N+1= ˆy(0)
N+1+K−1X
t=0
∆W(t)
1x1
N+1+ ∆W(t)
2x2
N+1
,(7)
where ∆W(t)
1and∆W(t)
2are the implicit updates represented by layer t. We use this multi-step
view as a linear-regime guide rather than as a literal claim about frozen LLM computation. In later
sections, this view motivates a forward-only mechanism that adapts how a frozen generator uses
retrieved evidence.
4

0 2500 5000 7500
Training steps1.01.11.21.3LossRAG
Trained TF
0 2500 5000 7500
Training steps0.00.51.01.52.0
0.0LossPartial diff
Preds diffPartial diff
Preds diff
0.00.20.40.60.81.0
Cosine simPartial cosinePartial cosine
0 2500 5000 7500
Training steps0.60.70.80.9LossRAG
Trained TF
0 2500 5000 7500
Training steps0.00.51.01.52.0
0.0LossPartial diff
Preds diffPartial diff
Preds diff
0.00.20.40.60.81.0
Cosine simPartial cosinePartial cosineFigure 1: Single-layer LSA reproduces one gradient-descent step on the unified linearized RAG loss.
The two left panels report the projection-based interface, and the two right panels report the dot-
product interface. Across both variants, the trained LSA layer and the constructed gradient-descent
predictor are nearly indistinguishable on held-out tasks.
4 Testing the Boundary of the Linear Correspondence
Lemma 1 gives an exact correspondence in a controlled linear setting. We now ask how far this
correspondence remains predictive when the setting is varied. The experiments have two goals: first,
to verify that a trained linear self-attention layer can reproduce the constructed gradient-descent
predictor; second, to identify where the correspondence begins to break beyond the exact regime.
4.1 Linear-Regime Verification
Each token concatenates an input feature, a retrieval-derived feature, and a target, ei= (x i, zi, yi)
fori= 1, . . . , N . The auxiliary slot ziinstantiates the unified RAG view. For the projection-
based interface, ziis a document-derived feature. For the dot-product interface, zi=x i, and
document information is injected into the keys and values. We train an LSA layer θto minimize
expected squared error across tasks, using minibatch SGD over freshly sampled tasks. Following
prior work [ 9,41], each task is generated from a teacher with weights Wτ∼ N(0, I) . Inputs are
sampled as xτ,i∼ U(−1,1)nI, and targets are generated by yτ,i=W1
τx1
τ,i+W2
τx2
τ,i. We set
N=n I= 10 and sweep the document count k∈ {2,5,10,25} . We compare the trained layer θ∗
with the constructed predictor that exactly realizes one gradient-descent step on the unified RAG
loss. On Tval= 104held-out tasks, we report the prediction difference ∥ˆyθ∗−ˆyθ,rag∥2, the cosine
similarity between input sensitivities ∂ˆy/∂x test, and the corresponding sensitivity ℓ2difference.
Details are in Appendix C. Figure 1 verifies the construction for both retrieval interfaces. The trained
LSA layer closely matches the constructed predictor: the loss difference is small, the sensitivity
cosine is close to 1, and the sensitivity ℓ2difference is negligible. This numerically confirms the
algebraic correspondence in Lemma 1.
4.2 Controlled Stress Tests
We next test whether the agreement persists under controlled changes within the linear regime. We
vary the document count, shift the test-input distribution, and stack LSA layers with shared parameters.
When sweeping n∈ {2,5,10,25} and shifting the test-input range to α∈ {0.5,1,1.5,2} while
keeping training fixed at α= 1 , the loss difference between the trained Transformer and the gradient
predictor remains small (Figure 5, Appendix C). Stacking LSA layers further supports the multi-step
picture in Eq. 7. At depths 2and5, the loss and prediction differences remain small across document
counts. The residual gap at Docs = 25 also shrinks as depth increases (Figure 6, Appendix C).
The projection-based interface shows similar behavior (Appendix E). These results suggest that the
linear correspondence is not a fragile single-step artifact, but remains stable under controlled linear
extensions.
4.3 Nonlinear Stress Test
We then examine where the correspondence begins to break. We add MLP layers after the input
embedding and evaluate on four real-world regression datasets: California Housing, Bike Sharing,
Wine Quality, and Predict Calorie Expenditure. We focus on the dot-product interface throughout.
Dataset details are in Appendix F. This experiment is diagnostic. We do not claim that normalization
solves RAG adaptation. Instead, we use normalization to control the feature geometry that interacts
with dot-product retrieval. We compare Z-score [ 5], Min–Max [ 5], rank-based normalization [ 7],
5

0 2000 4000
Training steps0.00.20.40.6loss
Z-Score
Min-Max
Rank
T anh
0 2000 4000
Training steps0.40.50.60.70.80.91.0Cosine simZ-Score
Min-Max
Rank
T anh
0 2000 4000 6000 8000
Training steps0.20.30.40.50.6Model diffZ-Score
Min-Max
Rank
T anh
0 2500 5000 7500
Training steps0.00.20.40.60.81.0Preds diffZ-Score
Min-Max
Rank
T anh
0 2000 4000
Training steps0.00.20.40.6loss
Z-Score
Min-Max
Rank
T anh
0 2000 4000
Training steps0.40.50.60.70.80.91.0Cosine simZ-Score
Min-Max
Rank
T anh
0 2000 4000 6000 8000
Training steps5101520Model diffZ-Score
Min-Max
Rank
T anh
0 2500 5000 7500
Training steps0.00.20.40.60.81.0Preds diffZ-Score
Min-Max
Rank
T anhFigure 2: Effect of input normalization on the alignment between the trained nonlinear Transformer
and the gradient-descent predictor under the dot-product interface.Top row:Bike Sharing.Bottom
row:California Housing. Columns report loss difference, sensitivity cosine, model difference, and
prediction difference. Min–Max normalization closely matches the gradient-descent predictor on
Bike Sharing, where features are bounded and roughly uniform. On California Housing, where
features are skewed and heavy-tailed, the alignment degrades.
and Tanh normalization. The training set is used as the retrieval corpus and is normalized with
Z-score throughout. Only the input-side normalization is varied, and alignment is measured using
the same metrics as in Section 4.1. Figure 2 shows two representative cases. On Bike Sharing,
Min–Max normalization gives the closest agreement between the trained nonlinear Transformer
and the gradient-descent predictor, likely because the features are bounded and not dominated by
outliers. On California Housing, the alignment is weaker: skewed and heavy-tailed features make
dot-product geometry more sensitive to outliers. The sensitivity cosine drops, the model difference
grows, and prediction differences become less stable. The same pattern appears on the remaining
datasets. Predict Calorie Expenditure behaves similarly to Bike Sharing, while Wine Quality behaves
more like California Housing (Figure 7, Appendix D). Overall, the linear optimization view remains
informative when feature geometry is stable, but becomes less predictive when retrieval-derived dot
products are dominated by skewed or heavy-tailed features. This empirical boundary supports our use
of the linear construction as a guide for adaptation, rather than as a literal model of LLM computation.
Next, we use this view to design a forward-only update to the generator-side evidence-use interface
in LLM-scale RAG.
5 LLM-Scale RAG: Amortizing the Gradient-Descent Update
We now instantiate this view as RAG-GD, a forward-only adaptation method for frozen billion-
parameter RAG LLMs. In this setting, we do not assume an exact equivalence between an LLM
forward pass and gradient descent. Instead, we use gradient descent as an operational target for
adapting how the generator uses retrieval-conditioned information. Given a few RAG-formatted
demonstrations, we use autograd during training to compute the update that gradient descent would
make to the generator-side retrieval interface. We then train a lightweight predictor to approximate
this update. At inference time, the predictor produces the update in a single forward pass, without
further training the RAG system or backpropagating through the frozen LLM.
Concretely, we first train a base retrieval adapter Wret
0on RAG-formatted examples from the NQ
training split. It is a low-rank LoRA perturbation to the Q/K/V projections of every attention layer in
a frozen LLM. All inputs are RAG-formatted instances (x,D x, y), where yis the gold answer. In
this work, we use a fixed external retriever, either BM25 or E5, to select the retrieved documents Dx
for each question xfrom a fixed corpus. This adapter serves as a generator-side retrieval interface: it
does not select documents, but modulates how the generator uses evidence selected by BM25 or E5.
The predictor gϕis then meta-trained on few-shot support contexts C={(x i,Di, yi)}N
i=1from NQ,
6

TriviaQA, HotpotQA, 2WikiMultiHopQA, and MuSiQue. PopQA and Bamboogle are held out from
both stages and used only for evaluation.
5.1 Supervision Target: Autograd-Defined Interface Update
The supervision target is the update that KSGD steps would produce on the generator-side retrieval
interface using a support contextC. Starting fromW(0)
ret=Wret
0, we run
∆W(K)
GD(C) =W(K)
ret−Wret
0, W(t+1)
ret =W(t)
ret−η∇L(W(t)
ret;C),(8)
fort= 0, . . . , K−1 , where Lis the answer-token cross-entropy conditioned on the question
and retrieved documents. We compute ∆W(K)
GD(C)with autograd and detach it from the predictor
optimizer. Equation 8 is not a theorem-preserving lift of Lemma 1. The setting changes from squared
regression with a linear predictor to cross-entropy training of a deep LLM, and the adapted parameter
becomes a low-rank Q/K/V LoRA interface. Its role is practical: it provides an optimization-derived
target for how RAG demonstrations should adjust generator-side evidence use.
5.2 Predictor Architecture and Matching Objective
The predictor gϕis a context encoder with per-layer, per-projection update heads. Each demonstration
(xi,Di, yi)∈C is formatted by concatenating the question, retrieved documents, and gold answer.
The frozen LLM with the base adapter encodes each sequence, and we use the EOS hidden state
hi∈Rdh. We aggregate demonstrations by mean pooling, ¯h(C) =1
NPN
i=1hi. For each layer ℓ
and projection type π∈ {Q, K, V} , the update head outputs g∆W ℓ,π=U ℓ,πV⊤
ℓ,π∈Rd×d,where
Uℓ,π, Vℓ,π∈Rd×rare generated by a two-layer MLP from ¯h(C) . The rank rmatches the base
adapter, so gϕ(C)has the same shape as Wret
0. We train gϕto match the autograd-defined target per
layer and projection type. Let∆W⋆
ℓ,π≜[∆W(K)
GD(C)] ℓ,π.The matching loss is
Lmatch (ϕ;C) =X
ℓ,π"
1−⟨g∆W ℓ,π,∆W⋆
ℓ,π⟩
∥g∆W ℓ,π∥F∥∆W⋆
ℓ,π∥F+λlog∥g∆W ℓ,π∥F
∥∆W⋆
ℓ,π∥F#
,(9)
where the cosine term matches direction and the log-magnitude term matches scale. We use λ= 0.1
throughout. At deployment, the predictor emits g∆W(C) in one forward pass, and the frozen LLM
answers with the adapted interfaceWret
0+g∆W(C).
5.3 Benchmarks, Baselines, and Metrics
BenchmarksWe evaluate on seven open-domain QA benchmarks: NQ, TriviaQA, PopQA, Hot-
potQA, 2WikiMultiHopQA, MuSiQue, and Bamboogle. NQ, TriviaQA, and PopQA are single-hop,
while the remaining four are multi-hop. We use Qwen 2.5-7B-Instruct [ 31] and Llama 3.1-8B-
Instruct [ 11] as frozen backbones. Each query is augmented with the top five documents from BM25
or E5-large [ 43], using the same retrieval cache for all methods. The support size is N= 3 , and we
train predictors withK∈ {1,5,10}.
Baselines and MetricsWe compare RAG-GD with six baselines.Query Onlyuses no retrieved
documents.Vanilla RAGprepends retrieved documents to the prompt.Base adapterapplies Wret
0
without context-conditioned perturbation.+ few shotvariants concatenate support demonstrations
into Vanilla RAG or Base adapter prompts.Prompt tuning[ 22] learns a soft prefix using the same
supervision pool as Wret
0.HyperTuning[ 29] uses the same predictor architecture as RAG-GD, but
trains through downstream task loss rather than the autograd-defined target.TT-SGDperforms K
SGD steps on Cat test time and serves as the non-amortized reference. RAG-GD shares Wret
0with
Base adapter and adds only g∆W(C) . Table 1 reports the headline comparison against no-perturbation
baselines. Full results for Prompt tuning, HyperTuning, TT-SGD, and + few shot variants are in
Appendix I. We report SQuAD-style [32] exact match (EM) and token-overlap F1.
5.4 Results
Table 1 reports the headline comparison between RAG-GD ( K=5 ) and the no-perturbation baselines.
Figures 3 and 4 compare against additional context-conditioned methods, including Prompt tuning,
7

Table 1: Main QA results across seven benchmarks, two frozen LLM backbones, and two retrievers.
RAG-GD ( K=5 ) applies a context-conditioned update on top of the same static retrieval adapter
Wret
0used by Base adapter. Bold values mark the best result per column within each backbone block.
Full context-conditioned baseline results are in Appendix I.
Method RetrieverSingle-Hop QA Multi-Hop QA Avg.
NQ TriviaQA PopQA HotpotQA 2Wiki MuSiQue Bamboogle
EM F1 EM F1 EM F1 EM F1 EM F1 EM F1 EM F1 EM F1
Qwen-2.5-7B
Query Only – 15.95 24.28 43.33 49.51 16.02 19.76 18.40 25.39 23.91 28.12 3.80 10.57 11.20 18.02 18.94 25.09
Vanilla RAGBM25 27.61 36.66 58.24 65.77 28.84 33.18 31.28 41.25 27.87 33.24 5.87 13.05 10.40 21.16 27.16 34.90
E5 39.16 50.03 62.99 70.80 44.03 50.21 32.45 42.21 25.48 31.43 5.79 12.77 18.40 26.74 32.61 40.60
Base adapterBM25 32.57 41.45 60.11 67.93 31.55 35.63 32.31 43.45 28.22 34.13 6.41 15.46 16.80 26.01 29.71 37.72
E5 41.77 51.22 63.31 71.62 47.05 51.82 33.78 44.64 27.89 33.97 6.95 15.76 18.40 28.78 34.16 42.54
RAG-GD (K=5)BM2534.46 43.54 63.27 70.69 33.22 37.69 35.54 47.14 28.86 34.48 9.26 19.73 22.40 32.85 32.43 40.87
E542.91 52.71 65.98 73.60 48.12 52.61 35.54 47.00 29.67 35.47 9.14 19.26 25.60 35.12 36.71 45.11
Llama-3.1-8B
Query Only – 22.46 32.51 52.67 59.79 20.63 25.09 18.31 25.71 26.39 31.04 3.81 9.51 6.40 12.88 21.52 28.08
Vanilla RAGBM25 31.41 40.95 60.43 68.35 31.08 35.43 31.92 42.46 26.07 31.82 5.75 12.44 14.40 22.93 28.72 36.34
E5 40.72 52.23 64.42 72.58 45.85 51.48 32.78 43.19 23.44 29.46 6.04 12.25 24.80 32.12 34.01 41.90
Base adapterBM25 38.47 49.42 62.66 72.46 37.80 42.29 37.35 50.41 33.66 39.55 11.46 21.9829.60 41.2935.86 45.34
E5 43.46 54.60 63.90 74.05 51.69 55.74 37.31 49.90 33.45 39.45 11.83 22.07 30.40 40.48 38.86 48.04
RAG-GD (K=5)BM2540.22 50.01 66.13 74.28 37.81 42.19 38.99 51.14 34.15 40.04 12.54 22.6128.00 39.6136.83 45.70
E545.68 55.84 67.66 76.07 52.31 56.48 39.01 50.94 33.94 39.83 13.20 23.47 32.00 42.08 40.54 49.24
30 32 34 36 38
Avg EM across 7 datasetsVanilla RAGVanilla + few-shotBase + few-shotPrompt TuningHyperTuningBase adapterTT-SGDRAG-GD (K=5)
32.6132.1436.2234.5935.8534.1636.5436.71
(a) Method-family comparison.
1 5 10
Inner GD steps K31323334353637Avg EM across 7 datasets36.3536.71 36.68
32.2532.43 32.40Qwen 2.5-7B
RAG-GD (E5)
RAG-GD (BM25)
1 5 10
Inner GD steps K363738394041Avg EM across 7 datasets39.9340.54 40.59
36.9336.8336.96Llama 3.1-8B
RAG-GD (E5)
RAG-GD (BM25) (b) Robustness to inner GD depthK.
Figure 3:The gradient-derived update improves context-conditioned adaptation and is relatively
insensitive to K. (a)Average EM across seven QA benchmarks on Qwen 2.5-7B with E5 retrieval.
RAG-GD matches the TT-SGD reference using one predictor forward pass.(b)Average EM across
K∈ {1,5,10}on both backbones and retrievers. Solid lines use E5, and dashed lines use BM25.
HyperTuning, TT-SGD, and + few shot variants. Full per-method and per-benchmark results are in
Appendix I, and Algorithm 1 gives the deployment procedure.
The predicted update improves the base retrieval adapter.Across all backbone and retriever
configurations, RAG-GD improves average EM and F1 over Base adapter. This comparison is
controlled: both methods share the external retriever, retrieval cache, frozen backbone, and base
adapter Wret
0. The only difference is the predicted perturbation g∆W(C) , which isolates the effect of
adapting the generator-side retrieval interface from the support context.
The learned update transfers to held-out tasks.PopQA and Bamboogle are held out from training
for both Wret
0andgϕ. On PopQA, RAG-GD improves EM over Base adapter in every backbone
and retriever setting, with F1 close or improved in most cases. On Bamboogle, gains are strongest
on Qwen, while Llama with BM25 stays close to Base adapter. The transfer is consistent but not
uniform, suggesting that gϕlearns a reusable update rule for generator-side evidence use rather than
only fitting the meta-training tasks.
Gradient-update supervision matters.Figure 3a compares RAG-GD with context-conditioned
baselines on Qwen-2.5-7B with E5 retrieval. HyperTuning uses the same predictor architecture but
trains through downstream task loss rather than matching the autograd-defined update. RAG-GD
improves average EM and F1 over HyperTuning and Prompt tuning, indicating that the gradient-
derived target contributes beyond the predictor architecture.
8

Performance is largely insensitive to K.Figure 3b sweeps the inner-loop depth K∈ {1,5,10}
across both backbones and retrievers. A single amortized step already recovers most of the gain,
while additional steps bring only small and configuration-dependent changes. Thus, the K=5 setting
in Table 1 is representative. Full per-Kresults are in Appendix I.
Amortization approaches test-time adaptation at lower cost.Figure 4 shows the EM-cost
tradeoff for Qwen-2.5-7B with E5 retrieval. TT-SGD performs inner-loop backpropagation through
the 7B LLM at test time, while RAG-GD moves this computation into training and uses only one
forward pass through gϕat inference. As a result, RAG-GD reaches a similar average EM and F1
operating point at substantially lower per-query cost.
5.5 Discussion
102103
Per-query inference cost (ms, log scale)3234363840Avg EM across 7 datasets
test-time gradient zoneLower cost, comparable EM
Vanilla RAG
Vanilla + few-shotBase + few-shot
Prompt TuningHyperTuning
Base adapterTT-SGDRAG-GD (K=5)
Figure 4:EM-cost tradeoffon Qwen 2.5-7B
with E5 retrieval. Per-query cost is shown on
a log scale. The shaded region marks methods
that run inner GD at test time.These results complete the theory-to-practice arc.
The linear regime gives an exact gradient-descent
correspondence, while the nonlinear experiments
show where this correspondence becomes feature-
dependent. At LLM scale, we do not claim that
frozen RAG LLMs implement the linear equivalence
internally. Instead, the autograd-defined update pro-
vides a practical target for context-conditioned adap-
tation. Because RAG-GD and Base adapter share
the same retriever, retrieval cache, backbone, and
Wret
0, the gains isolate the contribution of the pre-
dicted update to the generator-side retrieval interface.
The held-out transfer and cost-performance tradeoff
suggest that gradient-supervised retrieval-interface
adaptation is a promising forward-only alternative to
test-time backpropagation for RAG.
6 Conclusion
We studied retrieval-augmented generation through an in-context optimization lens. In a controlled
linear setting, we showed that one linear self-attention layer can implement one gradient-descent
step on a unified linearized RAG loss covering projection-based and dot-product retrieval interfaces.
Empirical tests verified this construction in the exact regime and revealed a structured boundary
under nonlinear architectures and real regression data, where alignment becomes sensitive to feature
distribution. At LLM scale, we turned this view into a forward-only adaptation method by using
the autograd-defined K-step update to a generator-side Q/K/V LoRA interface as supervision for
a lightweight predictor. Across seven QA benchmarks, two retrievers, and two frozen backbones,
the predicted update improved a shared-adapter baseline, transferred to held-out domains, and was
largely insensitive to K. Overall, these results suggest that retrieved evidence can be treated not only
as external context, but also as a signal for context-induced adaptation in RAG.
7 Limitations
Our linear construction is an analytical starting point rather than a literal account of modern RAG
LLMs, and the nonlinear experiments show that the correspondence depends on architecture and
feature distribution. At LLM scale, we instantiate the view with a generator-side Q/K/V LoRA
interface while keeping the retriever and backbone fixed. Open questions remain about the inner-loop
optimizer, update parameterization, predictor capacity, scaling to larger backbones, and robustness
across broader retrieval settings. Future work should characterize when context-induced updates
improve evidence use, when they should be suppressed, and how uncertainty-aware gating can make
such updates robust to noisy retrieval.
9

References
[1]Kwangjun Ahn, Xiang Cheng, Hadi Daneshmand, and Suvrit Sra. Transformers learn to
implement preconditioned gradient descent for in-context learning. InNeurIPS, 2023.
[2]Ekin Akyürek, Dale Schuurmans, Jacob Andreas, Tengyu Ma, and Denny Zhou. What learning
algorithm is in-context learning? investigations with linear models. InICLR, 2023.
[3]Akari Asai, Zeqiu Wu, Yizhong Wang, Avirup Sil, and Hannaneh Hajishirzi. Self-rag: Learning
to retrieve, generate, and critique through self-reflection. InICLR, 2024.
[4]Yu Bai, Fan Chen, Huan Wang, Caiming Xiong, and Song Mei. Transformers as statisticians:
Provable in-context learning with in-context algorithm selection. InNeurIPS, 2023.
[5] C.M. Bishop.Pattern recognition and machine learning. Springer, 2006.
[6]Sebastian Borgeaud, Arthur Mensch, Jordan Hoffmann, Trevor Cai, Eliza Rutherford, Katie
Millican, George van den Driessche, Jean-Baptiste Lespiau, et al. Improving language models
by retrieving from trillions of tokens. InICML, 2022.
[7] William Jay Conover.Practical nonparametric statistics. Wiley, 1999.
[8]Damai Dai, Yutao Sun, Li Dong, Yaru Hao, Shuming Ma, Zhifang Sui, and Furu Wei. Why can
gpt learn in-context? language models implicitly perform gradient descent as meta-optimizers.
InACL, 2023.
[9]Shivam Garg, Dimitris Tsipras, Percy Liang, and Gregory Valiant. What can transformers learn
in-context? a case study of simple function classes. InNeurIPS, 2022.
[10] Khashayar Gatmiry, Nikunj Saunshi, Sashank J. Reddi, Stefanie Jegelka, and Sanjiv Kumar.
Can looped transformers learn to implement multi-step gradient descent for in-context learning?
InICML, 2024.
[11] Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian,
Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Alex Vaughan, Amy Yang,
Angela Fan, Anirudh Goyal, Anthony Hartshorn, Aobo Yang, Archi Mitra, Archie Sravankumar,
Artem Korenev, Arthur Hinsvark, Arun Rao, et al. The llama 3 herd of models.CoRR, 2024.
[12] Kelvin Guu, Kenton Lee, Zora Tung, Panupong Pasupat, and Ming-Wei Chang. Realm:
Retrieval-augmented language model pre-training. InICML, 2020.
[13] Xanh Ho, Anh-Khoa Duong Nguyen, Saku Sugawara, and Akiko Aizawa. Constructing a
multi-hop qa dataset for comprehensive evaluation of reasoning steps. InCOLING, 2020.
[14] Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang,
Lu Wang, and Weizhu Chen. Lora: Low-rank adaptation of large language models. InICLR,
2022.
[15] Jiatan Huang, Mingchen Li, Zonghai Yao, Dawei Li, Yuxin Zhang, Zhichao Yang, Yongkang
Xiao, Feiyun Ouyang, Xiaohan Li, Shuo Han, and Hong Yu. Ritek: A dataset for large language
models complex reasoning over textual knowledge graphs in medicine. InACL, 2026.
[16] Gautier Izacard and Edouard Grave. Leveraging passage retrieval with generative models for
open domain question answering. InEACL, 2021.
[17] Jiajie Jin, Yutao Zhu, Zhicheng Dou, Guanting Dong, Xinyu Yang, Chenghao Zhang, Tong Zhao,
Zhao Yang, and Ji-Rong Wen. Flashrag: A modular toolkit for efficient retrieval-augmented
generation research. InWWW, 2025.
[18] Mandar Joshi, Eunsol Choi, Daniel S. Weld, and Luke Zettlemoyer. Triviaqa: A large scale
distantly supervised challenge dataset for reading comprehension. InACL, 2017.
[19] Vladimir Karpukhin, Barlas O ˘guz, Sewon Min, Patrick Lewis, Ledell Wu, Sergey Edunov,
Danqi Chen, and Wen tau Yih. Dense passage retrieval for open-domain question answering. In
EMNLP, 2020.
10

[20] Donggyun Kim, Chanwoo Kim, and Seunghoon Hong. Hyperflow: Gradient-free emulation of
few-shot fine-tuning.CoRR, 2025.
[21] Tom Kwiatkowski, Jennimaria Palomaki, Olivia Redfield, Michael Collins, Ankur Parikh, Chris
Alberti, Danielle Epstein, Illia Polosukhin, Jacob Devlin, Kenton Lee, Kristina Toutanova, Llion
Jones, Matthew Kelcey, Ming-Wei Chang, Andrew M. Dai, Jakob Uszkoreit, Quoc Le, and Slav
Petrov. Natural questions: A benchmark for question answering research. InTACL, 2019.
[22] Brian Lester, Rami Al-Rfou, and Noah Constant. The power of scale for parameter-efficient
prompt tuning. InEMNLP, 2021.
[23] Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman
Goyal, Heinrich Küttler, Mike Lewis, Wen tau Yih, Tim Rocktäschel, Sebastian Riedel, and
Douwe Kiela. Retrieval-augmented generation for knowledge-intensive nlp tasks. InNeurIPS,
2020.
[24] Mingchen Li, Zaifu Zhan, Han Yang, Yongkang Xiao, Jiatan Huang, and Rui Zhang. Bench-
marking retrieval-augmented large language models in biomedical nlp: Application, robustness,
and self-awareness.Science Advances, 2025.
[25] Arvind Mahankali, Tatsunori B. Hashimoto, and Tengyu Ma. One step of gradient descent is
provably the optimal in-context learner with one layer of linear self-attention.CoRR, 2023.
[26] Alex Mallen, Akari Asai, Victor Zhong, Rajarshi Das, Daniel Khashabi, and Hannaneh Ha-
jishirzi. When not to trust language models: Investigating effectiveness of parametric and
non-parametric memories. InACL, 2023.
[27] Eric Mitchell, Charles Lin, Antoine Bosselut, Chelsea Finn, and Christopher D. Manning. Fast
model editing at scale. InICLR, 2022.
[28] Marius Mosbach, Tiago Pimentel, Shauli Ravfogel, Dietrich Klakow, and Yanai Elazar. Few-
shot fine-tuning vs. in-context learning: A fair comparison and evaluation. InACL, 2023.
[29] Jason Phang, Yi Mao, Pengcheng He, and Weizhu Chen. Hypertuning: Toward adapting large
language models without back-propagation. InICML, 2023.
[30] Ofir Press, Muru Zhang, Sewon Min, Ludwig Schmidt, Noah A. Smith, and Mike Lewis.
Measuring and narrowing the compositionality gap in language models. InEMNLP, 2023.
[31] Qwen, :, An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu,
Chengyuan Li, Dayiheng Liu, Fei Huang, Haoran Wei, Huan Lin, Jian Yang, Jianhong Tu,
Jianwei Zhang, Jianxin Yang, Jiaxi Yang, Jingren Zhou, Junyang Lin, Kai Dang, Keming Lu,
Keqin Bao, Kexin Yang, Le Yu, et al. Qwen2.5 technical report.CoRR, 2024.
[32] Pranav Rajpurkar, Jian Zhang, Konstantin Lopyrev, and Percy Liang. Squad: 100,000+ questions
for machine comprehension of text. InEMNLP, 2016.
[33] Ori Ram, Yoav Levine, Itay Dalmedigos, Dor Muhlgay, Amnon Shashua, Kevin Leyton-Brown,
and Yoav Shoham. In-context retrieval-augmented language models. InTACL, 2023.
[34] Ruifeng Ren and Yong Liu. Towards understanding how transformers learn in-context through
a representation learning lens. InNeurIPS, 2024.
[35] Zhaiming Shen, Alexander Hsu, Rongjie Lai, and Wenjing Liao. Understanding in-context
learning on structured manifolds: Bridging attention to kernel methods. InICLR, 2026.
[36] Yu Sun, Xiaolong Wang, Zhuang Liu, John Miller, Alexei A. Efros, and Moritz Hardt. Test-time
training with self-supervision for generalization under distribution shifts. InICML, 2020.
[37] Jihoon Tack, Jaehyung Kim, Eric Mitchell, Jinwoo Shin, Yee Whye Teh, and Jonathan Richard
Schwarz. Online adaptation of language models with a memory of amortized contexts. In
NeurIPS, 2024.
[38] Harsh Trivedi, Niranjan Balasubramanian, Tushar Khot, and Ashish Sabharwal. Musique:
Multihop questions via single-hop question composition. InTACL, 2022.
11

[39] Laurens van der Maaten and Geoffrey Hinton. Visualizing data using t-sne.JMLR, 2008.
[40] Max Vladymyrov, Johannes von Oswald, Mark Sandler, and Rong Ge. Linear transformers are
versatile in-context learners. InICML, 2024.
[41] Johannes von Oswald, Eyvind Niklasson, Ettore Randazzo, João Sacramento, Alexander Mord-
vintsev, Andrey Zhmoginov, and Max Vladymyrov. Transformers learn in-context by gradient
descent. InICML, 2023.
[42] Fan Wang, Chuan Lin, Yang Cao, and Yu Kang. Benchmarking general-purpose in-context
learning.CoRR, 2024.
[43] Liang Wang, Nan Yang, Xiaolong Huang, Binxing Jiao, Linjun Yang, Daxin Jiang, Rangan
Majumder, and Furu Wei. Text embeddings by weakly-supervised contrastive pre-training.
CoRR, 2024.
[44] Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio, William W. Cohen, Ruslan Salakhut-
dinov, and Christopher D. Manning. Hotpotqa: A dataset for diverse, explainable multi-hop
question answering. InEMNLP, 2018.
[45] Qinggang Zhang, Shengyuan Chen, Yuanchen Bei, Zheng Yuan, Huachi Zhou, Zijin Hong, Hao
Chen, Yilin Xiao, Chuang Zhou, Junnan Dong, Yi Chang, and Xiao Huang. A survey of graph
retrieval-augmented generation for customized large language models.CoRR, 2025.
[46] Yedi Zhang, Aaditya K. Singh, Peter E. Latham, and Andrew Saxe. Training dynamics of
in-context learning in linear attention. InICML, 2025.
Broader Impacts
The method we propose lowers the cost of adapting a deployed large language model to a new task or
domain. At inference, the LLM still runs a forward pass to generate the answer, but no backward pass
through the LLM is required: adapting to the context costs only a single small forward pass through a
context-conditional weight predictor. If adopted at scale, this could reduce the energy footprint of
test-time adaptation pipelines for retrieval-augmented systems. The flip side is that lower adaptation
cost could also accelerate the deployment of models in domains for which the underlying LLM has
not been carefully evaluated, including settings where retrieval can amplify biases in the corpus. We
encourage practitioners adopting this style of adaptation to retain the same evaluation discipline that
would apply to a fine-tuned model. All datasets used in our evaluation are publicly available QA and
regression benchmarks that do not contain personally identifiable or sensitive information, and our
work does not raise additional ethical concerns beyond those discussed above.
Reproducibility Statement
All datasets used in our main evaluation are publicly available QA benchmarks (NQ, TriviaQA,
PopQA, HotpotQA, 2WikiMultiHopQA, MuSiQue, Bamboogle). The supplementary regression
datasets discussed in the appendix are also public. Preprocessing steps, dataset splits, and the training
pools used for the reference retrieval adapter Wret
0and the predictor gϕare documented in the
appendix.
Our implementation uses PyTorch with the Hugging Face Transformers library on top of frozen
Qwen 2.5-7B-Instruct [ 31] and Llama 3.1-8B-Instruct [ 11] backbones. Hyperparameters for the
reference retrieval adapter Wret
0, the predictor gϕ, and the inner-loop SGD target ( η,K, number of
demonstrations N) are listed in Appendix G. All experiments were conducted on NVIDIA A100
GPUs.
A Proof of Lemma 1
Statement.Given a 1-head linear-attention layer and tokens ej= (xj
1, xj
2, yj)forj= 1, . . . , N ,
we construct key, query, and value matrices WK,WQ,WVand a projection Psuch that one linear
12

self-attention update on each ejmatches one gradient-descent step on the unified RAG loss of
Section 3. The update modifies only they-coordinate of each token:
ej←ej+ 
0,0,∆W 1xj
1+ ∆W 2xj
2
=e j+P V K⊤qj,(1)
where∆W 1and∆W 2are the gradient-step updates of Eq. 4 in the main text.
Setup.We are given Ncontext tokens, each of the form ei= (xi
1, xi
2, yi)corresponding to one
training pair, plus a query token eN+1= (xN+1
1, xN+1
2,0)at position N+1 . The model is asked to
predict the updatedy-value at positionN+1.
Step 1: expand the post-update prediction.Writing the post-update prediction y′as the original
prediction plus the contributions of∆W 1and∆W 2,
y′=W′
1x1+W′
2x2
= (W 1+ ∆W 1)x1+ (W 2+ ∆W 2)x2
=W 1x1+W 2x2+ ∆W 1x1+ ∆W 2x2.(2)
Step 2: gradient step on the unified RAG loss.Under the squared loss L(W 1, W2) =
1
2NPN
i=1∥W1xi
1+W 2xi
2−yi∥2, one gradient step with learning rateηyields
∆W 1=−η∇ W1L=−η
NNX
i=1 
W1xi
1+W 2xi
2−yi
(xi
1)⊤,(3)
∆W 2=−η∇ W2L=−η
NNX
i=1 
W1xi
1+W 2xi
2−yi
(xi
2)⊤,(4)
∆y= ∆W 1x1+ ∆W 2x2.(5)
Step 3: rewrite ∆yas a sum of outer-product contractions.Substituting Eqs. 3–4 into Eq. 5 and
evaluating at the query tokenj,
∆y=−η
NNX
i=1 
W1xi
1+W 2xi
2−yi
(xi
1)⊤xj
1−η
NNX
i=1 
W1xi
1+W 2xi
2−yi
(xi
2)⊤xj
2.(6)
Equivalently, the update applied to the token at positionjis

xj
1
xj
2
yj
←
xj
1
xj
2
yj
+ 0
0
∆y!
,with 0
0
∆y!
= 0
0
∆W 1x1+ ∆W 2x2!
.(7)
Step 4: cast the update as a linear self-attention output.Using the identity a b⊤c= (a⊗b⊤)c
and grouping terms, Eq. 6 can be written as
 0
0
∆y!
=−η
NNX
i=1
0
0
W1xi
1+W 2xi
2−yi

| {z }
value vectorv i⊗ 
xi
1xi
20
|{z}
key vectork⊤
i
xj
1
xj
2
0

|{z}
query vectorq j.(8)
Each factor in Eq. 8 can be obtained by applying a fixed linear projection to the token ei= (xi
1, xi
2, yi)
orej:
vi= 0 0 0
0 0 0
W1W2−Iy!
| {z }
WVei, k i= Ix0 0
0I x0
0 0 0!
|{z }
WKei, q j= Ix0 0
0I x0
0 0 0!
|{z }
WQej.(9)
13

Step 5: explicit construction.Combining Eqs. 8 and 9, the gradient step of Eqs. 3–4 is realized as
a linear self-attention update with the closed-form projections

xj
1
xj
2
yj
←
xj
1
xj
2
yj
−η
NPN
i=1  0 0 0
0 0 0
W1W2−Iy!
| {z }
WV
xi
1
xi
2
yi
!
⊗  Ix0 0
0I x0
0 0 0!
|{z }
WK
xi
1
xi
2
yi
!⊤  Ix0 0
0I x0
0 0 0!
|{z }
WQ
xj
1
xj
2
yj
!
(10)
The projection Pis taken to be the identity on the y-coordinate, so that the value contribution lands
in the y-slot of ej. Comparing the right-hand side of Eq. 10 to the gradient step in Eqs. 3–4, the two
sides agree term by term. One linear self-attention update therefore reproduces one gradient-descent
step on the unified RAG loss, as claimed.
B Linear RAG: derivation of the unified retriever formulation
Main function.
y= (W q, Wz)xqPn
i=1(Wexq)⊤(Wedi)di
=W qxq+W znX
i=1(Wexq)⊤(Wedi)di.(11)
Define Mand rewrite the similarity.We adopt the shared-encoder simplification, where the same
linear encoder Wemaps both queries and documents into the retrieval space. The general DPR
formulation [ 19] permits separate WqandWd, in which case the analysis below carries through with
M=W⊤
qWdbutMneed not be symmetric. Defining
M≜W⊤
eWe⇒(W exq)⊤(Wedi) =x⊤
qW⊤
eWedi=x⊤
qMdi.(12)
Hence,
y=W qxq+W znX
i=1(x⊤
qMdi)di.(13)
Converting “scalar ×vector” into “matrix ×vector.”Note that x⊤
qMdiis a scalar, and the
following identity holds:
(x⊤
qMdi)di=di(d⊤
iM⊤xq) = (d id⊤
i)M⊤xq.(14)
Therefore,
nX
i=1(x⊤
qMdi)di=nX
i=1(did⊤
i)M⊤xq=nX
i=1did⊤
i
M⊤xq.(15)
Define the document second-moment matrixD.
D≜nX
i=1did⊤
i⇒nX
i=1(x⊤
qMdi)di=DM⊤xq.(16)
Substituting back intoy.
y=W qxq+W zDM⊤xq.(17)
ThenM=W⊤
eWeis symmetric, i.e.,M⊤=M. Thus the expression simplifies to
y=W qxq+W zDMx q.(18)
The right-hand side is grouped into an equivalent linear mapping:
y= (W q+W zDM)x q.(19)
C Linear-regime equivalence: full setup and additional figures
This appendix supplements Section 4.1 of the main paper with a full description of the synthetic-
regression setup and the additional robustness and stacked-layer figures referenced there.
14

C.1 Setup details
Tokens.Each token concatenates an input vector, a retrieval-derived feature, and a target,
ei= (x i, zi, yi), i= 1, . . . , N,(20)
where Nis the number of in-context examples for a single task τ. The auxiliary slot ziinstantiates
the unified RAG view of Section 3. Under the linear-projection retriever, ziis a document-derived
feature; under the dot-product retriever, zi=xiand the document information is injected through the
keys and values rather than through the token (see “dot-product injection” below).
Pre-training objective.We train an LSA layer parameterized by θto minimize the expected
squared prediction error across tasks:
L(θ) =1
BBX
τ=1ˆyθ 
{eτ,i}N
i=1, eτ,N+1
−yτ,N+12,(21)
where the query token at position N+ 1 iseτ,N+1 = (x test, ztest,0)andyτ,N+1 is its target. The
objective is optimized with minibatch SGD over a fresh batch of tasks at each iteration. We denote
the parameters at convergence byθ∗.
Synthetic data.Following [ 9,41], we generate each task τfrom a teacher with weights Wτ∼
N(0, I) . Inputs are drawn from xτ,i∼ U(−1,1)nIand targets are constructed as yτ,i=W1
τx1
τ,i+
W2
τx2
τ,i. We set N=n I= 10 with output dimension 1, and we sweep the document count
k∈ {2,5,10,25}.
Dot-product injection.Under the linear-projection retriever the document is included in the token,
ei= (x i,D, y i), and the LSA layer can learn to select relevant documents during pre-training.
Under the dot-product retriever the document is not concatenated into the token; instead, document
information is injected directly into the key and value matrices,
K=
Kctx
hd
, V=
Vctx
hd
,(22)
where Kctx, Vctxare the contextual key/value rows and hd=f(D)∈RB×dim(x)is a fixed projection
of the document setD={d 1, . . . , d n}into the input dimension.
Constructed reference predictor.The trained LSA layer (parameters θ∗) is compared against
aconstructedpredictor that realizes one gradient-descent step on the unified RAG loss exactly.
Following the construction of Eq. 10, we set the value, key, and query projections so that one LSA
update reproduces the gradient step. For the linear-projection retriever, W1andW2inside WVare
initialized to zero, following [ 41]. For the dot-product retriever, W2=W z P
idid⊤
i
M⊤, with
Wz∈Rdy×ddandM∈Rdd×ddsampled independently from N(0, σ2), and document features
C∼ U(−1
2,1
2)k×dd. The inner-loop learning rate ηfor the constructed predictor is chosen by line
search to minimize the constructed model’s loss over 104training tasks. We write the resulting
predictionˆy θ,rag(xtest).
Evaluation metrics.On Tval= 104held-out validation tasks, following [ 41], we report the
mean of three quantities between the trained and constructed predictors: (i) the prediction differenceˆyθ∗(xτ,test)−ˆy θ,rag(xτ,test)
2; (ii) the cosine similarity between the input-sensitivities ∂ˆyθ,rag/∂x test
and∂ˆy θ∗/∂x test; and (iii) the correspondingℓ 2sensitivity difference.
C.2 Robustness to distribution shift and document count
To probe whether the trained LSA layer captures a generalizable update rule rather than memorizing
the training distribution, we vary two factors at test time. First, we sweep the document count
n∈ {2,5,10,25} and recompute the comparison; second, we sample test inputs from U(−α, α)nI
withα∈ {0.5,1,1.5,2} while training is fixed at α= 1 . Figure 5 reports the resulting loss curves.
With the linear-projection retriever, the absolute loss rises with document count (the projected
15

2 5 10 25
Num document246810LossRAG
Trained TF
2 5 10 25
Num document0.00.20.40.60.81.0LossRAG
Trained TF
0.5 1 1.5 2
where xU(,)
02505007501000LossRAG
Trained TF
0.5 1 1.5 2
where xU(,)
02004006008001000LossRAG
Trained TFFigure 5: Robustness of the single-layer agreement of Section 4.1.Left two:loss as a function
of document count for the linear-projection (left) and dot-product (centre-left) retrievers.Right
two:loss under input distribution shift, with test inputs drawn from U(−α, α)nIfor varying α, for
the linear-projection (centre-right) and dot-product (right) retrievers. The trained Transformer, the
constructed gradient-descent predictor, and their interpolation track each other closely in all settings.
0 2500 5000 7500
Training steps0.00.20.40.6loss
Docs=2
Docs=5
Docs=10
Docs=25
0 2500 5000 7500
Training steps0.40.50.60.70.80.91.0Cosine simDocs=2
Docs=5
Docs=10
Docs=25
0 2500 5000 7500
Training steps0.00.20.40.60.81.0Model diffDocs=2
Docs=5
Docs=10
Docs=25
0 2500 5000 7500
Training steps0.20.40.60.8Preds diffDocs=2
Docs=5
Docs=10
Docs=25
0 2500 5000 7500
Training steps0.00.20.40.6loss
Docs=2
Docs=5
Docs=10
Docs=25
0 2500 5000 7500
Training steps0.40.50.60.70.80.91.0Cosine simDocs=2
Docs=5
Docs=10
Docs=25
0 2500 5000 7500
Training steps0.00.51.01.52.0Model diffDocs=2
Docs=5
Docs=10
Docs=25
0 2500 5000 7500
Training steps0.20.40.60.81.0Preds diffDocs=2
Docs=5
Docs=10
Docs=25
Figure 6: Stacked-layer agreement under the dot-product retriever (Section 4.1).Top row:2-layer
model.Bottom row:5-layer model. Columns: (a) loss difference between trained Transformer and
constructed gradient-descent predictor, (b) sensitivity cosine, (c) model difference, (d) prediction
difference. Agreement remains close at both depths; the small residual gap at Docs= 25 in the
2-layer setting shrinks as depth increases to 5.
document features carry more variance) but the LSA layer follows the gradient predictor in lockstep.
With the dot-product retriever, where the document information enters through the second-moment
matrixP
idid⊤
i, the loss is largely insensitive to document count. The dot-product variant is also
computationally cheaper, since no per-document projection is required.
C.3 Stacked-layer agreement under the dot-product retriever
Figure 6 reports the dot-product variant at depths 2 and 5. The loss differences between the trained
Transformer and the constructed predictor remain small across document counts, and the prediction
differences converge to similar values. The number of retrieved documents has a depth-dependent
effect on the residual: at depth 2, the prediction difference is smaller for Docs= 2 than for Docs= 25 ,
but this gap narrows at depth 5. The corresponding analysis for the linear-projection retriever is
reported in Appendix E.
D Normalization analysis: per-dataset extended results
This appendix supplements Section 4.3 of the main paper. Section 4.3 reports the headline result
on Bike Sharing and California Housing; here we cover the remaining two datasets, Predict Calorie
Expenditure and Wine Quality. Datasets and normalization methods are as defined in Section 4.3 and
Appendix F. The training set is used as the retrieval corpus and is normalized with Z-score throughout;
16

0 2000 4000
Training steps0.000.020.040.060.080.10loss
Z-Score
Min-Max
Rank
T anh
0 2000 4000
Training steps0.800.850.900.951.00Cosine simZ-Score
Min-Max
Rank
T anh
0 2000 4000 6000 8000
Training steps0.20.30.40.5Model diffZ-Score
Min-Max
Rank
T anh
0 2000 4000 6000
Training steps0.000.050.100.150.200.250.30Preds diffZ-Score
Min-Max
Rank
T anh
0 2000 4000
Training steps0.15
0.10
0.05
0.000.050.10loss
Z-Score
Min-Max
Rank
T anh
0 1000 2000 3000 4000
Training steps0.20.40.60.81.0Cosine simZ-Score
Min-Max
Rank
T anh
0 2000 4000 6000 8000
Training steps0.30.40.50.60.70.8Model diffZ-Score
Min-Max
Rank
T anh
0 2000 4000 6000
Training steps0.10.20.30.40.5Preds diffZ-Score
Min-Max
Rank
T anhFigure 7: Per-dataset normalization results.Top row:Predict Calorie Expenditure.Bottom row:
Wine Quality. Each column reports a different evaluation metric: loss difference with the trained
Transformer, training loss of RAG, sensitivity cosine, model difference, and prediction difference.
The four normalization schemes (Z-score, Min–Max, rank-based, Tanh) are overlaid within each
panel.
0 20000 40000
Training steps0.00.10.20.30.40.5Loss
Docs=2
Docs=5
Docs=10
Docs=15
0 20000 40000
Training steps0.40.50.60.70.80.91.0Cosine simDocs=2
Docs=5
Docs=10
Docs=15
0 20000 40000
Training steps0.51.01.52.02.53.0Model diffDocs=2
Docs=5
Docs=10
Docs=15
0 5000 10000
Training steps0.10.20.30.40.50.60.7Preds diffDocs=2
Docs=5
Docs=10
Docs=15
0 20000 40000
Training steps0.00.20.40.6Loss
Docs=2
Docs=5
Docs=10
Docs=15
0 20000 40000
Training steps0.800.850.900.951.00Cosine simDocs=2
Docs=5
Docs=10
Docs=15
0 20000 40000
Training steps0.51.01.52.02.53.0Model diffDocs=2
Docs=5
Docs=10
Docs=15
0 5000 10000
Training steps0.00.20.40.6Preds diffDocs=2
Docs=5
Docs=10
Docs=15
Figure 8: Stacked-layer agreement under the projection-based retriever.Top row:2-layer model.Bot-
tom row:5-layer model. Columns: (a) loss difference between trained Transformer and constructed
gradient-descent predictor, (b) sensitivity cosine, (c) model difference, (d) prediction difference.
only the input-side normalization is varied, between Z-score [ 5], Min–Max [ 5], rank-based [ 7], and
Tanh [39].
On Predict Calorie Expenditure, the trained Transformer continues to track the gradient-descent
predictor closely, mirroring the alignment seen on Bike Sharing in Section 4.3. Wine Quality is the
harder case. Under Min–Max normalization, a few outliers dominate the scaling and compress most of
the samples near zero. Two effects follow. The sensitivity cosine drops because the sensitivity vectors
diverge from those of the gradient-descent predictor. The prediction difference also fluctuates more
strongly, indicating instability in the alignment between RAG and ICL dynamics under heavy-tailed
feature distributions. This is consistent with the California Housing pattern in Section 4.3: when
retrieval-derived dot products are dominated by skewed features, the linear correspondence becomes
less predictive.
17

E Stacked-layer agreement under the projection-based retriever
This appendix complements the dot-product analysis of Appendix C with the projection-based
retriever. Under this interface, the retrieved documents are concatenated with the input tokens,
ei= (x i,D, y i), so each per-document feature is processed jointly with the query through every
stacked layer. As the document count grows, the variance of the per-token features grows with
it, which amplifies the discrepancy between the trained Transformer and the constructed gradient-
descent predictor at any fixed depth. Stacking layers reduces this discrepancy: at depth 5, the loss
and prediction differences are uniformly smaller across document counts than at depth 2, and the
sensitivity cosine is closer to 1.
Whereas the dot-product sweep in Appendix C reports results at 2, 5, 10, and 25 documents, the
projection-based sweep is restricted to 2, 5, 10, and 15. The compute cost of stacking concatenated-
document tokens grows substantially with retrieval size, and the growing-residual trend is already
clear at 15 documents, so the 25-document run is omitted.
F Dataset details
QA benchmarks (main evaluation).The seven question-answering benchmarks used in the main
evaluation are all publicly available:
•Natural Questions (NQ)[21]: open-domain factoid QA over Wikipedia.
•TriviaQA[18]: large-scale trivia question answering with evidence documents.
•PopQA[26]: popularity-stratified entity-centric QA (held out from training).
•HotpotQA[44]: multi-hop QA with comparison and bridge questions.
•2WikiMultiHopQA[13]: multi-hop questions grounded in Wikipedia article pairs.
•MuSiQue[38]: compositional multi-hop QA constructed by composing single-hop pairs.
•Bamboogle[30]: small multi-hop benchmark on long-tail entities (held out from training).
For each benchmark we use the standard FlashRAG [ 17] corpus and retrieval splits. NQ is used to
train the source retrieval adapter Wret
0. NQ, TriviaQA, HotpotQA, 2WikiMultiHopQA, and MuSiQue
are used to meta-train the predictor gϕ. PopQA and Bamboogle are held out from both adapter
training and predictor meta-training, and are used only for evaluation.
Synthetic and tabular regression datasets (linear-attention sanity check and normalization
analysis).
•California Housing: Given eight features, [’MedInc’, ’HouseAge’, ’AveRooms’,
’AveBedrms’, ’Population’, ’AveOccup’, ’Latitude’, ’Longitude’] , the task
is to predict MedHouseVal . The dataset is split into 16,640 training samples and 2,000 test
samples.
•Bike Sharing: Using the features [’season’, ’yr’, ’mnth’, ’hr’, ’holiday’,
’weekday’, ’workingday’, ’weathersit’, ’temp’, ’atemp’, ’hum’,
’windspeed’, ’casual’, ’registered’] , the task is to predict count . The dataset
contains 15,641 training samples and 1,738 test samples.
•Wine Quality: Given eleven physicochemical features, [fixed acidity, volatile
acidity, citric acid, residual sugar, chlorides, free sulfur dioxide,
total sulfur dioxide, density, pH, sulphates, alcohol] the task is to predict
the wine quality (a sensory score ranging from 0 to 10). The dataset is split into 4,408
training samples and 490 test samples.
•Predict Calorie Expenditure: Using the features [Gender, Age, Height, Weight,
Duration, Heart_Rate, Body_Temp] , the task is to predict the number of Calories
expended. The dataset is split into 13,500 training samples and 1,540 test samples.
18

G Implementation Details
Source retrieval adapter Wret
0.We implement the generator-side retrieval interface with LoRA
modules on the {q, k, v} projections of every transformer block. The LoRA rank is 16, with α=32
and dropout 0. The LLM backbone is kept frozen and loaded in 4-bit NF4 precision. We train
Wret
0with AdamW using learning rate 10−4, weight decay 0.01, gradient clipping 1.0, and gradient
accumulation 4for3,000 steps on RAG-formatted examples from the NQ training split. Each example
is paired with the top- Kret=5retrieved documents from the fixed external retriever. For each retriever
setting, we use the corresponding fixed retrieval cache and keep this cache unchanged across methods.
This adapter serves only as a generator-side evidence-use interface: it does not select documents, but
modulates how the frozen generator uses the retrieved evidence.
Predictor gϕ.The predictor maps a support context C={(x i, Di, yi)}N
i=1withN=3 demonstra-
tions to a context-conditioned update for the generator-side retrieval interface. Each demonstration is
formatted by concatenating the question, retrieved documents, and gold answer, and is encoded by
the frozen LLM equipped with Wret
0. We take the EOS hidden state hiof each demonstration and
aggregate the support context by mean pooling,
¯h(C) =1
NNX
i=1hi.
The pooled representation is passed to a two-layer MLP encoder with hidden dimension 256and
output dimension 64, followed by per-layer and per-projection update heads for the {q, k, v} LoRA
modules. The update heads output low-rank perturbations with the same LoRA rank as Wret
0, so
thatgϕ(C)has the same parameter shape as the base retrieval adapter. We train gϕwith AdamW
using learning rate 5×10−4, weight decay 0.01, gradient clipping 1.0, and gradient accumulation
4for3,000 steps. The predictor is meta-trained on support contexts sampled from NQ, TriviaQA,
HotpotQA, 2WikiMultiHopQA, and MuSiQue. PopQA and Bamboogle are excluded from both
adapter training and predictor meta-training, and are used only for held-out evaluation.
Matching objective.The predictor is trained to match the autograd-defined inner-GD target for
each layer and projection type. The matching loss uses a cosine term for update direction and a
log-magnitude term for update scale, as defined in Section 5.2. We set the magnitude weight to λ=0.1
throughout. No downstream answer loss is applied when training gϕfor the main RAG-GD results.
Inner GD target.For each support context, the supervision target is computed by running Ksteps
of SGD on the N=3 demonstrations, starting from Wret
0. The inner-loop learning rate is η=10−2,
and we evaluate K∈ {1,5,10} . Gradients are taken only with respect to the LoRA parameters of
the generator-side retrieval interface; the external retriever and the LLM backbone remain fixed.
Compute.All experiments are run on NVIDIA A100 80GB GPUs. A single training run for Wret
0
takes approximately 50–60minutes. Training gϕtakes approximately 1.5hours for K=1 ,4hours
forK=5, and7hours forK=10. The full result table requires approximately50A100-hours.
H Algorithm and inference procedure for RAG-GD
Algorithm 1 summarises the deployment-time procedure of RAG-GD: build a RAG-formatted
support context, predict the retrieval-interface update with a single forward pass through gϕ, and
generate with the perturbed interface. No backward pass through the LLM is required at deployment.
I Full per-method comparison on QA benchmarks
We split the per-benchmark numbers into two tables. Table 2 reports the methods that we ran on
bothQwen-2.5-7B-Instruct and Llama-3.1-8B-Instruct: Query Only, Vanilla RAG, Base adapter, and
RAG-GD at K∈ {1,5,10} . Table 3 reports the additional context-conditioned baselines that we ran
only on Qwen due to compute constraints: Vanilla RAG + few shot, Base adapter + few shot, Prompt
tuning, HyperTuning, and TT-SGD at K=5 . Together they complement the slim main-text Table 1 and
19

Algorithm 1:RAG-GD: Forward-Only Retrieval-Interface Adaptation
Input:External retrieverR, frozen LLMf, source adapterWret
0, predictorg ϕ, support sizeN,
queryq
Output:Generated answerˆy q
1// Phase 1: Build RAG-formatted support context
2C← ∅
3fori= 1, . . . , Ndo
4Sample support pair(x i, yi)from the task support pool
5D i←R(x i)// retrieve top-kdocuments
6C←C∪ {(x i,Di, yi)}
7// Phase 2: Predict retrieval-interface update
8for(x i,Di, yi)∈Cdo
9h i←fWret
0(xi,Di, yi)EOS
10¯h(C)←1
NPN
i=1hi
11g∆W(C)←g ϕ(¯h(C))
12// Phase 3: Generate with adapted interface
13D q←R(q)
14ˆyq←fWret
0+g∆W(C)(q,D q)
15returnˆy q
the per-family aggregates in Figures 3 and 4.HyperTuning[ 29] uses the same predictor architecture
as RAG-GD but supervises against a downstream task loss instead of the SGD-update target, so
the contrast between the HyperTuning rows in Table 3 and the RAG-GD ( K=5 ) rows for Qwen in
Table 2 isolates the choice of supervision signal.TT-SGDperforms K=5 inner gradient-descent
steps at test time and serves as a reference for what test-time gradient adaptation would achieve at the
same backbone and retriever.+ few shotvariants concatenate the support demonstrations into the
prompt under the corresponding base configuration.
20

Table 2: Methods run on both backbones. Per-benchmark exact match (EM) and F1 for Query Only,
Vanilla RAG, Base adapter, and RAG-GD atK∈ {1,5,10}.
Method RetrieverSingle-Hop QA Multi-Hop QA Avg.
NQ TriviaQA PopQA HotpotQA 2Wiki MuSiQue Bamboogle
EM F1 EM F1 EM F1 EM F1 EM F1 EM F1 EM F1 EM F1
Qwen-2.5-7B
Query Only – 15.95 24.28 43.33 49.51 16.02 19.76 18.40 25.39 23.91 28.12 3.80 10.57 11.20 18.02 18.94 25.09
Vanilla RAGBM25 27.61 36.66 58.24 65.77 28.84 33.18 31.28 41.25 27.87 33.24 5.87 13.05 10.40 21.16 27.16 34.90
E5 39.16 50.03 62.99 70.80 44.03 50.21 32.45 42.21 25.48 31.43 5.79 12.77 18.40 26.74 32.61 40.60
Base adapterBM25 32.57 41.45 60.11 67.93 31.55 35.63 32.31 43.45 28.22 34.13 6.41 15.46 16.80 26.01 29.71 37.72
E5 41.77 51.22 63.31 71.62 47.05 51.82 33.78 44.64 27.89 33.97 6.95 15.76 18.40 28.78 34.16 42.54
RAG-GD (K=1)BM25 33.88 43.20 63.02 70.66 33.25 37.73 35.25 46.79 28.84 34.59 9.14 19.62 22.40 33.06 32.25 40.81
E5 42.49 52.32 65.61 73.51 48.41 52.93 35.34 46.94 29.66 35.50 8.93 19.17 24.00 34.18 36.35 44.94
RAG-GD (K=5)BM25 34.46 43.54 63.27 70.69 33.22 37.69 35.54 47.14 28.86 34.48 9.26 19.73 22.40 32.85 32.43 40.87
E5 42.91 52.71 65.98 73.60 48.12 52.61 35.54 47.00 29.67 35.47 9.14 19.26 25.60 35.12 36.71 45.11
RAG-GD (K=10)BM25 34.57 43.72 63.26 70.57 33.11 37.57 35.45 47.07 28.64 34.32 9.35 19.58 22.40 31.92 32.40 40.68
E5 42.46 52.14 65.93 73.49 47.96 52.43 35.69 47.05 29.43 35.14 8.90 19.09 26.40 34.67 36.68 44.86
Llama-3.1-8B
Query Only – 22.46 32.51 52.67 59.79 20.63 25.09 18.31 25.71 26.39 31.04 3.81 9.51 6.40 12.88 21.52 28.08
Vanilla RAGBM25 31.41 40.95 60.43 68.35 31.08 35.43 31.92 42.46 26.07 31.82 5.75 12.44 14.40 22.93 28.72 36.34
E5 40.72 52.23 64.42 72.58 45.85 51.48 32.78 43.19 23.44 29.46 6.04 12.25 24.80 32.12 34.01 41.90
Base adapterBM25 38.47 49.42 62.66 72.46 37.80 42.29 37.35 50.41 33.66 39.55 11.46 21.98 29.60 41.29 35.86 45.34
E5 43.46 54.60 63.90 74.05 51.69 55.74 37.31 49.90 33.45 39.45 11.83 22.07 30.40 40.48 38.86 48.04
RAG-GD (K=1)BM25 39.58 49.60 66.13 74.13 37.89 42.38 38.53 50.67 33.72 39.61 12.25 22.48 30.40 41.69 36.93 45.79
E5 45.10 55.19 67.63 75.98 52.38 56.61 38.27 50.09 33.44 39.47 12.32 22.52 30.40 40.72 39.93 48.65
RAG-GD (K=5)BM25 40.22 50.01 66.13 74.28 37.81 42.19 38.99 51.14 34.15 40.04 12.54 22.61 28.00 39.61 36.83 45.70
E5 45.68 55.84 67.66 76.07 52.31 56.48 39.01 50.94 33.94 39.83 13.20 23.47 32.00 42.08 40.54 49.24
RAG-GD (K=10)BM25 40.28 50.29 65.93 74.07 37.73 42.10 39.10 51.20 34.26 40.08 12.62 22.86 28.80 39.92 36.96 45.79
E5 45.48 55.51 67.71 76.03 52.38 56.49 39.10 50.96 34.03 39.79 13.41 23.41 32.00 42.83 40.59 49.29
Table 3: Additional context-conditioned baselines, run on Qwen-2.5-7B only. For comparison anchors
(Qwen Base adapter and RAG-GD at the same retriever and benchmark), see Table 2.
Method RetrieverSingle-Hop QA Multi-Hop QA Avg.
NQ TriviaQA PopQA HotpotQA 2Wiki MuSiQue Bamboogle
EM F1 EM F1 EM F1 EM F1 EM F1 EM F1 EM F1 EM F1
Vanilla RAG + few shotBM25 27.72 36.80 58.41 65.98 28.80 33.18 31.73 41.43 26.27 32.05 5.88 13.58 11.20 20.27 27.14 34.76
E5 38.78 49.75 63.14 70.94 43.82 50.00 32.28 42.05 24.20 30.80 5.99 13.21 16.80 25.42 32.14 40.31
Base adapter + few shotBM25 34.07 43.18 62.26 69.97 32.94 37.65 34.65 46.18 28.91 34.56 9.06 19.66 19.20 30.19 31.58 40.20
E5 41.52 51.25 65.03 73.11 47.85 52.38 34.51 46.08 29.81 35.70 8.44 18.67 26.40 36.08 36.22 44.75
Prompt tuningBM25 28.94 40.81 58.47 69.11 32.42 37.14 32.43 44.98 32.18 38.54 7.36 17.79 22.40 33.15 30.60 40.22
E5 35.92 48.16 61.09 71.73 47.10 51.96 32.72 45.14 31.60 38.04 7.28 17.22 26.40 32.98 34.59 43.60
HyperTuningBM25 33.62 43.21 61.38 69.90 33.71 37.97 34.55 46.34 31.87 37.81 7.52 17.60 19.20 31.69 31.69 40.65
E5 40.72 50.51 64.19 72.57 48.88 53.25 34.76 46.16 31.47 37.44 6.95 16.74 24.00 32.83 35.85 44.21
TT-SGD (K=5)BM25 34.37 43.32 63.32 70.75 32.94 37.63 35.16 47.13 28.76 34.53 8.77 19.51 23.20 33.26 32.36 40.88
E5 42.52 52.14 65.50 73.45 47.95 52.51 35.62 47.43 29.67 35.59 8.93 19.13 25.60 34.88 36.54 45.02
J Use of large language models
Large language models (LLMs) were only used to assist with language polishing and minor gram-
matical editing of this manuscript.
21