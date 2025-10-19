# The Role of Parametric Injection-A Systematic Study of Parametric Retrieval-Augmented Generation

**Authors**: Minghao Tang, Shiyu Ni, Jingtong Wu, Zengxin Han, Keping Bi

**Published**: 2025-10-14 16:05:01

**PDF URL**: [http://arxiv.org/pdf/2510.12668v1](http://arxiv.org/pdf/2510.12668v1)

## Abstract
Retrieval-augmented generation (RAG) enhances large language models (LLMs) by
retrieving external documents. As an emerging form of RAG, parametric
retrieval-augmented generation (PRAG) encodes documents as model parameters
(i.e., LoRA modules) and injects these representations into the model during
inference, enabling interaction between the LLM and documents at parametric
level. Compared with directly placing documents in the input context, PRAG is
more efficient and has the potential to offer deeper model-document
interaction. Despite its growing attention, the mechanism underlying parametric
injection remains poorly understood. In this work, we present a systematic
study of PRAG to clarify the role of parametric injection, showing that
parameterized documents capture only partial semantic information of documents,
and relying on them alone yields inferior performance compared to interaction
at text level. However, these parametric representations encode high-level
document information that can enhance the model's understanding of documents
within the input context. When combined parameterized documents with textual
documents, the model can leverage relevant information more effectively and
become more robust to noisy inputs, achieving better performance than either
source alone. We recommend jointly using parameterized and textual documents
and advocate for increasing the information content of parametric
representations to advance PRAG.

## Full Text


<!-- PDF content starts -->

The Role of Parametric Injection-A Systematic Study of
Parametric Retrieval-Augmented Generation
Minghao Tang
State Key Laboratory of AI Safety,
ICT, Chinese Academy of Sciences
University of Chinese Academy of Sciences
Beijing, China
tangminghao25@mails.ucas.ac.cnShiyu Ni
State Key Laboratory of AI Safety,
ICT, Chinese Academy of Sciences
University of Chinese Academy of Sciences
Beijing, China
nishiyu23z@ict.ac.cn
Jingtong Wu, Zengxin Han
wangzhenlingwu@163.com
zengxin.hanzx@gmail.comKeping Bi
State Key Laboratory of AI Safety,
ICT, Chinese Academy of Sciences
University of Chinese Academy of Sciences
Beijing, China
bikeping@ict.ac.cn
Abstract
Retrieval-augmented generation (RAG) enhances large language
models (LLMs) by retrieving external documents. As an emerging
form of RAG, parametric retrieval-augmented generation (PRAG)
encodes documents as model parameters (i.e., LoRA modules) and
injects these representations into the model during inference, en-
abling interaction between the LLM and documents at parametric
level. Compared with directly placing documents in the input con-
text, PRAG is more efficient and has the potential to offer deeper
model–document interaction. Despite its growing attention, the
mechanism underlying parametric injection remains poorly un-
derstood. In this work, we present a systematic study of PRAG to
clarify the role of parametric injection, showing that parameterized
documents capture only partial semantic information of documents,
and relying on them alone yields inferior performance compared
to interaction at text level. However, these parametric representa-
tions encode high-level document information that can enhance
the model’s understanding of documents within the input context.
When combined parameterized documents with textual documents,
the model can leverage relevant information more effectively and
become more robust to noisy inputs, achieving better performance
than either source alone. We recommend jointly using parameter-
ized and textual documents and advocate for increasing the infor-
mation content of parametric representations to advance PRAG.
CCS Concepts
•Information systems→Novelty in information retrieval.
Permission to make digital or hard copies of all or part of this work for personal or
classroom use is granted without fee provided that copies are not made or distributed
for profit or commercial advantage and that copies bear this notice and the full citation
on the first page. Copyrights for components of this work owned by others than the
author(s) must be honored. Abstracting with credit is permitted. To copy otherwise, or
republish, to post on servers or to redistribute to lists, requires prior specific permission
and/or a fee. Request permissions from permissions@acm.org.
Conference acronym ’XX, June 03–05, 2018, Woodstock, NY
©2018 Copyright held by the owner/author(s). Publication rights licensed to ACM.
ACM ISBN 978-1-4503-XXXX-X/18/06
https://doi.org/XXXXXXX.XXXXXXXKeywords
Retrieval-Augmented Generation; Parametric RAG; LoRA
ACM Reference Format:
Minghao Tang, Shiyu Ni, Jingtong Wu, Zengxin Han, and Keping Bi. 2018.
The Role of Parametric Injection-A Systematic Study of Parametric Retrieval-
Augmented Generation. InProceedings of Make sure to enter the correct
conference title from your rights confirmation emai (Conference acronym ’XX).
ACM, New York, NY, USA, 12 pages. https://doi.org/XXXXXXX.XXXXXXX
1 Introduction
Despite the remarkable capabilities of large language models (LLMs)
across a wide range of tasks [ 5,6,29], their knowledge is limited by
the training data. When confronted with questions that fall outside
their knowledge boundary, LLMs often hallucinate—generating flu-
ent yet factually incorrect responses [ 13,17,18]. Retrieval-augmented
generation (RAG) addresses this limitation by retrieving relevant
external documents to supplement the model’s internal knowl-
edge [ 11,15], and has become an effective approach for knowledge-
intensive tasks such as factual question answering (QA) [20, 36].
A key component in RAG is how to interact with the retrieved
documents. Recent advances have explored diverse strategies for
this interaction, which broadly fall into three categories: (1) Token-
level augmentation [ 28,32,34]: Retrieved documents are directly
inserted to the input context, allowing the model to attend to them
through its self-attention mechanism. Although this approach is
simple and compatible with off-the-shelf LLMs, it substantially in-
creases the context length, leading to higher inference costs and lim-
ited accessible content under a fixed context window. Furthermore,
as the model interacts with the document only through attention,
it may fail to fully comprehend the content due to shallow interac-
tion. (2) Embedding-level fusion [ 4,10,33]: To reduce the inference
overhead of long contexts, these approaches encode documents
offline—typically using the LLM or a dedicated encoder—and inject
the resulting document embeddings into the LLM during inference
via cross-attention mechanisms, thereby decoupling retrieved doc-
uments from the input context. However, these methods typically
require additional training and their reliance on static embeddingsarXiv:2510.12668v1  [cs.IR]  14 Oct 2025

Conference acronym ’XX, June 03–05, 2018, Woodstock, NY Minghao Tang et al.
often leads to more limited interaction. (3) Parametric-level adapta-
tion [ 3,24,27]: Su et al . [24] propose parametric RAG (PRAG) which
encodes documents as model parameters (i.e., LoRA modules) and
use such parameters to updated the LLM during inference. Since it
does not require increasing the context length and has the potential
to enable deep interaction with documents, PRAG has attracted
significant attention [3, 23, 27].
However, existing efforts on PRAG have primarily focused on
optimizing the offline storage overhead and improving RAG per-
formance. The actual role of parametric injection remains under-
explored—for instance, it is unclear whether the injected parame-
ters genuinely store document knowledge or merely activate the
model’s inherent ability to answer questions.
In this work, we conduct a systematic analysis of PRAG to un-
cover the underlying mechanisms of parametric knowledge in-
jection. We first present a modified reproduction of the original
work [ 24], addressing several confounding design choices in its set-
tings. Our replication yields two key observations that motivate our
hypotheses: (1) PRAG outperforms the vanilla LLM (i.e., direct an-
swering without retrieval) but underperforms standard RAG (i.e., di-
rectly appending retrieved documents to the input prompt), suggest-
ing thatthe parameterized document may not encode full fac-
tual content; and (2) PRAG-Combine, a hybrid variant that injects
parametric knowledge while also retaining textual documents in the
context, achieves the best performance. Since the text already con-
tains all fine-grained facts, we hypothesize thatthe parameterized
document may encode high-level informationthat enhances
the model’s understanding of the textual information. Such en-
hanced understanding may yield two benefits: (i) better utilization
of relevant content, and (ii) greater robustness to noisy documents.
We first examine how much document information is encoded in
the injected parameters. To ensure that the model must rely on the
document to answer correctly, we construct a new dataset compris-
ing facts after the LLM’s knowledge cut-off date and complement
this with analyses of the model’s internal states. Results show that
parametric representations do encode semantic information from
the documents, but the encoding is incomplete, lacking sufficient
fine-grained factual detail. Nevertheless, these representations con-
tain high-level semantic information that can enhance the model’s
understanding of the documents in the input context.
We further analyze how this high-level information enhances
the model’s understanding of documents: whether it enables fuller
use of relevant documents or greater robustness to noisy ones. Eval-
uations on challenging multi-hop QA tasks, conducted with gold
passages, show that parametric injection helps the model interpret
and leverage the provided context more effectively. This benefit gen-
eralizes across downstream tasks, suggesting that the improvement
reflects genuine document understanding rather than mere task-
specific adaptation. Finally, to assess robustness to retrieval noise,
we inject artificial distractors into the retrieved passages. Mod-
els with parametric injection degrade significantly less than their
non-injected counterparts and maintain higher performance—even
when all passages are replaced with noise.
Although parametric injection can enhance the model’s under-
standing of documents in the context, enabling better use of relevant
documents and greater robustness to noise, the performance of re-
lying solely on parameterized documents remains limited, as theycapture only partial document content. Therefore, at the current
stage, we recommend using PRAG-Combine. However, this does
not offer an efficiency advantage compared to directly injecting
documents into the context. We argue that enhancing the ability
of parametric representations to encode fine-grained document
information is key to optimizing PRAG.
2 Related Work
2.1 Retrieval-Augmented Generation
Retrieval-augmented generation (RAG) retrieves knowledge from
external corpora to supplement large language models (LLMs) with
missing information, providing an effective approach to improving
performance on knowledge-intensive tasks and mitigating halluci-
nation [ 11,15,20,36]. The effectiveness of RAG hinges on two core
challenges: accurate information acquisition [ 22,37,38], and effec-
tive integration of the retrieved content [ 25,39]. Existing efforts
on enabling LLMs to better integrate retrieved knowledge can be
broadly categorized into three directions: 1) Token-level augmenta-
tion [ 28,32,34]: These methods optimize how retrieved documents
are incorporated into the input prompt, typically by context re-
finement or enhancement. While simple and widely adopted, they
are subject to limitations such as high inference cost and relatively
shallow interaction. 2) Embedding-level fusion [ 4,10,33]: Instead
of concatenating documents into the prompt, this paradigm encode
documents offline and inject the resulting embeddings into LLMs
during inference via cross-attention. This strategy alleviates the
computational burden of long context. However, the interaction
between LLMs and the retrieved knowledge is even shallower, of-
ten leading to performance degradation—particularly when only
a limited number of documents are used. 3) Parametric-level adap-
tation [ 3,24,27]: A recent and emerging direction that transforms
documents into model parameters through offline encoding, and in-
jects them into the model during inference. Su et al . [24] claims that
injecting documents in a parametric form (e.g., LoRA) enables deep
interactions between LLMs and the retrieved knowledge, while also
reducing inference overhead by avoiding inclusion of documents
in the prompt. Our work centers on on parametric-level adaptation,
with a particular focus on PRAG [ 24], the first work in this para-
digm. We conduct a systematic analysis of PRAG, aiming to validate
its capability for information preservation and deep interaction.
2.2 Parametric RAG
Parametric RAG (PRAG) [ 24] is a novel RAG paradigm that avoids
inserting documents into the context of LLM input. Instead, it en-
codes each document offline into a parametric representation (e.g.,
LoRA) and injects this representation into the LLM during infer-
ence, thereby incorporating external knowledge through parameter
updates rather than context augmentation. To obtain document-
specific parameters, PRAG performs data augmentation for each
document, then trains a dedicated LoRA module on this augmented
data. However, this process requires pre-computing and storing a
LoRA for every document in the corpus, leading to significant com-
putational and storage overhead. To mitigate this, DyPRAG [ 27]
introduces a parameter translator that maps documents directly
to LoRAs at inference time, achieving comparable performance
with substantially reduced cost. Yet, existing studies on parametric

The Role of Parametric Injection-A Systematic Study of Parametric Retrieval-Augmented Generation Conference acronym ’XX, June 03–05, 2018, Woodstock, NY
RAG remain largely focused on improving RAG performance, over-
looking a fundamental question: whether the injected parameters
actually encode and convey factual knowledge to the LLM. For in-
stance, the injected LoRA may act merely as a task-specific adapter,
improving answer formatting rather than conveying actual knowl-
edge. In contrast, our work investigates the underlying mechanisms
of parametric knowledge injection, examining whether LoRA mod-
ules indeed encode knowledge, to what extent such knowledge is
preserved, and whether it can be effectively utilized by LLMs.
3 Preliminary
This section formalizes the inference pipeline of standard RAG and
parametric RAG, and describes how PRAG encodes documents into
model parameters for knowledge injection.
Standard RAG.Given a query 𝑞and a set of top-k relevant doc-
uments{𝑑1,𝑑2,...,𝑑𝑘}retrieved from a large corpus 𝐶using a re-
triever𝑅, standard RAG constructs an augmented input by concate-
nating the retrieved documents with the query:
𝑥=concat(𝑑 1,𝑑2,...,𝑑𝑘,𝑞).(1)
An LLM with parameters 𝜃then generates the output sequence
conditioned on this augmented input:
𝑦RAG=arg max
𝑦𝑃(𝑦|𝑥;𝜃).(2)
Parametric RAG.In standard RAG, attention allows only shallow
interaction with the retrieved documents. As the context length in-
creases, inference costs grow substantially, and the limited context
window further restricts the accessible content. To address such
limitations, PRAG proposes a paradigm shift: instead of inserting
retrieved documents into the input context, it represents each doc-
ument as model parameters. During inference, these parametric
representations are injected into the LLM, enabling the the model to
interact with documents at the parameter level without increasing
the input context length.
Formally, each document 𝑑𝑖∈𝐶is pre-encoded into a parametric
representation Δ𝜃𝑖=𝐹(𝑑𝑖), where the mapping function 𝐹is imple-
mented in PRAG by training a LoRA module on document-specific
augmented data. At inference time, the parametric representations
of the retrieved top-𝑘documents are merged:
Δ𝜃merged =𝑘∑︁
𝑖=1Δ𝜃𝑖,(3)
and injected into the LLM. The output is then generated conditioned
only on the query 𝑞, but with the model parameters adapted to the
retrieved knowledge:
𝑦PRAG=arg max
𝑦𝑃(𝑦|𝑞;𝜃+Δ𝜃 merged).(4)
PRAG can also be combined with standard RAG, yielding a hybrid
variant referred to asPRAG-Combine. In this setting, retrieved
documents are included in the input prompt while the merged
parametric representation is simultaneously injected, resulting in:
𝑦PRAG-Combine=arg max
𝑦𝑃(𝑦|𝑥;𝜃+Δ𝜃 merged),(5)
where𝑥=concat(𝑑 1,𝑑2,...,𝑑𝑘,𝑞)as in standard RAG.
Document Parameterization.PRAG encodes each document
𝑑𝑖into a parametric representation Δ𝜃𝑖by training a LoRA mod-
ule [ 9] on document-specific data. However, as noted in priorwork [ 1], training solely on raw document text via next-token
prediction often fails to internalize factual knowledge effectively.
To address this, PRAG adopts a data augmentation strategy that
enriches the learning signal by generating question–answer (QA)
pairs grounded in 𝑑𝑖, and constructs training sequences in the form
of document–question–answer triples.
Specifically, for each 𝑑𝑖, it generates multiple rewritten variants
{𝑑1
𝑖,𝑑2
𝑖,...,𝑑𝑛
𝑖}and a set of QA pairs {(𝑞1
𝑖,𝑎1
𝑖),(𝑞2
𝑖,𝑎2
𝑖),...,(𝑞𝑚
𝑖,𝑎𝑚
𝑖)}.
These are combined to form an augmented dataset:
𝐷𝑖={(𝑑𝑗
𝑖,𝑞𝑙
𝑖,𝑎𝑙
𝑖)|𝑗∈[1,𝑛],𝑙∈[1,𝑚]}.(6)
Each triple is concatenated into a single sequence 𝑧=concat(𝑑𝑗
𝑖,𝑞𝑙
𝑖,𝑎𝑙
𝑖)
and used as a training sample. The LoRA parametersΔ𝜃 𝑖are then
optimized by minimizing the negative log-likelihood over all tokens
in the augmented sequences:
min
Δ𝜃𝑖∑︁
(𝑑𝑗
𝑖,𝑞𝑙
𝑖,𝑎𝑙
𝑖)∈𝐷 𝑖𝑇∑︁
𝑡=1−log𝑃𝜃+Δ𝜃 𝑖(𝑧𝑡|𝑧<𝑡).(7)
4 Reproduction of PRAG
The original PRAG study [ 24] contains certain experimental set-
tings that may confound the analysis of parametric injection. To
better understand how parameterized documents influence model
behavior, we conduct a modified reproduction of PRAG. Based on
our results, we formulate several hypotheses about the parametric
injection mechanism, which we validate in subsequent sections.
4.1 Experimental Setup
Our reproduction largely follows the original PRAG implementa-
tion, but incorporates targeted adjustments to a few problematic
settings to ensure a fairer and more interpretable evaluation.
4.1.1 Evaluation Metric.The original PARG used F1 score for eval-
uation. However, F1 is sensitive to surface-level formatting varia-
tions—such as inclusion of explanatory phrases—and often yields
high scores for incorrect answers (e.g., predicting “University of
Washington” when the ground truth is “University of Chicago”).
Consequently, it fails to accurately reflect whether the model has
truly acquired the correct knowledge (see detailed examples in Ap-
pendix A). To address this limitation, we adopt an LLM-as-a-judge
evaluation, which provides a more reliable assessment of factual
correctness by leveraging a strong LLM to compare model outputs
against ground-truth answers [7].
Specifically, we use Qwen2.5-32B-Instruct [ 29] as the judge,
prompting it with the original question, the ground-truth answer,
and the model’s prediction to assess factual consistency. We re-
port accuracy based on LLM judgments—i.e., the percentage of
predictions deemed correct—as our primary evaluation metric.
4.1.2 Parameterization Settings.In the original PRAG setup, few-
shot prompts were used during both parameterization and inference
on several datasets. However, this practice conflates task-specific
patterns with the factual content of the documents, causing the
resulting parametric representations to act more as task adapters
than as carriers of document knowledge. Since our goal is to isolate
and study the effect of knowledge injection via parameters, we
remove all few-shot examples from both training and inference.

Conference acronym ’XX, June 03–05, 2018, Woodstock, NY Minghao Tang et al.
Table 1: Reproduction results of four methods, evaluated by accuracy based on LLM judgments. Bold numbers indicate the best
performance under each model, and the second-best results are underlined. CWQ denotes ComplexWebQuestions.
LLM Method2WikiMultihopQA HotpotQAPopQA CWQ Average
Compare Bridge Inference Compose Total Bridge Compare Total
LLaMA3.2-
1B-InstructVanilla 43.00 43.00 0.66 3.66 21.00 9.00 42.33 16.00 10.66 30.33 21.96
RAG 30.00 34.00 7.33 8.33 21.00 24.00 45.33 30.33 48.6631.33 28.03
PRAG45.00 43.662.00 5.0023.6614.66 48.66 21.00 23.33 29.66 25.56
PRAG-Combine 36.33 37.337.66 9.6622.00 25.33 51.33 31.00 48.66 36.66 30.60
Qwen2.5-
1.5B-InstructVanilla28.3329.33 0.66 5.00 14.00 7.00 41.00 13.33 13.00 27.66 17.93
RAG 24.33 21.66 5.66 4.00 14.66 23.33 46.66 27.33 50.0023.33 24.10
PRAG 27.6634.662.66 5.00 16.3310.00 38.00 14.33 21.3331.6620.16
PRAG-Combine 28.00 27.337.66 8.0015.66 25.33 52.33 31.6646.00 28.99 27.10
Qwen2.5-
7B-InstructVanilla 49.66 47.66 1.66 7.00 25.00 14.00 61.66 20.33 18.00 33.66 27.86
RAG 45.00 41.33 10.66 8.00 22.66 31.66 54.65 34.66 36.66 26.00 31.33
PRAG56.33 49.002.33 10.66 28.6620.0063.0026.66 31.3344.6633.26
PRAG-Combine 46.66 37.3312.00 11.3325.00 35.3357.6638.66 43.3337.00 34.43
All other settings follow the original implementation: each doc-
ument is paired with one rewritten and three QA pairs for data
augmentation; LoRA modules are trained for one epoch with a
learning rate of3×10−4, applied exclusively to the feed-forward
network (FFN) layers with rank𝑟=2and scaling factor𝛼=32.
4.1.3 Datasets.We adopt the same four datasets as in the original
PRAG work: 2WikiMultihopQA [ 8], HotpotQA [ 35], ComplexWe-
bQuestions [ 26], and PopQA [ 16]. Following the original setup, we
evaluate on the first 300 questions per dataset. For 2WikiMulti-
hopQA and HotpotQA, the “Total” column in Table 1 and Table 3
reports results on the first 300 questions of the full dataset, while
each sub-task column shows results on the first 300 questions within
that sub-task. For each question, we retrieve the top-3 passages
from a Wikipedia dump using BM25 [21].
4.1.4 Methods and Models.We evaluate four methods:Vanilla
(base LLM without retrieval),RAG(directly appending documents
to the input prompt),PRAG(pure parametric injection), andPRAG-
Combine(hybrid of RAG and PRAG). Experiments are conducted
on three open-source LLMs: LLaMA3.2-1B-Instruct [ 5], Qwen2.5-
1.5B-Instruct, and Qwen2.5-7B-Instruct [ 29], spanning different
model families and scales. All generations use greedy decoding.
4.2 Reproduction Results
Table 1 presents our reproduction results using LLM-judged accu-
racy. The results suggest several hypotheses about parametric injec-
tion, which we systematically validate in subsequent sections: (1)
Parametric representations may not fully encode the factual
content of documents.While PRAG consistently outperforms
Vanilla across most datasets, it generally underperforms standard
RAG under our LLM-based evaluation. This implies that the para-
metric encoding may miss fine-grained details or nuanced facts,
limiting its utility as a standalone knowledge source. The original
PRAG study reported stronger performance for PRAG, we attribute
this discrepancy to the use of F1 score. As shown in Appendix A,
F1 often assigns high scores to fluent but factually incorrect out-
puts—a bias that particularly benefits PRAG, as parametric adap-
tation makes the model more prone to generating template-likeresponses—potentially inflating its apparent effectiveness. (2)Para-
metric injection may enhance the model’s comprehension
of the provided context.PRAG-Combine consistently improves
over RAG. Since RAG already supplies the full document content in
the prompt, this improvement suggests that parametric knowledge
does not merely duplicate information, but instead helps the model
better interpret the given context. This enhanced comprehension
may lead to: (i) more effective utilization of relevant passages, or
(ii) greater robustness to irrelevant or noisy retrieval results.
5 How Much Knowledge is Encoded in
Parametric Representations
To validate our hypothesis that parametric representations may not
fully encode the factual content of documents, we design controlled
experiments and conduct detailed analyses in this section.
5.1 Experimental Setup
Since the goal is to measure what and how much knowledge is
stored in parametric representations, it is necessary to exclude
the influence of the model’s internal knowledge. To achieve this,
we construct a dataset containing knowledge that emerged after
the LLM’s knowledge cutoff date, where the model must rely on
external documents to answer the question correctly.
Specifically, we collect 300 news articles published in 2025, all
of which postdate the the training cutoffs of LLMs used in this
work. Each article is split into at most three passages, with lengths
matched to those in the Wikipedia dump used in Section 4.1. We
use Qwen2.5-32B-Instruct to generate two types of QA pairs for
each article: (1) Factual QA: simple factual questions based on the
article content; (2) Multihop QA: questions that require combining
multiple facts from the article. During inference, no retrieval is
performed. Instead, the model is provided with the question and
all passages from the source article, and the corresponding docu-
ment parameters (trained on those passages) are injected. All other
settings—including document parameterization, model selection,
and evaluation metric—follow Section 4.1 exactly.

The Role of Parametric Injection-A Systematic Study of Parametric Retrieval-Augmented Generation Conference acronym ’XX, June 03–05, 2018, Woodstock, NY
/uni0000003e/uni0000003e/uni00000102/uni00000044/uni00000004/uni00000372/uni000003ed/uni00000011 /uni00000059/uni000001c1/uni0000011e/uni00000176/uni00000372/uni000003ed/uni00000358/uni000003f1/uni00000011 /uni00000059/uni000001c1/uni0000011e/uni00000176/uni00000372/uni000003f3/uni00000011/uni000003ec/uni00000358/uni000003ec/uni000003ec/uni000003ef/uni000003ec/uni00000358/uni000003ec/uni000003ec/uni000003f2/uni000003ec/uni00000358/uni000003ec/uni000003ec/uni000003f5/uni000003ec/uni00000358/uni000003ec/uni000003ec/uni00000004/uni00000110/uni00000110/uni000001b5/uni0000018c/uni00000102/uni00000110/uni000001c7/uni00000026/uni00000102/uni00000110/uni0000019a/uni000001b5/uni00000102/uni0000016f/uni00000003/uni00000059/uni00000004
/uni0000003e/uni0000003e/uni00000102/uni00000044/uni00000004/uni00000372/uni000003ed/uni00000011 /uni00000059/uni000001c1/uni0000011e/uni00000176/uni00000372/uni000003ed/uni00000358/uni000003f1/uni00000011 /uni00000059/uni000001c1/uni0000011e/uni00000176/uni00000372/uni000003f3/uni00000011/uni000003ec/uni00000358/uni000003ec/uni000003ec/uni000003ef/uni000003ec/uni00000358/uni000003ec/uni000003ec/uni000003f2/uni000003ec/uni00000358/uni000003ec/uni000003ec/uni000003f5/uni000003ec/uni00000358/uni000003ec/uni000003ec/uni00000004/uni00000110/uni00000110/uni000001b5/uni0000018c/uni00000102/uni00000110/uni000001c7/uni00000044/uni000001b5/uni0000016f/uni0000019a/uni0000015d/uni0000015a/uni0000017d/uni00000189/uni00000003/uni00000059/uni00000004
/uni00000073/uni00000102/uni00000176/uni0000015d/uni0000016f/uni0000016f/uni00000102 /uni0000005a/uni00000004/uni00000027 /uni00000057/uni0000005a/uni00000004/uni00000027 /uni00000057/uni0000005a/uni00000004/uni00000027/uni00000372/uni00000012/uni0000017d/uni00000175/uni0000010f/uni0000015d/uni00000176/uni0000011e
Figure 1: Performance of different methods on the new-
knowledge dataset, where the model must rely on exter-
nal documents to answer questions correctly. LLaMA-1B
denotes LLaMA3.2-1B-Instruct, Qwen-1.5 denotes Qwen2.5-
1.5B-Instruct, and Qwen-7B denotes Qwen2.5-7B-Instruct.
5.2 Experimental Results
Figure 1 shows performance on our new-knowledge dataset. The
results confirm and refine our initial hypotheses, yielding three
key findings: (1)Parametric representations do encode factual
knowledge.PRAG consistently outperforms Vanilla across both
question types, demonstrating that parametric injection can success-
fully endow the model with new, previously unknown information.
As illustrated in Figure 2, when queried about a event outside its
training horizon, Vanilla hallucinates, while PRAG produces the cor-
rect answer—direct evidence that knowledge is stored in the param-
eters. (2)Parametric representations fails to fully capture the
knowledge.PRAG lags substantially behind RAG, indicating that
current document parameterization protocols do not yet achieve
comprehensive knowledge encoding. In other words, while some
knowledge is encoded, it is insufficient to reliably support question
answering on novel content. (3)Parametric representations may
encode high-level semantic knowledge.PRAG-Combine also
achieves the strongest performance on this new-knowledge bench-
mark. From the perspective of representational content, we con-
jecture that although parametric representations lack fine-grained
factual detail, they capture high-level semantic structures—such
as relational patterns or discourse-level cues. We quantitatively
investigate this in Section 5.3.2.
Passages: ... Singer, actor and reality TV star Tamar Braxton said Tuesday that she “almost died” in a weekend accident that she doesn't remember. “I was found in a pool of blood from my friend with a face injury,” Braxton wrote in an Instagram post. “I fractured my nose, lost some teeth and mobility.”...Question:What did Tamar Braxton fracture in the accident?Answer:her noseVanilla Output: Tamar Braxton fractured her anklein an accident.PRAG Output:her nose.
Figure 2: An example from the new-knowledge dataset. When
asked a question about an event outside its training data,
the Vanilla LLM hallucinates, whereas PRAG produces the
correct answer. More cases are provided in Appendix B.5.3 Further Analysis
We further investigate, from a parametric perspective, whether
these representations capture only partial document information
and whether they contain high-level information that helps the
model in better understanding the document.
5.3.1 Similarity Between Parametric Representations.To further
validate our hypotheses—that parametric representations only en-
code part of the document knowledge—we examine the similarity of
parametric representations across different documents. Specifically,
we compute cosine similarities between flattened LoRA weight
matrices for two types of passage pairs: (i) relevant pairs, which are
segmented from the same article; and (ii) irrelevant pairs from differ-
ent articles. If the parameters capture document-specific semantics,
relevant pairs should be more similar than irrelevant ones.
As shown in Figure 3 (using Qwen2.5-1.5B-Instruct as a rep-
resentative model; full results in Appendix C), this is indeed the
case: relevant pairs exhibit higher average similarity, indicating
that parametric representations encode shared semantic and factual
content. However, the margin is modest: even irrelevant pairs show
a mean similarity of approximately 0.65. This suggests that the
representations fail to fully isolate document-unique information,
supporting that the encoded knowledge remains incomplete.
5.3.2 Quantifying Parametric Knowledge in the Residual.To in-
vestigate whether parametric representations contain high-level
semantic knowledge—such as relational patterns or discourse-level
cues—rather than merely surface facts, we analyze their impact on
the model’s internal states using the parametric knowledge score
(PKS) [ 25], a metric that quantifies the knowledge each FFN layer
contributes to the residual stream.
Specifically, for each generated token 𝑥𝑛in the response and
each layer𝑙, we compute the Jensen–Shannon divergence (JSD)
between the vocabulary distributions before and after the FFN block,
obtained via LogitLens [19]:
LogitLens(𝑥)=LayerNorm(𝑥)𝑊 𝑈,(8)
𝑃𝑙
𝑛=JSD(𝑞(𝑥𝑙,𝑏𝑒𝑓𝑜𝑟𝑒
𝑛)||𝑞(𝑥𝑙,𝑎𝑓𝑡𝑒𝑟
𝑛)),(9)
where𝑊𝑈is the unembedding matrix of the LLM and 𝑞(𝑥)=
softmax(LogitLens(𝑥)) . The PKS for layer 𝑙is obtained by aver-
aging𝑃𝑙
𝑛over all tokens in the response.
Figure 4 shows the per-layer difference in PKS between mod-
els with and without parametric injection (i.e., PRAG vs. Vanilla;
PRAG-Combine vs. RAG). While early layers exhibit inconsistent
changes, the last few layers consistently show substantially higher
PKS across all LLMs when parametric knowledge is injected. Prior
work [ 12,30] has shown that deeper transformer layers are pri-
marily responsible for high-level semantic processing—such as in-
tegrating information across tokens, resolving coreference, and
constructing structured event representations. The concentration
of PKS gains in these layers suggests that parametric representa-
tions do not merely store isolated facts, butencode high-level
semantic knowledgethat may contribute to more advanced com-
prehension of the input context.

Conference acronym ’XX, June 03–05, 2018, Woodstock, NY Minghao Tang et al.
0.55 0.60 0.65 0.70 0.75 0.80
Cosine Similarity02468101214DensitySimilarity Distribution of down_proj
Relevant Pairs
Irrelevant Pairs
0.50 0.55 0.60 0.65 0.70 0.75
Cosine Similarity02468101214Similarity Distribution of gate_proj
Relevant Pairs
Irrelevant Pairs
0.45 0.50 0.55 0.60 0.65 0.70 0.75
Cosine Similarity02468101214Similarity Distribution of up_proj
Relevant Pairs
Irrelevant Pairs
Figure 3: Similarity distribution of LoRA modules (averaged over all layers) for Qwen2.5-1.5B-Instruct. Relevant passage
pairs exhibit slightly higher similarity than irrelevant pairs, implying that LoRAs encode some semantic and factual content.
However, the gap is modest, suggesting limited encoding of fine-grained, document-specific knowledge.
0.02
0.01
0.000.010.020.030.040.05Score DifferenceDifference in Parametric Knowledge Score (PRAG - Vanilla)
Avg Diff: -0.0002
0.02
0.000.020.040.060.080.100.120.14Difference in Parametric Knowledge Score (PRAG - Vanilla)
Avg Diff: 0.0169
0.02
0.01
0.000.010.020.030.040.05Difference in Parametric Knowledge Score (PRAG - Vanilla)
Avg Diff: 0.0036
0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15
Layer0.02
0.000.020.04Score DifferenceDifference in Parametric Knowledge Score (Combine - RAG)
Avg Diff: 0.0032
0123456789101112131415161718192021222324252627
Layer0.02
0.000.020.040.060.080.100.12Difference in Parametric Knowledge Score (Combine - RAG)
Avg Diff: 0.0164
0123456789101112131415161718192021222324252627
Layer0.02
0.01
0.000.010.020.030.040.05Difference in Parametric Knowledge Score (Combine - RAG)
Avg Diff: 0.0045
(a) LLaMA3.2-1B-Instruct (b) Qwen2.5-1.5B-Instruct (c) Qwen2.5-7B-Instruct
Figure 4: Difference in parametric knowledge score between LLMs with and without parametric injection. Most comparisons
show a positive average difference, and all models exhibit a pronounced increase in the final layers, indicating that parametric
injection increases parametric knowledge in the residual stream and that the representations encode high-level knowledge.
6 Does Parametric Injection Enhance
Utilization of Relevant Passages
Our analysis so far shows that parametric representations encode
not only some factual knowledge but also high-level semantic
knowledge. This provides preliminary support for our second hy-
pothesis—that parametric injection enhances the model’s under-
standing of the provided context. As hypothesized in Section 4.2,
this enhanced understanding may manifest in two complemen-
tary ways: (i) more effective utilization of relevant passages, or (ii)
greater robustness to irrelevant or noisy retrieval results.
In this section, we empirically test the first mechanism: whether
the high-level knowledge encoded in parametric representations
helps the model better utilize relevant retrieved passages. We ex-
amine the second mechanism—robustness to retrieval noise—in the
following section.6.1 Experimental Setup
To rigorously evaluate whether parametric injection enhances the
model’s ability to utilize relevant passages, we design experiments
using gold passages and complex questions—a setting where effec-
tive utilization of the provided context is essential. By bypassing
retrieval, we ensure that performance difference reflects the model’s
capacity to interpret and integrate the given passages.
Since PRAG’s document parameterization is trained on QA-
formatted data (i.e., document–question–answer triples), any ob-
served improvement in document utilization might stem not from
deeper document understanding, but from better adaptation to
the QA task—i.e., learning the QA-task-specific patterns. To rule
out this alternative explanation, we conduct two complementary

The Role of Parametric Injection-A Systematic Study of Parametric Retrieval-Augmented Generation Conference acronym ’XX, June 03–05, 2018, Woodstock, NY
analyses: (i) probing for QA-task-specific components in the rep-
resentations, and (ii) testing whether the parametric knowledge
injection generalizes to non-QA tasks.
Gold-Passages Evaluation.We evaluate on HotpotQA [ 35] and
2WikiMultihopQA [ 8], using the first 300 questions from each. For
every question, we provide the gold supporting passages as input
context and inject LoRA parameters trained on those passages. All
other settings follow Section 4.1.
Probing for QA-Specific Task Knowledge.To directly probe
the presence of QA-specific adaptation, we train a dedicated QA-
task LoRA on the gold passages of 200 questions from the datasets
above, using the same QA-based data augmentation and training
protocol as in Section 4.1. We then analyze its contribution to model
performance to assess whether QA-specific adaptation plays a role
in the observed gains.
Cross-Task Generalization Test.To determine whether the en-
coded knowledge is general or QA-specific, we evaluate paramet-
ric injection on two non-QA tasks: fact-checking on FEVER [ 31],
measured by label accuracy; and slot-filling on Zero-Shot-RE [ 14],
measured by F1 score. For each question, we retrieve top-3 pas-
sages, use the same QA-based parameterization protocol, and apply
task-specific prompts during inference.
6.2 Experimental Results
Table 2 presents the performance of all methods and their variants
augmented with the QA-specific LoRA on gold passages. The re-
sults show the following: (1)Parametric injection enhances the
model’s ability to utilize relevant passages.PRAG-Combine
consistently outperforms RAG by a notable margin, especially on
such complex multi-hop questions, demonstrating that the high-
level knowledge encoded in parametric representations actively sup-
ports more effective context utilization. (2)The high-level knowl-
edge in parametric representations inherently includes QA-
specific task patterns.Adding a separately trained QA-specific
LoRA to PRAG or PRAG-Combine yields little to no improvement,
indicating that the task-adaptive signals it provides are already em-
bedded within the document-parameterized LoRA. (3)Parametric
injection provides more than task-specific cues-it encodes
general document understanding.While both Vanilla and RAG
benefit from the QA-specific LoRA, they still fall short of their
parametric-injection counterparts. Moreover, this advantage gen-
eralizes: as shown in Figure 5, PRAG and PRAG-Combine also
outperform baselines on non-QA tasks, following the same per-
formance trend. Together, these results confirm that parametric
representations encode general semantic and structural knowledge
of documents, going beyond both surface facts and QA-task-specific
patterns, and thereby enabling robust contextual comprehension
across diverse tasks.
6.3 Further Analysis on Context Faithfulness
Given that parametric injection enhances the model’s ability to
utilize provided passages, we expect it to also increase context
faithfulness—the tendency to ground answers in the given relevant
context even when it contradicts the model’s internal knowledge.
To verify this, we evaluate on the ConFiQA dataset [ 2], which
consists of questions paired with counterfactual passages. TheseTable 2: Performance of all methods and their variants aug-
mented with the QA-specific LoRA on gold passages from
2WikiMultihopQA (2Wiki) and HotpotQA. PRAG-Combine
(Combine) consistently outperforms RAG, indicating that
parametric injection enhances document utilization.
LLM MethodWithout QA-LoRA With QA-LoRA
2Wiki HotpotQA 2Wiki HotpotQA
LLaMA3.2-
1B-InstructVanilla 21.00 16.00 20.66 16.66
RAG 39.33 63.66 45.66 63.66
PRAG 23.33 21.66 23.66 22.33
Combine46.00 69.33 46.33 69.33
Qwen2.5-
1.5B-InstructVanilla 14.00 13.33 24.66 15.66
RAG 28.33 53.66 47.3363.00
PRAG 18.33 15.00 18.66 16.00
Combine40.00 67.0040.00 67.33
Qwen2.5-
7B-InstructVanilla 25.00 20.33 28.00 20.33
RAG 56.99 67.33 54.33 73.00
PRAG 30.66 32.33 30.66 32.66
Combine64.00 80.66 64.00 80.33
/uni0000003e/uni0000003e/uni00000102/uni00000044/uni00000004/uni00000372/uni000003ed/uni00000011 /uni00000059/uni000001c1/uni0000011e/uni00000176/uni00000372/uni000003ed/uni00000358/uni000003f1/uni00000011 /uni00000059/uni000001c1/uni0000011e/uni00000176/uni00000372/uni000003f3/uni00000011/uni000003ec/uni00000358/uni000003ec/uni000003ec/uni000003ee/uni000003f1/uni00000358/uni000003ec/uni000003ec/uni000003f1/uni000003ec/uni00000358/uni000003ec/uni000003ec/uni000003f3/uni000003f1/uni00000358/uni000003ec/uni000003ec/uni000003ed/uni000003ec/uni000003ec/uni00000358/uni000003ec/uni000003ec/uni00000004/uni00000110/uni00000110/uni000001b5/uni0000018c/uni00000102/uni00000110/uni000001c7/uni00000026/uni0000001c/uni00000073/uni0000001c/uni0000005a
/uni0000003e/uni0000003e/uni00000102/uni00000044/uni00000004/uni00000372/uni000003ed/uni00000011 /uni00000059/uni000001c1/uni0000011e/uni00000176/uni00000372/uni000003ed/uni00000358/uni000003f1/uni00000011 /uni00000059/uni000001c1/uni0000011e/uni00000176/uni00000372/uni000003f3/uni00000011/uni000003ec/uni00000358/uni000003ec/uni000003ec/uni000003ee/uni000003ec/uni00000358/uni000003ec/uni000003ec/uni000003f0/uni000003ec/uni00000358/uni000003ec/uni000003ec/uni000003f2/uni000003ec/uni00000358/uni000003ec/uni000003ec/uni00000026/uni000003ed/uni00000003/uni0000005e/uni00000110/uni0000017d/uni0000018c/uni0000011e/uni0000007f/uni0000011e/uni0000018c/uni0000017d/uni00000372/uni0000005e/uni0000015a/uni0000017d/uni0000019a/uni00000372/uni0000005a/uni0000001c
/uni00000073/uni00000102/uni00000176/uni0000015d/uni0000016f/uni0000016f/uni00000102 /uni0000005a/uni00000004/uni00000027 /uni00000057/uni0000005a/uni00000004/uni00000027 /uni00000057/uni0000005a/uni00000004/uni00000027/uni00000372/uni00000012/uni0000017d/uni00000175/uni0000010f/uni0000015d/uni00000176/uni0000011e
Figure 5: Performance of different methods on fact-checking
and slot filling. LLaMA-1B denotes LLaMA3.2-1B-Instruct,
Qwen-1.5 denotes Qwen2.5-1.5B-Instruct, and Qwen-7B de-
notes Qwen2.5-7B-Instruct. Methods with parametric injec-
tion achieve stronger performance on both tasks, indicating
that the benefits generalize to other downstream tasks.
passages are constructed by replacing key entities in original gold
passages with plausible same-type substitutes, preserving topical
coherence while introducing factual inaccuracies. We sample the
first 900 questions and use the counterfactual passages both as
input context and for document parameterization. Faithfulness is
measured by the proportion of outputs that align with the counter-
factual context (i.e., counterfactual answers).
Figure 6 presents the distribution of output answer types across
methods and models. We observe that: (1) PRAG-Combine consis-
tently generates more counterfactual answers than RAG, indicating
that parametric injectionstrengthens context faithfulness. (2)
PRAG generally produces more counterfactual answers and fewer
original ones than Vanilla, suggesting that parametric injection can
alter the model’s internal knowledgeto some extent.
7 Does Parametric Injection Improve
Robustness to Noise Passages
The previous section demonstrated that parametric injection en-
hances the model’s ability to utilize relevant passages. In this sec-
tion, we investigate the second hypothesized mechanism: whether

Conference acronym ’XX, June 03–05, 2018, Woodstock, NY Minghao Tang et al.
0 100 200 300 400 500 600 700 800 900
Number of ExamplesVanilla
RAG
PRAG
CombineMethods317
41
398163
79
140
74719
504
719
428LLaMA3.2-1B-Instruct
0 100 200 300 400 500 600 700 800 900
Number of ExamplesVanilla
RAG
PRAG
Combine31
431
53
465226
76
205
91643
393
642
344Qwen2.5-1.5B-Instruct
0 100 200 300 400 500 600 700 800 900
Number of ExamplesVanilla
RAG
PRAG
Combine372
80
504238
119
300
92636
409
520
304Qwen2.5-7B-Instruct
Counterfactual Answer Original Answer Others
Figure 6: Distribution of output answer types on ConFiQA across methods and models. PRAG-Combine (denoted as Combine)
exhibits stronger context faithfulness than RAG, and PRAG also generates more counterfactual answers than Vanilla.
BM25 T op3 Replace Last Replace First Replace All
Noise Condition1618202224262830Accuracy
LLaMA3.2-1B-Instruct
BM25 T op3 Replace Last Replace First Replace All
Noise Condition1214161820222426Accuracy
Qwen2.5-1.5B-Instruct
BM25 T op3 Replace Last Replace First Replace All
Noise Condition101520253035Accuracy
Qwen2.5-7B-Instruct
Vanilla RAG PRAG PRAG-Combine
Figure 7: Performance under varying noise conditions. All parametric-injection variants consistently match or outperform
their non-injected counterparts, suggesting that parametric injection enhances robustness to retrieval noise and that LLMs can
recognize and ignore irrelevant knowledge encoded in the injected parameters.
the high-level knowledge encoded in parametric representations
also helps the model better handle irrelevant or noisy retrieved
documents, thereby improving robustness to retrieval noise.
7.1 Experimental Setup
To assess whether parametric injection enhances robustness to
retrieval noise, we introduce controlled artificial noise into the
retrieved passages. Specifically, for each question, we begin with
the top-3 passages retrieved by BM25 and construct four variants
by replacing one or more of them with random, irrelevant passages:
•BM25 Top3: the original top-3 BM25-retrieved passages (no
noise injected);
•Replace Last: the least relevant passage (rank 3) is replaced with
a random noise passage;
•Replace First: the most relevant passage (rank 1) is replaced
with a random noise passage;
•Replace All: all three passages are replaced with random noise
passages.
We evaluate all methods on the same four datasets used in Sec-
tion 4.1, using identical document parameterization, model selec-
tion, and evaluation metrics. For each noise condition, we report
the average accuracy across the four datasets.
7.2 Experimental Results
Figure 7 presents the performance of all methods under varying
levels of retrieval noise. Our analysis yields two key findings: (1)Parametric injection enhances robustness to retrieval noise.
As expected, all methods suffer performance degradation as noise
increases. Nevertheless, PRAG-Combine consistently outperforms
RAG across all noise conditions—even when all retrieved passages
are replaced with irrelevant ones—demonstrating that parametric
injection effectively mitigates the adverse impact of noisy context.
(2)LLMs can recognize irrelevant knowledge encoded in para-
metric representations. PRAG’s performance gradually declines
as more retrieved passages are corrupted, eventually converging to
Vanilla under full noise. This confirms that the injected parameters
indeed encode document-specific information. Crucially, even in
the full-noise setting—where the injected parameters encode only
irrelevant content—PRAG never underperforms its non-injected
counterpart. This suggests that the model can detect irrelevant
parametric knowledge and avoid being misled by it.
8 Conclusion and Discussion
In this paper, we conduct a systematic analysis of parametric RAG
to uncover the underlying mechanisms of parametric knowledge
injection. Motivated by two central hypotheses—that (1) paramet-
ric representations may not fully encode the factual content of
documents, and (2) parametric injection may enhance the model’s
comprehension of the provided context—we design a series of con-
trolled experiments and internal analyses. Our findings show that
parametric representations do encode document-related knowl-
edge, including high-level semantic knowledge, but the encoding is

The Role of Parametric Injection-A Systematic Study of Parametric Retrieval-Augmented Generation Conference acronym ’XX, June 03–05, 2018, Woodstock, NY
incomplete, lacking sufficient fine-grained factual detail. This high-
level knowledge enables the model to better interpret the provided
context, leading to more effective utilization of relevant passages
and greater robustness to irrelevant or noisy passages.
Our analysis reveals a fundamental limitation of current para-
metric RAG approaches: the injected parameters do not encode
sufficient factual knowledge to support question answering on
their own. As a result, PRAG cannot fully replace standard RAG.
Although, the hybrid PRAG-Combine achieves strong performance
by complementing context with high-level knowledge, this comes
at a cost: it forfeits the original efficiency motivation of PRAG (i.e.,
avoiding token-level context expansion) and introduces additional
computational and storage overhead for document parameteriza-
tion. We argue that the most pressing challenge for this paradigm
is to increase the information content of parametric representa-
tions—encoding richer, more complete factual content, which calls
for carefully designed parameterization strategies.

Conference acronym ’XX, June 03–05, 2018, Woodstock, NY Minghao Tang et al.
References
[1]Zeyuan Allen-Zhu and Yuanzhi Li. 2023. Physics of language models: Part 3.1,
knowledge storage and extraction.arXiv preprint arXiv:2309.14316(2023).
[2]Baolong Bi, Shaohan Huang, Yiwei Wang, Tianchi Yang, Zihan Zhang, Haizhen
Huang, Lingrui Mei, Junfeng Fang, Zehao Li, Furu Wei, et al .2024. Context-
dpo: Aligning language models for context-faithfulness.arXiv preprint
arXiv:2412.15280(2024).
[3]Jinwen Chen, Hainan Zhang, Liang Pang, Yongxin Tong, Haibo Zhou, Yuan
Zhan, Wei Lin, and Zhiming Zheng. 2025. Privacy-Preserving Reasoning with
Knowledge-Distilled Parametric Retrieval Augmented Generation.arXiv preprint
arXiv:2509.01088(2025).
[4]Qian Dong, Qingyao Ai, Hongning Wang, Yiding Liu, Haitao Li, Weihang Su,
Yiqun Liu, Tat-Seng Chua, and Shaoping Ma. 2025. Decoupling Knowledge and
Context: An Efficient and Effective Retrieval Augmented Generation Framework
via Cross Attention. InProceedings of the ACM on Web Conference 2025. 4386–
4395.
[5]Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek
Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Alex
Vaughan, et al .2024. The llama 3 herd of models.arXiv preprint arXiv:2407.21783
(2024).
[6]Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin
Xu, Qihao Zhu, Shirong Ma, Peiyi Wang, Xiao Bi, et al .2025. Deepseek-r1:
Incentivizing reasoning capability in llms via reinforcement learning.arXiv
preprint arXiv:2501.12948(2025).
[7]Xanh Ho, Jiahao Huang, Florian Boudin, and Akiko Aizawa. 2025. LLM-as-a-
Judge: Reassessing the Performance of LLMs in Extractive QA.arXiv preprint
arXiv:2504.11972(2025).
[8]Xanh Ho, Anh-Khoa Duong Nguyen, Saku Sugawara, and Akiko Aizawa. 2020.
Constructing A Multi-hop QA Dataset for Comprehensive Evaluation of Reason-
ing Steps. InProceedings of the 28th International Conference on Computational
Linguistics. 6609–6625.
[9]Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean
Wang, Lu Wang, Weizhu Chen, et al .2022. Lora: Low-rank adaptation of large
language models.ICLR1, 2 (2022), 3.
[10] Gautier Izacard and Edouard Grave. 2020. Leveraging passage retrieval with
generative models for open domain question answering.arXiv preprint
arXiv:2007.01282(2020).
[11] Gautier Izacard, Patrick Lewis, Maria Lomeli, Lucas Hosseini, Fabio Petroni,
Timo Schick, Jane Dwivedi-Yu, Armand Joulin, Sebastian Riedel, and Edouard
Grave. 2023. Atlas: Few-shot learning with retrieval augmented language models.
Journal of Machine Learning Research24, 251 (2023), 1–43.
[12] Ganesh Jawahar, Benoît Sagot, and Djamé Seddah. 2019. What does BERT
learn about the structure of language?. InACL 2019-57th Annual Meeting of the
Association for Computational Linguistics.
[13] Adam Tauman Kalai, Ofir Nachum, Santosh S Vempala, and Edwin Zhang. 2025.
Why language models hallucinate.arXiv preprint arXiv:2509.04664(2025).
[14] Omer Levy, Minjoon Seo, Eunsol Choi, and Luke Zettlemoyer. 2017. Zero-shot
relation extraction via reading comprehension.arXiv preprint arXiv:1706.04115
(2017).
[15] Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin,
Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel,
et al.2020. Retrieval-augmented generation for knowledge-intensive nlp tasks.
Advances in neural information processing systems33 (2020), 9459–9474.
[16] Alex Mallen, Akari Asai, Victor Zhong, Rajarshi Das, Daniel Khashabi, and
Hannaneh Hajishirzi. 2023. When Not to Trust Language Models: Investigating
Effectiveness of Parametric and Non-Parametric Memories. InProceedings of the
61st Annual Meeting of the Association for Computational Linguistics (Volume 1:
Long Papers). 9802–9822.
[17] Shiyu Ni, Keping Bi, Jiafeng Guo, and Xueqi Cheng. 2024. When Do LLMs
Need Retrieval Augmentation? Mitigating LLMs’ Overconfidence Helps Retrieval
Augmentation. InFindings of the Association for Computational Linguistics ACL
2024. 11375–11388.
[18] Shiyu Ni, Keping Bi, Jiafeng Guo, Lulu Yu, Baolong Bi, and Xueqi Cheng. 2025.
Towards fully exploiting llm internal states to enhance knowledge boundary
perception.arXiv preprint arXiv:2502.11677(2025).
[19] nostalgebraist. 2020. Interpreting GPT: the logit lens.AI Align-
ment Forum. https://www.alignmentforum.org/posts/AcKRB8wDpdaN6v6ru/
interpreting-gpt-the-logit-lens
[20] Ori Ram, Yoav Levine, Itay Dalmedigos, Dor Muhlgay, Amnon Shashua, Kevin
Leyton-Brown, and Yoav Shoham. 2023. In-context retrieval-augmented language
models.Transactions of the Association for Computational Linguistics11 (2023),
1316–1331.
[21] Stephen Robertson, Hugo Zaragoza, et al .2009. The probabilistic relevance
framework: BM25 and beyond.Foundations and Trends®in Information Retrieval
3, 4 (2009), 333–389.
[22] Weijia Shi, Sewon Min, Michihiro Yasunaga, Minjoon Seo, Rich James, Mike
Lewis, Luke Zettlemoyer, and Wen-tau Yih. 2023. Replug: Retrieval-augmented
black-box language models.arXiv preprint arXiv:2301.12652(2023).[23] Weihang Su, Qingyao Ai, Jingtao Zhan, Qian Dong, and Yiqun Liu. 2025. Dy-
namic and parametric retrieval-augmented generation. InProceedings of the 48th
International ACM SIGIR Conference on Research and Development in Information
Retrieval. 4118–4121.
[24] Weihang Su, Yichen Tang, Qingyao Ai, Junxi Yan, Changyue Wang, Hongning
Wang, Ziyi Ye, Yujia Zhou, and Yiqun Liu. 2025. Parametric retrieval augmented
generation. InProceedings of the 48th International ACM SIGIR Conference on
Research and Development in Information Retrieval. 1240–1250.
[25] Zhongxiang Sun, Xiaoxue Zang, Kai Zheng, Yang Song, Jun Xu, Xiao Zhang, Wei-
jie Yu, and Han Li. 2024. Redeep: Detecting hallucination in retrieval-augmented
generation via mechanistic interpretability.arXiv preprint arXiv:2410.11414
(2024).
[26] Alon Talmor and Jonathan Berant. 2018. The Web as a Knowledge-Base for
Answering Complex Questions. InProceedings of the 2018 Conference of the
North American Chapter of the Association for Computational Linguistics: Human
Language Technologies, Volume 1 (Long Papers). 641–651.
[27] Yuqiao Tan, Shizhu He, Huanxuan Liao, Jun Zhao, and Kang Liu. 2025. Dynamic
parametric retrieval augmented generation for test-time knowledge enhancement.
arXiv preprint arXiv:2503.23895(2025).
[28] Minghao Tang, Shiyu Ni, Jiafeng Guo, and Keping Bi. 2025. Injecting External
Knowledge into the Reasoning Process Enhances Retrieval-Augmented Genera-
tion.arXiv preprint arXiv:2507.19333(2025).
[29] Qwen Team. 2024. Qwen2.5: A Party of Foundation Models. https://qwenlm.
github.io/blog/qwen2.5/
[30] Ian Tenney, Dipanjan Das, and Ellie Pavlick. 2019. BERT Rediscovers the Classical
NLP Pipeline. InProceedings of the 57th Annual Meeting of the Association for
Computational Linguistics. 4593–4601.
[31] James Thorne, Andreas Vlachos, Christos Christodoulopoulos, and Arpit Mittal.
2018. FEVER: a large-scale dataset for fact extraction and VERification.arXiv
preprint arXiv:1803.05355(2018).
[32] Harsh Trivedi, Niranjan Balasubramanian, Tushar Khot, and Ashish Sabharwal.
2022. Interleaving retrieval with chain-of-thought reasoning for knowledge-
intensive multi-step questions.arXiv preprint arXiv:2212.10509(2022).
[33] Xi Wang, Taketomo Isazawa, Liana Mikaelyan, and James Hensman. 2024. Kblam:
Knowledge base augmented language model.arXiv preprint arXiv:2410.10450
(2024).
[34] Zihao Wang, Anji Liu, Haowei Lin, Jiaqi Li, Xiaojian Ma, and Yitao Liang. 2024.
Rat: Retrieval augmented thoughts elicit context-aware reasoning in long-horizon
generation.arXiv preprint arXiv:2403.05313(2024).
[35] Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio, William Cohen, Ruslan
Salakhutdinov, and Christopher D Manning. 2018. HotpotQA: A Dataset for
Diverse, Explainable Multi-hop Question Answering. InProceedings of the 2018
Conference on Empirical Methods in Natural Language Processing. 2369–2380.
[36] Hamed Zamani, Fernando Diaz, Mostafa Dehghani, Donald Metzler, and Michael
Bendersky. 2022. Retrieval-enhanced machine learning. InProceedings of the 45th
International ACM SIGIR Conference on Research and Development in Information
Retrieval. 2875–2886.
[37] Hengran Zhang, Keping Bi, Jiafeng Guo, Jiaming Zhang, Shuaiqiang Wang, Dawei
Yin, and Xueqi Cheng. 2025. Distilling a Small Utility-Based Passage Selector
to Enhance Retrieval-Augmented Generation.arXiv preprint arXiv:2507.19102
(2025).
[38] Hengran Zhang, Minghao Tang, Keping Bi, Jiafeng Guo, Shihao Liu, Daiting
Shi, Dawei Yin, and Xueqi Cheng. 2025. Leveraging LLMs for Utility-Focused
Annotation: Reducing Manual Effort for Retrieval and RAG.arXiv preprint
arXiv:2504.05220(2025).
[39] Tianjun Zhang, Shishir G Patil, Naman Jain, Sheng Shen, Matei Zaharia, Ion
Stoica, and Joseph E Gonzalez. 2024. Raft: Adapting language model to domain
specific rag.arXiv preprint arXiv:2403.10131(2024).
A Reproduced F1 Results
Table 3 presents our reproduction results evaluated using F1 score.
Compared to the LLM-judged results in Table 1, these results are
much closer to those reported in the original work [ 24]—PRAG
shows stronger performance than RAG. We attribute this discrep-
ancy to the limitations of the F1 metric.
As illustrated in Figure 8, F1 can be highly misleading: (i) it pe-
nalizes correct answers that include additional explanatory phrases;
(ii) it assigns high scores to factually incorrect answers that happen
to share surface tokens with the ground truth; and (iii) it is sen-
sitive to normalization choices, and inappropriate normalization
(e.g., punctuation handling) can lead to completely incorrect evalu-
ations. These issues can inflate PRAG’s apparent performance, as

The Role of Parametric Injection-A Systematic Study of Parametric Retrieval-Augmented Generation Conference acronym ’XX, June 03–05, 2018, Woodstock, NY
Table 3: Reproduction results of four methods, evaluated by F1 score. Bold numbers indicate the best performance under each
model, and the second-best results are underlined. CWQ denotes ComplexWebQuestions.
LLM Method2WikiMultihopQA HotpotQAPopQA CWQ Average
Compare Bridge Inference Compose Total Bridge Compare Total
LLaMA3.2-
1B-InstructVanilla 41.15 44.41 16.65 4.27 22.39 11.21 39.97 15.59 4.34 36.28 23.63
RAG 24.41 33.1925.659.47 20.92 20.22 41.31 25.95 15.95 37.49 25.46
PRAG45.70 45.7119.57 5.9326.4615.0548.6221.30 17.64 34.30 28.03
PRAG-Combine 34.03 38.49 25.14 11.0723.65 22.7046.78 28.47 31.87 40.39 30.26
Qwen2.5-
1.5B-InstructVanilla 19.71 31.21 13.49 5.07 15.00 8.66 27.90 11.14 5.55 34.31 17.20
RAG 21.02 28.84 19.45 6.52 16.74 20.43 44.65 24.72 9.99 28.23 22.06
PRAG 20.7033.9219.17 5.8617.6211.65 32.53 15.45 16.40 35.8320.91
PRAG-Combine23.4329.0822.41 9.1717.23 22.31 45.64 26.61 19.0730.8924.58
Qwen2.5-
7B-InstructVanilla 46.43 46.03 20.60 6.83 27.15 15.14 52.10 20.47 4.20 36.23 27.52
RAG 46.11 41.92 24.84 8.44 23.98 28.92 52.03 32.05 8.57 32.07 29.83
PRAG53.28 48.6422.0711.58 30.7718.2757.5724.72 19.92 45.7733.26
PRAG-Combine 47.62 39.7129.5910.83 27.1131.5053.62 35.08 27.5841.96 34.46
question: "What is Jacob Kraemer's occupation?"answer: ["actor", "actress", ...]output: " Jacob Kraemer is a Canadian actor."normalized_output: "Jacob Kraemer is a Canadian actor"F1 score:0.33(a)question: "What college did Bill Clinton attend that is in Eastern Time Zone?"answer: ["Georgetown University", ...]output: " University of Arkansas."normalized_output: "University of Arkansas"F1 score:0.4(b)question: "What is George Rankin's occupation?"answer: ["politician", "polit.", ...]output: "1. George Rankin was an Australian soldier and politician."normalized_output: "1"F1 score:0(c)
Figure 8: Examples illustrating the limitations of F1 score. (a) A correct answer with additional explanation is penalized,
yielding a low F1 score. (b) An incorrect answer sharing surface tokens with the ground truth receives a high F1 score. (c)
Inappropriate normalization leads to completely incorrect evaluations.
Passages: ... A speechwriter for the Trump administration’s Department of Homeland Security (DHS) has come under scrutiny after he was linked to hate speech online. Eric Lendrum compared the circumstances of American conservatives to that of enslaved people ...Question:Which government department appointed a speechwriter who was linked to hate speech online?Answer:Department of Homeland Security (DHS)Vanilla Output: 2019-04-15, when the Chinese Ministry of Education announced that itPRAG Output:TheDepartment of Homeland Security (DHS).
Figure 10: An example from the new-knowledge dataset.
Passages: ... killing at least 13 people during morning prayers, local authorities said. No one immediately claimed responsibility for the attack in the town of UnguwanMantau, in the state of Katsina, but such attacks are common in Nigeria’s northwestern and north-central regions where local herders and farmers often clash over limited access to land and water ...Question:In which Nigerian state did the mosque attack take place?Answer:KatsinaVanilla Output: Borno State.The attack took place in the town of Maiduguri, which is located in ...PRAG Output:Katsina State.
Figure 9: An example from the new-knowledge dataset.its parametric adaptation encourage concise, template-like outputs
that align fortuitously with F1’s token-matching behavior—without
necessarily improving factual correctness.
B Examples on New-Knowledge Dataset
Figure 9 and Figure 10 show two examples from our new-knowledge
dataset, where the questions involve facts that emerged after the
LLM’s knowledge cutoff. In both cases, Vanilla hallucinates and
produces factually incorrect answers. In contrast, PRAG—equipped
with document-specific parametric knowledge—successfully re-
trieves the correct information and generates accurate responses.
C Similarity Distribution of LoRA Msodules
Figure 11 and Figure 12 show the similarity distributions of LoRA
modules (averaged over all layers) for LLaMA3.2-1B-Instruct and
Qwen2.5-7B-Instruct, respectively. The trends align with our ob-
servations in Section 5.3.1: relevant pairs exhibit higher average
similarity than irrelevant ones, but the margin remains modest.
This suggests that parametric representations encode shared se-
mantic and factual content across related documents, yet fail to
fully isolate document-unique information.

Conference acronym ’XX, June 03–05, 2018, Woodstock, NY Minghao Tang et al.
0.55 0.60 0.65 0.70 0.75 0.80 0.85
Cosine Similarity02468101214DensitySimilarity Distribution of down_proj
Relevant Pairs
Irrelevant Pairs
0.45 0.50 0.55 0.60 0.65 0.70 0.75
Cosine Similarity02468101214Similarity Distribution of gate_proj
Relevant Pairs
Irrelevant Pairs
0.50 0.55 0.60 0.65 0.70 0.75
Cosine Similarity02468101214Similarity Distribution of up_proj
Relevant Pairs
Irrelevant Pairs
Figure 11: Similarity distribution of LoRA modules (averaged over all layers) for LLaMA3.2-1B-Instruct.
0.35 0.40 0.45 0.50 0.55 0.60 0.65 0.70
Cosine Similarity024681012DensitySimilarity Distribution of down_proj
Relevant Pairs
Irrelevant Pairs
0.30 0.35 0.40 0.45 0.50 0.55 0.60 0.65
Cosine Similarity024681012Similarity Distribution of gate_proj
Relevant Pairs
Irrelevant Pairs
0.30 0.35 0.40 0.45 0.50 0.55 0.60 0.65
Cosine Similarity024681012Similarity Distribution of up_proj
Relevant Pairs
Irrelevant Pairs
Figure 12: Similarity distribution of LoRA modules (averaged over all layers) for Qwen2.5-7B-Instruct.