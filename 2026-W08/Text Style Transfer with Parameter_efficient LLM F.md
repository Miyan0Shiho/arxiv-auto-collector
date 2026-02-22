# Text Style Transfer with Parameter-efficient LLM Finetuning and Round-trip Translation

**Authors**: Ruoxi Liu, Philipp Koehn

**Published**: 2026-02-16 18:52:43

**PDF URL**: [https://arxiv.org/pdf/2602.15013v1](https://arxiv.org/pdf/2602.15013v1)

## Abstract
This paper proposes a novel method for Text Style Transfer (TST) based on parameter-efficient fine-tuning of Large Language Models (LLMs). Addressing the scarcity of parallel corpora that map between styles, the study employs roundtrip translation to synthesize such parallel datasets from monolingual corpora. This approach creates 'neutralized' text devoid of stylistic attributes, essentially creating a shared input style at training-time and inference-time. Experimental results demonstrate consistent superiority of this method over zero-shot prompting and fewshot ICL techniques measured by BLEU scores and style accuracy scores across four investigated domains. Furthermore, the integration of retrieval-augmented generation (RAG) for terminology and name knowledge enhances robustness and stylistic consistency.

## Full Text


<!-- PDF content starts -->

Text Style Transfer with Parameter-efficient LLM Finetuning and
Round-trip Translation
Ruoxi Liu
Department of Computer Science
Johns Hopkins University
rliu79@jh.eduPhilipp Koehn
Department of Computer Science
Johns Hopkins University
phi@jhu.edu
Abstract
This paper proposes a novel method for Text
Style Transfer (TST) based on parameter-
efficient fine-tuning of Large Language
Models (LLMs). Addressing the scarcity of
parallel corpora that map between styles, the
study employs roundtrip translation to syn-
thesize such parallel datasets from mono-
lingual corpora. This approach creates
’neutralized’ text devoid of stylistic at-
tributes, essentially creating a shared input
style at training-time and inference-time.
Experimental results demonstrate consis-
tent superiority of this method over zero-
shot prompting and fewshot ICL techniques
measured by BLEU scores and style ac-
curacy scores across four investigated do-
mains. Furthermore, the integration of
retrieval-augmented generation (RAG) for
terminology and name knowledge enhances
robustness and stylistic consistency.
1 Introduction
Text Style Transfer (TST) is the task of rephras-
ing text by modifying stylistic attributes while
preserving its core attribute-independent seman-
tics and intent (Shen et al., 2017; Toshevska and
Gievska, 2024). These stylistic attributes encom-
pass formality, attitude, verbosity, preferred termi-
nology, and other characteristics inherent to the
text. A significant challenge in TST lies in the
scarcity of annotated parallel corpora, which hin-
ders the application of fully supervised learning or
finetuning methods (Pan et al., 2024) in most text
style domains.
Roundtrip translation is a machine translation
technique where a sentence is translated from one
language to a pivot language and then back to the
original language. It has been previously used
to evaluate MT system robustness and generation
quality (Somers, 2005; Moon et al., 2020). Prior
en-de
NMTde-en
NMTIn vain we roam: Each in
the end must call a
strange land home.Vergeblich wandern wir
umher: Am Ende muss
jeder ein fremdes Land
sein Zuhause nennen.Futilely drifting: Eventrually ,
every person's home will be
a strange place.en-de general domain
parallel corpora
Style-rich
English inputMT-translated
GermanRoundtrip-destylized
English output
TST Finetuning
Synthetic parallel
corpusFigure 1: Our proposed workflow for finetuning large
language models (LLMs) for text style transfer (TST)
using only non-parallel dataset in the target domain.
A bilingual general-domain parallel dataset is used
to train a pair of neural machine translation (NMT)
models capable of translating between English and a
pivot language. We then obtain machine-translated
style-neutral texts of the original in-domain texts by
roundtrip translating the in-domain set with the NMT
models. This enables supervised finetuning of LLMs
for TST, where we finetune LLMs for MT-output-
domain to target-domain transfer using the synthetic
parallel corpus.
work on TST has observed that roundtrip translat-
ing a sentence effectively diminishes stylistic at-
tributes specific to the author, yielding a neutral-
ized style while retaining the content (Sennrich
et al., 2016; Rabinovich et al., 2017). This ob-
servation motivates the use of roundtrip transla-
tion pipelines as autoencoders in many encoder-
decoder styled TST frameworks to extract destyl-
ized latent vectors from input text, so that style-
specific decoders can be trained in a supervised
fashion even when input style domains are unpre-
dictable (Prabhumoye et al., 2018).
In this paper, we propose a novel TST method
that adapts LLMs for style transfer tasks using
monolingual in-style corpora and roundtrip trans-
lation (Figure 1). Our workflow involves firstarXiv:2602.15013v1  [cs.CL]  16 Feb 2026

Stylistically neutral
Informal
Formal
Literary
ShakespeareanHe is really worried about it.
He’s losing sleep over it!
He is experiencing considerable
distress about this matter .
He poured rocks in the dungeon
of his mind.
He doth fret greatly upon it.Figure 2: Example sentences illustrating semantically
equivalent content in various styles. Outputs of our
roundtrip translation pipeline is considered as stylisti-
cally neutral.
training two neural machine translation models
that serve as the roundtrip translation pipeline, us-
ing large-scale general-domain bilingual parallel
corpora. We then roundtrip translate a monolin-
gual, stylistically consistent corpus using the pre-
trained NMT models to construct a style-neutral to
target-domain pseudo-parallel corpus. This corpus
can thus be used to finetune LLMs for TST tasks.
Furthermore, to enhance the model’s robustness
to unseen or complex style domains, we imple-
mented an inference-time workflow that roundtrip
translates queries before doing inference, improv-
ing training and inference time coherence (§3.1).
We evaluated (§4) our style transfer method on
several text styles with distinctive style features
(§4.1.1), and compared its performance against
two state-of-the-art methods: Few-shot In Con-
text Learning (ICL) and Automatic Post-Editing
(APE) (Liu et al., 2024b; Moon et al., 2022). Fol-
lowing prior research, style transfer quality is eval-
uated using BERT-based style classifiers trained
on held-out data and the BLEU score (Subrama-
nian et al., 2018; Wan et al., 2023; Aycock and
Bawden, 2024).
Ourmain contributionsare:
•Pseudo-parallel dataset construction
(§3.1). We propose a roundtrip translation
method for generation synthetic parallel
corpus, enabling TST with supervised
finetuning in domains lacking bitext.•Retrieval augmentation for finetuning and
inference(§3.2.3). We propose the use of
retrieval-augmentation forfinetuning, care-
fully experiment with RAG in both finetuning
and inference prompts, and validate its effec-
tiveness beyond prompting.
•Methods for TST-finetuning(§4). We
systematically evaluate finetuned TST-LLMs
employing several different models, prompts,
RAG methods, and inference methods, com-
pared against state-of-the-art baselines.
2 Related Work
Supervised TSTSeveral parallel corpora for
TST have been released (V oigt et al., 2018; Rao
and Tetreault, 2018) that motivated supervised
TST on these pre-selected domains with sufficient
parallel data, such as Jhamtani et al. (2017)’s work
on Shakespearizing modern English. Their ap-
proach is limited to domains with parallel corpora.
Unsupervised / Semi-supervised Text Style
TransferDue to the scarcity of parallel TST
data in most domains, one major focus of prior
TST research (Lai et al., 2021; Hu et al., 2022;
Nouri, 2022) is the seq2seq encoder-decoder mod-
els for unsupervised training with non-parallel
target-side data. Central to these frameworks are
effective disentanglement of latent representations
of styles (Nangi et al., 2021; V oigt et al., 2018) and
the preservation of original content through the
TST pipeline (Tian et al., 2018). There is recent
work on UTST frameworks using LLM prompting
and attention masking (Pan et al., 2024).
Roundtrip translation for TSTPrior works
observed that roundtrip translation tend to re-
duce authors’ stylistic features while preserving
the style-independent content (Prabhumoye et al.,
2018; Rabinovich et al., 2017). This observation
motivates the use of roundtrip translation as auto-
encoders to extract destylized latent vectors from
text with various input style domains that repre-
sent content. Style-specific decoders then trans-
form these latent vectors to output texts with the
same content and the target style (Prabhumoye
et al., 2018; Riley et al., 2021). In these settings,
roundtrip translation is believed to transform in-
stances in various domains to the same latent rep-
resentation, essentially turning the task of transfer-
ring from varying domains to the simpler task of
decoding destylized latent vectors to target style

Arbitrary Domain
InputRoundtrip
TranslationMT domain Pseudo-parallel corpus
from the R T workflow
Finetune
TST-LLMMT domain
query
K-shots &
terminologiesRAG
retrievalInference
Prompt A
Arbitrary
domain query
K-shots &
terminologiesInference
Prompt B
RAG
retrievalTarget domain
outputFigure 3:Our proposed workflow.We showtwo inference routesthat we tested on:route i(blue in figure) in-
volves first roundtrip translate the input to match the training-time input domains and then perform RAG-enhanced
TST-LLM inference with two retrievers we built (§3.2) on the intermediary text, where asroute ii(red in figure)
directly performs RAG-enhanced TST-LLM inference using the original input. Controlled experiments on these
methods demonstrate that roundtrip translating the input first significantly enhances model’s performance, bringing
especially considerable improvements facing stylistically diverse and complex queries. Findings in this experiment
are described in §4.4.
.
generation, which can be achieved in a supervised
fashion.
LLM-supported TSTRecent research indi-
cates that state-of-the-art Large Language Mod-
els (LLMs) possess the capability to perform TST
tasks when appropriately prompted or finetuned
(Liu et al., 2024a; Zhang et al., 2024; Mukherjee
et al., 2024). Prior works have developed prompt
learning methods for TST that use non-parallel
data (Liu et al., 2024a; Wan et al., 2023; Aycock
and Bawden, 2024; Zhang et al., 2024). These
strategies typically involve augmenting prompts
with retrieved data (Liu et al., 2024b; Zhang et al.,
2024) and a limited set of in-domain, non-parallel
examples (“shots”) (Chen, 2024; Liu et al., 2024a;
Bhandarkar et al., 2024) in optimized prompt con-
figurations (Liu et al., 2024a). However, these
methods are limited to prompts, lacking the abil-
ity to introduce parameter-level adjustments that
could enhance LLM adaptability to specific TST
or domain adaptation contexts.
Parameter-efficient finetuning for TSThas
been investigated very recently (Liu et al., 2024b;
Mukherjee et al., 2024), but only limited to do-
mains with existing parallel corpora.3 Methods
3.1 Roundtrip Translation
We propose a novel TST framework that adapts
LLMs for style transfer tasks using only mono-
lingual in-style corpora (Figure 3). We first train
a pair of neural machine translation (NMT) mod-
els using Marian (Junczys-Dowmunt et al., 2018)
and a large-scale generic bilingual corpus between
English and a selected pivot language. This pair
of generic NMT models constitutes a roundtrip
translation pipeline, which reduces stylistic fea-
tures of input texts with rich and diverse styles to
roundtrip translated style-neutral output. We use
this pipeline to build a parallel corpus that pairs
with its style-neutral equivalent. Then we finetune
a LLM on this dataset to warp specialize in the
transfer task to MT-destylized text to in-style text.
A potential issue with our method is that, during
finetuning we essentially provide supervision for
the transfer of text fromRT-destylized domainto
the target domain, rather thanarbitrary domain
to target domain supervision. We make such dis-
tinction since machine-translated texts tend to be
neutralized and style-homogenized, whereas arbi-
trary inference-time inputs may differ. To mitigate
this issue, we designed an inference-time work-

flow where the input sentence is also roundtrip-
translated to its stylistically neutral counterpart be-
fore processed by the finetuned LLM. We com-
pared direct inference and our RT-first inference
method in the experiments section (§4.4), and re-
port that RT-first inference yields noticeably better
generation quality when dealing with unseen text
styles.
3.2 RAG Retrievers for TST-LLM
Lewis et al. (2020) proposed a Retrieval Aug-
mented Generation (RAG) framework in which
a retriever-generator model is trained end-to-end
to enhance coherence between the pre-trained re-
triever and generator subsystems. Inspired by
this approach, we incorporate RAG into both
the finetuning and inference stages of our TST-
LLM approach to enhance the LLM’s adaption
to retrieval-enhanced prompts at inference-time,
unlike previous TST-with-LLM methods where
RAG is primarily considered a prompting tech-
nique (Liu et al., 2024a; Wan et al., 2023; Aycock
and Bawden, 2024; Zhang et al., 2024) and fine-
tuning experiments are largely limited to the zero-
shot strategy with various prompt templates (Liu
et al., 2024b; Mukherjee et al., 2024). In §4.3,
we present a comparative implementation of re-
trieval augmentation at both training time and in-
ference time, demonstrating that RAG introduced
at training-time aligns the inference-time retriever
with the generator, producing considerable im-
provements at test time.
3.2.1 Training-time Similarity-based
Example Retrieval
Our example retrieval augmentation method in-
volves obtaining sentence pairs as instructions for
how we would like the query to be transferred.
In order for the example transfer sentence pairs
to be instructive, we adapt a similarity-based re-
trieval method (Figure 4) to retrieve transfer sen-
tence pairs that are similar to the task objective,
using cosine distance obtained through the Faiss
vectorization library (Douze et al., 2024).
Since we providesentence pairsas examples,
an issue that naturally arises is whether to pro-
vide example pairs whosesource-sidesare simi-
lar to the query or thosetarget-sidesare similar
to the expected output. We consider this distinc-
tion necessary and vital to the quality of the re-
trieved shots. Consider the informal phrase "I’m
good". Transferring it to the formal domain wouldhave many valid answers, such as "I do not re-
quire anything further", "I am content with the
arrangement", or straightforwardly "I appreciate
your concern; I am in good health." Searching
with target-side would more likely providerele-
vantanswers, whereas searching with source-side
would potentially yield misleading examples.
Essentially this is the difference between
searching with thequestionsand searching with
theanswers. At training-time, providing ex-
amples pairs with relevant target-sides is rather
straightforward, since the actual output text (or
the "completion") is present. We constructed a
Faiss vector bank for the monolingual in-domain
corpus. Then, for each instance in the pseudo-
parallel dataset obtained from roundtrip transla-
tion, we take its target-side text, search for top-k
most similar examples excluding itself, and look
up the source-side counterparts of these retrieved
sentences to form example transfer sentence pairs
to be put into finetuning prompts.
3.2.2 Inference-time "Sketch-first" Example
Retrieval
At inference time when only the out-of-domain-
side input is present, we follow Wang et al.
(2022)’s schema to use a "sketch-first" example re-
trieval augmentation logic (Figure 4). We first per-
form few-shot inference withrandomly selected
examples to generate a sketch output that resem-
bles the in-domain transferred generation, though
with limited quality due to the randomly selected
shots. We then use the sketch as the query to re-
trieve examples with high similarity from the Faiss
vector bank to enhance the second-round inference
that yields the refined output. In §4.4, we report
on inference-time experiments on the inference-
time example retrieval augmentation methods de-
scribed above and the RT-first inference pipeline
described in §3.1.
3.2.3 Terminology and Name List Retrieval
Diction and word preferences are an important as-
pect of text style domains. The same concept or
object can be referred to by different terminologies
in different domains, such as "football" in British
English and "soccer" in American English, so con-
sistently using the correct terminology for the tar-
get domain is vital for semantic correctness and
style consistency. In literary translation domains,
there is a similar issue ofnaming consistency,
where machine-translated works may use seman-

Input (source-
side) textTarget-side
example set
Source-side
example setPseudo-parallel
dataset from R TSimilarity
Search
Index
LookupTerminology and
Name RetrieverSimilar Example
Retriever
TST-LLM
Target domain
"sketch" Random k-shotPseudo-parallel
dataset from R T
Prompt 1: Identify
terminologies or character
names in this sentence.Prompt 2: Find the counterpart of
this word in a similar sentence
from another translator.Prompting 1
Prompting 2
Terminology
Pair ListInput (source-
side) textWord preference
knowledge setTriggeringFigure 4:Retrieval augmentation workflow. Left (a):Similarity-based example retriever. We vectorize and
indexthe target-side textsof the parallel synthetic datasets for nearest-neighbor search. For each query, we first
do k-shot inference with the finetuned TST-LLM to obtain an "in-domain" sketch, which is used as search query
in the target-side dataset to obtain k most similar pairs. Note that this is forinference-time RAG. For finetuning
prompts, we can search with the target side texts directly without the need for an in-domain sketch.Right (b)
Terminology and name retriever: For each instance in the synthetic parallel datasets, the first LLM call extracts
relevant words from the source side, then the second call matches them with their counterparts in the target side,
yielding a terminology pair list for each domain. During inference, each input is checked against these term pairs;
where relevant matches are found, a concise guiding sentence is appended to the prompt.
tic translations and direct translations in different
contexts to refer to the same characters, causing
confusion and inconsistency (Matusov, 2019).
We improve our TST model’s terminology cor-
rectness and long-term consistency by retrieving
aterminology and name listfrom our pseudo-
parallel corpus, and add relevant domain-specific
term instructions to prompts when some trigger
words are present in the query (Figure 4). For
each data point in the pseudo-parallel corpus, we
first prompt a LLM with the source side of the
paired sentences and ask it to identify any domain-
specific terms or names in it. Then, we do a second
round of prompting with the target side of the sen-
tence pairs alongside the retrieved domain-specific
terms, and ask the LLM to find their counterparts.
Through this pipeline, we construct a list of source
domain to target domain preferred terminologies
pairs. If any of the source-side words are present
in the query, we add a one-sentence instruction
in the inference prompt that provides terminology
and name transfer guidance. Prompts we used are
in Appendix A.
3.3 Parameter Efficient LoRA Finetuning
LoRA (Low-Rank Adaptation of Large Language
Models) is an efficient approach that reduces com-
putational and memory costs by using low-rank
approximation techniques (Hu et al., 2021). The
LoRA approach involves freezing the pre-trained
model’s weight matrices and introducing trainable
low-rank decomposition matrices into the model’s
layers. This approach allows us to finetune the 7Band 8B LLMs with 2 NVIDIA A100 GPUs, each
with 81GB of memory. Hyperparameters and con-
figurations we used are put in Appendix B.
3.4 Evaluation: BLEU and Style
Classification Accuracy Score
We primarily evaluate two aspects of our models,
namely style transfer quality and content preserva-
tion ability. We train a BERT-based (Devlin et al.,
2018) style classifier for each style domain us-
ing held-out in-domain data, in the same fashion
as Liu et al. (2024b,a); Mukherjee et al. (2024)’s
prior works. The trained classifier classifies a
given text to be either in-domain or out-of-domain,
thus the generation from our TST models is tested
with these classifiers to yield a style classification
accuracy, as a measure of how well the generated
texts aligns with the target domain in terms of
text style. BLEU scores between generation and
source texts are used to evaluate to what extent the
original meaning is preserved after transfer.
4 Experiments
4.1 Experiment Setup
4.1.1 Datasets, Synthetic Data Generation,
and Baselines
A large-scale generic parallel training setis
used to train the Neural Machine Translation
model pairs for each pivot language. We used
Marian (Junczys-Dowmunt et al., 2018) for these
Neural Machine Translation models. Detailed
configurations we used are given in Appendix B.

Dataset Language # Sentence # Word
WMT24 en–de 75,991,652 1,160,839,966
WMT24 en–zh 72,192,512 857,631,464
IRS en monolingual 455,733 7,349,231
Treasury en monolingual 408,004 8,990,216
NCBI en monolingual 201,888 3,509,166
Literary en monolingual 105,030 3,643,974
Table 1:Datasets.The WMT24 datasets are used to
train generic NMT models for roundtrip translation.
We selected Chinese and German as the pivot lan-
guages.
Four monolingual style-consistent corpora are
roundtrip translated to construct pseudo-parallel
datasets for finetuning, which are: (a) corpus of
administrative documentation from the Internal
Revenue Service (IRS) website; (b) corpus of offi-
cial communication corpus from the U.S. Depart-
ment of Treasury; (c) scientific publications from
the National Center for Biotechnology Informa-
tion (NCBI) database; (d) the corpus of literary
translations of pre-modern Chinese texts by six
productive translators, including David Hawkes
and John Minford. Dataset sizes are presented in
Table 1. These domain-specific corpora served
as the foundation for creating parallel finetuning
datasets.
Baselines:First, We set up anIn Context
Learning (ICL)baseline method by prompting
the LLMs with the same prompt from our own
method (Figure 5), with the addition of few-
shot examples retrieved based on similarity and
the absence of finetuning. Furthermore, we train
an additional Marian NMT model for each do-
main on the roundtrip-translated versus in-domain
English corpora, so that it performs English-to-
English "translations" that brings MT-style texts
to in-domain. This is known asAutomatic Post-
editing (APE)in Machine Translation, where the
additional APE module learns to correct system-
atic errors in the MT system through NMT train-
ing. We consider this as another baseline to com-
pare against.
4.1.2 TST Prompt Templates
We experimented on three potential prompt tem-
plates for TST finetuning. Prompt details and ex-
periments on prompts are in Appendix A. After
testing and careful evaluation, we decided to use
the prompt template in Figure 5 throughout our ex-
periments.Rewrite the given sentence into the style of
[style name].
Here are [n] examples:
Input: [example input i]. Output: [example
output i]. ......
Note that you may want to rewrite "[input
term]" to "[output term]" for contextual consis-
tency.
Now go ahead: Input: [query input]. The [style
name] output:
Figure 5: The prompt template we use for Text Style
Transfer Finetuning. Performances of other prompts
that we experimented on are put in Appendix A.
4.2 Experiments on Pretrained LLMs
We experimented on various LLMs to evaluate
their potentials for TST finetuning with synthetic
parallel data. For all models, we performed
sketch-first 5-shot finetuning without any other
knowledge retrieval. A BERT classifier is trained
for each text style domain and used on the gener-
ated text to yield the style accuracy score for each
experimental group. Results are shown in Table 2.
Out of the four models we investigated,Llama-
3-8B-InstructandGorilla-openfunctions-v2
have the best overall performances across the
four tested style domains, with the finetuned
Gorilla LLM having the highest average BLEU
score and the finetuned Llama-3 LLM having the
highest average style accuracy score. We will
use Llama-3-8B-Instruct as the base model for
prompting and finetuning for other experiments in
the rest of this section.
4.3 Experiments on Retrieval Augmentation
Methods
Here we present the experiment results with re-
gards to various RAG methods that we used
during both finetuning and inference (Table 3).
The random k-shot example retrieval method re-
trieves k randompairsof style-neutral to target-
domain sentences for each finetuning prompt and
each inference prompt (Figure 5). Similar k-shot
method retrieve the k most similar examples pairs,
which is achieved throughdirect cosine distance
searchat finetuning time, and throughsketch-
first method(§3.2.2) at inference time. Terminol-
ogy and name retrieval are achieved by construct-
ing adomain-specific termpair bank(§3.2.3).
Note that these groups in Table 3 are using dif-

Pretrained LLMsIRS style Literary style Treasury style NCBI style
BLEU Acc. BLEU Acc. BLEU Acc. BLEU Acc.
RT output (no transfer) 22.53 0.391 21.90 0.172 24.15 0.245 19.87 0.354
5-shot ICL (baseline) 27.79 0.591 25.90 0.613 24.72 0.541 27.87 0.462
meta-llama/Llama-3.1-8B-Instruct48.89 0.82641.420.72145.220.81246.300.896
gorilla-llm/gorilla-openfunctions-v2 47.40 0.75642.310.66347.800.71449.620.823
mistralai/Mistral-7B iii 43.30 0.742 36.85 0.701 40.12 0.710 38.43 0.734
facebook/opt-2.7b 38.12 0.640 35.15 0.570 42.00 0.820 41.27 0.676
Table 2: TST Finetuning performance with Various Base LLMs (random 5-shot instructions finetuning). Pivot
language is chosen to be Chinese for this experiment. We evaluate the effectiveness of TST-finetuning through
comparing various finetuned LLMs against baseline, which we chose to be 5-shot ICL. We use Llama-3.1-8B-
Instruct for the baseline method. BLEU score for the raw sentence pairs from the roundtrip-translation workflow
is also presented for reference. All four tested models exhibit strong potential in performing TST tasks after
finetuning, with Llama-3.1-8B-Instruct and Gorilla-openfunctions-v2 having considerably higher performance in
both content preservation and style adaptation across the four tested domains.
ferent finetuning methodsanddifferent inference
methods, since we also include the retrieved in-
formation in the finetuning prompts. Sketch-first
similar 5-shot finetuning consistently outperforms
the prompting and zero-shot finetuning baselines
across the four tested domains, with a highest
BLEU score of 52.35 and highest Style Accuracy
score of 0.865 both in the Pre-modern Literary do-
main. The effect of example retrieval on the BLEU
score is more consistent and stable that its effect
on the style classification accuracy. For style clas-
sification accuracy, the similar 5-shot model is still
predominantly the best-performing model, though
random 3-shot and 5-shot models have a 0.030 -
0.037 higher classification acc. in the IRS domain
and the NCBI health domain. We attribute this to
the fact that the IRS and NCBI domainsare closer
to the general domainthan the Literary and Trea-
sury domains, making the classification of gener-
ated texts for these domains more nuanced and un-
predictable.
Looking into the generated text across the ex-
perimental groups and the style domains, we ob-
served that similarity-based n-shot finetuning is
much more stable than random n-shot finetuning,
especially for the Literary domain, where sentence
length, diction, and phrasing habits vary to a great
extent throughout the corpus. When provided with
irrelevant examples at inference time, such as one
word long sentence examples for long discourses
or descriptive sentences provided as examples for
character speeches, the examples can even mis-
lead the model and lower the generation quality
compared to zero-shot inference. Similarity-based3-shot and 5-shot finetuning, on the other hand,
exhibits a much more stable improvement in gen-
eration quality, as it always provides examples
with similar length and overlapping words with
the query sentence. It yields up to 12.22 increase
in BLEU score and 0.191 increase in style classi-
fication accuracy across the four tested style do-
mains.
We also observed that terminology and name re-
trieval has stronger influence on prompting than
on finetuning – adding the terminology paraphrase
guidance results in a 7.29% average improvement
on the Acc. score for 5-shot finetuning, and a
18.62% average improvement on the Acc. score
for 5-shot ICL.
4.4 Experiments on Inference Methods
We also conducted controlled experiments on var-
ious inference-time workflows. All inference
groups utilized the LLama3.1-8B-instruct model,
finetuned with the same 5-shot approach.They
only differ in inference methods.The 0-shot in-
ference setting employed inference prompts con-
taining only task descriptions without additional
knowledge. The RT-first inference method in-
volved roundtrip translation (RT) of queries to
align with the finetuning input domain (§3.1) be-
fore feeding them into the LLM. The similar k-
shot inference method retrieves and provides rel-
evant examples in a sketch-first manner, as elabo-
rated in §3.2.2.
Results indicate that both RT-first and similar-
shot approaches bring significant enhancements
to style classification accuracy, while similar-

RAG methodsIRS style Literary style Treasury style NCBI style
BLEU Acc. BLEU Acc. BLEU Acc. BLEU Acc.
5-shot ICL 27.79 0.591 25.90 0.613 24.72 0.541 27.87 0.462
APE with Marian 36.81 0.642 35.72 0.649 36.37 0.621 35.95 0.659
Zero-shot finetuning 42.39 0.793 40.39 0.742 41.43 0.826 39.30 0.742
Random 3-shot finetuning 47.23 0.839 39.96 0.732 44.41 0.796 42.07 0.823
Random 5-shot finetuning 48.89 0.826 41.42 0.721 45.22 0.812 46.300.896
Similar 3-shot finetuning 47.79 0.749 48.83 0.812 47.79 0.820 49.01 0.776
Similar 5-shot finetuning 49.500.796 52.35 0.86550.460.876 49.96 0.831
5-shot ICL w/
terminology and name retrieval28.53 0.672 26.25 0.669 26.69 0.729 29.31 0.586
Similar 5-shot finetuning w/
terminology and name retrieval49.280.895 52.61 0.93350.250.894 50.370.872
Table 3: TST performance with various retrieval augmentation methods and scale (Using Llama-3.1-8B-Instruct).
The ICL method prompts the LLM with k in-domain example sentences as context knowledge. Random k-shot
finetuning provides random examples at both finetuning and inference time; Similar k-shot provides similar ex-
amples for finetuning prompts through cosine distance search, and for inference prompts in a sketch-first manner
(§3.2.2). Terminology and name retrieval constructs a term pair bank, which is added to the prompt when triggered
(§3.2.3). Providing LLMs with examples at both training and inference time brings considerable improvements,
especially when providingsimilarexamples. 5-shot groups tend to have stronger effects on both BLEU score and
Acc. than 3-shot and 0-shot groups.
shot inference also yields a moderate improve-
ment in BLEU score. However, we observed
that roundtrip translation can reduce BLEU scores,
suggesting potential semantic drift when queries
are mapped to the MT-output style neutral domain.
The extent of this information loss is likely influ-
enced by several factors, includingpivot language
selection, thequality of NMT models, and the
complexity of the style. Despite this trade-off, the
substantial improvement in style classification ac-
curacy underscores the importance of the RT-first
workflow.
5 Conclusion
This study has established a robust method
for Text Style Transfer (TST) that leverages
parameter-efficient finetuning of Large Language
Models (LLMs) combined with roundtrip trans-
lation to address the challenges posed by the
scarcity of parallel corpora in most stylistic do-
mains. Through roundtrip translation, we pro-
duce synthesized pseudo-parallel texts that recon-
struct a supervised Text Style Transfer setting
from MT-neutralized domain to target style do-
main. The MT-neutralized style serves as a shared
input style, so that inputs with unseen stylistic fea-
tures better match the finetuned LLM at inference
time, enhancing adaptability and robustness whenfacing out-of-domain input sentences. Our ex-
periments across four distinct styles demonstrate
that the roundtrip translation augmented finetun-
ing method consistently outperforms state-of-the-
art approaches, such as In-Context Learning and
Automatic Post-Editing for TST.
We also found that retrieval-augmented genera-
tion (RAG) effectively enhances terminology and
name consistency within our roundtrip translation
augmented finetuning framework. Our compre-
hensive experiments show that incorporating re-
trieved examples and generation guidance helps
maintain long-term stylistic consistency and im-
proves overall generation quality. These findings
demonstrate that the application of knowledge and
example retrieval augmentation can go beyond
prompting.
Our TST finetuning method has the potential
to extend beyond single-domain adaptation. Fu-
ture work could explore multi-style transfer within
a single finetuned LLM and investigate more nu-
anced, non-binary style transfer tasks, such as for-
mality editing.
Limitations
The main limitations of our work are as follows:
•Semantic drift and error propagation.Our
method relies on machine translation models

Inference methodsIRS style Literary style Treasury style NCBI style
BLEU Acc. BLEU Acc. BLEU Acc. BLEU Acc.
0-shot inference 43.21 0.811 46.68 0.842 42.25 0.742 46.63 0.696
RT & random 5-shot inference 45.53 0.809 47.12 0.792 43.31 0.782 45.51 0.742
similar 5-shot inference48.730.829 52.33 0.82050.470.833 49.96 0.793
RT & similar 5-shot inference 46.280.895 51.61 0.93350.250.894 50.37 0.872
Table 4: TST Finetuning performance with various inference-time workflows. All groups are inferences with a
LLama3.1-8B-instruct that is finetuned with similar 5-shot and terminology RA from the previous experiment
(§4.3). 0-shot inference uses prompts that do not provide any additional knowledge besides task description. RT-
first inferences means we roundtrip translate the queries to match finetuning input domains (§3.1) before being
given to the LLMs. Results suggest a significant boost in style classification accuracy brought by RT-first and
similar shots, and a moderate improvement in BLEU score brought by similar shots.
to generate parallel finetuning datasets. As a
result, its performance depends on the qual-
ity of the underlying NMT systems and their
training data. We observed that when these
models introduce errors or cause semantic
drift during roundtrip translation, such inac-
curacies become embedded in the synthetic
parallel corpus used for finetuning. We ap-
plied post-processing steps to mitigate such
effects, and further efforts could also be made
to test various NMT methods or architectures
to find the most ideal configuration for the
TST task. These improvements and post-
editing works, however, are beyond the scope
of this study.
•Alternatives for the Current Roundtrip
Translation Pipeline. In this work, we pri-
marily used Marian to train the NMT mod-
els and did not explore alternative methods
or workflows for performing roundtrip trans-
lation. An intriguing potential alternative is
to employ large language models to perform
machine translation, either by ICL or fine-
tuning, which might yield better results com-
pared to the current Marian-based approach.
However, we did not test these alternative ap-
proaches in the current study due to the limit
of time and length.
•Limits to domains with available corpus.
Due to data availability constraints, our ex-
periments are conducted on six style do-
mains, which may not fully capture the range
of stylistic variations encountered in real-
world scenarios. This limitation could intro-
duce biases into our analysis and potentiallyrestrict the generalizability of our methods.
We selected domains that are as diverse and
distinctive as possible—from literary to gov-
ernmental and medical texts—in an effort to
enhance the overall robustness and applica-
bility of our method. We strive to enhance
the generalizability of our experiments and
demonstrate the effectiveness of our method
in different domains and conditions.
References
Seth Aycock and Rachel Bawden. 2024. Topic-
guided example selection for domain adaptation
in llm-based machine translation.
Avanti Bhandarkar, Ronald Wilson, Anushka
Swarup, and Damon Woodard. 2024. Emulat-
ing author style: A feasibility study of prompt-
enabled text stylization with off-the-shelf llms.
Dimitar F do Carmo, D Shterionov, Joss
Moorkens, Joachim Wagner, Murhaf Hossari,
Eric Paquin, Dag Schmidtke, Declan Groves,
and Andy Way. 2021. A review of the state-
of-the-art in automatic post-editing.Machine
Translation, 35:101–143.
Jianlin Chen. 2024. Lmstyle benchmark: Evaluat-
ing text style transfer for chatbots.
Yue Chen, Chen Huang, Yang Deng, Wenqiang
Lei, Dingnan Jin, Jia Liu, and Tat-Seng Chua.
2024. Style: Improving domain transferabil-
ity of asking clarification questions in large lan-
guage model powered conversational agents.

Jacob Devlin, Ming-Wei Chang, Kenton Lee, and
Kristina N. Toutanova. 2018. Bert: Pre-training
of deep bidirectional transformers for language
understanding.
Matthijs Douze, Alexandr Guzhva, Chengqi
Deng, Jeff Johnson, Gergely Szilvasy, Pierre-
Emmanuel Mazaré, Maria Lomeli, Lucas Hos-
seini, and Hervé Jégou. 2024. The faiss library.
Johannes Eschbach-Dymanus, Frank Essenberger,
Bianka Buschbeck, and Miriam Exel. 2024.
Exploring the effectiveness of llm domain adap-
tation for business it machine translation.
Junxian He, Xinyi Wang, Graham Neubig, and
Taylor Berg-Kirkpatrick. 2020. A probabilistic
formulation of unsupervised text style transfer.
Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan
Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang,
and Weizhu Chen. 2021. Lora: Low-rank adap-
tation of large language models.arXiv preprint
arXiv:2106.09685.
Zhiqiang Hu, Roy Ka-Wei Lee, Charu C. Ag-
garwal, and Aston Zhang. 2022. Text style
transfer: A review and experimental evaluation.
SIGKDD Explor. Newsl., 24(1):14–45.
Harsh Jhamtani, Varun Gangal, Eduard Hovy,
and Eric Nyberg. 2017. Shakespearizing mod-
ern language using copy-enriched sequence
to sequence models. InProceedings of the
Workshop on Stylistic Variation, pages 10–19,
Copenhagen, Denmark. Association for Com-
putational Linguistics.
Marcin Junczys-Dowmunt, Roman Grund-
kiewicz, Tomasz Dwojak, Hieu Hoang,
Kenneth Heafield, Tom Neckermann, Frank
Seide, Ulrich Germann, Alham Fikri Aji,
Nikolay Bogoychev, André F. T. Martins, and
Alexandra Birch. 2018. Marian: Fast neural
machine translation in c++.
Huiyuan Lai, Antonio Toral, and Malvina Nissim.
2021. Thank you BART! rewarding pre-trained
models improves formality style transfer. In
Proceedings of the 59th Annual Meeting of the
Association for Computational Linguistics and
the 11th International Joint Conference on Nat-
ural Language Processing (Volume 2: Short Pa-
pers), pages 484–494, Online. Association for
Computational Linguistics.Patrick Lewis, Ethan Perez, Aleksandra Piktus,
Fabio Petroni, Vladimir Karpukhin, Naman
Goyal, Heinrich Küttler, Mike Lewis, Wen-
tau Yih, Tim Rocktäschel, Sebastian Riedel,
and Douwe Kiela. 2020. Retrieval-augmented
generation for knowledge-intensive NLP tasks.
InAdvances in Neural Information Processing
Systems, volume 33, pages 9459–9474. Curran
Associates, Inc.
Jinpeng Li, Zekai Zhang, Quan Tu, Xin Cheng,
Dongyan Zhao, and Rui Yan. 2024. Stylechat:
Learning recitation-augmented memory in llms
for stylized dialogue generation.
Qingyi Liu, Jinghui Qin, Wenxuan Ye, Hao Mou,
Yuxuan He, and Keze Wang. 2024a. Adaptive
prompt routing for arbitrary text style transfer
with pre-trained language models.
Xinyue Liu, Harshita Diddee, and Daphne
Ippolito. 2024b. Customizing large lan-
guage model generation style using parameter-
efficient finetuning.
Yinhong Liu, Yimai Fang, David Vandyke, and
Nigel Collier. 2024c. Toad: Task-oriented au-
tomatic dialogs with diverse response styles.
Evgeny Matusov. 2019. The challenges of us-
ing neural machine translation for literature. In
Proceedings of the Qualities of Literary Ma-
chine Translation, pages 10–19, Dublin, Ire-
land. European Association for Machine Trans-
lation.
Hyeonseok Moon, Chanjun Park, Jaehyung Seo,
Sugyeong Eo, and Heuiseok Lim. 2022. An
automatic post editing with efficient and sim-
ple data generation method.IEEE Access,
10:21032–21040.
Jihyung Moon, Hyunchang Cho, and Eunjeong L.
Park. 2020. Revisiting round-trip translation for
quality estimation.CoRR, abs/2004.13937.
Sourabrata Mukherjee, Atul Kr Ojha, and Ond ˇrej
Dušek. 2024. Are large language models actu-
ally good at text style transfer?
Sharmila Reddy Nangi, Niyati Chhaya, Sopan
Khosla, Nikhil Kaushik, and Harshit Nyati.
2021. Counterfactuals to control latent disen-
tangled text representations for style transfer.

Jianmo Ni, Jiacheng Li, and Julian Mcauley. 2019.
Justifying recommendations using distantly-
labeled reviews and fine-grained aspects.
Nasim Nouri. 2022. Text style transfer via optimal
transport.
WMT24 Organizers. 2024. Findings of the 2024
conference on machine translation (WMT24).
InProceedings of the 2024 Conference on Ma-
chine Translation, TBD. Association for Com-
putational Linguistics.
Lei Pan, Yunshi Lan, Yang Li, and Weining Qian.
2024. Unsupervised text style transfer via llms
and attention masking with multi-way interac-
tions.
Shishir G. Patil, Tianjun Zhang, Xin Wang, and
Joseph E. Gonzalez. 2023. Gorilla: Large lan-
guage model connected with massive apis.
Shrimai Prabhumoye, Yulia Tsvetkov, Ruslan
Salakhutdinov, and Alan W Black. 2018. Style
transfer through back-translation.
Ella Rabinovich, Raj Nath Patel, Shachar Mirkin,
Lucia Specia, and Shuly Wintner. 2017. Per-
sonalized machine translation: Preserving orig-
inal author traits. InProceedings of the 15th
Conference of the European Chapter of the As-
sociation for Computational Linguistics: Vol-
ume 1, Long Papers, pages 1074–1084, Valen-
cia, Spain. Association for Computational Lin-
guistics.
Sudha Rao and Joel Tetreault. 2018. Dear sir or
madam, may I introduce the GYAFC dataset:
Corpus, benchmarks and metrics for formal-
ity style transfer. InProceedings of the 2018
Conference of the North American Chapter
of the Association for Computational Linguis-
tics: Human Language Technologies, Volume 1
(Long Papers), pages 129–140, New Orleans,
Louisiana. Association for Computational Lin-
guistics.
Emily Reif, Daphne Ippolito, Ann Yuan, Andy
Coenen, Chris Callison-Burch, and Jason Wei.
2021. A recipe for arbitrary text style transfer
with large language models.
Parker Riley, Noah Constant, Mandy Guo, Girish
Kumar, David Uthus, and Zarana Parekh. 2021.
Textsettr: Few-shot text style extraction and
tunable targeted restyling.Rico Sennrich, Barry Haddow, and Alexandra
Birch. 2016. Improving neural machine transla-
tion models with monolingual data. InProceed-
ings of the 54th Annual Meeting of the Associ-
ation for Computational Linguistics (Volume 1:
Long Papers), pages 86–96, Berlin, Germany.
Association for Computational Linguistics.
Tianxiao Shen, Tao Lei, Regina Barzilay, Tommi
Jaakkola, and Mit Csail. 2017. Style transfer
from non-parallel text by cross-alignment.
Harold Somers. 2005. Round-trip translation:
What is it good for? InProceedings of the
Australasian Language Technology Workshop
2005, pages 127–133.
Sandeep Subramanian, Guillaume Lample,
Eric Michael Smith, Ludovic Denoyer,
Marc’Aurelio Ranzato, and Y-Lan Boureau.
2018. Multiple-attribute text style transfer.
Zhen Tao, Dinghao Xi, Zhiyu Li, Liumin Tang,
and Wei Xu. 2024. Cat-llm: Prompting large
language models with text style definition for
chinese article-style transfer.
Youzhi Tian, Zhiting Hu, and Zhou Yu. 2018.
Structured content preservation for unsuper-
vised text style transfer.
Antonio Toral and Andy Way. 2018.What Level
of Quality Can Neural Machine Translation At-
tain on Literary Text?Springer International
Publishing, Cham.
Martina Toshevska and Sonja Gievska. 2024.
Large Language Models for Text Style Transfer:
Exploratory Analysis of Prompting and Knowl-
edge Augmentation Techniques.
Rob V oigt, David Jurgens, Vinodkumar Prab-
hakaran, Dan Jurafsky, and Yulia Tsvetkov.
2018. Rtgender: A corpus for studying differ-
ential responses to gender.
Zhen Wan, Yating Zhang, Yexiang Wang, Fei
Cheng, and Sadao Kurohashi. 2023. Refor-
mulating domain adaptation of large language
models as adapt-retrieve-revise: A case study
on chinese legal domain.
Yifan Wang, Zewei Sun, Shanbo Cheng, Weiguo
Zheng, and Mingxuan Wang. 2022. Controlling
styles in neural machine translation with activa-
tion prompt.

Chiyu Zhang, Honglong Cai, Yuezhang, Li,
Yuexin Wu, Le Hou, and Muhammad Abdul-
Mageed. 2024. Distilling text style transfer
with self-explanation from llms.
Zhirui Zhang, Shuo Ren, Shujie Liu, Jianyong
Wang, Peng Chen, Mu Li, Ming Zhou, and En-
hong Chen. 2018. Style transfer as unsuper-
vised machine translation.
A Prompt Templates
TST finetuning prompts:
We experimented on three potential prompt
templates for text style transfer (TST) finetuning
with synthetic parallel data (Table 5). These
prompts organize the query input sentence and
several example sentence pairs into a prompt,
with proper task descriptions and guidance for
the generation. Template (I) and (II) explicitly
states the rewriting task, but have different orders
of the example and query content. Template (III)
is a classic Machine Translation prompt template
with demonstrated effectiveness for Machine
Translation with LLM. By changing language
name to style domain name, we adapt it to guide
LLM for text style transfer task.
In this experiment (Table 6), we conducted ran-
dom 5-shot finetuning with terminology retrieval
on Llama3.1-8B-Instruct with the different tem-
plates, while leaving other conditions unchanged.
Template (I) has the overall highest score in the
two tested domains. This is potentially because
the query input in template (I) is closer to the
end, while in the second template there are many
examples separating the query input and the
expected generation output. The phrasing of
the text style transfer task in prompt (I) is also
more ideal than the simplified version in template
(III) and better describes the task. Noticeably,
template (III), though simple and concise, also
has consistently high style accuracy scores in the
tested domains.
Terminology RAG prompts:
We retrieved domain-specific term and name pair
lists for each domain to enhance TST perfor-
mances, by calling the LLM twice for each in-
stance in the synthetic parallel corpus. The
prompts we used are shown in Table 7.Table 5: Prompts for TST finetuning
Prompt
Template
IndexPrompt Template Text
I Rewrite the following sentence into
the style of [style name]. Here are [n]
examples: Input: [example input i].
Output: [example output i]. Note that
word [input term] should be rewritten
to [output term] for contextual consis-
tency. Now go ahead: Input: [query
input]. The [style name] output:
II Rewrite the following sentence into
the style of [style name]: Input: [query
input]. Here are [n] examples: Input:
[example input i]. Output: [example
output i]. Note that word [input term]
should be rewritten to [output term] for
contextual consistency.
III Note that word [input term] should be
rewritten to [output term] for contex-
tual consistency. General domain: [ex-
ample input i]. [style name] domain:
[example output i]. general domain:
Input: [query input]. [style name] do-
main:
Prompt templates for LLM style transfer finetuning and
inference.The sentences containing [example input i] and
[example output i] placeholders are removed from the tem-
plates for zero-shot finetuning and inference.

TemplateIRS domain Literary domain
BLEU Acc. BLEU Acc.
Baseline 22.53 0.391 21.90 0.172
Template I48.89 0.826 41.420.721
Template II 45.40 0.542 38.29 0.563
Template III 46.28 0.781 37.710.794
Table 6: BLEU and acc. score across IRS and Liter-
ary domains for three potential templates. Template
(I) has consistently higher BLEU score compared to
template (II) and (III), indicating superior ability in
content-preservation. Both Template (I) and (III) have
stablly high style classification accuracy, indicating ro-
bust ability in transferring to target style.
Table 7: Prompts for terminology retrieval
Prompt
typePrompt Text
First round Identify terminologies or character
names in the sentence and return
in comma separated format, without
any additional explanation. Sentence:
[source-side sentence]. Terminologies
and names:
Second
roundFind the counterpart of the word
[source-side retrieved word] in the fol-
lowing sentence and return a single
word, without any additional explana-
tion. Sentence: [target-side sentence]:
Prompts for terminology retrieval. The first prompt re-
trieves a list of terminologies and names from the source side
sentence of each parallel instance, and for each of these re-
trieved words, the second prompt retrieves its counterpart in
the corresponding target side sentence.
B Hyperparameters and experiment
configurations
LoRA finetuning hyperparameters:
We set the learning rate to 2e-4, rank for the
low-rank approximation is set to 512, the scaling
factor is set to 256, and we use float16 data type.
A dropout rate of 0.05 is applied. We save and
evaluate the model every 2000 steps.
Marian Configurations:
We used the Marian framework for the roundtrip
translation NMT models. In our system we used
the Transformer architecture with R2L Reranking,
with learning rate 0.0001, 49500 BPE operations,
and step size 20000.