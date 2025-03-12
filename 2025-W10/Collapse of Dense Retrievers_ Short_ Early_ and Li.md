# Collapse of Dense Retrievers: Short, Early, and Literal Biases Outranking Factual Evidence

**Authors**: Mohsen Fayyaz, Ali Modarressi, Hinrich Schuetze, Nanyun Peng

**Published**: 2025-03-06 23:23:13

**PDF URL**: [http://arxiv.org/pdf/2503.05037v1](http://arxiv.org/pdf/2503.05037v1)

## Abstract
Dense retrieval models are commonly used in Information Retrieval (IR)
applications, such as Retrieval-Augmented Generation (RAG). Since they often
serve as the first step in these systems, their robustness is critical to avoid
failures. In this work, by repurposing a relation extraction dataset (e.g.
Re-DocRED), we design controlled experiments to quantify the impact of
heuristic biases, such as favoring shorter documents, in retrievers like
Dragon+ and Contriever. Our findings reveal significant vulnerabilities:
retrievers often rely on superficial patterns like over-prioritizing document
beginnings, shorter documents, repeated entities, and literal matches.
Additionally, they tend to overlook whether the document contains the query's
answer, lacking deep semantic understanding. Notably, when multiple biases
combine, models exhibit catastrophic performance degradation, selecting the
answer-containing document in less than 3% of cases over a biased document
without the answer. Furthermore, we show that these biases have direct
consequences for downstream applications like RAG, where retrieval-preferred
documents can mislead LLMs, resulting in a 34% performance drop than not
providing any documents at all.

## Full Text


<!-- PDF content starts -->

Collapse of Dense Retrievers:
Short, Early, and Literal Biases Outranking Factual Evidence
Mohsen Fayyaz1Ali Modarressi2,3Hinrich Schütze2,3Nanyun Peng1
1University of California, Los Angeles
2CIS, LMU Munich3Munich Center for Machine Learning
mohsenfayyaz@cs.ucla.edu amodaresi@cis.lmu.de violetpeng@cs.ucla.edu
Abstract
Dense retrieval models are commonly used in
Information Retrieval (IR) applications, such as
Retrieval-Augmented Generation (RAG). Since
they often serve as the first step in these sys-
tems, their robustness is critical to avoid fail-
ures. In this work, by repurposing a relation
extraction dataset (e.g. Re-DocRED), we de-
sign controlled experiments to quantify the
impact of heuristic biases, such as favoring
shorter documents, in retrievers like Dragon+
and Contriever. Our findings reveal significant
vulnerabilities: retrievers often rely on super-
ficial patterns like over-prioritizing document
beginnings, shorter documents, repeated enti-
ties, and literal matches . Additionally, they
tend to overlook whether the document con-
tains the query’s answer. Notably, when multi-
ple biases combine, models exhibit catastrophic
performance degradation , selecting the answer-
containing document in less than 3% of cases
over a synthetic biased document without the
answer. Furthermore, we show that these biases
have direct consequences for downstream ap-
plications like RAG , where retrieval-preferred
documents can mislead LLMs, resulting in a
34% performance drop than not providing any
documents at all.1
1 Introduction
Retrieval-based language models have demon-
strated strong performance on a range of
knowledge-intensive NLP tasks (Lewis et al., 2020;
Asai et al., 2023; Gao et al., 2024). At the core of
these models is a retriever that identifies relevant
context to ground the generated output. Dense re-
trieval methods such as Contriever (Izacard et al.,
2021)—where passages or documents are stored as
learned embeddings—are especially appealing for
their scalability across large knowledge bases and
handling lexical gaps (Ni et al., 2022; Shao et al.,
1∗Code and benchmark dataset are available at
https://huggingface.co/datasets/mohsenfayyaz/ColDeR.
Brevity Bias ⬇
(Shorter Doc Adv .)
Answer Importance ⬆
(Answer's Existence)
Repetition Bias ⬇
(Head Repetition Adv .)Position Bias ⬇
(Early F ocus)Literal Bias ⬇
(Lexical Matching)
05101520
Model Dragon+ Dragon R oBER Ta 
Contriever MSMAR CO
RetroMAE MSMAR CO FT
COCO -DR Base MSMAR COFigure 1: Paired t-test statistics comparing retriever
scores between document pairs ( D1vs.D2) across five
evaluation aspects. Document pairs are designed for
controlled experiments shown in Table 1. Positive val-
ues indicate a retriever’s preference for the more biased
document in each bias scenario, while for answer impor-
tance, they reflect a preference for answer-containing
documents. The results show that retrieval biases often
outweigh the impact of answer presence.
2024), compared to alternatives like BM25 (Robert-
son and Zaragoza, 2009) or ColBERT(Khattab and
Zaharia, 2020). Despite their widespread use, rel-
atively little is understood about how these dense
models encode and organize information, leaving
key questions about their robustness against adver-
sarial attacks unanswered.
Existing evaluations of retrieval models often
focus on downstream task performance, as seen in
benchmarks like BEIR (Thakur et al., 2021), with-
out probing the underlying behavior of retrievers.
Some studies have analyzed specific issues in in-
formation retrieval (IR) models, such as position
bias (Coelho et al., 2024) or lexical overlap (Ram
et al., 2023).
In this work, we systematically study multiple
biases’ impact on retrievers—both individually andarXiv:2503.05037v1  [cs.CL]  6 Mar 2025

Document 1 (Higher Query Document Similarity Score) -D1 Document 2 (Lower Query Document Similarity Score) -D2Answer ImpactQuery: What is the sister city of Leonessa ?
Document: Leonessa istwinned with theFrench town ofGonesse .
Its population in 2008 was around 2,700 . Situated in a small plain at the foot of Mt.
Terminillo .....Query: What is the sister city of Leonessa ?
Document: Leonessa is a town and comune in the far northeastern part of the Province
of Rieti in the Lazio region of central Italy .
Its population in 2008 was around 2,700 . Situated in a small plain at the foot of Mt.
Terminillo .....Position BiasQuery: Which country is Wony ongSung a citizen of?
Document: Wony ongSung (born 1950s ),South Korean professorofelectronic
engineering Won - yong is a Korean masculine given name ..... People with this name
include : Kang Won - yong ( 1917 – 2006 ) ..... , South Korean swimmerQuery: Which country is Wony ongSung a citizen of?
Document: Won - yong is a Korean masculine given name ..... People with this name
include : .... Jung Won - yong ( born 1992 ) , South Korean swimmer Wony ongSung (
born 1950s ),South Korean professorofelectronic engineeringLiteral BiasQuery: When was Seyhunborn?
Document: Seyhun,(August22,1920 –May 26,2014 )wasanIranian architect,
sculp tor,painter ,scholar andprofessor. He studied fine arts at .....Query: When was Seyhunborn?
Document: Houshang Seyhoun ,(August22,1920 –May 26,2014 )wasanIranian
architect,sculp tor,painter ,scholar andprofessor. He studied fine arts at .....Brevity BiasQuery: What series is Lost Verizonpart of?
Document: "Lost Verizon"isthesecondepisode ofTheSimp sons ’twen tiethseason.Query: What series is Lost Verizonpart of?
Document: "Lost Verizon"isthesecondepisode ofTheSimp sons ’twen tiethseason.
It first aired on the Fox network in the United States on October 5 , 2008 . Bart becomes
jealous of his friends and their cell phones . Working at a golf course , Bart takes the
cell phone of Denis Leary .....Repetition BiasQuery: Where was James Paul Maherborn?
Document: Born inBrook lyn,New York ,Mahergraduated from St.Patrick ’s
Academy inBrook lyn.James Paul Maher( November 3 , 1865 – July 31 , 1946 ) was a
U.S. Representative from New York . Maherwas elected as a Democrat to the Sixty -
second and to the four succeeding Congresses ( March 4 , 1911 – March 4 , 1921 ) .Query: Where was James Paul Maherborn?
Document: Born inBrook lyn,New York ,Mahergraduated from St.Patrick ’s
Academy inBrook lyn. Apprenticed to the hatter ’s trade , he moved to Danbury ,
Connecticut in 1887 and was employed as a journeyman hatter . He became treasurer of
the United Hatters of North America in 1897 .Foil vs. Evide.Query: Who is the publisher of Assassin’sCreed Unity ?
Document: "Assassin’sCreed Unity " "Assassin’sCreed Unity "Assassin’sCreed
Unity received mixed reviews upon its release .Query: Who is the publisher of Assassin’sCreed Unity ?
Document: Isa is a town and Local Government Area in the state of Sokoto in Nigeria
. It shares borders with ..... Assassin’sCreed Unity isanaction -adventure video
game developed byUbisoft Mon treal andpublished byUbisoft . Isa is a town and Local
Government Area in the state of Sokoto in Nigeria . It shares borders with .....
Table 1: Examples from our framework highlighting Evidence, Head Entity, and TailEntity. In all cases, retrieval
models favor Document 1 over Document 2, assigning higher retrieval scores accordingly. (Explained in §3.3)
in combination—for the first time. To enable fine-
grained control over document structure and factual
positioning, we repurpose a document-level rela-
tion extraction dataset (Re-DocRED (Tan et al.,
2022)).
We first investigate biases individually, identify-
ing tendencies such as an over-prioritization of
document beginnings, document brevity, repeti-
tion of matching entities, and literal matches at
the expense of ignoring answer presence. Our sta-
tistical approach, illustrated in Figure 1, allows for
comparative analysis across different biases. Ad-
ditionally, we explore the interplay between these
biases and propose an adversarial benchmark that
combines multiple vulnerabilities.
We further study combining multiple biases
and reveal concerning patterns in current retriever
architectures. When exposed to multiple in-
teracting biases, even top-performing mod-
els exhibit dramatic degradation , selecting the
answer-containing document over the foil docu-
ment—filled with biases—less than 3% of the time.
Moreover, we demonstrate that these biases can be
exploited to manipulate Retrieval-Augmented
Generation (RAG) , causing retrievers to favor mis-
leading or adversarially constructed documents,
which misguide LLMs into using incorrect infor-
mation and ultimately degrade its performance.2 Related Work
Benchmarking in Information Retrieval Pop-
ular benchmarks like BEIR (Thakur et al., 2021;
Guo et al., 2021; Petroni et al., 2021; Muennighoff
et al., 2023) have played a crucial role in evaluating
retrieval models across diverse datasets and tasks.
In addition to general IR benchmarks, domain-
specific benchmarks such as COIR (Li et al., 2024)
for code retrieval and LitSearch (Ajith et al., 2024)
for scientific literature search address retrieval chal-
lenges in specialized domains. While these bench-
marks have advanced the evaluation of IR models,
they primarily focus on downstream performance
rather than conducting systematic analyses of bi-
ases inherent in retrieval systems.
Information Retrieval Model Analysis Prior
work in information retrieval has explored various
dimensions of retrieval performance, including po-
sitional biases (Coelho et al., 2024). Studies have
also examined how dense retrievers exhibit biases
towards common entities and struggle with OOD
scenarios (Sciavolino et al., 2021). Furthermore,
analysis by projecting representations to the vo-
cabulary space has shown that supervised dense
retrievers tend to learn relying heavily on lexical
overlap during training (Ram et al., 2023). Sim-
ilarly, BehnamGhader et al. (2023) has indicated
that Dense Passage Retrieval (DPR) models often

fail to retrieve statements requiring reasoning be-
yond surface-level similarity. Furthermore, neu-
ral IR models have been shown to exhibit over-
penalization for extra information, where adding a
relevant sentence to a document can unexpectedly
decrease its ranking (Usuha et al., 2024). Addi-
tionally, Reichman and Heck (2024) takes a mech-
anistic approach to analyze the impact of DPR fine-
tuning, showing that while fine-tuned models gain
better access to pre-trained knowledge, their re-
trieval capabilities remain constrained by the pre-
existing knowledge in their base models. Further,
MacAvaney et al. (2022) provides a framework for
analyzing neural IR models, identifying key biases
and sensitivities in these models.
Adversarial Attacks in Information Retrieval
Numerous studies have explored various dimen-
sions of robustness in information retrieval, in-
cluding aspects related to adversarial robustness
(Liu et al., 2024). Adversarial perturbations, for
instance, have been shown to significantly degrade
BERT-based rankers’ performance, revealing their
brittleness to subtle modifications (Wang et al.,
2022). Existing retrieval attack methods primar-
ily encompass corpus poisoning (Lin et al., 2024;
Zhong et al., 2023), backdoor attacks (Long et al.,
2024), and encoding attacks (Boucher et al., 2023).
While previous work has analyzed some retrieval
biases, most studies focus on task-specific super-
vised models and a single aspect in isolation. Our
work provides a comprehensive comparative anal-
ysis of popular retrieval models across multiple
dimensions of vulnerability. We systematically in-
vestigate how these biases interact and affect the
retrieval capabilities of dense retrievers. By re-
purposing a relation extraction dataset, we gain
precise control over factual information in docu-
ments, enabling a rigorous evaluation of retrieval
robustness. This multi-dimensional approach pro-
vides a nuanced understanding of the strengths and
weaknesses of dense retrievers.
3 Experiments
3.1 A Framework for Identifying Biases in
Retrievers
To gain fine-grained control over the facts present
in a document, we take a novel approach by repur-
posing a relation extraction dataset that provides
relation-level fact granularity. This enables a struc-
tured analysis of retrieval biases by explicitly link-
ing queries to individual factual statements.Model pooling nDCG@10 Recall@10
Dragon RoBERTa cls 0.55 0.75
Dragon+ cls 0.54 0.74
COCO-DR Base MSMARCO cls 0.50 0.71
Contriever MSMARCO avg 0.50 0.71
RetroMAE MSMARCO FT cls 0.48 0.68
Contriever avg 0.25 0.41
Table 2: Models’ performance on NQ dataset with test
set queries and 2,681,468 corpus size.
One such dataset is DocRED (Yao et al., 2019),
a relation extraction dataset constructed from
Wikipedia and Wikidata. DocRED consists of
human-annotated triplets ( head entity, relation, tail
entity)—for example, (Albert Einstein, educated
at, University of Zurich). However, DocRED suf-
fers from a significant false negative issue, as many
valid relations are missing from the annotations.
To address this, we use Re-DocRED (Tan et al.,
2022), a re-annotated version of DocRED that re-
covers missing facts, leading to more complete and
reliable annotations.
To construct a retrieval dataset from Re-
DocRED, we map each relation to a query tem-
plate. For example, for the relation "educated at,"
we use the template "Where was {Head Entity}
educated?" This transformation allows us to sys-
tematically examine how retrievers handle different
types of factual queries.
The answers to these queries are the tail entities
found in the evidence sentences provided by the
dataset. For our analysis, we ensure that each query
has a single evidence sentence ( Sev) within the
original document ( S∈Dorig) that contains both
the head and tail entities. This constraint makes
the sentence self-contained, allowing for precise
control over the document structure in subsequent
sections. We also introduce the notation S+h
−tfor
sentences in Dorigthat contain the head entity but
not the tail entity, and S−h
−tfor sentences that do
not contain either entity. In each of the following
sections, we will use this notation to construct a
pair of document sets, D1andD2, enabling a sys-
tematic investigation of retrieval score variations
and potential biases. As a result, for each of our
six analysis settings, we compile 250 queries, each
with a single corresponding gold document, based
on the test and validation sets of Re-DocRED.
3.2 Models Performance & Bias Discovery
First, we evaluate several dense retrievers on the
NQ dataset (Kwiatkowski et al., 2019), compar-
ing their performance using nDCG@10 and Re-

Figure 2: Visualization of the contribution of each query and document token to the final retrieval score using
DecompX. Literal Bias reflects the model’s preference for exact word matches, such as failing to match "esteban
goemz" with "estevao gomes." Position Bias indicates a preference for entities earlier in the document receiving
more attention. Repetition Bias shows that repeating an entity multiple times increases its score. Lastly, Answer
Importance demonstrates that the query’s answer entity receives less attention compared to head entity matches.
call@10 metrics. Table 2 shows that Dragon mod-
els lead in performance, and the significant im-
provement of fine-tuned Contriever over its unsu-
pervised counterpart highlights the importance of
supervision and task-specific adaptation. Models
also differ in pooling mechanisms, with Contriever
using average pooling and others using CLS pool-
ing. For details, refer to the appendix A.1.
In our preliminary analysis, we utilized De-
compX (Modarressi et al., 2023, 2022), a method
that decomposes the representations of encoder-
based models such as BERT into their constituent
token-based representations. By applying De-
compX to the embeddings generated by dense re-
trievers, we obtain decomposed representations for
both the query and the document. Instead of using
the original embeddings, we compute the similarity
score via a dot product of the decomposed vectors.
This approach enables us to visualize the contribu-
tion of each query and document token to the final
similarity score as a heatmap (Figure 2), revealing
biases in token-level interactions.
In our preliminary error analysis of 60 retrieval
failure examples, we identified potential biases
and limitations in the models (Table A.3). Fig-
ure 2 highlights some of these biases, such as Lit-
eral Bias, where the term "esteban gomez" fails to
match "estevao gomez," reflecting a preference for
exact matching. In subsequent sections, we design
experiments and perform statistical tests to evaluate
these observed biases.
3.3 Bias Types in Dense Retrieval
The following experiments are meticulously de-
signed to control for all other factors and biases,isolating the specific bias under evaluation.
3.3.1 Answer Importance
An effective retrieval model should accurately iden-
tify the query’s intent. It should retrieve relevant
documents that address the query, rather than just
matching entities. To assess whether dense retrieval
models truly recognize the presence of answers or
merely focus on entity matching, we developed a
controlled experimental framework. Our experi-
mental design contrasts two carefully constructed
document types. 1. Document with Evidence :
Contains a leading evidence sentence with both the
head entity and the tail entity (answer). 2. Docu-
ment without Evidence D2:Contains a leading
sentence with only the head entity but no tail.
D1:=Sev+X
S−h
−t∈DorigS−h
−t
D2:=S+h
−t+X
S−h
−t∈DorigS−h
−t(1)
Here, S+h
−tis another sentence from Dorigthat re-
places the original evidence Sevwhile containing
the head entity but not containing the tail entity
to isolate the impact of answer presence. The re-
mainder of both documents consists of neutral sen-
tences S−h
−t∈Dorig, carefully filtered to exclude
any sentences containing similar head relations or
tail entities. This ensures the answer information
appears exclusively in the leading sentence of the
evidence document. We strategically positioned the
key sentences at the beginning of both documents
to mitigate potential position bias effects, which
we analyze in subsequent sections. An example of
this setup is presented in Table 1 (Answer Impact).

5
 0 5 10 15
Paired t-T est StatisticContriever 
COCO-DR Base MSMARCO
Contriever MSMARCO
RetroMAE MSMARCO FT
Dragon RoBERT a 
Dragon+ Model-5.92
9.59
10.07
10.13
11.98
12.69Answer Importance:
Answer-Present vs. Answer-Absent DocsFigure 3: Paired t-test statistics comparing dot product
similarity between the first sentence containing both
head and tail (Answer) entities versus only the head
entity, with 95% CI error bars. Higher values indicate
recognition of the answer’s importance.
To quantify the models’ ability to distin-
guish between these document types, we employ
Paired t-Test2to analyze the difference in simi-
larity scores. The t-test statistic (t) is calculated
as:
t=¯d
SE(¯d)=Average Difference
Standard Error(2)
where ¯d=mean (R(D1)−R(D2))is the
mean difference between paired observations3, and
SE(¯d)is the standard error of these differences4.
A positive t-statistic indicates higher scores for
D1documents, while negative values suggest a
preference for D2documents. In this scenario,
positive values are desirable as they indicate the
model prefers D1which contains the answer over
D2which does not.
As shown in Figure 3, our analysis reveals
variations across models. Dragon+ and Dragon-
RoBERTa demonstrate superior tail recognition,
achieving the highest positive t-statistics. In con-
trast, Contriever exhibits poor performance, yield-
ing negative t-statistics that indicate a failure to
properly distinguish answer-containing passages.
The vanilla Contriever’s underwhelming perfor-
mance can be attributed to its unsupervised training
methodology, which differs from models trained
on MS MARCO (Bajaj et al., 2018). While MS
MARCO provides supervised training with ex-
plicit query-passage relevance labels, Contriever
employs unsupervised contrastive learning. It gen-
erates positive pairs through data augmentation
2Using ttest_rel function of SciPy (Virtanen et al., 2020).
3Ris the retriever’s score
4SE=σ√n
1 2 3 4 5 6 7 8 9 10
Evidence Sentence Position20
15
10
5
0Paired t-T est Statistic
w.r.t. Initial Evidence Position
Position Bias:
Moving Evidence
RetroMAE MSMARCO FT
Dragon+ 
Dragon RoBERT a 
Contriever MSMARCO
COCO-DR Base MSMARCOFigure 4: Paired t-test statistics comparing the effect of
moving the evidence sentence position within the docu-
ment to keeping it in the first position. Negative values
indicate a bias towards the beginning of the document.
from document segments and derives negative ex-
amples implicitly via in-batch sampling from other
texts. This training approach, while efficient for
general text representation, appears insufficient for
developing the fine-grained discrimination needed
to understand query intent in retrieval tasks.
3.3.2 Position Bias
Position bias refers to the preference of retrieval
models for information located in specific positions
within a document, typically favoring content at the
beginning over content appearing later. This bias
is problematic as it may lead to the underrepre-
sentation of relevant information that is positioned
deeper within documents, thus reducing the overall
retrieval quality and fairness.
Our analysis reveals a strong positional bias in
dense retrievers, with models consistently priori-
tizing information at the beginning of documents.
As shown in Figure 4, we conducted paired t-
tests comparing retrieval scores when the evidence
sentence is placed at different positions to scores
when it is placed at the document’s beginning
(R(Di)−R(D1)).
D1:=Sev+1S−h
−t+2S−h
−t+3S−h
−t+...+nS−h
−t
D2:=1S−h
−t+Sev+2S−h
−t+3S−h
−t+...+nS−h
−t
D3:=1S−h
−t+2S−h
−t+Sev+3S−h
−t+...+nS−h
−t(3)
To ensure fairness, the examples were curated
so that the remaining content was free of any ev-
idence or head entity ( S−h
−t) like the last section.
This design ensured that the evidence’s position
was the sole factor under evaluation. The consis-
tently negative t-statistics across models in Fig-

ModelContriever
MSMARCODragon+
Q1D1Q2D2
long long long short +21.05 +21.04
short long +22.04 +13.40
short short long short +4.62 +9.04
short long +14.37 +16.62
Table 3: Paired t-test statistics (p-values < 0.05) compar-
ing retrieval scores between exact name matches ( Q1-
D1) and variant name pairs ( Q2-D2). Positive statistics
indicate a preference for exact literal matches over se-
mantically equivalent name variants (e.g., “US”-“US”
over “US”-“United States”). (All models in Table A.7.)
ure 4 confirm a strong bias favoring content at doc-
ument beginnings.5This bias is most pronounced
in Dragon-RoBERTa and Contriever-MSMARCO,
which show the most negative t-statistics, indicat-
ing severe degradation in recognizing evidence fur-
ther into the document. While Dragon+ and Retro-
MAE perform better, their negative t-statistics still
confirm position bias in these models.
These findings align with recent research by
Coelho et al. (2024), who demonstrated that po-
sitional biases emerge during the contrastive pre-
training phase and worsened through fine-tuning on
MS MARCO dataset with T5 (Raffel et al., 2020)
and RepLLaMA (Ma et al., 2023) models. This can
significantly impact retrieval performance when rel-
evant information appears later in documents.
3.3.3 Literal Bias
Retrieval models should ideally recognize semantic
equivalence across different surface forms of the
same entity. For instance, a robust model should
understand that "Gomes" and "Gomez" refer to
the same person, or that "US" and "United States"
represent the same entity. However, our analysis
reveals that current models exhibit a strong bias
toward exact literal matches rather than semantic
matching.
In our dataset, each head entity can be repre-
sented by multiple alternative names. To investi-
gate literal bias, we created different combinations
of query and document by replacing all head enti-
ties with the shortest or longest name variants as
illustrated in Table 1 (Literal Bias). For example,
an entity might be represented as "NYC" (shortest)
or "New York City" (longest), allowing us to test
how the model performs when matching different
5Fig. 1 shows the impact of evidence placement (beginning
vs. end), detailed in Appendix A, with an example in Table 1.combinations of these representations.
Table 3 presents the paired t-test statistics com-
paring different combinations of name selections
in queries and documents. The results consistently
show positive statistics when Query 1 and Docu-
ment 1 contain similar name representations. For
our subsequent analysis of bias interplay, we specif-
ically examine the comparison between two scenar-
ios (Figure A.1): one where both query and docu-
ment use the shortest name variant (short-short) ver-
sus cases where the query uses the short name but
the document contains the long name variant (short-
long). This corresponds to +14.37 and +16.62 in
Table 3 for Contriever and Dragon+, respectively.6
3.3.4 Brevity Bias
Brevity bias refers to the tendency of retrievers
to favor concise text, such as a single evidence
sentence, over longer documents that include the
same evidence alongside additional context. This
bias is problematic because retrievers may favor
a shorter, non-relevant document over a relevant
one. We will discuss this potential hazard further
in Section 3.5.
Here, we performed paired t-tests to compare the
similarity scores of queries with two sets of docu-
ments: (1)Single Evidence , consisting of only the
evidence sentence, and (2)Evidence+Document ,
consisting of the evidence sentence followed by the
rest of the document. The examples are carefully
selected to ensure the evidence sentence includes
both the head and tail entity and the rest of the doc-
ument contains no repetition of the head entity or
additional evidence.
D1:=Sev
D2:=Sev+X
S−h
−t∈DorigS−h
−t(4)
Figure 1 and A.4, illustrate the paired t-test
statistics, where significant positive values indi-
cate a strong bias toward brevity, as models assign
higher scores to concise texts ( D1) than to longer
ones with the same evidence ( D2). This behavior
likely stems from the way dense passage retrievers
compress document representations. Most retriev-
ers use either a mean-pooling strategy or a [CLS]
token-based method. Both methods struggle with
integrating useful evidence into the representation
6We avoid long-long combinations to control for confound-
ing effects, as they span multiple tokens and may introduce
repetition bias due to token overlap

(0,/uni00A01] (1,/uni00A02] (2,/uni00A03] (3,/uni00A010]
Number/uni00A0of/uni00A0Head/uni00A0Entity/uni00A0Mentions(230,/uni00A0512]
(190,/uni00A0230]
(160,/uni00A0190]
(130,/uni00A0160]Document/uni00A0Length1.11 1.36 1.52 1.57
1.20 1.44 1.64 1.63
1.21 1.53 1.61 1.66
1.28 1.54 1.64 1.68Average/uni00A0Retrieval/uni00A0Score
Contriever/uni00A0MSMARCO
1.21.31.41.51.6
Figure 5: The average retrieval score of Contriever MS-
MARCO increases with head entity repetitions but de-
creases with document length (all models in Figure A.5).
when unrelated content is present, leading to a “pol-
lution effect.” As a result, the additional context
in longer documents dilutes the importance of the
evidence, causing retrievers to favor concise input.
3.3.5 Repetition Bias
Repetition bias refers to the tendency of retrieval
models to prioritize documents or passages with
repetitive content, particularly repeated mentions
of head entities present in the query. This bias is
problematic as it may skew retrieval results toward
redundant or verbose documents, undermining the
goal of surfacing concise and diverse information.
To analyze repetition bias, we conducted an ex-
periment evaluating the average retrieval dot prod-
uct score of the models for samples with varying
document lengths and head entity repetitions (Fig-
ure 5 and A.5). A key concern is that longer doc-
uments naturally have a higher chance of lexical
overlap with the query, as they may contain more
repeated mentions of the head entity. This makes
it difficult to disentangle the effects of document
length from the number of entity repetitions. There-
fore, we structure our analysis to separately exam-
ine these two factors. Our findings (Figure 5) reveal
that the retrieval score increases with the number
of head entity mentions, indicating a preference
for documents with repeated entities. Conversely,
the retrieval score decreases as document length
increases, suggesting that longer documents are pe-
nalized despite potential relevance. Figure A.5 in
the appendix generalizes these observations across
all models. This experiment highlights the trade-off
between repetition and document length, emphasiz-
ing the need for retrieval systems to balance these
factors to mitigate bias.
We further explored this phenomenon through
the results shown in Figures 1 and A.3. Here, weModel AccuracyPaired t-Test
Statisticp-value
Contriever 0.4% -34.58 < 0.01
RetroMAE MSMARCO FT 0.4% -41.49 < 0.01
Contriever MSMARCO 0.8% -42.25 < 0.01
Dragon RoBERTa 0.8% -36.53 < 0.01
Dragon+ 1.2% -40.94 < 0.01
COCO-DR Base MSMARCO 2.4% -32.92 < 0.01
Table 4: The accuracy and paired t-test comparing a foil
document (exploiting biases but lacking the answer) to
a second document with evidence embedded in unre-
lated sentences. All retrieval models perform extremely
poorly (<3% accuracy), highlighting their inability to
distinguish biased distractors from genuine evidence.
performed paired t-tests to compare the dot product
similarity scores of queries with two sets of docu-
ments: (1)More Heads , comprising an evidence
sentence and two sentences containing head men-
tions but no tails, and (2)Fewer Heads , comprising
an evidence sentence and two sentences without
head or tail mentions from the document (Table 1).
D1:=Sev+S+h
−t+S+h
−t
D2:=Sev+S−h
−t+S−h
−t(5)
Positive paired t-test values indicate higher simi-
larity for sentences with more head mentions (Fig-
ure A.3). The results strongly suggest that the
model favors sentences with repeated heads, con-
firming the presence of repetition bias.
3.4 Interplay Between Bias Types
To understand how different biases interact and
amplify retrieval model weaknesses, we conduct a
systematic analysis using a controlled 250-sample
dataset across all experiments. This consistent sam-
ple size ensures comparability of paired t-test statis-
tics across bias types and provides a robust basis
for evaluating their interplay.
As illustrated in Figure 1, the paired t-test results
reveal that brevity bias, literal bias, and position
bias are the most problematic for dense retrievers.
In contrast, repetition bias, while still detrimental,
exhibits a relatively lower impact, suggesting that
models are slightly more robust against this type
of bias. Answer importance demonstrates an ac-
ceptable distinction between evidence-containing
and no-evidence documents. However, the scores
are not as strong as one would expect from models
designed for accurate answer retrieval, highlighting
the need for further improvement in this area.
To further investigate the compounded effects of
multiple biases, we conducted another experiment

MODEL Poison Doc Foil Doc No Doc Evidence Doc
gpt-4o-mini 32.0% 44.0% 52.0% 88.0%
gpt-4o 30.8% 62.8% 64.8% 93.6%
Table 5: RAG accuracy when using different document
versions as references. The poisoned document, pre-
ferred by retrievers 100% of the time (Table A.4), re-
sults in worse performance than providing no document,
highlighting the impact of retriever biases on RAG.
that combines several bias types into a single chal-
lenging setup. In this experiment, we created two
document types. 1) Foil Document with Multiple
Biases: This document contains multiple biases,
such as repetition and position biases. It includes
two repeated mentions of the head entity in the
opening sentence, followed by a sentence that men-
tions the head but not the tail (answer). So it does
not include the evidence. 2) Evidence Document
with Unrelated Content: This document includes
four unrelated sentences from another document,
followed by the evidence sentence with both the
head and tail entities. The document ends with
the same four unrelated sentences. An example is
shown in Table 1 (Foil vs. Evide.).7
D1:= 2×h+S+h
−t
D2:= 4×˜S−h
−t+Sev+ 4×˜S−h
−t(6)
Table 4 presents the accuracy (proportion of
times the model prefers D2overD1), paired t-test
statistics, and p-values. The results are striking: all
models exhibit extremely poor performance, with
accuracy dropping below 3%. The paired t-test
statistics are highly negative across all models, in-
dicating a consistent preference for foil documents
over the correct evidence-containing ones. This
outcome highlights the severity of bias interplay
and its detrimental impact on model reliability. Fur-
thermore, a sufficient number of biased documents
can potentially cause the model to select all top-k
documents from only biased results.
3.5 Impact on RAG
To assess the impact of the identified vulnerabilities
on RAG systems, we use GPT-4o models (OpenAI
et al., 2024) and provide them with different ver-
sions of the reference document for a given query.
Additionally, we construct a poisoned document
by modifying the foil document from §3.4, intro-
ducing a poisoned evidence sentence (Table A.5).
7˜Sare sentences from an unrelated documentSpecifically, we generate this sentence using GPT-
4o by replacing the tail entity with a contextually
plausible but entirely incorrect entity. This ap-
proach ensures that the poisoned document both
exploits the previously discussed retrieval biases
and contains an incorrect answer to the query.8
D1:= 2×h+S+h
−t+S+h
+PoisonTail
D2:= 4×˜S−h
−t+Sev+ 4×˜S−h
−t(7)
Table 5 reports the RAG accuracy,9showing that,
as expected, providing the evidence document en-
ables the LLM to achieve high accuracy. However,
since retrievers prefer the foil document from §3.4,
which lacks evidence, LLM performance drops
to levels near10the no-document condition. This
preference is concerning, as it allows biases to be
exploited, making certain documents more likely to
be retrieved despite embedding incorrect informa-
tion. This is evident with the poisoned document,
which degrades performance even worse than pre-
senting no document by introducing false facts. In
summary, retriever biases can mislead RAG sys-
tems by providing poisoned or non-informative
documents , ultimately harming performance.
4 Conclusions
In this work, we introduced a comprehensive frame-
work for analyzing biases in dense retrieval mod-
els. By leveraging a relation extraction dataset
(Re-DocRED), we constructed a diverse set of con-
trolled experiments to isolate and evaluate specific
biases, including literal, position, repetition, and
brevity biases as well as the answer’s importance.
Our findings reveal that retrieval models often
prioritize superficial patterns, such as exact string
matches, repetitive content, or information posi-
tioned early in documents, over deeper semantic un-
derstanding and the existence of the answer. More-
over, when multiple biases combine, retriever per-
formance deteriorates dramatically.
Furthermore, Our analysis shows that retriever
biases can undermine RAG’s reliability by favor-
ing poisoned or non-informative documents over
evidence-containing ones, leading to degraded per-
formance of LLMs. These findings underscore the
need for dense retrieval models that are robust to bi-
ases and capable of prioritizing semantic relevance.
8Despite this, retrievers prefer the poisoned document over
the evidence document in 100% of cases (Table A.4).
9Evaluated using GPT-4o. Prompts in Table A.6
10Slightly lower, as the model sometimes abstains by stat-
ing, “The document does not provide information.”

Limitations
Quality of the Relation Extraction Dataset Our
framework relies on a relation extraction dataset,
making both annotation accuracy (precision) and
completeness (recall) critical. We use Re-DocRED,
which addresses annotation issues in DocRED, but
it may still contain imperfections that introduce
minor noise into our experiments. To mitigate this,
we employ statistical tests and report error mar-
gins and p-values to ensure the robustness of our
findings.
Limitations of RAG Evaluation by LLMs In
our RAG experiments, we utilized GPT-4o models
and carefully designed prompts (Table A.6) to poi-
son documents, generate answers using RAG, and
evaluate the results against gold-standard answers.
Although GPT-4o is one of the most advanced mod-
els available, it is not infallible and may introduce
some variance in the RAG results and evaluations.
Nevertheless, we believe the observed trends and
findings remain valid given the model’s high per-
formance and the consistency of our experimental
setup.
References
Anirudh Ajith, Mengzhou Xia, Alexis Chevalier, Tanya
Goyal, Danqi Chen, and Tianyu Gao. 2024. Lit-
Search: A retrieval benchmark for scientific literature
search. In Proceedings of the 2024 Conference on
Empirical Methods in Natural Language Processing ,
pages 15068–15083, Miami, Florida, USA. Associa-
tion for Computational Linguistics.
Akari Asai, Sewon Min, Zexuan Zhong, and Danqi
Chen. 2023. Retrieval-based language models and
applications. In Proceedings of the 61st Annual Meet-
ing of the Association for Computational Linguistics
(Volume 6: Tutorial Abstracts) , pages 41–46, Toronto,
Canada. Association for Computational Linguistics.
Payal Bajaj, Daniel Campos, Nick Craswell, Li Deng,
Jianfeng Gao, Xiaodong Liu, Rangan Majumder,
Andrew McNamara, Bhaskar Mitra, Tri Nguyen,
Mir Rosenberg, Xia Song, Alina Stoica, Saurabh
Tiwary, and Tong Wang. 2018. Ms marco: A human
generated machine reading comprehension dataset.
Preprint , arXiv:1611.09268.
Parishad BehnamGhader, Santiago Miret, and Siva
Reddy. 2023. Can retriever-augmented language
models reason? the blame game between the re-
triever and the language model. In Findings of the
Association for Computational Linguistics: EMNLP
2023 , pages 15492–15509, Singapore. Association
for Computational Linguistics.Nicholas Boucher, Luca Pajola, Ilia Shumailov, Ross
Anderson, and Mauro Conti. 2023. Boosting big
brother: Attacking search engines with encodings. In
Proceedings of the 26th International Symposium on
Research in Attacks, Intrusions and Defenses , RAID
’23, page 700–713, New York, NY , USA. Association
for Computing Machinery.
João Coelho, Bruno Martins, Joao Magalhaes, Jamie
Callan, and Chenyan Xiong. 2024. Dwell in the
beginning: How language models embed long docu-
ments for dense retrieval. In Proceedings of the 62nd
Annual Meeting of the Association for Computational
Linguistics (Volume 2: Short Papers) , pages 370–377,
Bangkok, Thailand. Association for Computational
Linguistics.
Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang Jia,
Jinliu Pan, Yuxi Bi, Yi Dai, Jiawei Sun, Meng Wang,
and Haofen Wang. 2024. Retrieval-augmented gener-
ation for large language models: A survey. Preprint ,
arXiv:2312.10997.
Mandy Guo, Yinfei Yang, Daniel Cer, Qinlan Shen, and
Noah Constant. 2021. MultiReQA: A cross-domain
evaluation forRetrieval question answering models.
InProceedings of the Second Workshop on Domain
Adaptation for NLP , pages 94–104, Kyiv, Ukraine.
Association for Computational Linguistics.
Gautier Izacard, Mathilde Caron, Lucas Hosseini, Se-
bastian Riedel, Piotr Bojanowski, Armand Joulin,
and Edouard Grave. 2021. Unsupervised dense infor-
mation retrieval with contrastive learning.
Omar Khattab and Matei Zaharia. 2020. Colbert: Effi-
cient and effective passage search via contextualized
late interaction over bert. In Proceedings of the 43rd
International ACM SIGIR Conference on Research
and Development in Information Retrieval , SIGIR
’20, page 39–48, New York, NY , USA. Association
for Computing Machinery.
Tom Kwiatkowski, Jennimaria Palomaki, Olivia Red-
field, Michael Collins, Ankur Parikh, Chris Alberti,
Danielle Epstein, Illia Polosukhin, Jacob Devlin, Ken-
ton Lee, Kristina Toutanova, Llion Jones, Matthew
Kelcey, Ming-Wei Chang, Andrew M. Dai, Jakob
Uszkoreit, Quoc Le, and Slav Petrov. 2019. Natu-
ral questions: A benchmark for question answering
research. Transactions of the Association for Compu-
tational Linguistics , 7:452–466.
Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio
Petroni, Vladimir Karpukhin, Naman Goyal, Hein-
rich Küttler, Mike Lewis, Wen-tau Yih, Tim Rock-
täschel, Sebastian Riedel, and Douwe Kiela. 2020.
Retrieval-augmented generation for knowledge-
intensive nlp tasks. In Advances in Neural Infor-
mation Processing Systems , volume 33, pages 9459–
9474. Curran Associates, Inc.
Xiangyang Li, Kuicai Dong, Yi Quan Lee, Wei Xia,
Yichun Yin, Hao Zhang, Yong Liu, Yasheng Wang,
and Ruiming Tang. 2024. Coir: A comprehensive

benchmark for code information retrieval models.
Preprint , arXiv:2407.02883.
Sheng-Chieh Lin, Akari Asai, Minghan Li, Barlas Oguz,
Jimmy Lin, Yashar Mehdad, Wen-tau Yih, and Xilun
Chen. 2023. How to train your dragon: Diverse aug-
mentation towards generalizable dense retrieval. In
Findings of the Association for Computational Lin-
guistics: EMNLP 2023 , pages 6385–6400, Singapore.
Association for Computational Linguistics.
Zilong Lin, Zhengyi Li, Xiaojing Liao, XiaoFeng Wang,
and Xiaozhong Liu. 2024. Mawseo: Adversarial
wiki search poisoning for illicit online promotion. In
2024 IEEE Symposium on Security and Privacy (SP) ,
pages 388–406.
Yu-An Liu, Ruqing Zhang, Jiafeng Guo, and Maarten
de Rijke. 2024. Robust information retrieval. In
Proceedings of the 47th International ACM SIGIR
Conference on Research and Development in Infor-
mation Retrieval , SIGIR ’24, page 3009–3012, New
York, NY , USA. Association for Computing Machin-
ery.
Quanyu Long, Yue Deng, LeiLei Gan, Wenya Wang,
and Sinno Jialin Pan. 2024. Whispers in grammars:
Injecting covert backdoors to compromise dense re-
trieval systems. Preprint , arXiv:2402.13532.
Xueguang Ma, Liang Wang, Nan Yang, Furu Wei, and
Jimmy Lin. 2023. Fine-tuning llama for multi-stage
text retrieval. arXiv:2310.08319 .
Sean MacAvaney, Sergey Feldman, Nazli Goharian,
Doug Downey, and Arman Cohan. 2022. ABNIRML:
Analyzing the behavior of neural IR models. Trans-
actions of the Association for Computational Linguis-
tics, 10:224–239.
Ali Modarressi, Mohsen Fayyaz, Ehsan Aghazadeh,
Yadollah Yaghoobzadeh, and Mohammad Taher Pile-
hvar. 2023. DecompX: Explaining transformers deci-
sions by propagating token decomposition. In Pro-
ceedings of the 61st Annual Meeting of the Associa-
tion for Computational Linguistics (Volume 1: Long
Papers) , pages 2649–2664, Toronto, Canada. Associ-
ation for Computational Linguistics.
Ali Modarressi, Mohsen Fayyaz, Yadollah
Yaghoobzadeh, and Mohammad Taher Pile-
hvar. 2022. GlobEnc: Quantifying global token
attribution by incorporating the whole encoder
layer in transformers. In Proceedings of the 2022
Conference of the North American Chapter of the
Association for Computational Linguistics: Human
Language Technologies , pages 258–271, Seattle,
United States. Association for Computational
Linguistics.
Niklas Muennighoff, Nouamane Tazi, Loic Magne, and
Nils Reimers. 2023. MTEB: Massive text embedding
benchmark. In Proceedings of the 17th Conference
of the European Chapter of the Association for Com-
putational Linguistics , pages 2014–2037, Dubrovnik,
Croatia. Association for Computational Linguistics.Jianmo Ni, Chen Qu, Jing Lu, Zhuyun Dai, Gustavo Her-
nandez Abrego, Ji Ma, Vincent Zhao, Yi Luan, Keith
Hall, Ming-Wei Chang, et al. 2022. Large dual en-
coders are generalizable retrievers. In Proceedings
of the 2022 Conference on Empirical Methods in
Natural Language Processing , pages 9844–9855.
OpenAI, :, Aaron Hurst, Adam Lerer, Adam P. Goucher,
Adam Perelman, Aditya Ramesh, Aidan Clark,
AJ Ostrow, Akila Welihinda, Alan Hayes, Alec
Radford, Aleksander M ˛ adry, Alex Baker-Whitcomb,
Alex Beutel, Alex Borzunov, Alex Carney, Alex
Chow, Alex Kirillov, Alex Nichol, Alex Paino, Alex
Renzin, Alex Tachard Passos, Alexander Kirillov,
Alexi Christakis, Alexis Conneau, Ali Kamali, Allan
Jabri, Allison Moyer, Allison Tam, Amadou Crookes,
Amin Tootoochian, Amin Tootoonchian, Ananya
Kumar, Andrea Vallone, Andrej Karpathy, Andrew
Braunstein, Andrew Cann, Andrew Codispoti, An-
drew Galu, Andrew Kondrich, Andrew Tulloch, An-
drey Mishchenko, Angela Baek, Angela Jiang, An-
toine Pelisse, Antonia Woodford, Anuj Gosalia, Arka
Dhar, Ashley Pantuliano, Avi Nayak, Avital Oliver,
Barret Zoph, Behrooz Ghorbani, Ben Leimberger,
Ben Rossen, Ben Sokolowsky, Ben Wang, Benjamin
Zweig, Beth Hoover, Blake Samic, Bob McGrew,
Bobby Spero, Bogo Giertler, Bowen Cheng, Brad
Lightcap, Brandon Walkin, Brendan Quinn, Brian
Guarraci, Brian Hsu, Bright Kellogg, Brydon East-
man, Camillo Lugaresi, Carroll Wainwright, Cary
Bassin, Cary Hudson, Casey Chu, Chad Nelson,
Chak Li, Chan Jun Shern, Channing Conger, Char-
lotte Barette, Chelsea V oss, Chen Ding, Cheng Lu,
Chong Zhang, Chris Beaumont, Chris Hallacy, Chris
Koch, Christian Gibson, Christina Kim, Christine
Choi, Christine McLeavey, Christopher Hesse, Clau-
dia Fischer, Clemens Winter, Coley Czarnecki, Colin
Jarvis, Colin Wei, Constantin Koumouzelis, Dane
Sherburn, Daniel Kappler, Daniel Levin, Daniel Levy,
David Carr, David Farhi, David Mely, David Robin-
son, David Sasaki, Denny Jin, Dev Valladares, Dim-
itris Tsipras, Doug Li, Duc Phong Nguyen, Duncan
Findlay, Edede Oiwoh, Edmund Wong, Ehsan As-
dar, Elizabeth Proehl, Elizabeth Yang, Eric Antonow,
Eric Kramer, Eric Peterson, Eric Sigler, Eric Wal-
lace, Eugene Brevdo, Evan Mays, Farzad Khorasani,
Felipe Petroski Such, Filippo Raso, Francis Zhang,
Fred von Lohmann, Freddie Sulit, Gabriel Goh,
Gene Oden, Geoff Salmon, Giulio Starace, Greg
Brockman, Hadi Salman, Haiming Bao, Haitang
Hu, Hannah Wong, Haoyu Wang, Heather Schmidt,
Heather Whitney, Heewoo Jun, Hendrik Kirchner,
Henrique Ponde de Oliveira Pinto, Hongyu Ren,
Huiwen Chang, Hyung Won Chung, Ian Kivlichan,
Ian O’Connell, Ian O’Connell, Ian Osband, Ian Sil-
ber, Ian Sohl, Ibrahim Okuyucu, Ikai Lan, Ilya
Kostrikov, Ilya Sutskever, Ingmar Kanitscheider,
Ishaan Gulrajani, Jacob Coxon, Jacob Menick, Jakub
Pachocki, James Aung, James Betker, James Crooks,
James Lennon, Jamie Kiros, Jan Leike, Jane Park,
Jason Kwon, Jason Phang, Jason Teplitz, Jason
Wei, Jason Wolfe, Jay Chen, Jeff Harris, Jenia Var-
avva, Jessica Gan Lee, Jessica Shieh, Ji Lin, Jiahui
Yu, Jiayi Weng, Jie Tang, Jieqi Yu, Joanne Jang,

Joaquin Quinonero Candela, Joe Beutler, Joe Lan-
ders, Joel Parish, Johannes Heidecke, John Schul-
man, Jonathan Lachman, Jonathan McKay, Jonathan
Uesato, Jonathan Ward, Jong Wook Kim, Joost
Huizinga, Jordan Sitkin, Jos Kraaijeveld, Josh Gross,
Josh Kaplan, Josh Snyder, Joshua Achiam, Joy Jiao,
Joyce Lee, Juntang Zhuang, Justyn Harriman, Kai
Fricke, Kai Hayashi, Karan Singhal, Katy Shi, Kavin
Karthik, Kayla Wood, Kendra Rimbach, Kenny Hsu,
Kenny Nguyen, Keren Gu-Lemberg, Kevin Button,
Kevin Liu, Kiel Howe, Krithika Muthukumar, Kyle
Luther, Lama Ahmad, Larry Kai, Lauren Itow, Lau-
ren Workman, Leher Pathak, Leo Chen, Li Jing, Lia
Guy, Liam Fedus, Liang Zhou, Lien Mamitsuka, Lil-
ian Weng, Lindsay McCallum, Lindsey Held, Long
Ouyang, Louis Feuvrier, Lu Zhang, Lukas Kon-
draciuk, Lukasz Kaiser, Luke Hewitt, Luke Metz,
Lyric Doshi, Mada Aflak, Maddie Simens, Madelaine
Boyd, Madeleine Thompson, Marat Dukhan, Mark
Chen, Mark Gray, Mark Hudnall, Marvin Zhang,
Marwan Aljubeh, Mateusz Litwin, Matthew Zeng,
Max Johnson, Maya Shetty, Mayank Gupta, Meghan
Shah, Mehmet Yatbaz, Meng Jia Yang, Mengchao
Zhong, Mia Glaese, Mianna Chen, Michael Jan-
ner, Michael Lampe, Michael Petrov, Michael Wu,
Michele Wang, Michelle Fradin, Michelle Pokrass,
Miguel Castro, Miguel Oom Temudo de Castro,
Mikhail Pavlov, Miles Brundage, Miles Wang, Mi-
nal Khan, Mira Murati, Mo Bavarian, Molly Lin,
Murat Yesildal, Nacho Soto, Natalia Gimelshein, Na-
talie Cone, Natalie Staudacher, Natalie Summers,
Natan LaFontaine, Neil Chowdhury, Nick Ryder,
Nick Stathas, Nick Turley, Nik Tezak, Niko Felix,
Nithanth Kudige, Nitish Keskar, Noah Deutsch, Noel
Bundick, Nora Puckett, Ofir Nachum, Ola Okelola,
Oleg Boiko, Oleg Murk, Oliver Jaffe, Olivia Watkins,
Olivier Godement, Owen Campbell-Moore, Patrick
Chao, Paul McMillan, Pavel Belov, Peng Su, Pe-
ter Bak, Peter Bakkum, Peter Deng, Peter Dolan,
Peter Hoeschele, Peter Welinder, Phil Tillet, Philip
Pronin, Philippe Tillet, Prafulla Dhariwal, Qiming
Yuan, Rachel Dias, Rachel Lim, Rahul Arora, Ra-
jan Troll, Randall Lin, Rapha Gontijo Lopes, Raul
Puri, Reah Miyara, Reimar Leike, Renaud Gaubert,
Reza Zamani, Ricky Wang, Rob Donnelly, Rob
Honsby, Rocky Smith, Rohan Sahai, Rohit Ramchan-
dani, Romain Huet, Rory Carmichael, Rowan Zellers,
Roy Chen, Ruby Chen, Ruslan Nigmatullin, Ryan
Cheu, Saachi Jain, Sam Altman, Sam Schoenholz,
Sam Toizer, Samuel Miserendino, Sandhini Agar-
wal, Sara Culver, Scott Ethersmith, Scott Gray, Sean
Grove, Sean Metzger, Shamez Hermani, Shantanu
Jain, Shengjia Zhao, Sherwin Wu, Shino Jomoto, Shi-
rong Wu, Shuaiqi, Xia, Sonia Phene, Spencer Papay,
Srinivas Narayanan, Steve Coffey, Steve Lee, Stew-
art Hall, Suchir Balaji, Tal Broda, Tal Stramer, Tao
Xu, Tarun Gogineni, Taya Christianson, Ted Sanders,
Tejal Patwardhan, Thomas Cunninghman, Thomas
Degry, Thomas Dimson, Thomas Raoux, Thomas
Shadwell, Tianhao Zheng, Todd Underwood, Todor
Markov, Toki Sherbakov, Tom Rubin, Tom Stasi,
Tomer Kaftan, Tristan Heywood, Troy Peterson, Tyce
Walters, Tyna Eloundou, Valerie Qi, Veit Moeller,
Vinnie Monaco, Vishal Kuo, Vlad Fomenko, WayneChang, Weiyi Zheng, Wenda Zhou, Wesam Manassra,
Will Sheu, Wojciech Zaremba, Yash Patil, Yilei Qian,
Yongjik Kim, Youlong Cheng, Yu Zhang, Yuchen
He, Yuchen Zhang, Yujia Jin, Yunxing Dai, and
Yury Malkov. 2024. Gpt-4o system card. Preprint ,
arXiv:2410.21276.
Fabio Petroni, Aleksandra Piktus, Angela Fan, Patrick
Lewis, Majid Yazdani, Nicola De Cao, James Thorne,
Yacine Jernite, Vladimir Karpukhin, Jean Maillard,
Vassilis Plachouras, Tim Rocktäschel, and Sebastian
Riedel. 2021. KILT: a benchmark for knowledge
intensive language tasks. In Proceedings of the 2021
Conference of the North American Chapter of the
Association for Computational Linguistics: Human
Language Technologies , pages 2523–2544, Online.
Association for Computational Linguistics.
Colin Raffel, Noam Shazeer, Adam Roberts, Katherine
Lee, Sharan Narang, Michael Matena, Yanqi Zhou,
Wei Li, and Peter J Liu. 2020. Exploring the lim-
its of transfer learning with a unified text-to-text
transformer. Journal of machine learning research ,
21(140):1–67.
Ori Ram, Liat Bezalel, Adi Zicher, Yonatan Belinkov,
Jonathan Berant, and Amir Globerson. 2023. What
are you token about? dense retrieval as distributions
over the vocabulary. In Proceedings of the 61st An-
nual Meeting of the Association for Computational
Linguistics (Volume 1: Long Papers) , pages 2481–
2498, Toronto, Canada. Association for Computa-
tional Linguistics.
Benjamin Reichman and Larry Heck. 2024. Dense
passage retrieval: Is it retrieving? In Findings of the
Association for Computational Linguistics: EMNLP
2024 , pages 13540–13553, Miami, Florida, USA.
Association for Computational Linguistics.
Stephen Robertson and Hugo Zaragoza. 2009. The prob-
abilistic relevance framework: Bm25 and beyond.
Foundations and Trends ®in Information Retrieval ,
3(4):333–389.
Christopher Sciavolino, Zexuan Zhong, Jinhyuk Lee,
and Danqi Chen. 2021. Simple entity-centric ques-
tions challenge dense retrievers. In Proceedings of
the 2021 Conference on Empirical Methods in Natu-
ral Language Processing , pages 6138–6148, Online
and Punta Cana, Dominican Republic. Association
for Computational Linguistics.
Rulin Shao, Jacqueline He, Akari Asai, Weijia Shi, Tim
Dettmers, Sewon Min, Luke Zettlemoyer, and Pang
Wei W Koh. 2024. Scaling retrieval-based language
models with a trillion-token datastore. In Advances in
Neural Information Processing Systems , volume 37,
pages 91260–91299. Curran Associates, Inc.
Qingyu Tan, Lu Xu, Lidong Bing, Hwee Tou Ng, and
Sharifah Mahani Aljunied. 2022. Revisiting Do-
cRED - addressing the false negative problem in
relation extraction. In Proceedings of the 2022 Con-
ference on Empirical Methods in Natural Language

Processing , pages 8472–8487, Abu Dhabi, United
Arab Emirates. Association for Computational Lin-
guistics.
Nandan Thakur, Nils Reimers, Andreas Rücklé, Ab-
hishek Srivastava, and Iryna Gurevych. 2021. BEIR:
A heterogeneous benchmark for zero-shot evaluation
of information retrieval models. In Thirty-fifth Con-
ference on Neural Information Processing Systems
Datasets and Benchmarks Track (Round 2) .
Kota Usuha, Makoto P. Kato, and Sumio Fujita. 2024.
Over-penalization for extra information in neural ir
models. In Proceedings of the 33rd ACM Interna-
tional Conference on Information and Knowledge
Management , CIKM ’24, page 4096–4100, New
York, NY , USA. Association for Computing Machin-
ery.
Pauli Virtanen, Ralf Gommers, Travis E. Oliphant, Matt
Haberland, Tyler Reddy, David Cournapeau, Ev-
geni Burovski, Pearu Peterson, Warren Weckesser,
Jonathan Bright, Stéfan J. van der Walt, Matthew
Brett, Joshua Wilson, K. Jarrod Millman, Nikolay
Mayorov, Andrew R. J. Nelson, Eric Jones, Robert
Kern, Eric Larson, C J Carey, ˙Ilhan Polat, Yu Feng,
Eric W. Moore, Jake VanderPlas, Denis Laxalde,
Josef Perktold, Robert Cimrman, Ian Henriksen, E. A.
Quintero, Charles R. Harris, Anne M. Archibald, An-
tônio H. Ribeiro, Fabian Pedregosa, Paul van Mul-
bregt, and SciPy 1.0 Contributors. 2020. SciPy 1.0:
Fundamental Algorithms for Scientific Computing in
Python. Nature Methods , 17:261–272.
Yumeng Wang, Lijun Lyu, and Avishek Anand. 2022.
Bert rankers are brittle: A study using adversarial
document perturbations. In Proceedings of the 2022
ACM SIGIR International Conference on Theory of
Information Retrieval , ICTIR ’22, page 115–120,
New York, NY , USA. Association for Computing
Machinery.
Shitao Xiao, Zheng Liu, Yingxia Shao, and Zhao Cao.
2022. RetroMAE: Pre-training retrieval-oriented lan-
guage models via masked auto-encoder. In Proceed-
ings of the 2022 Conference on Empirical Methods in
Natural Language Processing , pages 538–548, Abu
Dhabi, United Arab Emirates. Association for Com-
putational Linguistics.
Yuan Yao, Deming Ye, Peng Li, Xu Han, Yankai Lin,
Zhenghao Liu, Zhiyuan Liu, Lixin Huang, Jie Zhou,
and Maosong Sun. 2019. DocRED: A large-scale
document-level relation extraction dataset. In Pro-
ceedings of the 57th Annual Meeting of the Associa-
tion for Computational Linguistics , pages 764–777,
Florence, Italy. Association for Computational Lin-
guistics.
Yue Yu, Chenyan Xiong, Si Sun, Chao Zhang, and
Arnold Overwijk. 2022. COCO-DR: Combating the
distribution shift in zero-shot dense retrieval with con-
trastive and distributionally robust learning. In Pro-
ceedings of the 2022 Conference on Empirical Meth-
ods in Natural Language Processing , pages 1462–1479, Abu Dhabi, United Arab Emirates. Association
for Computational Linguistics.
Zexuan Zhong, Ziqing Huang, Alexander Wettig, and
Danqi Chen. 2023. Poisoning retrieval corpora by
injecting adversarial passages. In Proceedings of the
2023 Conference on Empirical Methods in Natural
Language Processing , pages 13764–13775, Singa-
pore. Association for Computational Linguistics.

A Appendix
A.1 Models Downstream Performance
We evaluate several dense retrievers on the Nat-
ural Questions (NQ) dataset (Kwiatkowski et al.,
2019), comparing their performance using standard
retrieval metrics: nDCG@10 and Recall@1011.
The models differ in training objectives, datasets,
and pooling mechanisms, offering a comprehensive
view of their retrieval capabilities in our experimen-
tal setup. Table 2 (and A.2) summarizes the results.
Dragon RoBERTa and Dragon+ (Lin et al., 2023)
demonstrate the highest performances due to di-
verse data augmentations and multiple supervision
sources, which progressively enhance their gener-
alization.12
COCO-DR (Yu et al., 2022) adopts continuous
contrastive learning and implicit distributionally
robust optimization (DRO) to address distribution
shifts in dense retrieval tasks. It exhibits moderate
performance, scoring lower than Dragon models.
Contriever (Izacard et al., 2021) uses unsuper-
vised contrastive learning but performs poorly with-
out fine-tuning (nDCG@10: 0.25). Fine-tuning
on MSMARCO significantly improves its perfor-
mance (nDCG@10: 0.50), underscoring the impor-
tance of fine-tuning for robust retrieval.
RetroMAE (Xiao et al., 2022), which introduces
a retrieval-oriented pre-training paradigm based on
Masked Auto-Encoder (MAE), featuring innova-
tive designs like asymmetric masking, achieves
slightly lower performance (nDCG@10: 0.48)
compared to fine-tuned Contriever.
The models also differ in their pooling mech-
anisms. Contriever uses average pooling, where
token representations are averaged to form a dense
vector for retrieval. In contrast, the other models
use CLS pooling, where the representation of the
[CLS] token is taken as the sentence embedding.
In summary, Dragon models lead in perfor-
mance, and the significant improvement of fine-
tuned Contriever over its unsupervised counterpart
highlights the importance of supervision and task-
specific adaptation in dense retrieval.
A.2 Position Bias: First vs. Last
Further evidence is provided in Figure A.2,
where we compared two document variants:
11Using BEIR framework (Thakur et al., 2021)
12Dragon RoBERTa is initialized from RoBERTa and
Dragon+ from RetroMAE1. Beginning-Evidence Document D1:The ev-
idence sentence is positioned at the start of the
document. 2. End-Evidence Document D2:The
same evidence sentence is positioned at the end of
the document.
D1:=Sev+X
S−h
−t∈DorigS−h
−t
D2:=X
S−h
−t∈DorigS−h
−t+Sev(8)
An example of the document pairs (Position
Bias) is shown in Table 1. The resulting t-statistics
(Figure 1 and A.2), where higher positive values
indicate a stronger preference for evidence at the
beginning ( D1) over the end ( D2), provide another
clear metric of positional bias. These results serve
as a foundation for our subsequent analysis in the
interplay between biases section.
Model Citation
facebook/dragon-plus-query-encoder Lin et al. (2023)
facebook/dragon-plus-context-encoder
facebook/dragon-roberta-query-encoder Lin et al. (2023)
facebook/dragon-roberta-context-encoder
facebook/contriever-msmarco Izacard et al. (2021)
facebook/contriever Izacard et al. (2021)
OpenMatch/cocodr-base-msmarco Yu et al. (2022)
Shitao/RetroMAE_MSMARCO_finetune Xiao et al. (2022)
gpt-4o-mini-2024-07-18 OpenAI et al. (2024)
gpt-4o-2024-08-06 OpenAI et al. (2024)
Table A.1: The details of the models we used in this
work.
Model Pooling nDCG@10 Recall@10
Dragon+ cls 0.55 0.63
Dragon RoBERTa cls 0.53 0.59
Contriever MSMARCO avg 0.52 0.59
Contriever avg 0.50 0.59
RetroMAE MSMARCO FT cls 0.49 0.55
COCO-DR Base MSMARCO cls 0.48 0.53
Table A.2: Models’ performance on our refined redo-
cred dataset with 7170 queries and 105925 corpus size.
Issue Count Percentage
Long Document 33 55%
Missing Answer 19 32%
Literal Bias 11 18%
Repetition 6 10%
Numbers 2 3%
Position Bias 2 3%
Table A.3: Preliminary findings from our annotation of
60 retrieval errors based on DecompX

0.0 2.5 5.0 7.5 10.0 12.5 15.0 17.5
Paired/uni00A0t/uni00ADTest/uni00A0StatisticContriever/uni00A0
COCO/uni00ADDR/uni00A0Base/uni00A0MSMARCO
RetroMAE/uni00A0MSMARCO/uni00A0FT
Contriever/uni00A0MSMARCO
Dragon+/uni00A0
Dragon/uni00A0RoBERTa/uni00A0Model13.31
13.33
13.67
14.32
16.58
17.18Literal/uni00A0Bias:/uni00A0Matching/uni00A0vs./uni00A0Different/uni00A0NamesFigure A.1: Paired t-test statistics comparing retrieval
scores between two scenarios: (1) when both query and
document use the shortest name variant, and (2) when
the query uses the short name but the document con-
tains the long name variant of the same entity. Positive
statistics indicate that models favor exact string matches
over semantic matching of equivalent entity names.
0 5 10 15 20
Paired/uni00A0t/uni00ADTest/uni00A0StatisticRetroMAE/uni00A0MSMARCO/uni00A0FT
Contriever/uni00A0
Dragon+/uni00A0
COCO/uni00ADDR/uni00A0Base/uni00A0MSMARCO
Contriever/uni00A0MSMARCO
Dragon/uni00A0RoBERTa/uni00A0Model3.86
6.06
6.35
8.86
12.04
18.83Position/uni00A0Bias:
Evidence/uni00A0as/uni00A0the/uni00A0First/uni00A0vs./uni00A0Last/uni00A0Sentence
Figure A.2: Paired t-test statistics comparing document
scores based on the position of the evidence sentence
(beginning vs. end). Higher positive values reflect a
preference for evidence at the beginning, indicating
positional bias.
Model AccuracyPaired t-Test
Statisticp-value
Dragon+ 0.0% -55.16 < 0.01
Dragon RoBERTa 0.0% -49.17 < 0.01
Contriever MSMARCO 0.0% -46.96 < 0.01
COCO-DR Base MSMARCO 0.0% -40.19 < 0.01
RetroMAE MSMARCO FT 0.0% -48.10 < 0.01
Contriever 1.2% -33.60 < 0.01
Table A.4: The accuracy, paired t-test statistics, and
p-values comparing a poison document , designed to
exploit biases and having a wrong answer (tail), against
a second document containing the evidence sentence
embedded in the middle of eight unrelated sentences
from a different document. All retrieval models perform
extremely poorly (less than 2% accuracy).
0 2 4 6 8 10
Paired/uni00A0t/uni00ADTest/uni00A0StatisticContriever/uni00A0MSMARCO
COCO/uni00ADDR/uni00A0Base/uni00A0MSMARCO
Contriever/uni00A0
Dragon+/uni00A0
Dragon/uni00A0RoBERTa/uni00A0
RetroMAE/uni00A0MSMARCO/uni00A0FTModel5.56
6.36
6.90
6.93
7.98
8.05Repetition/uni00A0Bias:
More/uni00A0Heads/uni00A0vs./uni00A0Fewer/uni00A0HeadsFigure A.3: Paired t-test statistics comparing the dot
product similarity of queries with two sets of sentences:
(1)More Heads , consisting of evidence and two sen-
tences with head mentions but no tails, and (2) Fewer
Heads , consisting of evidence and two sentences with-
out head or tail mentions. Positive values indicate higher
similarity for sentences with more heads.
0 5 10 15 20
Paired/uni00A0t/uni00ADTest/uni00A0StatisticContriever/uni00A0
COCO/uni00ADDR/uni00A0Base/uni00A0MSMARCO
Contriever/uni00A0MSMARCO
Dragon/uni00A0RoBERTa/uni00A0
RetroMAE/uni00A0MSMARCO/uni00A0FT
Dragon+/uni00A0Model9.46
13.83
15.17
17.33
18.44
20.51Brevity/uni00A0Bias:
Individual/uni00A0Evidence/uni00A0vs./uni00A0Evidence/uni00A0+/uni00A0Document
Figure A.4: Paired t-test statistics comparing scores for
documents containing only the evidence sentence versus
those containing the evidence plus the full document.
Higher positive values indicate a stronger model bias
toward brevity.

Method 1 (Higher Query Document Similarity Score) Method 2 (Lower Query Document Similarity Score)Foil vs. Evide.Query: Who is the publisher of Assassin’sCreed Unity ?
Document: "Assassin’sCreed Unity " "Assassin’sCreed Unity "Assassin’sCreed
Unity received mixed reviews upon its release .Query: Who is the publisher of Assassin’sCreed Unity ?
Document: Isa is a town and Local Government Area in the state of Sokoto in Nigeria
. It shares borders with ..... Assassin’sCreed Unity isanaction -adventure video
game developed byUbisoft Mon treal andpublished byUbisoft . Isa is a town and Local
Government Area in the state of Sokoto in Nigeria . It shares borders with .....Poison vs. Evide.Query: Who is the publisher of Assassin’sCreed Unity ?
Document: "Assassin’sCreed Unity " "Assassin’sCreed Unity "Assassin’sCreed
Unity received mixed reviews upon its release . Assassin’sCreed Unity isanaction-
adventurevideo game developed byElectronic Arts Mon treal andpublished byElec-
tronic ArtsQuery: Who is the publisher of Assassin’sCreed Unity ?
Document: Isa is a town and Local Government Area in the state of Sokoto in Nigeria
. It shares borders with ..... Assassin’sCreed Unity isanaction -adventure video
game developed byUbisoft Mon treal andpublished byUbisoft . Isa is a town and Local
Government Area in the state of Sokoto in Nigeria . It shares borders with .....
Table A.5: Examples from our framework for poison document and evidence document highlighting Evidence,
Head Entity,TailEntityandPoisonreplacing true tail entity. In all cases, retrieval models favor Method 1 over
Method 2, assigning higher retrieval scores accordingly.
Prompt Utility Prompt
Poisoning In the sentence: ’{evidence}’, replace the entity ’{tail}’ with a different entity that
makes sense in context but is completely different. Output only the replacement
entity. replacement entity:
RAG Answer the question based on the given document. Only give me the complete
answer and do not output any other words. The following is the given document.
Document: {doc}
Question: {query}
Answer:
RAG for No Doc Answer the question. Only give me the answer and do not output any other words.
Question: {query}
Answer:
Evaluation Query: {query}
Evidence: {evidence_sentence}
Gold Answer: {gold_answer}
Model Answer: {model_answer}
Does the Model Answer contain or imply the Gold Answer based on the evidence?
YES or NO :
Table A.6: The prompts utilized for RAG.
ModelCOCO-DR
Base MSMARCORetroMAE
MSMARCO FTContrieverContriever
MSMARCODragon+Dragon
RoBERTa
Query Name 1 Doc Name 1 Query Name 2 Doc Name 2
long long long short 20.67 21.92 19.22 21.05 21.03 21.64
short long 23.41 23.53 21.46 22.01 13.40 7.55
short short 18.43 19.60 16.41 17.35 4.99 1.75
short short long short 2.19 3.86 2.32 4.65 9.05 5.57
short long 13.33 13.67 13.31 14.32 16.58 17.18
Table A.7: Paired t-test statistics comparing retrieval scores between exact name matches (Q1-D1) and variant name
pairs (Q2-D2). Positive statistics indicate model preference for exact literal matches over semantically equivalent
name variants (e.g., preferring “US”-“US” over “US”-“United States”).

(0, 1] (1, 2] (2, 3] (3, 10]
Head Count(230, 512]
(190, 230]
(160, 190]
(130, 160]Doc Len378.54 383.93 387.89 386.47
380.79 384.58 386.79 387.88
380.82 385.57 388.68 390.04
382.37 387.82 388.50 388.37Dragon+ 
380382384386388390
(0, 1] (1, 2] (2, 3] (3, 10]
Head Count(230, 512]
(190, 230]
(160, 190]
(130, 160]Doc Len360.23 363.64 366.06 368.53
362.06 365.41 367.01 367.61
362.33 368.26 370.83 371.55
364.08 369.70 369.55 371.81Dragon RoBERTa 
362364366368370
(0, 1] (1, 2] (2, 3] (3, 10]
Head Count(230, 512]
(190, 230]
(160, 190]
(130, 160]Doc Len1.11 1.36 1.52 1.57
1.20 1.44 1.64 1.63
1.21 1.53 1.61 1.66
1.28 1.54 1.64 1.68Contriever MSMARCO
1.21.31.41.51.6
(0, 1] (1, 2] (2, 3] (3, 10]
Head Count(230, 512]
(190, 230]
(160, 190]
(130, 160]Doc Len0.75 0.88 0.99 1.02
0.79 0.91 1.01 1.07
0.78 0.95 1.01 1.04
0.82 0.99 1.02 1.06Contriever 
0.800.850.900.951.001.05
(0, 1] (1, 2] (2, 3] (3, 10]
Head Count(230, 512]
(190, 230]
(160, 190]
(130, 160]Doc Len208.59 212.05 213.67 214.52
209.76 213.01 215.45 215.43
210.09 213.87 215.10 215.71
210.89 214.46 215.41 216.05COCO-DR Base MSMARCO
209210211212213214215216
(0, 1] (1, 2] (2, 3] (3, 10]
Head Count(230, 512]
(190, 230]
(160, 190]
(130, 160]Doc Len41.25 46.10 50.21 50.32
43.00 47.58 51.16 50.84
43.50 49.11 50.50 51.67
44.43 50.12 51.73 51.75RetroMAE MSMARCO FT
4244464850
Average Dot ScoreFigure A.5: The average retrieval dot product score for samples with different document lengths and head entity
repetitions. (See Figure A.6 for the number of examples in each bin)
(0,/uni00A01] (1,/uni00A02] (2,/uni00A03] (3,/uni00A010]
Number/uni00A0of/uni00A0Head/uni00A0Entity/uni00A0Mentions(230,/uni00A0512]
(190,/uni00A0230]
(160,/uni00A0190]
(130,/uni00A0160]Document/uni00A0Length1001 224 127 300
896 192 166 237
1133 228 212 273
1298 417 247 196Number/uni00A0of/uni00A0Samples/uni00A0in/uni00A0Each/uni00A0Bin
20040060080010001200
Figure A.6: The number of samples in each bin of Figures A.5 and 5.

[CLS]
american
airlines
group
inc
.
is
an
american
publicly
traded
airline
holding
company
headquartered
in
fort
worth
,
texas
.
it
was
formed
december
9
,
2013
,
in
the
merger
of
am
##r
corporation
,
the
parent
company
of
american
airlines
,
and
us
airways
group
,
the
Gold Document[CLS]
which
administrative
territorial
entity
is
american
airlines
located
in
?
[SEP]Query
1.0
0.5
0.00.51.0Figure A.7: Visualization of token-wise effects on retriever scores using DecompX.
[CLS]
wood
##law
##n
is
an
unincorporated
community
and
census
-
designated
place
in
baltimore
county
,
maryland
,
united
states
.
the
population
was
37
,
87
##9
at
the
2010
census
.
it
is
home
to
the
headquarters
of
the
social
security
administration
(
ss
##a
)
and
Gold Document[CLS]
where
is
the
headquarters
of
social
security
administration
located
?
[SEP]Query
0.4
0.2
0.00.20.4
Figure A.8: Visualization of token-wise effects on retriever scores using DecompX.
[CLS]
"
united
states
health
care
reform
:
progress
to
date
and
next
steps
"
is
a
review
article
by
then
-
president
of
the
united
states
barack
obama
in
which
he
reviews
the
effects
of
the
affordable
care
act
(
ac
##a
)
,
a
major
health
care
law
he
signed
in
2010
,
and
recommends
health
care
policy
Gold Document[CLS]
when
did
affordable
care
act
start
?
[SEP]Query
0.3
0.2
0.1
0.00.10.20.3
Figure A.9: Visualization of token-wise effects on retriever scores using DecompX.