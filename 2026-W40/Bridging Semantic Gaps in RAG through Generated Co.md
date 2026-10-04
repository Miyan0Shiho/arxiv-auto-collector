# Bridging Semantic Gaps in RAG through Generated Context Knowledge Fusion

**Authors**: Xinkai Du, Chao Lv, Yalin Sun, Quanjie Han, Lei Yao, Maosong Sun

**Published**: 2026-09-29 09:57:57

**PDF URL**: [https://arxiv.org/pdf/2609.37171v1](https://arxiv.org/pdf/2609.37171v1)

## Abstract
Retrieval-Augmented Generation has established itself as a fundamental framework in natural language processing, seamlessly integrating information retrieval with the generative capabilities of large language models. However, this process is fundamentally constrained by a critical challenge: semantic space mismatch between queries and retrieved contexts. We propose Knowledge-Aware Semantic Bridging (KASB), a novel framework that improves passage selection quality through semantic space alignment between queries and retrieved documents through intelligent knowledge fusion. Our approach leverages the complementary strengths of generative and retrieval-based knowledge through a multistage process that enhances both relevance and accuracy. We evaluate KASB on three popular open-domain Question Answering datasets to demonstrate the effectiveness of our approach.

## Full Text


<!-- PDF content starts -->

Bridging Semantic Gaps in RAG through
Generated Context Knowledge Fusion
Xinkai Du1,2, Chao Lv1, Yalin Sun3, Quanjie Han1, Lei Yao1, and Maosong
Sun2⋆
1Beijing Wanlian Zhilian Technology Corporation Limited, Beijing, China
{duxinkai, lvchao, hanquanjie, yaolei}@wanlianyida.com
2Department of Computer Science and Technology, Tsinghua University, Beijing,
China
sms@tsinghua.edu.cn
3Sunshine Digital Intelligence Tech Co., Ltd., Beijing, China
sunyalin-ghq@sinosig.com
Abstract.Retrieval-Augmented Generation has established itself as
a fundamental framework in natural language processing, seamlessly
integrating information retrieval with the generative capabilities of large
language models. However, this process is fundamentally constrained
by a critical challenge: semantic space mismatch between queries and
retrieved contexts. We propose Knowledge-Aware Semantic Bridging
(KASB), a novel framework that improves passage selection quality
through semantic space alignment between queries and retrieved docu-
ments through intelligent knowledge fusion. Our approach leverages the
complementary strengths of generative and retrieval-based knowledge
through a multistage process that enhances both relevance and accuracy.
We evaluate KASB on three popular open-domain Question Answering
datasets to demonstrate the effectiveness of our approach.
Keywords:Large Language Model·Direct Preference Optimization·
Retrieval Augmented Generation·Question Answering .
1 Introduction
Retrieval-Augmented Generation (RAG) [11] has become a cornerstone frame-
work in natural language processing, combining information retrieval with the
generative power of large language models (LLMs). The standard RAG pipeline
consists of three stages: retrieval, re-ranking, and generation. In the retrieval
phase, relevant documents are identified based on semantic similarity between
the query and a knowledge corpus. However, this process is hindered by a key
limitation: semantic space mismatch between queries and retrieved contexts.
This mismatch arises from differences in linguistic expression, abstraction levels,
and terminology, even for semantically aligned content [6].
⋆Corresponding Author
arXiv:2609.37171v1  [cs.CL]  29 Sep 2026

2 Xinkai Du, Chao Lv , Yalin Sun , Quanjie Han , Lei Yao, and Maosong Sun
LLMshavingacquiredvastworldknowledgethroughself-supervisedpretrain-
ing [2], can generate contextually coherent responses during question answering.
Yet, this generative capability is prone to hallucination: producing plausible but
factually inaccurate information [7]. In contrast, retrieved knowledge from cu-
rated corpora tends to be more factually reliable but often lacks fine-grained
relevance to the query. This trade-off between the relevance of generated con-
tent and the accuracy of retrieved content remains a fundamental challenge in
current RAG systems.
We propose Knowledge-Aware Semantic Bridging (KASB), a novel frame-
work that dynamically aligns the semantic spaces between queries and retrieved
documents via adaptive knowledge fusion. Our method exploits the comple-
mentary advantages of generative knowledge and retrieval-based knowledge in
a multi-stage paradigm, to substantially boost semantic relevance and factual
accuracy simultaneously.
Our contributions can be summarized as follows:
–We propose KASB, a reinforcement learning-enhanced methodology for dy-
namically aligning query and document representations.
–We introduce an unsupervised LLM generated context based evidence selec-
tor within the RAG framework, which is an effective approach that leverages
explicit hypothetical document as signals to align the preferences of different
components.
–Extensive experiments are conducted on three open domain QA datasets to
demonstrate the effectiveness of KASB.
Fig. 1.Overview of the KASB Framework. A finetuned generator is derived via the
DPO algorithm on the training set to produce a more effective generative context gen-
erator, which then serves in the test phase. During inference, the finetuned generator
first generates query-aligned generative contextsG; meanwhile, the evidence (repre-
senting the complete retrieved knowledge base) is processed through the generated
context-based selector to identify relevant retrieved contextsC, which are then fused
with the generated contexts[C;G]for final answer generation.

Knowledge-Aware Semantic Bridging 3
2 Related Work
In knowledge-intensive tasks, the retrieve-then-read paradigm has been domi-
nant [11,9,6]. Early approaches relied on sparse retrievers such as BM25, while
recent advances leverage dense contextualized representations, exemplified DPR
[9], which significantly outperform traditional methods. Various strategies have
since been explored to integrate the retrieved passages with large language mod-
els to improve the generation of answers [13].
LLMs have also been employed to generate auxiliary knowledge for down-
stream reasoning [12]. Building on this capability, recent work has explored using
LLM-generated knowledge to enhance retrieval, particularly when relevant in-
formation is sparse or insufficiently covered by a fixed corpus. This has led to the
emergence of the generate-then-read framework, which generates query-specific
contexts instead of retrieving them from a fixed corpus [19]. Methods like HyDE
[4] further refine retrieval by generating hypothetical document embeddings for
dense indexing.
Building on retrieval-augmented and generation-augmented paradigms, re-
cent work has explored hybrid approaches to mitigate their respective limita-
tions: retrieved passages may be irrelevant, while generated ones may be plau-
sible but inaccurate. To address this, [20] proposes merging passages based on
compatibility scoring, and [3] introduces an unsupervised bi-reranking frame-
work for fusing generated and retrieved knowledge. Our method integrates the
complementary strengths of both paradigms through a multistage process that
improves relevance and accuracy through a lightweight embedding-space selec-
tion mechanism.
KASBdiffersfromHyDE[4]inthatHyDEprimarilygeneratesahypothetical
document to construct a retrieval representation, whereas KASB starts from an
arbitrary retriever’s candidate set and uses multiple generated contexts to select
real evidence after retrieval. Compared with COMBO [20] and BRMGR [3],
KASBusesgeneratedcontextsnotonlyforfusionbutalsoasaquery-conditioned
signal for passage scoring and adaptive evidence selection.
3 KASB: Knowledge-Aware Semantic Bridging Guided
by Generated Context
3.1 Generated Context Generation
RAG inherently faces a semantic space mismatch between queries and retrieved
contexts. This gap arises from discrepancies in linguistic expression, abstraction
levels, and terminology usage, even when the core semantics are aligned. LLMs
having internalized extensive world knowledge through self-supervised pretrain-
ing, can generate hypothetical relevant contexts tailored to the query’s semantic
style. These generated contexts act as a semantic bridge to connect the query
with potentially misaligned retrieved documents, addressing the fundamental
limitation of traditional RAG’s retrieval phase.

4 Xinkai Du, Chao Lv , Yalin Sun , Quanjie Han , Lei Yao, and Maosong Sun
For each queryq, an LLM is prompted to generate a contextG, formulated
as
G=LLM(q,prompt)(1)
3.2 Preference-Tuned Generated Context Generator
While LLMs can generate contexts well-aligned with user queries, their outputs
frequently suffer from hallucinations, plausible but factually incorrect content
that significantly undermine the reliability of downstream retrieval and gen-
eration processes [7,14]. Traditional approaches to mitigating this issue often
employ Reinforcement Learning from Human Feedback (RLHF) [15], which re-
quires training and maintaining a separate reward model, introducing additional
complexity and computational overhead.
Direct Preference Optimization (DPO) [16] provides an efficient alternative
bydirectlyoptimizingthepolicyfrompreferencepairswithoutanexplicitreward
model. To enhance the contextual generation capability of our language model,
we fine-tune the generator on the training split of each dataset using preference
pairs constructed through an automated process.
Specifically, we leverage the Exact Match (EM) metric [17] to evaluate gener-
ated contexts against ground-truth answers, designating contexts containing the
correct answer as positive samples and those without as negative samples. This
creates a training signal that explicitly encourages factually accurate context
generation while maintaining semantic alignment with the query. The generator
is subsequently optimized through DPO, resulting in more reliable and factually
grounded contexts for semantic bridging without requiring human-annotated
preference data.
LDPO(πθ;πref) =−E (q,yw,yl)∼D[logσ(βf(q, y w, yl))](2)
f(q, y w, yl) = logπθ(yw|q)
πref(yw|q)−logπθ(yl|q)
πref(yl|q)(3)
whereπ θdenotes the model to be optimized,π refrepresents the original
model to generate question contexts, andθstands for the model optimization
parameters.
This fine-tuning process aligns the LLM’s generated contexts with factual
accuracy, ensuring they serve as reliable signals for subsequent retrieval.
3.3 Unsupervised Generated Context Based Retrieved Context
Selection
Retrieved contexts from curated corpora are factually reliable but often contain
irrelevant noise. Given the fine-tuned knowledge-generating model’s efficacy in
producing query-relevant context, we posit that strong alignment between re-
trieved documents and this generated context serves as a reliable indicator of
answer adequacy in top-ranked results.

Knowledge-Aware Semantic Bridging 5
Formally, we implement the document encoderenc das a dual-encoder archi-
tecture, where for each queryq j, the similarity between retrieved contexts and
the fine-tuned model’s generated knowledge is computed by:
S(gi, ej) =< enc d(gi), enc d(ej)>(4)
All the related context for queryq jis retrieved as follows:
Ev=
arg max
ej∈ES(gi, ej)|gi∈ G
(5)
whereGdenotes all the generated context information produced by fine-
tuning the large model for the queryq j, andE vrepresents the retrieved context
information corresponding to the queryq j.
3.4 Generated Context and Retrieved Context Combination
To determine the optimal number of retrieved documents, we introduce an adap-
tive top-k selection mechanism based on similarity score distribution analysis.
For each query, we first compute the centroid of generated context embeddings:
g=1
|G|X
gi∈GSBERT(g i)(6)
whereGdenotes the set of contexts generated by the fine-tuned model, and
SBERT refers to the pre-trained sentence embedding model from the sentence-
transformers library.
Subsequently, we calculate cosine similarities between the centroid gand
all retrieved contexts, yielding an ordered sequence{s 1, s2, . . . , s n}sorted in
descending order. To identify the optimal cutoff point, we analyze the first-order
differences of these similarity scores:
∆k=sk−sk+1 (7)
We use z-score to standardize all the first-order difference values. We select
the first indexk∗where∆ ksignificantly deviates from the mean z-score. If no
such decline is found, we will use the second-order difference method to find the
point with the maximum curvature to obtain the corresponding indexk∗.
Finally, the selected top-k∗retrieved contextsCand the generated contexts
produced by the fine-tuned large model are taken as a whole as the context
information of the query. The resulting context is then fed into the fine-tuned
large language model to generate the answer.
4 Experimental Setup
To demonstrate the effectiveness of our method, we evaluate it on three popular
open-domain Question Answering datasets.

6 Xinkai Du, Chao Lv , Yalin Sun , Quanjie Han , Lei Yao, and Maosong Sun
Datasets:Following prior work [3], we conduct experiments on three widely-
used open-domain Question Answering (QA) datasets: TriviaQA [8], Natural
Questions (NQ) [10], and WebQuestions (WebQ) [1].
Evaluation Metric:For question answering, we employ the exact match
(EM)score[17].Fortheretrievaltask,weadoptthestandardtop-KretrievalEM
metric, which measures the proportion of questions where at least one passage
among the top-K retrieved ones contains a text span that exactly matches the
human-annotated answer.
Baselines:Comparable to [20], we employ single and two knowledge sources
in comparison experiments to demonstrate our method’s effectiveness.
– Retri-Only: Utilizes only the original retrieved knowledge directly as the
input for the downstream reader model.
– Gen-Only: Adopts only the original LLM-generated knowledge as the sole
input for the reader model.
– HyDE: A classic retrieval-augmented baseline that generates a hypothetical
document for each query, encodes it into a dense representation, and uses
this representation to retrieve relevant real documents [4].
– COMBO: Matches LLM-generated passages with their retrieved counter-
parts to form compatible knowledge pairs and fuses the paired knowledge
for open question answering [20].
– BRMGR: An unsupervised Bi-Reranking for Merging Generated and Re-
trieved knowledge method, which optimizes the fusion of two knowledge
sources via bi-directional reranking of retrieved and generated passages [3].
DPO Training Details:We implement DPO optimization using the
LLaMA Factory framework [21] with a learning rate of5×10−6, batch size 4,
β= 0.1, and 3 training epochs. The reference model is the original Meta-Llama-
3.1-8B-Instruct. Positive and negative contexts are constructed automatically
from answer containment: a generated context containing the ground-truth
answer is preferred over one that does not. We therefore refer to these as
automatically constructed answer-containment preferences, rather than human
preferences.
We use Meta-Llama-3.1-8B-Instruct with the prompt “Question: {ques-
tion} Provide effective generated contexts for retrieving relevant informa-
tion.” Five relevant contexts are generated per query and encoded with
ms-marco-MiniLM-L6-v2.
For each query, if the generated contexts include at least one positive and
one negative instance, the query and the corresponding positive/negative con-
texts are formatted into ShareGPT style. This construction is scalable but can
produce false positives for polysemous strings and false negatives for aliases or
paraphrases. The data statistics for DPO training are summarized in Table 1:

Knowledge-Aware Semantic Bridging 7
Table 1.Datasets statistics.
Datasets Train Dev
TriviaQA 75675 8750
NQ 91334 10039
WebQ 3906 -
5 Experimental Results
5.1 Main Results
Finetuned LLM Generation AbilityTo enhance the LLM’s capability in
generating accurate and relevant contextual knowledge, we fine-tuned the gen-
erator using the DPO algorithm with carefully constructed positive and nega-
tive sample pairs. As demonstrated in Figure 2, the finetuned LLM generator
achieved substantial improvements in average retrieval exact match scores, with
absolute gains of 14.85%, 18.08%, and 16.75% on the TriviaQA, NQ, and WebQ
test sets, respectively.
Question AnsweringIn order to assess the performance of our method in
open question answering, we employ the finetuned LLM generator as the reader
model. The results of this evaluation are presented in Table 2.
Table 2 clearly illustrates that our proposed method KASB achieves the
strongest overall performance across all three datasets. Specifically, methods
leveraging two knowledge sources (retrieved and generated) consistently outper-
form those relying on a single knowledge source alone. For example, even the
weakest two-source method BRMGR surpasses the best single-source method on
NQ and WebQ, highlighting the advantage of fusing complementary retrieved
and generated knowledge for open question answering.
Among the two-source baselines, KASB delivers the most substantial gains.
This underscores the effectiveness of our knowledge alignment and semantic
boosting strategies in optimizing the fusion of dual knowledge sources. Addi-
tionally, when examining single-source methods, Gen-Only outperforms Retri-
Only on TriviaQA and NQ, while Retri-Only exhibits stronger performance on
WebQ, revealing the complementary strengths of generated and retrieved knowl-
edge across datasets with different characteristics.
5.2 Ablation Analysis
When both retrieved and generated knowledge are available, we use them as
contextual input to prompt the fine-tuned LLM, which then directly generates
the answer. The model’s accuracy is evaluated using exact match (EM) score
against the ground truth.

8 Xinkai Du, Chao Lv , Yalin Sun , Quanjie Han , Lei Yao, and Maosong Sun
Table 2.Exact match scores on test dataset.
MethodsTriviaQA NQ WebQ
Single Knowledge
Retri-Only 62.2 48.6 48.3
Gen-Only 67.5 50.3 41.6
Two Knowledge
HyDE 72.3 53.4 54.6
COMBO 74.6 54.2 53.0
BRMGR 68.6 52.2 53.4
KASB75.4 59.6 61.2
Retrieval Performance Boost via Semantic BridgingTo assess the im-
provements in retrieval performance brought about by semantic bridging, we
conduct dedicated experiments where passages are retrieved using both unsu-
pervised and supervised retrievers. The unsupervised retrievers selected for our
experiments include BM25, MSS [18], and Contriever [5], whereas the supervised
retrievers consist of DPR [9] and MSS-DPR [18].
Retrieval accuracy at Top-20 and Top-100 on the test sets of the three
datasets, evaluated on the basis of the top-1000 retrieved passages, is reported
in Table 3. As illustrated in the table, our proposed KASB method yields con-
sistent improvements in retrieval performance across all retrievers, with notably
substantial gains achieved for unsupervised models including BM25, MSS [18],
and Contriever [5].
Table 3.Top-20, 100 retrieval accuracy on the test set of datasets for the top-1000
retrieved passages.
Retriever TriviaQA NQ WebQ
Top-20 Top-100 Top-20 Top-100 Top-20 Top-100
Unsupervised Retrievers
MSS 67.2 79.1 60.0 75.6 49.2 68.4
MSS + KASB 81.3 85.0 77.3 81.5 62.8 75.8
BM25 76.4 83.2 62.9 78.3 62.4 75.5
BM25 + KASB 84.3 87.2 75.6 84.3 65.6 77.4
Contriever 73.9 82.9 67.9 80.6 65.7 80.1
Contriever + KASB 85.7 87.4 81.2 86.5 70.1 81.3
Supervised Retrievers
DPR 79.8 85.1 79.2 85.7 74.6 81.6
DPR + KASB 84.8 87.4 83.5 87.4 78.2 84.2
MSS-DPR 81.9 86.6 81.4 88.1 76.9 84.6
MSS-DPR + KASB 87.5 88.3 84.3 89.7 81.4 85.3

Knowledge-Aware Semantic Bridging 9
Fig. 2.Average retrieval exact match score obtained by finetuned vs original LLM
generator on generated contexts.
Effect of the finetuned Generated Context GeneratorComparing KASB
with the w/o DPO variant in Table 4, we observe that the fine-tuned LLM
achieves significantly higher performance across all three dataset compared to
the baseline model without direct preference optimization. This indicates that
the fine-tuning process, particularly through direct preference optimization en-
hances the model’s ability to generate high-quality responses. The substantial
improvement in exact match scores suggests that the fine-tuned LLM aligns
better with the automatically constructed answer-containment preferences and
produces more accurate answers when integrated with retrieved and generated
knowledge.
Importance of the Retrieved and Generated Knowledge FusionCom-
paring KASB with the ablated variants in Table 4, both knowledge sources con-
tribute to performance. Removing generated knowledge (GK) causes a larger
drop than removing retrieved knowledge (RK), while removing DPO has a
smaller effect. This supports the interpretation that the main benefit comes from
the interaction of generated-context guidance, evidence selection, and two-source
fusion rather than from DPO alone.

10 Xinkai Du, Chao Lv , Yalin Sun , Quanjie Han , Lei Yao, and Maosong Sun
Table 4.Question answering performance (Exact Match). GK and RK denote gener-
ated knowledge and retrieved knowledge, respectively. The combined knowledge results
are computed by union of single knowledge sources.
Model TriviaQA NQ WebQ
KASB75.4 59.6 61.2
w/o GK 69.9 54.9 56.3
w/o RK 73.2 56.8 59.5
w/o DPO 74.6 57.7 59.8
5.3 Parameter Sensitivity Analysis
Figure 3 examines the sensitivity of KASB to two critical hyperparameters: the
DPO coefficientβand the initial number of retrieved candidates. Performance
peaks atβ= 0.1suggesting that moderate preference strength optimally bal-
ances alignment with retained knowledge diversity. For retrieved candidates,
performance stabilizes after 800 documents, indicating that KASB can work ef-
fectivelywithstandardretrievalpoolsizeswithoutrequiringexhaustiveretrieval.
The adaptive top-kmechanism yields averagek⋆values of 24.3 (TriviaQA),
18.7 (NQ), and 15.2 (WebQ). To directly examine whether adaptive selection is
preferable to a globally fixed cutoff, we further compare fixedk∈ {5,10,20,50}
inTable5.Thefixed-kresultsshowaconsistentnon-monotonictrend:increasing
kinitially improves performance by adding useful evidence, while overly large
kintroduces more less-relevant passages and degrades performance. In contrast,
theadaptivestrategyachievesthebestresultonallthreedatasets.Thisindicates
that a single globalkis suboptimal because the appropriate amount of evidence
varies with the query-specific score distribution.
Table 5.Fixed-kversus adaptive evidence selection. EM is reported on the three QA
datasets; Avg. is the mean across datasets.
Selection TriviaQA NQ WebQ Avg.
Fixed-k,k= 573.2 58.2 59.2 63.5
Fixed-k,k= 1073.8 58.5 59.8 64.0
Fixed-k,k= 2074.2 58.9 59.7 64.3
Fixed-k,k= 5073.5 58.1 59.4 63.7
Adaptivek⋆75.4 59.6 61.2 65.4
5.4 Discussion
Despite KASB’s effectiveness, several limitations warrant discussion. First, the
quality of semantic bridging is constrained by the generative model’s knowledge

Knowledge-Aware Semantic Bridging 11
Fig. 3.ParametersensitivityanalysisshowingtheeffectofDPOcoefficientβandinitial
candidate pool size on EM performance across datasets.
coverage and accuracy; generated contexts may be outdated, redundant, hal-
lucinated, or conflicting with retrieved evidence. Second, our DPO preferences
areautomaticallyconstructedfromanswercontainmentratherthanhumanjudg-
ments, so they may reward surface overlap and do not establish semantic correct-
ness; answer-masked or answer-absent evaluation would be a useful diagnostic.
Third, the fixed-kcomparison shows that performance first improves as more
evidence is included and then declines when overly large cutoffs introduce less-
relevantpassages,supportingtheuseofquery-specificadaptiveselection.Fourth,
we do not explicitly perform deduplication or contradiction resolution between
generated and retrieved knowledge. Finally, the current evaluation is limited to
single-turn open-domain QA, and we do not claim cross-dataset generalization
of the DPO-tuned generator without additional transfer experiments.
6 Conclusion
In this paper, we propose Knowledge-Aware Semantic Bridging (KASB), a
lightweight post-retrieval framework that uses generated contexts to guide evi-
dence selection and fuse selected retrieved passages with generated knowledge.
Experiments on three single-turn open-domain QA datasets show consistent
improvements in retrieval accuracy and answer correctness across diverse re-
trievers. The results support generated-context-guided evidence selection as a
practical system-level integration for improving RAG.
Acknowledgements
This work is supported by Beijing Municipal Science and Technology Plan
Project(Z241100001324025).

12 Xinkai Du, Chao Lv , Yalin Sun , Quanjie Han , Lei Yao, and Maosong Sun
References
1. Berant, J., Chou, A., Frostig, R., Liang, P.: Semantic parsing on Freebase from
question-answer pairs. In: Proceedings of the 2013 Conference on Empirical Meth-
ods in Natural Language Processing. pp. 1533–1544 (2013)
2. Brown, T., Mann, B., Ryder, N., Subbiah, M., Kaplan, J.D., Dhariwal, P., Nee-
lakantan,A.,Shyam,P.,Sastry,G.,Askell,A.,etal.:Languagemodelsarefew-shot
learners.AdvancesinNeuralInformationProcessingSystems33,1877–1901(2020)
3. Du, X., Han, Q., Lv, C., Liu, Y., Sun, Y., Shu, H., Shan, H., Sun, M.: Improving
generated and retrieved knowledge combination through zero-shot generation. In:
ICASSP 2025 - 2025 IEEE International Conference on Acoustics, Speech and
Signal Processing (ICASSP). pp. 1–5. IEEE (2025)
4. Gao, L., Ma, X., Lin, J., Callan, J.: Precise zero-shot dense retrieval without rel-
evance labels. In: Proceedings of the 61st Annual Meeting of the Association for
Computational Linguistics (Volume 1: Long Papers). pp. 1762–1777 (2023)
5. Izacard, G., Caron, M., Hosseini, L., Riedel, S., Bojanowski, P., Joulin, A., Grave,
E.: Unsupervised dense information retrieval with contrastive learning. arXiv
preprint arXiv:2112.09118 (2021)
6. Izacard, G., Grave, E.: Leveraging passage retrieval with generative models for
open domain question answering. In: Proceedings of the 16th Conference of the
EuropeanChapteroftheAssociationforComputationalLinguistics:MainVolume.
pp. 874–880 (2021)
7. Ji,Z.,Lee,N.,Frieske,R.,Yu,T.,Su,D.,Xu,Y.,Ishii,E.,Bang,Y.J.,Madotto,A.,
Fung, P.: Survey of hallucination in natural language generation. ACM Computing
Surveys55(12), 1–38 (2023)
8. Joshi, M., Choi, E., Weld, D.S., Zettlemoyer, L.: TriviaQA: A large scale distantly
supervised challenge dataset for reading comprehension. In: Proceedings of the
55th Annual Meeting of the Association for Computational Linguistics (Volume 1:
Long Papers). pp. 1601–1611 (2017)
9. Karpukhin, V., Oguz, B., Min, S., Lewis, P., Wu, L., Edunov, S., Chen, D., Yih,
W.t.: Dense passage retrieval for open-domain question answering. In: Proceedings
of the 2020 Conference on Empirical Methods in Natural Language Processing
(EMNLP). pp. 6769–6781 (2020)
10. Kwiatkowski, T., Palomaki, J., Redfield, O., Collins, M., Parikh, A., Alberti, C.,
Epstein, D., Polosukhin, I., Devlin, J., Lee, K., et al.: Natural questions: a bench-
mark for question answering research. Transactions of the Association for Compu-
tational Linguistics7, 453–466 (2019)
11. Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V., Goyal, N., Küttler,
H., Lewis, M., Yih, W.t., Rocktäschel, T., et al.: Retrieval-augmented generation
for knowledge-intensive NLP tasks. Advances in Neural Information Processing
Systems33, 9459–9474 (2020)
12. Liu, J., Liu, A., Lu, X., Welleck, S., West, P., Le Bras, R., Choi, Y., Hajishirzi, H.:
Generatedknowledgepromptingforcommonsensereasoning.In:Proceedingsofthe
60th Annual Meeting of the Association for Computational Linguistics (Volume 1:
Long Papers). pp. 3154–3169 (2022)
13. Liu, Y., Yavuz, S., Meng, R., Moorthy, M., Joty, S., Xiong, C., Zhou, Y.: Exploring
the integration strategies of retriever and large language models. arXiv preprint
arXiv:2308.12574 (2023)
14. Maynez, J., Narayan, S., Bohnet, B., McDonald, R.: On faithfulness and factuality
in abstractive summarization. In: Proceedings of the 58th Annual Meeting of the
Association for Computational Linguistics. pp. 1906–1919 (2020)

Knowledge-Aware Semantic Bridging 13
15. Ouyang, L., Wu, J., Jiang, X., Almeida, D., Wainwright, C., Mishkin, P., Zhang,
C., Agarwal, S., Slama, K., Ray, A., et al.: Training language models to follow
instructions with human feedback. In: Advances in Neural Information Processing
Systems. vol. 35, pp. 27730–27744 (2022)
16. Rafailov, R., Sharma, A., Mitchell, E., Manning, C.D., Ermon, S., Finn, C.: Direct
preferenceoptimization:Yourlanguagemodelissecretlyarewardmodel.Advances
in Neural Information Processing Systems36, 53728–53741 (2023)
17. Rajpurkar, P., Zhang, J., Lopyrev, K., Liang, P.: SQuAD: 100,000+ questions
for machine comprehension of text. In: Proceedings of the 2016 Conference on
Empirical Methods in Natural Language Processing. pp. 2383–2392 (2016)
18. Sachan, D., Patwary, M., Shoeybi, M., Kant, N., Ping, W., Hamilton, W.L., Catan-
zaro, B.: End-to-end training of neural retrievers for open-domain question answer-
ing. In: Proceedings of the 59th Annual Meeting of the Association for Computa-
tionalLinguisticsandthe11thInternationalJointConferenceonNaturalLanguage
Processing (Volume 1: Long Papers). pp. 6648–6662 (2021)
19. Yu, W., Iter, D., Wang, S., Xu, Y., Ju, M., Sanyal, S., Zhu, C., Zeng, M., Jiang,
M.: Generate rather than retrieve: Large language models are strong context gen-
erators. In: The Eleventh International Conference on Learning Representations
(2023)
20. Zhang, Y., Khalifa, M., Logeswaran, L., Lee, M., Lee, H., Wang, L.: Merging
generated and retrieved knowledge for open-domain QA. In: Proceedings of the
2023 Conference on Empirical Methods in Natural Language Processing. pp. 4710–
4728 (2023)
21. Zheng, Y., Zhang, R., Zhang, J., Ye, Y., Luo, Z.: LlamaFactory: Unified efficient
fine-tuning of 100+ language models. In: Proceedings of the 62nd Annual Meeting
of the Association for Computational Linguistics (Volume 3: System Demonstra-
tions). pp. 400–410 (2024)