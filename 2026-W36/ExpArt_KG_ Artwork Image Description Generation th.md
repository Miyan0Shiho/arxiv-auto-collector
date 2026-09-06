# ExpArt-KG: Artwork Image Description Generation through Iterative Exploration of Knowledge Graphs

**Authors**: Yuta Kato, Shintaro Ozaki, Kazuki Hayashi, Yusuke Sakai, Hidetaka Kamigaito, Katsuhiko Hayashi, Taro Watanabe

**Published**: 2026-09-01 03:08:14

**PDF URL**: [https://arxiv.org/pdf/2609.00629v1](https://arxiv.org/pdf/2609.00629v1)

## Abstract
Large Vision-Language Models (LVLMs) achieve strong performance on image-grounded text generation and visual question answering. However, it remains difficult for them to comprehensively and accurately describe the factual relations among the entities and concepts associated with the objects depicted in an image. In this work, we propose a framework that efficiently exploits factual information from a knowledge graph via retrieval-augmented generation (RAG), with the goal of enabling LVLMs to generate detailed and accurate image explanations. Specifically, our method alternates between answer generation and knowledge-graph retrieval, and controls the search using a correctness judgment, thereby acquiring the necessary and sufficient factual information efficiently. We also construct a knowledge graph for the artwork domain (ExpArt-KG), in which the correspondence between images and entities is unambiguous. Applying the proposed method to this knowledge graph, we show experimentally that it improves the level of detail of artwork explanations and reduces the retrieval cost of external knowledge while maintaining generation quality comparable to that of iterating a fixed number of times.

## Full Text


<!-- PDF content starts -->

ExpArt-KG: Artwork Image Description Generation
through Iterative Exploration of Knowledge Graphs
Yuta Kato1, Shintaro Ozaki2, Kazuki Hayashi2, Yusuke Sakai2,
Hidetaka Kamigaito2,Katsuhiko Hayashi1, 2,Taro Watanabe2
1The University of Tokyo,2Nara Institute of Science and Technology (NAIST)
{ukato6209, katsuhiko-hayashi}@g.ecc.u-tokyo.ac.jp
ozaki.shintaro.ou6@naist.ac.jp
{hayashi.kazuki.hl4, sakai.yusuke.sr9, kamigaito.h, taro}@is.naist.jp
Abstract
Large Vision-Language Models (LVLMs)
achieve strong performance on image-
grounded text generation and visual question
answering. However, it remains difficult
for them to comprehensively and accurately
describe the factual relations among the
entities and concepts associated with the
objects depicted in an image. In this work, we
propose a framework that efficiently exploits
factual information from a knowledge graph
via retrieval-augmented generation (RAG),
with the goal of enabling LVLMs to generate
detailed and accurate image explanations.
Specifically, our method alternates between
answer generation and knowledge-graph
retrieval, and controls the search using a
correctness judgment, thereby acquiring the
necessary and sufficient factual information
efficiently. We also construct a knowledge
graph for the artwork domain (ExpArt-KG),
in which the correspondence between images
and entities is unambiguous. Applying the
proposed method to this knowledge graph, we
show experimentally that it improves the level
of detail of artwork explanations and reduces
the retrieval cost of external knowledge while
maintaining generation quality comparable to
that of iterating a fixed number of times.
1 Introduction
Large Vision-Language Models (LVLMs) (Liu
et al., 2023; Bai et al., 2025b; Abdin et al., 2024)
achieve strong performance on image-grounded
text generation and visual question answering (Li
et al., 2025b; Bai et al., 2023; Dai et al., 2023),
one instance of which is the ability to generate de-
tailed explanations of an image. In such settings,
an LVLM is required not merely to recognize the
people and objects contained in an image, but also
to connect them with their surrounding knowledge.However, it has been pointed out that LVLMs strug-
gle to comprehensively and accurately explain the
factual relations among the entities associated with
what they recognize (Hayashi et al., 2024).
A practical way to compensate for this limita-
tion of LVLMs is retrieval-augmented generation
(RAG) (Lewis et al., 2020), which utilizes exter-
nal knowledge. This framework generates answers
on the basis of retrieved and reliable external in-
formation, thereby suppressing hallucinations and
supplying the model with detailed knowledge. In
particular, for handling complex factual relations
among entities comprehensively and accurately, it
is effective to exploit a knowledge graph, whose
systematic structure makes the factual relations
among entities explicit. While such an approach
can extend the knowledge of a model without ad-
ditional training, existing knowledge-graph-based
methods fix the search depth or the number of iter-
ations in advance, which creates a trade-off: a shal-
low search yields insufficient information, whereas
a deep search increases the retrieval cost (Li et al.,
2025a). Consequently, a general framework for
efficiently retrieving useful factual relations and
properly using them for explanation generation has
not been sufficiently established.
In this work, we propose an iterative RAG frame-
work that introduces a correctness judgment of
the generated answer by a Large Language Model
(LLM) (Yang et al., 2025; Achiam et al., 2023;
Comanici et al., 2025; Grattafiori et al., 2024) act-
ing as a judge (Zheng et al., 2023), and that dy-
namically explores a knowledge graph and regen-
erates the answer on the basis of that judgment,
aiming at detailed image explanation generation
by LVLMs. In addition, we construct a knowledge
graph (ExpArt-KG) for extending the knowledge
of LVLMs, targeting the artwork domain, in which
a primary image and its entity are in one-to-one
1
arXiv:2609.00629v1  [cs.CL]  1 Sep 2026

correspondence.
Evaluating the explanation generation ability of
an LVLM using the proposed method together with
ExpArt-KG under evaluation metrics that cover
several viewpoints, we confirm that iterative an-
swer generation explores the knowledge graph ef-
ficiently and improves the level of detail of the
generated explanations. We further show that it-
eration control based on the correctness judgment
reduces the retrieval cost of external knowledge
while maintaining generation quality comparable
to that of iterating a fixed number of times.
2 Proposed Method
The proposed retrieval method iteratively performs
a local exploration of the nodes corresponding to
the entities contained in the query given to the
LVLM, together with the generation and verifica-
tion of an answer that uses the retrieved factual
information. In this way, the method obtains use-
ful factual information while keeping the retrieval
cost of external knowledge low. Following prior
work in which combining LLM self-verification
with RAG through iterative answering was shown
to improve performance (Hayashi et al., 2025), an-
swer verification by an LLM is introduced so as to
achieve both efficient exploration and high-quality
explanation generation. The prompts used in the
method are detailed in Appendix A.1.
The concrete procedure of the proposed method
is as follows.
(1) Triple retrieval.Among the entities con-
tained in the query, those that exist as nodes of
the knowledge graph are extracted, and the triples
containing them are retrieved.
(2) Triple selection.The triples obtained in Step
1 are ranked by TF-IDF (Salton and Buckley, 1988),
and the top 10 triples are selected for each entity.
In this ranking, the constituents of a triple are re-
garded as terms, and the set of triples containing
a particular entity as a document. The detailed
computation of TF-IDF is given in Appendix A.4.
(3) Answer generation.An augmented query is
constructed by appending the triples selected in
Step 2 to the original query, and it is given to the
LVLM together with the target artwork (image) to
generate an answer.
(4) Answer verification.The original query and
the answer generated in Step 3 are given to an LLM,which verifies whether the answer is correct. Using
forced decoding, the LLM outputs either “True”
or “False”. If “True” is obtained, the answer is
taken as the final answer of the LVLM; if “False”
is obtained, triple retrieval and answer generation
are performed again.
(5) Triple re-retrieval.When “False” is obtained
in Step 4, triples are retrieved again by the proce-
dures of Steps 1 and 2, using either the original
query or the answer generated immediately before,
and the answer is then regenerated and verified
through Steps 3 and 4. This process is repeated
until “True” is obtained or the maximum number
of iterations is reached.
3 Knowledge Graph Construction
This section introduces a general procedure for
constructing a knowledge graph suited to gener-
ating explanations of images. Based on the tex-
tual information contained in an arbitrary question-
answering dataset, entities and their relations are
extracted by reference to an external knowledge
base, and a knowledge graph is thereby constructed.
The concrete construction procedure is as follows.
(1) Extraction of node candidates.From the
strings contained in the questions, reference expla-
nations, and any available metadata (such as the ti-
tles of images) of a question-answering dataset, the
expressions that are candidate nodes of the knowl-
edge graph are extracted.
(2) Node selection.Among the node candidates
extracted in Step 1, those that exist as titles of
English Wikipedia articles are adopted as entities.
Furthermore, highly ambiguous words, for which
multiple corresponding Wikidata (Vrande ˇci´c and
Krötzsch, 2014) identifiers exist, and expressions
denoting general concepts are excluded from the
nodes, so that the resulting knowledge graph is
dense and specialized to the target domain. This
leaves only entities that are specific to, and unam-
biguous within, the target domain.
(3) Edge construction.For each node selected
in Step 2, the corresponding Wikidata entity is re-
ferred to, and the predicates defined among them
are obtained. Defining these predicates as edges
that express semantic relations between nodes
yields a knowledge graph in which the factual rela-
tions among entities are structured.
2

In this work, we applied this procedure to Ex-
pArt (Hayashi et al., 2024) to construct ExpArt-KG,
a knowledge graph about artworks.
4 Experimental Setup
Models.We used Qwen3-VL (Bai et al., 2025a)
as the LVLM that generates image explanations,
and Qwen3 (Yang et al., 2025) as the LLM that ver-
ifies answers (we refer to it as the validator LLM).
Dataset.From the test data of ExpArt,1we tar-
geted the instances whose image URLs2were valid
as of December 2025 and whose answers contained
at least one entity of the constructed ExpArt-KG.
From these we sampled approximately 25%, which
we used for evaluation. Details are given in Ap-
pendix A.3.
Input settings.In this experiment, we used the
following two question settings for each image.
(1)With Title: The LVLM generates an explana-
tion from the image and a question that contains
the title. The validator LLM judges correctness on
the basis of the question containing the title and the
answer.
(2)Without Title: When generating the image
explanation, the LVLM is given the image and a
question that does not contain the title. The val-
idator LLM is likewise given only the question
without the title and the answer, so that the correct-
ness judgment is made solely on the basis of the
logical consistency of the answer.
Through these settings, we examine the differ-
ence in generation and verification performance
depending on the presence of the title, as well as
the effect of the image title on the correctness judg-
ment of an answer.
Knowledge graph retrieval.We define three
variants of TF-IDF-based triple ranking: (1) treat-
ing predicates as terms and using their scores (PID),
(2) treating adjacent entities as terms and using
their scores (QID), and (3) treating both as terms
and using the sum of the two scores (PID-QID).
Details are given in Appendix A.4.
RAG settings.To examine the effectiveness of
ExpArt-KG and of the proposed method (RAG-
Validate), we compared RAG-Validate with a
method that generates an answer only once without
1https://huggingface.co/datasets/
naist-nlp/ExpArt
2https://commons.wikimedia.org/wiki/
Main_PageRAG (Baseline) and a method that uses RAG and
repeats answer generation five times (RAG-Loop5).
Evaluation metrics.For evaluation, we used
BLEU (Papineni et al., 2002), ROUGE (Lin, 2004),
and BERTScore (Zhang et al., 2020), which are
standard in natural language generation tasks, in
order to measure the lexical and semantic similarity
between the generated text and the reference expla-
nation. In addition, to measure the level of detail of
explanations about artworks, we used Entity Cover-
age, Entity F1, and Entity Cooccurrence (Hayashi
et al., 2024). The corresponding formulas are given
in Appendix A.5.
(1)Entity Coverageevaluates the extent to which
the generated text covers the artwork-related enti-
ties in the reference explanation, under two settings:
exact match and partial match.
(2)Entity F1compares the artwork-related entities
contained in the generated text and in the reference
explanation, and evaluates the frequency of entity
occurrences and their appropriate usage.
(3)Entity Cooccurrencefocuses on the cooccur-
rence relations among entities across the whole
text and evaluates the coverage of cooccurrences.
It adopts a length penalty analogous to the brevity
penalty of BLEU (Papineni et al., 2002) to prevent
overrating redundant explanations.
5 Results and Analysis
Table 1 reports the scores of the evaluation met-
rics under each setting. For all metrics except
BERTScore, RAG-Validate and RAG-Loop5 im-
proved over the Baseline. While BERTScore stays
at a high level regardless of whether RAG is used,
marked improvements are observed for Entity F1,
Entity Coverage, and Entity Cooccurrence. These
results indicate that, although the Baseline already
produces grammatically fluent text, using ExpArt-
KG and the proposed method substantially im-
proves the accuracy and comprehensiveness of the
entities contained in the explanations, and can raise
the level of detail of the explanations with respect
to the depicted subject and its surrounding knowl-
edge. Regarding the way TF-IDF scores are com-
puted, the largest improvement in the With Title
setting was observed when only adjacent entities
were used as the constituent terms. This suggests
that using the entities of a triple, rather than its pred-
icates, for retrieval improves the ability to identify
highly specific entities. Moreover, for questions
containing the title of the artwork, RAG-Validate
3

RAG setting TF-IDF BLEUROUGEBERT
ScoreEntity
F1 (↑)Entity Coverage (↑) Entity Cooccurrence (↑) Output
length1 2 L Exact Partial n=0 n=1 n=2 n=∞
With Title
RAG-Loop5 PID 3.22 30.73 9.02 19.02 84.51 41.47 38.52 45.07 7.93 8.77 8.89 8.90 121.6
RAG-Loop5 PID-QID 3.12 30.89 9.01 18.97 84.53 43.24 39.95 46.36 8.16 9.11 9.53 9.45 121.2
RAG-Loop5 QID 3.2231.17 9.11 19.2084.5443.85 40.73 47.10 8.60 9.61 9.92 9.79123.8
RAG-Validate PID 3.15 30.55 8.92 18.90 84.53 41.89 37.93 44.45 7.89 8.43 8.46 8.41 116.3
RAG-Validate PID-QID 3.18 30.66 8.96 18.92 84.55 42.17 38.69 45.34 8.47 8.71 9.01 8.95 118.1
RAG-Validate QID 3.22 30.85 9.07 18.96 84.56 42.96 39.46 45.99 8.53 9.19 9.52 9.45 120.1
Baseline – 1.18 24.89 5.53 15.94 83.62 16.93 14.82 22.67 1.76 1.55 1.53 1.44 91.3
Without Title
RAG-Loop5 PID 0.54 23.92 3.95 14.88 82.61 11.29 10.02 17.33 1.05 1.20 1.27 1.24 119.6
RAG-Loop5 PID-QID 0.60 24.024.0614.91 82.6411.75 10.66 17.80 1.49 1.68 1.78 1.73120.7
RAG-Loop5 QID0.61 24.183.9814.9482.60 11.12 10.33 17.62 1.41 1.60 1.73 1.66 123.8
RAG-Validate PID 0.43 22.58 3.51 14.14 82.62 8.59 7.94 15.22 0.76 0.95 0.97 0.95 102.7
RAG-Validate PID-QID 0.43 22.51 3.52 14.08 82.61 8.63 7.76 15.21 0.77 0.75 0.78 0.78 103.8
RAG-Validate QID 0.44 22.52 3.52 14.10 82.60 8.65 7.77 15.08 0.69 0.86 0.90 0.89 104.9
Baseline – 0.29 20.59 3.18 13.26 82.65 5.95 5.79 12.55 0.54 0.52 0.50 0.51 82.1
Table 1: Scores obtained under the various settings of our experiments.Boldindicates the best value for each
evaluation metric. Among the shaded rows, cyan denotes the proposed method and gray denotes the Baseline.
attains scores comparable to RAG-Loop5.
Figure 1 shows the transition of scores against
the number of iterative answer generations for the
Baseline, RAG-Loop5, and RAG-Validate. In the
Without Title setting, the score of the first answer
is markedly lower than in the With Title setting, but
the score was confirmed to increase continuously
as RAG-Loop5 repeated answer generation. In
the With Title setting, on the other hand, the score
also increases, but the magnitude of the increase
tends to diminish after a certain number of itera-
tions. This suggests that traversing the nodes of the
knowledge graph multiple times makes it possible
to gradually acquire factual information useful for
answer generation. In particular, the fact that score
improvements were confirmed even when the ini-
tial answer contained few related entities and thus
offered few cues for exploration supports the view
that iterative retrieval functions as multi-hop rea-
soning over the knowledge graph and effectively
explores entities that are useful for answer genera-
tion.
Furthermore, the average number of iterative
answer generations of RAG-Validate is 3.6 in
the With Title setting; despite requiring fewer
iterations than the fixed-count RAG-Loop5, it
maintains answer scores at a level comparable to
RAG-Loop5. This result shows that the proposed
method—which judges the correctness of the an-
swer generated by the LVLM with a validator LLM
and terminates the iteration once the answer is
deemed valid—avoids the unnecessary explorationthat arises when a fixed number of iterations is al-
ways performed, and thus reduces the retrieval cost
of external knowledge without degrading genera-
tion quality.
Finally, whereas RAG-Validate maintains scores
comparable to RAG-Loop5 in the With Title set-
ting, it obtains lower scores than RAG-Loop5 in the
Without Title setting. The likely cause of this degra-
dation is that the validator lacks the knowledge that
serves as the basis for the correctness judgment,
so that it fails to properly reject incorrect answers
and terminates the exploration prematurely. This
indicates that verification in RAG-Validate is based
not on the mere fluency of the generated text, but
on the factual consistency between the input title
and the content of the answer.
6 Conclusion
Aiming to improve the level of detail of image
explanations, we proposed a retrieval-augmented
generation method that makes effective use of a
knowledge graph. The proposed method iteratively
performs correctness judgment and regeneration
of the generated answer and dynamically explores
the knowledge graph, which makes it possible to
efficiently acquire entities that are useful for ex-
planation generation. In addition, we constructed
ExpArt-KG, a knowledge graph consisting of enti-
ties related to artworks. Our experimental results
confirm that, by dynamically controlling the num-
ber of iterations according to the validity of the
answer, the proposed method reduces the total num-
4

82.583.083.584.084.5BERT Score
141618ROUGE-L
123BLEU
10203040Entity F1
10203040Exact Entity Coverage
1 1.7 2 3 3.6 4 50246810Entity Cooccurrence (n=inf)
IterationsScore
Baseline
RAG-Loop5 (Trajectory)RAG-Loop5 (Final)
RAG-ValidateWith Title
Without TitleFigure 1: Scores obtained at each iteration of RAG-
Loop5, together with those of the Baseline and RAG-
Validate, under the QID setting. The horizontal axis
denotes the number of iterations and the vertical axis
denotes the score. Solid lines show the transition of
the RAG-Loop5 scores, and the shaded regions indi-
cate their 95% confidence intervals. Crosses mark the
Baseline, diamonds the final RAG-Loop5 scores, and
stars RAG-Validate, plotted at its average number of
iterations (3.6 with the title and 1.7 without it).
ber of iterations while maintaining generation qual-
ity comparable to that of iterating a fixed number
of times, thereby achieving both high-quality ex-
planation generation and a reduced retrieval cost
for external knowledge.Acknowledgments
This work was supported by JSPS KAKENHI
Grant Numbers JP23K28148 and JP24K02993.
References
Marah Abdin, Jyoti Aneja, Harkirat Behl, Sébastien
Bubeck, Ronen Eldan, Suriya Gunasekar, Michael
Harrison, Russell J Hewett, Mojan Javaheripi, Piero
Kauffmann, et al. 2024. Phi-4 technical report.arXiv
preprint arXiv:2412.08905.
Josh Achiam, Steven Adler, Sandhini Agarwal, Lama
Ahmad, Ilge Akkaya, Florencia Leoni Aleman,
Diogo Almeida, Janko Altenschmidt, Sam Altman,
Shyamal Anadkat, et al. 2023. GPT-4 technical re-
port.arXiv preprint arXiv:2303.08774.
Jinze Bai, Shuai Bai, Shusheng Yang, Shijie Wang,
Sinan Tan, Peng Wang, Junyang Lin, Chang Zhou,
and Jingren Zhou. 2023. Qwen-VL: A versa-
tile vision-language model for understanding, lo-
calization, text reading, and beyond.Preprint,
arXiv:2308.12966.
Shuai Bai, Yuxuan Cai, Ruizhe Chen, Keqin Chen,
Xionghui Chen, Zesen Cheng, Lianghao Deng, Wei
Ding, Chang Gao, et al. 2025a. Qwen3-VL technical
report.Preprint, arXiv:2511.21631.
Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wen-
bin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie
Wang, Jun Tang, et al. 2025b. Qwen2.5-VL technical
report.arXiv preprint arXiv:2502.13923.
Gheorghe Comanici, Eric Bieber, Mike Schaekermann,
Ice Pasupat, Noveen Sachdeva, Inderjit Dhillon, Mar-
cel Blistein, Ori Ram, Dan Zhang, Evan Rosen, et al.
2025. Gemini 2.5: Pushing the frontier with ad-
vanced reasoning, multimodality, long context, and
next generation agentic capabilities.arXiv preprint
arXiv:2507.06261.
Wenliang Dai, Junnan Li, Dongxu Li, Anthony Tiong,
Junqi Zhao, Weisheng Wang, Boyang Li, Pascale N
Fung, and Steven Hoi. 2023. InstructBLIP: Towards
general-purpose vision-language models with instruc-
tion tuning.Advances in Neural Information Process-
ing Systems, 36:49250–49267.
Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri,
Abhinav Pandey, Abhishek Kadian, Ahmad Al-
Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten,
Alex Vaughan, et al. 2024. The Llama 3 herd of
models.arXiv preprint arXiv:2407.21783.
Kazuki Hayashi, Hidetaka Kamigaito, Shinya Kouda,
and Taro Watanabe. 2025. IterKey: Iterative key-
word generation with LLMs for enhanced retrieval
augmented generation. InSecond Conference on
Language Modeling.
5

Kazuki Hayashi, Yusuke Sakai, Hidetaka Kamigaito,
Katsuhiko Hayashi, and Taro Watanabe. 2024. To-
wards artwork explanation in large-scale vision lan-
guage models. InProceedings of the 62nd Annual
Meeting of the Association for Computational Lin-
guistics (Volume 2: Short Papers), pages 705–729,
Bangkok, Thailand. Association for Computational
Linguistics.
Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio
Petroni, Vladimir Karpukhin, Naman Goyal, Hein-
rich Küttler, Mike Lewis, Wen-tau Yih, Tim Rock-
täschel, et al. 2020. Retrieval-augmented generation
for knowledge-intensive NLP tasks.Advances in
Neural Information Processing Systems, 33:9459–
9474.
Mufei Li, Siqi Miao, and Pan Li. 2025a. Simple is ef-
fective: The roles of graphs and large language mod-
els in knowledge-graph-based retrieval-augmented
generation. InICLR 2025 Workshop on Foundation
Models in the Wild.
Zongxia Li, Xiyang Wu, Hongyang Du, Fuxiao Liu,
Huy Nghiem, and Guangyao Shi. 2025b. A sur-
vey of state of the art large vision language models:
Alignment, benchmark, evaluations and challenges.
Preprint, arXiv:2501.02189.
Chin-Yew Lin. 2004. ROUGE: A package for auto-
matic evaluation of summaries. InText Summariza-
tion Branches Out, pages 74–81, Barcelona, Spain.
Association for Computational Linguistics.
Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae
Lee. 2023. Visual instruction tuning.Advances in
Neural Information Processing Systems, 36:34892–
34916.
Kishore Papineni, Salim Roukos, Todd Ward, and Wei-
Jing Zhu. 2002. Bleu: a method for automatic evalu-
ation of machine translation. InProceedings of the
40th Annual Meeting of the Association for Compu-
tational Linguistics, pages 311–318, Philadelphia,
Pennsylvania, USA. Association for Computational
Linguistics.
Gerard Salton and Christopher Buckley. 1988. Term-
weighting approaches in automatic text retrieval.In-
formation Processing & Management, 24(5):513–
523.
Denny Vrande ˇci´c and Markus Krötzsch. 2014. Wiki-
data: A free collaborative knowledge base.Commu-
nications of the ACM, 57:78–85.
An Yang, Anfeng Li, Baosong Yang, Beichen Zhang,
Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao,
Chengen Huang, Chenxu Lv, et al. 2025. Qwen3
technical report.arXiv preprint arXiv:2505.09388.
Tianyi Zhang, Varsha Kishore, Felix Wu, Kilian Q.
Weinberger, and Yoav Artzi. 2020. BERTScore:
Evaluating text generation with BERT. InInterna-
tional Conference on Learning Representations.Usage Prompt
LVLM
w/o KGSystem:You are an assistant helping with
visual question answering.
User:Question: {question}
Answer the question based on the provided
image.
Do not include any additional text other than
the answer itself.
LVLM
w/ KGSystem:You are an assistant helping with
visual question answering using external
knowledge.
User:Below are factual triples retrieved
from a knowledge graph to help answer the
question.
Each triple is formatted as: [SUBJECT,
PREDICATE, OBJECT]
Retrieved Facts: {triplets}
Question: {question}
Answer the question by integrating the image,
your internal knowledge, and any retrieved
facts that are relevant to the question.
Do not include any additional text other than
the answer itself.
Validator
LLMSystem:You are an assistant that validates
answers.
User:Question: {question}
Answer: {answer}
Judge whether the provided answer is correct
and addresses the question.
Respond with only "True" or "False". Do not
provide any additional explanation or text.
Table 2: Prompts used in the proposed method.
Abbreviation Hugging Face Model ID
Qwen3-VL Qwen/Qwen3-VL-8B-Instruct
Qwen3 Qwen/Qwen3-4B-Instruct-2507
Table 3: Correspondence between the models used and
their Hugging Face IDs.
Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan
Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin,
Zhuohan Li, Dacheng Li, Eric Xing, Hao Zhang,
Joseph E. Gonzalez, and Ion Stoica. 2023. Judging
LLM-as-a-judge with MT-bench and chatbot arena.
InThirty-seventh Conference on Neural Information
Processing Systems Datasets and Benchmarks Track.
A Appendix
A.1 Prompts
The prompts used in the proposed method are
shown in Table 2.
A.2 Details of the Models Used
Table 3 shows the correspondence between the ab-
breviations used in this paper and the Hugging Face
model IDs.
6

Breakdown # Questions
ExpArt test data 5,227
After filtering 4,823
After sampling (evaluation data) 1,199
Table 4: Dataset statistics.
A.3 Dataset Statistics
Table 4 shows the number of questions at each
processing stage of the dataset. Note that, by the
specification of the data, each question comes in
two settings depending on whether the title is in-
cluded.
A.4 Details of TF-IDF-Based Triple Selection
In our method, the set of triples connected to an
entity eis regarded as a document De, and a con-
stituent of a triple T∈D e(its predicate por its
adjacent node q) is regarded as a term t. LetEbe
the set of all entities of the knowledge graph and
N=|E| its size, let Terms(D) denote the set of
terms occurring in the triples of a document D, let
ft,ebe the number of times the term toccurs in
De, and let nt=|{e′∈ E:t∈Terms(D e′)}|be
the number of entities whose document contains t.
The TF-IDF value w(t, D e), which represents the
importance of the term t, is then defined as follows,
where logarithmic normalization and smoothing
are applied for numerical stability.
w(t, D e) = log(1 +f t,e)·logN+ 1
nt+ 1
(1)
The score S(T) of a triple Twhose predicate is
pand whose adjacent node is qis computed as
follows under each setting. (1)PID: S(T) =
w(p, D e), (2)QID: S(T) =w(q, D e), (3)PID-
QID: S(T) =w(p, D e) +w(q, D e). Based on
this score, we select the top-ranked triples for each
entity.
A.5 Definitions of the Entity Evaluation
Metrics
Let a generated explanation consisting of ksen-
tences be G={g 1, . . . , g k}and a reference
explanation consisting of msentences be R=
{r1, . . . , r m}. Let the sets of entities extracted
from these texts be EG= Entity(G) andER=
Entity(R) , and let |G|and|R|denote their respec-
tive total numbers of tokens. We define the length
penalty LP(G, R) for redundant generated text as
follows.
LP(G, R) = exp
−max
0.0,|G|
|R|−1
(2)Each evaluation metric is computed as follows. In
what follows, Cov(X, Y) denotes the proportion
of the elements ofYthat are covered byX.
(1)Entity Coverage (EC): the proportion of the
entities contained in Rthat are covered by G, com-
puted by the following equation. The longest com-
mon subsequence is used to compute the partial-
match coverage rate.
EC(G, R) = Cov(E G, ER)(3)
(2)Entity F1 (EF1): the harmonic mean of
the precision Prec and the recall Rec based on
the frequency of entity occurrences. Letting
#(e,·) be the number of occurrences of entity
ein the target text and Count clip(e, G, R) =
min(#(e, G),#(e, R)) , they are defined by the
following equations.
EF1=2×Prec×Rec
Prec+Rec(4)
Prec=P
e∈EGCount clip(e, G, R)P
e∈EG#(e, G)(5)
Rec=P
e∈ERCount clip(e, G, R)P
e∈ER#(e, R)(6)
(3)Entity Cooccurrence (ECooc): letting Con(·)
be a function that returns the pairs of entities cooc-
curring within a context window consisting of a
sentence and its surrounding nsentences, it is com-
puted by the following equation. We used the sen-
tence splitter of NLTK to split the text into sen-
tences.
ECooc(G, R) = LP(G, R)×Cov(Co n(G),Co n(R))(7)
7