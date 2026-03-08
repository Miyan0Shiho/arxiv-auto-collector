# ReFeed: Retrieval Feedback-Guided Dataset Construction for Style-Aware Query Rewriting

**Authors**: Jiyoon Myung, Jungki Son, Kyungro Lee, Jihyeon Park, Joohyung Han

**Published**: 2026-03-02 03:43:53

**PDF URL**: [https://arxiv.org/pdf/2603.01417v1](https://arxiv.org/pdf/2603.01417v1)

## Abstract
Retrieval systems often fail when user queries differ stylistically or semantically from the language used in domain documents. Query rewriting has been proposed to bridge this gap, improving retrieval by reformulating user queries into semantically equivalent forms. However, most existing approaches overlook the stylistic characteristics of target documents-their domain-specific phrasing, tone, and structure-which are crucial for matching real-world data distributions. We introduce a retrieval feedback-driven dataset generation framework that automatically identifies failed retrieval cases, leverages large language models to rewrite queries in the style of relevant documents, and verifies improvement through re-retrieval. The resulting corpus of (original, rewritten) query pairs enables the training of rewriter models that are explicitly aware of document style and retrieval feedback. This work highlights a new direction in data-centric information retrieval, emphasizing how feedback loops and document-style alignment can enhance the reasoning and adaptability of RAG systems in real-world, domain-specific contexts.

## Full Text


<!-- PDF content starts -->

ReFeed: Retrieval Feedback–Guided Dataset Construction for Style-Aware Query
Rewriting
Jiyoon Myung1,2*, Jungki Son1, Kyungro Lee1, 3, Jihyeon Park1, 4, Joohyung Han1, 5
1MODULABS,2Samsung SDS,3Dable,4MIRI D.I.H Co., Ltd.,5Lomin
{jiyoon0424, aeolian83, lkr981147, milhaud1201, ddang8jh}@gmail.com
Abstract
Retrieval systems often fail when user queries differ stylis-
tically or semantically from the language used in domain
documents. Query rewriting has been proposed to bridge
this gap, improving retrieval by reformulating user queries
into semantically equivalent forms. However, most exist-
ing approaches overlook the stylistic characteristics of target
documents—their domain-specific phrasing, tone, and struc-
ture—which are crucial for matching real-world data distribu-
tions. We introduce a retrieval feedback–driven dataset gener-
ation framework that automatically identifies failed retrieval
cases, leverages large language models to rewrite queries in
the style of relevant documents, and verifies improvement
through re-retrieval. The resulting corpus of (original, rewrit-
ten) query pairs enables the training of rewriter models that
are explicitly aware of document style and retrieval feed-
back. This work highlights a new direction in data-centric
information retrieval, emphasizing how feedback loops and
document-style alignment can enhance the reasoning and
adaptability of RAG systems in real-world, domain-specific
contexts.
Introduction
Information retrieval (IR) systems often fail when user
queries differ stylistically or semantically from the lan-
guage used in domain documents. In many real-world appli-
cations—such as technical manuals, customer-service chat
logs, and enterprise knowledge bases—users express prob-
lems or requests in informal, context-rich ways, whereas
documents are written in formal and domain-specific styles.
This stylistic mismatch leads to retrieval failures even when
the underlying intent of the query is correct, limiting the re-
liability of retrieval-augmented generation (RAG) systems
and conversational assistants.
Query rewriting has emerged as a promising approach to
bridge this gap. By reformulating user queries into semanti-
cally equivalent forms, rewriting helps retrievers locate rel-
evant documents more effectively. However, most existing
rewriting approaches remain semantic rather than stylistic:
they focus on preserving meaning while ignoring how in-
*This work was conducted independently and does not repre-
sent the views of the affiliated organizations.
Copyright © 2026, Association for the Advancement of Artificial
Intelligence (www.aaai.org). All rights reserved.formation is actually phrased in the target corpus. As a re-
sult, rewritten queries may still deviate from the linguistic
and structural patterns found in domain-specific documents,
leaving retrieval performance suboptimal.
Recent advances in feedback-driven query rewriting have
shown that retrieval signals can guide better reformulation
and ranking. However, most approaches integrate such feed-
back onlyduringmodel training, using it to fine-tune rewrit-
ers via reinforcement objectives. In contrast, we investigate
how retrieval outcomes can be leveragedbeforetraining—to
automatically curate high-quality supervision data for style-
aware query rewriting.
To address these gaps, we proposeReFeed, a retrieval
feedback–guided dataset generation framework for style-
aware query rewriting. Our method automatically identi-
fies failed retrieval cases from existing QA datasets, lever-
ages large language models (LLMs) to rewrite queries in
the style of relevant documents, and validates improvement
through re-retrieval. Only rewritten queries that successfully
retrieve the correct document are retained, resulting in a
high-precision corpus of (original, rewritten) query pairs that
naturally encode both retrieval feedback and document-style
adaptation. This corpus can be used for few-shot prompt-
ing or supervised fine-tuning (SFT) to train rewriter models
that are explicitly aware of document style and retrieval out-
comes.
Preliminary downstream experiments suggest that incor-
porating these examples into few-shot prompts can improve
retrieval recall and RAG response accuracy, indicating the
potential utility of style-aware rewriting for robust informa-
tion retrieval. By emphasizing feedback loops and stylistic
alignment, our work lays the groundwork for training fu-
turestyle-aware rewriter models, providing a data-centric
foundation for intelligent and adaptive information retrieval
systems.
Related Work
Query Rewriting for Information Retrieval.Query
rewriting aims to reformulate user queries into more effec-
tive forms for retrieval or generation tasks. Recent works
have explored explicit rewriting mechanisms optimized for
retrieval, including structured rewriting frameworks (Baek
et al. 2024) and generative reformulation approaches (Dhole
et al. 2024). These studies show that leveraging large lan-arXiv:2603.01417v1  [cs.IR]  2 Mar 2026

Figure 1: Overview of the retrieval feedback–driven dataset generation framework. Missed queries are rewritten via LLMs to
match the style of the correct documents and validated through re-retrieval before being assembled into a style-aware corpus.
guage models for rewriting can improve recall and ranking
quality, but they primarily emphasize semantic reformula-
tion or expansion rather than aligning rewritten queries with
the stylistic patterns of target domain documents.
Feedback-Driven Query Reformulation.Another line of
research integrates retrieval or ranking feedback into the
rewriting process. For example, Mao et al. (2024) lever-
age reranker-derived ranking signals to fine-tune rewrit-
ers, and Wang et al. (2024) incorporate multi-aspect feed-
back—including retrieval, generation, and document simi-
larity—to optimize rewriting for RAG systems. Similarly,
Shi et al. (2023) augment black-box language models with
retrieval feedback to enhance evidence grounding. While
these methods demonstrate that feedback can guide rewriter
optimization, they typically employ reinforcement learning
or reward modeling to train models directly. In contrast, our
framework uses retrieval feedback as adata curation sig-
nal, constructing reusable (original, rewritten) query pairs
that encode both retrieval success and document-style align-
ment.
Existing Query Rewriting Datasets.Large-scale rewrit-
ing datasets such asCANARD(Elgohary, Peskov, and
Boyd-Graber 2019) andQReCC(Anantha et al. 2021) sup-
port supervised learning for semantic reformulation, provid-
ing pairs of conversational and standalone queries. However,
these datasets do not capture retrieval feedback or stylis-
tic variation. Our proposed dataset complements these re-
sources by leveraging retrieval outcomes to ensure effective-
ness and by explicitly conditioning rewrites on the linguistic
style of relevant domain documents.
Methodology
Our goal is to construct aretrieval feedback–driven dataset
forstyle-aware query rewriting—a corpus of (original,
rewritten) query pairs that reflect both retrieval outcomes
and the stylistic characteristics of domain documents. The
overall framework, illustrated in Figure 1, consists of four
stages: (1) Initial Retrieval, (2) LLM-Guided Rewriting, (3)
Verification via Re-Retrieval, and (4) Dataset Assembly.
Initial Retrieval
We begin with an existing QA datasetD={(q i, ai)}, where
each queryq iis paired with its relevant document or passage
ai. Each query is embedded and used to retrieve the top-kdocuments from the corpus using a dense retriever (e.g., E5-
base or BGE-large). If the ground-truth documenta iis not
included in the retrieved top-k, the instance is marked as a
miss case. These failure cases serve as informative starting
points for generating retrieval-aware rewrites.
LLM-Guided Rewriting
For each miss case, we prompt a large language model
(LLM) to rewrite the query so that it better matches the lan-
guage and style of the relevant document. The input prompt
includes (1) the original queryq i, (2) the top-kretrieved but
incorrect documentsD neg, and (3) the ground-truth docu-
ment or its summaryD pos. A simplified prompt template is
shown below:
Your goal is to refine the given query
so that it successfully retrieves
the correct document among multiple
embedded candidates.
Original query: [q i]
Correct document: [D pos]
Retrieved but irrelevant documents:
[Dneg1 ... D negk]
Rewrite the query to better match
the style and phrasing of the correct
document, so that it can be retrieved
within the top-k results.
Do not copy or use any specific
information from the correct document.
Revise the query only based on the
content and intent of the original
query itself.
The LLM then produces one or more rewritten queries
q′
i. By conditioning on both negative and positive contexts,
the model implicitly learns to align the query with the tar-
get document’s linguistic and stylistic properties rather than
performing purely semantic paraphrasing.
Verification via Re-Retrieval
Each rewritten queryq′
iis re-submitted to the same retriever.
If the ground-truth documenta inow appears within the top-
kretrieved results, the rewrite is considered ahit case. Oth-
erwise, the sample is discarded or optionally re-attempted
through iterative rewriting. Thisre-retrievalstage ensures
that the resulting pairs are empirically validated: every re-

tained rewrite demonstrably improves retrieval performance
relative to the original query.
Dataset Assembly
After verification, we collect all successful pairs(q i, q′
i)
into a new corpusD rewrite . Each entry is stored with meta-
data such as the retrieval model used, the original ranking
position, and the relative improvement in similarity score.
This corpus forms aretrieval-grounded and style-aligned
dataset that can be directly used for training or prompting
rewriter models.
Experiments
Setup
We conduct experiments on theSQuAD v1.1training
split to automatically construct a retrieval feedback–driven
rewriting dataset. Each question is treated as a queryq i, and
its corresponding passage as the ground-truth documenta i.
All passages are embedded using thee5-base-v2model and
indexed withFAISSfor dense retrieval. For each query, we
retrieve the top-3 documents based on cosine similarity.
A query is labeled as amiss caseif its ground-truth pas-
sage does not appear within the top-3 retrieved results. Only
these miss cases are used in the rewriting stage.
Dataset Construction
For each miss case, we prompt a large language model
(GPT-5, temperature = 1.0) following the template de-
scribed in Section . Each rewritten queryq′
iis re-embedded
and re-retrieved using the same retriever and index. If the
ground-truth passage appears within the top-3 results, the
pair(q i, q′
i)is accepted as avalid rewrite pair.
Out of the SQuAD training split (approximately 87k sam-
ples), about 16k (18.7%) of the queries were initially classi-
fied as miss cases. Among these, the LLM successfully pro-
duced valid rewrites for67.5%of the cases, resulting in a
final dataset of roughly11,044 verified (original↔rewrit-
ten)pairs.
Qualitative Examples.Interestingly, the LLM does not
always make queries longer or simpler; instead, it performs
context-sensitive refinementbased on the retrieval feed-
back. Simple or ambiguous questions tend to be expanded
for clarity, while overly complex or redundant ones are sim-
plified to better align with the document style. Represen-
tative examples from the constructed dataset are shown in
Table 1.
These examples illustrate that theconstructed dataset
itselfcaptures a balance betweensemantic fidelityandre-
trieval effectiveness, producing rewritten queries that are
stylistically and contextually aligned with the target corpus.
Consequently, when used to train or prompt rewriter models,
this dataset can help theminternalize retrieval feedback
signalsbeyond mere paraphrasing, enabling more effective
corpus-style reformulation.Few-Shot Validation with Constructed Dataset
To assess whether the constructed dataset can improve re-
trieval performance in real-world rewriting scenarios, we
conducted a lightweightoffline few-shot validationexperi-
ment. Rather than fine-tuning a rewriter model, we treat the
dataset as a retrieval-augmented source of in-context exam-
ples. Each validation query is embedded, and the five most
similar (original, rewritten) pairs from our dataset are re-
trieved as few-shot demonstrations. The following prompt
template was used:
You are a query rewriter for retrieval.
Rewrite the user query into a
corpus-style, declarative search query
that best matches documentation
phrasing.
Maintain meaning, avoid hallucinations,
prefer canonical terminology,
and keep it concise.
<Few-shot Examples>
Example 1
Input: [original query]
Rewritten: [refined query]
...
<Task>
User Query: [user query]
Rewrite only the query. Respond with
the rewritten query text only|no
explanations.
This setup simulates a realistic offline environment where
rewrites are generated on demand using retrieved exemplars,
without task-specific training.
Discussion.This experiment demonstrates that the con-
structed dataset isoperationally usefuleven without super-
vised training. When used as few-shot exemplars, the ver-
ified pairs help the LLM produce retrieval-aligned rewrites
that better reflect the linguistic style of documentation. Be-
cause the SQuAD dataset primarily contains short and well-
structured factual questions, rewriting is not beneficial for
all queries—many original questions already align with the
corpus style.
Nevertheless, for complex or stylistically divergent
queries, such as those involving descriptive expressions, im-
plicit entities, or person-specific phrasing, few-shot rewrit-
ing notably improved the ranking of relevant documents.
Representative examples are shown in Table 2. Overall,
while rewriting may not universally enhance retrieval on
simple datasets, these results highlight the potential of re-
trieval feedback–driven data as a foundation for developing
future style-aware rewriter models.
Moreover, since the rewriting process implicitly adapts to
the phrasing style, terminology, and formality of the corpus,
the generated queries naturally align with how information
is expressed within each domain—suggesting strong appli-
cability to real-world RAG systems.

Original Query Rewritten Query
Certain dogs are bred to help fishermen with what? What specific task are certain dog breeds traditionally used for by
fishermen?
What was Gaddafi’s nationality? What was Muammar Gaddafi’s nationality (the country he was a
citizen of)?
When copper and tin is mixed, what is made? What alloy is formed by combining copper and tin?
What was the new name given to Plymouth Dock in 1824? What was Plymouth Dock renamed to in 1824?
How many did Crazy in Love sell to become one of the greatest
selling singles in history?How many copies did the single “Crazy in Love” sell worldwide?
Table 1: Examples of (original, rewritten) query pairs from the constructed dataset. The rewriting model dynamically adjusts
query specificity and phrasing to better align with the style and terminology of the target corpus—sometimes simplifying overly
detailed questions, and other times expanding vague ones for clarity.
Original Query Rewritten Query Ground-Truth Rank (Top-k)
What is the name of a Time Lord that Doc-
tor Who has fought?Which Time Lord adversaries has the Doc-
tor fought in theDoctor Whoseries?From not in top-10→Top-2
In lands attributed to what tribe are found
remains of large settlements?In territories associated with which tribe
have archaeologists discovered remains of
large settlements?From Top-8→Top-1
Besides the walk to the church, what else
was left out of the day’s celebration?What other parts of the day’s celebra-
tion were omitted besides the walk to the
church?From Top-5→Top-2
Most imperialism was carried out using
which method of transport?Which mode of transportation was primar-
ily used to carry out imperial expansion?From not in top-10→Top-1
Table 2: Few-shot rewriting results using five retrieved example pairs from the constructed dataset. Each rewritten query im-
proves retrieval ranking without explicit fine-tuning, indicating that the dataset serves as an effective retrieval-grounded refer-
ence for few-shot rewriting.
Conclusion and Future Work
In this work, we presentedReFeed, a framework for con-
structing aretrieval feedback–driven, style-aware query
rewriting dataset. By leveraging retrieval outcomes as weak
supervision, the framework systematically collects (original,
rewritten) query pairs that empirically improve document
recall. Through LLM-guided rewriting, re-retrieval verifi-
cation, and curated dataset assembly, the resulting corpus
achieves a natural balance betweensemantic fidelityandre-
trieval effectiveness.
The rewriting process implicitly adapts to the phrasing
style, terminology, and formality of the corpus, allowing the
generated queries to naturally align with how information is
expressed within each domain. This corpus-adaptive prop-
erty makes the approach particularly suitable for retrieval-
augmented generation systems, where query formulation
must reflect the linguistic characteristics of domain-specific
documents.
Preliminary few-shot validation further demonstrated that
these verified examples can improve retrieval performance
even without model fine-tuning, highlighting the practical
utility of the proposed framework for real-world applica-
tions.Future Work.Building on this foundation, our next step
is to develop anoffline style-aware rewriting framework
using this dataset to train small language models (SLMs).
Unlike the current few-shot setup, the trained model will di-
rectly internalize corpus-specific linguistic styles, enabling
efficient and robust rewriting for domain retrieval. To en-
sure scalability, we plan to integrate aselective rewrit-
ing modulethat determines whether a user query requires
rewriting before retrieval—avoiding unnecessary computa-
tion while preserving recall gains. We also aim to evaluate
this end-to-end system on industrial RAG pipelines, inves-
tigating trade-offs between rewrite coverage, retrieval accu-
racy, and latency. Ultimately, we envision this framework as
a lightweight yet effective approach for improving informa-
tion access in retrieval-augmented conversational systems.
References
Anantha, R.; Vakulenko, S.; Tu, Z.; Longpre, S.; Pulman,
S.; and Chappidi, S. 2021. Open-Domain Question Answer-
ing Goes Conversational via Question Rewriting Transfor-
mation. InProceedings of NAACL 2021.
Baek, S.; Oh, B.-K.; Park, Y .; and Seo, M. 2024. Crafting the
Path: Robust Query Rewriting for Information Retrieval. In

Findings of the Association for Computational Linguistics:
ACL 2024.
Dhole, K.; Zhao, X.; Tang, Y .; and Ma, X. 2024. Generative
Query Reformulation for Effective Ad-hoc Search.arXiv
preprint arXiv:2308.00415.
Elgohary, A.; Peskov, D.; and Boyd-Graber, J. 2019. Can
You Rephrase That? A Benchmark for Clarification Ques-
tion Generation and Answering. InProceedings of ACL
2019.
Mao, Y .; Li, Z.; Song, Y .; and Zhang, Y . 2024. RaFe:
Ranking Feedback Improves Query Rewriting for Retrieval-
Augmented Generation. InFindings of EMNLP 2024.
Shi, W.; Wang, H.; Wang, Y .; Gao, J.; Chen, Y .; Ma, Y .; and
Liu, S. 2023. REPLUG: Retrieval-Augmented Black-Box
Language Models. InNeurIPS 2023.
Wang, B.; Zhu, Y .; Zhao, J.; Chen, M.; and Xu, R. 2024.
MaFeRw: Query Rewriting with Multi-Aspect Feedbacks
for Retrieval-Augmented Large Language Models. InPro-
ceedings of ACL 2024.