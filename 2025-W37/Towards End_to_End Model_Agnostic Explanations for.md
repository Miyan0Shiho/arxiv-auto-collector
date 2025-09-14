# Towards End-to-End Model-Agnostic Explanations for RAG Systems

**Authors**: Viju Sudhi, Sinchana Ramakanth Bhat, Max Rudat, Roman Teucher, Nicolas Flores-Herr

**Published**: 2025-09-09 11:47:40

**PDF URL**: [http://arxiv.org/pdf/2509.07620v1](http://arxiv.org/pdf/2509.07620v1)

## Abstract
Retrieval Augmented Generation (RAG) systems, despite their growing
popularity for enhancing model response reliability, often struggle with
trustworthiness and explainability. In this work, we present a novel, holistic,
model-agnostic, post-hoc explanation framework leveraging perturbation-based
techniques to explain the retrieval and generation processes in a RAG system.
We propose different strategies to evaluate these explanations and discuss the
sufficiency of model-agnostic explanations in RAG systems. With this work, we
further aim to catalyze a collaborative effort to build reliable and
explainable RAG systems.

## Full Text


<!-- PDF content starts -->

Towards End-to-End Model-Agnostic
Explanations for RAG Systems
Viju Sudhi∗, Sinchana Ramakanth Bhat†, Max Rudat†, Roman Teucher†and
Nicolas Flores-Herr†
Abstract
Retrieval Augmented Generation (RAG) systems, despite their growing popular-
ity for enhancing model response reliability, often struggle with trustworthiness and
explainability. In this work, we present a novel, holistic, model-agnostic, post-hoc ex-
planation framework leveraging perturbation-based techniques to explain the retrieval
and generation processes in a RAG system. We propose different strategies to evaluate
these explanations and discuss the sufficiency of model-agnostic explanations in RAG
systems. With this work, we further aim to catalyze a collaborative effort to build
reliable and explainable RAG systems.
Keywords:Explainability,RetrievalAugmentedGeneration,LargeLanguageMod-
els
1 Introduction
RAG systems aim at improving response generation of Large Language Models (LLMs) [1, 3,
4]. A typical RAG system is composed of a retriever in conjunction with a generator. Given a
user questionq, the retriever from a collection of documents returns the most relevant documents
di. These documents together with an instruction compose the promptxwhich is then fed to
the LLM-based generator. The generator finally returns a responseyto the user - more reliable
than the one it generates from its model weights alone. However, since the models used are not
intrinsically explainable, end-users often find such RAG systems less trustworthy [6]. To mitigate
this, in this work§, we borrow ideas presented in our earlier works attempting to individually
explain the retriever [2] and the generator [5]; and combine these strategies to build a holistic
end-to-end framework towards model agnostic explanations for RAG systems. Our framework can
explain retrievers (utilizing dense embedding models) and generators (open-source or proprietary)
in an open-book QA setup. We present the framework as a "one-fits-all" solution considering the
plethora of emerging embedding and generator models.
2 Methodology
In the proposed end-to-end explanation framework, we aim at:(i) explaining the retrieval
processto answer why the retriever retrieveddgivenq, and(ii) explaining the generation
processto answer why the generator generatedygivenx. While the former helps the user under-
stand the contribution of individual document terms, the latter helps them to understand which
parts of the input the LLM focused on to generate the final answer. By presenting component-wise
explanations, we allow the user to be a better judge in carefully choosing different retriever and
generator models to finally compose their reliable and explainable RAG system. An exemplary
visualization is presented in 2.
The explainer starts with deciding what should be explained:d iin the case of the retriever
andxin the case of the generator. As illustrated in Figure 1, we then employ the same set of
components as outlined below to explain the retrieval and generation processes.
∗Correspondence to viju.sudhi@uni-bielefeld.de
∗Affiliation: Bielefeld University (Work done while the author was affiliated with Fraunhofer IAIS)
†Affiliation: Fraunhofer IAIS
§Code: https://github.com/fraunhofer-iais/explainable-lmsarXiv:2509.07620v1  [cs.IR]  9 Sep 2025

2 Towards End-to-End Model-Agnostic Explanations for RAG Systems
Figure 1: Overview of the explanation framework. For retriever explanations, the retrieved
documentdisfedintotheexplainer. Thisisthendecomposedtoobtainfeaturesf iandperturbedto
obtain perturbationsp i. The similarity between the perturbed documentp iand the user question
is computed ass i. This is compared against the reference scores dfinally resulting in feature
importance weightw i. For generator explanations, the input to the explainer is the promptx.
This is decomposed to obtainf iand based on each feature, the input is perturbed to obtainp i.
For each perturbed input, the generator generates a responser iwhich is then compared against
the reference responseyleading to the feature importance weightw i.
•Decompose:Firstly, the inputs (dorx) are decomposed to the featuresf iaccording to
the preferred granularity. For retriever explanations, we useword-levelgranularity to study
the significance of individual terms in the documentd. The generator explanations should
shed light into the different parts of the input promptxand therefore, we usesentence-level
granularity for generator explanations.
•Perturb:Based on each of the decomposed featuresf i, the inputs (dorx) are perturbed to
yield the perturbationsp i. We examined different perturbation strategies∗and observed that
the simplestleave one feature outstrategy (where each feature in the input is individually
left out) yields the most intuitive explanations.
•Retrieve or Generate:The perturbed inputsp iare then fed to the corresponding com-
ponent. For retriever explanations, we (i) first, utilize the base retriever which yielded the
relevant document to now embed the perturbed inputs and (ii) then, compute the cosine
similarity score against the user questions i. For generator explanations, we (i) first, feed the
perturbed inputs to the base generator and (ii) then, generate responsesr ifor eachp i.
•Compare:The resulting retriever scoress iand the generator responsesr iare compared
against the reference retriever scores dand the reference generator responseyrespectively to
study the relative similarities (of the scores and the texts). We quantify the importance of the
featuref ito the similarity scores further normalized and negated to obtain the dissimilarity
scores denoted asw i. These weights indicate the feature importance for the corresponding
process.
The components summarized above are agnostic to the language and models used, thereby
allowing the usage of any retriever or generator model. In our open-source user interface, the
features are marked with different color scales to help users easily identify and distinguish the
most important features from the others.
∗We advise the readers to find more details about the strategies in the work [5].

Sudhi et al. 3
Figure 2: An exemplary visualization of the explanation framework.
3 Evaluation
We evaluated the explanations from our framework considering different facets and report our
results briefly as follows. We advise the readers to refer to our work BiTe-REx [2] and RAG-Ex
[5] for more details on evaluation methods and metrics.
Intuitiveness of the explanationsAs the first step, we wanted to assess how intuitive the end-
users find the explanations from the framework. We designed user studies in which we instructed
users to annotate features that they believed were significant for retrieval or generation. Upon
comparing these features against the ones the explainer yielded as significant for the process, we
were able to understand how similar the user and the framework "explain" the processes. Our
retriever explanations were found to be 64.7%completeand the generator explanations yielded an
F1 score of 76. 9% against the end-user annotations.
Co-relation with downstream task model performanceWe observed that the models’
downstream task performance was consistently higher when the generator explainer yielded more
intuitive features as the significant ones. In other words, if the modellookedinto the most relevant
parts of the context - in this use case, the question itself and the part of the context where the
potential answer lies, its chances to generate the correct answer are higher.
Sufficiency of model-agnostic explanationsWe also studied how users received the model-
agnostic generator explanations against the available model-intrinsic approaches. They rated our
frameworktobe3.42completeand3.45correctagainst3.98and4.04, respectively, forthecompared
model-intrinsic approach. Despite the gap in these ratings, we strongly advocate model-agnostic
explanations for RAG systems, given the flexibility it offers by accommodating both open-source
and proprietary models.
4 Limitations
We acknowledge that the framework is limited in its utility considering use cases other than RAG
systems. We chooseword-levelgranularity for explaining the retrieval process overlooking how
dense embedding models are otherwise trained. Unlike model-intrinsic approaches which aim at
understandingthetechnicalworkingsofthelanguagemodelsingeneral, model-agnosticapproaches
like ours tend to provideapproximatedexplanations to the user. This caters rather better to the
end-users of the system, but does not necessarily expose the working of the model by itself.

4 Towards End-to-End Model-Agnostic Explanations for RAG Systems
5 Conclusion
We present an end-to-end model agnostic framework to explain the retrieval and generation
processes in RAG systems. As future work, we plan to extend the evaluation of our framework
by extending our qualitative analysis to evaluate intuitiveness and satisfaction; asking participants
to rate explanations based on clarity, relevance, and trustworthiness. We also aim to investigate
better quantitative measures for evaluating explanations.
References
[1] Patrick Lewis et al. “Retrieval-augmented generation for knowledge-intensive nlp tasks”. In:
Advances in neural information processing systems33 (2020), pp. 9459–9474.
[2] Viju Sudhi et al. “BiTe-REx: An Explainable Bilingual Text Retrieval System in the Au-
tomotive Domain”. In:Proceedings of the 45th International ACM SIGIR Conference on
Research and Development in Information Retrieval. SIGIR ’22. Madrid, Spain: Association
for Computing Machinery, 2022, pp. 3251–3255.isbn: 9781450387323.doi: 10.1145/3477495.
3531665.url: https://doi.org/10.1145/3477495.3531665.
[3] Yunfan Gao et al. “Retrieval-augmented generation for large language models: A survey”. In:
arXiv preprint arXiv:2312.109972 (2023).
[4] ShailjaGupta,RajeshRanjan,andSuryaNarayanSingh.“Acomprehensivesurveyofretrieval-
augmented generation (rag): Evolution, current landscape and future directions”. In:arXiv
preprint arXiv:2410.12837(2024).
[5] Viju Sudhi et al. “RAG-Ex: A Generic Framework for Explaining Retrieval Augmented Gen-
eration”. In:Proceedings of the 47th International ACM SIGIR Conference on Research and
Development in Information Retrieval. SIGIR ’24. Washington DC, USA: Association for
Computing Machinery, 2024, pp. 2776–2780.isbn: 9798400704314.doi: 10.1145/3626772.
3657660.url: https://doi.org/10.1145/3626772.3657660.
[6] Yujia Zhou et al. “Trustworthiness in retrieval-augmented generation systems: A survey”. In:
arXiv preprint arXiv:2409.10102(2024).