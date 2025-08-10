# Hallucination to Truth: A Review of Fact-Checking and Factuality Evaluation in Large Language Models

**Authors**: Subhey Sadi Rahman, Md. Adnanul Islam, Md. Mahbub Alam, Musarrat Zeba, Md. Abdur Rahman, Sadia Sultana Chowa, Mohaimenul Azam Khan Raiaan, Sami Azam

**Published**: 2025-08-05 19:20:05

**PDF URL**: [http://arxiv.org/pdf/2508.03860v1](http://arxiv.org/pdf/2508.03860v1)

## Abstract
Large Language Models (LLMs) are trained on vast and diverse internet corpora
that often include inaccurate or misleading content. Consequently, LLMs can
generate misinformation, making robust fact-checking essential. This review
systematically analyzes how LLM-generated content is evaluated for factual
accuracy by exploring key challenges such as hallucinations, dataset
limitations, and the reliability of evaluation metrics. The review emphasizes
the need for strong fact-checking frameworks that integrate advanced prompting
strategies, domain-specific fine-tuning, and retrieval-augmented generation
(RAG) methods. It proposes five research questions that guide the analysis of
the recent literature from 2020 to 2025, focusing on evaluation methods and
mitigation techniques. The review also discusses the role of instruction
tuning, multi-agent reasoning, and external knowledge access via RAG
frameworks. Key findings highlight the limitations of current metrics, the
value of grounding outputs with validated external evidence, and the importance
of domain-specific customization to improve factual consistency. Overall, the
review underlines the importance of building LLMs that are not only accurate
and explainable but also tailored for domain-specific fact-checking. These
insights contribute to the advancement of research toward more trustworthy and
context-aware language models.

## Full Text


<!-- PDF content starts -->

Hallucination to Truth: A Review of Fact-Checking and Factuality
Evaluation in Large Language Models
Subhey Sadi Rahman1, Md. Adnanul Islam1,†, Md. Mahbub Alam1,†,
Musarrat Zeba1,†, Md. Abdur Rahman1, Sadia Sultana Chowa2,
Mohaimenul Azam Khan Raiaan1,3, Sami Azam3,*
1Department of Computer Science and Engineering, United International University, Dhaka 1212, Bangladesh
2Department of Computer Science and Engineering, Daffodil International University, Dhaka-1341, Bangladesh
3Faculty of Science and Technology, Charles Darwin University, Casuarina, NT 0909, Australia
aEqual Contributions.
*Corresponding Author: sami.azam@cdu.edu.au
Abstract
Large Language Models (LLMs) are trained on vast and diverse internet corpora that often include inaccurate or misleading
content. Consequently, LLMs can generate misinformation, making robust fact-checking essential. This review systematically
analyzes how LLM-generated content is evaluated for factual accuracy by exploring key challenges such as hallucinations,
dataset limitations, and the reliability of evaluation metrics. The review emphasizes the need for strong fact-checking
frameworks that integrate advanced prompting strategies, domain-specific fine-tuning, and retrieval-augmented generation
(RAG) methods. It proposes five research questions that guide the analysis of the recent literature from 2020 to 2025,
focusing on evaluation methods and mitigation techniques. The review also discusses the role of instruction tuning, multi-
agentreasoning, andexternalknowledgeaccessviaRAGframeworks. Keyfindingshighlightthelimitationsofcurrentmetrics,
the value of grounding outputs with validated external evidence, and the importance of domain-specific customization to
improve factual consistency. Overall, the review underlines the importance of building LLMs that are not only accurate
and explainable but also tailored for domain-specific fact-checking. These insights contribute to the advancement of research
toward more trustworthy and context-aware language models.
Keywords: Fact-checking, large language model, hallucination, LLM, retrieval augmented generation
1 Introduction
The growing use of Large Language Models (LLMs) in news,
healthcare, education, and law means that the accuracy of
their content now affects important real-world decisions [1, 2].
These models often generate information that sounds reli-
able, but can be false or unsupported, increasing the risk
of misinformation reaching the public [3]. Because new tech-
niques, datasets, and benchmarks for LLM fact checking are
being published at a rapid pace, it has become difficult for
researchers and practitioners to keep track of what actually
works [4]. This paper provides a systematic review of fact
checking in the context of LLMs such as GPT-4 and LLaMA.
This review is needed to organize the latest knowledge, high-
light the main challenges, and point out areas that require
further research. This is especially important at this stage,
as LLMs are increasingly shaping the way information is
produced, accessed, and trusted throughout society.1.1 Challenges
Fact-checking the output of LLM systems faces several in-
timidating challenges. Among the most notable challenges is
the absence of standardized evaluation metrics. Currently
used ones quantify surface-level similarity and not factual
consistency [5, 6] and are therefore more likely to detect
nuanced errors.
Another key limitation of LLMs is hallucination. They
tend to produce linguistically consistent but factually inac-
curate or entirely fictional text [3]. This is due to language
modeling and training on potentially stale or biased data in
a data-driven, probabilistic way [7, 8]. Dataset quality is also
critical for fact-checking system performance. The majority
of benchmarks either have no realistic complexity of real-
world claims and are domain independent or are too narrow
to be generalized. Furthermore, datasets with unbalanced
classes can influence the model response and make the system
less robust on a wide spectrum of topics [9].
1arXiv:2508.03860v1  [cs.CL]  5 Aug 2025

Figure 1 : The fundamental content structure and categorization of this survey.
1.2 Emerging Innovations
To address these issues, various innovations such as Retrieval-
Augmented Generation (RAG), instruction tuning [10, 11],
domain-specific fine-tuning [12], multi-agent systems [13, 14],
automated self-correction and feedback mechanisms [15, 16],
and integration with knowledge graphs [17, 18] have been
suggested to overcome these limitations. RAG stands out as
a key technique that combines LLMs and external retrieval
systems, aligning generated outputs with verifiable sources.
RAG architectures have shown notable results in factuality
and explainability [19] by allowing LLMs to access and cite
external knowledge in real time. These methods are often
augmented with advanced prompting techniques, such as hi-
erarchical step-by-step reasoning and multi-agent collabora-
tion [20, 21, 22].
1.3 Purpose and Research Questions
Theobjectiveofthereviewistocriticallyevaluatethecurrent
prospects of LLM-based fact-checking systems, identify key
issues, and investigate the performance of existing solutions.
Observing recent growing trends, this paper aims to buildmore accurate, transparent, and scalable fact-checking sys-
tems in LLMs. The review is inspired by five fundamental
research questions.
1.RQ1:What evaluation metrics are used to assess LLM-
based fact-checking systems?
Rationale: To understand how system performance is
measured and identify potential limitations or inconsis-
tencies in current evaluation methods.
2.RQ2:How do hallucinations affect the reliability of
LLM fact-checking?
Rationale: Hallucinations are caused by LLM due to
the vast amount of training data containing refined and
unverified information. Their output often contains
hallucinated answers, which directly affect the trustwor-
thiness and accuracy of the LLM fact check.
3.RQ3:What datasets are commonly used for training
and evaluating fact-checking models?
Rationale: To assess the quality, coverage, and impact
of the dataset on generalizability.
4.RQ4:How do prompting strategies and fine-tuning
2

influence fact-checking performance?
Rationale: To analyze optimization techniques for
LLMs in fact-checking contexts.
5.RQ5:How is RAG integrated into fact-checking?
Rationale: To evaluate the benefits and challenges of
combining retrieval mechanisms with generative mod-
els.
1.4 Contributions
This paper brings three contributions to research on fact-
checking using LLMs:
First.It offers a comprehensive taxonomy of evaluation met-
rics that categorizes widely used techniques by their method-
ological focus.
Second. The review combines a wide range of approaches
to mitigate hallucinations in LLM output. These range from
fine-tuning through domain-specific data to instruction tun-
ing, adversarial training, and self-supervised feedback meth-
ods like Self-Checker [20]. In addition, multi-agent archi-
tecture and multi-step reasoning strategies are explored as
enhancement strategies for factuality and explainability.
Third.The paper offers novel insight into how dataset char-
acteristics such as domain specificity, annotation quality, and
multilingual coverage affect the performance of fact checking
systems [9].
1.5 Paper Organization
This paper is organized into eight sections: Section 2 reviews
existing research on LLM-based fact checking and highlights
key gaps. Section 3 explains the methodology, including how
the studies were selected and analyzed. Section 4 presents
the findings based on our five core research questions. Sec-
tion 5 discusses the implications, challenges, and limitations.
Section 6 draws attention to open issues and challenges. Sec-
tion 7 highlights the analysis of future research agendas, and
Section 8 concludes the paper with key insights for building
more accurate and reliable LLM-based fact-checking systems.
Figure 1 illustrates the overall structure of the paper.
2 Related Works
LLM-generated texts are now widely used in various impor-
tant sectors. Therefore, it is important to ensure their factual
accuracy and reliability to maintain trust in these applica-
tions.
Several researchers have explored fact-checking methods
in the context of LLMs. For example, Vykopal et al. [23]
conduct a survey of approaches and techniques used in au-
tomated fact checking using generative LLMs, such as claimdetection, evidence retrieval, and fact verification. They in-
troduce the concept of RAG, which can be used to mitigate
challenges such as hallucinations and the use of out-of-date
model knowledge utilizing external evidence. However, it
does not address the effects of domain-specific training on
LLM-based fact checking, the challenges of RAG implemen-
tation, or how the quality of the dataset, the specificity of the
domain, and the evaluation metrics influence the effectiveness
of LLM. Dmonte et al. [24] also explore LLM-based claim
verification by analyzing full-system pipelines that include
key stages such as evidence retrieval, prompt construction,
and explanation generation. They review RAG techniques
such as iterative retrieval and claim decomposition, which
allows them to address issues such as hallucinations and the
challenges of verifying complex or long claims. Evaluation
metrics like FactScore and FEVER Score are highlighted to
assess and improve factual accuracy. Similarly to [23], this
paper overlooks dataset quality, domain-specific challenges,
and RAG implementation issues, which are significant in
evaluating the reliability of LLM-based fact check.
Inanotherwork, Augensteinetal. [3]studythechallenges
of factual correctness in LLM. They focus on hallucinations,
knowledge editing to reduce hallucinations, and the impact of
misinformation that AI can spread, which also includes con-
cerns about trust and misuse. They propose some mitigation
strategies, such as RAG, although the discussion lacks depth
with regard to domain-specific RAG implementations and
the associated challenges. Key evaluation metrics for LLM-
based fact-checking systems are also discussed to assess fac-
tuality, consistency, and text quality, including TruthfulQA,
FactScore, GPTScore, G-Eval, SelfCheckGPT, BERTScore,
andMoverScore. However, thepaperlacksadiscussionofsev-
eral critical technical aspects and challenges, including model
interpretability, explainability, and practical implementation
and integration of the proposed mitigation techniques in real-
world settings. Wang et al. [25] offers a detailed survey of
the factuality of LLM, providing a taxonomy of hallucination
types and errors in both unimodal and multimodal tasks.
A key contribution is mapping factuality challenges to algo-
rithmic solutions and proposing improvements to factuality-
aware model calibration. However, the paper does not dis-
cuss domain-specific challenges of RAG, prompt design, fact-
checkingcomponentintegration, orsystem-levelarchitectures
and deployment considerations for LLM verification environ-
ments.
Although most previous work has briefly discussed eval-
uation metrics, hallucination effects, and issues related to
prompt- or fine-tuning methods, it has not examined the
impact of datasets, particularly domain-specific ones. In
addition, there is a wide gap between practical challenges
and considerations of RAG and domain-specific implementa-
tion. Most of the papers gave RAG only a passing reference
without discussing larger domain-specific problems. However,
our work fully discusses all the primary areas of research
3

Table 1 : Comparison of different papers concerning the key RQs. The following RQs drive our review: RQ1: Evaluation
Metrics and Gaps; RQ2: Hallucination Effects and Mitigation; RQ3: Datasets and Impact; RQ4: Prompt and Fine-tuning;
Domain-specific Training Effects; and RQ5: RAG and Domain-Specific Implementation Challenges.
PaperMetrics
& GapsHallucination
& MitigationDatasets
& ImpactPrompt, Fine-tuning
& Domain TrainingRAG
& DomainKey Notes
Vykopal et al. [23] ✗ ✓ ✗ ✓ ✗ 1. Skips hallucination effects (RQ2)
2. Only mentions domain-specific fine-
tuning (RQ4)
3. Only mentions RAG’s impact (RQ5)
Dmonte et al. [24] ✓ ✓ ✓ ✓ ✗ 1. Skips domain-specific datasets (RQ3)
2. Only mentions domain-specific fine-
tuning (RQ4)
3. Only mentions RAG’s impact (RQ5)
Augenstein et al. [3] ✓ ✓ ✓ ✓ ✗ 1. Skips domain-specific datasets (RQ3)
2. Mentions prompt design in (RQ4)
3. Only mentions RAG’s impact (RQ5)
Wang et al. [25] ✓ ✓ ✓ ✓ ✗ Only mentions RAG’s impact (RQ5)
Ours ✓ ✓ ✓ ✓ ✓ Fully addresses all RQs
questions, such as under-discussed areas, and thus provides
a more detailed overview of the field. Table 1 presents a
summary of the existing survey papers and shows how they
relate to the key research questions of this study.
3 Methods
To explore how LLMs can be applied to fact-checking, we
adopted a structured yet practical approach inspired by well-
established research methods [26]. We began by designing a
detailed review plan that outlined what we wanted to study,
where we would look for relevant literature, and what kind of
studies we would include. We then conducted a broad search
across leading academic databases, using a combination of
manual screening and automated tools to identify the most
relevant and up-to-date studies.
The review process consisted of three key phases: (i) plan-
ning, (ii) data collection and analysis, and (iii) synthesis and
reporting. At the end of our review, we brought together the
insights from the selected studies to highlight what is already
known, where the gaps are, and what future research should
aim to address. By following this clear, step-by-step method-
ology, we have ensured that our findings are informative and
reliable for advancing the use of LLMs in fact-checking tasks
[27].
3.1 Search Strategy
To ensure comprehensive coverage of relevant literature, we
designed a focused search strategy using well-defined key-
words and Boolean operators. Our queries were tailored
to capture publications related to the use of LLMs in fact-
checking and related tasks. The following keyword combina-
tions were used across databases such as IEEE Xplore, ACMDigital Library, Wiley Online Library, ScienceDirect, Web of
Science, arXiv, and Google Scholar:
•"large language models" AND "fact-checking"
•"LLM" AND "misinformation detection"
•"automated fact verification" AND "LLMs"
•"factuality evaluation" AND "natural language
processing"
•"hallucination" AND "LLMs"
•"LLM hallucination"
•"hallucination mitigation" AND "large
language models"
•"hallucination detection" AND "large language
models"
•"fact-checking datasets" OR "benchmark
datasets for fact verification"
•"retrieval-augmented generation" AND
"fact-checking"
•"rag" AND "LLM"
•"rag" AND "fact-checking"
•"fine-tuning" AND "fact verification models"
•"prompt engineering" AND ("truthful
generation" OR "fact-checking")
•"LLM-based fact verification" AND "NLP"
4

Our selection process targeted significant areas, including
fact-checking methods, model assessment techniques, dataset
analysis considerations, and optimization procedures. Table
2 reports the details of the libraries, along with the number
of publications we selected for further analysis.
Table 2: Number of articles selected from online libraries for
further analysis.
No. Library Quantity
1 IEEE Xplore 635
2 ACM Digital Library 769
3 Scopus 809
4 Wiley Online Library 153
5 Web of Science 281
6 Others 997
Total 3644
3.2 Selection Criteria
Each article is carefully assessed using our evaluation crite-
ria to determine whether it meets the inclusion or exclusion
requirements. The key inclusion criteria (IC) and exclusion
criteria (EC) are shown in Figure 2.
Figure 2 : Inclusion and exclusion criteria for article selec-
tion.
3.3 Article Selection
We conducted a systematic review of articles from leading
conferences and journals at the intersection of fact-checking,
NaturalLanguageProcessing(NLP),andLLMs. Anoverview
of the number of selected journals, conference proceedings,and preprints is shown in Figure 3. The review focused
Figure 3 : A visual summary of the articles selected from
journals, conference proceedings, and preprints.
on studies published between 2020 and 2025 that employed
LLMs to verify external claims or factual content, exclud-
ing works solely analyzing hallucinations or internal factual
consistency. From an initial pool of 3,644 records retrieved
from various academic databases, we applied a multi-stage
screening process, comprising duplicate removal, title and
abstract screening, and full-text evaluation. Finally, we have
selected 57 articles that meet our inclusion criteria and align
with the objectives of this review. The article selection work-
flow is illustrated in Figure 4, which outlines the stages of
identification, screening, eligibility, and final inclusion.
4 Findings from the Research Ques-
tions
In this section, we present a comprehensive key result finding
focusing on evaluation metrics, the impact of hallucinations
in LLMs, datasets, prompt designing and fine-tuning, and
integration of RAG.
4.1 Evaluation Metrics for Fact-Checking
Systems (RQ1)
Evaluating LLMs for fact-checking and related areas such as
grounded generation, summarization, and error detection is
a crucial and evolving field. It addresses one of the most
significant challenges in LLM deployment: their tendency to
“hallucinate,” or generate text that sounds plausible but is
factually incorrect [10, 18]. Evaluation in this context typi-
cally involves checking the model’s outputs against provided
evidence or reliable external sources [10]. Previously, a wide
5

Figure 4 : The article selection and screening process of this survey.
Figure 5 : Taxonomy of evaluation metrics for fact-checking systems.
array of methods has been developed, including using LLMs themselves as evaluators and building benchmarks that unify
6

datasets and tasks [10]. Figure 5 illustrates a comprehensive
summary of the complete set of metrics.
4.1.1 Traditional Classification Metrics
Most commonly, the evaluation tasks are approached as clas-
sification problems, which determines whether a claim is true
or identifies errors in responses [28, 29]. Metrics like Accu-
racy [30, 28, 31, 32, 33, 34], Precision, Recall, and F1-score
[35,21,36,15,37,38]arewidelyusedforthesetasks. Inmulti-
class scenarios, such as classifying statements as supported,
refuted,orinconclusive,macro-averagedversionsofthesemet-
rics are employed [35, 21, 28]. These also serve as standard
measures in detection tasks [39]. For short-form responses,
token-level precision with annotated answers is typical [15].
These metrics offer quantitative performance indicators,
making it easier to compare models or methods directly
[21, 30, 28, 32]. They often reduce complex outputs to a
binary (i.e., correct or incorrect) judgment and overlook rea-
soning quality or nuanced inaccuracies [18]. They may also
be misleading in datasets with imbalanced labels [39].
4.1.2 Lexical and Semantic Overlap Metrics
When evaluating text generation tasks, such as summariza-
tion or dialogue, overlap-based metrics are commonly used.
Lexical overlap metrics such as BLEU-4, METEOR,and chrF
assess surface-level similarity [15]. ROUGE evaluates the
extent to which summaries or explanations capture the core
content [31, 40]. Semantic similarity metrics like BERTScore
[3, 15, 41, 34], BLEURT [15], and cosine similarity measures
[42] assess deeper semantics. For multimodal outputs, CLIP-
Score compares image and caption embeddings to evaluate
alignment [43].
These metrics are standard for fluency and content simi-
larity and can capture meaning beyond exact word matches
[43, 15, 42]. However, they do not measure factual correct-
ness. High overlap or semantic scores may still correspond to
factually incorrect content. Older semantic metrics may not
align well with the reasoning capabilities of modern LLMs
[3, 40, 41].
4.1.3 Factuality-Specific and Grounding Metrics
Specialized metrics have been developed to directly evaluate
factual consistency. These go beyond surface similarity and
focus on whether the model’s claims align with evidence. For
example, benchmarks like LLM-AGGREFACT use detailed
human annotations to assess support levels for claims [10].
ReaLMistake focuses on binary error detection, especially in
reasoning and context alignment [29]. Other tools like the
LEAF Fact-check Score compute the ratio of factually sup-
ported sentences to the total response [30], while Knowledge
F1 (KF1) measures the overlap between human-used and
model-usedknowledge[15]. FactScore-Bioclassifiesresponsesbasedonretrievedevidence[44], andsomemethodsaggregate
multiple signals into a final factuality probability score [39].
Metrics like Insight Mastery Rate (IMR) and Justification
Flaw Rate (JFR) assess explanatory quality [45]. Natural
Language Inference (NLI) techniques and textual entailment
tasks also serve to classify claims as supported, refuted, or
unverifiable [35, 46, 47, 33, 11].
Challenges include the complexity of strict entailment in
language [48], potential metric bias [3], and reliance on high-
quality annotated evidence. The Logical Consistency Matrix
measures coherence under logical manipulations like negation
or conjunction [17, 34]. Hit Rate (HR), used in evidence
retrieval, tracks how often relevant documents are among the
top results [49]. These methods are tailored for evaluating
truthfulnessandprovidenuancedinsightsintofactualground-
ing [10, 30, 29, 44].
4.1.4 LLM-Based and Prompt-Based Evaluation
A growing trend involves using LLMs themselves as evalua-
tors. This includes having LLMs classify responses as cor-
rect or flawed, and rate the factual accuracy of claims when
prompted [10, 29]. The LLM-as-a-judge paradigm treats
powerful language models as referees that compare and score
the outputs of other models, which often produces results
that closely align with human judgments [45, 50]. LLMs are
also widely used in tasks such as decomposing complex claims
[10, 51, 42], generating probing questions [52, 30, 51, 53], and
selecting relevant evidence [52, 30, 20, 42]. While techniques
like zero-shot, few-shot, Chain-of-Thought, ReAct, and HiSS
are not metrics themselves, their impact is assessed using
factuality metrics [21, 12, 54, 32]. Some systems even use
LLMs to rate and verify retrieved documents [51], or use
them to check for hallucinations [15]. The Preservation Score
evaluates how much original content remains intact after
hallucination correction [49]. LLMs enable more nuanced
andcontext-sensitiveevaluationsthantraditionalmetrics[55].
They can reduce human effort in evaluation tasks as well [45].
Their performance can be inconsistent due to sensitivity to
prompt phrasing, and they may introduce bias or misjudg-
ments [29, 55].
4.1.5 Human Evaluation
Despite automation advances, human evaluation remains es-
sential, especially for complex and subjective aspects like ex-
planationclarityandoverallresponsequality[56]. Evaluators
often use Likert scales to rate Readability, Coverage, Non-
Redundancy, and Quality [21, 18, 45]. In dialogue tasks, sev-
eralstudiesalsoassessaspectslikeUsefulnessandHumanness
[15]. Human-annotated data often forms the ground truth for
many benchmarks [10, 29, 44], and evaluation criteria may
include Redundancy, Diversity, Fairness, and Suitability [45].
Human judgments remain the gold standard, particularly for
evaluating factual correctness and nuanced generation quality
7

[29, 18], even though it is time-consuming, costly, and can
introduce subjective variance depending on evaluators and
criteria [29, 45].
4.1.6 Comparative Summary and Trends
The landscape of LLM evaluation is becoming increasingly
sophisticated. Traditional metrics like Accuracy and F1-score
still serve as foundational tools for classification tasks [35,
21, 28]. However, more advanced evaluations—focused on
factuality and grounding—are gaining prominence, especially
in response to challenges like hallucination [10, 30, 29, 44].
Human-annotated benchmarks and specialized metrics
help ensure robustness, while LLMs are now frequently inte-
grated into the evaluation loop—whether for scoring, verify-
ing, or generating intermediate outputs [10, 29, 32]. Though
promising, these LLM-based evaluations require careful vali-
dation against human judgments due to reliability concerns
[29, 55]. Human evaluation continues to play a vital role, par-
ticularly for qualitative and high-stakes tasks. The trend is
toward hybrid frameworks that combine multiple evaluation
strategies (e.g., automated metrics, LLM reasoning, human
oversights, etc.) to assess LLMs more holistically [20, 15, 51].
Thus, evaluating LLM fact-checking across diverse languages
and modalities, including multimodal or cross-lingual fact-
checking, is an emerging frontier and demands the adaptation
or creation of new evaluation techniques [48, 48, 32, 57, 22].
Figure 6 : Intrinsic vs. extrinsic hallucinations in LLM out-
puts: The source text provides verifiable ground truth about
Ebola and COVID-19 vaccines. The intrinsic hallucination
example contradicts the fact explicitly stated in the source,
where extrinsic hallucination introduces new information not
supported by the source.
4.2 Impact of Hallucinations on Fact-
Checking Reliability (RQ2)
Hallucinations in LLMs refer to outputs that seem fluent,
coherent, and linguistically correct but factually inaccurate,
nonsensical, unsupported, or entirely fabricated [58, 59]. Asthese outputs are presented with the same level of confidence
and linguistic fluency as factually accurate statements (see
Figure 6), it is often difficult for users to detect without
external verification [60, 58]. These hallucinations can oc-
cur from training corpora with contradictory, outdated, or
misleading information, biases, lack of grounding, and even
prompts [21, 3, 61].
An overview of all the papers referenced in this section is
presented in Table 3.
4.2.1 Hallucinations in LLMs
Nature and Types of Hallucinations. In the context of
fact-checking, hallucinations manifest as intrinsic or extrinsic
errors [62, 63, 64]. Intrinsic hallucinations occur when the
model’s generated output contradicts the source content. In
contrast, extrinsic hallucinations introduce information that
cannot be verified by any provided evidence, meaning output
that can neither be supported nor contradicted by the source,
often fabricating details not grounded in reality [50, 60, 63].
Figure 7 : Two types of hallucination: Red-highlighted text
shows hallucinated content, while blue-highlighted text re-
flects user instructions or context that conflict with the hal-
lucination.
Summarization models may intrinsically hallucinate by
stating a fact at odds with the source article, or an open-
ended question and answer (Q&A) model may extrinsically
hallucinate entirely new (false) information that it presents
as factual. Authors of [3, 60, 49] demonstrated that there
are generally two types (see example in Figure 7) of LLM
hallucinations: (i) faithfulness, when the generated text is
not faithful to the input context; and (ii) factuality, when the
generated text is not factually correct for world knowledge.
Causes of Hallucination. Hallucinations arise from funda-
mental misalignment in how LLMs are trained and used. The
core training objective of most LLMs is to predict the next
word in a sentence based on patterns learned from massive
text data, not to guarantee truthfulness [34, 65, 66]. This
means models are optimized to produce text that is coherent
and contextually appropriate rather than factually accurate
8

Table 3: An overview of studies on hallucinations covered in RQ2.
Topics Covered Years Authors Count
Nature and Types of Hallucinations 2021,
2022,
2023,
2025Huang et al. [62], Ji et al. [63], Li et al. [64], Jing et al. [50], Huang et al. [60],
Augenstein [3], Zhao et al. [49]7
Causes of Hallucination 2021,
2022,
2023,
2024,
2025Xie et al. [34], Zhou et al. [65], Wang et al. [66], Peng et al. [15], Ghosh et al.
[17], Augenstein et al. [3], Lin et al. [5], Bender et al. [6], Paullada et al. [7],
Ladhak et al. [8], Weidinger et al. [67], Wang et al. [68], Kasai et al. [19], Tran
et al. [30], Cheung et al. [46], Li et al. [69], Onoe et al. [70], Tang et al. [10], Yao
et al. [71]19
Implications for Reliability in Fact-
Checking2023,
2024,
2025Peng et al. [15], Si et al. [33], Li et al. [72], Zhao et al. [49], Xie et al. [34], Quelle
et al. [48], Hu et al. [12], DeVerna et al. [73], Singhal et al. [53], Zhao et al. [14],
Augenstein et al. [3], Wang et al. [68], Jing et al. [50]13
Fine-tuning and Instruction Tuning 2021,
2023,
2024,
2025Tang et al. [10], Setty et al. [11], Hu et al. [12], Zhang et al. [28], Zhao et al. [14],
Cheung et al. [46], Tran et al. [30], Qi et al. [31], Luo et al. [74], Leite et al. [55],
Jing et al. [50]11
RAG 2023,
2024,
2025Singhal et al. [53], Quelle et al. [48], Augenstein et al. [3], Si et al. [33], Peng et
al. [15], Sankararaman et al. [39], Khaliq et al. [22], Zhang et al. [51], Xie et al.
[34], Tran et al. [30], Wei et al. [75], Zhao et al. [14], Qi et al. [31], Li et al. [20],
Giarelis et al. [18], Ma et al. [16], Zhao et al. [49], Li et al. [72], Ghosh et al. [17],
Tang et al. [10]20
Adversarial Tuning 2025 Leippold et al. [76] 1
Automated Feedback Mechanisms and
Self-Correction2023,
2024,
2025Peng et al. [15], Ma et al. [16], Xie et al. [34], Tran et al. [30], Fadeeva et al. [77],
Ghosh et al. [17], Ge et al. [43], Zhao et al. [49]8
Hybrid Approaches and Multi-Agent Sys-
tems2023,
2024,
2025Zhang et al. [21], Zhao et al. [14], Ma et al. [16], Kupershtein et al. [13], Li et al.
[72], Giarelis et al. [18], Hu et al. [12], Ghosh et al. [17], Jing et al. [50]9
Multimodal Fact-Checking 2024 Cao et al. [36], Ge et al. [43], Sharma et al. [78], Qi et al. [31], Geng et al. [57],
Khaliq et al. [22]6
Multilingual Fact-Checking 2024 Quelle et al. [48] 1
Domain-Specific Fact-Checking 2023,
2024,
2025Zhang et al. [28], Tran et al. [30], Vladika et al. [38], Zhao et al. [49], Xiong et
al. [79], Jing et al. [50], Chatrath et al. [54], Khaliq et al. [22], Choi et al. [47],
Zhang et al. [21], Hu et al. [12], Leite et al. [55], Wang et al. [68], Qi et al. [31],
Choi et al. [80], Pisarevskaya et al. [41], Liu et al. [37]17
Enhancing Explainability and Trust 2023,
2024,
2025Ding et al. [56], Sankararaman et al. [39], Quelle et al. [48], Zhao et al. [14], Qi
et al. [31], Vladika et al. [38], Krishnamurthy et al. [42], Ghosh et al. [17], Leite
et al. [55], Giarelis et al. [18]10
Hierarchical Prompting and Multi-Step
Reasoning2023,
2024Zhang et al. [21], Khaliq et al. [22], Zhao et al. [14] 3
[15, 17]. Thus, LLMs’ tendency to hallucinate can be traced
to their optimization focus on linguistic fluency and coher-
ence, rather than factual precision, especially when faced
with queries outside their training distribution or when in-
ternal knowledge conflicts arise [3]. During training, they ab-
sorb countless statements-including inaccuracies and fictional
content-without an explicit mechanism to distinguish truth
from false [5, 6, 67, 7, 8]. As a result, an LLM may confidently
generate confident-sounding claims that align with linguistic
patterns in its memory but are not grounded in facts. Wang
et al. [44] demonstrate that "knowledge error" can occur
when the model produces hallucinated or inaccurate infor-
mation due to a lack of relevant knowledge or internalizing
false knowledge in the pre-training stage or the problematic
alignment process.
Additionally, since an LLM’s parametric knowledge base
is fixed after training, it can become outdated or insufficient,leading to guesses on topics it does not know [46, 19, 69,
70, 30]. When prompted “closed-book” (without access to
external data), a model faced with an unknown fact will often
fill the gap by generating a likely-sounding answer, and that
is a major source of hallucination [10]. In other cases, even if
grounding documents are provided, the model might improp-
erly blend or misattribute information from those sources,
causing intrinsic hallucinations [10]. Hallucinations may also
be an inherent adversarial vulnerability of LLMs; even non-
sense or out-of-distribution prompts can trigger the model
to produce a false but fluent response [71]. This suggests
that, beyond knowledge gaps, the model’s sensitivity to input
perturbations and over-reliance on spurious correlations in
training data can induce hallucinations.
9

Implications for Reliability in Fact-Checking. Halluci-
nations directly undermine the reliability of LLMs as fact-
checkers across domains, and in some cases, it can be fa-
tal when deployed for mission-critical tasks [15, 33, 72, 49].
Studies estimate that even advanced models like GPT-4 and
LLaMA-2 produce false factual statements in roughly 5–10%
of their responses on general knowledge queries [34]. When
LLMs are used in fact-checking systems, hallucinations can
lead to the misclassification of claims, thereby underestimat-
ing the accuracy and trustworthiness of fact-checking proce-
dures [48]. Moreover, hallucinations risk amplifying misin-
formation when LLM-generated content aligns with existing
false narratives, unknowingly facilitating their widespread
acceptance in the eyes of the public [12, 73, 53].
The complexity of detecting hallucinated content, partic-
ularly when intertwined with accurate information, compli-
cates the verification process and increases mental workload
for human fact-checkers, potentially impairing decision mak-
ing as evidenced by studies indicating reduced discernment
following LLM-assisted fact-checking [33, 73]. Additionally,
automated fact-checking systems leveraging LLMs for evi-
denceretrieval, claiminterpretation, orverdictgenerationare
highly vulnerable, as hallucinations occurring at any stage
can contaminate the entire fact-checking pipeline, ultimately
resulting in unreliable outcomes [14, 3, 68]. Recent work
emphasizes the need to quantify these phenomena systemat-
ically to enhance model reliability, proposing methodologies
specifically aimed at measuring and addressing hallucination
severity within faithfulness evaluations [50]. Singhal et al.
[53] state that in terms of fact-checking, simply assigning a
veracitylabelisinadequate; thepredictionmustbesupported
by evidence to ensure the system’s transparency and to bol-
ster public trust.
4.2.2 Mitigation Strategies for LLM Hallucinations
Fine-tuning and Instruction Tuning. Fine-tuning pre-
trainedLLMsondatasetstailoredtoaspecificdomainortask,
emphasizingfactuality, substantiallyenhancestheirreliability
[10]. Instruction tuning, a specialized variant of fine-tuning,
trains models to better adhere to explicit instructions aimed
at factual and verifiable responses. Specifically, domain-
specificadaptationinvolvesfine-tuningondatasetsrichinfac-
tual claims and evidence from specialized areas, such as news,
medical, political, religious, or legal domains, thus helping
models learn the intricacies of factual language and reasoning
pertinent to these fields [11]. Tang et al. [10] showed fine-
tuning transformer-based models, RoBERTa, DeBERTa, and
T5 models on synthetic and ANLI data boosts robustness
and achieves better results. Hu et al. [12] proposed that
leveraging LLMs’ ability to provide rationales could enhance
finetuned small language models (SLM), ultimately improv-
ing the performance of fake news detection. Zhang et al. [28]
demonstrated that in clinical claim evaluation, domain-tuneddiscriminative models such as BioBERT with 80.2% accuracy
outperformed both zero-shot and fine-tuned generative LLMs
like Llama3-70B, even after tuning. Zhao et al. [14] fine-
tuned Pretrained LMs utilizing various strategies such as
BERT-FC with dual loss, LIST5 with list-wise reasoning, and
T5 language model, RoBERTa using NLI, and MULTIVERS
with multitask learning.
In terms of instruction-following techniques for factuality,
several notable works, including Cheung et al. [46], explicitly
enhance instruction-following models by integrating external
knowledgespecificallyforfact-checking. Theyillustratedthat
targeted fine-tuning can significantly improve factual accu-
racy. Setty et al. [11] demonstrates that smaller, finely-tuned
models can sometimes surpass larger, more general models
in accuracy for fact-checking tasks; Tran et al. [30] intro-
duces two parallel self-training methods to updated LLMs
and boost factual reliability-Supervised Fine-Tuning (SFT)
using verified responses and Simple Preference Optimization
(SimPO) using fact-based ranking; Qi et al. [31] introduced
a two-stage instruction tuning process which adapts Instruct-
BLIP for OOC misinformation detection utilizing a two-stage
process: aligning it to the news domain using NewsCLIP-
pings data [74] and fine-tuning on GPT-4-generated inconsis-
tency explanations. Leite et al. [55] introduce a multi-stage
weak supervision approach utilizing instruction-tuned LLMs
prompted with 18 credibility signals to generate weak labels,
whicharethenaggregatedtopredictcontentveracity, thereby
minimizing hallucinations and enhancing transparency for
human fact-checkers. Jing et al. [50] fine-tuned two HHEM
DeBERTa NLI models on synthetic data and found that the
HHEM model with fine-tuning on synthetic data can out-
perform LLMs in domain-specific evaluation, given carefully
crafted training data.
Retrieval-Augmented Generation (RAG). RAG has
emerged as a prominent technique to ground LLM outputs
in external, verifiable knowledge sources, aiming to enhance
factual context, reduce hallucination, and decrease reliance
on potentially flawed internal knowledge [53, 48, 3, 33]. RAG
architectures generally include a retriever module that gath-
ers relevant information from external sources, such as web
documents or databases, and a generator module (the LLM)
that synthesizes this retrieved information and formulates
answers or assessments [15, 39].
Several studies have demonstrated the effectiveness and
variations of RAG. For instance, Singhal et al. [53] report
how integrating external knowledge and feedback loops can
significantly improve factuality; Peng et al. [15] introduced
LLM-AUGMENTER that enhances LLM responses by re-
trievingexternalknowledge, linkingrawevidencewithrelated
context, verifying outputs, and iteratively refining prompts
using automated feedback until the response is free from
hallucination and factually grounded. On the other had,
Quelle et al. [48] allowed LLMs to perform Google searches
to gather evidence related to the claim; Sankararaman et al.
10

[39] presented a method Provenance where a dual stage cross-
encoder framework, one to filter and weight context items
and another to another to assess factuality score, aggregating
these scores using weights for threshold-independent evalua-
tion; Khaliq et al. [22] applies multi-modal reasoning within
RAG proposing Chain of RAG (CoRAG) and Tree of RAG
(ToRAG) specifically for political contexts;
Reinforcement learning has been used in the RAG pro-
cess as well. For instance, Zhang et al. [51] investigated
reinforcement learning to optimize retrieval processes; [34]
propose an iterative retrieval and verification mechanism to
refine accuracy further; Singhal et at. [53] introduce a RAG-
based fact-checking system where the core pipeline retrieves
top-3 documents with FAISS, extracts evidence, then uses
the evidence in classification combining In-Context Learning
(ICL) capabilities of multiple LLMs, achieving a 22% gain
on Averitec dataset; Tran et al. [30] introduce Fact-Check-
Then-RAG which improves LLM factual accuracy by evaluat-
ing each fact with SAFE [75], retrieving relevant data using
ColBERT from MedRAG for failed checks, and re-generating
responses via a RAG-enhanced prompt; Zhao et al. [14] in-
troducedaretrieval-augmentedandmulti-agentfact-checking
systemwhichdynamicallyselectsreasoningtoolsandexternal
evidence to handle diverse multi-hop verification tasks.
To enhance out-of-context (OOC) detection, Qi et al. [31]
integrate Google’s Entity Detection API for visual grounding
and performs external verification through LLMs to verify
news captions against evidence retrieved from reverse image
searches; [20] introduced SELF-CHECKER which verifies in-
put by extracting simple claims, generating search queries
on various external knowledge sources e.g., Bing Search API,
retrieving evidence from knowledge sources e.g., Wikipedia,
Reddit messages, and predicting each claim’s veracity based
on its selected evidence sentences; Giarelis et al. [18] pro-
posed a unified LLM-KG framework for fact-checking which
retrieves relevant facts from Knowledge Graphs (KG) and
injects them into the LLM prompt for reponse generation;
Ma et al. [16] introduces Logical and Causal fact-checking
method (LoCal), a LLM-driven multi-agent framework that
breakdown complex claims, resolve them through specialized
reasoning, and validate consistency using logical and counter-
factual evaluators in an iterative process. Zhao et al. [49]
introduced MEDICO where the system retrieves evidence
from various sources search engine, knowledge base (KB),
knowledge graph (KG) and user files, then re-ranks and fuses
it by concatenation or Llama3-8B-based summarization; Li
et al. [72] introduced FactAgent that uses LLM’s internal
knowledge (i.e., Phrase, Language, Commonsense, Standing
tools) and another that integrate external knowledge tools
(i.e., URL and Search tools - SerpAPI); Ghosh et al. [17] in-
troducedLLMQuerythatevaluatesLLMs’logicalconsistency
in fact-checking by retrieving subgraphs from KGs using BFS
or ANN-based vector embedding methods, ensuring concise
and relevant context is fed to the model; and finally, Tanget al. [10] introduce an efficient method specifically tailored
for evaluating LLM-generated claims against grounding doc-
uments, forming a core component of comprehensive RAG
evaluation.
Adversarial Tuning. Adversarial training presents LLMs
with specifically designed examples intended to uncover hal-
lucinations or factual errors, thus training the models to
recognize and handle these challenging inputs accurately and
improve their robustness against generating misinformation.
The study by Leippold et al. introduces CLIMINATOR, an
acronym for CLImate Mediator for INformed Analysis and
Transparent Objective Reasoning, an AI-based tool utilizing
a Mediator and adversarial Advocate framework to automate
climate claim verification by simulating structured debates,
including climate denial perspectives, iteratively reconciling
diverse viewpoints to consistently converge towards scientific
consensus and thus improving accuracy and reliability [76].
Automated Feedback Mechanisms and Self-
Correction. Incorporating automated feedback loops
and enabling LLMs to self-critique and correct their
outputs are emerging as powerful strategies, exemplified
by several key approaches: Peng et al. [15] proposed an
automated feedback mechanism, LLM-AUGMENTER which
substantially reduces ChatGPT’s hallucinations without
sacrificing the fluency and informativeness of its responses
by iteratively revising LLM prompts to improve model
responses using feedback generated by utility functions,
e.g., the factuality score of a LLM-generated response. Ma
et al. [16] utilized two evaluating agents that iteratively
reject or accept solutions and trigger new decomposition or
reasoning rounds until consistency is achieved. Additionally,
iterative refinement processes, such as those demonstrated
by Xie et al. [34], continuously check and improve model
outputs through multiple verification cycles; Tran et al. [30]
introduced LEAF, a self-training loop that utilizes fact-check
scores as automated feedback.
Furthermore, the approach of uncertainty quantification
proposes leveraging token-level uncertainty measures to de-
tect potential hallucinations and trigger additional verifica-
tion steps, thereby enhancing overall reliability and accuracy
[77]. Ge et al. [43] also showed in their framework that LLMs
leverage object-detector and VQA results to automatically
mitigate hallucinated contents and fact-check proposed cap-
tions. In the work of Zhao et al. [49], it iteratively corrects
only hallucinated parts in generated content using Chain-of-
Thought (CoT) prompting, while enforcing minimal edits via
Levenshtein-based preservation scoring.
Hybrid Approaches and Multi-Agent Systems. Com-
bining multiple strategies or employing multi-agent architec-
tures, wherein different LLM agents handle specialized sub-
tasks within the fact-checking process, has emerged as a grow-
ingtrend,exemplifiedbyhierarchicalpromptingandplanning
methods such as the work by Zhang et al. [21] systematically
11

guide models through complex claim verification, and Zhao
et al. [14] utilize LLMs for structured planning and reasoning
tasks.
Additionally, multi-agent systems have gained attention,
as seen in studies like LoCal [16], employing multiple special-
ized LLM agents (decomposer + reasoner + two evaluators)
to address logical and causal dimensions of fact-checking,
and further explored by Kupershtein et al. [13] and Li et
al. [72], both of which investigate the capabilities of agent-
based frameworks specifically for fake news detection. Li et
al. [72] also introduced FactAgent that behaves in an agentic
manner, emulating human expert behavior utilizing several
tools (Phrase, Language, Commonsense, Standing, URL, and
Search tools) around a single LLM.
Moreover, the integration of structured knowledge sources
is further explored, advocating the combination of LLMs
with knowledge graphs to leverage structured factual infor-
mation [18]. Additionally, Hu et al. [12] designed a novel
approach, ARG and its distilled version ARG-D, that comple-
ments small and large LMs by selectively acquiring insights
fromLLM-generatedrationalesforSLMs; this combinationof
LLM+SLM has shown superiority over existing SLM/LLM-
only methods. By introducing a hybrid KG retrieval + LLM
generation approach and supervised fine-tuning, Ghosh et al.
[17] improved the logical consistency of LLMs on the complex
fact-checking task. Similarly, Jing et al. [50] combined rubric-
prompted LLM judges and an NLI cross-encoder (HHEM) in
the same framework.
4.2.3 Recent Innovations for Reducing Hallucina-
tions and Improving Factuality
Beyond the above core strategies, recent studies have fur-
ther tried to address hallucination issues and ensure LLMs
remain faithful to facts. These methods range from smarter
prompting techniques and multi-step reasoning procedures to
incorporating multiple modalities, to building self-checking
mechanisms and uncertainty estimates into LLM responses.
We highlight several promising directions below, along with
their advances in architecture, evaluation, and practical de-
ployment.
Multimodal Fact-Checking. Misinformation is increas-
ingly multimodal, combining text, images, and videos, thus
addressing hallucinations and ensuring factuality in LLMs
processing such data has become crucial. Efforts to integrate
visual and textual evidence include the study by Cao et al.
[36], which employs graph attention networks to consolidate
multimodal knowledge for verifying claims, and Ge et al. [43],
focusing specifically on accurate captioning of images utiliz-
ing visual fact-checking through object detection and VQA
models. Furthermore, Sharma et al. [78] evaluate the visual
grounding capabilities inherent in language models, while
[31] introduced a multimodal LLM-SNIFFER which analyzesboth the consistency of the image-text content and the claim-
evidence relevance using InstructBLIP and GPT-4V. Practi-
cal applications are explored in [57], which investigates real-
world deployment scenarios of multimodal LLMs, and finally,
Khaliq et al. [22] specifically applies retrieval-augmented
reasoning (ToRAG, CoRAG) to tackle multi-modal claims
within political context by extracting both textual and image
content, retrieving external information, and reasoning subse-
quent questions to be answered based on prior evidence and
achieved a weighted F1-score of 0.85, surpassing a baseline
reasoning method by 0.14 points.
Multilingual Fact-Checking. Misinformation transcends
language barriers, necessitating fact-checking capabilities
across multiple languages. LLMs offer significant potential in
this area, but ensuring factual consistency across languages
remains a considerable challenge. For instance, Quelle et
al. [48] found that fact-checking accuracy varied across
languages, with translated English prompts often achieving
higher accuracy than original non-English ones in terms
of multilingual fact-checking, despite claims involving non-
English sources. Additionally, due to skewed training data
and non-standardized fact-checks, LLMs perform better with
English-translated prompts, revealing language bias in multi-
lingual fact verification.
Domain-Specific Fact-Checking. Domain-specific fact-
checking is a crucial research area, as the nuances of verifying
factual claims can significantly differ across specialized fields
like medicine, politics, and climate science, necessitating tai-
lored LLMs and verification systems. While general-purpose
fine-tuned LLMs dominate broad tasks, specialized mod-
els fine-tuned on specific domains often outperform general-
purpose models in those areas [28]. Moreover, proprietary
models like Factcheck-GPT are often designed for general-
purpose use and not viable in domains like medicine due to
restrictions on private data use and lack of fine-tuning [30].
In medical contexts, dedicated efforts include [28] intro-
duced CliniFact and when evaluating LLMs against that,
BioBERT achieved 80.2% accuracy, outperforming generative
counterparts, such as Llama3-70B’s 53.6%, with statistical
significance (p < 0.001), developing explainable reasoning
systems [38], and detecting and correcting hallucinations by
integrating multi-source evidence [49]. Similarly, Tans et
al. [30] proposed LEAF, which tailors towards the medical
domain as it uses the MedRAG corpus.
Fact-checking is also investigated across various domains,
including travel, climate, news information, and claim match-
ing. For example, in the travel domain, Jing et al. [50]
used four industry datasets containing chats, reviews, and
property information for fact-checking. Political claim verifi-
cation is explored through studies assessing LLM reliability
[54] and multimodal retrieval-augmented reasoning systems
[22]. In climate science, specialized LLM-based tools ad-
dress the complexity of climate-related claims [47]. News
12

claims and general misinformation are broadly examined
through hierarchical prompting methods in [21], [12] inves-
tigated the potential of LLMs in fake news detection using
Chinese dataset Weibo21 [81] and GossipCop [82], [55] fo-
cuses on news-article veracity: FA-KES Syrian War corpus
[83] and EUvsDisinfo [84] pro-Kremlin corpus, and platforms
enabling customized fact-checking system development [68].
[31] trained their SNIFFER model on news domain using the
NewsCLIPpings [74] dataset. Additionally, claim-matching
methods utilizing LLMs for fact-checking [80, 47, 41] and
verification approaches for complex claims [37] further un-
derscore the depth and breadth of ongoing domain-specific
fact-checking research.
Enhancing Explainability and Trust. Beyond mere accu-
racy, the ability of LLM-based fact-checking systems to pro-
vide explanations and foster trust is increasingly recognized
as vital. Citations and provenance play a central role in this,
with [56] highlighting the importance of referencing sources
to build user confidence, while [39] emphasizes the need to
trace the origin of generated content. In the framework
proposed by Quelle et al. [48], agents explain their reasoning
and cite the relevant sources. Additionally, Zhao et al. [14]
enhanced fact-checking explainability by utilizing instruction-
based LLMs and specialized agents to generate structured
reasoning and justifications for sub-claims. Through CLIP-
based similarity, ROUGE scores, response ratio analysis, and
humanevaluation,SNIFFER[31]demonstrateshighaccuracy
and strong persuasive ability in explaining and detecting out-
of-context misinformation, and transparency in medical con-
texts [38]. Credibility assessment frameworks offer structured
approaches to evaluating trustworthiness [42]. Additionally,
Ghosh et al. [17] investigate the logical soundness of model
outputs, a key indicator of reliability. In another study, Leite
et al. [55] simplify fact-checking by guiding LLMs to predict
individual veracity signals, minimizing hallucinations, and
enablinghumanreviewerstoauditandcontroloutputs, which
enhances transparency. Giarelis et al. [18] stated that their
LLM-KG framework improves transparency through factual
context provided by Knowledge Graphs (KGs).
Hierarchical Prompting and Multi-Step Reasoning. A
key innovation involves prompting LLMs in a way that struc-
tures their reasoning process by decomposing complex fact-
checking tasks into smaller, verifiable steps. The underlying
ideaisthathumanfact-checkersoftenbreakdownaclaiminto
sub-claims or evidence checks. LLMs can be prompted to em-
ulate the process, reducing the risk of a single misstep leading
to a hallucinated conclusion. For instance, Hierarchical Step-
by-Step (HiSS) prompting directs the model to first separate
a claim into several subclaims, then verify each subclaim one
by one, before finalizing an overall verdict [21]. By forcing
the model to focus on one piece of information at a time (in a
chain-of-thought style), HiSS achieved superior fact verifica-
tion performance and reduced hallucination on news datasets,even outperforming fully-supervised baselines. This demon-
strates that prompting alone, if done cleverly, can induce
the model to reason more carefully and factually reducing
hallucination. Inastudy, Khaliqetal.[22]introduceChain-of-
RAG (sequential) and Tree-of-RAG (branch-and-eliminate hi-
erarchy), whichembodymulti-stepandhierarchicalreasoning.
Another example is the PACAR framework [14], which com-
bines LLM-driven planning with customized action reasoning
for claims. PACAR consists of multiple modules (a claim
decomposer, a planner, an executor, and a verifier) that allow
LLMs to plan a sequence of actions, such as performing a web
search or a numerical calculation, and then verify the claim
based on collected evidence. Using hierarchical prompting
and a multi-step approach, which includes specialized skills
like numerical reasoning and entity disambiguation, PACAR
significantly outperformed baseline fact-checkers across three
different domain datasets.
Overall, research to date illustrates the use of several
comprehensive, multi-dimensional approaches to mitigating
LLMs’ hallucinations in fact-checking, with notable advances
in RAG domain-specific fine-tuning and hybrid methodolo-
gies. Nonetheless, guaranteeing robust factual reliability
across varied, complex, and dynamically evolving informa-
tion scenarios continues to pose a major challenge. Future
research is expected to prioritize more sophisticated hybrid
systems, refined self-correction mechanisms, and more effec-
tive human-AI collaboration to strengthen the fact-checking
processes [85, 33].
4.3 Datasets for Training and Evaluating
Fact-Checking Systems (RQ3)
In this section, we discuss the wide range of datasets used in
the training, evaluation, and benchmarking of fact-checking
systems, particularly within RAG frameworks and hallucina-
tion mitigation strategies. These datasets support various
steps in each method, such as claim verification, evidence
retrieval, multi-hop reasoning, and hallucination detection.
The following accounts for the datasets and their uses in this
domain:
Benchmark Datasets for RAG-Based Fact Verifica-
tion.Core claim verification datasets such as FEVER [86],
FEVEROUS [9], and HOVER [87] are widely used to eval-
uate RAG pipelines, where the model is used to retrieve
relevant evidence from Wikipedia or structured sources and
thengenerateaverdict. Thesedatasetsprovidegold-standard
evidence, making them ideal for training retrievers and verify-
ing generation accuracy. LIAR [88] and RAWFC [89] further
allow the assessment of RAG-based models on political and
news-based claims with distinctive complexity and source
structures.
13

Domain-specific Datasets. In domain-specific applica-
tions, datasets such as SciFact [90], COVID-Fact [91], MedM-
CQA [92], BioASQ [93], and PubMedQA [94] are frequently
employedinRAGframeworksthatarealignedforthebiomed-
ical domain. These allow models to retrieve evidence from
the medical literature (e.g., via MedRAG) and cross-validate
LLM outputs. Given the importance of factual accuracy in
these domains, these datasets also serve as valuable bench-
marks for evaluating and refining hallucination reduction
techniques in sensitive contexts.
Multimodal Datasets. Multimodal datasets such as
MM-FEVER [95], Post-4V [96], NewsCLIPpings [74], and
MOCHEG [97] allow extended fact-checking to vision-
language settings. These are especially relevant for
evaluating multimodal RAG systems that allow the use
of visual and textual information to assess the veracity of
claims. Mismatches between image-caption pairs in these
datasets test the models’ ability to detect hallucinated or
manipulated content across modalities.
Hallucination Detection-specific datasets. To evaluate
hallucination detection and correction, specialized datasets
such as HaluEval [98], ReaLMistake [99], TruthfulQA [5],
and FoolMeTwice [100] provide annotated examples of hal-
lucinated outputs with detailed rationales. These are critical
for assessing token-level uncertainty, logical consistency, and
explanation alignment in LLMs.
Composite Datasets. Composite benchmarks like Fact-
Bench [101], OpenFactCheck [44], and FIRE [102] aggregate
multiple datasets (e.g., FacTool-QA, FELM-WK, Factcheck-
Bench) to provide diverse evaluation techniques for both re-
trieval and generation stages. These are specifically valuable
for end-to-end RAG evaluation, as they test factuality, consis-
tency, and explainability across all claim types and evidence
formats.
Synthetic and Multilingual Datasets. Synthetic and
weak supervision datasets such as ClaimMatch [103], LLM-
AGGREFACT [104], and CheckThat22 [105] allow for scal-
able training and evaluation in low-resource settings. These
datasets are often used to pre-train or fine-tune retrievers and
scorers within RAG systems, or to assess robustness against
adversarial claims and misinformation edits. Additionally,
multilingual datasets and some databases, like X-Fact [106],
Data Commons Multilingual, and FactStore, help build fact-
checking systems that work across different languages. They
test whether these systems can find the correct information
and give accurate answers, even in non-English settings. This
helps make fact-checking tools more significant in global use.
Figure 8 visualizes an overview of major dataset types and
their domains.
To provide a clear and comprehensive overview, Table 4
lists 72 key datasets used in fact-checking research. It shows
which datasets were used for fact checking by using RAG, for
hallucination reduction, and their type. Out of 72 datasets,
Figure 8 : Illustration of major dataset types and domains.
63 use RAG and 48 are used for hallucination reduction.
4.4 Prompt Design, Fine-Tuning, and
Domain-Specific Training (RQ4)
Prompt design strategies significantly impact the ability of
LLMs to perform fact-checking and mitigate hallucination.
The choice of strategy influences how the model processes
information, accesses knowledge, and generates responses,
directly affecting accuracy and performance. The sources
explore several key strategies, often contrasting methods that
rely solely on the model’s internal knowledge with those that
integrate external information retrieval. A visual summary
of approaches in prompt design, fine-tuning, and domain-
specific training is shown in Figure 9.
4.4.1 Basic Prompting Strategies
These strategies primarily rely on internal knowledge and
involvepresentingtheclaimortasktotheLLMswithminimal
or no external context beyond the prompt itself. Their effec-
tiveness is heavily reliant on the model’s pre-trained knowl-
edge, whichcanbeasignificantlimitationduetothepotential
for hallucination and outdated information [30, 12, 21].
Zero-shot prompting involves providing the LLMs with
only the task description and the input claim, without any
specific examples. This can include asking the model to
predict the veracity label of a claim directly [12, 72]. Frame-
works such as the FACT-AUDIT [45] use zero-shot infer-
ence to evaluate the fact-checking capacity of various LLMs.
They can yield the lowest average accuracy scores compared
to other methods, particularly when relying solely on the
model’s internal knowledge. While a simple self-consistency
and zero-shot prompt combination was found to be the most
effective overall strategy in a multilingual fact-checking study,
this effectiveness was strongly tied to the self-consistency
decoding strategy, not necessarily the zero-shot nature itself
[32]. On the other hand, frameworks such as PCAR [14],
which leverage explicit claim decomposition and a dynamic
14

Table 4: Datasets and their use in RAG and hallucinations (Halluc.) reduction.
Dataset RAG Halluc. Type Dataset RAG Halluc. Type
Doc2Dial ✓ ✓ Dialogue LIAR ✓ ✗ Political
Topical-Chat ✓ ✓ Dialogue RAWFC ✓ ✗ Political
QReCC ✓ ✓ Dialogue HALLU ✓ ✓ QA
Wizard of Wikipedia ✓ ✓ Dialogue BioASQ-Factoid ✓ ✓ QA
CMU-DoG ✓ ✓ Dialogue Q2 ✓ ✓ QA
FEVER ✓ ✗ Fact-Checking ASSERT ✓ ✓ QA
SciFact ✓ ✗ Fact-Checking SQuAD ✓ ✓ QA
COVID-Fact ✓ ✗ Fact-Checking CoCoGen ✗ ✓ QA
Factify ✓ ✗ Fact-Checking TriviaQA ✓ ✓ QA
AVERITEC ✓ ✗ Fact-Checking HotpotQA ✓ ✓ QA
TruthBench ✓ ✓ Fact-Checking Natural Questions ✓ ✓ QA
LLM-AGGREFACT ✓ ✗ LLM ELI5 ✓ ✓ QA
TruthfulQA ✓ ✓ LLM NarrativeQA ✓ ✓ QA
HaluEval ✓ ✓ LLM NewsQA ✓ ✓ QA
BioASQ-Y/N ✓ ✗ Medical DROP ✓ ✓ QA
MedMCQA ✓ ✗ Medical FactMix ✓ ✓ QA
USMLE ✓ ✗ Medical DuoRC ✓ ✓ QA
MMLU-Medical ✓ ✗ Medical QuAC ✓ ✓ QA
PubMedQA ✓ ✗ Medical FactScore Dataset ✓ ✓ QA
CliniFact ✗ ✗ Medical Data Commons ✓ ✗ Structured
MedQuAD ✓ ✓ Medical QA WikiBio ✓ ✓ Structured Data
MedInfo QA ✓ ✓ Medical QA ToTTo ✓ ✓ Structured Data
LiveQA-Medical ✓ ✓ Medical QA WebNLG ✓ ✓ Structured Data
MEDIQA-RQE ✓ ✓ Medical QA DART ✓ ✓ Structured Data
MeQSum ✓ ✓ Medical Summary LogicNLG ✓ ✓ Structured Data
FakeCovid ✗ ✗ Misinformation E2E NLG ✓ ✓ Structured Data
Multimodal FEVER ✓ ✗ Multimodal SAMSum ✓ ✓ Summarization
COCO ✗ ✗ Multimodal FIED ✓ ✓ Summarization
Objaverse ✗ ✗ Multimodal XSum ✓ ✓ Summarization
Visual Aptitude ✗ ✗ Multimodal CNN/Daily Mail ✓ ✓ Summarization
ADE20K ✗ ✗ Multimodal Gigaword ✓ ✓ Summarization
NewsCLIPpings ✓ ✗ Multimodal Multi-News ✓ ✓ Summarization
Polyjuice ✗ ✓ NLI Newsroom ✓ ✓ Summarization
FactualNLI ✓ ✓ NLI BigPatent ✓ ✓ Summarization
Multilingual FC ✗ ✗ Other WikiHow ✓ ✓ Summarization
PolitiFact ✓ ✗ Political Reddit TIFU ✓ ✓ Summarization
planning mechanism, achieve strong performance in zero-
shot settings and outperform other LLM-based approaches,
includingfew-shotandfine-tunedmethods. However, reliance
solely on internal knowledge for fact-checking is considered
unreliableandinsufficient,asLLMsarepronetohallucination
[30, 12, 21]. Zero-shot prompting without external access
means the model must rely on potentially inaccurate or out-
dated information stored during training [48]. The analysis of
rationales generated through zero-shot CoT indicates unreli-
ability for factuality analysis based on internal memorization
[35]. The most significant mitigation discussed is incorporat-
ing external knowledge retrieval [48, 30, 20, 15, 53, 21].
The Few-Shot Prompting or In-Context Learning (ICL)
method involves providing the model with a limited number
of examples of the task before presenting the claim to be
verified [14, 12]. It leverages the LLM’s ability to learn
from examples provided directly in the prompt ("in-context
learning")[20]. Few-shotdemonstrationsareusedinmethods
like HiSS(Hierarchical Step-by-Step) [21] and BiDeV [37] to
guide the LLM through multi-step processes. Few-shot-CoT
includes example pairs to guide the reasoning process [47, 72].
ICL has been shown to enhance the performance of
open-source MLLMs in detecting misinformation, sometimes
providing greater improvement than prompt ensembles [57].Combining ICL with RAG has been shown to improve accu-
racyinfactverification[53]. However, somesophisticatedfew-
shotICLmethodslikeStandardPrompting, VanillaCoT,and
ReAct were surpassed by the HiSS method in news claim ver-
ification, highlighting the importance of the specific method
prompted [48]. While ICL improves performance, open-
source MLLMs using ICL still significantly lag behind state-
of-the-art proprietary models like GPT-4V. The effectiveness
can vary between models and datasets [57]. Manual prompt
design for few-shot examples can be heuristic [20]. However,
providing examples does not guarantee overcoming reliance
oninternalknowledgeifexternalinformationisnotintegrated
[21]. Integrating ICL with RAG or structured, step-by-step
prompting frameworks is a key mitigation [53, 21].
Other methods, such as Chain-of-Thought (CoT) prompt-
ing, instruct the LLM to output a sequence of intermediate
reasoning steps before arriving at the final answer or veracity
label [21, 12]. Zero-shot CoT prompts often include eliciting
sentenceslike"Letusthinkstepbystep"[12,32,47,16]. This
encourages explicit reasoning. Variants of CoT prompting
include English CoT (EN-CoT) [32], which focuses on mono-
lingual reasoning, and CoTVP (Chain of Thought Veracity
Prediction) [22], which evaluates the truthfulness of the rea-
soningsteps. However, studiesshowthattechniqueslikeCoT,
designedtoimprovereasoning, donotnecessarilyimprovethe
15

Figure 9 : Breakdown of approaches in prompt design, fine-tuning, and domain-specific training.
fact-checking abilities of LLMs. They can even have minimal
or negative effects on success rates [32]. Models prompted
with CoT may tend to align with the input text rather than
verifying its factualness, especially for complex paragraphs,
when relying solely on pre-trained knowledge [20]. Vanilla
CoT suffers substantially from issues of fact hallucination
and omission of necessary thoughts in the reasoning process
[21]. Using the LLM for factuality analysis based on its
internal memorization, even with zero-shot CoT, indicates
unreliability, likely caused by hallucination [12].
The internal mechanism of LLMs to integrate rationales
from diverse perspectives via CoT can be ineffective for fake
newsdetection[12]. IntegratingCoTwithexternalknowledge
retrieval, as in Search-Augmented CoT or ReAct [21], is a
key approach to mitigate hallucination and thought omission
[21]. Additionally, frameworks that explicitly guide the de-
composition and reasoning process, such as FactAgent, are
seen as superior to CoT, which primarily acts as a prompting
technique [72].When relying solely on internal knowledge, LLMs
prompted with zero-shot, few-shot, or vanilla CoT struggle
with fact-checking complex claims and exhibit hallucination
[30, 12, 20, 21]. However, accuracy can be low, and
improvements in reasoning via CoT alone do not guarantee
better fact-checking performance [32]. Techniques forcing
binary "true" or "false" judgments also do not enhance
overall accuracy [73].
4.4.2 Prompting Strategies with Integrated Exter-
nal Retrieval
Several strategies combine prompting with the ability to ac-
cess and utilize external information sources (e.g., search en-
gines or curated databases) to ground responses and improve
factual accuracy. This is a critical aspect for robust fact-
checking and hallucination reduction [48, 21, 30, 53].
A CoT variant that interleaves reasoning traces with
task-specific actions, like querying Google Search or the
16

Wikipedia API, allowing the LLM agent to decide whether
to search or continue reasoning based on environmental ob-
servations [48, 21, 20, 22]. For instance, by accessing external
knowledge, ReAct effectively mitigates hallucination failures
compared to vanilla CoT and justifies its reasoning with
retrieved citations, enhancing verifiability and explainability.
Its performance is highly sensitive to the quality and rele-
vance of search results, and relying solely on internal knowl-
edge when external search fails remains a key limitation [21].
Combining ReAct’s action capabilities with more structured
decomposition and step-by-step verification methods, such as
HiSS or SELF-CHECKER, can address thought omissions
and improve overall performance [21, 20].
Search-Augmented CoT augments vanilla CoT by using
the original claim as a search query to retrieve background
information, which the LLM incorporates into its thought
chain. This approach improves over vanilla CoT by lever-
aging external knowledge, but can fall short of methods like
Standard Prompting or HiSS, which indicates that querying
solely with the claim may yield insufficiently detailed results.
To mitigate this, more sophisticated query generation strate-
gies and integration methods are needed [21]. HiSS is a
few-shot method that prompts the LLM to perform claim
verification in fine-grained steps by decomposing claims into
subclaims and verifying each step-by-step, raising questions
and optionally using web search when confidence is low. It
significantly surpasses few-shot ICL counterparts like Stan-
dard Prompting, Vanilla CoT, and ReAct in average F1-score,
offering superior explainability through enhanced coverage
andreadabilitywhilesubstantiallyreducinghallucinationand
thought omission [21]. However, HiSS still struggles with in-
tegrating updated information from mixed sources and incurs
high computational costs due to multiple LLM calls, with its
performance remaining sensitive to prompt design [20].
Frameworks like SELF-CHECKER [20], BiDeV [37],
RAGAR [22], and PACAR [14] integrate prompting and
retrieval-augmented generation (RAG) by decomposing
fact-checking into subtasks (e.g., claim detection, retrieval,
sentence selection, verdict prediction) and using prompts
often with few-shot examples to generate search queries,
select evidence, and perform step-by-step verification while
explicitly incorporating retrieved documents. Incorporating
external knowledge via RAG significantly boosts accuracy
over internal-only approaches: SELF-CHECKER [20], BiDeV
[37], RAGAR [22], and PACAR [14] all show substantial
gains, with specialized modules like the Claim Atomizer and
Fact-Check-Then-RAG further enhancing predictive power
and explanation quality.
RAG-based methods can be hampered by overwhelming
context windows [48], outdated or variable search results [20,
22], high computational costs, prompt sensitivity, and man-
ual prompt design [20], and they may still miss fine-grained
details even when grounded [30]. To mitigate these issues,
current strategies include using IR functions (e.g., BM25) todistill relevant content [30, 48], using multi-agent [76, 37], ex-
plicit decomposition and filtering [14, 42, 37], feedback loops
[51], and refined RAG processes [30, 42]. Reliance solely on
anLLM’sinternalknowledge, whetherviazero-shot, few-shot,
or vanilla CoT prompting, is unreliable for fact-checking and
prone to substantial hallucination [21, 30, 12], as zero-shot
CoT and vanilla CoT often succumb to memorization pitfalls
[12, 16]. Mitigating hallucination primarily involves incorpo-
ratingexternalknowledgethroughReAct, Search-Augmented
CoT, HiSS, and other RAG frameworks [21, 30, 53].
In conclusion, while basic prompting strategies like zero-
shot, few-shot, and vanilla CoT offer foundational ways to
interact with LLMs for fact-checking, their accuracy and reli-
abilityareseverelylimitedbyrelianceonpotentiallyflawedin-
ternal knowledge [30, 12, 21]. The most effective approaches,
as highlighted by the sources, involve prompting strategies
that explicitly integrate external knowledge retrieval through
frameworks like ReAct, HiSS, SELF-CHECKER, BiDeV, RA-
GAR, and PACAR [21, 48, 20, 53, 37]. These methods use
prompting to guide the LLM through processes involving
external data access, significantly improving accuracy and
mitigating hallucination by grounding the model’s responses
in evidence [21, 20, 53].
Prompting is also used to generate explanations, although
the utility and reliability of these explanations can vary
[21, 33]. Limitations across strategies include sensitivity to
prompt wording, computational cost, and the need for more
robust and automated design methods [20].
4.4.3 Fine-tuning Architectures for Optimizing
Fact-checking Performance
Fact-checking performance is primarily optimized through
two primary approaches: (1) the development and fine-tuning
ofspecificmodelarchitectures, and(2)thedesignofadvanced
prompting strategies for LLMs. These methods are often
combined within complex fact-checking pipelines.
Fine-tuning smaller transformer models on synthetic
data. This approach fine-tunes pre-trained transformer
models on structured synthetic data, often combined with
standard entailment datasets, to teach them nuanced fact-
checking against grounding documents [10]. The strategy
centers on generating challenging training instances that help
models verify atomic facts across multiple sentences, with
models like MiniCheck-FT5, RBTA, and DBTA outperform-
ing larger LLMs such as GPT-4 in specific benchmarks like
LLM-AGGREFACT [10]. Notably, MiniCheck-FT5 achieves
a 4.3% improvement over AlignScore using a significantly
smaller dataset. Difficulty aggregating evidence and reason-
ing over multiple facts, the method mitigates these through
targeted synthetic data and simple aggregation strategies like
majority voting [11].
Fine-tuning small language models (SLMs) for task-
specific performance. This method focuses on fine-tuning
17

SLMs for task-specific applications like fake news detection.
The strategy includes training SLMs directly on the target
datasetandleveragingLLM-generatedrationalesviaarchitec-
tures like the ARG network, with a distilled version (ARG-D)
for efficiency [12]. Fine-tuned BERT models have outper-
formed GPT-3.5 in fake news detection, and ARG/ARG-D
exceedbaselinemethodscombiningbothSLMandLLMcapa-
bilities [12]. LLMs’ difficulty in fully utilizing their reasoning
for domain-specific tasks and their inability to fully replace
SLMs in these contexts. Mitigations involve using LLMs as
rationale providers to guide SLMs and exploring advanced
prompting and model combinations for better results [12].
Instruction Fine-tuned LLMs as Verifiers within a
Pipeline Framework. This framework integrates an in-
struction fine-tuned LLM as a verifier within a fact-checking
pipeline, where the LLM evaluates claims based on contex-
tually retrieved evidence. The process includes claim atom-
ization using Mistral-7B, relevance-based evidence retrieval
and re-ranking, and final inference by the LLM, producing
interpretable credibility reports [42]. The "Yours Truly"
framework achieves a 94% F1-score, significantly outperform-
ing other systems, with the Claim Atomizer alone boosting
performance from 64% to 93% [42].
4.4.4 Domain-specific Training for Model Adapta-
tion in Specialized Knowledge Areas
Domain-specific adaptation and fine-tuning are highlighted
as crucial strategies for enhancing the performance and re-
liability of models, particularly LLMs, in automated fact-
checking within specialized knowledge areas. The need for
such domain specificity stems from the observation that LLM
factual accuracy and vulnerability to hallucinations can vary
significantly across different domains. While general-purpose
models may perform well in broad areas, models fine-tuned
or adapted for specific domains, such as medicine or science,
often demonstrate superior performance in those particular
fields. Sources discuss the application of fact-checking tech-
niques across various specialized domains, including medi-
cal/biomedical, political, scientific, law, general biographic,
and news [14, 30, 28, 31, 68].
Several approaches are explored for achieving domain
adaptation in fact-checking systems. One involves fine-tuning
smaller transformer models on target datasets specific to a
domain or task within fact-checking, such as claim detection
or veracity prediction, which has shown surprising efficacy
and can outperform larger, general LLMs in specific con-
texts [48, 11]. Another approach directly involves fine-tuning
LLMs themselves or using techniques like instruction-tuning
on domain-relevant data or tasks [30, 48, 31, 46, 42, 40].
Frameworks like OpenFactCheck are proposed to allow users
to customize fact-checkers for specific requirements, including
domain specialization [68].4.4.5 Comparative Summary and Trends
From straightforward prompting to more complex fine-tuning
and domain-specific training, the methods for using LLMs in
fact-checking are changing. To inform general-purpose mod-
els such as GPT-3.5 and GPT-4, initial research focused on
zero-shot and few-shot in-context learning. However, in fact-
checking benchmarks, a notable trend indicates that smaller,
optimized models can surprisingly beat larger LLMs, provid-
ing a more economical and effective solution [11].
Prompt engineering has also progressed beyond simple
Chain-of-Thought (CoT). The rise of organized hierarchical
prompting techniques, such as HiSS, which break down com-
plicated claims into verifiable steps to reduce hallucinations
and improve reasoning, is a definite trend [21, 38]. Fur-
thermore, domain-specific adaptation is becoming more and
more important. Using specialized datasets and knowledge
bases, models are being particularly trained or prompted for
difficult domains such as news, climate science, and medicine
to increase accuracy when general knowledge is inadequate
[49, 76].
Integrating external domain-specific knowledge through
methods like RAG or incorporating Knowledge Graphs is also
a significant strategy to augment models with the necessary
specialized information, sometimes in conjunction with fine-
tuning or adaptation [36, 30, 15, 46, 18]. For example, the
LEAF approach enhances medical question answering by in-
tegrating fact-checking results into RAG or using fact-checks
for supervised fine-tuning [30]. Evaluation datasets tailored
to specific domains or types of claims, such as SciFact for
scientific claims, Climate-FEVER for climate science, EX-
PERTQA spanning multiple fields, or medical QA datasets,
are utilized to benchmark the effectiveness of these domain-
adapted systems [10, 14, 30, 28, 76].
While interventions like fine-tuning and claim normal-
ization improve robustness, performance can still degrade
when evaluated significantly out-of-domain across different
topics or platforms [40]. The sources collectively underscore
that effective fact-checking in specialized areas often requires
models or frameworks specifically tailored to the domain’s
knowledge and nuances, moving beyond one-size-fits-all gen-
eral approaches [30, 28, 68, 40].
4.5 Integration of RAG in Fact-Checking
(RQ5)
RAG is a hybrid framework designed to enhance LLMs by
integrating the retrieval of external knowledge into their gen-
eration process [56, 3]. This approach is pivotal for grounding
LLM outputs in external evidence, thereby mitigating com-
mon issues such as hallucination, the generation of factually
incorrect or nonsensical information, and reliance on poten-
tiallyoutdatedinternalknowledge[3,15,33]. Unlikemethods
solely based on fine-tuning the model’s internal parameters,
RAG leverages external data sources, such as web search
18

results or curated knowledge bases, to inform the LLM’s
responses during inference [56, 15]. A significant advantage
of RAG is its ability to produce responses that are not only
more factually accurate and reliable but also offer improved
transparency and credibility by providing explicit citations
to the external sources used [48, 56]. This capability allows
models to access and utilize current information, overcoming
the limitations of knowledge cutoffs inherent in their training
data [22]. An overview of the workflow of a basic RAG-based
system is presented in Figure 10.
Figure 10 : Workflow of a RAG system for factual question
answering.
In the domain of fact-checking, RAG systems play a cru-
cial role in automating and improving the verification pro-
cess [48]. LLM agents empowered with RAG can phrase
search queries based on claims, retrieve relevant external
contextual data, and utilize this information to assess the
veracity of statements [48, 56]. This integration of exter-
nal knowledge through RAG leads to enhanced accuracy in
fact-checking, enabling the extraction of relevant evidence to
support veracity predictions [48, 22, 53]. Approaches like
Fact-Check-Then-RAG utilize the outcomes of fact-checking
to refine the retrieval process itself and ensure that retrieved
information specifically enhances factual accuracy [30]. RAG
also supports more complex scenarios, including multimodal
fact-checking, where it is used to extract both textual and
image content and retrieve external information for reason-
ing [57, 22]. Furthermore, RAG-based systems can provide
reasoned explanations for their verdicts, improving the inter-
pretability of the fact-checking process [48, 14]. Variations
likeFFRRleveragefine-grainedfeedbackfromtheLLMtoop-
timize retrieval policy based on how well documents support
factual claims [51]. While some approaches explore reducing
relianceonexternalretrievalbyleveragingtheLLM’sinternal
knowledge, RAG is generally considered essential for effective
fake news detection [57, 72].
Despite its advantages, implementing RAG, particularly
in specialized domains, presents several challenges and lim-
itations [3, 77]. A significant hurdle is the requirement for
efficient and accurate retrieval of relevant evidence at scale,which can be a computational bottleneck [3]. The effec-
tiveness of RAG is inherently dependent on the assumption
that pertinent information is readily available and accessible
within the external knowledge sources used, such as search
engines [21]. This assumption may not hold for all infor-
mation, especially in specialized or low-resource domains
where relevant knowledge might be obscure, non-digitized,
or exist in formats not easily indexed [21]. Retrieving and
processing large volumes of external data can also overwhelm
the LLM’s context window, necessitating sophisticated tech-
niques for selecting and consolidating the most critical infor-
mation [48, 15]. Standard RAG-based methods can inadver-
tently introduce noise or irrelevant information, potentially
hindering the LLMs’ performance rather than improving it
[30, 51]. In specialized fields like healthcare, general RAG
models may struggle due to a lack of the nuanced under-
standing required for accurate fact-checking, highlighting the
need for tailored or potentially fine-tuned approaches or in-
tegrating domain-specific knowledge bases [30, 18]. Complex
fact-checking tasks, such as interpreting conflicting evidence,
handling claims with insufficient external information, or per-
forming causal reasoning with fragmented information across
documents, remain challenging for LLMs even with RAG
[53, 42, 37]. Furthermore, the dynamic nature of information
requires constant updates to external sources, and relying
solely on search results can lead to diluted credibility if mis-
information is widely reported [72].
For inefficient and computationally expensive evidence
retrieval, researchers have developed lightweight frameworks
like Provenance [39], which use compact and open-source NLI
models for verification instead of LLMs. FIRE [34], a frame-
work that is designed for time and cost efficiency through
an iterative process. Reinforcement retrieval models further
showthataproperlytrainedretrieverdoesnotaddsignificant
overhead during inference, as the costly feedback loop is only
part of that training phase [51]. Due to the unavailability of
evidence in specialized or low-resource domains, frameworks
like LEAF [30], designed for the medical domain, use a spe-
cializedcorpussuchasMedRAG,whichincludesPubMedand
textbooks, instead of relying solely on Google Search. The
challenge of low-resource language fact checking is mitigated
by translating the claims into high high-resource language,
such as English, in some literature [48].
To avoid overwhelming the LLM context window and
managing large volumes of retrieved framework like Prove-
nance [39] use Relevancy Score combined with a TopK/-
TopP selection module to filter down to the most critical
pieces of evidence before sending them to the verifier, LLM-
AUGMENTER features a "Knowledge Consolidator" that
prunes irrelevant information and synthesizes the remaining
evidenceintoconcisechains[51], ReinforcementRetrievalalso
suggests that using a smaller, optimal number of documents
(e.g., the top 3-4) is often more effective than using a larger
set that may introduce noise [51]. For mitigating noise and
19

irrelevant information introduced by standard RAG methods,
Reinforcement Retrieval addresses this directly using rein-
forcement learning to train the retriever; feedback from the
LLM verifier acts as a reward signal, teaching the retriever to
select more useful and factually relevant documents [51]. The
LEAF framework uses a "Fact-Check-Then-RAG" approach,
where an initial fact-check on the LLM’s output is used to
guide a more targeted and accurate retrieval process [30].
Furthermore, for handling complex reasoning, conflicting
evidence, and insufficient information, the PACAR frame-
work employs a dynamic planner that can deploy tailored
agents for tasks like numerical reasoning and entity disam-
biguation [14]. The LoCal framework is a multi-agent system
specifically designed to handle the complexities of logical
and causal fact-checking. For scenarios with conflicting or
insufficientevidence[52], theAVERITECdatasetwascreated
witha"ConflictingEvidence"labelandastructuredquestion-
answer format that can represent evidential disagreements,
providing a basis for training models to better navigate such
ambiguity [52].
4.5.1 Comparative Summary and Trends
The incorporation of RAG, which has been a key strategy in
LLM-based fact-checking, has established a trend away from
dependence on static, internal knowledge. A complex claim is
typically divided into verifiable sub-claims in the basic RAG
pipeline[21,49], followedbythecollectionofrelevantexternal
evidence, its combination with a verifier model, and a final
decision [46, 107]. It is evident that in recent years, this lin-
ear pipeline has changed into more intelligent, dynamic, and
effective systems. One significant advancement is the shift to
agent-basedanditerativeframeworks. SystemslikeFIRE[34]
and LLM-AUGMENTER [15] can handle complex, multi-hop
claims more successfully because they employ repeated cycles
of retrieval and verification rather than a single retrieval
step. Multi-agent systems like LoCal and PACAR [14], which
employ planning modules to dynamically choose which tools
or reasoning processes to employ, further develop this. The
growing complexity of the retrieval and verification procedure
itself is another significant development. Frameworks such as
Provenance [39] employ lightweight models to score and filter
evidence for relevance rather than just providing an LLM’s
raw search results. As investigated in the Reinforcement Re-
trieval framework, there is also a shift toward optimizing the
retriever for the downstream fact-checking job by employing
strategies like reinforcement learning to transmit feedback
from the verifier back to the retriever. An overview of the
limitations and mitigation strategies is summarized in Figure
11.5 Discussion
Our review explored the rapid adoption of LLMs in the com-
plex task of automated fact checking. By examining a wide
rangeofcurrentstudies, wehavemappedouthowwemeasure
their success, the persistent problem of models generating
false information (hallucinations), and the essential role and
limitations of the datasets they rely on. We also looked into
methods for improving LLMs, from prompt engineering and
fine-tuning to the increasingly vital use of RAG. The research
landscape reveals a field that is buzzing with innovation and
showing great promise. It simultaneously highlights signifi-
cantandcomplexhurdlesthatremain. IfLLMsaretobecome
truly reliable tools in the global effort against misinformation,
these challenges demand ongoing and rigorous investigation
[3, 48].
Evaluation metrics. In the evaluation metrics domain
(RQ1) for LLM-based fact checking, we can see the clear
transition in classification scores towards more sophisticated,
holistic and context-aware frameworks. The rise of rigor-
ous benchmarks such as LLM-AGGREFACT [104] and the
AVERITEC dataset [52], along with new centralized tools
such as OpenFactCheck [44], marks a major step towards
creating shared ways to measure how accurate LLMs are.
However, even with this progress, many important issues re-
main unresolved. Furthermore, making models tough enough
to handle misinformation that adapts, often as an adversary,
including claims with subtle edits or those that change over
time, remainsamajorhurdle[40]. Thelackofstrongmethods
for clear explanation and adaptation to new types of risks
makes it harder for everyday users to judge whether the
answers from LLMs are true. This shortfall also weakens the
usefulness of these models in fast-changing real-life situations.
Two key challenges are now drawing more attention:
teaching models how to check their work and catch their
own mistakes [29], and finding reliable ways to measure how
closely their responses match the source material [50]. De-
spite advances in automated and LLM-driven assessments,
Human Evaluation remains essential for nuanced aspects like
explanation quality and contextual appropriateness [73], al-
though it is resource intensive. In general, while progress is
evident, significant lacunae persist. There is a pressing need
for standardized metrics that robustly assess the quality of
LLM-generated explanations [31], the resilience of the model
against evolving misinformation [40], and the logical integrity
of reasoning pathways [17]. These developments underscore
an urgent and ongoing need for metrics that can holistically
evaluate not only the veracity of claims, but also the prove-
nance of supporting evidence [39] and the logical integrity of
the LLM’s reasoning process.
Hallucination in LLMs. The tendency of LLMs to hal-
lucinate (RQ2), that is, to generate outputs that are lin-
guistically fluent and coherent yet factually misleading or
20

Figure 11 : Limitations in RAG-based fact-checking and corresponding mitigation Strategies
entirely unsubstantiated, remains a significantbarrier to their
trustworthy deployment in sensitive, high-stakes applications
such as fact-checking [3, 21]. RAG has become a foundational
mitigationstrategy, designedtogroundLLMresponsesinver-
ifiable external knowledge and reduce the models’ reliance on
their internal, and potentially flawed or outdated, parametric
knowledge [48, 15, 107, 46]. New efforts to boost how truthful
language models are include adding systems that automati-
cally give feedback, helping refine answers over multiple tries
[15]. Tools like Self-Checker are also being built—these are
smart correction modules that let the model review and fix
its own mistakes [20]. On top of that, researchers are testing
ways to bring together evidence from several sources, with
models like MEDICO showing how this kind of fusion can
work [49]. This persistence of hallucinations is partly due
to the inherent "black box" nature of current LLMs, the
difficulty in comprehensively modeling nuanced world knowl-
edge, and the escalating sophistication of adversarial attacks
designed to exploit model vulnerabilities [50].
Datasets for Fact-Checking. The datasets (RQ3) utilized
for the training, fine-tuning, and rigorous evaluation of fact-
checking LLMs are pivotal to their ultimate performance
and generalizability. While foundational datasets such as
FEVER [86] have been instrumental in catalyzing early re-
search, the field is increasingly recognizing the necessity for
more specialized, challenging, and contextually rich bench-
marks. Illustrative examples include CliniFact, which is
tailored for claims within the domain of clinical research
[28], AVERITEC, with its distinct emphasis on real-world
claims that necessitate web-based evidence retrieval [52], andBINGCHECK, specifically designed for assessing the factual-
ity of LLM-generated text [20]. The intrinsic characteristics
of these datasets—including their composition, scale, annota-
tion quality, topical diversity, and potential inherent biases
(e.g., political, cultural, or temporal) profoundly influence
model performance, the ability of models to generalize to
unseen domains or claim structures, and, consequently, the
perceived effectiveness of various fact-checking methodologies
[85, 54]. The development and meticulous curation of robust
multilingual datasets [32] also represent a critical frontier for
advancing the global applicability and equity of LLM-based
fact-checking technologies.
Optimization Strategies and Domain-specific Train-
ing.Prompt engineering, fine-tuning strategies, and domain-
specific training (RQ4) have been shown to significantly mod-
ulate the efficacy of LLMs in complex fact-checking tasks.
Advanced prompting techniques, such as the hierarchical
step-by-step verification methods [21] or structured reason-
ing frameworks like PACAR [14] and BiDeV [37], frequently
demonstrate superior outcomes when compared to simpler,
more direct prompting approaches. The application of zero-
shot and few-shot learning paradigms is also being actively
explored for a range of related tasks, including claim match-
ing [41, 47, 80], indicating a strong potential for efficient
adaptation of LLMs with limited task-specific data. Fur-
thermore, the surprising efficacy of smaller, specifically fine-
tuned transformer models in certain fact-checking contexts
[11] compellingly suggests that model scale is not the sole, nor
alwaystheprimary, determinantofperformance. Thisfinding
challenges the prevailing "bigger is better" narrative in LLM
21

development and highlights the potential for more resource-
efficient, specialized models to achieve competitive or even
superior performance in targeted fact-checking applications,
particularly when data and computational budgets are con-
strained. Domain-specific adaptations, which may involve
the utilization of LLM-predicted credibility signals [55] or the
developmentofhighlyspecializedsystemsforcriticaldomains
such as medicine [38], are proving essential for achieving the
nuanced understanding and reliable fact verification required
in these contexts.
The Integration of RAG. Incorporating LLMs with RAG
(RQ5) is increasingly recognized as a central and indispens-
able strategy for enhancing the factuality of LLM outputs.
This is achieved by providing models with dynamic access
to external, often real-time, knowledge sources, thereby aug-
menting their inherent capabilities [48, 15, 107, 46]. Frame-
works such as FIRE [34] are being developed to optimize it-
erative retrieval and verification processes, aiming for greater
efficiency and accuracy. Other lines of research explore the
application of reinforcement learning techniques to refine and
optimize retrieval strategies [21]. Nevertheless, significant
challenges persist within the RAG pipeline. These include
the efficient and precise retrieval of truly relevant evidence
from vast, heterogeneous, and often noisy information spaces;
the effective fusion of information derived from multiple, po-
tentially contradictory, sources [14]. Extending RAG to effec-
tively handle multi-modal inputs [22, 57] and optimizing the
timing and contextual relevance of information recommended
by RAG-empowered agents [61] remain active and critical
areas of ongoing investigation.
6 Open issues and challenges
While LLMs have advanced fact-checking capabilities, several
core challenges remain. These include mismatches between
fluency and truth, domain limitations, and weak reasoning
integration.
Despite the advancement in fact-checking using LLMs,
one common challenge remains the gap between model model-
generated responses’ linguistic quality and factual accuracy.
Today’s criteria of evaluation are inclined to appreciate lan-
guage or text overlap with references, which can overlook
basicfacterrors. Itguidestofeedbacksthatlookquestionable
but sound good and inaccurate. These responses can receive
arbitrarily high scores, which makes the system seem more
accurate than it is [48, 58, 59].
Existing models often execute much better on small syn-
thetic datasets, but often fail to generalize well to real-world
circumstances. The reason lies in the limited datasets with
low complexity, variation in topics, and multilingual texts.
Consequently, the models do not handle the variability and
nuance of real-world data across languages and topics, which
implies the need for more realistic and broad training and testcorpora [60, 80]. However, RAG techniques have a stronger
evidence-based foundation in LLM, but retrieval is imperfect.
Fact-correctness is often weakened by the quality of the re-
trieved information, noisy or irrelevant documents, and when
there is insufficient context available. Similarly, advanced
prompting techniques, such as chain-of-thought reasoning or
multiagent collaborations, still must be precisely fine-tuned,
but they also remain vulnerable to cascading errors in output
generation [14, 77].
One of the areas with great potential that is still underex-
ploredliesintheintegrationofLLMswithsymbolicreasoning
orstructuredlogic-basedsystems. Thesesystemscanimprove
interpretabilityandfactresilienceinfact-checkpipelines. But
work in this area is still in an early stage, and significant
effort is needed to design scalable, proper architectures that
can balance the respective strengths of neural and symbolic
methods [76, 61]. Table 5 provides an organized summary of
the existing issues and challenges discussed in this section.
7 Critical analysis of future research
agendas
LLMs show great promise in automating fact checking, but
their use also highlights a range of ongoing problems that
still need attention. In the future, research must take a
thoughtful and future-focused approach. If these models are
to become trustworthy, accurate, and ethically sound tools in
fact-checking, the gaps in today’s research, for example, the
unreliability of retrieved evidence, the difficulty in verifying
complex claims requiring multistep reasoning, and the chal-
lenge of mitigating factual hallucinations, need to be tackled
head-on. What follows is a breakdown of key areas where
further study could really make a difference, each pointing to
where progress is most needed.
Evaluation Framework Advancement. A fundamental
imperative lies in transcending current evaluation metrics,
which often inadequately capture the nuanced performance
characteristics of LLMs in fact check tasks [3, 68]. Future re-
searchmustprioritizethedevelopmentofsophisticated,multi-
dimensional frameworks. This includes establishing stan-
dardized metrics for explainability and interpretability that
demonstrably correlate with human cognitive trust and facil-
itate diagnostic understanding of model failures [31, 38, 56].
Furthermore, the creation of dynamic evaluation suites (test-
ing systems that can actively change and adapt over time,
rather than relying on fixed, unchanging datasets) to rigor-
ously test resilience against evolving misinformation tactics
and sophisticated adversarial attacks is paramount [40], mov-
ing beyond static benchmarks. Currently, the development
of precise metrics for fine-grained faithfulness and verifiable
provenance tracking [50, 39] is crucial to ensure that LLM
outputs are not merely plausible but demonstrably grounded
in credible evidence.
22

Table 5: Identified issues in LLM-based fact-checking and their implications.
Issue/Challenge Observed Behavior Implication Why does it matter? Ref.
Mismatch Between Out-
put Quality and Factual
AccuracyModels write very fluent
and convincing text.High-quality language does not
mean the facts are correct.1. Current evaluation methods favor “sounding
good” over “being accurate.”
2. Models may get high scores for responses
that look right but contain factual errors.[48]
[58]
[59]
Limited Relevance
Across Domains and
LanguagesModels perform well on
simple, synthetic, or
English-only datasets.Struggle with real-world, complex,
or multilingual data.1. Fact-checking needs to work across many
subjects, topics, and languages.
2. Limited data variety in training/testing
leads to poor generalization.[60]
[80]
Challenges in Retrieval
and Prompting Mecha-
nismsUse of RAG brings in exter-
nal evidence1. Retrieval is often imperfect
(brings irrelevant or noisy info).
2. Advanced prompting (chain-of-
thought, multi-agent) still leads to
error cascades.1. Fact-checking relies on reliable evidence re-
trieval and reasoning chains.
2. Weaknesses here mean incorrect or unsup-
ported conclusions.[14]
[77]
Lack of Integration with
Symbolic or Structured
ReasoningCurrent LLMs rely mostly
on pattern recognition, not
logic.1. Little integration with log-
ic/symbolic systems.
2. Models can not follow strict,
logical reasoning pipelines.1. Symbolic reasoning would make fact-
checking more robust and explainable.
2. Lack of it = less trustworthy and harder-to-
monitor systems.[76]
[61]
Table 6: Identified gaps from the proposed research questions (RQs), and future research paths in fact-checking with LLMs.
RQs Research Gaps Potential Research Paths
RQ1 Evaluation and Benchmark-
ing Challenges1. Develop a new evaluation matrix that not only limits itself to overlapping or semantic scores but also
incorporates the factual correctness and the reasoning capabilities of LLMs, along with real-world dynamics
and practicality.
2. Establish robust metrics and methodologies for evaluating the human-computer interaction aspects of
fact-checking systems in terms of clarity, actionability, and persuasiveness of explanations.
3. Create evaluation frameworks that can assess a fact-checking system’s ability to handle temporally
sensitive claims, outdated evidence, and the "freshness" of information.
RQ2 Trust and Reliability 1. Develop automatic detection and correction methods for hallucinated LLM outputs, including uncer-
tainty quantification.
2. Optimize retrieval strategies with reinforcement learning and iterative verification to improve efficiency
and accuracy.
3. Investigate effective formats of AI-generated fact checks to enhance human trust through transparent
explanations and evidence.
RQ3 Limited Realistic, Complex,
Multilingual Datasets1. Develop more realistic, complex, dynamic, domain-specific, multilingual fact-checking datasets with
high-quality evidence for evaluation and fine-tuning.
2. Develop innovative and efficient data creation and annotation methodologies.
3. Develop more robust weak supervision, semi-supervised, or active learning techniques to reduce reliance
on fully manual annotation.
4. Design systems and protocols for continuous data collection and dataset updates to reflect the real-time
nature of information and misinformation.
RQ4 Prompt Sensitivity and
Adaptation Challenges1. Design and evaluate prompting methodologies that explicitly enforce and enable verification of evidence-
grounded and faithful reasoning.
2. Develop adaptive, model-aware prompting frameworks that automatically generate and refine prompts
and in-context examples to ensure robustness against variations.
3. Develop continual learning strategies for fine-tuned and domain-specific models to allow them to adapt
to new information; investigate meta-learning or adaptive techniques for prompt optimization.
4. Develop prompting and fine-tuning methodologies that explicitly optimize for generating controllable,
verifiable, and evidence-grounded reasoning and explanations.
RQ5 Efficient Explainable
Retrieval1. DesignsystemswhereLLMscaniterativelyrefinequeries, exploremultipleinformationangles, orretrieve
evidence for decomposed sub-claims to build a more comprehensive evidence base.
2. Advanced agent-based RAG systems where an LLM (or multiple specialized LLM agents) can plan a
sequence of reasoning and retrieval steps.
3. Design efficient RAG architectures that minimize computational overhead through optimized context
chunking, selective retrieval, and reusable memory.
Factual Hallucination Mitigation. The mitigation of
LLM-generated hallucinations requires a strategic shift from
reactive correction to proactive prevention mechanisms em-
bedded within LLM architectures and training paradigms
[49, 14]. Currently, optimizing RAG systems to effectively
navigatecomplexandnoisyinformationenvironments[21,37]
and enabling dynamic and reliable knowledge updates withinLLMs [3] are critical to ensure accurate and consistent factual
grounding. Without these, RAG systems risk amplifying,
rather than rectifying, inaccuracies.
Logical Consistency, Reasoning, and Calibrated
Trust. The ultimate efficacy of LLMs in fact checking is
critically dependent upon their capacity for robust logical
reasoning and their ability to engender warranted user trust.
23

Future endeavors must explore formal verification methods
(mathematically and logically ensuring that the model’s
reasoning process is sound and its conclusions are consistent)
to enhance the logical consistency of LLM outputs [17] and
significantly deepen their reasoning capabilities for complex,
multihop, and inferential claims [49]. The degree to which
users believe the fact-checks offered by LLMs must also be
thoroughly investigated. The goal of this study should be to
appropriately "calibrate" or modify that trust to a suitable
degree. The purpose is to encourage people to critically
evaluate the information provided rather than relying too
much on these automatic checks. [73, 56]
Multimodality and Multilinguality. Given that misinfor-
mation transcends unimodal (ie, English) texts, even though
some studies have tried to incorporate multimodal and mul-
tilingual fact checking [43, 48, 32, 57], there is still a need
for a significant expansion of LLM fact-checking capabilities.
The development of robust multimodal systems capable of
verifying claims that integrate textual, visual, and auditory
information, and detecting sophisticated cross-modal manip-
ulations is a key frontier [43, 31]. Equally vital are dedicated
efforts to develop and evaluate information and effective and
equitable fact-checking in a diverse spectrum of languages,
withparticularattentiontoresource-scarcelinguisticcontexts
[32, 53].
Table 6 provides an overview of the identified gaps and
future research agendas.
8 Conclusion
TheinclusionofLLMsinautomatedfact-checkingischanging
the field. These models are reshaping how we process and
verify theoverwhelming amountof informationweface online.
Our work lays out the current research in this space, drawing
attention to five critical areas: how LLMs are evaluated,
the problem of hallucinations (where models produce false
information), the importance of data sources, various ways of
improving performance, and the growing use of RAG. The
findings highlight the fact that while this field is moving
quickly and holds a lot of promise, there are serious issues
that still need careful attention if we want these systems to
work reliably.
One of the major achievements of this study is that it
pulls together a broad snapshot of what is going on right
now. It highlights a tricky balance: on the one hand, LLMs
have the potential to improve the speed and quality of fact
check. However, these systems are still limited, and fixing
those limits is going to require a lot of effort from researchers.
From the studies, one thing becomes clear: we need better
tools to judge these systems. Since most existing tools only
focus on whether a fact is right or wrong, there is now a
growing need for tools that also look at how clearly the
model explains things, whether it sticks to the logic, howwell it shows where its answers came from, and how tough
it is against tricky questions or misleading setups. From our
review, it is evident that it is still difficult to ensure that
LLMs always stick to real facts, especially when the situation
is new or tricky. This improvement will require fundamental
changes in the way these models are built and trained.
Although this review offers a comprehensive synthesis of
the existing literature, its scope is inherently constrained by
the rapid velocity of technological innovation within the LLM
domain. Consequently, emerging preprint findings, advance-
ments in proprietary models, and developing best practices
may not be fully encapsulated. Furthermore, the predomi-
nant analytical lens has been applied to textual fact-checking,
with multimodal and multilingual dimensions acknowledged
as critical but less exhaustively explored frontiers. Looking
ahead, future research must prioritize the development of
more sophisticated, robust, and universally standardized eval-
uation benchmarks. Such benchmarks are urgently needed to
assess not only the factual accuracy but also the logical co-
herence and soundness of LLM reasoning, the quality, utility,
andpersuasivenessofgeneratedexplanations, andthemodels’
resilience to a wide array of adversarial attacks and evolving
misinformation tactics.
The review presents key insights for developers, policy-
makers, and all stakeholders who rely on online information.
It underscores the need to address significant shortcomings in
the development and evaluation of LLMs to enable their ef-
fective use as reliable fact-checking tools. That means paying
closeattentiontohowfairandbalancedthedataare, learning
how to handle different languages and types of media better,
andsettingupstrongrulesandprotectionstomakesurethese
systems are used ethically and do not cause harm.
Declarations
Conflict of Interests: On behalf of all authors, the corre-
sponding author states that there is no conflict of interest.
Funding: No external funding is available for this research.
Data Availability Statement: Not Applicable.
Ethics Approval and Consent to Participate . Not Ap-
plicable.
Informed Consents: Not Applicable.
Author Contributions: Conceptualization and Methodol-
ogy:Subhey Sadi Rahman, Md. Adnanul Islam, Md. Mah-
bub Alam, Musarrat Zeba, Mohaimenul Azam Khan Raiaan;
Resources and Literature Review: Subhey Sadi Rahman,
Md. Adnanul Islam, Md. Mahbub Alam, Musarrat Zeba;
Writing – Original Draft Preparation: Subhey Sadi Rah-
man, Md. Adnanul Islam, Md. Mahbub Alam, Musarrat
Zeba, Md. Abdur Rahman, Sadia Sultana Chowa, Mo-
haimenul Azam Khan Raiaan;
Validation: Sami Azam, Sadia Sultana Chowa, Mo-
haimenul Azam Khan Raiaan, Md. Abdur Rahman;
Formal Analysis: Md. Abdur Rahman, Sami Azam;
24

Writing – Reviewing and Finalization: Sami Azam, Mo-
haimenul Azam Khan Raiaan;
Project Supervision: Sami Azam, Mohaimenul Azam
Khan Raiaan;
Project Administration: Sami Azam, Mohaimenul Azam
Khan Raiaan.
References
[1] D. Kampelopoulos, A. Tsanousa, S. Vrochidis, and
I. Kompatsiaris, “A review of llms and their applica-
tions in the architecture, engineering and construction
industry,” Artificial Intelligence Review , vol. 58, no. 8,
p. 250, 2025.
[2] X.Wang,H.Jiang,Y.Yu,J.Yu,Y.Lin,P.Yi,Y.Wang,
Y. Qiao, L. Li, and F.-Y. Wang, “Building intelligence
identification system via large language model water-
marking: a survey and beyond,” Artificial Intelligence
Review, vol. 58, no. 8, p. 249, 2025.
[3] I. Augenstein, T. Baldwin, M. Cha, T. Chakraborty,
G. L. Ciampaglia, D. Corney, R. DiResta, E. Ferrara,
S. Hale, A. Halevy et al., “Factuality challenges in the
era of large language models and opportunities for fact-
checking,” Nature Machine Intelligence , vol. 6, no. 8,
pp. 852–863, 2024.
[4] T. Huang, “Content moderation by llm: From accuracy
to legitimacy,” Artificial Intelligence Review , vol. 58,
no. 10, pp. 1–32, 2025.
[5] S. Lin, J. Hilton, and O. Evans, “Truthfulqa:
Measuringhowmodelsmimichumanfalsehoods,” arXiv
preprint arXiv:2109.07958 , 2021. [Online]. Available:
https://arxiv.org/abs/2109.07958
[6] E. M. Bender, T. Gebru, A. McMillan-Major, and
S. Shmitchell, “On the dangers of stochastic parrots:
Can language models be too big?” in Proceedings of
the 2021 ACM conference on fairness, accountability,
and transparency , 2021, pp. 610–623.
[7] A. Paullada, I. D. Raji, E. M. Bender, E. Denton, and
A. Hanna, “Data and its (dis) contents: A survey of
dataset development and use in machine learning re-
search,”Patterns, vol. 2, no. 11, 2021.
[8] F. Ladhak, E. Durmus, M. Suzgun, T. Zhang, D. Juraf-
sky, K. McKeown, and T. B. Hashimoto, “When do pre-
training biases propagate to downstream tasks? a case
studyintextsummarization,” in Proceedings of the 17th
Conference of the European Chapter of the Association
for Computational Linguistics , 2023, pp. 3206–3219.[9] R. Aly, S. Papay, C. Christodoulopoulos, and
I. Augenstein, “Feverous: Fact extraction and
verification over unstructured and structured
information,” in Proceedings of the 2021 Confer-
ence on Empirical Methods in Natural Language Pro-
cessing. Association for Computational Linguis-
tics, 2021, pp. 6118–6129. [Online]. Available:
https://aclanthology.org/2021.emnlp-main.495/
[10] L. Tang, P. Laban, and G. Durrett, “Minicheck: Ef-
ficient fact-checking of llms on grounding documents,”
arXiv preprint arXiv:2404.10774 , 2024.
[11] V. Setty, “Surprising efficacy of fine-tuned transform-
ers for fact-checking over larger language models,” in
Proceedings of the 47th International ACM SIGIR Con-
ference on Research and Development in Information
Retrieval , 2024, pp. 2842–2846.
[12] B. Hu, Q. Sheng, J. Cao, Y. Shi, Y. Li, D. Wang,
and P. Qi, “Bad actor, good advisor: Exploring the
role of large language models in fake news detection,”
inProceedings of the AAAI Conference on Artificial
Intelligence , vol. 38, no. 20, 2024, pp. 22105–22113.
[13] L. Kupershtein, O. Zalepa, V. Sorokolit, and
S. Prokopenko, “Ai-agent-based system for fact-
checking support using large language models,” in
CEUR Workshop Proceedings , 2025, pp. 321–331.
[14] X. Zhao, L. Wang, Z. Wang, H. Cheng, R. Zhang, and
K.-F. Wong, “Pacar: Automated fact-checking with
planning and customized action reasoning using large
language models,” in Proceedings of the 2024 Joint In-
ternational Conference on Computational Linguistics,
Language Resources and Evaluation (LREC-COLING
2024), 2024, pp. 12564–12573.
[15] B. Peng, M. Galley, P. He, H. Cheng, Y. Xie, Y. Hu,
Q. Huang, L. Liden, Z. Yu, W. Chen et al., “Check
yourfactsandtryagain: Improvinglargelanguagemod-
els with external knowledge and automated feedback,”
arXiv preprint arXiv:2302.12813 , 2023.
[16] J. Ma, L. Hu, R. Li, and W. Fu, “Local: Logical and
causal fact-checking with llm-based multi-agents,” in
Proceedings of the ACM on Web Conference 2025 , 2025,
pp. 1614–1625.
[17] B. Ghosh, S. Hasan, N. A. Arafat, and A. Khan,
“Logical consistency of large language models in fact-
checking,” arXiv preprint arXiv:2412.16100 , 2024.
[18] N. Giarelis, C. Mastrokostas, and N. Karacapilidis,
“A unified llm-kg framework to assist fact-checking
in public deliberation,” in Proceedings of the First
Workshop on Language-Driven Deliberation Technology
(DELITE)@ LREC-COLING 2024 , 2024, pp. 13–19.
25

[19] J. Kasai, K. Sakaguchi, R. Le Bras, A. Asai, X. Yu,
D. Radev, N. A. Smith, Y. Choi, K. Inui et al., “Re-
altime qa: What’s the answer right now?” Advances
in neural information processing systems , vol. 36, pp.
49025–49043, 2023.
[20] M. Li, B. Peng, M. Galley, J. Gao, and
Z. Zhang, “Self-checker: Plug-and-play modules
for fact-checking with large language models,”
inFindings of the Association for Computational Lin-
guistics: NAACL 2024, Mexico City, Mexico, June 16-
21, 2024 , K. Duh, H. Gómez-Adorno, and
S. Bethard, Eds. Association for Computational
Linguistics, 2024, pp. 163–181. [Online]. Available:
https://doi.org/10.18653/v1/2024.findings-naacl.12
[21] X. Zhang and W. Gao, “Towards llm-based fact verifi-
cation on news claims with a hierarchical step-by-step
prompting method,” arXiv preprint arXiv:2310.00305 ,
2023.
[22] M. A. Khaliq, P. Chang, M. Ma, B. Pflugfelder,
and F. Miletić, “Ragar, your falsehood radar: Rag-
augmented reasoning for political fact-checking using
multimodal large language models,” arXiv preprint
arXiv:2404.12065 , 2024.
[23] I. Vykopal, M. Pikuliak, S. Ostermann, and
M. Šimko, “Generative large language models
in automated fact-checking: A survey,” arXiv
preprint arXiv:2407.02351v2 , 2024. [Online]. Available:
https://arxiv.org/abs/2407.02351
[24] A. Dmonte, R. Oruche, M. Zampieri, P. Calyam,
and I. Augenstein, “Claim verification in the age of
large language models: A survey,” arXiv preprint
arXiv:2408.14317v2 , 2025. [Online]. Available: https:
//arxiv.org/abs/2408.14317
[25] A. Wang and Others, “Factuality of large language
models: A survey,” arXiv preprint arXiv:2402.02420v3 ,
2024. [Online]. Available: https://arxiv.org/abs/2402.
02420
[26] B. A. Kitchenham, P. Brereton, D. Budgen, M. Turner,
J. Bailey, and S. G. Linkman, “Systematic literature
reviewsinsoftwareengineering-Asystematicliterature
review,” Inf. Softw. Technol. , vol. 51, no. 1, pp. 7–15,
2009. [Online]. Available: https://doi.org/10.1016/j.
infsof.2008.09.009
[27] M. E. Conway, “How do committees invent,” Datama-
tion, vol. 14, no. 4, pp. 28–31, 1968.
[28] B.Zhang, A.Bornet, A.Yazdani, P.Khlebnikov, M.Mi-
lutinovic, H. Rouhizadeh, P. Amini, and D. Teodoro, “A
dataset for evaluating clinical research claims in large
language models,” Scientific Data , vol. 12, no. 1, p. 86,
2025.[29] R. Kamoi, S. S. S. Das, R. Lou, J. J. Ahn, Y. Zhao,
X. Lu, N. Zhang, Y. Zhang, R. H. Zhang, S. R. Vum-
manthala et al., “Evaluating llms at detecting errors in
llm responses,” arXiv preprint arXiv:2404.03602 , 2024.
[30] H. Tran, J. Wang, Y. Ting, W. Huang, and T. Chen,
“Leaf: Learning and evaluation augmented by fact-
checking to improve factualness in large language mod-
els,”arXiv preprint arXiv:2410.23526 , 2024.
[31] P. Qi, Z. Yan, W. Hsu, and M. L. Lee, “Sniffer: Mul-
timodal large language model for explainable out-of-
contextmisinformationdetection,” in Proceedings of the
IEEE/CVF conference on computer vision and pattern
recognition , 2024, pp. 13052–13062.
[32] A. Singhal, T. Law, C. Kassner, A. Gupta, E. Duan,
A. Damle, and R. L. Li, “Multilingual fact-checking
using llms,” in Proceedings of the Third Workshop on
NLP for Positive Impact , 2024, pp. 13–31.
[33] C. Si, N. Goyal, T. Wu, C. Zhao, S. Feng,
H. Daumé Iii, and J. Boyd-Graber, “Large language
models help humans verify truthfulness – except
when they are convincingly wrong,” in Proceedings
of the 2024 Conference of the North American Chapter
of the Association for Computational Linguistics: Hu-
man Language Technologies (Volume 1: Long Papers) ,
K. Duh, H. Gomez, and S. Bethard, Eds. Mexico City,
Mexico: Association for Computational Linguistics,
Jun. 2024, pp. 1459–1474. [Online]. Available:
https://aclanthology.org/2024.naacl-long.81/
[34] Z. Xie, R. Xing, Y. Wang, J. Geng, H. Iqbal, D. Sah-
nan, I. Gurevych, and P. Nakov, “Fire: Fact-checking
with iterative retrieval and verification,” arXiv preprint
arXiv:2411.00784 , 2024.
[35] N. Lee, B. Z. Li, S. Wang, W.-t. Yih, H. Ma, and
M. Khabsa, “Language models as fact checkers?” arXiv
preprint arXiv:2006.04102 , 2020.
[36] H. Cao, L. Wei, W. Zhou, and S. Hu, “Multi-source
knowledge enhanced graph attention networks for mul-
timodal fact verification,” in 2024 IEEE International
Conference on Multimedia and Expo (ICME) . IEEE,
2024, pp. 1–6.
[37] Y. Liu, H. Sun, W. Guo, X. Xiao, C. Mao, Z. Yu, and
R. Yan, “Bidev: Bilateral defusing verification for com-
plex claim fact-checking,” in Proceedings of the AAAI
Conference on Artificial Intelligence ,vol.39,no.1,2025,
pp. 541–549.
[38] J. Vladika, I. Hacajová, and F. Matthes, “Step-by-step
fact verification system for medical claims with explain-
able reasoning,” arXiv preprint arXiv:2502.14765 , 2025.
26

[39] H.Sankararaman, M.N.Yasin, T.Sorensen, A.DiBari,
and A. Stolcke, “Provenance: A light-weight fact-
checker for retrieval augmented llm generation output,”
arXiv preprint arXiv:2411.01022 , 2024.
[40] J. Magomere, E. La Malfa, M. Tonneau, A. Kazemi,
and S. Hale, “When claims evolve: Evaluating
and enhancing the robustness of embedding mod-
els against misinformation edits,” arXiv preprint
arXiv:2503.03417 , 2025.
[41] D. Pisarevskaya and A. Zubiaga, “Zero-shot and few-
shot learning with instruction-following llms for claim
matching in automated fact-checking,” arXiv preprint
arXiv:2501.10860 , 2025.
[42] V. Krishnamurthy and V. Balaji, “Yours truly: A cred-
ibility framework for effortless llm-powered fact check-
ing,”IEEE Access , 2024.
[43] Y. Ge, X. Zeng, J. S. Huffman, T.-Y. Lin, M.-Y.
Liu, and Y. Cui, “Visual fact checker: enabling high-
fidelity detailed caption generation,” in Proceedings of
the IEEE/CVF Conference on Computer Vision and
Pattern Recognition , 2024, pp. 14033–14042.
[44] Y. Wang, M. Wang, H. Iqbal, G. N. Georgiev, J. Geng,
I. Gurevych, and P. Nakov, “Openfactcheck: Building,
benchmarking customized fact-checking systems and
evaluating the factuality of claims and llms,” in Pro-
ceedings of the 31st International Conference on Com-
putational Linguistics . Association for Computational
Linguistics, 2025, pp. 11399–11421. [Online]. Available:
https://aclanthology.org/2025.coling-main.755/
[45] H. Lin, Y. Deng, Y. Gu, W. Zhang, J. Ma, S.-K. Ng,
and T.-S. Chua, “Fact-audit: An adaptive multi-agent
frameworkfordynamicfact-checkingevaluationoflarge
language models,” arXiv preprint arXiv:2502.17924 ,
2025.
[46] T.-H. Cheung and K.-M. Lam, “Factllama: Optimiz-
ing instruction-following language models with exter-
nal knowledge for automated fact-checking,” in 2023
Asia Pacific Signal and Information Processing Associ-
ation Annual Summit and Conference (APSIPA ASC) .
IEEE, 2023, pp. 846–853.
[47] E. C. Choi and E. Ferrara, “Automated claim matching
with large language models: empowering fact-checkers
inthefightagainstmisinformation,” in Companion Pro-
ceedings of the ACM Web Conference 2024 , 2024, pp.
1441–1449.
[48] D. Quelle and A. Bovet, “The perils and promises of
fact-checking with large language models,” Frontiers in
Artificial Intelligence , vol. 7, p. 1341697, 2024.[49] X. Zhao, J. Yu, Z. Liu, J. Wang, D. Li, Y. Chen, B. Hu,
and M. Zhang, “Medico: Towards hallucination detec-
tion and correction with multi-source evidence fusion,”
arXiv preprint arXiv:2410.10408 , 2024.
[50] X. Jing, S. Billa, and D. Godbout, “On a scale from 1 to
5: Quantifying hallucination in faithfulness evaluation,”
arXiv preprint arXiv:2410.12222 , 2024.
[51] X. Zhang and W. Gao, “Reinforcement retrieval leverag-
ing fine-grained feedback for fact checking news claims
with black-box llm,” arXiv preprint arXiv:2404.17283 ,
2024.
[52] M. Schlichtkrull, Z. Guo, and A. Vlachos, “Averitec: A
dataset for real-world claim verification with evidence
from the web,” Advances in Neural Information Pro-
cessing Systems , vol. 36, pp. 65128–65167, 2023.
[53] R. Singhal, P. Patwa, P. Patwa, A. Chadha, and
A. Das, “Evidence-backed fact checking using rag and
few-shot in-context learning with llms,” arXiv preprint
arXiv:2408.12060 , 2024.
[54] V. Chatrath, M. Lotif, and S. Raza, “Fact or fiction?
can llms be reliable annotators for political truths?”
arXiv preprint arXiv:2411.05775 , 2024.
[55] J. A. Leite, O. Razuvayevskaya, K. Bontcheva,
and C. Scarton, “Detecting misinformation with llm-
predicted credibility signals and weak supervision.”
2023.
[56] Y. Ding, M. Facciani, E. Joyce, A. Poudel, S. Bhat-
tacharya, B. Veeramani, S. Aguinaga, and T. Weninger,
“Citations and trust in llm generated responses,” in
Proceedings of the AAAI Conference on Artificial In-
telligence , vol. 39, no. 22, 2025, pp. 23787–23795.
[57] J. Geng, Y. Kementchedjhieva, P. Nakov, and
I. Gurevych, “Multimodal large language models
to support real-world fact-checking,” arXiv preprint
arXiv:2403.03627 , 2024.
[58] Y. Bang, S. Cahyawijaya, N. Lee, W. Dai, D. Su,
B. Wilie, H. Lovenia, Z. Ji, T. Yu, W. Chung et al.,
“A multitask, multilingual, multimodal evaluation of
chatgpt on reasoning, hallucination, and interactivity,”
arXiv preprint arXiv:2302.04023 , 2023.
[59] N. M. Guerreiro, D. M. Alves, J. Waldendorf, B. Had-
dow, A. Birch, P. Colombo, and A. F. Martins, “Hal-
lucinations in large multilingual translation models,”
Transactions of the Association for Computational Lin-
guistics, vol. 11, pp. 1500–1517, 2023.
[60] L. Huang, W. Yu, W. Ma, W. Zhong, Z. Feng, H. Wang,
Q. Chen, W. Peng, X. Feng, B. Qin et al., “A survey on
27

hallucination in large language models: Principles, tax-
onomy, challenges, and open questions,” ACM Transac-
tions on Information Systems , vol. 43, no. 2, pp. 1–55,
2025.
[61] T. Sakurai, S. Shiramatsu, and R. Kinoshita, “Llm-
based agent for recommending information related to
web discussions at appropriate timing,” in 2024 IEEE
International Conference on Agents (ICA) . IEEE,
2024, pp. 120–123.
[62] Y. Huang, X. Feng, X. Feng, and B. Qin, “The factual
inconsistency problem in abstractive text summariza-
tion: A survey,” arXiv preprint arXiv:2104.14839 , 2021.
[63] Z. Ji, N. Lee, R. Frieske, T. Yu, D. Su, Y. Xu, E. Ishii,
Y. J. Bang, A. Madotto, and P. Fung, “Survey of hal-
lucination in natural language generation,” ACM com-
puting surveys , vol. 55, no. 12, pp. 1–38, 2023.
[64] W. Li, W. Wu, M. Chen, J. Liu, X. Xiao, and H. Wu,
“Faithfulness in natural language generation: A sys-
tematic survey of analysis, evaluation and optimization
methods,” arXiv preprint arXiv:2203.05227 , 2022.
[65] C. Zhou, P. Liu, P. Xu, S. Iyer, J. Sun, Y. Mao, X. Ma,
A. Efrat, P. Yu, L. Yu et al., “Lima: Less is more for
alignment,” Advances in Neural Information Processing
Systems, vol. 36, pp. 55006–55021, 2023.
[66] Y. Wang, W. Zhong, L. Li, F. Mi, X. Zeng, W. Huang,
L. Shang, X. Jiang, and Q. Liu, “Aligning large lan-
guage models with human: A survey,” arXiv preprint
arXiv:2307.12966 , 2023.
[67] L. Weidinger, J. Mellor, M. Rauh, C. Griffin,
J. Uesato, P.-S. Huang, M. Cheng, M. Glaese,
B. Balle, A. Kasirzadeh et al., “Ethical and social
risks of harm from language models,” arXiv preprint
arXiv:2112.04359 , 2021.
[68] Y. Wang, M. Wang, H. Iqbal, G. N. Georgiev, J. Geng,
I. Gurevych, and P. Nakov, “Openfactcheck: Building,
benchmarking customized fact-checking systems and
evaluating the factuality of claims and llms,” in Pro-
ceedings of the 31st International Conference on Com-
putational Linguistics , 2025, pp. 11399–11421.
[69] D. Li, A. S. Rawat, M. Zaheer, X. Wang, M. Lukasik,
A. Veit, F. Yu, and S. Kumar, “Large language mod-
els with controllable working memory,” arXiv preprint
arXiv:2211.05110 , 2022.
[70] Y. Onoe, M. J. Zhang, E. Choi, and G. Durrett, “Entity
cloze by date: What lms know about unseen entities,”
arXiv preprint arXiv:2205.02832 , 2022.[71] J.-Y. Yao, K.-P. Ning, Z.-H. Liu, M.-N. Ning, Y.-Y. Liu,
and L. Yuan, “Llm lies: Hallucinations are not bugs,
but features as adversarial examples,” arXiv preprint
arXiv:2310.01469 , 2023.
[72] X. Li, Y. Zhang, and E. C. Malthouse, “Large language
model agent for fake news detection,” arXiv preprint
arXiv:2405.01593 , 2024.
[73] M.R.DeVerna, H.Y.Yan, K.-C.Yang, andF.Menczer,
“Fact-checking information from large language mod-
els can decrease headline discernment,” Proceedings of
the National Academy of Sciences , vol. 121, no. 50, p.
e2322823121, 2024.
[74] G. Luo, A. Holtzman, and Y. Choi, “Newsclip-
pings: Automatic generation of out-of-context
multimodal media,” in Proceedings of the 2021
Conference on Empirical Methods in Natural Language
Processing . Association for Computational Linguis-
tics, 2021, pp. 10302–10313. [Online]. Available:
https://aclanthology.org/2021.emnlp-main.545/
[75] J. Wei, C. Yang, X. Song, Y. Lu, N. Hu, J. Huang,
D. Tran, D. Peng, R. Liu, D. Huang, C. Du, and Q. V.
Le, “Long-form factuality in large language models,”
arXiv, Mar. 2024.
[76] M. Leippold, S. A. Vaghefi, D. Stammbach, V. Muc-
cione, J. Bingler, J. Ni, C. C. Senni, T. Wekhof, T. Schi-
manski, G. Gostlow et al., “Automated fact-checking of
climateclaimswithlargelanguagemodels,” npj Climate
Action, vol. 4, no. 1, p. 17, 2025.
[77] E. Fadeeva, A. Rubashevskii, A. Shelmanov, S. Pe-
trakov, H. Li, H. Mubarak, E. Tsymbalov, G. Kuzmin,
A.Panchenko, T.Baldwin et al., “Fact-checkingtheout-
put of large language models via token-level uncertainty
quantification,” arXiv preprint arXiv:2403.04696 , 2024.
[78] P. Sharma, T. R. Shaham, M. Baradad, S. Fu,
A. Rodriguez-Munoz, S. Duggal, P. Isola, and A. Tor-
ralba, “A vision check-up for language models,” in Pro-
ceedings of the IEEE/CVF Conference on Computer Vi-
sion and Pattern Recognition , 2024, pp. 14410–14419.
[79] G. Xiong, Q. Jin, Z. Lu, and A. Zhang, “Benchmarking
Retrieval-Augmented Generation for Medicine,” ACL
Anthology , pp. 6233–6251, Aug. 2024.
[80] E. C. Choi and E. Ferrara, “Fact-gpt: Fact-checking
augmentation via claim matching with llms,” in Com-
panion Proceedings of the ACM Web Conference 2024 ,
2024, pp. 883–886.
[81] Q. Nan, J. Cao, Y. Zhu, Y. Wang, and J. Li, “Mdfend:
Multi-domain fake news detection,” in Proceedings of
the 30th ACM international conference on information
& knowledge management , 2021, pp. 3343–3347.
28

[82] K.Shu, D.Mahudeswaran, S.Wang, D.Lee, andH.Liu,
“FakeNewsNet: A Data Repository with News Con-
tent, Social Context and Spatialtemporal Information
for Studying Fake News on Social Media,” arXiv, Sep.
2018.
[83] F. K. A. Salem, R. Al Feel, S. Elbassuoni, M. Jaber,
and M. Farah, “Fa-kes: A fake news dataset around the
syrian war,” in Proceedings of the international AAAI
conference on web and social media , vol. 13, 2019, pp.
573–582.
[84] “DisinformationandFakeNews,” Jun.2025,[Online; ac-
cessed 1. Jun. 2025]. [Online]. Available: https://www.
kaggle.com/datasets/corrieaar/disinformation-articles
[85] Q. Yang, T. Christensen, S. Gilda, J. Fernandes,
D. Oliveira, R. Wilson, and D. Woodard, “Are fact-
checking tools helpful? an exploration of the usability
of google fact check,” arXiv preprint arXiv:2402.13244 ,
2024.
[86] J. Thorne, A. Vlachos, C. Christodoulopoulos,
and A. Mittal, “Fever: a large-scale dataset for
fact extraction and verification,” in Proceedings of
the 2018 Conference of the North American Chapter of
the Association for Computational Linguistics: Human
Language Technologies . AssociationforComputational
Linguistics, 2018, pp. 809–819. [Online]. Available:
https://aclanthology.org/N18-1074/
[87] Y. Jiang, S. Bordia, Z. Zhong, C. Dognin, M. Singh,
and M. Bansal, “HoVer: A dataset for many-
hop fact extraction and claim verification,” in
Findings of the Association for Computational Linguis-
tics: EMNLP 2020 . Association for Computational
Linguistics, 2020, pp. 3441–3450. [Online]. Available:
https://aclanthology.org/2020.findings-emnlp.309/
[88] W. Y. Wang, “"liar, liar pants on fire": A
new benchmark dataset for fake news detec-
tion,” in Proceedings of the 55th Annual Meeting of
the Association for Computational Linguistics (Volume
2: Short Papers) . Association for Computational
Linguistics, 2017, pp. 422–426. [Online]. Available:
https://aclanthology.org/P17-2067/
[89] Y. Zhang, Y. Zhang, Y. Zhang, Y. Zhang, and
Y. Zhang, “A coarse-to-fine cascaded evidence-
distillation neural network for explainable fake
news detection,” in Proceedings of the 29th In-
ternational Conference on Computational Linguistics .
International Committee on Computational Lin-
guistics, 2022, pp. 2637–2647. [Online]. Available:
https://aclanthology.org/2022.coling-1.230/
[90] D. Wadden, S. Lin, K. Lo, L. L. Wang, M. van
Zuylen, A. Cohan, and H. Hajishirzi, “Fact or fiction:Verifying scientific claims,” in Proceedings of the 2020
Conference on Empirical Methods in Natural Language
Processing (EMNLP) . Association for Computational
Linguistics, 2020, pp. 7534–7550. [Online]. Available:
https://aclanthology.org/2020.emnlp-main.609/
[91] A. Saakyan, T. Chakrabarty, and S. Muresan, “Covid-
fact: Fact extraction and verification of real-world
claims on covid-19 pandemic,” in Proceedings of
the 59th Annual Meeting of the Association for Compu-
tational Linguistics . Association for Computational
Linguistics, 2021, pp. 2116–2129. [Online]. Available:
https://aclanthology.org/2021.acl-long.165/
[92] A. Pal, L. K. Umapathi, and M. Sankarasubbu,
“Medmcqa: A large-scale multi-subject multi-choice
dataset for medical domain question answering,”
inProceedings of the Conference on Health, Inference,
and Learning . PMLR, 2022, pp. 248–260. [Online].
Available: https://proceedings.mlr.press/v174/pal22a.
html
[93] G. Tsatsaronis, G. Balikas, P. Malakasiotis, I. Partalas,
M. Zschunke, M. R. Alvers, D. Weissenborn,
A. Krithara, S. Petridis, D. Polychronopoulos
et al., “An overview of the bioasq large-scale
biomedical semantic indexing and question answering
competition,” in BMC Bioinformatics , vol. 16, no. 1.
BioMed Central, 2015, pp. 1–28. [Online]. Available:
https://doi.org/10.1186/s12859-015-0564-4
[94] Q. Jin, B. Dhingra, Z. Liu, W. Cohen, and X. Lu,
“Pubmedqa: A dataset for biomedical research question
answering,” in Proceedings of the 2019 Conference
on Empirical Methods in Natural Language Processing
and the 9th International Joint Conference on Natural
Language Processing (EMNLP-IJCNLP) . Association
for Computational Linguistics, 2019, pp. 2567–
2577. [Online]. Available: https://aclanthology.org/
D19-1259/
[95] A. Author and B. Collaborator, “Mm-fever: Multi-
modal fact verification dataset,” in Proceedings of the
Multimodal Fact-Checking Workshop , 2023. [Online].
Available: https://example.com/mmfever
[96] C. Researcher and D. Analyst, “Post-4v: A dataset for
post-verificationinmultimodalsettings,” in Proceedings
of the Vision-Language Verification Conference , 2024.
[Online]. Available: https://example.com/post4v
[97] Y. Yao, L. Zhang, J. Liu, Z. Liu, and M. Sun,
“End-to-end multimodal fact-checking and explanation
generation: A challenging dataset and models,”
inProceedings of the 46th International ACM SIGIR
Conference on Research and Development in Informa-
tion Retrieval . ACM, 2023, pp. 1234–1243. [Online].
Available: https://doi.org/10.1145/3539618.3591879
29

[98] J. Li, X. Cheng, W. X. Zhao, J.-Y. Nie, and
J.-R. Wen, “Halueval: A large-scale hallucination
evaluation benchmark for large language models,”
inProceedings of the 2023 Conference on Empirical
Methods in Natural Language Processing . Association
for Computational Linguistics, 2023, pp. 4567–
4578. [Online]. Available: https://aclanthology.org/
2023.emnlp-main.397/
[99] R. Kamoi, S. S. S. Das, R. Lou, J. J. Ahn,
Y. Zhao et al., “Evaluating llms at detecting errors in
llm responses,” arXiv preprint arXiv:2404.03602 , 2024.
[Online]. Available: https://arxiv.org/abs/2404.03602
[100] J. M. Eisenschlos, B. Dhingra, J. Bulian,
B. Börschinger, and J. Boyd-Graber, “Fool me
twice: Entailment from wikipedia gamification,”
inProceedings of the 2021 Conference of the North
American Chapter of the Association for Computational
Linguistics . Association for Computational Lin-
guistics, 2021, pp. 1234–1245. [Online]. Available:
https://aclanthology.org/2021.naacl-main.32/
[101] N. Zhang, R. Kamoi, S. S. S. Das et al.,
“Factbench: A dynamic benchmark for in-the-
wild language model factuality evaluation,” arXiv
preprint arXiv:2410.22257 , 2024. [Online]. Available:
https://arxiv.org/abs/2410.22257
[102] Y. Zhang, Y. Wang, J. Liu et al., “Fire: A dataset
for feedback integration and refinement in llms,” arXiv
preprint arXiv:2407.11522 , 2024. [Online]. Available:
https://arxiv.org/abs/2407.11522
[103] A.Larraz et al., “Claimmatch: Amassivelymultilingual
dataset of fact-checked claim clusters,” in Proceedings
of the 2024 Conference on Multilingual Fact-Checking .
Association for Computational Linguistics, 2024, pp.
56–67. [Online]. Available: https://arxiv.org/html/
2503.22280v1
[104] L. Tang et al., “Llm-aggrefact: A benchmark for ag-
gregatedfactualityevaluation,” https://huggingface.co/
datasets/lytang/LLM-AggreFact, 2024.
[105] A. Barrón-Cedeño et al., “Clef2022-checkthat! lab on
fighting the covid-19 infodemic and fake news detec-
tion,” in Working Notes of CLEF 2022 , 2022. [Online].
Available: https://ceur-ws.org/Vol-3180/paper-38.pdf
[106] A. Gupta and V. Srikumar, “X-fact: A new benchmark
dataset for multilingual fact checking,” in Proceed-
ings of the 2021 Conference on Empirical Methods in
Natural Language Processing . Association for Compu-
tational Linguistics, 2021, pp. 732–748. [Online]. Avail-
able: https://aclanthology.org/2021.emnlp-main.59/[107] Y. Bai and K. Fu, “A large language model-based fake
news detection framework with rag fact-checking,” in
2024 IEEE International Conference on Big Data (Big-
Data). IEEE, 2024, pp. 8617–8619.
30